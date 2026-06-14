#!/usr/bin/env python3
"""
run_stage1_dev_label_crop_ablation_diagnostic.py

Second-Stage Lesion Refiner v1 — Phase 5.8 Stage1_dev Label/Crop Ablation Diagnostic

목적:
- stage1_dev 154명 전체에서 label_basis, overlap threshold, crop size 후보 비교
- 11개 ablation config에 대해 config별 통계 출력
- console 출력 전용 — 파일 저장 없음

IMPORTANT:
- stage1_dev만 처리 (stage2_holdout 절대 포함 금지)
- 봉인 환자 감지 즉시 중단: LUNG1-089, LUNG1-231, LUNG1-372
- v1 outputs 하위 쓰기 금지
- v2 경로 접근 금지
- 파일 저장 없음 (CSV/JSON/PNG/NPZ 생성 금지)
- runtime_summary.csv / error.csv append 금지
- 학습/평가/score 재계산 금지
- candidate seed: p95 threshold=14.3774 고정
- p92.5/p90 사용 금지
- adaptive crop 구현 금지 (TODO 주석으로만 기록)

ablation config (11개):
  baseline_fixed96_thr010 : fixed_crop_z_center, crop=96, thr=0.10
  fixed96_thr005          : fixed_crop_z_center, crop=96, thr=0.05
  fixed96_thr001          : fixed_crop_z_center, crop=96, thr=0.01
  fixed96_gt0             : fixed_crop_z_center, crop=96, overlap>0
  rawbbox_thr010          : raw_bbox_z_center,   crop=96 ref, thr=0.10
  rawbbox_thr005          : raw_bbox_z_center,   crop=96 ref, thr=0.05
  context_thr005          : context_z_center,    crop=96 ref, thr=0.05
  fixed96_zstack_thr005   : fixed_crop_zstack_max, crop=96, thr=0.05
  fixed128_thr010         : fixed_crop_z_center, crop=128, thr=0.10
  fixed128_thr005         : fixed_crop_z_center, crop=128, thr=0.05
  fixed128_zstack_thr005  : fixed_crop_zstack_max, crop=128, thr=0.05

# TODO (후속 후보 — 이번 스크립트에서 구현하지 않음):
#   adaptive crop: bbox 크기 기반 adaptive crop_size
#   crop size 96/128 비교 후 필요할 때 검토

py_compile 확인 (실행 아님):
  python -m py_compile scripts/run_stage1_dev_label_crop_ablation_diagnostic.py

실행 (별도 승인 후):
  source ~/ai_env/bin/activate && \\
  python scripts/run_stage1_dev_label_crop_ablation_diagnostic.py
"""

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yaml

# ============================================================
# 경로 상수 (v1 고정 — v2 접근 금지)
# ============================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
SCORE_DIR = (
    REPO_ROOT / "outputs/position-aware-padim-v1/scores/padim_v1/lesion_by_patient"
)
SPLIT_CSV = (
    REPO_ROOT
    / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"
)
PATHS_CONFIG = REPO_ROOT / "configs/paths.local.yaml"

V1_OUTPUT_DIR = REPO_ROOT / "outputs/position-aware-padim-v1"

SCRIPT_NAME = "run_stage1_dev_label_crop_ablation_diagnostic.py"

# ============================================================
# 안전 상수
# ============================================================
P95_THRESHOLD = 14.3774
DEFAULT_CONTEXT_MARGIN = 32
DEFAULT_Z_RANGE = 1
MAX_BBOX_SIZE = 256  # 기존 스크립트(build_stage1_dev_candidate_manifest.py)와 동일

STAGE2_HOLDOUT_BLOCKED: Set[str] = {"LUNG1-089", "LUNG1-231", "LUNG1-372"}
WEAK_CASES: List[str] = ["MSD_lung_043", "MSD_lung_079", "MSD_lung_096"]

BASELINE_CONFIG_NAME = "baseline_fixed96_thr010"

# replacement_50 참고 기준 (decision 문서 기록용 — 직접 수치 비교 아님)
# phase5_6_fallback_diagnostic_result_decision_v1.md 참고
# hybrid_p95_replace_topk_zero_lc_50: n_candidates=8672, zero_lc=90명, hn:pos=42.0:1
_REPLACEMENT50_REF = {
    "n_candidates": 8672,
    "n_zero_lc": 90,
    "hn_pos_ratio": 42.0,
}


# ============================================================
# Ablation Config
# ============================================================
@dataclass(frozen=True)
class AblationConfig:
    name: str
    basis: str
    crop_size: int   # raw_bbox/context basis의 경우 overlap 계산에 직접 미사용 (reference only)
    threshold: float
    is_gt0: bool = False  # threshold가 ">0" 모드인 경우 True


ABLATION_CONFIGS: List[AblationConfig] = [
    AblationConfig("baseline_fixed96_thr010", "fixed_crop_z_center", 96, 0.10),
    AblationConfig("fixed96_thr005",          "fixed_crop_z_center", 96, 0.05),
    AblationConfig("fixed96_thr001",          "fixed_crop_z_center", 96, 0.01),
    AblationConfig("fixed96_gt0",             "fixed_crop_z_center", 96, 0.0,  is_gt0=True),
    AblationConfig("rawbbox_thr010",          "raw_bbox_z_center",   96, 0.10),
    AblationConfig("rawbbox_thr005",          "raw_bbox_z_center",   96, 0.05),
    # context_thr005: phase5_7 결과에서 fixed_crop_z_center와 수치 거의 동일 — 보조 비교로만 해석
    AblationConfig("context_thr005",          "context_z_center",    96, 0.05),
    AblationConfig("fixed96_zstack_thr005",   "fixed_crop_zstack_max", 96,  0.05),
    AblationConfig("fixed128_thr010",         "fixed_crop_z_center", 128, 0.10),
    AblationConfig("fixed128_thr005",         "fixed_crop_z_center", 128, 0.05),
    AblationConfig("fixed128_zstack_thr005",  "fixed_crop_zstack_max", 128, 0.05),
]

# ablation에서 사용하는 crop_size 목록 (자동 수집)
_ABLATION_CROP_SIZES: List[int] = sorted(
    set(cfg.crop_size for cfg in ABLATION_CONFIGS
        if cfg.basis in ("fixed_crop_z_center", "fixed_crop_zstack_max"))
)


# ============================================================
# 안전 가드
# ============================================================
def check_stage1_dev(patient_id: str, stage_split: str) -> None:
    """stage2_holdout 또는 봉인 환자 감지 시 즉시 중단."""
    if patient_id in STAGE2_HOLDOUT_BLOCKED:
        raise RuntimeError(
            f"[GUARD] 봉인 환자 감지됨: {patient_id} — 즉시 중단.\n"
            "봉인 환자(LUNG1-089, LUNG1-231, LUNG1-372)는 포함 금지."
        )
    if stage_split != "stage1_dev":
        raise RuntimeError(
            f"[GUARD] stage1_dev가 아닌 환자 감지: {patient_id} "
            f"(stage_split={stage_split}) — 즉시 중단."
        )


def check_no_v2_path(path: Path) -> None:
    """v2 경로 접근 시도 시 즉시 중단."""
    path_str = str(path).lower()
    if "v2" in path_str and "outputs" in path_str:
        raise RuntimeError(
            f"[GUARD] v2 경로 접근 금지: {path}\n"
            "v2 관련 파일/프로세스/로그/출력 접근 금지."
        )


# ============================================================
# 경로 로딩
# ============================================================
def load_dataset_root() -> Path:
    """configs/paths.local.yaml에서 v1 dataset 경로 로드."""
    if not PATHS_CONFIG.exists():
        raise FileNotFoundError(f"paths.local.yaml 없음: {PATHS_CONFIG}")
    with open(PATHS_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    key = "nsclc_msd_usable_only_v1"
    val = cfg.get(key, "")
    if not val:
        raise ValueError(f"paths.local.yaml에 '{key}' 없거나 비어 있음")
    root = Path(val)
    check_no_v2_path(root)
    return root


# ============================================================
# 데이터 로딩
# ============================================================
def load_split(
    patients: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """split CSV 로드 → stage1_dev 필터 → 봉인 환자 이중 확인."""
    df = pd.read_csv(SPLIT_CSV, encoding="utf-8-sig")

    required_cols = ["patient_id", "stage_split", "group", "safe_id"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"split CSV에 필수 컬럼이 없습니다: {missing_cols}")

    df_dev = df[df["stage_split"] == "stage1_dev"].copy()

    for pid in df_dev["patient_id"]:
        if pid in STAGE2_HOLDOUT_BLOCKED:
            raise RuntimeError(
                f"[GUARD] split CSV에 봉인 환자가 stage1_dev로 포함됨: {pid}"
            )

    if patients:
        blocked_requested = [p for p in patients if p in STAGE2_HOLDOUT_BLOCKED]
        if blocked_requested:
            raise RuntimeError(
                f"[GUARD] 봉인 환자가 --patients에 포함됨: {blocked_requested}\n"
                "봉인 환자는 포함 금지. 실행 중단."
            )

        dev_pids = set(df_dev["patient_id"].tolist())
        holdout_pids = set(
            df[df["stage_split"] == "stage2_holdout"]["patient_id"].tolist()
        )
        requested_set = set(patients)
        missing_set = requested_set - dev_pids

        if missing_set:
            in_holdout = sorted(missing_set & holdout_pids)
            not_in_csv = sorted(missing_set - holdout_pids)
            msg = f"지정한 환자 중 stage1_dev에 없는 환자: {sorted(missing_set)}\n"
            if in_holdout:
                msg += f"  → stage2_holdout 소속 (봉인 대상): {in_holdout}\n"
            if not_in_csv:
                msg += f"  → split CSV에 없는 환자: {not_in_csv}\n"
            msg += "지정한 모든 환자가 stage1_dev에 있어야 합니다."
            raise ValueError(msg)

        df_dev = df_dev[df_dev["patient_id"].isin(patients)]

    df_dev = df_dev.reset_index(drop=True)

    if limit is not None:
        df_dev = df_dev.iloc[:limit]

    return df_dev


def load_score_csv(patient_id: str) -> pd.DataFrame:
    """v1 score CSV read-only 로드."""
    csv_path = SCORE_DIR / f"{patient_id}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"score CSV 없음: {csv_path}")
    return pd.read_csv(csv_path, encoding="utf-8-sig")


def load_patient_arrays(
    row: pd.Series,
    dataset_root: Path,
) -> Tuple[np.ndarray, np.ndarray, int, int, int]:
    """ct_hu.npy, lesion_mask_model_roi.npy를 mmap read-only로 로드."""
    safe_id = str(row["safe_id"])
    vol_dir = dataset_root / "volumes_npy" / safe_id
    ct_path = vol_dir / "ct_hu.npy"
    mask_path = vol_dir / "lesion_mask_model_roi.npy"

    if not ct_path.exists():
        raise FileNotFoundError(f"CT volume 없음: {ct_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"lesion mask 없음: {mask_path}")

    ct_vol = np.load(ct_path, mmap_mode="r")
    lesion_mask = np.load(mask_path, mmap_mode="r")
    ct_z, ct_h, ct_w = ct_vol.shape
    return ct_vol, lesion_mask, ct_z, ct_h, ct_w


# ============================================================
# Candidate 생성 (기존 스크립트 로직 유지 — 원본 미수정)
# ============================================================
def merge_patches_in_slice(
    patches: List[Tuple[int, int, int, int, float]],
) -> List[Dict]:
    """같은 slice의 positive patch를 8-neighbor touch/overlap 기준으로 merge."""
    n = len(patches)
    if n == 0:
        return []

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    def touching(a: Tuple, b: Tuple) -> bool:
        y_gap = max(0, max(a[0], b[0]) - min(a[2], b[2]))
        x_gap = max(0, max(a[1], b[1]) - min(a[3], b[3]))
        return y_gap == 0 and x_gap == 0

    for i in range(n):
        for j in range(i + 1, n):
            if touching(patches[i], patches[j]):
                union(i, j)

    groups: Dict[int, List] = defaultdict(list)
    for i, p in enumerate(patches):
        groups[find(i)].append(p)

    result = []
    for grp in groups.values():
        y0 = min(p[0] for p in grp)
        x0 = min(p[1] for p in grp)
        y1 = max(p[2] for p in grp)
        x1 = max(p[3] for p in grp)
        scores = [p[4] for p in grp]
        result.append({
            "y0": y0, "x0": x0, "y1": y1, "x1": x1,
            "n_patches": len(grp),
            "mean_score": float(np.mean(scores)),
            "max_score": float(np.max(scores)),
        })
    return result


def bboxes_overlap(a: Dict, b: Dict) -> bool:
    """두 bbox가 xy 평면에서 겹치는지 확인."""
    vy = a["y0"] < b["y1"] and b["y0"] < a["y1"]
    vx = a["x0"] < b["x1"] and b["x0"] < a["x1"]
    return vy and vx


def build_candidates_3d(slice_bboxes: Dict[int, List[Dict]]) -> List[Dict]:
    """slice별 merged bbox를 연속 z 기준으로 묶어 3D candidate 생성."""
    candidates: List[Dict] = []
    active: List[Dict] = []

    for z in sorted(slice_bboxes.keys()):
        current_bboxes = slice_bboxes[z]
        new_active: List[Dict] = []
        used: Set[int] = set()

        for cand in active:
            if z != cand["z_end"] + 1:
                candidates.append(cand)
                continue

            matched = False
            for i, bbox in enumerate(current_bboxes):
                if i in used:
                    continue
                if bboxes_overlap(cand["last_bbox"], bbox):
                    total = cand["n_patches"] + bbox["n_patches"]
                    cand["mean_score"] = (
                        cand["mean_score"] * cand["n_patches"]
                        + bbox["mean_score"] * bbox["n_patches"]
                    ) / total
                    cand["max_score"] = max(cand["max_score"], bbox["max_score"])
                    cand["n_patches"] = total
                    cand["y0"] = min(cand["y0"], bbox["y0"])
                    cand["x0"] = min(cand["x0"], bbox["x0"])
                    cand["y1"] = max(cand["y1"], bbox["y1"])
                    cand["x1"] = max(cand["x1"], bbox["x1"])
                    cand["z_end"] = z
                    cand["last_bbox"] = bbox
                    used.add(i)
                    new_active.append(cand)
                    matched = True
                    break
            if not matched:
                candidates.append(cand)

        for i, bbox in enumerate(current_bboxes):
            if i not in used:
                new_active.append({
                    "z_start": z, "z_end": z,
                    "y0": bbox["y0"], "x0": bbox["x0"],
                    "y1": bbox["y1"], "x1": bbox["x1"],
                    "n_patches": bbox["n_patches"],
                    "mean_score": bbox["mean_score"],
                    "max_score": bbox["max_score"],
                    "last_bbox": bbox,
                })

        active = new_active

    candidates.extend(active)
    for cand in candidates:
        cand["z_center"] = (cand["z_start"] + cand["z_end"]) // 2
        cand.pop("last_bbox", None)
    return candidates


def apply_context_margin(
    y0: int, x0: int, y1: int, x1: int,
    margin: int, ct_h: int, ct_w: int,
) -> Tuple[int, int, int, int]:
    return (
        max(0, y0 - margin),
        max(0, x0 - margin),
        min(ct_h, y1 + margin),
        min(ct_w, x1 + margin),
    )


def compute_fixed_crop_coords(
    bbox_center_y: int, bbox_center_x: int,
    crop_size: int, ct_h: int, ct_w: int,
) -> Tuple[int, int, int, int]:
    """fixed crop 좌표 계산 (기존 스크립트와 동일한 center-crop 로직)."""
    half = crop_size // 2
    y0 = max(0, bbox_center_y - half)
    x0 = max(0, bbox_center_x - half)
    y1 = min(ct_h, y0 + crop_size)
    x1 = min(ct_w, x0 + crop_size)
    y0 = max(0, y1 - crop_size)
    x0 = max(0, x1 - crop_size)
    return y0, x0, y1, x1


# ============================================================
# Overlap 계산
# ============================================================
def compute_lesion_overlap(
    lesion_mask: np.ndarray,
    z: int, y0: int, x0: int, y1: int, x1: int,
) -> Tuple[int, float]:
    """z slice에서 crop bbox 영역의 lesion mask overlap 계산."""
    if z < 0 or z >= lesion_mask.shape[0]:
        return 0, 0.0
    crop = lesion_mask[z, y0:y1, x0:x1]
    area = (y1 - y0) * (x1 - x0)
    if area == 0:
        return 0, 0.0
    lesion_px = int(np.sum(crop > 0))
    return lesion_px, float(lesion_px / area)


# ============================================================
# Enriched candidate 생성 (crop_size 복수 지원)
# ============================================================
def build_enriched_candidates_multi_crop(
    row: pd.Series,
    ct_vol: np.ndarray,
    lesion_mask: np.ndarray,
    args: argparse.Namespace,
    crop_sizes: List[int],
) -> List[Dict]:
    """
    p95 threshold 기준 candidate 생성 + 모든 basis overlap 계산.
    - crop_sizes: fixed_crop overlap을 계산할 crop_size 목록
    - top-k/fallback 구현 금지
    - 파일 저장 없음, read-only 작업만 수행
    """
    patient_id = str(row["patient_id"])
    stage_split = str(row["stage_split"])

    check_stage1_dev(patient_id, stage_split)

    score_df = load_score_csv(patient_id)
    pos_df = score_df[score_df["padim_score"] >= args.threshold_p95].copy()

    if pos_df.empty:
        return []

    ct_z, ct_h, ct_w = ct_vol.shape

    slice_bboxes: Dict[int, List[Dict]] = {}
    for local_z, grp in pos_df.groupby("local_z"):
        patches = [
            (
                int(r["y0"]), int(r["x0"]),
                int(r["y1"]), int(r["x1"]),
                float(r["padim_score"]),
            )
            for _, r in grp.iterrows()
        ]
        merged = merge_patches_in_slice(patches)
        if merged:
            slice_bboxes[int(local_z)] = merged

    if not slice_bboxes:
        return []

    cands_raw = build_candidates_3d(slice_bboxes)
    candidates: List[Dict] = []

    for cand in cands_raw:
        z_center = int(cand["z_center"])

        cy0, cx0, cy1, cx1 = apply_context_margin(
            cand["y0"], cand["x0"], cand["y1"], cand["x1"],
            args.context_margin, ct_h, ct_w,
        )

        z_lo = max(0, z_center - args.z_range)
        z_hi = min(ct_z - 1, z_center + args.z_range)

        # large_bbox 판정 (context bbox h/w 기준 — phase5_5 스크립트와 동일)
        bbox_h = cy1 - cy0
        bbox_w = cx1 - cx0
        bbox_too_large = bbox_h > MAX_BBOX_SIZE or bbox_w > MAX_BBOX_SIZE

        # crop_size 독립 basis overlap
        _, ctx_ratio = compute_lesion_overlap(
            lesion_mask, z_center, cy0, cx0, cy1, cx1
        )
        _, raw_ratio = compute_lesion_overlap(
            lesion_mask, z_center,
            cand["y0"], cand["x0"], cand["y1"], cand["x1"],
        )

        # zstack_max (context bbox 기반 z_lo~z_hi 전체 순회 — 기존 스크립트와 동일)
        zstack_ratio = 0.0
        for z in range(z_lo, z_hi + 1):
            _, r = compute_lesion_overlap(lesion_mask, z, cy0, cx0, cy1, cx1)
            if r > zstack_ratio:
                zstack_ratio = r

        bbox_center_y = (cand["y0"] + cand["y1"]) // 2
        bbox_center_x = (cand["x0"] + cand["x1"]) // 2

        cand_dict: Dict = {
            "patient_id": patient_id,
            "z_center": z_center,
            "z_lo": z_lo,
            "z_hi": z_hi,
            "raw_y0": cand["y0"], "raw_x0": cand["x0"],
            "raw_y1": cand["y1"], "raw_x1": cand["x1"],
            "mean_score": cand["mean_score"],
            "max_score": cand["max_score"],
            "bbox_too_large": bbox_too_large,
            "overlap_context_z_center": ctx_ratio,
            "overlap_raw_bbox_z_center": raw_ratio,
            "overlap_zstack_max": zstack_ratio,
        }

        # crop_size별 fixed_crop overlap
        for cs in crop_sizes:
            fc_y0, fc_x0, fc_y1, fc_x1 = compute_fixed_crop_coords(
                bbox_center_y, bbox_center_x, cs, ct_h, ct_w,
            )
            _, fc_z_ratio = compute_lesion_overlap(
                lesion_mask, z_center, fc_y0, fc_x0, fc_y1, fc_x1
            )
            # fixed_crop_zstack_max: fixed crop bbox 기반 z_lo~z_hi 전체 순회
            fc_zstack_ratio = 0.0
            for z in range(z_lo, z_hi + 1):
                _, r = compute_lesion_overlap(
                    lesion_mask, z, fc_y0, fc_x0, fc_y1, fc_x1
                )
                if r > fc_zstack_ratio:
                    fc_zstack_ratio = r

            cand_dict[f"overlap_fixed_crop_z_center_{cs}"] = fc_z_ratio
            cand_dict[f"overlap_fixed_crop_zstack_max_{cs}"] = fc_zstack_ratio

        candidates.append(cand_dict)

    return candidates


# ============================================================
# Label 계산
# ============================================================
def get_overlap_value(candidate: Dict, cfg: AblationConfig) -> float:
    """candidate에서 ablation config에 해당하는 basis overlap 값 반환."""
    basis = cfg.basis
    if basis in ("fixed_crop_z_center", "fixed_crop_zstack_max"):
        key = f"overlap_{basis}_{cfg.crop_size}"
    else:
        key = f"overlap_{basis}"
    return float(candidate.get(key, 0.0))


def assign_labels_for_config(overlap: float, cfg: AblationConfig) -> Tuple[str, str]:
    """
    overlap과 config 기준으로 rd4ad_label, binary_label 반환.

    gt0 모드:
      - overlap > 0  → lesion_candidate / positive
      - overlap == 0 → normal_like / hard_negative
      - ambiguous 0개 (구조상)

    일반 모드:
      - overlap == 0          → normal_like / hard_negative
      - 0 < overlap < thr     → ambiguous / ambiguous
      - overlap >= thr        → lesion_candidate / positive
    """
    if cfg.is_gt0:
        if overlap > 0:
            return "lesion_candidate", "positive"
        return "normal_like", "hard_negative"

    if overlap == 0.0:
        return "normal_like", "hard_negative"
    if overlap >= cfg.threshold:
        return "lesion_candidate", "positive"
    return "ambiguous", "ambiguous"


# ============================================================
# 환자 단위 처리
# ============================================================
@dataclass
class PatientConfigResult:
    n_lesion_candidate: int = 0
    n_ambiguous: int = 0
    n_normal_like: int = 0
    n_positive: int = 0
    n_hard_negative: int = 0


def process_patient(
    row: pd.Series,
    dataset_root: Path,
    args: argparse.Namespace,
    crop_sizes: List[int],
) -> Optional[Dict]:
    """
    단일 환자 처리.
    반환: {
        "patient_id": str,
        "n_candidates": int,
        "n_large_bbox": int,
        "failed": bool,
        "config_results": {config_name: PatientConfigResult}
    }
    """
    patient_id = str(row["patient_id"])

    try:
        ct_vol, lesion_mask, ct_z, ct_h, ct_w = load_patient_arrays(
            row, dataset_root
        )
    except FileNotFoundError as e:
        print(f"  [ERROR] {patient_id}: 파일 없음 — {e}")
        return {
            "patient_id": patient_id,
            "n_candidates": 0,
            "n_large_bbox": 0,
            "failed": True,
            "config_results": {},
        }

    try:
        candidates = build_enriched_candidates_multi_crop(
            row, ct_vol, lesion_mask, args, crop_sizes
        )
    except Exception as e:
        print(f"  [ERROR] {patient_id}: candidate 생성 실패 — {e}")
        return {
            "patient_id": patient_id,
            "n_candidates": 0,
            "n_large_bbox": 0,
            "failed": True,
            "config_results": {},
        }

    n_candidates = len(candidates)
    n_large_bbox = sum(1 for c in candidates if c.get("bbox_too_large", False))

    config_results: Dict[str, PatientConfigResult] = {}
    for cfg in ABLATION_CONFIGS:
        res = PatientConfigResult()
        for cand in candidates:
            overlap = get_overlap_value(cand, cfg)
            rd4ad_label, binary_label = assign_labels_for_config(overlap, cfg)
            if rd4ad_label == "lesion_candidate":
                res.n_lesion_candidate += 1
            elif rd4ad_label == "ambiguous":
                res.n_ambiguous += 1
            else:
                res.n_normal_like += 1
            if binary_label == "positive":
                res.n_positive += 1
            elif binary_label == "hard_negative":
                res.n_hard_negative += 1
        config_results[cfg.name] = res

    return {
        "patient_id": patient_id,
        "n_candidates": n_candidates,
        "n_large_bbox": n_large_bbox,
        "failed": False,
        "config_results": config_results,
    }


# ============================================================
# 통계 집계
# ============================================================
def aggregate_stats(
    patient_results: List[Dict],
    n_patients_target: int,
    args: argparse.Namespace,
) -> Dict[str, Dict]:
    """
    전체 환자 결과를 config별로 집계.
    반환: {config_name: 집계 dict}
    """
    scored_results = [r for r in patient_results if not r["failed"]]
    failed_results = [r for r in patient_results if r["failed"]]

    aggregated: Dict[str, Dict] = {}

    for cfg in ABLATION_CONFIGS:
        cname = cfg.name
        n_patients_scored = len(scored_results)
        n_patients_failed = len(failed_results)

        candidate_counts = [r["n_candidates"] for r in scored_results]
        large_bbox_counts = [r["n_large_bbox"] for r in scored_results]
        n_candidates_total = sum(candidate_counts)
        n_large_bbox_total = sum(large_bbox_counts)

        per_patient_lc: Dict[str, int] = {}
        per_patient_lb: Dict[str, int] = {}
        per_patient_n: Dict[str, int] = {}
        n_total_lc = 0
        n_total_amb = 0
        n_total_nl = 0
        n_total_pos = 0
        n_total_hn = 0

        zero_lc_patients: List[str] = []
        amb_only_patients: List[str] = []
        weak_case_info: Dict[str, Dict] = {}

        for r in scored_results:
            pid = r["patient_id"]
            res = r["config_results"].get(cname, PatientConfigResult())
            per_patient_lc[pid] = res.n_lesion_candidate
            per_patient_lb[pid] = r["n_large_bbox"]
            per_patient_n[pid] = r["n_candidates"]

            n_total_lc += res.n_lesion_candidate
            n_total_amb += res.n_ambiguous
            n_total_nl += res.n_normal_like
            n_total_pos += res.n_positive
            n_total_hn += res.n_hard_negative

            if res.n_lesion_candidate == 0:
                zero_lc_patients.append(pid)
                if res.n_ambiguous > 0:
                    amb_only_patients.append(pid)

            if pid in WEAK_CASES:
                weak_case_info[pid] = {
                    "n_lesion_candidate": res.n_lesion_candidate,
                    "n_ambiguous": res.n_ambiguous,
                    "n_normal_like": res.n_normal_like,
                    "n_candidates": r["n_candidates"],
                }

        cand_counts_arr = np.array(candidate_counts) if candidate_counts else np.array([0])
        hn_pos_ratio = (
            round(n_total_hn / n_total_pos, 1) if n_total_pos > 0 else float("inf")
        )
        large_bbox_ratio = (
            round(n_large_bbox_total / n_candidates_total, 4)
            if n_candidates_total > 0
            else 0.0
        )

        aggregated[cname] = {
            "cfg": cfg,
            "n_patients_target": n_patients_target,
            "n_patients_scored": n_patients_scored,
            "n_patients_failed": n_patients_failed,
            "n_candidates_total": n_candidates_total,
            "candidate_count_min": int(cand_counts_arr.min()),
            "candidate_count_median": float(np.median(cand_counts_arr)),
            "candidate_count_max": int(cand_counts_arr.max()),
            "n_normal_like": n_total_nl,
            "n_ambiguous": n_total_amb,
            "n_lesion_candidate": n_total_lc,
            "n_hard_negative": n_total_hn,
            "n_positive": n_total_pos,
            "n_large_bbox": n_large_bbox_total,
            "large_bbox_ratio": large_bbox_ratio,
            "hn_pos_ratio": hn_pos_ratio,
            "n_zero_lc_patients": len(zero_lc_patients),
            "zero_lc_patients": sorted(zero_lc_patients),
            "n_amb_only_patients": len(amb_only_patients),
            "amb_only_patients": sorted(amb_only_patients),
            "weak_case_info": weak_case_info,
            "per_patient_lc": per_patient_lc,
            "per_patient_lb": per_patient_lb,
            "per_patient_n": per_patient_n,
        }

    return aggregated


# ============================================================
# 출력 함수
# ============================================================
def _hn_pos_str(ratio: float) -> str:
    if ratio == float("inf"):
        return "∞ (positive=0)"
    return f"{ratio:.1f}:1"


def print_config_result(
    stats: Dict,
    baseline_stats: Optional[Dict],
    no_top_lists: bool,
) -> None:
    cfg: AblationConfig = stats["cfg"]
    thr_str = ">0 (gt0)" if cfg.is_gt0 else f"{cfg.threshold:.2f}"
    basis_note = ""
    if cfg.basis in ("raw_bbox_z_center", "context_z_center"):
        basis_note = " (crop_size는 overlap 계산에 미사용 — raw/context bbox 기준)"
    if cfg.name == "context_thr005":
        basis_note += " [주의: phase5_7에서 fixed_crop_z_center와 수치 거의 동일 — 보조 비교]"

    print(f"\n{'=' * 70}")
    print(f"  CONFIG: {cfg.name}")
    print(f"{'=' * 70}")
    print(f"  basis      : {cfg.basis}{basis_note}")
    print(f"  crop_size  : {cfg.crop_size}")
    print(f"  threshold  : {thr_str}")
    print(f"  n_patients : target={stats['n_patients_target']}"
          f"  scored={stats['n_patients_scored']}"
          f"  failed={stats['n_patients_failed']}")
    print()
    print(f"  n_candidates_total : {stats['n_candidates_total']}")
    print(f"  per-patient        : min={stats['candidate_count_min']}"
          f"  median={stats['candidate_count_median']:.1f}"
          f"  max={stats['candidate_count_max']}")
    print()
    print(f"  rd4ad_label 분포:")
    print(f"    normal_like      : {stats['n_normal_like']}")
    print(f"    ambiguous        : {stats['n_ambiguous']}")
    print(f"    lesion_candidate : {stats['n_lesion_candidate']}")
    print(f"  binary_label 분포:")
    print(f"    hard_negative    : {stats['n_hard_negative']}")
    print(f"    ambiguous        : {stats['n_ambiguous']}")
    print(f"    positive         : {stats['n_positive']}")
    print()
    print(f"  lesion_candidate=0 환자 : {stats['n_zero_lc_patients']}명")
    print(f"  ambiguous만 있는 환자   : {stats['n_amb_only_patients']}명")
    print(f"  hard_negative:positive  : {_hn_pos_str(stats['hn_pos_ratio'])}")
    print(f"  large_bbox              : {stats['n_large_bbox']}"
          f" ({100 * stats['large_bbox_ratio']:.1f}%)")
    print()

    # weak_case 3명 label 분포
    print(f"  [weak_case 3명 label 분포]")
    for wc in WEAK_CASES:
        info = stats["weak_case_info"].get(wc)
        if info is None:
            print(f"    {wc:<20} : 처리되지 않음 (failed 또는 대상 외)")
        else:
            print(
                f"    {wc:<20} : lc={info['n_lesion_candidate']}"
                f"  amb={info['n_ambiguous']}"
                f"  nl={info['n_normal_like']}"
                f"  (candidates={info['n_candidates']})"
            )
    print()

    # baseline 대비 변화량
    if baseline_stats is not None and cfg.name != BASELINE_CONFIG_NAME:
        d_lc = stats["n_lesion_candidate"] - baseline_stats["n_lesion_candidate"]
        d_zero_lc = stats["n_zero_lc_patients"] - baseline_stats["n_zero_lc_patients"]
        d_amb = stats["n_ambiguous"] - baseline_stats["n_ambiguous"]
        d_lb = stats["n_large_bbox"] - baseline_stats["n_large_bbox"]
        b_ratio = baseline_stats["hn_pos_ratio"]
        c_ratio = stats["hn_pos_ratio"]
        ratio_str = (
            f"{b_ratio:.1f} → {c_ratio:.1f}"
            if c_ratio != float("inf") and b_ratio != float("inf")
            else f"{_hn_pos_str(b_ratio)} → {_hn_pos_str(c_ratio)}"
        )
        print(f"  [baseline_fixed96_thr010 대비]")
        print(f"    lesion_candidate 변화   : {d_lc:+d}")
        print(f"    lesion_candidate=0 환자 : {d_zero_lc:+d}")
        print(f"    ambiguous 변화          : {d_amb:+d}")
        print(f"    hn:pos 비율             : {ratio_str}")
        print(f"    large_bbox 변화         : {d_lb:+d}")
        print()

    # lesion_candidate=0 환자 목록
    zero_lc = stats["zero_lc_patients"]
    if zero_lc:
        print(f"  lesion_candidate=0 환자 목록 ({len(zero_lc)}명):")
        for pid in zero_lc:
            n_cand = stats["per_patient_n"].get(pid, 0)
            n_amb = stats["weak_case_info"].get(pid, {}).get("n_ambiguous", "?")
            # weak_case가 아닌 환자는 config_results에서 ambiguous 수를 별도로 조회
            print(f"    {pid} (candidates={n_cand})")
    print()

    if not no_top_lists:
        # candidate 수 Top 10
        top_cand = sorted(
            stats["per_patient_n"].items(), key=lambda x: x[1], reverse=True
        )[:10]
        print(f"  candidate 수 Top 10:")
        print(f"    {'patient_id':<22} {'n_candidates':>12}")
        for pid, cnt in top_cand:
            print(f"    {pid:<22} {cnt:>12}")
        print()

        # large_bbox Top 10
        top_lb = sorted(
            stats["per_patient_lb"].items(), key=lambda x: x[1], reverse=True
        )[:10]
        print(f"  large_bbox Top 10:")
        print(f"    {'patient_id':<22} {'large_bbox':>10}")
        for pid, cnt in top_lb:
            if cnt > 0:
                print(f"    {pid:<22} {cnt:>10}")
        print()

        # lesion_candidate Top 10
        top_lc = sorted(
            stats["per_patient_lc"].items(), key=lambda x: x[1], reverse=True
        )[:10]
        print(f"  lesion_candidate Top 10:")
        print(f"    {'patient_id':<22} {'lesion_candidate':>16}")
        for pid, cnt in top_lc:
            if cnt > 0:
                print(f"    {pid:<22} {cnt:>16}")
        print()


def print_summary_table(aggregated: Dict[str, Dict], baseline_stats: Dict) -> None:
    """config 전체 요약 표 출력."""
    print(f"\n{'=' * 70}")
    print(f"  CONFIG 요약 표")
    print(f"{'=' * 70}")
    header = (
        f"  {'config_name':<30}"
        f" {'n_total':>8}"
        f" {'lc':>6}"
        f" {'zero_lc':>8}"
        f" {'amb':>6}"
        f" {'hn:pos':>9}"
        f" {'large_bb':>8}"
    )
    print(header)
    print(f"  {'-' * 78}")

    for cfg in ABLATION_CONFIGS:
        s = aggregated[cfg.name]
        hn_str = _hn_pos_str(s["hn_pos_ratio"])
        marker = " ←base" if cfg.name == BASELINE_CONFIG_NAME else ""
        print(
            f"  {cfg.name:<30}"
            f" {s['n_candidates_total']:>8}"
            f" {s['n_lesion_candidate']:>6}"
            f" {s['n_zero_lc_patients']:>8}"
            f" {s['n_ambiguous']:>6}"
            f" {hn_str:>9}"
            f" {s['n_large_bbox']:>8}"
            f"{marker}"
        )

    print(f"\n  * replacement_50 참고 기준 (phase5_6 decision 문서):")
    print(f"    n_candidates={_REPLACEMENT50_REF['n_candidates']}"
          f"  zero_lc={_REPLACEMENT50_REF['n_zero_lc']}명"
          f"  hn:pos={_REPLACEMENT50_REF['hn_pos_ratio']:.1f}:1")
    print(f"    (직접 수치 비교 아님 — 기존 참고 기준으로만 기록)")


# ============================================================
# argparse
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Second-Stage Lesion Refiner v1 Phase 5.8 — "
            "Stage1_dev Label/Crop Ablation Diagnostic "
            "(console-only, no file write)"
        )
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        default=None,
        help=(
            "처리할 patient_id 목록 (기본 None → stage1_dev 전체). "
            "stage1_dev에 없는 환자가 들어오면 중단. "
            "stage2_holdout이면 중단."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="debug용 처리 환자 수 제한 (기본 None)",
    )
    parser.add_argument(
        "--threshold-p95",
        type=float,
        default=P95_THRESHOLD,
        dest="threshold_p95",
        help=f"p95 candidate seed threshold (기본 {P95_THRESHOLD}). p92.5/p90 사용 금지.",
    )
    parser.add_argument(
        "--context-margin",
        type=int,
        default=DEFAULT_CONTEXT_MARGIN,
        dest="context_margin",
        help=f"context margin px (기본 {DEFAULT_CONTEXT_MARGIN})",
    )
    parser.add_argument(
        "--z-range",
        type=int,
        default=DEFAULT_Z_RANGE,
        dest="z_range",
        help=f"z 범위 +/-z_range (기본 {DEFAULT_Z_RANGE})",
    )
    parser.add_argument(
        "--crop-sizes",
        nargs="+",
        type=int,
        default=[96, 128],
        dest="crop_sizes",
        help="fixed_crop overlap을 계산할 crop_size 목록 (기본 96 128)",
    )
    parser.add_argument(
        "--no-top-lists",
        action="store_true",
        dest="no_top_lists",
        help="Top 10 목록 출력 생략",
    )
    return parser.parse_args()


# ============================================================
# 메인
# ============================================================
def main() -> None:
    args = parse_args()

    # ablation config에서 사용하는 crop_size를 자동으로 crop_sizes에 포함
    required_crop_sizes = set(_ABLATION_CROP_SIZES)
    user_crop_sizes = set(args.crop_sizes)
    combined_crop_sizes = sorted(required_crop_sizes | user_crop_sizes)
    args.crop_sizes = combined_crop_sizes

    print(
        f"\n{'=' * 70}\n"
        f"  Second-Stage Lesion Refiner v1 — Phase 5.8\n"
        f"  Stage1_dev Label/Crop Ablation Diagnostic (console-only)\n"
        f"  실행 일시       : {datetime.now().isoformat(timespec='seconds')}\n"
        f"  threshold_p95  : {args.threshold_p95}\n"
        f"  context_margin : {args.context_margin}\n"
        f"  z_range        : +/-{args.z_range}\n"
        f"  crop_sizes     : {args.crop_sizes}\n"
        f"  patients       : {args.patients if args.patients else '(stage1_dev 전체)'}\n"
        f"  limit          : {args.limit}\n"
        f"\n"
        f"  ablation config ({len(ABLATION_CONFIGS)}개):\n"
    )
    for cfg in ABLATION_CONFIGS:
        thr_str = ">0" if cfg.is_gt0 else str(cfg.threshold)
        print(f"    {cfg.name:<30} basis={cfg.basis}  crop={cfg.crop_size}  thr={thr_str}")

    print(
        f"\n"
        f"  [GUARD]\n"
        f"  stage2_holdout 포함 금지\n"
        f"  봉인 환자 포함 금지: {sorted(STAGE2_HOLDOUT_BLOCKED)}\n"
        f"  v1 outputs 쓰기 금지\n"
        f"  v2 경로 접근 금지\n"
        f"  파일 저장 없음 (console 출력 전용)\n"
        f"  runtime_summary.csv / error.csv append 금지\n"
        f"  crop npz / overlay PNG 생성 금지\n"
        f"  학습 / 평가 / score 재계산 금지\n"
        f"  p92.5/p90 사용 금지\n"
        f"{'=' * 70}\n"
    )

    dataset_root = load_dataset_root()
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root 없음: {dataset_root}")
    print(f"[INFO] dataset_root: {dataset_root}")

    split_df = load_split(patients=args.patients, limit=args.limit)
    n_target = len(split_df)
    print(f"[INFO] 처리 대상 환자: {n_target}명\n")

    patient_results: List[Dict] = []
    for idx, (_, row) in enumerate(split_df.iterrows()):
        patient_id = str(row["patient_id"])
        print(f"  [{idx + 1:>3}/{n_target}] {patient_id} ...", end=" ", flush=True)
        result = process_patient(row, dataset_root, args, args.crop_sizes)
        if result is None:
            print("SKIP (None 반환)")
            continue
        patient_results.append(result)
        if result["failed"]:
            print(f"FAILED")
        else:
            print(f"→ {result['n_candidates']}개 candidate (large_bbox={result['n_large_bbox']})")

    print(f"\n[INFO] 환자 처리 완료: {len(patient_results)}명")
    n_failed = sum(1 for r in patient_results if r["failed"])
    n_scored = len(patient_results) - n_failed
    print(f"[INFO] scored={n_scored}  failed={n_failed}\n")

    # 집계
    aggregated = aggregate_stats(patient_results, n_target, args)
    baseline_stats = aggregated.get(BASELINE_CONFIG_NAME)

    # config별 출력
    for cfg in ABLATION_CONFIGS:
        print_config_result(
            aggregated[cfg.name],
            baseline_stats if cfg.name != BASELINE_CONFIG_NAME else None,
            args.no_top_lists,
        )

    # 요약 표
    if baseline_stats is not None:
        print_summary_table(aggregated, baseline_stats)

    print(f"\n{'=' * 70}")
    print(f"  Phase 5.8 Label/Crop Ablation Diagnostic 완료")
    print(f"  완료 일시: {datetime.now().isoformat(timespec='seconds')}")
    print()
    print(f"  [확인]")
    print(f"  파일 생성/수정            : 0건")
    print(f"  runtime_summary.csv append: 없음")
    print(f"  error.csv append          : 없음")
    print(f"  crop npz / overlay PNG    : 없음")
    print(f"  v2 접촉                   : 없음")
    print(f"  stage2_holdout 접촉       : 없음")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
