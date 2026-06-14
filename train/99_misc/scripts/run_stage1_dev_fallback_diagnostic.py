#!/usr/bin/env python3
"""
run_stage1_dev_fallback_diagnostic.py

Second-Stage Lesion Refiner v1 — Phase 5.6 Fallback Diagnostic

목적:
- p95-only 기준과 여러 fallback 후보 규칙을 비교하는 diagnostic 전용 스크립트
- 파일 저장 없음 — console 출력 전용
- runtime_summary.csv / error.csv append 금지
- crop npz / overlay PNG 생성 금지
- p90/p92.5 임의 추정 금지: --threshold-p925 / --threshold-p90 값이 없으면 해당 rule skip

IMPORTANT:
- stage1_dev 154명만 처리 (stage2_holdout 154명 절대 포함 금지)
- 봉인 환자 감지 즉시 중단: LUNG1-089, LUNG1-231, LUNG1-372
- v1 outputs 하위 쓰기 금지
- v2 경로/파일 접근 금지
- ChatGPT 검토 + 사용자 승인 후에만 실행 가능
- 기존 build_stage1_dev_candidate_manifest.py 수정 금지

비교 rule:
  baseline_p95                  : p95 threshold (기본 14.3774), 모든 환자
  threshold_p92_5               : p92.5 threshold, 모든 환자 (--threshold-p925 필요)
  threshold_p90                 : p90 threshold, 모든 환자 (--threshold-p90 필요)
  patient_topk_K                : 환자별 top-K patch 기반 (K in --topk-values)
  hybrid_p95_replace_topk_zero_lc_K : p95 baseline + lesion=0 환자, p95 후보를 top-K로 교체 (replacement)
  hybrid_p95_add_topk_zero_lc_K     : p95 baseline + lesion=0 환자, top-K 후보를 p95에 추가 (additive)
  hybrid_p95_p92_5_zero_lc      : p95 baseline + lesion=0 환자에 p92.5 fallback
  hybrid_p95_p90_zero_lc        : p95 baseline + lesion=0 환자에 p90 fallback

py_compile 확인 (실행 아님):
  python -m py_compile scripts/run_stage1_dev_fallback_diagnostic.py

실행 (ChatGPT 검토 + 사용자 승인 후):
  source ~/ai_env/bin/activate && \\
  python scripts/run_stage1_dev_fallback_diagnostic.py
"""

import argparse
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yaml

# ============================================================
# 경로 상수 (v1 고정 — v2 접근 금지)
# ============================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
SCORE_DIR = REPO_ROOT / "outputs/position-aware-padim-v1/scores/padim_v1/lesion_by_patient"
SPLIT_CSV = (
    REPO_ROOT
    / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"
)
OUTPUT_BASE = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1"
PATHS_CONFIG = REPO_ROOT / "configs/paths.local.yaml"

# 쓰기 절대 금지
V1_OUTPUT_DIR = REPO_ROOT / "outputs/position-aware-padim-v1"

SCRIPT_NAME = "run_stage1_dev_fallback_diagnostic.py"

# ============================================================
# 안전 상수
# ============================================================
P95_THRESHOLD = 14.3774
DEFAULT_CROP_SIZE = 96
DEFAULT_CONTEXT_MARGIN = 32
DEFAULT_Z_RANGE = 1
MAX_BBOX_SIZE = 256

STAGE2_HOLDOUT_BLOCKED: Set[str] = {"LUNG1-089", "LUNG1-231", "LUNG1-372"}
WEAK_CASES: List[str] = ["MSD_lung_043", "MSD_lung_079", "MSD_lung_096"]


# ============================================================
# 안전 가드
# ============================================================
def check_not_v1_output(path: Path) -> None:
    """v1 outputs 하위 경로에 쓰기 시도 시 즉시 중단."""
    try:
        path.relative_to(V1_OUTPUT_DIR)
        raise RuntimeError(
            f"[GUARD] v1 outputs 하위 쓰기 금지: {path}\n"
            "v1 score CSV / lesion_mask는 read-only."
        )
    except ValueError:
        pass  # 정상: v1 output 하위가 아님


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
    return Path(val)


# ============================================================
# 데이터 로딩
# ============================================================
def load_split(
    limit: Optional[int] = None,
    patients: Optional[List[str]] = None,
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

    n_total_dev = len(df_dev)

    if patients:
        blocked_requested = [p for p in patients if p in STAGE2_HOLDOUT_BLOCKED]
        if blocked_requested:
            raise RuntimeError(
                f"[GUARD] 봉인 환자가 --patients에 포함됨: {blocked_requested}"
            )
        dev_pids = set(df_dev["patient_id"].tolist())
        requested_set = set(patients)
        missing_set = requested_set - dev_pids
        if missing_set:
            holdout_pids = set(
                df[df["stage_split"] == "stage2_holdout"]["patient_id"].tolist()
            )
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

    if limit is not None and limit > 0 and len(df_dev) > limit:
        df_dev = df_dev.head(limit)
        print(
            f"[INFO] --limit {limit} 적용: {len(df_dev)}명만 처리 "
            f"(전체 stage1_dev={n_total_dev}명)"
        )

    return df_dev.reset_index(drop=True)


def load_score_csv(patient_id: str) -> pd.DataFrame:
    """v1 score CSV 로드 (read-only)."""
    csv_path = SCORE_DIR / f"{patient_id}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"score CSV 없음: {csv_path}")
    return pd.read_csv(csv_path, encoding="utf-8-sig")


# ============================================================
# Candidate 생성 (기존 스크립트 로직 복사 — 원본 미수정)
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


def compute_lesion_overlap(
    lesion_mask: np.ndarray,
    z: int, y0: int, x0: int, y1: int, x1: int,
) -> Tuple[int, float]:
    if z < 0 or z >= lesion_mask.shape[0]:
        return 0, 0.0
    crop = lesion_mask[z, y0:y1, x0:x1]
    area = (y1 - y0) * (x1 - x0)
    if area == 0:
        return 0, 0.0
    lesion_px = int(np.sum(crop > 0))
    return lesion_px, float(lesion_px / area)


def compute_overlap_by_basis(
    lesion_mask: np.ndarray,
    label_basis: str,
    z_center: int, z_lo: int, z_hi: int,
    cy0: int, cx0: int, cy1: int, cx1: int,
    fc_y0: int, fc_x0: int, fc_y1: int, fc_x1: int,
    raw_y0: int, raw_x0: int, raw_y1: int, raw_x1: int,
) -> float:
    """label_basis에 따라 선택된 lesion overlap ratio 반환."""
    if label_basis == "context_z_center":
        _, ratio = compute_lesion_overlap(lesion_mask, z_center, cy0, cx0, cy1, cx1)
    elif label_basis == "fixed_crop_z_center":
        _, ratio = compute_lesion_overlap(lesion_mask, z_center, fc_y0, fc_x0, fc_y1, fc_x1)
    elif label_basis == "raw_bbox_z_center":
        _, ratio = compute_lesion_overlap(lesion_mask, z_center, raw_y0, raw_x0, raw_y1, raw_x1)
    elif label_basis == "zstack_max":
        ratio = 0.0
        for z in range(z_lo, z_hi + 1):
            _, r = compute_lesion_overlap(lesion_mask, z, cy0, cx0, cy1, cx1)
            if r > ratio:
                ratio = r
    else:
        _, ratio = compute_lesion_overlap(lesion_mask, z_center, fc_y0, fc_x0, fc_y1, fc_x1)
    return ratio


def assign_labels(lesion_overlap_ratio: float) -> Tuple[str, str]:
    """
    rd4ad_label, binary_label 반환.
    RD4AD 학습에서 lesion_candidate는 학습 target 아님 — 평가 ground truth 전용.
    """
    if lesion_overlap_ratio == 0.0:
        return "normal_like", "hard_negative"
    elif lesion_overlap_ratio >= 0.10:
        return "lesion_candidate", "positive"
    else:
        return "ambiguous", "ambiguous"


# ============================================================
# 환자 단위 처리 (threshold 또는 topk 방식)
# ============================================================
def build_candidates_for_patient(
    row: pd.Series,
    dataset_root: Path,
    args: argparse.Namespace,
    threshold: Optional[float] = None,
    topk: Optional[int] = None,
) -> List[Dict]:
    """
    단일 환자의 candidate 목록 생성.
    threshold 또는 topk 중 하나를 반드시 지정해야 한다.
    파일 저장 없음 — read-only 작업만 수행.
    crop npz / overlay PNG 생성 없음.
    """
    if threshold is None and topk is None:
        raise ValueError("threshold 또는 topk 중 하나는 반드시 지정해야 합니다.")

    patient_id = str(row["patient_id"])
    safe_id = str(row["safe_id"])
    stage_split = str(row["stage_split"])

    check_stage1_dev(patient_id, stage_split)

    score_df = load_score_csv(patient_id)

    if threshold is not None:
        pos_df = score_df[score_df["padim_score"] >= threshold].copy()
    else:
        # topk: 전체에서 상위 K개 patch를 score 기준으로 추출
        pos_df = score_df.nlargest(topk, "padim_score").copy()

    if pos_df.empty:
        return []

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

        bbox_h = cy1 - cy0
        bbox_w = cx1 - cx0
        bbox_too_large = bbox_h > MAX_BBOX_SIZE or bbox_w > MAX_BBOX_SIZE

        z_lo = max(0, z_center - args.z_range)
        z_hi = min(ct_z - 1, z_center + args.z_range)

        # fixed_crop 좌표 (RD4AD 고정 입력용)
        bbox_center_y = (cand["y0"] + cand["y1"]) // 2
        bbox_center_x = (cand["x0"] + cand["x1"]) // 2
        half_crop = args.crop_size // 2
        fc_y0 = max(0, bbox_center_y - half_crop)
        fc_x0 = max(0, bbox_center_x - half_crop)
        fc_y1 = min(ct_h, fc_y0 + args.crop_size)
        fc_x1 = min(ct_w, fc_x0 + args.crop_size)
        fc_y0 = max(0, fc_y1 - args.crop_size)
        fc_x0 = max(0, fc_x1 - args.crop_size)

        selected_ratio = compute_overlap_by_basis(
            lesion_mask, args.label_basis,
            z_center, z_lo, z_hi,
            cy0, cx0, cy1, cx1,
            fc_y0, fc_x0, fc_y1, fc_x1,
            cand["y0"], cand["x0"], cand["y1"], cand["x1"],
        )

        rd4ad_label, binary_label = assign_labels(selected_ratio)

        candidates.append({
            "patient_id": patient_id,
            "z_center": z_center,
            "y0": cand["y0"],
            "x0": cand["x0"],
            "y1": cand["y1"],
            "x1": cand["x1"],
            "y0_fixed_crop": fc_y0,
            "x0_fixed_crop": fc_x0,
            "y1_fixed_crop": fc_y1,
            "x1_fixed_crop": fc_x1,
            "rd4ad_label": rd4ad_label,
            "binary_label": binary_label,
            "bbox_too_large": bbox_too_large,
            "lesion_overlap_ratio": round(selected_ratio, 6),
            "mean_padim_score": round(cand["mean_score"], 6),
            "max_padim_score": round(cand["max_score"], 6),
        })

    return candidates


# ============================================================
# Candidate union helper (additive hybrid용)
# ============================================================
def merge_candidate_lists_unique(
    base_cands: List[Dict],
    fallback_cands: List[Dict],
) -> List[Dict]:
    """
    base_cands와 fallback_cands를 합쳐 중복 제거.
    중복 key: (patient_id, z_center, y0, x0, y1, x1)
    중복 시 max_padim_score 큰 후보 유지.
    파일 저장 없음 — 메모리 연산만 수행.
    """
    merged: Dict[Tuple, Dict] = {}
    for c in base_cands + fallback_cands:
        key = (
            c["patient_id"],
            c["z_center"],
            c["y0"],
            c["x0"],
            c["y1"],
            c["x1"],
        )
        if key not in merged or c["max_padim_score"] > merged[key]["max_padim_score"]:
            merged[key] = c
    return list(merged.values())


# ============================================================
# Rule 실행
# ============================================================
def run_rule(
    rule_name: str,
    split_df: pd.DataFrame,
    dataset_root: Path,
    args: argparse.Namespace,
    threshold: Optional[float] = None,
    topk: Optional[int] = None,
    zero_lc_pids: Optional[Set[str]] = None,
    baseline_patient_cands: Optional[Dict[str, List[Dict]]] = None,
    additive: bool = False,
) -> Dict[str, Any]:
    """
    단일 rule을 실행하고 환자별 candidate 목록과 실패 목록을 반환한다.

    zero_lc_pids가 지정되면 hybrid mode:
      - zero_lc_pids가 아닌 환자: baseline_patient_cands 그대로 사용
      - zero_lc_pids 환자:
          additive=False (replacement): threshold/topk 후보로 교체
          additive=True  (additive):    baseline 후보 유지 + topk/threshold 후보를 추가
    파일 저장 없음.
    """
    patient_cands: Dict[str, List[Dict]] = {}
    failed_patients: List[str] = []

    for _, row in split_df.iterrows():
        pid = str(row["patient_id"])

        if zero_lc_pids is not None and pid not in zero_lc_pids:
            # hybrid: 이 환자는 baseline 결과 그대로 사용
            patient_cands[pid] = (baseline_patient_cands or {}).get(pid, [])
            continue

        try:
            cands = build_candidates_for_patient(
                row, dataset_root, args,
                threshold=threshold,
                topk=topk,
            )
            if additive and zero_lc_pids is not None and pid in zero_lc_pids:
                # additive: p95 후보 유지 + topk/threshold 후보 추가 (중복 제거)
                base = (baseline_patient_cands or {}).get(pid, [])
                cands = merge_candidate_lists_unique(base, cands)
            patient_cands[pid] = cands
        except Exception as exc:
            patient_cands[pid] = []
            failed_patients.append(pid)

    return {
        "rule_name": rule_name,
        "patient_cands": patient_cands,
        "failed_patients": failed_patients,
    }


# ============================================================
# 지표 계산
# ============================================================
def compute_metrics(
    rule_result: Dict[str, Any],
    split_df: pd.DataFrame,
    baseline_metrics: Optional[Dict] = None,
) -> Dict[str, Any]:
    rule_name = rule_result["rule_name"]
    patient_cands = rule_result["patient_cands"]
    failed_patients = rule_result["failed_patients"]

    all_cands = [c for cands in patient_cands.values() for c in cands]
    n_total = len(all_cands)
    n_patients_target = len(split_df)
    n_patients_failed = len(failed_patients)
    n_patients_scored = n_patients_target - n_patients_failed

    rd4ad_summary: Dict[str, int] = defaultdict(int)
    binary_summary: Dict[str, int] = defaultdict(int)
    patient_lc: Dict[str, int] = {}
    patient_total: Dict[str, int] = {}
    patient_hn: Dict[str, int] = {}
    patient_lb: Dict[str, int] = {}
    patient_amb: Dict[str, int] = {}

    for pid, cands in patient_cands.items():
        lc = hn = lb = amb = 0
        for c in cands:
            rd4ad_summary[c["rd4ad_label"]] += 1
            binary_summary[c["binary_label"]] += 1
            if c["rd4ad_label"] == "lesion_candidate":
                lc += 1
            if c["binary_label"] == "hard_negative":
                hn += 1
            if c.get("bbox_too_large", False):
                lb += 1
            if c["rd4ad_label"] == "ambiguous":
                amb += 1
        patient_lc[pid] = lc
        patient_total[pid] = len(cands)
        patient_hn[pid] = hn
        patient_lb[pid] = lb
        patient_amb[pid] = amb

    n_lc = rd4ad_summary.get("lesion_candidate", 0)
    n_hn = binary_summary.get("hard_negative", 0)
    n_pos = binary_summary.get("positive", 0)
    n_amb = rd4ad_summary.get("ambiguous", 0)
    n_lb = sum(patient_lb.values())

    zero_lc_pids = sorted([pid for pid, cnt in patient_lc.items() if cnt == 0])
    n_zero_lc = len(zero_lc_pids)

    counts_arr = np.array(list(patient_total.values()), dtype=float)
    count_min = int(counts_arr.min()) if len(counts_arr) > 0 else 0
    count_median = float(np.median(counts_arr)) if len(counts_arr) > 0 else 0.0
    count_max = int(counts_arr.max()) if len(counts_arr) > 0 else 0

    weak_case_info: Dict[str, Dict] = {}
    for wc in WEAK_CASES:
        cands = patient_cands.get(wc, [])
        weak_case_info[wc] = {
            "total": len(cands),
            "lesion_candidate": sum(1 for c in cands if c["rd4ad_label"] == "lesion_candidate"),
            "ambiguous": sum(1 for c in cands if c["rd4ad_label"] == "ambiguous"),
        }

    hn_pos_ratio = (n_hn / n_pos) if n_pos > 0 else None

    delta_total: Optional[int] = None
    delta_total_pct: Optional[float] = None
    delta_zero_lc: Optional[int] = None
    if baseline_metrics is not None:
        base_total = baseline_metrics.get("n_candidates_total", 0)
        base_zero_lc = baseline_metrics.get("n_zero_lc", 0)
        delta_total = n_total - base_total
        delta_total_pct = (delta_total / base_total * 100) if base_total > 0 else None
        delta_zero_lc = n_zero_lc - base_zero_lc

    top10_cand = sorted(patient_total.items(), key=lambda x: x[1], reverse=True)[:10]
    top10_lb = sorted(patient_lb.items(), key=lambda x: x[1], reverse=True)[:10]
    top10_lc = sorted(patient_lc.items(), key=lambda x: x[1], reverse=True)[:10]
    top10_hn = sorted(patient_hn.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "rule_name": rule_name,
        "rule_status": "ran",
        "skip_reason": None,
        "n_patients_target": n_patients_target,
        "n_patients_scored": n_patients_scored,
        "n_patients_failed": n_patients_failed,
        "failed_patients": failed_patients,
        "n_candidates_total": n_total,
        "count_min": count_min,
        "count_median": count_median,
        "count_max": count_max,
        "rd4ad_summary": dict(rd4ad_summary),
        "binary_summary": dict(binary_summary),
        "n_large_bbox": n_lb,
        "large_bbox_ratio": round(n_lb / n_total, 4) if n_total > 0 else 0.0,
        "n_ambiguous": n_amb,
        "ambiguous_ratio": round(n_amb / n_total, 4) if n_total > 0 else 0.0,
        "n_lesion_candidate": n_lc,
        "n_zero_lc": n_zero_lc,
        "zero_lc_pids": zero_lc_pids,
        "n_hard_negative": n_hn,
        "n_positive": n_pos,
        "hn_pos_ratio": round(hn_pos_ratio, 1) if hn_pos_ratio is not None else None,
        "weak_case_info": weak_case_info,
        "delta_total": delta_total,
        "delta_total_pct": delta_total_pct,
        "delta_zero_lc": delta_zero_lc,
        "top10_cand": top10_cand,
        "top10_lb": top10_lb,
        "top10_lc": top10_lc,
        "top10_hn": top10_hn,
        "patient_cands": patient_cands,
    }


def make_skipped_metrics(rule_name: str, skip_reason: str) -> Dict[str, Any]:
    return {
        "rule_name": rule_name,
        "rule_status": "skipped",
        "skip_reason": skip_reason,
    }


# ============================================================
# 출력
# ============================================================
def print_rule_result(m: Dict[str, Any]) -> None:
    if m["rule_status"] == "skipped":
        print(
            f"\n{'=' * 60}\n"
            f"  Rule: {m['rule_name']}\n"
            f"  Status: SKIPPED — {m['skip_reason']}\n"
            f"{'=' * 60}"
        )
        return

    n_total = m["n_candidates_total"]
    hn_str = (
        f"{m['hn_pos_ratio']:.1f}:1"
        if m["hn_pos_ratio"] is not None
        else "inf (positive=0)"
    )

    delta_lines = ""
    if m["delta_total"] is not None:
        pct_str = (
            f"{m['delta_total_pct']:+.1f}%"
            if m["delta_total_pct"] is not None
            else "N/A"
        )
        delta_lines += f"  p95 대비 candidate 증가: {m['delta_total']:+d} ({pct_str})\n"
    if m["delta_zero_lc"] is not None:
        delta_lines += (
            f"  p95 대비 lesion_candidate=0 환자 변화: {m['delta_zero_lc']:+d}\n"
        )

    print(
        f"\n{'=' * 60}\n"
        f"  Rule: {m['rule_name']}  [Status: {m['rule_status']}]\n"
        f"{'=' * 60}\n"
        f"  환자: target={m['n_patients_target']}, "
        f"scored={m['n_patients_scored']}, failed={m['n_patients_failed']}\n"
        f"  candidate 총 수: {n_total}\n"
        f"  환자별 candidate: "
        f"min={m['count_min']}, median={m['count_median']:.1f}, max={m['count_max']}\n"
        f"  rd4ad_label  : {m['rd4ad_summary']}\n"
        f"  binary_label : {m['binary_summary']}\n"
        f"  large_bbox   : {m['n_large_bbox']} "
        f"({100 * m['large_bbox_ratio']:.1f}%)\n"
        f"  ambiguous    : {m['n_ambiguous']} "
        f"({100 * m['ambiguous_ratio']:.1f}%)\n"
        f"  lesion_candidate : {m['n_lesion_candidate']}\n"
        f"  lesion_candidate=0 환자: {m['n_zero_lc']}명 / {m['n_patients_target']}명\n"
        f"  hard_negative:positive 비율: {hn_str}\n"
        f"{delta_lines}"
    )

    print("  weak_case별 label 분포:")
    for wc, info in m["weak_case_info"].items():
        print(
            f"    {wc}: total={info['total']}, "
            f"lesion_candidate={info['lesion_candidate']}, "
            f"ambiguous={info['ambiguous']}"
        )

    if m["zero_lc_pids"]:
        print(
            f"\n  lesion_candidate=0 환자 목록 ({m['n_zero_lc']}명):\n"
            f"    {m['zero_lc_pids']}"
        )

    print(f"\n  candidate 수 상위 Top 10:")
    print(f"    {'patient_id':<22} {'total':>7}")
    for pid, cnt in m["top10_cand"]:
        print(f"    {pid:<22} {cnt:>7}")

    print(f"\n  large_bbox 상위 Top 10:")
    print(f"    {'patient_id':<22} {'large_bbox':>10}")
    for pid, cnt in m["top10_lb"]:
        print(f"    {pid:<22} {cnt:>10}")

    print(f"\n  lesion_candidate 상위 Top 10:")
    print(f"    {'patient_id':<22} {'lesion_cand':>12}")
    for pid, cnt in m["top10_lc"]:
        print(f"    {pid:<22} {cnt:>12}")

    print(f"\n  hard_negative 상위 Top 10:")
    print(f"    {'patient_id':<22} {'hard_negative':>14}")
    for pid, cnt in m["top10_hn"]:
        print(f"    {pid:<22} {cnt:>14}")

    if m["n_patients_failed"] > 0:
        print(
            f"\n  [WARN] 실패 환자 ({m['n_patients_failed']}명): "
            f"{m['failed_patients']}"
        )


def print_comparison_table(all_metrics: List[Dict[str, Any]]) -> None:
    print(f"\n{'=' * 70}")
    print("  Fallback Diagnostic 비교 요약")
    print(f"{'=' * 70}")
    hdr = (
        f"  {'rule':<38} {'status':<8}"
        f" {'n_cand':>8} {'delta':>8}"
        f" {'n_lc':>6} {'zero_lc':>8}"
        f" {'hn:pos':>9} {'n_lb':>6}"
    )
    print(hdr)
    print(f"  {'-' * 97}")
    for m in all_metrics:
        if m["rule_status"] == "skipped":
            print(f"  {m['rule_name']:<38} {'SKIP':<8}")
            continue
        delta_str = (
            f"{m['delta_total']:+d}"
            if m["delta_total"] is not None
            else "N/A"
        )
        hn_str = (
            f"{m['hn_pos_ratio']:.1f}:1"
            if m["hn_pos_ratio"] is not None
            else "inf"
        )
        print(
            f"  {m['rule_name']:<38} {'ran':<8}"
            f" {m['n_candidates_total']:>8} {delta_str:>8}"
            f" {m['n_lesion_candidate']:>6} {m['n_zero_lc']:>8}"
            f" {hn_str:>9} {m['n_large_bbox']:>6}"
        )
    print(f"{'=' * 70}")


# ============================================================
# argparse
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Second-Stage Lesion Refiner v1 Phase 5.6 — "
            "Fallback Diagnostic (console-only, no file write)"
        )
    )
    parser.add_argument(
        "--patients", nargs="+", default=None,
        help="처리할 patient_id 목록 (stage1_dev 내에서만 허용, debug용)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="처리할 환자 수 상한 (debug용)",
    )
    parser.add_argument(
        "--threshold-p95", type=float, default=P95_THRESHOLD, dest="threshold_p95",
        help=f"baseline p95 threshold (기본 {P95_THRESHOLD})",
    )
    parser.add_argument(
        "--threshold-p925", type=float, default=None, dest="threshold_p925",
        help=(
            "p92.5 threshold 값. 없으면 해당 rule skip. "
            "임의 추정 금지 — 검증된 값만 사용."
        ),
    )
    parser.add_argument(
        "--threshold-p90", type=float, default=None, dest="threshold_p90",
        help=(
            "p90 threshold 값. 없으면 해당 rule skip. "
            "임의 추정 금지 — 검증된 값만 사용."
        ),
    )
    parser.add_argument(
        "--topk-values", nargs="+", type=int, default=[20, 50], dest="topk_values",
        help="patient top-k 후보에서 사용할 K 값 목록 (기본 [20, 50])",
    )
    parser.add_argument(
        "--label-basis",
        dest="label_basis",
        type=str,
        choices=["context_z_center", "fixed_crop_z_center", "raw_bbox_z_center", "zstack_max"],
        default="fixed_crop_z_center",
        help="label 계산 기준 (기본 fixed_crop_z_center)",
    )
    parser.add_argument(
        "--context-margin", type=int, default=DEFAULT_CONTEXT_MARGIN, dest="context_margin",
        help=f"context margin px (기본 {DEFAULT_CONTEXT_MARGIN}, 변경 금지)",
    )
    parser.add_argument(
        "--z-range", type=int, default=DEFAULT_Z_RANGE, dest="z_range",
        help=f"2.5D z 범위 ±z_range (기본 {DEFAULT_Z_RANGE}, 변경 금지)",
    )
    parser.add_argument(
        "--crop-size", type=int, default=DEFAULT_CROP_SIZE, dest="crop_size",
        help=f"fixed crop 크기 px (기본 {DEFAULT_CROP_SIZE}, 변경 금지)",
    )
    return parser.parse_args()


# ============================================================
# 메인
# ============================================================
def main() -> None:
    args = parse_args()

    print(
        f"\n{'=' * 60}\n"
        f"  Second-Stage Lesion Refiner v1 — Phase 5.6\n"
        f"  Fallback Diagnostic (console-only)\n"
        f"  실행 일시   : {datetime.now().isoformat(timespec='seconds')}\n"
        f"  label_basis  : {args.label_basis}\n"
        f"  threshold_p95: {args.threshold_p95}\n"
        f"  threshold_p925: {args.threshold_p925} "
        f"({'사용 가능' if args.threshold_p925 is not None else 'SKIP'})\n"
        f"  threshold_p90 : {args.threshold_p90} "
        f"({'사용 가능' if args.threshold_p90 is not None else 'SKIP'})\n"
        f"  topk_values  : {args.topk_values}\n"
        f"\n"
        f"  [GUARD]\n"
        f"  stage2_holdout 포함 금지\n"
        f"  봉인 환자 포함 금지: {sorted(STAGE2_HOLDOUT_BLOCKED)}\n"
        f"  v1 outputs 쓰기 금지\n"
        f"  v2 경로 접근 금지\n"
        f"  파일 저장 없음 (console 출력 전용)\n"
        f"  runtime_summary.csv / error.csv append 금지\n"
        f"  crop npz / overlay PNG 생성 금지\n"
        f"{'=' * 60}\n"
    )

    dataset_root = load_dataset_root()
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root 없음: {dataset_root}")

    split_df = load_split(args.limit, args.patients)
    n_target = len(split_df)
    print(f"[INFO] 처리 대상 환자: {n_target}명\n")

    start_total = datetime.now()
    all_metrics: List[Dict[str, Any]] = []
    baseline_metrics: Optional[Dict] = None
    baseline_patient_cands: Optional[Dict[str, List[Dict]]] = None
    baseline_zero_lc_pids: Optional[Set[str]] = None

    # ── Rule 1: baseline_p95 ──────────────────────────────────
    rule_label = "baseline_p95"
    print(f"[Rule] {rule_label} 실행 중...")
    r1 = run_rule(rule_label, split_df, dataset_root, args, threshold=args.threshold_p95)
    m1 = compute_metrics(r1, split_df, baseline_metrics=None)
    all_metrics.append(m1)
    print_rule_result(m1)
    baseline_metrics = m1
    baseline_patient_cands = r1["patient_cands"]
    baseline_zero_lc_pids = set(m1["zero_lc_pids"])

    # ── Rule 2: threshold_p92_5 ───────────────────────────────
    rule_label = "threshold_p92_5"
    print(f"\n[Rule] {rule_label} 실행 중...")
    if args.threshold_p925 is None:
        m2 = make_skipped_metrics(
            rule_label,
            "--threshold-p925 값 없음. 임의 추정 금지. 값 확인 후 재실행.",
        )
    else:
        r2 = run_rule(rule_label, split_df, dataset_root, args, threshold=args.threshold_p925)
        m2 = compute_metrics(r2, split_df, baseline_metrics=baseline_metrics)
    all_metrics.append(m2)
    print_rule_result(m2)

    # ── Rule 3: threshold_p90 ─────────────────────────────────
    rule_label = "threshold_p90"
    print(f"\n[Rule] {rule_label} 실행 중...")
    if args.threshold_p90 is None:
        m3 = make_skipped_metrics(
            rule_label,
            "--threshold-p90 값 없음. 임의 추정 금지. 값 확인 후 재실행.",
        )
    else:
        r3 = run_rule(rule_label, split_df, dataset_root, args, threshold=args.threshold_p90)
        m3 = compute_metrics(r3, split_df, baseline_metrics=baseline_metrics)
    all_metrics.append(m3)
    print_rule_result(m3)

    # ── Rule 4+: patient_topk_K ───────────────────────────────
    topk_metrics: List[Dict] = []
    for k in args.topk_values:
        rule_label = f"patient_topk_{k}"
        print(f"\n[Rule] {rule_label} 실행 중...")
        r_k = run_rule(rule_label, split_df, dataset_root, args, topk=k)
        m_k = compute_metrics(r_k, split_df, baseline_metrics=baseline_metrics)
        all_metrics.append(m_k)
        topk_metrics.append(m_k)
        print_rule_result(m_k)

    # ── Rule: hybrid_p95_replace_topk_zero_lc_K (replacement) ─
    # lesion=0 환자에 대해 p95 후보를 top-k 후보로 교체 (replacement fallback)
    for k in args.topk_values:
        rule_label = f"hybrid_p95_replace_topk_zero_lc_{k}"
        print(f"\n[Rule] {rule_label} 실행 중...")
        r_h = run_rule(
            rule_label, split_df, dataset_root, args,
            topk=k,
            zero_lc_pids=baseline_zero_lc_pids,
            baseline_patient_cands=baseline_patient_cands,
            additive=False,
        )
        m_h = compute_metrics(r_h, split_df, baseline_metrics=baseline_metrics)
        all_metrics.append(m_h)
        print_rule_result(m_h)

    # ── Rule: hybrid_p95_add_topk_zero_lc_K (additive) ────────
    # lesion=0 환자에 대해 p95 후보를 유지하면서 top-k 후보를 추가 (true additive fallback)
    for k in args.topk_values:
        rule_label = f"hybrid_p95_add_topk_zero_lc_{k}"
        print(f"\n[Rule] {rule_label} 실행 중...")
        r_ha = run_rule(
            rule_label, split_df, dataset_root, args,
            topk=k,
            zero_lc_pids=baseline_zero_lc_pids,
            baseline_patient_cands=baseline_patient_cands,
            additive=True,
        )
        m_ha = compute_metrics(r_ha, split_df, baseline_metrics=baseline_metrics)
        all_metrics.append(m_ha)
        print_rule_result(m_ha)

    # ── Rule: hybrid_p95_p92_5_zero_lc ───────────────────────
    rule_label = "hybrid_p95_p92_5_zero_lc"
    print(f"\n[Rule] {rule_label} 실행 중...")
    if args.threshold_p925 is None:
        m_h925 = make_skipped_metrics(
            rule_label,
            "--threshold-p925 값 없음. 임의 추정 금지. 값 확인 후 재실행.",
        )
    else:
        r_h925 = run_rule(
            rule_label, split_df, dataset_root, args,
            threshold=args.threshold_p925,
            zero_lc_pids=baseline_zero_lc_pids,
            baseline_patient_cands=baseline_patient_cands,
        )
        m_h925 = compute_metrics(r_h925, split_df, baseline_metrics=baseline_metrics)
    all_metrics.append(m_h925)
    print_rule_result(m_h925)

    # ── Rule: hybrid_p95_p90_zero_lc ─────────────────────────
    rule_label = "hybrid_p95_p90_zero_lc"
    print(f"\n[Rule] {rule_label} 실행 중...")
    if args.threshold_p90 is None:
        m_h90 = make_skipped_metrics(
            rule_label,
            "--threshold-p90 값 없음. 임의 추정 금지. 값 확인 후 재실행.",
        )
    else:
        r_h90 = run_rule(
            rule_label, split_df, dataset_root, args,
            threshold=args.threshold_p90,
            zero_lc_pids=baseline_zero_lc_pids,
            baseline_patient_cands=baseline_patient_cands,
        )
        m_h90 = compute_metrics(r_h90, split_df, baseline_metrics=baseline_metrics)
    all_metrics.append(m_h90)
    print_rule_result(m_h90)

    # ── 비교 요약 출력 ────────────────────────────────────────
    print_comparison_table(all_metrics)

    elapsed = (datetime.now() - start_total).total_seconds()
    print(
        f"\n{'=' * 60}\n"
        f"  Fallback Diagnostic 완료\n"
        f"  완료 일시     : {datetime.now().isoformat(timespec='seconds')}\n"
        f"  총 경과 시간  : {elapsed:.1f}s\n"
        f"\n"
        f"  [확인]\n"
        f"  파일 생성/수정: 0건\n"
        f"  runtime_summary.csv append: 없음\n"
        f"  error.csv append: 없음\n"
        f"  crop npz / overlay PNG: 없음\n"
        f"  v2 접촉: 없음\n"
        f"\n"
        f"  다음 단계: ChatGPT 검토 후 최적 fallback 선택 → 별도 승인 후 manifest 생성\n"
        f"{'=' * 60}\n"
    )


if __name__ == "__main__":
    main()
