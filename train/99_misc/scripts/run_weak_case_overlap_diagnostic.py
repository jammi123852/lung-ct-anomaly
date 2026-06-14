#!/usr/bin/env python3
"""
run_weak_case_overlap_diagnostic.py

Second-Stage Lesion Refiner v1 — Phase 5.7 Weak Case Overlap Diagnostic

목적:
- weak_case 3명(MSD_lung_043, MSD_lung_079, MSD_lung_096)이 왜 lesion_candidate=0인지 확인
- 5가지 label_basis × 2가지 candidate 조건 × 4가지 overlap threshold 비교
- console 출력 전용 — 파일 저장 없음

IMPORTANT:
- stage1_dev만 처리 (stage2_holdout 절대 포함 금지)
- 봉인 환자 감지 즉시 중단: LUNG1-089, LUNG1-231, LUNG1-372
- v1 outputs 하위 쓰기 금지
- v2 경로 접근 금지
- 파일 저장 없음 (CSV/JSON/PNG/NPZ 생성 금지)
- runtime_summary.csv / error.csv append 금지
- 학습/평가/score 재계산 금지
- 스크립트 실행은 ChatGPT 검토 + 사용자 승인 후에만 가능

비교 기준:
  candidate 조건 1: p95 threshold (기본 14.3774)
  candidate 조건 2: patient top-k (기본 50개, replacement_50 맥락)

  label_basis:
    context_z_center       : context margin bbox + z_center slice
    fixed_crop_z_center    : fixed 96x96 crop + z_center slice (현재 기준)
    raw_bbox_z_center      : raw merge bbox (margin 없음) + z_center slice
    zstack_max             : context bbox 기반 z_lo~z_hi 전체 중 max overlap
    fixed_crop_zstack_max  : fixed crop bbox 기반 z_lo~z_hi 전체 중 max overlap (diagnostic 신규)

  overlap threshold: >0, >=0.01, >=0.05, >=0.10

py_compile 확인 (실행 아님):
  python -m py_compile scripts/run_weak_case_overlap_diagnostic.py

실행 (ChatGPT 검토 + 사용자 승인 후):
  source ~/ai_env/bin/activate && \\
  python scripts/run_weak_case_overlap_diagnostic.py
"""

import argparse
import sys
from collections import defaultdict
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
SCORE_DIR = REPO_ROOT / "outputs/position-aware-padim-v1/scores/padim_v1/lesion_by_patient"
SPLIT_CSV = (
    REPO_ROOT
    / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"
)
PATHS_CONFIG = REPO_ROOT / "configs/paths.local.yaml"

# 쓰기 절대 금지
V1_OUTPUT_DIR = REPO_ROOT / "outputs/position-aware-padim-v1"

SCRIPT_NAME = "run_weak_case_overlap_diagnostic.py"

# ============================================================
# 안전 상수
# ============================================================
P95_THRESHOLD = 14.3774
DEFAULT_CROP_SIZE = 96
DEFAULT_CONTEXT_MARGIN = 32
DEFAULT_Z_RANGE = 1

STAGE2_HOLDOUT_BLOCKED: Set[str] = {"LUNG1-089", "LUNG1-231", "LUNG1-372"}
WEAK_CASES: List[str] = ["MSD_lung_043", "MSD_lung_079", "MSD_lung_096"]

ALL_LABEL_BASES: List[str] = [
    "context_z_center",
    "fixed_crop_z_center",
    "raw_bbox_z_center",
    "zstack_max",
    "fixed_crop_zstack_max",
]


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
        pass


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
def load_split(patients: Optional[List[str]] = None) -> pd.DataFrame:
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

    return df_dev.reset_index(drop=True)


def load_score_csv(patient_id: str) -> pd.DataFrame:
    """v1 score CSV 로드 (read-only)."""
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


def compute_all_basis_overlap(
    lesion_mask: np.ndarray,
    z_center: int, z_lo: int, z_hi: int,
    cy0: int, cx0: int, cy1: int, cx1: int,
    fc_y0: int, fc_x0: int, fc_y1: int, fc_x1: int,
    raw_y0: int, raw_x0: int, raw_y1: int, raw_x1: int,
) -> Dict:
    """5가지 label_basis 전체의 overlap ratio 반환."""
    # context_z_center
    _, ctx_ratio = compute_lesion_overlap(
        lesion_mask, z_center, cy0, cx0, cy1, cx1
    )
    # fixed_crop_z_center
    _, fc_ratio = compute_lesion_overlap(
        lesion_mask, z_center, fc_y0, fc_x0, fc_y1, fc_x1
    )
    # raw_bbox_z_center
    _, raw_ratio = compute_lesion_overlap(
        lesion_mask, z_center, raw_y0, raw_x0, raw_y1, raw_x1
    )

    # zstack_max (context bbox 기반 z_lo~z_hi 전체 순회)
    zstack_ratio = 0.0
    zstack_z_of_max = z_center
    for z in range(z_lo, z_hi + 1):
        _, r = compute_lesion_overlap(lesion_mask, z, cy0, cx0, cy1, cx1)
        if r > zstack_ratio:
            zstack_ratio = r
            zstack_z_of_max = z

    # fixed_crop_zstack_max (fixed crop bbox 기반 z_lo~z_hi 전체 순회 — diagnostic 신규 기준)
    fc_zstack_ratio = 0.0
    fc_zstack_z_of_max = z_center
    for z in range(z_lo, z_hi + 1):
        _, r = compute_lesion_overlap(lesion_mask, z, fc_y0, fc_x0, fc_y1, fc_x1)
        if r > fc_zstack_ratio:
            fc_zstack_ratio = r
            fc_zstack_z_of_max = z

    return {
        "context_z_center": ctx_ratio,
        "fixed_crop_z_center": fc_ratio,
        "raw_bbox_z_center": raw_ratio,
        "zstack_max": zstack_ratio,
        "fixed_crop_zstack_max": fc_zstack_ratio,
        "zstack_z_of_max": zstack_z_of_max,
        "fc_zstack_z_of_max": fc_zstack_z_of_max,
    }


# ============================================================
# Enriched candidate 생성 (모든 basis overlap 포함)
# ============================================================
def build_enriched_candidates(
    row: pd.Series,
    ct_vol: np.ndarray,
    lesion_mask: np.ndarray,
    args: argparse.Namespace,
    threshold: Optional[float] = None,
    topk: Optional[int] = None,
) -> List[Dict]:
    """
    단일 환자 candidate 생성 — 5가지 basis overlap 전부 포함.
    파일 저장 없음, read-only 작업만 수행.
    """
    if threshold is None and topk is None:
        raise ValueError("threshold 또는 topk 중 하나는 반드시 지정해야 합니다.")

    patient_id = str(row["patient_id"])
    stage_split = str(row["stage_split"])

    check_stage1_dev(patient_id, stage_split)

    score_df = load_score_csv(patient_id)

    if threshold is not None:
        pos_df = score_df[score_df["padim_score"] >= threshold].copy()
    else:
        pos_df = score_df.nlargest(topk, "padim_score").copy()

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

        overlap = compute_all_basis_overlap(
            lesion_mask,
            z_center, z_lo, z_hi,
            cy0, cx0, cy1, cx1,
            fc_y0, fc_x0, fc_y1, fc_x1,
            cand["y0"], cand["x0"], cand["y1"], cand["x1"],
        )

        candidates.append({
            "patient_id": patient_id,
            "z_center": z_center,
            "z_lo": z_lo,
            "z_hi": z_hi,
            # raw bbox (margin 없음)
            "raw_y0": cand["y0"],
            "raw_x0": cand["x0"],
            "raw_y1": cand["y1"],
            "raw_x1": cand["x1"],
            # context bbox (margin 추가)
            "ctx_y0": cy0,
            "ctx_x0": cx0,
            "ctx_y1": cy1,
            "ctx_x1": cx1,
            # fixed crop 좌표
            "fc_y0": fc_y0,
            "fc_x0": fc_x0,
            "fc_y1": fc_y1,
            "fc_x1": fc_x1,
            # score 정보
            "mean_score": cand["mean_score"],
            "max_score": cand["max_score"],
            # 5가지 basis overlap
            "overlap_context_z_center": overlap["context_z_center"],
            "overlap_fixed_crop_z_center": overlap["fixed_crop_z_center"],
            "overlap_raw_bbox_z_center": overlap["raw_bbox_z_center"],
            "overlap_zstack_max": overlap["zstack_max"],
            "overlap_fixed_crop_zstack_max": overlap["fixed_crop_zstack_max"],
            "zstack_z_of_max": overlap["zstack_z_of_max"],
            "fc_zstack_z_of_max": overlap["fc_zstack_z_of_max"],
        })

    return candidates


# ============================================================
# 출력 함수
# ============================================================
def _overlap_key(basis: str) -> str:
    return f"overlap_{basis}"


def print_lesion_mask_info(lesion_mask: np.ndarray) -> None:
    """lesion mask 기본 정보 출력."""
    total_px = int(np.sum(lesion_mask > 0))
    slices_with_lesion = [
        z for z in range(lesion_mask.shape[0])
        if np.any(lesion_mask[z] > 0)
    ]
    n_slices = len(slices_with_lesion)
    z_min = min(slices_with_lesion) if slices_with_lesion else -1
    z_max = max(slices_with_lesion) if slices_with_lesion else -1

    print("  [Lesion Mask]")
    print(f"    total lesion pixel count : {total_px}")
    print(f"    lesion slice 수          : {n_slices}")
    print(f"    lesion z range           : [{z_min}, {z_max}]")
    if slices_with_lesion:
        if n_slices <= 10:
            print(f"    lesion slices            : {slices_with_lesion}")
        else:
            head = slices_with_lesion[:5]
            tail = slices_with_lesion[-5:]
            print(f"    lesion slices (요약)     : {head} ... {tail} (총 {n_slices}개)")


def print_candidate_info(
    candidates: List[Dict],
    lesion_mask: np.ndarray,
    cond_label: str,
) -> None:
    """candidate z 범위와 병변 slice 겹침 여부 출력."""
    slices_with_lesion: Set[int] = set(
        z for z in range(lesion_mask.shape[0])
        if np.any(lesion_mask[z] > 0)
    )
    cand_z_list = [c["z_center"] for c in candidates]
    cand_z_set = set(cand_z_list)

    z_min_cand = min(cand_z_list) if cand_z_list else -1
    z_max_cand = max(cand_z_list) if cand_z_list else -1
    overlap_slices = sorted(cand_z_set & slices_with_lesion)

    print(f"\n  [Candidate 정보 — {cond_label}]")
    print(f"    candidate 총수                    : {len(candidates)}")
    print(f"    candidate z range                 : [{z_min_cand}, {z_max_cand}]")
    if overlap_slices:
        print(f"    병변 slice와 candidate z 겹침     : 있음 {overlap_slices}")
    else:
        print(f"    병변 slice와 candidate z 겹침     : 없음")

    if slices_with_lesion and cand_z_list:
        min_dist = min(
            min(abs(cz - lz) for lz in slices_with_lesion)
            for cz in cand_z_list
        )
        print(f"    병변 slice와 가장 가까운 candidate z 거리: {min_dist}")
    else:
        print(f"    병변 slice와 가장 가까운 candidate z 거리: N/A")


def print_basis_overlap_summary(
    candidates: List[Dict],
    basis: str,
    label_thresholds: List[float],
) -> None:
    """단일 basis에 대한 overlap 요약 출력."""
    key = _overlap_key(basis)

    if not candidates:
        print(f"    [{basis}]")
        print(f"      candidate 없음")
        return

    overlaps = [c[key] for c in candidates]
    max_overlap = max(overlaps)
    max_idx = overlaps.index(max_overlap)
    mc = candidates[max_idx]

    print(f"    [{basis}]")
    print(f"      max overlap : {max_overlap:.6f}")
    print(f"      max overlap candidate:")
    print(f"        z_center    : {mc['z_center']}")
    print(f"        z_lo / z_hi : {mc['z_lo']} / {mc['z_hi']}")
    print(f"        raw bbox    : y=[{mc['raw_y0']}, {mc['raw_y1']}), x=[{mc['raw_x0']}, {mc['raw_x1']})")
    print(f"        context bbox: y=[{mc['ctx_y0']}, {mc['ctx_y1']}), x=[{mc['ctx_x0']}, {mc['ctx_x1']})")
    print(f"        fixed crop  : y=[{mc['fc_y0']}, {mc['fc_y1']}), x=[{mc['fc_x0']}, {mc['fc_x1']})")

    for thr in label_thresholds:
        if thr == 0.0:
            cnt = sum(1 for o in overlaps if o > 0)
            print(f"      overlap > 0        : {cnt}개")
        else:
            cnt = sum(1 for o in overlaps if o >= thr)
            print(f"      overlap >= {thr:<5.2f}  : {cnt}개")


def print_decision_assistance(
    candidates: List[Dict],
    lesion_mask: np.ndarray,
    cond_label: str,
) -> None:
    """판단 보조 출력 (5가지 패턴 확인)."""
    slices_with_lesion: Set[int] = set(
        z for z in range(lesion_mask.shape[0])
        if np.any(lesion_mask[z] > 0)
    )
    cand_z_set: Set[int] = set(c["z_center"] for c in candidates)

    if not candidates:
        print(f"\n  [판단 보조 — {cond_label}]")
        print(f"    candidate 없음 — 판단 보조 생략")
        return

    # 각 basis별 max overlap
    max_by_basis = {
        b: max(c[_overlap_key(b)] for c in candidates)
        for b in ALL_LABEL_BASES
    }

    z_center_bases = ["context_z_center", "fixed_crop_z_center", "raw_bbox_z_center"]
    zstack_bases = ["zstack_max", "fixed_crop_zstack_max"]

    messages: List[str] = []

    # 패턴 1: 모든 basis에서 max overlap < 0.001
    if all(v < 0.001 for v in max_by_basis.values()):
        messages.append("후보가 병변 위치를 실제로 못 올림")

    # 패턴 2: 어느 basis든 overlap > 0 있고 모든 basis에서 overlap >= 0.10 없음
    any_gt0 = any(c[_overlap_key(b)] > 0 for c in candidates for b in ALL_LABEL_BASES)
    any_ge10 = any(c[_overlap_key(b)] >= 0.10 for c in candidates for b in ALL_LABEL_BASES)
    if any_gt0 and not any_ge10:
        messages.append("label threshold 0.10이 작은 병변에 엄격할 가능성")

    # 패턴 3: z_center 3개 basis에서 >= 0.10 없고 zstack 기반에서 >= 0.10 있음
    any_z_center_ge10 = any(
        c[_overlap_key(b)] >= 0.10 for c in candidates for b in z_center_bases
    )
    any_zstack_ge10 = any(
        c[_overlap_key(b)] >= 0.10 for c in candidates for b in zstack_bases
    )
    if any_zstack_ge10 and not any_z_center_ge10:
        messages.append("z_center 기준이 문제일 가능성")

    # 패턴 4: raw_bbox_z_center >= 0.10 있고 fixed_crop_z_center >= 0.10 없음
    any_raw_ge10 = any(c["overlap_raw_bbox_z_center"] >= 0.10 for c in candidates)
    any_fc_ge10 = any(c["overlap_fixed_crop_z_center"] >= 0.10 for c in candidates)
    if any_raw_ge10 and not any_fc_ge10:
        messages.append("fixed crop 중심 또는 크기 문제 가능성")

    # 패턴 5: 병변 slice에 candidate z_center 없고 병변 slice ±3 이내에 candidate 있음
    in_lesion_slice = bool(cand_z_set & slices_with_lesion)
    if not in_lesion_slice and slices_with_lesion and cand_z_set:
        nearby = any(
            abs(cz - lz) <= 3
            for cz in cand_z_set
            for lz in slices_with_lesion
        )
        if nearby:
            messages.append("slice-level fallback 또는 z 연결 기준 검토 필요")

    print(f"\n  [판단 보조 — {cond_label}]")
    if messages:
        for msg in messages:
            print(f"    → {msg}")
    else:
        print(f"    (해당 패턴 없음)")


# ============================================================
# 환자 단위 diagnostic
# ============================================================
def print_patient_diagnostic(
    row: pd.Series,
    dataset_root: Path,
    args: argparse.Namespace,
) -> None:
    """단일 환자에 대한 전체 diagnostic 출력."""
    patient_id = str(row["patient_id"])
    group = str(row["group"])
    stage_split = str(row["stage_split"])
    safe_id = str(row["safe_id"])

    print(f"\n{'=' * 70}")
    print(f"  Patient: {patient_id}")
    print(f"{'=' * 70}")

    # 섹션 1: 환자 기본 정보
    print(f"  patient_id  : {patient_id}")
    print(f"  group       : {group}")
    print(f"  stage_split : {stage_split}")
    print(f"  safe_id     : {safe_id}")

    # 배열 로드 (1회, read-only)
    try:
        ct_vol, lesion_mask, ct_z, ct_h, ct_w = load_patient_arrays(row, dataset_root)
    except FileNotFoundError as e:
        print(f"\n  [ERROR] 파일 없음: {e}")
        return

    # 섹션 2: lesion mask 정보
    print()
    print_lesion_mask_info(lesion_mask)

    # ── 조건 1: p95 기준 ─────────────────────────────────────
    cond_label_p95 = f"p95 (threshold={args.threshold_p95})"
    print(f"\n  {'─' * 60}")
    print(f"  조건 1: {cond_label_p95}")
    print(f"  {'─' * 60}")

    cands_p95 = build_enriched_candidates(
        row, ct_vol, lesion_mask, args,
        threshold=args.threshold_p95,
    )

    print_candidate_info(cands_p95, lesion_mask, cond_label_p95)

    if cands_p95:
        print(f"\n  [Basis별 Overlap — {cond_label_p95}]")
        for basis in ALL_LABEL_BASES:
            print_basis_overlap_summary(cands_p95, basis, args.label_thresholds)
    else:
        print(f"\n  p95 candidate 없음 (positive patch 0개 또는 merge 결과 없음)")

    print_decision_assistance(cands_p95, lesion_mask, cond_label_p95)

    # ── 조건 2: patient_topk 기준 (replacement_50 맥락) ──────
    cond_label_topk = f"topk_{args.topk} (replacement_50 맥락)"
    print(f"\n  {'─' * 60}")
    print(f"  조건 2: {cond_label_topk}")
    print(f"  {'─' * 60}")

    cands_topk = build_enriched_candidates(
        row, ct_vol, lesion_mask, args,
        topk=args.topk,
    )

    print_candidate_info(cands_topk, lesion_mask, cond_label_topk)

    if cands_topk:
        print(f"\n  [Basis별 Overlap — {cond_label_topk}]")
        for basis in ALL_LABEL_BASES:
            print_basis_overlap_summary(cands_topk, basis, args.label_thresholds)
    else:
        print(f"\n  topk_{args.topk} candidate 없음")

    print_decision_assistance(cands_topk, lesion_mask, cond_label_topk)


# ============================================================
# argparse
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Second-Stage Lesion Refiner v1 Phase 5.7 — "
            "Weak Case Overlap Diagnostic (console-only, no file write)"
        )
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        default=list(WEAK_CASES),
        help=(
            f"처리할 patient_id 목록 (기본값: {' '.join(WEAK_CASES)}). "
            "stage1_dev에 없는 환자가 들어오면 중단. "
            "stage2_holdout이면 중단."
        ),
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=50,
        help="patient top-k 후보 수 (기본 50, replacement_50 맥락)",
    )
    parser.add_argument(
        "--threshold-p95",
        type=float,
        default=P95_THRESHOLD,
        dest="threshold_p95",
        help=f"p95 threshold (기본 {P95_THRESHOLD})",
    )
    parser.add_argument(
        "--label-thresholds",
        nargs="+",
        type=float,
        default=[0.0, 0.01, 0.05, 0.10],
        dest="label_thresholds",
        help=(
            "overlap threshold 목록 (기본 0.0 0.01 0.05 0.10). "
            "0.0은 overlap > 0으로 해석."
        ),
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=DEFAULT_CROP_SIZE,
        dest="crop_size",
        help=f"fixed crop 크기 px (기본 {DEFAULT_CROP_SIZE})",
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
    return parser.parse_args()


# ============================================================
# 메인
# ============================================================
def main() -> None:
    args = parse_args()

    print(
        f"\n{'=' * 70}\n"
        f"  Second-Stage Lesion Refiner v1 — Phase 5.7\n"
        f"  Weak Case Overlap Diagnostic (console-only)\n"
        f"  실행 일시        : {datetime.now().isoformat(timespec='seconds')}\n"
        f"  대상 환자        : {args.patients}\n"
        f"  threshold_p95    : {args.threshold_p95}\n"
        f"  topk             : {args.topk}\n"
        f"  label_thresholds : {args.label_thresholds}\n"
        f"  crop_size        : {args.crop_size}\n"
        f"  context_margin   : {args.context_margin}\n"
        f"  z_range          : +/-{args.z_range}\n"
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
        f"{'=' * 70}\n"
    )

    dataset_root = load_dataset_root()
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root 없음: {dataset_root}")

    split_df = load_split(args.patients)
    n_target = len(split_df)
    print(f"[INFO] 처리 대상 환자: {n_target}명")
    for _, r in split_df.iterrows():
        print(f"  - {r['patient_id']} ({r['group']}, {r['stage_split']})")

    for _, row in split_df.iterrows():
        print_patient_diagnostic(row, dataset_root, args)

    print(f"\n{'=' * 70}")
    print(f"  Weak Case Overlap Diagnostic 완료")
    print(f"  완료 일시: {datetime.now().isoformat(timespec='seconds')}")
    print()
    print(f"  [확인]")
    print(f"  파일 생성/수정          : 0건")
    print(f"  runtime_summary.csv append: 없음")
    print(f"  error.csv append          : 없음")
    print(f"  crop npz / overlay PNG    : 없음")
    print(f"  v2 접촉                   : 없음")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
