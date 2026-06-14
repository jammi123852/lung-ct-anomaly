#!/usr/bin/env python3
"""
build_stage1_dev_candidate_manifest.py

Second-Stage Lesion Refiner v1 — Phase 5.5 Stage1_dev Candidate Manifest 생성

IMPORTANT:
- stage1_dev 154명만 처리 (stage2_holdout 154명 절대 포함 금지)
- 봉인 환자 감지 즉시 중단: LUNG1-089, LUNG1-231, LUNG1-372
- v1 outputs 하위에는 절대 쓰기 금지
- crop(npz) 저장 미구현 — manifest/좌표 검증 전용
- 이 스크립트 결과는 학습 데이터로 바로 사용하지 않음
- ChatGPT 검토 + 사용자 승인 후에만 실행 가능
- overlay 생성 없음 (smoke_test 전용, 이 스크립트에서는 미구현)

TODO (미구현 — 설계 필요):
- lower-threshold fallback (p90, p92.5): stage1_dev 결과 확인 후 별도 설계
- top-k 보조 후보 (환자별/slice별): stage1_dev 결과 확인 후 별도 설계

실행 명령 초안:
  [dry-run — 파일 생성/수정 없음]:
    source ~/ai_env/bin/activate && \\
    python scripts/build_stage1_dev_candidate_manifest.py --dry-run

  [특정 환자만 debug]:
    source ~/ai_env/bin/activate && \\
    python scripts/build_stage1_dev_candidate_manifest.py \\
      --patients MSD_lung_043 LUNG1-148 --dry-run

  [전체 실행 — ChatGPT 검토 + 사용자 승인 후]:
    source ~/ai_env/bin/activate && \\
    python scripts/build_stage1_dev_candidate_manifest.py

  [fixed96_thr001 dry-run — Phase 5.10, 임시 1순위 후보, 최종 확정 아님]:
    source ~/ai_env/bin/activate && \\
    python scripts/build_stage1_dev_candidate_manifest.py \\
      --dry-run \\
      --no-runtime-append \\
      --label-basis fixed_crop_z_center \\
      --label-threshold 0.01 \\
      --crop-size 96 \\
      --output-tag stage1_dev_fixed96_thr001_v1

※ --force 없이는 기존 출력 덮어쓰기 금지
※ 실제 실행은 ChatGPT 검토 + 사용자 승인 대기
"""

import argparse
import csv
import json
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# ============================================================
# 경로 상수
# ============================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
SCORE_DIR = REPO_ROOT / "outputs/position-aware-padim-v1/scores/padim_v1/lesion_by_patient"
SPLIT_CSV = (
    REPO_ROOT
    / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"
)
OUTPUT_BASE = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1"
PATHS_CONFIG = REPO_ROOT / "configs/paths.local.yaml"

# v1 outputs (쓰기 절대 금지)
V1_OUTPUT_DIR = REPO_ROOT / "outputs/position-aware-padim-v1"

SCRIPT_NAME = "build_stage1_dev_candidate_manifest.py"
DATASET_TAG = "stage1_dev_v1"

# ============================================================
# 안전 상수
# ============================================================
P95_THRESHOLD = 14.3774
DEFAULT_CROP_SIZE = 96
DEFAULT_CONTEXT_MARGIN = 32
DEFAULT_Z_RANGE = 1          # z_center ± 1
MAX_BBOX_SIZE = 256          # 이 이상이면 경고 + bbox_too_large=True

# stage2_holdout 봉인 환자 (감지 즉시 중단)
STAGE2_HOLDOUT_BLOCKED: set = {"LUNG1-089", "LUNG1-231", "LUNG1-372"}


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


def check_not_overwrite(path: Path, force: bool) -> None:
    """기존 파일 존재 시 force=False면 중단."""
    if path.exists() and not force:
        raise FileExistsError(
            f"[GUARD] 출력 파일이 이미 존재합니다: {path}\n"
            "--force 없이는 덮어쓰기 금지."
        )


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
            f"(stage_split={stage_split})\n"
            "stage1_dev 환자만 허용."
        )


# ============================================================
# 데이터 로딩
# ============================================================
def load_split(
    limit: Optional[int] = None,
    patients: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    split CSV 로드 → stage1_dev 필터 → 봉인 환자 이중 확인 → limit 적용.

    limit=None이면 전체 stage1_dev 처리.
    """
    df = pd.read_csv(SPLIT_CSV, encoding="utf-8-sig")

    required_cols = ["patient_id", "stage_split", "group"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"split CSV에 필수 컬럼이 없습니다: {missing_cols}\n"
            f"경로: {SPLIT_CSV}"
        )
    if "safe_id" not in df.columns:
        raise ValueError(
            f"split CSV에 safe_id 컬럼이 없습니다 (경로: {SPLIT_CSV})\n"
            "safe_id는 patient manifest 또는 확정된 mapping에서 확인 필요.\n"
            "patient_id == safe_id로 임의 가정하지 않음. 실행 중단."
        )

    df_dev = df[df["stage_split"] == "stage1_dev"].copy()

    # 봉인 환자가 stage1_dev에 잘못 포함됐는지 이중 확인
    for pid in df_dev["patient_id"]:
        if pid in STAGE2_HOLDOUT_BLOCKED:
            raise RuntimeError(
                f"[GUARD] split CSV에 봉인 환자가 stage1_dev로 포함됨: {pid}"
            )

    n_total_dev = len(df_dev)

    if patients:
        # 봉인 환자가 --patients에 포함됐는지 먼저 확인
        blocked_requested = [p for p in patients if p in STAGE2_HOLDOUT_BLOCKED]
        if blocked_requested:
            raise RuntimeError(
                f"[GUARD] 봉인 환자가 --patients에 포함됨: {blocked_requested}\n"
                "봉인 환자는 포함 금지. 실행 중단."
            )

        dev_pids = set(df_dev["patient_id"].tolist())
        requested_set = set(patients)
        found_set = requested_set & dev_pids
        missing_set = requested_set - dev_pids

        if missing_set:
            holdout_pids = set(df[df["stage_split"] == "stage2_holdout"]["patient_id"].tolist())
            in_holdout = sorted(missing_set & holdout_pids)
            not_in_csv = sorted(missing_set - holdout_pids)
            msg = f"지정한 환자 중 stage1_dev에 없는 환자: {sorted(missing_set)}\n"
            msg += f"  요청: {sorted(requested_set)} / stage1_dev 확인됨: {sorted(found_set)}\n"
            if in_holdout:
                msg += f"  → stage2_holdout 소속 (봉인 대상): {in_holdout}\n"
            if not_in_csv:
                msg += f"  → split CSV에 없는 환자: {not_in_csv}\n"
            msg += "조용히 일부만 실행하지 않음. 지정한 모든 환자가 stage1_dev에 있어야 합니다."
            raise ValueError(msg)

        df_dev = df_dev[df_dev["patient_id"].isin(patients)]

    if limit is not None and limit > 0 and len(df_dev) > limit:
        df_dev = df_dev.head(limit)
        print(f"[INFO] --limit {limit} 적용: {len(df_dev)}명만 처리 (전체 stage1_dev={n_total_dev}명)")

    return df_dev.reset_index(drop=True)


def load_score_csv(patient_id: str) -> pd.DataFrame:
    """v1 score CSV 로드 (read-only)."""
    csv_path = SCORE_DIR / f"{patient_id}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"score CSV 없음: {csv_path}")
    return pd.read_csv(csv_path, encoding="utf-8-sig")


# ============================================================
# Candidate 생성
# ============================================================
def merge_patches_in_slice(
    patches: List[Tuple[int, int, int, int, float]],
) -> List[Dict]:
    """
    같은 slice의 positive patch 목록을 bbox touch/overlap 기준(8-neighbor)으로
    묶어 merged bbox 목록을 반환.

    patches: [(y0, x0, y1, x1, padim_score), ...]
    반환: [{'y0', 'x0', 'y1', 'x1', 'n_patches', 'mean_score', 'max_score'}, ...]
    """
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
    """두 bbox가 xy 평면에서 겹치는지 확인 (z 방향 candidate 연결에 사용)."""
    vy = a["y0"] < b["y1"] and b["y0"] < a["y1"]
    vx = a["x0"] < b["x1"] and b["x0"] < a["x1"]
    return vy and vx


def build_candidates_3d(
    slice_bboxes: Dict[int, List[Dict]],
) -> List[Dict]:
    """
    slice별 merged bbox 목록을 연속 z slice 기준으로 묶어 3D candidate 생성.
    연속 z에서 xy bbox가 겹치면 같은 candidate로 연결.
    """
    candidates: List[Dict] = []
    active: List[Dict] = []

    for z in sorted(slice_bboxes.keys()):
        current_bboxes = slice_bboxes[z]
        new_active: List[Dict] = []
        used: set = set()

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
                    "z_start": z,
                    "z_end": z,
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
    margin: int,
    ct_h: int, ct_w: int,
) -> Tuple[int, int, int, int]:
    """context margin 추가 후 CT 경계 내로 clip."""
    return (
        max(0, y0 - margin),
        max(0, x0 - margin),
        min(ct_h, y1 + margin),
        min(ct_w, x1 + margin),
    )


def compute_lesion_overlap(
    lesion_mask: np.ndarray,
    z: int,
    y0: int, x0: int, y1: int, x1: int,
) -> Tuple[int, float]:
    """z_center slice에서 crop bbox 영역의 lesion mask overlap 계산."""
    if z < 0 or z >= lesion_mask.shape[0]:
        return 0, 0.0
    crop = lesion_mask[z, y0:y1, x0:x1]
    area = (y1 - y0) * (x1 - x0)
    if area == 0:
        return 0, 0.0
    lesion_px = int(np.sum(crop > 0))
    return lesion_px, float(lesion_px / area)


def compute_multi_basis_overlap(
    lesion_mask: np.ndarray,
    z_center: int,
    z_lo: int,
    z_hi: int,
    context_y0: int, context_x0: int, context_y1: int, context_x1: int,
    fc_y0: int, fc_x0: int, fc_y1: int, fc_x1: int,
    raw_y0: int, raw_x0: int, raw_y1: int, raw_x1: int,
) -> Dict:
    """4가지 기준으로 lesion overlap을 계산해 진단 컬럼을 반환한다."""
    ctx_px, ctx_ratio = compute_lesion_overlap(
        lesion_mask, z_center, context_y0, context_x0, context_y1, context_x1
    )
    fc_px, fc_ratio = compute_lesion_overlap(
        lesion_mask, z_center, fc_y0, fc_x0, fc_y1, fc_x1
    )
    raw_px, raw_ratio = compute_lesion_overlap(
        lesion_mask, z_center, raw_y0, raw_x0, raw_y1, raw_x1
    )

    zstack_max_ratio = 0.0
    zstack_max_px = 0
    z_of_max = z_center
    for z in range(z_lo, z_hi + 1):
        zpx, zratio = compute_lesion_overlap(
            lesion_mask, z, context_y0, context_x0, context_y1, context_x1
        )
        if zratio > zstack_max_ratio:
            zstack_max_ratio = zratio
            zstack_max_px = zpx
            z_of_max = z

    return {
        "lesion_pixel_count_context": ctx_px,
        "lesion_overlap_ratio_context": round(ctx_ratio, 6),
        "lesion_pixel_count_fixed_crop": fc_px,
        "lesion_overlap_ratio_fixed_crop": round(fc_ratio, 6),
        "lesion_pixel_count_raw_bbox": raw_px,
        "lesion_overlap_ratio_raw_bbox": round(raw_ratio, 6),
        "lesion_pixel_count_zstack_max": zstack_max_px,
        "lesion_overlap_ratio_zstack_max": round(zstack_max_ratio, 6),
        "z_of_max_overlap": z_of_max,
    }


def assign_labels(
    lesion_overlap_ratio: float,
    label_threshold: float = 0.10,
) -> Tuple[bool, str, str]:
    """
    candidate에 label 부여.
    반환: (coverage_label, rd4ad_label, binary_label)

    label 규칙 (label_threshold 파라미터 기준):
    - overlap == 0          → normal_like / hard_negative
    - 0 < overlap < threshold → ambiguous / ambiguous
    - overlap >= threshold  → lesion_candidate / positive

    주의:
    - RD4AD 학습에서 lesion_candidate는 학습 target이 아님 — 평가 ground truth 전용.
    - ambiguous는 RD4AD 학습 및 binary classifier baseline 기본 제외.
    - coverage_label은 0.01 고정 기준 (진단용 별도 컬럼).
    """
    coverage_label = lesion_overlap_ratio >= 0.01

    if lesion_overlap_ratio == 0.0:
        rd4ad_label = "normal_like"
        binary_label = "hard_negative"
    elif lesion_overlap_ratio >= label_threshold:
        rd4ad_label = "lesion_candidate"
        binary_label = "positive"
    else:
        rd4ad_label = "ambiguous"
        binary_label = "ambiguous"

    return coverage_label, rd4ad_label, binary_label


# ============================================================
# 런타임 / 에러 기록
# ============================================================
def record_runtime(
    reports_dir: Path,
    row: Dict,
    dry_run: bool = False,
    no_runtime_append: bool = False,
) -> None:
    if dry_run or no_runtime_append:
        return
    rt_path = reports_dir / "runtime_summary.csv"
    fieldnames = ["timestamp", "script", "metric", "value"]
    rt_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not rt_path.exists()
    with open(rt_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def record_error(
    reports_dir: Path,
    patient_id: str,
    error: str,
    dry_run: bool = False,
) -> None:
    if dry_run:
        return
    err_path = reports_dir / "error.csv"
    fieldnames = ["timestamp", "script", "patient_id", "error"]
    err_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not err_path.exists()
    with open(err_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "script": SCRIPT_NAME,
            "patient_id": patient_id,
            "error": error,
        })


# ============================================================
# 환자 단위 처리
# ============================================================
def process_patient(
    row: pd.Series,
    dataset_root: Path,
    args: argparse.Namespace,
    candidate_idx_start: int,
) -> Tuple[List[Dict], int]:
    """
    단일 환자 candidate 생성.
    반환: (candidates, n_candidates)
    """
    patient_id = str(row["patient_id"])
    safe_id = str(row["safe_id"])
    stage_split = str(row["stage_split"])
    group = str(row["group"])

    # 안전 가드 (stage1_dev + 봉인 환자 이중 확인)
    check_stage1_dev(patient_id, stage_split)

    # v1 score CSV 로드 (read-only)
    score_df = load_score_csv(patient_id)

    # positive patch 추출
    pos_df = score_df[score_df["padim_score"] >= P95_THRESHOLD].copy()
    if pos_df.empty:
        print(f"  [WARN] {patient_id}: p95 이상 positive patch 없음")
        return [], 0

    # CT / lesion mask 로드 (read-only, mmap)
    vol_dir = dataset_root / "volumes_npy" / safe_id
    ct_path = vol_dir / "ct_hu.npy"
    mask_path = vol_dir / "lesion_mask_model_roi.npy"

    if not ct_path.exists():
        raise FileNotFoundError(f"CT volume 없음: {ct_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"lesion mask 없음: {mask_path}")

    ct_vol = np.load(ct_path, mmap_mode="r")        # [Z, H, W]
    lesion_mask = np.load(mask_path, mmap_mode="r")  # [Z, H, W]
    ct_z, ct_h, ct_w = ct_vol.shape

    # slice별 positive patch → merged bbox
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
        return [], 0

    # 3D candidate 생성
    cands_raw = build_candidates_3d(slice_bboxes)

    candidates: List[Dict] = []
    for i, cand in enumerate(cands_raw):
        z_center = int(cand["z_center"])

        # context margin + clip
        cy0, cx0, cy1, cx1 = apply_context_margin(
            cand["y0"], cand["x0"], cand["y1"], cand["x1"],
            args.context_margin, ct_h, ct_w,
        )

        # bbox 크기 경고
        bbox_h = cy1 - cy0
        bbox_w = cx1 - cx0
        bbox_too_large = bbox_h > MAX_BBOX_SIZE or bbox_w > MAX_BBOX_SIZE
        if bbox_too_large:
            print(
                f"  [WARN] 큰 bbox: {patient_id} z={z_center} "
                f"h={bbox_h} w={bbox_w} (>{MAX_BBOX_SIZE}px) — bbox_too_large=True 기록"
            )

        # z 범위 확인
        z_lo = max(0, z_center - args.z_range)
        z_hi = min(ct_z - 1, z_center + args.z_range)
        if z_lo == z_center - args.z_range and z_hi == z_center + args.z_range:
            z_boundary_ok = True
        else:
            z_boundary_ok = False
            print(
                f"  [WARN] z 경계 클리핑: {patient_id} z_center={z_center} "
                f"→ [{z_lo}, {z_hi}] (원래 ±{args.z_range})"
            )

        # center-crop 좌표 (RD4AD 고정 입력용: crop_size × crop_size)
        bbox_center_y = (cand["y0"] + cand["y1"]) // 2
        bbox_center_x = (cand["x0"] + cand["x1"]) // 2
        half_crop = args.crop_size // 2
        fc_y0 = max(0, bbox_center_y - half_crop)
        fc_x0 = max(0, bbox_center_x - half_crop)
        fc_y1 = min(ct_h, fc_y0 + args.crop_size)
        fc_x1 = min(ct_w, fc_x0 + args.crop_size)
        fc_y0 = max(0, fc_y1 - args.crop_size)
        fc_x0 = max(0, fc_x1 - args.crop_size)

        # lesion overlap (z_center slice, context 기준 — 하위 호환 보존용)
        lesion_px_ctx, overlap_ratio_ctx = compute_lesion_overlap(
            lesion_mask, z_center, cy0, cx0, cy1, cx1
        )

        # multi-basis overlap 진단 컬럼
        multi_overlap = compute_multi_basis_overlap(
            lesion_mask,
            z_center=z_center,
            z_lo=z_lo,
            z_hi=z_hi,
            context_y0=cy0, context_x0=cx0, context_y1=cy1, context_x1=cx1,
            fc_y0=fc_y0, fc_x0=fc_x0, fc_y1=fc_y1, fc_x1=fc_x1,
            raw_y0=cand["y0"], raw_x0=cand["x0"], raw_y1=cand["y1"], raw_x1=cand["x1"],
        )

        # label_basis에 따라 selected overlap 선택
        _basis = args.label_basis
        if _basis == "context_z_center":
            selected_lesion_px = lesion_px_ctx
            selected_overlap_ratio = overlap_ratio_ctx
        elif _basis == "fixed_crop_z_center":
            selected_lesion_px = multi_overlap["lesion_pixel_count_fixed_crop"]
            selected_overlap_ratio = multi_overlap["lesion_overlap_ratio_fixed_crop"]
        elif _basis == "raw_bbox_z_center":
            selected_lesion_px = multi_overlap["lesion_pixel_count_raw_bbox"]
            selected_overlap_ratio = multi_overlap["lesion_overlap_ratio_raw_bbox"]
        elif _basis == "zstack_max":
            selected_lesion_px = multi_overlap["lesion_pixel_count_zstack_max"]
            selected_overlap_ratio = multi_overlap["lesion_overlap_ratio_zstack_max"]
        else:
            selected_lesion_px = lesion_px_ctx
            selected_overlap_ratio = overlap_ratio_ctx

        # label 부여 (selected_overlap_ratio 기준, label_threshold 파라미터 사용)
        coverage_label, rd4ad_label, binary_label = assign_labels(
            selected_overlap_ratio, args.label_threshold
        )

        candidate_id = f"{patient_id}_c{candidate_idx_start + i:05d}"

        candidates.append({
            "candidate_id": candidate_id,
            "patient_id": patient_id,
            "safe_id": safe_id,
            "group": group,
            "stage_split": stage_split,
            "dataset_tag": args.output_tag,
            # z 정보
            "z_center": z_center,
            "z_start": cand["z_start"],
            "z_end": cand["z_end"],
            "z_lo": z_lo,
            "z_hi": z_hi,
            "z_boundary_ok": z_boundary_ok,
            # 원래 bbox (margin 전)
            "y0": cand["y0"],
            "x0": cand["x0"],
            "y1": cand["y1"],
            "x1": cand["x1"],
            # crop bbox (margin 후, clipped)
            "y0_crop": cy0,
            "x0_crop": cx0,
            "y1_crop": cy1,
            "x1_crop": cx1,
            # center-crop 고정 좌표 (RD4AD 입력용, crop_size × crop_size)
            "y0_fixed_crop": fc_y0,
            "x0_fixed_crop": fc_x0,
            "y1_fixed_crop": fc_y1,
            "x1_fixed_crop": fc_x1,
            "fixed_crop_size": args.crop_size,
            # score 정보
            "n_positive_patch": cand["n_patches"],
            "mean_padim_score": round(cand["mean_score"], 6),
            "max_padim_score": round(cand["max_score"], 6),
            # label_basis, label_threshold 및 selected overlap
            "label_basis": args.label_basis,
            "label_threshold": args.label_threshold,
            "lesion_pixel_count": selected_lesion_px,
            "lesion_overlap_ratio": round(selected_overlap_ratio, 6),
            "selected_lesion_pixel_count": selected_lesion_px,
            "selected_lesion_overlap_ratio": round(selected_overlap_ratio, 6),
            # labels
            "coverage_label": coverage_label,
            "rd4ad_label": rd4ad_label,
            "binary_label": binary_label,
            # 경고 플래그
            "bbox_too_large": bbox_too_large,
            # multi-basis overlap 진단 컬럼
            "lesion_pixel_count_context": multi_overlap["lesion_pixel_count_context"],
            "lesion_overlap_ratio_context": multi_overlap["lesion_overlap_ratio_context"],
            "lesion_pixel_count_fixed_crop": multi_overlap["lesion_pixel_count_fixed_crop"],
            "lesion_overlap_ratio_fixed_crop": multi_overlap["lesion_overlap_ratio_fixed_crop"],
            "lesion_pixel_count_raw_bbox": multi_overlap["lesion_pixel_count_raw_bbox"],
            "lesion_overlap_ratio_raw_bbox": multi_overlap["lesion_overlap_ratio_raw_bbox"],
            "lesion_pixel_count_zstack_max": multi_overlap["lesion_pixel_count_zstack_max"],
            "lesion_overlap_ratio_zstack_max": multi_overlap["lesion_overlap_ratio_zstack_max"],
            "z_of_max_overlap": multi_overlap["z_of_max_overlap"],
        })

    return candidates, len(candidates)


# ============================================================
# argparse
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Second-Stage Lesion Refiner v1 Phase 5.5 — "
            "stage1_dev 154명 전체 candidate manifest 생성"
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "처리할 환자 수 상한 (기본 None=전체). "
            "smoke/debug 목적으로만 사용할 것."
        ),
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        default=None,
        help=(
            "처리할 patient_id 목록 "
            "(예: --patients MSD_lung_043 LUNG1-148). "
            "지정 시 stage1_dev에서 해당 환자만 처리. "
            "지정 환자가 stage1_dev에 없으면 중단."
        ),
    )
    parser.add_argument(
        "--context-margin",
        type=int,
        default=DEFAULT_CONTEXT_MARGIN,
        help=f"candidate crop context margin px (기본 {DEFAULT_CONTEXT_MARGIN})",
    )
    parser.add_argument(
        "--z-range",
        type=int,
        default=DEFAULT_Z_RANGE,
        help=f"2.5D z 범위: z_center ± z_range (기본 {DEFAULT_Z_RANGE})",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=DEFAULT_CROP_SIZE,
        help=f"crop 목표 크기 px (기본 {DEFAULT_CROP_SIZE})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="기존 출력 파일 덮어쓰기 허용 (기본 금지)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "실제 파일 저장 없이 검증만 수행. "
            "stage1_dev 전체를 읽고 예상 candidate 수/label 분포만 콘솔 출력."
        ),
    )
    parser.add_argument(
        "--label-basis",
        dest="label_basis",
        type=str,
        choices=["context_z_center", "fixed_crop_z_center", "raw_bbox_z_center", "zstack_max"],
        default="fixed_crop_z_center",
        help=(
            "rd4ad_label / binary_label 계산 기준 (기본: fixed_crop_z_center). "
            "context/raw_bbox/zstack_max는 진단용 컬럼으로 항상 기록됨."
        ),
    )
    parser.add_argument(
        "--label-threshold",
        dest="label_threshold",
        type=float,
        default=0.10,
        help=(
            "rd4ad_label / binary_label 판정 threshold (기본 0.10 — 기존 동작 유지). "
            "fixed96_thr001 실행 시 0.01로 지정. "
            "overlap>=threshold → lesion_candidate, "
            "0<overlap<threshold → ambiguous, "
            "overlap==0 → normal_like."
        ),
    )
    parser.add_argument(
        "--output-tag",
        dest="output_tag",
        type=str,
        default="stage1_dev_v1",
        help=(
            "출력 경로 및 파일명에 사용할 tag (기본: stage1_dev_v1 — 기존 동작 유지). "
            "fixed96_thr001 실행 시: stage1_dev_fixed96_thr001_v1. "
            "기존 stage1_dev_v1 경로와 절대 섞이지 않도록 다른 tag 사용 필요."
        ),
    )
    parser.add_argument(
        "--no-runtime-append",
        action="store_true",
        default=False,
        help="runtime_summary.csv append를 건너뜀 (기본 append 허용)",
    )
    return parser.parse_args()


# ============================================================
# 메인
# ============================================================
def main() -> None:
    args = parse_args()

    print(
        f"\n{'=' * 60}\n"
        f"  Second-Stage Lesion Refiner v1 — Phase 5.5 Stage1_dev\n"
        f"  output_tag      : {args.output_tag}\n"
        f"  limit           : {args.limit if args.limit is not None else '전체 (None)'}\n"
        f"  z_range         : ±{args.z_range}\n"
        f"  margin          : {args.context_margin}px\n"
        f"  crop_size       : {args.crop_size}×{args.crop_size}\n"
        f"  label_basis     : {args.label_basis}\n"
        f"  label_threshold : {args.label_threshold}\n"
        f"  dry_run         : {args.dry_run}\n"
        f"  p95_threshold   : {P95_THRESHOLD}\n"
        f"\n"
        f"  [구현 상태]\n"
        f"  manifest 생성: 구현됨\n"
        f"  overlay 저장: 미구현 (stage1_dev 전용, smoke_test와 분리)\n"
        f"  crop(npz) 저장: 미구현 — manifest/좌표 검증 전용\n"
        f"\n"
        f"  [TODO — 미구현]\n"
        f"  lower-threshold fallback (p90/p92.5): 설계 필요\n"
        f"  top-k 보조 후보: 설계 필요\n"
        f"\n"
        f"  [GUARD]\n"
        f"  stage2_holdout 포함 금지\n"
        f"  봉인 환자 포함 금지: {sorted(STAGE2_HOLDOUT_BLOCKED)}\n"
        f"  v1 outputs 쓰기 금지\n"
        f"{'=' * 60}\n"
    )

    # 출력 경로 설계 (output_tag 기반으로 분리 — 기본: stage1_dev_v1)
    stage1_dev_cand_dir = OUTPUT_BASE / "candidates" / args.output_tag
    reports_dir = OUTPUT_BASE / "reports"

    manifest_path = stage1_dev_cand_dir / f"candidate_manifest_{args.output_tag}.csv"
    summary_path = reports_dir / f"{args.output_tag}_summary.json"

    # v1 outputs 쓰기 가드
    check_not_v1_output(manifest_path)
    check_not_v1_output(stage1_dev_cand_dir)

    # overwrite 가드
    if not args.dry_run:
        check_not_overwrite(manifest_path, args.force)
        check_not_overwrite(summary_path, args.force)

    # dataset root 로드
    dataset_root = load_dataset_root()
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root 없음: {dataset_root}")

    # split 로드 + stage1_dev 필터
    split_df = load_split(args.limit, args.patients)
    n_target = len(split_df)

    if args.patients is None and args.limit is None:
        if n_target != 154:
            raise RuntimeError(
                f"[GUARD] 전체 stage1_dev 실행은 154명이어야 합니다. 현재 n_target={n_target}"
            )

    print(f"[INFO] 처리 대상 환자 {n_target}명:")
    for _, r in split_df.iterrows():
        wf = " [weak_case]" if r.get("weak_case_flag", 0) else ""
        print(f"  - {r['patient_id']} ({r['group']}, {r['stage_split']}){wf}")

    # 출력 디렉터리 생성 (dry_run이 아닐 때만)
    if not args.dry_run:
        for d in [stage1_dev_cand_dir, reports_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # 환자별 처리
    all_candidates: List[Dict] = []
    patient_candidate_counts: Dict[str, int] = {}
    patient_rd4ad_label_counts: Dict[str, Dict[str, int]] = {}
    patient_binary_label_counts: Dict[str, Dict[str, int]] = {}
    patient_large_bbox_counts: Dict[str, int] = {}
    patient_ambiguous_counts: Dict[str, int] = {}
    patient_lesion_candidate_counts: Dict[str, int] = {}
    patient_normal_like_counts: Dict[str, int] = {}
    failed_patients: List[Dict] = []
    n_scored = 0
    n_failed = 0
    n_large_bbox = 0
    candidate_idx = 0
    start_time = datetime.now()

    for _, row in split_df.iterrows():
        patient_id = str(row["patient_id"])
        print(f"\n[{n_scored + n_failed + 1}/{n_target}] {patient_id} ...")
        try:
            cands, n_cand = process_patient(
                row, dataset_root, args, candidate_idx
            )
            all_candidates.extend(cands)
            patient_candidate_counts[patient_id] = n_cand
            candidate_idx += n_cand
            n_scored += 1
            n_large = sum(1 for c in cands if c.get("bbox_too_large", False))
            n_large_bbox += n_large
            # 환자별 label count 집계
            _p_rd4ad: Dict[str, int] = defaultdict(int)
            _p_binary: Dict[str, int] = defaultdict(int)
            for _c in cands:
                _p_rd4ad[_c["rd4ad_label"]] += 1
                _p_binary[_c["binary_label"]] += 1
            patient_rd4ad_label_counts[patient_id] = dict(_p_rd4ad)
            patient_binary_label_counts[patient_id] = dict(_p_binary)
            patient_large_bbox_counts[patient_id] = n_large
            patient_ambiguous_counts[patient_id] = int(_p_rd4ad["ambiguous"])
            patient_lesion_candidate_counts[patient_id] = int(_p_rd4ad["lesion_candidate"])
            patient_normal_like_counts[patient_id] = int(_p_rd4ad["normal_like"])
            print(f"  → candidate {n_cand}개 생성 (large_bbox={n_large})")
            record_runtime(
                reports_dir,
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "script": SCRIPT_NAME,
                    "metric": f"n_candidates_{patient_id}",
                    "value": n_cand,
                },
                dry_run=args.dry_run,
                no_runtime_append=args.no_runtime_append,
            )
        except Exception as exc:
            print(f"  [ERROR] {patient_id}: {exc}")
            record_error(reports_dir, patient_id, str(exc), dry_run=args.dry_run)
            patient_candidate_counts[patient_id] = 0
            patient_rd4ad_label_counts[patient_id] = {}
            patient_binary_label_counts[patient_id] = {}
            patient_large_bbox_counts[patient_id] = 0
            patient_ambiguous_counts[patient_id] = 0
            patient_lesion_candidate_counts[patient_id] = 0
            patient_normal_like_counts[patient_id] = 0
            failed_patients.append({"patient_id": patient_id, "error": str(exc)})
            n_failed += 1
            continue

    elapsed = (datetime.now() - start_time).total_seconds()

    # 결과 저장
    # partial manifest 저장 금지 guard
    if n_failed > 0:
        failed_list_str = "; ".join(
            f"{fp['patient_id']}({fp['error'][:80]})" for fp in failed_patients
        )
        msg = (
            f"[GUARD] 실패 환자가 있어 manifest/summary를 저장하지 않습니다.\n"
            f"  실패 환자 수: {n_failed}\n"
            f"  실패 환자 목록: {failed_list_str}\n"
            f"  partial manifest는 허용하지 않습니다."
        )
        if args.dry_run:
            print(f"\n[DRY-RUN] 파일 생성 없음, 실패 환자 있음.\n{msg}")
            sys.exit(1)
        else:
            raise RuntimeError(msg)

    if not args.dry_run and all_candidates:
        cand_df = pd.DataFrame(all_candidates)
        cand_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")
        print(f"\n[INFO] manifest 저장: {manifest_path} ({len(all_candidates)}개)")
    elif args.dry_run:
        print(
            f"\n[DRY-RUN] 파일 생성/수정 없음. "
            f"candidate {len(all_candidates)}개 생성 예상 (실제 저장 안 됨)."
        )

    # label 요약
    binary_summary: Dict[str, int] = defaultdict(int)
    rd4ad_summary: Dict[str, int] = defaultdict(int)
    for c in all_candidates:
        binary_summary[c["binary_label"]] += 1
        rd4ad_summary[c["rd4ad_label"]] += 1

    # 환자별 candidate 수 통계
    counts = list(patient_candidate_counts.values())
    if counts:
        counts_arr = np.array(counts)
        patient_count_min = int(counts_arr.min())
        patient_count_median = float(np.median(counts_arr))
        patient_count_max = int(counts_arr.max())
    else:
        patient_count_min = 0
        patient_count_median = 0.0
        patient_count_max = 0

    # weak_case 후보 유지 여부 (split CSV에 weak_case_flag가 있으면 확인)
    weak_case_info: Dict[str, int] = {}
    if "weak_case_flag" in split_df.columns:
        for _, r in split_df.iterrows():
            pid = str(r["patient_id"])
            if r.get("weak_case_flag", 0):
                weak_case_info[pid] = patient_candidate_counts.get(pid, 0)

    n_total = len(all_candidates)
    n_ambiguous = rd4ad_summary.get("ambiguous", 0)
    n_lesion_candidate = rd4ad_summary.get("lesion_candidate", 0)
    n_normal_like = rd4ad_summary.get("normal_like", 0)

    summary = {
        "script": SCRIPT_NAME,
        "dataset_tag": args.output_tag,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_patients_target": n_target,
        "n_patients_scored": n_scored,
        "n_patients_failed": n_failed,
        "n_candidates_total": n_total,
        "n_candidates_large_bbox": n_large_bbox,
        "large_bbox_ratio": round(n_large_bbox / n_total, 4) if n_total > 0 else 0.0,
        "n_ambiguous": n_ambiguous,
        "ambiguous_ratio": round(n_ambiguous / n_total, 4) if n_total > 0 else 0.0,
        "n_lesion_candidate": n_lesion_candidate,
        "n_normal_like": n_normal_like,
        "rd4ad_label_summary": dict(rd4ad_summary),
        "binary_label_summary": dict(binary_summary),
        "patient_candidate_counts": patient_candidate_counts,
        "patient_rd4ad_label_counts": patient_rd4ad_label_counts,
        "patient_binary_label_counts": patient_binary_label_counts,
        "patient_large_bbox_counts": patient_large_bbox_counts,
        "patient_count_min": patient_count_min,
        "patient_count_median": patient_count_median,
        "patient_count_max": patient_count_max,
        "weak_case_candidate_counts": weak_case_info,
        "elapsed_seconds": round(elapsed, 2),
        "dry_run": args.dry_run,
        "label_basis": args.label_basis,
        "label_threshold": args.label_threshold,
        "output_tag": args.output_tag,
        "candidate_seed_rule": "p95_score_threshold",
        "p95_threshold": P95_THRESHOLD,
        "crop_size": args.crop_size,
        "z_range": args.z_range,
        "context_margin": args.context_margin,
        "args": {
            "limit": args.limit,
            "patients": args.patients,
            "context_margin": args.context_margin,
            "z_range": args.z_range,
            "crop_size": args.crop_size,
            "label_basis": args.label_basis,
            "label_threshold": args.label_threshold,
            "output_tag": args.output_tag,
            "no_runtime_append": args.no_runtime_append,
        },
        "note": (
            "stage1_dev 전체 candidate manifest. "
            "stage2_holdout 미포함 확인됨. "
            "lower-threshold/top-k fallback 미구현 — 설계 필요. "
            "fixed96_thr001은 임시 1순위 후보이며 최종 성능 우위 확정 아님."
        ),
    }

    if not args.dry_run:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[INFO] summary 저장: {summary_path}")

    record_runtime(
        reports_dir,
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "script": SCRIPT_NAME,
            "metric": "stage1_dev_manifest_elapsed_seconds",
            "value": round(elapsed, 2),
        },
        dry_run=args.dry_run,
        no_runtime_append=args.no_runtime_append,
    )

    dry_run_note = "  [DRY-RUN] 파일 생성/수정 없음\n" if args.dry_run else ""
    print(
        f"\n{'=' * 60}\n"
        f"  완료: scored={n_scored}, failed={n_failed}\n"
        f"  output_tag    : {args.output_tag}\n"
        f"  label_basis   : {args.label_basis}\n"
        f"  label_threshold: {args.label_threshold}\n"
        f"  candidates={n_total}, large_bbox={n_large_bbox}\n"
        f"  binary_label  : {dict(binary_summary)}\n"
        f"  rd4ad_label   : {dict(rd4ad_summary)}\n"
        f"  환자별 candidate: min={patient_count_min}, "
        f"median={patient_count_median:.1f}, max={patient_count_max}\n"
        f"  elapsed       : {elapsed:.1f}s\n"
        f"\n"
        f"{dry_run_note}"
        f"  [9B 보고 필수] 학습 단계 전 아래 항목 확인:\n"
        f"  - 전체 candidate 수, 환자별 분포\n"
        f"  - large_bbox 수/비율 ({n_large_bbox}/{n_total})\n"
        f"  - ambiguous 수/비율 ({n_ambiguous}/{n_total})\n"
        f"  - lesion_candidate 수 ({n_lesion_candidate})\n"
        f"  - weak_case 후보 유지 여부: {list(weak_case_info.keys())}\n"
        f"  - hard_negative sampling 필요 여부\n"
        f"  - fallback 후보 설계 필요 여부\n"
        f"{'=' * 60}\n"
    )

    # dry-run 진단 요약 (예상 분포 콘솔 출력)
    if args.dry_run and all_candidates:
        n01_fc = sum(1 for c in all_candidates if c["lesion_overlap_ratio_fixed_crop"] >= 0.01)
        n10_fc = sum(1 for c in all_candidates if c["lesion_overlap_ratio_fixed_crop"] >= 0.10)
        nthr_fc = sum(1 for c in all_candidates if c["lesion_overlap_ratio_fixed_crop"] >= args.label_threshold)
        n01_ctx = sum(1 for c in all_candidates if c["lesion_overlap_ratio_context"] >= 0.01)
        n10_ctx = sum(1 for c in all_candidates if c["lesion_overlap_ratio_context"] >= 0.10)

        print(
            f"\n[DRY-RUN 진단 요약]\n"
            f"  예상 candidate 수: {n_total}\n"
            f"  output_tag: {args.output_tag}\n"
            f"  label 분포 (label_basis={args.label_basis}, label_threshold={args.label_threshold}):\n"
            f"    rd4ad_label  : {dict(rd4ad_summary)}\n"
            f"    binary_label : {dict(binary_summary)}\n"
            f"  기준별 후보 수 (>=0.01 / >=label_threshold({args.label_threshold}) / >=0.10):\n"
            f"    fixed_crop : >=0.01={n01_fc}, >={args.label_threshold}={nthr_fc}, >=0.10={n10_fc}\n"
            f"    context    : >=0.01={n01_ctx}, >=0.10={n10_ctx}\n"
            f"  large_bbox 수: {n_large_bbox} "
            f"({100 * n_large_bbox / n_total:.1f}%)\n"
            f"  ambiguous 수: {n_ambiguous} "
            f"({100 * n_ambiguous / n_total:.1f}%)\n"
            f"  lesion_candidate 수: {n_lesion_candidate}\n"
            f"  환자별 candidate 수: "
            f"min={patient_count_min}, median={patient_count_median:.1f}, max={patient_count_max}\n"
        )

        # [1] lesion_candidate=0 환자 목록
        _zero_lc = sorted([
            pid for pid, cnt in patient_lesion_candidate_counts.items() if cnt == 0
        ])
        print(
            f"  [1] lesion_candidate=0 환자:\n"
            f"    {len(_zero_lc)}명 / {n_target}명\n"
            f"    {_zero_lc}\n"
        )

        # [2] ambiguous>0 and lesion_candidate=0 환자
        _amb_only = sorted([
            pid for pid in _zero_lc
            if patient_ambiguous_counts.get(pid, 0) > 0
        ])
        print(
            f"  [2] ambiguous>0 and lesion_candidate=0 환자:\n"
            f"    {len(_amb_only)}명\n"
            f"    {_amb_only}\n"
        )

        # [3] weak_case별 label 분포
        _WEAK_CASES = ["MSD_lung_043", "MSD_lung_079", "MSD_lung_096"]
        print(f"  [3] weak_case별 label 분포:")
        for _wc in _WEAK_CASES:
            _wc_total = patient_candidate_counts.get(_wc, "N/A(미처리)")
            _wc_nl = patient_normal_like_counts.get(_wc, "N/A")
            _wc_amb = patient_ambiguous_counts.get(_wc, "N/A")
            _wc_lc = patient_lesion_candidate_counts.get(_wc, "N/A")
            _wc_lb = patient_large_bbox_counts.get(_wc, "N/A")
            print(
                f"    {_wc}: total={_wc_total}, normal_like={_wc_nl}, "
                f"ambiguous={_wc_amb}, lesion_candidate={_wc_lc}, large_bbox={_wc_lb}"
            )
        print()

        # [4] large_bbox 상위 환자 Top 10
        _sorted_lb = sorted(
            patient_large_bbox_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        print(f"  [4] large_bbox 상위 환자 Top 10:")
        print(f"    {'patient_id':<22} {'total_cand':>10} {'large_bbox':>10} {'large_bbox%':>12}")
        for _pid, _lb in _sorted_lb:
            _tot = patient_candidate_counts.get(_pid, 0)
            _ratio = _lb / _tot if _tot > 0 else 0.0
            print(f"    {_pid:<22} {_tot:>10} {_lb:>10} {_ratio:>11.1%}")
        print()

        # [5] candidate 수 상위 환자 Top 10
        _sorted_tot = sorted(
            patient_candidate_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        print(f"  [5] candidate 수 상위 환자 Top 10:")
        print(f"    {'patient_id':<22} {'total':>7} {'normal_like':>12} {'ambiguous':>10} {'lesion_cand':>12} {'large_bbox':>10}")
        for _pid, _tot in _sorted_tot:
            _nl = patient_normal_like_counts.get(_pid, 0)
            _amb = patient_ambiguous_counts.get(_pid, 0)
            _lc = patient_lesion_candidate_counts.get(_pid, 0)
            _lb = patient_large_bbox_counts.get(_pid, 0)
            print(f"    {_pid:<22} {_tot:>7} {_nl:>12} {_amb:>10} {_lc:>12} {_lb:>10}")
        print()

        # [6] lesion_candidate 상위 환자 Top 10
        _sorted_lc = sorted(
            patient_lesion_candidate_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        print(f"  [6] lesion_candidate 상위 환자 Top 10:")
        print(f"    {'patient_id':<22} {'lesion_cand':>12} {'total_cand':>10}")
        for _pid, _lc in _sorted_lc:
            _tot = patient_candidate_counts.get(_pid, 0)
            print(f"    {_pid:<22} {_lc:>12} {_tot:>10}")
        print()

        # [7] hard_negative sampling 판단용 요약
        _hn_vals = np.array([
            d.get("hard_negative", 0) for d in patient_binary_label_counts.values()
        ], dtype=float)
        _hn_total = int(binary_summary.get("hard_negative", 0))
        _pos_total = int(rd4ad_summary.get("lesion_candidate", 0))
        _hn_pos_ratio = _hn_total / _pos_total if _pos_total > 0 else float("inf")
        _hn_max = int(_hn_vals.max()) if len(_hn_vals) > 0 else 0
        _hn_median = float(np.median(_hn_vals)) if len(_hn_vals) > 0 else 0.0
        print(
            f"  [7] hard_negative sampling 판단용 요약:\n"
            f"    total hard_negative        : {_hn_total}\n"
            f"    total positive             : {_pos_total}\n"
            f"    hard_negative:positive 비율: {_hn_pos_ratio:.1f}:1\n"
            f"    환자별 hard_negative max   : {_hn_max}\n"
            f"    환자별 hard_negative median: {_hn_median:.1f}\n"
        )
        _sorted_hn = sorted(
            [(pid, d.get("hard_negative", 0)) for pid, d in patient_binary_label_counts.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        print(f"    hard_negative 상위 환자 Top 10 (cap 판단용):")
        print(f"    {'patient_id':<22} {'hard_negative':>14} {'total_cand':>10}")
        for _pid, _hn in _sorted_hn:
            _tot = patient_candidate_counts.get(_pid, 0)
            print(f"    {_pid:<22} {_hn:>14} {_tot:>10}")
        print()


if __name__ == "__main__":
    main()
