#!/usr/bin/env python3
"""
build_second_stage_smoke_candidates.py

Second-Stage Lesion Refiner v1 — Phase 4 Smoke Test
candidate 생성 로직 검증 스크립트 (3~5명 stage1_dev 환자 한정)

IMPORTANT:
- 이 스크립트 실행 결과는 학습 데이터가 아님 (smoke test 전용)
- 성능 결론 도출 금지
- stage1_dev 환자만 대상, stage2_holdout 봉인 유지
- ChatGPT 검토 + 사용자 승인 후에만 실행 가능

현재 구현 상태:
- manifest/좌표 검증 전용 (manifest-only smoke, npz 저장 없음)
- overlay 저장: 구현됨 (cv2, smoke_test 전용 경로)
- crop(npz) 저장: 미구현 — 별도 구현 필요
- center-crop 좌표: 구현됨 (crop_size×crop_size, RD4AD 입력용)

large_bbox 처리 방침:
- context margin 포함 bbox > 256px 시 warning 기록 (제외하지 않음)
- RD4AD 입력은 원본 bbox 중심 기준 crop_size×crop_size 고정 crop 사용
- manifest에 y0_fixed_crop/x0_fixed_crop/y1_fixed_crop/x1_fixed_crop 포함

권장 smoke test 환자 조합 (stage1_dev 전용):
- MSD_lung_043: weak_case 대표 (MSD_Lung)
- LUNG1-148: high recall NSCLC (patch_recall=0.86) — lesion_candidate 예상
- LUNG1-228: high recall NSCLC (patch_recall=0.80) — lesion_candidate 예상
- LUNG1-108: high FP NSCLC (patch_recall=0.28, continuous_hit=1.0) — FP 검증

실행 명령 초안:
  [dry-run 확인용 — 파일 생성/수정 없음]:
    source ~/ai_env/bin/activate && \\
    python scripts/build_second_stage_smoke_candidates.py \\
      --patients MSD_lung_043 LUNG1-148 LUNG1-228 LUNG1-108 --limit 4 --dry-run

  [실제 manifest-only smoke test — ChatGPT 검토 + 사용자 승인 후]:
    source ~/ai_env/bin/activate && \\
    python scripts/build_second_stage_smoke_candidates.py \\
      --patients MSD_lung_043 LUNG1-148 LUNG1-228 LUNG1-108 --limit 4 \\
      --max-overlays-per-patient 30
  ※ crop(npz) 저장 미구현 상태이므로 위 명령은 manifest-only smoke임
  ※ overlay는 smoke_test 전용 경로에 생성됨 (v1 폴더 사용 금지)
  ※ 기존 overlay PNG가 있으면 --force 없이 즉시 중단됨 (자동 삭제 안 함)
  ※ --force는 사용자가 명시 승인할 때만 추가한다
  ※ 실제 smoke test 실행은 ChatGPT 검토 후 별도 승인 대기
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

SCRIPT_NAME = "build_second_stage_smoke_candidates.py"
DATASET_TAG = "smoke_test"

# ============================================================
# 안전 상수
# ============================================================
P95_THRESHOLD = 14.3774
DEFAULT_CROP_SIZE = 96
DEFAULT_CONTEXT_MARGIN = 32
DEFAULT_Z_RANGE = 1          # z_center ± 1
DEFAULT_LIMIT = 5
MAX_LIMIT = 5
MAX_BBOX_SIZE = 256          # 이 이상이면 경고

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
            "stage2_holdout 봉인 환자는 smoke test 포함 금지."
        )
    if stage_split != "stage1_dev":
        raise RuntimeError(
            f"[GUARD] stage1_dev가 아닌 환자 감지: {patient_id} "
            f"(stage_split={stage_split})\n"
            "smoke test는 stage1_dev 환자만 허용."
        )


# ============================================================
# 데이터 로딩
# ============================================================
def load_split(
    limit: int,
    patients: Optional[List[str]] = None,
) -> pd.DataFrame:
    """split CSV 로드 → stage1_dev 필터 → limit 적용."""
    df = pd.read_csv(SPLIT_CSV, encoding="utf-8-sig")

    # required columns 확인
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

    if patients:
        # 봉인 환자(STAGE2_HOLDOUT_BLOCKED)가 --patients에 포함됐는지 먼저 확인
        blocked_requested = [p for p in patients if p in STAGE2_HOLDOUT_BLOCKED]
        if blocked_requested:
            raise RuntimeError(
                f"[GUARD] 봉인 환자(stage2_holdout blocked)가 --patients에 포함됨: {blocked_requested}\n"
                "봉인 환자는 smoke test 포함 금지. 실행 중단."
            )

        # 요청 환자 vs stage1_dev 실제 환자 비교
        dev_pids = set(df_dev["patient_id"].tolist())
        requested_set = set(patients)
        found_set = requested_set & dev_pids
        missing_set = requested_set - dev_pids

        if missing_set:
            # 누락 원인을 세분화: stage2_holdout 소속 vs CSV 자체에 없음
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

    if len(df_dev) > limit:
        df_dev = df_dev.head(limit)
        print(f"[INFO] --limit {limit} 적용: {len(df_dev)}명만 처리")

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
        # a, b = (y0, x0, y1, x1, score)
        # 두 bbox가 맞닿거나 겹치면 True (gap=0 허용)
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

    반환: [{'z_start', 'z_end', 'z_center', 'y0', 'x0', 'y1', 'x1',
             'n_patches', 'mean_score', 'max_score'}, ...]
    """
    candidates: List[Dict] = []
    active: List[Dict] = []  # 현재 열린 candidate 목록

    for z in sorted(slice_bboxes.keys()):
        current_bboxes = slice_bboxes[z]
        new_active: List[Dict] = []
        used: set = set()

        for cand in active:
            if z != cand["z_end"] + 1:
                # z gap → candidate 종료
                candidates.append(cand)
                continue

            # 현재 slice bbox와 매칭 시도 (첫 번째 overlap 기준)
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

        # 매칭 안 된 bbox → 새 candidate 시작
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
    """
    context margin 추가 후 CT 경계 내로 clip.
    clipping 방식 사용: 경계 밖 HU값을 가정하지 않아 안전함.
    """
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
    """
    z_center slice에서 crop bbox 영역의 lesion mask overlap 계산.
    반환: (lesion_pixel_count, lesion_overlap_ratio)
    """
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
    """
    4가지 기준으로 lesion overlap을 계산해 진단 컬럼을 반환한다.
    기존 lesion_overlap_ratio(context_z_center)는 이 함수로 대체하지 않는다.
    """
    # context bbox, z_center
    ctx_px, ctx_ratio = compute_lesion_overlap(
        lesion_mask, z_center, context_y0, context_x0, context_y1, context_x1
    )

    # fixed_crop bbox, z_center
    fc_px, fc_ratio = compute_lesion_overlap(
        lesion_mask, z_center, fc_y0, fc_x0, fc_y1, fc_x1
    )

    # raw_bbox, z_center
    raw_px, raw_ratio = compute_lesion_overlap(
        lesion_mask, z_center, raw_y0, raw_x0, raw_y1, raw_x1
    )

    # zstack_max: z_lo ~ z_hi 범위 각 slice에서 context bbox로 overlap 계산 후 max
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
) -> Tuple[bool, str, str]:
    """
    candidate에 label 부여.
    반환: (coverage_label, rd4ad_label, binary_label)

    coverage_label : lesion_overlap_ratio >= 0.01
    rd4ad_label    : 'normal_like' | 'lesion_candidate' | 'ambiguous'
    binary_label   : 'positive' | 'hard_negative' | 'ambiguous'

    주의:
    - RD4AD 학습에서 positive(lesion_candidate)는 학습 target이 아님.
      rd4ad_label='lesion_candidate'는 평가 ground truth 전용.
    - 모든 candidate는 padim_score >= p95 기준이므로
      binary_label에서 easy_negative는 발생하지 않음.
    """
    coverage_label = lesion_overlap_ratio >= 0.01

    if lesion_overlap_ratio == 0.0:
        rd4ad_label = "normal_like"
        binary_label = "hard_negative"
    elif lesion_overlap_ratio >= 0.10:
        rd4ad_label = "lesion_candidate"
        binary_label = "positive"
    else:
        rd4ad_label = "ambiguous"
        binary_label = "ambiguous"

    return coverage_label, rd4ad_label, binary_label


# ============================================================
# overlay 생성 (smoke_test 전용)
# ============================================================
def generate_overlay(
    out_dir: Path,
    patient_id: str,
    safe_id: str,
    candidate: Dict,
    dataset_root: Path,
) -> None:
    """
    CT slice 위 candidate bbox + lesion mask를 overlay한 이미지 저장.

    IMPORTANT:
    - smoke_test 전용 경로에만 저장.
    - v1 visualizations 폴더에는 절대 쓰지 않음.
    - HU windowing: [-1000, 400]
    - green rectangle: context margin crop bbox (두께 2)
    - cyan rectangle: fixed-size center crop bbox (RD4AD 입력용, 두께 2)
    - yellow dot: fixed crop 중심점
    - red fill (30% 투명): lesion mask 영역
    - red contour (두께 2): lesion mask 외곽
    - 병변 없는 z: "no lesion contour on this z" 텍스트
    - 텍스트: rd4ad_label, selected/context/fixed_crop overlap, max_padim_score
    """
    import cv2  # lazy import — cv2 없으면 건너뜀

    z = int(candidate["z_center"])
    vol_dir = dataset_root / "volumes_npy" / safe_id
    ct_path = vol_dir / "ct_hu.npy"
    mask_path = vol_dir / "lesion_mask_model_roi.npy"

    if not ct_path.exists() or not mask_path.exists():
        warnings.warn(
            f"[OVERLAY] CT/mask 없음, 건너뜀: {patient_id}", UserWarning, stacklevel=2
        )
        return

    ct_vol = np.load(ct_path, mmap_mode="r")
    lesion_mask = np.load(mask_path, mmap_mode="r")

    if z >= ct_vol.shape[0]:
        warnings.warn(
            f"[OVERLAY] z={z} 범위 초과, 건너뜀: {patient_id}", UserWarning, stacklevel=2
        )
        return

    # HU windowing → uint8
    HU_MIN, HU_MAX = -1000, 400
    slice_f = np.array(ct_vol[z], dtype=np.float32)
    slice_norm = np.clip((slice_f - HU_MIN) / (HU_MAX - HU_MIN), 0.0, 1.0)
    overlay = cv2.cvtColor((slice_norm * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # context margin crop bbox (green)
    cv2.rectangle(
        overlay,
        (int(candidate["x0_crop"]), int(candidate["y0_crop"])),
        (int(candidate["x1_crop"]), int(candidate["y1_crop"])),
        (0, 255, 0), 2,
    )

    # fixed center crop bbox (cyan, 두께 2), if present
    if "x0_fixed_crop" in candidate:
        cv2.rectangle(
            overlay,
            (int(candidate["x0_fixed_crop"]), int(candidate["y0_fixed_crop"])),
            (int(candidate["x1_fixed_crop"]), int(candidate["y1_fixed_crop"])),
            (255, 255, 0), 2,
        )
        # fixed crop 중심점 (노란색 작은 원)
        cx = (int(candidate["x0_fixed_crop"]) + int(candidate["x1_fixed_crop"])) // 2
        cy = (int(candidate["y0_fixed_crop"]) + int(candidate["y1_fixed_crop"])) // 2
        cv2.circle(overlay, (cx, cy), 4, (0, 255, 255), -1)

    # lesion mask: 반투명 red fill + 두꺼운 contour
    mask_slice = np.array(lesion_mask[z], dtype=np.uint8)
    contours, _ = cv2.findContours(mask_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        fill_layer = overlay.copy()
        cv2.drawContours(fill_layer, contours, -1, (0, 0, 255), -1)
        cv2.addWeighted(fill_layer, 0.3, overlay, 0.7, 0, overlay)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    else:
        cv2.putText(
            overlay, "no lesion contour on this z",
            (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1,
        )

    # 텍스트 (여러 줄): rd4ad_label, selected/context/fixed_crop overlap, max_padim_score
    rd4ad_lbl = candidate.get("rd4ad_label", "?")
    sel_overlap = candidate.get("selected_lesion_overlap_ratio", candidate.get("lesion_overlap_ratio", 0.0))
    ctx_overlap = candidate.get("lesion_overlap_ratio_context", 0.0)
    fc_overlap = candidate.get("lesion_overlap_ratio_fixed_crop", 0.0)
    max_score = candidate.get("max_padim_score", 0.0)
    label_basis = candidate.get("label_basis", "?")
    text_lines = [
        f"{rd4ad_lbl}  basis={label_basis}",
        f"sel={sel_overlap:.3f}  ctx={ctx_overlap:.3f}",
        f"fc={fc_overlap:.3f}  score={max_score:.2f}",
    ]
    for li, line in enumerate(text_lines):
        cv2.putText(
            overlay, line,
            (5, 16 + li * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{patient_id}_z{z:04d}_{candidate['candidate_id']}.png"
    check_not_v1_output(out_path)
    cv2.imwrite(str(out_path), overlay)


# ============================================================
# 런타임 / 에러 기록
# ============================================================
def record_runtime(reports_dir: Path, row: Dict, dry_run: bool = False) -> None:
    if dry_run:
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


def record_error(reports_dir: Path, patient_id: str, error: str, dry_run: bool = False) -> None:
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
                f"h={bbox_h} w={bbox_w} (>{MAX_BBOX_SIZE}px) — summary에 기록"
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
        # 원본 bbox 중심 기준으로 계산 (context margin 확장 전)
        bbox_center_y = (cand["y0"] + cand["y1"]) // 2
        bbox_center_x = (cand["x0"] + cand["x1"]) // 2
        half_crop = args.crop_size // 2
        fc_y0 = max(0, bbox_center_y - half_crop)
        fc_x0 = max(0, bbox_center_x - half_crop)
        fc_y1 = min(ct_h, fc_y0 + args.crop_size)
        fc_x1 = min(ct_w, fc_x0 + args.crop_size)
        # 경계 클리핑 후 재조정 (크기 보장)
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

        # label 부여 (selected_overlap_ratio 기준)
        coverage_label, rd4ad_label, binary_label = assign_labels(selected_overlap_ratio)

        candidate_id = f"{patient_id}_c{candidate_idx_start + i:05d}"

        candidates.append({
            "candidate_id": candidate_id,
            "patient_id": patient_id,
            "safe_id": safe_id,
            "group": group,
            "stage_split": stage_split,
            "dataset_tag": DATASET_TAG,
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
            # score 정보
            "n_positive_patch": cand["n_patches"],
            "mean_padim_score": round(cand["mean_score"], 6),
            "max_padim_score": round(cand["max_score"], 6),
            # lesion overlap (기존 컬럼 — selected_lesion_overlap_ratio와 동일값)
            "lesion_pixel_count": selected_lesion_px,
            "lesion_overlap_ratio": round(selected_overlap_ratio, 6),
            # labels
            "coverage_label": coverage_label,
            "rd4ad_label": rd4ad_label,       # normal_like | lesion_candidate | ambiguous
            "binary_label": binary_label,     # positive | hard_negative | ambiguous
            # center-crop 고정 좌표 (RD4AD 입력용, crop_size × crop_size)
            "y0_fixed_crop": fc_y0,
            "x0_fixed_crop": fc_x0,
            "y1_fixed_crop": fc_y1,
            "x1_fixed_crop": fc_x1,
            "fixed_crop_size": args.crop_size,
            # 경고 플래그
            "bbox_too_large": bbox_too_large,
            # selected label 기준 컬럼
            "label_basis": args.label_basis,
            "selected_lesion_overlap_ratio": round(selected_overlap_ratio, 6),
            "selected_lesion_pixel_count": selected_lesion_px,
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
            "Second-Stage Lesion Refiner v1 Phase 4 Smoke Test — "
            "stage1_dev 환자 3~5명 기준 candidate 생성 검증"
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=(
            f"처리할 환자 수 상한 (기본 {DEFAULT_LIMIT}, 최대 {MAX_LIMIT}). "
            "smoke test는 항상 소수 환자만."
        ),
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        default=None,
        help=(
            "처리할 patient_id 목록 "
            "(예: --patients MSD_lung_043 MSD_lung_079). "
            "지정 시 stage1_dev에서 해당 환자만 처리."
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
        help="실제 파일 저장 없이 검증만 수행",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        default=False,
        help="overlay 이미지 생성 건너뜀 (현재 stub)",
    )
    parser.add_argument(
        "--max-overlays-per-patient",
        type=int,
        default=30,
        help="환자당 overlay 최대 저장 수 (기본 30)",
    )
    parser.add_argument(
        "--overlay-labels",
        type=str,
        nargs="+",
        default=["ambiguous", "lesion_candidate"],
        help="overlay 저장 우선 대상 rd4ad_label (기본: ambiguous lesion_candidate)",
    )
    parser.add_argument(
        "--include-large-bbox-overlays",
        action="store_true",
        default=True,
        help="large_bbox=True인 candidate도 overlay 저장 대상에 포함 (기본 True)",
    )
    parser.add_argument(
        "--label-basis",
        dest="label_basis",
        type=str,
        choices=["context_z_center", "fixed_crop_z_center", "raw_bbox_z_center", "zstack_max"],
        default="fixed_crop_z_center",
        help=(
            "rd4ad_label / binary_label 계산 기준 (기본: fixed_crop_z_center). "
            "RD4AD 입력은 fixed 96×96 crop이므로 기본 label도 fixed_crop 기준. "
            "context/raw_bbox/zstack_max는 진단용 컬럼으로 유지."
        ),
    )
    parser.add_argument(
        "--visualization-subdir",
        dest="visualization_subdir",
        type=str,
        default="smoke_test",
        help=(
            "overlay 이미지 저장 서브디렉터리 이름 "
            "(기본: smoke_test). "
            "예: smoke_test_mask_enhanced — 기존 smoke_test PNG를 덮어쓰지 않고 새 폴더에 저장."
        ),
    )
    parser.add_argument(
        "--overlay-only",
        dest="overlay_only",
        action="store_true",
        default=False,
        help=(
            "기존 candidate_manifest_smoke.csv를 read-only로 읽고 overlay만 재생성. "
            "manifest/summary 재생성 없음. --visualization-subdir 폴더에만 PNG 저장."
        ),
    )
    return parser.parse_args()


# ============================================================
# 메인
# ============================================================
def main() -> None:
    args = parse_args()

    # limit 상한 강제
    if args.limit > MAX_LIMIT:
        print(
            f"[WARN] --limit {args.limit} > 최대 {MAX_LIMIT}. "
            f"{MAX_LIMIT}으로 제한."
        )
        args.limit = MAX_LIMIT
    if args.limit <= 0:
        print("[ERROR] --limit은 1 이상이어야 합니다.")
        sys.exit(1)

    # --max-overlays-per-patient guard
    if args.max_overlays_per_patient < 0:
        print("[ERROR] --max-overlays-per-patient는 0 이상이어야 합니다.")
        sys.exit(1)
    if args.max_overlays_per_patient == 0:
        print("[INFO] --max-overlays-per-patient=0 : overlay 저장 안 함으로 처리.")
        args.no_overlay = True

    print(
        f"\n{'=' * 60}\n"
        f"  Second-Stage Lesion Refiner v1 — Phase 4 Smoke Test\n"
        f"  dataset_tag : {DATASET_TAG}\n"
        f"  limit       : {args.limit}\n"
        f"  z_range     : ±{args.z_range}\n"
        f"  margin      : {args.context_margin}px\n"
        f"  crop_size   : {args.crop_size}×{args.crop_size}\n"
        f"  dry_run     : {args.dry_run}\n"
        f"  label_basis : {args.label_basis}\n"
        f"\n"
        f"  [구현 상태]\n"
        f"  overlay 저장: 구현됨 (cv2, smoke_test 전용 경로)\n"
        f"  crop(npz) 저장: 미구현 — manifest/좌표 검증 전용\n"
        f"  center-crop 좌표: 구현됨 (crop_size×crop_size, RD4AD 입력용)\n"
        f"  이 실행은 manifest-only smoke (npz 저장 없음)\n"
        f"\n"
        f"  [IMPORTANT] smoke test 결과 = 학습 데이터 아님\n"
        f"              성능 결론 도출 금지\n"
        f"              stage2_holdout 봉인 유지\n"
        f"{'=' * 60}\n"
    )

    # 출력 경로
    smoke_cand_dir = OUTPUT_BASE / "candidates" / "smoke_test"
    smoke_crops_dir = smoke_cand_dir / "crops"
    smoke_vis_dir = OUTPUT_BASE / "visualizations" / args.visualization_subdir
    smoke_reports_dir = OUTPUT_BASE / "reports"

    manifest_path = smoke_cand_dir / "candidate_manifest_smoke.csv"
    summary_path = smoke_reports_dir / "smoke_test_summary.json"

    # v1 outputs 쓰기 가드
    check_not_v1_output(manifest_path)
    check_not_v1_output(smoke_crops_dir)
    check_not_v1_output(smoke_vis_dir)

    # overwrite 가드 (overlay-only 모드에서는 manifest/summary 쓰기 없음)
    if not args.overlay_only and not args.dry_run:
        check_not_overwrite(manifest_path, args.force)
        check_not_overwrite(summary_path, args.force)

    # overlay PNG overwrite 가드
    if not args.dry_run and not args.no_overlay and args.max_overlays_per_patient > 0:
        existing_pngs = list(smoke_vis_dir.glob("*.png"))
        if existing_pngs and not args.force:
            raise FileExistsError(
                f"[GUARD] overlay 폴더에 기존 PNG {len(existing_pngs)}개 존재: {smoke_vis_dir}\n"
                "--force 없이는 overlay 덮어쓰기 금지.\n"
                "--force는 사용자가 명시 승인할 때만 사용한다.\n"
                "자동 삭제는 하지 않습니다. 폴더를 직접 확인하거나 --force를 추가하세요."
            )

    # dataset root 로드
    dataset_root = load_dataset_root()
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root 없음: {dataset_root}")

    # ── overlay-only 모드 (manifest read-only, overlay 재생성만) ────────────────
    if args.overlay_only:
        if not manifest_path.exists():
            print(f"[ERROR] --overlay-only: manifest 없음: {manifest_path}")
            sys.exit(1)
        manifest_df = pd.read_csv(manifest_path, encoding="utf-8-sig")
        print(
            f"[INFO] --overlay-only: manifest {len(manifest_df)}개 로드 (read-only)\n"
            f"  → overlay 저장 경로: {smoke_vis_dir}\n"
            f"  → visualization_subdir: {args.visualization_subdir}"
        )
        smoke_vis_dir.mkdir(parents=True, exist_ok=True)
        oo_existing = list(smoke_vis_dir.glob("*.png"))
        if oo_existing and not args.force:
            raise FileExistsError(
                f"[GUARD] overlay 폴더에 기존 PNG {len(oo_existing)}개 존재: {smoke_vis_dir}\n"
                "--force 없이는 overlay 덮어쓰기 금지.\n"
                "--force는 사용자가 명시 승인할 때만 사용한다.\n"
                "자동 삭제는 하지 않습니다. 폴더를 직접 확인하거나 --force를 추가하세요."
            )
        oo_saved_by_patient: Dict[str, int] = {}
        for patient_id_val, grp in manifest_df.groupby("patient_id"):
            patient_id_val = str(patient_id_val)
            cands_oo = grp.to_dict("records")
            safe_id_oo = str(cands_oo[0]["safe_id"])
            p1 = [c for c in cands_oo if c.get("rd4ad_label") in args.overlay_labels]
            p2 = [
                c for c in cands_oo
                if str(c.get("bbox_too_large", "False")).lower() == "true"
                and args.include_large_bbox_overlays
                and c not in p1
            ]
            p3 = sorted(
                [
                    c for c in cands_oo
                    if c.get("rd4ad_label") == "normal_like"
                    and c not in p1
                    and c not in p2
                ],
                key=lambda x: -float(x.get("max_padim_score", 0.0)),
            )[:5]
            selected_oo = (p1 + p2 + p3)[: args.max_overlays_per_patient]
            n_saved = 0
            for cand_oo in selected_oo:
                generate_overlay(smoke_vis_dir, patient_id_val, safe_id_oo, cand_oo, dataset_root)
                n_saved += 1
            oo_saved_by_patient[patient_id_val] = n_saved
            print(f"  {patient_id_val}: overlay {n_saved}개 → {smoke_vis_dir}")
        total_oo = sum(oo_saved_by_patient.values())
        print(
            f"\n[INFO] --overlay-only 완료: 총 {total_oo}개 PNG → {smoke_vis_dir}\n"
            f"  환자별: {oo_saved_by_patient}"
        )
        return
    # ── overlay-only 끝 ──────────────────────────────────────────────────────

    # split 로드 + stage1_dev 필터
    split_df = load_split(args.limit, args.patients)
    print(f"[INFO] 처리 대상 환자 {len(split_df)}명:")
    for _, r in split_df.iterrows():
        wf = " [weak_case]" if r.get("weak_case_flag", 0) else ""
        print(f"  - {r['patient_id']} ({r['group']}, {r['stage_split']}){wf}")

    # 출력 디렉터리 생성 (dry_run이 아닐 때만)
    # smoke_crops_dir: crop(npz) 저장 미구현 → 생성 안 함
    # smoke_vis_dir: --no-overlay 아닐 때만 생성 (overlay 내부 mkdir로 처리)
    if not args.dry_run:
        for d in [smoke_cand_dir, smoke_reports_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # 환자별 처리
    all_candidates: List[Dict] = []
    n_scored = 0
    n_failed = 0
    n_large_bbox = 0
    candidate_idx = 0
    overlay_saved_by_patient: Dict[str, int] = {}
    start_time = datetime.now()

    for _, row in split_df.iterrows():
        patient_id = str(row["patient_id"])
        print(f"\n[{n_scored + n_failed + 1}/{len(split_df)}] {patient_id} ...")
        try:
            cands, n_cand = process_patient(
                row, dataset_root, args, candidate_idx
            )
            all_candidates.extend(cands)
            candidate_idx += n_cand
            n_scored += 1
            n_large = sum(1 for c in cands if c.get("bbox_too_large", False))
            n_large_bbox += n_large
            print(f"  → candidate {n_cand}개 생성 (large_bbox={n_large})")
            record_runtime(
                smoke_reports_dir,
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "script": SCRIPT_NAME,
                    "metric": f"n_candidates_{patient_id}",
                    "value": n_cand,
                },
                dry_run=args.dry_run,
            )
        except Exception as exc:
            print(f"  [ERROR] {patient_id}: {exc}")
            record_error(smoke_reports_dir, patient_id, str(exc), dry_run=args.dry_run)
            n_failed += 1
            continue

        # overlay 생성 (smoke_test 전용 경로)
        if not args.no_overlay and not args.dry_run:
            safe_id_for_overlay = str(row["safe_id"])
            # 저장 정책: rd4ad_label 우선순위 → large_bbox → normal_like 상위
            priority_1 = [c for c in cands if c["rd4ad_label"] in args.overlay_labels]
            priority_2 = [
                c for c in cands
                if c.get("bbox_too_large") and args.include_large_bbox_overlays and c not in priority_1
            ]
            priority_3 = sorted(
                [
                    c for c in cands
                    if c["rd4ad_label"] == "normal_like" and c not in priority_1 and c not in priority_2
                ],
                key=lambda x: -x["max_padim_score"],
            )[:5]
            selected = (priority_1 + priority_2 + priority_3)[:args.max_overlays_per_patient]
            n_overlays_saved = 0
            for cand in selected:
                generate_overlay(smoke_vis_dir, patient_id, safe_id_for_overlay, cand, dataset_root)
                n_overlays_saved += 1
            overlay_saved_by_patient[patient_id] = n_overlays_saved
            if len(cands) > n_overlays_saved:
                print(
                    f"  [INFO] overlay 저장: {n_overlays_saved}/{len(cands)}개 "
                    f"(--max-overlays-per-patient={args.max_overlays_per_patient})"
                )

    elapsed = (datetime.now() - start_time).total_seconds()

    # 결과 저장
    if not args.dry_run and all_candidates:
        cand_df = pd.DataFrame(all_candidates)
        cand_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")
        print(f"\n[INFO] manifest 저장: {manifest_path} ({len(all_candidates)}개)")
    elif args.dry_run:
        print(
            f"\n[DRY-RUN] 파일 생성/수정 없음. "
            f"candidate {len(all_candidates)}개 생성 예정 (실제 저장 안 됨)."
        )

    # label 요약
    binary_summary: Dict[str, int] = defaultdict(int)
    rd4ad_summary: Dict[str, int] = defaultdict(int)
    for c in all_candidates:
        binary_summary[c["binary_label"]] += 1
        rd4ad_summary[c["rd4ad_label"]] += 1

    summary = {
        "script": SCRIPT_NAME,
        "dataset_tag": DATASET_TAG,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_patients_requested": len(split_df),
        "n_patients_scored": n_scored,
        "n_patients_failed": n_failed,
        "n_candidates_total": len(all_candidates),
        "n_candidates_large_bbox": n_large_bbox,
        "binary_label_summary": dict(binary_summary),
        "rd4ad_label_summary": dict(rd4ad_summary),
        "elapsed_seconds": round(elapsed, 2),
        "dry_run": args.dry_run,
        "label_basis": args.label_basis,
        "selected_rd4ad_label_summary": dict(rd4ad_summary),
        "selected_binary_label_summary": dict(binary_summary),
        "args": {
            "limit": args.limit,
            "context_margin": args.context_margin,
            "z_range": args.z_range,
            "crop_size": args.crop_size,
            "label_basis": args.label_basis,
        },
        "overlay_saved_by_patient": overlay_saved_by_patient,
        "overlay_saved_total": sum(overlay_saved_by_patient.values()),
        "max_overlays_per_patient": args.max_overlays_per_patient,
        "overlay_labels": args.overlay_labels,
        "include_large_bbox_overlays": args.include_large_bbox_overlays,
        "overlay_status": "구현됨 (cv2, smoke_test 전용 경로, --no-overlay로 건너뜀 가능)",
        "crop_save_status": "미구현 — manifest/좌표 검증 전용 (Phase 4에서 crop 저장 필요 시 별도 구현)",
        "note": (
            "smoke test 결과는 학습 데이터가 아님. "
            "성능 결론 도출 금지. "
            "stage2_holdout 미포함 확인됨."
        ),
    }

    if not args.dry_run:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 요약 저장: {summary_path}")

    record_runtime(
        smoke_reports_dir,
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "script": SCRIPT_NAME,
            "metric": "smoke_test_elapsed_seconds",
            "value": round(elapsed, 2),
        },
        dry_run=args.dry_run,
    )

    dry_run_note = "  [DRY-RUN] 파일 생성/수정 없음\n" if args.dry_run else ""
    print(
        f"\n{'=' * 60}\n"
        f"  완료: scored={n_scored}, failed={n_failed}\n"
        f"  candidates={len(all_candidates)}, large_bbox={n_large_bbox}\n"
        f"  binary_label  : {dict(binary_summary)}\n"
        f"  rd4ad_label   : {dict(rd4ad_summary)}\n"
        f"  elapsed       : {elapsed:.1f}s\n"
        f"\n"
        f"{dry_run_note}"
        f"  [REMINDER] 이 결과는 학습 데이터로 승격 금지\n"
        f"{'=' * 60}\n"
    )

    # 진단 요약 (dry-run 여부 무관하게 all_candidates가 있으면 출력)
    if all_candidates:
        n01_ctx  = sum(1 for c in all_candidates if c["lesion_overlap_ratio_context"]  >= 0.01)
        n05_ctx  = sum(1 for c in all_candidates if c["lesion_overlap_ratio_context"]  >= 0.05)
        n10_ctx  = sum(1 for c in all_candidates if c["lesion_overlap_ratio_context"]  >= 0.10)
        n01_fc   = sum(1 for c in all_candidates if c["lesion_overlap_ratio_fixed_crop"] >= 0.01)
        n05_fc   = sum(1 for c in all_candidates if c["lesion_overlap_ratio_fixed_crop"] >= 0.05)
        n10_fc   = sum(1 for c in all_candidates if c["lesion_overlap_ratio_fixed_crop"] >= 0.10)
        n01_raw  = sum(1 for c in all_candidates if c["lesion_overlap_ratio_raw_bbox"] >= 0.01)
        n05_raw  = sum(1 for c in all_candidates if c["lesion_overlap_ratio_raw_bbox"] >= 0.05)
        n10_raw  = sum(1 for c in all_candidates if c["lesion_overlap_ratio_raw_bbox"] >= 0.10)
        n01_z    = sum(1 for c in all_candidates if c["lesion_overlap_ratio_zstack_max"] >= 0.01)
        n05_z    = sum(1 for c in all_candidates if c["lesion_overlap_ratio_zstack_max"] >= 0.05)
        n10_z    = sum(1 for c in all_candidates if c["lesion_overlap_ratio_zstack_max"] >= 0.10)

        # 환자별 max overlap (context_z_center 기준)
        patient_max: Dict[str, float] = {}
        for c in all_candidates:
            pid = c["patient_id"]
            v = c["lesion_overlap_ratio_context"]
            if pid not in patient_max or v > patient_max[pid]:
                patient_max[pid] = v
        patient_max_lines = "\n".join(
            f"    {pid}: {v:.4f}" for pid, v in sorted(patient_max.items())
        )

        # ambiguous 후보 (rd4ad_label == "ambiguous") overlap 범위
        ambiguous_overlaps = [
            c["lesion_overlap_ratio_context"]
            for c in all_candidates if c["rd4ad_label"] == "ambiguous"
        ]
        n_ambiguous = len(ambiguous_overlaps)
        if n_ambiguous > 0:
            min_ambiguous = min(ambiguous_overlaps)
            max_ambiguous = max(ambiguous_overlaps)
            ambiguous_line = f"count={n_ambiguous}, min={min_ambiguous:.4f}, max={max_ambiguous:.4f}"
        else:
            ambiguous_line = "count=0 (없음)"

        # selected 기준 rd4ad_label / binary_label 분포
        sel_rd4ad_counts: Dict[str, int] = {}
        sel_binary_counts: Dict[str, int] = {}
        for c in all_candidates:
            sel_rd4ad_counts[c["rd4ad_label"]] = sel_rd4ad_counts.get(c["rd4ad_label"], 0) + 1
            sel_binary_counts[c["binary_label"]] = sel_binary_counts.get(c["binary_label"], 0) + 1

        print(
            f"\n[DRY-RUN 진단 요약]\n"
            f"  기준별 max overlap:\n"
            f"    context_z_center : max={max(c['lesion_overlap_ratio_context'] for c in all_candidates):.4f}\n"
            f"    fixed_crop       : max={max(c['lesion_overlap_ratio_fixed_crop'] for c in all_candidates):.4f}\n"
            f"    raw_bbox         : max={max(c['lesion_overlap_ratio_raw_bbox'] for c in all_candidates):.4f}\n"
            f"    zstack_max       : max={max(c['lesion_overlap_ratio_zstack_max'] for c in all_candidates):.4f}\n"
            f"\n"
            f"  기준별 후보 수 (>=0.01 / >=0.05 / >=0.10):\n"
            f"    context  : >=0.01={n01_ctx}, >=0.05={n05_ctx}, >=0.10={n10_ctx}\n"
            f"    fixed    : >=0.01={n01_fc},  >=0.05={n05_fc},  >=0.10={n10_fc}\n"
            f"    raw_bbox : >=0.01={n01_raw}, >=0.05={n05_raw}, >=0.10={n10_raw}\n"
            f"    zstack   : >=0.01={n01_z},   >=0.05={n05_z},   >=0.10={n10_z}\n"
            f"\n"
            f"  환자별 max overlap (context_z_center 기준):\n"
            f"{patient_max_lines}\n"
            f"\n"
            f"  ambiguous 후보 overlap 범위 (context_z_center 기준):\n"
            f"    {ambiguous_line}\n"
            f"\n"
            f"  [selected label_basis={args.label_basis}]\n"
            f"  rd4ad_label (selected) : {sel_rd4ad_counts}\n"
            f"  binary_label (selected): {sel_binary_counts}\n"
        )


if __name__ == "__main__":
    main()
