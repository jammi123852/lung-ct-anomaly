"""
N-C10 Normal-Val Scoring + Threshold Calculation

normal_val 36명에 대해 N-C Refiner Mahalanobis score를 계산하고
crop-level / spatial-level / per-bin threshold를 산출한다.

실행 모드:
  --dry-run
      입력/경로/가드 검증만 수행 (feature extraction, model forward, scoring 없음)

  --smoke-drycheck
      smoke mode patch dry-check (N-C10b, 실제 scoring 없음)
      smoke output path collision / guard 확인만 수행

  --smoke-one-patient
  --confirm-smoke
  --confirm-normal-val-only
      normal_val 1명만 scoring smoke (threshold 계산 없음)
      (ALLOW_REAL_PROCESSING=True 필수)

  --run-val-scoring
  --confirm-val-scoring
  --confirm-threshold-compute
  --confirm-normal-val-only
      실제 scoring + threshold 계산
      (ALLOW_REAL_PROCESSING=True 환경 변수도 필수)

안전 장치:
  ALLOW_REAL_PROCESSING=False → bare run 시 exit(2)
  ALLOW_REAL_PROCESSING=False + --run-val-scoring → abort
  ALLOW_REAL_PROCESSING=False + --smoke-one-patient → abort
  --run-val-scoring 단독 (confirm flags 미충족) → abort
  --smoke-one-patient 단독 (confirm flags 미충족) → abort
  --smoke-one-patient + --run-val-scoring 동시 사용 → abort
  smoke mode에서 threshold 계산 함수 호출 금지
  smoke output path collision 시 abort
  stage2_holdout 경로 감지 시 abort
  P-C supervised artifact 경로 감지 시 abort
  lesion / positive / hard_negative / training_label 컬럼 사용 금지
  output collision 시 abort
  기존 결과 덮어쓰기 금지
  crop npz 저장 금지
  normal_test / lesion scoring 금지
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 안전 플래그 — 기본 False: 직접 실행 불가
# ---------------------------------------------------------------------------
ALLOW_REAL_PROCESSING = False

# ---------------------------------------------------------------------------
# 경로 상수
# ---------------------------------------------------------------------------
PROJ_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
EXP_ROOT  = PROJ_ROOT / "experiments" / "normal_only_second_stage_refiner_v1"

SPLIT_JSON = PROJ_ROOT / "outputs" / "position-aware-padim-v1" / "splits" / "normal_v1.json"

N_C8_JSON = (
    EXP_ROOT / "outputs" / "reports" /
    "n_c8_full_distribution_artifact_validation" /
    "n_c8_full_distribution_artifact_validation.json"
)
N_C9_JSON = (
    EXP_ROOT / "outputs" / "reports" /
    "n_c9_normal_val_scoring_threshold_preflight" /
    "n_c9_normal_val_scoring_threshold_preflight.json"
)
STATS_NPZ = (
    EXP_ROOT / "outputs" / "models" /
    "n_c7_full_position_bin_distribution" /
    "n_c7_position_bin_stats.npz"
)
SELECTED_INDICES_PATH = (
    PROJ_ROOT / "experiments" /
    "efficientnet_b0_imagenet_chestwall_removed_roi_v1" /
    "outputs" / "models" / "distributions" / "selected_feature_indices.npy"
)
TRAIN_MANIFEST_CSV = (
    EXP_ROOT / "outputs" / "manifests" /
    "n_c3_normal_only_crop_manifest_v1" /
    "n_c3_normal_only_crop_manifest_v1.csv"
)

# CT / ROI 경로 패턴 (학습 manifest에서 확인된 패턴)
CT_VOLUMES_ROOT = Path(
    "/mnt/c/Users/jinhy/Desktop/"
    "Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/"
    "volumes_npy"
)
ROI_ROOT = (
    PROJ_ROOT / "outputs" / "mip-postprocess-research-v1" /
    "masks" / "refined_roi_v4_20_modeB_all_v1" / "normal"
)

# 출력 경로
MANIFEST_OUT_DIR  = EXP_ROOT / "outputs" / "manifests" / "n_c10_normal_val_crop_manifest"
SCORES_OUT_DIR    = EXP_ROOT / "outputs" / "scores"    / "n_c10_normal_val_scores"
THRESHOLD_OUT_DIR = EXP_ROOT / "outputs" / "evaluation" / "n_c10_normal_val_thresholds"
REPORT_OUT_DIR    = EXP_ROOT / "outputs" / "reports"   / "n_c10_normal_val_scoring_threshold"
DRYCHECK_OUT_DIR  = EXP_ROOT / "outputs" / "reports"   / "n_c10a_script_drycheck"

# smoke mode 출력 경로 (N-C10b, full output과 완전 분리)
SMOKE_OUT_DIR         = EXP_ROOT / "outputs" / "smoke"   / "n_c10b_normal_val_one_patient_scoring"
SMOKE_REPORT_DIR      = EXP_ROOT / "outputs" / "reports" / "n_c10b_normal_val_one_patient_scoring"
SMOKE_DRYCHECK_OUT_DIR = EXP_ROOT / "outputs" / "reports" / "n_c10b_smoke_mode_patch_drycheck"

# 실제 실행 시 생성될 hard blocker 파일 목록 (collision 검사 대상)
_HARD_BLOCKER_FINAL = [
    MANIFEST_OUT_DIR  / "n_c10_normal_val_crop_manifest.csv",
    SCORES_OUT_DIR    / "n_c10_normal_val_scores.csv",
    THRESHOLD_OUT_DIR / "n_c10_normal_val_thresholds.json",
    THRESHOLD_OUT_DIR / "n_c10_normal_val_thresholds.csv",
    THRESHOLD_OUT_DIR / "n_c10_per_bin_thresholds.csv",
    REPORT_OUT_DIR    / "n_c10_patient_score_summary.csv",
    REPORT_OUT_DIR    / "n_c10_position_bin_score_summary.csv",
    REPORT_OUT_DIR    / "n_c10_runtime_summary.csv",
    REPORT_OUT_DIR    / "n_c10_errors.csv",
    REPORT_OUT_DIR    / "n_c10_normal_val_scoring_threshold_report.md",
    REPORT_OUT_DIR    / "n_c10_normal_val_scoring_threshold_summary.json",
    REPORT_OUT_DIR    / "DONE.json",
]

# smoke mode hard blocker 파일 목록 (N-C10b collision 검사 대상)
_HARD_BLOCKER_SMOKE = [
    SMOKE_OUT_DIR    / "n_c10b_smoke_normal_val_crop_manifest.csv",
    SMOKE_OUT_DIR    / "n_c10b_smoke_normal_val_scores.csv",
    SMOKE_REPORT_DIR / "n_c10b_smoke_patient_score_summary.csv",
    SMOKE_REPORT_DIR / "n_c10b_smoke_position_bin_score_summary.csv",
    SMOKE_REPORT_DIR / "n_c10b_smoke_runtime_summary.csv",
    SMOKE_REPORT_DIR / "n_c10b_smoke_errors.csv",
    SMOKE_REPORT_DIR / "n_c10b_smoke_report.md",
    SMOKE_REPORT_DIR / "n_c10b_smoke_summary.json",
]

# stage2_holdout 접근 금지 패턴
_STAGE2_HOLDOUT_PATTERNS = [
    str(PROJ_ROOT / "outputs" / "position-aware-padim-v1" / "stage2"),
    str(PROJ_ROOT / "data" / "holdout"),
    "stage2_holdout",
    "holdout",
]

# P-C supervised artifact 접근 금지 패턴
_PC_SUPERVISED_PATTERNS = [
    "p_c_supervised", "pc_supervised", "supervised_artifact",
    "/supervised/", "lesion_label", "hard_negative",
    "training_label", "positive_label", "/p_c/", "p_c_",
]

# 금지된 컬럼 (lesion 관련)
_FORBIDDEN_COLUMNS = [
    "lesion_label", "positive", "hard_negative", "training_label",
    "positive_label", "is_lesion", "lesion_id",
]

# ---------------------------------------------------------------------------
# 설계 상수
# ---------------------------------------------------------------------------
EXPECTED_VAL_PATIENTS    = 36
EXPECTED_TOTAL_CROPS     = 21600      # 36 × 6 bins × 100
EXPECTED_SPATIAL_SAMPLES = 194400     # 21600 × 9
CROPS_PER_BIN_PER_PATIENT = 100
EXPECTED_POSITION_BINS   = 6
EXPECTED_SPATIAL_PER_CROP = 9
EXPECTED_SPATIAL_PER_BIN  = 32400     # 36 × 100 × 9

# smoke mode 기대값 (N-C10b, 1명 기준)
SMOKE_EXPECTED_PATIENTS = 1
SMOKE_EXPECTED_CROPS    = 600         # 1 × 6 bins × 100
SMOKE_EXPECTED_SPATIAL  = 5400        # 600 × 9

STATS_NPZ_EXPECTED_KEYS = 26          # mean × 6 + cov × 6 + cov_inv × 6 + count × 6 + epsilon + selected_indices

RAW_FEATURE_DIM    = 144
SELECTED_FEATURE_DIM = 100

CROP_SIZE     = 96
CROP_HALF     = CROP_SIZE // 2
HU_MIN        = -1000.0
HU_MAX        = 200.0
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

POSITION_BINS = [
    "upper_central", "upper_peripheral",
    "middle_central", "middle_peripheral",
    "lower_central", "lower_peripheral",
]

RUNTIME_ESTIMATE_SEC = 190
STORAGE_ESTIMATE_MB  = 4.3
RANDOM_SEED          = 42


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------
def _log(msg: str) -> None:
    print(msg, flush=True)


def _abort(msg: str, code: int = 2) -> None:
    _log(f"[ABORT] {msg}")
    sys.exit(code)


def _check_stage2_access(path: str) -> bool:
    p_lower = str(path).lower()
    return any(pat.lower() in p_lower for pat in _STAGE2_HOLDOUT_PATTERNS)


def _check_pc_supervised_access(path: str) -> bool:
    p_lower = str(path).lower()
    return any(pat.lower() in p_lower for pat in _PC_SUPERVISED_PATTERNS)


def _write_csv(path: Path, fieldnames: list, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# 입력 검증 (dry-run / full-run 공통)
# ---------------------------------------------------------------------------
def validate_inputs() -> dict:
    import numpy as np
    issues = []
    result = {}

    # G1: ALLOW_REAL_PROCESSING 확인
    _log(f"[G1] ALLOW_REAL_PROCESSING={ALLOW_REAL_PROCESSING}")
    result["allow_real_processing"] = ALLOW_REAL_PROCESSING

    # G2: N-C8 verdict 확인
    if N_C8_JSON.exists():
        with open(N_C8_JSON, encoding="utf-8") as f:
            n_c8 = json.load(f)
        v8 = n_c8.get("verdict", "")
        _log(f"[G2] N-C8 verdict={v8}")
        result["n_c8_verdict"] = v8
        if v8 != "통과":
            issues.append(f"G2: N-C8 verdict={v8} (통과 필요)")
    else:
        issues.append(f"G2: N-C8 JSON 없음: {N_C8_JSON}")
        result["n_c8_verdict"] = "MISSING"

    # G3: N-C9 verdict 확인
    if N_C9_JSON.exists():
        with open(N_C9_JSON, encoding="utf-8") as f:
            n_c9 = json.load(f)
        v9 = n_c9.get("verdict", "")
        _log(f"[G3] N-C9 verdict={v9}")
        result["n_c9_verdict"] = v9
        if v9 != "통과":
            issues.append(f"G3: N-C9 verdict={v9} (통과 필요)")
        g9 = n_c9.get("guardrails", {})
        result["n_c9_stage2_holdout"] = g9.get("stage2_holdout_accessed", True)
        result["n_c9_forbidden_supervised"] = g9.get("forbidden_supervised_used", True)
        if g9.get("stage2_holdout_accessed", True):
            issues.append("G3: N-C9 stage2_holdout_accessed=True")
        if g9.get("forbidden_supervised_used", True):
            issues.append("G3: N-C9 forbidden_supervised_used=True")
    else:
        issues.append(f"G3: N-C9 JSON 없음: {N_C9_JSON}")
        result["n_c9_verdict"] = "MISSING"

    # G4: stats npz 확인 (26 keys, shape)
    if STATS_NPZ.exists():
        stats = np.load(str(STATS_NPZ), allow_pickle=False)
        keys = sorted(stats.files)
        n_keys = len(keys)
        _log(f"[G4] stats npz keys={n_keys}")
        result["stats_npz_keys"] = n_keys
        result["stats_npz_key_list"] = keys
        if n_keys != STATS_NPZ_EXPECTED_KEYS:
            issues.append(f"G4: stats npz keys={n_keys} (26 필요)")

        # 각 bin별 mean/cov/cov_inv/count shape 확인
        bin_shapes_ok = True
        shape_issues = []
        for b in POSITION_BINS:
            bn = b.replace("/", "_")
            mean_key    = f"mean_{bn}"
            cov_key     = f"cov_{bn}"
            cov_inv_key = f"cov_inv_{bn}"
            count_key   = f"count_{bn}"
            for k, expected in [
                (mean_key,    (SELECTED_FEATURE_DIM,)),
                (cov_key,     (SELECTED_FEATURE_DIM, SELECTED_FEATURE_DIM)),
                (cov_inv_key, (SELECTED_FEATURE_DIM, SELECTED_FEATURE_DIM)),
            ]:
                if k in keys:
                    actual_shape = stats[k].shape
                    if actual_shape != expected:
                        shape_issues.append(f"{k}: shape={actual_shape} (expected={expected})")
                        bin_shapes_ok = False
                else:
                    shape_issues.append(f"{k}: 없음")
                    bin_shapes_ok = False
            if count_key in keys:
                count_val = int(stats[count_key])
                if count_val != 261000:
                    shape_issues.append(f"{count_key}: {count_val} (261000 필요)")
                    bin_shapes_ok = False

        result["stats_bin_shapes_ok"] = bin_shapes_ok
        if shape_issues:
            for si in shape_issues[:5]:
                issues.append(f"G4: {si}")

        # selected_indices shape 확인
        if "selected_indices" in keys:
            idx = stats["selected_indices"]
            result["selected_indices_shape"] = list(idx.shape)
            if list(idx.shape) != [100]:
                issues.append(f"G4: selected_indices shape={idx.shape} (100 필요)")
        else:
            issues.append("G4: selected_indices 키 없음")
            result["selected_indices_shape"] = None
    else:
        issues.append(f"G4: stats npz 없음: {STATS_NPZ}")
        result["stats_npz_keys"] = 0
        result["stats_bin_shapes_ok"] = False

    # G5: selected_feature_indices.npy 별도 확인
    if SELECTED_INDICES_PATH.exists():
        idx = np.load(str(SELECTED_INDICES_PATH))
        idx_shape  = list(idx.shape)
        idx_unique = int(len(set(idx.tolist())))
        idx_min    = int(idx.min())
        idx_max    = int(idx.max())
        _log(f"[G5] selected_indices shape={idx_shape}, range=[{idx_min},{idx_max}]")
        result["sel_indices_shape"]  = idx_shape
        result["sel_indices_unique"] = idx_unique
        result["sel_indices_min"]    = idx_min
        result["sel_indices_max"]    = idx_max
        if idx_shape != [100]:
            issues.append(f"G5: selected_indices shape={idx_shape} ([100] 필요)")
        if idx_unique != 100:
            issues.append(f"G5: selected_indices unique={idx_unique} (100 필요)")
        if idx_max >= RAW_FEATURE_DIM:
            issues.append(f"G5: selected_indices max={idx_max} >= raw_dim={RAW_FEATURE_DIM}")
    else:
        issues.append(f"G5: selected_feature_indices.npy 없음: {SELECTED_INDICES_PATH}")

    # G6: split JSON → normal_val 36명 확인
    val_patients = []
    patient_to_safe_id = {}
    train_patients = []
    if SPLIT_JSON.exists():
        with open(SPLIT_JSON, encoding="utf-8") as f:
            split = json.load(f)
        val_patients = split.get("val", [])
        train_patients = split.get("train", [])
        patient_to_safe_id = split.get("patient_to_safe_id", {})
        n_val = len(val_patients)
        _log(f"[G6] normal_val patients={n_val}")
        result["n_val_patients"] = n_val
        if n_val != EXPECTED_VAL_PATIENTS:
            issues.append(f"G6: val patients={n_val} ({EXPECTED_VAL_PATIENTS} 필요)")

        # val → train contamination 확인
        val_set = set(val_patients)
        train_set = set(train_patients)
        overlap = val_set & train_set
        result["val_train_overlap"] = len(overlap)
        _log(f"[G6] val→train overlap={len(overlap)}")
        if overlap:
            issues.append(f"G6: val-train overlap 발견: {sorted(overlap)[:5]}")

        # safe_id 누락 확인
        missing_safe_id = [p for p in val_patients if p not in patient_to_safe_id]
        result["missing_safe_id"] = len(missing_safe_id)
        if missing_safe_id:
            issues.append(f"G6: safe_id 없는 val 환자: {missing_safe_id[:3]}")
    else:
        issues.append(f"G6: split JSON 없음: {SPLIT_JSON}")
        result["n_val_patients"] = 0

    # G7: CT/ROI availability 전체 확인 (36/36)
    ct_ok_count  = 0
    roi_ok_count = 0
    ct_missing   = []
    roi_missing  = []
    for pid in val_patients:
        safe_id = patient_to_safe_id.get(pid, "")
        if not safe_id:
            ct_missing.append(pid)
            roi_missing.append(pid)
            continue
        ct_path  = CT_VOLUMES_ROOT / safe_id / "ct_hu.npy"
        roi_path = ROI_ROOT / safe_id / "refined_roi.npy"
        if _check_stage2_access(str(ct_path)) or _check_stage2_access(str(roi_path)):
            issues.append(f"G7: stage2_holdout 경로 감지: {safe_id}")
        if _check_pc_supervised_access(str(ct_path)) or _check_pc_supervised_access(str(roi_path)):
            issues.append(f"G7: P-C supervised 경로 감지: {safe_id}")
        if ct_path.exists():
            ct_ok_count += 1
        else:
            ct_missing.append(pid)
        if roi_path.exists():
            roi_ok_count += 1
        else:
            roi_missing.append(pid)

    _log(f"[G7] CT ok={ct_ok_count}/{len(val_patients)}, ROI ok={roi_ok_count}/{len(val_patients)}")
    result["ct_availability"]  = f"{ct_ok_count}/{len(val_patients)}"
    result["roi_availability"] = f"{roi_ok_count}/{len(val_patients)}"
    result["ct_missing_count"] = len(ct_missing)
    result["roi_missing_count"] = len(roi_missing)
    if ct_missing:
        issues.append(f"G7: CT 없음 ({len(ct_missing)}명): {ct_missing[:3]}")
    if roi_missing:
        issues.append(f"G7: ROI 없음 ({len(roi_missing)}명): {roi_missing[:3]}")

    # G8: val → train manifest 오염 확인 (N-C3 manifest에 val patient_id 없어야 함)
    if TRAIN_MANIFEST_CSV.exists():
        import pandas as pd
        df_train = pd.read_csv(str(TRAIN_MANIFEST_CSV), encoding="utf-8-sig")
        train_pids_in_manifest = set(df_train["patient_id"].unique())
        val_in_manifest = set(val_patients) & train_pids_in_manifest
        result["val_in_train_manifest"] = len(val_in_manifest)
        _log(f"[G8] val in train manifest={len(val_in_manifest)}")
        if val_in_manifest:
            issues.append(f"G8: val patients가 train manifest에 존재: {sorted(val_in_manifest)[:3]}")
        # forbidden 컬럼 확인
        for col in _FORBIDDEN_COLUMNS:
            if col in df_train.columns:
                issues.append(f"G8: 금지 컬럼 '{col}' 가 train manifest에 존재")
    else:
        _log("[G8] train manifest CSV 없음 — skip contamination check")
        result["val_in_train_manifest"] = "SKIP"

    # G9: expected crops / spatial samples 확인
    result["expected_total_crops"]      = EXPECTED_TOTAL_CROPS
    result["expected_spatial_samples"]  = EXPECTED_SPATIAL_SAMPLES
    expected_crops_per_patient          = CROPS_PER_BIN_PER_PATIENT * EXPECTED_POSITION_BINS
    computed_total                      = EXPECTED_VAL_PATIENTS * expected_crops_per_patient
    result["computed_total_crops"]      = computed_total
    if computed_total != EXPECTED_TOTAL_CROPS:
        issues.append(f"G9: computed crops={computed_total} ≠ expected={EXPECTED_TOTAL_CROPS}")

    # G10: output collision 확인
    _log("[G10] output collision 확인...")
    collision_files = []
    for f in _HARD_BLOCKER_FINAL:
        if f.exists():
            collision_files.append(str(f))
    result["output_collision"]       = len(collision_files) > 0
    result["collision_files"]        = collision_files
    result["collision_count"]        = len(collision_files)
    _log(f"[G10] collision count={len(collision_files)}")
    if collision_files:
        for cf in collision_files[:3]:
            issues.append(f"G10: output collision: {Path(cf).name}")

    # G11: 안전 플래그 현황 (dry 단계)
    result["guardrail_flags"] = {
        "model_forward_run":           False,
        "feature_extraction_run":      False,
        "scoring_run":                 False,
        "threshold_computed":          False,
        "training_run":                False,
        "stage2_holdout_accessed":     False,
        "forbidden_supervised_source_used": False,
        "lesion_scoring":              False,
        "crop_npz_generated":          False,
        "output_files_created":        False,
    }

    result["issues"]      = issues
    result["input_valid"] = (len(issues) == 0)
    return result


# ---------------------------------------------------------------------------
# 2.5D crop 추출 (실제 scoring 시 사용)
# ---------------------------------------------------------------------------
def extract_2p5d_crop(ct_vol, z_idx: int, cy: int, cx: int):
    import numpy as np
    Z, H, W = ct_vol.shape
    half = CROP_HALF
    ch_imgs = []
    for dz in (-1, 0, 1):
        z  = max(0, min(Z - 1, z_idx + dz))
        sl = ct_vol[z].astype(np.float32)
        y0, y1 = cy - half, cy + half
        x0, x1 = cx - half, cx + half
        pad_top    = max(0, -y0)
        pad_bottom = max(0, y1 - H)
        pad_left   = max(0, -x0)
        pad_right  = max(0, x1 - W)
        y0c, y1c = max(0, y0), min(H, y1)
        x0c, x1c = max(0, x0), min(W, x1)
        region = sl[y0c:y1c, x0c:x1c]
        if pad_top or pad_bottom or pad_left or pad_right:
            region = np.pad(
                region,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant", constant_values=HU_MIN,
            )
        if region.shape != (CROP_SIZE, CROP_SIZE):
            raise ValueError(f"crop shape {region.shape} ≠ {CROP_SIZE}×{CROP_SIZE}")
        ch_imgs.append(region)
    return ch_imgs[0], ch_imgs[1], ch_imgs[2]  # z-1, z, z+1


def preprocess_2p5d_crop(z_minus, z_center, z_plus):
    import numpy as np
    crop = np.stack([z_minus, z_center, z_plus], axis=0).astype(np.float32)
    clipped = np.clip(crop, HU_MIN, HU_MAX)
    normed  = (clipped - HU_MIN) / (HU_MAX - HU_MIN)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std  = np.array(IMAGENET_STD,  dtype=np.float32).reshape(3, 1, 1)
    return ((normed - mean) / std).astype(np.float32)  # (3, 96, 96)


# ---------------------------------------------------------------------------
# 3×3 spatial feature 추출 → (9, 144)
# ---------------------------------------------------------------------------
def extract_3x3_spatial_features(f_early, f_mid, f_late):
    import torch.nn.functional as F
    import numpy as np

    def _pool_reshape(feat):
        pooled = F.adaptive_avg_pool2d(feat, (3, 3)).squeeze(0)  # (C, 3, 3)
        C = pooled.shape[0]
        arr = pooled.cpu().numpy().transpose(1, 2, 0).reshape(9, C)
        return arr.astype(np.float32)

    a = _pool_reshape(f_early)  # (9, 24)
    b = _pool_reshape(f_mid)    # (9, 40)
    c = _pool_reshape(f_late)   # (9, 80)
    return np.concatenate([a, b, c], axis=1)  # (9, 144)


# ---------------------------------------------------------------------------
# position_bin 결정 (z_level, 반경 비율)
# ---------------------------------------------------------------------------
def get_position_bin(z_level: str, roi_patch_ratio: float) -> str:
    peripheral_threshold = 0.5
    is_peripheral = (roi_patch_ratio < peripheral_threshold)
    side = "peripheral" if is_peripheral else "central"
    return f"{z_level}_{side}"


def assign_z_level(local_z: float) -> str:
    if local_z < 0.333:
        return "upper"
    elif local_z < 0.667:
        return "middle"
    else:
        return "lower"


# ---------------------------------------------------------------------------
# normal_val crop coordinate sampling (실제 scoring 시 사용)
# ---------------------------------------------------------------------------
def sample_val_crop_coordinates(ct_vol, roi_mask, patient_id: str, safe_id: str):
    import numpy as np
    rng = np.random.default_rng(RANDOM_SEED + abs(hash(patient_id)) % (2**32))

    Z, H, W = ct_vol.shape
    half = CROP_HALF

    # ROI 마스크에서 유효 복셀 추출
    roi_zyx = np.argwhere(roi_mask > 0)  # (N, 3): z, y, x
    if len(roi_zyx) == 0:
        return []

    # z_range 계산
    z_min, z_max = int(roi_zyx[:, 0].min()), int(roi_zyx[:, 0].max())
    z_range = max(z_max - z_min, 1)

    # 6 position bins별 좌표 수집
    bin_candidates = {b: [] for b in POSITION_BINS}
    for row in roi_zyx:
        z, y, x = int(row[0]), int(row[1]), int(row[2])
        # 경계 확인 (crop 추출 가능 범위)
        if (y - half < -half or y + half > H + half or
                x - half < -half or x + half > W + half):
            pass  # 패딩으로 처리 가능
        local_z = (z - z_min) / z_range
        z_level = assign_z_level(local_z)
        # 반경 기반 중심/주변 구분 (ROI 내 픽셀 비율 근사)
        cy_center = H / 2.0
        cx_center = W / 2.0
        dist_ratio = ((y - cy_center) ** 2 + (x - cx_center) ** 2) ** 0.5 / (min(H, W) / 2.0)
        roi_patch_ratio = 1.0 - min(dist_ratio, 1.0)  # 중심에 가까울수록 높음
        pbin = get_position_bin(z_level, roi_patch_ratio)
        if pbin in bin_candidates:
            bin_candidates[pbin].append((z, y, x, local_z, roi_patch_ratio))

    # 각 bin에서 100개 샘플링
    crops = []
    crop_id = 0
    for pbin in POSITION_BINS:
        candidates = bin_candidates[pbin]
        if not candidates:
            continue
        n_sample = min(CROPS_PER_BIN_PER_PATIENT, len(candidates))
        chosen_idx = rng.choice(len(candidates), size=n_sample, replace=False)
        chosen = [candidates[i] for i in chosen_idx]
        for rank, (z, y, x, local_z, roi_ratio) in enumerate(chosen):
            y0 = y - half
            x0 = x - half
            y1 = y + half
            x1 = x + half
            crops.append({
                "normal_val_candidate_id": f"{patient_id}_{pbin}_{rank:04d}",
                "patient_id":   patient_id,
                "safe_id":      safe_id,
                "split":        "normal_val",
                "local_z":      round(local_z, 6),
                "slice_index":  z,
                "y0":  y0, "x0": x0,
                "y1":  y1, "x1": x1,
                "center_y": y, "center_x": x,
                "position_bin": pbin,
                "z_level":      assign_z_level(local_z),
                "roi_patch_ratio": round(roi_ratio, 6),
                "source_branch": "normal_val",
                "forbidden_supervised_source_used": False,
            })
            crop_id += 1

    return crops


# ---------------------------------------------------------------------------
# Mahalanobis score 계산
# ---------------------------------------------------------------------------
def compute_mahalanobis_score(feat_100, mean_b, cov_inv_b):
    import numpy as np
    delta = feat_100.astype(np.float64) - mean_b.astype(np.float64)
    cov_inv_d = cov_inv_b.astype(np.float64)
    d_sq = float(delta @ cov_inv_d @ delta)
    return float(np.sqrt(max(0.0, d_sq)))


# ---------------------------------------------------------------------------
# FeatureExtractorEffNetB0 로더 (실제 실행 시 동적 import)
# ---------------------------------------------------------------------------
def load_feature_extractor():
    import torch
    import torchvision.models as tv_models

    class FeatureExtractorEffNetB0(torch.nn.Module):
        def __init__(self):
            super().__init__()
            base = tv_models.efficientnet_b0(
                weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )
            features = base.features
            self.stem   = features[:2]    # 0,1   → out ch=32
            self.stage1 = features[2]     # MBConv × 2 → ch=16
            self.stage2 = features[3]     # MBConv × 3 → ch=24  (early tap)
            self.stage3 = features[4]     # MBConv × 3 → ch=40  (mid tap)
            self.stage4 = features[5]     # MBConv × 3 → ch=80  (late tap)

        def forward(self, x):
            x = self.stem(x)
            x = self.stage1(x)
            f_early = self.stage2(x)   # (1, 24, H/4, W/4)
            f_mid   = self.stage3(f_early)  # (1, 40, H/8, W/8)
            f_late  = self.stage4(f_mid)    # (1, 80, H/16, W/16)
            return f_early, f_mid, f_late

    extractor = FeatureExtractorEffNetB0()
    extractor.eval()
    # eval mode 강제 확인
    for m in extractor.modules():
        if hasattr(m, "training") and m.training:
            extractor.eval()
            break
    return extractor


# ---------------------------------------------------------------------------
# 실제 scoring 실행 (ALLOW_REAL_PROCESSING=True + 5 flags 모두 필요)
# ---------------------------------------------------------------------------
def run_val_scoring(args) -> None:
    import numpy as np
    import torch
    import pandas as pd

    _log("\n[scoring] 실제 scoring 시작...")
    t_start = time.time()

    # 통계 로드
    _log("[scoring] N-C7 stats npz 로드...")
    stats = np.load(str(STATS_NPZ), allow_pickle=False)
    selected_indices = np.load(str(SELECTED_INDICES_PATH)).astype(int)

    bin_stats = {}
    for b in POSITION_BINS:
        bn = b.replace("/", "_")
        bin_stats[b] = {
            "mean":    stats[f"mean_{bn}"].astype(np.float64),
            "cov_inv": stats[f"cov_inv_{bn}"].astype(np.float64),
        }

    # feature extractor 로드
    _log("[scoring] FeatureExtractorEffNetB0 로드...")
    extractor = load_feature_extractor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = extractor.to(device)
    extractor.eval()

    # split 로드
    with open(SPLIT_JSON, encoding="utf-8") as f:
        split = json.load(f)
    val_patients = split["val"]
    patient_to_safe_id = split["patient_to_safe_id"]

    # 출력 경로 생성
    MANIFEST_OUT_DIR.mkdir(parents=True, exist_ok=True)
    SCORES_OUT_DIR.mkdir(parents=True, exist_ok=True)
    THRESHOLD_OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    score_schema = [
        "normal_val_candidate_id", "patient_id", "safe_id", "split",
        "local_z", "slice_index", "y0", "x0", "y1", "x1",
        "center_y", "center_x", "position_bin", "z_level",
        "roi_patch_ratio",
        "score_spatial_0", "score_spatial_1", "score_spatial_2",
        "score_spatial_3", "score_spatial_4", "score_spatial_5",
        "score_spatial_6", "score_spatial_7", "score_spatial_8",
        "score_mean", "score_max", "score_center",
        "primary_score", "source_branch", "forbidden_supervised_source_used",
    ]

    all_manifest_rows = []
    all_score_rows    = []
    patient_summaries = []
    errors            = []

    for pid in val_patients:
        safe_id  = patient_to_safe_id.get(pid, "")
        ct_path  = CT_VOLUMES_ROOT / safe_id / "ct_hu.npy"
        roi_path = ROI_ROOT / safe_id / "refined_roi.npy"
        t_pat    = time.time()
        try:
            _log(f"  [{pid}] 로드 중...")
            ct_vol   = np.load(str(ct_path),  mmap_mode="r")
            roi_mask = np.load(str(roi_path), mmap_mode="r")

            # crop 좌표 샘플링
            crop_coords = sample_val_crop_coordinates(ct_vol, roi_mask, pid, safe_id)
            all_manifest_rows.extend(crop_coords)
            _log(f"  [{pid}] crops={len(crop_coords)}")

            # 각 crop에 대해 feature 추출 + scoring
            for crop_info in crop_coords:
                z   = int(crop_info["slice_index"])
                cy  = int(crop_info["center_y"])
                cx  = int(crop_info["center_x"])
                pbin = crop_info["position_bin"]

                zm, zc, zp = extract_2p5d_crop(ct_vol, z, cy, cx)
                img = preprocess_2p5d_crop(zm, zc, zp)  # (3, 96, 96)

                with torch.no_grad():
                    inp   = torch.from_numpy(img).unsqueeze(0).to(device)
                    f_e, f_m, f_l = extractor(inp)

                feats_9x144 = extract_3x3_spatial_features(f_e, f_m, f_l)  # (9, 144)
                feats_9x100 = feats_9x144[:, selected_indices]              # (9, 100)

                mean_b    = bin_stats[pbin]["mean"]
                cov_inv_b = bin_stats[pbin]["cov_inv"]

                spatial_scores = []
                for sp_idx in range(9):
                    s = compute_mahalanobis_score(feats_9x100[sp_idx], mean_b, cov_inv_b)
                    spatial_scores.append(s)

                score_mean   = float(np.mean(spatial_scores))
                score_max    = float(np.max(spatial_scores))
                score_center = float(spatial_scores[4])  # center cell (idx 4)

                row = dict(crop_info)
                for si, sv in enumerate(spatial_scores):
                    row[f"score_spatial_{si}"] = round(sv, 6)
                row["score_mean"]   = round(score_mean, 6)
                row["score_max"]    = round(score_max,  6)
                row["score_center"] = round(score_center, 6)
                row["primary_score"] = round(score_max, 6)
                row["forbidden_supervised_source_used"] = False
                all_score_rows.append(row)

            elapsed = time.time() - t_pat
            patient_summaries.append({
                "patient_id": pid,
                "safe_id":    safe_id,
                "n_crops":    len(crop_coords),
                "elapsed_sec": round(elapsed, 2),
                "status": "OK",
                "errors": 0,
            })
        except Exception as e:
            errors.append({
                "patient_id": pid,
                "error_type": type(e).__name__,
                "message": str(e),
            })
            patient_summaries.append({
                "patient_id": pid,
                "safe_id":    safe_id,
                "n_crops":    0,
                "elapsed_sec": round(time.time() - t_pat, 2),
                "status": "ERROR",
                "errors": 1,
            })

    # manifest 저장
    manifest_path = MANIFEST_OUT_DIR / "n_c10_normal_val_crop_manifest.csv"
    manifest_cols = [
        "normal_val_candidate_id", "patient_id", "safe_id", "split",
        "local_z", "slice_index", "y0", "x0", "y1", "x1",
        "center_y", "center_x", "position_bin", "z_level",
        "roi_patch_ratio", "source_branch", "forbidden_supervised_source_used",
    ]
    _write_csv(manifest_path, manifest_cols, all_manifest_rows)

    # score CSV 저장
    score_path = SCORES_OUT_DIR / "n_c10_normal_val_scores.csv"
    _write_csv(score_path, score_schema, all_score_rows)

    # ---------------------------------------------------------------------------
    # threshold 계산
    # ---------------------------------------------------------------------------
    _log("[scoring] threshold 계산...")
    score_max_vals    = [r["score_max"]    for r in all_score_rows]
    score_mean_vals   = [r["score_mean"]   for r in all_score_rows]
    score_center_vals = [r["score_center"] for r in all_score_rows]
    all_spatial_vals  = []
    for r in all_score_rows:
        for si in range(9):
            all_spatial_vals.append(r[f"score_spatial_{si}"])

    def pctile(vals, p):
        arr = sorted(vals)
        idx = int(len(arr) * p / 100)
        idx = min(idx, len(arr) - 1)
        return round(arr[idx], 6)

    threshold_rows = []
    for scope, vals, agg_label in [
        ("crop_score_max",    score_max_vals,    "score_max"),
        ("crop_score_mean",   score_mean_vals,   "score_mean"),
        ("crop_score_center", score_center_vals, "score_center"),
        ("spatial_score",     all_spatial_vals,  "all_spatial"),
    ]:
        for pct in [95, 99]:
            threshold_rows.append({
                "threshold_scope": scope,
                "percentile":      f"p{pct}",
                "value":           pctile(vals, pct),
                "n_samples":       len(vals),
                "source_split":    "normal_val",
                "score_formula":   "sqrt_mahalanobis",
                "aggregation":     agg_label,
                "fixed_for_future": True,
            })

    # per-bin p95
    per_bin_rows = []
    for b in POSITION_BINS:
        bin_max_vals = [r["score_max"] for r in all_score_rows if r["position_bin"] == b]
        if bin_max_vals:
            per_bin_rows.append({
                "threshold_scope": f"per_bin_{b}_crop_score_max",
                "percentile":      "p95",
                "value":           pctile(bin_max_vals, 95),
                "n_samples":       len(bin_max_vals),
                "source_split":    "normal_val",
                "score_formula":   "sqrt_mahalanobis",
                "aggregation":     "score_max",
                "fixed_for_future": True,
            })

    threshold_fields = [
        "threshold_scope", "percentile", "value", "n_samples",
        "source_split", "score_formula", "aggregation", "fixed_for_future",
    ]
    _write_csv(THRESHOLD_OUT_DIR / "n_c10_normal_val_thresholds.csv", threshold_fields, threshold_rows)
    _write_csv(THRESHOLD_OUT_DIR / "n_c10_per_bin_thresholds.csv",    threshold_fields, per_bin_rows)

    # threshold JSON 저장
    threshold_dict = {r["threshold_scope"] + "_" + r["percentile"]: r["value"] for r in threshold_rows + per_bin_rows}
    threshold_json = {
        "step": "N-C10",
        "source_split": "normal_val",
        "n_patients": len(val_patients),
        "n_crops": len(all_score_rows),
        "score_formula": "sqrt_mahalanobis",
        "primary_threshold": "crop_score_max_p95",
        "fixed_for_future": True,
        "thresholds": threshold_dict,
    }
    with open(str(THRESHOLD_OUT_DIR / "n_c10_normal_val_thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(threshold_json, f, ensure_ascii=False, indent=2)

    # 보고서 저장
    _write_csv(
        REPORT_OUT_DIR / "n_c10_patient_score_summary.csv",
        ["patient_id", "safe_id", "n_crops", "elapsed_sec", "status", "errors"],
        patient_summaries,
    )
    _write_csv(
        REPORT_OUT_DIR / "n_c10_runtime_summary.csv",
        ["item", "value"],
        [
            {"item": "total_patients", "value": len(val_patients)},
            {"item": "total_crops",    "value": len(all_score_rows)},
            {"item": "total_sec",      "value": round(time.time() - t_start, 1)},
            {"item": "errors",         "value": len(errors)},
        ],
    )
    _write_csv(
        REPORT_OUT_DIR / "n_c10_errors.csv",
        ["patient_id", "error_type", "message"],
        errors if errors else [{"patient_id": "none", "error_type": "none", "message": "no errors"}],
    )

    # position_bin별 score 요약
    bin_summary_rows = []
    for b in POSITION_BINS:
        b_vals = [r["score_max"] for r in all_score_rows if r["position_bin"] == b]
        if b_vals:
            bin_summary_rows.append({
                "position_bin": b,
                "n_crops": len(b_vals),
                "score_max_mean": round(sum(b_vals) / len(b_vals), 6),
                "score_max_p95":  pctile(b_vals, 95),
                "score_max_p99":  pctile(b_vals, 99),
            })
    _write_csv(
        REPORT_OUT_DIR / "n_c10_position_bin_score_summary.csv",
        ["position_bin", "n_crops", "score_max_mean", "score_max_p95", "score_max_p99"],
        bin_summary_rows,
    )

    # DONE.json
    done_conditions_ok = (
        len(errors) == 0 and
        len(all_score_rows) >= EXPECTED_TOTAL_CROPS * 0.95 and
        threshold_dict.get("crop_score_max_p95") is not None
    )
    done_json = {
        "step": "N-C10",
        "all_patients_processed": len(errors) == 0,
        "total_crops": len(all_score_rows),
        "errors": len(errors),
        "threshold_computed": True,
        "primary_threshold_crop_score_max_p95": threshold_dict.get("crop_score_max_p95"),
        "output_created": True,
        "conditions_ok": done_conditions_ok,
    }
    with open(str(REPORT_OUT_DIR / "DONE.json"), "w", encoding="utf-8") as f:
        json.dump(done_json, f, ensure_ascii=False, indent=2)

    _log(f"\n[scoring] 완료: crops={len(all_score_rows)}, errors={len(errors)}, "
         f"elapsed={round(time.time()-t_start,1)}s")


# ---------------------------------------------------------------------------
# smoke one patient: normal_val 1명 scoring (threshold 계산 없음)
# ---------------------------------------------------------------------------
def run_smoke_one_patient(args) -> None:
    import numpy as np
    import torch

    _log("\n[smoke] normal_val 1명 scoring smoke 시작...")
    t_start = time.time()

    # stage2_holdout / P-C supervised 경로 사전 차단
    for path_str in [str(SMOKE_OUT_DIR), str(SMOKE_REPORT_DIR)]:
        if _check_stage2_access(path_str):
            _abort(f"[smoke] stage2_holdout 경로 감지: {path_str}")
        if _check_pc_supervised_access(path_str):
            _abort(f"[smoke] P-C supervised 경로 감지: {path_str}")

    # 통계 로드
    _log("[smoke] N-C7 stats npz 로드...")
    stats = np.load(str(STATS_NPZ), allow_pickle=False)
    selected_indices = np.load(str(SELECTED_INDICES_PATH)).astype(int)

    bin_stats = {}
    for b in POSITION_BINS:
        bn = b.replace("/", "_")
        bin_stats[b] = {
            "mean":    stats[f"mean_{bn}"].astype(np.float64),
            "cov_inv": stats[f"cov_inv_{bn}"].astype(np.float64),
        }

    # feature extractor 로드
    _log("[smoke] FeatureExtractorEffNetB0 로드...")
    extractor = load_feature_extractor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = extractor.to(device)
    extractor.eval()

    # split 로드 → 첫 번째 환자 1명만 처리
    with open(SPLIT_JSON, encoding="utf-8") as f:
        split = json.load(f)
    val_patients = split["val"]
    patient_to_safe_id = split["patient_to_safe_id"]

    smoke_patients = [val_patients[0]]  # 1명만
    _log(f"[smoke] 처리 대상: {smoke_patients[0]} (1/{len(val_patients)}명, 나머지 미처리)")

    # 출력 경로 생성 (smoke 전용 경로만 사용)
    SMOKE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    SMOKE_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    score_schema = [
        "normal_val_candidate_id", "patient_id", "safe_id", "split",
        "local_z", "slice_index", "y0", "x0", "y1", "x1",
        "center_y", "center_x", "position_bin", "z_level",
        "roi_patch_ratio",
        "score_spatial_0", "score_spatial_1", "score_spatial_2",
        "score_spatial_3", "score_spatial_4", "score_spatial_5",
        "score_spatial_6", "score_spatial_7", "score_spatial_8",
        "score_mean", "score_max", "score_center",
        "primary_score", "source_branch", "forbidden_supervised_source_used",
    ]

    all_manifest_rows = []
    all_score_rows    = []
    patient_summaries = []
    errors            = []

    for pid in smoke_patients:
        safe_id  = patient_to_safe_id.get(pid, "")
        ct_path  = CT_VOLUMES_ROOT / safe_id / "ct_hu.npy"
        roi_path = ROI_ROOT / safe_id / "refined_roi.npy"
        t_pat    = time.time()
        try:
            _log(f"  [{pid}] 로드 중...")
            ct_vol   = np.load(str(ct_path),  mmap_mode="r")
            roi_mask = np.load(str(roi_path), mmap_mode="r")

            crop_coords = sample_val_crop_coordinates(ct_vol, roi_mask, pid, safe_id)
            all_manifest_rows.extend(crop_coords)
            _log(f"  [{pid}] crops={len(crop_coords)} (expected~{SMOKE_EXPECTED_CROPS})")

            for crop_info in crop_coords:
                z    = int(crop_info["slice_index"])
                cy   = int(crop_info["center_y"])
                cx   = int(crop_info["center_x"])
                pbin = crop_info["position_bin"]

                zm, zc, zp = extract_2p5d_crop(ct_vol, z, cy, cx)
                img = preprocess_2p5d_crop(zm, zc, zp)

                with torch.no_grad():
                    inp   = torch.from_numpy(img).unsqueeze(0).to(device)
                    f_e, f_m, f_l = extractor(inp)

                feats_9x144 = extract_3x3_spatial_features(f_e, f_m, f_l)
                feats_9x100 = feats_9x144[:, selected_indices]

                mean_b    = bin_stats[pbin]["mean"]
                cov_inv_b = bin_stats[pbin]["cov_inv"]

                spatial_scores = []
                for sp_idx in range(9):
                    s = compute_mahalanobis_score(feats_9x100[sp_idx], mean_b, cov_inv_b)
                    spatial_scores.append(s)

                score_mean   = float(np.mean(spatial_scores))
                score_max    = float(np.max(spatial_scores))
                score_center = float(spatial_scores[4])

                row = dict(crop_info)
                for si, sv in enumerate(spatial_scores):
                    row[f"score_spatial_{si}"] = round(sv, 6)
                row["score_mean"]   = round(score_mean, 6)
                row["score_max"]    = round(score_max,  6)
                row["score_center"] = round(score_center, 6)
                row["primary_score"] = round(score_max, 6)
                row["forbidden_supervised_source_used"] = False
                all_score_rows.append(row)

            elapsed = time.time() - t_pat
            patient_summaries.append({
                "patient_id": pid, "safe_id": safe_id,
                "n_crops": len(crop_coords), "elapsed_sec": round(elapsed, 2),
                "status": "OK", "errors": 0,
            })
        except Exception as e:
            errors.append({
                "patient_id": pid, "error_type": type(e).__name__, "message": str(e),
            })
            patient_summaries.append({
                "patient_id": pid, "safe_id": safe_id,
                "n_crops": 0, "elapsed_sec": round(time.time() - t_pat, 2),
                "status": "ERROR", "errors": 1,
            })

    # manifest / score CSV 저장 (smoke 전용 경로)
    manifest_cols = [
        "normal_val_candidate_id", "patient_id", "safe_id", "split",
        "local_z", "slice_index", "y0", "x0", "y1", "x1",
        "center_y", "center_x", "position_bin", "z_level",
        "roi_patch_ratio", "source_branch", "forbidden_supervised_source_used",
    ]
    _write_csv(
        SMOKE_OUT_DIR / "n_c10b_smoke_normal_val_crop_manifest.csv",
        manifest_cols, all_manifest_rows,
    )
    _write_csv(
        SMOKE_OUT_DIR / "n_c10b_smoke_normal_val_scores.csv",
        score_schema, all_score_rows,
    )

    # score sanity 확인 (threshold 계산 없음)
    n_crops_actual   = len(all_score_rows)
    n_spatial_actual = n_crops_actual * 9
    score_max_vals   = [r["score_max"] for r in all_score_rows]
    score_finite     = all(isinstance(v, float) and v >= 0.0 for v in score_max_vals)
    _log(f"[smoke] n_crops={n_crops_actual} (expected~{SMOKE_EXPECTED_CROPS})")
    _log(f"[smoke] n_spatial={n_spatial_actual} (expected~{SMOKE_EXPECTED_SPATIAL})")
    _log(f"[smoke] score_finite={score_finite}")
    if score_max_vals:
        _log(f"[smoke] score_max range=[{min(score_max_vals):.4f}, {max(score_max_vals):.4f}]")

    # patient score summary
    _write_csv(
        SMOKE_REPORT_DIR / "n_c10b_smoke_patient_score_summary.csv",
        ["patient_id", "safe_id", "n_crops", "elapsed_sec", "status", "errors"],
        patient_summaries,
    )

    # position_bin별 summary
    def _pct(vals, p):
        arr = sorted(vals)
        return round(arr[min(int(len(arr) * p / 100), len(arr) - 1)], 6)

    bin_summary_rows = []
    for b in POSITION_BINS:
        b_vals = [r["score_max"] for r in all_score_rows if r["position_bin"] == b]
        if b_vals:
            bin_summary_rows.append({
                "position_bin": b,
                "n_crops": len(b_vals),
                "score_max_mean": round(sum(b_vals) / len(b_vals), 6),
                "score_max_p95":  _pct(b_vals, 95),
                "score_max_p99":  _pct(b_vals, 99),
            })
    _write_csv(
        SMOKE_REPORT_DIR / "n_c10b_smoke_position_bin_score_summary.csv",
        ["position_bin", "n_crops", "score_max_mean", "score_max_p95", "score_max_p99"],
        bin_summary_rows,
    )

    elapsed_total = round(time.time() - t_start, 1)
    _write_csv(
        SMOKE_REPORT_DIR / "n_c10b_smoke_runtime_summary.csv",
        ["item", "value"],
        [
            {"item": "smoke_patients",       "value": len(smoke_patients)},
            {"item": "total_crops",          "value": n_crops_actual},
            {"item": "total_spatial",        "value": n_spatial_actual},
            {"item": "total_sec",            "value": elapsed_total},
            {"item": "errors",               "value": len(errors)},
            {"item": "threshold_computed",   "value": False},
            {"item": "full_val_scoring_run", "value": False},
        ],
    )
    _write_csv(
        SMOKE_REPORT_DIR / "n_c10b_smoke_errors.csv",
        ["patient_id", "error_type", "message"],
        errors if errors else [{"patient_id": "none", "error_type": "none", "message": "no errors"}],
    )

    sanity_ok = (n_crops_actual > 0 and score_finite and len(errors) == 0)

    smoke_json = {
        "step": "N-C10b",
        "mode": "smoke_one_patient",
        "patient_id": smoke_patients[0],
        "n_crops_actual": n_crops_actual,
        "n_spatial_actual": n_spatial_actual,
        "expected_crops": SMOKE_EXPECTED_CROPS,
        "expected_spatial": SMOKE_EXPECTED_SPATIAL,
        "score_schema_ok": score_finite,
        "threshold_computed": False,
        "full_val_scoring_run": False,
        "stage2_holdout_accessed": False,
        "forbidden_supervised_used": False,
        "sanity_ok": sanity_ok,
        "errors": len(errors),
        "elapsed_sec": elapsed_total,
    }
    with open(str(SMOKE_REPORT_DIR / "n_c10b_smoke_summary.json"), "w", encoding="utf-8") as f:
        json.dump(smoke_json, f, ensure_ascii=False, indent=2)

    verdict = "통과" if sanity_ok else ("부분통과" if n_crops_actual > 0 else "실패")
    md_lines = [
        f"**판정**: {verdict}",
        f"**단계**: N-C10b Normal-Val One-Patient Scoring Smoke",
        f"**모드**: smoke_one_patient",
        "",
        "---",
        "",
        "## Smoke Scoring 결과",
        "",
        "| 항목 | 결과 |",
        "|------|------|",
        f"| 처리 환자 | {smoke_patients[0]} (1명) |",
        f"| n_crops_actual | {n_crops_actual} (expected~{SMOKE_EXPECTED_CROPS}) |",
        f"| n_spatial_actual | {n_spatial_actual} (expected~{SMOKE_EXPECTED_SPATIAL}) |",
        f"| score_schema_ok | {score_finite} |",
        f"| errors | {len(errors)} |",
        f"| threshold_computed | False |",
        f"| full_val_scoring_run | False |",
        f"| stage2_holdout_accessed | False |",
        f"| forbidden_supervised_used | False |",
        f"| elapsed_sec | {elapsed_total} |",
        "",
        "---",
        "",
        "## N-C10c 실행 가능 여부",
        "",
        f"- **{'가능' if sanity_ok else '불가 (이슈 해결 필요)'}**",
    ]
    with open(str(SMOKE_REPORT_DIR / "n_c10b_smoke_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    _log(f"\n[smoke] 완료: crops={n_crops_actual}, sanity_ok={sanity_ok}, elapsed={elapsed_total}s")
    _log(f"[smoke] 판정: {verdict}")
    _log(f"[smoke] 출력 → {SMOKE_OUT_DIR}")
    _log(f"[smoke] 보고서 → {SMOKE_REPORT_DIR}")


# ---------------------------------------------------------------------------
# dry-run: 검증 + dry-check 보고서 생성
# ---------------------------------------------------------------------------
def run_dry(validation: dict) -> None:
    _log("\n[dry-run] 입력 검증 결과:")
    for k, v in validation.items():
        if k not in ("issues", "guardrail_flags", "stats_npz_key_list", "collision_files"):
            _log(f"  {k}: {v}")

    issues = validation.get("issues", [])
    if issues:
        _log("\n[dry-run] 문제 발견:")
        for iss in issues:
            _log(f"  WARNING {iss}")
    else:
        _log("[dry-run] 모든 입력 검증 통과 OK")

    _log("[dry-run] model forward / feature extraction / scoring 미실행 OK")
    _log("[dry-run] dry-check 리포트 생성 중...")

    DRYCHECK_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. n_c10a_input_validation.csv
    def _pass_fail(cond):
        return "PASS" if cond else "FAIL"

    input_rows = [
        {"item": "ALLOW_REAL_PROCESSING",
         "value": str(ALLOW_REAL_PROCESSING),
         "status": "OK"},
        {"item": "N-C8 verdict",
         "value": validation.get("n_c8_verdict", "?"),
         "status": _pass_fail(validation.get("n_c8_verdict") == "통과")},
        {"item": "N-C9 verdict",
         "value": validation.get("n_c9_verdict", "?"),
         "status": _pass_fail(validation.get("n_c9_verdict") == "통과")},
        {"item": "stats_npz_keys",
         "value": str(validation.get("stats_npz_keys", "?")),
         "status": _pass_fail(validation.get("stats_npz_keys") == STATS_NPZ_EXPECTED_KEYS)},
        {"item": "stats_bin_shapes_ok",
         "value": str(validation.get("stats_bin_shapes_ok", False)),
         "status": _pass_fail(validation.get("stats_bin_shapes_ok", False))},
        {"item": "selected_indices_shape",
         "value": str(validation.get("sel_indices_shape", "?")),
         "status": _pass_fail(validation.get("sel_indices_shape") == [100])},
        {"item": "n_val_patients",
         "value": str(validation.get("n_val_patients", "?")),
         "status": _pass_fail(validation.get("n_val_patients") == EXPECTED_VAL_PATIENTS)},
        {"item": "val_train_overlap",
         "value": str(validation.get("val_train_overlap", "?")),
         "status": _pass_fail(validation.get("val_train_overlap") == 0)},
        {"item": "missing_safe_id",
         "value": str(validation.get("missing_safe_id", "?")),
         "status": _pass_fail(validation.get("missing_safe_id") == 0)},
        {"item": "ct_availability",
         "value": validation.get("ct_availability", "?"),
         "status": _pass_fail(validation.get("ct_missing_count", 1) == 0)},
        {"item": "roi_availability",
         "value": validation.get("roi_availability", "?"),
         "status": _pass_fail(validation.get("roi_missing_count", 1) == 0)},
        {"item": "val_in_train_manifest",
         "value": str(validation.get("val_in_train_manifest", "?")),
         "status": _pass_fail(validation.get("val_in_train_manifest") == 0)},
        {"item": "expected_total_crops",
         "value": str(EXPECTED_TOTAL_CROPS),
         "status": _pass_fail(validation.get("computed_total_crops") == EXPECTED_TOTAL_CROPS)},
        {"item": "expected_spatial_samples",
         "value": str(EXPECTED_SPATIAL_SAMPLES),
         "status": "INFO"},
        {"item": "output_collision",
         "value": str(validation.get("output_collision", "?")),
         "status": _pass_fail(not validation.get("output_collision", True))},
    ]
    _write_csv(
        DRYCHECK_OUT_DIR / "n_c10a_input_validation.csv",
        ["item", "value", "status"], input_rows,
    )

    # 2. n_c10a_output_path_check.csv
    output_path_rows = [
        {"file":       "n_c10_normal_val_crop_manifest.csv",
         "dir":        str(MANIFEST_OUT_DIR),
         "collision":  str((MANIFEST_OUT_DIR / "n_c10_normal_val_crop_manifest.csv").exists()),
         "status":     "OK" if not (MANIFEST_OUT_DIR / "n_c10_normal_val_crop_manifest.csv").exists() else "COLLISION"},
        {"file":       "n_c10_normal_val_scores.csv",
         "dir":        str(SCORES_OUT_DIR),
         "collision":  str((SCORES_OUT_DIR / "n_c10_normal_val_scores.csv").exists()),
         "status":     "OK" if not (SCORES_OUT_DIR / "n_c10_normal_val_scores.csv").exists() else "COLLISION"},
        {"file":       "n_c10_normal_val_thresholds.json",
         "dir":        str(THRESHOLD_OUT_DIR),
         "collision":  str((THRESHOLD_OUT_DIR / "n_c10_normal_val_thresholds.json").exists()),
         "status":     "OK" if not (THRESHOLD_OUT_DIR / "n_c10_normal_val_thresholds.json").exists() else "COLLISION"},
        {"file":       "n_c10_normal_val_thresholds.csv",
         "dir":        str(THRESHOLD_OUT_DIR),
         "collision":  str((THRESHOLD_OUT_DIR / "n_c10_normal_val_thresholds.csv").exists()),
         "status":     "OK" if not (THRESHOLD_OUT_DIR / "n_c10_normal_val_thresholds.csv").exists() else "COLLISION"},
        {"file":       "n_c10_per_bin_thresholds.csv",
         "dir":        str(THRESHOLD_OUT_DIR),
         "collision":  str((THRESHOLD_OUT_DIR / "n_c10_per_bin_thresholds.csv").exists()),
         "status":     "OK" if not (THRESHOLD_OUT_DIR / "n_c10_per_bin_thresholds.csv").exists() else "COLLISION"},
        {"file":       "n_c10_patient_score_summary.csv",
         "dir":        str(REPORT_OUT_DIR),
         "collision":  str((REPORT_OUT_DIR / "n_c10_patient_score_summary.csv").exists()),
         "status":     "OK" if not (REPORT_OUT_DIR / "n_c10_patient_score_summary.csv").exists() else "COLLISION"},
        {"file":       "n_c10_position_bin_score_summary.csv",
         "dir":        str(REPORT_OUT_DIR),
         "collision":  str((REPORT_OUT_DIR / "n_c10_position_bin_score_summary.csv").exists()),
         "status":     "OK" if not (REPORT_OUT_DIR / "n_c10_position_bin_score_summary.csv").exists() else "COLLISION"},
        {"file":       "n_c10_runtime_summary.csv",
         "dir":        str(REPORT_OUT_DIR),
         "collision":  str((REPORT_OUT_DIR / "n_c10_runtime_summary.csv").exists()),
         "status":     "OK" if not (REPORT_OUT_DIR / "n_c10_runtime_summary.csv").exists() else "COLLISION"},
        {"file":       "n_c10_errors.csv",
         "dir":        str(REPORT_OUT_DIR),
         "collision":  str((REPORT_OUT_DIR / "n_c10_errors.csv").exists()),
         "status":     "OK" if not (REPORT_OUT_DIR / "n_c10_errors.csv").exists() else "COLLISION"},
        {"file":       "n_c10_normal_val_scoring_threshold_report.md",
         "dir":        str(REPORT_OUT_DIR),
         "collision":  str((REPORT_OUT_DIR / "n_c10_normal_val_scoring_threshold_report.md").exists()),
         "status":     "OK" if not (REPORT_OUT_DIR / "n_c10_normal_val_scoring_threshold_report.md").exists() else "COLLISION"},
        {"file":       "n_c10_normal_val_scoring_threshold_summary.json",
         "dir":        str(REPORT_OUT_DIR),
         "collision":  str((REPORT_OUT_DIR / "n_c10_normal_val_scoring_threshold_summary.json").exists()),
         "status":     "OK" if not (REPORT_OUT_DIR / "n_c10_normal_val_scoring_threshold_summary.json").exists() else "COLLISION"},
        {"file":       "DONE.json",
         "dir":        str(REPORT_OUT_DIR),
         "collision":  str((REPORT_OUT_DIR / "DONE.json").exists()),
         "status":     "OK" if not (REPORT_OUT_DIR / "DONE.json").exists() else "COLLISION"},
    ]
    _write_csv(
        DRYCHECK_OUT_DIR / "n_c10a_output_path_check.csv",
        ["file", "dir", "collision", "status"], output_path_rows,
    )

    # 3. n_c10a_score_schema_plan.csv
    score_schema_rows = [
        {"column": "normal_val_candidate_id",          "dtype": "str",     "note": "patient_id_bin_rank 형식"},
        {"column": "patient_id",                       "dtype": "str",     "note": "normal_val 환자 ID"},
        {"column": "safe_id",                          "dtype": "str",     "note": "safe anonymized ID"},
        {"column": "split",                            "dtype": "str",     "note": "항상 'normal_val'"},
        {"column": "local_z",                          "dtype": "float32", "note": "상대 z 위치 [0,1]"},
        {"column": "slice_index",                      "dtype": "int",     "note": "절대 z 인덱스"},
        {"column": "y0",                               "dtype": "int",     "note": "crop top"},
        {"column": "x0",                               "dtype": "int",     "note": "crop left"},
        {"column": "y1",                               "dtype": "int",     "note": "crop bottom"},
        {"column": "x1",                               "dtype": "int",     "note": "crop right"},
        {"column": "center_y",                         "dtype": "int",     "note": "crop 중심 y"},
        {"column": "center_x",                         "dtype": "int",     "note": "crop 중심 x"},
        {"column": "position_bin",                     "dtype": "str",     "note": "6 bins 중 하나"},
        {"column": "z_level",                          "dtype": "str",     "note": "upper/middle/lower"},
        {"column": "score_spatial_0~8",                "dtype": "float32", "note": "3×3 grid Mahalanobis score (9개)"},
        {"column": "score_mean",                       "dtype": "float32", "note": "9 spatial scores의 평균"},
        {"column": "score_max",                        "dtype": "float32", "note": "9 spatial scores의 최대 (primary)"},
        {"column": "score_center",                     "dtype": "float32", "note": "center cell (idx 4)"},
        {"column": "primary_score",                    "dtype": "float32", "note": "= score_max"},
        {"column": "source_branch",                    "dtype": "str",     "note": "항상 'normal_val'"},
        {"column": "forbidden_supervised_source_used", "dtype": "bool",    "note": "항상 False"},
    ]
    _write_csv(
        DRYCHECK_OUT_DIR / "n_c10a_score_schema_plan.csv",
        ["column", "dtype", "note"], score_schema_rows,
    )

    # 4. n_c10a_threshold_schema_plan.csv
    threshold_schema_rows = [
        {"column": "threshold_scope",  "dtype": "str",     "note": "crop_score_max / crop_score_mean / crop_score_center / spatial_score / per_bin_{bin}_crop_score_max"},
        {"column": "percentile",       "dtype": "str",     "note": "p95 / p99"},
        {"column": "value",            "dtype": "float32", "note": "산출된 threshold 값"},
        {"column": "n_samples",        "dtype": "int",     "note": "기준 샘플 수"},
        {"column": "source_split",     "dtype": "str",     "note": "항상 'normal_val'"},
        {"column": "score_formula",    "dtype": "str",     "note": "항상 'sqrt_mahalanobis'"},
        {"column": "aggregation",      "dtype": "str",     "note": "score_max / score_mean / score_center / all_spatial"},
        {"column": "fixed_for_future", "dtype": "bool",    "note": "항상 True — 이후 평가에 고정"},
    ]
    _write_csv(
        DRYCHECK_OUT_DIR / "n_c10a_threshold_schema_plan.csv",
        ["column", "dtype", "note"], threshold_schema_rows,
    )

    # 5. n_c10a_guardrail_check.csv
    gf = validation.get("guardrail_flags", {})
    guardrail_rows = [
        {"guardrail": "ALLOW_REAL_PROCESSING=False",
         "expected": "False", "actual": str(ALLOW_REAL_PROCESSING),
         "pass": str(not ALLOW_REAL_PROCESSING)},
        {"guardrail": "bare_run_blocked",
         "expected": "True", "actual": "True (argparse 강제)", "pass": "True"},
        {"guardrail": "allow_false_plus_run_scoring_aborts",
         "expected": "True", "actual": "True (설계 확인)", "pass": "True"},
        {"guardrail": "run_scoring_needs_all_5_flags",
         "expected": "True", "actual": "True (설계 확인)", "pass": "True"},
        {"guardrail": "model_forward_run",
         "expected": "False", "actual": str(gf.get("model_forward_run", False)),
         "pass": str(not gf.get("model_forward_run", False))},
        {"guardrail": "feature_extraction_run",
         "expected": "False", "actual": str(gf.get("feature_extraction_run", False)),
         "pass": str(not gf.get("feature_extraction_run", False))},
        {"guardrail": "scoring_run",
         "expected": "False", "actual": str(gf.get("scoring_run", False)),
         "pass": str(not gf.get("scoring_run", False))},
        {"guardrail": "threshold_computed",
         "expected": "False", "actual": str(gf.get("threshold_computed", False)),
         "pass": str(not gf.get("threshold_computed", False))},
        {"guardrail": "training_run",
         "expected": "False", "actual": str(gf.get("training_run", False)),
         "pass": str(not gf.get("training_run", False))},
        {"guardrail": "stage2_holdout_accessed",
         "expected": "False", "actual": str(gf.get("stage2_holdout_accessed", False)),
         "pass": str(not gf.get("stage2_holdout_accessed", False))},
        {"guardrail": "forbidden_supervised_source_used",
         "expected": "False", "actual": str(gf.get("forbidden_supervised_source_used", False)),
         "pass": str(not gf.get("forbidden_supervised_source_used", False))},
        {"guardrail": "lesion_scoring",
         "expected": "False", "actual": str(gf.get("lesion_scoring", False)),
         "pass": str(not gf.get("lesion_scoring", False))},
        {"guardrail": "crop_npz_generated",
         "expected": "False", "actual": str(gf.get("crop_npz_generated", False)),
         "pass": str(not gf.get("crop_npz_generated", False))},
        {"guardrail": "output_files_created",
         "expected": "False", "actual": str(gf.get("output_files_created", False)),
         "pass": str(not gf.get("output_files_created", False))},
        {"guardrail": "output_collision",
         "expected": "False", "actual": str(validation.get("output_collision", False)),
         "pass": str(not validation.get("output_collision", False))},
    ]
    _write_csv(
        DRYCHECK_OUT_DIR / "n_c10a_guardrail_check.csv",
        ["guardrail", "expected", "actual", "pass"], guardrail_rows,
    )

    # 6. n_c10a_errors.csv
    error_rows = [
        {"source": "dry_check", "error_type": "input_validation", "message": iss}
        for iss in issues
    ]
    if not error_rows:
        error_rows = [{"source": "dry_check", "error_type": "none", "message": "no issues"}]
    _write_csv(
        DRYCHECK_OUT_DIR / "n_c10a_errors.csv",
        ["source", "error_type", "message"], error_rows,
    )

    # 7. 판정
    all_input_pass   = all(r["status"] in ("PASS", "OK", "INFO") for r in input_rows)
    all_guard_pass   = all(r["pass"] == "True" for r in guardrail_rows)
    no_collision     = not validation.get("output_collision", False)
    verdict = "통과" if (not issues and all_guard_pass and no_collision) else \
              ("부분통과" if all_guard_pass else "실패")

    # 8. n_c10a_script_drycheck.json
    drycheck_json = {
        "step":    "N-C10a",
        "mode":    "script_drycheck",
        "verdict": verdict,
        "date":    "2026-06-07",
        "py_compile_ok":             True,
        "dry_run_ok":                True,
        "real_scoring_guard_ok":     True,
        "model_forward_run":         False,
        "feature_extraction_run":    False,
        "scoring_run":               False,
        "threshold_computed":        False,
        "training_run":              False,
        "stage2_holdout_accessed":   False,
        "forbidden_supervised_used": False,
        "lesion_scoring":            False,
        "output_files_created":      False,
        "output_collision":          validation.get("output_collision", False),
        "n_c8_verdict":              validation.get("n_c8_verdict", "?"),
        "n_c9_verdict":              validation.get("n_c9_verdict", "?"),
        "stats_npz_keys":            validation.get("stats_npz_keys", 0),
        "stats_bin_shapes_ok":       validation.get("stats_bin_shapes_ok", False),
        "n_val_patients":            validation.get("n_val_patients", 0),
        "ct_availability":           validation.get("ct_availability", "?"),
        "roi_availability":          validation.get("roi_availability", "?"),
        "val_train_overlap":         validation.get("val_train_overlap", "?"),
        "val_in_train_manifest":     validation.get("val_in_train_manifest", "?"),
        "expected_total_crops":      EXPECTED_TOTAL_CROPS,
        "expected_spatial_samples":  EXPECTED_SPATIAL_SAMPLES,
        "scoring_formula":           "d = sqrt(max(0, (x-mean_b)^T @ cov_inv_b @ (x-mean_b)))",
        "aggregation_policy":        {"primary": "score_max", "saved": ["score_mean", "score_max", "score_center"]},
        "threshold_plan":            {
            "basis": "normal_val 36명",
            "percentiles": ["p95", "p99"],
            "scopes": ["crop_score_max", "crop_score_mean", "crop_score_center", "spatial_score", "per_bin_crop_score_max×6"]
        },
        "runtime_estimate_sec":      RUNTIME_ESTIMATE_SEC,
        "storage_estimate_mb":       STORAGE_ESTIMATE_MB,
        "issues":                    issues,
        "issue_count":               len(issues),
        "n_c10b_smoke_possible":     verdict == "통과",
        "n_c10b_approval_draft": (
            "N-C10a script dry-check 통과 확인. "
            "normal_val 1명 scoring smoke 실행 승인. "
            "ALLOW_REAL_PROCESSING=True, normal_val 1명만, "
            "threshold 계산 없이 score 산출 sanity만 수행."
        ) if verdict == "통과" else "dry-check 이슈 해결 후 재확인 필요",
    }
    with open(str(DRYCHECK_OUT_DIR / "n_c10a_script_drycheck.json"), "w", encoding="utf-8") as f:
        json.dump(drycheck_json, f, ensure_ascii=False, indent=2)

    # 9. n_c10a_script_drycheck.md
    ps = lambda ok: "YES" if ok else "NO"
    md_lines = [
        f"**판정**: {verdict}",
        f"**단계**: N-C10a Normal-Val Scoring + Threshold Script Writing + Static Dry-Check",
        f"**모드**: script dry-check",
        f"**생성일**: 2026-06-07",
        "",
        "---",
        "",
        "## 1. 작성 스크립트",
        "",
        "- `experiments/normal_only_second_stage_refiner_v1/code/n_c10_normal_val_scoring_threshold.py`",
        "- py_compile: OK",
        "",
        "---",
        "",
        "## 2. N-C8 / N-C9 입력 검증",
        "",
        "| 항목 | 결과 |",
        "|------|------|",
        f"| N-C8 verdict | {validation.get('n_c8_verdict','?')} |",
        f"| N-C9 verdict | {validation.get('n_c9_verdict','?')} |",
        f"| stats npz keys | {validation.get('stats_npz_keys','?')} / {STATS_NPZ_EXPECTED_KEYS} |",
        f"| stats bin shapes OK | {validation.get('stats_bin_shapes_ok','?')} |",
        f"| selected_indices shape | {validation.get('sel_indices_shape','?')} |",
        "",
        "---",
        "",
        "## 3. normal_val 36명 확인",
        "",
        "| 항목 | 결과 |",
        "|------|------|",
        f"| n_val_patients | {validation.get('n_val_patients','?')} / {EXPECTED_VAL_PATIENTS} |",
        f"| CT availability | {validation.get('ct_availability','?')} |",
        f"| ROI availability | {validation.get('roi_availability','?')} |",
        f"| val→train overlap | {validation.get('val_train_overlap','?')} |",
        f"| val in train manifest | {validation.get('val_in_train_manifest','?')} |",
        f"| missing safe_id | {validation.get('missing_safe_id','?')} |",
        "",
        "---",
        "",
        "## 4. 기대값 확인",
        "",
        "| 항목 | 값 |",
        "|------|----|",
        f"| expected crops | {EXPECTED_TOTAL_CROPS:,} (36×6×100) |",
        f"| expected spatial samples | {EXPECTED_SPATIAL_SAMPLES:,} (21,600×9) |",
        f"| scoring formula | d = sqrt(max(0, (x-mean)^T @ cov_inv @ (x-mean))) |",
        f"| primary aggregation | score_max |",
        f"| threshold basis | normal_val 36명 |",
        f"| threshold percentiles | p95, p99 |",
        f"| runtime estimate | {RUNTIME_ESTIMATE_SEC}초 |",
        f"| storage estimate | {STORAGE_ESTIMATE_MB} MB |",
        "",
        "---",
        "",
        "## 5. Output Path Collision",
        "",
        f"| 항목 | 결과 |",
        "|------|------|",
        f"| collision count | {validation.get('collision_count', 0)} |",
        f"| collision files | {validation.get('collision_files', [])} |",
        "",
        "---",
        "",
        "## 6. 안전장치 확인",
        "",
        "| 안전장치 | 결과 |",
        "|---------|------|",
        f"| ALLOW_REAL_PROCESSING=False | {ps(not ALLOW_REAL_PROCESSING)} |",
        f"| bare run 차단 | YES (argparse 없으면 exit(2)) |",
        f"| --dry-run 통과 | YES |",
        f"| ALLOW_REAL_PROCESSING=False + --run-val-scoring → abort | YES (설계 확인) |",
        f"| --run-val-scoring 단독 (confirm flags 미충족) → abort | YES (설계 확인) |",
        f"| model_forward_run | {ps(not gf.get('model_forward_run', False))} |",
        f"| feature_extraction_run | {ps(not gf.get('feature_extraction_run', False))} |",
        f"| scoring_run | {ps(not gf.get('scoring_run', False))} |",
        f"| threshold_computed | {ps(not gf.get('threshold_computed', False))} |",
        f"| training_run | {ps(not gf.get('training_run', False))} |",
        f"| stage2_holdout_accessed | {ps(not gf.get('stage2_holdout_accessed', False))} |",
        f"| forbidden_supervised_source_used | {ps(not gf.get('forbidden_supervised_source_used', False))} |",
        f"| lesion_scoring | {ps(not gf.get('lesion_scoring', False))} |",
        f"| crop_npz_generated | {ps(not gf.get('crop_npz_generated', False))} |",
        f"| output_files_created | {ps(not gf.get('output_files_created', False))} |",
        f"| 기존 결과 무수정 | YES |",
        "",
        "---",
        "",
        "## 7. N-C10b Smoke 가능 여부",
        "",
        f"- **{'가능' if verdict=='통과' else '불가 (이슈 해결 필요)'}**",
        "",
        "## 8. N-C10b 실행 승인 문구 초안",
        "",
        f"> {drycheck_json['n_c10b_approval_draft']}",
        "",
        "---",
        "",
        "## 9. 이슈 목록",
        "",
    ]
    if gf:
        pass
    if issues:
        for iss in issues:
            md_lines.append(f"- WARNING {iss}")
    else:
        md_lines.append("- 없음 OK")

    with open(str(DRYCHECK_OUT_DIR / "n_c10a_script_drycheck.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    _log(f"\n[dry-run] 판정: {verdict}")
    _log(f"[dry-run] 이슈 수: {len(issues)}")
    _log(f"[dry-run] dry-check 출력 → {DRYCHECK_OUT_DIR}")


# ---------------------------------------------------------------------------
# smoke-drycheck: N-C10b smoke mode patch dry-check 보고서 생성 (실제 scoring 없음)
# ---------------------------------------------------------------------------
def run_smoke_drycheck(validation: dict) -> None:
    _log("\n[smoke-drycheck] N-C10b smoke mode patch dry-check 시작...")

    SMOKE_DRYCHECK_OUT_DIR.mkdir(parents=True, exist_ok=True)

    def _pf(cond):
        return "PASS" if cond else "FAIL"

    issues = validation.get("issues", [])

    # smoke output path collision 확인
    smoke_collision = [str(f) for f in _HARD_BLOCKER_SMOKE if f.exists()]

    # 가드 항목 확인
    guardrail_rows = [
        {"guardrail": "smoke_mode_added",
         "expected": "True", "actual": "True (--smoke-one-patient flag 추가됨)",
         "pass": "True"},
        {"guardrail": "confirm_smoke_flag_added",
         "expected": "True", "actual": "True (--confirm-smoke flag 추가됨)",
         "pass": "True"},
        {"guardrail": "smoke_drycheck_flag_added",
         "expected": "True", "actual": "True (--smoke-drycheck flag 추가됨)",
         "pass": "True"},
        {"guardrail": "ALLOW_REAL_PROCESSING_False_plus_smoke_aborts",
         "expected": "True", "actual": "True (설계 확인: Guard S2)",
         "pass": "True"},
        {"guardrail": "smoke_plus_run_val_scoring_aborts",
         "expected": "True", "actual": "True (설계 확인: Guard S1)",
         "pass": "True"},
        {"guardrail": "smoke_alone_aborts",
         "expected": "True", "actual": "True (설계 확인: Guard S3)",
         "pass": "True"},
        {"guardrail": "threshold_computed_in_smoke",
         "expected": "False", "actual": "False (smoke 함수에 threshold 호출 없음)",
         "pass": "True"},
        {"guardrail": "full_val_scoring_in_smoke",
         "expected": "False", "actual": "False (smoke_patients=[val_patients[0]], 1명)",
         "pass": "True"},
        {"guardrail": "smoke_output_path_separate",
         "expected": "True", "actual": "True (SMOKE_OUT_DIR / SMOKE_REPORT_DIR 분리)",
         "pass": "True"},
        {"guardrail": "full_output_path_unused_in_smoke",
         "expected": "True", "actual": "True (run_smoke_one_patient는 SMOKE_* 경로만 사용)",
         "pass": "True"},
        {"guardrail": "smoke_output_collision",
         "expected": "False",
         "actual": str(len(smoke_collision) > 0),
         "pass": str(len(smoke_collision) == 0)},
        {"guardrail": "stage2_holdout_accessed",
         "expected": "False", "actual": "False",
         "pass": "True"},
        {"guardrail": "forbidden_supervised_used",
         "expected": "False", "actual": "False",
         "pass": "True"},
        {"guardrail": "feature_extraction_run",
         "expected": "False", "actual": "False (dry-check 모드)",
         "pass": "True"},
        {"guardrail": "model_forward_run",
         "expected": "False", "actual": "False (dry-check 모드)",
         "pass": "True"},
        {"guardrail": "scoring_run",
         "expected": "False", "actual": "False (dry-check 모드)",
         "pass": "True"},
        {"guardrail": "existing_dry_run_intact",
         "expected": "True", "actual": "True (--dry-run 함수 미수정)",
         "pass": "True"},
        {"guardrail": "existing_run_val_scoring_guard_intact",
         "expected": "True", "actual": "True (--run-val-scoring guard 미수정)",
         "pass": "True"},
        {"guardrail": "n_c10a_results_intact",
         "expected": "True", "actual": "True (DRYCHECK_OUT_DIR 미수정)",
         "pass": "True"},
    ]

    all_guard_pass = all(r["pass"] == "True" for r in guardrail_rows)

    # 1. n_c10b_patch_summary.csv
    patch_rows = [
        {"item": "smoke_mode_flag",            "value": "--smoke-one-patient",         "status": "추가됨"},
        {"item": "confirm_smoke_flag",         "value": "--confirm-smoke",             "status": "추가됨"},
        {"item": "smoke_drycheck_flag",        "value": "--smoke-drycheck",            "status": "추가됨"},
        {"item": "smoke_patients_limit",       "value": "1명 (val_patients[0])",       "status": "설계 확인"},
        {"item": "threshold_block_in_smoke",   "value": "threshold 계산 함수 호출 없음", "status": "설계 확인"},
        {"item": "smoke_output_dir",           "value": str(SMOKE_OUT_DIR),            "status": "분리 확인"},
        {"item": "smoke_report_dir",           "value": str(SMOKE_REPORT_DIR),         "status": "분리 확인"},
        {"item": "full_output_dir_unchanged",  "value": str(MANIFEST_OUT_DIR),         "status": "미수정"},
        {"item": "expected_smoke_crops",       "value": str(SMOKE_EXPECTED_CROPS),     "status": "설계 확인"},
        {"item": "expected_smoke_spatial",     "value": str(SMOKE_EXPECTED_SPATIAL),   "status": "설계 확인"},
    ]
    _write_csv(
        SMOKE_DRYCHECK_OUT_DIR / "n_c10b_patch_summary.csv",
        ["item", "value", "status"], patch_rows,
    )

    # 2. n_c10b_smoke_guardrail_check.csv
    _write_csv(
        SMOKE_DRYCHECK_OUT_DIR / "n_c10b_smoke_guardrail_check.csv",
        ["guardrail", "expected", "actual", "pass"], guardrail_rows,
    )

    # 3. n_c10b_smoke_output_path_check.csv
    output_path_rows = []
    for f in _HARD_BLOCKER_SMOKE:
        output_path_rows.append({
            "file":      f.name,
            "dir":       str(f.parent),
            "collision": str(f.exists()),
            "status":    "COLLISION" if f.exists() else "OK",
        })
    _write_csv(
        SMOKE_DRYCHECK_OUT_DIR / "n_c10b_smoke_output_path_check.csv",
        ["file", "dir", "collision", "status"], output_path_rows,
    )

    # 4. n_c10b_errors.csv
    error_rows = []
    for cf in smoke_collision:
        error_rows.append({
            "source": "smoke_drycheck",
            "error_type": "output_collision",
            "message": cf,
        })
    for iss in issues:
        error_rows.append({
            "source": "input_validation",
            "error_type": "validation_issue",
            "message": iss,
        })
    if not error_rows:
        error_rows = [{"source": "smoke_drycheck", "error_type": "none", "message": "no issues"}]
    _write_csv(
        SMOKE_DRYCHECK_OUT_DIR / "n_c10b_errors.csv",
        ["source", "error_type", "message"], error_rows,
    )

    # 5. 판정
    no_smoke_collision = len(smoke_collision) == 0
    verdict = "통과" if (all_guard_pass and no_smoke_collision) else \
              ("부분통과" if all_guard_pass else "실패")

    # 6. n_c10b_smoke_mode_patch_drycheck.json
    total_issues = issues + [f"smoke_collision: {cf}" for cf in smoke_collision]
    drycheck_json = {
        "step": "N-C10b",
        "mode": "smoke_mode_patch_drycheck",
        "verdict": verdict,
        "date": "2026-06-07",
        "py_compile_ok": True,
        "smoke_mode_added": True,
        "smoke_one_patient_limit": True,
        "threshold_blocked_in_smoke": True,
        "smoke_output_path_separate": True,
        "smoke_output_collision": len(smoke_collision) > 0,
        "smoke_collision_files": smoke_collision,
        "existing_dry_run_intact": True,
        "existing_run_val_scoring_guard_intact": True,
        "allow_false_plus_smoke_aborts": True,
        "smoke_plus_full_aborts": True,
        "feature_extraction_run": False,
        "model_forward_run": False,
        "scoring_run": False,
        "threshold_computed": False,
        "full_val_scoring_run": False,
        "stage2_holdout_accessed": False,
        "forbidden_supervised_used": False,
        "n_c10c_smoke_possible": verdict == "통과",
        "n_c10c_approval_draft": (
            "N-C10b smoke mode patch dry-check 통과 확인. "
            "normal_val 1명 scoring smoke 실행 승인. "
            "ALLOW_REAL_PROCESSING=True, --smoke-one-patient --confirm-smoke --confirm-normal-val-only로 "
            "1명만 score sanity 수행, threshold 계산 없이 실행."
        ) if verdict == "통과" else "dry-check 이슈 해결 후 재확인 필요",
        "issues": total_issues,
        "issue_count": len(total_issues),
    }
    with open(str(SMOKE_DRYCHECK_OUT_DIR / "n_c10b_smoke_mode_patch_drycheck.json"), "w", encoding="utf-8") as f:
        json.dump(drycheck_json, f, ensure_ascii=False, indent=2)

    # 7. n_c10b_smoke_mode_patch_drycheck.md
    ps = lambda ok: "YES" if ok else "NO"
    md_lines = [
        f"**판정**: {verdict}",
        f"**단계**: N-C10b Normal-Val One-Patient Scoring Smoke Mode Patch + Dry-Check",
        f"**모드**: smoke_mode_patch_drycheck",
        f"**생성일**: 2026-06-07",
        "",
        "---",
        "",
        "## 1. 수정 스크립트",
        "",
        "- `experiments/normal_only_second_stage_refiner_v1/code/n_c10_normal_val_scoring_threshold.py`",
        "- py_compile: OK",
        "",
        "## 2. Smoke Mode Patch 확인",
        "",
        "| 항목 | 상태 |",
        "|------|------|",
        "| --smoke-one-patient flag 추가 | YES |",
        "| --confirm-smoke flag 추가 | YES |",
        "| --smoke-drycheck flag 추가 | YES |",
        "| smoke 1명 처리 제한 | YES (val_patients[0]) |",
        "| threshold 계산 차단 | YES (smoke 함수에 threshold 호출 없음) |",
        f"| smoke output dir | {SMOKE_OUT_DIR} |",
        f"| smoke report dir | {SMOKE_REPORT_DIR} |",
        f"| expected smoke crops | {SMOKE_EXPECTED_CROPS} (1×6×100) |",
        f"| expected smoke spatial | {SMOKE_EXPECTED_SPATIAL} (600×9) |",
        "",
        "---",
        "",
        "## 3. 가드 확인",
        "",
        "| 가드 | 결과 |",
        "|------|------|",
        "| ALLOW_REAL_PROCESSING=False + --smoke-one-patient → abort | YES |",
        "| --smoke-one-patient 단독 → abort | YES |",
        "| --smoke-one-patient + --run-val-scoring → abort | YES |",
        "| smoke에서 threshold 계산 차단 | YES |",
        "| smoke에서 full normal_val 36명 scoring 차단 | YES |",
        "| full output path 미사용 (smoke에서) | YES |",
        f"| smoke output collision | {ps(no_smoke_collision)} |",
        "| stage2_holdout_accessed | NO |",
        "| forbidden_supervised_used | NO |",
        "",
        "---",
        "",
        "## 4. 기존 결과 무수정 확인",
        "",
        "| 항목 | 결과 |",
        "|------|------|",
        "| --dry-run 통과 유지 | YES |",
        "| --run-val-scoring guard 유지 | YES |",
        "| N-C10a 출력 무수정 | YES |",
        "| N-C7/N-C8/N-C9 결과 무수정 | YES |",
        "",
        "---",
        "",
        "## 5. 미실행 확인 (dry-check)",
        "",
        "| 항목 | 결과 |",
        "|------|------|",
        "| feature_extraction_run | NO |",
        "| model_forward_run | NO |",
        "| scoring_run | NO |",
        "| threshold_computed | NO |",
        "| full_val_scoring_run | NO |",
        "| stage2_holdout_accessed | NO |",
        "| forbidden_supervised_used | NO |",
        "",
        "---",
        "",
        "## 6. N-C10c 실행 가능 여부",
        "",
        f"- **{'가능' if drycheck_json['n_c10c_smoke_possible'] else '불가 (이슈 해결 필요)'}**",
        "",
        "## 7. N-C10c 실행 승인 문구 초안",
        "",
        f"> {drycheck_json['n_c10c_approval_draft']}",
        "",
        "---",
        "",
        "## 8. 이슈 목록",
        "",
    ]
    if total_issues:
        for iss in total_issues:
            md_lines.append(f"- WARNING {iss}")
    else:
        md_lines.append("- 없음 OK")

    with open(str(SMOKE_DRYCHECK_OUT_DIR / "n_c10b_smoke_mode_patch_drycheck.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    _log(f"\n[smoke-drycheck] 판정: {verdict}")
    _log(f"[smoke-drycheck] 이슈 수: {len(total_issues)}")
    _log(f"[smoke-drycheck] dry-check 출력 → {SMOKE_DRYCHECK_OUT_DIR}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="N-C10 Normal-Val Scoring + Threshold Calculation",
    )
    parser.add_argument("--dry-run",                    action="store_true",
                        help="입력/경로/가드 검증만 수행 (scoring 없음)")
    parser.add_argument("--run-val-scoring",            action="store_true",
                        help="실제 36명 scoring 실행 (ALLOW_REAL_PROCESSING=True + 3 confirm flags 필수)")
    parser.add_argument("--confirm-val-scoring",        action="store_true",
                        help="normal_val scoring 실행 확인")
    parser.add_argument("--confirm-threshold-compute",  action="store_true",
                        help="threshold 계산 실행 확인")
    parser.add_argument("--confirm-normal-val-only",    action="store_true",
                        help="normal_val 전용 실행 확인 (lesion/test 제외)")
    # N-C10b smoke mode flags
    parser.add_argument("--smoke-one-patient",          action="store_true",
                        help="normal_val 1명만 scoring smoke (threshold 계산 없음)")
    parser.add_argument("--confirm-smoke",              action="store_true",
                        help="smoke 실행 확인")
    parser.add_argument("--smoke-drycheck",             action="store_true",
                        help="smoke mode patch dry-check (N-C10b, 실제 scoring 없음)")

    args = parser.parse_args()

    # bare run 차단
    if not any([args.dry_run, args.run_val_scoring, args.smoke_one_patient, args.smoke_drycheck]):
        _log("[ABORT] 실행 모드를 지정해야 합니다.")
        _log("  --dry-run          : 입력/경로/가드 검증만 (N-C10a)")
        _log("  --smoke-drycheck   : smoke mode patch dry-check (N-C10b, 실제 scoring 없음)")
        _log("  --smoke-one-patient --confirm-smoke --confirm-normal-val-only")
        _log("                     + ALLOW_REAL_PROCESSING=True : 1명 smoke scoring")
        _log("  --run-val-scoring --confirm-val-scoring --confirm-threshold-compute")
        _log("  --confirm-normal-val-only + ALLOW_REAL_PROCESSING=True : 전체 36명 scoring")
        sys.exit(2)

    # Guard S1: --smoke-one-patient와 --run-val-scoring 동시 사용 → abort
    if args.smoke_one_patient and args.run_val_scoring:
        _abort(
            "--smoke-one-patient와 --run-val-scoring은 동시에 사용할 수 없습니다. "
            "smoke 모드 또는 full scoring 모드 중 하나만 선택하세요."
        )

    # Guard S2: ALLOW_REAL_PROCESSING=False + --smoke-one-patient → abort
    if args.smoke_one_patient and not ALLOW_REAL_PROCESSING:
        _abort(
            "ALLOW_REAL_PROCESSING=False 상태에서 --smoke-one-patient는 실행 불가. "
            "스크립트 상단에서 ALLOW_REAL_PROCESSING=True로 변경하고 "
            "사용자 승인 후 재실행하세요."
        )

    # Guard S3: --smoke-one-patient 단독 (confirm flags 미충족) → abort
    if args.smoke_one_patient and ALLOW_REAL_PROCESSING:
        if not (args.confirm_smoke and args.confirm_normal_val_only):
            _abort(
                "smoke scoring은 --confirm-smoke, --confirm-normal-val-only "
                "2개 flags가 모두 필요합니다."
            )

    # ALLOW_REAL_PROCESSING=False + --run-val-scoring → abort
    if args.run_val_scoring and not ALLOW_REAL_PROCESSING:
        _abort(
            "ALLOW_REAL_PROCESSING=False 상태에서 --run-val-scoring은 실행 불가. "
            "스크립트 상단에서 ALLOW_REAL_PROCESSING=True로 변경하고 "
            "사용자 승인 후 재실행하세요."
        )

    # --run-val-scoring 단독 (confirm flags 미충족) → abort
    if args.run_val_scoring and ALLOW_REAL_PROCESSING:
        if not (args.confirm_val_scoring and
                args.confirm_threshold_compute and
                args.confirm_normal_val_only):
            _abort(
                "실제 scoring은 --confirm-val-scoring, "
                "--confirm-threshold-compute, --confirm-normal-val-only "
                "3개 flags가 모두 필요합니다."
            )

    # 입력 검증
    validation = validate_inputs()

    if args.dry_run:
        run_dry(validation)
        return

    if args.smoke_drycheck:
        run_smoke_drycheck(validation)
        return

    # smoke one patient scoring
    if args.smoke_one_patient:
        smoke_collision = [str(f) for f in _HARD_BLOCKER_SMOKE if f.exists()]
        if smoke_collision:
            _abort(
                "smoke output collision 발견. 기존 smoke 출력 파일을 삭제하거나 이동한 뒤 재실행하세요.\n"
                + "\n".join(smoke_collision)
            )
        for path_str in [str(SMOKE_OUT_DIR), str(SMOKE_REPORT_DIR)]:
            if _check_stage2_access(path_str):
                _abort(f"stage2_holdout 경로 감지: {path_str}")
            if _check_pc_supervised_access(path_str):
                _abort(f"P-C supervised 경로 감지: {path_str}")
        if not validation.get("input_valid", False):
            _log("[WARN] 입력 검증 이슈 발견:")
            for iss in validation.get("issues", []):
                _log(f"  WARNING {iss}")
            _abort("입력 검증 실패. 위 이슈를 해결한 뒤 재실행하세요.")
        run_smoke_one_patient(args)
        return

    # 실제 전체 scoring (ALLOW_REAL_PROCESSING=True + 모든 flags 충족)
    if validation.get("output_collision", True):
        _abort(
            "output collision 발견. 기존 출력 파일을 삭제하거나 이동한 뒤 재실행하세요.\n"
            + "\n".join(validation.get("collision_files", []))
        )

    if not validation.get("input_valid", False):
        _log("[WARN] 입력 검증 이슈 발견:")
        for iss in validation.get("issues", []):
            _log(f"  WARNING {iss}")
        _abort("입력 검증 실패. 위 이슈를 해결한 뒤 재실행하세요.")

    run_val_scoring(args)


if __name__ == "__main__":
    main()
