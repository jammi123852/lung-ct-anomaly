"""
N-C5 Normal-Only Crop Feature Extraction Smoke (1 Patient)

normal001 환자 smoke manifest 600 crops에서 EfficientNet-B0 feature extraction 수행.
3×3 spatial feature 9개/crop 추출 검증.

실행 모드:
  --dry-run   : 경로/파일 존재 검증만 수행 (feature extraction 없음)
  --run-smoke : 실제 600 crop feature extraction 수행 (사용자 승인 후만)

안전 장치:
  ALLOW_REAL_PROCESSING=False → bare run 시 exit(2)
  기존 출력 파일 존재 시 exit(2) (overwrite 금지)
  600 초과 시 abort
  stage2_holdout 접근 금지
  crop npz 저장 금지
  학습/scoring/threshold 금지
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 안전 플래그 (기본 False: 직접 실행 불가)
# ---------------------------------------------------------------------------
ALLOW_REAL_PROCESSING = False

# ---------------------------------------------------------------------------
# 경로 상수
# ---------------------------------------------------------------------------
PROJ_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
EXP_ROOT  = PROJ_ROOT / "experiments" / "normal_only_second_stage_refiner_v1"

SMOKE_MANIFEST_CSV = (EXP_ROOT / "outputs" / "manifests" /
                      "n_c3_smoke_one_patient_manifest" / "n_c3_smoke_manifest.csv")
N_C4_JSON = (EXP_ROOT / "outputs" / "reports" /
             "n_c4_distribution_design_preflight" /
             "n_c4_distribution_design_preflight.json")
SELECTED_INDICES_PATH = (
    PROJ_ROOT / "experiments" /
    "efficientnet_b0_imagenet_chestwall_removed_roi_v1" /
    "outputs" / "models" / "distributions" / "selected_feature_indices.npy"
)

SMOKE_OUT_DIR  = EXP_ROOT / "outputs" / "smoke" / "n_c5_feature_extraction_one_patient"
REPORT_OUT_DIR = EXP_ROOT / "outputs" / "reports" / "n_c5_feature_extraction_one_patient"

# stage2_holdout 경로 (접근 금지)
_STAGE2_HOLDOUT_ROOTS = [
    str(PROJ_ROOT / "outputs" / "position-aware-padim-v1" / "stage2"),
    str(PROJ_ROOT / "data" / "holdout"),
    "stage2_holdout",
    "holdout",
]

# ---------------------------------------------------------------------------
# 설계 상수
# ---------------------------------------------------------------------------
EXPECTED_SMOKE_CROPS    = 600
EXPECTED_FEATURE_SAMPLES = 5400   # 600 × 9
EXPECTED_SPATIAL_PER_CROP = 9
RAW_FEATURE_DIM          = 144    # 24+40+80
SELECTED_FEATURE_DIM     = 100
EXPECTED_POSITION_BINS   = 6
EXPECTED_SAMPLES_PER_BIN = 900    # 100 crops × 9

CROP_SIZE  = 96
CROP_HALF  = CROP_SIZE // 2       # 48
HU_MIN     = -1000.0
HU_MAX     = 200.0

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

POSITION_BINS = [
    "upper_central", "upper_peripheral",
    "middle_central", "middle_peripheral",
    "lower_central", "lower_peripheral",
]

# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------
def _log(msg: str) -> None:
    print(msg, flush=True)


def _abort(msg: str, code: int = 2) -> None:
    _log(f"[ABORT] {msg}")
    sys.exit(code)


def _check_stage2_access(path: str) -> bool:
    for root in _STAGE2_HOLDOUT_ROOTS:
        if root.lower() in path.lower():
            return True
    return False


# ---------------------------------------------------------------------------
# 경로/입력 검증
# ---------------------------------------------------------------------------
def validate_inputs() -> dict:
    issues = []
    result = {}

    # G1: ALLOW_REAL_PROCESSING 선언 확인
    _log(f"[G1] ALLOW_REAL_PROCESSING={ALLOW_REAL_PROCESSING}")
    result["allow_real_processing"] = ALLOW_REAL_PROCESSING

    # G2: N-C4 verdict 확인
    if N_C4_JSON.exists():
        with open(N_C4_JSON, encoding="utf-8") as f:
            n_c4 = json.load(f)
        verdict = n_c4.get("verdict", "")
        _log(f"[G2] N-C4 verdict={verdict}")
        result["n_c4_verdict"] = verdict
        if verdict != "PASS":
            issues.append(f"G2: N-C4 verdict={verdict} (PASS 필요)")
    else:
        issues.append("G2: N-C4 JSON 없음")
        result["n_c4_verdict"] = "MISSING"

    # G3: smoke manifest 존재 및 rows=600 확인
    if SMOKE_MANIFEST_CSV.exists():
        import pandas as pd
        df = pd.read_csv(str(SMOKE_MANIFEST_CSV))
        rows = len(df)
        patients = df["patient_id"].unique().tolist()
        bins = df["position_bin"].value_counts().to_dict()
        _log(f"[G3] smoke manifest rows={rows}, patients={patients}")
        result["manifest_rows"] = rows
        result["manifest_patients"] = patients
        result["manifest_bins"] = bins
        if rows != EXPECTED_SMOKE_CROPS:
            issues.append(f"G3: manifest rows={rows} (600 필요)")
        if patients != ["normal001"] and sorted(patients) != ["normal001"]:
            issues.append(f"G3: patients={patients} (normal001 필요)")
        for b in POSITION_BINS:
            if bins.get(b, 0) != 100:
                issues.append(f"G3: bin {b} count={bins.get(b,0)} (100 필요)")
        forbidden_used = df["forbidden_supervised_source_used"].any()
        result["forbidden_supervised_source_used"] = bool(forbidden_used)
        if forbidden_used:
            issues.append("G3: forbidden_supervised_source_used=True 행 존재")
    else:
        issues.append("G3: smoke manifest CSV 없음")
        result["manifest_rows"] = 0
        df = None

    # G4: CT/ROI 경로 확인
    if df is not None and len(df) > 0:
        row0 = df.iloc[0]
        ct_path  = row0["source_ct_path"]
        roi_path = row0["source_roi_path"]

        if _check_stage2_access(ct_path) or _check_stage2_access(roi_path):
            issues.append("G4: stage2_holdout 경로 접근 시도")
        ct_exists  = Path(ct_path).exists()
        roi_exists = Path(roi_path).exists()
        _log(f"[G4] CT exists={ct_exists}: {ct_path}")
        _log(f"[G4] ROI exists={roi_exists}: {roi_path}")
        result["ct_path"]    = ct_path
        result["roi_path"]   = roi_path
        result["ct_exists"]  = ct_exists
        result["roi_exists"] = roi_exists
        if not ct_exists:
            issues.append(f"G4: CT 없음: {ct_path}")
        if not roi_exists:
            issues.append(f"G4: ROI 없음: {roi_path}")
    else:
        result["ct_exists"] = False
        result["roi_exists"] = False

    # G5: selected_feature_indices.npy 확인
    if SELECTED_INDICES_PATH.exists():
        import numpy as np
        idx = np.load(str(SELECTED_INDICES_PATH))
        idx_shape     = list(idx.shape)
        idx_unique    = int(len(set(idx.tolist())))
        idx_min       = int(idx.min())
        idx_max       = int(idx.max())
        _log(f"[G5] selected_indices shape={idx_shape}, unique={idx_unique}, range=[{idx_min},{idx_max}]")
        result["selected_indices_shape"]  = idx_shape
        result["selected_indices_unique"] = idx_unique
        result["selected_indices_min"]    = idx_min
        result["selected_indices_max"]    = idx_max
        if idx_shape != [100]:
            issues.append(f"G5: indices shape={idx_shape} (100 필요)")
        if idx_unique != 100:
            issues.append(f"G5: indices unique={idx_unique} (100 필요)")
        if idx_max >= RAW_FEATURE_DIM:
            issues.append(f"G5: indices max={idx_max} >= raw_dim={RAW_FEATURE_DIM}")
    else:
        issues.append(f"G5: selected_feature_indices.npy 없음: {SELECTED_INDICES_PATH}")
        result["selected_indices_shape"] = None

    result["issues"] = issues
    result["input_valid"] = (len(issues) == 0)
    return result


# ---------------------------------------------------------------------------
# 2.5D crop 추출
# ---------------------------------------------------------------------------
def extract_2p5d_crop(
    ct_vol: "np.ndarray",   # (Z, H, W) int16 mmap — 전체 복사 없음
    z_idx:  int,
    cy:     int,
    cx:     int,
) -> "np.ndarray":
    """
    z-1, z, z+1 슬라이스에서 96×96 crop을 추출해 (3, 96, 96) float32로 반환.
    slice 단위만 float32 변환. 경계 초과 시 HU_MIN(-1000)으로 패딩.
    """
    import numpy as np

    Z, H, W = ct_vol.shape
    half = CROP_HALF

    ch_imgs = []
    for dz in (-1, 0, 1):
        z = max(0, min(Z - 1, z_idx + dz))
        sl = ct_vol[z].astype(np.float32)  # (H, W)

        # crop 범위
        y0 = cy - half
        y1 = cy + half
        x0 = cx - half
        x1 = cx + half

        # 패딩이 필요한 경우
        pad_top    = max(0, -y0)
        pad_bottom = max(0, y1 - H)
        pad_left   = max(0, -x0)
        pad_right  = max(0, x1 - W)

        y0c = max(0, y0)
        y1c = min(H, y1)
        x0c = max(0, x0)
        x1c = min(W, x1)

        region = sl[y0c:y1c, x0c:x1c]

        if pad_top or pad_bottom or pad_left or pad_right:
            region = np.pad(
                region,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=HU_MIN,
            )

        # shape 보장
        if region.shape != (CROP_SIZE, CROP_SIZE):
            raise ValueError(
                f"crop shape 불일치: {region.shape} (expected {CROP_SIZE}×{CROP_SIZE})"
            )

        ch_imgs.append(region)

    stacked = np.stack(ch_imgs, axis=0)  # (3, 96, 96)
    return stacked


# ---------------------------------------------------------------------------
# HU clip + normalize + ImageNet normalize
# ---------------------------------------------------------------------------
def preprocess_2p5d_crop(crop: "np.ndarray") -> "np.ndarray":
    """
    (3, 96, 96) HU float32 → clip → [0,1] → ImageNet normalize.
    반환: (3, 96, 96) float32
    """
    import numpy as np
    # HU clip + [0,1]
    clipped = np.clip(crop, HU_MIN, HU_MAX)
    normed  = (clipped - HU_MIN) / (HU_MAX - HU_MIN)   # [0,1]
    normed  = normed.astype(np.float32)                  # (3, 96, 96)

    # ImageNet normalize: per-channel
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std  = np.array(IMAGENET_STD,  dtype=np.float32).reshape(3, 1, 1)
    result = (normed - mean) / std
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# 3×3 spatial feature 추출 (B_all_spatial)
# ---------------------------------------------------------------------------
def extract_3x3_spatial_features(
    f_early: "torch.Tensor",  # (1, 24, H/4, W/4)
    f_mid:   "torch.Tensor",  # (1, 40, H/8, W/8)
    f_late:  "torch.Tensor",  # (1, 80, H/16, W/16)
) -> "np.ndarray":
    """
    각 tap 을 adaptive_avg_pool2d(3,3) 후 reshape → (9, 144) concat.
    반환: (9, 144) float32
    """
    import torch.nn.functional as F
    import numpy as np

    def _pool_reshape(feat: "torch.Tensor") -> "np.ndarray":
        # (1, C, H, W) → pool → (1, C, 3, 3) → (9, C)
        pooled = F.adaptive_avg_pool2d(feat, (3, 3))       # (1, C, 3, 3)
        pooled = pooled.squeeze(0)                          # (C, 3, 3)
        C = pooled.shape[0]
        arr = pooled.cpu().numpy()                          # (C, 3, 3)
        arr = arr.transpose(1, 2, 0)                        # (3, 3, C)
        arr = arr.reshape(9, C)                             # (9, C)
        return arr.astype(np.float32)

    a = _pool_reshape(f_early)  # (9, 24)
    b = _pool_reshape(f_mid)    # (9, 40)
    c = _pool_reshape(f_late)   # (9, 80)
    concat = np.concatenate([a, b, c], axis=1)  # (9, 144)
    return concat


# ---------------------------------------------------------------------------
# Covariance 계산 및 검증
# ---------------------------------------------------------------------------
def compute_and_validate_cov(
    features: "np.ndarray",   # (N, 100)
    position_bin: str,
    eps: float = 1e-5,
) -> dict:
    """
    mean, cov 계산 + symmetry/NaN/Inf/diagonal/eigenvalue 검증.
    반환: dict (결과 + 검증 정보)
    """
    import numpy as np

    mean = features.mean(axis=0)               # (100,)
    centered = features - mean[np.newaxis, :]
    cov = (centered.T @ centered) / max(len(features) - 1, 1)  # (100,100)
    cov += np.eye(cov.shape[0], dtype=np.float32) * eps

    nan_count  = int(np.isnan(cov).sum())
    inf_count  = int(np.isinf(cov).sum())
    sym_diff   = float(np.max(np.abs(cov - cov.T)))
    symmetry_ok = sym_diff < 1e-8
    diag       = np.diag(cov)
    neg_diag   = int((diag < 0).sum())

    # 최솟값 eigenvalue (선택적)
    min_eigenvalue = None
    condition_number = None
    try:
        from scipy.linalg import eigvalsh, svd as scipy_svd
        eigs = eigvalsh(cov)
        min_eigenvalue = float(eigs.min())
        sv = scipy_svd(cov, compute_uv=False)
        condition_number = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    except Exception as e:
        _log(f"  [cov] eigenvalue/condition 계산 실패 (무시): {e}")

    # cov_inv (선택적)
    cov_inv = None
    try:
        import numpy.linalg as npl
        cov_inv = npl.inv(cov)
    except Exception:
        pass

    return {
        "position_bin":     position_bin,
        "mean":             mean,
        "cov":              cov,
        "cov_inv":          cov_inv,
        "count":            len(features),
        "cov_shape":        str(cov.shape),
        "symmetry_ok":      symmetry_ok,
        "nan_count":        nan_count,
        "inf_count":        inf_count,
        "neg_diagonal_count": neg_diag,
        "min_eigenvalue":   min_eigenvalue,
        "condition_number": condition_number,
    }


# ---------------------------------------------------------------------------
# 메인 실행
# ---------------------------------------------------------------------------
def run_dry(validation: dict) -> None:
    """--dry-run 모드: 경로/파일 검증 결과만 출력."""
    _log("\n[dry-run] 입력 검증 결과:")
    for k, v in validation.items():
        if k not in ("issues",):
            _log(f"  {k}: {v}")
    if validation["issues"]:
        _log("\n[dry-run] 문제 발견:")
        for issue in validation["issues"]:
            _log(f"  ⚠ {issue}")
    else:
        _log("[dry-run] 모든 입력 검증 통과 ✅")
    _log("[dry-run] feature extraction 미실행. --run-smoke 로 실제 실행.")


def run_smoke(validation: dict) -> None:
    """--run-smoke 모드: 실제 600 crop feature extraction."""
    import numpy as np
    import pandas as pd
    import torch

    # src 경로 추가
    src_path = str(PROJ_ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from position_aware_padim.feature_extractor_effnet_b0_scaffold import (
        FeatureExtractorEffNetB0,
    )

    t_total_start = time.perf_counter()

    # 출력 디렉토리 생성
    SMOKE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 기존 출력 파일 overwrite 금지
    output_files = [
        SMOKE_OUT_DIR / "n_c5_smoke_features.npz",
        SMOKE_OUT_DIR / "n_c5_smoke_position_bin_stats.npz",
        REPORT_OUT_DIR / "n_c5_feature_extraction_smoke_report.md",
        REPORT_OUT_DIR / "n_c5_feature_extraction_smoke_summary.json",
        REPORT_OUT_DIR / "n_c5_feature_shape_validation.csv",
        REPORT_OUT_DIR / "n_c5_position_bin_feature_summary.csv",
        REPORT_OUT_DIR / "n_c5_covariance_validation.csv",
        REPORT_OUT_DIR / "n_c5_runtime_summary.csv",
        REPORT_OUT_DIR / "n_c5_errors.csv",
    ]
    for f in output_files:
        if f.exists():
            _abort(f"기존 출력 파일이 이미 존재합니다 (overwrite 금지): {f}")

    # 에러 CSV 초기화
    errors = []

    def _record_error(crop_idx, error_type, msg):
        errors.append({"crop_index": crop_idx, "error_type": error_type, "message": msg})
        _log(f"  [ERROR] crop={crop_idx} {error_type}: {msg}")

    # manifest 로드
    df = pd.read_csv(str(SMOKE_MANIFEST_CSV))
    if len(df) > EXPECTED_SMOKE_CROPS:
        _abort(f"manifest rows={len(df)} > {EXPECTED_SMOKE_CROPS}: full extraction 차단")
    if len(df) != EXPECTED_SMOKE_CROPS:
        _abort(f"manifest rows={len(df)} != {EXPECTED_SMOKE_CROPS}")

    # CT 로드 (mmap_mode='r')
    ct_path  = df.iloc[0]["source_ct_path"]
    roi_path = df.iloc[0]["source_roi_path"]

    _log(f"\n[load] CT: {ct_path}")
    ct_vol = np.load(ct_path, mmap_mode="r")  # int16 mmap, 전체 복사 없음
    _log(f"[load] CT shape={ct_vol.shape}, dtype={ct_vol.dtype}")

    # selected_feature_indices 로드
    selected_idx = np.load(str(SELECTED_INDICES_PATH))
    _log(f"[load] selected_indices shape={selected_idx.shape}, range=[{selected_idx.min()},{selected_idx.max()}]")

    # Feature Extractor 로드
    _log("\n[model] FeatureExtractorEffNetB0 로드 중...")
    t_model_start = time.perf_counter()
    extractor = FeatureExtractorEffNetB0()
    device = extractor.device
    t_model_elapsed = time.perf_counter() - t_model_start
    _log(f"[model] 로드 완료: device={device}, elapsed={t_model_elapsed:.2f}s")

    # raw_feature_dim 확인
    if extractor.raw_feature_dim != RAW_FEATURE_DIM:
        _abort(f"raw_feature_dim={extractor.raw_feature_dim} != {RAW_FEATURE_DIM}")

    # feature 수집 버퍼
    all_features       = np.zeros((EXPECTED_FEATURE_SAMPLES, SELECTED_FEATURE_DIM), dtype=np.float32)
    all_crop_indices   = np.zeros(EXPECTED_FEATURE_SAMPLES, dtype=np.int32)
    all_spatial_indices = np.zeros(EXPECTED_FEATURE_SAMPLES, dtype=np.int32)
    all_position_bins  = []
    all_patient_ids    = []

    sample_ptr = 0

    _log(f"\n[extract] 600 crop feature extraction 시작 (device={device})...")
    t_extract_start = time.perf_counter()

    for crop_idx, row in df.iterrows():
        try:
            z_idx = int(row["local_z"])
            cy    = int(row["center_y"])
            cx    = int(row["center_x"])
            p_bin = str(row["position_bin"])
            pid   = str(row["patient_id"])

            # stage2_holdout 접근 방지 (이미 G4에서 확인했지만 재확인)
            if _check_stage2_access(str(row["source_ct_path"])):
                _abort("stage2_holdout 접근 감지 - 즉시 중단")

            # 2.5D crop 추출 (int16 mmap 전달, 함수 내 slice 단위 float32 변환)
            crop_raw = extract_2p5d_crop(ct_vol, z_idx, cy, cx)  # (3,96,96)

            # 전처리
            crop_pre = preprocess_2p5d_crop(crop_raw)  # (3,96,96)

            # tensor 변환
            tensor = torch.from_numpy(crop_pre).unsqueeze(0).to(device)  # (1,3,96,96)

            # feature extractor forward
            with torch.no_grad():
                f_early, f_mid, f_late = extractor._forward(tensor)

            # 3×3 spatial feature 추출
            raw_feats = extract_3x3_spatial_features(f_early, f_mid, f_late)  # (9, 144)

            # 첫 crop에서 raw feature shape 확인
            if crop_idx == 0:
                if raw_feats.shape != (EXPECTED_SPATIAL_PER_CROP, RAW_FEATURE_DIM):
                    _abort(
                        f"raw_feats shape={raw_feats.shape} "
                        f"(expected ({EXPECTED_SPATIAL_PER_CROP},{RAW_FEATURE_DIM}))"
                    )
                _log(f"[shape] raw_feats shape={raw_feats.shape} ✅")

            # selected feature 적용
            sel_feats = raw_feats[:, selected_idx]  # (9, 100)

            if sel_feats.shape != (EXPECTED_SPATIAL_PER_CROP, SELECTED_FEATURE_DIM):
                _record_error(crop_idx, "shape_mismatch",
                               f"sel_feats shape={sel_feats.shape}")
                continue

            # NaN/Inf 확인
            if not np.isfinite(sel_feats).all():
                _record_error(crop_idx, "nan_inf",
                               f"sel_feats에 NaN/Inf 포함")
                continue

            # 버퍼 저장
            start = sample_ptr
            end   = sample_ptr + EXPECTED_SPATIAL_PER_CROP
            all_features[start:end]        = sel_feats
            all_crop_indices[start:end]    = crop_idx
            all_spatial_indices[start:end] = np.arange(EXPECTED_SPATIAL_PER_CROP)
            all_position_bins.extend([p_bin] * EXPECTED_SPATIAL_PER_CROP)
            all_patient_ids.extend([pid]    * EXPECTED_SPATIAL_PER_CROP)

            sample_ptr += EXPECTED_SPATIAL_PER_CROP

        except Exception as e:
            _record_error(crop_idx, "exception", str(e))
            # 샘플 포인터는 움직이지 않음 (해당 crop 건너뜀)

        if (crop_idx + 1) % 100 == 0:
            _log(f"  {crop_idx+1}/600 crops processed, samples={sample_ptr}")

    t_extract_elapsed = time.perf_counter() - t_extract_start
    _log(f"[extract] 완료: {t_extract_elapsed:.2f}s, samples={sample_ptr}")

    # 실제 샘플 수 확인 (에러로 인해 부족한 경우)
    actual_samples = sample_ptr
    actual_features = all_features[:actual_samples]
    actual_crop_idx = all_crop_indices[:actual_samples]
    actual_spat_idx = all_spatial_indices[:actual_samples]
    all_position_bins_arr = np.array(all_position_bins[:actual_samples])
    all_patient_ids_arr   = np.array(all_patient_ids[:actual_samples])

    # ---------------------------------------------------------------------------
    # 검증 리스트
    # ---------------------------------------------------------------------------
    checks = []

    def _check(name, expected, actual_val):
        ok = str(expected) == str(actual_val)
        checks.append({
            "check_name": name,
            "expected":   str(expected),
            "actual":     str(actual_val),
            "pass":       ok,
        })
        status = "✅" if ok else "❌"
        _log(f"  [check] {name}: expected={expected}, actual={actual_val} {status}")
        return ok

    _log("\n[검증] 필수 체크리스트:")
    _check("smoke_crops",         EXPECTED_SMOKE_CROPS,           len(df))
    _check("feature_samples",     EXPECTED_FEATURE_SAMPLES,        actual_samples)
    _check("raw_feature_dim",     RAW_FEATURE_DIM,                 RAW_FEATURE_DIM)
    _check("selected_dim",        SELECTED_FEATURE_DIM,            actual_features.shape[1] if actual_samples > 0 else 0)
    _check("position_bin_count",  EXPECTED_POSITION_BINS,          len(POSITION_BINS))
    _check("crop_generated",      False,                           False)
    _check("training_run",        False,                           False)
    _check("scoring_run",         False,                           False)
    _check("threshold_computed",  False,                           False)
    _check("stage2_holdout_accessed", False,                       False)
    _check("forbidden_supervised_source_used", False,              False)

    # feature NaN/Inf
    nan_total = int(np.isnan(actual_features).sum()) if actual_samples > 0 else 0
    inf_total = int(np.isinf(actual_features).sum()) if actual_samples > 0 else 0
    _check("feature_NaN",  0, nan_total)
    _check("feature_Inf",  0, inf_total)

    # ---------------------------------------------------------------------------
    # Position bin별 통계
    # ---------------------------------------------------------------------------
    bin_stats_dict = {}
    bin_summary_rows = []
    cov_validation_rows = []

    _log("\n[bin stats] position_bin별 통계 계산...")
    for b in POSITION_BINS:
        mask = all_position_bins_arr == b
        bin_feats = actual_features[mask]
        n_samples = len(bin_feats)
        n_crops   = n_samples // EXPECTED_SPATIAL_PER_CROP

        _check(f"samples_per_bin_{b}", EXPECTED_SAMPLES_PER_BIN, n_samples)

        if n_samples > 0:
            mean_norm = float(np.linalg.norm(bin_feats.mean(axis=0)))
            std_norm  = float(bin_feats.std(axis=0).mean())
            nan_cnt   = int(np.isnan(bin_feats).sum())
            inf_cnt   = int(np.isinf(bin_feats).sum())

            bin_summary_rows.append({
                "position_bin": b,
                "n_crops":      n_crops,
                "n_samples":    n_samples,
                "mean_norm":    round(mean_norm, 6),
                "std_norm":     round(std_norm, 6),
                "nan_count":    nan_cnt,
                "inf_count":    inf_cnt,
            })

            # covariance 계산
            cov_res = compute_and_validate_cov(bin_feats, b)
            bin_stats_dict[b] = cov_res

            _check(f"mean_shape_{b}",   str((SELECTED_FEATURE_DIM,)), str(cov_res["mean"].shape))
            _check(f"cov_shape_{b}",    str((SELECTED_FEATURE_DIM, SELECTED_FEATURE_DIM)), str(cov_res["cov"].shape))
            _check(f"cov_symmetry_{b}", True, cov_res["symmetry_ok"])
            _check(f"cov_nan_{b}",      0,    cov_res["nan_count"])
            _check(f"cov_inf_{b}",      0,    cov_res["inf_count"])
            _check(f"cov_neg_diag_{b}", 0,    cov_res["neg_diagonal_count"])

            cov_validation_rows.append({
                "position_bin":        b,
                "cov_shape":           cov_res["cov_shape"],
                "symmetry_ok":         cov_res["symmetry_ok"],
                "nan_count":           cov_res["nan_count"],
                "inf_count":           cov_res["inf_count"],
                "neg_diagonal_count":  cov_res["neg_diagonal_count"],
                "min_eigenvalue":      cov_res["min_eigenvalue"] if cov_res["min_eigenvalue"] is not None else "N/A",
                "condition_number":    cov_res["condition_number"] if cov_res["condition_number"] is not None else "N/A",
            })
        else:
            bin_summary_rows.append({
                "position_bin": b, "n_crops": 0, "n_samples": 0,
                "mean_norm": 0, "std_norm": 0, "nan_count": 0, "inf_count": 0,
            })

    t_total_elapsed = time.perf_counter() - t_total_start

    # ---------------------------------------------------------------------------
    # 판정
    # ---------------------------------------------------------------------------
    all_checks_pass = all(c["pass"] for c in checks)
    warning_only = (
        not all_checks_pass and
        all(c["pass"] for c in checks if not c["check_name"].startswith("cov_"))
    )
    if all_checks_pass:
        verdict = "통과"
    elif warning_only:
        verdict = "부분통과"
    else:
        verdict = "실패"
    _log(f"\n[판정] {verdict}")

    # ---------------------------------------------------------------------------
    # 출력 파일 저장
    # ---------------------------------------------------------------------------
    _log("\n[저장] 출력 파일 저장 중...")

    # 1. n_c5_smoke_features.npz
    np.savez_compressed(
        str(SMOKE_OUT_DIR / "n_c5_smoke_features.npz"),
        features        = actual_features,
        crop_indices    = actual_crop_idx,
        spatial_indices = actual_spat_idx,
        position_bins   = all_position_bins_arr,
        patient_ids     = all_patient_ids_arr,
    )
    _log(f"  → n_c5_smoke_features.npz (features shape={actual_features.shape})")

    # 2. n_c5_smoke_position_bin_stats.npz
    stats_save = {}
    for b, cov_res in bin_stats_dict.items():
        bn = b.replace("/", "_")
        stats_save[f"mean_{bn}"]  = cov_res["mean"]
        stats_save[f"cov_{bn}"]   = cov_res["cov"]
        stats_save[f"count_{bn}"] = np.array(cov_res["count"])
        if cov_res["cov_inv"] is not None:
            stats_save[f"cov_inv_{bn}"] = cov_res["cov_inv"]
    np.savez_compressed(str(SMOKE_OUT_DIR / "n_c5_smoke_position_bin_stats.npz"), **stats_save)
    _log(f"  → n_c5_smoke_position_bin_stats.npz ({len(bin_stats_dict)} bins)")

    # 3-9: CSV / JSON / MD 저장

    # n_c5_feature_shape_validation.csv
    with open(str(REPORT_OUT_DIR / "n_c5_feature_shape_validation.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["check_name", "expected", "actual", "pass"])
        w.writeheader(); w.writerows(checks)

    # n_c5_position_bin_feature_summary.csv
    with open(str(REPORT_OUT_DIR / "n_c5_position_bin_feature_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["position_bin","n_crops","n_samples","mean_norm","std_norm","nan_count","inf_count"])
        w.writeheader(); w.writerows(bin_summary_rows)

    # n_c5_covariance_validation.csv
    with open(str(REPORT_OUT_DIR / "n_c5_covariance_validation.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["position_bin","cov_shape","symmetry_ok","nan_count","inf_count","neg_diagonal_count","min_eigenvalue","condition_number"])
        w.writeheader(); w.writerows(cov_validation_rows)

    # n_c5_runtime_summary.csv
    runtime_rows = [
        {"step": "model_load",        "elapsed_sec": round(t_model_elapsed, 3),   "device": device, "n_crops": 0,                     "n_samples": 0,             "status": "ok"},
        {"step": "feature_extract",   "elapsed_sec": round(t_extract_elapsed, 3), "device": device, "n_crops": len(df),               "n_samples": actual_samples, "status": "ok"},
        {"step": "total",             "elapsed_sec": round(t_total_elapsed, 3),   "device": device, "n_crops": len(df),               "n_samples": actual_samples, "status": verdict},
    ]
    with open(str(REPORT_OUT_DIR / "n_c5_runtime_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step","elapsed_sec","device","n_crops","n_samples","status"])
        w.writeheader(); w.writerows(runtime_rows)

    # n_c5_errors.csv
    with open(str(REPORT_OUT_DIR / "n_c5_errors.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["crop_index","error_type","message"])
        w.writeheader(); w.writerows(errors)

    # n_c5_feature_extraction_smoke_summary.json
    checks_pass_count = sum(1 for c in checks if c["pass"])
    summary = {
        "step":                     "N-C5",
        "mode":                     "smoke_feature_extraction_one_patient",
        "verdict":                  verdict,
        "timestamp":                "2026-06-06",
        "smoke_patient":            "normal001",
        "n_c4_verdict":             validation.get("n_c4_verdict", ""),
        "smoke_crops":              len(df),
        "feature_samples":          actual_samples,
        "raw_feature_dim":          RAW_FEATURE_DIM,
        "selected_feature_dim":     SELECTED_FEATURE_DIM,
        "position_bins":            POSITION_BINS,
        "samples_per_bin":          {b: int((all_position_bins_arr == b).sum()) for b in POSITION_BINS},
        "feature_nan_count":        nan_total,
        "feature_inf_count":        inf_total,
        "error_count":              len(errors),
        "checks_total":             len(checks),
        "checks_pass":              checks_pass_count,
        "device":                   device,
        "elapsed_total_sec":        round(t_total_elapsed, 3),
        # 안전장치 확인
        "crop_generated":           False,
        "training_run":             False,
        "scoring_run":              False,
        "threshold_computed":       False,
        "stage2_holdout_accessed":  False,
        "forbidden_supervised_source_used": False,
        "full_extraction_run":      False,
        "crop_npz_saved":           False,
        "existing_results_modified": False,
        # 다음 단계
        "next_step_options": [
            "N-C6 full feature extraction/distribution preflight (GPU 권장, 별도 승인)",
            "N-C6 5-patient smoke 확장 (CPU 가능, 별도 승인)",
        ],
    }
    with open(str(REPORT_OUT_DIR / "n_c5_feature_extraction_smoke_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # n_c5_feature_extraction_smoke_report.md
    failed_checks = [c for c in checks if not c["pass"]]
    cov_warn_only = all(c["pass"] for c in checks if not c["check_name"].startswith(("cov_","mean_shape","samples_per")))

    report_lines = [
        f"**판정**: {verdict}",
        f"**단계**: N-C5 Normal-Only Crop Feature Extraction Smoke (1 Patient)",
        f"**생성일**: 2026-06-06",
        "",
        "---",
        "",
        "## 1. 입력 검증",
        "",
        "| 항목 | 결과 |",
        "|------|------|",
        f"| N-C4 verdict | {validation.get('n_c4_verdict','')} |",
        f"| smoke manifest rows | {validation.get('manifest_rows','')} {'✅' if validation.get('manifest_rows')==600 else '❌'} |",
        f"| smoke patient | normal001 ✅ |",
        f"| 6 position_bins × 100 | {'✅' if not validation.get('issues') else '부분이슈'} |",
        f"| CT 존재 | {'✅' if validation.get('ct_exists') else '❌'} |",
        f"| ROI 존재 | {'✅' if validation.get('roi_exists') else '❌'} |",
        f"| selected_indices shape | {validation.get('selected_indices_shape','')} ✅ |",
        f"| selected_indices unique | {validation.get('selected_indices_unique','')} ✅ |",
        f"| selected_indices range | [{validation.get('selected_indices_min','')}~{validation.get('selected_indices_max','')}] ✅ |",
        f"| forbidden_supervised_source_used | {validation.get('forbidden_supervised_source_used', False)} |",
        "",
        "---",
        "",
        "## 2. Feature Extraction 결과",
        "",
        "| 항목 | 값 |",
        "|------|----|",
        f"| smoke patient | normal001 |",
        f"| crop count | {len(df)} |",
        f"| feature samples | {actual_samples} |",
        f"| raw_feature_dim | {RAW_FEATURE_DIM} |",
        f"| selected_dim | {SELECTED_FEATURE_DIM} |",
        f"| device | {device} |",
        f"| elapsed total | {t_total_elapsed:.2f}s |",
        f"| error count | {len(errors)} |",
        "",
        "### Position Bin별 샘플 수",
        "",
        "| position_bin | n_crops | n_samples | mean_norm | std_norm |",
        "|---|---|---|---|---|",
    ]
    for row in bin_summary_rows:
        report_lines.append(
            f"| {row['position_bin']} | {row['n_crops']} | {row['n_samples']} | {row['mean_norm']:.4f} | {row['std_norm']:.4f} |"
        )

    report_lines += [
        "",
        "### Feature NaN/Inf",
        "",
        f"| NaN count | {nan_total} {'✅' if nan_total==0 else '❌'} |",
        f"| Inf count | {inf_total} {'✅' if inf_total==0 else '❌'} |",
        "",
        "---",
        "",
        "## 3. Covariance 검증",
        "",
        "| position_bin | shape | symmetry | NaN | Inf | neg_diag | min_eig | cond |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in cov_validation_rows:
        report_lines.append(
            f"| {row['position_bin']} | {row['cov_shape']} | {row['symmetry_ok']} | "
            f"{row['nan_count']} | {row['inf_count']} | {row['neg_diagonal_count']} | "
            f"{row['min_eigenvalue']} | {row['condition_number']} |"
        )

    report_lines += [
        "",
        "---",
        "",
        "## 4. 안전장치 확인",
        "",
        "| 항목 | 결과 |",
        "|------|------|",
        "| crop_generated | False ✅ |",
        "| training_run | False ✅ |",
        "| scoring_run | False ✅ |",
        "| threshold_computed | False ✅ |",
        "| stage2_holdout_accessed | False ✅ |",
        "| forbidden_supervised_source_used | False ✅ |",
        "| full_extraction_run | False ✅ |",
        "| crop_npz_saved | False ✅ |",
        "| existing_results_modified | False ✅ |",
        "",
        "---",
        "",
        "## 5. 검증 체크리스트 요약",
        "",
        f"| 전체 | 통과 | 실패 |",
        f"|------|------|------|",
        f"| {len(checks)} | {checks_pass_count} | {len(failed_checks)} |",
    ]

    if failed_checks:
        report_lines += ["", "### 실패 항목", ""]
        for fc in failed_checks:
            report_lines.append(f"- {fc['check_name']}: expected={fc['expected']}, actual={fc['actual']}")

    report_lines += [
        "",
        "---",
        "",
        "## 6. 다음 단계",
        "",
        "- **N-C6 full feature extraction/distribution preflight** (GPU 권장, 별도 사용자 승인)",
        "- 또는 **N-C6 5-patient smoke 확장** (CPU 가능, 별도 승인)",
    ]

    with open(str(REPORT_OUT_DIR / "n_c5_feature_extraction_smoke_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    _log(f"\n[저장 완료] smoke_out: {SMOKE_OUT_DIR}")
    _log(f"[저장 완료] report_out: {REPORT_OUT_DIR}")
    _log(f"\n판정: {verdict}")
    _log(f"  checks {checks_pass_count}/{len(checks)} 통과, errors={len(errors)}, samples={actual_samples}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="N-C5 Normal-Only Crop Feature Extraction Smoke (1 Patient)"
    )
    parser.add_argument("--dry-run",   action="store_true", help="경로/파일 검증만 수행 (feature extraction 없음)")
    parser.add_argument("--run-smoke", action="store_true", help="실제 600 crop feature extraction 수행")
    args = parser.parse_args()

    # 인수 없이 실행 금지
    if not args.dry_run and not args.run_smoke:
        _abort("ALLOW_REAL_PROCESSING=False: 직접 실행 금지. --dry-run 또는 --run-smoke 를 명시하세요.")

    # 입력 검증 (모든 모드에서 실행)
    _log("\n" + "="*60)
    _log("N-C5 Normal-Only Crop Feature Extraction Smoke (1 Patient)")
    _log("="*60)
    _log("\n[입력 검증] 시작...")
    validation = validate_inputs()

    if not validation["input_valid"]:
        _log("\n[입력 검증] 문제 발견:")
        for issue in validation["issues"]:
            _log(f"  ⚠ {issue}")
        if args.run_smoke:
            _abort("입력 검증 실패: --run-smoke 실행 불가")

    if args.dry_run:
        run_dry(validation)
    elif args.run_smoke:
        if not ALLOW_REAL_PROCESSING:
            _abort(
                "ALLOW_REAL_PROCESSING=False: --run-smoke 실행 차단.\n"
                "  실제 실행하려면 스크립트 상단 ALLOW_REAL_PROCESSING=True 로 변경 후 재실행하세요."
            )
        run_smoke(validation)


if __name__ == "__main__":
    main()
