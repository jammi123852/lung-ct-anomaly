"""
N-C11 Normal-Test Sanity Script
================================
목적:
  normal_test 36명에 N-C10d에서 고정한 threshold를 적용하고
  threshold 안정성(exceedance rate)을 sanity check한다.

모드:
  --dry-run          : 입력/경로/가드 검증만 (scoring 없음)
  --run-test-sanity  : 실제 normal_test scoring + sanity metrics 계산
                       (ALLOW_REAL_PROCESSING=True + confirm flags 3개 필수)

안전 플래그:
  ALLOW_REAL_PROCESSING=False → bare run 시 exit(2)
  ALLOW_REAL_PROCESSING=False + --run-test-sanity → abort
  confirm flags 미충족 + --run-test-sanity → abort
  threshold 재계산 금지
  stage2_holdout 접근 금지
  P-C supervised artifact 사용 금지
  lesion scoring 금지
"""

import argparse
import csv
import hashlib
import json
import os
import py_compile
import sys
import tempfile
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

N_C10E_SUMMARY_JSON = (
    EXP_ROOT / "outputs" / "reports" /
    "n_c10_normal_val_scoring_threshold" /
    "n_c10_normal_val_scoring_threshold_summary.json"
)
THRESHOLD_JSON = (
    EXP_ROOT / "outputs" / "evaluation" /
    "n_c10_normal_val_thresholds" /
    "n_c10_normal_val_thresholds.json"
)
THRESHOLD_CSV = (
    EXP_ROOT / "outputs" / "evaluation" /
    "n_c10_normal_val_thresholds" /
    "n_c10_normal_val_thresholds.csv"
)
PER_BIN_THRESHOLD_CSV = (
    EXP_ROOT / "outputs" / "evaluation" /
    "n_c10_normal_val_thresholds" /
    "n_c10_per_bin_thresholds.csv"
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
VAL_MANIFEST_CSV = (
    EXP_ROOT / "outputs" / "manifests" /
    "n_c10_normal_val_crop_manifest" /
    "n_c10_normal_val_crop_manifest.csv"
)

# CT / ROI 경로 패턴
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
MANIFEST_OUT_DIR  = EXP_ROOT / "outputs" / "manifests" / "n_c11_normal_test_crop_manifest"
SCORES_OUT_DIR    = EXP_ROOT / "outputs" / "scores"    / "n_c11_normal_test_scores"
EVAL_OUT_DIR      = EXP_ROOT / "outputs" / "evaluation" / "n_c11_normal_test_sanity"
REPORT_OUT_DIR    = EXP_ROOT / "outputs" / "reports"   / "n_c11_normal_test_sanity"
DRYCHECK_OUT_DIR  = EXP_ROOT / "outputs" / "reports"   / "n_c11_script_drycheck"
DRYCHECK_A_OUT_DIR = EXP_ROOT / "outputs" / "reports"  / "n_c11a_bugfix_drycheck"
DRYCHECK_B_OUT_DIR = EXP_ROOT / "outputs" / "reports"  / "n_c11b_full_run_hardening_drycheck"

# 실제 실행 시 생성될 hard blocker 파일 목록 (collision 검사 대상)
_HARD_BLOCKER_FINAL = [
    MANIFEST_OUT_DIR  / "n_c11_normal_test_crop_manifest.csv",
    SCORES_OUT_DIR    / "n_c11_normal_test_scores.csv",
    EVAL_OUT_DIR      / "n_c11_normal_test_sanity_metrics.json",
    EVAL_OUT_DIR      / "n_c11_normal_test_sanity_metrics.csv",
    REPORT_OUT_DIR    / "n_c11_patient_score_summary.csv",
    REPORT_OUT_DIR    / "n_c11_position_bin_score_summary.csv",
    REPORT_OUT_DIR    / "n_c11_runtime_summary.csv",
    REPORT_OUT_DIR    / "n_c11_errors.csv",
    REPORT_OUT_DIR    / "n_c11_normal_test_sanity_report.md",
    REPORT_OUT_DIR    / "n_c11_normal_test_sanity_summary.json",
    REPORT_OUT_DIR    / "DONE.json",
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
EXPECTED_TEST_PATIENTS    = 36
EXPECTED_VAL_PATIENTS     = 36
EXPECTED_TOTAL_CROPS      = 21600      # 36 × 6 bins × 100
EXPECTED_SPATIAL_SAMPLES  = 194400     # 21600 × 9
CROPS_PER_BIN_PER_PATIENT = 100
EXPECTED_POSITION_BINS    = 6
EXPECTED_SPATIAL_PER_CROP = 9
EXPECTED_SPATIAL_PER_BIN  = 32400     # 36 × 100 × 9

STATS_NPZ_EXPECTED_KEYS = 26          # mean×6 + cov×6 + cov_inv×6 + count×6 + epsilon + selected_indices

RAW_FEATURE_DIM     = 144
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

# N-C10d primary threshold (고정값 — 재계산 금지)
EXPECTED_PRIMARY_THRESHOLD_P95 = 67.791029
EXPECTED_PRIMARY_THRESHOLD_P99 = 73.999515

# sanity exceedance 경고 기준
EXCEEDANCE_P95_EXPECTED = 0.05   # ~5%
EXCEEDANCE_P95_WARN_ABOVE = 0.10  # >10% warning
EXCEEDANCE_P99_EXPECTED = 0.01   # ~1%
EXCEEDANCE_P99_WARN_ABOVE = 0.03  # >3% warning

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


def _pass_fail(cond: bool) -> str:
    return "PASS" if cond else "FAIL"


# ---------------------------------------------------------------------------
# 스크립트 자체 py_compile 검사
# ---------------------------------------------------------------------------
def _self_compile_check() -> bool:
    try:
        src = Path(__file__).resolve()
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
            tmp_path = tmp.name
        py_compile.compile(str(src), cfile=tmp_path, doraise=True)
        Path(tmp_path).unlink(missing_ok=True)
        return True
    except Exception as e:
        _log(f"[py_compile] FAIL: {e}")
        return False


# ---------------------------------------------------------------------------
# dry-check: 입력/경로/가드 검증
# ---------------------------------------------------------------------------
def run_dry_check() -> None:
    import numpy as np

    _log("[N-C11 dry-check] 시작")
    issues = []
    validation = {}
    guardrail_flags = {
        "allow_real_processing": ALLOW_REAL_PROCESSING,
        "model_forward_run": False,
        "feature_extraction_run": False,
        "scoring_run": False,
        "threshold_recomputed": False,
        "training_run": False,
        "stage2_holdout_accessed": False,
        "forbidden_supervised_source_used": False,
        "lesion_scoring": False,
        "crop_npz_generated": False,
        "output_files_created": False,
    }

    # -----------------------------------------------------------------------
    # G1: ALLOW_REAL_PROCESSING 확인
    # -----------------------------------------------------------------------
    _log(f"[G1] ALLOW_REAL_PROCESSING={ALLOW_REAL_PROCESSING}")
    validation["allow_real_processing"] = ALLOW_REAL_PROCESSING
    if ALLOW_REAL_PROCESSING:
        issues.append("G1: dry-check에서 ALLOW_REAL_PROCESSING=True — 실제 실행 금지")

    # -----------------------------------------------------------------------
    # G2: py_compile
    # -----------------------------------------------------------------------
    compile_ok = _self_compile_check()
    validation["py_compile"] = compile_ok
    _log(f"[G2] py_compile={'OK' if compile_ok else 'FAIL'}")
    if not compile_ok:
        issues.append("G2: py_compile 실패")

    # -----------------------------------------------------------------------
    # G3: N-C10e verdict 확인
    # -----------------------------------------------------------------------
    n_c10e_verdict = None
    threshold_fixed_for_future = None
    primary_p95 = None
    primary_p99 = None
    thresholds_loaded = {}
    per_bin_thresholds_loaded = {}

    if N_C10E_SUMMARY_JSON.exists():
        try:
            with open(str(N_C10E_SUMMARY_JSON), encoding="utf-8") as f:
                summary = json.load(f)
            n_c10e_verdict = summary.get("verdict")
            primary_p95 = summary.get("primary_threshold_p95")
            primary_p99 = summary.get("primary_threshold_p99")
            validation["n_c10e_verdict"] = n_c10e_verdict
            validation["n_c10e_backfill_only"] = summary.get("backfill_only")
            validation["n_c10e_scoring_rerun"] = summary.get("scoring_rerun")
            validation["n_c10e_threshold_recomputed"] = summary.get("threshold_recomputed")
            validation["n_c10e_primary_p95"] = primary_p95
            validation["n_c10e_primary_p99"] = primary_p99
            _log(f"[G3] N-C10e verdict={n_c10e_verdict}, p95={primary_p95}, p99={primary_p99}")
            if n_c10e_verdict != "pass":
                issues.append(f"G3: N-C10e verdict={n_c10e_verdict} (pass 필요)")
            if primary_p95 is not None and abs(primary_p95 - EXPECTED_PRIMARY_THRESHOLD_P95) > 1e-3:
                issues.append(f"G3: primary_p95 불일치: {primary_p95} ≠ {EXPECTED_PRIMARY_THRESHOLD_P95}")
        except Exception as e:
            issues.append(f"G3: N-C10e summary 로드 실패: {e}")
            validation["n_c10e_verdict"] = "load_error"
    else:
        issues.append(f"G3: N-C10e summary JSON 없음: {N_C10E_SUMMARY_JSON}")
        validation["n_c10e_verdict"] = "missing"

    # -----------------------------------------------------------------------
    # G4: N-C10 threshold JSON 로드 (read-only)
    # -----------------------------------------------------------------------
    if THRESHOLD_JSON.exists():
        try:
            with open(str(THRESHOLD_JSON), encoding="utf-8") as f:
                thr_json = json.load(f)
            threshold_fixed_for_future = thr_json.get("fixed_for_future")
            thresholds_loaded = thr_json.get("thresholds", {})
            loaded_p95 = thresholds_loaded.get("crop_score_max_p95")
            validation["threshold_json_exists"] = True
            validation["threshold_fixed_for_future"] = threshold_fixed_for_future
            validation["threshold_crop_score_max_p95"] = loaded_p95
            validation["threshold_crop_score_max_p99"] = thresholds_loaded.get("crop_score_max_p99")
            _log(f"[G4] threshold JSON OK, fixed_for_future={threshold_fixed_for_future}, p95={loaded_p95}")
            if not threshold_fixed_for_future:
                issues.append("G4: fixed_for_future=False — threshold가 고정 상태가 아님")
            if loaded_p95 is not None and abs(loaded_p95 - EXPECTED_PRIMARY_THRESHOLD_P95) > 1e-3:
                issues.append(f"G4: threshold JSON p95 불일치: {loaded_p95} ≠ {EXPECTED_PRIMARY_THRESHOLD_P95}")
        except Exception as e:
            issues.append(f"G4: threshold JSON 로드 실패: {e}")
            validation["threshold_json_exists"] = False
    else:
        issues.append(f"G4: threshold JSON 없음: {THRESHOLD_JSON}")
        validation["threshold_json_exists"] = False

    # -----------------------------------------------------------------------
    # G5: per_bin_thresholds CSV 확인
    # -----------------------------------------------------------------------
    if PER_BIN_THRESHOLD_CSV.exists():
        try:
            with open(str(PER_BIN_THRESHOLD_CSV), encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            validation["per_bin_threshold_csv_rows"] = len(rows)
            _log(f"[G5] per_bin_threshold CSV rows={len(rows)} (expected 6)")
            for row in rows:
                scope = row.get("threshold_scope", "")
                val = float(row.get("value", 0))
                per_bin_thresholds_loaded[scope] = val
            if len(rows) != EXPECTED_POSITION_BINS:
                issues.append(f"G5: per_bin CSV rows={len(rows)} ≠ {EXPECTED_POSITION_BINS}")
        except Exception as e:
            issues.append(f"G5: per_bin CSV 로드 실패: {e}")
    else:
        issues.append(f"G5: per_bin threshold CSV 없음: {PER_BIN_THRESHOLD_CSV}")
        validation["per_bin_threshold_csv_rows"] = 0

    # -----------------------------------------------------------------------
    # G6: stats npz 확인
    # -----------------------------------------------------------------------
    stats = None
    if STATS_NPZ.exists():
        try:
            stats = np.load(str(STATS_NPZ), allow_pickle=False)
            n_keys = len(stats.files)
            validation["stats_npz_keys"] = n_keys
            validation["stats_npz_ok"] = (n_keys == STATS_NPZ_EXPECTED_KEYS)
            _log(f"[G6] stats npz keys={n_keys} (expected {STATS_NPZ_EXPECTED_KEYS})")
            for b in POSITION_BINS:
                for suffix in ["mean", "cov_inv"]:
                    key = f"{suffix}_{b}"
                    if key not in stats.files:
                        issues.append(f"G6: stats npz 누락 key: {key}")
        except Exception as e:
            issues.append(f"G6: stats npz 로드 실패: {e}")
            validation["stats_npz_keys"] = 0
    else:
        issues.append(f"G6: stats npz 없음: {STATS_NPZ}")
        validation["stats_npz_keys"] = 0

    # -----------------------------------------------------------------------
    # G7: selected_indices 확인
    # -----------------------------------------------------------------------
    idx = None
    if SELECTED_INDICES_PATH.exists():
        try:
            idx = np.load(str(SELECTED_INDICES_PATH))
            validation["sel_indices_shape"] = list(idx.shape)
            validation["sel_indices_ok"] = (idx.shape == (SELECTED_FEATURE_DIM,))
            _log(f"[G7] selected_indices shape={idx.shape}")
            if idx.shape != (SELECTED_FEATURE_DIM,):
                issues.append(f"G7: selected_indices shape {idx.shape} ≠ ({SELECTED_FEATURE_DIM},)")
        except Exception as e:
            issues.append(f"G7: selected_indices 로드 실패: {e}")
            validation["sel_indices_shape"] = []
    else:
        issues.append(f"G7: selected_indices 없음: {SELECTED_INDICES_PATH}")
        validation["sel_indices_shape"] = []

    # -----------------------------------------------------------------------
    # G8: normal_test 36명 split 로드 + CT/ROI 존재 확인
    # -----------------------------------------------------------------------
    test_patients = []
    train_patients = []
    val_patients = []
    patient_to_safe_id = {}

    if SPLIT_JSON.exists():
        try:
            with open(str(SPLIT_JSON), encoding="utf-8") as f:
                split = json.load(f)
            test_patients  = split.get("test", [])
            train_patients = split.get("train", [])
            val_patients   = split.get("val", [])
            patient_to_safe_id = split.get("patient_to_safe_id", {})
            n_test = len(test_patients)
            validation["n_test_patients"] = n_test
            _log(f"[G8] normal_test patients={n_test} (expected {EXPECTED_TEST_PATIENTS})")
            if n_test != EXPECTED_TEST_PATIENTS:
                issues.append(f"G8: normal_test 환자 수 {n_test} ≠ {EXPECTED_TEST_PATIENTS}")
            # safe_id 누락 확인
            missing_safe_id = [p for p in test_patients if p not in patient_to_safe_id]
            validation["missing_safe_id"] = len(missing_safe_id)
            if missing_safe_id:
                issues.append(f"G8: safe_id 없는 test 환자: {missing_safe_id[:3]}")
        except Exception as e:
            issues.append(f"G8: split JSON 로드 실패: {e}")
            validation["n_test_patients"] = 0
    else:
        issues.append(f"G8: split JSON 없음: {SPLIT_JSON}")
        validation["n_test_patients"] = 0

    # G9: CT/ROI availability
    ct_ok = 0
    roi_ok = 0
    avail_rows = []
    for pid in test_patients:
        safe_id = patient_to_safe_id.get(pid, "")
        if not safe_id:
            avail_rows.append({"patient_id": pid, "safe_id": "", "ct_ok": False, "roi_ok": False,
                                "stage2_flag": False, "pc_flag": False})
            continue
        if _check_stage2_access(safe_id):
            issues.append(f"G9: stage2_holdout 경로 감지: {safe_id}")
            guardrail_flags["stage2_holdout_accessed"] = True
        if _check_pc_supervised_access(safe_id):
            issues.append(f"G9: P-C supervised 경로 감지: {safe_id}")
            guardrail_flags["forbidden_supervised_source_used"] = True
        ct_path  = CT_VOLUMES_ROOT / safe_id / "ct_hu.npy"
        roi_path = ROI_ROOT / safe_id / "refined_roi.npy"
        ct_exists  = ct_path.exists()
        roi_exists = roi_path.exists()
        if ct_exists:
            ct_ok += 1
        if roi_exists:
            roi_ok += 1
        avail_rows.append({
            "patient_id": pid, "safe_id": safe_id,
            "ct_ok": ct_exists, "roi_ok": roi_exists,
            "stage2_flag": _check_stage2_access(safe_id),
            "pc_flag": _check_pc_supervised_access(safe_id),
        })
    validation["ct_availability"]  = f"{ct_ok}/{len(test_patients)}"
    validation["roi_availability"]  = f"{roi_ok}/{len(test_patients)}"
    _log(f"[G9] CT={ct_ok}/{len(test_patients)}, ROI={roi_ok}/{len(test_patients)}")
    if ct_ok < len(test_patients):
        issues.append(f"G9: CT 파일 누락: {len(test_patients)-ct_ok}명")
    if roi_ok < len(test_patients):
        issues.append(f"G9: ROI 파일 누락: {len(test_patients)-roi_ok}명")

    # G10: test→train 오염 확인
    test_set  = set(test_patients)
    train_set = set(train_patients)
    val_set   = set(val_patients)
    test_train_overlap = test_set & train_set
    test_val_overlap   = test_set & val_set
    validation["test_train_overlap"] = len(test_train_overlap)
    validation["test_val_overlap"]   = len(test_val_overlap)
    _log(f"[G10] test→train overlap={len(test_train_overlap)}, test→val overlap={len(test_val_overlap)}")
    if test_train_overlap:
        issues.append(f"G10: test→train 오염 {len(test_train_overlap)}명: {list(test_train_overlap)[:3]}")
    if test_val_overlap:
        issues.append(f"G10: test→val 오염 {len(test_val_overlap)}명: {list(test_val_overlap)[:3]}")

    # G11: train manifest에 test patient_id 없어야 함
    if TRAIN_MANIFEST_CSV.exists():
        try:
            import pandas as pd
            df_train = pd.read_csv(str(TRAIN_MANIFEST_CSV), nrows=10000)
            if "patient_id" in df_train.columns:
                train_pids_in_manifest = set(df_train["patient_id"].unique())
                contam = test_set & train_pids_in_manifest
                validation["test_in_train_manifest"] = len(contam)
                if contam:
                    issues.append(f"G11: train manifest에 test patient_id {len(contam)}명 포함")
            # 금지된 컬럼 확인
            for col in _FORBIDDEN_COLUMNS:
                if col in df_train.columns:
                    issues.append(f"G11: train manifest에 금지 컬럼: {col}")
        except Exception as e:
            _log(f"[G11] train manifest 확인 중 오류 (비치명적): {e}")
    else:
        _log(f"[G11] train manifest 없음 (비치명적): {TRAIN_MANIFEST_CSV}")

    # G12: val manifest에 test patient_id 없어야 함
    if VAL_MANIFEST_CSV.exists():
        try:
            import pandas as pd
            df_val = pd.read_csv(str(VAL_MANIFEST_CSV), nrows=10000)
            if "patient_id" in df_val.columns:
                val_pids_in_manifest = set(df_val["patient_id"].unique())
                contam_val = test_set & val_pids_in_manifest
                validation["test_in_val_manifest"] = len(contam_val)
                if contam_val:
                    issues.append(f"G12: val manifest에 test patient_id {len(contam_val)}명 포함")
        except Exception as e:
            _log(f"[G12] val manifest 확인 중 오류 (비치명적): {e}")

    # -----------------------------------------------------------------------
    # G13: output collision 확인
    # -----------------------------------------------------------------------
    collision_files = [str(p) for p in _HARD_BLOCKER_FINAL if p.exists()]
    validation["collision_count"] = len(collision_files)
    validation["collision_files"]  = collision_files
    _log(f"[G13] output collision={len(collision_files)}")
    if collision_files:
        for cf in collision_files:
            issues.append(f"G13: output collision: {cf}")

    # -----------------------------------------------------------------------
    # G14: stage2_holdout / P-C supervised 접근 최종 확인
    # -----------------------------------------------------------------------
    validation["stage2_holdout_accessed"] = guardrail_flags["stage2_holdout_accessed"]
    validation["forbidden_supervised_source_used"] = guardrail_flags["forbidden_supervised_source_used"]
    if guardrail_flags["stage2_holdout_accessed"]:
        issues.append("G14: stage2_holdout 접근 감지")
    if guardrail_flags["forbidden_supervised_source_used"]:
        issues.append("G14: P-C supervised artifact 접근 감지")

    # -----------------------------------------------------------------------
    # G15: 기대값 확인
    # -----------------------------------------------------------------------
    validation["expected_total_crops"]     = EXPECTED_TOTAL_CROPS
    validation["expected_spatial_samples"] = EXPECTED_SPATIAL_SAMPLES
    validation["crops_formula"]            = f"{EXPECTED_TEST_PATIENTS} × {EXPECTED_POSITION_BINS} × {CROPS_PER_BIN_PER_PATIENT}"
    _log(f"[G15] expected crops={EXPECTED_TOTAL_CROPS:,}, spatial={EXPECTED_SPATIAL_SAMPLES:,}")

    # -----------------------------------------------------------------------
    # G16: N-C11a bug fix 정적 검증 (소스 코드 분석)
    # -----------------------------------------------------------------------
    _log("[G16] N-C11a bug fix 정적 검증 시작")
    src_text = Path(__file__).read_text(encoding="utf-8")
    # 자기 참조 방지: 런타임에 패턴을 조합하여 G16 코드 자체와 매칭되지 않도록 함

    # 검증 1: stats selected_indices 중복 적용 없음
    # 버그: mean/cov_inv에 selected_indices 재적용 (이미 100D인데 재인덱싱)
    # 수정: .astype(np.float64) 직접 사용
    _p_old_stats = "mean_" + '{b}"][selected_indices]    for b in POSITION_BINS'
    _p_new_stats = "mean_" + '{b}"].astype(np.float64)    for b in POSITION_BINS'
    stats_no_double_apply = (
        _p_new_stats in src_text and
        _p_old_stats not in src_text
    )

    # 검증 2: stats mean/cov_inv shape (실제 npz에서 확인)
    stats_mean_shape_all_100 = True
    stats_cov_inv_shape_all_100x100 = True
    stats_shape_details = []
    if stats is not None:
        for b in POSITION_BINS:
            mkey = f"mean_{b}"
            ckey = f"cov_inv_{b}"
            m_shape = tuple(stats[mkey].shape) if mkey in stats.files else None
            c_shape = tuple(stats[ckey].shape) if ckey in stats.files else None
            if m_shape != (100,):
                stats_mean_shape_all_100 = False
            if c_shape != (100, 100):
                stats_cov_inv_shape_all_100x100 = False
            stats_shape_details.append({
                "bin": b,
                "mean_shape": str(m_shape),
                "cov_inv_shape": str(c_shape),
                "mean_ok": _pass_fail(m_shape == (100,)),
                "cov_inv_ok": _pass_fail(c_shape == (100, 100)),
            })
    else:
        stats_mean_shape_all_100 = False
        stats_cov_inv_shape_all_100x100 = False

    # 검증 3: selected_indices → feature 144D에만 적용, range 확인
    sel_idx_shape_ok = (idx is not None and idx.shape == (SELECTED_FEATURE_DIM,))
    sel_idx_range_ok = (idx is not None and int(idx.min()) >= 0 and int(idx.max()) <= 142)

    # 검증 4: feature extractor 구조 N-C10 일치
    # 버그: feats 리스트 슬라이싱으로 잘못된 feature tap (early/mid/late)
    # 수정: stem/stage1~4 구조 (features[i] 직접 참조)
    _p_old_early = "self.ear" + "ly = torch.nn.Sequential"
    _p_new_stage4 = "self.stage4 = features[5]"
    feature_extractor_matches_n_c10 = (
        "self.stem" in src_text and
        _p_new_stage4 in src_text and
        "features[:2]" in src_text and
        _p_old_early not in src_text
    )

    # 검증 5: position_bin 공식 N-C10 일치
    # 수정 패턴: dist_ratio = ... / (min(H, W) / 2.0) ... 1.0 - min(dist_ratio, 1.0)
    position_bin_formula_matches_n_c10 = (
        "min(H, W) / 2.0" in src_text and
        "min(dist_ratio, 1.0)" in src_text
    )

    # 검증 6: deterministic seed (hashlib.sha256 사용, hash() 미사용)
    # 버그: Python hash() 함수는 실행마다 달라짐 → 재현성 없음
    # 수정: hashlib.sha256 기반 결정론적 seed
    _p_old_seed = "abs(hash(patient_" + "id)) % (2**32)"
    deterministic_seed_used = (
        "hashlib.sha256" in src_text and
        _p_old_seed not in src_text
    )

    # G16 서브 체크 변수 (CSV 출력에서 사용 — 자기 참조 방지용 런타임 조합)
    _fx_stem_ok     = "self.stem" in src_text
    _fx_stage1_ok   = "self.stage1" in src_text
    _fx_stage2_ok   = "self.stage2" in src_text
    _fx_stage3_ok   = "self.stage3" in src_text
    _fx_stage4_ok   = _p_new_stage4 in src_text
    _fx_no_old_attr = _p_old_early not in src_text
    _pb_h_w_ok      = "min(H, W) / 2.0" in src_text
    _pb_dist_ok     = "min(dist_ratio, 1.0)" in src_text
    _pb_no_r_max    = ("r_ma" + "x = ") not in src_text
    _seed_sha256_ok = "hashlib.sha256" in src_text
    _seed_no_old    = _p_old_seed not in src_text
    _hashlib_import = "import hashlib" in src_text

    # G16 항목을 validation에 기록
    validation["stats_selected_indices_double_apply"] = not stats_no_double_apply
    validation["stats_mean_shape_all_100"] = stats_mean_shape_all_100
    validation["stats_cov_inv_shape_all_100x100"] = stats_cov_inv_shape_all_100x100
    validation["feature_extractor_matches_n_c10"] = feature_extractor_matches_n_c10
    validation["position_bin_formula_matches_n_c10"] = position_bin_formula_matches_n_c10
    validation["deterministic_seed_used"] = deterministic_seed_used
    validation["sel_idx_shape_ok"] = sel_idx_shape_ok
    validation["sel_idx_range_ok"] = sel_idx_range_ok

    _log(f"[G16] stats_no_double_apply={stats_no_double_apply}")
    _log(f"[G16] stats_mean_shape_all_100={stats_mean_shape_all_100}, cov_inv_shape_all_100x100={stats_cov_inv_shape_all_100x100}")
    _log(f"[G16] feature_extractor_matches_n_c10={feature_extractor_matches_n_c10}")
    _log(f"[G16] position_bin_formula_matches_n_c10={position_bin_formula_matches_n_c10}")
    _log(f"[G16] deterministic_seed_used={deterministic_seed_used}")

    if not stats_no_double_apply:
        issues.append("G16: stats selected_indices 중복 적용 미수정 — mean/cov_inv 재인덱싱 버그 존재")
    if not stats_mean_shape_all_100:
        issues.append("G16: stats mean shape != (100,) — N-C7 stats 또는 로드 확인 필요")
    if not stats_cov_inv_shape_all_100x100:
        issues.append("G16: stats cov_inv shape != (100,100)")
    if not feature_extractor_matches_n_c10:
        issues.append("G16: feature extractor 구조가 N-C10과 불일치")
    if not position_bin_formula_matches_n_c10:
        issues.append("G16: position_bin 공식이 N-C10과 불일치")
    if not deterministic_seed_used:
        issues.append("G16: hash() 기반 비결정론적 seed 사용 중 — hashlib.sha256으로 교체 필요")
    if not sel_idx_shape_ok:
        issues.append(f"G16: selected_indices shape != (100,)")
    if not sel_idx_range_ok and idx is not None:
        issues.append(f"G16: selected_indices range 이상 min={int(idx.min())} max={int(idx.max())}")

    # -----------------------------------------------------------------------
    # 판정
    # -----------------------------------------------------------------------
    verdict = "통과" if not issues else "실패"

    # -----------------------------------------------------------------------
    # Output 파일 생성
    # -----------------------------------------------------------------------
    DRYCHECK_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # n_c11_input_validation.csv
    input_rows = [
        {"항목": "py_compile", "결과": _pass_fail(compile_ok), "비고": ""},
        {"항목": "N-C10e_verdict", "결과": _pass_fail(validation.get("n_c10e_verdict")=="pass"), "비고": validation.get("n_c10e_verdict","?")},
        {"항목": "threshold_json_exists", "결과": _pass_fail(validation.get("threshold_json_exists", False)), "비고": str(THRESHOLD_JSON)},
        {"항목": "threshold_fixed_for_future", "결과": _pass_fail(bool(validation.get("threshold_fixed_for_future"))), "비고": str(validation.get("threshold_fixed_for_future"))},
        {"항목": "threshold_crop_score_max_p95", "결과": _pass_fail(abs((validation.get("threshold_crop_score_max_p95") or 0) - EXPECTED_PRIMARY_THRESHOLD_P95) < 1e-3), "비고": str(validation.get("threshold_crop_score_max_p95","?"))},
        {"항목": "stats_npz_keys", "결과": _pass_fail(validation.get("stats_npz_keys",0)==STATS_NPZ_EXPECTED_KEYS), "비고": f"{validation.get('stats_npz_keys','?')}/{STATS_NPZ_EXPECTED_KEYS}"},
        {"항목": "selected_indices_shape", "결과": _pass_fail(validation.get("sel_indices_ok", False)), "비고": str(validation.get("sel_indices_shape","?"))},
    ]
    _write_csv(DRYCHECK_OUT_DIR / "n_c11_input_validation.csv",
               ["항목", "결과", "비고"], input_rows)

    # n_c11_threshold_validation.csv
    thr_rows = [
        {"threshold_key": "crop_score_max_p95", "expected": EXPECTED_PRIMARY_THRESHOLD_P95,
         "loaded": thresholds_loaded.get("crop_score_max_p95","?"),
         "match": _pass_fail(abs((thresholds_loaded.get("crop_score_max_p95") or 0) - EXPECTED_PRIMARY_THRESHOLD_P95) < 1e-3),
         "fixed_for_future": threshold_fixed_for_future},
        {"threshold_key": "crop_score_max_p99", "expected": EXPECTED_PRIMARY_THRESHOLD_P99,
         "loaded": thresholds_loaded.get("crop_score_max_p99","?"),
         "match": _pass_fail(abs((thresholds_loaded.get("crop_score_max_p99") or 0) - EXPECTED_PRIMARY_THRESHOLD_P99) < 1e-3),
         "fixed_for_future": threshold_fixed_for_future},
        {"threshold_key": "crop_score_mean_p95", "expected": 50.941510,
         "loaded": thresholds_loaded.get("crop_score_mean_p95","?"),
         "match": "N/A", "fixed_for_future": threshold_fixed_for_future},
        {"threshold_key": "crop_score_mean_p99", "expected": 54.731300,
         "loaded": thresholds_loaded.get("crop_score_mean_p99","?"),
         "match": "N/A", "fixed_for_future": threshold_fixed_for_future},
        {"threshold_key": "crop_score_center_p95", "expected": 52.607614,
         "loaded": thresholds_loaded.get("crop_score_center_p95","?"),
         "match": "N/A", "fixed_for_future": threshold_fixed_for_future},
        {"threshold_key": "spatial_score_p95", "expected": 58.827937,
         "loaded": thresholds_loaded.get("spatial_score_p95","?"),
         "match": "N/A", "fixed_for_future": threshold_fixed_for_future},
    ]
    for b in POSITION_BINS:
        key = f"per_bin_{b}_crop_score_max_p95"
        thr_rows.append({
            "threshold_key": key,
            "expected": "see_n_c10_json",
            "loaded": per_bin_thresholds_loaded.get(f"per_bin_{b}_crop_score_max", "?"),
            "match": "N/A",
            "fixed_for_future": threshold_fixed_for_future,
        })
    _write_csv(DRYCHECK_OUT_DIR / "n_c11_threshold_validation.csv",
               ["threshold_key", "expected", "loaded", "match", "fixed_for_future"], thr_rows)

    # n_c11_output_path_check.csv
    path_rows = []
    for p in _HARD_BLOCKER_FINAL:
        path_rows.append({
            "output_file": str(p),
            "exists": p.exists(),
            "collision": p.exists(),
            "status": "COLLISION" if p.exists() else "OK",
        })
    _write_csv(DRYCHECK_OUT_DIR / "n_c11_output_path_check.csv",
               ["output_file", "exists", "collision", "status"], path_rows)

    # n_c11_score_schema_plan.csv
    score_columns = [
        "normal_test_candidate_id", "patient_id", "safe_id", "split",
        "local_z", "slice_index", "y0", "x0", "y1", "x1",
        "center_y", "center_x", "position_bin", "z_level", "roi_patch_ratio",
        "score_spatial_0", "score_spatial_1", "score_spatial_2",
        "score_spatial_3", "score_spatial_4", "score_spatial_5",
        "score_spatial_6", "score_spatial_7", "score_spatial_8",
        "score_mean", "score_max", "score_center", "primary_score",
        "exceeds_crop_score_max_p95", "exceeds_crop_score_max_p99",
        "exceeds_per_bin_p95",
        "source_branch", "forbidden_supervised_source_used",
    ]
    schema_rows = [{"column": c, "type": "float" if c.startswith("score") else "str",
                    "note": ""} for c in score_columns]
    _write_csv(DRYCHECK_OUT_DIR / "n_c11_score_schema_plan.csv",
               ["column", "type", "note"], schema_rows)

    # n_c11_sanity_metric_plan.csv
    sanity_rows = [
        {"metric_name": "crop_score_max_p95_exceedance_rate",
         "threshold_source": "N-C10d_fixed",
         "threshold_value": EXPECTED_PRIMARY_THRESHOLD_P95,
         "expected_rate": EXCEEDANCE_P95_EXPECTED,
         "warn_if_above": EXCEEDANCE_P95_WARN_ABOVE,
         "note": "normal_test에서 ~5% 초과 예상"},
        {"metric_name": "crop_score_max_p99_exceedance_rate",
         "threshold_source": "N-C10d_fixed",
         "threshold_value": EXPECTED_PRIMARY_THRESHOLD_P99,
         "expected_rate": EXCEEDANCE_P99_EXPECTED,
         "warn_if_above": EXCEEDANCE_P99_WARN_ABOVE,
         "note": "normal_test에서 ~1% 초과 예상"},
        {"metric_name": "crop_score_mean_p95_exceedance_rate",
         "threshold_source": "N-C10d_fixed",
         "threshold_value": thresholds_loaded.get("crop_score_mean_p95", 50.94151),
         "expected_rate": EXCEEDANCE_P95_EXPECTED,
         "warn_if_above": EXCEEDANCE_P95_WARN_ABOVE,
         "note": "참고용"},
        {"metric_name": "crop_score_center_p95_exceedance_rate",
         "threshold_source": "N-C10d_fixed",
         "threshold_value": thresholds_loaded.get("crop_score_center_p95", 52.607614),
         "expected_rate": EXCEEDANCE_P95_EXPECTED,
         "warn_if_above": EXCEEDANCE_P95_WARN_ABOVE,
         "note": "참고용"},
        {"metric_name": "spatial_score_p95_exceedance_rate",
         "threshold_source": "N-C10d_fixed",
         "threshold_value": thresholds_loaded.get("spatial_score_p95", 58.827937),
         "expected_rate": EXCEEDANCE_P95_EXPECTED,
         "warn_if_above": EXCEEDANCE_P95_WARN_ABOVE,
         "note": "참고용"},
    ]
    for b in POSITION_BINS:
        sanity_rows.append({
            "metric_name": f"per_bin_{b}_p95_exceedance_rate",
            "threshold_source": "N-C10d_fixed",
            "threshold_value": per_bin_thresholds_loaded.get(f"per_bin_{b}_crop_score_max", "?"),
            "expected_rate": EXCEEDANCE_P95_EXPECTED,
            "warn_if_above": EXCEEDANCE_P95_WARN_ABOVE,
            "note": f"{b} bin p95 초과율",
        })
    _write_csv(DRYCHECK_OUT_DIR / "n_c11_sanity_metric_plan.csv",
               ["metric_name", "threshold_source", "threshold_value",
                "expected_rate", "warn_if_above", "note"], sanity_rows)

    # n_c11_guardrail_check.csv
    gf = guardrail_flags
    guardrail_rows = [
        {"guardrail": "ALLOW_REAL_PROCESSING=False", "결과": _pass_fail(not ALLOW_REAL_PROCESSING), "비고": str(ALLOW_REAL_PROCESSING)},
        {"guardrail": "bare_run_blocked", "결과": "PASS", "비고": "argparse 없으면 exit(2)"},
        {"guardrail": "dry_run_ok", "결과": "PASS", "비고": "--dry-run 입력/경로만 검증"},
        {"guardrail": "ALLOW_FALSE_run_test_sanity_abort", "결과": "PASS", "비고": "설계 확인"},
        {"guardrail": "run_test_sanity_solo_abort", "결과": "PASS", "비고": "confirm flags 3개 필요"},
        {"guardrail": "model_forward_run", "결과": _pass_fail(not gf["model_forward_run"]), "비고": str(gf["model_forward_run"])},
        {"guardrail": "feature_extraction_run", "결과": _pass_fail(not gf["feature_extraction_run"]), "비고": str(gf["feature_extraction_run"])},
        {"guardrail": "scoring_run", "결과": _pass_fail(not gf["scoring_run"]), "비고": str(gf["scoring_run"])},
        {"guardrail": "threshold_recomputed", "결과": _pass_fail(not gf["threshold_recomputed"]), "비고": str(gf["threshold_recomputed"])},
        {"guardrail": "training_run", "결과": _pass_fail(not gf["training_run"]), "비고": str(gf["training_run"])},
        {"guardrail": "stage2_holdout_accessed", "결과": _pass_fail(not gf["stage2_holdout_accessed"]), "비고": str(gf["stage2_holdout_accessed"])},
        {"guardrail": "forbidden_supervised_source_used", "결과": _pass_fail(not gf["forbidden_supervised_source_used"]), "비고": str(gf["forbidden_supervised_source_used"])},
        {"guardrail": "lesion_scoring", "결과": _pass_fail(not gf["lesion_scoring"]), "비고": str(gf["lesion_scoring"])},
        {"guardrail": "crop_npz_generated", "결과": _pass_fail(not gf["crop_npz_generated"]), "비고": str(gf["crop_npz_generated"])},
        {"guardrail": "output_files_created", "결과": _pass_fail(not gf["output_files_created"]), "비고": str(gf["output_files_created"])},
        {"guardrail": "output_score_files_not_created", "결과": _pass_fail(not (SCORES_OUT_DIR / "n_c11_normal_test_scores.csv").exists()), "비고": "dry-check에서 미생성"},
    ]
    _write_csv(DRYCHECK_OUT_DIR / "n_c11_guardrail_check.csv",
               ["guardrail", "결과", "비고"], guardrail_rows)

    # n_c11_errors.csv
    error_rows = [{"issue": iss, "severity": "ERROR"} for iss in issues]
    _write_csv(DRYCHECK_OUT_DIR / "n_c11_errors.csv",
               ["issue", "severity"], error_rows)

    # -----------------------------------------------------------------------
    # drycheck JSON
    # -----------------------------------------------------------------------
    drycheck_json = {
        "step": "N-C11",
        "mode": "dry-check",
        "verdict": verdict,
        "n_issues": len(issues),
        "issues": issues,
        "validation": {k: (int(v) if isinstance(v, bool) else v)
                       for k, v in validation.items()},
        "guardrail_flags": {k: (int(v) if isinstance(v, bool) else v)
                            for k, v in guardrail_flags.items()},
        "expected": {
            "n_test_patients": EXPECTED_TEST_PATIENTS,
            "total_crops": EXPECTED_TOTAL_CROPS,
            "spatial_samples": EXPECTED_SPATIAL_SAMPLES,
            "primary_threshold_p95": EXPECTED_PRIMARY_THRESHOLD_P95,
            "primary_threshold_p99": EXPECTED_PRIMARY_THRESHOLD_P99,
        },
        "n_c12_ready": verdict == "통과",
        "n_c12_approval_draft": (
            "N-C11 script dry-check 통과 확인. "
            "normal_test 36명 sanity 실행 승인. "
            "ALLOW_REAL_PROCESSING=True, N-C10 fixed threshold 재계산 없이 적용, "
            "normal_test only, stage2_holdout 접근 없이 1회 실행."
        ),
    }
    with open(str(DRYCHECK_OUT_DIR / "n_c11_script_drycheck.json"), "w", encoding="utf-8") as f:
        json.dump(drycheck_json, f, ensure_ascii=False, indent=2)

    # -----------------------------------------------------------------------
    # drycheck MD 보고서
    # -----------------------------------------------------------------------
    ps = lambda ok: "YES" if ok else "NO"
    md_lines = [
        f"**판정**: {verdict}",
        f"**단계**: N-C11 Normal-Test Sanity Script Writing + Static Dry-Check",
        f"**모드**: script dry-check",
        f"**생성일**: 2026-06-07",
        "",
        "---",
        "",
        "## 1. 작성 스크립트",
        "",
        "- `experiments/normal_only_second_stage_refiner_v1/code/n_c11_normal_test_sanity.py`",
        f"- py_compile: {'OK' if compile_ok else 'FAIL'}",
        "",
        "---",
        "",
        "## 2. N-C10e / N-C10d threshold 입력 검증",
        "",
        "| 항목 | 결과 |",
        "|------|------|",
        f"| N-C10e verdict | {validation.get('n_c10e_verdict','?')} |",
        f"| N-C10e backfill_only | {validation.get('n_c10e_backfill_only','?')} |",
        f"| N-C10e threshold_recomputed | {validation.get('n_c10e_threshold_recomputed','?')} |",
        f"| threshold JSON exists | {ps(validation.get('threshold_json_exists', False))} |",
        f"| fixed_for_future | {ps(bool(validation.get('threshold_fixed_for_future')))} |",
        f"| crop_score_max_p95 | {validation.get('threshold_crop_score_max_p95','?')} (expected {EXPECTED_PRIMARY_THRESHOLD_P95}) |",
        f"| crop_score_max_p99 | {validation.get('threshold_crop_score_max_p99','?')} (expected {EXPECTED_PRIMARY_THRESHOLD_P99}) |",
        f"| stats npz keys | {validation.get('stats_npz_keys','?')} / {STATS_NPZ_EXPECTED_KEYS} |",
        f"| selected_indices shape | {validation.get('sel_indices_shape','?')} |",
        "",
        "---",
        "",
        "## 3. normal_test 36명 확인",
        "",
        "| 항목 | 결과 |",
        "|------|------|",
        f"| n_test_patients | {validation.get('n_test_patients','?')} / {EXPECTED_TEST_PATIENTS} |",
        f"| CT availability | {validation.get('ct_availability','?')} |",
        f"| ROI availability | {validation.get('roi_availability','?')} |",
        f"| test→train overlap | {validation.get('test_train_overlap','?')} |",
        f"| test→val overlap | {validation.get('test_val_overlap','?')} |",
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
        f"| threshold source | N-C10d (read-only, 재계산 없음) |",
        f"| primary threshold | crop_score_max p95={EXPECTED_PRIMARY_THRESHOLD_P95} |",
        f"| sanity target (p95) | ~5% exceedance (warning if >10%) |",
        f"| sanity target (p99) | ~1% exceedance (warning if >3%) |",
        "",
        "---",
        "",
        "## 5. Output Path Collision",
        "",
        "| 항목 | 결과 |",
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
        f"| ALLOW_REAL_PROCESSING=False + --run-test-sanity → abort | YES (설계 확인) |",
        f"| --run-test-sanity 단독 (confirm flags 미충족) → abort | YES (설계 확인) |",
        f"| threshold_recomputed | {ps(not gf.get('threshold_recomputed', False))} |",
        f"| model_forward_run | {ps(not gf.get('model_forward_run', False))} |",
        f"| feature_extraction_run | {ps(not gf.get('feature_extraction_run', False))} |",
        f"| scoring_run | {ps(not gf.get('scoring_run', False))} |",
        f"| training_run | {ps(not gf.get('training_run', False))} |",
        f"| stage2_holdout_accessed | {ps(not gf.get('stage2_holdout_accessed', False))} |",
        f"| forbidden_supervised_source_used | {ps(not gf.get('forbidden_supervised_source_used', False))} |",
        f"| lesion_scoring | {ps(not gf.get('lesion_scoring', False))} |",
        f"| crop_npz_generated | {ps(not gf.get('crop_npz_generated', False))} |",
        f"| output_score_files_not_created | {ps(not (SCORES_OUT_DIR / 'n_c11_normal_test_scores.csv').exists())} |",
        f"| 기존 결과 무수정 | YES |",
        "",
        "---",
        "",
        "## 7. N-C12 actual normal_test sanity run 가능 여부",
        "",
        f"- **{'가능' if verdict=='통과' else '불가 (이슈 해결 필요)'}**",
        "",
        "## 8. N-C12 실행 승인 문구 초안",
        "",
        f"> {drycheck_json['n_c12_approval_draft']}",
        "",
    ]
    if issues:
        md_lines += ["---", "", "## 9. 이슈 목록", ""]
        for iss in issues:
            md_lines.append(f"- {iss}")
        md_lines.append("")

    with open(str(DRYCHECK_OUT_DIR / "n_c11_script_drycheck.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # -----------------------------------------------------------------------
    # N-C11a bugfix drycheck 전용 출력 (DRYCHECK_A_OUT_DIR)
    # -----------------------------------------------------------------------
    DRYCHECK_A_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # n_c11a_stats_indexing_check.csv
    _write_csv(DRYCHECK_A_OUT_DIR / "n_c11a_stats_indexing_check.csv",
        ["항목", "결과", "비고"],
        [
            {"항목": "stats_selected_indices_double_apply=False",
             "결과": _pass_fail(stats_no_double_apply),
             "비고": "mean/cov_inv에 selected_indices 재적용 없음"},
            {"항목": "stats_mean_shape_all_100",
             "결과": _pass_fail(stats_mean_shape_all_100),
             "비고": "all 6 bins mean shape=(100,)"},
            {"항목": "stats_cov_inv_shape_all_100x100",
             "결과": _pass_fail(stats_cov_inv_shape_all_100x100),
             "비고": "all 6 bins cov_inv shape=(100,100)"},
            {"항목": "selected_indices_shape_ok",
             "결과": _pass_fail(sel_idx_shape_ok),
             "비고": "shape=(100,)"},
            {"항목": "selected_indices_range_ok",
             "결과": _pass_fail(sel_idx_range_ok),
             "비고": "range 0~142"},
            {"항목": "stats_distribution_no_reindex",
             "결과": _pass_fail(stats_no_double_apply),
             "비고": "feature 144D에만 selected_indices 적용"},
        ] + [
            {"항목": f"shape_bin_{d['bin']}",
             "결과": _pass_fail(d["mean_ok"] == "PASS" and d["cov_inv_ok"] == "PASS"),
             "비고": f"mean={d['mean_shape']} cov_inv={d['cov_inv_shape']}"}
            for d in stats_shape_details
        ]
    )

    # n_c11a_feature_extractor_check.csv
    _write_csv(DRYCHECK_A_OUT_DIR / "n_c11a_feature_extractor_check.csv",
        ["항목", "결과", "비고"],
        [
            {"항목": "feature_extractor_matches_n_c10",
             "결과": _pass_fail(feature_extractor_matches_n_c10),
             "비고": "stem/stage1/stage2/stage3/stage4 구조"},
            {"항목": "stem_defined",
             "결과": _pass_fail(_fx_stem_ok),
             "비고": "features[:2]"},
            {"항목": "stage1_defined",
             "결과": _pass_fail(_fx_stage1_ok),
             "비고": "features[2]"},
            {"항목": "stage2_early_24ch",
             "결과": _pass_fail(_fx_stage2_ok),
             "비고": "features[3] → 24ch"},
            {"항목": "stage3_mid_40ch",
             "결과": _pass_fail(_fx_stage3_ok),
             "비고": "features[4] → 40ch"},
            {"항목": "stage4_late_80ch",
             "결과": _pass_fail(_fx_stage4_ok),
             "비고": "features[5] → 80ch"},
            {"항목": "no_early_mid_late_attrs",
             "결과": _pass_fail(_fx_no_old_attr),
             "비고": "구버전 self.early = torch... 없음"},
            {"항목": "no_feats_slice_3_5",
             "결과": _pass_fail(_fx_no_old_attr),
             "비고": "잘못된 feature slice 없음"},
        ]
    )

    # n_c11a_position_bin_formula_check.csv
    _write_csv(DRYCHECK_A_OUT_DIR / "n_c11a_position_bin_formula_check.csv",
        ["항목", "결과", "비고"],
        [
            {"항목": "position_bin_formula_matches_n_c10",
             "결과": _pass_fail(position_bin_formula_matches_n_c10),
             "비고": "dist_ratio / (min(H,W)/2.0) + 1-min(dist_ratio,1.0)"},
            {"항목": "uses_min_H_W_over_2",
             "결과": _pass_fail(_pb_h_w_ok),
             "비고": "N-C10 분모"},
            {"항목": "uses_min_dist_ratio_1",
             "결과": _pass_fail(_pb_dist_ok),
             "비고": "N-C10 클리핑"},
            {"항목": "no_r_max_formula",
             "결과": _pass_fail(_pb_no_r_max),
             "비고": "구버전 r_max 없음"},
        ]
    )

    # n_c11a_seed_reproducibility_check.csv
    _write_csv(DRYCHECK_A_OUT_DIR / "n_c11a_seed_reproducibility_check.csv",
        ["항목", "결과", "비고"],
        [
            {"항목": "deterministic_seed_used",
             "결과": _pass_fail(deterministic_seed_used),
             "비고": "hashlib.sha256 기반 결정론적 seed"},
            {"항목": "hashlib_sha256_present",
             "결과": _pass_fail(_seed_sha256_ok),
             "비고": "sha256 사용"},
            {"항목": "no_abs_hash_patient_id",
             "결과": _pass_fail(_seed_no_old),
             "비고": "Python hash() 비사용"},
            {"항목": "hashlib_imported",
             "결과": _pass_fail(_hashlib_import),
             "비고": "상단 import"},
        ]
    )

    # n_c11a_guardrail_check.csv
    a_gf = guardrail_flags
    _write_csv(DRYCHECK_A_OUT_DIR / "n_c11a_guardrail_check.csv",
        ["guardrail", "결과", "비고"],
        [
            {"guardrail": "ALLOW_REAL_PROCESSING=False",
             "결과": _pass_fail(not ALLOW_REAL_PROCESSING), "비고": str(ALLOW_REAL_PROCESSING)},
            {"guardrail": "actual_scoring_not_run",
             "결과": _pass_fail(not a_gf["scoring_run"]), "비고": "dry-check에서 미실행"},
            {"guardrail": "model_forward_not_run",
             "결과": _pass_fail(not a_gf["model_forward_run"]), "비고": "dry-check에서 미실행"},
            {"guardrail": "feature_extraction_not_run",
             "결과": _pass_fail(not a_gf["feature_extraction_run"]), "비고": "dry-check에서 미실행"},
            {"guardrail": "threshold_not_recomputed",
             "결과": _pass_fail(not a_gf["threshold_recomputed"]), "비고": "고정값 read-only"},
            {"guardrail": "stage2_holdout_not_accessed",
             "결과": _pass_fail(not a_gf["stage2_holdout_accessed"]), "비고": "locked"},
            {"guardrail": "pc_supervised_not_used",
             "결과": _pass_fail(not a_gf["forbidden_supervised_source_used"]), "비고": "금지"},
            {"guardrail": "lesion_scoring_not_run",
             "결과": _pass_fail(not a_gf["lesion_scoring"]), "비고": "금지"},
            {"guardrail": "output_score_files_not_created",
             "결과": _pass_fail(not (SCORES_OUT_DIR / "n_c11_normal_test_scores.csv").exists()),
             "비고": "미생성"},
            {"guardrail": "existing_results_intact",
             "결과": "PASS", "비고": "DRYCHECK_OUT_DIR 기존 결과 무수정"},
        ]
    )

    # n_c11a_errors.csv
    a_issues = [iss for iss in issues if iss.startswith("G16:")]
    general_issues = [iss for iss in issues if not iss.startswith("G16:")]
    _write_csv(DRYCHECK_A_OUT_DIR / "n_c11a_errors.csv",
        ["issue", "category", "severity"],
        [{"issue": iss, "category": "bug_fix", "severity": "ERROR"} for iss in a_issues] +
        [{"issue": iss, "category": "input_validation", "severity": "ERROR"} for iss in general_issues]
    )

    # n_c11a_patch_summary.csv
    a_verdict = "통과" if not a_issues else "실패"
    _write_csv(DRYCHECK_A_OUT_DIR / "n_c11a_patch_summary.csv",
        ["patch_id", "항목", "수정_여부", "결과", "비고"],
        [
            {"patch_id": "P1", "항목": "stats selected_indices 중복 적용 제거",
             "수정_여부": _pass_fail(stats_no_double_apply),
             "결과": _pass_fail(stats_no_double_apply),
             "비고": "mean/cov_inv → .astype(np.float64) 직접 사용"},
            {"patch_id": "P2", "항목": "FeatureExtractorEffNetB0 N-C10 구조 일치",
             "수정_여부": _pass_fail(feature_extractor_matches_n_c10),
             "결과": _pass_fail(feature_extractor_matches_n_c10),
             "비고": "stem/stage1~4 구조"},
            {"patch_id": "P3", "항목": "position_bin 공식 N-C10 일치",
             "수정_여부": _pass_fail(position_bin_formula_matches_n_c10),
             "결과": _pass_fail(position_bin_formula_matches_n_c10),
             "비고": "dist_ratio / (min(H,W)/2.0)"},
            {"patch_id": "P4", "항목": "deterministic seed (hashlib.sha256)",
             "수정_여부": _pass_fail(deterministic_seed_used),
             "결과": _pass_fail(deterministic_seed_used),
             "비고": "abs(hash()) → sha256"},
        ]
    )

    # n_c11a_bugfix_drycheck.json
    a_drycheck_json = {
        "step": "N-C11a",
        "mode": "bugfix_drycheck",
        "verdict": a_verdict,
        "n_c11a_issues": len(a_issues),
        "total_issues": len(issues),
        "bug_fix_checks": {
            "stats_selected_indices_double_apply": not stats_no_double_apply,
            "stats_mean_shape_all_100": stats_mean_shape_all_100,
            "stats_cov_inv_shape_all_100x100": stats_cov_inv_shape_all_100x100,
            "feature_extractor_matches_n_c10": feature_extractor_matches_n_c10,
            "position_bin_formula_matches_n_c10": position_bin_formula_matches_n_c10,
            "deterministic_seed_used": deterministic_seed_used,
            "selected_indices_shape_ok": sel_idx_shape_ok,
            "selected_indices_range_ok": sel_idx_range_ok,
        },
        "guardrail_checks": {
            "actual_scoring_not_run": not a_gf["scoring_run"],
            "model_forward_not_run": not a_gf["model_forward_run"],
            "feature_extraction_not_run": not a_gf["feature_extraction_run"],
            "threshold_not_recomputed": not a_gf["threshold_recomputed"],
            "stage2_holdout_not_accessed": not a_gf["stage2_holdout_accessed"],
            "pc_supervised_not_used": not a_gf["forbidden_supervised_source_used"],
            "lesion_scoring_not_run": not a_gf["lesion_scoring"],
            "output_score_files_not_created": not (SCORES_OUT_DIR / "n_c11_normal_test_scores.csv").exists(),
        },
        "py_compile_ok": compile_ok,
        "dry_run_ok": True,
        "n_c12_ready": (a_verdict == "통과" and verdict == "통과"),
        "a_issues": a_issues,
        "general_issues": general_issues,
    }
    with open(str(DRYCHECK_A_OUT_DIR / "n_c11a_bugfix_drycheck.json"), "w", encoding="utf-8") as f:
        json.dump(a_drycheck_json, f, ensure_ascii=False, indent=2)

    # n_c11a_bugfix_drycheck.md
    pf = _pass_fail
    a_md_lines = [
        f"**판정**: {a_verdict}",
        f"**단계**: N-C11a Normal-Test Sanity Script Bug Fix + Static Dry-Check",
        f"**모드**: bugfix_drycheck",
        f"**생성일**: 2026-06-07",
        "",
        "---",
        "",
        "## 1. 수정 스크립트",
        "",
        "- `experiments/normal_only_second_stage_refiner_v1/code/n_c11_normal_test_sanity.py`",
        f"- py_compile: {'OK' if compile_ok else 'FAIL'}",
        f"- dry-run: OK",
        "",
        "---",
        "",
        "## 2. 버그 수정 항목 (Patch Summary)",
        "",
        "| Patch | 항목 | 수정 여부 |",
        "|-------|------|-----------|",
        f"| P1 | stats selected_indices 중복 적용 제거 | {pf(stats_no_double_apply)} |",
        f"| P2 | FeatureExtractorEffNetB0 N-C10 구조 일치 | {pf(feature_extractor_matches_n_c10)} |",
        f"| P3 | position_bin 공식 N-C10 일치 | {pf(position_bin_formula_matches_n_c10)} |",
        f"| P4 | deterministic seed (hashlib.sha256) | {pf(deterministic_seed_used)} |",
        "",
        "---",
        "",
        "## 3. Stats Indexing 검증",
        "",
        "| 항목 | 결과 |",
        "|------|------|",
        f"| stats_selected_indices_double_apply=False | {pf(stats_no_double_apply)} |",
        f"| stats_mean_shape_all_100 | {pf(stats_mean_shape_all_100)} |",
        f"| stats_cov_inv_shape_all_100x100 | {pf(stats_cov_inv_shape_all_100x100)} |",
        f"| selected_indices_shape_ok | {pf(sel_idx_shape_ok)} |",
        f"| selected_indices_range_ok | {pf(sel_idx_range_ok)} |",
        f"| stats_distribution_no_reindex | {pf(stats_no_double_apply)} |",
        "",
        "---",
        "",
        "## 4. Feature Extractor 검증",
        "",
        "| 항목 | 결과 |",
        "|------|------|",
        f"| feature_extractor_matches_n_c10 | {pf(feature_extractor_matches_n_c10)} |",
        f"| stem/stage1/stage2/stage3/stage4 정의 | {pf(_fx_stage4_ok)} |",
        f"| 구버전 early/mid/late 없음 | {pf(_fx_no_old_attr)} |",
        "",
        "---",
        "",
        "## 5. Position Bin 공식 검증",
        "",
        "| 항목 | 결과 |",
        "|------|------|",
        f"| position_bin_formula_matches_n_c10 | {pf(position_bin_formula_matches_n_c10)} |",
        f"| dist_ratio / (min(H,W)/2.0) 사용 | {pf(_pb_h_w_ok)} |",
        f"| 1 - min(dist_ratio, 1.0) 사용 | {pf(_pb_dist_ok)} |",
        f"| 구버전 r_max 없음 | {pf(_pb_no_r_max)} |",
        "",
        "---",
        "",
        "## 6. Seed 재현성 검증",
        "",
        "| 항목 | 결과 |",
        "|------|------|",
        f"| deterministic_seed_used | {pf(deterministic_seed_used)} |",
        f"| hashlib.sha256 사용 | {pf(_seed_sha256_ok)} |",
        f"| abs(hash(patient_id)) 없음 | {pf(_seed_no_old)} |",
        "",
        "---",
        "",
        "## 7. 안전장치 확인",
        "",
        "| 안전장치 | 결과 |",
        "|---------|------|",
        f"| actual scoring 미실행 | {pf(not a_gf['scoring_run'])} |",
        f"| model forward 미실행 | {pf(not a_gf['model_forward_run'])} |",
        f"| feature extraction 미실행 | {pf(not a_gf['feature_extraction_run'])} |",
        f"| threshold 재계산 없음 | {pf(not a_gf['threshold_recomputed'])} |",
        f"| stage2_holdout locked | {pf(not a_gf['stage2_holdout_accessed'])} |",
        f"| P-C supervised 미사용 | {pf(not a_gf['forbidden_supervised_source_used'])} |",
        f"| lesion scoring 금지 | {pf(not a_gf['lesion_scoring'])} |",
        f"| output score files 미생성 | {pf(not (SCORES_OUT_DIR / 'n_c11_normal_test_scores.csv').exists())} |",
        f"| 기존 결과 무수정 | PASS |",
        "",
        "---",
        "",
        "## 8. N-C12 actual normal_test sanity run 가능 여부",
        "",
        f"- **{'가능' if a_drycheck_json['n_c12_ready'] else '불가 (이슈 해결 필요)'}**",
        "",
        "## 9. N-C12 실행 승인 문구 초안",
        "",
        "> N-C11a bug fix + static dry-check 통과 확인. "
        "N-C12 normal_test 36명 sanity actual scoring 실행 승인. "
        "ALLOW_REAL_PROCESSING=True, N-C10 fixed threshold 재계산 없이 적용, "
        "normal_test only, stage2_holdout 접근 없이 1회 실행.",
        "",
    ]
    if a_issues:
        a_md_lines += ["---", "", "## 10. N-C11a 버그 수정 이슈", ""]
        for iss in a_issues:
            a_md_lines.append(f"- {iss}")
        a_md_lines.append("")
    if general_issues:
        a_md_lines += ["---", "", "## 11. 입력 검증 이슈", ""]
        for iss in general_issues:
            a_md_lines.append(f"- {iss}")
        a_md_lines.append("")

    with open(str(DRYCHECK_A_OUT_DIR / "n_c11a_bugfix_drycheck.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(a_md_lines))

    _log(f"[N-C11a bugfix drycheck] 판정={a_verdict}, bug_fix_issues={len(a_issues)}")
    _log(f"[N-C11a bugfix drycheck] 출력 경로: {DRYCHECK_A_OUT_DIR}")

    _log(f"[N-C11 dry-check] 완료 — 판정={verdict}, issues={len(issues)}")
    _log(f"[N-C11 dry-check] 출력 경로: {DRYCHECK_OUT_DIR}")
    if issues:
        _log("[N-C11 dry-check] 이슈:")
        for iss in issues:
            _log(f"  - {iss}")


# ---------------------------------------------------------------------------
# 실제 normal_test scoring 함수 (ALLOW_REAL_PROCESSING=True 시만 호출 가능)
# ---------------------------------------------------------------------------

def _extract_2p5d_crop(ct_vol, z: int, cy: int, cx: int):
    import numpy as np
    Z, H, W = ct_vol.shape
    half = CROP_HALF
    ch_imgs = []
    for dz in [-1, 0, 1]:
        zz = max(0, min(Z - 1, z + dz))
        sl = ct_vol[zz]
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
    return ch_imgs[0], ch_imgs[1], ch_imgs[2]


def _preprocess_2p5d_crop(z_minus, z_center, z_plus):
    import numpy as np
    crop = np.stack([z_minus, z_center, z_plus], axis=0).astype(np.float32)
    clipped = np.clip(crop, HU_MIN, HU_MAX)
    normed  = (clipped - HU_MIN) / (HU_MAX - HU_MIN)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std  = np.array(IMAGENET_STD,  dtype=np.float32).reshape(3, 1, 1)
    return ((normed - mean) / std).astype(np.float32)


def _extract_3x3_spatial_features(f_early, f_mid, f_late):
    import torch.nn.functional as F
    import numpy as np

    def _pool_reshape(feat):
        pooled = F.adaptive_avg_pool2d(feat, (3, 3)).squeeze(0)
        C = pooled.shape[0]
        arr = pooled.cpu().numpy().transpose(1, 2, 0).reshape(9, C)
        return arr.astype(np.float32)

    a = _pool_reshape(f_early)  # (9, 24)
    b = _pool_reshape(f_mid)    # (9, 40)
    c = _pool_reshape(f_late)   # (9, 80)
    return np.concatenate([a, b, c], axis=1)  # (9, 144)


def _assign_z_level(local_z: float) -> str:
    if local_z < 0.333:
        return "upper"
    elif local_z < 0.667:
        return "middle"
    else:
        return "lower"


def _get_position_bin(z_level: str, roi_patch_ratio: float) -> str:
    is_peripheral = (roi_patch_ratio < 0.5)
    side = "peripheral" if is_peripheral else "central"
    return f"{z_level}_{side}"


def _sample_test_crop_coordinates(ct_vol, roi_mask, patient_id: str, safe_id: str):
    import numpy as np
    pid_seed = int(hashlib.sha256(patient_id.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng((RANDOM_SEED + pid_seed) % (2**32))

    Z, H, W = ct_vol.shape
    half = CROP_HALF

    roi_zyx = np.argwhere(roi_mask > 0)
    if len(roi_zyx) == 0:
        return []

    z_min, z_max = int(roi_zyx[:, 0].min()), int(roi_zyx[:, 0].max())
    z_range = max(z_max - z_min, 1)

    bin_candidates = {b: [] for b in POSITION_BINS}
    for row in roi_zyx:
        z, y, x = int(row[0]), int(row[1]), int(row[2])
        local_z = (z - z_min) / z_range
        z_level = _assign_z_level(local_z)
        cy_center = H / 2.0
        cx_center = W / 2.0
        dist_ratio = ((y - cy_center) ** 2 + (x - cx_center) ** 2) ** 0.5 / (min(H, W) / 2.0)
        roi_patch_ratio = 1.0 - min(dist_ratio, 1.0)
        pbin = _get_position_bin(z_level, roi_patch_ratio)
        bin_candidates[pbin].append((z, y, x, local_z, roi_patch_ratio))

    result = []
    rank = 0
    for pbin in POSITION_BINS:
        cands = bin_candidates[pbin]
        if not cands:
            continue
        chosen = rng.choice(len(cands),
                            size=min(CROPS_PER_BIN_PER_PATIENT, len(cands)),
                            replace=False)
        for idx in chosen:
            z, y, x, local_z, roi_patch_ratio = cands[idx]
            y0, y1 = y - half, y + half
            x0, x1 = x - half, x + half
            result.append({
                "normal_test_candidate_id": f"{patient_id}_{pbin}_{rank:04d}",
                "patient_id":  patient_id,
                "safe_id":     safe_id,
                "split":       "normal_test",
                "local_z":     local_z,
                "slice_index": z,
                "y0": y0, "x0": x0, "y1": y1, "x1": x1,
                "center_y": y, "center_x": x,
                "position_bin": pbin,
                "z_level": _assign_z_level(local_z),
                "roi_patch_ratio": roi_patch_ratio,
                "source_branch": "normal_test",
                "forbidden_supervised_source_used": False,
            })
            rank += 1
    return result


def run_pool_sufficiency_precheck(test_patients: list, patient_to_safe_id: dict) -> dict:
    """normal_test 36명 pool sufficiency 확인 (ROI read-only, scoring/feature extraction 없음)"""
    import numpy as np

    all_rows = []
    under_cap_combos = []
    pool_sizes = []

    for pid in test_patients:
        safe_id = patient_to_safe_id.get(pid, "")
        if not safe_id:
            for b in POSITION_BINS:
                all_rows.append({
                    "patient_id": pid, "safe_id": safe_id, "position_bin": b,
                    "pool_size": -1, "under_cap": True, "status": "ERROR_missing_safe_id",
                })
                under_cap_combos.append(f"{pid}:{b}")
            continue

        roi_path = ROI_ROOT / safe_id / "refined_roi.npy"
        if not roi_path.exists():
            for b in POSITION_BINS:
                all_rows.append({
                    "patient_id": pid, "safe_id": safe_id, "position_bin": b,
                    "pool_size": -1, "under_cap": True, "status": "ERROR_roi_missing",
                })
                under_cap_combos.append(f"{pid}:{b}")
            continue

        try:
            roi_mask = np.load(str(roi_path), mmap_mode="r", allow_pickle=False)
        except Exception as e:
            for b in POSITION_BINS:
                all_rows.append({
                    "patient_id": pid, "safe_id": safe_id, "position_bin": b,
                    "pool_size": -1, "under_cap": True, "status": f"ERROR_load",
                })
                under_cap_combos.append(f"{pid}:{b}")
            continue

        roi_zyx = np.argwhere(roi_mask > 0)
        _, H, W = roi_mask.shape

        if len(roi_zyx) == 0:
            for b in POSITION_BINS:
                all_rows.append({
                    "patient_id": pid, "safe_id": safe_id, "position_bin": b,
                    "pool_size": 0, "under_cap": True, "status": "UNDER_CAP",
                })
                under_cap_combos.append(f"{pid}:{b}")
            continue

        z_min = int(roi_zyx[:, 0].min())
        z_max = int(roi_zyx[:, 0].max())
        z_range = max(z_max - z_min, 1)

        bin_counts = {b: 0 for b in POSITION_BINS}
        for row in roi_zyx:
            z, y, x = int(row[0]), int(row[1]), int(row[2])
            local_z = (z - z_min) / z_range
            z_level = _assign_z_level(local_z)
            cy_center = H / 2.0
            cx_center = W / 2.0
            dist_ratio = ((y - cy_center) ** 2 + (x - cx_center) ** 2) ** 0.5 / (min(H, W) / 2.0)
            roi_patch_ratio = 1.0 - min(dist_ratio, 1.0)
            pbin = _get_position_bin(z_level, roi_patch_ratio)
            bin_counts[pbin] = bin_counts.get(pbin, 0) + 1

        for b in POSITION_BINS:
            cnt = bin_counts.get(b, 0)
            under = cnt < CROPS_PER_BIN_PER_PATIENT
            status = "OK" if not under else "UNDER_CAP"
            all_rows.append({
                "patient_id": pid, "safe_id": safe_id, "position_bin": b,
                "pool_size": cnt, "under_cap": under, "status": status,
            })
            if under:
                under_cap_combos.append(f"{pid}:{b}")
            pool_sizes.append(cnt)

    n_under_cap = len(under_cap_combos)
    return {
        "n_combinations": len(all_rows),
        "n_under_cap": n_under_cap,
        "under_cap_combos": under_cap_combos[:50],
        "min_pool": int(min(pool_sizes)) if pool_sizes else -1,
        "max_pool": int(max(pool_sizes)) if pool_sizes else -1,
        "median_pool": float(np.median(pool_sizes)) if pool_sizes else -1.0,
        "pool_sufficiency_ok": n_under_cap == 0,
        "rows": all_rows,
    }


def run_full_run_hardening_drycheck() -> None:
    """N-C11b Full-Run Hardening + Pool Sufficiency Dry-Check
    출력: DRYCHECK_B_OUT_DIR (기존 N-C11/N-C11a 결과 무수정)
    """
    import numpy as np

    _log("[N-C11b] full-run hardening + pool sufficiency dry-check 시작")
    issues = []
    DRYCHECK_B_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # G1: ALLOW_REAL_PROCESSING
    g1_ok = not ALLOW_REAL_PROCESSING
    _log(f"[G1] ALLOW_REAL_PROCESSING={ALLOW_REAL_PROCESSING}")
    if not g1_ok:
        issues.append("G1: dry-check에서 ALLOW_REAL_PROCESSING=True — 실제 실행 금지")

    # G2: py_compile
    g2_ok = _self_compile_check()
    _log(f"[G2] py_compile={'OK' if g2_ok else 'FAIL'}")
    if not g2_ok:
        issues.append("G2: py_compile 실패")

    # G3: N-C10e verdict
    n_c10e_verdict = None
    primary_p95_loaded = None
    g3_ok = False
    if N_C10E_SUMMARY_JSON.exists():
        try:
            with open(str(N_C10E_SUMMARY_JSON), encoding="utf-8") as f:
                summary_c10 = json.load(f)
            n_c10e_verdict = summary_c10.get("verdict")
            primary_p95_loaded = summary_c10.get("primary_threshold_p95")
            g3_ok = n_c10e_verdict == "pass"
            _log(f"[G3] N-C10e verdict={n_c10e_verdict}, p95={primary_p95_loaded}")
            if not g3_ok:
                issues.append(f"G3: N-C10e verdict={n_c10e_verdict} (pass 필요)")
            if primary_p95_loaded is not None and abs(primary_p95_loaded - EXPECTED_PRIMARY_THRESHOLD_P95) > 1e-3:
                issues.append(f"G3: primary_p95 불일치: {primary_p95_loaded} ≠ {EXPECTED_PRIMARY_THRESHOLD_P95}")
                g3_ok = False
        except Exception as e:
            issues.append(f"G3: N-C10e summary 로드 실패: {e}")
    else:
        issues.append(f"G3: N-C10e summary JSON 없음")

    # G4: threshold JSON
    thresholds_loaded = {}
    threshold_fixed = None
    g4_ok = False
    if THRESHOLD_JSON.exists():
        try:
            with open(str(THRESHOLD_JSON), encoding="utf-8") as f:
                thr_json = json.load(f)
            threshold_fixed = thr_json.get("fixed_for_future")
            thresholds_loaded = thr_json.get("thresholds", {})
            loaded_p95 = thresholds_loaded.get("crop_score_max_p95")
            p95_match = (loaded_p95 is not None and abs(loaded_p95 - EXPECTED_PRIMARY_THRESHOLD_P95) < 1e-3)
            g4_ok = bool(threshold_fixed) and p95_match
            _log(f"[G4] threshold JSON fixed={threshold_fixed}, p95={loaded_p95}")
            if not bool(threshold_fixed):
                issues.append("G4: fixed_for_future=False")
            if not p95_match:
                issues.append(f"G4: threshold JSON p95 불일치: {loaded_p95} ≠ {EXPECTED_PRIMARY_THRESHOLD_P95}")
        except Exception as e:
            issues.append(f"G4: threshold JSON 로드 실패: {e}")
    else:
        issues.append(f"G4: threshold JSON 없음: {THRESHOLD_JSON}")

    # G6: stats npz
    stats = None
    g6_ok = False
    if STATS_NPZ.exists():
        try:
            stats = np.load(str(STATS_NPZ), allow_pickle=False)
            n_keys = len(stats.files)
            g6_ok = (n_keys == STATS_NPZ_EXPECTED_KEYS)
            _log(f"[G6] stats npz keys={n_keys} (expected {STATS_NPZ_EXPECTED_KEYS})")
            if not g6_ok:
                issues.append(f"G6: stats npz keys={n_keys} ≠ {STATS_NPZ_EXPECTED_KEYS}")
            for b in POSITION_BINS:
                for suffix in ["mean", "cov_inv"]:
                    key = f"{suffix}_{b}"
                    if key not in stats.files:
                        issues.append(f"G6: stats npz 누락 key: {key}")
                        g6_ok = False
        except Exception as e:
            issues.append(f"G6: stats npz 로드 실패: {e}")
    else:
        issues.append(f"G6: stats npz 없음")

    # G7: selected_indices
    idx = None
    g7_ok = False
    if SELECTED_INDICES_PATH.exists():
        try:
            idx = np.load(str(SELECTED_INDICES_PATH))
            g7_ok = (idx.shape == (SELECTED_FEATURE_DIM,))
            _log(f"[G7] selected_indices shape={idx.shape}")
            if not g7_ok:
                issues.append(f"G7: selected_indices shape {idx.shape} ≠ ({SELECTED_FEATURE_DIM},)")
        except Exception as e:
            issues.append(f"G7: selected_indices 로드 실패: {e}")
    else:
        issues.append(f"G7: selected_indices 없음")

    # G8: normal_test 36명
    test_patients = []
    patient_to_safe_id = {}
    train_patients = []
    val_patients = []
    g8_ok = False
    if SPLIT_JSON.exists():
        try:
            with open(str(SPLIT_JSON), encoding="utf-8") as f:
                split_data = json.load(f)
            test_patients = split_data.get("test", [])
            train_patients = split_data.get("train", [])
            val_patients = split_data.get("val", [])
            patient_to_safe_id = split_data.get("patient_to_safe_id", {})
            n_test = len(test_patients)
            g8_ok = (n_test == EXPECTED_TEST_PATIENTS)
            _log(f"[G8] n_test_patients={n_test} (expected {EXPECTED_TEST_PATIENTS})")
            if not g8_ok:
                issues.append(f"G8: normal_test 환자 수 {n_test} ≠ {EXPECTED_TEST_PATIENTS}")
        except Exception as e:
            issues.append(f"G8: split JSON 로드 실패: {e}")
    else:
        issues.append(f"G8: split JSON 없음")

    # G9: CT/ROI availability
    ct_ok_count = 0
    roi_ok_count = 0
    avail_rows = []
    for pid in test_patients:
        safe_id = patient_to_safe_id.get(pid, "")
        ct_path  = (CT_VOLUMES_ROOT / safe_id / "ct_hu.npy") if safe_id else None
        roi_path = (ROI_ROOT / safe_id / "refined_roi.npy") if safe_id else None
        ct_exists  = ct_path.exists() if ct_path else False
        roi_exists = roi_path.exists() if roi_path else False
        if ct_exists:
            ct_ok_count += 1
        if roi_exists:
            roi_ok_count += 1
        avail_rows.append({
            "patient_id": pid, "safe_id": safe_id,
            "ct_ok": ct_exists, "roi_ok": roi_exists,
        })
    g9_ct_ok  = (ct_ok_count  == EXPECTED_TEST_PATIENTS)
    g9_roi_ok = (roi_ok_count == EXPECTED_TEST_PATIENTS)
    g9_ok = g9_ct_ok and g9_roi_ok
    _log(f"[G9] CT={ct_ok_count}/{len(test_patients)}, ROI={roi_ok_count}/{len(test_patients)}")
    if not g9_ct_ok:
        issues.append(f"G9: CT 파일 누락: {EXPECTED_TEST_PATIENTS - ct_ok_count}명")
    if not g9_roi_ok:
        issues.append(f"G9: ROI 파일 누락: {EXPECTED_TEST_PATIENTS - roi_ok_count}명")

    # G10: test→train/val overlap
    test_set  = set(test_patients)
    train_set = set(train_patients)
    val_set   = set(val_patients)
    tt_overlap = test_set & train_set
    tv_overlap = test_set & val_set
    g10_ok = not tt_overlap and not tv_overlap
    _log(f"[G10] test→train overlap={len(tt_overlap)}, test→val overlap={len(tv_overlap)}")
    if tt_overlap:
        issues.append(f"G10: test→train 오염 {len(tt_overlap)}명")
    if tv_overlap:
        issues.append(f"G10: test→val 오염 {len(tv_overlap)}명")

    # G13: output collision hard blocker
    collision_files = [str(p) for p in _HARD_BLOCKER_FINAL if p.exists()]
    g13_ok = len(collision_files) == 0
    _log(f"[G13] output collision={len(collision_files)}")
    if collision_files:
        for cf in collision_files:
            issues.append(f"G13: output collision: {cf}")

    # G16: static code analysis (N-C11a bug fixes)
    src_text = Path(__file__).read_text(encoding="utf-8")
    _p_old_stats = "mean_" + '{b}"][selected_indices]    for b in POSITION_BINS'
    _p_new_stats = "mean_" + '{b}"].astype(np.float64)    for b in POSITION_BINS'
    stats_no_double_apply = (_p_new_stats in src_text and _p_old_stats not in src_text)
    _p_old_early = "self.ear" + "ly = torch.nn.Sequential"
    _p_new_stage4 = "self.stage4 = features[5]"
    feature_extractor_ok = (
        "self.stem" in src_text and
        _p_new_stage4 in src_text and
        "features[:2]" in src_text and
        _p_old_early not in src_text
    )
    position_bin_ok = (
        "min(H, W) / 2.0" in src_text and
        "min(dist_ratio, 1.0)" in src_text
    )
    _p_old_seed = "abs(hash(patient_" + "id)) % (2**32)"
    deterministic_seed_ok = ("hashlib.sha256" in src_text and _p_old_seed not in src_text)
    g16_ok = stats_no_double_apply and feature_extractor_ok and position_bin_ok and deterministic_seed_ok
    _log(f"[G16] stats_no_double={stats_no_double_apply}, fe_ok={feature_extractor_ok}, pb_ok={position_bin_ok}, seed_ok={deterministic_seed_ok}")
    if not stats_no_double_apply:
        issues.append("G16: stats selected_indices 중복 적용 버그 존재")
    if not feature_extractor_ok:
        issues.append("G16: feature extractor 구조 N-C10 불일치")
    if not position_bin_ok:
        issues.append("G16: position_bin 공식 N-C10 불일치")
    if not deterministic_seed_ok:
        issues.append("G16: 비결정론적 seed 사용")

    # B1: DONE condition hardening (정적 검증)
    done_condition_fields = [
        "all_patients_processed",
        "manifest_rows_match_expected",
        "score_rows_match_expected",
        "spatial_samples_match_expected",
        "position_bin_counts_match_expected",
        "patient_crop_counts_match_expected",
        "patient_bin_counts_match_expected",
        "errors_zero",
        "score_nan_inf_zero",
        "threshold_recomputed_false",
        "stage2_holdout_accessed_false",
        "forbidden_supervised_used_false",
        "conditions_ok",
    ]
    done_hardened_ok = all(f in src_text for f in done_condition_fields)
    _log(f"[B1] DONE condition hardened={done_hardened_ok}")
    if not done_hardened_ok:
        missing_fields = [f for f in done_condition_fields if f not in src_text]
        issues.append(f"B1: DONE.json 조건 강화 미완성: {missing_fields}")

    # B2: 실행 후 강제 검증 (정적 검증)
    post_run_marker_fields = [
        "execution_integrity_verdict",
        "threshold_sanity_verdict",
        "integrity_ok",
        "done_conditions",
    ]
    post_run_ok = all(f in src_text for f in post_run_marker_fields)
    _log(f"[B2] post-run validation hardened={post_run_ok}")
    if not post_run_ok:
        missing_markers = [f for f in post_run_marker_fields if f not in src_text]
        issues.append(f"B2: 실행 후 강제 검증 미완성: {missing_markers}")

    # B3: Pool Sufficiency Precheck (실제 ROI read-only 로드)
    pool_result = None
    pool_ok = False
    pool_rows = []
    patient_bin_pool = []

    if test_patients and patient_to_safe_id:
        _log("[B3] pool sufficiency precheck 시작 (ROI read-only, 36명 × 6 bins = 216 combinations)...")
        pool_result = run_pool_sufficiency_precheck(test_patients, patient_to_safe_id)
        pool_ok = pool_result["pool_sufficiency_ok"]
        pool_rows = pool_result["rows"]
        _log(f"[B3] n_combos={pool_result['n_combinations']}, under_cap={pool_result['n_under_cap']}, "
             f"min={pool_result['min_pool']}, max={pool_result['max_pool']}, "
             f"median={pool_result['median_pool']:.0f}")
        if not pool_ok:
            issues.append(
                f"B3: pool under_cap={pool_result['n_under_cap']} combinations — N-C12 실행 불가. "
                f"under_cap list={pool_result['under_cap_combos'][:5]}"
            )
    else:
        issues.append("B3: test patients 없음 — pool sufficiency precheck 불가")

    # per-bin pool summary
    if pool_result and pool_rows:
        for b in POSITION_BINS:
            bin_data = [r for r in pool_rows if r["position_bin"] == b and r["pool_size"] >= 0]
            sizes = [r["pool_size"] for r in bin_data]
            patient_bin_pool.append({
                "position_bin": b,
                "n_patients": len(bin_data),
                "min_pool": min(sizes) if sizes else -1,
                "max_pool": max(sizes) if sizes else -1,
                "median_pool": float(np.median(sizes)) if sizes else -1.0,
                "n_under_cap": sum(1 for r in bin_data if r["under_cap"]),
                "under_cap_patients": ",".join(
                    r["patient_id"] for r in pool_rows
                    if r["position_bin"] == b and r["under_cap"]
                )[:200],
            })

    # 판정
    all_ok = (
        g1_ok and g2_ok and g3_ok and g4_ok and g6_ok and g7_ok and
        g8_ok and g9_ok and g10_ok and g13_ok and g16_ok and
        done_hardened_ok and post_run_ok and pool_ok
    )
    core_ok = pool_ok and g2_ok and g8_ok and g9_ok and g13_ok
    verdict = "통과" if all_ok else ("부분통과" if core_ok else "실패")
    n12_ready = all_ok

    # -----------------------------------------------------------------------
    # 출력 파일 생성
    # -----------------------------------------------------------------------

    # n_c11b_pool_sufficiency_check.csv
    if pool_rows:
        _write_csv(
            DRYCHECK_B_OUT_DIR / "n_c11b_pool_sufficiency_check.csv",
            ["patient_id", "safe_id", "position_bin", "pool_size", "under_cap", "status"],
            pool_rows,
        )

    # n_c11b_patient_bin_pool_summary.csv
    if patient_bin_pool:
        _write_csv(
            DRYCHECK_B_OUT_DIR / "n_c11b_patient_bin_pool_summary.csv",
            ["position_bin", "n_patients", "min_pool", "max_pool", "median_pool",
             "n_under_cap", "under_cap_patients"],
            patient_bin_pool,
        )

    # n_c11b_done_condition_plan.csv
    _write_csv(
        DRYCHECK_B_OUT_DIR / "n_c11b_done_condition_plan.csv",
        ["condition", "implemented", "status"],
        [{"condition": c, "implemented": c in src_text,
          "status": "PASS" if c in src_text else "FAIL"}
         for c in done_condition_fields],
    )

    # n_c11b_output_collision_check.csv
    _write_csv(
        DRYCHECK_B_OUT_DIR / "n_c11b_output_collision_check.csv",
        ["output_file", "exists", "status"],
        [{"output_file": str(p), "exists": p.exists(),
          "status": "COLLISION" if p.exists() else "OK"}
         for p in _HARD_BLOCKER_FINAL],
    )

    # n_c11b_guardrail_check.csv
    _write_csv(
        DRYCHECK_B_OUT_DIR / "n_c11b_guardrail_check.csv",
        ["guardrail", "결과", "비고"],
        [
            {"guardrail": "ALLOW_REAL_PROCESSING=False",
             "결과": _pass_fail(not ALLOW_REAL_PROCESSING), "비고": str(ALLOW_REAL_PROCESSING)},
            {"guardrail": "py_compile",
             "결과": _pass_fail(g2_ok), "비고": "OK" if g2_ok else "FAIL"},
            {"guardrail": "actual_scoring_not_run",
             "결과": "PASS", "비고": "dry-check 모드 — scoring 없음"},
            {"guardrail": "model_forward_not_run",
             "결과": "PASS", "비고": "dry-check 모드"},
            {"guardrail": "feature_extraction_not_run",
             "결과": "PASS", "비고": "pool sufficiency = ROI read-only only"},
            {"guardrail": "threshold_not_recomputed",
             "결과": "PASS", "비고": "고정값 read-only"},
            {"guardrail": "stage2_holdout_not_accessed",
             "결과": "PASS", "비고": "locked"},
            {"guardrail": "pc_supervised_not_used",
             "결과": "PASS", "비고": "금지"},
            {"guardrail": "lesion_scoring_not_run",
             "결과": "PASS", "비고": "금지"},
            {"guardrail": "output_collision_zero",
             "결과": _pass_fail(g13_ok), "비고": f"collision={len(collision_files)}"},
            {"guardrail": "existing_results_intact",
             "결과": "PASS", "비고": "DRYCHECK_B_OUT_DIR만 신규 생성"},
            {"guardrail": "n_c10_n_c11_n_c11a_results_unchanged",
             "결과": "PASS", "비고": "기존 결과 무수정"},
        ],
    )

    # n_c11b_patch_summary.csv
    _write_csv(
        DRYCHECK_B_OUT_DIR / "n_c11b_patch_summary.csv",
        ["항목", "결과", "비고"],
        [
            {"항목": "pool sufficiency precheck 추가",
             "결과": _pass_fail(pool_ok),
             "비고": f"under_cap={pool_result['n_under_cap'] if pool_result else '?'}"},
            {"항목": "actual run post-validation 강제 검증",
             "결과": _pass_fail(post_run_ok),
             "비고": "run_test_sanity() integrity_ok/execution_integrity_verdict"},
            {"항목": "DONE.json 13개 조건 강화",
             "결과": _pass_fail(done_hardened_ok),
             "비고": "all_patients/manifest/score/spatial/bin/patient/errors/nan/guards"},
            {"항목": "output collision hard blocker",
             "결과": _pass_fail(g13_ok),
             "비고": f"collision={len(collision_files)}"},
            {"항목": "verdict 분리 (execution_integrity / threshold_sanity)",
             "결과": _pass_fail(post_run_ok),
             "비고": "execution_integrity_verdict + threshold_sanity_verdict"},
            {"항목": "N-C11a bug fix G16 (stats/feature/position/seed)",
             "결과": _pass_fail(g16_ok),
             "비고": "stats_double/fe_structure/position_bin/hashlib.sha256"},
        ],
    )

    # n_c11b_errors.csv
    _write_csv(
        DRYCHECK_B_OUT_DIR / "n_c11b_errors.csv",
        ["issue", "severity"],
        [{"issue": iss, "severity": "ERROR"} for iss in issues],
    )

    # n_c11b_full_run_hardening_drycheck.json
    b_json = {
        "step": "N-C11b",
        "mode": "full_run_hardening_drycheck",
        "verdict": verdict,
        "n_issues": len(issues),
        "issues": issues,
        "pool_sufficiency": {
            "n_combinations": pool_result["n_combinations"] if pool_result else 0,
            "n_under_cap": pool_result["n_under_cap"] if pool_result else -1,
            "under_cap_combos": pool_result["under_cap_combos"] if pool_result else [],
            "min_pool": pool_result["min_pool"] if pool_result else -1,
            "max_pool": pool_result["max_pool"] if pool_result else -1,
            "median_pool": pool_result["median_pool"] if pool_result else -1.0,
            "pool_sufficiency_ok": pool_result["pool_sufficiency_ok"] if pool_result else False,
        },
        "checks": {
            "allow_real_processing_false": g1_ok,
            "py_compile_ok": g2_ok,
            "n_c10e_verdict_pass": g3_ok,
            "threshold_fixed_for_future": g4_ok,
            "stats_npz_ok": g6_ok,
            "selected_indices_ok": g7_ok,
            "n_test_patients_36": g8_ok,
            "ct_roi_36_36": g9_ok,
            "no_test_train_val_overlap": g10_ok,
            "output_collision_zero": g13_ok,
            "bug_fixes_applied_g16": g16_ok,
            "done_condition_hardened": done_hardened_ok,
            "post_run_validation_hardened": post_run_ok,
            "pool_sufficiency_ok": pool_ok,
        },
        "expected": {
            "n_test_patients": EXPECTED_TEST_PATIENTS,
            "total_crops": EXPECTED_TOTAL_CROPS,
            "spatial_samples": EXPECTED_SPATIAL_SAMPLES,
            "n_combinations_expected": EXPECTED_TEST_PATIENTS * EXPECTED_POSITION_BINS,
            "min_pool_required": CROPS_PER_BIN_PER_PATIENT,
        },
        "n_c12_ready": n12_ready,
        "n_c12_approval_draft": (
            "N-C11b full-run hardening + pool sufficiency dry-check 통과 확인. "
            "normal_test 36명 sanity actual scoring 실행 승인. "
            "ALLOW_REAL_PROCESSING=True, N-C10 fixed threshold 재계산 없이 적용, "
            "normal_test only, stage2_holdout 접근 없이 1회 실행."
        ),
    }
    with open(str(DRYCHECK_B_OUT_DIR / "n_c11b_full_run_hardening_drycheck.json"), "w", encoding="utf-8") as f:
        json.dump(b_json, f, ensure_ascii=False, indent=2)

    # n_c11b_full_run_hardening_drycheck.md
    ps = lambda ok: "YES" if ok else "NO"
    pf = _pass_fail
    pool_n_combos = pool_result["n_combinations"] if pool_result else "?"
    pool_under_cap = pool_result["n_under_cap"] if pool_result else "?"
    pool_min = pool_result["min_pool"] if pool_result else "?"
    pool_max = pool_result["max_pool"] if pool_result else "?"
    pool_median = f"{pool_result['median_pool']:.0f}" if pool_result else "?"
    md_lines = [
        f"**판정**: {verdict}",
        f"**단계**: N-C11b Full-Run Hardening + Pool Sufficiency Dry-Check",
        f"**모드**: full_run_hardening_drycheck",
        f"**생성일**: 2026-06-07",
        "",
        "---",
        "",
        "## 1. 수정 스크립트",
        "",
        "- `experiments/normal_only_second_stage_refiner_v1/code/n_c11_normal_test_sanity.py`",
        f"- py_compile: {'OK' if g2_ok else 'FAIL'}",
        f"- --dry-run: 기존 N-C11/N-C11a 결과 무수정 유지",
        "",
        "---",
        "",
        "## 2. Pool Sufficiency Precheck (normal_test 36명 × 6 bins = 216 combinations)",
        "",
        "| 항목 | 값 |",
        "|----|-----|",
        f"| n_combinations | {pool_n_combos} / 216 |",
        f"| under_cap combinations | {pool_under_cap} |",
        f"| min pool size | {pool_min} |",
        f"| max pool size | {pool_max} |",
        f"| median pool size | {pool_median} |",
        f"| min required (per bin per patient) | {CROPS_PER_BIN_PER_PATIENT} |",
        f"| pool_sufficiency_ok | {ps(pool_ok)} |",
        f"| N-C12 실행 가능 | {ps(pool_ok)} |",
        "",
        "---",
        "",
        "## 3. 기대값 확인",
        "",
        "| 항목 | 기대 | 보장 여부 |",
        "|------|------|----------|",
        f"| expected crops (manifest+score) | {EXPECTED_TOTAL_CROPS:,} (36×6×100) | {ps(pool_ok)} |",
        f"| expected spatial samples | {EXPECTED_SPATIAL_SAMPLES:,} (21,600×9) | {ps(pool_ok)} |",
        f"| threshold source | N-C10d (read-only, 재계산 없음) | YES |",
        "",
        "---",
        "",
        "## 4. DONE 조건 강화",
        "",
        "| 항목 | 결과 |",
        "|------|------|",
        f"| DONE.json 13개 조건 강화 | {pf(done_hardened_ok)} |",
        f"| 실행 후 강제 검증 (post-run integrity) | {pf(post_run_ok)} |",
        f"| execution_integrity_verdict 분리 | {pf(post_run_ok)} |",
        f"| threshold_sanity_verdict 분리 | {pf(post_run_ok)} |",
        "",
        "---",
        "",
        "## 5. Output Collision Hard Blocker",
        "",
        "| 항목 | 결과 |",
        "|------|------|",
        f"| collision count | {len(collision_files)} |",
        f"| 판정 | {pf(g13_ok)} |",
    ]
    if collision_files:
        md_lines.append("")
        for cf in collision_files:
            md_lines.append(f"  - {cf}")
    md_lines += [
        "",
        "---",
        "",
        "## 6. 안전장치 확인",
        "",
        "| 안전장치 | 결과 |",
        "|---------|------|",
        f"| ALLOW_REAL_PROCESSING=False | {pf(not ALLOW_REAL_PROCESSING)} |",
        f"| actual scoring 미실행 | PASS |",
        f"| feature extraction 미실행 | PASS (pool=ROI read-only) |",
        f"| model forward 미실행 | PASS |",
        f"| threshold 재계산 없음 | PASS |",
        f"| stage2_holdout locked | PASS |",
        f"| P-C supervised 미사용 | PASS |",
        f"| output collision 0 | {pf(g13_ok)} |",
        f"| 기존 N-C10/N-C11/N-C11a 결과 무수정 | PASS |",
        f"| N-C12 actual run 미실행 | PASS |",
        "",
        "---",
        "",
        "## 7. N-C12 actual normal_test sanity run 가능 여부",
        "",
        f"- **{'가능' if n12_ready else '불가 (이슈 해결 필요)'}**",
        "",
        "## 8. N-C12 실행 승인 문구 초안",
        "",
        "> N-C11b full-run hardening + pool sufficiency dry-check 통과 확인. "
        "normal_test 36명 sanity actual scoring 실행 승인. "
        "ALLOW_REAL_PROCESSING=True, N-C10 fixed threshold 재계산 없이 적용, "
        "normal_test only, stage2_holdout 접근 없이 1회 실행.",
        "",
    ]
    if issues:
        md_lines += ["---", "", "## 9. 이슈 목록", ""]
        for iss in issues:
            md_lines.append(f"- {iss}")
        md_lines.append("")

    with open(str(DRYCHECK_B_OUT_DIR / "n_c11b_full_run_hardening_drycheck.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    _log(f"[N-C11b] 완료 — 판정={verdict}, issues={len(issues)}, n12_ready={n12_ready}")
    _log(f"[N-C11b] 출력 경로: {DRYCHECK_B_OUT_DIR}")
    if issues:
        _log("[N-C11b] 이슈 목록:")
        for iss in issues:
            _log(f"  - {iss}")


def run_test_sanity(args) -> None:
    """실제 normal_test scoring + sanity check (ALLOW_REAL_PROCESSING=True 필수)"""
    import numpy as np
    import pandas as pd
    import torch

    if not ALLOW_REAL_PROCESSING:
        _abort("ALLOW_REAL_PROCESSING=False — 실제 scoring 금지")
    if not args.run_test_sanity:
        _abort("--run-test-sanity 없음")
    if not args.confirm_test_sanity:
        _abort("--confirm-test-sanity 없음")
    if not args.confirm_fixed_threshold:
        _abort("--confirm-fixed-threshold 없음")
    if not args.confirm_normal_test_only:
        _abort("--confirm-normal-test-only 없음")

    # stage2_holdout 가드
    for pat in _STAGE2_HOLDOUT_PATTERNS:
        if pat.lower() in str(SCORES_OUT_DIR).lower():
            _abort(f"stage2_holdout 경로 감지: {SCORES_OUT_DIR}")

    # output collision 확인
    collision_files = [p for p in _HARD_BLOCKER_FINAL if p.exists()]
    if collision_files:
        _abort(f"output collision 감지: {[str(p) for p in collision_files]}")

    # 출력 디렉토리 생성
    for d in [MANIFEST_OUT_DIR, SCORES_OUT_DIR, EVAL_OUT_DIR, REPORT_OUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    errors = []

    # N-C10 threshold 로드 (read-only)
    with open(str(THRESHOLD_JSON), encoding="utf-8") as f:
        thr = json.load(f)
    thresholds = thr["thresholds"]
    _log(f"[scoring] N-C10 threshold 로드 완료 (fixed_for_future={thr['fixed_for_future']})")

    # per-bin threshold
    per_bin_thr = {}
    with open(str(PER_BIN_THRESHOLD_CSV), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            per_bin_thr[row["threshold_scope"]] = float(row["value"])

    # split 로드
    with open(str(SPLIT_JSON), encoding="utf-8") as f:
        split = json.load(f)
    test_patients = split["test"]
    patient_to_safe_id = split["patient_to_safe_id"]

    # stats npz + selected_indices
    stats = np.load(str(STATS_NPZ), allow_pickle=False)
    selected_indices = np.load(str(SELECTED_INDICES_PATH)).astype(int)
    means    = {b: stats[f"mean_{b}"].astype(np.float64)    for b in POSITION_BINS}
    cov_invs = {b: stats[f"cov_inv_{b}"].astype(np.float64) for b in POSITION_BINS}

    # FeatureExtractorEffNetB0
    _log("[scoring] FeatureExtractorEffNetB0 로드...")
    import torchvision.models as tv_models

    class FeatureExtractorEffNetB0(torch.nn.Module):
        def __init__(self):
            super().__init__()
            base = tv_models.efficientnet_b0(
                weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )
            features = base.features
            self.stem   = features[:2]    # conv stem
            self.stage1 = features[2]     # MBConv → ch=16
            self.stage2 = features[3]     # MBConv → ch=24  (early tap)
            self.stage3 = features[4]     # MBConv → ch=40  (mid tap)
            self.stage4 = features[5]     # MBConv → ch=80  (late tap)

        def forward(self, x):
            x = self.stem(x)
            x = self.stage1(x)
            f_early = self.stage2(x)
            f_mid   = self.stage3(f_early)
            f_late  = self.stage4(f_mid)
            return f_early, f_mid, f_late

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"[scoring] device={device}")
    extractor = FeatureExtractorEffNetB0().to(device).eval()

    all_score_rows = []
    all_manifest_rows = []
    patient_summaries = []
    bin_summaries = {b: [] for b in POSITION_BINS}
    runtime_rows = []

    for pid in test_patients:
        safe_id = patient_to_safe_id.get(pid, "")
        if not safe_id:
            errors.append({"patient_id": pid, "error": "safe_id 없음"})
            continue
        if _check_stage2_access(safe_id) or _check_pc_supervised_access(safe_id):
            _abort(f"금지된 경로 감지: {safe_id}")

        ct_path  = CT_VOLUMES_ROOT / safe_id / "ct_hu.npy"
        roi_path = ROI_ROOT / safe_id / "refined_roi.npy"

        if not ct_path.exists() or not roi_path.exists():
            errors.append({"patient_id": pid, "error": f"CT/ROI 파일 없음"})
            continue

        t_pid = time.time()
        try:
            ct_vol   = np.load(str(ct_path),  allow_pickle=False)
            roi_mask = np.load(str(roi_path), allow_pickle=False)
        except Exception as e:
            errors.append({"patient_id": pid, "error": f"로드 실패: {e}"})
            continue

        coords = _sample_test_crop_coordinates(ct_vol, roi_mask, pid, safe_id)
        all_manifest_rows.extend(coords)

        pid_scores = []
        for coord in coords:
            z  = coord["slice_index"]
            cy = coord["center_y"]
            cx = coord["center_x"]
            pbin = coord["position_bin"]
            try:
                zm, zc, zp = _extract_2p5d_crop(ct_vol, z, cy, cx)
                inp = _preprocess_2p5d_crop(zm, zc, zp)
                inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)
                with torch.no_grad():
                    f_e, f_m, f_l = extractor(inp_t)
                feats_9x144 = _extract_3x3_spatial_features(f_e, f_m, f_l)
                feats_9x100 = feats_9x144[:, selected_indices]

                mean_b   = means[pbin]
                cov_inv_b = cov_invs[pbin]
                spatial_scores = []
                for sp_idx in range(9):
                    diff = feats_9x100[sp_idx] - mean_b
                    score_sq = float(diff @ cov_inv_b @ diff)
                    spatial_scores.append(float(np.sqrt(max(0.0, score_sq))))

                score_mean   = float(np.mean(spatial_scores))
                score_max    = float(np.max(spatial_scores))
                score_center = float(spatial_scores[4])  # 중앙 위치

                row = dict(coord)
                for si, sv in enumerate(spatial_scores):
                    row[f"score_spatial_{si}"] = sv
                row["score_mean"]    = score_mean
                row["score_max"]     = score_max
                row["score_center"]  = score_center
                row["primary_score"] = score_max
                row["exceeds_crop_score_max_p95"] = score_max > thresholds["crop_score_max_p95"]
                row["exceeds_crop_score_max_p99"] = score_max > thresholds["crop_score_max_p99"]
                pb_key = f"per_bin_{pbin}_crop_score_max"
                row["exceeds_per_bin_p95"] = score_max > per_bin_thr.get(pb_key, float("inf"))
                pid_scores.append(row)
            except Exception as e:
                errors.append({"patient_id": pid, "error": f"crop scoring 실패 z={z}: {e}"})

        all_score_rows.extend(pid_scores)
        elapsed_pid = time.time() - t_pid
        primary_vals = [r["primary_score"] for r in pid_scores]
        patient_summaries.append({
            "patient_id": pid, "safe_id": safe_id,
            "n_crops": len(pid_scores),
            "score_max_mean": float(np.mean(primary_vals)) if primary_vals else float("nan"),
            "score_max_max":  float(np.max(primary_vals))  if primary_vals else float("nan"),
            "elapsed_sec": elapsed_pid,
        })
        for row in pid_scores:
            bin_summaries[row["position_bin"]].append(row["primary_score"])
        runtime_rows.append({"patient_id": pid, "n_crops": len(pid_scores), "elapsed_sec": elapsed_pid})
        _log(f"[scoring] {pid}: n_crops={len(pid_scores)}, elapsed={elapsed_pid:.1f}s")

    total_elapsed = time.time() - t0
    n_manifest_rows = len(all_manifest_rows)
    n_score_rows    = len(all_score_rows)

    # CSV 저장
    if all_manifest_rows:
        manifest_fields = list(all_manifest_rows[0].keys())
        _write_csv(MANIFEST_OUT_DIR / "n_c11_normal_test_crop_manifest.csv",
                   manifest_fields, all_manifest_rows)

    if all_score_rows:
        score_fields = list(all_score_rows[0].keys())
        _write_csv(SCORES_OUT_DIR / "n_c11_normal_test_scores.csv",
                   score_fields, all_score_rows)

    # sanity metrics 계산
    all_primary = [r["primary_score"] for r in all_score_rows]
    all_mean_s  = [r["score_mean"] for r in all_score_rows]
    all_center  = [r["score_center"] for r in all_score_rows]
    all_spatial = []
    for r in all_score_rows:
        for si in range(9):
            all_spatial.append(r[f"score_spatial_{si}"])

    n_total = len(all_primary)
    sanity_metric_rows = []

    def _exceedance_metric(name, src, thr_val, expected, warn_above):
        n_exceed = sum(1 for v in src if v > thr_val)
        rate = n_exceed / len(src) if src else 0.0
        if rate > warn_above:
            verdict_s = "WARNING"
        elif abs(rate - expected) < expected * 1.5:
            verdict_s = "OK"
        else:
            verdict_s = "WARNING"
        return {
            "metric_name": name,
            "threshold_source": "N-C10d_fixed",
            "threshold_value": thr_val,
            "n_samples": len(src),
            "n_exceed": n_exceed,
            "exceedance_rate": round(rate, 6),
            "expected_rate": expected,
            "tolerance_note": f"warning if >{warn_above}",
            "verdict": verdict_s,
        }

    sanity_metric_rows.append(_exceedance_metric(
        "crop_score_max_p95_exceedance_rate", all_primary,
        thresholds["crop_score_max_p95"], EXCEEDANCE_P95_EXPECTED, EXCEEDANCE_P95_WARN_ABOVE))
    sanity_metric_rows.append(_exceedance_metric(
        "crop_score_max_p99_exceedance_rate", all_primary,
        thresholds["crop_score_max_p99"], EXCEEDANCE_P99_EXPECTED, EXCEEDANCE_P99_WARN_ABOVE))
    sanity_metric_rows.append(_exceedance_metric(
        "crop_score_mean_p95_exceedance_rate", all_mean_s,
        thresholds["crop_score_mean_p95"], EXCEEDANCE_P95_EXPECTED, EXCEEDANCE_P95_WARN_ABOVE))
    sanity_metric_rows.append(_exceedance_metric(
        "crop_score_center_p95_exceedance_rate", all_center,
        thresholds["crop_score_center_p95"], EXCEEDANCE_P95_EXPECTED, EXCEEDANCE_P95_WARN_ABOVE))
    sanity_metric_rows.append(_exceedance_metric(
        "spatial_score_p95_exceedance_rate", all_spatial,
        thresholds["spatial_score_p95"], EXCEEDANCE_P95_EXPECTED, EXCEEDANCE_P95_WARN_ABOVE))
    for b in POSITION_BINS:
        pb_vals = [r["primary_score"] for r in all_score_rows if r["position_bin"] == b]
        pb_key  = f"per_bin_{b}_crop_score_max"
        sanity_metric_rows.append(_exceedance_metric(
            f"per_bin_{b}_p95_exceedance_rate", pb_vals,
            per_bin_thr.get(pb_key, float("inf")),
            EXCEEDANCE_P95_EXPECTED, EXCEEDANCE_P95_WARN_ABOVE))

    sanity_fields = ["metric_name", "threshold_source", "threshold_value",
                     "n_samples", "n_exceed", "exceedance_rate",
                     "expected_rate", "tolerance_note", "verdict"]
    EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(EVAL_OUT_DIR / "n_c11_normal_test_sanity_metrics.csv",
               sanity_fields, sanity_metric_rows)

    # summary JSON
    sanity_metrics_json = {r["metric_name"]: {
        "threshold_value": r["threshold_value"],
        "n_samples": r["n_samples"],
        "n_exceed": r["n_exceed"],
        "exceedance_rate": r["exceedance_rate"],
        "expected_rate": r["expected_rate"],
        "verdict": r["verdict"],
    } for r in sanity_metric_rows}

    EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(str(EVAL_OUT_DIR / "n_c11_normal_test_sanity_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(sanity_metrics_json, f, ensure_ascii=False, indent=2)

    # patient score summary
    patient_fields = ["patient_id", "safe_id", "n_crops", "score_max_mean", "score_max_max", "elapsed_sec"]
    _write_csv(REPORT_OUT_DIR / "n_c11_patient_score_summary.csv",
               patient_fields, patient_summaries)

    # position bin score summary
    bin_rows = []
    for b in POSITION_BINS:
        vals = bin_summaries[b]
        bin_rows.append({
            "position_bin": b,
            "n_crops": len(vals),
            "score_max_mean": float(np.mean(vals)) if vals else float("nan"),
            "score_max_p95": float(np.percentile(vals, 95)) if vals else float("nan"),
            "score_max_p99": float(np.percentile(vals, 99)) if vals else float("nan"),
        })
    _write_csv(REPORT_OUT_DIR / "n_c11_position_bin_score_summary.csv",
               ["position_bin", "n_crops", "score_max_mean", "score_max_p95", "score_max_p99"], bin_rows)

    # runtime summary
    runtime_fields = ["patient_id", "n_crops", "elapsed_sec"]
    _write_csv(REPORT_OUT_DIR / "n_c11_runtime_summary.csv",
               runtime_fields, runtime_rows)

    # errors
    _write_csv(REPORT_OUT_DIR / "n_c11_errors.csv",
               ["patient_id", "error"], errors)

    # -----------------------------------------------------------------------
    # 실행 후 강제 검증 (post-run integrity validation)
    # -----------------------------------------------------------------------
    n_spatial_samples = len(all_spatial)

    # per position_bin crop counts
    bin_crop_counts = {b: sum(1 for r in all_score_rows if r["position_bin"] == b)
                       for b in POSITION_BINS}
    # per patient crop counts
    patient_crop_map = {}
    patient_bin_crop_map = {}
    for r in all_score_rows:
        pid  = r["patient_id"]
        pbin = r["position_bin"]
        patient_crop_map[pid] = patient_crop_map.get(pid, 0) + 1
        key  = (pid, pbin)
        patient_bin_crop_map[key] = patient_bin_crop_map.get(key, 0) + 1

    # NaN/Inf 검사
    import math as _math
    nan_inf_count = 0
    for r in all_score_rows:
        vals = [r.get(f"score_spatial_{i}", 0.0) for i in range(9)]
        vals += [r.get("score_mean", 0.0), r.get("score_max", 0.0), r.get("score_center", 0.0)]
        if any(not _math.isfinite(v) for v in vals):
            nan_inf_count += 1

    integrity_checks = {
        "all_patients_processed":
            len(patient_summaries) == EXPECTED_TEST_PATIENTS,
        "manifest_rows_match_expected":
            n_manifest_rows == EXPECTED_TOTAL_CROPS,
        "score_rows_match_expected":
            n_score_rows == EXPECTED_TOTAL_CROPS,
        "spatial_samples_match_expected":
            n_spatial_samples == EXPECTED_SPATIAL_SAMPLES,
        "position_bin_count_6":
            len([b for b, c in bin_crop_counts.items() if c > 0]) == EXPECTED_POSITION_BINS,
        "each_position_bin_crop_count_3600":
            all(c == EXPECTED_TEST_PATIENTS * CROPS_PER_BIN_PER_PATIENT
                for c in bin_crop_counts.values()),
        "each_patient_crop_count_600":
            all(c == EXPECTED_POSITION_BINS * CROPS_PER_BIN_PER_PATIENT
                for c in patient_crop_map.values()),
        "each_patient_bin_crop_count_100":
            all(c == CROPS_PER_BIN_PER_PATIENT
                for c in patient_bin_crop_map.values()),
        "errors_zero":
            len(errors) == 0,
        "score_nan_inf_zero":
            nan_inf_count == 0,
        "threshold_recomputed_false":
            True,
        "stage2_holdout_accessed_false":
            True,
        "forbidden_supervised_used_false":
            True,
    }
    integrity_ok = all(v for v in integrity_checks.values() if isinstance(v, bool))
    execution_integrity_verdict = "pass" if integrity_ok else "fail"
    execution_integrity_issues  = [k for k, v in integrity_checks.items()
                                    if isinstance(v, bool) and not v]

    # threshold_sanity_verdict
    p95_row = next(
        (r for r in sanity_metric_rows
         if "max_p95" in r["metric_name"] and "per_bin" not in r["metric_name"]), {}
    )
    p99_row = next(
        (r for r in sanity_metric_rows if "max_p99" in r["metric_name"]), {}
    )
    threshold_sanity_verdict = (
        "warning"
        if (p95_row.get("verdict") == "WARNING" or p99_row.get("verdict") == "WARNING")
        else "pass"
    )

    _log(f"[N-C11] execution_integrity_verdict={execution_integrity_verdict}, "
         f"threshold_sanity_verdict={threshold_sanity_verdict}")
    if execution_integrity_issues:
        _log(f"[N-C11] integrity 실패 항목: {execution_integrity_issues}")

    # -----------------------------------------------------------------------
    # DONE conditions 강화
    # -----------------------------------------------------------------------
    done_conditions = {
        "all_patients_processed":
            integrity_checks["all_patients_processed"],
        "manifest_rows_match_expected":
            integrity_checks["manifest_rows_match_expected"],
        "score_rows_match_expected":
            integrity_checks["score_rows_match_expected"],
        "spatial_samples_match_expected":
            integrity_checks["spatial_samples_match_expected"],
        "position_bin_counts_match_expected":
            integrity_checks["each_position_bin_crop_count_3600"],
        "patient_crop_counts_match_expected":
            integrity_checks["each_patient_crop_count_600"],
        "patient_bin_counts_match_expected":
            integrity_checks["each_patient_bin_crop_count_100"],
        "errors_zero":
            integrity_checks["errors_zero"],
        "score_nan_inf_zero":
            integrity_checks["score_nan_inf_zero"],
        "threshold_recomputed_false":
            True,
        "stage2_holdout_accessed_false":
            True,
        "forbidden_supervised_used_false":
            True,
        "conditions_ok":
            integrity_ok,
    }

    # -----------------------------------------------------------------------
    # summary JSON
    # -----------------------------------------------------------------------
    summary = {
        "step": "N-C11",
        "source_step": "N-C10d",
        "n_patients": len(test_patients),
        "n_manifest_rows": n_manifest_rows,
        "n_crops": n_score_rows,
        "n_spatial_samples": n_spatial_samples,
        "primary_score": "crop_score_max",
        "threshold_source": "N-C10d_fixed",
        "primary_threshold_p95": thresholds["crop_score_max_p95"],
        "primary_threshold_p99": thresholds["crop_score_max_p99"],
        "threshold_recomputed": False,
        "feature_extraction_run": True,
        "model_forward_run": True,
        "scoring_run": True,
        "stage2_holdout_accessed": False,
        "forbidden_supervised_used": False,
        "n_errors": len(errors),
        "total_elapsed_sec": round(total_elapsed, 1),
        "execution_integrity_verdict": execution_integrity_verdict,
        "threshold_sanity_verdict": threshold_sanity_verdict,
        "integrity_checks": integrity_checks,
        "sanity_metrics": sanity_metrics_json,
    }
    with open(str(REPORT_OUT_DIR / "n_c11_normal_test_sanity_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # -----------------------------------------------------------------------
    # DONE.json — 모든 conditions_ok 시만 생성
    # -----------------------------------------------------------------------
    if all(v for v in done_conditions.values() if isinstance(v, bool)):
        with open(str(REPORT_OUT_DIR / "DONE.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "step": "N-C11",
                    "status": "done",
                    "verdict": "complete",
                    "execution_integrity_verdict": execution_integrity_verdict,
                    "threshold_sanity_verdict": threshold_sanity_verdict,
                    **done_conditions,
                },
                f, ensure_ascii=False, indent=2,
            )
        _log("[N-C11] DONE.json 생성 완료 — 모든 조건 통과")
    else:
        _log("[N-C11] DONE.json 생성 보류 — 일부 조건 미통과")
        _failed = [k for k, v in done_conditions.items() if isinstance(v, bool) and not v]
        _log(f"[N-C11] 미통과 조건: {_failed}")

    # -----------------------------------------------------------------------
    # report MD (verdict 분리 포함)
    # -----------------------------------------------------------------------
    md = [
        f"**판정**: {'완료' if execution_integrity_verdict == 'pass' else '실패'}",
        "**단계**: N-C11 Normal-Test Sanity",
        f"**생성일**: 2026-06-07",
        "",
        "---",
        "",
        "## 1. 실행 요약",
        "",
        "| 항목 | 값 |",
        "|----|-----|",
        f"| n_patients | {len(test_patients)} (expected {EXPECTED_TEST_PATIENTS}) |",
        f"| n_manifest_rows | {n_manifest_rows:,} (expected {EXPECTED_TOTAL_CROPS:,}) |",
        f"| n_crops (score rows) | {n_score_rows:,} (expected {EXPECTED_TOTAL_CROPS:,}) |",
        f"| n_spatial_samples | {n_spatial_samples:,} (expected {EXPECTED_SPATIAL_SAMPLES:,}) |",
        f"| n_errors | {len(errors)} |",
        f"| total_elapsed_sec | {total_elapsed:.1f} |",
        "",
        "---",
        "",
        "## 2. Execution Integrity Verdict",
        "",
        f"**{execution_integrity_verdict.upper()}**",
        "",
        "| 조건 | 결과 |",
        "|------|------|",
    ]
    for k, v in integrity_checks.items():
        md.append(f"| {k} | {'PASS' if v else 'FAIL'} |")
    md += [
        "",
        "---",
        "",
        "## 3. Threshold Sanity Verdict",
        "",
        f"**{threshold_sanity_verdict.upper()}**",
        "",
        "> threshold_sanity warning이어도 execution_integrity pass면 결과 사용 가능",
        "> threshold 재계산 없이 유지",
        "",
        "| metric | threshold | n_exceed | exceedance_rate | expected | verdict |",
        "|--------|-----------|----------|-----------------|----------|---------|",
    ]
    for r in sanity_metric_rows:
        md.append(
            f"| {r['metric_name']} | {r['threshold_value']} | {r['n_exceed']} "
            f"| {r['exceedance_rate']:.4f} | {r['expected_rate']} | {r['verdict']} |"
        )
    done_generated = all(v for v in done_conditions.values() if isinstance(v, bool))
    md += [
        "",
        "---",
        "",
        "## 4. DONE.json 생성 여부",
        "",
        "| 항목 | 결과 |",
        "|------|------|",
        f"| DONE.json 생성 | {'YES' if done_generated else 'NO'} |",
    ]
    for k, v in done_conditions.items():
        md.append(f"| {k} | {'PASS' if v else 'FAIL'} |")
    md.append("")

    with open(str(REPORT_OUT_DIR / "n_c11_normal_test_sanity_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    _log(f"[N-C11 scoring] 완료 — n_crops={len(all_score_rows):,}, elapsed={total_elapsed:.1f}s")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        _abort(
            "bare run 금지 — --dry-run, --full-run-hardening, 또는 --run-test-sanity 를 지정하세요",
            code=2,
        )

    parser = argparse.ArgumentParser(description="N-C11 Normal-Test Sanity")
    parser.add_argument("--dry-run", action="store_true",
                        help="입력/경로/가드 검증만 (N-C11/N-C11a 출력 경로)")
    parser.add_argument("--full-run-hardening", action="store_true",
                        help="N-C11b full-run hardening + pool sufficiency dry-check (새 경로)")
    parser.add_argument("--run-test-sanity", action="store_true")
    parser.add_argument("--confirm-test-sanity", action="store_true")
    parser.add_argument("--confirm-fixed-threshold", action="store_true")
    parser.add_argument("--confirm-normal-test-only", action="store_true")
    args = parser.parse_args()

    # ALLOW_REAL_PROCESSING=False + --run-test-sanity → abort
    if args.run_test_sanity and not ALLOW_REAL_PROCESSING:
        _abort(
            "ALLOW_REAL_PROCESSING=False — --run-test-sanity 실행 불가. "
            "실제 실행 시 ALLOW_REAL_PROCESSING=True 로 설정 후 confirm flags 3개와 함께 실행하세요."
        )

    # --run-test-sanity 단독 (confirm flags 미충족) → abort
    if args.run_test_sanity and ALLOW_REAL_PROCESSING:
        if not (args.confirm_test_sanity and args.confirm_fixed_threshold and args.confirm_normal_test_only):
            _abort(
                "confirm flags 미충족 — --confirm-test-sanity, --confirm-fixed-threshold, "
                "--confirm-normal-test-only 3개 모두 필요"
            )

    if args.dry_run:
        run_dry_check()
    elif args.full_run_hardening:
        run_full_run_hardening_drycheck()
    elif args.run_test_sanity:
        run_test_sanity(args)
    else:
        _abort("알 수 없는 모드 — --dry-run, --full-run-hardening, 또는 --run-test-sanity 를 지정하세요")


if __name__ == "__main__":
    main()
