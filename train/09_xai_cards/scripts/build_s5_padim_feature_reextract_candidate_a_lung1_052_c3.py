"""
S5 Feature Re-extraction — Candidate A (LUNG1-052__c3)
=======================================================
대상: LUNG1-052__c3 / stage1_dev only
단계: smoke script (static 검사 단계 — 실제 CT load/model forward 금지)

기반: build_s5_padim_feature_reextract_1case.py (LUNG1-320__c2) 구조 유지
주의: 기존 LUNG1-320 script 수정 금지 — 이 파일은 candidate A 전용 신규 파일

중요 — local_z / report_slice 구분:
  CT_INDEX_Z = 51        ← CT npy indexing에 사용 (local_z)
  REPORT_SLICE_INDEX = 106  ← global/reporting 전용 (CT index 절대 사용 금지)
  LUNG1-052 offset = 55  (LUNG1-320 offset 51과 다름 — 환자별 독립 확인)

실행 방법:
  --selftest         : 정적 selftest 60+ 항목 (허용)
  --dry-run          : 경로 확인, CT load 없음 (허용)
  --plan-only        : 실행 계획 출력 (허용)
  --run-smoke --confirm-extract : 실제 실행 (guard True 시에만)

금지 (현재 모든 guard=False):
  bare 실행         → BLOCKED exit 2
  --run-smoke 단독  → BLOCKED exit 2
  --run-smoke --confirm-extract → BLOCKED exit 2 (guard False)

주의 사항 (진단/방법론):
  이 스크립트의 output은 XAI 설명 목적의 feature attribution이다.
  Grad-CAM이 아니며 pixel attribution이 아니다.
  특정 소견, 암, 병변을 진단하거나 확정하지 않는다.
"""

from __future__ import annotations

import argparse
import os
import sys

# ============================================================
# A. Path / Metadata constants
# ============================================================

REPO_ROOT = "/home/jinhy/project/lung-ct-anomaly"

CASE_ID            = "LUNG1-052__c3"
VOLUME_ID          = "NSCLC_LUNG1-052__d4a19cc211"
CT_INDEX_Z         = 51           # CT npy indexing에 사용하는 local z
REPORT_SLICE_INDEX = 106          # global/reporting index (CT index 아님)
PATCH_Y0           = 288
PATCH_X0           = 128
PATCH_Y1           = 320
PATCH_X1           = 160
PATCH_SIZE         = 32
POSITION_BIN       = "lower_central"
EXPECTED_SCORE     = 39.34
ROI_COVERAGE       = 1.000
SPLIT              = "stage1_dev"

# local_z indexing 정책 명시
LOCAL_Z_USED_FOR_CT_INDEXING   = True
SLICE_INDEX_USED_FOR_CT_INDEXING = False

# component bbox (metadata note 전용 — feature extraction에 사용 금지)
COMPONENT_BBOX_NOTE = "[y0=192, x0=112, y1=336, x1=192]"

# ============================================================
# B. preprocessing parity metadata
# ============================================================

HU_MIN               = -1000.0
HU_MAX               =   200.0
IMAGENET_MEAN        = [0.485, 0.456, 0.406]
IMAGENET_STD         = [0.229, 0.224, 0.225]
THREE_CHANNEL_POLICY = "1ch → 3ch stack (repeat)"
BACKBONE             = "ResNet18"
TAPPED_LAYERS        = ["layer1", "layer2", "layer3"]
RAW_FEATURE_DIM      = 448
SELECTED_FEATURE_DIM = 100

# ============================================================
# C. Input paths
# ============================================================

CT_HU_NPY_PATH = (
    "/mnt/c/Users/jinhy/Desktop/"
    "NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/"
    "volumes_npy/NSCLC_LUNG1-052__d4a19cc211/ct_hu.npy"
)

STATS_NPZ_PATH = os.path.join(
    REPO_ROOT,
    "outputs/position-aware-padim-v1/models/padim_v2_roi0_0/distributions/position_bin_stats.npz"
)

SELECTED_INDICES_NPY_PATH = os.path.join(
    REPO_ROOT,
    "outputs/position-aware-padim-v1/models/padim_v1/distributions/selected_feature_indices.npy"
)

FEATURE_EXTRACTOR_MODULE_PATH = os.path.join(
    REPO_ROOT,
    "src/position_aware_padim/feature_extractor.py"
)

PREPROCESSING_MODULE_PATH = os.path.join(
    REPO_ROOT,
    "src/position_aware_padim/preprocessing.py"
)

RESNET18_CACHE_PATH = (
    "/home/jinhy/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
)

# stage2_holdout 경로 (접근 금지 확인용 — 절대 load 금지)
_STAGE2_HOLDOUT_SENTINEL = "stage2_holdout"

# ============================================================
# D. Output root (smoke run 단계에서만 생성)
# ============================================================

OUTPUT_ROOT = os.path.join(
    REPO_ROOT,
    "outputs/position-aware-padim-v1/reports/explanation_cards/"
    "s5_lung1_052_c3_feature_reextract_smoke_v1"
)

# LUNG1-320 output root (충돌 방지 확인용 — 절대 수정 금지)
_LUNG1_320_OUTPUT_ROOT = os.path.join(
    REPO_ROOT,
    "outputs/position-aware-padim-v1/reports/explanation_cards/"
    "s5_feature_reextract_1case_smoke_v1"
)

# ============================================================
# E. Basic guards (all False — 실제 실행 금지)
# ============================================================

ALLOW_CT_LOAD            = False
ALLOW_MODEL_FORWARD      = False
ALLOW_FEATURE_EXTRACTION = False
ALLOW_WRITE_FEATURE_VECTOR = False
ALLOW_GPU                = False
ALLOW_STAGE2_HOLDOUT     = False
ALLOW_FULL_300           = False


# ============================================================
# F. local_z guard
# ============================================================

def _assert_local_z_guards():
    """local_z / slice_index 혼동 방지 guard."""
    assert CT_INDEX_Z == 51,          f"CT_INDEX_Z guard FAIL: got {CT_INDEX_Z}"
    assert REPORT_SLICE_INDEX == 106, f"REPORT_SLICE_INDEX guard FAIL: got {REPORT_SLICE_INDEX}"
    assert CT_INDEX_Z != REPORT_SLICE_INDEX, (
        f"CT_INDEX_Z({CT_INDEX_Z}) == REPORT_SLICE_INDEX({REPORT_SLICE_INDEX}) — 혼동 위험"
    )
    assert LOCAL_Z_USED_FOR_CT_INDEXING is True, \
        "LOCAL_Z_USED_FOR_CT_INDEXING must be True"
    assert SLICE_INDEX_USED_FOR_CT_INDEXING is False, \
        "SLICE_INDEX_USED_FOR_CT_INDEXING must be False"
    assert (PATCH_Y1 - PATCH_Y0) == PATCH_SIZE, \
        f"patch height mismatch: {PATCH_Y1 - PATCH_Y0} != {PATCH_SIZE}"
    assert (PATCH_X1 - PATCH_X0) == PATCH_SIZE, \
        f"patch width mismatch: {PATCH_X1 - PATCH_X0} != {PATCH_SIZE}"


# ============================================================
# G. selftest (60+ 항목)
# ============================================================

def run_selftest() -> int:
    """정적 selftest. 모든 guard / 상수 / 경로 / 정책 검증. 실제 load 없음."""
    passed = []
    failed = []

    def chk(name: str, cond: bool, msg: str = ""):
        if cond:
            passed.append(name)
        else:
            failed.append(f"{name}: {msg}")

    # --- 1. guards all False ---
    chk("01_guard_ct_load_false",            not ALLOW_CT_LOAD,             "ALLOW_CT_LOAD must be False")
    chk("02_guard_model_forward_false",      not ALLOW_MODEL_FORWARD,       "ALLOW_MODEL_FORWARD must be False")
    chk("03_guard_feature_extraction_false", not ALLOW_FEATURE_EXTRACTION,  "ALLOW_FEATURE_EXTRACTION must be False")
    chk("04_guard_write_feature_false",      not ALLOW_WRITE_FEATURE_VECTOR,"ALLOW_WRITE_FEATURE_VECTOR must be False")
    chk("05_guard_gpu_false",                not ALLOW_GPU,                 "ALLOW_GPU must be False")
    chk("06_guard_stage2_holdout_false",     not ALLOW_STAGE2_HOLDOUT,      "ALLOW_STAGE2_HOLDOUT must be False")
    chk("07_guard_full300_false",            not ALLOW_FULL_300,            "ALLOW_FULL_300 must be False")

    # --- 2. CASE_ID / VOLUME_ID exact ---
    chk("08_case_id_exact",   CASE_ID == "LUNG1-052__c3",
        f"got {CASE_ID}")
    chk("09_volume_id_exact", VOLUME_ID == "NSCLC_LUNG1-052__d4a19cc211",
        f"got {VOLUME_ID}")

    # --- 3. local_z / report_slice ---
    chk("10_ct_index_z_51",              CT_INDEX_Z == 51,
        f"got {CT_INDEX_Z}")
    chk("11_report_slice_index_106",     REPORT_SLICE_INDEX == 106,
        f"got {REPORT_SLICE_INDEX}")
    chk("12_ct_index_z_ne_report_slice", CT_INDEX_Z != REPORT_SLICE_INDEX,
        "local_z == report_slice — 혼동 위험")
    chk("13_local_z_used_for_ct_indexing",   LOCAL_Z_USED_FOR_CT_INDEXING is True,
        "LOCAL_Z_USED_FOR_CT_INDEXING must be True")
    chk("14_slice_index_not_for_ct",         SLICE_INDEX_USED_FOR_CT_INDEXING is False,
        "SLICE_INDEX_USED_FOR_CT_INDEXING must be False")
    import inspect as _inspect
    _mod_src = _inspect.getsource(sys.modules[__name__])
    chk("15_report_slice_policy_exists",
        "CT index 아님" in _mod_src,
        "local_z policy string 없음")

    # --- 4. patch bbox ---
    chk("16_patch_y0_288",     PATCH_Y0 == 288,  f"got {PATCH_Y0}")
    chk("17_patch_x0_128",     PATCH_X0 == 128,  f"got {PATCH_X0}")
    chk("18_patch_y1_320",     PATCH_Y1 == 320,  f"got {PATCH_Y1}")
    chk("19_patch_x1_160",     PATCH_X1 == 160,  f"got {PATCH_X1}")
    chk("20_patch_height_32",  (PATCH_Y1 - PATCH_Y0) == PATCH_SIZE,
        f"{PATCH_Y1 - PATCH_Y0} != 32")
    chk("21_patch_width_32",   (PATCH_X1 - PATCH_X0) == PATCH_SIZE,
        f"{PATCH_X1 - PATCH_X0} != 32")

    CT_SHAPE_Z, CT_SHAPE_H, CT_SHAPE_W = 213, 512, 512
    chk("22_patch_y_in_bounds",
        0 <= PATCH_Y0 < PATCH_Y1 <= CT_SHAPE_H,
        f"y0={PATCH_Y0}, y1={PATCH_Y1}, H={CT_SHAPE_H}")
    chk("23_patch_x_in_bounds",
        0 <= PATCH_X0 < PATCH_X1 <= CT_SHAPE_W,
        f"x0={PATCH_X0}, x1={PATCH_X1}, W={CT_SHAPE_W}")
    chk("24_local_z_in_bounds",
        0 <= CT_INDEX_Z < CT_SHAPE_Z,
        f"local_z={CT_INDEX_Z} out of z range [0, {CT_SHAPE_Z})")
    chk("25_patch_bbox_single_not_component",
        [PATCH_Y0, PATCH_X0, PATCH_Y1, PATCH_X1] == [288, 128, 320, 160],
        "single patch bbox must be [288,128,320,160]")
    chk("26_component_bbox_note_only",
        "192" in COMPONENT_BBOX_NOTE and "112" in COMPONENT_BBOX_NOTE,
        "component bbox note missing")

    # --- 5. position_bin / score / coverage ---
    chk("27_position_bin_lower_central",
        POSITION_BIN == "lower_central",
        f"got {POSITION_BIN}")
    chk("28_expected_score_39_34",
        abs(EXPECTED_SCORE - 39.34) < 0.01,
        f"got {EXPECTED_SCORE}")
    chk("29_roi_coverage_1_000",
        abs(ROI_COVERAGE - 1.000) < 1e-6,
        f"got {ROI_COVERAGE}")
    chk("30_split_stage1_dev",
        SPLIT == "stage1_dev",
        f"got {SPLIT}")

    # --- 6. preprocessing constants ---
    chk("31_hu_min_neg1000",       HU_MIN == -1000.0,         f"got {HU_MIN}")
    chk("32_hu_max_200",           HU_MAX == 200.0,            f"got {HU_MAX}")
    chk("33_imagenet_mean_3",      len(IMAGENET_MEAN) == 3,    "IMAGENET_MEAN must be length 3")
    chk("34_imagenet_std_3",       len(IMAGENET_STD) == 3,     "IMAGENET_STD must be length 3")
    chk("35_3ch_policy_exists",    "3ch" in THREE_CHANNEL_POLICY, "3ch repeat policy not defined")

    # --- 7. backbone / layers / dims ---
    chk("36_backbone_resnet18",    BACKBONE == "ResNet18",    f"got {BACKBONE}")
    chk("37_tapped_layers",        TAPPED_LAYERS == ["layer1", "layer2", "layer3"],
        f"got {TAPPED_LAYERS}")
    chk("38_raw_dim_448",          RAW_FEATURE_DIM == 448,    f"got {RAW_FEATURE_DIM}")
    chk("39_selected_dim_100",     SELECTED_FEATURE_DIM == 100, f"got {SELECTED_FEATURE_DIM}")

    # --- 8. no EfficientNet branch ---
    chk("40_no_effnet_in_backbone",
        "efficientnet" not in BACKBONE.lower(),
        f"EfficientNet branch 혼입 위험: {BACKBONE}")
    chk("41_no_effnet_in_extractor_path",
        "effnet" not in FEATURE_EXTRACTOR_MODULE_PATH.lower(),
        f"effnet extractor 경로 혼입: {FEATURE_EXTRACTOR_MODULE_PATH}")

    # --- 9. 파일 존재 확인 (load 없음) ---
    chk("42_selected_indices_exists",
        os.path.isfile(SELECTED_INDICES_NPY_PATH),
        f"not found: {SELECTED_INDICES_NPY_PATH}")
    chk("43_feature_extractor_exists",
        os.path.isfile(FEATURE_EXTRACTOR_MODULE_PATH),
        f"not found: {FEATURE_EXTRACTOR_MODULE_PATH}")
    chk("44_preprocessing_module_exists",
        os.path.isfile(PREPROCESSING_MODULE_PATH),
        f"not found: {PREPROCESSING_MODULE_PATH}")
    chk("45_stats_npz_exists",
        os.path.isfile(STATS_NPZ_PATH),
        f"not found: {STATS_NPZ_PATH}")
    chk("46_ct_path_recorded",
        len(CT_HU_NPY_PATH) > 10,
        "CT_HU_NPY_PATH empty")
    chk("47_resnet18_cache_exists",
        os.path.isfile(RESNET18_CACHE_PATH),
        f"not found: {RESNET18_CACHE_PATH}")

    # --- 10. stage2_holdout 방지 ---
    chk("48_stage2_sentinel_not_in_ct_path",
        _STAGE2_HOLDOUT_SENTINEL not in CT_HU_NPY_PATH,
        "CT 경로에 stage2_holdout 포함 위험")
    chk("49_stage2_holdout_guard_false",
        not ALLOW_STAGE2_HOLDOUT,
        "ALLOW_STAGE2_HOLDOUT must be False")

    # --- 11. output root 분리 ---
    chk("50_output_root_in_outputs_dir",
        "outputs/" in OUTPUT_ROOT,
        f"output root이 outputs/ 외부: {OUTPUT_ROOT}")
    chk("51_output_root_not_lung1_320",
        OUTPUT_ROOT != _LUNG1_320_OUTPUT_ROOT,
        "output root이 LUNG1-320과 충돌")
    chk("52_output_root_contains_052",
        "052" in OUTPUT_ROOT,
        f"output root에 052 없음: {OUTPUT_ROOT}")

    # --- 12. 실행 차단 상태 ---
    chk("53_ct_load_blocked",         not ALLOW_CT_LOAD,             "CT load guard 없음")
    chk("54_model_forward_blocked",   not ALLOW_MODEL_FORWARD,       "model forward guard 없음")
    chk("55_feature_blocked",         not ALLOW_FEATURE_EXTRACTION,  "feature extraction guard 없음")
    chk("56_write_blocked",           not ALLOW_WRITE_FEATURE_VECTOR,"write guard 없음")
    chk("57_gpu_blocked",             not ALLOW_GPU,                 "GPU guard 없음")
    chk("58_full300_blocked",         not ALLOW_FULL_300,            "full300 guard 없음")

    # --- 13. run_smoke에 --confirm-extract 체크 존재 (inspect) ---
    import inspect
    smoke_src = inspect.getsource(run_smoke)
    chk("59_run_smoke_requires_confirm",
        "confirm_extract" in smoke_src,
        "run_smoke에 --confirm-extract 체크 없음")

    # --- 14. run_smoke에 mmap_mode="r" 정책 존재 ---
    chk("60_ct_load_mmap_mode_r",
        'mmap_mode="r"' in smoke_src or "mmap_mode='r'" in smoke_src,
        "CT load에 mmap_mode='r' 없음 — 일반 np.load 금지")

    # --- 15. metadata schema 검증 ---
    schema_keys = _get_metadata_schema_keys()
    chk("61_schema_ct_index_z",
        "ct_index_z" in schema_keys, "metadata schema에 ct_index_z 없음")
    chk("62_schema_report_slice_index",
        "report_slice_index" in schema_keys, "metadata schema에 report_slice_index 없음")
    chk("63_schema_local_z_used",
        "local_z_used_for_ct_indexing" in schema_keys,
        "local_z_used_for_ct_indexing 필드 없음")
    chk("64_schema_slice_index_not_for_ct",
        "slice_index_used_for_ct_indexing" in schema_keys,
        "slice_index_used_for_ct_indexing 필드 없음")
    chk("65_schema_existing_artifacts_modified",
        "existing_artifacts_modified" in schema_keys,
        "existing_artifacts_modified 필드 없음")
    chk("66_schema_not_gradcam_caveat",
        "not_gradcam_caveat" in schema_keys,
        "not_gradcam_caveat 필드 없음")
    chk("67_schema_not_diagnostic_caveat",
        "not_diagnostic_caveat" in schema_keys,
        "not_diagnostic_caveat 필드 없음")
    chk("68_schema_position_bin",
        "position_bin" in schema_keys, "metadata schema에 position_bin 없음")
    chk("69_schema_preprocessing_parity",
        "preprocessing_hu_min" in schema_keys, "metadata schema에 preprocessing 항목 없음")
    chk("70_schema_score_recomputed_flag",
        "score_recomputed" in schema_keys, "score_recomputed 필드 없음")
    chk("71_schema_no_contribution_calc",
        "contribution_calculated" not in schema_keys,
        "contribution_calculated는 이 단계에서 schema에 포함 금지")

    # --- 결과 출력 ---
    total = len(passed) + len(failed)
    print(f"\n=== SELFTEST 결과 (candidate A: {CASE_ID}) ===")
    print(f"PASS: {len(passed)}/{total}")
    if failed:
        print(f"FAIL: {len(failed)}/{total}")
        for f in failed:
            print(f"  FAIL: {f}")
        print("\n판정: FAIL")
        return 1
    else:
        print("모든 selftest 통과.")
        print("판정: PASS")
        return 0


def _get_metadata_schema_keys() -> list:
    """run 단계에서 생성할 metadata JSON 필드 목록 (설계 참조)."""
    return [
        "case_id",
        "volume_id",
        "ct_index_z",
        "report_slice_index",
        "local_z_used_for_ct_indexing",
        "slice_index_used_for_ct_indexing",
        "patch_bbox_yxyx",
        "patch_size",
        "position_bin",
        "roi_coverage",
        "expected_score",
        "backbone",
        "tapped_layers",
        "raw_feature_dim",
        "selected_feature_dim",
        "selected_indices_source",
        "preprocessing_hu_min",
        "preprocessing_hu_max",
        "preprocessing_image_net_norm",
        "three_channel_policy",
        "feature_extractor_source",
        "ct_load_occurred",
        "model_forward_occurred",
        "feature_extraction_occurred",
        "gpu_used",
        "stage2_holdout_accessed",
        "score_recomputed",
        "threshold_recomputed",
        "existing_artifacts_modified",
        "not_gradcam_caveat",
        "not_diagnostic_caveat",
        "component_bbox_note",
    ]


# ============================================================
# H. dry-run
# ============================================================

def run_dry_run() -> int:
    """경로 확인 및 metadata 출력. CT load / model forward / feature extraction 없음."""
    print("\n=== DRY-RUN (candidate A: LUNG1-052__c3) ===")
    print(f"CASE_ID            : {CASE_ID}")
    print(f"VOLUME_ID          : {VOLUME_ID}")
    print(f"CT_INDEX_Z         : {CT_INDEX_Z}  ← CT npy indexing에 사용 (local_z)")
    print(f"REPORT_SLICE_INDEX : {REPORT_SLICE_INDEX}  ← global/reporting 전용 (CT index 절대 사용 금지)")
    print(f"LUNG1-052 offset   : {REPORT_SLICE_INDEX - CT_INDEX_Z}  (LUNG1-320 offset=51과 다름)")
    print(f"patch_bbox         : y=[{PATCH_Y0},{PATCH_Y1}), x=[{PATCH_X0},{PATCH_X1})  ({PATCH_SIZE}x{PATCH_SIZE})")
    print(f"component_bbox     : {COMPONENT_BBOX_NOTE}  ← metadata note 전용, extraction 사용 금지")
    print(f"position_bin       : {POSITION_BIN}")
    print(f"expected_score     : {EXPECTED_SCORE}")
    print(f"roi_coverage       : {ROI_COVERAGE}")
    print(f"split              : {SPLIT}")
    print()

    errors = []

    def chk_path(label, path):
        exists = os.path.isfile(path)
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {label}: {path}")
        if not exists:
            errors.append(f"{label} missing: {path}")

    print("--- 입력 경로 확인 (load 없음) ---")
    chk_path("CT_HU_NPY",        CT_HU_NPY_PATH)
    chk_path("STATS_NPZ",        STATS_NPZ_PATH)
    chk_path("SELECTED_INDICES", SELECTED_INDICES_NPY_PATH)
    chk_path("FEATURE_EXTRACTOR",FEATURE_EXTRACTOR_MODULE_PATH)
    chk_path("PREPROCESSING",    PREPROCESSING_MODULE_PATH)
    chk_path("RESNET18_CACHE",   RESNET18_CACHE_PATH)
    print()

    print("--- guard 상태 ---")
    print(f"  ALLOW_CT_LOAD              = {ALLOW_CT_LOAD}")
    print(f"  ALLOW_MODEL_FORWARD        = {ALLOW_MODEL_FORWARD}")
    print(f"  ALLOW_FEATURE_EXTRACTION   = {ALLOW_FEATURE_EXTRACTION}")
    print(f"  ALLOW_WRITE_FEATURE_VECTOR = {ALLOW_WRITE_FEATURE_VECTOR}")
    print(f"  ALLOW_GPU                  = {ALLOW_GPU}")
    print(f"  ALLOW_STAGE2_HOLDOUT       = {ALLOW_STAGE2_HOLDOUT}")
    print(f"  ALLOW_FULL_300             = {ALLOW_FULL_300}")
    print()

    print("--- local_z / report_slice 매핑 ---")
    print(f"  CT npy indexing  : ct_hu[{CT_INDEX_Z}]  ← LOCAL_Z_USED_FOR_CT_INDEXING=True")
    print(f"  report_slice=106 : title/metadata 전용  ← SLICE_INDEX_USED_FOR_CT_INDEXING=False")
    print(f"  offset           : {REPORT_SLICE_INDEX - CT_INDEX_Z}")
    print()

    print("--- stats / selected_indices policy ---")
    print(f"  STATS_NPZ      : {STATS_NPZ_PATH}")
    print(f"  position_bin   : {POSITION_BIN}  → {POSITION_BIN}_mean / {POSITION_BIN}_cov")
    print(f"  SELECTED_INDICES: {SELECTED_INDICES_NPY_PATH}")
    print()

    print("--- stage2_holdout 확인 ---")
    print(f"  ALLOW_STAGE2_HOLDOUT = {ALLOW_STAGE2_HOLDOUT} (False = 안전)")
    print(f"  CT 경로에 'stage2_holdout' 포함: "
          f"{'YES (위험)' if _STAGE2_HOLDOUT_SENTINEL in CT_HU_NPY_PATH else 'NO (안전)'}")
    print()

    print("--- output root 확인 ---")
    print(f"  OUTPUT_ROOT: {OUTPUT_ROOT}")
    print(f"  LUNG1-320 root (충돌 금지): {_LUNG1_320_OUTPUT_ROOT}")
    print(f"  충돌 여부: {'YES (위험)' if OUTPUT_ROOT == _LUNG1_320_OUTPUT_ROOT else 'NO (안전)'}")
    done_path = os.path.join(OUTPUT_ROOT, "DONE.json")
    print(f"  DONE.json 기존 존재: {'YES (BLOCKED)' if os.path.exists(done_path) else 'NO (안전)'}")
    if os.path.exists(done_path):
        errors.append("DONE.json already exists — smoke run BLOCKED")
    print()

    print("--- 실행 시 생성 예정 파일 (이번 단계에서는 생성 안 함) ---")
    smoke_files = [
        "feature_selected100.npy",
        "feature_raw448.npy",
        "metadata.json",
        "errors.csv",
        "DONE.json",
    ]
    for f in smoke_files:
        print(f"  - {f}")
    print()

    print("--- CT / model / feature / write = 0 확인 ---")
    print("  CT load    : 0 (dry-run)")
    print("  model fwd  : 0 (dry-run)")
    print("  feature ext: 0 (dry-run)")
    print("  write      : 0 (dry-run)")
    print()

    if errors:
        print(f"WARNING: {len(errors)}개 문제 감지")
        for e in errors:
            print(f"  {e}")
        print("판정: WARNING (smoke 전 해결 필요)")
        return 1

    print("판정: DRY-RUN PASS (경로 이상 없음, 실행 없음)")
    return 0


# ============================================================
# I. plan-only
# ============================================================

def run_plan_only() -> int:
    """실행 계획 출력. CT load / model forward / feature extraction 없음."""
    print("\n=== PLAN-ONLY (candidate A: LUNG1-052__c3) ===")
    print()
    print("+-----------------------------------------------------------------+")
    print("| S5 Feature Re-extraction — Candidate A Smoke Plan              |")
    print("| CASE: LUNG1-052__c3 / lower_central / stage1_dev               |")
    print("+-----------------------------------------------------------------+")
    print()
    print("! CT INDEX 주의:")
    print(f"   CT npy indexing 시 반드시 local_z = {CT_INDEX_Z} 사용")
    print(f"   report_slice = {REPORT_SLICE_INDEX} 는 global/reporting 전용 — CT index 절대 사용 금지")
    print(f"   local_z({CT_INDEX_Z}) != report_slice({REPORT_SLICE_INDEX}), offset={REPORT_SLICE_INDEX - CT_INDEX_Z}")
    print()
    print("! POSITION_BIN 주의:")
    print(f"   POSITION_BIN = '{POSITION_BIN}'  (lower_central)")
    print(f"   stats key: {POSITION_BIN}_mean / {POSITION_BIN}_cov")
    print()

    steps = [
        ("Step 1", "guard 확인",
         "ALLOW_CT_LOAD=True, ALLOW_MODEL_FORWARD=True 등 별도 승인 필요"),
        ("Step 2", "stats load",
         f"np.load('{STATS_NPZ_PATH}', allow_pickle=True)\n"
         f"           → {POSITION_BIN}_mean (100,) + {POSITION_BIN}_cov (100,100)"),
        ("Step 3", "selected_indices load",
         f"np.load('{SELECTED_INDICES_NPY_PATH}')  shape=(100,)"),
        ("Step 4", "CT load (ALLOW_CT_LOAD=True 시 — mmap_mode='r')",
         f"np.load('{CT_HU_NPY_PATH}', mmap_mode='r')\n"
         f"           → shape (213,512,512) int16"),
        ("Step 5", "slice 추출 (local_z 사용)",
         f"slice_2d = np.asarray(ct_hu[{CT_INDEX_Z}], dtype=np.float32)\n"
         f"           ← local_z={CT_INDEX_Z}  (report_slice={REPORT_SLICE_INDEX} 사용 절대 금지)"),
        ("Step 6", "preprocessing",
         f"preprocess_ct_slice(slice_2d, hu_min={HU_MIN}, hu_max={HU_MAX})\n"
         f"           → (3,512,512) float32, {THREE_CHANNEL_POLICY}, ImageNet norm"),
        ("Step 7", "FeatureExtractor init",
         f"FeatureExtractor(device='cpu')  {BACKBONE} {TAPPED_LAYERS}"),
        ("Step 8", "extract_patch_features (single patch)",
         f"fe.extract_patch_features(preprocessed, [({PATCH_Y0},{PATCH_X0},{PATCH_Y1},{PATCH_X1})])\n"
         f"           → features_448 shape (1, {RAW_FEATURE_DIM}) float32\n"
         f"           component_bbox={COMPONENT_BBOX_NOTE} 사용 금지"),
        ("Step 9", "dimension reduction",
         f"feature_100 = features_448[0][selected_indices]  shape ({SELECTED_FEATURE_DIM},)"),
        ("Step 10", "feature 저장 (ALLOW_WRITE_FEATURE_VECTOR=True 시)",
         f"np.save(OUTPUT_ROOT/feature_selected100.npy, feature_100)\n"
         f"np.save(OUTPUT_ROOT/feature_raw448.npy, features_448[0])"),
        ("Step 11", "score 재현 검증",
         f"diff = feature_100 - {POSITION_BIN}_mean\n"
         f"           cov_inv = np.linalg.inv(cov + 1e-5*I)\n"
         f"           d_sq = diff @ cov_inv @ diff\n"
         f"           score_recon = sqrt(max(0, d_sq))  expected ≈ {EXPECTED_SCORE}"),
        ("Step 12", "metadata + DONE 저장",
         "metadata.json (ct_index_z / report_slice / local_z_used / not_gradcam / not_diagnostic)\n"
         "           + DONE.json"),
    ]

    for step_id, step_name, detail in steps:
        print(f"  {step_id}: {step_name}")
        for line in detail.split("\n"):
            print(f"           {line}")
        print()

    print("--- 다음 smoke actual run 승인 요건 ---")
    print("  1. ALLOW_CT_LOAD            = True  (별도 승인)")
    print("  2. ALLOW_MODEL_FORWARD      = True  (별도 승인)")
    print("  3. ALLOW_FEATURE_EXTRACTION = True  (별도 승인)")
    print("  4. ALLOW_WRITE_FEATURE_VECTOR = True (별도 승인)")
    print("  5. ALLOW_GPU                = False (CPU only 유지)")
    print("  6. ALLOW_STAGE2_HOLDOUT     = False (유지)")
    print("  7. --run-smoke --confirm-extract 플래그 필요")
    print()
    print(f"  현재 단계: 정적 검사 (PASS 후) → smoke actual run 승인 대기")
    return 0


# ============================================================
# J. run_smoke (현재 guard False → BLOCKED)
# ============================================================

def run_smoke(confirm_extract: bool) -> int:
    """실제 feature re-extraction smoke run. 모든 guard True 시에만 실행."""
    if not confirm_extract:
        print("BLOCKED: --run-smoke 단독 실행 금지. --confirm-extract 필요.", file=sys.stderr)
        return 2

    if not ALLOW_CT_LOAD:
        print("BLOCKED: ALLOW_CT_LOAD=False. CT load 금지.", file=sys.stderr)
        return 2
    if not ALLOW_MODEL_FORWARD:
        print("BLOCKED: ALLOW_MODEL_FORWARD=False. model forward 금지.", file=sys.stderr)
        return 2
    if not ALLOW_FEATURE_EXTRACTION:
        print("BLOCKED: ALLOW_FEATURE_EXTRACTION=False. feature extraction 금지.", file=sys.stderr)
        return 2
    if not ALLOW_WRITE_FEATURE_VECTOR:
        print("BLOCKED: ALLOW_WRITE_FEATURE_VECTOR=False. feature write 금지.", file=sys.stderr)
        return 2
    if ALLOW_GPU:
        print("BLOCKED: ALLOW_GPU=True 시 CPU-only 정책 위반.", file=sys.stderr)
        return 2
    if ALLOW_STAGE2_HOLDOUT:
        print("BLOCKED: ALLOW_STAGE2_HOLDOUT=True는 허용되지 않습니다.", file=sys.stderr)
        return 2
    if ALLOW_FULL_300:
        print("BLOCKED: ALLOW_FULL_300=True는 허용되지 않습니다.", file=sys.stderr)
        return 2

    # DONE.json 존재 시 재실행 차단
    done_path = os.path.join(OUTPUT_ROOT, "DONE.json")
    if os.path.exists(done_path):
        print(f"BLOCKED: DONE.json 이미 존재 — 재실행 금지. {done_path}", file=sys.stderr)
        return 2

    # ----------------------------------------------------------------
    # guard 통과 후 실제 실행 로직 (현재 모든 guard=False이므로 도달 불가)
    # ----------------------------------------------------------------
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
    import json
    import numpy as np
    from position_aware_padim.preprocessing import preprocess_ct_slice
    from position_aware_padim.feature_extractor import FeatureExtractor

    # local_z guard (run path)
    assert CT_INDEX_Z == 51,          f"CT_INDEX_Z guard FAIL: {CT_INDEX_Z}"
    assert REPORT_SLICE_INDEX == 106, f"REPORT_SLICE_INDEX guard FAIL: {REPORT_SLICE_INDEX}"
    assert CT_INDEX_Z != REPORT_SLICE_INDEX, \
        f"CT_INDEX_Z({CT_INDEX_Z}) == REPORT_SLICE_INDEX({REPORT_SLICE_INDEX}) — 혼동 위험"
    assert LOCAL_Z_USED_FOR_CT_INDEXING is True
    assert SLICE_INDEX_USED_FOR_CT_INDEXING is False

    # CT load (mmap_mode="r" 필수)
    print(f"CT load (mmap_mode='r'): {CT_HU_NPY_PATH}")
    ct_hu = np.load(CT_HU_NPY_PATH, mmap_mode="r")
    assert ct_hu.ndim == 3, f"CT shape ndim mismatch: {ct_hu.ndim}"
    assert ct_hu.shape == (213, 512, 512), \
        f"CT shape mismatch: {ct_hu.shape} != (213,512,512)"
    assert 0 <= CT_INDEX_Z < ct_hu.shape[0], \
        f"local_z={CT_INDEX_Z} out of range [0, {ct_hu.shape[0]})"

    # slice_2d: local_z=51 사용 (report_slice=106 사용 금지)
    slice_2d = np.asarray(ct_hu[CT_INDEX_Z], dtype=np.float32)
    del ct_hu

    preprocessed = preprocess_ct_slice(slice_2d, hu_min=HU_MIN, hu_max=HU_MAX)

    stats = np.load(STATS_NPZ_PATH, allow_pickle=True)
    mean_vec = stats[f"{POSITION_BIN}_mean"].astype(np.float32)    # (100,)
    cov_mat  = stats[f"{POSITION_BIN}_cov"].astype(np.float32)     # (100,100)
    assert mean_vec.shape == (100,), f"mean shape mismatch: {mean_vec.shape}"
    assert cov_mat.shape == (100, 100), f"cov shape mismatch: {cov_mat.shape}"

    selected_indices = np.load(SELECTED_INDICES_NPY_PATH)          # (100,)
    assert selected_indices.shape == (100,), \
        f"selected_indices shape mismatch: {selected_indices.shape}"
    assert selected_indices.max() < RAW_FEATURE_DIM, \
        f"selected_indices max={selected_indices.max()} >= raw_dim={RAW_FEATURE_DIM}"

    device = "cpu"  # ALLOW_GPU=False 유지
    fe = FeatureExtractor(device=device)
    # single max-score patch만 사용 (component bbox 사용 금지)
    features_448 = fe.extract_patch_features(
        preprocessed, [(PATCH_Y0, PATCH_X0, PATCH_Y1, PATCH_X1)]
    )
    assert features_448.shape[1] == RAW_FEATURE_DIM, \
        f"raw feature dim mismatch: {features_448.shape[1]} != {RAW_FEATURE_DIM}"

    feature_100 = features_448[0][selected_indices].astype(np.float32)
    assert feature_100.shape == (100,), f"feature_100 shape mismatch: {feature_100.shape}"

    diff    = feature_100 - mean_vec
    cov_inv = np.linalg.inv(cov_mat + 1e-5 * np.eye(len(cov_mat)))
    d_sq    = float(diff @ cov_inv @ diff)
    score_recon = float(np.sqrt(max(0.0, d_sq)))

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    np.save(os.path.join(OUTPUT_ROOT, "feature_selected100.npy"), feature_100)
    np.save(os.path.join(OUTPUT_ROOT, "feature_raw448.npy"),
            features_448[0].astype(np.float32))

    metadata = {
        "case_id": CASE_ID,
        "volume_id": VOLUME_ID,
        "ct_index_z": CT_INDEX_Z,
        "report_slice_index": REPORT_SLICE_INDEX,
        "local_z_used_for_ct_indexing": LOCAL_Z_USED_FOR_CT_INDEXING,
        "slice_index_used_for_ct_indexing": SLICE_INDEX_USED_FOR_CT_INDEXING,
        "patch_bbox_yxyx": [PATCH_Y0, PATCH_X0, PATCH_Y1, PATCH_X1],
        "patch_size": PATCH_SIZE,
        "position_bin": POSITION_BIN,
        "roi_coverage": ROI_COVERAGE,
        "expected_score": EXPECTED_SCORE,
        "backbone": BACKBONE,
        "tapped_layers": TAPPED_LAYERS,
        "raw_feature_dim": RAW_FEATURE_DIM,
        "selected_feature_dim": SELECTED_FEATURE_DIM,
        "selected_indices_source": SELECTED_INDICES_NPY_PATH,
        "preprocessing_hu_min": HU_MIN,
        "preprocessing_hu_max": HU_MAX,
        "preprocessing_image_net_norm": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "three_channel_policy": THREE_CHANNEL_POLICY,
        "feature_extractor_source": FEATURE_EXTRACTOR_MODULE_PATH,
        "ct_load_occurred": True,
        "model_forward_occurred": True,
        "feature_extraction_occurred": True,
        "gpu_used": False,
        "stage2_holdout_accessed": False,
        "score_recomputed": True,
        "threshold_recomputed": False,
        "existing_artifacts_modified": False,
        "not_gradcam_caveat": "이 output은 Grad-CAM이 아님. feature attribution 목적.",
        "not_diagnostic_caveat": "특정 소견/암/병변 확정 진단 불가. 이상 후보 localization 목적.",
        "component_bbox_note": COMPONENT_BBOX_NOTE,
        "score_reconstructed": score_recon,
        "score_diff_from_expected": abs(score_recon - EXPECTED_SCORE),
        "parity_pass": abs(score_recon - EXPECTED_SCORE) < 0.5,
    }
    with open(os.path.join(OUTPUT_ROOT, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUTPUT_ROOT, "DONE.json"), "w") as f:
        json.dump({
            "status": "DONE",
            "case_id": CASE_ID,
            "ct_index_z": CT_INDEX_Z,
            "report_slice_index": REPORT_SLICE_INDEX,
            "score_reconstructed": score_recon,
            "parity_pass": metadata["parity_pass"],
        }, f, indent=2)

    print(f"score_reconstructed = {score_recon:.5f}  (expected ≈ {EXPECTED_SCORE})")
    print(f"diff_from_expected  = {abs(score_recon - EXPECTED_SCORE):.5f}")
    if metadata["parity_pass"]:
        print("판정: PARITY PASS")
    else:
        print("판정: PARITY FAIL — 재검토 필요")
    return 0


# ============================================================
# K. main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=f"S5 Feature Re-extraction — Candidate A ({CASE_ID})"
    )
    parser.add_argument("--selftest",        action="store_true", help="정적 selftest 실행")
    parser.add_argument("--dry-run",         action="store_true", help="경로 확인 (실행 없음)")
    parser.add_argument("--plan-only",       action="store_true", help="실행 계획 출력")
    parser.add_argument("--run-smoke",       action="store_true", help="실제 smoke run (guard 필요)")
    parser.add_argument("--confirm-extract", action="store_true", help="feature extraction 확인 플래그")
    args = parser.parse_args()

    # bare 실행 차단
    if not any([args.selftest, args.dry_run, args.plan_only, args.run_smoke]):
        print("BLOCKED: 실행 모드를 지정하세요.", file=sys.stderr)
        print("  허용: --selftest | --dry-run | --plan-only", file=sys.stderr)
        print("  금지 (guard 필요): --run-smoke --confirm-extract", file=sys.stderr)
        sys.exit(2)

    # --run-smoke 단독 차단
    if args.run_smoke and not args.confirm_extract:
        print("BLOCKED: --run-smoke 단독 실행 금지. --confirm-extract 필요.", file=sys.stderr)
        sys.exit(2)

    # 모든 경로에서 local_z guard 확인
    _assert_local_z_guards()

    if args.selftest:
        rc = run_selftest()
        sys.exit(rc)

    if args.dry_run:
        rc = run_dry_run()
        sys.exit(rc)

    if args.plan_only:
        rc = run_plan_only()
        sys.exit(rc)

    if args.run_smoke:
        rc = run_smoke(confirm_extract=args.confirm_extract)
        sys.exit(rc)


if __name__ == "__main__":
    main()
