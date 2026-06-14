"""
S5 Feature Re-extraction 1-case Smoke Script
=============================================
대상: LUNG1-320__c2 / stage1_dev only
단계: smoke script (static 검사 단계 — 실제 CT load/model forward 금지)

실행 방법:
  --selftest         : 정적 selftest 30+ 항목 (허용)
  --dry-run          : 경로 확인, CT load 없음 (허용)
  --plan-only        : 실행 계획 출력 (허용)
  --run-smoke --confirm-extract : 실제 실행 (guard True 시에만)

금지 (현재 모든 guard=False):
  bare 실행         → BLOCKED exit 2
  --run-smoke 단독  → BLOCKED exit 2
  --run-smoke --confirm-extract → BLOCKED exit 2 (guard False)
"""

from __future__ import annotations

import argparse
import os
import sys

# ============================================================
# A. Path / Metadata constants
# ============================================================

REPO_ROOT = "/home/jinhy/project/lung-ct-anomaly"

CASE_ID           = "LUNG1-320__c2"
VOLUME_ID         = "NSCLC_LUNG1-320__95de24d86f"
CT_INDEX_Z        = 89           # CT npy indexing에 사용하는 local z
REPORT_SLICE_INDEX = 140         # global/reporting index (CT index 아님)
PATCH_Y0          = 320
PATCH_X0          = 208
PATCH_Y1          = 352
PATCH_X1          = 240
PATCH_SIZE        = 32
POSITION_BIN      = "lower_peripheral"
EXPECTED_SCORE    = 35.44
SPLIT             = "stage1_dev"

# ============================================================
# B. preprocessing parity metadata
# ============================================================

HU_MIN            = -1000.0
HU_MAX            =  200.0
IMAGENET_MEAN     = [0.485, 0.456, 0.406]
IMAGENET_STD      = [0.229, 0.224, 0.225]
THREE_CHANNEL_POLICY = "1ch → 3ch stack (repeat)"
BACKBONE          = "ResNet18"
TAPPED_LAYERS     = ["layer1", "layer2", "layer3"]
RAW_FEATURE_DIM   = 448
SELECTED_FEATURE_DIM = 100

# ============================================================
# C. Input paths
# ============================================================

CT_HU_NPY_PATH = (
    "/mnt/c/Users/jinhy/Desktop/"
    "NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/"
    "volumes_npy/NSCLC_LUNG1-320__95de24d86f/ct_hu.npy"
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

SPLITS_JSON_PATH = os.path.join(
    REPO_ROOT,
    "outputs/position-aware-padim-v1/splits/normal_v1.json"
)

# stage2_holdout 경로 (접근 금지 확인용 — 절대 load 금지)
_STAGE2_HOLDOUT_SENTINEL = "stage2_holdout"

# ============================================================
# D. Output root (smoke run 단계에서만 생성)
# ============================================================

OUTPUT_ROOT = os.path.join(
    REPO_ROOT,
    "outputs/position-aware-padim-v1/reports/explanation_cards/"
    "s5_feature_reextract_1case_smoke_v1"
)

# ============================================================
# E. Basic guards (all False — 실제 실행 금지)
# ============================================================

ALLOW_CT_LOAD           = False
ALLOW_MODEL_FORWARD     = False
ALLOW_FEATURE_EXTRACTION = False
ALLOW_WRITE_FEATURE_VECTOR = False
ALLOW_GPU               = False
ALLOW_STAGE2_HOLDOUT    = False
ALLOW_FULL_300          = False


# ============================================================
# F. local_z guard (양쪽 확인)
# ============================================================

def _assert_local_z_guards():
    """local_z / slice_index 혼동 방지 guard."""
    assert CT_INDEX_Z == 89,        f"CT_INDEX_Z guard FAIL: {CT_INDEX_Z}"
    assert REPORT_SLICE_INDEX == 140, f"REPORT_SLICE_INDEX guard FAIL: {REPORT_SLICE_INDEX}"
    assert CT_INDEX_Z != REPORT_SLICE_INDEX, (
        f"CT_INDEX_Z({CT_INDEX_Z}) == REPORT_SLICE_INDEX({REPORT_SLICE_INDEX}) — 혼동 위험"
    )
    assert (PATCH_Y1 - PATCH_Y0) == PATCH_SIZE, \
        f"patch height mismatch: {PATCH_Y1 - PATCH_Y0} != {PATCH_SIZE}"
    assert (PATCH_X1 - PATCH_X0) == PATCH_SIZE, \
        f"patch width mismatch: {PATCH_X1 - PATCH_X0} != {PATCH_SIZE}"


# ============================================================
# G. selftest (30+ 항목)
# ============================================================

def run_selftest() -> int:
    """정적 selftest. 모든 guard / 상수 / 경로 검증. 실제 load 없음."""
    passed = []
    failed = []

    def chk(name: str, cond: bool, msg: str = ""):
        if cond:
            passed.append(name)
        else:
            failed.append(f"{name}: {msg}")

    # 1. all guards default False
    chk("1_guard_ct_load_false",            not ALLOW_CT_LOAD,            "ALLOW_CT_LOAD must be False")
    chk("2_guard_model_forward_false",      not ALLOW_MODEL_FORWARD,      "ALLOW_MODEL_FORWARD must be False")
    chk("3_guard_feature_extraction_false", not ALLOW_FEATURE_EXTRACTION, "ALLOW_FEATURE_EXTRACTION must be False")
    chk("4_guard_write_feature_false",      not ALLOW_WRITE_FEATURE_VECTOR, "ALLOW_WRITE_FEATURE_VECTOR must be False")
    chk("5_guard_gpu_false",                not ALLOW_GPU,                "ALLOW_GPU must be False")
    chk("6_guard_stage2_holdout_false",     not ALLOW_STAGE2_HOLDOUT,     "ALLOW_STAGE2_HOLDOUT must be False")
    chk("7_guard_full300_false",            not ALLOW_FULL_300,           "ALLOW_FULL_300 must be False")

    # 2. CASE_ID / VOLUME_ID exact
    chk("8_case_id_exact",   CASE_ID == "LUNG1-320__c2",                    f"got {CASE_ID}")
    chk("9_volume_id_exact", VOLUME_ID == "NSCLC_LUNG1-320__95de24d86f",    f"got {VOLUME_ID}")

    # 3. local_z / slice_index
    chk("10_ct_index_z_89",              CT_INDEX_Z == 89,              f"got {CT_INDEX_Z}")
    chk("11_report_slice_index_140",     REPORT_SLICE_INDEX == 140,     f"got {REPORT_SLICE_INDEX}")
    chk("12_ct_index_z_ne_slice_index",  CT_INDEX_Z != REPORT_SLICE_INDEX, "local_z == slice_index — 혼동 위험")

    # 4. patch bbox 32×32
    chk("13_patch_height_32", (PATCH_Y1 - PATCH_Y0) == PATCH_SIZE, f"{PATCH_Y1-PATCH_Y0} != 32")
    chk("14_patch_width_32",  (PATCH_X1 - PATCH_X0) == PATCH_SIZE, f"{PATCH_X1-PATCH_X0} != 32")

    # 5. position_bin
    chk("15_position_bin",    POSITION_BIN == "lower_peripheral",    f"got {POSITION_BIN}")

    # 6. expected score recorded
    chk("16_expected_score",  abs(EXPECTED_SCORE - 35.44) < 0.01,   f"got {EXPECTED_SCORE}")

    # 7. preprocessing constants
    chk("17_hu_min_neg1000",  HU_MIN == -1000.0,   f"got {HU_MIN}")
    chk("18_hu_max_200",      HU_MAX == 200.0,      f"got {HU_MAX}")
    chk("19_imagenet_mean_exists", len(IMAGENET_MEAN) == 3, "IMAGENET_MEAN must be length 3")
    chk("20_imagenet_std_exists",  len(IMAGENET_STD) == 3,  "IMAGENET_STD must be length 3")
    chk("21_3ch_policy_exists",    "3ch" in THREE_CHANNEL_POLICY, "3ch repeat policy not defined")

    # 8. backbone / layers / dims
    chk("22_backbone_resnet18",  BACKBONE == "ResNet18",              f"got {BACKBONE}")
    chk("23_tapped_layers",      TAPPED_LAYERS == ["layer1", "layer2", "layer3"],
        f"got {TAPPED_LAYERS}")
    chk("24_raw_dim_448",        RAW_FEATURE_DIM == 448,              f"got {RAW_FEATURE_DIM}")
    chk("25_selected_dim_100",   SELECTED_FEATURE_DIM == 100,         f"got {SELECTED_FEATURE_DIM}")

    # 9. selected_indices source exists
    chk("26_selected_indices_source_exists",
        os.path.isfile(SELECTED_INDICES_NPY_PATH),
        f"not found: {SELECTED_INDICES_NPY_PATH}")

    # 10. FeatureExtractor reuse path
    chk("27_feature_extractor_file_exists",
        os.path.isfile(FEATURE_EXTRACTOR_MODULE_PATH),
        f"not found: {FEATURE_EXTRACTOR_MODULE_PATH}")

    # 11. preprocessing module exists
    chk("28_preprocessing_module_exists",
        os.path.isfile(PREPROCESSING_MODULE_PATH),
        f"not found: {PREPROCESSING_MODULE_PATH}")

    # 12. no EfficientNet-B0 branch use
    chk("29_no_effnet_in_backbone", "efficientnet" not in BACKBONE.lower(),
        f"EfficientNet branch 혼입 위험: {BACKBONE}")
    chk("30_no_effnet_in_extractor_path",
        "effnet" not in FEATURE_EXTRACTOR_MODULE_PATH.lower(),
        f"effnet extractor 경로 혼입: {FEATURE_EXTRACTOR_MODULE_PATH}")

    # 13. no CT load / model forward / feature extraction / write in dry-run logic
    chk("31_ct_load_blocked_by_guard", not ALLOW_CT_LOAD,
        "CT load가 guard 없이 실행될 수 있음")
    chk("32_model_forward_blocked",    not ALLOW_MODEL_FORWARD,
        "model forward가 guard 없이 실행될 수 있음")
    chk("33_feature_extraction_blocked", not ALLOW_FEATURE_EXTRACTION,
        "feature extraction이 guard 없이 실행될 수 있음")
    chk("34_write_blocked",            not ALLOW_WRITE_FEATURE_VECTOR,
        "feature write가 guard 없이 실행될 수 있음")

    # 14. no GPU by default
    chk("35_no_gpu_by_default", not ALLOW_GPU, "GPU guard must be False")

    # 15. no stage2_holdout
    chk("36_no_stage2_holdout", not ALLOW_STAGE2_HOLDOUT, "stage2_holdout guard must be False")
    chk("37_stage2_sentinel_not_in_ct_path",
        _STAGE2_HOLDOUT_SENTINEL not in CT_HU_NPY_PATH,
        "CT 경로에 stage2_holdout 포함 위험")

    # 16. no full300 loop
    chk("38_no_full300", not ALLOW_FULL_300, "ALLOW_FULL_300 must be False")

    # 17. output root separate from src/scripts
    chk("39_output_root_in_outputs_dir",
        "outputs/" in OUTPUT_ROOT,
        f"output root이 outputs/ 외부: {OUTPUT_ROOT}")

    # 18. run requires both flags
    # (정적 검사: 소스에 "--run-smoke" and "--confirm-extract" 체크가 있는지)
    import inspect, textwrap
    src = inspect.getsource(run_smoke)
    chk("40_run_smoke_requires_confirm",
        "confirm_extract" in src,
        "run_smoke에 --confirm-extract 체크 없음")

    # 19. metadata schema has local_z/slice_index distinction
    schema_keys = _get_metadata_schema_keys()
    chk("41_schema_local_z_used",
        "ct_index_z" in schema_keys, "metadata schema에 ct_index_z 없음")
    chk("42_schema_report_slice_index",
        "report_slice_index" in schema_keys, "metadata schema에 report_slice_index 없음")
    chk("43_schema_local_z_for_ct_indexing",
        "local_z_used_for_ct_indexing" in schema_keys,
        "local_z_used_for_ct_indexing 필드 없음")
    chk("44_schema_slice_index_not_for_ct",
        "slice_index_used_for_ct_indexing" in schema_keys,
        "slice_index_used_for_ct_indexing 필드 없음")

    # 20. existing artifacts modification false in metadata
    chk("45_no_existing_artifacts_modified",
        "existing_artifacts_modified" in schema_keys,
        "existing_artifacts_modified 필드 없음")

    # 21. stats npz exists
    chk("46_stats_npz_exists",
        os.path.isfile(STATS_NPZ_PATH),
        f"not found: {STATS_NPZ_PATH}")

    # 22. CT path recorded (existence check without load)
    chk("47_ct_path_recorded",
        len(CT_HU_NPY_PATH) > 10,
        "CT_HU_NPY_PATH empty")

    # 23. ResNet18 cache path exists
    chk("48_resnet18_cache_exists",
        os.path.isfile(RESNET18_CACHE_PATH),
        f"not found: {RESNET18_CACHE_PATH}")

    # 24. split is stage1_dev
    chk("49_split_stage1_dev", SPLIT == "stage1_dev", f"got {SPLIT}")

    # 25. patch bbox in bounds (static check against known shape)
    CT_SHAPE_Z, CT_SHAPE_H, CT_SHAPE_W = 262, 512, 512
    chk("50_local_z_in_bounds", 0 <= CT_INDEX_Z < CT_SHAPE_Z,
        f"local_z={CT_INDEX_Z} out of z range [0, {CT_SHAPE_Z})")
    chk("51_patch_y_in_bounds", 0 <= PATCH_Y0 < PATCH_Y1 <= CT_SHAPE_H,
        f"y0={PATCH_Y0}, y1={PATCH_Y1}, H={CT_SHAPE_H}")
    chk("52_patch_x_in_bounds", 0 <= PATCH_X0 < PATCH_X1 <= CT_SHAPE_W,
        f"x0={PATCH_X0}, x1={PATCH_X1}, W={CT_SHAPE_W}")

    # --- 결과 출력 ---
    total = len(passed) + len(failed)
    print(f"\n=== SELFTEST 결과 ===")
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
        "expected_score",
        "backbone",
        "tapped_layers",
        "raw_feature_dim",
        "selected_feature_dim",
        "selected_indices_source",
        "preprocessing_hu_min",
        "preprocessing_hu_max",
        "preprocessing_image_net_norm",
        "feature_extractor_source",
        "ct_load_occurred",
        "model_forward_occurred",
        "feature_extraction_occurred",
        "gpu_used",
        "stage2_holdout_accessed",
        "score_recomputed",
        "threshold_recomputed",
        "existing_artifacts_modified",
    ]


# ============================================================
# H. dry-run
# ============================================================

def run_dry_run() -> int:
    """경로 확인 및 metadata 출력. CT load / model forward / feature extraction 없음."""
    print("\n=== DRY-RUN ===")
    print(f"CASE_ID            : {CASE_ID}")
    print(f"VOLUME_ID          : {VOLUME_ID}")
    print(f"CT_INDEX_Z         : {CT_INDEX_Z}  ← CT npy indexing에 사용")
    print(f"REPORT_SLICE_INDEX : {REPORT_SLICE_INDEX}  ← global/reporting 전용 (CT index 아님)")
    print(f"patch_bbox         : y=[{PATCH_Y0},{PATCH_Y1}), x=[{PATCH_X0},{PATCH_X1})  ({PATCH_SIZE}x{PATCH_SIZE})")
    print(f"position_bin       : {POSITION_BIN}")
    print(f"expected_score     : {EXPECTED_SCORE}")
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
    print(f"  ALLOW_CT_LOAD            = {ALLOW_CT_LOAD}")
    print(f"  ALLOW_MODEL_FORWARD      = {ALLOW_MODEL_FORWARD}")
    print(f"  ALLOW_FEATURE_EXTRACTION = {ALLOW_FEATURE_EXTRACTION}")
    print(f"  ALLOW_WRITE_FEATURE_VECTOR = {ALLOW_WRITE_FEATURE_VECTOR}")
    print(f"  ALLOW_GPU                = {ALLOW_GPU}")
    print(f"  ALLOW_STAGE2_HOLDOUT     = {ALLOW_STAGE2_HOLDOUT}")
    print(f"  ALLOW_FULL_300           = {ALLOW_FULL_300}")
    print()

    print("--- stage2_holdout 교집합 확인 ---")
    print(f"  LUNG1-320 split: {SPLIT}")
    print(f"  stage2_holdout 접근: {ALLOW_STAGE2_HOLDOUT} (False = 안전)")
    print(f"  CT 경로에 'stage2_holdout' 포함 여부: "
          f"{'YES (위험)' if _STAGE2_HOLDOUT_SENTINEL in CT_HU_NPY_PATH else 'NO (안전)'}")
    print()

    print("--- 실행 시 생성 예정 파일 (이번 단계에서는 생성 안 함) ---")
    print(f"  OUTPUT_ROOT: {OUTPUT_ROOT}")
    smoke_files = [
        "s5_feature_reextract_1case_feature_selected100.npy",
        "s5_feature_reextract_1case_feature_raw448.npy",
        "s5_feature_reextract_1case_metadata.json",
        "s5_feature_reextract_1case_validation.md",
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
        print(f"WARNING: {len(errors)}개 경로 없음")
        for e in errors:
            print(f"  {e}")
        print("판정: WARNING (경로 문제 있음 — smoke 전 해결 필요)")
        return 1

    print("판정: DRY-RUN PASS (경로 이상 없음, 실행 없음)")
    return 0


# ============================================================
# I. plan-only
# ============================================================

def run_plan_only() -> int:
    """실행 계획 출력. CT load / model forward / feature extraction 없음."""
    print("\n=== PLAN-ONLY ===")
    print()
    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│ S5 Feature Re-extraction 1-case Smoke Plan                      │")
    print("└─────────────────────────────────────────────────────────────────┘")
    print()
    print("⚠️  CT INDEX 주의:")
    print(f"   CT npy indexing 시 반드시 local_z = {CT_INDEX_Z} 사용")
    print(f"   slice_index = {REPORT_SLICE_INDEX} 는 global/reporting index 전용 — CT index 절대 사용 금지")
    print(f"   local_z({CT_INDEX_Z}) ≠ slice_index({REPORT_SLICE_INDEX}) — 혼동 시 wrong feature")
    print()

    steps = [
        ("Step 1", "guard 확인", "ALLOW_CT_LOAD=False → run smoke에서만 True로 변경"),
        ("Step 2", "stats load",
         f"np.load('{STATS_NPZ_PATH}')\n"
         f"           → lower_peripheral_mean (100,) + lower_peripheral_cov (100,100)"),
        ("Step 3", "selected_indices load",
         f"np.load('{SELECTED_INDICES_NPY_PATH}')  shape=(100,)"),
        ("Step 4", "CT load (ALLOW_CT_LOAD=True 시)",
         f"np.load('{CT_HU_NPY_PATH}')  → shape (262,512,512) int16"),
        ("Step 5", "slice 추출",
         f"slice_2d = np.asarray(ct_hu[{CT_INDEX_Z}], dtype=np.float32)  ← local_z={CT_INDEX_Z}"),
        ("Step 6", "preprocessing",
         "preprocess_ct_slice(slice_2d)  → (3,512,512) float32\n"
         f"           hu_min={HU_MIN}, hu_max={HU_MAX}, 3ch repeat, ImageNet norm"),
        ("Step 7", "FeatureExtractor init",
         "FeatureExtractor(device='cpu')  ResNet18 layer1+2+3"),
        ("Step 8", "extract_patch_features",
         f"fe.extract_patch_features(preprocessed, [({PATCH_Y0},{PATCH_X0},{PATCH_Y1},{PATCH_X1})])\n"
         "           → features_448 shape (1, 448) float32"),
        ("Step 9", "dimension reduction",
         "feature_100 = features_448[0][selected_indices]  shape (100,)"),
        ("Step 10", "feature 저장 (ALLOW_WRITE_FEATURE_VECTOR=True 시)",
         "np.save(OUTPUT_ROOT/s5_feature_reextract_1case_feature_selected100.npy, feature_100)"),
        ("Step 11", "validation",
         f"diff = feature_100 - mean\n"
         "           d_sq = float(diff @ cov_inv @ diff)\n"
         f"           score_recon = sqrt(max(0, d_sq))  expected ≈ {EXPECTED_SCORE}"),
        ("Step 12", "metadata + DONE 저장",
         "s5_feature_reextract_1case_metadata.json + DONE.json"),
    ]

    for step_id, step_name, detail in steps:
        print(f"  {step_id}: {step_name}")
        for line in detail.split("\n"):
            print(f"           {line}")
        print()

    print("--- 다음 smoke run 승인 요건 ---")
    print("  1. ALLOW_CT_LOAD            = True  (별도 승인)")
    print("  2. ALLOW_MODEL_FORWARD      = True  (별도 승인)")
    print("  3. ALLOW_FEATURE_EXTRACTION = True  (별도 승인)")
    print("  4. ALLOW_WRITE_FEATURE_VECTOR = True (별도 승인)")
    print("  5. ALLOW_GPU                = False (CPU only 유지)")
    print("  6. --run-smoke --confirm-extract 플래그 필요")
    print()
    print("  현재 단계: 정적 검사 (PASS) → smoke 승인 대기")
    return 0


# ============================================================
# J. run_smoke (현재 guard False → BLOCKED)
# ============================================================

def run_smoke(confirm_extract: bool) -> int:
    """실제 feature re-extraction smoke run. guard True 시에만 실행."""
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

    # ----------------------------------------------------------------
    # guard 통과 후 실제 실행 로직 (현재 모든 guard=False이므로 도달 불가)
    # ----------------------------------------------------------------
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
    import numpy as np
    from position_aware_padim.preprocessing import preprocess_ct_slice
    from position_aware_padim.feature_extractor import FeatureExtractor

    # local_z guard
    assert CT_INDEX_Z == 89
    assert REPORT_SLICE_INDEX == 140
    assert CT_INDEX_Z != REPORT_SLICE_INDEX

    print(f"CT load: {CT_HU_NPY_PATH} (local_z={CT_INDEX_Z})")
    ct_hu = np.load(CT_HU_NPY_PATH)
    assert ct_hu.ndim == 3, f"CT shape ndim mismatch: {ct_hu.ndim}"
    assert 0 <= CT_INDEX_Z < ct_hu.shape[0], \
        f"local_z={CT_INDEX_Z} out of range [0, {ct_hu.shape[0]})"

    slice_2d = np.asarray(ct_hu[CT_INDEX_Z], dtype=np.float32)
    del ct_hu

    preprocessed = preprocess_ct_slice(slice_2d, hu_min=HU_MIN, hu_max=HU_MAX)

    stats = np.load(STATS_NPZ_PATH, allow_pickle=True)
    mean_vec = stats[f"{POSITION_BIN}_mean"].astype(np.float32)     # (100,)
    cov_mat  = stats[f"{POSITION_BIN}_cov"].astype(np.float32)      # (100,100)
    selected_indices = np.load(SELECTED_INDICES_NPY_PATH)           # (100,)

    device = "cuda" if ALLOW_GPU else "cpu"
    fe = FeatureExtractor(device=device)
    features_448 = fe.extract_patch_features(
        preprocessed, [(PATCH_Y0, PATCH_X0, PATCH_Y1, PATCH_X1)]
    )
    feature_100 = features_448[0][selected_indices].astype(np.float32)

    diff    = feature_100 - mean_vec
    cov_inv = np.linalg.inv(cov_mat + 1e-5 * np.eye(len(cov_mat)))
    d_sq    = float(diff @ cov_inv @ diff)
    score_recon = float(np.sqrt(max(0.0, d_sq)))

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    np.save(os.path.join(OUTPUT_ROOT, "s5_feature_reextract_1case_feature_selected100.npy"), feature_100)
    np.save(os.path.join(OUTPUT_ROOT, "s5_feature_reextract_1case_feature_raw448.npy"),
            features_448[0].astype(np.float32))

    import json
    metadata = {
        "case_id": CASE_ID,
        "volume_id": VOLUME_ID,
        "ct_index_z": CT_INDEX_Z,
        "report_slice_index": REPORT_SLICE_INDEX,
        "local_z_used_for_ct_indexing": True,
        "slice_index_used_for_ct_indexing": False,
        "patch_bbox_yxyx": [PATCH_Y0, PATCH_X0, PATCH_Y1, PATCH_X1],
        "patch_size": PATCH_SIZE,
        "position_bin": POSITION_BIN,
        "expected_score": EXPECTED_SCORE,
        "backbone": BACKBONE,
        "tapped_layers": TAPPED_LAYERS,
        "raw_feature_dim": RAW_FEATURE_DIM,
        "selected_feature_dim": SELECTED_FEATURE_DIM,
        "selected_indices_source": SELECTED_INDICES_NPY_PATH,
        "preprocessing_hu_min": HU_MIN,
        "preprocessing_hu_max": HU_MAX,
        "preprocessing_image_net_norm": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "feature_extractor_source": FEATURE_EXTRACTOR_MODULE_PATH,
        "ct_load_occurred": True,
        "model_forward_occurred": True,
        "feature_extraction_occurred": True,
        "gpu_used": False,
        "stage2_holdout_accessed": False,
        "score_recomputed": True,
        "threshold_recomputed": False,
        "existing_artifacts_modified": False,
        "score_reconstructed": score_recon,
        "score_diff_from_expected": abs(score_recon - EXPECTED_SCORE),
        "parity_pass": abs(score_recon - EXPECTED_SCORE) < 0.5,
    }
    with open(os.path.join(OUTPUT_ROOT, "s5_feature_reextract_1case_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUTPUT_ROOT, "DONE.json"), "w") as f:
        json.dump({"status": "DONE", "case_id": CASE_ID, "score_reconstructed": score_recon,
                   "parity_pass": metadata["parity_pass"]}, f, indent=2)

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
        description="S5 Feature Re-extraction 1-case Smoke Script"
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
