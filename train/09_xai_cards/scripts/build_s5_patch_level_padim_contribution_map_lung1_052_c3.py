"""
S5 Patch-level PaDiM Contribution Map — LUNG1-052__c3 Smoke Script
===================================================================
대상: LUNG1-052__c3 / stage1_dev only
단계: 3×3 patch grid feature re-extraction + Mahalanobis contribution map

실행 방법:
  --selftest          : 정적 selftest 80+ 항목 (허용)
  --dry-run           : 경로 확인, load/extract/calc 없음 (허용)
  --plan-only         : 실행 계획 출력 (허용)
  --run-smoke --confirm-map : 실제 실행 (guard True 시에만)

금지 (현재 모든 guard=False):
  bare 실행           → BLOCKED exit 2
  --run-smoke 단독    → BLOCKED exit 2
  --run-smoke --confirm-map → BLOCKED exit 2 (guard False)

중요 — local_z / report_slice 구분:
  CT_INDEX_Z = 51        ← CT npy indexing에 사용 (local_z)
  REPORT_SLICE_INDEX = 106  ← global/reporting 전용 (CT index 아님)
  LUNG1-052 offset = 55  (LUNG1-320 offset=51과 다름 — 환자별 독립 확인 필수)

contribution formula (patch별 동일):
  diff   = x - mean                    # (100,)
  v      = cov_inv @ diff              # (100,)
  contrib_i = diff_i * v_i             # (100,) — 음수 가능
  d2     = diff.T @ cov_inv @ diff     # scalar (Mahalanobis^2)
  check: abs(sum(contrib) - d2) < tol

layer boundary (raw448):
  layer1: raw index [0,   64)
  layer2: raw index [64,  192)
  layer3: raw index [192, 448)

dim91 caveat:
  dim91/raw427/layer3는 LUNG1-320__c2와 LUNG1-052__c3 모두에서 강하게 나타났음.
  해부학적 구조로 해석 금지. covariance inverse amplification 또는
  공통 high-response feature-space pattern 가능성.
  multi-case review 전까지 단정 금지.

명칭 정책:
  허용: PaDiM patch-level contribution map
        patch-level Mahalanobis contribution map
        feature-space contribution summary map
        patch-grid contribution visualization
  금지: Grad-CAM / pixel attribution / lesion attribution map / 진단 heatmap

caveat:
  이 map은 patch 단위 feature-space contribution을 시각화하기 위한 중간 산출물이며,
  픽셀 단위 원인이나 해부학적 구조를 직접 의미하지 않습니다.
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
POSITION_BIN       = "lower_central"
SPLIT              = "stage1_dev"

# local_z indexing 정책 명시
LOCAL_Z_USED_FOR_CT_INDEXING    = True
SLICE_INDEX_USED_FOR_CT_INDEXING = False

# ============================================================
# B. Preprocessing parity metadata
# ============================================================

HU_MIN                = -1000.0
HU_MAX                =  200.0
IMAGENET_MEAN         = [0.485, 0.456, 0.406]
IMAGENET_STD          = [0.229, 0.224, 0.225]
THREE_CHANNEL_POLICY  = "1ch → 3ch stack (repeat)"
BACKBONE              = "ResNet18"
TAPPED_LAYERS         = ["layer1", "layer2", "layer3"]
RAW_FEATURE_DIM       = 448
SELECTED_FEATURE_DIM  = 100

# ============================================================
# C. Patch grid constants
# ============================================================

PATCH_SIZE      = 32
GRID_SIZE       = 3           # 3×3
CENTER_PATCH_ID = 4           # row-major: (1,1)
CENTER_Y0       = 288
CENTER_X0       = 128
CENTER_Y1       = 320
CENTER_X1       = 160
CT_SHAPE_H      = 512
CT_SHAPE_W      = 512

# center patch parity references
CENTER_REEXTRACT_SCORE    = 38.87256  # feature_selected100.npy 기반 재현 score
RECORDED_SCORE_REFERENCE  = 39.34    # 원래 기록 score (metadata/reporting 참고 전용 — score CSV 수정 금지)
ROI_COVERAGE              = 1.000

# dim91 상수 (raw427, layer3) — actual run 시 selected_indices[91] 일치 확인 필수
DIM91_SELECTED_INDEX = 91
DIM91_RAW_INDEX_EXPECTED = 427  # prior LUNG1-320__c2 분석 기반 예상값
DIM91_LAYER          = "layer3"

# 3×3 patch grid (row-major, stride=PATCH_SIZE=32)
# center patch bbox = [y0=288, x0=128, y1=320, x1=160]
PATCH_GRID = [
    {"patch_id": 0, "row": 0, "col": 0, "y0": 256, "x0":  96, "y1": 288, "x1": 128, "is_center": False},
    {"patch_id": 1, "row": 0, "col": 1, "y0": 256, "x0": 128, "y1": 288, "x1": 160, "is_center": False},
    {"patch_id": 2, "row": 0, "col": 2, "y0": 256, "x0": 160, "y1": 288, "x1": 192, "is_center": False},
    {"patch_id": 3, "row": 1, "col": 0, "y0": 288, "x0":  96, "y1": 320, "x1": 128, "is_center": False},
    {"patch_id": 4, "row": 1, "col": 1, "y0": 288, "x0": 128, "y1": 320, "x1": 160, "is_center": True},   # center
    {"patch_id": 5, "row": 1, "col": 2, "y0": 288, "x0": 160, "y1": 320, "x1": 192, "is_center": False},
    {"patch_id": 6, "row": 2, "col": 0, "y0": 320, "x0":  96, "y1": 352, "x1": 128, "is_center": False},
    {"patch_id": 7, "row": 2, "col": 1, "y0": 320, "x0": 128, "y1": 352, "x1": 160, "is_center": False},
    {"patch_id": 8, "row": 2, "col": 2, "y0": 320, "x0": 160, "y1": 352, "x1": 192, "is_center": False},
]

# ============================================================
# D. Input paths
# ============================================================

CT_HU_NPY_PATH = (
    "/mnt/c/Users/jinhy/Desktop/"
    "NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/"
    "volumes_npy/NSCLC_LUNG1-052__d4a19cc211/ct_hu.npy"
)

# WARNING: stats는 padim_v2_roi0_0, selected_indices는 padim_v1 — 혼동 금지
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

RESNET18_CACHE_PATH = "/home/jinhy/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"

# 기존 1-patch reextract 결과 (center patch parity 참조용 — 수정 금지)
PRIOR_REEXTRACT_FEATURE_PATH = os.path.join(
    REPO_ROOT,
    "outputs/position-aware-padim-v1/reports/explanation_cards/"
    "s5_lung1_052_c3_feature_reextract_smoke_v1/feature_selected100.npy"
)

# 기존 1-patch contribution 결과 (참조용 — 수정 금지)
PRIOR_CONTRIBUTION_SUMMARY_PATH = os.path.join(
    REPO_ROOT,
    "outputs/position-aware-padim-v1/reports/explanation_cards/"
    "s5_lung1_052_c3_feature_contribution_1case_smoke_v1/feature_contribution_summary.json"
)

# stage2_holdout sentinel (접근 금지 확인용 — 절대 load 금지)
_STAGE2_HOLDOUT_SENTINEL = "stage2_holdout"

# padim_v1 stats 경로 (stats로 쓰면 안 됨 — 경고 비교용)
_PADIM_V1_STATS_PATH = os.path.join(
    REPO_ROOT,
    "outputs/position-aware-padim-v1/models/padim_v1/distributions/position_bin_stats.npz"
)

# ============================================================
# E. Output roots (충돌 방지)
# ============================================================

OUTPUT_ROOT = os.path.join(
    REPO_ROOT,
    "outputs/position-aware-padim-v1/reports/explanation_cards/"
    "s5_lung1_052_c3_patch_level_contribution_map_1case_smoke_v1"
)

# 충돌 방지 참조 root들 (절대 수정 금지)
_LUNG1_320_PATCH_MAP_ROOT = os.path.join(
    REPO_ROOT,
    "outputs/position-aware-padim-v1/reports/explanation_cards/"
    "s5_patch_level_contribution_map_1case_smoke_v1"
)
_LUNG1_052_REEXTRACT_ROOT = os.path.join(
    REPO_ROOT,
    "outputs/position-aware-padim-v1/reports/explanation_cards/"
    "s5_lung1_052_c3_feature_reextract_smoke_v1"
)
_LUNG1_052_CONTRIBUTION_ROOT = os.path.join(
    REPO_ROOT,
    "outputs/position-aware-padim-v1/reports/explanation_cards/"
    "s5_lung1_052_c3_feature_contribution_1case_smoke_v1"
)

# ============================================================
# F. Contribution formula constants
# ============================================================

COVARIANCE_EPSILON        = 1e-5
CONTRIBUTION_METHOD       = "full_inverse_cov"
SUM_MATCH_TOLERANCE       = 1e-3

LAYER1_RAW_START  = 0
LAYER1_RAW_END    = 64
LAYER2_RAW_START  = 64
LAYER2_RAW_END    = 192
LAYER3_RAW_START  = 192
LAYER3_RAW_END    = 448

CONTRIBUTION_FORMULA_STRING = (
    "diff = x - mean; "
    "v = cov_inv @ diff; "
    "contrib_i = diff_i * v_i; "
    "d2 = diff.T @ cov_inv @ diff; "
    "check: abs(sum(contrib) - d2) < tol"
)

# ============================================================
# G. Caveat / policy strings
# ============================================================

DIAGNOSTIC_CAVEAT = (
    "이 map은 patch 단위 feature-space contribution을 시각화하기 위한 중간 산출물이며, "
    "픽셀 단위 원인이나 해부학적 구조를 직접 의미하지 않습니다. "
    "Grad-CAM 아님. pixel attribution 아님. 임상 진단 근거 아님."
)

DIM91_DOMINANCE_CAVEAT = (
    "dim91/raw427/layer3는 LUNG1-320__c2와 LUNG1-052__c3 모두에서 강하게 나타났음. "
    "해부학적 구조로 해석 금지. "
    "covariance inverse amplification 또는 공통 high-response feature-space pattern 가능성. "
    "multi-case review 전까지 단정 금지."
)

NEGATIVE_CONTRIBUTION_POLICY = (
    "음수 contribution은 '정상 분포 방향으로의 covariance interaction'으로 기록. "
    "진단 의미 부여 금지."
)

# policy strings (selftest 참조)
_DRY_RUN_POLICY = (
    "dry-run: path existence only. "
    "CT load 없음. model forward 없음. feature extraction 없음. "
    "contribution 계산 없음. write 없음."
)
_PLAN_ONLY_POLICY = (
    "plan-only: 3×3 patch grid + formula + output schema 출력. "
    "CT load 없음. model forward 없음. feature extraction 없음. "
    "contribution 계산 없음. write 없음."
)

# ============================================================
# H. Basic guards (all False — 실제 실행 금지)
# ============================================================

ALLOW_CT_LOAD              = False
ALLOW_MODEL_FORWARD        = False
ALLOW_FEATURE_EXTRACTION   = False
ALLOW_FEATURE_WRITE        = False
ALLOW_STATS_LOAD           = False
ALLOW_CONTRIBUTION_CALC    = False
ALLOW_CONTRIBUTION_WRITE   = False
ALLOW_MAP_JSON_WRITE       = False
ALLOW_GPU                  = False
ALLOW_STAGE2_HOLDOUT       = False
ALLOW_FULL_300             = False


# ============================================================
# I. local_z guard
# ============================================================

def _assert_local_z_guards():
    """local_z / slice_index 혼동 방지 guard."""
    assert CT_INDEX_Z == 51, \
        f"CT_INDEX_Z guard FAIL: {CT_INDEX_Z}"
    assert REPORT_SLICE_INDEX == 106, \
        f"REPORT_SLICE_INDEX guard FAIL: {REPORT_SLICE_INDEX}"
    assert CT_INDEX_Z != REPORT_SLICE_INDEX, (
        f"CT_INDEX_Z({CT_INDEX_Z}) == REPORT_SLICE_INDEX({REPORT_SLICE_INDEX}) — 혼동 위험"
    )
    assert LOCAL_Z_USED_FOR_CT_INDEXING is True, \
        "LOCAL_Z_USED_FOR_CT_INDEXING must be True"
    assert SLICE_INDEX_USED_FOR_CT_INDEXING is False, \
        "SLICE_INDEX_USED_FOR_CT_INDEXING must be False"


# ============================================================
# J. Patch grid validation helpers
# ============================================================

def _validate_patch_grid() -> list:
    """3×3 patch grid 좌표 정적 검증. 오류 리스트 반환."""
    errors = []
    if len(PATCH_GRID) != GRID_SIZE * GRID_SIZE:
        errors.append(f"PATCH_GRID len={len(PATCH_GRID)}, expected {GRID_SIZE*GRID_SIZE}")
        return errors
    center_found = False
    for p in PATCH_GRID:
        pid = p["patch_id"]
        y0, x0, y1, x1 = p["y0"], p["x0"], p["y1"], p["x1"]
        if (y1 - y0) != PATCH_SIZE:
            errors.append(f"patch_id={pid}: height={y1-y0} != {PATCH_SIZE}")
        if (x1 - x0) != PATCH_SIZE:
            errors.append(f"patch_id={pid}: width={x1-x0} != {PATCH_SIZE}")
        if not (0 <= y0 < y1 <= CT_SHAPE_H):
            errors.append(f"patch_id={pid}: y=[{y0},{y1}] out of [0,{CT_SHAPE_H}]")
        if not (0 <= x0 < x1 <= CT_SHAPE_W):
            errors.append(f"patch_id={pid}: x=[{x0},{x1}] out of [0,{CT_SHAPE_W}]")
        if p["is_center"]:
            center_found = True
            if pid != CENTER_PATCH_ID:
                errors.append(f"is_center patch_id={pid} != CENTER_PATCH_ID={CENTER_PATCH_ID}")
            if y0 != CENTER_Y0 or x0 != CENTER_X0 or y1 != CENTER_Y1 or x1 != CENTER_X1:
                errors.append(
                    f"center bbox mismatch: got ({y0},{x0},{y1},{x1}) "
                    f"expected ({CENTER_Y0},{CENTER_X0},{CENTER_Y1},{CENTER_X1})"
                )
    if not center_found:
        errors.append("no center patch found (is_center=True)")
    return errors


def _layer_label(raw_idx: int) -> str:
    if LAYER1_RAW_START <= raw_idx < LAYER1_RAW_END:
        return "layer1"
    elif LAYER2_RAW_START <= raw_idx < LAYER2_RAW_END:
        return "layer2"
    elif LAYER3_RAW_START <= raw_idx < LAYER3_RAW_END:
        return "layer3"
    return "unknown"


# ============================================================
# K. Schema helpers
# ============================================================

def _get_patch_contribution_summary_schema_keys() -> list:
    return [
        "patch_id", "grid_row", "grid_col",
        "y0", "x0", "y1", "x1",
        "is_center",
        "sqrt_mahalanobis", "d2",
        "sum_contribution", "sum_match_error", "sum_match_pass",
        "top1_abs_selected_dim", "top1_abs_raw_dim", "top1_abs_layer", "top1_abs_contribution",
        "top2_abs_contribution_sum",
        "top5_abs_contribution_sum", "top5_concentration_ratio",
        "top10_concentration_ratio",
        "layer1_fraction_abs", "layer2_fraction_abs", "layer3_fraction_abs",
        "dim91_contribution", "dim91_rank_abs",
        "n_positive_contrib", "n_negative_contrib",
    ]


def _get_contribution_map_json_schema_keys() -> list:
    return [
        "case_id", "position_bin",
        "ct_index_z", "report_slice_index",
        "local_z_used_for_ct_indexing", "slice_index_used_for_ct_indexing",
        "grid_size", "n_patches", "center_patch_id",
        "score_map_sqrt_mahalanobis",
        "top5_concentration_map",
        "layer3_fraction_map",
        "dim91_contribution_map",
        "center_score",
        "max_patch_id", "max_patch_score", "max_to_center_ratio",
        "not_gradcam", "not_pixel_attribution", "not_diagnostic",
        "caveat", "dim91_dominance_caveat",
    ]


def _get_xai_patch_map_bridge_schema_keys() -> list:
    return [
        "s5_case_id",
        "contribution_method",
        "grid_size",
        "n_patches",
        "center_patch_id",
        "map_type",
        "caveat",
        "not_diagnostic",
        "not_gradcam",
        "not_pixel_attribution",
        "not_spatial_heatmap_yet",
        "dim91_dominance_caveat",
        "note",
    ]


# ============================================================
# L. selftest (80+ 항목)
# ============================================================

def run_selftest() -> int:
    """정적 selftest 80+항목. 실제 load/extract/calc 없음."""
    passed = []
    failed = []

    def chk(name: str, cond: bool, msg: str = ""):
        if cond:
            passed.append(name)
        else:
            failed.append(f"{name}: {msg}")

    # --- 1. all guards default False (11항목) ---
    chk("01_guard_ct_load_false",              not ALLOW_CT_LOAD,              "ALLOW_CT_LOAD must be False")
    chk("02_guard_model_forward_false",        not ALLOW_MODEL_FORWARD,        "ALLOW_MODEL_FORWARD must be False")
    chk("03_guard_feature_extraction_false",   not ALLOW_FEATURE_EXTRACTION,   "ALLOW_FEATURE_EXTRACTION must be False")
    chk("04_guard_feature_write_false",        not ALLOW_FEATURE_WRITE,        "ALLOW_FEATURE_WRITE must be False")
    chk("05_guard_stats_load_false",           not ALLOW_STATS_LOAD,           "ALLOW_STATS_LOAD must be False")
    chk("06_guard_contribution_calc_false",    not ALLOW_CONTRIBUTION_CALC,    "ALLOW_CONTRIBUTION_CALC must be False")
    chk("07_guard_contribution_write_false",   not ALLOW_CONTRIBUTION_WRITE,   "ALLOW_CONTRIBUTION_WRITE must be False")
    chk("08_guard_map_json_write_false",       not ALLOW_MAP_JSON_WRITE,       "ALLOW_MAP_JSON_WRITE must be False")
    chk("09_guard_gpu_false",                  not ALLOW_GPU,                  "ALLOW_GPU must be False")
    chk("10_guard_stage2_holdout_false",       not ALLOW_STAGE2_HOLDOUT,       "ALLOW_STAGE2_HOLDOUT must be False")
    chk("11_guard_full300_false",              not ALLOW_FULL_300,             "ALLOW_FULL_300 must be False")

    # --- 2. case / volume ID ---
    chk("12_case_id_exact",   CASE_ID == "LUNG1-052__c3",                   f"got {CASE_ID}")
    chk("13_volume_id_exact", VOLUME_ID == "NSCLC_LUNG1-052__d4a19cc211",   f"got {VOLUME_ID}")

    # --- 3. position_bin ---
    chk("14_position_bin_lower_central", POSITION_BIN == "lower_central",   f"got {POSITION_BIN}")

    # --- 4. local_z / slice_index ---
    chk("15_ct_index_z_51",                CT_INDEX_Z == 51,                   f"got {CT_INDEX_Z}")
    chk("16_report_slice_index_106",       REPORT_SLICE_INDEX == 106,          f"got {REPORT_SLICE_INDEX}")
    chk("17_ct_index_z_ne_report_slice",   CT_INDEX_Z != REPORT_SLICE_INDEX,   "local_z == report_slice — 혼동 위험")
    chk("18_local_z_used_for_ct_indexing", LOCAL_Z_USED_FOR_CT_INDEXING is True,
        "LOCAL_Z_USED_FOR_CT_INDEXING must be True")
    chk("19_slice_index_not_for_ct",       SLICE_INDEX_USED_FOR_CT_INDEXING is False,
        "SLICE_INDEX_USED_FOR_CT_INDEXING must be False")

    import inspect as _inspect
    _mod_src = _inspect.getsource(sys.modules[__name__])

    # report_slice policy string 존재
    chk("20_report_slice_policy_in_src",
        "CT index 아님" in _mod_src,
        "report_slice CT index 금지 policy string 없음")
    # CT indexing이 CT_INDEX_Z만 사용
    chk("21_ct_indexing_uses_local_z_only",
        "ct_hu[CT_INDEX_Z]" in _mod_src,
        "CT indexing에 CT_INDEX_Z 사용 확인 필요")

    # --- 5. patch grid ---
    chk("22_grid_size_3",       GRID_SIZE == 3,             f"got {GRID_SIZE}")
    chk("23_patch_count_9",     len(PATCH_GRID) == 9,       f"got {len(PATCH_GRID)}")
    chk("24_center_patch_id_4", CENTER_PATCH_ID == 4,       f"got {CENTER_PATCH_ID}")
    chk("25_center_y0_288",     CENTER_Y0 == 288,           f"got {CENTER_Y0}")
    chk("26_center_x0_128",     CENTER_X0 == 128,           f"got {CENTER_X0}")
    chk("27_center_y1_320",     CENTER_Y1 == 320,           f"got {CENTER_Y1}")
    chk("28_center_x1_160",     CENTER_X1 == 160,           f"got {CENTER_X1}")

    grid_errors = _validate_patch_grid()
    chk("29_all_patches_valid",     len(grid_errors) == 0,
        f"patch grid errors: {grid_errors}")
    chk("30_all_patches_32x32",
        all((p["y1"]-p["y0"]) == 32 and (p["x1"]-p["x0"]) == 32 for p in PATCH_GRID),
        "not all patches are 32×32")
    chk("31_all_patches_in_512",
        all(0 <= p["y0"] and p["y1"] <= 512 and 0 <= p["x0"] and p["x1"] <= 512
            for p in PATCH_GRID),
        "some patch out of 512×512 bounds")
    chk("32_no_clamp_policy", PATCH_SIZE == 32, "patch_size must be 32 — no clamp needed")
    center_patches = [p for p in PATCH_GRID if p["is_center"]]
    chk("33_exactly_one_center",   len(center_patches) == 1, f"found {len(center_patches)} center patches")
    # component bbox는 metadata note 전용 — grid extraction에 사용 금지
    chk("34_component_bbox_not_in_patch_grid",
        all("COMPONENT" not in str(p) for p in PATCH_GRID),
        "COMPONENT bbox가 PATCH_GRID 항목에 혼입")

    # --- 6. preprocessing parity ---
    chk("35_hu_min_neg1000",       HU_MIN == -1000.0,                  f"got {HU_MIN}")
    chk("36_hu_max_200",           HU_MAX == 200.0,                    f"got {HU_MAX}")
    chk("37_imagenet_mean_len3",   len(IMAGENET_MEAN) == 3,            "IMAGENET_MEAN len must be 3")
    chk("38_imagenet_std_len3",    len(IMAGENET_STD) == 3,             "IMAGENET_STD len must be 3")
    chk("39_3ch_policy_exists",    "3ch" in THREE_CHANNEL_POLICY,      "3ch repeat policy missing")
    chk("40_backbone_resnet18",    BACKBONE == "ResNet18",              f"got {BACKBONE}")
    chk("41_tapped_layers",        TAPPED_LAYERS == ["layer1","layer2","layer3"], f"got {TAPPED_LAYERS}")
    chk("42_raw_dim_448",          RAW_FEATURE_DIM == 448,             f"got {RAW_FEATURE_DIM}")
    chk("43_selected_dim_100",     SELECTED_FEATURE_DIM == 100,        f"got {SELECTED_FEATURE_DIM}")
    chk("44_no_efficientnet",      "efficientnet" not in BACKBONE.lower(), "EfficientNet branch 혼입 위험")

    # --- 7. stats / selected_indices path policy ---
    chk("45_stats_path_padim_v2",
        "padim_v2_roi0_0" in STATS_NPZ_PATH,
        f"STATS_NPZ_PATH must contain padim_v2_roi0_0: {STATS_NPZ_PATH}")
    chk("46_selected_indices_path_padim_v1",
        "padim_v1" in SELECTED_INDICES_NPY_PATH and "padim_v2" not in SELECTED_INDICES_NPY_PATH,
        f"SELECTED_INDICES must be padim_v1 only: {SELECTED_INDICES_NPY_PATH}")
    chk("47_stats_not_padim_v1_only",
        "padim_v2" in STATS_NPZ_PATH,
        f"STATS must use padim_v2_roi0_0, not padim_v1: {STATS_NPZ_PATH}")

    # --- 8. planned stats keys (lower_central_mean / lower_central_cov) ---
    chk("48_lower_central_mean_key_planned",
        "lower_central_mean" in _mod_src,
        "lower_central_mean key not mentioned in source")
    chk("49_lower_central_cov_key_planned",
        "lower_central_cov" in _mod_src,
        "lower_central_cov key not mentioned in source")
    chk("50_selected_feature_indices_key_planned",
        "selected_feature_indices" in _mod_src,
        "selected_feature_indices key not mentioned in source")

    # --- 9. selected_indices equality guard ---
    chk("51_selected_indices_equality_guard_planned",
        "np.array_equal" in _mod_src,
        "selected_indices equality guard (np.array_equal) not in source")
    chk("52_selected_indices_final_from_file",
        "selected_indices_file" in _mod_src,
        "selected_indices_file not in source (layer mapping must use file-loaded indices)")

    # --- 10. cov epsilon ---
    chk("53_cov_epsilon_1e5", abs(COVARIANCE_EPSILON - 1e-5) < 1e-10, f"got {COVARIANCE_EPSILON}")

    # --- 11. contribution formula ---
    chk("54_formula_string_exists", len(CONTRIBUTION_FORMULA_STRING) > 0,    "formula string empty")
    chk("55_formula_has_diff",      "diff" in CONTRIBUTION_FORMULA_STRING,   "diff missing in formula")
    chk("56_formula_has_cov_inv",   "cov_inv" in CONTRIBUTION_FORMULA_STRING,"cov_inv missing in formula")
    chk("57_formula_has_contrib",   "contrib" in CONTRIBUTION_FORMULA_STRING,"contrib missing in formula")
    chk("58_formula_has_check",     "check" in CONTRIBUTION_FORMULA_STRING,  "sum-check missing in formula")
    summary_keys = _get_patch_contribution_summary_schema_keys()
    chk("59_sum_match_error_in_schema",
        "sum_match_error" in summary_keys,
        "sum_match_error missing from summary schema")

    # --- 12. center score references ---
    chk("60_center_reextract_score_38_87256",
        abs(CENTER_REEXTRACT_SCORE - 38.87256) < 1e-6,
        f"CENTER_REEXTRACT_SCORE mismatch: got {CENTER_REEXTRACT_SCORE}")
    chk("61_recorded_score_39_34",
        abs(RECORDED_SCORE_REFERENCE - 39.34) < 0.01,
        f"RECORDED_SCORE_REFERENCE mismatch: got {RECORDED_SCORE_REFERENCE}")
    chk("62_center_score_match_tolerance_1e3",
        "1e-3" in _mod_src,
        "1e-3 tolerance not found in source")

    # --- 13. patch summary schema ---
    chk("63_schema_has_dim91_contribution",
        "dim91_contribution" in summary_keys, "dim91_contribution missing from summary schema")
    chk("64_schema_has_dim91_rank_abs",
        "dim91_rank_abs" in summary_keys, "dim91_rank_abs missing from summary schema")
    chk("65_schema_has_top5_concentration_ratio",
        "top5_concentration_ratio" in summary_keys, "top5_concentration_ratio missing from summary schema")
    chk("66_schema_has_layer3_fraction_abs",
        "layer3_fraction_abs" in summary_keys, "layer3_fraction_abs missing from summary schema")

    # --- 14. map JSON schema ---
    map_keys = _get_contribution_map_json_schema_keys()
    chk("67_map_schema_has_score_map",
        "score_map_sqrt_mahalanobis" in map_keys, "score_map_sqrt_mahalanobis missing from map schema")
    chk("68_map_schema_has_dim91_map",
        "dim91_contribution_map" in map_keys, "dim91_contribution_map missing from map schema")
    chk("69_map_schema_has_top5_concentration_map",
        "top5_concentration_map" in map_keys, "top5_concentration_map missing from map schema")
    chk("70_map_schema_has_layer3_fraction_map",
        "layer3_fraction_map" in map_keys, "layer3_fraction_map missing from map schema")

    # --- 15. naming / caveat policy ---
    chk("71_diagnostic_caveat_exists",           len(DIAGNOSTIC_CAVEAT) > 0,          "caveat empty")
    chk("72_not_gradcam_in_caveat",              "Grad-CAM" in DIAGNOSTIC_CAVEAT,      "Grad-CAM disclaimer missing")
    chk("73_not_pixel_attribution_in_caveat",    "pixel attribution" in DIAGNOSTIC_CAVEAT,
        "pixel attribution disclaimer missing")
    chk("74_not_diagnostic_in_caveat",           "진단" in DIAGNOSTIC_CAVEAT,           "diagnostic disclaimer missing")
    chk("75_dim91_dominance_caveat_exists",      len(DIM91_DOMINANCE_CAVEAT) > 0,      "DIM91_DOMINANCE_CAVEAT empty")
    chk("76_dim91_caveat_no_anatomy",
        "해부학적 구조로 해석 금지" in DIM91_DOMINANCE_CAVEAT,
        "dim91 caveat must forbid anatomical interpretation")
    chk("77_no_lesion_cancer_in_formula",
        "암" not in CONTRIBUTION_FORMULA_STRING and "병변" not in CONTRIBUTION_FORMULA_STRING,
        "lesion/cancer wording in formula string")
    chk("78_no_gradcam_in_method",
        "Grad-CAM" not in CONTRIBUTION_METHOD,
        "Grad-CAM 표현이 CONTRIBUTION_METHOD에 혼입")

    # --- 16. execution policy ---
    import inspect
    run_smoke_src = inspect.getsource(run_smoke)
    chk("79_run_requires_confirm_map",
        "confirm_map" in run_smoke_src,
        "run_smoke에 --confirm-map 체크 없음")
    chk("80_run_requires_all_guards",
        "ALLOW_CT_LOAD" in run_smoke_src and "ALLOW_MODEL_FORWARD" in run_smoke_src,
        "run_smoke가 필수 guard를 확인하지 않음")
    chk("81_ct_load_uses_mmap_r",
        'mmap_mode="r"' in run_smoke_src,
        'CT load에 mmap_mode="r" 없음 — 일반 np.load 금지')

    # --- 17. dry-run / plan-only policy ---
    chk("82_dryrun_no_ct_load",
        "CT load 없음" in _DRY_RUN_POLICY, "dry-run must state no CT load")
    chk("83_dryrun_no_contrib_calc",
        "contribution 계산 없음" in _DRY_RUN_POLICY, "dry-run must state no contrib calc")
    chk("84_planonly_no_contrib_calc",
        "contribution 계산 없음" in _PLAN_ONLY_POLICY, "plan-only must state no contrib calc")

    # --- 18. output root separation ---
    chk("85_output_root_in_outputs",
        "outputs/" in OUTPUT_ROOT, f"output root outside outputs/: {OUTPUT_ROOT}")
    chk("86_output_root_map_specific",
        "patch_level_contribution_map" in OUTPUT_ROOT, "output root not map-specific")
    chk("87_output_root_contains_052",
        "052" in OUTPUT_ROOT, "output root does not contain '052'")
    chk("88_output_root_ne_lung1_320_map",
        OUTPUT_ROOT != _LUNG1_320_PATCH_MAP_ROOT,
        "output root 충돌: LUNG1-320 patch map과 동일")
    chk("89_output_root_ne_lung1_052_reextract",
        OUTPUT_ROOT != _LUNG1_052_REEXTRACT_ROOT,
        "output root 충돌: LUNG1-052 reextract와 동일")
    chk("90_output_root_ne_lung1_052_contribution",
        OUTPUT_ROOT != _LUNG1_052_CONTRIBUTION_ROOT,
        "output root 충돌: LUNG1-052 contribution과 동일")

    # --- 19. guard / safety invariants ---
    chk("91_no_gpu",             not ALLOW_GPU,             "GPU must be disabled")
    chk("92_no_stage2_holdout",  not ALLOW_STAGE2_HOLDOUT,  "stage2_holdout guard must be False")
    chk("93_no_full300",         not ALLOW_FULL_300,        "ALLOW_FULL_300 must be False")
    chk("94_stage2_sentinel_not_in_ct_path",
        _STAGE2_HOLDOUT_SENTINEL not in CT_HU_NPY_PATH,
        "CT 경로에 stage2_holdout 포함 위험")

    # --- 20. output files planned ---
    chk("95_errors_csv_planned",
        "errors.csv" in _mod_src, "errors.csv not planned in source")
    chk("96_done_json_planned",
        "DONE.json" in _mod_src, "DONE.json not planned in source")
    chk("97_patch_grid_plan_csv_planned",
        "patch_grid_plan.csv" in _mod_src, "patch_grid_plan.csv not planned in source")
    chk("98_patch_feature_vectors_planned",
        "patch_feature_vectors_selected100.npy" in _mod_src,
        "patch_feature_vectors_selected100.npy not planned")
    chk("99_contribution_full_planned",
        "patch_contribution_full.csv" in _mod_src,
        "patch_contribution_full.csv not planned")
    chk("100_contribution_summary_planned",
        "patch_contribution_summary.csv" in _mod_src,
        "patch_contribution_summary.csv not planned")
    chk("101_xai_bridge_planned",
        "xai_patch_map_bridge.json" in _mod_src,
        "xai_patch_map_bridge.json not planned")

    # --- 21. xai bridge schema ---
    bridge_keys = _get_xai_patch_map_bridge_schema_keys()
    chk("102_bridge_has_caveat",          "caveat" in bridge_keys,              "xai bridge missing caveat")
    chk("103_bridge_has_not_gradcam",     "not_gradcam" in bridge_keys,         "xai bridge missing not_gradcam")
    chk("104_bridge_has_not_diagnostic",  "not_diagnostic" in bridge_keys,      "xai bridge missing not_diagnostic")
    chk("105_bridge_has_dim91_caveat",    "dim91_dominance_caveat" in bridge_keys,
        "xai bridge missing dim91_dominance_caveat")

    # --- 22. DONE collision check planned ---
    chk("106_done_collision_check_planned",
        "DONE.json" in run_smoke_src and "isfile" in run_smoke_src,
        "DONE.json 충돌 확인 없음 in run_smoke")

    # --- 23. write guards ---
    chk("107_feature_write_guarded",
        "ALLOW_FEATURE_WRITE" in run_smoke_src,
        "feature write not guarded in run_smoke")
    chk("108_contribution_write_guarded",
        "ALLOW_CONTRIBUTION_WRITE" in run_smoke_src,
        "contribution write not guarded in run_smoke")
    chk("109_map_json_write_guarded",
        "ALLOW_MAP_JSON_WRITE" in run_smoke_src,
        "map JSON write not guarded in run_smoke")

    # --- 24. no score/threshold recalculation ---
    chk("110_no_score_csv_modification",
        "score CSV 수정 금지" in _mod_src,
        "score CSV 수정 금지 policy 없음")
    chk("111_no_threshold_recalculation",
        "threshold_recomputed" not in run_smoke_src or "False" in run_smoke_src,
        "threshold recalculation guard 없음")

    # --- 25. no PNG generation ---
    chk("112_no_png_generation",
        ".png" not in run_smoke_src,
        "PNG 생성 코드가 run_smoke에 포함됨")

    # --- 26. no heatmap wording ---
    chk("113_no_heatmap_final_wording",
        "진단 heatmap" not in CONTRIBUTION_METHOD,
        "진단 heatmap wording in CONTRIBUTION_METHOD")

    # --- 결과 출력 ---
    total = len(passed) + len(failed)
    print(f"\n=== SELFTEST 결과 (LUNG1-052__c3 patch-level map) ===")
    print(f"PASS: {len(passed)}/{total}")
    if failed:
        print(f"FAIL: {len(failed)}/{total}")
        for f_item in failed:
            print(f"  FAIL: {f_item}")
        print("\n판정: FAIL")
        return 1
    print(f"모든 {total}개 selftest 통과.")
    print("판정: PASS")
    return 0


# ============================================================
# M. dry-run
# ============================================================

def run_dry_run() -> int:
    """경로 확인. CT load / model forward / feature extraction / calc / write 없음."""
    print(f"\n=== DRY-RUN (LUNG1-052__c3 patch-level map) ===")
    print(f"정책: {_DRY_RUN_POLICY}")
    print()
    print(f"[대상]")
    print(f"  CASE_ID            = {CASE_ID}")
    print(f"  VOLUME_ID          = {VOLUME_ID}")
    print(f"  CT_INDEX_Z         = {CT_INDEX_Z}   ← CT npy indexing에 사용 (local_z)")
    print(f"  REPORT_SLICE_INDEX = {REPORT_SLICE_INDEX}  ← global/reporting 전용 (CT index 아님)")
    print(f"  LUNG1-052 offset   = {REPORT_SLICE_INDEX - CT_INDEX_Z}  (LUNG1-320 offset=51과 다름)")
    print(f"  POSITION_BIN       = {POSITION_BIN}")
    print(f"  GRID_SIZE          = {GRID_SIZE}×{GRID_SIZE} = {GRID_SIZE*GRID_SIZE} patches")
    print(f"  CENTER_PATCH_ID    = {CENTER_PATCH_ID}")
    print(f"  CENTER_BBOX        = [y0={CENTER_Y0}, x0={CENTER_X0}, y1={CENTER_Y1}, x1={CENTER_X1}]")
    print()

    errors = []

    def chk_path(label, path):
        exists = os.path.isfile(path)
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {label}")
        print(f"         {path}")
        if not exists:
            errors.append(f"{label} missing: {path}")

    print("[입력 경로 확인 (load 없음)]")
    chk_path("CT_HU_NPY",             CT_HU_NPY_PATH)
    chk_path("STATS_NPZ (v2_roi0_0)", STATS_NPZ_PATH)
    chk_path("SELECTED_INDICES (v1)", SELECTED_INDICES_NPY_PATH)
    chk_path("FEATURE_EXTRACTOR",     FEATURE_EXTRACTOR_MODULE_PATH)
    chk_path("PREPROCESSING",         PREPROCESSING_MODULE_PATH)
    chk_path("RESNET18_CACHE",        RESNET18_CACHE_PATH)
    chk_path("PRIOR_REEXTRACT_FEAT",  PRIOR_REEXTRACT_FEATURE_PATH)
    chk_path("PRIOR_CONTRIBUTION",    PRIOR_CONTRIBUTION_SUMMARY_PATH)
    print()

    print("[stats / selected_indices path policy 확인]")
    if "padim_v2_roi0_0" in STATS_NPZ_PATH:
        print("  [OK] stats → padim_v2_roi0_0 (정상)")
    else:
        print("  [WARN] stats 경로에 padim_v2_roi0_0 없음")
        errors.append("stats path not padim_v2_roi0_0")
    if "padim_v1" in SELECTED_INDICES_NPY_PATH and "padim_v2" not in SELECTED_INDICES_NPY_PATH:
        print("  [OK] selected_indices → padim_v1 only (정상)")
    else:
        print("  [WARN] selected_indices 경로 혼동 위험")
        errors.append("selected_indices path mismatch")
    print()

    print("[3×3 patch grid 확인 (clamp 금지, component bbox 사용 금지)]")
    grid_errors = _validate_patch_grid()
    if grid_errors:
        for ge in grid_errors:
            print(f"  [ERROR] {ge}")
            errors.append(ge)
    else:
        print("  [OK] 9개 patch 전부 valid (32×32, 0≤y,x≤512 범위 내)")
    for p in PATCH_GRID:
        cmark = "★center" if p["is_center"] else "      "
        print(f"    patch_id={p['patch_id']} row={p['row']} col={p['col']} "
              f"y=[{p['y0']},{p['y1']}) x=[{p['x0']},{p['x1']}) {cmark}")
    print()

    print("[output root 충돌 확인]")
    print(f"  OUTPUT_ROOT                  : {OUTPUT_ROOT}")
    print(f"  ≠ LUNG1-320 patch map        : {OUTPUT_ROOT != _LUNG1_320_PATCH_MAP_ROOT}")
    print(f"  ≠ LUNG1-052 reextract        : {OUTPUT_ROOT != _LUNG1_052_REEXTRACT_ROOT}")
    print(f"  ≠ LUNG1-052 1-patch contrib  : {OUTPUT_ROOT != _LUNG1_052_CONTRIBUTION_ROOT}")
    done_path = os.path.join(OUTPUT_ROOT, "DONE.json")
    done_exists = os.path.isfile(done_path)
    print(f"  DONE.json 기존 존재          : {'YES (BLOCKED)' if done_exists else 'NO (안전)'}")
    if done_exists:
        errors.append("DONE.json already exists — smoke run BLOCKED")
    print()

    print("[guard 상태]")
    for name, val in [
        ("ALLOW_CT_LOAD",            ALLOW_CT_LOAD),
        ("ALLOW_MODEL_FORWARD",      ALLOW_MODEL_FORWARD),
        ("ALLOW_FEATURE_EXTRACTION", ALLOW_FEATURE_EXTRACTION),
        ("ALLOW_FEATURE_WRITE",      ALLOW_FEATURE_WRITE),
        ("ALLOW_STATS_LOAD",         ALLOW_STATS_LOAD),
        ("ALLOW_CONTRIBUTION_CALC",  ALLOW_CONTRIBUTION_CALC),
        ("ALLOW_CONTRIBUTION_WRITE", ALLOW_CONTRIBUTION_WRITE),
        ("ALLOW_MAP_JSON_WRITE",     ALLOW_MAP_JSON_WRITE),
        ("ALLOW_GPU",                ALLOW_GPU),
        ("ALLOW_STAGE2_HOLDOUT",     ALLOW_STAGE2_HOLDOUT),
        ("ALLOW_FULL_300",           ALLOW_FULL_300),
    ]:
        status = "OK (False)" if not val else "WARN (True)"
        print(f"  [{status}] {name} = {val}")
    print()

    print("[feature extractor import 확인 (forward 없음)]")
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
    try:
        from position_aware_padim.feature_extractor import FeatureExtractor  # noqa: F401
        print("  [OK] FeatureExtractor import 성공 (forward 없음)")
    except ImportError as e:
        print(f"  [WARN] FeatureExtractor import 실패: {e}")
        errors.append(f"FeatureExtractor import failed: {e}")
    print()

    print("[safety 확인]")
    print("  CT load            : 0 (dry-run)")
    print("  model forward      : 0 (dry-run)")
    print("  feature extraction : 0 (dry-run)")
    print("  contribution calc  : 0 (dry-run)")
    print("  write              : 0 (dry-run)")
    print(f"  stage2_holdout     : {'YES (위험)' if _STAGE2_HOLDOUT_SENTINEL in CT_HU_NPY_PATH else 'NO (안전)'}")
    print()

    if errors:
        print(f"WARNING: {len(errors)}개 문제 발견")
        for e in errors:
            print(f"  - {e}")
        print("판정: WARNING (smoke 전 해결 필요)")
        return 1

    print("판정: DRY-RUN PASS")
    return 0


# ============================================================
# N. plan-only
# ============================================================

def run_plan_only() -> int:
    """실행 계획 출력. 계산/load/write 없음."""
    print(f"\n=== PLAN-ONLY (LUNG1-052__c3 patch-level map) ===")
    print(f"정책: {_PLAN_ONLY_POLICY}")
    print()
    print("+------------------------------------------------------------------------+")
    print("| S5 Patch-level PaDiM Contribution Map — LUNG1-052__c3 Smoke Plan      |")
    print("+------------------------------------------------------------------------+")
    print()

    print("⚠  CT INDEX 주의:")
    print(f"   CT npy indexing 시 반드시 local_z = {CT_INDEX_Z} 사용 (CT_INDEX_Z)")
    print(f"   report_slice = {REPORT_SLICE_INDEX} 는 global/reporting 전용 — CT index 절대 사용 금지")
    print(f"   offset = {REPORT_SLICE_INDEX - CT_INDEX_Z}  (LUNG1-320 offset=51과 다름)")
    print()

    print("[대상]")
    print(f"  CASE_ID                  = {CASE_ID}")
    print(f"  POSITION_BIN             = {POSITION_BIN}")
    print(f"  CENTER_PATCH_ID          = {CENTER_PATCH_ID}")
    print(f"  CENTER_BBOX              = [y0={CENTER_Y0},x0={CENTER_X0},y1={CENTER_Y1},x1={CENTER_X1}]")
    print(f"  CENTER_REEXTRACT_SCORE   = {CENTER_REEXTRACT_SCORE}")
    print(f"  RECORDED_SCORE_REFERENCE = {RECORDED_SCORE_REFERENCE}  (참고 parity, CSV 수정 금지)")
    print(f"  ROI_COVERAGE             = {ROI_COVERAGE}")
    print()

    print("[3×3 patch grid]")
    print(f"  {'patch_id':8} {'row':4} {'col':4} {'y0':5} {'x0':5} {'y1':5} {'x1':5} {'center':8}")
    for p in PATCH_GRID:
        cmark = "★" if p["is_center"] else " "
        print(f"  {p['patch_id']:8} {p['row']:4} {p['col']:4} "
              f"{p['y0']:5} {p['x0']:5} {p['y1']:5} {p['x1']:5} {cmark:8}")
    print()

    print("[stats / selected_indices 경로 정책]")
    print(f"  STATS_NPZ_PATH            = padim_v2_roi0_0")
    print(f"  SELECTED_INDICES_NPY_PATH = padim_v1")
    print(f"  lower_central_mean key    → {POSITION_BIN}_mean  shape=(100,)")
    print(f"  lower_central_cov key     → {POSITION_BIN}_cov   shape=(100,100)")
    print(f"  selected_feature_indices  → stats 내 + 외부 npy 동일성 확인 (np.array_equal)")
    print(f"  layer/raw index 매핑      → selected_indices_file 사용")
    print()

    print("[contribution formula]")
    print(f"  {CONTRIBUTION_FORMULA_STRING}")
    print(f"  cov_epsilon = {COVARIANCE_EPSILON}")
    print(f"  sum_match_tol = {SUM_MATCH_TOLERANCE}")
    print(f"  center parity:  abs(sqrt_mahalanobis - {CENTER_REEXTRACT_SCORE}) < 1e-3")
    print(f"  recorded parity: abs(sqrt_mahalanobis - {RECORDED_SCORE_REFERENCE}) < 0.5 (참고)")
    print()

    print("[dim91 caveat]")
    print(f"  {DIM91_DOMINANCE_CAVEAT}")
    print()

    print("[patch summary schema]")
    for k in _get_patch_contribution_summary_schema_keys():
        print(f"  - {k}")
    print()

    print("[단계별 계획]")
    steps = [
        ("Step 1",  "guard 확인 (11개 guard 전부 True 확인)"),
        ("Step 2",  f"stats load: np.load(padim_v2_roi0_0/position_bin_stats.npz)"),
        ("Step 3",  f"selected_indices load: np.load(padim_v1/selected_feature_indices.npy)"),
        ("Step 4",  "selected_indices 동일성 검증: np.array_equal(stats 내 vs 외부 npy)"),
        ("Step 5",  f"CT load: np.load(ct_hu.npy, mmap_mode='r') → (Z,512,512) int16"),
        ("Step 6",  f"slice 추출: ct_hu[CT_INDEX_Z]  (CT_INDEX_Z={CT_INDEX_Z}, local_z)"),
        ("Step 7",  "preprocessing: hu_clip → 3ch repeat → ImageNet norm → (3,512,512)"),
        ("Step 8",  "FeatureExtractor(device='cpu') 초기화"),
        ("Step 9",  "9개 patch bbox 전달 → features_448 shape (9, 448)"),
        ("Step 10", "dimension selection: features_100 = features_448[:, selected_indices_file] → (9, 100)"),
        ("Step 11", "patch별 contribution 계산 (9회)"),
        ("Step 12", f"center patch (patch_id=4) score 검증: abs(score - {CENTER_REEXTRACT_SCORE}) < 1e-3"),
        ("Step 13", "dim91 tracking per patch (dim91_contribution, dim91_rank_abs)"),
        ("Step 14", "patch_grid_plan.csv 저장"),
        ("Step 15", "patch_feature_vectors_selected100.npy (9×100) + raw448.npy (9×448) 저장"),
        ("Step 16", "patch_contribution_full.csv (900 rows) 저장"),
        ("Step 17", "patch_contribution_summary.csv (9 rows) 저장"),
        ("Step 18", "patch_contribution_map.json 저장"),
        ("Step 19", "xai_patch_map_bridge.json 저장"),
        ("Step 20", "errors.csv + DONE.json 저장"),
    ]
    for step, desc in steps:
        print(f"  {step}: {desc}")
    print()

    print("[output schema]")
    print(f"  출력 root: {OUTPUT_ROOT}")
    for fname in [
        "patch_grid_plan.csv",
        "patch_feature_vectors_selected100.npy  (shape: 9×100)",
        "patch_feature_vectors_raw448.npy        (shape: 9×448)",
        "patch_contribution_full.csv             (9×100 = 900 rows)",
        "patch_contribution_summary.csv          (9 rows)",
        "patch_contribution_map.json",
        "xai_patch_map_bridge.json",
        "errors.csv",
        "DONE.json",
    ]:
        print(f"  - {fname}")
    print()

    print("[run 승인 요건]")
    for name, note in [
        ("ALLOW_CT_LOAD",            "True (별도 승인)"),
        ("ALLOW_MODEL_FORWARD",      "True (별도 승인)"),
        ("ALLOW_FEATURE_EXTRACTION", "True (별도 승인)"),
        ("ALLOW_FEATURE_WRITE",      "True (별도 승인)"),
        ("ALLOW_STATS_LOAD",         "True (별도 승인)"),
        ("ALLOW_CONTRIBUTION_CALC",  "True (별도 승인)"),
        ("ALLOW_CONTRIBUTION_WRITE", "True (별도 승인)"),
        ("ALLOW_MAP_JSON_WRITE",     "True (별도 승인)"),
        ("ALLOW_GPU",                "False (CPU only 유지)"),
        ("ALLOW_STAGE2_HOLDOUT",     "False (유지)"),
        ("ALLOW_FULL_300",           "False (유지)"),
    ]:
        print(f"  {name} = {note}")
    print()
    print("  + --run-smoke --confirm-map 플래그 필요")
    print()

    print("[caveats]")
    print(f"  {DIAGNOSTIC_CAVEAT}")
    print()
    print(f"  dim91: {DIM91_DOMINANCE_CAVEAT}")
    return 0


# ============================================================
# O. run_smoke (현재 guard False → BLOCKED)
# ============================================================

def run_smoke(confirm_map: bool) -> int:
    """실제 9-patch feature extraction + contribution map smoke run. guard True 시에만 실행."""
    if not confirm_map:
        print("BLOCKED: --run-smoke 단독 실행 금지. --confirm-map 필요.", file=sys.stderr)
        return 2

    if not ALLOW_CT_LOAD:
        print("BLOCKED: ALLOW_CT_LOAD=False.", file=sys.stderr)
        return 2
    if not ALLOW_MODEL_FORWARD:
        print("BLOCKED: ALLOW_MODEL_FORWARD=False.", file=sys.stderr)
        return 2
    if not ALLOW_FEATURE_EXTRACTION:
        print("BLOCKED: ALLOW_FEATURE_EXTRACTION=False.", file=sys.stderr)
        return 2
    if not ALLOW_FEATURE_WRITE:
        print("BLOCKED: ALLOW_FEATURE_WRITE=False.", file=sys.stderr)
        return 2
    if not ALLOW_STATS_LOAD:
        print("BLOCKED: ALLOW_STATS_LOAD=False.", file=sys.stderr)
        return 2
    if not ALLOW_CONTRIBUTION_CALC:
        print("BLOCKED: ALLOW_CONTRIBUTION_CALC=False.", file=sys.stderr)
        return 2
    if not ALLOW_CONTRIBUTION_WRITE:
        print("BLOCKED: ALLOW_CONTRIBUTION_WRITE=False.", file=sys.stderr)
        return 2
    if not ALLOW_MAP_JSON_WRITE:
        print("BLOCKED: ALLOW_MAP_JSON_WRITE=False.", file=sys.stderr)
        return 2
    if ALLOW_GPU:
        print("BLOCKED: ALLOW_GPU=True는 CPU-only 정책 위반.", file=sys.stderr)
        return 2
    if ALLOW_STAGE2_HOLDOUT:
        print("BLOCKED: ALLOW_STAGE2_HOLDOUT=True는 허용되지 않습니다.", file=sys.stderr)
        return 2
    if ALLOW_FULL_300:
        print("BLOCKED: ALLOW_FULL_300=True는 허용되지 않습니다.", file=sys.stderr)
        return 2

    # DONE.json 충돌 확인
    done_path = os.path.join(OUTPUT_ROOT, "DONE.json")
    if os.path.isfile(done_path):
        print(f"BLOCKED: DONE.json already exists: {done_path}", file=sys.stderr)
        return 2

    # ----------------------------------------------------------------
    # guard 통과 후 실제 실행 (현재 모든 guard=False이므로 도달 불가)
    # ----------------------------------------------------------------
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
    import numpy as np
    import json
    import csv
    from position_aware_padim.preprocessing import preprocess_ct_slice
    from position_aware_padim.feature_extractor import FeatureExtractor

    # local_z guard (run path)
    assert CT_INDEX_Z == 51
    assert REPORT_SLICE_INDEX == 106
    assert CT_INDEX_Z != REPORT_SLICE_INDEX
    assert LOCAL_Z_USED_FOR_CT_INDEXING is True
    assert SLICE_INDEX_USED_FOR_CT_INDEXING is False

    # stats load
    stats = np.load(STATS_NPZ_PATH, allow_pickle=True)
    mean_vec = stats[f"{POSITION_BIN}_mean"].astype(np.float64)   # lower_central_mean shape=(100,)
    cov_mat  = stats[f"{POSITION_BIN}_cov"].astype(np.float64)    # lower_central_cov shape=(100,100)
    stats_sel_indices = stats["selected_feature_indices"].astype(np.int64)

    # selected_indices load (padim_v1)
    selected_indices_file = np.load(SELECTED_INDICES_NPY_PATH).astype(np.int64)

    # selected_indices 동일성 검증 (stats 내 vs 외부 npy)
    if not np.array_equal(stats_sel_indices, selected_indices_file):
        print("BLOCKED: selected_indices mismatch — stats vs padim_v1 npy 불일치", file=sys.stderr)
        return 2

    # layer/raw index 매핑에는 selected_indices_file 사용
    selected_indices = selected_indices_file
    raw_indices = selected_indices.tolist()
    layer_labels = [_layer_label(ri) for ri in raw_indices]

    # cov_inv 계산
    cov_reg = cov_mat + COVARIANCE_EPSILON * np.eye(SELECTED_FEATURE_DIM)
    cov_inv = np.linalg.inv(cov_reg)

    # CT load (mmap_mode="r" 필수 — 일반 np.load 금지)
    assert _STAGE2_HOLDOUT_SENTINEL not in CT_HU_NPY_PATH, "stage2_holdout path 혼입 위험"
    ct_hu = np.load(CT_HU_NPY_PATH, mmap_mode="r")
    assert ct_hu.ndim == 3, f"CT ndim mismatch: {ct_hu.ndim}"
    assert 0 <= CT_INDEX_Z < ct_hu.shape[0], \
        f"local_z={CT_INDEX_Z} out of [0,{ct_hu.shape[0]})"

    # slice 추출: CT_INDEX_Z=51 사용 (report_slice=106은 CT index로 절대 사용 금지)
    slice_2d = np.asarray(ct_hu[CT_INDEX_Z], dtype=np.float32)
    del ct_hu

    # preprocessing
    preprocessed = preprocess_ct_slice(slice_2d, hu_min=HU_MIN, hu_max=HU_MAX)

    # feature extraction (9 patches) — CPU only, EfficientNet 혼입 금지
    device = "cpu"  # ALLOW_GPU=False 강제
    fe = FeatureExtractor(device=device)
    patch_bboxes = [(p["y0"], p["x0"], p["y1"], p["x1"]) for p in PATCH_GRID]
    features_448_all = fe.extract_patch_features(preprocessed, patch_bboxes)
    # features_448_all: shape (9, 448)

    features_100_all = features_448_all[:, selected_indices].astype(np.float64)
    # features_100_all: shape (9, 100)

    # output
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # patch_grid_plan.csv
    plan_path = os.path.join(OUTPUT_ROOT, "patch_grid_plan.csv")
    with open(plan_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "patch_id", "row", "col", "y0", "x0", "y1", "x1", "patch_size",
            "is_center", "position_bin", "ct_index_z", "report_slice_index",
            "local_z_used_for_ct_indexing", "slice_index_used_for_ct_indexing",
        ])
        writer.writeheader()
        for p in PATCH_GRID:
            writer.writerow({
                "patch_id": p["patch_id"], "row": p["row"], "col": p["col"],
                "y0": p["y0"], "x0": p["x0"], "y1": p["y1"], "x1": p["x1"],
                "patch_size": PATCH_SIZE,
                "is_center": p["is_center"],
                "position_bin": POSITION_BIN,
                "ct_index_z": CT_INDEX_Z,
                "report_slice_index": REPORT_SLICE_INDEX,
                "local_z_used_for_ct_indexing": True,
                "slice_index_used_for_ct_indexing": False,
            })

    # feature vectors
    np.save(os.path.join(OUTPUT_ROOT, "patch_feature_vectors_selected100.npy"),
            features_100_all.astype(np.float32))
    np.save(os.path.join(OUTPUT_ROOT, "patch_feature_vectors_raw448.npy"),
            features_448_all.astype(np.float32))

    # contribution 계산 (9 patches)
    summary_rows = []
    full_rows    = []
    errors_list  = []

    for p_idx, p in enumerate(PATCH_GRID):
        pid = p["patch_id"]
        x = features_100_all[p_idx]   # (100,)
        try:
            diff    = x - mean_vec
            v       = cov_inv @ diff
            contrib = diff * v
            d2      = float(diff @ cov_inv @ diff)
            sum_c   = float(contrib.sum())
            sum_err = abs(sum_c - d2)
            sqrt_mah = float(d2 ** 0.5) if d2 >= 0 else 0.0

            abs_order = sorted(range(100), key=lambda i: -abs(contrib[i]))
            sum_abs   = sum(abs(contrib[i]) for i in range(100))
            top2_sum  = float(abs(contrib[abs_order[0]]) + abs(contrib[abs_order[1]]))
            top5_sum  = float(sum(abs(contrib[abs_order[i]]) for i in range(5)))
            top10_sum = float(sum(abs(contrib[abs_order[i]]) for i in range(10)))
            top5_ratio  = top5_sum  / d2 if d2 > 0 else 0.0
            top10_ratio = top10_sum / d2 if d2 > 0 else 0.0

            def _lf(lname):
                idxs = [i for i in range(100) if layer_labels[i] == lname]
                return sum(abs(contrib[i]) for i in idxs) / sum_abs if sum_abs > 0 else 0.0

            n_pos = int((contrib > 0).sum())
            n_neg = int((contrib < 0).sum())

            # dim91 tracking
            dim91_contrib   = float(contrib[DIM91_SELECTED_INDEX])
            dim91_rank_abs  = int(abs_order.index(DIM91_SELECTED_INDEX))

            # center patch parity 검증
            if p["is_center"]:
                center_err = abs(sqrt_mah - CENTER_REEXTRACT_SCORE)
                if center_err >= 1e-3:
                    errors_list.append({
                        "case_id": CASE_ID, "patch_id": pid,
                        "error_type": "center_score_mismatch",
                        "message": (
                            f"abs({sqrt_mah:.6f} - {CENTER_REEXTRACT_SCORE}) "
                            f"= {center_err:.6f} >= 1e-3"
                        ),
                    })

            summary_rows.append({
                "patch_id":                  pid,
                "grid_row":                  p["row"],
                "grid_col":                  p["col"],
                "y0": p["y0"], "x0": p["x0"], "y1": p["y1"], "x1": p["x1"],
                "is_center":                 p["is_center"],
                "sqrt_mahalanobis":          sqrt_mah,
                "d2":                        d2,
                "sum_contribution":          sum_c,
                "sum_match_error":           sum_err,
                "sum_match_pass":            sum_err < SUM_MATCH_TOLERANCE,
                "top1_abs_selected_dim":     abs_order[0],
                "top1_abs_raw_dim":          raw_indices[abs_order[0]],
                "top1_abs_layer":            layer_labels[abs_order[0]],
                "top1_abs_contribution":     float(abs(contrib[abs_order[0]])),
                "top2_abs_contribution_sum": top2_sum,
                "top5_abs_contribution_sum": top5_sum,
                "top5_concentration_ratio":  top5_ratio,
                "top10_concentration_ratio": top10_ratio,
                "layer1_fraction_abs":       _lf("layer1"),
                "layer2_fraction_abs":       _lf("layer2"),
                "layer3_fraction_abs":       _lf("layer3"),
                "dim91_contribution":        dim91_contrib,
                "dim91_rank_abs":            dim91_rank_abs,
                "n_positive_contrib":        n_pos,
                "n_negative_contrib":        n_neg,
            })

            # full rows (100 dims per patch)
            for i in range(100):
                full_rows.append({
                    "patch_id":           pid,
                    "is_center":          p["is_center"],
                    "selected_dim_index": i,
                    "raw_feature_index":  raw_indices[i],
                    "layer":              layer_labels[i],
                    "diff":               float(diff[i]),
                    "cov_inv_dot_diff":   float(v[i]),
                    "contribution":       float(contrib[i]),
                    "abs_contribution":   float(abs(contrib[i])),
                    "contribution_sign":  "positive" if contrib[i] >= 0 else "negative",
                    "position_bin":       POSITION_BIN,
                    "case_id":            CASE_ID,
                })

        except Exception as e:
            errors_list.append({
                "case_id": CASE_ID, "patch_id": pid,
                "error_type": "contribution_error", "message": str(e),
            })

    # patch_contribution_summary.csv (9 rows)
    summary_fieldnames = _get_patch_contribution_summary_schema_keys()
    _write_csv(os.path.join(OUTPUT_ROOT, "patch_contribution_summary.csv"),
               summary_rows, summary_fieldnames)

    # patch_contribution_full.csv (900 rows)
    full_fieldnames = [
        "patch_id", "is_center", "selected_dim_index", "raw_feature_index", "layer",
        "diff", "cov_inv_dot_diff", "contribution", "abs_contribution",
        "contribution_sign", "position_bin", "case_id",
    ]
    _write_csv(os.path.join(OUTPUT_ROOT, "patch_contribution_full.csv"),
               full_rows, full_fieldnames)

    # patch_contribution_map.json
    score_map       = [[None]*GRID_SIZE for _ in range(GRID_SIZE)]
    top5_conc_map   = [[None]*GRID_SIZE for _ in range(GRID_SIZE)]
    layer3_frac_map = [[None]*GRID_SIZE for _ in range(GRID_SIZE)]
    dim91_contr_map = [[None]*GRID_SIZE for _ in range(GRID_SIZE)]
    max_pid, max_score, center_score_val = -1, -1.0, None

    for r in summary_rows:
        rr, rc = r["grid_row"], r["grid_col"]
        s = round(r["sqrt_mahalanobis"], 6)
        score_map[rr][rc]       = s
        top5_conc_map[rr][rc]   = round(r["top5_concentration_ratio"], 6)
        layer3_frac_map[rr][rc] = round(r["layer3_fraction_abs"], 6)
        dim91_contr_map[rr][rc] = round(r["dim91_contribution"], 6)
        if s > max_score:
            max_score = s
            max_pid   = r["patch_id"]
        if r["is_center"]:
            center_score_val = s

    max_to_center_ratio = (
        max_score / center_score_val
        if center_score_val and center_score_val > 0 else None
    )

    map_json = {
        "case_id":                        CASE_ID,
        "position_bin":                   POSITION_BIN,
        "ct_index_z":                     CT_INDEX_Z,
        "report_slice_index":             REPORT_SLICE_INDEX,
        "local_z_used_for_ct_indexing":   True,
        "slice_index_used_for_ct_indexing": False,
        "grid_size":                      GRID_SIZE,
        "n_patches":                      len(PATCH_GRID),
        "center_patch_id":                CENTER_PATCH_ID,
        "score_map_sqrt_mahalanobis":     score_map,
        "top5_concentration_map":         top5_conc_map,
        "layer3_fraction_map":            layer3_frac_map,
        "dim91_contribution_map":         dim91_contr_map,
        "center_score":                   center_score_val,
        "max_patch_id":                   max_pid,
        "max_patch_score":                max_score,
        "max_to_center_ratio":            max_to_center_ratio,
        "not_gradcam":                    True,
        "not_pixel_attribution":          True,
        "not_diagnostic":                 True,
        "caveat":                         DIAGNOSTIC_CAVEAT,
        "dim91_dominance_caveat":         DIM91_DOMINANCE_CAVEAT,
        "stage2_holdout_accessed":        False,
        "ct_load_occurred":               True,
        "model_forward_occurred":         True,
        "feature_extraction_occurred":    True,
        "existing_artifacts_modified":    False,
    }
    with open(os.path.join(OUTPUT_ROOT, "patch_contribution_map.json"), "w") as f:
        json.dump(map_json, f, indent=2, ensure_ascii=False)

    # xai_patch_map_bridge.json
    bridge = {
        "s5_case_id":              CASE_ID,
        "contribution_method":     CONTRIBUTION_METHOD,
        "grid_size":               GRID_SIZE,
        "n_patches":               len(PATCH_GRID),
        "center_patch_id":         CENTER_PATCH_ID,
        "map_type":                "PaDiM patch-level contribution map",
        "score_map_sqrt_mahalanobis": score_map,
        "caveat":                  DIAGNOSTIC_CAVEAT,
        "not_diagnostic":          True,
        "not_gradcam":             True,
        "not_pixel_attribution":   True,
        "not_spatial_heatmap_yet": True,
        "dim91_dominance_caveat":  DIM91_DOMINANCE_CAVEAT,
        "note": (
            "feature dimension은 사람이 직접 이해 가능한 CT 구조가 아님. "
            "이 map은 Mahalanobis distance 기반 feature-space contribution의 "
            "patch-level 분포를 보여주는 탐색적 결과임."
        ),
    }
    with open(os.path.join(OUTPUT_ROOT, "xai_patch_map_bridge.json"), "w") as f:
        json.dump(bridge, f, indent=2, ensure_ascii=False)

    # errors.csv
    errors_path = os.path.join(OUTPUT_ROOT, "errors.csv")
    with open(errors_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "patch_id", "error_type", "message"])
        writer.writeheader()
        writer.writerows(errors_list)

    # DONE.json
    center_summary = next((r for r in summary_rows if r["is_center"]), None)
    with open(done_path, "w") as f:
        json.dump({
            "status":                  "DONE",
            "case_id":                 CASE_ID,
            "n_patches":               len(summary_rows),
            "n_errors":                len(errors_list),
            "center_sqrt_mahalanobis": center_summary["sqrt_mahalanobis"] if center_summary else None,
            "center_sum_match_pass":   center_summary["sum_match_pass"] if center_summary else None,
            "center_score_parity_pass": (
                abs(center_summary["sqrt_mahalanobis"] - CENTER_REEXTRACT_SCORE) < 1e-3
                if center_summary else None
            ),
        }, f, indent=2)

    # 결과 출력
    print(f"\n완료: {len(summary_rows)}개 patch contribution 계산")
    if center_summary:
        c = center_summary["sqrt_mahalanobis"]
        print(f"center patch (id={CENTER_PATCH_ID}): score={c:.6f}")
        print(f"  reextract parity: abs({c:.6f} - {CENTER_REEXTRACT_SCORE}) "
              f"= {abs(c - CENTER_REEXTRACT_SCORE):.6f} "
              f"({'PASS' if abs(c - CENTER_REEXTRACT_SCORE) < 1e-3 else 'FAIL'})")
        print(f"  recorded parity:  abs({c:.6f} - {RECORDED_SCORE_REFERENCE}) "
              f"= {abs(c - RECORDED_SCORE_REFERENCE):.6f} "
              f"({'PASS' if abs(c - RECORDED_SCORE_REFERENCE) < 0.5 else 'FAIL'})")
        print(f"  sum_match: {'PASS' if center_summary['sum_match_pass'] else 'FAIL'}")
    if errors_list:
        print(f"오류: {len(errors_list)}건 (errors.csv 참고)")
    print(f"\n[NOTE] {DIAGNOSTIC_CAVEAT}")
    print(f"[DIM91] {DIM91_DOMINANCE_CAVEAT}")
    return 0


def _write_csv(path: str, rows: list, fieldnames: list):
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# P. main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="S5 Patch-level PaDiM Contribution Map — LUNG1-052__c3 Smoke Script"
    )
    parser.add_argument("--selftest",     action="store_true", help="정적 selftest 80+ 항목 실행")
    parser.add_argument("--dry-run",      action="store_true", help="경로 확인 (실행 없음)")
    parser.add_argument("--plan-only",    action="store_true", help="실행 계획 출력")
    parser.add_argument("--run-smoke",    action="store_true", help="실제 smoke run (guard 필요)")
    parser.add_argument("--confirm-map",  action="store_true", help="map smoke 실행 확인 플래그")
    args = parser.parse_args()

    # bare 실행 차단
    if not any([args.selftest, args.dry_run, args.plan_only, args.run_smoke]):
        print("BLOCKED: 실행 모드를 지정하세요.", file=sys.stderr)
        print("  허용: --selftest | --dry-run | --plan-only", file=sys.stderr)
        print("  금지 (guard 필요): --run-smoke --confirm-map", file=sys.stderr)
        sys.exit(2)

    # --run-smoke 단독 차단
    if args.run_smoke and not args.confirm_map:
        print("BLOCKED: --run-smoke 단독 실행 금지. --confirm-map 필요.", file=sys.stderr)
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
        rc = run_smoke(confirm_map=args.confirm_map)
        sys.exit(rc)


if __name__ == "__main__":
    main()
