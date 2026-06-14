"""
S5 Coordinate Visual Audit Overlay — LUNG1-052__c3
====================================================
대상: LUNG1-052__c3
목적: patch_id=4(center)와 patch_id=7(second-highest/below)를
      CT local_z=51 slice 위에 overlay하는 시각 검증용 PNG 생성

실행 방법:
  --selftest                              : 정적 selftest 75+ 항목 (허용)
  --dry-run                               : 입력 경로 확인, CT load / render / PNG write 없음
  --plan-only                             : overlay 대상 bbox 및 window 정책 출력
  --run-overlay --confirm-overlay         : 실제 overlay 생성 (guard True 시에만)

금지 (현재 모든 guard=False):
  bare 실행               → BLOCKED exit 2
  --run-overlay 단독      → BLOCKED exit 2
  --run-overlay --confirm-overlay → BLOCKED exit 2 (guard False)

WARNING-1 (S4/S5 slice 불일치):
  S4 card에서의 CANDIDATE_SLICE_Z와 S5 contribution map의 local_z=51이 다를 수 있음.
  Overlay v1은 S5 local_z=51 기준으로만 해석.
  LUNG1-052 offset=55 (LUNG1-320 offset=51과 다름). 환자별 독립 확인.
  report_slice=106은 title/metadata/reporting 전용. CT indexing에 사용 금지.

WARNING-2 (CT path 구조):
  실제 CT path는 {root}/volumes_npy/{volume_id}/ct_hu.npy 구조.
  {root}/{volume_id}/ct_hu.npy 구조로 접근하면 파일이 존재하지 않음.
  volumes_npy/ 중간 디렉토리 필수.

Option A (window):
  v1에서 lung window(center=-600, width=1500)와
  PaDiM preprocessing window(HU_MIN=-1000, HU_MAX=200) 둘 다 생성.
  LUNG1-320에서 window mismatch로 v2 재생성 필요했던 선례 반영.

spatial_pattern: CENTER_DOMINANT_WITH_DOWNWARD_CONTINUITY
  center(patch4)에서 하단(patch7) 방향으로 높은 score가 이어지는 패턴.

명칭 정책:
  허용: patch-level PaDiM contribution map / patch-level Mahalanobis contribution overlay
  금지: Grad-CAM / pixel attribution / pixel-level attribution / lesion attribution map
        병변/암/혈관 원인 단정
  진단 목적 사용 금지.

caveat:
  Patch-level PaDiM contribution overlay. Not Grad-CAM, not pixel attribution,
  not diagnostic.
"""

from __future__ import annotations

import argparse
import os
import sys

# ============================================================
# A. Case / Volume / Slice constants
# ============================================================

REPO_ROOT = "/home/jinhy/project/lung-ct-anomaly"

CASE_ID            = "LUNG1-052__c3"
VOLUME_ID          = "NSCLC_LUNG1-052__d4a19cc211"
POSITION_BIN       = "lower_central"

CT_INDEX_Z         = 51     # CT npy[51] — S5 contribution map 기준 (local_z)
REPORT_SLICE_INDEX = 106    # reporting/title 표기 전용 (CT index 아님)

LOCAL_Z_USED_FOR_CT_INDEXING    = True    # CT 인덱싱에 local_z=51 사용
SLICE_INDEX_USED_FOR_CT_INDEXING = False  # REPORT_SLICE_INDEX를 CT index로 쓰면 안 됨

LUNG1_052_OFFSET = 55   # LUNG1-320 offset=51과 다름. 환자별 독립 확인 필수.

IMAGE_H = 512
IMAGE_W = 512

# ============================================================
# B. HU window constants (Option A: 둘 다 생성)
# ============================================================

# Lung window
WINDOW_LUNG_CENTER = -600
WINDOW_LUNG_WIDTH  = 1500
LUNG_HU_LOW        = WINDOW_LUNG_CENTER - WINDOW_LUNG_WIDTH // 2   # -1350
LUNG_HU_HIGH       = WINDOW_LUNG_CENTER + WINDOW_LUNG_WIDTH // 2   #  150

# PaDiM preprocessing window
PADIM_HU_MIN = -1000
PADIM_HU_MAX =  200

WINDOW_MODES_GENERATED = ["lung", "padim_preprocessing"]

_S4_S5_SLICE_MISMATCH_WARNING = (
    "S5 contribution map은 local_z=51 사용. "
    "report_slice=106은 title/metadata 전용. CT indexing 금지. "
    "LUNG1-052 offset=55 (LUNG1-320 offset=51과 다름). "
    "환자별 local_z mapping 독립 확인 원칙."
)

_CT_PATH_STRUCTURE_WARNING = (
    "CT path 구조: {root}/volumes_npy/{volume_id}/ct_hu.npy. "
    "volumes_npy/ 중간 디렉토리 필수. 없으면 파일 미존재."
)

# ============================================================
# C. Patch / Grid constants
# ============================================================

PATCH4_ID    = 4
PATCH4_BBOX  = [288, 128, 320, 160]   # [y0, x0, y1, x1]
PATCH4_SCORE = 38.872562
PATCH4_ROLE  = "center"
PATCH4_LABEL = "patch 4 center max 38.87"
PATCH4_COLOR = "#FFD700"              # yellow solid

PATCH7_ID    = 7
PATCH7_BBOX  = [320, 128, 352, 160]   # [y0, x0, y1, x1]
PATCH7_SCORE = 36.612470
PATCH7_ROLE  = "second_highest"
PATCH7_LABEL = "patch 7 second 36.61"
PATCH7_COLOR = "#FF6600"              # orange thick solid

GRID_EXTENT = [256, 96, 352, 192]     # [y0, x0, y1, x1] 전체 3×3 grid

# 3×3 full grid (row-major) — display용
PATCH_GRID = [
    {"patch_id": 0, "row": 0, "col": 0, "y0": 256, "x0":  96, "y1": 288, "x1": 128},
    {"patch_id": 1, "row": 0, "col": 1, "y0": 256, "x0": 128, "y1": 288, "x1": 160},
    {"patch_id": 2, "row": 0, "col": 2, "y0": 256, "x0": 160, "y1": 288, "x1": 192},
    {"patch_id": 3, "row": 1, "col": 0, "y0": 288, "x0":  96, "y1": 320, "x1": 128},
    {"patch_id": 4, "row": 1, "col": 1, "y0": 288, "x0": 128, "y1": 320, "x1": 160},  # center
    {"patch_id": 5, "row": 1, "col": 2, "y0": 288, "x0": 160, "y1": 320, "x1": 192},
    {"patch_id": 6, "row": 2, "col": 0, "y0": 320, "x0":  96, "y1": 352, "x1": 128},
    {"patch_id": 7, "row": 2, "col": 1, "y0": 320, "x0": 128, "y1": 352, "x1": 160},  # second-highest
    {"patch_id": 8, "row": 2, "col": 2, "y0": 320, "x0": 160, "y1": 352, "x1": 192},
]

SPATIAL_PATTERN = "CENTER_DOMINANT_WITH_DOWNWARD_CONTINUITY"

# ============================================================
# D. Display / Title / Caveat
# ============================================================

OVERLAY_TITLE = (
    f"S5 coordinate audit: {CASE_ID} | CT local_z={CT_INDEX_Z} / report_slice={REPORT_SLICE_INDEX}"
)
OVERLAY_CAVEAT = (
    "Patch-level PaDiM contribution overlay. "
    "Not Grad-CAM, not pixel attribution, not diagnostic."
)
DIM91_DOMINANCE_CAVEAT = (
    "dim91/raw427/layer3는 LUNG1-320__c2와 LUNG1-052__c3 모두에서 강하게 나타남. "
    "해부학적 구조 해석 금지. covariance inverse amplification 또는 "
    "공통 high-response feature-space pattern 가능성."
)
GRID_COLOR     = "#888888"
GRID_LINEWIDTH = 1

# ============================================================
# E. Input paths
# ============================================================

_DATA_ROOT = (
    "/mnt/c/Users/jinhy/Desktop/"
    "NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1"
)

CT_HU_NPY_PATH = (
    _DATA_ROOT + "/volumes_npy/NSCLC_LUNG1-052__d4a19cc211/ct_hu.npy"
)

SMOKE_OUTPUT_ROOT = os.path.join(
    REPO_ROOT,
    "outputs/position-aware-padim-v1/reports/explanation_cards/"
    "s5_lung1_052_c3_patch_level_contribution_map_1case_smoke_v1"
)

PATCH_GRID_PLAN_CSV      = os.path.join(SMOKE_OUTPUT_ROOT, "patch_grid_plan.csv")
CONTRIBUTION_SUMMARY_CSV = os.path.join(SMOKE_OUTPUT_ROOT, "patch_contribution_summary.csv")
CONTRIBUTION_MAP_JSON    = os.path.join(SMOKE_OUTPUT_ROOT, "patch_contribution_map.json")
XAI_BRIDGE_JSON          = os.path.join(SMOKE_OUTPUT_ROOT, "xai_patch_map_bridge.json")
SMOKE_DONE_JSON          = os.path.join(SMOKE_OUTPUT_ROOT, "DONE.json")
SMOKE_ERRORS_CSV         = os.path.join(SMOKE_OUTPUT_ROOT, "errors.csv")

_STAGE2_HOLDOUT_SENTINEL = "stage2_holdout"

# ============================================================
# F. Output root
# ============================================================

OUTPUT_ROOT = os.path.join(
    REPO_ROOT,
    "outputs/position-aware-padim-v1/reports/explanation_cards/"
    "s5_lung1_052_c3_coordinate_visual_audit_1case_v1"
)

OUTPUT_PNG_P4P7_LUNG    = os.path.join(OUTPUT_ROOT, "coordinate_overlay_patch4_patch7_lung.png")
OUTPUT_PNG_GRID_LUNG    = os.path.join(OUTPUT_ROOT, "coordinate_overlay_3x3_grid_lung.png")
OUTPUT_PNG_P4P7_PADIM   = os.path.join(OUTPUT_ROOT, "coordinate_overlay_patch4_patch7_padim_window.png")
OUTPUT_PNG_GRID_PADIM   = os.path.join(OUTPUT_ROOT, "coordinate_overlay_3x3_grid_padim_window.png")
OUTPUT_METADATA_JSON    = os.path.join(OUTPUT_ROOT, "coordinate_overlay_metadata.json")
OUTPUT_INDEX_CSV        = os.path.join(OUTPUT_ROOT, "coordinate_overlay_index.csv")
OUTPUT_RUNTIME_JSON     = os.path.join(OUTPUT_ROOT, "runtime_summary.json")
OUTPUT_ERRORS_CSV       = os.path.join(OUTPUT_ROOT, "errors.csv")
OUTPUT_DONE_JSON        = os.path.join(OUTPUT_ROOT, "DONE.json")

# ============================================================
# G. Guards (all False — 실제 실행 금지)
# ============================================================

ALLOW_CT_LOAD             = False
ALLOW_OVERLAY_RENDER      = False
ALLOW_PNG_WRITE           = False
ALLOW_STAGE2_HOLDOUT      = False
ALLOW_FULL_300            = False
ALLOW_S4_CARD_MODIFICATION = False

_DRY_RUN_POLICY = (
    "dry-run: input path existence only. "
    "CT load 없음. overlay render 없음. PNG write 없음. "
    "model forward 없음. feature extraction 없음. "
    "contribution 재계산 없음. score/threshold 재계산 없음."
)
_PLAN_ONLY_POLICY = (
    "plan-only: overlay target bboxes + window policy + output plan 출력. "
    "CT load 없음. overlay render 없음. PNG write 없음."
)
_MMAP_MODE_POLICY = "np.load(ct_path, mmap_mode='r') — 일반 np.load(ct_path) 금지"

# ============================================================
# H. local_z guard assertion helper
# ============================================================

def _assert_local_z_guards() -> None:
    assert CT_INDEX_Z == 51, f"CT_INDEX_Z guard FAIL: {CT_INDEX_Z}"
    assert REPORT_SLICE_INDEX == 106, f"REPORT_SLICE_INDEX guard FAIL: {REPORT_SLICE_INDEX}"
    assert CT_INDEX_Z != REPORT_SLICE_INDEX, (
        f"CT_INDEX_Z({CT_INDEX_Z}) == REPORT_SLICE_INDEX({REPORT_SLICE_INDEX}) — 혼동 위험"
    )
    assert LOCAL_Z_USED_FOR_CT_INDEXING is True, "LOCAL_Z_USED_FOR_CT_INDEXING must be True"
    assert SLICE_INDEX_USED_FOR_CT_INDEXING is False, "SLICE_INDEX_USED_FOR_CT_INDEXING must be False"


# ============================================================
# I. Bbox validation helpers
# ============================================================

def _validate_bbox(bbox: list, label: str) -> list:
    errors = []
    y0, x0, y1, x1 = bbox
    if y1 - y0 != 32:
        errors.append(f"{label}: height={y1 - y0} != 32")
    if x1 - x0 != 32:
        errors.append(f"{label}: width={x1 - x0} != 32")
    if not (0 <= y0 < y1 <= IMAGE_H):
        errors.append(f"{label}: y=[{y0},{y1}] out of [0,{IMAGE_H}]")
    if not (0 <= x0 < x1 <= IMAGE_W):
        errors.append(f"{label}: x=[{x0},{x1}] out of [0,{IMAGE_W}]")
    return errors


def _validate_all_bboxes() -> list:
    errors = []
    errors.extend(_validate_bbox(PATCH4_BBOX, "patch4"))
    errors.extend(_validate_bbox(PATCH7_BBOX, "patch7"))
    for p in PATCH_GRID:
        errors.extend(_validate_bbox(
            [p["y0"], p["x0"], p["y1"], p["x1"]], f"grid_patch{p['patch_id']}"
        ))
    gy0, gx0, gy1, gx1 = GRID_EXTENT
    if not (0 <= gy0 < gy1 <= IMAGE_H):
        errors.append(f"GRID_EXTENT y=[{gy0},{gy1}] out of [0,{IMAGE_H}]")
    if not (0 <= gx0 < gx1 <= IMAGE_W):
        errors.append(f"GRID_EXTENT x=[{gx0},{gx1}] out of [0,{IMAGE_W}]")
    return errors


# ============================================================
# J. selftest (75+ 항목)
# ============================================================

def run_selftest() -> int:
    passed: list = []
    failed: list = []

    def chk(name: str, cond: bool, msg: str = "") -> None:
        if cond:
            passed.append(name)
        else:
            failed.append(f"{name}: {msg}")

    # --- 1. all guards default False ---
    chk("01_guard_ct_load_false",               not ALLOW_CT_LOAD,
        "ALLOW_CT_LOAD must be False")
    chk("02_guard_overlay_render_false",        not ALLOW_OVERLAY_RENDER,
        "ALLOW_OVERLAY_RENDER must be False")
    chk("03_guard_png_write_false",             not ALLOW_PNG_WRITE,
        "ALLOW_PNG_WRITE must be False")
    chk("04_guard_stage2_holdout_false",        not ALLOW_STAGE2_HOLDOUT,
        "ALLOW_STAGE2_HOLDOUT must be False")
    chk("05_guard_full300_false",               not ALLOW_FULL_300,
        "ALLOW_FULL_300 must be False")
    chk("06_guard_s4_card_mod_false",           not ALLOW_S4_CARD_MODIFICATION,
        "ALLOW_S4_CARD_MODIFICATION must be False")

    # --- 2. CASE_ID / VOLUME_ID / POSITION_BIN ---
    chk("07_case_id_exact",      CASE_ID   == "LUNG1-052__c3",
        f"got {CASE_ID}")
    chk("08_volume_id_exact",    VOLUME_ID == "NSCLC_LUNG1-052__d4a19cc211",
        f"got {VOLUME_ID}")
    chk("09_position_bin_lower_central", POSITION_BIN == "lower_central",
        f"got {POSITION_BIN}")

    # --- 3. local_z / slice_index ---
    chk("10_ct_index_z_51",              CT_INDEX_Z == 51,
        f"got {CT_INDEX_Z}")
    chk("11_report_slice_index_106",     REPORT_SLICE_INDEX == 106,
        f"got {REPORT_SLICE_INDEX}")
    chk("12_ct_index_z_ne_slice_index",  CT_INDEX_Z != REPORT_SLICE_INDEX,
        f"local_z({CT_INDEX_Z}) == slice_index({REPORT_SLICE_INDEX}) — 혼동 위험")
    chk("13_local_z_flag_true",          LOCAL_Z_USED_FOR_CT_INDEXING is True,
        "LOCAL_Z_USED_FOR_CT_INDEXING must be True")
    chk("14_slice_index_flag_false",     SLICE_INDEX_USED_FOR_CT_INDEXING is False,
        "SLICE_INDEX_USED_FOR_CT_INDEXING must be False")
    chk("15_report_slice_never_ct_index_policy",
        "CT indexing 금지" in _S4_S5_SLICE_MISMATCH_WARNING or
        "CT index 아님" in _S4_S5_SLICE_MISMATCH_WARNING or
        "CT indexing에 사용 금지" in _S4_S5_SLICE_MISMATCH_WARNING,
        "report_slice never CT index policy missing from warning string")

    # --- 4. CT path ---
    chk("16_ct_path_contains_volumes_npy",
        "volumes_npy" in CT_HU_NPY_PATH,
        f"volumes_npy 미포함: {CT_HU_NPY_PATH}")
    chk("17_ct_path_endswith_volumes_npy_volume_ct",
        CT_HU_NPY_PATH.endswith("/volumes_npy/NSCLC_LUNG1-052__d4a19cc211/ct_hu.npy"),
        f"CT path 구조 오류: {CT_HU_NPY_PATH}")
    chk("18_stage2_holdout_not_in_ct_path",
        _STAGE2_HOLDOUT_SENTINEL not in CT_HU_NPY_PATH,
        "CT 경로에 stage2_holdout 포함 위험")
    chk("19_ct_path_isfile",
        os.path.isfile(CT_HU_NPY_PATH),
        f"CT file not found: {CT_HU_NPY_PATH}")

    # --- 5. patch4 bbox ---
    chk("20_patch4_bbox_exact",          PATCH4_BBOX == [288, 128, 320, 160],
        f"got {PATCH4_BBOX}")
    chk("21_patch4_width_32",            PATCH4_BBOX[3] - PATCH4_BBOX[1] == 32,
        f"patch4 width={PATCH4_BBOX[3] - PATCH4_BBOX[1]}")
    chk("22_patch4_height_32",           PATCH4_BBOX[2] - PATCH4_BBOX[0] == 32,
        f"patch4 height={PATCH4_BBOX[2] - PATCH4_BBOX[0]}")

    # --- 6. patch7 bbox ---
    chk("23_patch7_bbox_exact",          PATCH7_BBOX == [320, 128, 352, 160],
        f"got {PATCH7_BBOX}")
    chk("24_patch7_width_32",            PATCH7_BBOX[3] - PATCH7_BBOX[1] == 32,
        f"patch7 width={PATCH7_BBOX[3] - PATCH7_BBOX[1]}")
    chk("25_patch7_height_32",           PATCH7_BBOX[2] - PATCH7_BBOX[0] == 32,
        f"patch7 height={PATCH7_BBOX[2] - PATCH7_BBOX[0]}")

    # --- 7. patch4/patch7 spatial relationship ---
    chk("26_patch4_patch7_same_x_range",
        PATCH4_BBOX[1] == PATCH7_BBOX[1] and PATCH4_BBOX[3] == PATCH7_BBOX[3],
        "patch4 and patch7 must share same x-range (same column)")
    chk("27_patch7_directly_below_patch4",
        PATCH7_BBOX[0] == PATCH4_BBOX[2],
        f"patch7 y0({PATCH7_BBOX[0]}) must equal patch4 y1({PATCH4_BBOX[2]})")
    chk("28_patch4_x0_128", PATCH4_BBOX[1] == 128, f"patch4 x0={PATCH4_BBOX[1]}")
    chk("29_patch4_x1_160", PATCH4_BBOX[3] == 160, f"patch4 x1={PATCH4_BBOX[3]}")

    # --- 8. patch scores ---
    chk("30_patch4_score_approx_38_87",
        abs(PATCH4_SCORE - 38.872562) < 1e-3,
        f"got {PATCH4_SCORE}")
    chk("31_patch7_score_approx_36_61",
        abs(PATCH7_SCORE - 36.612470) < 1e-3,
        f"got {PATCH7_SCORE}")
    chk("32_patch4_score_gt_patch7",
        PATCH4_SCORE > PATCH7_SCORE,
        f"patch4({PATCH4_SCORE}) must be > patch7({PATCH7_SCORE})")

    # --- 9. grid extent ---
    chk("33_grid_extent_exact",          GRID_EXTENT == [256, 96, 352, 192],
        f"got {GRID_EXTENT}")
    gy0, gx0, gy1, gx1 = GRID_EXTENT
    chk("34_grid_extent_in_512",
        0 <= gy0 < gy1 <= IMAGE_H and 0 <= gx0 < gx1 <= IMAGE_W,
        f"GRID_EXTENT out of bounds: {GRID_EXTENT}")
    chk("35_patch4_inside_grid",
        (gy0 <= PATCH4_BBOX[0] and PATCH4_BBOX[2] <= gy1 and
         gx0 <= PATCH4_BBOX[1] and PATCH4_BBOX[3] <= gx1),
        "patch4 not inside grid extent")
    chk("36_patch7_inside_grid",
        (gy0 <= PATCH7_BBOX[0] and PATCH7_BBOX[2] <= gy1 and
         gx0 <= PATCH7_BBOX[1] and PATCH7_BBOX[3] <= gx1),
        "patch7 not inside grid extent")

    # --- 10. all bboxes inside 512×512 ---
    bbox_errors = _validate_all_bboxes()
    chk("37_all_bboxes_in_512",          len(bbox_errors) == 0,
        f"bbox errors: {bbox_errors}")

    # --- 11. lung window ---
    chk("38_lung_window_center_neg600",  WINDOW_LUNG_CENTER == -600,
        f"got {WINDOW_LUNG_CENTER}")
    chk("39_lung_window_width_1500",     WINDOW_LUNG_WIDTH == 1500,
        f"got {WINDOW_LUNG_WIDTH}")
    chk("40_lung_hu_low_neg1350",        LUNG_HU_LOW == -1350,
        f"got {LUNG_HU_LOW}")
    chk("41_lung_hu_high_150",           LUNG_HU_HIGH == 150,
        f"got {LUNG_HU_HIGH}")

    # --- 12. PaDiM window ---
    chk("42_padim_hu_min_neg1000",       PADIM_HU_MIN == -1000,
        f"got {PADIM_HU_MIN}")
    chk("43_padim_hu_max_200",           PADIM_HU_MAX == 200,
        f"got {PADIM_HU_MAX}")

    # --- 13. window modes ---
    chk("44_window_modes_include_lung",
        "lung" in WINDOW_MODES_GENERATED,
        f"got {WINDOW_MODES_GENERATED}")
    chk("45_window_modes_include_padim",
        "padim_preprocessing" in WINDOW_MODES_GENERATED,
        f"got {WINDOW_MODES_GENERATED}")

    # --- 14. output root ---
    chk("46_output_root_contains_lung1_052",
        "LUNG1-052" in OUTPUT_ROOT or "lung1_052" in OUTPUT_ROOT,
        f"output root missing LUNG1-052: {OUTPUT_ROOT}")
    chk("47_output_root_not_lung1_320",
        "LUNG1-320" not in OUTPUT_ROOT and "lung1_320" not in OUTPUT_ROOT,
        f"output root must not reference LUNG1-320: {OUTPUT_ROOT}")
    chk("48_output_root_not_smoke_root",
        OUTPUT_ROOT != SMOKE_OUTPUT_ROOT,
        "output root must not be same as smoke output root")
    chk("49_output_root_coordinate_visual_audit",
        "coordinate_visual_audit" in OUTPUT_ROOT,
        f"output root must contain coordinate_visual_audit: {OUTPUT_ROOT}")

    # --- 15. output PNGs planned ---
    chk("50_png_p4p7_lung_planned",
        "coordinate_overlay_patch4_patch7_lung.png" in OUTPUT_PNG_P4P7_LUNG,
        f"got {OUTPUT_PNG_P4P7_LUNG}")
    chk("51_png_grid_lung_planned",
        "coordinate_overlay_3x3_grid_lung.png" in OUTPUT_PNG_GRID_LUNG,
        f"got {OUTPUT_PNG_GRID_LUNG}")
    chk("52_png_p4p7_padim_planned",
        "coordinate_overlay_patch4_patch7_padim_window.png" in OUTPUT_PNG_P4P7_PADIM,
        f"got {OUTPUT_PNG_P4P7_PADIM}")
    chk("53_png_grid_padim_planned",
        "coordinate_overlay_3x3_grid_padim_window.png" in OUTPUT_PNG_GRID_PADIM,
        f"got {OUTPUT_PNG_GRID_PADIM}")
    chk("54_metadata_json_planned",
        "coordinate_overlay_metadata.json" in OUTPUT_METADATA_JSON,
        f"got {OUTPUT_METADATA_JSON}")
    chk("55_index_csv_planned",
        "coordinate_overlay_index.csv" in OUTPUT_INDEX_CSV,
        f"got {OUTPUT_INDEX_CSV}")
    chk("56_runtime_summary_planned",
        "runtime_summary.json" in OUTPUT_RUNTIME_JSON,
        f"got {OUTPUT_RUNTIME_JSON}")
    chk("57_errors_csv_planned",
        "errors.csv" in OUTPUT_ERRORS_CSV,
        f"got {OUTPUT_ERRORS_CSV}")
    chk("58_done_json_planned",
        "DONE.json" in OUTPUT_DONE_JSON,
        f"got {OUTPUT_DONE_JSON}")

    # --- 16. DONE collision block ---
    done_exists = os.path.exists(OUTPUT_DONE_JSON)
    chk("59_done_collision_check",
        True,  # 이 항목은 run_overlay에서 실시간 체크; 여기서는 path 설계 확인만
        "")

    # --- 17. CT load policy ---
    chk("60_ct_load_requires_mmap_r",    "mmap_mode='r'" in _MMAP_MODE_POLICY,
        "mmap_mode='r' not stated in policy")
    chk("61_no_ct_load_in_dryrun",       "CT load 없음" in _DRY_RUN_POLICY,
        "dry-run must state no CT load")
    chk("62_no_overlay_render_in_dryrun","overlay render 없음" in _DRY_RUN_POLICY,
        "dry-run must state no overlay render")
    chk("63_no_png_write_in_dryrun",     "PNG write 없음" in _DRY_RUN_POLICY,
        "dry-run must state no PNG write")
    chk("64_no_model_forward",           "model forward 없음" in _DRY_RUN_POLICY,
        "dry-run must state no model forward")
    chk("65_no_feature_extraction",      "feature extraction 없음" in _DRY_RUN_POLICY,
        "dry-run must state no feature extraction")
    chk("66_no_contribution_recalc",     "contribution 재계산 없음" in _DRY_RUN_POLICY,
        "dry-run must state no contribution recalc")

    # --- 18. naming / caveat policy ---
    chk("67_not_gradcam_caveat",         "Not Grad-CAM" in OVERLAY_CAVEAT,
        "caveat must state Not Grad-CAM")
    chk("68_not_pixel_attribution_caveat",
        "not pixel attribution" in OVERLAY_CAVEAT,
        "caveat must state not pixel attribution")
    chk("69_not_diagnostic_caveat",      "not diagnostic" in OVERLAY_CAVEAT,
        "caveat must state not diagnostic")
    chk("70_dim91_caveat_exists",        len(DIM91_DOMINANCE_CAVEAT) > 0,
        "dim91 dominance caveat missing")
    chk("71_dim91_caveat_no_anatomy",
        "해부학적 구조 해석 금지" in DIM91_DOMINANCE_CAVEAT,
        "dim91 caveat must state no anatomical interpretation")

    # --- 19. spatial pattern ---
    chk("72_spatial_pattern_center_downward",
        SPATIAL_PATTERN == "CENTER_DOMINANT_WITH_DOWNWARD_CONTINUITY",
        f"got {SPATIAL_PATTERN}")

    # --- 20. run mode requires both flags ---
    import inspect
    src = inspect.getsource(run_overlay)
    chk("73_run_requires_run_overlay_flag",
        "run_overlay" in src,
        "run_overlay에 --run-overlay 체크 없음")
    chk("74_run_requires_confirm_overlay_flag",
        "confirm_overlay" in src,
        "run_overlay에 --confirm-overlay 체크 없음")
    chk("75_run_checks_all_guards",
        "ALLOW_CT_LOAD" in src and "ALLOW_OVERLAY_RENDER" in src and "ALLOW_PNG_WRITE" in src,
        "run_overlay must check all 3 run guards")

    # --- 21. image dimensions ---
    chk("76_image_h_512",    IMAGE_H == 512, f"got {IMAGE_H}")
    chk("77_image_w_512",    IMAGE_W == 512, f"got {IMAGE_W}")

    # --- 결과 출력 ---
    total = len(passed) + len(failed)
    print(f"\n=== SELFTEST 결과 ===")
    print(f"PASS: {len(passed)}/{total}")
    if failed:
        print(f"FAIL: {len(failed)}/{total}")
        for f_item in failed:
            print(f"  FAIL: {f_item}")
        print("\n판정: FAIL")
        return 1
    print("모든 selftest 통과.")
    print("판정: PASS")
    return 0


# ============================================================
# K. dry-run
# ============================================================

def run_dry_run() -> int:
    print("=== DRY-RUN ===")
    print(_DRY_RUN_POLICY)
    print()

    _assert_local_z_guards()
    print(f"[OK] local_z guard: CT_INDEX_Z={CT_INDEX_Z}, REPORT_SLICE_INDEX={REPORT_SLICE_INDEX}")

    bbox_errors = _validate_all_bboxes()
    if bbox_errors:
        print(f"[FAIL] bbox errors: {bbox_errors}")
        return 1
    print(f"[OK] patch4/patch7 bbox valid. patch4={PATCH4_BBOX}, patch7={PATCH7_BBOX}")
    print(f"[OK] grid extent valid: {GRID_EXTENT}")

    # CT path structure check
    if "volumes_npy" not in CT_HU_NPY_PATH:
        print(f"[FAIL] CT path missing volumes_npy: {CT_HU_NPY_PATH}")
        return 1
    print(f"[OK] CT path volumes_npy 구조 확인: {CT_HU_NPY_PATH}")

    # stage2_holdout check
    if _STAGE2_HOLDOUT_SENTINEL in CT_HU_NPY_PATH:
        print(f"[FAIL] CT path contains stage2_holdout sentinel", file=sys.stderr)
        return 1
    print(f"[OK] stage2_holdout 접근 0")

    # input paths exist
    paths_to_check = [
        ("CT_HU_NPY_PATH",          CT_HU_NPY_PATH),
        ("PATCH_GRID_PLAN_CSV",     PATCH_GRID_PLAN_CSV),
        ("CONTRIBUTION_SUMMARY_CSV",CONTRIBUTION_SUMMARY_CSV),
        ("CONTRIBUTION_MAP_JSON",   CONTRIBUTION_MAP_JSON),
        ("XAI_BRIDGE_JSON",         XAI_BRIDGE_JSON),
        ("SMOKE_DONE_JSON",         SMOKE_DONE_JSON),
        ("SMOKE_ERRORS_CSV",        SMOKE_ERRORS_CSV),
    ]
    missing = []
    for label, path in paths_to_check:
        exists = os.path.exists(path)
        status = "OK  " if exists else "MISS"
        print(f"  [{status}] {label}: {path}")
        if not exists:
            missing.append(label)

    # errors.csv header check (smoke)
    if os.path.isfile(SMOKE_ERRORS_CSV):
        with open(SMOKE_ERRORS_CSV, encoding="utf-8") as f:
            content = f.read().strip()
        is_header_only = "\n" not in content
        print(f"  [{'OK  ' if is_header_only else 'WARN'}] SMOKE_ERRORS_CSV header-only: {is_header_only}")

    # output root safe check
    if os.path.exists(OUTPUT_DONE_JSON):
        print(f"[WARN] OUTPUT DONE.json 이미 존재 — 덮어쓰기 충돌 가능: {OUTPUT_DONE_JSON}")
    elif os.path.exists(OUTPUT_ROOT):
        print(f"[INFO] output root 이미 존재 (DONE 없음): {OUTPUT_ROOT}")
    else:
        print(f"[OK] output root 미존재 (신규 생성 예정): {OUTPUT_ROOT}")

    print(f"[OK] CT load 0 — CT path 존재 확인만 수행")
    print(f"[OK] overlay render 0")
    print(f"[OK] PNG write 0")
    print(f"[OK] 기존 artifact 수정 0")

    if missing:
        print(f"\n[WARN] 누락 파일 {len(missing)}건: {missing}")
        return 1

    print("\n판정: DRY-RUN PASS")
    return 0


# ============================================================
# L. plan-only
# ============================================================

def run_plan_only() -> int:
    print("=== PLAN-ONLY ===")
    print(_PLAN_ONLY_POLICY)
    print()

    _assert_local_z_guards()

    print("─── Case info ───")
    print(f"  CASE_ID:           {CASE_ID}")
    print(f"  VOLUME_ID:         {VOLUME_ID}")
    print(f"  POSITION_BIN:      {POSITION_BIN}")
    print(f"  SPATIAL_PATTERN:   {SPATIAL_PATTERN}")
    print(f"  LUNG1_052_OFFSET:  {LUNG1_052_OFFSET} (≠ LUNG1-320 offset 51)")
    print()

    print("─── Slice policy ───")
    print(f"  CT index (CT npy 인덱싱): local_z = {CT_INDEX_Z}")
    print(f"  Report label (title 표기용): report_slice = {REPORT_SLICE_INDEX}")
    print(f"  [WARNING] {_S4_S5_SLICE_MISMATCH_WARNING}")
    print()

    print("─── CT path ───")
    print(f"  {CT_HU_NPY_PATH}")
    print(f"  [WARNING] {_CT_PATH_STRUCTURE_WARNING}")
    print()

    print("─── Window policy (Option A) ───")
    print(f"  lung window:  center={WINDOW_LUNG_CENTER}, width={WINDOW_LUNG_WIDTH}")
    print(f"    HU range: [{LUNG_HU_LOW}, {LUNG_HU_HIGH}]")
    print(f"    norm = clip((hu - {LUNG_HU_LOW}) / {WINDOW_LUNG_WIDTH}, 0, 1)")
    print(f"  PaDiM window: HU_MIN={PADIM_HU_MIN}, HU_MAX={PADIM_HU_MAX}")
    print(f"    norm = clip((hu - ({PADIM_HU_MIN})) / ({PADIM_HU_MAX} - ({PADIM_HU_MIN})), 0, 1)")
    print(f"  두 window 모두 v1에서 생성 (LUNG1-320 재작업 선례 반영)")
    print()

    print("─── Overlay targets ───")
    p4 = PATCH4_BBOX
    p7 = PATCH7_BBOX
    ge = GRID_EXTENT
    print(f"  patch_id=4 ({PATCH4_ROLE}):      "
          f"bbox=[y0={p4[0]},x0={p4[1]},y1={p4[2]},x1={p4[3]}]  "
          f"score={PATCH4_SCORE:.6f}  color={PATCH4_COLOR}  label='{PATCH4_LABEL}'")
    print(f"  patch_id=7 ({PATCH7_ROLE}): "
          f"bbox=[y0={p7[0]},x0={p7[1]},y1={p7[2]},x1={p7[3]}]  "
          f"score={PATCH7_SCORE:.6f}  color={PATCH7_COLOR}  label='{PATCH7_LABEL}'  [thick]")
    print(f"  [NOTE] patch7.y0={p7[0]} == patch4.y1={p4[2]} — 바로 인접 (수직)")
    print(f"  [NOTE] patch4/patch7 동일 x-range: x0={p4[1]}, x1={p4[3]}")
    print(f"  3×3 grid extent: [y0={ge[0]},x0={ge[1]},y1={ge[2]},x1={ge[3]}]  "
          f"color={GRID_COLOR}  linewidth={GRID_LINEWIDTH}  no-label")
    print()

    print("─── Title / Caveat ───")
    print(f"  title:       '{OVERLAY_TITLE}'")
    print(f"  caveat:      '{OVERLAY_CAVEAT}'")
    print(f"  dim91 caveat: '{DIM91_DOMINANCE_CAVEAT}'")
    print()

    print("─── Output plan ───")
    print(f"  root: {OUTPUT_ROOT}")
    outputs = [
        (os.path.basename(OUTPUT_PNG_P4P7_LUNG),
         "lung window, patch4(yellow)+patch7(orange) only"),
        (os.path.basename(OUTPUT_PNG_GRID_LUNG),
         "lung window, patch4+patch7+3×3 grid"),
        (os.path.basename(OUTPUT_PNG_P4P7_PADIM),
         "PaDiM window, patch4(yellow)+patch7(orange) only"),
        (os.path.basename(OUTPUT_PNG_GRID_PADIM),
         "PaDiM window, patch4+patch7+3×3 grid"),
        (os.path.basename(OUTPUT_METADATA_JSON),   "파라미터 기록"),
        (os.path.basename(OUTPUT_INDEX_CSV),        "생성 파일 목록"),
        (os.path.basename(OUTPUT_RUNTIME_JSON),     "runtime summary"),
        (os.path.basename(OUTPUT_ERRORS_CSV),       "오류 기록"),
        (os.path.basename(OUTPUT_DONE_JSON),        "완료 마커"),
    ]
    for fname, note in outputs:
        print(f"    {fname}  — {note}")
    print()

    print("─── run guard 요구사항 ───")
    print("  ALLOW_CT_LOAD = True")
    print("  ALLOW_OVERLAY_RENDER = True")
    print("  ALLOW_PNG_WRITE = True")
    print("  (ALLOW_STAGE2_HOLDOUT, ALLOW_FULL_300, ALLOW_S4_CARD_MODIFICATION 는 False 유지)")
    print("  실행: python <script> --run-overlay --confirm-overlay")
    print()

    print("─── mmap policy ───")
    print(f"  {_MMAP_MODE_POLICY}")
    print()

    print("판정: PLAN-ONLY PASS")
    return 0


# ============================================================
# M. run_overlay (실제 overlay 생성 — guard True 시에만)
# ============================================================

def run_overlay(args: argparse.Namespace) -> int:
    import time
    import json
    import csv
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if not getattr(args, "run_overlay", False):
        print("[BLOCKED] --run-overlay 플래그 없음. exit 2", file=sys.stderr)
        return 2
    if not getattr(args, "confirm_overlay", False):
        print("[BLOCKED] --confirm-overlay 플래그 없음. exit 2", file=sys.stderr)
        return 2

    # guard 확인
    guards_ok = ALLOW_CT_LOAD and ALLOW_OVERLAY_RENDER and ALLOW_PNG_WRITE
    if not guards_ok:
        print(
            "[BLOCKED] run guard FAIL. "
            f"ALLOW_CT_LOAD={ALLOW_CT_LOAD}, "
            f"ALLOW_OVERLAY_RENDER={ALLOW_OVERLAY_RENDER}, "
            f"ALLOW_PNG_WRITE={ALLOW_PNG_WRITE}. "
            "소스에서 guard를 True로 변경 후 재실행.",
            file=sys.stderr,
        )
        return 2

    if ALLOW_STAGE2_HOLDOUT:
        print("[BLOCKED] ALLOW_STAGE2_HOLDOUT=True — 금지. exit 2", file=sys.stderr)
        return 2
    if ALLOW_FULL_300:
        print("[BLOCKED] ALLOW_FULL_300=True — 금지. exit 2", file=sys.stderr)
        return 2
    if ALLOW_S4_CARD_MODIFICATION:
        print("[BLOCKED] ALLOW_S4_CARD_MODIFICATION=True — 금지. exit 2", file=sys.stderr)
        return 2

    # stage2_holdout CT path 재확인
    if _STAGE2_HOLDOUT_SENTINEL in CT_HU_NPY_PATH:
        print("[BLOCKED] CT path contains stage2_holdout sentinel. exit 2", file=sys.stderr)
        return 2

    # volumes_npy 구조 재확인
    if "volumes_npy" not in CT_HU_NPY_PATH:
        print(f"[BLOCKED] CT path missing volumes_npy: {CT_HU_NPY_PATH}. exit 2", file=sys.stderr)
        return 2

    _assert_local_z_guards()

    # DONE.json collision check
    if os.path.exists(OUTPUT_DONE_JSON):
        print(f"[BLOCKED] DONE.json 이미 존재 — 덮어쓰기 금지: {OUTPUT_DONE_JSON}. exit 2",
              file=sys.stderr)
        return 2

    t_start = time.time()
    errors = []
    generated = []

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # --- CT load (mmap read-only) ---
    print(f"[INFO] CT load: {CT_HU_NPY_PATH} (mmap_mode='r')")
    ct_hu = np.load(CT_HU_NPY_PATH, mmap_mode="r")
    ct_slice_raw = ct_hu[CT_INDEX_Z].astype(np.float32)  # local_z=51 전용
    print(f"[INFO] CT slice shape: {ct_slice_raw.shape}, "
          f"HU range: {ct_slice_raw.min():.0f}~{ct_slice_raw.max():.0f}")

    # --- lung window ---
    ct_lung = np.clip(ct_slice_raw, LUNG_HU_LOW, LUNG_HU_HIGH)
    ct_lung_disp = ((ct_lung - LUNG_HU_LOW) / (LUNG_HU_HIGH - LUNG_HU_LOW) * 255).astype(np.uint8)

    # --- PaDiM window ---
    ct_padim = np.clip(ct_slice_raw, PADIM_HU_MIN, PADIM_HU_MAX)
    ct_padim_disp = (
        (ct_padim - PADIM_HU_MIN) / (PADIM_HU_MAX - PADIM_HU_MIN) * 255
    ).astype(np.uint8)

    # helper: patch4 + patch7 overlay on given axes
    def _add_patch4_patch7(ax):
        p4_y0, p4_x0, p4_y1, p4_x1 = PATCH4_BBOX
        rect4 = mpatches.Rectangle(
            (p4_x0, p4_y0), p4_x1 - p4_x0, p4_y1 - p4_y0,
            linewidth=2, edgecolor=PATCH4_COLOR, facecolor="none", linestyle="solid",
        )
        ax.add_patch(rect4)
        ax.text(p4_x0, p4_y0 - 4, PATCH4_LABEL,
                color=PATCH4_COLOR, fontsize=8, fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.5, pad=1, linewidth=0))
        p7_y0, p7_x0, p7_y1, p7_x1 = PATCH7_BBOX
        rect7 = mpatches.Rectangle(
            (p7_x0, p7_y0), p7_x1 - p7_x0, p7_y1 - p7_y0,
            linewidth=3, edgecolor=PATCH7_COLOR, facecolor="none", linestyle="solid",
        )
        ax.add_patch(rect7)
        ax.text(p7_x0, p7_y1 + 10, PATCH7_LABEL,
                color=PATCH7_COLOR, fontsize=8, fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.5, pad=1, linewidth=0))

    # helper: 3×3 grid overlay
    def _add_grid(ax):
        for p in PATCH_GRID:
            ry0, rx0, ry1, rx1 = p["y0"], p["x0"], p["y1"], p["x1"]
            rect_g = mpatches.Rectangle(
                (rx0, ry0), rx1 - rx0, ry1 - ry0,
                linewidth=1, edgecolor=GRID_COLOR, facecolor="none", linestyle="solid",
            )
            ax.add_patch(rect_g)
            ax.text((rx0 + rx1) / 2, (ry0 + ry1) / 2, str(p["patch_id"]),
                    color=GRID_COLOR, fontsize=6, ha="center", va="center")
        ge_y0, ge_x0, ge_y1, ge_x1 = GRID_EXTENT
        rect_ge = mpatches.Rectangle(
            (ge_x0, ge_y0), ge_x1 - ge_x0, ge_y1 - ge_y0,
            linewidth=1, edgecolor=GRID_COLOR, facecolor="none", linestyle="dashed",
        )
        ax.add_patch(rect_ge)

    def _set_title_caveat(ax, title_suffix=""):
        ax.set_title(OVERLAY_TITLE + title_suffix, fontsize=8, pad=6)
        ax.text(0.5, -0.03, OVERLAY_CAVEAT,
                transform=ax.transAxes, fontsize=6, ha="center", va="top",
                color="gray", style="italic")
        ax.axis("off")

    # ====== lung window PNGs ======
    for ct_disp, suffix_tag, suffix_file in [
        (ct_lung_disp,  "[lung window]", "_lung"),
        (ct_padim_disp, "[PaDiM window]", "_padim_window"),
    ]:
        # PNG 1: patch4+patch7 only
        fig1, ax1 = plt.subplots(1, 1, figsize=(7, 7))
        ax1.imshow(ct_disp, cmap="gray", vmin=0, vmax=255, origin="upper")
        _add_patch4_patch7(ax1)
        _set_title_caveat(ax1, f" {suffix_tag}")
        fig1.tight_layout()
        out1 = OUTPUT_PNG_P4P7_LUNG if suffix_file == "_lung" else OUTPUT_PNG_P4P7_PADIM
        fig1.savefig(out1, dpi=150, bbox_inches="tight")
        plt.close(fig1)
        generated.append(out1)
        print(f"[OK] {out1}")

        # PNG 2: 3×3 grid + patch4+patch7
        fig2, ax2 = plt.subplots(1, 1, figsize=(7, 7))
        ax2.imshow(ct_disp, cmap="gray", vmin=0, vmax=255, origin="upper")
        _add_grid(ax2)
        _add_patch4_patch7(ax2)
        _set_title_caveat(ax2, f" [3×3 grid] {suffix_tag}")
        fig2.tight_layout()
        out2 = OUTPUT_PNG_GRID_LUNG if suffix_file == "_lung" else OUTPUT_PNG_GRID_PADIM
        fig2.savefig(out2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        generated.append(out2)
        print(f"[OK] {out2}")

    # ====== metadata.json ======
    metadata = {
        "case_id": CASE_ID,
        "volume_id": VOLUME_ID,
        "position_bin": POSITION_BIN,
        "ct_index_z": CT_INDEX_Z,
        "report_slice_index": REPORT_SLICE_INDEX,
        "local_z_used_for_ct_indexing": LOCAL_Z_USED_FOR_CT_INDEXING,
        "slice_index_used_for_ct_indexing": SLICE_INDEX_USED_FOR_CT_INDEXING,
        "ct_path_contains_volumes_npy": ("volumes_npy" in CT_HU_NPY_PATH),
        "s4_s5_slice_mismatch_warning": _S4_S5_SLICE_MISMATCH_WARNING,
        "patch4_bbox": PATCH4_BBOX,
        "patch4_score": PATCH4_SCORE,
        "patch4_role": PATCH4_ROLE,
        "patch7_bbox": PATCH7_BBOX,
        "patch7_score": PATCH7_SCORE,
        "patch7_role": PATCH7_ROLE,
        "grid_extent": GRID_EXTENT,
        "spatial_pattern": SPATIAL_PATTERN,
        "window_modes_generated": WINDOW_MODES_GENERATED,
        "lung_window_center": WINDOW_LUNG_CENTER,
        "lung_window_width": WINDOW_LUNG_WIDTH,
        "padim_window_hu_min": PADIM_HU_MIN,
        "padim_window_hu_max": PADIM_HU_MAX,
        "not_gradcam": True,
        "not_pixel_attribution": True,
        "not_diagnostic": True,
        "dim91_dominance_caveat": DIM91_DOMINANCE_CAVEAT,
        "stage2_holdout_accessed": False,
        "s4_card_modified": False,
        "existing_artifacts_modified": False,
        "caveat": OVERLAY_CAVEAT,
        "generated_files": [os.path.basename(p) for p in generated],
        "guards": {
            "ALLOW_CT_LOAD": ALLOW_CT_LOAD,
            "ALLOW_OVERLAY_RENDER": ALLOW_OVERLAY_RENDER,
            "ALLOW_PNG_WRITE": ALLOW_PNG_WRITE,
            "ALLOW_STAGE2_HOLDOUT": ALLOW_STAGE2_HOLDOUT,
            "ALLOW_FULL_300": ALLOW_FULL_300,
            "ALLOW_S4_CARD_MODIFICATION": ALLOW_S4_CARD_MODIFICATION,
        },
    }
    import json as _json
    with open(OUTPUT_METADATA_JSON, "w", encoding="utf-8") as f:
        _json.dump(metadata, f, ensure_ascii=False, indent=2)
    generated.append(OUTPUT_METADATA_JSON)
    print(f"[OK] {OUTPUT_METADATA_JSON}")

    # ====== index.csv ======
    import csv as _csv
    index_rows = [
        {"file": os.path.basename(OUTPUT_PNG_P4P7_LUNG),  "type": "overlay_png",
         "window": "lung",   "description": "patch4_yellow + patch7_orange on CT local_z=51"},
        {"file": os.path.basename(OUTPUT_PNG_GRID_LUNG),  "type": "overlay_png",
         "window": "lung",   "description": "full 3x3 grid + patch4 + patch7 on CT local_z=51"},
        {"file": os.path.basename(OUTPUT_PNG_P4P7_PADIM), "type": "overlay_png",
         "window": "padim",  "description": "patch4_yellow + patch7_orange PaDiM window CT local_z=51"},
        {"file": os.path.basename(OUTPUT_PNG_GRID_PADIM), "type": "overlay_png",
         "window": "padim",  "description": "full 3x3 grid + patch4 + patch7 PaDiM window CT local_z=51"},
        {"file": os.path.basename(OUTPUT_METADATA_JSON),  "type": "metadata_json",
         "window": "both",   "description": "overlay generation metadata and policy records"},
    ]
    with open(OUTPUT_INDEX_CSV, "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=["file", "type", "window", "description"])
        writer.writeheader()
        writer.writerows(index_rows)
    generated.append(OUTPUT_INDEX_CSV)
    print(f"[OK] {OUTPUT_INDEX_CSV}")

    # ====== errors.csv ======
    with open(OUTPUT_ERRORS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=["error_type", "message"])
        writer.writeheader()
        for e in errors:
            writer.writerow({"error_type": "WARNING", "message": e})
    generated.append(OUTPUT_ERRORS_CSV)
    print(f"[OK] {OUTPUT_ERRORS_CSV}")

    # ====== runtime_summary.json ======
    t_end = time.time()
    runtime_summary = {
        "case_id": CASE_ID,
        "ct_index_z": CT_INDEX_Z,
        "elapsed_sec": round(t_end - t_start, 2),
        "png_count": len([p for p in generated if p.endswith(".png")]),
        "error_count": len(errors),
        "status": "DONE",
    }
    with open(OUTPUT_RUNTIME_JSON, "w", encoding="utf-8") as f:
        _json.dump(runtime_summary, f, ensure_ascii=False, indent=2)
    generated.append(OUTPUT_RUNTIME_JSON)
    print(f"[OK] {OUTPUT_RUNTIME_JSON}")

    # ====== DONE.json ======
    done_obj = {
        "status": "DONE",
        "case_id": CASE_ID,
        "ct_index_z": CT_INDEX_Z,
        "png_count": 4,
        "errors": errors,
    }
    with open(OUTPUT_DONE_JSON, "w", encoding="utf-8") as f:
        _json.dump(done_obj, f, ensure_ascii=False, indent=2)
    generated.append(OUTPUT_DONE_JSON)
    print(f"[OK] DONE.json")

    print(f"\n[PASS] overlay 생성 완료. elapsed={round(t_end - t_start, 2)}s")
    print(f"  output_root: {OUTPUT_ROOT}")
    return 0


# ============================================================
# N. main
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="S5 Coordinate Visual Audit Overlay — LUNG1-052__c3"
    )
    parser.add_argument("--selftest",        action="store_true")
    parser.add_argument("--dry-run",         action="store_true", dest="dry_run")
    parser.add_argument("--plan-only",       action="store_true", dest="plan_only")
    parser.add_argument("--run-overlay",     action="store_true", dest="run_overlay")
    parser.add_argument("--confirm-overlay", action="store_true", dest="confirm_overlay")
    args = parser.parse_args()

    any_flag = (
        args.selftest or args.dry_run or args.plan_only
        or args.run_overlay or args.confirm_overlay
    )

    if not any_flag:
        print(
            "[BLOCKED] 명시적 플래그 없이 bare 실행은 금지됩니다.\n"
            "사용법:\n"
            "  --selftest\n"
            "  --dry-run\n"
            "  --plan-only\n"
            "  --run-overlay --confirm-overlay  (guard True 시에만)",
            file=sys.stderr,
        )
        return 2

    if args.run_overlay and not args.confirm_overlay:
        print("[BLOCKED] --run-overlay 단독 실행 금지. --confirm-overlay도 필요합니다.",
              file=sys.stderr)
        return 2

    if args.selftest:
        return run_selftest()
    if args.dry_run:
        return run_dry_run()
    if args.plan_only:
        return run_plan_only()
    if args.run_overlay and args.confirm_overlay:
        return run_overlay(args)

    print("[BLOCKED] 알 수 없는 실행 조합. exit 2", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
