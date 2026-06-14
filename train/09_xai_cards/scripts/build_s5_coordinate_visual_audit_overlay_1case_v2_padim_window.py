"""
S5 Coordinate Visual Audit Overlay v2 — PaDiM preprocessing window
====================================================================
대상: LUNG1-320__c2
목적: patch_id=4(center)와 patch_id=5(max/right-adjacent)를
      CT local_z=89 slice 위에 PaDiM preprocessing window(HU_MIN=-1000, HU_MAX=200)
      기준으로 overlay하는 시각 검증용 PNG 생성.
      v1(lung window)과의 window 차이 보완이 목적이며, slice mismatch는 해결하지 않음.

실행 방법:
  --selftest                              : 정적 selftest 52+ 항목 (허용)
  --dry-run                               : 입력 경로 확인, CT load / render / PNG write 없음
  --plan-only                             : overlay 대상 bbox 및 window 정책 출력
  --run-overlay --confirm-overlay         : 실제 overlay 생성 (guard True 시에만)

금지 (현재 모든 guard=False):
  bare 실행               → BLOCKED exit 2
  --run-overlay 단독      → BLOCKED exit 2
  --run-overlay --confirm-overlay → BLOCKED exit 2 (guard False)

WARNING-1 (S4/S5 slice 불일치 — v2에서도 미해결):
  S4 v7_textfix는 CANDIDATE_SLICE_Z=140을 CT index처럼 사용.
  S5 contribution map은 CT local_z=89를 사용.
  v2도 S5 local_z=89 기준으로만 해석.
  S4 PNG와 pixel 직접 비교 금지.

WARNING-2 (HU window 차이 — v2가 보완하는 부분):
  v1 overlay: lung window center=-600, width=1500
  v2 overlay: PaDiM preprocessing window HU_MIN=-1000, HU_MAX=200
  v2는 PaDiM 실제 입력 기준으로 시각 재확인 목적.

명칭 정책:
  허용: patch-level PaDiM contribution map / patch-level Mahalanobis contribution overlay
  금지: Grad-CAM / pixel attribution / pixel-level attribution / lesion attribution map
       heatmap (진단 목적으로 해석 금지) / 병변·암·혈관으로 단정
  진단 목적 사용 금지.

caveat:
  Patch-level PaDiM contribution overlay using PaDiM preprocessing window.
  Not Grad-CAM, not pixel attribution, not diagnostic.
"""

from __future__ import annotations

import argparse
import os
import sys

# ============================================================
# A. Case / Volume / Slice constants
# ============================================================

REPO_ROOT = "/home/jinhy/project/lung-ct-anomaly"

CASE_ID            = "LUNG1-320__c2"
VOLUME_ID          = "NSCLC_LUNG1-320__95de24d86f"

CT_INDEX_Z         = 89     # CT npy[89] — S5 contribution map 기준, local_z 전용
REPORT_SLICE_INDEX = 140    # reporting/title 표기 전용 (CT index 절대 사용 금지)

LOCAL_Z_USED_FOR_CT_INDEXING    = True    # CT 인덱싱에 local_z=89 사용
SLICE_INDEX_USED_FOR_CT_INDEXING = False  # REPORT_SLICE_INDEX를 CT index로 쓰면 안 됨

IMAGE_H = 512
IMAGE_W = 512

# ============================================================
# B. HU window constants — v2: PaDiM preprocessing window
# ============================================================

# v2 overlay: PaDiM preprocessing window
WINDOW_MODE = "padim_preprocessing"
HU_MIN      = -1000
HU_MAX      =  200

# v1 lung window — metadata reference용 기록 (v2에서 실제 display에 사용 안 함)
V1_WINDOW_MODE        = "lung"
V1_LUNG_WINDOW_CENTER = -600
V1_LUNG_WINDOW_WIDTH  = 1500

# Mismatch / slice warning strings (selftest에서 확인)
_S4_S5_SLICE_MISMATCH_WARNING = (
    "S4 v7_textfix CANDIDATE_SLICE_Z=140 vs S5 local_z=89 다른 slice. "
    "v2에서도 미해결. S4 PNG와 pixel 직접 비교 금지."
)
_V2_RESOLVES_ONLY_WINDOW_NOTE = (
    "v2가 해결하는 것: lung window(v1) vs PaDiM preprocessing window(v2) 차이 보완. "
    "v2가 해결하지 못하는 것: S4/S5 slice mismatch(local_z=89 vs report_slice=140), "
    "pixel attribution, Grad-CAM, 병변/암/혈관 원인, 해부학적 원인 확정."
)

# ============================================================
# C. Patch / Grid constants
# ============================================================

PATCH4_ID    = 4
PATCH4_BBOX  = [320, 208, 352, 240]   # [y0, x0, y1, x1]
PATCH4_SCORE = 35.0866261463303
PATCH4_ROLE  = "center"
PATCH4_LABEL = "patch 4 center 35.09"
PATCH4_COLOR = "#FFD700"              # yellow solid

PATCH5_ID    = 5
PATCH5_BBOX  = [320, 240, 352, 272]   # [y0, x0, y1, x1]
PATCH5_SCORE = 51.17
PATCH5_ROLE  = "max_right_adjacent"
PATCH5_LABEL = "patch 5 max 51.17"
PATCH5_COLOR = "#FF4444"              # red thick solid

GRID_EXTENT = [288, 176, 384, 272]    # [y0, x0, y1, x1] 전체 3×3 grid

# 3×3 full grid (row-major) — display용
PATCH_GRID = [
    {"patch_id": 0, "row": 0, "col": 0, "y0": 288, "x0": 176, "y1": 320, "x1": 208},
    {"patch_id": 1, "row": 0, "col": 1, "y0": 288, "x0": 208, "y1": 320, "x1": 240},
    {"patch_id": 2, "row": 0, "col": 2, "y0": 288, "x0": 240, "y1": 320, "x1": 272},
    {"patch_id": 3, "row": 1, "col": 0, "y0": 320, "x0": 176, "y1": 352, "x1": 208},
    {"patch_id": 4, "row": 1, "col": 1, "y0": 320, "x0": 208, "y1": 352, "x1": 240},  # center
    {"patch_id": 5, "row": 1, "col": 2, "y0": 320, "x0": 240, "y1": 352, "x1": 272},  # max
    {"patch_id": 6, "row": 2, "col": 0, "y0": 352, "x0": 176, "y1": 384, "x1": 208},
    {"patch_id": 7, "row": 2, "col": 1, "y0": 352, "x0": 208, "y1": 384, "x1": 240},
    {"patch_id": 8, "row": 2, "col": 2, "y0": 352, "x0": 240, "y1": 384, "x1": 272},
]

# ============================================================
# D. Display / Title / Caveat
# ============================================================

OVERLAY_TITLE = (
    f"S5 coordinate audit v2 PaDiM-window: {CASE_ID} | "
    f"CT local_z={CT_INDEX_Z} / report_slice={REPORT_SLICE_INDEX}"
)
OVERLAY_CAVEAT = (
    "Patch-level PaDiM contribution overlay using PaDiM preprocessing window. "
    "Not Grad-CAM, not pixel attribution, not diagnostic."
)
GRID_COLOR     = "#888888"
GRID_LINEWIDTH = 1

# ============================================================
# E. Input paths
# ============================================================

CT_HU_NPY_PATH = (
    "/mnt/c/Users/jinhy/Desktop/"
    "NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/"
    "volumes_npy/NSCLC_LUNG1-320__95de24d86f/ct_hu.npy"
)

SMOKE_OUTPUT_ROOT = os.path.join(
    REPO_ROOT,
    "outputs/position-aware-padim-v1/reports/explanation_cards/"
    "s5_patch_level_contribution_map_1case_smoke_v1"
)

PATCH_GRID_PLAN_CSV      = os.path.join(SMOKE_OUTPUT_ROOT, "patch_grid_plan.csv")
CONTRIBUTION_SUMMARY_CSV = os.path.join(SMOKE_OUTPUT_ROOT, "patch_contribution_summary.csv")
CONTRIBUTION_MAP_JSON    = os.path.join(SMOKE_OUTPUT_ROOT, "patch_contribution_map.json")
XAI_BRIDGE_JSON          = os.path.join(SMOKE_OUTPUT_ROOT, "xai_patch_map_bridge.json")

PREFLIGHT_JSON = os.path.join(
    REPO_ROOT,
    "reports/explanation_cards/"
    "s5_coordinate_visual_audit_overlay_v2_padim_window_preflight_v1.json"
)

_STAGE2_HOLDOUT_SENTINEL = "stage2_holdout"

# v1 output root — 참조만, 수정/덮어쓰기 절대 금지
V1_OUTPUT_ROOT = os.path.join(
    REPO_ROOT,
    "outputs/position-aware-padim-v1/reports/explanation_cards/"
    "s5_coordinate_visual_audit_1case_v1"
)
V1_LUNG_WINDOW_REFERENCE_PATH = os.path.join(
    V1_OUTPUT_ROOT, "coordinate_overlay_patch4_patch5.png"
)

# ============================================================
# F. Output root (v2 — v1과 완전히 분리)
# ============================================================

OUTPUT_ROOT = os.path.join(
    REPO_ROOT,
    "outputs/position-aware-padim-v1/reports/explanation_cards/"
    "s5_coordinate_visual_audit_1case_v2_padim_window"
)

OUTPUT_PNG_PATCH4_PATCH5 = os.path.join(
    OUTPUT_ROOT, "coordinate_overlay_patch4_patch5_padim_window.png"
)
OUTPUT_PNG_3X3_GRID      = os.path.join(
    OUTPUT_ROOT, "coordinate_overlay_3x3_grid_padim_window.png"
)
OUTPUT_METADATA_JSON     = os.path.join(OUTPUT_ROOT, "coordinate_overlay_metadata.json")
OUTPUT_INDEX_CSV         = os.path.join(OUTPUT_ROOT, "coordinate_overlay_index.csv")
OUTPUT_RUNTIME_SUMMARY   = os.path.join(OUTPUT_ROOT, "runtime_summary.json")
OUTPUT_ERRORS_CSV        = os.path.join(OUTPUT_ROOT, "errors.csv")
OUTPUT_DONE_JSON         = os.path.join(OUTPUT_ROOT, "DONE.json")

# ============================================================
# G. Guards (all False — 실제 실행 금지)
# ============================================================

ALLOW_CT_LOAD              = False
ALLOW_OVERLAY_RENDER       = False
ALLOW_PNG_WRITE            = False
ALLOW_STAGE2_HOLDOUT       = False
ALLOW_FULL_300             = False
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
    assert CT_INDEX_Z == 89, f"CT_INDEX_Z guard FAIL: {CT_INDEX_Z}"
    assert REPORT_SLICE_INDEX == 140, f"REPORT_SLICE_INDEX guard FAIL: {REPORT_SLICE_INDEX}"
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
        errors.append(f"{label}: height={y1-y0} != 32")
    if x1 - x0 != 32:
        errors.append(f"{label}: width={x1-x0} != 32")
    if not (0 <= y0 < y1 <= IMAGE_H):
        errors.append(f"{label}: y=[{y0},{y1}] out of [0,{IMAGE_H}]")
    if not (0 <= x0 < x1 <= IMAGE_W):
        errors.append(f"{label}: x=[{x0},{x1}] out of [0,{IMAGE_W}]")
    return errors


def _validate_all_bboxes() -> list:
    errors = []
    errors.extend(_validate_bbox(PATCH4_BBOX, "patch4"))
    errors.extend(_validate_bbox(PATCH5_BBOX, "patch5"))
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
# J. selftest (52+ 항목)
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
    chk("01_guard_ct_load_false",
        not ALLOW_CT_LOAD, "ALLOW_CT_LOAD must be False")
    chk("02_guard_overlay_render_false",
        not ALLOW_OVERLAY_RENDER, "ALLOW_OVERLAY_RENDER must be False")
    chk("03_guard_png_write_false",
        not ALLOW_PNG_WRITE, "ALLOW_PNG_WRITE must be False")
    chk("04_guard_stage2_holdout_false",
        not ALLOW_STAGE2_HOLDOUT, "ALLOW_STAGE2_HOLDOUT must be False")
    chk("05_guard_full300_false",
        not ALLOW_FULL_300, "ALLOW_FULL_300 must be False")
    chk("06_guard_s4_card_mod_false",
        not ALLOW_S4_CARD_MODIFICATION, "ALLOW_S4_CARD_MODIFICATION must be False")

    # --- 2. CASE_ID / VOLUME_ID ---
    chk("07_case_id_exact",
        CASE_ID == "LUNG1-320__c2", f"got {CASE_ID}")
    chk("08_volume_id_exact",
        VOLUME_ID == "NSCLC_LUNG1-320__95de24d86f", f"got {VOLUME_ID}")

    # --- 3. local_z / slice_index ---
    chk("09_ct_index_z_89",
        CT_INDEX_Z == 89, f"got {CT_INDEX_Z}")
    chk("10_report_slice_index_140",
        REPORT_SLICE_INDEX == 140, f"got {REPORT_SLICE_INDEX}")
    chk("11_ct_index_z_ne_slice_index",
        CT_INDEX_Z != REPORT_SLICE_INDEX,
        f"local_z({CT_INDEX_Z}) == slice_index({REPORT_SLICE_INDEX}) — 혼동 위험")
    chk("12_local_z_flag_true",
        LOCAL_Z_USED_FOR_CT_INDEXING is True, "LOCAL_Z_USED_FOR_CT_INDEXING must be True")
    chk("13_slice_index_flag_false",
        SLICE_INDEX_USED_FOR_CT_INDEXING is False, "SLICE_INDEX_USED_FOR_CT_INDEXING must be False")

    # --- 4. patch4 bbox ---
    chk("14_patch4_bbox_exact",
        PATCH4_BBOX == [320, 208, 352, 240], f"got {PATCH4_BBOX}")
    chk("15_patch4_width_32",
        PATCH4_BBOX[3] - PATCH4_BBOX[1] == 32, f"patch4 width={PATCH4_BBOX[3]-PATCH4_BBOX[1]}")
    chk("16_patch4_height_32",
        PATCH4_BBOX[2] - PATCH4_BBOX[0] == 32, f"patch4 height={PATCH4_BBOX[2]-PATCH4_BBOX[0]}")

    # --- 5. patch5 bbox ---
    chk("17_patch5_bbox_exact",
        PATCH5_BBOX == [320, 240, 352, 272], f"got {PATCH5_BBOX}")
    chk("18_patch5_width_32",
        PATCH5_BBOX[3] - PATCH5_BBOX[1] == 32, f"patch5 width={PATCH5_BBOX[3]-PATCH5_BBOX[1]}")
    chk("19_patch5_height_32",
        PATCH5_BBOX[2] - PATCH5_BBOX[0] == 32, f"patch5 height={PATCH5_BBOX[2]-PATCH5_BBOX[0]}")

    # --- 6. patch4/patch5 spatial relationship ---
    chk("20_patch4_patch5_same_y_range",
        PATCH4_BBOX[0] == PATCH5_BBOX[0] and PATCH4_BBOX[2] == PATCH5_BBOX[2],
        "patch4 and patch5 must share same y-range (same row)")
    chk("21_patch5_right_adjacent_to_patch4",
        PATCH5_BBOX[1] == PATCH4_BBOX[3],
        f"patch5 x0({PATCH5_BBOX[1]}) must equal patch4 x1({PATCH4_BBOX[3]})")

    # --- 7. grid extent ---
    chk("22_grid_extent_exact",
        GRID_EXTENT == [288, 176, 384, 272], f"got {GRID_EXTENT}")
    gy0, gx0, gy1, gx1 = GRID_EXTENT
    chk("23_grid_extent_in_512",
        0 <= gy0 < gy1 <= IMAGE_H and 0 <= gx0 < gx1 <= IMAGE_W,
        f"GRID_EXTENT out of bounds: {GRID_EXTENT}")

    # --- 8. all bboxes inside 512×512 ---
    bbox_errors = _validate_all_bboxes()
    chk("24_all_bboxes_in_512",
        len(bbox_errors) == 0, f"bbox errors: {bbox_errors}")

    # --- 9. v2 window: padim_preprocessing ---
    chk("25_window_mode_padim_preprocessing",
        WINDOW_MODE == "padim_preprocessing", f"got {WINDOW_MODE}")
    chk("26_hu_min_neg1000",
        HU_MIN == -1000, f"got {HU_MIN}")
    chk("27_hu_max_200",
        HU_MAX == 200, f"got {HU_MAX}")

    # --- 10. v1 lung window recorded (reference) ---
    chk("28_v1_lung_window_center_neg600_recorded",
        V1_LUNG_WINDOW_CENTER == -600, f"got {V1_LUNG_WINDOW_CENTER}")
    chk("29_v1_lung_window_width_1500_recorded",
        V1_LUNG_WINDOW_WIDTH == 1500, f"got {V1_LUNG_WINDOW_WIDTH}")
    chk("30_v1_window_mode_lung_recorded",
        V1_WINDOW_MODE == "lung", f"got {V1_WINDOW_MODE}")

    # --- 11. padim window != v1 lung window ---
    v1_hu_low  = V1_LUNG_WINDOW_CENTER - V1_LUNG_WINDOW_WIDTH // 2   # -1350
    v1_hu_high = V1_LUNG_WINDOW_CENTER + V1_LUNG_WINDOW_WIDTH // 2   #  150
    chk("31_padim_window_ne_lung_window",
        HU_MIN != v1_hu_low or HU_MAX != v1_hu_high,
        "v2 padim window must differ from v1 lung window")

    # --- 12. S4/S5 slice mismatch warning ---
    chk("32_slice_mismatch_warning_exists",
        len(_S4_S5_SLICE_MISMATCH_WARNING) > 0, "slice mismatch warning string empty")
    chk("33_slice_mismatch_warning_has_140",
        "140" in _S4_S5_SLICE_MISMATCH_WARNING, "slice mismatch warning must mention 140")
    chk("34_slice_mismatch_warning_has_89",
        "89" in _S4_S5_SLICE_MISMATCH_WARNING, "slice mismatch warning must mention 89")
    chk("35_s4_pixel_comparison_forbidden",
        "직접 비교 금지" in _S4_S5_SLICE_MISMATCH_WARNING,
        "S4 pixel direct comparison prohibition missing")
    chk("36_v2_does_not_resolve_slice_mismatch",
        "미해결" in _S4_S5_SLICE_MISMATCH_WARNING,
        "slice mismatch warning must state v2 does not resolve this")

    # --- 13. v2 resolves only window mismatch note ---
    chk("37_v2_resolves_only_window_note_exists",
        len(_V2_RESOLVES_ONLY_WINDOW_NOTE) > 0, "_V2_RESOLVES_ONLY_WINDOW_NOTE empty")
    chk("38_v2_resolves_note_has_window_mismatch",
        "window" in _V2_RESOLVES_ONLY_WINDOW_NOTE.lower(),
        "v2 resolves note must mention window")
    chk("39_v2_resolves_note_states_slice_not_resolved",
        "slice mismatch" in _V2_RESOLVES_ONLY_WINDOW_NOTE.lower(),
        "v2 resolves note must state slice mismatch not resolved")

    # --- 14. mmap_mode policy ---
    chk("40_mmap_mode_policy_exists",
        "mmap_mode" in _MMAP_MODE_POLICY, "mmap_mode policy string missing")
    chk("41_mmap_mode_r_required",
        "mmap_mode='r'" in _MMAP_MODE_POLICY, "mmap_mode='r' not stated in policy")

    # --- 15. dry-run policy strings ---
    chk("42_dryrun_no_ct_load",
        "CT load 없음" in _DRY_RUN_POLICY, "dry-run must state no CT load")
    chk("43_dryrun_no_overlay_render",
        "overlay render 없음" in _DRY_RUN_POLICY, "dry-run must state no overlay render")
    chk("44_dryrun_no_png_write",
        "PNG write 없음" in _DRY_RUN_POLICY, "dry-run must state no PNG write")
    chk("45_dryrun_no_model_forward",
        "model forward 없음" in _DRY_RUN_POLICY, "dry-run must state no model forward")
    chk("46_dryrun_no_feature_extract",
        "feature extraction 없음" in _DRY_RUN_POLICY, "dry-run must state no feature extraction")
    chk("47_dryrun_no_contribution_recalc",
        "contribution 재계산 없음" in _DRY_RUN_POLICY, "dry-run must state no contribution recalc")

    # --- 16. naming/caveat policy ---
    chk("48_not_gradcam_caveat",
        "Not Grad-CAM" in OVERLAY_CAVEAT, "caveat must state Not Grad-CAM")
    chk("49_not_pixel_attribution_caveat",
        "not pixel attribution" in OVERLAY_CAVEAT, "caveat must state not pixel attribution")
    chk("50_not_diagnostic_caveat",
        "not diagnostic" in OVERLAY_CAVEAT, "caveat must state not diagnostic")
    chk("51_caveat_has_padim_window",
        "PaDiM preprocessing window" in OVERLAY_CAVEAT,
        "caveat must state PaDiM preprocessing window")

    # --- 17. title policy ---
    chk("52_title_has_padim_window",
        "PaDiM-window" in OVERLAY_TITLE, f"title missing PaDiM-window: {OVERLAY_TITLE}")
    chk("53_title_has_local_z_89",
        "local_z=89" in OVERLAY_TITLE, f"title must show local_z=89: {OVERLAY_TITLE}")
    chk("54_title_has_report_slice_140",
        "report_slice=140" in OVERLAY_TITLE, f"title must show report_slice=140: {OVERLAY_TITLE}")

    # --- 18. output root: v2, not v1 ---
    chk("55_output_root_v2_padim_window",
        "v2_padim_window" in OUTPUT_ROOT,
        f"output root must be v2_padim_window: {OUTPUT_ROOT}")
    chk("56_output_root_not_v1",
        OUTPUT_ROOT != V1_OUTPUT_ROOT,
        "v2 output root must differ from v1 output root")
    chk("57_v1_output_root_not_overwritten",
        "v1" not in OUTPUT_ROOT or "v2" in OUTPUT_ROOT,
        f"output root must not target v1: {OUTPUT_ROOT}")

    # --- 19. stage2_holdout not in CT path ---
    chk("58_stage2_not_in_ct_path",
        _STAGE2_HOLDOUT_SENTINEL not in CT_HU_NPY_PATH,
        "CT 경로에 stage2_holdout 포함 위험")

    # --- 20. run mode requires both flags (inspect run_overlay source) ---
    import inspect
    src = inspect.getsource(run_overlay)
    chk("59_run_requires_run_overlay_flag",
        "run_overlay" in src, "run_overlay에 --run-overlay 체크 없음")
    chk("60_run_requires_confirm_overlay_flag",
        "confirm_overlay" in src, "run_overlay에 --confirm-overlay 체크 없음")
    chk("61_run_checks_all_guards",
        "ALLOW_CT_LOAD" in src and "ALLOW_OVERLAY_RENDER" in src and "ALLOW_PNG_WRITE" in src,
        "run_overlay must check all 3 run guards")

    # --- 21. image dimensions ---
    chk("62_image_h_512", IMAGE_H == 512, f"got {IMAGE_H}")
    chk("63_image_w_512", IMAGE_W == 512, f"got {IMAGE_W}")

    # --- 22. output filenames have padim_window suffix ---
    chk("64_png_patch4_patch5_has_padim_window",
        "padim_window" in os.path.basename(OUTPUT_PNG_PATCH4_PATCH5),
        f"PNG filename must have padim_window: {os.path.basename(OUTPUT_PNG_PATCH4_PATCH5)}")
    chk("65_png_3x3_grid_has_padim_window",
        "padim_window" in os.path.basename(OUTPUT_PNG_3X3_GRID),
        f"PNG filename must have padim_window: {os.path.basename(OUTPUT_PNG_3X3_GRID)}")

    # --- 23. errors.csv and DONE.json planned ---
    chk("66_errors_csv_planned",
        "errors.csv" in OUTPUT_ERRORS_CSV, f"errors.csv path missing: {OUTPUT_ERRORS_CSV}")
    chk("67_done_json_planned",
        "DONE.json" in OUTPUT_DONE_JSON, f"DONE.json path missing: {OUTPUT_DONE_JSON}")

    # --- 24. no heatmap/lesion wording in caveat ---
    chk("68_no_heatmap_in_caveat",
        "heatmap" not in OVERLAY_CAVEAT.lower(),
        "caveat must not use 'heatmap' as final interpretation")
    chk("69_no_lesion_claim_in_caveat",
        "lesion" not in OVERLAY_CAVEAT.lower() and
        "cancer" not in OVERLAY_CAVEAT.lower() and
        "vessel" not in OVERLAY_CAVEAT.lower(),
        "caveat must not claim lesion/cancer/vessel")

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
    print(f"[OK] patch4/patch5 bbox valid. patch4={PATCH4_BBOX}, patch5={PATCH5_BBOX}")
    print(f"[OK] grid extent valid: {GRID_EXTENT}")

    # input paths exist
    paths_to_check = [
        ("CT_HU_NPY_PATH",           CT_HU_NPY_PATH),
        ("PATCH_GRID_PLAN_CSV",      PATCH_GRID_PLAN_CSV),
        ("CONTRIBUTION_SUMMARY_CSV", CONTRIBUTION_SUMMARY_CSV),
        ("CONTRIBUTION_MAP_JSON",    CONTRIBUTION_MAP_JSON),
        ("XAI_BRIDGE_JSON",          XAI_BRIDGE_JSON),
        ("PREFLIGHT_JSON",           PREFLIGHT_JSON),
        ("V1_OUTPUT_ROOT",           V1_OUTPUT_ROOT),
    ]
    missing = []
    for label, path in paths_to_check:
        exists = os.path.exists(path)
        status = "OK  " if exists else "MISS"
        print(f"  [{status}] {label}: {path}")
        if not exists:
            missing.append(label)

    # CT value load 명시적 0 확인
    print(f"[OK] CT load 0 — CT path 존재 확인만 수행")
    print(f"[OK] stage2_holdout 접근 0")
    print(f"[OK] overlay render 0")
    print(f"[OK] PNG write 0")
    print(f"[OK] 기존 artifact 수정 0")
    print(f"[OK] v1 output root 수정 0")

    # v2 output root safe check
    if os.path.exists(OUTPUT_ROOT):
        print(f"[INFO] v2 output root 이미 존재: {OUTPUT_ROOT}")
    else:
        print(f"[OK] v2 output root 미존재 (신규 생성 예정): {OUTPUT_ROOT}")

    # v2 != v1 output root
    if OUTPUT_ROOT == V1_OUTPUT_ROOT:
        print(f"[FAIL] v2 output root == v1 output root — 덮어쓰기 위험!", file=sys.stderr)
        return 2
    print(f"[OK] v2 output root != v1 output root")

    if missing:
        print(f"\n[WARN] 누락 파일 {len(missing)}건: {missing}")
        # CT path missing은 BLOCK, 나머지는 WARNING
        if "CT_HU_NPY_PATH" in missing:
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

    print("─── Slice policy ───")
    print(f"  CT index (CT npy 인덱싱): local_z = {CT_INDEX_Z}")
    print(f"  Report label (title 표기용 전용): report_slice = {REPORT_SLICE_INDEX}")
    print(f"  [WARNING] {_S4_S5_SLICE_MISMATCH_WARNING}")
    print()

    print("─── HU window policy ───")
    print(f"  v2 (this script, 실제 display): PaDiM preprocessing window")
    print(f"    HU_MIN = {HU_MIN}, HU_MAX = {HU_MAX}")
    print(f"    norm = clip((hu - {HU_MIN}) / ({HU_MAX} - {HU_MIN}), 0, 1)")
    v1_hu_low  = V1_LUNG_WINDOW_CENTER - V1_LUNG_WINDOW_WIDTH // 2
    v1_hu_high = V1_LUNG_WINDOW_CENTER + V1_LUNG_WINDOW_WIDTH // 2
    print(f"  v1 (reference only): lung window center={V1_LUNG_WINDOW_CENTER}, width={V1_LUNG_WINDOW_WIDTH}")
    print(f"    v1 HU range: [{v1_hu_low}, {v1_hu_high}]")
    print(f"  [NOTE] {_V2_RESOLVES_ONLY_WINDOW_NOTE}")
    print()

    print("─── Overlay targets ───")
    p4 = PATCH4_BBOX
    p5 = PATCH5_BBOX
    ge = GRID_EXTENT
    print(f"  patch_id=4 ({PATCH4_ROLE}): bbox=[y0={p4[0]},x0={p4[1]},y1={p4[2]},x1={p4[3]}]"
          f"  score={PATCH4_SCORE:.2f}  color={PATCH4_COLOR}  label='{PATCH4_LABEL}'")
    print(f"  patch_id=5 ({PATCH5_ROLE}): bbox=[y0={p5[0]},x0={p5[1]},y1={p5[2]},x1={p5[3]}]"
          f"  score={PATCH5_SCORE:.2f}  color={PATCH5_COLOR}  label='{PATCH5_LABEL}'  [thick]")
    print(f"  3×3 grid extent: [y0={ge[0]},x0={ge[1]},y1={ge[2]},x1={ge[3]}]"
          f"  color={GRID_COLOR}  linewidth={GRID_LINEWIDTH}  no-label")
    print()

    print("─── Title / Caveat ───")
    print(f"  title:  '{OVERLAY_TITLE}'")
    print(f"  caveat: '{OVERLAY_CAVEAT}'")
    print()

    print("─── Output plan ───")
    print(f"  root: {OUTPUT_ROOT}")
    outputs = [
        ("coordinate_overlay_patch4_patch5_padim_window.png",
         "PaDiM window, patch4(yellow)+patch5(red thick)"),
        ("coordinate_overlay_3x3_grid_padim_window.png",
         "PaDiM window, patch4(yellow)+patch5(red thick)+3×3 grid(gray)"),
        ("coordinate_overlay_metadata.json",  "파라미터 기록 (독립 key 포함)"),
        ("coordinate_overlay_index.csv",      "생성 파일 목록"),
        ("runtime_summary.json",              "실행 시간/결과 기록"),
        ("errors.csv",                        "오류 기록"),
        ("DONE.json",                         "완료 마커"),
    ]
    for fname, note in outputs:
        print(f"    {fname}  — {note}")
    print()

    print("─── 실제 run guard 요구사항 ───")
    print("  ALLOW_CT_LOAD = True")
    print("  ALLOW_OVERLAY_RENDER = True")
    print("  ALLOW_PNG_WRITE = True")
    print("  (ALLOW_STAGE2_HOLDOUT, ALLOW_FULL_300, ALLOW_S4_CARD_MODIFICATION 는 False 유지)")
    print("  실행: python <script> --run-overlay --confirm-overlay")
    print()

    print("─── mmap policy ───")
    print(f"  {_MMAP_MODE_POLICY}")
    print()

    print("─── metadata 독립 key 계획 ───")
    meta_keys = [
        "case_id", "volume_id", "ct_index_z", "report_slice_index",
        "local_z_used_for_ct_indexing", "slice_index_used_for_ct_indexing",
        "s4_s5_slice_mismatch_warning", "window_mode(=padim_preprocessing)",
        "hu_min", "hu_max", "v1_window_mode", "v1_lung_window_center", "v1_lung_window_width",
        "patch4_bbox", "patch5_bbox", "grid_extent",
        "not_gradcam", "not_pixel_attribution", "not_diagnostic",
        "stage2_holdout_accessed", "full_300_applied", "s4_card_modified",
    ]
    for k in meta_keys:
        print(f"    {k}")
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

    # v1 output root 보호
    if OUTPUT_ROOT == V1_OUTPUT_ROOT:
        print("[BLOCKED] v2 output root == v1 output root — 덮어쓰기 위험. exit 2", file=sys.stderr)
        return 2

    _assert_local_z_guards()

    t_start = time.time()
    errors = []
    generated = []

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # --- CT load (mmap read-only, PaDiM preprocessing window 적용) ---
    print(f"[INFO] CT load: {CT_HU_NPY_PATH} (mmap_mode='r')")
    ct_hu = np.load(CT_HU_NPY_PATH, mmap_mode="r")
    ct_slice_raw = ct_hu[CT_INDEX_Z].astype(np.float32)  # local_z=89 전용
    print(f"[INFO] CT slice shape: {ct_slice_raw.shape}, HU range: {ct_slice_raw.min():.0f}~{ct_slice_raw.max():.0f}")

    # --- PaDiM preprocessing window windowing ---
    ct_windowed = np.clip(ct_slice_raw, HU_MIN, HU_MAX)
    ct_display  = ((ct_windowed - HU_MIN) / (HU_MAX - HU_MIN) * 255).astype(np.uint8)

    # ============================================================
    # PNG 1: patch4(yellow) + patch5(red thick) — PaDiM window
    # ============================================================
    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 7))
    ax1.imshow(ct_display, cmap="gray", vmin=0, vmax=255, origin="upper")

    p4_y0, p4_x0, p4_y1, p4_x1 = PATCH4_BBOX
    rect4 = mpatches.Rectangle(
        (p4_x0, p4_y0), p4_x1 - p4_x0, p4_y1 - p4_y0,
        linewidth=2, edgecolor=PATCH4_COLOR, facecolor="none", linestyle="solid",
    )
    ax1.add_patch(rect4)
    ax1.text(p4_x0, p4_y0 - 4, PATCH4_LABEL,
             color=PATCH4_COLOR, fontsize=8, fontweight="bold",
             bbox=dict(facecolor="black", alpha=0.5, pad=1, linewidth=0))

    p5_y0, p5_x0, p5_y1, p5_x1 = PATCH5_BBOX
    rect5 = mpatches.Rectangle(
        (p5_x0, p5_y0), p5_x1 - p5_x0, p5_y1 - p5_y0,
        linewidth=3, edgecolor=PATCH5_COLOR, facecolor="none", linestyle="solid",
    )
    ax1.add_patch(rect5)
    ax1.text(p5_x1 + 2, p5_y0 - 4, PATCH5_LABEL,
             color=PATCH5_COLOR, fontsize=8, fontweight="bold",
             bbox=dict(facecolor="black", alpha=0.5, pad=1, linewidth=0))

    ax1.set_title(OVERLAY_TITLE, fontsize=8, pad=6)
    ax1.text(0.5, -0.03, OVERLAY_CAVEAT,
             transform=ax1.transAxes, fontsize=6, ha="center", va="top",
             color="gray", style="italic")
    ax1.axis("off")
    fig1.tight_layout()
    fig1.savefig(OUTPUT_PNG_PATCH4_PATCH5, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    generated.append(OUTPUT_PNG_PATCH4_PATCH5)
    print(f"[OK] {OUTPUT_PNG_PATCH4_PATCH5}")

    # ============================================================
    # PNG 2: 3×3 grid + patch4/patch5 강조 — PaDiM window
    # ============================================================
    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 7))
    ax2.imshow(ct_display, cmap="gray", vmin=0, vmax=255, origin="upper")

    for p in PATCH_GRID:
        pid = p["patch_id"]
        ry0, rx0, ry1, rx1 = p["y0"], p["x0"], p["y1"], p["x1"]
        rect_g = mpatches.Rectangle(
            (rx0, ry0), rx1 - rx0, ry1 - ry0,
            linewidth=1, edgecolor=GRID_COLOR, facecolor="none", linestyle="solid",
        )
        ax2.add_patch(rect_g)
        ax2.text((rx0 + rx1) / 2, (ry0 + ry1) / 2, str(pid),
                 color=GRID_COLOR, fontsize=6, ha="center", va="center")

    ge_y0, ge_x0, ge_y1, ge_x1 = GRID_EXTENT
    rect_ge = mpatches.Rectangle(
        (ge_x0, ge_y0), ge_x1 - ge_x0, ge_y1 - ge_y0,
        linewidth=1, edgecolor=GRID_COLOR, facecolor="none", linestyle="dashed",
    )
    ax2.add_patch(rect_ge)

    rect4g = mpatches.Rectangle(
        (p4_x0, p4_y0), p4_x1 - p4_x0, p4_y1 - p4_y0,
        linewidth=2, edgecolor=PATCH4_COLOR, facecolor="none", linestyle="solid",
    )
    ax2.add_patch(rect4g)

    rect5g = mpatches.Rectangle(
        (p5_x0, p5_y0), p5_x1 - p5_x0, p5_y1 - p5_y0,
        linewidth=3, edgecolor=PATCH5_COLOR, facecolor="none", linestyle="solid",
    )
    ax2.add_patch(rect5g)

    ax2.set_title(OVERLAY_TITLE + " [3×3 grid]", fontsize=8, pad=6)
    ax2.text(0.5, -0.03, OVERLAY_CAVEAT,
             transform=ax2.transAxes, fontsize=6, ha="center", va="top",
             color="gray", style="italic")
    ax2.axis("off")
    fig2.tight_layout()
    fig2.savefig(OUTPUT_PNG_3X3_GRID, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    generated.append(OUTPUT_PNG_3X3_GRID)
    print(f"[OK] {OUTPUT_PNG_3X3_GRID}")

    # ============================================================
    # metadata.json — 독립 key 전체 포함
    # ============================================================
    metadata = {
        "case_id": CASE_ID,
        "volume_id": VOLUME_ID,
        "ct_index_z": CT_INDEX_Z,
        "report_slice_index": REPORT_SLICE_INDEX,
        "local_z_used_for_ct_indexing": LOCAL_Z_USED_FOR_CT_INDEXING,
        "slice_index_used_for_ct_indexing": SLICE_INDEX_USED_FOR_CT_INDEXING,
        "ct_index_policy": "local_z=89 only — report_slice=140 is title/metadata only",
        "s4_s5_slice_mismatch_warning": _S4_S5_SLICE_MISMATCH_WARNING,
        "window_mode": WINDOW_MODE,
        "hu_min": HU_MIN,
        "hu_max": HU_MAX,
        "v1_window_mode": V1_WINDOW_MODE,
        "v1_lung_window_center": V1_LUNG_WINDOW_CENTER,
        "v1_lung_window_width": V1_LUNG_WINDOW_WIDTH,
        "v1_lung_window_reference_path": V1_LUNG_WINDOW_REFERENCE_PATH,
        "patch4_bbox_y0x0y1x1": PATCH4_BBOX,
        "patch4_score": PATCH4_SCORE,
        "patch4_role": PATCH4_ROLE,
        "patch5_bbox_y0x0y1x1": PATCH5_BBOX,
        "patch5_score": PATCH5_SCORE,
        "patch5_role": PATCH5_ROLE,
        "grid_extent_y0x0y1x1": GRID_EXTENT,
        "not_gradcam": True,
        "not_pixel_attribution": True,
        "not_diagnostic": True,
        "stage2_holdout_accessed": False,
        "full_300_applied": False,
        "s4_card_modified": False,
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
    with open(OUTPUT_METADATA_JSON, "w", encoding="utf-8") as f:
        import json as _json
        _json.dump(metadata, f, ensure_ascii=False, indent=2)
    generated.append(OUTPUT_METADATA_JSON)
    print(f"[OK] {OUTPUT_METADATA_JSON}")

    # ============================================================
    # index.csv
    # ============================================================
    index_rows = [
        {"file": os.path.basename(OUTPUT_PNG_PATCH4_PATCH5), "type": "overlay_png",
         "description": "patch4_yellow + patch5_red_thick on CT local_z=89 (PaDiM window)"},
        {"file": os.path.basename(OUTPUT_PNG_3X3_GRID), "type": "overlay_png",
         "description": "3x3 grid + patch4 + patch5 on CT local_z=89 (PaDiM window)"},
        {"file": os.path.basename(OUTPUT_METADATA_JSON), "type": "metadata_json",
         "description": "overlay generation metadata with independent keys"},
    ]
    with open(OUTPUT_INDEX_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "type", "description"])
        writer.writeheader()
        writer.writerows(index_rows)
    generated.append(OUTPUT_INDEX_CSV)
    print(f"[OK] {OUTPUT_INDEX_CSV}")

    # ============================================================
    # errors.csv
    # ============================================================
    with open(OUTPUT_ERRORS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["error_type", "message"])
        writer.writeheader()
        for e in errors:
            writer.writerow({"error_type": "WARNING", "message": e})
    generated.append(OUTPUT_ERRORS_CSV)

    # ============================================================
    # runtime_summary.json
    # ============================================================
    t_end = time.time()
    runtime_summary = {
        "case_id": CASE_ID,
        "ct_index_z": CT_INDEX_Z,
        "window_mode": WINDOW_MODE,
        "hu_min": HU_MIN,
        "hu_max": HU_MAX,
        "elapsed_sec": round(t_end - t_start, 2),
        "generated_count": len([p for p in generated if p.endswith(".png")]),
        "error_count": len(errors),
        "status": "DONE",
    }
    with open(OUTPUT_RUNTIME_SUMMARY, "w", encoding="utf-8") as f:
        import json as _json
        _json.dump(runtime_summary, f, ensure_ascii=False, indent=2)
    generated.append(OUTPUT_RUNTIME_SUMMARY)
    print(f"[OK] runtime_summary.json")

    # ============================================================
    # DONE.json
    # ============================================================
    done_obj = {
        "status": "DONE",
        "case_id": CASE_ID,
        "ct_index_z": CT_INDEX_Z,
        "window_mode": WINDOW_MODE,
        "png_count": 2,
        "errors": errors,
    }
    with open(OUTPUT_DONE_JSON, "w", encoding="utf-8") as f:
        import json as _json
        _json.dump(done_obj, f, ensure_ascii=False, indent=2)
    print(f"[OK] DONE.json")

    print(f"\n[PASS] overlay 생성 완료. elapsed={round(t_end - t_start, 2)}s")
    print(f"  output_root: {OUTPUT_ROOT}")
    return 0


# ============================================================
# N. main
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="S5 Coordinate Visual Audit Overlay v2 — PaDiM preprocessing window"
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
        print("[BLOCKED] --run-overlay 단독 실행 금지. --confirm-overlay도 필요합니다.", file=sys.stderr)
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
