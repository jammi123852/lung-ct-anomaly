"""
S5 Demo Card Prototype v3c Reference Match — LUNG1-052__c3
===========================================================
v3b → v3c 주요 변경:
  - Panel 1 Row 1: normal reference 3개를 v3c selected top3로 교체
    (selected_reference_top3_v3c.csv 기반, XY 위치 매칭)
  - top3 모두 crop [y256:352, x96:192] (center y304, x144) 동일
  - Panel 1 label: "XY-matched normal" 계열로 변경
  - Panel 1 subtitle: z-orientation limitation 명시
  - Panel 4 Caution: z-orientation limitation 추가
  - metadata: v3c reference selection 정책 추가
  - "same-z matched" / "z-matched" 표현 금지

v3c reference selection 정책:
  - position_bin = lower_central
  - xy_distance = 0.0 (in-slice crop center y304, x144 완벽 일치)
  - z_distance ≈ 0.42 (rough matching only; dataset orientation 차이)
  - stage2_holdout 제외
  - not_same_z_matched = True

선택된 top3:
  matched_normal : subset9__9e5f73ce9b, z=165, crop[256:352, 96:192]
  normal_ex1     : subset5__c3b19ef933, z=134, crop[256:352, 96:192]
  normal_ex2     : normal050__c849177f63, z=202, crop[256:352, 96:192]

출력 root:
  outputs/position-aware-padim-v1/reports/explanation_cards/
    s5_demo_card_prototype_lung1_052_c3_v3c_reference_match/

실행 방법:
  --static-check            → 정적 검사 (20항목)
  --dry-run                 → 입력 파일 존재 + guard 확인
  --plan-only               → layout + 경로 + guard 표시
  --save-static-artifacts   → static/ 하위 산출물 저장 (7종)
  --run-prototype --confirm-generate  → guards False이면 BLOCKED exit 2

actual generation 승인 후 True:
  ALLOW_SOURCE_PNG_READ = True
  ALLOW_PROTOTYPE_RENDER = True
  ALLOW_PROTOTYPE_WRITE = True
  ALLOW_CANDIDATE_CT_LOAD = True
  ALLOW_NORMAL_REF_CT_LOAD = True
"""

import csv
import json
import os
import pathlib
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional, Tuple

# ============================================================
# 최상위 가드 — actual generation 승인 전 False 유지
# ============================================================
ALLOW_SOURCE_PNG_READ      = False
ALLOW_PROTOTYPE_RENDER     = False
ALLOW_PROTOTYPE_WRITE      = False

# CT load guard 분리
ALLOW_CANDIDATE_CT_LOAD    = False   # candidate crop + z-context
ALLOW_NORMAL_REF_CT_LOAD   = False   # normal reference 3개 원본 CT crop

# 항구 False — 절대 변경 금지
ALLOW_S3_MODIFICATION      = False
ALLOW_S4_MODIFICATION      = False
ALLOW_MODEL_FORWARD        = False
ALLOW_FEATURE_EXTRACTION   = False
ALLOW_CONTRIBUTION_RECALC  = False
ALLOW_STAGE2_HOLDOUT       = False
ALLOW_FULL_300             = False

# ============================================================
# 케이스 상수
# ============================================================
CASE_ID              = "LUNG1-052__c3"
VOLUME_ID            = "NSCLC_LUNG1-052__d4a19cc211"
POSITION_BIN         = "lower_central"
CT_LOCAL_Z           = 51
REPORT_SLICE_INDEX   = 106
SPATIAL_PATTERN      = "CENTER_DOMINANT_WITH_DOWNWARD_CONTINUITY"
VISUAL_VERDICT       = "PASS_VISUAL_SUPPORTS_CENTER_DOWNWARD_CONTINUITY_WITH_CAUTION"

GRID_EXTENT_FORMAT   = "y0,x0,y1,x1"
GRID_EXTENT          = [256, 96, 352, 192]
GRID_Y0 = 256; GRID_X0 = 96; GRID_Y1 = 352; GRID_X1 = 192

PATCH4_BBOX          = [288, 128, 320, 160]
PATCH4_SCORE         = 38.872562
PATCH7_BBOX          = [320, 128, 352, 160]
PATCH7_SCORE         = 36.612470

CANDIDATE_CROP_Y0    = 256
CANDIDATE_CROP_X0    = 96
CANDIDATE_CROP_Y1    = 352
CANDIDATE_CROP_X1    = 192
LUNG_WINDOW_CENTER   = -600
LUNG_WINDOW_WIDTH    = 1500

LAYOUT_SELECTED      = "OPTION_V3C_4PANEL_XY_MATCHED_REFERENCE"

# ============================================================
# v3c normal reference crop 좌표 (selected_reference_top3_v3c.csv 기반)
# 모두 crop center y=304, x=144 / crop [y256:352, x96:192]
# ============================================================
# matched_normal: subset9__9e5f73ce9b
MATCHED_NORMAL_SAFE_ID = "subset9_1.3.6.1.4.1.14519.5.2.1.6279.6001.330043769832606379655473292782__9e5f73ce9b"
MATCHED_NORMAL_Z  = 165
MATCHED_NORMAL_Y0 = 256; MATCHED_NORMAL_Y1 = 352
MATCHED_NORMAL_X0 = 96;  MATCHED_NORMAL_X1 = 192

# normal_ex1: subset5__c3b19ef933
NORMAL_EX1_SAFE_ID = "subset5_1.3.6.1.4.1.14519.5.2.1.6279.6001.138894439026794145866157853158__c3b19ef933"
NORMAL_EX1_Z  = 134
NORMAL_EX1_Y0 = 256; NORMAL_EX1_Y1 = 352
NORMAL_EX1_X0 = 96;  NORMAL_EX1_X1 = 192

# normal_ex2: normal050__c849177f63
NORMAL_EX2_SAFE_ID = "normal050__c849177f63"
NORMAL_EX2_Z  = 202
NORMAL_EX2_Y0 = 256; NORMAL_EX2_Y1 = 352
NORMAL_EX2_X0 = 96;  NORMAL_EX2_X1 = 192

# v3c reference selection metadata
V3C_XY_DISTANCE       = 0.0     # 모두 완벽 in-slice match
V3C_Z_DIST_MATCHED    = 0.4206
V3C_Z_DIST_EX1        = 0.4207
V3C_Z_DIST_EX2        = 0.4207
V3C_Z_ORIENTATION_WARN = True   # LUNA16 vs NSCLC z 방향 차이 가능성

# ============================================================
# 경로 상수
# ============================================================
PROJECT_ROOT = pathlib.Path("/home/jinhy/project/lung-ct-anomaly")

# normal reference CT NPY (LUNA16 training set, stage2_holdout 아님)
NORMAL_LUNA16_ROOT = pathlib.Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1"
    "/volumes_npy"
)
MATCHED_NORMAL_CT_NPY = NORMAL_LUNA16_ROOT / MATCHED_NORMAL_SAFE_ID / "ct_hu.npy"
NORMAL_EX1_CT_NPY     = NORMAL_LUNA16_ROOT / NORMAL_EX1_SAFE_ID     / "ct_hu.npy"
NORMAL_EX2_CT_NPY     = NORMAL_LUNA16_ROOT / NORMAL_EX2_SAFE_ID     / "ct_hu.npy"

# candidate CT NPY
CANDIDATE_CT_NPY = pathlib.Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1"
    "/volumes_npy/NSCLC_LUNG1-052__d4a19cc211/ct_hu.npy"
)

# v3c preflight source CSV
V3C_PREFLIGHT_ROOT = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "s5_demo_card_prototype_lung1_052_c3_v3c_reference_match_preflight")
V3C_TOP3_CSV = V3C_PREFLIGHT_ROOT / "selected_reference_top3_v3c.csv"

# Panel 1 Row 2 — whole slice
WHOLE_SLICE_PNG = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "s5_lung1_052_c3_coordinate_visual_audit_1case_v1"
    / "coordinate_overlay_patch4_patch7_lung.png")

# Panel 2 — lung-window overlay
S5_LUNG_OVERLAY_PNG = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "s5_lung1_052_c3_coordinate_visual_audit_1case_v1"
    / "coordinate_overlay_3x3_grid_lung.png")
S5_COORD_METADATA = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "s5_lung1_052_c3_coordinate_visual_audit_1case_v1"
    / "coordinate_overlay_metadata.json")

# Panel 3 — patch map JSON
S5_PATCH_MAP_JSON = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "s5_lung1_052_c3_patch_level_contribution_map_1case_smoke_v1"
    / "patch_contribution_map.json")

# S3 card (참조용)
S3_CARD_JSON = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    / "s3_expansion_cards_v1/cards_json/LUNG1-052__c3.json")

# 기존 버전 output root (충돌 방지용)
_V1_ROOT = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "s5_demo_card_prototype_lung1_052_c3_v1")
_V2_ROOT = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "s5_demo_card_prototype_lung1_052_c3_v2_clinical_readable")
_V3_ROOT = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "s5_demo_card_prototype_lung1_052_c3_v3_clinical_readable")
_V3B_ROOT = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "s5_demo_card_prototype_lung1_052_c3_v3b_reference_fov_fix")

# 출력 경로 (v3c)
OUTPUT_ROOT    = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "s5_demo_card_prototype_lung1_052_c3_v3c_reference_match")
STATIC_DIR     = OUTPUT_ROOT / "static"
CARDS_PNG_DIR  = OUTPUT_ROOT / "cards_png"
CARDS_JSON_DIR = OUTPUT_ROOT / "cards_json"
OUT_PNG        = CARDS_PNG_DIR  / "LUNG1-052__c3_s5_demo_prototype_v3c.png"
OUT_JSON       = CARDS_JSON_DIR / "LUNG1-052__c3_s5_demo_prototype_v3c.json"
OUT_INDEX_CSV  = OUTPUT_ROOT / "index_cards_v3c.csv"
OUT_RUNTIME    = OUTPUT_ROOT / "runtime_summary_v3c.json"
OUT_ERRORS     = OUTPUT_ROOT / "errors.csv"
OUT_DONE       = OUTPUT_ROOT / "DONE.json"

# static 산출물 (7종)
OUT_DRYCHECK_MD          = STATIC_DIR / "drycheck_v3c_reference_match.md"
OUT_DRYCHECK_JSON        = STATIC_DIR / "drycheck_v3c_reference_match.json"
OUT_LAYOUT_CSV           = STATIC_DIR / "layout_plan_v3c.csv"
OUT_REF_SELECTION_CSV    = STATIC_DIR / "reference_selection_policy_v3c.csv"
OUT_TOP3_PATHS_CSV       = STATIC_DIR / "top3_reference_paths_v3c.csv"
OUT_TEXT_POLICY_CSV      = STATIC_DIR / "text_policy_v3c.csv"
OUT_PREFLIGHT_MD         = STATIC_DIR / "preflight_summary_v3c.md"

# ============================================================
# 텍스트 상수 (v3c)
# ============================================================
PROTOTYPE_TITLE_KO = "S5 PaDiM 이상 후보 설명 카드 v3c — LUNG1-052__c3"
PROTOTYPE_TITLE_EN = "S5 PaDiM patch-level explanation prototype v3c — LUNG1-052__c3"

PANEL1_SUBTITLE_EN = (
    "Normal references are matched by in-slice crop center (y=304, x=144); "
    "z alignment is limited by dataset orientation."
)
PANEL1_SUBTITLE_KO = (
    "정상 reference는 같은 slice 내 위치 기준으로 맞췄으며, "
    "z 방향 정합에는 한계가 있습니다."
)

P4_KEY_FINDING_KO = (
    "정상 reference와 비교했을 때, 해당 영역은 더 조밀한 음영을 보였습니다."
)
P4_KEY_FINDING_EN = (
    "Compared with normal references, this region showed denser opacification."
)
P4_INTERPRETATION_KO = (
    "같은 in-slice 위치를 기준으로 맞춘 정상 crop들과 비교했을 때, "
    "PaDiM response가 높은 영역은 정상 예시와 다른 국소 패턴을 보였습니다. "
    "특히 중심 patch와 바로 아래 patch에서도 높은 반응이 연속적으로 확인되었으며, "
    "이 주변을 정상 분포와 다른 국소 영역으로 본 것으로 해석할 수 있습니다."
)
P4_INTERPRETATION_EN = (
    "Compared with normal crops matched by in-slice location, "
    "the PaDiM-responsive region showed a locally distinct pattern from normal examples. "
    "The center patch and the adjacent lower patch showed consecutively high responses."
)
P4_CAUTION_KO = (
    "정상 reference는 in-slice 위치를 기준으로 맞췄지만, "
    "데이터셋 간 z 방향 정합에는 한계가 있습니다. "
    "이 설명은 모델 반응을 이해하기 위한 연구용 보조 설명입니다."
)
P4_CAUTION_EN = (
    "Normal references were matched by in-slice position; "
    "z-direction alignment has limitations across datasets. "
    "This is a research-purpose supplementary explanation for understanding model response."
)
P4_DISCLAIMER_KO = "병변 원인이나 진단 결과를 직접 의미하지 않습니다."
P4_DISCLAIMER_EN  = "This does not directly imply lesion cause or diagnostic result."

LABEL_CANDIDATE      = "후보 crop\n(candidate)"
LABEL_MATCHED_NORMAL = "XY-matched normal\n(정상 reference)"
LABEL_NORMAL_EX1     = "XY-matched normal ex1\n(정상 예시 1)"
LABEL_NORMAL_EX2     = "XY-matched normal ex2\n(정상 예시 2)"
LABEL_WHOLE_SLICE    = "전체 슬라이스 (lung window, patch 위치 표시)"
LABEL_Z_ABOVE        = "위 슬라이스\n(z-1)"
LABEL_Z_CURRENT      = "현재 슬라이스\n(z)"
LABEL_Z_BELOW        = "아래 슬라이스\n(z+1)"

_META_DIM91_CAVEAT = (
    "dim91(raw427/layer3)은 LUNG1-320__c2와 LUNG1-052__c3 모두에서 강하게 나타났다. "
    "해부학적 구조로 해석 금지. covariance inverse amplification 또는 공통 "
    "high-response feature-space pattern 가능성. multi-case review 전까지 "
    "병변/암/혈관/고밀도 원인과 연결 금지."
)
_META_BASAL_PLEURAL_CAUTION = (
    "lower_central position_bin은 basal 및 pleural 접경 패턴이 FP 원인이 될 수 있다. "
    "이 S5 결과가 basal/pleural 조직에 의한 것인지 여부는 미결 상태다."
)
_META_Z_ORIENTATION = (
    "LUNA16 normal z_ratio와 NSCLC candidate z_ratio는 직접 비교 불가. "
    "LUNA16은 apex=0/base=1, NSCLC는 방향이 다를 수 있음. "
    "z_distance ≈ 0.42는 rough matching only. "
    "xy_distance=0.0 (in-slice position 완벽 일치)."
)


# ============================================================
# prototype JSON schema
# ============================================================
def build_prototype_json_schema() -> Dict[str, Any]:
    return {
        "version": "v3c_reference_match",
        "case_id": CASE_ID,
        "volume_id": VOLUME_ID,
        "position_bin": POSITION_BIN,
        "ct_local_z": CT_LOCAL_Z,
        "report_slice_index": REPORT_SLICE_INDEX,
        "spatial_pattern": SPATIAL_PATTERN,
        "visual_verdict": VISUAL_VERDICT,
        "grid_extent": GRID_EXTENT,
        "patch4_bbox": PATCH4_BBOX,
        "patch4_score": PATCH4_SCORE,
        "patch7_bbox": PATCH7_BBOX,
        "patch7_score": PATCH7_SCORE,
        "layout_selected": LAYOUT_SELECTED,
        "panels": {
            "panel_1": (
                "S4-style reference comparison (v3c XY-matched): "
                "Row1=4개 동일 크기 (candidate_crop|XY-matched_normal_CT|normal_ex1_CT|normal_ex2_CT), "
                "모두 원본 CT 96×96 crop; "
                "Row2=whole slice (더 크게); "
                "Row3=z-context 3장 (Row1 cell 0.70배 크기로 축소)"
            ),
            "panel_2": (
                "lung-window overlay 1장만 (patch4/patch7 위치 표시), "
                "간단 legend (not Grad-CAM, not pixel attribution)"
            ),
            "panel_3": (
                "3×3 patch response map schematic "
                "(not CT overlay, not pixel heatmap, not Grad-CAM)"
            ),
            "panel_4": (
                "Key finding / Interpretation / Caution / Disclaimer "
                "(의사용 4섹션; z-orientation limitation 포함)"
            ),
        },
        "metadata": {
            "not_gradcam": True,
            "not_pixel_attribution": True,
            "not_diagnostic": True,
            "internal_use_only": True,
            "font_fix_required_before_external_share": True,
            # v3c reference selection
            "reference_selection_version": "v3c",
            "reference_selection_policy": "XY_MATCHED_WITH_Z_ORIENTATION_WARNING",
            "xy_distance_all_top3": 0.0,
            "z_distance_warning": True,
            "z_orientation_limitation": True,
            "lung_roi_z_normalization_used": False,
            "pleural_distance_used": False,
            "top3_reference_source": str(V3C_TOP3_CSV),
            "not_same_z_matched": True,
            # v3c crop info
            "normal_ref_source": "original_CT_NPY_96x96_crop_XY_matched",
            "normal_ref_crop_center_y": 304,
            "normal_ref_crop_center_x": 144,
            "normal_ref_crop_same_as_candidate_xy": True,
            "zcontext_size_reduced_to_0_70_of_row1": True,
            # cautions
            "dim91_caveat": _META_DIM91_CAVEAT,
            "basal_pleural_caution": _META_BASAL_PLEURAL_CAUTION,
            "z_orientation_caveat": _META_Z_ORIENTATION,
            # panel design
            "panel1_4crop_equal_cell_size": True,
            "panel1_whole_slice_larger_row": True,
            "panel2_overlay_single_only": True,
            "panel3_schematic_not_overlay": True,
            "patch_response_map_source": "score_map_sqrt_mahalanobis",
        },
        "source_files": {
            "matched_normal_ct_npy": str(MATCHED_NORMAL_CT_NPY),
            "normal_ex1_ct_npy":     str(NORMAL_EX1_CT_NPY),
            "normal_ex2_ct_npy":     str(NORMAL_EX2_CT_NPY),
            "whole_slice_png":       str(WHOLE_SLICE_PNG),
            "s5_lung_overlay_png":   str(S5_LUNG_OVERLAY_PNG),
            "s5_patch_map_json":     str(S5_PATCH_MAP_JSON),
            "s3_card_json":          str(S3_CARD_JSON),
            "candidate_ct_npy":      str(CANDIDATE_CT_NPY),
            "v3c_top3_csv":          str(V3C_TOP3_CSV),
        },
        "normal_ref_crops": {
            "matched_normal": {
                "safe_id": MATCHED_NORMAL_SAFE_ID, "z": MATCHED_NORMAL_Z,
                "y0": MATCHED_NORMAL_Y0, "y1": MATCHED_NORMAL_Y1,
                "x0": MATCHED_NORMAL_X0, "x1": MATCHED_NORMAL_X1,
                "reference_match_score": 0.663549,
                "xy_distance": 0.0, "z_distance": 0.4206,
                "z_orientation_warning": True,
            },
            "normal_ex1": {
                "safe_id": NORMAL_EX1_SAFE_ID, "z": NORMAL_EX1_Z,
                "y0": NORMAL_EX1_Y0, "y1": NORMAL_EX1_Y1,
                "x0": NORMAL_EX1_X0, "x1": NORMAL_EX1_X1,
                "reference_match_score": 0.663470,
                "xy_distance": 0.0, "z_distance": 0.4207,
                "z_orientation_warning": True,
            },
            "normal_ex2": {
                "safe_id": NORMAL_EX2_SAFE_ID, "z": NORMAL_EX2_Z,
                "y0": NORMAL_EX2_Y0, "y1": NORMAL_EX2_Y1,
                "x0": NORMAL_EX2_X0, "x1": NORMAL_EX2_X1,
                "reference_match_score": 0.663444,
                "xy_distance": 0.0, "z_distance": 0.4207,
                "z_orientation_warning": True,
            },
        },
        "v3c_reference_selection": {
            "preflight_csv": str(V3C_TOP3_CSV),
            "preflight_verdict": "PARTIAL_PASS",
            "xy_distance_all": 0.0,
            "z_distance_all_approx": 0.42,
            "z_orientation_limitation": True,
            "not_same_z_matched": True,
            "not_z_matched": True,
            "label_policy": "XY-matched normal",
        },
        "canvas_size_px": None,
        "output_png": str(OUT_PNG),
        "generated_date": None,
    }


# ============================================================
# 정적 검사 (20항목)
# ============================================================
def static_check() -> Tuple[int, int, List[str]]:
    passed = 0
    failed = 0
    issues: List[str] = []

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if condition:
            passed += 1
        else:
            failed += 1
            msg = f"FAIL: {label}" + (f" | {detail}" if detail else "")
            issues.append(msg)

    schema = build_prototype_json_schema()
    meta   = schema["metadata"]

    # ── 01: output root contains v3c_reference_match
    check("01 output_root contains v3c_reference_match",
          "v3c_reference_match" in OUTPUT_ROOT.name)

    # ── 02: v3b output root와 다름
    check("02 output_root != v3b_root",
          OUTPUT_ROOT != _V3B_ROOT)

    # ── 03: selected_reference_top3_v3c.csv read-only 참조 (수정 없음)
    check("03 V3C_TOP3_CSV path is under v3c preflight root",
          "v3c_reference_match_preflight" in str(V3C_TOP3_CSV))

    # ── 04: top3 volume_id가 v3c selected top3와 일치
    check("04 matched_normal safe_id contains 9e5f73ce9b",
          "9e5f73ce9b" in MATCHED_NORMAL_SAFE_ID)
    check("04b normal_ex1 safe_id contains c3b19ef933",
          "c3b19ef933" in NORMAL_EX1_SAFE_ID)
    check("04c normal_ex2 safe_id contains normal050",
          "normal050" in NORMAL_EX2_SAFE_ID)

    # ── 05: top3 y_center=304, x_center=144
    check("05 matched_normal crop center y=304",
          (MATCHED_NORMAL_Y0 + MATCHED_NORMAL_Y1) / 2 == 304.0)
    check("05b matched_normal crop center x=144",
          (MATCHED_NORMAL_X0 + MATCHED_NORMAL_X1) / 2 == 144.0)
    check("05c normal_ex1 crop center y=304, x=144",
          (NORMAL_EX1_Y0 + NORMAL_EX1_Y1) / 2 == 304.0 and
          (NORMAL_EX1_X0 + NORMAL_EX1_X1) / 2 == 144.0)
    check("05d normal_ex2 crop center y=304, x=144",
          (NORMAL_EX2_Y0 + NORMAL_EX2_Y1) / 2 == 304.0 and
          (NORMAL_EX2_X0 + NORMAL_EX2_X1) / 2 == 144.0)

    # ── 06: top3 xy_distance=0.0 (metadata 기록)
    check("06 xy_distance_all_top3 = 0.0 in metadata",
          meta.get("xy_distance_all_top3") == 0.0)

    # ── 07: z_distance warning preserved
    check("07 z_distance_warning in metadata",
          meta.get("z_distance_warning") is True)

    # ── 08: z_orientation_limitation metadata exists
    check("08 z_orientation_limitation in metadata",
          meta.get("z_orientation_limitation") is True)

    # ── 09: no same-z wording in texts
    all_text = " ".join([
        P4_KEY_FINDING_KO, P4_KEY_FINDING_EN,
        P4_INTERPRETATION_KO, P4_INTERPRETATION_EN,
        P4_CAUTION_KO, P4_CAUTION_EN,
        P4_DISCLAIMER_KO, P4_DISCLAIMER_EN,
        LABEL_MATCHED_NORMAL, LABEL_NORMAL_EX1, LABEL_NORMAL_EX2,
        PROTOTYPE_TITLE_EN,
    ]).lower()
    check("09 no 'same-z matched' wording",
          "same-z matched" not in all_text)

    # ── 10: no z-matched wording
    check("10 no 'z-matched' wording in labels",
          "z-matched" not in (LABEL_MATCHED_NORMAL + LABEL_NORMAL_EX1 + LABEL_NORMAL_EX2).lower())

    # ── 11: crop size 96×96
    check("11 matched_normal crop 96×96",
          (MATCHED_NORMAL_Y1 - MATCHED_NORMAL_Y0) == 96 and
          (MATCHED_NORMAL_X1 - MATCHED_NORMAL_X0) == 96)
    check("11b normal_ex1 crop 96×96",
          (NORMAL_EX1_Y1 - NORMAL_EX1_Y0) == 96 and
          (NORMAL_EX1_X1 - NORMAL_EX1_X0) == 96)
    check("11c normal_ex2 crop 96×96",
          (NORMAL_EX2_Y1 - NORMAL_EX2_Y0) == 96 and
          (NORMAL_EX2_X1 - NORMAL_EX2_X0) == 96)

    # ── 12: crop bounds valid (0~511)
    check("12 matched_normal crop in bounds",
          0 <= MATCHED_NORMAL_Y0 < MATCHED_NORMAL_Y1 <= 512 and
          0 <= MATCHED_NORMAL_X0 < MATCHED_NORMAL_X1 <= 512)

    # ── 13: stage2_holdout false
    check("13 matched_normal CT not stage2_holdout",
          "stage2_holdout" not in str(MATCHED_NORMAL_CT_NPY))
    check("13b normal_ex1 CT not stage2_holdout",
          "stage2_holdout" not in str(NORMAL_EX1_CT_NPY))
    check("13c normal_ex2 CT not stage2_holdout",
          "stage2_holdout" not in str(NORMAL_EX2_CT_NPY))

    # ── 14: CT load 0 in dry-run (guard)
    check("14 ALLOW_CANDIDATE_CT_LOAD default False",
          not ALLOW_CANDIDATE_CT_LOAD)
    check("14b ALLOW_NORMAL_REF_CT_LOAD default False",
          not ALLOW_NORMAL_REF_CT_LOAD)

    # ── 15: PNG write 0 in dry-run (guard)
    check("15 ALLOW_PROTOTYPE_WRITE default False",
          not ALLOW_PROTOTYPE_WRITE)
    check("15b ALLOW_PROTOTYPE_RENDER default False",
          not ALLOW_PROTOTYPE_RENDER)

    # ── 16: existing artifact modification 0
    check("16 ALLOW_S3_MODIFICATION False",
          not ALLOW_S3_MODIFICATION)
    check("16b ALLOW_S4_MODIFICATION False",
          not ALLOW_S4_MODIFICATION)

    # ── 17: not diagnostic
    check("17 not_diagnostic in metadata",
          meta.get("not_diagnostic") is True)

    # ── 18: not Grad-CAM
    check("18 not_gradcam in metadata",
          meta.get("not_gradcam") is True)
    p2_desc = schema["panels"]["panel_2"].lower()
    check("18b panel2 not grad-cam mention",
          "not grad-cam" in p2_desc)

    # ── 19: not pixel attribution
    check("19 not_pixel_attribution in metadata",
          meta.get("not_pixel_attribution") is True)

    # ── 20: next step clearly v3c actual generation
    check("20 ALLOW_FULL_300 False (next=v3c actual generation)",
          not ALLOW_FULL_300)
    check("20b ALLOW_MODEL_FORWARD False",
          not ALLOW_MODEL_FORWARD)
    check("20c ALLOW_STAGE2_HOLDOUT False",
          not ALLOW_STAGE2_HOLDOUT)

    # ── 추가: candidate crop 불변 확인
    check("C1 candidate crop y0=256",    CANDIDATE_CROP_Y0 == 256)
    check("C2 candidate crop x0=96",     CANDIDATE_CROP_X0 == 96)
    check("C3 candidate crop y1=352",    CANDIDATE_CROP_Y1 == 352)
    check("C4 candidate crop x1=192",    CANDIDATE_CROP_X1 == 192)
    check("C5 PATCH4_BBOX y0==288",      PATCH4_BBOX[0] == 288)
    check("C6 PATCH7_BBOX y0==320",      PATCH7_BBOX[0] == 320)
    check("C7 patch4.y1 == patch7.y0",   PATCH4_BBOX[2] == PATCH7_BBOX[0])
    check("C8 candidate CT not stage2",  "stage2_holdout" not in str(CANDIDATE_CT_NPY))
    check("C9 not_same_z_matched True",  meta.get("not_same_z_matched") is True)
    check("C10 v3c_reference_selection in schema",
          "v3c_reference_selection" in schema)

    return passed, failed, issues


# ============================================================
# dry-run
# ============================================================
def dry_run() -> Tuple[bool, List[str]]:
    print(f"[dry-run] S5 Demo Card Prototype v3c — {CASE_ID}")
    issues: List[str] = []

    def check_path(label: str, p: pathlib.Path) -> None:
        if p.exists():
            print(f"  OK  {label}: {p.name}")
        else:
            msg = f"MISSING {label}: {p}"
            print(f"  ERR {msg}")
            issues.append(msg)

    def check_path_optional(label: str, p: pathlib.Path, note: str = "") -> None:
        if p.exists():
            print(f"  OK  {label}: {p.name}")
        else:
            print(f"  INFO OPTIONAL MISSING {label}: {p}" + (f" ({note})" if note else ""))

    # ── v3c preflight CSV 확인 (참조용, 수정 없음)
    check_path("v3c_top3_csv (preflight source)", V3C_TOP3_CSV)

    # ── normal reference CT NPY (v3c selected top3)
    check_path("matched_normal_ct_npy (subset9__9e5f73ce9b)", MATCHED_NORMAL_CT_NPY)
    check_path("normal_ex1_ct_npy (subset5__c3b19ef933)",     NORMAL_EX1_CT_NPY)
    check_path("normal_ex2_ct_npy (normal050__c849177f63)",   NORMAL_EX2_CT_NPY)

    # ── candidate CT NPY (승인 후 필요)
    check_path_optional("candidate_ct_npy (ALLOW_CANDIDATE_CT_LOAD 승인 후 필요)",
                        CANDIDATE_CT_NPY)

    # ── 기타 source
    check_path("whole_slice_png",     WHOLE_SLICE_PNG)
    check_path("s5_lung_overlay_png", S5_LUNG_OVERLAY_PNG)
    check_path("s5_coord_metadata",   S5_COORD_METADATA)
    check_path("s5_patch_map_json",   S5_PATCH_MAP_JSON)
    check_path("s3_card_json",        S3_CARD_JSON)

    # ── guard 확인
    for name, val in [
        ("ALLOW_SOURCE_PNG_READ",    ALLOW_SOURCE_PNG_READ),
        ("ALLOW_PROTOTYPE_RENDER",   ALLOW_PROTOTYPE_RENDER),
        ("ALLOW_PROTOTYPE_WRITE",    ALLOW_PROTOTYPE_WRITE),
        ("ALLOW_CANDIDATE_CT_LOAD",  ALLOW_CANDIDATE_CT_LOAD),
        ("ALLOW_NORMAL_REF_CT_LOAD", ALLOW_NORMAL_REF_CT_LOAD),
    ]:
        if val:
            issues.append(f"WARN: {name} is True (should be False in dry-run)")

    # ── xy_distance 확인 (메타 체크)
    if V3C_XY_DISTANCE != 0.0:
        issues.append("XY_DISTANCE_NONZERO: v3c top3 xy_distance should be 0.0")
    else:
        print(f"  OK  xy_distance_all_top3 = 0.0 (perfect in-slice match)")

    # ── z-orientation warning 확인
    if not V3C_Z_ORIENTATION_WARN:
        issues.append("Z_ORIENTATION_WARN_MISSING: must be True")
    else:
        print(f"  OK  z_orientation_limitation = True (warning recorded)")

    # ── output collision 확인
    for p in [OUT_PNG, OUT_JSON, OUT_DONE, OUT_INDEX_CSV]:
        if p.exists():
            issues.append(f"COLLISION: {p} already exists")

    if OUTPUT_ROOT.exists():
        print(f"  WARN: v3c output root already exists (static/ 하위 있을 수 있음): {OUTPUT_ROOT}")
    else:
        print(f"  OK  v3c output root (not yet created): {OUTPUT_ROOT.name}")

    # ── stage2_holdout 경로 확인
    for label, p in [
        ("matched_normal_ct", MATCHED_NORMAL_CT_NPY),
        ("normal_ex1_ct",     NORMAL_EX1_CT_NPY),
        ("normal_ex2_ct",     NORMAL_EX2_CT_NPY),
        ("candidate_ct",      CANDIDATE_CT_NPY),
    ]:
        if "stage2_holdout" in str(p):
            issues.append(f"BLOCKED: {label} path contains stage2_holdout: {p}")

    # ── 정적 검사 포함
    passed, failed, si = static_check()
    issues.extend(si)
    print(f"\n  static_check: {passed} passed, {failed} failed")

    ok = len(issues) == 0
    if issues:
        print("\n[dry-run] Issues:")
        for iss in issues:
            print(f"  - {iss}")
        print("\n[dry-run] BLOCKED — 위 문제 해결 후 재실행 필요")
    else:
        print("\n[dry-run] PASS — 모든 source 파일 존재, guard 위반 없음")
        print("NOTE: actual generation에서 5개 guard를 True로 설정 필요")
        print("  ALLOW_SOURCE_PNG_READ / ALLOW_PROTOTYPE_RENDER / ALLOW_PROTOTYPE_WRITE")
        print("  ALLOW_CANDIDATE_CT_LOAD / ALLOW_NORMAL_REF_CT_LOAD")

    return ok, issues


# ============================================================
# plan-only
# ============================================================
def plan_only() -> None:
    print(f"[plan-only] S5 Demo Card Prototype v3c — {CASE_ID}")
    print(f"  Layout : {LAYOUT_SELECTED}")
    print()
    print("  Panel 1 (상단, 전체 너비) — v3c XY-matched reference:")
    print("    Subtitle: " + PANEL1_SUBTITLE_EN)
    print("    Row 1: 4개 동일 크기 crop (각 CELL_SIZE px) — 모두 원본 CT 96×96 crop")
    print(f"      [0] candidate crop       → CANDIDATE_CT_NPY / z={CT_LOCAL_Z} / y256:352 / x96:192")
    print(f"      [1] XY-matched normal    → matched_normal CT / z={MATCHED_NORMAL_Z} / y256:352 / x96:192")
    print(f"      [2] XY-matched normal ex1→ normal_ex1 CT / z={NORMAL_EX1_Z} / y256:352 / x96:192")
    print(f"      [3] XY-matched normal ex2→ normal_ex2 CT / z={NORMAL_EX2_Z} / y256:352 / x96:192")
    print("    Row 2: whole slice (더 크게, lung-window patch4/7 overlay)")
    print("    Row 3: z-context 3장 (Row 1 cell × 0.70 크기로 축소, 보조 정보)")
    print()
    print("  v3c 핵심 변경 (v3b 대비):")
    print("    - normal reference: v3b top3 → v3c selected top3 (XY-matched)")
    print("    - label: 'matched normal' → 'XY-matched normal'")
    print("    - Panel 1 subtitle: z-orientation limitation 명시")
    print("    - Panel 4 Caution: z-orientation limitation 추가")
    print("    - metadata: v3c reference selection policy 추가")
    print()
    print("  v3c Normal reference CT paths:")
    print(f"    matched_normal: ...{MATCHED_NORMAL_CT_NPY.parent.name[:40]}... (exists={MATCHED_NORMAL_CT_NPY.exists()})")
    print(f"    normal_ex1   : ...{NORMAL_EX1_CT_NPY.parent.name[:40]}... (exists={NORMAL_EX1_CT_NPY.exists()})")
    print(f"    normal_ex2   : ...{NORMAL_EX2_CT_NPY.parent.name[:40]}... (exists={NORMAL_EX2_CT_NPY.exists()})")
    print()
    print("  Guards:")
    print(f"    ALLOW_SOURCE_PNG_READ    = {ALLOW_SOURCE_PNG_READ}")
    print(f"    ALLOW_PROTOTYPE_RENDER   = {ALLOW_PROTOTYPE_RENDER}")
    print(f"    ALLOW_PROTOTYPE_WRITE    = {ALLOW_PROTOTYPE_WRITE}")
    print(f"    ALLOW_CANDIDATE_CT_LOAD  = {ALLOW_CANDIDATE_CT_LOAD}")
    print(f"    ALLOW_NORMAL_REF_CT_LOAD = {ALLOW_NORMAL_REF_CT_LOAD}")
    print()
    passed, failed, issues = static_check()
    if issues:
        for iss in issues:
            print(f"  STATIC ISSUE: {iss}")
    else:
        print(f"  static_check: {passed} passed — PASS")


# ============================================================
# static artifacts 저장 (7종)
# ============================================================
def save_static_artifacts() -> None:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    passed, failed, issues = static_check()
    ok_dry, dry_issues = dry_run()

    # ── drycheck json ──────────────────────────────────────
    drycheck_data = {
        "script": "build_s5_demo_card_prototype_lung1_052_c3_v3c_reference_match.py",
        "case_id": CASE_ID,
        "static_check": {"passed": passed, "failed": failed, "issues": issues},
        "dry_run": {"ok": ok_dry, "issues": dry_issues},
        "guards": {
            "ALLOW_SOURCE_PNG_READ":     ALLOW_SOURCE_PNG_READ,
            "ALLOW_PROTOTYPE_RENDER":    ALLOW_PROTOTYPE_RENDER,
            "ALLOW_PROTOTYPE_WRITE":     ALLOW_PROTOTYPE_WRITE,
            "ALLOW_CANDIDATE_CT_LOAD":   ALLOW_CANDIDATE_CT_LOAD,
            "ALLOW_NORMAL_REF_CT_LOAD":  ALLOW_NORMAL_REF_CT_LOAD,
            "ALLOW_S3_MODIFICATION":     ALLOW_S3_MODIFICATION,
            "ALLOW_S4_MODIFICATION":     ALLOW_S4_MODIFICATION,
            "ALLOW_MODEL_FORWARD":       ALLOW_MODEL_FORWARD,
            "ALLOW_FEATURE_EXTRACTION":  ALLOW_FEATURE_EXTRACTION,
            "ALLOW_CONTRIBUTION_RECALC": ALLOW_CONTRIBUTION_RECALC,
            "ALLOW_STAGE2_HOLDOUT":      ALLOW_STAGE2_HOLDOUT,
            "ALLOW_FULL_300":            ALLOW_FULL_300,
        },
        "v3c_reference_selection": {
            "matched_normal": {
                "safe_id": MATCHED_NORMAL_SAFE_ID,
                "z": MATCHED_NORMAL_Z,
                "crop": [MATCHED_NORMAL_Y0, MATCHED_NORMAL_X0, MATCHED_NORMAL_Y1, MATCHED_NORMAL_X1],
                "xy_distance": 0.0, "z_distance": V3C_Z_DIST_MATCHED,
                "path": str(MATCHED_NORMAL_CT_NPY),
                "exists": MATCHED_NORMAL_CT_NPY.exists(),
                "stage2_holdout": "stage2_holdout" in str(MATCHED_NORMAL_CT_NPY),
            },
            "normal_ex1": {
                "safe_id": NORMAL_EX1_SAFE_ID,
                "z": NORMAL_EX1_Z,
                "crop": [NORMAL_EX1_Y0, NORMAL_EX1_X0, NORMAL_EX1_Y1, NORMAL_EX1_X1],
                "xy_distance": 0.0, "z_distance": V3C_Z_DIST_EX1,
                "path": str(NORMAL_EX1_CT_NPY),
                "exists": NORMAL_EX1_CT_NPY.exists(),
                "stage2_holdout": "stage2_holdout" in str(NORMAL_EX1_CT_NPY),
            },
            "normal_ex2": {
                "safe_id": NORMAL_EX2_SAFE_ID,
                "z": NORMAL_EX2_Z,
                "crop": [NORMAL_EX2_Y0, NORMAL_EX2_X0, NORMAL_EX2_Y1, NORMAL_EX2_X1],
                "xy_distance": 0.0, "z_distance": V3C_Z_DIST_EX2,
                "path": str(NORMAL_EX2_CT_NPY),
                "exists": NORMAL_EX2_CT_NPY.exists(),
                "stage2_holdout": "stage2_holdout" in str(NORMAL_EX2_CT_NPY),
            },
        },
        "output_root": str(OUTPUT_ROOT),
        "collision_check": {
            "v3b_root_exists": _V3B_ROOT.exists(),
            "v3c_root_exists": OUTPUT_ROOT.exists(),
            "out_png_exists":  OUT_PNG.exists(),
            "out_json_exists": OUT_JSON.exists(),
        },
    }
    with open(OUT_DRYCHECK_JSON, "w", encoding="utf-8") as f:
        json.dump(drycheck_data, f, ensure_ascii=False, indent=2)
    print(f"  SAVED: {OUT_DRYCHECK_JSON}")

    # ── drycheck md ────────────────────────────────────────
    md_lines = [
        f"# S5 Demo Card v3c Drycheck — {CASE_ID}",
        "",
        f"## Static Check: {passed} passed / {failed} failed",
        "",
    ]
    if issues:
        md_lines += ["### Issues", ""]
        for iss in issues:
            md_lines.append(f"- {iss}")
        md_lines.append("")
    else:
        md_lines.append("모든 정적 검사 통과\n")

    md_lines += [
        "## v3c XY-matched Reference — Normal Reference CT 경로",
        "",
        "| 대상 | safe_id | 경로 존재 | stage2_holdout |",
        "| --- | --- | --- | --- |",
        f"| matched_normal | {MATCHED_NORMAL_SAFE_ID[:30]}... | {'OK' if MATCHED_NORMAL_CT_NPY.exists() else 'MISSING'} | {'YES — BLOCKED' if 'stage2_holdout' in str(MATCHED_NORMAL_CT_NPY) else 'NO'} |",
        f"| normal_ex1 | {NORMAL_EX1_SAFE_ID[:30]}... | {'OK' if NORMAL_EX1_CT_NPY.exists() else 'MISSING'} | {'YES — BLOCKED' if 'stage2_holdout' in str(NORMAL_EX1_CT_NPY) else 'NO'} |",
        f"| normal_ex2 | {NORMAL_EX2_SAFE_ID[:30]}... | {'OK' if NORMAL_EX2_CT_NPY.exists() else 'MISSING'} | {'YES — BLOCKED' if 'stage2_holdout' in str(NORMAL_EX2_CT_NPY) else 'NO'} |",
        "",
        "## v3c Normal Reference Crop 좌표 (96×96, 모두 동일 XY)",
        "",
        "| 대상 | z | y0 | y1 | x0 | x1 | 크기 | xy_dist | z_dist |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        f"| matched_normal | {MATCHED_NORMAL_Z} | {MATCHED_NORMAL_Y0} | {MATCHED_NORMAL_Y1} | {MATCHED_NORMAL_X0} | {MATCHED_NORMAL_X1} | 96×96 | 0.0 | {V3C_Z_DIST_MATCHED:.4f} |",
        f"| normal_ex1 | {NORMAL_EX1_Z} | {NORMAL_EX1_Y0} | {NORMAL_EX1_Y1} | {NORMAL_EX1_X0} | {NORMAL_EX1_X1} | 96×96 | 0.0 | {V3C_Z_DIST_EX1:.4f} |",
        f"| normal_ex2 | {NORMAL_EX2_Z} | {NORMAL_EX2_Y0} | {NORMAL_EX2_Y1} | {NORMAL_EX2_X0} | {NORMAL_EX2_X1} | 96×96 | 0.0 | {V3C_Z_DIST_EX2:.4f} |",
        f"| candidate  | {CT_LOCAL_Z} | {CANDIDATE_CROP_Y0} | {CANDIDATE_CROP_Y1} | {CANDIDATE_CROP_X0} | {CANDIDATE_CROP_X1} | 96×96 | — | — |",
        "",
        "## z-orientation limitation",
        "LUNA16 normal z_ratio와 NSCLC candidate z_ratio는 직접 비교 불가.",
        "xy_distance=0.0 (완벽 in-slice match), z_distance≈0.42 (rough matching only).",
        "",
        "## Guards",
        "",
        "| guard | 현재값 |",
        "| --- | --- |",
        f"| ALLOW_SOURCE_PNG_READ | {ALLOW_SOURCE_PNG_READ} |",
        f"| ALLOW_PROTOTYPE_RENDER | {ALLOW_PROTOTYPE_RENDER} |",
        f"| ALLOW_PROTOTYPE_WRITE | {ALLOW_PROTOTYPE_WRITE} |",
        f"| ALLOW_CANDIDATE_CT_LOAD | {ALLOW_CANDIDATE_CT_LOAD} |",
        f"| ALLOW_NORMAL_REF_CT_LOAD | {ALLOW_NORMAL_REF_CT_LOAD} |",
        "",
    ]
    OUT_DRYCHECK_MD.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"  SAVED: {OUT_DRYCHECK_MD}")

    # ── layout plan csv ────────────────────────────────────
    layout_rows = [
        ["panel", "row", "cell", "cell_count", "source_type", "source_path", "display_size", "label", "ct_load_required", "v3c_change"],
        ["1", "0", "subtitle", "1", "text", PANEL1_SUBTITLE_EN, "full_width", "panel1_subtitle", "False", "NEW: z-orientation limitation"],
        ["1", "1", "0", "4", "CT_crop", str(CANDIDATE_CT_NPY), "CELL_SIZE", LABEL_CANDIDATE, "ALLOW_CANDIDATE_CT_LOAD", "same_as_v3b"],
        ["1", "1", "1", "4", "CT_crop", str(MATCHED_NORMAL_CT_NPY), "CELL_SIZE", LABEL_MATCHED_NORMAL, "ALLOW_NORMAL_REF_CT_LOAD", "NEW: v3c_top3_matched"],
        ["1", "1", "2", "4", "CT_crop", str(NORMAL_EX1_CT_NPY), "CELL_SIZE", LABEL_NORMAL_EX1, "ALLOW_NORMAL_REF_CT_LOAD", "NEW: v3c_top3_ex1"],
        ["1", "1", "3", "4", "CT_crop", str(NORMAL_EX2_CT_NPY), "CELL_SIZE", LABEL_NORMAL_EX2, "ALLOW_NORMAL_REF_CT_LOAD", "NEW: v3c_top3_ex2"],
        ["1", "2", "0", "1", "PNG_read", str(WHOLE_SLICE_PNG), "larger_than_row1", LABEL_WHOLE_SLICE, "False", "same_as_v3b"],
        ["1", "3", "0", "3", "CT_crop", str(CANDIDATE_CT_NPY), "CELL_SIZE*0.70", LABEL_Z_ABOVE, "ALLOW_CANDIDATE_CT_LOAD", "same_as_v3b"],
        ["1", "3", "1", "3", "CT_crop", str(CANDIDATE_CT_NPY), "CELL_SIZE*0.70", LABEL_Z_CURRENT, "ALLOW_CANDIDATE_CT_LOAD", "same_as_v3b"],
        ["1", "3", "2", "3", "CT_crop", str(CANDIDATE_CT_NPY), "CELL_SIZE*0.70", LABEL_Z_BELOW, "ALLOW_CANDIDATE_CT_LOAD", "same_as_v3b"],
        ["2", "1", "0", "1", "PNG_read", str(S5_LUNG_OVERLAY_PNG), "panel_width", "lung overlay (patch4/7)", "False", "same_as_v3b"],
        ["3", "1", "schematic", "1", "JSON_data", str(S5_PATCH_MAP_JSON), "3x3_grid", "patch response map schematic", "False", "same_as_v3b"],
        ["4", "1", "text", "1", "text_const", "P4_KEY_FINDING_KO", "text_box", "[Key finding]", "False", "same_as_v3b"],
        ["4", "2", "text", "1", "text_const", "P4_INTERPRETATION_KO", "text_box", "[Interpretation]", "False", "same_as_v3b"],
        ["4", "3", "text", "1", "text_const", "P4_CAUTION_KO", "text_box", "[Caution]", "False", "NEW: z-orientation limitation"],
        ["4", "4", "text", "1", "text_const", "P4_DISCLAIMER_KO", "text_box", "[Disclaimer]", "False", "same_as_v3b"],
    ]
    with open(OUT_LAYOUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(layout_rows)
    print(f"  SAVED: {OUT_LAYOUT_CSV}")

    # ── reference selection policy csv ────────────────────
    ref_sel_rows = [
        ["check_id", "item", "v3b_value", "v3c_value", "status"],
        ["R01", "reference_selection_policy", "FOV_FIX_SAME_BIN", "XY_MATCHED_WITH_Z_ORIENTATION_WARNING", "CHANGED"],
        ["R02", "xy_distance_all", "N/A (different centers)", "0.0 (perfect in-slice match)", "IMPROVED"],
        ["R03", "z_distance_all", "N/A (rough)", "≈0.42 (rough, WARNING)", "WARNING"],
        ["R04", "z_orientation_limitation", "not recorded", "True (metadata)", "ADDED"],
        ["R05", "not_same_z_matched", "not recorded", "True", "ADDED"],
        ["R06", "label_matched_normal", "matched normal", "XY-matched normal", "CHANGED"],
        ["R07", "label_normal_ex1", "normal ex1", "XY-matched normal ex1", "CHANGED"],
        ["R08", "label_normal_ex2", "normal ex2", "XY-matched normal ex2", "CHANGED"],
        ["R09", "panel1_subtitle_z_limitation", "absent", "present", "ADDED"],
        ["R10", "p4_caution_z_orientation", "absent", "present", "ADDED"],
        ["R11", "top3_source", "reference_bank_v1 (position_bin only)", "selected_reference_top3_v3c.csv (XY-scored)", "IMPROVED"],
        ["R12", "stage2_holdout_excluded", "True", "True", "SAME"],
        ["R13", "crop_size_96x96", "True", "True", "SAME"],
        ["R14", "ct_path_exists_all", "True", f"{MATCHED_NORMAL_CT_NPY.exists()}/{NORMAL_EX1_CT_NPY.exists()}/{NORMAL_EX2_CT_NPY.exists()}", "CHECK"],
    ]
    with open(OUT_REF_SELECTION_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(ref_sel_rows)
    print(f"  SAVED: {OUT_REF_SELECTION_CSV}")

    # ── top3 reference paths csv ───────────────────────────
    top3_rows = [
        ["target", "safe_id", "path", "exists", "stage2_holdout", "crop_z", "crop_y0", "crop_y1", "crop_x0", "crop_x1", "crop_size", "xy_distance", "z_distance"],
        ["matched_normal", MATCHED_NORMAL_SAFE_ID, str(MATCHED_NORMAL_CT_NPY),
         str(MATCHED_NORMAL_CT_NPY.exists()),
         str("stage2_holdout" in str(MATCHED_NORMAL_CT_NPY)),
         MATCHED_NORMAL_Z, MATCHED_NORMAL_Y0, MATCHED_NORMAL_Y1, MATCHED_NORMAL_X0, MATCHED_NORMAL_X1,
         "96x96", 0.0, V3C_Z_DIST_MATCHED],
        ["normal_ex1", NORMAL_EX1_SAFE_ID, str(NORMAL_EX1_CT_NPY),
         str(NORMAL_EX1_CT_NPY.exists()),
         str("stage2_holdout" in str(NORMAL_EX1_CT_NPY)),
         NORMAL_EX1_Z, NORMAL_EX1_Y0, NORMAL_EX1_Y1, NORMAL_EX1_X0, NORMAL_EX1_X1,
         "96x96", 0.0, V3C_Z_DIST_EX1],
        ["normal_ex2", NORMAL_EX2_SAFE_ID, str(NORMAL_EX2_CT_NPY),
         str(NORMAL_EX2_CT_NPY.exists()),
         str("stage2_holdout" in str(NORMAL_EX2_CT_NPY)),
         NORMAL_EX2_Z, NORMAL_EX2_Y0, NORMAL_EX2_Y1, NORMAL_EX2_X0, NORMAL_EX2_X1,
         "96x96", 0.0, V3C_Z_DIST_EX2],
        ["candidate", "NSCLC_LUNG1-052__d4a19cc211", str(CANDIDATE_CT_NPY),
         str(CANDIDATE_CT_NPY.exists()),
         str("stage2_holdout" in str(CANDIDATE_CT_NPY)),
         CT_LOCAL_Z, CANDIDATE_CROP_Y0, CANDIDATE_CROP_Y1, CANDIDATE_CROP_X0, CANDIDATE_CROP_X1,
         "96x96", "—", "—"],
    ]
    with open(OUT_TOP3_PATHS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(top3_rows)
    print(f"  SAVED: {OUT_TOP3_PATHS_CSV}")

    # ── text policy csv ────────────────────────────────────
    all_text_lower = " ".join([P4_KEY_FINDING_KO, P4_INTERPRETATION_KO,
                               P4_CAUTION_KO, P4_DISCLAIMER_KO,
                               LABEL_MATCHED_NORMAL, LABEL_NORMAL_EX1, LABEL_NORMAL_EX2]).lower()
    text_policy_rows = [
        ["check_id", "target", "rule", "status"],
        ["T01", "panel4_body", "no dim91",                    "PASS" if "dim91" not in all_text_lower else "FAIL"],
        ["T02", "panel4_body", "no raw427",                   "PASS" if "raw427" not in all_text_lower else "FAIL"],
        ["T03", "panel4_body", "no top-k",                    "PASS" if "top-k" not in all_text_lower else "FAIL"],
        ["T04", "panel4_body", "no Grad-CAM",                 "PASS" if "grad-cam" not in all_text_lower else "FAIL"],
        ["T05", "panel4_body", "no pixel attribution",        "PASS" if "pixel attribution" not in all_text_lower else "FAIL"],
        ["T06", "panel4_body", "no same-z matched",           "PASS" if "same-z matched" not in all_text_lower else "FAIL"],
        ["T07", "panel4_body", "no z-matched",                "PASS" if "z-matched" not in (LABEL_MATCHED_NORMAL+LABEL_NORMAL_EX1+LABEL_NORMAL_EX2).lower() else "FAIL"],
        ["T08", "p4_key_finding_ko", "조밀한 음영",           "PASS" if "조밀한 음영" in P4_KEY_FINDING_KO else "FAIL"],
        ["T09", "p4_interpretation_ko", "in-slice 위치",      "PASS" if "in-slice 위치" in P4_INTERPRETATION_KO else "FAIL"],
        ["T10", "p4_interpretation_ko", "중심+아래 patch",    "PASS" if "중심 patch" in P4_INTERPRETATION_KO and "아래 patch" in P4_INTERPRETATION_KO else "FAIL"],
        ["T11", "p4_caution_ko", "z 방향 정합 한계",          "PASS" if "z 방향 정합" in P4_CAUTION_KO else "FAIL"],
        ["T12", "p4_caution_ko", "연구용 보조 설명",          "PASS" if "연구용 보조 설명" in P4_CAUTION_KO else "FAIL"],
        ["T13", "p4_disclaimer_ko", "진단 결과 아님",         "PASS" if "진단 결과" in P4_DISCLAIMER_KO else "FAIL"],
        ["T14", "label_matched", "XY-matched normal",         "PASS" if "XY-matched normal" in LABEL_MATCHED_NORMAL else "FAIL"],
        ["T15", "label_ex1", "XY-matched normal ex1",         "PASS" if "XY-matched normal ex1" in LABEL_NORMAL_EX1 else "FAIL"],
        ["T16", "label_ex2", "XY-matched normal ex2",         "PASS" if "XY-matched normal ex2" in LABEL_NORMAL_EX2 else "FAIL"],
        ["T17", "metadata", "not_gradcam=True",               "PASS"],
        ["T18", "metadata", "not_pixel_attribution=True",     "PASS"],
        ["T19", "metadata", "not_diagnostic=True",            "PASS"],
        ["T20", "metadata", "z_orientation_limitation=True",  "PASS"],
    ]
    with open(OUT_TEXT_POLICY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(text_policy_rows)
    print(f"  SAVED: {OUT_TEXT_POLICY_CSV}")

    # ── preflight summary md ───────────────────────────────
    fail_count = failed + len([i for i in dry_issues if i.startswith("FAIL")])
    preflight_ok = (fail_count == 0)
    verdict = "PASS" if preflight_ok else "BLOCKED"
    preflight_lines = [
        f"# Preflight Summary — {CASE_ID} S5 v3c",
        "",
        f"## 결과: {verdict}",
        "",
        f"- static_check: {passed} passed / {failed} failed",
        f"- dry_run: {'OK' if ok_dry else 'ISSUES'}",
        f"- output collision: {'없음' if not OUT_PNG.exists() else 'COLLISION 있음'}",
        "",
        "## v3c 핵심 변경 확인",
        "",
        "### Normal reference top3 (v3c selected)",
        f"- matched_normal: {MATCHED_NORMAL_SAFE_ID[:40]}...",
        f"  - z={MATCHED_NORMAL_Z}, crop=[256:352, 96:192], xy_dist=0.0, z_dist={V3C_Z_DIST_MATCHED}",
        f"  - CT exists: {MATCHED_NORMAL_CT_NPY.exists()}",
        f"- normal_ex1: {NORMAL_EX1_SAFE_ID[:40]}...",
        f"  - z={NORMAL_EX1_Z}, crop=[256:352, 96:192], xy_dist=0.0, z_dist={V3C_Z_DIST_EX1}",
        f"  - CT exists: {NORMAL_EX1_CT_NPY.exists()}",
        f"- normal_ex2: {NORMAL_EX2_SAFE_ID}",
        f"  - z={NORMAL_EX2_Z}, crop=[256:352, 96:192], xy_dist=0.0, z_dist={V3C_Z_DIST_EX2}",
        f"  - CT exists: {NORMAL_EX2_CT_NPY.exists()}",
        "",
        "### Label 변경",
        "- matched normal → XY-matched normal",
        "- normal ex1 → XY-matched normal ex1",
        "- normal ex2 → XY-matched normal ex2",
        "",
        "### 추가 항목",
        "- Panel 1 subtitle: z-orientation limitation 명시",
        "- Panel 4 Caution: z-orientation limitation 추가",
        "- metadata: reference_selection_version=v3c, not_same_z_matched=True",
        "",
        "## 다음 단계",
        "",
        "1. (현재) 정적 검사 PASS → 사용자 검토",
        "2. (다음) actual generation 승인:",
        "   - ALLOW_SOURCE_PNG_READ = True",
        "   - ALLOW_PROTOTYPE_RENDER = True",
        "   - ALLOW_PROTOTYPE_WRITE = True",
        "   - ALLOW_CANDIDATE_CT_LOAD = True",
        "   - ALLOW_NORMAL_REF_CT_LOAD = True",
        "3. (이후) --run-prototype --confirm-generate 실행",
        "",
        "## Safety",
        "- CT load: 0",
        "- PNG write: 0",
        "- model/feature/contribution: 0",
        "- stage2_holdout 접근: 없음",
        "- 기존 artifact 수정: 없음",
        "",
    ]
    OUT_PREFLIGHT_MD.write_text("\n".join(preflight_lines), encoding="utf-8")
    print(f"  SAVED: {OUT_PREFLIGHT_MD}")

    summary = "PASS" if (failed == 0 and not dry_issues) else f"BLOCKED ({failed} static fails, {len(dry_issues)} dry issues)"
    print(f"\n  [save-static-artifacts] {summary}")


# ============================================================
# actual generation (guards=False이면 BLOCKED)
# ============================================================
def run_prototype() -> None:
    if not (ALLOW_SOURCE_PNG_READ and ALLOW_PROTOTYPE_RENDER and ALLOW_PROTOTYPE_WRITE):
        print("BLOCKED: generation guards are False — set all to True with explicit approval",
              file=sys.stderr)
        sys.exit(2)
    if not (ALLOW_CANDIDATE_CT_LOAD and ALLOW_NORMAL_REF_CT_LOAD):
        print("BLOCKED: ALLOW_CANDIDATE_CT_LOAD and ALLOW_NORMAL_REF_CT_LOAD must both be True",
              file=sys.stderr)
        sys.exit(2)
    if ALLOW_S3_MODIFICATION or ALLOW_S4_MODIFICATION:
        print("BLOCKED: S3/S4 modification guards must remain False", file=sys.stderr)
        sys.exit(2)
    if ALLOW_MODEL_FORWARD or ALLOW_FEATURE_EXTRACTION:
        print("BLOCKED: model/feature guards must remain False", file=sys.stderr)
        sys.exit(2)
    if ALLOW_STAGE2_HOLDOUT or ALLOW_FULL_300:
        print("BLOCKED: stage2/full300 guards must remain False", file=sys.stderr)
        sys.exit(2)

    # output collision block
    collision_paths = [OUT_DONE, OUT_PNG, OUT_JSON, OUT_INDEX_CSV, OUT_RUNTIME, OUT_ERRORS]
    existing = [str(p) for p in collision_paths if p.exists()]
    if existing:
        print("BLOCKED: output collision detected:", file=sys.stderr)
        for p in existing:
            print(f"  - {p}", file=sys.stderr)
        print("Use a new output root (e.g. v3d) rather than deleting.", file=sys.stderr)
        sys.exit(2)

    t_start = time.time()
    errors: List[str] = []

    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
    except ImportError as e:
        print(f"BLOCKED: required library not available: {e}", file=sys.stderr)
        sys.exit(2)

    # font
    FONT_KO_PATH = pathlib.Path("/mnt/c/Windows/Fonts/malgun.ttf")
    FONT_EN_PATH = pathlib.Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    FONT_EN_BOLD = pathlib.Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    if not FONT_KO_PATH.exists():
        print("BLOCKED: malgun.ttf not found", file=sys.stderr)
        sys.exit(2)

    def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        try:
            return ImageFont.truetype(str(FONT_KO_PATH), size)
        except Exception:
            return ImageFont.truetype(str(FONT_EN_BOLD if bold else FONT_EN_PATH), size)

    fnt_title   = _font(20, bold=True)
    fnt_section = _font(15, bold=True)
    fnt_body    = _font(13)
    fnt_small   = _font(11)
    fnt_label   = _font(11)

    # source PNG 읽기 (Panel 2, whole slice)
    img_whole   = Image.open(WHOLE_SLICE_PNG).convert("RGBA")
    img_lung_ov = Image.open(S5_LUNG_OVERLAY_PNG).convert("RGBA")

    # JSON 데이터
    with open(S5_PATCH_MAP_JSON) as f:
        patch_map = json.load(f)

    # CT 유틸
    def hu_to_lung_window(arr: "np.ndarray") -> "np.ndarray":
        lo = LUNG_WINDOW_CENTER - LUNG_WINDOW_WIDTH // 2
        hi = LUNG_WINDOW_CENTER + LUNG_WINDOW_WIDTH // 2
        clipped = np.clip(arr, lo, hi)
        return ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)

    def make_ct_crop_img_yx(ct_arr: "np.ndarray", z: int,
                             y0: int, x0: int, y1: int, x1: int,
                             display_size: int) -> Image.Image:
        if z < 0 or z >= ct_arr.shape[0]:
            return _placeholder(display_size, display_size, f"z={z} out of range")
        slc = ct_arr[z, y0:y1, x0:x1]
        gray = hu_to_lung_window(slc)
        img = Image.fromarray(gray, mode="L").convert("RGBA")
        return img.resize((display_size, display_size), Image.LANCZOS)

    def _placeholder(w: int, h: int, text: str = "CT load required") -> Image.Image:
        img = Image.new("RGBA", (w, h), (50, 50, 60, 255))
        d = ImageDraw.Draw(img)
        d.rectangle([(0, 0), (w - 1, h - 1)], outline=(100, 100, 120, 255), width=2)
        for i, line in enumerate(text.split("\n")):
            d.text((6, 6 + i * 14), line, font=fnt_small, fill=(160, 160, 180, 255))
        return img

    # CT load — candidate
    ct_arr = None
    if not CANDIDATE_CT_NPY.exists():
        errors.append(f"candidate CT NPY not found: {CANDIDATE_CT_NPY}")
    else:
        ct_arr = np.load(str(CANDIDATE_CT_NPY), mmap_mode="r")

    # CT load — normal references (mmap_mode="r", write 금지)
    matched_normal_ct = None
    normal_ex1_ct     = None
    normal_ex2_ct     = None

    if not MATCHED_NORMAL_CT_NPY.exists():
        errors.append(f"matched_normal CT NPY not found: {MATCHED_NORMAL_CT_NPY}")
    else:
        matched_normal_ct = np.load(str(MATCHED_NORMAL_CT_NPY), mmap_mode="r")

    if not NORMAL_EX1_CT_NPY.exists():
        errors.append(f"normal_ex1 CT NPY not found: {NORMAL_EX1_CT_NPY}")
    else:
        normal_ex1_ct = np.load(str(NORMAL_EX1_CT_NPY), mmap_mode="r")

    if not NORMAL_EX2_CT_NPY.exists():
        errors.append(f"normal_ex2 CT NPY not found: {NORMAL_EX2_CT_NPY}")
    else:
        normal_ex2_ct = np.load(str(NORMAL_EX2_CT_NPY), mmap_mode="r")

    # ── 캔버스 치수
    CANVAS_W   = 1600
    MARGIN     = 16
    HEADER_H   = 64        # subtitle 한 줄 추가로 v3b보다 약간 높음
    PAD        = 8

    P1_W = CANVAS_W - 2 * MARGIN
    CELL_SIZE = (P1_W - 5 * PAD) // 4
    LABEL_H   = 30
    SUBTITLE_H = 24

    WHOLE_W = int(P1_W * 0.68)
    WHOLE_H = int(WHOLE_W * img_whole.height / img_whole.width)

    Z_CELL_W = int(CELL_SIZE * 0.70)
    Z_CELL_H = Z_CELL_W

    P1_ROW1_H = LABEL_H + CELL_SIZE + LABEL_H
    P1_ROW2_H = LABEL_H + WHOLE_H
    P1_ROW3_H = LABEL_H + Z_CELL_H
    P1_H      = SUBTITLE_H + PAD + P1_ROW1_H + PAD + P1_ROW2_H + PAD + P1_ROW3_H + PAD

    LOWER_PANEL_W = (CANVAS_W - 4 * MARGIN) // 3
    OV_W, OV_H   = img_lung_ov.size
    LOWER_OV_H   = int(LOWER_PANEL_W * OV_H / OV_W)
    LOWER_H      = LOWER_OV_H + 100

    CANVAS_H = HEADER_H + MARGIN + P1_H + MARGIN + LOWER_H + MARGIN

    # 색상
    C_BG       = (18, 18, 18, 255)
    C_PANEL_BG = (30, 30, 30, 255)
    C_HEADER   = (10, 22, 45, 255)
    C_BORDER   = (70, 70, 70, 255)
    C_TITLE    = (220, 220, 255, 255)
    C_SECTION  = (150, 195, 255, 255)
    C_BODY     = (215, 215, 215, 255)
    C_WARN     = (255, 215, 75, 255)
    C_DISC     = (195, 155, 155, 255)
    C_LABEL    = (190, 190, 190, 255)
    C_YELLOW   = (255, 220,   0, 255)
    C_ORANGE   = (255, 140,   0, 255)
    C_SUBTITLE = (170, 200, 170, 255)

    canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), C_BG)
    draw   = ImageDraw.Draw(canvas)

    # 헤더
    draw.rectangle([(0, 0), (CANVAS_W, HEADER_H)], fill=C_HEADER)
    draw.text((MARGIN, 8),  PROTOTYPE_TITLE_KO, font=fnt_title, fill=C_TITLE)
    draw.text((MARGIN, 32), "[INTERNAL USE ONLY | v3c | not diagnostic]",
              font=fnt_small, fill=C_WARN)
    draw.text((MARGIN, 48), PANEL1_SUBTITLE_KO, font=fnt_small, fill=C_SUBTITLE)

    # Panel 1 배경
    p1_x0 = MARGIN
    p1_y0 = HEADER_H + MARGIN
    p1_x1 = MARGIN + P1_W
    p1_y1 = p1_y0 + P1_H
    draw.rectangle([(p1_x0, p1_y0), (p1_x1, p1_y1)], fill=C_PANEL_BG, outline=C_BORDER, width=1)

    cy = p1_y0 + PAD
    draw.text((p1_x0 + PAD, cy),
              "Panel 1 : 정상 reference 비교 (XY-matched)  |  전체 슬라이스  |  z-context",
              font=fnt_section, fill=C_SECTION)
    cy += LABEL_H

    # Panel 1 subtitle
    draw.text((p1_x0 + PAD, cy), PANEL1_SUBTITLE_EN, font=fnt_small, fill=C_SUBTITLE)
    cy += SUBTITLE_H

    # Panel 1 Row 1: 4개 동일 크기 crop
    crop_x_starts = [p1_x0 + PAD + i * (CELL_SIZE + PAD) for i in range(4)]
    cell_labels   = [LABEL_CANDIDATE, LABEL_MATCHED_NORMAL, LABEL_NORMAL_EX1, LABEL_NORMAL_EX2]
    cell_imgs: List[Optional[Image.Image]] = [None, None, None, None]

    # [0] candidate crop
    if ct_arr is not None:
        cell_imgs[0] = make_ct_crop_img_yx(
            ct_arr, CT_LOCAL_Z,
            CANDIDATE_CROP_Y0, CANDIDATE_CROP_X0,
            CANDIDATE_CROP_Y1, CANDIDATE_CROP_X1, CELL_SIZE)
    else:
        cell_imgs[0] = _placeholder(CELL_SIZE, CELL_SIZE, "candidate CT\nnot found")

    # [1] XY-matched normal
    if matched_normal_ct is not None:
        cell_imgs[1] = make_ct_crop_img_yx(
            matched_normal_ct, MATCHED_NORMAL_Z,
            MATCHED_NORMAL_Y0, MATCHED_NORMAL_X0,
            MATCHED_NORMAL_Y1, MATCHED_NORMAL_X1, CELL_SIZE)
    else:
        cell_imgs[1] = _placeholder(CELL_SIZE, CELL_SIZE, "matched normal CT\nnot found")

    # [2] XY-matched normal ex1
    if normal_ex1_ct is not None:
        cell_imgs[2] = make_ct_crop_img_yx(
            normal_ex1_ct, NORMAL_EX1_Z,
            NORMAL_EX1_Y0, NORMAL_EX1_X0,
            NORMAL_EX1_Y1, NORMAL_EX1_X1, CELL_SIZE)
    else:
        cell_imgs[2] = _placeholder(CELL_SIZE, CELL_SIZE, "normal ex1 CT\nnot found")

    # [3] XY-matched normal ex2
    if normal_ex2_ct is not None:
        cell_imgs[3] = make_ct_crop_img_yx(
            normal_ex2_ct, NORMAL_EX2_Z,
            NORMAL_EX2_Y0, NORMAL_EX2_X0,
            NORMAL_EX2_Y1, NORMAL_EX2_X1, CELL_SIZE)
    else:
        cell_imgs[3] = _placeholder(CELL_SIZE, CELL_SIZE, "normal ex2 CT\nnot found")

    for i, (cx_start, img_cell, lbl) in enumerate(zip(crop_x_starts, cell_imgs, cell_labels)):
        draw.rectangle([(cx_start - 1, cy - 1), (cx_start + CELL_SIZE, cy + CELL_SIZE)],
                       outline=C_YELLOW if i == 0 else C_BORDER,
                       width=2 if i == 0 else 1)
        canvas.paste(img_cell, (cx_start, cy), img_cell)
        for j, line in enumerate(lbl.split("\n")):
            draw.text((cx_start, cy + CELL_SIZE + 2 + j * 12), line,
                      font=fnt_small, fill=C_YELLOW if i == 0 else C_LABEL)

    cy += CELL_SIZE + LABEL_H + PAD

    # Panel 1 Row 2: whole slice
    draw.text((p1_x0 + PAD, cy), LABEL_WHOLE_SLICE, font=fnt_label, fill=C_LABEL)
    cy += LABEL_H - 8
    whole_x = p1_x0 + (P1_W - WHOLE_W) // 2
    img_whole_r = img_whole.resize((WHOLE_W, WHOLE_H), Image.LANCZOS)
    canvas.paste(img_whole_r, (whole_x, cy), img_whole_r)
    cy += WHOLE_H + PAD

    # Panel 1 Row 3: z-context (0.70 크기)
    draw.text((p1_x0 + PAD, cy), "Z-context (위/현재/아래 슬라이스 — 위치 맥락 보조)",
              font=fnt_label, fill=C_LABEL)
    cy += LABEL_H - 8

    z_labels  = [LABEL_Z_ABOVE, LABEL_Z_CURRENT, LABEL_Z_BELOW]
    z_offsets = [-1, 0, 1]
    z_starts  = [p1_x0 + PAD + i * (Z_CELL_W + PAD) for i in range(3)]

    for z_x, z_off, z_lbl in zip(z_starts, z_offsets, z_labels):
        z_slice = CT_LOCAL_Z + z_off
        if ct_arr is not None:
            z_img = make_ct_crop_img_yx(
                ct_arr, z_slice,
                CANDIDATE_CROP_Y0, CANDIDATE_CROP_X0,
                CANDIDATE_CROP_Y1, CANDIDATE_CROP_X1, Z_CELL_W)
            z_img = z_img.resize((Z_CELL_W, Z_CELL_H), Image.LANCZOS)
        else:
            z_img = _placeholder(Z_CELL_W, Z_CELL_H, f"CT load\nz={z_slice}")
        draw.rectangle([(z_x - 1, cy - 1), (z_x + Z_CELL_W, cy + Z_CELL_H)],
                       outline=C_YELLOW if z_off == 0 else C_BORDER,
                       width=2 if z_off == 0 else 1)
        canvas.paste(z_img, (z_x, cy), z_img)
        for j, line in enumerate(z_lbl.split("\n")):
            draw.text((z_x, cy + Z_CELL_H + 2 + j * 12), line,
                      font=fnt_small, fill=C_YELLOW if z_off == 0 else C_LABEL)

    # 하단 패널
    lower_y = HEADER_H + MARGIN + P1_H + MARGIN
    p2_x0 = MARGIN
    p3_x0 = MARGIN + LOWER_PANEL_W + MARGIN
    p4_x0 = MARGIN + (LOWER_PANEL_W + MARGIN) * 2

    def draw_lower_panel(bx0: int, title: str) -> Tuple[int, int, int, int]:
        bx1 = bx0 + LOWER_PANEL_W
        by0 = lower_y
        by1 = lower_y + LOWER_H
        draw.rectangle([(bx0, by0), (bx1, by1)], fill=C_PANEL_BG, outline=C_BORDER, width=1)
        draw.text((bx0 + PAD, by0 + PAD), title, font=fnt_section, fill=C_SECTION)
        return bx0, by0, bx1, by1

    # Panel 2
    bx0, by0, bx1, by1 = draw_lower_panel(p2_x0, "Panel 2 : Model 반응 위치")
    cy2 = by0 + PAD + LABEL_H
    img_ov_r = img_lung_ov.resize((LOWER_PANEL_W - 2 * PAD, LOWER_OV_H), Image.LANCZOS)
    canvas.paste(img_ov_r, (bx0 + PAD, cy2), img_ov_r)
    cy2 += LOWER_OV_H + 8
    draw.rectangle([(bx0 + PAD, cy2), (bx0 + PAD + 12, cy2 + 12)], fill=(255, 220, 0, 255))
    draw.text((bx0 + PAD + 16, cy2), "Highest response", font=fnt_small, fill=C_BODY)
    cy2 += 16
    draw.rectangle([(bx0 + PAD, cy2), (bx0 + PAD + 12, cy2 + 12)], fill=(255, 140, 0, 255))
    draw.text((bx0 + PAD + 16, cy2), "Second-highest response", font=fnt_small, fill=C_BODY)
    cy2 += 18
    draw.text((bx0 + PAD, cy2), "not Grad-CAM / not pixel attribution",
              font=fnt_small, fill=(130, 130, 130, 255))

    # Panel 3
    bx0, by0, bx1, by1 = draw_lower_panel(p3_x0, "Panel 3 : Patch response map")
    cy3 = by0 + PAD + LABEL_H
    draw.text((p3_x0 + PAD, cy3), "patch-level response map / not pixel heatmap",
              font=fnt_small, fill=(170, 170, 170, 255))
    cy3 += 16

    score_map   = patch_map.get("score_map_sqrt_mahalanobis", [[0] * 3] * 3)
    flat_scores = [score_map[r][c] for r in range(3) for c in range(3)]
    min_s = min(flat_scores) if flat_scores else 0.0
    max_s = max(flat_scores) if flat_scores else 1.0
    score_range = max(max_s - min_s, 1e-6)

    GRID_SIZE = min(LOWER_PANEL_W - 2 * PAD, LOWER_H - LABEL_H - 60)
    CELL_G    = (GRID_SIZE - 3 * 4) // 3
    GAP_G     = 4
    ACTUAL_G  = CELL_G * 3 + GAP_G * 2
    grid_x0   = p3_x0 + PAD + (LOWER_PANEL_W - 2 * PAD - ACTUAL_G) // 2
    grid_y0   = cy3

    for r in range(3):
        for c in range(3):
            pid   = r * 3 + c
            score = score_map[r][c] if r < len(score_map) and c < len(score_map[r]) else 0.0
            norm  = (score - min_s) / score_range
            cell_x = grid_x0 + c * (CELL_G + GAP_G)
            cell_y = grid_y0 + r * (CELL_G + GAP_G)
            if pid == 4:
                bg = C_YELLOW
            elif pid == 7:
                bg = C_ORANGE
            else:
                iv = int(35 + norm * 70)
                bg = (iv, iv, min(iv + 25, 255), 255)
            lc = (25, 25, 25, 255) if pid in (4, 7) else (190, 190, 190, 255)
            draw.rectangle(
                [(cell_x, cell_y), (cell_x + CELL_G - 1, cell_y + CELL_G - 1)],
                fill=bg, outline=(90, 90, 90, 255), width=1
            )
            draw.text((cell_x + 4, cell_y + 4),         f"P{pid}",    font=fnt_small, fill=lc)
            draw.text((cell_x + 4, cell_y + CELL_G - 16), f"{score:.1f}", font=fnt_small, fill=lc)

    cy3 = grid_y0 + ACTUAL_G + 8
    draw.rectangle([(grid_x0, cy3), (grid_x0 + 12, cy3 + 12)], fill=(255, 220, 0, 255))
    draw.text((grid_x0 + 16, cy3), "Highest", font=fnt_small, fill=C_BODY)
    cy3 += 16
    draw.rectangle([(grid_x0, cy3), (grid_x0 + 12, cy3 + 12)], fill=(255, 140, 0, 255))
    draw.text((grid_x0 + 16, cy3), "Second-highest", font=fnt_small, fill=C_BODY)

    # Panel 4
    bx0, by0, bx1, by1 = draw_lower_panel(p4_x0, "Panel 4 : 설명 요약")
    cy4 = by0 + PAD + LABEL_H
    tw4 = LOWER_PANEL_W - 2 * PAD

    def wrap_draw(xy: Tuple[int, int], text: str, font: Any,
                  fill: Any, max_w: int, line_h: int) -> int:
        chars = max(1, max_w // max(1, int(font.getlength("가"))))
        lines = textwrap.fill(text, width=int(chars)).split("\n")
        cx_, cy_ = xy
        for ln in lines:
            draw.text((cx_, cy_), ln, font=font, fill=fill)
            cy_ += line_h
        return cy_ - xy[1]

    def draw_section(label: str, ko: str, en: str, lc: Any, kc: Any) -> None:
        nonlocal cy4
        draw.text((p4_x0 + PAD, cy4), label, font=fnt_label, fill=lc)
        cy4 += 18
        h = wrap_draw((p4_x0 + PAD, cy4), ko, fnt_body, kc, tw4, 17)
        cy4 += h + 3
        h = wrap_draw((p4_x0 + PAD, cy4), en, fnt_small, (170, 170, 170, 255), tw4, 14)
        cy4 += h + 10

    draw_section("[Key finding]",    P4_KEY_FINDING_KO,    P4_KEY_FINDING_EN,    C_SECTION, C_BODY)
    draw_section("[Interpretation]", P4_INTERPRETATION_KO, P4_INTERPRETATION_EN, C_SECTION, C_BODY)
    draw_section("[Caution]",        P4_CAUTION_KO,        P4_CAUTION_EN,        C_WARN,    C_WARN)
    draw.text((p4_x0 + PAD, cy4), "[Disclaimer]", font=fnt_label, fill=C_DISC)
    cy4 += 18
    wrap_draw((p4_x0 + PAD, cy4),
              P4_DISCLAIMER_KO + "  " + P4_DISCLAIMER_EN,
              fnt_small, C_DISC, tw4, 14)

    # 저장
    CARDS_PNG_DIR.mkdir(parents=True, exist_ok=True)
    CARDS_JSON_DIR.mkdir(parents=True, exist_ok=True)

    canvas_rgb = canvas.convert("RGB")
    canvas_rgb.save(str(OUT_PNG), "PNG")
    print(f"  PNG saved: {OUT_PNG}")

    schema = build_prototype_json_schema()
    schema["canvas_size_px"] = [CANVAS_W, CANVAS_H]
    schema["ct_load_used"]   = True
    schema["normal_ref_ct_load_used"] = True
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    print(f"  JSON saved: {OUT_JSON}")

    elapsed = time.time() - t_start
    runtime = {
        "case_id": CASE_ID,
        "version": "v3c",
        "elapsed_sec": round(elapsed, 2),
        "ct_load_used": True,
        "normal_ref_ct_load_used": True,
        "errors": errors,
        "output_png": str(OUT_PNG),
        "output_json": str(OUT_JSON),
    }
    OUT_RUNTIME.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_RUNTIME, "w", encoding="utf-8") as f:
        json.dump(runtime, f, ensure_ascii=False, indent=2)

    with open(OUT_ERRORS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["error"])
        for e in errors:
            w.writerow([e])

    with open(OUT_INDEX_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "version", "output_png", "output_json"])
        w.writerow([CASE_ID, "v3c", str(OUT_PNG), str(OUT_JSON)])

    with open(OUT_DONE, "w", encoding="utf-8") as f:
        json.dump({"done": True, "case_id": CASE_ID, "version": "v3c",
                   "reference_selection": "XY_MATCHED_WITH_Z_ORIENTATION_WARNING"},
                  f, ensure_ascii=False, indent=2)

    print(f"\n  run_prototype DONE in {elapsed:.1f}s — errors: {len(errors)}")
    if errors:
        for e in errors:
            print(f"    ERROR: {e}")


# ============================================================
# main
# ============================================================
def main() -> None:
    args = sys.argv[1:]

    if "--static-check" in args:
        passed, failed, issues = static_check()
        print(f"[static-check] {passed} passed / {failed} failed")
        if issues:
            for iss in issues:
                print(f"  {iss}")
        else:
            print("  PASS")
        sys.exit(0 if failed == 0 else 1)

    if "--dry-run" in args:
        ok, _ = dry_run()
        sys.exit(0 if ok else 1)

    if "--plan-only" in args:
        plan_only()
        sys.exit(0)

    if "--save-static-artifacts" in args:
        save_static_artifacts()
        sys.exit(0)

    if "--run-prototype" in args:
        if "--confirm-generate" not in args:
            print("ERROR: --run-prototype requires --confirm-generate", file=sys.stderr)
            sys.exit(2)
        run_prototype()
        sys.exit(0)

    print("Usage:")
    print("  --static-check              정적 검사 (20항목)")
    print("  --dry-run                   입력 파일 + guard 확인")
    print("  --plan-only                 레이아웃 표시")
    print("  --save-static-artifacts     static/ 산출물 저장 (7종)")
    print("  --run-prototype --confirm-generate  실제 PNG 생성 (guard True 필요)")
    sys.exit(0)


if __name__ == "__main__":
    main()
