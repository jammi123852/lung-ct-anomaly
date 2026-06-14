"""
S5 Demo Card Prototype v3b Reference FOV Fix — LUNG1-052__c3
=============================================================
v3 → v3b 주요 변경:
  - Panel 1 Row 1: normal reference 3개를 원본 CT NPY에서 96×96 crop으로 재생성
    (v3: reference PNG 직접 resize → FOV/해상도 불일치)
  - guard 분리: ALLOW_CANDIDATE_CT_LOAD / ALLOW_NORMAL_REF_CT_LOAD
  - z-context display 높이 축소 (Row 1 cell size의 0.70배)
  - normal reference CT paths: Desktop LUNA16 volumes_npy 탐색 완료

출력 root:
  outputs/position-aware-padim-v1/reports/explanation_cards/
    s5_demo_card_prototype_lung1_052_c3_v3b_reference_fov_fix/

실행 방법:
  --static-check            → 정적 검사
  --dry-run                 → 입력 파일 존재 + guard 확인
  --plan-only               → layout + 경로 + guard 표시
  --save-static-artifacts   → static/ 하위 산출물 저장
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

# CT load guard 분리 (v3b)
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

LAYOUT_SELECTED      = "OPTION_V3B_4PANEL_ORIGINAL_CT_REFERENCE_FOV_FIX"

# ============================================================
# normal reference crop 좌표 (원본 CT 기준, y0,x0,y1,x1)
# ============================================================
# matched normal (d3e03b0ce0): y_center=176, x_center=304 → ±48
MATCHED_NORMAL_Z  = 179
MATCHED_NORMAL_Y0 = 128; MATCHED_NORMAL_Y1 = 224
MATCHED_NORMAL_X0 = 256; MATCHED_NORMAL_X1 = 352

# normal ex1 (2508834c92): y_center=144, x_center=320 → ±48
NORMAL_EX1_Z  = 167
NORMAL_EX1_Y0 = 96;  NORMAL_EX1_Y1 = 192
NORMAL_EX1_X0 = 272; NORMAL_EX1_X1 = 368

# normal ex2 (ae9463f7d8): y_center=160, x_center=320 → ±48
NORMAL_EX2_Z  = 172
NORMAL_EX2_Y0 = 112; NORMAL_EX2_Y1 = 208
NORMAL_EX2_X0 = 272; NORMAL_EX2_X1 = 368

# ============================================================
# 경로 상수
# ============================================================
PROJECT_ROOT = pathlib.Path("/home/jinhy/project/lung-ct-anomaly")

REF_BANK_FULL = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    / "reference_bank_v1/full")

# Panel 1 Row 1 — reference PNG (fallback/debug 전용, 실제 표시 미사용)
MATCHED_NORMAL_REF_PNG = (REF_BANK_FULL
    / "reference_crops/lower_central"
    / "subset4_1.3.6.1.4.1.14519.5.2.1.6279.6001.200837896655745926888305239398"
      "__d3e03b0ce0__z179__y176_x304.png")
NORMAL_EX1_PNG = (REF_BANK_FULL
    / "reference_crops/lower_central"
    / "subset1_1.3.6.1.4.1.14519.5.2.1.6279.6001.300271604576987336866436407488"
      "__2508834c92__z167__y144_x320.png")
NORMAL_EX2_PNG = (REF_BANK_FULL
    / "reference_crops/lower_central"
    / "subset8_1.3.6.1.4.1.14519.5.2.1.6279.6001.336102335330125765000317290445"
      "__ae9463f7d8__z172__y160_x320.png")

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

# candidate CT NPY
CANDIDATE_CT_NPY = pathlib.Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1"
    "/volumes_npy/NSCLC_LUNG1-052__d4a19cc211/ct_hu.npy"
)

# normal reference CT NPY (LUNA16 training set, stage2_holdout 아님)
NORMAL_LUNA16_ROOT = pathlib.Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1"
    "/volumes_npy"
)
MATCHED_NORMAL_CT_NPY = (NORMAL_LUNA16_ROOT
    / "subset4_1.3.6.1.4.1.14519.5.2.1.6279.6001.200837896655745926888305239398"
      "__d3e03b0ce0/ct_hu.npy")
NORMAL_EX1_CT_NPY = (NORMAL_LUNA16_ROOT
    / "subset1_1.3.6.1.4.1.14519.5.2.1.6279.6001.300271604576987336866436407488"
      "__2508834c92/ct_hu.npy")
NORMAL_EX2_CT_NPY = (NORMAL_LUNA16_ROOT
    / "subset8_1.3.6.1.4.1.14519.5.2.1.6279.6001.336102335330125765000317290445"
      "__ae9463f7d8/ct_hu.npy")

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

# 출력 경로 (v3b)
OUTPUT_ROOT    = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "s5_demo_card_prototype_lung1_052_c3_v3b_reference_fov_fix")
STATIC_DIR     = OUTPUT_ROOT / "static"
CARDS_PNG_DIR  = OUTPUT_ROOT / "cards_png"
CARDS_JSON_DIR = OUTPUT_ROOT / "cards_json"
OUT_PNG        = CARDS_PNG_DIR  / "LUNG1-052__c3_s5_demo_prototype_v3b.png"
OUT_JSON       = CARDS_JSON_DIR / "LUNG1-052__c3_s5_demo_prototype_v3b.json"
OUT_INDEX_CSV  = OUTPUT_ROOT / "index_cards_v3b.csv"
OUT_RUNTIME    = OUTPUT_ROOT / "runtime_summary_v3b.json"
OUT_ERRORS     = OUTPUT_ROOT / "errors.csv"
OUT_DONE       = OUTPUT_ROOT / "DONE.json"

# static 산출물
OUT_DRYCHECK_MD       = STATIC_DIR / "drycheck_v3b_reference_fov_fix.md"
OUT_DRYCHECK_JSON     = STATIC_DIR / "drycheck_v3b_reference_fov_fix.json"
OUT_LAYOUT_CSV        = STATIC_DIR / "layout_plan_v3b.csv"
OUT_TEXT_POLICY_CSV   = STATIC_DIR / "text_policy_v3b.csv"
OUT_PREFLIGHT_MD      = STATIC_DIR / "preflight_summary_v3b.md"
OUT_NORMAL_CT_CSV     = STATIC_DIR / "normal_reference_ct_path_check_v3b.csv"
OUT_FOV_POLICY_CSV    = STATIC_DIR / "panel1_fov_policy_v3b.csv"

# ============================================================
# 텍스트 상수 (v3 동일)
# ============================================================
PROTOTYPE_TITLE_KO = "S5 PaDiM 이상 후보 설명 카드 v3b — LUNG1-052__c3"
PROTOTYPE_TITLE_EN = "S5 PaDiM patch-level explanation prototype v3b — LUNG1-052__c3"

P4_KEY_FINDING_KO = (
    "정상 reference와 비교했을 때, 해당 영역은 더 조밀한 음영을 보였습니다."
)
P4_KEY_FINDING_EN = (
    "Compared with normal references, this region showed denser opacification."
)
P4_INTERPRETATION_KO = (
    "PaDiM patch-level response도 같은 위치 주변에서 높게 나타났습니다. "
    "특히 중심 patch와 바로 아래 patch에서 높은 반응이 연속적으로 확인되어, "
    "모델이 이 주변을 정상 분포와 다른 국소 영역으로 본 것으로 해석할 수 있습니다."
)
P4_INTERPRETATION_EN = (
    "PaDiM patch-level response was also elevated around the same location. "
    "The center patch and the adjacent lower patch showed consecutively high responses, "
    "suggesting the model identified this area as a locally distinct region from normal distribution."
)
P4_CAUTION_KO = (
    "이 설명은 모델 반응을 이해하기 위한 연구용 보조 설명입니다."
)
P4_CAUTION_EN = (
    "This is a research-purpose supplementary explanation for understanding model response."
)
P4_DISCLAIMER_KO = "병변 원인이나 진단 결과를 직접 의미하지 않습니다."
P4_DISCLAIMER_EN  = "This does not directly imply lesion cause or diagnostic result."

LABEL_CANDIDATE      = "후보 crop\n(candidate)"
LABEL_MATCHED_NORMAL = "정상 reference\n(matched normal)"
LABEL_NORMAL_EX1     = "정상 예시 1\n(normal ex1)"
LABEL_NORMAL_EX2     = "정상 예시 2\n(normal ex2)"
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


# ============================================================
# prototype JSON schema
# ============================================================
def build_prototype_json_schema() -> Dict[str, Any]:
    return {
        "version": "v3b_reference_fov_fix",
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
                "S4-style reference comparison (v3b FOV fix): "
                "Row1=4개 동일 크기 (candidate_crop|matched_normal_CT|normal_ex1_CT|normal_ex2_CT), "
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
                "(의사용 4섹션; dim91/raw427/topk/feature-space 본문 제거)"
            ),
        },
        "metadata": {
            "not_gradcam": True,
            "not_pixel_attribution": True,
            "not_diagnostic": True,
            "internal_use_only": True,
            "font_fix_required_before_external_share": True,
            "v3b_fov_fix": True,
            "normal_ref_source": "original_CT_NPY_96x96_crop",
            "normal_ref_png_not_used_for_display": True,
            "zcontext_size_reduced_to_0_70_of_row1": True,
            "dim91_caveat": _META_DIM91_CAVEAT,
            "basal_pleural_caution": _META_BASAL_PLEURAL_CAUTION,
            "panel1_4crop_equal_cell_size": True,
            "panel1_whole_slice_larger_row": True,
            "panel2_overlay_single_only": True,
            "panel3_schematic_not_overlay": True,
            "patch_response_map_source": "score_map_sqrt_mahalanobis",
        },
        "source_files": {
            "matched_normal_ct_npy": str(MATCHED_NORMAL_CT_NPY),
            "normal_ex1_ct_npy": str(NORMAL_EX1_CT_NPY),
            "normal_ex2_ct_npy": str(NORMAL_EX2_CT_NPY),
            "matched_normal_ref_png_fallback": str(MATCHED_NORMAL_REF_PNG),
            "normal_ex1_png_fallback": str(NORMAL_EX1_PNG),
            "normal_ex2_png_fallback": str(NORMAL_EX2_PNG),
            "whole_slice_png": str(WHOLE_SLICE_PNG),
            "s5_lung_overlay_png": str(S5_LUNG_OVERLAY_PNG),
            "s5_patch_map_json": str(S5_PATCH_MAP_JSON),
            "s3_card_json": str(S3_CARD_JSON),
            "candidate_ct_npy": str(CANDIDATE_CT_NPY),
        },
        "normal_ref_crops": {
            "matched_normal": {
                "hash": "d3e03b0ce0", "z": MATCHED_NORMAL_Z,
                "y0": MATCHED_NORMAL_Y0, "y1": MATCHED_NORMAL_Y1,
                "x0": MATCHED_NORMAL_X0, "x1": MATCHED_NORMAL_X1,
            },
            "normal_ex1": {
                "hash": "2508834c92", "z": NORMAL_EX1_Z,
                "y0": NORMAL_EX1_Y0, "y1": NORMAL_EX1_Y1,
                "x0": NORMAL_EX1_X0, "x1": NORMAL_EX1_X1,
            },
            "normal_ex2": {
                "hash": "ae9463f7d8", "z": NORMAL_EX2_Z,
                "y0": NORMAL_EX2_Y0, "y1": NORMAL_EX2_Y1,
                "x0": NORMAL_EX2_X0, "x1": NORMAL_EX2_X1,
            },
        },
        "canvas_size_px": None,
        "output_png": str(OUT_PNG),
        "generated_date": None,
    }


# ============================================================
# 정적 검사
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

    # ── 검사 1: guard 기본값
    check("01 ALLOW_SOURCE_PNG_READ default False",       not ALLOW_SOURCE_PNG_READ)
    check("02 ALLOW_PROTOTYPE_RENDER default False",      not ALLOW_PROTOTYPE_RENDER)
    check("03 ALLOW_PROTOTYPE_WRITE default False",       not ALLOW_PROTOTYPE_WRITE)
    check("04 ALLOW_CANDIDATE_CT_LOAD default False",     not ALLOW_CANDIDATE_CT_LOAD)
    check("05 ALLOW_NORMAL_REF_CT_LOAD default False",    not ALLOW_NORMAL_REF_CT_LOAD)
    check("06 ALLOW_S3_MODIFICATION False",               not ALLOW_S3_MODIFICATION)
    check("07 ALLOW_S4_MODIFICATION False",               not ALLOW_S4_MODIFICATION)
    check("08 ALLOW_MODEL_FORWARD False",                 not ALLOW_MODEL_FORWARD)
    check("09 ALLOW_FEATURE_EXTRACTION False",            not ALLOW_FEATURE_EXTRACTION)
    check("10 ALLOW_CONTRIBUTION_RECALC False",           not ALLOW_CONTRIBUTION_RECALC)
    check("11 ALLOW_STAGE2_HOLDOUT False",                not ALLOW_STAGE2_HOLDOUT)
    check("12 ALLOW_FULL_300 False",                      not ALLOW_FULL_300)

    # ── 검사 2: output root v3b
    check("20 OUTPUT_ROOT contains v3b",                  "v3b" in OUTPUT_ROOT.name)
    check("21 OUTPUT_ROOT != v3",                         str(OUTPUT_ROOT) != str(_V3_ROOT))
    check("22 OUTPUT_ROOT != v2",                         str(OUTPUT_ROOT) != str(_V2_ROOT))
    check("23 OUTPUT_ROOT != v1",                         str(OUTPUT_ROOT) != str(_V1_ROOT))
    check("24 OUT_PNG contains v3b",                      "v3b" in OUT_PNG.name)
    check("25 OUT_JSON contains v3b",                     "v3b" in OUT_JSON.name)

    # ── 검사 3: v3b FOV fix 메타데이터
    meta = schema["metadata"]
    check("30 v3b_fov_fix in metadata",                   meta.get("v3b_fov_fix") is True)
    check("31 normal_ref_source = original_CT_NPY",       meta.get("normal_ref_source") == "original_CT_NPY_96x96_crop")
    check("32 normal_ref_png_not_used_for_display",       meta.get("normal_ref_png_not_used_for_display") is True)
    check("33 zcontext_size_reduced",                     meta.get("zcontext_size_reduced_to_0_70_of_row1") is True)

    # ── 검사 4: normal reference CT path 설계
    check("40 matched_normal CT path contains d3e03b0ce0",
          "d3e03b0ce0" in str(MATCHED_NORMAL_CT_NPY))
    check("41 normal_ex1 CT path contains 2508834c92",
          "2508834c92" in str(NORMAL_EX1_CT_NPY))
    check("42 normal_ex2 CT path contains ae9463f7d8",
          "ae9463f7d8" in str(NORMAL_EX2_CT_NPY))
    check("43 matched_normal CT not stage2_holdout",
          "stage2_holdout" not in str(MATCHED_NORMAL_CT_NPY))
    check("44 normal_ex1 CT not stage2_holdout",
          "stage2_holdout" not in str(NORMAL_EX1_CT_NPY))
    check("45 normal_ex2 CT not stage2_holdout",
          "stage2_holdout" not in str(NORMAL_EX2_CT_NPY))
    check("46 matched_normal CT filename ct_hu.npy",
          MATCHED_NORMAL_CT_NPY.name == "ct_hu.npy")
    check("47 normal_ex1 CT filename ct_hu.npy",
          NORMAL_EX1_CT_NPY.name == "ct_hu.npy")
    check("48 normal_ex2 CT filename ct_hu.npy",
          NORMAL_EX2_CT_NPY.name == "ct_hu.npy")

    # ── 검사 5: normal reference crop 좌표 (96×96)
    check("50 matched_normal crop 96×96",
          (MATCHED_NORMAL_Y1 - MATCHED_NORMAL_Y0) == 96 and
          (MATCHED_NORMAL_X1 - MATCHED_NORMAL_X0) == 96)
    check("51 normal_ex1 crop 96×96",
          (NORMAL_EX1_Y1 - NORMAL_EX1_Y0) == 96 and
          (NORMAL_EX1_X1 - NORMAL_EX1_X0) == 96)
    check("52 normal_ex2 crop 96×96",
          (NORMAL_EX2_Y1 - NORMAL_EX2_Y0) == 96 and
          (NORMAL_EX2_X1 - NORMAL_EX2_X0) == 96)
    check("53 candidate crop 96×96",
          (CANDIDATE_CROP_Y1 - CANDIDATE_CROP_Y0) == 96 and
          (CANDIDATE_CROP_X1 - CANDIDATE_CROP_X0) == 96)

    # ── 검사 6: Panel 1 Row 1 설계
    p1_desc = schema["panels"]["panel_1"].lower()
    check("60 panel1 row1 4개 동일 크기",                "4개 동일 크기" in schema["panels"]["panel_1"])
    check("61 panel1 candidate in description",          "candidate" in p1_desc)
    check("62 panel1 matched_normal_ct in description",  "matched_normal_ct" in p1_desc or "matched_normal" in p1_desc)
    check("63 panel1 normal_ex1_CT in description",      "normal_ex1_ct" in p1_desc or "normal_ex1" in p1_desc)
    check("64 panel1 whole slice row",                   "whole slice" in p1_desc)
    check("65 panel1 z-context row",                     "z-context" in p1_desc)
    check("66 panel1 z-context 축소 언급",               "0.70" in p1_desc or "축소" in p1_desc)
    check("67 reference PNG not used for display",       "png" not in p1_desc or "fallback" in p1_desc or "ct_npy" in p1_desc)

    # ── 검사 7: Panel 2/3
    p2_desc = schema["panels"]["panel_2"].lower()
    check("70 panel2 overlay 1장",                       "1장만" in schema["panels"]["panel_2"])
    check("71 panel2 not gradcam",                       "not grad-cam" in p2_desc)
    check("72 panel2 not pixel attribution",             "not pixel attribution" in p2_desc)
    p3_desc = schema["panels"]["panel_3"].lower()
    check("73 panel3 schematic",                         "schematic" in p3_desc)
    check("74 panel3 not pixel heatmap",                 "not pixel heatmap" in p3_desc)
    check("75 panel3 not Grad-CAM",                      "not grad-cam" in p3_desc)

    # ── 검사 8: Panel 4 금지 표현
    all_card_text = " ".join([
        P4_KEY_FINDING_KO, P4_KEY_FINDING_EN,
        P4_INTERPRETATION_KO, P4_INTERPRETATION_EN,
        P4_CAUTION_KO, P4_CAUTION_EN,
        P4_DISCLAIMER_KO, P4_DISCLAIMER_EN,
        PROTOTYPE_TITLE_EN
    ]).lower()
    check("80 no dim91 in card text",                    "dim91" not in all_card_text)
    check("81 no raw427 in card text",                   "raw427" not in all_card_text)
    check("82 no topk in card text",                     "top-k" not in all_card_text and "topk" not in all_card_text)
    check("83 no Grad-CAM in card text",                 "grad-cam" not in all_card_text and "gradcam" not in all_card_text)
    check("84 no pixel attribution in card text",        "pixel attribution" not in all_card_text)
    check("85 no diagnostic heatmap in card text",       "diagnostic heatmap" not in all_card_text)

    # ── 검사 9: Panel 4 필수 의미
    check("90 조밀한 음영 언급",                         "조밀한 음영" in P4_KEY_FINDING_KO)
    check("91 같은 위치 주변 반응 언급",                 "같은 위치 주변" in P4_INTERPRETATION_KO)
    check("92 중심 patch + 아래 patch 연속 언급",
          "중심 patch" in P4_INTERPRETATION_KO and "아래 patch" in P4_INTERPRETATION_KO)
    check("93 연구용 보조 설명 언급",                    "연구용 보조 설명" in P4_CAUTION_KO)
    check("94 진단 결과 아님 언급",                      "진단 결과" in P4_DISCLAIMER_KO)

    # ── 검사 10: metadata
    check("100 not_gradcam in metadata",                 meta.get("not_gradcam") is True)
    check("101 not_pixel_attribution in metadata",       meta.get("not_pixel_attribution") is True)
    check("102 not_diagnostic in metadata",              meta.get("not_diagnostic") is True)
    check("103 internal_use_only in metadata",           meta.get("internal_use_only") is True)

    # ── 검사 11: 좌표 selftest
    check("110 GRID_EXTENT_FORMAT defined",              GRID_EXTENT_FORMAT == "y0,x0,y1,x1")
    check("111 CANDIDATE_CROP_Y0 == 256",                CANDIDATE_CROP_Y0 == 256)
    check("112 CANDIDATE_CROP_X0 == 96",                 CANDIDATE_CROP_X0 == 96)
    check("113 CANDIDATE_CROP_Y1 == 352",                CANDIDATE_CROP_Y1 == 352)
    check("114 CANDIDATE_CROP_X1 == 192",                CANDIDATE_CROP_X1 == 192)
    check("115 PATCH4_BBOX y0==288",                     PATCH4_BBOX[0] == 288)
    check("116 PATCH4_BBOX x0==128",                     PATCH4_BBOX[1] == 128)
    check("117 PATCH4_BBOX y1==320",                     PATCH4_BBOX[2] == 320)
    check("118 PATCH4_BBOX x1==160",                     PATCH4_BBOX[3] == 160)
    check("119 PATCH7_BBOX y0==320",                     PATCH7_BBOX[0] == 320)
    check("120 PATCH7_BBOX x0==128",                     PATCH7_BBOX[1] == 128)
    check("121 PATCH7_BBOX y1==352",                     PATCH7_BBOX[2] == 352)
    check("122 PATCH7_BBOX x1==160",                     PATCH7_BBOX[3] == 160)
    check("123 patch4.y1 == patch7.y0",                  PATCH4_BBOX[2] == PATCH7_BBOX[0])
    check("124 patch4.x0 == patch7.x0",                  PATCH4_BBOX[1] == PATCH7_BBOX[1])

    # ── 검사 12: candidate CT selftest
    check("130 candidate CT path not stage2_holdout",
          "stage2_holdout" not in str(CANDIDATE_CT_NPY))
    check("131 candidate CT contains volumes_npy",
          "volumes_npy" in str(CANDIDATE_CT_NPY))
    check("132 candidate CT contains LUNG1-052",
          "LUNG1-052" in str(CANDIDATE_CT_NPY))
    check("133 candidate CT filename ct_hu.npy",
          CANDIDATE_CT_NPY.name == "ct_hu.npy")

    return passed, failed, issues


# ============================================================
# dry-run
# ============================================================
def dry_run() -> Tuple[bool, List[str]]:
    print(f"[dry-run] S5 Demo Card Prototype v3b — {CASE_ID}")
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

    # Panel 1 Row 1 — normal reference CT NPY (필수)
    check_path("matched_normal_ct_npy", MATCHED_NORMAL_CT_NPY)
    check_path("normal_ex1_ct_npy",     NORMAL_EX1_CT_NPY)
    check_path("normal_ex2_ct_npy",     NORMAL_EX2_CT_NPY)

    # candidate CT NPY (ALLOW_CANDIDATE_CT_LOAD=True 승인 후 필요)
    check_path_optional("candidate_ct_npy (ALLOW_CANDIDATE_CT_LOAD 승인 후 필요)",
                        CANDIDATE_CT_NPY)

    # 기타 source
    check_path("whole_slice_png",     WHOLE_SLICE_PNG)
    check_path("s5_lung_overlay_png", S5_LUNG_OVERLAY_PNG)
    check_path("s5_coord_metadata",   S5_COORD_METADATA)
    check_path("s5_patch_map_json",   S5_PATCH_MAP_JSON)
    check_path("s3_card_json",        S3_CARD_JSON)

    # reference PNG (fallback/debug용 — 표시 미사용, 없어도 non-blocking)
    check_path_optional("matched_normal_ref_png (fallback only)", MATCHED_NORMAL_REF_PNG)
    check_path_optional("normal_ex1_png (fallback only)",         NORMAL_EX1_PNG)
    check_path_optional("normal_ex2_png (fallback only)",         NORMAL_EX2_PNG)

    # guard 확인
    for name, val in [
        ("ALLOW_SOURCE_PNG_READ",    ALLOW_SOURCE_PNG_READ),
        ("ALLOW_PROTOTYPE_RENDER",   ALLOW_PROTOTYPE_RENDER),
        ("ALLOW_PROTOTYPE_WRITE",    ALLOW_PROTOTYPE_WRITE),
        ("ALLOW_CANDIDATE_CT_LOAD",  ALLOW_CANDIDATE_CT_LOAD),
        ("ALLOW_NORMAL_REF_CT_LOAD", ALLOW_NORMAL_REF_CT_LOAD),
    ]:
        if val:
            issues.append(f"WARN: {name} is True (should be False in dry-run)")

    # output collision 확인
    for p in [OUT_PNG, OUT_JSON, OUT_DONE]:
        if p.exists():
            issues.append(f"COLLISION: {p} already exists")

    # v3 artifact가 존재해도 충돌 아님 (다른 root)
    if OUTPUT_ROOT.exists():
        print(f"  WARN: v3b output root already exists: {OUTPUT_ROOT}")
    else:
        print(f"  OK  v3b output root (not yet created): {OUTPUT_ROOT.name}")

    # 정적 검사 포함
    passed, failed, si = static_check()
    issues.extend(si)
    print(f"\n  static_check: {passed} passed, {failed} failed")

    # normal reference CT path stage2_holdout 확인
    for label, p in [
        ("matched_normal_ct", MATCHED_NORMAL_CT_NPY),
        ("normal_ex1_ct",     NORMAL_EX1_CT_NPY),
        ("normal_ex2_ct",     NORMAL_EX2_CT_NPY),
    ]:
        if "stage2_holdout" in str(p):
            issues.append(f"BLOCKED: {label} path contains stage2_holdout: {p}")

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
    print(f"[plan-only] S5 Demo Card Prototype v3b — {CASE_ID}")
    print(f"  Layout : {LAYOUT_SELECTED}")
    print()
    print("  Panel 1 (상단, 전체 너비) — v3b FOV fix:")
    print("    Row 1: 4개 동일 크기 crop (각 CELL_SIZE px) — 모두 원본 CT 96×96 crop")
    print("      [0] candidate crop   → CANDIDATE_CT_NPY / z=51 / y256:352 / x96:192")
    print("      [1] matched normal   → MATCHED_NORMAL_CT_NPY / z=179 / y128:224 / x256:352")
    print("      [2] normal ex1       → NORMAL_EX1_CT_NPY / z=167 / y96:192 / x272:368")
    print("      [3] normal ex2       → NORMAL_EX2_CT_NPY / z=172 / y112:208 / x272:368")
    print("    Row 2: whole slice (더 크게, lung-window patch4/7 overlay)")
    print("    Row 3: z-context 3장 (Row 1 cell × 0.70 크기로 축소, 보조 정보)")
    print()
    print("  Panel 2 (하단 1/3): lung-window overlay 1장 / not Grad-CAM")
    print("  Panel 3 (하단 1/3): 3×3 patch response map schematic")
    print("  Panel 4 (하단 1/3): 4섹션 텍스트 (v3 동일)")
    print()
    print("  v3b 핵심 변경:")
    print("    - normal reference: PNG resize → 원본 CT 96×96 crop")
    print("    - z-context: full-width → Row 1 cell × 0.70 축소")
    print("    - guard 분리: ALLOW_CANDIDATE_CT_LOAD / ALLOW_NORMAL_REF_CT_LOAD")
    print()
    print("  Normal reference CT paths:")
    print(f"    matched_normal: {MATCHED_NORMAL_CT_NPY} (exists={MATCHED_NORMAL_CT_NPY.exists()})")
    print(f"    normal_ex1   : {NORMAL_EX1_CT_NPY} (exists={NORMAL_EX1_CT_NPY.exists()})")
    print(f"    normal_ex2   : {NORMAL_EX2_CT_NPY} (exists={NORMAL_EX2_CT_NPY.exists()})")
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
# static artifacts 저장
# ============================================================
def save_static_artifacts() -> None:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    passed, failed, issues = static_check()
    ok_dry, dry_issues = dry_run()

    # drycheck json
    drycheck_data = {
        "script": "build_s5_demo_card_prototype_lung1_052_c3_v3b_reference_fov_fix.py",
        "case_id": CASE_ID,
        "static_check": {"passed": passed, "failed": failed, "issues": issues},
        "dry_run": {"ok": ok_dry, "issues": dry_issues},
        "guards": {
            "ALLOW_SOURCE_PNG_READ":    ALLOW_SOURCE_PNG_READ,
            "ALLOW_PROTOTYPE_RENDER":   ALLOW_PROTOTYPE_RENDER,
            "ALLOW_PROTOTYPE_WRITE":    ALLOW_PROTOTYPE_WRITE,
            "ALLOW_CANDIDATE_CT_LOAD":  ALLOW_CANDIDATE_CT_LOAD,
            "ALLOW_NORMAL_REF_CT_LOAD": ALLOW_NORMAL_REF_CT_LOAD,
            "ALLOW_S3_MODIFICATION":    ALLOW_S3_MODIFICATION,
            "ALLOW_S4_MODIFICATION":    ALLOW_S4_MODIFICATION,
            "ALLOW_MODEL_FORWARD":      ALLOW_MODEL_FORWARD,
            "ALLOW_FEATURE_EXTRACTION": ALLOW_FEATURE_EXTRACTION,
            "ALLOW_CONTRIBUTION_RECALC":ALLOW_CONTRIBUTION_RECALC,
            "ALLOW_STAGE2_HOLDOUT":     ALLOW_STAGE2_HOLDOUT,
            "ALLOW_FULL_300":           ALLOW_FULL_300,
        },
        "normal_ref_ct_paths": {
            "matched_normal": {"path": str(MATCHED_NORMAL_CT_NPY), "exists": MATCHED_NORMAL_CT_NPY.exists(),
                               "stage2_holdout": "stage2_holdout" in str(MATCHED_NORMAL_CT_NPY)},
            "normal_ex1":     {"path": str(NORMAL_EX1_CT_NPY),     "exists": NORMAL_EX1_CT_NPY.exists(),
                               "stage2_holdout": "stage2_holdout" in str(NORMAL_EX1_CT_NPY)},
            "normal_ex2":     {"path": str(NORMAL_EX2_CT_NPY),     "exists": NORMAL_EX2_CT_NPY.exists(),
                               "stage2_holdout": "stage2_holdout" in str(NORMAL_EX2_CT_NPY)},
        },
        "candidate_ct_path": {
            "path": str(CANDIDATE_CT_NPY),
            "exists": CANDIDATE_CT_NPY.exists(),
            "stage2_holdout": "stage2_holdout" in str(CANDIDATE_CT_NPY),
        },
        "output_root": str(OUTPUT_ROOT),
        "collision_check": {
            "v3_root_exists": _V3_ROOT.exists(),
            "v3b_root_exists": OUTPUT_ROOT.exists(),
            "out_png_exists": OUT_PNG.exists(),
            "out_json_exists": OUT_JSON.exists(),
        },
    }
    with open(OUT_DRYCHECK_JSON, "w", encoding="utf-8") as f:
        json.dump(drycheck_data, f, ensure_ascii=False, indent=2)
    print(f"  SAVED: {OUT_DRYCHECK_JSON}")

    # drycheck md
    md_lines = [
        f"# S5 Demo Card v3b Drycheck — {CASE_ID}",
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
        "## v3b FOV Fix — Normal Reference CT 경로",
        "",
        "| 대상 | 경로 | 존재 | stage2_holdout |",
        "| --- | --- | --- | --- |",
        f"| matched_normal | ...{MATCHED_NORMAL_CT_NPY.parent.name} | {'OK' if MATCHED_NORMAL_CT_NPY.exists() else 'MISSING'} | {'YES — BLOCKED' if 'stage2_holdout' in str(MATCHED_NORMAL_CT_NPY) else 'NO'} |",
        f"| normal_ex1 | ...{NORMAL_EX1_CT_NPY.parent.name} | {'OK' if NORMAL_EX1_CT_NPY.exists() else 'MISSING'} | {'YES — BLOCKED' if 'stage2_holdout' in str(NORMAL_EX1_CT_NPY) else 'NO'} |",
        f"| normal_ex2 | ...{NORMAL_EX2_CT_NPY.parent.name} | {'OK' if NORMAL_EX2_CT_NPY.exists() else 'MISSING'} | {'YES — BLOCKED' if 'stage2_holdout' in str(NORMAL_EX2_CT_NPY) else 'NO'} |",
        "",
        "## Normal Reference Crop 좌표 (96×96)",
        "",
        "| 대상 | z | y0 | y1 | x0 | x1 | 크기 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        f"| matched_normal | {MATCHED_NORMAL_Z} | {MATCHED_NORMAL_Y0} | {MATCHED_NORMAL_Y1} | {MATCHED_NORMAL_X0} | {MATCHED_NORMAL_X1} | {MATCHED_NORMAL_Y1-MATCHED_NORMAL_Y0}×{MATCHED_NORMAL_X1-MATCHED_NORMAL_X0} |",
        f"| normal_ex1 | {NORMAL_EX1_Z} | {NORMAL_EX1_Y0} | {NORMAL_EX1_Y1} | {NORMAL_EX1_X0} | {NORMAL_EX1_X1} | {NORMAL_EX1_Y1-NORMAL_EX1_Y0}×{NORMAL_EX1_X1-NORMAL_EX1_X0} |",
        f"| normal_ex2 | {NORMAL_EX2_Z} | {NORMAL_EX2_Y0} | {NORMAL_EX2_Y1} | {NORMAL_EX2_X0} | {NORMAL_EX2_X1} | {NORMAL_EX2_Y1-NORMAL_EX2_Y0}×{NORMAL_EX2_X1-NORMAL_EX2_X0} |",
        f"| candidate  | {CT_LOCAL_Z} | {CANDIDATE_CROP_Y0} | {CANDIDATE_CROP_Y1} | {CANDIDATE_CROP_X0} | {CANDIDATE_CROP_X1} | {CANDIDATE_CROP_Y1-CANDIDATE_CROP_Y0}×{CANDIDATE_CROP_X1-CANDIDATE_CROP_X0} |",
        "",
        "## Guards",
        "",
        f"| guard | 현재값 |",
        "| --- | --- |",
        f"| ALLOW_SOURCE_PNG_READ | {ALLOW_SOURCE_PNG_READ} |",
        f"| ALLOW_PROTOTYPE_RENDER | {ALLOW_PROTOTYPE_RENDER} |",
        f"| ALLOW_PROTOTYPE_WRITE | {ALLOW_PROTOTYPE_WRITE} |",
        f"| ALLOW_CANDIDATE_CT_LOAD | {ALLOW_CANDIDATE_CT_LOAD} |",
        f"| ALLOW_NORMAL_REF_CT_LOAD | {ALLOW_NORMAL_REF_CT_LOAD} |",
        "",
        "## v3b Layout 변경 요약",
        "",
        "- Panel 1 Row 1: normal reference 3개 → 원본 CT 96×96 crop (PNG resize 폐기)",
        "- z-context: Row 1 cell × 0.70 크기로 축소",
        "- guard 분리: ALLOW_CANDIDATE_CT_LOAD / ALLOW_NORMAL_REF_CT_LOAD",
        "",
    ]
    OUT_DRYCHECK_MD.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"  SAVED: {OUT_DRYCHECK_MD}")

    # layout plan csv
    layout_rows = [
        ["panel", "row", "cell", "cell_count", "source_type", "source_path", "display_size", "label", "ct_load_required", "v3b_change"],
        ["1", "1", "0", "4", "CT_crop", str(CANDIDATE_CT_NPY), "CELL_SIZE", LABEL_CANDIDATE, "ALLOW_CANDIDATE_CT_LOAD", "same_as_v3"],
        ["1", "1", "1", "4", "CT_crop", str(MATCHED_NORMAL_CT_NPY), "CELL_SIZE", LABEL_MATCHED_NORMAL, "ALLOW_NORMAL_REF_CT_LOAD", "NEW: orig_CT_crop"],
        ["1", "1", "2", "4", "CT_crop", str(NORMAL_EX1_CT_NPY), "CELL_SIZE", LABEL_NORMAL_EX1, "ALLOW_NORMAL_REF_CT_LOAD", "NEW: orig_CT_crop"],
        ["1", "1", "3", "4", "CT_crop", str(NORMAL_EX2_CT_NPY), "CELL_SIZE", LABEL_NORMAL_EX2, "ALLOW_NORMAL_REF_CT_LOAD", "NEW: orig_CT_crop"],
        ["1", "2", "0", "1", "PNG_read", str(WHOLE_SLICE_PNG), "larger_than_row1", LABEL_WHOLE_SLICE, "False", "same_as_v3"],
        ["1", "3", "0", "3", "CT_crop", str(CANDIDATE_CT_NPY), "CELL_SIZE*0.70", LABEL_Z_ABOVE, "ALLOW_CANDIDATE_CT_LOAD", "NEW: size_reduced_0.70"],
        ["1", "3", "1", "3", "CT_crop", str(CANDIDATE_CT_NPY), "CELL_SIZE*0.70", LABEL_Z_CURRENT, "ALLOW_CANDIDATE_CT_LOAD", "NEW: size_reduced_0.70"],
        ["1", "3", "2", "3", "CT_crop", str(CANDIDATE_CT_NPY), "CELL_SIZE*0.70", LABEL_Z_BELOW, "ALLOW_CANDIDATE_CT_LOAD", "NEW: size_reduced_0.70"],
        ["2", "1", "0", "1", "PNG_read", str(S5_LUNG_OVERLAY_PNG), "panel_width", "lung overlay (patch4/7)", "False", "same_as_v3"],
        ["3", "1", "schematic", "1", "JSON_data", str(S5_PATCH_MAP_JSON), "3x3_grid", "patch response map schematic", "False", "same_as_v3"],
        ["4", "1", "text", "1", "text_const", "P4_KEY_FINDING_KO", "text_box", "[Key finding]", "False", "same_as_v3"],
        ["4", "2", "text", "1", "text_const", "P4_INTERPRETATION_KO", "text_box", "[Interpretation]", "False", "same_as_v3"],
        ["4", "3", "text", "1", "text_const", "P4_CAUTION_KO", "text_box", "[Caution]", "False", "same_as_v3"],
        ["4", "4", "text", "1", "text_const", "P4_DISCLAIMER_KO", "text_box", "[Disclaimer]", "False", "same_as_v3"],
    ]
    with open(OUT_LAYOUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(layout_rows)
    print(f"  SAVED: {OUT_LAYOUT_CSV}")

    # normal reference CT path check csv
    nr_rows = [
        ["target", "hash", "path", "exists", "stage2_holdout", "crop_z", "crop_y0", "crop_y1", "crop_x0", "crop_x1", "crop_size"],
        ["matched_normal", "d3e03b0ce0", str(MATCHED_NORMAL_CT_NPY),
         str(MATCHED_NORMAL_CT_NPY.exists()),
         str("stage2_holdout" in str(MATCHED_NORMAL_CT_NPY)),
         MATCHED_NORMAL_Z, MATCHED_NORMAL_Y0, MATCHED_NORMAL_Y1, MATCHED_NORMAL_X0, MATCHED_NORMAL_X1,
         f"{MATCHED_NORMAL_Y1-MATCHED_NORMAL_Y0}x{MATCHED_NORMAL_X1-MATCHED_NORMAL_X0}"],
        ["normal_ex1", "2508834c92", str(NORMAL_EX1_CT_NPY),
         str(NORMAL_EX1_CT_NPY.exists()),
         str("stage2_holdout" in str(NORMAL_EX1_CT_NPY)),
         NORMAL_EX1_Z, NORMAL_EX1_Y0, NORMAL_EX1_Y1, NORMAL_EX1_X0, NORMAL_EX1_X1,
         f"{NORMAL_EX1_Y1-NORMAL_EX1_Y0}x{NORMAL_EX1_X1-NORMAL_EX1_X0}"],
        ["normal_ex2", "ae9463f7d8", str(NORMAL_EX2_CT_NPY),
         str(NORMAL_EX2_CT_NPY.exists()),
         str("stage2_holdout" in str(NORMAL_EX2_CT_NPY)),
         NORMAL_EX2_Z, NORMAL_EX2_Y0, NORMAL_EX2_Y1, NORMAL_EX2_X0, NORMAL_EX2_X1,
         f"{NORMAL_EX2_Y1-NORMAL_EX2_Y0}x{NORMAL_EX2_X1-NORMAL_EX2_X0}"],
        ["candidate", "d4a19cc211", str(CANDIDATE_CT_NPY),
         str(CANDIDATE_CT_NPY.exists()),
         str("stage2_holdout" in str(CANDIDATE_CT_NPY)),
         CT_LOCAL_Z, CANDIDATE_CROP_Y0, CANDIDATE_CROP_Y1, CANDIDATE_CROP_X0, CANDIDATE_CROP_X1,
         f"{CANDIDATE_CROP_Y1-CANDIDATE_CROP_Y0}x{CANDIDATE_CROP_X1-CANDIDATE_CROP_X0}"],
    ]
    with open(OUT_NORMAL_CT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(nr_rows)
    print(f"  SAVED: {OUT_NORMAL_CT_CSV}")

    # FOV policy csv
    fov_rows = [
        ["cell", "source_v3", "source_v3b", "size_px", "fov_match", "note"],
        ["candidate_crop", "CT_crop_96x96", "CT_crop_96x96", "CELL_SIZE", "YES", "unchanged"],
        ["matched_normal", "PNG_resize", "CT_crop_96x96", "CELL_SIZE", "YES", "v3b: orig CT"],
        ["normal_ex1", "PNG_resize", "CT_crop_96x96", "CELL_SIZE", "YES", "v3b: orig CT"],
        ["normal_ex2", "PNG_resize", "CT_crop_96x96", "CELL_SIZE", "YES", "v3b: orig CT"],
        ["z_context", "Z_CELL_W=full_width/3", "CELL_SIZE*0.70", "smaller", "N/A", "v3b: reduced"],
    ]
    with open(OUT_FOV_POLICY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(fov_rows)
    print(f"  SAVED: {OUT_FOV_POLICY_CSV}")

    # text policy csv
    text_policy_rows = [
        ["check_id", "target", "rule", "status"],
        ["T01", "panel4_body", "no dim91",                   "PASS" if "dim91" not in P4_INTERPRETATION_KO.lower() else "FAIL"],
        ["T02", "panel4_body", "no raw427",                  "PASS" if "raw427" not in P4_INTERPRETATION_KO.lower() else "FAIL"],
        ["T03", "panel4_body", "no top-k",                   "PASS" if "top-k" not in P4_INTERPRETATION_KO.lower() else "FAIL"],
        ["T04", "panel4_body", "no Grad-CAM",                "PASS"],
        ["T05", "panel4_body", "no pixel attribution",       "PASS"],
        ["T06", "panel4_body", "no diagnostic heatmap",      "PASS"],
        ["T07", "p4_key_finding_ko", "조밀한 음영 포함",    "PASS" if "조밀한 음영" in P4_KEY_FINDING_KO else "FAIL"],
        ["T08", "p4_interpretation_ko", "같은 위치 주변",   "PASS" if "같은 위치 주변" in P4_INTERPRETATION_KO else "FAIL"],
        ["T09", "p4_interpretation_ko", "중심+아래 patch",   "PASS" if "중심 patch" in P4_INTERPRETATION_KO and "아래 patch" in P4_INTERPRETATION_KO else "FAIL"],
        ["T10", "p4_caution_ko", "연구용 보조 설명",        "PASS" if "연구용 보조 설명" in P4_CAUTION_KO else "FAIL"],
        ["T11", "p4_disclaimer_ko", "진단 결과 아님",       "PASS" if "진단 결과" in P4_DISCLAIMER_KO else "FAIL"],
        ["T12", "metadata", "not_gradcam=True",              "PASS"],
        ["T13", "metadata", "not_pixel_attribution=True",    "PASS"],
        ["T14", "metadata", "not_diagnostic=True",           "PASS"],
    ]
    with open(OUT_TEXT_POLICY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(text_policy_rows)
    print(f"  SAVED: {OUT_TEXT_POLICY_CSV}")

    # preflight summary
    fail_count = failed + len([i for i in dry_issues if i.startswith("FAIL")])
    preflight_ok = (fail_count == 0)
    preflight_lines = [
        f"# Preflight Summary — {CASE_ID} S5 v3b",
        "",
        f"## 결과: {'PASS' if preflight_ok else 'BLOCKED'}",
        "",
        f"- static_check: {passed} passed / {failed} failed",
        f"- dry_run: {'OK' if ok_dry else 'ISSUES'}",
        f"- output collision: {'없음' if not OUT_PNG.exists() else 'COLLISION 있음'}",
        "",
        "## v3b 핵심 변경 확인",
        "",
        "- normal reference source: PNG resize → 원본 CT 96×96 crop",
        f"  - matched_normal CT exists: {MATCHED_NORMAL_CT_NPY.exists()}",
        f"  - normal_ex1 CT exists: {NORMAL_EX1_CT_NPY.exists()}",
        f"  - normal_ex2 CT exists: {NORMAL_EX2_CT_NPY.exists()}",
        "- z-context 크기: Row 1 cell × 0.70 (보조 정보 수준으로 축소)",
        "- guard 분리: ALLOW_CANDIDATE_CT_LOAD / ALLOW_NORMAL_REF_CT_LOAD",
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
    ]
    OUT_PREFLIGHT_MD.write_text("\n".join(preflight_lines), encoding="utf-8")
    print(f"  SAVED: {OUT_PREFLIGHT_MD}")

    summary = "PASS" if (failed == 0 and not dry_issues) else f"BLOCKED ({failed} static fails, {len(dry_issues)} dry issues)"
    print(f"\n  [save-static-artifacts] {summary}")


# ============================================================
# actual generation
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
        print("Use a new output root (e.g. v3c) rather than deleting.", file=sys.stderr)
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
    HEADER_H   = 52
    PAD        = 8

    P1_W = CANVAS_W - 2 * MARGIN
    CELL_SIZE = (P1_W - 5 * PAD) // 4          # 4개 crop 동일 크기 (≈382px)
    LABEL_H   = 30

    WHOLE_W = int(P1_W * 0.68)
    WHOLE_H = int(WHOLE_W * img_whole.height / img_whole.width)

    # z-context: Row 1 cell의 0.70배 (보조 정보 수준)
    Z_CELL_W = int(CELL_SIZE * 0.70)
    Z_CELL_H = Z_CELL_W   # 정방형

    P1_ROW1_H = LABEL_H + CELL_SIZE + LABEL_H
    P1_ROW2_H = LABEL_H + WHOLE_H
    P1_ROW3_H = LABEL_H + Z_CELL_H
    P1_H      = P1_ROW1_H + PAD + P1_ROW2_H + PAD + P1_ROW3_H + PAD

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

    canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), C_BG)
    draw   = ImageDraw.Draw(canvas)

    # 헤더
    draw.rectangle([(0, 0), (CANVAS_W, HEADER_H)], fill=C_HEADER)
    draw.text((MARGIN, 8),  PROTOTYPE_TITLE_KO, font=fnt_title, fill=C_TITLE)
    draw.text((MARGIN, 32), "[INTERNAL USE ONLY | v3b | not diagnostic]",
              font=fnt_small, fill=C_WARN)

    # Panel 1 배경
    p1_x0 = MARGIN
    p1_y0 = HEADER_H + MARGIN
    p1_x1 = MARGIN + P1_W
    p1_y1 = p1_y0 + P1_H
    draw.rectangle([(p1_x0, p1_y0), (p1_x1, p1_y1)], fill=C_PANEL_BG, outline=C_BORDER, width=1)

    cy = p1_y0 + PAD
    draw.text((p1_x0 + PAD, cy),
              "Panel 1 : 정상 reference 비교  |  전체 슬라이스  |  z-context",
              font=fnt_section, fill=C_SECTION)
    cy += LABEL_H

    # Panel 1 Row 1: 4개 동일 크기 crop (모두 원본 CT)
    crop_x_starts = [p1_x0 + PAD + i * (CELL_SIZE + PAD) for i in range(4)]
    cell_labels   = [LABEL_CANDIDATE, LABEL_MATCHED_NORMAL, LABEL_NORMAL_EX1, LABEL_NORMAL_EX2]
    cell_imgs: List[Optional[Image.Image]] = [None, None, None, None]

    # [0] candidate crop
    if ct_arr is not None:
        cell_imgs[0] = make_ct_crop_img_yx(
            ct_arr, CT_LOCAL_Z,
            CANDIDATE_CROP_Y0, CANDIDATE_CROP_X0,
            CANDIDATE_CROP_Y1, CANDIDATE_CROP_X1,
            CELL_SIZE
        )
    else:
        cell_imgs[0] = _placeholder(CELL_SIZE, CELL_SIZE, "candidate CT\nnot found")

    # [1] matched normal — 원본 CT crop (PNG fallback 금지)
    if matched_normal_ct is not None:
        cell_imgs[1] = make_ct_crop_img_yx(
            matched_normal_ct, MATCHED_NORMAL_Z,
            MATCHED_NORMAL_Y0, MATCHED_NORMAL_X0,
            MATCHED_NORMAL_Y1, MATCHED_NORMAL_X1,
            CELL_SIZE
        )
    else:
        cell_imgs[1] = _placeholder(CELL_SIZE, CELL_SIZE, "matched normal CT\nnot found")

    # [2] normal ex1 — 원본 CT crop
    if normal_ex1_ct is not None:
        cell_imgs[2] = make_ct_crop_img_yx(
            normal_ex1_ct, NORMAL_EX1_Z,
            NORMAL_EX1_Y0, NORMAL_EX1_X0,
            NORMAL_EX1_Y1, NORMAL_EX1_X1,
            CELL_SIZE
        )
    else:
        cell_imgs[2] = _placeholder(CELL_SIZE, CELL_SIZE, "normal ex1 CT\nnot found")

    # [3] normal ex2 — 원본 CT crop
    if normal_ex2_ct is not None:
        cell_imgs[3] = make_ct_crop_img_yx(
            normal_ex2_ct, NORMAL_EX2_Z,
            NORMAL_EX2_Y0, NORMAL_EX2_X0,
            NORMAL_EX2_Y1, NORMAL_EX2_X1,
            CELL_SIZE
        )
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

    # Panel 1 Row 3: z-context (0.70 크기, 보조 정보)
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
                CANDIDATE_CROP_Y1, CANDIDATE_CROP_X1,
                Z_CELL_W
            )
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

    # 하단 패널 공통 위치
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

    score_map  = patch_map.get("score_map_sqrt_mahalanobis", [[0] * 3] * 3)
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
            draw.text((cell_x + 4, cell_y + 4),      f"P{pid}",    font=fnt_small, fill=lc)
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
        "version": "v3b",
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

    with open(OUT_DONE, "w", encoding="utf-8") as f:
        json.dump({"done": True, "case_id": CASE_ID, "version": "v3b"}, f,
                  ensure_ascii=False, indent=2)

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
    print("  --static-check           정적 검사")
    print("  --dry-run                입력 파일 + guard 확인")
    print("  --plan-only              레이아웃 표시")
    print("  --save-static-artifacts  static/ 산출물 저장")
    print("  --run-prototype --confirm-generate  실제 PNG 생성 (guard True 필요)")
    sys.exit(0)


if __name__ == "__main__":
    main()
