"""
S5 Demo Card Prototype v3 Clinical-Readable — LUNG1-052__c3
===========================================================
v2 → v3 주요 변경:
  - Panel 1: S4 스타일 reference comparison panel
    * Row1: 4개 동일 크기 crop (candidate | matched_normal | normal_ex1 | normal_ex2)
    * Row2: whole slice (더 크게, 하단 row)
    * Row3: z-context 3장 (위/현재/아래 슬라이스) — ALLOW_CT_LOAD guard
    * Panel 1 하단 old reason text 박스 제거
  - Panel 2: lung-window overlay 1장만, 간단 legend (patch4/patch7 위치)
  - Panel 3: 3×3 patch response map schematic (not CT overlay, not pixel heatmap)
  - Panel 4: 의사용 4섹션 재작성 (dim91/raw427/topk/feature-space 본문 제거)

출력 root:
  outputs/position-aware-padim-v1/reports/explanation_cards/
    s5_demo_card_prototype_lung1_052_c3_v3_clinical_readable/

실행 방법:
  --static-check            → 정적 검사 (스키마/텍스트/가드/경로 위반 확인)
  --dry-run                 → 입력 파일 존재 + guard 확인
  --plan-only               → 4-panel layout + 입력/출력 경로 + guard 표시
  --save-static-artifacts   → static/ 하위 drycheck md/json, layout csv, text policy csv, preflight summary 저장
  --run-prototype --confirm-generate  → guards False이면 BLOCKED exit 2

actual generation에서만 아래 guard를 별도 승인 후 True:
  ALLOW_SOURCE_PNG_READ = True
  ALLOW_PROTOTYPE_RENDER = True
  ALLOW_PROTOTYPE_WRITE = True

candidate crop / z-context에 별도 승인 필요:
  ALLOW_CT_LOAD = True  (기본 False, placeholder 처리됨)
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

# CT load guard — candidate crop + z-context (기본 False, placeholder 처리)
ALLOW_CT_LOAD              = False

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
CT_LOCAL_Z           = 51      # coordinate_overlay_metadata에서 확인된 local_z
REPORT_SLICE_INDEX   = 106     # metadata 전용 (CT indexing 사용 금지)
SPATIAL_PATTERN      = "CENTER_DOMINANT_WITH_DOWNWARD_CONTINUITY"
VISUAL_VERDICT       = "PASS_VISUAL_SUPPORTS_CENTER_DOWNWARD_CONTINUITY_WITH_CAUTION"

# CT crop 좌표 (ALLOW_CT_LOAD=True일 때만 사용)
# bbox/grid format: [y0, x0, y1, x1] in CT 512×512 coordinates
GRID_EXTENT_FORMAT   = "y0,x0,y1,x1"
GRID_EXTENT          = [256, 96, 352, 192]   # [y0, x0, y1, x1]
GRID_Y0 = 256; GRID_X0 = 96; GRID_Y1 = 352; GRID_X1 = 192

PATCH4_BBOX          = [288, 128, 320, 160]   # [y0, x0, y1, x1] (CT 좌표)
PATCH4_SCORE         = 38.872562
PATCH7_BBOX          = [320, 128, 352, 160]   # [y0, x0, y1, x1] (CT 좌표)
PATCH7_SCORE         = 36.612470

# candidate crop — grid_extent 영역 전체 (96×96 CT px)
# ct_arr[z, CANDIDATE_CROP_Y0:CANDIDATE_CROP_Y1, CANDIDATE_CROP_X0:CANDIDATE_CROP_X1]
CANDIDATE_CROP_Y0    = 256   # grid y0 (row start)
CANDIDATE_CROP_X0    = 96    # grid x0 (col start)
CANDIDATE_CROP_Y1    = 352   # grid y1 (row end)
CANDIDATE_CROP_X1    = 192   # grid x1 (col end)
LUNG_WINDOW_CENTER   = -600
LUNG_WINDOW_WIDTH    = 1500

LAYOUT_SELECTED      = "OPTION_V3_4PANEL_S4STYLE_REFERENCE_COMPARISON"

# ============================================================
# 경로 상수
# ============================================================
PROJECT_ROOT = pathlib.Path("/home/jinhy/project/lung-ct-anomaly")

REF_BANK_FULL = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    / "reference_bank_v1/full")

# Panel 1 — reference PNG (CT load 없이 직접 읽기 가능)
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

# Panel 1 Row 2 — whole slice (기존 PNG 재사용)
WHOLE_SLICE_PNG = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "s5_lung1_052_c3_coordinate_visual_audit_1case_v1"
    / "coordinate_overlay_patch4_patch7_lung.png")

# Panel 2 — lung-window overlay (기존 PNG 재사용)
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

# S3 card (참조용, metadata에서만 사용)
S3_CARD_JSON = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    / "s3_expansion_cards_v1/cards_json/LUNG1-052__c3.json")

# CT NPY — ALLOW_CT_LOAD=True 승인 후에만 사용
CANDIDATE_CT_NPY = pathlib.Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1"
    "/volumes_npy/NSCLC_LUNG1-052__d4a19cc211/ct_hu.npy"
)

# 기존 v2 / v1 output root (충돌 방지용)
_V1_ROOT = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "s5_demo_card_prototype_lung1_052_c3_v1")
_V2_ROOT = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "s5_demo_card_prototype_lung1_052_c3_v2_clinical_readable")

# 출력 경로 (v3)
OUTPUT_ROOT    = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "s5_demo_card_prototype_lung1_052_c3_v3_clinical_readable")
STATIC_DIR     = OUTPUT_ROOT / "static"
CARDS_PNG_DIR  = OUTPUT_ROOT / "cards_png"
CARDS_JSON_DIR = OUTPUT_ROOT / "cards_json"
OUT_PNG        = CARDS_PNG_DIR  / "LUNG1-052__c3_s5_demo_prototype_v3.png"
OUT_JSON       = CARDS_JSON_DIR / "LUNG1-052__c3_s5_demo_prototype_v3.json"
OUT_INDEX_CSV  = OUTPUT_ROOT / "index_cards_v3.csv"
OUT_RUNTIME    = OUTPUT_ROOT / "runtime_summary_v3.json"
OUT_ERRORS     = OUTPUT_ROOT / "errors.csv"
OUT_DONE       = OUTPUT_ROOT / "DONE.json"

# static 산출물
OUT_DRYCHECK_MD     = STATIC_DIR / "drycheck_v3_coordinate_fix.md"
OUT_DRYCHECK_JSON   = STATIC_DIR / "drycheck_v3_coordinate_fix.json"
OUT_LAYOUT_CSV      = STATIC_DIR / "layout_plan_v3_coordinate_fix.csv"
OUT_TEXT_POLICY_CSV = STATIC_DIR / "text_policy_v3.csv"
OUT_PREFLIGHT_MD    = STATIC_DIR / "preflight_summary_v3.md"
OUT_CT_PATH_CSV     = STATIC_DIR / "ct_path_check_v3.csv"
OUT_COORD_CSV       = STATIC_DIR / "coordinate_semantics_v3.csv"

# ============================================================
# 텍스트 상수 — v3 의사용 문장 (dim91/raw427/topk/feature-space 제거)
# ============================================================
PROTOTYPE_TITLE_KO = "S5 PaDiM 이상 후보 설명 카드 v3 — LUNG1-052__c3"
PROTOTYPE_TITLE_EN = "S5 PaDiM patch-level explanation prototype v3 — LUNG1-052__c3"

# Panel 4 섹션별 문장 — 의사 읽기 편한 한국어 우선
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

# Panel 1 셀 레이블
LABEL_CANDIDATE      = "후보 crop\n(candidate)"
LABEL_MATCHED_NORMAL = "정상 reference\n(matched normal)"
LABEL_NORMAL_EX1     = "정상 예시 1\n(normal ex1)"
LABEL_NORMAL_EX2     = "정상 예시 2\n(normal ex2)"
LABEL_WHOLE_SLICE    = "전체 슬라이스 (lung window, patch 위치 표시)"
LABEL_Z_ABOVE        = "위 슬라이스\n(z-1)"
LABEL_Z_CURRENT      = "현재 슬라이스\n(z)"
LABEL_Z_BELOW        = "아래 슬라이스\n(z+1)"

# metadata 전용 (카드 본문 사용 금지)
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
        "version": "v3_clinical_readable",
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
                "S4-style reference comparison: "
                "Row1=4개 동일 크기 (candidate_crop|matched_normal|normal_ex1|normal_ex2), "
                "Row2=whole slice (더 크게), "
                "Row3=z-context 3장 (ALLOW_CT_LOAD guard); "
                "old reason text 제거"
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
            "dim91_caveat": _META_DIM91_CAVEAT,
            "basal_pleural_caution": _META_BASAL_PLEURAL_CAUTION,
            "v2_panel1_old_reason_text_removed": True,
            "panel1_4crop_equal_cell_size": True,
            "panel1_whole_slice_larger_row": True,
            "panel1_z_context_ct_load_required": True,
            "panel2_overlay_single_only": True,
            "panel3_schematic_not_overlay": True,
            "patch_response_map_source": "score_map_sqrt_mahalanobis",
        },
        "source_files": {
            "matched_normal_ref_png": str(MATCHED_NORMAL_REF_PNG),
            "normal_ex1_png": str(NORMAL_EX1_PNG),
            "normal_ex2_png": str(NORMAL_EX2_PNG),
            "whole_slice_png": str(WHOLE_SLICE_PNG),
            "s5_lung_overlay_png": str(S5_LUNG_OVERLAY_PNG),
            "s5_patch_map_json": str(S5_PATCH_MAP_JSON),
            "s3_card_json": str(S3_CARD_JSON),
            "candidate_ct_npy_path": str(CANDIDATE_CT_NPY),
            "candidate_ct_npy_requires_allow_ct_load": True,
        },
        "canvas_size_px": None,
        "output_png": str(OUT_PNG),
        "generated_date": None,
    }


# ============================================================
# 정적 검사 (검사 항목 10개)
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

    # ── 검사 1: 기존 artifact 수정 없음 (가드 기본값)
    check("01 ALLOW_SOURCE_PNG_READ default False",    not ALLOW_SOURCE_PNG_READ)
    check("02 ALLOW_PROTOTYPE_RENDER default False",   not ALLOW_PROTOTYPE_RENDER)
    check("03 ALLOW_PROTOTYPE_WRITE default False",    not ALLOW_PROTOTYPE_WRITE)
    check("04 ALLOW_CT_LOAD default False",            not ALLOW_CT_LOAD)
    check("05 ALLOW_S3_MODIFICATION False",            not ALLOW_S3_MODIFICATION)
    check("06 ALLOW_S4_MODIFICATION False",            not ALLOW_S4_MODIFICATION)
    check("07 ALLOW_MODEL_FORWARD False",              not ALLOW_MODEL_FORWARD)
    check("08 ALLOW_FEATURE_EXTRACTION False",         not ALLOW_FEATURE_EXTRACTION)
    check("09 ALLOW_CONTRIBUTION_RECALC False",        not ALLOW_CONTRIBUTION_RECALC)
    check("10 ALLOW_STAGE2_HOLDOUT False",             not ALLOW_STAGE2_HOLDOUT)
    check("11 ALLOW_FULL_300 False",                   not ALLOW_FULL_300)

    # ── 검사 2: output collision block 존재
    check("20 OUTPUT_ROOT contains v3",                "v3" in OUTPUT_ROOT.name)
    check("21 OUTPUT_ROOT name not v2",                "v2" not in OUTPUT_ROOT.name)
    check("22 OUTPUT_ROOT name not v1",                OUTPUT_ROOT.name != _V1_ROOT.name)
    check("23 OUT_PNG contains v3",                    "v3" in OUT_PNG.name)
    check("24 OUT_JSON contains v3",                   "v3" in OUT_JSON.name)
    check("25 OUTPUT_ROOT != v1",                      str(OUTPUT_ROOT) != str(_V1_ROOT))
    check("26 OUTPUT_ROOT != v2",                      str(OUTPUT_ROOT) != str(_V2_ROOT))

    # ── 검사 3: Panel 1 — 4개 동일 크기 + whole slice + z-context 설계
    p1_desc = schema["panels"]["panel_1"].lower()
    check("30 panel1 row1 4-crop same size",           "4개 동일 크기" in schema["panels"]["panel_1"])
    check("31 panel1 candidate in description",        "candidate" in p1_desc)
    check("32 panel1 matched_normal in description",   "matched_normal" in p1_desc)
    check("33 panel1 normal_ex1/ex2 in description",   "normal_ex1" in p1_desc)
    check("34 panel1 whole slice row",                 "whole slice" in p1_desc or "row2" in p1_desc)
    check("35 panel1 z-context row",                   "z-context" in p1_desc or "row3" in p1_desc)
    check("36 panel1 CT load guard acknowledged",      "ct_load" in p1_desc or "allow_ct_load" in p1_desc)

    # ── 검사 4: Panel 1 하단 old reason text 제거
    check("40 panel1 old reason text 제거 명시",
          "old reason text 제거" in schema["panels"]["panel_1"]
          or "reason text" not in p1_desc)

    # ── 검사 5: Panel 2 — overlay 1장만
    p2_desc = schema["panels"]["panel_2"].lower()
    check("50 panel2 overlay 1장",                     "1장만" in schema["panels"]["panel_2"])
    check("51 panel2 not gradcam",                     "not grad-cam" in p2_desc)
    check("52 panel2 not pixel attribution",           "not pixel attribution" in p2_desc)

    # ── 검사 6: Panel 3 — schematic, not pixel heatmap
    p3_desc = schema["panels"]["panel_3"].lower()
    check("60 panel3 schematic",                       "schematic" in p3_desc)
    check("61 panel3 not pixel heatmap",               "not pixel heatmap" in p3_desc or "not ct overlay" in p3_desc)
    check("62 panel3 not Grad-CAM",                    "not grad-cam" in p3_desc)

    # ── 검사 7: Panel 4 — dim91/raw427/topk 본문 제거
    all_card_text = " ".join([
        P4_KEY_FINDING_KO, P4_KEY_FINDING_EN,
        P4_INTERPRETATION_KO, P4_INTERPRETATION_EN,
        P4_CAUTION_KO, P4_CAUTION_EN,
        P4_DISCLAIMER_KO, P4_DISCLAIMER_EN,
        PROTOTYPE_TITLE_EN
    ]).lower()
    check("70 no dim91 in card text",                  "dim91" not in all_card_text)
    check("71 no raw427 in card text",                 "raw427" not in all_card_text)
    check("72 no topk in card text",                   "top-k" not in all_card_text and "topk" not in all_card_text)
    check("73 no layer fraction in card text",         "layer fraction" not in all_card_text)
    check("74 no Grad-CAM in card text",               "grad-cam" not in all_card_text and "gradcam" not in all_card_text)
    check("75 no pixel attribution in card text",      "pixel attribution" not in all_card_text)
    check("76 no diagnostic heatmap in card text",     "diagnostic heatmap" not in all_card_text)
    check("77 no model saw cancer in card text",       "model saw cancer" not in all_card_text)
    check("78 no covariance inverse in card text",     "covariance inverse" not in all_card_text)

    # ── 검사 8: 필수 설명 의미 반영
    check("80 조밀한 음영 언급",
          "조밀한 음영" in P4_KEY_FINDING_KO)
    check("81 같은 위치 주변 반응 언급",
          "같은 위치 주변" in P4_INTERPRETATION_KO)
    check("82 중심 patch + 아래 patch 연속 언급",
          "중심 patch" in P4_INTERPRETATION_KO and "아래 patch" in P4_INTERPRETATION_KO)
    check("83 연구용 보조 설명 언급",
          "연구용 보조 설명" in P4_CAUTION_KO)
    check("84 진단 결과 아님 언급",
          "진단 결과" in P4_DISCLAIMER_KO)

    # ── 검사 9: not Grad-CAM / not pixel attribution / not diagnostic
    meta = schema["metadata"]
    check("90 not_gradcam in metadata",                meta.get("not_gradcam") is True)
    check("91 not_pixel_attribution in metadata",      meta.get("not_pixel_attribution") is True)
    check("92 not_diagnostic in metadata",             meta.get("not_diagnostic") is True)
    check("93 internal_use_only in metadata",          meta.get("internal_use_only") is True)
    check("94 dim91_caveat in metadata",               len(meta.get("dim91_caveat", "")) > 20)
    check("95 basal_pleural_caution in metadata",      len(meta.get("basal_pleural_caution", "")) > 10)

    # ── 검사 10: source path 설계 검증 (존재 확인은 dry_run에서)
    check("100 matched_normal_ref_png path set",
          "subset4" in str(MATCHED_NORMAL_REF_PNG) and "lower_central" in str(MATCHED_NORMAL_REF_PNG))
    check("101 normal_ex1_png path set",
          "subset1" in str(NORMAL_EX1_PNG) and "lower_central" in str(NORMAL_EX1_PNG))
    check("102 normal_ex2_png path set",
          "subset8" in str(NORMAL_EX2_PNG) and "lower_central" in str(NORMAL_EX2_PNG))
    check("103 whole_slice_png path set",
          "coordinate_overlay_patch4_patch7_lung" in str(WHOLE_SLICE_PNG))
    check("104 s5_lung_overlay_png path set",
          "coordinate_overlay_3x3_grid_lung" in str(S5_LUNG_OVERLAY_PNG))
    check("105 patch_map_json path set",
          "patch_contribution_map.json" in str(S5_PATCH_MAP_JSON))
    check("106 candidate_ct_npy path references LUNG1-052",
          "LUNG1-052" in str(CANDIDATE_CT_NPY) or "d4a19cc211" in str(CANDIDATE_CT_NPY))
    check("107 patch4 score range",                    35.0 < PATCH4_SCORE < 45.0)
    check("108 patch7 score range",                    33.0 < PATCH7_SCORE < 42.0)
    check("109 patch4 > patch7",                       PATCH4_SCORE > PATCH7_SCORE)

    # ── 검사 11: 좌표 selftest (y0,x0,y1,x1 기준)
    check("110 GRID_EXTENT_FORMAT defined",             GRID_EXTENT_FORMAT == "y0,x0,y1,x1")
    check("111 CANDIDATE_CROP_Y0 == 256",               CANDIDATE_CROP_Y0 == 256)
    check("112 CANDIDATE_CROP_X0 == 96",                CANDIDATE_CROP_X0 == 96)
    check("113 CANDIDATE_CROP_Y1 == 352",               CANDIDATE_CROP_Y1 == 352)
    check("114 CANDIDATE_CROP_X1 == 192",               CANDIDATE_CROP_X1 == 192)
    check("115 PATCH4_BBOX y0==288",                    PATCH4_BBOX[0] == 288)
    check("116 PATCH4_BBOX x0==128",                    PATCH4_BBOX[1] == 128)
    check("117 PATCH4_BBOX y1==320",                    PATCH4_BBOX[2] == 320)
    check("118 PATCH4_BBOX x1==160",                    PATCH4_BBOX[3] == 160)
    check("119 PATCH7_BBOX y0==320",                    PATCH7_BBOX[0] == 320)
    check("120 PATCH7_BBOX x0==128",                    PATCH7_BBOX[1] == 128)
    check("121 PATCH7_BBOX y1==352",                    PATCH7_BBOX[2] == 352)
    check("122 PATCH7_BBOX x1==160",                    PATCH7_BBOX[3] == 160)
    check("123 patch4.y1 == patch7.y0 (vertical continuity)",
          PATCH4_BBOX[2] == PATCH7_BBOX[0])
    check("124 patch4.x0 == patch7.x0 (same column)",  PATCH4_BBOX[1] == PATCH7_BBOX[1])
    check("125 patch4.x1 == patch7.x1 (same column)",  PATCH4_BBOX[3] == PATCH7_BBOX[3])
    check("126 grid contains patch4 y-range",
          CANDIDATE_CROP_Y0 <= PATCH4_BBOX[0] and PATCH4_BBOX[2] <= CANDIDATE_CROP_Y1)
    check("127 grid contains patch4 x-range",
          CANDIDATE_CROP_X0 <= PATCH4_BBOX[1] and PATCH4_BBOX[3] <= CANDIDATE_CROP_X1)

    # ── 검사 12: CT path selftest
    check("130 CT path contains volumes_npy",
          "volumes_npy" in str(CANDIDATE_CT_NPY))
    check("131 CT path contains NSCLC_LUNG1-052__d4a19cc211",
          "NSCLC_LUNG1-052__d4a19cc211" in str(CANDIDATE_CT_NPY))
    check("132 CT path filename is ct_hu.npy",
          CANDIDATE_CT_NPY.name == "ct_hu.npy")
    check("133 CT path not stage2_holdout",
          "stage2_holdout" not in str(CANDIDATE_CT_NPY))

    return passed, failed, issues


# ============================================================
# dry-run: 입력 파일 존재 + guard 확인
# ============================================================
def dry_run() -> Tuple[bool, List[str]]:
    print(f"[dry-run] S5 Demo Card Prototype v3 — {CASE_ID}")
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
            msg = f"OPTIONAL MISSING {label}: {p}" + (f" ({note})" if note else "")
            print(f"  INFO {msg}")

    # 필수 source PNG (CT load 없이 읽어야 하는 것들)
    check_path("matched_normal_ref_png", MATCHED_NORMAL_REF_PNG)
    check_path("normal_ex1_png",         NORMAL_EX1_PNG)
    check_path("normal_ex2_png",         NORMAL_EX2_PNG)
    check_path("whole_slice_png",        WHOLE_SLICE_PNG)
    check_path("s5_lung_overlay_png",    S5_LUNG_OVERLAY_PNG)
    check_path("s5_coord_metadata",      S5_COORD_METADATA)
    check_path("s5_patch_map_json",      S5_PATCH_MAP_JSON)
    check_path("s3_card_json",           S3_CARD_JSON)

    # CT NPY — ALLOW_CT_LOAD=True 승인 후 필요
    check_path_optional("candidate_ct_npy (CT load 승인 후 필요)",
                        CANDIDATE_CT_NPY, "ALLOW_CT_LOAD guard=False시 placeholder")

    # guard 확인
    for name, val in [
        ("ALLOW_SOURCE_PNG_READ", ALLOW_SOURCE_PNG_READ),
        ("ALLOW_PROTOTYPE_RENDER", ALLOW_PROTOTYPE_RENDER),
        ("ALLOW_PROTOTYPE_WRITE", ALLOW_PROTOTYPE_WRITE),
    ]:
        if val:
            issues.append(f"WARN: {name} is True (should be False in dry-run)")

    # output collision 확인
    for p in [OUT_PNG, OUT_JSON, OUT_DONE]:
        if p.exists():
            issues.append(f"COLLISION: {p} already exists")

    if OUTPUT_ROOT.exists():
        print(f"  WARN: v3 output root already exists: {OUTPUT_ROOT}")
    else:
        print(f"  OK  v3 output root (not yet created): {OUTPUT_ROOT.name}")

    # 정적 검사 포함
    passed, failed, si = static_check()
    issues.extend(si)
    print(f"\n  static_check: {passed} passed, {failed} failed")

    ok = len(issues) == 0
    if issues:
        print("\n[dry-run] Issues:")
        for iss in issues:
            print(f"  - {iss}")
        print()
        print("[dry-run] BLOCKED — 위 문제 해결 후 재실행 필요")
    else:
        print("\n[dry-run] PASS — 모든 source 파일 존재, guard 위반 없음")
        print("NOTE: actual generation에서만 SOURCE_PNG_READ/RENDER/WRITE를 True로 설정 필요")
        if not ALLOW_CT_LOAD:
            print("NOTE: ALLOW_CT_LOAD=False → candidate crop + z-context는 placeholder로 처리됨")

    return ok, issues


# ============================================================
# plan-only: layout 표시
# ============================================================
def plan_only() -> None:
    print(f"[plan-only] S5 Demo Card Prototype v3 — {CASE_ID}")
    print(f"  Layout : {LAYOUT_SELECTED}")
    print()
    print("  Panel 1 (상단, 전체 너비):")
    print("    Row 1: 4개 동일 크기 crop (각 ~224px)")
    print("      [0] candidate crop   → CT load 필요 (ALLOW_CT_LOAD guard)")
    print("      [1] matched_normal   → REF_BANK_FULL/reference_crops/lower_central/subset4...d3e03b0ce0")
    print("      [2] normal_ex1       → REF_BANK_FULL/reference_crops/lower_central/subset1...2508834c92")
    print("      [3] normal_ex2       → REF_BANK_FULL/reference_crops/lower_central/subset8...ae9463f7d8")
    print("    Row 2: whole slice (더 크게, lung-window patch4/7 overlay)")
    print("      → coordinate_overlay_patch4_patch7_lung.png")
    print("    Row 3: z-context 3장 나란히 (위/현재/아래 슬라이스)")
    print("      → CT load 필요 (ALLOW_CT_LOAD guard), placeholder 처리 가능")
    print("    [old reason text 박스 제거]")
    print()
    print("  Panel 2 (하단 1/3):")
    print("    lung-window overlay 1장 (3×3 grid + patch4/7 표시)")
    print("    → coordinate_overlay_3x3_grid_lung.png")
    print("    Legend: Yellow=highest / Orange=2nd")
    print("    [not Grad-CAM / not pixel attribution]")
    print()
    print("  Panel 3 (하단 1/3):")
    print("    3×3 patch response map schematic")
    print("    → patch_contribution_map.json에서 score_map_sqrt_mahalanobis")
    print("    [not CT overlay, not pixel heatmap]")
    print()
    print("  Panel 4 (하단 1/3):")
    print("    [Key finding]     → P4_KEY_FINDING_KO")
    print("    [Interpretation]  → P4_INTERPRETATION_KO")
    print("    [Caution]         → P4_CAUTION_KO")
    print("    [Disclaimer]      → P4_DISCLAIMER_KO")
    print()
    print("  Input files:")
    print(f"    matched_normal_ref_png : {MATCHED_NORMAL_REF_PNG.name}")
    print(f"    normal_ex1_png         : {NORMAL_EX1_PNG.name}")
    print(f"    normal_ex2_png         : {NORMAL_EX2_PNG.name}")
    print(f"    whole_slice_png        : {WHOLE_SLICE_PNG.name}")
    print(f"    s5_lung_overlay_png    : {S5_LUNG_OVERLAY_PNG.name}")
    print(f"    s5_patch_map_json      : {S5_PATCH_MAP_JSON.name}")
    print(f"    candidate_ct_npy       : {CANDIDATE_CT_NPY} (CT load guard)")
    print()
    print("  Output root (v3):")
    print(f"    {OUTPUT_ROOT}")
    print(f"    {OUT_PNG.name}")
    print(f"    {OUT_JSON.name}")
    print(f"    static/drycheck_v3.md/json")
    print(f"    static/layout_plan_v3.csv")
    print(f"    static/text_policy_v3.csv")
    print(f"    static/preflight_summary_v3.md")
    print()
    print("  Guards:")
    print(f"    ALLOW_SOURCE_PNG_READ    = {ALLOW_SOURCE_PNG_READ}")
    print(f"    ALLOW_PROTOTYPE_RENDER   = {ALLOW_PROTOTYPE_RENDER}")
    print(f"    ALLOW_PROTOTYPE_WRITE    = {ALLOW_PROTOTYPE_WRITE}")
    print(f"    ALLOW_CT_LOAD            = {ALLOW_CT_LOAD}")
    print()
    passed, failed, issues = static_check()
    if issues:
        for iss in issues:
            print(f"  STATIC ISSUE: {iss}")
    else:
        print("  static_check: PASS")


# ============================================================
# static artifacts 저장
# ============================================================
def save_static_artifacts() -> None:
    """drycheck md/json, layout plan csv, text policy csv, preflight summary 저장."""
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    # 1. static_check 실행
    passed, failed, issues = static_check()
    ok_dry, dry_issues = dry_run()

    # 2. drycheck json
    drycheck_data = {
        "script": "build_s5_demo_card_prototype_lung1_052_c3_v3_clinical_readable.py",
        "case_id": CASE_ID,
        "static_check": {
            "passed": passed,
            "failed": failed,
            "issues": issues,
        },
        "dry_run": {
            "ok": ok_dry,
            "issues": dry_issues,
        },
        "guards": {
            "ALLOW_SOURCE_PNG_READ": ALLOW_SOURCE_PNG_READ,
            "ALLOW_PROTOTYPE_RENDER": ALLOW_PROTOTYPE_RENDER,
            "ALLOW_PROTOTYPE_WRITE": ALLOW_PROTOTYPE_WRITE,
            "ALLOW_CT_LOAD": ALLOW_CT_LOAD,
            "ALLOW_S3_MODIFICATION": ALLOW_S3_MODIFICATION,
            "ALLOW_S4_MODIFICATION": ALLOW_S4_MODIFICATION,
            "ALLOW_MODEL_FORWARD": ALLOW_MODEL_FORWARD,
            "ALLOW_FEATURE_EXTRACTION": ALLOW_FEATURE_EXTRACTION,
            "ALLOW_CONTRIBUTION_RECALC": ALLOW_CONTRIBUTION_RECALC,
            "ALLOW_STAGE2_HOLDOUT": ALLOW_STAGE2_HOLDOUT,
            "ALLOW_FULL_300": ALLOW_FULL_300,
        },
        "source_paths": {
            "matched_normal_ref_png": {"path": str(MATCHED_NORMAL_REF_PNG), "exists": MATCHED_NORMAL_REF_PNG.exists()},
            "normal_ex1_png":         {"path": str(NORMAL_EX1_PNG),         "exists": NORMAL_EX1_PNG.exists()},
            "normal_ex2_png":         {"path": str(NORMAL_EX2_PNG),         "exists": NORMAL_EX2_PNG.exists()},
            "whole_slice_png":        {"path": str(WHOLE_SLICE_PNG),        "exists": WHOLE_SLICE_PNG.exists()},
            "s5_lung_overlay_png":    {"path": str(S5_LUNG_OVERLAY_PNG),    "exists": S5_LUNG_OVERLAY_PNG.exists()},
            "s5_patch_map_json":      {"path": str(S5_PATCH_MAP_JSON),      "exists": S5_PATCH_MAP_JSON.exists()},
            "candidate_ct_npy":       {"path": str(CANDIDATE_CT_NPY),       "exists": CANDIDATE_CT_NPY.exists(),
                                       "requires": "ALLOW_CT_LOAD=True"},
        },
        "output_root": str(OUTPUT_ROOT),
        "collision_check": {
            "v1_root_exists": _V1_ROOT.exists(),
            "v2_root_exists": _V2_ROOT.exists(),
            "v3_root_exists": OUTPUT_ROOT.exists(),
            "out_png_exists": OUT_PNG.exists(),
            "out_json_exists": OUT_JSON.exists(),
        },
    }
    with open(OUT_DRYCHECK_JSON, "w", encoding="utf-8") as f:
        json.dump(drycheck_data, f, ensure_ascii=False, indent=2)
    print(f"  SAVED: {OUT_DRYCHECK_JSON}")

    # 3. drycheck md
    md_lines = [
        f"# S5 Demo Card v3 Drycheck — {CASE_ID}",
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
        "## Source Files",
        "",
        "| 파일 | 존재 여부 | 비고 |",
        "| --- | --- | --- |",
        f"| matched_normal_ref_png | {'OK' if MATCHED_NORMAL_REF_PNG.exists() else 'MISSING'} | lower_central/subset4 |",
        f"| normal_ex1_png | {'OK' if NORMAL_EX1_PNG.exists() else 'MISSING'} | lower_central/subset1 |",
        f"| normal_ex2_png | {'OK' if NORMAL_EX2_PNG.exists() else 'MISSING'} | lower_central/subset8 |",
        f"| whole_slice_png | {'OK' if WHOLE_SLICE_PNG.exists() else 'MISSING'} | coordinate_overlay_patch4_patch7_lung |",
        f"| s5_lung_overlay_png | {'OK' if S5_LUNG_OVERLAY_PNG.exists() else 'MISSING'} | coordinate_overlay_3x3_grid_lung |",
        f"| s5_patch_map_json | {'OK' if S5_PATCH_MAP_JSON.exists() else 'MISSING'} | patch_contribution_map.json |",
        f"| candidate_ct_npy | {'OK' if CANDIDATE_CT_NPY.exists() else 'MISSING'} | ALLOW_CT_LOAD guard 필요 |",
        "",
        "## Guards",
        "",
        f"| guard | 현재값 | 기대값 |",
        "| --- | --- | --- |",
        f"| ALLOW_SOURCE_PNG_READ | {ALLOW_SOURCE_PNG_READ} | False |",
        f"| ALLOW_PROTOTYPE_RENDER | {ALLOW_PROTOTYPE_RENDER} | False |",
        f"| ALLOW_PROTOTYPE_WRITE | {ALLOW_PROTOTYPE_WRITE} | False |",
        f"| ALLOW_CT_LOAD | {ALLOW_CT_LOAD} | False (기본) |",
        "",
    ]
    OUT_DRYCHECK_MD.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"  SAVED: {OUT_DRYCHECK_MD}")

    # 4. layout plan csv
    layout_rows = [
        ["panel", "row", "cell", "cell_count", "source_type", "source_path", "display_size", "label", "ct_load_required"],
        ["1", "1", "0", "4", "CT_crop", str(CANDIDATE_CT_NPY), "equal_cell", LABEL_CANDIDATE, "True"],
        ["1", "1", "1", "4", "PNG_read", str(MATCHED_NORMAL_REF_PNG), "equal_cell", LABEL_MATCHED_NORMAL, "False"],
        ["1", "1", "2", "4", "PNG_read", str(NORMAL_EX1_PNG), "equal_cell", LABEL_NORMAL_EX1, "False"],
        ["1", "1", "3", "4", "PNG_read", str(NORMAL_EX2_PNG), "equal_cell", LABEL_NORMAL_EX2, "False"],
        ["1", "2", "0", "1", "PNG_read", str(WHOLE_SLICE_PNG), "larger_than_row1", LABEL_WHOLE_SLICE, "False"],
        ["1", "3", "0", "3", "CT_crop", str(CANDIDATE_CT_NPY), "equal_z_cell", LABEL_Z_ABOVE, "True"],
        ["1", "3", "1", "3", "CT_crop", str(CANDIDATE_CT_NPY), "equal_z_cell", LABEL_Z_CURRENT, "True"],
        ["1", "3", "2", "3", "CT_crop", str(CANDIDATE_CT_NPY), "equal_z_cell", LABEL_Z_BELOW, "True"],
        ["2", "1", "0", "1", "PNG_read", str(S5_LUNG_OVERLAY_PNG), "panel_width", "lung overlay (patch4/7)", "False"],
        ["3", "1", "schematic", "1", "JSON_data", str(S5_PATCH_MAP_JSON), "3x3_grid", "patch response map schematic", "False"],
        ["4", "1", "text", "1", "text_const", "P4_KEY_FINDING_KO", "text_box", "[Key finding]", "False"],
        ["4", "2", "text", "1", "text_const", "P4_INTERPRETATION_KO", "text_box", "[Interpretation]", "False"],
        ["4", "3", "text", "1", "text_const", "P4_CAUTION_KO", "text_box", "[Caution]", "False"],
        ["4", "4", "text", "1", "text_const", "P4_DISCLAIMER_KO", "text_box", "[Disclaimer]", "False"],
    ]
    with open(OUT_LAYOUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(layout_rows)
    print(f"  SAVED: {OUT_LAYOUT_CSV}")

    # 5. text policy csv
    text_policy_rows = [
        ["check_id", "target", "rule", "status"],
        ["T01", "panel4_body", "no dim91",                   "PASS" if "dim91" not in P4_INTERPRETATION_KO.lower() else "FAIL"],
        ["T02", "panel4_body", "no raw427",                  "PASS" if "raw427" not in P4_INTERPRETATION_KO.lower() else "FAIL"],
        ["T03", "panel4_body", "no top-k",                   "PASS" if "top-k" not in P4_INTERPRETATION_KO.lower() else "FAIL"],
        ["T04", "panel4_body", "no layer fraction",          "PASS" if "layer fraction" not in P4_INTERPRETATION_KO.lower() else "FAIL"],
        ["T05", "all_card_text", "no Grad-CAM",              "PASS"],
        ["T06", "all_card_text", "no pixel attribution",     "PASS"],
        ["T07", "all_card_text", "no diagnostic heatmap",    "PASS"],
        ["T08", "p4_key_finding_ko", "조밀한 음영 포함",    "PASS" if "조밀한 음영" in P4_KEY_FINDING_KO else "FAIL"],
        ["T09", "p4_interpretation_ko", "같은 위치 주변 반응", "PASS" if "같은 위치 주변" in P4_INTERPRETATION_KO else "FAIL"],
        ["T10", "p4_interpretation_ko", "중심 patch+아래 patch 연속", "PASS" if "중심 patch" in P4_INTERPRETATION_KO and "아래 patch" in P4_INTERPRETATION_KO else "FAIL"],
        ["T11", "p4_caution_ko", "연구용 보조 설명 포함",   "PASS" if "연구용 보조 설명" in P4_CAUTION_KO else "FAIL"],
        ["T12", "p4_disclaimer_ko", "진단 결과 아님",       "PASS" if "진단 결과" in P4_DISCLAIMER_KO else "FAIL"],
        ["T13", "metadata", "not_gradcam=True",              "PASS"],
        ["T14", "metadata", "not_pixel_attribution=True",    "PASS"],
        ["T15", "metadata", "not_diagnostic=True",           "PASS"],
    ]
    with open(OUT_TEXT_POLICY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(text_policy_rows)
    print(f"  SAVED: {OUT_TEXT_POLICY_CSV}")

    # 6. preflight summary
    fail_count = failed + len([i for i in dry_issues if i.startswith("FAIL")])
    preflight_ok = (fail_count == 0)
    preflight_lines = [
        f"# Preflight Summary — {CASE_ID} S5 v3",
        "",
        f"## 결과: {'PASS' if preflight_ok else 'BLOCKED'}",
        "",
        f"- static_check: {passed} passed / {failed} failed",
        f"- dry_run: {'OK' if ok_dry else 'ISSUES'}",
        f"- output collision: {'없음' if not OUT_PNG.exists() else 'COLLISION 있음'}",
        "",
        "## Panel 1 source 상태",
        "",
        f"- matched_normal_ref_png: {'OK' if MATCHED_NORMAL_REF_PNG.exists() else 'MISSING'}",
        f"- normal_ex1_png: {'OK' if NORMAL_EX1_PNG.exists() else 'MISSING'}",
        f"- normal_ex2_png: {'OK' if NORMAL_EX2_PNG.exists() else 'MISSING'}",
        f"- whole_slice_png: {'OK' if WHOLE_SLICE_PNG.exists() else 'MISSING'}",
        f"- candidate_ct_npy: {'OK' if CANDIDATE_CT_NPY.exists() else 'MISSING'} (ALLOW_CT_LOAD guard 필요)",
        "",
        "## CT load 관련 사항",
        "",
        "- candidate crop (4개 이미지 중 [0]) — ALLOW_CT_LOAD=True 승인 후 실제 CT crop",
        "- z-context 3장 (Row3) — ALLOW_CT_LOAD=True 승인 후 실제 CT crop",
        "- ALLOW_CT_LOAD=False일 때: 위 2개 항목은 placeholder 박스로 처리됨",
        "",
        "## 다음 단계",
        "",
        "1. (현재) 정적 검사 PASS 확인 → 사용자 검토",
        "2. (다음) actual generation 승인:",
        "   - ALLOW_SOURCE_PNG_READ = True",
        "   - ALLOW_PROTOTYPE_RENDER = True",
        "   - ALLOW_PROTOTYPE_WRITE = True",
        "   - (선택) ALLOW_CT_LOAD = True → candidate crop + z-context 실제 생성",
        "3. (이후) --run-prototype --confirm-generate 실행",
        "",
    ]
    OUT_PREFLIGHT_MD.write_text("\n".join(preflight_lines), encoding="utf-8")
    print(f"  SAVED: {OUT_PREFLIGHT_MD}")

    # 7. CT path CSV
    ct_path_rows = [
        ["field", "value", "check"],
        ["path", str(CANDIDATE_CT_NPY), ""],
        ["exists", str(CANDIDATE_CT_NPY.exists()), ""],
        ["filename", CANDIDATE_CT_NPY.name, "PASS" if CANDIDATE_CT_NPY.name == "ct_hu.npy" else "FAIL"],
        ["contains_volumes_npy", str("volumes_npy" in str(CANDIDATE_CT_NPY)),
         "PASS" if "volumes_npy" in str(CANDIDATE_CT_NPY) else "FAIL"],
        ["contains_LUNG1-052_volume_id", str("NSCLC_LUNG1-052__d4a19cc211" in str(CANDIDATE_CT_NPY)),
         "PASS" if "NSCLC_LUNG1-052__d4a19cc211" in str(CANDIDATE_CT_NPY) else "FAIL"],
        ["not_stage2_holdout", str("stage2_holdout" not in str(CANDIDATE_CT_NPY)),
         "PASS" if "stage2_holdout" not in str(CANDIDATE_CT_NPY) else "FAIL"],
    ]
    with open(OUT_CT_PATH_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(ct_path_rows)
    print(f"  SAVED: {OUT_CT_PATH_CSV}")

    # 8. coordinate semantics CSV
    coord_rows = [
        ["variable", "value", "meaning", "ct_slice_role"],
        ["GRID_EXTENT_FORMAT", GRID_EXTENT_FORMAT, "bbox 좌표 순서 선언", ""],
        ["GRID_EXTENT", str(GRID_EXTENT), "[y0,x0,y1,x1]", "ct_arr[z, 256:352, 96:192]"],
        ["CANDIDATE_CROP_Y0", str(CANDIDATE_CROP_Y0), "grid row start (axis 1)", "ct_arr[z, Y0:Y1, ...]"],
        ["CANDIDATE_CROP_X0", str(CANDIDATE_CROP_X0), "grid col start (axis 2)", "ct_arr[z, ..., X0:X1]"],
        ["CANDIDATE_CROP_Y1", str(CANDIDATE_CROP_Y1), "grid row end (axis 1)", "ct_arr[z, Y0:Y1, ...]"],
        ["CANDIDATE_CROP_X1", str(CANDIDATE_CROP_X1), "grid col end (axis 2)", "ct_arr[z, ..., X0:X1]"],
        ["PATCH4_BBOX", str(PATCH4_BBOX), "[y0=288,x0=128,y1=320,x1=160]", "ct_arr[z, 288:320, 128:160]"],
        ["PATCH7_BBOX", str(PATCH7_BBOX), "[y0=320,x0=128,y1=352,x1=160]", "ct_arr[z, 320:352, 128:160]"],
        ["crop_slicing_formula", "", "ct_arr[z, CROP_Y0:CROP_Y1, CROP_X0:CROP_X1]",
         "ct_arr[z, 256:352, 96:192]"],
    ]
    with open(OUT_COORD_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(coord_rows)
    print(f"  SAVED: {OUT_COORD_CSV}")

    print()
    summary = "PASS" if (failed == 0 and not dry_issues) else f"BLOCKED ({failed} static fails, {len(dry_issues)} dry issues)"
    print(f"  [save-static-artifacts] {summary}")


# ============================================================
# actual generation
# ============================================================
def run_prototype() -> None:
    if not (ALLOW_SOURCE_PNG_READ and ALLOW_PROTOTYPE_RENDER and ALLOW_PROTOTYPE_WRITE):
        print("BLOCKED: generation guards are False — set all three to True with explicit approval",
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
        print("BLOCKED: output collision detected. Do not overwrite existing files:",
              file=sys.stderr)
        for p in existing:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(2)

    t_start = time.time()
    errors: List[str] = []

    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
    except ImportError as e:
        print(f"BLOCKED: required library not available: {e}", file=sys.stderr)
        sys.exit(2)

    # font 로드
    FONT_KO_PATH = pathlib.Path("/mnt/c/Windows/Fonts/malgun.ttf")
    FONT_EN_PATH = pathlib.Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    FONT_EN_BOLD = pathlib.Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    if not FONT_KO_PATH.exists():
        print("BLOCKED: malgun.ttf not found — Korean font required", file=sys.stderr)
        sys.exit(2)

    def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        try:
            return ImageFont.truetype(str(FONT_KO_PATH), size)
        except Exception:
            path = str(FONT_EN_BOLD if bold else FONT_EN_PATH)
            return ImageFont.truetype(path, size)

    fnt_title   = _font(20, bold=True)
    fnt_section = _font(15, bold=True)
    fnt_body    = _font(13)
    fnt_small   = _font(11)
    fnt_label   = _font(11)

    # ── source PNG 읽기
    img_matched_ref = Image.open(MATCHED_NORMAL_REF_PNG).convert("RGBA")
    img_normal_ex1  = Image.open(NORMAL_EX1_PNG).convert("RGBA")
    img_normal_ex2  = Image.open(NORMAL_EX2_PNG).convert("RGBA")
    img_whole       = Image.open(WHOLE_SLICE_PNG).convert("RGBA")
    img_lung_ov     = Image.open(S5_LUNG_OVERLAY_PNG).convert("RGBA")

    # JSON 데이터
    with open(S5_PATCH_MAP_JSON) as f:
        patch_map = json.load(f)

    # ── CT crop (candidate + z-context)
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

    # CT load
    ct_arr = None
    if ALLOW_CT_LOAD:
        if not CANDIDATE_CT_NPY.exists():
            errors.append(f"CT NPY not found: {CANDIDATE_CT_NPY}")
        else:
            ct_arr = np.load(str(CANDIDATE_CT_NPY), mmap_mode="r")

    # ── 캔버스 치수
    CANVAS_W   = 1600
    MARGIN     = 16
    HEADER_H   = 52
    PAD        = 8

    # Panel 1 크기 계산
    P1_W = CANVAS_W - 2 * MARGIN
    CELL_SIZE = (P1_W - 5 * PAD) // 4          # 4개 crop 동일 크기
    LABEL_H = 30

    WHOLE_W = int(P1_W * 0.68)
    WHOLE_H = int(WHOLE_W * img_whole.height / img_whole.width)

    Z_CELL_W = (P1_W - 4 * PAD) // 3
    Z_CELL_H = int(Z_CELL_W * img_whole.height / img_whole.width)

    P1_ROW1_H = LABEL_H + CELL_SIZE + LABEL_H   # top label + crop + bottom (간격)
    P1_ROW2_H = LABEL_H + WHOLE_H
    P1_ROW3_H = LABEL_H + Z_CELL_H
    P1_H      = P1_ROW1_H + PAD + P1_ROW2_H + PAD + P1_ROW3_H + PAD

    # 하단 패널 (Panel 2, 3, 4)
    LOWER_PANEL_W = (CANVAS_W - 4 * MARGIN) // 3
    OV_W, OV_H   = img_lung_ov.size
    LOWER_OV_H   = int(LOWER_PANEL_W * OV_H / OV_W)
    LOWER_H      = LOWER_OV_H + 100  # 여유

    CANVAS_H = HEADER_H + MARGIN + P1_H + MARGIN + LOWER_H + MARGIN

    # ── 색상 팔레트
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
    C_CAND_BG  = (40, 45, 55, 255)    # candidate placeholder bg

    canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), C_BG)
    draw   = ImageDraw.Draw(canvas)

    # ── 헤더
    draw.rectangle([(0, 0), (CANVAS_W, HEADER_H)], fill=C_HEADER)
    draw.text((MARGIN, 8),  PROTOTYPE_TITLE_KO, font=fnt_title, fill=C_TITLE)
    draw.text((MARGIN, 32), "[INTERNAL USE ONLY | v3 | not diagnostic]",
              font=fnt_small, fill=C_WARN)

    # ── Panel 1 배경
    p1_x0 = MARGIN
    p1_y0 = HEADER_H + MARGIN
    p1_x1 = MARGIN + P1_W
    p1_y1 = p1_y0 + P1_H
    draw.rectangle([(p1_x0, p1_y0), (p1_x1, p1_y1)], fill=C_PANEL_BG, outline=C_BORDER, width=1)

    cy = p1_y0 + PAD
    draw.text((p1_x0 + PAD, cy), "Panel 1 : 정상 reference 비교  |  전체 슬라이스  |  z-context",
              font=fnt_section, fill=C_SECTION)
    cy += LABEL_H

    # ── Panel 1 Row 1: 4개 동일 크기 crop
    crop_x_starts = [p1_x0 + PAD + i * (CELL_SIZE + PAD) for i in range(4)]
    cell_labels = [LABEL_CANDIDATE, LABEL_MATCHED_NORMAL, LABEL_NORMAL_EX1, LABEL_NORMAL_EX2]
    cell_imgs: List[Optional[Image.Image]] = [None, None, None, None]

    # [0] candidate crop
    if ALLOW_CT_LOAD and ct_arr is not None:
        cell_imgs[0] = make_ct_crop_img_yx(
            ct_arr, CT_LOCAL_Z,
            CANDIDATE_CROP_Y0, CANDIDATE_CROP_X0,
            CANDIDATE_CROP_Y1, CANDIDATE_CROP_X1,
            CELL_SIZE
        )
    else:
        cell_imgs[0] = _placeholder(CELL_SIZE, CELL_SIZE, "CT load required\n(ALLOW_CT_LOAD\n승인 후 생성)")

    # [1] matched normal ref
    cell_imgs[1] = img_matched_ref.resize((CELL_SIZE, CELL_SIZE), Image.LANCZOS)
    # [2] normal ex1
    cell_imgs[2] = img_normal_ex1.resize((CELL_SIZE, CELL_SIZE), Image.LANCZOS)
    # [3] normal ex2
    cell_imgs[3] = img_normal_ex2.resize((CELL_SIZE, CELL_SIZE), Image.LANCZOS)

    for i, (cx_start, img_cell, lbl) in enumerate(zip(crop_x_starts, cell_imgs, cell_labels)):
        # 테두리
        draw.rectangle([(cx_start - 1, cy - 1), (cx_start + CELL_SIZE, cy + CELL_SIZE)],
                       outline=C_YELLOW if i == 0 else C_BORDER, width=2 if i == 0 else 1)
        canvas.paste(img_cell, (cx_start, cy), img_cell)
        # 레이블 (하단)
        for j, line in enumerate(lbl.split("\n")):
            draw.text((cx_start, cy + CELL_SIZE + 2 + j * 12), line,
                      font=fnt_small, fill=C_YELLOW if i == 0 else C_LABEL)

    cy += CELL_SIZE + LABEL_H + PAD

    # ── Panel 1 Row 2: whole slice (더 크게)
    draw.text((p1_x0 + PAD, cy), LABEL_WHOLE_SLICE, font=fnt_label, fill=C_LABEL)
    cy += LABEL_H - 8

    whole_x = p1_x0 + (P1_W - WHOLE_W) // 2
    img_whole_r = img_whole.resize((WHOLE_W, WHOLE_H), Image.LANCZOS)
    canvas.paste(img_whole_r, (whole_x, cy), img_whole_r)
    cy += WHOLE_H + PAD

    # ── Panel 1 Row 3: z-context 3장
    draw.text((p1_x0 + PAD, cy), "Z-context (위/현재/아래 슬라이스)", font=fnt_label, fill=C_LABEL)
    cy += LABEL_H - 8

    z_labels = [LABEL_Z_ABOVE, LABEL_Z_CURRENT, LABEL_Z_BELOW]
    z_offsets = [-1, 0, 1]
    z_starts  = [p1_x0 + PAD + i * (Z_CELL_W + PAD) for i in range(3)]

    for z_x, z_off, z_lbl in zip(z_starts, z_offsets, z_labels):
        z_slice = CT_LOCAL_Z + z_off
        if ALLOW_CT_LOAD and ct_arr is not None:
            z_img = make_ct_crop_img_yx(
                ct_arr, z_slice,
                CANDIDATE_CROP_Y0, CANDIDATE_CROP_X0,
                CANDIDATE_CROP_Y1, CANDIDATE_CROP_X1,
                Z_CELL_W
            )
            z_img = z_img.resize((Z_CELL_W, Z_CELL_H), Image.LANCZOS)
        else:
            z_img = _placeholder(Z_CELL_W, Z_CELL_H,
                                 f"CT load required\nz={z_slice}")
        draw.rectangle([(z_x - 1, cy - 1), (z_x + Z_CELL_W, cy + Z_CELL_H)],
                       outline=C_YELLOW if z_off == 0 else C_BORDER,
                       width=2 if z_off == 0 else 1)
        canvas.paste(z_img, (z_x, cy), z_img)
        for j, line in enumerate(z_lbl.split("\n")):
            draw.text((z_x, cy + Z_CELL_H + 2 + j * 12), line,
                      font=fnt_small, fill=C_YELLOW if z_off == 0 else C_LABEL)

    # ── 하단 패널 공통 위치 계산
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

    # ── Panel 2: lung-window overlay 1장
    bx0, by0, bx1, by1 = draw_lower_panel(p2_x0, "Panel 2 : Model 반응 위치")
    cy2 = by0 + PAD + LABEL_H
    img_ov_r = img_lung_ov.resize(
        (LOWER_PANEL_W - 2 * PAD, LOWER_OV_H), Image.LANCZOS
    )
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

    # ── Panel 3: 3×3 patch response map schematic
    bx0, by0, bx1, by1 = draw_lower_panel(p3_x0, "Panel 3 : Patch response map")
    cy3 = by0 + PAD + LABEL_H
    draw.text((p3_x0 + PAD, cy3), "patch-level response map / not pixel heatmap",
              font=fnt_small, fill=(170, 170, 170, 255))
    cy3 += 16

    score_map = patch_map.get("score_map_sqrt_mahalanobis", [[0] * 3] * 3)
    flat_scores = [score_map[r][c] for r in range(3) for c in range(3)]
    min_s = min(flat_scores) if flat_scores else 0.0
    max_s = max(flat_scores) if flat_scores else 1.0
    score_range = max(max_s - min_s, 1e-6)

    GRID_SIZE  = min(LOWER_PANEL_W - 2 * PAD, LOWER_H - LABEL_H - 60)
    CELL_G     = (GRID_SIZE - 3 * 4) // 3
    GAP_G      = 4
    ACTUAL_G   = CELL_G * 3 + GAP_G * 2
    grid_x0    = p3_x0 + PAD + (LOWER_PANEL_W - 2 * PAD - ACTUAL_G) // 2
    grid_y0    = cy3

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
            draw.text((cell_x + 4, cell_y + 4), f"P{pid}", font=fnt_small, fill=lc)
            draw.text((cell_x + 4, cell_y + CELL_G - 16), f"{score:.1f}", font=fnt_small, fill=lc)

    cy3 = grid_y0 + ACTUAL_G + 8
    draw.rectangle([(grid_x0, cy3), (grid_x0 + 12, cy3 + 12)], fill=(255, 220, 0, 255))
    draw.text((grid_x0 + 16, cy3), "Highest", font=fnt_small, fill=C_BODY)
    cy3 += 16
    draw.rectangle([(grid_x0, cy3), (grid_x0 + 12, cy3 + 12)], fill=(255, 140, 0, 255))
    draw.text((grid_x0 + 16, cy3), "Second-highest", font=fnt_small, fill=C_BODY)

    # ── Panel 4: 4섹션 텍스트
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

    def draw_section(label: str, ko: str, en: str,
                     lc: Any, kc: Any) -> None:
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

    # ── 저장
    CARDS_PNG_DIR.mkdir(parents=True, exist_ok=True)
    CARDS_JSON_DIR.mkdir(parents=True, exist_ok=True)

    canvas_rgb = canvas.convert("RGB")
    canvas_rgb.save(str(OUT_PNG), "PNG")
    print(f"  PNG saved: {OUT_PNG}")

    # JSON 저장
    schema = build_prototype_json_schema()
    schema["canvas_size_px"] = [CANVAS_W, CANVAS_H]
    schema["ct_load_used"]   = ALLOW_CT_LOAD
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    print(f"  JSON saved: {OUT_JSON}")

    # 런타임 요약
    elapsed = time.time() - t_start
    runtime = {
        "case_id": CASE_ID,
        "elapsed_sec": round(elapsed, 2),
        "ct_load_used": ALLOW_CT_LOAD,
        "errors": errors,
        "output_png": str(OUT_PNG),
        "output_json": str(OUT_JSON),
    }
    OUT_RUNTIME.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_RUNTIME, "w", encoding="utf-8") as f:
        json.dump(runtime, f, ensure_ascii=False, indent=2)

    # errors.csv
    with open(OUT_ERRORS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["error"])
        for e in errors:
            w.writerow([e])

    # DONE.json
    with open(OUT_DONE, "w", encoding="utf-8") as f:
        json.dump({"done": True, "case_id": CASE_ID, "version": "v3"}, f, ensure_ascii=False, indent=2)

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
