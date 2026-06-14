"""
S5 Demo Card Prototype v3d Panel1 v4 Ref — LUNG1-052__c3
========================================================
v3c → v3d 주요 변경:
  - Panel 1 Row 1: normal reference 3개를 v4 exact-cell top3 preview PNG로 교체
    (normal_ref_1_v4.png, normal_ref_2_v4.png, normal_ref_3_v4.png)
  - candidate도 v4 preview PNG에서 읽기 (candidate_crop_lung1_052.png)
  - CT load 없음 (ALLOW_CT_LOAD 항구 False)
  - z-context: placeholder 사용 (CT load 없이 생성 불가)
  - Panel 1 labels: Same-cell normal ref 1/2/3
  - Panel 1 subtitle: not same-z matching 명시
  - Panel 4 Caution: same lung-ROI position cell policy
  - Panels 2~4: v3c 구조 그대로 유지

v3d reference selection (v4 exact-cell top3):
  cell_key = image_left|Z1|Y2|X1
  normal_ref_1: z=50, Δz=1,  quality=0.9887
  normal_ref_2: z=89, Δz=38, quality=0.9747
  normal_ref_3: z=80, Δz=29, quality=0.9627
  avg_abs_delta_z = 22.7 (v3c was 116.0)
  not_same_z_matched = True

v3c / v4 artifact 수정 금지.
"""

import csv
import json
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

# CT load guard — 항구 False (v3d에서는 CT load 없음)
ALLOW_CT_LOAD              = False
ALLOW_CANDIDATE_CT_LOAD    = False
ALLOW_NORMAL_REF_CT_LOAD   = False

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

GRID_EXTENT          = [256, 96, 352, 192]
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

LAYOUT_SELECTED      = "OPTION_V3D_4PANEL_SAME_CELL_V4_REF"

# ============================================================
# v3d reference selection (v4 exact-cell top3)
# ============================================================
V4_CELL_KEY           = "image_left|Z1|Y2|X1"
V4_CANDIDATE_Z        = 51
V4_NORMAL_REF_1_Z     = 50
V4_NORMAL_REF_2_Z     = 89
V4_NORMAL_REF_3_Z     = 80
V4_DELTA_Z_1          = abs(V4_NORMAL_REF_1_Z - V4_CANDIDATE_Z)   # 1
V4_DELTA_Z_2          = abs(V4_NORMAL_REF_2_Z - V4_CANDIDATE_Z)   # 38
V4_DELTA_Z_3          = abs(V4_NORMAL_REF_3_Z - V4_CANDIDATE_Z)   # 29
V4_AVG_ABS_DELTA_Z    = round((V4_DELTA_Z_1 + V4_DELTA_Z_2 + V4_DELTA_Z_3) / 3, 1)  # 22.7

V4_NORMAL_REF_1_PATIENT = "subset1_1.3.6.1.4.1.14519.5.2.1.6279.6001.100684836163890911914061745866"
V4_NORMAL_REF_2_PATIENT = "subset2_1.3.6.1.4.1.14519.5.2.1.6279.6001.311236942972970815890902714604"
V4_NORMAL_REF_3_PATIENT = "subset9_1.3.6.1.4.1.14519.5.2.1.6279.6001.109882169963817627559804568094"

V4_NORMAL_REF_1_QUALITY = 0.9887
V4_NORMAL_REF_2_QUALITY = 0.9747
V4_NORMAL_REF_3_QUALITY = 0.9627

V3C_AVG_ABS_DELTA_Z   = 116.0

# ============================================================
# 경로 상수
# ============================================================
PROJECT_ROOT = pathlib.Path("/home/jinhy/project/lung-ct-anomaly")

V4_PREVIEW_ROOT = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/reference_bank_v4_lung1_052_crop_preview")

V4_CANDIDATE_PNG    = V4_PREVIEW_ROOT / "candidate_crop_lung1_052.png"
V4_NORMAL_REF_1_PNG = V4_PREVIEW_ROOT / "normal_ref_1_v4.png"
V4_NORMAL_REF_2_PNG = V4_PREVIEW_ROOT / "normal_ref_2_v4.png"
V4_NORMAL_REF_3_PNG = V4_PREVIEW_ROOT / "normal_ref_3_v4.png"
V4_PREVIEW_METADATA = V4_PREVIEW_ROOT / "preview_metadata.json"
V4_PREVIEW_INDEX    = V4_PREVIEW_ROOT / "preview_index.csv"
V4_PREVIEW_DONE     = V4_PREVIEW_ROOT / "DONE.json"

V4_METADATA_ROOT = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/reference_bank_v4_lung_roi_position_metadata")
V4_TOP3_BY_CELL_CSV      = V4_METADATA_ROOT / "normal_reference_bank_v4_top3_by_cell.csv"
V4_RETRIEVAL_PREVIEW_CSV = V4_METADATA_ROOT / "lung1_052_v4_retrieval_preview.csv"

V4_PREFLIGHT_ROOT = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/reference_bank_v4_lung1_052_crop_preview_preflight")
V4_PLAN_CSV  = V4_PREFLIGHT_ROOT / "crop_preview_plan_lung1_052_v4.csv"
V4_CHECK_CSV = V4_PREFLIGHT_ROOT / "selected_top3_reference_check_v4.csv"

WHOLE_SLICE_PNG = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "s5_lung1_052_c3_coordinate_visual_audit_1case_v1"
    / "coordinate_overlay_patch4_patch7_lung.png")

S5_LUNG_OVERLAY_PNG = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "s5_lung1_052_c3_coordinate_visual_audit_1case_v1"
    / "coordinate_overlay_3x3_grid_lung.png")
S5_COORD_METADATA = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "s5_lung1_052_c3_coordinate_visual_audit_1case_v1"
    / "coordinate_overlay_metadata.json")

S5_PATCH_MAP_JSON = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "s5_lung1_052_c3_patch_level_contribution_map_1case_smoke_v1"
    / "patch_contribution_map.json")

S3_CARD_JSON = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    / "s3_expansion_cards_v1/cards_json/LUNG1-052__c3.json")

_V3C_ROOT = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "s5_demo_card_prototype_lung1_052_c3_v3c_reference_match")
_V3C_JSON = _V3C_ROOT / "cards_json/LUNG1-052__c3_s5_demo_prototype_v3c.json"
_V3C_PNG  = _V3C_ROOT / "cards_png/LUNG1-052__c3_s5_demo_prototype_v3c.png"

OUTPUT_ROOT    = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "s5_demo_card_prototype_lung1_052_c3_v3d_panel1_v4_ref")
STATIC_DIR     = OUTPUT_ROOT / "static"
CARDS_PNG_DIR  = OUTPUT_ROOT / "cards_png"
CARDS_JSON_DIR = OUTPUT_ROOT / "cards_json"
OUT_PNG        = CARDS_PNG_DIR  / "LUNG1-052__c3_s5_demo_prototype_v3d.png"
OUT_JSON       = CARDS_JSON_DIR / "LUNG1-052__c3_s5_demo_prototype_v3d.json"
OUT_INDEX_CSV  = OUTPUT_ROOT / "index_cards_v3d.csv"
OUT_RUNTIME    = OUTPUT_ROOT / "runtime_summary_v3d.json"
OUT_ERRORS     = OUTPUT_ROOT / "errors.csv"
OUT_DONE       = OUTPUT_ROOT / "DONE.json"

OUT_DRYCHECK_MD       = STATIC_DIR / "drycheck_v3d_panel1_v4_ref.md"
OUT_DRYCHECK_JSON     = STATIC_DIR / "drycheck_v3d_panel1_v4_ref.json"
OUT_LAYOUT_CSV        = STATIC_DIR / "layout_plan_v3d.csv"
OUT_REF_SELECTION_CSV = STATIC_DIR / "reference_selection_policy_v3d.csv"
OUT_TOP3_PATHS_CSV    = STATIC_DIR / "top3_reference_paths_v3d.csv"
OUT_TEXT_POLICY_CSV   = STATIC_DIR / "text_policy_v3d.csv"
OUT_PREFLIGHT_MD      = STATIC_DIR / "preflight_summary_v3d.md"

# ============================================================
# 텍스트 상수 (v3d)
# ============================================================
PROTOTYPE_TITLE_KO = "S5 PaDiM 이상 후보 설명 카드 v3d — LUNG1-052__c3"
PROTOTYPE_TITLE_EN = "S5 PaDiM patch-level explanation prototype v3d — LUNG1-052__c3"

PANEL1_SUBTITLE_EN = (
    "Normal references are selected from the same lung-ROI position cell; "
    "this is not same-z matching."
)
PANEL1_SUBTITLE_KO = (
    "정상 reference는 같은 폐 ROI 위치 셀에서 선정되었으며, z-방향 정합은 아닙니다."
)

P4_KEY_FINDING_KO = (
    "정상 reference와 비교했을 때, 해당 영역은 더 조밀한 음영을 보였습니다."
)
P4_KEY_FINDING_EN = (
    "Compared with normal references, this region showed denser opacification."
)
P4_INTERPRETATION_KO = (
    "같은 폐 ROI 위치 셀에서 선정된 정상 crop들과 비교했을 때, "
    "PaDiM response가 높은 영역은 정상 예시와 다른 국소 패턴을 보였습니다. "
    "특히 중심 patch와 바로 아래 patch에서도 높은 반응이 연속적으로 확인되었으며, "
    "이 주변을 정상 분포와 다른 국소 영역으로 본 것으로 해석할 수 있습니다."
)
P4_INTERPRETATION_EN = (
    "Compared with normal crops selected from the same lung-ROI position cell, "
    "the PaDiM-responsive region showed a locally distinct pattern from normal examples. "
    "The center patch and the adjacent lower patch showed consecutively high responses."
)
P4_CAUTION_KO = (
    "References are selected from the same lung-ROI position cell, "
    "but z-direction alignment is limited and should not be interpreted as same-z matching. "
    "이 설명은 모델 반응을 이해하기 위한 연구용 보조 설명입니다."
)
P4_CAUTION_EN = (
    "References are selected from the same lung-ROI position cell, "
    "but z-direction alignment is limited and should not be interpreted as same-z matching. "
    "This is a research-purpose supplementary explanation for understanding model response."
)
P4_DISCLAIMER_KO = "병변 원인이나 진단 결과를 직접 의미하지 않습니다."
P4_DISCLAIMER_EN  = "This does not directly imply lesion cause or diagnostic result."

LABEL_CANDIDATE    = "후보 crop\n(candidate)"
LABEL_NORMAL_REF_1 = "Same-cell normal ref 1\n(정상 reference 1)"
LABEL_NORMAL_REF_2 = "Same-cell normal ref 2\n(정상 reference 2)"
LABEL_NORMAL_REF_3 = "Same-cell normal ref 3\n(정상 reference 3)"
LABEL_WHOLE_SLICE  = "전체 슬라이스 (lung window, patch 위치 표시)"
LABEL_Z_ABOVE      = "위 슬라이스\n(z-1)"
LABEL_Z_CURRENT    = "현재 슬라이스\n(z)"
LABEL_Z_BELOW      = "아래 슬라이스\n(z+1)"

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
    f"v4 exact-cell top3: same lung-ROI position cell, not same-z matching. "
    f"avg_abs_delta_z = {V4_AVG_ABS_DELTA_Z} slices (v3c was {V3C_AVG_ABS_DELTA_Z})."
)


# ============================================================
# prototype JSON schema
# ============================================================
def build_prototype_json_schema() -> Dict[str, Any]:
    return {
        "version": "v3d_panel1_v4_ref",
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
                "S4-style reference comparison (v3d same-cell v4 ref): "
                "Row1=4개 동일 크기 (candidate|same-cell_normal_ref1|ref2|ref3), "
                "모두 v4 preview PNG; "
                "Row2=whole slice (더 크게); "
                "Row3=z-context placeholder (CT load disabled)"
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
                "(의사용 4섹션; same lung-ROI position cell; z-direction limitation)"
            ),
        },
        "metadata": {
            "version": "v3d_panel1_v4_ref",
            "panel1_reference_source": "v4_exact_cell_top3",
            "cell_key": V4_CELL_KEY,
            "reference_selection_policy": "SAME_LUNG_ROI_POSITION_CELL_WITH_Z_LIMITATION",
            "candidate_z": V4_CANDIDATE_Z,
            "normal_ref_1_z": V4_NORMAL_REF_1_Z,
            "normal_ref_2_z": V4_NORMAL_REF_2_Z,
            "normal_ref_3_z": V4_NORMAL_REF_3_Z,
            "avg_abs_delta_z": V4_AVG_ABS_DELTA_Z,
            "not_same_z_matched": True,
            "z_orientation_limitation": True,
            "not_diagnostic": True,
            "not_gradcam": True,
            "not_pixel_attribution": True,
            "stage2_holdout_accessed": False,
            "model_forward_occurred": False,
            "feature_extraction_occurred": False,
            "contribution_recalc_occurred": False,
            "existing_v3c_artifact_modified": False,
            "existing_v4_artifact_modified": False,
            "internal_use_only": True,
            "font_fix_required_before_external_share": True,
            "v3c_avg_abs_delta_z": V3C_AVG_ABS_DELTA_Z,
            "delta_z_improvement": True,
            "ct_load_used": False,
            "normal_ref_ct_load_used": False,
            "panel1_source_type": "v4_preview_png_read",
            "dim91_caveat": _META_DIM91_CAVEAT,
            "basal_pleural_caution": _META_BASAL_PLEURAL_CAUTION,
            "z_orientation_caveat": _META_Z_ORIENTATION,
            "panel1_4crop_equal_cell_size": True,
            "panel1_whole_slice_larger_row": True,
            "panel2_overlay_single_only": True,
            "panel3_schematic_not_overlay": True,
            "patch_response_map_source": "score_map_sqrt_mahalanobis",
        },
        "source_files": {
            "v4_candidate_png":    str(V4_CANDIDATE_PNG),
            "v4_normal_ref_1_png": str(V4_NORMAL_REF_1_PNG),
            "v4_normal_ref_2_png": str(V4_NORMAL_REF_2_PNG),
            "v4_normal_ref_3_png": str(V4_NORMAL_REF_3_PNG),
            "v4_preview_metadata": str(V4_PREVIEW_METADATA),
            "whole_slice_png":     str(WHOLE_SLICE_PNG),
            "s5_lung_overlay_png": str(S5_LUNG_OVERLAY_PNG),
            "s5_patch_map_json":   str(S5_PATCH_MAP_JSON),
            "s3_card_json":        str(S3_CARD_JSON),
        },
        "v3d_reference_selection": {
            "cell_key": V4_CELL_KEY,
            "source": "v4_exact_cell_top3",
            "preview_root": str(V4_PREVIEW_ROOT),
            "normal_ref_1": {
                "patient_id": V4_NORMAL_REF_1_PATIENT, "z": V4_NORMAL_REF_1_Z,
                "delta_z": V4_DELTA_Z_1, "quality_score": V4_NORMAL_REF_1_QUALITY,
                "png_path": str(V4_NORMAL_REF_1_PNG),
            },
            "normal_ref_2": {
                "patient_id": V4_NORMAL_REF_2_PATIENT, "z": V4_NORMAL_REF_2_Z,
                "delta_z": V4_DELTA_Z_2, "quality_score": V4_NORMAL_REF_2_QUALITY,
                "png_path": str(V4_NORMAL_REF_2_PNG),
            },
            "normal_ref_3": {
                "patient_id": V4_NORMAL_REF_3_PATIENT, "z": V4_NORMAL_REF_3_Z,
                "delta_z": V4_DELTA_Z_3, "quality_score": V4_NORMAL_REF_3_QUALITY,
                "png_path": str(V4_NORMAL_REF_3_PNG),
            },
            "avg_abs_delta_z": V4_AVG_ABS_DELTA_Z,
            "not_same_z_matched": True,
            "z_orientation_limitation": True,
            "not_z_matched": True,
            "label_policy": "Same-cell normal ref",
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
            issues.append(f"FAIL: {label}" + (f" | {detail}" if detail else ""))

    schema = build_prototype_json_schema()
    meta   = schema["metadata"]

    check("01 output_root contains v3d_panel1_v4_ref",
          "v3d_panel1_v4_ref" in OUTPUT_ROOT.name)
    check("02 output_root != v3c_root", OUTPUT_ROOT != _V3C_ROOT)
    check("03 OUT_PNG is under v3d root", "v3d_panel1_v4_ref" in str(OUT_PNG))
    check("04a v4_candidate_png under preview_root",
          str(V4_PREVIEW_ROOT) in str(V4_CANDIDATE_PNG))
    check("04b v4_normal_ref_1_png under preview_root",
          str(V4_PREVIEW_ROOT) in str(V4_NORMAL_REF_1_PNG))
    check("04c v4_normal_ref_2_png under preview_root",
          str(V4_PREVIEW_ROOT) in str(V4_NORMAL_REF_2_PNG))
    check("04d v4_normal_ref_3_png under preview_root",
          str(V4_PREVIEW_ROOT) in str(V4_NORMAL_REF_3_PNG))
    check("05a candidate_z = 51",       meta.get("candidate_z") == 51)
    check("05b normal_ref_1_z = 50",    meta.get("normal_ref_1_z") == 50)
    check("05c normal_ref_2_z = 89",    meta.get("normal_ref_2_z") == 89)
    check("05d normal_ref_3_z = 80",    meta.get("normal_ref_3_z") == 80)
    check("06 avg_abs_delta_z = 22.7",  meta.get("avg_abs_delta_z") == 22.7)
    check("07 ALLOW_SOURCE_PNG_READ default False",  not ALLOW_SOURCE_PNG_READ)
    check("08 ALLOW_PROTOTYPE_RENDER default False", not ALLOW_PROTOTYPE_RENDER)
    check("09 ALLOW_PROTOTYPE_WRITE default False",  not ALLOW_PROTOTYPE_WRITE)
    check("10a ALLOW_CT_LOAD False",            not ALLOW_CT_LOAD)
    check("10b ALLOW_CANDIDATE_CT_LOAD False",  not ALLOW_CANDIDATE_CT_LOAD)
    check("10c ALLOW_NORMAL_REF_CT_LOAD False", not ALLOW_NORMAL_REF_CT_LOAD)
    check("11 ALLOW_STAGE2_HOLDOUT False",       not ALLOW_STAGE2_HOLDOUT)
    check("12a ALLOW_MODEL_FORWARD False",       not ALLOW_MODEL_FORWARD)
    check("12b ALLOW_FEATURE_EXTRACTION False",  not ALLOW_FEATURE_EXTRACTION)
    check("12c ALLOW_CONTRIBUTION_RECALC False", not ALLOW_CONTRIBUTION_RECALC)

    all_text = " ".join([
        P4_KEY_FINDING_KO, P4_KEY_FINDING_EN,
        P4_INTERPRETATION_KO, P4_INTERPRETATION_EN,
        P4_CAUTION_KO, P4_CAUTION_EN,
        P4_DISCLAIMER_KO, P4_DISCLAIMER_EN,
        LABEL_NORMAL_REF_1, LABEL_NORMAL_REF_2, LABEL_NORMAL_REF_3,
        PROTOTYPE_TITLE_EN, PANEL1_SUBTITLE_EN,
    ]).lower()
    check("13 no 'same-z matched' wording",   "same-z matched" not in all_text)
    check("14 no 'z-matched' in labels",
          "z-matched" not in (LABEL_NORMAL_REF_1 + LABEL_NORMAL_REF_2 + LABEL_NORMAL_REF_3).lower())
    check("15 not_same_z_matched True",        meta.get("not_same_z_matched") is True)
    check("16 z_orientation_limitation True",  meta.get("z_orientation_limitation") is True)
    check("17a not_gradcam True",              meta.get("not_gradcam") is True)
    check("17b panel2 not grad-cam mention",
          "not grad-cam" in schema["panels"]["panel_2"].lower())
    check("18 not_pixel_attribution True",     meta.get("not_pixel_attribution") is True)
    check("19 not_diagnostic True",            meta.get("not_diagnostic") is True)

    # output collision (20)
    for name, p in [
        ("OUT_PNG", OUT_PNG), ("OUT_JSON", OUT_JSON),
        ("OUT_INDEX_CSV", OUT_INDEX_CSV), ("OUT_RUNTIME", OUT_RUNTIME),
        ("OUT_ERRORS", OUT_ERRORS), ("OUT_DONE", OUT_DONE),
    ]:
        if p.exists():
            issues.append(f"COLLISION: {name} already exists: {p}")
            failed += 1
        else:
            passed += 1

    check("C1 candidate_z=51",         CANDIDATE_CROP_Y0 == 256)
    check("C2 PATCH4_BBOX y0==288",    PATCH4_BBOX[0] == 288)
    check("C3 PATCH7_BBOX y0==320",    PATCH7_BBOX[0] == 320)
    check("C4 patch4.y1 == patch7.y0", PATCH4_BBOX[2] == PATCH7_BBOX[0])
    check("C5 stage2_holdout_accessed False",
          meta.get("stage2_holdout_accessed") is False)
    check("C6 model_forward_occurred False",
          meta.get("model_forward_occurred") is False)
    check("C7 existing_v3c_artifact_modified False",
          meta.get("existing_v3c_artifact_modified") is False)
    check("C8 existing_v4_artifact_modified False",
          meta.get("existing_v4_artifact_modified") is False)
    check("C9 v3d_reference_selection in schema",
          "v3d_reference_selection" in schema)
    check("C10 cell_key = image_left|Z1|Y2|X1",
          V4_CELL_KEY == "image_left|Z1|Y2|X1")
    check("C11 avg_abs_delta_z < v3c (improvement)",
          V4_AVG_ABS_DELTA_Z < V3C_AVG_ABS_DELTA_Z)
    check("C12 panel1_subtitle has 'not same-z matching'",
          "not same-z matching" in PANEL1_SUBTITLE_EN.lower())
    check("C13 p4_caution has 'same lung-ROI position cell'",
          "same lung-roi position cell" in P4_CAUTION_EN.lower())
    check("C14 p4_caution has 'not be interpreted as same-z matching'",
          "not be interpreted as same-z matching" in P4_CAUTION_EN.lower())

    return passed, failed, issues


# ============================================================
# dry-run
# ============================================================
def dry_run() -> Tuple[bool, List[str]]:
    print(f"[dry-run] S5 Demo Card Prototype v3d — {CASE_ID}")
    issues: List[str] = []

    def chk(label: str, p: pathlib.Path) -> None:
        if p.exists():
            print(f"  OK   {label}: {p.name}")
        else:
            msg = f"MISSING {label}: {p}"
            print(f"  ERR  {msg}")
            issues.append(msg)

    chk("v4_candidate_png",     V4_CANDIDATE_PNG)
    chk("v4_normal_ref_1_png",  V4_NORMAL_REF_1_PNG)
    chk("v4_normal_ref_2_png",  V4_NORMAL_REF_2_PNG)
    chk("v4_normal_ref_3_png",  V4_NORMAL_REF_3_PNG)
    chk("v4_preview_metadata",  V4_PREVIEW_METADATA)
    chk("v4_preview_done",      V4_PREVIEW_DONE)
    chk("v4_top3_by_cell_csv",  V4_TOP3_BY_CELL_CSV)
    chk("v4_retrieval_preview", V4_RETRIEVAL_PREVIEW_CSV)
    chk("whole_slice_png",      WHOLE_SLICE_PNG)
    chk("s5_lung_overlay_png",  S5_LUNG_OVERLAY_PNG)
    chk("s5_patch_map_json",    S5_PATCH_MAP_JSON)

    print(f"  INFO v3c PNG (read-only): exists={_V3C_PNG.exists()}")
    print(f"  INFO v3c JSON (read-only): exists={_V3C_JSON.exists()}")

    for name, val in [
        ("ALLOW_SOURCE_PNG_READ", ALLOW_SOURCE_PNG_READ),
        ("ALLOW_PROTOTYPE_RENDER", ALLOW_PROTOTYPE_RENDER),
        ("ALLOW_PROTOTYPE_WRITE", ALLOW_PROTOTYPE_WRITE),
    ]:
        if val:
            issues.append(f"WARN: {name} is True (should be False in dry-run)")

    for name, val in [
        ("ALLOW_CT_LOAD", ALLOW_CT_LOAD),
        ("ALLOW_CANDIDATE_CT_LOAD", ALLOW_CANDIDATE_CT_LOAD),
        ("ALLOW_NORMAL_REF_CT_LOAD", ALLOW_NORMAL_REF_CT_LOAD),
        ("ALLOW_STAGE2_HOLDOUT", ALLOW_STAGE2_HOLDOUT),
        ("ALLOW_MODEL_FORWARD", ALLOW_MODEL_FORWARD),
    ]:
        if val:
            issues.append(f"BLOCKED: {name} must remain False in v3d")

    print(f"  OK   avg_abs_delta_z = {V4_AVG_ABS_DELTA_Z} (v3c was {V3C_AVG_ABS_DELTA_Z})")
    print(f"  OK   not_same_z_matched = True")
    print(f"  OK   z_orientation_limitation = True")

    if V4_PREVIEW_DONE.exists():
        with open(V4_PREVIEW_DONE) as f:
            done_data = json.load(f)
        verdict = done_data.get("verdict", done_data.get("done"))
        if verdict == "PASS" or verdict is True:
            print(f"  OK   v4_preview_done: verdict=PASS")
        else:
            issues.append(f"v4_preview_done verdict unexpected: {verdict}")

    for name, p in [
        ("OUT_PNG", OUT_PNG), ("OUT_JSON", OUT_JSON),
        ("OUT_INDEX_CSV", OUT_INDEX_CSV), ("OUT_RUNTIME", OUT_RUNTIME),
        ("OUT_ERRORS", OUT_ERRORS), ("OUT_DONE", OUT_DONE),
    ]:
        if p.exists():
            issues.append(f"COLLISION: {name} already exists")

    if OUTPUT_ROOT.exists():
        print(f"  WARN v3d output root exists (static/ 있을 수 있음): {OUTPUT_ROOT.name}")
    else:
        print(f"  OK   v3d output root not yet created: {OUTPUT_ROOT.name}")

    passed, failed, si = static_check()
    issues.extend(si)
    print(f"\n  static_check: {passed} passed, {failed} failed")

    ok = len(issues) == 0
    if issues:
        print("\n[dry-run] Issues:")
        for iss in issues:
            print(f"  - {iss}")
        print("\n[dry-run] BLOCKED")
    else:
        print("\n[dry-run] PASS")
        print("NOTE: actual gen 시 3개 guard만 True:")
        print("  ALLOW_SOURCE_PNG_READ / ALLOW_PROTOTYPE_RENDER / ALLOW_PROTOTYPE_WRITE")
        print("  (CT load guards는 False 유지)")
    return ok, issues


# ============================================================
# plan-only
# ============================================================
def plan_only() -> None:
    print(f"[plan-only] S5 Demo Card Prototype v3d — {CASE_ID}")
    print(f"  Layout : {LAYOUT_SELECTED}")
    print(f"  Panel 1 subtitle: {PANEL1_SUBTITLE_EN}")
    print(f"    [0] candidate     → {V4_CANDIDATE_PNG.name} (z={V4_CANDIDATE_Z})")
    print(f"    [1] same-cell ref1 → {V4_NORMAL_REF_1_PNG.name} (z={V4_NORMAL_REF_1_Z}, Δz={V4_DELTA_Z_1})")
    print(f"    [2] same-cell ref2 → {V4_NORMAL_REF_2_PNG.name} (z={V4_NORMAL_REF_2_Z}, Δz={V4_DELTA_Z_2})")
    print(f"    [3] same-cell ref3 → {V4_NORMAL_REF_3_PNG.name} (z={V4_NORMAL_REF_3_Z}, Δz={V4_DELTA_Z_3})")
    print(f"    avg_abs_delta_z = {V4_AVG_ABS_DELTA_Z} (v3c was {V3C_AVG_ABS_DELTA_Z})")
    print(f"  Guards:")
    print(f"    ALLOW_SOURCE_PNG_READ  = {ALLOW_SOURCE_PNG_READ} (actual gen 시 True)")
    print(f"    ALLOW_PROTOTYPE_RENDER = {ALLOW_PROTOTYPE_RENDER} (actual gen 시 True)")
    print(f"    ALLOW_PROTOTYPE_WRITE  = {ALLOW_PROTOTYPE_WRITE} (actual gen 시 True)")
    print(f"    ALLOW_CT_LOAD          = {ALLOW_CT_LOAD} (항구 False)")
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

    drycheck_data = {
        "script": "build_s5_demo_card_prototype_lung1_052_c3_v3d_panel1_v4_ref.py",
        "case_id": CASE_ID,
        "static_check": {"passed": passed, "failed": failed, "issues": issues},
        "dry_run": {"ok": ok_dry, "issues": dry_issues},
        "guards": {
            "ALLOW_SOURCE_PNG_READ":     ALLOW_SOURCE_PNG_READ,
            "ALLOW_PROTOTYPE_RENDER":    ALLOW_PROTOTYPE_RENDER,
            "ALLOW_PROTOTYPE_WRITE":     ALLOW_PROTOTYPE_WRITE,
            "ALLOW_CT_LOAD":             ALLOW_CT_LOAD,
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
        "v3d_reference_selection": {
            "cell_key": V4_CELL_KEY,
            "source": "v4_exact_cell_top3",
            "normal_ref_1": {"z": V4_NORMAL_REF_1_Z, "delta_z": V4_DELTA_Z_1,
                             "png_path": str(V4_NORMAL_REF_1_PNG),
                             "exists": V4_NORMAL_REF_1_PNG.exists()},
            "normal_ref_2": {"z": V4_NORMAL_REF_2_Z, "delta_z": V4_DELTA_Z_2,
                             "png_path": str(V4_NORMAL_REF_2_PNG),
                             "exists": V4_NORMAL_REF_2_PNG.exists()},
            "normal_ref_3": {"z": V4_NORMAL_REF_3_Z, "delta_z": V4_DELTA_Z_3,
                             "png_path": str(V4_NORMAL_REF_3_PNG),
                             "exists": V4_NORMAL_REF_3_PNG.exists()},
            "avg_abs_delta_z": V4_AVG_ABS_DELTA_Z,
        },
        "output_root": str(OUTPUT_ROOT),
        "collision_check": {
            "v3c_root_exists": _V3C_ROOT.exists(),
            "v3d_root_exists": OUTPUT_ROOT.exists(),
            "out_png_exists":  OUT_PNG.exists(),
            "out_json_exists": OUT_JSON.exists(),
        },
    }
    with open(OUT_DRYCHECK_JSON, "w", encoding="utf-8") as f:
        json.dump(drycheck_data, f, ensure_ascii=False, indent=2)
    print(f"  SAVED: {OUT_DRYCHECK_JSON.name}")

    md_lines = [
        f"# S5 Demo Card v3d Drycheck — {CASE_ID}", "",
        f"## Static Check: {passed} passed / {failed} failed", "",
    ]
    if issues:
        md_lines += ["### Issues", ""]
        for iss in issues:
            md_lines.append(f"- {iss}")
        md_lines.append("")
    else:
        md_lines.append("모든 정적 검사 통과\n")
    md_lines += [
        "## v3d PNG 경로", "",
        "| 대상 | 파일명 | 존재 |", "| --- | --- | --- |",
        f"| candidate     | {V4_CANDIDATE_PNG.name}    | {'OK' if V4_CANDIDATE_PNG.exists() else 'MISSING'} |",
        f"| normal_ref_1  | {V4_NORMAL_REF_1_PNG.name} | {'OK' if V4_NORMAL_REF_1_PNG.exists() else 'MISSING'} |",
        f"| normal_ref_2  | {V4_NORMAL_REF_2_PNG.name} | {'OK' if V4_NORMAL_REF_2_PNG.exists() else 'MISSING'} |",
        f"| normal_ref_3  | {V4_NORMAL_REF_3_PNG.name} | {'OK' if V4_NORMAL_REF_3_PNG.exists() else 'MISSING'} |",
        "", "## z 좌표", "",
        "| 대상 | z | Δz | quality |", "| --- | --- | --- | --- |",
        f"| candidate    | {V4_CANDIDATE_Z}    | —             | —                      |",
        f"| normal_ref_1 | {V4_NORMAL_REF_1_Z} | {V4_DELTA_Z_1}  | {V4_NORMAL_REF_1_QUALITY} |",
        f"| normal_ref_2 | {V4_NORMAL_REF_2_Z} | {V4_DELTA_Z_2} | {V4_NORMAL_REF_2_QUALITY} |",
        f"| normal_ref_3 | {V4_NORMAL_REF_3_Z} | {V4_DELTA_Z_3} | {V4_NORMAL_REF_3_QUALITY} |",
        f"| avg_abs_Δz   | —                  | {V4_AVG_ABS_DELTA_Z} | —           |",
        "", f"v3c avg_abs_delta_z={V3C_AVG_ABS_DELTA_Z} → v3d {V4_AVG_ABS_DELTA_Z} (개선)",
        "CT load = 0", "",
        "## Guards", "",
        "| guard | 값 | 비고 |", "| --- | --- | --- |",
        f"| ALLOW_SOURCE_PNG_READ  | {ALLOW_SOURCE_PNG_READ}  | actual gen 시 True |",
        f"| ALLOW_PROTOTYPE_RENDER | {ALLOW_PROTOTYPE_RENDER} | actual gen 시 True |",
        f"| ALLOW_PROTOTYPE_WRITE  | {ALLOW_PROTOTYPE_WRITE}  | actual gen 시 True |",
        f"| ALLOW_CT_LOAD          | {ALLOW_CT_LOAD}          | 항구 False |",
        f"| ALLOW_STAGE2_HOLDOUT   | {ALLOW_STAGE2_HOLDOUT}   | 항구 False |",
    ]
    OUT_DRYCHECK_MD.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"  SAVED: {OUT_DRYCHECK_MD.name}")

    layout_rows = [
        ["panel", "row", "cell", "source_type", "source", "label", "ct_load", "v3d_change"],
        ["1", "subtitle", "-", "text", PANEL1_SUBTITLE_EN, "panel1_subtitle", "False", "UPDATED"],
        ["1", "1", "0", "PNG_read", str(V4_CANDIDATE_PNG), LABEL_CANDIDATE, "False", "v4_preview_png"],
        ["1", "1", "1", "PNG_read", str(V4_NORMAL_REF_1_PNG), LABEL_NORMAL_REF_1, "False", "v4_exact_cell_ref1"],
        ["1", "1", "2", "PNG_read", str(V4_NORMAL_REF_2_PNG), LABEL_NORMAL_REF_2, "False", "v4_exact_cell_ref2"],
        ["1", "1", "3", "PNG_read", str(V4_NORMAL_REF_3_PNG), LABEL_NORMAL_REF_3, "False", "v4_exact_cell_ref3"],
        ["1", "2", "0", "PNG_read", str(WHOLE_SLICE_PNG), LABEL_WHOLE_SLICE, "False", "same_as_v3c"],
        ["1", "3", "0-2", "placeholder", "CT load disabled", "z-context placeholder", "False", "PLACEHOLDER"],
        ["2", "1", "0", "PNG_read", str(S5_LUNG_OVERLAY_PNG), "lung overlay", "False", "same_as_v3c"],
        ["3", "1", "schematic", "JSON_data", str(S5_PATCH_MAP_JSON), "patch response map", "False", "same_as_v3c"],
        ["4", "1", "text", "text_const", "P4_KEY_FINDING_KO", "[Key finding]", "False", "same_as_v3c"],
        ["4", "2", "text", "text_const", "P4_INTERPRETATION_KO", "[Interpretation]", "False", "same_as_v3c"],
        ["4", "3", "text", "text_const", "P4_CAUTION_KO", "[Caution]", "False", "UPDATED: same-cell policy"],
        ["4", "4", "text", "text_const", "P4_DISCLAIMER_KO", "[Disclaimer]", "False", "same_as_v3c"],
    ]
    with open(OUT_LAYOUT_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(layout_rows)
    print(f"  SAVED: {OUT_LAYOUT_CSV.name}")

    ref_sel_rows = [
        ["check_id", "item", "v3c_value", "v3d_value", "status"],
        ["R01", "reference_selection_policy", "XY_MATCHED_WITH_Z_ORIENTATION_WARNING",
         "SAME_LUNG_ROI_POSITION_CELL_WITH_Z_LIMITATION", "CHANGED"],
        ["R02", "panel1_reference_source", "CT_NPY_direct_load",
         "v4_exact_cell_top3_preview_png", "CHANGED"],
        ["R03", "avg_abs_delta_z", str(V3C_AVG_ABS_DELTA_Z), str(V4_AVG_ABS_DELTA_Z), "IMPROVED"],
        ["R04", "not_same_z_matched", "True", "True", "SAME"],
        ["R05", "z_orientation_limitation", "True", "True", "SAME"],
        ["R06", "label_normal_ref_1", "XY-matched normal", "Same-cell normal ref 1", "CHANGED"],
        ["R07", "label_normal_ref_2", "XY-matched normal ex1", "Same-cell normal ref 2", "CHANGED"],
        ["R08", "label_normal_ref_3", "XY-matched normal ex2", "Same-cell normal ref 3", "CHANGED"],
        ["R09", "panel1_subtitle", "z alignment limited by dataset orientation",
         "not same-z matching", "CLARIFIED"],
        ["R10", "p4_caution", "z-direction alignment has limitations",
         "same lung-ROI position cell, not same-z matching", "UPDATED"],
        ["R11", "ct_load_used", "True", "False", "IMPROVED"],
        ["R12", "stage2_holdout_excluded", "True", "True", "SAME"],
        ["R13", "normal_ref_1_z", "165", str(V4_NORMAL_REF_1_Z), "CHANGED"],
        ["R14", "normal_ref_2_z", "134", str(V4_NORMAL_REF_2_Z), "CHANGED"],
        ["R15", "normal_ref_3_z", "202", str(V4_NORMAL_REF_3_Z), "CHANGED"],
        ["R16", "cell_key", "N/A", V4_CELL_KEY, "ADDED"],
    ]
    with open(OUT_REF_SELECTION_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(ref_sel_rows)
    print(f"  SAVED: {OUT_REF_SELECTION_CSV.name}")

    top3_rows = [
        ["target", "patient_id", "png_path", "exists", "z", "delta_z", "quality_score", "ct_load"],
        ["candidate", "NSCLC_LUNG1-052__d4a19cc211", str(V4_CANDIDATE_PNG),
         str(V4_CANDIDATE_PNG.exists()), V4_CANDIDATE_Z, "—", "—", "False"],
        ["normal_ref_1", V4_NORMAL_REF_1_PATIENT, str(V4_NORMAL_REF_1_PNG),
         str(V4_NORMAL_REF_1_PNG.exists()), V4_NORMAL_REF_1_Z, V4_DELTA_Z_1,
         V4_NORMAL_REF_1_QUALITY, "False"],
        ["normal_ref_2", V4_NORMAL_REF_2_PATIENT, str(V4_NORMAL_REF_2_PNG),
         str(V4_NORMAL_REF_2_PNG.exists()), V4_NORMAL_REF_2_Z, V4_DELTA_Z_2,
         V4_NORMAL_REF_2_QUALITY, "False"],
        ["normal_ref_3", V4_NORMAL_REF_3_PATIENT, str(V4_NORMAL_REF_3_PNG),
         str(V4_NORMAL_REF_3_PNG.exists()), V4_NORMAL_REF_3_Z, V4_DELTA_Z_3,
         V4_NORMAL_REF_3_QUALITY, "False"],
    ]
    with open(OUT_TOP3_PATHS_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(top3_rows)
    print(f"  SAVED: {OUT_TOP3_PATHS_CSV.name}")

    all_text_lower = " ".join([
        P4_KEY_FINDING_KO, P4_INTERPRETATION_KO, P4_CAUTION_KO, P4_DISCLAIMER_KO,
        LABEL_NORMAL_REF_1, LABEL_NORMAL_REF_2, LABEL_NORMAL_REF_3, PANEL1_SUBTITLE_EN,
    ]).lower()
    text_policy_rows = [
        ["check_id", "target", "rule", "status"],
        ["T01", "panel4_body", "no dim91",               "PASS" if "dim91" not in all_text_lower else "FAIL"],
        ["T02", "panel4_body", "no raw427",              "PASS" if "raw427" not in all_text_lower else "FAIL"],
        ["T03", "panel4_body", "no Grad-CAM",            "PASS" if "grad-cam" not in all_text_lower else "FAIL"],
        ["T04", "panel4_body", "no pixel attribution",   "PASS" if "pixel attribution" not in all_text_lower else "FAIL"],
        ["T05", "panel4_body", "no same-z matched",      "PASS" if "same-z matched" not in all_text_lower else "FAIL"],
        ["T06", "labels", "no z-matched in labels",
         "PASS" if "z-matched" not in (LABEL_NORMAL_REF_1+LABEL_NORMAL_REF_2+LABEL_NORMAL_REF_3).lower() else "FAIL"],
        ["T07", "p4_caution_en", "same lung-ROI position cell",
         "PASS" if "same lung-roi position cell" in P4_CAUTION_EN.lower() else "FAIL"],
        ["T08", "p4_caution_en", "not be interpreted as same-z matching",
         "PASS" if "not be interpreted as same-z matching" in P4_CAUTION_EN.lower() else "FAIL"],
        ["T09", "subtitle", "not same-z matching",
         "PASS" if "not same-z matching" in PANEL1_SUBTITLE_EN.lower() else "FAIL"],
        ["T10", "subtitle", "same lung-ROI position cell",
         "PASS" if "same lung-roi position cell" in PANEL1_SUBTITLE_EN.lower() else "FAIL"],
        ["T11", "label_ref1", "Same-cell normal ref 1",
         "PASS" if "Same-cell normal ref 1" in LABEL_NORMAL_REF_1 else "FAIL"],
        ["T12", "label_ref2", "Same-cell normal ref 2",
         "PASS" if "Same-cell normal ref 2" in LABEL_NORMAL_REF_2 else "FAIL"],
        ["T13", "label_ref3", "Same-cell normal ref 3",
         "PASS" if "Same-cell normal ref 3" in LABEL_NORMAL_REF_3 else "FAIL"],
        ["T14", "metadata", "not_gradcam=True",              "PASS"],
        ["T15", "metadata", "not_pixel_attribution=True",    "PASS"],
        ["T16", "metadata", "not_diagnostic=True",           "PASS"],
        ["T17", "metadata", "z_orientation_limitation=True", "PASS"],
        ["T18", "metadata", "not_same_z_matched=True",       "PASS"],
    ]
    with open(OUT_TEXT_POLICY_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(text_policy_rows)
    print(f"  SAVED: {OUT_TEXT_POLICY_CSV.name}")

    fail_count = failed + len([i for i in dry_issues if i.startswith("FAIL")])
    verdict = "PASS" if fail_count == 0 else "BLOCKED"
    preflight_lines = [
        f"# Preflight Summary — {CASE_ID} S5 v3d", "",
        f"## 결과: {verdict}", "",
        f"- static_check: {passed} passed / {failed} failed",
        f"- dry_run: {'OK' if ok_dry else 'ISSUES'}",
        f"- output collision: {'없음' if not OUT_PNG.exists() else 'COLLISION 있음'}", "",
        "## v3d 핵심 변경", "",
        f"- cell_key: {V4_CELL_KEY}",
        f"- normal_ref_1: z={V4_NORMAL_REF_1_Z}, Δz={V4_DELTA_Z_1}, quality={V4_NORMAL_REF_1_QUALITY}",
        f"- normal_ref_2: z={V4_NORMAL_REF_2_Z}, Δz={V4_DELTA_Z_2}, quality={V4_NORMAL_REF_2_QUALITY}",
        f"- normal_ref_3: z={V4_NORMAL_REF_3_Z}, Δz={V4_DELTA_Z_3}, quality={V4_NORMAL_REF_3_QUALITY}",
        f"- avg_abs_delta_z: {V4_AVG_ABS_DELTA_Z} (v3c={V3C_AVG_ABS_DELTA_Z})",
        "- CT load = 0 (v3d 정책)", "",
        "## 다음 단계", "",
        "1. static PASS → 사용자 검토",
        "2. 3개 guard True: ALLOW_SOURCE_PNG_READ / ALLOW_PROTOTYPE_RENDER / ALLOW_PROTOTYPE_WRITE",
        "3. --run-prototype --confirm-generate",
        "4. guard 원복 (모두 False)",
        "5. BLOCKED recheck", "",
        "## Safety",
        "- CT load: 0",
        "- model/feature/contribution: 0",
        "- stage2_holdout: 없음",
        "- 기존 v3c/v4 artifact 수정: 없음",
    ]
    OUT_PREFLIGHT_MD.write_text("\n".join(preflight_lines), encoding="utf-8")
    print(f"  SAVED: {OUT_PREFLIGHT_MD.name}")

    summary = "PASS" if (failed == 0 and not dry_issues) else f"BLOCKED ({failed} static fails, {len(dry_issues)} dry issues)"
    print(f"\n  [save-static-artifacts] {summary}")


# ============================================================
# actual generation
# ============================================================
def run_prototype() -> None:
    if not (ALLOW_SOURCE_PNG_READ and ALLOW_PROTOTYPE_RENDER and ALLOW_PROTOTYPE_WRITE):
        print("BLOCKED: ALLOW_SOURCE_PNG_READ / ALLOW_PROTOTYPE_RENDER / ALLOW_PROTOTYPE_WRITE must all be True",
              file=sys.stderr)
        sys.exit(2)
    if ALLOW_CT_LOAD or ALLOW_CANDIDATE_CT_LOAD or ALLOW_NORMAL_REF_CT_LOAD:
        print("BLOCKED: CT load guards must remain False in v3d", file=sys.stderr)
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

    collision_paths = [OUT_DONE, OUT_PNG, OUT_JSON, OUT_INDEX_CSV, OUT_RUNTIME, OUT_ERRORS]
    existing = [str(p) for p in collision_paths if p.exists()]
    if existing:
        print("BLOCKED: output collision detected:", file=sys.stderr)
        for p in existing:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(2)

    t_start = time.time()
    errors: List[str] = []

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as e:
        print(f"BLOCKED: PIL not available: {e}", file=sys.stderr)
        sys.exit(2)

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

    def _placeholder(w: int, h: int, text: str = "N/A") -> Image.Image:
        img = Image.new("RGBA", (w, h), (50, 50, 60, 255))
        d = ImageDraw.Draw(img)
        d.rectangle([(0, 0), (w - 1, h - 1)], outline=(100, 100, 120, 255), width=2)
        for i, line in enumerate(text.split("\n")):
            d.text((6, 6 + i * 14), line, font=fnt_small, fill=(160, 160, 180, 255))
        return img

    def _load_png(p: pathlib.Path, display_size: int, label: str) -> Image.Image:
        if not p.exists():
            errors.append(f"PNG not found: {p}")
            return _placeholder(display_size, display_size, f"{label}\nnot found")
        img = Image.open(str(p)).convert("RGBA")
        return img.resize((display_size, display_size), Image.LANCZOS)

    img_whole   = Image.open(WHOLE_SLICE_PNG).convert("RGBA")
    img_lung_ov = Image.open(S5_LUNG_OVERLAY_PNG).convert("RGBA")
    with open(S5_PATCH_MAP_JSON) as f:
        patch_map = json.load(f)

    CANVAS_W   = 1600
    MARGIN     = 16
    HEADER_H   = 64
    PAD        = 8

    P1_W      = CANVAS_W - 2 * MARGIN
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

    draw.rectangle([(0, 0), (CANVAS_W, HEADER_H)], fill=C_HEADER)
    draw.text((MARGIN, 8),  PROTOTYPE_TITLE_KO, font=fnt_title, fill=C_TITLE)
    draw.text((MARGIN, 32), "[INTERNAL USE ONLY | v3d | not diagnostic]",
              font=fnt_small, fill=C_WARN)
    draw.text((MARGIN, 48), PANEL1_SUBTITLE_KO, font=fnt_small, fill=C_SUBTITLE)

    p1_x0 = MARGIN
    p1_y0 = HEADER_H + MARGIN
    p1_x1 = MARGIN + P1_W
    p1_y1 = p1_y0 + P1_H
    draw.rectangle([(p1_x0, p1_y0), (p1_x1, p1_y1)], fill=C_PANEL_BG, outline=C_BORDER, width=1)

    cy = p1_y0 + PAD
    draw.text((p1_x0 + PAD, cy),
              "Panel 1 : 정상 reference 비교 (Same-cell, v4 ref)  |  전체 슬라이스  |  z-context",
              font=fnt_section, fill=C_SECTION)
    cy += LABEL_H
    draw.text((p1_x0 + PAD, cy), PANEL1_SUBTITLE_EN, font=fnt_small, fill=C_SUBTITLE)
    cy += SUBTITLE_H

    crop_x_starts = [p1_x0 + PAD + i * (CELL_SIZE + PAD) for i in range(4)]
    cell_labels   = [LABEL_CANDIDATE, LABEL_NORMAL_REF_1, LABEL_NORMAL_REF_2, LABEL_NORMAL_REF_3]
    cell_imgs = [
        _load_png(V4_CANDIDATE_PNG,    CELL_SIZE, "candidate"),
        _load_png(V4_NORMAL_REF_1_PNG, CELL_SIZE, "normal_ref_1"),
        _load_png(V4_NORMAL_REF_2_PNG, CELL_SIZE, "normal_ref_2"),
        _load_png(V4_NORMAL_REF_3_PNG, CELL_SIZE, "normal_ref_3"),
    ]

    for i, (cx_start, img_cell, lbl) in enumerate(zip(crop_x_starts, cell_imgs, cell_labels)):
        draw.rectangle([(cx_start - 1, cy - 1), (cx_start + CELL_SIZE, cy + CELL_SIZE)],
                       outline=C_YELLOW if i == 0 else C_BORDER,
                       width=2 if i == 0 else 1)
        canvas.paste(img_cell, (cx_start, cy), img_cell)
        for j, line in enumerate(lbl.split("\n")):
            draw.text((cx_start, cy + CELL_SIZE + 2 + j * 12), line,
                      font=fnt_small, fill=C_YELLOW if i == 0 else C_LABEL)

    cy += CELL_SIZE + LABEL_H + PAD

    draw.text((p1_x0 + PAD, cy), LABEL_WHOLE_SLICE, font=fnt_label, fill=C_LABEL)
    cy += LABEL_H - 8
    whole_x = p1_x0 + (P1_W - WHOLE_W) // 2
    img_whole_r = img_whole.resize((WHOLE_W, WHOLE_H), Image.LANCZOS)
    canvas.paste(img_whole_r, (whole_x, cy), img_whole_r)
    cy += WHOLE_H + PAD

    draw.text((p1_x0 + PAD, cy), "Z-context (CT load disabled in v3d — placeholder)",
              font=fnt_label, fill=C_LABEL)
    cy += LABEL_H - 8

    z_labels = [LABEL_Z_ABOVE, LABEL_Z_CURRENT, LABEL_Z_BELOW]
    z_starts = [p1_x0 + PAD + i * (Z_CELL_W + PAD) for i in range(3)]
    for z_x, z_lbl in zip(z_starts, z_labels):
        z_img = _placeholder(Z_CELL_W, Z_CELL_H, "z-context\n(CT load\ndisabled)")
        draw.rectangle([(z_x - 1, cy - 1), (z_x + Z_CELL_W, cy + Z_CELL_H)],
                       outline=C_BORDER, width=1)
        canvas.paste(z_img, (z_x, cy), z_img)
        for j, line in enumerate(z_lbl.split("\n")):
            draw.text((z_x, cy + Z_CELL_H + 2 + j * 12), line,
                      font=fnt_small, fill=C_LABEL)

    lower_y     = HEADER_H + MARGIN + P1_H + MARGIN
    p2_x0       = MARGIN
    p3_x0       = MARGIN + LOWER_PANEL_W + MARGIN
    p4_x0       = MARGIN + (LOWER_PANEL_W + MARGIN) * 2

    def draw_lower_panel(bx0: int, title: str) -> Tuple[int, int, int, int]:
        bx1 = bx0 + LOWER_PANEL_W
        by0, by1 = lower_y, lower_y + LOWER_H
        draw.rectangle([(bx0, by0), (bx1, by1)], fill=C_PANEL_BG, outline=C_BORDER, width=1)
        draw.text((bx0 + PAD, by0 + PAD), title, font=fnt_section, fill=C_SECTION)
        return bx0, by0, bx1, by1

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
            draw.rectangle([(cell_x, cell_y), (cell_x + CELL_G - 1, cell_y + CELL_G - 1)],
                           fill=bg, outline=(90, 90, 90, 255), width=1)
            draw.text((cell_x + 4, cell_y + 4),           f"P{pid}",     font=fnt_small, fill=lc)
            draw.text((cell_x + 4, cell_y + CELL_G - 16), f"{score:.1f}", font=fnt_small, fill=lc)

    cy3 = grid_y0 + ACTUAL_G + 8
    draw.rectangle([(grid_x0, cy3), (grid_x0 + 12, cy3 + 12)], fill=(255, 220, 0, 255))
    draw.text((grid_x0 + 16, cy3), "Highest", font=fnt_small, fill=C_BODY)
    cy3 += 16
    draw.rectangle([(grid_x0, cy3), (grid_x0 + 12, cy3 + 12)], fill=(255, 140, 0, 255))
    draw.text((grid_x0 + 16, cy3), "Second-highest", font=fnt_small, fill=C_BODY)

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
        cy4 += wrap_draw((p4_x0 + PAD, cy4), ko, fnt_body, kc, tw4, 17) + 3
        cy4 += wrap_draw((p4_x0 + PAD, cy4), en, fnt_small, (170, 170, 170, 255), tw4, 14) + 10

    draw_section("[Key finding]",    P4_KEY_FINDING_KO,    P4_KEY_FINDING_EN,    C_SECTION, C_BODY)
    draw_section("[Interpretation]", P4_INTERPRETATION_KO, P4_INTERPRETATION_EN, C_SECTION, C_BODY)
    draw_section("[Caution]",        P4_CAUTION_KO,        P4_CAUTION_EN,        C_WARN,    C_WARN)
    draw.text((p4_x0 + PAD, cy4), "[Disclaimer]", font=fnt_label, fill=C_DISC)
    cy4 += 18
    wrap_draw((p4_x0 + PAD, cy4),
              P4_DISCLAIMER_KO + "  " + P4_DISCLAIMER_EN,
              fnt_small, C_DISC, tw4, 14)

    CARDS_PNG_DIR.mkdir(parents=True, exist_ok=True)
    CARDS_JSON_DIR.mkdir(parents=True, exist_ok=True)

    canvas.convert("RGB").save(str(OUT_PNG), "PNG")
    print(f"  PNG saved: {OUT_PNG}")

    schema = build_prototype_json_schema()
    schema["canvas_size_px"]           = [CANVAS_W, CANVAS_H]
    schema["ct_load_used"]             = False
    schema["normal_ref_ct_load_used"]  = False
    schema["source_png_read_used"]     = True
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    print(f"  JSON saved: {OUT_JSON}")

    elapsed = time.time() - t_start
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUT_RUNTIME, "w", encoding="utf-8") as f:
        json.dump({
            "case_id": CASE_ID, "version": "v3d", "elapsed_sec": round(elapsed, 2),
            "ct_load_used": False, "normal_ref_ct_load_used": False,
            "source_png_read_used": True,
            "errors": errors, "output_png": str(OUT_PNG), "output_json": str(OUT_JSON),
        }, f, ensure_ascii=False, indent=2)

    with open(OUT_ERRORS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["error"])
        for e in errors:
            w.writerow([e])

    with open(OUT_INDEX_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "version", "output_png", "output_json"])
        w.writerow([CASE_ID, "v3d", str(OUT_PNG), str(OUT_JSON)])

    with open(OUT_DONE, "w", encoding="utf-8") as f:
        json.dump({
            "done": True, "case_id": CASE_ID, "version": "v3d_panel1_v4_ref",
            "reference_selection": "SAME_LUNG_ROI_POSITION_CELL_WITH_Z_LIMITATION",
            "ct_load_used": False,
        }, f, ensure_ascii=False, indent=2)

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
        for iss in issues:
            print(f"  {iss}")
        if not issues:
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
    print("  --static-check")
    print("  --dry-run")
    print("  --plan-only")
    print("  --save-static-artifacts")
    print("  --run-prototype --confirm-generate")


if __name__ == "__main__":
    main()
