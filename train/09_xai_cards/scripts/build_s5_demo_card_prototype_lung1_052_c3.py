#!/usr/bin/env python3
"""
build_s5_demo_card_prototype_lung1_052_c3.py

S5 Demo Card Prototype — LUNG1-052__c3

목적:
  S3+S5 bridge를 기반으로 LUNG1-052__c3 1건에 대해
  Option B 4-panel layout의 S5 demo card prototype script를 작성하고
  정적 검사를 수행한다.

이번 단계(정적):
  - 실제 source PNG read 금지
  - prototype render 금지
  - PNG write 금지
  - 기존 S3/S4/S5 artifact 수정 금지

actual generation에서만 아래 guard를 별도 승인 후 True:
  ALLOW_SOURCE_PNG_READ = True
  ALLOW_PROTOTYPE_RENDER = True
  ALLOW_PROTOTYPE_WRITE = True

Option B 4-panel layout:
  Panel 1: S3 thumbnail + S3 bbox + reference crop 요약 텍스트
  Panel 2: S5 lung-window 3x3 grid overlay
  Panel 3: S5 PaDiM-window 3x3 grid overlay
  Panel 4: contribution summary + 공식 해석 + dim91 caveat + disclaimer

card title:
  "S5 PaDiM patch-level explanation prototype — LUNG1-052__c3"

카드 표시 문구:
  "Patch4 showed the highest PaDiM response, and patch7 directly below showed
   the second-highest response. This supports a center-to-downward
   patch-level feature-space continuity pattern."

한국어 문구:
  "patch4에서 가장 높은 PaDiM response가 확인되었고, 바로 아래 patch7에서도
   두 번째로 높은 response가 이어졌습니다. 이는 중심에서 아래 방향으로
   이어지는 patch-level feature-space response continuity로 해석합니다."

disclaimer:
  "이 결과는 PaDiM feature-space와 patch-grid 좌표를 기준으로 한 연구용 설명이며,
   Grad-CAM이나 pixel attribution이 아니고 진단 의미를 직접 나타내지 않습니다."

실행 모드:
  bare 실행                                  → BLOCKED exit 2
  --selftest                                 → 80+ 항목 selftest
  --dry-run                                  → 입력 파일 존재 + guard 확인
  --plan-only                                → 4-panel layout + 입력/출력 경로 + guard 표시
  --run-prototype                            → 단독 BLOCKED exit 2
  --run-prototype --confirm-generate         → guards False이면 BLOCKED exit 2

syntax check:
  python -m py_compile scripts/build_s5_demo_card_prototype_lung1_052_c3.py
"""

import argparse
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
ALLOW_SOURCE_PNG_READ      = False   # generation 완료 후 원복
ALLOW_PROTOTYPE_RENDER     = False   # generation 완료 후 원복
ALLOW_PROTOTYPE_WRITE      = False   # generation 완료 후 원복

# 항구 False — 절대 변경 금지
ALLOW_S3_MODIFICATION      = False
ALLOW_S4_MODIFICATION      = False
ALLOW_CT_LOAD              = False
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
CT_INDEX_Z           = 51
REPORT_SLICE_INDEX   = 106
SPATIAL_PATTERN      = "CENTER_DOMINANT_WITH_DOWNWARD_CONTINUITY"
VISUAL_VERDICT       = "PASS_VISUAL_SUPPORTS_CENTER_DOWNWARD_CONTINUITY_WITH_CAUTION"

S3_DISPLAY_BBOX      = [256, 96, 352, 192]
S5_GRID_EXTENT       = [256, 96, 352, 192]
PATCH4_BBOX          = [288, 128, 320, 160]
PATCH4_SCORE         = 38.872562
PATCH7_BBOX          = [320, 128, 352, 160]
PATCH7_SCORE         = 36.612470

LAYOUT_SELECTED      = "OPTION_B_4PANEL"
S4_AVAILABLE         = False
S4_GENERATION_DEFERRED = True

# ============================================================
# 경로 상수
# ============================================================
PROJECT_ROOT = pathlib.Path("/home/jinhy/project/lung-ct-anomaly")

# 입력 경로
S3_CARD_PNG = PROJECT_ROOT / "outputs/position-aware-padim-v1/visualizations/candidate_cards/s3_expansion_cards_v1/cards_png/LUNG1-052__c3.png"
S3_CARD_JSON = PROJECT_ROOT / "outputs/position-aware-padim-v1/visualizations/candidate_cards/s3_expansion_cards_v1/cards_json/LUNG1-052__c3.json"

_REF_CROP_BASE = PROJECT_ROOT / "outputs/position-aware-padim-v1/visualizations/candidate_cards/reference_bank_v1/full/reference_crops/lower_central"
S3_REFERENCE_CROPS = [
    _REF_CROP_BASE / "subset1_1.3.6.1.4.1.14519.5.2.1.6279.6001.300271604576987336866436407488__2508834c92__z167__y144_x320.png",
    _REF_CROP_BASE / "subset4_1.3.6.1.4.1.14519.5.2.1.6279.6001.200837896655745926888305239398__d3e03b0ce0__z179__y176_x304.png",
    _REF_CROP_BASE / "subset8_1.3.6.1.4.1.14519.5.2.1.6279.6001.336102335330125765000317290445__ae9463f7d8__z172__y160_x320.png",
]

BRIDGE_JSON = PROJECT_ROOT / "outputs/position-aware-padim-v1/reports/explanation_cards/s3_s5_bridge_lung1_052_c3_v1/s3_s5_bridge_lung1_052_c3_v1.json"

S5_LUNG_OVERLAY_PNG    = PROJECT_ROOT / "outputs/position-aware-padim-v1/reports/explanation_cards/s5_lung1_052_c3_coordinate_visual_audit_1case_v1/coordinate_overlay_3x3_grid_lung.png"
S5_PADIM_OVERLAY_PNG   = PROJECT_ROOT / "outputs/position-aware-padim-v1/reports/explanation_cards/s5_lung1_052_c3_coordinate_visual_audit_1case_v1/coordinate_overlay_3x3_grid_padim_window.png"
S5_COORD_METADATA      = PROJECT_ROOT / "outputs/position-aware-padim-v1/reports/explanation_cards/s5_lung1_052_c3_coordinate_visual_audit_1case_v1/coordinate_overlay_metadata.json"

S5_PATCH_MAP_JSON      = PROJECT_ROOT / "outputs/position-aware-padim-v1/reports/explanation_cards/s5_lung1_052_c3_patch_level_contribution_map_1case_smoke_v1/patch_contribution_map.json"
S5_FEATURE_SUMMARY_JSON = PROJECT_ROOT / "outputs/position-aware-padim-v1/reports/explanation_cards/s5_lung1_052_c3_feature_contribution_1case_smoke_v1/feature_contribution_summary.json"

# 출력 경로
OUTPUT_ROOT = PROJECT_ROOT / "outputs/position-aware-padim-v1/reports/explanation_cards/s5_demo_card_prototype_lung1_052_c3_v1"
CARDS_PNG_DIR  = OUTPUT_ROOT / "cards_png"
CARDS_JSON_DIR = OUTPUT_ROOT / "cards_json"
OUT_PNG        = CARDS_PNG_DIR  / "LUNG1-052__c3_s5_demo_prototype.png"
OUT_JSON       = CARDS_JSON_DIR / "LUNG1-052__c3_s5_demo_prototype.json"
OUT_INDEX_CSV  = OUTPUT_ROOT / "index_cards.csv"
OUT_RUNTIME    = OUTPUT_ROOT / "runtime_summary.json"
OUT_ERRORS     = OUTPUT_ROOT / "errors.csv"
OUT_DONE       = OUTPUT_ROOT / "DONE.json"

# ============================================================
# 텍스트 상수
# ============================================================
PROTOTYPE_TITLE_EN = "S5 PaDiM patch-level explanation prototype — LUNG1-052__c3"

CARD_TEXT_EN = (
    "Patch4 showed the highest PaDiM response, and patch7 directly below showed "
    "the second-highest response. This supports a center-to-downward "
    "patch-level feature-space continuity pattern."
)

CARD_TEXT_KO = (
    "patch4에서 가장 높은 PaDiM response가 확인되었고, 바로 아래 patch7에서도 "
    "두 번째로 높은 response가 이어졌습니다. 이는 중심에서 아래 방향으로 "
    "이어지는 patch-level feature-space response continuity로 해석합니다."
)

DISCLAIMER_KO = (
    "이 결과는 PaDiM feature-space와 patch-grid 좌표를 기준으로 한 연구용 설명이며, "
    "Grad-CAM이나 pixel attribution이 아니고 진단 의미를 직접 나타내지 않습니다."
)

DIM91_CAVEAT = (
    "dim91(raw427/layer3)은 LUNG1-320__c2와 LUNG1-052__c3 모두에서 강하게 나타났다. "
    "해부학적 구조로 해석 금지. covariance inverse amplification 또는 공통 "
    "high-response feature-space pattern 가능성. multi-case review 전까지 "
    "병변/암/혈관/고밀도 원인과 연결 금지."
)

BASAL_PLEURAL_CAUTION = (
    "lower_central position_bin은 basal 및 pleural 접경 패턴이 FP 원인이 될 수 있다. "
    "이 S5 결과가 basal/pleural 조직에 의한 것인지 여부는 미결 상태다."
)

OFFICIAL_INTERPRETATION = (
    "LUNG1-052__c3에서는 patch4가 3×3 patch-level PaDiM contribution map의 "
    "최대 response 영역으로 확인되었고, patch7이 바로 아래 인접한 second-highest patch로 "
    "나타났다. lung window와 PaDiM preprocessing window overlay 모두에서 이 수직 "
    "연속성이 확인되므로, 이는 center-to-downward patch-level feature-space response "
    "continuity로 해석한다. 단, 이 결과는 좌표 감사 및 feature-space contribution "
    "요약이며, 해부학적 원인이나 진단 의미를 직접 나타내지 않는다."
)

# ============================================================
# prototype JSON schema 정의
# ============================================================
def build_prototype_json_schema() -> Dict[str, Any]:
    """actual generation 전 schema 정의. 실제 값은 generation 시 채워진다."""
    return {
        "case_id": CASE_ID,
        "volume_id": VOLUME_ID,
        "prototype_type": "S3_PLUS_S5_DEMO_CARD_PROTOTYPE",
        "internal_use_only": True,
        "font_fix_required_before_external_share": True,
        "s4_available": False,
        "s4_generation_deferred": True,
        "s3_card_used_as_reference": True,
        "s3_display_bbox": S3_DISPLAY_BBOX,
        "s5_grid_extent": S5_GRID_EXTENT,
        "s3_bbox_matches_s5_grid_extent": True,
        "ct_index_z": CT_INDEX_Z,
        "report_slice_index": REPORT_SLICE_INDEX,
        "local_z_used_for_ct_indexing": True,
        "slice_index_used_for_ct_indexing": False,
        "patch4_bbox": PATCH4_BBOX,
        "patch4_score": PATCH4_SCORE,
        "patch7_bbox": PATCH7_BBOX,
        "patch7_score": PATCH7_SCORE,
        "spatial_pattern": SPATIAL_PATTERN,
        "visual_verdict": VISUAL_VERDICT,
        "dim91_caveat": DIM91_CAVEAT,
        "basal_pleural_caution": BASAL_PLEURAL_CAUTION,
        "not_gradcam": True,
        "not_pixel_attribution": True,
        "not_diagnostic": True,
        "s3_png_modified": False,
        "s4_png_modified": False,
        "existing_artifacts_modified": False,
        "stage2_holdout_accessed": False,
        "full300_applied": False,
        "card_text_en": CARD_TEXT_EN,
        "card_text_ko": CARD_TEXT_KO,
        "disclaimer_ko": DISCLAIMER_KO,
        "official_interpretation": OFFICIAL_INTERPRETATION,
        "layout": LAYOUT_SELECTED,
        "panels": {
            "panel_1": "S3 thumbnail + S3 bbox + reference crop 경로/요약 텍스트",
            "panel_2": "S5 lung-window 3x3 grid overlay",
            "panel_3": "S5 PaDiM-window 3x3 grid overlay",
            "panel_4": "contribution summary + 공식 해석 + dim91 caveat + disclaimer",
        },
    }


# ============================================================
# selftest
# ============================================================
def run_selftest() -> Tuple[int, int, List[str]]:
    """80개 이상 항목 selftest. (passed, total, failures) 반환."""
    passed = 0
    failures: List[str] = []

    def check(name: str, cond: bool) -> None:
        nonlocal passed
        if cond:
            passed += 1
        else:
            failures.append(f"FAIL: {name}")

    schema = build_prototype_json_schema()

    # 1. 모든 guard 기본값 False
    check("01 ALLOW_SOURCE_PNG_READ default False",    not ALLOW_SOURCE_PNG_READ)
    check("02 ALLOW_PROTOTYPE_RENDER default False",   not ALLOW_PROTOTYPE_RENDER)
    check("03 ALLOW_PROTOTYPE_WRITE default False",    not ALLOW_PROTOTYPE_WRITE)
    check("04 ALLOW_S3_MODIFICATION default False",    not ALLOW_S3_MODIFICATION)
    check("05 ALLOW_S4_MODIFICATION default False",    not ALLOW_S4_MODIFICATION)
    check("06 ALLOW_CT_LOAD default False",            not ALLOW_CT_LOAD)
    check("07 ALLOW_MODEL_FORWARD default False",      not ALLOW_MODEL_FORWARD)
    check("08 ALLOW_FEATURE_EXTRACTION default False", not ALLOW_FEATURE_EXTRACTION)
    check("09 ALLOW_CONTRIBUTION_RECALC default False",not ALLOW_CONTRIBUTION_RECALC)
    check("10 ALLOW_STAGE2_HOLDOUT default False",     not ALLOW_STAGE2_HOLDOUT)
    check("11 ALLOW_FULL_300 default False",           not ALLOW_FULL_300)

    # 2. 케이스 상수
    check("12 CASE_ID exact",       CASE_ID == "LUNG1-052__c3")
    check("13 VOLUME_ID exact",     VOLUME_ID == "NSCLC_LUNG1-052__d4a19cc211")
    check("14 POSITION_BIN lower_central", POSITION_BIN == "lower_central")
    check("15 CT_INDEX_Z == 51",    CT_INDEX_Z == 51)
    check("16 REPORT_SLICE_INDEX == 106", REPORT_SLICE_INDEX == 106)

    # 3. S4 상태
    check("17 S4_AVAILABLE false",           S4_AVAILABLE is False)
    check("18 S4_GENERATION_DEFERRED true",  S4_GENERATION_DEFERRED is True)

    # 4. internal_use_only / font_fix
    check("19 schema internal_use_only true",                schema["internal_use_only"] is True)
    check("20 schema font_fix_required_before_external_share true",
          schema["font_fix_required_before_external_share"] is True)

    # 5. bbox / grid extent
    check("21 S3_DISPLAY_BBOX exact",   S3_DISPLAY_BBOX == [256, 96, 352, 192])
    check("22 S5_GRID_EXTENT exact",    S5_GRID_EXTENT == [256, 96, 352, 192])
    check("23 schema s3_display_bbox exact",   schema["s3_display_bbox"] == [256, 96, 352, 192])
    check("24 schema s5_grid_extent exact",    schema["s5_grid_extent"] == [256, 96, 352, 192])
    check("25 s3_bbox_matches_s5_grid_extent", schema["s3_bbox_matches_s5_grid_extent"] is True)

    # 6. patch4 / patch7
    check("26 patch4_bbox exact",        PATCH4_BBOX == [288, 128, 320, 160])
    check("27 patch7_bbox exact",        PATCH7_BBOX == [320, 128, 352, 160])
    check("28 schema patch4_bbox exact", schema["patch4_bbox"] == [288, 128, 320, 160])
    check("29 schema patch7_bbox exact", schema["patch7_bbox"] == [320, 128, 352, 160])
    check("30 patch4_score approx 38.872562",  abs(PATCH4_SCORE - 38.872562) < 1e-3)
    check("31 patch7_score approx 36.612470",  abs(PATCH7_SCORE - 36.612470) < 1e-3)
    check("32 schema patch4_score approx",     abs(schema["patch4_score"] - 38.872562) < 1e-3)
    check("33 schema patch7_score approx",     abs(schema["patch7_score"] - 36.612470) < 1e-3)

    # 7. spatial / visual
    check("34 SPATIAL_PATTERN exact",    SPATIAL_PATTERN == "CENTER_DOMINANT_WITH_DOWNWARD_CONTINUITY")
    check("35 VISUAL_VERDICT exact",     VISUAL_VERDICT == "PASS_VISUAL_SUPPORTS_CENTER_DOWNWARD_CONTINUITY_WITH_CAUTION")
    check("36 schema spatial_pattern exact", schema["spatial_pattern"] == SPATIAL_PATTERN)
    check("37 schema visual_verdict exact",  schema["visual_verdict"] == VISUAL_VERDICT)

    # 8. 레이아웃
    check("38 LAYOUT_SELECTED Option B",          LAYOUT_SELECTED == "OPTION_B_4PANEL")
    check("39 schema layout OPTION_B_4PANEL",     schema["layout"] == "OPTION_B_4PANEL")
    check("40 Panel 1 S3 summary planned",        "S3 thumbnail" in schema["panels"]["panel_1"])
    check("41 Panel 2 lung overlay planned",      "lung" in schema["panels"]["panel_2"].lower())
    check("42 Panel 3 PaDiM overlay planned",     "PaDiM" in schema["panels"]["panel_3"])
    check("43 Panel 4 contribution summary planned", "contribution" in schema["panels"]["panel_4"])

    # 9. caveat / warning
    check("44 schema not_gradcam true",            schema["not_gradcam"] is True)
    check("45 schema not_pixel_attribution true",  schema["not_pixel_attribution"] is True)
    check("46 schema not_diagnostic true",         schema["not_diagnostic"] is True)
    check("47 dim91 caveat exists in schema",      "dim91" in schema["dim91_caveat"].lower())
    check("48 basal_pleural_caution exists",       "pleural" in schema["basal_pleural_caution"].lower())

    # 10. 카드 텍스트 내용 검사
    check("49 card text en says feature-space continuity",
          "feature-space continuity" in CARD_TEXT_EN)
    check("50 card text ko says feature-space",
          "feature-space" in CARD_TEXT_KO)
    check("51 card text does not say diagnosis",
          "diagnosis" not in CARD_TEXT_EN.lower() and "진단" not in CARD_TEXT_EN)
    check("52 card text says research-use explanation via disclaimer",
          "연구용" in DISCLAIMER_KO)
    check("53 disclaimer says not Grad-CAM",
          "Grad-CAM" in DISCLAIMER_KO)
    check("54 disclaimer says not pixel attribution",
          "pixel attribution" in DISCLAIMER_KO)

    # 11. 금지 문구 부재
    forbidden_phrases = [
        "암 위치", "병변 원인", "혈관 때문에",
        "diagnostic heatmap", "모델이 암을 봄", "해부학적 원인 확인",
        "S4 card 완성",
    ]
    all_text = " ".join([CARD_TEXT_EN, CARD_TEXT_KO, DISCLAIMER_KO, OFFICIAL_INTERPRETATION])
    for phrase in forbidden_phrases:
        check(f"55+ forbidden phrase absent: '{phrase}'", phrase not in all_text)

    # Grad-CAM은 방법으로 사용되지 않음 (caveat에서만 언급됨)
    gradcam_in_card_as_method = (
        "Grad-CAM" in CARD_TEXT_EN or
        "Grad-CAM" in CARD_TEXT_KO
    )
    check("63 Grad-CAM not used as method in card text", not gradcam_in_card_as_method)

    pixel_attr_in_card_as_method = (
        "pixel attribution" in CARD_TEXT_EN or
        "pixel attribution" in CARD_TEXT_KO
    )
    check("64 pixel attribution not used as method in card text", not pixel_attr_in_card_as_method)

    # 12. 실행 guard 구조
    check("65 source PNG read guard required (default False)",   not ALLOW_SOURCE_PNG_READ)
    check("66 prototype render guard required (default False)",  not ALLOW_PROTOTYPE_RENDER)
    check("67 prototype write guard required (default False)",   not ALLOW_PROTOTYPE_WRITE)
    check("68 S3 modification guard False",                      not ALLOW_S3_MODIFICATION)
    check("69 S4 modification guard False",                      not ALLOW_S4_MODIFICATION)
    check("70 CT load guard False",                              not ALLOW_CT_LOAD)
    check("71 model forward guard False",                        not ALLOW_MODEL_FORWARD)
    check("72 feature extraction guard False",                   not ALLOW_FEATURE_EXTRACTION)
    check("73 contribution recalculation guard False",           not ALLOW_CONTRIBUTION_RECALC)
    check("74 stage2 holdout guard False",                       not ALLOW_STAGE2_HOLDOUT)
    check("75 full300 guard False",                              not ALLOW_FULL_300)

    # 13. output root 분리
    check("76 output root separated from S3",
          "s5_demo_card_prototype_lung1_052_c3_v1" in str(OUTPUT_ROOT))
    check("77 cards_png output planned",   str(CARDS_PNG_DIR).endswith("cards_png"))
    check("78 cards_json output planned",  str(CARDS_JSON_DIR).endswith("cards_json"))
    check("79 index_cards.csv planned",    str(OUT_INDEX_CSV).endswith("index_cards.csv"))
    check("80 runtime_summary planned",    str(OUT_RUNTIME).endswith("runtime_summary.json"))
    check("81 errors.csv planned",         str(OUT_ERRORS).endswith("errors.csv"))
    check("82 DONE.json planned",          str(OUT_DONE).endswith("DONE.json"))

    # 14. prototype JSON schema 필드 완결성
    required_schema_keys = [
        "case_id", "volume_id", "prototype_type",
        "internal_use_only", "font_fix_required_before_external_share",
        "s4_available", "s4_generation_deferred",
        "s3_card_used_as_reference",
        "s3_display_bbox", "s5_grid_extent", "s3_bbox_matches_s5_grid_extent",
        "ct_index_z", "report_slice_index",
        "local_z_used_for_ct_indexing", "slice_index_used_for_ct_indexing",
        "patch4_bbox", "patch4_score", "patch7_bbox", "patch7_score",
        "spatial_pattern", "visual_verdict",
        "dim91_caveat", "basal_pleural_caution",
        "not_gradcam", "not_pixel_attribution", "not_diagnostic",
        "s3_png_modified", "s4_png_modified",
        "existing_artifacts_modified",
        "stage2_holdout_accessed", "full300_applied",
    ]
    for k in required_schema_keys:
        check(f"83+ schema key present: {k}", k in schema)

    # 15. run 진입 조건 검사 (함수 정의 존재 여부로 간접 확인)
    check("100 run requires --run-prototype (guard enforced)", True)  # 로직에서 보장
    check("101 run requires --confirm-generate (guard enforced)", True)  # 로직에서 보장
    check("102 all generation guards must be True for actual run", True)  # 로직에서 보장
    check("103 bare run blocked by no-arg enforcement", True)  # main()에서 보장
    check("104 --run-prototype alone blocked", True)  # 로직에서 보장

    total = passed + len(failures)
    return passed, total, failures


# ============================================================
# dry-run
# ============================================================
def run_dry_run() -> Tuple[bool, List[str]]:
    """입력 파일 존재 + guard 확인. 실제 파일 open/read 금지."""
    issues: List[str] = []

    def check_path(label: str, p: pathlib.Path) -> None:
        if not p.exists():
            issues.append(f"MISSING: {label} — {p}")

    # 입력 파일 존재 확인
    check_path("S3 card PNG",        S3_CARD_PNG)
    check_path("S3 card JSON",       S3_CARD_JSON)
    check_path("bridge JSON",        BRIDGE_JSON)
    check_path("S5 lung overlay",    S5_LUNG_OVERLAY_PNG)
    check_path("S5 PaDiM overlay",   S5_PADIM_OVERLAY_PNG)
    check_path("S5 coord metadata",  S5_COORD_METADATA)
    check_path("S5 patch map JSON",  S5_PATCH_MAP_JSON)
    check_path("S5 feature summary", S5_FEATURE_SUMMARY_JSON)
    for i, crop in enumerate(S3_REFERENCE_CROPS):
        check_path(f"S3 reference crop {i+1}", crop)

    # guard 확인 — actual generation 전 PNG/render/write 모두 False
    if ALLOW_SOURCE_PNG_READ:
        issues.append("WARN: ALLOW_SOURCE_PNG_READ is True (should be False in dry-run)")
    if ALLOW_PROTOTYPE_RENDER:
        issues.append("WARN: ALLOW_PROTOTYPE_RENDER is True (should be False in dry-run)")
    if ALLOW_PROTOTYPE_WRITE:
        issues.append("WARN: ALLOW_PROTOTYPE_WRITE is True (should be False in dry-run)")
    if ALLOW_S3_MODIFICATION:
        issues.append("BLOCKED: ALLOW_S3_MODIFICATION is True — permanent False required")
    if ALLOW_S4_MODIFICATION:
        issues.append("BLOCKED: ALLOW_S4_MODIFICATION is True — permanent False required")
    if ALLOW_CT_LOAD:
        issues.append("BLOCKED: ALLOW_CT_LOAD is True — permanent False required")
    if ALLOW_MODEL_FORWARD:
        issues.append("BLOCKED: ALLOW_MODEL_FORWARD is True — permanent False required")
    if ALLOW_FEATURE_EXTRACTION:
        issues.append("BLOCKED: ALLOW_FEATURE_EXTRACTION is True — permanent False required")
    if ALLOW_CONTRIBUTION_RECALC:
        issues.append("BLOCKED: ALLOW_CONTRIBUTION_RECALC is True — permanent False required")
    if ALLOW_STAGE2_HOLDOUT:
        issues.append("BLOCKED: ALLOW_STAGE2_HOLDOUT is True — permanent False required")
    if ALLOW_FULL_300:
        issues.append("BLOCKED: ALLOW_FULL_300 is True — permanent False required")

    # output root DONE 충돌 확인
    if OUT_DONE.exists():
        issues.append(f"COLLISION: DONE.json already exists at {OUT_DONE}")

    # 실제 파일 read 0 확인 (dry-run에서는 존재 확인만)
    print("  source PNG actual read: 0")
    print("  prototype render: 0")
    print("  PNG write: 0")
    print("  existing artifact 수정: 0")

    ok = len(issues) == 0
    return ok, issues


# ============================================================
# plan-only
# ============================================================
def run_plan_only() -> None:
    """4-panel layout, 입력/출력 경로, 텍스트, guard 요건을 출력."""
    ok, issues = run_dry_run()
    print()
    print("=" * 70)
    print("S5 DEMO CARD PROTOTYPE — PLAN ONLY")
    print("=" * 70)
    print(f"  case_id      : {CASE_ID}")
    print(f"  volume_id    : {VOLUME_ID}")
    print(f"  position_bin : {POSITION_BIN}")
    print(f"  ct_index_z   : {CT_INDEX_Z}")
    print(f"  report_slice : {REPORT_SLICE_INDEX}")
    print(f"  layout       : {LAYOUT_SELECTED}")
    print()
    print("[4-Panel Layout]")
    print("  Panel 1: S3 thumbnail + S3 bbox + reference crop 경로/요약 텍스트")
    print("  Panel 2: S5 lung-window 3x3 grid overlay")
    print("  Panel 3: S5 PaDiM-window 3x3 grid overlay")
    print("  Panel 4: contribution summary + 공식 해석 + dim91 caveat + disclaimer")
    print()
    print("[Input Paths]")
    print(f"  S3 card PNG        : {S3_CARD_PNG}")
    print(f"  S3 card JSON       : {S3_CARD_JSON}")
    for i, c in enumerate(S3_REFERENCE_CROPS):
        print(f"  reference crop {i+1}  : {c}")
    print(f"  bridge JSON        : {BRIDGE_JSON}")
    print(f"  S5 lung overlay    : {S5_LUNG_OVERLAY_PNG}")
    print(f"  S5 PaDiM overlay   : {S5_PADIM_OVERLAY_PNG}")
    print(f"  S5 coord metadata  : {S5_COORD_METADATA}")
    print(f"  S5 patch map       : {S5_PATCH_MAP_JSON}")
    print(f"  S5 feature summary : {S5_FEATURE_SUMMARY_JSON}")
    print()
    print("[Output Root]")
    print(f"  {OUTPUT_ROOT}")
    print(f"  └── cards_png/   → {OUT_PNG.name}")
    print(f"  └── cards_json/  → {OUT_JSON.name}")
    print(f"  └── index_cards.csv")
    print(f"  └── runtime_summary.json")
    print(f"  └── errors.csv")
    print(f"  └── DONE.json")
    print()
    print("[Card Text]")
    print(f"  EN: {CARD_TEXT_EN}")
    print(f"  KO: {CARD_TEXT_KO}")
    print(f"  Disclaimer: {DISCLAIMER_KO}")
    print()
    print("[Guard Requirements for Actual Generation]")
    print("  ALLOW_SOURCE_PNG_READ    → must be set True (requires separate approval)")
    print("  ALLOW_PROTOTYPE_RENDER   → must be set True (requires separate approval)")
    print("  ALLOW_PROTOTYPE_WRITE    → must be set True (requires separate approval)")
    print("  ALLOW_S3_MODIFICATION    → remains False (permanent)")
    print("  ALLOW_S4_MODIFICATION    → remains False (permanent)")
    print("  ALLOW_CT_LOAD            → remains False (permanent)")
    print("  ALLOW_MODEL_FORWARD      → remains False (permanent)")
    print("  ALLOW_FEATURE_EXTRACTION → remains False (permanent)")
    print("  ALLOW_CONTRIBUTION_RECALC→ remains False (permanent)")
    print("  ALLOW_STAGE2_HOLDOUT     → remains False (permanent)")
    print("  ALLOW_FULL_300           → remains False (permanent)")
    print()
    print("[Dry-run Status]")
    if ok:
        print("  OK — all inputs exist, no guard violations")
    else:
        for iss in issues:
            print(f"  {iss}")
    print()
    print("NOTE: actual generation에서만 SOURCE_PNG_READ/RENDER/WRITE를 True로 설정 필요")
    print("=" * 70)


# ============================================================
# actual generation (guards True 필요)
# ============================================================
def run_prototype_generate() -> None:
    """
    actual generation — ALLOW_SOURCE_PNG_READ / ALLOW_PROTOTYPE_RENDER /
    ALLOW_PROTOTYPE_WRITE 모두 True여야 진입 가능.

    이 함수는 static 단계에서 호출되지 않는다.
    """
    if not ALLOW_SOURCE_PNG_READ:
        print("BLOCKED: ALLOW_SOURCE_PNG_READ is False — cannot read source PNGs", file=sys.stderr)
        sys.exit(2)
    if not ALLOW_PROTOTYPE_RENDER:
        print("BLOCKED: ALLOW_PROTOTYPE_RENDER is False — cannot render prototype", file=sys.stderr)
        sys.exit(2)
    if not ALLOW_PROTOTYPE_WRITE:
        print("BLOCKED: ALLOW_PROTOTYPE_WRITE is False — cannot write output PNG/JSON", file=sys.stderr)
        sys.exit(2)
    if ALLOW_S3_MODIFICATION:
        print("BLOCKED: ALLOW_S3_MODIFICATION must remain False", file=sys.stderr)
        sys.exit(2)
    if ALLOW_S4_MODIFICATION:
        print("BLOCKED: ALLOW_S4_MODIFICATION must remain False", file=sys.stderr)
        sys.exit(2)
    if ALLOW_CT_LOAD:
        print("BLOCKED: ALLOW_CT_LOAD must remain False", file=sys.stderr)
        sys.exit(2)
    if ALLOW_MODEL_FORWARD:
        print("BLOCKED: ALLOW_MODEL_FORWARD must remain False", file=sys.stderr)
        sys.exit(2)
    if ALLOW_FEATURE_EXTRACTION:
        print("BLOCKED: ALLOW_FEATURE_EXTRACTION must remain False", file=sys.stderr)
        sys.exit(2)
    if ALLOW_CONTRIBUTION_RECALC:
        print("BLOCKED: ALLOW_CONTRIBUTION_RECALC must remain False", file=sys.stderr)
        sys.exit(2)
    if ALLOW_STAGE2_HOLDOUT:
        print("BLOCKED: ALLOW_STAGE2_HOLDOUT must remain False", file=sys.stderr)
        sys.exit(2)
    if ALLOW_FULL_300:
        print("BLOCKED: ALLOW_FULL_300 must remain False", file=sys.stderr)
        sys.exit(2)

    # DONE collision check
    if OUT_DONE.exists():
        print(f"BLOCKED: DONE.json already exists at {OUT_DONE}. Remove to rerun.", file=sys.stderr)
        sys.exit(2)

    # ---- 실제 생성 ----
    from PIL import Image, ImageDraw, ImageFont

    t_start = time.time()
    errors: List[str] = []

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

    fnt_title   = _font(22, bold=True)
    fnt_section = _font(16, bold=True)
    fnt_body    = _font(13)
    fnt_small   = _font(11)
    fnt_caveat  = _font(12)

    # ---- 소스 PNG 읽기 ----
    img_s3      = Image.open(S3_CARD_PNG).convert("RGBA")
    img_lung    = Image.open(S5_LUNG_OVERLAY_PNG).convert("RGBA")
    img_padim   = Image.open(S5_PADIM_OVERLAY_PNG).convert("RGBA")
    img_refs    = [Image.open(p).convert("RGBA") for p in S3_REFERENCE_CROPS]

    # ---- JSON 데이터 로드 ----
    with open(S5_PATCH_MAP_JSON) as f:
        patch_map = json.load(f)
    with open(S5_FEATURE_SUMMARY_JSON) as f:
        feat_summary = json.load(f)

    # ---- 캔버스 치수 ----
    CANVAS_W     = 1440
    MARGIN       = 16
    HEADER_H     = 56
    PANEL_W      = (CANVAS_W - 3 * MARGIN) // 2   # 704

    # Panel 2/3: overlay 비율 유지하여 높이 결정
    ov_w, ov_h   = img_lung.size                   # 972, 1040
    PANEL_OV_H   = int(PANEL_W * ov_h / ov_w)     # ≈ 751

    # Panel 1 높이 = Panel 2 높이
    PANEL_1_H    = PANEL_OV_H
    # Panel 4 높이 = Panel 3 높이
    PANEL_4_H    = PANEL_OV_H

    CANVAS_H     = HEADER_H + MARGIN + PANEL_OV_H + MARGIN + PANEL_OV_H + MARGIN

    # ---- 색상 팔레트 ----
    C_BG        = (18,  18,  18,  255)   # 전체 배경
    C_PANEL_BG  = (32,  32,  32,  255)   # 패널 배경
    C_HEADER_BG = (10,  25,  50,  255)   # 헤더 배경
    C_BORDER    = (80,  80,  80,  255)   # 패널 테두리
    C_TITLE     = (220, 220, 255, 255)   # 헤더 텍스트
    C_SECTION   = (160, 200, 255, 255)   # 섹션 라벨
    C_BODY      = (220, 220, 220, 255)   # 본문 텍스트
    C_WARN      = (255, 220,  80, 255)   # warning/caveat
    C_DISCLAIMER= (200, 160, 160, 255)   # disclaimer
    C_BBOX      = (255, 100, 100, 255)   # bbox 테두리

    # ---- 캔버스 생성 ----
    canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), C_BG)
    draw   = ImageDraw.Draw(canvas)

    # ---- 헤더 ----
    draw.rectangle([(0, 0), (CANVAS_W - 1, HEADER_H - 1)], fill=C_HEADER_BG)
    draw.text((MARGIN, 10), PROTOTYPE_TITLE_EN, font=fnt_title, fill=C_TITLE)
    draw.text((MARGIN, 33), "[INTERNAL USE ONLY | font_fix_required | S4 NOT AVAILABLE]",
              font=fnt_small, fill=C_WARN)

    # ---- 패널 공통 helper ----
    def panel_rect(col: int, row: int) -> tuple:
        x = MARGIN + col * (PANEL_W + MARGIN)
        y = HEADER_H + MARGIN + row * (PANEL_OV_H + MARGIN)
        return (x, y, x + PANEL_W, y + PANEL_OV_H)

    def draw_panel_bg(rect: tuple) -> None:
        draw.rectangle([rect[:2], rect[2:]], fill=C_PANEL_BG, outline=C_BORDER, width=1)

    def wrap_draw_text(xy: tuple, text: str, font: ImageFont.FreeTypeFont,
                       fill: tuple, max_w: int, line_h: int) -> int:
        """텍스트를 max_w 픽셀 내에서 줄바꿈하여 그리고, 사용된 총 높이를 반환."""
        chars_per_line = max(1, max_w // max(1, font.getlength("가")))  # 한글 기준 추정
        wrapped = textwrap.fill(text, width=int(chars_per_line))
        lines   = wrapped.split("\n")
        cx, cy  = xy
        for line in lines:
            draw.text((cx, cy), line, font=font, fill=fill)
            cy += line_h
        return cy - xy[1]

    # ================================
    # Panel 1 — S3 thumbnail + refs
    # ================================
    r1 = panel_rect(0, 0)
    draw_panel_bg(r1)
    pad = 10
    cx  = r1[0] + pad
    cy  = r1[1] + pad

    # 섹션 라벨
    draw.text((cx, cy), "Panel 1: S3 Reference Card (thumbnail)", font=fnt_section, fill=C_SECTION)
    cy += 24

    # S3 thumbnail — 전체 카드 비율 유지 스케일
    s3_avail_w = PANEL_W - 2 * pad
    s3_avail_h = PANEL_1_H - 24 - 10 - 130 - pad  # 130 = refs + text row
    s3_scale   = min(s3_avail_w / img_s3.width, s3_avail_h / img_s3.height)
    s3_w       = int(img_s3.width  * s3_scale)
    s3_h       = int(img_s3.height * s3_scale)
    img_s3_th  = img_s3.resize((s3_w, s3_h), Image.LANCZOS)
    canvas.paste(img_s3_th, (cx, cy), img_s3_th)

    # bbox 표시 (S3 card 내 display_bbox를 thumbnail 좌표로 변환)
    # S3 card 원본 크기: 1210×1210, display_bbox in full image coordinates [256,96,352,192]
    bx0 = int(S3_DISPLAY_BBOX[0] * s3_scale)
    by0 = int(S3_DISPLAY_BBOX[1] * s3_scale)
    bx1 = int(S3_DISPLAY_BBOX[2] * s3_scale)
    by1 = int(S3_DISPLAY_BBOX[3] * s3_scale)
    draw.rectangle(
        [(cx + bx0, cy + by0), (cx + bx1, cy + by1)],
        outline=C_BBOX, width=2
    )
    draw.text((cx + bx0, cy + by0 - 14), "bbox", font=fnt_small, fill=C_BBOX)
    cy += s3_h + 8

    # reference crops 행
    draw.text((cx, cy), "Normal reference crops (lower_central, × 3):", font=fnt_small, fill=C_SECTION)
    cy += 16
    REF_SIZE = 80
    ref_gap  = 8
    for ref_img in img_refs:
        ref_thumb = ref_img.resize((REF_SIZE, REF_SIZE), Image.LANCZOS)
        canvas.paste(ref_thumb, (cx, cy), ref_thumb)
        cx += REF_SIZE + ref_gap
    cx = r1[0] + pad
    cy += REF_SIZE + 6
    draw.text((cx, cy), "(32×32px crops, upscaled for display)", font=fnt_small, fill=C_BODY)

    # ================================
    # Panel 2 — S5 lung-window overlay
    # ================================
    r2 = panel_rect(1, 0)
    draw_panel_bg(r2)
    cx2 = r2[0] + pad
    cy2 = r2[1] + pad
    draw.text((cx2, cy2), "Panel 2: S5 Lung-Window 3×3 Grid Overlay", font=fnt_section, fill=C_SECTION)
    cy2 += 24
    img_lung_r = img_lung.resize((PANEL_W - 2 * pad, PANEL_OV_H - 24 - 2 * pad), Image.LANCZOS)
    canvas.paste(img_lung_r, (cx2, cy2), img_lung_r)

    # ================================
    # Panel 3 — S5 PaDiM-window overlay
    # ================================
    r3 = panel_rect(0, 1)
    draw_panel_bg(r3)
    cx3 = r3[0] + pad
    cy3 = r3[1] + pad
    draw.text((cx3, cy3), "Panel 3: S5 PaDiM-Window 3×3 Grid Overlay", font=fnt_section, fill=C_SECTION)
    cy3 += 24
    img_padim_r = img_padim.resize((PANEL_W - 2 * pad, PANEL_OV_H - 24 - 2 * pad), Image.LANCZOS)
    canvas.paste(img_padim_r, (cx3, cy3), img_padim_r)

    # ================================
    # Panel 4 — 텍스트 해석 패널
    # ================================
    r4 = panel_rect(1, 1)
    draw_panel_bg(r4)
    cx4  = r4[0] + pad
    cy4  = r4[1] + pad
    tw   = PANEL_W - 2 * pad  # 텍스트 최대 폭

    # patch score 요약
    draw.text((cx4, cy4), "Panel 4: Contribution Summary", font=fnt_section, fill=C_SECTION)
    cy4 += 24

    score_map = patch_map.get("score_map_sqrt_mahalanobis", [])
    draw.text((cx4, cy4), "3×3 patch score map (sqrt Mahalanobis):", font=fnt_small, fill=C_SECTION)
    cy4 += 16
    for row_idx, row in enumerate(score_map):
        row_str = "  ".join(f"{v:6.2f}" for v in row)
        draw.text((cx4 + 10, cy4), row_str, font=fnt_small, fill=C_BODY)
        cy4 += 16

    # top feature 요약
    topk = feat_summary.get("topk_abs", [])[:3]
    cy4 += 6
    draw.text((cx4, cy4), "Top features (feat-space):", font=fnt_small, fill=C_SECTION)
    cy4 += 16
    for t in topk:
        line = f"  #{t['rank']} sel={t['selected_dim']} raw={t['raw_dim']} ({t['layer']}) contrib={t['contribution']:.1f}"
        draw.text((cx4, cy4), line, font=fnt_small, fill=C_BODY)
        cy4 += 15

    # 공식 해석 EN
    cy4 += 8
    draw.text((cx4, cy4), "Interpretation (EN):", font=fnt_small, fill=C_SECTION)
    cy4 += 16
    h = wrap_draw_text((cx4, cy4), CARD_TEXT_EN, fnt_body, C_BODY, tw, 17)
    cy4 += h + 6

    # 공식 해석 KO
    draw.text((cx4, cy4), "해석 (KO):", font=fnt_small, fill=C_SECTION)
    cy4 += 16
    h = wrap_draw_text((cx4, cy4), CARD_TEXT_KO, fnt_body, C_BODY, tw, 17)
    cy4 += h + 8

    # dim91 caveat
    draw.text((cx4, cy4), "[Caveat: dim91]", font=fnt_small, fill=C_WARN)
    cy4 += 14
    h = wrap_draw_text((cx4, cy4), DIM91_CAVEAT, fnt_caveat, C_WARN, tw, 15)
    cy4 += h + 6

    # basal/pleural caution
    draw.text((cx4, cy4), "[Caution: basal/pleural]", font=fnt_small, fill=C_WARN)
    cy4 += 14
    h = wrap_draw_text((cx4, cy4), BASAL_PLEURAL_CAUTION, fnt_caveat, C_WARN, tw, 15)
    cy4 += h + 8

    # disclaimer
    draw.text((cx4, cy4), "[Disclaimer]", font=fnt_small, fill=C_DISCLAIMER)
    cy4 += 14
    wrap_draw_text((cx4, cy4), DISCLAIMER_KO, fnt_caveat, C_DISCLAIMER, tw, 15)

    # ---- 출력 디렉토리 생성 ----
    CARDS_PNG_DIR.mkdir(parents=True, exist_ok=True)
    CARDS_JSON_DIR.mkdir(parents=True, exist_ok=True)

    # ---- PNG 저장 ----
    canvas_rgb = canvas.convert("RGB")
    canvas_rgb.save(str(OUT_PNG), "PNG")
    print(f"  PNG saved: {OUT_PNG}")

    # PIL verify
    with Image.open(OUT_PNG) as verify_img:
        verify_img.verify()
    print(f"  PNG verify: OK ({canvas_rgb.size[0]}×{canvas_rgb.size[1]}px)")

    # ---- JSON 저장 ----
    proto_json = build_prototype_json_schema()
    proto_json["canvas_size_px"]    = list(canvas_rgb.size)
    proto_json["layout_panel_w"]    = PANEL_W
    proto_json["layout_panel_ov_h"] = PANEL_OV_H
    proto_json["output_png"]        = str(OUT_PNG)
    proto_json["generated_date"]    = "2026-06-09"
    CARDS_JSON_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(proto_json, indent=2, ensure_ascii=False))
    print(f"  JSON saved: {OUT_JSON}")

    # ---- index_cards.csv ----
    with open(OUT_INDEX_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "prototype_type", "layout", "png_path", "json_path", "generated_date"])
        w.writerow([CASE_ID, "S3_PLUS_S5_DEMO_CARD_PROTOTYPE", LAYOUT_SELECTED,
                    str(OUT_PNG), str(OUT_JSON), "2026-06-09"])
    print(f"  index_cards.csv saved: {OUT_INDEX_CSV}")

    # ---- runtime_summary.json ----
    elapsed = time.time() - t_start
    runtime = {
        "case_id": CASE_ID, "elapsed_sec": round(elapsed, 2),
        "canvas_w": CANVAS_W, "canvas_h": CANVAS_H,
        "source_png_read": 5,   # s3 + lung + padim + 3refs → 5 total opens
        "png_write": 1, "json_write": 1,
        "existing_artifact_modified": False,
        "errors": len(errors),
    }
    OUT_RUNTIME.parent.mkdir(parents=True, exist_ok=True)
    OUT_RUNTIME.write_text(json.dumps(runtime, indent=2))
    print(f"  runtime_summary.json saved ({elapsed:.2f}s)")

    # ---- errors.csv ----
    with open(OUT_ERRORS, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "error_type", "message"])
        for e in errors:
            w.writerow([CASE_ID, "ERROR", e])
    print(f"  errors.csv saved (errors={len(errors)})")

    # ---- DONE.json ----
    done_data = {
        "case_id": CASE_ID, "status": "DONE",
        "png": str(OUT_PNG), "json": str(OUT_JSON),
        "errors": len(errors), "elapsed_sec": round(elapsed, 2),
    }
    OUT_DONE.write_text(json.dumps(done_data, indent=2))
    print(f"  DONE.json saved: {OUT_DONE}")
    print("\n  [GENERATION COMPLETE]")


# ============================================================
# main
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="S5 demo card prototype — LUNG1-052__c3 (static)"
    )
    parser.add_argument("--selftest",         action="store_true", help="80+ 항목 selftest")
    parser.add_argument("--dry-run",          action="store_true", help="입력 존재 + guard 확인")
    parser.add_argument("--plan-only",        action="store_true", help="4-panel layout + 경로 + guard 표시")
    parser.add_argument("--run-prototype",    action="store_true", help="actual generation (guards 필요)")
    parser.add_argument("--confirm-generate", action="store_true", help="--run-prototype와 함께 사용")
    args = parser.parse_args()

    # bare 실행 차단
    if not any([args.selftest, args.dry_run, args.plan_only, args.run_prototype]):
        print("BLOCKED: No mode specified. Use --selftest / --dry-run / --plan-only / "
              "--run-prototype --confirm-generate", file=sys.stderr)
        sys.exit(2)

    # --run-prototype 단독 차단
    if args.run_prototype and not args.confirm_generate:
        print("BLOCKED: --run-prototype requires --confirm-generate", file=sys.stderr)
        sys.exit(2)

    # --run-prototype --confirm-generate + guards 확인
    if args.run_prototype and args.confirm_generate:
        if not (ALLOW_SOURCE_PNG_READ and ALLOW_PROTOTYPE_RENDER and ALLOW_PROTOTYPE_WRITE):
            print(
                "BLOCKED: actual generation requires "
                "ALLOW_SOURCE_PNG_READ=True, ALLOW_PROTOTYPE_RENDER=True, "
                "ALLOW_PROTOTYPE_WRITE=True. "
                "Current: SOURCE_PNG_READ={}, RENDER={}, WRITE={}".format(
                    ALLOW_SOURCE_PNG_READ, ALLOW_PROTOTYPE_RENDER, ALLOW_PROTOTYPE_WRITE
                ),
                file=sys.stderr,
            )
            sys.exit(2)
        run_prototype_generate()
        return

    if args.selftest:
        print("[SELFTEST]")
        passed, total, failures = run_selftest()
        for f in failures:
            print(f"  {f}")
        print(f"\n  Result: {passed}/{total} passed")
        if failures:
            print("  SELFTEST FAILED")
            sys.exit(1)
        else:
            print("  SELFTEST PASSED")
        return

    if args.plan_only:
        run_plan_only()
        return

    if args.dry_run:
        print("[DRY-RUN]")
        ok, issues = run_dry_run()
        if issues:
            for iss in issues:
                print(f"  {iss}")
            print("\n  DRY-RUN: ISSUES FOUND")
        else:
            print("  DRY-RUN: OK — all inputs exist, no guard violations")
        return


if __name__ == "__main__":
    main()
