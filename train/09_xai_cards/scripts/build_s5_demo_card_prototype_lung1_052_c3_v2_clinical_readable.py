"""
S5 Demo Card Prototype v2 Clinical-Readable — LUNG1-052__c3
===========================================================
v1 문제점 수정:
  - Panel 1: S3 thumbnail 크게, reference crops 제거
  - Panel 2: lung-window overlay + 범례 (메인 CT overlay)
  - Panel 3: 3×3 patch response map schematic (PaDiM-window overlay 대체)
  - Panel 4: 의사용 쉬운 문장 (dim91/topk feature 제거)

출력 root:
  outputs/position-aware-padim-v1/reports/explanation_cards/
    s5_demo_card_prototype_lung1_052_c3_v2_clinical_readable/

실행 방법:
  --dry-run          → 입력 파일 존재 + guard 확인
  --plan-only        → 4-panel layout + 입력/출력 경로 + guard 표시
  --static-check     → 정적 검사 (주석/schema/텍스트 정책 위반 확인)
  --run-prototype --confirm-generate  → guards False이면 BLOCKED exit 2

actual generation에서만 아래 guard를 별도 승인 후 True:
  ALLOW_SOURCE_PNG_READ = True
  ALLOW_PROTOTYPE_RENDER = True
  ALLOW_PROTOTYPE_WRITE = True
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

LAYOUT_SELECTED      = "OPTION_D_4PANEL_CLINICAL_READABLE"
S4_AVAILABLE         = False
S4_GENERATION_DEFERRED = True

# ============================================================
# 경로 상수
# ============================================================
PROJECT_ROOT = pathlib.Path("/home/jinhy/project/lung-ct-anomaly")

# 입력 경로
S3_CARD_PNG  = PROJECT_ROOT / "outputs/position-aware-padim-v1/visualizations/candidate_cards/s3_expansion_cards_v1/cards_png/LUNG1-052__c3.png"
S3_CARD_JSON = PROJECT_ROOT / "outputs/position-aware-padim-v1/visualizations/candidate_cards/s3_expansion_cards_v1/cards_json/LUNG1-052__c3.json"

BRIDGE_JSON = PROJECT_ROOT / "outputs/position-aware-padim-v1/reports/explanation_cards/s3_s5_bridge_lung1_052_c3_v1/s3_s5_bridge_lung1_052_c3_v1.json"

S5_LUNG_OVERLAY_PNG  = PROJECT_ROOT / "outputs/position-aware-padim-v1/reports/explanation_cards/s5_lung1_052_c3_coordinate_visual_audit_1case_v1/coordinate_overlay_3x3_grid_lung.png"
S5_COORD_METADATA    = PROJECT_ROOT / "outputs/position-aware-padim-v1/reports/explanation_cards/s5_lung1_052_c3_coordinate_visual_audit_1case_v1/coordinate_overlay_metadata.json"
S5_PATCH_MAP_JSON    = PROJECT_ROOT / "outputs/position-aware-padim-v1/reports/explanation_cards/s5_lung1_052_c3_patch_level_contribution_map_1case_smoke_v1/patch_contribution_map.json"
S5_FEATURE_SUMMARY_JSON = PROJECT_ROOT / "outputs/position-aware-padim-v1/reports/explanation_cards/s5_lung1_052_c3_feature_contribution_1case_smoke_v1/feature_contribution_summary.json"

# 출력 경로 (v2)
OUTPUT_ROOT    = PROJECT_ROOT / "outputs/position-aware-padim-v1/reports/explanation_cards/s5_demo_card_prototype_lung1_052_c3_v2_clinical_readable"
CARDS_PNG_DIR  = OUTPUT_ROOT / "cards_png"
CARDS_JSON_DIR = OUTPUT_ROOT / "cards_json"
OUT_PNG        = CARDS_PNG_DIR  / "LUNG1-052__c3_s5_demo_prototype_v2.png"
OUT_JSON       = CARDS_JSON_DIR / "LUNG1-052__c3_s5_demo_prototype_v2.json"
OUT_INDEX_CSV  = OUTPUT_ROOT / "index_cards_v2.csv"
OUT_RUNTIME    = OUTPUT_ROOT / "runtime_summary_v2.json"
OUT_ERRORS     = OUTPUT_ROOT / "errors.csv"
OUT_DONE       = OUTPUT_ROOT / "DONE.json"

# ============================================================
# 텍스트 상수 — v2 의사용 문장
# ============================================================
PROTOTYPE_TITLE_EN = "S5 PaDiM patch-level explanation prototype v2 — LUNG1-052__c3"

# Panel 4 섹션별 문장
P4_KEY_FINDING_KO = (
    "노란 박스는 가장 높은 PaDiM 반응을 보인 패치이고 (patch 4, 중심), "
    "주황 박스는 바로 아래에서 두 번째로 높은 반응을 보인 패치입니다 (patch 7)."
)
P4_KEY_FINDING_EN = (
    "The yellow box marks the highest PaDiM response (patch 4, center). "
    "The orange box marks the second-highest response directly below (patch 7)."
)

P4_INTERPRETATION_KO = (
    "두 패치가 위아래로 이어져 있어, 모델 반응이 한 지점에만 튄 것이 아니라 "
    "주변 영역으로 연속되는 패턴으로 보입니다."
)
P4_INTERPRETATION_EN = (
    "The response continues from the center patch toward the lower adjacent patch, "
    "suggesting a localized patch-level response pattern."
)

P4_CAUTION_KO = (
    "이 결과는 patch 단위의 feature-space 반응을 요약한 것이며, "
    "특정 해부학적 원인이나 병변 원인을 직접 의미하지 않습니다."
)
P4_CAUTION_EN = (
    "This is a patch-level feature-space summary, "
    "not a pixel-level attribution or diagnostic result."
)

P4_DISCLAIMER_KO = "연구용 설명 자료이며 진단 결과가 아닙니다."
P4_DISCLAIMER_EN = "Research-use explanation only. Not diagnostic."

# metadata/report 전용 (카드 본문 사용 금지)
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
# prototype JSON schema 정의
# ============================================================
def build_prototype_json_schema() -> Dict[str, Any]:
    return {
        "version": "v2_clinical_readable",
        "case_id": CASE_ID,
        "volume_id": VOLUME_ID,
        "position_bin": POSITION_BIN,
        "ct_index_z": CT_INDEX_Z,
        "report_slice_index": REPORT_SLICE_INDEX,
        "spatial_pattern": SPATIAL_PATTERN,
        "visual_verdict": VISUAL_VERDICT,
        "s3_display_bbox": S3_DISPLAY_BBOX,
        "patch4_bbox": PATCH4_BBOX,
        "patch4_score": PATCH4_SCORE,
        "patch7_bbox": PATCH7_BBOX,
        "patch7_score": PATCH7_SCORE,
        "layout_selected": LAYOUT_SELECTED,
        "s4_available": S4_AVAILABLE,
        "panels": {
            "panel_1": "S3 thumbnail + bbox (reference crops 제거, thumbnail 크게)",
            "panel_2": "S5 lung-window 3x3 grid overlay + 범례",
            "panel_3": "3x3 patch response map schematic (not CT overlay, not pixel heatmap, not Grad-CAM)",
            "panel_4": "Key finding / Interpretation / Caution / Disclaimer (의사용 4섹션)"
        },
        "metadata": {
            "not_gradcam": True,
            "not_pixel_attribution": True,
            "not_diagnostic": True,
            "internal_use_only": True,
            "font_fix_required_before_external_share": True,
            "dim91_caveat": _META_DIM91_CAVEAT,
            "basal_pleural_caution": _META_BASAL_PLEURAL_CAUTION,
            "s3_card_used_as_reference": True,
            "s4_unavailable": True,
            "patch_response_map_source": "score_map_sqrt_mahalanobis from patch_contribution_map.json"
        },
        "source_files": {
            "s3_card_png": str(S3_CARD_PNG),
            "s3_card_json": str(S3_CARD_JSON),
            "s5_lung_overlay_png": str(S5_LUNG_OVERLAY_PNG),
            "s5_patch_map_json": str(S5_PATCH_MAP_JSON),
            "s5_feature_summary_json": str(S5_FEATURE_SUMMARY_JSON),
            "bridge_json": str(BRIDGE_JSON)
        },
        "canvas_size_px": None,
        "layout_panel_w": None,
        "layout_panel_ov_h": None,
        "output_png": str(OUT_PNG),
        "generated_date": None
    }


# ============================================================
# 정적 검사
# ============================================================
def static_check() -> Tuple[int, int, List[str]]:
    passed = 0
    failed = 0
    issues: List[str] = []

    def check(label: str, condition: bool) -> None:
        nonlocal passed, failed
        if condition:
            passed += 1
        else:
            failed += 1
            issues.append(f"FAIL: {label}")

    schema = build_prototype_json_schema()

    # 1. guard 기본값 확인
    check("01 ALLOW_SOURCE_PNG_READ default False",    not ALLOW_SOURCE_PNG_READ)
    check("02 ALLOW_PROTOTYPE_RENDER default False",   not ALLOW_PROTOTYPE_RENDER)
    check("03 ALLOW_PROTOTYPE_WRITE default False",    not ALLOW_PROTOTYPE_WRITE)
    check("04 ALLOW_S3_MODIFICATION False",            not ALLOW_S3_MODIFICATION)
    check("05 ALLOW_S4_MODIFICATION False",            not ALLOW_S4_MODIFICATION)
    check("06 ALLOW_CT_LOAD False",                    not ALLOW_CT_LOAD)
    check("07 ALLOW_MODEL_FORWARD False",              not ALLOW_MODEL_FORWARD)
    check("08 ALLOW_FEATURE_EXTRACTION False",         not ALLOW_FEATURE_EXTRACTION)
    check("09 ALLOW_CONTRIBUTION_RECALC False",        not ALLOW_CONTRIBUTION_RECALC)
    check("10 ALLOW_STAGE2_HOLDOUT False",             not ALLOW_STAGE2_HOLDOUT)
    check("11 ALLOW_FULL_300 False",                   not ALLOW_FULL_300)

    # 2. 케이스 상수
    check("12 CASE_ID exact",       CASE_ID == "LUNG1-052__c3")
    check("13 VOLUME_ID exact",     VOLUME_ID == "NSCLC_LUNG1-052__d4a19cc211")
    check("14 PATCH4_SCORE range",  35.0 < PATCH4_SCORE < 45.0)
    check("15 PATCH7_SCORE range",  33.0 < PATCH7_SCORE < 42.0)
    check("16 PATCH4 > PATCH7",     PATCH4_SCORE > PATCH7_SCORE)

    # 3. 경로 상수 — v2 전용 확인
    check("17 OUTPUT_ROOT contains v2",          "v2" in str(OUTPUT_ROOT))
    check("18 OUTPUT_ROOT name not v1",           "v1" not in OUTPUT_ROOT.name)
    check("19 OUT_PNG contains v2",              "v2" in OUT_PNG.name)
    check("20 OUT_JSON contains v2",             "v2" in OUT_JSON.name)
    check("21 S3 card PNG path set",             S3_CARD_PNG.parts[-1] == "LUNG1-052__c3.png")
    check("22 S5 lung overlay path set",         "coordinate_overlay_3x3_grid_lung.png" in str(S5_LUNG_OVERLAY_PNG))
    check("23 patch map JSON path set",          "patch_contribution_map.json" in str(S5_PATCH_MAP_JSON))

    # 4. v1 output root와 충돌 없음
    v1_root = str(PROJECT_ROOT / "outputs/position-aware-padim-v1/reports/explanation_cards/s5_demo_card_prototype_lung1_052_c3_v1")
    check("24 OUTPUT_ROOT != v1 root",           str(OUTPUT_ROOT) != v1_root)

    # 5. 텍스트 정책 확인 (금지 표현)
    all_card_text = " ".join([
        P4_KEY_FINDING_KO, P4_KEY_FINDING_EN,
        P4_INTERPRETATION_KO, P4_INTERPRETATION_EN,
        P4_CAUTION_KO, P4_CAUTION_EN,
        P4_DISCLAIMER_KO, P4_DISCLAIMER_EN,
        PROTOTYPE_TITLE_EN
    ]).lower()
    check("30 no Grad-CAM in card text",          "grad-cam" not in all_card_text)
    check("31 no gradcam in card text",           "gradcam" not in all_card_text)
    check("32 no pixel attribution in card text", "pixel attribution" not in all_card_text)
    check("33 no diagnostic heatmap in card text","diagnostic heatmap" not in all_card_text)
    check("34 no cancer location in card text",   "cancer location" not in all_card_text)
    check("35 no lesion cause in card text",      "lesion cause" not in all_card_text)
    check("36 no model saw cancer in card text",  "model saw cancer" not in all_card_text)
    check("37 no covariance inverse in card text","covariance inverse" not in all_card_text)

    # 6. metadata에 필수 항목 유지
    meta = schema["metadata"]
    check("40 not_gradcam in metadata",              meta.get("not_gradcam") is True)
    check("41 not_pixel_attribution in metadata",    meta.get("not_pixel_attribution") is True)
    check("42 not_diagnostic in metadata",           meta.get("not_diagnostic") is True)
    check("43 internal_use_only in metadata",        meta.get("internal_use_only") is True)
    check("44 dim91_caveat in metadata",             len(meta.get("dim91_caveat", "")) > 20)
    check("45 basal_pleural_caution in metadata",    len(meta.get("basal_pleural_caution", "")) > 10)

    # 7. layout 결정
    check("50 LAYOUT_SELECTED v2",                  "CLINICAL_READABLE" in LAYOUT_SELECTED)
    check("51 panel_1 thumbnail (no crops)",        "thumbnail" in schema["panels"]["panel_1"].lower())
    check("52 panel_2 lung overlay",                "lung" in schema["panels"]["panel_2"].lower())
    check("53 panel_3 schematic (not overlay)",     "schematic" in schema["panels"]["panel_3"].lower())
    check("54 panel_3 not pixel heatmap",           "not pixel heatmap" in schema["panels"]["panel_3"].lower())
    check("55 panel_3 not Grad-CAM",                "not grad-cam" in schema["panels"]["panel_3"].lower())
    check("56 panel_4 key finding planned",         "key finding" in schema["panels"]["panel_4"].lower())
    check("57 panel_4 caution planned",             "caution" in schema["panels"]["panel_4"].lower())
    check("58 panel_4 disclaimer planned",          "disclaimer" in schema["panels"]["panel_4"].lower())

    # 8. S3/S4/S5 원본 수정 guard
    check("60 S3 modification guard False",         not ALLOW_S3_MODIFICATION)
    check("61 S4 modification guard False",         not ALLOW_S4_MODIFICATION)
    check("62 CT load guard False",                 not ALLOW_CT_LOAD)
    check("63 model forward guard False",           not ALLOW_MODEL_FORWARD)
    check("64 feature extraction guard False",      not ALLOW_FEATURE_EXTRACTION)
    check("65 contribution recalc guard False",     not ALLOW_CONTRIBUTION_RECALC)
    check("66 stage2 holdout guard False",          not ALLOW_STAGE2_HOLDOUT)
    check("67 full300 guard False",                 not ALLOW_FULL_300)

    # 9. source PNG 목록 — v2에서 PaDiM overlay / reference crops 읽지 않음
    check("70 S5_PADIM_OVERLAY not required for v2 panel3", True)  # schematic이므로 CT overlay 불필요
    check("71 S3_REFERENCE_CROPS not required for v2 panel1", True)  # panel1에서 제거됨

    # 10. 실행 guard 구조
    check("80 source PNG read guard (default False)",   not ALLOW_SOURCE_PNG_READ)
    check("81 prototype render guard (default False)",  not ALLOW_PROTOTYPE_RENDER)
    check("82 prototype write guard (default False)",   not ALLOW_PROTOTYPE_WRITE)

    return passed, failed, issues


# ============================================================
# dry-run: 입력 파일 존재 + guard 확인
# ============================================================
def dry_run() -> None:
    print(f"[dry-run] S5 Demo Card Prototype v2 — {CASE_ID}")
    issues: List[str] = []

    def check_path(label: str, p: pathlib.Path) -> None:
        if p.exists():
            print(f"  OK  {label}: {p.name}")
        else:
            msg = f"MISSING {label}: {p}"
            print(f"  ERR {msg}")
            issues.append(msg)

    check_path("S3 card PNG",        S3_CARD_PNG)
    check_path("S3 card JSON",       S3_CARD_JSON)
    check_path("BRIDGE JSON",        BRIDGE_JSON)
    check_path("S5 lung overlay",    S5_LUNG_OVERLAY_PNG)
    check_path("S5 coord metadata",  S5_COORD_METADATA)
    check_path("S5 patch map JSON",  S5_PATCH_MAP_JSON)
    check_path("S5 feature summary", S5_FEATURE_SUMMARY_JSON)

    # v2에서 사용 안 하는 파일 (참고용 존재 확인만)
    _padim_ov = PROJECT_ROOT / "outputs/position-aware-padim-v1/reports/explanation_cards/s5_lung1_052_c3_coordinate_visual_audit_1case_v1/coordinate_overlay_3x3_grid_padim_window.png"
    print(f"  REF S5 padim overlay (v2 사용 안함): {'exists' if _padim_ov.exists() else 'missing'}")

    # guard 확인
    if ALLOW_SOURCE_PNG_READ:
        issues.append("WARN: ALLOW_SOURCE_PNG_READ is True (should be False in dry-run)")
    if ALLOW_PROTOTYPE_RENDER:
        issues.append("WARN: ALLOW_PROTOTYPE_RENDER is True (should be False in dry-run)")
    if ALLOW_PROTOTYPE_WRITE:
        issues.append("WARN: ALLOW_PROTOTYPE_WRITE is True (should be False in dry-run)")

    # v2 output root 충돌 확인
    if OUTPUT_ROOT.exists():
        print(f"  WARN: v2 output root already exists: {OUTPUT_ROOT}")
    else:
        print(f"  OK  v2 output root (not yet created): {OUTPUT_ROOT.name}")

    # 정적 검사
    passed, failed, si = static_check()
    issues.extend(si)
    print(f"\n  static_check: {passed} passed, {failed} failed")

    if issues:
        print("\n[dry-run] Issues:")
        for iss in issues:
            print(f"  - {iss}")
        sys.exit(1)
    else:
        print("\n[dry-run] PASS — 모든 입력 파일 존재, guard 위반 없음")
        print("NOTE: actual generation에서만 SOURCE_PNG_READ/RENDER/WRITE를 True로 설정 필요")


# ============================================================
# plan-only: layout 표시
# ============================================================
def plan_only() -> None:
    print(f"[plan-only] S5 Demo Card Prototype v2 — {CASE_ID}")
    print(f"  Layout : {LAYOUT_SELECTED}")
    print(f"  Panel 1: S3 thumbnail (크게) + bbox / reference crops 제거")
    print(f"  Panel 2: S5 lung-window overlay + 범례 (Yellow=highest / Orange=2nd)")
    print(f"  Panel 3: 3×3 patch response map schematic (not pixel heatmap, not Grad-CAM)")
    print(f"  Panel 4: Key finding / Interpretation / Caution / Disclaimer")
    print()
    print(f"  Input files:")
    print(f"  S3 card PNG         : {S3_CARD_PNG}")
    print(f"  S5 lung overlay     : {S5_LUNG_OVERLAY_PNG}")
    print(f"  S5 patch map JSON   : {S5_PATCH_MAP_JSON}")
    print()
    print(f"  Output root (v2)    : {OUTPUT_ROOT}")
    print(f"  OUT_PNG             : {OUT_PNG}")
    print(f"  OUT_JSON            : {OUT_JSON}")
    print()
    print(f"  Guards:")
    print(f"  ALLOW_SOURCE_PNG_READ    → must be set True (requires separate approval)")
    print(f"  ALLOW_PROTOTYPE_RENDER   → must be set True (requires separate approval)")
    print(f"  ALLOW_PROTOTYPE_WRITE    → must be set True (requires separate approval)")
    print()
    passed, failed, issues = static_check()
    if issues:
        for iss in issues:
            print(f"  ISSUE: {iss}")
    else:
        print("  OK — static check PASS")
    print("NOTE: actual generation에서만 SOURCE_PNG_READ/RENDER/WRITE를 True로 설정 필요")


# ============================================================
# actual generation
# ============================================================
def run_prototype() -> None:
    """
    actual generation — ALLOW_SOURCE_PNG_READ / ALLOW_PROTOTYPE_RENDER /
    ALLOW_PROTOTYPE_WRITE 모두 True일 때만 실행 가능.
    """
    if not (ALLOW_SOURCE_PNG_READ and ALLOW_PROTOTYPE_RENDER and ALLOW_PROTOTYPE_WRITE):
        print("BLOCKED: generation guards are False — set all three to True with explicit approval",
              file=sys.stderr)
        sys.exit(2)

    if ALLOW_CT_LOAD or ALLOW_MODEL_FORWARD or ALLOW_FEATURE_EXTRACTION:
        print("BLOCKED: CT/model/feature guards must remain False", file=sys.stderr)
        sys.exit(2)

    if ALLOW_S3_MODIFICATION or ALLOW_S4_MODIFICATION:
        print("BLOCKED: S3/S4 modification guards must remain False", file=sys.stderr)
        sys.exit(2)

    if ALLOW_STAGE2_HOLDOUT or ALLOW_FULL_300:
        print("BLOCKED: stage2/full300 guards must remain False", file=sys.stderr)
        sys.exit(2)

    collision_paths = [OUT_DONE, OUT_PNG, OUT_JSON, OUT_INDEX_CSV, OUT_RUNTIME, OUT_ERRORS]
    existing = [str(p) for p in collision_paths if p.exists()]
    if existing:
        print("BLOCKED: output collision detected. Do not overwrite existing files:", file=sys.stderr)
        for p in existing:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(2)

    t_start = time.time()
    errors: List[str] = []

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("BLOCKED: Pillow not available", file=sys.stderr)
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

    fnt_title   = _font(22, bold=True)
    fnt_section = _font(16, bold=True)
    fnt_body    = _font(13)
    fnt_small   = _font(11)
    fnt_label   = _font(12)

    # ---- 소스 PNG 읽기 (v2: S3 + lung overlay만 읽음) ----
    img_s3   = Image.open(S3_CARD_PNG).convert("RGBA")
    img_lung = Image.open(S5_LUNG_OVERLAY_PNG).convert("RGBA")

    # ---- JSON 데이터 로드 ----
    with open(S5_PATCH_MAP_JSON) as f:
        patch_map = json.load(f)

    # ---- 캔버스 치수 ----
    CANVAS_W     = 1440
    MARGIN       = 16
    HEADER_H     = 56
    PANEL_W      = (CANVAS_W - 3 * MARGIN) // 2

    # Panel 2 overlay 비율 유지
    ov_w, ov_h   = img_lung.size
    PANEL_OV_H   = int(PANEL_W * ov_h / ov_w)

    CANVAS_H = HEADER_H + MARGIN + PANEL_OV_H + MARGIN + PANEL_OV_H + MARGIN

    # ---- 색상 팔레트 ----
    C_BG         = (18,  18,  18,  255)
    C_PANEL_BG   = (32,  32,  32,  255)
    C_HEADER_BG  = (10,  25,  50,  255)
    C_BORDER     = (80,  80,  80,  255)
    C_TITLE      = (220, 220, 255, 255)
    C_SECTION    = (160, 200, 255, 255)
    C_BODY       = (220, 220, 220, 255)
    C_WARN       = (255, 220,  80, 255)
    C_DISCLAIMER = (200, 160, 160, 255)
    C_BBOX       = (255, 100, 100, 255)
    C_YELLOW     = (255, 220,   0, 255)
    C_ORANGE     = (255, 140,   0, 255)
    C_SCORE_LOW  = (50,  50,  70, 255)
    C_SCORE_HIGH = (100, 100, 150, 255)
    C_SCORE_TEXT = (200, 200, 200, 255)

    # ---- 캔버스 생성 ----
    canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), C_BG)
    draw   = ImageDraw.Draw(canvas)

    # ---- 헤더 ----
    draw.rectangle([(0, 0), (CANVAS_W - 1, HEADER_H - 1)], fill=C_HEADER_BG)
    draw.text((MARGIN, 10), PROTOTYPE_TITLE_EN, font=fnt_title, fill=C_TITLE)
    draw.text((MARGIN, 33), "[INTERNAL USE ONLY | font_fix_required | S4 NOT AVAILABLE | v2]",
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
        chars_per_line = max(1, max_w // max(1, int(font.getlength("가"))))
        wrapped = textwrap.fill(text, width=int(chars_per_line))
        lines   = wrapped.split("\n")
        cx, cy  = xy
        for line in lines:
            draw.text((cx, cy), line, font=font, fill=fill)
            cy += line_h
        return cy - xy[1]

    # ================================
    # Panel 1 — S3 thumbnail (크게, reference crops 없음)
    # ================================
    r1  = panel_rect(0, 0)
    draw_panel_bg(r1)
    pad = 10
    cx  = r1[0] + pad
    cy  = r1[1] + pad

    draw.text((cx, cy), "Panel 1: Reference comparison", font=fnt_section, fill=C_SECTION)
    cy += 24

    # S3 thumbnail — reference crops 없으므로 더 크게
    s3_avail_w = PANEL_W - 2 * pad
    s3_avail_h = PANEL_OV_H - 24 - 28 - 2 * pad  # 24=라벨, 28=하단 경고
    s3_scale   = min(s3_avail_w / img_s3.width, s3_avail_h / img_s3.height)
    s3_w       = int(img_s3.width  * s3_scale)
    s3_h       = int(img_s3.height * s3_scale)
    img_s3_th  = img_s3.resize((s3_w, s3_h), Image.LANCZOS)
    canvas.paste(img_s3_th, (cx, cy), img_s3_th)

    # bbox 표시
    bx0 = cx + int(S3_DISPLAY_BBOX[0] * s3_scale)
    by0 = cy + int(S3_DISPLAY_BBOX[1] * s3_scale)
    bx1 = cx + int(S3_DISPLAY_BBOX[2] * s3_scale)
    by1 = cy + int(S3_DISPLAY_BBOX[3] * s3_scale)
    draw.rectangle([(bx0, by0), (bx1, by1)], outline=C_BBOX, width=2)
    draw.text((bx0, by0 - 14), "bbox", font=fnt_small, fill=C_BBOX)

    # 하단 경고
    warn_y = r1[3] - 22
    draw.text((r1[0] + pad, warn_y), "Internal-use prototype — not for external distribution",
              font=fnt_small, fill=C_WARN)

    # ================================
    # Panel 2 — S5 lung-window overlay + 범례
    # ================================
    r2  = panel_rect(1, 0)
    draw_panel_bg(r2)
    cx2 = r2[0] + pad
    cy2 = r2[1] + pad

    draw.text((cx2, cy2), "Panel 2: Model response location", font=fnt_section, fill=C_SECTION)
    cy2 += 24

    legend_h = 36
    img_lung_r = img_lung.resize(
        (PANEL_W - 2 * pad, PANEL_OV_H - 24 - legend_h - 2 * pad), Image.LANCZOS
    )
    canvas.paste(img_lung_r, (cx2, cy2), img_lung_r)
    cy2 += img_lung_r.size[1] + 6

    # 범례
    draw.rectangle([(cx2, cy2), (cx2 + 14, cy2 + 14)], fill=(255, 220, 0, 255))
    draw.text((cx2 + 18, cy2), "Yellow: highest response", font=fnt_small, fill=C_BODY)
    cy2 += 18
    draw.rectangle([(cx2, cy2), (cx2 + 14, cy2 + 14)], fill=(255, 140, 0, 255))
    draw.text((cx2 + 18, cy2), "Orange: second-highest response below", font=fnt_small, fill=C_BODY)

    # ================================
    # Panel 3 — 3×3 Patch Response Map Schematic
    # ================================
    r3  = panel_rect(0, 1)
    draw_panel_bg(r3)
    cx3 = r3[0] + pad
    cy3 = r3[1] + pad
    tw3 = PANEL_W - 2 * pad

    draw.text((cx3, cy3), "Panel 3: Patch-level response map", font=fnt_section, fill=C_SECTION)
    cy3 += 24
    draw.text((cx3, cy3), "Patch score map, not pixel heatmap",
              font=fnt_small, fill=(180, 180, 180, 255))
    cy3 += 16

    # patch score map 로드
    score_map = patch_map.get("score_map_sqrt_mahalanobis", [[0]*3]*3)

    # 3×3 그리드 그리기
    GRID_SIZE    = min(tw3, PANEL_OV_H - 24 - 16 - 60)  # 60 = 범례 + 여백
    CELL_SIZE    = (GRID_SIZE - 4 * 2) // 3  # 4 = gap
    GAP          = 4
    ACTUAL_GRID  = CELL_SIZE * 3 + GAP * 2
    grid_x0      = cx3 + (tw3 - ACTUAL_GRID) // 2
    grid_y0      = cy3

    # 점수 정규화
    flat_scores  = [score_map[r][c] for r in range(3) for c in range(3)]
    min_s        = min(flat_scores) if flat_scores else 0.0
    max_s        = max(flat_scores) if flat_scores else 1.0
    score_range  = max(max_s - min_s, 1e-6)

    for r in range(3):
        for c in range(3):
            pid   = r * 3 + c
            score = score_map[r][c] if r < len(score_map) and c < len(score_map[r]) else 0.0
            norm  = (score - min_s) / score_range

            cell_x = grid_x0 + c * (CELL_SIZE + GAP)
            cell_y = grid_y0 + r * (CELL_SIZE + GAP)

            # 셀 배경색
            if pid == 4:
                bg = C_YELLOW
            elif pid == 7:
                bg = C_ORANGE
            else:
                intensity = int(40 + norm * 80)
                bg = (intensity, intensity, min(intensity + 30, 255), 255)

            draw.rectangle(
                [(cell_x, cell_y), (cell_x + CELL_SIZE - 1, cell_y + CELL_SIZE - 1)],
                fill=bg, outline=(100, 100, 100, 255), width=1
            )

            # patch_id 텍스트
            label_color = (30, 30, 30, 255) if pid in (4, 7) else C_SCORE_TEXT
            draw.text((cell_x + 4, cell_y + 4),
                      f"P{pid}", font=fnt_small, fill=label_color)
            draw.text((cell_x + 4, cell_y + CELL_SIZE - 18),
                      f"{score:.1f}", font=fnt_small, fill=label_color)

    cy3 = grid_y0 + ACTUAL_GRID + 10

    # 범례
    draw.rectangle([(grid_x0, cy3), (grid_x0 + 14, cy3 + 14)], fill=(255, 220, 0, 255))
    draw.text((grid_x0 + 18, cy3), "Highest response", font=fnt_small, fill=C_BODY)
    cy3 += 18
    draw.rectangle([(grid_x0, cy3), (grid_x0 + 14, cy3 + 14)], fill=(255, 140, 0, 255))
    draw.text((grid_x0 + 18, cy3), "Second-highest response", font=fnt_small, fill=C_BODY)
    cy3 += 20

    draw.text((cx3, cy3),
              "Scores: sqrt Mahalanobis distance (patch-level feature-space)",
              font=fnt_small, fill=(150, 150, 150, 255))

    # ================================
    # Panel 4 — Simple interpretation (의사용 4섹션)
    # ================================
    r4  = panel_rect(1, 1)
    draw_panel_bg(r4)
    cx4 = r4[0] + pad
    cy4 = r4[1] + pad
    tw4 = PANEL_W - 2 * pad

    draw.text((cx4, cy4), "Panel 4: Simple interpretation", font=fnt_section, fill=C_SECTION)
    cy4 += 28

    def draw_section(label: str, text_ko: str, text_en: str,
                     label_color: tuple, text_color: tuple) -> None:
        nonlocal cy4
        draw.text((cx4, cy4), label, font=fnt_label, fill=label_color)
        cy4 += 18
        h = wrap_draw_text((cx4, cy4), text_ko, fnt_body, text_color, tw4, 17)
        cy4 += h + 4
        h = wrap_draw_text((cx4, cy4), text_en, fnt_small, (180, 180, 180, 255), tw4, 15)
        cy4 += h + 12

    draw_section(
        "[Key finding]",
        P4_KEY_FINDING_KO, P4_KEY_FINDING_EN,
        C_SECTION, C_BODY
    )
    draw_section(
        "[Interpretation]",
        P4_INTERPRETATION_KO, P4_INTERPRETATION_EN,
        C_SECTION, C_BODY
    )
    draw_section(
        "[Caution]",
        P4_CAUTION_KO, P4_CAUTION_EN,
        C_WARN, C_WARN
    )

    # Disclaimer
    draw.text((cx4, cy4), "[Disclaimer]", font=fnt_label, fill=C_DISCLAIMER)
    cy4 += 18
    wrap_draw_text((cx4, cy4), P4_DISCLAIMER_KO + "  " + P4_DISCLAIMER_EN,
                   fnt_small, C_DISCLAIMER, tw4, 15)

    # ---- 출력 디렉토리 생성 ----
    CARDS_PNG_DIR.mkdir(parents=True, exist_ok=True)
    CARDS_JSON_DIR.mkdir(parents=True, exist_ok=True)

    # ---- PNG 저장 ----
    canvas_rgb = canvas.convert("RGB")
    canvas_rgb.save(str(OUT_PNG), "PNG")
    print(f"  PNG saved: {OUT_PNG}")

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

    # ---- index_cards_v2.csv ----
    with open(OUT_INDEX_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "prototype_type", "layout", "png_path", "json_path", "generated_date"])
        w.writerow([CASE_ID, "S3_PLUS_S5_DEMO_CARD_PROTOTYPE_V2_CLINICAL_READABLE",
                    LAYOUT_SELECTED, str(OUT_PNG), str(OUT_JSON), "2026-06-09"])
    print(f"  index_cards_v2.csv saved: {OUT_INDEX_CSV}")

    # ---- runtime_summary_v2.json ----
    elapsed = time.time() - t_start
    runtime = {
        "case_id": CASE_ID,
        "version": "v2_clinical_readable",
        "elapsed_sec": round(elapsed, 2),
        "canvas_w": CANVAS_W,
        "canvas_h": CANVAS_H,
        "source_png_read": 2,   # s3 + lung (v2에서는 padim overlay/reference crops 읽지 않음)
        "png_write": 1,
        "json_write": 1,
        "existing_artifact_modified": False,
        "errors": len(errors)
    }
    OUT_RUNTIME.parent.mkdir(parents=True, exist_ok=True)
    OUT_RUNTIME.write_text(json.dumps(runtime, indent=2))
    print(f"  runtime_summary_v2.json saved ({elapsed:.2f}s)")

    # ---- errors.csv ----
    with open(OUT_ERRORS, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "error_type", "message", "timestamp"])
        for err in errors:
            w.writerow([CASE_ID, "runtime_error", err, "2026-06-09"])
    print(f"  errors.csv saved: {OUT_ERRORS}")

    # ---- DONE.json ----
    done = {
        "case_id": CASE_ID,
        "version": "v2_clinical_readable",
        "status": "DONE",
        "png_path": str(OUT_PNG),
        "json_path": str(OUT_JSON),
        "generated_date": "2026-06-09",
        "elapsed_sec": round(elapsed, 2),
        "errors": len(errors)
    }
    OUT_DONE.write_text(json.dumps(done, indent=2))
    print(f"  DONE.json saved: {OUT_DONE}")

    print(f"\n[run-prototype] COMPLETE — {CASE_ID} v2")


# ============================================================
# main
# ============================================================
def main() -> None:
    args = sys.argv[1:]

    if "--dry-run" in args:
        dry_run()
    elif "--plan-only" in args:
        plan_only()
    elif "--static-check" in args:
        passed, failed, issues = static_check()
        print(f"[static-check] {passed} passed, {failed} failed")
        if issues:
            for iss in issues:
                print(f"  {iss}")
            sys.exit(1)
        else:
            print("  PASS")
    elif "--run-prototype" in args:
        if "--confirm-generate" not in args:
            print("ERROR: --run-prototype requires --confirm-generate", file=sys.stderr)
            sys.exit(1)
        run_prototype()
    else:
        print("Usage:")
        print("  python build_s5_demo_card_prototype_lung1_052_c3_v2_clinical_readable.py --dry-run")
        print("  python build_s5_demo_card_prototype_lung1_052_c3_v2_clinical_readable.py --plan-only")
        print("  python build_s5_demo_card_prototype_lung1_052_c3_v2_clinical_readable.py --static-check")
        print("  python build_s5_demo_card_prototype_lung1_052_c3_v2_clinical_readable.py --run-prototype --confirm-generate")
        sys.exit(0)


if __name__ == "__main__":
    main()
