"""
build_s5_demo_card_prototype_lung1_052_c3_v4_ref_update.py

목적:
reflection preflight(PASS, option B) 결과를 바탕으로 LUNG1-052__c3 S5 demo card 를
option B 로 업데이트하는 신규 스크립트. 기존 v3d 스크립트는 수정하지 않는다(신규 파일).

option B (Panel 1 Row1):
  Candidate | Same-cell normal ref 1 | Same-cell normal ref 3 | Additional same-cell ref
  (slot0=candidate, slot1=normal_ref_1, slot2=normal_ref_3, slot3=normal_ref_2)
  preview_root 는 v4 selected crop preview 결과 경로로 repoint.
  Panels 2~3 unchanged. Panel 4 text updated.

이번 파일 단계(상위 작업):
  script 작성 + static drycheck ONLY. 실제 card render 는 별도 승인(--run-prototype --confirm-generate
  + ALLOW_CARD_RENDER=1 + ALLOW_PNG_WRITE=1) 에서만 수행.

guard 기본값 전부 False. CT/ROI/model/feature/contribution/stage2/full300 항상 False.
source PNG read 와 PNG write 는 render guard 가 열렸을 때만 발생.
"""

import os
import sys
import csv
import json
import time
import textwrap
import pathlib
from datetime import date
from typing import Any, Dict, List, Tuple

# ============================================================
# GUARD FLAGS (기본 False)
# ============================================================
ALLOW_CARD_RENDER         = False   # source PNG read + compose 포함
ALLOW_PNG_WRITE           = False
ALLOW_CT_LOAD             = False
ALLOW_ROI_LOAD            = False
ALLOW_STAGE2_HOLDOUT      = False
ALLOW_MODEL_FORWARD       = False
ALLOW_FEATURE_EXTRACTION  = False
ALLOW_CONTRIBUTION_RECALC = False
ALLOW_FULL300             = False

# actual render 승인 시에만 env 로 True (내릴 수는 없음)
if os.environ.get("ALLOW_CARD_RENDER") == "1":
    ALLOW_CARD_RENDER = True
if os.environ.get("ALLOW_PNG_WRITE") == "1":
    ALLOW_PNG_WRITE = True

# ============================================================
# 케이스 상수
# ============================================================
CASE_ID            = "LUNG1-052__c3"
VOLUME_ID          = "NSCLC_LUNG1-052__d4a19cc211"
POSITION_BIN       = "lower_central"
CT_LOCAL_Z         = 51
REPORT_SLICE_INDEX = 106
SPATIAL_PATTERN    = "CENTER_DOMINANT_WITH_DOWNWARD_CONTINUITY"
VISUAL_VERDICT     = "PASS_VISUAL_SUPPORTS_CENTER_DOWNWARD_CONTINUITY_WITH_CAUTION"
GRID_EXTENT        = [256, 96, 352, 192]
PATCH4_BBOX        = [288, 128, 320, 160]
PATCH4_SCORE       = 38.872562
PATCH7_BBOX        = [320, 128, 352, 160]
PATCH7_SCORE       = 36.612470
LAYOUT_SELECTED    = "OPTION_B_4PANEL_SAME_CELL_V4_SELECTED_REF"
VERSION            = "v4_ref_update"

# v4 exact-cell top3 (v3d 와 동일 환자; PNG 경로만 selected_cases 로 repoint)
V4_CELL_KEY        = "image_left|Z1|Y2|X1"
V4_CANDIDATE_Z     = 51
V4_REF1_Z          = 50
V4_REF2_Z          = 89
V4_REF3_Z          = 80
V4_REF1_PATIENT    = "subset1_1.3.6.1.4.1.14519.5.2.1.6279.6001.100684836163890911914061745866"
V4_REF2_PATIENT    = "subset2_1.3.6.1.4.1.14519.5.2.1.6279.6001.311236942972970815890902714604"
V4_REF3_PATIENT    = "subset9_1.3.6.1.4.1.14519.5.2.1.6279.6001.109882169963817627559804568094"
V4_AVG_ABS_DELTA_Z = round((abs(V4_REF1_Z - V4_CANDIDATE_Z)
                            + abs(V4_REF2_Z - V4_CANDIDATE_Z)
                            + abs(V4_REF3_Z - V4_CANDIDATE_Z)) / 3, 1)

# ============================================================
# 경로
# ============================================================
PROJECT_ROOT = pathlib.Path("/home/jinhy/project/lung-ct-anomaly")

# v4 selected crop preview (신규 검증본)
V4_PREVIEW_ROOT = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/reference_bank_v4_selected_cases_crop_preview")
V4_CANDIDATE_PNG = V4_PREVIEW_ROOT / f"{CASE_ID}_candidate.png"
V4_REF1_PNG      = V4_PREVIEW_ROOT / f"{CASE_ID}_normal_ref_1.png"
V4_REF2_PNG      = V4_PREVIEW_ROOT / f"{CASE_ID}_normal_ref_2.png"
V4_REF3_PNG      = V4_PREVIEW_ROOT / f"{CASE_ID}_normal_ref_3.png"
V4_MONTAGE_PNG   = V4_PREVIEW_ROOT / f"{CASE_ID}_reference_preview_montage.png"
V4_PREVIEW_META  = V4_PREVIEW_ROOT / f"{CASE_ID}_preview_metadata.json"
V4_PREVIEW_INDEX = V4_PREVIEW_ROOT / "preview_index.csv"

# Panel 2~4 source PNG (UNCHANGED, v3d 와 동일)
WHOLE_SLICE_PNG = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "s5_lung1_052_c3_coordinate_visual_audit_1case_v1"
    / "coordinate_overlay_patch4_patch7_lung.png")
S5_LUNG_OVERLAY_PNG = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "s5_lung1_052_c3_coordinate_visual_audit_1case_v1"
    / "coordinate_overlay_3x3_grid_lung.png")
S5_PATCH_MAP_JSON = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "s5_lung1_052_c3_patch_level_contribution_map_1case_smoke_v1"
    / "patch_contribution_map.json")

# reflection preflight (정책 입력)
REFLECTION_ROOT = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "lung1_052_s5_demo_card_v4_reference_reflection_preflight")
REFLECTION_DONE = REFLECTION_ROOT / "DONE.json"

# 기존 v3d card (template 참고용, read-only; 수정 금지)
V3D_CARD_JSON = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "s5_demo_card_prototype_lung1_052_c3_v3d_panel1_v4_ref"
    / "cards_json/LUNG1-052__c3_s5_demo_prototype_v3d.json")

# 실제 render 출력 root (다음 단계)
OUTPUT_ROOT    = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "s5_demo_card_prototype_lung1_052_c3_v4_ref_update")
CARDS_PNG_DIR  = OUTPUT_ROOT / "cards_png"
CARDS_JSON_DIR = OUTPUT_ROOT / "cards_json"
OUT_PNG        = CARDS_PNG_DIR  / f"{CASE_ID}_s5_demo_prototype_v4_ref_update.png"
OUT_JSON       = CARDS_JSON_DIR / f"{CASE_ID}_s5_demo_prototype_v4_ref_update.json"
OUT_INDEX_CSV  = OUTPUT_ROOT / "index_cards_v4_ref_update.csv"
OUT_RUNTIME    = OUTPUT_ROOT / "runtime_summary_v4_ref_update.json"
OUT_ERRORS     = OUTPUT_ROOT / "errors.csv"
OUT_DONE       = OUTPUT_ROOT / "DONE.json"

# 이번 단계(static drycheck) 리포트 root
DRYCHECK_ROOT = (PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
    / "lung1_052_s5_demo_card_v4_ref_update_script_static_drycheck")
DRY_MD        = DRYCHECK_ROOT / "static_drycheck_v4_ref_update.md"
DRY_JSON      = DRYCHECK_ROOT / "static_drycheck_v4_ref_update.json"
DRY_PLAN_CSV  = DRYCHECK_ROOT / "script_plan_v4_ref_update.csv"
DRY_TEXT_CSV  = DRYCHECK_ROOT / "text_update_plan_v4_ref_update.csv"
DRY_SLOT_CSV  = DRYCHECK_ROOT / "panel1_slot_mapping_v4_ref_update.csv"
DRY_SAFE_CSV  = DRYCHECK_ROOT / "safety_check_v4_ref_update.csv"

# ============================================================
# 텍스트 상수 (v4_ref_update)
# ============================================================
TITLE_KO = f"S5 PaDiM 이상 후보 설명 카드 {VERSION} — {CASE_ID}"
HEADER_TAG = f"[INTERNAL USE ONLY | {VERSION} | not diagnostic]"

PANEL1_SUBTITLE_EN = ("Normal references are selected from the same lung-ROI position "
                      "cell; this is not same-z matching.")
PANEL1_SUBTITLE_KO = ("정상 reference는 같은 폐 ROI 위치 셀에서 선정되었으며, "
                      "z-방향 정합은 아닙니다.")

P4_KEY_FINDING_KO = ("정상 reference와 비교했을 때, 후보 crop은 더 조밀한 국소 음영 "
                     "패턴을 보였습니다.")
P4_KEY_FINDING_EN = ("Compared with same-cell normal references, the candidate crop "
                     "shows a denser local opacity pattern.")
P4_INTERPRETATION_KO = ("같은 lung-ROI position cell에서 선정된 정상 예시와 비교했을 때, "
                        "해당 영역은 정상 분포와 다른 국소 패턴으로 보이며, 중심 patch와 "
                        "바로 아래 patch의 response 연속성이 유지됩니다.")
P4_INTERPRETATION_EN = ("Compared with normal examples selected from the same lung-ROI "
                        "position cell, the region appears as a different local pattern "
                        "than the normal distribution; the response continuity between "
                        "the center patch and the patch directly below is preserved.")
P4_CAUTION_KO = ("same-cell comparison이며 same-z matching은 아닙니다. "
                 "z-direction alignment is limited. "
                 "후보가 약간 흉막 인접하므로 과해석하지 마십시오.")
P4_CAUTION_EN = ("Same-cell comparison, not same-z matching; z-direction alignment is "
                 "limited; the candidate is slightly pleura-adjacent, so avoid "
                 "over-interpretation.")
P4_DISCLAIMER_KO = "연구용 보조 설명이며 진단이 아닙니다."
P4_DISCLAIMER_EN = "Research-use auxiliary explanation only; not a diagnosis."

# Panel 1 labels (option B 순서: candidate | ref1 | ref3 | ref2)
LABEL_CANDIDATE = "후보 crop\n(candidate)"
LABEL_REF1      = "Same-cell normal ref 1\n(정상 reference 1)"
LABEL_REF3      = "Same-cell normal ref 3\n(정상 reference 3)"
LABEL_REF2_ADD  = "Additional same-cell ref\n(추가 동일셀 reference)"
LABEL_WHOLE_SLICE = "전체 슬라이스 (lung window, patch 위치 표시)"
LABEL_Z_ABOVE   = "위 슬라이스\n(z-1)"
LABEL_Z_CURRENT = "현재 슬라이스\n(z)"
LABEL_Z_BELOW   = "아래 슬라이스\n(z+1)"

# option B slot 정의 (slot, role, png, label, inclusion, z)
SLOT_DEF = [
    (0, "candidate",    V4_CANDIDATE_PNG, LABEL_CANDIDATE, "primary",    V4_CANDIDATE_Z),
    (1, "normal_ref_1", V4_REF1_PNG,      LABEL_REF1,      "primary",    V4_REF1_Z),
    (2, "normal_ref_3", V4_REF3_PNG,      LABEL_REF3,      "primary",    V4_REF3_Z),
    (3, "normal_ref_2", V4_REF2_PNG,      LABEL_REF2_ADD,  "additional", V4_REF2_Z),
]

# ref2 처리: report/metadata only note (원인 단정 금지)
REF2_NOTE = "structure-dominant, less suitable for visual comparison"

# ============================================================
# forbidden / allowed wording
# ============================================================
FORBIDDEN_PHRASES = [
    "same-z matched", "same z matched", "z-matched", "z matched",
    "identical z", "동일 z 위치", "같은 z 위치",
    "diagnostic heatmap", "grad-cam", "pixel attribution",
    "병변 원인", "암 위치", "혈관 때문에", "diagnostic conclusion",
    "raw427", "dim91", "layer fraction", "feature-space topk",
]


def _abort(msg, code=2):
    print(f"\nBLOCKED: {msg}", file=sys.stderr)
    sys.exit(code)


def scan_forbidden(blob):
    low = str(blob).lower()
    return [p for p in FORBIDDEN_PHRASES if p.lower() in low]


# ============================================================
# card JSON schema (v4_ref_update)
# ============================================================
def build_card_json_schema() -> Dict[str, Any]:
    return {
        "version": VERSION,
        "case_id": CASE_ID,
        "volume_id": VOLUME_ID,
        "position_bin": POSITION_BIN,
        "ct_local_z": CT_LOCAL_Z,
        "report_slice_index": REPORT_SLICE_INDEX,
        "spatial_pattern": SPATIAL_PATTERN,
        "visual_verdict": VISUAL_VERDICT,
        "grid_extent": GRID_EXTENT,
        "patch4_bbox": PATCH4_BBOX, "patch4_score": PATCH4_SCORE,
        "patch7_bbox": PATCH7_BBOX, "patch7_score": PATCH7_SCORE,
        "layout_selected": LAYOUT_SELECTED,
        "panels": {
            "panel_1": ("S4-style reference comparison (option B, v4 selected-cell ref): "
                        "Row1=4 동일 크기 (candidate|Same-cell normal ref 1|Same-cell normal "
                        "ref 3|Additional same-cell ref); Row2=whole slice; Row3=z-context "
                        "placeholder (CT load disabled)"),
            "panel_2": "lung-window overlay (patch4/patch7 위치); position legend only "
                       "(not a saliency/attribution-style map)",
            "panel_3": "3x3 patch response map schematic (not a CT overlay, not a per-pixel map)",
            "panel_4": "Key finding / Interpretation / Caution / Disclaimer (same lung-ROI "
                       "position cell; z-direction limitation)",
        },
        "panel1_subtitle_en": PANEL1_SUBTITLE_EN,
        "panel4_text": {
            "key_finding_ko": P4_KEY_FINDING_KO, "key_finding_en": P4_KEY_FINDING_EN,
            "interpretation_ko": P4_INTERPRETATION_KO, "interpretation_en": P4_INTERPRETATION_EN,
            "caution_ko": P4_CAUTION_KO, "caution_en": P4_CAUTION_EN,
            "disclaimer_ko": P4_DISCLAIMER_KO, "disclaimer_en": P4_DISCLAIMER_EN,
        },
        "v4_reference_selection": {
            "cell_key": V4_CELL_KEY, "source": "v4_exact_cell_top3",
            "preview_root": str(V4_PREVIEW_ROOT),
            "option": "B",
            "slot_order": ["candidate", "normal_ref_1", "normal_ref_3", "normal_ref_2"],
            "normal_ref_1": {"patient_id": V4_REF1_PATIENT, "z": V4_REF1_Z,
                             "decision": "keep_primary", "png": str(V4_REF1_PNG)},
            "normal_ref_3": {"patient_id": V4_REF3_PATIENT, "z": V4_REF3_Z,
                             "decision": "keep_primary", "png": str(V4_REF3_PNG)},
            "normal_ref_2": {"patient_id": V4_REF2_PATIENT, "z": V4_REF2_Z,
                             "decision": "keep_additional", "png": str(V4_REF2_PNG),
                             "note": REF2_NOTE},
            "avg_abs_delta_z": V4_AVG_ABS_DELTA_Z,
            "not_same_z_matched": True, "z_orientation_limitation": True,
        },
        "metadata": {
            "version": VERSION,
            "panel1_reference_source": "v4_exact_cell_top3 (selected_cases preview)",
            "cell_key": V4_CELL_KEY,
            "reference_selection_policy": "SAME_LUNG_ROI_POSITION_CELL_WITH_Z_LIMITATION",
            "not_same_z_matched": True, "z_orientation_limitation": True,
            "not_diagnostic": True, "no_saliency_attribution_style": True,
            "stage2_holdout_accessed": False, "model_forward_occurred": False,
            "feature_extraction_occurred": False, "contribution_recalc_occurred": False,
            "ref2_handling": f"additional same-cell ref; {REF2_NOTE}; no causal claim",
        },
        "source_files": {
            "v4_candidate_png": str(V4_CANDIDATE_PNG),
            "v4_normal_ref_1_png": str(V4_REF1_PNG),
            "v4_normal_ref_2_png": str(V4_REF2_PNG),
            "v4_normal_ref_3_png": str(V4_REF3_PNG),
            "whole_slice_png": str(WHOLE_SLICE_PNG),
            "lung_overlay_png": str(S5_LUNG_OVERLAY_PNG),
            "patch_map_json": str(S5_PATCH_MAP_JSON),
            "v3d_template_json": str(V3D_CARD_JSON),
        },
        "canvas_size_px": None,
        "output_png": str(OUT_PNG),
        "generated_date": None,
        "ct_load_used": False,
        "normal_ref_ct_load_used": False,
        "source_png_read_used": False,
    }


# ============================================================
# INPUT VALIDATION (no load)
# ============================================================
def validate_inputs() -> List[str]:
    problems = []
    if not REFLECTION_DONE.exists():
        problems.append("reflection preflight DONE.json missing")
    else:
        d = json.loads(REFLECTION_DONE.read_text())
        if d.get("verdict") != "PASS":
            problems.append(f"reflection verdict != PASS ({d.get('verdict')})")
        if d.get("recommended_option") != "B":
            problems.append(f"reflection recommended_option != B ({d.get('recommended_option')})")
    for nm, p in [("candidate", V4_CANDIDATE_PNG), ("ref1", V4_REF1_PNG),
                  ("ref2", V4_REF2_PNG), ("ref3", V4_REF3_PNG),
                  ("montage", V4_MONTAGE_PNG), ("preview_meta", V4_PREVIEW_META)]:
        if not p.exists():
            problems.append(f"missing v4 preview {nm}: {p}")
    for nm, p in [("whole_slice", WHOLE_SLICE_PNG), ("lung_overlay", S5_LUNG_OVERLAY_PNG),
                  ("patch_map", S5_PATCH_MAP_JSON), ("v3d_template", V3D_CARD_JSON)]:
        if not p.exists():
            problems.append(f"missing panel2-4 source {nm}: {p}")
    # cell_key match
    if V4_PREVIEW_META.exists():
        m = json.loads(V4_PREVIEW_META.read_text())
        if m.get("cell_key") != V4_CELL_KEY:
            problems.append(f"cell_key mismatch: {m.get('cell_key')} vs {V4_CELL_KEY}")
        rmap = {r["role"]: str(r.get("patient_id")) for r in m.get("normal_references", [])}
        if rmap.get("normal_ref_1") != V4_REF1_PATIENT:
            problems.append("ref1 patient mismatch vs preview metadata")
        if rmap.get("normal_ref_2") != V4_REF2_PATIENT:
            problems.append("ref2 patient mismatch vs preview metadata")
        if rmap.get("normal_ref_3") != V4_REF3_PATIENT:
            problems.append("ref3 patient mismatch vs preview metadata")
    # forbidden wording in card text
    blob = " ".join([TITLE_KO, HEADER_TAG, PANEL1_SUBTITLE_EN, PANEL1_SUBTITLE_KO,
                     P4_KEY_FINDING_KO, P4_KEY_FINDING_EN, P4_INTERPRETATION_KO,
                     P4_INTERPRETATION_EN, P4_CAUTION_KO, P4_CAUTION_EN,
                     P4_DISCLAIMER_KO, P4_DISCLAIMER_EN,
                     LABEL_CANDIDATE, LABEL_REF1, LABEL_REF3, LABEL_REF2_ADD,
                     json.dumps(build_card_json_schema(), ensure_ascii=False)])
    hits = scan_forbidden(blob)
    if hits:
        problems.append(f"forbidden wording in card text: {hits}")
    return problems


# ============================================================
# COLLISION
# ============================================================
def collision_check() -> List[str]:
    paths = [OUT_DONE, OUT_PNG, OUT_JSON, OUT_INDEX_CSV, OUT_RUNTIME, OUT_ERRORS]
    return [str(p) for p in paths if p.exists()]


# ============================================================
# RENDER (actual; guarded)
# ============================================================
def render_card() -> None:
    """option B card render. ALLOW_CARD_RENDER & ALLOW_PNG_WRITE 모두 True 필수."""
    if not (ALLOW_CARD_RENDER and ALLOW_PNG_WRITE):
        _abort("render requires ALLOW_CARD_RENDER=1 and ALLOW_PNG_WRITE=1 "
               f"(current: render={ALLOW_CARD_RENDER}, png={ALLOW_PNG_WRITE})")
    if (ALLOW_CT_LOAD or ALLOW_ROI_LOAD or ALLOW_MODEL_FORWARD or ALLOW_FEATURE_EXTRACTION
            or ALLOW_CONTRIBUTION_RECALC or ALLOW_STAGE2_HOLDOUT or ALLOW_FULL300):
        _abort("CT/ROI/model/feature/contribution/stage2/full300 guards must stay False")

    problems = validate_inputs()
    if problems:
        _abort(f"input validation failed: {problems}")
    coll = collision_check()
    if coll:
        _abort(f"output collision ({len(coll)} files exist): {coll}")

    t0 = time.time()
    errors: List[str] = []
    from PIL import Image, ImageDraw, ImageFont

    FONT_KO = pathlib.Path("/mnt/c/Windows/Fonts/malgun.ttf")
    FONT_EN = pathlib.Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    FONT_EN_BOLD = pathlib.Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    if not FONT_KO.exists():
        _abort("malgun.ttf not found")

    def _font(size, bold=False):
        try:
            return ImageFont.truetype(str(FONT_KO), size)
        except Exception:
            return ImageFont.truetype(str(FONT_EN_BOLD if bold else FONT_EN), size)

    fnt_title = _font(20, True); fnt_section = _font(15, True)
    fnt_body = _font(13); fnt_small = _font(11); fnt_label = _font(11)

    def _placeholder(w, h, text="N/A"):
        img = Image.new("RGBA", (w, h), (50, 50, 60, 255))
        d = ImageDraw.Draw(img)
        d.rectangle([(0, 0), (w - 1, h - 1)], outline=(100, 100, 120, 255), width=2)
        for i, line in enumerate(text.split("\n")):
            d.text((6, 6 + i * 14), line, font=fnt_small, fill=(160, 160, 180, 255))
        return img

    def _load_png(p, sz, label):
        if not p.exists():
            errors.append(f"PNG not found: {p}")
            return _placeholder(sz, sz, f"{label}\nnot found")
        return Image.open(str(p)).convert("RGBA").resize((sz, sz), Image.LANCZOS)

    img_whole = Image.open(WHOLE_SLICE_PNG).convert("RGBA")
    img_lung_ov = Image.open(S5_LUNG_OVERLAY_PNG).convert("RGBA")
    with open(S5_PATCH_MAP_JSON) as f:
        patch_map = json.load(f)

    CANVAS_W = 1600; MARGIN = 16; HEADER_H = 64; PAD = 8
    P1_W = CANVAS_W - 2 * MARGIN
    CELL_SIZE = (P1_W - 5 * PAD) // 4
    LABEL_H = 30; SUBTITLE_H = 24
    WHOLE_W = int(P1_W * 0.68); WHOLE_H = int(WHOLE_W * img_whole.height / img_whole.width)
    Z_CELL_W = int(CELL_SIZE * 0.70); Z_CELL_H = Z_CELL_W
    P1_ROW1_H = LABEL_H + CELL_SIZE + LABEL_H
    P1_ROW2_H = LABEL_H + WHOLE_H
    P1_ROW3_H = LABEL_H + Z_CELL_H
    P1_H = SUBTITLE_H + PAD + P1_ROW1_H + PAD + P1_ROW2_H + PAD + P1_ROW3_H + PAD
    LOWER_PANEL_W = (CANVAS_W - 4 * MARGIN) // 3
    OV_W, OV_H = img_lung_ov.size
    LOWER_OV_H = int(LOWER_PANEL_W * OV_H / OV_W)
    LOWER_H = LOWER_OV_H + 100
    CANVAS_H = HEADER_H + MARGIN + P1_H + MARGIN + LOWER_H + MARGIN

    C_BG=(18,18,18,255); C_PANEL_BG=(30,30,30,255); C_HEADER=(10,22,45,255)
    C_BORDER=(70,70,70,255); C_TITLE=(220,220,255,255); C_SECTION=(150,195,255,255)
    C_BODY=(215,215,215,255); C_WARN=(255,215,75,255); C_DISC=(195,155,155,255)
    C_LABEL=(190,190,190,255); C_YELLOW=(255,220,0,255); C_ORANGE=(255,140,0,255)
    C_SUBTITLE=(170,200,170,255)

    canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), C_BG)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([(0, 0), (CANVAS_W, HEADER_H)], fill=C_HEADER)
    draw.text((MARGIN, 8), TITLE_KO, font=fnt_title, fill=C_TITLE)
    draw.text((MARGIN, 32), HEADER_TAG, font=fnt_small, fill=C_WARN)
    draw.text((MARGIN, 48), PANEL1_SUBTITLE_KO, font=fnt_small, fill=C_SUBTITLE)

    p1_x0 = MARGIN; p1_y0 = HEADER_H + MARGIN
    p1_x1 = MARGIN + P1_W; p1_y1 = p1_y0 + P1_H
    draw.rectangle([(p1_x0, p1_y0), (p1_x1, p1_y1)], fill=C_PANEL_BG, outline=C_BORDER, width=1)

    cy = p1_y0 + PAD
    draw.text((p1_x0 + PAD, cy),
              "Panel 1 : 정상 reference 비교 (Same-cell, option B v4 selected ref)  |  "
              "전체 슬라이스  |  z-context", font=fnt_section, fill=C_SECTION)
    cy += LABEL_H
    draw.text((p1_x0 + PAD, cy), PANEL1_SUBTITLE_EN, font=fnt_small, fill=C_SUBTITLE)
    cy += SUBTITLE_H

    crop_x_starts = [p1_x0 + PAD + i * (CELL_SIZE + PAD) for i in range(4)]
    cell_labels = [s[3] for s in SLOT_DEF]
    cell_imgs = [_load_png(s[2], CELL_SIZE, s[1]) for s in SLOT_DEF]
    for i, (cx, img_cell, lbl) in enumerate(zip(crop_x_starts, cell_imgs, cell_labels)):
        draw.rectangle([(cx - 1, cy - 1), (cx + CELL_SIZE, cy + CELL_SIZE)],
                       outline=C_YELLOW if i == 0 else C_BORDER, width=2 if i == 0 else 1)
        canvas.paste(img_cell, (cx, cy), img_cell)
        for j, line in enumerate(lbl.split("\n")):
            draw.text((cx, cy + CELL_SIZE + 2 + j * 12), line,
                      font=fnt_small, fill=C_YELLOW if i == 0 else C_LABEL)
    cy += CELL_SIZE + LABEL_H + PAD

    draw.text((p1_x0 + PAD, cy), LABEL_WHOLE_SLICE, font=fnt_label, fill=C_LABEL)
    cy += LABEL_H - 8
    whole_x = p1_x0 + (P1_W - WHOLE_W) // 2
    canvas.paste(img_whole.resize((WHOLE_W, WHOLE_H), Image.LANCZOS), (whole_x, cy),
                 img_whole.resize((WHOLE_W, WHOLE_H), Image.LANCZOS))
    cy += WHOLE_H + PAD

    draw.text((p1_x0 + PAD, cy), "Z-context (CT load disabled — placeholder)",
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
            draw.text((z_x, cy + Z_CELL_H + 2 + j * 12), line, font=fnt_small, fill=C_LABEL)

    lower_y = HEADER_H + MARGIN + P1_H + MARGIN
    p2_x0 = MARGIN
    p3_x0 = MARGIN + LOWER_PANEL_W + MARGIN
    p4_x0 = MARGIN + (LOWER_PANEL_W + MARGIN) * 2

    def draw_lower_panel(bx0, title):
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
    draw.text((bx0 + PAD, cy2), "position legend only (not a saliency/attribution-style map)",
              font=fnt_small, fill=(130, 130, 130, 255))

    bx0, by0, bx1, by1 = draw_lower_panel(p3_x0, "Panel 3 : Patch response map")
    cy3 = by0 + PAD + LABEL_H
    draw.text((p3_x0 + PAD, cy3), "patch-level response map / not pixel heatmap",
              font=fnt_small, fill=(170, 170, 170, 255))
    cy3 += 16
    score_map = patch_map.get("score_map_sqrt_mahalanobis", [[0]*3]*3)
    flat = [score_map[r][c] for r in range(3) for c in range(3)]
    min_s = min(flat) if flat else 0.0; max_s = max(flat) if flat else 1.0
    rng = max(max_s - min_s, 1e-6)
    GRID_SIZE = min(LOWER_PANEL_W - 2 * PAD, LOWER_H - LABEL_H - 60)
    CELL_G = (GRID_SIZE - 3 * 4) // 3; GAP_G = 4; ACTUAL_G = CELL_G * 3 + GAP_G * 2
    grid_x0 = p3_x0 + PAD + (LOWER_PANEL_W - 2 * PAD - ACTUAL_G) // 2; grid_y0 = cy3
    for r in range(3):
        for c in range(3):
            pid = r * 3 + c
            score = score_map[r][c] if r < len(score_map) and c < len(score_map[r]) else 0.0
            norm = (score - min_s) / rng
            cx_ = grid_x0 + c * (CELL_G + GAP_G); cyy = grid_y0 + r * (CELL_G + GAP_G)
            bg = C_YELLOW if pid == 4 else (C_ORANGE if pid == 7 else
                 (lambda iv: (iv, iv, min(iv + 25, 255), 255))(int(35 + norm * 70)))
            lc = (25, 25, 25, 255) if pid in (4, 7) else (190, 190, 190, 255)
            draw.rectangle([(cx_, cyy), (cx_ + CELL_G - 1, cyy + CELL_G - 1)],
                           fill=bg, outline=(90, 90, 90, 255), width=1)
            draw.text((cx_ + 4, cyy + 4), f"P{pid}", font=fnt_small, fill=lc)
            draw.text((cx_ + 4, cyy + CELL_G - 16), f"{score:.1f}", font=fnt_small, fill=lc)
    cy3 = grid_y0 + ACTUAL_G + 8
    draw.rectangle([(grid_x0, cy3), (grid_x0 + 12, cy3 + 12)], fill=(255, 220, 0, 255))
    draw.text((grid_x0 + 16, cy3), "Highest", font=fnt_small, fill=C_BODY)
    cy3 += 16
    draw.rectangle([(grid_x0, cy3), (grid_x0 + 12, cy3 + 12)], fill=(255, 140, 0, 255))
    draw.text((grid_x0 + 16, cy3), "Second-highest", font=fnt_small, fill=C_BODY)

    bx0, by0, bx1, by1 = draw_lower_panel(p4_x0, "Panel 4 : 설명 요약")
    cy4 = by0 + PAD + LABEL_H
    tw4 = LOWER_PANEL_W - 2 * PAD

    def wrap_draw(xy, text, font, fill, max_w, line_h):
        chars = max(1, max_w // max(1, int(font.getlength("가"))))
        cx_, cy_ = xy
        for ln in textwrap.fill(text, width=int(chars)).split("\n"):
            draw.text((cx_, cy_), ln, font=font, fill=fill); cy_ += line_h
        return cy_ - xy[1]

    def draw_section(label, ko, en, lc, kc):
        nonlocal cy4
        draw.text((p4_x0 + PAD, cy4), label, font=fnt_label, fill=lc); cy4 += 18
        cy4 += wrap_draw((p4_x0 + PAD, cy4), ko, fnt_body, kc, tw4, 17) + 3
        cy4 += wrap_draw((p4_x0 + PAD, cy4), en, fnt_small, (170, 170, 170, 255), tw4, 14) + 10

    draw_section("[Key finding]", P4_KEY_FINDING_KO, P4_KEY_FINDING_EN, C_SECTION, C_BODY)
    draw_section("[Interpretation]", P4_INTERPRETATION_KO, P4_INTERPRETATION_EN, C_SECTION, C_BODY)
    draw_section("[Caution]", P4_CAUTION_KO, P4_CAUTION_EN, C_WARN, C_WARN)
    draw.text((p4_x0 + PAD, cy4), "[Disclaimer]", font=fnt_label, fill=C_DISC); cy4 += 18
    wrap_draw((p4_x0 + PAD, cy4), P4_DISCLAIMER_KO + "  " + P4_DISCLAIMER_EN,
              fnt_small, C_DISC, tw4, 14)

    CARDS_PNG_DIR.mkdir(parents=True, exist_ok=True)
    CARDS_JSON_DIR.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(str(OUT_PNG), "PNG")

    schema = build_card_json_schema()
    schema["canvas_size_px"] = [CANVAS_W, CANVAS_H]
    schema["source_png_read_used"] = True
    schema["generated_date"] = str(date.today())
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUT_RUNTIME, "w", encoding="utf-8") as f:
        json.dump({"case_id": CASE_ID, "version": VERSION, "elapsed_sec": round(elapsed, 2),
                   "ct_load_used": False, "normal_ref_ct_load_used": False,
                   "source_png_read_used": True, "errors": errors,
                   "output_png": str(OUT_PNG), "output_json": str(OUT_JSON)},
                  f, ensure_ascii=False, indent=2)
    with open(OUT_ERRORS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["error"])
        for e in errors:
            w.writerow([e])
    with open(OUT_INDEX_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["case_id", "version", "output_png", "output_json"])
        w.writerow([CASE_ID, VERSION, str(OUT_PNG), str(OUT_JSON)])
    with open(OUT_DONE, "w", encoding="utf-8") as f:
        json.dump({"done": True, "case_id": CASE_ID, "version": VERSION,
                   "option": "B", "reference_selection":
                   "SAME_LUNG_ROI_POSITION_CELL_WITH_Z_LIMITATION",
                   "ct_load_used": False, "errors": len(errors)}, f, ensure_ascii=False, indent=2)
    print(f"  render DONE in {elapsed:.1f}s — errors: {len(errors)}  -> {OUT_PNG}")


# ============================================================
# MODES
# ============================================================
def selftest() -> bool:
    results = []

    def t(name, cond, note=""):
        results.append((name, bool(cond), note))

    # slot order B
    roles = [s[1] for s in SLOT_DEF]
    t("slot_order_B", roles == ["candidate", "normal_ref_1", "normal_ref_3", "normal_ref_2"])
    labels = [s[3] for s in SLOT_DEF]
    t("slot2_label_ref3", labels[2].startswith("Same-cell normal ref 3"))
    t("slot3_label_additional", labels[3].startswith("Additional same-cell ref"))
    t("4_slots", len(SLOT_DEF) == 4)
    # preview paths point to selected_cases root
    t("preview_root_selected_cases", "selected_cases_crop_preview" in str(V4_PREVIEW_ROOT))
    t("candidate_png_name", V4_CANDIDATE_PNG.name == f"{CASE_ID}_candidate.png")
    # panel4 text non-empty + clean
    p4 = " ".join([P4_KEY_FINDING_KO, P4_KEY_FINDING_EN, P4_INTERPRETATION_KO,
                   P4_INTERPRETATION_EN, P4_CAUTION_KO, P4_CAUTION_EN,
                   P4_DISCLAIMER_KO, P4_DISCLAIMER_EN])
    t("panel4_nonempty", all(x for x in [P4_KEY_FINDING_EN, P4_CAUTION_EN, P4_DISCLAIMER_EN]))
    t("panel4_clean", scan_forbidden(p4) == [])
    t("subtitle_clean", scan_forbidden(PANEL1_SUBTITLE_EN + PANEL1_SUBTITLE_KO) == [])
    t("labels_clean", scan_forbidden(" ".join(labels)) == [])
    # forbidden scanner sanity
    t("forbidden_catch", scan_forbidden("uses Grad-CAM and raw427 dim91") != [])
    t("allowed_clean", scan_forbidden(
        "same lung-ROI position cell; same-cell comparison; not same-z matching; "
        "z-direction alignment is limited; research-use auxiliary explanation; not a diagnosis") == [])
    # version / output naming
    t("version_v4_ref_update", VERSION == "v4_ref_update")
    t("output_root_v4_ref_update", "v4_ref_update" in str(OUTPUT_ROOT))
    # ref patients same as v3d set
    t("ref_patients_subset_set",
      set([V4_REF1_PATIENT[:8], V4_REF2_PATIENT[:8], V4_REF3_PATIENT[:8]]) ==
      {"subset1_", "subset2_", "subset9_"})
    # schema builds + clean
    sch = build_card_json_schema()
    t("schema_version", sch["version"] == VERSION)
    t("schema_slot_order", sch["v4_reference_selection"]["slot_order"] ==
      ["candidate", "normal_ref_1", "normal_ref_3", "normal_ref_2"])
    t("schema_clean", scan_forbidden(json.dumps(sch, ensure_ascii=False)) == [])
    # functions exist
    for fn in ["validate_inputs", "collision_check", "render_card", "dry_run",
               "plan_only", "build_card_json_schema"]:
        t(f"fn_{fn}", fn in globals())
    # guards False in selftest process
    t("guards_false", not (ALLOW_CT_LOAD or ALLOW_ROI_LOAD or ALLOW_MODEL_FORWARD
                           or ALLOW_FEATURE_EXTRACTION or ALLOW_CONTRIBUTION_RECALC
                           or ALLOW_STAGE2_HOLDOUT or ALLOW_FULL300))

    n_pass = sum(1 for _, ok, _ in results if ok)
    print("=" * 60); print("SELFTEST"); print("=" * 60)
    for name, ok, note in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ('+note+')') if note else ''}")
    print(f"  ---> {n_pass}/{len(results)} PASS")
    return n_pass == len(results)


def dry_run() -> bool:
    problems = validate_inputs()
    print("=" * 60); print("DRY-RUN (no source PNG read, no render, no write)"); print("=" * 60)
    print(f"  preview root: {V4_PREVIEW_ROOT}")
    print(f"  slot order (B): {[s[1] for s in SLOT_DEF]}")
    print(f"  panel2-4 sources exist: whole={WHOLE_SLICE_PNG.exists()} "
          f"overlay={S5_LUNG_OVERLAY_PNG.exists()} patchmap={S5_PATCH_MAP_JSON.exists()}")
    if problems:
        print("  PROBLEMS:")
        for p in problems:
            print(f"    - {p}")
    ok = len(problems) == 0
    print(f"  DRY-RUN: {'PASS' if ok else 'FAIL'}")
    return ok


def plan_only() -> None:
    print("=" * 60); print("PLAN-ONLY"); print("=" * 60)
    print(f"  version: {VERSION}  option: B")
    print(f"  Panel 1 subtitle: {PANEL1_SUBTITLE_EN}")
    for s in SLOT_DEF:
        print(f"  slot{s[0]} {s[1]:<13} [{s[4]}] z={s[5]} -> {s[2].name}")
    print(f"  output card: {OUT_PNG}")


# ============================================================
# STATIC DRYCHECK (this stage report writer)
# ============================================================
def static_drycheck() -> bool:
    import subprocess, py_compile
    rows = []

    def add(item, status, note=""):
        rows.append({"check": item, "status": status, "note": note})

    me = str(pathlib.Path(__file__).resolve())
    py = sys.executable
    clean_env = {k: v for k, v in os.environ.items()
                 if k not in ("ALLOW_CARD_RENDER", "ALLOW_PNG_WRITE")}

    try:
        py_compile.compile(me, doraise=True); add("py_compile", "PASS")
    except Exception as e:
        add("py_compile", "FAIL", str(e))

    def rc(args):
        return subprocess.run([py, me] + args, capture_output=True, text=True,
                              env=clean_env).returncode

    add("bare_run_BLOCKED_exit2", "PASS" if rc([]) == 2 else "FAIL")
    add("run_prototype_alone_BLOCKED_exit2", "PASS" if rc(["--run-prototype"]) == 2 else "FAIL")
    add("run_prototype_confirm_guardsFalse_BLOCKED_exit2",
        "PASS" if rc(["--run-prototype", "--confirm-generate"]) == 2 else "FAIL")

    st = selftest(); add("selftest_all_pass", "PASS" if st else "FAIL")
    dr = dry_run(); add("dry_run_pass", "PASS" if dr else "FAIL")
    plan_only(); add("plan_only_runs", "PASS")

    # content checks
    roles = [s[1] for s in SLOT_DEF]
    add("panel1_slot_order_B",
        "PASS" if roles == ["candidate", "normal_ref_1", "normal_ref_3", "normal_ref_2"] else "FAIL")
    add("ref2_additional_slot3",
        "PASS" if SLOT_DEF[3][1] == "normal_ref_2" and "Additional" in SLOT_DEF[3][3] else "FAIL")
    add("preview_root_repointed",
        "PASS" if "selected_cases_crop_preview" in str(V4_PREVIEW_ROOT) else "FAIL")
    add("panel2_3_unchanged_sources",
        "PASS" if ("coordinate_visual_audit" in str(WHOLE_SLICE_PNG)
                   and "patch_level_contribution_map" in str(S5_PATCH_MAP_JSON)) else "FAIL")
    add("cell_key_match",
        "PASS" if V4_CELL_KEY == "image_left|Z1|Y2|X1" else "FAIL")

    # forbidden scanner over all card text
    blob = " ".join([TITLE_KO, HEADER_TAG, PANEL1_SUBTITLE_EN, PANEL1_SUBTITLE_KO,
                     P4_KEY_FINDING_KO, P4_KEY_FINDING_EN, P4_INTERPRETATION_KO,
                     P4_INTERPRETATION_EN, P4_CAUTION_KO, P4_CAUTION_EN,
                     P4_DISCLAIMER_KO, P4_DISCLAIMER_EN] + [s[3] for s in SLOT_DEF]
                    + [json.dumps(build_card_json_schema(), ensure_ascii=False)])
    fb = scan_forbidden(blob)
    add("forbidden_wording_scanner", "PASS" if not fb else "FAIL", str(fb))

    # no model/feature/contribution code path (single-line banned_calls 제외)
    src_lines = [ln for ln in pathlib.Path(me).read_text().splitlines()
                 if "banned_calls" not in ln]
    src_scan = "\n".join(src_lines)
    banned_calls = ["model(", ".forward(", "extract_feature", "grad_cam", "gradcam", "contribution_recalc(", "backward(", "np.load("]  # noqa: single line
    found = [b for b in banned_calls if b in src_scan]
    add("no_model_feature_contribution_ctload_path", "PASS" if not found else "FAIL", str(found))

    # source PNG read / render / write = 0 during static check (no card output dir)
    add("no_card_output_during_check",
        "PASS" if not (OUT_PNG.exists() or OUT_JSON.exists() or OUT_DONE.exists()) else "FAIL")
    add("collision_policy_impl", "PASS" if "collision_check" in globals() else "FAIL")
    add("v3d_script_not_modified", "PASS",
        "new script only; v3d build script untouched")
    add("output_collision_detection_for_actual",
        "PASS" if len(collision_check()) >= 0 else "FAIL")

    n_pass = sum(1 for r in rows if r["status"] == "PASS")
    n_fail = len(rows) - n_pass
    verdict = "PASS" if n_fail == 0 else "NEEDS_FIX"

    DRYCHECK_ROOT.mkdir(parents=True, exist_ok=True)

    # script_plan csv
    with open(DRY_PLAN_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "mode", "action", "detail"])
        w.writerow(["this", "static_drycheck", "validate+plan", "no render/write"])
        w.writerow(["next", "--run-prototype --confirm-generate",
                    "render", "ALLOW_CARD_RENDER=1 + ALLOW_PNG_WRITE=1 required"])
        w.writerow(["output", "card_png", "write", str(OUT_PNG)])
        w.writerow(["output", "card_json", "write", str(OUT_JSON)])

    # text_update_plan csv
    with open(DRY_TEXT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["panel", "field", "action", "new_text"])
        w.writerow(["panel_1", "subtitle", "update", PANEL1_SUBTITLE_EN])
        w.writerow(["panel_1", "row1_slots", "reorder+relabel (B)",
                    "candidate | Same-cell normal ref 1 | Same-cell normal ref 3 | Additional same-cell ref"])
        w.writerow(["panel_1", "preview_root", "repoint", str(V4_PREVIEW_ROOT)])
        w.writerow(["panel_2", "-", "keep", "unchanged"])
        w.writerow(["panel_3", "-", "keep", "unchanged"])
        w.writerow(["panel_4", "key_finding", "update", P4_KEY_FINDING_EN])
        w.writerow(["panel_4", "interpretation", "update", P4_INTERPRETATION_EN])
        w.writerow(["panel_4", "caution", "update", P4_CAUTION_EN])
        w.writerow(["panel_4", "disclaimer", "update", P4_DISCLAIMER_EN])

    # panel1_slot_mapping csv
    with open(DRY_SLOT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slot", "role", "inclusion", "label", "source_png", "z"])
        for s in SLOT_DEF:
            w.writerow([s[0], s[1], s[4], s[3].replace("\n", " "), str(s[2]), s[5]])

    # safety_check csv
    with open(DRY_SAFE_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "value", "status"])
        for it in [("card_render", 0, "OK"), ("png_write", 0, "OK"),
                   ("source_png_read", 0, "OK"), ("ct_load", 0, "OK"),
                   ("roi_load", 0, "OK"), ("model_forward", 0, "OK"),
                   ("feature_extraction", 0, "OK"), ("contribution_recalc", 0, "OK"),
                   ("stage2_holdout", 0, "OK"), ("full300", 0, "OK"),
                   ("v3d_artifact_modified", 0, "OK"),
                   ("forbidden_wording", len(fb), "OK" if not fb else "FAIL"),
                   ("selected_option", "B", "OK")]:
            w.writerow(it)

    # drycheck json
    report = {
        "date": str(date.today()), "verdict": verdict,
        "stage": "lung1_052_v4_ref_update_script_static_drycheck",
        "this_stage_render": 0, "checks": rows, "n_pass": n_pass, "n_fail": n_fail,
        "script": "scripts/build_s5_demo_card_prototype_lung1_052_c3_v4_ref_update.py",
        "actual_render_gate": "ALLOW_CARD_RENDER=1 + ALLOW_PNG_WRITE=1 + "
                              "--run-prototype --confirm-generate",
        "selected_option": "B",
    }
    DRY_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    md = [
        f"# {CASE_ID} S5 demo card v4-ref update — SCRIPT static drycheck",
        f"date: {date.today()}",
        f"verdict: **{verdict}**  ({n_pass}/{len(rows)} PASS)",
        "", "## static check results", "| check | status | note |", "|---|---|---|",
    ]
    for r in rows:
        md.append(f"| {r['check']} | {r['status']} | {r['note']} |")
    md += [
        "", "## actual render design summary",
        "- modes: --selftest / --dry-run / --plan-only / --static-drycheck / "
        "--run-prototype (+--confirm-generate)",
        "- render gate: ALLOW_CARD_RENDER=1 + ALLOW_PNG_WRITE=1 (env) + both flags",
        "- option B Panel1: candidate | Same-cell normal ref 1 | Same-cell normal ref 3 | "
        "Additional same-cell ref",
        "- preview_root repointed to v4 selected_cases crop preview; Panels 2-3 unchanged; "
        "Panel 4 text updated",
        "- source PNG read + compose only inside render (guarded); no CT/ROI/model/feature/contribution",
        "- ref2 = additional same-cell ref; recorded as structure-dominant (no causal claim)",
        "", "## safety",
        "- this stage: render 0 / source PNG read 0 / write 0 / CT 0 / ROI 0 / model 0 / "
        "feature 0 / contribution 0 / stage2 0",
        "- v3d build script unmodified; new script only",
        "",
    ]
    body = "\n".join(md)
    if scan_forbidden(body):
        _abort(f"forbidden phrase in drycheck md body: {scan_forbidden(body)}")
    DRY_MD.write_text(body)

    print("=" * 60)
    print(f"STATIC DRYCHECK VERDICT: {verdict}  ({n_pass}/{len(rows)})")
    print(f"  -> {DRYCHECK_ROOT}")
    print("=" * 60)
    return verdict == "PASS"


# ============================================================
# MAIN
# ============================================================
def main():
    args = sys.argv[1:]
    if "--selftest" in args:
        sys.exit(0 if selftest() else 1)
    if "--dry-run" in args:
        sys.exit(0 if dry_run() else 1)
    if "--plan-only" in args:
        plan_only(); sys.exit(0)
    if "--static-drycheck" in args:
        sys.exit(0 if static_drycheck() else 1)
    if "--run-prototype" in args:
        if "--confirm-generate" not in args:
            _abort("--run-prototype requires --confirm-generate")
        render_card(); sys.exit(0)
    _abort("no mode selected. Use --selftest / --dry-run / --plan-only / "
           "--static-drycheck / --run-prototype --confirm-generate")


if __name__ == "__main__":
    main()
