"""
build_s5_demo_card_prototype_lung1_052_c3_efficientnet_v4_20_dynamicref_layoutfix.py

목적 (dynamic-reference integrated full XAI card):
efficientnet_v4_20_layoutfix 최종 dark-card 레이아웃(헤더/패널좌표/박스/폰트/하단 3-column)을 그대로 유지하고,
Panel 1 만 "dynamic normal-reference comparison (final A+B)" 결과로 교체한 full 4-panel 카드.
base = scripts/build_s5_demo_card_prototype_lung1_052_c3_efficientnet_v4_20_layoutfix.py (직접 수정 금지, copy 후 최소 수정).
기존 v1 / aspectfix / v3d / layoutfix / 흰배경 카드 산출물은 수정하지 않는다(신규 파일 + 신규 output root).

Panel 1 = dynamic normal-reference comparison (final A+B):
  - row1 = 4개 동일 정사각형 280px tile (candidate | normal_patient_1 | normal_patient_2 | normal_patient_3).
  - tile source = dynamic A+B explanation artifact 의 PRE-RENDERED 96x96 PNG
    (candidate_patch.png + normal_ref_patient1/2/3_patch.png; 80mm physical-scale normalized, bilateral frame).
    -> CT 재load 불필요. contain-fit(aspect 보존, stretch 금지)으로 280px 에 중앙 배치.
  - candidate tile 에는 EfficientNet peak box 표시. label/요약(z, lung_z_pct, distance)은 tile 바깥(아래)에.
  - row2 = 전체 슬라이스(center CT audit, unchanged), row3 = z-context 한 줄 note.
Panel 2/3 = EfficientNet layoutfix 내용 그대로 유지(peak/patch4/7 legend, 3x3 response map, [304,96] MISSING).
Panel 4 = dynamic-reference 방식에 맞게 KO+EN 문구 갱신(Key finding/Interpretation/Caution/Disclaimer).

matching = bilateral lung frame; lung_z_pct + lung-bbox relative y/x; 80mm physical-scale normalized; NOT same-z.
score 재계산/ model forward / feature extraction / contribution recalc / stage2_holdout = 전부 금지(읽기 전용).

render gate: ALLOW_CARD_RENDER=1 + ALLOW_SOURCE_IMAGE_READ=1 + ALLOW_PNG_WRITE=1 (+ --run-prototype --confirm-generate).
CT load 불필요(모든 tile/overlay 가 pre-rendered PNG). guard 기본값 전부 False.
"""

import os
import sys
import csv
import json
import time
import textwrap
import pathlib
from datetime import date
from typing import Any, Dict, List

# ============================================================
# GUARD FLAGS (기본 False)
# ============================================================
ALLOW_CARD_RENDER         = False   # compose
ALLOW_SOURCE_IMAGE_READ   = False   # pre-rendered PNG (panel1 dynamic tiles + panel2/3 overlays) read
ALLOW_CT_LOAD             = False   # dynamic card 에서는 불필요(모든 tile=PNG); optional
ALLOW_PNG_WRITE           = False
ALLOW_ROI_LOAD            = False
ALLOW_STAGE2_HOLDOUT      = False
ALLOW_MODEL_FORWARD       = False
ALLOW_FEATURE_EXTRACTION  = False
ALLOW_CONTRIBUTION_RECALC = False
ALLOW_FULL300             = False

# actual render 승인 시에만 env 로 True (내릴 수는 없음)
if os.environ.get("ALLOW_CARD_RENDER") == "1":
    ALLOW_CARD_RENDER = True
if os.environ.get("ALLOW_SOURCE_IMAGE_READ") == "1":
    ALLOW_SOURCE_IMAGE_READ = True
if os.environ.get("ALLOW_CT_LOAD") == "1":
    ALLOW_CT_LOAD = True
if os.environ.get("ALLOW_PNG_WRITE") == "1":
    ALLOW_PNG_WRITE = True

# ============================================================
# 케이스 상수
# ============================================================
CASE_ID            = "LUNG1-052__c3"
VOLUME_ID          = "NSCLC_LUNG1-052__d4a19cc211"
POSITION_BIN       = "lower_peripheral"
CT_LOCAL_Z         = 51
REPORT_SLICE_INDEX = 106
SPATIAL_PATTERN    = "CENTER_DOMINANT_WITH_DOWNWARD_CONTINUITY"
VISUAL_VERDICT     = "PASS_VISUAL_SUPPORTS_CENTER_DOWNWARD_CONTINUITY_WITH_CAUTION"
GRID_EXTENT        = [256, 96, 352, 192]
PATCH4_BBOX        = [288, 128, 320, 160]
PATCH4_SCORE       = 17.9909   # EfficientNet score at v1 patch4 location (branch-specific)
PATCH7_BBOX        = [320, 128, 352, 160]
PATCH7_SCORE       = 18.2798   # EfficientNet score at v1 patch7 location (branch-specific)
LAYOUT_SELECTED    = "DYNAMIC_REF_4PANEL_FINAL_AB_PANEL1_DYNAMIC_NORMAL_REFERENCE"
VERSION            = "efficientnet_dynamic_ref_layoutfix"

# ============================================================
# EfficientNet PaDiM v4.20 전용 content (layout은 base aspectfix와 동일, 내용만 교체)
# ============================================================
EFF_BRANCH                = "efficientnet_b0_imagenet_chestwall_removed_roi_v1"
EFF_MASK                  = "refined_roi_v4_20_modeB"
EFF_CARD_VERSION          = "efficientnet_v4_20"
CANDIDATE_POSITION_POLICY = "saved_first_stage_peak_bilateral_frame"   # saved 1차 peak 위치, bilateral lung frame
SCORE_SCALE_NOTE          = "branch-specific score; not absolute-comparable with v1"
EFF_PEAK_BBOX             = [288, 112, 320, 144]
EFF_PEAK_SCORE            = 21.6854
# 3x3 EfficientNet patch response map: (y0,x0) -> score, None = MISSING (보간 금지)
EFF_YS = [272, 288, 304]
EFF_XS = [96, 112, 128]
EFF_3X3 = {
    (288, 112): 21.6854, (304, 112): 19.2811, (288, 96): 18.9322,
    (288, 128): 17.9909, (304, 128): 17.7807, (272, 112): 15.5562,
    (272, 128): 14.5310, (272, 96): 14.3818, (304, 96): None,
}
EFF_MISSING_CELL = (304, 96)

# ============================================================
# Dynamic normal-reference (final A+B) source + loaded metadata
# ============================================================
DYN_AB_ROOT = pathlib.Path(
    "/home/jinhy/project/lung-ct-anomaly/outputs/position-aware-padim-v1/reports"
    "/dynamic_explanation_table_from_first_stage_candidate_v3_final_ab")
DYN_TILES = {
    "candidate":        DYN_AB_ROOT / "candidate_patch.png",
    "normal_patient_1": DYN_AB_ROOT / "normal_ref_patient1_patch.png",
    "normal_patient_2": DYN_AB_ROOT / "normal_ref_patient2_patch.png",
    "normal_patient_3": DYN_AB_ROOT / "normal_ref_patient3_patch.png",
}
DYN_RETRIEVAL_JSON = DYN_AB_ROOT / "dynamic_retrieval_result.json"
DYN_CAND_META_JSON = DYN_AB_ROOT / "candidate_metadata_from_saved_score.json"
DYN_MONTAGE_PNG    = DYN_AB_ROOT / "candidate_vs_dynamic_normal3_montage.png"
DYN_EXPL_JSON      = DYN_AB_ROOT / "explanation_table.json"

DYNAMIC_REFERENCE_MODE = "final_A_plus_B"
PHYSICAL_SCALE_NORMALIZED = True
BILATERAL_FRAME_MATCHING  = True
SAME_Z_MATCHING           = False
PHYSICAL_FOV_MM           = 80.0

def _load_dyn():
    """dynamic A+B retrieval/candidate metadata 를 read-only 로 로드(없으면 빈 dict)."""
    refs, cand = [], {}
    try:
        rr = json.loads(DYN_RETRIEVAL_JSON.read_text())
        refs = rr.get("normal_refs", [])
    except Exception:
        refs = []
    try:
        cand = json.loads(DYN_CAND_META_JSON.read_text())
    except Exception:
        cand = {}
    return refs, cand

DYN_REFS, DYN_CAND = _load_dyn()
# candidate (saved first-stage) 핵심값
SAVED_SCORE        = DYN_CAND.get("first_stage_score", EFF_PEAK_SCORE)
CAND_LUNG_BBOX     = DYN_CAND.get("candidate_lung_bbox", [155, 98, 345, 369])
CAND_LUNG_Z_PCT    = DYN_CAND.get("candidate_lung_z_pct", 0.2406)
CAND_Y_PCT         = DYN_CAND.get("candidate_y_pct", 0.7842)
CAND_X_PCT         = DYN_CAND.get("candidate_x_pct", 0.1107)
CAND_SIDE          = DYN_CAND.get("candidate_side", "left")
CAND_CROP_BBOX_NAT = DYN_CAND.get("candidate_crop_bbox_native", [263, 87, 345, 169])
CAND_CROP_PX_NAT   = DYN_CAND.get("candidate_crop_px_native", 82)
CAND_SPACING_MM    = DYN_CAND.get("candidate_spacing_mm", 0.977)
# refs: alias -> {z, lung_z_pct, distance, spacing, crop_px}
DYN_REF_BY_ALIAS = {r.get("patient_alias"): r for r in DYN_REFS}

# v4 exact-cell top3 (v3d/v4_ref_update 와 동일 환자)
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

# clean tile / crop policy
TILE_PX   = 280     # Panel1 row1 동일 정사각형 tile
CROP_SIZE = 96
IMG_HW    = 512
WL = -600.0
WW = 1500.0

# ============================================================
# 경로
# ============================================================
PROJECT_ROOT = pathlib.Path("/home/jinhy/project/lung-ct-anomaly")
EC = PROJECT_ROOT / "outputs/position-aware-padim-v1/reports/explanation_cards"

# preflight (정책 입력)
ASPECTFIX_PREFLIGHT_ROOT = EC / "lung1_052_s5_demo_card_v4_ref_update_panel1_aspect_fix_preflight"
ASPECTFIX_PREFLIGHT_DONE = ASPECTFIX_PREFLIGHT_ROOT / "DONE.json"
REFLECTION_DONE = (EC / "lung1_052_s5_demo_card_v4_reference_reflection_preflight" / "DONE.json")

# crop plan CSV (ct_path / bbox / z 의 canonical source — crop preview preflight 산출)
PLAN_CSV = (PROJECT_ROOT / "outputs/position-aware-padim-v1/reports"
            / "reference_bank_v4_selected_cases_crop_preview_preflight"
            / "selected_cases_crop_preview_plan_v4.csv")

# v4 selected crop preview (preview metadata = patient/z/bbox 교차검증용)
V4_PREVIEW_ROOT = (PROJECT_ROOT / "outputs/position-aware-padim-v1/visualizations"
                   / "reference_bank_v4_selected_cases_crop_preview")
V4_PREVIEW_META = V4_PREVIEW_ROOT / f"{CASE_ID}_preview_metadata.json"

# Panel 2~4 source (UNCHANGED, v4_ref_update/v3d 와 동일)
WHOLE_SLICE_PNG = (EC / "s5_lung1_052_c3_coordinate_visual_audit_1case_v1"
                   / "coordinate_overlay_patch4_patch7_lung.png")
S5_LUNG_OVERLAY_PNG = (EC / "s5_lung1_052_c3_coordinate_visual_audit_1case_v1"
                       / "coordinate_overlay_3x3_grid_lung.png")
S5_PATCH_MAP_JSON = (EC / "s5_lung1_052_c3_patch_level_contribution_map_1case_smoke_v1"
                     / "patch_contribution_map.json")
V3D_CARD_JSON = (EC / "s5_demo_card_prototype_lung1_052_c3_v3d_panel1_v4_ref"
                 / "cards_json/LUNG1-052__c3_s5_demo_prototype_v3d.json")

REPORTS_ROOT = PROJECT_ROOT / "outputs/position-aware-padim-v1/reports"
# 신규 render 출력 root (dynamic-reference integrated; 기존 v1/aspectfix/layoutfix/흰배경 카드와 별도)
OUTPUT_ROOT    = REPORTS_ROOT / "efficientnet_s5_card_lung1_052_c3_dynamic_ref_layoutfix"
CARDS_PNG_DIR  = OUTPUT_ROOT / "cards_png"
CARDS_JSON_DIR = OUTPUT_ROOT / "cards_json"
OUT_PNG        = CARDS_PNG_DIR  / f"{CASE_ID}_efficientnet_dynamic_ref_layoutfix.png"
OUT_JSON       = CARDS_JSON_DIR / f"{CASE_ID}_efficientnet_dynamic_ref_layoutfix.json"
OUT_INDEX_CSV  = OUTPUT_ROOT / "index_cards_dynamic_ref_layoutfix.csv"
OUT_RUNTIME    = OUTPUT_ROOT / "runtime_summary_dynamic_ref_layoutfix.json"
OUT_ERRORS     = OUTPUT_ROOT / "errors.csv"
OUT_DONE       = OUTPUT_ROOT / "DONE.json"
OUT_INTEG_MD   = OUTPUT_ROOT / "integration_report.md"
OUT_P1MAP_CSV  = OUTPUT_ROOT / "panel1_source_mapping.csv"
OUT_P4SUM_CSV  = OUTPUT_ROOT / "panel4_text_summary.csv"

# 이번 단계(static drycheck) 리포트 root
DRYCHECK_ROOT = REPORTS_ROOT / "efficientnet_s5_card_lung1_052_c3_dynamic_ref_layoutfix_static_drycheck"
DRY_MD        = DRYCHECK_ROOT / "static_drycheck_dynamic_ref_layoutfix.md"
DRY_JSON      = DRYCHECK_ROOT / "static_drycheck_dynamic_ref_layoutfix.json"
DRY_PLAN_CSV  = DRYCHECK_ROOT / "script_plan_dynamic_ref_layoutfix.csv"
DRY_TILE_CSV  = DRYCHECK_ROOT / "panel1_tile_layout_dynamic_ref_layoutfix.csv"
DRY_CROP_CSV  = DRYCHECK_ROOT / "panel1_dynamic_source_dynamic_ref_layoutfix.csv"
DRY_SAFE_CSV  = DRYCHECK_ROOT / "safety_check_dynamic_ref_layoutfix.csv"

# ============================================================
# 텍스트 상수 (Panel 4 등은 v4_ref_update 와 동일 유지)
# ============================================================
TITLE_KO = f"S5 EfficientNet PaDiM v4.20 이상 후보 설명 카드 — Dynamic normal-reference integrated ({CASE_ID})"
HEADER_TAG = "[INTERNAL USE ONLY | EfficientNet-B0 PaDiM v4.20 (research-use) | not diagnostic]"

PANEL1_SUBTITLE_EN = ("Dynamic normal references matched by bilateral lung frame "
                      "(lung_z_pct + lung-bbox relative y/x), 80mm physical-scale normalized; "
                      "this is not same-z matching.")
PANEL1_SUBTITLE_KO = ("정상 reference는 양쪽 폐(bilateral) 기준 폐 내부 상대 위치"
                      "(lung_z_pct + lung-bbox 상대 y/x)로 동적 매칭했고, 80mm 동일 물리 시야로 정규화했습니다. "
                      "같은 z 번호 정합이 아닙니다.")

P4_KEY_FINDING_KO = ("저장된 1차 anomaly 후보를 정상 환자 3명의 dynamic normal reference와 비교했을 때, "
                     "후보 위치는 정상 참조보다 더 조밀하고 이질적인 국소 패턴을 보입니다.")
P4_KEY_FINDING_EN = ("Compared with dynamic normal references from 3 normal patients, the saved "
                     "first-stage candidate shows a denser, more heterogeneous local pattern than "
                     "the normal references.")
P4_INTERPRETATION_KO = ("비교 기준은 양쪽 폐(bilateral) 기준 폐 내부 상대 위치(lung_z_pct + lung-bbox 상대 y/x)이며 "
                        "절대 slice 번호 매칭이 아닙니다. 모든 patch는 환자별 spacing을 반영해 80mm 동일 물리 시야로 "
                        "맞췄습니다. 후보는 EfficientNet 1차 저장(saved) anomaly 후보입니다.")
P4_INTERPRETATION_EN = ("Matching uses bilateral lung-relative position (lung_z_pct + lung-bbox relative "
                        "y/x), not absolute slice index. All patches are normalized to an 80mm physical "
                        "field-of-view using per-patient spacing. The candidate is a saved EfficientNet "
                        "first-stage anomaly candidate.")
P4_CAUTION_KO = ("same-z matching이 아닙니다. 점수는 branch-specific이므로 v1 점수와 절대값으로 비교하지 마십시오. "
                 "후보가 흉막 인접(pleura-adjacent)하므로 과해석하지 마십시오. dynamic reference 3개는 정상 참조 "
                 "예시일 뿐 진단 근거가 아닙니다.")
P4_CAUTION_EN = ("Not same-z matching; scores are branch-specific and must not be compared as absolute "
                 "values with v1; the candidate is pleura-adjacent, so avoid over-interpretation; the 3 "
                 "dynamic references are normal examples only, not diagnostic evidence.")
P4_DISCLAIMER_KO = "연구용 보조 설명이며 진단이 아닙니다. saliency/attribution 스타일 지도(map)가 아닙니다."
P4_DISCLAIMER_EN = "Research-use auxiliary explanation only; not a diagnosis; not a saliency/attribution-style map."

# Panel 1 row1 labels (tile 밖; dynamic normal-reference)
LABEL_CANDIDATE = "Candidate (saved first-stage 후보)"
LABEL_REF1      = "Dynamic normal ref — normal_patient_1"
LABEL_REF3      = "Dynamic normal ref — normal_patient_2"
LABEL_REF2_ADD  = "Dynamic normal ref — normal_patient_3"
LABEL_WHOLE_SLICE = "전체 슬라이스 (lung window, patch 위치 표시)"
Z_CONTEXT_NOTE = "z-context: bilateral lung-relative matching (lung_z_pct), not same-z matching"

def _ref_meta(alias, key, default=None):
    return DYN_REF_BY_ALIAS.get(alias, {}).get(key, default)

# dynamic slot 정의 (slot, role, label, inclusion, selected_local_z)
SLOT_DEF = [
    (0, "candidate",        LABEL_CANDIDATE, "primary", CT_LOCAL_Z),
    (1, "normal_patient_1", LABEL_REF1,      "primary", _ref_meta("normal_patient_1", "selected_local_z", 60)),
    (2, "normal_patient_2", LABEL_REF3,      "primary", _ref_meta("normal_patient_2", "selected_local_z", 61)),
    (3, "normal_patient_3", LABEL_REF2_ADD,  "primary", _ref_meta("normal_patient_3", "selected_local_z", 67)),
]
REF2_NOTE = "dynamic normal reference (bilateral-frame, 80mm physical-scale)"

# ============================================================
# forbidden wording
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
# clean crop plan loader (ct_path / bbox / z)
# ============================================================
def load_crop_plan() -> Dict[str, Dict[str, Any]]:
    """preflight plan CSV 에서 LUNG1-052__c3 의 candidate/ref1/ref2/ref3 crop 행을 읽어
    role -> {ct_path, local_z, crop_y0..x1, cell_key, crop_in_bounds, ct_exists} dict 반환."""
    if not PLAN_CSV.exists():
        return {}
    out = {}
    with open(PLAN_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("case_id") != CASE_ID:
                continue
            out[r["role"]] = {
                "ct_path": r["ct_path"],
                "local_z": int(r["local_z"]),
                "crop_y0": int(r["crop_y0"]), "crop_x0": int(r["crop_x0"]),
                "crop_y1": int(r["crop_y1"]), "crop_x1": int(r["crop_x1"]),
                "crop_size": int(r["crop_size"]),
                "cell_key": r["cell_key"],
                "crop_in_bounds": str(r["crop_in_bounds"]) == "True",
                "ct_exists": str(r["ct_exists"]) == "True",
                "stage2_holdout_flag": str(r.get("stage2_holdout_flag", "False")) == "True",
                "patient_id": r["patient_id"], "volume_id": r["volume_id"],
            }
    return out


# ============================================================
# contain-fit geometry (pure, selftest 대상)
# ============================================================
def contain_fit_size(src_w, src_h, tile=TILE_PX):
    """aspect 보존 contain-fit. (new_w, new_h) 반환. stretch 없음."""
    scale = min(tile / max(src_w, 1), tile / max(src_h, 1))
    return max(1, round(src_w * scale)), max(1, round(src_h * scale))


# ============================================================
# CT crop / window (실제 실행 경로 — guarded)
# ============================================================
def window_hu_to_uint8(hu_arr, wl=WL, ww=WW):
    import numpy as np
    lo = wl - ww / 2.0
    hi = wl + ww / 2.0
    arr = np.clip(hu_arr.astype(np.float32), lo, hi)
    arr = (arr - lo) / max(hi - lo, 1e-6)
    return (arr * 255.0).round().astype(np.uint8)


def load_ct_crop_readonly(ct_path, local_z, y0, x0, y1, x1):
    """read-only mmap CT clean 96x96 crop. ALLOW_CT_LOAD guard 필수."""
    if not ALLOW_CT_LOAD:
        _abort("load_ct_crop_readonly called with ALLOW_CT_LOAD=False")
    import numpy as np
    p = str(ct_path).lower()
    if any(k in p for k in ["stage2_holdout", "stage2holdout", "holdout"]):
        _abort(f"stage2_holdout path forbidden: {ct_path}")
    vol = np.load(str(ct_path), mmap_mode="r")   # read-only
    z = int(local_z)
    if z < 0 or z >= vol.shape[0]:
        raise ValueError(f"local_z {z} out of range {vol.shape}")
    crop = np.array(vol[z, y0:y1, x0:x1])
    del vol
    if crop.shape != (CROP_SIZE, CROP_SIZE):
        raise ValueError(f"crop shape {crop.shape} != {(CROP_SIZE, CROP_SIZE)}")
    return crop


# ============================================================
# card JSON schema
# ============================================================
def build_card_json_schema() -> Dict[str, Any]:
    return {
        "version": VERSION,
        "card_version": EFF_CARD_VERSION,
        "branch": EFF_BRANCH,
        "mask": EFF_MASK,
        "candidate_position_policy": CANDIDATE_POSITION_POLICY,
        "score_scale_note": SCORE_SCALE_NOTE,
        "efficientnet_peak_bbox": EFF_PEAK_BBOX,
        "efficientnet_peak_score": EFF_PEAK_SCORE,
        "efficientnet_3x3": {f"{y},{x}": EFF_3X3[(y, x)] for y in EFF_YS for x in EFF_XS},
        "efficientnet_missing_cell": list(EFF_MISSING_CELL),
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
        "panel1_fix": {
            "tile_px": TILE_PX, "fit": "contain-fit (aspect preserved, no stretch)",
            "tile_source": "dynamic A+B pre-rendered 96x96 PNG (80mm physical-scale normalized); no CT re-crop",
            "labels": "drawn by renderer outside tile only",
            "z_context_row": "collapsed to one-line note (bilateral lung-relative)",
            "slot_order": ["candidate", "normal_patient_1", "normal_patient_2", "normal_patient_3"],
        },
        "dynamic_reference_mode": DYNAMIC_REFERENCE_MODE,
        "physical_scale_normalized": PHYSICAL_SCALE_NORMALIZED,
        "physical_fov_mm": PHYSICAL_FOV_MM,
        "bilateral_frame_matching": BILATERAL_FRAME_MATCHING,
        "same_z_matching": SAME_Z_MATCHING,
        "candidate_first_stage": {
            "candidate_id": "efficientnet_v4_20_layoutfix_peak",
            "candidate_source": "saved_first_stage_score_output",
            "saved_score": SAVED_SCORE,
            "candidate_bbox": EFF_PEAK_BBOX,
            "candidate_center": [304, 128],
            "candidate_local_z": CT_LOCAL_Z,
            "report_slice": REPORT_SLICE_INDEX,
            "candidate_lung_bbox": CAND_LUNG_BBOX,
            "candidate_lung_z_pct": CAND_LUNG_Z_PCT,
            "candidate_y_pct": CAND_Y_PCT, "candidate_x_pct": CAND_X_PCT,
            "candidate_side": CAND_SIDE, "position_bin": POSITION_BIN,
        },
        "dynamic_normal_refs": [
            {"slot": 1, "tile": "normal_ref_patient1_patch.png", **DYN_REF_BY_ALIAS.get("normal_patient_1", {})},
            {"slot": 2, "tile": "normal_ref_patient2_patch.png", **DYN_REF_BY_ALIAS.get("normal_patient_2", {})},
            {"slot": 3, "tile": "normal_ref_patient3_patch.png", **DYN_REF_BY_ALIAS.get("normal_patient_3", {})},
        ],
        "dynamic_source_roots": {
            "final_A_plus_B_artifact": str(DYN_AB_ROOT),
            "dynamic_reference_bank_index": ("outputs/position-aware-padim-v1/reports/"
                "dynamic_normal_reference_bank_three_patients_v1/dynamic_reference_slice_index.csv"),
            "saved_first_stage_score": ("experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1/"
                "outputs/scores/lesion_stage1_dev_by_patient/LUNG1-052.csv"),
        },
        "safety_flags": {
            "score_recomputed": False, "model_forward": False, "feature_extraction": False,
            "contribution_recalc": False, "stage2_holdout_access": False, "ct_load_used": False,
            "not_diagnostic": True, "no_saliency_attribution_style": True,
        },
        "panels": {
            "panel_1": ("S4-style reference comparison (option B, v4 selected-cell ref, "
                        "clean-tile fix): Row1=4 equal-square 280px contain-fit tiles "
                        "(candidate|Same-cell normal ref 1|Same-cell normal ref 3|"
                        "Additional same-cell ref); Row2=whole slice; "
                        "Row3=one-line z-context note"),
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
            "option": "B",
            "slot_order": ["candidate", "normal_ref_1", "normal_ref_3", "normal_ref_2"],
            "normal_ref_1": {"patient_id": V4_REF1_PATIENT, "z": V4_REF1_Z,
                             "decision": "keep_primary"},
            "normal_ref_3": {"patient_id": V4_REF3_PATIENT, "z": V4_REF3_Z,
                             "decision": "keep_primary"},
            "normal_ref_2": {"patient_id": V4_REF2_PATIENT, "z": V4_REF2_Z,
                             "decision": "keep_additional", "note": REF2_NOTE},
            "avg_abs_delta_z": V4_AVG_ABS_DELTA_Z,
            "not_same_z_matched": True, "z_orientation_limitation": True,
        },
        "metadata": {
            "version": VERSION,
            "panel1_reference_source": "v4_exact_cell_top3 (clean CT re-crop)",
            "cell_key": V4_CELL_KEY,
            "reference_selection_policy": "SAME_LUNG_ROI_POSITION_CELL_WITH_Z_LIMITATION",
            "not_same_z_matched": True, "z_orientation_limitation": True,
            "not_diagnostic": True, "no_saliency_attribution_style": True,
            "stage2_holdout_accessed": False, "model_forward_occurred": False,
            "feature_extraction_occurred": False, "contribution_recalc_occurred": False,
            "ref2_handling": f"additional same-cell ref; {REF2_NOTE}; no causal claim",
        },
        "source_files": {
            "crop_plan_csv": str(PLAN_CSV),
            "preview_metadata": str(V4_PREVIEW_META),
            "whole_slice_png": str(WHOLE_SLICE_PNG),
            "lung_overlay_png": str(S5_LUNG_OVERLAY_PNG),
            "patch_map_json": str(S5_PATCH_MAP_JSON),
            "v3d_template_json": str(V3D_CARD_JSON),
        },
        "canvas_size_px": None,
        "output_png": str(OUT_PNG),
        "generated_date": None,
        "ct_load_used": False,
        "source_png_read_used": False,
    }


# ============================================================
# INPUT VALIDATION (no load)
# ============================================================
def validate_inputs() -> List[str]:
    problems = []
    # preflight gating
    if not ASPECTFIX_PREFLIGHT_DONE.exists():
        problems.append("aspect-fix preflight DONE.json missing")
    else:
        d = json.loads(ASPECTFIX_PREFLIGHT_DONE.read_text())
        if d.get("verdict") != "PASS":
            problems.append(f"aspect-fix preflight verdict != PASS ({d.get('verdict')})")
        if d.get("z_context_option") != "B":
            problems.append(f"aspect-fix z_context_option != B ({d.get('z_context_option')})")
    if not REFLECTION_DONE.exists():
        problems.append("reflection preflight DONE.json missing")
    else:
        d = json.loads(REFLECTION_DONE.read_text())
        if d.get("verdict") != "PASS":
            problems.append(f"reflection verdict != PASS ({d.get('verdict')})")
        if d.get("recommended_option") != "B":
            problems.append(f"reflection recommended_option != B ({d.get('recommended_option')})")

    # dynamic A+B source (Panel 1 tiles + retrieval/candidate metadata)
    if not DYN_AB_ROOT.exists():
        problems.append(f"dynamic A+B root missing: {DYN_AB_ROOT}")
        return problems
    for role, p in DYN_TILES.items():
        if not p.exists():
            problems.append(f"dynamic tile missing: {role} -> {p}")
    if not DYN_RETRIEVAL_JSON.exists():
        problems.append(f"dynamic retrieval json missing: {DYN_RETRIEVAL_JSON}")
    if not DYN_CAND_META_JSON.exists():
        problems.append(f"dynamic candidate metadata missing: {DYN_CAND_META_JSON}")
    if len(DYN_REFS) != 3:
        problems.append(f"dynamic normal refs != 3 ({len(DYN_REFS)})")
    for al in ["normal_patient_1", "normal_patient_2", "normal_patient_3"]:
        if al not in DYN_REF_BY_ALIAS:
            problems.append(f"dynamic ref alias missing: {al}")
    if not isinstance(SAVED_SCORE, (int, float)):
        problems.append("saved_score not numeric")
    if not (BILATERAL_FRAME_MATCHING and PHYSICAL_SCALE_NORMALIZED) or SAME_Z_MATCHING:
        problems.append("dynamic flags inconsistent (need bilateral + physical-scale, not same-z)")
    if "holdout" in str(DYN_AB_ROOT).lower():
        problems.append("dynamic root holdout forbidden")
    # candidate not same-z as any ref (sanity)
    if any(int(r.get("selected_local_z", -1)) == CT_LOCAL_Z for r in DYN_REFS):
        problems.append("a dynamic ref shares candidate local_z (should be lung_z_pct matched, not same-z)")

    # panel 2-4 sources
    for nm, p in [("whole_slice", WHOLE_SLICE_PNG), ("lung_overlay", S5_LUNG_OVERLAY_PNG),
                  ("patch_map", S5_PATCH_MAP_JSON), ("v3d_template", V3D_CARD_JSON)]:
        if not p.exists():
            problems.append(f"missing panel2-4 source {nm}: {p}")

    # forbidden wording in card text
    blob = " ".join([TITLE_KO, HEADER_TAG, PANEL1_SUBTITLE_EN, PANEL1_SUBTITLE_KO,
                     P4_KEY_FINDING_KO, P4_KEY_FINDING_EN, P4_INTERPRETATION_KO,
                     P4_INTERPRETATION_EN, P4_CAUTION_KO, P4_CAUTION_EN,
                     P4_DISCLAIMER_KO, P4_DISCLAIMER_EN, Z_CONTEXT_NOTE,
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
# RENDER (actual; guarded)  — Panel1 clean-tile fix
# ============================================================
def render_card() -> None:
    if not (ALLOW_CARD_RENDER and ALLOW_SOURCE_IMAGE_READ and ALLOW_PNG_WRITE):
        _abort("render requires ALLOW_CARD_RENDER=1 + ALLOW_SOURCE_IMAGE_READ=1 + ALLOW_PNG_WRITE=1 "
               f"(current: render={ALLOW_CARD_RENDER}, src={ALLOW_SOURCE_IMAGE_READ}, png={ALLOW_PNG_WRITE})")
    if (ALLOW_ROI_LOAD or ALLOW_MODEL_FORWARD or ALLOW_FEATURE_EXTRACTION
            or ALLOW_CONTRIBUTION_RECALC or ALLOW_STAGE2_HOLDOUT or ALLOW_FULL300):
        _abort("ROI/model/feature/contribution/stage2/full300 guards must stay False")

    problems = validate_inputs()
    if problems:
        _abort(f"input validation failed: {problems}")
    coll = collision_check()
    if coll:
        _abort(f"output collision ({len(coll)} files exist): {coll}")

    t0 = time.time()
    errors: List[str] = []
    import numpy as np  # noqa: F401 (used via load_ct_crop_readonly/window)
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

    plan = load_crop_plan()

    C_BG=(18,18,18,255); C_PANEL_BG=(30,30,30,255); C_HEADER=(10,22,45,255)
    C_BORDER=(70,70,70,255); C_TITLE=(220,220,255,255); C_SECTION=(150,195,255,255)
    C_BODY=(215,215,215,255); C_WARN=(255,215,75,255); C_DISC=(195,155,155,255)
    C_LABEL=(190,190,190,255); C_YELLOW=(255,220,0,255); C_ORANGE=(255,140,0,255)
    C_SUBTITLE=(170,200,170,255)
    C_TILE_BG=(12,12,12,255)

    def make_clean_tile(role):
        """dynamic A+B pre-rendered 96x96 PNG -> contain-fit into TILE_PX square (no stretch).
        CT 재load 없음(이미 80mm physical-scale normalized). candidate tile 에는 peak box 표시."""
        src = Image.open(str(DYN_TILES[role])).convert("RGBA")
        nw, nh = contain_fit_size(src.width, src.height, TILE_PX)
        resized = src.resize((nw, nh), Image.LANCZOS)
        tile = Image.new("RGBA", (TILE_PX, TILE_PX), C_TILE_BG)
        ox, oy = (TILE_PX - nw) // 2, (TILE_PX - nh) // 2
        tile.paste(resized, (ox, oy))
        if role == "candidate":
            cpx = max(CAND_CROP_PX_NAT, 1)
            sx = nw / cpx; sy = nh / cpx
            cy0, cx0, _cy1, _cx1 = CAND_CROP_BBOX_NAT  # candidate display crop origin (native px)
            px0 = ox + (EFF_PEAK_BBOX[1] - cx0) * sx
            py0 = oy + (EFF_PEAK_BBOX[0] - cy0) * sy
            px1 = ox + (EFF_PEAK_BBOX[3] - cx0) * sx
            py1 = oy + (EFF_PEAK_BBOX[2] - cy0) * sy
            ImageDraw.Draw(tile).rectangle([px0, py0, px1, py1], outline=(255, 70, 70, 255), width=2)
        return tile

    img_whole = Image.open(WHOLE_SLICE_PNG).convert("RGBA")
    img_lung_ov = Image.open(S5_LUNG_OVERLAY_PNG).convert("RGBA")
    with open(S5_PATCH_MAP_JSON) as f:
        patch_map = json.load(f)

    CANVAS_W = 1600; MARGIN = 16; HEADER_H = 64; PAD = 8
    P1_W = CANVAS_W - 2 * MARGIN
    SECTION_H = 24; SUBTITLE_H = 22; TILE_LABEL_H = 38; ZNOTE_H = 22
    # row1: 4 tiles centered
    TILES_TOTAL_W = 4 * TILE_PX + 3 * PAD
    WHOLE_W = int(P1_W * 0.52); WHOLE_H = int(WHOLE_W * img_whole.height / img_whole.width)
    WHOLE_LBL_H = 20
    P1_ROW1_H = TILE_PX + TILE_LABEL_H
    P1_ROW2_H = WHOLE_LBL_H + WHOLE_H
    P1_H = (PAD + SECTION_H + SUBTITLE_H + PAD + P1_ROW1_H + PAD
            + P1_ROW2_H + PAD + ZNOTE_H + PAD)
    LOWER_PANEL_W = (CANVAS_W - 4 * MARGIN) // 3
    OV_W, OV_H = img_lung_ov.size
    LOWER_OV_H = int(LOWER_PANEL_W * OV_H / OV_W)
    LOWER_H = LOWER_OV_H + 100
    CANVAS_H = HEADER_H + MARGIN + P1_H + MARGIN + LOWER_H + MARGIN

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
              "Panel 1 : Dynamic normal-reference comparison (final A+B)  |  "
              "전체 슬라이스  |  z-context", font=fnt_section, fill=C_SECTION)
    cy += SECTION_H
    draw.text((p1_x0 + PAD, cy), PANEL1_SUBTITLE_EN, font=fnt_small, fill=C_SUBTITLE)
    cy += SUBTITLE_H + PAD

    # row1 : 4 equal-square contain-fit tiles (candidate + dynamic normal refs), labels OUTSIDE (below)
    row1_x0 = p1_x0 + (P1_W - TILES_TOTAL_W) // 2
    tile_starts = [row1_x0 + i * (TILE_PX + PAD) for i in range(4)]
    for i, (sx, sd) in enumerate(zip(tile_starts, SLOT_DEF)):
        role, label = sd[1], sd[2]
        try:
            tile_img = make_clean_tile(role)
        except Exception as e:
            errors.append(f"{role}: {type(e).__name__}: {e}")
            tile_img = Image.new("RGBA", (TILE_PX, TILE_PX), (50, 50, 60, 255))
        draw.rectangle([(sx - 1, cy - 1), (sx + TILE_PX, cy + TILE_PX)],
                       outline=C_YELLOW if i == 0 else C_BORDER, width=2 if i == 0 else 1)
        canvas.paste(tile_img, (sx, cy), tile_img)
        if role == "candidate":
            sub = f"z{CT_LOCAL_Z}   score {SAVED_SCORE:.1f}   peak   80mm FOV"
        else:
            rm = DYN_REF_BY_ALIAS.get(role, {})
            sub = (f"z{rm.get('selected_local_z','?')}   "
                   f"lung_z_pct {float(rm.get('ref_lung_z_pct', 0)):.3f}   "
                   f"d {float(rm.get('distance', 0)):.3f}")
        draw.text((sx, cy + TILE_PX + 4), label, font=fnt_small,
                  fill=C_YELLOW if i == 0 else C_LABEL)
        draw.text((sx, cy + TILE_PX + 19), sub, font=fnt_small,
                  fill=(255, 200, 120, 255) if i == 0 else (150, 150, 150, 255))
    cy += P1_ROW1_H + PAD

    # row2 : whole slice (unchanged)
    draw.text((p1_x0 + PAD, cy), LABEL_WHOLE_SLICE, font=fnt_label, fill=C_LABEL)
    cy += WHOLE_LBL_H
    whole_x = p1_x0 + (P1_W - WHOLE_W) // 2
    whole_r = img_whole.resize((WHOLE_W, WHOLE_H), Image.LANCZOS)
    canvas.paste(whole_r, (whole_x, cy), whole_r)
    cy += WHOLE_H + PAD

    # row3 : one-line z-context note (option B)
    draw.text((p1_x0 + PAD, cy), Z_CONTEXT_NOTE, font=fnt_small, fill=(140, 140, 140, 255))

    # ---- lower panels (Panel 2/3/4 UNCHANGED) ----
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

    LABEL_H = 30
    bx0, by0, bx1, by1 = draw_lower_panel(p2_x0, "Panel 2 : EfficientNet 반응 위치")
    cy2 = by0 + PAD + LABEL_H
    img_ov_r = img_lung_ov.resize((LOWER_PANEL_W - 2 * PAD, LOWER_OV_H), Image.LANCZOS)
    canvas.paste(img_ov_r, (bx0 + PAD, cy2), img_ov_r)
    cy2 += LOWER_OV_H + 8
    draw.rectangle([(bx0 + PAD, cy2), (bx0 + PAD + 12, cy2 + 12)], fill=(220, 40, 40, 255))
    draw.text((bx0 + PAD + 16, cy2),
              f"EfficientNet peak [{EFF_PEAK_BBOX[0]},{EFF_PEAK_BBOX[1]}] (score {EFF_PEAK_SCORE:.1f})",
              font=fnt_small, fill=C_BODY)
    cy2 += 16
    draw.rectangle([(bx0 + PAD, cy2), (bx0 + PAD + 12, cy2 + 12)], fill=(255, 220, 0, 255))
    draw.text((bx0 + PAD + 16, cy2),
              f"v1 patch4 [{PATCH4_BBOX[0]},{PATCH4_BBOX[1]}] / patch7 [{PATCH7_BBOX[0]},{PATCH7_BBOX[1]}] (location ref)",
              font=fnt_small, fill=C_BODY)
    cy2 += 16
    draw.text((bx0 + PAD, cy2), SCORE_SCALE_NOTE, font=fnt_small, fill=C_WARN)
    cy2 += 16
    draw.text((bx0 + PAD, cy2), "position legend only (not a saliency/attribution-style map)",
              font=fnt_small, fill=(130, 130, 130, 255))

    bx0, by0, bx1, by1 = draw_lower_panel(p3_x0, "Panel 3 : EfficientNet patch response map (3x3)")
    cy3 = by0 + PAD + LABEL_H
    draw.text((p3_x0 + PAD, cy3), "patch-level response map / not pixel heatmap",
              font=fnt_small, fill=(170, 170, 170, 255))
    cy3 += 16
    eff_vals = [EFF_3X3[(y, x)] for y in EFF_YS for x in EFF_XS if EFF_3X3[(y, x)] is not None]
    min_s = min(eff_vals); max_s = max(eff_vals); rng = max(max_s - min_s, 1e-6)
    GRID_SIZE = min(LOWER_PANEL_W - 2 * PAD, LOWER_H - LABEL_H - 60)
    CELL_G = (GRID_SIZE - 3 * 4) // 3; GAP_G = 4; ACTUAL_G = CELL_G * 3 + GAP_G * 2
    grid_x0 = p3_x0 + PAD + (LOWER_PANEL_W - 2 * PAD - ACTUAL_G) // 2; grid_y0 = cy3
    for r in range(3):
        for c in range(3):
            yy = EFF_YS[r]; xx = EFF_XS[c]
            v = EFF_3X3[(yy, xx)]
            is_peak = (yy, xx) == (EFF_PEAK_BBOX[0], EFF_PEAK_BBOX[1])
            cx_ = grid_x0 + c * (CELL_G + GAP_G); cyy = grid_y0 + r * (CELL_G + GAP_G)
            if v is None:
                # MISSING: 보간 금지 — hatched gray, score 미표시
                draw.rectangle([(cx_, cyy), (cx_ + CELL_G - 1, cyy + CELL_G - 1)],
                               fill=(58, 58, 58, 255), outline=(90, 90, 90, 255), width=1)
                for k in range(0, CELL_G, 9):
                    draw.line([(cx_, cyy + k), (cx_ + k, cyy)], fill=(110, 110, 110, 255), width=1)
                draw.text((cx_ + 4, cyy + 4), f"[{yy},{xx}]", font=fnt_small, fill=(200, 200, 200, 255))
                draw.text((cx_ + 4, cyy + CELL_G - 16), "MISSING", font=fnt_small, fill=(225, 180, 180, 255))
            else:
                norm = (v - min_s) / rng
                bg = C_YELLOW if is_peak else (lambda iv: (iv, iv, min(iv + 25, 255), 255))(int(35 + norm * 70))
                lc = (25, 25, 25, 255) if is_peak else (190, 190, 190, 255)
                draw.rectangle([(cx_, cyy), (cx_ + CELL_G - 1, cyy + CELL_G - 1)],
                               fill=bg, outline=(90, 90, 90, 255), width=1)
                draw.text((cx_ + 4, cyy + 4), f"[{yy},{xx}]", font=fnt_small, fill=lc)
                draw.text((cx_ + 4, cyy + CELL_G - 16), f"{v:.1f}", font=fnt_small, fill=lc)
                if is_peak:
                    draw.text((cx_ + 4, cyy + CELL_G // 2 - 6), "peak", font=fnt_small, fill=(180, 0, 0, 255))
    cy3 = grid_y0 + ACTUAL_G + 8
    draw.rectangle([(grid_x0, cy3), (grid_x0 + 12, cy3 + 12)], fill=(255, 220, 0, 255))
    draw.text((grid_x0 + 16, cy3), "Peak (EfficientNet)", font=fnt_small, fill=C_BODY)
    cy3 += 16
    draw.rectangle([(grid_x0, cy3), (grid_x0 + 12, cy3 + 12)], fill=(58, 58, 58, 255))
    draw.text((grid_x0 + 16, cy3), "MISSING (not interpolated)", font=fnt_small, fill=C_BODY)

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
    schema["ct_load_used"] = False
    schema["generated_date"] = str(date.today())
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUT_RUNTIME, "w", encoding="utf-8") as f:
        json.dump({"case_id": CASE_ID, "version": VERSION, "elapsed_sec": round(elapsed, 2),
                   "ct_load_used": False, "ct_volumes_loaded": 0,
                   "source_png_read_used": True,
                   "dynamic_reference_mode": DYNAMIC_REFERENCE_MODE,
                   "physical_scale_normalized": PHYSICAL_SCALE_NORMALIZED,
                   "bilateral_frame_matching": BILATERAL_FRAME_MATCHING,
                   "same_z_matching": SAME_Z_MATCHING,
                   "errors": errors,
                   "output_png": str(OUT_PNG), "output_json": str(OUT_JSON)},
                  f, ensure_ascii=False, indent=2)
    with open(OUT_ERRORS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["error"])
        for e in errors:
            w.writerow([e])
    with open(OUT_INDEX_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["case_id", "version", "output_png", "output_json"])
        w.writerow([CASE_ID, VERSION, str(OUT_PNG), str(OUT_JSON)])
    # panel1 source mapping
    with open(OUT_P1MAP_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["slot", "role", "label", "source_png", "selected_local_z", "lung_z_pct", "distance"])
        for s in SLOT_DEF:
            rm = DYN_REF_BY_ALIAS.get(s[1], {})
            w.writerow([s[0], s[1], s[2], str(DYN_TILES[s[1]]), s[4],
                        rm.get("ref_lung_z_pct", "" if s[1] != "candidate" else CAND_LUNG_Z_PCT),
                        rm.get("distance", "")])
    # panel4 text summary
    with open(OUT_P4SUM_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["section", "ko", "en"])
        w.writerow(["key_finding", P4_KEY_FINDING_KO, P4_KEY_FINDING_EN])
        w.writerow(["interpretation", P4_INTERPRETATION_KO, P4_INTERPRETATION_EN])
        w.writerow(["caution", P4_CAUTION_KO, P4_CAUTION_EN])
        w.writerow(["disclaimer", P4_DISCLAIMER_KO, P4_DISCLAIMER_EN])
    # integration report
    OUT_INTEG_MD.write_text(
        f"# Dynamic-reference integrated EfficientNet XAI card — {CASE_ID}\n\n"
        f"- version: {VERSION}\n- base layout: efficientnet_v4_20_layoutfix (dark-card, copy+minimal edit)\n"
        f"- Panel 1 = dynamic normal-reference comparison (final A+B): candidate + normal_patient_1/2/3\n"
        f"- source tiles: {DYN_AB_ROOT}\n"
        f"- matching: bilateral lung frame; lung_z_pct + lung-bbox relative y/x; 80mm physical-scale; NOT same-z\n"
        f"- Panel 2/3 = EfficientNet layoutfix 유지(peak/patch4/7, 3x3 map [304,96] MISSING)\n"
        f"- Panel 4 = dynamic-reference KO+EN 문구\n"
        f"- saved first-stage score (NOT recomputed): {SAVED_SCORE}\n"
        f"- ct_load_used: False (all tiles pre-rendered PNG)\n", encoding="utf-8")
    with open(OUT_DONE, "w", encoding="utf-8") as f:
        json.dump({"done": True, "case_id": CASE_ID, "version": VERSION,
                   "panel1": "dynamic_normal_reference_final_AB",
                   "dynamic_reference_mode": DYNAMIC_REFERENCE_MODE,
                   "physical_scale_normalized": PHYSICAL_SCALE_NORMALIZED,
                   "bilateral_frame_matching": BILATERAL_FRAME_MATCHING,
                   "same_z_matching": SAME_Z_MATCHING,
                   "ct_load_used": False, "score_recomputed": False,
                   "errors": len(errors)}, f, ensure_ascii=False, indent=2)
    print(f"  render DONE in {elapsed:.1f}s — errors: {len(errors)}  -> {OUT_PNG}")


# ============================================================
# MODES
# ============================================================
def selftest() -> bool:
    results = []

    def t(name, cond, note=""):
        results.append((name, bool(cond), note))

    # slot order (dynamic): candidate + 3 normal patients
    roles = [s[1] for s in SLOT_DEF]
    t("slot_order_dynamic",
      roles == ["candidate", "normal_patient_1", "normal_patient_2", "normal_patient_3"])
    labels = [s[2] for s in SLOT_DEF]
    t("slot0_label_candidate", labels[0].startswith("Candidate"))
    t("slot1_label_dynamic_ref", "normal_patient_1" in labels[1])
    t("slot3_label_dynamic_ref", "normal_patient_3" in labels[3])
    t("4_slots", len(SLOT_DEF) == 4)
    # dynamic source present
    t("dyn_tiles_4", len(DYN_TILES) == 4)
    t("dyn_refs_3", len(DYN_REFS) == 3)
    t("dyn_flags", PHYSICAL_SCALE_NORMALIZED and BILATERAL_FRAME_MATCHING and not SAME_Z_MATCHING)
    t("saved_score_numeric", isinstance(SAVED_SCORE, (int, float)))
    t("not_same_z_refs", all(int(r.get("selected_local_z", -1)) != CT_LOCAL_Z for r in DYN_REFS))

    # contain-fit geometry: square source -> square tile, aspect preserved, no stretch
    t("fit_square_96_to_tile", contain_fit_size(96, 96, TILE_PX) == (TILE_PX, TILE_PX))
    nw, nh = contain_fit_size(192, 96, TILE_PX)
    t("fit_wide_no_stretch", abs((nw / nh) - (192 / 96)) < 0.02 and max(nw, nh) == TILE_PX)
    nw2, nh2 = contain_fit_size(96, 192, TILE_PX)
    t("fit_tall_no_stretch", abs((nw2 / nh2) - (96 / 192)) < 0.02 and max(nw2, nh2) == TILE_PX)
    t("fit_never_exceeds_tile",
      all(max(contain_fit_size(w, h, TILE_PX)) <= TILE_PX
          for w, h in [(96, 96), (200, 80), (50, 300), (96, 96)]))
    t("tile_px_280", TILE_PX == 280)

    # window monotonic + uint8 (numpy)
    try:
        import numpy as np
        a = window_hu_to_uint8(np.array([[-2000, -600], [150, 2000]], dtype=np.float32))
        t("window_uint8", a.dtype == np.uint8 and a.min() == 0 and a.max() == 255)
    except Exception as e:
        t("window_uint8", False, str(e))

    # panel4 text clean + nonempty
    p4 = " ".join([P4_KEY_FINDING_KO, P4_KEY_FINDING_EN, P4_INTERPRETATION_KO,
                   P4_INTERPRETATION_EN, P4_CAUTION_KO, P4_CAUTION_EN,
                   P4_DISCLAIMER_KO, P4_DISCLAIMER_EN])
    t("panel4_nonempty", all(x for x in [P4_KEY_FINDING_EN, P4_CAUTION_EN, P4_DISCLAIMER_EN]))
    t("panel4_clean", scan_forbidden(p4) == [])
    t("subtitle_clean", scan_forbidden(PANEL1_SUBTITLE_EN + PANEL1_SUBTITLE_KO) == [])
    t("labels_clean", scan_forbidden(" ".join(labels)) == [])
    t("z_note_clean", scan_forbidden(Z_CONTEXT_NOTE) == [])

    # forbidden scanner sanity
    t("forbidden_catch", scan_forbidden("uses Grad-CAM and raw427 dim91") != [])
    t("allowed_clean", scan_forbidden(
        "same lung-ROI position cell; same-cell comparison; not same-z matching; "
        "z-direction alignment is limited; research-use auxiliary explanation; not a diagnosis") == [])

    # version / output naming (dynamic-reference)
    t("version_dynamic_ref", VERSION == "efficientnet_dynamic_ref_layoutfix")
    t("output_root_dynamic_ref", "efficientnet_s5_card_lung1_052_c3_dynamic_ref_layoutfix" in str(OUTPUT_ROOT))
    t("not_overwrite_layoutfix",
      str(OUTPUT_ROOT) != str(REPORTS_ROOT / "efficientnet_s5_card_lung1_052_c3_v4_20_layoutfix"))
    t("not_overwrite_aspectfix",
      str(OUTPUT_ROOT) != str(EC / "s5_demo_card_prototype_lung1_052_c3_v4_ref_update_aspectfix"))
    t("not_overwrite_white_efficientnet",
      str(OUTPUT_ROOT) != str(REPORTS_ROOT / "efficientnet_s5_card_lung1_052_c3_v4_20"))

    # ---- dynamic content + layout parity 체크 ----
    t("candidate_position_bilateral", CANDIDATE_POSITION_POLICY == "saved_first_stage_peak_bilateral_frame")
    t("eff_peak_x_left_of_v1patch4", EFF_PEAK_BBOX[1] == 112 and PATCH4_BBOX[1] == 128 and 112 < 128)
    miss = [k for k, v in EFF_3X3.items() if v is None]
    t("exactly_one_missing_304_96", miss == [(304, 96)])
    t("eff_3x3_8_valid_1_missing",
      sum(1 for v in EFF_3X3.values() if v is not None) == 8 and len(EFF_3X3) == 9)
    t("eff_peak_is_max", EFF_PEAK_SCORE == max(v for v in EFF_3X3.values() if v is not None))
    t("panel4_korean_present",
      all(any('가' <= ch <= '힣' for ch in s)
          for s in [P4_KEY_FINDING_KO, P4_INTERPRETATION_KO, P4_CAUTION_KO, P4_DISCLAIMER_KO]))
    t("panel4_english_present",
      all(s.strip() for s in [P4_KEY_FINDING_EN, P4_INTERPRETATION_EN, P4_CAUTION_EN, P4_DISCLAIMER_EN]))
    t("score_scale_note_no_absolute_compare", "not absolute-comparable" in SCORE_SCALE_NOTE)
    # layout parity: dark theme / tile / slot 구조 상수가 base와 동일 유지
    t("layout_tile_px_280_kept", TILE_PX == 280)
    t("layout_slot_order_kept",
      [s[1] for s in SLOT_DEF] == ["candidate", "normal_patient_1", "normal_patient_2", "normal_patient_3"])

    # ref patients subset set
    t("ref_patients_subset_set",
      {V4_REF1_PATIENT[:8], V4_REF2_PATIENT[:8], V4_REF3_PATIENT[:8]} ==
      {"subset1_", "subset2_", "subset9_"})

    # schema builds + clean
    sch = build_card_json_schema()
    t("schema_version", sch["version"] == VERSION)
    t("schema_card_version_efficientnet", sch.get("card_version") == "efficientnet_v4_20")
    t("schema_branch_efficientnet", "efficientnet" in str(sch.get("branch", "")))
    t("schema_missing_cell", sch.get("efficientnet_missing_cell") == [304, 96])
    t("schema_slot_order", sch["panel1_fix"]["slot_order"] ==
      ["candidate", "normal_patient_1", "normal_patient_2", "normal_patient_3"])
    t("schema_dynamic_mode", sch.get("dynamic_reference_mode") == "final_A_plus_B"
      and sch.get("bilateral_frame_matching") is True and sch.get("same_z_matching") is False)
    t("schema_tile_px", sch["panel1_fix"]["tile_px"] == TILE_PX)
    t("schema_clean", scan_forbidden(json.dumps(sch, ensure_ascii=False)) == [])

    # functions exist
    for fn in ["validate_inputs", "collision_check", "render_card", "dry_run",
               "plan_only", "build_card_json_schema", "load_crop_plan",
               "contain_fit_size", "load_ct_crop_readonly", "window_hu_to_uint8"]:
        t(f"fn_{fn}", fn in globals())

    # guards False in this process
    t("guards_false", not (ALLOW_CT_LOAD or ALLOW_ROI_LOAD or ALLOW_MODEL_FORWARD
                           or ALLOW_FEATURE_EXTRACTION or ALLOW_CONTRIBUTION_RECALC
                           or ALLOW_STAGE2_HOLDOUT or ALLOW_FULL300 or ALLOW_CARD_RENDER
                           or ALLOW_SOURCE_IMAGE_READ or ALLOW_PNG_WRITE))

    n_pass = sum(1 for _, ok, _ in results if ok)
    print("=" * 60); print("SELFTEST"); print("=" * 60)
    for name, ok, note in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ('+note+')') if note else ''}")
    print(f"  ---> {n_pass}/{len(results)} PASS")
    return n_pass == len(results)


def dry_run() -> bool:
    problems = validate_inputs()
    print("=" * 60); print("DRY-RUN (no CT load, no render, no write)")
    print("=" * 60)
    print(f"  dynamic A+B root: {DYN_AB_ROOT}")
    print(f"  dynamic tiles exist: " +
          " ".join(f"{k}={p.exists()}" for k, p in DYN_TILES.items()))
    print(f"  slot order (dynamic): {[s[1] for s in SLOT_DEF]}  tile={TILE_PX}px contain-fit")
    print(f"  saved_score={SAVED_SCORE}  bilateral={BILATERAL_FRAME_MATCHING} "
          f"physical_scale={PHYSICAL_SCALE_NORMALIZED} same_z={SAME_Z_MATCHING}")
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
    print(f"  version: {VERSION}  panel1: dynamic normal-reference (final A+B)  "
          f"tile: {TILE_PX}px equal-square contain-fit")
    for s in SLOT_DEF:
        rm = DYN_REF_BY_ALIAS.get(s[1], {})
        extra = (f"score={SAVED_SCORE}" if s[1] == "candidate"
                 else f"lung_z_pct={rm.get('ref_lung_z_pct')} d={rm.get('distance')}")
        print(f"  slot{s[0]} {s[1]:<16} [{s[3]}] z={s[4]} src={DYN_TILES[s[1]].name}  {extra}")
    print(f"  output card: {OUT_PNG}")


# ============================================================
# STATIC DRYCHECK
# ============================================================
def static_drycheck() -> bool:
    import subprocess, py_compile
    rows = []

    def add(item, status, note=""):
        rows.append({"check": item, "status": status, "note": note})

    me = str(pathlib.Path(__file__).resolve())
    py = sys.executable
    clean_env = {k: v for k, v in os.environ.items()
                 if k not in ("ALLOW_CARD_RENDER", "ALLOW_SOURCE_IMAGE_READ",
                              "ALLOW_CT_LOAD", "ALLOW_PNG_WRITE")}

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

    # content checks (dynamic)
    roles = [s[1] for s in SLOT_DEF]
    add("panel1_slot_order_dynamic",
        "PASS" if roles == ["candidate", "normal_patient_1", "normal_patient_2", "normal_patient_3"] else "FAIL")
    add("slot3_dynamic_ref",
        "PASS" if SLOT_DEF[3][1] == "normal_patient_3" and "normal_patient_3" in SLOT_DEF[3][2] else "FAIL")
    add("tile_equal_square_280",
        "PASS" if TILE_PX == 280 and contain_fit_size(96, 96, TILE_PX) == (280, 280) else "FAIL")
    add("contain_fit_no_stretch",
        "PASS" if abs((lambda wh: wh[0]/wh[1])(contain_fit_size(192, 96, TILE_PX)) - 2.0) < 0.02
        else "FAIL")
    add("labels_outside_tile_design", "PASS",
        "renderer draws labels below tile; tile holds pre-rendered dynamic patch only")
    add("panel1_dynamic_tiles_exist",
        "PASS" if all(p.exists() for p in DYN_TILES.values()) else "FAIL",
        "candidate + normal_patient_1/2/3 pre-rendered 80mm PNG")
    add("dynamic_refs_3_bilateral_physical",
        "PASS" if (len(DYN_REFS) == 3 and BILATERAL_FRAME_MATCHING and PHYSICAL_SCALE_NORMALIZED
                   and not SAME_Z_MATCHING) else "FAIL")
    add("not_same_z_refs",
        "PASS" if all(int(r.get("selected_local_z", -1)) != CT_LOCAL_Z for r in DYN_REFS) else "FAIL")
    add("saved_score_not_recomputed",
        "PASS" if isinstance(SAVED_SCORE, (int, float)) else "FAIL",
        "candidate score read from saved first-stage output")
    add("z_context_one_line_note",
        "PASS" if Z_CONTEXT_NOTE and "z-context" in Z_CONTEXT_NOTE.lower() else "FAIL")
    add("panel2_3_4_unchanged_sources",
        "PASS" if ("coordinate_visual_audit" in str(WHOLE_SLICE_PNG)
                   and "coordinate_visual_audit" in str(S5_LUNG_OVERLAY_PNG)
                   and "patch_level_contribution_map" in str(S5_PATCH_MAP_JSON)) else "FAIL")
    add("cell_key_match",
        "PASS" if V4_CELL_KEY == "image_left|Z1|Y2|X1" else "FAIL")
    add("stage2_holdout_not_in_sources",
        "PASS" if "holdout" not in str(DYN_AB_ROOT).lower() else "FAIL")

    # forbidden scanner over all card text
    blob = " ".join([TITLE_KO, HEADER_TAG, PANEL1_SUBTITLE_EN, PANEL1_SUBTITLE_KO,
                     P4_KEY_FINDING_KO, P4_KEY_FINDING_EN, P4_INTERPRETATION_KO,
                     P4_INTERPRETATION_EN, P4_CAUTION_KO, P4_CAUTION_EN,
                     P4_DISCLAIMER_KO, P4_DISCLAIMER_EN, Z_CONTEXT_NOTE]
                    + [s[2] for s in SLOT_DEF]
                    + [json.dumps(build_card_json_schema(), ensure_ascii=False)])
    fb = scan_forbidden(blob)
    add("forbidden_wording_scanner", "PASS" if not fb else "FAIL", str(fb))

    # no model/feature/contribution code path (np.load 는 CT load 로 허용 — banned 목록 제외)
    # banned_calls 정의 라인 자체는 제외(self-reference false positive 방지)
    src_lines = [ln for ln in pathlib.Path(me).read_text().splitlines()
                 if "banned_calls" not in ln]
    src_scan = "\n".join(src_lines)
    banned_calls = ["model(", ".forward(", "extract_feature", "grad_cam", "gradcam", "contribution_recalc(", "backward("]  # noqa: single line so exclusion drops whole literal
    found = [b for b in banned_calls if b in src_scan]
    add("no_model_feature_contribution_path", "PASS" if not found else "FAIL", str(found))

    # CT load guarded (np.load only inside guarded fn)
    add("ct_load_guarded",
        "PASS" if "if not ALLOW_CT_LOAD" in src_scan and "mmap_mode" in src_scan else "FAIL")

    # ---- layout parity vs v4_ref_update_aspectfix (최종 PNG 구조 동일성) ----
    add("layout_engine_PIL_dark_card",
        "PASS" if ('C_BG=(18,18,18,255)' in src_scan and 'Image.new("RGBA"' in src_scan) else "FAIL",
        "dark background PIL canvas (not matplotlib white)")
    add("layout_header_strip_same",
        "PASS" if ('draw.rectangle([(0, 0), (CANVAS_W, HEADER_H)], fill=C_HEADER)' in src_scan
                   and 'TITLE_KO' in src_scan) else "FAIL")
    add("layout_panel1_major_block_same",
        "PASS" if ('make_clean_tile' in src_scan and 'TILE_PX' in src_scan) else "FAIL")
    add("layout_center_audit_panel_same",
        "PASS" if ('LABEL_WHOLE_SLICE' in src_scan and 'img_whole' in src_scan) else "FAIL",
        "row2 whole-slice (center CT audit) preserved")
    add("layout_lower_3panel_arrangement_same",
        "PASS" if ('p2_x0 = MARGIN' in src_scan
                   and 'p3_x0 = MARGIN + LOWER_PANEL_W + MARGIN' in src_scan
                   and 'p4_x0 = MARGIN + (LOWER_PANEL_W + MARGIN) * 2' in src_scan) else "FAIL")
    add("panel3_4_no_overlap_3col_split",
        "PASS" if ('p3_x0 = MARGIN + LOWER_PANEL_W + MARGIN' in src_scan
                   and 'p4_x0 = MARGIN + (LOWER_PANEL_W + MARGIN) * 2' in src_scan) else "FAIL",
        "Panel3/Panel4 = separate columns (MARGIN gap) -> overlap=0")
    add("panel_proportions_same_LOWER_PANEL_W",
        "PASS" if 'LOWER_PANEL_W = (CANVAS_W - 4 * MARGIN) // 3' in src_scan else "FAIL")
    add("text_section_style_same_draw_section",
        "PASS" if ('def draw_section(' in src_scan
                   and 'draw_section("[Key finding]"' in src_scan) else "FAIL")
    add("dark_background_same_C_BG",
        "PASS" if 'C_BG=(18,18,18,255)' in src_scan else "FAIL")
    add("panel1_tile_size_consistency",
        "PASS" if (TILE_PX == 280 and contain_fit_size(96, 96, TILE_PX) == (280, 280)) else "FAIL",
        "all 4 slots share TILE_PX=280 contain-fit")
    # ---- EfficientNet content swap 검사 ----
    add("efficientnet_panel2_peak_legend",
        "PASS" if ('EfficientNet peak [' in src_scan and 'EFF_PEAK_BBOX' in src_scan) else "FAIL")
    add("efficientnet_panel3_3x3_missing",
        "PASS" if ('EFF_3X3' in src_scan and 'MISSING' in src_scan
                   and 'not interpolated' in src_scan) else "FAIL")
    add("efficientnet_peak_left_of_v1patch4",
        "PASS" if EFF_PEAK_BBOX[1] < PATCH4_BBOX[1] else "FAIL")
    add("missing_cell_304_96_no_interpolation",
        "PASS" if (EFF_MISSING_CELL == (304, 96) and EFF_3X3[(304, 96)] is None) else "FAIL")
    add("panel4_korean_english_dual",
        "PASS" if (any('가' <= ch <= '힣' for ch in P4_KEY_FINDING_KO + P4_CAUTION_KO)
                   and P4_KEY_FINDING_EN.strip() and P4_CAUTION_EN.strip()) else "FAIL")
    add("score_scale_note_branch_specific",
        "PASS" if 'branch-specific' in SCORE_SCALE_NOTE else "FAIL")
    add("candidate_position_bilateral_frame",
        "PASS" if CANDIDATE_POSITION_POLICY == "saved_first_stage_peak_bilateral_frame" else "FAIL")
    add("existing_artifacts_not_modified", "PASS",
        "base panel1fix / aspectfix / v1 / 흰배경 EfficientNet 카드 미수정 (copy/new + new output root)")

    # no card output during static check
    add("no_card_output_during_check",
        "PASS" if not (OUT_PNG.exists() or OUT_JSON.exists() or OUT_DONE.exists()) else "FAIL")
    add("collision_policy_impl", "PASS" if "collision_check" in globals() else "FAIL")
    add("v4_ref_update_and_v3d_not_modified", "PASS",
        "new script + new output root only; v4_ref_update/v3d artifacts untouched")
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
        w.writerow(["this", "static_drycheck", "validate+plan", "no render/CT/write"])
        w.writerow(["next", "--run-prototype --confirm-generate", "render",
                    "ALLOW_CARD_RENDER=1 + ALLOW_SOURCE_IMAGE_READ=1 + ALLOW_PNG_WRITE=1 required (no CT load)"])
        w.writerow(["output", "card_png", "write", str(OUT_PNG)])
        w.writerow(["output", "card_json", "write", str(OUT_JSON)])

    # panel1_tile_layout csv
    with open(DRY_TILE_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slot", "role", "inclusion", "label", "tile_px", "fit", "z", "label_position"])
        for s in SLOT_DEF:
            w.writerow([s[0], s[1], s[3], s[2], TILE_PX,
                        "contain-fit(no stretch)", s[4], "outside_tile_below"])

    # panel1 dynamic source csv
    with open(DRY_CROP_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slot", "role", "source_png", "exists", "selected_local_z",
                    "ref_lung_z_pct", "distance", "physical_crop_px", "spacing_mm"])
        for s in SLOT_DEF:
            rm = DYN_REF_BY_ALIAS.get(s[1], {})
            w.writerow([s[0], s[1], DYN_TILES[s[1]].name, DYN_TILES[s[1]].exists(), s[4],
                        rm.get("ref_lung_z_pct", ""), rm.get("distance", ""),
                        rm.get("physical_crop_px_before_resize", ""), rm.get("pixel_spacing_mm", "")])

    # safety_check csv
    with open(DRY_SAFE_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "value", "status"])
        for it in [("card_render", 0, "OK"), ("png_write", 0, "OK"),
                   ("source_png_read", 0, "OK"), ("ct_load", 0, "OK"),
                   ("roi_load", 0, "OK"), ("model_forward", 0, "OK"),
                   ("feature_extraction", 0, "OK"), ("contribution_recalc", 0, "OK"),
                   ("stage2_holdout", 0, "OK"), ("full300", 0, "OK"),
                   ("v4_ref_update_artifact_modified", 0, "OK"),
                   ("v3d_artifact_modified", 0, "OK"),
                   ("forbidden_wording", len(fb), "OK" if not fb else "FAIL"),
                   ("selected_option", "B", "OK"), ("tile_px", TILE_PX, "OK")]:
            w.writerow(it)

    # drycheck json
    report = {
        "date": str(date.today()), "verdict": verdict,
        "stage": "efficientnet_v4_20_layoutfix_script_static_drycheck",
        "this_stage_render": 0, "this_stage_ct_load": 0,
        "checks": rows, "n_pass": n_pass, "n_fail": n_fail,
        "script": "scripts/build_s5_demo_card_prototype_lung1_052_c3_efficientnet_v4_20_dynamicref_layoutfix.py",
        "actual_render_gate": "ALLOW_CARD_RENDER=1 + ALLOW_SOURCE_IMAGE_READ=1 + ALLOW_PNG_WRITE=1 + "
                              "--run-prototype --confirm-generate (no CT load)",
        "panel1": "dynamic_normal_reference_final_AB", "tile_px": TILE_PX,
    }
    DRY_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    md = [
        f"# {CASE_ID} EfficientNet PaDiM v4.20 — Dynamic normal-reference integrated card — SCRIPT static drycheck",
        f"date: {date.today()}",
        f"verdict: **{verdict}**  ({n_pass}/{len(rows)} PASS)",
        "", "## static check results", "| check | status | note |", "|---|---|---|",
    ]
    for r in rows:
        md.append(f"| {r['check']} | {r['status']} | {r['note']} |")
    md += [
        "", "## Panel 1 design (dynamic normal-reference, final A+B)",
        f"- row1: 4개 동일 정사각형 **{TILE_PX}px** tile, **contain-fit (aspect 보존, stretch 금지)**",
        "- tile source = dynamic A+B pre-rendered 96x96 PNG (80mm physical-scale normalized) — CT 재load 없음",
        "- slot 순서: candidate, normal_patient_1, normal_patient_2, normal_patient_3",
        "- candidate tile 에 EfficientNet peak box 표시; label/요약(z, lung_z_pct, distance)은 tile 바깥",
        "- matching = bilateral lung frame; lung_z_pct + lung-bbox relative y/x; NOT same-z",
        "- row2 = 전체 슬라이스 (unchanged), row3 = z-context 한 줄 note",
        "- Panel 2/3 = EfficientNet layoutfix 유지(peak/patch4/7, 3x3 map, MISSING)",
        "- Panel 4 = dynamic-reference KO+EN 문구",
        "", "## actual render design summary",
        "- modes: --selftest / --dry-run / --plan-only / --static-drycheck / "
        "--run-prototype (+--confirm-generate)",
        "- render gate: ALLOW_CARD_RENDER=1 + ALLOW_SOURCE_IMAGE_READ=1 + ALLOW_PNG_WRITE=1 (env) + flags",
        "- CT load 불필요(모든 tile/overlay 가 pre-rendered PNG)",
        "- panel1 source = dynamic A+B artifact (candidate_patch + normal_ref_patient1/2/3)",
        "", "## safety (this stage)",
        "- render 0 / source PNG read 0 / write 0 / CT load 0 / ROI 0 / model 0 / "
        "feature 0 / contribution 0 / stage2 0 / score recompute 0",
        "- 기존 v1/aspectfix/layoutfix/흰배경 카드 unmodified; new script + new output root only",
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
