#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explanation Card S3 Expansion v2 Font-Fix Smoke

기준:
  - reports/explanation_cards/s3_font_fix_smoke_preflight_v1.md (PASS)
  - 기존 expansion v1: build_explanation_card_s3_expansion_cards_v1.py (보존 / 수정 금지)

v2_fontfix_smoke 변경:
  - 한글 폰트 설정 추가: resolve_korean_font() → Malgun Gothic (PRIMARY) / fallback 2종
  - OUT_DIR: s3_expansion_cards_v2_fontfix_smoke/ (신규; 기존 expansion v1 보존)
  - SMOKE_IDS: 6장만 처리 (full 300 옵션 없음)
  - runtime_summary에 selected_font_path / selected_font_family 기록

변경하지 않은 것:
  - B2 display_bbox (max-score patch bbox + margin32)
  - overmerge caution annotation (Panel A 내부)
  - apex_caution annotation (Panel A 내부)
  - internal_use_only = True
  - font_fix_required_before_external_share = True (smoke 단계; 최종 수용 후 갱신)
  - JSON schema (component_union_bbox + display_bbox 모두 보존)
  - threshold 14.0921
  - diagnostic guard (FORBIDDEN_TERMS)
  - Panel A/B/C/D 4-panel 구조
  - z-context logic
  - holdout denylist assert
  - DONE.json guard / 잔여 산출물 BLOCKED guard
  - CARD_JSON_FIELDS / INDEX_FIELDS

가드:
  - 플래그 없으면 BLOCKED (exit 2).
  - --run-cards는 --confirm-generate 동반 필요.
  - DONE.json / 잔여 산출물 있으면 BLOCKED.
  - SMOKE_IDS 외 case_id 처리 BLOCKED.
  - --overwrite 옵션 없음.
  - holdout 교집합 0 assert.
  - CT/mask 로드는 run-cards에서만 np.load(mmap_mode="r").
  - expansion v1 output 보존 (수정/삭제/덮어쓰기 금지).
  - full 300 옵션 없음.
"""

import argparse
import csv
import inspect
import json
import os
import sys
from datetime import datetime

import numpy as np

csv.field_size_limit(10 ** 9)

# ----------------------------------------------------------------------------
# 상수 (v1과 동일 — 수치/문자열 변경 금지)
# ----------------------------------------------------------------------------
THRESHOLD_P95 = 14.0921
THRESHOLD_TYPE = "p95"
N_SMOKE_EXPECTED = 6          # smoke 전용; full 300 옵션 없음
N_EXPECTED_UNIQUE_SIDS = 100  # manifest 원본 기준 (plan-only 참고용)
LUNG_WINDOW_CENTER = -600.0
LUNG_WINDOW_WIDTH = 1500.0
DISPLAY_MARGIN = 32
DISPLAY_BBOX_MODE = "max_score_patch_margin32"
SCORE_TOL = 1e-3
REF_CROP_MAX = 3
CONTOUR_RGB = (0, 255, 0)

LARGE_AREA = 25000
LARGE_ZSPAN = 30
LARGE_PC = 500
EXTREME_AREA = 100000
EXTREME_ZSPAN = 50
EXTREME_PC = 1000

ROLE_NORMAL = "normal_control"
ROLE_LESION = "lesion_candidate"

POSITION_BINS = ("upper_central", "upper_peripheral", "middle_central",
                 "middle_peripheral", "lower_central", "lower_peripheral")

FORBIDDEN_PATH_TOKENS = ("stage2_holdout", "holdout")

FORBIDDEN_TERMS = (
    "cancer", "malignancy", "malignant", "benign", "adenocarcinoma", "carcinoma",
    "tumor", "tumour", "nodule 확정", "pulmonary nodule 확정", "ground-glass nodule 확정",
    "ggn 확정", "폐암", "악성", "양성", "선암", "종양", "결절로 진단", "유리결절로 진단",
)

OVERMERGE_CAUTION_EN = ("Broad component response: the union area is wide, "
                        "so this card focuses on the highest-score local region.")
OVERMERGE_CAUTION_KO = ("광역 component 반응: component 전체 범위가 넓어, "
                        "카드는 최고 점수 인근 국소 영역을 중심으로 표시합니다.")

OVERMERGE_ANNOTATION_EN = "Broad component: red box marks the highest-score local region."
OVERMERGE_ANNOTATION_KO = "광역 component: 빨간 박스는 최고 점수 국소 영역입니다."

APEX_CAUTION_TEXT_EN = "Apex/upper-peripheral caution: review surrounding lung context carefully."
APEX_CAUTION_TEXT_KO = "상부 말초부 주의: 주변 폐 실질 맥락을 함께 확인해야 합니다."

SAFE_ID_MAX_LEN = 16

CARD_JSON_FIELDS = (
    "expansion_case_id", "prototype_role", "patient_id", "safe_id", "component_id",
    "rank_in_patient", "position_bin", "slice_index_min", "slice_index_max",
    "max_score_slice_index", "z_span", "z_ratio", "component_union_bbox", "display_bbox",
    "display_bbox_mode", "display_bbox_margin", "display_bbox_area", "component_bbox_area",
    "display_bbox_area_reduction_ratio", "overmerge_flag", "overmerge_level",
    "overmerge_reason", "overmerge_caution_text", "apex_caution", "apex_caution_reason",
    "apex_caution_text", "patch_count", "max_padim_score", "mean_padim_score",
    "threshold", "threshold_type", "roi_0_0_patch_ratio_mean", "central_peripheral",
    "left_right_metadata", "internal_use_only", "font_fix_required_before_external_share",
    "normal_reference_crops", "card_png_path", "explanation_text", "fp_caution_text",
    "stage_split_safety_flag", "diagnostic_terms_blocked", "diagnostic_guard_passed",
)

INDEX_FIELDS = (
    "expansion_case_id", "prototype_role", "patient_id", "safe_id", "component_id",
    "rank_in_patient", "position_bin", "max_padim_score", "threshold", "z_span",
    "patch_count", "display_bbox_mode", "overmerge_flag", "overmerge_level",
    "apex_caution", "display_bbox_area_reduction_ratio",
    "internal_use_only", "font_fix_required_before_external_share",
    "card_png_path", "card_json_path", "status", "error",
)

PLANNED_ARTIFACTS = (
    "cards_png/<expansion_case_id>.png", "cards_json/<expansion_case_id>.json",
    "index_cards.csv", "runtime_summary.json", "errors.csv", "DONE.json",
)

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_BLOCKED = 2

# ----------------------------------------------------------------------------
# 폰트 후보 (PRIMARY → FALLBACK 순서)
# ----------------------------------------------------------------------------
FONT_CANDIDATES = [
    "/mnt/c/Windows/Fonts/malgun.ttf",                   # PRIMARY: Malgun Gothic
    "/mnt/c/Windows/Fonts/Hancom Gothic Regular.ttf",   # FALLBACK_1: Hancom Gothic
    "/mnt/c/Windows/Fonts/HANDotum.ttf",                # FALLBACK_2: HCR Dotum
]

# ----------------------------------------------------------------------------
# Smoke 대상 (6장 고정; 다른 case_id → BLOCKED)
# ----------------------------------------------------------------------------
SMOKE_IDS = frozenset({
    "LUNG1-284__c1",
    "LUNG1-220__c3",
    "LUNG1-402__c1",
    "LUNG1-305__c1",
    "MSD_lung_054__c1",
    "subset9_1.3.6.1.4.1.14519.5.2.1.6279.6001.291156498203266896953765649282__c1",
})

HOLD_IDS = frozenset({
    "LUNG1-284__c1",
    "LUNG1-220__c3",
    "LUNG1-402__c1",
})

# ----------------------------------------------------------------------------
# 경로
# ----------------------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_CSV = os.path.join(
    REPO, "outputs/position-aware-padim-v1/candidates/s3_expansion_manifest_v1"
          "/s3_expansion_candidate_manifest_v1.csv")
PATIENT_SUMMARY_CSV = os.path.join(
    REPO, "outputs/position-aware-padim-v1/candidates/s3_expansion_manifest_v1"
          "/s3_expansion_patient_summary_v1.csv")
CAND_ROOT = os.path.join(
    REPO, "outputs/position-aware-padim-v1/candidates/padim_v2_roi0_0_explanation_candidates_v1")
PATCH_CSV = os.path.join(CAND_ROOT, "patch_candidates.csv")
REF_BANK_FULL = os.path.join(
    REPO, "outputs/position-aware-padim-v1/visualizations/candidate_cards/reference_bank_v1/full")
REF_CROP_MANIFEST = os.path.join(REF_BANK_FULL, "reference_crop_manifest.csv")
SPLIT_CSV = os.path.join(
    REPO, "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1.csv")

NORMAL_CT_ROOT = (
    "/mnt/c/Users/jinhy/Desktop/"
    "Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/volumes_npy")
LESION_CT_ROOT = (
    "/mnt/c/Users/jinhy/Desktop/"
    "NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy")
MASK_ROOT = os.path.join(
    REPO, "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1")

OUT_DIR = os.path.join(
    REPO, "outputs/position-aware-padim-v1/visualizations/candidate_cards"
          "/s3_expansion_cards_v2_fontfix_smoke")

# 보존 대상 (수정/삭제/덮어쓰기 금지)
V1_EXPANSION_OUT_DIR = os.path.join(
    REPO, "outputs/position-aware-padim-v1/visualizations/candidate_cards/s3_expansion_cards_v1")
V1_OUT_DIR = os.path.join(
    REPO, "outputs/position-aware-padim-v1/visualizations/candidate_cards/s3_prototype_cards_v1")
V2_OUT_DIR = os.path.join(
    REPO, "outputs/position-aware-padim-v1/visualizations/candidate_cards/s3_prototype_cards_v2")
V3_OUT_DIR = os.path.join(
    REPO, "outputs/position-aware-padim-v1/visualizations/candidate_cards/s3_prototype_cards_v3")


# ----------------------------------------------------------------------------
# 가드 / 경로 헬퍼
# ----------------------------------------------------------------------------
def safe_path(path):
    low = str(path).replace("\\", "/").lower()
    for tok in FORBIDDEN_PATH_TOKENS:
        if tok in low:
            raise RuntimeError("FORBIDDEN path token '%s' in: %s" % (tok, path))
    return path


def resolve_ct_path(safe_id, role):
    root = NORMAL_CT_ROOT if role == ROLE_NORMAL else LESION_CT_ROOT
    return os.path.join(root, safe_id, "ct_hu.npy")


def resolve_mask_path(safe_id, role):
    sub = "normal" if role == ROLE_NORMAL else "lesion"
    return os.path.join(MASK_ROOT, sub, safe_id, "refined_roi.npy")


def assert_no_holdout(pids, sids, hp, hs):
    ip, is_ = set(pids) & set(hp), set(sids) & set(hs)
    if ip or is_:
        raise RuntimeError("HOLDOUT LEAK -> BLOCKED pid=%s sid=%s" % (sorted(ip)[:5], sorted(is_)[:5]))
    return True


# ----------------------------------------------------------------------------
# 한글 폰트 해결 함수 (v2 신규)
# ----------------------------------------------------------------------------
def resolve_korean_font():
    """FONT_CANDIDATES 순서대로 시도. 성공 시 dict 반환. 전부 실패 시 RuntimeError."""
    import matplotlib.font_manager as fm
    attempted = []
    failed = []
    for path in FONT_CANDIDATES:
        attempted.append(path)
        if not os.path.isfile(path):
            failed.append(path + " (file not found)")
            continue
        try:
            fm.fontManager.addfont(path)
            family = fm.FontProperties(fname=path).get_name()
            return {
                "selected_font_path": path,
                "selected_font_family": family,
                "attempted_fonts": attempted,
                "failed_fonts": failed,
            }
        except Exception as e:
            failed.append(path + " (%s)" % str(e)[:80])
    raise RuntimeError(
        "[BLOCKED] 한글 폰트 후보 전부 실패. "
        "설치 필요: NanumGothic (sudo apt install fonts-nanum) 또는 Noto Sans CJK KR. "
        "tried=%s failed=%s" % (attempted, failed)
    )


# ----------------------------------------------------------------------------
# 타이틀 / suptitle / annotation 빌더 (v1과 동일)
# ----------------------------------------------------------------------------
def _short_safe_id(safe_id, maxlen=SAFE_ID_MAX_LEN):
    s = str(safe_id)
    return (s[:maxlen] + "...") if len(s) > maxlen else s


def build_panel_a_title(zc, position_bin, rank):
    return "A. Whole slice | rank=%s | bin=%s" % (rank, position_bin)


def build_panel_b_title(max_score, threshold):
    return "B. Local crop | score=%.2f / thr=%s" % (float(max_score), threshold)


def build_suptitle(case_id, role, safe_id):
    return "%s | %s | %s" % (case_id, role, _short_safe_id(safe_id))


def _overmerge_annotation_text():
    return OVERMERGE_ANNOTATION_EN + "  " + OVERMERGE_ANNOTATION_KO


def _apex_caution_text():
    return APEX_CAUTION_TEXT_EN + "  " + APEX_CAUTION_TEXT_KO


# ----------------------------------------------------------------------------
# 순수 계산 (v1과 동일)
# ----------------------------------------------------------------------------
def bbox_area(b):
    return int((b[2] - b[0]) * (b[3] - b[1]))


def display_bbox_from_patch(patch_bbox, margin=DISPLAY_MARGIN):
    y0, x0, y1, x1 = patch_bbox
    return [int(y0) - margin, int(x0) - margin, int(y1) + margin, int(x1) + margin]


def clip_bbox(b, H, W):
    return [max(0, int(b[0])), max(0, int(b[1])), min(int(H), int(b[2])), min(int(W), int(b[3]))]


def area_reduction_ratio(union_area, display_area):
    if union_area <= 0:
        return 0.0
    return round(1.0 - (display_area / union_area), 4)


def overmerge_eval(patch_count, z_span, area):
    pc, z, a = int(patch_count), int(z_span), int(area)
    extreme = (a >= EXTREME_AREA or z >= EXTREME_ZSPAN or pc >= EXTREME_PC)
    large = (a >= LARGE_AREA or z >= LARGE_ZSPAN or pc >= LARGE_PC)
    if extreme:
        return True, "extreme_union", "area=%d z_span=%d patch_count=%d" % (a, z, pc)
    if large:
        return True, "large_union", "area=%d z_span=%d patch_count=%d" % (a, z, pc)
    return False, "none", ""


def scan_forbidden_terms(text):
    low = str(text).lower()
    return [t for t in FORBIDDEN_TERMS if t.lower() in low]


def build_explanation_text(row, overmerge_flag):
    rd = row["role"]
    if rd == ROLE_NORMAL:
        head = ("Normal control / FP review case. 정상 control FP 검토용. "
                "This card is for reviewing false-positive-like high PaDiM response in a normal case.")
    else:
        head = "Stage1-dev candidate."
    contrast = ("같은 위치 bin의 정상 reference와 비교했을 때, 이 후보는 PaDiM 이상 점수가 높게 나타난 영역입니다. "
                "(Compared with normal reference crops from the same position_bin, "
                "this candidate shows a higher PaDiM anomaly score.)")
    meas = ("position_bin=%s, rank=%s, slice=%s~%s, z_span=%s, patch_count=%s, "
            "max_padim_score=%.4f (threshold=%s p95), roi_ratio_mean=%s, %s, %s." % (
                row["position_bin"], row["rank_in_patient"], row["slice_index_min"],
                row["slice_index_max"], row["z_span"], row["patch_count"],
                float(row["max_padim_score"]), THRESHOLD_P95, row["roi_0_0_patch_ratio_mean"],
                row["central_peripheral"], row["left_right_metadata"]))
    om = (" " + OVERMERGE_CAUTION_KO + " " + OVERMERGE_CAUTION_EN) if overmerge_flag else ""
    return "%s %s %s%s" % (head, contrast, meas, om)


def build_fp_caution_text(row, overmerge_flag):
    parts = []
    if row["role"] == ROLE_NORMAL:
        parts.append("정상 control / structural FP 검토용 (not an abnormality).")
    if str(row["central_peripheral"]).strip().endswith("peripheral") or \
            str(row["position_bin"]).endswith("peripheral"):
        parts.append("경계/흉막 인접 가능성 (boundary/pleura-adjacent possible).")
    if overmerge_flag:
        parts.append(OVERMERGE_CAUTION_KO)
    return " ".join(parts) if parts else "(no specific caution)"


def window_to_uint8(hu, center=LUNG_WINDOW_CENTER, width=LUNG_WINDOW_WIDTH):
    lo, hi = center - width / 2.0, center + width / 2.0
    x = (np.asarray(hu, dtype=np.float32) - lo) / (hi - lo)
    return (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def z_context_slices(z, depth):
    return [s for s in (int(z) - 1, int(z), int(z) + 1) if 0 <= s < int(depth)]


def mask_contour(mask_bool):
    m = np.asarray(mask_bool, dtype=bool)
    er = np.zeros_like(m)
    if m.shape[0] >= 3 and m.shape[1] >= 3:
        er[1:-1, 1:-1] = (m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1]
                          & m[1:-1, :-2] & m[1:-1, 2:])
    return m & (~er)


def select_reference_crops(ref_rows, position_bin, k=REF_CROP_MAX):
    same = [r for r in ref_rows if r.get("position_bin") == position_bin]
    return same[:k]


def pick_max_score_patch(patches, max_score_slice, max_score, union_bbox):
    cands = [p for p in patches
             if p["slice"] == int(max_score_slice) and abs(p["score"] - float(max_score)) <= SCORE_TOL]
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    uy0, ux0, uy1, ux1 = union_bbox
    def inside(p):
        return uy0 <= p["y0"] and ux0 <= p["x0"] and p["y1"] <= uy1 and p["x1"] <= ux1
    ins = [p for p in cands if inside(p)]
    pool = ins if ins else cands
    ucy, ucx = (uy0 + uy1) / 2.0, (ux0 + ux1) / 2.0
    return min(pool, key=lambda p: ((p["y0"] + p["y1"]) / 2 - ucy) ** 2
               + ((p["x0"] + p["x1"]) / 2 - ucx) ** 2)


# ----------------------------------------------------------------------------
# read-only 로더 (v1과 동일)
# ----------------------------------------------------------------------------
def load_holdout_denylist():
    hp, hs = set(), set()
    with open(safe_path(SPLIT_CSV), "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        r.fieldnames = [c.strip() for c in r.fieldnames]
        for row in r:
            if (row.get("stage_split") or "").strip() == "stage2_holdout":
                if row.get("patient_id"):
                    hp.add(row["patient_id"].strip())
                if row.get("safe_id"):
                    hs.add(row["safe_id"].strip())
    return hp, hs


def load_manifest():
    with open(safe_path(MANIFEST_CSV), "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_reference_crop_rows():
    with open(safe_path(REF_CROP_MANIFEST), "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_expansion_patches(safe_ids):
    want = set(safe_ids)
    out = {}
    with open(safe_path(PATCH_CSV), "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sid = row["safe_id"].strip()
            if sid not in want:
                continue
            out.setdefault((sid, row["position_bin"]), []).append({
                "slice": int(row["slice_index"]), "score": float(row["padim_score"]),
                "y0": int(row["y0"]), "x0": int(row["x0"]),
                "y1": int(row["y1"]), "x1": int(row["x1"]),
                "ps": int(row["patch_size"])})
    return out


# ----------------------------------------------------------------------------
# 카드 생성 (--run-cards --confirm-generate; 이번 단계 미실행)
# ----------------------------------------------------------------------------
def _generate_cards(out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    # ── 한글 폰트 설정 (v2 신규) ──────────────────────────────
    font_info = resolve_korean_font()
    plt.rcParams['font.family'] = font_info["selected_font_family"]

    png_dir = os.path.join(out_dir, "cards_png")
    json_dir = os.path.join(out_dir, "cards_json")
    index_path = os.path.join(out_dir, "index_cards.csv")
    runtime_path = os.path.join(out_dir, "runtime_summary.json")
    errors_path = os.path.join(out_dir, "errors.csv")
    done_path = os.path.join(out_dir, "DONE.json")

    if os.path.exists(safe_path(done_path)):
        sys.stderr.write("[BLOCKED] DONE.json 존재: %s\n" % done_path)
        return EXIT_BLOCKED
    if os.path.isdir(safe_path(out_dir)):
        leftovers = [p for p in (index_path, runtime_path, errors_path) if os.path.exists(p)]
        for d in (png_dir, json_dir):
            if os.path.isdir(d) and any(os.scandir(safe_path(d))):
                leftovers.append(d)
        if leftovers:
            sys.stderr.write("[BLOCKED] 잔여 산출물 존재: %s\n" % leftovers)
            return EXIT_BLOCKED

    hp, hs = load_holdout_denylist()
    rows_all = load_manifest()

    # ── smoke 필터 (v2 신규) ───────────────────────────────────
    rows = [r for r in rows_all if r["expansion_case_id"].strip() in SMOKE_IDS]
    if len(rows) != N_SMOKE_EXPECTED:
        sys.stderr.write("[BLOCKED] smoke 대상 %d장 기대, manifest에서 %d장 발견\n"
                         % (N_SMOKE_EXPECTED, len(rows)))
        return EXIT_BLOCKED
    for r in rows:
        cid = r["expansion_case_id"].strip()
        if cid not in SMOKE_IDS:
            sys.stderr.write("[BLOCKED] smoke 외 case_id BLOCKED: %s\n" % cid)
            return EXIT_BLOCKED

    ref_rows = load_reference_crop_rows()
    pids = set(r["patient_id"].strip() for r in rows)
    sids = set(r["safe_id"].strip() for r in rows)
    assert_no_holdout(pids, sids, hp, hs)
    patches_by_key = load_expansion_patches(sids)

    os.makedirs(safe_path(png_dir), exist_ok=True)
    os.makedirs(safe_path(json_dir), exist_ok=True)

    errors, index_rows = [], []
    started = datetime.now()
    vol_cache = {}
    n_ok = 0

    def _load_vol(safe_id, role):
        if safe_id not in vol_cache:
            ct = np.load(safe_path(resolve_ct_path(safe_id, role)), mmap_mode="r")
            mask = np.load(safe_path(resolve_mask_path(safe_id, role)), mmap_mode="r")
            vol_cache[safe_id] = (ct, mask)
        return vol_cache[safe_id]

    for row in rows:
        cid = row["expansion_case_id"]
        role = row["role"]
        safe_id = row["safe_id"].strip()
        png_rel = os.path.join("cards_png", "%s.png" % cid)
        json_rel = os.path.join("cards_json", "%s.json" % cid)
        png_abs = os.path.join(out_dir, png_rel)
        json_abs = os.path.join(out_dir, json_rel)
        try:
            uy0, ux0, uy1, ux1 = int(row["y0"]), int(row["x0"]), int(row["y1"]), int(row["x1"])
            union_bbox = [uy0, ux0, uy1, ux1]
            comp_area = bbox_area(union_bbox)
            pc = int(row["patch_count"])
            zsp = int(row["z_span"])
            om_flag = row["overmerge_flag"].strip() == "True"
            om_level = row["overmerge_level"].strip()
            om_reason = row["overmerge_reason"].strip()
            apex_c = row["apex_caution"].strip() == "True"
            apex_reason = row["apex_caution_reason"].strip()

            key = (safe_id, row["position_bin"])
            mxp = pick_max_score_patch(patches_by_key.get(key, []),
                                       row["max_score_slice_index"], row["max_padim_score"], union_bbox)
            if mxp is None:
                raise RuntimeError("max-score patch not found for %s" % cid)
            disp = display_bbox_from_patch([mxp["y0"], mxp["x0"], mxp["y1"], mxp["x1"]])

            ct, mask = _load_vol(safe_id, role)
            depth, H, W = ct.shape[0], ct.shape[1], ct.shape[2]
            zc = min(max(int(row["max_score_slice_index"]), 0), depth - 1)
            disp_c = clip_bbox(disp, H, W)
            disp_area = bbox_area(disp_c)
            red = area_reduction_ratio(comp_area, disp_area)
            zs = z_context_slices(zc, depth)

            ref_sel = select_reference_crops(ref_rows, row["position_bin"])

            expl = build_explanation_text(row, om_flag)
            caution = build_fp_caution_text(row, om_flag)
            bad = scan_forbidden_terms(expl) + scan_forbidden_terms(caution)
            if bad:
                raise RuntimeError("forbidden diagnostic term: %s" % bad)

            # ---- 4-panel ----
            fig, axes = plt.subplots(2, 2, figsize=(11, 11))
            base = window_to_uint8(np.asarray(ct[zc]))
            axA = axes[0, 0]
            axA.imshow(base, cmap="gray")
            axA.axis("off")
            axA.set_title(build_panel_a_title(zc, row["position_bin"], row["rank_in_patient"]))
            axA.add_patch(plt.Rectangle((ux0, uy0), ux1 - ux0, uy1 - uy0, fill=False,
                                        edgecolor="yellow", lw=0.8, linestyle="--", alpha=0.7))
            axA.add_patch(plt.Rectangle((disp_c[1], disp_c[0]), disp_c[3] - disp_c[1], disp_c[2] - disp_c[0],
                                        fill=False, edgecolor="red", lw=2.0))
            if om_flag:
                axA.text(0.01, 0.01, _overmerge_annotation_text(),
                         transform=axA.transAxes, fontsize=6, color="darkorange",
                         verticalalignment="bottom",
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                   alpha=0.75, edgecolor="orange"))
            if apex_c:
                axA.text(0.01, 0.99, _apex_caution_text(),
                         transform=axA.transAxes, fontsize=6, color="navy",
                         verticalalignment="top",
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                   alpha=0.75, edgecolor="navy"))
            crop_hu = np.asarray(ct[zc, disp_c[0]:disp_c[2], disp_c[1]:disp_c[3]])
            crop_rgb = np.stack([window_to_uint8(crop_hu)] * 3, axis=-1)
            cont = mask_contour(np.asarray(mask[zc, disp_c[0]:disp_c[2], disp_c[1]:disp_c[3]]).astype(bool))
            for ch, v in enumerate(CONTOUR_RGB):
                crop_rgb[..., ch][cont] = v
            axB = axes[0, 1]
            axB.imshow(crop_rgb)
            axB.axis("off")
            axB.set_title(build_panel_b_title(float(row["max_padim_score"]), THRESHOLD_P95))
            axC = axes[1, 0]
            axC.axis("off")
            axC.set_title("C. normal reference (%s)  n=%d" % (row["position_bin"], len(ref_sel)))
            ref_imgs = []
            for rr in ref_sel:
                rp = os.path.join(REF_BANK_FULL, rr.get("crop_png_path", ""))
                if os.path.isfile(rp):
                    ref_imgs.append(plt.imread(rp))
            if ref_imgs:
                concat = np.concatenate([np.asarray(im)[:, :, :3] if np.asarray(im).ndim == 3
                                         else np.stack([im] * 3, -1) for im in ref_imgs], axis=1)
                axC.imshow(concat)
            axD = axes[1, 1]
            axD.axis("off")
            axD.set_title("D. z-context %s" % zs)
            if zs:
                zimgs = [window_to_uint8(np.asarray(ct[s, disp_c[0]:disp_c[2], disp_c[1]:disp_c[3]])) for s in zs]
                axD.imshow(np.concatenate(zimgs, axis=1), cmap="gray")

            fig.suptitle(build_suptitle(cid, role, safe_id), fontsize=10)
            fig.tight_layout()
            fig.savefig(safe_path(png_abs), dpi=110)
            plt.close(fig)

            z_ratio_val = float(row["z_ratio"]) if row.get("z_ratio") else None
            card = {
                "expansion_case_id": cid,
                "prototype_role": row["prototype_role"],
                "patient_id": row["patient_id"],
                "safe_id": safe_id,
                "component_id": row["component_id"],
                "rank_in_patient": int(row["rank_in_patient"]),
                "position_bin": row["position_bin"],
                "slice_index_min": int(row["slice_index_min"]),
                "slice_index_max": int(row["slice_index_max"]),
                "max_score_slice_index": int(row["max_score_slice_index"]),
                "z_span": zsp,
                "z_ratio": z_ratio_val,
                "component_union_bbox": [int(v) for v in union_bbox],
                "display_bbox": [int(v) for v in disp_c],
                "display_bbox_mode": DISPLAY_BBOX_MODE,
                "display_bbox_margin": int(DISPLAY_MARGIN),
                "display_bbox_area": int(disp_area),
                "component_bbox_area": int(comp_area),
                "display_bbox_area_reduction_ratio": float(red),
                "overmerge_flag": bool(om_flag),
                "overmerge_level": om_level,
                "overmerge_reason": om_reason,
                "overmerge_caution_text": (OVERMERGE_CAUTION_KO + " " + OVERMERGE_CAUTION_EN) if om_flag else "",
                "apex_caution": bool(apex_c),
                "apex_caution_reason": apex_reason,
                "apex_caution_text": (APEX_CAUTION_TEXT_KO + " " + APEX_CAUTION_TEXT_EN) if apex_c else "",
                "patch_count": pc,
                "max_padim_score": float(row["max_padim_score"]),
                "mean_padim_score": float(row["mean_padim_score"]),
                "threshold": float(THRESHOLD_P95),
                "threshold_type": THRESHOLD_TYPE,
                "roi_0_0_patch_ratio_mean": (float(row["roi_0_0_patch_ratio_mean"])
                                             if row["roi_0_0_patch_ratio_mean"] else None),
                "central_peripheral": row["central_peripheral"],
                "central_distance_ratio_mean": (float(row["central_distance_ratio_mean"])
                                                if row.get("central_distance_ratio_mean") else None),
                "left_right_metadata": row["left_right_metadata"],
                "internal_use_only": True,
                "font_fix_required_before_external_share": True,
                "normal_reference_crops": [rr.get("crop_png_path", "") for rr in ref_sel],
                "card_png_path": png_rel,
                "explanation_text": expl,
                "fp_caution_text": caution,
                "stage_split_safety_flag": row["stage_split_safety_flag"],
                "diagnostic_terms_blocked": list(FORBIDDEN_TERMS),
                "diagnostic_guard_passed": bool(not bad),
            }
            with open(safe_path(json_abs), "w", encoding="utf-8") as jf:
                json.dump(card, jf, ensure_ascii=False, indent=2)

            index_rows.append({
                "expansion_case_id": cid,
                "prototype_role": row["prototype_role"],
                "patient_id": row["patient_id"],
                "safe_id": safe_id,
                "component_id": row["component_id"],
                "rank_in_patient": int(row["rank_in_patient"]),
                "position_bin": row["position_bin"],
                "max_padim_score": float(row["max_padim_score"]),
                "threshold": float(THRESHOLD_P95),
                "z_span": zsp,
                "patch_count": pc,
                "display_bbox_mode": DISPLAY_BBOX_MODE,
                "overmerge_flag": bool(om_flag),
                "overmerge_level": om_level,
                "apex_caution": bool(apex_c),
                "display_bbox_area_reduction_ratio": float(red),
                "internal_use_only": True,
                "font_fix_required_before_external_share": True,
                "card_png_path": png_rel,
                "card_json_path": json_rel,
                "status": "ok",
                "error": "",
            })
            n_ok += 1

        except Exception as e:
            errors.append({"expansion_case_id": cid, "safe_id": safe_id,
                           "stage": "card", "detail": str(e)[:300]})
            index_rows.append({
                "expansion_case_id": cid,
                "prototype_role": row.get("prototype_role", ""),
                "patient_id": row.get("patient_id", ""),
                "safe_id": safe_id,
                "component_id": row.get("component_id", ""),
                "rank_in_patient": row.get("rank_in_patient", ""),
                "position_bin": row.get("position_bin", ""),
                "max_padim_score": row.get("max_padim_score", ""),
                "threshold": float(THRESHOLD_P95),
                "z_span": row.get("z_span", ""),
                "patch_count": row.get("patch_count", ""),
                "display_bbox_mode": DISPLAY_BBOX_MODE,
                "overmerge_flag": "",
                "overmerge_level": "",
                "apex_caution": "",
                "display_bbox_area_reduction_ratio": "",
                "internal_use_only": True,
                "font_fix_required_before_external_share": True,
                "card_png_path": "",
                "card_json_path": "",
                "status": "error",
                "error": str(e)[:200],
            })

    with open(safe_path(index_path), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(INDEX_FIELDS))
        w.writeheader()
        for r in index_rows:
            w.writerow(r)
    with open(safe_path(errors_path), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["expansion_case_id", "safe_id", "stage", "detail"])
        w.writeheader()
        for e in errors:
            w.writerow(e)

    summary = {
        "mode": "s3_expansion_cards_v2_fontfix_smoke",
        "display_bbox_mode": DISPLAY_BBOX_MODE,
        "n_smoke_expected": N_SMOKE_EXPECTED,
        "smoke_ids": sorted(SMOKE_IDS),
        "n_manifest_rows_total": len(rows_all),
        "n_smoke_rows": len(rows),
        "n_cards_ok": n_ok,
        "n_errors": len(errors),
        "n_overmerge_flagged": sum(1 for r in index_rows if r.get("overmerge_flag") is True),
        "n_apex_caution": sum(1 for r in index_rows if r.get("apex_caution") is True),
        "threshold": float(THRESHOLD_P95),
        "threshold_type": THRESHOLD_TYPE,
        "selected_font_path": font_info["selected_font_path"],
        "selected_font_family": font_info["selected_font_family"],
        "holdout_intersection": 0,
        "unique_volumes": len(vol_cache),
        "source_manifest": os.path.relpath(MANIFEST_CSV, REPO),
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "done": True,
    }
    with open(safe_path(runtime_path), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(safe_path(done_path), "w", encoding="utf-8") as f:
        json.dump({"done": True, "summary": summary}, f, ensure_ascii=False, indent=2)

    print("[run-cards smoke v2] 완료. cards_ok=%d errors=%d font=%s -> %s"
          % (n_ok, len(errors), font_info["selected_font_family"], out_dir))
    return EXIT_OK


# ----------------------------------------------------------------------------
# 모드
# ----------------------------------------------------------------------------
def mode_dry_run():
    print("[MODE] --dry-run (입력 read-only + 출력 계획; npy 미열람)")
    ok = True
    def chk(n, c, e=""):
        nonlocal ok
        ok = ok and bool(c)
        print("  [%s] %s %s" % ("OK" if c else "MISS", n, e))
    chk("manifest", os.path.isfile(safe_path(MANIFEST_CSV)))
    chk("patient_summary", os.path.isfile(safe_path(PATIENT_SUMMARY_CSV)))
    chk("patch_candidates", os.path.isfile(safe_path(PATCH_CSV)))
    chk("reference crop manifest", os.path.isfile(safe_path(REF_CROP_MANIFEST)))
    chk("split", os.path.isfile(safe_path(SPLIT_CSV)))
    chk("expansion v1 output 보존(존재)", os.path.isdir(V1_EXPANSION_OUT_DIR))
    chk("prototype v1 output 보존(존재)", os.path.isdir(V1_OUT_DIR))
    chk("prototype v2 output 보존(존재)", os.path.isdir(V2_OUT_DIR))
    chk("prototype v3 output 보존(존재)", os.path.isdir(V3_OUT_DIR))
    chk("smoke DONE 부재", not os.path.exists(os.path.join(OUT_DIR, "DONE.json")),
        "(존재시 BLOCKED)")
    print("  [FONT] 후보 %d개:" % len(FONT_CANDIDATES))
    for i, fp in enumerate(FONT_CANDIDATES):
        exists = os.path.isfile(fp)
        print("    [%d] %s -> %s" % (i, fp, "EXISTS" if exists else "MISSING"))
    print("  [PLAN] smoke 대상 %d장: %s" % (len(SMOKE_IDS), sorted(SMOKE_IDS)))
    print("  [PLAN] 출력:", os.path.relpath(OUT_DIR, REPO), " display_bbox_mode=", DISPLAY_BBOX_MODE)
    for a in PLANNED_ARTIFACTS:
        print("     -", a)
    return EXIT_OK if ok else EXIT_FAIL


def mode_plan_only():
    print("[MODE] --plan-only (smoke 6장 CT/mask path 존재만; npy 미열람)")
    rows_all = load_manifest()
    rows = [r for r in rows_all if r["expansion_case_id"].strip() in SMOKE_IDS]
    hp, hs = load_holdout_denylist()
    pids = set(r["patient_id"].strip() for r in rows)
    sids = set(r["safe_id"].strip() for r in rows)
    print("  smoke 대상:", len(rows), "(기대 %d)" % N_SMOKE_EXPECTED)
    try:
        assert_no_holdout(pids, sids, hp, hs)
        inter = 0
    except RuntimeError as e:
        print("  [BLOCKED]", e)
        return EXIT_BLOCKED
    ct_ok = mask_ok = 0
    for r in rows:
        sid, role = r["safe_id"].strip(), r["role"]
        if os.path.isfile(safe_path(resolve_ct_path(sid, role))):
            ct_ok += 1
        if os.path.isfile(safe_path(resolve_mask_path(sid, role))):
            mask_ok += 1
    print("  CT path 존재:", ct_ok, "/", len(rows))
    print("  mask path 존재:", mask_ok, "/", len(rows))
    print("  holdout 교집합:", inter)
    print("  font candidates:")
    for fp in FONT_CANDIDATES:
        print("    %s -> %s" % (fp, "EXISTS" if os.path.isfile(fp) else "MISSING"))
    print("  expansion v1 output 보존:", os.path.isdir(V1_EXPANSION_OUT_DIR))
    print("  smoke IDs:", sorted(SMOKE_IDS))
    ok = (len(rows) == N_SMOKE_EXPECTED
          and ct_ok == len(rows) and mask_ok == len(rows) and inter == 0)
    print("  [%s] plan readiness" % ("PASS" if ok else "CHECK"))
    return EXIT_OK if ok else EXIT_FAIL


def mode_selftest():
    print("[MODE] --selftest")
    results = []
    def expect(n, c):
        results.append(bool(c))
        print("  [%s] %s" % ("PASS" if c else "FAIL", n))

    # ── v2 smoke 전용 확인 ──────────────────────────────────────
    # S1. smoke 대상 수 = 6
    expect("S1. SMOKE_IDS 수 = 6", len(SMOKE_IDS) == 6)

    # S2. HOLD_IDS 3건 모두 SMOKE_IDS에 포함
    expect("S2. HOLD_IDS ⊆ SMOKE_IDS", HOLD_IDS.issubset(SMOKE_IDS))
    expect("S2b. HOLD_IDS 수 = 3", len(HOLD_IDS) == 3)

    # S3. FONT_CANDIDATES 비어 있지 않음
    expect("S3. FONT_CANDIDATES 비어 있지 않음", len(FONT_CANDIDATES) > 0)

    # S4. PRIMARY 폰트 경로 = malgun.ttf
    expect("S4. PRIMARY = malgun.ttf", "malgun.ttf" in FONT_CANDIDATES[0])

    # S5. fallback 순서 보장 (리스트 순서)
    expect("S5. fallback 2종 존재", len(FONT_CANDIDATES) >= 2)

    # S6. resolve_korean_font 소스: addfont 호출
    src_font = inspect.getsource(resolve_korean_font)
    expect("S6a. addfont 호출", "addfont(" in src_font)
    expect("S6b. get_name() 호출", "get_name()" in src_font)
    expect("S6c. RuntimeError 발생 (전부 실패 시)", "RuntimeError" in src_font)
    expect("S6d. DejaVu로 조용히 넘어가지 않음", "DejaVu" not in src_font)

    # S7. N_SMOKE_EXPECTED = 6 (full 300 옵션 없음)
    expect("S7. N_SMOKE_EXPECTED == 6", N_SMOKE_EXPECTED == 6)
    expect("S7b. full 300 상수 없음 (N_EXPECTED_ROWS 미정의)", not hasattr(sys.modules[__name__], "N_EXPECTED_ROWS"))

    # S8. OUT_DIR에 smoke 포함
    expect("S8. OUT_DIR에 fontfix_smoke 포함", "fontfix_smoke" in OUT_DIR)

    # S9. expansion v1 output 보존 확인
    expect("S9. V1_EXPANSION_OUT_DIR 존재", os.path.isdir(V1_EXPANSION_OUT_DIR))

    # S10. smoke 필터가 _generate_cards에 존재
    src_gen = inspect.getsource(_generate_cards)
    expect("S10a. smoke 필터 (SMOKE_IDS 필터링)", "SMOKE_IDS" in src_gen)
    expect("S10b. N_SMOKE_EXPECTED 체크", "N_SMOKE_EXPECTED" in src_gen)
    expect("S10c. smoke 외 case_id BLOCKED", "smoke 외 case_id BLOCKED" in src_gen)

    # S11. font 설정이 _generate_cards에 존재
    expect("S11a. resolve_korean_font() 호출", "resolve_korean_font()" in src_gen)
    expect("S11b. rcParams font.family 설정", "rcParams['font.family']" in src_gen)

    # S12. runtime_summary에 font 정보 포함
    expect("S12a. selected_font_path in summary", '"selected_font_path"' in src_gen)
    expect("S12b. selected_font_family in summary", '"selected_font_family"' in src_gen)
    expect("S12c. mode = v2_fontfix_smoke", '"s3_expansion_cards_v2_fontfix_smoke"' in src_gen)

    # ── 공통 확인 (v1에서 유지) ──────────────────────────────────
    # 1. bare guard
    bare_blocked = "EXIT_BLOCKED" in inspect.getsource(main) and "run_cards" in inspect.getsource(main)
    expect("1. bare guard (main에 EXIT_BLOCKED 존재)", bare_blocked)

    # 2. run-cards confirm guard
    src_run = inspect.getsource(mode_run_cards)
    expect("2. run-cards confirm guard", "confirm_generate" in src_run and "EXIT_BLOCKED" in src_run)

    # 3. expansion output DONE/잔여 guard
    expect("3. expansion DONE 가드", "DONE.json 존재" in src_gen and "잔여 산출물" in src_gen)

    # 4~6. prototype output 보존 확인
    expect("4. prototype v1 output 보존", os.path.isdir(V1_OUT_DIR))
    expect("5. prototype v2 output 보존", os.path.isdir(V2_OUT_DIR))
    expect("6. prototype v3 output 보존", os.path.isdir(V3_OUT_DIR))

    # 9. holdout denylist 교집합 차단
    blk = False
    try:
        assert_no_holdout({"H"}, set(), {"H"}, set())
    except RuntimeError:
        blk = True
    expect("9. holdout denylist 교집합 차단", blk)
    expect("9b. holdout 무교집합 통과", assert_no_holdout({"P"}, {"S"}, {"H"}, {"HS"}) is True)

    # 10~11. JSON 필드 유지
    expect("10. internal_use_only in CARD_JSON_FIELDS", "internal_use_only" in CARD_JSON_FIELDS)
    expect("11. font_fix_required_before_external_share in CARD_JSON_FIELDS",
           "font_fix_required_before_external_share" in CARD_JSON_FIELDS)
    expect("12. overmerge_flag in CARD_JSON_FIELDS", "overmerge_flag" in CARD_JSON_FIELDS)
    expect("13. apex_caution in CARD_JSON_FIELDS", "apex_caution" in CARD_JSON_FIELDS)

    # 14~15. Panel title 유지
    pa = build_panel_a_title(512, "lower_peripheral", 9)
    pb = build_panel_b_title(999.99, THRESHOLD_P95)
    expect("14a. Panel A title ≤50자", len(pa) <= 50)
    expect("14b. Panel B title ≤60자", len(pb) <= 60)
    expect("15a. Panel A title에 broad/광역 없음",
           "broad" not in pa.lower() and "광역" not in pa and "caution" not in pa.lower())
    expect("15b. Panel B title에 mode 없음", DISPLAY_BBOX_MODE not in pb)

    # 16~17. overmerge/apex annotation 유지
    expect("16. axA.text overmerge annotation 존재", "axA.text(" in src_gen)
    expect("16b. _overmerge_annotation_text() 호출", "_overmerge_annotation_text()" in src_gen)
    expect("17. _apex_caution_text() 호출", "_apex_caution_text()" in src_gen)

    # 18. caution 문구 금지어 0
    ann_om = _overmerge_annotation_text()
    ann_apex = _apex_caution_text()
    expect("18a. overmerge annotation 금지어 0", scan_forbidden_terms(ann_om) == [])
    expect("18b. apex caution annotation 금지어 0", scan_forbidden_terms(ann_apex) == [])

    # 19. B2 display_bbox 유지
    disp = display_bbox_from_patch([100, 100, 132, 132], 32)
    expect("19. display bbox B2+margin32", disp == [68, 68, 164, 164])
    expect("19b. bbox clip 범위", clip_bbox([-10, -10, 1000, 1000], 512, 512) == [0, 0, 512, 512])

    # 20~21. overmerge_flag 기준 유지
    f1, l1, _ = overmerge_eval(4160, 96, 101376)
    expect("20. extreme_union", f1 and l1 == "extreme_union")
    f2, l2, _ = overmerge_eval(300, 35, 30000)
    expect("21a. large_union(z>=30)", f2 and l2 == "large_union")
    f3, l3, _ = overmerge_eval(50, 5, 5000)
    expect("21b. non-overmerge", (not f3) and l3 == "none")

    # 22. JSON numeric casting 유지
    expect("22a. int slice_index_min 캐스팅", "int(row[\"slice_index_min\"])" in src_gen)
    expect("22b. float max_padim_score 캐스팅", "float(row[\"max_padim_score\"])" in src_gen)
    expect("22c. bool overmerge_flag 캐스팅", "bool(om_flag)" in src_gen)

    # 23. diagnostic guard 유지
    expect("23a. forbidden term 검출", "adenocarcinoma" in scan_forbidden_terms("x adenocarcinoma y"))
    expect("23b. clean 통과", scan_forbidden_terms("광역 component 반응") == [])
    expect("23c. diagnostic_terms_blocked list", "list(FORBIDDEN_TERMS)" in src_gen)

    # 24. same-bin reference 유지
    refs = [{"position_bin": "upper_central", "crop_png_path": "a"},
            {"position_bin": "lower_central", "crop_png_path": "b"},
            {"position_bin": "upper_central", "crop_png_path": "c"},
            {"position_bin": "upper_central", "crop_png_path": "d"}]
    sel = select_reference_crops(refs, "upper_central")
    expect("24. same-bin reference 최대3", len(sel) == 3 and all(r["position_bin"] == "upper_central" for r in sel))

    # 25. z-context clipping 유지
    expect("25. z-context 클립", z_context_slices(0, 5) == [0, 1] and z_context_slices(10, 50) == [9, 10, 11])

    # 26. lung window 변환 유지
    w = window_to_uint8(np.array([[-1350.0, 150.0]]))
    expect("26. lung window uint8", w.dtype == np.uint8 and w[0, 0] == 0 and w[0, 1] == 255)

    # 27. mask contour 유지
    m = np.ones((4, 4), bool)
    cont = mask_contour(m)
    expect("27. mask contour 경계만", bool(cont[0, 0]) and (not bool(cont[1, 1])))

    # 28. run-cards _generate_cards 연결
    expect("28. run-cards _generate_cards 연결", "_generate_cards(OUT_DIR)" in src_run)

    # 29~30. npy load 위치 확인
    src_dry = inspect.getsource(mode_dry_run)
    src_plan = inspect.getsource(mode_plan_only)
    expect("29. np.load mmap 생성부에만", 'np.load(' in src_gen
           and 'np.load(' not in src_dry and 'np.load(' not in src_plan)
    expect("30. mmap_mode=\"r\" 생성부에만", 'mmap_mode="r"' in src_gen)

    # 31. output 경로 구분
    expect("31. OUT_DIR != V1_EXPANSION_OUT_DIR", OUT_DIR != V1_EXPANSION_OUT_DIR)
    expect("31b. OUT_DIR에 fontfix_smoke 포함", "fontfix_smoke" in OUT_DIR)

    # 32. FORBIDDEN_PATH_TOKENS에 stage2_holdout
    expect("32. FORBIDDEN_PATH_TOKENS에 stage2_holdout", "stage2_holdout" in FORBIDDEN_PATH_TOKENS)

    # 33. ROLE 상수 유지
    expect("33. ROLE_NORMAL == normal_control", ROLE_NORMAL == "normal_control")
    expect("33b. ROLE_LESION == lesion_candidate", ROLE_LESION == "lesion_candidate")

    # 34. index schema 포함
    required_idx = ("expansion_case_id", "prototype_role", "overmerge_flag", "apex_caution",
                    "internal_use_only", "font_fix_required_before_external_share",
                    "card_png_path", "card_json_path", "status", "error")
    expect("34. index schema 필수 필드 포함", all(k in INDEX_FIELDS for k in required_idx))

    # 35. runtime_summary 포함 항목
    expect("35a. runtime_summary n_smoke_rows", '"n_smoke_rows"' in src_gen)
    expect("35b. runtime_summary n_cards_ok", '"n_cards_ok"' in src_gen)
    expect("35c. runtime_summary mode v2", '"s3_expansion_cards_v2_fontfix_smoke"' in src_gen)
    expect("35d. runtime_summary n_apex_caution", '"n_apex_caution"' in src_gen)

    # extra
    expect("extra. lesion GT mask 미사용", "lesion_mask" not in src_gen)
    expect("extra. explanation_text build", "build_explanation_text(" in src_gen)
    expect("extra. fp_caution_text build", "build_fp_caution_text(" in src_gen)
    expect("extra. component_union_bbox JSON", "component_union_bbox" in src_gen and "display_bbox" in src_gen)
    expect("extra. threshold 14.0921 유지", "THRESHOLD_P95" in src_gen)
    expect("extra. smoke 6장 기록", '"smoke_ids"' in src_gen)

    n = sum(1 for x in results if x)
    print("\n[SELFTEST] %d/%d PASS" % (n, len(results)))
    return EXIT_OK if n == len(results) else EXIT_FAIL


def mode_run_cards(confirm_generate):
    if not confirm_generate:
        sys.stderr.write("[BLOCKED] --run-cards는 --confirm-generate 동반 + 사용자 승인 필요.\n")
        return EXIT_BLOCKED
    return _generate_cards(OUT_DIR)


def build_parser():
    p = argparse.ArgumentParser(
        description="Explanation Card S3 Expansion v2 Font-Fix Smoke 카드 생성기 (가드 필수).")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--run-cards", action="store_true")
    p.add_argument("--confirm-generate", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.selftest:
        return mode_selftest()
    if args.dry_run:
        return mode_dry_run()
    if args.plan_only:
        return mode_plan_only()
    if args.run_cards:
        return mode_run_cards(args.confirm_generate)
    sys.stderr.write("[BLOCKED] 가드 플래그 필요: --selftest | --dry-run | --plan-only "
                     "| (--run-cards --confirm-generate)\n")
    return EXIT_BLOCKED


if __name__ == "__main__":
    sys.exit(main())
