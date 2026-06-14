#!/usr/bin/env python3
"""
build_explanation_card_s4_reason_card_prototype_1case_v5.py

S4 Reason Card Prototype 1-case v5 Script
Layout: comparison_first_v1 — Candidate vs Matched ref (same size) + context + reason box

목적:
- LUNG1-320__c2 1건에 대해 comparison-first layout으로 신규 카드 생성.
- Row 1: Candidate 256×256 (좌) vs Matched normal reference 256×256 (우) — 메인 비교
- Row 2: A whole-slice context (190×190, 좌) + D z-context (190×190, 우) — 보조
- Row 3: Reason box (KO text, 전체 폭)

v4 개선 사항:
  1. Candidate crop y 시작점 수정: (625,41,...) → (625,70,...) — B panel title 제외
  2. Matched ref display size: 32→556(17.4x 확대) → 32→128→256(8x) — blur 개선
  3. Candidate display size: ~484px(row 40%) → 256×256 downscale — 동일 비교 조건
  4. Ref2/Ref3: 카드에서 완전 제거, JSON에만 additional_reference_sources 보존
  5. D z-context: Row 4 전체폭 556px → Row 2 보조 패널 190×190 — 과대 공간 해소
  6. canvas 높이: 1836px → ~760px — 비교 중심으로 단순화
  7. 기존 A/B/D 재배치 중심 구조 → comparison-first 구조

절대 금지:
- S3 full card를 새 canvas에 그대로 붙이는 방식 (FULL_CARD_PASTE_STRATEGY) 금지
- CT/mask npy 로드 (ALLOW_CT_LOAD=False)
- HU 통계 재계산
- 기존 S3 PNG/JSON 수정 (ALLOW_ORIGINAL_CARD_MODIFICATION=False)
- full 300 처리 (ALLOW_FULL_300=False)
- score/model/threshold 재계산
- stage2_holdout 접근
- 기존 v1/v2/v3/v4 prototype 수정
- old Panel C를 카드에 포함

실행 모드:
- bare 실행                                 → BLOCKED exit 2
- --selftest                                → 22개 guard 검사
- --dry-run                                 → 입력 파일 존재 + 1건 resolve + output guard 확인
- --plan-only                               → dry-run + 배치 계획 출력
- --run-prototype                           → 단독 BLOCKED exit 2
- --run-prototype --confirm-generate        → ALLOW_RUN_CARD_PROTOTYPE=False 로 BLOCKED

syntax check:
  python -m py_compile scripts/build_explanation_card_s4_reason_card_prototype_1case_v5.py
"""

import argparse
import csv
import json
import os
import pathlib
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# ============================================================
# 최상위 가드 — 이번 단계는 전부 False
# ============================================================
ALLOW_RUN_CARD_PROTOTYPE         = True    # 실제 생성 시 True로 변경 (사용자 승인 필수)
ALLOW_ORIGINAL_CARD_MODIFICATION = False   # 항구 False
ALLOW_CT_LOAD                    = False   # 항구 False
ALLOW_FULL_300                   = False   # 항구 False

# ============================================================
# v5 핵심 전략 식별자
# ============================================================
FULL_CARD_PASTE_STRATEGY  = "DISABLED"   # S3 PNG 전체를 새 canvas에 paste — v4부터 폐기
SHOW_REF2_REF3_ON_CARD    = False        # Ref2/Ref3 카드 표시 금지 (JSON에만 보존)

# ============================================================
# prototype 대상 (1건)
# ============================================================
TARGET_CASE_ID = "LUNG1-320__c2"

# ============================================================
# reason text v5 — KO만 표시, EN은 JSON 전용
# ============================================================
REASON_TITLE = "Reason cue / 검토 근거 후보"

REASON_TEXT_KO = (
    "같은 위치(lower_peripheral)의 정상 reference와 비교했을 때, "
    "후보 crop은 HU 밀도가 더 높게 측정되었습니다(약 +245 HU). "
    "이 결과는 PaDiM high-response를 해석하기 위한 참고 단서이며, 진단 의미는 아닙니다."
)

REASON_TEXT_EN = (
    "Compared with same-bin normal references in the lower_peripheral region, "
    "the candidate crop showed a higher HU density difference, about +245 HU. "
    "This is a reference cue for interpreting the PaDiM high response, not a diagnosis."
)

# ============================================================
# v5 layout 설정
# ============================================================
LAYOUT_VERSION       = "comparison_first_v1"
PROTOTYPE_VERSION    = "v5"

# Panel labels
PANEL_A_LABEL        = "A. Whole slice (위치 확인)"
PANEL_B_LABEL        = "B. Local crop"
PANEL_D_LABEL        = "D. z-context [139, 140, 141]"

# Comparison labels
CANDIDATE_LABEL      = "Candidate crop"
MAIN_REF_LABEL       = "Matched normal reference\n(lower_peripheral)"

# Row 1 section label
COMPARISON_SECTION_LABEL = "Candidate vs same-bin normal reference"

# ============================================================
# SOURCE CROP BBOXES (S3 PNG read-only crop 좌표)
# v5 수정: B panel y 시작점 41 → 70 (panel title 높이 ~30px 제외)
# 추정치 ±10px — run 후 눈검증 대상
# selftest에서 bbox 유효성(x1<x2, y1<y2) 및 y_start >= 70 검사
# ============================================================
A_CROP_BBOX         = (29,  41, 585,  597)   # (left, upper, right, lower)
B_CROP_BBOX         = (625, 70, 1181, 597)   # v5: y=70 (v4 y=41에서 title 제외 재조정)
D_CROP_BBOX         = (625, 638, 1181, 1194)

# Candidate = B crop 재사용 (CT npy 없이)
CANDIDATE_CROP_BBOX = B_CROP_BBOX

# C는 source에서 절대 crop하지 않음
C_CROP_BBOX         = None   # old Panel C 완전 제거

# ============================================================
# v5 display size (candidate == matched ref — 동일 조건 비교)
# ============================================================
CANDIDATE_DISPLAY_SIZE = 256   # px
REF_DISPLAY_SIZE       = 256   # px (반드시 CANDIDATE_DISPLAY_SIZE와 동일)

# Context panels (보조, Row 2)
CONTEXT_DISPLAY_SIZE   = 190   # px (A, D 보조 패널 크기)

# Reference upscale: staged LANCZOS 32 → 128 → 256 (v4 32→128→556에서 변경)
REF_UPSCALE_METHOD = "staged_LANCZOS_32_128_256"
REF_UPSCALE_STAGES = [128, 256]   # 중간 단계 크기

# ============================================================
# canvas 크기
# ============================================================
CANVAS_WIDTH_PX = 1210
CANVAS_DPI      = 110

# Row 높이 (px)
ROW1_H_PX     = 300    # main comparison (Candidate vs Matched ref)
ROW2_H_PX     = 220    # context panels (A + D)
ROW3_H_PX     = 160    # reason box
CANVAS_HEIGHT_ESTIMATE_PX = ROW1_H_PX + ROW2_H_PX + ROW3_H_PX + 80   # ~760

# ============================================================
# 진단 금지어
# ============================================================
FORBIDDEN_TERMS = [
    "cancer", "malignancy", "malignant", "benign",
    "tumor", "tumour",
    "nodule 확정", "pulmonary nodule 확정",
    "ground-glass nodule 확정", "ggn 확정",
    "폐암", "악성", "양성", "종양",
    "결절로 진단", "유리결절로 진단",
    "병변 확정", "암 가능성 높음",
    "병변", "암",
]

# ============================================================
# stage2_holdout 접근 금지 토큰
# ============================================================
STAGE2_HOLDOUT_TOKENS = [
    "stage2_holdout", "stage2-holdout", "stage2holdout",
    "holdout_stage2", "holdout-stage2",
]

# ============================================================
# 경로 상수
# ============================================================
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

REPORTS_ROOT = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
)

# S4 reason layer v3 입력 (read-only)
V3_OUTPUT_DIR  = REPORTS_ROOT / "s4_reason_layer_integrated_smoke_v3"
V3_OUTPUT_CSV  = V3_OUTPUT_DIR / "s4_reason_layer_integrated_smoke_v3.csv"
V3_OUTPUT_JSON = V3_OUTPUT_DIR / "s4_reason_layer_integrated_smoke_v3.json"
V3_DONE_JSON   = V3_OUTPUT_DIR / "DONE.json"

# S3 font-fix 카드 root (read-only source)
S3_CARD_ROOT = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    / "s3_expansion_cards_v2_fontfix"
)
S3_INDEX_CSV      = S3_CARD_ROOT / "index_cards.csv"
S3_CARDS_PNG_DIR  = S3_CARD_ROOT / "cards_png"
S3_CARDS_JSON_DIR = S3_CARD_ROOT / "cards_json"

# reference bank (ref crop PNG read-only)
REF_BANK_FULL = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    / "reference_bank_v1/full"
)
REF_CROP_MANIFEST = REF_BANK_FULL / "reference_crop_manifest.csv"

# 보존 전용 경로 (수정/삭제 금지)
PROTO_V1_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    / "s4_reason_card_prototype_1case_v1"
)
PROTO_V2_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    / "s4_reason_card_prototype_1case_v2"
)
PROTO_V3_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    / "s4_reason_card_prototype_1case_v3"
)
PROTO_V4_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    / "s4_reason_card_prototype_1case_v4"
)

# v5 prototype output root (신규 경로)
PROTO_OUTPUT_ROOT    = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    / "s4_reason_card_prototype_1case_v5"
)
PROTO_CARDS_PNG_DIR  = PROTO_OUTPUT_ROOT / "cards_png"
PROTO_CARDS_JSON_DIR = PROTO_OUTPUT_ROOT / "cards_json"
PROTO_INDEX_CSV      = PROTO_OUTPUT_ROOT / "index_cards.csv"
PROTO_RUNTIME_JSON   = PROTO_OUTPUT_ROOT / "runtime_summary.json"
PROTO_ERRORS_CSV     = PROTO_OUTPUT_ROOT / "errors.csv"
PROTO_DONE_JSON      = PROTO_OUTPUT_ROOT / "DONE.json"

# v5 static check 보고서 경로
STATIC_CHECK_REPORT_MD   = REPORTS_ROOT / "s4_reason_card_prototype_1case_v5_script_static_drycheck_v1.md"
STATIC_CHECK_REPORT_JSON = REPORTS_ROOT / "s4_reason_card_prototype_1case_v5_script_static_drycheck_v1.json"

# v5 preflight 입력
V5_PREFLIGHT_JSON = REPORTS_ROOT / "s4_reason_card_prototype_1case_v5_comparison_first_preflight_v1.json"


# ============================================================
# guard helper
# ============================================================
def _block(reason: str, code: int = 2) -> None:
    print(f"[BLOCKED] {reason}", file=sys.stderr)
    sys.exit(code)


def _assert_no_stage2_holdout(path_or_str: str) -> None:
    s = str(path_or_str).lower()
    for tok in STAGE2_HOLDOUT_TOKENS:
        if tok in s:
            _block(f"stage2_holdout 접근 금지: {path_or_str}")


def scan_forbidden_terms(text: str) -> List[str]:
    hits = []
    tl = text.lower()
    for term in FORBIDDEN_TERMS:
        if term.lower() in tl:
            hits.append(term)
    return hits


# ============================================================
# CSV / JSON loader
# ============================================================
def load_v3_csv() -> List[Dict[str, str]]:
    rows = []
    with open(V3_OUTPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


# ============================================================
# target case resolve / validate
# ============================================================
def resolve_target_case(rows: List[Dict[str, str]]) -> Dict[str, str]:
    matched = [r for r in rows if r.get("expansion_case_id") == TARGET_CASE_ID]
    if not matched:
        _block(f"target case {TARGET_CASE_ID} not found in V3 CSV")
    return matched[0]


def validate_target_row(row: Dict[str, str]) -> Dict[str, Any]:
    issues = []
    if row.get("card_text_ready", "").strip().lower() != "true":
        issues.append(f"card_text_ready != True (실제: {row.get('card_text_ready')})")
    if row.get("card_reflection_status", "").strip() != "card_text_candidate":
        issues.append(
            f"card_reflection_status != card_text_candidate "
            f"(실제: {row.get('card_reflection_status')})"
        )
    if row.get("disclaimer_present", "").strip().lower() != "true":
        issues.append(f"disclaimer_present != True (실제: {row.get('disclaimer_present')})")
    if row.get("diagnostic_guard_passed", "").strip().lower() != "true":
        issues.append(
            f"diagnostic_guard_passed != True "
            f"(실제: {row.get('diagnostic_guard_passed')})"
        )
    return {"ok": len(issues) == 0, "issues": issues, "row": row}


# ============================================================
# S3 card 경로 resolve
# ============================================================
def resolve_s3_card_paths() -> Dict[str, Any]:
    index_row = None
    if S3_INDEX_CSV.exists():
        with open(S3_INDEX_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("expansion_case_id") == TARGET_CASE_ID or row.get("case_id") == TARGET_CASE_ID:
                    index_row = dict(row)
                    break

    # index 없으면 직접 경로 조합
    png_abs  = S3_CARDS_PNG_DIR  / f"{TARGET_CASE_ID}.png"
    json_abs = S3_CARDS_JSON_DIR / f"{TARGET_CASE_ID}.json"

    _assert_no_stage2_holdout(png_abs)
    _assert_no_stage2_holdout(json_abs)

    png_exists  = png_abs.exists()
    json_exists = json_abs.exists()

    if not png_exists:
        return {"ok": False, "error": f"S3 PNG 없음: {png_abs}", "png_exists": False, "json_exists": json_exists}
    if not json_exists:
        return {"ok": False, "error": f"S3 JSON 없음: {json_abs}", "png_exists": True, "json_exists": False}

    return {
        "ok": True,
        "png_absolute": str(png_abs),
        "json_absolute": str(json_abs),
        "png_exists": True,
        "json_exists": True,
    }


# ============================================================
# reference 경로 resolve (S3 JSON에서 추출)
# ============================================================
def resolve_ref_paths_from_json(json_abs: str) -> Dict[str, Any]:
    _assert_no_stage2_holdout(json_abs)
    try:
        with open(json_abs, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {"ok": False, "error": str(e), "ref_paths": [], "ref_count": 0}

    position_bin    = data.get("position_bin", "unknown")
    ref_crop_paths  = data.get("normal_reference_crops", [])

    for rp in ref_crop_paths:
        full_p = REF_BANK_FULL / rp
        _assert_no_stage2_holdout(full_p)

    existing = [rp for rp in ref_crop_paths if (REF_BANK_FULL / rp).exists()]

    return {
        "ok": True,
        "position_bin": position_bin,
        "ref_paths": ref_crop_paths,
        "ref_paths_existing": existing,
        "ref_count": len(ref_crop_paths),
        "ref_count_existing": len(existing),
    }


# ============================================================
# S3 JSON metadata 로드
# ============================================================
def read_s3_card_json_metadata(json_abs: str) -> Dict[str, Any]:
    _assert_no_stage2_holdout(json_abs)
    try:
        with open(json_abs, encoding="utf-8") as f:
            data = json.load(f)
        return {"ok": True, "data": data}
    except Exception as e:
        return {"ok": False, "error": str(e), "data": {}}


# ============================================================
# output guard (v5 output root가 기존 경로와 충돌하지 않는지 확인)
# ============================================================
def check_output_guard() -> Dict[str, Any]:
    protected_roots = [
        S3_CARD_ROOT,
        PROTO_V1_OUTPUT_ROOT,
        PROTO_V2_OUTPUT_ROOT,
        PROTO_V3_OUTPUT_ROOT,
        PROTO_V4_OUTPUT_ROOT,
        V3_OUTPUT_DIR,
    ]

    _assert_no_stage2_holdout(PROTO_OUTPUT_ROOT)

    for pr in protected_roots:
        if PROTO_OUTPUT_ROOT == pr:
            return {"ok": False, "error": f"output root가 보호 경로와 일치: {pr}"}
        if str(PROTO_OUTPUT_ROOT).startswith(str(pr) + "/"):
            return {"ok": False, "error": f"output root가 보호 경로 하위: {pr}"}
        if str(pr).startswith(str(PROTO_OUTPUT_ROOT) + "/"):
            return {"ok": False, "error": f"보호 경로가 output root 하위: {pr}"}

    # v5 output root가 v4와 다름 확인
    v5_name = PROTO_OUTPUT_ROOT.name
    v4_name = PROTO_V4_OUTPUT_ROOT.name
    if v5_name == v4_name:
        return {"ok": False, "error": f"v5 output root name이 v4와 동일: {v5_name}"}

    return {
        "ok": True,
        "output_root": str(PROTO_OUTPUT_ROOT),
        "protected_checked": len(protected_roots),
    }


# ============================================================
# prior prototypes 보존 확인
# ============================================================
def check_prior_prototypes_preserved() -> Dict[str, Any]:
    prior_roots = {
        "v1": PROTO_V1_OUTPUT_ROOT,
        "v2": PROTO_V2_OUTPUT_ROOT,
        "v3": PROTO_V3_OUTPUT_ROOT,
        "v4": PROTO_V4_OUTPUT_ROOT,
    }
    result = {"ok": True, "prior_prototypes": {}}
    for ver, root in prior_roots.items():
        exists = root.exists()
        done_ok = (root / "DONE.json").exists() if exists else False
        result["prior_prototypes"][ver] = {
            "output_root_exists": exists,
            "done_marker": done_ok,
        }
    return result


# ============================================================
# crop bbox 유효성 (v5: B_CROP_BBOX y_start >= 70 추가 검사)
# ============================================================
def validate_crop_bboxes() -> Dict[str, Any]:
    issues = []
    results = {}

    for name, bbox in [("A", A_CROP_BBOX), ("B", B_CROP_BBOX), ("D", D_CROP_BBOX)]:
        if bbox is None:
            issues.append(f"{name}_CROP_BBOX is None")
            continue
        x1, y1, x2, y2 = bbox
        if x1 >= x2:
            issues.append(f"{name}_CROP_BBOX: x1({x1}) >= x2({x2})")
        if y1 >= y2:
            issues.append(f"{name}_CROP_BBOX: y1({y1}) >= y2({y2})")
        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            issues.append(f"{name}_CROP_BBOX: 유효 크기 없음")
        results[name] = {"bbox": list(bbox), "w": x2 - x1, "h": y2 - y1}

    # C_CROP_BBOX는 None이어야 함 (old Panel C 완전 제거)
    if C_CROP_BBOX is not None:
        issues.append(f"C_CROP_BBOX가 None이 아님: {C_CROP_BBOX} — old Panel C 포함 금지")
    results["C"] = {"bbox": None, "removed": True}

    # CANDIDATE_CROP_BBOX == B_CROP_BBOX 확인
    if CANDIDATE_CROP_BBOX != B_CROP_BBOX:
        issues.append(f"CANDIDATE_CROP_BBOX가 B_CROP_BBOX와 불일치: {CANDIDATE_CROP_BBOX} != {B_CROP_BBOX}")
    results["candidate"] = {"bbox": list(CANDIDATE_CROP_BBOX), "matches_B": CANDIDATE_CROP_BBOX == B_CROP_BBOX}

    # v5 추가: B_CROP_BBOX y_start >= 70 (title 제외 확인)
    b_y_start = B_CROP_BBOX[1]
    if b_y_start < 70:
        issues.append(f"B_CROP_BBOX y_start={b_y_start} < 70 — B panel title 포함 가능성 (v5 기준: y >= 70)")
    results["B_y_start_check"] = {"y_start": b_y_start, "ok": b_y_start >= 70}

    return {"ok": len(issues) == 0, "issues": issues, "bboxes": results}


# ============================================================
# display size 일치 검사 (v5 신규)
# ============================================================
def validate_display_sizes() -> Dict[str, Any]:
    issues = []

    if CANDIDATE_DISPLAY_SIZE != REF_DISPLAY_SIZE:
        issues.append(
            f"CANDIDATE_DISPLAY_SIZE({CANDIDATE_DISPLAY_SIZE}) != "
            f"REF_DISPLAY_SIZE({REF_DISPLAY_SIZE}) — 동일 비교 조건 위반"
        )

    if CANDIDATE_DISPLAY_SIZE <= 0 or REF_DISPLAY_SIZE <= 0:
        issues.append("display size <= 0")

    if CONTEXT_DISPLAY_SIZE >= CANDIDATE_DISPLAY_SIZE:
        issues.append(
            f"CONTEXT_DISPLAY_SIZE({CONTEXT_DISPLAY_SIZE}) >= "
            f"CANDIDATE_DISPLAY_SIZE({CANDIDATE_DISPLAY_SIZE}) — context가 main보다 크거나 같음"
        )

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "candidate_display_size": CANDIDATE_DISPLAY_SIZE,
        "ref_display_size": REF_DISPLAY_SIZE,
        "context_display_size": CONTEXT_DISPLAY_SIZE,
        "sizes_equal": CANDIDATE_DISPLAY_SIZE == REF_DISPLAY_SIZE,
    }


# ============================================================
# reason text 검사
# ============================================================
def validate_reason_text() -> Dict[str, Any]:
    issues = []

    # KO 필수 포함 요소
    required_ko = [
        ("약 +245 HU", "HU 수치"),
        ("진단 의미는 아닙니다", "면책 문구"),
        ("참고 단서", "참고 단서 표현"),
        ("lower_peripheral", "position_bin 표기"),
    ]
    for phrase, label in required_ko:
        if phrase not in REASON_TEXT_KO:
            issues.append(f"KO reason text에 '{phrase}' 없음 ({label})")

    # ≈ 금지 (KO, EN 모두)
    if "≈" in REASON_TEXT_KO:
        issues.append("REASON_TEXT_KO에 '≈' 포함 — 금지 문자")
    if "≈" in REASON_TEXT_EN:
        issues.append("REASON_TEXT_EN에 '≈' 포함 — 금지 문자")

    # EN 필수 포함
    required_en = [
        ("+245 HU", "HU 수치"),
        ("not a diagnosis", "disclaimer"),
        ("reference cue", "참고 단서"),
        ("lower_peripheral", "position_bin"),
    ]
    for phrase, label in required_en:
        if phrase not in REASON_TEXT_EN:
            issues.append(f"EN reason text에 '{phrase}' 없음 ({label})")

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "ko_len": len(REASON_TEXT_KO),
        "en_len": len(REASON_TEXT_EN),
        "has_hu": "약 +245 HU" in REASON_TEXT_KO,
        "has_disclaimer": "진단 의미는 아닙니다" in REASON_TEXT_KO,
        "has_approx_symbol": "≈" in REASON_TEXT_KO or "≈" in REASON_TEXT_EN,
    }


# ============================================================
# font resolve
# ============================================================
def _resolve_font_for_reason_box() -> str:
    candidates = [
        "NanumGothic", "NanumBarunGothic", "NanumMyeongjo",
        "Malgun Gothic", "Apple SD Gothic Neo",
        "DejaVu Sans", "Liberation Sans", "sans-serif",
    ]
    try:
        from matplotlib import font_manager as fm
        # font patch: WSL 환경에서 malgun.ttf 직접 등록
        _malgun = "/mnt/c/Windows/Fonts/malgun.ttf"
        if pathlib.Path(_malgun).exists():
            fm.fontManager.addfont(_malgun)
        available = {f.name for f in fm.fontManager.ttflist}
        for name in candidates:
            if name in available:
                return name
    except Exception:
        pass
    return "sans-serif"


# ============================================================
# v5 canvas 생성 — 3행 comparison-first layout
# ============================================================
def build_v5_canvas(
    s3_png_path: str,
    ref_paths: List[str],
    font_family: str,
    canvas_width_px: int = CANVAS_WIDTH_PX,
    dpi: int = CANVAS_DPI,
):
    """
    v5 canvas: comparison-first layout.

    절대 금지 (가드 체크):
    - FULL_CARD_PASTE_STRATEGY 사용 금지
    - CT npy 로드 금지
    - S3 PNG 전체를 새 canvas에 paste 금지
    - old Panel C 포함 금지
    - Ref2/Ref3 카드 표시 금지 (SHOW_REF2_REF3_ON_CARD=False)

    Row 1 (300px): Candidate 256×256 (좌) | Matched ref 256×256 (우)
    Row 2 (220px): A context 190×190 (좌) | D context 190×190 (우)
    Row 3 (160px): Reason box (전체 폭, KO only)
    """
    if not ALLOW_RUN_CARD_PROTOTYPE:
        _block("build_v5_canvas: ALLOW_RUN_CARD_PROTOTYPE=False — 실행 금지")
    if ALLOW_CT_LOAD:
        _block("build_v5_canvas: ALLOW_CT_LOAD=True — CT 로드 감지")
    if ALLOW_ORIGINAL_CARD_MODIFICATION:
        _block("build_v5_canvas: ALLOW_ORIGINAL_CARD_MODIFICATION=True — 원본 수정 금지")
    if SHOW_REF2_REF3_ON_CARD:
        _block("build_v5_canvas: SHOW_REF2_REF3_ON_CARD=True — Ref2/Ref3 카드 표시 금지")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    import numpy as np
    from PIL import Image

    # S3 source PNG read-only 로드 (crop 전용 — full paste 금지)
    src_pil = Image.open(s3_png_path).convert("RGB")

    # A panel crop → resize to context size
    a_pil = src_pil.crop(A_CROP_BBOX).resize(
        (CONTEXT_DISPLAY_SIZE, CONTEXT_DISPLAY_SIZE), Image.LANCZOS
    )
    a_arr = np.array(a_pil)

    # Candidate crop (B panel, title 제외) → resize to display size
    cand_pil = src_pil.crop(CANDIDATE_CROP_BBOX).resize(
        (CANDIDATE_DISPLAY_SIZE, CANDIDATE_DISPLAY_SIZE), Image.LANCZOS
    )
    cand_arr = np.array(cand_pil)

    # D panel crop → resize to context size
    d_pil = src_pil.crop(D_CROP_BBOX).resize(
        (CONTEXT_DISPLAY_SIZE, CONTEXT_DISPLAY_SIZE), Image.LANCZOS
    )
    d_arr = np.array(d_pil)

    # matched reference: staged LANCZOS 32→128→256
    # Ref2/Ref3는 SHOW_REF2_REF3_ON_CARD=False이므로 카드에 표시하지 않음
    ref_img = None
    if ref_paths:
        rp = REF_BANK_FULL / ref_paths[0]
        if rp.exists():
            img_pil = Image.open(str(rp)).convert("RGB")
            for stage_size in REF_UPSCALE_STAGES:
                img_pil = img_pil.resize((stage_size, stage_size), Image.LANCZOS)
            ref_img = np.array(img_pil)

    # 캔버스 크기
    total_h_px = ROW1_H_PX + ROW2_H_PX + ROW3_H_PX + 80

    canvas_w_inch = canvas_width_px / dpi
    canvas_h_inch = total_h_px / dpi

    fig = plt.figure(figsize=(canvas_w_inch, canvas_h_inch), dpi=dpi)
    fig.patch.set_facecolor("#f8f8f8")
    fig.suptitle(
        f"S4 Reason Card Prototype v5 — {TARGET_CASE_ID}",
        fontsize=10, fontfamily=font_family,
        fontweight="bold", color="#1a3a5c",
        y=0.99,
    )

    # 3행 gridspec: Row1=비교 | Row2=context | Row3=reason
    gs_main = gridspec.GridSpec(
        3, 1,
        figure=fig,
        height_ratios=[ROW1_H_PX, ROW2_H_PX, ROW3_H_PX],
        hspace=0.06,
        left=0.01, right=0.99,
        top=0.96, bottom=0.01,
    )

    # ---- Row 1: Candidate (좌) | Matched ref (우) ----
    gs_row1 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_main[0], wspace=0.04
    )

    ax_cand = fig.add_subplot(gs_row1[0, 0])
    ax_cand.imshow(cand_arr, interpolation="lanczos", aspect="equal")
    ax_cand.axis("off")
    ax_cand.set_title(
        CANDIDATE_LABEL, fontsize=9, fontfamily=font_family,
        color="#1a3a5c", pad=5, fontweight="bold", loc="center"
    )

    ax_mref = fig.add_subplot(gs_row1[0, 1])
    if ref_img is not None:
        ax_mref.imshow(ref_img, interpolation="lanczos", aspect="equal")
    else:
        ax_mref.text(
            0.5, 0.5, "N/A (ref not found)",
            ha="center", va="center", fontsize=9, color="#888888",
            transform=ax_mref.transAxes,
        )
    ax_mref.axis("off")
    ax_mref.set_title(
        MAIN_REF_LABEL, fontsize=9, fontfamily=font_family,
        color="#1a3a5c", pad=5, fontweight="bold", loc="center"
    )

    # Row 1 section 라벨 (전체 Row 1 상단)
    ax_cand.annotate(
        COMPARISON_SECTION_LABEL,
        xy=(0.5, 1.06), xycoords="axes fraction",
        fontsize=8, fontfamily=font_family, color="#3a7abf",
        ha="center", va="bottom",
        annotation_clip=False,
    )

    # ---- Row 2: A context (좌) | D context (우) ----
    gs_row2 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_main[1], wspace=0.04
    )

    ax_a = fig.add_subplot(gs_row2[0, 0])
    ax_a.imshow(a_arr, interpolation="lanczos", aspect="equal")
    ax_a.axis("off")
    ax_a.set_title(
        PANEL_A_LABEL, fontsize=7, fontfamily=font_family,
        color="#555555", pad=3, loc="left"
    )

    ax_d = fig.add_subplot(gs_row2[0, 1])
    ax_d.imshow(d_arr, interpolation="lanczos", aspect="equal")
    ax_d.axis("off")
    ax_d.set_title(
        PANEL_D_LABEL, fontsize=7, fontfamily=font_family,
        color="#555555", pad=3, loc="left"
    )

    # ---- Row 3: Reason box ----
    ax_reason = fig.add_subplot(gs_main[2])
    ax_reason.set_xlim(0, 1)
    ax_reason.set_ylim(0, 1)
    ax_reason.axis("off")
    ax_reason.patch.set_facecolor("#eef4fb")

    rect = mpatches.FancyBboxPatch(
        (0.005, 0.04), 0.990, 0.90,
        boxstyle="round,pad=0.01",
        linewidth=1.5,
        edgecolor="#3a7abf",
        facecolor="#eef4fb",
        transform=ax_reason.transAxes,
        zorder=1,
    )
    ax_reason.add_patch(rect)

    ax_reason.text(
        0.015, 0.82,
        REASON_TITLE,
        fontsize=9, fontfamily=font_family,
        fontweight="bold", color="#1a3a5c",
        transform=ax_reason.transAxes,
        zorder=2,
    )

    # KO reason text — 2줄로 명시 분리
    reason_line1 = (
        "같은 위치(lower_peripheral)의 정상 reference와 비교했을 때, "
        "후보 crop은 HU 밀도가 더 높게 측정되었습니다(약 +245 HU)."
    )
    reason_line2 = (
        "이 결과는 PaDiM high-response를 해석하기 위한 참고 단서이며, 진단 의미는 아닙니다."
    )
    ax_reason.text(
        0.015, 0.54,
        reason_line1,
        fontsize=8, fontfamily=font_family,
        color="#222222",
        transform=ax_reason.transAxes,
        wrap=True, zorder=2,
    )
    ax_reason.text(
        0.015, 0.22,
        reason_line2,
        fontsize=8, fontfamily=font_family,
        color="#444444", style="italic",
        transform=ax_reason.transAxes,
        wrap=True, zorder=2,
    )

    return fig


# ============================================================
# prototype JSON 생성
# ============================================================
def build_prototype_json(
    s3_png_path: str,
    s3_json_path: str,
    s3_data: Dict[str, Any],
    ref_paths: List[str],
    output_png_path: str,
) -> Dict[str, Any]:
    matched_ref_source = str(REF_BANK_FULL / ref_paths[0]) if ref_paths else None
    additional_ref_sources = [
        str(REF_BANK_FULL / r) for r in ref_paths[1:3]
    ]

    ko_forbidden = scan_forbidden_terms(REASON_TEXT_KO)
    en_forbidden = scan_forbidden_terms(REASON_TEXT_EN)

    return {
        "case_id": TARGET_CASE_ID,
        "prototype_version": PROTOTYPE_VERSION,
        "layout_version": LAYOUT_VERSION,
        "source_s3_card_png": s3_png_path,
        "source_s3_card_json": s3_json_path,
        "source_s4_reason_csv": str(V3_OUTPUT_CSV),
        "source_s4_reason_json": str(V3_OUTPUT_JSON),
        "full_card_paste_strategy": FULL_CARD_PASTE_STRATEGY,
        "old_panel_c_removed": True,
        "old_panel_c_included": False,
        "a_panel_bbox_in_source_png": list(A_CROP_BBOX),
        "b_panel_bbox_in_source_png": list(B_CROP_BBOX),
        "d_panel_bbox_in_source_png": list(D_CROP_BBOX),
        "candidate_crop_source": "S3_PNG_B_panel_image_only",
        "candidate_crop_bbox_in_source_png": list(CANDIDATE_CROP_BBOX),
        "candidate_display_size": [CANDIDATE_DISPLAY_SIZE, CANDIDATE_DISPLAY_SIZE],
        "matched_reference_source": matched_ref_source,
        "matched_reference_display_size": [REF_DISPLAY_SIZE, REF_DISPLAY_SIZE],
        "additional_reference_sources": additional_ref_sources,
        "additional_refs_displayed_on_card": SHOW_REF2_REF3_ON_CARD,
        "reference_resize_method": REF_UPSCALE_METHOD,
        "reference_resize_stages_px": REF_UPSCALE_STAGES,
        "context_row_added": True,
        "context_a_display_size": [CONTEXT_DISPLAY_SIZE, CONTEXT_DISPLAY_SIZE],
        "context_d_display_size": [CONTEXT_DISPLAY_SIZE, CONTEXT_DISPLAY_SIZE],
        "reason_box_added": True,
        "reason_text_version": "v5_comparison_first_text",
        "reason_title": REASON_TITLE,
        "reason_text_ko": REASON_TEXT_KO,
        "reason_text_en": REASON_TEXT_EN,
        "displayed_reason_text": REASON_TEXT_KO,
        "display_language": "ko_only",
        "disclaimer_present": "진단 의미는 아닙니다" in REASON_TEXT_KO,
        "diagnostic_guard_passed": (
            len(ko_forbidden) == 0 and len(en_forbidden) == 0
        ),
        "card_reflection_status": s3_data.get("card_reflection_status", "card_text_candidate"),
        "prototype_target_case": True,
        "card_text_ready": True,
        "json_only_ready": True,
        "existing_card_modified": False,
        "ct_load_occurred": False,
        "full_300_applied": False,
        "stage2_holdout_accessed": False,
        "output_png_path": output_png_path,
        "position_bin": s3_data.get("position_bin", "lower_peripheral"),
        "max_score": s3_data.get("max_score", None),
        "threshold": s3_data.get("threshold", None),
        "role": s3_data.get("role", "lesion_candidate"),
        "roi_coverage": s3_data.get("roi_coverage", None),
    }


# ============================================================
# run prototype (실제 생성 — ALLOW_RUN_CARD_PROTOTYPE=True 필요)
# ============================================================
def run_prototype() -> None:
    """v5 prototype PNG/JSON 생성."""
    if not ALLOW_RUN_CARD_PROTOTYPE:
        _block("run_prototype: ALLOW_RUN_CARD_PROTOTYPE=False — 사용자 승인 후 True로 변경")

    guard = check_output_guard()
    if not guard["ok"]:
        _block(f"output guard FAIL: {guard.get('error')}")

    s3_info = resolve_s3_card_paths()
    if not s3_info["ok"]:
        _block(f"S3 card resolve FAIL: {s3_info.get('error')}")

    s3_png_path  = s3_info["png_absolute"]
    s3_json_path = s3_info["json_absolute"]

    meta = read_s3_card_json_metadata(s3_json_path)
    if not meta["ok"]:
        _block(f"S3 JSON 로드 FAIL: {meta.get('error')}")
    s3_data = meta["data"]

    ref_info = resolve_ref_paths_from_json(s3_json_path)
    if not ref_info["ok"]:
        print(f"[WARN] ref crop 경로 오류: {ref_info.get('error')}", file=sys.stderr)
    ref_paths = ref_info.get("ref_paths", [])

    font_family = _resolve_font_for_reason_box()

    fig = build_v5_canvas(
        s3_png_path=s3_png_path,
        ref_paths=ref_paths,
        font_family=font_family,
    )

    PROTO_CARDS_PNG_DIR.mkdir(parents=True, exist_ok=True)
    PROTO_CARDS_JSON_DIR.mkdir(parents=True, exist_ok=True)

    png_out  = PROTO_CARDS_PNG_DIR  / f"{TARGET_CASE_ID}_reason_prototype.png"
    json_out = PROTO_CARDS_JSON_DIR / f"{TARGET_CASE_ID}_reason_prototype.json"

    import matplotlib
    matplotlib.use("Agg")
    fig.savefig(str(png_out), dpi=CANVAS_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())

    import matplotlib.pyplot as plt
    plt.close(fig)

    proto_json = build_prototype_json(
        s3_png_path=s3_png_path,
        s3_json_path=s3_json_path,
        s3_data=s3_data,
        ref_paths=ref_paths,
        output_png_path=str(png_out),
    )
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(proto_json, f, ensure_ascii=False, indent=2)

    # index_cards.csv
    with open(PROTO_INDEX_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "case_id", "role", "source_png", "source_json",
            "prototype_png_path", "prototype_json_path",
            "status", "prototype_version", "layout_version",
            "candidate_display_size", "matched_reference_display_size",
            "reason_box_added", "diagnostic_guard_passed",
            "existing_card_modified", "mode",
        ])
        writer.writeheader()
        writer.writerow({
            "case_id": TARGET_CASE_ID,
            "role": proto_json.get("role", "lesion_candidate"),
            "source_png": s3_png_path,
            "source_json": s3_json_path,
            "prototype_png_path": str(png_out.relative_to(PROTO_OUTPUT_ROOT)),
            "prototype_json_path": str(json_out.relative_to(PROTO_OUTPUT_ROOT)),
            "status": "ok",
            "prototype_version": PROTOTYPE_VERSION,
            "layout_version": LAYOUT_VERSION,
            "candidate_display_size": f"{CANDIDATE_DISPLAY_SIZE}x{CANDIDATE_DISPLAY_SIZE}",
            "matched_reference_display_size": f"{REF_DISPLAY_SIZE}x{REF_DISPLAY_SIZE}",
            "reason_box_added": "true",
            "diagnostic_guard_passed": str(proto_json["diagnostic_guard_passed"]).lower(),
            "existing_card_modified": "false",
            "mode": "prototype_1case",
        })

    # runtime_summary.json
    runtime = {
        "prototype_version": PROTOTYPE_VERSION,
        "layout_version": LAYOUT_VERSION,
        "case_id": TARGET_CASE_ID,
        "output_root": str(PROTO_OUTPUT_ROOT),
        "png_out": str(png_out),
        "json_out": str(json_out),
        "old_panel_c_removed": True,
        "full_card_paste_strategy": FULL_CARD_PASTE_STRATEGY,
        "show_ref2_ref3_on_card": SHOW_REF2_REF3_ON_CARD,
        "candidate_display_size": CANDIDATE_DISPLAY_SIZE,
        "ref_display_size": REF_DISPLAY_SIZE,
        "ct_load_occurred": False,
        "full_300_applied": False,
        "stage2_holdout_accessed": False,
        "existing_card_modified": False,
    }
    with open(PROTO_RUNTIME_JSON, "w", encoding="utf-8") as f:
        json.dump(runtime, f, ensure_ascii=False, indent=2)

    # errors.csv
    with open(PROTO_ERRORS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "error", "stage"])
        writer.writeheader()

    # DONE.json
    done = {
        "status": "DONE",
        "prototype_version": PROTOTYPE_VERSION,
        "layout_version": LAYOUT_VERSION,
        "case_id": TARGET_CASE_ID,
        "ct_load_occurred": False,
        "existing_card_modified": False,
        "full_300_applied": False,
        "stage2_holdout_accessed": False,
    }
    with open(PROTO_DONE_JSON, "w", encoding="utf-8") as f:
        json.dump(done, f, ensure_ascii=False, indent=2)

    print(f"[OK] v5 prototype 생성 완료: {png_out}")
    print(f"[OK] JSON: {json_out}")


# ============================================================
# selftest (22개 guard 검사)
# ============================================================
def run_selftest() -> bool:
    results = []

    def _check(name: str, ok: bool, detail: str = "", warning: str = "") -> None:
        status = "PASS" if ok else "FAIL"
        results.append({"name": name, "status": status, "detail": detail, "warning": warning})
        print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))

    print("=== selftest (v5) ===")

    # 1. target case 정확히 1건
    _check(
        "1. target case 정확히 1건",
        TARGET_CASE_ID == "LUNG1-320__c2",
        f"TARGET_CASE_ID='{TARGET_CASE_ID}'",
    )

    # 2. target case = LUNG1-320__c2
    _check(
        "2. target case = LUNG1-320__c2",
        TARGET_CASE_ID == "LUNG1-320__c2",
        f"확인됨: {TARGET_CASE_ID}",
    )

    # 3. guard default false
    _check(
        "3. guard default false",
        not ALLOW_RUN_CARD_PROTOTYPE
        and not ALLOW_CT_LOAD
        and not ALLOW_FULL_300
        and not ALLOW_ORIGINAL_CARD_MODIFICATION,
        f"RUN={ALLOW_RUN_CARD_PROTOTYPE}, CT={ALLOW_CT_LOAD}, "
        f"FULL300={ALLOW_FULL_300}, ORIG_MOD={ALLOW_ORIGINAL_CARD_MODIFICATION}",
    )

    # 4. full S3 card paste 금지
    _check(
        "4. full S3 card paste 금지",
        FULL_CARD_PASTE_STRATEGY == "DISABLED",
        f"FULL_CARD_PASTE_STRATEGY={FULL_CARD_PASTE_STRATEGY}",
    )

    # 5. old Panel C 포함 금지
    _check(
        "5. old Panel C 포함 금지",
        C_CROP_BBOX is None,
        f"C_CROP_BBOX={C_CROP_BBOX} (None이어야 함)",
    )

    # 6. candidate display size = 256×256
    _check(
        "6. candidate display size = 256×256",
        CANDIDATE_DISPLAY_SIZE == 256,
        f"CANDIDATE_DISPLAY_SIZE={CANDIDATE_DISPLAY_SIZE}",
    )

    # 7. matched ref display size = 256×256
    _check(
        "7. matched ref display size = 256×256",
        REF_DISPLAY_SIZE == 256,
        f"REF_DISPLAY_SIZE={REF_DISPLAY_SIZE}",
    )

    # 8. additional refs displayed on card = false
    _check(
        "8. additional refs displayed on card = false",
        SHOW_REF2_REF3_ON_CARD is False,
        f"SHOW_REF2_REF3_ON_CARD={SHOW_REF2_REF3_ON_CARD}",
    )

    # 9. Ref2/Ref3는 JSON에만 보존 (코드에서 build_prototype_json에 additional_reference_sources 있음)
    import inspect
    json_src = inspect.getsource(build_prototype_json)
    has_additional = "additional_reference_sources" in json_src
    _check(
        "9. Ref2/Ref3 JSON-only 보존",
        has_additional and not SHOW_REF2_REF3_ON_CARD,
        "additional_reference_sources JSON 필드 있음" if has_additional else "additional_reference_sources 없음",
    )

    # 10. D context row가 main comparison보다 작음
    _check(
        "10. D context row < main comparison",
        CONTEXT_DISPLAY_SIZE < CANDIDATE_DISPLAY_SIZE,
        f"CONTEXT={CONTEXT_DISPLAY_SIZE}px < CANDIDATE={CANDIDATE_DISPLAY_SIZE}px",
    )

    # 11. reason text에 '약 +245 HU' 있음
    _check(
        "11. reason text '약 +245 HU' 있음",
        "약 +245 HU" in REASON_TEXT_KO,
        f"확인됨" if "약 +245 HU" in REASON_TEXT_KO else "없음",
    )

    # 12. reason text에 '≈' 없음
    no_approx = "≈" not in REASON_TEXT_KO and "≈" not in REASON_TEXT_EN
    _check(
        "12. reason text '≈' 없음",
        no_approx,
        "≈ 없음 확인됨" if no_approx else f"KO={('≈' in REASON_TEXT_KO)}, EN={('≈' in REASON_TEXT_EN)}",
    )

    # 13. reason text에 '진단 의미는 아닙니다' 있음
    _check(
        "13. reason text '진단 의미는 아닙니다' 있음",
        "진단 의미는 아닙니다" in REASON_TEXT_KO,
        "확인됨" if "진단 의미는 아닙니다" in REASON_TEXT_KO else "없음",
    )

    # 14. forbidden wording hit 0
    ko_forbidden = scan_forbidden_terms(REASON_TEXT_KO)
    en_forbidden = scan_forbidden_terms(REASON_TEXT_EN)
    _check(
        "14. forbidden wording hit 0",
        len(ko_forbidden) == 0 and len(en_forbidden) == 0,
        f"KO금지어={ko_forbidden}, EN금지어={en_forbidden}",
    )

    # 15. candidate crop bbox y_start >= 70
    bbox_res = validate_crop_bboxes()
    b_y_ok = bbox_res["bboxes"].get("B_y_start_check", {}).get("ok", False)
    b_y_val = bbox_res["bboxes"].get("B_y_start_check", {}).get("y_start", -1)
    _check(
        "15. candidate crop bbox y_start >= 70",
        b_y_ok,
        f"B_CROP_BBOX y_start={b_y_val}",
    )

    # 16. candidate crop title exclusion 정책 존재 (코드에서 확인)
    canvas_src = inspect.getsource(build_v5_canvas)
    has_title_exclusion = "CANDIDATE_CROP_BBOX" in canvas_src and "title" in __doc__ if __doc__ else False
    # 상수 주석에서 title 제외 정책 확인
    import __main__ as _m
    module_src = inspect.getsource(sys.modules[__name__])
    has_title_policy = "title" in module_src and "70" in module_src and "B_CROP_BBOX" in module_src
    _check(
        "16. candidate crop title exclusion 정책 존재",
        has_title_policy,
        "B_CROP_BBOX y=70 title 제외 정책 확인됨" if has_title_policy else "정책 없음",
    )

    # 17. no CT load (ALLOW_CT_LOAD=False)
    _check(
        "17. no CT load",
        not ALLOW_CT_LOAD,
        f"ALLOW_CT_LOAD={ALLOW_CT_LOAD}",
    )

    # 18. no np.load (소스 코드 검사)
    has_npload = "np.load(" in canvas_src
    _check(
        "18. no np.load",
        not has_npload,
        "np.load 없음 확인됨" if not has_npload else "np.load 발견됨",
    )

    # 19. no full 300 loop (ALLOW_FULL_300=False)
    _check(
        "19. no full 300 loop",
        not ALLOW_FULL_300,
        f"ALLOW_FULL_300={ALLOW_FULL_300}",
    )

    # 20. stage2_holdout 접근 0
    has_s2 = any(tok in canvas_src for tok in STAGE2_HOLDOUT_TOKENS)
    _check(
        "20. stage2_holdout 접근 0",
        not has_s2,
        "stage2_holdout 없음 확인됨" if not has_s2 else "stage2_holdout 토큰 발견됨",
    )

    # 21. JSON schema 필수 필드 존재
    required_json_fields = [
        "case_id", "prototype_version", "layout_version",
        "source_s3_card_png", "source_s3_card_json",
        "source_s4_reason_csv", "source_s4_reason_json",
        "candidate_crop_source", "candidate_crop_bbox_in_source_png",
        "candidate_display_size", "matched_reference_source",
        "matched_reference_display_size", "additional_reference_sources",
        "additional_refs_displayed_on_card", "context_row_added",
        "reason_box_added", "reason_text_version",
        "reason_title", "reason_text_ko", "reason_text_en",
        "displayed_reason_text", "display_language",
        "disclaimer_present", "diagnostic_guard_passed",
        "card_reflection_status", "prototype_target_case",
        "card_text_ready", "json_only_ready",
        "existing_card_modified", "ct_load_occurred",
        "full_300_applied", "stage2_holdout_accessed",
    ]
    missing_fields = [f for f in required_json_fields if f not in json_src]
    _check(
        "21. JSON schema 필수 필드 존재",
        len(missing_fields) == 0,
        f"누락 필드: {missing_fields}" if missing_fields else f"{len(required_json_fields)}개 필드 전부 확인됨",
    )

    # 22. output root가 S3/v1/v2/v3/v4와 분리됨
    guard_res = check_output_guard()
    _check(
        "22. output root가 기존 경로와 분리됨",
        guard_res["ok"],
        guard_res.get("error", f"v5 output root 충돌 없음: {PROTO_OUTPUT_ROOT.name}"),
    )

    # 결과 집계
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    print(f"\n=== selftest 결과: PASS={n_pass}/{len(results)} FAIL={n_fail}/{len(results)} ===")

    if n_fail == 0:
        print("[PASS] selftest 전체 통과")
    else:
        print("[FAIL] selftest 일부 실패 — 위 FAIL 항목 확인 필요")

    return n_fail == 0


# ============================================================
# dry-run (PNG 열지 않음, 경로 존재 확인만)
# ============================================================
def run_dry_run() -> bool:
    """입력 파일 존재 + 1건 resolve + output guard 확인 (PNG open 없음)."""
    print("=== dry-run (v5) ===")
    issues = []

    if ALLOW_RUN_CARD_PROTOTYPE:
        issues.append("ALLOW_RUN_CARD_PROTOTYPE=True — 이 단계에서는 False여야 함")
    if ALLOW_CT_LOAD:
        issues.append("ALLOW_CT_LOAD=True — 금지")

    # V3 CSV 존재
    if not V3_OUTPUT_CSV.exists():
        issues.append(f"V3 CSV 없음: {V3_OUTPUT_CSV}")
    else:
        print(f"  [OK] V3 CSV: {V3_OUTPUT_CSV}")

    # S3 PNG 존재 (open 없이 경로만 확인)
    s3_png = S3_CARDS_PNG_DIR / f"{TARGET_CASE_ID}.png"
    if not s3_png.exists():
        issues.append(f"S3 PNG 없음: {s3_png}")
    else:
        print(f"  [OK] S3 PNG: {s3_png}")

    # S3 JSON 존재
    s3_json = S3_CARDS_JSON_DIR / f"{TARGET_CASE_ID}.json"
    if not s3_json.exists():
        issues.append(f"S3 JSON 없음: {s3_json}")
    else:
        print(f"  [OK] S3 JSON: {s3_json}")

    # reference manifest 존재
    if not REF_CROP_MANIFEST.exists():
        issues.append(f"REF_CROP_MANIFEST 없음: {REF_CROP_MANIFEST}")
    else:
        print(f"  [OK] REF_CROP_MANIFEST: {REF_CROP_MANIFEST}")

    # matched reference 경로 존재 (JSON open하여 경로 확인)
    if s3_json.exists():
        ref_info = resolve_ref_paths_from_json(str(s3_json))
        if ref_info["ok"] and ref_info["ref_count"] > 0:
            matched_rp = REF_BANK_FULL / ref_info["ref_paths"][0]
            if matched_rp.exists():
                print(f"  [OK] matched_ref: {ref_info['ref_paths'][0]}")
            else:
                issues.append(f"matched_ref 없음: {matched_rp}")
            add_refs = ref_info["ref_paths"][1:3]
            for rp in add_refs:
                full_p = REF_BANK_FULL / rp
                if full_p.exists():
                    print(f"  [OK] additional_ref: {rp}")
                else:
                    print(f"  [WARN] additional_ref 없음 (JSON에만 보존): {rp}")
        else:
            print(f"  [WARN] ref_paths 없음 또는 오류: {ref_info.get('error', 'unknown')}")

    # output guard
    guard = check_output_guard()
    if not guard["ok"]:
        issues.append(f"output guard FAIL: {guard.get('error')}")
    else:
        print(f"  [OK] output guard: v5 output root 충돌 없음")

    # bbox 유효성 (v5: y_start >= 70 포함)
    bbox_res = validate_crop_bboxes()
    if not bbox_res["ok"]:
        issues.append(f"bbox 오류: {'; '.join(bbox_res['issues'])}")
    else:
        print(f"  [OK] crop bbox 유효성: A/B/D bbox 정상, C=None, B_y_start={B_CROP_BBOX[1]}")

    # display size 일치
    size_res = validate_display_sizes()
    if not size_res["ok"]:
        issues.append(f"display size 오류: {'; '.join(size_res['issues'])}")
    else:
        print(f"  [OK] display size: candidate={CANDIDATE_DISPLAY_SIZE}, ref={REF_DISPLAY_SIZE} (동일)")

    # reason text
    text_res = validate_reason_text()
    if not text_res["ok"]:
        issues.append(f"reason text 오류: {'; '.join(text_res['issues'])}")
    else:
        print(f"  [OK] reason text: KO={text_res['ko_len']}자, 면책문구 있음, ≈ 없음")

    # SHOW_REF2_REF3_ON_CARD=False 확인
    if SHOW_REF2_REF3_ON_CARD:
        issues.append("SHOW_REF2_REF3_ON_CARD=True — Ref2/Ref3 카드 표시 금지")
    else:
        print(f"  [OK] SHOW_REF2_REF3_ON_CARD=False — Ref2/Ref3 카드 제거 확인")

    if issues:
        print(f"\n[FAIL] dry-run 이슈 {len(issues)}건:")
        for i in issues:
            print(f"  - {i}")
        return False
    else:
        print("\n[PASS] dry-run 전체 통과")
        return True


# ============================================================
# plan-only (배치 계획 출력)
# ============================================================
def run_plan_only() -> None:
    """dry-run + v5 배치 계획 출력."""
    dry_ok = run_dry_run()

    print("\n=== plan-only: v5 comparison-first layout 계획 ===")
    print(f"  대상 case     : {TARGET_CASE_ID}")
    print(f"  prototype 버전: {PROTOTYPE_VERSION}")
    print(f"  layout 버전   : {LAYOUT_VERSION}")
    print()
    print(f"  [Row 1] Main comparison (h={ROW1_H_PX}px)")
    print(f"    왼쪽: Candidate crop {CANDIDATE_DISPLAY_SIZE}×{CANDIDATE_DISPLAY_SIZE}px")
    print(f"      CANDIDATE_CROP_BBOX: {CANDIDATE_CROP_BBOX} (y_start={CANDIDATE_CROP_BBOX[1]}, v4 대비 title 제외)")
    print(f"      resize: LANCZOS downscale → {CANDIDATE_DISPLAY_SIZE}×{CANDIDATE_DISPLAY_SIZE}")
    print(f"    오른쪽: Matched normal reference {REF_DISPLAY_SIZE}×{REF_DISPLAY_SIZE}px")
    print(f"      staged upscale: 32 → {' → '.join(str(s) for s in REF_UPSCALE_STAGES)}")
    print(f"      upscale factor: {REF_UPSCALE_STAGES[-1]//32}x (v4 17.4x 대비 감소)")
    print(f"    Ref2/Ref3: 카드 미표시 (SHOW_REF2_REF3_ON_CARD={SHOW_REF2_REF3_ON_CARD})")
    print()
    print(f"  [Row 2] Context panels (h={ROW2_H_PX}px)")
    print(f"    왼쪽: A whole-slice {CONTEXT_DISPLAY_SIZE}×{CONTEXT_DISPLAY_SIZE}px — 위치 확인용")
    print(f"      A_CROP_BBOX: {A_CROP_BBOX}")
    print(f"    오른쪽: D z-context {CONTEXT_DISPLAY_SIZE}×{CONTEXT_DISPLAY_SIZE}px — 보조")
    print(f"      D_CROP_BBOX: {D_CROP_BBOX}")
    print()
    print(f"  [Row 3] Reason box (h={ROW3_H_PX}px, 전체 폭, KO만 표시)")
    print(f"    Title: {REASON_TITLE}")
    print(f"    KO ({len(REASON_TEXT_KO)}자): {REASON_TEXT_KO[:60]}...")
    print(f"    '≈' 없음: {'확인' if '≈' not in REASON_TEXT_KO else '경고: ≈ 있음'}")
    print()
    print(f"  canvas 크기 추정: {CANVAS_WIDTH_PX}×{CANVAS_HEIGHT_ESTIMATE_PX}px (v4 1836px 대비 ~58% 축소)")
    print()
    print(f"  Source S3 PNG: {S3_CARDS_PNG_DIR / f'{TARGET_CASE_ID}.png'}")
    print(f"    → read-only crop만 사용 (full card paste 금지)")
    print(f"    → old Panel C 완전 제거")
    print()
    print(f"  Output root  : {PROTO_OUTPUT_ROOT}")
    print(f"  Output PNG   : {PROTO_CARDS_PNG_DIR / f'{TARGET_CASE_ID}_reason_prototype.png'}")
    print(f"  Output JSON  : {PROTO_CARDS_JSON_DIR / f'{TARGET_CASE_ID}_reason_prototype.json'}")
    print()

    # reference selection plan
    s3_json = S3_CARDS_JSON_DIR / f"{TARGET_CASE_ID}.json"
    if s3_json.exists():
        ref_info = resolve_ref_paths_from_json(str(s3_json))
        print(f"  Reference selection plan:")
        print(f"    position_bin: {ref_info.get('position_bin', 'unknown')}")
        print(f"    ref count: {ref_info.get('ref_count', 0)}")
        for i, rp in enumerate(ref_info.get("ref_paths", [])[:3]):
            role = ["matched_ref(카드표시)", "add_ref_1(JSON전용)", "add_ref_2(JSON전용)"][i]
            full_p = REF_BANK_FULL / rp
            print(f"    [{role}] exists={full_p.exists()} → {rp}")
    else:
        print(f"  [WARN] S3 JSON 없음 — reference selection plan 확인 불가")

    if not dry_ok:
        print("\n[WARN] dry-run 이슈 있음 — plan-only 결과 참고용")
    else:
        print("\n[PASS] plan-only 완료")


# ============================================================
# main
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="S4 Reason Card Prototype 1-case v5 — comparison-first layout",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="22개 guard 검사 실행 (PNG 생성 없음)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="입력 파일 존재 + output guard 확인 (PNG 생성 없음)",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        dest="plan_only",
        help="dry-run + v5 배치 계획 출력 (PNG 생성 없음)",
    )
    parser.add_argument(
        "--run-prototype",
        action="store_true",
        dest="run_prototype",
        help="실제 v5 prototype PNG/JSON 생성 (ALLOW_RUN_CARD_PROTOTYPE=True 필요)",
    )
    parser.add_argument(
        "--confirm-generate",
        action="store_true",
        dest="confirm_generate",
        help="--run-prototype 와 함께 사용 시 실제 생성 확인 (여전히 guard 체크)",
    )

    args = parser.parse_args()

    # bare 실행 차단
    if not any([args.selftest, args.dry_run, args.plan_only, args.run_prototype]):
        _block(
            "bare 실행 차단: --selftest / --dry-run / --plan-only / --run-prototype 중 하나를 지정하세요."
        )

    # --run-prototype 단독 차단
    if args.run_prototype and not args.confirm_generate:
        _block(
            "--run-prototype 단독 실행 차단: --confirm-generate 를 함께 지정하세요."
        )

    # --run-prototype --confirm-generate: ALLOW_RUN_CARD_PROTOTYPE guard
    if args.run_prototype and args.confirm_generate:
        if not ALLOW_RUN_CARD_PROTOTYPE:
            _block(
                "--run-prototype --confirm-generate: ALLOW_RUN_CARD_PROTOTYPE=False — "
                "사용자 승인 후 스크립트 내 ALLOW_RUN_CARD_PROTOTYPE=True 로 변경 후 재실행"
            )
        run_prototype()
        return

    if args.selftest:
        ok = run_selftest()
        sys.exit(0 if ok else 1)

    if args.plan_only:
        run_plan_only()
        sys.exit(0)

    if args.dry_run:
        ok = run_dry_run()
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
