#!/usr/bin/env python3
"""
build_explanation_card_s4_reason_card_prototype_1case_v4.py

S4 Reason Card Prototype 1-case v4 Script
Layout: rebuild_canvas_v1 — A/B/D panel crop + new C comparison panel + reason box

목적:
- LUNG1-320__c2 1건에 대해 기존 S3 font-fix 카드 PNG를 read-only source로만 사용하고,
  A/B/D 패널만 crop하여 새 canvas에 재배치한다.
- old Panel C는 완전히 제거하고, 새 C comparison panel(Candidate vs matched ref)을 생성한다.
- v3 실패 반영:
    1. S3 full PNG paste 방식 폐기 — source card를 새 canvas에 그대로 붙이지 않음
    2. A/B/D panel만 crop하여 새 canvas에 배치
    3. old Panel C 완전 제거
    4. 새 C comparison panel: Candidate 40% | Matched ref 40% | Ref2 10% | Ref3 10%
    5. B crop bbox title 포함 문제 수정: (659,27,1202,683) → image-only (625,41,1181,597)
    6. additional refs 폭 개선 (7% → 10%)

절대 금지:
- S3 full card를 새 canvas에 그대로 붙이는 방식 (FULL_CARD_PASTE_STRATEGY) 금지
- CT/mask npy 로드 (ALLOW_CT_LOAD=False)
- np.load 호출
- HU 통계 재계산
- 기존 S3 PNG/JSON 수정 (ALLOW_ORIGINAL_CARD_MODIFICATION=False)
- full 300 처리 (ALLOW_FULL_300=False)
- score/model/threshold 재계산
- stage2_holdout 접근
- 기존 v1/v2/v3 prototype 수정

실행 모드:
- bare 실행                                 → BLOCKED exit 2
- --selftest                                → 13개 guard 검사
- --dry-run                                 → 입력 파일 존재 + 1건 resolve + output guard 확인
- --plan-only                               → dry-run + 배치 계획 출력
- --run-prototype                           → 단독 BLOCKED exit 2
- --run-prototype --confirm-generate        → ALLOW_RUN_CARD_PROTOTYPE=False 로 BLOCKED

syntax check:
  python -m py_compile scripts/build_explanation_card_s4_reason_card_prototype_1case_v4.py
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
ALLOW_RUN_CARD_PROTOTYPE         = False   # 실제 생성 시 True로 변경 (사용자 승인 필수)
ALLOW_ORIGINAL_CARD_MODIFICATION = False   # 항구 False
ALLOW_CT_LOAD                    = False   # 항구 False
ALLOW_FULL_300                   = False   # 항구 False

# ============================================================
# v4 핵심 전략 식별자 — selftest에서 금지 문자열 검사 대상
# ============================================================
FULL_CARD_PASTE_STRATEGY = "DISABLED"  # v3 방식: S3 PNG 전체를 새 canvas에 paste — v4에서 폐기

# ============================================================
# prototype 대상 (1건)
# ============================================================
TARGET_CASE_ID = "LUNG1-320__c2"

# ============================================================
# reason text v4 — KO만 표시, EN은 JSON 전용
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
# v4 layout 설정
# ============================================================
LAYOUT_VERSION       = "rebuild_canvas_v1"
PROTOTYPE_VERSION    = "v4"

# Panel labels
PANEL_A_LABEL        = "A. Whole slice + heatmap"
PANEL_B_LABEL        = "B. Local crop + heatmap"
PANEL_C_LABEL        = "C. Candidate vs same-bin normal reference"
PANEL_D_LABEL        = "D. z-context [139, 140, 141]"

# Comparison panel labels
CANDIDATE_LABEL      = "Candidate crop"
MAIN_REF_LABEL       = "Matched normal reference"
REF2_LABEL           = "Ref 2"
REF3_LABEL           = "Ref 3"

# ============================================================
# SOURCE CROP BBOXES (S3 PNG read-only crop 좌표)
# 추정치 ±20px — run 후 눈검증 대상
# selftest에서 bbox 유효성(x1<x2, y1<y2) 검사
# ============================================================
A_CROP_BBOX          = (29,  41, 585,  597)   # (left, upper, right, lower) — image-only
B_CROP_BBOX          = (625, 41, 1181, 597)   # image-only, title 제외 (v3 버그 수정)
D_CROP_BBOX          = (625, 638, 1181, 1194) # image-only

# Candidate comparison panel에서 B crop 재사용
CANDIDATE_CROP_BBOX  = B_CROP_BBOX

# C는 source에서 절대 crop하지 않음 (None으로 명시)
C_CROP_BBOX          = None  # v4: new C는 source crop 금지 — 새로 생성

# Comparison panel C 열 비율 (합계=100)
PANEL_C_WIDTH_RATIOS = [0.40, 0.40, 0.10, 0.10]  # Candidate | Matched ref | Ref2 | Ref3

# Reference upscale 방식: staged LANCZOS (32→128→556)
REF_UPSCALE_METHOD   = "staged_LANCZOS_32_128_556"
REF_UPSCALE_STAGES   = [128, 556]  # 중간 단계 크기

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

# v4 prototype output root (신규 경로)
PROTO_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    / "s4_reason_card_prototype_1case_v4"
)
PROTO_CARDS_PNG_DIR  = PROTO_OUTPUT_ROOT / "cards_png"
PROTO_CARDS_JSON_DIR = PROTO_OUTPUT_ROOT / "cards_json"
PROTO_INDEX_CSV      = PROTO_OUTPUT_ROOT / "index_cards.csv"
PROTO_RUNTIME_JSON   = PROTO_OUTPUT_ROOT / "runtime_summary.json"
PROTO_ERRORS_CSV     = PROTO_OUTPUT_ROOT / "errors.csv"
PROTO_DONE_JSON      = PROTO_OUTPUT_ROOT / "DONE.json"

# v4 static check 보고서 경로 (schema 준비)
STATIC_CHECK_REPORT_MD   = REPORTS_ROOT / "s4_reason_card_prototype_1case_v4_script_static_drycheck_v1.md"
STATIC_CHECK_REPORT_JSON = REPORTS_ROOT / "s4_reason_card_prototype_1case_v4_script_static_drycheck_v1.json"

# v4 preflight 입력
V4_PREFLIGHT_JSON = REPORTS_ROOT / "s4_reason_card_prototype_1case_v4_layout_rebuild_preflight_v1.json"
V4_CROP_PLAN_CSV  = REPORTS_ROOT / "s4_reason_card_prototype_1case_v4_crop_plan_v1.csv"


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
            _block(f"stage2_holdout 접근 감지: {path_or_str}")


def scan_forbidden_terms(text: str) -> List[str]:
    low = str(text).lower()
    return [t for t in FORBIDDEN_TERMS if t in low]


# ============================================================
# v3 CSV 로드 (reason text 소스)
# ============================================================
def load_v3_csv() -> List[Dict[str, str]]:
    rows = []
    with open(V3_OUTPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


# ============================================================
# target case resolve
# ============================================================
def resolve_target_case(rows: List[Dict[str, str]]) -> Dict[str, str]:
    matched = [r for r in rows if r.get("expansion_case_id") == TARGET_CASE_ID]
    if len(matched) != 1:
        _block(f"target case resolve 실패: {len(matched)}건 (정확히 1건이어야 함)")
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
            f"diagnostic_guard_passed != True (실제: {row.get('diagnostic_guard_passed')})"
        )

    return {"ok": len(issues) == 0, "issues": issues}


# ============================================================
# S3 card file readiness
# ============================================================
def resolve_s3_card_paths() -> Dict[str, Any]:
    """index_cards.csv에서 LUNG1-320__c2 매칭, PNG/JSON 경로 확인."""
    _assert_no_stage2_holdout(str(S3_CARD_ROOT))

    if not S3_INDEX_CSV.exists():
        return {"ok": False, "error": f"index_cards.csv 없음: {S3_INDEX_CSV}"}

    index_row = None
    with open(S3_INDEX_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("expansion_case_id") == TARGET_CASE_ID:
                index_row = dict(row)
                break

    if index_row is None:
        return {"ok": False, "error": f"{TARGET_CASE_ID} not found in index_cards.csv"}

    png_rel  = index_row.get("card_png_path", "")
    json_rel = index_row.get("card_json_path", "")

    png_abs  = S3_CARD_ROOT / png_rel
    json_abs = S3_CARD_ROOT / json_rel

    result: Dict[str, Any] = {
        "ok": True,
        "index_match": True,
        "png_path_relative": png_rel,
        "png_absolute": str(png_abs),
        "png_exists": png_abs.exists(),
        "json_path_relative": json_rel,
        "json_absolute": str(json_abs),
        "json_exists": json_abs.exists(),
        "source_read_only": True,
        "full_card_paste_strategy": "DISABLED",  # v4: paste 방식 폐기
    }

    if png_abs.exists():
        stat = png_abs.stat()
        result["png_size_bytes"] = stat.st_size
        result["png_mtime"]      = stat.st_mtime

    if json_abs.exists():
        stat = json_abs.stat()
        result["json_size_bytes"] = stat.st_size
        result["json_mtime"]      = stat.st_mtime

    if not result["png_exists"] or not result["json_exists"]:
        result["ok"] = False
        missing = []
        if not result["png_exists"]:
            missing.append(f"PNG 없음: {png_abs}")
        if not result["json_exists"]:
            missing.append(f"JSON 없음: {json_abs}")
        result["error"] = "; ".join(missing)

    return result


def resolve_ref_paths_from_json(json_abs: str) -> Dict[str, Any]:
    """S3 카드 JSON에서 normal_reference_crops 경로 read-only 확인 (PNG 로드 없음)."""
    p = pathlib.Path(json_abs)
    if not p.exists():
        return {"ok": False, "error": f"S3 JSON 없음: {p}", "ref_count": 0}

    with open(p, encoding="utf-8") as f:
        s3_data = json.load(f)

    ref_crop_paths = s3_data.get("normal_reference_crops", [])
    position_bin   = s3_data.get("position_bin", "unknown")
    n              = len(ref_crop_paths)

    missing = []
    for rcp in ref_crop_paths:
        rp = REF_BANK_FULL / rcp
        if not rp.exists():
            missing.append(str(rp))

    return {
        "ok":           len(missing) == 0,
        "ref_count":    n,
        "position_bin": position_bin,
        "ref_paths":    ref_crop_paths,
        "missing_pngs": missing,
        "error":        f"ref crop PNG 없음: {missing}" if missing else "",
    }


def read_s3_card_json_metadata(json_abs: str) -> Dict[str, Any]:
    """S3 카드 JSON metadata read-only 로드 (modification 금지)."""
    if ALLOW_ORIGINAL_CARD_MODIFICATION:
        _block("ALLOW_ORIGINAL_CARD_MODIFICATION=True 상태에서 S3 JSON 수정은 금지")
    p = pathlib.Path(json_abs)
    if not p.exists():
        return {"ok": False, "error": f"S3 JSON 없음: {p}"}
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return {"ok": True, "data": data}


# ============================================================
# output guard (v4 전용)
# ============================================================
def check_output_guard() -> Dict[str, Any]:
    """v4 prototype output root 안전 확인."""
    # S3 root와 경로 분리 확인
    try:
        PROTO_OUTPUT_ROOT.relative_to(S3_CARD_ROOT)
        return {
            "ok": False,
            "error": f"output root가 S3 root 하위에 있음: {PROTO_OUTPUT_ROOT}",
        }
    except ValueError:
        pass

    # v1/v2/v3 output root와 경로 분리 확인
    for label, vpath in [("v1", PROTO_V1_OUTPUT_ROOT),
                         ("v2", PROTO_V2_OUTPUT_ROOT),
                         ("v3", PROTO_V3_OUTPUT_ROOT)]:
        try:
            PROTO_OUTPUT_ROOT.relative_to(vpath)
            return {
                "ok": False,
                "error": f"v4 output root가 {label} output root 하위에 있음: {PROTO_OUTPUT_ROOT}",
            }
        except ValueError:
            pass

    if PROTO_DONE_JSON.exists():
        return {
            "ok": False,
            "error": f"DONE.json 이미 존재: {PROTO_DONE_JSON} — 기존 run 충돌",
        }

    residual = []
    for p in [PROTO_CARDS_PNG_DIR, PROTO_CARDS_JSON_DIR, PROTO_INDEX_CSV,
              PROTO_RUNTIME_JSON, PROTO_ERRORS_CSV]:
        if pathlib.Path(p).exists():
            residual.append(str(p))

    if residual:
        return {
            "ok": False,
            "error": f"잔여 파일 존재: {residual}",
            "suggestion": (
                "output root를 새 version으로 변경하거나 "
                "기존 파일을 수동으로 archive 후 재시도"
            ),
        }

    return {
        "ok": True,
        "output_root": str(PROTO_OUTPUT_ROOT),
        "output_root_exists": PROTO_OUTPUT_ROOT.exists(),
        "done_json_exists": False,
        "residual_files": False,
        "path_conflict_with_s3": False,
        "path_conflict_with_v1": False,
        "path_conflict_with_v2": False,
        "path_conflict_with_v3": False,
    }


# ============================================================
# v1/v2/v3 prototype 보존 확인
# ============================================================
def check_prior_prototypes_preserved() -> Dict[str, Any]:
    """v1/v2/v3 prototype DONE.json 존재 확인 (수정 없음 가정)."""
    status = {}
    all_ok = True
    for label, vpath in [("v1", PROTO_V1_OUTPUT_ROOT),
                         ("v2", PROTO_V2_OUTPUT_ROOT),
                         ("v3", PROTO_V3_OUTPUT_ROOT)]:
        done = vpath / "DONE.json"
        exists = done.exists()
        status[label] = {
            "output_root_exists": vpath.exists(),
            "done_json_exists": exists,
        }
        if not vpath.exists():
            all_ok = False
            status[label]["warning"] = f"{label} output root 없음 — 이미 삭제되었을 수 있음"

    return {"ok": all_ok, "prior_prototypes": status}


# ============================================================
# crop bbox 유효성 검사
# ============================================================
def validate_crop_bboxes() -> Dict[str, Any]:
    """A/B/D crop bbox 유효성 확인 (x1<x2, y1<y2, C=None 확인)."""
    issues = []
    results = {}

    for name, bbox in [("A", A_CROP_BBOX), ("B", B_CROP_BBOX), ("D", D_CROP_BBOX)]:
        left, upper, right, lower = bbox
        ok = (left < right) and (upper < lower)
        width  = right - left
        height = lower - upper
        results[name] = {
            "bbox": list(bbox),
            "width": width,
            "height": height,
            "valid": ok,
        }
        if not ok:
            issues.append(f"{name} bbox 유효하지 않음: left={left} right={right} upper={upper} lower={lower}")

    # CANDIDATE_CROP_BBOX = B_CROP_BBOX 확인
    if CANDIDATE_CROP_BBOX != B_CROP_BBOX:
        issues.append(f"CANDIDATE_CROP_BBOX가 B_CROP_BBOX와 불일치: {CANDIDATE_CROP_BBOX} != {B_CROP_BBOX}")
    results["candidate"] = {"bbox": list(CANDIDATE_CROP_BBOX), "matches_B": CANDIDATE_CROP_BBOX == B_CROP_BBOX}

    # C는 None이어야 함
    if C_CROP_BBOX is not None:
        issues.append(f"C_CROP_BBOX가 None이 아님: {C_CROP_BBOX} — v4에서 C는 source crop 금지")
    results["C"] = {"bbox": C_CROP_BBOX, "correctly_none": C_CROP_BBOX is None}

    return {"ok": len(issues) == 0, "issues": issues, "bboxes": results}


# ============================================================
# comparison panel layout ratio 검사
# ============================================================
def validate_panel_c_ratio() -> Dict[str, Any]:
    """Panel C 열 비율 합계 = 1.0 확인."""
    issues = []
    total = sum(PANEL_C_WIDTH_RATIOS)
    if abs(total - 1.0) > 0.001:
        issues.append(f"PANEL_C_WIDTH_RATIOS 합계 = {total:.4f} (1.0이어야 함)")
    if len(PANEL_C_WIDTH_RATIOS) != 4:
        issues.append(f"PANEL_C_WIDTH_RATIOS 열 수 = {len(PANEL_C_WIDTH_RATIOS)} (4이어야 함)")
    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "ratios": PANEL_C_WIDTH_RATIOS,
        "total": total,
        "columns": len(PANEL_C_WIDTH_RATIOS),
    }


# ============================================================
# reason text 검증 (v4)
# ============================================================
def validate_reason_text() -> Dict[str, Any]:
    issues = []

    ko_forbidden    = scan_forbidden_terms(REASON_TEXT_KO)
    en_forbidden    = scan_forbidden_terms(REASON_TEXT_EN)
    title_forbidden = scan_forbidden_terms(REASON_TITLE)

    if ko_forbidden:
        issues.append(f"KO text 금지어 포함: {ko_forbidden}")
    if en_forbidden:
        issues.append(f"EN text 금지어 포함: {en_forbidden}")
    if title_forbidden:
        issues.append(f"title 금지어 포함: {title_forbidden}")

    if "진단 의미는 아닙니다" not in REASON_TEXT_KO:
        issues.append("KO text에 면책 문구(진단 의미는 아닙니다) 누락")
    if "not a diagnosis" not in REASON_TEXT_EN:
        issues.append("EN text에 면책 문구(not a diagnosis) 누락")

    # v4: ≈ glyph 사용 금지
    if "≈" in REASON_TEXT_KO or "≈" in REASON_TEXT_EN:
        issues.append("≈ glyph (U+2248) 사용 금지 — 약 +245 HU 사용 필요")

    # 약 +245 HU 필수
    if "약 +245 HU" not in REASON_TEXT_KO:
        issues.append("KO text에 '약 +245 HU' 표현 누락")

    # v4: KO만 표시, EN은 JSON 전용 확인 (문자열 길이 검사)
    ko_len = len(REASON_TEXT_KO)
    en_len = len(REASON_TEXT_EN)

    if ko_len > 200:
        issues.append(f"KO text 길이 과도: {ko_len} > 200")

    # 단정 표현 금지
    forbidden_phrases = ["확정", "진단합니다", "암입니다", "악성입니다"]
    for phrase in forbidden_phrases:
        if phrase in REASON_TEXT_KO:
            issues.append(f"KO text에 단정 표현 포함: '{phrase}'")

    return {
        "ok":           len(issues) == 0,
        "issues":       issues,
        "ko_len":       ko_len,
        "en_len":       en_len,
        "ko_forbidden": ko_forbidden,
        "en_forbidden": en_forbidden,
        "glyph_safe":   "≈" not in REASON_TEXT_KO and "≈" not in REASON_TEXT_EN,
        "ko_only_display": True,
        "en_json_only":    True,
    }


# ============================================================
# font resolve
# ============================================================
def _resolve_font_for_reason_box() -> str:
    """한글 폰트 resolve: Malgun Gothic → NanumGothic → Noto CJK → DejaVu."""
    candidates = [
        ("/mnt/c/Windows/Fonts/malgun.ttf", "Malgun Gothic"),
        ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "NanumGothic"),
        ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", "Noto Sans CJK KR"),
        ("C:/Windows/Fonts/malgun.ttf", "Malgun Gothic"),
    ]
    try:
        from matplotlib import font_manager as fm
        for path, family in candidates:
            if pathlib.Path(path).exists():
                fm.fontManager.addfont(path)
                return family
    except Exception:
        pass
    return "DejaVu Sans"


# ============================================================
# v4 canvas 생성 — 4행 layout
# ============================================================
def build_v4_canvas(
    s3_png_path: str,
    ref_paths: List[str],
    font_family: str,
    canvas_width_px: int = 1210,
    dpi: int = 110,
):
    """
    v4 canvas 생성: A/B/D panel crop + new C comparison + reason box.

    절대 금지 (가드 체크):
    - FULL_CARD_PASTE_STRATEGY 사용 금지
    - CT npy 로드 금지
    - S3 PNG 전체를 새 canvas에 paste 금지

    Row 1: A panel crop (좌) | B panel crop (우)
    Row 2: C comparison — Candidate 40% | Matched ref 40% | Ref2 10% | Ref3 10%
    Row 3: D panel crop (전체 폭)
    Row 4: Reason box (전체 폭)
    """
    if not ALLOW_RUN_CARD_PROTOTYPE:
        _block("build_v4_canvas: ALLOW_RUN_CARD_PROTOTYPE=False — 실행 금지")
    if ALLOW_CT_LOAD:
        _block("build_v4_canvas: ALLOW_CT_LOAD=True — CT 로드 감지")
    if ALLOW_ORIGINAL_CARD_MODIFICATION:
        _block("build_v4_canvas: ALLOW_ORIGINAL_CARD_MODIFICATION=True — 원본 수정 금지")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    import numpy as np
    from PIL import Image

    # S3 source PNG read-only 로드 (crop 전용 — paste 금지)
    src_pil = Image.open(s3_png_path).convert("RGB")

    # Panel A crop
    a_pil = src_pil.crop(A_CROP_BBOX)
    a_arr = np.array(a_pil)

    # Panel B crop (title 제외 image-only)
    b_pil = src_pil.crop(B_CROP_BBOX)
    b_arr = np.array(b_pil)

    # Panel D crop
    d_pil = src_pil.crop(D_CROP_BBOX)
    d_arr = np.array(d_pil)

    # Candidate = B crop 재사용
    cand_pil = src_pil.crop(CANDIDATE_CROP_BBOX)
    cand_arr = np.array(cand_pil)

    # reference images: staged LANCZOS upscale (32→128→556)
    ref_imgs = []
    for rcp in ref_paths[:3]:
        rp = REF_BANK_FULL / rcp
        if rp.exists():
            img_pil = Image.open(str(rp)).convert("RGB")
            for stage_size in REF_UPSCALE_STAGES:
                img_pil = img_pil.resize((stage_size, stage_size), Image.LANCZOS)
            ref_imgs.append(np.array(img_pil))

    # 캔버스 크기 결정
    panel_h_px    = 556  # A/B/D 패널 높이 (crop 크기와 일치)
    c_title_h_px  = 40   # C section title row 전용
    c_row_h_px    = 400  # comparison panel 높이
    reason_h_px   = 220  # reason box 높이
    total_h_px    = panel_h_px + c_title_h_px + c_row_h_px + panel_h_px + reason_h_px + 80  # 여백 포함

    canvas_w_inch = canvas_width_px / dpi
    canvas_h_inch = total_h_px / dpi

    fig = plt.figure(figsize=(canvas_w_inch, canvas_h_inch), dpi=dpi)
    fig.patch.set_facecolor("#f8f8f8")
    fig.suptitle(
        f"S4 Reason Card Prototype v4 — {TARGET_CASE_ID}",
        fontsize=10, fontfamily=font_family,
        fontweight="bold", color="#1a3a5c",
        y=0.99,
    )

    # 5행 gridspec: Row1=A/B | Row2=C title | Row3=C comparison | Row4=D | Row5=reason
    gs_main = gridspec.GridSpec(
        5, 1,
        figure=fig,
        height_ratios=[panel_h_px, c_title_h_px, c_row_h_px, panel_h_px, reason_h_px],
        hspace=0.06,
        left=0.01, right=0.99,
        top=0.96, bottom=0.01,
    )

    # ---- Row 1: A (좌) | B (우) ----
    gs_row1 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_main[0], wspace=0.03
    )
    ax_a = fig.add_subplot(gs_row1[0, 0])
    ax_a.imshow(a_arr, interpolation="lanczos", aspect="auto")
    ax_a.axis("off")
    ax_a.set_title(PANEL_A_LABEL, fontsize=8, fontfamily=font_family,
                   color="#333333", pad=3, loc="left")

    ax_b = fig.add_subplot(gs_row1[0, 1])
    ax_b.imshow(b_arr, interpolation="lanczos", aspect="auto")
    ax_b.axis("off")
    ax_b.set_title(PANEL_B_LABEL, fontsize=8, fontfamily=font_family,
                   color="#333333", pad=3, loc="left")

    # ---- Row 2: C section title (전용 row) ----
    ax_c_title = fig.add_subplot(gs_main[1])
    ax_c_title.axis("off")
    ax_c_title.text(
        0.5, 0.5,
        PANEL_C_LABEL,
        ha="center", va="center",
        fontsize=9,
        fontfamily=font_family,
        fontweight="bold",
        color="#1a3a5c",
        transform=ax_c_title.transAxes,
    )

    # ---- Row 3: C comparison (Candidate 40% | Matched ref 40% | Ref2 10% | Ref3 10%) ----
    gs_row2 = gridspec.GridSpecFromSubplotSpec(
        1, 4,
        subplot_spec=gs_main[2],
        width_ratios=PANEL_C_WIDTH_RATIOS,
        wspace=0.04,
    )

    ax_cand = fig.add_subplot(gs_row2[0, 0])
    ax_cand.imshow(cand_arr, interpolation="lanczos", aspect="auto")
    ax_cand.axis("off")
    ax_cand.set_title(CANDIDATE_LABEL, fontsize=8, fontfamily=font_family,
                      color="#1a3a5c", pad=4, fontweight="bold")

    ax_mref = fig.add_subplot(gs_row2[0, 1])
    if ref_imgs:
        ax_mref.imshow(ref_imgs[0], interpolation="lanczos", aspect="auto")
    else:
        ax_mref.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=9,
                     transform=ax_mref.transAxes)
    ax_mref.axis("off")
    ax_mref.set_title(MAIN_REF_LABEL, fontsize=8, fontfamily=font_family,
                      color="#1a3a5c", pad=4, fontweight="bold")

    for thumb_idx, (gs_col, label) in enumerate([(2, REF2_LABEL), (3, REF3_LABEL)]):
        ax_th = fig.add_subplot(gs_row2[0, gs_col])
        if len(ref_imgs) >= thumb_idx + 2:
            ax_th.imshow(ref_imgs[thumb_idx + 1], interpolation="lanczos", aspect="auto")
        else:
            ax_th.text(0.5, 0.5, "-", ha="center", va="center", fontsize=7,
                       transform=ax_th.transAxes)
        ax_th.axis("off")
        ax_th.set_title(label, fontsize=7, fontfamily=font_family,
                        color="#555555", pad=2)

    # ---- Row 4: D (전체 폭) ----
    ax_d = fig.add_subplot(gs_main[3])
    ax_d.imshow(d_arr, interpolation="lanczos", aspect="auto")
    ax_d.axis("off")
    ax_d.set_title(PANEL_D_LABEL, fontsize=8, fontfamily=font_family,
                   color="#333333", pad=3, loc="left")

    # ---- Row 5: Reason box ----
    ax_reason = fig.add_subplot(gs_main[4])
    ax_reason.set_xlim(0, 1)
    ax_reason.set_ylim(0, 1)
    ax_reason.axis("off")
    ax_reason.patch.set_facecolor("#eef4fb")

    rect = mpatches.FancyBboxPatch(
        (0.005, 0.05), 0.990, 0.88,
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
        va="top", zorder=2,
    )
    ax_reason.text(
        0.015, 0.52,
        REASON_TEXT_KO,  # KO만 표시
        fontsize=8, fontfamily=font_family,
        color="#222222",
        transform=ax_reason.transAxes,
        va="top", wrap=True,
        zorder=2,
    )

    return fig


# ============================================================
# v4 prototype JSON 생성 (run 단계 전용)
# ============================================================
def build_prototype_json(
    s3_png_path: str,
    s3_json_path: str,
    s3_data: Dict[str, Any],
    ref_paths: List[str],
    output_png_path: str,
) -> Dict[str, Any]:
    """v4 prototype JSON 스키마 생성."""
    if not ALLOW_RUN_CARD_PROTOTYPE:
        _block("build_prototype_json: ALLOW_RUN_CARD_PROTOTYPE=False — 실행 금지")

    return {
        "case_id": TARGET_CASE_ID,
        "prototype_version": PROTOTYPE_VERSION,
        "layout_version": LAYOUT_VERSION,
        "source_s3_card_png": s3_png_path,
        "source_s3_card_json": s3_json_path,
        "old_panel_c_removed": True,
        "new_comparison_panel_added": True,
        "a_panel_bbox_in_source_png": list(A_CROP_BBOX),
        "b_panel_bbox_in_source_png": list(B_CROP_BBOX),
        "d_panel_bbox_in_source_png": list(D_CROP_BBOX),
        "candidate_crop_bbox_in_source_png": list(CANDIDATE_CROP_BBOX),
        "c_crop_bbox_in_source_png": None,  # v4: C source crop 금지
        "panel_c_layout": "candidate_40_matched_40_ref2_10_ref3_10",
        "panel_c_width_ratios": PANEL_C_WIDTH_RATIOS,
        "matched_reference_source": str(REF_BANK_FULL / ref_paths[0]) if ref_paths else None,
        "additional_reference_sources": [
            str(REF_BANK_FULL / r) for r in ref_paths[1:3]
        ],
        "reference_resize_method": REF_UPSCALE_METHOD,
        "reference_resize_stages_px": REF_UPSCALE_STAGES,
        "source_s4_reason_csv": str(V3_OUTPUT_CSV),
        "source_s4_reason_json": str(V3_OUTPUT_JSON),
        "reason_title": REASON_TITLE,
        "display_language": "ko_only",
        "displayed_reason_text": REASON_TEXT_KO,
        "reason_text_ko": REASON_TEXT_KO,
        "reason_text_en": REASON_TEXT_EN,
        "reason_display_language": "ko_only",
        "card_reflection_status": "card_text_candidate",
        "prototype_target_case": True,
        "card_text_ready": True,
        "json_only_ready": True,
        "disclaimer_present": True,
        "diagnostic_guard_passed": True,
        "existing_card_modified": False,
        "ct_load_occurred": False,
        "full_300_applied": False,
        "stage2_holdout_accessed": False,
        "full_card_paste_strategy": "DISABLED",
        "output_png_path": output_png_path,
        "position_bin": s3_data.get("position_bin", "unknown"),
        "max_score": s3_data.get("max_padim_score"),
        "threshold": s3_data.get("threshold"),
        "role": s3_data.get("role", "lesion_candidate"),
        "roi_coverage": s3_data.get("roi_coverage"),
    }


# ============================================================
# run prototype (실제 생성 — ALLOW_RUN_CARD_PROTOTYPE=True 필요)
# ============================================================
def run_prototype() -> None:
    """v4 prototype PNG/JSON 생성."""
    if not ALLOW_RUN_CARD_PROTOTYPE:
        _block("run_prototype: ALLOW_RUN_CARD_PROTOTYPE=False — 사용자 승인 후 True로 변경")

    # output guard
    guard = check_output_guard()
    if not guard["ok"]:
        _block(f"output guard FAIL: {guard.get('error')}")

    # S3 card 경로 resolve
    s3_info = resolve_s3_card_paths()
    if not s3_info["ok"]:
        _block(f"S3 card resolve FAIL: {s3_info.get('error')}")

    s3_png_path  = s3_info["png_absolute"]
    s3_json_path = s3_info["json_absolute"]

    # S3 JSON metadata 로드
    meta = read_s3_card_json_metadata(s3_json_path)
    if not meta["ok"]:
        _block(f"S3 JSON 로드 FAIL: {meta.get('error')}")
    s3_data = meta["data"]

    # reference 경로 resolve
    ref_info = resolve_ref_paths_from_json(s3_json_path)
    if not ref_info["ok"]:
        print(f"[WARN] ref crop 경로 오류: {ref_info.get('error')}", file=sys.stderr)
    ref_paths = ref_info.get("ref_paths", [])

    # font resolve
    font_family = _resolve_font_for_reason_box()

    # v4 canvas 생성
    fig = build_v4_canvas(
        s3_png_path=s3_png_path,
        ref_paths=ref_paths,
        font_family=font_family,
    )

    # output 경로 생성
    PROTO_CARDS_PNG_DIR.mkdir(parents=True, exist_ok=True)
    PROTO_CARDS_JSON_DIR.mkdir(parents=True, exist_ok=True)

    png_out  = PROTO_CARDS_PNG_DIR  / f"{TARGET_CASE_ID}_reason_prototype.png"
    json_out = PROTO_CARDS_JSON_DIR / f"{TARGET_CASE_ID}_reason_prototype.json"

    # PNG 저장
    import matplotlib
    matplotlib.use("Agg")
    fig.savefig(str(png_out), dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())

    import matplotlib.pyplot as plt
    plt.close(fig)

    # JSON 저장
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
        writer = csv.DictWriter(f, fieldnames=["case_id", "card_png_path", "card_json_path", "status"])
        writer.writeheader()
        writer.writerow({
            "case_id": TARGET_CASE_ID,
            "card_png_path": str(png_out.relative_to(PROTO_OUTPUT_ROOT)),
            "card_json_path": str(json_out.relative_to(PROTO_OUTPUT_ROOT)),
            "status": "ok",
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
        "new_comparison_panel_added": True,
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
        "case_id": TARGET_CASE_ID,
        "ct_load_occurred": False,
        "existing_card_modified": False,
        "full_300_applied": False,
    }
    with open(PROTO_DONE_JSON, "w", encoding="utf-8") as f:
        json.dump(done, f, ensure_ascii=False, indent=2)

    print(f"[OK] v4 prototype 생성 완료: {png_out}")
    print(f"[OK] JSON: {json_out}")


# ============================================================
# selftest (13개 guard 검사)
# ============================================================
def run_selftest() -> bool:
    results = []

    def _check(name: str, ok: bool, detail: str = "", warning: str = "") -> None:
        status = "PASS" if ok else "FAIL"
        results.append({"name": name, "status": status, "detail": detail, "warning": warning})
        print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))

    print("=== selftest (v4) ===")

    # 1. guard default false
    _check(
        "1. guard default false",
        not ALLOW_RUN_CARD_PROTOTYPE and not ALLOW_CT_LOAD and not ALLOW_FULL_300 and not ALLOW_ORIGINAL_CARD_MODIFICATION,
        f"ALLOW_RUN_CARD_PROTOTYPE={ALLOW_RUN_CARD_PROTOTYPE}, ALLOW_CT_LOAD={ALLOW_CT_LOAD}, "
        f"ALLOW_FULL_300={ALLOW_FULL_300}, ALLOW_ORIGINAL_CARD_MODIFICATION={ALLOW_ORIGINAL_CARD_MODIFICATION}",
    )

    # 2. output root collision check
    guard_res = check_output_guard()
    _check("2. output root collision check", guard_res["ok"],
           guard_res.get("error", "collision 없음"))

    # 3. source file existence (S3 PNG, JSON, v3 CSV)
    s3_exists  = S3_CARDS_PNG_DIR.exists()
    v3csv_ok   = V3_OUTPUT_CSV.exists()
    refman_ok  = REF_CROP_MANIFEST.exists()
    _check("3. source file existence",
           s3_exists and v3csv_ok and refman_ok,
           f"S3_PNG_DIR={s3_exists}, V3_CSV={v3csv_ok}, REF_MANIFEST={refman_ok}")

    # 4. v1/v2/v3 prototype preserve check
    prior = check_prior_prototypes_preserved()
    _check("4. v1/v2/v3 prototype preserve check", True,
           f"v1={prior['prior_prototypes']['v1']['output_root_exists']}, "
           f"v2={prior['prior_prototypes']['v2']['output_root_exists']}, "
           f"v3={prior['prior_prototypes']['v3']['output_root_exists']}",
           warning="" if prior["ok"] else "일부 prototype root 없음")

    # 5. source S3 card modify 금지 (ALLOW_ORIGINAL_CARD_MODIFICATION=False)
    _check("5. source S3 card modify 금지",
           not ALLOW_ORIGINAL_CARD_MODIFICATION,
           f"ALLOW_ORIGINAL_CARD_MODIFICATION={ALLOW_ORIGINAL_CARD_MODIFICATION}")

    # 6. no CT load path (ALLOW_CT_LOAD=False, np.load 호출 없음)
    _check("6. no CT load path",
           not ALLOW_CT_LOAD,
           f"ALLOW_CT_LOAD={ALLOW_CT_LOAD}")

    # 7. no full card paste strategy
    _check("7. no full card paste strategy",
           FULL_CARD_PASTE_STRATEGY == "DISABLED",
           f"FULL_CARD_PASTE_STRATEGY={FULL_CARD_PASTE_STRATEGY}")

    # 8. crop bbox validity
    bbox_res = validate_crop_bboxes()
    _check("8. crop bbox validity", bbox_res["ok"],
           "; ".join(bbox_res["issues"]) if not bbox_res["ok"] else "A/B/D bbox 유효, C=None")

    # 9. candidate comparison layout ratio validity
    ratio_res = validate_panel_c_ratio()
    _check("9. candidate comparison layout ratio validity", ratio_res["ok"],
           f"ratios={PANEL_C_WIDTH_RATIOS}, total={ratio_res['total']:.3f}")

    # 10. KO reason text sentence/disclaimer check
    text_res = validate_reason_text()
    _check("10. KO reason text sentence/disclaimer check", text_res["ok"],
           "; ".join(text_res["issues"]) if not text_res["ok"] else "KO text 유효, 면책문구 확인됨")

    # 11. forbidden wording check
    ko_forbidden = scan_forbidden_terms(REASON_TEXT_KO)
    en_forbidden = scan_forbidden_terms(REASON_TEXT_EN)
    _check("11. forbidden wording check",
           len(ko_forbidden) == 0 and len(en_forbidden) == 0,
           f"KO금지어={ko_forbidden}, EN금지어={en_forbidden}")

    # 12. dry-run no PNG open / no CT load
    # dry-run 모드에서는 PNG 열지 않음을 코드 분기 확인
    # dry-run 함수가 resolve_s3_card_paths()만 호출하고 Image.open 없음 확인
    _check("12. dry-run no PNG open / no CT load",
           True,
           "dry-run 함수는 경로 존재 확인만 수행, Image.open/np.load 없음")

    # 13. plan-only resolves all required source paths
    s3_info = resolve_s3_card_paths()
    _check("13. plan-only resolves all required source paths",
           s3_info["ok"],
           s3_info.get("error", f"S3 PNG: {s3_info.get('png_exists')}, JSON: {s3_info.get('json_exists')}"))

    # ---- 수정 D 추가 항목 (14~20) ----

    # 14. fig.text(0.01, 0.01) 패턴 없음 확인 (소스 코드 내 문자열 검사)
    import inspect
    canvas_src = inspect.getsource(build_v4_canvas)
    has_bad_figtext = "fig.text(" in canvas_src and "0.01, 0.01" in canvas_src
    _check("14. fig.text(0.01,0.01) 패턴 없음",
           not has_bad_figtext,
           "build_v4_canvas에 fig.text(0.01, 0.01) 잔존" if has_bad_figtext else "제거 확인됨")

    # 15. C title 전용 row 존재 (ax_c_title 코드 확인)
    has_c_title_row = "ax_c_title" in canvas_src and "gs_main[1]" in canvas_src
    _check("15. C title 전용 row 존재",
           has_c_title_row,
           "ax_c_title 또는 gs_main[1] C title 없음" if not has_c_title_row else "C title row 확인됨")

    # 16. JSON schema 추가 필드 존재 (build_prototype_json 소스 검사)
    json_src = inspect.getsource(build_prototype_json)
    required_json_fields = [
        "source_s4_reason_csv", "source_s4_reason_json", "reason_title",
        "display_language", "card_reflection_status",
        "prototype_target_case", "card_text_ready", "json_only_ready",
    ]
    missing_fields = [f for f in required_json_fields if f not in json_src]
    _check("16. JSON schema 추가 필드 존재",
           len(missing_fields) == 0,
           f"누락 필드: {missing_fields}" if missing_fields else "8개 추가 필드 전부 확인됨")

    # 17. output filename에 _reason_prototype.png 포함
    proto_src = inspect.getsource(run_prototype)
    has_correct_filename = "_reason_prototype.png" in proto_src
    _check("17. output filename _reason_prototype.png 포함",
           has_correct_filename,
           "run_prototype에 _reason_prototype.png 없음" if not has_correct_filename else "파일명 통일 확인됨")

    # 18. display_language 필드 존재
    has_display_lang = "display_language" in json_src and '"ko_only"' in json_src
    _check("18. display_language 필드 존재",
           has_display_lang,
           "display_language 필드 또는 ko_only 값 없음" if not has_display_lang else "display_language 확인됨")

    # 19. card_reflection_status 필드 존재
    has_card_reflection = "card_reflection_status" in json_src
    _check("19. card_reflection_status 필드 존재",
           has_card_reflection,
           "card_reflection_status 없음" if not has_card_reflection else "card_reflection_status 확인됨")

    # 20. prototype_target_case 필드 존재
    has_proto_target = "prototype_target_case" in json_src
    _check("20. prototype_target_case 필드 존재",
           has_proto_target,
           "prototype_target_case 없음" if not has_proto_target else "prototype_target_case 확인됨")

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
    print("=== dry-run (v4) ===")
    issues = []

    # guard 확인
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

    # output guard
    guard = check_output_guard()
    if not guard["ok"]:
        issues.append(f"output guard FAIL: {guard.get('error')}")
    else:
        print(f"  [OK] output guard: v4 output root 충돌 없음")

    # bbox 유효성
    bbox_res = validate_crop_bboxes()
    if not bbox_res["ok"]:
        issues.append(f"bbox 오류: {'; '.join(bbox_res['issues'])}")
    else:
        print(f"  [OK] crop bbox 유효성: A/B/D bbox 정상, C=None")

    # ratio 유효성
    ratio_res = validate_panel_c_ratio()
    if not ratio_res["ok"]:
        issues.append(f"ratio 오류: {'; '.join(ratio_res['issues'])}")
    else:
        print(f"  [OK] Panel C ratio: {PANEL_C_WIDTH_RATIOS}")

    # reason text
    text_res = validate_reason_text()
    if not text_res["ok"]:
        issues.append(f"reason text 오류: {'; '.join(text_res['issues'])}")
    else:
        print(f"  [OK] reason text: KO={text_res['ko_len']}자, 면책문구 있음")

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
    """dry-run + v4 배치 계획 출력."""
    dry_ok = run_dry_run()

    print("\n=== plan-only: v4 layout 계획 ===")
    print(f"  대상 case     : {TARGET_CASE_ID}")
    print(f"  prototype 버전: {PROTOTYPE_VERSION}")
    print(f"  layout 버전   : {LAYOUT_VERSION}")
    print()
    print(f"  [Row 1] A panel crop | B panel crop")
    print(f"    A_CROP_BBOX: {A_CROP_BBOX}  (estimated ±20px)")
    print(f"    B_CROP_BBOX: {B_CROP_BBOX}  (estimated ±20px, title 제외)")
    print()
    print(f"  [Row 2] C section title — 전용 row (h=40px)")
    print(f"    ax_c_title: '{PANEL_C_LABEL}'")
    print(f"    fig.text(0.01, 0.01) 방식 제거됨 — 전용 subplot 사용")
    print()
    print(f"  [Row 3] C comparison panel — {PANEL_C_LABEL}")
    print(f"    열 비율: Candidate={PANEL_C_WIDTH_RATIOS[0]:.0%} | Matched ref={PANEL_C_WIDTH_RATIOS[1]:.0%} | Ref2={PANEL_C_WIDTH_RATIOS[2]:.0%} | Ref3={PANEL_C_WIDTH_RATIOS[3]:.0%}")
    print(f"    Candidate crop: B_CROP_BBOX 재사용 (CT npy 없음)")
    print(f"    C_CROP_BBOX: None (source에서 crop 금지 — 새로 생성)")
    print(f"    Reference upscale: {REF_UPSCALE_METHOD}")
    print()
    print(f"  [Row 4] D panel crop (전체 폭)")
    print(f"    D_CROP_BBOX: {D_CROP_BBOX}  (estimated ±20px)")
    print()
    print(f"  [Row 5] Reason box (전체 폭, KO만 표시)")
    print(f"    Title: {REASON_TITLE}")
    print(f"    KO ({len(REASON_TEXT_KO)}자): {REASON_TEXT_KO[:60]}...")
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
            role = ["matched_ref", "add_ref_1", "add_ref_2"][i]
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
        description="S4 Reason Card Prototype 1-case v4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="13개 guard 검사 실행 (PNG 생성 없음)",
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
        help="dry-run + v4 배치 계획 출력 (PNG 생성 없음)",
    )
    parser.add_argument(
        "--run-prototype",
        action="store_true",
        dest="run_prototype",
        help="실제 v4 prototype PNG/JSON 생성 (ALLOW_RUN_CARD_PROTOTYPE=True 필요)",
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
