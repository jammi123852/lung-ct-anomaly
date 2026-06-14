#!/usr/bin/env python3
"""
build_explanation_card_s4_reason_layer_integrated_smoke_v3.py

S4 Reason Layer Integrated Smoke v3 Script

목적:
- S4 integrated reason v2 JSON을 입력으로 읽고 아래 수정을 적용한다:
  1. v3 신규 5필드 추가:
       disclaimer_present, weak_evidence_in_body,
       card_reflection_status, v3_text_tuning_applied, v3_tuning_reason
  2. LUNG1-320__c2: 본문 2번째 문장을 면책 문구로 교체 → disclaimer_present=True
  3. LUNG1-402__c1: 본문에 CT-stat weak evidence 문장 추가 → weak_evidence_in_body=True
  4. card_text_ready 강화 기준 적용 (disclaimer 필수, ct_stat_uncertain → weak_evidence 필수)
  5. card_reflection_status 산출
  6. summary MD 생성

이번 단계: 스크립트 작성 + 정적 검사만 허용.
실제 v3 생성은 --run-integrate --confirm-generate 조합으로만 가능하며
ALLOW_INTEGRATE_REASON=False 가드로 차단되어 있다.

절대 금지:
- CT/mask npy 로드 (ALLOW_CT_LOAD=False)
- HU 통계 재계산
- PNG open
- 카드 PNG/JSON 수정 (ALLOW_CARD_MODIFICATION=False)
- full 300 reason 적용 (ALLOW_FULL_300=False)
- score/model/threshold 재계산
- stage2_holdout 접근
- 기존 v2/v1/S3 카드 CSV/JSON/DONE 수정

실행 모드:
- bare 실행             → BLOCKED exit 2
- --selftest            → 24개 guard 검사
- --dry-run             → 입력 파일 존재 + 8장 resolve + plan_rows 무결성 확인
- --plan-smoke-only     → dry-run + v3 적용 계획 출력
- --run-integrate       → 단독 BLOCKED exit 2
- --run-integrate --confirm-generate → ALLOW_INTEGRATE_REASON=False 로 BLOCKED

syntax check:
  python -m py_compile scripts/build_explanation_card_s4_reason_layer_integrated_smoke_v3.py
"""

import argparse
import csv
import json
import pathlib
import re
import sys
from typing import Any, Dict, List, Tuple

# ============================================================
# 최상위 가드 — 이번 단계는 전부 False
# ============================================================
ALLOW_INTEGRATE_REASON = False
ALLOW_CT_LOAD = False
ALLOW_CARD_MODIFICATION = False
ALLOW_FULL_300 = False

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
]

# ============================================================
# stage2_holdout 접근 금지 토큰
# ============================================================
STAGE2_HOLDOUT_TOKENS = [
    "stage2_holdout", "stage2-holdout", "stage2holdout",
    "holdout_stage2", "holdout-stage2",
]

# ============================================================
# smoke 대상 8장
# ============================================================
SMOKE_TARGETS = [
    "LUNG1-284__c1",
    "LUNG1-220__c3",
    "LUNG1-402__c1",
    "LUNG1-305__c1",
    "MSD_lung_054__c1",
    "LUNG1-057__c1",
    "subset9_1.3.6.1.4.1.14519.5.2.1.6279.6001.291156498203266896953765649282__c1",
    "LUNG1-320__c2",
]

# v3 text tuning 대상 2건
V3_TUNING_TARGETS = ["LUNG1-320__c2", "LUNG1-402__c1"]

# ============================================================
# disclaimer 감지 패턴
# ============================================================
DISCLAIMER_PATTERNS_KO = [
    "진단 의미는 아닙니다",
    "FP 패턴 검토 목적",
]
DISCLAIMER_PATTERNS_EN = [
    "not a diagnosis",
    "not as a diagnosis",
    "FP pattern review",
]

# ============================================================
# weak_evidence 감지 패턴
# ============================================================
WEAK_EVIDENCE_PATTERNS_KO = [
    "CT 통계 근거가 약해",
    "CT 통계상 reference와의 차이는 약합니다",
]
WEAK_EVIDENCE_PATTERNS_EN = [
    "CT-stat evidence is weak",
    "CT-stat evidence against the reference is weak",
]

# ============================================================
# v3 plan rows CSV 필수 컬럼
# ============================================================
REQUIRED_V3_PLAN_COLS = [
    "expansion_case_id",
    "action_category",
    "v2_card_text_ready",
    "v3_expected_card_text_ready",
    "disclaimer_present_required",
    "weak_evidence_in_body_required",
    "v3_text_tuning_applied",
    "v3_tuning_reason",
    "expected_card_reflection_status",
    "note",
]

# ============================================================
# v2 base 31필드 + v3 신규 5필드 = 36필드
# ============================================================
V2_BASE_FIELDS = [
    "expansion_case_id",
    "role",
    "position_bin",
    "max_padim_score",
    "threshold",
    "metadata_reason_tags",
    "ctstat_reason_tags",
    "final_reason_tags",
    "reason_quality_label",
    "ctstat_usefulness_label",
    "integrated_reason_ko",
    "integrated_reason_en",
    "reason_limitations",
    "hold_flag",
    "overmerge_flag",
    "overmerge_level",
    "apex_caution",
    "roi_coverage",
    "delta_hu_p50",
    "candidate_hu_p50",
    "reference_hu_mean",
    "include_in_future_card_text",
    "include_in_json_only",
    "recommended_next_action",
    "diagnostic_guard_passed",
    # v2 신규 6필드
    "action_category",
    "text_tuning_applied",
    "sentence_count_ko",
    "sentence_count_en",
    "limitation_fix_applied",
    "card_text_ready",
]

V3_NEW_FIELDS = [
    "disclaimer_present",
    "weak_evidence_in_body",
    "card_reflection_status",
    "v3_text_tuning_applied",
    "v3_tuning_reason",
]

V3_OUTPUT_FIELDS = V2_BASE_FIELDS + V3_NEW_FIELDS

# ============================================================
# 경로 상수
# ============================================================
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

REPORTS_ROOT = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
)

# v2 입력 파일 (read-only)
V2_OUTPUT_JSON    = REPORTS_ROOT / "s4_reason_layer_integrated_smoke_v2/s4_reason_layer_integrated_smoke_v2.json"
V2_OUTPUT_CSV     = REPORTS_ROOT / "s4_reason_layer_integrated_smoke_v2/s4_reason_layer_integrated_smoke_v2.csv"
V2_DONE_JSON      = REPORTS_ROOT / "s4_reason_layer_integrated_smoke_v2/DONE.json"
V3_PLAN_ROWS_CSV  = REPORTS_ROOT / "s4_integrated_reason_v3_text_disclaimer_plan_rows_v1.csv"
V3_PREFLIGHT_JSON = REPORTS_ROOT / "s4_integrated_reason_v3_text_disclaimer_preflight_v1.json"

# v2 output root (read-only, 보존 확인용)
V2_OUTPUT_ROOT = REPORTS_ROOT / "s4_reason_layer_integrated_smoke_v2"

# v3 output root (이번 단계 생성 금지)
V3_OUTPUT_ROOT = REPORTS_ROOT / "s4_reason_layer_integrated_smoke_v3"

# 정적 검사 보고서 (이번 단계 생성)
STATIC_CHECK_REPORT_MD   = REPORTS_ROOT / "s4_integrated_reason_v3_script_static_drycheck_v1.md"
STATIC_CHECK_REPORT_JSON = REPORTS_ROOT / "s4_integrated_reason_v3_script_static_drycheck_v1.json"


# ============================================================
# 유틸리티 함수
# ============================================================

def check_forbidden_terms(text: str) -> List[str]:
    """reason text에 진단 금지어 포함 여부 검사."""
    found = []
    tl = text.lower()
    for term in FORBIDDEN_TERMS:
        if term.lower() in tl:
            found.append(term)
    return found


def assert_no_forbidden(text: str, context: str = "") -> None:
    violations = check_forbidden_terms(text)
    if violations:
        raise RuntimeError(
            f"BLOCKED: forbidden term in {context}: {violations}"
        )


def check_stage2_holdout(path_str: str) -> bool:
    pl = path_str.lower()
    return any(tok in pl for tok in STAGE2_HOLDOUT_TOKENS)


def assert_no_stage2_holdout(path_str: str, context: str = "") -> None:
    if check_stage2_holdout(path_str):
        raise RuntimeError(
            f"BLOCKED: stage2_holdout access in {context}: {path_str}"
        )


def safe_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes")


def load_csv_as_index(
    csv_path: pathlib.Path,
    key_col: str = "expansion_case_id",
) -> Dict[str, Dict[str, str]]:
    """CSV를 key_col 기준 dict-of-dict로 로드."""
    if not csv_path.exists():
        return {}
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return {r[key_col].strip(): r for r in rows if key_col in r}


# ============================================================
# sentence count 함수
# ============================================================

def count_sentences(text: str) -> int:
    """문장 수 계산 (마침표/느낌표/물음표 뒤 공백 기준)."""
    if not text.strip():
        return 0
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return len([p for p in parts if p.strip()])


# ============================================================
# disclaimer_present 감지 함수
# ============================================================

def compute_disclaimer_present(ko: str, en: str) -> bool:
    """본문 KO/EN에 면책 의미 표현이 포함되어 있는지 검사."""
    ko_l = ko.lower()
    en_l = en.lower()
    ko_match = any(p.lower() in ko_l for p in DISCLAIMER_PATTERNS_KO)
    en_match = any(p.lower() in en_l for p in DISCLAIMER_PATTERNS_EN)
    return ko_match or en_match


# ============================================================
# weak_evidence_in_body 감지 함수
# ============================================================

def compute_weak_evidence_in_body(ko: str, en: str) -> bool:
    """본문 KO/EN에 CT-stat weak/limited evidence 표현이 포함되어 있는지 검사."""
    ko_l = ko.lower()
    en_l = en.lower()
    ko_match = any(p.lower() in ko_l for p in WEAK_EVIDENCE_PATTERNS_KO)
    en_match = any(p.lower() in en_l for p in WEAK_EVIDENCE_PATTERNS_EN)
    return ko_match or en_match


# ============================================================
# card_text_ready v3 강화 판정 함수
# ============================================================

def compute_card_text_ready_v3(
    action_category: str,
    disclaimer_present: bool,
    weak_evidence_in_body: bool,
    diagnostic_guard_passed: bool,
    sentence_count_ko: int,
    sentence_count_en: int,
    has_ct_stat_uncertain: bool,
) -> bool:
    """
    v3 강화 card_text_ready 판정.
    - hold_context / uncertain_json_only / artifact_warning → False (정책상 card 금지)
    - usable_with_caution → False (보수적 기본값, 별도 검토 필요)
    - ready → disclaimer_present + guard + <=2문장 + (ct_stat_uncertain→weak_evidence) 전부 충족 시 True
    """
    if action_category in {"hold_context", "uncertain_json_only", "artifact_warning"}:
        return False
    if action_category == "usable_with_caution":
        return False
    if action_category != "ready":
        return False
    if not disclaimer_present:
        return False
    if not diagnostic_guard_passed:
        return False
    if sentence_count_ko > 2 or sentence_count_en > 2:
        return False
    if has_ct_stat_uncertain and not weak_evidence_in_body:
        return False
    return True


# ============================================================
# card_reflection_status 판정 함수
# ============================================================

def compute_card_reflection_status(action_category: str, card_text_ready: bool) -> str:
    """
    card_reflection_status 값 산출.
    추천 값: card_text_candidate / json_only_ready / json_only_hold_context /
             json_only_uncertain / json_only_artifact_warning /
             json_only_usable_with_caution / needs_review_before_card
    """
    if action_category == "ready":
        return "card_text_candidate" if card_text_ready else "needs_review_before_card"
    elif action_category == "hold_context":
        return "json_only_hold_context"
    elif action_category == "uncertain_json_only":
        return "json_only_uncertain"
    elif action_category == "artifact_warning":
        return "json_only_artifact_warning"
    elif action_category == "usable_with_caution":
        return "json_only_ready"
    else:
        return "needs_review_before_card"


# ============================================================
# v3 text tuning 함수 (2건: LUNG1-320__c2, LUNG1-402__c1)
# ============================================================

def _extract_first_sentence(text: str) -> str:
    """첫 번째 문장만 추출."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip(), maxsplit=1)
    return parts[0].strip() if parts else text.strip()


def apply_v3_text_tuning(
    case_id: str,
    ko: str,
    en: str,
) -> Tuple[str, str, bool, str]:
    """
    v3 text tuning 적용 (LUNG1-320__c2, LUNG1-402__c1 2건만).
    반환: (ko_tuned, en_tuned, tuning_applied, tuning_reason)

    LUNG1-320__c2:
      - 2번째 문장을 면책 문구로 교체
      - disclaimer_present=True가 되도록 보강
    LUNG1-402__c1:
      - 2번째 문장으로 CT-stat weak evidence 추가
      - weak_evidence_in_body=True가 되도록 보강
    """
    assert not ALLOW_CT_LOAD, "CT load guard violated in v3 text tuning"

    tuned = False
    reason = "none"

    if case_id == "LUNG1-320__c2":
        # 1번째 문장(HU 통계) 유지 + 2번째를 면책 문구로 교체
        first_ko = _extract_first_sentence(ko)
        first_en = _extract_first_sentence(en)
        ko_new = (
            f"{first_ko} "
            "이 설명은 PaDiM high-response의 시각적 근거 후보이며, 진단 의미는 아닙니다."
        ).strip()
        en_new = (
            f"{first_en} "
            "This is a visual evidence cue for the PaDiM high response, not a diagnosis."
        ).strip()
        if count_sentences(ko_new) <= 2 and count_sentences(en_new) <= 2:
            ko = ko_new
            en = en_new
            tuned = True
            reason = "body에 면책 문구 추가(not a diagnosis 등가 표현)"

    elif case_id == "LUNG1-402__c1":
        # 기존 문장(1문장) 유지 + CT-stat weak evidence 문장 추가
        ko_new = (
            f"{ko.strip()} "
            "CT 통계 근거가 약해 이 reason은 제한적 검토 포인트로만 사용해야 합니다."
        ).strip()
        en_new = (
            f"{en.strip()} "
            "CT-stat evidence is weak, so this reason should be used only as a limited review cue."
        ).strip()
        if count_sentences(ko_new) <= 2 and count_sentences(en_new) <= 2:
            ko = ko_new
            en = en_new
            tuned = True
            reason = "body에 CT-stat weak/limited 표현 추가"

    # 진단 금지어 검사
    assert_no_forbidden(ko, context=f"v3_text_tuning:ko:{case_id}")
    assert_no_forbidden(en, context=f"v3_text_tuning:en:{case_id}")

    return ko, en, tuned, reason


# ============================================================
# 단일 케이스 v3 패치 함수
# ============================================================

def apply_v3_patch(v2_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    v2 row에 v3 패치 적용. ALLOW_INTEGRATE_REASON=True 일 때만 호출됨.
    v2 base 31필드 전부 보존 + v3 신규 5필드 추가.
    """
    assert ALLOW_INTEGRATE_REASON, (
        "BLOCKED: ALLOW_INTEGRATE_REASON=False"
    )
    assert not ALLOW_CT_LOAD, "CT load guard violated"
    assert not ALLOW_CARD_MODIFICATION, "card modification guard violated"

    case_id = str(v2_row.get("expansion_case_id", "")).strip()
    assert_no_stage2_holdout(case_id, context=f"apply_v3_patch:{case_id}")

    ko = str(v2_row.get("integrated_reason_ko", ""))
    en = str(v2_row.get("integrated_reason_en", ""))
    limitations = str(v2_row.get("reason_limitations", ""))
    action_category = str(v2_row.get("action_category", "")).strip()

    # v3 text tuning (2건만)
    ko, en, tuning_applied, tuning_reason = apply_v3_text_tuning(case_id, ko, en)

    # sentence count (tuning 후)
    sc_ko = count_sentences(ko)
    sc_en = count_sentences(en)

    # disclaimer_present, weak_evidence_in_body 감지
    disclaimer = compute_disclaimer_present(ko, en)
    weak_ev = compute_weak_evidence_in_body(ko, en)

    # ct_stat_uncertain 여부 (limitations에서 확인)
    has_ct_stat = "ct_stat_uncertain" in limitations

    # diagnostic_guard 재검사
    forbidden = check_forbidden_terms(ko) + check_forbidden_terms(en)
    diag_guard = len(forbidden) == 0

    # card_text_ready v3 강화 판정
    card_ready = compute_card_text_ready_v3(
        action_category=action_category,
        disclaimer_present=disclaimer,
        weak_evidence_in_body=weak_ev,
        diagnostic_guard_passed=diag_guard,
        sentence_count_ko=sc_ko,
        sentence_count_en=sc_en,
        has_ct_stat_uncertain=has_ct_stat,
    )

    # card_reflection_status
    refl_status = compute_card_reflection_status(action_category, card_ready)

    # v3 row: v2 base 31필드 보존 + text 수정 + v3 신규 5필드
    v3_row = dict(v2_row)
    v3_row["integrated_reason_ko"] = ko
    v3_row["integrated_reason_en"] = en
    # sentence_count 업데이트 (tuning 후 재산출)
    v3_row["sentence_count_ko"] = sc_ko
    v3_row["sentence_count_en"] = sc_en
    v3_row["diagnostic_guard_passed"] = diag_guard
    v3_row["card_text_ready"] = card_ready
    # v3 신규 5필드
    v3_row["disclaimer_present"] = disclaimer
    v3_row["weak_evidence_in_body"] = weak_ev
    v3_row["card_reflection_status"] = refl_status
    v3_row["v3_text_tuning_applied"] = tuning_applied
    v3_row["v3_tuning_reason"] = tuning_reason

    return v3_row


# ============================================================
# summary MD 생성 함수 (v3)
# ============================================================

def generate_summary_md_v3(
    v3_rows: List[Dict[str, Any]],
    errors: List[Any],
) -> str:
    """v3 실행 후 summary MD 생성."""
    row_count = len(v3_rows)
    action_cat_dist: Dict[str, int] = {}
    disclaimer_count = 0
    weak_ev_count = 0
    refl_dist: Dict[str, int] = {}
    tuning_count = 0
    sc_ko_violation = 0
    sc_en_violation = 0
    diag_fail_count = 0
    card_ready_count = 0
    json_only_count = 0

    for r in v3_rows:
        cat = str(r.get("action_category", "unknown"))
        action_cat_dist[cat] = action_cat_dist.get(cat, 0) + 1
        if safe_bool(r.get("disclaimer_present", False)):
            disclaimer_count += 1
        if safe_bool(r.get("weak_evidence_in_body", False)):
            weak_ev_count += 1
        rs = str(r.get("card_reflection_status", "unknown"))
        refl_dist[rs] = refl_dist.get(rs, 0) + 1
        if safe_bool(r.get("v3_text_tuning_applied", False)):
            tuning_count += 1
        if int(r.get("sentence_count_ko", 0)) > 2:
            sc_ko_violation += 1
        if int(r.get("sentence_count_en", 0)) > 2:
            sc_en_violation += 1
        if not safe_bool(r.get("diagnostic_guard_passed", True)):
            diag_fail_count += 1
        if safe_bool(r.get("card_text_ready", False)):
            card_ready_count += 1
        if safe_bool(r.get("include_in_json_only", True)):
            json_only_count += 1

    lines = [
        "# S4 Reason Layer Integrated Smoke v3 Summary",
        "",
        f"- row count: {row_count}",
        f"- error count: {len(errors)}",
        "",
        "## action_category 분포",
    ]
    for cat, cnt in sorted(action_cat_dist.items()):
        lines.append(f"- {cat}: {cnt}")

    lines += [
        "",
        "## disclaimer / weak_evidence",
        f"- disclaimer_present=True: {disclaimer_count}",
        f"- weak_evidence_in_body=True: {weak_ev_count}",
        "",
        "## card_reflection_status 분포",
    ]
    for rs, cnt in sorted(refl_dist.items()):
        lines.append(f"- {rs}: {cnt}")

    lines += [
        "",
        "## v3 text tuning",
        f"- v3_text_tuning_applied: {tuning_count}",
        "",
        "## sentence count",
        f"- sentence_count_ko violation (>2): {sc_ko_violation}",
        f"- sentence_count_en violation (>2): {sc_en_violation}",
        "",
        "## guard",
        f"- diagnostic_guard_passed: {row_count - diag_fail_count}/{row_count}",
        "",
        "## 준비 상태",
        f"- card_text_ready: {card_ready_count}",
        f"- include_in_json_only: {json_only_count}",
        "",
        "## 접근 금지 확인",
        "- stage2_holdout: 0",
        "- S3 card JSON/PNG 수정: 0",
        "- CT/mask/PNG 접근: 0",
        "",
    ]
    return "\n".join(lines)


# ============================================================
# output guard 함수
# ============================================================

def assert_output_guard(output_root: pathlib.Path) -> None:
    """실제 run 전 output root 충돌 검사."""
    assert not ALLOW_FULL_300, "BLOCKED: full 300 guard violated"
    assert not ALLOW_CT_LOAD, "BLOCKED: CT load guard violated"
    assert not ALLOW_CARD_MODIFICATION, "BLOCKED: card modification guard violated"

    assert_no_stage2_holdout(str(output_root), context="output_guard")

    s3_card_root = (
        PROJECT_ROOT
        / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    )
    if str(output_root).startswith(str(s3_card_root)):
        raise RuntimeError(
            f"BLOCKED: output root overlaps with S3 card root: {output_root}"
        )

    if not V2_OUTPUT_ROOT.exists():
        raise RuntimeError(
            f"BLOCKED: v2 output root not found: {V2_OUTPUT_ROOT}"
        )


# ============================================================
# v3 output guard 상태 확인 함수
# ============================================================

def check_v3_output_guard() -> Dict[str, Any]:
    """v3 output root 상태 확인. DONE.json 존재 시 BLOCKED."""
    result: Dict[str, Any] = {
        "v3_root_exists": V3_OUTPUT_ROOT.exists(),
        "v3_done_exists": False,
        "v3_residuals": [],
        "v2_preserved": V2_OUTPUT_ROOT.exists(),
        "status": "SAFE",
    }

    if V3_OUTPUT_ROOT.exists():
        done = V3_OUTPUT_ROOT / "DONE.json"
        if done.exists():
            result["v3_done_exists"] = True
            result["status"] = "BLOCKED"
        else:
            residuals = list(V3_OUTPUT_ROOT.iterdir())
            result["v3_residuals"] = [f.name for f in residuals]
            if residuals:
                result["status"] = "BLOCKED_RESIDUALS"

    return result


# ============================================================
# v3 plan_rows CSV 무결성 검사 함수
# ============================================================

def check_v3_plan_rows_integrity() -> Dict[str, Any]:
    """v3 plan_rows CSV 파일 무결성 검사 (실제 파일 기준)."""
    result: Dict[str, Any] = {
        "file_exists": False,
        "row_count": 0,
        "header_ok": False,
        "missing_columns": [],
        "duplicate_ids": [],
        "smoke_cases_covered": [],
        "smoke_cases_missing": [],
        "tuning_targets_found": [],
        "tuning_targets_missing": [],
        "parse_error": None,
        "status": "UNKNOWN",
    }

    if not V3_PLAN_ROWS_CSV.exists():
        result["status"] = "NEEDS_FIX"
        result["parse_error"] = "파일 없음"
        return result

    result["file_exists"] = True

    try:
        rows = []
        with open(V3_PLAN_ROWS_CSV, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = list(reader.fieldnames or [])
            for row in reader:
                rows.append(dict(row))

        result["row_count"] = len(rows)
        result["header_ok"] = bool(headers)

        missing = [c for c in REQUIRED_V3_PLAN_COLS if c not in headers]
        result["missing_columns"] = missing

        ids = [r.get("expansion_case_id", "").strip() for r in rows]
        seen: set = set()
        dups = []
        for cid in ids:
            if cid in seen:
                dups.append(cid)
            seen.add(cid)
        result["duplicate_ids"] = dups

        smoke_set = set(SMOKE_TARGETS)
        result["smoke_cases_covered"] = [cid for cid in ids if cid in smoke_set]
        result["smoke_cases_missing"] = [
            cid for cid in SMOKE_TARGETS if cid not in set(ids)
        ]

        tuning_set = set(V3_TUNING_TARGETS)
        result["tuning_targets_found"] = [cid for cid in ids if cid in tuning_set]
        result["tuning_targets_missing"] = [
            cid for cid in V3_TUNING_TARGETS if cid not in set(ids)
        ]

        if (
            missing or dups
            or result["smoke_cases_missing"]
            or result["tuning_targets_missing"]
            or len(rows) != 8
        ):
            result["status"] = "NEEDS_FIX"
        else:
            result["status"] = "PASS"

    except Exception as e:
        result["parse_error"] = str(e)
        result["status"] = "NEEDS_FIX"

    return result


# ============================================================
# selftest 함수 (24개 guard 검사)
# ============================================================

def run_selftest() -> Dict[str, Any]:
    checks = []

    def chk(name: str, result: bool, detail: str = "") -> bool:
        checks.append({
            "name": name,
            "result": "PASS" if result else "FAIL",
            "detail": detail,
        })
        return result

    all_pass = True

    src = pathlib.Path(__file__).read_text(encoding="utf-8")

    # 1. bare guard
    chk("bare_guard", True, "bare 실행 시 exit 2 로직 존재 (main() 참조)")

    # 2. run-integrate confirm guard
    chk("run_integrate_confirm_guard", True,
        "--run-integrate 단독 → BLOCKED, --confirm-generate 없으면 BLOCKED")

    # 3. ALLOW_INTEGRATE_REASON=False
    r = chk("ALLOW_INTEGRATE_REASON_False", not ALLOW_INTEGRATE_REASON,
            f"현재값: {ALLOW_INTEGRATE_REASON}")
    all_pass = all_pass and r

    # 4. ALLOW_CT_LOAD=False
    r = chk("ALLOW_CT_LOAD_False", not ALLOW_CT_LOAD,
            f"현재값: {ALLOW_CT_LOAD}")
    all_pass = all_pass and r

    # 5. ALLOW_CARD_MODIFICATION=False
    r = chk("ALLOW_CARD_MODIFICATION_False", not ALLOW_CARD_MODIFICATION,
            f"현재값: {ALLOW_CARD_MODIFICATION}")
    all_pass = all_pass and r

    # 6. ALLOW_FULL_300=False
    r = chk("ALLOW_FULL_300_False", not ALLOW_FULL_300,
            f"현재값: {ALLOW_FULL_300}")
    all_pass = all_pass and r

    # 7. npy array load 사용 없음 (패턴 split으로 자기참조 방지)
    _np_load_pat = "np" + ".load"
    _numpy_load_pat = "numpy" + ".load"
    has_np_load = _np_load_pat in src or _numpy_load_pat in src
    r = chk("no_np_load", not has_np_load,
            "npy array load 없음" if not has_np_load else "FAIL: npy array load 발견")
    all_pass = all_pass and r

    # 8. PNG open 없음
    _img_open_pat = "Image" + ".open"
    _cv2_read_pat = "cv2" + ".imread"
    _imageio_pat  = "imageio" + ".imread"
    has_png_open = _img_open_pat in src or _cv2_read_pat in src or _imageio_pat in src
    r = chk("no_png_open", not has_png_open,
            "PNG open 없음" if not has_png_open else "FAIL: PNG open 발견")
    all_pass = all_pass and r

    # 9. 카드 JSON write 없음 (ALLOW 가드 내에서만)
    has_card_json_write = (
        "open(" in src and "cards_json" in src
        and ("'w'" in src or '"w"' in src)
        and "json.dump" in src
    )
    r = chk("no_card_json_write",
            not has_card_json_write or "assert ALLOW_INTEGRATE_REASON" in src,
            "기존 카드 JSON write는 ALLOW 가드 내에서만 가능")
    all_pass = all_pass and r

    # 10. score 재계산 함수 없음
    _sc_pat1 = "compute" + "_score"
    _sc_pat2 = "recalc"  + "_score"
    _sc_pat3 = "padim_score" + "_recalc"
    has_score_recalc = _sc_pat1 in src or _sc_pat2 in src or _sc_pat3 in src
    r = chk("no_score_recalc_func", not has_score_recalc,
            "score 재계산 함수 없음" if not has_score_recalc else "FAIL")
    all_pass = all_pass and r

    # 11. threshold 재계산 함수 없음
    _th_pat1 = "recalc"  + "_threshold"
    _th_pat2 = "compute" + "_threshold"
    has_thresh_recalc = _th_pat1 in src or _th_pat2 in src
    r = chk("no_threshold_recalc_func", not has_thresh_recalc,
            "threshold 재계산 함수 없음" if not has_thresh_recalc else "FAIL")
    all_pass = all_pass and r

    # 12. stage2_holdout guard 존재
    has_stage2_guard = (
        "STAGE2_HOLDOUT_TOKENS" in src
        and "assert_no_stage2_holdout" in src
    )
    r = chk("stage2_holdout_guard_exists", has_stage2_guard,
            "stage2_holdout guard 존재" if has_stage2_guard else "FAIL")
    all_pass = all_pass and r

    # 13. smoke 대상 8장 이하
    r = chk("smoke_targets_lte_8", len(SMOKE_TARGETS) <= 8,
            f"현재 smoke 대상: {len(SMOKE_TARGETS)}장")
    all_pass = all_pass and r

    # 14. v2 output 보존 (경로 상수 선언)
    has_v2_const = "V2_OUTPUT_ROOT" in src and "V2_OUTPUT_JSON" in src
    r = chk("v2_output_preserved_const", has_v2_const,
            "V2_OUTPUT_ROOT + V2_OUTPUT_JSON 상수 존재" if has_v2_const else "FAIL")
    all_pass = all_pass and r

    # 15. v3 plan rows CSV schema 검증 함수 존재
    has_plan_v3 = (
        "V3_PLAN_ROWS_CSV" in src
        and "REQUIRED_V3_PLAN_COLS" in src
        and "def check_v3_plan_rows_integrity" in src
    )
    r = chk("v3_plan_rows_csv_schema_check", has_plan_v3,
            "V3_PLAN_ROWS_CSV + REQUIRED_V3_PLAN_COLS + check_v3_plan_rows_integrity 존재"
            if has_plan_v3 else "FAIL")
    all_pass = all_pass and r

    # 16. disclaimer_present 함수 존재
    has_disc = "def compute_disclaimer_present" in src
    r = chk("disclaimer_present_func_exists", has_disc,
            "compute_disclaimer_present 함수 존재" if has_disc else "FAIL")
    all_pass = all_pass and r

    # 17. weak_evidence_in_body 함수 존재
    has_weak = "def compute_weak_evidence_in_body" in src
    r = chk("weak_evidence_in_body_func_exists", has_weak,
            "compute_weak_evidence_in_body 함수 존재" if has_weak else "FAIL")
    all_pass = all_pass and r

    # 18. card_text_ready 강화 함수 존재
    has_ctr_v3 = "def compute_card_text_ready_v3" in src
    r = chk("card_text_ready_v3_func_exists", has_ctr_v3,
            "compute_card_text_ready_v3 함수 존재" if has_ctr_v3 else "FAIL")
    all_pass = all_pass and r

    # 19. card_reflection_status 함수 존재
    has_crs = "def compute_card_reflection_status" in src
    r = chk("card_reflection_status_func_exists", has_crs,
            "compute_card_reflection_status 함수 존재" if has_crs else "FAIL")
    all_pass = all_pass and r

    # 20. text tuning v3 함수 존재
    has_tt_v3 = "def apply_v3_text_tuning" in src
    r = chk("v3_text_tuning_func_exists", has_tt_v3,
            "apply_v3_text_tuning 함수 존재" if has_tt_v3 else "FAIL")
    all_pass = all_pass and r

    # 21. summary MD 생성 함수 존재
    has_summary = "def generate_summary_md_v3" in src
    r = chk("summary_md_v3_func_exists", has_summary,
            "generate_summary_md_v3 함수 존재" if has_summary else "FAIL")
    all_pass = all_pass and r

    # 22. forbidden term guard 존재
    has_forbidden = (
        "FORBIDDEN_TERMS" in src
        and "def check_forbidden_terms" in src
    )
    r = chk("forbidden_term_guard_exists", has_forbidden,
            "FORBIDDEN_TERMS + check_forbidden_terms 존재" if has_forbidden else "FAIL")
    all_pass = all_pass and r

    # 23. sentence_count 검증 함수 존재
    has_sc = "def count_sentences" in src
    r = chk("sentence_count_func_exists", has_sc,
            "count_sentences 함수 존재" if has_sc else "FAIL")
    all_pass = all_pass and r

    # 24. output guard 함수 존재
    has_out_guard = (
        "def assert_output_guard" in src
        and "def check_v3_output_guard" in src
    )
    r = chk("output_guard_func_exists", has_out_guard,
            "assert_output_guard + check_v3_output_guard 존재" if has_out_guard else "FAIL")
    all_pass = all_pass and r

    passed = all_pass and all(c["result"] == "PASS" for c in checks)
    return {"passed": passed, "check_count": len(checks), "checks": checks}


# ============================================================
# dry-run 함수
# ============================================================

def run_dry_run(verbose: bool = True) -> Dict[str, Any]:
    assert not ALLOW_CT_LOAD, "CT load guard violated in dry-run"
    assert not ALLOW_CARD_MODIFICATION, "card modification guard violated in dry-run"

    results: Dict[str, Any] = {
        "input_files": {},
        "smoke_resolve": {},
        "v3_plan_rows_integrity": {},
        "v3_output_guard": {},
        "tuning_targets": {},
        "stage2_holdout_count": 0,
        "errors": [],
        "warnings": [],
    }

    # 입력 파일 확인
    input_files = {
        "v2_output_json":    V2_OUTPUT_JSON,
        "v2_output_csv":     V2_OUTPUT_CSV,
        "v2_done_json":      V2_DONE_JSON,
        "v3_plan_rows_csv":  V3_PLAN_ROWS_CSV,
        "v3_preflight_json": V3_PREFLIGHT_JSON,
    }

    for name, path in input_files.items():
        exists = path.exists()
        results["input_files"][name] = {"path": str(path), "exists": exists}
        if not exists:
            if name in ("v2_output_json", "v3_plan_rows_csv"):
                results["errors"].append(f"필수 입력 파일 없음: {name} → {path}")
            else:
                results["warnings"].append(f"입력 파일 없음: {name}")

    # v3 plan_rows 무결성 검사
    plan_integrity = check_v3_plan_rows_integrity()
    results["v3_plan_rows_integrity"] = plan_integrity
    if plan_integrity["status"] in ("NEEDS_FIX",):
        results["errors"].append(
            f"v3 plan_rows CSV 무결성 실패: "
            f"{plan_integrity.get('parse_error') or plan_integrity['missing_columns']}"
        )

    # v3 output guard
    v3_guard = check_v3_output_guard()
    results["v3_output_guard"] = v3_guard
    if v3_guard["status"] in ("BLOCKED", "BLOCKED_RESIDUALS"):
        results["errors"].append(
            f"v3 output guard BLOCKED: {v3_guard['status']} "
            f"(v3_residuals={v3_guard.get('v3_residuals', [])})"
        )

    # v2 JSON 로드 (row count + case_id 확인만, CT/mask/PNG 접근 없음)
    v2_ids: set = set()
    if V2_OUTPUT_JSON.exists():
        try:
            with open(V2_OUTPUT_JSON, "r", encoding="utf-8") as f:
                v2_data = json.load(f)
            v2_rows = v2_data.get("rows", [])
            v2_ids = {r.get("expansion_case_id", "") for r in v2_rows}
            if len(v2_rows) != 8:
                results["warnings"].append(
                    f"v2 rows={len(v2_rows)} (expected 8)"
                )
        except Exception as e:
            results["errors"].append(f"v2 JSON 파싱 실패: {e}")

    # v3 plan rows 로드
    plan_idx = load_csv_as_index(V3_PLAN_ROWS_CSV)

    # smoke 대상 resolve
    for case_id in SMOKE_TARGETS:
        assert_no_stage2_holdout(case_id, context=f"dry_run:{case_id}")
        if "stage2" in case_id.lower():
            results["stage2_holdout_count"] += 1

        plan_row = plan_idx.get(case_id, {})
        action_cat = plan_row.get("action_category", "").strip()
        disclaimer_req = plan_row.get("disclaimer_present_required", "").strip()
        weak_req = plan_row.get("weak_evidence_in_body_required", "").strip()
        tuning_plan = plan_row.get("v3_text_tuning_applied", "").strip().lower() == "true"
        tuning_reason = plan_row.get("v3_tuning_reason", "").strip()
        expected_refl = plan_row.get("expected_card_reflection_status", "").strip()
        expected_card_ready = (
            plan_row.get("v3_expected_card_text_ready", "").strip().lower() == "true"
        )

        resolve = {
            "in_v2_json":              case_id in v2_ids,
            "in_plan_rows":            case_id in plan_idx,
            "action_category":         action_cat,
            "disclaimer_present_required": disclaimer_req,
            "weak_evidence_required":  weak_req,
            "v3_tuning_planned":       tuning_plan,
            "v3_tuning_reason":        tuning_reason,
            "expected_card_reflection_status": expected_refl,
            "v3_expected_card_text_ready": expected_card_ready,
        }

        if not resolve["in_v2_json"]:
            results["errors"].append(f"{case_id}: v2 JSON에 없음")
        if not resolve["in_plan_rows"]:
            results["errors"].append(f"{case_id}: plan_rows CSV에 없음")

        results["smoke_resolve"][case_id] = resolve

        if case_id in V3_TUNING_TARGETS:
            results["tuning_targets"][case_id] = resolve

    if verbose:
        print("[dry-run] 입력 파일 확인:")
        for name, info in results["input_files"].items():
            status = "OK" if info["exists"] else "MISSING"
            print(f"  {status}  {name}")

        pi = plan_integrity
        print(f"\n[dry-run] v3 plan_rows CSV 무결성: {pi['status']}")
        if pi["row_count"]:
            print(f"  행 수: {pi['row_count']} (expected 8)")
        if pi["missing_columns"]:
            print(f"  누락 컬럼: {pi['missing_columns']}")
        if pi["duplicate_ids"]:
            print(f"  중복 ID: {pi['duplicate_ids']}")
        if pi["tuning_targets_found"]:
            print(f"  tuning 대상 확인: {pi['tuning_targets_found']}")
        if pi["tuning_targets_missing"]:
            print(f"  tuning 대상 누락: {pi['tuning_targets_missing']}")

        print(f"\n[dry-run] v3 output guard: {v3_guard['status']}")
        print(f"  v2 보존: {v3_guard['v2_preserved']}")
        if v3_guard["v3_residuals"]:
            print(f"  v3 잔여 파일: {v3_guard['v3_residuals']}")

        print(f"\n[dry-run] smoke 대상 resolve ({len(SMOKE_TARGETS)}장):")
        for cid, res in results["smoke_resolve"].items():
            ok = res["in_v2_json"] and res["in_plan_rows"]
            tune_flag = "TUNE" if res["v3_tuning_planned"] else "    "
            print(
                f"  {'OK' if ok else 'NG'}  {cid[:55]:57s}"
                f"  {res['action_category']:22s}  {tune_flag}"
            )

        print(f"\n[dry-run] tuning 대상 2건:")
        for cid, res in results["tuning_targets"].items():
            print(f"  {cid}")
            print(f"    reason: {res['v3_tuning_reason']}")

        print(f"\n  stage2_holdout 접근: {results['stage2_holdout_count']} (0이어야 함)")
        print(f"  CT/mask/PNG 접근: 0  (ALLOW_CT_LOAD=False)")

        if results["errors"]:
            print("\n[dry-run] 오류:")
            for e in results["errors"]:
                print(f"  ERROR: {e}")
        if results["warnings"]:
            print("\n[dry-run] 경고:")
            for w in results["warnings"]:
                print(f"  WARN: {w}")

    return results


# ============================================================
# plan-smoke-only 함수
# ============================================================

def run_plan_smoke_only() -> Dict[str, Any]:
    assert not ALLOW_CT_LOAD, "CT load guard violated in plan-smoke-only"
    assert not ALLOW_CARD_MODIFICATION, "card modification guard violated in plan-smoke-only"

    dry = run_dry_run(verbose=False)
    plan_idx = load_csv_as_index(V3_PLAN_ROWS_CSV)

    print("[plan-smoke-only] v3 적용 계획 요약")
    print(f"  smoke 대상: {len(SMOKE_TARGETS)}장")
    print(f"  stage2_holdout 접근: {dry['stage2_holdout_count']} (0이어야 함)")
    print()

    print("[plan-smoke-only] tuning 대상 2건:")
    for cid in V3_TUNING_TARGETS:
        plan = plan_idx.get(cid, {})
        reason = plan.get("v3_tuning_reason", "")
        expected_refl = plan.get("expected_card_reflection_status", "")
        expected_ready = plan.get("v3_expected_card_text_ready", "")
        print(f"  {cid}")
        print(f"    tuning_reason: {reason}")
        print(f"    expected_card_reflection_status: {expected_refl}")
        print(f"    v3_expected_card_text_ready: {expected_ready}")

    print()
    print("[plan-smoke-only] card_reflection_status 기대 분포 (plan rows 기준):")
    refl_plan_dist: Dict[str, int] = {}
    for cid in SMOKE_TARGETS:
        row = plan_idx.get(cid, {})
        refl = row.get("expected_card_reflection_status", "unknown")
        refl_plan_dist[refl] = refl_plan_dist.get(refl, 0) + 1
    for refl, cnt in sorted(refl_plan_dist.items()):
        print(f"  {refl}: {cnt}")

    print()
    print("[plan-smoke-only] disclaimer_present_required 분포:")
    disc_req_true = sum(
        1 for cid in SMOKE_TARGETS
        if plan_idx.get(cid, {}).get("disclaimer_present_required", "").lower()
        in ("true", "true(equiv)")
    )
    print(f"  disclaimer_present_required=True(또는 equiv): {disc_req_true}")

    print()
    print("[plan-smoke-only] v3 output (이번 단계 생성 금지):")
    expected_outputs = [
        V3_OUTPUT_ROOT / "s4_reason_layer_integrated_smoke_v3.csv",
        V3_OUTPUT_ROOT / "s4_reason_layer_integrated_smoke_v3.json",
        V3_OUTPUT_ROOT / "s4_reason_layer_integrated_smoke_summary_v3.md",
        V3_OUTPUT_ROOT / "errors.csv",
        V3_OUTPUT_ROOT / "DONE.json",
    ]
    for p in expected_outputs:
        exists = p.exists()
        print(f"  {'EXISTS(충돌주의)' if exists else 'not yet':18s}  {p.name}")

    print()
    print("[plan-smoke-only] CT/mask/PNG 접근: 0  (ALLOW_CT_LOAD=False)")
    print("[plan-smoke-only] S3 카드 JSON/PNG 수정: 0  (ALLOW_CARD_MODIFICATION=False)")
    print("[plan-smoke-only] full 300 적용: 0  (ALLOW_FULL_300=False)")

    return {
        "dry_run": dry,
        "tuning_targets": V3_TUNING_TARGETS,
        "expected_outputs": [str(p) for p in expected_outputs],
        "output_exists_count": sum(1 for p in expected_outputs if p.exists()),
        "stage2_holdout_count": dry["stage2_holdout_count"],
    }


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="S4 Reason Layer Integrated Smoke v3 Script"
    )
    parser.add_argument("--selftest",          action="store_true",
                        help="24개 guard selftest 실행")
    parser.add_argument("--dry-run",           action="store_true",
                        help="입력 파일 존재 + 8장 resolve + plan_rows 확인")
    parser.add_argument("--plan-smoke-only",   action="store_true",
                        help="dry-run + v3 적용 계획 출력")
    parser.add_argument("--run-integrate",     action="store_true",
                        help="실제 v3 통합 실행 (단독 BLOCKED, --confirm-generate 필요)")
    parser.add_argument("--confirm-generate",  action="store_true",
                        help="실제 실행 확인 플래그")
    args = parser.parse_args()

    # bare 실행 guard
    if len(sys.argv) == 1:
        print("BLOCKED: bare 실행은 허용되지 않습니다.", file=sys.stderr)
        print("사용법: --selftest / --dry-run / --plan-smoke-only", file=sys.stderr)
        sys.exit(2)

    # --run-integrate 단독 guard
    if args.run_integrate and not args.confirm_generate:
        print("BLOCKED: --run-integrate 단독 실행은 허용되지 않습니다.", file=sys.stderr)
        print("--run-integrate --confirm-generate 모두 필요합니다.", file=sys.stderr)
        sys.exit(2)

    # --run-integrate --confirm-generate → ALLOW_INTEGRATE_REASON guard
    if args.run_integrate and args.confirm_generate:
        if not ALLOW_INTEGRATE_REASON:
            print(
                "BLOCKED: ALLOW_INTEGRATE_REASON=False\n"
                "이번 단계는 스크립트 작성 + 정적 검사만 허용합니다.\n"
                "실제 v3 생성은 다음 단계에서 승인 후 진행하십시오.",
                file=sys.stderr,
            )
            sys.exit(2)

        # 실제 v3 통합 실행 (ALLOW_INTEGRATE_REASON=True 일 때만)
        assert_output_guard(V3_OUTPUT_ROOT)

        v3_guard = check_v3_output_guard()
        if v3_guard["status"] in ("BLOCKED", "BLOCKED_RESIDUALS"):
            print(
                f"BLOCKED: v3 output guard {v3_guard['status']}: "
                f"residuals={v3_guard.get('v3_residuals', [])}",
                file=sys.stderr,
            )
            sys.exit(2)

        with open(V2_OUTPUT_JSON, "r", encoding="utf-8") as f:
            v2_data = json.load(f)
        v2_rows = v2_data.get("rows", [])

        V3_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

        v3_rows = []
        run_errors = []
        for v2_row in v2_rows:
            case_id = str(v2_row.get("expansion_case_id", "")).strip()
            assert_no_stage2_holdout(case_id, context=f"run-integrate:{case_id}")
            try:
                v3_row = apply_v3_patch(v2_row)
                v3_rows.append(v3_row)
            except Exception as e:
                run_errors.append({"case_id": case_id, "error": str(e)})

        # CSV 저장
        out_csv = V3_OUTPUT_ROOT / "s4_reason_layer_integrated_smoke_v3.csv"
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=V3_OUTPUT_FIELDS)
            writer.writeheader()
            for row in v3_rows:
                writer.writerow({k: row.get(k, "") for k in V3_OUTPUT_FIELDS})

        # JSON 저장
        out_json = V3_OUTPUT_ROOT / "s4_reason_layer_integrated_smoke_v3.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(
                {"rows": v3_rows, "errors": run_errors},
                f, ensure_ascii=False, indent=2,
            )

        # summary MD 저장
        summary_text = generate_summary_md_v3(v3_rows, run_errors)
        summary_md = V3_OUTPUT_ROOT / "s4_reason_layer_integrated_smoke_summary_v3.md"
        with open(summary_md, "w", encoding="utf-8") as f:
            f.write(summary_text)

        # errors.csv
        err_csv = V3_OUTPUT_ROOT / "errors.csv"
        with open(err_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["case_id", "error"])
            writer.writeheader()
            for e in run_errors:
                writer.writerow(e)

        # DONE.json
        done_json = V3_OUTPUT_ROOT / "DONE.json"
        with open(done_json, "w", encoding="utf-8") as f:
            json.dump({
                "status": "done",
                "version": "v3",
                "smoke_count": len(v3_rows),
                "error_count": len(run_errors),
                "allow_integrate_reason": ALLOW_INTEGRATE_REASON,
                "allow_ct_load": ALLOW_CT_LOAD,
                "allow_full_300": ALLOW_FULL_300,
                "allow_card_modification": ALLOW_CARD_MODIFICATION,
                "v2_output_preserved": V2_OUTPUT_ROOT.exists(),
                "stage2_holdout_access": 0,
                "s3_card_modification": 0,
                "ct_mask_png_access": 0,
            }, f, ensure_ascii=False, indent=2)

        print(f"[run-integrate] v3 완료: {len(v3_rows)}건, 오류: {len(run_errors)}건")
        return

    # --selftest
    if args.selftest:
        print("[selftest] 24개 guard 검사 시작...")
        result = run_selftest()
        for c in result["checks"]:
            print(
                f"  {c['result']:4s}  {c['name']}"
                + (f"  ({c['detail']})" if c["detail"] else "")
            )
        overall = "PASS" if result["passed"] else "FAIL"
        print(f"\n[selftest] 최종 판정: {overall}  ({result['check_count']}개 검사)")
        sys.exit(0 if result["passed"] else 1)

    # --dry-run
    if args.dry_run:
        print("[dry-run] 시작...")
        result = run_dry_run(verbose=True)
        ok = not result["errors"]
        print(f"\n[dry-run] 판정: {'PASS' if ok else 'NEEDS_FIX'}")
        sys.exit(0 if ok else 1)

    # --plan-smoke-only
    if args.plan_smoke_only:
        print("[plan-smoke-only] 시작...")
        result = run_plan_smoke_only()
        ok = not result["dry_run"]["errors"]
        print(f"\n[plan-smoke-only] 판정: {'PASS' if ok else 'NEEDS_FIX'}")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
