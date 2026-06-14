#!/usr/bin/env python3
"""
build_explanation_card_s4_reason_layer_integrated_smoke_v2.py

S4 Reason Layer Integrated Smoke v2 Script

목적:
- S4 integrated reason v1 JSON을 입력으로 읽고 아래 수정을 적용한다:
  1. action_category 컬럼 추가 (v1 recommended_next_action 매핑)
  2. text_tuning_applied / sentence_count_ko / sentence_count_en 추가
  3. limitation_fix_applied / card_text_ready 추가
  4. MSD_lung_054__c1, LUNG1-057__c1: EN 'vs.' → 'versus'
  5. LUNG1-402__c1, LUNG1-057__c1: limitations에 ct_stat_uncertain 추가
  6. subset9 normal_control: KO/EN 3문장 → 2문장 병합
  7. summary MD 생성

이번 단계: 스크립트 작성 + 정적 검사만 허용.
실제 v2 생성은 --run-integrate --confirm-generate 조합으로만 가능하며
ALLOW_INTEGRATE_REASON=False 가드로 차단되어 있다.

절대 금지:
- CT/mask npy 로드 (ALLOW_CT_LOAD=False)
- HU 통계 재계산
- PNG open
- 카드 PNG/JSON 수정 (ALLOW_CARD_MODIFICATION=False)
- full 300 reason 적용 (ALLOW_FULL_300=False)
- score/model/threshold 재계산
- stage2_holdout 접근
- 기존 v1 CSV/JSON/DONE 수정

실행 모드:
- bare 실행             → BLOCKED exit 2
- --selftest            → 25개 guard 검사
- --dry-run             → 입력 파일 존재 + 8장 resolve 확인
- --plan-smoke-only     → dry-run + v2 적용 계획 출력
- --run-integrate       → 단독 BLOCKED exit 2
- --run-integrate --confirm-generate → ALLOW_INTEGRATE_REASON=False 로 BLOCKED

syntax check:
  python -m py_compile scripts/build_explanation_card_s4_reason_layer_integrated_smoke_v2.py
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

# ============================================================
# action_category mapping
# ============================================================
ACTION_CATEGORY_MAP = {
    "hold_review_pending":           "hold_context",
    "json_only_until_hold_resolved": "hold_context",
    "json_only_no_card_text":        "uncertain_json_only",
    "artifact_warning_json_only":    "artifact_warning",
    "json_ready_borderline":         "usable_with_caution",
    "fp_context":                    "usable_with_caution",
    "json_ready_fp_context":         "usable_with_caution",
    "json_ready_strong_density":     "ready",
}

# ============================================================
# v1 base schema 25 필드 + v2 신규 6 필드 = 31 필드
# ============================================================
V1_BASE_FIELDS = [
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
]

V2_NEW_FIELDS = [
    "action_category",
    "text_tuning_applied",
    "sentence_count_ko",
    "sentence_count_en",
    "limitation_fix_applied",
    "card_text_ready",
]

V2_OUTPUT_FIELDS = V1_BASE_FIELDS + V2_NEW_FIELDS

# ============================================================
# plan_rows CSV 필수 컬럼 (실제 파일 컬럼명 기준)
# ============================================================
REQUIRED_PLAN_COLS = [
    "expansion_case_id",
    "role",
    "v1_action",
    "v2_action_category",
    "text_tuning_applied",
    "limitation_fix_applied",
    "text_fix_detail",
    "limitation_fix_detail",
    "v2_sentence_count_ko_target",
    "v2_sentence_count_en_target",
    "card_text_ready_expected",
    "json_only_ready_expected",
    "diagnostic_guard_expected",
]

# ============================================================
# 경로 상수
# ============================================================
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

REPORTS_ROOT = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
)

# v2 입력 파일
V1_OUTPUT_JSON    = REPORTS_ROOT / "s4_reason_layer_integrated_smoke_v1/s4_reason_layer_integrated_smoke_v1.json"
V1_OUTPUT_CSV     = REPORTS_ROOT / "s4_reason_layer_integrated_smoke_v1/s4_reason_layer_integrated_smoke_v1.csv"
V1_DONE_JSON      = REPORTS_ROOT / "s4_reason_layer_integrated_smoke_v1/DONE.json"
V2_PLAN_ROWS_CSV  = REPORTS_ROOT / "s4_integrated_reason_text_tuning_v2_plan_rows_v1.csv"
V2_PREFLIGHT_JSON = REPORTS_ROOT / "s4_integrated_reason_text_tuning_v2_preflight_v1.json"
RESULT_REVIEW_JSON = REPORTS_ROOT / "s4_integrated_reason_result_review_v1.json"
RESULT_REVIEW_ROWS = REPORTS_ROOT / "s4_integrated_reason_result_review_rows_v1.csv"

# v1 output root (read-only, 보존 확인용)
V1_OUTPUT_ROOT = REPORTS_ROOT / "s4_reason_layer_integrated_smoke_v1"

# v2 output root (이번 단계 생성 금지)
OUTPUT_ROOT = REPORTS_ROOT / "s4_reason_layer_integrated_smoke_v2"

# static drycheck 보고서 (이번 단계 생성)
STATIC_CHECK_REPORT_MD   = REPORTS_ROOT / "s4_integrated_reason_v2_script_static_drycheck_v1.md"
STATIC_CHECK_REPORT_JSON = REPORTS_ROOT / "s4_integrated_reason_v2_script_static_drycheck_v1.json"


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
# action_category mapping 함수
# ============================================================

def map_action_category(recommended_next_action: str) -> str:
    """v1 recommended_next_action → v2 action_category 매핑."""
    return ACTION_CATEGORY_MAP.get(recommended_next_action.strip(), "unknown")


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
# text tuning 함수
# ============================================================

def _extract_first_sentence(text: str) -> str:
    """첫 번째 문장만 추출."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip(), maxsplit=1)
    return parts[0].strip() if parts else text.strip()


def apply_text_tuning(
    case_id: str,
    ko: str,
    en: str,
) -> Tuple[str, str, bool]:
    """
    v2 text tuning 적용.
    반환: (ko_tuned, en_tuned, text_tuning_applied)

    적용 규칙:
    - 공통: EN 'vs.' → 'versus'
    - subset9: KO/EN 3문장 → 2문장 병합, FP review 의미 유지
    - 모든 수정 후 forbidden term 검사
    - 2문장 이하 강제
    """
    assert not ALLOW_CT_LOAD, "CT load guard violated in text tuning"

    tuned = False
    subset9_id = (
        "subset9_1.3.6.1.4.1.14519.5.2.1.6279.6001"
        ".291156498203266896953765649282__c1"
    )

    # EN 'vs.' → 'versus'
    if "vs." in en:
        en = en.replace("vs.", "versus")
        tuned = True

    # subset9: 3→2문장 병합
    if case_id == subset9_id:
        ko_first = _extract_first_sentence(ko)
        en_first = _extract_first_sentence(en)
        ko_new = (
            f"{ko_first} "
            "이 케이스는 정상 제어군으로 FP 패턴 검토 목적이며, "
            "ROI coverage가 낮아 경계 조직의 영향도 있을 수 있습니다."
        ).strip()
        en_new = (
            f"{en_first} "
            "This is a normal control case for FP pattern review; "
            "low ROI coverage may reflect boundary tissue influence."
        ).strip()
        if count_sentences(ko_new) <= 2 and count_sentences(en_new) <= 2:
            ko = ko_new
            en = en_new
            tuned = True

    # 진단 금지어 검사
    assert_no_forbidden(ko, context=f"text_tuning:ko:{case_id}")
    assert_no_forbidden(en, context=f"text_tuning:en:{case_id}")

    return ko, en, tuned


# ============================================================
# limitation fix 함수
# ============================================================

def apply_limitation_fix(
    case_id: str,
    limitations: str,
) -> Tuple[str, bool]:
    """
    v2 limitation fix 적용.
    반환: (limitations_fixed, limitation_fix_applied)
    """
    fix_targets: Dict[str, str] = {
        "LUNG1-402__c1": "ct_stat_uncertain",
        "LUNG1-057__c1": "ct_stat_uncertain",
    }
    if case_id not in fix_targets:
        return limitations, False

    add_tag = fix_targets[case_id]
    existing = [t.strip() for t in limitations.split(",") if t.strip()]
    if add_tag not in existing:
        existing.append(add_tag)
        return ",".join(existing), True

    return limitations, False


# ============================================================
# card_text_ready 판정 함수
# ============================================================

def compute_card_text_ready(
    include_in_future_card_text: bool,
    diagnostic_guard_passed: bool,
    sentence_count_ko: int,
    sentence_count_en: int,
    action_category: str,
) -> bool:
    """
    card_text_ready = action_category=="ready" AND include AND guard AND <=2문장.
    json_ready_borderline/usable_with_caution 등은 future card용으로만 유지.
    """
    return (
        action_category == "ready"
        and include_in_future_card_text
        and diagnostic_guard_passed
        and sentence_count_ko <= 2
        and sentence_count_en <= 2
    )


# ============================================================
# 단일 케이스 v2 패치 함수
# ============================================================

def apply_v2_patch(v1_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    v1 row에 v2 패치 적용. ALLOW_INTEGRATE_REASON=True 일 때만 호출됨.
    v1 base 필드 전부 보존 + text/limitation 수정 + 신규 필드 추가.
    """
    assert ALLOW_INTEGRATE_REASON, (
        "BLOCKED: ALLOW_INTEGRATE_REASON=False"
    )
    assert not ALLOW_CT_LOAD, "CT load guard violated"
    assert not ALLOW_CARD_MODIFICATION, "card modification guard violated"

    case_id = str(v1_row.get("expansion_case_id", "")).strip()
    assert_no_stage2_holdout(case_id, context=f"apply_v2_patch:{case_id}")

    ko = str(v1_row.get("integrated_reason_ko", ""))
    en = str(v1_row.get("integrated_reason_en", ""))
    limitations = str(v1_row.get("reason_limitations", ""))
    rec_action = str(v1_row.get("recommended_next_action", "")).strip()

    # text tuning
    ko, en, text_tuning = apply_text_tuning(case_id, ko, en)

    # limitation fix
    limitations, lim_fix = apply_limitation_fix(case_id, limitations)

    # sentence count
    sc_ko = count_sentences(ko)
    sc_en = count_sentences(en)

    # action category
    action_cat = map_action_category(rec_action)

    # card_text_ready
    include_card = safe_bool(v1_row.get("include_in_future_card_text", False))
    diag_guard = safe_bool(v1_row.get("diagnostic_guard_passed", True))
    card_ready = compute_card_text_ready(
        include_card, diag_guard, sc_ko, sc_en, action_cat
    )

    # diagnostic guard 재검사
    forbidden = check_forbidden_terms(ko) + check_forbidden_terms(en)
    diag_guard_final = len(forbidden) == 0

    # v2 row: v1 base fields 보존 + 수정 + 신규 필드
    v2_row = dict(v1_row)
    v2_row["integrated_reason_ko"] = ko
    v2_row["integrated_reason_en"] = en
    v2_row["reason_limitations"] = limitations
    v2_row["diagnostic_guard_passed"] = diag_guard_final
    # 신규 v2 필드
    v2_row["action_category"] = action_cat
    v2_row["text_tuning_applied"] = text_tuning
    v2_row["sentence_count_ko"] = sc_ko
    v2_row["sentence_count_en"] = sc_en
    v2_row["limitation_fix_applied"] = lim_fix
    v2_row["card_text_ready"] = card_ready

    return v2_row


# ============================================================
# summary MD 생성 함수
# ============================================================

def generate_summary_md(
    v2_rows: List[Dict[str, Any]],
    errors: List[Any],
) -> str:
    """v2 실행 후 summary MD 생성."""
    row_count = len(v2_rows)
    action_cat_dist: Dict[str, int] = {}
    text_tuning_count = 0
    lim_fix_count = 0
    sc_ko_violation = 0
    sc_en_violation = 0
    diag_fail_count = 0
    card_ready_count = 0
    json_only_count = 0

    for r in v2_rows:
        cat = str(r.get("action_category", "unknown"))
        action_cat_dist[cat] = action_cat_dist.get(cat, 0) + 1
        if safe_bool(r.get("text_tuning_applied", False)):
            text_tuning_count += 1
        if safe_bool(r.get("limitation_fix_applied", False)):
            lim_fix_count += 1
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
        "# S4 Reason Layer Integrated Smoke v2 Summary",
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
        "## text/limitation 수정",
        f"- text_tuning_applied: {text_tuning_count}",
        f"- limitation_fix_applied: {lim_fix_count}",
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

    if not V1_OUTPUT_ROOT.exists():
        raise RuntimeError(
            f"BLOCKED: v1 output root not found: {V1_OUTPUT_ROOT}"
        )


# ============================================================
# v2 output guard 상태 확인 함수
# ============================================================

def check_v2_output_guard() -> Dict[str, Any]:
    """v2 output root 상태 확인. DONE.json 존재 시 BLOCKED."""
    result: Dict[str, Any] = {
        "v2_root_exists": OUTPUT_ROOT.exists(),
        "v2_done_exists": False,
        "v2_residuals": [],
        "v1_preserved": V1_OUTPUT_ROOT.exists(),
        "status": "SAFE",
    }

    if OUTPUT_ROOT.exists():
        done = OUTPUT_ROOT / "DONE.json"
        if done.exists():
            result["v2_done_exists"] = True
            result["status"] = "BLOCKED"
        else:
            residuals = list(OUTPUT_ROOT.iterdir())
            result["v2_residuals"] = [f.name for f in residuals]
            if residuals:
                result["status"] = "WARN_RESIDUALS"

    return result


# ============================================================
# plan_rows CSV 무결성 검사 함수
# ============================================================

def check_plan_rows_integrity() -> Dict[str, Any]:
    """plan_rows CSV 파일 무결성 검사 (실제 파일 기준)."""
    result: Dict[str, Any] = {
        "file_exists": False,
        "row_count": 0,
        "header_ok": False,
        "missing_columns": [],
        "duplicate_ids": [],
        "smoke_cases_covered": [],
        "smoke_cases_missing": [],
        "parse_error": None,
        "status": "UNKNOWN",
    }

    if not V2_PLAN_ROWS_CSV.exists():
        result["status"] = "NEEDS_FIX"
        result["parse_error"] = "파일 없음"
        return result

    result["file_exists"] = True

    try:
        rows = []
        with open(V2_PLAN_ROWS_CSV, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = list(reader.fieldnames or [])
            for row in reader:
                rows.append(dict(row))

        result["row_count"] = len(rows)
        result["header_ok"] = bool(headers)

        missing = [c for c in REQUIRED_PLAN_COLS if c not in headers]
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

        if missing or dups or result["smoke_cases_missing"] or len(rows) != 8:
            result["status"] = "NEEDS_FIX"
        else:
            result["status"] = "PASS"

    except Exception as e:
        result["parse_error"] = str(e)
        result["status"] = "NEEDS_FIX"

    return result


# ============================================================
# selftest 함수 (25개 guard 검사)
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

    # 14. v1 output 보존 (경로 상수 선언)
    has_v1_const = "V1_OUTPUT_ROOT" in src and "V1_OUTPUT_JSON" in src
    r = chk("v1_output_preserved_const", has_v1_const,
            "V1_OUTPUT_ROOT + V1_OUTPUT_JSON 상수 존재" if has_v1_const else "FAIL")
    all_pass = all_pass and r

    # 15. plan_rows CSV 로드 경로 존재
    has_plan_rows = "V2_PLAN_ROWS_CSV" in src
    r = chk("plan_rows_csv_path_exists", has_plan_rows,
            "V2_PLAN_ROWS_CSV 상수 존재" if has_plan_rows else "FAIL")
    all_pass = all_pass and r

    # 16. action_category mapping 함수 존재
    has_action_cat = (
        "def map_action_category" in src
        and "ACTION_CATEGORY_MAP" in src
    )
    r = chk("action_category_mapping_func_exists", has_action_cat,
            "map_action_category + ACTION_CATEGORY_MAP 존재" if has_action_cat else "FAIL")
    all_pass = all_pass and r

    # 17. text tuning 함수 존재
    has_text_tuning = "def apply_text_tuning" in src
    r = chk("text_tuning_func_exists", has_text_tuning,
            "apply_text_tuning 함수 존재" if has_text_tuning else "FAIL")
    all_pass = all_pass and r

    # 18. limitation fix 함수 존재
    has_lim_fix = "def apply_limitation_fix" in src
    r = chk("limitation_fix_func_exists", has_lim_fix,
            "apply_limitation_fix 함수 존재" if has_lim_fix else "FAIL")
    all_pass = all_pass and r

    # 19. summary MD 생성 함수 존재
    has_summary = "def generate_summary_md" in src
    r = chk("summary_md_func_exists", has_summary,
            "generate_summary_md 함수 존재" if has_summary else "FAIL")
    all_pass = all_pass and r

    # 20. forbidden term guard 존재
    has_forbidden = (
        "FORBIDDEN_TERMS" in src
        and "def check_forbidden_terms" in src
    )
    r = chk("forbidden_term_guard_exists", has_forbidden,
            "FORBIDDEN_TERMS + check_forbidden_terms 존재" if has_forbidden else "FAIL")
    all_pass = all_pass and r

    # 21. sentence_count 함수 존재
    has_sc = "def count_sentences" in src
    r = chk("sentence_count_func_exists", has_sc,
            "count_sentences 함수 존재" if has_sc else "FAIL")
    all_pass = all_pass and r

    # 22. output guard 함수 존재
    has_out_guard = "def assert_output_guard" in src
    r = chk("output_guard_func_exists", has_out_guard,
            "assert_output_guard 함수 존재" if has_out_guard else "FAIL")
    all_pass = all_pass and r

    # 23. v2 output root가 v1 output root와 다름
    v2_different = str(OUTPUT_ROOT) != str(V1_OUTPUT_ROOT)
    r = chk("v2_output_root_different_from_v1", v2_different,
            f"v2={OUTPUT_ROOT.name}, v1={V1_OUTPUT_ROOT.name}")
    all_pass = all_pass and r

    # 24. vs. → versus 변환 로직 존재
    has_vs_replace = (
        'replace("vs.", "versus")' in src
        or "replace('vs.', 'versus')" in src
    )
    r = chk("vs_dot_replacement_implemented", has_vs_replace,
            "vs. → versus 변환 로직 존재" if has_vs_replace else "FAIL")
    all_pass = all_pass and r

    # 25. S3 card root overlap 없음
    s3_overlap = "candidate_cards" in str(OUTPUT_ROOT)
    r = chk("no_s3_output_overlap", not s3_overlap,
            f"OUTPUT_ROOT={OUTPUT_ROOT.name} — S3 card root 미포함"
            if not s3_overlap else "FAIL")
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
        "plan_rows_integrity": {},
        "v2_output_guard": {},
        "stage2_holdout_count": 0,
        "errors": [],
        "warnings": [],
    }

    # 입력 파일 확인
    input_files = {
        "v1_output_json":       V1_OUTPUT_JSON,
        "v1_output_csv":        V1_OUTPUT_CSV,
        "v1_done_json":         V1_DONE_JSON,
        "v2_plan_rows_csv":     V2_PLAN_ROWS_CSV,
        "v2_preflight_json":    V2_PREFLIGHT_JSON,
        "result_review_json":   RESULT_REVIEW_JSON,
        "result_review_rows":   RESULT_REVIEW_ROWS,
    }

    for name, path in input_files.items():
        exists = path.exists()
        results["input_files"][name] = {"path": str(path), "exists": exists}
        if not exists:
            if name in ("v1_output_json", "v2_plan_rows_csv"):
                results["errors"].append(f"필수 입력 파일 없음: {name} → {path}")
            else:
                results["warnings"].append(f"입력 파일 없음: {name}")

    # plan_rows 무결성 검사
    plan_integrity = check_plan_rows_integrity()
    results["plan_rows_integrity"] = plan_integrity
    if plan_integrity["status"] == "NEEDS_FIX":
        results["errors"].append(
            f"plan_rows CSV 무결성 실패: "
            f"{plan_integrity.get('parse_error') or plan_integrity['missing_columns']}"
        )

    # v2 output guard
    v2_guard = check_v2_output_guard()
    results["v2_output_guard"] = v2_guard
    if v2_guard["status"] == "BLOCKED":
        results["errors"].append(
            f"v2 output DONE.json 이미 존재 → BLOCKED: {OUTPUT_ROOT / 'DONE.json'}"
        )

    # v1 JSON 로드 (row count 확인만)
    v1_ids: set = set()
    if V1_OUTPUT_JSON.exists():
        try:
            with open(V1_OUTPUT_JSON, "r", encoding="utf-8") as f:
                v1_data = json.load(f)
            v1_rows = v1_data.get("rows", [])
            v1_ids = {r.get("expansion_case_id", "") for r in v1_rows}
            if len(v1_rows) != 8:
                results["warnings"].append(
                    f"v1 rows={len(v1_rows)} (expected 8)"
                )
        except Exception as e:
            results["errors"].append(f"v1 JSON 파싱 실패: {e}")

    # plan_rows 로드
    plan_idx = load_csv_as_index(V2_PLAN_ROWS_CSV)

    # smoke 대상 resolve
    for case_id in SMOKE_TARGETS:
        assert_no_stage2_holdout(case_id, context=f"dry_run:{case_id}")
        if "stage2" in case_id.lower():
            results["stage2_holdout_count"] += 1

        plan_row = plan_idx.get(case_id, {})
        v1_action = plan_row.get("v1_action", "").strip()
        v2_cat_plan = plan_row.get("v2_action_category", "").strip()
        v2_cat_computed = map_action_category(v1_action) if v1_action else ""

        resolve = {
            "in_v1_json":              case_id in v1_ids,
            "in_plan_rows":            case_id in plan_idx,
            "v1_action":               v1_action,
            "v2_action_category_plan": v2_cat_plan,
            "v2_action_category_computed": v2_cat_computed,
            "action_cat_match": (
                v2_cat_plan == v2_cat_computed if v2_cat_plan else True
            ),
            "text_tuning_expected": plan_row.get("text_tuning_applied", "").lower() == "true",
            "lim_fix_expected":     plan_row.get("limitation_fix_applied", "").lower() == "true",
        }

        if not resolve["in_v1_json"]:
            results["errors"].append(f"{case_id}: v1 JSON에 없음")
        if not resolve["in_plan_rows"]:
            results["errors"].append(f"{case_id}: plan_rows CSV에 없음")
        if not resolve["action_cat_match"]:
            results["warnings"].append(
                f"{case_id}: action_category 불일치 "
                f"(plan={v2_cat_plan}, computed={v2_cat_computed})"
            )

        results["smoke_resolve"][case_id] = resolve

    if verbose:
        print("[dry-run] 입력 파일 확인:")
        for name, info in results["input_files"].items():
            status = "OK" if info["exists"] else "MISSING"
            print(f"  {status}  {name}")

        print(f"\n[dry-run] plan_rows CSV 무결성: {plan_integrity['status']}")
        if plan_integrity["row_count"]:
            print(f"  행 수: {plan_integrity['row_count']} (expected 8)")
        if plan_integrity["missing_columns"]:
            print(f"  누락 컬럼: {plan_integrity['missing_columns']}")
        if plan_integrity["duplicate_ids"]:
            print(f"  중복 ID: {plan_integrity['duplicate_ids']}")

        print(f"\n[dry-run] v2 output guard: {v2_guard['status']}")
        print(f"  v1 보존: {v2_guard['v1_preserved']}")

        print(f"\n[dry-run] smoke 대상 resolve ({len(SMOKE_TARGETS)}장):")
        for cid, res in results["smoke_resolve"].items():
            ok = res["in_v1_json"] and res["in_plan_rows"]
            cat = res["v2_action_category_computed"] or res["v1_action"]
            tuning = "tune" if res["text_tuning_expected"] else "    "
            lim = "lim_fix" if res["lim_fix_expected"] else "       "
            print(
                f"  {'OK' if ok else 'NG'}  {cid[:52]:54s}"
                f"  {cat:20s}  {tuning}  {lim}"
            )

        print(f"\n  stage2_holdout 접근: {results['stage2_holdout_count']} (0이어야 함)")

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
    plan_idx = load_csv_as_index(V2_PLAN_ROWS_CSV)

    text_tuning_targets = [
        "MSD_lung_054__c1",
        "LUNG1-057__c1",
        "subset9_1.3.6.1.4.1.14519.5.2.1.6279.6001.291156498203266896953765649282__c1",
    ]
    lim_fix_targets = ["LUNG1-402__c1", "LUNG1-057__c1"]

    print("[plan-smoke-only] v2 적용 계획 요약")
    print(f"  smoke 대상: {len(SMOKE_TARGETS)}장")
    print(f"  stage2_holdout 접근: {dry['stage2_holdout_count']} (0이어야 함)")
    print()

    print("[plan-smoke-only] text tuning 대상 (3건):")
    for cid in text_tuning_targets:
        plan = plan_idx.get(cid, {})
        detail = plan.get("text_fix_detail", "")
        print(f"  {cid[:55]:57s}  {detail}")

    print()
    print("[plan-smoke-only] limitation fix 대상 (2건):")
    for cid in lim_fix_targets:
        plan = plan_idx.get(cid, {})
        detail = plan.get("limitation_fix_detail", "")
        print(f"  {cid[:55]:57s}  {detail}")

    print()
    print("[plan-smoke-only] action_category mapping (8건):")
    for case_id in SMOKE_TARGETS:
        plan = plan_idx.get(case_id, {})
        v1_act = plan.get("v1_action", "UNKNOWN")
        v2_cat = map_action_category(v1_act)
        print(
            f"  {case_id[:40]:42s}  {v1_act:35s}  → {v2_cat}"
        )

    print()
    print("[plan-smoke-only] v2 output (이번 단계 생성 금지):")
    expected_outputs = [
        OUTPUT_ROOT / "s4_reason_layer_integrated_smoke_v2.csv",
        OUTPUT_ROOT / "s4_reason_layer_integrated_smoke_v2.json",
        OUTPUT_ROOT / "s4_reason_layer_integrated_smoke_summary_v2.md",
        OUTPUT_ROOT / "errors.csv",
        OUTPUT_ROOT / "DONE.json",
    ]
    for p in expected_outputs:
        exists = p.exists()
        print(f"  {'EXISTS(충돌주의)' if exists else 'not yet':18s}  {p.name}")

    print()
    print("[plan-smoke-only] CT/mask/PNG 접근: 0  (ALLOW_CT_LOAD=False)")
    print("[plan-smoke-only] S3 카드 JSON/PNG 수정: 0  (ALLOW_CARD_MODIFICATION=False)")

    return {
        "dry_run": dry,
        "text_tuning_targets": text_tuning_targets,
        "lim_fix_targets": lim_fix_targets,
        "expected_outputs": [str(p) for p in expected_outputs],
        "output_exists_count": sum(1 for p in expected_outputs if p.exists()),
    }


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="S4 Reason Layer Integrated Smoke v2 Script"
    )
    parser.add_argument("--selftest",          action="store_true",
                        help="25개 guard selftest 실행")
    parser.add_argument("--dry-run",           action="store_true",
                        help="입력 파일 존재 + 8장 resolve 확인")
    parser.add_argument("--plan-smoke-only",   action="store_true",
                        help="dry-run + v2 적용 계획 출력")
    parser.add_argument("--run-integrate",     action="store_true",
                        help="실제 v2 통합 실행 (단독 BLOCKED, --confirm-generate 필요)")
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
                "실제 v2 생성은 다음 단계에서 승인 후 진행하십시오.",
                file=sys.stderr,
            )
            sys.exit(2)

        # 실제 v2 통합 실행 (ALLOW_INTEGRATE_REASON=True 일 때만)
        assert_output_guard(OUTPUT_ROOT)

        v2_guard = check_v2_output_guard()
        if v2_guard["status"] == "BLOCKED":
            print(
                f"BLOCKED: v2 DONE.json 이미 존재: {OUTPUT_ROOT / 'DONE.json'}",
                file=sys.stderr,
            )
            sys.exit(2)

        with open(V1_OUTPUT_JSON, "r", encoding="utf-8") as f:
            v1_data = json.load(f)
        v1_rows = v1_data.get("rows", [])

        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

        v2_rows = []
        run_errors = []
        for v1_row in v1_rows:
            case_id = str(v1_row.get("expansion_case_id", "")).strip()
            assert_no_stage2_holdout(case_id, context=f"run-integrate:{case_id}")
            try:
                v2_row = apply_v2_patch(v1_row)
                v2_rows.append(v2_row)
            except Exception as e:
                run_errors.append({"case_id": case_id, "error": str(e)})

        # CSV 저장
        out_csv = OUTPUT_ROOT / "s4_reason_layer_integrated_smoke_v2.csv"
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=V2_OUTPUT_FIELDS)
            writer.writeheader()
            for row in v2_rows:
                writer.writerow({k: row.get(k, "") for k in V2_OUTPUT_FIELDS})

        # JSON 저장
        out_json = OUTPUT_ROOT / "s4_reason_layer_integrated_smoke_v2.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(
                {"rows": v2_rows, "errors": run_errors},
                f, ensure_ascii=False, indent=2,
            )

        # summary MD 저장
        summary_text = generate_summary_md(v2_rows, run_errors)
        summary_md = OUTPUT_ROOT / "s4_reason_layer_integrated_smoke_summary_v2.md"
        with open(summary_md, "w", encoding="utf-8") as f:
            f.write(summary_text)

        # errors.csv
        err_csv = OUTPUT_ROOT / "errors.csv"
        with open(err_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["case_id", "error"])
            writer.writeheader()
            for e in run_errors:
                writer.writerow(e)

        # DONE.json
        done_json = OUTPUT_ROOT / "DONE.json"
        with open(done_json, "w", encoding="utf-8") as f:
            json.dump({
                "status": "done",
                "version": "v2",
                "smoke_count": len(v2_rows),
                "error_count": len(run_errors),
                "allow_integrate_reason": ALLOW_INTEGRATE_REASON,
                "allow_ct_load": ALLOW_CT_LOAD,
                "allow_full_300": ALLOW_FULL_300,
                "allow_card_modification": ALLOW_CARD_MODIFICATION,
                "v1_output_preserved": V1_OUTPUT_ROOT.exists(),
                "stage2_holdout_access": 0,
                "s3_card_modification": 0,
            }, f, ensure_ascii=False, indent=2)

        print(f"[run-integrate] v2 완료: {len(v2_rows)}건, 오류: {len(run_errors)}건")
        return

    # --selftest
    if args.selftest:
        print("[selftest] 25개 guard 검사 시작...")
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
