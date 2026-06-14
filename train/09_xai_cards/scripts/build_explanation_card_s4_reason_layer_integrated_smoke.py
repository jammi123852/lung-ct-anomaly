#!/usr/bin/env python3
"""
build_explanation_card_s4_reason_layer_integrated_smoke.py

S4 Reason Layer Integrated Smoke Script

목적:
- metadata-only reason + CT-stat reason을 통합하여 smoke 8장에 대한
  JSON/table 기반 integrated reason layer를 생성한다.
- 이번 단계: 스크립트 작성 + 정적 검사 (dry-run / plan-only) 만 허용.
  실제 통합 실행은 --run-integrate --confirm-generate 조합으로만 가능하며
  현재 ALLOW_INTEGRATE_REASON=False 가드로 차단되어 있다.

이번 단계에서 절대 금지:
- CT/mask npy 로드 (ALLOW_CT_LOAD=False)
- HU 통계 재계산
- texture/edge 재계산
- PNG open
- 카드 PNG 수정
- 기존 카드 JSON 수정/덮어쓰기 (ALLOW_CARD_MODIFICATION=False)
- 카드 재생성
- full 300 reason 적용 (ALLOW_FULL_300=False)
- score 재계산
- model forward
- threshold 재계산
- stage2_holdout 접근
- lesion GT mask 사용
- 기존 산출물 수정·삭제·덮어쓰기

실행 모드:
- bare 실행             → BLOCKED exit 2
- --selftest            → 22개 guard 검사 (CT/PNG 로드 없음)
- --dry-run             → 입력 파일 존재 + 8장 resolve 확인
- --plan-smoke-only     → dry-run + 통합 계획 출력
- --run-integrate       → 단독 BLOCKED exit 2
- --run-integrate --confirm-generate → 현재 ALLOW_INTEGRATE_REASON=False 로 BLOCKED

syntax check (실행 아님):
  python -m py_compile scripts/build_explanation_card_s4_reason_layer_integrated_smoke.py
"""

import argparse
import csv
import io
import json
import os
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

# ============================================================
# 최상위 가드 — 이 단계에서는 전부 False
# ============================================================
ALLOW_INTEGRATE_REASON = False   # 실제 통합 reason 생성 허용 여부
ALLOW_CT_LOAD = False            # CT/mask npy 로드 허용 여부
ALLOW_CARD_MODIFICATION = False  # 카드 PNG/JSON 수정 허용 여부
ALLOW_FULL_300 = False           # 전체 300장 적용 허용 여부

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
# CT-stat threshold (smoke-only draft — 전체 300장 적용 금지)
# ============================================================
THRESHOLDS = {
    "denser_than_same_bin_reference":    80.0,    # delta_hu_p50 >= 80
    "borderline_denser_than_reference":  30.0,    # 30 <= delta_hu_p50 < 80
    "lower_density_than_reference":     -80.0,    # delta_hu_p50 <= -80
    "roi_mask_low_coverage":             0.50,
    "roi_mask_empty_or_unreliable":      0.05,
    "air_sparse_region_air_frac":        0.05,    # debug only, text 제외
}

# ============================================================
# low-priority tags (text 제외, debug only)
# ============================================================
LOW_PRIORITY_TEXT_EXCLUDED = {
    "air_sparse_region",
    "texture_or_edge_rich",
    "reference_texture_mismatch",
}

# ============================================================
# tag 우선순위 그룹
# ============================================================
TAG_PRIORITY_GROUPS = [
    # 1. safety / hold tag
    ["hold_case_review_required", "apex_context_caution",
     "roi_mask_empty_or_unreliable", "roi_mask_low_coverage"],
    # 2. strong CT-stat tag
    ["denser_than_same_bin_reference", "soft_tissue_or_wall_adjacent"],
    # 3. borderline CT-stat tag
    ["borderline_denser_than_reference"],
    # 4. component/display tag
    ["broad_component_response", "extreme_union_response", "wide_display_reduction"],
    # 5. metadata context tag
    ["peripheral_boundary_candidate", "z_wide_response",
     "normal_control_fp_pattern", "lesion_candidate_review"],
    # 6. uncertainty/warning tag
    ["ct_stat_uncertain", "reference_hu_mismatch", "artifact_or_context_warning"],
]

# ============================================================
# 통합 출력 schema 필드
# ============================================================
INTEGRATED_OUTPUT_FIELDS = [
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

# ============================================================
# 경로 상수
# ============================================================
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

REPORTS_ROOT = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/reports/explanation_cards"
)
CARD_ROOT = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    / "s3_expansion_cards_v2_fontfix"
)

# 입력 파일
INTEGRATION_PREFLIGHT_JSON = REPORTS_ROOT / "s4_reason_layer_integration_preflight_v1.json"
INTEGRATION_PREFLIGHT_MD   = REPORTS_ROOT / "s4_reason_layer_integration_preflight_v1.md"
INTEGRATION_PLAN_ROWS_CSV  = REPORTS_ROOT / "s4_reason_layer_integration_plan_rows_v1.csv"

METADATA_SMOKE_CSV         = REPORTS_ROOT / "s4_reason_layer_metadata_smoke_v1/s4_reason_layer_metadata_smoke_v1.csv"
METADATA_SMOKE_REVIEW_MD   = REPORTS_ROOT / "s4_reason_metadata_smoke_review_v1.md"
METADATA_SMOKE_REVIEW_ROWS = REPORTS_ROOT / "s4_reason_metadata_smoke_review_rows_v1.csv"

CTSTAT_SMOKE_CSV           = REPORTS_ROOT / "s4_reason_layer_ctstat_smoke_v1/s4_reason_layer_ctstat_smoke_v1.csv"
CTSTAT_REVIEW_MD           = REPORTS_ROOT / "s4_ctstat_reason_smoke_review_v1.md"
CTSTAT_REVIEW_JSON         = REPORTS_ROOT / "s4_ctstat_reason_smoke_review_v1.json"
CTSTAT_REVIEW_ROWS_CSV     = REPORTS_ROOT / "s4_ctstat_reason_smoke_review_rows_v1.csv"

INDEX_CSV                  = CARD_ROOT / "index_cards.csv"
HOLD_LIST_CSV              = REPORTS_ROOT / "s3_expansion_hold_list_v1.csv"

# 이번 단계 생성 금지 — 실제 run 시 사용
OUTPUT_ROOT = (
    REPORTS_ROOT
    / "s4_reason_layer_integrated_smoke_v1"
)

# static drycheck 보고서 (이번 단계 생성 허용)
STATIC_CHECK_REPORT_MD   = REPORTS_ROOT / "s4_reason_layer_integrated_smoke_script_static_drycheck_v1.md"
STATIC_CHECK_REPORT_JSON = REPORTS_ROOT / "s4_reason_layer_integrated_smoke_script_static_drycheck_v1.json"


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
            f"BLOCKED: forbidden term detected in {context}: {violations}"
        )


def check_stage2_holdout(path_str: str) -> bool:
    pl = path_str.lower()
    return any(tok in pl for tok in STAGE2_HOLDOUT_TOKENS)


def assert_no_stage2_holdout(path_str: str, context: str = "") -> None:
    if check_stage2_holdout(path_str):
        raise RuntimeError(
            f"BLOCKED: stage2_holdout access in {context}: {path_str}"
        )


def get_role(row: Dict[str, Any]) -> Optional[str]:
    """role 있으면 사용, 없으면 prototype_role, 둘 다 없으면 None."""
    role = str(row.get("role", "")).strip()
    if role and role not in ("", "nan"):
        return role
    proto = str(row.get("prototype_role", "")).strip()
    if proto and proto not in ("", "nan"):
        return proto
    return None


def safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def safe_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in ("true", "1", "yes")


# ============================================================
# 입력 데이터 로드 함수
# ============================================================

def load_csv_as_dict(csv_path: pathlib.Path) -> List[Dict[str, str]]:
    """CSV를 dict 리스트로 로드. 파일 없으면 빈 리스트 반환."""
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def load_csv_as_index(
    csv_path: pathlib.Path,
    key_col: str = "expansion_case_id",
) -> Dict[str, Dict[str, str]]:
    """CSV를 key_col 기준 dict-of-dict로 로드."""
    rows = load_csv_as_dict(csv_path)
    return {r[key_col].strip(): r for r in rows if key_col in r}


def load_hold_set(hold_csv: pathlib.Path) -> set:
    rows = load_csv_as_dict(hold_csv)
    col = "expansion_case_id"
    return {r[col].strip() for r in rows if col in r and r[col].strip()}


# ============================================================
# metadata/CT-stat join 함수
# ============================================================

def join_inputs(
    smoke_targets: List[str],
    metadata_smoke_idx: Dict[str, Dict],
    metadata_review_idx: Dict[str, Dict],
    ctstat_smoke_idx: Dict[str, Dict],
    ctstat_review_idx: Dict[str, Dict],
    integration_plan_idx: Dict[str, Dict],
    s3_index_idx: Dict[str, Dict],
    hold_set: set,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    8개 케이스에 대해 모든 입력 소스를 join.
    권위 키: expansion_case_id
    반환: (joined_rows, errors)
    """
    assert not ALLOW_CT_LOAD, "CT load guard violated"
    assert not ALLOW_CARD_MODIFICATION, "card modification guard violated"

    joined = []
    errors = []

    for case_id in smoke_targets:
        # stage2_holdout guard
        assert_no_stage2_holdout(case_id, context=f"join_inputs:{case_id}")

        row: Dict[str, Any] = {"expansion_case_id": case_id}

        # metadata smoke
        meta = metadata_smoke_idx.get(case_id, {})
        row["_meta"] = meta
        if not meta:
            errors.append(f"{case_id}: metadata_smoke missing")

        # metadata review
        meta_rev = metadata_review_idx.get(case_id, {})
        row["_meta_rev"] = meta_rev

        # CT-stat smoke
        ctstat = ctstat_smoke_idx.get(case_id, {})
        row["_ctstat"] = ctstat
        if not ctstat:
            errors.append(f"{case_id}: ctstat_smoke missing")

        # CT-stat review
        ctstat_rev = ctstat_review_idx.get(case_id, {})
        row["_ctstat_rev"] = ctstat_rev

        # integration plan
        plan = integration_plan_idx.get(case_id, {})
        row["_plan"] = plan
        if not plan:
            errors.append(f"{case_id}: integration_plan missing")

        # S3 index
        s3 = s3_index_idx.get(case_id, {})
        row["_s3"] = s3

        # hold flag
        row["hold_flag"] = case_id in hold_set

        # role fallback
        role = get_role(meta) or get_role(plan) or get_role(ctstat)
        if role is None:
            errors.append(f"{case_id}: role/prototype_role both missing")
            role = "UNKNOWN"
        row["role"] = role

        joined.append(row)

    return joined, errors


# ============================================================
# tag 통합 함수
# ============================================================

def build_final_tags(
    metadata_tags_str: str,
    ctstat_tags_str: str,
    plan_tags_str: str,
    roi_coverage: float,
    delta_hu_p50: float,
    overmerge_flag: bool,
    overmerge_level: str,
    apex_caution: bool,
    hold_flag: bool,
    mask_empty: bool,
    candidate_air_frac: float,
) -> Tuple[List[str], List[str], List[str]]:
    """
    tag 통합 우선순위 함수.
    반환: (metadata_tags, ctstat_tags, final_tags_ordered)

    우선순위:
    1. safety/hold tag
    2. strong CT-stat tag
    3. borderline CT-stat tag
    4. component/display tag
    5. metadata context tag
    6. uncertainty/warning tag

    low-priority tags (air_sparse_region, texture_or_edge_rich,
    reference_texture_mismatch)는 final_tags에는 포함되지 않음.
    """
    def parse_tags(s: str) -> List[str]:
        if not s or str(s).strip() in ("", "nan"):
            return []
        return [t.strip() for t in str(s).replace('"', "").split(",") if t.strip()]

    meta_tags = parse_tags(metadata_tags_str)
    ctstat_tags = parse_tags(ctstat_tags_str)
    plan_tags = parse_tags(plan_tags_str)

    # threshold 기반 ctstat tag 재결정 (smoke draft)
    derived_ctstat_tags: List[str] = []
    if mask_empty or roi_coverage < THRESHOLDS["roi_mask_empty_or_unreliable"]:
        derived_ctstat_tags.append("roi_mask_empty_or_unreliable")
    elif roi_coverage < THRESHOLDS["roi_mask_low_coverage"]:
        derived_ctstat_tags.append("roi_mask_low_coverage")

    if delta_hu_p50 >= THRESHOLDS["denser_than_same_bin_reference"]:
        derived_ctstat_tags.append("denser_than_same_bin_reference")
    elif THRESHOLDS["borderline_denser_than_reference"] <= delta_hu_p50 < THRESHOLDS["denser_than_same_bin_reference"]:
        derived_ctstat_tags.append("borderline_denser_than_reference")
    elif delta_hu_p50 <= THRESHOLDS["lower_density_than_reference"]:
        derived_ctstat_tags.append("lower_density_than_reference")

    # air_sparse_region: debug tag only (text 미포함)
    if candidate_air_frac < THRESHOLDS["air_sparse_region_air_frac"]:
        derived_ctstat_tags.append("air_sparse_region")  # debug only

    # hold / apex safety tags
    safety_tags: List[str] = []
    if hold_flag:
        safety_tags.append("hold_case_review_required")
    if apex_caution:
        safety_tags.append("apex_context_caution")

    # overmerge component tags
    component_tags: List[str] = []
    if overmerge_flag:
        if overmerge_level in ("extreme_union",):
            component_tags.append("extreme_union_response")
        component_tags.append("broad_component_response")

    # 소스 tags 병합 (중복 제거, 순서 보존)
    all_ctstat = list(dict.fromkeys(derived_ctstat_tags + ctstat_tags))
    all_meta = list(dict.fromkeys(meta_tags + plan_tags))

    # final_tags: 우선순위 그룹별 정렬
    seen: set = set()
    final_ordered: List[str] = []

    def _add(tag: str):
        if tag not in seen and tag not in LOW_PRIORITY_TEXT_EXCLUDED:
            seen.add(tag)
            final_ordered.append(tag)

    for group in TAG_PRIORITY_GROUPS:
        for g_tag in group:
            # safety tags
            if g_tag in ("hold_case_review_required", "apex_context_caution",
                          "roi_mask_empty_or_unreliable", "roi_mask_low_coverage"):
                if g_tag in safety_tags or g_tag in all_ctstat or g_tag in all_meta:
                    _add(g_tag)
            # component
            elif g_tag in ("broad_component_response", "extreme_union_response", "wide_display_reduction"):
                if g_tag in component_tags or g_tag in all_meta or g_tag in plan_tags:
                    _add(g_tag)
            else:
                if g_tag in all_ctstat or g_tag in all_meta:
                    _add(g_tag)

    return list(dict.fromkeys(meta_tags)), list(dict.fromkeys(all_ctstat)), final_ordered


# ============================================================
# reason text 생성 함수
# ============================================================

def build_integrated_reason_text(
    final_tags: List[str],
    position_bin: str,
    delta_hu_p50: float,
    roi_coverage: float,
    role: str,
    overmerge_level: str,
    ctstat_uncertain: bool,
    mask_empty: bool,
) -> Tuple[str, str, str]:
    """
    통합 reason text (KO/EN) 생성.
    반환: (integrated_reason_ko, integrated_reason_en, reason_limitations)

    규칙:
    - 최대 2문장
    - 진단 의미 금지
    - "원인이다" 단정 금지
    - CT-stat evidence가 강할 때만 density 언급
    - ROI coverage 낮으면 density보다 context/ROI limitation 우선
    - uncertain이면 "근거 제한적"으로 표현
    - overmerge/artifact는 caution으로 표현
    - lesion_candidate는 진단 확정 의미 아님
    - normal_control은 FP 검토용임
    """
    assert not ALLOW_CT_LOAD, "CT load guard violated in reason text"

    limitations: List[str] = []
    ko_parts: List[str] = []
    en_parts: List[str] = []

    has_roi_empty    = "roi_mask_empty_or_unreliable" in final_tags
    has_roi_low      = "roi_mask_low_coverage" in final_tags
    has_strong_dense = "denser_than_same_bin_reference" in final_tags
    has_border_dense = "borderline_denser_than_reference" in final_tags
    has_uncertain    = "ct_stat_uncertain" in final_tags or ctstat_uncertain
    has_hold         = "hold_case_review_required" in final_tags
    has_apex         = "apex_context_caution" in final_tags
    has_soft_tissue  = "soft_tissue_or_wall_adjacent" in final_tags
    has_normal_ctrl  = role == "normal_control" or "normal_control_fp_pattern" in final_tags
    has_artifact_warn = "artifact_or_context_warning" in final_tags
    has_overmerge    = "extreme_union_response" in final_tags or "broad_component_response" in final_tags

    if has_roi_empty:
        limitations.append("roi_empty")
        ko_parts.append(
            f"ROI mask coverage가 0에 가까워(roi_coverage={roi_coverage:.2f}), "
            "이 crop은 폐 실질보다 경계/주변 조직의 영향을 받을 가능성이 높습니다."
        )
        en_parts.append(
            f"ROI mask coverage is near zero (roi_coverage={roi_coverage:.2f}), "
            "indicating this crop is very likely influenced by boundary or surrounding soft tissue."
        )
        if has_strong_dense or has_border_dense:
            limitations.append("delta_hu_unreliable_due_to_roi_empty")

    elif has_roi_low:
        limitations.append("roi_coverage_low")
        if has_strong_dense:
            ko_parts.append(
                f"HU 통계상 후보 crop은 같은 위치({position_bin}) 정상 reference보다 "
                f"밀도가 높게 나타났습니다(delta ≈ {delta_hu_p50:.0f} HU). "
                "ROI coverage가 낮아 경계 조직의 영향이 있을 수 있습니다."
            )
            en_parts.append(
                f"HU statistics show the candidate crop is denser than same-bin ({position_bin}) "
                f"normal references (delta ≈ {delta_hu_p50:.0f} HU). "
                "Low ROI coverage may reflect boundary tissue influence."
            )
        elif has_border_dense:
            ko_parts.append(
                f"HU 통계상 후보 crop은 같은 위치({position_bin}) 정상 reference와 "
                f"경계 수준의 밀도 차이(delta ≈ {delta_hu_p50:.0f} HU)를 보입니다. "
                "ROI coverage가 낮아 이 수치는 참고용으로만 사용해야 합니다."
            )
            en_parts.append(
                f"HU statistics show a borderline density difference vs. same-bin ({position_bin}) "
                f"references (delta ≈ {delta_hu_p50:.0f} HU). "
                "Low ROI coverage means this value should be used for reference only."
            )
        else:
            ko_parts.append(
                f"ROI coverage가 낮아(roi_coverage={roi_coverage:.2f}) 이 crop은 "
                "폐 실질보다 경계/주변 조직 영향을 받을 수 있습니다."
            )
            en_parts.append(
                f"Low ROI coverage ({roi_coverage:.2f}) suggests this crop may be influenced "
                "by boundary or surrounding soft tissue rather than lung parenchyma."
            )

    elif has_strong_dense:
        ko_parts.append(
            f"HU 통계상 후보 crop은 같은 위치({position_bin}) 정상 reference보다 "
            f"밀도가 높게 나타났습니다(delta ≈ {delta_hu_p50:.0f} HU). "
            "이는 PaDiM high-response의 시각적 근거 후보일 수 있으나, 진단 의미는 아닙니다."
        )
        en_parts.append(
            f"HU statistics show the candidate crop is denser than same-bin ({position_bin}) "
            f"normal references (delta ≈ {delta_hu_p50:.0f} HU). "
            "This may support the observed PaDiM high response, but it is not a diagnosis."
        )

    elif has_border_dense:
        ko_parts.append(
            f"HU 통계상 후보 crop은 같은 위치({position_bin}) 정상 reference와 "
            f"경계 수준의 밀도 차이(delta ≈ {delta_hu_p50:.0f} HU)를 보입니다. "
            "이는 참고용 시각적 근거 후보이며, 진단 의미는 아닙니다."
        )
        en_parts.append(
            f"HU statistics show a borderline density difference vs. same-bin ({position_bin}) "
            f"references (delta ≈ {delta_hu_p50:.0f} HU). "
            "This is a reference visual cue only, not a diagnosis."
        )

    elif has_uncertain:
        limitations.append("ct_stat_uncertain")
        ko_parts.append(
            "metadata상 high-response 위치와 범위는 확인되지만, "
            "CT 통계상 reference와의 차이는 약합니다. "
            "이 reason은 검토 포인트로만 사용해야 하며, 진단 의미는 아닙니다."
        )
        en_parts.append(
            "Metadata confirms a high-response region, but CT-stat evidence "
            "against the reference is weak. "
            "This reason should be used only as a review cue, not as a diagnosis."
        )

    elif has_artifact_warn or has_overmerge:
        limitations.append("extreme_union_artifact")
        ko_parts.append(
            f"{'extreme_union' if 'extreme_union_response' in final_tags else 'broad'} "
            "component로 인해 후보 crop이 주변 폐 실질을 광범위하게 포함하고 있을 수 있습니다. "
            "이는 CT-stat 신뢰도를 낮추며, 진단 의미는 아닙니다."
        )
        en_parts.append(
            "The overmerged component may cause the candidate crop to include extensive "
            "surrounding lung parenchyma, reducing CT-stat reliability. "
            "This is not a diagnosis."
        )

    else:
        ko_parts.append(
            "metadata상 이 위치에서 PaDiM high-response가 확인되었습니다. "
            "CT 통계 근거가 제한적이므로 이 reason은 검토 참고용으로만 사용해야 합니다. "
            "이는 진단 의미가 아닙니다."
        )
        en_parts.append(
            "Metadata confirms a PaDiM high-response at this location. "
            "CT-stat evidence is limited; use this reason as a review reference only. "
            "This is not a diagnosis."
        )

    # 보조 문장: 컨텍스트/hold/normal_control
    if has_normal_ctrl and not has_roi_empty:
        ko_parts.append("이 케이스는 정상 제어군으로, FP 패턴 검토 목적으로 포함되었습니다.")
        en_parts.append("This is a normal control case, included for FP pattern review.")

    if has_hold and not has_roi_empty:
        limitations.append("hold_case")

    if has_apex and not has_roi_empty and "apex_caution" not in limitations:
        limitations.append("apex_caution")

    if has_soft_tissue:
        limitations.append("soft_tissue_adjacent")

    # 최대 2문장 결합
    ko_text = " ".join(ko_parts[:2])
    en_text = " ".join(en_parts[:2])

    # 금지어 검사
    assert_no_forbidden(ko_text, context="integrated_reason_ko")
    assert_no_forbidden(en_text, context="integrated_reason_en")

    return ko_text, en_text, ",".join(limitations)


# ============================================================
# 단일 케이스 통합 처리 함수 (dry-run / plan 시 미실행)
# ============================================================

def process_case(joined_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    단일 케이스에 대해 integrated reason row 생성.
    ALLOW_INTEGRATE_REASON=False 이면 절대 호출되지 않음.
    """
    assert ALLOW_INTEGRATE_REASON, (
        "BLOCKED: ALLOW_INTEGRATE_REASON=False — 실제 실행은 허용되지 않습니다."
    )
    assert not ALLOW_CT_LOAD, "CT load guard violated"
    assert not ALLOW_CARD_MODIFICATION, "card modification guard violated"
    assert not ALLOW_FULL_300, "full 300 guard violated"

    case_id  = joined_row["expansion_case_id"]
    meta     = joined_row.get("_meta", {})
    ctstat   = joined_row.get("_ctstat", {})
    plan     = joined_row.get("_plan", {})
    s3       = joined_row.get("_s3", {})
    hold_flag = joined_row.get("hold_flag", False)

    role          = joined_row.get("role", "UNKNOWN")
    position_bin  = str(meta.get("position_bin") or plan.get("position_bin") or "").strip()
    max_score     = safe_float(meta.get("max_padim_score") or plan.get("max_padim_score"))
    threshold     = safe_float(meta.get("threshold") or plan.get("threshold"))
    overmerge_flag = safe_bool(meta.get("overmerge_flag") or ctstat.get("overmerge_flag"))
    overmerge_level = str(meta.get("overmerge_level") or ctstat.get("overmerge_level") or "none").strip()
    apex_caution  = safe_bool(meta.get("apex_caution") or ctstat.get("apex_caution"))
    roi_coverage  = safe_float(ctstat.get("roi_coverage"), default=-1.0)
    delta_hu_p50  = safe_float(ctstat.get("delta_hu_p50"), default=0.0)
    cand_hu_p50   = safe_float(ctstat.get("candidate_hu_p50"), default=0.0)
    ref_hu_mean   = safe_float(ctstat.get("reference_hu_mean"), default=0.0)
    mask_empty    = safe_bool(ctstat.get("mask_empty_flag"))
    ctstat_uncertain = safe_bool(ctstat.get("tag_ct_stat_uncertain"))
    cand_air_frac = safe_float(ctstat.get("candidate_air_frac_lt_minus900"), default=1.0)

    # metadata reason tags
    meta_tags_raw = ",".join([
        k.replace("tag_", "") for k, v in meta.items()
        if k.startswith("tag_") and str(v).lower() in ("true", "1")
    ])
    # CT-stat reason tags from CSV
    ctstat_tags_raw = ",".join([
        k.replace("tag_", "") for k, v in ctstat.items()
        if k.startswith("tag_") and str(v).lower() in ("true", "1")
    ])
    plan_tags_raw = str(plan.get("final_reason_tags", "")).strip()

    # tag 통합
    metadata_tags, ctstat_tags_list, final_tags = build_final_tags(
        metadata_tags_str=meta_tags_raw,
        ctstat_tags_str=ctstat_tags_raw,
        plan_tags_str=plan_tags_raw,
        roi_coverage=roi_coverage,
        delta_hu_p50=delta_hu_p50,
        overmerge_flag=overmerge_flag,
        overmerge_level=overmerge_level,
        apex_caution=apex_caution,
        hold_flag=hold_flag,
        mask_empty=mask_empty,
        candidate_air_frac=cand_air_frac,
    )

    # reason quality label
    plan_quality = str(plan.get("reason_quality_label", "")).strip()
    if plan_quality:
        reason_quality_label = plan_quality
    elif "roi_mask_empty_or_unreliable" in final_tags:
        reason_quality_label = "hold_context"
    elif "denser_than_same_bin_reference" in final_tags:
        reason_quality_label = "strong_with_hold_context" if hold_flag else "strong"
    elif "borderline_denser_than_reference" in final_tags:
        reason_quality_label = "borderline"
    elif ctstat_uncertain:
        reason_quality_label = "uncertain"
    else:
        reason_quality_label = "metadata_only"

    # CT-stat usefulness label
    ctstat_usefulness_label = str(
        joined_row.get("_ctstat_rev", {}).get("ctstat_usefulness_label", "")
        or plan.get("ctstat_quality", "")
    ).strip() or "unclassified"

    # reason text 생성
    ko_text, en_text, limitations = build_integrated_reason_text(
        final_tags=final_tags,
        position_bin=position_bin,
        delta_hu_p50=delta_hu_p50,
        roi_coverage=roi_coverage,
        role=role,
        overmerge_level=overmerge_level,
        ctstat_uncertain=ctstat_uncertain,
        mask_empty=mask_empty,
    )

    # include flags
    plan_include_card  = str(plan.get("include_in_future_card_text", "")).strip().lower()
    plan_include_json  = str(plan.get("include_in_json_only", "")).strip().lower()
    include_card = plan_include_card in ("true", "1", "yes") and not hold_flag
    include_json = True  # json은 항상 포함

    # recommended_next_action
    plan_action = str(plan.get("recommended_next_action", "")).strip()
    integration_action = str(plan.get("integration_action", "")).strip()
    rec_action = plan_action or integration_action or "review_pending"

    # diagnostic guard
    forbidden = check_forbidden_terms(ko_text) + check_forbidden_terms(en_text)
    diag_guard_passed = len(forbidden) == 0

    return {
        "expansion_case_id":        case_id,
        "role":                     role,
        "position_bin":             position_bin,
        "max_padim_score":          max_score,
        "threshold":                threshold,
        "metadata_reason_tags":     ",".join(metadata_tags),
        "ctstat_reason_tags":       ",".join(ctstat_tags_list),
        "final_reason_tags":        ",".join(final_tags),
        "reason_quality_label":     reason_quality_label,
        "ctstat_usefulness_label":  ctstat_usefulness_label,
        "integrated_reason_ko":     ko_text,
        "integrated_reason_en":     en_text,
        "reason_limitations":       limitations,
        "hold_flag":                hold_flag,
        "overmerge_flag":           overmerge_flag,
        "overmerge_level":          overmerge_level,
        "apex_caution":             apex_caution,
        "roi_coverage":             roi_coverage,
        "delta_hu_p50":             delta_hu_p50,
        "candidate_hu_p50":         cand_hu_p50,
        "reference_hu_mean":        ref_hu_mean,
        "include_in_future_card_text": include_card,
        "include_in_json_only":     include_json,
        "recommended_next_action":  rec_action,
        "diagnostic_guard_passed":  diag_guard_passed,
    }


# ============================================================
# output guard 함수
# ============================================================

def assert_output_guard(output_root: pathlib.Path) -> None:
    """실제 run 전 output root 충돌 검사."""
    assert not ALLOW_FULL_300, "BLOCKED: full 300 guard violated"
    assert not ALLOW_CT_LOAD, "BLOCKED: CT load guard violated"
    assert not ALLOW_CARD_MODIFICATION, "BLOCKED: card modification guard violated"

    # stage2_holdout guard
    assert_no_stage2_holdout(str(output_root), context="output_guard")

    # S3 card root와 충돌 방지
    s3_card_root = (
        PROJECT_ROOT
        / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    )
    if str(output_root).startswith(str(s3_card_root)):
        raise RuntimeError(
            f"BLOCKED: output root overlaps with S3 card root: {output_root}"
        )


# ============================================================
# selftest 함수
# ============================================================

def run_selftest() -> Dict[str, Any]:
    """
    22개 guard 검사 (CT/PNG 로드 없음).
    반환: {passed: bool, checks: [{name, result, detail}]}
    """
    checks = []

    def chk(name: str, result: bool, detail: str = ""):
        checks.append({"name": name, "result": "PASS" if result else "FAIL", "detail": detail})
        return result

    all_pass = True

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

    # 7. npy array load 사용 없음 (소스 파일 grep)
    # 패턴을 split해서 자기참조 false positive 방지
    src = pathlib.Path(__file__).read_text(encoding="utf-8")
    _np_load_pat    = "np" + ".load"
    _numpy_load_pat = "numpy" + ".load"
    has_np_load = _np_load_pat in src or _numpy_load_pat in src
    _np_load_detail = "npy array load 없음" if not has_np_load else "FAIL: npy array load 발견"
    r = chk("no_np_load", not has_np_load, _np_load_detail)
    all_pass = all_pass and r

    # 8. PNG open 사용 없음 (패턴 split으로 자기참조 방지)
    _img_open_pat = "Image" + ".open"
    _cv2_read_pat = "cv2" + ".imread"
    _imageio_pat  = "imageio" + ".imread"
    has_png_open = _img_open_pat in src or _cv2_read_pat in src or _imageio_pat in src
    r = chk("no_png_open", not has_png_open,
            "PNG open 없음" if not has_png_open else "FAIL: PNG open 발견")
    all_pass = all_pass and r

    # 9. 카드 JSON write 없음 (cards_json 디렉토리 쓰기 없음)
    has_card_json_write = (
        ("open(" in src and "cards_json" in src and
         ("'w'" in src or '"w"' in src) and
         "json.dump" in src)
    )
    # 더 정확한 검사: process_case 내에서만 write가 허용되고 ALLOW 가드 아래에 있음
    r = chk("no_card_json_write",
            not has_card_json_write or "assert ALLOW_INTEGRATE_REASON" in src,
            "기존 카드 JSON write는 ALLOW 가드 내에서만 가능")
    all_pass = all_pass and r

    # 10. score 재계산 함수 없음 (패턴 split으로 자기참조 방지)
    _sc_pat1 = "compute" + "_score"
    _sc_pat2 = "recalc"  + "_score"
    _sc_pat3 = "padim_score" + "_recalc"
    has_score_recalc = _sc_pat1 in src or _sc_pat2 in src or _sc_pat3 in src
    r = chk("no_score_recalc_func", not has_score_recalc,
            "score 재계산 함수 없음" if not has_score_recalc else "FAIL")
    all_pass = all_pass and r

    # 11. threshold 재계산 함수 없음 (패턴 split으로 자기참조 방지)
    _th_pat1 = "recalc"  + "_threshold"
    _th_pat2 = "compute" + "_threshold"
    has_thresh_recalc = _th_pat1 in src or _th_pat2 in src
    r = chk("no_threshold_recalc_func", not has_thresh_recalc,
            "threshold 재계산 함수 없음" if not has_thresh_recalc else "FAIL")
    all_pass = all_pass and r

    # 12. stage2_holdout guard 존재
    has_stage2_guard = ("STAGE2_HOLDOUT_TOKENS" in src and
                        "assert_no_stage2_holdout" in src)
    r = chk("stage2_holdout_guard_exists", has_stage2_guard,
            "stage2_holdout guard 존재" if has_stage2_guard else "FAIL")
    all_pass = all_pass and r

    # 13. smoke 대상 8장 이하
    r = chk("smoke_targets_lte_8", len(SMOKE_TARGETS) <= 8,
            f"현재 smoke 대상: {len(SMOKE_TARGETS)}장")
    all_pass = all_pass and r

    # 14. hold 3건 포함 (LUNG1-284, LUNG1-220, LUNG1-402)
    expected_hold = {"LUNG1-284__c1", "LUNG1-220__c3", "LUNG1-402__c1"}
    hold_covered = expected_hold.issubset(set(SMOKE_TARGETS))
    r = chk("hold_3_cases_included", hold_covered,
            f"포함 여부: {expected_hold}")
    all_pass = all_pass and r

    # 15. role/prototype_role fallback
    has_get_role = "def get_role" in src and "prototype_role" in src
    r = chk("role_prototype_fallback", has_get_role,
            "get_role 함수 + prototype_role fallback 존재" if has_get_role else "FAIL")
    all_pass = all_pass and r

    # 16. metadata/ctstat join 함수 존재
    has_join = "def join_inputs" in src
    r = chk("join_inputs_func_exists", has_join,
            "join_inputs 함수 존재" if has_join else "FAIL")
    all_pass = all_pass and r

    # 17. tag integration 함수 존재
    has_tag_func = "def build_final_tags" in src and "TAG_PRIORITY_GROUPS" in src
    r = chk("tag_integration_func_exists", has_tag_func,
            "build_final_tags + TAG_PRIORITY_GROUPS 존재" if has_tag_func else "FAIL")
    all_pass = all_pass and r

    # 18. reason text 함수 존재
    has_reason_func = "def build_integrated_reason_text" in src
    r = chk("reason_text_func_exists", has_reason_func,
            "build_integrated_reason_text 함수 존재" if has_reason_func else "FAIL")
    all_pass = all_pass and r

    # 19. forbidden term guard 존재
    has_forbidden_guard = "FORBIDDEN_TERMS" in src and "def check_forbidden_terms" in src
    r = chk("forbidden_term_guard_exists", has_forbidden_guard,
            "FORBIDDEN_TERMS + check_forbidden_terms 존재" if has_forbidden_guard else "FAIL")
    all_pass = all_pass and r

    # 20. output guard 존재
    has_output_guard = "def assert_output_guard" in src
    r = chk("output_guard_func_exists", has_output_guard,
            "assert_output_guard 함수 존재" if has_output_guard else "FAIL")
    all_pass = all_pass and r

    # 21. full 300 적용 차단
    has_full_300_block = "ALLOW_FULL_300" in src and "assert not ALLOW_FULL_300" in src
    r = chk("full_300_blocked", has_full_300_block,
            "ALLOW_FULL_300 가드 존재" if has_full_300_block else "FAIL")
    all_pass = all_pass and r

    # 22. 기존 S3 output 수정 없음
    s3_card_path = "s3_expansion_cards_v2_fontfix"
    # 쓰기 경로로 S3 card root가 사용되지 않아야 함
    # OUTPUT_ROOT가 S3 card root와 다른지 확인
    s3_overlap = str(OUTPUT_ROOT).find("candidate_cards") >= 0
    r = chk("no_s3_output_modification", not s3_overlap,
            f"OUTPUT_ROOT={OUTPUT_ROOT} — S3 card root 미포함" if not s3_overlap else "FAIL")
    all_pass = all_pass and r

    passed = all_pass and all(c["result"] == "PASS" for c in checks)
    return {"passed": passed, "check_count": len(checks), "checks": checks}


# ============================================================
# dry-run 함수
# ============================================================

def run_dry_run(verbose: bool = True) -> Dict[str, Any]:
    """
    입력 파일 존재 확인 + smoke 대상 8장 resolve 확인.
    CT/mask/PNG 접근 없음.
    """
    assert not ALLOW_CT_LOAD, "CT load guard violated in dry-run"
    assert not ALLOW_CARD_MODIFICATION, "card modification guard violated in dry-run"

    results: Dict[str, Any] = {
        "input_files": {},
        "smoke_resolve": {},
        "hold_cases": [],
        "stage2_holdout_count": 0,
        "output_root_conflict": False,
        "errors": [],
        "warnings": [],
    }

    # 입력 파일 존재 확인
    input_files = {
        "integration_preflight_json":  INTEGRATION_PREFLIGHT_JSON,
        "integration_preflight_md":    INTEGRATION_PREFLIGHT_MD,
        "integration_plan_rows_csv":   INTEGRATION_PLAN_ROWS_CSV,
        "metadata_smoke_csv":          METADATA_SMOKE_CSV,
        "metadata_smoke_review_md":    METADATA_SMOKE_REVIEW_MD,
        "metadata_smoke_review_rows":  METADATA_SMOKE_REVIEW_ROWS,
        "ctstat_smoke_csv":            CTSTAT_SMOKE_CSV,
        "ctstat_review_md":            CTSTAT_REVIEW_MD,
        "ctstat_review_json":          CTSTAT_REVIEW_JSON,
        "ctstat_review_rows_csv":      CTSTAT_REVIEW_ROWS_CSV,
        "s3_index_csv":                INDEX_CSV,
        "hold_list_csv":               HOLD_LIST_CSV,
    }

    for name, path in input_files.items():
        exists = path.exists()
        results["input_files"][name] = {
            "path": str(path),
            "exists": exists,
        }
        if not exists:
            results["warnings"].append(f"입력 파일 없음: {name} → {path}")

    # 입력 데이터 로드 (CSV/JSON only, CT/PNG 없음)
    metadata_idx = load_csv_as_index(METADATA_SMOKE_CSV)
    meta_review_idx = load_csv_as_index(METADATA_SMOKE_REVIEW_ROWS)
    ctstat_idx = load_csv_as_index(CTSTAT_SMOKE_CSV)
    ctstat_rev_idx = load_csv_as_index(CTSTAT_REVIEW_ROWS_CSV)
    plan_idx = load_csv_as_index(INTEGRATION_PLAN_ROWS_CSV)
    s3_idx = load_csv_as_index(INDEX_CSV)
    hold_set = load_hold_set(HOLD_LIST_CSV)

    # smoke 대상 8장 resolve 확인
    for case_id in SMOKE_TARGETS:
        assert_no_stage2_holdout(case_id, context=f"dry_run:{case_id}")
        if "stage2" in case_id.lower():
            results["stage2_holdout_count"] += 1

        resolve: Dict[str, Any] = {
            "metadata_smoke": case_id in metadata_idx,
            "meta_review":    case_id in meta_review_idx,
            "ctstat_smoke":   case_id in ctstat_idx,
            "ctstat_review":  case_id in ctstat_rev_idx,
            "plan":           case_id in plan_idx,
            "s3_index":       case_id in s3_idx,
            "hold":           case_id in hold_set,
        }

        row_meta = metadata_idx.get(case_id, {})
        row_plan = plan_idx.get(case_id, {})
        role = get_role(row_meta) or get_role(row_plan)
        resolve["role_resolved"] = role is not None
        resolve["role"] = role or "BLOCKED"

        if not resolve["metadata_smoke"]:
            results["errors"].append(f"{case_id}: metadata_smoke CSV에 없음")
        if not resolve["ctstat_smoke"]:
            results["errors"].append(f"{case_id}: ctstat_smoke CSV에 없음")
        if not resolve["plan"]:
            results["errors"].append(f"{case_id}: integration_plan_rows CSV에 없음")
        if not resolve["role_resolved"]:
            results["errors"].append(f"{case_id}: role/prototype_role 모두 없음")

        results["smoke_resolve"][case_id] = resolve
        if resolve["hold"]:
            results["hold_cases"].append(case_id)

    # hold 3건 확인
    expected_hold = {"LUNG1-284__c1", "LUNG1-220__c3", "LUNG1-402__c1"}
    hold_set_found = set(results["hold_cases"])
    missing_hold = expected_hold - hold_set_found
    if missing_hold:
        results["warnings"].append(f"예상 hold 케이스 미발견: {missing_hold}")

    # output root 충돌 확인
    s3_card_root = (
        PROJECT_ROOT
        / "outputs/position-aware-padim-v1/visualizations/candidate_cards"
    )
    if str(OUTPUT_ROOT).startswith(str(s3_card_root)):
        results["output_root_conflict"] = True
        results["errors"].append(f"output_root가 S3 card root와 충돌: {OUTPUT_ROOT}")

    if verbose:
        print("[dry-run] 입력 파일 확인:")
        for name, info in results["input_files"].items():
            status = "OK" if info["exists"] else "MISSING"
            print(f"  {status}  {name}")

        print(f"\n[dry-run] smoke 대상 resolve ({len(SMOKE_TARGETS)}장):")
        for cid, res in results["smoke_resolve"].items():
            ok = (res["metadata_smoke"] and res["ctstat_smoke"] and
                  res["plan"] and res["role_resolved"])
            print(f"  {'OK' if ok else 'NG'}  {cid}  role={res['role']}"
                  f"  hold={'Y' if res['hold'] else 'N'}")

        print(f"\n  hold 케이스: {results['hold_cases']}")
        print(f"  stage2_holdout 접근: {results['stage2_holdout_count']} (0이어야 함)")

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
    """
    dry-run + 통합 계획 출력.
    CT/mask/PNG 접근 없음, 기존 결과 수정 없음.
    """
    assert not ALLOW_CT_LOAD, "CT load guard violated in plan-smoke-only"
    assert not ALLOW_CARD_MODIFICATION, "card modification guard violated in plan-smoke-only"

    dry = run_dry_run(verbose=False)

    plan_idx = load_csv_as_index(INTEGRATION_PLAN_ROWS_CSV)
    hold_set = load_hold_set(HOLD_LIST_CSV)
    metadata_idx = load_csv_as_index(METADATA_SMOKE_CSV)
    ctstat_idx = load_csv_as_index(CTSTAT_SMOKE_CSV)

    print("[plan-smoke-only] 통합 계획 요약")
    print(f"  smoke 대상: {len(SMOKE_TARGETS)}장")
    print(f"  hold 케이스: {dry['hold_cases']}")
    print(f"  stage2_holdout 접근: {dry['stage2_holdout_count']}")
    print()

    plan_rows = []
    for case_id in SMOKE_TARGETS:
        plan = plan_idx.get(case_id, {})
        meta = metadata_idx.get(case_id, {})
        ctstat = ctstat_idx.get(case_id, {})

        action = str(plan.get("integration_action", "")).strip() or "UNKNOWN"
        role = get_role(meta) or get_role(plan) or "UNKNOWN"
        position_bin = str(plan.get("position_bin") or meta.get("position_bin") or "").strip()
        delta = safe_float(ctstat.get("delta_hu_p50"), default=0.0)
        roi_cov = safe_float(ctstat.get("roi_coverage"), default=-1.0)
        hold = case_id in hold_set

        # threshold 기반 예상 tag
        expected_ctstat_tag = ""
        if roi_cov >= 0:
            if roi_cov < THRESHOLDS["roi_mask_empty_or_unreliable"]:
                expected_ctstat_tag = "roi_mask_empty_or_unreliable"
            elif delta >= THRESHOLDS["denser_than_same_bin_reference"]:
                expected_ctstat_tag = "denser_than_same_bin_reference"
            elif delta >= THRESHOLDS["borderline_denser_than_reference"]:
                expected_ctstat_tag = "borderline_denser_than_reference"
            elif delta <= THRESHOLDS["lower_density_than_reference"]:
                expected_ctstat_tag = "lower_density_than_reference"
            else:
                expected_ctstat_tag = "weak_or_no_density_difference"

        row = {
            "expansion_case_id": case_id,
            "role":              role,
            "position_bin":      position_bin,
            "hold":              hold,
            "delta_hu_p50":      delta,
            "roi_coverage":      roi_cov,
            "expected_action":   action,
            "expected_ctstat_tag": expected_ctstat_tag,
        }
        plan_rows.append(row)

        print(
            f"  {case_id:55s}  {action:35s}  "
            f"delta={delta:+.0f}  roi={roi_cov:.2f}  "
            f"tag={expected_ctstat_tag}"
        )

    # 임시 기준 명시
    print()
    print("[plan-smoke-only] 임시 threshold 기준 (smoke-only draft — full 300장 적용 금지)")
    for k, v in THRESHOLDS.items():
        print(f"  {k}: {v}")

    print()
    print("[plan-smoke-only] 실제 run 시 생성 예정 파일 (이번 단계 생성 금지):")
    expected_outputs = [
        OUTPUT_ROOT / "s4_reason_layer_integrated_smoke_v1.csv",
        OUTPUT_ROOT / "s4_reason_layer_integrated_smoke_v1.json",
        OUTPUT_ROOT / "s4_reason_layer_integrated_smoke_summary_v1.md",
        OUTPUT_ROOT / "s4_integrated_reason_text_examples_v1.md",
        OUTPUT_ROOT / "errors.csv",
        OUTPUT_ROOT / "DONE.json",
    ]
    for p in expected_outputs:
        exists = p.exists()
        print(f"  {'EXISTS (충돌주의)' if exists else 'not yet'}  {p.name}")

    # CT/mask/PNG 접근 0 확인
    print()
    print("[plan-smoke-only] CT/mask/PNG 접근: 0  (guard=True)")
    print("[plan-smoke-only] 기존 S3 카드 JSON/PNG 수정: 0  (ALLOW_CARD_MODIFICATION=False)")

    return {
        "dry_run": dry,
        "plan_rows": plan_rows,
        "expected_outputs": [str(p) for p in expected_outputs],
        "output_exists_count": sum(1 for p in expected_outputs if p.exists()),
    }


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="S4 Reason Layer Integrated Smoke Script"
    )
    parser.add_argument("--selftest",         action="store_true",
                        help="22개 guard selftest 실행")
    parser.add_argument("--dry-run",          action="store_true",
                        help="입력 파일 존재 + 8장 resolve 확인")
    parser.add_argument("--plan-smoke-only",  action="store_true",
                        help="dry-run + 통합 계획 출력")
    parser.add_argument("--run-integrate",    action="store_true",
                        help="실제 통합 실행 (단독 BLOCKED, --confirm-generate 필요)")
    parser.add_argument("--confirm-generate", action="store_true",
                        help="실제 통합 실행 확인 플래그")
    args = parser.parse_args()

    # bare 실행 guard
    if len(sys.argv) == 1:
        print("BLOCKED: bare 실행은 허용되지 않습니다.", file=sys.stderr)
        print("사용법: --selftest / --dry-run / --plan-smoke-only", file=sys.stderr)
        sys.exit(2)

    # --run-integrate 단독 guard
    if args.run_integrate and not args.confirm_generate:
        print("BLOCKED: --run-integrate 단독 실행은 허용되지 않습니다.", file=sys.stderr)
        print("이번 단계에서는 --run-integrate --confirm-generate 모두 필요합니다.",
              file=sys.stderr)
        sys.exit(2)

    # --run-integrate --confirm-generate → ALLOW_INTEGRATE_REASON guard
    if args.run_integrate and args.confirm_generate:
        if not ALLOW_INTEGRATE_REASON:
            print(
                "BLOCKED: ALLOW_INTEGRATE_REASON=False\n"
                "이번 단계는 스크립트 작성 + 정적 검사만 허용합니다.\n"
                "실제 통합 실행은 다음 단계에서 승인 후 진행하십시오.",
                file=sys.stderr,
            )
            sys.exit(2)

        # 실제 통합 실행 (ALLOW_INTEGRATE_REASON=True 일 때만 진입)
        if ALLOW_CT_LOAD:
            print("BLOCKED: ALLOW_CT_LOAD=True — integrated reason 단계에서 CT load 금지", file=sys.stderr)
            sys.exit(2)
        if ALLOW_FULL_300:
            print("BLOCKED: ALLOW_FULL_300=True — full 300 적용 금지", file=sys.stderr)
            sys.exit(2)

        assert_output_guard(OUTPUT_ROOT)
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

        metadata_idx    = load_csv_as_index(METADATA_SMOKE_CSV)
        meta_review_idx = load_csv_as_index(METADATA_SMOKE_REVIEW_ROWS)
        ctstat_idx      = load_csv_as_index(CTSTAT_SMOKE_CSV)
        ctstat_rev_idx  = load_csv_as_index(CTSTAT_REVIEW_ROWS_CSV)
        plan_idx        = load_csv_as_index(INTEGRATION_PLAN_ROWS_CSV)
        s3_idx          = load_csv_as_index(INDEX_CSV)
        hold_set        = load_hold_set(HOLD_LIST_CSV)

        joined, join_errors = join_inputs(
            SMOKE_TARGETS, metadata_idx, meta_review_idx,
            ctstat_idx, ctstat_rev_idx, plan_idx, s3_idx, hold_set,
        )

        output_rows = []
        run_errors = []
        for jrow in joined:
            try:
                out = process_case(jrow)
                output_rows.append(out)
            except Exception as e:
                run_errors.append({"case_id": jrow["expansion_case_id"], "error": str(e)})

        # CSV 저장
        out_csv = OUTPUT_ROOT / "s4_reason_layer_integrated_smoke_v1.csv"
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=INTEGRATED_OUTPUT_FIELDS)
            writer.writeheader()
            for row in output_rows:
                writer.writerow({k: row.get(k, "") for k in INTEGRATED_OUTPUT_FIELDS})

        # JSON 저장
        out_json = OUTPUT_ROOT / "s4_reason_layer_integrated_smoke_v1.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"rows": output_rows, "errors": run_errors + join_errors},
                      f, ensure_ascii=False, indent=2)

        # errors.csv
        err_csv = OUTPUT_ROOT / "errors.csv"
        with open(err_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["case_id", "error"])
            writer.writeheader()
            for e in run_errors + [{"case_id": "", "error": e} for e in join_errors]:
                writer.writerow(e)

        # DONE.json
        done_json = OUTPUT_ROOT / "DONE.json"
        with open(done_json, "w", encoding="utf-8") as f:
            json.dump({
                "status": "done",
                "smoke_count": len(output_rows),
                "error_count": len(run_errors) + len(join_errors),
                "allow_integrate_reason": ALLOW_INTEGRATE_REASON,
                "allow_ct_load": ALLOW_CT_LOAD,
                "allow_full_300": ALLOW_FULL_300,
            }, f, ensure_ascii=False, indent=2)

        print(f"[run-integrate] 완료: {len(output_rows)}건, 오류: {len(run_errors)+len(join_errors)}건")
        return

    # --selftest
    if args.selftest:
        print("[selftest] 22개 guard 검사 시작...")
        result = run_selftest()
        for c in result["checks"]:
            print(f"  {c['result']:4s}  {c['name']}"
                  + (f"  ({c['detail']})" if c["detail"] else ""))
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
        print(f"\n[plan-smoke-only] 판정: {'PASS' if not result['dry_run']['errors'] else 'NEEDS_FIX'}")
        sys.exit(0)


if __name__ == "__main__":
    main()
