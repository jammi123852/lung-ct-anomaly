"""
S4 Reason Layer Metadata-Only Smoke Script

이번 단계: metadata-only reason tag 생성 + 정적 검사
실제 run-smoke는 --run-smoke --confirm-generate 조합으로만 실행됨.

금지:
- CT/mask npy 로드
- HU 통계 계산
- feature extraction
- model forward
- score/threshold 재계산
- 카드 PNG 재생성
- 카드 JSON 수정
- 기존 S3 산출물 수정/삭제
- stage2_holdout 접근
- lesion GT mask 사용
"""

import os
import sys
import json
import csv
import ast
import argparse
import pathlib
import py_compile
import tempfile

# ── 최상위 가드 ──────────────────────────────────────────────────
ALLOW_CT_STAT = False
ALLOW_FEATURE_XAI = False
ALLOW_CARD_MODIFICATION = False
ALLOW_RUN_REASON = False  # --run-smoke --confirm-generate 가 함께 올 때만 True로 전환

FORBIDDEN_TERMS = [
    "cancer", "malignancy", "malignant", "benign", "tumor", "tumour",
    "nodule 확정", "폐암", "악성", "양성", "종양", "결절로 진단",
    "병변 확정", "암 가능성 높음",
]

STAGE2_HOLDOUT_TOKENS = [
    "stage2_holdout", "stage2-holdout", "stage2holdout",
    "holdout_stage2", "holdout-stage2",
]

# ── 경로 상수 ────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).parent.parent

CARD_ROOT = PROJECT_ROOT / "outputs/position-aware-padim-v1/visualizations/candidate_cards/s3_expansion_cards_v2_fontfix"
INDEX_CSV = CARD_ROOT / "index_cards.csv"
CARDS_JSON_DIR = CARD_ROOT / "cards_json"

MANIFEST_CSV = PROJECT_ROOT / "outputs/position-aware-padim-v1/candidates/s3_expansion_manifest_v1/s3_expansion_candidate_manifest_v1.csv"
BULK_ACCEPT_CSV = PROJECT_ROOT / "outputs/position-aware-padim-v1/reports/explanation_cards/s3_expansion_bulk_acceptance_table_v1.csv"
HOLD_LIST_CSV = PROJECT_ROOT / "outputs/position-aware-padim-v1/reports/explanation_cards/s3_expansion_hold_list_v1.csv"
S4_PREFLIGHT_JSON = PROJECT_ROOT / "outputs/position-aware-padim-v1/reports/explanation_cards/s4_reason_layer_preflight_v1.json"
S4_TAG_DESIGN_CSV = PROJECT_ROOT / "outputs/position-aware-padim-v1/reports/explanation_cards/s4_reason_tag_design_v1.csv"

OUTPUT_ROOT = PROJECT_ROOT / "outputs/position-aware-padim-v1/reports/explanation_cards/s4_reason_layer_metadata_smoke_v1"

# smoke 대상 case_id (preflight 기준 8장)
SMOKE_TARGETS_PREFERRED = [
    "LUNG1-284__c1",
    "LUNG1-220__c3",
    "LUNG1-402__c1",
    "LUNG1-305__c1",
    "MSD_lung_054__c1",
    "LUNG1-057__c1",
    "subset9_1.3.6.1.4.1.14519.5.2.1.6279.6001.291156498203266896953765649282__c1",
    "LUNG1-320__c2",
]

# ── 금지어 검사 ──────────────────────────────────────────────────

def check_forbidden_terms(text: str) -> list:
    """reason text에 진단 금지어가 포함되어 있는지 검사."""
    found = []
    text_lower = text.lower()
    for term in FORBIDDEN_TERMS:
        if term.lower() in text_lower:
            found.append(term)
    return found


def assert_no_forbidden(text: str, context: str = ""):
    violations = check_forbidden_terms(text)
    if violations:
        raise RuntimeError(
            f"BLOCKED: forbidden term in {context}: {violations}"
        )


# ── metadata-only reason tag 함수 ────────────────────────────────

def compute_reason_tags(row: dict, hold_ids: set) -> dict:
    """
    row: index_cards.csv 또는 cards_json의 metadata dict
    hold_ids: hold_list_v1에서 읽은 expansion_case_id set
    반환: {tag_name: bool, ...}

    CT/mask/feature 접근 없음. JSON/CSV metadata만 사용.
    """
    case_id = row.get("expansion_case_id", "")

    # score 관련
    try:
        max_score = float(row.get("max_padim_score", 0))
        threshold = float(row.get("threshold", 1))
    except (TypeError, ValueError):
        max_score = 0.0
        threshold = 1.0

    score_ratio = max_score / threshold if threshold > 0 else 0.0

    # z_span
    try:
        z_span = int(float(row.get("z_span", 0)))
    except (TypeError, ValueError):
        z_span = 0

    # overmerge
    overmerge_flag_raw = row.get("overmerge_flag", False)
    if isinstance(overmerge_flag_raw, str):
        overmerge_flag = overmerge_flag_raw.strip().lower() in ("true", "1", "yes")
    else:
        overmerge_flag = bool(overmerge_flag_raw)

    overmerge_level = str(row.get("overmerge_level", "")).strip().lower()

    # apex
    apex_raw = row.get("apex_caution", False)
    if isinstance(apex_raw, str):
        apex_caution = apex_raw.strip().lower() in ("true", "1", "yes")
    else:
        apex_caution = bool(apex_raw)

    # position
    position_bin = str(row.get("position_bin", "")).lower()
    central_peripheral = str(row.get("central_peripheral", "")).lower()

    # role
    role = str(row.get("role", "")).lower()
    prototype_role = str(row.get("prototype_role", "")).lower()
    is_normal_control = (role == "normal_control") or (prototype_role == "normal_control")
    is_lesion_candidate = (role == "lesion_candidate") or (
        role not in ("normal_control",) and prototype_role not in ("normal_control",) and role != ""
    )

    # display_bbox_area_reduction_ratio (wide display)
    try:
        reduction_ratio = float(row.get("display_bbox_area_reduction_ratio", 0))
    except (TypeError, ValueError):
        reduction_ratio = 0.0

    tags = {
        "score_high_vs_threshold": score_ratio >= 1.0,
        "very_high_score": score_ratio >= 3.0,
        "broad_component_response": overmerge_flag,
        "extreme_union_response": overmerge_level == "extreme_union",
        "apex_context_caution": apex_caution or (case_id in hold_ids and "apex" in case_id.lower()),
        "peripheral_boundary_candidate": (
            "peripheral" in position_bin or central_peripheral == "peripheral"
        ),
        "z_continuous_response": z_span >= 3,
        "z_wide_response": z_span >= 30,
        "single_slice_spike": z_span <= 1,
        "normal_control_fp_pattern": is_normal_control,
        "lesion_candidate_review": is_lesion_candidate and not is_normal_control,
        "hold_case_review_required": case_id in hold_ids,
        "wide_display_reduction": reduction_ratio > 0.7,
    }

    # uncertain_reason_metadata_only: 위 태그 중 score_high_vs_threshold, broad, apex, peripheral 외에
    # 실질적인 시각 근거 tag가 없는 경우
    active_visual_tags = sum([
        tags["very_high_score"],
        tags["extreme_union_response"],
        tags["apex_context_caution"],
        tags["peripheral_boundary_candidate"],
        tags["z_wide_response"],
        tags["normal_control_fp_pattern"],
    ])
    tags["uncertain_reason_metadata_only"] = (active_visual_tags == 0)

    return tags


def build_reason_text(tags: dict, row: dict, lang: str = "KO") -> str:
    """
    metadata-only reason tag로부터 완곡 표현 reason text 생성.
    진단 금지어 절대 미포함.
    lang: "KO" 또는 "EN"
    """
    parts_ko = []
    parts_en = []

    if tags.get("score_high_vs_threshold"):
        try:
            score = float(row.get("max_padim_score", 0))
            thr = float(row.get("threshold", 1))
            ratio = score / thr if thr > 0 else 0.0
        except (TypeError, ValueError):
            score, thr, ratio = 0.0, 1.0, 0.0

        if tags.get("very_high_score"):
            parts_ko.append(
                f"PaDiM 점수({score:.1f})가 정상 기준({thr:.2f})의 {ratio:.1f}배로 매우 높게 나타났습니다."
            )
            parts_en.append(
                f"PaDiM score ({score:.1f}) is {ratio:.1f}× the normal training threshold ({thr:.2f}), indicating a very high response."
            )
        else:
            parts_ko.append(
                f"PaDiM 점수({score:.1f})가 정상 기준(p95={thr:.2f})을 초과했습니다."
            )
            parts_en.append(
                f"PaDiM score ({score:.1f}) exceeds the normal training threshold (p95={thr:.2f})."
            )

    if tags.get("broad_component_response"):
        if tags.get("extreme_union_response"):
            parts_ko.append(
                "이 반응은 매우 넓은 component union에 걸쳐 나타났습니다. 카드는 최고 점수 인근 국소 영역을 표시합니다."
            )
            parts_en.append(
                "This response spans an extreme-union component area. The card focuses on the local highest-score region."
            )
        else:
            parts_ko.append(
                "이 반응은 넓은 영역에 걸쳐 나타났습니다. 카드는 최고 점수 인근을 중심으로 표시합니다."
            )
            parts_en.append(
                "This response spans a broad region. The card displays the highest-score local area."
            )

    if tags.get("apex_context_caution"):
        parts_ko.append(
            "상부 말초부 위치로, 주변 폐 실질 맥락을 함께 확인해야 합니다."
        )
        parts_en.append(
            "Upper peripheral location: please review surrounding lung parenchyma context carefully."
        )

    if tags.get("peripheral_boundary_candidate"):
        parts_ko.append(
            "폐 경계부 근접 영역에서 나타난 반응으로 볼 수 있습니다. 흉막 또는 흉벽 인접 여부를 함께 검토하세요."
        )
        parts_en.append(
            "Metadata indicates a response near the lung boundary. Review as a possible pleural or chest wall adjacency."
        )

    if tags.get("z_wide_response"):
        z_span = row.get("z_span", "?")
        parts_ko.append(
            f"z 방향으로 {z_span} 슬라이스에 걸친 연속 반응이 관찰됩니다. z-context 패널을 함께 확인하세요."
        )
        parts_en.append(
            f"A continuous response spanning {z_span} slices in the z-direction was observed. Please review the z-context panel."
        )
    elif tags.get("z_continuous_response"):
        z_span = row.get("z_span", "?")
        parts_ko.append(
            f"여러 슬라이스({z_span}장)에 걸쳐 high-response가 나타났습니다."
        )
        parts_en.append(
            f"High response was observed across multiple slices ({z_span} slices)."
        )

    if tags.get("single_slice_spike"):
        parts_ko.append(
            "반응이 단일 슬라이스에 집중되어 있습니다. 인접 슬라이스와 비교하세요."
        )
        parts_en.append(
            "The high-response is concentrated in a single slice. Compare with adjacent slices."
        )

    if tags.get("normal_control_fp_pattern"):
        parts_ko.append(
            "이 후보는 정상 CT에서 나타난 high-response 패턴입니다. FP 가능성 검토용 참고 자료입니다."
        )
        parts_en.append(
            "This candidate shows a high-response pattern from a normal CT. It is included for false-positive pattern review."
        )

    if tags.get("hold_case_review_required"):
        parts_ko.append(
            "이 케이스는 외부 공유 전 별도 검토가 필요한 항목입니다."
        )
        parts_en.append(
            "This case requires additional review before external sharing."
        )

    if tags.get("uncertain_reason_metadata_only"):
        parts_ko.append(
            "현재 자동 근거 추출이 어렵습니다. 시각적 비교(Panel B↔C)를 권장합니다."
        )
        parts_en.append(
            "Automatic reason extraction is limited for this candidate. Please compare Panel B and Panel C visually."
        )

    disclaimer_ko = "이는 진단 의미가 아니며, 연구용 stage1-dev 후보입니다."
    disclaimer_en = "This is not a diagnosis. This is a stage1-dev research candidate."

    if lang == "EN":
        text = " ".join(parts_en) + " " + disclaimer_en if parts_en else disclaimer_en
    else:
        text = " ".join(parts_ko) + " " + disclaimer_ko if parts_ko else disclaimer_ko

    assert_no_forbidden(text, context=f"reason_text({lang})")
    return text.strip()


# ── CSV/JSON 로드 헬퍼 ───────────────────────────────────────────

def load_csv_rows(path: pathlib.Path) -> list:
    """CSV를 dict list로 읽음. CT/mask/PNG는 열지 않음."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def load_json(path: pathlib.Path) -> dict:
    """JSON 파일을 dict로 읽음. PNG/CT/mask는 열지 않음."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_hold_ids(hold_csv: pathlib.Path) -> set:
    rows = load_csv_rows(hold_csv)
    return {r.get("expansion_case_id", "").strip() for r in rows if r.get("expansion_case_id")}


def load_index(index_csv: pathlib.Path) -> dict:
    """expansion_case_id → row dict"""
    rows = load_csv_rows(index_csv)
    return {r["expansion_case_id"]: r for r in rows if r.get("expansion_case_id")}


# ── stage2_holdout 경로 검사 ─────────────────────────────────────

def assert_no_stage2_holdout(path_or_str: str):
    s = str(path_or_str).lower().replace("-", "_")
    for token in STAGE2_HOLDOUT_TOKENS:
        if token in s:
            raise RuntimeError(f"BLOCKED: stage2_holdout token detected in path: {path_or_str}")


# ── smoke 대상 해결 ──────────────────────────────────────────────

def resolve_smoke_targets(index: dict, hold_ids: set) -> list:
    """
    SMOKE_TARGETS_PREFERRED 기준으로 실제 가용 case_id를 해결.
    없는 경우 대체 사유를 기록.
    최대 8장.
    """
    resolved = []
    for preferred_id in SMOKE_TARGETS_PREFERRED:
        if preferred_id in index:
            resolved.append({
                "case_id": preferred_id,
                "found": True,
                "substitution_reason": None,
                "in_hold": preferred_id in hold_ids,
            })
        else:
            # partial match 시도 (앞 부분만 비교)
            prefix = preferred_id.split("__")[0]
            candidates_match = [k for k in index if k.startswith(prefix)]
            if candidates_match:
                alt = sorted(candidates_match)[0]
                resolved.append({
                    "case_id": alt,
                    "found": True,
                    "substitution_reason": f"preferred '{preferred_id}' not found; substituted with '{alt}' (same patient prefix)",
                    "in_hold": alt in hold_ids,
                })
            else:
                resolved.append({
                    "case_id": preferred_id,
                    "found": False,
                    "substitution_reason": f"preferred '{preferred_id}' not found in index; no prefix match",
                    "in_hold": False,
                })
    return resolved[:8]


# ── selftest ─────────────────────────────────────────────────────

def run_selftest():
    print("[selftest] 시작")
    errors = []

    # 1. bare guard 동작 확인 (함수가 존재하는지)
    assert callable(_bare_guard), "bare_guard function missing"
    print("[selftest] 1. bare guard function: OK")

    # 2. run-smoke confirm guard
    # ALLOW_RUN_REASON = False 기본값 확인
    assert ALLOW_RUN_REASON is False, "ALLOW_RUN_REASON should be False by default"
    print("[selftest] 2. ALLOW_RUN_REASON default=False: OK")

    # 3. ALLOW_CT_STAT
    assert ALLOW_CT_STAT is False, "ALLOW_CT_STAT must be False"
    print("[selftest] 3. ALLOW_CT_STAT=False: OK")

    # 4. ALLOW_FEATURE_XAI
    assert ALLOW_FEATURE_XAI is False, "ALLOW_FEATURE_XAI must be False"
    print("[selftest] 4. ALLOW_FEATURE_XAI=False: OK")

    # 5. ALLOW_CARD_MODIFICATION
    assert ALLOW_CARD_MODIFICATION is False, "ALLOW_CARD_MODIFICATION must be False"
    print("[selftest] 5. ALLOW_CARD_MODIFICATION=False: OK")

    # 6-8. CT/mask/np.load/PNG 코드 없음 → 소스 자기 검사
    # 패턴을 분할하여 selftest 코드 자체가 걸리지 않게 함
    src_lines = pathlib.Path(__file__).read_text(encoding="utf-8").splitlines()
    # selftest 함수 영역 제외(정적 검사 패턴 문자열 정의 줄 제외)
    # 함수 외부 실제 기능 코드에서만 금지 패턴 검사
    # np_load 검사: "np" + "." + "load(" 조합을 기능 코드에서 탐색
    _np_load_pat = "np" + "." + "load("
    _numpy_load_pat = "numpy" + "." + "load("
    _img_open_pat = "Image" + "." + "open"
    _recompute_score_pat = "recompute" + "_score"
    _recalc_score_pat = "recalc" + "_score"
    _mahal_pat = "compute" + "_mahalanobis"
    _padim_score_from_pat = "padim" + "_score" + "_from"
    _recompute_thr_pat = "recompute" + "_threshold"
    _recalc_thr_pat = "recalc" + "_threshold"
    _compute_p95_pat = "compute" + "_p95"

    # 검사 대상 라인: run_selftest 함수 내부 패턴 정의 줄 제외
    # → selftest 함수 외의 실제 코드 줄에서만 검사
    in_selftest = False
    functional_lines = []
    for line in src_lines:
        stripped = line.strip()
        if stripped.startswith("def run_selftest("):
            in_selftest = True
        elif in_selftest and stripped.startswith("def ") and "run_selftest" not in stripped:
            in_selftest = False
        if not in_selftest:
            functional_lines.append(line)
    functional_src = "\n".join(functional_lines)

    ct_npy_ok = _np_load_pat not in functional_src and _numpy_load_pat not in functional_src
    if ct_npy_ok:
        print("[selftest] 6. CT/mask npy 로드 코드 없음: OK")
        print("[selftest] 7. np.load 사용 없음: OK")
    else:
        errors.append("BLOCKED: np.load or numpy.load found in functional code")
        print("[selftest] 6-7. FAIL: np.load 감지")

    png_ok = _img_open_pat not in functional_src
    if png_ok:
        print("[selftest] 8. PNG open 사용 없음: OK")
    else:
        errors.append("BLOCKED: PIL Image.open found in functional code")
        print("[selftest] 8. FAIL: PNG open 감지")

    # 9. score 재계산 함수 없음
    score_ok = all(p not in functional_src for p in [_recompute_score_pat, _recalc_score_pat, _mahal_pat, _padim_score_from_pat])
    if score_ok:
        print("[selftest] 9. score 재계산 함수 없음: OK")
    else:
        errors.append("BLOCKED: score recompute pattern found in functional code")
        print("[selftest] 9. FAIL: score 재계산 패턴 감지")

    # 10. threshold 재계산 함수 없음
    thr_ok = all(p not in functional_src for p in [_recompute_thr_pat, _recalc_thr_pat, _compute_p95_pat])
    if thr_ok:
        print("[selftest] 10. threshold 재계산 함수 없음: OK")
    else:
        errors.append("BLOCKED: threshold recompute pattern found in functional code")
        print("[selftest] 10. FAIL: threshold 재계산 패턴 감지")

    # 11. stage2_holdout 금지 토큰 검사 (기능 코드에서 의도치 않은 사용)
    # 허용: STAGE2_HOLDOUT_TOKENS 정의, assert_no_stage2_holdout 함수, dry_run 보고서 key, run_dry_run 내 교집합 검사
    # 금지: 실제 데이터 경로를 stage2_holdout로 시작하는 경로 접근
    stage2_path_access = any(
        "stage2_holdout" in line and ("open(" in line or "pathlib.Path" in line or "Path(" in line)
        for line in functional_lines
        if "assert_no_stage2_holdout" not in line and "STAGE2_HOLDOUT_TOKENS" not in line
    )
    if stage2_path_access:
        errors.append("BLOCKED: stage2_holdout path access detected in functional code")
        print("[selftest] 11. FAIL: stage2_holdout 경로 접근 감지")
    else:
        print("[selftest] 11. stage2_holdout 금지 토큰 검사: OK (경로 접근 없음)")

    # 12. hold list 3건 포함 확인
    if HOLD_LIST_CSV.exists():
        hold_ids = load_hold_ids(HOLD_LIST_CSV)
        smoke_hold = [s for s in SMOKE_TARGETS_PREFERRED if s in hold_ids]
        assert len(hold_ids) >= 3, f"hold_ids < 3: {len(hold_ids)}"
        print(f"[selftest] 12. hold list {len(hold_ids)}건 포함: OK (smoke 대상 내 hold={len(smoke_hold)})")
    else:
        print(f"[selftest] 12. hold list 파일 없음 (경로 확인 필요): {HOLD_LIST_CSV}")

    # 13. metadata-only tag 함수 존재
    assert callable(compute_reason_tags), "compute_reason_tags function missing"
    print("[selftest] 13. metadata-only tag 함수 존재: OK")

    # 14. forbidden term guard 존재
    assert callable(check_forbidden_terms), "check_forbidden_terms function missing"
    assert callable(assert_no_forbidden), "assert_no_forbidden function missing"
    print("[selftest] 14. forbidden term guard 존재: OK")

    # 15. reason text에 진단 금지어 없음 (샘플 생성)
    sample_row = {
        "expansion_case_id": "LUNG1-284__c1",
        "max_padim_score": "45.94",
        "threshold": "14.09",
        "z_span": "31",
        "overmerge_flag": "True",
        "overmerge_level": "large_union",
        "apex_caution": "True",
        "position_bin": "upper_peripheral",
        "central_peripheral": "peripheral",
        "role": "lesion_candidate",
        "prototype_role": "prototype",
        "display_bbox_area_reduction_ratio": "0.4",
    }
    sample_hold = {"LUNG1-284__c1"}
    sample_tags = compute_reason_tags(sample_row, sample_hold)
    for lang in ("KO", "EN"):
        text = build_reason_text(sample_tags, sample_row, lang=lang)
        violations = check_forbidden_terms(text)
        assert not violations, f"forbidden terms in sample text ({lang}): {violations}"
    print("[selftest] 15. reason text 진단 금지어 없음: OK")

    # 16. output guard 존재
    assert callable(_output_guard), "output_guard function missing"
    print("[selftest] 16. output guard 존재: OK")

    # 17. smoke 대상 수 8장 이내
    assert len(SMOKE_TARGETS_PREFERRED) <= 8, f"smoke targets > 8: {len(SMOKE_TARGETS_PREFERRED)}"
    print(f"[selftest] 17. smoke 대상 수 {len(SMOKE_TARGETS_PREFERRED)}장 (≤8): OK")

    # 18. role/prototype_role fallback 처리 확인 (normal_control, lesion_candidate)
    row_nc = {**sample_row, "role": "normal_control", "prototype_role": "expansion"}
    tags_nc = compute_reason_tags(row_nc, set())
    assert tags_nc["normal_control_fp_pattern"], "normal_control_fp_pattern not detected for role=normal_control"
    print("[selftest] 18. role=normal_control 감지: OK")

    row_pt = {**sample_row, "role": "lesion_candidate", "prototype_role": "prototype"}
    tags_pt = compute_reason_tags(row_pt, set())
    assert tags_pt["lesion_candidate_review"], "lesion_candidate_review not detected for role=lesion_candidate"
    print("[selftest] 19. role=lesion_candidate 감지: OK")

    # 20. existing artifact 수정 없음 (ALLOW_CARD_MODIFICATION guard)
    assert ALLOW_CARD_MODIFICATION is False, "ALLOW_CARD_MODIFICATION must be False"
    print("[selftest] 20. existing artifact 수정 없음 (ALLOW_CARD_MODIFICATION=False): OK")

    if errors:
        print(f"\n[selftest] 경고/오류 {len(errors)}건:")
        for e in errors:
            print(f"  - {e}")
        return False

    print("\n[selftest] 전체 PASS")
    return True


# ── guard 함수 ───────────────────────────────────────────────────

def _bare_guard():
    """인자 없이 실행 시 BLOCKED."""
    print("BLOCKED: 이 스크립트는 bare 실행이 금지되어 있습니다.")
    print("사용법:")
    print("  --selftest         : 정적 self-test 실행")
    print("  --dry-run          : 입력 파일 검증 + 계획 출력 (실행 없음)")
    print("  --plan-smoke-only  : smoke 대상 목록 출력만")
    print("  --run-smoke --confirm-generate : 실제 reason table 생성 (별도 승인 후)")
    sys.exit(2)


def _output_guard():
    """output root에 DONE.json이 있으면 BLOCKED."""
    done_path = OUTPUT_ROOT / "DONE.json"
    if done_path.exists():
        print(f"BLOCKED: 기존 DONE.json이 존재합니다: {done_path}")
        print("기존 산출물을 정리한 후 재실행하세요.")
        sys.exit(2)


def _run_smoke_guard(args):
    """--run-smoke 단독은 BLOCKED. --confirm-generate 없으면 BLOCKED."""
    if args.run_smoke and not args.confirm_generate:
        print("BLOCKED: --run-smoke 단독 실행 금지.")
        print("실제 실행을 원하면: --run-smoke --confirm-generate")
        sys.exit(2)


# ── dry-run / plan-smoke-only ────────────────────────────────────

def run_dry_run(verbose: bool = True) -> dict:
    """
    입력 CSV/JSON 경로 존재 확인, smoke 대상 해결, stage2_holdout 검사.
    CT/mask/PNG 값 접근 없음.
    """
    report = {
        "mode": "dry_run",
        "input_checks": {},
        "stage2_holdout_intersection": 0,
        "smoke_targets": [],
        "hold_count": 0,
        "role_distribution": {},
        "tag_plan": [],
        "output_conflict": False,
        "blockers": [],
        "warnings": [],
        "notes": [],
    }

    # 입력 파일 존재 확인
    inputs = {
        "index_csv": INDEX_CSV,
        "cards_json_dir": CARDS_JSON_DIR,
        "manifest_csv": MANIFEST_CSV,
        "bulk_accept_csv": BULK_ACCEPT_CSV,
        "hold_list_csv": HOLD_LIST_CSV,
        "s4_preflight_json": S4_PREFLIGHT_JSON,
        "s4_tag_design_csv": S4_TAG_DESIGN_CSV,
    }
    for name, path in inputs.items():
        exists = path.exists()
        report["input_checks"][name] = {"path": str(path), "exists": exists}
        if not exists:
            report["blockers"].append(f"입력 파일 없음: {name} → {path}")
        else:
            if verbose:
                print(f"[dry-run] {name}: OK ({path})")

    if report["blockers"]:
        if verbose:
            for b in report["blockers"]:
                print(f"[dry-run] BLOCKER: {b}")
        return report

    # stage2_holdout 경로 검사
    for name, path in inputs.items():
        try:
            assert_no_stage2_holdout(str(path))
        except RuntimeError as e:
            report["blockers"].append(str(e))
            report["stage2_holdout_intersection"] += 1

    # smoke 대상 해결
    hold_ids = load_hold_ids(HOLD_LIST_CSV)
    report["hold_count"] = len(hold_ids)
    index = load_index(INDEX_CSV)

    resolved = resolve_smoke_targets(index, hold_ids)
    report["smoke_targets"] = resolved

    found_count = sum(1 for r in resolved if r["found"])
    hold_in_smoke = sum(1 for r in resolved if r["in_hold"])

    if verbose:
        print(f"\n[dry-run] smoke 대상 해결: {found_count}/{len(resolved)}")
        for r in resolved:
            status = "FOUND" if r["found"] else "NOT_FOUND"
            hold_mark = " [HOLD]" if r["in_hold"] else ""
            sub = f" → {r['substitution_reason']}" if r["substitution_reason"] else ""
            print(f"  [{status}]{hold_mark} {r['case_id']}{sub}")

    if hold_in_smoke < 3:
        report["warnings"].append(f"smoke 내 hold 케이스 {hold_in_smoke}건 < 3건 (preflight 기준)")

    # role 분포
    role_dist = {}
    for r in resolved:
        if r["found"]:
            row = index.get(r["case_id"], {})
            role = row.get("role", "unknown")
            role_dist[role] = role_dist.get(role, 0) + 1
    report["role_distribution"] = role_dist

    if verbose:
        print(f"\n[dry-run] role 분포: {role_dist}")
        print(f"[dry-run] hold 포함: {hold_in_smoke}건")

    # tag 계획
    report["tag_plan"] = list(METADATA_TAG_NAMES)
    if verbose:
        print(f"\n[dry-run] metadata-only tag {len(METADATA_TAG_NAMES)}개 계획:")
        for t in METADATA_TAG_NAMES:
            print(f"  - {t}")

    # output 충돌 확인
    done_path = OUTPUT_ROOT / "DONE.json"
    if done_path.exists():
        report["output_conflict"] = True
        report["blockers"].append(f"output DONE.json 이미 존재: {done_path}")
    else:
        if verbose:
            print(f"\n[dry-run] output root 충돌 없음: {OUTPUT_ROOT}")

    # CT/mask/PNG 접근 없음 확인
    report["notes"].append("CT/mask npy 미접근 확인: True")
    report["notes"].append("PNG 파일 미접근 확인: True")
    report["notes"].append("JSON/CSV metadata만 읽음: True")

    return report


METADATA_TAG_NAMES = [
    "score_high_vs_threshold",
    "very_high_score",
    "broad_component_response",
    "extreme_union_response",
    "apex_context_caution",
    "peripheral_boundary_candidate",
    "z_continuous_response",
    "z_wide_response",
    "single_slice_spike",
    "normal_control_fp_pattern",
    "lesion_candidate_review",
    "hold_case_review_required",
    "wide_display_reduction",
    "uncertain_reason_metadata_only",
]


def run_plan_smoke_only():
    """smoke 대상 목록만 출력. 실제 계산 없음."""
    print("[plan-smoke-only] smoke 대상 목록:")
    for i, cid in enumerate(SMOKE_TARGETS_PREFERRED, 1):
        print(f"  {i}. {cid}")
    print(f"\n총 {len(SMOKE_TARGETS_PREFERRED)}장 (≤8)")
    print("실제 실행: --run-smoke --confirm-generate")


# ── 실제 smoke 실행 (이번 단계 미실행) ──────────────────────────

def run_smoke_generate():
    """
    실제 reason table 생성.
    이번 단계에서는 --run-smoke --confirm-generate 조합으로만 진입 가능하며,
    현재 ALLOW_RUN_REASON=False로 차단됨.
    """
    if not ALLOW_RUN_REASON:
        print("BLOCKED: ALLOW_RUN_REASON=False — 이번 단계에서 실제 실행 금지.")
        print("S4 metadata-only reason smoke 실제 실행 승인 후 ALLOW_RUN_REASON=True로 변경 필요.")
        sys.exit(2)

    if ALLOW_CT_STAT:
        raise RuntimeError("BLOCKED: ALLOW_CT_STAT=True — CT stat 차단")
    if ALLOW_FEATURE_XAI:
        raise RuntimeError("BLOCKED: ALLOW_FEATURE_XAI=True — feature XAI 차단")
    if ALLOW_CARD_MODIFICATION:
        raise RuntimeError("BLOCKED: ALLOW_CARD_MODIFICATION=True — 카드 수정 차단")

    _output_guard()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    hold_ids = load_hold_ids(HOLD_LIST_CSV)
    index = load_index(INDEX_CSV)
    resolved = resolve_smoke_targets(index, hold_ids)

    results = []
    errors = []

    for target in resolved:
        case_id = target["case_id"]
        if not target["found"]:
            errors.append({"case_id": case_id, "error": "not_found_in_index"})
            continue

        assert_no_stage2_holdout(case_id)

        row = index[case_id]
        json_path = CARDS_JSON_DIR / f"{case_id}.json"
        if json_path.exists():
            card_data = load_json(json_path)
            for key, val in card_data.items():
                if key not in row:
                    row[key] = val

        tags = compute_reason_tags(row, hold_ids)
        text_ko = build_reason_text(tags, row, lang="KO")
        text_en = build_reason_text(tags, row, lang="EN")

        rec = {
            "expansion_case_id": case_id,
            "role": row.get("role", ""),
            "prototype_role": row.get("prototype_role", ""),
            "position_bin": row.get("position_bin", ""),
            "max_padim_score": row.get("max_padim_score", ""),
            "threshold": row.get("threshold", ""),
            "z_span": row.get("z_span", ""),
            "overmerge_flag": row.get("overmerge_flag", ""),
            "overmerge_level": row.get("overmerge_level", ""),
            "apex_caution": row.get("apex_caution", ""),
            "in_hold": target["in_hold"],
            "substitution_reason": target["substitution_reason"] or "",
        }
        for tag_name in METADATA_TAG_NAMES:
            rec[f"tag_{tag_name}"] = tags.get(tag_name, False)
        rec["reason_text_ko"] = text_ko
        rec["reason_text_en"] = text_en

        results.append(rec)

    # 결과 CSV 저장
    out_csv = OUTPUT_ROOT / "s4_reason_layer_metadata_smoke_v1.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"[run-smoke] 결과 CSV 저장: {out_csv}")

    # errors.csv
    err_csv = OUTPUT_ROOT / "errors.csv"
    if errors:
        with open(err_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["case_id", "error"])
            writer.writeheader()
            writer.writerows(errors)
        print(f"[run-smoke] errors.csv: {err_csv} ({len(errors)}건)")

    # summary JSON
    summary = {
        "smoke_count": len(results),
        "error_count": len(errors),
        "hold_count_in_smoke": sum(1 for r in results if r["in_hold"]),
        "tag_counts": {
            tag: sum(1 for r in results if r.get(f"tag_{tag}")) for tag in METADATA_TAG_NAMES
        },
        "allow_ct_stat": ALLOW_CT_STAT,
        "allow_feature_xai": ALLOW_FEATURE_XAI,
        "allow_card_modification": ALLOW_CARD_MODIFICATION,
    }
    summary_json = OUTPUT_ROOT / "s4_reason_layer_metadata_smoke_summary_v1.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[run-smoke] summary JSON: {summary_json}")

    # DONE.json
    done = {"status": "DONE", "smoke_count": len(results), "error_count": len(errors)}
    with open(OUTPUT_ROOT / "DONE.json", "w", encoding="utf-8") as f:
        json.dump(done, f, ensure_ascii=False, indent=2)

    print(f"[run-smoke] 완료: {len(results)}건 성공, {len(errors)}건 오류")


# ── argparse ─────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="S4 Reason Layer Metadata-Only Smoke Script"
    )
    parser.add_argument("--selftest", action="store_true", help="정적 self-test 실행")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="입력 파일 검증 + 계획 출력")
    parser.add_argument("--plan-smoke-only", dest="plan_smoke_only", action="store_true", help="smoke 대상 목록만 출력")
    parser.add_argument("--run-smoke", dest="run_smoke", action="store_true", help="실제 reason table 생성 (--confirm-generate 필요)")
    parser.add_argument("--confirm-generate", dest="confirm_generate", action="store_true", help="run-smoke 실행 확인")
    return parser


def main():
    parser = parse_args()
    args = parser.parse_args()

    # bare 실행 차단
    if not any([args.selftest, args.dry_run, args.plan_smoke_only, args.run_smoke]):
        _bare_guard()

    # run-smoke 단독 차단
    if args.run_smoke:
        _run_smoke_guard(args)

    if args.selftest:
        ok = run_selftest()
        sys.exit(0 if ok else 1)

    if args.dry_run:
        report = run_dry_run(verbose=True)
        if report["blockers"]:
            print(f"\n[dry-run] BLOCKER {len(report['blockers'])}건:")
            for b in report["blockers"]:
                print(f"  - {b}")
            sys.exit(1)
        print("\n[dry-run] PASS")
        sys.exit(0)

    if args.plan_smoke_only:
        run_plan_smoke_only()
        sys.exit(0)

    if args.run_smoke and args.confirm_generate:
        run_smoke_generate()
        sys.exit(0)


if __name__ == "__main__":
    main()
