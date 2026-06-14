"""
B1-D4g Rule-B3 Dev-Only Soft Penalty ×0.5 Extended Preview
============================================================
목적: B1-D4f preflight PASS 이후, Option-M 범위(B1-D2 candidate 30 + safety 26, dedup 54)에서
     R001/R015/R016/R028 4개에만 ×0.5 감점을 adjusted_score_preview로 생성.

금지 사항:
- real 실행 금지 (ALLOW_REAL_PROCESSING = False)
- original_score 수정 금지
- adjusted_score 실제 컬럼/파일 생성 금지
- threshold 재계산 금지
- stage2_holdout 접근 금지
- score CSV / mask / ROI 수정 금지
- suppression_weight / refined_score 생성 금지
- ×0.25 / exclude 실행 금지
- 기존 B1-D1~D4 산출물 수정 금지
"""

import sys
import json
import csv
from pathlib import Path

# ── 실행 차단 ─────────────────────────────────────────────────────────────────
ALLOW_REAL_PROCESSING = False
if not ALLOW_REAL_PROCESSING:
    # bare-run 감지: sys.argv[0] 가 직접 실행인 경우 exit(2)로 차단
    # dry-run 플래그 없이 실행하면 abort
    if "--dry-run" not in sys.argv and "--run" not in sys.argv:
        print("[BLOCKED] ALLOW_REAL_PROCESSING=False. 실행하려면 --run 플래그 필요.")
        print("  dry-run (로직 검증만): python <script> --dry-run")
        print("  실제 실행:            python <script> --run  (사용자 승인 후)")
        sys.exit(2)

DRY_RUN = "--dry-run" in sys.argv

# ── 경로 ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "b1d1_wall_mediastinum_fp_cause_diagnostic_v1"

# ── 입력 파일 ────────────────────────────────────────────────────────────────
INPUT_FILES = {
    "b1d4f_plan":       OUT_DIR / "b1d4f_rule_b3_soft_penalty_0_5_extended_preflight_plan.csv",
    "b1d4f_safety":     OUT_DIR / "b1d4f_rule_b3_soft_penalty_0_5_extended_safety_plan.csv",
    "b1d4f_summary":    OUT_DIR / "b1d4f_rule_b3_soft_penalty_0_5_extended_preflight_summary.json",
    "b1d4e_summary":    OUT_DIR / "b1d4e_rule_b3_penalty_preview_checkpoint_summary.json",
    "b1d4d_csv":        OUT_DIR / "b1d4d_rule_b3_soft_penalty_0_5_preview.csv",
    "b1d2_candidate":   OUT_DIR / "b1d2_candidate_groups_preview.csv",
    "b1d2_safety":      OUT_DIR / "b1d2_safety_set_preview.csv",
}

# ── 출력 파일 ────────────────────────────────────────────────────────────────
OUTPUT_CSV     = OUT_DIR / "b1d4g_rule_b3_soft_penalty_0_5_extended_preview.csv"
OUTPUT_SUMMARY = OUT_DIR / "b1d4g_rule_b3_soft_penalty_0_5_extended_preview_summary.json"
OUTPUT_REPORT  = OUT_DIR / "b1d4g_rule_b3_soft_penalty_0_5_extended_preview_report.md"

# ── 상수 ─────────────────────────────────────────────────────────────────────
PENALTY_ALPHA             = 0.5
PENALIZED_IDS             = {"R001", "R015", "R016", "R028"}
PROTECTED_HARD_CASES      = {"R018", "R024"}
PROTECTED_GATE_CANDIDATES = {"R002", "R005", "R006", "R009", "R012", "R030"}
UNREVIEWED_BOUNDARY_HOLD  = {"R020", "R021", "R025", "R027", "R029"}
DUPLICATE_IDS             = {"R018", "R024"}   # b1d2_safety_set_DUPLICATE 행

EXPECTED_PLAN_ROWS       = 56
EXPECTED_DEDUP_ROWS      = 54
EXPECTED_PENALTY_COUNT   = 4
EXPECTED_DUPLICATE_COUNT = 2

STAGE2_KEYWORDS = ["stage2", "holdout", "test_split"]

# ── output schema ─────────────────────────────────────────────────────────────
OUTPUT_FIELDNAMES = [
    "row_id", "review_id", "patient_id",
    "source_table", "source_group", "safety_type",
    "cause_class", "human_label", "visual_label",
    "original_score", "candidate_score",
    "rule_b3_extended_candidate",
    "penalty_alpha",
    "soft_penalty_applied",
    "adjusted_score_preview",
    "score_delta", "score_delta_percent",
    "protected_hard_case", "protected_lesion_safety",
    "protected_gate_candidate", "protected_observation",
    "protected_unreviewed_boundary",
    "duplicate_source_flag", "dedup_status",
    "fail_condition_hit", "fail_reason",
    "holdout_flag",
]


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def abort_if_stage2_path(path: Path):
    for kw in STAGE2_KEYWORDS:
        if kw in str(path).lower():
            print(f"[BLOCKED] stage2/holdout 경로 감지: {path}")
            sys.exit(2)


def abort_if_stage2_value(val: str, context: str = ""):
    for kw in STAGE2_KEYWORDS:
        if kw in val.lower():
            print(f"[BLOCKED] stage2/holdout 값 감지 ({context}): {val}")
            sys.exit(2)


def check_output_collision():
    for p in [OUTPUT_CSV, OUTPUT_SUMMARY, OUTPUT_REPORT]:
        if p.exists():
            print(f"[ABORT] 출력 파일이 이미 존재합니다: {p}")
            print("덮어쓰기 금지. 기존 파일을 보존합니다.")
            sys.exit(1)


def record_input_mtimes() -> dict:
    mtimes = {}
    for key, path in INPUT_FILES.items():
        abort_if_stage2_path(path)
        if not path.exists():
            print(f"[ERROR] 필수 입력 파일 없음: {path}")
            sys.exit(1)
        mtimes[key] = path.stat().st_mtime
    return mtimes


def verify_input_mtimes_unchanged(initial_mtimes: dict) -> bool:
    unchanged = True
    for key, path in INPUT_FILES.items():
        if path.stat().st_mtime != initial_mtimes[key]:
            print(f"[WARN] 입력 파일 mtime 변경 감지: {key} → {path}")
            unchanged = False
    return unchanged


def parse_float(val: str):
    try:
        return float(str(val).strip())
    except (ValueError, TypeError):
        return None


def parse_int(val: str, default: int = 0) -> int:
    try:
        return int(str(val).strip())
    except (ValueError, TypeError):
        return default


# ── 입력 로드 ────────────────────────────────────────────────────────────────

def load_b1d4f_summary() -> dict:
    with open(INPUT_FILES["b1d4f_summary"], "r", encoding="utf-8") as f:
        return json.load(f)


def validate_b1d4f_summary(s: dict):
    assert s.get("verdict") == "PASS", \
        f"B1-D4f verdict != PASS: {s.get('verdict')}"
    assert s.get("stage2_holdout_access", 1) == 0, \
        f"B1-D4f stage2_holdout_access != 0"
    assert s.get("recommended_extended_scope") == "Option-M", \
        f"recommended_extended_scope != Option-M: {s.get('recommended_extended_scope')}"
    assert float(s.get("expected_penalty_alpha", 0)) == 0.5, \
        f"expected_penalty_alpha != 0.5: {s.get('expected_penalty_alpha')}"
    assert set(s.get("expected_penalized_review_ids", [])) == PENALIZED_IDS, \
        f"expected_penalized_review_ids 불일치: {s.get('expected_penalized_review_ids')}"
    assert s.get("option_m_total_unique", 0) == EXPECTED_DEDUP_ROWS, \
        f"option_m_total_unique != 54: {s.get('option_m_total_unique')}"
    print("[OK] B1-D4f summary 검증 완료")


def load_plan_csv() -> list:
    rows = []
    with open(INPUT_FILES["b1d4f_plan"], "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            abort_if_stage2_value(row.get("patient_id", ""), "patient_id")
            rows.append(row)
    if len(rows) != EXPECTED_PLAN_ROWS:
        print(f"[ERROR] plan CSV 행 수 != {EXPECTED_PLAN_ROWS}: {len(rows)}")
        sys.exit(1)
    print(f"[OK] plan CSV 로드: {len(rows)}행")
    return rows


def load_visual_label_map() -> dict:
    """b1d2 candidate + safety에서 review_id → best_visual_label 매핑."""
    vl_map = {}
    for key in ("b1d2_candidate", "b1d2_safety"):
        with open(INPUT_FILES[key], "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rid = row.get("review_id", "")
                if rid and rid not in vl_map:
                    vl_map[rid] = row.get("best_visual_label", "")
    return vl_map


# ── dedup ────────────────────────────────────────────────────────────────────

def dedup_plan_rows(rows: list) -> tuple:
    """
    review_id 기준으로 중복 제거.
    source_table == 'b1d2_safety_set_DUPLICATE' 행(R018/R024 두 번째 출현)을 제거.
    반환: (deduped_rows, duplicate_dropped_rows)
    """
    seen = {}
    deduped = []
    dropped = []
    for row in rows:
        rid = row["review_id"]
        src = row.get("source_table", "")
        if rid in seen:
            # 이미 본 review_id → DUPLICATE
            dropped.append(row)
        else:
            seen[rid] = True
            deduped.append(row)

    # 검증: dropped 행은 전부 b1d2_safety_set_DUPLICATE여야 함
    for d in dropped:
        if d.get("source_table") != "b1d2_safety_set_DUPLICATE":
            print(f"[ERROR] 예상치 못한 DUPLICATE 행: {d['review_id']} / {d.get('source_table')}")
            sys.exit(1)
        if d["review_id"] not in DUPLICATE_IDS:
            print(f"[ERROR] DUPLICATE 처리된 review_id가 예상 외: {d['review_id']}")
            sys.exit(1)

    if len(deduped) != EXPECTED_DEDUP_ROWS:
        print(f"[ERROR] dedup 후 행 수 != {EXPECTED_DEDUP_ROWS}: {len(deduped)}")
        sys.exit(1)
    if len(dropped) != EXPECTED_DUPLICATE_COUNT:
        print(f"[ERROR] DUPLICATE 제거 행 수 != {EXPECTED_DUPLICATE_COUNT}: {len(dropped)}")
        sys.exit(1)

    duplicate_review_ids = [d["review_id"] for d in dropped]
    print(f"[OK] dedup 완료: {len(deduped)}행 유지, {len(dropped)}행 제거: {duplicate_review_ids}")
    return deduped, dropped


# ── 보호/보류 분류 ────────────────────────────────────────────────────────────

def classify_protection(row: dict) -> dict:
    """
    protection_status 컬럼과 review_id 기반으로 보호 플래그 결정.
    반환: {protected_hard_case, protected_lesion_safety, protected_gate_candidate,
           protected_observation, protected_unreviewed_boundary}
    """
    rid = row.get("review_id", "")
    ps  = row.get("protection_status", "")

    return {
        "protected_hard_case":       rid in PROTECTED_HARD_CASES or ps == "protected_hard_case",
        "protected_lesion_safety":   ps in ("protected_lesion_kept", "protected_lesion_risk_partial"),
        "protected_gate_candidate":  rid in PROTECTED_GATE_CANDIDATES or ps == "protected_gate_candidate",
        "protected_observation":     ps == "protected_obs_excluded",
        "protected_unreviewed_boundary": rid in UNREVIEWED_BOUNDARY_HOLD or ps == "unreviewed_hold",
    }


def is_ad_wall_med(row: dict) -> bool:
    """AD_wall_med_inside / Gate-P2 후보 여부."""
    bg = row.get("b1d2_group", "")
    cc = row.get("cause_class", "")
    return bg == "patchcore_gate_candidate" or cc == "AD_wall_med_inside"


def is_ad_other(row: dict) -> bool:
    """AD_other_inside 여부."""
    return row.get("cause_class", "") == "AD_other_inside"


# ── soft penalty 적용 ─────────────────────────────────────────────────────────

def apply_soft_penalty(row: dict, prot: dict, vl_map: dict, dropped_ids: set, row_counter: int) -> dict:
    """
    단일 행에 ×0.5 soft penalty 적용 로직.
    adjusted_score_preview만 생성 — adjusted_score 컬럼 생성 금지.
    """
    rid = row["review_id"]

    # holdout 검증
    hf = parse_int(row.get("holdout_flag", "0"))
    if hf != 0:
        print(f"[BLOCKED] holdout_flag != 0 (review_id={rid}, holdout_flag={hf})")
        sys.exit(2)

    # stage2 값 검사 (patient_id)
    abort_if_stage2_value(row.get("patient_id", ""), f"patient_id row={rid}")

    # AD_wall_med / AD_other 추가 보호
    ad_wall_med = is_ad_wall_med(row)
    ad_other    = is_ad_other(row)

    # 감점 적용 조건 (전부 true여야 함)
    should_penalize = (
        rid in PENALIZED_IDS
        and not prot["protected_hard_case"]
        and not prot["protected_lesion_safety"]
        and not prot["protected_gate_candidate"]
        and not prot["protected_observation"]
        and not prot["protected_unreviewed_boundary"]
        and not ad_wall_med
        and not ad_other
        and hf == 0
    )

    # 추가 안전망: PENALIZED_IDS 외 row가 감점되면 즉시 abort
    if should_penalize and rid not in PENALIZED_IDS:
        print(f"[BLOCKED] 예상 외 row 감점 시도: {rid}")
        sys.exit(2)

    # 보호 대상이 should_penalize=true로 분류되면 즉시 abort
    protected_any = (
        prot["protected_hard_case"] or prot["protected_lesion_safety"]
        or prot["protected_gate_candidate"] or prot["protected_observation"]
        or prot["protected_unreviewed_boundary"] or ad_wall_med or ad_other
    )
    if should_penalize and protected_any:
        print(f"[BLOCKED] 보호 대상 row 감점 시도: {rid}")
        sys.exit(2)

    # original_score / candidate_score (original_score 컬럼 우선)
    orig_score = parse_float(row.get("candidate_score", ""))
    if orig_score is None:
        print(f"[ERROR] score 파싱 실패: row={rid}")
        sys.exit(1)

    # adjusted_score_preview (내부 preview 전용 컬럼, adjusted_score 아님)
    if should_penalize:
        adj_score_preview = orig_score * PENALTY_ALPHA
        score_delta       = adj_score_preview - orig_score
        score_delta_pct   = (score_delta / orig_score * 100) if orig_score != 0 else 0.0
    else:
        adj_score_preview = orig_score
        score_delta       = 0.0
        score_delta_pct   = 0.0

    # fail_condition 검사
    fail_condition_hit = False
    fail_reason        = ""
    # 추가 이중 검증: PENALIZED_IDS 외에서 감점 발생 여부
    if should_penalize and rid not in PENALIZED_IDS:
        fail_condition_hit = True
        fail_reason        = f"예상 외 review_id 감점: {rid}"

    # visual_label 조회
    visual_label = vl_map.get(rid, "")

    # duplicate_source_flag: 원본 행이 b1d2_safety_set_DUPLICATE에서 왔으면 true
    # (dedup 후 남은 행이므로 이 플래그는 항상 False; dropped 행은 여기 오지 않음)
    dup_flag   = "false"
    dedup_stat = "first_occurrence" if rid not in dropped_ids else "duplicate_dropped"

    # rule_b3_extended_candidate: boundary_rule_candidate 행
    rule_b3_ext = str(row.get("b1d2_group", "") == "boundary_rule_candidate").lower()

    return {
        "row_id":                     row_counter,
        "review_id":                  rid,
        "patient_id":                 row.get("patient_id", ""),
        "source_table":               row.get("source_table", ""),
        "source_group":               row.get("b1d2_group", ""),
        "safety_type":                row.get("safety_role", ""),
        "cause_class":                row.get("cause_class", ""),
        "human_label":                row.get("human_label", ""),
        "visual_label":               visual_label,
        "original_score":             f"{orig_score:.6f}",
        "candidate_score":            f"{orig_score:.6f}",
        "rule_b3_extended_candidate": rule_b3_ext,
        "penalty_alpha":              str(PENALTY_ALPHA) if should_penalize else "",
        "soft_penalty_applied":       str(should_penalize).lower(),
        "adjusted_score_preview":     f"{adj_score_preview:.6f}",
        "score_delta":                f"{score_delta:.6f}",
        "score_delta_percent":        f"{score_delta_pct:.4f}",
        "protected_hard_case":        str(prot["protected_hard_case"]).lower(),
        "protected_lesion_safety":    str(prot["protected_lesion_safety"]).lower(),
        "protected_gate_candidate":   str(prot["protected_gate_candidate"]).lower(),
        "protected_observation":      str(prot["protected_observation"]).lower(),
        "protected_unreviewed_boundary": str(prot["protected_unreviewed_boundary"]).lower(),
        "duplicate_source_flag":      dup_flag,
        "dedup_status":               dedup_stat,
        "fail_condition_hit":         str(fail_condition_hit).lower(),
        "fail_reason":                fail_reason,
        "holdout_flag":               str(hf),
    }


# ── safety 검증 ───────────────────────────────────────────────────────────────

def run_safety_checks(output_rows: list) -> list:
    """
    전체 output 행에 대해 safety 조건을 검증.
    위반 즉시 fail_conditions에 추가.
    """
    fails = []

    # 행 수 검증
    if len(output_rows) != EXPECTED_DEDUP_ROWS:
        fails.append(f"output rows != {EXPECTED_DEDUP_ROWS}: {len(output_rows)}")

    # soft_penalty_applied 검증
    penalized     = [r for r in output_rows if r["soft_penalty_applied"] == "true"]
    penalized_ids = {r["review_id"] for r in penalized}

    if len(penalized) != EXPECTED_PENALTY_COUNT:
        fails.append(f"soft_penalty_applied count != {EXPECTED_PENALTY_COUNT}: {len(penalized)}")
    if penalized_ids != PENALIZED_IDS:
        fails.append(f"penalized_review_ids 불일치: {penalized_ids}")

    # 보호 대상 감점 여부
    for r in output_rows:
        rid = r["review_id"]
        if r["soft_penalty_applied"] == "true" and rid not in PENALIZED_IDS:
            fails.append(f"PENALIZED_IDS 외 감점: {rid}")
        if r["soft_penalty_applied"] == "true" and rid in PROTECTED_HARD_CASES:
            fails.append(f"protected_hard_case 감점: {rid}")
        if r["soft_penalty_applied"] == "true" and r["protected_lesion_safety"] == "true":
            fails.append(f"protected_lesion_safety 감점: {rid}")
        if r["soft_penalty_applied"] == "true" and r["protected_gate_candidate"] == "true":
            fails.append(f"protected_gate_candidate 감점: {rid}")
        if r["soft_penalty_applied"] == "true" and r["protected_observation"] == "true":
            fails.append(f"protected_observation 감점: {rid}")
        if r["soft_penalty_applied"] == "true" and r["protected_unreviewed_boundary"] == "true":
            fails.append(f"protected_unreviewed_boundary 감점: {rid}")
        if r["soft_penalty_applied"] == "true" and rid in UNREVIEWED_BOUNDARY_HOLD:
            fails.append(f"unreviewed_boundary_hold 감점: {rid}")
        if r["soft_penalty_applied"] == "true" and rid in PROTECTED_GATE_CANDIDATES:
            fails.append(f"gate_candidate 감점 (by id): {rid}")

    # holdout_flag != 0 행 없음 확인
    for r in output_rows:
        if parse_int(r.get("holdout_flag", "0")) != 0:
            fails.append(f"holdout_flag != 0: {r['review_id']}")

    # adjusted_score_preview가 아닌 adjusted_score 컬럼 없음 확인
    if output_rows:
        if "adjusted_score" in output_rows[0]:
            fails.append("adjusted_score 컬럼 생성됨 (금지)")

    return fails


# ── rank 계산 ─────────────────────────────────────────────────────────────────

def compute_rank_changes(output_rows: list) -> tuple:
    """
    54행 내부 기준 original_score / adjusted_score_preview rank 계산.
    반환: (rows_with_rank, rank_summary_for_penalized)
    """
    # rank: 내림차순 (높은 score = rank 1)
    orig_sorted  = sorted(output_rows, key=lambda r: -parse_float(r["original_score"]))
    adj_sorted   = sorted(output_rows, key=lambda r: -parse_float(r["adjusted_score_preview"]))

    orig_rank = {r["review_id"]: i + 1 for i, r in enumerate(orig_sorted)}
    adj_rank  = {r["review_id"]: i + 1 for i, r in enumerate(adj_sorted)}

    for r in output_rows:
        r["original_rank_54"]  = orig_rank[r["review_id"]]
        r["adjusted_rank_54"]  = adj_rank[r["review_id"]]
        r["rank_change"]       = orig_rank[r["review_id"]] - adj_rank[r["review_id"]]

    rank_summary = []
    for r in output_rows:
        if r["soft_penalty_applied"] == "true":
            rank_summary.append({
                "review_id":               r["review_id"],
                "original_score":          r["original_score"],
                "adjusted_score_preview":  r["adjusted_score_preview"],
                "score_delta":             r["score_delta"],
                "original_rank_54":        r["original_rank_54"],
                "adjusted_rank_54":        r["adjusted_rank_54"],
                "rank_change":             r["rank_change"],
            })
    return output_rows, rank_summary


# ── report 작성 ───────────────────────────────────────────────────────────────

def write_report(summary: dict, rank_summary: list, fail_conditions: list):
    verdict = "PASS" if summary["fail_count"] == 0 else "FAIL"
    lines = [
        "# B1-D4g Rule-B3 ×0.5 Extended Dev-Only Preview Report",
        "",
        f"**판정: {verdict}**",
        "",
        "## B1-D4f Preflight 요약",
        "- verdict: PASS",
        "- recommended_extended_scope: Option-M",
        "- expected_penalty_alpha: 0.5",
        "- expected_penalized_ids: R001/R015/R016/R028",
        "- option_m_total_unique: 54",
        "",
        "## Option-M 범위",
        "- B1-D2 candidate_groups 30행 + B1-D2 safety_set 26행",
        "- 중복 review_id(R018/R024)는 b1d2_safety_set_DUPLICATE 행을 제거",
        "- dedup 후 54개 고유 review_id",
        "",
        "## Dedup 처리",
        f"- plan input rows: {summary['input_plan_rows']}",
        f"- dedup output rows: {summary['dedup_output_rows']}",
        f"- duplicate_review_ids: {summary['duplicate_review_ids']}",
        f"- duplicate_rows_handled: {summary['duplicate_rows_handled']}",
        "- 처리 기준: source_table == 'b1d2_safety_set_DUPLICATE' 행을 DUPLICATE로 처리.",
        "  첫 출현(b1d2_candidate_groups)을 유지하고 두 번째 출현을 제거.",
        "",
        "## 감점된 4개 후보",
        "| review_id | original_score | adjusted_score_preview | score_delta | orig_rank | adj_rank | rank_change |",
        "|-----------|----------------|------------------------|-------------|-----------|----------|-------------|",
    ]
    for rs in rank_summary:
        lines.append(
            f"| {rs['review_id']} | {rs['original_score']} | {rs['adjusted_score_preview']} "
            f"| {rs['score_delta']} | {rs['original_rank_54']} | {rs['adjusted_rank_54']} "
            f"| {rs['rank_change']} |"
        )
    lines += [
        "",
        "## 보호/보류 대상 감점 없음 확인",
        f"- protected_hard_case (R018/R024) penalty count: {summary['protected_hard_case_penalty_count']} (must=0)",
        f"- protected_lesion_safety penalty count: {summary['protected_lesion_safety_penalty_count']} (must=0)",
        f"- protected_gate_candidate penalty count: {summary['protected_gate_candidate_penalty_count']} (must=0)",
        f"- protected_observation penalty count: {summary['protected_observation_penalty_count']} (must=0)",
        f"- protected_unreviewed_boundary penalty count: {summary['protected_unreviewed_boundary_penalty_count']} (must=0)",
        f"- protected_ad_wall_med penalty count: {summary['protected_ad_wall_med_penalty_count']} (must=0)",
        f"- protected_ad_other penalty count: {summary['protected_ad_other_penalty_count']} (must=0)",
        "",
        "## Score 원본 무수정 확인",
        f"- original_score_modified: {summary['original_score_modified']} (must=false)",
        f"- threshold_recomputed: {summary['threshold_recomputed']} (must=false)",
        f"- adjusted_score_preview_created: {summary['adjusted_score_preview_created']} (preview csv 내부 컬럼)",
        f"- adjusted_score_created: {summary['adjusted_score_created']} (must=false)",
        f"- suppression_weight_created: {summary['suppression_weight_created']} (must=false)",
        f"- refined_score_created: {summary['refined_score_created']} (must=false)",
        "",
        "## 54행 내부 Rank 변화 요약",
        f"- {summary['rank_change_summary']}",
        "",
        "## 한계",
        "- 이 결과는 54행 Option-M preview 내부 rank 변화일 뿐이다.",
        "- 전체 dev metric(FROC/AUROC)이 아니다.",
        "- threshold 재계산을 포함하지 않는다.",
        "- stage2_holdout 데이터를 사용하지 않는다.",
        "- adjusted_score_preview는 이 CSV 내부 컬럼이며, 실제 score CSV에 반영되지 않는다.",
        "",
        "## 다음 단계",
    ]
    if verdict == "PASS":
        lines.append("- **B1-D4h Rule-B3 extended preview checkpoint**")
    else:
        lines.append("- FAIL: Rule-B3 조건 재설계 필요")
        for fc in fail_conditions:
            lines.append(f"  - {fc}")

    if not DRY_RUN:
        OUTPUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
        print(f"[OK] report 저장: {OUTPUT_REPORT}")
    else:
        print("[DRY-RUN] report 내용 미저장 (dry-run 모드)")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"[START] B1-D4g {'(DRY-RUN)' if DRY_RUN else ''}")

    # 1. 출력 파일 충돌 확인
    if not DRY_RUN:
        check_output_collision()

    # 2. 입력 파일 mtime 기록
    initial_mtimes = record_input_mtimes()

    # 3. B1-D4f summary 검증
    b1d4f_sum = load_b1d4f_summary()
    validate_b1d4f_summary(b1d4f_sum)

    # 4. plan CSV 로드
    plan_rows = load_plan_csv()

    # 5. visual_label 매핑 로드
    vl_map = load_visual_label_map()

    # 6. dedup
    deduped_rows, dropped_rows = dedup_plan_rows(plan_rows)
    dropped_ids = {r["review_id"] for r in dropped_rows}

    # 7. soft penalty 적용
    output_rows = []
    for i, row in enumerate(deduped_rows, start=1):
        prot = classify_protection(row)
        out  = apply_soft_penalty(row, prot, vl_map, dropped_ids, i)
        output_rows.append(out)

    # 8. rank 계산
    output_rows, rank_summary = compute_rank_changes(output_rows)

    # 9. safety 검증
    fail_conditions = run_safety_checks(output_rows)

    # 10. 추가 보호 카운트 집계
    def count_penalty_for(cond_fn):
        return sum(1 for r in output_rows if r["soft_penalty_applied"] == "true" and cond_fn(r))

    p_hard     = count_penalty_for(lambda r: r["protected_hard_case"] == "true")
    p_lesion   = count_penalty_for(lambda r: r["protected_lesion_safety"] == "true")
    p_gate     = count_penalty_for(lambda r: r["protected_gate_candidate"] == "true")
    p_obs      = count_penalty_for(lambda r: r["protected_observation"] == "true")
    p_unrev    = count_penalty_for(lambda r: r["protected_unreviewed_boundary"] == "true")
    p_adwall   = count_penalty_for(lambda r: r["source_group"] == "patchcore_gate_candidate")
    p_adother  = count_penalty_for(lambda r: r["cause_class"] == "AD_other_inside")

    for cnt, label in [
        (p_hard, "protected_hard_case"), (p_lesion, "protected_lesion_safety"),
        (p_gate, "protected_gate_candidate"), (p_obs, "protected_observation"),
        (p_unrev, "protected_unreviewed_boundary"), (p_adwall, "ad_wall_med"),
        (p_adother, "ad_other"),
    ]:
        if cnt > 0:
            fail_conditions.append(f"{label} 감점 발생: {cnt}건")

    # 11. rank_change_summary 문자열
    rc_desc = ", ".join(
        f"{rs['review_id']} {rs['original_rank_54']}→{rs['adjusted_rank_54']}(Δ{rs['rank_change']})"
        for rs in rank_summary
    ) if rank_summary else "없음"

    # 12. summary
    penalized_out = [r["review_id"] for r in output_rows if r["soft_penalty_applied"] == "true"]
    summary = {
        "stage":                          "B1-D4g",
        "dry_run":                        DRY_RUN,
        "stage2_holdout_access":          0,
        "input_plan_rows":                len(plan_rows),
        "dedup_output_rows":              len(output_rows),
        "duplicate_review_ids":           list(dropped_ids),
        "duplicate_rows_handled":         len(dropped_rows),
        "penalty_alpha":                  PENALTY_ALPHA,
        "soft_penalty_applied_count":     len(penalized_out),
        "penalized_review_ids":           sorted(penalized_out),
        "protected_hard_case_penalty_count":          p_hard,
        "protected_lesion_safety_penalty_count":      p_lesion,
        "protected_gate_candidate_penalty_count":     p_gate,
        "protected_observation_penalty_count":        p_obs,
        "protected_unreviewed_boundary_penalty_count": p_unrev,
        "protected_ad_wall_med_penalty_count":        p_adwall,
        "protected_ad_other_penalty_count":           p_adother,
        "original_score_modified":        False,
        "threshold_recomputed":           False,
        "adjusted_score_preview_created": True,
        "adjusted_score_created":         False,
        "suppression_weight_created":     False,
        "refined_score_created":          False,
        "fail_count":                     len(fail_conditions),
        "fail_reasons":                   fail_conditions,
        "rank_change_summary":            rc_desc,
        "verdict":                        "PASS" if len(fail_conditions) == 0 else "FAIL",
        "next_recommended_step":          "B1-D4h Rule-B3 extended preview checkpoint"
                                          if len(fail_conditions) == 0 else "Rule-B3 조건 재설계",
    }

    # 13. 입력 mtime 무변경 확인
    if not verify_input_mtimes_unchanged(initial_mtimes):
        fail_conditions.append("입력 파일 mtime 변경 감지")
        summary["fail_count"] += 1
        summary["verdict"] = "FAIL"

    # 14. 결과 출력 / 저장
    verdict = summary["verdict"]
    print(f"\n[RESULT] verdict={verdict}, penalized={penalized_out}, "
          f"output_rows={len(output_rows)}, fail_count={len(fail_conditions)}")
    if fail_conditions:
        for fc in fail_conditions:
            print(f"  [FAIL] {fc}")

    if DRY_RUN:
        print("[DRY-RUN] 출력 파일 미저장.")
        return

    # 출력 CSV (rank 컬럼은 summary용이므로 OUTPUT_FIELDNAMES에 추가 포함)
    csv_fields = OUTPUT_FIELDNAMES + ["original_rank_54", "adjusted_rank_54", "rank_change"]
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"[OK] CSV 저장: {OUTPUT_CSV}")

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] summary 저장: {OUTPUT_SUMMARY}")

    write_report(summary, rank_summary, fail_conditions)

    print(f"\n[DONE] B1-D4g {verdict}")


if __name__ == "__main__":
    main()
