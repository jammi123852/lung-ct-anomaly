"""
B1-D4c Rule-B3 Dev-Only Soft Penalty ×0.75 Preview
=====================================================
목적: B1-D4b flag_only PASS 이후, flagged 4개(R001/R015/R016/R028)에만
      ×0.75 감점을 preview로 적용한 adjusted_score_preview 컬럼을 생성.
- original_score 무수정
- adjusted_score (기존 score CSV 반영용 컬럼명) 생성 금지
- threshold 재계산 금지
- stage2_holdout 접근 금지
- suppression_weight / refined_score 생성 금지
- ×0.5 / ×0.25 계산 금지
"""

import sys
import json
import csv
from pathlib import Path

# ── 프로젝트 루트 ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "b1d1_wall_mediastinum_fp_cause_diagnostic_v1"

# ── 입력 파일 ─────────────────────────────────────────────────────────────
INPUT_FILES = {
    "b1d4b_csv":         OUT_DIR / "b1d4b_rule_b3_flag_only_preview.csv",
    "b1d4b_summary":     OUT_DIR / "b1d4b_rule_b3_flag_only_preview_summary.json",
    "adjustment_plan":   OUT_DIR / "b1d4_rule_b3_adjustment_plan_preview.csv",
    "safety_sentinel":   OUT_DIR / "b1d4_rule_b3_safety_sentinel_plan.csv",
    "dry_smoke_results": OUT_DIR / "b1d3b_rule_b3_dry_smoke_results.csv",
}

# ── 출력 파일 ─────────────────────────────────────────────────────────────
OUTPUT_CSV     = OUT_DIR / "b1d4c_rule_b3_soft_penalty_0_75_preview.csv"
OUTPUT_SUMMARY = OUT_DIR / "b1d4c_rule_b3_soft_penalty_0_75_preview_summary.json"
OUTPUT_REPORT  = OUT_DIR / "b1d4c_rule_b3_soft_penalty_0_75_preview_report.md"

# ── 상수 ──────────────────────────────────────────────────────────────────
PENALTY_ALPHA          = 0.75
FLAG_CANDIDATES        = {"R001", "R015", "R016", "R028"}
PROTECTED_HARD_CASES   = {"R018", "R024"}
PROTECTED_GATE_CANDIDATES = {"R002", "R005", "R006", "R009", "R012", "R030"}
OBSERVATION_OTHER      = {"R003", "R008"}
ALL_LESION_SAFETY_TYPES = {"lesion_kept", "lesion_risk_partial"}

STAGE2_KEYWORDS = ["stage2", "holdout", "test_split"]


# ── 유틸 ──────────────────────────────────────────────────────────────────

def abort_if_stage2(val: str):
    for kw in STAGE2_KEYWORDS:
        if kw in val.lower():
            print(f"[BLOCKED] stage2/holdout 접근 감지: {val}")
            sys.exit(2)


def check_output_exists():
    for p in [OUTPUT_CSV, OUTPUT_SUMMARY, OUTPUT_REPORT]:
        if p.exists():
            print(f"[ABORT] 출력 파일이 이미 존재합니다: {p}")
            print("기존 파일 보존. 재실행이 필요하면 새 output version을 사용하세요.")
            sys.exit(1)


def record_input_mtimes() -> dict:
    mtimes = {}
    for key, path in INPUT_FILES.items():
        abort_if_stage2(str(path))
        if not path.exists():
            print(f"[ERROR] 필수 입력 파일 없음: {path}")
            sys.exit(1)
        mtimes[key] = path.stat().st_mtime
    return mtimes


def verify_input_mtimes_unchanged(initial_mtimes: dict) -> bool:
    unchanged = True
    for key, path in INPUT_FILES.items():
        if path.stat().st_mtime != initial_mtimes[key]:
            print(f"[WARN] 입력 파일 mtime 변경 감지: {path}")
            unchanged = False
    return unchanged


def load_b1d4b_summary() -> dict:
    with open(INPUT_FILES["b1d4b_summary"], "r", encoding="utf-8") as f:
        return json.load(f)


def validate_b1d4b_summary(s: dict):
    assert s.get("verdict") == "PASS", \
        f"B1-D4b verdict != PASS: {s.get('verdict')}"
    assert s.get("stage2_holdout_access", 1) == 0, \
        f"B1-D4b stage2_holdout_access != 0: {s.get('stage2_holdout_access')}"
    assert s.get("output_rows", 0) == 26, \
        f"B1-D4b output_rows != 26: {s.get('output_rows')}"
    assert s.get("flag_true_count", 0) == 4, \
        f"B1-D4b flag_true_count != 4: {s.get('flag_true_count')}"
    assert set(s.get("flagged_review_ids", [])) == FLAG_CANDIDATES, \
        f"B1-D4b flagged_review_ids 불일치: {s.get('flagged_review_ids')}"
    for cnt_key in [
        "protected_hard_case_flag_count",
        "protected_lesion_safety_flag_count",
        "protected_gate_candidate_flag_count",
        "protected_observation_flag_count",
    ]:
        assert s.get(cnt_key, 1) == 0, \
            f"B1-D4b {cnt_key} != 0: {s.get(cnt_key)}"
    print("[OK] B1-D4b summary 검증 완료")


def abort_if_stage2_value(val: str, context: str = ""):
    """값(경로·patient_id 등)에 stage2/holdout 키워드가 있으면 중단.
    컬럼명 자체는 체크 대상이 아님 — 파일 경로와 값만 체크."""
    for kw in STAGE2_KEYWORDS:
        if kw in val.lower():
            print(f"[BLOCKED] stage2/holdout 값 감지 ({context}): {val}")
            sys.exit(2)


def load_b1d4b_csv() -> list[dict]:
    rows = []
    with open(INPUT_FILES["b1d4b_csv"], "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # patient_id, stage_split_check 값만 체크 (컬럼명 자체는 제외)
            abort_if_stage2_value(row.get("patient_id", ""), "patient_id")
            abort_if_stage2_value(row.get("stage_split_check", ""), "stage_split_check")
            rows.append(row)
    assert len(rows) == 26, f"B1-D4b CSV 행 수 != 26: {len(rows)}"
    return rows


def parse_bool(val: str) -> bool:
    return str(val).strip().lower() in ("true", "1", "yes")


def parse_score(val: str) -> float | None:
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def apply_soft_penalty(row: dict) -> dict:
    """
    flag_only == true이고 보호 대상 아닌 행에만 ×0.75 적용.
    adjusted_score_preview = original_score × 0.75 (flagged)
    adjusted_score_preview = original_score         (나머지)
    """
    rid = row.get("review_id", "")
    flag_only = parse_bool(row.get("rule_b3_flag_only", "false"))
    protected_hard   = parse_bool(row.get("protected_hard_case", "false"))
    protected_lesion = parse_bool(row.get("protected_lesion_safety", "false"))
    protected_gate   = parse_bool(row.get("protected_gate_candidate", "false"))
    protected_obs    = parse_bool(row.get("protected_observation", "false"))

    orig_score = parse_score(row.get("original_score", ""))
    if orig_score is None:
        orig_score = parse_score(row.get("candidate_score", ""))

    # holdout 검증
    hf_raw = row.get("holdout_flag", "0").strip()
    try:
        hf = int(hf_raw) if hf_raw else 0
    except ValueError:
        hf = -1
    if hf != 0:
        print(f"[BLOCKED] holdout_flag != 0 (review_id={rid}, holdout_flag={hf_raw!r})")
        sys.exit(2)

    apply = (
        flag_only
        and rid in FLAG_CANDIDATES
        and not protected_hard
        and not protected_lesion
        and not protected_gate
        and not protected_obs
        and hf == 0
    )

    if orig_score is not None:
        adj_score = round(orig_score * PENALTY_ALPHA, 6) if apply else orig_score
        delta = round(adj_score - orig_score, 6) if apply else 0.0
        delta_pct = round((delta / orig_score) * 100, 4) if (apply and orig_score != 0) else 0.0
    else:
        adj_score = None
        delta = None
        delta_pct = None

    return {
        "row_id":                   row.get("row_id", ""),
        "review_id":                rid,
        "patient_id":               row.get("patient_id", ""),
        "source_group":             row.get("source_group", ""),
        "safety_type":              row.get("safety_type", ""),
        "safety_role":              row.get("safety_role", ""),
        "human_label":              row.get("human_label", ""),
        "original_score":           orig_score,
        "candidate_score":          row.get("candidate_score", ""),
        "rule_b3_flag_only":        flag_only,
        "penalty_alpha":            PENALTY_ALPHA if apply else "",
        "soft_penalty_applied":     apply,
        "adjusted_score_preview":   adj_score,   # preview 전용 — 기존 score CSV 반영 안 함
        "score_delta":              delta,
        "score_delta_percent":      delta_pct,
        "protected_hard_case":      protected_hard,
        "protected_lesion_safety":  protected_lesion,
        "protected_gate_candidate": protected_gate,
        "protected_observation":    protected_obs,
        "fail_condition_hit":       False,
        "fail_reason":              "",
        "stage_split_check":        "stage1_dev_only",
        "holdout_flag":             hf,
    }


def compute_fail_conditions(output_rows: list[dict]) -> tuple[int, list[str]]:
    fails = []
    applied_ids = [r["review_id"] for r in output_rows if r["soft_penalty_applied"]]

    if len(applied_ids) != 4:
        fails.append(f"soft_penalty_applied_count != 4: {len(applied_ids)}")
    if set(applied_ids) != FLAG_CANDIDATES:
        fails.append(f"penalized_review_ids 불일치: {sorted(applied_ids)} != {sorted(FLAG_CANDIDATES)}")

    for r in output_rows:
        rid = r["review_id"]
        applied = r["soft_penalty_applied"]
        if applied and rid not in FLAG_CANDIDATES:
            fails.append(f"flag 대상 외 감점: {rid}")
        if rid in PROTECTED_HARD_CASES and applied:
            fails.append(f"protected_hard_case 감점: {rid}")
        st = str(r.get("safety_type", "")).strip().lower()
        if st in ALL_LESION_SAFETY_TYPES and applied:
            fails.append(f"lesion_safety 감점: {rid} ({st})")
        if rid in PROTECTED_GATE_CANDIDATES and applied:
            fails.append(f"gate_candidate 감점: {rid}")
        sg = str(r.get("source_group", "")).strip().lower()
        if (sg == "observation_other" or rid in OBSERVATION_OTHER) and applied:
            fails.append(f"observation_other 감점: {rid}")
        # original_score 변경 검사는 적용 행이 아닌 경우만 (비적용 행의 adjusted == original)
        if not applied:
            orig = r["original_score"]
            adj  = r["adjusted_score_preview"]
            if orig is not None and adj is not None and round(float(orig), 8) != round(float(adj), 8):
                fails.append(f"비적용 행 score 변경: {rid} orig={orig} adj={adj}")

    # 각 행에 fail 기록
    fail_set = set(fails)
    for r in output_rows:
        if any(r["review_id"] in f for f in fail_set):
            r["fail_condition_hit"] = True
            r["fail_reason"] = "; ".join(f for f in fail_set if r["review_id"] in f)

    return len(fails), fails


def compute_rank_summary(output_rows: list[dict]) -> list[dict]:
    """26행 내부 rank (original_score 기준 vs adjusted_score_preview 기준)."""
    valid = [r for r in output_rows if r["original_score"] is not None]

    orig_ranked = sorted(valid, key=lambda r: float(r["original_score"]), reverse=True)
    adj_ranked  = sorted(valid, key=lambda r: float(r["adjusted_score_preview"]), reverse=True)

    orig_rank = {r["review_id"]: i + 1 for i, r in enumerate(orig_ranked)}
    adj_rank  = {r["review_id"]: i + 1 for i, r in enumerate(adj_ranked)}

    summary = []
    for r in output_rows:
        if not r["soft_penalty_applied"]:
            continue
        rid = r["review_id"]
        or_ = orig_rank.get(rid, "?")
        ar_ = adj_rank.get(rid, "?")
        rc  = (or_ - ar_) if isinstance(or_, int) and isinstance(ar_, int) else "?"
        summary.append({
            "review_id":               rid,
            "original_score":          r["original_score"],
            "adjusted_score_preview":  r["adjusted_score_preview"],
            "score_delta":             r["score_delta"],
            "score_delta_percent":     r["score_delta_percent"],
            "original_rank_within_26": or_,
            "adjusted_rank_within_26": ar_,
            "rank_change":             rc,
        })
    return summary


def write_output_csv(output_rows: list[dict]):
    fieldnames = [
        "row_id", "review_id", "patient_id", "source_group", "safety_type",
        "safety_role", "human_label", "original_score", "candidate_score",
        "rule_b3_flag_only", "penalty_alpha", "soft_penalty_applied",
        "adjusted_score_preview", "score_delta", "score_delta_percent",
        "protected_hard_case", "protected_lesion_safety",
        "protected_gate_candidate", "protected_observation",
        "fail_condition_hit", "fail_reason", "stage_split_check", "holdout_flag",
    ]
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"[OK] CSV 생성: {OUTPUT_CSV}")


def write_summary_json(output_rows: list[dict], fail_count: int, fail_reasons: list[str],
                       rank_summary: list[dict], mtime_unchanged: bool):
    applied_ids = sorted(r["review_id"] for r in output_rows if r["soft_penalty_applied"])
    hard_penalty   = sum(1 for r in output_rows if r["protected_hard_case"] and r["soft_penalty_applied"])
    lesion_penalty = sum(1 for r in output_rows if r["protected_lesion_safety"] and r["soft_penalty_applied"])
    gate_penalty   = sum(1 for r in output_rows if r["protected_gate_candidate"] and r["soft_penalty_applied"])
    obs_penalty    = sum(1 for r in output_rows if r["protected_observation"] and r["soft_penalty_applied"])

    verdict = "PASS" if (
        fail_count == 0
        and len(applied_ids) == 4
        and set(applied_ids) == FLAG_CANDIDATES
        and hard_penalty == 0
        and lesion_penalty == 0
        and gate_penalty == 0
        and obs_penalty == 0
    ) else "FAIL"

    summary = {
        "step": "B1-D4c_Rule_B3_soft_penalty_0_75_preview",
        "verdict": verdict,
        "stage2_holdout_access": 0,
        "input_rows": len(output_rows),
        "output_rows": len(output_rows),
        "penalty_alpha": PENALTY_ALPHA,
        "soft_penalty_applied_count": len(applied_ids),
        "penalized_review_ids": applied_ids,
        "expected_penalized_ids": sorted(FLAG_CANDIDATES),
        "protected_hard_case_penalty_count": hard_penalty,
        "protected_lesion_safety_penalty_count": lesion_penalty,
        "protected_gate_candidate_penalty_count": gate_penalty,
        "protected_observation_penalty_count": obs_penalty,
        "original_score_modified": False,
        "threshold_recomputed": False,
        "adjusted_score_preview_created": True,
        "adjusted_score_created": False,          # 기존 score CSV 반영 없음
        "suppression_weight_created": False,
        "refined_score_created": False,
        "fail_count": fail_count,
        "fail_reasons": fail_reasons,
        "rank_change_summary": rank_summary,
        "input_mtime_unchanged": mtime_unchanged,
        "next_recommended_step": (
            "B1-D4d Rule-B3 soft penalty ×0.5 preview 또는 B1-D4e preview checkpoint"
            if verdict == "PASS"
            else "fail_reasons 확인 후 수정"
        ),
    }

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] Summary JSON 생성: {OUTPUT_SUMMARY}")
    return summary


def write_report_md(output_rows: list[dict], summary: dict, rank_summary: list[dict]):
    verdict = summary["verdict"]
    lines = [
        "# B1-D4c Rule-B3 Soft Penalty ×0.75 Preview Report",
        "",
        f"**판정: {verdict}**",
        "",
        "## B1-D4b Flag-Only PASS 요약",
        "",
        "B1-D4b flag_only preview PASS 완료."
        " flagged 4개(R001/R015/R016/R028), 보호 대상 flag 0개, score 무수정.",
        "",
        "## ×0.75 Preview 목적",
        "",
        "score를 변경하지 않고 `adjusted_score_preview` 컬럼만 생성하여,"
        " ×0.75 감점 시 candidate-level score 변화를 dev-only로 확인한다."
        " 전체 PaDiM score CSV 및 threshold 미변경.",
        "",
        "## 감점된 4개 후보",
        "",
        "| review_id | original_score | adjusted_score_preview | score_delta | score_delta_% |",
        "|-----------|---------------|------------------------|-------------|---------------|",
    ]
    for r in rank_summary:
        lines.append(
            f"| {r['review_id']} | {r['original_score']:.4f}"
            f" | {r['adjusted_score_preview']:.4f}"
            f" | {r['score_delta']:.4f}"
            f" | {r['score_delta_percent']:.2f}% |"
        )

    lines += [
        "",
        "## 보호 대상 감점 없음 확인",
        "",
        "| 보호 카테고리 | 감점 발생 수 | 결과 |",
        "|--------------|-------------|------|",
        f"| boundary_hard_case (R018/R024) | {summary['protected_hard_case_penalty_count']} | {'PASS' if summary['protected_hard_case_penalty_count']==0 else 'FAIL'} |",
        f"| lesion_kept / lesion_risk_partial | {summary['protected_lesion_safety_penalty_count']} | {'PASS' if summary['protected_lesion_safety_penalty_count']==0 else 'FAIL'} |",
        f"| patchcore_gate_candidate | {summary['protected_gate_candidate_penalty_count']} | {'PASS' if summary['protected_gate_candidate_penalty_count']==0 else 'FAIL'} |",
        f"| observation_other | {summary['protected_observation_penalty_count']} | {'PASS' if summary['protected_observation_penalty_count']==0 else 'FAIL'} |",
        "",
        "## Candidate-Level Rank 변화 (26행 내부)",
        "",
        "| review_id | original_rank | adjusted_rank | rank_change |",
        "|-----------|--------------|---------------|-------------|",
    ]
    for r in rank_summary:
        rc = r["rank_change"]
        rc_str = f"+{rc} (상승)" if isinstance(rc, int) and rc > 0 else (f"{rc} (하락)" if isinstance(rc, int) and rc < 0 else str(rc))
        lines.append(
            f"| {r['review_id']} | {r['original_rank_within_26']}"
            f" | {r['adjusted_rank_within_26']}"
            f" | {rc_str} |"
        )

    lines += [
        "",
        "※ rank는 26행 preview 내부 순위. 전체 dev set 순위 아님.",
        "",
        "## Score 원본 무수정 확인",
        "",
        "- `original_score` 컬럼: 변경 없음",
        "- `adjusted_score_preview`: 이 파일 내부에만 존재하는 preview 값",
        "- 기존 PaDiM score CSV: 수정 없음",
        "- `adjusted_score` 컬럼 (기존 파일 반영용): 미생성",
        "- `suppression_weight` / `refined_score`: 미생성",
        "- threshold 재계산: 없음",
        "",
        "## 한계",
        "",
        "- 26행 preview 내부 변화일 뿐, 전체 dev set ranking 변화 아님",
        "- 전체 FROC/AUROC 성능지표 아님",
        "- threshold 재계산 없음 (실제 FP 감소 미확인)",
        "- stage2_holdout 미사용",
        "- ×0.75 감점이 FP 억제에 충분한지는 checkpoint에서 판단 필요",
        "",
        "## Fail 조건 검사",
        "",
        f"- fail_count: {summary['fail_count']}",
    ]
    if summary["fail_reasons"]:
        for fr in summary["fail_reasons"]:
            lines.append(f"  - {fr}")
    else:
        lines.append("  - 없음")

    lines += [
        "",
        "## 다음 단계",
        "",
    ]
    if verdict == "PASS":
        lines += [
            "- **PASS** → 아래 중 선택:",
            "  - B1-D4d Rule-B3 soft penalty ×0.5 preview (추가 감점 비교)",
            "  - B1-D4e Rule-B3 preview checkpoint (×0.75가 충분히 보수적이라고 판단 시)",
        ]
    else:
        lines.append("- **FAIL** → fail_reasons 확인 후 조건 수정")

    lines.append("")

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] Report MD 생성: {OUTPUT_REPORT}")


def main():
    print("=== B1-D4c Rule-B3 Soft Penalty ×0.75 Preview ===")

    # 1. 출력 파일 존재 확인
    check_output_exists()

    # 2. 입력 파일 mtime 기록
    initial_mtimes = record_input_mtimes()
    print("[OK] 입력 파일 mtime 기록 완료")

    # 3. B1-D4b summary 검증
    b1d4b_summary = load_b1d4b_summary()
    validate_b1d4b_summary(b1d4b_summary)

    # 4. B1-D4b CSV 로드
    b1d4b_rows = load_b1d4b_csv()
    print(f"[OK] B1-D4b CSV 로드: {len(b1d4b_rows)}행")

    # 5. soft penalty 적용
    output_rows = [apply_soft_penalty(row) for row in b1d4b_rows]
    applied_count = sum(1 for r in output_rows if r["soft_penalty_applied"])
    applied_ids   = sorted(r["review_id"] for r in output_rows if r["soft_penalty_applied"])
    print(f"[OK] soft_penalty_applied_count={applied_count}, ids={applied_ids}")

    # 6. fail 조건 검사
    fail_count, fail_reasons = compute_fail_conditions(output_rows)
    if fail_count > 0:
        print(f"[FAIL] fail_count={fail_count}")
        for fr in fail_reasons:
            print(f"  - {fr}")

    # 7. rank 변화 계산
    rank_summary = compute_rank_summary(output_rows)

    # 8. 입력 mtime 최종 확인
    mtime_unchanged = verify_input_mtimes_unchanged(initial_mtimes)
    if not mtime_unchanged:
        print("[WARN] 입력 파일 mtime 변경됨")

    # 9. 출력 쓰기
    write_output_csv(output_rows)
    summary = write_summary_json(output_rows, fail_count, fail_reasons, rank_summary, mtime_unchanged)
    write_report_md(output_rows, summary, rank_summary)

    # 10. 최종 보고
    verdict = summary["verdict"]
    print()
    print(f"=== 판정: {verdict} ===")
    print(f"  input_rows                 : {len(output_rows)}")
    print(f"  output_rows                : {len(output_rows)}")
    print(f"  penalty_alpha              : {PENALTY_ALPHA}")
    print(f"  soft_penalty_applied_count : {summary['soft_penalty_applied_count']}")
    print(f"  penalized_review_ids       : {summary['penalized_review_ids']}")
    print(f"  hard_case_penalty          : {summary['protected_hard_case_penalty_count']}")
    print(f"  lesion_safety_penalty      : {summary['protected_lesion_safety_penalty_count']}")
    print(f"  gate_candidate_penalty     : {summary['protected_gate_candidate_penalty_count']}")
    print(f"  observation_penalty        : {summary['protected_observation_penalty_count']}")
    print(f"  original_score_modified    : {summary['original_score_modified']}")
    print(f"  adjusted_score_created     : {summary['adjusted_score_created']}")
    print(f"  threshold_recomputed       : {summary['threshold_recomputed']}")
    print(f"  fail_count                 : {summary['fail_count']}")
    print(f"  mtime_unchanged            : {summary['input_mtime_unchanged']}")
    print(f"  next                       : {summary['next_recommended_step']}")
    print()
    print("  [rank_change_summary]")
    for r in rank_summary:
        print(f"    {r['review_id']}: orig_score={r['original_score']:.4f}"
              f" -> adj={r['adjusted_score_preview']:.4f}"
              f" delta={r['score_delta']:.4f}"
              f" rank {r['original_rank_within_26']} -> {r['adjusted_rank_within_26']}"
              f" (change={r['rank_change']})")

    if verdict == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
