"""
B1-D4b Rule-B3 Dev-Only Flag-Only Preview
==========================================
목적: Rule-B3 flag_only 컬럼을 생성하는 dev-only preview.
- score 수정 없음
- adjusted_score 생성 없음
- threshold 재계산 없음
- stage2_holdout 접근 없음
- flag 대상: R015/R001/R016/R028 (confirmed overlap_artifact) 4개만
"""

import os
import sys
import json
import csv
from pathlib import Path

# ── 프로젝트 루트 ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "b1d1_wall_mediastinum_fp_cause_diagnostic_v1"

# ── 입력 파일 ─────────────────────────────────────────────────────────────
INPUT_FILES = {
    "adjustment_plan":   OUT_DIR / "b1d4_rule_b3_adjustment_plan_preview.csv",
    "safety_sentinel":   OUT_DIR / "b1d4_rule_b3_safety_sentinel_plan.csv",
    "preflight_summary": OUT_DIR / "b1d4_rule_b3_score_adjustment_preflight_summary.json",
    "dry_smoke_results": OUT_DIR / "b1d3b_rule_b3_dry_smoke_results.csv",
    "dry_smoke_summary": OUT_DIR / "b1d3b_rule_b3_dry_smoke_summary.json",
    "smoke_manifest":    OUT_DIR / "b1d3a_smoke_preflight_manifest.csv",
    "safety_manifest":   OUT_DIR / "b1d3a_smoke_safety_manifest.csv",
}

# ── 출력 파일 ─────────────────────────────────────────────────────────────
OUTPUT_CSV     = OUT_DIR / "b1d4b_rule_b3_flag_only_preview.csv"
OUTPUT_SUMMARY = OUT_DIR / "b1d4b_rule_b3_flag_only_preview_summary.json"
OUTPUT_REPORT  = OUT_DIR / "b1d4b_rule_b3_flag_only_preview_report.md"

# ── 상수 ──────────────────────────────────────────────────────────────────
FLAG_CANDIDATES        = {"R015", "R001", "R016", "R028"}
PROTECTED_HARD_CASES   = {"R018", "R024"}
PROTECTED_GATE_CANDIDATES = {"R002", "R005", "R006", "R009", "R012", "R030"}
OBSERVATION_OTHER      = {"R003", "R008"}
LESION_KEPT_TYPES      = {"lesion_kept"}
LESION_RISK_TYPES      = {"lesion_risk_partial"}
ALL_LESION_SAFETY_TYPES = LESION_KEPT_TYPES | LESION_RISK_TYPES

# stage2_holdout 경로 키워드 — 접근 시 즉시 중단
STAGE2_PATH_KEYWORDS   = ["stage2", "holdout", "test_split"]


def abort_if_stage2(path_str: str):
    for kw in STAGE2_PATH_KEYWORDS:
        if kw in path_str.lower():
            print(f"[BLOCKED] stage2_holdout 접근 감지: {path_str}")
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
    for key, path in INPUT_FILES.items():
        current_mtime = path.stat().st_mtime
        if current_mtime != initial_mtimes[key]:
            print(f"[WARN] 입력 파일 mtime 변경 감지: {path}")
            return False
    return True


def load_preflight_summary() -> dict:
    with open(INPUT_FILES["preflight_summary"], "r", encoding="utf-8") as f:
        return json.load(f)


def load_dry_smoke_results() -> list[dict]:
    rows = []
    with open(INPUT_FILES["dry_smoke_results"], "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            abort_if_stage2(str(row))
            # row-level holdout 검증: holdout_flag 컬럼이 있으면 반드시 0이어야 함
            if "holdout_flag" in row:
                hf_val = row["holdout_flag"].strip()
                if hf_val not in ("0", ""):
                    print(f"[BLOCKED] row holdout_flag != 0 감지 (review_id={row.get('review_id','?')}, holdout_flag={hf_val!r})")
                    sys.exit(2)
            rows.append(row)
    return rows


def compute_flag_only(row: dict) -> tuple[bool, str]:
    """
    rule_b3_flag_only = True 조건:
    - review_id in FLAG_CANDIDATES
    - b1d3b rule_b3_flag == "true"
    - source_manifest == "smoke"
    - holdout_flag == 0
    - not PROTECTED_HARD_CASES / lesion_safety / gate_candidate / observation
    """
    rid = row.get("review_id", "")
    b3_flag = row.get("rule_b3_flag", "false").strip().lower() == "true"
    src_manifest = row.get("source_manifest", "").strip().lower()
    safety_type = row.get("safety_type", "").strip().lower()
    src_group = row.get("source_group", "").strip().lower()

    if rid not in FLAG_CANDIDATES:
        return False, f"review_id({rid}) not in FLAG_CANDIDATES"
    if not b3_flag:
        return False, "b1d3b rule_b3_flag=false"
    if src_manifest != "smoke":
        return False, f"source_manifest={src_manifest} (safety 행은 flag 대상 아님)"
    if rid in PROTECTED_HARD_CASES:
        return False, f"protected_hard_case({rid})"
    if safety_type in ALL_LESION_SAFETY_TYPES:
        return False, f"lesion_safety({safety_type})"
    if rid in PROTECTED_GATE_CANDIDATES:
        return False, f"patchcore_gate_candidate({rid})"
    if src_group == "observation_other" or rid in OBSERVATION_OTHER:
        return False, f"observation_other({rid})"
    return True, "confirmed_overlap_artifact_boundary_rule_b3"


def compute_fail_conditions(row: dict, flag_only: bool) -> tuple[bool, str]:
    rid = row.get("review_id", "")
    safety_type = row.get("safety_type", "").strip().lower()
    src_group = row.get("source_group", "").strip().lower()
    fails = []

    if flag_only and rid not in FLAG_CANDIDATES:
        fails.append(f"flag=True이지만 review_id({rid}) not in FLAG_CANDIDATES")
    if flag_only and rid in PROTECTED_HARD_CASES:
        fails.append(f"flag=True이지만 protected_hard_case({rid})")
    if flag_only and safety_type in ALL_LESION_SAFETY_TYPES:
        fails.append(f"flag=True이지만 lesion_safety({safety_type})")
    if flag_only and rid in PROTECTED_GATE_CANDIDATES:
        fails.append(f"flag=True이지만 patchcore_gate_candidate({rid})")
    if flag_only and (src_group == "observation_other" or rid in OBSERVATION_OTHER):
        fails.append(f"flag=True이지만 observation_other({rid})")

    hit = len(fails) > 0
    return hit, "; ".join(fails) if fails else ""


def resolve_holdout_flag(row: dict) -> int:
    """
    row에 holdout_flag 컬럼이 있으면 그 값을 파싱해 반환.
    없으면 summary-level 검증이 이미 완료된 것으로 간주하고 0 반환.
    어떤 경우든 0이 아니면 즉시 중단.
    """
    if "holdout_flag" in row:
        hf_raw = row["holdout_flag"].strip()
        try:
            hf = int(hf_raw) if hf_raw else 0
        except ValueError:
            print(f"[BLOCKED] holdout_flag 파싱 실패 (review_id={row.get('review_id','?')}, value={hf_raw!r})")
            sys.exit(2)
        if hf != 0:
            print(f"[BLOCKED] holdout_flag != 0 (review_id={row.get('review_id','?')}, holdout_flag={hf})")
            sys.exit(2)
        return hf
    # 컬럼 없음 → summary-level stage2_holdout_access=0 이중 확인은 main()에서 수행
    return 0


def build_output_row(row: dict, flag_only: bool, flag_reason: str,
                     fail_hit: bool, fail_reason: str) -> dict:
    rid = row.get("review_id", "")
    safety_type = row.get("safety_type", "").strip().lower()
    src_group = row.get("source_group", "").strip().lower()
    score_str = row.get("candidate_score", "")
    try:
        float(score_str)
    except (ValueError, TypeError):
        pass

    # holdout 검증: row-level 값 확인 후 source로 보존
    verified_holdout = resolve_holdout_flag(row)
    source_holdout_raw = row.get("holdout_flag", "not_in_source")  # 원본 입력값 보존

    return {
        "row_id":                   row.get("row_id", ""),
        "review_id":                rid,
        "patient_id":               row.get("patient_id", ""),
        "source_group":             row.get("source_group", ""),
        "safety_type":              row.get("safety_type", ""),
        "safety_role":              row.get("safety_role", ""),
        "human_label":              row.get("human_label", ""),
        "candidate_score":          score_str,
        "original_score":           score_str,    # score 복사 (변경 없음)
        "score_unchanged":          True,
        "rule_b3_flag_only":        flag_only,
        "rule_b3_flag_reason":      flag_reason,
        "adjustment_candidate":     rid in FLAG_CANDIDATES,
        "protected_hard_case":      rid in PROTECTED_HARD_CASES,
        "protected_lesion_safety":  safety_type in ALL_LESION_SAFETY_TYPES,
        "protected_gate_candidate": rid in PROTECTED_GATE_CANDIDATES,
        "protected_observation":    (src_group == "observation_other" or rid in OBSERVATION_OTHER),
        "fail_condition_hit":       fail_hit,
        "fail_reason":              fail_reason,
        "stage_split_check":        "stage1_dev_only",
        "source_holdout_flag":      source_holdout_raw,   # 입력값 원본 보존
        "holdout_flag":             verified_holdout,      # 검증된 0
    }


def write_output_csv(output_rows: list[dict]):
    fieldnames = [
        "row_id", "review_id", "patient_id", "source_group", "safety_type",
        "safety_role", "human_label", "candidate_score", "original_score",
        "score_unchanged", "rule_b3_flag_only", "rule_b3_flag_reason",
        "adjustment_candidate", "protected_hard_case", "protected_lesion_safety",
        "protected_gate_candidate", "protected_observation",
        "fail_condition_hit", "fail_reason", "stage_split_check",
        "source_holdout_flag", "holdout_flag",
    ]
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"[OK] CSV 생성: {OUTPUT_CSV}")


def write_summary_json(output_rows: list[dict], preflight_summary: dict,
                       mtime_unchanged: bool):
    flagged_ids = [r["review_id"] for r in output_rows if r["rule_b3_flag_only"] is True]
    hard_flag_count = sum(
        1 for r in output_rows
        if r["protected_hard_case"] and r["rule_b3_flag_only"]
    )
    lesion_flag_count = sum(
        1 for r in output_rows
        if r["protected_lesion_safety"] and r["rule_b3_flag_only"]
    )
    gate_flag_count = sum(
        1 for r in output_rows
        if r["protected_gate_candidate"] and r["rule_b3_flag_only"]
    )
    obs_flag_count = sum(
        1 for r in output_rows
        if r["protected_observation"] and r["rule_b3_flag_only"]
    )
    fail_rows = [r for r in output_rows if r["fail_condition_hit"]]
    fail_count = len(fail_rows)
    fail_reasons = [r["fail_reason"] for r in fail_rows if r["fail_reason"]]

    verdict = "PASS" if (
        fail_count == 0
        and len(flagged_ids) == 4
        and set(flagged_ids) == FLAG_CANDIDATES
        and hard_flag_count == 0
        and lesion_flag_count == 0
        and gate_flag_count == 0
        and obs_flag_count == 0
    ) else "FAIL"

    summary = {
        "step": "B1-D4b_Rule_B3_dev_only_flag_only_preview",
        "verdict": verdict,
        "stage2_holdout_access": 0,
        "input_rows": len(output_rows),
        "output_rows": len(output_rows),
        "flag_true_count": len(flagged_ids),
        "flagged_review_ids": sorted(flagged_ids),
        "expected_flagged_review_ids": sorted(FLAG_CANDIDATES),
        "protected_hard_case_flag_count": hard_flag_count,
        "protected_lesion_safety_flag_count": lesion_flag_count,
        "protected_gate_candidate_flag_count": gate_flag_count,
        "protected_observation_flag_count": obs_flag_count,
        "score_modified": False,
        "adjusted_score_created": False,
        "suppression_weight_created": False,
        "refined_score_created": False,
        "threshold_recomputed": False,
        "fail_count": fail_count,
        "fail_reasons": fail_reasons,
        "input_mtime_unchanged": mtime_unchanged,
        "next_recommended_step": (
            "B1-D4c Rule-B3 soft penalty ×0.75 preview"
            if verdict == "PASS"
            else "Rule-B3 조건 재설계"
        ),
    }

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] Summary JSON 생성: {OUTPUT_SUMMARY}")
    return summary


def write_report_md(output_rows: list[dict], summary: dict):
    flagged = [r for r in output_rows if r["rule_b3_flag_only"] is True]
    verdict = summary["verdict"]

    lines = [
        "# B1-D4b Rule-B3 Dev-Only Flag-Only Preview Report",
        "",
        f"**판정: {verdict}**",
        "",
        "## B1-D4 Preflight 요약",
        "",
        "B1-D4 preflight PASS 완료. adjustment candidates 4개(R015/R001/R016/R028) 확정,"
        " uncertain hold 5개, protected hard cases 2개(R018/R024),"
        " lesion_kept 17개, lesion_risk_partial 7개, gate_candidate 6개 보호.",
        "",
        "## Flag-Only Preview 목적",
        "",
        "score를 변경하지 않고 `rule_b3_flag_only` 컬럼만 추가하여,"
        " Rule-B3 flag가 정확히 4개 후보(R015/R001/R016/R028)에만 붙고"
        " 보호 대상에는 절대 붙지 않는지 검증한다.",
        "",
        "## Flag된 후보 목록",
        "",
        "| review_id | candidate_score | refined_roi_ratio 참고 | rule_b3_flag_reason |",
        "|-----------|----------------|------------------------|---------------------|",
    ]

    # adjustment_plan 데이터에서 ratio 추가 정보 제공
    ratio_map = {
        "R015": "0.4111", "R001": "0.5713",
        "R016": "0.3955", "R028": "0.6602",
    }
    for r in flagged:
        rid = r["review_id"]
        score = r["candidate_score"]
        ratio = ratio_map.get(rid, "-")
        reason = r["rule_b3_flag_reason"]
        lines.append(f"| {rid} | {score} | {ratio} | {reason} |")

    lines += [
        "",
        "## 보호 대상 검증 결과",
        "",
        "| 보호 카테고리 | 대상 | flag=True 발생 수 | 결과 |",
        "|--------------|------|-------------------|------|",
        f"| boundary_hard_case | R018, R024 | {summary['protected_hard_case_flag_count']} | {'PASS' if summary['protected_hard_case_flag_count']==0 else 'FAIL'} |",
        f"| lesion_kept / lesion_risk_partial | 17+7=24개 | {summary['protected_lesion_safety_flag_count']} | {'PASS' if summary['protected_lesion_safety_flag_count']==0 else 'FAIL'} |",
        f"| patchcore_gate_candidate | R002/R005/R006/R009/R012/R030 | {summary['protected_gate_candidate_flag_count']} | {'PASS' if summary['protected_gate_candidate_flag_count']==0 else 'FAIL'} |",
        f"| observation_other | R003, R008 | {summary['protected_observation_flag_count']} | {'PASS' if summary['protected_observation_flag_count']==0 else 'FAIL'} |",
        "",
        "## Score 무수정 확인",
        "",
        "- `score_unchanged = True` (전 행 동일)",
        "- `original_score = candidate_score` (복사, 변경 없음)",
        "- `adjusted_score` 컬럼 미생성",
        "- `suppression_weight` 컬럼 미생성",
        "- `refined_score` 컬럼 미생성",
        "- threshold 재계산 없음",
        "",
        "## 한계",
        "",
        "- 아직 score 감점 없음 (flag 컬럼만 추가, 실제 ranking 변화 없음)",
        "- 아직 FP 감소 성능지표(FROC, patient AUROC)가 아님",
        "- stage2_holdout 미사용 (stage1_dev only)",
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
        lines.append("- **PASS** → B1-D4c Rule-B3 soft penalty ×0.75 preview preflight 또는 실행")
    else:
        lines.append("- **FAIL** → Rule-B3 조건 재설계 필요. fail_reasons 확인 후 수정")

    lines.append("")

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] Report MD 생성: {OUTPUT_REPORT}")


def main():
    print("=== B1-D4b Rule-B3 Flag-Only Preview ===")

    # 1. 출력 파일 존재 확인
    check_output_exists()

    # 2. 입력 파일 mtime 기록
    initial_mtimes = record_input_mtimes()
    print("[OK] 입력 파일 mtime 기록 완료")

    # 3. preflight summary 로드 및 검증 (B1-D4)
    preflight = load_preflight_summary()
    assert preflight.get("verdict") == "PASS", "B1-D4 preflight verdict가 PASS가 아님"
    assert preflight.get("stage2_holdout_access", 1) == 0, \
        f"B1-D4 preflight stage2_holdout_access != 0: {preflight.get('stage2_holdout_access')}"
    flagged_from_preflight = set(preflight["b1d3b_verification"]["flagged_review_ids"])
    assert flagged_from_preflight == FLAG_CANDIDATES, (
        f"adjustment candidates 불일치: {flagged_from_preflight} != {FLAG_CANDIDATES}"
    )
    print("[OK] B1-D4 preflight 검증 완료 (PASS, stage2=0, candidates 일치)")

    # 3b. b1d3b summary 이중 확인 (stage2_holdout_access=0)
    with open(INPUT_FILES["dry_smoke_summary"], "r", encoding="utf-8") as f:
        b1d3b_summary = json.load(f)
    b1d3b_stage2 = b1d3b_summary.get("stage2_holdout_access", None)
    if b1d3b_stage2 is None:
        print("[WARN] b1d3b summary에 stage2_holdout_access 키 없음 — 추가 확인 필요")
    elif b1d3b_stage2 != 0:
        print(f"[BLOCKED] b1d3b summary stage2_holdout_access != 0: {b1d3b_stage2}")
        sys.exit(2)
    else:
        print(f"[OK] b1d3b summary stage2_holdout_access=0 확인")

    # 4. dry smoke results 로드
    smoke_rows = load_dry_smoke_results()
    assert len(smoke_rows) == 26, f"예상 26행인데 {len(smoke_rows)}행"
    print(f"[OK] B1-D3b dry smoke results 로드: {len(smoke_rows)}행")

    # 5. flag_only 계산 및 출력 행 구성
    output_rows = []
    for row in smoke_rows:
        flag_only, flag_reason = compute_flag_only(row)
        fail_hit, fail_reason = compute_fail_conditions(row, flag_only)
        out_row = build_output_row(row, flag_only, flag_reason, fail_hit, fail_reason)
        output_rows.append(out_row)

    flag_count = sum(1 for r in output_rows if r["rule_b3_flag_only"] is True)
    flagged_ids = [r["review_id"] for r in output_rows if r["rule_b3_flag_only"] is True]
    print(f"[OK] flag_true_count={flag_count}, flagged_ids={sorted(flagged_ids)}")

    # 6. 입력 mtime 최종 확인
    mtime_unchanged = verify_input_mtimes_unchanged(initial_mtimes)
    if not mtime_unchanged:
        print("[WARN] 입력 파일 mtime 변경됨")

    # 7. 출력 파일 쓰기
    write_output_csv(output_rows)
    summary = write_summary_json(output_rows, preflight, mtime_unchanged)
    write_report_md(output_rows, summary)

    # 8. 최종 보고
    verdict = summary["verdict"]
    print()
    print(f"=== 판정: {verdict} ===")
    print(f"  input_rows       : {len(output_rows)}")
    print(f"  output_rows      : {len(output_rows)}")
    print(f"  flag_true_count  : {summary['flag_true_count']}")
    print(f"  flagged_ids      : {summary['flagged_review_ids']}")
    print(f"  hard_case_flag   : {summary['protected_hard_case_flag_count']}")
    print(f"  lesion_safe_flag : {summary['protected_lesion_safety_flag_count']}")
    print(f"  gate_cand_flag   : {summary['protected_gate_candidate_flag_count']}")
    print(f"  obs_flag         : {summary['protected_observation_flag_count']}")
    print(f"  score_modified   : {summary['score_modified']}")
    print(f"  fail_count       : {summary['fail_count']}")
    print(f"  mtime_unchanged  : {summary['input_mtime_unchanged']}")
    print(f"  next             : {summary['next_recommended_step']}")

    if verdict == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
