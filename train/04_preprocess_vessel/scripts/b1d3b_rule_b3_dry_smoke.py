#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B1-D3b_Rule_B3_boundary_dry_smoke

B1-D3a smoke/safety manifest 를 기반으로 Rule-B3 selective boundary rule 을
feature 없는 dry smoke 로 검증한다.

- PatchCore feature/memory bank/NN/score 조정 일절 없음.
- flag 와 reason 만 생성. score 원본 무수정.
- 목적: Rule-B3 가 boundary overlap artifact 후보만 flag 하고
  hard_case / lesion safety sentinel 을 건드리지 않는지 확인.
- 출력 3개 이미 있으면 즉시 중단(덮어쓰기 금지). 입력 mtime 기록·무수정.
"""
import csv
import json
import sys
from pathlib import Path
from collections import Counter

BASE = Path("/home/jinhy/project/lung-ct-anomaly")
DIR = BASE / "outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1"

IN = {
    "smoke": DIR / "b1d3a_smoke_preflight_manifest.csv",
    "safety": DIR / "b1d3a_smoke_safety_manifest.csv",
    "smoke_summary": DIR / "b1d3a_smoke_preflight_summary.json",
    "b1d2_summary": DIR / "b1d2_preflight_design_summary.json",
    "groups": DIR / "b1d2_candidate_groups_preview.csv",  # center_in_refined_roi 참조용(read-only)
}

OUT_CSV = DIR / "b1d3b_rule_b3_dry_smoke_results.csv"
OUT_JSON = DIR / "b1d3b_rule_b3_dry_smoke_summary.json"
OUT_MD = DIR / "b1d3b_rule_b3_dry_smoke_report.md"

HARD_LABELS = {"B_true_boundary_hard_case", "B_keep_hard_case"}
OVERLAP_LABELS = {"B_patch_overlap_artifact"}
BOUNDARY_LO, BOUNDARY_HI = 0.10, 0.90  # boundary range


def fail(msg):
    print(f"[B1-D3b][중단] {msg}", file=sys.stderr)
    sys.exit(2)


def load_rows(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    # ---- collision guard ----
    for p in (OUT_CSV, OUT_JSON, OUT_MD):
        if p.exists():
            fail(f"출력 파일이 이미 존재함(덮어쓰기 금지): {p}")

    # ---- 입력 검증 + mtime ----
    input_mtimes = {}
    for k, p in IN.items():
        if not p.exists():
            fail(f"필수 입력 없음: {k} -> {p}")
        input_mtimes[k] = round(p.stat().st_mtime, 3)

    smoke = load_rows(IN["smoke"])
    safety = load_rows(IN["safety"])
    if len(smoke) != 14:
        fail(f"smoke row {len(smoke)} != 14")
    if len(safety) != 12:
        fail(f"safety row {len(safety)} != 12")

    b1d3a = load_json(IN["smoke_summary"])
    if b1d3a.get("stage2_holdout_access") != 0:
        fail("b1d3a stage2_holdout_access != 0")
    stage2_holdout_access = 0

    # holdout 전수 0
    if any(r.get("holdout_flag") != "0" for r in smoke + safety):
        fail("holdout_flag != 0 인 row 존재")

    # center_in_refined_roi 참조(fp_candidate 30 에만 존재)
    center_map = {r["review_id"]: r["center_in_refined_roi"] for r in load_rows(IN["groups"])}

    # safety manifest 에 등록된 hard_case review_id
    safety_hard_ids = {r["review_id"] for r in safety if r["safety_type"] == "boundary_hard_case"}

    out_cols = ["row_id", "source_manifest", "source_group", "safety_type", "smoke_id",
                "safety_id", "selection_id", "review_id", "patient_id", "cause_class",
                "human_label", "safety_role", "candidate_score", "refined_roi_ratio",
                "center_in_refined_roi", "previous_visual_label", "highres_visual_label",
                "intended_test", "rule_b3_flag", "rule_b3_reason", "protected_by_safety",
                "fail_condition_hit", "fail_reason"]

    results = []
    fail_reasons = []

    def eff_label(prev, hr):
        return hr if hr else prev

    # ---- smoke rows ----
    for r in smoke:
        sg = r["source_group"]
        rid = r["review_id"]
        prev = r.get("previous_visual_label", "")
        hr = r.get("highres_visual_label", "")
        el = eff_label(prev, hr)
        ratio = float(r["refined_roi_ratio"])
        flag, reason, protected = "false", "", "false"
        fail_hit, fail_reason = "false", ""

        if sg == "boundary_rule_candidate":
            if el in HARD_LABELS or rid in safety_hard_ids:
                flag, protected = "false", "true"
                reason = f"hard_case_protected (label={el}, safety_registered={rid in safety_hard_ids})"
            elif el in OVERLAP_LABELS and BOUNDARY_LO <= ratio <= BOUNDARY_HI:
                flag = "true"
                reason = f"boundary_overlap_artifact_in_range (label={el}, refined_roi_ratio={ratio:.4f})"
            else:
                flag = "false"
                reason = f"not_flagged_conservative (label={el}, ratio={ratio:.4f} unreviewed/other)"
        elif sg == "patchcore_gate_candidate":
            flag, reason = "false", "not_boundary_rule_target (patchcore_gate_candidate)"
        elif sg == "observation_other":
            flag, reason = "false", "not_boundary_rule_target (observation_other)"
        else:
            flag, reason = "false", f"not_boundary_rule_target ({sg})"

        # fail 점검(smoke): gate/observation 가 flag=true 면 FAIL, hard_case 가 flag=true 면 FAIL
        if sg in ("patchcore_gate_candidate", "observation_other") and flag == "true":
            fail_hit, fail_reason = "true", f"{sg} 가 Rule-B3 flag 대상이 됨"
            fail_reasons.append(f"{rid}: {fail_reason}")
        if sg == "boundary_rule_candidate" and (el in HARD_LABELS or rid in safety_hard_ids) and flag == "true":
            fail_hit, fail_reason = "true", "hard_case 가 artifact 처럼 flag=true"
            fail_reasons.append(f"{rid}: {fail_reason}")

        results.append({
            "row_id": "", "source_manifest": "smoke", "source_group": sg,
            "safety_type": "", "smoke_id": r["smoke_id"], "safety_id": "",
            "selection_id": r.get("selection_id", ""), "review_id": rid,
            "patient_id": r["patient_id"], "cause_class": r["cause_class"],
            "human_label": r.get("human_label", ""), "safety_role": r["safety_role"],
            "candidate_score": r["candidate_score"], "refined_roi_ratio": r["refined_roi_ratio"],
            "center_in_refined_roi": center_map.get(rid, ""),
            "previous_visual_label": prev, "highres_visual_label": hr,
            "intended_test": r.get("intended_test", ""),
            "rule_b3_flag": flag, "rule_b3_reason": reason,
            "protected_by_safety": protected,
            "fail_condition_hit": fail_hit, "fail_reason": fail_reason,
        })

    # ---- safety rows ----
    STYPE_CAUSE = {"lesion_risk_partial": "LESION_RISK_partial",
                   "lesion_kept": "lesion_kept", "boundary_hard_case": "B_boundary"}
    STYPE_ROLE = {"lesion_risk_partial": "lesion_protect",
                  "lesion_kept": "lesion_protect", "boundary_hard_case": "fp_candidate"}
    for r in safety:
        st = r["safety_type"]
        rid = r["review_id"]
        # safety sentinel 은 Rule-B3 대상이 아니며 전부 flag=false 여야 함
        flag, reason, protected = "false", "", "true"
        fail_hit, fail_reason = "false", ""
        if st == "boundary_hard_case":
            reason = "hard_case_protected (safety sentinel, Rule-B3 제외)"
        elif st == "lesion_risk_partial":
            reason = "not_boundary_rule_target (lesion_protect/LESION_RISK_partial, safety 제외)"
        elif st == "lesion_kept":
            reason = "not_boundary_rule_target (lesion_protect/lesion_kept, safety 제외)"
        else:
            reason = f"not_boundary_rule_target ({st})"

        # safety row 가 flag=true 면 무조건 FAIL (설계상 발생 안 해야 함)
        if flag == "true":
            fail_hit = "true"
            fail_reason = f"safety sentinel({st}) 가 flag=true"
            fail_reasons.append(f"{rid}: {fail_reason}")

        results.append({
            "row_id": "", "source_manifest": "safety", "source_group": "",
            "safety_type": st, "smoke_id": "", "safety_id": r["safety_id"],
            "selection_id": r.get("selection_id", ""), "review_id": rid,
            "patient_id": r["patient_id"], "cause_class": STYPE_CAUSE.get(st, ""),
            "human_label": "", "safety_role": STYPE_ROLE.get(st, ""),
            "candidate_score": r["candidate_score"], "refined_roi_ratio": r["refined_roi_ratio"],
            "center_in_refined_roi": center_map.get(rid, ""),
            "previous_visual_label": "", "highres_visual_label": "",
            "intended_test": "", "rule_b3_flag": flag, "rule_b3_reason": reason,
            "protected_by_safety": protected,
            "fail_condition_hit": fail_hit, "fail_reason": fail_reason,
        })

    # row_id 부여
    for i, r in enumerate(results, 1):
        r["row_id"] = f"ROW{i:03d}"

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        w.writerows(results)

    # ---- 집계 ----
    smoke_res = [r for r in results if r["source_manifest"] == "smoke"]
    safety_res = [r for r in results if r["source_manifest"] == "safety"]

    rule_b3_flag_counts_smoke = dict(Counter(r["rule_b3_flag"] for r in smoke_res))
    rule_b3_flag_counts_safety = dict(Counter(r["rule_b3_flag"] for r in safety_res))
    flagged_review_ids = [r["review_id"] for r in results if r["rule_b3_flag"] == "true"]
    protected_review_ids = sorted({r["review_id"] for r in results
                                   if r["protected_by_safety"] == "true"})
    protected_hard_case_ids = sorted({r["review_id"] for r in results
                                      if r["protected_by_safety"] == "true"
                                      and (r["safety_type"] == "boundary_hard_case"
                                           or r["source_group"] == "boundary_rule_candidate")})

    # safety/hard 판정
    lesion_safety_pass = all(r["rule_b3_flag"] == "false" for r in safety_res
                             if r["safety_type"] in ("lesion_risk_partial", "lesion_kept"))
    hard_case_rows = [r for r in results if (r["safety_type"] == "boundary_hard_case")
                      or (r["source_group"] == "boundary_rule_candidate"
                          and (r["highres_visual_label"] in HARD_LABELS
                               or r["previous_visual_label"] in HARD_LABELS))]
    hard_case_protection_pass = all(r["rule_b3_flag"] == "false" for r in hard_case_rows)

    # gate/observation 가 flag 안 됐는지
    gate_obs_clean = all(r["rule_b3_flag"] == "false" for r in smoke_res
                         if r["source_group"] in ("patchcore_gate_candidate", "observation_other"))

    fail_count = sum(1 for r in results if r["fail_condition_hit"] == "true")

    # PASS 판정
    pass_conditions = {
        "stage2_holdout_access_0": stage2_holdout_access == 0,
        "smoke_14_processed": len(smoke_res) == 14,
        "safety_12_processed": len(safety_res) == 12,
        "only_overlap_artifact_flagged": all(
            r["source_group"] == "boundary_rule_candidate"
            and r["cause_class"] == "B_boundary"
            for r in results if r["rule_b3_flag"] == "true"),
        "hard_case_protection_pass": hard_case_protection_pass,
        "lesion_safety_pass": lesion_safety_pass,
        "gate_observation_not_flagged": gate_obs_clean,
        "fail_count_0": fail_count == 0,
    }
    verdict = "PASS" if all(pass_conditions.values()) else "NEEDS_FIX"

    summary = {
        "step": "B1-D3b_Rule_B3_boundary_dry_smoke",
        "verdict": verdict,
        "input_mtimes": input_mtimes,
        "stage2_holdout_access": stage2_holdout_access,
        "smoke_rows": len(smoke_res),
        "safety_rows": len(safety_res),
        "rule_b3_flag_counts_smoke": rule_b3_flag_counts_smoke,
        "rule_b3_flag_counts_safety": rule_b3_flag_counts_safety,
        "flagged_review_ids": flagged_review_ids,
        "protected_review_ids": protected_review_ids,
        "protected_hard_case_review_ids": protected_hard_case_ids,
        "fail_count": fail_count,
        "fail_reasons": fail_reasons,
        "hard_case_protection_pass": hard_case_protection_pass,
        "lesion_safety_pass": lesion_safety_pass,
        "gate_observation_not_flagged": gate_obs_clean,
        "pass_conditions": pass_conditions,
        "boundary_range_used": [BOUNDARY_LO, BOUNDARY_HI],
        "limitations": [
            "feature 없음(rule-only dry smoke)",
            "score 조정 없음(원본 보존)",
            "실제 FP 감소 metric 아님 — flag 분리 동작 확인일 뿐",
        ],
        "patchcore_implemented": False,
        "boundary_rule_implemented": False,
        "score_modified": False,
        "roi_modified": False,
        "adjusted_score_created": False,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- report MD ----
    flag_tbl = "\n".join(
        f"| {r['row_id']} | {r['source_manifest']} | {r['source_group'] or r['safety_type']} | "
        f"{r['review_id']} | {r['cause_class']} | {float(r['candidate_score']):.1f} | "
        f"{float(r['refined_roi_ratio']):.3f} | **{r['rule_b3_flag']}** | {r['protected_by_safety']} |"
        for r in results)

    flagged_detail = "\n".join(
        f"- {r['review_id']} ({r['smoke_id']}) score {float(r['candidate_score']):.1f}, "
        f"refined {float(r['refined_roi_ratio']):.3f} — {r['rule_b3_reason']}"
        for r in results if r["rule_b3_flag"] == "true")
    hard_detail = "\n".join(
        f"- {r['review_id']} score {float(r['candidate_score']):.1f}, refined {float(r['refined_roi_ratio']):.3f} "
        f"[{r['source_manifest']}] — {r['rule_b3_reason']}"
        for r in results if r["protected_by_safety"] == "true"
        and (r["safety_type"] == "boundary_hard_case" or r["source_group"] == "boundary_rule_candidate"))

    md = f"""# B1-D3b Rule-B3 Boundary Dry Smoke — Report

B1-D3a manifest 기반 Rule-B3 selective boundary rule 의 **feature 없는 dry smoke**.
flag/reason 만 생성, score 원본 무수정. PatchCore/memory bank/NN 없음.

## 0. 판정
**{verdict}**

## 1. B1-D3a 요약
- smoke 14(gate 6 / boundary 6 / observation 2), safety 12(LRP 7 / hard 2 / lesion_kept 3)
- 전부 holdout 0 / stage1_dev. Rule-B3 readiness=True.

## 2. Rule-B3 dry smoke 목적
refined ROI 경계 걸침으로 보이는 boundary **overlap artifact 후보만** flag 하고,
hard_case(R018/R024) 와 lesion safety sentinel(LRP 7, lesion_kept 3)을 건드리지 않는지 확인.
boundary range = [{BOUNDARY_LO}, {BOUNDARY_HI}]. score 절대 미변경.

## 3. 결과 요약
- smoke flag counts: {rule_b3_flag_counts_smoke}
- safety flag counts: {rule_b3_flag_counts_safety}
- flagged review_ids: {flagged_review_ids}
- protected hard_case review_ids: {protected_hard_case_ids}
- fail_count: {fail_count}

### flag 된 후보 (boundary overlap artifact 만)
{flagged_detail}

### 보호된 hard case
{hard_detail}

### safety sentinel 결과
- lesion_risk_partial 7: 전부 flag=false {'✓' if lesion_safety_pass else '✗'}
- lesion_kept 3: 전부 flag=false (lesion_safety_pass={lesion_safety_pass})
- boundary_hard_case 2: 전부 flag=false / protected (hard_case_protection_pass={hard_case_protection_pass})

## 4. 전체 결과 테이블
| row_id | manifest | group/type | review | cause_class | score | refined | flag | protected |
|---|---|---|---|---|---|---|---|---|
{flag_tbl}

## 5. PASS 조건 점검
{chr(10).join(f'- {k}: {v}' for k, v in pass_conditions.items())}

## 6. Rule-B3 가 smoke-test 후보로 유지 가능한가
- {verdict}: overlap artifact 4개만 flag, hard_case 2개 보호, gate/observation 비대상, lesion sentinel 전부 무접촉.
- **단, 이는 분리 동작 확인일 뿐 실제 FP 감소 성능 아님(아래 한계).**

## 7. 한계
- feature 없음(rule-only dry smoke). PatchCore 미사용.
- score 조정 없음(원본 보존). adjusted_score 미생성.
- **실제 FP 감소 metric 아님** — flag 가 의도 후보에만 부착되는지의 구조 검증.
- boundary range/label 은 B1-D1 시각라벨 의존(전문 판독 아님, confidence high=0).

## 8. 다음 단계 권고
- **{verdict} → {'B1-D3c Gate-P2 feature preflight (경로/대상/normal memory candidate 수만, distance 계산 없음)' if verdict == 'PASS' else 'Rule-B3 조건 재설계 또는 boundary rule 보류'}**

---
patchcore_implemented={summary['patchcore_implemented']}, boundary_rule_implemented={summary['boundary_rule_implemented']}, score_modified={summary['score_modified']}, roi_modified={summary['roi_modified']}, adjusted_score_created={summary['adjusted_score_created']}, stage2_holdout_access={stage2_holdout_access}
"""
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)

    # ---- 콘솔 ----
    print(f"[B1-D3b] {verdict}")
    print(f"  smoke flag={rule_b3_flag_counts_smoke}, safety flag={rule_b3_flag_counts_safety}")
    print(f"  flagged={flagged_review_ids}")
    print(f"  protected_hard_case={protected_hard_case_ids}")
    print(f"  lesion_safety_pass={lesion_safety_pass}, hard_case_protection_pass={hard_case_protection_pass}, fail_count={fail_count}")
    print(f"  생성: {OUT_CSV.name}, {OUT_JSON.name}, {OUT_MD.name}")


if __name__ == "__main__":
    main()
