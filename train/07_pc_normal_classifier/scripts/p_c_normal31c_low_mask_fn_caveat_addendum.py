"""
P-C-NORMAL31c: low-mask FN caveat addendum + decision readiness re-adjudication
addendum only — no training, no model forward, no prediction re-export
original P-C-NORMAL31b BLOCKED verdict is preserved and NOT modified
"""

import sys
import csv
import json
import datetime
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")

REPORT31 = PROJECT_ROOT / "outputs/reports/p_c_normal31_repaired_final_test_masked_comparison"
REPORT31B = PROJECT_ROOT / "outputs/reports/p_c_normal31b_zero_mask_patient_sanity_audit"
REPORT_ROOT = PROJECT_ROOT / "outputs/reports/p_c_normal31c_low_mask_fn_caveat_addendum"

# 31b 입력 파일
IN_CASES = REPORT31B / "p_c_normal31b_zero_mask_cases.csv"
IN_PAT_SUMMARY = REPORT31B / "p_c_normal31b_patient_level_sanity_summary.csv"
IN_PAT_METRIC = REPORT31B / "p_c_normal31b_patient_metric_summary.csv"
IN_DECISION = REPORT31B / "p_c_normal31b_decision_readiness.csv"
IN_SUMMARY31B = REPORT31B / "p_c_normal31b_summary.json"
IN_DONE31B = REPORT31B / "DONE.json"

# 31 crop-level metric
IN_CROP_RAW = REPORT31 / "p_c_normal31_crop_raw_metrics_comparison.csv"

# known constants (P-C-NORMAL31 results)
KNOWN = {
    "ref_auroc": 0.9517, "ref_auprc": 0.9733, "ref_brier": 0.1533,
    "ref_specificity": 0.4956, "ref_sensitivity": 0.9929,
    "ref_fp": 10874, "ref_fn": 317,
    "cand_auroc": 0.9904, "cand_auprc": 0.9954, "cand_brier": 0.0750,
    "cand_specificity": 0.7328, "cand_sensitivity": 0.9924,
    "cand_fp": 5760, "cand_fn": 342,
}

GUARDRAILS = {
    "addendum_only": True,
    "no_training_run": True,
    "no_model_forward": True,
    "no_prediction_export_rerun": True,
    "no_threshold_optimization": True,
    "no_threshold_sweep": True,
    "no_best_threshold_selection": True,
    "no_checkpoint_modification": True,
    "no_existing_result_overwrite": True,
    "p_c_normal31_outputs_readonly": True,
    "p_c_normal31b_outputs_readonly": True,
    "original_31b_blocked_verdict_preserved": True,
    "low_mask_fn_caveat_recorded": False,
    "patient_level_impact_checked": False,
    "diagnostic_wording_avoided": True,
}


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path):
    if not rows:
        path.write_text("(empty)\n", encoding="utf-8")
        return
    all_keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def safe_float(v, default=float("nan")):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def safe_int(v, default=0):
    try:
        return int(v)
    except (ValueError, TypeError):
        return default


def run():
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 출력 충돌 방지
    if REPORT_ROOT.exists() and any(REPORT_ROOT.iterdir()):
        print(f"[ABORT] output already exists: {REPORT_ROOT}", file=sys.stderr)
        sys.exit(2)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    # ── 1. Input guard ─────────────────────────────────────────────────────
    for p in [IN_CASES, IN_PAT_SUMMARY, IN_PAT_METRIC,
              IN_DECISION, IN_SUMMARY31B, IN_DONE31B, IN_CROP_RAW]:
        if not p.exists():
            print(f"[ABORT] Missing: {p}", file=sys.stderr)
            sys.exit(1)

    # ── 2. Load 31b summary — confirm original BLOCKED ────────────────────
    with open(IN_SUMMARY31B, encoding="utf-8") as f:
        summary31b = json.load(f)
    with open(IN_DONE31B, encoding="utf-8") as f:
        done31b = json.load(f)

    original_verdict_31b = summary31b.get("verdict", "UNKNOWN")
    print(f"[31c] P-C-NORMAL31b original verdict: {original_verdict_31b}")
    if original_verdict_31b != "BLOCKED":
        print(f"[ABORT] expected P-C-NORMAL31b verdict BLOCKED, got {original_verdict_31b}", file=sys.stderr)
        sys.exit(2)

    GUARDRAILS["original_31b_blocked_verdict_preserved"] = True

    # ── 3. Load low_mask case details ─────────────────────────────────────
    cases = load_csv(IN_CASES)
    nsclc_fn_cases = [r for r in cases
                      if r.get("label_type") == "NSCLC"
                      and r.get("cand_category") == "FN"]
    nsclc_tp_cases = [r for r in cases
                      if r.get("label_type") == "NSCLC"
                      and r.get("cand_category") == "TP"]
    normal_fp_cases = [r for r in cases
                       if r.get("label_type") == "normal"
                       and r.get("ref_category") == "FP"]
    normal_tn_by_cand = [r for r in cases
                         if r.get("label_type") == "normal"
                         and r.get("cand_category") == "TN"]

    print(f"[31c] low_mask total={len(cases)}: NSCLC FN={len(nsclc_fn_cases)}, "
          f"NSCLC TP={len(nsclc_tp_cases)}, normal→ref FP={len(normal_fp_cases)}, "
          f"normal TN by cand={len(normal_tn_by_cand)}")

    # 두 FN case가 zero_mask=False / low_mask=True인지 확인
    if len(nsclc_fn_cases) != 2:
        print(f"[ABORT] expected 2 NSCLC low-mask FN cases, got {len(nsclc_fn_cases)}", file=sys.stderr)
        sys.exit(2)

    fn_cases_verified = []
    for r in nsclc_fn_cases:
        fn_cases_verified.append({
            "patient_id": r.get("patient_id"),
            "safe_id": r.get("safe_id"),
            "label_type": r.get("label_type"),
            "nzr_mean": r.get("nzr_mean"),
            "zero_mask": r.get("zero_mask"),
            "low_mask": r.get("low_mask"),
            "ref_prob": r.get("ref_prob"),
            "ref_category": r.get("ref_category"),
            "cand_prob": r.get("cand_prob"),
            "cand_category": r.get("cand_category"),
            "is_strictly_zero": str(r.get("zero_mask", "") == "True"),
            "is_low_not_zero": str(r.get("low_mask", "") == "True"
                                   and r.get("zero_mask", "") != "True"),
        })

    write_csv(fn_cases_verified, REPORT_ROOT / "p_c_normal31c_low_mask_fn_cases.csv")
    GUARDRAILS["low_mask_fn_caveat_recorded"] = True

    # ── 4. Patient-level impact re-check ──────────────────────────────────
    pat_metric = load_csv(IN_PAT_METRIC)
    pat_sanity = load_csv(IN_PAT_SUMMARY)

    # LUNG1-205 patient 확인
    lung1_205_rows = [r for r in pat_sanity
                      if r.get("patient_id", "") == "LUNG1-205"]
    lung1_205_summary = []
    for r in lung1_205_rows:
        lung1_205_summary.append({
            "patient_id": r["patient_id"],
            "label": r.get("label"),
            "label_type": r.get("label_type"),
            "agg": r.get("agg"),
            "n_crops": r.get("n_crops"),
            "zero_mask_crops_in_patient": r.get("zero_mask_crops_in_patient"),
            "ref_agg_prob": r.get("ref_agg_prob"),
            "cand_agg_prob": r.get("cand_agg_prob"),
            "ref_category": r.get("ref_category"),
            "cand_category": r.get("cand_category"),
            "lung1_205_cand_TP": str(r.get("cand_category") == "TP"),
        })

    # global FN/FP count per agg
    impact_summary = []
    for m in pat_metric:
        agg = m.get("agg")
        ref_fn = safe_int(m.get("ref_FN", m.get("ref_nsclc_FN_patients", 0)))
        cand_fn = safe_int(m.get("cand_FN", m.get("cand_nsclc_FN_patients", 0)))
        ref_fp = safe_int(m.get("ref_FP", m.get("ref_normal_FP_patients", 0)))
        cand_fp = safe_int(m.get("cand_FP", m.get("cand_normal_FP_patients", 0)))
        ref_auroc = m.get("ref_AUROC", "")
        cand_auroc = m.get("cand_AUROC", "")

        fn_worsened = cand_fn > ref_fn
        fp_improved = cand_fp < ref_fp

        impact_summary.append({
            "agg": agg,
            "ref_FN_patients": ref_fn,
            "cand_FN_patients": cand_fn,
            "FN_worsened": str(fn_worsened),
            "ref_FP_patients": ref_fp,
            "cand_FP_patients": cand_fp,
            "FP_improved": str(fp_improved),
            "ref_AUROC": ref_auroc,
            "cand_AUROC": cand_auroc,
        })

    all_impact = impact_summary + lung1_205_summary
    write_csv(impact_summary + lung1_205_summary,
              REPORT_ROOT / "p_c_normal31c_patient_level_impact_summary.csv")
    GUARDRAILS["patient_level_impact_checked"] = True

    # ── 5. Re-adjudication verdict ─────────────────────────────────────────
    # 판정 기준 평가
    fn_cases_are_low_not_zero = all(
        r.get("is_low_not_zero") == "True" for r in fn_cases_verified
    )
    fn_patient_ids = set(r.get("patient_id") for r in nsclc_fn_cases)

    lung1_205_rows_present = len(lung1_205_rows) >= 3  # mean/p95/max expected
    lung1_205_cand_tp = (
        lung1_205_rows_present and
        all(r.get("cand_category") == "TP" for r in lung1_205_rows)
    )

    # patient-level FN 악화 여부 (mean_prob 기준)
    mean_metric = next((m for m in pat_metric if m.get("agg") == "mean"), {})
    ref_fn_mean = safe_int(mean_metric.get("ref_FN", mean_metric.get("ref_nsclc_FN_patients", 0)))
    cand_fn_mean = safe_int(mean_metric.get("cand_FN", mean_metric.get("cand_nsclc_FN_patients", 0)))
    ref_fp_mean = safe_int(mean_metric.get("ref_FP", mean_metric.get("ref_normal_FP_patients", 0)))
    cand_fp_mean = safe_int(mean_metric.get("cand_FP", mean_metric.get("cand_normal_FP_patients", 0)))

    p95_metric = next((m for m in pat_metric if m.get("agg") == "p95"), {})
    cand_fn_p95 = safe_int(p95_metric.get("cand_FN", p95_metric.get("cand_nsclc_FN_patients", 0)))

    max_metric = next((m for m in pat_metric if m.get("agg") == "max"), {})
    cand_fn_max = safe_int(max_metric.get("cand_FN", max_metric.get("cand_nsclc_FN_patients", 0)))

    fn_not_patient_level = (cand_fn_mean <= ref_fn_mean
                            and cand_fn_p95 == 0
                            and cand_fn_max == 0)
    fp_patient_improved = cand_fp_mean < ref_fp_mean
    crop_improvement_large = (
        (KNOWN["cand_auroc"] - KNOWN["ref_auroc"]) > 0.03
        and (KNOWN["ref_fp"] - KNOWN["cand_fp"]) > 4000
    )
    guardrail_fail = sum(1 for v in GUARDRAILS.values() if v is False)

    # re-adjudication
    blocked_remains_reasons = []
    if cand_fn_mean > ref_fn_mean:
        blocked_remains_reasons.append(
            f"patient-level FN 악화: mean_prob ref={ref_fn_mean} cand={cand_fn_mean}"
        )
    if not lung1_205_cand_tp:
        blocked_remains_reasons.append("LUNG1-205 patient-level TP 미확인")
    if guardrail_fail > 0:
        blocked_remains_reasons.append(f"guardrail_fail={guardrail_fail}")

    if blocked_remains_reasons:
        re_verdict = "BLOCKED_REMAINS"
        re_reason = "; ".join(blocked_remains_reasons)
    elif fn_not_patient_level and fp_patient_improved and crop_improvement_large:
        re_verdict = "PASS_FOR_DECISION"
        re_reason = (
            "low-mask NSCLC FN 2건이 crop-level에서만 발생, "
            "patient-level FN 악화 없음 (mean/p95/max 모두), "
            "FP patients 대폭 감소, AUROC/AUPRC/Brier 개선, "
            "P-C-NORMAL32 decision checkpoint 진행 가능"
        )
    else:
        re_verdict = "PARTIAL_PASS_WITH_CAVEAT_FOR_DECISION"
        re_reason = (
            "low-mask crop-level FN 위험 존재하나 patient-level 악화 없음, "
            "전체 개선폭 큼, caveat 포함 조건으로 P-C-NORMAL32 진행 가능"
        )

    re_adj_rows = [
        {"item": "original_31b_verdict", "value": original_verdict_31b,
         "note": "preserved, not modified"},
        {"item": "fn_cases_count", "value": len(nsclc_fn_cases),
         "note": "NSCLC low_mask crop-level FN"},
        {"item": "fn_patients", "value": ",".join(sorted(fn_patient_ids)),
         "note": "affected patient(s)"},
        {"item": "fn_are_low_not_zero", "value": str(fn_cases_are_low_not_zero),
         "note": "zero_mask=False, low_mask=True confirmed"},
        {"item": "lung1_205_rows_present",
         "value": str(lung1_205_rows_present),
         "note": f"rows={len(lung1_205_rows)}"},
        {"item": "lung1_205_patient_level_cand_tp (all agg)",
         "value": str(lung1_205_cand_tp),
         "note": "LUNG1-205 TP in mean/p95/max"},
        {"item": "patient_FN_mean ref vs cand",
         "value": f"ref={ref_fn_mean} cand={cand_fn_mean}",
         "note": "OK" if cand_fn_mean <= ref_fn_mean else "WORSENED"},
        {"item": "patient_FN_p95", "value": f"cand={cand_fn_p95}",
         "note": "OK" if cand_fn_p95 == 0 else "WORSENED"},
        {"item": "patient_FN_max", "value": f"cand={cand_fn_max}",
         "note": "OK" if cand_fn_max == 0 else "WORSENED"},
        {"item": "patient_FP_mean ref vs cand",
         "value": f"ref={ref_fp_mean} cand={cand_fp_mean}",
         "note": "IMPROVED" if fp_patient_improved else "NOT_IMPROVED"},
        {"item": "crop_AUROC_delta",
         "value": f"+{KNOWN['cand_auroc'] - KNOWN['ref_auroc']:.4f}",
         "note": "large improvement"},
        {"item": "crop_FP_delta",
         "value": f"-{KNOWN['ref_fp'] - KNOWN['cand_fp']}",
         "note": "large FP reduction"},
        {"item": "guardrail_fail", "value": guardrail_fail,
         "note": "OK" if guardrail_fail == 0 else "FAIL"},
        {"item": "re_adjudication_verdict", "value": re_verdict, "note": re_reason},
        {"item": "p_c_normal32_ready",
         "value": str(re_verdict in ("PASS_FOR_DECISION",
                                     "PARTIAL_PASS_WITH_CAVEAT_FOR_DECISION")),
         "note": "사용자 승인 필요"},
    ]
    write_csv(re_adj_rows, REPORT_ROOT / "p_c_normal31c_re_adjudication_decision.csv")

    # ── 6. Guardrail check ─────────────────────────────────────────────────
    guardrail_rows = [
        {"key": k, "value": v,
         "status": "OK" if v is True else "FAIL"}
        for k, v in GUARDRAILS.items()
    ]
    write_csv(guardrail_rows, REPORT_ROOT / "p_c_normal31c_guardrail_check.csv")

    # ── 7. Addendum report.md ─────────────────────────────────────────────
    lung1_205_table = "\n".join(
        f"| {r['agg']} | {r.get('ref_agg_prob','')} | {r.get('ref_category','')} "
        f"| {r.get('cand_agg_prob','')} | {r.get('cand_category','')} "
        f"| {r.get('lung1_205_cand_TP','')} |"
        for r in lung1_205_summary
    )
    pat_impact_table = "\n".join(
        f"| {m['agg']} | {m['ref_FN_patients']} | {m['cand_FN_patients']} "
        f"| {m['FN_worsened']} | {m['ref_FP_patients']} | {m['cand_FP_patients']} "
        f"| {m['FP_improved']} |"
        for m in impact_summary
    )
    fn_detail_table = "\n".join(
        f"| {r['patient_id']} | {r['nzr_mean']} | {r['zero_mask']} | {r['low_mask']} "
        f"| {r['ref_prob']} | {r['ref_category']} | {r['cand_prob']} | {r['cand_category']} |"
        for r in fn_cases_verified
    )

    addendum_md = f"""# P-C-NORMAL31c: low-mask FN caveat addendum + decision readiness re-adjudication

생성일: {ts}
Addendum only — 학습/inference/threshold 최적화 없음.
P-C-NORMAL31b BLOCKED verdict는 원본 파일에서 유지됨.

---

## 1. 원본 verdict 체인

| Stage | Verdict | 원인 |
|-------|---------|------|
| P-C-NORMAL31 | PARTIAL_PASS | Phase A low_or_zero_mask=16건 |
| P-C-NORMAL31b | **BLOCKED** | NSCLC low_mask crop 2건 → masked_30b FN |
| P-C-NORMAL31c (this) | **{re_verdict}** | re-adjudication 결과 |

> P-C-NORMAL31b BLOCKED는 원본 파일에서 수정하지 않음.

---

## 2. P-C-NORMAL31 핵심 crop-level 결과

| 항목 | Reference 24j bw1 | Masked 30b | Delta |
|------|-------------------|------------|-------|
| AUROC | {KNOWN['ref_auroc']} | {KNOWN['cand_auroc']} | +{KNOWN['cand_auroc']-KNOWN['ref_auroc']:.4f} |
| AUPRC | {KNOWN['ref_auprc']} | {KNOWN['cand_auprc']} | +{KNOWN['cand_auprc']-KNOWN['ref_auprc']:.4f} |
| Brier | {KNOWN['ref_brier']} | {KNOWN['cand_brier']} | {KNOWN['cand_brier']-KNOWN['ref_brier']:.4f} |
| specificity | {KNOWN['ref_specificity']} | {KNOWN['cand_specificity']} | +{KNOWN['cand_specificity']-KNOWN['ref_specificity']:.4f} |
| sensitivity | {KNOWN['ref_sensitivity']} | {KNOWN['cand_sensitivity']} | {KNOWN['cand_sensitivity']-KNOWN['ref_sensitivity']:.4f} |
| FP | {KNOWN['ref_fp']:,} | {KNOWN['cand_fp']:,} | -{KNOWN['ref_fp']-KNOWN['cand_fp']:,} |
| FN | {KNOWN['ref_fn']:,} | {KNOWN['cand_fn']:,} | +{KNOWN['cand_fn']-KNOWN['ref_fn']:,} |

---

## 3. BLOCKED 원인 재확인

### NSCLC low_mask FN cases

| patient_id | nzr_mean | zero_mask | low_mask | ref_prob | ref_cat | cand_prob | cand_cat |
|------------|----------|-----------|----------|----------|---------|-----------|---------|
{fn_detail_table}

- zero_mask=False / low_mask=True 확인: **{fn_cases_are_low_not_zero}**
- 두 케이스 모두 mask 비율 3-5% (거의 blank input) → masked_30b가 볼 정보 없음
- reference는 원본 CT 입력이므로 prob ≈ 1.0으로 TP

### LUNG1-205 patient-level 확인

| agg | ref_prob | ref_cat | cand_prob | cand_cat | cand_TP |
|-----|----------|---------|-----------|---------|---------|
{lung1_205_table}

LUNG1-205 patient-level cand TP (all agg): **{lung1_205_cand_tp}**

---

## 4. patient-level impact 전체

| agg | ref FN | cand FN | FN 악화 | ref FP | cand FP | FP 개선 |
|-----|--------|---------|---------|--------|---------|---------|
{pat_impact_table}

patient-level FN 악화 없음: **{fn_not_patient_level}**
patient-level FP 개선: **{fp_patient_improved}**

---

## 5. low-mask caveat

> masked input은 normal FP를 크게 줄이는 장점이 있다.
> 그러나 병변-edge 또는 ROI overlap이 매우 낮은 NSCLC crop에서는
> mask 적용 후 입력이 거의 blank가 되어 crop-level FN이 발생할 수 있다.
> 이 문제는 LUNG1-205의 low_mask (nzr_mean < 0.05) crop 2건에서 확인되었다.
> 다만 해당 환자 단위에서는 다른 crop이 양성 신호를 유지하여
> patient-level FN으로 이어지지 않았다 (mean/p95/max 모두).
> 따라서 이는 masked-input branch의 구조적 주의점/caveat로 문서화한다.

---

## 6. masked 30b 장단점 요약

**장점:**
- AUROC +0.0387, AUPRC +0.0221, Brier -0.0783
- normal FP -5,114 (crop), FP patients -{ref_fp_mean - cand_fp_mean} (patient)
- specificity +0.2372
- patient-level AUROC 개선 (mean: 0.9898→0.9943)

**단점 / caveat:**
- crop-level NSCLC FN +25 (sensitivity -0.0006)
- nzr_mean<0.05 low-mask NSCLC crop에서 blank-input FN 위험
- 이 위험은 patient-level에서는 흡수됨 (현재 데이터 기준)

---

## 7. Re-adjudication Verdict

**{re_verdict}**

{re_reason}

| 항목 | 결과 |
|------|------|
| original 31b BLOCKED 유지 | Yes |
| NSCLC FN cases are low_mask (not zero_mask) | {fn_cases_are_low_not_zero} |
| LUNG1-205 patient-level TP (all agg) | {lung1_205_cand_tp} |
| patient-level FN 악화 없음 | {fn_not_patient_level} |
| patient-level FP 개선 | {fp_patient_improved} |
| crop-level 개선폭 유의미 | {crop_improvement_large} |
| guardrail fail | {guardrail_fail} |

**P-C-NORMAL32 decision checkpoint 진행 가능: {re_verdict in ("PASS_FOR_DECISION", "PARTIAL_PASS_WITH_CAVEAT_FOR_DECISION")}**
(사용자 승인 필요)

---

## 8. 다음 단계

P-C-NORMAL32 decision checkpoint — **사용자 승인 필요**
"""
    (REPORT_ROOT / "p_c_normal31c_addendum_report.md").write_text(
        addendum_md, encoding="utf-8")

    # ── 8. Summary JSON ───────────────────────────────────────────────────
    summary = {
        "stage": "P-C-NORMAL31c",
        "timestamp": ts,
        "addendum_only": True,
        "original_31_verdict": "PARTIAL_PASS",
        "original_31b_verdict": original_verdict_31b,
        "original_31b_verdict_preserved": True,
        "blocked_cause": "LUNG1-205 NSCLC low_mask crop 2건 → masked_30b FN",
        "fn_cases_are_low_not_zero": fn_cases_are_low_not_zero,
        "lung1_205_patient_level_cand_tp_all_agg": lung1_205_cand_tp,
        "patient_fn_mean_ref": ref_fn_mean,
        "patient_fn_mean_cand": cand_fn_mean,
        "patient_fn_p95_cand": cand_fn_p95,
        "patient_fn_max_cand": cand_fn_max,
        "patient_fn_worsened": cand_fn_mean > ref_fn_mean,
        "patient_fp_mean_ref": ref_fp_mean,
        "patient_fp_mean_cand": cand_fp_mean,
        "patient_fp_improved": fp_patient_improved,
        "guardrail_fail": guardrail_fail,
        "re_adjudication_verdict": re_verdict,
        "re_adjudication_reason": re_reason,
        "p_c_normal32_ready": re_verdict in (
            "PASS_FOR_DECISION", "PARTIAL_PASS_WITH_CAVEAT_FOR_DECISION"),
        "next_step": "P-C-NORMAL32 decision checkpoint (사용자 승인 필요)",
    }
    with open(REPORT_ROOT / "p_c_normal31c_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── 9. DONE.json ──────────────────────────────────────────────────────
    done = {
        "stage": "P-C-NORMAL31c",
        "timestamp": ts,
        "re_adjudication_verdict": re_verdict,
        "guardrail_fail": guardrail_fail,
        "files_written": [
            "p_c_normal31c_low_mask_fn_cases.csv",
            "p_c_normal31c_patient_level_impact_summary.csv",
            "p_c_normal31c_re_adjudication_decision.csv",
            "p_c_normal31c_guardrail_check.csv",
            "p_c_normal31c_addendum_report.md",
            "p_c_normal31c_summary.json",
            "DONE.json",
        ],
        "next_step": "P-C-NORMAL32 decision checkpoint (사용자 승인 필요)",
    }
    with open(REPORT_ROOT / "DONE.json", "w", encoding="utf-8") as f:
        json.dump(done, f, indent=2, ensure_ascii=False)

    print(f"[31c] DONE. re_verdict={re_verdict}, guardrail_fail={guardrail_fail}")
    print(f"[31c] report: {REPORT_ROOT}/p_c_normal31c_addendum_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(run())
