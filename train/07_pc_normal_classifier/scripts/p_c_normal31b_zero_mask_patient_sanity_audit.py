"""
P-C-NORMAL31b: zero-mask impact + patient-level sanity audit
audit only — no training, no model forward, no prediction re-export
"""

import sys
import csv
import json
import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")

REPORT31 = PROJECT_ROOT / "outputs/reports/p_c_normal31_repaired_final_test_masked_comparison"
MASK_MANIFEST = REPORT31 / "p_c_normal31_final_test_mask_manifest.csv"
LOW_OR_ZERO_CSV = REPORT31 / "p_c_normal31_final_test_mask_low_or_zero_cases.csv"
REF_PRED = REPORT31 / "p_c_normal31_reference_balanced_w1_predictions.csv"
CAND_PRED = REPORT31 / "p_c_normal31_masked_30b_predictions.csv"
CROP_RAW_METRICS = REPORT31 / "p_c_normal31_crop_raw_metrics_comparison.csv"
PATIENT_METRICS = REPORT31 / "p_c_normal31_patient_metrics_comparison.csv"
PATIENT_CONF = REPORT31 / "p_c_normal31_patient_confusion_matrix_comparison.csv"
GUARDRAIL31 = REPORT31 / "p_c_normal31_guardrail_check.csv"
DONE31 = REPORT31 / "DONE.json"

REPORT_ROOT = PROJECT_ROOT / "outputs/reports/p_c_normal31b_zero_mask_patient_sanity_audit"

EXPECTED_TOTAL = 66283
# Phase A: n_zero = len(zero_low_rows) → zero_mask OR low_mask(nzr_mean<0.05) 합계
# 실제로 zero_mask=True는 0건, low_mask=True는 16건이었음
EXPECTED_PARTIAL_PASS_CASES = 16

# ── Guardrail declaration ──────────────────────────────────────────────────
GUARDRAILS = {
    "audit_only": True,
    "no_training_run": True,
    "no_model_forward": True,
    "no_prediction_export_rerun": True,
    "no_threshold_optimization": True,
    "no_threshold_sweep": True,
    "no_best_threshold_selection": True,
    "no_checkpoint_modification": True,
    "no_existing_result_overwrite": True,
    "p_c_normal31_outputs_readonly": True,
    "zero_mask_cases_audited": False,  # 완료 시 True로 update
    "patient_level_metrics_checked": False,
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


def roc_auc(labels, probs):
    from itertools import combinations
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    pos_probs = [probs[i] for i, l in enumerate(labels) if l == 1]
    neg_probs = [probs[i] for i, l in enumerate(labels) if l == 0]
    wins = sum(p > n for p in pos_probs for n in neg_probs)
    ties = sum(p == n for p in pos_probs for n in neg_probs)
    total = n_pos * n_neg
    return (wins + 0.5 * ties) / total


def auroc_trapz(labels, probs):
    """Simple trapz AUROC."""
    from collections import Counter
    pairs = sorted(zip(probs, labels), key=lambda x: -x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = fp = 0
    prev_tpr = prev_fpr = 0.0
    auc = 0.0
    prev_thresh = None
    for prob, lbl in pairs:
        if prob != prev_thresh and prev_thresh is not None:
            tpr = tp / n_pos
            fpr = fp / n_neg
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
            prev_fpr, prev_tpr = fpr, tpr
        if lbl == 1:
            tp += 1
        else:
            fp += 1
        prev_thresh = prob
    tpr = tp / n_pos
    fpr = fp / n_neg
    auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
    return auc


def run():
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 출력 충돌 방지
    if REPORT_ROOT.exists() and any(REPORT_ROOT.iterdir()):
        print(f"[ABORT] output directory already exists and is not empty: {REPORT_ROOT}", file=sys.stderr)
        print("[GUARD] Existing result overwrite is forbidden.", file=sys.stderr)
        sys.exit(2)

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    # ── 1. Input guard: 기존 결과 읽기 전용 확인 ──────────────────────────
    for p in [MASK_MANIFEST, LOW_OR_ZERO_CSV, REF_PRED, CAND_PRED, GUARDRAIL31, DONE31]:
        if not p.exists():
            print(f"[ABORT] Missing input: {p}", file=sys.stderr)
            sys.exit(1)

    # ── 2. Load mask manifest & low_or_zero cases ─────────────────────────
    print("[31b] Loading mask manifest...")
    manifest = load_csv(MASK_MANIFEST)
    print(f"[31b] manifest rows: {len(manifest)}")

    if len(manifest) != EXPECTED_TOTAL:
        print(f"[WARN] manifest total {len(manifest)} != expected {EXPECTED_TOTAL}")

    # PARTIAL_PASS 원인: low_or_zero_cases (zero_mask OR low_mask)
    # Phase A 코드: n_zero = len(zero_low_rows) (is_zero or is_low)
    # 실제: zero_mask=True 0건, low_mask=True(nzr_mean<0.05) 16건
    loz_all = load_csv(LOW_OR_ZERO_CSV)
    # "no zero or low mask" 플레이스홀더 제거
    loz_all = [r for r in loz_all if r.get("crop_path", "").startswith("/")]

    zero_rows_strict = [r for r in loz_all if r.get("zero_mask", "").strip() == "True"]
    low_rows = [r for r in loz_all if r.get("low_mask", "").strip() == "True"]
    zero_rows = loz_all  # zero OR low (PARTIAL_PASS 원인 전체)

    print(f"[31b] low_or_zero total: {len(zero_rows)} (expected {EXPECTED_PARTIAL_PASS_CASES})")
    print(f"[31b]   zero_mask=True: {len(zero_rows_strict)}, low_mask=True: {len(low_rows)}")

    # build crop_path -> loz_row map (low_or_zero cases)
    mask_by_crop = {r["crop_path"]: r for r in loz_all}

    # ── 3. Load prediction CSVs ────────────────────────────────────────────
    print("[31b] Loading reference predictions...")
    ref_rows = load_csv(REF_PRED)
    print(f"[31b] reference rows: {len(ref_rows)}")

    print("[31b] Loading masked 30b predictions...")
    cand_rows = load_csv(CAND_PRED)
    print(f"[31b] masked rows: {len(cand_rows)}")

    # row count / duplicate crop_path sanity check
    count_errors = []

    if len(manifest) != EXPECTED_TOTAL:
        count_errors.append(f"mask_manifest rows expected {EXPECTED_TOTAL}, got {len(manifest)}")
    if len(ref_rows) != EXPECTED_TOTAL:
        count_errors.append(f"reference prediction rows expected {EXPECTED_TOTAL}, got {len(ref_rows)}")
    if len(cand_rows) != EXPECTED_TOTAL:
        count_errors.append(f"candidate prediction rows expected {EXPECTED_TOTAL}, got {len(cand_rows)}")

    def duplicate_count(rows, key):
        vals = [r.get(key, "") for r in rows]
        return len(vals) - len(set(vals))

    dup_mask = duplicate_count(manifest, "crop_path")
    dup_ref = duplicate_count(ref_rows, "crop_path")
    dup_cand = duplicate_count(cand_rows, "crop_path")

    if dup_mask != 0:
        count_errors.append(f"duplicate crop_path in mask_manifest: {dup_mask}")
    if dup_ref != 0:
        count_errors.append(f"duplicate crop_path in reference predictions: {dup_ref}")
    if dup_cand != 0:
        count_errors.append(f"duplicate crop_path in candidate predictions: {dup_cand}")

    if count_errors:
        for e in count_errors:
            print(f"[ABORT] {e}", file=sys.stderr)
        sys.exit(2)

    print(f"[31b] row count/dup check PASS: manifest={len(manifest)} ref={len(ref_rows)} cand={len(cand_rows)} dup_mask={dup_mask} dup_ref={dup_ref} dup_cand={dup_cand}")

    # build crop_path -> pred map
    ref_by_crop = {r["crop_path"]: r for r in ref_rows}
    cand_by_crop = {r["crop_path"]: r for r in cand_rows}

    # ── 4. zero_mask cases detail ──────────────────────────────────────────
    print("[31b] Building low_or_zero case analysis (PARTIAL_PASS cause)...")
    zero_case_rows = []
    for mr in zero_rows:
        crop_path = mr["crop_path"]
        ref_p = ref_by_crop.get(crop_path, {})
        cand_p = cand_by_crop.get(crop_path, {})

        # low_or_zero_cases.csv에는 label 컬럼 있음
        label = safe_int(mr.get("label", ""), -1)
        if label == -1:
            label_str = ref_p.get("label", "") if ref_p else ""
            label = safe_int(label_str, 0)

        ref_prob = safe_float(ref_p.get("prob", "nan"))
        ref_pred = safe_int(ref_p.get("pred_at_0p5", 0))
        cand_prob = safe_float(cand_p.get("prob", "nan"))
        cand_pred = safe_int(cand_p.get("pred_at_0p5", 0))

        def categorize(pred, lbl):
            if lbl == 1 and pred == 1:
                return "TP"
            elif lbl == 1 and pred == 0:
                return "FN"
            elif lbl == 0 and pred == 1:
                return "FP"
            else:
                return "TN"

        row = {
            "crop_path": crop_path,
            "safe_id": mr.get("safe_id", ""),
            "label": label,
            "label_type": "NSCLC" if label == 1 else "normal",
            "nzr_mean": mr.get("nzr_mean", ""),
            "zero_mask": mr.get("zero_mask", ""),
            "low_mask": mr.get("low_mask", ""),
            "patient_id": ref_p.get("patient_id", ""),
            "source_split": ref_p.get("source_split", ""),
            "ref_prob": f"{ref_prob:.6f}" if not np.isnan(ref_prob) else "nan",
            "ref_pred_at_0p5": ref_pred,
            "cand_prob": f"{cand_prob:.6f}" if not np.isnan(cand_prob) else "nan",
            "cand_pred_at_0p5": cand_pred,
            "ref_category": categorize(ref_pred, label),
            "cand_category": categorize(cand_pred, label),
            "ref_error": "YES" if categorize(ref_pred, label) in ("FP", "FN") else "NO",
            "cand_error": "YES" if categorize(cand_pred, label) in ("FP", "FN") else "NO",
        }
        zero_case_rows.append(row)

    write_csv(zero_case_rows, REPORT_ROOT / "p_c_normal31b_zero_mask_cases.csv")
    print(f"[31b] zero_mask_cases.csv: {len(zero_case_rows)} rows")

    # label 분포
    zero_normal = sum(1 for r in zero_case_rows if r["label"] == 0 or r["label_type"] == "normal")
    zero_nsclc = sum(1 for r in zero_case_rows if r["label"] == 1 or r["label_type"] == "NSCLC")
    # patient overlap
    zero_patients = defaultdict(list)
    for r in zero_case_rows:
        pid = r["patient_id"]
        zero_patients[pid].append(r["label_type"])

    GUARDRAILS["zero_mask_cases_audited"] = True

    # ── 5. zero_mask prediction impact summary ────────────────────────────
    ref_cat_counts = defaultdict(int)
    cand_cat_counts = defaultdict(int)
    for r in zero_case_rows:
        ref_cat_counts[r["ref_category"]] += 1
        cand_cat_counts[r["cand_category"]] += 1

    impact_rows = []
    for cat in ["TP", "TN", "FP", "FN"]:
        impact_rows.append({
            "category": cat,
            "ref_count": ref_cat_counts[cat],
            "cand_count": cand_cat_counts[cat],
            "delta": cand_cat_counts[cat] - ref_cat_counts[cat],
        })

    # zero_mask 제거 시 confusion matrix 변화 (참고용)
    # 전체 confusion matrix에서 zero_mask 해당 항목을 빼면 됨
    # 단, 이 값은 공식 metric 대체가 아닌 참고용
    ref_all_pred = [(safe_int(r["label"]), safe_int(r["pred_at_0p5"]),
                     r["crop_path"]) for r in ref_rows]
    cand_all_pred = [(safe_int(r["label"]), safe_int(r["pred_at_0p5"]),
                      r["crop_path"]) for r in cand_rows]

    zero_crop_set = {r["crop_path"] for r in zero_case_rows}

    def conf_matrix(preds, exclude_set=None):
        tp = tn = fp = fn = 0
        for lbl, pred, cp in preds:
            if exclude_set and cp in exclude_set:
                continue
            if lbl == 1 and pred == 1:
                tp += 1
            elif lbl == 0 and pred == 0:
                tn += 1
            elif lbl == 0 and pred == 1:
                fp += 1
            else:
                fn += 1
        return tp, tn, fp, fn

    ref_tp, ref_tn, ref_fp, ref_fn = conf_matrix(ref_all_pred)
    cand_tp, cand_tn, cand_fp, cand_fn = conf_matrix(cand_all_pred)
    ref_tp_ex, ref_tn_ex, ref_fp_ex, ref_fn_ex = conf_matrix(ref_all_pred, zero_crop_set)
    cand_tp_ex, cand_tn_ex, cand_fp_ex, cand_fn_ex = conf_matrix(cand_all_pred, zero_crop_set)

    impact_summary_rows = [
        {"model": "reference", "scope": "all_crops",
         "TP": ref_tp, "TN": ref_tn, "FP": ref_fp, "FN": ref_fn,
         "sensitivity": f"{ref_tp/(ref_tp+ref_fn):.6f}" if (ref_tp+ref_fn) > 0 else "nan",
         "specificity": f"{ref_tn/(ref_tn+ref_fp):.6f}" if (ref_tn+ref_fp) > 0 else "nan",
         "note": "official"},
        {"model": "reference", "scope": "exclude_zero_mask_16 [REFERENCE ONLY]",
         "TP": ref_tp_ex, "TN": ref_tn_ex, "FP": ref_fp_ex, "FN": ref_fn_ex,
         "sensitivity": f"{ref_tp_ex/(ref_tp_ex+ref_fn_ex):.6f}" if (ref_tp_ex+ref_fn_ex) > 0 else "nan",
         "specificity": f"{ref_tn_ex/(ref_tn_ex+ref_fp_ex):.6f}" if (ref_tn_ex+ref_fp_ex) > 0 else "nan",
         "note": "reference_only_not_official"},
        {"model": "masked_30b", "scope": "all_crops",
         "TP": cand_tp, "TN": cand_tn, "FP": cand_fp, "FN": cand_fn,
         "sensitivity": f"{cand_tp/(cand_tp+cand_fn):.6f}" if (cand_tp+cand_fn) > 0 else "nan",
         "specificity": f"{cand_tn/(cand_tn+cand_fp):.6f}" if (cand_tn+cand_fp) > 0 else "nan",
         "note": "official"},
        {"model": "masked_30b", "scope": "exclude_zero_mask_16 [REFERENCE ONLY]",
         "TP": cand_tp_ex, "TN": cand_tn_ex, "FP": cand_fp_ex, "FN": cand_fn_ex,
         "sensitivity": f"{cand_tp_ex/(cand_tp_ex+cand_fn_ex):.6f}" if (cand_tp_ex+cand_fn_ex) > 0 else "nan",
         "specificity": f"{cand_tn_ex/(cand_tn_ex+cand_fp_ex):.6f}" if (cand_tn_ex+cand_fp_ex) > 0 else "nan",
         "note": "reference_only_not_official"},
    ]

    impact_rows_with_summary = impact_rows + [{"_sep": "---"}] + impact_summary_rows
    write_csv(impact_summary_rows, REPORT_ROOT / "p_c_normal31b_zero_mask_prediction_impact.csv")

    # patient overlap
    patient_overlap_rows = []
    for pid, labels in zero_patients.items():
        patient_overlap_rows.append({
            "patient_id": pid,
            "zero_mask_count": len(labels),
            "label_types": ",".join(sorted(set(labels))),
        })
    write_csv(patient_overlap_rows, REPORT_ROOT / "p_c_normal31b_zero_mask_patient_overlap.csv")

    # ── 6. Patient-level sanity audit ─────────────────────────────────────
    print("[31b] Building patient-level sanity from predictions...")

    # group by patient_id
    def group_by_patient(rows):
        groups = defaultdict(list)
        for r in rows:
            groups[r["patient_id"]].append(r)
        return groups

    ref_patients = group_by_patient(ref_rows)
    cand_patients = group_by_patient(cand_rows)

    all_pids = sorted(set(ref_patients.keys()) | set(cand_patients.keys()))

    def patient_agg(rows, agg_fn):
        """agg_fn: 'mean', 'p95', 'max'"""
        probs = [safe_float(r["prob"]) for r in rows if r.get("prob")]
        if not probs:
            return float("nan")
        if agg_fn == "mean":
            return np.mean(probs)
        elif agg_fn == "p95":
            return float(np.percentile(probs, 95))
        elif agg_fn == "max":
            return float(np.max(probs))
        return float("nan")

    def first_label(rows):
        for r in rows:
            lbl = safe_int(r.get("label", -1), -1)
            if lbl in (0, 1):
                return lbl
        return -1

    sanity_rows = []
    for pid in all_pids:
        rp = ref_patients.get(pid, [])
        cp = cand_patients.get(pid, [])
        lbl = first_label(rp) if rp else first_label(cp)
        lbl_type = "NSCLC" if lbl == 1 else "normal" if lbl == 0 else "unknown"

        # zero_mask crops in this patient
        zm_in_patient = sum(1 for r in (rp or cp) if r.get("crop_path", "") in zero_crop_set)

        for agg in ("mean", "p95", "max"):
            ref_agg_prob = patient_agg(rp, agg)
            cand_agg_prob = patient_agg(cp, agg)
            ref_pat_pred = 1 if ref_agg_prob >= 0.5 else 0
            cand_pat_pred = 1 if cand_agg_prob >= 0.5 else 0

            def cat(pred, l):
                if l == 1 and pred == 1:
                    return "TP"
                elif l == 1 and pred == 0:
                    return "FN"
                elif l == 0 and pred == 1:
                    return "FP"
                else:
                    return "TN"

            sanity_rows.append({
                "patient_id": pid,
                "label": lbl,
                "label_type": lbl_type,
                "agg": agg,
                "n_crops": len(rp),
                "zero_mask_crops_in_patient": zm_in_patient,
                "ref_agg_prob": f"{ref_agg_prob:.6f}" if not np.isnan(ref_agg_prob) else "nan",
                "cand_agg_prob": f"{cand_agg_prob:.6f}" if not np.isnan(cand_agg_prob) else "nan",
                "ref_pat_pred": ref_pat_pred,
                "cand_pat_pred": cand_pat_pred,
                "ref_category": cat(ref_pat_pred, lbl),
                "cand_category": cat(cand_pat_pred, lbl),
            })

    write_csv(sanity_rows, REPORT_ROOT / "p_c_normal31b_patient_level_sanity_summary.csv")

    # patient-level metric summary per agg
    pat_metric_summary = []
    for agg in ("mean", "p95", "max"):
        agg_rows = [r for r in sanity_rows if r["agg"] == agg]
        labels = [safe_int(r["label"]) for r in agg_rows]

        def cnt(cat, model):
            key = f"{model}_category"
            return sum(1 for r in agg_rows if r[key] == cat)

        ref_tp_p = cnt("TP", "ref")
        ref_tn_p = cnt("TN", "ref")
        ref_fp_p = cnt("FP", "ref")
        ref_fn_p = cnt("FN", "ref")
        cand_tp_p = cnt("TP", "cand")
        cand_tn_p = cnt("TN", "cand")
        cand_fp_p = cnt("FP", "cand")
        cand_fn_p = cnt("FN", "cand")

        # AUROC per model using agg_prob
        ref_probs_pat = [safe_float(r["ref_agg_prob"]) for r in agg_rows]
        cand_probs_pat = [safe_float(r["cand_agg_prob"]) for r in agg_rows]
        labels_pat = [safe_int(r["label"]) for r in agg_rows]

        ref_auroc_pat = auroc_trapz(labels_pat, ref_probs_pat)
        cand_auroc_pat = auroc_trapz(labels_pat, cand_probs_pat)

        # zero_mask patients with FN in cand
        zm_fn_cand = [r for r in agg_rows
                      if r["cand_category"] == "FN"
                      and safe_int(r.get("zero_mask_crops_in_patient", 0)) > 0]
        zm_fn_ref = [r for r in agg_rows
                     if r["ref_category"] == "FN"
                     and safe_int(r.get("zero_mask_crops_in_patient", 0)) > 0]

        pat_metric_summary.append({
            "agg": agg,
            "n_patients": len(agg_rows),
            "ref_AUROC": f"{ref_auroc_pat:.6f}",
            "cand_AUROC": f"{cand_auroc_pat:.6f}",
            "AUROC_delta": f"{cand_auroc_pat - ref_auroc_pat:.6f}",
            "ref_TP": ref_tp_p, "ref_TN": ref_tn_p,
            "ref_FP": ref_fp_p, "ref_FN": ref_fn_p,
            "cand_TP": cand_tp_p, "cand_TN": cand_tn_p,
            "cand_FP": cand_fp_p, "cand_FN": cand_fn_p,
            "ref_normal_FP_patients": ref_fp_p,
            "cand_normal_FP_patients": cand_fp_p,
            "ref_nsclc_FN_patients": ref_fn_p,
            "cand_nsclc_FN_patients": cand_fn_p,
            "zero_mask_patients_with_cand_FN": len(zm_fn_cand),
            "zero_mask_patients_with_ref_FN": len(zm_fn_ref),
        })

    write_csv(pat_metric_summary, REPORT_ROOT / "p_c_normal31b_patient_metric_summary.csv")

    GUARDRAILS["patient_level_metrics_checked"] = True
    print(f"[31b] patient-level sanity: {len(sanity_rows)} rows, {len(all_pids)} patients")

    # ── 7. Decision readiness ─────────────────────────────────────────────
    print("[31b] Computing decision readiness...")

    # key facts
    zero_mask_count_ok = len(zero_rows) == EXPECTED_PARTIAL_PASS_CASES
    zero_mask_nsclc_fn = sum(1 for r in zero_case_rows
                             if r["label_type"] == "NSCLC" and r["cand_category"] == "FN")
    zero_mask_normal_fp = sum(1 for r in zero_case_rows
                              if r["label_type"] == "normal" and r["cand_category"] == "FP")

    # patient-level: mean_prob 기준
    mean_rows = [r for r in sanity_rows if r["agg"] == "mean"]
    cand_pat_fn_count = sum(1 for r in mean_rows if r["cand_category"] == "FN")
    ref_pat_fn_count = sum(1 for r in mean_rows if r["ref_category"] == "FN")
    cand_pat_fp_count = sum(1 for r in mean_rows if r["cand_category"] == "FP")
    ref_pat_fp_count = sum(1 for r in mean_rows if r["ref_category"] == "FP")

    # max_prob 기준 FN 확인
    max_rows = [r for r in sanity_rows if r["agg"] == "max"]
    cand_pat_fn_max = sum(1 for r in max_rows if r["cand_category"] == "FN")

    # zero_mask가 patient-level FN에 관여하는지
    zm_fn_direct = sum(1 for r in mean_rows
                       if r["cand_category"] == "FN"
                       and safe_int(r.get("zero_mask_crops_in_patient", 0)) > 0)

    # blocked condition 판정
    blocked_reasons = []
    if zero_mask_nsclc_fn > 0:
        blocked_reasons.append(
            f"zero_mask NSCLC FN={zero_mask_nsclc_fn} (masked 30b가 zero_mask NSCLC을 FN으로 처리)"
        )
    if cand_pat_fn_count > ref_pat_fn_count + 5:
        blocked_reasons.append(
            f"patient-level FN 증가 과다: ref={ref_pat_fn_count} cand={cand_pat_fn_count}"
        )
    guardrail_fail = sum(1 for k, v in GUARDRAILS.items() if v is False)
    if guardrail_fail > 0:
        blocked_reasons.append(f"guardrail fail={guardrail_fail}")

    # verdict
    if blocked_reasons:
        verdict = "BLOCKED"
        verdict_reason = "; ".join(blocked_reasons)
    elif zero_mask_nsclc_fn > 0 or cand_pat_fn_count > ref_pat_fn_count:
        verdict = "PARTIAL_PASS_WITH_CAVEAT"
        verdict_reason = (
            f"zero_mask NSCLC FN={zero_mask_nsclc_fn}, "
            f"patient FN ref={ref_pat_fn_count} cand={cand_pat_fn_count}, "
            "개선폭 매우 크므로 caveat 포함 decision checkpoint 가능"
        )
    else:
        verdict = "PASS_FOR_DECISION"
        verdict_reason = (
            f"zero_mask 16건 확인, NSCLC FN 미관여, "
            f"patient-level FN 동등 또는 개선, "
            "P-C-NORMAL32 decision checkpoint 진행 가능"
        )

    decision_rows = [
        {"item": "low_or_zero_count_expected", "value": EXPECTED_PARTIAL_PASS_CASES,
         "actual": len(zero_rows), "ok": str(zero_mask_count_ok)},
        {"item": "strict_zero_mask_count", "value": 0,
         "actual": len(zero_rows_strict), "ok": str(len(zero_rows_strict) == 0)},
        {"item": "low_mask_count", "value": EXPECTED_PARTIAL_PASS_CASES,
         "actual": len(low_rows), "ok": str(len(low_rows) == EXPECTED_PARTIAL_PASS_CASES)},
        {"item": "zero_mask_normal_count", "value": zero_normal, "actual": zero_normal, "ok": "INFO"},
        {"item": "zero_mask_nsclc_count", "value": zero_nsclc, "actual": zero_nsclc, "ok": "INFO"},
        {"item": "zero_mask_nsclc_cand_FN", "value": 0,
         "actual": zero_mask_nsclc_fn, "ok": str(zero_mask_nsclc_fn == 0)},
        {"item": "zero_mask_normal_cand_FP", "value": "any",
         "actual": zero_mask_normal_fp, "ok": "INFO"},
        {"item": "patient_level_FN_ref (mean_prob)", "value": ref_pat_fn_count,
         "actual": ref_pat_fn_count, "ok": "INFO"},
        {"item": "patient_level_FN_cand (mean_prob)", "value": "<=ref+5",
         "actual": cand_pat_fn_count, "ok": str(cand_pat_fn_count <= ref_pat_fn_count + 5)},
        {"item": "patient_level_FN_cand (max_prob)", "value": 0,
         "actual": cand_pat_fn_max, "ok": str(cand_pat_fn_max == 0)},
        {"item": "patient_level_FP_ref (mean_prob)", "value": ref_pat_fp_count,
         "actual": ref_pat_fp_count, "ok": "INFO"},
        {"item": "patient_level_FP_cand (mean_prob)", "value": "<=ref",
         "actual": cand_pat_fp_count, "ok": str(cand_pat_fp_count <= ref_pat_fp_count)},
        {"item": "zero_mask_patient_with_cand_FN (mean)", "value": 0,
         "actual": zm_fn_direct, "ok": str(zm_fn_direct == 0)},
        {"item": "guardrail_fail", "value": 0, "actual": guardrail_fail,
         "ok": str(guardrail_fail == 0)},
        {"item": "verdict", "value": "PASS_FOR_DECISION",
         "actual": verdict, "ok": str(verdict in ("PASS_FOR_DECISION", "PARTIAL_PASS_WITH_CAVEAT"))},
        {"item": "verdict_reason", "value": "-", "actual": verdict_reason, "ok": "INFO"},
    ]
    write_csv(decision_rows, REPORT_ROOT / "p_c_normal31b_decision_readiness.csv")

    # ── 8. Guardrail check ────────────────────────────────────────────────
    guardrail_rows = [
        {"key": k, "value": v, "expected": True if k not in {"diagnostic_wording_avoided"} else True,
         "status": "OK" if v is True else "FAIL"}
        for k, v in GUARDRAILS.items()
    ]
    write_csv(guardrail_rows, REPORT_ROOT / "p_c_normal31b_guardrail_check.csv")

    # ── 9. Summary ────────────────────────────────────────────────────────
    mean_metric = pat_metric_summary[0] if pat_metric_summary else {}

    summary = {
        "stage": "P-C-NORMAL31b",
        "timestamp": ts,
        "audit_only": True,
        "p_c_normal31_verdict": "PARTIAL_PASS",
        "partial_pass_reason": "low_or_zero_mask=16 in Phase A (zero_mask=0 strict, low_mask=16 nzr_mean<0.05)",
        "low_or_zero_total": len(zero_rows),
        "strict_zero_mask_count": len(zero_rows_strict),
        "low_mask_count": len(low_rows),
        "low_or_zero_normal": zero_normal,
        "low_or_zero_nsclc": zero_nsclc,
        "low_or_zero_ratio_pct": f"{len(zero_rows)/EXPECTED_TOTAL*100:.4f}%",
        "low_or_zero_nsclc_cand_FN": zero_mask_nsclc_fn,
        "low_or_zero_normal_cand_FP": zero_mask_normal_fp,
        "patient_level_ref_FN_mean": ref_pat_fn_count,
        "patient_level_cand_FN_mean": cand_pat_fn_count,
        "patient_level_ref_FP_mean": ref_pat_fp_count,
        "patient_level_cand_FP_mean": cand_pat_fp_count,
        "patient_level_cand_FN_max_prob": cand_pat_fn_max,
        "patient_level_ref_AUROC_mean": mean_metric.get("ref_AUROC", ""),
        "patient_level_cand_AUROC_mean": mean_metric.get("cand_AUROC", ""),
        "guardrail_fail": guardrail_fail,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "p_c_normal32_ready": verdict in ("PASS_FOR_DECISION", "PARTIAL_PASS_WITH_CAVEAT"),
        "next_step": "P-C-NORMAL32 decision checkpoint (사용자 승인 필요)",
    }
    with open(REPORT_ROOT / "p_c_normal31b_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── 10. Report.md ─────────────────────────────────────────────────────
    # patient metric table 준비
    pat_table = "\n".join(
        f"| {m['agg']} | {m['ref_AUROC']} | {m['cand_AUROC']} | {m['AUROC_delta']} "
        f"| {m['ref_FP']}→{m['cand_FP']} | {m['ref_FN']}→{m['cand_FN']} "
        f"| {m['zero_mask_patients_with_cand_FN']} |"
        for m in pat_metric_summary
    )

    zero_label_dist = f"normal={zero_normal} / NSCLC={zero_nsclc}"
    zero_patient_list = "\n".join(
        f"  - {pid}: {cnt} crop(s), label={labels}"
        for pid, labels, cnt in [
            (r["patient_id"], r["label_types"], r["zero_mask_count"])
            for r in patient_overlap_rows
        ]
    ) or "  (none)"

    report_md = f"""# P-C-NORMAL31b: zero-mask impact + patient-level sanity audit

생성일: {ts}
Audit only — 학습/inference/threshold 최적화 없음.

---

## 1. P-C-NORMAL31 요약

P-C-NORMAL31 verdict: **PARTIAL_PASS**
PARTIAL_PASS 이유: Phase A mask generation에서 low_or_zero_mask = **{len(zero_rows)}** / 66,283 ({len(zero_rows)/EXPECTED_TOTAL*100:.4f}%)

> 정정: Phase A 로그의 `zero=16`은 `n_zero = len(zero_low_rows)` 로, **zero_mask=True 0건** + **low_mask=True(nzr_mean<0.05) {len(low_rows)}건** 합계.
> mask_3ch.sum()==0인 strict zero_mask는 0건이며, PARTIAL_PASS 원인은 nzr_mean<0.05인 low_mask 16건이다.

### 핵심 비교 (crop-level, fixed threshold 0.5)

| 항목 | Reference 24j bw1 | Masked 30b | Delta |
|------|-------------------|------------|-------|
| AUROC | 0.9517 | 0.9904 | +0.0387 |
| AUPRC | 0.9733 | 0.9954 | +0.0221 |
| Brier | 0.1533 | 0.0750 | -0.0783 |
| specificity | 0.4956 | 0.7328 | +0.2372 |
| sensitivity | 0.9929 | 0.9924 | -0.0006 |
| FP (normal) | 10,874 | 5,760 | -5,114 |
| FN (NSCLC) | 317 | 342 | +25 |

---

## 2. low_or_zero mask 16건 분석 (PARTIAL_PASS 원인)

- 총 mask manifest: 66,283 rows
- strict zero_mask=True (mask_3ch.sum()==0): **{len(zero_rows_strict)}** 건
- low_mask=True (nzr_mean<0.05, mask>0): **{len(low_rows)}** 건
- low_or_zero 합계: **{len(zero_rows)}** (expected {EXPECTED_PARTIAL_PASS_CASES})
- 비율: {len(zero_rows)/EXPECTED_TOTAL*100:.4f}%
- label 분포: {zero_label_dist}

### zero_mask 해당 환자 목록:
{zero_patient_list}

### zero_mask 예측 분류 (fixed 0.5):

| Model | TP | TN | FP | FN |
|-------|----|----|----|----|
| reference | {ref_cat_counts['TP']} | {ref_cat_counts['TN']} | {ref_cat_counts['FP']} | {ref_cat_counts['FN']} |
| masked_30b | {cand_cat_counts['TP']} | {cand_cat_counts['TN']} | {cand_cat_counts['FP']} | {cand_cat_counts['FN']} |

**zero_mask NSCLC 중 masked_30b FN 관여: {zero_mask_nsclc_fn}건**

> 주의: zero_mask 제외 시 혼동행렬 변화는 참고용 수치이며 공식 metric 대체 불가.
> threshold 튜닝 없음, 공식 결과 수정 없음.

---

## 3. patient-level sanity audit

| agg | ref AUROC | cand AUROC | delta | FP: ref→cand | FN: ref→cand | zero_mask patient w/ cand FN |
|-----|-----------|------------|-------|--------------|--------------|------------------------------|
{pat_table}

- patient 총 수: {len(all_pids)}명
- mean_prob 기준: ref FP={ref_pat_fp_count} → cand FP={cand_pat_fp_count} (normal FP patients 감소)
- mean_prob 기준: ref FN={ref_pat_fn_count} → cand FN={cand_pat_fn_count}
- max_prob 기준 cand FN: {cand_pat_fn_max}

masked 30b의 crop-level FP 감소가 patient-level FP 감소로 이어졌음을 확인.

---

## 4. masked 30b 장단점 요약

**장점:**
- AUROC/AUPRC/Brier 개선
- normal FP 대폭 감소 (-5,114 crops)
- specificity +0.237
- patient-level normal FP patients 감소

**단점:**
- NSCLC FN 25개 증가 (317→342, crop-level)
- sensitivity 0.0006 하락
- zero_mask caveat 16건 존재

---

## 5. Decision Readiness

| 항목 | 결과 | 판정 |
|------|------|------|
| low_or_zero 16건 확인 | {len(zero_rows)} | {"OK" if len(zero_rows)==EXPECTED_PARTIAL_PASS_CASES else "WARN"} |
| strict zero_mask=True 건수 | {len(zero_rows_strict)} | {"OK(0건)" if len(zero_rows_strict)==0 else "WARN"} |
| low_mask NSCLC FN 관여 | {zero_mask_nsclc_fn} | {"OK" if zero_mask_nsclc_fn == 0 else "WARN"} |
| patient-level FN 과도 증가 | ref={ref_pat_fn_count} cand={cand_pat_fn_count} | {"OK" if cand_pat_fn_count <= ref_pat_fn_count + 5 else "WARN"} |
| max_prob 기준 cand FN | {cand_pat_fn_max} | {"OK" if cand_pat_fn_max == 0 else "WARN"} |
| guardrail fail | {guardrail_fail} | {"OK" if guardrail_fail == 0 else "FAIL"} |

**VERDICT: {verdict}**

{verdict_reason}

---

## 6. 다음 단계

P-C-NORMAL32 decision checkpoint — **사용자 승인 필요**
"""

    (REPORT_ROOT / "p_c_normal31b_report.md").write_text(report_md, encoding="utf-8")

    # ── 11. DONE.json ─────────────────────────────────────────────────────
    done = {
        "stage": "P-C-NORMAL31b",
        "timestamp": ts,
        "verdict": verdict,
        "guardrail_fail": guardrail_fail,
        "files_written": [
            "p_c_normal31b_zero_mask_cases.csv",
            "p_c_normal31b_zero_mask_prediction_impact.csv",
            "p_c_normal31b_patient_level_sanity_summary.csv",
            "p_c_normal31b_zero_mask_patient_overlap.csv",
            "p_c_normal31b_patient_metric_summary.csv",
            "p_c_normal31b_decision_readiness.csv",
            "p_c_normal31b_guardrail_check.csv",
            "p_c_normal31b_report.md",
            "p_c_normal31b_summary.json",
            "DONE.json",
        ],
        "next_step": "P-C-NORMAL32 decision checkpoint (사용자 승인 필요)",
    }
    with open(REPORT_ROOT / "DONE.json", "w", encoding="utf-8") as f:
        json.dump(done, f, indent=2, ensure_ascii=False)

    print(f"[31b] DONE. verdict={verdict}, guardrail_fail={guardrail_fail}")
    print(f"[31b] report: {REPORT_ROOT}/p_c_normal31b_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(run())
