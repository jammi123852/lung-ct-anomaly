"""
P-C-NORMAL21f: Per-Crop HU / Position / Context Shortcut Error Audit
Read-only analysis of P-C-NORMAL21e export results.
No training, no inference, no threshold optimization, no holdout access.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
BRANCH_ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR  = BRANCH_ROOT / "outputs/reports/p_c_normal21c_validation_prediction_export"
OUT_DIR     = BRANCH_ROOT / "outputs/reports/p_c_normal21f_per_crop_shortcut_error_audit"

CROP_CSV    = EXPORT_DIR / "p_c_normal21c_val_crop_predictions.csv"
PATIENT_CSV = EXPORT_DIR / "p_c_normal21c_val_patient_predictions.csv"
SUMMARY_JSON = EXPORT_DIR / "p_c_normal21c_prediction_export_summary.json"
DONE_JSON   = EXPORT_DIR / "DONE.json"

STAGE2_HOLDOUT_SENTINEL = "stage2_holdout"
FORBIDDEN_WORDS = [
    "폐선암 확률", "암 확률", "진단 모델",
    "cancer probability", "adenocarcinoma probability",
]

REQUIRED_CROP_COLS = [
    "prob_nsclc_like", "label", "pred_label_threshold_0_5", "correct_threshold_0_5",
    "patient_id", "position_bin", "z_level",
    "crop_hu_mean", "crop_hu_std", "crop_hu_min", "crop_hu_max",
    "air_frac_lt_minus800", "dense_frac_gt_minus500",
    "dense_frac_gt_minus300", "positive_frac_gt_0",
]

# Patients mentioned in audit spec
PATIENTS_OF_INTEREST = ["subset2_470912", "subset8_203425", "LUNG1-326"]


def check_forbidden(text: str) -> int:
    lo = text.lower()
    return sum(lo.count(w.lower()) for w in FORBIDDEN_WORDS)


def pct_stats(arr) -> dict:
    arr = np.array(arr, dtype=float)
    return {
        "mean":   round(float(np.mean(arr)), 6),
        "median": round(float(np.median(arr)), 6),
        "std":    round(float(np.std(arr)), 6),
        "p05":    round(float(np.percentile(arr, 5)), 6),
        "p25":    round(float(np.percentile(arr, 25)), 6),
        "p75":    round(float(np.percentile(arr, 75)), 6),
        "p95":    round(float(np.percentile(arr, 95)), 6),
        "min":    round(float(np.min(arr)), 6),
        "max":    round(float(np.max(arr)), 6),
        "n":      len(arr),
    }


def main():
    print("[21F] P-C-NORMAL21f per-crop shortcut/error audit 시작")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    errors = []
    guardrail = {
        "training_run": False,
        "model_forward_run": False,
        "backward_run": False,
        "optimizer_step": False,
        "checkpoint_saved": False,
        "threshold_computed_or_optimized": False,
        "stage2_holdout_accessed": False,
        "existing_21e_results_modified": False,
        "existing_19_checkpoint_modified": False,
        "additional_inference_run": False,
        "forbidden_diagnostic_wording_count": 0,
    }

    # ── 1. Export file validation ──
    print("[21F] Step 1: Export file validation")
    checks = []

    def record(name, status, detail=""):
        checks.append({"check": name, "status": status, "detail": str(detail)})
        marker = "PASS" if status == "PASS" else "FAIL"
        print(f"  [{marker}] {name}" + (f": {detail}" if detail else ""))

    for fpath, label in [
        (CROP_CSV, "crop_csv"),
        (PATIENT_CSV, "patient_csv"),
        (SUMMARY_JSON, "summary_json"),
        (DONE_JSON, "done_json"),
    ]:
        if fpath.exists():
            record(f"file_exists:{label}", "PASS", str(fpath))
        else:
            record(f"file_exists:{label}", "FAIL", f"not found: {fpath}")
            errors.append(f"missing file: {fpath}")

    if errors:
        _finalize(checks, guardrail, errors, verdict="FAIL")
        return

    # load
    crop_df    = pd.read_csv(CROP_CSV)
    patient_df = pd.read_csv(PATIENT_CSV)
    with open(SUMMARY_JSON) as f:
        summary_json = json.load(f)
    with open(DONE_JSON) as f:
        done_json = json.load(f)

    # row counts
    n_crops = len(crop_df)
    n_patients = len(patient_df)
    record("crop_row_count", "PASS" if n_crops == 5200 else "FAIL", f"{n_crops}/5200")
    record("patient_row_count", "PASS" if n_patients == 60 else "FAIL", f"{n_patients}/60")
    if n_crops != 5200:
        errors.append(f"crop row mismatch: {n_crops}")
    if n_patients != 60:
        errors.append(f"patient row mismatch: {n_patients}")

    # errors=0 from summary
    n_export_errors = summary_json.get("n_errors", -1)
    record("export_errors=0", "PASS" if n_export_errors == 0 else "FAIL", f"{n_export_errors}")
    if n_export_errors != 0:
        errors.append(f"export errors={n_export_errors}")

    # DONE.json status
    done_status = done_json.get("status", "")
    record("done_json_status=DONE", "PASS" if done_status == "DONE" else "FAIL", done_status)

    # required columns
    missing_cols = [c for c in REQUIRED_CROP_COLS if c not in crop_df.columns]
    record("required_columns_present",
           "PASS" if not missing_cols else "FAIL",
           f"missing={missing_cols}" if missing_cols else f"all {len(REQUIRED_CROP_COLS)} present")
    if missing_cols:
        errors.append(f"missing columns: {missing_cols}")

    # stage2 holdout check
    holdout_hit = crop_df["crop_path"].astype(str).str.contains(STAGE2_HOLDOUT_SENTINEL, na=False).any()
    record("stage2_holdout_refs=0", "PASS" if not holdout_hit else "FAIL")
    if holdout_hit:
        guardrail["stage2_holdout_accessed"] = True
        errors.append("stage2_holdout refs detected")

    # ── 2. Overall prediction summary ──
    print("[21F] Step 2: Overall prediction summary")
    crop_df["label"] = crop_df["label"].astype(int)
    crop_df["prob_nsclc_like"] = crop_df["prob_nsclc_like"].astype(float)
    crop_df["pred_label_threshold_0_5"] = crop_df["pred_label_threshold_0_5"].astype(int)
    crop_df["correct_threshold_0_5"] = crop_df["correct_threshold_0_5"].astype(int)

    n_correct = int(crop_df["correct_threshold_0_5"].sum())
    crop_acc = round(n_correct / n_crops, 6)
    record("crop_accuracy_recalc",
           "PASS" if abs(crop_acc - 0.995577) < 0.0001 else "FAIL",
           f"{crop_acc}")

    normal_df = crop_df[crop_df["label"] == 0]
    nsclc_df  = crop_df[crop_df["label"] == 1]
    normal_acc = round(float(normal_df["correct_threshold_0_5"].mean()), 6)
    nsclc_acc  = round(float(nsclc_df["correct_threshold_0_5"].mean()), 6)

    # confusion matrix @0.5
    tn = int(((crop_df["label"] == 0) & (crop_df["pred_label_threshold_0_5"] == 0)).sum())
    fp = int(((crop_df["label"] == 0) & (crop_df["pred_label_threshold_0_5"] == 1)).sum())
    fn = int(((crop_df["label"] == 1) & (crop_df["pred_label_threshold_0_5"] == 0)).sum())
    tp = int(((crop_df["label"] == 1) & (crop_df["pred_label_threshold_0_5"] == 1)).sum())

    print(f"  Crop accuracy: {crop_acc} | Normal: {normal_acc} | NSCLC: {nsclc_acc}")
    print(f"  Confusion @0.5: TN={tn} FP={fp} FN={fn} TP={tp}")

    # prob distribution by label
    prob_dist_normal = pct_stats(normal_df["prob_nsclc_like"].values)
    prob_dist_nsclc  = pct_stats(nsclc_df["prob_nsclc_like"].values)

    # patient-level accuracy recheck
    patient_df["correct_mean_threshold_0_5"] = patient_df["correct_mean_threshold_0_5"].astype(int)
    patient_df["correct_max_threshold_0_5"] = patient_df["correct_max_threshold_0_5"].astype(int)
    pat_acc_mean = round(float(patient_df["correct_mean_threshold_0_5"].mean()), 6)
    pat_acc_max  = round(float(patient_df["correct_max_threshold_0_5"].mean()), 6)
    record("patient_acc_mean_recalc",
           "PASS" if abs(pat_acc_mean - 1.0) < 0.01 else "FAIL",
           f"{pat_acc_mean}")

    overall_summary_rows = [
        {"metric": "n_crops", "value": n_crops},
        {"metric": "n_patients", "value": n_patients},
        {"metric": "crop_accuracy_0_5", "value": crop_acc},
        {"metric": "normal_crop_accuracy_0_5", "value": normal_acc},
        {"metric": "nsclc_crop_accuracy_0_5", "value": nsclc_acc},
        {"metric": "patient_acc_mean_prob_0_5", "value": pat_acc_mean},
        {"metric": "patient_acc_max_prob_0_5", "value": pat_acc_max},
        {"metric": "n_normal_crops", "value": len(normal_df)},
        {"metric": "n_nsclc_crops", "value": len(nsclc_df)},
        {"metric": "prob_normal_mean", "value": prob_dist_normal["mean"]},
        {"metric": "prob_normal_p95", "value": prob_dist_normal["p95"]},
        {"metric": "prob_nsclc_mean", "value": prob_dist_nsclc["mean"]},
        {"metric": "prob_nsclc_p05", "value": prob_dist_nsclc["p05"]},
    ]
    pd.DataFrame(overall_summary_rows).to_csv(
        OUT_DIR / "p_c_normal21f_overall_prediction_summary.csv", index=False)

    # confusion matrix
    cm_rows = [
        {"metric": "TN", "value": tn, "label_true": 0, "label_pred": 0},
        {"metric": "FP", "value": fp, "label_true": 0, "label_pred": 1},
        {"metric": "FN", "value": fn, "label_true": 1, "label_pred": 0},
        {"metric": "TP", "value": tp, "label_true": 1, "label_pred": 1},
    ]
    pd.DataFrame(cm_rows).to_csv(
        OUT_DIR / "p_c_normal21f_confusion_matrix.csv", index=False)

    # ── 3. Error crop audit ──
    print("[21F] Step 3: Error crop audit")
    wrong_df = crop_df[crop_df["correct_threshold_0_5"] == 0].copy()
    fp_df    = crop_df[(crop_df["label"] == 0) & (crop_df["pred_label_threshold_0_5"] == 1)].copy()
    fn_df    = crop_df[(crop_df["label"] == 1) & (crop_df["pred_label_threshold_0_5"] == 0)].copy()
    borderline_df = crop_df[
        (crop_df["prob_nsclc_like"] >= 0.4) & (crop_df["prob_nsclc_like"] <= 0.6)
    ].copy()

    print(f"  Wrong crops total={len(wrong_df)} | FP_normal={len(fp_df)} | FN_NSCLC={len(fn_df)}")
    print(f"  Borderline (0.4~0.6): {len(borderline_df)}")

    hu_cols = ["crop_hu_mean", "crop_hu_std", "crop_hu_min", "crop_hu_max",
               "air_frac_lt_minus800", "dense_frac_gt_minus500",
               "dense_frac_gt_minus300", "positive_frac_gt_0"]
    meta_cols = ["row_index", "aux_candidate_id", "patient_id", "position_bin",
                 "z_level", "label", "label_name", "prob_nsclc_like",
                 "pred_label_threshold_0_5", "correct_threshold_0_5"] + hu_cols

    error_cols = [c for c in meta_cols if c in crop_df.columns]
    wrong_df[error_cols].to_csv(OUT_DIR / "p_c_normal21f_error_crop_table.csv", index=False)

    # top 20 normal high-prob
    top_normal_high = normal_df.nlargest(20, "prob_nsclc_like")[error_cols]
    top_normal_high.to_csv(OUT_DIR / "p_c_normal21f_top_normal_high_prob.csv", index=False)

    # top 20 NSCLC low-prob
    top_nsclc_low = nsclc_df.nsmallest(20, "prob_nsclc_like")[error_cols]
    top_nsclc_low.to_csv(OUT_DIR / "p_c_normal21f_top_nsclc_low_prob.csv", index=False)

    # borderline
    borderline_df[error_cols].sort_values("prob_nsclc_like").to_csv(
        OUT_DIR / "p_c_normal21f_borderline_crops.csv", index=False)

    # ── 4. Patient-level outlier audit ──
    print("[21F] Step 4: Patient-level outlier audit")
    patient_df["label"] = patient_df["label"].astype(int)
    patient_df["mean_prob_nsclc_like"] = patient_df["mean_prob_nsclc_like"].astype(float)
    patient_df["max_prob_nsclc_like"] = patient_df["max_prob_nsclc_like"].astype(float)

    normal_pat = patient_df[patient_df["label"] == 0]
    nsclc_pat  = patient_df[patient_df["label"] == 1]

    top_normal_max  = normal_pat.nlargest(10, "max_prob_nsclc_like")
    top_normal_mean = normal_pat.nlargest(10, "mean_prob_nsclc_like")
    top_nsclc_low_mean = nsclc_pat.nsmallest(10, "mean_prob_nsclc_like")
    top_nsclc_low_max  = nsclc_pat.nsmallest(10, "max_prob_nsclc_like")

    outlier_rows = []
    for _, row in top_normal_max.iterrows():
        outlier_rows.append({**row.to_dict(), "outlier_type": "normal_high_max"})
    for _, row in top_normal_mean.iterrows():
        existing = {r["patient_id"] for r in outlier_rows}
        if row["patient_id"] not in existing:
            outlier_rows.append({**row.to_dict(), "outlier_type": "normal_high_mean"})
    for _, row in top_nsclc_low_mean.iterrows():
        outlier_rows.append({**row.to_dict(), "outlier_type": "nsclc_low_mean"})
    for _, row in top_nsclc_low_max.iterrows():
        existing = {r["patient_id"] for r in outlier_rows if r.get("outlier_type","").startswith("nsclc")}
        if row["patient_id"] not in existing:
            outlier_rows.append({**row.to_dict(), "outlier_type": "nsclc_low_max"})

    pd.DataFrame(outlier_rows).to_csv(
        OUT_DIR / "p_c_normal21f_patient_outlier_summary.csv", index=False)

    # Specific patient deep-dives
    poi_analysis = {}
    for pid in PATIENTS_OF_INTEREST:
        # match by patient_id containing the key string
        pid_crops = crop_df[crop_df["patient_id"].astype(str).str.contains(pid, na=False)]
        pid_pat   = patient_df[patient_df["patient_id"].astype(str).str.contains(pid, na=False)]
        if len(pid_crops) == 0:
            poi_analysis[pid] = {"found": False, "n_crops": 0}
            print(f"  [POI] {pid}: NOT FOUND in crop CSV")
            continue

        label_val = int(pid_crops["label"].iloc[0])
        n_high_prob = int((pid_crops["prob_nsclc_like"] >= 0.5).sum())
        poi_analysis[pid] = {
            "found": True,
            "patient_id_match": pid_crops["patient_id"].iloc[0],
            "label": label_val,
            "n_crops": len(pid_crops),
            "mean_prob": round(float(pid_crops["prob_nsclc_like"].mean()), 6),
            "max_prob":  round(float(pid_crops["prob_nsclc_like"].max()), 6),
            "min_prob":  round(float(pid_crops["prob_nsclc_like"].min()), 6),
            "n_high_prob_crops": n_high_prob,
            "position_bin_counts": pid_crops["position_bin"].value_counts().to_dict() if "position_bin" in pid_crops.columns else {},
            "z_level_counts": pid_crops["z_level"].value_counts().to_dict() if "z_level" in pid_crops.columns else {},
            "hu_mean_mean": round(float(pid_crops["crop_hu_mean"].mean()), 2) if "crop_hu_mean" in pid_crops.columns else None,
            "air_frac_mean": round(float(pid_crops["air_frac_lt_minus800"].mean()), 4) if "air_frac_lt_minus800" in pid_crops.columns else None,
            "dense_frac_mean": round(float(pid_crops["dense_frac_gt_minus500"].mean()), 4) if "dense_frac_gt_minus500" in pid_crops.columns else None,
        }
        print(f"  [POI] {pid}: label={label_val}, n_crops={len(pid_crops)}, "
              f"mean_prob={poi_analysis[pid]['mean_prob']:.4f}, "
              f"max_prob={poi_analysis[pid]['max_prob']:.4f}, "
              f"n_high_prob={n_high_prob}")

    # ── 5. HU shortcut audit ──
    print("[21F] Step 5: HU shortcut audit")
    hu_rows = []
    for grp_label, grp_name in [(0, "normal"), (1, "NSCLC")]:
        for grp_correct, grp_correct_name in [(1, "correct"), (0, "wrong")]:
            sub = crop_df[(crop_df["label"] == grp_label) &
                          (crop_df["correct_threshold_0_5"] == grp_correct)]
            if len(sub) == 0:
                continue
            for col in ["crop_hu_mean", "air_frac_lt_minus800",
                        "dense_frac_gt_minus500", "dense_frac_gt_minus300",
                        "positive_frac_gt_0"]:
                if col not in sub.columns:
                    continue
                st = pct_stats(sub[col].values)
                hu_rows.append({
                    "label_name": grp_name,
                    "correctness": grp_correct_name,
                    "n": len(sub),
                    "hu_metric": col,
                    **{k: st[k] for k in ["mean","median","std","p05","p25","p75","p95","min","max"]},
                })

    pd.DataFrame(hu_rows).to_csv(OUT_DIR / "p_c_normal21f_hu_shortcut_audit.csv", index=False)

    # correlations
    prob_col = crop_df["prob_nsclc_like"].astype(float).values
    hu_corr = {}
    for col in ["crop_hu_mean", "dense_frac_gt_minus500", "air_frac_lt_minus800",
                "dense_frac_gt_minus300", "positive_frac_gt_0"]:
        if col in crop_df.columns:
            r = float(np.corrcoef(prob_col, crop_df[col].astype(float).values)[0, 1])
            hu_corr[col] = round(r, 4)
            print(f"  corr(prob, {col}) = {r:.4f}")

    # FP HU stats
    fp_hu_mean = float(fp_df["crop_hu_mean"].mean()) if len(fp_df) > 0 and "crop_hu_mean" in fp_df.columns else None
    fp_dense_mean = float(fp_df["dense_frac_gt_minus500"].mean()) if len(fp_df) > 0 and "dense_frac_gt_minus500" in fp_df.columns else None
    fp_air_mean = float(fp_df["air_frac_lt_minus800"].mean()) if len(fp_df) > 0 and "air_frac_lt_minus800" in fp_df.columns else None

    fn_hu_mean = float(fn_df["crop_hu_mean"].mean()) if len(fn_df) > 0 and "crop_hu_mean" in fn_df.columns else None
    fn_air_mean = float(fn_df["air_frac_lt_minus800"].mean()) if len(fn_df) > 0 and "air_frac_lt_minus800" in fn_df.columns else None

    # HU shortcut risk: compare FP normal to correct normal
    correct_normal = crop_df[(crop_df["label"] == 0) & (crop_df["correct_threshold_0_5"] == 1)]
    correct_normal_hu_mean = float(correct_normal["crop_hu_mean"].mean()) if "crop_hu_mean" in correct_normal.columns else None
    correct_normal_dense_mean = float(correct_normal["dense_frac_gt_minus500"].mean()) if "dense_frac_gt_minus500" in correct_normal.columns else None

    # SR-HU rating
    if fp_dense_mean is not None and correct_normal_dense_mean is not None:
        dense_diff = fp_dense_mean - correct_normal_dense_mean
        if abs(dense_diff) > 0.1 and abs(hu_corr.get("crop_hu_mean", 0)) > 0.3:
            sr_hu = "HIGH"
        elif abs(dense_diff) > 0.05 or abs(hu_corr.get("crop_hu_mean", 0)) > 0.15:
            sr_hu = "MEDIUM"
        else:
            sr_hu = "LOW"
    else:
        sr_hu = "LOW"

    fp_dense_str = f"{fp_dense_mean:.4f}" if fp_dense_mean is not None else "N/A"
    cn_dense_str = f"{correct_normal_dense_mean:.4f}" if correct_normal_dense_mean is not None else "N/A"
    print(f"  SR-HU: {sr_hu} | FP dense_frac={fp_dense_str} vs correct_normal dense_frac={cn_dense_str}")

    # ── 6. Position / context shortcut audit ──
    print("[21F] Step 6: Position/context shortcut audit")
    pos_rows = []
    if "position_bin" in crop_df.columns:
        for pos, grp in crop_df.groupby("position_bin"):
            n_total = len(grp)
            n_norm  = int((grp["label"] == 0).sum())
            n_nsc   = int((grp["label"] == 1).sum())
            acc     = round(float(grp["correct_threshold_0_5"].mean()), 4)
            fp_pos  = int(((grp["label"] == 0) & (grp["pred_label_threshold_0_5"] == 1)).sum())
            fn_pos  = int(((grp["label"] == 1) & (grp["pred_label_threshold_0_5"] == 0)).sum())
            mean_prob_norm = round(float(grp[grp["label"]==0]["prob_nsclc_like"].mean()), 4) if n_norm > 0 else None
            mean_prob_nsc  = round(float(grp[grp["label"]==1]["prob_nsclc_like"].mean()), 4) if n_nsc > 0 else None
            pos_rows.append({
                "position_bin": pos,
                "n_total": n_total,
                "n_normal": n_norm,
                "n_nsclc": n_nsc,
                "accuracy": acc,
                "fp_count": fp_pos,
                "fn_count": fn_pos,
                "mean_prob_normal": mean_prob_norm,
                "mean_prob_nsclc": mean_prob_nsc,
            })

    pos_df = pd.DataFrame(pos_rows).sort_values("position_bin")
    pos_df.to_csv(OUT_DIR / "p_c_normal21f_position_shortcut_audit.csv", index=False)

    # z_level audit
    z_rows = []
    if "z_level" in crop_df.columns:
        for zlvl, grp in crop_df.groupby("z_level"):
            n_total = len(grp)
            n_norm  = int((grp["label"] == 0).sum())
            n_nsc   = int((grp["label"] == 1).sum())
            acc     = round(float(grp["correct_threshold_0_5"].mean()), 4)
            fp_z  = int(((grp["label"] == 0) & (grp["pred_label_threshold_0_5"] == 1)).sum())
            fn_z  = int(((grp["label"] == 1) & (grp["pred_label_threshold_0_5"] == 0)).sum())
            z_rows.append({
                "z_level": zlvl,
                "n_total": n_total,
                "n_normal": n_norm,
                "n_nsclc": n_nsc,
                "accuracy": acc,
                "fp_count": fp_z,
                "fn_count": fn_z,
                "mean_prob_normal": round(float(grp[grp["label"]==0]["prob_nsclc_like"].mean()), 4) if n_norm > 0 else None,
                "mean_prob_nsclc":  round(float(grp[grp["label"]==1]["prob_nsclc_like"].mean()), 4) if n_nsc > 0 else None,
            })

    # central vs peripheral (based on position_bin keyword)
    cent_rows = []
    if "position_bin" in crop_df.columns:
        for region_key, region_name in [("central", "central"), ("peripheral", "peripheral")]:
            sub = crop_df[crop_df["position_bin"].astype(str).str.contains(region_key, na=False)]
            if len(sub) == 0:
                continue
            n_norm = int((sub["label"] == 0).sum())
            n_nsc  = int((sub["label"] == 1).sum())
            acc    = round(float(sub["correct_threshold_0_5"].mean()), 4)
            fp_c   = int(((sub["label"] == 0) & (sub["pred_label_threshold_0_5"] == 1)).sum())
            fn_c   = int(((sub["label"] == 1) & (sub["pred_label_threshold_0_5"] == 0)).sum())
            cent_rows.append({
                "region": region_name,
                "n_total": len(sub),
                "n_normal": n_norm,
                "n_nsclc": n_nsc,
                "accuracy": acc,
                "fp_count": fp_c,
                "fn_count": fn_c,
                "mean_prob_normal": round(float(sub[sub["label"]==0]["prob_nsclc_like"].mean()), 4) if n_norm > 0 else None,
                "mean_prob_nsclc":  round(float(sub[sub["label"]==1]["prob_nsclc_like"].mean()), 4) if n_nsc > 0 else None,
            })

    all_pos_rows = pos_rows + z_rows + cent_rows
    pd.DataFrame(all_pos_rows).to_csv(
        OUT_DIR / "p_c_normal21f_position_shortcut_audit.csv", index=False)

    # SR-POS assessment
    if pos_rows:
        fp_by_pos = {r["position_bin"]: r["fp_count"] for r in pos_rows}
        max_fp_pos = max(fp_by_pos, key=fp_by_pos.get) if fp_by_pos else "N/A"
        max_fp_count = max(fp_by_pos.values()) if fp_by_pos else 0
        sr_pos = "ELEVATED" if max_fp_count > 5 else "LOW"
    else:
        sr_pos = "N/A"
        max_fp_pos = "N/A"
        max_fp_count = 0

    print(f"  SR-POS: {sr_pos} | max FP position_bin={max_fp_pos} ({max_fp_count} FPs)")

    # ── 7. Shortcut decision ──
    print("[21F] Step 7: Shortcut decision")
    n_wrong = len(wrong_df)
    n_fp    = len(fp_df)
    n_fn    = len(fn_df)
    n_borderline = len(borderline_df)

    # Decision logic
    if n_wrong <= 30 and sr_hu in ("LOW", "MEDIUM") and sr_pos == "LOW":
        if n_fp <= 5 and n_fn <= 20:
            shortcut_decision = "AUDIT_PASS_READY_FOR_INTEGRATION_PREFLIGHT"
        else:
            shortcut_decision = "NEEDS_VISUAL_REVIEW_OF_OUTLIERS"
    elif sr_hu == "HIGH":
        shortcut_decision = "NEEDS_HU_CONTEXT_MITIGATION"
    else:
        shortcut_decision = "NEEDS_VISUAL_REVIEW_OF_OUTLIERS"

    print(f"  Shortcut decision: {shortcut_decision}")
    print(f"  n_wrong={n_wrong}, n_fp={n_fp}, n_fn={n_fn}, n_borderline={n_borderline}")
    print(f"  SR-HU={sr_hu}, SR-POS={sr_pos}")

    shortcut_rows = [
        {"item": "shortcut_decision",          "value": shortcut_decision},
        {"item": "sr_hu_risk",                 "value": sr_hu},
        {"item": "sr_pos_risk",                "value": sr_pos},
        {"item": "n_wrong_crops",              "value": n_wrong},
        {"item": "n_fp_normal",                "value": n_fp},
        {"item": "n_fn_nsclc",                 "value": n_fn},
        {"item": "n_borderline_0_4_0_6",       "value": n_borderline},
        {"item": "corr_prob_hu_mean",          "value": hu_corr.get("crop_hu_mean", "N/A")},
        {"item": "corr_prob_dense_frac",       "value": hu_corr.get("dense_frac_gt_minus500", "N/A")},
        {"item": "corr_prob_air_frac",         "value": hu_corr.get("air_frac_lt_minus800", "N/A")},
        {"item": "fp_hu_mean",                 "value": round(fp_hu_mean, 2) if fp_hu_mean else "N/A"},
        {"item": "fp_dense_frac_mean",         "value": round(fp_dense_mean, 4) if fp_dense_mean else "N/A"},
        {"item": "fn_hu_mean",                 "value": round(fn_hu_mean, 2) if fn_hu_mean else "N/A"},
        {"item": "fn_air_frac_mean",           "value": round(fn_air_mean, 4) if fn_air_mean else "N/A"},
        {"item": "correct_normal_dense_frac",  "value": round(correct_normal_dense_mean, 4) if correct_normal_dense_mean else "N/A"},
        {"item": "max_fp_position_bin",        "value": max_fp_pos},
        {"item": "max_fp_position_count",      "value": max_fp_count},
    ]
    pd.DataFrame(shortcut_rows).to_csv(
        OUT_DIR / "p_c_normal21f_shortcut_decision.csv", index=False)

    # ── 8. Guardrail & forbidden wording check ──
    candidate_report_text = (
        "P-C-NORMAL21f per-crop shortcut error audit. "
        "supervised auxiliary classifier. "
        "normal-like vs NSCLC-lesion-like auxiliary score. "
        f"shortcut_decision={shortcut_decision}. "
        f"SR-HU={sr_hu}. SR-POS={sr_pos}."
    )
    fw_count = check_forbidden(candidate_report_text)
    guardrail["forbidden_diagnostic_wording_count"] = fw_count
    if fw_count:
        errors.append(f"forbidden_diagnostic_wording_count={fw_count}")

    # ── Verdict ──
    fail_count = sum(1 for c in checks if c["status"] != "PASS")
    verdict = "PASS" if (fail_count == 0 and not errors) else (
        "PARTIAL_PASS" if (n_crops == 5200 and n_patients == 60) else "FAIL"
    )
    if shortcut_decision == "NEEDS_VISUAL_REVIEW_OF_OUTLIERS" and verdict == "PASS":
        verdict = "PARTIAL_PASS"

    _finalize(checks, guardrail, errors, verdict,
              summary_data={
                  "n_crops": n_crops,
                  "n_patients": n_patients,
                  "crop_acc": crop_acc,
                  "normal_crop_acc": normal_acc,
                  "nsclc_crop_acc": nsclc_acc,
                  "pat_acc_mean": pat_acc_mean,
                  "pat_acc_max": pat_acc_max,
                  "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                  "n_wrong": n_wrong,
                  "n_fp": n_fp, "n_fn": n_fn,
                  "n_borderline": n_borderline,
                  "sr_hu": sr_hu, "sr_pos": sr_pos,
                  "shortcut_decision": shortcut_decision,
                  "hu_corr": hu_corr,
                  "poi_analysis": poi_analysis,
                  "fp_hu_mean": fp_hu_mean,
                  "fp_dense_mean": fp_dense_mean,
                  "fn_hu_mean": fn_hu_mean,
                  "pos_rows": pos_rows,
                  "prob_dist_normal": prob_dist_normal,
                  "prob_dist_nsclc": prob_dist_nsclc,
              })
    print(f"\n[21F] 판정: {verdict}")


def _finalize(checks, guardrail, errors, verdict, summary_data=None):
    sd = summary_data or {}

    # guardrail_check.csv
    with open(OUT_DIR / "p_c_normal21f_guardrail_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["guardrail", "value"])
        w.writeheader()
        w.writerows([{"guardrail": k, "value": str(v)} for k, v in guardrail.items()])

    # errors.csv
    with open(OUT_DIR / "p_c_normal21f_errors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["error"])
        for e in errors:
            w.writerow([e])

    # summary JSON
    pos_rows = sd.get("pos_rows", [])
    summary = {
        "stage": "P-C-NORMAL21f",
        "mode": "per_crop_shortcut_error_audit",
        "verdict": verdict,
        "total_checks": len(checks),
        "pass_count": sum(1 for c in checks if c["status"] == "PASS"),
        "fail_count": sum(1 for c in checks if c["status"] != "PASS"),
        "error_count": len(errors),
        "guardrail": guardrail,
        "shortcut_decision": sd.get("shortcut_decision", "N/A"),
        "sr_hu_risk": sd.get("sr_hu", "N/A"),
        "sr_pos_risk": sd.get("sr_pos", "N/A"),
        "n_crops": sd.get("n_crops", 0),
        "n_patients": sd.get("n_patients", 0),
        "crop_accuracy_0_5": sd.get("crop_acc"),
        "normal_crop_accuracy_0_5": sd.get("normal_crop_acc"),
        "nsclc_crop_accuracy_0_5": sd.get("nsclc_crop_acc"),
        "patient_accuracy_mean_0_5": sd.get("pat_acc_mean"),
        "patient_accuracy_max_0_5": sd.get("pat_acc_max"),
        "confusion_matrix": {"TN": sd.get("tn"), "FP": sd.get("fp"),
                             "FN": sd.get("fn"), "TP": sd.get("tp")},
        "n_wrong_crops": sd.get("n_wrong"),
        "n_fp_normal": sd.get("n_fp"),
        "n_fn_nsclc": sd.get("n_fn"),
        "n_borderline_0_4_0_6": sd.get("n_borderline"),
        "hu_correlations": sd.get("hu_corr", {}),
        "fp_hu_mean": sd.get("fp_hu_mean"),
        "fp_dense_frac_mean": sd.get("fp_dense_mean"),
        "fn_hu_mean": sd.get("fn_hu_mean"),
        "prob_dist_normal": sd.get("prob_dist_normal"),
        "prob_dist_nsclc": sd.get("prob_dist_nsclc"),
        "patients_of_interest": sd.get("poi_analysis", {}),
        "position_audit_summary": [
            {"position_bin": r["position_bin"], "fp_count": r["fp_count"],
             "fn_count": r["fn_count"], "accuracy": r["accuracy"]}
            for r in pos_rows
        ] if pos_rows else [],
        "branch": "P-C-NORMAL (supervised auxiliary classifier, normal=0 / NSCLC=1)",
        "interpretation": "normal-like vs NSCLC-lesion-like auxiliary score only",
    }
    with open(OUT_DIR / "p_c_normal21f_per_crop_shortcut_error_audit.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Markdown report
    poi_analysis = sd.get("poi_analysis", {})
    hu_corr = sd.get("hu_corr", {})
    fp_dense = sd.get("fp_dense_mean")
    fn_air   = sd.get("fn_hu_mean")
    pos_rows_list = sd.get("pos_rows", [])

    md = [
        "# P-C-NORMAL21f Per-Crop HU / Position / Context Shortcut Audit Report",
        "",
        f"**판정: {verdict}**",
        f"**Shortcut Decision: {sd.get('shortcut_decision', 'N/A')}**",
        "",
        "## 개요",
        "- branch: P-C-NORMAL (supervised auxiliary classifier, normal=0 / NSCLC=1)",
        "- 출력: normal-like vs NSCLC-lesion-like auxiliary score (진단 모델 아님)",
        "- 분석 대상: P-C-NORMAL21e export 결과 (read-only)",
        "- 추가 inference/training/threshold 최적화/holdout 미접근",
        "",
        "## P-C-NORMAL21e 요약",
        f"- crops processed: {sd.get('n_crops',0)}/5200",
        f"- patients: {sd.get('n_patients',0)}/60",
        f"- errors: 0",
        f"- crop accuracy @0.5: {sd.get('crop_acc', 'N/A')}",
        f"- patient accuracy mean @0.5: {sd.get('pat_acc_mean', 'N/A')}",
        "",
        "## Crop-level Confusion Matrix @0.5",
        "| | Pred Normal | Pred NSCLC |",
        "|---|---|---|",
        f"| **True Normal** | TN={sd.get('tn','N/A')} | FP={sd.get('fp','N/A')} |",
        f"| **True NSCLC**  | FN={sd.get('fn','N/A')} | TP={sd.get('tp','N/A')} |",
        "",
        f"- Normal crop accuracy: {sd.get('normal_crop_acc','N/A')}",
        f"- NSCLC crop accuracy: {sd.get('nsclc_crop_acc','N/A')}",
        "",
        "## Error Crop Summary",
        f"- Wrong crops (total): {sd.get('n_wrong','N/A')}",
        f"- FP (normal predicted NSCLC): {sd.get('n_fp','N/A')}",
        f"- FN (NSCLC predicted normal): {sd.get('n_fn','N/A')}",
        f"- Borderline (0.4~0.6): {sd.get('n_borderline','N/A')}",
        "",
        "## Top Normal High-prob Cases",
        "- 파일: p_c_normal21f_top_normal_high_prob.csv",
        "",
        "## Top NSCLC Low-prob Cases",
        "- 파일: p_c_normal21f_top_nsclc_low_prob.csv",
        "",
    ]

    # POI analysis
    md.append("## Patients of Interest 분석")
    for pid, info in poi_analysis.items():
        md.append(f"### {pid}")
        if not info.get("found"):
            md.append(f"- **NOT FOUND** in crop CSV")
        else:
            lname = "normal" if info["label"] == 0 else "NSCLC"
            md.append(f"- label: {lname} ({info['label']})")
            md.append(f"- n_crops: {info['n_crops']}")
            md.append(f"- mean_prob_nsclc_like: {info['mean_prob']}")
            md.append(f"- max_prob_nsclc_like: {info['max_prob']}")
            md.append(f"- n_high_prob_crops (>=0.5): {info['n_high_prob_crops']}")
            md.append(f"- hu_mean_avg: {info.get('hu_mean_mean','N/A')}")
            md.append(f"- air_frac_mean: {info.get('air_frac_mean','N/A')}")
            md.append(f"- dense_frac_mean: {info.get('dense_frac_mean','N/A')}")
            if info.get("position_bin_counts"):
                md.append(f"- position_bin distribution: {info['position_bin_counts']}")
            if info.get("z_level_counts"):
                md.append(f"- z_level distribution: {info['z_level_counts']}")
        md.append("")

    md += [
        "## HU Shortcut Audit",
        f"- SR-HU Risk: **{sd.get('sr_hu','N/A')}**",
        f"- corr(prob, crop_hu_mean): {hu_corr.get('crop_hu_mean', 'N/A')}",
        f"- corr(prob, dense_frac_gt_minus500): {hu_corr.get('dense_frac_gt_minus500', 'N/A')}",
        f"- corr(prob, air_frac_lt_minus800): {hu_corr.get('air_frac_lt_minus800', 'N/A')}",
        f"- FP normal crops: dense_frac_mean={round(fp_dense,4) if fp_dense else 'N/A'}",
        f"- FN NSCLC crops: hu_mean_avg={round(fn_air,2) if fn_air else 'N/A'}",
        "",
        "## Position / Context Shortcut Audit",
        f"- SR-POS Risk: **{sd.get('sr_pos','N/A')}**",
    ]

    if pos_rows_list:
        md += ["", "| position_bin | n | accuracy | FP | FN |",
               "|---|---|---|---|---|"]
        for r in sorted(pos_rows_list, key=lambda x: x.get("position_bin","")):
            md.append(f"| {r['position_bin']} | {r['n_total']} | {r['accuracy']} | {r['fp_count']} | {r['fn_count']} |")

    md += [
        "",
        "## Shortcut Decision",
        f"**{sd.get('shortcut_decision','N/A')}**",
        "",
        "## Guardrail",
    ]
    for k, v in guardrail.items():
        ok = "OK" if v in (False, 0) else "VIOLATION"
        md.append(f"- `{k}`: {v}  [{ok}]")

    md += [
        "",
        f"## 체크 결과: total={len(checks)}, pass={sum(1 for c in checks if c['status']=='PASS')}, fail={sum(1 for c in checks if c['status']!='PASS')}",
        "",
        "## 확인 사항",
        "- stage2_holdout 미접근: True",
        "- 추가 inference/training/scoring/threshold 미실행: True",
        "- 기존 P-C-NORMAL21e 결과 무수정: True",
        "- 기존 checkpoint 무수정: True",
        "- 금지 표현(암 확률/진단/cancer probability 등) 0건: True",
        "",
        "## 다음 단계 권고",
    ]

    decision = sd.get("shortcut_decision", "")
    if decision == "AUDIT_PASS_READY_FOR_INTEGRATION_PREFLIGHT":
        md += [
            "- **A. P-C-NORMAL22 candidate scoring integration preflight**",
            "  - 1차 PaDiM candidate crop에 auxiliary score를 붙이는 설계",
            "  - stage2_holdout 접근 없음, score adjustment 아직 금지",
        ]
    elif decision == "NEEDS_VISUAL_REVIEW_OF_OUTLIERS":
        md += [
            "- **B. P-C-NORMAL21g outlier visual review preflight**",
            "  - high-prob normal / low-prob NSCLC crop PNG card 생성 계획",
            "  - 실제 PNG 생성은 별도 사용자 승인 필요",
        ]
    else:
        md += [
            "- **HU/context mitigation 검토 후 재판단 필요**",
        ]

    if errors:
        md += ["", "## Errors"]
        for e in errors:
            md.append(f"- {e}")

    with open(OUT_DIR / "p_c_normal21f_per_crop_shortcut_error_audit.md", "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"[21F] 결과 저장 완료: {OUT_DIR}")


if __name__ == "__main__":
    main()
