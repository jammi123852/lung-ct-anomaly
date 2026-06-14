"""
P-C-NORMAL23c: Final Baseline Test Metrics Calculation
=======================================================
입력: P-C-NORMAL23b prediction CSVs (read-only)
출력: crop/patient metrics, FP/FN lists, error analysis handoff CSV

금지:
- model forward
- prediction export 재실행
- threshold 최적화
- training / backward / optimizer / checkpoint 저장
- crop/manifest 생성
- 기존 prediction 파일 수정
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path('/home/jinhy/project/lung-ct-anomaly')
INPUT_DIR = (
    PROJECT_ROOT / 'outputs' / 'reports'
    / 'p_c_normal23_final_baseline_test_prediction_export'
)
OUTPUT_DIR = (
    PROJECT_ROOT / 'outputs' / 'reports'
    / 'p_c_normal23c_final_baseline_test_metrics'
)

CROP_PRED_CSV    = INPUT_DIR / 'p_c_normal23_final_test_crop_predictions.csv'
PATIENT_PRED_CSV = INPUT_DIR / 'p_c_normal23_final_test_patient_predictions.csv'
DONE_23B_JSON    = INPUT_DIR / 'DONE.json'

# ──────────────────────────────────────────────────────────────────────────────
# Expectations
# ──────────────────────────────────────────────────────────────────────────────
EXPECTED_CROP_ROWS       = 66323
EXPECTED_PATIENT_ROWS    = 158
EXPECTED_NORMAL_ROWS     = 21600
EXPECTED_NSCLC_ROWS      = 44723
EXPECTED_NORMAL_PATIENTS = 36
EXPECTED_NSCLC_PATIENTS  = 122
FIXED_THRESHOLD          = 0.5

# ──────────────────────────────────────────────────────────────────────────────
# Guardrail (this stage)
# ──────────────────────────────────────────────────────────────────────────────
GUARDRAIL = {
    "prediction_export_run":           False,
    "model_forward_run":               False,
    "threshold_optimized":             False,
    "training_run":                    False,
    "backward_run":                    False,
    "optimizer_step":                  False,
    "checkpoint_saved":                False,
    "crop_generated":                  False,
    "manifest_generated":              False,
    "existing_prediction_files_modified": False,
    "forbidden_diagnostic_wording_count": 0,
    "metrics_computed":                True,
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
HANDOFF_CROP_COLS = [
    'patient_id', 'safe_id', 'label', 'label_name',
    'prob_nsclc_like', 'pred_label_threshold_0_5',
    'source_split', 'position_bin', 'z_level', 'local_z',
    'slice_index', 'center_y', 'center_x', 'crop_path',
    'crop_hu_mean', 'crop_hu_std',
    'air_frac_lt_minus800', 'dense_frac_gt_minus500',
    'dense_frac_gt_minus300', 'positive_frac_gt_0',
]


# ──────────────────────────────────────────────────────────────────────────────
# Metric helpers (numpy-only, no sklearn)
# ──────────────────────────────────────────────────────────────────────────────
def _roc_auc(y_true, y_score):
    """Mann-Whitney U AUC."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    # count pairs where pos > neg (vectorised, chunked to avoid OOM)
    auc = 0.0
    chunk = 4096
    for i in range(0, n_pos, chunk):
        p_chunk = pos[i:i+chunk]
        wins = (p_chunk[:, None] > neg[None, :]).sum()
        ties = (p_chunk[:, None] == neg[None, :]).sum()
        auc += wins + 0.5 * ties
    return float(auc / (n_pos * n_neg))


def _pr_auc(y_true, y_score):
    """Area under precision-recall curve (trapezoid, sorted by decreasing score)."""
    order = np.argsort(y_score)[::-1]
    ys = y_true[order]
    n_pos = ys.sum()
    if n_pos == 0:
        return float('nan')
    tp_cum   = np.cumsum(ys)
    fp_cum   = np.cumsum(1 - ys)
    precision = tp_cum / (tp_cum + fp_cum)
    recall    = tp_cum / n_pos
    # prepend (recall=0, prec=precision[0]) and append (recall=1, prec=0)
    prec_ext = np.concatenate([[precision[0]], precision])
    rec_ext  = np.concatenate([[0.0], recall])
    return float(np.trapz(prec_ext, rec_ext))


def _confusion(y_true, y_pred):
    """Returns (TN, FP, FN, TP) for binary {0,1}."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tn, fp, fn, tp


def prob_summary(probs, label, label_name):
    return {
        "label": label, "label_name": label_name,
        "count":  len(probs),
        "mean":   float(np.mean(probs)),
        "median": float(np.median(probs)),
        "p05":    float(np.percentile(probs, 5)),
        "p25":    float(np.percentile(probs, 25)),
        "p75":    float(np.percentile(probs, 75)),
        "p95":    float(np.percentile(probs, 95)),
        "min":    float(np.min(probs)),
        "max":    float(np.max(probs)),
    }


def compute_metrics(y_true, y_prob, y_pred, agg_name):
    tn, fp, fn, tp = _confusion(y_true, y_pred)
    n = len(y_true)
    acc    = (tp + tn) / n if n > 0 else 0.0
    sens   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec   = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1     = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    brier  = float(np.mean((y_prob - y_true) ** 2))
    return {
        "aggregation":        agg_name,
        "threshold":          FIXED_THRESHOLD,
        "n_total":            int(n),
        "accuracy":           float(acc),
        "sensitivity_nsclc":  float(sens),
        "specificity_normal": float(spec),
        "precision_nsclc":    float(prec),
        "f1_nsclc":           float(f1),
        "auroc":              _roc_auc(y_true, y_prob),
        "auprc":              _pr_auc(y_true, y_prob),
        "brier_score":        brier,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    }


def crop_rows_for_patients(crop_df, patient_ids):
    return crop_df[crop_df['patient_id'].isin(patient_ids)]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    errors = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. 23b DONE.json input validation ──────────────────────────────────
    print("[23c] Loading 23b DONE.json...")
    with open(DONE_23B_JSON) as f:
        done23b = json.load(f)

    checks = [
        ("conditions_ok",        done23b.get('conditions_ok')        == True),
        ("crop_prediction_rows", done23b.get('crop_prediction_rows') == EXPECTED_CROP_ROWS),
        ("patient_rows",         done23b.get('patient_prediction_rows') == EXPECTED_PATIENT_ROWS),
        ("n_errors",             done23b.get('n_errors')             == 0),
        ("prediction_export_run", done23b.get('prediction_export_run') == True),
        ("model_forward_run",    done23b.get('model_forward_run')    == True),
        ("threshold_optimized",  done23b.get('threshold_optimized')  == False),
        ("metrics_computed_False", done23b.get('metrics_computed')   == False),
        ("training_run",         done23b.get('training_run')         == False),
        ("checkpoint_saved",     done23b.get('checkpoint_saved')     == False),
    ]
    for name, ok in checks:
        if not ok:
            errors.append(f"23b DONE.json check failed: {name}")

    if errors:
        _fail(errors, "23b DONE.json validation failed")
        return

    # ── 2. Load CSVs ──────────────────────────────────────────────────────
    print("[23c] Loading prediction CSVs...")
    crop_df    = pd.read_csv(CROP_PRED_CSV)
    patient_df = pd.read_csv(PATIENT_PRED_CSV)

    # ── 3. Crop-level validation ──────────────────────────────────────────
    print("[23c] Validating crop predictions...")
    if len(crop_df) != EXPECTED_CROP_ROWS:
        errors.append(f"crop rows {len(crop_df)} != {EXPECTED_CROP_ROWS}")
    if (crop_df['label'] == 0).sum() != EXPECTED_NORMAL_ROWS:
        errors.append(f"normal crop rows mismatch")
    if (crop_df['label'] == 1).sum() != EXPECTED_NSCLC_ROWS:
        errors.append(f"NSCLC crop rows mismatch")
    if crop_df['patient_id'].nunique() != EXPECTED_PATIENT_ROWS:
        errors.append(f"patient count {crop_df['patient_id'].nunique()} != {EXPECTED_PATIENT_ROWS}")
    if not np.isfinite(crop_df['prob_nsclc_like'].values).all():
        errors.append("prob_nsclc_like contains non-finite values")
    if not ((crop_df['prob_nsclc_like'] >= 0) & (crop_df['prob_nsclc_like'] <= 1)).all():
        errors.append("prob_nsclc_like out of [0,1]")
    if not np.isfinite(crop_df['logit'].values).all():
        errors.append("logit contains non-finite values")
    if not set(crop_df['pred_label_threshold_0_5'].unique()).issubset({0, 1}):
        errors.append("pred_label_threshold_0_5 has unexpected values")
    if not set(crop_df['correct_threshold_0_5'].unique()).issubset({0, 1}):
        errors.append("correct_threshold_0_5 has unexpected values")
    if crop_df['msd_lung_flag'].sum() != 0:
        errors.append("msd_lung_flag has True rows")
    if crop_df['hard_negative_flag'].sum() != 0:
        errors.append("hard_negative_flag has True rows")
    if 'LUNG1-295' in crop_df['patient_id'].values:
        errors.append("LUNG1-295 present in crop_df")
    if 'LUNG1-415' in crop_df['patient_id'].values:
        errors.append("LUNG1-415 present in crop_df")
    if crop_df['crop_path'].nunique() != EXPECTED_CROP_ROWS:
        errors.append(f"crop_path unique {crop_df['crop_path'].nunique()} != {EXPECTED_CROP_ROWS}")

    # ── 4. Patient-level validation ───────────────────────────────────────
    print("[23c] Validating patient predictions...")
    if len(patient_df) != EXPECTED_PATIENT_ROWS:
        errors.append(f"patient rows {len(patient_df)} != {EXPECTED_PATIENT_ROWS}")
    if (patient_df['label'] == 0).sum() != EXPECTED_NORMAL_PATIENTS:
        errors.append(f"normal patients mismatch")
    if (patient_df['label'] == 1).sum() != EXPECTED_NSCLC_PATIENTS:
        errors.append(f"NSCLC patients mismatch")
    if patient_df['n_crops'].sum() != EXPECTED_CROP_ROWS:
        errors.append(f"n_crops sum {patient_df['n_crops'].sum()} != {EXPECTED_CROP_ROWS}")
    for col in ['mean_prob_nsclc_like', 'max_prob_nsclc_like',
                'median_prob_nsclc_like', 'p95_prob_nsclc_like']:
        if not np.isfinite(patient_df[col].values).all():
            errors.append(f"{col} non-finite")
        if not ((patient_df[col] >= 0) & (patient_df[col] <= 1)).all():
            errors.append(f"{col} out of [0,1]")
    if patient_df['msd_lung_flag_any'].sum() != 0:
        errors.append("msd_lung_flag_any has True rows")

    if errors:
        _fail(errors, "Input validation failed")
        return

    print(f"[23c] Input validation PASS. crops={len(crop_df)}, patients={len(patient_df)}")

    # ── 5. Crop-level metrics ─────────────────────────────────────────────
    print("[23c] Computing crop-level metrics...")
    y_true = crop_df['label'].values
    y_prob = crop_df['prob_nsclc_like'].values
    y_pred = crop_df['pred_label_threshold_0_5'].values

    crop_metrics = compute_metrics(y_true, y_prob, y_pred, "crop")

    # prob summary by label
    prob_summary_rows = [
        prob_summary(crop_df.loc[crop_df['label'] == 0, 'prob_nsclc_like'].values, 0, 'normal'),
        prob_summary(crop_df.loc[crop_df['label'] == 1, 'prob_nsclc_like'].values, 1, 'NSCLC'),
    ]

    # ── 6. Patient-level metrics ──────────────────────────────────────────
    print("[23c] Computing patient-level metrics...")
    patient_metrics_rows = []
    patient_cm_rows      = []
    patient_fp_fn_rows   = []

    for agg in ['mean', 'max', 'p95']:
        prob_col  = f"{agg}_prob_nsclc_like"
        pt_y_true = patient_df['label'].values
        pt_y_prob = patient_df[prob_col].values
        pt_y_pred = (pt_y_prob >= FIXED_THRESHOLD).astype(int)

        row = compute_metrics(pt_y_true, pt_y_prob, pt_y_pred, agg)
        patient_metrics_rows.append(row)
        patient_cm_rows.append({
            "aggregation": agg,
            "TP": row['TP'], "FP": row['FP'], "TN": row['TN'], "FN": row['FN'],
        })

        fp_mask = (patient_df['label'] == 0) & (pt_y_pred == 1)
        fn_mask = (patient_df['label'] == 1) & (pt_y_pred == 0)
        for _, prow in patient_df[fp_mask].iterrows():
            patient_fp_fn_rows.append({"aggregation": agg, "error_type": "FP",
                                       "prob_used": float(prow[prob_col]), **prow.to_dict()})
        for _, prow in patient_df[fn_mask].iterrows():
            patient_fp_fn_rows.append({"aggregation": agg, "error_type": "FN",
                                       "prob_used": float(prow[prob_col]), **prow.to_dict()})

    # ── 7. Crop FP/FN/TP/TN lists ─────────────────────────────────────────
    print("[23c] Generating FP/FN/TP/TN crop lists...")
    fp_crops = crop_df[(crop_df['label'] == 0) & (crop_df['pred_label_threshold_0_5'] == 1)].copy()
    fn_crops = crop_df[(crop_df['label'] == 1) & (crop_df['pred_label_threshold_0_5'] == 0)].copy()
    tp_crops = crop_df[(crop_df['label'] == 1) & (crop_df['pred_label_threshold_0_5'] == 1)].copy()
    tn_crops = crop_df[(crop_df['label'] == 0) & (crop_df['pred_label_threshold_0_5'] == 0)].copy()

    fp_top200 = fp_crops.sort_values('prob_nsclc_like', ascending=False).head(200).copy()
    fn_top200 = fn_crops.sort_values('prob_nsclc_like', ascending=True).head(200).copy()
    tp_top100 = tp_crops.sort_values('prob_nsclc_like', ascending=False).head(100).copy()
    tn_top100 = tn_crops.sort_values('prob_nsclc_like', ascending=True).head(100).copy()

    # ── 8. Error analysis handoff CSV ─────────────────────────────────────
    print("[23c] Building error analysis handoff CSV...")
    handoff_rows = []

    def add_crop_handoff(df, error_type, aggregation):
        for _, row in df.iterrows():
            d = {"level": "crop", "error_type": error_type, "aggregation": aggregation}
            for col in HANDOFF_CROP_COLS:
                d[col] = row[col] if col in row.index else np.nan
            handoff_rows.append(d)

    add_crop_handoff(fp_top200, "FP", "crop")
    add_crop_handoff(fn_top200, "FN", "crop")
    add_crop_handoff(tp_top100, "TP", "crop")
    add_crop_handoff(tn_top100, "TN", "crop")

    for agg in ['mean', 'max', 'p95']:
        prob_col  = f"{agg}_prob_nsclc_like"
        pt_y_pred = (patient_df[prob_col] >= FIXED_THRESHOLD).astype(int)
        fp_pids   = patient_df.loc[(patient_df['label'] == 0) & (pt_y_pred == 1), 'patient_id'].tolist()
        fn_pids   = patient_df.loc[(patient_df['label'] == 1) & (pt_y_pred == 0), 'patient_id'].tolist()

        for pids, etype in [(fp_pids, 'FP'), (fn_pids, 'FN')]:
            if not pids:
                continue
            subset = crop_df[crop_df['patient_id'].isin(pids)]
            for _, row in subset.iterrows():
                d = {"level": "patient", "error_type": etype, "aggregation": f"patient_{agg}"}
                for col in HANDOFF_CROP_COLS:
                    d[col] = row[col] if col in row.index else np.nan
                handoff_rows.append(d)

    handoff_df = pd.DataFrame(handoff_rows)

    # ── 9. Save all outputs ───────────────────────────────────────────────
    print("[23c] Saving outputs...")

    pd.DataFrame([crop_metrics]).to_csv(
        OUTPUT_DIR / 'p_c_normal23c_crop_level_metrics.csv', index=False)

    pd.DataFrame(patient_metrics_rows).to_csv(
        OUTPUT_DIR / 'p_c_normal23c_patient_level_metrics.csv', index=False)

    pd.DataFrame([{
        "level": "crop", "aggregation": "crop",
        "TP": crop_metrics['TP'], "FP": crop_metrics['FP'],
        "TN": crop_metrics['TN'], "FN": crop_metrics['FN'],
    }]).to_csv(OUTPUT_DIR / 'p_c_normal23c_crop_confusion_matrix.csv', index=False)

    pd.DataFrame(patient_cm_rows).to_csv(
        OUTPUT_DIR / 'p_c_normal23c_patient_confusion_matrices.csv', index=False)

    pd.DataFrame(prob_summary_rows).to_csv(
        OUTPUT_DIR / 'p_c_normal23c_prob_summary_by_label.csv', index=False)

    fp_top200.to_csv(OUTPUT_DIR / 'p_c_normal23c_crop_false_positive_top200.csv', index=False)
    fn_top200.to_csv(OUTPUT_DIR / 'p_c_normal23c_crop_false_negative_top200.csv', index=False)
    tp_top100.to_csv(OUTPUT_DIR / 'p_c_normal23c_crop_true_positive_top100.csv', index=False)
    tn_top100.to_csv(OUTPUT_DIR / 'p_c_normal23c_crop_true_negative_top100.csv', index=False)

    # patient FP/FN lists — split into FP and FN files
    if patient_fp_fn_rows:
        pt_fpfn_df = pd.DataFrame(patient_fp_fn_rows)
        pt_fpfn_df[pt_fpfn_df['error_type'] == 'FP'].to_csv(
            OUTPUT_DIR / 'p_c_normal23c_patient_false_positive_list.csv', index=False)
        pt_fpfn_df[pt_fpfn_df['error_type'] == 'FN'].to_csv(
            OUTPUT_DIR / 'p_c_normal23c_patient_false_negative_list.csv', index=False)
    else:
        empty_cols = ['aggregation', 'error_type', 'patient_id', 'label', 'prob_used']
        pd.DataFrame(columns=empty_cols).to_csv(
            OUTPUT_DIR / 'p_c_normal23c_patient_false_positive_list.csv', index=False)
        pd.DataFrame(columns=empty_cols).to_csv(
            OUTPUT_DIR / 'p_c_normal23c_patient_false_negative_list.csv', index=False)

    handoff_df.to_csv(OUTPUT_DIR / 'p_c_normal23c_error_analysis_candidates.csv', index=False)

    pd.DataFrame(columns=['error']).to_csv(
        OUTPUT_DIR / 'p_c_normal23c_errors.csv', index=False)

    pd.DataFrame([{"guardrail": k, "value": str(v)} for k, v in GUARDRAIL.items()]).to_csv(
        OUTPUT_DIR / 'p_c_normal23c_guardrail_check.csv', index=False)

    # ── 10. Metrics JSON ──────────────────────────────────────────────────
    metrics_json = {
        "stage": "P-C-NORMAL23c",
        "fixed_threshold": FIXED_THRESHOLD,
        "note_threshold": "fixed reporting threshold 0.5. threshold optimization not performed.",
        "crop_level_metrics":   crop_metrics,
        "patient_level_metrics": patient_metrics_rows,
        "prob_summary_by_label": prob_summary_rows,
        "fp_crop_count":  int(len(fp_crops)),
        "fn_crop_count":  int(len(fn_crops)),
        "tp_crop_count":  int(len(tp_crops)),
        "tn_crop_count":  int(len(tn_crops)),
        "patient_fp_fn_summary": {
            agg: {
                "FP": int(sum(1 for r in patient_fp_fn_rows if r['aggregation'] == agg and r['error_type'] == 'FP')),
                "FN": int(sum(1 for r in patient_fp_fn_rows if r['aggregation'] == agg and r['error_type'] == 'FN')),
            } for agg in ['mean', 'max', 'p95']
        },
        "guardrail": GUARDRAIL,
        "interpretation": "normal-like vs NSCLC-lesion-like auxiliary score only",
        "next_step": "P-C-NORMAL23d lung_z_percentile/crop_lung_roi_ratio metadata join + error analysis",
    }
    with open(OUTPUT_DIR / 'p_c_normal23c_final_baseline_test_metrics.json', 'w') as f:
        json.dump(metrics_json, f, indent=2)

    # ── 11. Report MD ─────────────────────────────────────────────────────
    cm = crop_metrics
    pm_mean = next(r for r in patient_metrics_rows if r['aggregation'] == 'mean')
    pm_max  = next(r for r in patient_metrics_rows if r['aggregation'] == 'max')
    pm_p95  = next(r for r in patient_metrics_rows if r['aggregation'] == 'p95')

    patient_fpfn_summary_lines = []
    for agg in ['mean', 'max', 'p95']:
        n_fp = sum(1 for r in patient_fp_fn_rows if r['aggregation'] == agg and r['error_type'] == 'FP')
        n_fn = sum(1 for r in patient_fp_fn_rows if r['aggregation'] == agg and r['error_type'] == 'FN')
        patient_fpfn_summary_lines.append(f"| patient_{agg} | {n_fp} | {n_fn} |")

    report_lines = [
        "# P-C-NORMAL23c Final Baseline Test Metrics",
        "",
        "**Branch**: supervised normal-vs-NSCLC auxiliary classifier (normal=0, NSCLC=1)",
        "**출력 해석**: normal-like vs NSCLC-lesion-like auxiliary score only",
        "**fixed threshold**: 0.5 (threshold optimization not performed)",
        "",
        "## Crop-level Metrics",
        "",
        f"| metric | value |",
        f"|--------|-------|",
        f"| AUROC | {cm['auroc']:.6f} |",
        f"| AUPRC | {cm['auprc']:.6f} |",
        f"| Brier score | {cm['brier_score']:.6f} |",
        f"| Accuracy @0.5 | {cm['accuracy']:.4f} |",
        f"| Sensitivity (NSCLC) @0.5 | {cm['sensitivity_nsclc']:.4f} |",
        f"| Specificity (normal) @0.5 | {cm['specificity_normal']:.4f} |",
        f"| Precision @0.5 | {cm['precision_nsclc']:.4f} |",
        f"| F1 @0.5 | {cm['f1_nsclc']:.4f} |",
        "",
        "### Confusion Matrix (crop @0.5)",
        "",
        f"| | pred=0 | pred=1 |",
        f"|--|--------|--------|",
        f"| true=0 (normal) | {cm['TN']} | {cm['FP']} |",
        f"| true=1 (NSCLC)  | {cm['FN']} | {cm['TP']} |",
        "",
        "## Patient-level Metrics",
        "",
        f"| aggregation | AUROC | AUPRC | Brier | Acc | Sens | Spec | Prec | F1 | TP | FP | TN | FN |",
        f"|-------------|-------|-------|-------|-----|------|------|------|----|----|----|----|----|",
        f"| mean  | {pm_mean['auroc']:.4f} | {pm_mean['auprc']:.4f} | {pm_mean['brier_score']:.4f} | {pm_mean['accuracy']:.4f} | {pm_mean['sensitivity_nsclc']:.4f} | {pm_mean['specificity_normal']:.4f} | {pm_mean['precision_nsclc']:.4f} | {pm_mean['f1_nsclc']:.4f} | {pm_mean['TP']} | {pm_mean['FP']} | {pm_mean['TN']} | {pm_mean['FN']} |",
        f"| max   | {pm_max['auroc']:.4f} | {pm_max['auprc']:.4f} | {pm_max['brier_score']:.4f} | {pm_max['accuracy']:.4f} | {pm_max['sensitivity_nsclc']:.4f} | {pm_max['specificity_normal']:.4f} | {pm_max['precision_nsclc']:.4f} | {pm_max['f1_nsclc']:.4f} | {pm_max['TP']} | {pm_max['FP']} | {pm_max['TN']} | {pm_max['FN']} |",
        f"| p95   | {pm_p95['auroc']:.4f} | {pm_p95['auprc']:.4f} | {pm_p95['brier_score']:.4f} | {pm_p95['accuracy']:.4f} | {pm_p95['sensitivity_nsclc']:.4f} | {pm_p95['specificity_normal']:.4f} | {pm_p95['precision_nsclc']:.4f} | {pm_p95['f1_nsclc']:.4f} | {pm_p95['TP']} | {pm_p95['FP']} | {pm_p95['TN']} | {pm_p95['FN']} |",
        "",
        "## FP/FN Counts",
        "",
        "### Crop-level",
        f"- FP (normal predicted as NSCLC): {len(fp_crops)}",
        f"- FN (NSCLC predicted as normal): {len(fn_crops)}",
        f"- TP: {len(tp_crops)}",
        f"- TN: {len(tn_crops)}",
        "",
        "### Patient-level",
        "",
        "| aggregation | FP patients | FN patients |",
        "|-------------|-------------|-------------|",
    ] + patient_fpfn_summary_lines + [
        "",
        "## Probability Summary by Label",
        "",
        "| label | count | mean | median | p05 | p25 | p75 | p95 | min | max |",
        "|-------|-------|------|--------|-----|-----|-----|-----|-----|-----|",
    ]
    for ps in prob_summary_rows:
        report_lines.append(
            f"| {ps['label_name']} | {ps['count']} | {ps['mean']:.4f} | {ps['median']:.4f} "
            f"| {ps['p05']:.4f} | {ps['p25']:.4f} | {ps['p75']:.4f} | {ps['p95']:.4f} "
            f"| {ps['min']:.4f} | {ps['max']:.4f} |"
        )

    report_lines += [
        "",
        "## Guardrails",
        "",
        "- threshold 0.5는 fixed reporting threshold이며 최적화하지 않았음",
        "- model forward: 미실행",
        "- prediction export 재실행: 미실행",
        "- training/backward/optimizer/checkpoint: 미실행",
        "- 금지 표현 (cancer probability / adenocarcinoma probability 등): 0",
        "",
        "## 다음 단계",
        "",
        "**P-C-NORMAL23d**: lung_z_percentile / crop_lung_roi_ratio metadata join + error analysis",
    ]

    with open(OUTPUT_DIR / 'p_c_normal23c_final_baseline_test_metrics.md', 'w') as f:
        f.write('\n'.join(report_lines) + '\n')

    # ── 12. DONE.json ─────────────────────────────────────────────────────
    done = {
        "stage":                       "P-C-NORMAL23c",
        "status":                      "done",
        "conditions_ok":               True,
        "input_crop_prediction_rows":  EXPECTED_CROP_ROWS,
        "input_patient_prediction_rows": EXPECTED_PATIENT_ROWS,
        "metrics_computed":            True,
        "threshold_optimized":         False,
        "fixed_threshold_0_5_used":    True,
        "model_forward_run":           False,
        "prediction_export_run":       False,
        "training_run":                False,
        "checkpoint_saved":            False,
        "error_analysis_handoff_created": True,
        "n_errors":                    0,
        "crop_auroc":                  crop_metrics['auroc'],
        "crop_auprc":                  crop_metrics['auprc'],
        "patient_mean_auroc":          pm_mean['auroc'],
        "patient_max_auroc":           pm_max['auroc'],
        "patient_p95_auroc":           pm_p95['auroc'],
        "fp_crop_count":               int(len(fp_crops)),
        "fn_crop_count":               int(len(fn_crops)),
        "next_step":                   "P-C-NORMAL23d metadata-based error analysis",
    }
    with open(OUTPUT_DIR / 'DONE.json', 'w') as f:
        json.dump(done, f, indent=2)

    # ── 13. Print summary ─────────────────────────────────────────────────
    print(f"\n[23c] ============================================================")
    print(f"[23c] P-C-NORMAL23c COMPLETE")
    print(f"[23c] Crop  AUROC={crop_metrics['auroc']:.4f}  AUPRC={crop_metrics['auprc']:.4f}  Brier={crop_metrics['brier_score']:.4f}")
    print(f"[23c] Crop  Acc={crop_metrics['accuracy']:.4f}  Sens={crop_metrics['sensitivity_nsclc']:.4f}  Spec={crop_metrics['specificity_normal']:.4f}")
    print(f"[23c] Crop  CM: TP={crop_metrics['TP']}  FP={crop_metrics['FP']}  TN={crop_metrics['TN']}  FN={crop_metrics['FN']}")
    for pm in patient_metrics_rows:
        print(f"[23c] Patient[{pm['aggregation']}] AUROC={pm['auroc']:.4f}  FP={pm['FP']}  FN={pm['FN']}")
    print(f"[23c] Output: {OUTPUT_DIR}")
    print(f"[23c] ============================================================\n")


# ──────────────────────────────────────────────────────────────────────────────
def _fail(errors, msg):
    failed = {
        "stage": "P-C-NORMAL23c",
        "status": "FAILED_DO_NOT_USE",
        "conditions_ok": False,
        "errors": errors,
        "message": msg,
    }
    fail_path = OUTPUT_DIR / 'FAILED_DO_NOT_USE.json'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(fail_path, 'w') as f:
        json.dump(failed, f, indent=2)
    err_path = OUTPUT_DIR / 'p_c_normal23c_errors.csv'
    with open(err_path, 'w') as f:
        f.write("error\n")
        for e in errors:
            f.write(e.replace('\n', ' ') + "\n")
    print(f"[23c] FAILED: {msg}", file=sys.stderr)
    for e in errors:
        print(f"  - {e}", file=sys.stderr)
    sys.exit(1)


if __name__ == '__main__':
    main()
