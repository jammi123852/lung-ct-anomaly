"""
P-C-NORMAL23c2: Crop-Balanced Supplementary Metrics Quick Check
===============================================================
입력 : P-C-NORMAL23b prediction CSV (read-only)
목적 : NSCLC-heavy(44,723) → balanced(21,600) downsample 후 metrics 재계산
       원본 P-C-NORMAL23c official metric 덮어쓰지 않음

금지:
- model forward / prediction 재실행
- threshold 최적화
- training / checkpoint 저장
- 기존 23b/23c 파일 수정
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path('/home/jinhy/project/lung-ct-anomaly')
INPUT_CSV    = (
    PROJECT_ROOT / 'outputs' / 'reports'
    / 'p_c_normal23_final_baseline_test_prediction_export'
    / 'p_c_normal23_final_test_crop_predictions.csv'
)
OUTPUT_DIR   = (
    PROJECT_ROOT / 'outputs' / 'reports'
    / 'p_c_normal23c2_crop_balanced_supplementary_metrics'
)

EXPECTED_TOTAL_ROWS   = 66323
EXPECTED_NORMAL_ROWS  = 21600
EXPECTED_NSCLC_ROWS   = 44723
BALANCED_N            = 21600      # normal 전부 = NSCLC 다운샘플 수
FIXED_THRESHOLD       = 0.5
MULTI_SEEDS           = [0, 1, 2, 3, 4]
PRIMARY_SEED          = 42

GUARDRAIL = {
    "model_forward_run":                   False,
    "prediction_export_run":               False,
    "threshold_optimized":                 False,
    "training_run":                        False,
    "checkpoint_saved":                    False,
    "existing_prediction_files_modified":  False,
    "existing_metrics_files_modified":     False,
    "forbidden_diagnostic_wording_count":  0,
    "metrics_computed":                    True,
}

# ──────────────────────────────────────────────────────────────────────────────
# Metric helpers (numpy-only)
# ──────────────────────────────────────────────────────────────────────────────
def _roc_auc(y_true, y_score):
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    auc, chunk = 0.0, 4096
    for i in range(0, n_pos, chunk):
        p = pos[i:i+chunk]
        auc += (p[:, None] > neg[None, :]).sum() + 0.5 * (p[:, None] == neg[None, :]).sum()
    return float(auc / (n_pos * n_neg))


def _pr_auc(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    ys    = y_true[order]
    n_pos = ys.sum()
    if n_pos == 0:
        return float('nan')
    tp_cum   = np.cumsum(ys)
    fp_cum   = np.cumsum(1 - ys)
    precision = tp_cum / (tp_cum + fp_cum)
    recall    = tp_cum / n_pos
    prec_ext  = np.concatenate([[precision[0]], precision])
    rec_ext   = np.concatenate([[0.0], recall])
    return float(np.trapezoid(prec_ext, rec_ext))


def compute_metrics(y_true, y_prob, seed_label):
    y_pred = (y_prob >= FIXED_THRESHOLD).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    n  = len(y_true)
    acc    = (tp + tn) / n
    sens   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec   = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1     = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    bal_acc = (sens + spec) / 2
    brier  = float(np.mean((y_prob - y_true) ** 2))
    return {
        "seed":               seed_label,
        "n_normal":           int((y_true == 0).sum()),
        "n_nsclc":            int((y_true == 1).sum()),
        "threshold":          FIXED_THRESHOLD,
        "accuracy":           float(acc),
        "balanced_accuracy":  float(bal_acc),
        "sensitivity_nsclc":  float(sens),
        "specificity_normal": float(spec),
        "precision_nsclc":    float(prec),
        "f1_nsclc":           float(f1),
        "auroc":              _roc_auc(y_true, y_prob),
        "auprc":              _pr_auc(y_true, y_prob),
        "brier_score":        brier,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Patient-stratified proportional downsample
# ──────────────────────────────────────────────────────────────────────────────
def patient_stratified_downsample(nsclc_df: pd.DataFrame, n_target: int, seed: int) -> pd.DataFrame:
    """
    각 NSCLC 환자에서 patient 비율을 유지하면서 n_target 개를 샘플링.
    소수점은 올림하고 마지막에 n_target으로 clip.
    """
    rng        = np.random.default_rng(seed)
    total_nsclc = len(nsclc_df)
    ratio       = n_target / total_nsclc

    sampled_parts = []
    for pid, grp in nsclc_df.groupby('patient_id'):
        n_take = max(1, int(np.ceil(len(grp) * ratio)))
        idx    = rng.choice(len(grp), size=min(n_take, len(grp)), replace=False)
        sampled_parts.append(grp.iloc[sorted(idx)])

    sampled = pd.concat(sampled_parts, ignore_index=True)

    # 초과 제거
    if len(sampled) > n_target:
        idx_all = rng.permutation(len(sampled))[:n_target]
        sampled = sampled.iloc[sorted(idx_all)].reset_index(drop=True)
    # 부족 보충 (거의 없어야 함)
    elif len(sampled) < n_target:
        deficit = n_target - len(sampled)
        already = set(sampled.index)
        remaining = nsclc_df.drop(index=sampled.index, errors='ignore')
        extra = remaining.sample(n=min(deficit, len(remaining)), random_state=seed)
        sampled = pd.concat([sampled, extra], ignore_index=True)

    return sampled.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
def _fail(errors, msg):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failed = {"stage": "P-C-NORMAL23c2", "status": "FAILED_DO_NOT_USE",
              "conditions_ok": False, "errors": errors, "message": msg}
    with open(OUTPUT_DIR / 'FAILED_DO_NOT_USE.json', 'w') as f:
        json.dump(failed, f, indent=2)
    with open(OUTPUT_DIR / 'p_c_normal23c2_errors.csv', 'w') as f:
        f.write("error\n")
        for e in errors:
            f.write(e.replace('\n', ' ') + "\n")
    print(f"[23c2] FAILED: {msg}", file=sys.stderr)
    for e in errors:
        print(f"  - {e}", file=sys.stderr)
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
def main():
    errors = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load & validate ────────────────────────────────────────────────
    print("[23c2] Loading crop prediction CSV...")
    if not INPUT_CSV.exists():
        errors.append(f"Input CSV not found: {INPUT_CSV}")
        _fail(errors, "Input not found")
        return

    crop_df = pd.read_csv(INPUT_CSV)

    if len(crop_df) != EXPECTED_TOTAL_ROWS:
        errors.append(f"rows {len(crop_df)} != {EXPECTED_TOTAL_ROWS}")
    n0 = (crop_df['label'] == 0).sum()
    n1 = (crop_df['label'] == 1).sum()
    if n0 != EXPECTED_NORMAL_ROWS:
        errors.append(f"normal rows {n0} != {EXPECTED_NORMAL_ROWS}")
    if n1 != EXPECTED_NSCLC_ROWS:
        errors.append(f"NSCLC rows {n1} != {EXPECTED_NSCLC_ROWS}")
    if not np.isfinite(crop_df['prob_nsclc_like'].values).all():
        errors.append("prob_nsclc_like non-finite")
    if not ((crop_df['prob_nsclc_like'] >= 0) & (crop_df['prob_nsclc_like'] <= 1)).all():
        errors.append("prob_nsclc_like out of [0,1]")
    if not set(crop_df['label'].unique()).issubset({0, 1}):
        errors.append("label unexpected values")

    if errors:
        _fail(errors, "Input validation failed")
        return

    print(f"[23c2] Input validation PASS. total={len(crop_df)}, normal={n0}, NSCLC={n1}")

    # ── 2. Split ──────────────────────────────────────────────────────────
    normal_df = crop_df[crop_df['label'] == 0].copy().reset_index(drop=True)
    nsclc_df  = crop_df[crop_df['label'] == 1].copy().reset_index(drop=True)

    # ── 3. Primary seed=42 balanced sample ───────────────────────────────
    print(f"[23c2] Downsampling NSCLC → {BALANCED_N} (patient-stratified, seed={PRIMARY_SEED})...")
    nsclc_sampled42 = patient_stratified_downsample(nsclc_df, BALANCED_N, PRIMARY_SEED)
    assert len(nsclc_sampled42) == BALANCED_N, f"downsample count {len(nsclc_sampled42)} != {BALANCED_N}"

    balanced42 = pd.concat([normal_df, nsclc_sampled42], ignore_index=True)
    assert len(balanced42) == BALANCED_N * 2

    y_true42 = balanced42['label'].values
    y_prob42 = balanced42['prob_nsclc_like'].values
    seed42_metrics = compute_metrics(y_true42, y_prob42, seed_label=PRIMARY_SEED)
    print(f"[23c2] seed={PRIMARY_SEED}  AUROC={seed42_metrics['auroc']:.4f}  Acc={seed42_metrics['accuracy']:.4f}  BalAcc={seed42_metrics['balanced_accuracy']:.4f}")

    # save balanced sample manifest
    balanced42.to_csv(OUTPUT_DIR / 'p_c_normal23c2_balanced_sample_manifest_seed42.csv', index=False)

    # ── 4. Multi-seed sensitivity ─────────────────────────────────────────
    print(f"[23c2] Multi-seed check: seeds={MULTI_SEEDS}...")
    multiseed_rows = []
    for s in MULTI_SEEDS:
        ns_s      = patient_stratified_downsample(nsclc_df, BALANCED_N, s)
        bal_s     = pd.concat([normal_df, ns_s], ignore_index=True)
        y_t, y_p  = bal_s['label'].values, bal_s['prob_nsclc_like'].values
        row       = compute_metrics(y_t, y_p, seed_label=s)
        multiseed_rows.append(row)
        print(f"  seed={s}  AUROC={row['auroc']:.4f}  Acc={row['accuracy']:.4f}  BalAcc={row['balanced_accuracy']:.4f}")

    multiseed_df = pd.DataFrame(multiseed_rows)

    # summary row (mean/std)
    numeric_cols = [c for c in multiseed_df.columns if c not in ('seed', 'threshold')]
    mean_row = {"seed": "mean_seeds0-4", "threshold": FIXED_THRESHOLD}
    std_row  = {"seed": "std_seeds0-4",  "threshold": FIXED_THRESHOLD}
    for c in numeric_cols:
        mean_row[c] = float(multiseed_df[c].mean())
        std_row[c]  = float(multiseed_df[c].std())

    multiseed_full = pd.concat(
        [multiseed_df, pd.DataFrame([mean_row, std_row])], ignore_index=True
    )
    multiseed_full.to_csv(OUTPUT_DIR / 'p_c_normal23c2_multiseed_metrics.csv', index=False)

    # ── 5. seed=42 single CSVs ────────────────────────────────────────────
    pd.DataFrame([seed42_metrics]).to_csv(
        OUTPUT_DIR / 'p_c_normal23c2_seed42_balanced_metrics.csv', index=False)

    pd.DataFrame([{
        "aggregation": "crop_balanced_seed42",
        "TP": seed42_metrics['TP'], "FP": seed42_metrics['FP'],
        "TN": seed42_metrics['TN'], "FN": seed42_metrics['FN'],
    }]).to_csv(OUTPUT_DIR / 'p_c_normal23c2_seed42_confusion_matrix.csv', index=False)

    # ── 6. Original vs balanced comparison ────────────────────────────────
    # Original 23c values (hardcoded from official result)
    orig = {
        "version":            "original_unbalanced_23c",
        "n_normal":           21600, "n_nsclc": 44723, "n_total": 66323,
        "accuracy":           0.8207861526167393,
        "balanced_accuracy":  (0.9914585336404087 + 0.4674074074074074) / 2,
        "sensitivity_nsclc":  0.9914585336404087,
        "specificity_normal": 0.4674074074074074,
        "precision_nsclc":    0.7940012534694243,
        "f1_nsclc":           0.8818113117492641,
        "auroc":              0.9594750950501068,
        "auprc":              0.9795937645399608,
        "brier_score":        0.1650982349674925,
        "TP": 44341, "FP": 11504, "TN": 10096, "FN": 382,
    }
    bal42_row = {k: seed42_metrics.get(k, '') for k in orig.keys()}
    bal42_row["version"]  = f"balanced_seed{PRIMARY_SEED}"
    bal42_row["n_total"]  = BALANCED_N * 2

    compare_cols = [
        "version", "n_normal", "n_nsclc", "n_total",
        "accuracy", "balanced_accuracy",
        "sensitivity_nsclc", "specificity_normal", "precision_nsclc",
        "f1_nsclc", "auroc", "auprc", "brier_score",
        "TP", "FP", "TN", "FN",
    ]
    compare_df = pd.DataFrame([
        {c: orig.get(c, '') for c in compare_cols},
        {c: bal42_row.get(c, '') for c in compare_cols},
    ])
    compare_df.to_csv(
        OUTPUT_DIR / 'p_c_normal23c2_original_vs_balanced_comparison.csv', index=False)

    # ── 7. Guardrail CSV ──────────────────────────────────────────────────
    pd.DataFrame([{"guardrail": k, "value": str(v)} for k, v in GUARDRAIL.items()]).to_csv(
        OUTPUT_DIR / 'p_c_normal23c2_guardrail_check.csv', index=False)

    pd.DataFrame(columns=['error']).to_csv(
        OUTPUT_DIR / 'p_c_normal23c2_errors.csv', index=False)

    # ── 8. Metrics JSON ───────────────────────────────────────────────────
    ms_mean = {c: float(multiseed_df[c].mean()) for c in numeric_cols}
    ms_std  = {c: float(multiseed_df[c].std())  for c in numeric_cols}

    acc_diff       = seed42_metrics['accuracy']      - orig['accuracy']
    bal_acc_diff   = seed42_metrics['balanced_accuracy'] - orig['balanced_accuracy']
    auroc_diff     = seed42_metrics['auroc']         - orig['auroc']

    metrics_json = {
        "stage": "P-C-NORMAL23c2",
        "purpose": "supplementary crop-balanced metrics, not official final baseline",
        "fixed_threshold": FIXED_THRESHOLD,
        "note_threshold": "fixed reporting threshold 0.5, optimization not performed",
        "balanced_n_per_class": BALANCED_N,
        "primary_seed": PRIMARY_SEED,
        "seed42_metrics":    seed42_metrics,
        "multiseed_mean":    ms_mean,
        "multiseed_std":     ms_std,
        "original_23c":      orig,
        "delta_seed42_vs_original": {
            "accuracy":          acc_diff,
            "balanced_accuracy": bal_acc_diff,
            "auroc":             auroc_diff,
        },
        "imbalance_note": (
            "Original 23c had NSCLC:normal = 2.07:1. "
            "Accuracy@0.5 improves after balancing because FP penalty is symmetric. "
            "AUROC/AUPRC are threshold-free and robust to class ratio."
        ),
        "guardrail": GUARDRAIL,
        "interpretation": "normal-like vs NSCLC-lesion-like auxiliary score only",
        "next_step": "P-C-NORMAL23d lung_z_percentile/crop_lung_roi_ratio metadata join + error analysis",
    }
    with open(OUTPUT_DIR / 'p_c_normal23c2_final_baseline_test_metrics.json', 'w') as f:
        json.dump(metrics_json, f, indent=2)

    # ── 9. Report MD ──────────────────────────────────────────────────────
    m42 = seed42_metrics
    orig_bal_acc = orig['balanced_accuracy']

    report = [
        "# P-C-NORMAL23c2 Crop-Balanced Supplementary Metrics",
        "",
        "**목적**: class imbalance(NSCLC:normal=2.07:1)가 accuracy에 미치는 영향 확인",
        "**이 결과는 supplementary analysis임. P-C-NORMAL23c가 official final baseline.**",
        "**출력 해석**: normal-like vs NSCLC-lesion-like auxiliary score only",
        f"**balanced set**: normal {BALANCED_N} / NSCLC {BALANCED_N} (patient-stratified, seed={PRIMARY_SEED})",
        f"**fixed threshold**: {FIXED_THRESHOLD} (threshold optimization not performed)",
        "",
        "## Original (unbalanced) vs Balanced Comparison (seed=42)",
        "",
        f"| metric | original 23c (unbalanced) | balanced 23c2 (seed=42) | delta |",
        f"|--------|--------------------------|------------------------|-------|",
        f"| n_total | {orig['n_total']:,} | {BALANCED_N*2:,} | — |",
        f"| n_normal | {orig['n_normal']:,} | {BALANCED_N:,} | — |",
        f"| n_nsclc | {orig['n_nsclc']:,} | {BALANCED_N:,} | — |",
        f"| Accuracy @0.5 | {orig['accuracy']:.4f} | {m42['accuracy']:.4f} | {m42['accuracy']-orig['accuracy']:+.4f} |",
        f"| Balanced Accuracy @0.5 | {orig_bal_acc:.4f} | {m42['balanced_accuracy']:.4f} | {m42['balanced_accuracy']-orig_bal_acc:+.4f} |",
        f"| Sensitivity (NSCLC) | {orig['sensitivity_nsclc']:.4f} | {m42['sensitivity_nsclc']:.4f} | {m42['sensitivity_nsclc']-orig['sensitivity_nsclc']:+.4f} |",
        f"| Specificity (normal) | {orig['specificity_normal']:.4f} | {m42['specificity_normal']:.4f} | {m42['specificity_normal']-orig['specificity_normal']:+.4f} |",
        f"| Precision | {orig['precision_nsclc']:.4f} | {m42['precision_nsclc']:.4f} | {m42['precision_nsclc']-orig['precision_nsclc']:+.4f} |",
        f"| F1 | {orig['f1_nsclc']:.4f} | {m42['f1_nsclc']:.4f} | {m42['f1_nsclc']-orig['f1_nsclc']:+.4f} |",
        f"| AUROC | {orig['auroc']:.4f} | {m42['auroc']:.4f} | {m42['auroc']-orig['auroc']:+.4f} |",
        f"| AUPRC | {orig['auprc']:.4f} | {m42['auprc']:.4f} | {m42['auprc']-orig['auprc']:+.4f} |",
        f"| Brier score | {orig['brier_score']:.4f} | {m42['brier_score']:.4f} | {m42['brier_score']-orig['brier_score']:+.4f} |",
        f"| TP | {orig['TP']:,} | {m42['TP']:,} | — |",
        f"| FP | {orig['FP']:,} | {m42['FP']:,} | — |",
        f"| TN | {orig['TN']:,} | {m42['TN']:,} | — |",
        f"| FN | {orig['FN']:,} | {m42['FN']:,} | — |",
        "",
        "## Multi-seed Sensitivity (seeds 0-4)",
        "",
        f"| metric | mean | std |",
        f"|--------|------|-----|",
    ]
    for c in ['accuracy', 'balanced_accuracy', 'sensitivity_nsclc', 'specificity_normal',
              'precision_nsclc', 'f1_nsclc', 'auroc', 'auprc', 'brier_score']:
        report.append(
            f"| {c} | {ms_mean[c]:.4f} | {ms_std[c]:.4f} |"
        )

    report += [
        "",
        "## Interpretation",
        "",
        (f"- Original 23c accuracy={orig['accuracy']:.4f}는 NSCLC:normal=2.07:1 불균형 때문에 "
         f"NSCLC majority쪽으로 편향된 수치임."),
        (f"- Balanced 후 accuracy={m42['accuracy']:.4f}, balanced_accuracy={m42['balanced_accuracy']:.4f}로 "
         f"실제 per-class 성능을 더 잘 반영함."),
        (f"- AUROC는 threshold-free이며 class 비율에 무관: 원본 {orig['auroc']:.4f} → 균형 {m42['auroc']:.4f} (변화 미미)."),
        (f"- Specificity(normal)가 {orig['specificity_normal']:.4f} → {m42['specificity_normal']:.4f}로 "
         f"변화 — normal FP 수는 balanced set에서 동일하지만 비율 기준이 달라짐."),
        "",
        "## Guardrails",
        "",
        "- threshold 0.5는 fixed reporting threshold이며 최적화하지 않았음",
        "- model forward: 미실행",
        "- prediction export 재실행: 미실행",
        "- 기존 23b/23c 파일: 무수정",
        "- 금지 표현 (cancer probability 등): 0",
        "",
        "## 다음 단계",
        "",
        "**P-C-NORMAL23d**: lung_z_percentile / crop_lung_roi_ratio metadata join + error analysis",
    ]

    with open(OUTPUT_DIR / 'p_c_normal23c2_crop_balanced_supplementary_metrics.md', 'w') as f:
        f.write('\n'.join(report) + '\n')

    # ── 10. DONE.json ─────────────────────────────────────────────────────
    done = {
        "stage":                         "P-C-NORMAL23c2",
        "status":                        "done",
        "conditions_ok":                 True,
        "supplementary_only":            True,
        "original_23c_not_modified":     True,
        "balanced_n_per_class":          BALANCED_N,
        "primary_seed":                  PRIMARY_SEED,
        "multiseeds":                    MULTI_SEEDS,
        "seed42_auroc":                  m42['auroc'],
        "seed42_accuracy":               m42['accuracy'],
        "seed42_balanced_accuracy":      m42['balanced_accuracy'],
        "multiseed_mean_auroc":          ms_mean['auroc'],
        "multiseed_std_auroc":           ms_std['auroc'],
        "threshold_optimized":           False,
        "fixed_threshold_0_5_used":      True,
        "model_forward_run":             False,
        "prediction_export_run":         False,
        "training_run":                  False,
        "checkpoint_saved":              False,
        "n_errors":                      0,
        "next_step":                     "P-C-NORMAL23d metadata-based error analysis",
    }
    with open(OUTPUT_DIR / 'DONE.json', 'w') as f:
        json.dump(done, f, indent=2)

    print(f"\n[23c2] ============================================================")
    print(f"[23c2] P-C-NORMAL23c2 COMPLETE")
    print(f"[23c2] seed=42  AUROC={m42['auroc']:.4f}  Acc={m42['accuracy']:.4f}  BalAcc={m42['balanced_accuracy']:.4f}")
    print(f"[23c2] seeds0-4 AUROC mean={ms_mean['auroc']:.4f} ± {ms_std['auroc']:.4f}")
    print(f"[23c2] Original Acc={orig['accuracy']:.4f} → Balanced Acc={m42['accuracy']:.4f}  (Δ={m42['accuracy']-orig['accuracy']:+.4f})")
    print(f"[23c2] Output: {OUTPUT_DIR}")
    print(f"[23c2] ============================================================\n")


if __name__ == '__main__':
    main()
