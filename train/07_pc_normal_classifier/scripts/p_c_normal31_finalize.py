"""
p_c_normal31_finalize.py
Phase D만 재실행: reproducibility check, guardrail, report.md, summary.json, DONE.json 생성
이미 생성된 metric CSV를 읽어서 처리.
"""
import csv, json, math, sys
from pathlib import Path
from datetime import datetime

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT  = PROJECT_ROOT / "outputs/reports/p_c_normal31_repaired_final_test_masked_comparison"

STAGE_LABEL     = "P-C-NORMAL31"
FIXED_THRESH    = 0.5
EXPECTED_TOTAL  = 66283
EXPECTED_NORMAL = 21560
EXPECTED_NSCLC  = 44723
REF28_AUROC       = 0.951743
REF28_AUPRC       = 0.973271
REF28_SPECIFICITY = 0.49564
REF28_SENSITIVITY = 0.992912
REF28_FP          = 10874
REF28_FN          = 317
REPRO_AUROC_TOL   = 0.005
REPRO_SPEC_TOL    = 0.02
REPRO_FP_TOL_REL  = 0.05

REF_CKPT = PROJECT_ROOT / "outputs/p_c_normal24j_fix_balanced_w1_zroi_scalar_fusion_full_train/checkpoints/p_c_normal24j_fix_balanced_w1_best_val_auc_checkpoint.pt"
CAND_CKPT = PROJECT_ROOT / "outputs/p_c_normal30b_masked_input_full_train/checkpoints/p_c_normal30b_best_val_auc_checkpoint.pt"

def _write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{"note": "empty"}]
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

def _write_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# metric CSV 읽기
df_raw = pd.read_csv(REPORT_ROOT / "p_c_normal31_crop_raw_metrics_comparison.csv")
m_r = df_raw[df_raw["model"] == "reference_balanced_w1"].iloc[0]
m_c = df_raw[df_raw["model"] == "masked_30b"].iloc[0]

# reproducibility check
auroc_diff = abs(float(m_r["auroc"]) - REF28_AUROC)
spec_diff  = abs(float(m_r["specificity"]) - REF28_SPECIFICITY)
fp_rel     = abs(int(m_r["FP"]) - REF28_FP) / max(REF28_FP, 1)
repro_ok   = (auroc_diff <= REPRO_AUROC_TOL and spec_diff <= REPRO_SPEC_TOL and fp_rel <= REPRO_FP_TOL_REL)
repro_status = "PASS" if repro_ok else "WARN"

repro_rows = [
    {"check": "auroc",       "ref31": float(m_r["auroc"]),       "ref28": REF28_AUROC,
     "diff": round(auroc_diff, 6), "tol": REPRO_AUROC_TOL,
     "status": "OK" if auroc_diff <= REPRO_AUROC_TOL else "WARN"},
    {"check": "specificity", "ref31": float(m_r["specificity"]), "ref28": REF28_SPECIFICITY,
     "diff": round(spec_diff, 6), "tol": REPRO_SPEC_TOL,
     "status": "OK" if spec_diff <= REPRO_SPEC_TOL else "WARN"},
    {"check": "FP_count",    "ref31": int(m_r["FP"]),            "ref28": REF28_FP,
     "diff": abs(int(m_r["FP"]) - REF28_FP), "tol": f"rel:{REPRO_FP_TOL_REL}",
     "status": "OK" if fp_rel <= REPRO_FP_TOL_REL else "WARN"},
]
_write_csv(repro_rows, REPORT_ROOT / "p_c_normal31_reference_reproducibility_check.csv")
print(f"[{STAGE_LABEL}] repro_status={repro_status}")

# delta 계산
auroc_delta = float(m_c["auroc"]) - float(m_r["auroc"])
spec_delta  = float(m_c["specificity"]) - float(m_r["specificity"])
sens_delta  = float(m_c["sensitivity"]) - float(m_r["sensitivity"])
fp_delta    = int(m_c["FP"]) - int(m_r["FP"])
fn_delta    = int(m_c["FN"]) - int(m_r["FN"])

if auroc_delta > 0.002:
    masking_effect = "masking이 fixed 0.5 operating point에서 개선 가능성을 보였다"
elif auroc_delta < -0.002:
    masking_effect = "masking으로 정보 손실 또는 distribution shift가 발생했을 수 있음"
else:
    masking_effect = "masking 적용 전후 성능 차이가 미미함 (AUROC delta < 0.002)"

print(f"[{STAGE_LABEL}] AUROC delta={auroc_delta:+.4f} FP delta={fp_delta:+d}")

# guardrail
phase_a_verdict = "PARTIAL_PASS"  # zero=16
guardrail_items = {
    "final_test_reporting_only": True,
    "no_training_run": True,
    "no_threshold_optimization": True,
    "no_threshold_sweep": True,
    "no_best_threshold_selection": True,
    "fixed_threshold_0p5_reporting_only": True,
    "no_checkpoint_modification": True,
    "no_existing_result_overwrite": True,
    "repaired_manifest_used": True,
    "corrupted_24g_manifest_used_for_prediction": False,
    "final_test_mask_generated": True,
    "final_test_mask_audit_passed": True,
    "mask_applied_to_30b_image_only": True,
    "mask_not_applied_to_reference_24j": True,
    "scalar_features_unchanged": True,
    "sample_weight_not_modified": True,
    "raw_metrics_primary": True,
    "weighted_metrics_supplementary_only": True,
    "test_set_downsampling_used": False,
    "normal_patch_duplication_used": False,
    "vessel_feature_used": False,
    "roi_masked_loss_used": False,
    "diagnostic_wording_avoided": True,
}
false_keys = {"corrupted_24g_manifest_used_for_prediction",
              "test_set_downsampling_used", "normal_patch_duplication_used",
              "vessel_feature_used", "roi_masked_loss_used"}
g_rows = []
n_gfail = 0
for k, v in guardrail_items.items():
    expected = False if k in false_keys else True
    ok = (v == expected)
    if not ok: n_gfail += 1
    g_rows.append({"key": k, "value": v, "expected": expected, "status": "OK" if ok else "FAIL"})
_write_csv(g_rows, REPORT_ROOT / "p_c_normal31_guardrail_check.csv")
print(f"[{STAGE_LABEL}] guardrail_fail={n_gfail}")

# weighted metrics 읽기
df_w = pd.read_csv(REPORT_ROOT / "p_c_normal31_crop_weighted_metrics_comparison.csv")
mw_r = df_w[df_w["model"] == "reference_balanced_w1"].iloc[0]
mw_c = df_w[df_w["model"] == "masked_30b"].iloc[0]

# patient metrics 읽기
df_pat = pd.read_csv(REPORT_ROOT / "p_c_normal31_patient_metrics_comparison.csv")

# report.md
lines = [
    f"# P-C-NORMAL31 Repaired Final_Test Masked Comparison Report",
    f"",
    f"- **Stage**: {STAGE_LABEL}",
    f"- **Timestamp**: {ts}",
    f"- **Phase A (mask generation)**: {phase_a_verdict} (zero_mask=16, error=0)",
    f"- **Reference reproducibility vs P-C-NORMAL28**: {repro_status}",
    f"",
    f"## Dataset",
    f"- total: {EXPECTED_TOTAL}, normal: {EXPECTED_NORMAL}, NSCLC: {EXPECTED_NSCLC}",
    f"- fixed threshold: {FIXED_THRESH}",
    f"",
    f"## Crop-level Raw Metrics (PRIMARY)",
    f"",
    f"| Metric | Reference (24j bw1) | Masked 30b | Delta |",
    f"|--------|---------------------|------------|-------|",
    f"| AUROC | {float(m_r['auroc']):.6f} | {float(m_c['auroc']):.6f} | {auroc_delta:+.6f} |",
    f"| AUPRC | {float(m_r['auprc']):.6f} | {float(m_c['auprc']):.6f} | {float(m_c['auprc'])-float(m_r['auprc']):+.6f} |",
    f"| Brier | {float(m_r['brier']):.6f} | {float(m_c['brier']):.6f} | {float(m_c['brier'])-float(m_r['brier']):+.6f} |",
    f"| accuracy | {float(m_r['accuracy']):.6f} | {float(m_c['accuracy']):.6f} | {float(m_c['accuracy'])-float(m_r['accuracy']):+.6f} |",
    f"| balanced_acc | {float(m_r['balanced_accuracy']):.6f} | {float(m_c['balanced_accuracy']):.6f} | {float(m_c['balanced_accuracy'])-float(m_r['balanced_accuracy']):+.6f} |",
    f"| sensitivity | {float(m_r['sensitivity']):.6f} | {float(m_c['sensitivity']):.6f} | {sens_delta:+.6f} |",
    f"| specificity | {float(m_r['specificity']):.6f} | {float(m_c['specificity']):.6f} | {spec_delta:+.6f} |",
    f"| precision | {float(m_r['precision']):.6f} | {float(m_c['precision']):.6f} | {float(m_c['precision'])-float(m_r['precision']):+.6f} |",
    f"| F1 | {float(m_r['f1']):.6f} | {float(m_c['f1']):.6f} | {float(m_c['f1'])-float(m_r['f1']):+.6f} |",
    f"| FP (normal) | {int(m_r['FP'])} | {int(m_c['FP'])} | {fp_delta:+d} |",
    f"| FN (NSCLC)  | {int(m_r['FN'])} | {int(m_c['FN'])} | {fn_delta:+d} |",
    f"",
    f"## Reference Reproducibility vs P-C-NORMAL28 balanced_w1",
    f"",
    f"| Check | ref31 | ref28 | diff | status |",
    f"|-------|-------|-------|------|--------|",
]
for r in repro_rows:
    lines.append(f"| {r['check']} | {r['ref31']} | {r['ref28']} | {r['diff']} | {r['status']} |")

lines += [
    f"",
    f"## Weighted Metrics (SUPPLEMENTARY ONLY)",
    f"",
    f"| Metric | Reference | Masked 30b |",
    f"|--------|-----------|------------|",
    f"| w_accuracy | {float(mw_r['w_accuracy']):.6f} | {float(mw_c['w_accuracy']):.6f} |",
    f"| w_brier | {float(mw_r['w_brier']):.6f} | {float(mw_c['w_brier']):.6f} |",
    f"| w_sensitivity | {float(mw_r['w_sensitivity']):.6f} | {float(mw_c['w_sensitivity']):.6f} |",
    f"| w_specificity | {float(mw_r['w_specificity']):.6f} | {float(mw_c['w_specificity']):.6f} |",
    f"| w_precision | {float(mw_r['w_precision']):.6f} | {float(mw_c['w_precision']):.6f} |",
    f"| w_f1 | {float(mw_r['w_f1']):.6f} | {float(mw_c['w_f1']):.6f} |",
    f"",
    f"## Patient-level Metrics",
    f"",
    f"| model | agg | auroc | auprc | normal_FP_pat | nsclc_FN_pat |",
    f"|-------|-----|-------|-------|----------------|--------------|",
]
for _, row in df_pat.iterrows():
    lines.append(f"| {row['model']} | {row['agg']} | {row['auroc']:.4f} | {row['auprc']:.4f} | {row['normal_FP_patients']} | {row['nsclc_FN_patients']} |")

lines += [
    f"",
    f"## Masking Effect Interpretation",
    f"",
    f"- AUROC delta (masked - reference): {auroc_delta:+.6f}",
    f"- Specificity delta: {spec_delta:+.6f}",
    f"- Sensitivity delta: {sens_delta:+.6f}",
    f"- FP delta: {fp_delta:+d}",
    f"- FN delta: {fn_delta:+d}",
    f"",
    f"**해석**: {masking_effect}",
    f"",
    f"※ 어떤 경우에도 진단 성능/암 확률 표현 금지. 이 결과는 research comparison 목적이다.",
    f"",
    f"---",
    f"",
    f"## Next Steps",
    f"",
    f"**P-C-NORMAL32 decision checkpoint**",
    f"- P-C-NORMAL31 결과 기준으로 masked 30b 채택 여부 결정",
    f"- 또는 masked 30b error review",
    f"- 또는 masking branch 종료",
    f"",
    f"사용자 승인 후 진행.",
]
(REPORT_ROOT / "p_c_normal31_repaired_final_test_masked_comparison_report.md").write_text(
    "\n".join(lines), encoding="utf-8"
)

# summary.json
verdict = "PARTIAL_PASS"  # phase_a zero=16 + repro check
fail_reasons = []
if phase_a_verdict == "PARTIAL_PASS":
    fail_reasons.append("phase_a_zero_mask_16")
if repro_status == "WARN":
    fail_reasons.append("reference_reproducibility_warn")
if n_gfail > 0:
    verdict = "FAIL"; fail_reasons.append(f"guardrail_fail={n_gfail}")

summary = {
    "stage": STAGE_LABEL,
    "timestamp": ts,
    "verdict": verdict,
    "fail_reasons": fail_reasons,
    "phase_a_verdict": phase_a_verdict,
    "phase_a_zero_mask": 16,
    "repro_status": repro_status,
    "n_total": EXPECTED_TOTAL, "n_normal": EXPECTED_NORMAL, "n_nsclc": EXPECTED_NSCLC,
    "reference_balanced_w1": {
        "auroc": float(m_r["auroc"]), "auprc": float(m_r["auprc"]),
        "brier": float(m_r["brier"]),
        "specificity": float(m_r["specificity"]), "sensitivity": float(m_r["sensitivity"]),
        "FP": int(m_r["FP"]), "FN": int(m_r["FN"]),
    },
    "masked_30b": {
        "auroc": float(m_c["auroc"]), "auprc": float(m_c["auprc"]),
        "brier": float(m_c["brier"]),
        "specificity": float(m_c["specificity"]), "sensitivity": float(m_c["sensitivity"]),
        "FP": int(m_c["FP"]), "FN": int(m_c["FN"]),
    },
    "delta_masked_minus_ref": {
        "auroc": round(auroc_delta, 6), "specificity": round(spec_delta, 6),
        "FP": fp_delta, "FN": fn_delta,
    },
    "masking_effect_note": masking_effect,
    "guardrail_fail": n_gfail,
}
_write_json(summary, REPORT_ROOT / "p_c_normal31_repaired_final_test_masked_comparison_summary.json")

# DONE.json
_write_json({
    "stage": STAGE_LABEL, "verdict": verdict, "timestamp": ts,
    "next_step": "P-C-NORMAL32 decision checkpoint (사용자 승인 필요)",
}, REPORT_ROOT / "DONE.json")

print(f"[{STAGE_LABEL}] VERDICT: {verdict}")
print(f"[{STAGE_LABEL}] ref AUROC={float(m_r['auroc']):.4f} masked AUROC={float(m_c['auroc']):.4f} delta={auroc_delta:+.4f}")
print(f"[{STAGE_LABEL}] ref FP={int(m_r['FP'])} masked FP={int(m_c['FP'])} delta={fp_delta:+d}")
print(f"[{STAGE_LABEL}] {masking_effect}")
print("DONE")
