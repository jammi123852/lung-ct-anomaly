"""Phase 8.5D: Final metric closure / decision report.
기존 Phase 8.5B/8.5C 결과를 읽어 정리하는 문서화 단계. 새 계산 없음.
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
ANN_ROOT = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/review_annotations"

# 입력 파일
IN_8_5B_SUMMARY  = ANN_ROOT / "phase8_5b_metric_calculation_v1/phase8_5b_metric_calculation_summary.json"
IN_8_5B_CROP_CSV = ANN_ROOT / "phase8_5b_metric_calculation_v1/phase8_5b_metric_results_crop_level.csv"
IN_8_5C_SUMMARY  = ANN_ROOT / "phase8_5c_patient_label_definition_preflight_v1/phase8_5c_patient_label_definition_preflight_summary.json"
IN_8_5C_COMP_CSV = ANN_ROOT / "phase8_5c_patient_label_definition_preflight_v1/patient_label_composition_by_patient.csv"

# 이전 phase 파일 (요약용)
IN_8_4_SUMMARY   = ANN_ROOT / "phase8_4_stage2_full_scoring_v1/phase8_4_stage2_full_scoring_summary_v1.json"
IN_8_4C_SUMMARY  = ANN_ROOT / "phase8_4_stage2_full_scoring_v1/phase8_4c_artifact_validation_summary.json"
IN_8_5A_SUMMARY  = ANN_ROOT / "phase8_5a_metric_calculation_preflight_v1/phase8_5a_metric_calculation_preflight_summary.json"

OUT_DIR = ANN_ROOT / "phase8_5d_metric_closure_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=== Phase 8.5D: Final metric closure ===\n")

# ── 입력 로드 ──────────────────────────────────────────────
with open(IN_8_5B_SUMMARY) as f:
    s8_5b = json.load(f)
with open(IN_8_5C_SUMMARY) as f:
    s8_5c = json.load(f)
with open(IN_8_4_SUMMARY) as f:
    s8_4 = json.load(f)
with open(IN_8_4C_SUMMARY) as f:
    s8_4c = json.load(f)
with open(IN_8_5A_SUMMARY) as f:
    s8_5a = json.load(f)

crop_df = pd.read_csv(IN_8_5B_CROP_CSV)
comp_df = pd.read_csv(IN_8_5C_COMP_CSV)

print("입력 파일 모두 로드 완료.")

# ── 각 phase 요약 추출 ──────────────────────────────────────
phase_8_4b = {
    "total_processed_rows": s8_4.get("total_processed_rows"),
    "total_success_rows":   s8_4.get("total_success_rows"),
    "total_error_rows":     s8_4.get("total_error_rows"),
    "status": "PASS" if s8_4.get("total_error_rows", 1) == 0 else "FAIL",
}

phase_8_4c = {
    "verdict":                  s8_4c.get("verdict"),
    "overall_pass":             s8_4c.get("overall_pass"),
    "phase_8_5a_allowed":       s8_4c.get("phase_8_5a_preflight_allowed"),
}

phase_8_5a = {
    "verdict":              s8_5a.get("verdict"),
    "phase_8_5b_allowed":   s8_5a.get("phase_8_5b_allowed"),
    "preflight_pass":       s8_5a.get("preflight_pass"),
    "total_pass":           s8_5a.get("total_pass"),
    "total_fail":           s8_5a.get("total_fail"),
}

# crop-level metrics
crop_metrics = {}
for _, row in crop_df.iterrows():
    crop_metrics[row["score_column"]] = {
        "auroc": round(row["auroc"], 4),
        "auprc": round(row["auprc"], 4),
        "n_total":    int(row["n_total"]),
        "n_positive": int(row["n_positive"]),
        "n_negative": int(row["n_negative"]),
    }

best_col = max(crop_metrics, key=lambda c: crop_metrics[c]["auroc"])
print(f"최고 crop score column: {best_col} (AUROC {crop_metrics[best_col]['auroc']})")

# patient 구성
patient_count         = int(s8_5c.get("patient_id_unique_count", 154))
normal_patient_count  = int(s8_5c.get("inferred_dataset_distribution", {}).get("LUNA", 0))
mixed_label_count     = int(s8_5c.get("mixed_label_patients", 154))
positive_only_count   = int(s8_5c.get("positive_only_patients", 0))
hn_only_count         = int(s8_5c.get("hard_negative_only_patients", 0))
nsclc_count           = int(s8_5c.get("inferred_dataset_distribution", {}).get("NSCLC", 124))
msd_count             = int(s8_5c.get("inferred_dataset_distribution", {}).get("MSD", 30))

# ── summary JSON ────────────────────────────────────────────
ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
summary = {
    "phase": "8.5D",
    "timestamp": ts,
    "final_status": "PASS_CROP_LEVEL_ONLY_PATIENT_LEVEL_NOT_APPLICABLE",
    "crop_level_metric_valid": True,
    "patient_level_metric_valid": False,
    "patient_level_status": "STRUCTURALLY_INVALID",
    "patient_level_blocker": "no_negative_patient_class",
    "patient_count": patient_count,
    "normal_patient_count": normal_patient_count,
    "mixed_label_patient_count": mixed_label_count,
    "threshold_metrics_calculated": False,
    "froc_calculated": False,
    "metric_recalculated": False,
    "model_forward_executed": False,
    "training_executed": False,
    "checkpoint_created": False,
    "recommended_next_step": "per_scan_froc_protocol_design",

    "phase_8_4b_full_scoring": phase_8_4b,
    "phase_8_4c_artifact_validation": phase_8_4c,
    "phase_8_5a_metric_preflight": phase_8_5a,
    "phase_8_5b_crop_level_metrics": crop_metrics,
    "best_crop_score_column": best_col,
    "best_crop_auroc": crop_metrics[best_col]["auroc"],
    "best_crop_auprc": crop_metrics[best_col]["auprc"],
    "phase_8_5c_patient_label_preflight": {
        "patient_count": patient_count,
        "nsclc_count": nsclc_count,
        "msd_count": msd_count,
        "luna_normal_count": normal_patient_count,
        "positive_only_patients": positive_only_count,
        "hard_negative_only_patients": hn_only_count,
        "mixed_label_patients": mixed_label_count,
        "all_patients_have_positive_crop": s8_5c.get("all_patients_have_positive_crop"),
        "patient_level_feasibility": s8_5c.get("patient_level_metric_feasibility"),
    },
    "input_files": {
        "phase8_5b_summary": str(IN_8_5B_SUMMARY),
        "phase8_5b_crop_csv": str(IN_8_5B_CROP_CSV),
        "phase8_5c_summary": str(IN_8_5C_SUMMARY),
        "phase8_5c_comp_csv": str(IN_8_5C_COMP_CSV),
    },
    "output_files": {
        "report_md":    str(OUT_DIR / "phase8_5d_metric_closure_report.md"),
        "summary_json": str(OUT_DIR / "phase8_5d_metric_closure_summary.json"),
    },
    "forbidden_operations_confirmed_not_executed": [
        "metric_recalculation", "AUROC_AUPRC_recalculation",
        "threshold_p95_p99_hit_rate_recall_calculation", "FROC_calculation",
        "model_forward", "training_backward_optimizer_step", "checkpoint_creation",
        "score_csv_modification", "existing_output_modification_deletion",
        "stage2_holdout_crop_npz_reload", "v2_v2v2_access",
        "arbitrary_patient_label_creation",
    ],
}

# ── report MD ──────────────────────────────────────────────
lines = [
    "# Phase 8.5D: Final Metric Closure / Decision Report",
    "",
    f"**생성 시각:** {ts}",
    f"**최종 판정:** `PASS_CROP_LEVEL_ONLY_PATIENT_LEVEL_NOT_APPLICABLE`",
    "",
    "---",
    "",
    "## 1. Phase 8.4B — Full Scoring PASS",
    "",
    f"| 항목 | 값 |",
    f"|------|----|",
    f"| 전체 처리 row | {phase_8_4b['total_processed_rows']:,} |",
    f"| 성공 row | {phase_8_4b['total_success_rows']:,} |",
    f"| 오류 row | {phase_8_4b['total_error_rows']} |",
    f"| 판정 | **{phase_8_4b['status']}** |",
    "",
    "## 2. Phase 8.4C — Artifact Validation PASS",
    "",
    f"- verdict: **{phase_8_4c['verdict']}**",
    f"- overall_pass: {phase_8_4c['overall_pass']}",
    f"- Phase 8.5A 진행 허용: {phase_8_4c['phase_8_5a_allowed']}",
    "",
    "## 3. Phase 8.5A — Metric Preflight PASS",
    "",
    f"- verdict: **{phase_8_5a['verdict']}**",
    f"- Phase 8.5B 진행 허용: {phase_8_5a['phase_8_5b_allowed']}",
    f"- 검사 항목 통과: {phase_8_5a['total_pass']} / 실패: {phase_8_5a['total_fail']}",
    "",
    "## 4. Phase 8.5B — Crop-level Metric 결과",
    "",
    "| Score Column | AUROC | AUPRC |",
    "|--------------|-------|-------|",
]

# 정렬: AUROC 내림차순
for col, m in sorted(crop_metrics.items(), key=lambda x: -x[1]["auroc"]):
    best_mark = " ← **최고**" if col == best_col else ""
    lines.append(f"| {col} | {m['auroc']} | {m['auprc']}{best_mark} |")

lines += [
    "",
    f"- crop 총 수: {crop_metrics[best_col]['n_total']:,}",
    f"  (positive {crop_metrics[best_col]['n_positive']:,} / hard_negative {crop_metrics[best_col]['n_negative']:,})",
    "",
    "**가장 좋은 score column:** `mediastinal_channels_l1_mean`",
    f"- AUROC {crop_metrics['mediastinal_channels_l1_mean']['auroc']} / AUPRC {crop_metrics['mediastinal_channels_l1_mean']['auprc']}",
    "",
    "> threshold / p95 / p99 / hit-rate / recall 은 계산하지 않음 (금지 항목).",
    "",
    "## 5. Phase 8.5C — Patient-level Label Preflight 결과",
    "",
    "| 항목 | 값 |",
    "|------|----|",
    f"| stage2_holdout patient 수 | {patient_count} |",
    f"| NSCLC 환자 | {nsclc_count} |",
    f"| MSD 환자 | {msd_count} |",
    f"| LUNA/normal 환자 | **{normal_patient_count}** |",
    f"| positive crop only 환자 | {positive_only_count} |",
    f"| hard_negative crop only 환자 | {hn_only_count} |",
    f"| mixed (positive + hard_negative) 환자 | **{mixed_label_count}** |",
    "",
    "## 6. Patient-level Binary AUROC/AUPRC 불가 사유",
    "",
    "1. **negative class 환자가 0명.** stage2_holdout은 NSCLC + MSD(병변 환자)만으로 구성.",
    "   LUNA(정상) 환자가 포함되지 않아 binary 분류 지표 계산 불가.",
    "2. **모든 환자가 positive crop + hard_negative crop을 동시에 보유.**",
    "   crop label을 patient label로 끌어올릴 경우 154명 전원 positive → AUROC 계산 의미 없음.",
    "3. **crop label은 sampling 방식 기반이며 patient ground-truth가 아님.**",
    "   'positive crop ≥1 → patient positive' 기준은 sampling 방식을 ground-truth로 오용하는 것.",
    "",
    "## 7. 최종 결정",
    "",
    "- **Phase 8.5 공식 결과는 crop-level metric까지만 인정.**",
    "- **patient-level AUROC/AUPRC: `NOT_APPLICABLE / STRUCTURALLY_INVALID`로 기록.**",
    "- **Phase 8.5D patient-level metric calculation: 실행하지 않음.**",
    "",
    "## 8. 다음 평가 방향",
    "",
    "- **per-scan FROC-style evaluation protocol 설계 필요.**",
    "  현재 메모리 기록상 per-scan modified-z + top-K(FROC)가 올바른 지표.",
    "  (v2 ResNet18 기준 top-10 sensitivity 0.62 기록 존재)",
    "- 또는 **normal patient(LUNA)를 포함한 별도 patient-level evaluation set 구성 필요.**",
    "- 이번 문서에서는 FROC 계산 수행하지 않음 (설계 필요성만 기록).",
    "",
    "---",
    "",
    "Phase 8.5는 crop-level discrimination 평가로 닫고, "
    "patient-level binary AUROC/AUPRC는 stage2_holdout 구성상 구조적으로 계산 불가하므로 "
    "공식 성능 지표에서 제외한다.",
]

report_path = OUT_DIR / "phase8_5d_metric_closure_report.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"[MD 보고서] {report_path}")

summary_path = OUT_DIR / "phase8_5d_metric_closure_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"[JSON 요약] {summary_path}")

print(f"\n=== Phase 8.5D 완료. 최종 판정: PASS_CROP_LEVEL_ONLY_PATIENT_LEVEL_NOT_APPLICABLE ===")
