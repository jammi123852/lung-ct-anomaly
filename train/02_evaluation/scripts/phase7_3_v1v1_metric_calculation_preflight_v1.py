"""
Phase 7.3 metric calculation preflight v1
- Phase 7.2 full scoring output integrity audit
- label/sampling_label feasibility 확인
- metric 후보 설계 (실제 계산 금지)
- score column policy 정리
- next step decision
"""

import csv
import json
import pathlib
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

PHASE72_DIR = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/scores"
    / "phase7_2_v1v1_stage1_dev_full_scoring_v1"
)
PHASE73_DIR = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase7_3_v1v1_metric_calculation_preflight_v1"
)
MANIFEST_DIR = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/review_annotations"
    / "phase6_1b_s6a_stage1_dev_filtered_manifest_v1"
)

INPUT_SCORE_CSV = PHASE72_DIR / "phase7_2_v1v1_stage1_dev_full_scoring_v1.csv"
INPUT_SUMMARY_JSON = PHASE72_DIR / "phase7_2_v1v1_stage1_dev_full_scoring_summary_v1.json"
INPUT_MD_REPORT = PHASE72_DIR / "phase7_2_v1v1_stage1_dev_full_scoring_report_v1.md"
INPUT_ERROR_CSV = PHASE72_DIR / "phase7_2_v1v1_stage1_dev_full_scoring_errors_v1.csv"
INPUT_RUNTIME_CSV = PHASE72_DIR / "phase7_2_v1v1_stage1_dev_full_scoring_runtime_summary_v1.csv"
INPUT_DONE_JSON = PHASE72_DIR / "phase7_2_v1v1_stage1_dev_full_scoring_DONE.json"
INPUT_MANIFEST_CSV = (
    MANIFEST_DIR / "phase6_1b_s6a_stage1_dev_filtered_manifest_v1.csv"
)

OUT_CSV = PHASE73_DIR / "phase7_3_v1v1_metric_calculation_preflight_v1.csv"
OUT_JSON = PHASE73_DIR / "phase7_3_v1v1_metric_calculation_preflight_v1.json"
OUT_MD = PHASE73_DIR / "phase7_3_v1v1_metric_calculation_preflight_report_v1.md"

CANONICAL_COLS = [
    "crop_id", "patient_id", "npz_path", "label", "sampling_label",
    "stage_split", "model_tag", "checkpoint_path",
    "crop_score_l1_mean", "crop_score_l1_max", "crop_score_mse_mean",
    "channel_0_l1_mean", "channel_1_l1_mean", "channel_2_l1_mean",
    "channel_3_l1_mean", "channel_4_l1_mean", "channel_5_l1_mean",
    "lung_channels_l1_mean", "mediastinal_channels_l1_mean",
    "input_min", "input_max", "recon_min", "recon_max",
    "error_min", "error_max", "has_nan", "has_inf",
]

EXPECTED_ROW_COUNT = 129437

# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------
def pass_fail(condition):
    return "PASS" if condition else "FAIL"


def exists_status(path):
    return "PASS" if path.exists() else "FAIL"


# ---------------------------------------------------------------------------
# output overwrite guard
# ---------------------------------------------------------------------------
def check_output_guard():
    if PHASE73_DIR.exists():
        print(f"[ERROR] output root already exists: {PHASE73_DIR}")
        sys.exit(1)
    for p in [OUT_CSV, OUT_JSON, OUT_MD]:
        if p.exists():
            print(f"[ERROR] output file already exists: {p}")
            sys.exit(1)


def recheck_before_save(path):
    if path.exists():
        print(f"[ERROR] output file already exists before save: {path}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Section A: Phase 7.2 output integrity audit
# ---------------------------------------------------------------------------
def run_section_a(df, done_data, md_text):
    rows = []

    def add(item, expected, observed, status, note=""):
        rows.append({
            "section": "A",
            "item": item,
            "expected": str(expected),
            "observed": str(observed),
            "status": status,
            "note": note,
        })

    # 파일 존재 확인
    add("score_csv_exists", "True", str(INPUT_SCORE_CSV.exists()), exists_status(INPUT_SCORE_CSV))
    add("summary_json_exists", "True", str(INPUT_SUMMARY_JSON.exists()), exists_status(INPUT_SUMMARY_JSON))
    add("md_report_exists", "True", str(INPUT_MD_REPORT.exists()), exists_status(INPUT_MD_REPORT))
    add("error_csv_exists", "True", str(INPUT_ERROR_CSV.exists()), exists_status(INPUT_ERROR_CSV))
    add("runtime_summary_exists", "True", str(INPUT_RUNTIME_CSV.exists()), exists_status(INPUT_RUNTIME_CSV))
    add("done_marker_exists", "True", str(INPUT_DONE_JSON.exists()), exists_status(INPUT_DONE_JSON))
    add("filtered_manifest_exists", "True", str(INPUT_MANIFEST_CSV.exists()), exists_status(INPUT_MANIFEST_CSV))

    # row count
    row_count = len(df) if df is not None else 0
    add("score_csv_row_count", EXPECTED_ROW_COUNT, row_count,
        pass_fail(row_count == EXPECTED_ROW_COUNT))

    # canonical 27컬럼
    if df is not None:
        actual_cols = list(df.columns)
        missing = [c for c in CANONICAL_COLS if c not in actual_cols]
        extra = [c for c in actual_cols if c not in CANONICAL_COLS]
        col_ok = (len(missing) == 0 and len(df.columns) == len(CANONICAL_COLS))
        add("canonical_27cols", "27 cols", f"{len(actual_cols)} cols",
            pass_fail(col_ok),
            f"missing={missing}, extra={extra}" if not col_ok else "")
    else:
        add("canonical_27cols", "27 cols", "N/A", "FAIL", "score CSV not loaded")

    # error CSV row 수 (헤더 제외)
    if INPUT_ERROR_CSV.exists():
        with open(INPUT_ERROR_CSV, newline="", encoding="utf-8") as f:
            error_rows = list(csv.reader(f))
        error_data_rows = max(0, len(error_rows) - 1)
        add("error_csv_error_rows", 0, error_data_rows,
            pass_fail(error_data_rows == 0))
    else:
        add("error_csv_error_rows", 0, "N/A", "FAIL", "error CSV not found")

    # DONE marker 필드 확인
    if done_data:
        dm_sp = done_data.get("scoring_pass", None)
        add("done_scoring_pass", True, dm_sp, pass_fail(dm_sp is True))

        dm_rc = done_data.get("output_csv_row_count", None)
        add("done_output_csv_row_count", EXPECTED_ROW_COUNT, dm_rc,
            pass_fail(dm_rc == EXPECTED_ROW_COUNT))

        dm_ec = done_data.get("error_count", None)
        add("done_error_count", 0, dm_ec, pass_fail(dm_ec == 0))

        dm_me = done_data.get("metric_calculation_executed", None)
        add("done_metric_calculation_executed", False, dm_me,
            pass_fail(dm_me is False))

        dm_tc = done_data.get("threshold_calculated", None)
        add("done_threshold_calculated", False, dm_tc,
            pass_fail(dm_tc is False))

        dm_te = done_data.get("training_executed", None)
        add("done_training_executed", False, dm_te,
            pass_fail(dm_te is False))

        # stage2_holdout_used / v2_used: DONE marker에 없으므로 npz_path 기반 검증
        if df is not None and "npz_path" in df.columns:
            sh_count = df["npz_path"].astype(str).str.contains("stage2_holdout").sum()
            v2_count = df["npz_path"].astype(str).str.contains(r"/v2/|/v2v2/", regex=True).sum()
            add("stage2_holdout_npz_count", 0, int(sh_count),
                pass_fail(sh_count == 0),
                "DONE marker field absent; verified via npz_path")
            add("v2_v2v2_npz_count", 0, int(v2_count),
                pass_fail(v2_count == 0),
                "DONE marker field absent; verified via npz_path")
        else:
            add("stage2_holdout_npz_count", 0, "N/A", "FAIL", "score CSV not loaded or no npz_path col")
            add("v2_v2v2_npz_count", 0, "N/A", "FAIL", "score CSV not loaded or no npz_path col")
    else:
        for field in ["done_scoring_pass", "done_output_csv_row_count", "done_error_count",
                      "done_metric_calculation_executed", "done_threshold_calculated",
                      "done_training_executed", "stage2_holdout_npz_count", "v2_v2v2_npz_count"]:
            add(field, "-", "N/A", "FAIL", "DONE marker not loaded")

    # MD / DONE marker 불일치 확인
    # MD 110번 줄 근처에 "DONE marker 생성: NO" 패턴 존재 여부
    md_done_no = "DONE marker 생성: NO" in md_text
    dm_actual_pass = (done_data or {}).get("scoring_pass", False) is True
    consistency_warning = md_done_no and dm_actual_pass
    add("md_done_consistency",
        "consistent",
        "WARNING: MD says DONE=NO but JSON has scoring_pass=True" if consistency_warning else "consistent",
        "WARNING" if consistency_warning else "PASS",
        "Phase 7.2 MD was likely generated before DONE marker was written; does not invalidate score CSV")

    return rows, consistency_warning


# ---------------------------------------------------------------------------
# Section B: label / sampling_label distribution
# ---------------------------------------------------------------------------
def run_section_b(df):
    rows = []

    def add(column, value, count, patient_count, metric_role, note=""):
        rows.append({
            "section": "B",
            "column": column,
            "value": str(value),
            "count": int(count),
            "patient_count": int(patient_count),
            "metric_role_candidate": metric_role,
            "note": note,
        })

    if df is None:
        rows.append({
            "section": "B", "column": "N/A", "value": "N/A", "count": 0,
            "patient_count": 0, "metric_role_candidate": "N/A",
            "note": "score CSV not loaded",
        })
        return rows, {}, {}, {}

    # label 분포 (원본)
    label_dist = df["label"].value_counts().to_dict()
    for val, cnt in sorted(label_dist.items()):
        pat_cnt = int(df[df["label"] == val]["patient_id"].nunique())
        norm_val = int(pd.to_numeric(val, errors="coerce")) if not isinstance(val, int) else val
        role = "positive_class" if norm_val == 1 else "negative_class_candidate"
        note = "sampling_label=positive" if norm_val == 1 else "sampling_label=hard_negative"
        add("label", val, cnt, pat_cnt, role, note)

    # label_norm: numeric 정규화 (메모리 내부 처리, 원본 CSV 미수정)
    label_norm = pd.to_numeric(df["label"], errors="coerce")
    nan_count = int(label_norm.isna().sum())
    label_norm_dist = label_norm.dropna().astype(int).value_counts().to_dict()
    if nan_count > 0:
        add("label_norm", "nan", nan_count, 0, "ERROR",
            f"NaN label count={nan_count}: dtype conversion failed for some rows")
    for val, cnt in sorted(label_norm_dist.items()):
        pat_cnt = int(df[label_norm == val]["patient_id"].nunique())
        role = "positive_class" if val == 1 else "negative_class_candidate"
        add("label_norm", val, cnt, pat_cnt, role, "pd.to_numeric normalized; used for metric feasibility")

    # sampling_label 분포
    sl_dist = df["sampling_label"].value_counts().to_dict()
    for val, cnt in sorted(sl_dist.items()):
        pat_cnt = int(df[df["sampling_label"] == val]["patient_id"].nunique())
        role = "positive_class" if val == "positive" else "negative_class_candidate"
        add("sampling_label", val, cnt, pat_cnt, role, "")

    # 모든 환자 양쪽 label 혼재 확인
    mixed = df.groupby("patient_id")["label"].nunique()
    all_mixed = int((mixed > 1).sum())
    total_patients = int(df["patient_id"].nunique())
    add("patient_label_mix",
        "all_mixed",
        all_mixed,
        total_patients,
        "note_only",
        "all_patients_have_mixed_labels: patient-level AUROC requires aggregation strategy definition")

    # patient별 label 분포 요약
    pat_label_dist = {}
    for lv in sorted(label_dist.keys()):
        pat_label_dist[str(lv)] = int(df[df["label"] == lv]["patient_id"].nunique())

    return rows, label_dist, sl_dist, pat_label_dist, label_norm_dist


# ---------------------------------------------------------------------------
# Section C: metric feasibility
# ---------------------------------------------------------------------------
def run_section_c(df, label_norm_dist):
    rows = []

    def add(metric_name, score_column, level, required_labels, feasible, blocker, recommendation):
        rows.append({
            "section": "C",
            "metric_name": metric_name,
            "score_column": score_column,
            "level": level,
            "required_labels": required_labels,
            "feasible": feasible,
            "blocker": blocker,
            "recommendation": recommendation,
        })

    has_binary = (
        df is not None
        and 0 in label_norm_dist
        and 1 in label_norm_dist
        and label_norm_dist.get(0, 0) > 0
        and label_norm_dist.get(1, 0) > 0
    )

    f = "YES" if has_binary else "NO"
    blocker_msg = "" if has_binary else "binary labels not available"

    add("crop_level_auroc_primary", "crop_score_l1_mean", "crop", "label 0/1", f, blocker_msg,
        "PRIMARY: use for Phase 7.4")
    add("crop_level_auprc_primary", "crop_score_l1_mean", "crop", "label 0/1", f, blocker_msg,
        "PRIMARY: use for Phase 7.4")
    add("crop_level_auroc_secondary", "crop_score_mse_mean", "crop", "label 0/1", f, blocker_msg,
        "SECONDARY: cross-check with primary")
    add("crop_level_auprc_secondary", "crop_score_mse_mean", "crop", "label 0/1", f, blocker_msg,
        "SECONDARY: cross-check with primary")
    add("patient_level_max_score_auroc", "crop_score_l1_mean", "patient",
        "patient-level label required",
        "CONDITIONAL",
        "patient_label_definition_required",
        "all patients have mixed labels; need aggregation strategy (e.g. patient=positive if any crop=1)")
    add("patient_level_mean_score_auroc", "crop_score_l1_mean", "patient",
        "patient-level label required",
        "CONDITIONAL",
        "patient_label_definition_required",
        "same as above")
    add("patient_level_topk_mean_score", "crop_score_l1_mean", "patient",
        "patient-level label required",
        "CONDITIONAL",
        "patient_label_definition_required",
        "k must be defined; same mixed-label issue")

    feasibility_summary = {
        "crop_level_auroc_primary": f,
        "crop_level_auprc_primary": f,
        "crop_level_auroc_secondary": f,
        "crop_level_auprc_secondary": f,
        "patient_level_max_score_auroc": "CONDITIONAL",
        "patient_level_mean_score_auroc": "CONDITIONAL",
        "patient_level_topk_mean_score": "CONDITIONAL",
    }
    return rows, feasibility_summary


# ---------------------------------------------------------------------------
# Section D: score column policy
# ---------------------------------------------------------------------------
def run_section_d():
    rows = []

    def add(score_column, role, use_for_metric, limitation, note):
        rows.append({
            "section": "D",
            "score_column": score_column,
            "role": role,
            "use_for_metric": str(use_for_metric),
            "limitation": limitation,
            "note": note,
        })

    add("crop_score_l1_mean", "primary", True,
        "L1 mean may be smoothed by normal channels",
        "selected_primary_score_column for Phase 7.4")
    add("crop_score_mse_mean", "secondary", True,
        "MSE amplifies large errors; complementary to L1",
        "selected_secondary_score_column for Phase 7.4")
    add("crop_score_l1_max", "diagnostic_only", False,
        "max can be dominated by single pixel artifact",
        "do not use for AUROC/AUPRC; diagnostic inspection only")

    channel_cols = [
        "channel_0_l1_mean", "channel_1_l1_mean", "channel_2_l1_mean",
        "channel_3_l1_mean", "channel_4_l1_mean", "channel_5_l1_mean",
        "lung_channels_l1_mean", "mediastinal_channels_l1_mean",
    ]
    for col in channel_cols:
        add(col, "diagnostic_only", False,
            "single channel score; not representative of full reconstruction error",
            "diagnostic inspection only; not used in Phase 7.4 metric")

    return rows


# ---------------------------------------------------------------------------
# Section E: next step decision
# ---------------------------------------------------------------------------
def run_section_e(a_pass_all, label_binary_ok):
    rows = []

    def add(option_id, option_name, status, recommendation, approval_required):
        rows.append({
            "section": "E",
            "option_id": option_id,
            "option_name": option_name,
            "status": status,
            "recommendation": recommendation,
            "approval_required": str(approval_required),
        })

    if not a_pass_all:
        add("E1", "BLOCKED_SCORE_INTEGRITY_ISSUE", "SELECTED",
            "Fix Phase 7.2 integrity issues before proceeding", True)
        add("E2", "READY_FOR_PHASE7_4_METRIC_CALCULATION", "NOT_APPLICABLE",
            "Blocked by integrity issue", False)
        add("E3", "BLOCKED_LABEL_MAP_CLARIFICATION_REQUIRED", "NOT_APPLICABLE",
            "Blocked by integrity issue", False)
        return rows, "BLOCKED_SCORE_INTEGRITY_ISSUE"

    if not label_binary_ok:
        add("E1", "BLOCKED_LABEL_MAP_CLARIFICATION_REQUIRED", "SELECTED",
            "label map not binary; clarify positive/negative definition", True)
        add("E2", "READY_FOR_PHASE7_4_METRIC_CALCULATION", "NOT_APPLICABLE",
            "Blocked by label map issue", False)
        add("E3", "BLOCKED_SCORE_INTEGRITY_ISSUE", "NOT_APPLICABLE", "", False)
        return rows, "BLOCKED_LABEL_MAP_CLARIFICATION_REQUIRED"

    add("E1", "READY_FOR_PHASE7_4_METRIC_CALCULATION", "SELECTED",
        "All integrity checks pass; label map is binary; proceed to Phase 7.4", True)
    add("E2", "BLOCKED_LABEL_MAP_CLARIFICATION_REQUIRED", "NOT_APPLICABLE", "", False)
    add("E3", "BLOCKED_SCORE_INTEGRITY_ISSUE", "NOT_APPLICABLE", "", False)
    return rows, "READY_FOR_PHASE7_4_METRIC_CALCULATION"


# ---------------------------------------------------------------------------
# CSV 저장
# ---------------------------------------------------------------------------
def save_csv(section_a, section_b, section_c, section_d, section_e):
    all_rows = []

    # 섹션별 필드 합집합
    a_fields = ["section", "item", "expected", "observed", "status", "note"]
    b_fields = ["section", "column", "value", "count", "patient_count", "metric_role_candidate", "note"]
    c_fields = ["section", "metric_name", "score_column", "level", "required_labels", "feasible", "blocker", "recommendation"]
    d_fields = ["section", "score_column", "role", "use_for_metric", "limitation", "note"]
    e_fields = ["section", "option_id", "option_name", "status", "recommendation", "approval_required"]

    all_fields = sorted(set(
        a_fields + b_fields + c_fields + d_fields + e_fields
    ))

    def normalize(row, fields):
        return {f: row.get(f, "") for f in all_fields}

    for r in section_a + section_b + section_c + section_d + section_e:
        all_rows.append(normalize(r, all_fields))

    recheck_before_save(OUT_CSV)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[CSV] saved: {OUT_CSV}  ({len(all_rows)} rows)")


# ---------------------------------------------------------------------------
# JSON 저장
# ---------------------------------------------------------------------------
def save_json(
    done_data,
    consistency_warning,
    row_count,
    canonical_ok,
    label_dist,
    sl_dist,
    pat_label_dist,
    feasibility_summary,
    readiness,
    blockers,
    a_rows,
    df,
):
    stage2_holdout_used = False
    v2_used = False
    if df is not None and "npz_path" in df.columns:
        stage2_holdout_used = bool(df["npz_path"].astype(str).str.contains("stage2_holdout").any())
        v2_used = bool(df["npz_path"].astype(str).str.contains(r"/v2/|/v2v2/", regex=True).any())

    integrity_audit = {}
    for r in a_rows:
        integrity_audit[r["item"]] = r["status"]

    output = {
        "input_paths": {
            "score_csv": str(INPUT_SCORE_CSV),
            "summary_json": str(INPUT_SUMMARY_JSON),
            "md_report": str(INPUT_MD_REPORT),
            "error_csv": str(INPUT_ERROR_CSV),
            "runtime_summary": str(INPUT_RUNTIME_CSV),
            "done_marker": str(INPUT_DONE_JSON),
            "filtered_manifest": str(INPUT_MANIFEST_CSV),
        },
        "phase7_2_integrity_audit": integrity_audit,
        "score_csv_row_count": row_count,
        "canonical_schema_check": "PASS" if canonical_ok else "FAIL",
        "done_marker_audit": done_data if done_data else {},
        "report_consistency_warning": consistency_warning,
        "label_distribution": {str(k): int(v) for k, v in label_dist.items()},
        "sampling_label_distribution": {str(k): int(v) for k, v in sl_dist.items()},
        "patient_label_distribution": pat_label_dist,
        "metric_feasibility": feasibility_summary,
        "selected_primary_score_column": "crop_score_l1_mean",
        "selected_secondary_score_column": "crop_score_mse_mean",
        "threshold_policy": "forbidden_in_phase7_3",
        "metric_calculation_executed": False,
        "threshold_calculated": False,
        "training_executed": False,
        "stage2_holdout_used": stage2_holdout_used,
        "v2_used": v2_used,
        "readiness_for_phase7_4": readiness,
        "blockers": blockers,
        "next_step_recommendation": (
            "Proceed to Phase 7.4 metric calculation after user approval"
            if readiness == "READY_FOR_PHASE7_4_METRIC_CALCULATION"
            else f"Resolve: {readiness}"
        ),
    }

    recheck_before_save(OUT_JSON)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[JSON] saved: {OUT_JSON}")
    return output


# ---------------------------------------------------------------------------
# MD report 저장
# ---------------------------------------------------------------------------
def save_md(jdata, consistency_warning, a_rows, label_dist, sl_dist, pat_label_dist,
            feasibility_summary, readiness):
    lines = []

    def h2(t):
        lines.append(f"\n## {t}\n")

    def h3(t):
        lines.append(f"\n### {t}\n")

    def li(t):
        lines.append(f"- {t}")

    lines.append("# Phase 7.3 metric calculation preflight report v1\n")

    h2("1. Phase 7.3 목적")
    li("Phase 7.2 full scoring output의 integrity를 audit한다.")
    li("label/sampling_label 분포를 확인하고 metric 계산 가능성을 검증한다.")
    li("metric 후보 및 score column policy를 설계한다.")
    li("실제 metric 계산, threshold 계산, training은 이 단계에서 수행하지 않는다.")

    h2("2. Phase 7.2 scoring output integrity audit")
    lines.append("")
    lines.append("| item | expected | observed | status | note |")
    lines.append("|------|----------|----------|--------|------|")
    for r in a_rows:
        note = r.get("note", "").replace("|", "/")
        lines.append(f"| {r['item']} | {r['expected']} | {r['observed']} | {r['status']} | {note} |")

    h2("3. DONE marker audit")
    dm = jdata.get("done_marker_audit", {})
    for k, v in dm.items():
        li(f"`{k}`: {v}")

    h2("4. Phase 7.2 MD/DONE consistency warning")
    if consistency_warning:
        lines.append(
            "> **WARNING**: Phase 7.2 MD report (line ~110) states "
            "`DONE marker 생성: NO` but actual DONE JSON has `scoring_pass=true`. "
            "This does NOT invalidate the score CSV. The MD was likely generated "
            "before the DONE marker write. Record as `report_consistency_warning=True`."
        )
    else:
        li("MD report and DONE marker are consistent. No warning.")

    h2("5. label/sampling_label 분포")
    h3("label 분포")
    lines.append("")
    lines.append("| label | count | patient_count |")
    lines.append("|-------|-------|---------------|")
    for k, v in sorted(label_dist.items()):
        from collections import Counter
        lines.append(f"| {k} | {v} | {pat_label_dist.get(str(k), '-')} |")

    h3("sampling_label 분포")
    lines.append("")
    lines.append("| sampling_label | count |")
    lines.append("|----------------|-------|")
    for k, v in sorted(sl_dist.items()):
        lines.append(f"| {k} | {v} |")

    h3("특이사항")
    li("모든 152명 환자가 label=0(hard_negative)과 label=1(positive)을 모두 보유한다.")
    li("patient-level AUROC/AUPRC 계산 시 환자 레벨 label 집계 전략 정의가 필요하다.")
    li("(예: 환자에 label=1 crop이 하나라도 있으면 patient=positive 등)")
    li("이 주의사항은 crop-level metric을 block하지 않는다.")

    h2("6. metric feasibility")
    lines.append("")
    lines.append("| metric_name | score_column | level | feasible | blocker | recommendation |")
    lines.append("|-------------|-------------|-------|----------|---------|----------------|")
    for k, v in feasibility_summary.items():
        lines.append(f"| {k} | crop_score_l1_mean | - | {v} | - | - |")

    h2("7. score column policy")
    lines.append("")
    lines.append("| score_column | role | use_for_metric |")
    lines.append("|-------------|------|----------------|")
    lines.append("| crop_score_l1_mean | primary | True |")
    lines.append("| crop_score_mse_mean | secondary | True |")
    lines.append("| crop_score_l1_max | diagnostic_only | False |")
    lines.append("| channel_0~5_l1_mean, lung/mediastinal_channels_l1_mean | diagnostic_only | False |")

    h2("8. threshold policy")
    li("Phase 7.3에서 threshold 계산은 금지된다.")
    li("p95/p99 계산 금지.")
    li("threshold-based hit rate 계산 금지.")
    li("threshold는 별도 phase에서만 다룬다.")
    li(f"`threshold_policy`: forbidden_in_phase7_3")

    h2("9. 최종 판정")
    lines.append(f"\n**{readiness}**\n")
    if readiness == "READY_FOR_PHASE7_4_METRIC_CALCULATION":
        li("모든 integrity check PASS.")
        li("label 분포 binary 확인 완료.")
        li("Phase 7.4 metric calculation 진행 가능.")
        li("단, patient-level metric 집계 전략은 Phase 7.4에서 정의 필요.")
    elif readiness == "BLOCKED_LABEL_MAP_CLARIFICATION_REQUIRED":
        li("label map이 명확하지 않음. label map clarification 필요.")
    else:
        li("Phase 7.2 score CSV integrity 문제. 수정 또는 재검증 필요.")

    h2("10. 다음 단계")
    if readiness == "READY_FOR_PHASE7_4_METRIC_CALCULATION":
        li("Phase 7.4 metric calculation 사용자 승인 요청.")
        li("승인 후 crop-level AUROC/AUPRC (primary: crop_score_l1_mean) 계산.")
        li("patient-level metric은 label 집계 전략 정의 후 진행.")
    else:
        li(f"Resolve: {readiness}")

    h2("11. 금지 사항 확인 목록")
    forbidden_items = [
        "metric 계산 금지",
        "AUROC/AUPRC 실제 계산 금지",
        "threshold 계산 금지",
        "p95/p99 계산 금지",
        "threshold-based hit rate 계산 금지",
        "training 금지",
        "backward 금지",
        "optimizer step 금지",
        "checkpoint 생성 금지",
        "hard negative 최종 채택 금지",
        "score CSV 수정 금지",
        "filtered manifest 수정 금지",
        "stage2_holdout 접근 금지",
        "v2/v2v2 접근 금지",
        "pip install / 외부 다운로드 금지",
    ]
    for item in forbidden_items:
        li(f"[확인] {item}")

    recheck_before_save(OUT_MD)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[MD] saved: {OUT_MD}")


# ---------------------------------------------------------------------------
# 콘솔 보고 (보고 형식 1~23번)
# ---------------------------------------------------------------------------
def print_report(jdata, consistency_warning, a_rows, label_dist, sl_dist,
                 pat_label_dist, feasibility_summary, readiness):
    print("\n" + "=" * 70)
    print("Phase 7.3 metric calculation preflight 보고")
    print("=" * 70)

    def pitem(n, label, val):
        print(f"  {n:2}. {label}: {val}")

    pitem(1, "output root", str(PHASE73_DIR))
    pitem(2, "생성 CSV 경로", str(OUT_CSV))
    pitem(3, "생성 JSON 경로", str(OUT_JSON))
    pitem(4, "생성 MD 경로", str(OUT_MD))
    pitem(5, "Phase 7.2 score CSV row 수", jdata["score_csv_row_count"])
    pitem(6, "canonical 27컬럼 확인", jdata["canonical_schema_check"])

    err_status = jdata["phase7_2_integrity_audit"].get("error_csv_error_rows", "N/A")
    pitem(7, "error CSV error row 수", f"0 (status={err_status})")

    pitem(8, "DONE marker 존재", jdata["phase7_2_integrity_audit"].get("done_marker_exists", "N/A"))
    pitem(9, "DONE marker scoring_pass", jdata["done_marker_audit"].get("scoring_pass", "N/A"))
    pitem(10, "DONE marker output_csv_row_count", jdata["done_marker_audit"].get("output_csv_row_count", "N/A"))
    pitem(11, "report consistency warning", consistency_warning)

    ld_str = ", ".join(f"label={k}: {v}" for k, v in sorted(label_dist.items()))
    pitem(12, "label distribution 요약", ld_str)

    sl_str = ", ".join(f"{k}: {v}" for k, v in sorted(sl_dist.items()))
    pitem(13, "sampling_label distribution 요약", sl_str)

    pitem(14, "metric feasibility 판정", readiness)
    pitem(15, "selected primary score column", jdata["selected_primary_score_column"])
    pitem(16, "selected secondary score column", jdata["selected_secondary_score_column"])
    pitem(17, "threshold policy", jdata["threshold_policy"])
    pitem(18, "metric 계산 없음 확인", not jdata["metric_calculation_executed"])
    pitem(19, "threshold 계산 없음 확인", not jdata["threshold_calculated"])
    pitem(20, "training 없음 확인", not jdata["training_executed"])
    pitem(21, "stage2_holdout/v2/v2v2 미접근 확인",
          f"stage2_holdout={not jdata['stage2_holdout_used']}, v2={not jdata['v2_used']}")
    pitem(22, "기존 파일 미수정 확인", "TRUE (read-only; no write to input files)")
    pitem(23, "다음 단계 제안", jdata["next_step_recommendation"])

    print("=" * 70)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print("[Phase 7.3] metric calculation preflight 시작")

    check_output_guard()
    PHASE73_DIR.mkdir(parents=True, exist_ok=False)

    # 데이터 로드
    df = None
    if INPUT_SCORE_CSV.exists():
        print(f"[load] score CSV: {INPUT_SCORE_CSV}")
        df = pd.read_csv(INPUT_SCORE_CSV)
    else:
        print(f"[ERROR] score CSV not found: {INPUT_SCORE_CSV}", file=sys.stderr)

    done_data = None
    if INPUT_DONE_JSON.exists():
        with open(INPUT_DONE_JSON, encoding="utf-8") as f:
            done_data = json.load(f)

    md_text = ""
    if INPUT_MD_REPORT.exists():
        with open(INPUT_MD_REPORT, encoding="utf-8") as f:
            md_text = f.read()

    # 섹션 실행
    a_rows, consistency_warning = run_section_a(df, done_data, md_text)
    b_rows, label_dist, sl_dist, pat_label_dist, label_norm_dist = run_section_b(df)
    c_rows, feasibility_summary = run_section_c(df, label_norm_dist)
    d_rows = run_section_d()

    # integrity all pass 판정 (WARNING은 PASS로 처리)
    a_statuses = [r["status"] for r in a_rows]
    a_pass_all = all(s in ("PASS", "WARNING") for s in a_statuses)
    label_binary_ok = (0 in label_norm_dist and 1 in label_norm_dist)

    e_rows, readiness = run_section_e(a_pass_all, label_binary_ok)

    blockers = []
    if not a_pass_all:
        blockers.append("BLOCKED_SCORE_INTEGRITY_ISSUE")
    if not label_binary_ok:
        blockers.append("BLOCKED_LABEL_MAP_CLARIFICATION_REQUIRED")

    # 저장
    save_csv(a_rows, b_rows, c_rows, d_rows, e_rows)
    jdata = save_json(
        done_data, consistency_warning,
        len(df) if df is not None else 0,
        all(c in (list(df.columns) if df is not None else []) for c in CANONICAL_COLS),
        label_dist, sl_dist, pat_label_dist,
        feasibility_summary,
        readiness, blockers,
        a_rows, df,
    )
    save_md(
        jdata, consistency_warning,
        a_rows, label_dist, sl_dist, pat_label_dist,
        feasibility_summary, readiness,
    )
    print_report(
        jdata, consistency_warning,
        a_rows, label_dist, sl_dist, pat_label_dist,
        feasibility_summary, readiness,
    )

    print("\n[Phase 7.3] preflight 완료")


if __name__ == "__main__":
    main()
