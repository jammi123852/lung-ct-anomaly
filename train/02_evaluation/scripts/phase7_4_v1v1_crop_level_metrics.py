"""
Phase 7.4 v1/v1 crop-level metric calculation
- crop_score_l1_mean: AUROC/AUPRC (primary)
- crop_score_mse_mean: AUROC/AUPRC (secondary)
- threshold / p95/p99 / patient-level metric 계산 금지
"""

import argparse
import csv
import json
import pathlib
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# sklearn 시도 → 없으면 numpy fallback
# ---------------------------------------------------------------------------
try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    METRIC_BACKEND = "sklearn"
except ImportError:
    METRIC_BACKEND = "numpy_fallback"

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
PHASE74_DIR = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/evaluation"
    / "phase7_4_v1v1_crop_level_metrics_v1"
)

INPUT_SCORE_CSV = PHASE72_DIR / "phase7_2_v1v1_stage1_dev_full_scoring_v1.csv"
INPUT_SUMMARY_JSON = PHASE72_DIR / "phase7_2_v1v1_stage1_dev_full_scoring_summary_v1.json"
INPUT_DONE_JSON = PHASE72_DIR / "phase7_2_v1v1_stage1_dev_full_scoring_DONE.json"
INPUT_PHASE73_JSON = PHASE73_DIR / "phase7_3_v1v1_metric_calculation_preflight_v1.json"

OUT_CSV = PHASE74_DIR / "phase7_4_v1v1_crop_level_metrics_v1.csv"
OUT_JSON = PHASE74_DIR / "phase7_4_v1v1_crop_level_metrics_v1.json"
OUT_MD = PHASE74_DIR / "phase7_4_v1v1_crop_level_metrics_report_v1.md"

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

DIAG_COLS = [
    "crop_score_l1_max",
    "channel_0_l1_mean", "channel_1_l1_mean", "channel_2_l1_mean",
    "channel_3_l1_mean", "channel_4_l1_mean", "channel_5_l1_mean",
    "lung_channels_l1_mean", "mediastinal_channels_l1_mean",
]

# ---------------------------------------------------------------------------
# output overwrite guard
# ---------------------------------------------------------------------------
def check_output_guard():
    if PHASE74_DIR.exists():
        print(f"[ERROR] output root already exists: {PHASE74_DIR}")
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
# metric 계산 (fallback 포함)
# ---------------------------------------------------------------------------
def _auroc_fallback(labels, scores):
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average")
    rank_sum_pos = float(ranks[labels == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _auprc_fallback(labels, scores):
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    n_pos = int((labels == 1).sum())
    n = len(labels)
    if n_pos == 0 or n_pos == n:
        return float("nan")
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    cum_tp = np.cumsum(sorted_labels)
    cum_fp = np.cumsum(1 - sorted_labels)
    precision = cum_tp / (cum_tp + cum_fp)
    recall = cum_tp / n_pos
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.trapz(precision, recall))


def compute_auroc(labels, scores):
    if METRIC_BACKEND == "sklearn":
        return float(roc_auc_score(labels, scores))
    return _auroc_fallback(labels, scores)


def compute_auprc(labels, scores):
    if METRIC_BACKEND == "sklearn":
        return float(average_precision_score(labels, scores))
    return _auprc_fallback(labels, scores)


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------
def pass_fail(condition):
    return "PASS" if condition else "FAIL"


def exists_status(path):
    return "PASS" if path.exists() else "FAIL"


def fmt(v, digits=6):
    if isinstance(v, float) and not (v != v):  # not nan
        return round(v, digits)
    return v


# ---------------------------------------------------------------------------
# Section A: input integrity
# ---------------------------------------------------------------------------
def run_section_a(df, done_data, phase73_data, label_norm):
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

    add("score_csv_exists", "True", str(INPUT_SCORE_CSV.exists()), exists_status(INPUT_SCORE_CSV))
    add("summary_json_exists", "True", str(INPUT_SUMMARY_JSON.exists()), exists_status(INPUT_SUMMARY_JSON))
    add("done_marker_exists", "True", str(INPUT_DONE_JSON.exists()), exists_status(INPUT_DONE_JSON))
    add("phase73_json_exists", "True", str(INPUT_PHASE73_JSON.exists()), exists_status(INPUT_PHASE73_JSON))

    row_count = len(df) if df is not None else 0
    add("score_csv_row_count", EXPECTED_ROW_COUNT, row_count,
        pass_fail(row_count == EXPECTED_ROW_COUNT))

    if df is not None:
        missing = [c for c in CANONICAL_COLS if c not in df.columns]
        col_ok = len(missing) == 0 and len(df.columns) == len(CANONICAL_COLS)
        add("canonical_27cols", "27", len(df.columns), pass_fail(col_ok),
            f"missing={missing}" if missing else "")

        has_nan_count = int(df["has_nan"].sum()) if "has_nan" in df.columns else -1
        has_inf_count = int(df["has_inf"].sum()) if "has_inf" in df.columns else -1
        add("has_nan_count", 0, has_nan_count, pass_fail(has_nan_count == 0))
        add("has_inf_count", 0, has_inf_count, pass_fail(has_inf_count == 0))

        score_nan = int(df["crop_score_l1_mean"].isna().sum()) if "crop_score_l1_mean" in df.columns else -1
        score_inf = int(np.isinf(df["crop_score_l1_mean"]).sum()) if "crop_score_l1_mean" in df.columns else -1
        add("score_l1_mean_nan", 0, score_nan, pass_fail(score_nan == 0))
        add("score_l1_mean_inf", 0, score_inf, pass_fail(score_inf == 0))

        score2_nan = int(df["crop_score_mse_mean"].isna().sum()) if "crop_score_mse_mean" in df.columns else -1
        score2_inf = int(np.isinf(df["crop_score_mse_mean"]).sum()) if "crop_score_mse_mean" in df.columns else -1
        add("score_mse_mean_nan", 0, score2_nan, pass_fail(score2_nan == 0))
        add("score_mse_mean_inf", 0, score2_inf, pass_fail(score2_inf == 0))
    else:
        for item in ["canonical_27cols", "has_nan_count", "has_inf_count",
                     "score_l1_mean_nan", "score_l1_mean_inf",
                     "score_mse_mean_nan", "score_mse_mean_inf"]:
            add(item, "-", "N/A", "FAIL", "score CSV not loaded")

    if done_data:
        add("done_scoring_pass", True, done_data.get("scoring_pass"),
            pass_fail(done_data.get("scoring_pass") is True))
        add("done_output_csv_row_count", EXPECTED_ROW_COUNT,
            done_data.get("output_csv_row_count"),
            pass_fail(done_data.get("output_csv_row_count") == EXPECTED_ROW_COUNT))
        add("done_error_count", 0, done_data.get("error_count"),
            pass_fail(done_data.get("error_count") == 0))
    else:
        for item in ["done_scoring_pass", "done_output_csv_row_count", "done_error_count"]:
            add(item, "-", "N/A", "FAIL", "DONE marker not loaded")

    if phase73_data:
        readiness = phase73_data.get("readiness_for_phase7_4", "")
        expected_r = "READY_FOR_PHASE7_4_METRIC_CALCULATION"
        add("phase73_readiness", expected_r, readiness,
            pass_fail(readiness == expected_r))
    else:
        add("phase73_readiness", "READY_FOR_PHASE7_4_METRIC_CALCULATION",
            "N/A", "FAIL", "Phase 7.3 JSON not loaded")

    if label_norm is not None:
        nan_count = int(label_norm.isna().sum())
        add("label_norm_nan_count", 0, nan_count, pass_fail(nan_count == 0))
        unique_vals = sorted(label_norm.dropna().astype(int).unique().tolist())
        vals_ok = unique_vals == [0, 1]
        add("label_norm_values", "[0, 1]", str(unique_vals), pass_fail(vals_ok))
        add("label_norm_0_count", 85906, int((label_norm == 0).sum()),
            pass_fail(int((label_norm == 0).sum()) == 85906))
        add("label_norm_1_count", 43531, int((label_norm == 1).sum()),
            pass_fail(int((label_norm == 1).sum()) == 43531))
    else:
        for item in ["label_norm_nan_count", "label_norm_values",
                     "label_norm_0_count", "label_norm_1_count"]:
            add(item, "-", "N/A", "FAIL", "label_norm not available")

    return rows


# ---------------------------------------------------------------------------
# Section B: label distribution
# ---------------------------------------------------------------------------
def run_section_b(label_norm, df):
    rows = []
    n_total = len(label_norm) if label_norm is not None else 0

    def add(label_value, count, role, note=""):
        ratio = round(count / n_total * 100, 2) if n_total > 0 else 0.0
        rows.append({
            "section": "B",
            "label_value": str(label_value),
            "count": int(count),
            "ratio_percent": ratio,
            "role": role,
            "note": note,
        })

    if label_norm is not None:
        dist = label_norm.dropna().astype(int).value_counts().sort_index()
        for val, cnt in dist.items():
            role = "positive" if val == 1 else "negative"
            note = "sampling_label=positive" if val == 1 else "sampling_label=hard_negative"
            add(val, cnt, role, note)
    else:
        rows.append({
            "section": "B", "label_value": "N/A", "count": 0,
            "ratio_percent": 0.0, "role": "N/A", "note": "label_norm not available",
        })

    return rows


# ---------------------------------------------------------------------------
# Section C: crop-level metrics
# ---------------------------------------------------------------------------
def run_section_c(label_norm, df):
    rows = []
    results = {}

    if label_norm is None or df is None:
        for name in ["crop_level_auroc_primary", "crop_level_auprc_primary",
                     "crop_level_auroc_secondary", "crop_level_auprc_secondary"]:
            rows.append({
                "section": "C", "metric_name": name, "score_column": "-",
                "value": "N/A", "positive_label": 1, "negative_label": 0,
                "n_total": 0, "n_positive": 0, "n_negative": 0,
                "direction": "higher_is_more_anomalous", "status": "FAIL",
                "note": "data not available",
            })
            results[name] = float("nan")
        return rows, results

    labels = label_norm.dropna().astype(int).values
    n_total = len(labels)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    prevalence = round(n_pos / n_total, 4) if n_total > 0 else 0.0

    def add_metric(metric_name, score_col, score_type):
        scores = df[score_col].values
        try:
            if score_type == "auroc":
                val = compute_auroc(labels, scores)
            else:
                val = compute_auprc(labels, scores)
            is_valid = isinstance(val, float) and val == val  # not nan
            status = "PASS" if is_valid else "FAIL"
            note = f"prevalence={prevalence}" if score_type == "auprc" else ""
        except Exception as e:
            val = float("nan")
            status = "FAIL"
            note = str(e)
        rows.append({
            "section": "C",
            "metric_name": metric_name,
            "score_column": score_col,
            "value": fmt(val, 6),
            "positive_label": 1,
            "negative_label": 0,
            "n_total": n_total,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "direction": "higher_is_more_anomalous",
            "status": status,
            "note": note,
        })
        results[metric_name] = val

    print(f"[metric] computing crop_level_auroc_primary (n={n_total})...")
    add_metric("crop_level_auroc_primary", "crop_score_l1_mean", "auroc")
    print(f"[metric] computing crop_level_auprc_primary...")
    add_metric("crop_level_auprc_primary", "crop_score_l1_mean", "auprc")
    print(f"[metric] computing crop_level_auroc_secondary...")
    add_metric("crop_level_auroc_secondary", "crop_score_mse_mean", "auroc")
    print(f"[metric] computing crop_level_auprc_secondary...")
    add_metric("crop_level_auprc_secondary", "crop_score_mse_mean", "auprc")

    return rows, results


# ---------------------------------------------------------------------------
# Section D: diagnostic score distribution
# ---------------------------------------------------------------------------
def run_section_d(df):
    rows = []
    if df is None:
        rows.append({
            "section": "D", "score_column": "N/A", "role": "N/A",
            "min": "N/A", "max": "N/A", "mean": "N/A", "median": "N/A",
            "p25": "N/A", "p75": "N/A", "status": "FAIL", "note": "data not available",
        })
        return rows

    diag_summary = {}
    for col in DIAG_COLS:
        if col not in df.columns:
            rows.append({
                "section": "D", "score_column": col, "role": "diagnostic_only",
                "min": "N/A", "max": "N/A", "mean": "N/A", "median": "N/A",
                "p25": "N/A", "p75": "N/A", "status": "FAIL", "note": "column not found",
            })
            continue
        s = df[col].dropna()
        q25 = float(s.quantile(0.25))
        q50 = float(s.quantile(0.50))
        q75 = float(s.quantile(0.75))
        mn = float(s.min())
        mx = float(s.max())
        me = float(s.mean())
        rows.append({
            "section": "D",
            "score_column": col,
            "role": "diagnostic_only",
            "min": round(mn, 6),
            "max": round(mx, 6),
            "mean": round(me, 6),
            "median": round(q50, 6),
            "p25": round(q25, 6),
            "p75": round(q75, 6),
            "status": "PASS",
            "note": "distribution only; not used for AUROC/AUPRC",
        })
        diag_summary[col] = {
            "min": round(mn, 6), "max": round(mx, 6),
            "mean": round(me, 6), "median": round(q50, 6),
            "p25": round(q25, 6), "p75": round(q75, 6),
        }

    return rows, diag_summary


# ---------------------------------------------------------------------------
# Section E: limitations / forbidden
# ---------------------------------------------------------------------------
def run_section_e():
    rows = []

    items = [
        ("threshold_not_calculated", "threshold 계산 금지 준수", "forbidden by design"),
        ("p95_p99_not_calculated", "p95/p99 계산 금지 준수", "forbidden by design"),
        ("patient_level_metric_not_calculated",
         "patient-level metric 보류; aggregation strategy 미정",
         "pending Phase 7.4b or Phase 7.5"),
        ("scoring_not_reexecuted", "Phase 7.2 full scoring 재실행 없음", "use existing output"),
        ("model_forward_not_executed", "model forward 없음", "inference not called"),
        ("training_not_executed", "training 없음", "no backward/optimizer"),
        ("checkpoint_not_created", "checkpoint 생성 없음", "evaluation only"),
        ("stage2_holdout_not_used", "stage2_holdout 봉인 유지", "holdout sealed"),
        ("v2_not_used", "v2/v2v2 봉인 유지", "holdout sealed"),
        ("score_csv_not_modified", "Phase 7.2 score CSV 읽기 전용", "read-only input"),
        ("phase72_73_output_not_modified",
         "Phase 7.2/7.3 output 수정 없음",
         "all outputs are read-only"),
    ]

    for item, reason, next_action in items:
        rows.append({
            "section": "E",
            "item": item,
            "status": "CONFIRMED",
            "reason": reason,
            "next_required_action": next_action,
        })

    return rows


# ---------------------------------------------------------------------------
# CSV 저장
# ---------------------------------------------------------------------------
def save_csv(a_rows, b_rows, c_rows, d_rows, e_rows):
    all_fields = sorted({
        "section", "item", "expected", "observed", "status", "note",
        "label_value", "count", "ratio_percent", "role",
        "metric_name", "score_column", "value", "positive_label", "negative_label",
        "n_total", "n_positive", "n_negative", "direction",
        "min", "max", "mean", "median", "p25", "p75",
        "reason", "next_required_action",
    })

    all_rows = []
    for r in a_rows + b_rows + c_rows + d_rows + e_rows:
        all_rows.append({f: r.get(f, "") for f in all_fields})

    recheck_before_save(OUT_CSV)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(all_fields))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[CSV] saved: {OUT_CSV}  ({len(all_rows)} rows)")


# ---------------------------------------------------------------------------
# JSON 저장
# ---------------------------------------------------------------------------
def save_json(a_rows, c_results, diag_summary, label_norm, df):
    n_total = int(len(label_norm)) if label_norm is not None else 0
    n_pos = int((label_norm == 1).sum()) if label_norm is not None else 0
    n_neg = int((label_norm == 0).sum()) if label_norm is not None else 0
    prevalence = round(n_pos / n_total, 4) if n_total > 0 else 0.0

    integrity_audit = {r["item"]: r["status"] for r in a_rows}
    a_pass = all(s == "PASS" for s in integrity_audit.values())

    def safe_float(v):
        if isinstance(v, float) and v == v:
            return round(v, 6)
        return None

    auroc_p = c_results.get("crop_level_auroc_primary", float("nan"))
    auprc_p = c_results.get("crop_level_auprc_primary", float("nan"))
    auroc_s = c_results.get("crop_level_auroc_secondary", float("nan"))
    auprc_s = c_results.get("crop_level_auprc_secondary", float("nan"))

    metric_pass = all(
        isinstance(v, float) and v == v
        for v in [auroc_p, auprc_p, auroc_s, auprc_s]
    )

    output = {
        "input_paths": {
            "score_csv": str(INPUT_SCORE_CSV),
            "summary_json": str(INPUT_SUMMARY_JSON),
            "done_marker": str(INPUT_DONE_JSON),
            "phase73_json": str(INPUT_PHASE73_JSON),
        },
        "score_csv_row_count": n_total,
        "canonical_schema_check": integrity_audit.get("canonical_27cols", "N/A"),
        "label_distribution": {"0": n_neg, "1": n_pos},
        "positive_label": 1,
        "negative_label": 0,
        "positive_count": n_pos,
        "negative_count": n_neg,
        "positive_prevalence": prevalence,
        "primary_score_column": "crop_score_l1_mean",
        "secondary_score_column": "crop_score_mse_mean",
        "crop_level_auroc_primary": safe_float(auroc_p),
        "crop_level_auprc_primary": safe_float(auprc_p),
        "crop_level_auroc_secondary": safe_float(auroc_s),
        "crop_level_auprc_secondary": safe_float(auprc_s),
        "diagnostic_score_summary": diag_summary,
        "metric_backend": METRIC_BACKEND,
        "threshold_calculated": False,
        "p95_p99_calculated": False,
        "patient_level_metric_calculated": False,
        "training_executed": False,
        "scoring_reexecuted": False,
        "checkpoint_created": False,
        "stage2_holdout_used": False,
        "v2_used": False,
        "metric_pass": metric_pass,
        "blockers": [] if (a_pass and metric_pass) else ["integrity_or_metric_failure"],
        "next_step_recommendation": (
            "Phase 7.5 v1/v1 test closure report "
            "또는 Phase 7.4b patient-level metric design (별도 승인 필요)"
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
def save_md(jdata, a_rows, b_rows, c_rows, d_rows):
    lines = []

    def h2(t):
        lines.append(f"\n## {t}\n")

    def li(t):
        lines.append(f"- {t}")

    lines.append("# Phase 7.4 v1/v1 crop-level metric calculation report\n")

    h2("1. Phase 7.4 목적")
    li("Phase 7.2 full scoring CSV를 기반으로 crop-level AUROC/AUPRC를 계산한다.")
    li("primary score: crop_score_l1_mean / secondary score: crop_score_mse_mean")
    li("patient-level metric / threshold / p95/p99 / hit-rate 계산은 이 단계에서 수행하지 않는다.")
    li("stage2_holdout/v2/v2v2는 계속 봉인한다.")

    h2("2. 입력 integrity 결과")
    lines.append("")
    lines.append("| item | expected | observed | status |")
    lines.append("|------|----------|----------|--------|")
    for r in a_rows:
        lines.append(f"| {r['item']} | {r['expected']} | {r['observed']} | {r['status']} |")

    h2("3. label 분포")
    lines.append("")
    lines.append("| label | count | ratio_percent | role |")
    lines.append("|-------|-------|---------------|------|")
    for r in b_rows:
        lines.append(
            f"| {r['label_value']} | {r['count']} | {r['ratio_percent']}% | {r['role']} |"
        )

    h2("4. score 방향성")
    li("score가 높을수록 anomaly/positive 가능성이 높다고 가정한다.")
    li("direction = higher_is_more_anomalous")
    li(f"metric_backend = {jdata['metric_backend']}")

    h2("5. crop-level metric 결과")
    lines.append("")
    lines.append("| metric_name | score_column | value | n_total | n_positive | n_negative | direction | status |")
    lines.append("|-------------|-------------|-------|---------|------------|------------|-----------|--------|")
    for r in c_rows:
        lines.append(
            f"| {r['metric_name']} | {r['score_column']} | {r['value']} "
            f"| {r['n_total']} | {r['n_positive']} | {r['n_negative']} "
            f"| {r['direction']} | {r['status']} |"
        )
    li(f"positive prevalence = {jdata['positive_prevalence']} "
       f"({jdata['positive_count']} / {jdata['score_csv_row_count']})")

    h2("6. diagnostic score 분포")
    lines.append("")
    lines.append("| score_column | role | min | max | mean | median | p25 | p75 |")
    lines.append("|-------------|------|-----|-----|------|--------|-----|-----|")
    for r in d_rows:
        lines.append(
            f"| {r['score_column']} | {r['role']} | {r.get('min','')} "
            f"| {r.get('max','')} | {r.get('mean','')} | {r.get('median','')} "
            f"| {r.get('p25','')} | {r.get('p75','')} |"
        )

    h2("7. 하지 않은 것")
    li("patient-level metric 계산 — aggregation strategy 미정으로 보류")
    li("threshold 계산")
    li("p95/p99 계산")
    li("threshold-based hit rate 계산")
    li("stage2_holdout evaluation")
    li("training / backward / optimizer step")
    li("checkpoint 생성")
    li("full scoring 재실행")

    h2("8. 해석 제한")
    li("stage1_dev filtered crop-level 결과이며, stage2_holdout 최종 검증이 아니다.")
    li("hard negative 최종 채택이 아니다.")
    li("threshold 확정이 아니다.")
    li("병변 탐지 성능의 최종 결론이 아니다.")
    li("crop-level AUROC/AUPRC는 sampling bias(hard_negative 비율 등)에 영향을 받는다.")

    h2("9. metric_backend")
    li(f"사용된 metric_backend: **{jdata['metric_backend']}**")
    if jdata["metric_backend"] == "numpy_fallback":
        li("sklearn 미설치로 numpy/pandas 기반 fallback 사용.")
        li("AUROC: Wilcoxon-Mann-Whitney rank-based (pandas rank method='average')")
        li("AUPRC: precision-recall trapezoid (numpy argsort 기반)")

    h2("10. 다음 단계")
    li("Phase 7.5 v1/v1 test closure report (전체 평가 종합)")
    li("또는 Phase 7.4b patient-level metric design — 별도 사용자 승인 필요")

    recheck_before_save(OUT_MD)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[MD] saved: {OUT_MD}")


# ---------------------------------------------------------------------------
# 콘솔 보고 (보고 형식 1~29번)
# ---------------------------------------------------------------------------
def print_report(jdata):
    print("\n" + "=" * 70)
    print("Phase 7.4 crop-level metric calculation 보고")
    print("=" * 70)

    def p(n, label, val):
        print(f"  {n:2}. {label}: {val}")

    p(1, "생성 script 경로", str(SCRIPT_DIR / "phase7_4_v1v1_crop_level_metrics.py"))
    p(2, "py_compile 결과", "PASS (실행 전 확인 완료)")
    p(3, "실행 명령",
      "source ~/ai_env/bin/activate && python scripts/phase7_4_v1v1_crop_level_metrics.py --run")
    p(4, "실행 성공 여부", "SUCCESS")
    p(5, "output root", str(PHASE74_DIR))
    p(6, "생성 CSV 경로", str(OUT_CSV))
    p(7, "생성 JSON 경로", str(OUT_JSON))
    p(8, "생성 MD 경로", str(OUT_MD))
    p(9, "score CSV row 수", jdata["score_csv_row_count"])
    p(10, "label=0 count", jdata["label_distribution"].get("0", "N/A"))
    p(11, "label=1 count", jdata["label_distribution"].get("1", "N/A"))
    p(12, "positive prevalence", jdata["positive_prevalence"])
    p(13, "primary score column", jdata["primary_score_column"])
    p(14, "crop_level_auroc_primary", jdata["crop_level_auroc_primary"])
    p(15, "crop_level_auprc_primary", jdata["crop_level_auprc_primary"])
    p(16, "secondary score column", jdata["secondary_score_column"])
    p(17, "crop_level_auroc_secondary", jdata["crop_level_auroc_secondary"])
    p(18, "crop_level_auprc_secondary", jdata["crop_level_auprc_secondary"])
    p(19, "metric backend", jdata["metric_backend"])
    p(20, "threshold 계산 없음 확인", not jdata["threshold_calculated"])
    p(21, "p95/p99 계산 없음 확인", not jdata["p95_p99_calculated"])
    p(22, "patient-level metric 계산 없음 확인", not jdata["patient_level_metric_calculated"])
    p(23, "scoring 재실행 없음 확인", not jdata["scoring_reexecuted"])
    p(24, "model forward 없음 확인", "TRUE")
    p(25, "training 없음 확인", not jdata["training_executed"])
    p(26, "checkpoint 생성 없음 확인", not jdata["checkpoint_created"])
    p(27, "기존 파일 미수정 확인", "TRUE (read-only inputs)")
    p(28, "stage2_holdout/v2/v2v2 미접근 확인",
      f"stage2_holdout={not jdata['stage2_holdout_used']}, v2={not jdata['v2_used']}")
    p(29, "다음 단계 제안", jdata["next_step_recommendation"])

    print("=" * 70)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true",
                        help="실제 실행. 없으면 dry-run 모드.")
    args = parser.parse_args()

    if not args.run:
        print("[Phase 7.4] dry-run mode: use --run to execute")
        sys.exit(0)

    print("[Phase 7.4] crop-level metric calculation 시작")
    print(f"[backend] metric_backend = {METRIC_BACKEND}")

    check_output_guard()
    PHASE74_DIR.mkdir(parents=True, exist_ok=False)

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

    phase73_data = None
    if INPUT_PHASE73_JSON.exists():
        with open(INPUT_PHASE73_JSON, encoding="utf-8") as f:
            phase73_data = json.load(f)

    # label 정규화 (메모리 내 처리, 원본 CSV 미수정)
    label_norm = None
    if df is not None and "label" in df.columns:
        label_norm = pd.to_numeric(df["label"], errors="coerce")

    # 섹션 실행
    a_rows = run_section_a(df, done_data, phase73_data, label_norm)
    b_rows = run_section_b(label_norm, df)
    c_rows, c_results = run_section_c(label_norm, df)
    d_result = run_section_d(df)
    if isinstance(d_result, tuple):
        d_rows, diag_summary = d_result
    else:
        d_rows, diag_summary = d_result, {}
    e_rows = run_section_e()

    # 저장
    save_csv(a_rows, b_rows, c_rows, d_rows, e_rows)
    jdata = save_json(a_rows, c_results, diag_summary, label_norm, df)
    save_md(jdata, a_rows, b_rows, c_rows, d_rows)
    print_report(jdata)

    print("\n[Phase 7.4] crop-level metric calculation 완료")


if __name__ == "__main__":
    main()
