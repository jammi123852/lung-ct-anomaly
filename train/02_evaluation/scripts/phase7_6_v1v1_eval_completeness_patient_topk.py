"""
Phase 7.6 v1/v1 evaluation completeness + patient/top-k metric calculation
- Phase 7.4 crop-level 이후 남은 성능평가 항목 정리
- patient-level AUROC/AUPRC feasibility 검증 및 계산 (가능한 경우만)
- threshold-free top-k ranking diagnostic metric 계산
- threshold / p95/p99 / stage2_holdout 평가 금지
- scoring 재실행 / model forward / training 금지
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

BASE = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1"

PHASE72_CSV = (
    BASE / "scores/phase7_2_v1v1_stage1_dev_full_scoring_v1"
    / "phase7_2_v1v1_stage1_dev_full_scoring_v1.csv"
)
PHASE74_JSON = (
    BASE / "evaluation/phase7_4_v1v1_crop_level_metrics_v1"
    / "phase7_4_v1v1_crop_level_metrics_v1.json"
)
PHASE74_MD = (
    BASE / "evaluation/phase7_4_v1v1_crop_level_metrics_v1"
    / "phase7_4_v1v1_crop_level_metrics_report_v1.md"
)
PHASE75_JSON = (
    BASE / "review_annotations/phase7_5_v1v1_test_closure_report_v1"
    / "phase7_5_v1v1_test_closure_report_v1.json"
)
PHASE75_MD = (
    BASE / "review_annotations/phase7_5_v1v1_test_closure_report_v1"
    / "phase7_5_v1v1_test_closure_report_v1.md"
)

OUT_DIR = (
    BASE / "evaluation/phase7_6_v1v1_eval_completeness_patient_topk_v1"
)
OUT_CSV = OUT_DIR / "phase7_6_v1v1_eval_completeness_patient_topk_v1.csv"
OUT_JSON = OUT_DIR / "phase7_6_v1v1_eval_completeness_patient_topk_v1.json"
OUT_MD = OUT_DIR / "phase7_6_v1v1_eval_completeness_patient_topk_report_v1.md"

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
EXPECTED_LABEL_0 = 85906
EXPECTED_LABEL_1 = 43531

SCORE_COLS = {
    "primary": "crop_score_l1_mean",
    "secondary": "crop_score_mse_mean",
}
AGGREGATIONS = [
    "patient_max",
    "patient_mean",
    "patient_top1_percent_mean",
    "patient_top5_percent_mean",
    "patient_top10_percent_mean",
]
PATIENT_LABEL_RULES = [
    "patient_positive_if_any_label1",
    "patient_positive_if_majority_label1",
    "patient_positive_if_positive_ratio_above_0",
]
TOPK_PERCENTS = [1, 5, 10, 20]


# ---------------------------------------------------------------------------
# output overwrite guard
# ---------------------------------------------------------------------------
def check_output_guard():
    if OUT_DIR.exists():
        print(f"[ERROR] output root already exists: {OUT_DIR}")
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


def safe_auroc_auprc(labels, scores):
    """feasibility 검사 후 계산. 불가하면 NOT_APPLICABLE 반환."""
    labels_arr = np.asarray(labels, dtype=float)
    n_pos = int((labels_arr == 1).sum())
    n_neg = int((labels_arr == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return "NOT_APPLICABLE", "NOT_APPLICABLE", f"label class 한쪽만 존재 (pos={n_pos}, neg={n_neg})"
    auroc = compute_auroc(labels_arr, np.asarray(scores, dtype=float))
    auprc = compute_auprc(labels_arr, np.asarray(scores, dtype=float))
    return round(auroc, 6), round(auprc, 6), "OK"


# ---------------------------------------------------------------------------
# has_nan / has_inf robust parser
# ---------------------------------------------------------------------------
def normalize_bool_series(s):
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({"false": 0, "true": 1, "0": 0, "1": 1})
    )


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------
def pass_fail(condition):
    return "PASS" if condition else "FAIL"


def fmt(v, digits=6):
    if isinstance(v, float) and v == v:
        return round(v, digits)
    return v


# ---------------------------------------------------------------------------
# 작업 1: score CSV integrity
# ---------------------------------------------------------------------------
def run_integrity(df):
    rows = []

    def add(item, expected, observed, status, note=""):
        rows.append({
            "section": "A_integrity",
            "item": item,
            "expected": str(expected),
            "observed": str(observed),
            "status": status,
            "note": note,
        })

    add("row_count", EXPECTED_ROW_COUNT, len(df), pass_fail(len(df) == EXPECTED_ROW_COUNT))
    add("n_cols", 27, len(df.columns), pass_fail(len(df.columns) == 27))

    missing_cols = [c for c in CANONICAL_COLS if c not in df.columns]
    add(
        "canonical_cols_present",
        "all 27",
        "PASS" if not missing_cols else str(missing_cols),
        pass_fail(not missing_cols),
    )

    score_nan = int(df[["crop_score_l1_mean", "crop_score_mse_mean", "label"]].isnull().sum().sum())
    add("key_col_has_nan", 0, score_nan, pass_fail(score_nan == 0))

    has_nan_sum = int(df["has_nan"].sum()) if "has_nan" in df.columns else -1
    add("has_nan_col_sum", 0, has_nan_sum, pass_fail(has_nan_sum == 0))

    has_inf_sum = int(df["has_inf"].sum()) if "has_inf" in df.columns else -1
    add("has_inf_col_sum", 0, has_inf_sum, pass_fail(has_inf_sum == 0))

    label_vals = set(df["label"].dropna().astype(int).unique().tolist())
    add("label_values", "{0,1}", str(label_vals), pass_fail(label_vals <= {0, 1}))

    label0 = int((df["label"] == 0).sum())
    label1 = int((df["label"] == 1).sum())
    add("label_0_count", EXPECTED_LABEL_0, label0, pass_fail(label0 == EXPECTED_LABEL_0))
    add("label_1_count", EXPECTED_LABEL_1, label1, pass_fail(label1 == EXPECTED_LABEL_1))

    if "npz_path" in df.columns:
        v2_paths = int(df["npz_path"].str.contains("v2v2|/v2/", na=False, regex=True).sum())
        add("v2_v2v2_path_count", 0, v2_paths, pass_fail(v2_paths == 0))

    if "stage_split" in df.columns:
        holdout_rows = int((df["stage_split"] == "stage2_holdout").sum())
        add("stage2_holdout_row_count", 0, holdout_rows, pass_fail(holdout_rows == 0))

    integrity_pass = all(r["status"] == "PASS" for r in rows)
    print(f"[작업 1] integrity: {'PASS' if integrity_pass else 'FAIL'} ({len(rows)} items checked)")
    return rows, integrity_pass


# ---------------------------------------------------------------------------
# 작업 2: patient label feasibility
# ---------------------------------------------------------------------------
def run_patient_label_feasibility(df):
    rows = []

    patient_label_stats = df.groupby("patient_id")["label"].agg(["min", "max"])
    all_pos_only = int(((patient_label_stats["min"] == 1) & (patient_label_stats["max"] == 1)).sum())
    all_neg_only = int(((patient_label_stats["min"] == 0) & (patient_label_stats["max"] == 0)).sum())
    mixed = int(((patient_label_stats["min"] == 0) & (patient_label_stats["max"] == 1)).sum())

    for rule in PATIENT_LABEL_RULES:
        if rule == "patient_positive_if_any_label1":
            patient_labels = df.groupby("patient_id")["label"].max().astype(int)
        elif rule == "patient_positive_if_majority_label1":
            patient_labels = (df.groupby("patient_id")["label"].mean() >= 0.5).astype(int)
        elif rule == "patient_positive_if_positive_ratio_above_0":
            patient_labels = (df.groupby("patient_id")["label"].mean() > 0).astype(int)
        else:
            continue

        pos_count = int((patient_labels == 1).sum())
        neg_count = int((patient_labels == 0).sum())
        feasible = pos_count > 0 and neg_count > 0
        reason = (
            "OK - both classes present"
            if feasible
            else f"NOT_APPLICABLE - pos={pos_count}, neg={neg_count} (한쪽 class만 존재)"
        )

        rows.append({
            "section": "B_patient_label_feasibility",
            "patient_label_rule": rule,
            "positive_patient_count": pos_count,
            "negative_patient_count": neg_count,
            "mixed_crop_patient_count": mixed,
            "all_positive_crop_patient_count": all_pos_only,
            "all_negative_crop_patient_count": all_neg_only,
            "feasible_for_auroc_auprc": "YES" if feasible else "NO",
            "reason": reason,
        })
        print(f"[작업 2] {rule}: pos={pos_count}, neg={neg_count} → {'feasible' if feasible else 'NOT_APPLICABLE'}")

    return rows


# ---------------------------------------------------------------------------
# 작업 3: patient aggregation metrics
# ---------------------------------------------------------------------------
def _top_percent_mean(s, pct):
    n = max(1, int(np.ceil(len(s) * pct / 100)))
    return float(s.nlargest(n).mean())


def run_patient_aggregation(df, feasibility_rows):
    rows = []

    feasibility = {
        r["patient_label_rule"]: r["feasible_for_auroc_auprc"] == "YES"
        for r in feasibility_rows
    }

    for score_tag, score_col in SCORE_COLS.items():
        for rule in PATIENT_LABEL_RULES:
            if rule == "patient_positive_if_any_label1":
                patient_labels = df.groupby("patient_id")["label"].max().astype(int)
            elif rule == "patient_positive_if_majority_label1":
                patient_labels = (df.groupby("patient_id")["label"].mean() >= 0.5).astype(int)
            elif rule == "patient_positive_if_positive_ratio_above_0":
                patient_labels = (df.groupby("patient_id")["label"].mean() > 0).astype(int)
            else:
                continue

            for agg in AGGREGATIONS:
                if agg == "patient_max":
                    patient_scores = df.groupby("patient_id")[score_col].max()
                elif agg == "patient_mean":
                    patient_scores = df.groupby("patient_id")[score_col].mean()
                elif agg == "patient_top1_percent_mean":
                    patient_scores = df.groupby("patient_id")[score_col].apply(
                        lambda s: _top_percent_mean(s, 1)
                    )
                elif agg == "patient_top5_percent_mean":
                    patient_scores = df.groupby("patient_id")[score_col].apply(
                        lambda s: _top_percent_mean(s, 5)
                    )
                elif agg == "patient_top10_percent_mean":
                    patient_scores = df.groupby("patient_id")[score_col].apply(
                        lambda s: _top_percent_mean(s, 10)
                    )
                else:
                    continue

                common_ids = patient_labels.index.intersection(patient_scores.index)
                labels_arr = patient_labels.loc[common_ids].values
                scores_arr = patient_scores.loc[common_ids].values

                if feasibility.get(rule, False):
                    auroc, auprc, note = safe_auroc_auprc(labels_arr, scores_arr)
                    status = "DONE" if auroc != "NOT_APPLICABLE" else "NOT_APPLICABLE"
                else:
                    auroc = "NOT_APPLICABLE"
                    auprc = "NOT_APPLICABLE"
                    note = "label feasibility FAIL"
                    status = "NOT_APPLICABLE"

                rows.append({
                    "section": "C_patient_aggregation",
                    "score_column": score_col,
                    "aggregation": agg,
                    "patient_label_rule": rule,
                    "metric_name": "patient_level_auroc",
                    "value": str(auroc),
                    "status": status,
                    "note": note,
                })
                rows.append({
                    "section": "C_patient_aggregation",
                    "score_column": score_col,
                    "aggregation": agg,
                    "patient_label_rule": rule,
                    "metric_name": "patient_level_auprc",
                    "value": str(auprc),
                    "status": status,
                    "note": note,
                })

    done_count = sum(
        1 for r in rows if r["status"] == "DONE" and r["metric_name"] == "patient_level_auroc"
    )
    na_count = sum(
        1 for r in rows if r["status"] == "NOT_APPLICABLE" and r["metric_name"] == "patient_level_auroc"
    )
    print(f"[작업 3] patient aggregation: DONE={done_count}, NOT_APPLICABLE={na_count}")
    return rows


# ---------------------------------------------------------------------------
# 작업 4: threshold-free top-k ranking
# ---------------------------------------------------------------------------
def run_topk_ranking(df):
    rows = []

    for score_tag, score_col in SCORE_COLS.items():
        total_crops = len(df)

        # 전체 crop 기준
        for pct in TOPK_PERCENTS:
            n_selected = max(1, int(np.ceil(total_crops * pct / 100)))
            top_idx = df[score_col].nlargest(n_selected).index
            n_pos_selected = int(df.loc[top_idx, "label"].sum())
            pos_ratio = round(n_pos_selected / n_selected, 6) if n_selected > 0 else 0.0

            rows.append({
                "section": "D_topk_ranking",
                "score_column": score_col,
                "level": "global",
                "topk_percent": pct,
                "positive_crop_ratio": pos_ratio,
                "n_selected": n_selected,
                "n_positive_selected": n_pos_selected,
                "status": "DONE",
                "note": "threshold-free ranking diagnostic only",
            })

        # patient별 평균
        for pct in TOPK_PERCENTS:
            per_patient_ratios = []
            for _pid, grp in df.groupby("patient_id"):
                n_sel = max(1, int(np.ceil(len(grp) * pct / 100)))
                top_idx_p = grp[score_col].nlargest(n_sel).index
                n_pos_p = int(grp.loc[top_idx_p, "label"].sum())
                per_patient_ratios.append(n_pos_p / n_sel if n_sel > 0 else 0.0)

            avg_ratio = round(float(np.mean(per_patient_ratios)), 6)
            rows.append({
                "section": "D_topk_ranking",
                "score_column": score_col,
                "level": "per_patient_mean",
                "topk_percent": pct,
                "positive_crop_ratio": avg_ratio,
                "n_selected": "per_patient",
                "n_positive_selected": "per_patient",
                "status": "DONE",
                "note": "threshold-free ranking diagnostic only",
            })

    print(f"[작업 4] top-k ranking: {len(rows)} rows 생성")
    return rows


# ---------------------------------------------------------------------------
# 작업 5: evaluation completeness
# ---------------------------------------------------------------------------
def run_completeness(feasibility_rows, patient_rows, topk_rows):
    any_done = any(
        r["status"] == "DONE"
        for r in patient_rows
        if r["metric_name"] == "patient_level_auroc"
    )
    all_na = all(
        r["status"] == "NOT_APPLICABLE"
        for r in patient_rows
        if r["metric_name"] == "patient_level_auroc"
    )
    patient_status = "DONE" if any_done else ("NOT_APPLICABLE" if all_na else "PARTIAL")
    topk_done = len(topk_rows) > 0

    rows = [
        {
            "section": "E_completeness",
            "evaluation_item": "crop_level_auroc_auprc",
            "status": "DONE",
            "evidence": "Phase 7.4 완료 (crop_score_l1_mean AUROC=0.649008, AUPRC=0.397344)",
            "limitation": "stage1_dev filtered crop-level only",
            "next_required_action": "none",
        },
        {
            "section": "E_completeness",
            "evaluation_item": "patient_level_auroc_auprc",
            "status": patient_status,
            "evidence": "Phase 7.6 feasibility 검증 결과",
            "limitation": "stage1_dev filtered crop-level only; patient label rule 의존",
            "next_required_action": (
                "none"
                if patient_status in ("DONE", "NOT_APPLICABLE")
                else "feasibility 재검토 필요"
            ),
        },
        {
            "section": "E_completeness",
            "evaluation_item": "threshold_free_topk_ranking",
            "status": "DONE" if topk_done else "FAIL",
            "evidence": f"{len(topk_rows)} rows 생성",
            "limitation": "threshold 확정 아님; ranking diagnostic only",
            "next_required_action": "none",
        },
        {
            "section": "E_completeness",
            "evaluation_item": "threshold_p95_p99_hit_rate",
            "status": "NOT_DONE",
            "evidence": "threshold policy 미결정",
            "limitation": "별도 승인 필요",
            "next_required_action": "threshold policy preflight 별도 승인 후 진행",
        },
        {
            "section": "E_completeness",
            "evaluation_item": "stage2_holdout_evaluation",
            "status": "LOCKED",
            "evidence": "stage2_holdout 봉인 유지",
            "limitation": "별도 승인 필요",
            "next_required_action": "stage2_holdout preflight 별도 승인 후 진행",
        },
        {
            "section": "E_completeness",
            "evaluation_item": "training_retraining",
            "status": "NOT_DONE",
            "evidence": "이번 단계에서 수행하지 않음",
            "limitation": "별도 결정 필요",
            "next_required_action": "별도 승인 후 진행",
        },
        {
            "section": "E_completeness",
            "evaluation_item": "hard_negative_final_manifest",
            "status": "NOT_DONE",
            "evidence": "이번 단계에서 수행하지 않음",
            "limitation": "별도 결정 필요",
            "next_required_action": "별도 승인 후 진행",
        },
    ]

    key_statuses = [
        r["status"]
        for r in rows
        if r["evaluation_item"] in (
            "crop_level_auroc_auprc",
            "patient_level_auroc_auprc",
            "threshold_free_topk_ranking",
        )
    ]
    final_ready = all(s in ("DONE", "NOT_APPLICABLE") for s in key_statuses)
    final_status = (
        "READY_FOR_PHASE7_7_FINAL_PERFORMANCE_CLOSURE"
        if final_ready
        else "BLOCKED"
    )

    print(f"[작업 5] completeness: {final_status}")
    return rows, final_status


# ---------------------------------------------------------------------------
# CSV 출력
# ---------------------------------------------------------------------------
def write_csv(integrity_rows, feasibility_rows, patient_rows, topk_rows, completeness_rows):
    recheck_before_save(OUT_CSV)

    all_rows = integrity_rows + feasibility_rows + patient_rows + topk_rows + completeness_rows

    all_keys = []
    for r in all_rows:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: r.get(k, "") for k in all_keys})

    print(f"[CSV] {OUT_CSV} ({len(all_rows)} rows)")


# ---------------------------------------------------------------------------
# JSON 출력
# ---------------------------------------------------------------------------
def write_json(
    integrity_rows, feasibility_rows, patient_rows, topk_rows, completeness_rows,
    label_dist, integrity_pass, final_status, phase74_metrics,
):
    recheck_before_save(OUT_JSON)

    patient_auroc_done = {}
    for r in patient_rows:
        if r["metric_name"] == "patient_level_auroc" and r["status"] == "DONE":
            key = f"{r['score_column']}__{r['aggregation']}__{r['patient_label_rule']}"
            try:
                patient_auroc_done[key] = float(r["value"])
            except (ValueError, TypeError):
                patient_auroc_done[key] = r["value"]

    patient_metric_status = "DONE" if patient_auroc_done else "NOT_APPLICABLE"

    output = {
        "input_paths": {
            "phase7_2_score_csv": str(PHASE72_CSV),
            "phase7_4_json": str(PHASE74_JSON),
            "phase7_4_md": str(PHASE74_MD),
            "phase7_5_json": str(PHASE75_JSON),
            "phase7_5_md": str(PHASE75_MD),
        },
        "score_csv_integrity": {
            "pass": integrity_pass,
            "row_count": label_dist["total"],
            "n_cols": 27,
        },
        "label_distribution": label_dist,
        "patient_label_feasibility": [
            {k: v for k, v in r.items() if k != "section"}
            for r in feasibility_rows
        ],
        "patient_level_metrics": patient_auroc_done,
        "patient_level_metric_status": patient_metric_status,
        "topk_ranking_metrics": [
            {k: v for k, v in r.items() if k != "section"}
            for r in topk_rows
        ],
        "evaluation_completeness_status": {
            r["evaluation_item"]: r["status"] for r in completeness_rows
        },
        "crop_level_metrics_from_phase7_4": phase74_metrics,
        "final_recommendation": final_status,
        "notes": {
            "no_scoring_reexecution": True,
            "no_model_forward": True,
            "no_training": True,
            "no_threshold": True,
            "no_p95_p99": True,
            "no_stage2_holdout": True,
            "no_v2": True,
        },
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[JSON] {OUT_JSON}")


# ---------------------------------------------------------------------------
# MD report 출력
# ---------------------------------------------------------------------------
def write_md(
    feasibility_rows, patient_rows, topk_rows, completeness_rows,
    label_dist, integrity_pass, final_status, phase74_metrics,
):
    recheck_before_save(OUT_MD)

    lines = []
    lines.append("# Phase 7.6 v1/v1 evaluation completeness + patient/top-k metric")
    lines.append("")
    lines.append("## 1. Phase 7.6 목적")
    lines.append("")
    lines.append("- Phase 7.4 crop-level AUROC/AUPRC 이후 남은 성능평가 항목 정리")
    lines.append("- patient-level AUROC/AUPRC feasibility 검증 및 계산 (feasibility 통과 시만)")
    lines.append("- threshold-free top-k ranking diagnostic metric 계산")
    lines.append("- threshold / p95/p99 / stage2_holdout 평가 금지")
    lines.append("- scoring 재실행 / model forward / training 금지")
    lines.append("")

    lines.append("## 2. score CSV integrity")
    lines.append("")
    lines.append(f"- row_count: {label_dist['total']} (expected: {EXPECTED_ROW_COUNT}) → {'PASS' if label_dist['total'] == EXPECTED_ROW_COUNT else 'FAIL'}")
    lines.append(f"- label=0: {label_dist['label_0']} (expected: {EXPECTED_LABEL_0})")
    lines.append(f"- label=1: {label_dist['label_1']} (expected: {EXPECTED_LABEL_1})")
    lines.append(f"- positive_prevalence: {label_dist['positive_prevalence']}")
    lines.append(f"- integrity_pass: {'PASS' if integrity_pass else 'FAIL'}")
    lines.append("")

    lines.append("## 3. patient-level AUROC/AUPRC 가능 여부")
    lines.append("")
    lines.append("| patient_label_rule | pos_patients | neg_patients | feasible |")
    lines.append("|---|---|---|---|")
    for r in feasibility_rows:
        lines.append(
            f"| {r['patient_label_rule']} "
            f"| {r['positive_patient_count']} "
            f"| {r['negative_patient_count']} "
            f"| {r['feasible_for_auroc_auprc']} |"
        )
    lines.append("")

    lines.append("## 4. patient-level metric 결과")
    lines.append("")
    feasible_rules = [
        r["patient_label_rule"]
        for r in feasibility_rows
        if r["feasible_for_auroc_auprc"] == "YES"
    ]
    if not feasible_rules:
        lines.append("**NOT_APPLICABLE**: 모든 patient label rule에서 양쪽 class가 동시에 존재하지 않아 AUROC/AUPRC 계산 불가.")
        lines.append("")
        lines.append("- stage1_dev filtered crop-level 데이터에서 patient-level binary label이 양쪽 class로 분리되지 않음.")
        lines.append("- 이는 patient AUROC가 정의되지 않는 정상적인 상황이며, 결과 생략이 오류가 아니다.")
    else:
        lines.append("| score_column | aggregation | patient_label_rule | AUROC | AUPRC |")
        lines.append("|---|---|---|---|---|")
        auroc_map = {}
        auprc_map = {}
        for r in patient_rows:
            key = (r["score_column"], r["aggregation"], r["patient_label_rule"])
            if r["metric_name"] == "patient_level_auroc":
                auroc_map[key] = r["value"]
            elif r["metric_name"] == "patient_level_auprc":
                auprc_map[key] = r["value"]
        for key in auroc_map:
            lines.append(
                f"| {key[0]} | {key[1]} | {key[2]} "
                f"| {auroc_map.get(key, '')} | {auprc_map.get(key, '')} |"
            )
    lines.append("")

    lines.append("## 5. threshold-free top-k ranking 결과")
    lines.append("")
    lines.append("> **주의**: threshold 확정이 아니며 ranking diagnostic metric이다.")
    lines.append("")
    for score_tag, score_col in SCORE_COLS.items():
        lines.append(f"### {score_col} ({score_tag})")
        lines.append("")
        lines.append("**global level**")
        lines.append("")
        lines.append("| top-k% | n_selected | n_positive | positive_ratio |")
        lines.append("|---|---|---|---|")
        for r in topk_rows:
            if r["score_column"] == score_col and r["level"] == "global":
                lines.append(f"| {r['topk_percent']}% | {r['n_selected']} | {r['n_positive_selected']} | {r['positive_crop_ratio']} |")
        lines.append("")
        lines.append("**per-patient mean level**")
        lines.append("")
        lines.append("| top-k% | avg_positive_ratio |")
        lines.append("|---|---|")
        for r in topk_rows:
            if r["score_column"] == score_col and r["level"] == "per_patient_mean":
                lines.append(f"| {r['topk_percent']}% | {r['positive_crop_ratio']} |")
        lines.append("")

    lines.append("## 6. Phase 7.4 + Phase 7.6 성능평가 요약")
    lines.append("")
    lines.append("| metric | score_column | value | scope |")
    lines.append("|---|---|---|---|")
    lines.append(f"| crop_level_auroc | crop_score_l1_mean | {phase74_metrics.get('crop_level_auroc_primary', 'N/A')} | stage1_dev crop |")
    lines.append(f"| crop_level_auprc | crop_score_l1_mean | {phase74_metrics.get('crop_level_auprc_primary', 'N/A')} | stage1_dev crop |")
    lines.append(f"| crop_level_auroc | crop_score_mse_mean | {phase74_metrics.get('crop_level_auroc_secondary', 'N/A')} | stage1_dev crop |")
    lines.append(f"| crop_level_auprc | crop_score_mse_mean | {phase74_metrics.get('crop_level_auprc_secondary', 'N/A')} | stage1_dev crop |")
    if not feasible_rules:
        lines.append("| patient_level_auroc_auprc | — | NOT_APPLICABLE | — |")
    lines.append("")

    lines.append("## 7. 아직 하지 않은 것")
    lines.append("")
    for r in completeness_rows:
        if r["status"] in ("NOT_DONE", "LOCKED"):
            lines.append(f"- **{r['evaluation_item']}**: {r['status']} — {r['next_required_action']}")
    lines.append("")

    lines.append("## 8. 최종 판정")
    lines.append("")
    lines.append(f"**{final_status}**")
    lines.append("")

    lines.append("## 9. 해석 제한")
    lines.append("")
    lines.append("- 본 결과는 stage1_dev filtered crop-level 범위에 한정된다.")
    lines.append("- stage2_holdout 최종 검증 결과가 아니다.")
    lines.append("- threshold 기반 성능 평가가 수행되지 않았다.")
    lines.append("- patient-level label이 feasible한 경우에만 patient AUROC/AUPRC를 산출하였다.")
    lines.append("- top-k ranking은 threshold 확정이 아닌 ranking diagnostic metric이다.")
    lines.append("- 병변 성능 최종 결론이 아니다.")
    lines.append("")

    lines.append("## 10. 금지 사항 확인")
    lines.append("")
    lines.append("- [확인] scoring 재실행 금지")
    lines.append("- [확인] model forward 금지")
    lines.append("- [확인] training 금지")
    lines.append("- [확인] threshold 계산 금지")
    lines.append("- [확인] p95/p99 계산 금지")
    lines.append("- [확인] stage2_holdout 접근 금지")
    lines.append("- [확인] v2/v2v2 접근 금지")
    lines.append("- [확인] 기존 Phase 6/7 output 수정 금지")
    lines.append("")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[MD] {OUT_MD}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Phase 7.6 evaluation completeness + patient/top-k metric"
    )
    parser.add_argument("--run", action="store_true", help="실제 실행 (없으면 dry-run)")
    args = parser.parse_args()

    if not args.run:
        print("dry-run: --run 플래그를 붙여 실행하세요.")
        print(f"  score CSV: {PHASE72_CSV}")
        print(f"  phase7_4 JSON: {PHASE74_JSON}")
        print(f"  phase7_5 JSON: {PHASE75_JSON}")
        print(f"  output: {OUT_DIR}")
        return

    # input 파일 존재 확인
    for p in [PHASE72_CSV, PHASE74_JSON, PHASE74_MD, PHASE75_JSON, PHASE75_MD]:
        if not p.exists():
            print(f"[ERROR] input file not found: {p}")
            sys.exit(1)

    # output guard
    check_output_guard()

    # Phase 7.4 metrics 로드
    with open(PHASE74_JSON, encoding="utf-8") as f:
        phase74_data = json.load(f)
    phase74_metrics = {
        "crop_level_auroc_primary": phase74_data.get("crop_level_auroc_primary"),
        "crop_level_auprc_primary": phase74_data.get("crop_level_auprc_primary"),
        "crop_level_auroc_secondary": phase74_data.get("crop_level_auroc_secondary"),
        "crop_level_auprc_secondary": phase74_data.get("crop_level_auprc_secondary"),
        "positive_prevalence": phase74_data.get("positive_prevalence"),
        "metric_backend": METRIC_BACKEND,
    }

    # score CSV 로드
    print("[로드] score CSV 읽는 중...")
    df = pd.read_csv(PHASE72_CSV, dtype={"label": int})
    print(f"[로드] {len(df):,} rows 완료")
    # has_nan / has_inf robust parsing (원본 CSV 수정 없음, 메모리 내부만)
    for col in ["has_nan", "has_inf"]:
        if col in df.columns:
            df[col] = normalize_bool_series(df[col])
            if df[col].isnull().any():
                print(f"[ERROR] {col} 정규화 후 NaN 존재 — 허용값(0/1/False/True) 외 값 포함")
                sys.exit(1)

    label_dist = {
        "total": len(df),
        "label_0": int((df["label"] == 0).sum()),
        "label_1": int((df["label"] == 1).sum()),
        "positive_prevalence": round(float((df["label"] == 1).mean()), 4),
    }

    # 작업 실행
    integrity_rows, integrity_pass = run_integrity(df)
    feasibility_rows = run_patient_label_feasibility(df)
    patient_rows = run_patient_aggregation(df, feasibility_rows)
    topk_rows = run_topk_ranking(df)
    completeness_rows, final_status = run_completeness(feasibility_rows, patient_rows, topk_rows)

    # 출력
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    write_csv(integrity_rows, feasibility_rows, patient_rows, topk_rows, completeness_rows)
    write_json(
        integrity_rows, feasibility_rows, patient_rows, topk_rows, completeness_rows,
        label_dist, integrity_pass, final_status, phase74_metrics,
    )
    write_md(
        feasibility_rows, patient_rows, topk_rows, completeness_rows,
        label_dist, integrity_pass, final_status, phase74_metrics,
    )

    print(f"\n=== Phase 7.6 완료 ===")
    print(f"최종 판정: {final_status}")
    print(f"출력 경로: {OUT_DIR}")


if __name__ == "__main__":
    main()
