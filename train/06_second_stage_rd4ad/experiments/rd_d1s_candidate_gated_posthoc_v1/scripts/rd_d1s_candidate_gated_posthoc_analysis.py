"""
RD-D1s Candidate-Gated Post-hoc Analysis
실험 root: experiments/rd_d1s_candidate_gated_posthoc_v1/
- 기존 파일 수정 없음 (read-only)
- 모델 forward / 재학습 / 새 scoring 없음
- label은 evaluation용으로만 사용 (selector 아님)
- stage2_holdout 접근 없음
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# ── 경로 설정 ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
SCORE_CSV = PROJECT_ROOT / "outputs/normal_based_stage2_verifier_audit/rd_d1s_medi3ch_true_rd4ad_shard_run_v1/rd_d1s_stage1dev_candidate_score.csv"
MANIFEST_CSV = PROJECT_ROOT / "outputs/normal_based_stage2_verifier_audit/rd_c2_effb0_v420_candidate_rd4ad_retest_v1/rd_c2_effb0_v420_candidate_manifest.csv"
STAGE1_DEV_PATIENTS_TXT = PROJECT_ROOT / "outputs/mip-postprocess-research-v1/manifests/stage1_dev_patients.txt"
OUT_ROOT = PROJECT_ROOT / "experiments/rd_d1s_candidate_gated_posthoc_v1"
REPORT_DIR = OUT_ROOT / "reports"
MANIFEST_DIR = OUT_ROOT / "manifests"
LOG_DIR = OUT_ROOT / "logs"

# ── Selector 컬럼 명세 ────────────────────────────────────────────────────────
# selector로 허용된 컬럼만 사용함을 명시
SELECTOR_USED_COLUMNS = [
    "patient_id",   # groupby key
    "local_z",      # z-continuity 기준
    "first_stage_score",    # PaDiM 1차 score (calibrated threshold 비교용)
    "threshold_p99",        # calibrated normal_val threshold (global fixed)
]
FORBIDDEN_SELECTOR_COLUMNS = [
    "label", "positive", "hard_negative", "lesion", "gt",
    "mask_overlap", "lesion_overlap", "is_positive", "class", "target",
]

# ── Selector 타입 명세 ────────────────────────────────────────────────────────
SELECTOR_META = {
    "selector_type": "global_calibrated_normal_val_threshold_p99",
    "threshold_source": "normal_val_threshold.json (EfficientNet-B0 normal validation set)",
    "threshold_scope": "global_fixed",           # 모든 환자에 동일 threshold_p99 적용
    "threshold_p99_value": 15.472385,
    "calibrated_detector": True,                 # 고정 threshold, per-patient percentile 아님
    "candidate_scheduler": True,                 # 이상 판정이 아닌 2차 후보 감소 규칙
    "note": (
        "threshold_p99가 전체 152환자에 동일값(15.472385) 적용 확인. "
        "이는 환자별 top-percentile이 아닌 normal_val calibrated threshold임."
    ),
}

GUARDRAILS = {
    "stage2_holdout_accessed": False,
    "model_forward_executed": False,
    "training_executed": False,
    "scoring_executed": False,
    "checkpoint_loaded": False,
    "existing_artifact_modified": False,
    "existing_script_modified": False,
    "selector_used_columns": SELECTOR_USED_COLUMNS,
    "forbidden_selector_columns": FORBIDDEN_SELECTOR_COLUMNS,
    "forbidden_selector_columns_intersection": [],  # 실행 시 업데이트
    "label_used_as_selector": False,
    "label_used_for_evaluation_only": True,
    "threshold_recalculated": False,
    "output_overwrite": False,
}

errors = []


def log_error(msg: str):
    errors.append({"timestamp": datetime.now().isoformat(), "error": msg})
    print(f"[ERROR] {msg}", file=sys.stderr)


def compute_auroc_auprc(df: pd.DataFrame) -> dict:
    """binary label: positive=1, hard_negative=0. label leakage 없음."""
    y_true = (df["label"] == "positive").astype(int)
    y_score = df["rd_d1s_medi3ch_rd4ad_score"]
    n_pos = y_true.sum()
    n_neg = (y_true == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return {"auroc": None, "auprc": None, "n_pos": int(n_pos), "n_neg": int(n_neg)}
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    return {"auroc": round(auroc, 4), "auprc": round(auprc, 4), "n_pos": int(n_pos), "n_neg": int(n_neg)}


def compute_variant_metrics(df_variant: pd.DataFrame, df_a0: pd.DataFrame, variant_name: str) -> dict:
    """각 variant의 metric 계산"""
    n_total_a0 = len(df_a0)
    n_variant = len(df_variant)
    reduction_rate = round(1.0 - n_variant / n_total_a0, 4) if n_total_a0 > 0 else None

    # patient metrics
    patients_a0 = set(df_a0["patient_id"].unique())
    patients_variant = set(df_variant["patient_id"].unique())
    patient_retained = len(patients_variant)
    patient_coverage = round(len(patients_variant) / len(patients_a0), 4) if len(patients_a0) > 0 else None

    # positive crop retention
    pos_a0 = df_a0[df_a0["label"] == "positive"]
    pos_variant = df_variant[df_variant["label"] == "positive"]
    pos_retention = round(len(pos_variant) / len(pos_a0), 4) if len(pos_a0) > 0 else None

    # hard_negative retention
    hn_a0 = df_a0[df_a0["label"] == "hard_negative"]
    hn_variant = df_variant[df_variant["label"] == "hard_negative"]
    hn_retention = round(len(hn_variant) / len(hn_a0), 4) if len(hn_a0) > 0 else None
    hn_suppression = round(1.0 - len(hn_variant) / len(hn_a0), 4) if len(hn_a0) > 0 else None

    # lesion slice coverage: unique (patient_id, local_z) pairs with positive label
    pos_slices_a0 = set(zip(pos_a0["patient_id"], pos_a0["local_z"]))
    pos_slices_variant = set(zip(pos_variant["patient_id"], pos_variant["local_z"]))
    lesion_slice_coverage = round(len(pos_slices_variant) / len(pos_slices_a0), 4) if len(pos_slices_a0) > 0 else None

    # lesion patient coverage
    pos_patients_a0 = set(pos_a0["patient_id"].unique())
    pos_patients_variant = set(pos_variant["patient_id"].unique())
    lesion_patient_coverage = round(len(pos_patients_variant) / len(pos_patients_a0), 4) if len(pos_patients_a0) > 0 else None

    # pat_all_sup: patients where all positive candidates were suppressed
    pat_all_sup_list = sorted(pos_patients_a0 - pos_patients_variant)
    pat_all_sup_count = len(pat_all_sup_list)

    # AUROC / AUPRC on selected set (eval only)
    auroc_auprc = compute_auroc_auprc(df_variant) if len(df_variant) > 0 else {"auroc": None, "auprc": None}

    # threshold sweep: lesion_rate ≤1%, ≤3%, ≤5%
    hn_suppression_by_rate = {}
    if len(df_variant) > 0:
        scores_sorted = df_variant["rd_d1s_medi3ch_rd4ad_score"].sort_values(ascending=False)
        for rate in [0.01, 0.03, 0.05]:
            topk = max(1, int(len(df_variant) * rate))
            threshold_at_rate = scores_sorted.iloc[topk - 1] if topk <= len(scores_sorted) else scores_sorted.iloc[-1]
            above = df_variant[df_variant["rd_d1s_medi3ch_rd4ad_score"] >= threshold_at_rate]
            hn_above = (above["label"] == "hard_negative").sum()
            hn_total_variant = (df_variant["label"] == "hard_negative").sum()
            hn_suppression_at_rate = round(1.0 - hn_above / hn_total_variant, 4) if hn_total_variant > 0 else None
            hn_suppression_by_rate[f"lesion_rate_{int(rate*100)}pct_hn_suppression"] = hn_suppression_at_rate

    # patient-level max score
    pat_max_score = df_variant.groupby("patient_id")["rd_d1s_medi3ch_rd4ad_score"].max()
    pat_top3_mean = df_variant.groupby("patient_id")["rd_d1s_medi3ch_rd4ad_score"].apply(
        lambda x: x.nlargest(3).mean()
    )

    # z 분포
    z_q25, z_q50, z_q75 = (
        float(df_variant["local_z"].quantile(0.25)) if len(df_variant) > 0 else None,
        float(df_variant["local_z"].quantile(0.50)) if len(df_variant) > 0 else None,
        float(df_variant["local_z"].quantile(0.75)) if len(df_variant) > 0 else None,
    )

    return {
        "variant": variant_name,
        "n_total_a0": n_total_a0,
        "n_variant": n_variant,
        "candidate_reduction_rate": reduction_rate,
        "patient_retained": patient_retained,
        "patient_coverage": patient_coverage,
        "lesion_slice_coverage": lesion_slice_coverage,
        "lesion_patient_coverage": lesion_patient_coverage,
        "positive_crop_retention": pos_retention,
        "hard_negative_retention": hn_retention,
        "hard_negative_suppression": hn_suppression,
        "pat_all_sup_count": pat_all_sup_count,
        "pat_all_sup_list": pat_all_sup_list,
        "auroc": auroc_auprc.get("auroc"),
        "auprc": auroc_auprc.get("auprc"),
        "n_positive_in_variant": int(auroc_auprc.get("n_pos", 0)),
        "n_hard_negative_in_variant": int(auroc_auprc.get("n_neg", 0)),
        **hn_suppression_by_rate,
        "z_q25": z_q25,
        "z_q50": z_q50,
        "z_q75": z_q75,
        "patient_max_score_mean": round(float(pat_max_score.mean()), 6) if len(pat_max_score) > 0 else None,
        "patient_top3mean_score_mean": round(float(pat_top3_mean.mean()), 6) if len(pat_top3_mean) > 0 else None,
    }


def get_problem_patients(df_variant: pd.DataFrame, df_a0: pd.DataFrame, variant_name: str) -> pd.DataFrame:
    """coverage <50%, <80%, <95% 환자 목록"""
    pos_a0 = df_a0[df_a0["label"] == "positive"]
    pos_variant = df_variant[df_variant["label"] == "positive"]

    # per-patient positive slice coverage
    def coverage_for_patient(pid):
        slices_a0 = set(pos_a0[pos_a0["patient_id"] == pid]["local_z"])
        slices_variant = set(pos_variant[pos_variant["patient_id"] == pid]["local_z"])
        if len(slices_a0) == 0:
            return None
        return len(slices_variant) / len(slices_a0)

    pos_patients = sorted(pos_a0["patient_id"].unique())
    rows = []
    for pid in pos_patients:
        cov = coverage_for_patient(pid)
        if cov is None:
            continue
        flags = {
            "coverage_lt_50": cov < 0.5,
            "coverage_lt_80": cov < 0.8,
            "coverage_lt_95": cov < 0.95,
        }
        n_pos_a0 = int((pos_a0["patient_id"] == pid).sum())
        n_pos_var = int((pos_variant["patient_id"] == pid).sum())
        rows.append({
            "variant": variant_name,
            "patient_id": pid,
            "slice_coverage": round(cov, 4),
            "n_pos_a0": n_pos_a0,
            "n_pos_variant": n_pos_var,
            **flags,
        })

    return pd.DataFrame(rows)


def main():
    # BLOCKER 1 수정: --run-posthoc --confirm-readonly 필수 인자로 실수 실행 방지
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-posthoc", action="store_true", required=True,
                        help="post-hoc analysis 실행 확인 플래그")
    parser.add_argument("--confirm-readonly", action="store_true", required=True,
                        help="기존 파일 read-only 사용 확인 플래그")
    args_cli = parser.parse_args()
    if not (args_cli.run_posthoc and args_cli.confirm_readonly):
        print("[ABORT] --run-posthoc --confirm-readonly 양쪽 모두 필요합니다.", file=sys.stderr)
        sys.exit(1)

    # BLOCKER 2 수정: selector 컬럼 금지 교집합 사전 점검
    merged_cols_to_use = set(SELECTOR_USED_COLUMNS)
    forbidden_intersection = sorted(merged_cols_to_use & set(FORBIDDEN_SELECTOR_COLUMNS))
    GUARDRAILS["forbidden_selector_columns_intersection"] = forbidden_intersection
    if forbidden_intersection:
        GUARDRAILS["label_used_as_selector"] = True
        log_error(f"FORBIDDEN selector columns detected: {forbidden_intersection}")
        print("[ABORT] label leakage 발생. 실행 중단.", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("RD-D1s Candidate-Gated Post-hoc Analysis")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    print("\n[1/6] 데이터 로드 (read-only)")
    if not SCORE_CSV.exists():
        log_error(f"score CSV 없음: {SCORE_CSV}")
        sys.exit(1)
    if not MANIFEST_CSV.exists():
        log_error(f"manifest CSV 없음: {MANIFEST_CSV}")
        sys.exit(1)

    df_score = pd.read_csv(SCORE_CSV)
    df_manifest = pd.read_csv(MANIFEST_CSV)
    print(f"  score CSV: {len(df_score):,} rows")
    print(f"  manifest CSV: {len(df_manifest):,} rows")

    # stage1_dev only
    df_score = df_score[df_score["stage_split"] == "stage1_dev"].copy()
    df_manifest = df_manifest[df_manifest["stage_split"] == "stage1_dev"].copy()
    print(f"  stage1_dev score: {len(df_score):,}, manifest: {len(df_manifest):,}")

    # stage2_holdout 접근 확인 (filter 전 원본에서 체크)
    raw_score = pd.read_csv(SCORE_CSV, usecols=["stage_split"])
    if (raw_score["stage_split"] == "stage2_holdout").any():
        GUARDRAILS["stage2_holdout_accessed"] = True
        log_error("stage2_holdout detected in score CSV!")
    raw_manifest = pd.read_csv(MANIFEST_CSV, usecols=["stage_split"])
    if (raw_manifest["stage_split"] == "stage2_holdout").any():
        GUARDRAILS["stage2_holdout_accessed"] = True
        log_error("stage2_holdout detected in manifest CSV!")
    del raw_score, raw_manifest

    # WARNING 2: 152 vs 154 환자 수 불일치 기록
    stage1_dev_total_expected = 154
    stage1_dev_all_patients = []
    if STAGE1_DEV_PATIENTS_TXT.exists():
        stage1_dev_all_patients = STAGE1_DEV_PATIENTS_TXT.read_text().strip().splitlines()
    score_patients = set(df_score["patient_id"].unique())
    missing_patients = sorted(set(stage1_dev_all_patients) - score_patients)
    patient_count_note = {
        "stage1_dev_total_expected": stage1_dev_total_expected,
        "candidate_score_patients": len(score_patients),
        "missing_from_candidate_score": missing_patients,
        "missing_count": len(missing_patients),
        "missing_reason": (
            "RD-C2 manifest 생성 단계에서 제외됨 "
            "(positive lesion candidate 없거나 candidate rule에서 누락). "
            "RD-D1s scoring 평가 대상은 candidate 존재 환자만 포함."
        ),
    }
    print(f"  stage1_dev expected: {stage1_dev_total_expected}, "
          f"in score CSV: {len(score_patients)}, "
          f"missing: {missing_patients}")

    # ── 병합 ─────────────────────────────────────────────────────────────────
    print("\n[2/6] candidate_id 기준 병합")
    df_manifest_slim = df_manifest[[
        "candidate_id", "first_stage_score", "threshold_p95", "threshold_p99",
        "candidate_rule", "sampling_reason", "slice_index"
    ]].copy()
    df = df_score.merge(df_manifest_slim, on="candidate_id", how="inner")
    print(f"  merged: {len(df):,} rows (expected 113,447)")
    if len(df) != len(df_score):
        log_error(f"merge 행수 불일치: {len(df)} vs {len(df_score)}")

    # ── PaDiM 선택 z 정의 ────────────────────────────────────────────────────
    print("\n[3/6] PaDiM p99 selected z-slices 정의 (per patient)")
    # selector: first_stage_score > threshold_p99 (PaDiM calibrated threshold, NOT GT label)
    # threshold_p99 = 15.472385 (global fixed, normal_val_threshold.json 기반)
    # 모든 152환자 동일값 확인 → calibrated_detector=True, per_patient_percentile=False
    actual_threshold_p99_unique = df["threshold_p99"].unique()
    if len(actual_threshold_p99_unique) != 1:
        log_error(f"threshold_p99가 환자별로 다름: {actual_threshold_p99_unique}")
    SELECTOR_META["threshold_p99_value_verified"] = float(actual_threshold_p99_unique[0]) if len(actual_threshold_p99_unique) == 1 else list(actual_threshold_p99_unique)

    df["padim_p99_exceed"] = df["first_stage_score"] > df["threshold_p99"]
    # label 컬럼은 selector에서 접근하지 않음 (GUARDRAILS["label_used_as_selector"] = False 유지)

    # A1_direct selector: per patient, z-slices where padim_p99_exceed == True
    selected_z_per_patient = (
        df[df["padim_p99_exceed"]]
        .groupby("patient_id")["local_z"]
        .apply(set)
        .to_dict()
    )
    n_patients_with_hot_z = len(selected_z_per_patient)
    total_p99_slices = sum(len(v) for v in selected_z_per_patient.values())
    print(f"  patients with p99-hot z-slices: {n_patients_with_hot_z}/{len(score_patients)}")
    print(f"  total p99-hot (patient, z) pairs: {total_p99_slices:,}")
    if n_patients_with_hot_z < len(score_patients):
        missing_hot = sorted(score_patients - set(selected_z_per_patient.keys()))
        log_error(f"p99 hot z-slice 없는 환자: {missing_hot}")

    # ── A0 기준선 ────────────────────────────────────────────────────────────
    print("\n[4/6] Variant 계산")
    df_a0 = df.copy()
    print(f"  A0_all: {len(df_a0):,} candidates")

    # ── Variant 필터 함수 ────────────────────────────────────────────────────
    def filter_by_z_expansion(df_src, expansion: int):
        """per patient, PaDiM p99 선택 z ± expansion 으로 필터"""
        keep_masks = []
        for pid, group in df_src.groupby("patient_id"):
            hot_z = selected_z_per_patient.get(pid, set())
            expanded_z = set()
            for z in hot_z:
                for dz in range(-expansion, expansion + 1):
                    expanded_z.add(z + dz)
            mask = group["local_z"].isin(expanded_z)
            keep_masks.append(mask)
        if not keep_masks:
            return df_src.iloc[:0].copy()
        combined = pd.concat(keep_masks)
        return df_src[combined].copy()

    # A1_direct: 확장 없음 (expansion=0)
    df_a1 = filter_by_z_expansion(df_a0, 0)
    print(f"  A1_direct: {len(df_a1):,}")

    df_a2 = filter_by_z_expansion(df_a0, 1)
    print(f"  A2_z_pm1: {len(df_a2):,}")

    df_a3 = filter_by_z_expansion(df_a0, 2)
    print(f"  A3_z_pm2: {len(df_a3):,}")

    df_a4 = filter_by_z_expansion(df_a0, 3)
    print(f"  A4_z_pm3: {len(df_a4):,}")

    df_a5 = filter_by_z_expansion(df_a0, 5)
    print(f"  A5_z_pm5: {len(df_a5):,}")

    # BLOCKER 2: label leakage 판정 — 컬럼 기반 (positive rate 기반 아님)
    # selector 함수 filter_by_z_expansion이 사용한 컬럼: patient_id, local_z, first_stage_score, threshold_p99
    # forbidden 교집합은 main() 진입 시 이미 체크했으므로 여기서는 참고용 로그만 남김
    a1_label_leak_flag = bool(GUARDRAILS["forbidden_selector_columns_intersection"])
    a1_pos_rate = (df_a1["label"] == "positive").mean()
    a0_pos_rate = (df_a0["label"] == "positive").mean()
    # positive rate는 참고값으로만 기록 (판정 기준 아님)
    print(f"  A1 positive rate (참고): {a1_pos_rate:.3f} (A0: {a0_pos_rate:.3f})")
    print(f"  label_used_as_selector (컬럼 기반 판정): {GUARDRAILS['label_used_as_selector']}")

    # ── Metric 계산 ───────────────────────────────────────────────────────────
    print("\n[5/6] Metric 계산")
    variants = [
        ("A0_all", df_a0),
        ("A1_direct", df_a1),
        ("A2_z_pm1", df_a2),
        ("A3_z_pm2", df_a3),
        ("A4_z_pm3", df_a4),
        ("A5_z_pm5", df_a5),
    ]

    summary_rows = []
    problem_patient_rows = []
    patient_summary_rows = []

    for vname, df_v in variants:
        print(f"  computing {vname}...", end=" ")
        m = compute_variant_metrics(df_v, df_a0, vname)
        summary_rows.append(m)
        print(f"candidates={m['n_variant']:,}, lesion_slice_cov={m['lesion_slice_coverage']}, hn_supp={m['hard_negative_suppression']}")

        prob = get_problem_patients(df_v, df_a0, vname)
        problem_patient_rows.append(prob)

        # patient-level summary
        for pid in df_a0["patient_id"].unique():
            df_pid = df_v[df_v["patient_id"] == pid]
            pos_a0_pid = df_a0[(df_a0["patient_id"] == pid) & (df_a0["label"] == "positive")]
            pos_v_pid = df_pid[df_pid["label"] == "positive"]
            patient_summary_rows.append({
                "variant": vname,
                "patient_id": pid,
                "n_candidates": len(df_pid),
                "n_positive": len(pos_v_pid),
                "n_hard_negative": len(df_pid[df_pid["label"] == "hard_negative"]),
                "positive_a0": len(pos_a0_pid),
                "pos_slice_cov": round(len(set(pos_v_pid["local_z"])) / len(set(pos_a0_pid["local_z"])), 4) if len(pos_a0_pid) > 0 else None,
                "max_rd_d1s_score": round(float(df_pid["rd_d1s_medi3ch_rd4ad_score"].max()), 6) if len(df_pid) > 0 else None,
                "top3_mean_rd_d1s_score": round(float(df_pid["rd_d1s_medi3ch_rd4ad_score"].nlargest(3).mean()), 6) if len(df_pid) > 0 else None,
            })

    # ── 결과 저장 ─────────────────────────────────────────────────────────────
    print("\n[6/6] 결과 저장")

    # variant summary CSV
    df_variant_summary = pd.DataFrame(summary_rows)
    # pat_all_sup_list는 JSON으로 직렬화
    df_variant_summary["pat_all_sup_list"] = df_variant_summary["pat_all_sup_list"].apply(json.dumps)
    variant_csv = MANIFEST_DIR / "rd_d1s_candidate_gated_variant_summary.csv"
    df_variant_summary.drop(columns=["pat_all_sup_list"]).to_csv(variant_csv, index=False)
    print(f"  saved: {variant_csv}")

    # patient summary CSV
    df_patient_summary = pd.DataFrame(patient_summary_rows)
    patient_csv = MANIFEST_DIR / "rd_d1s_candidate_gated_patient_summary.csv"
    df_patient_summary.to_csv(patient_csv, index=False)
    print(f"  saved: {patient_csv}")

    # problem patients CSV
    df_problems = pd.concat(problem_patient_rows, ignore_index=True)
    problem_csv = MANIFEST_DIR / "rd_d1s_candidate_gated_problem_patients.csv"
    df_problems.to_csv(problem_csv, index=False)
    print(f"  saved: {problem_csv}")

    # errors CSV
    errors_csv = LOG_DIR / "errors.csv"
    pd.DataFrame(errors if errors else [{"timestamp": datetime.now().isoformat(), "error": "none"}]).to_csv(errors_csv, index=False)
    print(f"  saved: {errors_csv}")

    # ── 판정 ─────────────────────────────────────────────────────────────────
    m_a3 = next(m for m in summary_rows if m["variant"] == "A3_z_pm2")
    m_a4 = next(m for m in summary_rows if m["variant"] == "A4_z_pm3")

    a3_pass = (
        (m_a3["lesion_slice_coverage"] or 0) >= 0.97
        and (m_a3["lesion_patient_coverage"] or 0) >= 1.0
        and (m_a3["candidate_reduction_rate"] or 0) > 0
        and m_a3["pat_all_sup_count"] == 0
        and not GUARDRAILS["stage2_holdout_accessed"]
        and not GUARDRAILS["label_used_as_selector"]
        and not a1_label_leak_flag
    )
    a4_fallback = not a3_pass and (
        (m_a4["lesion_slice_coverage"] or 0) >= 0.97
        and (m_a4["lesion_patient_coverage"] or 0) >= 1.0
    )

    if a3_pass:
        recommended_variant = "A3_z_pm2"
        verdict = "PASS"
    elif a4_fallback:
        recommended_variant = "A4_z_pm3"
        verdict = "PARTIAL_PASS"
    else:
        recommended_variant = "A5_z_pm5 (upper-bound only)"
        verdict = "FAIL"

    # A1 label leakage annotation
    a1_status = "evaluation_upper_bound" if a1_label_leak_flag else "valid_selector"

    # ── Summary JSON ─────────────────────────────────────────────────────────
    summary_json = {
        "analysis": "rd_d1s_candidate_gated_posthoc_v1",
        "timestamp": datetime.now().isoformat(),
        "guardrails": GUARDRAILS,
        "selector_meta": SELECTOR_META,
        "patient_count_note": patient_count_note,
        "verdict": verdict,
        "recommended_variant": recommended_variant,
        "a1_status": a1_status,
        "a1_label_leak_flag": a1_label_leak_flag,
        "a1_label_leak_basis": "column_intersection_check (forbidden_selector_columns_intersection)",
        "a1_positive_rate_reference": round(float(a1_pos_rate), 4),
        "a0_positive_rate_reference": round(float(a0_pos_rate), 4),
        "n_errors": len(errors),
        "variant_summary": summary_rows,
        "next_step": (
            "RD-D1s 96×96 center-region score aggregation preflight"
            if verdict == "PASS" else
            "coverage 개선 필요 후 재판정"
        ),
    }
    summary_path = REPORT_DIR / "rd_d1s_candidate_gated_posthoc_summary.json"
    summary_path.write_text(json.dumps(summary_json, indent=2, ensure_ascii=False))
    print(f"  saved: {summary_path}")

    # ── Markdown report ───────────────────────────────────────────────────────
    m_a0 = next(m for m in summary_rows if m["variant"] == "A0_all")
    m_a1 = next(m for m in summary_rows if m["variant"] == "A1_direct")
    m_a2 = next(m for m in summary_rows if m["variant"] == "A2_z_pm1")
    m_a5 = next(m for m in summary_rows if m["variant"] == "A5_z_pm5")

    report_lines = [
        "# RD-D1s Candidate-Gated Post-hoc Analysis Report",
        f"\n생성일시: {datetime.now().isoformat()}",
        "\n## Guardrails",
        "| 항목 | 값 |",
        "|------|-----|",
    ]
    for k, v in GUARDRAILS.items():
        report_lines.append(f"| {k} | {v} |")

    report_lines += [
        "\n## 판정",
        f"**{verdict}**  ",
        f"추천 variant: **{recommended_variant}**  ",
        f"A1_direct 상태: **{a1_status}**  ",
        f"label_used_as_selector: {GUARDRAILS['label_used_as_selector']}",
        f"forbidden_selector_columns_intersection: {GUARDRAILS['forbidden_selector_columns_intersection']}",
        "\n## Selector 명세 (WARNING 1)",
        f"- selector_type: `{SELECTOR_META['selector_type']}`",
        f"- threshold_scope: `{SELECTOR_META['threshold_scope']}`",
        f"- threshold_p99_value: `{SELECTOR_META['threshold_p99_value']}`",
        f"- calibrated_detector: `{SELECTOR_META['calibrated_detector']}`",
        f"- candidate_scheduler: `{SELECTOR_META['candidate_scheduler']}`",
        f"- 비고: {SELECTOR_META['note']}",
        "\n## 환자 수 불일치 (WARNING 2)",
        f"- stage1_dev_total_expected: {patient_count_note['stage1_dev_total_expected']}",
        f"- candidate_score_patients: {patient_count_note['candidate_score_patients']}",
        f"- missing_patients: {patient_count_note['missing_from_candidate_score']}",
        f"- 이유: {patient_count_note['missing_reason']}",
        "\n## Variant Summary",
        "| Variant | Candidates | Reduction | LesionSliceCov | PatientCov | PosRetention | HN_Supp | PatAllSup | AUROC | AUPRC |",
        "|---------|-----------|-----------|----------------|------------|--------------|---------|-----------|-------|-------|",
    ]
    for m in summary_rows:
        report_lines.append(
            f"| {m['variant']} "
            f"| {m['n_variant']:,} "
            f"| {m['candidate_reduction_rate']:.1%} "
            f"| {m['lesion_slice_coverage']:.4f} "
            f"| {m['lesion_patient_coverage']:.4f} "
            f"| {m['positive_crop_retention']:.4f} "
            f"| {m['hard_negative_suppression']:.4f} "
            f"| {m['pat_all_sup_count']} "
            f"| {m['auroc']} "
            f"| {m['auprc']} |"
        )

    report_lines += [
        "\n## 판정 근거 (A3_z_pm2)",
        f"- lesion_slice_coverage: {m_a3['lesion_slice_coverage']:.4f} (기준 ≥0.97: {'O' if (m_a3['lesion_slice_coverage'] or 0)>=0.97 else 'X'})",
        f"- lesion_patient_coverage: {m_a3['lesion_patient_coverage']:.4f} (기준 1.0: {'O' if (m_a3['lesion_patient_coverage'] or 0)>=1.0 else 'X'})",
        f"- candidate_reduction_rate: {m_a3['candidate_reduction_rate']:.4f} (기준 >0: {'O' if (m_a3['candidate_reduction_rate'] or 0)>0 else 'X'})",
        f"- pat_all_sup_count: {m_a3['pat_all_sup_count']} (기준 0: {'O' if m_a3['pat_all_sup_count']==0 else 'X'})",
        f"- stage2_holdout_accessed: {GUARDRAILS['stage2_holdout_accessed']} (기준 False: {'O' if not GUARDRAILS['stage2_holdout_accessed'] else 'X'})",
        f"- label_used_as_selector: {GUARDRAILS['label_used_as_selector']} (기준 False: {'O' if not GUARDRAILS['label_used_as_selector'] else 'X'})",
        f"- A1 label_leak_flag: {a1_label_leak_flag} (기준 False: {'O' if not a1_label_leak_flag else 'X'})",
        "\n## Coverage <95% 환자 수 (A3_z_pm2)",
    ]
    a3_problems = df_problems[(df_problems["variant"] == "A3_z_pm2") & df_problems["coverage_lt_95"]]
    report_lines.append(f"- coverage<95%: {len(a3_problems)} 환자")
    a3_80 = df_problems[(df_problems["variant"] == "A3_z_pm2") & df_problems["coverage_lt_80"]]
    report_lines.append(f"- coverage<80%: {len(a3_80)} 환자")
    a3_50 = df_problems[(df_problems["variant"] == "A3_z_pm2") & df_problems["coverage_lt_50"]]
    report_lines.append(f"- coverage<50%: {len(a3_50)} 환자")

    report_lines += [
        "\n## 해석 주의사항",
        "- selected subset AUROC는 A0_all AUROC와 직접 비교 불가 (평가 분포 변화)",
        "- 주요 판단 기준: lesion/slice/patient coverage 유지 + HN/FP 후보 감소",
        "- A5_z_pm5는 upper-bound 참고용이며 기본 채택 금지",
        "\n## 오류 목록",
        f"- 오류 수: {len(errors)}",
    ]
    for e in errors:
        report_lines.append(f"  - [{e['timestamp']}] {e['error']}")

    report_lines += [
        "\n## 다음 단계",
        summary_json["next_step"],
    ]

    report_path = REPORT_DIR / "rd_d1s_candidate_gated_posthoc_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  saved: {report_path}")

    # ── DONE.json ─────────────────────────────────────────────────────────────
    done = {
        "status": "DONE" if len(errors) == 0 else "DONE_WITH_ERRORS",
        "timestamp": datetime.now().isoformat(),
        "verdict": verdict,
        "recommended_variant": recommended_variant,
        "guardrails": GUARDRAILS,
        "n_errors": len(errors),
    }
    (OUT_ROOT / "DONE.json").write_text(json.dumps(done, indent=2, ensure_ascii=False))
    print(f"\n  saved: {OUT_ROOT / 'DONE.json'}")

    # ── 최종 출력 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"판정: {verdict}")
    print(f"추천 variant: {recommended_variant}")
    print(f"A1_direct 상태: {a1_status}")
    print(f"\n핵심 결과:")
    print(f"  A0 전체 후보: {m_a0['n_variant']:,}")
    for m in summary_rows[1:]:
        print(
            f"  {m['variant']}: {m['n_variant']:,} candidates "
            f"(reduction={m['candidate_reduction_rate']:.1%}, "
            f"lesion_slice_cov={m['lesion_slice_coverage']:.4f}, "
            f"hn_supp={m['hard_negative_suppression']:.4f})"
        )
    print(f"\n  label leakage: {GUARDRAILS['label_used_as_selector']}")
    print(f"  errors: {len(errors)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
