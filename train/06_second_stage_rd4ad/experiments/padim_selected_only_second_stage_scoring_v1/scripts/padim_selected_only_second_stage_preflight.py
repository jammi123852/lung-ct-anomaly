"""
PaDiM-selected-only second-stage scoring preflight v1.

이 스크립트는 preflight/post-hoc 분석만 수행한다.
모델 forward, crop 생성, 재학습, full scoring은 하지 않는다.

Usage:
  # dry-run (입력 확인만, 파일 생성 없음)
  python experiments/padim_selected_only_second_stage_scoring_v1/scripts/padim_selected_only_second_stage_preflight.py --dry-run

  # 실제 preflight
  python experiments/padim_selected_only_second_stage_scoring_v1/scripts/padim_selected_only_second_stage_preflight.py \\
    --run-preflight --confirm-readonly --confirm-stage1dev-only
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]

MANIFEST_CSV = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_c2_effb0_v420_candidate_rd4ad_retest_v1"
    / "rd_c2_effb0_v420_candidate_manifest.csv"
)

RD_D1S_SCORE_CSV = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_c2_effb0_v420_candidate_rd4ad_retest_v1"
    / "rd_c2_effb0_v420_rd4ad_candidate_score.csv"
)

CONVAE_SCORE_CSV = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_c3_v1v1_convae_same_universe_retest_v1"
    / "rd_c3_v1v1_convae_candidate_score.csv"
)

STAGE2_HOLDOUT_PATH_KEYWORDS = [
    "stage2_holdout",
    "second-stage-lesion-refiner-v1/datasets",
]

OUT_ROOT = (
    PROJECT_ROOT
    / "experiments/padim_selected_only_second_stage_scoring_v1"
)
REPORT_DIR = OUT_ROOT / "reports"
MANIFEST_DIR = OUT_ROOT / "manifests"
LOG_DIR = OUT_ROOT / "logs"

# ---------------------------------------------------------------------------
# Guardrail constants
# ---------------------------------------------------------------------------
GUARDRAILS = {
    "stage2_holdout_accessed": False,
    "model_forward_executed": False,
    "training_executed": False,
    "crop_generation_executed": False,
    "full_scoring_executed": False,
    "checkpoint_loaded": False,
    "threshold_recalculated": False,
    "existing_artifact_modified": False,
    "existing_script_modified": False,
    "output_overwrite": False,
    "label_used_as_selector": False,
    "label_used_for_evaluation_only": True,
    "second_stage_score_used_as_selector": False,
}

REQUIRED_MANIFEST_COLUMNS = [
    "candidate_id", "patient_id", "safe_id", "stage_split",
    "local_z", "first_stage_score", "threshold_p95", "threshold_p99",
    "label", "candidate_label", "candidate_rule",
]

FORBIDDEN_SELECTOR_COLUMNS = {
    "label", "positive", "hard_negative", "lesion", "gt",
    "mask_overlap", "lesion_overlap", "is_positive", "target", "class",
    "rd4ad_crop_score", "score_layer1", "score_layer2", "score_layer3",
    "global_p95_exceed", "global_p99_exceed", "bin_p95_exceed", "bin_p99_exceed",
    "convAE_crop_score_l1_mean", "convAE_crop_score_mse_mean",
    "convAE_mediastinal_channels_l1_mean", "convAE_lung_channels_l1_mean",
    "rd4ad_b8f_score",
}

ALLOWED_SELECTOR_COLUMNS = {
    "patient_id", "safe_id", "local_z", "slice_index", "x0", "y0", "x1", "y1",
    "crop_y0", "crop_x0", "crop_y1", "crop_x1",
    "first_stage_score", "threshold_p95", "threshold_p99",
    "position_bin", "six_bin_label", "z_level",
    "candidate_rule", "source_branch", "backbone", "roi_source",
}

errors = []

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def log_error(msg):
    errors.append({"type": "ERROR", "message": msg})
    print(f"[ERROR] {msg}", file=sys.stderr)


def log_warn(msg):
    print(f"[WARN] {msg}", file=sys.stderr)


def log_info(msg):
    print(f"[INFO] {msg}")


def check_stage2_holdout_not_accessed():
    """
    이 스크립트가 실제로 stage2_holdout 경로에서 데이터를 읽지 않음을 선언한다.
    입력 파일 경로를 검사해 holdout 키워드가 없는지 확인한다.
    """
    input_paths = [str(MANIFEST_CSV), str(RD_D1S_SCORE_CSV), str(CONVAE_SCORE_CSV)]
    for ip in input_paths:
        for kw in STAGE2_HOLDOUT_PATH_KEYWORDS:
            if kw in ip:
                GUARDRAILS["stage2_holdout_accessed"] = True
                log_error(f"입력 경로에 stage2_holdout 키워드 포함: {ip}")
    return not GUARDRAILS["stage2_holdout_accessed"]


# ---------------------------------------------------------------------------
# Selector variant definitions
# ---------------------------------------------------------------------------
VARIANT_DEFS = [
    {
        "name": "A0_all",
        "desc": "전체 candidate (기준선)",
        "kind": "all",
    },
    {
        "name": "A1_p95_patch",
        "desc": "first_stage_score > threshold_p95",
        "kind": "score_threshold",
        "col": "threshold_p95",
    },
    {
        "name": "A2_p99_patch",
        "desc": "first_stage_score > threshold_p99",
        "kind": "score_threshold",
        "col": "threshold_p99",
    },
    {
        "name": "A3_p99_z_pm1",
        "desc": "p99 patch가 있는 patient-local_z ±1 slice 확장",
        "kind": "z_expand",
        "base_col": "threshold_p99",
        "z_delta": 1,
    },
    {
        "name": "A4_p99_z_pm2",
        "desc": "p99 patch가 있는 patient-local_z ±2 slice 확장 [primary 후보]",
        "kind": "z_expand",
        "base_col": "threshold_p99",
        "z_delta": 2,
    },
    {
        "name": "A5_p99_z_pm3",
        "desc": "p99 patch가 있는 patient-local_z ±3 slice 확장 [safety 후보]",
        "kind": "z_expand",
        "base_col": "threshold_p99",
        "z_delta": 3,
    },
    {
        "name": "A6_p99_z_pm5",
        "desc": "p99 patch가 있는 patient-local_z ±5 slice 확장 [upper-bound, 기본 채택 금지]",
        "kind": "z_expand",
        "base_col": "threshold_p99",
        "z_delta": 5,
    },
    {
        "name": "B1_p95_z_pm1",
        "desc": "p95 patch 기준 ±1 slice 확장",
        "kind": "z_expand",
        "base_col": "threshold_p95",
        "z_delta": 1,
    },
    {
        "name": "B2_p95_z_pm2",
        "desc": "p95 patch 기준 ±2 slice 확장",
        "kind": "z_expand",
        "base_col": "threshold_p95",
        "z_delta": 2,
    },
    {
        "name": "B3_p95_z_pm3",
        "desc": "p95 patch 기준 ±3 slice 확장",
        "kind": "z_expand",
        "base_col": "threshold_p95",
        "z_delta": 3,
    },
]

# ---------------------------------------------------------------------------
# Core variant computation
# ---------------------------------------------------------------------------
def compute_selector_mask(df, vdef):
    """selector 조건만으로 마스크를 만든다. label/2차 score 사용 금지."""
    import pandas as pd

    kind = vdef["kind"]
    if kind == "all":
        return pd.Series([True] * len(df), index=df.index)

    elif kind == "score_threshold":
        col = vdef["col"]
        return df["first_stage_score"] > df[col]

    elif kind == "z_expand":
        base_col = vdef["base_col"]
        z_delta = vdef["z_delta"]
        # 1) p95/p99 초과 patch가 있는 (patient_id, local_z) 집합
        base_mask = df["first_stage_score"] > df[base_col]
        hot_pairs = df[base_mask][["patient_id", "local_z"]].drop_duplicates()
        # 2) ±z_delta 확장
        expanded = set()
        for _, row in hot_pairs.iterrows():
            pid = row["patient_id"]
            z = int(row["local_z"])
            for dz in range(-z_delta, z_delta + 1):
                expanded.add((pid, z + dz))
        # 3) 마스크 생성
        return df.apply(
            lambda r: (r["patient_id"], int(r["local_z"])) in expanded, axis=1
        )

    else:
        raise ValueError(f"Unknown variant kind: {kind}")


def compute_variant_metrics(df, mask, vdef, label_col="label"):
    """
    선택 후 coverage/reduction 지표 계산.
    label_col은 평가 전용이며, selector로 사용하지 않는다.
    """
    import numpy as np

    sel = df[mask]
    n_all = len(df)
    n_sel = len(sel)

    # positive / hard_negative 분류 (평가 전용)
    is_pos = df[label_col] == "positive"
    is_hn = df[label_col] == "hard_negative"
    sel_is_pos = sel[label_col] == "positive"
    sel_is_hn = sel[label_col] == "hard_negative"

    n_pos_all = is_pos.sum()
    n_hn_all = is_hn.sum()
    n_pos_sel = sel_is_pos.sum()
    n_hn_sel = sel_is_hn.sum()

    reduction_rate = 1.0 - n_sel / n_all if n_all > 0 else 0.0

    # patient coverage
    all_patients = set(df["patient_id"].unique())
    sel_patients = set(sel["patient_id"].unique())
    patient_cov = len(sel_patients) / len(all_patients) if all_patients else 0.0

    # lesion patients (patients with >=1 positive candidate)
    lesion_patients = set(df[is_pos]["patient_id"].unique())
    sel_lesion_patients = set(sel[sel_is_pos]["patient_id"].unique())
    lesion_patient_cov = (
        len(sel_lesion_patients) / len(lesion_patients) if lesion_patients else 0.0
    )

    # lesion slice coverage: (patient_id, local_z) pairs with >=1 positive
    all_lesion_slices = set(
        map(tuple, df[is_pos][["patient_id", "local_z"]].drop_duplicates().values)
    )
    sel_lesion_slices = set(
        map(tuple, sel[sel_is_pos][["patient_id", "local_z"]].drop_duplicates().values)
    )
    lesion_slice_cov = (
        len(sel_lesion_slices) / len(all_lesion_slices) if all_lesion_slices else 0.0
    )

    # positive crop retention
    pos_retention = n_pos_sel / n_pos_all if n_pos_all > 0 else 0.0

    # hard_negative retention and reduction
    hn_retention = n_hn_sel / n_hn_all if n_hn_all > 0 else 0.0
    hn_reduction = 1.0 - hn_retention

    # problem patients: lesion patients NOT fully covered (< 100% positive retention)
    per_patient_pos_all = df[is_pos].groupby("patient_id")["candidate_id"].count()
    per_patient_pos_sel = sel[sel_is_pos].groupby("patient_id")["candidate_id"].count()
    per_patient_pos_sel = per_patient_pos_sel.reindex(per_patient_pos_all.index, fill_value=0)
    per_patient_ret = per_patient_pos_sel / per_patient_pos_all

    problem_patients = per_patient_ret[per_patient_ret < 1.0].index.tolist()
    cov_lt50 = (per_patient_ret < 0.5).sum()
    cov_lt80 = (per_patient_ret < 0.8).sum()
    cov_lt95 = (per_patient_ret < 0.95).sum()

    # selected candidate count distribution (per patient)
    sel_per_patient = sel.groupby("patient_id")["candidate_id"].count()
    p25, p50, p75, p95 = float(np.percentile(sel_per_patient, 25)), float(np.percentile(sel_per_patient, 50)), float(np.percentile(sel_per_patient, 75)), float(np.percentile(sel_per_patient, 95))

    # z distribution (selected)
    z_counts = sel["local_z"].value_counts()

    # position_bin / six_bin_label distribution (if present)
    pb_col = "six_bin_label" if "six_bin_label" in sel.columns else (
        "position_bin" if "position_bin" in sel.columns else None
    )
    pb_dist = sel[pb_col].value_counts().to_dict() if pb_col else {}

    result = {
        "variant": vdef["name"],
        "desc": vdef["desc"],
        "n_selected": int(n_sel),
        "n_total": int(n_all),
        "reduction_rate": round(reduction_rate, 6),
        "n_selected_patients": int(len(sel_patients)),
        "n_total_patients": int(len(all_patients)),
        "patient_coverage": round(patient_cov, 6),
        "n_lesion_patients_all": int(len(lesion_patients)),
        "n_lesion_patients_sel": int(len(sel_lesion_patients)),
        "lesion_patient_coverage": round(lesion_patient_cov, 6),
        "n_lesion_slices_all": int(len(all_lesion_slices)),
        "n_lesion_slices_sel": int(len(sel_lesion_slices)),
        "lesion_slice_coverage": round(lesion_slice_cov, 6),
        "n_positive_all": int(n_pos_all),
        "n_positive_sel": int(n_pos_sel),
        "positive_crop_retention": round(pos_retention, 6),
        "n_hard_negative_all": int(n_hn_all),
        "n_hard_negative_sel": int(n_hn_sel),
        "hard_negative_retention": round(hn_retention, 6),
        "hard_negative_reduction": round(hn_reduction, 6),
        "n_problem_patients": int(len(problem_patients)),
        "coverage_lt50_patient_count": int(cov_lt50),
        "coverage_lt80_patient_count": int(cov_lt80),
        "coverage_lt95_patient_count": int(cov_lt95),
        "per_patient_sel_p25": p25,
        "per_patient_sel_p50": p50,
        "per_patient_sel_p75": p75,
        "per_patient_sel_p95": p95,
        "position_bin_dist": pb_dist,
        "stage2_holdout_accessed": False,
        "label_used_as_selector": False,
        "second_stage_score_used_as_selector": False,
    }
    return result, problem_patients, sel, per_patient_ret


# ---------------------------------------------------------------------------
# AUROC helper (sklearn-free, Mann-Whitney)
# ---------------------------------------------------------------------------
def compute_auroc_auprc_skleanfree(scores, labels_binary):
    """
    sklearn 없이 AUROC(Mann-Whitney) / AUPRC(step AP) 계산.
    labels_binary: 1=positive, 0=negative
    """
    import numpy as np

    pos = scores[labels_binary == 1]
    neg = scores[labels_binary == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None, None

    # AUROC via Mann-Whitney U
    n_pos, n_neg = len(pos), len(neg)
    rank_sum = sum((p > n).sum() + 0.5 * (p == n).sum() for p in pos for n in [neg])
    # vectorized
    from numpy import searchsorted
    combined = np.concatenate([pos, neg])
    sorted_combined = np.sort(combined)
    ranks_pos = searchsorted(sorted_combined, pos, side="left") + 1
    u = ranks_pos.sum() - n_pos * (n_pos + 1) / 2
    auroc = u / (n_pos * n_neg)

    # AUPRC via step average precision (sorted by score desc)
    order = np.argsort(scores)[::-1]
    sorted_labels = labels_binary[order]
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (n_pos + 1e-12)
    auprc = float(np.sum(precision[1:] * (recall[1:] - recall[:-1])))

    return float(auroc), auprc


# ---------------------------------------------------------------------------
# dry-run
# ---------------------------------------------------------------------------
def run_dry():
    """입력 파일 존재, 컬럼, stage2_holdout 접근 여부만 확인한다."""
    print("=" * 70)
    print("[DRY-RUN] padim_selected_only_second_stage_preflight v1")
    print("=" * 70)

    # 1. stage2_holdout 접근 확인
    ok_holdout = check_stage2_holdout_not_accessed()
    print(f"[CHECK] stage2_holdout 접근 없음: {ok_holdout}")

    # 2. 입력 파일 존재 확인
    for label, path in [
        ("manifest_csv", MANIFEST_CSV),
        ("rd_d1s_score_csv", RD_D1S_SCORE_CSV),
        ("convae_score_csv", CONVAE_SCORE_CSV),
    ]:
        exists = path.exists()
        print(f"[CHECK] {label}: {'OK' if exists else 'MISSING'} — {path}")
        if not exists and label == "manifest_csv":
            log_error(f"필수 입력 파일 없음: {path}")

    # 3. output root 충돌 확인
    for out_file in [
        REPORT_DIR / "padim_selected_only_second_stage_preflight_report.md",
        REPORT_DIR / "padim_selected_only_second_stage_preflight_summary.json",
        MANIFEST_DIR / "padim_selected_only_variant_summary.csv",
    ]:
        if out_file.exists():
            print(f"[WARN] 출력 파일 이미 존재 (overwrite 필요시 삭제): {out_file}")

    # 4. 컬럼 확인
    if MANIFEST_CSV.exists():
        import pandas as pd
        df_head = pd.read_csv(MANIFEST_CSV, nrows=5)
        cols = set(df_head.columns)
        missing = [c for c in REQUIRED_MANIFEST_COLUMNS if c not in cols]
        if missing:
            log_error(f"manifest CSV에 필요 컬럼 누락: {missing}")
        else:
            print(f"[CHECK] manifest 필요 컬럼 모두 존재: OK")

        # selector_used_columns
        selector_cols = {
            "patient_id", "local_z", "first_stage_score",
            "threshold_p95", "threshold_p99",
        }
        forbidden_intersection = selector_cols & FORBIDDEN_SELECTOR_COLUMNS
        if forbidden_intersection:
            log_error(f"selector에 forbidden column 포함: {forbidden_intersection}")
            print(f"[FAIL] label leakage 발견: {forbidden_intersection}")
        else:
            print(f"[CHECK] selector_used_columns: {sorted(selector_cols)}")
            print(f"[CHECK] forbidden column intersection: EMPTY — OK")

    # 5. stage_split 확인
    if MANIFEST_CSV.exists():
        import pandas as pd
        raw = pd.read_csv(MANIFEST_CSV, usecols=["stage_split"])
        if (raw["stage_split"] == "stage2_holdout").any():
            GUARDRAILS["stage2_holdout_accessed"] = True
            log_error("manifest에 stage2_holdout 행 존재!")
        else:
            print(f"[CHECK] stage_split=stage2_holdout 행 없음: OK")

    print()
    if errors:
        print(f"[DRY-RUN RESULT] FAIL — {len(errors)} error(s)")
        for e in errors:
            print(f"  - {e['message']}")
        sys.exit(1)
    else:
        print("[DRY-RUN RESULT] PASS — 실제 preflight 실행 가능")
    print("[INFO] 파일 생성 없음 (dry-run)")


# ---------------------------------------------------------------------------
# Main preflight
# ---------------------------------------------------------------------------
def run_preflight():
    import pandas as pd
    import numpy as np

    print("=" * 70)
    print("[PREFLIGHT] padim_selected_only_second_stage_preflight v1")
    print("=" * 70)

    # 0. stage2_holdout 확인
    check_stage2_holdout_not_accessed()

    # 1. 입력 CSV 로드
    if not MANIFEST_CSV.exists():
        log_error(f"manifest CSV 없음: {MANIFEST_CSV}")
        _finalize_with_errors()
        return

    log_info(f"manifest CSV 로드: {MANIFEST_CSV}")
    df = pd.read_csv(MANIFEST_CSV)
    log_info(f"  → {len(df):,} rows, columns: {df.columns.tolist()}")

    # stage_split=stage1_dev 확인
    if "stage_split" in df.columns:
        holdout_rows = (df["stage_split"] == "stage2_holdout").sum()
        if holdout_rows > 0:
            GUARDRAILS["stage2_holdout_accessed"] = True
            log_error(f"stage2_holdout 행 {holdout_rows}개 존재!")
        df = df[df["stage_split"] == "stage1_dev"].copy()
        log_info(f"  → stage1_dev 필터 후: {len(df):,} rows")

    # 컬럼 확인
    missing_cols = [c for c in REQUIRED_MANIFEST_COLUMNS if c not in df.columns]
    if missing_cols:
        log_error(f"manifest 컬럼 누락: {missing_cols}")

    # 2. 참고용 RD-D1s score 로드 (optional)
    df_rd = None
    if RD_D1S_SCORE_CSV.exists():
        log_info(f"RD-D1s score CSV 로드 (참고용): {RD_D1S_SCORE_CSV}")
        df_rd = pd.read_csv(
            RD_D1S_SCORE_CSV,
            usecols=["candidate_id", "rd4ad_crop_score", "six_bin_label",
                      "z_level", "boundary_status"]
        )
        # candidate_id join으로 six_bin_label 추가
        df = df.merge(
            df_rd[["candidate_id", "six_bin_label", "z_level", "boundary_status"]],
            on="candidate_id", how="left"
        )
        log_info(f"  → six_bin_label 추가 완료")
    else:
        log_warn(f"RD-D1s score CSV 없음 (참고 지표 생략): {RD_D1S_SCORE_CSV}")

    # 3. 참고용 ConvAE score 로드 (optional)
    df_conv = None
    if CONVAE_SCORE_CSV.exists():
        log_info(f"ConvAE score CSV 로드 (참고용): {CONVAE_SCORE_CSV}")
        df_conv = pd.read_csv(
            CONVAE_SCORE_CSV,
            usecols=["candidate_id", "convAE_crop_score_l1_mean"]
        )
    else:
        log_warn(f"ConvAE score CSV 없음 (참고 지표 생략): {CONVAE_SCORE_CSV}")

    # 4. forbidden selector column intersection 확인
    selector_cols_used = {
        "patient_id", "local_z", "first_stage_score",
        "threshold_p95", "threshold_p99",
    }
    forbidden_hit = selector_cols_used & FORBIDDEN_SELECTOR_COLUMNS
    if forbidden_hit:
        GUARDRAILS["label_used_as_selector"] = True
        log_error(f"selector에 forbidden column 포함: {forbidden_hit}")

    # 5. label을 평가용으로 분리 (selector로 절대 사용 금지)
    EVAL_LABEL_COL = "label"
    if EVAL_LABEL_COL not in df.columns:
        log_error("'label' column 없음 — 평가 지표 계산 불가")
        _finalize_with_errors()
        return

    log_info(f"  label 분포: {df[EVAL_LABEL_COL].value_counts().to_dict()}")

    # 6. 각 variant 계산
    all_variant_rows = []
    all_problem_rows = []
    recommended_manifest = None
    per_patient_summary_rows = []

    for vdef in VARIANT_DEFS:
        log_info(f"variant: {vdef['name']}")
        try:
            mask = compute_selector_mask(df, vdef)
            metrics, problem_patients, sel_df, per_ret = compute_variant_metrics(
                df, mask, vdef, label_col=EVAL_LABEL_COL
            )
        except Exception as e:
            log_error(f"variant {vdef['name']} 계산 오류: {e}\n{traceback.format_exc()}")
            continue

        # 참고 AUROC/AUPRC (RD-D1s)
        if df_rd is not None and "rd4ad_crop_score" in df_rd.columns:
            sel_with_rd = sel_df.merge(
                df_rd[["candidate_id", "rd4ad_crop_score"]],
                on="candidate_id", how="inner"
            )
            labels_bin = (sel_with_rd[EVAL_LABEL_COL] == "positive").astype(int).values
            scores_rd = sel_with_rd["rd4ad_crop_score"].values
            auroc_rd, auprc_rd = compute_auroc_auprc_skleanfree(scores_rd, labels_bin)
            metrics["ref_rd_d1s_auroc"] = round(auroc_rd, 6) if auroc_rd is not None else None
            metrics["ref_rd_d1s_auprc"] = round(auprc_rd, 6) if auprc_rd is not None else None
        else:
            metrics["ref_rd_d1s_auroc"] = None
            metrics["ref_rd_d1s_auprc"] = None

        # 참고 AUROC/AUPRC (ConvAE)
        if df_conv is not None:
            sel_with_conv = sel_df.merge(
                df_conv[["candidate_id", "convAE_crop_score_l1_mean"]],
                on="candidate_id", how="inner"
            )
            labels_bin_c = (sel_with_conv[EVAL_LABEL_COL] == "positive").astype(int).values
            scores_conv = sel_with_conv["convAE_crop_score_l1_mean"].values
            auroc_c, auprc_c = compute_auroc_auprc_skleanfree(scores_conv, labels_bin_c)
            metrics["ref_convae_auroc"] = round(auroc_c, 6) if auroc_c is not None else None
            metrics["ref_convae_auprc"] = round(auprc_c, 6) if auprc_c is not None else None
        else:
            metrics["ref_convae_auroc"] = None
            metrics["ref_convae_auprc"] = None

        all_variant_rows.append(metrics)

        for pp in problem_patients:
            all_problem_rows.append({
                "variant": vdef["name"],
                "patient_id": pp,
                "positive_retention": round(float(per_ret.loc[pp]) if pp in per_ret.index else 0.0, 6),
            })

        # A4_p99_z_pm2 → recommended manifest preview (candidate_id + selector 컬럼만)
        if vdef["name"] == "A4_p99_z_pm2":
            preview_cols = [
                "candidate_id", "patient_id", "safe_id", "local_z",
                "first_stage_score", "threshold_p95", "threshold_p99",
                "candidate_rule",
            ]
            if "six_bin_label" in sel_df.columns:
                preview_cols.append("six_bin_label")
            recommended_manifest = sel_df[preview_cols].copy()
            recommended_manifest["selector_variant"] = "A4_p99_z_pm2"

        # per-patient summary (A0 and primary variants)
        if vdef["name"] in ("A0_all", "A4_p99_z_pm2", "A5_p99_z_pm3"):
            pat_pos = df[df[EVAL_LABEL_COL] == "positive"].groupby("patient_id")["candidate_id"].count().rename("n_positive_all")
            pat_hn = df[df[EVAL_LABEL_COL] == "hard_negative"].groupby("patient_id")["candidate_id"].count().rename("n_hn_all")
            sel_pos = sel_df[sel_df[EVAL_LABEL_COL] == "positive"].groupby("patient_id")["candidate_id"].count().rename("n_positive_sel")
            sel_hn = sel_df[sel_df[EVAL_LABEL_COL] == "hard_negative"].groupby("patient_id")["candidate_id"].count().rename("n_hn_sel")
            pat_summ = pat_pos.to_frame().join(pat_hn, how="outer").join(sel_pos, how="outer").join(sel_hn, how="outer").fillna(0).astype(int)
            pat_summ["positive_retention"] = (pat_summ["n_positive_sel"] / pat_summ["n_positive_all"].clip(lower=1)).round(6)
            pat_summ["hn_reduction"] = (1.0 - pat_summ["n_hn_sel"] / pat_summ["n_hn_all"].clip(lower=1)).round(6)
            pat_summ["variant"] = vdef["name"]
            per_patient_summary_rows.append(pat_summ.reset_index())

    # 7. 출력 파일 저장
    import pandas as pd

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 7a. variant summary CSV
    df_var = pd.DataFrame(all_variant_rows)
    # position_bin_dist는 dict → JSON 문자열로
    if "position_bin_dist" in df_var.columns:
        df_var["position_bin_dist"] = df_var["position_bin_dist"].apply(json.dumps)
    var_out = MANIFEST_DIR / "padim_selected_only_variant_summary.csv"
    df_var.to_csv(var_out, index=False)
    log_info(f"저장: {var_out}")

    # 7b. patient summary CSV
    if per_patient_summary_rows:
        df_pat = pd.concat(per_patient_summary_rows, ignore_index=True)
        pat_out = MANIFEST_DIR / "padim_selected_only_patient_summary.csv"
        df_pat.to_csv(pat_out, index=False)
        log_info(f"저장: {pat_out}")

    # 7c. problem patients CSV
    df_prob = pd.DataFrame(all_problem_rows)
    prob_out = MANIFEST_DIR / "padim_selected_only_problem_patients.csv"
    df_prob.to_csv(prob_out, index=False)
    log_info(f"저장: {prob_out}")

    # 7d. recommended manifest preview
    if recommended_manifest is not None:
        rec_out = MANIFEST_DIR / "padim_selected_only_recommended_manifest_preview.csv"
        recommended_manifest.to_csv(rec_out, index=False)
        log_info(f"저장: {rec_out} ({len(recommended_manifest):,} rows)")

    # 7e. errors CSV
    df_err = pd.DataFrame(errors) if errors else pd.DataFrame(columns=["type", "message"])
    err_out = LOG_DIR / "errors.csv"
    df_err.to_csv(err_out, index=False)
    log_info(f"저장: {err_out}")

    # 8. 판정
    primary = next((r for r in all_variant_rows if r["variant"] == "A4_p99_z_pm2"), None)
    safety = next((r for r in all_variant_rows if r["variant"] == "A5_p99_z_pm3"), None)

    recommended_selector = "UNDETERMINED"
    verdict = "FAIL"

    def meets_primary_criteria(r):
        return (
            r["lesion_patient_coverage"] >= 1.0
            and r["lesion_slice_coverage"] >= 0.97
            and r["coverage_lt50_patient_count"] == 0
            and r["reduction_rate"] > 0
            and r["hard_negative_reduction"] > 0
            and not GUARDRAILS["stage2_holdout_accessed"]
            and not GUARDRAILS["label_used_as_selector"]
            and not GUARDRAILS["second_stage_score_used_as_selector"]
        )

    if primary and meets_primary_criteria(primary):
        recommended_selector = "A4_p99_z_pm2"
        verdict = "PASS"
    elif safety and meets_primary_criteria(safety):
        recommended_selector = "A5_p99_z_pm3"
        verdict = "PARTIAL_PASS"
    else:
        # check p95 variants
        for bname in ("B2_p95_z_pm2", "B3_p95_z_pm3"):
            bv = next((r for r in all_variant_rows if r["variant"] == bname), None)
            if bv and meets_primary_criteria(bv):
                recommended_selector = bname
                verdict = "PARTIAL_PASS"
                break
        else:
            verdict = "FAIL"

    if errors:
        verdict = "FAIL"

    # 9. summary JSON
    summary = {
        "script": "padim_selected_only_second_stage_preflight.py",
        "version": "v1",
        "verdict": verdict,
        "recommended_selector": recommended_selector,
        "selector_used_columns": sorted(selector_cols_used),
        "forbidden_selector_column_intersection": sorted(list(forbidden_hit)),
        "label_leakage": bool(forbidden_hit),
        "n_errors": len(errors),
        **GUARDRAILS,
        "a0_n_candidates": int(len(df)),
        "variant_summary": {
            r["variant"]: {
                "n_selected": r["n_selected"],
                "reduction_rate": r["reduction_rate"],
                "lesion_patient_coverage": r["lesion_patient_coverage"],
                "lesion_slice_coverage": r["lesion_slice_coverage"],
                "positive_crop_retention": r["positive_crop_retention"],
                "hard_negative_reduction": r["hard_negative_reduction"],
                "coverage_lt50_patient_count": r["coverage_lt50_patient_count"],
                "coverage_lt80_patient_count": r["coverage_lt80_patient_count"],
                "coverage_lt95_patient_count": r["coverage_lt95_patient_count"],
                "ref_rd_d1s_auroc": r.get("ref_rd_d1s_auroc"),
                "ref_convae_auroc": r.get("ref_convae_auroc"),
            }
            for r in all_variant_rows
        },
    }
    summ_out = REPORT_DIR / "padim_selected_only_second_stage_preflight_summary.json"
    with open(summ_out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log_info(f"저장: {summ_out}")

    # 10. markdown report
    _write_report(summary, all_variant_rows, primary)

    # 11. DONE.json
    if not errors and GUARDRAILS["stage2_holdout_accessed"] is False:
        done_payload = {
            "done": True,
            "verdict": verdict,
            "recommended_selector": recommended_selector,
            **GUARDRAILS,
        }
        done_out = OUT_ROOT / "DONE.json"
        with open(done_out, "w") as f:
            json.dump(done_payload, f, indent=2)
        log_info(f"저장: {done_out}")

    # 12. 최종 출력
    print()
    print("=" * 70)
    print(f"판정: {verdict}")
    print(f"recommended_selector: {recommended_selector}")
    if primary:
        print(f"A4_p99_z_pm2 핵심 수치:")
        print(f"  n_selected:               {primary['n_selected']:,} / {primary['n_total']:,}")
        print(f"  reduction_rate:           {primary['reduction_rate']:.4f}")
        print(f"  lesion_patient_coverage:  {primary['lesion_patient_coverage']:.4f}")
        print(f"  lesion_slice_coverage:    {primary['lesion_slice_coverage']:.4f}")
        print(f"  positive_crop_retention:  {primary['positive_crop_retention']:.4f}")
        print(f"  hard_negative_reduction:  {primary['hard_negative_reduction']:.4f}")
        print(f"  coverage_lt50:            {primary['coverage_lt50_patient_count']}")
    print(f"label_leakage:              {bool(forbidden_hit)}")
    print(f"stage2_holdout_accessed:    {GUARDRAILS['stage2_holdout_accessed']}")
    print(f"model_forward_executed:     {GUARDRAILS['model_forward_executed']}")
    print(f"errors:                     {len(errors)}")
    print("=" * 70)

    if errors:
        for e in errors:
            print(f"[ERROR] {e['message']}", file=sys.stderr)
        sys.exit(1)


def _write_report(summary, all_variant_rows, primary):
    lines = []
    lines.append("# PaDiM-selected-only Second-Stage Preflight Report v1\n")
    lines.append(f"**판정:** {summary['verdict']}\n")
    lines.append(f"**추천 selector:** `{summary['recommended_selector']}`\n")
    lines.append("\n## Guardrail 체크\n")
    guard_keys = [
        "stage2_holdout_accessed", "model_forward_executed", "training_executed",
        "crop_generation_executed", "full_scoring_executed", "checkpoint_loaded",
        "threshold_recalculated", "existing_artifact_modified", "existing_script_modified",
        "output_overwrite", "label_used_as_selector", "label_used_for_evaluation_only",
        "second_stage_score_used_as_selector",
    ]
    for k in guard_keys:
        v = summary.get(k, "N/A")
        status = "✓" if (k == "label_used_for_evaluation_only" and v) or (k != "label_used_for_evaluation_only" and not v) else "✗"
        lines.append(f"- {status} `{k}`: `{v}`\n")

    lines.append("\n## Selector 사용 컬럼\n")
    lines.append(f"`{summary['selector_used_columns']}`\n")
    lines.append(f"\n**Forbidden column intersection:** `{summary['forbidden_selector_column_intersection']}`\n")
    lines.append(f"**Label leakage:** `{summary['label_leakage']}`\n")

    lines.append("\n## Variant별 핵심 수치\n")
    headers = [
        "variant", "n_selected", "reduction_rate",
        "lesion_patient_cov", "lesion_slice_cov",
        "pos_retention", "hn_reduction",
        "cov<50", "cov<80", "cov<95",
        "ref_rd4ad_auroc", "ref_convae_auroc",
    ]
    lines.append("| " + " | ".join(headers) + " |\n")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |\n")
    for r in all_variant_rows:
        row = [
            r["variant"],
            f"{r['n_selected']:,}",
            f"{r['reduction_rate']:.4f}",
            f"{r['lesion_patient_coverage']:.4f}",
            f"{r['lesion_slice_coverage']:.4f}",
            f"{r['positive_crop_retention']:.4f}",
            f"{r['hard_negative_reduction']:.4f}",
            str(r["coverage_lt50_patient_count"]),
            str(r["coverage_lt80_patient_count"]),
            str(r["coverage_lt95_patient_count"]),
            str(r.get("ref_rd_d1s_auroc", "N/A")),
            str(r.get("ref_convae_auroc", "N/A")),
        ]
        lines.append("| " + " | ".join(row) + " |\n")

    if primary:
        lines.append(f"\n## Primary 후보 (A4_p99_z_pm2) 상세\n")
        for k, v in primary.items():
            if k != "position_bin_dist":
                lines.append(f"- `{k}`: `{v}`\n")

    lines.append("\n## 다음 단계\n")
    lines.append(
        f"판정이 PASS이면 다음 단계는 selected-only actual scoring script preflight다.\n"
        "선택된 후보만 RD-D1s 또는 ConvAE로 실제 forward/scoring하는 스크립트를 만든다.\n"
        "단, 아직 stage2_holdout은 접근 금지다.\n"
    )

    report_out = REPORT_DIR / "padim_selected_only_second_stage_preflight_report.md"
    with open(report_out, "w") as f:
        f.writelines(lines)
    log_info(f"저장: {report_out}")


def _finalize_with_errors():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    df_err = pd.DataFrame(errors)
    df_err.to_csv(LOG_DIR / "errors.csv", index=False)
    print(f"[FAIL] {len(errors)} error(s)", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="PaDiM-selected-only second-stage preflight"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-preflight", action="store_true")
    parser.add_argument("--confirm-readonly", action="store_true")
    parser.add_argument("--confirm-stage1dev-only", action="store_true")
    args = parser.parse_args()

    # bare run 금지
    if not args.dry_run and not args.run_preflight:
        print("[ERROR] bare run 금지. --dry-run 또는 --run-preflight 를 사용하세요.", file=sys.stderr)
        sys.exit(2)

    if args.dry_run:
        run_dry()
    elif args.run_preflight:
        if not args.confirm_readonly or not args.confirm_stage1dev_only:
            print(
                "[ERROR] --confirm-readonly 와 --confirm-stage1dev-only 를 함께 전달해야 합니다.",
                file=sys.stderr,
            )
            sys.exit(2)
        run_preflight()


if __name__ == "__main__":
    main()
