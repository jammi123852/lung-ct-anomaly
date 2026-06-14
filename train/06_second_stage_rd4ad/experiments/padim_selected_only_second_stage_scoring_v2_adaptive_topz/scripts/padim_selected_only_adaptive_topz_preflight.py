"""
PaDiM-selected-only second-stage scoring preflight v2 — patient-adaptive top-z selector.

v1에서 global p99/p95 threshold 기반 selector가 실패했으므로
환자 내부에서 상대적으로 의심도가 높은 z-slice를 선택하는 방식으로 전환한다.

이 스크립트는 preflight/post-hoc 분석만 수행한다.
모델 forward, crop 생성, 재학습, full scoring은 하지 않는다.

Usage:
  # dry-run (입력 확인만, 파일 생성 없음)
  python experiments/padim_selected_only_second_stage_scoring_v2_adaptive_topz/scripts/padim_selected_only_adaptive_topz_preflight.py --dry-run

  # 실제 preflight
  python experiments/padim_selected_only_second_stage_scoring_v2_adaptive_topz/scripts/padim_selected_only_adaptive_topz_preflight.py \\
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

V1_SUMMARY_JSON = (
    PROJECT_ROOT
    / "experiments/padim_selected_only_second_stage_scoring_v1"
    / "reports/padim_selected_only_second_stage_preflight_summary.json"
)

STAGE2_HOLDOUT_PATH_KEYWORDS = [
    "stage2_holdout",
    "second-stage-lesion-refiner-v1/datasets",
]

OUT_ROOT = (
    PROJECT_ROOT
    / "experiments/padim_selected_only_second_stage_scoring_v2_adaptive_topz"
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
    "local_z", "first_stage_score",
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

# v2 selector는 이 컬럼만 사용한다
SELECTOR_COLS_USED = {
    "patient_id", "local_z", "first_stage_score",
}

# v2 slice_score 후보
SLICE_SCORE_VARIANTS = ["slice_top20_mean", "slice_top10_mean", "slice_max"]

# 문제 환자 (v1에서 coverage 실패, v2에서 회복 여부 확인)
V1_PROBLEM_PATIENTS = ["LUNG1-086", "LUNG1-386", "LUNG1-399"]

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
    input_paths = [str(MANIFEST_CSV), str(RD_D1S_SCORE_CSV), str(CONVAE_SCORE_CSV)]
    for ip in input_paths:
        for kw in STAGE2_HOLDOUT_PATH_KEYWORDS:
            if kw in ip:
                GUARDRAILS["stage2_holdout_accessed"] = True
                log_error(f"입력 경로에 stage2_holdout 키워드 포함: {ip}")
    return not GUARDRAILS["stage2_holdout_accessed"]


# ---------------------------------------------------------------------------
# Slice score computation
# ---------------------------------------------------------------------------
def compute_slice_scores(df):
    """
    각 (patient_id, local_z) 단위로 slice_score 3종을 계산한다.
    selector로 사용 가능한 컬럼만 사용: patient_id, local_z, first_stage_score
    """
    import pandas as pd
    import numpy as np

    def top_n_mean(scores, n):
        arr = np.array(scores)
        if len(arr) == 0:
            return 0.0
        k = min(n, len(arr))
        return float(np.sort(arr)[-k:].mean())

    grp = df.groupby(["patient_id", "local_z"])["first_stage_score"]

    slice_top20_mean = grp.apply(lambda x: top_n_mean(x.values, 20)).rename("slice_top20_mean")
    slice_top10_mean = grp.apply(lambda x: top_n_mean(x.values, 10)).rename("slice_top10_mean")
    slice_max = grp.max().rename("slice_max")

    slice_df = pd.concat([slice_top20_mean, slice_top10_mean, slice_max], axis=1).reset_index()
    return slice_df


# ---------------------------------------------------------------------------
# Adaptive top-z selector
# ---------------------------------------------------------------------------
def select_adaptive_topz(df, slice_df, slice_score_col, top_n_z, z_delta, cap=None):
    """
    patient별 top_n_z z-slice를 slice_score_col 기준으로 선택하고 ±z_delta 확장.
    cap이 있으면 (per_slice_top50 등) selected z-slice 내에서 추가 필터 적용.

    반환: bool 마스크 (df 기준)
    """
    import pandas as pd

    # patient별 top_n_z z 선택
    def get_top_z_set(group):
        """patient 내 z-slice score 상위 top_n_z 개 z 반환 (z_delta 포함)"""
        sorted_z = group.nlargest(top_n_z, slice_score_col)["local_z"].values
        expanded = set()
        for z in sorted_z:
            for dz in range(-z_delta, z_delta + 1):
                expanded.add(int(z) + dz)
        return expanded

    patient_z_sets = (
        slice_df.groupby("patient_id")
        .apply(get_top_z_set)
        .to_dict()
    )

    # 기본 마스크: selected z-slice에 속하는 candidate
    mask = df.apply(
        lambda r: int(r["local_z"]) in patient_z_sets.get(r["patient_id"], set()),
        axis=1,
    )

    if cap is None or cap == "no_cap":
        return mask

    # cap variant 적용
    sel_idx = mask[mask].index.tolist()
    sel_df = df.loc[sel_idx].copy()

    if cap.startswith("per_slice_top"):
        m = int(cap.replace("per_slice_top", ""))
        keep_idx = set()
        for (pid, z), grp in sel_df.groupby(["patient_id", "local_z"]):
            top_idx = grp.nlargest(m, "first_stage_score").index
            keep_idx.update(top_idx)
        final_mask = df.index.isin(keep_idx)
        return final_mask

    elif cap.startswith("per_patient_top"):
        m = int(cap.replace("per_patient_top", ""))
        keep_idx = set()
        for pid, grp in sel_df.groupby("patient_id"):
            top_idx = grp.nlargest(m, "first_stage_score").index
            keep_idx.update(top_idx)
        final_mask = df.index.isin(keep_idx)
        return final_mask

    else:
        raise ValueError(f"Unknown cap: {cap}")


# ---------------------------------------------------------------------------
# Coverage / reduction metrics
# ---------------------------------------------------------------------------
def compute_variant_metrics(df, mask, variant_name, desc, label_col="label"):
    """
    선택 후 coverage/reduction 지표 계산.
    label_col은 평가 전용이며 selector로 사용하지 않는다.
    """
    import numpy as np

    sel = df[mask]
    n_all = len(df)
    n_sel = len(sel)

    is_pos = df[label_col] == "positive"
    is_hn = df[label_col] == "hard_negative"
    sel_is_pos = sel[label_col] == "positive"
    sel_is_hn = sel[label_col] == "hard_negative"

    n_pos_all = int(is_pos.sum())
    n_hn_all = int(is_hn.sum())
    n_pos_sel = int(sel_is_pos.sum())
    n_hn_sel = int(sel_is_hn.sum())

    reduction_rate = 1.0 - n_sel / n_all if n_all > 0 else 0.0

    all_patients = set(df["patient_id"].unique())
    sel_patients = set(sel["patient_id"].unique())
    patient_cov = len(sel_patients) / len(all_patients) if all_patients else 0.0

    lesion_patients = set(df[is_pos]["patient_id"].unique())
    sel_lesion_patients = set(sel[sel_is_pos]["patient_id"].unique())
    lesion_patient_cov = (
        len(sel_lesion_patients) / len(lesion_patients) if lesion_patients else 0.0
    )

    all_lesion_slices = set(
        map(tuple, df[is_pos][["patient_id", "local_z"]].drop_duplicates().values)
    )
    sel_lesion_slices = set(
        map(tuple, sel[sel_is_pos][["patient_id", "local_z"]].drop_duplicates().values)
    )
    lesion_slice_cov = (
        len(sel_lesion_slices) / len(all_lesion_slices) if all_lesion_slices else 0.0
    )

    pos_retention = n_pos_sel / n_pos_all if n_pos_all > 0 else 0.0
    hn_retention = n_hn_sel / n_hn_all if n_hn_all > 0 else 0.0
    hn_reduction = 1.0 - hn_retention

    per_patient_pos_all = df[is_pos].groupby("patient_id")["candidate_id"].count()
    per_patient_pos_sel = sel[sel_is_pos].groupby("patient_id")["candidate_id"].count()
    per_patient_pos_sel = per_patient_pos_sel.reindex(per_patient_pos_all.index, fill_value=0)
    per_patient_ret = per_patient_pos_sel / per_patient_pos_all

    problem_patients = per_patient_ret[per_patient_ret < 1.0].index.tolist()
    cov_lt50 = int((per_patient_ret < 0.5).sum())
    cov_lt80 = int((per_patient_ret < 0.8).sum())
    cov_lt95 = int((per_patient_ret < 0.95).sum())

    sel_per_patient = sel.groupby("patient_id")["candidate_id"].count()
    p25 = float(np.percentile(sel_per_patient, 25))
    p50 = float(np.percentile(sel_per_patient, 50))
    p75 = float(np.percentile(sel_per_patient, 75))
    p95 = float(np.percentile(sel_per_patient, 95))

    # position_bin 분포
    pb_col = "six_bin_label" if "six_bin_label" in sel.columns else (
        "position_bin" if "position_bin" in sel.columns else None
    )
    pb_dist = sel[pb_col].value_counts().to_dict() if pb_col else {}

    # v1 문제 환자 coverage 회복 여부
    problem_patient_recovery = {}
    for pp in V1_PROBLEM_PATIENTS:
        if pp in per_patient_pos_all.index:
            ret_val = float(per_patient_ret.get(pp, 0.0))
            problem_patient_recovery[pp] = {
                "retention": round(ret_val, 6),
                "recovered": ret_val >= 1.0,
            }
        else:
            problem_patient_recovery[pp] = {"retention": None, "recovered": None}

    # selected z 통계
    sel_z_per_patient = sel.groupby("patient_id")["local_z"].nunique()
    z_nunique_p50 = float(np.percentile(sel_z_per_patient, 50))
    z_nunique_p95 = float(np.percentile(sel_z_per_patient, 95))

    result = {
        "variant": variant_name,
        "desc": desc,
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
        "n_positive_all": n_pos_all,
        "n_positive_sel": n_pos_sel,
        "positive_crop_retention": round(pos_retention, 6),
        "n_hard_negative_all": n_hn_all,
        "n_hard_negative_sel": n_hn_sel,
        "hard_negative_retention": round(hn_retention, 6),
        "hard_negative_reduction": round(hn_reduction, 6),
        "n_problem_patients": int(len(problem_patients)),
        "coverage_lt50_patient_count": cov_lt50,
        "coverage_lt80_patient_count": cov_lt80,
        "coverage_lt95_patient_count": cov_lt95,
        "per_patient_sel_p25": p25,
        "per_patient_sel_p50": p50,
        "per_patient_sel_p75": p75,
        "per_patient_sel_p95": p95,
        "sel_z_per_patient_p50": z_nunique_p50,
        "sel_z_per_patient_p95": z_nunique_p95,
        "position_bin_dist": pb_dist,
        "v1_problem_patient_recovery": problem_patient_recovery,
        "stage2_holdout_accessed": False,
        "label_used_as_selector": False,
        "second_stage_score_used_as_selector": False,
    }
    return result, problem_patients, sel, per_patient_ret


# ---------------------------------------------------------------------------
# AUROC/AUPRC helper (sklearn-free)
# ---------------------------------------------------------------------------
def compute_auroc_auprc(scores, labels_binary):
    import numpy as np
    from numpy import searchsorted

    pos = scores[labels_binary == 1]
    neg = scores[labels_binary == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None, None

    n_pos, n_neg = len(pos), len(neg)
    combined = np.concatenate([pos, neg])
    sorted_combined = np.sort(combined)
    ranks_pos = searchsorted(sorted_combined, pos, side="left") + 1
    u = ranks_pos.sum() - n_pos * (n_pos + 1) / 2
    auroc = u / (n_pos * n_neg)

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
    print("=" * 70)
    print("[DRY-RUN] padim_selected_only_adaptive_topz_preflight v2")
    print("=" * 70)

    # 1. stage2_holdout 접근 확인
    ok_holdout = check_stage2_holdout_not_accessed()
    print(f"[CHECK] stage2_holdout 접근 없음: {ok_holdout}")

    # 2. 입력 파일 존재 확인
    for label, path in [
        ("manifest_csv", MANIFEST_CSV),
        ("rd_d1s_score_csv", RD_D1S_SCORE_CSV),
        ("convae_score_csv", CONVAE_SCORE_CSV),
        ("v1_summary_json", V1_SUMMARY_JSON),
    ]:
        exists = path.exists()
        print(f"[CHECK] {label}: {'OK' if exists else 'MISSING'} — {path}")
        if not exists and label == "manifest_csv":
            log_error(f"필수 입력 파일 없음: {path}")

    # 3. output root 충돌 확인
    for out_file in [
        REPORT_DIR / "padim_selected_only_adaptive_topz_preflight_report.md",
        REPORT_DIR / "padim_selected_only_adaptive_topz_preflight_summary.json",
        MANIFEST_DIR / "padim_selected_only_adaptive_topz_variant_summary.csv",
    ]:
        if out_file.exists():
            print(f"[WARN] 출력 파일 이미 존재 (overwrite 방지 — 삭제 후 재실행): {out_file}")

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

        # selector_used_columns 확인 (v2: threshold 컬럼 불필요)
        forbidden_intersection = SELECTOR_COLS_USED & FORBIDDEN_SELECTOR_COLUMNS
        if forbidden_intersection:
            log_error(f"selector에 forbidden column 포함: {forbidden_intersection}")
            print(f"[FAIL] label leakage 발견: {forbidden_intersection}")
        else:
            print(f"[CHECK] selector_used_columns (v2): {sorted(SELECTOR_COLS_USED)}")
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
        n_dev = (raw["stage_split"] == "stage1_dev").sum()
        print(f"[CHECK] stage1_dev 행 수: {n_dev:,}")

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
    print("[PREFLIGHT] padim_selected_only_adaptive_topz_preflight v2")
    print("=" * 70)

    # 0. stage2_holdout 확인
    check_stage2_holdout_not_accessed()

    # 1. manifest CSV 로드
    if not MANIFEST_CSV.exists():
        log_error(f"manifest CSV 없음: {MANIFEST_CSV}")
        _finalize_with_errors()
        return

    log_info(f"manifest CSV 로드: {MANIFEST_CSV}")
    df = pd.read_csv(MANIFEST_CSV)
    log_info(f"  → {len(df):,} rows, columns: {df.columns.tolist()}")

    if "stage_split" in df.columns:
        holdout_rows = (df["stage_split"] == "stage2_holdout").sum()
        if holdout_rows > 0:
            GUARDRAILS["stage2_holdout_accessed"] = True
            log_error(f"stage2_holdout 행 {holdout_rows}개 존재!")
        df = df[df["stage_split"] == "stage1_dev"].copy()
        log_info(f"  → stage1_dev 필터 후: {len(df):,} rows")

    missing_cols = [c for c in REQUIRED_MANIFEST_COLUMNS if c not in df.columns]
    if missing_cols:
        log_error(f"manifest 컬럼 누락: {missing_cols}")

    # 2. RD-D1s score 로드 (참고용, six_bin_label 추가)
    if RD_D1S_SCORE_CSV.exists():
        log_info(f"RD-D1s score CSV 로드 (참고용): {RD_D1S_SCORE_CSV}")
        df_rd = pd.read_csv(
            RD_D1S_SCORE_CSV,
            usecols=["candidate_id", "rd4ad_crop_score", "six_bin_label", "z_level", "boundary_status"],
        )
        df = df.merge(
            df_rd[["candidate_id", "six_bin_label", "z_level", "boundary_status"]],
            on="candidate_id", how="left",
        )
        log_info("  → six_bin_label 추가 완료")
    else:
        log_warn(f"RD-D1s score CSV 없음 (참고 지표 생략): {RD_D1S_SCORE_CSV}")

    # 3. ConvAE score 로드 (참고용)
    df_conv = None
    if CONVAE_SCORE_CSV.exists():
        log_info(f"ConvAE score CSV 로드 (참고용): {CONVAE_SCORE_CSV}")
        df_conv = pd.read_csv(
            CONVAE_SCORE_CSV, usecols=["candidate_id", "convAE_crop_score_l1_mean"]
        )
    else:
        log_warn(f"ConvAE score CSV 없음 (참고 지표 생략): {CONVAE_SCORE_CSV}")

    # 4. selector column 검증
    forbidden_hit = SELECTOR_COLS_USED & FORBIDDEN_SELECTOR_COLUMNS
    if forbidden_hit:
        GUARDRAILS["label_used_as_selector"] = True
        log_error(f"selector에 forbidden column 포함: {forbidden_hit}")

    EVAL_LABEL_COL = "label"
    if EVAL_LABEL_COL not in df.columns:
        log_error("'label' column 없음 — 평가 지표 계산 불가")
        _finalize_with_errors()
        return

    log_info(f"  label 분포: {df[EVAL_LABEL_COL].value_counts().to_dict()}")

    # 5. slice_score 계산 (v2 핵심)
    log_info("slice_score 계산 중 (top20_mean / top10_mean / max) ...")
    slice_df = compute_slice_scores(df)
    log_info(f"  → {len(slice_df):,} (patient, z) pairs")

    # 6. variant 정의
    # C1~C5: primary selector variants (slice_top20_mean)
    # D1~D4: ablation (slice_top10_mean, slice_max)
    # cap: no_cap / per_slice_top50 / per_slice_top100 / per_patient_top300 / per_patient_top500
    PRIMARY_VARIANTS = [
        # (name, slice_score_col, top_n_z, z_delta, cap, desc)
        ("C1_top3z_pm2",  "slice_top20_mean", 3,  2, "no_cap",  "top3z pm2 (top20_mean)"),
        ("C2_top5z_pm2",  "slice_top20_mean", 5,  2, "no_cap",  "top5z pm2 (top20_mean) [primary 후보]"),
        ("C3_top5z_pm3",  "slice_top20_mean", 5,  3, "no_cap",  "top5z pm3 (top20_mean) [safety 후보]"),
        ("C4_top10z_pm2", "slice_top20_mean", 10, 2, "no_cap",  "top10z pm2 (top20_mean)"),
        ("C5_top10z_pm3", "slice_top20_mean", 10, 3, "no_cap",  "top10z pm3 (top20_mean) [upper-bound]"),
    ]
    ABLATION_VARIANTS = [
        ("D1_top5z_pm2_slice_top10_mean",  "slice_top10_mean", 5,  2, "no_cap", "top5z pm2 (top10_mean)"),
        ("D2_top5z_pm2_slice_max",          "slice_max",        5,  2, "no_cap", "top5z pm2 (slice_max)"),
        ("D3_top5z_pm3_slice_top10_mean",  "slice_top10_mean", 5,  3, "no_cap", "top5z pm3 (top10_mean)"),
        ("D4_top5z_pm3_slice_max",          "slice_max",        5,  3, "no_cap", "top5z pm3 (slice_max)"),
    ]
    # cap variant combinations: C2, C3 기준으로만 계산
    CAP_VARIANTS = []
    for base_name, ssc, tnz, zdelta, _, base_desc in [
        ("C2_top5z_pm2",  "slice_top20_mean", 5, 2, None, "top5z pm2"),
        ("C3_top5z_pm3",  "slice_top20_mean", 5, 3, None, "top5z pm3"),
    ]:
        for cap in ["per_slice_top50", "per_slice_top100", "per_patient_top300", "per_patient_top500"]:
            CAP_VARIANTS.append(
                (f"{base_name}_{cap}", ssc, tnz, zdelta, cap, f"{base_desc} cap={cap}")
            )

    # baseline
    BASELINE = [("A0_all", None, None, None, None, "전체 candidate (기준선)")]

    ALL_VARIANTS = BASELINE + PRIMARY_VARIANTS + ABLATION_VARIANTS + CAP_VARIANTS

    # 7. 각 variant 계산
    all_variant_rows = []
    all_problem_rows = []
    per_patient_summary_rows = []
    primary_c2_sel_df = None

    for (vname, ssc, tnz, zdelta, cap, vdesc) in ALL_VARIANTS:
        log_info(f"variant: {vname}")
        try:
            if vname == "A0_all":
                mask = pd.Series([True] * len(df), index=df.index)
            else:
                mask = select_adaptive_topz(df, slice_df, ssc, tnz, zdelta, cap=cap)

            metrics, prob_patients, sel_df, per_ret = compute_variant_metrics(
                df, mask, vname, vdesc, label_col=EVAL_LABEL_COL
            )

            # 참고용 AUROC 계산 (RD-D1s)
            if RD_D1S_SCORE_CSV.exists():
                df_rd_full = pd.read_csv(RD_D1S_SCORE_CSV, usecols=["candidate_id", "rd4ad_crop_score"])
                rd_merged = sel_df.merge(df_rd_full, on="candidate_id", how="left")
                rd_scores = rd_merged["rd4ad_crop_score"].fillna(0.0).values
                rd_labels = (rd_merged[EVAL_LABEL_COL] == "positive").astype(int).values
                rd_auroc, rd_auprc = compute_auroc_auprc(rd_scores, rd_labels)
                metrics["ref_rd_d1s_auroc"] = round(rd_auroc, 6) if rd_auroc is not None else None
                metrics["ref_rd_d1s_auprc"] = round(rd_auprc, 6) if rd_auprc is not None else None
            else:
                metrics["ref_rd_d1s_auroc"] = None
                metrics["ref_rd_d1s_auprc"] = None

            # 참고용 AUROC 계산 (ConvAE)
            if df_conv is not None:
                conv_merged = sel_df.merge(df_conv, on="candidate_id", how="left")
                conv_scores = conv_merged["convAE_crop_score_l1_mean"].fillna(0.0).values
                conv_labels = (conv_merged[EVAL_LABEL_COL] == "positive").astype(int).values
                conv_auroc, conv_auprc = compute_auroc_auprc(conv_scores, conv_labels)
                metrics["ref_convae_auroc"] = round(conv_auroc, 6) if conv_auroc is not None else None
                metrics["ref_convae_auprc"] = round(conv_auprc, 6) if conv_auprc is not None else None
            else:
                metrics["ref_convae_auroc"] = None
                metrics["ref_convae_auprc"] = None

            all_variant_rows.append(metrics)

            if vname == "C2_top5z_pm2":
                primary_c2_sel_df = sel_df.copy()

            # per-patient 기록
            for pid in df["patient_id"].unique():
                n_pos_all_p = int((df[df["patient_id"] == pid][EVAL_LABEL_COL] == "positive").sum())
                n_hn_all_p = int((df[df["patient_id"] == pid][EVAL_LABEL_COL] == "hard_negative").sum())
                if pid in sel_df["patient_id"].values:
                    n_pos_sel_p = int((sel_df[sel_df["patient_id"] == pid][EVAL_LABEL_COL] == "positive").sum())
                    n_hn_sel_p = int((sel_df[sel_df["patient_id"] == pid][EVAL_LABEL_COL] == "hard_negative").sum())
                else:
                    n_pos_sel_p = 0
                    n_hn_sel_p = 0
                pos_ret_p = n_pos_sel_p / n_pos_all_p if n_pos_all_p > 0 else None
                hn_red_p = 1.0 - n_hn_sel_p / n_hn_all_p if n_hn_all_p > 0 else None
                per_patient_summary_rows.append({
                    "patient_id": pid,
                    "n_positive_all": n_pos_all_p,
                    "n_hn_all": n_hn_all_p,
                    "n_positive_sel": n_pos_sel_p,
                    "n_hn_sel": n_hn_sel_p,
                    "positive_retention": pos_ret_p,
                    "hn_reduction": hn_red_p,
                    "variant": vname,
                })

            # 문제 환자 기록
            for pp in prob_patients:
                all_problem_rows.append({
                    "variant": vname,
                    "patient_id": pp,
                    "positive_retention": round(float(per_ret.get(pp, 0.0)), 6),
                })

        except Exception as e:
            log_error(f"variant {vname} 계산 오류: {e}\n{traceback.format_exc()}")
            continue

    # 8. v1 비교 데이터 로드
    v1_compare = {}
    if V1_SUMMARY_JSON.exists():
        log_info(f"v1 summary 로드 (비교용): {V1_SUMMARY_JSON}")
        with open(V1_SUMMARY_JSON) as f:
            v1_data = json.load(f)
        for vname in ["A4_p99_z_pm2", "A5_p99_z_pm3", "A6_p99_z_pm5", "B2_p95_z_pm2"]:
            if vname in v1_data.get("variant_summary", {}):
                v1_compare[vname] = v1_data["variant_summary"][vname]
    else:
        log_warn("v1 summary JSON 없음 — v1 비교 생략")

    # 9. recommended selector 판정
    def check_primary_criteria(m):
        return (
            m.get("lesion_patient_coverage", 0) >= 1.0
            and m.get("lesion_slice_coverage", 0) >= 0.97
            and m.get("positive_crop_retention", 0) >= 0.97
            and m.get("coverage_lt50_patient_count", 999) == 0
            and all(
                m.get("v1_problem_patient_recovery", {}).get(pp, {}).get("recovered", False)
                for pp in V1_PROBLEM_PATIENTS
            )
            and m.get("reduction_rate", 0) > 0.0
            and m.get("hard_negative_reduction", 0) > 0.0
        )

    variant_map = {r["variant"]: r for r in all_variant_rows}
    recommended_selector = "UNDETERMINED"
    for cand in ["C2_top5z_pm2", "C3_top5z_pm3", "C4_top10z_pm2", "C5_top10z_pm3"]:
        if cand in variant_map and check_primary_criteria(variant_map[cand]):
            recommended_selector = cand
            break

    # cap 채택 판정 (recommended_selector 기준)
    recommended_cap = "no_cap"
    if recommended_selector != "UNDETERMINED":
        for cap_suf in ["per_slice_top50", "per_slice_top100", "per_patient_top300", "per_patient_top500"]:
            cap_vname = f"{recommended_selector}_{cap_suf}"
            if cap_vname in variant_map:
                cm = variant_map[cap_vname]
                if (
                    cm.get("lesion_patient_coverage", 0) >= 1.0
                    and cm.get("coverage_lt50_patient_count", 999) == 0
                    and cm.get("lesion_slice_coverage", 0) >= 0.97
                    and cm.get("positive_crop_retention", 0) >= 0.97
                    and cm.get("reduction_rate", 0) > variant_map[recommended_selector].get("reduction_rate", 0)
                ):
                    recommended_cap = cap_suf
                    break

    # 10. 출력 파일 생성
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # variant_summary.csv
    variant_df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("position_bin_dist", "v1_problem_patient_recovery")}
        for r in all_variant_rows
    ])
    # position_bin_dist를 JSON string으로
    variant_df["position_bin_dist"] = [
        json.dumps(r.get("position_bin_dist", {})) for r in all_variant_rows
    ]
    variant_df["v1_problem_patient_recovery"] = [
        json.dumps(r.get("v1_problem_patient_recovery", {})) for r in all_variant_rows
    ]
    variant_df.to_csv(
        MANIFEST_DIR / "padim_selected_only_adaptive_topz_variant_summary.csv",
        index=False,
    )
    log_info("variant_summary.csv 저장 완료")

    # patient_summary.csv (C2만 저장, 전체는 너무 큼)
    if per_patient_summary_rows:
        patient_df = pd.DataFrame(per_patient_summary_rows)
        patient_df_c2 = patient_df[patient_df["variant"] == "C2_top5z_pm2"].copy()
        patient_df_c2.to_csv(
            MANIFEST_DIR / "padim_selected_only_adaptive_topz_patient_summary.csv",
            index=False,
        )
        log_info("patient_summary.csv (C2 only) 저장 완료")

    # problem_patients.csv
    if all_problem_rows:
        prob_df = pd.DataFrame(all_problem_rows)
        prob_df.to_csv(
            MANIFEST_DIR / "padim_selected_only_adaptive_topz_problem_patients.csv",
            index=False,
        )
    else:
        pd.DataFrame(columns=["variant", "patient_id", "positive_retention"]).to_csv(
            MANIFEST_DIR / "padim_selected_only_adaptive_topz_problem_patients.csv",
            index=False,
        )
    log_info("problem_patients.csv 저장 완료")

    # recommended_manifest_preview.csv (recommended selector + cap)
    if primary_c2_sel_df is not None:
        preview_cols = [
            "candidate_id", "patient_id", "safe_id", "local_z", "slice_index",
            "first_stage_score", "label", "candidate_rule",
            "crop_y0", "crop_x0", "crop_y1", "crop_x1",
        ]
        preview_cols_avail = [c for c in preview_cols if c in primary_c2_sel_df.columns]
        primary_c2_sel_df[preview_cols_avail].to_csv(
            MANIFEST_DIR / "padim_selected_only_adaptive_topz_recommended_manifest_preview.csv",
            index=False,
        )
        log_info("recommended_manifest_preview.csv (C2) 저장 완료")

    # summary.json
    a0 = variant_map.get("A0_all", {})
    summary = {
        "script": "padim_selected_only_adaptive_topz_preflight.py",
        "version": "v2",
        "verdict": "PASS" if recommended_selector != "UNDETERMINED" and not errors else (
            "PARTIAL_PASS" if recommended_selector != "UNDETERMINED" else "FAIL"
        ),
        "recommended_selector": recommended_selector,
        "recommended_cap": recommended_cap,
        "selector_used_columns": sorted(SELECTOR_COLS_USED),
        "forbidden_selector_column_intersection": sorted(
            SELECTOR_COLS_USED & FORBIDDEN_SELECTOR_COLUMNS
        ),
        "label_leakage": bool(GUARDRAILS["label_used_as_selector"]),
        "n_errors": len(errors),
        "errors": errors,
        # guardrails
        **GUARDRAILS,
        "a0_n_candidates": a0.get("n_selected"),
        "variant_summary": {r["variant"]: {
            k: v for k, v in r.items()
            if k not in ("desc", "variant", "position_bin_dist")
        } for r in all_variant_rows},
        "v1_comparison": v1_compare,
    }
    with open(REPORT_DIR / "padim_selected_only_adaptive_topz_preflight_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log_info("summary.json 저장 완료")

    # report.md
    _write_report(summary, all_variant_rows, v1_compare, variant_map)

    # errors.csv
    err_df = pd.DataFrame(errors) if errors else pd.DataFrame(columns=["type", "message"])
    err_df.to_csv(LOG_DIR / "errors.csv", index=False)

    # DONE.json
    done = {
        "done": True,
        "version": "v2",
        "verdict": summary["verdict"],
        "recommended_selector": recommended_selector,
        "recommended_cap": recommended_cap,
        "n_errors": len(errors),
        "guardrails": {k: GUARDRAILS[k] for k in GUARDRAILS},
    }
    with open(OUT_ROOT / "DONE.json", "w") as f:
        json.dump(done, f, indent=2, ensure_ascii=False)
    log_info("DONE.json 저장 완료")

    # 최종 보고
    print()
    print("=" * 70)
    print(f"[RESULT] 판정: {summary['verdict']}")
    print(f"  A0 전체 candidate 수: {a0.get('n_selected'):,}")
    print(f"  recommended selector: {recommended_selector}")
    print(f"  recommended cap: {recommended_cap}")
    print(f"  label leakage: {summary['label_leakage']}")
    print(f"  stage2_holdout 접근: {GUARDRAILS['stage2_holdout_accessed']}")
    print(f"  model forward: {GUARDRAILS['model_forward_executed']}")
    print(f"  기존 artifact 수정: {GUARDRAILS['existing_artifact_modified']}")
    print()
    for vname in ["C2_top5z_pm2", "C3_top5z_pm3", "C4_top10z_pm2", "C5_top10z_pm3"]:
        m = variant_map.get(vname)
        if m:
            print(
                f"  {vname}: n={m['n_selected']:,} "
                f"reduction={m['reduction_rate']:.4f} "
                f"lp_cov={m['lesion_patient_coverage']:.4f} "
                f"ls_cov={m['lesion_slice_coverage']:.4f} "
                f"pos_ret={m['positive_crop_retention']:.4f} "
                f"hn_red={m['hard_negative_reduction']:.4f} "
                f"lt50={m['coverage_lt50_patient_count']}"
            )
    print("=" * 70)

    if errors:
        print(f"[WARN] {len(errors)} error(s) 발생 — errors.csv 확인")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------
def _write_report(summary, all_variant_rows, v1_compare, variant_map):
    lines = []
    lines.append("# PaDiM Selected-Only Second-Stage Scoring v2 — Adaptive Top-Z Preflight Report\n")
    lines.append(f"**판정: {summary['verdict']}**\n")
    lines.append(f"- recommended selector: `{summary['recommended_selector']}`\n")
    lines.append(f"- recommended cap: `{summary['recommended_cap']}`\n")
    lines.append(f"- label leakage: `{summary['label_leakage']}`\n")
    lines.append(f"- stage2_holdout 접근: `{summary['stage2_holdout_accessed']}`\n")
    lines.append(f"- model forward: `{summary['model_forward_executed']}`\n")
    lines.append(f"- 기존 artifact 수정: `{summary['existing_artifact_modified']}`\n")
    lines.append(f"- selector 사용 컬럼: `{summary['selector_used_columns']}`\n")
    lines.append(f"- forbidden column intersection: `{summary['forbidden_selector_column_intersection']}`\n")
    lines.append("\n## Variant 결과 요약\n")
    lines.append(
        "| variant | n_selected | reduction | lp_cov | ls_cov | pos_ret | hn_red | lt50 | lt80 | lt95 |\n"
        "|---------|-----------|-----------|--------|--------|---------|--------|------|------|------|\n"
    )
    for r in all_variant_rows:
        lines.append(
            f"| {r['variant']} | {r['n_selected']:,} | {r['reduction_rate']:.4f} "
            f"| {r['lesion_patient_coverage']:.4f} | {r['lesion_slice_coverage']:.4f} "
            f"| {r['positive_crop_retention']:.4f} | {r['hard_negative_reduction']:.4f} "
            f"| {r['coverage_lt50_patient_count']} | {r['coverage_lt80_patient_count']} "
            f"| {r['coverage_lt95_patient_count']} |\n"
        )

    lines.append("\n## v1 비교 (참고용)\n")
    if v1_compare:
        lines.append(
            "| v1 variant | n_selected | reduction | lp_cov | ls_cov | pos_ret | hn_red | lt50 |\n"
            "|------------|-----------|-----------|--------|--------|---------|--------|------|\n"
        )
        for vname, m in v1_compare.items():
            lines.append(
                f"| {vname} | {m.get('n_selected', '?'):,} | {m.get('reduction_rate', 0):.4f} "
                f"| {m.get('lesion_patient_coverage', 0):.4f} | {m.get('lesion_slice_coverage', 0):.4f} "
                f"| {m.get('positive_crop_retention', 0):.4f} | {m.get('hard_negative_reduction', 0):.4f} "
                f"| {m.get('coverage_lt50_patient_count', '?')} |\n"
            )
    else:
        lines.append("v1 summary JSON 없음\n")

    lines.append("\n## v1 문제 환자 coverage 회복 여부\n")
    for vname in ["C2_top5z_pm2", "C3_top5z_pm3"]:
        m = variant_map.get(vname)
        if m and "v1_problem_patient_recovery" in m:
            lines.append(f"### {vname}\n")
            for pp, info in m["v1_problem_patient_recovery"].items():
                lines.append(f"- {pp}: retention={info.get('retention')} recovered={info.get('recovered')}\n")

    lines.append("\n## Guardrails\n")
    for k, v in summary.items():
        if k in (
            "stage2_holdout_accessed", "model_forward_executed", "training_executed",
            "crop_generation_executed", "full_scoring_executed", "checkpoint_loaded",
            "threshold_recalculated", "existing_artifact_modified", "existing_script_modified",
            "output_overwrite", "label_used_as_selector", "label_used_for_evaluation_only",
            "second_stage_score_used_as_selector",
        ):
            lines.append(f"- {k}: `{v}`\n")

    lines.append("\n## 다음 단계\n")
    if summary["verdict"] == "PASS":
        lines.append(
            f"PASS. recommended selector = `{summary['recommended_selector']}`, cap = `{summary['recommended_cap']}`.\n"
            "다음 단계: selected-only actual scoring script preflight (RD-D1s 또는 ConvAE forward).\n"
            "단, stage2_holdout 접근은 아직 금지.\n"
        )
    else:
        lines.append(
            f"판정: {summary['verdict']}. coverage 조건 미충족 variant 검토 필요.\n"
            "C5_top10z_pm3를 upper-bound로 재확인할 것.\n"
        )

    report_path = REPORT_DIR / "padim_selected_only_adaptive_topz_preflight_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    log_info(f"report.md 저장 완료: {report_path}")


# ---------------------------------------------------------------------------
# Error finalization
# ---------------------------------------------------------------------------
def _finalize_with_errors():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    err_df = pd.DataFrame(errors)
    err_df.to_csv(LOG_DIR / "errors.csv", index=False)
    print(f"[FAIL] {len(errors)} critical error(s). errors.csv 확인.")
    sys.exit(2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="PaDiM selected-only second-stage scoring preflight v2 — adaptive top-z"
    )
    parser.add_argument("--dry-run", action="store_true", help="입력 확인만 수행, 파일 생성 없음")
    parser.add_argument("--run-preflight", action="store_true", help="실제 preflight 실행")
    parser.add_argument("--confirm-readonly", action="store_true", help="입력 파일 read-only 확인 동의")
    parser.add_argument("--confirm-stage1dev-only", action="store_true", help="stage1_dev만 사용 확인 동의")
    args = parser.parse_args()

    if args.dry_run:
        run_dry()
    elif args.run_preflight:
        if not args.confirm_readonly or not args.confirm_stage1dev_only:
            print("[ERROR] --confirm-readonly 와 --confirm-stage1dev-only 플래그가 필요합니다.")
            print("Usage: --run-preflight --confirm-readonly --confirm-stage1dev-only")
            sys.exit(2)
        run_preflight()
    else:
        print("[ERROR] --dry-run 또는 --run-preflight 플래그가 필요합니다.")
        print("bare run은 금지됩니다 (exit 2).")
        sys.exit(2)


if __name__ == "__main__":
    main()
