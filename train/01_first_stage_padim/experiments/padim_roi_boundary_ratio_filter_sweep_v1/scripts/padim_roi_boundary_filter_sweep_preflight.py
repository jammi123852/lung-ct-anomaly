"""
PaDiM ROI-boundary ratio filter threshold sweep preflight v1.

roi_0_0_patch_ratio 기반 hard filter 및 soft penalty preview가
G3 boundary artifact를 줄이면서 lesion coverage를 유지하는지
read-only post-hoc 분석으로 검증한다.

Usage:
  # dry-run (파일 생성 없음)
  python experiments/padim_roi_boundary_ratio_filter_sweep_v1/scripts/padim_roi_boundary_filter_sweep_preflight.py --dry-run

  # 실제 preflight
  python experiments/padim_roi_boundary_ratio_filter_sweep_v1/scripts/padim_roi_boundary_filter_sweep_preflight.py \\
    --run-preflight --confirm-readonly --confirm-stage1dev-only
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


def roc_auc_score(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    order = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[order]
    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)
    tpr = tp / y_true.sum()
    fpr = fp / (len(y_true) - y_true.sum())
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    return float(np.trapz(tpr, fpr))

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

SOURCE_SCORE_BASE = (
    PROJECT_ROOT
    / "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1"
    / "outputs/scores"
)

STAGE2_HOLDOUT_PATH_KEYWORDS = [
    "stage2_holdout",
    "second-stage-lesion-refiner-v1/datasets",
]

OUT_ROOT = PROJECT_ROOT / "experiments/padim_roi_boundary_ratio_filter_sweep_v1"
REPORT_DIR = OUT_ROOT / "reports"
MANIFEST_OUT = OUT_ROOT / "manifests"
LOG_DIR = OUT_ROOT / "logs"

PROBLEM_PATIENTS = ["LUNG1-086", "LUNG1-386", "LUNG1-399"]

# ---------------------------------------------------------------------------
# Guardrails
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
    "label_used_for_evaluation_only": True,
    "label_used_as_selector": False,
    "adjusted_score_preview_only": True,
    "original_first_stage_score_modified": False,
}

errors = []
warnings_list = []


def log_error(msg):
    errors.append({"type": "ERROR", "message": msg})
    print(f"[ERROR] {msg}", file=sys.stderr)


def log_warn(msg):
    warnings_list.append(msg)
    print(f"[WARN] {msg}", file=sys.stderr)


def log_info(msg):
    print(f"[INFO] {msg}")


# ---------------------------------------------------------------------------
# Filter threshold definitions
# ---------------------------------------------------------------------------
FILTER_THRESHOLDS = [
    ("F0_no_filter", None),
    ("F1_roi_ge_0_50", 0.50),
    ("F2_roi_ge_0_60", 0.60),
    ("F3_roi_ge_0_70", 0.70),
    ("F4_roi_ge_0_80", 0.80),
    ("F5_roi_ge_0_90", 0.90),
    ("F6_roi_ge_0_95", 0.95),
    ("F7_roi_eq_1_00", 0.999),
]

TOP_BANDS = [1, 5, 10, 20]

ADAPTIVE_TOPZ_SELECTORS = [
    ("C2_top5z_pm2", 5, 2),
    ("C3_top5z_pm3", 5, 3),
    ("C4_top10z_pm2", 10, 2),
    ("C5_top10z_pm3", 10, 3),
]

# lambda values for P4
P4_LAMBDAS = [2.0, 5.0, 10.0]

# ---------------------------------------------------------------------------
# Stage2 holdout guard
# ---------------------------------------------------------------------------
def check_no_stage2_holdout(path_str: str):
    for kw in STAGE2_HOLDOUT_PATH_KEYWORDS:
        if kw in str(path_str):
            GUARDRAILS["stage2_holdout_accessed"] = True
            log_error(f"STAGE2 HOLDOUT 접근 감지: {path_str}")
            sys.exit(3)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_manifest():
    check_no_stage2_holdout(MANIFEST_CSV)
    log_info(f"manifest 로드: {MANIFEST_CSV}")
    df = pd.read_csv(MANIFEST_CSV)

    # stage1_dev only
    if "stage_split" in df.columns:
        before = len(df)
        df = df[df["stage_split"] == "stage1_dev"].copy()
        after = len(df)
        log_info(f"stage1_dev 필터: {before:,} → {after:,} rows")
    else:
        log_warn("stage_split 컬럼 없음 — 전체 manifest 사용")

    return df


def load_roi_ratio_from_source_csvs(df):
    """
    manifest 각 candidate에 roi_0_0_patch_ratio, position_bin을
    per-patient CSV에서 join.
    join key: (patient_id, local_z, y0=crop_y0+32, x0=crop_x0+32)
    """
    df = df.copy()
    df["anchor_y0"] = (df["crop_y0"] + 32).astype(int)
    df["anchor_x0"] = (df["crop_x0"] + 32).astype(int)

    patients = df["patient_id"].unique()
    all_src_rows = []

    for pid in patients:
        src_path = SOURCE_SCORE_BASE / "lesion_stage1_dev_by_patient" / f"{pid}.csv"
        check_no_stage2_holdout(src_path)
        if not src_path.exists():
            log_warn(f"per-patient source CSV 없음: {src_path}")
            continue
        try:
            usecols = ["patient_id", "local_z", "y0", "x0",
                       "roi_0_0_patch_ratio", "position_bin"]
            src = pd.read_csv(src_path, usecols=usecols)
            all_src_rows.append(src)
        except Exception as e:
            log_warn(f"source CSV 로드 실패 ({pid}): {e}")

    if not all_src_rows:
        log_error("per-patient source CSV 로드 실패 — boundary ratio 없음")
        return df, False

    src_all = pd.concat(all_src_rows, ignore_index=True)
    src_all = src_all.rename(columns={"y0": "anchor_y0", "x0": "anchor_x0"})
    src_all["anchor_y0"] = src_all["anchor_y0"].astype(int)
    src_all["anchor_x0"] = src_all["anchor_x0"].astype(int)
    src_all = src_all.drop_duplicates(subset=["patient_id", "local_z", "anchor_y0", "anchor_x0"])

    df_merged = df.merge(
        src_all[["patient_id", "local_z", "anchor_y0", "anchor_x0",
                 "roi_0_0_patch_ratio", "position_bin"]],
        on=["patient_id", "local_z", "anchor_y0", "anchor_x0"],
        how="left",
    )

    # position_bin: merged에서 없으면 원본 컬럼 사용
    if "position_bin_x" in df_merged.columns:
        df_merged["position_bin"] = df_merged["position_bin_y"].fillna(df_merged["position_bin_x"])
        df_merged = df_merged.drop(columns=["position_bin_x", "position_bin_y"])

    n_matched = df_merged["roi_0_0_patch_ratio"].notna().sum()
    n_total = len(df_merged)
    log_info(f"roi_0_0_patch_ratio join: {n_matched:,}/{n_total:,} matched ({100*n_matched/n_total:.1f}%)")

    if n_matched < n_total * 0.8:
        log_warn(f"join match rate < 80% — boundary ratio 불완전 ({100*n_matched/n_total:.1f}%)")

    df_merged["boundary_like_ratio"] = 1.0 - df_merged["roi_0_0_patch_ratio"].fillna(0.0)
    return df_merged, n_matched > 0


# ---------------------------------------------------------------------------
# Group assignment
# ---------------------------------------------------------------------------
def assign_groups(df):
    def _assign(row):
        if row["label"] == "positive":
            return "G1_lesion_positive"
        blr = row.get("boundary_like_ratio", np.nan)
        if pd.isna(blr):
            return "G4_interior"
        if blr > 0.0:
            return "G3_boundary_lt50"
        else:
            return "G4_interior"
    df = df.copy()
    df["group"] = df.apply(_assign, axis=1)
    return df


# ---------------------------------------------------------------------------
# Apply filter
# ---------------------------------------------------------------------------
def apply_filter(df, threshold):
    if threshold is None:
        return df.copy()
    return df[df["roi_0_0_patch_ratio"] >= threshold].copy()


# ---------------------------------------------------------------------------
# Per-filter metrics
# ---------------------------------------------------------------------------
def compute_filter_metrics(df_full, df_filtered, filter_name, threshold):
    n_total_full = len(df_full)
    n_filtered = len(df_filtered)
    reduction_rate = 1.0 - n_filtered / n_total_full if n_total_full > 0 else 0.0

    g1_full = df_full[df_full["group"] == "G1_lesion_positive"]
    g1_filt = df_filtered[df_filtered["group"] == "G1_lesion_positive"]
    g3_full = df_full[df_full["group"] == "G3_boundary_lt50"]
    g3_filt = df_filtered[df_filtered["group"] == "G3_boundary_lt50"]
    g4_full = df_full[df_full["group"] == "G4_interior"]
    g4_filt = df_filtered[df_filtered["group"] == "G4_interior"]
    hn_full = df_full[df_full["label"] == "hard_negative"]
    hn_filt = df_filtered[df_filtered["label"] == "hard_negative"]

    # Positive retention
    pos_retention = len(g1_filt) / len(g1_full) if len(g1_full) > 0 else 1.0

    # Lesion patient coverage
    full_patients_with_pos = set(g1_full["patient_id"].unique())
    filt_patients_with_pos = set(g1_filt["patient_id"].unique())
    patient_coverage = len(filt_patients_with_pos) / len(full_patients_with_pos) if full_patients_with_pos else 1.0

    # Lesion slice coverage
    if len(g1_full) > 0:
        slice_ids_full = set(zip(g1_full["patient_id"], g1_full["local_z"]))
        slice_ids_filt = set(zip(g1_filt["patient_id"], g1_filt["local_z"]))
        slice_coverage = len(slice_ids_filt) / len(slice_ids_full) if slice_ids_full else 1.0
    else:
        slice_coverage = 1.0

    # G3 reduction
    g3_reduction = 1.0 - len(g3_filt) / len(g3_full) if len(g3_full) > 0 else 0.0

    # G4 retention
    g4_retention = len(g4_filt) / len(g4_full) if len(g4_full) > 0 else 1.0

    # hard_negative reduction
    hn_reduction = 1.0 - len(hn_filt) / len(hn_full) if len(hn_full) > 0 else 0.0

    # Coverage threshold counts (per patient)
    patient_pos_retention = {}
    for pid in full_patients_with_pos:
        full_pos = len(g1_full[g1_full["patient_id"] == pid])
        filt_pos = len(g1_filt[g1_filt["patient_id"] == pid])
        patient_pos_retention[pid] = filt_pos / full_pos if full_pos > 0 else 1.0

    cov_lt50 = sum(1 for v in patient_pos_retention.values() if v < 0.50)
    cov_lt80 = sum(1 for v in patient_pos_retention.values() if v < 0.80)
    cov_lt95 = sum(1 for v in patient_pos_retention.values() if v < 0.95)

    # Problem patient coverage
    pp_coverage = {}
    for pp in PROBLEM_PATIENTS:
        full_pp = len(g1_full[g1_full["patient_id"] == pp])
        filt_pp = len(g1_filt[g1_filt["patient_id"] == pp])
        pp_coverage[pp] = filt_pp / full_pp if full_pp > 0 else float("nan")

    return {
        "filter_name": filter_name,
        "threshold": threshold,
        "n_candidates_full": n_total_full,
        "n_candidates_filtered": n_filtered,
        "candidate_reduction_rate": round(reduction_rate, 4),
        "g1_lesion_full": len(g1_full),
        "g1_lesion_filtered": len(g1_filt),
        "g1_positive_retention": round(pos_retention, 4),
        "lesion_patient_coverage": round(patient_coverage, 4),
        "lesion_patient_n_full": len(full_patients_with_pos),
        "lesion_patient_n_filtered": len(filt_patients_with_pos),
        "lesion_slice_coverage": round(slice_coverage, 4),
        "g3_boundary_full": len(g3_full),
        "g3_boundary_filtered": len(g3_filt),
        "g3_boundary_reduction": round(g3_reduction, 4),
        "g4_interior_full": len(g4_full),
        "g4_interior_filtered": len(g4_filt),
        "g4_interior_retention": round(g4_retention, 4),
        "hard_negative_full": len(hn_full),
        "hard_negative_filtered": len(hn_filt),
        "hard_negative_reduction": round(hn_reduction, 4),
        "coverage_lt50_patients": cov_lt50,
        "coverage_lt80_patients": cov_lt80,
        "coverage_lt95_patients": cov_lt95,
        "pp_LUNG1_086_retention": round(pp_coverage.get("LUNG1-086", float("nan")), 4),
        "pp_LUNG1_386_retention": round(pp_coverage.get("LUNG1-386", float("nan")), 4),
        "pp_LUNG1_399_retention": round(pp_coverage.get("LUNG1-399", float("nan")), 4),
    }


# ---------------------------------------------------------------------------
# Top score band composition
# ---------------------------------------------------------------------------
def compute_top_band_composition(df_full, df_filtered, filter_name, threshold):
    rows = []
    for band_pct in TOP_BANDS:
        n_band = max(1, int(len(df_filtered) * band_pct / 100))
        band = df_filtered.nlargest(n_band, "first_stage_score")

        g1 = (band["group"] == "G1_lesion_positive").sum()
        g3 = (band["group"] == "G3_boundary_lt50").sum()
        g4 = (band["group"] == "G4_interior").sum()
        n = len(band)

        # Lesion patient/slice in band
        band_pos = band[band["group"] == "G1_lesion_positive"]
        full_pos_patients = set(df_full[df_full["group"] == "G1_lesion_positive"]["patient_id"].unique())
        band_pos_patients = set(band_pos["patient_id"].unique())
        full_pos_slices = set(zip(
            df_full[df_full["group"] == "G1_lesion_positive"]["patient_id"],
            df_full[df_full["group"] == "G1_lesion_positive"]["local_z"]
        ))
        band_pos_slices = set(zip(band_pos["patient_id"], band_pos["local_z"]))

        pp_in_band = {pp: pp in band_pos_patients for pp in PROBLEM_PATIENTS}

        rows.append({
            "filter_name": filter_name,
            "threshold": threshold,
            "top_band_pct": band_pct,
            "n_band": n,
            "G1_lesion_count": g1,
            "G3_boundary_lt50_count": g3,
            "G4_interior_count": g4,
            "G1_pct": round(100 * g1 / n, 2) if n > 0 else 0,
            "G3_pct": round(100 * g3 / n, 2) if n > 0 else 0,
            "G4_pct": round(100 * g4 / n, 2) if n > 0 else 0,
            "lesion_patient_coverage": round(len(band_pos_patients) / len(full_pos_patients), 4) if full_pos_patients else 0,
            "lesion_slice_coverage": round(len(band_pos_slices) / len(full_pos_slices), 4) if full_pos_slices else 0,
            "pp_LUNG1_086_in_band": pp_in_band.get("LUNG1-086", False),
            "pp_LUNG1_386_in_band": pp_in_band.get("LUNG1-386", False),
            "pp_LUNG1_399_in_band": pp_in_band.get("LUNG1-399", False),
        })
    return rows


# ---------------------------------------------------------------------------
# Adaptive top-z selector
# ---------------------------------------------------------------------------
def compute_slice_top20_mean(patient_df):
    """patient 단위로 first_stage_score 상위 20개 평균"""
    scores = patient_df["first_stage_score"].values
    top_k = scores[np.argsort(scores)[-min(20, len(scores)):]]
    return float(np.mean(top_k))


def adaptive_topz_selector(df_filtered, top_n_z, pm_slices):
    """
    Per-patient 상위 top_n_z local_z를 선택하고,
    각 z에서 ±pm_slices 범위의 후보를 선택한다.
    """
    selected_ids = []
    for pid, pdf in df_filtered.groupby("patient_id"):
        # slice_top20_mean per z
        z_scores = pdf.groupby("local_z").apply(
            lambda g: pd.Series({"slice_top20_mean": compute_slice_top20_mean(g)})
        ).reset_index()

        top_zs = z_scores.nlargest(top_n_z, "slice_top20_mean")["local_z"].values

        selected_zs = set()
        for z in top_zs:
            for dz in range(-pm_slices, pm_slices + 1):
                selected_zs.add(z + dz)

        mask = pdf["local_z"].isin(selected_zs)
        selected_ids.extend(pdf[mask].index.tolist())

    return df_filtered.loc[selected_ids].copy() if selected_ids else df_filtered.iloc[0:0].copy()


def compute_adaptive_topz_metrics(df_full, df_filter_base, filter_name, threshold, selector_name, top_n_z, pm_slices):
    df_selected = adaptive_topz_selector(df_filter_base, top_n_z, pm_slices)

    g1_full = df_full[df_full["group"] == "G1_lesion_positive"]
    g1_sel = df_selected[df_selected["group"] == "G1_lesion_positive"]
    g3_sel = df_selected[df_selected["group"] == "G3_boundary_lt50"]
    hn_sel = df_selected[df_selected["label"] == "hard_negative"]
    hn_full = df_full[df_full["label"] == "hard_negative"]

    pos_patients_full = set(g1_full["patient_id"].unique())
    pos_patients_sel = set(g1_sel["patient_id"].unique())

    full_pos_slices = set(zip(g1_full["patient_id"], g1_full["local_z"]))
    sel_pos_slices = set(zip(g1_sel["patient_id"], g1_sel["local_z"]))

    patient_coverage = len(pos_patients_sel) / len(pos_patients_full) if pos_patients_full else 1.0
    slice_coverage = len(sel_pos_slices) / len(full_pos_slices) if full_pos_slices else 1.0
    pos_retention = len(g1_sel) / len(g1_full) if len(g1_full) > 0 else 1.0
    g3_reduction = 1.0 - len(g3_sel) / max(1, len(df_full[df_full["group"] == "G3_boundary_lt50"]))
    hn_reduction = 1.0 - len(hn_sel) / len(hn_full) if len(hn_full) > 0 else 0.0

    # per-patient coverage <50%
    pp_retention = {}
    for pid in pos_patients_full:
        full_pp = len(g1_full[g1_full["patient_id"] == pid])
        sel_pp = len(g1_sel[g1_sel["patient_id"] == pid])
        pp_retention[pid] = sel_pp / full_pp if full_pp > 0 else 1.0
    cov_lt50 = sum(1 for v in pp_retention.values() if v < 0.50)

    # positive z in top-z check: per patient, does at least one positive z appear?
    pos_z_in_topz_count = 0
    g3_dominant_count = 0
    total_patients = len(df_filter_base["patient_id"].unique())

    for pid, pdf in df_filter_base.groupby("patient_id"):
        z_scores = pdf.groupby("local_z").apply(
            lambda g: pd.Series({"slice_top20_mean": compute_slice_top20_mean(g)})
        ).reset_index()
        top_zs_sorted = z_scores.nlargest(top_n_z, "slice_top20_mean")["local_z"].values

        if len(top_zs_sorted) == 0:
            continue

        # positive z가 top-z에 들어오는지
        pos_zs = set(pdf[pdf["group"] == "G1_lesion_positive"]["local_z"].unique())
        if pos_zs and any(z in pos_zs for z in top_zs_sorted):
            pos_z_in_topz_count += 1

        # top-z가 G3 dominant인지 (top-z 중 가장 높은 slice가 G3 다수인지)
        if len(top_zs_sorted) > 0:
            top1_z = top_zs_sorted[0]
            top1_slice = pdf[pdf["local_z"] == top1_z]
            if len(top1_slice) > 0:
                g3_in_top1 = (top1_slice["group"] == "G3_boundary_lt50").sum()
                if g3_in_top1 / len(top1_slice) > 0.5:
                    g3_dominant_count += 1

    patients_with_pos = len(pos_patients_full)
    pos_z_in_topz_rate = pos_z_in_topz_count / patients_with_pos if patients_with_pos > 0 else 0.0
    g3_dominant_rate = g3_dominant_count / total_patients if total_patients > 0 else 0.0

    pp_cov = {}
    for pp in PROBLEM_PATIENTS:
        full_pp = len(g1_full[g1_full["patient_id"] == pp])
        sel_pp = len(g1_sel[g1_sel["patient_id"] == pp])
        pp_cov[pp] = round(sel_pp / full_pp, 4) if full_pp > 0 else float("nan")

    return {
        "filter_name": filter_name,
        "threshold": threshold,
        "selector_name": selector_name,
        "top_n_z": top_n_z,
        "pm_slices": pm_slices,
        "n_candidates_in_filter_base": len(df_filter_base),
        "n_selected": len(df_selected),
        "candidate_reduction_rate": round(1.0 - len(df_selected) / max(1, len(df_full)), 4),
        "lesion_patient_coverage": round(patient_coverage, 4),
        "lesion_slice_coverage": round(slice_coverage, 4),
        "positive_crop_retention": round(pos_retention, 4),
        "g3_boundary_reduction": round(g3_reduction, 4),
        "hard_negative_reduction": round(hn_reduction, 4),
        "coverage_lt50_patients": cov_lt50,
        "pos_z_in_topz_rate": round(pos_z_in_topz_rate, 4),
        "g3_dominant_top1z_rate": round(g3_dominant_rate, 4),
        "pp_LUNG1_086_retention": pp_cov.get("LUNG1-086", float("nan")),
        "pp_LUNG1_386_retention": pp_cov.get("LUNG1-386", float("nan")),
        "pp_LUNG1_399_retention": pp_cov.get("LUNG1-399", float("nan")),
    }


# ---------------------------------------------------------------------------
# Penalty preview
# ---------------------------------------------------------------------------
def compute_penalty_preview(df_full, df_base, filter_name, threshold):
    """
    기존 first_stage_score는 수정하지 않고 preview score만 별도 컬럼으로 계산.
    AUROC, top-band composition 변화를 확인한다.
    """
    df = df_base.copy()
    r = df["roi_0_0_patch_ratio"].clip(0.0, 1.0)
    blr = df["boundary_like_ratio"].clip(0.0, 1.0)
    s = df["first_stage_score"]

    df["adjusted_score_p1"] = s * r
    df["adjusted_score_p2"] = s * np.sqrt(r)
    df["adjusted_score_p3"] = s * r.clip(lower=0.5)
    for lam in P4_LAMBDAS:
        df[f"adjusted_score_p4_lam{int(lam)}"] = s - lam * blr

    g1 = df["group"] == "G1_lesion_positive"
    g_npos = ~g1
    y_true = g1.astype(int).values

    def safe_auc(scores):
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            return float("nan")
        try:
            return float(roc_auc_score(y_true, scores))
        except Exception:
            return float("nan")

    penalty_cols = (
        ["first_stage_score", "adjusted_score_p1", "adjusted_score_p2", "adjusted_score_p3"]
        + [f"adjusted_score_p4_lam{int(lam)}" for lam in P4_LAMBDAS]
    )

    rows = []
    g1_full = df_full[df_full["group"] == "G1_lesion_positive"]
    full_pos_patients = set(g1_full["patient_id"].unique())
    full_pos_slices = set(zip(g1_full["patient_id"], g1_full["local_z"]))

    for col in penalty_cols:
        if col not in df.columns:
            continue
        auc = safe_auc(df[col].values)

        # top 1/5/10% band G3 ratio
        band_rows = {}
        for band_pct in [1, 5, 10]:
            n_band = max(1, int(len(df) * band_pct / 100))
            band = df.nlargest(n_band, col)
            g3_pct = round(100 * (band["group"] == "G3_boundary_lt50").sum() / max(1, len(band)), 2)
            g1_pct = round(100 * (band["group"] == "G1_lesion_positive").sum() / max(1, len(band)), 2)
            band_rows[f"top{band_pct}pct_G3_pct"] = g3_pct
            band_rows[f"top{band_pct}pct_G1_pct"] = g1_pct

        # adaptive top-z C2/C3 coverage preview (using C2: top5z pm2)
        c2_result = adaptive_topz_selector(df, top_n_z=5, pm_slices=2)
        c2_pos = c2_result[c2_result["group"] == "G1_lesion_positive"]
        c2_pat_cov = len(set(c2_pos["patient_id"])) / len(full_pos_patients) if full_pos_patients else 0.0

        # problem patient coverage in top5% band
        top5_n = max(1, int(len(df) * 5 / 100))
        top5 = df.nlargest(top5_n, col)
        top5_pos = top5[top5["group"] == "G1_lesion_positive"]
        pp_cov = {pp: pp in set(top5_pos["patient_id"]) for pp in PROBLEM_PATIENTS}

        row = {
            "filter_name": filter_name,
            "threshold": threshold,
            "score_col": col,
            "g1_vs_g3_auroc": round(auc, 4) if not np.isnan(auc) else None,
            **band_rows,
            "adaptive_c2_patient_coverage": round(c2_pat_cov, 4),
            "pp_LUNG1_086_in_top5pct": pp_cov.get("LUNG1-086", False),
            "pp_LUNG1_386_in_top5pct": pp_cov.get("LUNG1-386", False),
            "pp_LUNG1_399_in_top5pct": pp_cov.get("LUNG1-399", False),
        }
        rows.append(row)

    return rows, df


# ---------------------------------------------------------------------------
# Problem patient audit
# ---------------------------------------------------------------------------
def compute_problem_patient_audit(df_full, filter_results, topz_results):
    """문제 환자 3명의 coverage 변화를 filter + selector 조합별로 정리"""
    rows = []
    g1_full = df_full[df_full["group"] == "G1_lesion_positive"]

    for pp in PROBLEM_PATIENTS:
        full_pos = len(g1_full[g1_full["patient_id"] == pp])
        for fr in filter_results:
            rows.append({
                "patient_id": pp,
                "analysis_type": "filter_only",
                "filter_name": fr["filter_name"],
                "threshold": fr["threshold"],
                "selector_name": None,
                "full_positive": full_pos,
                "filtered_positive_retention": fr.get(f"pp_{pp.replace('-', '_')}_retention", float("nan")),
            })
        for tr in topz_results:
            rows.append({
                "patient_id": pp,
                "analysis_type": "filter+topz",
                "filter_name": tr["filter_name"],
                "threshold": tr["threshold"],
                "selector_name": tr["selector_name"],
                "full_positive": full_pos,
                "filtered_positive_retention": tr.get(f"pp_{pp.replace('-', '_')}_retention", float("nan")),
            })
    return rows


# ---------------------------------------------------------------------------
# Recommended manifest preview
# ---------------------------------------------------------------------------
def build_recommended_manifest_preview(df_full, best_filter_threshold, best_selector_name, best_top_n_z, best_pm_slices):
    df_filtered = apply_filter(df_full, best_filter_threshold)
    df_selected = adaptive_topz_selector(df_filtered, best_top_n_z, best_pm_slices)
    out_cols = [c for c in [
        "candidate_id", "patient_id", "safe_id", "stage_split",
        "local_z", "slice_index", "crop_y0", "crop_x0", "crop_y1", "crop_x1",
        "first_stage_score", "label", "group",
        "roi_0_0_patch_ratio", "boundary_like_ratio", "position_bin",
    ] if c in df_selected.columns]
    return df_selected[out_cols].copy()


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
def determine_verdict(filter_results, topz_results):
    pass_candidates = []
    partial_pass = []

    for fr in filter_results:
        if fr["filter_name"] == "F0_no_filter":
            continue
        if (
            fr["lesion_patient_coverage"] >= 1.0
            and fr["coverage_lt50_patients"] == 0
            and fr["lesion_slice_coverage"] >= 0.97
            and fr["g1_positive_retention"] >= 0.97
            and fr["g3_boundary_reduction"] > 0.0
        ):
            pass_candidates.append(fr["filter_name"])
        elif fr["lesion_patient_coverage"] >= 0.95 and fr["g3_boundary_reduction"] > 0.0:
            partial_pass.append(fr["filter_name"])

    if pass_candidates:
        return "PASS_CANDIDATE", pass_candidates
    elif partial_pass:
        return "PARTIAL_PASS_EXPLORATORY", partial_pass
    else:
        return "FAIL", []


# ---------------------------------------------------------------------------
# DRY-RUN
# ---------------------------------------------------------------------------
def dry_run():
    print("[DRY-RUN] 입력 파일 및 설정 확인")

    # manifest
    if MANIFEST_CSV.exists():
        print(f"[OK] manifest: {MANIFEST_CSV}")
    else:
        print(f"[MISSING] manifest: {MANIFEST_CSV}")

    # source CSV dir
    src_dir = SOURCE_SCORE_BASE / "lesion_stage1_dev_by_patient"
    if src_dir.exists():
        n_src = len(list(src_dir.glob("*.csv")))
        print(f"[OK] per-patient source CSV dir: {src_dir} ({n_src}개)")
        if n_src > 0:
            sample = next(src_dir.glob("*.csv"))
            import pandas as pd
            sample_df = pd.read_csv(sample, nrows=1)
            cols = list(sample_df.columns)
            has_ratio = "roi_0_0_patch_ratio" in cols
            has_pos = "position_bin" in cols
            print(f"[CHECK] roi_0_0_patch_ratio: {'OK' if has_ratio else 'MISSING'}")
            print(f"[CHECK] position_bin: {'OK' if has_pos else 'MISSING'}")
    else:
        print(f"[MISSING] source CSV dir: {src_dir}")

    # stage2 holdout guard
    for kw in STAGE2_HOLDOUT_PATH_KEYWORDS:
        if kw in str(MANIFEST_CSV):
            print(f"[FAIL] stage2_holdout 접근 감지: {MANIFEST_CSV}")
        else:
            print(f"[OK] stage2_holdout 없음: {kw}")

    # output root 충돌
    if OUT_ROOT.exists() and any(OUT_ROOT.iterdir()):
        print(f"[WARN] output root 이미 존재: {OUT_ROOT}")
    else:
        print(f"[OK] output root 신규: {OUT_ROOT}")

    print("[DRY-RUN] 완료 — 파일 생성 없음")
    sys.exit(0)


# ---------------------------------------------------------------------------
# MAIN PREFLIGHT
# ---------------------------------------------------------------------------
def run_preflight():
    # 1. Load manifest
    df = load_manifest()
    log_info(f"manifest rows: {len(df):,}")

    # 2. Join roi_0_0_patch_ratio
    df, roi_joined = load_roi_ratio_from_source_csvs(df)
    if not roi_joined:
        log_error("roi_0_0_patch_ratio join 실패 — 분석 불가")
        return None

    # 3. Assign groups
    df = assign_groups(df)
    log_info(f"group distribution:\n{df['group'].value_counts().to_string()}")

    # 4. Sanity: roi_0_0_patch_ratio distribution
    log_info(f"roi_0_0_patch_ratio stats:\n{df['roi_0_0_patch_ratio'].describe().round(4).to_string()}")

    # 5. Per-filter analysis
    filter_results = []
    band_rows_all = []
    topz_results = []
    penalty_rows_all = []
    penalty_dfs = {}

    for filter_name, threshold in FILTER_THRESHOLDS:
        log_info(f"--- Filter: {filter_name} (threshold={threshold}) ---")
        df_filtered = apply_filter(df, threshold)

        # 5a. Filter metrics
        fm = compute_filter_metrics(df, df_filtered, filter_name, threshold)
        filter_results.append(fm)
        log_info(f"  n_filtered={fm['n_candidates_filtered']:,} "
                 f"pos_ret={fm['g1_positive_retention']:.3f} "
                 f"pat_cov={fm['lesion_patient_coverage']:.3f} "
                 f"g3_red={fm['g3_boundary_reduction']:.3f}")

        # 5b. Top score band composition
        band_rows = compute_top_band_composition(df, df_filtered, filter_name, threshold)
        band_rows_all.extend(band_rows)

        # 5c. Adaptive top-z (using filtered base)
        for sel_name, top_n_z, pm_slices in ADAPTIVE_TOPZ_SELECTORS:
            tz_m = compute_adaptive_topz_metrics(
                df, df_filtered, filter_name, threshold, sel_name, top_n_z, pm_slices
            )
            topz_results.append(tz_m)

        # 5d. Penalty preview
        pen_rows, df_pen = compute_penalty_preview(df, df_filtered, filter_name, threshold)
        penalty_rows_all.extend(pen_rows)
        penalty_dfs[(filter_name, threshold)] = df_pen

    # 6. Problem patient audit
    pp_rows = compute_problem_patient_audit(df, filter_results, topz_results)

    # 7. Verdict
    verdict, verdict_candidates = determine_verdict(filter_results, topz_results)
    log_info(f"판정: {verdict} ({verdict_candidates})")

    # 8. Recommended manifest preview
    # 가장 좋은 filter는 lesion coverage 100% 유지하면서 g3 reduction이 가장 큰 것
    best_filter = None
    best_g3_red = -1.0
    for fr in filter_results:
        if fr["filter_name"] == "F0_no_filter":
            continue
        if (fr["lesion_patient_coverage"] >= 1.0 and
                fr["coverage_lt50_patients"] == 0 and
                fr["g1_positive_retention"] >= 0.97):
            if fr["g3_boundary_reduction"] > best_g3_red:
                best_g3_red = fr["g3_boundary_reduction"]
                best_filter = fr

    if best_filter is None:
        # coverage 조건을 완화해서 가장 coverage 높은 것
        best_filter = max(filter_results, key=lambda x: (x["lesion_patient_coverage"], x["g3_boundary_reduction"]))

    recommended_df = build_recommended_manifest_preview(
        df, best_filter["threshold"], "C2_top5z_pm2", 5, 2
    )

    return {
        "filter_results": filter_results,
        "band_rows_all": band_rows_all,
        "topz_results": topz_results,
        "penalty_rows_all": penalty_rows_all,
        "pp_rows": pp_rows,
        "recommended_df": recommended_df,
        "verdict": verdict,
        "verdict_candidates": verdict_candidates,
        "best_filter": best_filter,
        "df_full": df,
    }


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
def save_outputs(results):
    for d in [OUT_ROOT, REPORT_DIR, MANIFEST_OUT, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    planned_outputs = [
        MANIFEST_OUT / "filter_threshold_summary.csv",
        MANIFEST_OUT / "filter_top_score_band_composition.csv",
        MANIFEST_OUT / "filter_adaptive_topz_summary.csv",
        MANIFEST_OUT / "penalty_preview_summary.csv",
        MANIFEST_OUT / "problem_patient_filter_audit.csv",
        MANIFEST_OUT / "recommended_manifest_preview.csv",
        REPORT_DIR / "padim_roi_boundary_filter_sweep_report.md",
        REPORT_DIR / "padim_roi_boundary_filter_sweep_summary.json",
        LOG_DIR / "errors.csv",
        OUT_ROOT / "DONE.json",
    ]
    existing = [p for p in planned_outputs if p.exists()]
    if existing:
        GUARDRAILS["output_overwrite"] = True
        raise RuntimeError(f"[ABORT] output overwrite risk: {existing}")

    filter_results = results["filter_results"]
    band_rows_all = results["band_rows_all"]
    topz_results = results["topz_results"]
    penalty_rows_all = results["penalty_rows_all"]
    pp_rows = results["pp_rows"]
    recommended_df = results["recommended_df"]
    verdict = results["verdict"]
    verdict_candidates = results["verdict_candidates"]
    best_filter = results["best_filter"]

    # filter_threshold_summary.csv
    pd.DataFrame(filter_results).to_csv(
        MANIFEST_OUT / "filter_threshold_summary.csv", index=False
    )
    log_info("filter_threshold_summary.csv 저장")

    # filter_top_score_band_composition.csv
    pd.DataFrame(band_rows_all).to_csv(
        MANIFEST_OUT / "filter_top_score_band_composition.csv", index=False
    )
    log_info("filter_top_score_band_composition.csv 저장")

    # filter_adaptive_topz_summary.csv
    pd.DataFrame(topz_results).to_csv(
        MANIFEST_OUT / "filter_adaptive_topz_summary.csv", index=False
    )
    log_info("filter_adaptive_topz_summary.csv 저장")

    # penalty_preview_summary.csv
    pd.DataFrame(penalty_rows_all).to_csv(
        MANIFEST_OUT / "penalty_preview_summary.csv", index=False
    )
    log_info("penalty_preview_summary.csv 저장")

    # problem_patient_filter_audit.csv
    pd.DataFrame(pp_rows).to_csv(
        MANIFEST_OUT / "problem_patient_filter_audit.csv", index=False
    )
    log_info("problem_patient_filter_audit.csv 저장")

    # recommended_manifest_preview.csv
    recommended_df.to_csv(
        MANIFEST_OUT / "recommended_manifest_preview.csv", index=False
    )
    log_info("recommended_manifest_preview.csv 저장")

    # errors.csv
    pd.DataFrame(errors).to_csv(LOG_DIR / "errors.csv", index=False)
    log_info("errors.csv 저장")

    # --- Report markdown ---
    f0 = next((fr for fr in filter_results if fr["filter_name"] == "F0_no_filter"), {})
    report_lines = [
        "# PaDiM ROI Boundary Ratio Filter Sweep Preflight Report v1",
        "",
        f"## 판정: {verdict}",
        f"PASS_CANDIDATE 필터: {verdict_candidates}",
        f"추천 filter: {best_filter['filter_name']} (threshold={best_filter['threshold']})",
        "",
        "## 기본 현황 (F0 no filter)",
        f"- 전체 candidate: {f0.get('n_candidates_full', 'N/A'):,}" if isinstance(f0.get('n_candidates_full'), int) else f"- 전체 candidate: {f0.get('n_candidates_full', 'N/A')}",
        f"- G1 lesion positive: {f0.get('g1_lesion_full', 'N/A')}",
        f"- G3 boundary lt50: {f0.get('g3_boundary_full', 'N/A')}",
        f"- G4 interior: {f0.get('g4_interior_full', 'N/A')}",
        f"- lesion patient coverage: {f0.get('lesion_patient_coverage', 'N/A')}",
        f"- coverage <50% patients: {f0.get('coverage_lt50_patients', 'N/A')}",
        "",
        "## Filter Threshold Summary",
        "| filter | threshold | n_filtered | pos_retention | pat_cov | slice_cov | g3_reduction | cov_lt50 |",
        "|--------|-----------|-----------|---------------|---------|-----------|--------------|----------|",
    ]
    for fr in filter_results:
        report_lines.append(
            f"| {fr['filter_name']} | {fr['threshold']} | {fr['n_candidates_filtered']:,} "
            f"| {fr['g1_positive_retention']:.4f} | {fr['lesion_patient_coverage']:.4f} "
            f"| {fr['lesion_slice_coverage']:.4f} | {fr['g3_boundary_reduction']:.4f} "
            f"| {fr['coverage_lt50_patients']} |"
        )

    report_lines += [
        "",
        "## Top 1% Band G3 ratio (Filter comparison)",
        "| filter | G3_pct@top1% | G1_pct@top1% | G3_pct@top5% | G1_pct@top5% |",
        "|--------|-------------|-------------|-------------|-------------|",
    ]
    for filter_name, _ in FILTER_THRESHOLDS:
        top1 = next((b for b in band_rows_all if b["filter_name"] == filter_name and b["top_band_pct"] == 1), {})
        top5 = next((b for b in band_rows_all if b["filter_name"] == filter_name and b["top_band_pct"] == 5), {})
        report_lines.append(
            f"| {filter_name} | {top1.get('G3_pct', 'N/A')} | {top1.get('G1_pct', 'N/A')} "
            f"| {top5.get('G3_pct', 'N/A')} | {top5.get('G1_pct', 'N/A')} |"
        )

    report_lines += [
        "",
        "## Problem Patient Coverage",
        "| patient | F0 | F2_0.60 | F3_0.70 | F4_0.80 | F5_0.90 |",
        "|---------|----|---------|---------|---------|---------| ",
    ]
    for pp in PROBLEM_PATIENTS:
        pp_key = f"pp_{pp.replace('-', '_')}_retention"
        vals = {fr["filter_name"]: fr.get(pp_key, float("nan")) for fr in filter_results}
        report_lines.append(
            f"| {pp} | {vals.get('F0_no_filter', 'N/A')} "
            f"| {vals.get('F2_roi_ge_0_60', 'N/A')} "
            f"| {vals.get('F3_roi_ge_0_70', 'N/A')} "
            f"| {vals.get('F4_roi_ge_0_80', 'N/A')} "
            f"| {vals.get('F5_roi_ge_0_90', 'N/A')} |"
        )

    report_lines += [
        "",
        "## Adaptive Top-Z C2 (top5z pm2) Coverage by Filter",
        "| filter | lesion_pat_cov | lesion_slice_cov | pos_z_in_topz | g3_dom_rate | cov_lt50 |",
        "|--------|---------------|-----------------|---------------|-------------|----------|",
    ]
    for filter_name, _ in FILTER_THRESHOLDS:
        tz = next((t for t in topz_results if t["filter_name"] == filter_name and t["selector_name"] == "C2_top5z_pm2"), {})
        report_lines.append(
            f"| {filter_name} | {tz.get('lesion_patient_coverage', 'N/A')} "
            f"| {tz.get('lesion_slice_coverage', 'N/A')} "
            f"| {tz.get('pos_z_in_topz_rate', 'N/A')} "
            f"| {tz.get('g3_dominant_top1z_rate', 'N/A')} "
            f"| {tz.get('coverage_lt50_patients', 'N/A')} |"
        )

    report_lines += [
        "",
        "## Penalty Preview (G1 vs G3 AUROC)",
        "| filter | score_col | auroc | top1%_G3 | top5%_G3 | c2_pat_cov |",
        "|--------|-----------|-------|----------|----------|------------|",
    ]
    for pr in penalty_rows_all:
        report_lines.append(
            f"| {pr['filter_name']} | {pr['score_col']} | {pr.get('g1_vs_g3_auroc', 'N/A')} "
            f"| {pr.get('top1pct_G3_pct', 'N/A')} | {pr.get('top5pct_G3_pct', 'N/A')} "
            f"| {pr.get('adaptive_c2_patient_coverage', 'N/A')} |"
        )

    report_lines += [
        "",
        "## Guardrails",
        f"- stage2_holdout_accessed: {GUARDRAILS['stage2_holdout_accessed']}",
        f"- model_forward_executed: {GUARDRAILS['model_forward_executed']}",
        f"- training_executed: {GUARDRAILS['training_executed']}",
        f"- existing_artifact_modified: {GUARDRAILS['existing_artifact_modified']}",
        f"- original_first_stage_score_modified: {GUARDRAILS['original_first_stage_score_modified']}",
        f"- adjusted_score_preview_only: {GUARDRAILS['adjusted_score_preview_only']}",
        "",
        "## Critical Errors",
        f"- count: {len([e for e in errors if e['type'] == 'ERROR'])}",
    ]
    for e in errors:
        if e["type"] == "ERROR":
            report_lines.append(f"  - {e['message']}")

    (REPORT_DIR / "padim_roi_boundary_filter_sweep_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    log_info("report.md 저장")

    # --- Summary JSON ---
    summary = {
        "verdict": verdict,
        "verdict_pass_candidates": verdict_candidates,
        "best_filter": best_filter["filter_name"] if best_filter else None,
        "best_filter_threshold": best_filter["threshold"] if best_filter else None,
        "critical_error_count": len([e for e in errors if e["type"] == "ERROR"]),
        "warning_count": len(warnings_list),
        "guardrails": GUARDRAILS,
        "filter_summary": [
            {
                "filter_name": fr["filter_name"],
                "threshold": fr["threshold"],
                "n_filtered": fr["n_candidates_filtered"],
                "pos_retention": fr["g1_positive_retention"],
                "lesion_patient_coverage": fr["lesion_patient_coverage"],
                "lesion_slice_coverage": fr["lesion_slice_coverage"],
                "g3_boundary_reduction": fr["g3_boundary_reduction"],
                "coverage_lt50_patients": fr["coverage_lt50_patients"],
                "pp_LUNG1_086": fr["pp_LUNG1_086_retention"],
                "pp_LUNG1_386": fr["pp_LUNG1_386_retention"],
                "pp_LUNG1_399": fr["pp_LUNG1_399_retention"],
            }
            for fr in filter_results
        ],
    }
    (REPORT_DIR / "padim_roi_boundary_filter_sweep_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log_info("summary.json 저장")

    # --- DONE.json ---
    done = {
        "status": "DONE",
        "verdict": verdict,
        "critical_error_count": len([e for e in errors if e["type"] == "ERROR"]),
        "guardrails": GUARDRAILS,
        "outputs": [
            str(MANIFEST_OUT / "filter_threshold_summary.csv"),
            str(MANIFEST_OUT / "filter_top_score_band_composition.csv"),
            str(MANIFEST_OUT / "filter_adaptive_topz_summary.csv"),
            str(MANIFEST_OUT / "penalty_preview_summary.csv"),
            str(MANIFEST_OUT / "problem_patient_filter_audit.csv"),
            str(MANIFEST_OUT / "recommended_manifest_preview.csv"),
            str(REPORT_DIR / "padim_roi_boundary_filter_sweep_report.md"),
            str(REPORT_DIR / "padim_roi_boundary_filter_sweep_summary.json"),
            str(LOG_DIR / "errors.csv"),
        ],
    }
    (OUT_ROOT / "DONE.json").write_text(
        json.dumps(done, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log_info("DONE.json 저장")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-preflight", action="store_true")
    parser.add_argument("--confirm-readonly", action="store_true")
    parser.add_argument("--confirm-stage1dev-only", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    if not (args.run_preflight and args.confirm_readonly and args.confirm_stage1dev_only):
        print("[ABORT] bare run 금지. --run-preflight --confirm-readonly --confirm-stage1dev-only 필요", file=sys.stderr)
        sys.exit(2)

    # Validate guardrails at exit
    try:
        results = run_preflight()
        if results is None:
            log_error("preflight 실패 — 결과 없음")
            sys.exit(1)

        # Final guardrail check
        assert not GUARDRAILS["stage2_holdout_accessed"], "STAGE2 HOLDOUT ACCESSED"
        assert not GUARDRAILS["model_forward_executed"], "MODEL FORWARD"
        assert not GUARDRAILS["training_executed"], "TRAINING"
        assert not GUARDRAILS["existing_artifact_modified"], "ARTIFACT MODIFIED"
        assert not GUARDRAILS["original_first_stage_score_modified"], "SCORE MODIFIED"

        save_outputs(results)

        n_errors = len([e for e in errors if e["type"] == "ERROR"])
        print(f"\n[DONE] 판정: {results['verdict']} | critical errors: {n_errors}")
        print(f"[DONE] 추천 filter: {results['best_filter']['filter_name']} (threshold={results['best_filter']['threshold']})")

        if n_errors > 0:
            sys.exit(1)

    except AssertionError as ae:
        log_error(f"GUARDRAIL VIOLATION: {ae}")
        sys.exit(3)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
