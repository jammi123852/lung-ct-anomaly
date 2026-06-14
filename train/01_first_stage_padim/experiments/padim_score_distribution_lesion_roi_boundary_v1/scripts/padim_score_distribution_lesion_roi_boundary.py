"""
PaDiM score distribution audit by lesion and ROI-boundary groups v1.

first_stage_score 분포를 4개 그룹(G1 lesion / G2 ROI경계>=50% / G3 경계<50% / G4 내부)으로 나누어
top-z selector 실패 원인을 확인한다.

Usage:
  # dry-run
  python experiments/padim_score_distribution_lesion_roi_boundary_v1/scripts/padim_score_distribution_lesion_roi_boundary.py --dry-run

  # 실제 분석
  python experiments/padim_score_distribution_lesion_roi_boundary_v1/scripts/padim_score_distribution_lesion_roi_boundary.py \\
    --run-audit --confirm-readonly --confirm-stage1dev-only
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

# per-patient source score CSV base dir
SOURCE_SCORE_BASE = (
    PROJECT_ROOT
    / "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1"
    / "outputs/scores"
)

STAGE2_HOLDOUT_PATH_KEYWORDS = [
    "stage2_holdout",
    "second-stage-lesion-refiner-v1/datasets",
]

OUT_ROOT = PROJECT_ROOT / "experiments/padim_score_distribution_lesion_roi_boundary_v1"
REPORT_DIR = OUT_ROOT / "reports"
MANIFEST_DIR = OUT_ROOT / "manifests"
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
    "label_used_for_group_analysis_only": True,
    "label_used_as_selector": False,
    "boundary_available": False,
    "boundary_source": "unknown",
}

errors = []


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
                log_error(f"stage2_holdout 키워드 포함: {ip}")
    return not GUARDRAILS["stage2_holdout_accessed"]


# ---------------------------------------------------------------------------
# ROI boundary ratio: join from per-patient source CSVs
# join key: patient_id, local_z, y0 = crop_y0+32, x0 = crop_x0+32
# ---------------------------------------------------------------------------
def load_roi_ratio_from_source_csvs(df):
    """
    manifest 각 candidate에 roi_0_0_patch_ratio를 per-patient CSV에서 join.
    join key: (patient_id, local_z, y0=crop_y0+32, x0=crop_x0+32)
    boundary_like_ratio = 1 - roi_0_0_patch_ratio
    """
    import pandas as pd

    df = df.copy()
    df["anchor_y0"] = df["crop_y0"] + 32
    df["anchor_x0"] = df["crop_x0"] + 32

    patients = df["patient_id"].unique()
    all_src_rows = []

    for pid in patients:
        src_path = SOURCE_SCORE_BASE / "lesion_stage1_dev_by_patient" / f"{pid}.csv"
        if not src_path.exists():
            log_warn(f"per-patient source CSV 없음: {src_path}")
            continue
        try:
            src = pd.read_csv(src_path, usecols=["patient_id", "local_z", "y0", "x0",
                                                   "roi_0_0_patch_ratio", "lesion_patch_ratio"])
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
    df["anchor_y0"] = df["anchor_y0"].astype(int)
    df["anchor_x0"] = df["anchor_x0"].astype(int)

    src_all = src_all.drop_duplicates(subset=["patient_id", "local_z", "anchor_y0", "anchor_x0"])

    df_merged = df.merge(
        src_all[["patient_id", "local_z", "anchor_y0", "anchor_x0",
                 "roi_0_0_patch_ratio", "lesion_patch_ratio"]],
        on=["patient_id", "local_z", "anchor_y0", "anchor_x0"],
        how="left",
    )

    n_matched = df_merged["roi_0_0_patch_ratio"].notna().sum()
    n_total = len(df_merged)
    log_info(f"roi_0_0_patch_ratio join: {n_matched:,}/{n_total:,} matched ({100*n_matched/n_total:.1f}%)")

    if n_matched < n_total * 0.8:
        log_warn(f"join match rate < 80% — boundary ratio 불완전")

    df_merged["boundary_like_ratio"] = 1.0 - df_merged["roi_0_0_patch_ratio"].fillna(0.0)
    return df_merged, n_matched > 0


# ---------------------------------------------------------------------------
# Group assignment
# Priority: G1 > G2 > G3 > G4
# ---------------------------------------------------------------------------
def assign_groups(df, boundary_available):
    """
    G1_lesion_positive: label == positive
    G2_roi_boundary_ge50: non-lesion, boundary_like_ratio >= 0.5
    G3_roi_boundary_lt50: non-lesion, 0 < boundary_like_ratio < 0.5
    G4_other: non-lesion, boundary_like_ratio == 0 (fully inside)
    """
    import numpy as np

    def _assign(row):
        if row["label"] == "positive":
            return "G1_lesion_positive"
        if not boundary_available or "boundary_like_ratio" not in row or np.isnan(row.get("boundary_like_ratio", float("nan"))):
            return "G4_other_boundary_unknown"
        blr = row["boundary_like_ratio"]
        if blr >= 0.5:
            return "G2_roi_boundary_ge50"
        elif blr > 0.0:
            return "G3_roi_boundary_lt50"
        else:
            return "G4_other_nonlesion_nonboundary"

    df["group"] = df.apply(_assign, axis=1)
    return df


# ---------------------------------------------------------------------------
# Score distribution stats
# ---------------------------------------------------------------------------
PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]


def score_distribution(scores):
    import numpy as np
    if len(scores) == 0:
        return {"count": 0}
    arr = scores.values if hasattr(scores, "values") else np.array(scores)
    arr = arr[~np.isnan(arr)]
    result = {
        "count": len(arr),
        "mean": round(float(np.mean(arr)), 6),
        "std": round(float(np.std(arr)), 6),
        "min": round(float(np.min(arr)), 6),
        "max": round(float(np.max(arr)), 6),
    }
    for p in PERCENTILES:
        result[f"p{p:02d}"] = round(float(np.percentile(arr, p)), 6)
    return result


# ---------------------------------------------------------------------------
# AUROC / AUPRC
# ---------------------------------------------------------------------------
def compute_auroc_auprc(scores, labels_binary):
    import numpy as np
    from numpy import searchsorted

    arr = np.array(scores, dtype=float)
    lab = np.array(labels_binary, dtype=int)
    valid = ~np.isnan(arr)
    arr, lab = arr[valid], lab[valid]

    pos = arr[lab == 1]
    neg = arr[lab == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None, None

    n_pos, n_neg = len(pos), len(neg)
    combined = np.concatenate([pos, neg])
    sorted_combined = np.sort(combined)
    ranks_pos = searchsorted(sorted_combined, pos, side="left") + 1
    u = ranks_pos.sum() - n_pos * (n_pos + 1) / 2
    auroc = float(u / (n_pos * n_neg))

    order = np.argsort(arr)[::-1]
    sorted_labels = lab[order]
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (n_pos + 1e-12)
    auprc = float(np.sum(precision[1:] * (recall[1:] - recall[:-1])))

    return auroc, auprc


# ---------------------------------------------------------------------------
# Pairwise group comparison
# ---------------------------------------------------------------------------
def pairwise_comparison(df, group_a, group_b, score_col="first_stage_score"):
    import numpy as np

    a = df[df["group"] == group_a][score_col].dropna().values
    b = df[df["group"] == group_b][score_col].dropna().values

    if len(a) == 0 or len(b) == 0:
        return {"group_a": group_a, "group_b": group_b, "n_a": len(a), "n_b": len(b)}

    combined_scores = np.concatenate([a, b])
    combined_labels = np.array([1] * len(a) + [0] * len(b))
    auroc, auprc = compute_auroc_auprc(combined_scores, combined_labels)

    median_diff = float(np.median(a) - np.median(b))
    p90_diff = float(np.percentile(a, 90) - np.percentile(b, 90))
    p95_diff = float(np.percentile(a, 95) - np.percentile(b, 95))

    # fraction of a scores > median(b)
    a_gt_b_median = float((a > np.median(b)).mean())
    # fraction of a scores > p90(b)
    a_gt_b_p90 = float((a > np.percentile(b, 90)).mean())

    return {
        "group_a": group_a,
        "group_b": group_b,
        "score_col": score_col,
        "n_a": len(a),
        "n_b": len(b),
        "auroc_a_vs_b": round(auroc, 6) if auroc is not None else None,
        "auprc_a_vs_b": round(auprc, 6) if auprc is not None else None,
        "median_a": round(float(np.median(a)), 6),
        "median_b": round(float(np.median(b)), 6),
        "median_diff_a_minus_b": round(median_diff, 6),
        "p90_a": round(float(np.percentile(a, 90)), 6),
        "p90_b": round(float(np.percentile(b, 90)), 6),
        "p90_diff_a_minus_b": round(p90_diff, 6),
        "p95_a": round(float(np.percentile(a, 95)), 6),
        "p95_b": round(float(np.percentile(b, 95)), 6),
        "p95_diff_a_minus_b": round(p95_diff, 6),
        "frac_a_gt_b_median": round(a_gt_b_median, 6),
        "frac_a_gt_b_p90": round(a_gt_b_p90, 6),
    }


# ---------------------------------------------------------------------------
# dry-run
# ---------------------------------------------------------------------------
def run_dry():
    print("=" * 70)
    print("[DRY-RUN] padim_score_distribution_lesion_roi_boundary v1")
    print("=" * 70)

    check_stage2_holdout_not_accessed()
    print(f"[CHECK] stage2_holdout 접근 없음: {not GUARDRAILS['stage2_holdout_accessed']}")

    for label, path in [
        ("manifest_csv", MANIFEST_CSV),
        ("rd_d1s_score_csv", RD_D1S_SCORE_CSV),
        ("convae_score_csv", CONVAE_SCORE_CSV),
        ("source_score_base", SOURCE_SCORE_BASE),
    ]:
        exists = path.exists()
        print(f"[CHECK] {label}: {'OK' if exists else 'MISSING'} — {path}")
        if not exists and label == "manifest_csv":
            log_error(f"필수 입력 없음: {path}")

    # 컬럼 확인
    if MANIFEST_CSV.exists():
        import pandas as pd
        df_head = pd.read_csv(MANIFEST_CSV, nrows=5)
        cols = set(df_head.columns)
        required = {"candidate_id", "patient_id", "local_z", "crop_y0", "crop_x0",
                    "first_stage_score", "label", "stage_split"}
        missing = required - cols
        if missing:
            log_error(f"manifest 컬럼 누락: {missing}")
        else:
            print(f"[CHECK] manifest 필요 컬럼 존재: OK")

        # stage_split 확인
        raw = pd.read_csv(MANIFEST_CSV, usecols=["stage_split"])
        if (raw["stage_split"] == "stage2_holdout").any():
            log_error("stage2_holdout 행 존재!")
        else:
            n_dev = (raw["stage_split"] == "stage1_dev").sum()
            print(f"[CHECK] stage1_dev 행 수: {n_dev:,}, stage2_holdout: 없음")

    # boundary 컬럼 탐색
    boundary_col_candidates = [
        "roi_boundary_overlap_ratio", "boundary_overlap_ratio", "boundary_ratio",
        "roi_edge_overlap_ratio", "refined_roi_boundary_overlap",
        "refined_roi_ratio", "roi_inside_ratio", "mask_inside_ratio",
        "roi_overlap_ratio", "roi_0_0_patch_ratio",
    ]
    if MANIFEST_CSV.exists():
        import pandas as pd
        df_head = pd.read_csv(MANIFEST_CSV, nrows=5)
        found_direct = [c for c in boundary_col_candidates if c in df_head.columns]
        print(f"[CHECK] manifest 내 boundary 관련 컬럼 탐색: {found_direct or '없음'}")

    if SOURCE_SCORE_BASE.exists():
        sample_csvs = list((SOURCE_SCORE_BASE / "lesion_stage1_dev_by_patient").glob("*.csv"))
        n_src = len(sample_csvs)
        print(f"[CHECK] per-patient source CSV 수: {n_src}")
        if sample_csvs:
            import pandas as pd
            src_head = pd.read_csv(sample_csvs[0], nrows=2)
            found_src = [c for c in boundary_col_candidates + ["roi_0_0_patch_ratio", "lesion_patch_ratio"]
                         if c in src_head.columns]
            print(f"[CHECK] per-patient source CSV boundary 컬럼: {found_src}")
            print(f"[CHECK] join key 확인: y0 + 32 = anchor_y0, x0 + 32 = anchor_x0")
            print(f"        → boundary_source = proxy_from_roi_0_0_patch_ratio")

    print()
    if errors:
        print(f"[DRY-RUN RESULT] FAIL — {len(errors)} error(s)")
        for e in errors:
            print(f"  - {e['message']}")
        sys.exit(1)
    else:
        print("[DRY-RUN RESULT] PASS")
    print("[INFO] 파일 생성 없음 (dry-run)")


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------
def run_audit():
    import pandas as pd
    import numpy as np

    print("=" * 70)
    print("[AUDIT] padim_score_distribution_lesion_roi_boundary v1")
    print("=" * 70)

    check_stage2_holdout_not_accessed()

    # 1. manifest 로드
    if not MANIFEST_CSV.exists():
        log_error(f"manifest CSV 없음: {MANIFEST_CSV}")
        _finalize_with_errors()
        return

    log_info(f"manifest 로드: {MANIFEST_CSV}")
    df = pd.read_csv(MANIFEST_CSV)
    log_info(f"  → {len(df):,} rows")

    if "stage_split" in df.columns:
        if (df["stage_split"] == "stage2_holdout").any():
            GUARDRAILS["stage2_holdout_accessed"] = True
            log_error("stage2_holdout 행 존재!")
        df = df[df["stage_split"] == "stage1_dev"].copy()
        log_info(f"  → stage1_dev 필터 후: {len(df):,} rows")

    # 2. RD-D1s score join (boundary_status, six_bin_label)
    if RD_D1S_SCORE_CSV.exists():
        log_info(f"RD-D1s score 로드 (boundary_status/six_bin_label 추가)")
        df_rd = pd.read_csv(RD_D1S_SCORE_CSV,
            usecols=["candidate_id", "rd4ad_crop_score", "boundary_status",
                     "six_bin_label", "z_level", "boundary_status"])
        df = df.merge(df_rd[["candidate_id", "rd4ad_crop_score", "boundary_status",
                              "six_bin_label", "z_level"]],
                      on="candidate_id", how="left")

    # 3. ConvAE score join
    df_conv = None
    if CONVAE_SCORE_CSV.exists():
        log_info(f"ConvAE score 로드")
        df_conv = pd.read_csv(CONVAE_SCORE_CSV,
            usecols=["candidate_id", "convAE_crop_score_l1_mean",
                     "convAE_mediastinal_channels_l1_mean"])
        df = df.merge(df_conv[["candidate_id", "convAE_crop_score_l1_mean",
                                "convAE_mediastinal_channels_l1_mean"]],
                      on="candidate_id", how="left")

    # 4. per-patient source CSV join → roi_0_0_patch_ratio
    log_info("per-patient source CSV join (roi_0_0_patch_ratio) ...")
    df, roi_joined = load_roi_ratio_from_source_csvs(df)

    if roi_joined:
        GUARDRAILS["boundary_available"] = True
        GUARDRAILS["boundary_source"] = "proxy_from_roi_0_0_patch_ratio_computed_as_1_minus_ratio"
        log_info(f"boundary_like_ratio 계산 완료 (1 - roi_0_0_patch_ratio)")
    else:
        GUARDRAILS["boundary_available"] = False
        GUARDRAILS["boundary_source"] = "not_available"
        log_warn("boundary ratio join 실패 — G1 vs nonlesion 분석으로 제한")

    # 5. 그룹 배정
    df = assign_groups(df, GUARDRAILS["boundary_available"])
    group_counts = df["group"].value_counts().to_dict()
    log_info(f"그룹 배정 결과: {group_counts}")

    ALL_GROUPS = ["G1_lesion_positive", "G2_roi_boundary_ge50", "G3_roi_boundary_lt50",
                  "G4_other_nonlesion_nonboundary", "G4_other_boundary_unknown"]

    SCORE_COLS = ["first_stage_score"]
    if "rd4ad_crop_score" in df.columns:
        SCORE_COLS.append("rd4ad_crop_score")
    if "convAE_crop_score_l1_mean" in df.columns:
        SCORE_COLS.append("convAE_crop_score_l1_mean")
    if "convAE_mediastinal_channels_l1_mean" in df.columns:
        SCORE_COLS.append("convAE_mediastinal_channels_l1_mean")

    # 6. group별 score 분포
    group_dist_rows = []
    for grp in ALL_GROUPS:
        sub = df[df["group"] == grp]
        if len(sub) == 0:
            continue
        for sc in SCORE_COLS:
            if sc not in sub.columns:
                continue
            dist = score_distribution(sub[sc])
            dist["group"] = grp
            dist["score_col"] = sc
            group_dist_rows.append(dist)

    # 7. pairwise comparison
    pair_rows = []
    pairs = [
        ("G1_lesion_positive", "G2_roi_boundary_ge50"),
        ("G1_lesion_positive", "G3_roi_boundary_lt50"),
        ("G1_lesion_positive", "G4_other_nonlesion_nonboundary"),
        ("G2_roi_boundary_ge50", "G3_roi_boundary_lt50"),
        ("G2_roi_boundary_ge50", "G4_other_nonlesion_nonboundary"),
        ("G1_lesion_positive", "G4_other_boundary_unknown"),
    ]
    for ga, gb in pairs:
        if df["group"].isin([ga]).any() and df["group"].isin([gb]).any():
            for sc in SCORE_COLS:
                row = pairwise_comparison(df, ga, gb, score_col=sc)
                pair_rows.append(row)

    # 8. top score band analysis
    top_band_rows = []
    n_total = len(df)
    for band_pct in [1, 5, 10, 20]:
        k = max(1, int(n_total * band_pct / 100))
        top_k_idx = df.nlargest(k, "first_stage_score").index
        top_k = df.loc[top_k_idx]
        grp_dist = top_k["group"].value_counts()

        lesion_patients = set(df[df["label"] == "positive"]["patient_id"].unique())
        top_lesion_patients = set(top_k[top_k["label"] == "positive"]["patient_id"].unique())
        all_lesion_slices = set(
            map(tuple, df[df["label"] == "positive"][["patient_id", "local_z"]].drop_duplicates().values)
        )
        top_lesion_slices = set(
            map(tuple, top_k[top_k["label"] == "positive"][["patient_id", "local_z"]].drop_duplicates().values)
        )

        row = {
            "top_band_pct": band_pct,
            "n_candidates": k,
            "G1_lesion_positive_count": int(grp_dist.get("G1_lesion_positive", 0)),
            "G2_roi_boundary_ge50_count": int(grp_dist.get("G2_roi_boundary_ge50", 0)),
            "G3_roi_boundary_lt50_count": int(grp_dist.get("G3_roi_boundary_lt50", 0)),
            "G4_other_count": int(grp_dist.get("G4_other_nonlesion_nonboundary", 0)
                                  + grp_dist.get("G4_other_boundary_unknown", 0)),
            "G1_lesion_pct": round(grp_dist.get("G1_lesion_positive", 0) / k * 100, 2),
            "G2_boundary_ge50_pct": round(grp_dist.get("G2_roi_boundary_ge50", 0) / k * 100, 2),
            "G3_boundary_lt50_pct": round(grp_dist.get("G3_roi_boundary_lt50", 0) / k * 100, 2),
            "n_patients": int(top_k["patient_id"].nunique()),
            "lesion_patient_coverage": round(
                len(top_lesion_patients) / len(lesion_patients), 6) if lesion_patients else 0,
            "lesion_slice_coverage": round(
                len(top_lesion_slices) / len(all_lesion_slices), 6) if all_lesion_slices else 0,
        }
        top_band_rows.append(row)

    # 9. problem patient analysis (LUNG1-086, LUNG1-386, LUNG1-399)
    prob_rows = []
    all_lesion_patients = set(df[df["label"] == "positive"]["patient_id"].unique())
    for pid in PROBLEM_PATIENTS:
        if pid not in df["patient_id"].values:
            prob_rows.append({"patient_id": pid, "note": "not_in_dataset"})
            continue
        pdf = df[df["patient_id"] == pid].copy()
        g1 = pdf[pdf["group"] == "G1_lesion_positive"]
        g2 = pdf[pdf["group"] == "G2_roi_boundary_ge50"]
        g3 = pdf[pdf["group"] == "G3_roi_boundary_lt50"]
        g4_all = pdf[~pdf["group"].isin(["G1_lesion_positive", "G2_roi_boundary_ge50",
                                          "G3_roi_boundary_lt50"])]

        g1_scores = g1["first_stage_score"].values
        all_scores = pdf["first_stage_score"].values
        g1_max = float(g1_scores.max()) if len(g1_scores) > 0 else None
        # percentile rank of g1_max in all scores
        g1_max_percentile = float(np.mean(all_scores <= g1_max) * 100) if g1_max is not None else None
        # count of G2/G3/G4 patches above g1_max
        n_above_g1max = int((pdf[pdf["group"] != "G1_lesion_positive"]["first_stage_score"] > g1_max).sum()) if g1_max is not None else None

        # per-patient top-z analysis
        from_slice_df = pdf.groupby("local_z").agg(
            n_g1=("group", lambda x: (x == "G1_lesion_positive").sum()),
            slice_top20_mean=("first_stage_score", lambda x: float(np.sort(x.values)[-min(20, len(x)):].mean())),
        ).reset_index()
        from_slice_df = from_slice_df.sort_values("slice_top20_mean", ascending=False).reset_index(drop=True)
        g1_z = set(g1["local_z"].unique())
        in_top5z = g1_z & set(from_slice_df.head(5)["local_z"].values)
        in_top10z = g1_z & set(from_slice_df.head(10)["local_z"].values)

        row = {
            "patient_id": pid,
            "n_total": int(len(pdf)),
            "n_g1_lesion": int(len(g1)),
            "n_g2_boundary_ge50": int(len(g2)),
            "n_g3_boundary_lt50": int(len(g3)),
            "n_g4_other": int(len(g4_all)),
            "g1_score_min": round(float(g1_scores.min()), 6) if len(g1_scores) > 0 else None,
            "g1_score_p50": round(float(np.median(g1_scores)), 6) if len(g1_scores) > 0 else None,
            "g1_score_p90": round(float(np.percentile(g1_scores, 90)), 6) if len(g1_scores) > 0 else None,
            "g1_score_max": round(g1_max, 6) if g1_max is not None else None,
            "g2_score_p50": round(float(np.median(g2["first_stage_score"].values)), 6) if len(g2) > 0 else None,
            "g2_score_p90": round(float(np.percentile(g2["first_stage_score"].values, 90)), 6) if len(g2) > 0 else None,
            "g2_score_max": round(float(g2["first_stage_score"].max()), 6) if len(g2) > 0 else None,
            "g3_score_p50": round(float(np.median(g3["first_stage_score"].values)), 6) if len(g3) > 0 else None,
            "g3_score_max": round(float(g3["first_stage_score"].max()), 6) if len(g3) > 0 else None,
            "g4_score_p50": round(float(np.median(g4_all["first_stage_score"].values)), 6) if len(g4_all) > 0 else None,
            "g4_score_max": round(float(g4_all["first_stage_score"].max()), 6) if len(g4_all) > 0 else None,
            "g1_max_percentile_in_patient": round(g1_max_percentile, 2) if g1_max_percentile is not None else None,
            "n_nonlesion_above_g1_max": n_above_g1max,
            "g1_z_count": int(len(g1_z)),
            "g1_z_in_top5z_count": int(len(in_top5z)),
            "g1_z_in_top10z_count": int(len(in_top10z)),
            "top5z_covers_any_lesion_z": len(in_top5z) > 0,
            "top10z_covers_any_lesion_z": len(in_top10z) > 0,
        }
        prob_rows.append(row)

    # 10. z-slice group audit (all patients)
    log_info("z-slice 분석 중 ...")
    lesion_slices_all = set(
        map(tuple, df[df["label"] == "positive"][["patient_id", "local_z"]].drop_duplicates().values)
    )

    z_rows = []
    for (pid, z), grp_df in df.groupby(["patient_id", "local_z"]):
        n_g1 = int((grp_df["group"] == "G1_lesion_positive").sum())
        n_g2 = int((grp_df["group"] == "G2_roi_boundary_ge50").sum())
        n_g3 = int((grp_df["group"] == "G3_roi_boundary_lt50").sum())
        n_g4 = int((~grp_df["group"].isin(["G1_lesion_positive", "G2_roi_boundary_ge50",
                                             "G3_roi_boundary_lt50"])).sum())
        scores = grp_df["first_stage_score"].values
        top20_mean = float(np.sort(scores)[-min(20, len(scores)):].mean())
        slice_max = float(scores.max())
        is_positive_slice = (pid, z) in lesion_slices_all
        n_total_z = len(grp_df)
        dominant = "G2" if n_g2 >= n_g1 and n_g2 >= n_g3 and n_g2 >= n_g4 else (
            "G3" if n_g3 >= n_g1 and n_g3 >= n_g4 else (
                "G1" if n_g1 > 0 else "G4"
            )
        )
        z_rows.append({
            "patient_id": pid,
            "local_z": int(z),
            "n_total": n_total_z,
            "n_g1_lesion": n_g1,
            "n_g2_boundary_ge50": n_g2,
            "n_g3_boundary_lt50": n_g3,
            "n_g4_other": n_g4,
            "slice_top20_mean": round(top20_mean, 6),
            "slice_max": round(slice_max, 6),
            "is_positive_slice": is_positive_slice,
            "dominant_group": dominant,
        })

    z_df = pd.DataFrame(z_rows)

    # positive z-slice가 top-N z에 들어오는 비율 (per patient)
    log_info("positive z-slice top-N coverage 계산 중 ...")
    top5z_pos_coverage = []
    top10z_pos_coverage = []
    for pid, pz_df in z_df.groupby("patient_id"):
        pz_sorted = pz_df.sort_values("slice_top20_mean", ascending=False).reset_index(drop=True)
        pos_z = set(pz_df[pz_df["is_positive_slice"]]["local_z"].values)
        if not pos_z:
            continue
        top5z = set(pz_sorted.head(5)["local_z"].values)
        top10z = set(pz_sorted.head(10)["local_z"].values)
        # expand ±2
        top5z_exp = set()
        for z in top5z:
            for dz in range(-2, 3):
                top5z_exp.add(z + dz)
        top10z_exp = set()
        for z in top10z:
            for dz in range(-2, 3):
                top10z_exp.add(z + dz)
        covered5 = len(pos_z & top5z_exp) / len(pos_z)
        covered10 = len(pos_z & top10z_exp) / len(pos_z)
        top5z_pos_coverage.append(covered5)
        top10z_pos_coverage.append(covered10)

    import numpy as np
    top5z_cov_mean = float(np.mean(top5z_pos_coverage)) if top5z_pos_coverage else None
    top10z_cov_mean = float(np.mean(top10z_pos_coverage)) if top10z_pos_coverage else None
    top5z_cov_100pct = float(np.mean(np.array(top5z_pos_coverage) >= 1.0)) if top5z_pos_coverage else None
    top10z_cov_100pct = float(np.mean(np.array(top10z_pos_coverage) >= 1.0)) if top10z_pos_coverage else None

    log_info(f"  top5z ±2 positive z coverage: mean={top5z_cov_mean:.3f}, 100%={top5z_cov_100pct:.3f}")
    log_info(f"  top10z ±2 positive z coverage: mean={top10z_cov_mean:.3f}, 100%={top10z_cov_100pct:.3f}")

    # G2 dominant slice가 top-z를 차지하는 비율
    g2_dominated_top5z_ratio = []
    g2_dominated_top10z_ratio = []
    for pid, pz_df in z_df.groupby("patient_id"):
        pz_sorted = pz_df.sort_values("slice_top20_mean", ascending=False).reset_index(drop=True)
        top5 = pz_sorted.head(5)
        top10 = pz_sorted.head(10)
        g2_dom5 = (top5["dominant_group"] == "G2").mean()
        g2_dom10 = (top10["dominant_group"] == "G2").mean()
        g2_dominated_top5z_ratio.append(g2_dom5)
        g2_dominated_top10z_ratio.append(g2_dom10)

    g2_top5z_ratio = float(np.mean(g2_dominated_top5z_ratio)) if g2_dominated_top5z_ratio else None
    g2_top10z_ratio = float(np.mean(g2_dominated_top10z_ratio)) if g2_dominated_top10z_ratio else None

    log_info(f"  G2_boundary_ge50 top5z 점유율: {g2_top5z_ratio:.3f}")
    log_info(f"  G2_boundary_ge50 top10z 점유율: {g2_top10z_ratio:.3f}")

    # 11. 출력 파일 저장
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(group_dist_rows).to_csv(
        MANIFEST_DIR / "group_distribution_by_score.csv", index=False)
    log_info("group_distribution_by_score.csv 저장")

    pd.DataFrame(pair_rows).to_csv(
        MANIFEST_DIR / "group_pairwise_comparison.csv", index=False)
    log_info("group_pairwise_comparison.csv 저장")

    pd.DataFrame(top_band_rows).to_csv(
        MANIFEST_DIR / "top_score_band_composition.csv", index=False)
    log_info("top_score_band_composition.csv 저장")

    pd.DataFrame(prob_rows).to_csv(
        MANIFEST_DIR / "problem_patient_score_audit.csv", index=False)
    log_info("problem_patient_score_audit.csv 저장")

    z_df.to_csv(MANIFEST_DIR / "z_slice_group_audit.csv", index=False)
    log_info("z_slice_group_audit.csv 저장")

    # 12. verdict 판정
    if not GUARDRAILS["boundary_available"]:
        verdict = "PARTIAL_PASS_BOUNDARY_COLUMN_MISSING"
    elif errors:
        verdict = "FAIL"
    else:
        verdict = "PASS"

    # 13. 핵심 수치 수집
    g1_dist = next((r for r in group_dist_rows
                    if r["group"] == "G1_lesion_positive" and r["score_col"] == "first_stage_score"), {})
    g2_dist = next((r for r in group_dist_rows
                    if r["group"] == "G2_roi_boundary_ge50" and r["score_col"] == "first_stage_score"), {})
    g3_dist = next((r for r in group_dist_rows
                    if r["group"] == "G3_roi_boundary_lt50" and r["score_col"] == "first_stage_score"), {})
    g4_dist = next((r for r in group_dist_rows
                    if r["group"] == "G4_other_nonlesion_nonboundary" and r["score_col"] == "first_stage_score"), {})

    g1_vs_g2 = next((r for r in pair_rows if r["group_a"] == "G1_lesion_positive"
                     and r["group_b"] == "G2_roi_boundary_ge50" and r["score_col"] == "first_stage_score"), {})
    g1_vs_g3 = next((r for r in pair_rows if r["group_a"] == "G1_lesion_positive"
                     and r["group_b"] == "G3_roi_boundary_lt50" and r["score_col"] == "first_stage_score"), {})
    g1_vs_g4 = next((r for r in pair_rows if r["group_a"] == "G1_lesion_positive"
                     and r["group_b"] == "G4_other_nonlesion_nonboundary" and r["score_col"] == "first_stage_score"), {})

    # 14. summary JSON
    summary = {
        "script": "padim_score_distribution_lesion_roi_boundary.py",
        "version": "v1",
        "verdict": verdict,
        **GUARDRAILS,
        "n_errors": len(errors),
        "group_counts": group_counts,
        "g1_first_stage_score": {k: v for k, v in g1_dist.items() if k not in ("group", "score_col")},
        "g2_first_stage_score": {k: v for k, v in g2_dist.items() if k not in ("group", "score_col")},
        "g3_first_stage_score": {k: v for k, v in g3_dist.items() if k not in ("group", "score_col")},
        "g4_first_stage_score": {k: v for k, v in g4_dist.items() if k not in ("group", "score_col")},
        "g1_vs_g2_auroc": g1_vs_g2.get("auroc_a_vs_b"),
        "g1_vs_g2_median_diff": g1_vs_g2.get("median_diff_a_minus_b"),
        "g1_vs_g3_auroc": g1_vs_g3.get("auroc_a_vs_b"),
        "g1_vs_g4_auroc": g1_vs_g4.get("auroc_a_vs_b"),
        "top1pct_g1_pct": next((r["G1_lesion_pct"] for r in top_band_rows if r["top_band_pct"] == 1), None),
        "top5pct_g1_pct": next((r["G1_lesion_pct"] for r in top_band_rows if r["top_band_pct"] == 5), None),
        "top10pct_g1_pct": next((r["G1_lesion_pct"] for r in top_band_rows if r["top_band_pct"] == 10), None),
        "top1pct_g2_pct": next((r["G2_boundary_ge50_pct"] for r in top_band_rows if r["top_band_pct"] == 1), None),
        "top5pct_g2_pct": next((r["G2_boundary_ge50_pct"] for r in top_band_rows if r["top_band_pct"] == 5), None),
        "top10pct_g2_pct": next((r["G2_boundary_ge50_pct"] for r in top_band_rows if r["top_band_pct"] == 10), None),
        "top5z_pm2_positive_z_coverage_mean": top5z_cov_mean,
        "top5z_pm2_positive_z_coverage_100pct_rate": top5z_cov_100pct,
        "top10z_pm2_positive_z_coverage_mean": top10z_cov_mean,
        "top10z_pm2_positive_z_coverage_100pct_rate": top10z_cov_100pct,
        "g2_dominant_in_top5z_mean": g2_top5z_ratio,
        "g2_dominant_in_top10z_mean": g2_top10z_ratio,
        "problem_patients": {r["patient_id"]: r for r in prob_rows if "n_g1_lesion" in r},
    }

    with open(REPORT_DIR / "padim_score_distribution_lesion_roi_boundary_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    log_info("summary.json 저장")

    # 15. report.md
    _write_report(summary, group_dist_rows, pair_rows, top_band_rows, prob_rows)

    # errors.csv
    err_df = pd.DataFrame(errors) if errors else pd.DataFrame(columns=["type", "message"])
    err_df.to_csv(LOG_DIR / "errors.csv", index=False)

    # DONE.json
    with open(OUT_ROOT / "DONE.json", "w") as f:
        json.dump({"done": True, "verdict": verdict, "n_errors": len(errors),
                   "boundary_available": GUARDRAILS["boundary_available"],
                   "boundary_source": GUARDRAILS["boundary_source"]}, f, indent=2)
    log_info("DONE.json 저장")

    # 최종 출력
    print()
    print("=" * 70)
    print(f"[RESULT] 판정: {verdict}")
    print(f"  boundary_source: {GUARDRAILS['boundary_source']}")
    print(f"  그룹 수: {group_counts}")
    print(f"  G1 first_stage_score median={g1_dist.get('p50')} p90={g1_dist.get('p90')}")
    print(f"  G2 first_stage_score median={g2_dist.get('p50')} p90={g2_dist.get('p90')}")
    print(f"  G3 first_stage_score median={g3_dist.get('p50')} p90={g3_dist.get('p90')}")
    print(f"  G4 first_stage_score median={g4_dist.get('p50')} p90={g4_dist.get('p90')}")
    print(f"  G1 vs G2 AUROC: {g1_vs_g2.get('auroc_a_vs_b')} (G1이 높으면 > 0.5)")
    print(f"  G1 vs G3 AUROC: {g1_vs_g3.get('auroc_a_vs_b')}")
    print(f"  G1 vs G4 AUROC: {g1_vs_g4.get('auroc_a_vs_b')}")
    print(f"  top1% G1비율: {summary['top1pct_g1_pct']}%  G2비율: {summary['top1pct_g2_pct']}%")
    print(f"  top5% G1비율: {summary['top5pct_g1_pct']}%  G2비율: {summary['top5pct_g2_pct']}%")
    print(f"  top10% G1비율: {summary['top10pct_g1_pct']}%  G2비율: {summary['top10pct_g2_pct']}%")
    print(f"  top5z±2 positive z coverage mean: {top5z_cov_mean:.3f}")
    print(f"  G2 top5z 점유율: {g2_top5z_ratio:.3f}")
    print("=" * 70)

    if errors:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------
def _write_report(summary, group_dist_rows, pair_rows, top_band_rows, prob_rows):
    lines = []
    lines.append("# PaDiM Score Distribution Audit — Lesion vs ROI-Boundary Groups v1\n\n")
    lines.append(f"**판정: {summary['verdict']}**\n\n")
    lines.append(f"- boundary_source: `{summary['boundary_source']}`\n")
    lines.append(f"- boundary_available: `{summary['boundary_available']}`\n")
    lines.append(f"- stage2_holdout 접근: `{summary['stage2_holdout_accessed']}`\n")
    lines.append(f"- model forward: `{summary['model_forward_executed']}`\n\n")

    lines.append("## 그룹 수\n\n")
    for g, n in summary["group_counts"].items():
        lines.append(f"- {g}: {n:,}\n")
    lines.append("\n")

    lines.append("## first_stage_score 분포 비교\n\n")
    lines.append("| group | count | mean | p50 | p90 | p95 | p99 | max |\n")
    lines.append("|-------|-------|------|-----|-----|-----|-----|-----|\n")
    for r in group_dist_rows:
        if r["score_col"] != "first_stage_score":
            continue
        lines.append(
            f"| {r['group']} | {r['count']:,} | {r.get('mean','?'):.3f} "
            f"| {r.get('p50','?'):.3f} | {r.get('p90','?'):.3f} "
            f"| {r.get('p95','?'):.3f} | {r.get('p99','?'):.3f} | {r.get('max','?'):.3f} |\n"
        )
    lines.append("\n")

    lines.append("## Pairwise AUROC\n\n")
    lines.append("| 비교 | score_col | AUROC | median_diff | p90_diff |\n")
    lines.append("|------|-----------|-------|-------------|----------|\n")
    for r in pair_rows:
        lines.append(
            f"| {r['group_a']} vs {r['group_b']} | {r['score_col']} "
            f"| {r.get('auroc_a_vs_b','?')} | {r.get('median_diff_a_minus_b','?')} "
            f"| {r.get('p90_diff_a_minus_b','?')} |\n"
        )
    lines.append("\n")

    lines.append("## Top Score Band 구성\n\n")
    lines.append("| top% | n | G1_lesion% | G2_boundary_ge50% | G3_boundary_lt50% | G4% | lp_cov | ls_cov |\n")
    lines.append("|------|---|-----------|-----------------|-----------------|-----|--------|--------|\n")
    for r in top_band_rows:
        lines.append(
            f"| top{r['top_band_pct']}% | {r['n_candidates']:,} "
            f"| {r['G1_lesion_pct']}% | {r['G2_boundary_ge50_pct']}% "
            f"| {r['G3_boundary_lt50_pct']}% | {r['G4_other_count']} "
            f"| {r['lesion_patient_coverage']:.4f} | {r['lesion_slice_coverage']:.4f} |\n"
        )
    lines.append("\n")

    lines.append("## 문제 환자 분석\n\n")
    for r in prob_rows:
        pid = r["patient_id"]
        if "n_g1_lesion" not in r:
            lines.append(f"### {pid}: 데이터셋에 없음\n\n")
            continue
        lines.append(f"### {pid}\n\n")
        lines.append(f"- G1 lesion: {r['n_g1_lesion']}개, score p50={r['g1_score_p50']} p90={r['g1_score_p90']} max={r['g1_score_max']}\n")
        lines.append(f"- G2 boundary_ge50: {r['n_g2_boundary_ge50']}개, score p50={r['g2_score_p50']} max={r['g2_score_max']}\n")
        lines.append(f"- G3 boundary_lt50: {r['n_g3_boundary_lt50']}개\n")
        lines.append(f"- G4 other: {r['n_g4_other']}개\n")
        lines.append(f"- G1 max score의 환자 내 percentile: {r['g1_max_percentile_in_patient']}%\n")
        lines.append(f"- G1 max 초과 비병변 patch 수: {r['n_nonlesion_above_g1_max']}\n")
        lines.append(f"- 병변 z-slice 수: {r['g1_z_count']}, top5z에 포함: {r['g1_z_in_top5z_count']}, top10z: {r['g1_z_in_top10z_count']}\n\n")

    lines.append("## top-z selector 실패 원인 분석\n\n")
    g2_ratio = summary.get("g2_dominant_in_top5z_mean", 0) or 0
    g1_top5 = summary.get("top5pct_g1_pct", 0) or 0
    g2_top5 = summary.get("top5pct_g2_pct", 0) or 0
    lines.append(f"- top5% score band: G1={g1_top5}%, G2={g2_top5}%\n")
    lines.append(f"- top5z±2 positive z coverage (mean across patients): {summary.get('top5z_pm2_positive_z_coverage_mean'):.3f}\n")
    lines.append(f"- G2_boundary_ge50가 top5z 슬라이스를 차지하는 비율: {g2_ratio:.3f}\n\n")

    if g2_top5 > g1_top5 * 2:
        lines.append("**결론: G2_roi_boundary_ge50 patch가 first_stage_score 상위 band를 장악.**\n")
        lines.append("ROI 경계에 걸친 patch의 Mahalanobis distance가 병변 patch보다 높아,\n")
        lines.append("score 기반 top-z/top-k selector는 경계 artifact를 병변보다 먼저 선택한다.\n")
        lines.append("→ first_stage_score 기반 adaptive selector는 구조적으로 불가능.\n\n")
        lines.append("**다음 방향 추천:**\n")
        lines.append("1. ROI boundary penalty 또는 interior-only 제한 필터\n")
        lines.append("2. six_bin_label/boundary_status 기반 taxonomy flag로 경계 patch 분리\n")
        lines.append("3. supervised lesion-vs-hard-negative second-stage verifier\n")
    else:
        lines.append("**G2 score가 G1보다 높지 않음 — top-z 실패 원인이 다른 곳에 있을 수 있음.**\n")
        lines.append("병변 z-slice가 너무 넓게 분산되어 있어 top-N z만으로 커버가 불가능한 구조일 수 있음.\n")

    lines.append("\n## Guardrails\n\n")
    for k in ["stage2_holdout_accessed", "model_forward_executed", "training_executed",
               "existing_artifact_modified", "label_used_as_selector",
               "label_used_for_group_analysis_only"]:
        lines.append(f"- {k}: `{summary.get(k)}`\n")

    with open(REPORT_DIR / "padim_score_distribution_lesion_roi_boundary_report.md", "w", encoding="utf-8") as f:
        f.writelines(lines)
    log_info("report.md 저장")


def _finalize_with_errors():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(errors).to_csv(LOG_DIR / "errors.csv", index=False)
    print(f"[FAIL] {len(errors)} critical error(s).")
    sys.exit(2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-audit", action="store_true")
    parser.add_argument("--confirm-readonly", action="store_true")
    parser.add_argument("--confirm-stage1dev-only", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        run_dry()
    elif args.run_audit:
        if not args.confirm_readonly or not args.confirm_stage1dev_only:
            print("[ERROR] --confirm-readonly 와 --confirm-stage1dev-only 필요")
            sys.exit(2)
        run_audit()
    else:
        print("[ERROR] --dry-run 또는 --run-audit 플래그 필요. bare run 금지 (exit 2).")
        sys.exit(2)


if __name__ == "__main__":
    main()
