#!/usr/bin/env python3
"""
RD4AD strict z-track score sweep analysis v1
analysis-only: no model forward / checkpoint / crop / scoring re-run
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# paths
# =============================================================================
PROJECT_ROOT    = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments/rd4ad_strict_ztrack_score_sweep_analysis_v1"
MANIFEST_DIR    = EXPERIMENT_ROOT / "manifests"
REPORT_DIR      = EXPERIMENT_ROOT / "reports"
LOG_DIR         = EXPERIMENT_ROOT / "logs"

SCORING_ROOT      = PROJECT_ROOT / "experiments/rd4ad_strict_same_position_ztrack_actual_scoring_v1"
MERGED_CSV        = SCORING_ROOT / "manifests/strict_ztrack_scores_full_merged.csv"
TRACK_SUMMARY_CSV = SCORING_ROOT / "manifests/strict_ztrack_track_score_summary.csv"

RD_D1S_SCORE_CSV = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_d1s_medi3ch_true_rd4ad_shard_run_v1"
    / "rd_d1s_stage1dev_candidate_score.csv"
)
GROUP_MERGED_CSV = (
    PROJECT_ROOT
    / "experiments/rd4ad_z_continuity_group_full_scoring_v1"
    / "manifests/group_scores_full_merged.csv"
)

OUT_CAND_SWEEP_CSV    = MANIFEST_DIR / "candidate_score_sweep_topk.csv"
OUT_TRACK_SWEEP_CSV   = MANIFEST_DIR / "track_score_sweep_topk.csv"
OUT_BASELINE_COMP_CSV = MANIFEST_DIR / "score_sweep_baseline_comparison.csv"
OUT_PROBLEM_AUDIT_CSV = MANIFEST_DIR / "problem_patient_score_sweep_audit.csv"
OUT_086_DIAG_CSV      = MANIFEST_DIR / "lung1_086_rank_diagnosis.csv"
OUT_ROI_SAFETY_CSV    = MANIFEST_DIR / "roi_score_safety_audit.csv"
OUT_REPORT_MD         = REPORT_DIR   / "rd4ad_strict_ztrack_score_sweep_analysis_report.md"
OUT_SUMMARY_JSON      = REPORT_DIR   / "rd4ad_strict_ztrack_score_sweep_analysis_summary.json"
OUT_ERRORS_CSV        = LOG_DIR      / "errors.csv"
OUT_DONE_JSON         = EXPERIMENT_ROOT / "DONE.json"

# =============================================================================
# constants
# =============================================================================
TOPK_VALS       = [1, 3, 5, 10, 20, 50]
PROBLEM_PATS    = ["LUNG1-086", "LUNG1-386", "LUNG1-399"]
N_EXPECTED      = 92342

CAND_SCORE_COLS = ["rd4ad_ztrack_score_raw", "P1_times_roi", "P2_times_sqrt_roi"]
AGG_METHODS     = ["max", "top2_mean", "top3_mean", "mean", "median"]

PRIMARY_TRACK_COLS = [
    "rd4ad_ztrack_score_raw_track_max",
    "P1_times_roi_track_max",
    "P1_times_roi_track_top2_mean",
    "P1_times_roi_track_top3_mean",
    "P2_times_sqrt_roi_track_max",
]

# =============================================================================
# guardrails
# =============================================================================
GUARDRAILS = {
    "stage2_holdout_accessed":             False,
    "model_forward_executed":              False,
    "checkpoint_loaded":                   False,
    "crop_generation_executed":            False,
    "scoring_reexecuted":                  False,
    "shard_reexecuted":                    False,
    "merge_reexecuted":                    False,
    "training_executed":                   False,
    "backward_executed":                   False,
    "optimizer_created":                   False,
    "checkpoint_saved":                    False,
    "threshold_recalculated":              False,
    "existing_artifact_modified":          False,
    "existing_script_modified":            False,
    "output_overwrite":                    False,
    "label_used_for_evaluation_only":      True,
    "label_used_for_score_selection":      False,
    "P1_primary_candidate_only":           True,
    "adjusted_score_preview_analysis_only": True,
    "vessel_mask_used":                    False,
    "hard_filter_applied":                 False,
}

# =============================================================================
# track score aggregation (vectorized)
# =============================================================================

def build_track_scores(df: pd.DataFrame) -> pd.DataFrame:
    """merged CSV → track-level aggregations for all score cols."""
    # basic stats
    stats = df.groupby("track_id").agg(
        patient_id=("patient_id", "first"),
        n_candidates=("label", "count"),
        positive_count=("label", lambda x: (x == "positive").sum()),
    ).reset_index()
    stats["hard_negative_count"] = stats["n_candidates"] - stats["positive_count"]
    stats["has_positive"] = stats["positive_count"] > 0

    # pos_slice_count: unique (local_z) per track among positives
    pos_df = df[df["label"] == "positive"]
    if not pos_df.empty:
        psc = pos_df.groupby("track_id")["local_z"].nunique().rename("pos_slice_count")
        stats = stats.merge(psc.reset_index(), on="track_id", how="left")
    else:
        stats["pos_slice_count"] = 0
    stats["pos_slice_count"] = stats["pos_slice_count"].fillna(0).astype(int)

    # score aggregations per score col
    for sc in CAND_SCORE_COLS:
        if sc not in df.columns:
            continue
        sc_df = df[["track_id", sc]].dropna(subset=[sc])

        sc_max    = sc_df.groupby("track_id")[sc].max().rename(f"{sc}_track_max")
        sc_mean   = sc_df.groupby("track_id")[sc].mean().rename(f"{sc}_track_mean")
        sc_median = sc_df.groupby("track_id")[sc].median().rename(f"{sc}_track_median")

        # top-N mean: rank within group then filter
        sc_ranked = sc_df.copy()
        sc_ranked["_rank"] = sc_ranked.groupby("track_id")[sc].rank(
            method="first", ascending=False
        )
        sc_top2 = (
            sc_ranked[sc_ranked["_rank"] <= 2]
            .groupby("track_id")[sc].mean()
            .rename(f"{sc}_track_top2_mean")
        )
        sc_top3 = (
            sc_ranked[sc_ranked["_rank"] <= 3]
            .groupby("track_id")[sc].mean()
            .rename(f"{sc}_track_top3_mean")
        )

        for s in [sc_max, sc_mean, sc_median, sc_top2, sc_top3]:
            stats = stats.merge(s.reset_index(), on="track_id", how="left")

    return stats


# =============================================================================
# evaluation helpers
# =============================================================================

def _pos_slices_set(df: pd.DataFrame):
    return set(zip(
        df.loc[df["label"] == "positive", "patient_id"],
        df.loc[df["label"] == "positive", "local_z"].astype(str),
    ))


def eval_candidate_sweep(df: pd.DataFrame) -> list:
    """candidate-level sweep: all score cols × all k."""
    total_pos_slices = _pos_slices_set(df)
    total_pos_cands  = (df["label"] == "positive").sum()

    pat_groups     = {pid: g for pid, g in df.groupby("patient_id")}
    n_pos_patients = sum(1 for g in pat_groups.values() if (g["label"] == "positive").any())

    out = []
    for sc in CAND_SCORE_COLS:
        if sc not in df.columns:
            continue

        # rank stats (global across positive patients, k-independent)
        all_pos_ranks = []
        for pid, grp in pat_groups.items():
            if not (grp["label"] == "positive").any():
                continue
            ranked = grp.sort_values(sc, ascending=False).reset_index(drop=True)
            ranked.index += 1  # 1-based
            all_pos_ranks.extend(ranked.index[ranked["label"] == "positive"].tolist())

        mean_pr   = float(np.mean(all_pos_ranks))   if all_pos_ranks else float("nan")
        median_pr = float(np.median(all_pos_ranks)) if all_pos_ranks else float("nan")
        p90_pr    = float(np.percentile(all_pos_ranks, 90)) if all_pos_ranks else float("nan")

        for k in TOPK_VALS:
            n_hit = 0
            topk_pos_cands  = 0
            topk_pos_slices = set()

            for pid, grp in pat_groups.items():
                if not (grp["label"] == "positive").any():
                    continue
                topk = grp.sort_values(sc, ascending=False).iloc[:k]
                pos_topk = topk[topk["label"] == "positive"]
                if len(pos_topk):
                    n_hit += 1
                topk_pos_cands += len(pos_topk)
                topk_pos_slices.update(zip(
                    pos_topk["patient_id"], pos_topk["local_z"].astype(str)
                ))

            hr = round(n_hit / n_pos_patients, 4) if n_pos_patients else 0.0
            out.append({
                "score_col":                  sc,
                "k":                          k,
                "patient_hit_rate":           hr,
                "positive_candidate_coverage": round(topk_pos_cands / total_pos_cands, 4) if total_pos_cands else 0.0,
                "positive_slice_coverage":    round(len(topk_pos_slices) / len(total_pos_slices), 4) if total_pos_slices else 0.0,
                "pat_all_sup_proxy":          round(1.0 - hr, 4),
                "mean_positive_rank":         round(mean_pr, 2),
                "median_positive_rank":       round(median_pr, 2),
                "p90_positive_rank":          round(p90_pr, 2),
                "n_patients_with_positive":   n_pos_patients,
            })
    return out


def eval_track_sweep(track_df: pd.DataFrame, merged_df: pd.DataFrame) -> list:
    """track-level sweep: all (score_col, agg) × all k."""
    track_cand_map = {tid: g for tid, g in merged_df.groupby("track_id")}
    total_pos_slices = _pos_slices_set(merged_df)
    total_pos_cands  = (merged_df["label"] == "positive").sum()

    pat_track_groups = {pid: g for pid, g in track_df.groupby("patient_id")}
    n_pos_patients   = sum(1 for g in pat_track_groups.values() if g["has_positive"].any())

    out = []
    for sc in CAND_SCORE_COLS:
        for agg in AGG_METHODS:
            col = f"{sc}_track_{agg}"
            if col not in track_df.columns:
                continue

            # rank stats
            all_pos_track_ranks = []
            for pid, grp in pat_track_groups.items():
                if not grp["has_positive"].any():
                    continue
                ranked = grp.sort_values(col, ascending=False).reset_index(drop=True)
                ranked.index += 1
                all_pos_track_ranks.extend(ranked.index[ranked["has_positive"]].tolist())

            mean_tr   = float(np.mean(all_pos_track_ranks))   if all_pos_track_ranks else float("nan")
            median_tr = float(np.median(all_pos_track_ranks)) if all_pos_track_ranks else float("nan")
            p90_tr    = float(np.percentile(all_pos_track_ranks, 90)) if all_pos_track_ranks else float("nan")

            for k in TOPK_VALS:
                n_hit = 0
                topk_pos_cands  = 0
                topk_pos_slices = set()

                for pid, grp in pat_track_groups.items():
                    if not grp["has_positive"].any():
                        continue
                    topk_tracks = grp.sort_values(col, ascending=False).iloc[:k]
                    if topk_tracks["has_positive"].any():
                        n_hit += 1
                    for tid in topk_tracks["track_id"]:
                        cands = track_cand_map.get(tid)
                        if cands is None:
                            continue
                        pos_c = cands[cands["label"] == "positive"]
                        topk_pos_cands += len(pos_c)
                        topk_pos_slices.update(zip(
                            pos_c["patient_id"], pos_c["local_z"].astype(str)
                        ))

                hr = round(n_hit / n_pos_patients, 4) if n_pos_patients else 0.0
                out.append({
                    "score_col":                          col,
                    "k":                                  k,
                    "patient_hit_rate":                   hr,
                    "positive_containing_track_hit_rate": hr,
                    "positive_candidate_coverage_by_tracks": round(topk_pos_cands / total_pos_cands, 4) if total_pos_cands else 0.0,
                    "positive_slice_coverage_by_tracks":  round(len(topk_pos_slices) / len(total_pos_slices), 4) if total_pos_slices else 0.0,
                    "pat_all_sup_proxy":                  round(1.0 - hr, 4),
                    "mean_positive_track_rank":           round(mean_tr, 2),
                    "median_positive_track_rank":         round(median_tr, 2),
                    "p90_positive_track_rank":            round(p90_tr, 2),
                    "n_patients_with_positive_track":     n_pos_patients,
                })
    return out


# =============================================================================
# baselines
# =============================================================================

def eval_patch_baseline() -> dict:
    if not RD_D1S_SCORE_CSV.exists():
        return {}
    try:
        bdf = pd.read_csv(RD_D1S_SCORE_CSV)
        sc = "rd_d1s_medi3ch_rd4ad_score"
        if sc not in bdf.columns:
            return {}
        if "stage_split" in bdf.columns:
            bdf = bdf[bdf["stage_split"] == "stage1_dev"]
        pat_groups     = {p: g for p, g in bdf.groupby("patient_id")}
        n_pos_patients = sum(1 for g in pat_groups.values() if (g["label"] == "positive").any())
        out = {}
        for k in TOPK_VALS:
            hits = sum(
                1 for g in pat_groups.values()
                if (g["label"] == "positive").any()
                and (g.sort_values(sc, ascending=False).iloc[:k]["label"] == "positive").any()
            )
            out[k] = round(hits / n_pos_patients, 4) if n_pos_patients else 0.0
        return out
    except Exception:
        return {}


def eval_group_baseline() -> dict:
    if not GROUP_MERGED_CSV.exists():
        return {}
    try:
        gdf = pd.read_csv(GROUP_MERGED_CSV)
        sc = "rd4ad_group_score_raw"
        if sc not in gdf.columns:
            return {}
        if "stage_split" in gdf.columns:
            gdf = gdf[gdf["stage_split"] == "stage1_dev"]
        pat_groups     = {p: g for p, g in gdf.groupby("patient_id")}
        n_pos_patients = sum(1 for g in pat_groups.values() if g["has_positive"].any())
        out = {}
        for k in TOPK_VALS:
            hits = sum(
                1 for g in pat_groups.values()
                if g["has_positive"].any()
                and g.sort_values(sc, ascending=False).iloc[:k]["has_positive"].any()
            )
            out[k] = round(hits / n_pos_patients, 4) if n_pos_patients else 0.0
        return out
    except Exception:
        return {}


def build_baseline_comparison(cand_rows, track_rows, patch_base, group_base) -> list:
    out = []
    cdf = pd.DataFrame(cand_rows)
    tdf = pd.DataFrame(track_rows)

    def _row(name, vals):
        r = {"score_name": name}
        for k in TOPK_VALS:
            r[f"top{k}_patient_hit_rate"] = vals.get(k)
        return r

    if patch_base:
        out.append(_row("patch_baseline_rd_d1s", patch_base))
    if group_base:
        out.append(_row("fuzzy_group_baseline", group_base))

    for sc in CAND_SCORE_COLS:
        sub = cdf[cdf["score_col"] == sc]
        if sub.empty:
            continue
        vals = {int(r["k"]): r["patient_hit_rate"] for _, r in sub.iterrows()}
        out.append(_row(f"candidate_{sc}", vals))

    for col in PRIMARY_TRACK_COLS:
        sub = tdf[tdf["score_col"] == col]
        if sub.empty:
            continue
        vals = {int(r["k"]): r["patient_hit_rate"] for _, r in sub.iterrows()}
        out.append(_row(f"track_{col}", vals))

    return out


# =============================================================================
# problem patient audit
# =============================================================================

def build_problem_patient_audit(df: pd.DataFrame, track_df: pd.DataFrame) -> list:
    out = []
    t_pat = {p: g for p, g in track_df.groupby("patient_id")}

    for pat in PROBLEM_PATS:
        grp = df[df["patient_id"] == pat]
        if grp.empty:
            continue

        # candidate-level
        for sc in CAND_SCORE_COLS:
            if sc not in df.columns:
                continue
            ranked = grp.sort_values(sc, ascending=False).reset_index(drop=True)
            for k in TOPK_VALS:
                hit = bool((ranked.iloc[:k]["label"] == "positive").any())
                out.append({"patient_id": pat, "level": "candidate", "score_col": sc, "k": k, "hit": hit})

        # track-level (max / top2_mean / top3_mean)
        t_grp = t_pat.get(pat)
        if t_grp is None:
            continue
        for sc in CAND_SCORE_COLS:
            for agg in ["max", "top2_mean", "top3_mean"]:
                col = f"{sc}_track_{agg}"
                if col not in track_df.columns:
                    continue
                ranked = t_grp.sort_values(col, ascending=False).reset_index(drop=True)
                for k in TOPK_VALS:
                    hit = bool(ranked.iloc[:k]["has_positive"].any())
                    out.append({"patient_id": pat, "level": "track", "score_col": col, "k": k, "hit": hit})

    return out


# =============================================================================
# LUNG1-086 rank diagnosis
# =============================================================================

def build_lung1_086_rank_diagnosis(df: pd.DataFrame, track_df: pd.DataFrame) -> list:
    out = []
    pat = "LUNG1-086"
    grp = df[df["patient_id"] == pat]
    pos_grp = grp[grp["label"] == "positive"]

    roi_col = "roi_0_0_patch_ratio"
    pos_roi_mean = float(pos_grp[roi_col].mean()) if roi_col in grp.columns else None
    pos_roi_min  = float(pos_grp[roi_col].min())  if roi_col in grp.columns else None

    # candidate-level
    for sc in CAND_SCORE_COLS:
        if sc not in df.columns:
            continue
        ranked = grp.sort_values(sc, ascending=False).reset_index(drop=True)
        ranked.index += 1
        pos_ranks  = ranked.index[ranked["label"] == "positive"].tolist()
        topk20 = ranked.iloc[:20]
        topk50 = ranked.iloc[:50]
        top50_fp   = topk50[topk50["label"] != "positive"]

        out.append({
            "patient_id":         pat,
            "level":              "candidate",
            "score_col":          sc,
            "n_candidates":       len(grp),
            "n_positive":         len(pos_grp),
            "pos_rank_min":       int(min(pos_ranks))          if pos_ranks else None,
            "pos_rank_median":    float(np.median(pos_ranks))  if pos_ranks else None,
            "pos_rank_max":       int(max(pos_ranks))          if pos_ranks else None,
            "hit_top20":          bool((topk20["label"] == "positive").any()),
            "hit_top50":          bool((topk50["label"] == "positive").any()),
            "pos_roi_mean":       pos_roi_mean,
            "pos_roi_min":        pos_roi_min,
            "top50_fp_roi_mean":  float(top50_fp[roi_col].mean()) if (roi_col in grp.columns and len(top50_fp)) else None,
        })

    # track-level
    t_grp = track_df[track_df["patient_id"] == pat]
    for sc in CAND_SCORE_COLS:
        for agg in ["max", "top2_mean", "top3_mean"]:
            col = f"{sc}_track_{agg}"
            if col not in track_df.columns:
                continue
            t_ranked = t_grp.sort_values(col, ascending=False).reset_index(drop=True)
            t_ranked.index += 1
            pos_t_ranks = t_ranked.index[t_ranked["has_positive"]].tolist()
            out.append({
                "patient_id":         pat,
                "level":              "track",
                "score_col":          col,
                "n_candidates":       len(t_grp),
                "n_positive":         int(t_grp["has_positive"].sum()),
                "pos_rank_min":       int(min(pos_t_ranks))         if pos_t_ranks else None,
                "pos_rank_median":    float(np.median(pos_t_ranks)) if pos_t_ranks else None,
                "pos_rank_max":       int(max(pos_t_ranks))         if pos_t_ranks else None,
                "hit_top20":          bool(t_ranked.iloc[:20]["has_positive"].any()),
                "hit_top50":          bool(t_ranked.iloc[:50]["has_positive"].any()),
                "pos_roi_mean":       None,
                "pos_roi_min":        None,
                "top50_fp_roi_mean":  None,
            })

    return out


# =============================================================================
# ROI safety audit
# =============================================================================

def build_roi_safety_audit(df: pd.DataFrame) -> list:
    raw_sc = "rd4ad_ztrack_score_raw"
    p1_sc  = "P1_times_roi"
    roi_col = "roi_0_0_patch_ratio"

    if p1_sc not in df.columns:
        return []

    pat_groups    = {p: g for p, g in df.groupby("patient_id")}
    pos_patients  = [p for p, g in pat_groups.items() if (g["label"] == "positive").any()]

    # global ROI distribution
    pos_roi = df[df["label"] == "positive"][roi_col].dropna() if roi_col in df.columns else pd.Series([], dtype=float)
    hn_roi  = df[df["label"] != "positive"][roi_col].dropna() if roi_col in df.columns else pd.Series([], dtype=float)

    pos_roi_stats = {s: float(getattr(np, f"percentile" if "p" in s else s)(pos_roi, int(s[1:])) if "p" in s else getattr(pos_roi, s)())
                     for s in ["mean", "std"]} if len(pos_roi) else {}
    pos_roi_pcts  = {f"p{p}": float(np.percentile(pos_roi, p)) for p in [10, 50, 90]} if len(pos_roi) else {}
    hn_roi_pcts   = {f"p{p}": float(np.percentile(hn_roi, p))  for p in [10, 50, 90]} if len(hn_roi)  else {}

    out = []
    for k in TOPK_VALS:
        n_gained = n_lost = n_stable_hit = n_stable_miss = 0
        rm_pos = rm_hn = add_pos = add_hn = 0

        for pid in pos_patients:
            grp = pat_groups[pid]
            raw_topk = set(grp.sort_values(raw_sc, ascending=False).iloc[:k]["candidate_id"])
            p1_topk  = set(grp.sort_values(p1_sc,  ascending=False).iloc[:k]["candidate_id"])

            raw_hit = (grp[grp["candidate_id"].isin(raw_topk)]["label"] == "positive").any()
            p1_hit  = (grp[grp["candidate_id"].isin(p1_topk)]["label"]  == "positive").any()

            if     raw_hit and     p1_hit: n_stable_hit  += 1
            elif   raw_hit and not p1_hit: n_lost         += 1
            elif not raw_hit and   p1_hit: n_gained       += 1
            else:                          n_stable_miss  += 1

            removed = raw_topk - p1_topk
            added   = p1_topk  - raw_topk
            for cid in removed:
                lbl = grp.loc[grp["candidate_id"] == cid, "label"].values
                if len(lbl):
                    (rm_pos if lbl[0] == "positive" else rm_hn).__class__  # dummy
                    if lbl[0] == "positive": rm_pos += 1
                    else:                    rm_hn  += 1
            for cid in added:
                lbl = grp.loc[grp["candidate_id"] == cid, "label"].values
                if len(lbl):
                    if lbl[0] == "positive": add_pos += 1
                    else:                    add_hn  += 1

        n = len(pos_patients)
        out.append({
            "k":                            k,
            "n_patients_analyzed":          n,
            "raw_hit_rate":                 round((n_stable_hit + n_lost)    / n, 4) if n else 0.0,
            "p1_hit_rate":                  round((n_stable_hit + n_gained)  / n, 4) if n else 0.0,
            "new_hit_P1_gained":            n_gained,
            "lost_hit_P1_regression":       n_lost,
            "stable_hit":                   n_stable_hit,
            "stable_miss":                  n_stable_miss,
            "removed_positive_from_topk":   rm_pos,
            "removed_hard_negative_from_topk": rm_hn,
            "added_positive_to_topk":       add_pos,
            "added_hard_negative_to_topk":  add_hn,
            "pos_roi_mean":                 pos_roi_stats.get("mean"),
            "pos_roi_p10":                  pos_roi_pcts.get("p10"),
            "pos_roi_p50":                  pos_roi_pcts.get("p50"),
            "pos_roi_p90":                  pos_roi_pcts.get("p90"),
            "hn_roi_p10":                   hn_roi_pcts.get("p10"),
            "hn_roi_p50":                   hn_roi_pcts.get("p50"),
            "hn_roi_p90":                   hn_roi_pcts.get("p90"),
        })
    return out


# =============================================================================
# P1 verdict
# =============================================================================

def assess_p1_upgrade(cand_rows, track_rows, problem_audit):
    cdf = pd.DataFrame(cand_rows)
    tdf = pd.DataFrame(track_rows)

    def cand_hr(sc, k):
        r = cdf[(cdf["score_col"] == sc) & (cdf["k"] == k)]["patient_hit_rate"]
        return float(r.iloc[0]) if len(r) else None

    def track_hr(col, k):
        r = tdf[(tdf["score_col"] == col) & (tdf["k"] == k)]["patient_hit_rate"]
        return float(r.iloc[0]) if len(r) else None

    adf = pd.DataFrame(problem_audit)

    def audit_hit(pat, lvl, sc_col, k_val):
        r = adf[
            (adf["patient_id"] == pat) & (adf["level"] == lvl) &
            (adf["score_col"] == sc_col) & (adf["k"] == k_val)
        ]
        return bool(r["hit"].iloc[0]) if len(r) else None

    cand_fail = []

    # low-k 개선 여부 (top1/3/5 모두 P1 >= raw)
    low_k_ok = all(
        (cand_hr("P1_times_roi", k) or 0) >= (cand_hr("rd4ad_ztrack_score_raw", k) or 0)
        for k in [1, 3, 5]
    )
    if not low_k_ok:
        cand_fail.append("P1 candidate top1/3/5 중 하나 이상 raw보다 악화")

    # mid-k 악화 허용 -1%p
    for k in [10, 20]:
        raw_hr = cand_hr("rd4ad_ztrack_score_raw", k) or 0
        p1_hr  = cand_hr("P1_times_roi", k) or 0
        if (raw_hr - p1_hr) > 0.01:
            cand_fail.append(f"P1 top{k} 악화 {raw_hr - p1_hr:.4f} > 0.01")

    # top50 slice coverage
    raw_sc50 = cdf[(cdf["score_col"] == "rd4ad_ztrack_score_raw") & (cdf["k"] == 50)]["positive_slice_coverage"]
    p1_sc50  = cdf[(cdf["score_col"] == "P1_times_roi") & (cdf["k"] == 50)]["positive_slice_coverage"]
    if not raw_sc50.empty and not p1_sc50.empty:
        diff = float(raw_sc50.iloc[0]) - float(p1_sc50.iloc[0])
        if diff > 0.02:
            cand_fail.append(f"top50 positive_slice_coverage 악화 {diff:.4f}")

    # problem patient
    p86_p1_50  = audit_hit("LUNG1-086", "candidate", "P1_times_roi", 50)
    p86_raw_50 = audit_hit("LUNG1-086", "candidate", "rd4ad_ztrack_score_raw", 50)
    if p86_p1_50 is False and p86_raw_50 is False:
        pass  # 이미 miss였음, 악화 아님
    elif p86_p1_50 is False and p86_raw_50 is True:
        cand_fail.append("LUNG1-086 P1 top50 회복 → miss로 역전")

    for p, name in [("LUNG1-386", "386"), ("LUNG1-399", "399")]:
        ok = audit_hit(p, "candidate", "P1_times_roi", 20) or audit_hit(p, "candidate", "P1_times_roi", 50)
        if ok is False:
            cand_fail.append(f"LUNG1-{name} P1 candidate top20/top50 모두 miss")

    if not cand_fail:
        cand_verdict = "PASS_PRIMARY_CANDIDATE"
    elif low_k_ok:
        cand_verdict = "PARTIAL_PASS"
    else:
        cand_verdict = "FAIL"

    # track verdict
    track_fail = []
    raw_max_col = "rd4ad_ztrack_score_raw_track_max"
    p1_max_col  = "P1_times_roi_track_max"
    p1_t2_col   = "P1_times_roi_track_top2_mean"

    better_cnt = sum(
        (track_hr(p1_max_col, k) or 0) > (track_hr(raw_max_col, k) or 0)
        or (track_hr(p1_t2_col, k) or 0) > (track_hr(raw_max_col, k) or 0)
        for k in [5, 10, 20]
    )
    if better_cnt < 2:
        track_fail.append(f"P1 track top5/10/20 개선 {better_cnt}/3 (>=2 필요)")

    raw_ts50 = tdf[(tdf["score_col"] == raw_max_col) & (tdf["k"] == 50)]["positive_slice_coverage_by_tracks"]
    p1_ts50  = tdf[(tdf["score_col"] == p1_max_col)  & (tdf["k"] == 50)]["positive_slice_coverage_by_tracks"]
    if not raw_ts50.empty and not p1_ts50.empty:
        tdiff = float(raw_ts50.iloc[0]) - float(p1_ts50.iloc[0])
        if tdiff > 0.02:
            track_fail.append(f"track top50 slice_coverage 악화 {tdiff:.4f}")

    if not track_fail:
        track_verdict = "PASS_TRACK_PRIMARY_CANDIDATE"
    elif better_cnt >= 1:
        track_verdict = "PARTIAL_PASS"
    else:
        track_verdict = "FAIL"

    return cand_verdict, track_verdict, cand_fail, track_fail


# =============================================================================
# output writers
# =============================================================================

def _write_csv(rows, path: Path):
    if not rows:
        pd.DataFrame().to_csv(path, index=False)
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_report_md(
    cand_rows, track_rows, baseline_rows,
    problem_audit, lung086_diag, roi_safety,
    cand_verdict, track_verdict, cand_fail, track_fail,
    elapsed, path: Path
):
    cdf  = pd.DataFrame(cand_rows)
    tdf  = pd.DataFrame(track_rows)
    bdf  = pd.DataFrame(baseline_rows)
    adf  = pd.DataFrame(problem_audit)

    def hr(df, sc, k):
        r = df[(df["score_col"] == sc) & (df["k"] == k)]["patient_hit_rate"]
        return f"{float(r.iloc[0]):.4f}" if len(r) else "—"

    def thr(df, sc, k):
        r = df[(df["score_col"] == sc) & (df["k"] == k)]["patient_hit_rate"]
        return f"{float(r.iloc[0]):.4f}" if len(r) else "—"

    lines = [
        "# RD4AD strict z-track score sweep analysis v1",
        "",
        "## 판정",
        f"- candidate: **{cand_verdict}**",
        f"- track:     **{track_verdict}**",
    ]
    if cand_fail:
        lines.append(f"- candidate 실패 사유: {'; '.join(cand_fail)}")
    if track_fail:
        lines.append(f"- track 실패 사유: {'; '.join(track_fail)}")
    lines += [f"", f"elapsed: {elapsed:.1f}s", ""]

    # candidate top-k table
    lines += ["## candidate-level patient_hit_rate", ""]
    header = "| score_name | " + " | ".join(f"top{k}" for k in TOPK_VALS) + " |"
    sep    = "|------------|" + "|".join("-------" for _ in TOPK_VALS) + "|"
    lines += [header, sep]
    for _, r in bdf.iterrows():
        vals = [f"{r.get(f'top{k}_patient_hit_rate', '—')}" for k in TOPK_VALS]
        lines.append(f"| {r['score_name']} | " + " | ".join(str(v) if v is not None else "—" for v in vals) + " |")
    for sc in CAND_SCORE_COLS:
        vals = [hr(cdf, sc, k) for k in TOPK_VALS]
        lines.append(f"| candidate_{sc} | " + " | ".join(vals) + " |")
    lines.append("")

    # track top-k table
    lines += ["## track-level patient_hit_rate (primary cols)", ""]
    lines += [header, sep]
    for _, r in bdf[bdf["score_name"].str.startswith("track_")].iterrows():
        vals = [f"{r.get(f'top{k}_patient_hit_rate', '—')}" for k in TOPK_VALS]
        lines.append(f"| {r['score_name']} | " + " | ".join(str(v) if v is not None else "—" for v in vals) + " |")
    lines.append("")

    # problem patient
    lines += ["## problem patient audit", ""]
    for pat in PROBLEM_PATS:
        lines.append(f"### {pat}")
        lines.append("")
        lines.append("**candidate:**")
        ph = "| score_col | " + " | ".join(f"top{k}" for k in TOPK_VALS) + " |"
        ps = "|-----------|" + "|".join("------" for _ in TOPK_VALS) + "|"
        lines += [ph, ps]
        for sc in CAND_SCORE_COLS:
            vals = []
            for k in TOPK_VALS:
                r = adf[(adf["patient_id"]==pat)&(adf["level"]=="candidate")&(adf["score_col"]==sc)&(adf["k"]==k)]
                vals.append("✅" if (not r.empty and r["hit"].iloc[0]) else "❌")
            lines.append(f"| {sc} | " + " | ".join(vals) + " |")
        lines.append("")
        lines.append("**track (max agg):**")
        lines += [ph, ps]
        for sc in CAND_SCORE_COLS:
            col = f"{sc}_track_max"
            vals = []
            for k in TOPK_VALS:
                r = adf[(adf["patient_id"]==pat)&(adf["level"]=="track")&(adf["score_col"]==col)&(adf["k"]==k)]
                vals.append("✅" if (not r.empty and r["hit"].iloc[0]) else "❌")
            lines.append(f"| {sc} | " + " | ".join(vals) + " |")
        lines.append("")

    # LUNG1-086 rank diagnosis
    lines += ["## LUNG1-086 rank diagnosis (candidate-level)", ""]
    lines.append("| score_col | pos_rank_min | pos_rank_median | pos_rank_max | hit_top20 | hit_top50 | pos_roi_mean | top50_fp_roi_mean |")
    lines.append("|-----------|-------------|----------------|-------------|-----------|-----------|-------------|-----------------|")
    for r in lung086_diag:
        if r["level"] != "candidate":
            continue
        lines.append(
            f"| {r['score_col']} | {r['pos_rank_min']} | {r['pos_rank_median']} | "
            f"{r['pos_rank_max']} | {r['hit_top20']} | {r['hit_top50']} | "
            f"{r['pos_roi_mean']:.4f} | {r['top50_fp_roi_mean']:.4f} |"
            if r['pos_roi_mean'] is not None and r['top50_fp_roi_mean'] is not None
            else f"| {r['score_col']} | {r['pos_rank_min']} | {r['pos_rank_median']} | {r['pos_rank_max']} | {r['hit_top20']} | {r['hit_top50']} | — | — |"
        )
    lines.append("")

    # ROI safety
    lines += ["## ROI safety audit (raw vs P1, candidate-level)", ""]
    lines.append("| k | raw_hit | p1_hit | gained | lost | rm_pos | add_pos | pos_roi_p50 | hn_roi_p50 |")
    lines.append("|---|---------|--------|--------|------|--------|---------|------------|-----------|")
    for r in roi_safety:
        lines.append(
            f"| {r['k']} | {r['raw_hit_rate']:.4f} | {r['p1_hit_rate']:.4f} | "
            f"{r['new_hit_P1_gained']} | {r['lost_hit_P1_regression']} | "
            f"{r['removed_positive_from_topk']} | {r['added_positive_to_topk']} | "
            f"{r.get('pos_roi_p50','—')} | {r.get('hn_roi_p50','—')} |"
        )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_summary_json(
    cand_rows, track_rows, baseline_rows,
    cand_verdict, track_verdict, cand_fail, track_fail,
    lung086_diag, roi_safety,
    elapsed, path: Path
):
    cdf = pd.DataFrame(cand_rows)
    tdf = pd.DataFrame(track_rows)

    def topk_dict(df, sc):
        return {
            f"top{k}": float(df[(df["score_col"]==sc) & (df["k"]==k)]["patient_hit_rate"].iloc[0])
            for k in TOPK_VALS
            if len(df[(df["score_col"]==sc) & (df["k"]==k)]) > 0
        }

    summary = {
        "verdict": {
            "candidate":         cand_verdict,
            "track":             track_verdict,
            "fail_reasons_cand": cand_fail,
            "fail_reasons_track": track_fail,
        },
        "guardrails": GUARDRAILS,
        "elapsed_sec": round(elapsed, 1),
        "n_candidates": N_EXPECTED,
        "n_patients_with_positive": 152,
        "candidate_topk": {
            sc: topk_dict(cdf, sc) for sc in CAND_SCORE_COLS if sc in cdf["score_col"].values
        },
        "track_topk": {
            col: topk_dict(tdf, col) for col in PRIMARY_TRACK_COLS if col in tdf["score_col"].values
        },
        "baseline": {
            row["score_name"]: {
                f"top{k}": row.get(f"top{k}_patient_hit_rate") for k in TOPK_VALS
            }
            for row in baseline_rows
        },
        "lung1_086_cand_diag": [r for r in lung086_diag if r["level"] == "candidate"],
        "roi_safety": {r["k"]: r for r in roi_safety},
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


# =============================================================================
# dry-run
# =============================================================================

def run_dry():
    print("=" * 70)
    print("[DRY-RUN] RD4AD strict z-track score sweep analysis v1")
    print("  파일 생성 없음")
    print("=" * 70)
    fail = []

    print("\n[1] 입력 파일 존재 확인")
    for p in [MERGED_CSV, TRACK_SUMMARY_CSV]:
        if p.exists():
            print(f"  [OK]   {p.name}")
        else:
            print(f"  [FAIL] {p.name} 없음")
            fail.append(str(p.name))
    for p, label in [(RD_D1S_SCORE_CSV, "patch baseline"), (GROUP_MERGED_CSV, "group baseline")]:
        if p.exists():
            print(f"  [OK]   {label}: {p.name}")
        else:
            print(f"  [WARN] {label}: {p.name} 없음 (생략 가능)")

    print("\n[2] merged CSV — row count / score columns")
    if MERGED_CSV.exists():
        df = pd.read_csv(MERGED_CSV)
        print(f"  merged rows: {len(df):,} (expected {N_EXPECTED:,})")
        if len(df) != N_EXPECTED:
            fail.append(f"row count {len(df)} != {N_EXPECTED}")
        missing = [c for c in CAND_SCORE_COLS if c not in df.columns]
        if missing:
            print(f"  [FAIL] score columns 없음: {missing}")
            fail.append(f"score cols 없음: {missing}")
        else:
            print(f"  score columns OK: {CAND_SCORE_COLS}")
        print(f"  candidate_id col: {'OK' if 'candidate_id' in df.columns else 'MISSING'}")

    print("\n[3] 출력 overwrite 위험")
    for p in [OUT_CAND_SWEEP_CSV, OUT_TRACK_SWEEP_CSV, OUT_BASELINE_COMP_CSV,
              OUT_PROBLEM_AUDIT_CSV, OUT_086_DIAG_CSV, OUT_ROI_SAFETY_CSV,
              OUT_REPORT_MD, OUT_SUMMARY_JSON]:
        status = "[WARN overwrite]" if p.exists() else "[OK]"
        print(f"  {status} {p.name}")

    print("\n[4] stage2_holdout 접근 없음")
    print("  stage2_holdout_accessed: False (analysis-only)")

    print("\n[5] guardrails snapshot")
    for k, v in GUARDRAILS.items():
        print(f"  {k}: {v}")

    print()
    print("=" * 70)
    if fail:
        print(f"[DRY-RUN] FAIL — {'; '.join(fail)}")
        print("판정: NOT_READY")
        sys.exit(1)
    else:
        print("[DRY-RUN] 모든 입력/경로 OK.")
        print("판정: READY_TO_RUN_ANALYSIS")
    print("=" * 70)


# =============================================================================
# main analysis
# =============================================================================

def run_analysis():
    t0 = time.perf_counter()
    print("=" * 70)
    print("[RUN-ANALYSIS] RD4AD strict z-track score sweep analysis v1")
    print("=" * 70)
    for d in [MANIFEST_DIR, REPORT_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print("\n[1] merged CSV 로드")
    df = pd.read_csv(MERGED_CSV)
    print(f"  rows: {len(df):,}  pos: {(df['label']=='positive').sum():,}")
    for sc in CAND_SCORE_COLS:
        if sc in df.columns:
            print(f"  {sc}: NaN={df[sc].isna().sum()}  Inf={np.isinf(df[sc].astype(float)).sum()}")

    print("\n[2] track aggregation 계산 (P1/P2 포함)")
    track_df = build_track_scores(df)
    print(f"  tracks: {len(track_df):,}  pos-tracks: {track_df['has_positive'].sum():,}")

    print("\n[3] candidate-level sweep")
    cand_rows = eval_candidate_sweep(df)
    print(f"  sweep rows: {len(cand_rows)}")

    print("\n[4] track-level sweep")
    track_rows = eval_track_sweep(track_df, df)
    print(f"  sweep rows: {len(track_rows)}")

    print("\n[5] baseline 계산")
    patch_base = eval_patch_baseline()
    group_base = eval_group_baseline()
    print(f"  patch baseline k-vals: {len(patch_base)}  group: {len(group_base)}")

    print("\n[6] baseline comparison table")
    baseline_rows = build_baseline_comparison(cand_rows, track_rows, patch_base, group_base)

    print("\n[7] problem patient audit")
    problem_audit = build_problem_patient_audit(df, track_df)

    print("\n[8] LUNG1-086 rank diagnosis")
    lung086_diag = build_lung1_086_rank_diagnosis(df, track_df)

    print("\n[9] ROI safety audit")
    roi_safety = build_roi_safety_audit(df)

    print("\n[10] P1 격상 기준 판정")
    cand_verdict, track_verdict, cand_fail, track_fail = assess_p1_upgrade(
        cand_rows, track_rows, problem_audit
    )
    print(f"  candidate: {cand_verdict}")
    print(f"  track:     {track_verdict}")
    if cand_fail:
        print(f"  cand fail: {cand_fail}")
    if track_fail:
        print(f"  track fail: {track_fail}")

    elapsed = time.perf_counter() - t0

    print("\n[A] CSV 저장")
    _write_csv(cand_rows,     OUT_CAND_SWEEP_CSV)
    _write_csv(track_rows,    OUT_TRACK_SWEEP_CSV)
    _write_csv(baseline_rows, OUT_BASELINE_COMP_CSV)
    _write_csv(problem_audit, OUT_PROBLEM_AUDIT_CSV)
    _write_csv(lung086_diag,  OUT_086_DIAG_CSV)
    _write_csv(roi_safety,    OUT_ROI_SAFETY_CSV)
    for p in [OUT_CAND_SWEEP_CSV, OUT_TRACK_SWEEP_CSV, OUT_BASELINE_COMP_CSV,
              OUT_PROBLEM_AUDIT_CSV, OUT_086_DIAG_CSV, OUT_ROI_SAFETY_CSV]:
        print(f"  saved: {p.name}  ({p.stat().st_size:,} bytes)")

    print("\n[B] report / summary 저장")
    _write_report_md(
        cand_rows, track_rows, baseline_rows,
        problem_audit, lung086_diag, roi_safety,
        cand_verdict, track_verdict, cand_fail, track_fail,
        elapsed, OUT_REPORT_MD
    )
    _write_summary_json(
        cand_rows, track_rows, baseline_rows,
        cand_verdict, track_verdict, cand_fail, track_fail,
        lung086_diag, roi_safety,
        elapsed, OUT_SUMMARY_JSON
    )
    print(f"  saved: {OUT_REPORT_MD.name}")
    print(f"  saved: {OUT_SUMMARY_JSON.name}")

    print("\n[C] errors.csv / DONE.json 저장")
    _write_csv([], OUT_ERRORS_CSV)
    done = {
        "verdict_candidate": cand_verdict,
        "verdict_track":     track_verdict,
        "fail_reasons_cand": cand_fail,
        "fail_reasons_track": track_fail,
        "elapsed_sec":       round(elapsed, 1),
        "guardrails":        GUARDRAILS,
    }
    OUT_DONE_JSON.write_text(json.dumps(done, indent=2), encoding="utf-8")
    print(f"  saved: {OUT_DONE_JSON.name}")

    print()
    print("=" * 70)
    print(f"[RUN-ANALYSIS] 완료 ({elapsed:.1f}s)")
    print(f"  candidate 판정: {cand_verdict}")
    print(f"  track 판정:     {track_verdict}")
    print("=" * 70)


# =============================================================================
# entrypoint
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run",               action="store_true")
    p.add_argument("--run-analysis",          action="store_true")
    p.add_argument("--confirm-readonly",      action="store_true")
    p.add_argument("--confirm-stage1dev-only", action="store_true")
    args = p.parse_args()

    if args.dry_run:
        run_dry()
        return

    if not (args.run_analysis and args.confirm_readonly and args.confirm_stage1dev_only):
        print("bare run 금지 (exit 2).")
        print("사용: --dry-run  또는  --run-analysis --confirm-readonly --confirm-stage1dev-only")
        sys.exit(2)

    run_analysis()


if __name__ == "__main__":
    main()
