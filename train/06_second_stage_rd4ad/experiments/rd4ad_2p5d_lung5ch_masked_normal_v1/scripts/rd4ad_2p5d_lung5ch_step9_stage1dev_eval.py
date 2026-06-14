"""
Step 9: stage1_dev candidate-level / track-level ranking & evaluation
evaluation only — no training, no model forward, no checkpoint save/modify, no stage2 access
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── 경로 ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
SCORE_CSV   = ROOT / "scoring/step8_stage1dev_v1/rd4ad_lung5ch_stage1dev_scores_v1.csv"
DONE_STEP8  = ROOT / "DONE_STEP8_SCORING.json"
PLAN_LOCK   = ROOT / "docs/FINAL_PLAN_LOCK.json"
OUT_MANIFESTS = ROOT / "manifests"
OUT_REPORTS   = ROOT / "reports"
OUT_LOGS      = ROOT / "logs"

BASELINE_CSV = Path("outputs/normal_based_stage2_verifier_audit"
                    "/rd_e1c2_lung_mip3ch_roipx_true_rd4ad_shard_run_v1"
                    "/rd_d1s_stage1dev_candidate_score.csv")

PROBLEM_PATIENTS = ["LUNG1-086", "LUNG1-386", "LUNG1-399"]
TOPK_LIST = [1, 3, 5, 10, 20, 50]
DENOMINATOR_CANDIDATES = 95995


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run-eval", action="store_true")
    ap.add_argument("--confirm-plan-lock", action="store_true")
    ap.add_argument("--confirm-no-stage2", action="store_true")
    ap.add_argument("--confirm-eval-only", action="store_true")
    return ap.parse_args()


# ── 전처리 헬퍼 ──────────────────────────────────────────────────────────────
def topk_mean(arr, k):
    if len(arr) == 0:
        return float("nan")
    a = sorted(arr, reverse=True)
    return float(np.mean(a[:k]))


def compute_track_scores(df_pt):
    """track_id별 aggregate score를 계산하고 DataFrame을 반환."""
    rows = []
    for tid, grp in df_pt.groupby("track_id"):
        raw_vals  = grp["rd4ad_lung5ch_score_raw"].tolist()
        p1_vals   = grp["P1_times_roi"].tolist()
        p2_vals   = grp["P2_times_sqrt_roi"].tolist()
        is_pos    = int((grp["label"] == "positive").any())
        rows.append({
            "track_id": tid,
            "patient_id": grp["patient_id"].iloc[0],
            "track_len": len(grp),
            "is_positive": is_pos,
            "raw_track_max":        max(raw_vals),
            "raw_track_top2_mean":  topk_mean(raw_vals, 2),
            "raw_track_top3_mean":  topk_mean(raw_vals, 3),
            "raw_track_mean":       float(np.mean(raw_vals)),
            "P1_track_max":         max(p1_vals),
            "P1_track_top2_mean":   topk_mean(p1_vals, 2),
            "P1_track_top3_mean":   topk_mean(p1_vals, 3),
            "P1_track_mean":        float(np.mean(p1_vals)),
            "P2_track_max":         max(p2_vals),
            "P2_track_top2_mean":   topk_mean(p2_vals, 2),
            "P2_track_top3_mean":   topk_mean(p2_vals, 3),
            "P2_track_mean":        float(np.mean(p2_vals)),
        })
    return pd.DataFrame(rows)


def hit_at_k(series_sorted, k):
    """상위 k개 안에 positive가 있으면 1."""
    return int((series_sorted.iloc[:k] == "positive").any())


def compute_candidate_topk(df, score_col, k_list):
    """candidate-level top-k 통계 (per score column)."""
    pos_patients = set(df.loc[df["label"] == "positive", "patient_id"].unique())
    n_pos_pts = len(pos_patients)

    results = {}
    all_pos_candidates = set(df.index[df["label"] == "positive"])

    for k in k_list:
        hit_count = 0
        pos_cov_num = 0  # top-k 안에 든 positive candidate 수
        rank_list = []   # positive candidate의 patient 내 rank

        for pid, grp in df.groupby("patient_id"):
            if pid not in pos_patients:
                continue
            grp_sorted = grp.sort_values(score_col, ascending=False).reset_index()
            top_k = grp_sorted.iloc[:k]
            if (top_k["label"] == "positive").any():
                hit_count += 1
            # positive coverage
            pos_cov_num += int((top_k["label"] == "positive").sum())
            # rank of positive candidates
            for rank_i, row in grp_sorted.iterrows():
                if row["label"] == "positive":
                    rank_list.append(rank_i + 1)  # 1-based

        total_pos_cands = int((df["label"] == "positive").sum())
        results[k] = {
            "patient_hit_rate": round(hit_count / n_pos_pts, 4) if n_pos_pts else 0,
            "hit_patient_count": hit_count,
            "total_positive_patient_count": n_pos_pts,
            "positive_candidate_coverage": round(pos_cov_num / total_pos_cands, 4) if total_pos_cands else 0,
            "mean_positive_rank": round(float(np.mean(rank_list)), 2) if rank_list else float("nan"),
            "median_positive_rank": round(float(np.median(rank_list)), 2) if rank_list else float("nan"),
            "p90_positive_rank": round(float(np.percentile(rank_list, 90)), 2) if rank_list else float("nan"),
        }
    return results, n_pos_pts


def compute_track_topk(df_tracks_all, score_col, k_list):
    """track-level top-k 통계."""
    pos_patients = set(df_tracks_all.loc[df_tracks_all["is_positive"] == 1, "patient_id"].unique())
    n_pos_pts = len(pos_patients)
    results = {}

    for k in k_list:
        hit_count = 0
        pos_track_cov_num = 0
        rank_list = []

        for pid, grp in df_tracks_all.groupby("patient_id"):
            if pid not in pos_patients:
                continue
            grp_sorted = grp.sort_values(score_col, ascending=False).reset_index(drop=True)
            top_k = grp_sorted.iloc[:k]
            if (top_k["is_positive"] == 1).any():
                hit_count += 1
            pos_track_cov_num += int((top_k["is_positive"] == 1).sum())
            for rank_i, row in grp_sorted.iterrows():
                if row["is_positive"] == 1:
                    rank_list.append(rank_i + 1)

        total_pos_tracks = int((df_tracks_all["is_positive"] == 1).sum())
        results[k] = {
            "patient_hit_rate": round(hit_count / n_pos_pts, 4) if n_pos_pts else 0,
            "hit_patient_count": hit_count,
            "total_positive_patient_count": n_pos_pts,
            "positive_track_coverage": round(pos_track_cov_num / total_pos_tracks, 4) if total_pos_tracks else 0,
            "mean_positive_track_rank": round(float(np.mean(rank_list)), 2) if rank_list else float("nan"),
            "median_positive_track_rank": round(float(np.median(rank_list)), 2) if rank_list else float("nan"),
            "p90_positive_track_rank": round(float(np.percentile(rank_list, 90)), 2) if rank_list else float("nan"),
        }
    return results, n_pos_pts


def compute_complete_miss(df, score_col, k, label="label"):
    """top-k에서 완전히 놓친 positive patient 목록."""
    pos_patients = set(df.loc[df[label] == "positive", "patient_id"].unique())
    missed = []
    for pid in pos_patients:
        grp = df[df["patient_id"] == pid].sort_values(score_col, ascending=False)
        if not (grp.iloc[:k][label] == "positive").any():
            missed.append(pid)
    return sorted(missed)


def score_distribution(series, name):
    s = series.dropna()
    inf_count = int((s.abs() == float("inf")).sum())
    nan_count = int(series.isna().sum())
    s = s[s.abs() != float("inf")]
    return {
        "score": name,
        "count": len(s),
        "mean": round(float(s.mean()), 6),
        "std": round(float(s.std()), 6),
        "min": round(float(s.min()), 6),
        "p1": round(float(s.quantile(0.01)), 6),
        "p5": round(float(s.quantile(0.05)), 6),
        "p10": round(float(s.quantile(0.10)), 6),
        "p25": round(float(s.quantile(0.25)), 6),
        "p50": round(float(s.quantile(0.50)), 6),
        "p75": round(float(s.quantile(0.75)), 6),
        "p90": round(float(s.quantile(0.90)), 6),
        "p95": round(float(s.quantile(0.95)), 6),
        "p99": round(float(s.quantile(0.99)), 6),
        "max": round(float(s.max()), 6),
        "nan_count": nan_count,
        "inf_count": inf_count,
    }


def main():
    args = parse_args()

    if not args.dry_run and not args.run_eval:
        print("bare run blocked — use --dry-run or --run-eval with confirm flags")
        sys.exit(2)

    if args.dry_run:
        print("=" * 64)
        print("Step 9 Stage1Dev Eval — DRY-RUN PLAN")
        print("=" * 64)
        print()
        print("[선행 조건 확인]")
        print(f"  DONE_STEP8_SCORING.json : {DONE_STEP8}")
        print(f"  scoring CSV             : {SCORE_CSV}")
        print(f"  FINAL_PLAN_LOCK.json    : {PLAN_LOCK}")
        print(f"  baseline CSV            : {BASELINE_CSV} ({'존재' if BASELINE_CSV.exists() else '없음'})")
        print()
        print("[평가 내용]")
        print("  candidate-level top-k  : raw / P1 / P2  |  k=1,3,5,10,20,50")
        print("  track-level top-k      : raw_track_top3_mean, P1_track_top3_mean, P2_track_top3_mean")
        print("  patient hit summary    : 152 patients")
        print("  problem patient audit  : LUNG1-086, LUNG1-386, LUNG1-399")
        print("  score distribution     : raw/P1/P2/layer1/2/3")
        print("  baseline comparison    : RD-D1s medi3ch  (if available)")
        print()
        print("[금지]")
        print("  training / model forward / checkpoint / stage2 / threshold tuning")
        print()
        print("[생성 파일]")
        for f in [
            "manifests/step9_candidate_level_topk.csv",
            "manifests/step9_track_level_scores.csv",
            "manifests/step9_track_level_topk.csv",
            "manifests/step9_patient_hit_summary.csv",
            "manifests/step9_problem_patient_audit.csv",
            "manifests/step9_score_distribution_summary.csv",
            "manifests/step9_baseline_comparison.csv",
            "reports/step9_stage1dev_eval_report.md",
            "reports/step9_stage1dev_eval_summary.json",
            "logs/step9_stage1dev_eval_errors.csv",
            "DONE_STEP9_STAGE1DEV_EVAL.json",
        ]:
            print(f"  {ROOT / f}")
        print()
        print("[실행 명령]")
        print("  python experiments/rd4ad_2p5d_lung5ch_masked_normal_v1/scripts/"
              "rd4ad_2p5d_lung5ch_step9_stage1dev_eval.py \\")
        print("    --run-eval --confirm-plan-lock --confirm-no-stage2 --confirm-eval-only")
        print()
        print("DRY-RUN 완료.")
        return

    # ── 실행 모드 ──────────────────────────────────────────────────────────
    if not (args.confirm_plan_lock and args.confirm_no_stage2 and args.confirm_eval_only):
        print("BLOCKED: confirm flags missing")
        sys.exit(2)

    for d in [OUT_MANIFESTS, OUT_REPORTS, OUT_LOGS]:
        d.mkdir(parents=True, exist_ok=True)

    errors = []
    guardrail = {
        "plan_lock_loaded": False,
        "step8_scoring_passed": False,
        "evaluation_only": True,
        "training_executed": False,
        "model_forward_executed": False,
        "checkpoint_saved": False,
        "checkpoint_modified": False,
        "stage2_holdout_accessed": False,
        "positive_label_used_for_metric_only": True,
        "positive_label_used_for_training": False,
        "lesion_mask_used_for_training": False,
        "threshold_tuning_executed": False,
        "candidate_deletion_executed": False,
        "representative_only_scoring_used": False,
        "score_selection_changed": False,
        "convae_branch_created": False,
        "image_reconstruction_loss_used": False,
        "denominator_candidates": DENOMINATOR_CANDIDATES,
    }

    print("[1] 선행 조건 확인")

    # Plan lock
    if not PLAN_LOCK.exists():
        errors.append({"step": "plan_lock", "msg": f"FINAL_PLAN_LOCK.json not found: {PLAN_LOCK}"})
        print(f"  WARN: plan lock not found — {PLAN_LOCK}")
    else:
        guardrail["plan_lock_loaded"] = True
        print(f"  plan lock: OK")

    # DONE_STEP8
    if not DONE_STEP8.exists():
        print("BLOCKED: DONE_STEP8_SCORING.json not found")
        sys.exit(2)
    with open(DONE_STEP8) as f:
        done8 = json.load(f)
    if done8.get("verdict") != "PASS_STEP8_SCORING":
        print(f"BLOCKED: step8 verdict = {done8.get('verdict')}")
        sys.exit(2)
    if done8.get("scored_crops") != DENOMINATOR_CANDIDATES:
        print(f"BLOCKED: scored_crops={done8.get('scored_crops')} != {DENOMINATOR_CANDIDATES}")
        sys.exit(2)
    if done8.get("failed_crops", 1) != 0:
        print(f"BLOCKED: failed_crops={done8.get('failed_crops')}")
        sys.exit(2)
    guardrail["step8_scoring_passed"] = True
    print(f"  DONE_STEP8: PASS (scored={done8['scored_crops']}, failed=0)")

    # Score CSV
    if not SCORE_CSV.exists():
        print(f"BLOCKED: score CSV not found: {SCORE_CSV}")
        sys.exit(2)

    print()
    print("[2] Step 8 결과 로드 및 검증")
    df = pd.read_csv(str(SCORE_CSV))
    row_count = len(df)
    nan_count = int(df["rd4ad_lung5ch_score_raw"].isna().sum())
    inf_count = int((df["rd4ad_lung5ch_score_raw"].abs() == float("inf")).sum())

    print(f"  rows     : {row_count}")
    print(f"  NaN      : {nan_count}")
    print(f"  Inf      : {inf_count}")

    if row_count != DENOMINATOR_CANDIDATES:
        print(f"BLOCKED: row count {row_count} != {DENOMINATOR_CANDIDATES}")
        sys.exit(2)
    if nan_count > 0 or inf_count > 0:
        print(f"BLOCKED: NaN={nan_count} Inf={inf_count} in score")
        sys.exit(2)

    # label
    label_counts = df["label"].value_counts().to_dict()
    pos_patients_all = sorted(df.loc[df["label"] == "positive", "patient_id"].unique())
    n_pos_pts = len(pos_patients_all)
    n_total_pts = df["patient_id"].nunique()
    n_pos_cands = int((df["label"] == "positive").sum())
    n_total_tracks = df["track_id"].nunique()
    print(f"  label dist: {label_counts}")
    print(f"  patients   : {n_total_pts} total, {n_pos_pts} positive")
    print(f"  tracks     : {n_total_tracks}")

    # denominator note
    denominator_note = (
        "Step 8에서 실제 scoring manifest 기준으로 p90 + same-position z-continuity >= 2 "
        "후보를 재계산한 결과 95,995개로 확정했다. "
        "이후 stage1_dev ranking/evaluation의 공식 denominator는 95,995 candidates이다."
    )

    print()
    print("[3] candidate-level top-k 계산")
    score_cols = {
        "raw": "rd4ad_lung5ch_score_raw",
        "P1": "P1_times_roi",
        "P2": "P2_times_sqrt_roi",
    }

    cand_topk_rows = []
    cand_topk_by_score = {}
    for sname, scol in score_cols.items():
        if scol not in df.columns:
            print(f"  SKIPPED: {scol} not in CSV")
            continue
        stats, n_pos = compute_candidate_topk(df, scol, TOPK_LIST)
        cand_topk_by_score[sname] = stats
        for k, v in stats.items():
            row = {"score_type": sname, "score_col": scol, "k": k}
            row.update(v)
            cand_topk_rows.append(row)
        print(f"  {sname}: top1={stats[1]['patient_hit_rate']:.4f}  "
              f"top5={stats[5]['patient_hit_rate']:.4f}  "
              f"top20={stats[20]['patient_hit_rate']:.4f}")

    pd.DataFrame(cand_topk_rows).to_csv(OUT_MANIFESTS / "step9_candidate_level_topk.csv", index=False)

    print()
    print("[4] track-level score 집계")
    track_rows_list = []
    for pid, grp in df.groupby("patient_id"):
        tdf = compute_track_scores(grp)
        track_rows_list.append(tdf)
    df_tracks = pd.concat(track_rows_list, ignore_index=True)
    print(f"  total tracks: {len(df_tracks)}")
    print(f"  positive tracks: {int(df_tracks['is_positive'].sum())}")
    df_tracks.to_csv(OUT_MANIFESTS / "step9_track_level_scores.csv", index=False)

    track_score_cols = {
        "raw_top3": "raw_track_top3_mean",
        "raw_max": "raw_track_max",
        "P1_top3": "P1_track_top3_mean",
        "P1_max": "P1_track_max",
        "P2_top3": "P2_track_top3_mean",
        "P2_max": "P2_track_max",
    }

    print()
    print("[5] track-level top-k 계산")
    track_topk_rows = []
    track_topk_by_score = {}
    for tsname, tscol in track_score_cols.items():
        if tscol not in df_tracks.columns:
            print(f"  SKIPPED: {tscol}")
            continue
        stats, _ = compute_track_topk(df_tracks, tscol, TOPK_LIST)
        track_topk_by_score[tsname] = stats
        for k, v in stats.items():
            row = {"score_type": tsname, "score_col": tscol, "k": k}
            row.update(v)
            track_topk_rows.append(row)
        print(f"  {tsname}: top1={stats[1]['patient_hit_rate']:.4f}  "
              f"top5={stats[5]['patient_hit_rate']:.4f}  "
              f"top20={stats[20]['patient_hit_rate']:.4f}")

    pd.DataFrame(track_topk_rows).to_csv(OUT_MANIFESTS / "step9_track_level_topk.csv", index=False)

    print()
    print("[6] patient hit summary")
    hit_rows = []
    for pid, grp in df.groupby("patient_id"):
        is_pos_pt = pid in pos_patients_all
        pt_tracks = df_tracks[df_tracks["patient_id"] == pid]
        row = {"patient_id": pid, "is_positive_patient": int(is_pos_pt),
               "n_candidates": len(grp), "n_tracks": len(pt_tracks),
               "n_positive_candidates": int((grp["label"] == "positive").sum()),
               "n_positive_tracks": int(pt_tracks["is_positive"].sum()) if len(pt_tracks) else 0}

        for sname, scol in score_cols.items():
            if scol not in grp.columns:
                continue
            grp_sorted = grp.sort_values(scol, ascending=False).reset_index(drop=True)
            for k in [1, 5, 10, 20, 50]:
                row[f"hit@{k}_{sname}"] = hit_at_k(grp_sorted["label"], k) if is_pos_pt else -1

        for tsname, tscol in track_score_cols.items():
            if tscol not in pt_tracks.columns or len(pt_tracks) == 0:
                continue
            pt_tr_sorted = pt_tracks.sort_values(tscol, ascending=False).reset_index(drop=True)
            for k in [1, 5, 10, 20]:
                row[f"track_hit@{k}_{tsname}"] = hit_at_k(
                    pt_tr_sorted["is_positive"].map({1: "positive", 0: "hard_negative"}), k
                ) if is_pos_pt else -1

        hit_rows.append(row)

    df_hit = pd.DataFrame(hit_rows)
    df_hit.to_csv(OUT_MANIFESTS / "step9_patient_hit_summary.csv", index=False)

    # complete miss (top20, raw)
    miss_top20_raw = compute_complete_miss(df, "rd4ad_lung5ch_score_raw", 20)
    miss_top20_P1  = compute_complete_miss(df, "P1_times_roi", 20)
    print(f"  complete miss @top20 raw: {len(miss_top20_raw)}  {miss_top20_raw[:5]}")
    print(f"  complete miss @top20 P1 : {len(miss_top20_P1)}  {miss_top20_P1[:5]}")

    print()
    print("[7] problem patient audit")
    audit_rows = []
    for pid in PROBLEM_PATIENTS:
        grp = df[df["patient_id"] == pid]
        pt_tr = df_tracks[df_tracks["patient_id"] == pid]
        if len(grp) == 0:
            audit_rows.append({"patient_id": pid, "note": "not in scoring CSV"})
            print(f"  {pid}: not in CSV")
            continue

        n_pos_cands_pt = int((grp["label"] == "positive").sum())
        n_pos_tracks_pt = int(pt_tr["is_positive"].sum()) if len(pt_tr) else 0

        audit = {
            "patient_id": pid,
            "total_candidates": len(grp),
            "total_tracks": len(pt_tr),
            "positive_candidates": n_pos_cands_pt,
            "positive_tracks": n_pos_tracks_pt,
            "nearest_repeat_ratio": round(float(grp["nearest_repeat_used"].mean()), 4),
            "lung_z_pct_mean": round(float(grp["lung_z_percentile"].mean()), 4),
            "lung_z_pct_p10": round(float(grp["lung_z_percentile"].quantile(0.10)), 4),
            "lung_z_pct_p90": round(float(grp["lung_z_percentile"].quantile(0.90)), 4),
        }

        for sname, scol in score_cols.items():
            if scol not in grp.columns:
                continue
            grp_sorted = grp.sort_values(scol, ascending=False).reset_index(drop=True)
            # best positive rank
            pos_ranks = [i+1 for i, r in grp_sorted.iterrows() if r["label"] == "positive"]
            audit[f"best_positive_rank_{sname}"] = pos_ranks[0] if pos_ranks else -1
            for k in [10, 20, 50]:
                audit[f"top{k}_hit_{sname}"] = (
                    int((grp_sorted.iloc[:k]["label"] == "positive").any())
                    if n_pos_cands_pt > 0 else 0
                )

        for tsname, tscol in [("raw_top3", "raw_track_top3_mean"),
                               ("P1_top3", "P1_track_top3_mean")]:
            if tscol in pt_tr.columns and len(pt_tr):
                pt_tr_sorted = pt_tr.sort_values(tscol, ascending=False).reset_index(drop=True)
                pos_tr_ranks = [i+1 for i, r in pt_tr_sorted.iterrows() if r["is_positive"] == 1]
                audit[f"best_pos_track_rank_{tsname}"] = pos_tr_ranks[0] if pos_tr_ranks else -1

        if n_pos_cands_pt == 0:
            audit["note"] = "no positive candidates — hard_negative patient only"
        elif pid == "LUNG1-399":
            audit["note"] = f"positive candidates exist ({n_pos_cands_pt}); check rank"
        else:
            audit["note"] = "check"

        audit_rows.append(audit)
        print(f"  {pid}: cands={len(grp)}, pos={n_pos_cands_pt}, tracks={len(pt_tr)}")

    pd.DataFrame(audit_rows).to_csv(OUT_MANIFESTS / "step9_problem_patient_audit.csv", index=False)

    print()
    print("[8] score distribution")
    dist_rows = []
    for sname, scol in {
        "raw": "rd4ad_lung5ch_score_raw",
        "P1": "P1_times_roi",
        "P2": "P2_times_sqrt_roi",
        "layer1": "score_layer1",
        "layer2": "score_layer2",
        "layer3": "score_layer3",
    }.items():
        if scol in df.columns:
            dist_rows.append(score_distribution(df[scol], sname))
    pd.DataFrame(dist_rows).to_csv(OUT_MANIFESTS / "step9_score_distribution_summary.csv", index=False)
    print("  완료")

    print()
    print("[9] baseline comparison")
    baseline_found = BASELINE_CSV.exists()
    baseline_note = "baseline file not found; comparison deferred"
    cmp_rows = []

    if baseline_found:
        df_bl = pd.read_csv(str(BASELINE_CSV))
        # 새 모델 scoring 대상과 같은 (patient_id + local_z + crop_y0 + crop_x0) 기준으로 join
        merge_key = ["patient_id", "local_z", "crop_y0", "crop_x0"]
        df_bl_filt = df_bl.merge(
            df[merge_key].drop_duplicates(), on=merge_key, how="inner"
        )
        print(f"  baseline filtered to matching candidates: {len(df_bl_filt)}")

        # 새 모델 raw top-k
        new_cand_topk, _ = compute_candidate_topk(df, "rd4ad_lung5ch_score_raw", TOPK_LIST)
        new_P1_topk, _   = compute_candidate_topk(df, "P1_times_roi", TOPK_LIST)
        new_tr_topk, _   = compute_track_topk(df_tracks, "P1_track_top3_mean", TOPK_LIST)

        # baseline raw top-k (on matched subset)
        bl_cand_topk, _ = compute_candidate_topk(df_bl_filt, "rd_d1s_medi3ch_rd4ad_score", TOPK_LIST)

        for k in TOPK_LIST:
            cmp_rows.append({
                "k": k,
                "new_lung5ch_raw_hit_rate": new_cand_topk[k]["patient_hit_rate"],
                "new_lung5ch_P1_hit_rate": new_P1_topk[k]["patient_hit_rate"],
                "new_lung5ch_P1_track_top3_hit_rate": new_tr_topk[k]["patient_hit_rate"],
                "baseline_rd_d1s_raw_hit_rate": bl_cand_topk[k]["patient_hit_rate"],
                "baseline_candidate_count": len(df_bl_filt),
                "new_candidate_count": len(df),
                "note": "baseline filtered to same candidate set; no P1/track data for baseline"
            })
        baseline_note = f"RD-D1s medi3ch baseline: {len(df_bl_filt)} matched candidates"
        print(f"  {baseline_note}")
        for k in [1, 5, 10, 20, 50]:
            r = next(x for x in cmp_rows if x["k"] == k)
            print(f"  @{k:2d}: new_raw={r['new_lung5ch_raw_hit_rate']:.4f}  "
                  f"new_P1={r['new_lung5ch_P1_hit_rate']:.4f}  "
                  f"bl_raw={r['baseline_rd_d1s_raw_hit_rate']:.4f}")
    else:
        # new model only
        new_cand_topk, _ = compute_candidate_topk(df, "rd4ad_lung5ch_score_raw", TOPK_LIST)
        new_P1_topk, _   = compute_candidate_topk(df, "P1_times_roi", TOPK_LIST)
        new_tr_topk, _   = compute_track_topk(df_tracks, "P1_track_top3_mean", TOPK_LIST)
        for k in TOPK_LIST:
            cmp_rows.append({
                "k": k,
                "new_lung5ch_raw_hit_rate": new_cand_topk[k]["patient_hit_rate"],
                "new_lung5ch_P1_hit_rate": new_P1_topk[k]["patient_hit_rate"],
                "new_lung5ch_P1_track_top3_hit_rate": new_tr_topk[k]["patient_hit_rate"],
                "baseline_rd_d1s_raw_hit_rate": "N/A",
                "note": baseline_note,
            })
        print(f"  {baseline_note}")

    pd.DataFrame(cmp_rows).to_csv(OUT_MANIFESTS / "step9_baseline_comparison.csv", index=False)

    # ── report ────────────────────────────────────────────────────────────
    print()
    print("[10] report 생성")

    # best score family 결정 (P1_track_top3 @top20)
    if "P1_top3" in track_topk_by_score:
        best_top20 = track_topk_by_score["P1_top3"][20]["patient_hit_rate"]
        best_family = "P1_track_top3_mean"
    else:
        best_top20 = cand_topk_by_score.get("P1", {}).get(20, {}).get("patient_hit_rate", 0)
        best_family = "P1_times_roi (candidate)"

    report_lines = []
    report_lines.append("# Step 9 Stage1Dev Eval Report")
    report_lines.append("")
    report_lines.append("## Denominator Note")
    report_lines.append("")
    report_lines.append(denominator_note)
    report_lines.append("")
    report_lines.append("## Summary")
    report_lines.append("")
    report_lines.append(f"| 항목 | 값 |")
    report_lines.append(f"|---|---|")
    report_lines.append(f"| denominator candidates | {DENOMINATOR_CANDIDATES:,} |")
    report_lines.append(f"| total tracks | {len(df_tracks):,} |")
    report_lines.append(f"| total patients | {n_total_pts} |")
    report_lines.append(f"| positive patients | {n_pos_pts} |")
    report_lines.append(f"| positive candidates | {n_pos_cands:,} |")
    report_lines.append(f"| hard_negative candidates | {DENOMINATOR_CANDIDATES - n_pos_cands:,} |")
    report_lines.append(f"| NaN score | 0 |")
    report_lines.append(f"| Inf score | 0 |")
    report_lines.append(f"| stage2 accessed | False |")
    report_lines.append("")
    report_lines.append("## Candidate-Level Top-K (patient hit rate)")
    report_lines.append("")
    report_lines.append("| k | raw | P1 | P2 |")
    report_lines.append("|---|---|---|---|")
    for k in TOPK_LIST:
        r = cand_topk_by_score.get("raw", {}).get(k, {})
        p1 = cand_topk_by_score.get("P1", {}).get(k, {})
        p2 = cand_topk_by_score.get("P2", {}).get(k, {})
        report_lines.append(
            f"| {k} | {r.get('patient_hit_rate','N/A')} "
            f"| {p1.get('patient_hit_rate','N/A')} "
            f"| {p2.get('patient_hit_rate','N/A')} |"
        )
    report_lines.append("")
    report_lines.append("## Track-Level Top-K (patient hit rate)")
    report_lines.append("")
    report_lines.append("| k | raw_top3 | raw_max | P1_top3 | P1_max | P2_top3 | P2_max |")
    report_lines.append("|---|---|---|---|---|---|---|")
    for k in TOPK_LIST:
        cols = ["raw_top3", "raw_max", "P1_top3", "P1_max", "P2_top3", "P2_max"]
        vals = [track_topk_by_score.get(c, {}).get(k, {}).get("patient_hit_rate", "N/A") for c in cols]
        report_lines.append(f"| {k} | " + " | ".join(str(v) for v in vals) + " |")
    report_lines.append("")
    report_lines.append(
        "> **Caution**: Candidate-level top-k와 track-level top-k는 선택 단위가 다르므로 "
        "직접적인 수치 비교는 주의해야 한다. "
        "Track-level 결과는 후보 pool 압축 후 환자별 positive-containing track을 "
        "얼마나 상위에 배치하는지 보는 지표다."
    )
    report_lines.append("")
    report_lines.append("## Best Score Family")
    report_lines.append("")
    report_lines.append(f"best: `{best_family}` — @top20 patient_hit_rate = {best_top20:.4f}")
    report_lines.append("")
    report_lines.append("## Complete Miss @top20 (raw)")
    report_lines.append("")
    report_lines.append(f"count: {len(miss_top20_raw)}")
    report_lines.append(f"patients: {miss_top20_raw}")
    report_lines.append("")
    report_lines.append("## Problem Patient Audit")
    report_lines.append("")
    for a in audit_rows:
        report_lines.append(f"**{a['patient_id']}**: {a.get('note','')}")
        for kk, vv in a.items():
            if kk not in ("patient_id", "note"):
                report_lines.append(f"  - {kk}: {vv}")
        report_lines.append("")
    report_lines.append("## Baseline Comparison")
    report_lines.append("")
    report_lines.append(f"_{baseline_note}_")
    report_lines.append("")
    if cmp_rows and cmp_rows[0].get("baseline_rd_d1s_raw_hit_rate") != "N/A":
        report_lines.append("| k | new_raw | new_P1 | new_P1_track_top3 | baseline_raw |")
        report_lines.append("|---|---|---|---|---|")
        for r in cmp_rows:
            report_lines.append(
                f"| {r['k']} | {r['new_lung5ch_raw_hit_rate']} "
                f"| {r['new_lung5ch_P1_hit_rate']} "
                f"| {r['new_lung5ch_P1_track_top3_hit_rate']} "
                f"| {r['baseline_rd_d1s_raw_hit_rate']} |"
            )
    else:
        report_lines.append("baseline file not found; comparison deferred")
    report_lines.append("")
    report_lines.append("## Guardrail")
    report_lines.append("")
    for kk, vv in guardrail.items():
        report_lines.append(f"- {kk}: {vv}")

    report_path = OUT_REPORTS / "step9_stage1dev_eval_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  report: {report_path}")

    # summary json
    summary = {
        "step": "step9_stage1dev_eval",
        "verdict": "PASS_STEP9_STAGE1DEV_EVAL",
        "denominator_note": denominator_note,
        "denominator_candidates": DENOMINATOR_CANDIDATES,
        "total_tracks": len(df_tracks),
        "total_patients": n_total_pts,
        "positive_patients": n_pos_pts,
        "positive_candidates": n_pos_cands,
        "nan_count": 0,
        "inf_count": 0,
        "stage2_accessed": False,
        "best_score_family": best_family,
        "candidate_topk": {k: {s: cand_topk_by_score[s][k]["patient_hit_rate"]
                                for s in cand_topk_by_score}
                            for k in TOPK_LIST},
        "track_topk": {k: {s: track_topk_by_score[s][k]["patient_hit_rate"]
                            for s in track_topk_by_score}
                        for k in TOPK_LIST},
        "complete_miss_top20_raw": miss_top20_raw,
        "complete_miss_top20_P1": miss_top20_P1,
        "baseline_note": baseline_note,
        "guardrail": guardrail,
    }
    summary_path = OUT_REPORTS / "step9_stage1dev_eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # error CSV
    err_path = OUT_LOGS / "step9_stage1dev_eval_errors.csv"
    with open(err_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "msg"])
        writer.writeheader()
        for e in errors:
            writer.writerow(e)

    # DONE
    done_path = ROOT / "DONE_STEP9_STAGE1DEV_EVAL.json"
    with open(done_path, "w") as f:
        json.dump({
            "step": "step9_stage1dev_eval",
            "verdict": "PASS_STEP9_STAGE1DEV_EVAL",
            "created": "2026-06-10",
            "denominator_candidates": DENOMINATOR_CANDIDATES,
            "positive_patients": n_pos_pts,
            "total_patients": n_total_pts,
            "total_tracks": len(df_tracks),
            "best_score_family": best_family,
            "report": str(report_path),
            "summary_json": str(summary_path),
        }, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 64)
    print("판정: PASS_STEP9_STAGE1DEV_EVAL")
    print("=" * 64)
    print(f"  denominator  : {DENOMINATOR_CANDIDATES:,} candidates")
    print(f"  tracks       : {len(df_tracks):,}")
    print(f"  patients     : {n_total_pts} total / {n_pos_pts} positive")
    print(f"  best family  : {best_family}  @top20={best_top20:.4f}")
    print(f"  stage2 accessed : False")
    print()
    print("  candidate-level top-k (patient hit rate):")
    header = "  k   |" + "".join(f" {s:>8} |" for s in score_cols)
    print(header)
    for k in TOPK_LIST:
        vals = "".join(f" {cand_topk_by_score[s][k]['patient_hit_rate']:8.4f} |"
                       for s in score_cols if s in cand_topk_by_score)
        print(f"  {k:3d} |{vals}")
    print()
    print("  track-level top-k P1_top3 (patient hit rate):")
    for k in TOPK_LIST:
        v = track_topk_by_score.get("P1_top3", {}).get(k, {}).get("patient_hit_rate", "N/A")
        print(f"    @top{k:2d}: {v}")
    print()
    print("다음 단계: Step 10 decision checkpoint (lung5ch vs RD-D1s)")


if __name__ == "__main__":
    main()
