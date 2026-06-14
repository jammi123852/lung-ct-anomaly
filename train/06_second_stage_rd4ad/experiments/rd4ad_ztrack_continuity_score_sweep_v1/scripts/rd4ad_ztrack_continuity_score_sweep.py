"""
rd4ad_ztrack_continuity_score_sweep.py
목적: z-track 연결 길이(track_len)를 score에 반영했을 때 patient_hit_rate 변화 분석.

입력 (read-only):
  - rd4ad_strict_same_position_ztrack_actual_scoring_v1/manifests/strict_ztrack_scores_full_merged.csv
  - rd4ad_strict_same_position_ztrack_survival_preflight_v1/manifests/ztrack_manifest_minrun2.csv

출력:
  - reports/continuity_score_sweep_report.md
  - reports/continuity_score_sweep_summary.json
  - manifests/continuity_score_sweep_topk.csv

가드레일:
  - model forward / training / checkpoint / stage2_holdout / score 수정 금지
  - read-only CSV 분석만
"""

import json
import math
import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT     = Path(__file__).resolve().parents[1]

MERGED_CSV   = (PROJECT_ROOT
    / "experiments/rd4ad_strict_same_position_ztrack_actual_scoring_v1"
    / "manifests/strict_ztrack_scores_full_merged.csv")
MANIFEST_CSV = (PROJECT_ROOT
    / "experiments/rd4ad_strict_same_position_ztrack_survival_preflight_v1"
    / "manifests/ztrack_manifest_minrun2.csv")

REPORT_MD    = EXP_ROOT / "reports/continuity_score_sweep_report.md"
SUMMARY_JSON = EXP_ROOT / "reports/continuity_score_sweep_summary.json"
TOPK_CSV     = EXP_ROOT / "manifests/continuity_score_sweep_topk.csv"
DONE_JSON    = EXP_ROOT / "DONE.json"

GUARDRAILS = {
    "stage2_holdout_accessed": False,
    "model_forward_executed": False,
    "training_executed": False,
    "checkpoint_loaded": False,
    "scoring_reexecuted": False,
    "existing_artifact_modified": False,
    "existing_script_modified": False,
    "output_overwrite": False,
    "label_used_for_evaluation_only": True,
    "label_used_as_selector": False,
    "original_score_modified": False,
}

TOP_KS = [1, 3, 5, 10, 20, 50]

# 기존 baseline (sweep_analysis_v1 결과)
BASELINE = {
    "patch_baseline_rd_d1s":            {1:0.2697, 3:0.4079, 5:0.4803, 10:0.5855, 20:0.6184, 50:0.7105},
    "P1_track_max":                     {1:0.4474, 3:0.5855, 5:0.6382, 10:0.7237, 20:0.8158, 50:0.9079},
    "P1_track_top3_mean":               {1:0.4671, 3:0.5789, 5:0.6513, 10:0.7237, 20:0.8224, 50:0.9079},
}


def patient_hit_rate(df_tracks, score_col, k, n_positive_patients):
    """track-level DataFrame에서 top-k 기준 patient hit rate 계산."""
    topk = (df_tracks
            .sort_values(score_col, ascending=False)
            .groupby("patient_id")
            .head(k))
    hit = topk[topk["has_positive"] == True]["patient_id"].nunique()
    return round(hit / n_positive_patients, 4) if n_positive_patients > 0 else 0.0


def main():
    print("[1] 데이터 로드...")
    merged   = pd.read_csv(MERGED_CSV)
    manifest = pd.read_csv(MANIFEST_CSV)
    print(f"  merged: {len(merged):,}행  manifest: {len(manifest):,}행")

    # track_len join
    manifest_sub = manifest[["track_id", "track_len", "has_positive"]].copy()
    merged = merged.merge(manifest_sub, on="track_id", how="left")
    n_missing = merged["track_len"].isna().sum()
    if n_missing > 0:
        print(f"  [경고] track_len 미매칭: {n_missing}행 → 기본값 2로 채움")
        merged["track_len"] = merged["track_len"].fillna(2)
    merged["track_len"] = merged["track_len"].astype(int)
    print(f"  track_len: min={merged['track_len'].min()}, "
          f"median={merged['track_len'].median()}, max={merged['track_len'].max()}")

    # ── 새 score 계산 (patch-level)
    print("[2] 연결성 기반 score 계산...")
    p1 = merged["P1_times_roi"].values
    tl = merged["track_len"].values

    merged["P3_log_len"]   = p1 * np.log(tl)          # log(2)=0.69, log(10)=2.3
    merged["P4_sqrt_len"]  = p1 * np.sqrt(tl)          # sqrt(2)=1.41, sqrt(10)=3.16
    merged["P5_len_norm"]  = p1 * (tl / 3.0)           # median=3 기준 정규화

    # 진단
    for col in ["P3_log_len", "P4_sqrt_len", "P5_len_norm"]:
        print(f"  {col}: mean={merged[col].mean():.4f}, "
              f"max={merged[col].max():.4f}")

    # ── track-level aggregation
    print("[3] track-level aggregation...")
    score_cols = ["P1_times_roi", "P3_log_len", "P4_sqrt_len", "P5_len_norm"]

    def agg_tracks(df, score_col):
        """track별 max / top3_mean 집계."""
        grp = df.groupby("track_id")

        # has_positive는 manifest에서 왔으므로 track별로 일정
        meta = df.groupby("track_id").agg(
            patient_id=("patient_id", "first"),
            has_positive=("has_positive", "first"),
            track_len=("track_len", "first"),
        ).reset_index()

        track_max = grp[score_col].max().reset_index()
        track_max.columns = ["track_id", f"{score_col}_track_max"]

        def top3_mean(x):
            return x.nlargest(min(3, len(x))).mean()

        track_t3 = grp[score_col].apply(top3_mean).reset_index()
        track_t3.columns = ["track_id", f"{score_col}_track_top3_mean"]

        result = meta.merge(track_max, on="track_id").merge(track_t3, on="track_id")
        return result

    track_dfs = {}
    for sc in score_cols:
        track_dfs[sc] = agg_tracks(merged, sc)
        print(f"  {sc}: {len(track_dfs[sc]):,} tracks")

    # ── patient_hit_rate 계산
    print("[4] patient hit rate 계산...")
    positive_patients = set(merged[merged["label"] == "positive"]["patient_id"].unique())
    n_pos = len(positive_patients)
    print(f"  positive patients: {n_pos}")

    rows = []
    for sc in score_cols:
        tdf = track_dfs[sc]
        for agg in ("track_max", "track_top3_mean"):
            col = f"{sc}_{agg}"
            row = {"score_col": col}
            for k in TOP_KS:
                row[f"top{k}"] = patient_hit_rate(tdf, col, k, n_pos)
            rows.append(row)

    results_df = pd.DataFrame(rows)

    # ── 출력
    print("\n[5] 결과 출력")
    print(f"\n{'score_col':45s}", end="")
    for k in TOP_KS:
        print(f"  top{k:>2}", end="")
    print()
    print("-" * 110)

    # baseline 먼저
    for bname, bvals in BASELINE.items():
        print(f"  [baseline] {bname:35s}", end="")
        for k in TOP_KS:
            print(f"  {bvals[k]:.4f}", end="")
        print()
    print()

    # 새 결과
    for _, row in results_df.iterrows():
        col = row["score_col"]
        print(f"  {col:45s}", end="")
        for k in TOP_KS:
            print(f"  {row[f'top{k}']:.4f}", end="")
        print()

    # ── 최강 비교
    print("\n[6] 기존 P1_track_top3_mean 대비 개선 여부")
    ref = BASELINE["P1_track_top3_mean"]
    for _, row in results_df.iterrows():
        if "top3_mean" not in row["score_col"]:
            continue
        diffs = {k: round(row[f"top{k}"] - ref[k], 4) for k in TOP_KS}
        better = sum(1 for v in diffs.values() if v > 0)
        worse  = sum(1 for v in diffs.values() if v < -0.001)
        print(f"  {row['score_col']:45s}  better={better}/6  worse={worse}/6  "
              f"top1Δ={diffs[1]:+.4f}  top10Δ={diffs[10]:+.4f}  top50Δ={diffs[50]:+.4f}")

    # ── 저장
    print("\n[7] 저장...")
    results_df.to_csv(TOPK_CSV, index=False)
    print(f"  → {TOPK_CSV}")

    # summary json
    best_top10 = results_df.loc[results_df["top10"].idxmax(), "score_col"]
    best_top1  = results_df.loc[results_df["top1"].idxmax(), "score_col"]

    summary = {
        "verdict": "DONE",
        "n_candidates": len(merged),
        "n_tracks": len(track_dfs[score_cols[0]]),
        "n_positive_patients": n_pos,
        "track_len_stats": {
            "min": int(merged["track_len"].min()),
            "median": float(merged["track_len"].median()),
            "mean": round(float(merged["track_len"].mean()), 2),
            "max": int(merged["track_len"].max()),
        },
        "best_top1": best_top1,
        "best_top10": best_top10,
        "baseline_P1_track_top3_mean": BASELINE["P1_track_top3_mean"],
        "new_scores": {
            row["score_col"]: {f"top{k}": row[f"top{k}"] for k in TOP_KS}
            for _, row in results_df.iterrows()
        },
        "guardrails": GUARDRAILS,
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  → {SUMMARY_JSON}")

    # report md
    lines = ["# RD4AD Z-Track Continuity Score Sweep v1\n",
             "## 목적\n",
             "track_len(z-축 연결 길이)을 P1_times_roi에 곱해서 연결성이 높을수록 score를 올리는 방식 평가.\n\n",
             "## 새 Score 정의\n",
             "| score | 수식 |\n|-------|------|\n",
             "| P3_log_len | P1_times_roi × log(track_len) |\n",
             "| P4_sqrt_len | P1_times_roi × sqrt(track_len) |\n",
             "| P5_len_norm | P1_times_roi × (track_len / 3) |\n\n",
             "## track_len 분포\n",
             f"- min={summary['track_len_stats']['min']}, ",
             f"median={summary['track_len_stats']['median']}, ",
             f"mean={summary['track_len_stats']['mean']}, ",
             f"max={summary['track_len_stats']['max']}\n\n",
             "## Baseline 비교\n",
             "| score_col | top1 | top3 | top5 | top10 | top20 | top50 |\n",
             "|-----------|------|------|------|-------|-------|-------|\n"]

    for bname, bvals in BASELINE.items():
        lines.append(f"| {bname} | " + " | ".join(f"{bvals[k]:.4f}" for k in TOP_KS) + " |\n")
    for _, row in results_df.iterrows():
        lines.append(f"| {row['score_col']} | " +
                     " | ".join(f"{row[f'top{k}']:.4f}" for k in TOP_KS) + " |\n")

    lines += ["\n## 판정\n",
              f"- best_top1: `{best_top1}`\n",
              f"- best_top10: `{best_top10}`\n"]

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"  → {REPORT_MD}")

    # DONE
    done = {"verdict": "DONE", "guardrails": GUARDRAILS}
    with open(DONE_JSON, "w", encoding="utf-8") as f:
        json.dump(done, f, indent=2)
    print(f"  → {DONE_JSON}")
    print("\n[완료]")


if __name__ == "__main__":
    main()
