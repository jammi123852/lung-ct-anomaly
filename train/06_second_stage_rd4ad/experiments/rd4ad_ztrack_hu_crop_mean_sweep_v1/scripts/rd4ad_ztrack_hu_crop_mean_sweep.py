"""
rd4ad_ztrack_hu_crop_mean_sweep.py
목적: crop 영역의 v4_20 ROI 마스크 내부 정규화 HU mean을 계산하고,
      기존 P1_times_roi + track_len과 결합한 score sweep 평가.

입력 (read-only):
  - strict_ztrack_scores_full_merged.csv
  - ztrack_manifest_minrun2.csv
  - /mnt/c/.../volumes_npy/{safe_id}/ct_hu.npy  (메모리맵, crop 영역만)
  - outputs/.../refined_roi_v4_20_modeB_all_v1/{lesion|normal}/{safe_id}/refined_roi.npy

출력:
  - reports/hu_crop_mean_sweep_report.md
  - reports/hu_crop_mean_sweep_summary.json
  - manifests/hu_crop_mean_sweep_topk.csv
  - DONE.json

가드레일:
  - model forward / training / checkpoint / stage2_holdout / score 수정 금지
  - read-only CSV + npy 분석만
"""

import json
from pathlib import Path
from collections import OrderedDict

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

CT_ROOT  = Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1"
    "/volumes_npy"
)
MASK_ROOT = (PROJECT_ROOT
    / "outputs/mip-postprocess-research-v1/masks"
    / "refined_roi_v4_20_modeB_all_v1")

REPORT_MD    = EXP_ROOT / "reports/hu_crop_mean_sweep_report.md"
SUMMARY_JSON = EXP_ROOT / "reports/hu_crop_mean_sweep_summary.json"
TOPK_CSV     = EXP_ROOT / "manifests/hu_crop_mean_sweep_topk.csv"
DONE_JSON    = EXP_ROOT / "DONE.json"

HU_MIN, HU_MAX = -160.0, 240.0
TOP_KS = [1, 3, 5, 10, 20, 30, 50]

BASELINE = {
    "patch_baseline_rd_d1s":       {1:0.2697, 3:0.4079, 5:0.4803, 10:0.5855, 20:0.6184, 30:None, 50:0.7105},
    "P1_track_top3_mean":          {1:0.4671, 3:0.5789, 5:0.6513, 10:0.7237, 20:0.8224, 30:None, 50:0.9079},
    "P5_len_norm_track_top3_mean": {1:0.6579, 3:0.8355, 5:0.8553, 10:0.9145, 20:0.9474, 30:None, 50:0.9737},
    "P9_p1_sqrthu_len_top3_mean":  {1:0.6776, 3:0.7961, 5:0.8421, 10:0.8947, 20:0.9342, 30:None, 50:0.9605},
}


def find_mask_path(safe_id: str) -> Path | None:
    for sub in ("lesion", "normal"):
        p = MASK_ROOT / sub / safe_id / "refined_roi.npy"
        if p.exists():
            return p
    return None


class NpyCache:
    """환자별 npy를 최대 max_size 개 LRU 캐시."""
    def __init__(self, max_size=8):
        self._cache = OrderedDict()
        self._max   = max_size

    def get(self, path: Path):
        key = str(path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        arr = np.load(path, mmap_mode="r")
        if len(self._cache) >= self._max:
            self._cache.popitem(last=False)
        self._cache[key] = arr
        return arr


def compute_hu_mean(ct_arr, roi_arr, local_z, y0, x0, y1, x1):
    """crop 영역 내 v4_20 ROI 마스크 교집합 HU mean (정규화 [0,1])."""
    Z, H, W = ct_arr.shape
    # 경계 clamp
    y0c, y1c = max(y0, 0), min(y1, H)
    x0c, x1c = max(x0, 0), min(x1, W)
    z = max(0, min(local_z, Z - 1))
    if y1c <= y0c or x1c <= x0c:
        return np.nan

    hu_crop  = ct_arr[z, y0c:y1c, x0c:x1c].astype(np.float32)
    roi_crop = roi_arr[z, y0c:y1c, x0c:x1c].astype(bool)

    # ROI mask shape mismatch 방어
    if roi_crop.shape != hu_crop.shape:
        roi_crop = roi_crop[:hu_crop.shape[0], :hu_crop.shape[1]]

    inside = hu_crop[roi_crop]
    if len(inside) == 0:
        # ROI 내 픽셀 없으면 전체 crop 평균
        inside = hu_crop.ravel()
    if len(inside) == 0:
        return np.nan

    norm = (inside.clip(HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)
    return float(norm.mean())


def patient_hit_rate(df_tracks, score_col, k, n_positive_patients):
    topk = (df_tracks
            .sort_values(score_col, ascending=False)
            .groupby("patient_id")
            .head(k))
    hit = topk[topk["has_positive"] == True]["patient_id"].nunique()
    return round(hit / n_positive_patients, 4) if n_positive_patients > 0 else 0.0


def agg_tracks(df, score_col):
    grp  = df.groupby("track_id")
    meta = df.groupby("track_id").agg(
        patient_id=("patient_id", "first"),
        has_positive=("has_positive", "first"),
        track_len=("track_len", "first"),
    ).reset_index()
    t_max = grp[score_col].max().reset_index()
    t_max.columns = ["track_id", f"{score_col}_track_max"]
    def top3(x):
        return x.nlargest(min(3, len(x))).mean()
    t_t3 = grp[score_col].apply(top3).reset_index()
    t_t3.columns = ["track_id", f"{score_col}_track_top3_mean"]
    return meta.merge(t_max, on="track_id").merge(t_t3, on="track_id")


def main():
    print("[1] 데이터 로드...")
    merged   = pd.read_csv(MERGED_CSV)
    manifest = pd.read_csv(MANIFEST_CSV)
    print(f"  merged: {len(merged):,}행  manifest: {len(manifest):,}행")

    manifest_sub = manifest[["track_id", "track_len", "has_positive"]].copy()
    merged = merged.merge(manifest_sub, on="track_id", how="left")
    merged["track_len"] = merged["track_len"].fillna(2).astype(int)

    print("[2] HU mean 계산 (메모리맵 crop)...")
    ct_cache   = NpyCache(max_size=8)
    mask_cache = NpyCache(max_size=8)

    hu_means  = []
    n_missing = 0
    n_no_mask = 0

    for i, row in merged.iterrows():
        if i % 5000 == 0:
            print(f"  {i:,}/{len(merged):,}  missing={n_missing}  no_mask={n_no_mask}")

        safe_id = row["safe_id"]
        ct_path   = CT_ROOT / safe_id / "ct_hu.npy"
        mask_path = find_mask_path(safe_id)

        if not ct_path.exists():
            hu_means.append(np.nan)
            n_missing += 1
            continue
        if mask_path is None:
            n_no_mask += 1
            # ROI mask 없으면 전체 crop 평균으로 계산
            try:
                ct_arr = ct_cache.get(ct_path)
                # mask=None → 전체 crop
                Z, H, W = ct_arr.shape
                z  = int(row["local_z"])
                y0, x0, y1, x1 = int(row["crop_y0"]), int(row["crop_x0"]), int(row["crop_y1"]), int(row["crop_x1"])
                y0c, y1c = max(y0,0), min(y1,H)
                x0c, x1c = max(x0,0), min(x1,W)
                z = max(0, min(z, Z-1))
                patch = ct_arr[z, y0c:y1c, x0c:x1c].astype(np.float32).ravel()
                norm  = (patch.clip(HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)
                hu_means.append(float(norm.mean()))
            except Exception:
                hu_means.append(np.nan)
            continue

        try:
            ct_arr   = ct_cache.get(ct_path)
            roi_arr  = mask_cache.get(mask_path)
            val = compute_hu_mean(ct_arr, roi_arr,
                                  int(row["local_z"]),
                                  int(row["crop_y0"]), int(row["crop_x0"]),
                                  int(row["crop_y1"]), int(row["crop_x1"]))
            hu_means.append(val)
        except Exception as e:
            hu_means.append(np.nan)
            n_missing += 1

    merged["hu_norm_mean"] = hu_means
    nan_cnt = merged["hu_norm_mean"].isna().sum()
    print(f"  완료: NaN={nan_cnt}  no_mask={n_no_mask}  ct_missing={n_missing}")
    print(f"  hu_norm_mean: mean={merged['hu_norm_mean'].mean():.4f}  "
          f"min={merged['hu_norm_mean'].min():.4f}  max={merged['hu_norm_mean'].max():.4f}")

    # NaN → 전체 median으로 채움
    hu_median = merged["hu_norm_mean"].median()
    merged["hu_norm_mean"] = merged["hu_norm_mean"].fillna(hu_median)

    print("[3] 상관관계 분석...")
    from scipy import stats
    track_hu = merged.groupby("track_id").agg(
        label_bin=("label", lambda x: 1 if (x == "positive").any() else 0),
        hu_norm_mean=("hu_norm_mean", "mean"),
    ).reset_index()
    r, p = stats.pointbiserialr(track_hu["label_bin"], track_hu["hu_norm_mean"])
    print(f"  hu_norm_mean vs label: r={r:.4f}, p={p:.4e}")

    # track_len vs hu_norm_mean
    track_info = merged.groupby("track_id").agg(
        track_len=("track_len", "first"),
        hu_norm_mean=("hu_norm_mean", "mean"),
    ).reset_index()
    r2, p2 = stats.pearsonr(track_info["track_len"], track_info["hu_norm_mean"])
    print(f"  track_len vs hu_norm_mean: r={r2:.4f}, p={p2:.4e}")

    print("[4] score 계산...")
    p1 = merged["P1_times_roi"].values
    hu = merged["hu_norm_mean"].values
    tl = merged["track_len"].values

    eps = 1e-4
    # P1 포함 버전
    merged["P6_p1_hu"]          = p1 * (hu + eps)
    merged["P7_p1_hu_len"]      = p1 * (hu + eps) * (tl / 3.0)
    merged["P8_p1_hu_loglen"]   = p1 * (hu + eps) * np.log(np.maximum(tl, 1))
    merged["P9_p1_sqrthu_len"]  = p1 * np.sqrt(np.maximum(hu, 0)) * (tl / 3.0)
    # P1 제거 버전 (hu + track_len 신호만)
    merged["PA_hu_only"]        = hu + eps
    merged["PB_hu_len"]         = (hu + eps) * (tl / 3.0)
    merged["PC_hu_loglen"]      = (hu + eps) * np.log(np.maximum(tl, 1))
    merged["PD_sqrthu_len"]     = np.sqrt(np.maximum(hu, 0)) * (tl / 3.0)

    all_score_cols = ["P6_p1_hu","P7_p1_hu_len","P8_p1_hu_loglen","P9_p1_sqrthu_len",
                      "PA_hu_only","PB_hu_len","PC_hu_loglen","PD_sqrthu_len"]
    for col in all_score_cols:
        print(f"  {col}: mean={merged[col].mean():.4f}  max={merged[col].max():.4f}")

    print("[5] track-level aggregation & patient_hit_rate...")
    score_cols = ["P6_p1_hu","P7_p1_hu_len","P8_p1_hu_loglen","P9_p1_sqrthu_len",
                  "PA_hu_only","PB_hu_len","PC_hu_loglen","PD_sqrthu_len"]

    positive_patients = set(merged[merged["label"] == "positive"]["patient_id"].unique())
    n_pos = len(positive_patients)
    print(f"  positive patients: {n_pos}")

    rows = []
    for sc in score_cols:
        tdf = agg_tracks(merged, sc)
        for agg in ("track_max", "track_top3_mean"):
            col = f"{sc}_{agg}"
            row = {"score_col": col}
            for k in TOP_KS:
                row[f"top{k}"] = patient_hit_rate(tdf, col, k, n_pos)
            rows.append(row)

    results_df = pd.DataFrame(rows)

    print("\n[6] 결과 출력")
    print(f"\n{'score_col':50s}", end="")
    for k in TOP_KS:
        print(f"  top{k:>2}", end="")
    print()
    print("-" * 120)

    for bname, bvals in BASELINE.items():
        print(f"  [base] {bname:45s}", end="")
        for k in TOP_KS:
            v = bvals[k]
            print(f"  {'--   ' if v is None else f'{v:.4f}'}", end="")
        print()
    print()

    for _, row in results_df.iterrows():
        col = row["score_col"]
        tag = "[P1제거]" if col.startswith("P") and col[1] in "ABCD" else "      "
        print(f"  {tag} {col:50s}", end="")
        for k in TOP_KS:
            print(f"  {row[f'top{k}']:.4f}", end="")
        print()

    print("\n[7] P1_track_top3_mean 대비 개선 (top3_mean만)")
    ref = BASELINE["P1_track_top3_mean"]
    ref30 = None  # top30 baseline 없음
    for _, row in results_df.iterrows():
        if "track_top3_mean" not in row["score_col"]:
            continue
        diffs = {}
        for k in TOP_KS:
            rb = ref[k]
            if rb is None:
                diffs[k] = None
            else:
                diffs[k] = round(row[f"top{k}"] - rb, 4)
        better = sum(1 for v in diffs.values() if v is not None and v > 0)
        d1  = diffs[1]  if diffs[1]  is not None else float('nan')
        d10 = diffs[10] if diffs[10] is not None else float('nan')
        d50 = diffs[50] if diffs[50] is not None else float('nan')
        print(f"  {row['score_col']:52s}  better={better}  "
              f"top1Δ={d1:+.4f}  top10Δ={d10:+.4f}  top50Δ={d50:+.4f}")

    print("\n[8] 저장...")
    results_df.to_csv(TOPK_CSV, index=False)
    print(f"  → {TOPK_CSV}")

    best_top1  = results_df.loc[results_df["top1"].idxmax(),  "score_col"]
    best_top10 = results_df.loc[results_df["top10"].idxmax(), "score_col"]

    summary = {
        "verdict": "DONE",
        "n_candidates": len(merged),
        "n_tracks": len(merged["track_id"].unique()),
        "n_positive_patients": n_pos,
        "hu_nan_filled": int(nan_cnt),
        "hu_no_mask": int(n_no_mask),
        "corr_hu_label": {"r": round(r, 4), "p": float(p)},
        "corr_hu_tracklen": {"r": round(r2, 4), "p": float(p2)},
        "best_top1": best_top1,
        "best_top10": best_top10,
        "baseline": BASELINE,
        "new_scores": {
            row["score_col"]: {f"top{k}": row[f"top{k}"] for k in TOP_KS}
            for _, row in results_df.iterrows()
        },
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  → {SUMMARY_JSON}")

    lines = [
        "# RD4AD Z-Track HU Crop Mean Score Sweep v1\n\n",
        "## 목적\n",
        "v4_20 ROI 마스크 내부 정규화 HU mean을 P1_times_roi + track_len과 결합한 score 평가.\n\n",
        "## HU 상관관계\n",
        f"- hu_norm_mean vs label(positive): r={r:.4f}, p={p:.4e}\n",
        f"- track_len vs hu_norm_mean: r={r2:.4f}, p={p2:.4e}\n\n",
        "## Score 정의\n",
        "| score | 수식 |\n|-------|------|\n",
        "| P6_p1_hu | P1 × (hu_norm_mean + ε) |\n",
        "| P7_p1_hu_len | P1 × (hu_norm_mean + ε) × (track_len/3) |\n",
        "| P8_p1_hu_loglen | P1 × (hu_norm_mean + ε) × log(track_len) |\n",
        "| P9_p1_sqrthu_len | P1 × sqrt(hu_norm_mean) × (track_len/3) |\n\n",
        "## 결과\n",
        "| score_col | top1 | top3 | top5 | top10 | top20 | top50 |\n",
        "|-----------|------|------|------|-------|-------|-------|\n",
    ]
    for bname, bvals in BASELINE.items():
        lines.append(f"| {bname} | " + " | ".join(f"{bvals[k]:.4f}" for k in TOP_KS) + " |\n")
    for _, row in results_df.iterrows():
        lines.append(f"| {row['score_col']} | " +
                     " | ".join(f"{row[f'top{k}']:.4f}" for k in TOP_KS) + " |\n")
    lines += [f"\n- best_top1: `{best_top1}`\n", f"- best_top10: `{best_top10}`\n"]
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"  → {REPORT_MD}")

    with open(DONE_JSON, "w", encoding="utf-8") as f:
        json.dump({"verdict": "DONE", "best_top1": best_top1, "best_top10": best_top10}, f, indent=2)
    print(f"  → {DONE_JSON}")
    print("\n[완료]")


if __name__ == "__main__":
    main()
