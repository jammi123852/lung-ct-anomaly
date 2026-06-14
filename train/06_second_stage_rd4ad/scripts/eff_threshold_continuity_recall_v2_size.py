"""
threshold별 남은 패치 수 + 병변 크기별 손실률
lesion_pixels 기준 small/medium/large/xlarge 분류
"""
import os, glob, json
import pandas as pd
import numpy as np
from pathlib import Path

REPO = Path(__file__).parent.parent
NORMAL_VAL_DIR = REPO / "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1/outputs/scores/normal_val_by_patient"
LESION_DEV_DIR = REPO / "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1/outputs/scores/lesion_stage1_dev_by_patient"
OUT_DIR = REPO / "outputs/position-aware-padim-v1/reports/eff_threshold_continuity_recall_v1"

PERCENTILES = [80, 90, 95]

# ── 1. threshold 계산
nv_files = glob.glob(str(NORMAL_VAL_DIR / "*.csv"))
nv_scores = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in nv_files], ignore_index=True)["padim_score"].dropna().values
thresholds = {p: float(np.percentile(nv_scores, p)) for p in PERCENTILES}

# ── 2. lesion_stage1_dev 로드
lv_files = glob.glob(str(LESION_DEV_DIR / "*.csv"))
lv_all = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in lv_files], ignore_index=True)
lv_all["is_lesion"] = ((lv_all["has_lesion_patch"] > 0) | (lv_all["lesion_pixels"] > 0)).astype(int)

# 병변 패치만 추출 + 크기 분류
lesion_df = lv_all[lv_all["is_lesion"] == 1].copy()
total_lesion = len(lesion_df)
bins   = [0, 50, 200, 500, np.inf]
labels = ["small(<50px)", "medium(50-200px)", "large(200-500px)", "xlarge(>500px)"]
lesion_df["size_bin"] = pd.cut(lesion_df["lesion_pixels"], bins=bins, labels=labels, right=False)
size_counts = lesion_df["size_bin"].value_counts().sort_index()
print("\n[병변 패치 크기 분포]")
for k, v in size_counts.items():
    print(f"  {k}: {v:,}개 ({v/total_lesion*100:.1f}%)")

# ── 3. threshold별 z-continuity 필터 후 크기별 손실 계산
def get_track_lesion_set(df, thr):
    above = df[df["padim_score"] > thr].copy()
    above = above.sort_values(["patient_id", "y0", "x0", "local_z"])
    track_idx = []
    for (pid, y0, x0), grp in above.groupby(["patient_id", "y0", "x0"], sort=False):
        zvals = grp["local_z"].sort_values().values
        idxvals = grp.sort_values("local_z").index.values
        if len(zvals) < 2:
            continue
        run_start = 0
        for i in range(1, len(zvals) + 1):
            if i == len(zvals) or zvals[i] - zvals[i-1] != 1:
                if i - run_start >= 2:
                    track_idx.extend(idxvals[run_start:i])
                run_start = i
    return set(track_idx)

print("\n[threshold별 패치 수 & 크기별 병변 손실]")
print(f"{'':30s} {'p80':>12} {'p90':>12} {'p95':>12}")
print(f"{'threshold':30s} {thresholds[80]:>12.4f} {thresholds[90]:>12.4f} {thresholds[95]:>12.4f}")

track_sets = {}
for p in PERCENTILES:
    track_sets[p] = get_track_lesion_set(lv_all, thresholds[p])

# 전체 패치 수
for label, key in [("전체 남은 패치(track)", "track"), ("  중 병변 패치", "lesion")]:
    row = {}
    for p in PERCENTILES:
        tidx = track_sets[p]
        if key == "track":
            row[p] = len(tidx)
        else:
            row[p] = len(set(lesion_df.index) & tidx)
    print(f"{label:30s} {row[80]:>12,} {row[90]:>12,} {row[95]:>12,}")

print(f"\n{'병변 recall(track)':30s} {0.7887:>12.4f} {0.6568:>12.4f} {0.5354:>12.4f}")

# 크기별 손실률 = (전체 크기별 병변 - track에 남은 크기별 병변) / 전체 크기별 병변
print(f"\n[크기별 병변 손실률 (잃은 비율)]")
print(f"{'size_bin':25s} {'총 병변':>8} {'p80 손실%':>10} {'p90 손실%':>10} {'p95 손실%':>10}")
for sbin in labels:
    sub = lesion_df[lesion_df["size_bin"] == sbin]
    total_s = len(sub)
    if total_s == 0:
        continue
    row_vals = []
    for p in PERCENTILES:
        survived = len(set(sub.index) & track_sets[p])
        lost_pct = (total_s - survived) / total_s * 100
        row_vals.append(f"{lost_pct:>10.1f}%")
    print(f"{sbin:25s} {total_s:>8,} {'  '.join(row_vals)}")

# ── 4. 저장
size_rows = []
for sbin in labels:
    sub = lesion_df[lesion_df["size_bin"] == sbin]
    total_s = len(sub)
    row = {"size_bin": sbin, "total_lesion_patches": total_s}
    for p in PERCENTILES:
        survived = len(set(sub.index) & track_sets[p])
        row[f"p{p}_survived"] = survived
        row[f"p{p}_lost_pct"] = round((total_s - survived) / total_s * 100, 2) if total_s > 0 else 0
    size_rows.append(row)
pd.DataFrame(size_rows).to_csv(OUT_DIR / "size_bin_lesion_loss.csv", index=False)
print(f"\n저장: {OUT_DIR}/size_bin_lesion_loss.csv")
