"""
EfficientNet v4_20: threshold(p80/p90/p95) 초과 패치 → z-continuity 필터 → 병변 recall
- 입력: normal_val scores(threshold 계산), lesion_stage1_dev scores(recall 계산)
- 출력: outputs/position-aware-padim-v1/reports/eff_threshold_continuity_recall_v1/
"""
import os, glob, json
import pandas as pd
import numpy as np
from pathlib import Path

REPO = Path(__file__).parent.parent
NORMAL_VAL_DIR  = REPO / "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1/outputs/scores/normal_val_by_patient"
LESION_DEV_DIR  = REPO / "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1/outputs/scores/lesion_stage1_dev_by_patient"
OUT_DIR = REPO / "outputs/position-aware-padim-v1/reports/eff_threshold_continuity_recall_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PERCENTILES = [80, 90, 95]

# ── 1. normal_val 전체 점수 로드 → threshold 계산
print("[1] normal_val scores 로드 중...")
nv_files = glob.glob(str(NORMAL_VAL_DIR / "*.csv"))
nv_dfs = [pd.read_csv(f, encoding="utf-8-sig") for f in nv_files]
nv_all = pd.concat(nv_dfs, ignore_index=True)
nv_scores = nv_all["padim_score"].dropna().values
thresholds = {p: float(np.percentile(nv_scores, p)) for p in PERCENTILES}
print(f"  normal_val patches: {len(nv_scores):,}")
for p, v in thresholds.items():
    print(f"  p{p} threshold: {v:.6f}")

# ── 2. lesion_stage1_dev 전체 패치 로드
print("[2] lesion_stage1_dev scores 로드 중...")
lv_files = glob.glob(str(LESION_DEV_DIR / "*.csv"))
lv_dfs = [pd.read_csv(f, encoding="utf-8-sig") for f in lv_files]
lv_all = pd.concat(lv_dfs, ignore_index=True)
lv_all["is_lesion"] = ((lv_all["has_lesion_patch"] > 0) | (lv_all["lesion_pixels"] > 0)).astype(int)
total_lesion_patches = int(lv_all["is_lesion"].sum())
print(f"  lesion_stage1_dev patches: {len(lv_all):,}  / lesion patches: {total_lesion_patches:,}")

# ── 3. threshold별 → z-continuity 필터 → recall
def calc_continuity_recall(df, thr, pct):
    # threshold 초과 패치만
    above = df[df["padim_score"] > thr].copy()
    if above.empty:
        return {"percentile": pct, "threshold": thr,
                "above_patches": 0, "above_lesion_patches": 0,
                "track_patches": 0, "track_lesion_patches": 0,
                "lesion_recall_above": 0.0, "lesion_recall_track": 0.0,
                "patient_hit_rate_track": 0.0}

    above = above.sort_values(["patient_id", "y0", "x0", "local_z"])

    # z-continuity track: same (patient_id, y0, x0)에서 local_z diff==1 run, length≥2
    track_rows = []
    for (pid, y0, x0), grp in above.groupby(["patient_id", "y0", "x0"], sort=False):
        zvals = grp["local_z"].sort_values().values
        if len(zvals) < 2:
            continue
        # 연속 run 탐지
        run_start = 0
        for i in range(1, len(zvals) + 1):
            if i == len(zvals) or zvals[i] - zvals[i-1] != 1:
                run_len = i - run_start
                if run_len >= 2:
                    run_zs = set(zvals[run_start:i])
                    mask = grp["local_z"].isin(run_zs)
                    track_rows.append(grp[mask])
                run_start = i

    above_lesion = int(above["is_lesion"].sum())
    lesion_recall_above = above_lesion / total_lesion_patches if total_lesion_patches > 0 else 0.0

    if not track_rows:
        return {"percentile": pct, "threshold": round(thr, 6),
                "above_patches": len(above), "above_lesion_patches": above_lesion,
                "track_patches": 0, "track_lesion_patches": 0,
                "lesion_recall_above": round(lesion_recall_above, 4),
                "lesion_recall_track": 0.0,
                "patient_hit_rate_track": 0.0}

    track_df = pd.concat(track_rows, ignore_index=True)
    track_lesion = int(track_df["is_lesion"].sum())
    lesion_recall_track = track_lesion / total_lesion_patches if total_lesion_patches > 0 else 0.0

    # patient hit rate: 병변 있는 환자 중 track에 병변 패치가 1개라도 있는 비율
    lesion_patients = set(lv_all[lv_all["is_lesion"] == 1]["patient_id"].unique())
    hit_patients = set(track_df[track_df["is_lesion"] == 1]["patient_id"].unique())
    patient_hit_rate = len(hit_patients) / len(lesion_patients) if lesion_patients else 0.0

    return {
        "percentile": pct,
        "threshold": round(thr, 6),
        "above_patches": len(above),
        "above_lesion_patches": above_lesion,
        "lesion_recall_above": round(lesion_recall_above, 4),
        "track_patches": len(track_df),
        "track_lesion_patches": track_lesion,
        "lesion_recall_track": round(lesion_recall_track, 4),
        "patient_hit_rate_track": round(patient_hit_rate, 4),
    }

print("[3] threshold별 계산 중...")
results = []
for p in PERCENTILES:
    thr = thresholds[p]
    r = calc_continuity_recall(lv_all, thr, p)
    results.append(r)
    print(f"  p{p}(>{thr:.4f}): above recall={r['lesion_recall_above']:.4f} "
          f"| track recall={r['lesion_recall_track']:.4f} "
          f"| patient hit={r['patient_hit_rate_track']:.4f}")

# ── 4. 저장
out_csv = OUT_DIR / "threshold_continuity_recall.csv"
out_json = OUT_DIR / "threshold_continuity_recall.json"
pd.DataFrame(results).to_csv(out_csv, index=False)
with open(out_json, "w") as f:
    json.dump({"total_lesion_patches": total_lesion_patches, "results": results}, f, indent=2)
print(f"[4] 저장 완료: {out_csv}")
