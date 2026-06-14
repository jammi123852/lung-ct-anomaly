#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
oracle_vessel_score_adjust_1case_v1
- 목적: oracle 혈관 라벨로 E2 stage1_dev 점수 양방향 보정이 의도대로 동작하는지 1명 검증
- 보정식: score_adj = score_raw + ALPHA*risky_ratio - BETA*normal_ratio
    risky_vessel_candidate  = GT lesion           (가점)
    normal_vessel_candidate = vessel & ~lesion     (감점)
- stage2 절대 미접근 (stage1_dev 점수만 사용)
- read-only 입력, 새 폴더 저장, 원본/score CSV 수정 없음 (보정은 새 CSV로만)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.filters import frangi
from skimage.morphology import disk, remove_small_objects, white_tophat

PROJECT = Path("/home/jinhy/project/lung-ct-anomaly")
NROOT   = Path("/mnt/c/Users/jinhy/Desktop/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy")
SCORE_CSV = PROJECT / "outputs/normal_based_stage2_verifier_audit/rd_e1e2_lung3ch_effb0_shard_run_v1/rd_d1s_stage1dev_candidate_score.csv"
SCORE_COL = "rd_d1s_medi3ch_rd4ad_score"
SAFE_ID   = "NSCLC_LUNG1-001__5d369af301"
OUT = PROJECT / "outputs/reports/oracle_vessel_score_adjust_1case_v1"
OUT.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05   # risky(병변) 가점 계수
BETA  = 0.05   # normal(혈관) 감점 계수

# ─ b1c4z59 동일 vessel 엔진 ─────────────────────────────────────────────────
SIGMAS, TOPHAT_RADIUS, PERCENTILE, MIN_AREA = (0.5, 1.0, 1.5, 2.0), 10, 85.0, 10

def lung_window(arr, level=-600, width=1500):
    lo, hi = level - width / 2, level + width / 2
    return np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)

def hu_normalize(arr):
    return np.clip((arr.astype(np.float32) + 1000.0) / 1600.0, 0.0, 1.0)

def compute_vessel_mask_s3(ct_vol, roi_vol, local_z):
    lz = int(local_z); nz = ct_vol.shape[0]
    z0, z1 = max(0, lz - 1), min(nz, lz + 2)
    mip   = np.max(ct_vol[z0:z1].astype(np.float32), axis=0)
    valid = np.any(roi_vol[z0:z1] > 0, axis=0)
    if not valid.any():
        return np.zeros(mip.shape, dtype=bool)
    lw = lung_window(mip);  m1 = (lw > np.percentile(lw[valid], PERCENTILE)) & valid
    hn = hu_normalize(mip)
    th = white_tophat(hn, disk(TOPHAT_RADIUS)); m2 = (th > np.percentile(th[valid], PERCENTILE)) & valid
    fr = frangi(hn, sigmas=SIGMAS, black_ridges=False); m3 = (fr > np.percentile(fr[valid], PERCENTILE)) & valid
    union = m1 | m2 | m3
    if MIN_AREA > 0 and union.any():
        union = remove_small_objects(union.copy(), min_size=MIN_AREA)
    return union

# ─ 데이터 로드 ──────────────────────────────────────────────────────────────
df = pd.read_csv(SCORE_CSV)
sub = df[df["safe_id"] == SAFE_ID].copy().reset_index(drop=True)
print(f"[INFO] {SAFE_ID}: {len(sub)} patches  label={sub['label'].value_counts().to_dict()}", flush=True)

pdir = NROOT / SAFE_ID
ct  = np.load(pdir / "ct_hu.npy")
roi = np.load(pdir / "roi_0_0.npy")
les = np.load(pdir / "lesion_mask_roi_0_0.npy")
print(f"[INFO] ct={ct.shape} lesion_vox={int(les.sum())}", flush=True)

# ─ z별 vessel mask 캐시 ─────────────────────────────────────────────────────
risky_ratios, normal_ratios = [], []
cache = {}
uniq_z = sorted(sub["local_z"].unique())
print(f"[INFO] 고유 z 수: {len(uniq_z)} → vessel mask 생성 시작", flush=True)
for i, z in enumerate(uniq_z):
    v = compute_vessel_mask_s3(ct, roi, z)
    lz = les[int(z)] > 0
    cache[z] = (lz, v & ~lz)   # (risky=lesion, normal=vessel&~lesion)
    if (i + 1) % 40 == 0:
        print(f"  ... {i+1}/{len(uniq_z)} z done", flush=True)

for _, r in sub.iterrows():
    z = int(r["local_z"]); y0, x0, y1, x1 = int(r.crop_y0), int(r.crop_x0), int(r.crop_y1), int(r.crop_x1)
    risky, normal = cache[z]
    rr = float(risky[y0:y1, x0:x1].mean())
    nn = float(normal[y0:y1, x0:x1].mean())
    risky_ratios.append(rr); normal_ratios.append(nn)

sub["risky_ratio"]  = risky_ratios
sub["normal_ratio"] = normal_ratios
sub["score_raw"]    = sub[SCORE_COL]
sub["score_adj"]    = sub["score_raw"] + ALPHA * sub["risky_ratio"] - BETA * sub["normal_ratio"]
sub["delta"]        = sub["score_adj"] - sub["score_raw"]

# ─ 검증 지표 ────────────────────────────────────────────────────────────────
pos = sub[sub["label"] == "positive"]
neg = sub[sub["label"] == "hard_negative"]
pos_on_lesion = pos[pos["risky_ratio"] > 0]
neg_on_vessel = neg[neg["normal_ratio"] > 0]
print("\n========== 보정 동작 검증 ==========", flush=True)
print(f"positive patch (병변겹침>0) {len(pos_on_lesion)}/{len(pos)}: 평균 delta = {pos_on_lesion['delta'].mean():+.5f}  (양수 기대)", flush=True)
print(f"hard_neg patch (혈관겹침>0) {len(neg_on_vessel)}/{len(neg)}: 평균 delta = {neg_on_vessel['delta'].mean():+.5f}  (음수 기대)", flush=True)
print(f"전체 positive 평균 delta = {pos['delta'].mean():+.5f}", flush=True)
print(f"전체 hard_neg  평균 delta = {neg['delta'].mean():+.5f}", flush=True)
print(f"max 가점 = {sub['delta'].max():+.5f}   max 감점 = {sub['delta'].min():+.5f}", flush=True)

out_csv = OUT / f"score_adjust_{SAFE_ID}.csv"
sub.to_csv(out_csv, index=False)
print(f"\n[OK] saved: {out_csv}", flush=True)

# ─ 가장 크게 가점/감점된 patch overlay 1장씩 ───────────────────────────────
def overlay_patch(row, tag):
    z = int(row["local_z"]); risky, normal = cache[z]
    ct_disp = lung_window(ct[z])
    fig, ax = plt.subplots(figsize=(6, 6)); ax.axis("off")
    ax.imshow(ct_disp, cmap="gray")
    ov = np.zeros((*ct_disp.shape, 4))
    ov[normal] = [0, 0.4, 1, 0.45]; ov[risky] = [1, 0, 0, 0.55]
    ax.imshow(ov)
    import matplotlib.patches as mpatches
    ax.add_patch(mpatches.Rectangle((row.crop_x0, row.crop_y0), 96, 96, fill=False, edgecolor="yellow", lw=2))
    ax.set_title(f"{tag} z={z} {row['label']}\nraw={row.score_raw:.3f} adj={row.score_adj:.3f} d={row.delta:+.3f}")
    p = OUT / f"patch_{tag}_z{z}.png"
    plt.tight_layout(); plt.savefig(p, dpi=100, bbox_inches="tight"); plt.close()
    print(f"[OK] {p}", flush=True)

overlay_patch(sub.loc[sub["delta"].idxmax()], "max_up")
overlay_patch(sub.loc[sub["delta"].idxmin()], "max_down")
print("\n[DONE]", flush=True)
