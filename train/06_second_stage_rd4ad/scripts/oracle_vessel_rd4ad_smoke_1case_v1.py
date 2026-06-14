#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
oracle_vessel_rd4ad_smoke_1case_v1
- 목적: GT lesion 마스크로 oracle 혈관 softmask 만드는 방식 1명 시각 검증
- 방식: compute_vessel_mask_s3 (b1c4z59 동일) 로 혈관후보 → GT lesion subtract → oracle clean vessel
- 출력: overlay PNG (CT + 혈관(감점,파랑) + 병변(가점,빨강))
- read-only 입력, 새 폴더에만 저장. 원본 수정 없음. score 재계산 없음.
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
E2_SHARD0 = PROJECT / "experiments/rd_e1_abc_stage2_eval_v1/E2/shards/shard_0/stage2_rd4ad_scores_shard_0.csv"
OUT = PROJECT / "outputs/reports/oracle_vessel_rd4ad_smoke_1case_v1"
OUT.mkdir(parents=True, exist_ok=True)

# ─ b1c4z59 동일 파라미터/엔진 ──────────────────────────────────────────────
SIGMAS, TOPHAT_RADIUS, PERCENTILE, MIN_AREA = (0.5, 1.0, 1.5, 2.0), 10, 85.0, 10

def lung_window(arr, level=-600, width=1500):
    lo, hi = level - width / 2, level + width / 2
    return np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)

def hu_normalize(arr):
    return np.clip((arr.astype(np.float32) + 1000.0) / 1600.0, 0.0, 1.0)

def compute_vessel_mask_s3(ct_vol, roi_vol, local_z, axis="axial"):
    lz = int(local_z)
    nz = ct_vol.shape[0]
    z0, z1 = max(0, lz - 1), min(nz, lz + 2)
    ct_slab  = ct_vol[z0:z1].astype(np.float32)
    roi_slab = roi_vol[z0:z1]
    mip   = np.max(ct_slab, axis=0)
    valid = np.any(roi_slab > 0, axis=0)
    if not valid.any():
        return np.zeros(mip.shape, dtype=bool), mip
    lw = lung_window(mip)
    m1 = (lw > np.percentile(lw[valid], PERCENTILE)) & valid
    hn = hu_normalize(mip)
    th = white_tophat(hn, disk(TOPHAT_RADIUS))
    m2 = (th > np.percentile(th[valid], PERCENTILE)) & valid
    fr = frangi(hn, sigmas=SIGMAS, black_ridges=False)
    m3 = (fr > np.percentile(fr[valid], PERCENTILE)) & valid
    union = m1 | m2 | m3
    if MIN_AREA > 0 and union.any():
        union = remove_small_objects(union.copy(), min_size=MIN_AREA)
    return union, mip

# ─ 1명 선택: E2 score에서 label==1 첫 환자 ─────────────────────────────────
df = pd.read_csv(E2_SHARD0)
pos = df[df["label"] == 1]
if pos.empty:
    print("[ERROR] label==1 환자 없음 in shard_0", file=sys.stderr); sys.exit(1)
safe_id = str(pos.iloc[0]["safe_id"])
pat = str(pos.iloc[0]["patient_id"])
print(f"[INFO] 선택 환자: patient_id={pat}  safe_id={safe_id}")

pdir = NROOT / safe_id
ct   = np.load(pdir / "ct_hu.npy")
roi  = np.load(pdir / "roi_0_0.npy")
les  = np.load(pdir / "lesion_mask_roi_0_0.npy")
print(f"[INFO] shapes ct={ct.shape} roi={roi.shape} lesion={les.shape}  lesion_vox={int(les.sum())}")

# lesion voxel 가장 많은 axial slice 선택
les_per_z = les.reshape(les.shape[0], -1).sum(axis=1)
zsel = int(np.argmax(les_per_z))
print(f"[INFO] lesion 최다 slice z={zsel} (vox={int(les_per_z[zsel])})")

# ─ oracle 라벨 마스크 생성 ─────────────────────────────────────────────────
#   risky  vessel candidate = GT 병변 마스크          (점수 가점 대상)
#   normal vessel candidate = 혈관 후보 ∩ 병변 밖     (점수 감점 대상)
vessel, mip = compute_vessel_mask_s3(ct, roi, zsel, "axial")
lesion2d = les[zsel] > 0
risky_vessel_candidate  = lesion2d.copy()
normal_vessel_candidate = vessel & ~lesion2d
print(f"[INFO] vessel_raw_vox={int(vessel.sum())}  "
      f"risky(lesion)_vox={int(risky_vessel_candidate.sum())}  "
      f"normal_vessel_vox={int(normal_vessel_candidate.sum())}  "
      f"vessel_on_lesion={int((vessel & lesion2d).sum())}")

# ─ 라벨 마스크 저장 (.npz, 점수 보정 단계에서 재사용) ──────────────────────
out_npz = OUT / f"oracle_vessel_labels_{pat}_z{zsel}.npz"
np.savez_compressed(
    out_npz,
    patient_id=pat, safe_id=safe_id, z=zsel,
    vessel_candidate_raw=vessel,
    risky_vessel_candidate=risky_vessel_candidate,
    normal_vessel_candidate=normal_vessel_candidate,
)
print(f"[OK] saved labels: {out_npz}")

# ─ overlay PNG (영문 라벨) ─────────────────────────────────────────────────
ct_disp = lung_window(ct[zsel])
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
for a in ax: a.axis("off")
ax[0].imshow(ct_disp, cmap="gray"); ax[0].set_title(f"CT z={zsel} ({pat})")

ax[1].imshow(ct_disp, cmap="gray")
ov1 = np.zeros((*ct_disp.shape, 4))
ov1[vessel] = [0, 0.4, 1, 0.5]                   # raw vessel candidate = blue
ax[1].imshow(ov1); ax[1].set_title("vessel candidate (raw)")

ax[2].imshow(ct_disp, cmap="gray")
ov2 = np.zeros((*ct_disp.shape, 4))
ov2[normal_vessel_candidate] = [0, 0.4, 1, 0.5]  # normal vessel candidate = blue (score down)
ov2[risky_vessel_candidate]  = [1, 0, 0, 0.6]    # risky vessel candidate = red (score up)
ax[2].imshow(ov2)
ax[2].set_title("ORACLE labels: blue=normal vessel (down) / red=risky vessel (up)")

out_png = OUT / f"oracle_vessel_{pat}_z{zsel}.png"
plt.tight_layout(); plt.savefig(out_png, dpi=110, bbox_inches="tight"); plt.close()
print(f"[OK] saved: {out_png}")
