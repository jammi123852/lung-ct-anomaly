#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
oracle_vessel_score_adjust_stage2_v1
- ★stage2 봉인셋: stage1_dev에서 확정한 방법 그대로 1회 적용 (튜닝 금지, upper-bound 참고용)
- E2 stage2 saved score에 oracle 혈관 라벨 risky/normal ratio 계산 (score_adj는 sweep에서)
- read-only 입력, 새 폴더 저장. score 재계산 없음.
"""
import time, glob
from pathlib import Path
import numpy as np
import pandas as pd
from multiprocessing import Pool
from skimage.filters import frangi
from skimage.morphology import disk, remove_small_objects, white_tophat

PROJECT = Path("/home/jinhy/project/lung-ct-anomaly")
NROOT   = Path("/mnt/c/Users/jinhy/Desktop/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy")
SHARDS  = sorted(glob.glob(str(PROJECT / "experiments/rd_e1_abc_stage2_eval_v1/E2/shards/shard_*/stage2_rd4ad_scores_shard_*.csv")))
SCORE_COL = "rd4ad_ztrack_score_raw"
OUT = PROJECT / "outputs/reports/oracle_vessel_score_adjust_stage2_v1"
OUT.mkdir(parents=True, exist_ok=True)
N_WORKERS = 8
SIGMAS, TOPHAT_RADIUS, PERCENTILE, MIN_AREA = (0.5, 1.0, 1.5, 2.0), 10, 85.0, 10

def lung_window(a, l=-600, w=1500):
    lo, hi = l - w/2, l + w/2; return np.clip((a.astype(np.float32)-lo)/(hi-lo), 0, 1)
def hu_normalize(a): return np.clip((a.astype(np.float32)+1000)/1600, 0, 1)

def vessel_mask(ct, roi, z):
    z0, z1 = max(0, z-1), min(ct.shape[0], z+2)
    mip = np.max(ct[z0:z1].astype(np.float32), axis=0); valid = np.any(roi[z0:z1] > 0, axis=0)
    if not valid.any(): return np.zeros(mip.shape, bool)
    a = lung_window(mip); m1 = (a > np.percentile(a[valid], PERCENTILE)) & valid
    h = hu_normalize(mip); th = white_tophat(h, disk(TOPHAT_RADIUS)); m2 = (th > np.percentile(th[valid], PERCENTILE)) & valid
    fr = frangi(h, sigmas=SIGMAS, black_ridges=False); m3 = (fr > np.percentile(fr[valid], PERCENTILE)) & valid
    u = m1 | m2 | m3
    return remove_small_objects(u.copy(), min_size=MIN_AREA) if u.any() else u

def process_patient(args):
    safe_id, rows = args
    try:
        pdir = NROOT / safe_id
        ct = np.load(pdir / "ct_hu.npy"); roi = np.load(pdir / "roi_0_0.npy"); les = np.load(pdir / "lesion_mask_roi_0_0.npy")
        cache = {}; rr, nn = [], []
        for _, r in rows.iterrows():
            z = int(r["local_z"])
            if z not in cache:
                v = vessel_mask(ct, roi, z); lz = les[z] > 0; cache[z] = (lz, v & ~lz)
            risky, normal = cache[z]
            y0, x0, y1, x1 = int(r.crop_y0), int(r.crop_x0), int(r.crop_y1), int(r.crop_x1)
            rr.append(float(risky[y0:y1, x0:x1].mean())); nn.append(float(normal[y0:y1, x0:x1].mean()))
        out = rows.copy(); out["risky_ratio"] = rr; out["normal_ratio"] = nn; out["score_raw"] = out[SCORE_COL]
        return ("ok", safe_id, out)
    except Exception as e:
        return ("err", safe_id, str(e))

def main():
    df = pd.concat([pd.read_csv(f) for f in SHARDS], ignore_index=True)
    groups = [(sid, g) for sid, g in df.groupby("safe_id")]
    print(f"[INFO] stage2 환자 {len(groups)}명, patch {len(df)}, worker {N_WORKERS}", flush=True)
    t0 = time.time(); results = []; errs = []; done = 0
    with Pool(N_WORKERS) as pool:
        for status, sid, payload in pool.imap_unordered(process_patient, groups):
            done += 1
            if status == "ok": results.append(payload)
            else: errs.append({"safe_id": sid, "error": payload}); print(f"[ERR] {sid}: {payload}", flush=True)
            if done % 20 == 0 or done == len(groups):
                el = time.time()-t0; print(f"  {done}/{len(groups)} done  {el:.0f}s  eta {el/done*(len(groups)-done):.0f}s", flush=True)
    pd.DataFrame(errs).to_csv(OUT / "errors.csv", index=False)
    alldf = pd.concat(results, ignore_index=True)
    out_csv = OUT / "score_adjust_all_stage2.csv"; alldf.to_csv(out_csv, index=False)
    print(f"\n[OK] saved: {out_csv}  rows={len(alldf)}  errors={len(errs)}  소요 {time.time()-t0:.0f}s", flush=True)
    print("[DONE]", flush=True)

if __name__ == "__main__":
    main()
