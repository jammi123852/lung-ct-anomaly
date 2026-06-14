#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
oracle_vessel_score_adjust_full_v1
- 152명 stage1_dev 전체에 oracle 혈관 보정 적용 (병렬)
- score_adj = score_raw + ALPHA*risky_ratio - BETA*normal_ratio
- stage2 절대 미접근. read-only 입력. 새 폴더에만 저장.
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from multiprocessing import Pool
from skimage.filters import frangi
from skimage.morphology import disk, remove_small_objects, white_tophat

PROJECT = Path("/home/jinhy/project/lung-ct-anomaly")
NROOT   = Path("/mnt/c/Users/jinhy/Desktop/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy")
SCORE_CSV = PROJECT / "outputs/normal_based_stage2_verifier_audit/rd_e1e2_lung3ch_effb0_shard_run_v1/rd_d1s_stage1dev_candidate_score.csv"
SCORE_COL = "rd_d1s_medi3ch_rd4ad_score"
OUT = PROJECT / "outputs/reports/oracle_vessel_score_adjust_full_v1"
OUT.mkdir(parents=True, exist_ok=True)
ERR_CSV = OUT / "errors.csv"

ALPHA, BETA = 0.05, 0.05
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
    if u.any(): u = remove_small_objects(u.copy(), min_size=MIN_AREA)
    return u

def process_patient(args):
    safe_id, rows = args
    try:
        pdir = NROOT / safe_id
        ct = np.load(pdir / "ct_hu.npy"); roi = np.load(pdir / "roi_0_0.npy"); les = np.load(pdir / "lesion_mask_roi_0_0.npy")
        cache = {}
        rr_list, nn_list = [], []
        for _, r in rows.iterrows():
            z = int(r["local_z"])
            if z not in cache:
                v = vessel_mask(ct, roi, z); lz = les[z] > 0
                cache[z] = (lz, v & ~lz)
            risky, normal = cache[z]
            y0, x0, y1, x1 = int(r.crop_y0), int(r.crop_x0), int(r.crop_y1), int(r.crop_x1)
            rr_list.append(float(risky[y0:y1, x0:x1].mean()))
            nn_list.append(float(normal[y0:y1, x0:x1].mean()))
        out = rows.copy()
        out["risky_ratio"] = rr_list; out["normal_ratio"] = nn_list
        out["score_raw"] = out[SCORE_COL]
        out["score_adj"] = out["score_raw"] + ALPHA*out["risky_ratio"] - BETA*out["normal_ratio"]
        return ("ok", safe_id, out)
    except Exception as e:
        return ("err", safe_id, str(e))

def main():
    df = pd.read_csv(SCORE_CSV)
    groups = [(sid, g) for sid, g in df.groupby("safe_id")]
    print(f"[INFO] 환자 {len(groups)}명, patch {len(df)}개, worker {N_WORKERS}", flush=True)
    t0 = time.time(); results = []; errs = []; done = 0
    with Pool(N_WORKERS) as pool:
        for status, sid, payload in pool.imap_unordered(process_patient, groups):
            done += 1
            if status == "ok":
                results.append(payload)
            else:
                errs.append({"safe_id": sid, "error": payload}); print(f"[ERR] {sid}: {payload}", flush=True)
            if done % 20 == 0 or done == len(groups):
                el = time.time()-t0
                print(f"  {done}/{len(groups)} done  {el:.0f}s  (eta {el/done*(len(groups)-done):.0f}s)", flush=True)
    pd.DataFrame(errs).to_csv(ERR_CSV, index=False)
    alldf = pd.concat(results, ignore_index=True)
    out_csv = OUT / "score_adjust_all_stage1dev.csv"
    alldf.to_csv(out_csv, index=False)
    # 요약
    pos = alldf[alldf["label"] == "positive"]; neg = alldf[alldf["label"] == "hard_negative"]
    print("\n========== 전체 요약 ==========", flush=True)
    print(f"saved: {out_csv}  rows={len(alldf)}  errors={len(errs)}", flush=True)
    print(f"positive 평균 delta = {(pos['score_adj']-pos['score_raw']).mean():+.5f}", flush=True)
    print(f"hard_neg 평균 delta = {(neg['score_adj']-neg['score_raw']).mean():+.5f}", flush=True)
    print(f"총 소요 {time.time()-t0:.0f}s", flush=True)
    print("[DONE]", flush=True)

if __name__ == "__main__":
    main()
