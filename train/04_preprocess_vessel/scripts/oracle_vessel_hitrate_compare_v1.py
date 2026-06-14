#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
oracle_vessel_hitrate_compare_v1
- baseline(score_raw) vs adjusted(score_adj) z-track patient hit rate 비교
- z-track 로직은 rd_e1_abc_ztrack_eval.py 와 동일 (min_run=2, track=top3 mean, P5=p1_top3*len/3)
- stage1_dev only. read-only 입력(생성된 score_adjust CSV). 새 폴더에 결과 저장.
"""
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

PROJECT = Path("/home/jinhy/project/lung-ct-anomaly")
IN_CSV  = PROJECT / "outputs/reports/oracle_vessel_score_adjust_full_v1/score_adjust_all_stage1dev.csv"
ROI_SRC = PROJECT / "experiments/rd4ad_strict_same_position_ztrack_actual_scoring_v1/manifests/strict_ztrack_scores_full_merged.csv"
OUT     = PROJECT / "outputs/reports/oracle_vessel_score_adjust_full_v1"
MIN_RUN = 2
TOP_KS  = [5, 10, 20, 30, 50]

def build_tracks(rows, score_field):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["patient_id"], r["y0"], r["x0"], r["y1"], r["x1"])].append(r)
    track_meta = {}
    for key, grp in groups.items():
        grp.sort(key=lambda x: x["local_z"]); pid = key[0]
        runs, cur = [], [grp[0]]
        for i in range(1, len(grp)):
            if grp[i]["local_z"] - grp[i-1]["local_z"] == 1: cur.append(grp[i])
            else: runs.append(cur); cur = [grp[i]]
        runs.append(cur)
        for ri, run in enumerate(runs):
            if len(run) < MIN_RUN: continue
            tid = f"{pid}|{key[1]}_{key[2]}_{key[3]}_{key[4]}|{run[0]['local_z']}_{run[-1]['local_z']}|{ri}"
            tlen = len(run)
            sc   = sorted([r[score_field] for r in run], reverse=True)
            p1   = sorted([r[score_field]*r["roi"] for r in run], reverse=True)
            raw_top3 = sum(sc[:3])/min(3, tlen)
            p1_top3  = sum(p1[:3])/min(3, tlen)
            track_meta[tid] = {
                "patient_id": pid, "has_positive": any(r["label"]=="positive" for r in run),
                "raw_top3": raw_top3, "p1_top3": p1_top3, "p5": p1_top3*(tlen/3.0),
            }
    return track_meta

def hit_rate(track_meta, score_key):
    pt = defaultdict(list)
    for tm in track_meta.values():
        pt[tm["patient_id"]].append((tm[score_key], tm["has_positive"]))
    pos_patients = [p for p in pt if any(hp for _, hp in pt[p])]
    n = len(pos_patients); res = {}
    for k in TOP_KS:
        hit = 0
        for pid in pos_patients:
            topk = sorted(pt[pid], key=lambda x: x[0], reverse=True)[:k]
            if any(hp for _, hp in topk): hit += 1
        res[f"top{k}"] = round(hit/n, 4) if n else 0.0
    return res, n

def main():
    df = pd.read_csv(IN_CSV)
    roi_df = pd.read_csv(ROI_SRC, usecols=["candidate_id", "roi_0_0_patch_ratio"])
    roi_map = dict(zip(roi_df["candidate_id"], roi_df["roi_0_0_patch_ratio"]))
    print(f"[INFO] roi_map={len(roi_map)} entries")
    miss = 0
    rows = []
    for r in df.itertuples():
        roi = roi_map.get(r.candidate_id)
        if roi is None:
            roi = 1.0; miss += 1
        rows.append({"patient_id": r.patient_id, "local_z": int(r.local_z),
                     "y0": int(r.crop_y0), "x0": int(r.crop_x0), "y1": int(r.crop_y1), "x1": int(r.crop_x1),
                     "label": r.label, "roi": float(roi),
                     "raw": float(r.score_raw), "adj": float(r.score_adj)})
    print(f"[INFO] rows={len(rows)} patients={df['patient_id'].nunique()} roi_missing={miss}")

    out = {}
    for tag, fld in [("baseline_raw", "raw"), ("oracle_adj", "adj")]:
        tm = build_tracks(rows, fld)
        out[tag] = {"n_tracks": len(tm)}
        for sk in ["raw_top3", "p1_top3", "p5"]:
            hr, n = hit_rate(tm, sk)
            out[tag][sk] = hr; out[tag]["positive_patients"] = n

    # 출력 표
    print(f"\n{'='*60}\n  baseline(raw) vs oracle(adj) — stage1_dev hit rate\n{'='*60}")
    for sk in ["raw_top3", "p1_top3", "p5"]:
        print(f"\n[{sk}]  (top-k: hit rate)")
        print(f"  {'k':>5} | {'baseline':>9} | {'oracle':>9} | {'Δ':>8}")
        for k in TOP_KS:
            b = out["baseline_raw"][sk][f"top{k}"]; a = out["oracle_adj"][sk][f"top{k}"]
            print(f"  {k:>5} | {b:>9.4f} | {a:>9.4f} | {a-b:>+8.4f}")

    (OUT / "hitrate_compare_baseline_vs_oracle.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[OK] saved: {OUT/'hitrate_compare_baseline_vs_oracle.json'}")
    print("[DONE]")

if __name__ == "__main__":
    main()
