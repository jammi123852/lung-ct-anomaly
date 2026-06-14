#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
oracle_vessel_alpha_sweep_v1
- 보정강도 α(=β) sweep: score_adj = raw + α*risky_ratio - α*normal_ratio
- risky/normal ratio는 score_adjust_all CSV에 이미 있음 → 재계산 없음
- raw_top3 / p5 hit rate를 α별 비교. stage1_dev only.
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
ALPHAS  = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]   # 0.0 = baseline

def build_tracks(rows, get_score):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["patient_id"], r["y0"], r["x0"], r["y1"], r["x1"])].append(r)
    tm = {}
    for key, grp in groups.items():
        grp.sort(key=lambda x: x["local_z"]); pid = key[0]
        runs, cur = [], [grp[0]]
        for i in range(1, len(grp)):
            if grp[i]["local_z"] - grp[i-1]["local_z"] == 1: cur.append(grp[i])
            else: runs.append(cur); cur = [grp[i]]
        runs.append(cur)
        for ri, run in enumerate(runs):
            if len(run) < MIN_RUN: continue
            tid = f"{pid}|{key[1]}_{key[2]}_{key[3]}_{key[4]}|{ri}|{run[0]['local_z']}"
            tlen = len(run)
            sc = sorted([get_score(r) for r in run], reverse=True)
            p1 = sorted([get_score(r)*r["roi"] for r in run], reverse=True)
            raw_top3 = sum(sc[:3])/min(3, tlen)
            p1_top3  = sum(p1[:3])/min(3, tlen)
            tm[tid] = {"patient_id": pid, "has_positive": any(r["label"]=="positive" for r in run),
                       "raw_top3": raw_top3, "p5": p1_top3*(tlen/3.0)}
    return tm

def hit_rate(tm, key):
    pt = defaultdict(list)
    for t in tm.values(): pt[t["patient_id"]].append((t[key], t["has_positive"]))
    pos = [p for p in pt if any(hp for _, hp in pt[p])]; n = len(pos); res = {}
    for k in TOP_KS:
        hit = sum(1 for pid in pos if any(hp for _, hp in sorted(pt[pid], key=lambda x: x[0], reverse=True)[:k]))
        res[k] = round(hit/n, 4) if n else 0.0
    return res

def main():
    df = pd.read_csv(IN_CSV)
    roi_df = pd.read_csv(ROI_SRC, usecols=["candidate_id", "roi_0_0_patch_ratio"])
    roi_map = dict(zip(roi_df["candidate_id"], roi_df["roi_0_0_patch_ratio"]))
    rows = []
    for r in df.itertuples():
        rows.append({"patient_id": r.patient_id, "local_z": int(r.local_z),
                     "y0": int(r.crop_y0), "x0": int(r.crop_x0), "y1": int(r.crop_y1), "x1": int(r.crop_x1),
                     "label": r.label, "roi": float(roi_map.get(r.candidate_id, 1.0)),
                     "raw": float(r.score_raw), "risky": float(r.risky_ratio), "normal": float(r.normal_ratio)})
    print(f"[INFO] rows={len(rows)} patients={df['patient_id'].nunique()}")

    out = {}
    for a in ALPHAS:
        gs = (lambda r, a=a: r["raw"] + a*r["risky"] - a*r["normal"])
        tm = build_tracks(rows, gs)
        out[f"alpha_{a}"] = {"raw_top3": hit_rate(tm, "raw_top3"), "p5": hit_rate(tm, "p5")}

    for metric in ["raw_top3", "p5"]:
        print(f"\n{'='*64}\n  [{metric}]  α별 hit rate  (α=0.0 은 baseline)\n{'='*64}")
        print(f"  {'α':>5} | " + " | ".join(f"top{k:>2}" for k in TOP_KS))
        base = out["alpha_0.0"][metric]
        for a in ALPHAS:
            hr = out[f"alpha_{a}"][metric]
            cells = " | ".join(f"{hr[k]:.4f}" for k in TOP_KS)
            tag = "  <-- baseline" if a == 0.0 else ""
            print(f"  {a:>5} | {cells}{tag}")
        # Δ from baseline (max)
        print("  Δmax vs baseline:", {k: round(max(out[f'alpha_{a}'][metric][k] for a in ALPHAS)-base[k], 4) for k in TOP_KS})

    (OUT / "alpha_sweep_hitrate.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[OK] saved: {OUT/'alpha_sweep_hitrate.json'}")
    print("[DONE]")

if __name__ == "__main__":
    main()
