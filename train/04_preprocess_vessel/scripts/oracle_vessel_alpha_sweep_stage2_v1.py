#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
oracle_vessel_alpha_sweep_stage2_v1
- ★stage2 봉인셋, stage1 확정 방법 그대로 적용 (튜닝 금지)
- label=0/1, roi=roi_0_0_patch_ratio 직접 컬럼(roi_missing 없음)
"""
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

PROJECT = Path("/home/jinhy/project/lung-ct-anomaly")
IN_CSV  = PROJECT / "outputs/reports/oracle_vessel_score_adjust_stage2_v1/score_adjust_all_stage2.csv"
OUT     = PROJECT / "outputs/reports/oracle_vessel_score_adjust_stage2_v1"
MIN_RUN = 2
TOP_KS  = [5, 10, 20, 30, 50]
ALPHAS  = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

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
            tm[tid] = {"patient_id": pid, "has_positive": any(r["label"] == 1 for r in run),
                       "raw_top3": sum(sc[:3])/min(3, tlen), "p5": (sum(p1[:3])/min(3, tlen))*(tlen/3.0)}
    return tm

def hit_rate(tm, key):
    pt = defaultdict(list)
    for t in tm.values(): pt[t["patient_id"]].append((t[key], t["has_positive"]))
    pos = [p for p in pt if any(hp for _, hp in pt[p])]; n = len(pos); res = {}
    for k in TOP_KS:
        hit = sum(1 for pid in pos if any(hp for _, hp in sorted(pt[pid], key=lambda x: x[0], reverse=True)[:k]))
        res[k] = round(hit/n, 4) if n else 0.0
    return res, n

def main():
    df = pd.read_csv(IN_CSV)
    rows = [{"patient_id": r.patient_id, "local_z": int(r.local_z),
             "y0": int(r.crop_y0), "x0": int(r.crop_x0), "y1": int(r.crop_y1), "x1": int(r.crop_x1),
             "label": int(r.label), "roi": float(r.roi_0_0_patch_ratio),
             "raw": float(r.score_raw), "risky": float(r.risky_ratio), "normal": float(r.normal_ratio)}
            for r in df.itertuples()]
    n_pat = df['patient_id'].nunique()
    print(f"[INFO] stage2 rows={len(rows)} patients={n_pat}")

    out = {}
    for a in ALPHAS:
        gs = (lambda r, a=a: r["raw"] + a*r["risky"] - a*r["normal"])
        tm = build_tracks(rows, gs)
        hr_raw, n = hit_rate(tm, "raw_top3"); hr_p5, _ = hit_rate(tm, "p5")
        out[f"alpha_{a}"] = {"raw_top3": hr_raw, "p5": hr_p5, "positive_patients": n}

    for metric in ["raw_top3", "p5"]:
        print(f"\n{'='*64}\n  STAGE2 [{metric}]  α별 hit rate  (α=0.0 baseline)\n{'='*64}")
        print(f"  {'α':>5} | " + " | ".join(f"top{k:>2}" for k in TOP_KS))
        for a in ALPHAS:
            hr = out[f"alpha_{a}"][metric]
            tag = "  <-- baseline" if a == 0.0 else ""
            print(f"  {a:>5} | " + " | ".join(f"{hr[k]:.4f}" for k in TOP_KS) + tag)

    (OUT / "alpha_sweep_hitrate_stage2.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[OK] saved: {OUT/'alpha_sweep_hitrate_stage2.json'}")
    print("[DONE]")

if __name__ == "__main__":
    main()
