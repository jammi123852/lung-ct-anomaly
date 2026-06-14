"""
RD-E1 A/B/C 전체 평가 스크립트
- D1s sweep과 동일한 모든 scoring method 적용
- hu_norm_mean은 CT에서 직접 계산 (medi window [-160,240], ROI mask)
- 결과: outputs/end/rd4ad_padim_final_eval_summary_v1/rd_e1_abc_full_eval.json
"""
import csv, json, math, sys
from collections import defaultdict, OrderedDict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")

CT_ROOT = Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1"
    "/volumes_npy"
)
MASK_ROOT = (
    PROJECT_ROOT
    / "outputs/mip-postprocess-research-v1/masks"
    / "refined_roi_v4_20_modeB_all_v1"
)
D1S_MERGED = (
    PROJECT_ROOT
    / "experiments/rd4ad_strict_same_position_ztrack_actual_scoring_v1"
    / "manifests/strict_ztrack_scores_full_merged.csv"
)
EXPERIMENTS = {
    "A_lung3ch": (
        PROJECT_ROOT
        / "outputs/normal_based_stage2_verifier_audit"
        / "rd_e1a_lung3ch_true_rd4ad_shard_run_v1"
        / "rd_d1s_stage1dev_candidate_score.csv"
    ),
    "B_medi_mip3ch": (
        PROJECT_ROOT
        / "outputs/normal_based_stage2_verifier_audit"
        / "rd_e1b_medi_mip3ch_true_rd4ad_shard_run_v1"
        / "rd_d1s_stage1dev_candidate_score.csv"
    ),
    "C_lung_mip3ch": (
        PROJECT_ROOT
        / "outputs/normal_based_stage2_verifier_audit"
        / "rd_e1c_lung_mip3ch_true_rd4ad_shard_run_v1"
        / "rd_d1s_stage1dev_candidate_score.csv"
    ),
    "C2_lung_mip3ch_roipx": (
        PROJECT_ROOT
        / "outputs/normal_based_stage2_verifier_audit"
        / "rd_e1c2_lung_mip3ch_roipx_true_rd4ad_shard_run_v1"
        / "rd_d1s_stage1dev_candidate_score.csv"
    ),
    "A2_lung3ch_roipx": (
        PROJECT_ROOT
        / "outputs/normal_based_stage2_verifier_audit"
        / "rd_e1a2_lung3ch_roipx_true_rd4ad_shard_run_v1"
        / "rd_d1s_stage1dev_candidate_score.csv"
    ),
    "E1_lung_mip3ch_effb0": (
        PROJECT_ROOT
        / "outputs/normal_based_stage2_verifier_audit"
        / "rd_e1e1_lung_mip3ch_effb0_shard_run_v1"
        / "rd_d1s_stage1dev_candidate_score.csv"
    ),
    "E2_lung3ch_effb0": (
        PROJECT_ROOT
        / "outputs/normal_based_stage2_verifier_audit"
        / "rd_e1e2_lung3ch_effb0_shard_run_v1"
        / "rd_d1s_stage1dev_candidate_score.csv"
    ),
}
OUT_JSON = (
    PROJECT_ROOT
    / "outputs/end/rd4ad_padim_final_eval_summary_v1"
    / "rd_e1_abc_full_eval.json"
)
HU_CACHE_CSV = (
    PROJECT_ROOT
    / "outputs/end/rd4ad_padim_final_eval_summary_v1"
    / "rd_e1_candidate_hu_cache.csv"
)

HU_MIN, HU_MAX = -160.0, 240.0
MIN_RUN = 2
TOP_KS  = [1, 3, 5, 10, 20, 30, 50]
EPS     = 1e-6


# ── utils ────────────────────────────────────────────────────────────────────

class NpyCache:
    def __init__(self, max_size=10):
        self._c = OrderedDict()
        self._max = max_size
    def get(self, path):
        k = str(path)
        if k in self._c:
            self._c.move_to_end(k)
            return self._c[k]
        arr = np.load(path, mmap_mode="r")
        if len(self._c) >= self._max:
            self._c.popitem(last=False)
        self._c[k] = arr
        return arr


def find_mask(safe_id):
    for sub in ("lesion", "normal"):
        p = MASK_ROOT / sub / safe_id / "refined_roi.npy"
        if p.exists():
            return p
    return None


def compute_hu_norm(ct_arr, roi_arr, z, y0, x0, y1, x1):
    Z, H, W = ct_arr.shape
    y0c, y1c = max(y0, 0), min(y1, H)
    x0c, x1c = max(x0, 0), min(x1, W)
    z = max(0, min(z, Z - 1))
    if y1c <= y0c or x1c <= x0c:
        return np.nan
    hu = ct_arr[z, y0c:y1c, x0c:x1c].astype(np.float32)
    if roi_arr is not None:
        roi = roi_arr[z, y0c:y1c, x0c:x1c].astype(bool)
        if roi.shape != hu.shape:
            roi = roi[:hu.shape[0], :hu.shape[1]]
        inside = hu[roi]
        if len(inside) == 0:
            inside = hu.ravel()
    else:
        inside = hu.ravel()
    if len(inside) == 0:
        return np.nan
    norm = (inside.clip(HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)
    return float(norm.mean())


# ── step 1: load D1s roi_map ─────────────────────────────────────────────────

def load_roi_map():
    roi_map = {}
    with open(D1S_MERGED, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            roi_map[row["candidate_id"]] = float(row["roi_0_0_patch_ratio"])
    print(f"  roi_map: {len(roi_map)} entries")
    return roi_map


# ── step 2: load candidate rows (A as base: positions identical) ─────────────

def load_base_rows(roi_map):
    """A 스코어 CSV에서 공통 필드 로드 (position, label, roi 포함)."""
    rows = []
    path = EXPERIMENTS["A_lung3ch"]
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["candidate_id"]
            rows.append({
                "candidate_id": cid,
                "patient_id":   row["patient_id"],
                "safe_id":      row["safe_id"],
                "local_z":      int(float(row["local_z"])),
                "y0": int(row["crop_y0"]), "x0": int(row["crop_x0"]),
                "y1": int(row["crop_y1"]), "x1": int(row["crop_x1"]),
                "label": row["label"],
                "roi":   roi_map.get(cid, 1.0),
            })
    print(f"  base rows: {len(rows)}")
    return rows


# ── step 3: compute or load hu_norm_mean ─────────────────────────────────────

def get_hu_map(base_rows):
    if HU_CACHE_CSV.exists():
        print(f"  hu cache 존재 → 로드: {HU_CACHE_CSV}")
        hu_map = {}
        with open(HU_CACHE_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                hu_map[row["candidate_id"]] = float(row["hu_norm_mean"]) if row["hu_norm_mean"] else np.nan
        print(f"  hu_map: {len(hu_map)} entries")
        return hu_map

    print(f"  hu cache 없음 → CT에서 계산 ({len(base_rows):,} candidates)...")
    ct_cache   = NpyCache(max_size=10)
    mask_cache = NpyCache(max_size=10)
    hu_map = {}
    n_miss = 0

    for i, row in enumerate(base_rows):
        if i % 10000 == 0:
            print(f"    {i:,}/{len(base_rows):,}  miss={n_miss}")
        cid     = row["candidate_id"]
        safe_id = row["safe_id"]
        ct_path = CT_ROOT / safe_id / "ct_hu.npy"
        if not ct_path.exists():
            hu_map[cid] = np.nan
            n_miss += 1
            continue
        mask_path = find_mask(safe_id)
        try:
            ct_arr  = ct_cache.get(ct_path)
            roi_arr = mask_cache.get(mask_path) if mask_path else None
            val = compute_hu_norm(ct_arr, roi_arr,
                                  row["local_z"],
                                  row["y0"], row["x0"], row["y1"], row["x1"])
            hu_map[cid] = val
        except Exception:
            hu_map[cid] = np.nan
            n_miss += 1

    print(f"  완료: miss={n_miss}")
    nan_cnt = sum(1 for v in hu_map.values() if v != v)
    print(f"  NaN={nan_cnt} → 중앙값으로 대체")
    vals = [v for v in hu_map.values() if v == v]
    median_hu = float(np.median(vals)) if vals else 0.0
    for k, v in hu_map.items():
        if v != v:
            hu_map[k] = median_hu

    # 캐시 저장
    HU_CACHE_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(HU_CACHE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "hu_norm_mean"])
        for cid, val in hu_map.items():
            writer.writerow([cid, val])
    print(f"  캐시 저장: {HU_CACHE_CSV}")
    return hu_map


# ── step 4: apply z-track ────────────────────────────────────────────────────

def apply_ztrack_all(base_rows, hu_map, exp_scores):
    """
    exp_scores: {candidate_id: {exp_name: raw_score}}
    반환: track_meta dict keyed by track_id
    """
    groups = defaultdict(list)
    for r in base_rows:
        key = (r["patient_id"], r["y0"], r["x0"], r["y1"], r["x1"])
        groups[key].append(r)

    track_meta = {}
    survived_count = 0

    for key, grp in groups.items():
        grp.sort(key=lambda x: x["local_z"])
        pid = key[0]

        runs = []
        cur = [grp[0]]
        for i in range(1, len(grp)):
            if grp[i]["local_z"] - grp[i-1]["local_z"] == 1:
                cur.append(grp[i])
            else:
                runs.append(cur); cur = [grp[i]]
        runs.append(cur)

        for run in runs:
            if len(run) < MIN_RUN:
                continue
            survived_count += len(run)
            z0 = run[0]["local_z"]; z1 = run[-1]["local_z"]
            tid = f"{pid}|{key[1]}_{key[2]}_{key[3]}_{key[4]}|{z0}_{z1}"
            has_pos = any(r["label"] == "positive" for r in run)
            tlen    = len(run)

            roi_vals = [r["roi"] for r in run]
            hu_vals  = [hu_map[r["candidate_id"]] for r in run]
            hu_mean  = float(np.mean(hu_vals))

            # per-experiment raw scores → P1 = raw × roi
            exp_raw     = {}
            exp_p1      = {}
            exp_raw_max = {}
            exp_p1_max  = {}
            for exp in exp_scores:
                raws = sorted([exp_scores[exp].get(r["candidate_id"], 0.0) for r in run], reverse=True)
                p1s  = sorted([exp_scores[exp].get(r["candidate_id"], 0.0) * r["roi"] for r in run], reverse=True)
                exp_raw[exp]     = sum(raws[:3]) / min(3, tlen)
                exp_p1[exp]      = sum(p1s[:3])  / min(3, tlen)
                exp_raw_max[exp] = raws[0] if raws else 0.0
                exp_p1_max[exp]  = p1s[0]  if p1s  else 0.0

            track_meta[tid] = {
                "patient_id":   pid,
                "track_len":    tlen,
                "has_positive": has_pos,
                "hu":           hu_mean,
                "exp_raw":      exp_raw,
                "exp_p1":       exp_p1,
                "exp_raw_max":  exp_raw_max,
                "exp_p1_max":   exp_p1_max,
            }

    print(f"  survived_rows={survived_count}, tracks={len(track_meta)}")
    return track_meta


# ── step 5: load all experiment raw scores ───────────────────────────────────

def load_exp_scores():
    exp_scores = {}
    for exp_name, csv_path in EXPERIMENTS.items():
        sm = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sm[row["candidate_id"]] = float(row["rd_d1s_medi3ch_rd4ad_score"])
        exp_scores[exp_name] = sm
        print(f"  {exp_name}: {len(sm)} scores")
    return exp_scores


# ── step 6: compute all scores per track ─────────────────────────────────────

def compute_all_scores(track_meta, exp_names):
    """track_meta에 모든 score columns 추가. top3_mean + max 두 집계 방식."""
    for tid, tm in track_meta.items():
        tl  = tm["track_len"]
        hu  = tm["hu"]
        for exp in exp_names:
            p1     = tm["exp_p1"][exp]       # top3_mean 집계
            p1_max = tm["exp_p1_max"][exp]   # max 집계
            # ── top3_mean 집계 ──────────────────────────────────────────────
            tm[f"{exp}_RAW"]  = tm["exp_raw"][exp]
            tm[f"{exp}_P1"]   = p1
            tm[f"{exp}_P3"]   = p1 * math.log(max(tl, 1))
            tm[f"{exp}_P4"]   = p1 * math.sqrt(tl)
            tm[f"{exp}_P5"]   = p1 * (tl / 3.0)
            tm[f"{exp}_P6"]   = p1 * (hu + EPS)
            tm[f"{exp}_P7"]   = p1 * (hu + EPS) * (tl / 3.0)
            tm[f"{exp}_P8"]   = p1 * (hu + EPS) * math.log(max(tl, 1))
            tm[f"{exp}_P9"]   = p1 * math.sqrt(max(hu, 0)) * (tl / 3.0)
            tm[f"{exp}_PA"]   = hu + EPS
            tm[f"{exp}_PB"]   = (hu + EPS) * (tl / 3.0)
            tm[f"{exp}_PC"]   = (hu + EPS) * math.log(max(tl, 1))
            tm[f"{exp}_PD"]   = math.sqrt(max(hu, 0)) * (tl / 3.0)
            # ── max 집계 ───────────────────────────────────────────────────
            tm[f"{exp}_RAW_max"] = tm["exp_raw_max"][exp]
            tm[f"{exp}_P1_max"]  = p1_max
            tm[f"{exp}_P3_max"]  = p1_max * math.log(max(tl, 1))
            tm[f"{exp}_P4_max"]  = p1_max * math.sqrt(tl)
            tm[f"{exp}_P5_max"]  = p1_max * (tl / 3.0)
            tm[f"{exp}_P6_max"]  = p1_max * (hu + EPS)
            tm[f"{exp}_P7_max"]  = p1_max * (hu + EPS) * (tl / 3.0)
            tm[f"{exp}_P8_max"]  = p1_max * (hu + EPS) * math.log(max(tl, 1))
            tm[f"{exp}_P9_max"]  = p1_max * math.sqrt(max(hu, 0)) * (tl / 3.0)


# ── step 7: patient hit rate ─────────────────────────────────────────────────

def patient_hit_rate(track_meta, score_col, top_ks):
    patient_tracks = defaultdict(list)
    for tm in track_meta.values():
        patient_tracks[tm["patient_id"]].append((tm[score_col], tm["has_positive"]))

    positive_patients = [p for p, tracks in patient_tracks.items()
                         if any(hp for _, hp in tracks)]
    n_pos = len(positive_patients)

    results = {}
    for k in top_ks:
        hit = 0
        for pid in positive_patients:
            topk = sorted(patient_tracks[pid], key=lambda x: -x[0])[:k]
            if any(hp for _, hp in topk):
                hit += 1
        results[f"top{k}"] = round(hit / n_pos, 4) if n_pos > 0 else 0.0
    return results, n_pos


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=== [1] Load roi_map ===")
    roi_map = load_roi_map()

    print("\n=== [2] Load base rows ===")
    base_rows = load_base_rows(roi_map)

    print("\n=== [3] Compute hu_norm_mean ===")
    hu_map = get_hu_map(base_rows)

    print("\n=== [4] Load experiment scores ===")
    exp_scores = load_exp_scores()

    print("\n=== [5] Apply z-track ===")
    track_meta = apply_ztrack_all(base_rows, hu_map, exp_scores)

    print("\n=== [6] Compute all score cols ===")
    exp_names = list(EXPERIMENTS.keys())
    compute_all_scores(track_meta, exp_names)

    print("\n=== [7] Patient hit rate ===")

    # top3_mean 집계 방식
    SCORE_METHODS = ["RAW", "P1", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "PA", "PB", "PC", "PD"]
    # max 집계 방식 (track 내 최대 patch score 기반)
    SCORE_METHODS_MAX = ["RAW_max", "P1_max", "P3_max", "P4_max", "P5_max", "P6_max", "P7_max", "P8_max", "P9_max"]
    SCORE_DESC = {
        "RAW": "raw rd4ad score top3",
        "P1":  "raw × roi_ratio top3",
        "P3":  "P1 × log(track_len)",
        "P4":  "P1 × sqrt(track_len)",
        "P5":  "P1 × track_len/3",
        "P6":  "P1 × hu_norm",
        "P7":  "P1 × hu_norm × len/3",
        "P8":  "P1 × hu_norm × log(len)",
        "P9":  "P1 × sqrt(hu_norm) × len/3",
        "PA":  "hu_norm only",
        "PB":  "hu_norm × len/3",
        "PC":  "hu_norm × log(len)",
        "PD":  "sqrt(hu_norm) × len/3",
        "RAW_max": "raw rd4ad score MAX",
        "P1_max":  "raw × roi_ratio MAX",
        "P3_max":  "P1_max × log(track_len)",
        "P4_max":  "P1_max × sqrt(track_len)",
        "P5_max":  "P1_max × track_len/3",
        "P6_max":  "P1_max × hu_norm",
        "P7_max":  "P1_max × hu_norm × len/3",
        "P8_max":  "P1_max × hu_norm × log(len)",
        "P9_max":  "P1_max × sqrt(hu_norm) × len/3",
    }

    out = {
        "description": "RD-E1 A/B/C 전체 scoring method 비교 (stage1_dev, z-track min_run=2)",
        "note": (
            "hu_norm_mean: medi window [-160,240], ROI mask v4_20. "
            "D1s 기존 결과는 rd4ad_eval.json 참조."
        ),
        "score_methods": SCORE_DESC,
        "experiments": {}
    }

    for exp in exp_names:
        print(f"\n  -- {exp} --")
        exp_result = {}
        for method in SCORE_METHODS + SCORE_METHODS_MAX:
            col = f"{exp}_{method}"
            hr, n_pos = patient_hit_rate(track_meta, col, TOP_KS)
            exp_result[f"{method} ({SCORE_DESC[method]})"] = hr
            print(f"    {method:8s}: top10={hr.get('top10')}, top20={hr.get('top20')}")
        out["experiments"][exp] = {
            "eval_set": f"stage1_dev {n_pos}명 positive only",
            "scores": exp_result
        }

    # D1s 기존 결과 포함
    out["experiments"]["D1s_baseline (ref: rd4ad_eval.json)"] = {
        "eval_set": "stage1_dev 152명 positive only",
        "note": "기존 평가 결과. P9/PD는 hu sweep 결과.",
        "scores": {
            "P1 (raw × roi_ratio top3)":         {"top1":0.4671,"top3":0.5789,"top5":0.6513,"top10":0.7237,"top20":0.8224,"top30":0.8618,"top50":0.9079},
            "P3 (P1 × log(track_len))":          {"top1":0.6316,"top3":0.7697,"top5":0.8355,"top10":0.9079,"top20":0.9474,"top30":None,"top50":0.9737},
            "P4 (P1 × sqrt(track_len))":         {"top1":0.6316,"top3":0.7829,"top5":0.8355,"top10":0.9079,"top20":0.9474,"top30":None,"top50":0.9737},
            "P5 (P1 × track_len/3)":             {"top1":0.6579,"top3":0.8355,"top5":0.8553,"top10":0.9145,"top20":0.9474,"top30":0.9671,"top50":0.9737},
            "P9 (P1 × sqrt(hu) × len/3)":        {"top1":0.6776,"top3":0.7961,"top5":0.8421,"top10":0.8947,"top20":0.9342,"top30":None,"top50":0.9605},
            "PD (sqrt(hu) × len/3)":             {"top1":0.6974,"top3":0.8158,"top5":0.8684,"top10":0.8882,"top20":0.9342,"top30":0.9539,"top50":0.9737},
        }
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {OUT_JSON}")


if __name__ == "__main__":
    main()
