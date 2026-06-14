"""
RD-E1 A/B/C z-track 평가 스크립트
- 각 실험의 stage1_dev candidate score CSV 로드
- strict same-position z-track (min_run=2) 적용
- P1(×roi), P5(P1×len/3) 계산
- patient hit rate top-k 계산
- 결과를 rd4ad_eval.json 동일 포맷으로 저장
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")

# 입력
D1S_MERGED = (
    PROJECT_ROOT
    / "experiments/rd4ad_strict_same_position_ztrack_actual_scoring_v1"
    / "manifests/strict_ztrack_scores_full_merged.csv"
)

EXPERIMENTS = {
    "A_lung3ch": PROJECT_ROOT
        / "outputs/normal_based_stage2_verifier_audit"
        / "rd_e1a_lung3ch_true_rd4ad_shard_run_v1"
        / "rd_d1s_stage1dev_candidate_score.csv",
    "B_medi_mip3ch": PROJECT_ROOT
        / "outputs/normal_based_stage2_verifier_audit"
        / "rd_e1b_medi_mip3ch_true_rd4ad_shard_run_v1"
        / "rd_d1s_stage1dev_candidate_score.csv",
    "C_lung_mip3ch": PROJECT_ROOT
        / "outputs/normal_based_stage2_verifier_audit"
        / "rd_e1c_lung_mip3ch_true_rd4ad_shard_run_v1"
        / "rd_d1s_stage1dev_candidate_score.csv",
}

OUT_DIR = PROJECT_ROOT / "outputs/end/rd4ad_padim_final_eval_summary_v1"
OUT_JSON = OUT_DIR / "rd_e1_abc_eval.json"

TOP_KS = [1, 3, 5, 10, 20, 30, 50]
MIN_RUN = 2


def load_roi_map(d1s_path):
    """candidate_id → roi_0_0_patch_ratio"""
    roi_map = {}
    with open(d1s_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            roi_map[row["candidate_id"]] = float(row["roi_0_0_patch_ratio"])
    print(f"  roi_map loaded: {len(roi_map)} entries")
    return roi_map


def load_scores(csv_path, roi_map):
    """CSV 로드 → 후보별 dict 리스트"""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["candidate_id"]
            roi = roi_map.get(cid, 1.0)
            raw = float(row["rd_d1s_medi3ch_rd4ad_score"])
            rows.append({
                "candidate_id": cid,
                "patient_id":   row["patient_id"],
                "local_z":      int(float(row["local_z"])),
                "y0": int(row["crop_y0"]), "x0": int(row["crop_x0"]),
                "y1": int(row["crop_y1"]), "x1": int(row["crop_x1"]),
                "label":  row["label"],
                "raw":    raw,
                "roi":    roi,
                "P1":     raw * roi,
            })
    return rows


def apply_ztrack(rows):
    """strict same-position z-track, min_run=2.
    z축 gap(diff>1)에서 track 분리 → 각 연속 run이 독립 track.
    """
    # group by (patient, y0,x0,y1,x1)
    groups = defaultdict(list)
    for r in rows:
        key = (r["patient_id"], r["y0"], r["x0"], r["y1"], r["x1"])
        groups[key].append(r)

    survived_rows = []
    track_meta = {}  # track_id → {track_len, has_positive, raw_top3, p1_top3, p5}

    for key, grp in groups.items():
        grp.sort(key=lambda x: x["local_z"])
        pid = key[0]

        # z gap에서 연속 run 분리
        runs = []
        cur = [grp[0]]
        for i in range(1, len(grp)):
            if grp[i]["local_z"] - grp[i-1]["local_z"] == 1:
                cur.append(grp[i])
            else:
                runs.append(cur)
                cur = [grp[i]]
        runs.append(cur)

        for run in runs:
            if len(run) < MIN_RUN:
                continue

            z_start = run[0]["local_z"]
            z_end   = run[-1]["local_z"]
            track_id = f"{pid}|{key[1]}_{key[2]}_{key[3]}_{key[4]}|{z_start}_{z_end}"

            has_pos = any(r["label"] == "positive" for r in run)
            raw_vals = sorted([r["raw"] for r in run], reverse=True)
            p1_vals  = sorted([r["P1"]  for r in run], reverse=True)
            tlen = len(run)

            top3_raw = sum(raw_vals[:3]) / min(3, tlen)
            top3_p1  = sum(p1_vals[:3])  / min(3, tlen)
            p5_score = top3_p1 * (tlen / 3.0)

            track_meta[track_id] = {
                "patient_id":   pid,
                "track_len":    tlen,
                "has_positive": has_pos,
                "raw_top3":     top3_raw,
                "p1_top3":      top3_p1,
                "p5":           p5_score,
            }
            survived_rows.extend(run)

    return track_meta, survived_rows


def patient_hit_rate(track_meta, score_key, top_ks):
    """환자별 top-k tracks 선택 → positive 환자 hit 비율"""
    # patient별 track 목록 (score, has_positive)
    patient_tracks = defaultdict(list)
    for tm in track_meta.values():
        patient_tracks[tm["patient_id"]].append((tm[score_key], tm["has_positive"]))

    all_patients = list(patient_tracks.keys())
    positive_patients = [p for p in all_patients
                         if any(hp for _, hp in patient_tracks[p])]
    n_pos = len(positive_patients)
    print(f"    total patients={len(all_patients)}, positive={n_pos}")

    results = {}
    for k in top_ks:
        hit = 0
        for pid in positive_patients:
            tracks = sorted(patient_tracks[pid], key=lambda x: -x[0])
            topk = tracks[:k]
            if any(hp for _, hp in topk):
                hit += 1
        hit_rate = hit / n_pos if n_pos > 0 else 0.0
        results[f"top{k}"] = round(hit_rate, 4)

    return results, n_pos


def run():
    print("Loading roi_map...")
    roi_map = load_roi_map(D1S_MERGED)

    results = {}

    for exp_name, csv_path in EXPERIMENTS.items():
        print(f"\n=== {exp_name} ===")
        rows = load_scores(csv_path, roi_map)
        print(f"  loaded {len(rows)} rows")

        track_meta, survived = apply_ztrack(rows)
        n_tracks = len(track_meta)
        n_survived = len(survived)
        print(f"  survived rows={n_survived}, tracks={n_tracks}")

        print("  RAW top3:")
        raw_hr, n_pos = patient_hit_rate(track_meta, "raw_top3", TOP_KS)
        print("  P1 top3:")
        p1_hr, _     = patient_hit_rate(track_meta, "p1_top3", TOP_KS)
        print("  P5 top3:")
        p5_hr, _     = patient_hit_rate(track_meta, "p5", TOP_KS)

        results[exp_name] = {
            "eval_set": f"stage1_dev {n_pos}명 (positive only)",
            "total_candidates": len(rows),
            "survived_candidates": n_survived,
            "total_z_tracks": n_tracks,
            "positive_patients": n_pos,
            "scores": {
                "RAW (raw_top3)": raw_hr,
                "P1 (raw×roi top3)": p1_hr,
                "P5 (P1×len/3)": p5_hr,
            }
        }
        print(f"  P5 top10={p5_hr.get('top10')}, top20={p5_hr.get('top20')}")

    # D1s 기존 결과 포함 (rd4ad_eval.json에서)
    d1s_ref = {
        "D1s_medi3ch (baseline)": {
            "eval_set": "stage1_dev 152명 (positive only) — 기존 평가",
            "note": "P1/P5/PD: rd4ad_eval.json 기존 결과",
            "scores": {
                "P1 (raw×roi top3)": {
                    "top1": 0.4671, "top3": 0.5789, "top5": 0.6513,
                    "top10": 0.7237, "top20": 0.8224, "top30": 0.8618, "top50": 0.9079
                },
                "P5 (P1×len/3)": {
                    "top1": 0.6579, "top3": 0.8355, "top5": 0.8553,
                    "top10": 0.9145, "top20": 0.9474, "top30": 0.9671, "top50": 0.9737
                },
                "PD (sqrthu×len/3)": {
                    "top1": 0.6974, "top3": 0.8158, "top5": 0.8684,
                    "top10": 0.8882, "top20": 0.9342, "top30": 0.9539, "top50": 0.9737
                },
            }
        }
    }

    out = {
        "description": "RD-E1 A/B/C vs D1s baseline 비교 (stage1_dev, z-track min_run=2)",
        "note": "A/B/C: RAW·P1·P5만 계산 (hu_norm 미포함). D1s: 기존 rd4ad_eval.json 값",
        **d1s_ref,
        **results
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {OUT_JSON}")


if __name__ == "__main__":
    run()
