"""
rd_e1_abc_stage2_eval.py

목적:
  A/B/C/C2/A2/E1/E2 실험 stage2_holdout shard CSV를 합산하여
  P1/P3/P4/P5/P9/PD 스코어 방식별 patient hit rate 계산.
  D1s stage2 결과와 비교표 생성.

실행:
  python rd_e1_abc_stage2_eval.py --run

출력:
  experiments/rd_e1_abc_stage2_eval_v1/eval/
    rd_e1_stage2_eval_per_exp.json    (실험별 전체 결과)
    rd_e1_stage2_eval_comparison.csv  (P5 top-k 비교표)
    rd_e1_stage2_eval_best_summary.md (최강 실험 요약)
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

# =============================================================================
# 경로 상수
# =============================================================================

PROJECT_ROOT    = Path("/home/jinhy/project/lung-ct-anomaly")
EVAL_ROOT       = PROJECT_ROOT / "experiments/rd_e1_abc_stage2_eval_v1"
SHARDS_BASE     = EVAL_ROOT
STAGE2_MANIFEST = (
    PROJECT_ROOT
    / "experiments/stage2_strict_ztrack_rd4ad_scoring_preflight_v1"
    / "manifests/stage2_rd4ad_scoring_manifest_minrun2.csv"
)
OUTPUT_DIR = EVAL_ROOT / "eval"

EXPS        = ["A", "B", "C", "C2", "A2", "E1", "E2", "E2z"]
SHARD_COUNT = 8
TOPK_LIST   = [1, 3, 5, 10, 20, 50]

# D1s stage2 기준값 — stage2_strict_ztrack_rd4ad_scoring_preflight_v1 원본 데이터로 재계산 (2026-06-11)
# 이전 하드코딩 P5 top10=0.9346은 오류 (과대계상), 정정값=0.9020
D1S_STAGE2 = {
    "RAW": {1: 0.2680, 3: 0.4314, 5: 0.4837, 10: 0.5621, 20: 0.6928, 50: 0.8301},
    "P1":  {1: 0.3856, 3: 0.5490, 5: 0.5817, 10: 0.6732, 20: 0.7451, 50: 0.9020},
    "P5":  {1: 0.5425, 3: 0.7124, 5: 0.7908, 10: 0.9020, 20: 0.9673, 50: 1.0000},
    "P9":  {1: 0.6209, 3: 0.7582, 5: 0.7974, 10: 0.8627, 20: 0.9412, 50: 0.9804},
    "PD":  {1: 0.6275, 3: 0.7647, 5: 0.8235, 10: 0.8824, 20: 0.9477, 50: 0.9935},
}

# =============================================================================
# 스코어 방식 정의 (후보 레벨 → track top3 mean으로 집계)
# =============================================================================

SCORE_METHODS = ["RAW", "P1", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "PA", "PB", "PC", "PD"]

SCORE_DESCS = {
    "RAW": "rd4ad_raw top3",
    "P1":  "raw × roi_ratio top3",
    "P3":  "P1 × log(track_len)",
    "P4":  "P1 × sqrt(track_len)",
    "P5":  "P1 × track_len/3",
    "P6":  "P1 × hu_mean",
    "P7":  "P1 × hu_mean × len/3",
    "P8":  "P1 × hu_mean × log(len)",
    "P9":  "P1 × sqrt(hu_mean) × len/3",
    "PA":  "hu_mean only",
    "PB":  "hu_mean × len/3",
    "PC":  "hu_mean × log(len)",
    "PD":  "sqrt(hu_mean) × len/3",
}


def compute_cand_scores(row: dict) -> dict:
    """후보 1개에 대해 각 스코어 방식의 값 계산."""
    raw = float(row["rd4ad_ztrack_score_raw"])
    p1  = float(row["P1_times_roi"]) if row["P1_times_roi"] != "" else 0.0
    tl  = float(row["track_len"])    if row["track_len"]    != "" else 1.0
    hu  = float(row["crop_hu_mean"]) if row["crop_hu_mean"] != "" else 0.0

    if not math.isfinite(p1):  p1 = 0.0
    if not math.isfinite(hu):  hu = 0.0
    if tl <= 0:                tl = 1.0

    return {
        "RAW": raw,
        "P1":  p1,
        "P3":  p1 * math.log(max(tl, 1)),
        "P4":  p1 * math.sqrt(tl),
        "P5":  p1 * (tl / 3.0),
        "P6":  p1 * hu,
        "P7":  p1 * hu * (tl / 3.0),
        "P8":  p1 * hu * math.log(max(tl, 1)),
        "P9":  p1 * math.sqrt(max(hu, 0.0)) * (tl / 3.0),
        "PA":  hu,
        "PB":  hu * (tl / 3.0),
        "PC":  hu * math.log(max(tl, 1)),
        "PD":  math.sqrt(max(hu, 0.0)) * (tl / 3.0),
    }


# =============================================================================
# 데이터 로드
# =============================================================================

def load_full_positive_patients() -> set:
    """stage2 manifest에서 전체 positive patient_id 집합 반환 (denominator)."""
    pos = set()
    with open(str(STAGE2_MANIFEST), encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("label", "") == "1" and row.get("patient_id", ""):
                pos.add(row["patient_id"])
    return pos


def load_exp_rows(exp_id: str) -> list:
    """실험의 8개 shard CSV를 합산하여 row list 반환."""
    rows = []
    for sid in range(SHARD_COUNT):
        csv_path = SHARDS_BASE / exp_id / "shards" / f"shard_{sid}" / f"stage2_rd4ad_scores_shard_{sid}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"shard CSV 없음: {csv_path}")
        with open(str(csv_path), encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
    return rows


# =============================================================================
# track-level 집계 + patient hit rate
# =============================================================================

def evaluate_exp(rows: list, positive_patients: set) -> dict:
    """
    rows: 128,827개 후보.
    각 스코어 방식에 대해:
      - track_label = 1 if any candidate in track has label==1, else 0
      - track_score = top3 mean of candidate scores within (patient_id, track_id)
      - 환자별 top-k tracks 중 label=1 track이 있으면 hit
      - hit rate = hit_patients / len(positive_patients)
    """
    # 후보 레벨 스코어 + track 집계
    # track_key = (patient_id, track_id)
    track_cands: dict = {}
    for row in rows:
        try:
            cand_scores = compute_cand_scores(row)
        except Exception:
            cand_scores = {m: 0.0 for m in SCORE_METHODS}
        pid = row["patient_id"]
        tid = row.get("track_id", "")
        lbl = row.get("label", "0")
        key = (pid, tid)
        if key not in track_cands:
            track_cands[key] = {
                "patient_id":  pid,
                "is_positive": False,        # track에 label=1 후보가 있으면 True
                "scores":      {m: [] for m in SCORE_METHODS},
            }
        if lbl == "1":
            track_cands[key]["is_positive"] = True
        for m in SCORE_METHODS:
            track_cands[key]["scores"][m].append(cand_scores[m])

    # track 레벨: top3 mean score + is_positive
    # pat_tracks[patient_id] = list of {"score_M": float, "is_positive": bool}
    pat_tracks: dict = {}
    for (pid, tid), td in track_cands.items():
        entry = {"is_positive": td["is_positive"]}
        for m in SCORE_METHODS:
            vals = sorted(td["scores"][m], reverse=True)
            top3 = vals[:3]
            entry[m] = sum(top3) / len(top3) if top3 else 0.0
        if pid not in pat_tracks:
            pat_tracks[pid] = []
        pat_tracks[pid].append(entry)

    # 환자별 top-k tracks hit rate
    results  = {}
    total_pos = len(positive_patients)

    for method in SCORE_METHODS:
        hit_count = {k: 0 for k in TOPK_LIST}

        for pid in positive_patients:
            tracks = pat_tracks.get(pid, [])
            if not tracks:
                continue  # 이 환자 후보 없음 → miss
            ranked = sorted(tracks, key=lambda t: t[method], reverse=True)
            for k in TOPK_LIST:
                top_k = ranked[:k]
                if any(t["is_positive"] for t in top_k):
                    hit_count[k] += 1

        topk_hit = {k: round(hit_count[k] / max(1, total_pos), 4) for k in TOPK_LIST}
        scored_pos = sum(1 for pid in positive_patients if pid in pat_tracks)
        results[method] = {
            "topk_hit_rate":        topk_hit,
            "total_positive_denom": total_pos,
            "scored_positive":      scored_pos,
        }

    return results


# =============================================================================
# 비교표 생성
# =============================================================================

def build_comparison_csv(all_results: dict, output_path: Path) -> None:
    rows = []
    for method in SCORE_METHODS:
        for k in TOPK_LIST:
            row = {"method": method, "desc": SCORE_DESCS[method], "topk": k}
            for exp_id in EXPS:
                val = all_results.get(exp_id, {}).get(method, {}).get("topk_hit_rate", {}).get(k)
                row[exp_id] = f"{val:.4f}" if val is not None else ""
            # D1s reference
            d1s_val = D1S_STAGE2.get(method, {}).get(k)
            row["D1s_stage2"] = f"{d1s_val:.4f}" if d1s_val is not None else ""
            rows.append(row)

    fieldnames = ["method", "desc", "topk"] + EXPS + ["D1s_stage2"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  saved: {output_path}")


def build_summary_md(all_results: dict, output_path: Path) -> None:
    lines = ["# RD-E1 Stage2 Holdout Evaluation Summary\n"]
    lines.append("## P5 (P1 × track_len/3) Patient Hit Rate — top10 / top20 / top50\n")
    lines.append(f"| Exp | Description | top10 | top20 | top50 | vs D1s top20 |")
    lines.append(f"|-----|-------------|-------|-------|-------|-------------|")

    d1s_top20_p5 = D1S_STAGE2["P5"][20]
    for exp_id in EXPS + ["D1s"]:
        if exp_id == "D1s":
            t10 = "-"
            t20 = f"{D1S_STAGE2['P5'][20]:.4f}"
            t50 = f"{D1S_STAGE2['P5'][50]:.4f}"
            delta = "baseline"
            desc = "medi3ch, [-160,240], ResNet18"
        else:
            r = all_results.get(exp_id, {}).get("P5", {}).get("topk_hit_rate", {})
            t10 = f"{r.get(10, 0):.4f}" if r.get(10) is not None else "-"
            t20 = f"{r.get(20, 0):.4f}" if r.get(20) is not None else "-"
            t50 = f"{r.get(50, 0):.4f}" if r.get(50) is not None else "-"
            v20 = r.get(20, 0) or 0
            delta_v = v20 - (d1s_top20_p5 or 0)
            delta = f"{delta_v:+.4f}"
            exp_descs = {
                "A":  "lung3ch [-1000,600] ResNet18",
                "B":  "medi_mip3ch [-160,240] ResNet18",
                "C":  "lung_mip3ch [-1000,600] ResNet18",
                "C2": "lung_mip3ch+ROImask [-1000,600] ResNet18",
                "A2": "lung3ch+per-ch-ROImask [-1000,600] ResNet18",
                "E1": "lung_mip3ch [-1000,600] EfficientNet-B0",
                "E2": "lung3ch [-1000,600] EfficientNet-B0",
                "E2z": "lung3ch [-1000,600] EfficientNet-B0 +z_pct",
                "D1s": "medi3ch [-160,240] ResNet18 (베이스라인)",
            }
            desc = exp_descs.get(exp_id, "")
        lines.append(f"| {exp_id} | {desc} | {t10} | {t20} | {t50} | {delta} |")

    lines.append("\n## Best Method per Experiment (top20)\n")
    lines.append("| Exp | Best method | top20 hit rate |")
    lines.append("|-----|-------------|----------------|")
    for exp_id in EXPS:
        best_method = None
        best_val    = -1.0
        for method in SCORE_METHODS:
            val = all_results.get(exp_id, {}).get(method, {}).get("topk_hit_rate", {}).get(20, 0) or 0
            if val > best_val:
                best_val    = val
                best_method = method
        lines.append(f"| {exp_id} | {best_method} | {best_val:.4f} |")

    lines.append("\n## D1s Stage2 Reference\n")
    lines.append("| method | top10 | top20 | top50 |")
    lines.append("|--------|-------|-------|-------|")
    for m in ["P1", "P5", "P9", "PD"]:
        r = D1S_STAGE2.get(m, {})
        t10 = f"{r[10]:.4f}" if r.get(10) else "-"
        t20 = f"{r[20]:.4f}" if r.get(20) else "-"
        t50 = f"{r[50]:.4f}" if r.get(50) else "-"
        lines.append(f"| {m} | {t10} | {t20} | {t50} |")

    lines.append(f"\n*denominator = 154 full positive patients (stage2 manifest)*\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  saved: {output_path}")


# =============================================================================
# main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--exp-ids", nargs="+", choices=EXPS, default=EXPS,
                        help="평가할 실험 (기본: 전체)")
    args = parser.parse_args()

    t0_total = time.perf_counter()

    print("=" * 70)
    print("[RD-E1 STAGE2 EVAL] A/B/C/C2/A2/E1/E2 stage2 holdout evaluation")
    print("=" * 70)

    print("\n[1] denominator 로드 (stage2 manifest positive patients)")
    positive_patients = load_full_positive_patients()
    print(f"  full positive patients: {len(positive_patients)}")

    all_results = {}

    for exp_id in args.exp_ids:
        t0 = time.perf_counter()
        print(f"\n[2] {exp_id} 로드 및 평가")
        rows = load_exp_rows(exp_id)
        print(f"  loaded {len(rows):,} rows")
        results = evaluate_exp(rows, positive_patients)
        elapsed = time.perf_counter() - t0
        all_results[exp_id] = results

        # P5 top10/20/50 빠른 확인
        p5 = results.get("P5", {}).get("topk_hit_rate", {})
        print(f"  P5: top10={p5.get(10):.4f}  top20={p5.get(20):.4f}  top50={p5.get(50):.4f}  ({elapsed:.0f}s)")

    # 출력 저장
    print("\n[3] 결과 저장")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / "rd_e1_stage2_eval_per_exp.json"
    with open(str(json_path), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"  saved: {json_path}")

    build_comparison_csv(all_results, OUTPUT_DIR / "rd_e1_stage2_eval_comparison.csv")
    build_summary_md(all_results, OUTPUT_DIR / "rd_e1_stage2_eval_best_summary.md")

    total_elapsed = time.perf_counter() - t0_total

    # 터미널 요약 출력
    print("\n" + "=" * 70)
    print("=== P5 top20 hit rate 비교 ===")
    print(f"  {'Exp':<6} {'P5-top20':>10} {'P5-top10':>10} {'PD-top20':>10} {'vs D1s-P5':>10}")
    print(f"  {'D1s':<6} {'0.9673':>10} {'-':>10} {'0.9477':>10} {'baseline':>10}")
    for exp_id in args.exp_ids:
        p5  = all_results[exp_id]["P5"]["topk_hit_rate"]
        pd_ = all_results[exp_id]["PD"]["topk_hit_rate"]
        v20  = p5.get(20, 0)
        v10  = p5.get(10, 0)
        pd20 = pd_.get(20, 0)
        delta = (v20 or 0) - 0.9673
        print(f"  {exp_id:<6} {v20:>10.4f} {v10:>10.4f} {pd20:>10.4f} {delta:>+10.4f}")
    print(f"\n  총 소요: {total_elapsed:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
