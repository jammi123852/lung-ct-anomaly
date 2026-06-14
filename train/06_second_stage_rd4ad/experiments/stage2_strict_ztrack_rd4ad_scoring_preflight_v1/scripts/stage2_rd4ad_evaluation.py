"""
stage2_rd4ad_evaluation.py

목적:
  8 shard scoring 결과를 merge하고 track-level aggregation 후
  patient hit rate (top1/3/5/10/20/30/50) 평가.

고정 조건 (stage1_dev 확정, 변경 금지):
  - primary_candidate_score = P1_times_roi
  - primary_track_score = P1_track_top3_mean
  - auxiliary_track_score = P1_track_top2_mean
  - method tuning 없음 / threshold 변경 없음

가드레일:
  - model forward / checkpoint load / scoring 없음
  - 기존 artifact 수정 없음
  - stage2 결과를 보고 stage1 방법 변경 금지
"""

import csv
import json
import math
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT    = Path("/home/jinhy/project/lung-ct-anomaly")
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments/stage2_strict_ztrack_rd4ad_scoring_preflight_v1"

SHARDS_DIR      = EXPERIMENT_ROOT / "shards"
ZTRACK_MANIFEST = (
    PROJECT_ROOT
    / "experiments/stage2_strict_ztrack_schema_survival_preflight_v1"
    / "manifests/stage2_ztrack_manifest_minrun2.csv"
)

OUT_MERGED   = EXPERIMENT_ROOT / "manifests/stage2_rd4ad_scores_full_merged.csv"
OUT_REPORT   = EXPERIMENT_ROOT / "reports/stage2_rd4ad_evaluation_report.md"
OUT_SUMMARY  = EXPERIMENT_ROOT / "reports/stage2_rd4ad_evaluation_summary.json"
OUT_DONE     = EXPERIMENT_ROOT / "DONE_eval.json"

SHARD_COUNT = 8
TOP_KS      = [1, 3, 5, 10, 20, 30, 50]

GUARDRAILS = {
    "stage2_holdout_used_for_method_tuning": False,
    "model_forward_executed":               False,
    "checkpoint_loaded":                    False,
    "scoring_executed":                     False,
    "existing_artifact_modified":           False,
    "label_used_for_evaluation_only":       True,
    "label_used_as_selector":               False,
    "method_changed_based_on_stage2":       False,
    "primary_candidate_score":              "P1_times_roi",
    "primary_track_score":                  "P1_track_top3_mean",
    "auxiliary_track_score":                "P1_track_top2_mean",
}

# stage1_dev baseline (변경 금지)
STAGE1_BASELINE = {
    "patch_baseline_rd_d1s":            {1:0.2697, 3:0.4079, 5:0.4803, 10:0.5855, 20:0.6184, 30:None, 50:0.7105},
    "P1_track_top3_mean":               {1:0.4671, 3:0.5789, 5:0.6513, 10:0.7237, 20:0.8224, 30:None, 50:0.9079},
    "P5_len_norm_track_top3_mean":      {1:0.6579, 3:0.8355, 5:0.8553, 10:0.9145, 20:0.9474, 30:None, 50:0.9737},
    "PD_sqrthu_len_track_top3_mean":    {1:0.6974, 3:0.8158, 5:0.8684, 10:0.8882, 20:0.9342, 30:0.9539, 50:0.9737},
}


def patient_hit_rate(track_df, score_col, k, positive_patients):
    """track DataFrame (list of dict)에서 top-k patient hit rate 계산."""
    sorted_tracks = sorted(track_df, key=lambda r: r[score_col], reverse=True)

    # 환자별 상위 k 트랙 선택
    per_patient_count = defaultdict(int)
    hit_patients = set()
    for r in sorted_tracks:
        pid = r["patient_id"]
        if per_patient_count[pid] < k:
            per_patient_count[pid] += 1
            if r["has_positive"]:
                hit_patients.add(pid)

    n_pos = len(positive_patients)
    return round(len(hit_patients) / n_pos, 4) if n_pos > 0 else 0.0


def main():
    # ── [1] shard merge ──────────────────────────────────────────────────────
    print("[1] shard merge...")
    all_rows = []
    for sid in range(SHARD_COUNT):
        p = SHARDS_DIR / f"shard_{sid}" / f"stage2_rd4ad_scores_shard_{sid}.csv"
        if not p.exists():
            print(f"  [ERROR] missing: {p}", file=sys.stderr)
            sys.exit(1)
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        all_rows.extend(rows)
        print(f"  shard {sid}: {len(rows):,}행")

    print(f"  합계: {len(all_rows):,}행")
    if len(all_rows) != 128827:
        print(f"  [WARN] 기대 128,827 != 실제 {len(all_rows):,}")

    # ── [2] ztrack manifest join (has_positive) ───────────────────────────────
    print("[2] ztrack manifest join...")
    ztrack_info = {}
    with open(ZTRACK_MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ztrack_info[r["track_id"]] = {
                "track_len":   int(r.get("track_len", 2)),
                "n_positive":  int(r.get("n_positive", 0)),
                "has_positive": int(r.get("n_positive", 0)) > 0,
            }
    print(f"  ztrack entries: {len(ztrack_info):,}")

    # ── [3] 점수 계산 ─────────────────────────────────────────────────────────
    print("[3] 점수 계산...")
    eps = 1e-9
    for r in all_rows:
        try:
            p1  = float(r["P1_times_roi"]) if r.get("P1_times_roi") else 0.0
            tl  = int(r.get("track_len", 2) or 2)
            hu_raw = r.get("crop_hu_mean", "")
            hu  = float(hu_raw) if hu_raw else 0.0

            r["_p1"]  = p1
            r["_tl"]  = tl
            r["_hu"]  = hu

            r["P1_len_norm"]   = p1 * (tl / 3.0)
            r["P1_sqrt_len"]   = p1 * math.sqrt(tl)
            r["P1_log_len"]    = p1 * math.log(max(tl, 1))
            r["PD_sqrthu_len"] = math.sqrt(max(hu, 0)) * (tl / 3.0)
            r["PB_hu_len"]     = (hu + eps) * (tl / 3.0)

            # has_positive from ztrack
            tid = r.get("track_id", "")
            zi  = ztrack_info.get(tid, {})
            r["has_positive"] = zi.get("has_positive", False)
        except Exception as e:
            r["P1_len_norm"] = r["P1_sqrt_len"] = r["P1_log_len"] = 0.0
            r["PD_sqrthu_len"] = r["PB_hu_len"] = 0.0
            r["has_positive"] = False

    # ── [4] track-level aggregation ───────────────────────────────────────────
    print("[4] track-level aggregation...")

    score_cols = [
        "P1_times_roi",
        "P1_len_norm",
        "P1_sqrt_len",
        "P1_log_len",
        "PD_sqrthu_len",
        "PB_hu_len",
    ]

    def top_n_mean(vals, n):
        s = sorted(vals, reverse=True)
        return sum(s[:n]) / max(len(s[:n]), 1)

    # group by track_id
    track_groups = defaultdict(list)
    for r in all_rows:
        track_groups[r["track_id"]].append(r)

    track_df = []
    for tid, members in track_groups.items():
        meta_r = members[0]
        entry = {
            "track_id":   tid,
            "patient_id": meta_r["patient_id"],
            "has_positive": meta_r["has_positive"],
            "track_len":  meta_r.get("track_len", 2),
            "n_members":  len(members),
        }
        for sc in score_cols:
            vals = []
            for r in members:
                try:
                    v = float(r[sc])
                    if math.isfinite(v):
                        vals.append(v)
                except Exception:
                    pass
            if not vals:
                vals = [0.0]
            entry[f"{sc}_track_max"]       = max(vals)
            entry[f"{sc}_track_top3_mean"] = top_n_mean(vals, 3)
            entry[f"{sc}_track_top2_mean"] = top_n_mean(vals, 2)
        track_df.append(entry)

    print(f"  tracks: {len(track_df):,}")

    # ── [5] patient hit rate 계산 ─────────────────────────────────────────────
    print("[5] patient hit rate 계산...")
    positive_patients = set(
        r["patient_id"] for r in all_rows if r.get("label", "").strip() == "1"
    )
    print(f"  positive patients: {len(positive_patients)}")

    agg_variants = ["track_max", "track_top3_mean", "track_top2_mean"]
    eval_cols = []
    for sc in score_cols:
        for agg in agg_variants:
            eval_cols.append(f"{sc}_{agg}")

    rows_result = []
    for col in eval_cols:
        row = {"score_col": col}
        for k in TOP_KS:
            row[f"top{k}"] = patient_hit_rate(track_df, col, k, positive_patients)
        rows_result.append(row)

    # ── [6] 출력 ─────────────────────────────────────────────────────────────
    print("\n[6] 결과 출력")

    header = f"{'score_col':52s}" + "".join(f"  top{k:>2}" for k in TOP_KS)
    sep    = "-" * (len(header) + 4)
    print(f"\n{header}")
    print(sep)

    # stage1 baseline
    print("  [stage1 baseline]")
    for bname, bvals in STAGE1_BASELINE.items():
        line = f"  {bname:50s}"
        for k in TOP_KS:
            v = bvals.get(k)
            line += f"  {'N/A ':6s}" if v is None else f"  {v:.4f}"
        print(line)
    print()

    # stage2 결과
    print("  [stage2 holdout eval-only]")
    for row in rows_result:
        col  = row["score_col"]
        line = f"  {col:50s}"
        for k in TOP_KS:
            line += f"  {row[f'top{k}']:.4f}"
        print(line)

    # primary 결과 강조
    primary_col = "P1_times_roi_track_top3_mean"
    primary_row = next((r for r in rows_result if r["score_col"] == primary_col), None)
    print(f"\n★ PRIMARY (stage1 확정): {primary_col}")
    if primary_row:
        for k in TOP_KS:
            s1 = STAGE1_BASELINE["P1_track_top3_mean"].get(k)
            s2 = primary_row[f"top{k}"]
            delta = f"({s2 - s1:+.4f})" if s1 is not None else "(N/A)"
            print(f"  top{k:>2}: stage2={s2:.4f}  stage1={s1 if s1 else 'N/A':>6}  Δ={delta}")

    # ── [7] merged CSV 저장 ───────────────────────────────────────────────────
    print("\n[7] merged CSV 저장...")
    OUT_MERGED.parent.mkdir(parents=True, exist_ok=True)
    if all_rows:
        extra_cols = ["P1_len_norm", "P1_sqrt_len", "P1_log_len", "PD_sqrthu_len", "PB_hu_len", "has_positive"]
        base_fields = list(all_rows[0].keys())
        # _p1/_tl/_hu 제거
        base_fields = [f for f in base_fields if not f.startswith("_")]
        all_fields  = base_fields + [c for c in extra_cols if c not in base_fields]
        with open(OUT_MERGED, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)
        print(f"  → {OUT_MERGED} ({len(all_rows):,}행)")

    # ── [8] summary JSON ──────────────────────────────────────────────────────
    summary = {
        "verdict":                "DONE",
        "n_candidates_merged":    len(all_rows),
        "n_tracks":               len(track_df),
        "n_positive_patients":    len(positive_patients),
        "complete_miss_known":    ["LUNG1-415"],
        "stage1_baseline":        STAGE1_BASELINE,
        "stage2_results":         {r["score_col"]: {f"top{k}": r[f"top{k}"] for k in TOP_KS} for r in rows_result},
        "primary_score":          primary_col,
        "primary_stage2":         {f"top{k}": primary_row[f"top{k}"] for k in TOP_KS} if primary_row else {},
        "guardrails":             GUARDRAILS,
    }
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  → {OUT_SUMMARY}")

    # ── [9] report MD ─────────────────────────────────────────────────────────
    lines = [
        "# Stage2 Holdout RD4AD Evaluation — Eval-Only\n\n",
        "## 조건\n",
        "- stage2_holdout eval-only (method tuning 금지)\n",
        "- primary_candidate_score: P1_times_roi\n",
        "- primary_track_score: P1_track_top3_mean\n",
        "- complete_miss: LUNG1-415 (z-track min_run=2 구조적 탈락, 변경 금지)\n\n",
        "## patient_hit_rate 결과\n\n",
        "| score_col | top1 | top3 | top5 | top10 | top20 | top30 | top50 |\n",
        "|-----------|------|------|------|-------|-------|-------|-------|\n",
    ]
    for bname, bvals in STAGE1_BASELINE.items():
        vals_str = " | ".join(
            f"{bvals[k]:.4f}" if bvals.get(k) is not None else "N/A"
            for k in TOP_KS
        )
        lines.append(f"| **[S1] {bname}** | {vals_str} |\n")
    for row in rows_result:
        vals_str = " | ".join(f"{row[f'top{k}']:.4f}" for k in TOP_KS)
        lines.append(f"| {row['score_col']} | {vals_str} |\n")

    lines += [
        "\n## PRIMARY 결과 상세\n\n",
        f"primary: `{primary_col}`\n\n",
        "| top-k | stage1 | stage2 | Δ |\n",
        "|-------|--------|--------|---|\n",
    ]
    if primary_row:
        for k in TOP_KS:
            s1 = STAGE1_BASELINE["P1_track_top3_mean"].get(k)
            s2 = primary_row[f"top{k}"]
            delta = f"{s2 - s1:+.4f}" if s1 is not None else "N/A"
            lines.append(f"| top{k} | {s1 if s1 else 'N/A'} | {s2:.4f} | {delta} |\n")

    lines += ["\n## 가드레일\n\n"]
    for k, v in GUARDRAILS.items():
        lines.append(f"- {k}: {v}\n")

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"  → {OUT_REPORT}")

    # DONE
    with open(OUT_DONE, "w", encoding="utf-8") as f:
        json.dump({"verdict": "DONE", "guardrails": GUARDRAILS}, f, indent=2)
    print(f"  → {OUT_DONE}")

    print("\n[완료]")


if __name__ == "__main__":
    main()
