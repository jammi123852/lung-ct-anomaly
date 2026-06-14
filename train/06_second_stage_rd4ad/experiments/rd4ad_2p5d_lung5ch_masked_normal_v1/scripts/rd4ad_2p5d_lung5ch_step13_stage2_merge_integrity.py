"""
Step 13: Stage2 Fixed Scoring Merge + Integrity Check
rd4ad_2p5d_lung5ch_masked_normal_v1

대상: step12 shard_000~007_scores.csv → 단일 merged CSV
      merge/integrity check only — label 평가 금지, metric 금지

bare run → exit 2
dry-run  → 계획 출력
--run-merge → actual merge + integrity check
"""

import sys
import json
import csv
import math
import time
import argparse
from pathlib import Path
from datetime import date

# ── bare run 차단 ──────────────────────────────────────────────────────────────
if len(sys.argv) == 1:
    print("[BLOCKED] bare run 금지. --dry-run 또는 --run-merge를 사용하세요.", file=sys.stderr)
    sys.exit(2)

# ── 경로 상수 ─────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]

PLAN_LOCK_JSON   = ROOT / "docs" / "FINAL_PLAN_LOCK.json"
DONE_STEP12_JSON = ROOT / "DONE_STEP12_STAGE2_FIXED_SCORING.json"
STEP11_MANIFEST  = ROOT / "manifests" / "step11_stage2_scoring_plan_manifest.csv"

SHARD_DIR = ROOT / "scoring" / "step12_stage2_fixed_v1"
OUT_DIR   = ROOT / "scoring" / "step13_stage2_fixed_merged_v1"

MERGED_CSV    = OUT_DIR / "stage2_lung5ch_fixed_scores_merged.csv"
INTEGRITY_CSV = ROOT / "manifests" / "step13_stage2_merge_integrity_summary.csv"
SCORE_DIST_CSV = ROOT / "manifests" / "step13_stage2_score_distribution.csv"
TRACK_CSV     = ROOT / "manifests" / "step13_stage2_track_level_scores.csv"
PATIENT_CSV   = ROOT / "manifests" / "step13_stage2_patient_summary.csv"
REPORT_MD     = ROOT / "reports"   / "step13_stage2_merge_integrity_report.md"
SUMMARY_JSON  = ROOT / "reports"   / "step13_stage2_merge_integrity_summary.json"
ERRORS_CSV    = ROOT / "logs"      / "step13_stage2_merge_integrity_errors.csv"
DONE_OUT      = ROOT / "DONE_STEP13_STAGE2_MERGE_INTEGRITY.json"

N_SHARDS             = 8
EXPECTED_TOTAL_ROWS  = 127947
PRIMARY_CAND_SCORE   = "rd4ad_lung5ch_score_raw"
PRIMARY_TRACK_SCORE  = "raw_track_top3_mean"
SCORE_COLS           = ["rd4ad_lung5ch_score_raw", "score_layer1", "score_layer2", "score_layer3"]
DIST_COLS            = ["rd4ad_lung5ch_score_raw", "score_layer1", "score_layer2", "score_layer3",
                        "roi_ratio", "mask_area_center", "mask_area_5ch_mean"]
COMPOSITE_KEY_COLS   = ["patient_id", "safe_id", "local_z", "crop_y0", "crop_x0", "crop_y1", "crop_x1"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run-merge", action="store_true")
    ap.add_argument("--confirm-step12-pass", action="store_true")
    ap.add_argument("--confirm-merge-only", action="store_true")
    ap.add_argument("--confirm-no-label-eval", action="store_true")
    return ap.parse_args()


def print_dry_run():
    print()
    print("=" * 64)
    print("Step 13 Stage2 Merge + Integrity Check — DRY-RUN PLAN")
    print("=" * 64)
    print()
    print("[입력]")
    for i in range(N_SHARDS):
        print(f"  {SHARD_DIR}/shard_{i:03d}_scores.csv")
    print(f"  {STEP11_MANIFEST}  (planned candidate 비교용)")
    print()
    print("[출력]")
    for fp in [MERGED_CSV, INTEGRITY_CSV, SCORE_DIST_CSV, TRACK_CSV,
               PATIENT_CSV, REPORT_MD, SUMMARY_JSON, ERRORS_CSV, DONE_OUT]:
        print(f"  {fp}")
    print()
    print("[integrity 항목]")
    for item in [
        "shard 파일 존재 + verdict PASS",
        "merged rows = 127,947",
        "failed rows = 0",
        "NaN/Inf = 0",
        "duplicate rows = 0  (row_id 기준)",
        "missing planned rows = 0  (step11 manifest 비교)",
        "crop 96x96 sanity",
        "score distribution (7 컬럼, 16 통계)",
        "track-level score  (raw_track_top3_mean primary)",
        "patient summary  (label 사용 금지)",
    ]:
        print(f"  {item}")
    print()
    print("[금지]")
    print("  stage2 label 사용 / metric 계산 / threshold 튜닝 / score family 변경")
    print("  P1/P2 primary 재지정 / candidate 삭제 / track score 공식 변경")
    print()
    rel = Path("experiments/rd4ad_2p5d_lung5ch_masked_normal_v1/scripts") / Path(__file__).name
    print("[실행 명령]")
    print(f"  python {rel} \\")
    print(f"    --run-merge --confirm-step12-pass --confirm-merge-only --confirm-no-label-eval")
    print()
    print("DRY-RUN 완료.")


def check_guards():
    errors = []
    if not DONE_STEP12_JSON.exists():
        errors.append("DONE_STEP12_STAGE2_FIXED_SCORING.json 없음")
    else:
        with open(DONE_STEP12_JSON) as f:
            d = json.load(f)
        if d.get("verdict") != "PASS_STEP12_STAGE2_FIXED_SCORING":
            errors.append(f"Step12 verdict={d.get('verdict')}")
        if d.get("scored_rows") != EXPECTED_TOTAL_ROWS:
            errors.append(f"Step12 scored_rows={d.get('scored_rows')} != {EXPECTED_TOTAL_ROWS}")
        if d.get("failed_rows", 1) != 0:
            errors.append(f"Step12 failed_rows={d.get('failed_rows')} != 0")

    for i in range(N_SHARDS):
        sf = SHARD_DIR / f"shard_{i:03d}_summary.json"
        if not sf.exists():
            errors.append(f"shard {i} summary 없음")
        else:
            with open(sf) as f:
                d = json.load(f)
            if not d.get("verdict", "").startswith("PASS"):
                errors.append(f"shard {i} verdict={d.get('verdict')}")
        cf = SHARD_DIR / f"shard_{i:03d}_scores.csv"
        if not cf.exists():
            errors.append(f"shard {i} CSV 없음")

    if not STEP11_MANIFEST.exists():
        errors.append(f"step11 manifest 없음: {STEP11_MANIFEST}")

    return errors


def topk_mean(values, k):
    """sorted descending, mean of top-k."""
    if not values:
        return float("nan")
    top = sorted(values, reverse=True)[:k]
    return sum(top) / len(top)


def compute_percentile(values, pct):
    """단순 linear interpolation percentile."""
    if not values:
        return float("nan")
    sv = sorted(values)
    n = len(sv)
    idx = pct / 100.0 * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sv[lo] * (1 - frac) + sv[hi] * frac


def score_stats(values):
    n = len(values)
    if n == 0:
        return {k: float("nan") for k in
                ["count","mean","std","min","p1","p5","p10","p25","p50","p75","p90","p95","p99","max","nan_count","inf_count"]}
    nan_c = sum(1 for v in values if math.isnan(v))
    inf_c = sum(1 for v in values if math.isinf(v))
    finite = [v for v in values if not (math.isnan(v) or math.isinf(v))]
    if not finite:
        return {"count": n, "mean": float("nan"), "std": float("nan"), "min": float("nan"),
                "p1": float("nan"), "p5": float("nan"), "p10": float("nan"), "p25": float("nan"),
                "p50": float("nan"), "p75": float("nan"), "p90": float("nan"), "p95": float("nan"),
                "p99": float("nan"), "max": float("nan"),
                "nan_count": nan_c, "inf_count": inf_c}
    mean = sum(finite) / len(finite)
    var  = sum((x - mean) ** 2 for x in finite) / max(1, len(finite) - 1)
    std  = math.sqrt(var)
    return {
        "count"    : n,
        "mean"     : mean,
        "std"      : std,
        "min"      : min(finite),
        "p1"       : compute_percentile(finite, 1),
        "p5"       : compute_percentile(finite, 5),
        "p10"      : compute_percentile(finite, 10),
        "p25"      : compute_percentile(finite, 25),
        "p50"      : compute_percentile(finite, 50),
        "p75"      : compute_percentile(finite, 75),
        "p90"      : compute_percentile(finite, 90),
        "p95"      : compute_percentile(finite, 95),
        "p99"      : compute_percentile(finite, 99),
        "max"      : max(finite),
        "nan_count": nan_c,
        "inf_count": inf_c,
    }


def run_merge(args):
    import collections

    t_start = time.perf_counter()
    print()
    print("=" * 64)
    print("Step 13 Stage2 Merge + Integrity Check — ACTUAL RUN")
    print("=" * 64)

    # [0] guards
    print("\n[0] Guards 확인")
    guard_errors = check_guards()
    if guard_errors:
        print("  [BLOCKED] Guard 실패:")
        for e in guard_errors:
            print(f"    - {e}")
        sys.exit(1)
    print("  [PASS]")

    # 출력 디렉토리
    for d in [OUT_DIR, ROOT / "manifests", ROOT / "reports", ROOT / "logs"]:
        d.mkdir(parents=True, exist_ok=True)

    # [1] shard 확인 + merge
    print("\n[1] shard 파일 merge 시작")
    shard_row_counts = {}
    error_rows = []
    all_rows = []

    for i in range(N_SHARDS):
        cf = SHARD_DIR / f"shard_{i:03d}_scores.csv"
        with open(cf, newline="") as f:
            rows = list(csv.DictReader(f))
        shard_row_counts[i] = len(rows)
        all_rows.extend(rows)
        print(f"  shard {i:03d}: {len(rows):,}행")

    total_loaded = len(all_rows)
    print(f"  합계: {total_loaded:,}행")

    # [2] row count 확인
    print(f"\n[2] row count 확인")
    if total_loaded != EXPECTED_TOTAL_ROWS:
        print(f"  [BLOCKED] {total_loaded:,} != {EXPECTED_TOTAL_ROWS:,}", file=sys.stderr)
        sys.exit(1)
    print(f"  {total_loaded:,} == {EXPECTED_TOTAL_ROWS:,}  [OK]")

    # [3] failed/NaN/Inf 확인
    print("\n[3] failed/NaN/Inf 확인")
    failed_rows_list = [r for r in all_rows if r.get("status", "SCORED") != "SCORED"]
    nan_count  = sum(1 for r in all_rows if r.get("rd4ad_lung5ch_score_raw","") in ("nan","NaN",""))
    inf_count  = sum(1 for r in all_rows if r.get("rd4ad_lung5ch_score_raw","") in ("inf","Inf","-inf","-Inf"))

    # 더 정확한 float 확인
    nan_count_f = inf_count_f = 0
    for r in all_rows:
        v = r.get("rd4ad_lung5ch_score_raw", "")
        try:
            fv = float(v)
            if math.isnan(fv): nan_count_f += 1
            elif math.isinf(fv): inf_count_f += 1
        except (ValueError, TypeError):
            nan_count_f += 1
    nan_count = nan_count_f
    inf_count = inf_count_f

    print(f"  failed rows : {len(failed_rows_list)}")
    print(f"  NaN         : {nan_count}")
    print(f"  Inf         : {inf_count}")

    # [4] duplicate 확인
    print("\n[4] duplicate 확인 (row_id)")
    row_ids = [r.get("row_id", "") for r in all_rows]
    if all(rid != "" for rid in row_ids):
        dup_count = len(row_ids) - len(set(row_ids))
        print(f"  row_id 기준 duplicate: {dup_count}")
    else:
        # composite key fallback
        keys = [tuple(r.get(c, "") for c in COMPOSITE_KEY_COLS) for r in all_rows]
        dup_count = len(keys) - len(set(keys))
        print(f"  composite key 기준 duplicate: {dup_count}")

    # [5] missing candidate 확인 (step11 manifest vs merged)
    print("\n[5] missing candidate 확인 (step11 manifest)")
    planned_row_ids = set()
    with open(STEP11_MANIFEST, newline="") as f:
        for r in csv.DictReader(f):
            planned_row_ids.add(r["row_id"])

    merged_row_ids = set(r.get("row_id", "") for r in all_rows)
    missing_planned = planned_row_ids - merged_row_ids
    extra_merged    = merged_row_ids - planned_row_ids
    print(f"  planned:       {len(planned_row_ids):,}")
    print(f"  merged:        {len(merged_row_ids):,}")
    print(f"  missing:       {len(missing_planned)}")
    print(f"  extra:         {len(extra_merged)}")

    # [6] coordinate sanity
    print("\n[6] coordinate sanity")
    coord_errors = 0
    neg_coord = 0
    for r in all_rows:
        try:
            cy0 = int(r["crop_y0"]); cy1 = int(r["crop_y1"])
            cx0 = int(r["crop_x0"]); cx1 = int(r["crop_x1"])
            if cy1 - cy0 != 96 or cx1 - cx0 != 96:
                coord_errors += 1
            if cy0 < 0 or cx0 < 0:
                neg_coord += 1
        except (ValueError, KeyError):
            coord_errors += 1
    print(f"  crop size error (non-96x96): {coord_errors}")
    print(f"  negative coord: {neg_coord}")

    # [7] score distribution
    print("\n[7] score distribution 계산")
    dist_data = {}
    for col in DIST_COLS:
        vals = []
        for r in all_rows:
            v = r.get(col, "")
            try:
                vals.append(float(v))
            except (ValueError, TypeError):
                pass
        dist_data[col] = vals

    dist_rows = []
    for col in DIST_COLS:
        st = score_stats(dist_data[col])
        row = {"column": col}
        row.update({k: round(v, 8) if isinstance(v, float) and not math.isnan(v) else v
                    for k, v in st.items()})
        dist_rows.append(row)
        print(f"  {col}: mean={st['mean']:.4f} p50={st['p50']:.4f} p90={st['p90']:.4f} max={st['max']:.4f}")

    dist_fields = ["column","count","mean","std","min","p1","p5","p10","p25","p50","p75","p90","p95","p99","max","nan_count","inf_count"]
    with open(SCORE_DIST_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=dist_fields)
        w.writeheader()
        w.writerows(dist_rows)
    print(f"  [SAVED] {SCORE_DIST_CSV}")

    # [8] track-level score 생성
    print("\n[8] track-level score 생성")
    track_scores = collections.defaultdict(list)
    track_meta   = {}
    for r in all_rows:
        tid = r.get("track_id", "")
        try:
            raw = float(r["rd4ad_lung5ch_score_raw"])
        except (ValueError, TypeError):
            continue
        track_scores[tid].append(raw)
        if tid not in track_meta:
            track_meta[tid] = {
                "track_id"    : tid,
                "track_len"   : r.get("track_len", ""),
                "track_z_start": r.get("track_z_start", ""),
                "track_z_end" : r.get("track_z_end", ""),
                "patient_id"  : r.get("patient_id", ""),
                "safe_id"     : r.get("safe_id", ""),
            }

    track_rows = []
    for tid, scores in track_scores.items():
        meta = track_meta[tid]
        track_rows.append({
            "track_id"            : tid,
            "track_len"           : meta["track_len"],
            "track_z_start"       : meta["track_z_start"],
            "track_z_end"         : meta["track_z_end"],
            "patient_id"          : meta["patient_id"],
            "safe_id"             : meta["safe_id"],
            "raw_track_max"       : round(max(scores), 8),
            "raw_track_top2_mean" : round(topk_mean(scores, 2), 8),
            "raw_track_top3_mean" : round(topk_mean(scores, 3), 8),
            "raw_track_mean"      : round(sum(scores) / len(scores), 8),
            "n_candidates"        : len(scores),
        })

    track_fields = ["track_id","track_len","track_z_start","track_z_end","patient_id","safe_id",
                    "raw_track_max","raw_track_top2_mean","raw_track_top3_mean","raw_track_mean","n_candidates"]
    with open(TRACK_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=track_fields)
        w.writeheader()
        w.writerows(track_rows)

    n_tracks = len(track_rows)
    print(f"  unique tracks: {n_tracks:,}")
    print(f"  primary track score: {PRIMARY_TRACK_SCORE}")
    print(f"  [SAVED] {TRACK_CSV}")

    # [9] patient summary (label 사용 금지)
    print("\n[9] patient summary 생성 (label 없음)")
    pat_scores = collections.defaultdict(list)
    pat_tracks  = collections.defaultdict(set)
    pat_nearest = collections.defaultdict(list)
    pat_roi     = collections.defaultdict(list)
    pat_failed  = collections.defaultdict(int)
    pat_safe    = {}

    for r in all_rows:
        pid = r.get("patient_id", "")
        pat_safe[pid] = r.get("safe_id", "")
        status = r.get("status", "SCORED")
        if status != "SCORED":
            pat_failed[pid] += 1
            continue
        try:
            raw = float(r["rd4ad_lung5ch_score_raw"])
            pat_scores[pid].append(raw)
        except (ValueError, TypeError):
            pass
        pat_tracks[pid].add(r.get("track_id", ""))
        try:
            pat_nearest[pid].append(1 if r.get("nearest_repeat_used","False") == "True" else 0)
        except Exception:
            pass
        try:
            pat_roi[pid].append(float(r.get("roi_ratio", 0)))
        except (ValueError, TypeError):
            pass

    # track top score per patient
    track_top_by_pat = collections.defaultdict(float)
    for tr in track_rows:
        pid = tr["patient_id"]
        top3 = float(tr["raw_track_top3_mean"])
        if top3 > track_top_by_pat[pid]:
            track_top_by_pat[pid] = top3

    pat_rows = []
    for pid in sorted(set(r.get("patient_id","") for r in all_rows)):
        sc = pat_scores[pid]
        nr = pat_nearest[pid]
        roi = pat_roi[pid]
        pat_rows.append({
            "patient_id"              : pid,
            "safe_id"                 : pat_safe.get(pid, ""),
            "candidate_count"         : len(sc),
            "track_count"             : len(pat_tracks[pid]),
            "raw_score_max"           : round(max(sc), 8) if sc else "",
            "raw_score_p95"           : round(compute_percentile(sc, 95), 8) if sc else "",
            "raw_score_mean"          : round(sum(sc) / len(sc), 8) if sc else "",
            "top_track_raw_top3_mean" : round(track_top_by_pat[pid], 8),
            "nearest_repeat_ratio"    : round(sum(nr) / max(1, len(nr)), 4),
            "roi_ratio_mean"          : round(sum(roi) / max(1, len(roi)), 4) if roi else "",
            "failed_count"            : pat_failed.get(pid, 0),
        })

    pat_fields = ["patient_id","safe_id","candidate_count","track_count",
                  "raw_score_max","raw_score_p95","raw_score_mean",
                  "top_track_raw_top3_mean","nearest_repeat_ratio","roi_ratio_mean","failed_count"]
    with open(PATIENT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pat_fields)
        w.writeheader()
        w.writerows(pat_rows)
    n_patients = len(pat_rows)
    print(f"  unique patients: {n_patients}")
    print(f"  [SAVED] {PATIENT_CSV}")

    # [10] merged CSV 저장
    print("\n[10] merged CSV 저장")
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(MERGED_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)
        print(f"  [SAVED] {MERGED_CSV}  ({total_loaded:,} rows)")

    # [11] integrity summary CSV
    integrity_items = [
        ("shard_count", N_SHARDS, N_SHARDS, "PASS"),
        ("total_rows", total_loaded, EXPECTED_TOTAL_ROWS,
         "PASS" if total_loaded == EXPECTED_TOTAL_ROWS else "FAIL"),
        ("failed_rows", len(failed_rows_list), 0,
         "PASS" if not failed_rows_list else "FAIL"),
        ("nan_count", nan_count, 0, "PASS" if nan_count == 0 else "FAIL"),
        ("inf_count", inf_count, 0, "PASS" if inf_count == 0 else "FAIL"),
        ("duplicate_rows", dup_count, 0, "PASS" if dup_count == 0 else "BLOCKED"),
        ("missing_planned_rows", len(missing_planned), 0,
         "PASS" if not missing_planned else "BLOCKED"),
        ("extra_merged_rows", len(extra_merged), 0,
         "PASS" if not extra_merged else "WARN"),
        ("coord_errors", coord_errors, 0, "PASS" if coord_errors == 0 else "FAIL"),
        ("neg_coord", neg_coord, 0, "PASS" if neg_coord == 0 else "FAIL"),
        ("track_count", n_tracks, 19263, "PASS" if n_tracks > 0 else "FAIL"),
        ("patient_count", n_patients, 154, "PASS" if n_patients > 0 else "FAIL"),
    ]
    with open(INTEGRITY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item","actual","expected","status"])
        w.writeheader()
        for name, actual, expected, status in integrity_items:
            w.writerow({"item": name, "actual": actual, "expected": expected, "status": status})
    print(f"\n  [SAVED] {INTEGRITY_CSV}")

    # 오류 행 저장
    if error_rows or failed_rows_list:
        with open(ERRORS_CSV, "w", newline="") as f:
            fields = ["row_id","patient_id","safe_id","local_z","status","error_message"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in failed_rows_list:
                w.writerow({k: r.get(k,"") for k in fields})
        print(f"  [SAVED] {ERRORS_CSV}")

    # ── 판정 ───────────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    fail_items = [name for name, actual, expected, status in integrity_items
                  if status in ("FAIL", "BLOCKED")]
    warn_items = [name for name, actual, expected, status in integrity_items
                  if status == "WARN"]

    if not fail_items:
        verdict = "PASS_STEP13_STAGE2_MERGE_INTEGRITY"
    elif any(s == "BLOCKED" for _, _, _, s in integrity_items if s == "BLOCKED"):
        verdict = f"BLOCKED_STEP13: {','.join(fail_items)}"
    else:
        verdict = f"PARTIAL_PASS_STEP13: {','.join(fail_items)}"

    print()
    print("=" * 64)
    print(f"판정: {verdict}")
    print(f"  merged rows  : {total_loaded:,}")
    print(f"  failed       : {len(failed_rows_list)}")
    print(f"  NaN/Inf      : {nan_count}/{inf_count}")
    print(f"  duplicate    : {dup_count}")
    print(f"  missing      : {len(missing_planned)}")
    print(f"  coord_errors : {coord_errors}")
    print(f"  tracks       : {n_tracks:,}")
    print(f"  patients     : {n_patients}")
    print(f"  runtime      : {elapsed:.1f}s")
    if warn_items:
        print(f"  WARN items   : {warn_items}")
    print("=" * 64)

    # ── guardrail + summary JSON ───────────────────────────────────────────────
    guardrail = {
        "step12_passed"                 : True,
        "merge_only"                    : True,
        "stage2_label_used_for_metric"  : False,
        "stage2_label_used_for_tuning"  : False,
        "threshold_tuning_executed"     : False,
        "score_family_changed"          : False,
        "P1_rejected_for_lung5ch"       : True,
        "P2_rejected_for_lung5ch"       : True,
        "primary_candidate_score_locked": PRIMARY_CAND_SCORE,
        "primary_track_score_locked"    : PRIMARY_TRACK_SCORE,
        "candidate_deletion_executed"   : False,
        "representative_only_scoring_used": False,
        "all_survived_candidates_scored": True,
        "expected_rows"                 : EXPECTED_TOTAL_ROWS,
        "merged_rows"                   : total_loaded,
        "duplicate_rows"                : dup_count,
        "missing_rows"                  : len(missing_planned),
        "failed_rows"                   : len(failed_rows_list),
        "nan_count"                     : nan_count,
        "inf_count"                     : inf_count,
    }

    summary = {
        "step"                 : "step13_stage2_merge_integrity",
        "verdict"              : verdict,
        "created"              : str(date.today()),
        "merged_rows"          : total_loaded,
        "expected_rows"        : EXPECTED_TOTAL_ROWS,
        "failed_rows"          : len(failed_rows_list),
        "nan_count"            : nan_count,
        "inf_count"            : inf_count,
        "duplicate_rows"       : dup_count,
        "missing_planned_rows" : len(missing_planned),
        "coord_errors"         : coord_errors,
        "n_tracks"             : n_tracks,
        "n_patients"           : n_patients,
        "shard_row_counts"     : {str(k): v for k, v in shard_row_counts.items()},
        "primary_candidate_score": PRIMARY_CAND_SCORE,
        "primary_track_score"    : PRIMARY_TRACK_SCORE,
        "merged_csv"           : str(MERGED_CSV),
        "track_csv"            : str(TRACK_CSV),
        "patient_csv"          : str(PATIENT_CSV),
        "score_dist_csv"       : str(SCORE_DIST_CSV),
        "guardrail"            : guardrail,
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [SAVED] {SUMMARY_JSON}")

    # ── report.md ─────────────────────────────────────────────────────────────
    raw_st = score_stats(dist_data["rd4ad_lung5ch_score_raw"])
    lines = [
        f"# Step 13 Stage2 Merge + Integrity Report",
        f"",
        f"**verdict**: {verdict}",
        f"",
        f"## Merge 결과",
        f"",
        f"| 항목 | 값 |",
        f"|---|---|",
        f"| merged_rows | {total_loaded:,} |",
        f"| expected | {EXPECTED_TOTAL_ROWS:,} |",
        f"| failed_rows | {len(failed_rows_list)} |",
        f"| NaN/Inf | {nan_count}/{inf_count} |",
        f"| duplicate_rows | {dup_count} |",
        f"| missing_planned | {len(missing_planned)} |",
        f"| coord_errors | {coord_errors} |",
        f"| n_tracks | {n_tracks:,} |",
        f"| n_patients | {n_patients} |",
        f"",
        f"## Score Distribution ({PRIMARY_CAND_SCORE})",
        f"",
        f"| 통계 | 값 |",
        f"|---|---|",
    ]
    for k in ["mean","std","min","p10","p25","p50","p75","p90","p95","p99","max"]:
        v = raw_st.get(k, float("nan"))
        lines.append(f"| {k} | {v:.6f} |" if isinstance(v, float) and not math.isnan(v) else f"| {k} | {v} |")
    lines += [
        f"",
        f"## Shard Row Counts",
        f"",
    ]
    for i, cnt in shard_row_counts.items():
        lines.append(f"- shard {i:03d}: {cnt:,}행")
    lines += [
        f"",
        f"## Guardrail",
        f"",
        f"- stage2_label_used: False",
        f"- threshold_tuning: False",
        f"- score_family_changed: False",
        f"- primary_candidate_score: {PRIMARY_CAND_SCORE}",
        f"- primary_track_score: {PRIMARY_TRACK_SCORE}",
        f"",
        f"## Next Step",
        f"",
        f"Step 14: stage2 fixed evaluation (label 기반 patient hit rate / track hit rate)",
    ]
    with open(REPORT_MD, "w") as f:
        f.write("\n".join(lines))
    print(f"  [SAVED] {REPORT_MD}")

    # ── DONE ──────────────────────────────────────────────────────────────────
    done = {
        "step"       : "step13_stage2_merge_integrity",
        "verdict"    : verdict,
        "created"    : str(date.today()),
        "merged_rows": total_loaded,
        "n_tracks"   : n_tracks,
        "n_patients" : n_patients,
        "merged_csv" : str(MERGED_CSV),
        "track_csv"  : str(TRACK_CSV),
        "summary_json": str(SUMMARY_JSON),
    }
    with open(DONE_OUT, "w") as f:
        json.dump(done, f, indent=2)
    print(f"  [SAVED] {DONE_OUT}")

    if verdict == "PASS_STEP13_STAGE2_MERGE_INTEGRITY":
        print("\nStep 13 완료. 다음 단계: Step 14 stage2 fixed evaluation (사용자 승인 후)")
    else:
        print(f"\nStep 13 {verdict}. 결과 확인 후 다음 단계 결정.")


def main():
    args = parse_args()

    if args.dry_run:
        print_dry_run()
        sys.exit(0)

    if args.run_merge:
        missing = []
        if not args.confirm_step12_pass:
            missing.append("--confirm-step12-pass")
        if not args.confirm_merge_only:
            missing.append("--confirm-merge-only")
        if not args.confirm_no_label_eval:
            missing.append("--confirm-no-label-eval")
        if missing:
            print(f"[BLOCKED] 필수 flags 누락: {missing}", file=sys.stderr)
            sys.exit(2)
        run_merge(args)
        return

    print("[BLOCKED] --dry-run 또는 --run-merge 필요", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
