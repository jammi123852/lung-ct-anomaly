"""
Step 14: Stage2 Fixed Evaluation
rd4ad_2p5d_lung5ch_masked_normal_v1

대상: step13 merged score (127,947 rows) + stage2 label merge
      candidate-level / track-level top-k evaluation
      label은 metric 계산에만 사용 — score/threshold/rule 변경 금지

bare run → exit 2
dry-run  → 계획 출력
--run-eval → actual evaluation
"""

import sys
import json
import csv
import math
import time
import argparse
import collections
from pathlib import Path
from datetime import date

if len(sys.argv) == 1:
    print("[BLOCKED] bare run 금지. --dry-run 또는 --run-eval을 사용하세요.", file=sys.stderr)
    sys.exit(2)

# ── 경로 상수 ─────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]

DONE_STEP13_JSON  = ROOT / "DONE_STEP13_STAGE2_MERGE_INTEGRITY.json"
MERGED_CSV        = ROOT / "scoring" / "step13_stage2_fixed_merged_v1" / "stage2_lung5ch_fixed_scores_merged.csv"
TRACK_CSV         = ROOT / "manifests" / "step13_stage2_track_level_scores.csv"
PATIENT_CSV_13    = ROOT / "manifests" / "step13_stage2_patient_summary.csv"
STEP9_SUMMARY     = ROOT / "reports" / "step9_stage1dev_eval_summary.json"

STAGE2_MANIFEST_ORIG = PROJECT_ROOT / "outputs" / \
    "second-stage-lesion-refiner-v1" / "datasets" / \
    "s6a_stage2_holdout_candidate_coordinate_manifest_v1.csv"

# 기존 RD4AD stage2 baseline
BASELINE_STAGE2_SUMMARY = ROOT.parents[0] / \
    "stage2_strict_ztrack_rd4ad_scoring_preflight_v1" / \
    "reports" / "stage2_rd4ad_evaluation_summary.json"

OUT_MANIFESTS = ROOT / "manifests"
OUT_REPORTS   = ROOT / "reports"
OUT_LOGS      = ROOT / "logs"
DONE_OUT      = ROOT / "DONE_STEP14_STAGE2_FIXED_EVAL.json"

CAND_TOPK_CSV     = OUT_MANIFESTS / "step14_candidate_level_topk.csv"
TRACK_TOPK_CSV    = OUT_MANIFESTS / "step14_track_level_topk.csv"
PATIENT_HIT_CSV   = OUT_MANIFESTS / "step14_patient_hit_summary.csv"
COMPLETE_MISS_CSV = OUT_MANIFESTS / "step14_complete_miss_audit.csv"
PROB_PATIENT_CSV  = OUT_MANIFESTS / "step14_problem_patient_audit.csv"
COMPARISON_CSV    = OUT_MANIFESTS / "step14_stage1_vs_stage2_comparison.csv"
REPORT_MD         = OUT_REPORTS   / "step14_stage2_fixed_eval_report.md"
SUMMARY_JSON      = OUT_REPORTS   / "step14_stage2_fixed_eval_summary.json"
ERRORS_CSV        = OUT_LOGS      / "step14_stage2_fixed_eval_errors.csv"

PRIMARY_CAND_SCORE  = "rd4ad_lung5ch_score_raw"
PRIMARY_TRACK_SCORE = "raw_track_top3_mean"
TOPK_LIST           = [1, 3, 5, 10, 20, 50]
PROBLEM_PATIENTS    = ["LUNG1-086", "LUNG1-386", "LUNG1-399"]
EXPECTED_ROWS       = 127947


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run-eval", action="store_true")
    ap.add_argument("--confirm-step13-pass", action="store_true")
    ap.add_argument("--confirm-fixed-eval-only", action="store_true")
    ap.add_argument("--confirm-no-tuning", action="store_true")
    return ap.parse_args()


def print_dry_run():
    print()
    print("=" * 64)
    print("Step 14 Stage2 Fixed Evaluation — DRY-RUN PLAN")
    print("=" * 64)
    print()
    print("[입력]")
    print(f"  merged CSV   : {MERGED_CSV}")
    print(f"  track CSV    : {TRACK_CSV}")
    print(f"  stage2 label : {STAGE2_MANIFEST_ORIG}")
    print(f"  step9 summary: {STEP9_SUMMARY}")
    print()
    print("[할 일]")
    for item in [
        "1. integrity 재확인 (step13 DONE)",
        "2. label merge  (row_id 기준, label=0/1)",
        "3. candidate-level top-k evaluation",
        "4. track-level top-k evaluation (raw_track_top3_mean primary)",
        "5. stage1_dev vs stage2_holdout comparison",
        "6. complete miss audit (top20 기준)",
        "7. problem patient audit (LUNG1-086/386/399)",
        "8. baseline comparison (rd4ad stage2 baseline)",
        "9. guardrail 기록",
    ]:
        print(f"  {item}")
    print()
    print("[고정 rule]")
    print(f"  primary candidate: {PRIMARY_CAND_SCORE}")
    print(f"  primary track    : {PRIMARY_TRACK_SCORE}")
    print(f"  p90 threshold    : 12.196394  (재계산 금지)")
    print(f"  P1/P2            : REJECT")
    print()
    print("[절대 금지]")
    print("  label 보고 score/threshold/rule 변경")
    print("  stage2 결과 기반 후처리 설계")
    print("  training / model forward / checkpoint 수정")
    print()
    rel = Path("experiments/rd4ad_2p5d_lung5ch_masked_normal_v1/scripts") / Path(__file__).name
    print("[실행 명령]")
    print(f"  python {rel} \\")
    print(f"    --run-eval --confirm-step13-pass --confirm-fixed-eval-only --confirm-no-tuning")
    print()
    print("DRY-RUN 완료.")


def check_guards():
    errors = []
    if not DONE_STEP13_JSON.exists():
        errors.append("DONE_STEP13 없음")
    else:
        with open(DONE_STEP13_JSON) as f:
            d = json.load(f)
        if d.get("verdict") != "PASS_STEP13_STAGE2_MERGE_INTEGRITY":
            errors.append(f"Step13 verdict={d.get('verdict')}")
        if d.get("merged_rows") != EXPECTED_ROWS:
            errors.append(f"merged_rows={d.get('merged_rows')} != {EXPECTED_ROWS}")
    for fp, label in [(MERGED_CSV, "merged CSV"), (TRACK_CSV, "track CSV"),
                      (STAGE2_MANIFEST_ORIG, "stage2 manifest")]:
        if not fp.exists():
            errors.append(f"{label} 없음: {fp}")
    return errors


def percentile(values, pct):
    if not values:
        return float("nan")
    sv = sorted(values)
    n = len(sv)
    idx = pct / 100.0 * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    return sv[lo] * (1 - (idx - lo)) + sv[hi] * (idx - lo)


def run_eval(args):
    t_start = time.perf_counter()
    print()
    print("=" * 64)
    print("Step 14 Stage2 Fixed Evaluation — ACTUAL RUN")
    print("=" * 64)

    # [0] guards
    print("\n[0] Guards 확인")
    guard_errors = check_guards()
    if guard_errors:
        print("  [BLOCKED]:")
        for e in guard_errors:
            print(f"    - {e}")
        sys.exit(1)
    print("  [PASS]")

    for d in [OUT_MANIFESTS, OUT_REPORTS, OUT_LOGS]:
        d.mkdir(parents=True, exist_ok=True)

    # [1] merged CSV 로드
    print("\n[1] Merged score CSV 로드")
    with open(MERGED_CSV, newline="") as f:
        score_rows = list(csv.DictReader(f))
    print(f"  {len(score_rows):,}행 로드")
    if len(score_rows) != EXPECTED_ROWS:
        print(f"  [BLOCKED] {len(score_rows)} != {EXPECTED_ROWS}", file=sys.stderr)
        sys.exit(1)

    # NaN/Inf 재확인
    nan_c = inf_c = 0
    for r in score_rows:
        try:
            v = float(r["rd4ad_lung5ch_score_raw"])
            if math.isnan(v): nan_c += 1
            elif math.isinf(v): inf_c += 1
        except (ValueError, TypeError):
            nan_c += 1
    if nan_c > 0 or inf_c > 0:
        print(f"  [BLOCKED] NaN={nan_c} Inf={inf_c}", file=sys.stderr)
        sys.exit(1)
    print(f"  NaN={nan_c} Inf={inf_c}  [OK]")

    # [2] label merge
    print("\n[2] label merge (row_id 기준, stage2 manifest)")
    label_lookup = {}        # row_id_str -> '0' or '1'
    full_positive_patients = set()   # label=1인 patient 전체 (p90+z-cont 필터 이전)
    pid_to_safe_id_manifest = {}     # manifest에서 safe_id 조회 (필터된 환자 보조)
    with open(STAGE2_MANIFEST_ORIG, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rid = row.get("row_id", "")
            lbl = row.get("label", "")
            pid = row.get("patient_id", "")
            if rid:
                label_lookup[rid] = lbl
            if lbl == "1" and pid:
                full_positive_patients.add(pid)
            if pid and row.get("safe_id", ""):
                pid_to_safe_id_manifest.setdefault(pid, row["safe_id"])

    full_stage2_positive_patients = len(full_positive_patients)
    print(f"  label lookup: {len(label_lookup):,}건")
    print(f"  full_positive_patients (manifest 전체): {full_stage2_positive_patients}")
    # 양성/음성 변환 규칙: '1' → positive=True
    merge_matched = merge_missing = 0
    label_conv_rule = "label='1' → positive=True, label='0' → positive=False"
    for r in score_rows:
        rid = r.get("row_id", "")
        lbl = label_lookup.get(rid, "")
        if lbl:
            r["_label"] = lbl
            r["_is_positive"] = (lbl == "1")
            merge_matched += 1
        else:
            r["_label"] = ""
            r["_is_positive"] = False
            merge_missing += 1

    print(f"  matched: {merge_matched:,}  missing: {merge_missing}")
    if merge_missing > 0:
        print(f"  [WARN] label 미매칭 {merge_missing}건 — positive=False로 처리")
    print(f"  label conv rule: {label_conv_rule}")

    # 집계
    total_positive_cands = sum(1 for r in score_rows if r["_is_positive"])
    total_negative_cands = sum(1 for r in score_rows if not r["_is_positive"])
    scored_positive_patients = set(r["patient_id"] for r in score_rows if r["_is_positive"])
    all_patients = sorted(set(r["patient_id"] for r in score_rows))
    total_patients = len(all_patients)

    # denominator guard: full_positive_patients (p90+z-cont 필터 이전 전체)
    positive_patients = full_positive_patients            # hit rate denominator
    total_positive_patients = full_stage2_positive_patients
    positive_patients_missing_after_filter = sorted(full_positive_patients - scored_positive_patients)
    n_missing_after_filter = len(positive_patients_missing_after_filter)
    positive_patient_retention = round(len(scored_positive_patients) / max(1, total_positive_patients), 4)
    denominator_same = (len(scored_positive_patients) == total_positive_patients)

    print(f"  총 candidate: {EXPECTED_ROWS:,}")
    print(f"  positive cand: {total_positive_cands:,}")
    print(f"  negative cand: {total_negative_cands:,}")
    print(f"  full_positive_patients (denominator): {total_positive_patients}")
    print(f"  scored_positive_patients (p90+z-cont 통과): {len(scored_positive_patients)}")
    print(f"  missing_after_filter: {n_missing_after_filter}")
    print(f"  positive_patient_retention: {positive_patient_retention:.4f}")
    if denominator_same:
        print(f"  [INFO] denominator 동일 — unguarded 결과와 동일")
    else:
        print(f"  [WARN] {n_missing_after_filter}명 positive patient가 p90+z-cont 필터로 탈락 — denominator 보정 적용")
    print(f"  total patients (scored): {total_patients}")

    # [3] candidate-level top-k
    print(f"\n[3] candidate-level top-k  (primary: {PRIMARY_CAND_SCORE})")

    # patient별 score 내림차순 정렬
    pat_cands = collections.defaultdict(list)
    for r in score_rows:
        pat_cands[r["patient_id"]].append(r)
    for pid in pat_cands:
        pat_cands[pid].sort(key=lambda x: float(x[PRIMARY_CAND_SCORE]), reverse=True)

    cand_topk_results = {}
    cand_best_rank = {}   # positive patient → rank of best positive candidate (1-indexed)

    for k in TOPK_LIST:
        hit_count = 0
        covered_pos_cands = 0
        total_pos_cands = total_positive_cands
        miss_pids = []

        for pid in positive_patients:
            ranked = pat_cands[pid]
            topk_rows = ranked[:k]
            has_hit = any(r["_is_positive"] for r in topk_rows)
            if has_hit:
                hit_count += 1
                covered_pos_cands += sum(1 for r in topk_rows if r["_is_positive"])
            else:
                miss_pids.append(pid)

        cand_topk_results[k] = {
            "k": k,
            "hit_patient_count": hit_count,
            "total_positive_patients": total_positive_patients,
            "patient_hit_rate": round(hit_count / max(1, total_positive_patients), 4),
            "positive_candidate_coverage": round(covered_pos_cands / max(1, total_pos_cands), 4),
            "complete_miss_patient_count": len(miss_pids),
            "complete_miss_patient_ids": sorted(miss_pids),
        }
        print(f"  top{k:2d}: hit_rate={cand_topk_results[k]['patient_hit_rate']:.4f}  "
              f"hit={hit_count}/{total_positive_patients}(full)  "
              f"miss={len(miss_pids)}")

    # best positive rank per patient
    for pid in positive_patients:
        ranked = pat_cands[pid]
        for rank_i, r in enumerate(ranked, 1):
            if r["_is_positive"]:
                cand_best_rank[pid] = rank_i
                break

    best_ranks = list(cand_best_rank.values())
    cand_rank_stats = {
        "mean":   round(sum(best_ranks) / max(1, len(best_ranks)), 2) if best_ranks else None,
        "median": round(percentile(best_ranks, 50), 2) if best_ranks else None,
        "p90":    round(percentile(best_ranks, 90), 2) if best_ranks else None,
    }

    # [4] track-level top-k
    print(f"\n[4] track-level top-k  (primary: {PRIMARY_TRACK_SCORE})")

    # track CSV 로드
    with open(TRACK_CSV, newline="") as f:
        track_rows = list(csv.DictReader(f))

    # track별 positive 여부 결정: track_id에 속한 candidate 중 positive 있으면 positive track
    track_pos_cand_count = collections.defaultdict(int)
    for r in score_rows:
        tid = r.get("track_id", "")
        if r["_is_positive"] and tid:
            track_pos_cand_count[tid] += 1

    # track 데이터에 positive 여부 추가
    for tr in track_rows:
        tid = tr.get("track_id", "")
        tr["_is_positive"] = track_pos_cand_count.get(tid, 0) > 0
        tr["_pos_cand_count"] = track_pos_cand_count.get(tid, 0)

    # patient별 track 정렬
    pat_tracks = collections.defaultdict(list)
    for tr in track_rows:
        pat_tracks[tr["patient_id"]].append(tr)
    for pid in pat_tracks:
        pat_tracks[pid].sort(key=lambda x: float(x[PRIMARY_TRACK_SCORE]), reverse=True)

    positive_track_patients = set(
        tr["patient_id"] for tr in track_rows if tr["_is_positive"]
    )
    total_positive_tracks = sum(1 for tr in track_rows if tr["_is_positive"])

    track_topk_results = {}
    track_best_rank = {}

    for k in TOPK_LIST:
        hit_count = 0
        covered_pos_tracks = 0
        miss_pids = []

        # denominator guard: full_positive_patients (필터된 환자도 포함)
        for pid in full_positive_patients:
            ranked = pat_tracks.get(pid, [])
            topk_trs = ranked[:k]
            has_hit = any(tr["_is_positive"] for tr in topk_trs)
            if has_hit:
                hit_count += 1
                covered_pos_tracks += sum(1 for tr in topk_trs if tr["_is_positive"])
            else:
                miss_pids.append(pid)

        track_topk_results[k] = {
            "k": k,
            "hit_patient_count": hit_count,
            "total_positive_patients": total_positive_patients,
            "patient_hit_rate": round(hit_count / max(1, total_positive_patients), 4),
            "positive_track_coverage": round(covered_pos_tracks / max(1, total_positive_tracks), 4),
            "complete_miss_patient_count": len(miss_pids),
            "complete_miss_patient_ids": sorted(miss_pids),
        }
        print(f"  top{k:2d}: hit_rate={track_topk_results[k]['patient_hit_rate']:.4f}  "
              f"hit={hit_count}/{total_positive_patients}(full)  "
              f"miss={len(miss_pids)}")

    for pid in positive_track_patients:
        ranked = pat_tracks[pid]
        for rank_i, tr in enumerate(ranked, 1):
            if tr["_is_positive"]:
                track_best_rank[pid] = rank_i
                break

    track_br = list(track_best_rank.values())
    track_rank_stats = {
        "mean":   round(sum(track_br) / max(1, len(track_br)), 2) if track_br else None,
        "median": round(percentile(track_br, 50), 2) if track_br else None,
        "p90":    round(percentile(track_br, 90), 2) if track_br else None,
    }

    # [5] stage1_dev vs stage2 비교
    print("\n[5] stage1_dev vs stage2_holdout 비교")
    stage1_cand_topk = {}
    stage1_track_topk = {}
    stage1_pos_patients = None

    if STEP9_SUMMARY.exists():
        with open(STEP9_SUMMARY) as f:
            s9 = json.load(f)
        cand_raw = s9.get("candidate_topk", {})
        track_raw = s9.get("track_topk", {})
        stage1_pos_patients = s9.get("positive_patients")
        for k in TOPK_LIST:
            kstr = str(k)
            if isinstance(cand_raw, dict) and kstr in cand_raw:
                v = cand_raw[kstr]
                stage1_cand_topk[k] = v.get("raw") if isinstance(v, dict) else v
            if isinstance(track_raw, dict) and kstr in track_raw:
                v = track_raw[kstr]
                stage1_track_topk[k] = v.get("raw_top3") if isinstance(v, dict) else v
        print(f"  step9 stage1_dev loaded (pos_patients={stage1_pos_patients})")
    else:
        print(f"  [WARN] step9 summary 없음 — hardcoded 값 사용")
        stage1_cand_topk  = {1:0.4600,3:0.5867,5:0.6000,10:0.6933,20:0.7600,50:0.8133}
        stage1_track_topk = {1:0.5000,3:0.6467,5:0.7133,10:0.7800,20:0.8333,50:0.9400}

    comparison_rows = []
    for k in TOPK_LIST:
        s1c = stage1_cand_topk.get(k)
        s2c = cand_topk_results[k]["patient_hit_rate"]
        s1t = stage1_track_topk.get(k)
        s2t = track_topk_results[k]["patient_hit_rate"]

        def delta(a, b):
            if a is None or b is None: return None
            return round(b - a, 4)

        comparison_rows.append({
            "k"                        : k,
            "stage1_dev_cand_raw"      : s1c,
            "stage2_holdout_cand_raw"  : s2c,
            "cand_delta"               : delta(s1c, s2c),
            "stage1_dev_track_raw_top3": s1t,
            "stage2_holdout_track_raw_top3": s2t,
            "track_delta"              : delta(s1t, s2t),
            "generalization_note"      : (
                "DROP" if (delta(s1c, s2c) is not None and delta(s1c, s2c) < -0.05)
                else ("HOLD" if (delta(s1c, s2c) is not None and delta(s1c, s2c) >= -0.05)
                else "N/A")
            ),
        })
        print(f"  top{k:2d}: cand {s1c}→{s2c} ({'+' if (delta(s1c,s2c) or 0)>=0 else ''}{delta(s1c,s2c)})  "
              f"track {s1t}→{s2t} ({'+' if (delta(s1t,s2t) or 0)>=0 else ''}{delta(s1t,s2t)})")

    comp_fields = ["k","stage1_dev_cand_raw","stage2_holdout_cand_raw","cand_delta",
                   "stage1_dev_track_raw_top3","stage2_holdout_track_raw_top3","track_delta","generalization_note"]
    with open(COMPARISON_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=comp_fields)
        w.writeheader()
        w.writerows(comparison_rows)
    print(f"  [SAVED] {COMPARISON_CSV}")

    # [6] complete miss audit (top20 기준)
    print("\n[6] complete miss audit (top20 기준)")
    cand_miss_top20 = set(cand_topk_results[20]["complete_miss_patient_ids"])
    track_miss_top20 = set(track_topk_results[20]["complete_miss_patient_ids"])
    audit_pids = sorted(cand_miss_top20 | track_miss_top20)
    print(f"  cand miss@20  : {len(cand_miss_top20)}")
    print(f"  track miss@20 : {len(track_miss_top20)}")
    print(f"  union         : {len(audit_pids)}")

    audit_rows = []
    for pid in audit_pids:
        safe_id = (pat_cands[pid][0]["safe_id"] if pat_cands[pid]
                   else pid_to_safe_id_manifest.get(pid, ""))
        all_cands = pat_cands[pid]
        pos_cands = [r for r in all_cands if r["_is_positive"]]
        pt_tracks = pat_tracks.get(pid, [])
        pos_tracks = [tr for tr in pt_tracks if tr["_is_positive"]]

        best_cand_rank = cand_best_rank.get(pid)
        best_track_rank = track_best_rank.get(pid)

        def hit_at_k(k, is_cand=True):
            results = cand_topk_results if is_cand else track_topk_results
            return pid not in results[k]["complete_miss_patient_ids"]

        # failure mode 분류 (데이터 기반)
        if not pos_cands:
            mode = "no_positive_candidate_in_filtered_set"
        elif best_cand_rank is not None and best_cand_rank <= 50:
            mode = "borderline_rank_21_50"
        elif best_cand_rank is not None and best_cand_rank > 100:
            mode = "buried_deep_100_plus"
        elif len(pos_cands) < 3:
            mode = "low_positive_density"
        elif not pos_tracks:
            mode = "no_positive_track"
        else:
            mode = "unknown"

        audit_rows.append({
            "patient_id"                 : pid,
            "safe_id"                    : safe_id,
            "total_candidates"           : len(all_cands),
            "total_tracks"               : len(pt_tracks),
            "positive_candidates"        : len(pos_cands),
            "positive_tracks"            : len(pos_tracks),
            "best_positive_candidate_rank_raw": best_cand_rank,
            "best_positive_track_rank_raw_top3": best_track_rank,
            "candidate_hit_top10"        : hit_at_k(10, is_cand=True),
            "candidate_hit_top20"        : hit_at_k(20, is_cand=True),
            "candidate_hit_top50"        : hit_at_k(50, is_cand=True),
            "track_hit_top10"            : hit_at_k(10, is_cand=False),
            "track_hit_top20"            : hit_at_k(20, is_cand=False),
            "track_hit_top50"            : hit_at_k(50, is_cand=False),
            "likely_failure_mode"        : mode,
            "note"                       : "",
        })

    audit_fields = ["patient_id","safe_id","total_candidates","total_tracks",
                    "positive_candidates","positive_tracks",
                    "best_positive_candidate_rank_raw","best_positive_track_rank_raw_top3",
                    "candidate_hit_top10","candidate_hit_top20","candidate_hit_top50",
                    "track_hit_top10","track_hit_top20","track_hit_top50",
                    "likely_failure_mode","note"]
    with open(COMPLETE_MISS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=audit_fields)
        w.writeheader()
        w.writerows(audit_rows)
    print(f"  [SAVED] {COMPLETE_MISS_CSV}")

    # [7] problem patient audit
    print("\n[7] problem patient audit (LUNG1-086/386/399)")
    prob_rows = []
    for pid in PROBLEM_PATIENTS:
        if pid not in set(all_patients):
            prob_rows.append({"patient_id": pid, "note": "not_in_stage2_holdout",
                              **{f: "" for f in audit_fields[2:]}})
            print(f"  {pid}: not_in_stage2_holdout")
            continue

        all_cands = pat_cands.get(pid, [])
        pos_cands = [r for r in all_cands if r["_is_positive"]]
        pt_tracks = pat_tracks.get(pid, [])
        pos_tracks = [tr for tr in pt_tracks if tr["_is_positive"]]
        safe_id = all_cands[0]["safe_id"] if all_cands else ""

        best_cand_rank = cand_best_rank.get(pid)
        best_track_rank = track_best_rank.get(pid)

        def hit_k(k, cand=True):
            res = cand_topk_results if cand else track_topk_results
            return pid not in res[k]["complete_miss_patient_ids"]

        note = f"cand_count={len(all_cands)},pos_cands={len(pos_cands)},best_rank={best_cand_rank}"
        prob_rows.append({
            "patient_id"                 : pid,
            "safe_id"                    : safe_id,
            "total_candidates"           : len(all_cands),
            "total_tracks"               : len(pt_tracks),
            "positive_candidates"        : len(pos_cands),
            "positive_tracks"            : len(pos_tracks),
            "best_positive_candidate_rank_raw": best_cand_rank,
            "best_positive_track_rank_raw_top3": best_track_rank,
            "candidate_hit_top10"        : hit_k(10, cand=True),
            "candidate_hit_top20"        : hit_k(20, cand=True),
            "candidate_hit_top50"        : hit_k(50, cand=True),
            "track_hit_top10"            : hit_k(10, cand=False),
            "track_hit_top20"            : hit_k(20, cand=False),
            "track_hit_top50"            : hit_k(50, cand=False),
            "likely_failure_mode"        : "problem_patient_audit",
            "note"                       : note,
        })
        print(f"  {pid}: best_cand_rank={best_cand_rank}  best_track_rank={best_track_rank}  "
              f"hit@10={hit_k(10)},@20={hit_k(20)},@50={hit_k(50)}")

    with open(PROB_PATIENT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=audit_fields)
        w.writeheader()
        w.writerows(prob_rows)
    print(f"  [SAVED] {PROB_PATIENT_CSV}")

    # [8] baseline comparison
    print("\n[8] baseline comparison (rd4ad stage2)")
    baseline_note = ""
    baseline_data = {}
    if BASELINE_STAGE2_SUMMARY.exists():
        with open(BASELINE_STAGE2_SUMMARY) as f:
            bl = json.load(f)
        # PD_sqrthu_len_track_top3_mean이 best 컬럼
        stage2_results = bl.get("stage2_results", {})
        best_key = "PD_sqrthu_len_track_top3_mean"
        if best_key in stage2_results:
            baseline_data = stage2_results[best_key]
            baseline_note = f"rd_d1s_stage2 {best_key} (denominator={bl.get('n_candidates_merged','-')})"
        print(f"  baseline loaded: {baseline_note}")
        print(f"  caution: denominator 다를 수 있음 (rd_d1s: 128,827 vs lung5ch: 127,947)")
    else:
        baseline_note = "baseline comparison deferred / file not found"
        print(f"  [INFO] {baseline_note}")

    # [9] patient hit summary CSV
    print("\n[9] patient hit summary 생성")
    # full_positive_patients 중 scored set에 없는 환자도 포함
    all_patients_for_summary = sorted(set(all_patients) | full_positive_patients)
    pat_hit_rows = []
    for pid in all_patients_for_summary:
        all_cands = pat_cands.get(pid, [])
        pos_cands = [r for r in all_cands if r["_is_positive"]]
        safe_id = (all_cands[0]["safe_id"] if all_cands
                   else pid_to_safe_id_manifest.get(pid, ""))
        is_positive_patient = pid in full_positive_patients

        row = {
            "patient_id"         : pid,
            "safe_id"            : safe_id,
            "is_positive_patient": is_positive_patient,
            "candidate_count"    : len(all_cands),
            "positive_candidate_count": len(pos_cands),
            "best_positive_cand_rank": cand_best_rank.get(pid, ""),
            "best_positive_track_rank": track_best_rank.get(pid, ""),
        }
        for k in TOPK_LIST:
            row[f"cand_hit_top{k}"] = (pid not in cand_topk_results[k]["complete_miss_patient_ids"]) if is_positive_patient else ""
            row[f"track_hit_top{k}"] = (pid not in track_topk_results[k]["complete_miss_patient_ids"]) if is_positive_patient else ""
        pat_hit_rows.append(row)

    pat_hit_fields = (["patient_id","safe_id","is_positive_patient","candidate_count",
                       "positive_candidate_count","best_positive_cand_rank","best_positive_track_rank"]
                      + [f"cand_hit_top{k}" for k in TOPK_LIST]
                      + [f"track_hit_top{k}" for k in TOPK_LIST])
    with open(PATIENT_HIT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pat_hit_fields)
        w.writeheader()
        w.writerows(pat_hit_rows)
    print(f"  [SAVED] {PATIENT_HIT_CSV}")

    # candidate_topk/track_topk CSV
    cand_tk_fields = ["k","hit_patient_count","total_positive_patients","patient_hit_rate",
                      "positive_candidate_coverage","complete_miss_patient_count"]
    with open(CAND_TOPK_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cand_tk_fields)
        w.writeheader()
        for k in TOPK_LIST:
            r = cand_topk_results[k]
            w.writerow({fi: r[fi] for fi in cand_tk_fields})
    print(f"  [SAVED] {CAND_TOPK_CSV}")

    track_tk_fields = ["k","hit_patient_count","total_positive_patients","patient_hit_rate",
                       "positive_track_coverage","complete_miss_patient_count"]
    with open(TRACK_TOPK_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=track_tk_fields)
        w.writeheader()
        for k in TOPK_LIST:
            r = track_topk_results[k]
            w.writerow({fi: r[fi] for fi in track_tk_fields})
    print(f"  [SAVED] {TRACK_TOPK_CSV}")

    # ── 판정 ───────────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    label_issues = merge_missing > 0

    if label_issues:
        verdict = "PARTIAL_PASS_STEP14_LABEL_MERGE_WARN"
    else:
        verdict = "PASS_STEP14_STAGE2_FIXED_EVAL"

    # ── summary JSON ───────────────────────────────────────────────────────────
    guardrail = {
        "step13_merge_integrity_passed"  : True,
        "fixed_evaluation_only"          : True,
        "stage2_label_used_for_metric"   : True,
        "stage2_label_used_for_tuning"   : False,
        "score_family_changed_on_stage2" : False,
        "threshold_tuning_executed"      : False,
        "p90_recomputed_on_stage2"       : False,
        "primary_candidate_score_locked" : PRIMARY_CAND_SCORE,
        "primary_track_score_locked"     : PRIMARY_TRACK_SCORE,
        "P1_rejected_for_lung5ch"        : True,
        "P2_rejected_for_lung5ch"        : True,
        "candidate_deletion_executed"    : False,
        "representative_only_eval_used"  : False,
        "training_executed"              : False,
        "model_forward_executed"         : False,
        "checkpoint_saved"               : False,
        "checkpoint_modified"            : False,
        "denominator_candidates"                       : EXPECTED_ROWS,
        "denominator_tracks"                           : len(track_rows),
        "denominator_patients"                         : total_patients,
        "denominator_guard_applied"                    : True,
        "denominator_corrected"                        : not denominator_same,
        "full_stage2_positive_patients_denominator"    : total_positive_patients,
        "scored_positive_patients_after_filter"        : len(scored_positive_patients),
    }

    summary = {
        "step"                                        : "step14_stage2_fixed_eval",
        "verdict"                                     : verdict,
        "created"                                     : str(date.today()),
        "label_conv_rule"                             : label_conv_rule,
        "label_merge_matched"                         : merge_matched,
        "label_merge_missing"                         : merge_missing,
        "total_candidates"                            : EXPECTED_ROWS,
        "total_positive_candidates"                   : total_positive_cands,
        "total_negative_candidates"                   : total_negative_cands,
        "total_patients"                              : total_patients,
        "full_stage2_positive_patients"               : total_positive_patients,
        "scored_positive_patients"                    : len(scored_positive_patients),
        "positive_patient_retention_after_p90_ztrack" : positive_patient_retention,
        "positive_patients_missing_after_filter"      : n_missing_after_filter,
        "positive_patients_missing_ids"               : positive_patients_missing_after_filter,
        "denominator_same_as_unguarded"               : denominator_same,
        "denominator_note"                            : (
            "full_positive_patients == scored_positive_patients: results identical to unguarded version"
            if denominator_same else
            f"{n_missing_after_filter} positive patients dropped by p90+z-continuity filter — denominator corrected"
        ),
        "total_tracks"               : len(track_rows),
        "total_positive_tracks"      : total_positive_tracks,
        "candidate_topk"             : {
            str(k): {
                "patient_hit_rate"           : cand_topk_results[k]["patient_hit_rate"],
                "hit_patient_count"          : cand_topk_results[k]["hit_patient_count"],
                "complete_miss_patient_count": cand_topk_results[k]["complete_miss_patient_count"],
            } for k in TOPK_LIST
        },
        "track_topk"                 : {
            str(k): {
                "patient_hit_rate"           : track_topk_results[k]["patient_hit_rate"],
                "hit_patient_count"          : track_topk_results[k]["hit_patient_count"],
                "complete_miss_patient_count": track_topk_results[k]["complete_miss_patient_count"],
            } for k in TOPK_LIST
        },
        "candidate_rank_stats"       : cand_rank_stats,
        "track_rank_stats"           : track_rank_stats,
        "complete_miss_cand_top20_count" : len(cand_miss_top20),
        "complete_miss_track_top20_count": len(track_miss_top20),
        "baseline_note"              : baseline_note,
        "baseline_data_key"          : "PD_sqrthu_len_track_top3_mean",
        "baseline_caution"           : "denominator 다를 수 있음 (direct comparison caution)",
        "guardrail"                  : guardrail,
        "runtime_s"                  : round(elapsed, 2),
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [SAVED] {SUMMARY_JSON}")

    # ── report.md ─────────────────────────────────────────────────────────────
    s2_cand_top20 = cand_topk_results[20]["patient_hit_rate"]
    s2_track_top20 = track_topk_results[20]["patient_hit_rate"]
    denom_guard_note = (
        "full_positive_patients == scored_positive_patients → denominator 보정 없음 (결과 동일)"
        if denominator_same else
        f"⚠️ {n_missing_after_filter}명 p90+z-cont 필터 탈락 → denominator 보정 적용"
    )
    lines = [
        f"# Step 14 Stage2 Fixed Evaluation Report",
        f"",
        f"**verdict**: {verdict}",
        f"",
        f"## Denominator Guard",
        f"",
        f"| 항목 | 값 |",
        f"|---|---|",
        f"| full_stage2_positive_patients | {total_positive_patients} |",
        f"| scored_positive_patients (p90+z-cont 통과) | {len(scored_positive_patients)} |",
        f"| positive_patient_retention | {positive_patient_retention:.4f} |",
        f"| missing_after_filter | {n_missing_after_filter} |",
        f"| denominator_same_as_unguarded | {denominator_same} |",
        f"",
        f"{denom_guard_note}",
        f"",
        f"## 기본 통계",
        f"",
        f"| 항목 | 값 |",
        f"|---|---|",
        f"| total_candidates | {EXPECTED_ROWS:,} |",
        f"| positive_candidates | {total_positive_cands:,} |",
        f"| total_patients | {total_patients} |",
        f"| positive_patients | {total_positive_patients} |",
        f"| total_tracks | {len(track_rows):,} |",
        f"| positive_tracks | {total_positive_tracks:,} |",
        f"| label_merge_missing | {merge_missing} |",
        f"",
        f"## Candidate-level Top-k  ({PRIMARY_CAND_SCORE})",
        f"",
        f"| k | hit_rate | hit/total | miss |",
        f"|---|---|---|---|",
    ]
    for k in TOPK_LIST:
        r = cand_topk_results[k]
        lines.append(f"| top{k} | {r['patient_hit_rate']:.4f} | "
                     f"{r['hit_patient_count']}/{total_positive_patients} | "
                     f"{r['complete_miss_patient_count']} |")

    lines += [
        f"",
        f"## Track-level Top-k  ({PRIMARY_TRACK_SCORE})",
        f"",
        f"| k | hit_rate | hit/total | miss |",
        f"|---|---|---|---|",
    ]
    for k in TOPK_LIST:
        r = track_topk_results[k]
        lines.append(f"| top{k} | {r['patient_hit_rate']:.4f} | "
                     f"{r['hit_patient_count']}/{r['total_positive_patients']} | "
                     f"{r['complete_miss_patient_count']} |")

    lines += [
        f"",
        f"## Stage1_dev vs Stage2_holdout 비교",
        f"",
        f"| k | stage1 cand | stage2 cand | Δcand | stage1 track | stage2 track | Δtrack |",
        f"|---|---|---|---|---|---|---|",
    ]
    for cr in comparison_rows:
        lines.append(
            f"| top{cr['k']} | {cr['stage1_dev_cand_raw']} | {cr['stage2_holdout_cand_raw']} | "
            f"{cr['cand_delta']} | {cr['stage1_dev_track_raw_top3']} | "
            f"{cr['stage2_holdout_track_raw_top3']} | {cr['track_delta']} |"
        )

    if baseline_data:
        lines += [
            f"",
            f"## Baseline Comparison ({baseline_note})",
            f"",
            f"⚠️ denominator 다를 수 있음 (direct comparison caution)",
            f"",
            f"| k | baseline track_top3 | lung5ch track_top3 |",
            f"|---|---|---|",
        ]
        for k in TOPK_LIST:
            kkey = f"top{k}"
            bl_v = baseline_data.get(kkey, "N/A")
            l5_v = track_topk_results[k]["patient_hit_rate"]
            lines.append(f"| top{k} | {bl_v} | {l5_v} |")

    lines += [
        f"",
        f"## Complete Miss Audit (top20 기준)",
        f"",
        f"- cand_miss@20: {len(cand_miss_top20)}",
        f"- track_miss@20: {len(track_miss_top20)}",
        f"- union: {len(audit_pids)}",
        f"",
        f"## Problem Patient (LUNG1-086/386/399)",
        f"",
    ]
    for pr in prob_rows:
        lines.append(f"- {pr['patient_id']}: {pr.get('note','')}")

    lines += [
        f"",
        f"## Best Positive Rank Stats",
        f"",
        f"candidate: mean={cand_rank_stats['mean']}  median={cand_rank_stats['median']}  p90={cand_rank_stats['p90']}",
        f"track    : mean={track_rank_stats['mean']}  median={track_rank_stats['median']}  p90={track_rank_stats['p90']}",
        f"",
        f"## Guardrail",
        f"",
        f"- score_family_changed: False",
        f"- threshold_tuning: False",
        f"- stage2_label_used_for_tuning: False",
        f"- training/model_forward/checkpoint_saved: False",
        f"",
        f"## Next Step",
        f"",
        f"Step 15: final stage2 interpretation / branch decision report",
    ]
    with open(REPORT_MD, "w") as f:
        f.write("\n".join(lines))
    print(f"  [SAVED] {REPORT_MD}")

    # ── DONE ──────────────────────────────────────────────────────────────────
    done = {
        "step"                       : "step14_stage2_fixed_eval",
        "verdict"                    : verdict,
        "created"                    : str(date.today()),
        "total_positive_patients"    : total_positive_patients,
        "candidate_hit_top20"        : s2_cand_top20,
        "track_hit_top20"            : s2_track_top20,
        "complete_miss_top20"        : len(cand_miss_top20),
        "summary_json"               : str(SUMMARY_JSON),
        "report_md"                  : str(REPORT_MD),
    }
    with open(DONE_OUT, "w") as f:
        json.dump(done, f, indent=2)
    print(f"  [SAVED] {DONE_OUT}")

    # ── 최종 판정 출력 ─────────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print(f"판정: {verdict}")
    print(f"  positive patients  : {total_positive_patients} / {total_patients}")
    print(f"  cand  top20 hit    : {s2_cand_top20:.4f}  ({cand_topk_results[20]['hit_patient_count']}/{total_positive_patients})")
    print(f"  track top20 hit    : {s2_track_top20:.4f}  ({track_topk_results[20]['hit_patient_count']}/{len(positive_track_patients)})")
    print(f"  cand  top50 hit    : {cand_topk_results[50]['patient_hit_rate']:.4f}")
    print(f"  track top50 hit    : {track_topk_results[50]['patient_hit_rate']:.4f}")
    print(f"  complete miss@20   : cand={len(cand_miss_top20)}  track={len(track_miss_top20)}")
    print(f"  runtime            : {elapsed:.1f}s")
    print("=" * 64)

    if verdict.startswith("PASS"):
        print("\nStep 14 완료. 다음 단계: Step 15 final interpretation / branch decision report (사용자 승인 후)")
    else:
        print(f"\nStep 14 {verdict}. 결과 확인 후 다음 단계 결정.")


def main():
    args = parse_args()

    if args.dry_run:
        print_dry_run()
        sys.exit(0)

    if args.run_eval:
        missing = []
        if not args.confirm_step13_pass:
            missing.append("--confirm-step13-pass")
        if not args.confirm_fixed_eval_only:
            missing.append("--confirm-fixed-eval-only")
        if not args.confirm_no_tuning:
            missing.append("--confirm-no-tuning")
        if missing:
            print(f"[BLOCKED] 필수 flags 누락: {missing}", file=sys.stderr)
            sys.exit(2)
        run_eval(args)
        return

    print("[BLOCKED] --dry-run 또는 --run-eval 필요", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
