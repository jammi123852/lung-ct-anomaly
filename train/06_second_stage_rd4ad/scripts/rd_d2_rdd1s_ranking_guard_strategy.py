"""
RD-D2: RD-D1s true RD4AD ranking/guard strategy analysis
목적: RD-D1s score를 삭제 필터가 아닌 ranking/refiner score로 활용 가능한지 분석.

모드:
  bare run        -> exit 2
  --dry-plan      -> 입력 확인 (분석 실행 없음)
  --run-analysis  -> read-only CSV analysis + ranking/top-k guard simulation + DONE
"""

import sys
import csv
import json
import math
import collections
from pathlib import Path

ALLOWED_MODES = {"--dry-plan", "--run-analysis"}
if not any(m in sys.argv for m in ALLOWED_MODES):
    print("오류: 실행 모드 지정 필요.")
    print("  --dry-plan      : 입력 확인 (분석 실행 없음)")
    print("  --run-analysis  : read-only analysis + ranking simulation + DONE")
    sys.exit(2)

IS_DRY_PLAN     = "--dry-plan"     in sys.argv
IS_RUN_ANALYSIS = "--run-analysis" in sys.argv

# ── 경로 설정
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")

RD_D1S_SCORE_CSV = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_d1s_medi3ch_true_rd4ad_shard_run_v1"
    / "rd_d1s_stage1dev_candidate_score.csv"
)
RD_D1S_THRESHOLD_JSON = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_d1s_medi3ch_true_rd4ad_shard_run_v1"
    / "rd_d1s_normal_val_threshold_summary.json"
)
RD_D1S_SWEEP_CSV = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_d1s_medi3ch_true_rd4ad_shard_run_v1"
    / "rd_d1s_safety_constrained_threshold_sweep.csv"
)
RD_D1S_SUMMARY_JSON = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_d1s_medi3ch_true_rd4ad_shard_run_v1"
    / "rd_d1s_medi3ch_true_rd4ad_shard_run_summary.json"
)
RD_C3_SCORE_CSV = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_c3_v1v1_convae_same_universe_retest_v1"
    / "rd_c3_v1v1_convae_candidate_score.csv"
)
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_d2_rdd1s_ranking_guard_strategy_v1"
)

FORBIDDEN_KEYWORDS = [
    "stage2_holdout", "second-stage-lesion-refiner",
    "test_lesion", "lesion_refiner",
]

INPUT_ROWS_EXPECTED = 113447
POSITIVE_EXPECTED   = 35247
HN_EXPECTED         = 78200
RD_D1S_AUROC_REF    = 0.7506
RD_D1S_AUPRC_REF    = 0.5048


def assert_path_safe(p):
    for kw in FORBIDDEN_KEYWORDS:
        if kw.lower() in str(p).lower():
            raise RuntimeError(f"[SAFETY] 금지 경로 접근 차단: {p!r} (keyword={kw!r})")


def fmt_float(x, ndigits=4):
    if x is None:
        return "N/A"
    try:
        if math.isnan(float(x)):
            return "N/A"
    except Exception:
        pass
    return f"{float(x):.{ndigits}f}"


def compute_auroc_mann_whitney(y_true, y_score):
    import numpy as np
    y_true  = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    valid   = np.isfinite(y_score)
    y_true  = y_true[valid]
    y_score = y_score[valid]
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order        = np.argsort(y_score)
    sorted_scores = y_score[order]
    ranks = np.empty(len(sorted_scores), dtype=float)
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[i:j] = avg_rank
        i = j
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    rank_sum_pos = float(original_ranks[y_true == 1].sum())
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def compute_average_precision(y_true, y_score):
    import numpy as np
    y_true  = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    valid   = np.isfinite(y_score)
    y_true  = y_true[valid]
    y_score = y_score[valid]
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return None
    order         = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    score_sorted  = y_score[order]
    tp = 0; fp = 0; prev_recall = 0.0; ap = 0.0; i = 0
    while i < len(score_sorted):
        j = i + 1
        while j < len(score_sorted) and score_sorted[j] == score_sorted[i]:
            j += 1
        group = y_true_sorted[i:j]
        tp += int((group == 1).sum())
        fp += int((group == 0).sum())
        recall    = tp / n_pos
        precision = tp / max(tp + fp, 1)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        i = j
    return float(ap)


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"  saved: {path.name}")


# =============================================================================
# dry-plan
# =============================================================================

def run_dry_plan():
    print("=" * 72)
    print("RD-D2: RD-D1s ranking/guard strategy analysis [DRY-PLAN]")
    print("=" * 72)

    ok_all = True

    print("\n[1] 입력 파일 존재 확인")
    required_inputs = [
        ("RD_D1S_SCORE_CSV",    RD_D1S_SCORE_CSV),
        ("RD_D1S_THRESHOLD_JSON", RD_D1S_THRESHOLD_JSON),
        ("RD_D1S_SWEEP_CSV",    RD_D1S_SWEEP_CSV),
        ("RD_D1S_SUMMARY_JSON", RD_D1S_SUMMARY_JSON),
    ]
    optional_inputs = [
        ("RD_C3_SCORE_CSV",     RD_C3_SCORE_CSV),
    ]
    for label, p in required_inputs:
        assert_path_safe(p)
        exists = p.exists()
        print(f"  {'OK' if exists else 'MISSING'}: {label}")
        if not exists:
            ok_all = False
    for label, p in optional_inputs:
        assert_path_safe(p)
        exists = p.exists()
        print(f"  {'OK(optional)' if exists else 'MISSING(optional)'}: {label}")

    print("\n[2] output root guard")
    if OUTPUT_ROOT.exists():
        print(f"  CONFLICT: OUTPUT_ROOT 이미 존재 -> {OUTPUT_ROOT}")
        ok_all = False
    else:
        print(f"  OK: OUTPUT_ROOT 없음")

    print("\n[3] RD-D1s score CSV row/count 확인")
    d1s_rows = []
    if RD_D1S_SCORE_CSV.exists():
        with open(RD_D1S_SCORE_CSV, newline="", encoding="utf-8") as f:
            d1s_rows = list(csv.DictReader(f))
        n_rows    = len(d1s_rows)
        n_pos     = sum(1 for r in d1s_rows if r.get("label") == "positive")
        n_hn      = sum(1 for r in d1s_rows if r.get("label") == "hard_negative")
        n_nan     = sum(1 for r in d1s_rows if r.get("score_nan","0") == "1")
        n_inf     = sum(1 for r in d1s_rows if r.get("score_inf","0") == "1")
        n_holdout = sum(1 for r in d1s_rows if r.get("stage_split","") == "stage2_holdout")
        ok_rows   = (n_rows == INPUT_ROWS_EXPECTED and n_pos == POSITIVE_EXPECTED
                     and n_hn == HN_EXPECTED and n_nan == 0 and n_inf == 0 and n_holdout == 0)
        print(f"  rows={n_rows:,}  positive={n_pos:,}  hard_negative={n_hn:,}")
        print(f"  score_nan={n_nan}  score_inf={n_inf}  holdout={n_holdout}  {'OK' if ok_rows else 'FAIL'}")
        if not ok_rows:
            ok_all = False

    print("\n[4] RD-C3 join 가능성 확인")
    if RD_C3_SCORE_CSV.exists() and d1s_rows:
        with open(RD_C3_SCORE_CSV, newline="", encoding="utf-8") as f:
            c3_rows = list(csv.DictReader(f))
        c3_ids   = {r.get("candidate_id","") for r in c3_rows}
        d1s_ids  = {r.get("candidate_id","") for r in d1s_rows}
        overlap  = len(c3_ids & d1s_ids)
        has_fs   = "first_stage_score" in (c3_rows[0] if c3_rows else {})
        print(f"  c3_rows={len(c3_rows):,}  overlap={overlap:,}/{len(d1s_rows):,}  "
              f"first_stage_score={has_fs}")
    else:
        print("  RD-C3 없음 또는 d1s 미로드")

    print("\n[5] RD-D1s sweep threshold 확인")
    if RD_D1S_SWEEP_CSV.exists():
        with open(RD_D1S_SWEEP_CSV, newline="", encoding="utf-8") as f:
            for sw in csv.DictReader(f):
                print(f"  le{float(sw['target_lesion_rate'])*100:.0f}%: "
                      f"thr={sw['threshold']}  hn_sup={sw['hn_suppressed_rate']}")

    print()
    verdict = "DRY-PLAN OK" if ok_all else "DRY-PLAN FAIL"
    print(f"판정: {verdict}")
    if ok_all:
        print("  사용자 승인 후:")
        print("  source ~/ai_env/bin/activate && \\")
        print("  python scripts/rd_d2_rdd1s_ranking_guard_strategy.py --run-analysis \\")
        print("  2>&1 | tee /tmp/rd_d2_rdd1s_ranking_guard_strategy_log.txt")
    return ok_all


# =============================================================================
# run_analysis
# =============================================================================

def run_analysis():
    import numpy as np

    print("=" * 72)
    print("RD-D2: RD-D1s ranking/guard strategy analysis [RUN-ANALYSIS]")
    print("=" * 72)

    if OUTPUT_ROOT.exists():
        print(f"[ABORT] OUTPUT_ROOT 이미 존재: {OUTPUT_ROOT}", file=sys.stderr)
        sys.exit(1)
    OUTPUT_ROOT.mkdir(parents=True)

    error_rows = []

    # ── [1/6] 입력 검증
    print("\n[1/6] 입력 검증")
    for p in [RD_D1S_SCORE_CSV, RD_D1S_THRESHOLD_JSON, RD_D1S_SWEEP_CSV, RD_D1S_SUMMARY_JSON]:
        assert_path_safe(p)
        if not p.exists():
            print(f"[ABORT] 입력 없음: {p}", file=sys.stderr)
            sys.exit(1)

    with open(RD_D1S_SCORE_CSV, newline="", encoding="utf-8") as f:
        d1s_rows = list(csv.DictReader(f))

    n_rows    = len(d1s_rows)
    n_pos     = sum(1 for r in d1s_rows if r.get("label") == "positive")
    n_hn      = sum(1 for r in d1s_rows if r.get("label") == "hard_negative")
    n_nan     = sum(1 for r in d1s_rows if r.get("score_nan","0") == "1")
    n_inf     = sum(1 for r in d1s_rows if r.get("score_inf","0") == "1")
    n_holdout = sum(1 for r in d1s_rows if r.get("stage_split","") == "stage2_holdout")
    cids      = [r.get("candidate_id","") for r in d1s_rows]
    n_dup_cid = len(cids) - len(set(cids))
    n_null_pid = sum(1 for r in d1s_rows if not r.get("patient_id","").strip())
    n_null_sid = sum(1 for r in d1s_rows if not r.get("safe_id","").strip())

    labels      = np.array([1 if r["label"] == "positive" else 0 for r in d1s_rows], dtype=int)
    d1s_scores  = np.array([float(r["rd_d1s_medi3ch_rd4ad_score"]) for r in d1s_rows], dtype=float)
    patient_ids = [r["patient_id"] for r in d1s_rows]

    auroc = compute_auroc_mann_whitney(labels.tolist(), d1s_scores.tolist())
    auprc = compute_average_precision(labels.tolist(), d1s_scores.tolist())
    auroc_ok = (auroc is not None and abs(auroc - RD_D1S_AUROC_REF) < 0.001)
    auprc_ok = (auprc is not None and abs(auprc - RD_D1S_AUPRC_REF) < 0.001)

    print(f"  rows={n_rows:,}  positive={n_pos:,}  hard_negative={n_hn:,}")
    print(f"  score_nan={n_nan}  score_inf={n_inf}  holdout={n_holdout}  "
          f"dup_cid={n_dup_cid}  null_pid={n_null_pid}")
    print(f"  AUROC={fmt_float(auroc,4)} (ref={RD_D1S_AUROC_REF}) {'OK' if auroc_ok else 'MISMATCH'}")
    print(f"  AUPRC={fmt_float(auprc,4)} (ref={RD_D1S_AUPRC_REF}) {'OK' if auprc_ok else 'MISMATCH'}")

    val_checks = [
        {"check": "input_rows",         "result": n_rows,        "expected": INPUT_ROWS_EXPECTED, "pass": n_rows == INPUT_ROWS_EXPECTED},
        {"check": "positive_count",     "result": n_pos,         "expected": POSITIVE_EXPECTED,   "pass": n_pos == POSITIVE_EXPECTED},
        {"check": "hard_negative_count","result": n_hn,          "expected": HN_EXPECTED,         "pass": n_hn == HN_EXPECTED},
        {"check": "score_nan",          "result": n_nan,         "expected": 0,                   "pass": n_nan == 0},
        {"check": "score_inf",          "result": n_inf,         "expected": 0,                   "pass": n_inf == 0},
        {"check": "stage2_holdout",     "result": n_holdout,     "expected": 0,                   "pass": n_holdout == 0},
        {"check": "dup_candidate_id",   "result": n_dup_cid,     "expected": 0,                   "pass": n_dup_cid == 0},
        {"check": "null_patient_id",    "result": n_null_pid,    "expected": 0,                   "pass": n_null_pid == 0},
        {"check": "null_safe_id",       "result": n_null_sid,    "expected": 0,                   "pass": n_null_sid == 0},
        {"check": "auroc_reproduced",   "result": fmt_float(auroc,4), "expected": RD_D1S_AUROC_REF, "pass": auroc_ok},
        {"check": "auprc_reproduced",   "result": fmt_float(auprc,4), "expected": RD_D1S_AUPRC_REF, "pass": auprc_ok},
    ]
    write_csv(
        OUTPUT_ROOT / "rd_d2_input_validation.csv",
        ["check", "result", "expected", "pass"],
        val_checks,
    )
    input_ok = all(c["pass"] for c in val_checks)
    if not input_ok:
        print("[ABORT] 입력 검증 실패", file=sys.stderr)
        sys.exit(1)
    print("  입력 검증 PASS")

    # ── RD-C3 join
    print("\n  RD-C3 score join")
    assert_path_safe(RD_C3_SCORE_CSV)
    first_stage_joined = False
    convae_joined      = False
    fs_map  = {}
    cae_map = {}

    if RD_C3_SCORE_CSV.exists():
        with open(RD_C3_SCORE_CSV, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                cid = r.get("candidate_id","")
                fs  = r.get("first_stage_score","")
                ca  = r.get("convAE_crop_score_l1_mean","")
                try:
                    if cid and fs:
                        fs_map[cid] = float(fs)
                except ValueError:
                    pass
                try:
                    if cid and ca:
                        cae_map[cid] = float(ca)
                except ValueError:
                    pass
        fs_overlap  = sum(1 for r in d1s_rows if r.get("candidate_id","") in fs_map)
        cae_overlap = sum(1 for r in d1s_rows if r.get("candidate_id","") in cae_map)
        first_stage_joined = (fs_overlap == n_rows)
        convae_joined      = (cae_overlap == n_rows)
        print(f"  first_stage_score joined={fs_overlap:,}/{n_rows:,} ({first_stage_joined})")
        print(f"  convAE_score joined={cae_overlap:,}/{n_rows:,} ({convae_joined})")
    else:
        print("  RD-C3 없음 - join 불가")

    fs_scores  = np.full(n_rows, np.nan, dtype=float)
    cae_scores = np.full(n_rows, np.nan, dtype=float)
    for i, r in enumerate(d1s_rows):
        cid = r.get("candidate_id","")
        if cid in fs_map:
            fs_scores[i] = fs_map[cid]
        if cid in cae_map:
            cae_scores[i] = cae_map[cid]

    # patient 그룹
    pat_groups = collections.defaultdict(list)
    for i, r in enumerate(d1s_rows):
        pat_groups[r["patient_id"]].append(i)

    # sweep threshold 로드
    with open(RD_D1S_SWEEP_CSV, newline="", encoding="utf-8") as f:
        sweep_ref = {float(r["target_lesion_rate"]): float(r["threshold"])
                     for r in csv.DictReader(f)}
    with open(RD_D1S_THRESHOLD_JSON, encoding="utf-8") as f:
        thr_json = json.load(f)
    global_p95 = float(thr_json["global_p95"])
    global_p99 = float(thr_json["global_p99"])

    # ── [2/6] patient-level 위험 환자
    print("\n[2/6] patient-level 위험 환자 확인")
    risk_rows = []
    for target_rate in [0.01, 0.03, 0.05]:
        thr = sweep_ref.get(target_rate)
        if thr is None:
            continue
        for pid, idxs in pat_groups.items():
            pos_idxs = [i for i in idxs if labels[i] == 1]
            if not pos_idxs:
                continue
            if not all(d1s_scores[i] <= thr for i in pos_idxs):
                continue
            pos_sc = [d1s_scores[i] for i in pos_idxs]
            fs_vals = [fs_scores[i] for i in pos_idxs if np.isfinite(fs_scores[i])]
            sorted_pat = sorted(idxs, key=lambda i: -d1s_scores[i])
            rank_map = {idx: rk + 1 for rk, idx in enumerate(sorted_pat)}
            best_pos_rank = min(rank_map[i] for i in pos_idxs)
            risk_rows.append({
                "target_lesion_rate":     target_rate,
                "threshold":              round(thr, 6),
                "patient_id":             pid,
                "positive_count":         len(pos_idxs),
                "total_candidates":       len(idxs),
                "pos_score_min":          round(float(np.min(pos_sc)), 6),
                "pos_score_max":          round(float(np.max(pos_sc)), 6),
                "pos_score_mean":         round(float(np.mean(pos_sc)), 6),
                "fs_score_min":           round(float(np.min(fs_vals)), 6) if fs_vals else "",
                "fs_score_max":           round(float(np.max(fs_vals)), 6) if fs_vals else "",
                "best_pos_rank_in_patient": best_pos_rank,
            })
    write_csv(
        OUTPUT_ROOT / "rd_d2_patient_all_suppressed_risk_cases.csv",
        ["target_lesion_rate", "threshold", "patient_id", "positive_count",
         "total_candidates", "pos_score_min", "pos_score_max", "pos_score_mean",
         "fs_score_min", "fs_score_max", "best_pos_rank_in_patient"],
        risk_rows,
    )
    risk_le1 = sum(1 for r in risk_rows if r["target_lesion_rate"] == 0.01)
    risk_le3 = sum(1 for r in risk_rows if r["target_lesion_rate"] == 0.03)
    risk_le5 = sum(1 for r in risk_rows if r["target_lesion_rate"] == 0.05)
    print(f"  pat_all_sup: ≤1%={risk_le1}명  ≤3%={risk_le3}명  ≤5%={risk_le5}명")

    # ── [3/6] patient top-k guard simulation
    print("\n[3/6] patient top-k guard simulation")

    thr_candidates = []
    for rate in sorted(sweep_ref):
        thr_candidates.append({"name": f"sweep_le{int(rate*100)}pct", "threshold": sweep_ref[rate]})
    for pct in [1, 3, 5, 10, 15]:
        thr_candidates.append({"name": f"d1s_p{pct}", "threshold": float(np.percentile(d1s_scores, pct))})
    thr_candidates.append({"name": "global_p95", "threshold": global_p95})
    thr_candidates.append({"name": "global_p99", "threshold": global_p99})

    guard_rows = []
    for tc in thr_candidates:
        thr_name = tc["name"]
        thr_val  = tc["threshold"]
        for topk in [1, 3, 5, 10]:
            pat_topk = {pid: set(sorted(idxs, key=lambda i: -d1s_scores[i])[:topk])
                        for pid, idxs in pat_groups.items()}

            sup = np.array([
                d1s_scores[i] <= thr_val and i not in pat_topk.get(patient_ids[i], set())
                for i in range(n_rows)
            ])
            total_rd = int(sup.sum())
            pos_rd   = int((sup & (labels == 1)).sum())
            hn_rd    = int((sup & (labels == 0)).sum())
            pos_rate = pos_rd / n_pos if n_pos > 0 else 0.0
            hn_rate  = hn_rd  / n_hn  if n_hn  > 0 else 0.0

            all_rd_pats    = 0
            at_least1_kept = 0
            guard_trigger  = 0
            for pid, idxs in pat_groups.items():
                pos_idxs = [i for i in idxs if labels[i] == 1]
                if not pos_idxs:
                    continue
                if all(sup[i] for i in pos_idxs):
                    all_rd_pats += 1
                else:
                    at_least1_kept += 1
                for i in pos_idxs:
                    if d1s_scores[i] <= thr_val and i in pat_topk.get(pid, set()):
                        guard_trigger += 1
                        break

            safety_pass = (all_rd_pats == 0 and pos_rate <= 0.03 and hn_rate > 0.05)
            guard_rows.append({
                "threshold_name":                     thr_name,
                "threshold_value":                    round(thr_val, 6),
                "topk":                               topk,
                "rank_down_candidate_count":          total_rd,
                "positive_rank_down_count":           pos_rd,
                "positive_rank_down_rate":            round(pos_rate, 4),
                "hard_negative_rank_down_count":      hn_rd,
                "hard_negative_rank_down_rate":       round(hn_rate, 4),
                "lesion_patient_all_rank_down_count": all_rd_pats,
                "lesion_patient_at_least_one_kept_count": at_least1_kept,
                "patient_topk_guard_trigger_count":   guard_trigger,
                "ranking_safety_pass":                safety_pass,
            })

    write_csv(
        OUTPUT_ROOT / "rd_d2_threshold_topk_guard_simulation.csv",
        ["threshold_name", "threshold_value", "topk",
         "rank_down_candidate_count", "positive_rank_down_count", "positive_rank_down_rate",
         "hard_negative_rank_down_count", "hard_negative_rank_down_rate",
         "lesion_patient_all_rank_down_count", "lesion_patient_at_least_one_kept_count",
         "patient_topk_guard_trigger_count", "ranking_safety_pass"],
        guard_rows,
    )

    pass_rows  = [r for r in guard_rows if r["ranking_safety_pass"]]
    best_guard = max(pass_rows, key=lambda r: r["hard_negative_rank_down_rate"]) if pass_rows else None
    print(f"  safety_pass: {len(pass_rows)}/{len(guard_rows)} 조합")
    if best_guard:
        print(f"  best: {best_guard['threshold_name']} topk={best_guard['topk']} "
              f"hn_rd={best_guard['hard_negative_rank_down_rate']:.2%} "
              f"pos_rd={best_guard['positive_rank_down_rate']:.2%} "
              f"all_rd_pats={best_guard['lesion_patient_all_rank_down_count']}")

    # ── [4/6] first-stage score guard simulation
    print("\n[4/6] first-stage score guard simulation")
    fs_guard_rows = []

    if first_stage_joined:
        fs_filled = fs_scores.copy()
        fs_filled[~np.isfinite(fs_filled)] = float(np.nanmedian(fs_filled))

        sweep_small = [t for t in thr_candidates
                       if t["name"] in {"sweep_le1pct", "sweep_le3pct", "sweep_le5pct",
                                        "d1s_p1", "d1s_p3", "d1s_p5"}]
        for tc in sweep_small:
            thr_name = tc["name"]
            thr_val  = tc["threshold"]
            low_mask = d1s_scores <= thr_val

            # patient fs top-k guard
            for fs_topk in [1, 3]:
                pat_fs_topk = {pid: set(sorted(idxs, key=lambda i: -fs_filled[i])[:fs_topk])
                               for pid, idxs in pat_groups.items()}
                sup = np.array([
                    low_mask[i] and i not in pat_fs_topk.get(patient_ids[i], set())
                    for i in range(n_rows)
                ])
                total_rd = int(sup.sum())
                pos_rd   = int((sup & (labels == 1)).sum())
                hn_rd    = int((sup & (labels == 0)).sum())
                pos_rate = pos_rd / n_pos if n_pos > 0 else 0.0
                hn_rate  = hn_rd  / n_hn  if n_hn  > 0 else 0.0
                all_rd_pats = sum(
                    1 for pid, idxs in pat_groups.items()
                    if any(labels[i] == 1 for i in idxs)
                    and all(sup[i] for i in idxs if labels[i] == 1)
                )
                safety_pass = (all_rd_pats == 0 and pos_rate <= 0.03 and hn_rate > 0.05)
                fs_guard_rows.append({
                    "guard_type": f"fs_patient_top{fs_topk}_keep",
                    "d1s_threshold_name": thr_name,
                    "d1s_threshold":      round(thr_val, 6),
                    "rank_down_count":    total_rd,
                    "positive_rank_down_count": pos_rd,
                    "positive_rank_down_rate":  round(pos_rate, 4),
                    "hn_rank_down_count": hn_rd,
                    "hn_rank_down_rate":  round(hn_rate, 4),
                    "lesion_patient_all_rank_down": all_rd_pats,
                    "ranking_safety_pass": safety_pass,
                })

            # global fs top-pct% guard
            for global_pct in [5, 10]:
                n_keep = max(1, int(n_rows * global_pct / 100))
                keep_set = set(np.argsort(-fs_filled)[:n_keep].tolist())
                sup = np.array([
                    low_mask[i] and i not in keep_set
                    for i in range(n_rows)
                ])
                total_rd = int(sup.sum())
                pos_rd   = int((sup & (labels == 1)).sum())
                hn_rd    = int((sup & (labels == 0)).sum())
                pos_rate = pos_rd / n_pos if n_pos > 0 else 0.0
                hn_rate  = hn_rd  / n_hn  if n_hn  > 0 else 0.0
                all_rd_pats = sum(
                    1 for pid, idxs in pat_groups.items()
                    if any(labels[i] == 1 for i in idxs)
                    and all(sup[i] for i in idxs if labels[i] == 1)
                )
                safety_pass = (all_rd_pats == 0 and pos_rate <= 0.03 and hn_rate > 0.05)
                fs_guard_rows.append({
                    "guard_type": f"fs_global_top{global_pct}pct_keep",
                    "d1s_threshold_name": thr_name,
                    "d1s_threshold":      round(thr_val, 6),
                    "rank_down_count":    total_rd,
                    "positive_rank_down_count": pos_rd,
                    "positive_rank_down_rate":  round(pos_rate, 4),
                    "hn_rank_down_count": hn_rd,
                    "hn_rank_down_rate":  round(hn_rate, 4),
                    "lesion_patient_all_rank_down": all_rd_pats,
                    "ranking_safety_pass": safety_pass,
                })

        # combined: d1s top-k + fs top-k guard
        for tc in [t for t in thr_candidates if t["name"] in {"sweep_le1pct", "sweep_le3pct"}]:
            thr_name = tc["name"]
            thr_val  = tc["threshold"]
            low_mask = d1s_scores <= thr_val
            for d1s_topk in [1, 3]:
                pat_d1s_tk = {pid: set(sorted(idxs, key=lambda i: -d1s_scores[i])[:d1s_topk])
                              for pid, idxs in pat_groups.items()}
                for fs_topk in [1, 3]:
                    pat_fs_tk = {pid: set(sorted(idxs, key=lambda i: -fs_filled[i])[:fs_topk])
                                 for pid, idxs in pat_groups.items()}
                    sup = np.array([
                        low_mask[i]
                        and i not in pat_d1s_tk.get(patient_ids[i], set())
                        and i not in pat_fs_tk.get(patient_ids[i], set())
                        for i in range(n_rows)
                    ])
                    total_rd = int(sup.sum())
                    pos_rd   = int((sup & (labels == 1)).sum())
                    hn_rd    = int((sup & (labels == 0)).sum())
                    pos_rate = pos_rd / n_pos if n_pos > 0 else 0.0
                    hn_rate  = hn_rd  / n_hn  if n_hn  > 0 else 0.0
                    all_rd_pats = sum(
                        1 for pid, idxs in pat_groups.items()
                        if any(labels[i] == 1 for i in idxs)
                        and all(sup[i] for i in idxs if labels[i] == 1)
                    )
                    safety_pass = (all_rd_pats == 0 and pos_rate <= 0.03 and hn_rate > 0.05)
                    fs_guard_rows.append({
                        "guard_type": f"combined_d1s_top{d1s_topk}+fs_top{fs_topk}",
                        "d1s_threshold_name": thr_name,
                        "d1s_threshold":      round(thr_val, 6),
                        "rank_down_count":    total_rd,
                        "positive_rank_down_count": pos_rd,
                        "positive_rank_down_rate":  round(pos_rate, 4),
                        "hn_rank_down_count": hn_rd,
                        "hn_rank_down_rate":  round(hn_rate, 4),
                        "lesion_patient_all_rank_down": all_rd_pats,
                        "ranking_safety_pass": safety_pass,
                    })
    else:
        fs_guard_rows.append({
            "guard_type": "skipped_no_first_stage_join",
            "d1s_threshold_name": "N/A", "d1s_threshold": "",
            "rank_down_count": "", "positive_rank_down_count": "",
            "positive_rank_down_rate": "", "hn_rank_down_count": "",
            "hn_rank_down_rate": "", "lesion_patient_all_rank_down": "",
            "ranking_safety_pass": "",
        })

    write_csv(
        OUTPUT_ROOT / "rd_d2_first_stage_guard_simulation.csv",
        ["guard_type", "d1s_threshold_name", "d1s_threshold",
         "rank_down_count", "positive_rank_down_count", "positive_rank_down_rate",
         "hn_rank_down_count", "hn_rank_down_rate",
         "lesion_patient_all_rank_down", "ranking_safety_pass"],
        fs_guard_rows,
    )
    fs_pass_rows = [r for r in fs_guard_rows if r.get("ranking_safety_pass") is True]
    print(f"  fs guard safety_pass: {len(fs_pass_rows)}/{len(fs_guard_rows)}")

    # ── [5/6] ranking fusion simulation
    print("\n[5/6] ranking fusion simulation")

    fusion_rows       = []
    pat_retention_rows = []
    recommended_alpha = None
    best_fusion_auroc_val = -1.0

    if first_stage_joined:
        def robust_z(arr):
            valid = arr[np.isfinite(arr)]
            if len(valid) == 0:
                return arr.copy()
            med = float(np.median(valid))
            q75 = float(np.percentile(valid, 75))
            q25 = float(np.percentile(valid, 25))
            iqr = q75 - q25
            if iqr < 1e-9:
                return arr.copy()
            return (arr - med) / iqr

        fs_norm  = robust_z(fs_scores)
        d1s_norm = robust_z(d1s_scores)

        fs_z  = fs_norm.copy();  fs_z[~np.isfinite(fs_z)]   = float(np.nanmedian(fs_z))
        d1s_z = d1s_norm.copy(); d1s_z[~np.isfinite(d1s_z)] = float(np.nanmedian(d1s_z))

        for alpha in [0.0, 0.1, 0.2, 0.3, 0.5]:
            final_sc = fs_z + alpha * d1s_z
            fa_auroc = compute_auroc_mann_whitney(labels.tolist(), final_sc.tolist())
            fa_auprc = compute_average_precision(labels.tolist(), final_sc.tolist())

            topk_local = []
            for k in [1, 3, 5]:
                pos_in_topk = 0; hn_in_topk = 0; total_pos_pats = 0
                for pid, idxs in pat_groups.items():
                    pos_idxs = [i for i in idxs if labels[i] == 1]
                    if not pos_idxs:
                        continue
                    total_pos_pats += 1
                    topk_set = set(sorted(idxs, key=lambda i: -final_sc[i])[:k])
                    if any(i in topk_set for i in pos_idxs):
                        pos_in_topk += 1
                    hn_in_topk += sum(1 for i in idxs if labels[i] == 0 and i in topk_set)
                topk_local.append({
                    "alpha": alpha, "topk": k,
                    "pos_patient_at_least1_in_topk": pos_in_topk,
                    "total_pos_patients":           total_pos_pats,
                    "pos_topk_retention": round(pos_in_topk / total_pos_pats, 4) if total_pos_pats > 0 else 0,
                    "hn_in_topk_total": hn_in_topk,
                })
            pat_retention_rows.extend(topk_local)

            top1_ret = next((r["pos_topk_retention"] for r in topk_local if r["topk"] == 1), 0)
            top3_ret = next((r["pos_topk_retention"] for r in topk_local if r["topk"] == 3), 0)

            fusion_rows.append({
                "alpha":               alpha,
                "formula":             f"fs_norm + {alpha} * d1s_norm" if alpha > 0 else "fs_norm",
                "auroc":               round(fa_auroc, 4) if fa_auroc else "N/A",
                "auprc":               round(fa_auprc, 4) if fa_auprc else "N/A",
                "pos_top1_retention":  round(top1_ret, 4),
                "pos_top3_retention":  round(top3_ret, 4),
            })

            if fa_auroc is not None and top1_ret >= 0.99:
                if fa_auroc > best_fusion_auroc_val:
                    best_fusion_auroc_val = fa_auroc
                    recommended_alpha = alpha

        if recommended_alpha is None:
            for fr in fusion_rows:
                if (fr["pos_top1_retention"] != "N/A"
                        and fr["pos_top1_retention"] >= 0.95
                        and fr["auroc"] != "N/A"
                        and float(fr["auroc"]) > best_fusion_auroc_val):
                    best_fusion_auroc_val = float(fr["auroc"])
                    recommended_alpha = fr["alpha"]
    else:
        fusion_rows.append({
            "alpha": "N/A", "formula": "skipped_no_first_stage_score",
            "auroc": "N/A", "auprc": "N/A",
            "pos_top1_retention": "N/A", "pos_top3_retention": "N/A",
        })

    write_csv(
        OUTPUT_ROOT / "rd_d2_ranking_fusion_simulation.csv",
        ["alpha", "formula", "auroc", "auprc", "pos_top1_retention", "pos_top3_retention"],
        fusion_rows,
    )
    write_csv(
        OUTPUT_ROOT / "rd_d2_patient_level_topk_retention_summary.csv",
        ["alpha", "topk", "pos_patient_at_least1_in_topk", "total_pos_patients",
         "pos_topk_retention", "hn_in_topk_total"],
        pat_retention_rows,
    )
    print(f"  recommended_alpha={recommended_alpha}  "
          f"best_fusion_auroc={fmt_float(best_fusion_auroc_val,4)}")
    for fr in fusion_rows:
        print(f"    alpha={fr['alpha']}: AUROC={fr['auroc']} AUPRC={fr['auprc']} "
              f"top1_ret={fr['pos_top1_retention']} top3_ret={fr['pos_top3_retention']}")

    # ── [6/6] 최종 추천
    print("\n[6/6] 최종 추천")

    lesion_rd_min = min((r["lesion_patient_all_rank_down_count"] for r in guard_rows), default=999)

    best_guard_name = best_guard["threshold_name"] if best_guard else "none"
    best_guard_topk = best_guard["topk"] if best_guard else None
    best_hn_rd      = best_guard["hard_negative_rank_down_rate"] if best_guard else 0.0
    best_pos_rd     = best_guard["positive_rank_down_rate"] if best_guard else 0.0

    best_hn_le1 = next(
        (r["hard_negative_rank_down_rate"] for r in guard_rows
         if "le1pct" in r["threshold_name"]
         and r["topk"] == (best_guard_topk or 3)
         and r["lesion_patient_all_rank_down_count"] == 0), None
    )
    best_hn_le3 = next(
        (r["hard_negative_rank_down_rate"] for r in guard_rows
         if "le3pct" in r["threshold_name"]
         and r["topk"] == (best_guard_topk or 3)
         and r["lesion_patient_all_rank_down_count"] == 0), None
    )

    alpha0_auroc = next(
        (float(fr["auroc"]) for fr in fusion_rows
         if fr["alpha"] == 0.0 and fr["auroc"] != "N/A"), None
    )
    fusion_better = (
        best_fusion_auroc_val > 0
        and alpha0_auroc is not None
        and best_fusion_auroc_val > alpha0_auroc
    )

    if best_guard is not None and recommended_alpha is not None:
        if lesion_rd_min == 0 and best_pos_rd <= 0.01 and fusion_better:
            final_decision = "RD4AD_RANKING_READY"
        elif lesion_rd_min == 0:
            final_decision = "RD4AD_RANKING_WITH_CAUTION"
        else:
            final_decision = "RD4AD_ANALYSIS_ONLY"
    elif best_guard is not None:
        final_decision = "RD4AD_RANKING_WITH_CAUTION" if lesion_rd_min == 0 else "RD4AD_ANALYSIS_ONLY"
    else:
        final_decision = "RD4AD_ANALYSIS_ONLY"

    print(f"  final_decision : {final_decision}")
    print(f"  best_guard: {best_guard_name} topk={best_guard_topk} "
          f"hn_rd={fmt_float(best_hn_rd,4)} pos_rd={fmt_float(best_pos_rd,4)}")
    print(f"  lesion_patient_all_rank_down_min: {lesion_rd_min}")

    rec_formula = (
        f"fs_norm + {recommended_alpha} * d1s_norm"
        if recommended_alpha is not None else "fs_norm"
    )

    write_csv(
        OUTPUT_ROOT / "rd_d2_recommended_strategy.csv",
        ["strategy", "threshold_name", "threshold_value", "topk",
         "positive_rank_down_rate", "hn_rank_down_rate",
         "lesion_patient_all_rank_down",
         "recommended_alpha", "recommended_formula", "final_decision"],
        [{
            "strategy":                   best_guard_name,
            "threshold_name":             best_guard_name,
            "threshold_value":            best_guard["threshold_value"] if best_guard else "",
            "topk":                       best_guard_topk,
            "positive_rank_down_rate":    best_pos_rd,
            "hn_rank_down_rate":          best_hn_rd,
            "lesion_patient_all_rank_down": lesion_rd_min,
            "recommended_alpha":          recommended_alpha,
            "recommended_formula":        rec_formula,
            "final_decision":             final_decision,
        }],
    )

    write_csv(OUTPUT_ROOT / "rd_d2_errors.csv", ["step", "detail"], error_rows)

    # ── summary JSON
    all_checks_passed = (
        input_ok
        and n_rows    == INPUT_ROWS_EXPECTED
        and n_pos     == POSITIVE_EXPECTED
        and n_hn      == HN_EXPECTED
        and n_nan     == 0
        and n_inf     == 0
        and n_holdout == 0
        and auroc_ok
        and auprc_ok
    )

    summary = {
        "input_rows":                              n_rows,
        "positive_count":                          n_pos,
        "hard_negative_count":                     n_hn,
        "rd_d1s_auroc_recomputed":                 round(auroc, 4) if auroc else None,
        "rd_d1s_auprc_recomputed":                 round(auprc, 4) if auprc else None,
        "first_stage_score_joined":                first_stage_joined,
        "convAE_score_joined":                     convae_joined,
        "best_topk_guard":                         f"{best_guard_name}_top{best_guard_topk}" if best_guard else "none",
        "best_threshold_strategy":                 best_guard_name,
        "best_hn_rank_down_rate_at_positive_le1pct": round(best_hn_le1, 4) if best_hn_le1 is not None else None,
        "best_hn_rank_down_rate_at_positive_le3pct": round(best_hn_le3, 4) if best_hn_le3 is not None else None,
        "lesion_patient_all_rank_down_min":        lesion_rd_min,
        "recommended_alpha":                       recommended_alpha,
        "recommended_final_score_formula":         rec_formula,
        "final_decision":                          final_decision,
        "suppression_applied":                     False,
        "threshold_applied":                       False,
        "candidate_deleted":                       False,
        "first_stage_score_modified":              False,
        "model_forward_run":                       False,
        "training_run":                            False,
        "stage2_holdout_access":                   0,
        "all_checks_passed":                       all_checks_passed,
    }
    with open(OUTPUT_ROOT / "rd_d2_rdd1s_ranking_guard_strategy_summary.json",
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print("  saved: rd_d2_rdd1s_ranking_guard_strategy_summary.json")

    # ── report MD
    verdict_str = "PASS" if all_checks_passed else "FAIL"
    md_lines = [
        "# RD-D2: RD-D1s true RD4AD Ranking/Guard Strategy Analysis",
        "",
        f"## 판정: {verdict_str} / {final_decision}",
        "",
        "## 입력 검증",
        f"- rows={n_rows:,}  positive={n_pos:,}  hard_negative={n_hn:,}",
        f"- AUROC={fmt_float(auroc,4)} (ref={RD_D1S_AUROC_REF}) {'OK' if auroc_ok else 'MISMATCH'}",
        f"- AUPRC={fmt_float(auprc,4)} (ref={RD_D1S_AUPRC_REF}) {'OK' if auprc_ok else 'MISMATCH'}",
        f"- first_stage_score_joined={first_stage_joined}  convAE_joined={convae_joined}",
        "",
        "## 위험 환자 (pat_all_sup)",
        f"- ≤1%: {risk_le1}명  ≤3%: {risk_le3}명  ≤5%: {risk_le5}명",
        "",
        "## Top-k guard simulation",
        f"- safety_pass 조합: {len(pass_rows)}/{len(guard_rows)}",
    ]
    if best_guard:
        md_lines.append(
            f"- best: {best_guard_name} topk={best_guard_topk} "
            f"hn_rd={fmt_float(best_hn_rd,4)} pos_rd={fmt_float(best_pos_rd,4)}"
        )
    md_lines += [
        "",
        "## Ranking fusion",
    ]
    for fr in fusion_rows:
        if fr["alpha"] != "N/A":
            md_lines.append(
                f"- alpha={fr['alpha']}: AUROC={fr['auroc']} AUPRC={fr['auprc']} "
                f"top1={fr['pos_top1_retention']} top3={fr['pos_top3_retention']}"
            )
    md_lines += [
        f"- recommended_alpha={recommended_alpha}",
        f"- formula: {rec_formula}",
        "",
        f"## all_checks_passed: {all_checks_passed}",
    ]
    with open(OUTPUT_ROOT / "rd_d2_rdd1s_ranking_guard_strategy_report.md",
              "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print("  saved: rd_d2_rdd1s_ranking_guard_strategy_report.md")

    # DONE marker
    if all_checks_passed:
        (OUTPUT_ROOT / "DONE").write_text("DONE\n", encoding="utf-8")
        print("\n[DONE] all_checks_passed:", all_checks_passed)
    else:
        (OUTPUT_ROOT / "FAILED").write_text(
            f"FAILED\nall_checks_passed={all_checks_passed}\nfinal_decision={final_decision}\n",
            encoding="utf-8",
        )
        print("\n[FAILED] all_checks_passed:", all_checks_passed, file=sys.stderr)
        sys.exit(1)


# =============================================================================
# entry point
# =============================================================================
if IS_DRY_PLAN:
    run_dry_plan()
elif IS_RUN_ANALYSIS:
    run_analysis()
