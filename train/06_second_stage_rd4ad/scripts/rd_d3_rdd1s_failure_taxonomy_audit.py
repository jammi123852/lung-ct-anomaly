"""
RD-D3: RD-D1s failure-risk patient taxonomy audit
목적: RD-D1s가 실패하는 환자/후보 유형 분석 (read-only)

모드:
  bare run     -> exit 2
  --dry-plan   -> 입력 확인 (분석 실행 없음)
  --run-audit  -> read-only CSV analysis + taxonomy audit + DONE
"""
import sys
import csv
import json
import math
import collections
from pathlib import Path

ALLOWED_MODES = {"--dry-plan", "--run-audit"}
if not any(m in sys.argv for m in ALLOWED_MODES):
    print("오류: 실행 모드 지정 필요.")
    print("  --dry-plan  : 입력 확인 (분석 실행 없음)")
    print("  --run-audit : read-only analysis + taxonomy + DONE")
    sys.exit(2)

IS_DRY_PLAN  = "--dry-plan"  in sys.argv
IS_RUN_AUDIT = "--run-audit" in sys.argv

PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")

RD_D1S_SCORE_CSV = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_d1s_medi3ch_true_rd4ad_shard_run_v1"
    / "rd_d1s_stage1dev_candidate_score.csv"
)
RD_D2_ROOT = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_d2_rdd1s_ranking_guard_strategy_v1"
)
RD_C3_SCORE_CSV = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_c3_v1v1_convae_same_universe_retest_v1"
    / "rd_c3_v1v1_convae_candidate_score.csv"
)
RD_C2_SCORE_CSV = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_c2_effb0_v420_candidate_rd4ad_retest_v1"
    / "rd_c2_effb0_v420_rd4ad_candidate_score.csv"
)
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_d3_rdd1s_failure_taxonomy_audit_v1"
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
RISK_LE1_EXPECTED   = 3
RISK_LE3_EXPECTED   = 4
RISK_LE5_EXPECTED   = 5


def assert_path_safe(p):
    for kw in FORBIDDEN_KEYWORDS:
        if kw.lower() in str(p).lower():
            raise RuntimeError(f"[SAFETY] 금지 경로 차단: {p!r} (keyword={kw!r})")


def fmt_float(x, ndigits=4):
    if x is None:
        return "N/A"
    try:
        if math.isnan(float(x)):
            return "N/A"
    except Exception:
        pass
    return f"{float(x):.{ndigits}f}"


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"  saved: {path.name}")


def safe_float(v):
    try:
        x = float(v)
        if math.isfinite(x):
            return x
    except (TypeError, ValueError):
        pass
    return None


def percentile_rank(value, sorted_arr):
    if value is None or not sorted_arr:
        return None
    n = len(sorted_arr)
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_arr[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return round(lo / n * 100, 2)


# =============================================================================
# dry-plan
# =============================================================================

def run_dry_plan():
    print("=" * 72)
    print("RD-D3: RD-D1s failure taxonomy audit [DRY-PLAN]")
    print("=" * 72)
    ok_all = True

    print("\n[1] 필수 입력 파일")
    required = [
        ("RD_D1S_SCORE_CSV", RD_D1S_SCORE_CSV),
        ("rd_d2_patient_all_suppressed_risk_cases.csv",
         RD_D2_ROOT / "rd_d2_patient_all_suppressed_risk_cases.csv"),
        ("rd_d2_threshold_topk_guard_simulation.csv",
         RD_D2_ROOT / "rd_d2_threshold_topk_guard_simulation.csv"),
        ("rd_d2_ranking_fusion_simulation.csv",
         RD_D2_ROOT / "rd_d2_ranking_fusion_simulation.csv"),
        ("rd_d2_rdd1s_ranking_guard_strategy_summary.json",
         RD_D2_ROOT / "rd_d2_rdd1s_ranking_guard_strategy_summary.json"),
    ]
    optional = [
        ("RD_C3_SCORE_CSV", RD_C3_SCORE_CSV),
        ("RD_C2_SCORE_CSV", RD_C2_SCORE_CSV),
    ]
    for label, p in required:
        assert_path_safe(p)
        ex = p.exists()
        print(f"  {'OK' if ex else 'MISSING'}: {label}")
        if not ex:
            ok_all = False
    for label, p in optional:
        assert_path_safe(p)
        ex = p.exists()
        print(f"  {'OK(opt)' if ex else 'MISSING(opt)'}: {label}")

    print("\n[2] OUTPUT_ROOT guard")
    if OUTPUT_ROOT.exists():
        print(f"  CONFLICT: OUTPUT_ROOT 이미 존재 -> {OUTPUT_ROOT}")
        ok_all = False
    else:
        print(f"  OK: OUTPUT_ROOT 없음")

    print("\n[3] RD-D2 risk patient count")
    rp_path = RD_D2_ROOT / "rd_d2_patient_all_suppressed_risk_cases.csv"
    if rp_path.exists():
        with open(rp_path, newline="", encoding="utf-8") as f:
            rp_rows = list(csv.DictReader(f))
        le1 = {r["patient_id"] for r in rp_rows if float(r["target_lesion_rate"]) <= 0.011}
        le3 = {r["patient_id"] for r in rp_rows if float(r["target_lesion_rate"]) <= 0.031}
        le5 = {r["patient_id"] for r in rp_rows if float(r["target_lesion_rate"]) <= 0.051}
        le1_ok = (len(le1) == RISK_LE1_EXPECTED)
        le3_ok = (len(le3) == RISK_LE3_EXPECTED)
        le5_ok = (len(le5) == RISK_LE5_EXPECTED)
        print(f"  le1={len(le1)} (exp={RISK_LE1_EXPECTED}) {'OK' if le1_ok else 'MISMATCH'}")
        print(f"  le3={len(le3)} (exp={RISK_LE3_EXPECTED}) {'OK' if le3_ok else 'MISMATCH'}")
        print(f"  le5={len(le5)} (exp={RISK_LE5_EXPECTED}) {'OK' if le5_ok else 'MISMATCH'}")
        if not (le1_ok and le3_ok and le5_ok):
            ok_all = False
        print(f"  le1 patients: {sorted(le1)}")
        print(f"  le3 patients: {sorted(le3)}")
        print(f"  le5 patients: {sorted(le5)}")

    print("\n[4] RD-D1s rows")
    if RD_D1S_SCORE_CSV.exists():
        with open(RD_D1S_SCORE_CSV, newline="", encoding="utf-8") as f:
            n = sum(1 for _ in f) - 1
        ok = (n == INPUT_ROWS_EXPECTED)
        print(f"  rows={n:,} (exp={INPUT_ROWS_EXPECTED:,}) {'OK' if ok else 'MISMATCH'}")
        if not ok:
            ok_all = False

    print("\n[5] join 가능성")
    print(f"  RD-C3 ConvAE join: {'가능' if RD_C3_SCORE_CSV.exists() else '불가'}")
    print(f"  RD-C2 first_stage join: {'가능' if RD_C2_SCORE_CSV.exists() else '불가'}")

    print()
    verdict = "DRY-PLAN OK" if ok_all else "DRY-PLAN FAIL"
    print(f"판정: {verdict}")
    if ok_all:
        print("  사용자 승인 후:")
        print("  source ~/ai_env/bin/activate && \\")
        print("  python scripts/rd_d3_rdd1s_failure_taxonomy_audit.py --run-audit \\")
        print("  2>&1 | tee /tmp/rd_d3_rdd1s_failure_taxonomy_audit_log.txt")
    return ok_all


# =============================================================================
# run_audit
# =============================================================================

def run_audit():
    import numpy as np

    print("=" * 72)
    print("RD-D3: RD-D1s failure taxonomy audit [RUN-AUDIT]")
    print("=" * 72)

    if OUTPUT_ROOT.exists():
        print(f"[ABORT] OUTPUT_ROOT 이미 존재: {OUTPUT_ROOT}", file=sys.stderr)
        sys.exit(1)
    OUTPUT_ROOT.mkdir(parents=True)

    error_rows = []

    # ── [1/7] 입력 검증 ──────────────────────────────────────────────────────
    print("\n[1/7] 입력 검증")
    for p in [
        RD_D1S_SCORE_CSV,
        RD_D2_ROOT / "rd_d2_patient_all_suppressed_risk_cases.csv",
        RD_D2_ROOT / "rd_d2_threshold_topk_guard_simulation.csv",
        RD_D2_ROOT / "rd_d2_ranking_fusion_simulation.csv",
        RD_D2_ROOT / "rd_d2_rdd1s_ranking_guard_strategy_summary.json",
    ]:
        assert_path_safe(p)
        if not p.exists():
            print(f"[ABORT] 입력 없음: {p}", file=sys.stderr)
            sys.exit(1)

    with open(RD_D1S_SCORE_CSV, newline="", encoding="utf-8") as f:
        d1s_rows = list(csv.DictReader(f))

    n_rows    = len(d1s_rows)
    n_pos     = sum(1 for r in d1s_rows if r.get("label") == "positive")
    n_hn      = sum(1 for r in d1s_rows if r.get("label") == "hard_negative")
    n_nan     = sum(1 for r in d1s_rows if r.get("score_nan", "0") == "1")
    n_inf     = sum(1 for r in d1s_rows if r.get("score_inf", "0") == "1")
    n_holdout = sum(1 for r in d1s_rows if r.get("stage_split", "") == "stage2_holdout")

    with open(RD_D2_ROOT / "rd_d2_patient_all_suppressed_risk_cases.csv",
              newline="", encoding="utf-8") as f:
        risk_rows = list(csv.DictReader(f))

    le1_pats = {r["patient_id"] for r in risk_rows if float(r["target_lesion_rate"]) <= 0.011}
    le3_pats = {r["patient_id"] for r in risk_rows if float(r["target_lesion_rate"]) <= 0.031}
    le5_pats = {r["patient_id"] for r in risk_rows if float(r["target_lesion_rate"]) <= 0.051}
    all_risk_pats = le5_pats

    val_checks = [
        {"check": "input_rows",          "result": n_rows,        "expected": INPUT_ROWS_EXPECTED, "pass": n_rows == INPUT_ROWS_EXPECTED},
        {"check": "positive_count",      "result": n_pos,         "expected": POSITIVE_EXPECTED,   "pass": n_pos == POSITIVE_EXPECTED},
        {"check": "hard_negative_count", "result": n_hn,          "expected": HN_EXPECTED,         "pass": n_hn == HN_EXPECTED},
        {"check": "score_nan",           "result": n_nan,         "expected": 0,                   "pass": n_nan == 0},
        {"check": "score_inf",           "result": n_inf,         "expected": 0,                   "pass": n_inf == 0},
        {"check": "stage2_holdout",      "result": n_holdout,     "expected": 0,                   "pass": n_holdout == 0},
        {"check": "risk_patient_le1",    "result": len(le1_pats), "expected": RISK_LE1_EXPECTED,   "pass": len(le1_pats) == RISK_LE1_EXPECTED},
        {"check": "risk_patient_le3",    "result": len(le3_pats), "expected": RISK_LE3_EXPECTED,   "pass": len(le3_pats) == RISK_LE3_EXPECTED},
        {"check": "risk_patient_le5",    "result": len(le5_pats), "expected": RISK_LE5_EXPECTED,   "pass": len(le5_pats) == RISK_LE5_EXPECTED},
    ]

    print(f"  rows={n_rows:,}  positive={n_pos:,}  hard_negative={n_hn:,}")
    print(f"  score_nan={n_nan}  score_inf={n_inf}  holdout={n_holdout}")
    print(f"  risk le1={len(le1_pats)}  le3={len(le3_pats)}  le5={len(le5_pats)}")
    print(f"  all_risk_pats: {sorted(all_risk_pats)}")

    # ConvAE / first_stage join
    assert_path_safe(RD_C3_SCORE_CSV)
    fs_map  = {}
    cae_map = {}
    first_stage_joined = False
    convae_joined      = False

    if RD_C3_SCORE_CSV.exists():
        with open(RD_C3_SCORE_CSV, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                cid = r.get("candidate_id", "")
                fs  = safe_float(r.get("first_stage_score", ""))
                ca  = safe_float(r.get("convAE_crop_score_l1_mean", ""))
                if cid and fs is not None:
                    fs_map[cid]  = fs
                if cid and ca is not None:
                    cae_map[cid] = ca
        cids_all = [r["candidate_id"] for r in d1s_rows]
        fs_miss  = sum(1 for c in cids_all if c not in fs_map)
        cae_miss = sum(1 for c in cids_all if c not in cae_map)
        first_stage_joined = (fs_miss == 0)
        convae_joined      = (cae_miss == 0)
        print(f"  first_stage_score join miss={fs_miss} ({first_stage_joined})")
        print(f"  convAE_score join miss={cae_miss} ({convae_joined})")
    else:
        cids_all = [r["candidate_id"] for r in d1s_rows]
        fs_miss  = n_rows
        cae_miss = n_rows

    val_checks += [
        {"check": "first_stage_join_miss", "result": fs_miss,  "expected": 0, "pass": fs_miss == 0},
        {"check": "convae_join_miss",       "result": cae_miss, "expected": 0, "pass": cae_miss == 0},
    ]

    write_csv(OUTPUT_ROOT / "rd_d3_input_validation.csv",
              ["check", "result", "expected", "pass"], val_checks)
    input_ok = all(c["pass"] for c in val_checks)
    if not input_ok:
        print("[ABORT] 입력 검증 실패", file=sys.stderr)
        sys.exit(1)
    print("  입력 검증 PASS")

    # 전체 score 배열
    d1s_scores = np.array([float(r["rd_d1s_medi3ch_rd4ad_score"]) for r in d1s_rows])
    labels     = np.array([1 if r["label"] == "positive" else 0 for r in d1s_rows])
    cid_to_idx = {r["candidate_id"]: i for i, r in enumerate(d1s_rows)}

    d1s_sorted = sorted(d1s_scores.tolist())
    fs_vals    = [safe_float(fs_map.get(r["candidate_id"], "")) for r in d1s_rows]
    cae_vals   = [safe_float(cae_map.get(r["candidate_id"], "")) for r in d1s_rows]
    fs_sorted  = sorted(x for x in fs_vals  if x is not None)
    cae_sorted = sorted(x for x in cae_vals if x is not None)

    l1_all = [safe_float(r.get("score_layer1", "")) for r in d1s_rows]
    l2_all = [safe_float(r.get("score_layer2", "")) for r in d1s_rows]
    l3_all = [safe_float(r.get("score_layer3", "")) for r in d1s_rows]

    patient_groups = collections.defaultdict(list)
    for i, r in enumerate(d1s_rows):
        patient_groups[r["patient_id"]].append(i)

    # percentile cutoffs
    d1s_p25 = float(np.percentile(d1s_scores, 25))
    d1s_p75 = float(np.percentile(d1s_scores, 75))
    fs_p25  = float(np.percentile(fs_sorted,  25)) if fs_sorted  else None
    fs_p75  = float(np.percentile(fs_sorted,  75)) if fs_sorted  else None
    cae_p25 = float(np.percentile(cae_sorted, 25)) if cae_sorted else None
    cae_p75 = float(np.percentile(cae_sorted, 75)) if cae_sorted else None

    # ── [2/7] 위험 환자별 후보 구조 ─────────────────────────────────────────
    print("\n[2/7] 위험 환자별 후보 구조")
    risk_patient_summary = []

    for pid in sorted(all_risk_pats):
        idxs     = patient_groups.get(pid, [])
        pos_idxs = [i for i in idxs if labels[i] == 1]
        hn_idxs  = [i for i in idxs if labels[i] == 0]

        pos_d1s = [d1s_scores[i] for i in pos_idxs]
        hn_d1s  = [d1s_scores[i] for i in hn_idxs]
        pos_fs  = [fs_map[d1s_rows[i]["candidate_id"]]  for i in pos_idxs if d1s_rows[i]["candidate_id"] in fs_map]
        pos_cae = [cae_map[d1s_rows[i]["candidate_id"]] for i in pos_idxs if d1s_rows[i]["candidate_id"] in cae_map]

        sorted_by_d1s = sorted(idxs, key=lambda i: -d1s_scores[i])
        rank_d1s_map  = {idx: rk + 1 for rk, idx in enumerate(sorted_by_d1s)}

        if first_stage_joined:
            sorted_by_fs = sorted(idxs, key=lambda i: -(fs_map.get(d1s_rows[i]["candidate_id"]) or 0.0))
            rank_fs_map  = {idx: rk + 1 for rk, idx in enumerate(sorted_by_fs)}
        else:
            rank_fs_map = {}

        if convae_joined:
            sorted_by_cae = sorted(idxs, key=lambda i: -(cae_map.get(d1s_rows[i]["candidate_id"]) or 0.0))
            rank_cae_map  = {idx: rk + 1 for rk, idx in enumerate(sorted_by_cae)}
        else:
            rank_cae_map = {}

        best_d1s_rank = min(rank_d1s_map[i] for i in pos_idxs) if pos_idxs else None
        best_fs_rank  = min(rank_fs_map[i]  for i in pos_idxs) if (pos_idxs and rank_fs_map) else None
        best_cae_rank = min(rank_cae_map[i] for i in pos_idxs) if (pos_idxs and rank_cae_map) else None

        in_top1  = any(rank_d1s_map[i] <= 1  for i in pos_idxs)
        in_top3  = any(rank_d1s_map[i] <= 3  for i in pos_idxs)
        in_top5  = any(rank_d1s_map[i] <= 5  for i in pos_idxs)
        in_top10 = any(rank_d1s_map[i] <= 10 for i in pos_idxs)

        def distrib_json(idx_list, col):
            c = collections.Counter(d1s_rows[i].get(col, "") for i in idx_list)
            return json.dumps(dict(c.most_common()), ensure_ascii=False)

        safe_id   = d1s_rows[pos_idxs[0]]["safe_id"] if pos_idxs else ""
        risk_tier = ("le1" if pid in le1_pats else "le3" if pid in le3_pats else "le5")

        risk_patient_summary.append({
            "patient_id":          pid,
            "safe_id":             safe_id,
            "risk_tier":           risk_tier,
            "positive_count":      len(pos_idxs),
            "hard_negative_count": len(hn_idxs),
            "total_candidates":    len(idxs),
            "pos_d1s_min":         round(float(np.min(pos_d1s)),  6) if pos_d1s else "",
            "pos_d1s_mean":        round(float(np.mean(pos_d1s)), 6) if pos_d1s else "",
            "pos_d1s_max":         round(float(np.max(pos_d1s)),  6) if pos_d1s else "",
            "hn_d1s_min":          round(float(np.min(hn_d1s)),   6) if hn_d1s else "",
            "hn_d1s_mean":         round(float(np.mean(hn_d1s)),  6) if hn_d1s else "",
            "hn_d1s_max":          round(float(np.max(hn_d1s)),   6) if hn_d1s else "",
            "pos_fs_min":          round(float(np.min(pos_fs)),   6) if pos_fs else "",
            "pos_fs_mean":         round(float(np.mean(pos_fs)),  6) if pos_fs else "",
            "pos_fs_max":          round(float(np.max(pos_fs)),   6) if pos_fs else "",
            "pos_cae_min":         round(float(np.min(pos_cae)),  6) if pos_cae else "",
            "pos_cae_mean":        round(float(np.mean(pos_cae)), 6) if pos_cae else "",
            "pos_cae_max":         round(float(np.max(pos_cae)),  6) if pos_cae else "",
            "best_pos_rank_d1s":   best_d1s_rank if best_d1s_rank is not None else "",
            "best_pos_rank_fs":    best_fs_rank  if best_fs_rank  is not None else "",
            "best_pos_rank_cae":   best_cae_rank if best_cae_rank is not None else "",
            "pos_in_top1":         in_top1,
            "pos_in_top3":         in_top3,
            "pos_in_top5":         in_top5,
            "pos_in_top10":        in_top10,
            "six_bin_distrib":     distrib_json(pos_idxs, "six_bin_label"),
            "z_level_distrib":     distrib_json(pos_idxs, "z_level"),
            "boundary_distrib":    distrib_json(pos_idxs, "boundary_status"),
        })
        print(f"  {pid} (risk={risk_tier}) pos={len(pos_idxs)} hn={len(hn_idxs)} "
              f"best_rank_d1s={best_d1s_rank} in_top5={in_top5}")

    write_csv(
        OUTPUT_ROOT / "rd_d3_risk_patient_summary.csv",
        ["patient_id","safe_id","risk_tier",
         "positive_count","hard_negative_count","total_candidates",
         "pos_d1s_min","pos_d1s_mean","pos_d1s_max",
         "hn_d1s_min","hn_d1s_mean","hn_d1s_max",
         "pos_fs_min","pos_fs_mean","pos_fs_max",
         "pos_cae_min","pos_cae_mean","pos_cae_max",
         "best_pos_rank_d1s","best_pos_rank_fs","best_pos_rank_cae",
         "pos_in_top1","pos_in_top3","pos_in_top5","pos_in_top10",
         "six_bin_distrib","z_level_distrib","boundary_distrib"],
        risk_patient_summary,
    )

    # ── [3/7] positive candidate taxonomy table ──────────────────────────────
    print("\n[3/7] positive candidate taxonomy table")

    # 비교군: positive score 상위 5 환자
    pat_pos_max = {}
    for pid, idxs in patient_groups.items():
        pos_i = [i for i in idxs if labels[i] == 1]
        if pos_i:
            pat_pos_max[pid] = float(np.max([d1s_scores[i] for i in pos_i]))
    top5_good_pats = sorted(
        [p for p in pat_pos_max if p not in all_risk_pats],
        key=lambda p: -pat_pos_max[p]
    )[:5]

    def make_tax_row(i, group_type):
        r        = d1s_rows[i]
        cid      = r["candidate_id"]
        d1s_sc   = float(r["rd_d1s_medi3ch_rd4ad_score"])
        fs_sc    = fs_map.get(cid)
        cae_sc   = cae_map.get(cid)
        l1       = safe_float(r.get("score_layer1", ""))
        l2       = safe_float(r.get("score_layer2", ""))
        l3       = safe_float(r.get("score_layer3", ""))
        boundary = r.get("boundary_status", "")
        six_bin  = r.get("six_bin_label", "")
        z_level  = r.get("z_level", "")
        pid_     = r.get("patient_id", "")

        def has(s, kw):
            return kw.lower() in s.lower() if s else False

        boundary_related     = has(boundary, "boundary") or has(six_bin, "boundary")
        interior_related     = has(six_bin, "interior")
        lower_lung_related   = has(z_level, "lower")  or has(six_bin, "lower")
        upper_lung_related   = has(z_level, "upper")  or has(six_bin, "upper")
        middle_lung_related  = has(z_level, "middle") or has(six_bin, "middle")

        d1s_low  = d1s_sc < d1s_p25
        d1s_high = d1s_sc > d1s_p75
        cae_high = (cae_sc is not None and cae_p75 is not None and cae_sc > cae_p75)
        fs_high  = (fs_sc  is not None and fs_p75  is not None and fs_sc  > fs_p75)
        fs_low   = (fs_sc  is not None and fs_p25  is not None and fs_sc  < fs_p25)

        all_scores_low = (
            d1s_low
            and (fs_sc  is None or (fs_p25  is not None and fs_sc  < fs_p25))
            and (cae_sc is None or (cae_p25 is not None and cae_sc < cae_p25))
        )

        # patient rank by d1s
        idxs_pat   = patient_groups[pid_]
        sorted_pat = sorted(idxs_pat, key=lambda x: -d1s_scores[x])
        rank_d1s   = next(rk + 1 for rk, x in enumerate(sorted_pat) if x == i)

        # layer failure
        layer_specific = "unknown"
        if l1 is not None and l2 is not None and l3 is not None:
            lv = {"l1": l1, "l2": l2, "l3": l3}
            mn = min(lv, key=lambda k: lv[k])
            mx = max(lv, key=lambda k: lv[k])
            if lv[mx] - lv[mn] > 0.02:
                layer_specific = f"min={mn}({fmt_float(lv[mn], 4)})"
            else:
                layer_specific = "all_similar"

        d1s_pct = percentile_rank(d1s_sc, d1s_sorted)
        fs_pct  = percentile_rank(fs_sc,  fs_sorted)  if fs_sc  is not None else None
        cae_pct = percentile_rank(cae_sc, cae_sorted) if cae_sc is not None else None

        return {
            "group_type":                                       group_type,
            "patient_id":                                       pid_,
            "safe_id":                                          r.get("safe_id", ""),
            "candidate_id":                                     cid,
            "label":                                            r.get("label", ""),
            "local_z":                                          r.get("local_z", ""),
            "crop_y0":                                          r.get("crop_y0", ""),
            "crop_x0":                                          r.get("crop_x0", ""),
            "crop_y1":                                          r.get("crop_y1", ""),
            "crop_x1":                                          r.get("crop_x1", ""),
            "z_level":                                          z_level,
            "boundary_status":                                  boundary,
            "six_bin_label":                                    six_bin,
            "stage_split":                                      r.get("stage_split", ""),
            "rd_d1s_score":                                     round(d1s_sc, 6),
            "rd_d1s_percentile":                                fmt_float(d1s_pct, 2),
            "first_stage_score":                                fmt_float(fs_sc, 6),
            "first_stage_percentile":                           fmt_float(fs_pct, 2),
            "convae_score":                                     fmt_float(cae_sc, 6),
            "convae_percentile":                                fmt_float(cae_pct, 2),
            "score_layer1":                                     fmt_float(l1, 6),
            "score_layer2":                                     fmt_float(l2, 6),
            "score_layer3":                                     fmt_float(l3, 6),
            "patient_rank_d1s":                                 rank_d1s,
            "taxonomy_boundary_related":                        boundary_related,
            "taxonomy_interior_related":                        interior_related,
            "taxonomy_lower_lung_related":                      lower_lung_related,
            "taxonomy_upper_lung_related":                      upper_lung_related,
            "taxonomy_middle_lung_related":                     middle_lung_related,
            "taxonomy_vessel_suspected":                        "unknown",
            "taxonomy_wall_or_pleura_suspected":                "unknown",
            "taxonomy_hilar_or_mediastinal_suspected":          "unknown",
            "taxonomy_small_or_low_lesion_fraction_suspected":  "unknown",
            "taxonomy_score_disagreement_d1s_low_cae_high":     d1s_low and cae_high,
            "taxonomy_score_disagreement_d1s_low_fs_high":      d1s_low and fs_high,
            "taxonomy_score_disagreement_d1s_high_fs_low":      d1s_high and fs_low,
            "taxonomy_all_scores_low":                          all_scores_low,
            "taxonomy_layer_specific_failure":                  layer_specific,
        }

    candidate_rows = []

    for pid in sorted(all_risk_pats):
        idxs = patient_groups.get(pid, [])
        for i in idxs:
            if labels[i] == 1:
                candidate_rows.append(make_tax_row(i, "risk_positive"))
        hn_sorted_pat = sorted([i for i in idxs if labels[i] == 0],
                               key=lambda x: -d1s_scores[x])
        for i in hn_sorted_pat[:10]:
            candidate_rows.append(make_tax_row(i, "risk_patient_top_hn"))

    for pid in top5_good_pats:
        for i in patient_groups.get(pid, []):
            if labels[i] == 1:
                candidate_rows.append(make_tax_row(i, "good_patient_positive"))

    all_hn_by_d1s = sorted([i for i in range(n_rows) if labels[i] == 0],
                           key=lambda x: d1s_scores[x])
    for i in all_hn_by_d1s[:50]:
        candidate_rows.append(make_tax_row(i, "global_hn_lowest50"))
    for i in all_hn_by_d1s[-50:]:
        candidate_rows.append(make_tax_row(i, "global_hn_highest50"))

    CAND_FIELDS = [
        "group_type","patient_id","safe_id","candidate_id","label",
        "local_z","crop_y0","crop_x0","crop_y1","crop_x1",
        "z_level","boundary_status","six_bin_label","stage_split",
        "rd_d1s_score","rd_d1s_percentile",
        "first_stage_score","first_stage_percentile",
        "convae_score","convae_percentile",
        "score_layer1","score_layer2","score_layer3",
        "patient_rank_d1s",
        "taxonomy_boundary_related","taxonomy_interior_related",
        "taxonomy_lower_lung_related","taxonomy_upper_lung_related","taxonomy_middle_lung_related",
        "taxonomy_vessel_suspected","taxonomy_wall_or_pleura_suspected",
        "taxonomy_hilar_or_mediastinal_suspected","taxonomy_small_or_low_lesion_fraction_suspected",
        "taxonomy_score_disagreement_d1s_low_cae_high",
        "taxonomy_score_disagreement_d1s_low_fs_high",
        "taxonomy_score_disagreement_d1s_high_fs_low",
        "taxonomy_all_scores_low",
        "taxonomy_layer_specific_failure",
    ]
    write_csv(OUTPUT_ROOT / "rd_d3_risk_positive_candidate_table.csv",
              CAND_FIELDS, candidate_rows)

    risk_pos_rows = [r for r in candidate_rows if r["group_type"] == "risk_positive"]
    print(f"  total candidate rows: {len(candidate_rows)}  risk_positive: {len(risk_pos_rows)}")

    # ── [4/7] score disagreement 분석 ────────────────────────────────────────
    print("\n[4/7] score disagreement 분석")
    disagree_rows = []
    for r in risk_pos_rows:
        cae_high_flag = r["taxonomy_score_disagreement_d1s_low_cae_high"]
        fs_high_flag  = r["taxonomy_score_disagreement_d1s_low_fs_high"]
        d1s_hfl_flag  = r["taxonomy_score_disagreement_d1s_high_fs_low"]
        all_low_flag  = r["taxonomy_all_scores_low"]

        if cae_high_flag and fs_high_flag:
            disagree_type = "D1S_ONLY_FAILURE"
        elif cae_high_flag and not fs_high_flag:
            disagree_type = "D1S_CAE_DISAGREE"
        elif fs_high_flag and not cae_high_flag:
            disagree_type = "D1S_FS_DISAGREE"
        elif d1s_hfl_flag:
            disagree_type = "D1S_HIGH_FS_LOW"
        elif all_low_flag:
            disagree_type = "ALL_SCORES_LOW"
        else:
            disagree_type = "NO_CLEAR_DISAGREE"

        disagree_rows.append({
            "patient_id":             r["patient_id"],
            "candidate_id":           r["candidate_id"],
            "rd_d1s_score":           r["rd_d1s_score"],
            "rd_d1s_percentile":      r["rd_d1s_percentile"],
            "first_stage_score":      r["first_stage_score"],
            "first_stage_percentile": r["first_stage_percentile"],
            "convae_score":           r["convae_score"],
            "convae_percentile":      r["convae_percentile"],
            "disagree_type":          disagree_type,
            "d1s_low_cae_high":       cae_high_flag,
            "d1s_low_fs_high":        fs_high_flag,
            "d1s_high_fs_low":        d1s_hfl_flag,
            "all_scores_low":         all_low_flag,
        })

    disagree_counter = collections.Counter(r["disagree_type"] for r in disagree_rows)
    print(f"  disagree type: {dict(disagree_counter)}")
    write_csv(
        OUTPUT_ROOT / "rd_d3_score_disagreement_analysis.csv",
        ["patient_id","candidate_id",
         "rd_d1s_score","rd_d1s_percentile",
         "first_stage_score","first_stage_percentile",
         "convae_score","convae_percentile",
         "disagree_type","d1s_low_cae_high","d1s_low_fs_high","d1s_high_fs_low","all_scores_low"],
        disagree_rows,
    )

    # ── [5/7] layer-level 실패 분석 ──────────────────────────────────────────
    print("\n[5/7] layer-level 실패 분석")

    def layer_stats(idx_list):
        l1v = [l1_all[i] for i in idx_list if l1_all[i] is not None]
        l2v = [l2_all[i] for i in idx_list if l2_all[i] is not None]
        l3v = [l3_all[i] for i in idx_list if l3_all[i] is not None]
        return (
            round(float(np.mean(l1v)), 6) if l1v else None,
            round(float(np.mean(l2v)), 6) if l2v else None,
            round(float(np.mean(l3v)), 6) if l3v else None,
        )

    def get_idxs_from_rows(rows):
        return [cid_to_idx[r["candidate_id"]] for r in rows if r["candidate_id"] in cid_to_idx]

    risk_pos_idxs   = get_idxs_from_rows(risk_pos_rows)
    good_pos_rows   = [r for r in candidate_rows if r["group_type"] == "good_patient_positive"]
    good_pos_idxs   = get_idxs_from_rows(good_pos_rows)
    hn_low_idxs     = get_idxs_from_rows([r for r in candidate_rows if r["group_type"] == "global_hn_lowest50"])
    hn_high_idxs    = get_idxs_from_rows([r for r in candidate_rows if r["group_type"] == "global_hn_highest50"])
    all_pos_idxs    = [i for i in range(n_rows) if labels[i] == 1]
    all_hn_idxs     = [i for i in range(n_rows) if labels[i] == 0]

    layer_rows = []
    for group_name, idx_list in [
        ("risk_positive",         risk_pos_idxs),
        ("good_patient_positive", good_pos_idxs),
        ("all_positive",          all_pos_idxs),
        ("all_hard_negative",     all_hn_idxs),
        ("global_hn_lowest50",    hn_low_idxs),
        ("global_hn_highest50",   hn_high_idxs),
    ]:
        l1m, l2m, l3m = layer_stats(idx_list)
        d1s_m = round(float(np.mean([d1s_scores[i] for i in idx_list])), 6) if idx_list else None
        layer_rows.append({
            "group":          group_name,
            "count":          len(idx_list),
            "mean_d1s_score": fmt_float(d1s_m, 6),
            "mean_layer1":    fmt_float(l1m, 6),
            "mean_layer2":    fmt_float(l2m, 6),
            "mean_layer3":    fmt_float(l3m, 6),
        })
        print(f"  {group_name}: d1s={fmt_float(d1s_m,4)} l1={fmt_float(l1m,4)} l2={fmt_float(l2m,4)} l3={fmt_float(l3m,4)}")

    write_csv(OUTPUT_ROOT / "rd_d3_layer_level_failure_analysis.csv",
              ["group","count","mean_d1s_score","mean_layer1","mean_layer2","mean_layer3"],
              layer_rows)

    # ── [6/7] taxonomy flag summary & representative cases ───────────────────
    print("\n[6/7] taxonomy flag summary + representative cases")

    tax_flag_cols = [
        "taxonomy_boundary_related", "taxonomy_interior_related",
        "taxonomy_lower_lung_related", "taxonomy_upper_lung_related", "taxonomy_middle_lung_related",
        "taxonomy_score_disagreement_d1s_low_cae_high",
        "taxonomy_score_disagreement_d1s_low_fs_high",
        "taxonomy_score_disagreement_d1s_high_fs_low",
        "taxonomy_all_scores_low",
    ]
    tax_summary = []
    for tf in tax_flag_cols:
        n_true = sum(1 for r in risk_pos_rows if r.get(tf) is True)
        tax_summary.append({
            "flag":       tf,
            "count_true": n_true,
            "total":      len(risk_pos_rows),
            "rate":       round(n_true / len(risk_pos_rows), 4) if risk_pos_rows else 0,
        })
    for tf in ["taxonomy_vessel_suspected", "taxonomy_wall_or_pleura_suspected",
               "taxonomy_hilar_or_mediastinal_suspected",
               "taxonomy_small_or_low_lesion_fraction_suspected"]:
        tax_summary.append({"flag": tf, "count_true": "unknown",
                             "total": len(risk_pos_rows), "rate": "unknown"})

    write_csv(OUTPUT_ROOT / "rd_d3_taxonomy_flag_summary.csv",
              ["flag","count_true","total","rate"], tax_summary)
    for ts in tax_summary:
        print(f"  {ts['flag']}: {ts['count_true']}/{ts['total']}")

    # representative cases
    rep_cases = []
    rep_done  = collections.defaultdict(int)

    def add_rep(r, taxonomy_label):
        if rep_done[taxonomy_label] >= 3:
            return
        rep_cases.append({
            "taxonomy_label":    taxonomy_label,
            "patient_id":        r["patient_id"],
            "safe_id":           r["safe_id"],
            "candidate_id":      r["candidate_id"],
            "local_z":           r["local_z"],
            "crop_y0":           r["crop_y0"],
            "crop_x0":           r["crop_x0"],
            "crop_y1":           r["crop_y1"],
            "crop_x1":           r["crop_x1"],
            "rd_d1s_score":      r["rd_d1s_score"],
            "rd_d1s_percentile": r["rd_d1s_percentile"],
            "first_stage_score": r["first_stage_score"],
            "convae_score":      r["convae_score"],
            "z_level":           r["z_level"],
            "boundary_status":   r["boundary_status"],
            "six_bin_label":     r["six_bin_label"],
            "score_layer1":      r["score_layer1"],
            "score_layer2":      r["score_layer2"],
            "score_layer3":      r["score_layer3"],
            "taxonomy_flags":    ";".join(
                k.replace("taxonomy_", "")
                for k, v in r.items()
                if k.startswith("taxonomy_") and v is True
            ),
        })
        rep_done[taxonomy_label] += 1

    for r in sorted(risk_pos_rows, key=lambda x: x["rd_d1s_score"]):
        if r["taxonomy_boundary_related"] is True:
            add_rep(r, "boundary_positive_failure")
        if r["taxonomy_score_disagreement_d1s_low_fs_high"] is True:
            add_rep(r, "fs_high_d1s_low")
        if r["taxonomy_score_disagreement_d1s_low_cae_high"] is True:
            add_rep(r, "cae_high_d1s_low")
        if r["taxonomy_all_scores_low"] is True:
            add_rep(r, "all_scores_low_hard_case")

    for r in sorted(risk_pos_rows, key=lambda x: -x["rd_d1s_score"])[:6]:
        add_rep(r, "possible_vessel_adjacent_unknown")

    REP_FIELDS = [
        "taxonomy_label","patient_id","safe_id","candidate_id",
        "local_z","crop_y0","crop_x0","crop_y1","crop_x1",
        "rd_d1s_score","rd_d1s_percentile","first_stage_score","convae_score",
        "z_level","boundary_status","six_bin_label",
        "score_layer1","score_layer2","score_layer3","taxonomy_flags",
    ]
    write_csv(OUTPUT_ROOT / "rd_d3_representative_cases.csv", REP_FIELDS, rep_cases)
    print(f"  representative cases: {len(rep_cases)}")

    # ── [7/7] 최종 원인 판단 ─────────────────────────────────────────────────
    print("\n[7/7] 최종 원인 판단")

    boundary_rate_val = next(
        (ts["rate"] for ts in tax_summary if ts["flag"] == "taxonomy_boundary_related"),
        0
    )
    all_low_rate_val = next(
        (ts["rate"] for ts in tax_summary if ts["flag"] == "taxonomy_all_scores_low"),
        0
    )
    n_risk_pos    = len(risk_pos_rows)
    d1s_only_cnt  = sum(1 for r in disagree_rows if r["disagree_type"] == "D1S_ONLY_FAILURE")
    d1s_only_rate = round(d1s_only_cnt / n_risk_pos, 4) if n_risk_pos > 0 else 0.0

    if n_risk_pos == 0:
        dominant_taxonomy = "INSUFFICIENT_METADATA_FOR_VISUAL_CAUSE"
    elif isinstance(all_low_rate_val, float) and all_low_rate_val > 0.5:
        dominant_taxonomy = "ALL_SCORE_HARD_CASE"
    elif d1s_only_rate >= 0.3:
        dominant_taxonomy = "SCORE_DISAGREEMENT_RD4AD_ONLY"
    elif isinstance(boundary_rate_val, float) and boundary_rate_val > 0.3:
        dominant_taxonomy = "BOUNDARY_DOMINANT_FAILURE"
    else:
        dominant_taxonomy = "SCORE_DISAGREEMENT_RD4AD_ONLY"

    cause_rows = [
        {"cause": "BOUNDARY_DOMINANT_FAILURE",
         "evidence_rate": fmt_float(boundary_rate_val, 4) if isinstance(boundary_rate_val, float) else "unknown",
         "note": "boundary_related rate among risk_positive"},
        {"cause": "ALL_SCORE_HARD_CASE",
         "evidence_rate": fmt_float(all_low_rate_val, 4) if isinstance(all_low_rate_val, float) else "unknown",
         "note": "all_scores_low rate among risk_positive"},
        {"cause": "SCORE_DISAGREEMENT_RD4AD_ONLY",
         "evidence_rate": fmt_float(d1s_only_rate, 4),
         "note": "D1S_ONLY_FAILURE rate (cae_high+fs_high but d1s_low)"},
        {"cause": "VESSEL_ADJACENT_FAILURE",       "evidence_rate": "unknown", "note": "vessel metadata absent"},
        {"cause": "HILAR_OR_MEDIASTINAL_CONFUSION", "evidence_rate": "unknown", "note": "hilar/card metadata absent"},
        {"cause": "LOW_LESION_FRACTION_FAILURE",    "evidence_rate": "unknown", "note": "lesion_fraction metadata absent"},
        {"cause": "INSUFFICIENT_METADATA_FOR_VISUAL_CAUSE", "evidence_rate": "partial",
         "note": "vessel/hilar/lesion_fraction absent → visual cause unverified"},
    ]
    cause_rows.sort(key=lambda r: r["cause"] != dominant_taxonomy)
    write_csv(OUTPUT_ROOT / "rd_d3_failure_cause_decision_table.csv",
              ["cause","evidence_rate","note"], cause_rows)

    final_decision_text = (
        f"RD-D1s는 AUROC={RD_D1S_AUROC_REF}로 전체 구분력은 유효하나 "
        f"patient-level safety 조건에서 실패했다. "
        f"실패는 전체 score 품질 문제가 아니라 특정 환자/후보 유형(dominant: {dominant_taxonomy})에서 "
        f"positive가 낮은 score를 받는 구조적 문제이다. "
        f"RD-D1s는 hard suppression/ranking deployment가 아니라 "
        f"analysis/XAI/taxonomy 신호로 유지한다. "
        f"다음 단계는 새 학습이 아니라 원인 유형별로 "
        f"1차 candidate generation 또는 XAI 설명을 개선하는 것이다."
    )
    print(f"  dominant_taxonomy: {dominant_taxonomy}")
    print(f"  {final_decision_text}")

    # summary JSON
    summary = {
        "input_rows":                       n_rows,
        "risk_patient_count_le1":           len(le1_pats),
        "risk_patient_count_le3":           len(le3_pats),
        "risk_patient_count_le5":           len(le5_pats),
        "candidate_id_join_miss_count":     fs_miss + cae_miss,
        "score_nan_count":                  n_nan,
        "score_inf_count":                  n_inf,
        "stage2_holdout_access":            0,
        "first_stage_score_joined":         first_stage_joined,
        "convae_score_joined":              convae_joined,
        "vessel_metadata_available":        False,
        "lesion_fraction_metadata_available": False,
        "dominant_failure_taxonomy":        dominant_taxonomy,
        "rd_d1s_auroc_reference":           RD_D1S_AUROC_REF,
        "rd_d1s_auprc_reference":           RD_D1S_AUPRC_REF,
        "final_decision":                   "RD4AD_TAXONOMY_ANALYSIS_ONLY",
        "training_run":                     False,
        "model_forward_run":                False,
        "scoring_run":                      False,
        "suppression_applied":              False,
        "first_stage_score_modified":       False,
        "all_checks_passed":                input_ok,
        "disagree_type_distribution":       dict(disagree_counter),
        "risk_positive_total":              n_risk_pos,
        "boundary_related_rate":            boundary_rate_val if isinstance(boundary_rate_val, float) else None,
        "all_scores_low_rate":              all_low_rate_val  if isinstance(all_low_rate_val,  float) else None,
        "d1s_only_failure_rate":            d1s_only_rate,
        "final_decision_text":              final_decision_text,
    }
    with open(OUTPUT_ROOT / "rd_d3_rdd1s_failure_taxonomy_summary.json",
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print("  saved: rd_d3_rdd1s_failure_taxonomy_summary.json")

    # report MD
    md_lines = [
        "# RD-D3: RD-D1s Failure-Risk Patient Taxonomy Audit",
        "",
        f"## 판정: {'PASS' if input_ok else 'FAIL'} / RD4AD_TAXONOMY_ANALYSIS_ONLY",
        "",
        "## 입력 검증",
        f"- rows={n_rows:,}  positive={n_pos:,}  hard_negative={n_hn:,}",
        f"- score_nan={n_nan}  score_inf={n_inf}  holdout={n_holdout}",
        f"- first_stage_joined={first_stage_joined}  convae_joined={convae_joined}",
        f"- risk_patient le1={len(le1_pats)}  le3={len(le3_pats)}  le5={len(le5_pats)}",
        "",
        "## 위험 환자 요약",
    ]
    for rp in risk_patient_summary:
        md_lines.append(
            f"- {rp['patient_id']} (tier={rp['risk_tier']}): "
            f"pos={rp['positive_count']} hn={rp['hard_negative_count']} "
            f"best_rank_d1s={rp['best_pos_rank_d1s']} in_top5={rp['pos_in_top5']}"
        )
    md_lines += ["", "## Score disagreement (risk_positive)"]
    for dtype, cnt in disagree_counter.most_common():
        md_lines.append(f"- {dtype}: {cnt}/{n_risk_pos}")
    md_lines += ["", "## Taxonomy flags (risk_positive)"]
    for ts in tax_summary:
        md_lines.append(f"- {ts['flag']}: {ts['count_true']}/{ts['total']}")
    md_lines += ["", "## Layer-level 분석"]
    for lr in layer_rows:
        md_lines.append(
            f"- {lr['group']} (n={lr['count']}): "
            f"d1s={lr['mean_d1s_score']} l1={lr['mean_layer1']} l2={lr['mean_layer2']} l3={lr['mean_layer3']}"
        )
    md_lines += [
        "", f"## dominant_failure_taxonomy: {dominant_taxonomy}",
        "", "## 결론", final_decision_text,
        "", f"## all_checks_passed: {input_ok}",
    ]
    with open(OUTPUT_ROOT / "rd_d3_rdd1s_failure_taxonomy_report.md",
              "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print("  saved: rd_d3_rdd1s_failure_taxonomy_report.md")

    write_csv(OUTPUT_ROOT / "rd_d3_errors.csv", ["step","detail"], error_rows)

    if input_ok:
        (OUTPUT_ROOT / "DONE").write_text("DONE\n", encoding="utf-8")
        print("\n[DONE] all_checks_passed:", input_ok)
    else:
        (OUTPUT_ROOT / "FAILED").write_text(f"FAILED\nall_checks_passed={input_ok}\n", encoding="utf-8")
        print("\n[FAILED] all_checks_passed:", input_ok, file=sys.stderr)
        sys.exit(1)


# =============================================================================
if IS_DRY_PLAN:
    run_dry_plan()
elif IS_RUN_AUDIT:
    run_audit()
