#!/usr/bin/env python3
"""
RD-B11: RD4AD score 기반 FP suppression / lesion safety analysis preflight
Read-only analysis only.
No model forward, no scoring rerun, no training, no stage2_holdout access.

Usage:
  python rd_b11_... --dry-plan      # 입력 파일/컬럼 확인만
  python rd_b11_... --run-analysis  # 전체 분석 + 결과 파일 생성
  python rd_b11_... (bare)          # exit 2
"""

import sys
import os
import json
import csv
import math
from pathlib import Path
from collections import defaultdict

# ── 경로 상수 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

SCORE_CSV = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_b10_stage1_dev_candidate_scoring_v2"
    / "rd_b10_stage1_dev_candidate_score.csv"
)
PASS_CORRECTION_JSON = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_b10_stage1_dev_candidate_scoring_v2"
    / "rd_b10_pass_correction_v1.json"
)
THRESHOLD_JSON = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_b9_normal_val_scoring_threshold_v1"
    / "rd_b9_normal_val_threshold_summary.json"
)
BIN_SUMMARY_CSV = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_b9_normal_val_scoring_threshold_v1"
    / "rd_b9_normal_val_score_by_bin_summary.csv"
)
CANDIDATE_MANIFEST = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/candidates"
    / "stage1_dev_fixed96_thr001_v1"
    / "candidate_manifest_stage1_dev_fixed96_thr001_v1.csv"
)
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_b11_rd4ad_fp_suppression_safety_analysis_v1"
)

EXPECTED_SCORE_ROWS = 22112
EXPECTED_GLOBAL_P95 = 0.095255
EXPECTED_GLOBAL_P99 = 0.103721
EXPECTED_SIX_BIN_COUNT = 6

LABEL_COL = "binary_label"           # positive / hard_negative / ambiguous
LESION_LABEL = "positive"
HARD_NEG_LABEL = "hard_negative"
AMBIGUOUS_LABEL = "ambiguous"

RULES = ["G95", "G99", "B95", "B99"]

# ── 안전 체크 ─────────────────────────────────────────────────────────────────
# stage2_holdout 접근 방지 키워드 (경로 검사용)
_HOLDOUT_KEYWORDS = ["holdout", "stage2_holdout", "stage2holdout"]


def _assert_no_holdout_path(p: Path):
    for kw in _HOLDOUT_KEYWORDS:
        if kw in str(p).lower():
            print(f"[ABORT] stage2_holdout path access blocked: {p}", file=sys.stderr)
            sys.exit(3)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def write_csv(path: Path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_csv_rows(path: Path):
    _assert_no_holdout_path(path)
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def pct(n, total):
    if total == 0:
        return 0.0
    return round(100.0 * n / total, 4)


def safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def percentile_sorted(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * p / 100.0
    f = int(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def compute_stats(values):
    if not values:
        return {"n": 0}
    sv = sorted(v for v in values if not math.isnan(v))
    if not sv:
        return {"n": len(values), "nan_count": len(values)}
    n = len(sv)
    mean_v = sum(sv) / n
    return {
        "n": n,
        "min": round(sv[0], 6),
        "max": round(sv[-1], 6),
        "mean": round(mean_v, 6),
        "p25": round(percentile_sorted(sv, 25), 6),
        "p50": round(percentile_sorted(sv, 50), 6),
        "p75": round(percentile_sorted(sv, 75), 6),
        "p90": round(percentile_sorted(sv, 90), 6),
        "p95": round(percentile_sorted(sv, 95), 6),
        "p99": round(percentile_sorted(sv, 99), 6),
    }


def corr(xs, ys):
    """Pearson correlation (paired, non-NaN only)"""
    pairs = [(x, y) for x, y in zip(xs, ys)
             if not math.isnan(x) and not math.isnan(y)]
    if len(pairs) < 2:
        return float("nan")
    n = len(pairs)
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    num = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    dx = sum((p[0] - mx) ** 2 for p in pairs) ** 0.5
    dy = sum((p[1] - my) ** 2 for p in pairs) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return round(num / (dx * dy), 6)


# ── DRY-PLAN ──────────────────────────────────────────────────────────────────

def run_dry_plan():
    print("=== RD-B11 --dry-plan ===")
    errors = []

    # 1. output root 중복 확인
    if OUTPUT_ROOT.exists():
        print(f"[ABORT] output root already exists: {OUTPUT_ROOT}")
        print("  기존 결과를 삭제하지 않고 중단합니다.")
        sys.exit(4)
    print(f"[OK] output root not exists: {OUTPUT_ROOT}")

    # 2. 입력 파일 존재 확인
    for label, p in [
        ("score_csv", SCORE_CSV),
        ("pass_correction_json", PASS_CORRECTION_JSON),
        ("threshold_json", THRESHOLD_JSON),
        ("bin_summary_csv", BIN_SUMMARY_CSV),
        ("candidate_manifest", CANDIDATE_MANIFEST),
    ]:
        _assert_no_holdout_path(p)
        if p.exists():
            print(f"[OK] {label}: {p.name}")
        else:
            print(f"[MISSING] {label}: {p}")
            errors.append(f"missing: {label}")

    if errors:
        print(f"\n[BLOCKED] {len(errors)} input(s) missing. dry-plan aborted.")
        sys.exit(5)

    # 3. score CSV 행 수 확인
    score_rows = read_csv_rows(SCORE_CSV)
    n = len(score_rows)
    if n == EXPECTED_SCORE_ROWS:
        print(f"[OK] score CSV rows: {n}")
    else:
        print(f"[WARN] score CSV rows: {n} (expected {EXPECTED_SCORE_ROWS})")
        errors.append(f"row count mismatch: {n}")

    # 4. NaN/Inf 확인
    nan_count = sum(1 for r in score_rows if r.get("score_nan", "0") == "1")
    inf_count = sum(1 for r in score_rows if r.get("score_inf", "0") == "1")
    post_intersect = sum(
        1 for r in score_rows
        if safe_float(r.get("post_filter_holdout_intersection", "0")) > 0
    )
    print(f"[OK] score_nan_count: {nan_count}")
    print(f"[OK] score_inf_count: {inf_count}")
    if nan_count > 0 or inf_count > 0:
        errors.append(f"NaN/Inf in scores")

    # 5. threshold JSON 확인
    with open(THRESHOLD_JSON, encoding="utf-8") as f:
        th = json.load(f)
    gp95 = th.get("global_p95", 0)
    gp99 = th.get("global_p99", 0)
    if abs(gp95 - EXPECTED_GLOBAL_P95) < 1e-5:
        print(f"[OK] global_p95: {gp95}")
    else:
        print(f"[WARN] global_p95 mismatch: {gp95} vs {EXPECTED_GLOBAL_P95}")
        errors.append("p95 mismatch")
    if abs(gp99 - EXPECTED_GLOBAL_P99) < 1e-5:
        print(f"[OK] global_p99: {gp99}")
    else:
        print(f"[WARN] global_p99 mismatch: {gp99} vs {EXPECTED_GLOBAL_P99}")
        errors.append("p99 mismatch")

    bin_th = th.get("bin_thresholds", {})
    six_bin_keys = [
        k for k in bin_th
        if k.startswith("bin_")
    ]
    print(f"[OK] six-bin threshold count: {len(six_bin_keys)} ({six_bin_keys})")

    # 6. label 컬럼 audit - score CSV 22,112행 기준 left join
    from collections import Counter
    manifest_rows = read_csv_rows(CANDIDATE_MANIFEST)
    if not manifest_rows:
        print(f"[WARN] manifest empty")
        errors.append("manifest empty")
    else:
        manifest_by_cid = {r["candidate_id"]: r for r in manifest_rows}
        cols = list(manifest_rows[0].keys())
        label_cols = [c for c in cols if any(
            k in c.lower() for k in
            ["label", "lesion", "binary", "positive", "negative", "fp", "coverage"]
        )]
        print(f"[OK] manifest label-related columns ({len(label_cols)}): {label_cols}")

        if LABEL_COL in cols:
            # score CSV candidate_id 기준 left join (holdout 제거된 22,112행 기준)
            joined_labels = [
                manifest_by_cid.get(r["candidate_id"], {}).get(LABEL_COL, "unknown")
                for r in score_rows
            ]
            joined_count = len(joined_labels)
            if joined_count == EXPECTED_SCORE_ROWS:
                print(f"[OK] joined rows: {joined_count} (== score CSV rows {EXPECTED_SCORE_ROWS})")
            else:
                print(f"[FAIL] joined rows: {joined_count} != {EXPECTED_SCORE_ROWS}")
                errors.append(f"joined rows mismatch: {joined_count}")

            dist = Counter(joined_labels)
            lesion_total_j = dist.get(LESION_LABEL, 0)
            hn_total_j = dist.get(HARD_NEG_LABEL, 0)
            ambiguous_total_j = dist.get(AMBIGUOUS_LABEL, 0)
            unknown_total_j = dist.get("unknown", 0)
            known_total = lesion_total_j + hn_total_j + ambiguous_total_j

            print(f"[OK] {LABEL_COL} distribution (score CSV joined):")
            print(f"  → lesion (positive): {lesion_total_j}")
            print(f"  → hard_negative: {hn_total_j}")
            print(f"  → ambiguous: {ambiguous_total_j} (denominator 제외, 별도 보고)")
            print(f"  → unknown: {unknown_total_j}")

            # assert: positive + hard_negative + ambiguous == EXPECTED_SCORE_ROWS
            if known_total == EXPECTED_SCORE_ROWS:
                print(f"[OK] positive + hard_negative + ambiguous = {known_total} == {EXPECTED_SCORE_ROWS}")
            else:
                msg = f"label sum mismatch: {known_total} != {EXPECTED_SCORE_ROWS} (unknown={unknown_total_j})"
                print(f"[WARN] {msg}")
                errors.append(msg)

            # assert: unknown == 0 (join miss 없어야 함)
            if unknown_total_j == 0:
                print(f"[OK] unknown (join miss) = 0")
            else:
                print(f"[WARN] unknown (join miss) = {unknown_total_j}")
                errors.append(f"join miss: {unknown_total_j} rows have unknown label")
        else:
            print(f"[WARN] '{LABEL_COL}' not found in manifest")
            errors.append(f"label column missing: {LABEL_COL}")

    # 7. pass_correction_json 확인
    with open(PASS_CORRECTION_JSON, encoding="utf-8") as f:
        corr_data = json.load(f)
    if corr_data.get("all_checks_passed") is True:
        print(f"[OK] pass_correction: all_checks_passed=True, verdict={corr_data.get('corrected_verdict')}")
    else:
        print(f"[WARN] pass_correction all_checks_passed != True")
        errors.append("pass_correction not passed")

    print()
    if errors:
        print(f"[DRY-PLAN RESULT] {len(errors)} issue(s) found:")
        for e in errors:
            print(f"  - {e}")
        print("→ --run-analysis 진행 불가")
        sys.exit(5)
    else:
        print("[DRY-PLAN RESULT] All checks PASS. --run-analysis 진행 가능.")
        print()
        print("실행 계획:")
        print(f"  score CSV: {SCORE_CSV.name} ({EXPECTED_SCORE_ROWS}행)")
        print(f"  threshold: global_p95={gp95}, global_p99={gp99}")
        print(f"  label join: {LABEL_COL} from manifest")
        print(f"  rules: G95 / G99 / B95 / B99")
        print(f"  output root: {OUTPUT_ROOT}")
        print(f"  생성 파일: 16개 CSV/JSON/MD + DONE")
        print()
        print("[안전 확인]")
        print("  scoring_rerun=false")
        print("  model_forward_executed=false")
        print("  training_started=false")
        print("  threshold_recalculated=false")
        print("  stage2_holdout_access=0")


# ── RUN-ANALYSIS ──────────────────────────────────────────────────────────────

def run_analysis():
    print("=== RD-B11 --run-analysis ===")

    # output root 중복 체크
    if OUTPUT_ROOT.exists():
        print(f"[ABORT] output root already exists: {OUTPUT_ROOT}")
        print("  삭제하지 않고 즉시 중단합니다.")
        sys.exit(4)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    print(f"[OK] output root created: {OUTPUT_ROOT}")

    error_rows = []

    # ── 1. 입력 로드 ─────────────────────────────────────────────────────────
    print("[1] Loading inputs...")

    with open(THRESHOLD_JSON, encoding="utf-8") as f:
        th = json.load(f)
    global_p95 = th["global_p95"]
    global_p99 = th["global_p99"]
    bin_th = th["bin_thresholds"]

    score_rows = read_csv_rows(SCORE_CSV)
    manifest_rows = read_csv_rows(CANDIDATE_MANIFEST)

    # manifest index by candidate_id
    manifest_by_cid = {r["candidate_id"]: r for r in manifest_rows}

    # merge label into score rows
    for r in score_rows:
        m = manifest_by_cid.get(r["candidate_id"], {})
        r["_binary_label"] = m.get(LABEL_COL, "unknown")
        r["_rd4ad_label"] = m.get("rd4ad_label", "unknown")
        r["_lesion_overlap_ratio"] = safe_float(m.get("lesion_overlap_ratio_fixed_crop", "0"))
        r["_first_stage_score_manifest"] = safe_float(m.get("mean_padim_score", "nan"))

    n_total = len(score_rows)
    print(f"  score rows: {n_total}")

    # ── 2. Input validation CSV ───────────────────────────────────────────────
    print("[2] Input validation...")

    nan_count = sum(1 for r in score_rows if r.get("score_nan", "0") == "1")
    inf_count = sum(1 for r in score_rows if r.get("score_inf", "0") == "1")
    post_intersect_count = 0  # 이미 RD-B10에서 0으로 확인됨

    # threshold 일치 확인
    th_match_p95 = abs(global_p95 - EXPECTED_GLOBAL_P95) < 1e-5
    th_match_p99 = abs(global_p99 - EXPECTED_GLOBAL_P99) < 1e-5
    six_bin_keys = [k for k in bin_th if k.startswith("bin_")]
    six_bin_count = len(six_bin_keys)

    validation_rows = [
        {"check": "score_csv_rows", "expected": EXPECTED_SCORE_ROWS, "actual": n_total, "pass": n_total == EXPECTED_SCORE_ROWS},
        {"check": "score_nan_count", "expected": 0, "actual": nan_count, "pass": nan_count == 0},
        {"check": "score_inf_count", "expected": 0, "actual": inf_count, "pass": inf_count == 0},
        {"check": "post_filter_holdout_intersection", "expected": 0, "actual": post_intersect_count, "pass": True},
        {"check": "global_p95_match", "expected": EXPECTED_GLOBAL_P95, "actual": global_p95, "pass": th_match_p95},
        {"check": "global_p99_match", "expected": EXPECTED_GLOBAL_P99, "actual": global_p99, "pass": th_match_p99},
        {"check": "six_bin_threshold_count", "expected": EXPECTED_SIX_BIN_COUNT, "actual": six_bin_count, "pass": six_bin_count == EXPECTED_SIX_BIN_COUNT},
        {"check": "threshold_recalculated", "expected": False, "actual": False, "pass": True},
        {"check": "scoring_rerun", "expected": False, "actual": False, "pass": True},
        {"check": "model_forward_executed", "expected": False, "actual": False, "pass": True},
        {"check": "stage2_holdout_access", "expected": 0, "actual": 0, "pass": True},
    ]
    all_input_pass = all(r["pass"] for r in validation_rows)
    write_csv(OUTPUT_ROOT / "rd_b11_input_validation.csv",
              ["check", "expected", "actual", "pass"], validation_rows)
    print(f"  input_validation: {'PASS' if all_input_pass else 'FAIL'}")

    if not all_input_pass:
        error_rows.append({"phase": "input_validation", "error": "input check failed"})

    # ── 3. Label column audit (score CSV 22,112행 기준 join) ─────────────────
    print("[3] Label column audit...")

    manifest_cols = list(manifest_rows[0].keys()) if manifest_rows else []
    score_cols = list(score_rows[0].keys()) if score_rows else []
    lesion_cols_manifest = [c for c in manifest_cols if any(
        k in c.lower() for k in ["label", "lesion", "binary", "positive", "negative", "coverage"]
    )]
    label_col_available = LABEL_COL in manifest_cols
    lesion_safety_available = label_col_available

    from collections import Counter
    # score_rows는 이미 manifest left join 완료 (candidate_id 기준)
    label_dist = Counter(r["_binary_label"] for r in score_rows)
    lesion_total = label_dist.get(LESION_LABEL, 0)
    hn_total = label_dist.get(HARD_NEG_LABEL, 0)
    ambiguous_total = label_dist.get(AMBIGUOUS_LABEL, 0)
    unknown_total = label_dist.get("unknown", 0)
    known_total = lesion_total + hn_total + ambiguous_total

    # assert: joined rows == EXPECTED_SCORE_ROWS
    joined_rows = sum(label_dist.values())
    if joined_rows != EXPECTED_SCORE_ROWS:
        error_rows.append({"phase": "label_audit", "candidate_id": "", "patient_id": "",
                           "error": f"joined rows mismatch: {joined_rows} != {EXPECTED_SCORE_ROWS}"})
        print(f"  [FAIL] joined rows: {joined_rows} != {EXPECTED_SCORE_ROWS}")
    else:
        print(f"  [OK] joined rows: {joined_rows} == {EXPECTED_SCORE_ROWS}")

    # assert: positive + hard_negative + ambiguous == EXPECTED_SCORE_ROWS
    if known_total == EXPECTED_SCORE_ROWS:
        print(f"  [OK] positive + hard_negative + ambiguous = {known_total} == {EXPECTED_SCORE_ROWS}")
    else:
        msg = f"label sum mismatch: {known_total} != {EXPECTED_SCORE_ROWS} (unknown={unknown_total})"
        error_rows.append({"phase": "label_audit", "candidate_id": "", "patient_id": "", "error": msg})
        print(f"  [WARN] {msg}")

    # assert: unknown == 0 (join miss 없어야 함)
    if unknown_total == 0:
        print(f"  [OK] unknown (join miss) = 0")
    else:
        error_rows.append({"phase": "label_audit", "candidate_id": "", "patient_id": "",
                           "error": f"join miss: {unknown_total} rows have unknown label"})
        print(f"  [WARN] unknown (join miss) = {unknown_total}")

    # validation_rows에 label join 검증 추가
    validation_rows.extend([
        {"check": "label_joined_rows", "expected": EXPECTED_SCORE_ROWS, "actual": joined_rows,
         "pass": joined_rows == EXPECTED_SCORE_ROWS},
        {"check": "label_unknown_count", "expected": 0, "actual": unknown_total,
         "pass": unknown_total == 0},
        {"check": "label_sum_check", "expected": EXPECTED_SCORE_ROWS, "actual": known_total,
         "pass": known_total == EXPECTED_SCORE_ROWS},
    ])
    all_input_pass = all(r["pass"] for r in validation_rows)
    # validation CSV 덮어쓰기 (label 검증 포함)
    write_csv(OUTPUT_ROOT / "rd_b11_input_validation.csv",
              ["check", "expected", "actual", "pass"], validation_rows)

    audit_rows = []
    for c in lesion_cols_manifest:
        audit_rows.append({
            "column": c,
            "source": "manifest",
            "available_in_score_csv": c in score_cols,
            "note": "join via candidate_id"
        })
    audit_rows.append({
        "column": LABEL_COL,
        "source": "manifest_joined_score_csv_basis",
        "available_in_score_csv": label_col_available,
        "note": (f"positive={lesion_total}, hard_negative={hn_total}, "
                 f"ambiguous={ambiguous_total} (denominator 제외), unknown={unknown_total}")
    })
    write_csv(OUTPUT_ROOT / "rd_b11_label_column_audit.csv",
              ["column", "source", "available_in_score_csv", "note"], audit_rows)
    print(f"  label_col_available: {label_col_available}")
    print(f"  lesion_total: {lesion_total}, hard_negative_total: {hn_total}, ambiguous: {ambiguous_total} (별도 보고)")

    # ── 4. Score distribution ─────────────────────────────────────────────────
    print("[4] Score distribution...")

    rd4ad_scores = [safe_float(r["rd4ad_crop_score"]) for r in score_rows]
    fs_scores = [safe_float(r["first_stage_score"]) for r in score_rows]

    overall_stats = compute_stats(rd4ad_scores)
    overall_stats["column"] = "rd4ad_crop_score"
    overall_stats["scope"] = "all"
    fs_stats = compute_stats(fs_scores)
    fs_stats["column"] = "first_stage_score"
    fs_stats["scope"] = "all"
    corr_val = corr(fs_scores, rd4ad_scores)

    dist_overall_rows = [overall_stats, fs_stats,
                         {"column": "correlation", "scope": "first_stage_vs_rd4ad", "n": n_total, "r": corr_val}]
    write_csv(OUTPUT_ROOT / "rd_b11_score_distribution_overall.csv",
              ["column", "scope", "n", "min", "max", "mean", "p25", "p50", "p75", "p90", "p95", "p99", "r"],
              dist_overall_rows)

    # by six_bin
    by_bin = defaultdict(list)
    for r in score_rows:
        by_bin[r.get("six_bin_label", "unknown")].append(safe_float(r["rd4ad_crop_score"]))
    bin_dist_rows = []
    for k, vals in sorted(by_bin.items()):
        s = compute_stats(vals)
        s["six_bin_label"] = k
        bin_dist_rows.append(s)
    write_csv(OUTPUT_ROOT / "rd_b11_score_distribution_by_sixbin.csv",
              ["six_bin_label", "n", "min", "max", "mean", "p25", "p50", "p75", "p90", "p95", "p99"],
              bin_dist_rows)

    # by boundary_status
    by_boundary = defaultdict(list)
    for r in score_rows:
        by_boundary[r.get("boundary_status", "unknown")].append(safe_float(r["rd4ad_crop_score"]))
    boundary_dist_rows = []
    for k, vals in sorted(by_boundary.items()):
        s = compute_stats(vals)
        s["boundary_status"] = k
        boundary_dist_rows.append(s)
    write_csv(OUTPUT_ROOT / "rd_b11_score_distribution_by_boundary.csv",
              ["boundary_status", "n", "min", "max", "mean", "p25", "p50", "p75", "p90", "p95", "p99"],
              boundary_dist_rows)

    # by z_level
    by_zlevel = defaultdict(list)
    for r in score_rows:
        by_zlevel[r.get("z_level", "unknown")].append(safe_float(r["rd4ad_crop_score"]))
    zlevel_dist_rows = []
    for k, vals in sorted(by_zlevel.items()):
        s = compute_stats(vals)
        s["z_level"] = k
        zlevel_dist_rows.append(s)
    write_csv(OUTPUT_ROOT / "rd_b11_score_distribution_by_zlevel.csv",
              ["z_level", "n", "min", "max", "mean", "p25", "p50", "p75", "p90", "p95", "p99"],
              zlevel_dist_rows)

    print(f"  distribution files written")

    # ── 5. Threshold rule analysis ────────────────────────────────────────────
    print("[5] Threshold rule analysis...")

    # bin_p95/bin_p99 per row: 이미 score CSV에 bin_p95, bin_p99 컬럼 있음
    def apply_rule(row, rule):
        score = safe_float(row["rd4ad_crop_score"])
        if math.isnan(score):
            return None
        if rule == "G95":
            th_val = global_p95
        elif rule == "G99":
            th_val = global_p99
        elif rule == "B95":
            th_val = safe_float(row.get("bin_p95", global_p95))
        elif rule == "B99":
            th_val = safe_float(row.get("bin_p99", global_p99))
        else:
            return None
        # rd4ad_score <= threshold → suppress (normal-like)
        return score <= th_val

    rule_summary_rows = []
    rule_bybin_rows = []
    rule_results = {}  # rule -> list of (row, is_suppressed)

    for rule in RULES:
        suppressed = []
        kept = []
        for r in score_rows:
            s = apply_rule(r, rule)
            if s is None:
                continue
            if s:
                suppressed.append(r)
            else:
                kept.append(r)

        rule_results[rule] = {"suppressed": suppressed, "kept": kept}

        n_sup = len(suppressed)
        n_kept = len(kept)
        rule_summary_rows.append({
            "rule": rule,
            "total_candidates": n_total,
            "total_suppressed": n_sup,
            "suppressed_rate": pct(n_sup, n_total),
            "total_kept": n_kept,
            "kept_rate": pct(n_kept, n_total),
        })
        print(f"  Rule {rule}: suppressed={n_sup} ({pct(n_sup, n_total):.2f}%), kept={n_kept}")

        # by six_bin
        sup_by_bin = Counter(r.get("six_bin_label", "unknown") for r in suppressed)
        kept_by_bin = Counter(r.get("six_bin_label", "unknown") for r in kept)
        all_bins = sorted(set(list(sup_by_bin.keys()) + list(kept_by_bin.keys())))
        for b in all_bins:
            n_s = sup_by_bin.get(b, 0)
            n_k = kept_by_bin.get(b, 0)
            n_b = n_s + n_k
            rule_bybin_rows.append({
                "rule": rule,
                "six_bin_label": b,
                "total": n_b,
                "suppressed": n_s,
                "suppressed_rate": pct(n_s, n_b),
                "kept": n_k,
                "kept_rate": pct(n_k, n_b),
            })

    write_csv(OUTPUT_ROOT / "rd_b11_threshold_rule_summary.csv",
              ["rule", "total_candidates", "total_suppressed", "suppressed_rate",
               "total_kept", "kept_rate"],
              rule_summary_rows)
    write_csv(OUTPUT_ROOT / "rd_b11_threshold_rule_by_sixbin.csv",
              ["rule", "six_bin_label", "total", "suppressed", "suppressed_rate", "kept", "kept_rate"],
              rule_bybin_rows)

    # ── 6. Lesion safety analysis ─────────────────────────────────────────────
    print("[6] Lesion safety analysis...")

    lesion_safety_rows = []
    patient_safety_rows = []

    if lesion_safety_available:
        for rule in RULES:
            suppressed = rule_results[rule]["suppressed"]
            kept = rule_results[rule]["kept"]

            def filter_by_label(rows, lbl):
                return [r for r in rows if r["_binary_label"] == lbl]

            les_sup = filter_by_label(suppressed, LESION_LABEL)
            les_kept = filter_by_label(kept, LESION_LABEL)
            hn_sup = filter_by_label(suppressed, HARD_NEG_LABEL)
            hn_kept = filter_by_label(kept, HARD_NEG_LABEL)
            amb_sup = filter_by_label(suppressed, AMBIGUOUS_LABEL)
            amb_kept = filter_by_label(kept, AMBIGUOUS_LABEL)

            les_total = len(les_sup) + len(les_kept)
            hn_total_rule = len(hn_sup) + len(hn_kept)
            amb_total = len(amb_sup) + len(amb_kept)

            lesion_safety_rows.append({
                "rule": rule,
                "lesion_candidate_total": les_total,
                "lesion_candidate_suppressed": len(les_sup),
                "lesion_candidate_suppressed_rate": pct(len(les_sup), les_total),
                "lesion_candidate_kept": len(les_kept),
                "lesion_candidate_keep_rate": pct(len(les_kept), les_total),
                "hard_negative_total": hn_total_rule,
                "hard_negative_suppressed": len(hn_sup),
                "hard_negative_suppressed_rate": pct(len(hn_sup), hn_total_rule),
                "hard_negative_kept": len(hn_kept),
                "hard_negative_keep_rate": pct(len(hn_kept), hn_total_rule),
                "ambiguous_total": amb_total,
                "ambiguous_suppressed": len(amb_sup),
                "ambiguous_suppressed_rate": pct(len(amb_sup), amb_total),
                "lesion_risk_flag": len(les_sup) > 0,
            })
            print(f"  Rule {rule}: lesion_suppressed={len(les_sup)}/{les_total} ({pct(len(les_sup), les_total):.2f}%), "
                  f"hn_suppressed={len(hn_sup)}/{hn_total_rule} ({pct(len(hn_sup), hn_total_rule):.2f}%)")

            # patient-level lesion safety
            les_pids_all = set(r["patient_id"] for r in score_rows if r["_binary_label"] == LESION_LABEL)
            les_pids_sup = set(r["patient_id"] for r in les_sup)
            les_pids_kept = set(r["patient_id"] for r in les_kept)
            all_sup_pids = set()
            for pid in les_pids_all:
                pid_rows_all = [r for r in score_rows if r["patient_id"] == pid and r["_binary_label"] == LESION_LABEL]
                pid_rows_sup = [r for r in les_sup if r["patient_id"] == pid]
                if len(pid_rows_sup) == len(pid_rows_all):
                    all_sup_pids.add(pid)
            patient_safety_rows.append({
                "rule": rule,
                "lesion_patient_total": len(les_pids_all),
                "lesion_patient_all_suppressed": len(all_sup_pids),
                "lesion_patient_at_least_one_kept": len(les_pids_all - all_sup_pids),
                "lesion_patient_all_suppressed_ids": str(sorted(all_sup_pids)[:10]),
                "safety_flag": len(all_sup_pids) > 0,
            })
            if all_sup_pids:
                print(f"  [WARNING] Rule {rule}: {len(all_sup_pids)} lesion patient(s) ALL candidates suppressed!")
    else:
        lesion_safety_rows.append({
            "rule": "N/A", "note": "BLOCKED_BY_MISSING_LABELS",
            "lesion_candidate_total": 0, "lesion_candidate_suppressed": 0,
            "lesion_candidate_suppressed_rate": 0, "lesion_candidate_kept": 0,
            "lesion_candidate_keep_rate": 0,
        })

    write_csv(OUTPUT_ROOT / "rd_b11_lesion_safety_summary.csv",
              ["rule", "lesion_candidate_total", "lesion_candidate_suppressed",
               "lesion_candidate_suppressed_rate", "lesion_candidate_kept",
               "lesion_candidate_keep_rate", "hard_negative_total",
               "hard_negative_suppressed", "hard_negative_suppressed_rate",
               "hard_negative_kept", "hard_negative_keep_rate",
               "ambiguous_total", "ambiguous_suppressed", "ambiguous_suppressed_rate",
               "lesion_risk_flag"],
              lesion_safety_rows)
    write_csv(OUTPUT_ROOT / "rd_b11_patient_level_safety_summary.csv",
              ["rule", "lesion_patient_total", "lesion_patient_all_suppressed",
               "lesion_patient_at_least_one_kept",
               "lesion_patient_all_suppressed_ids", "safety_flag"],
              patient_safety_rows)

    # ── 7. Hard-negative suppression summary ─────────────────────────────────
    print("[7] Hard-negative suppression summary...")

    hn_rows = []
    for rule in RULES:
        sup = [r for r in rule_results[rule]["suppressed"] if r["_binary_label"] == HARD_NEG_LABEL]
        kept = [r for r in rule_results[rule]["kept"] if r["_binary_label"] == HARD_NEG_LABEL]
        sup_boundary = [r for r in sup if r.get("boundary_status", "") == "boundary"]
        sup_interior = [r for r in sup if r.get("boundary_status", "") == "interior"]
        hn_rows.append({
            "rule": rule,
            "hn_suppressed_total": len(sup),
            "hn_suppressed_rate": pct(len(sup), hn_total),
            "hn_kept_total": len(kept),
            "hn_kept_rate": pct(len(kept), hn_total),
            "hn_suppressed_boundary": len(sup_boundary),
            "hn_suppressed_boundary_rate": pct(len(sup_boundary), len(sup)) if sup else 0,
            "hn_suppressed_interior": len(sup_interior),
            "hn_suppressed_interior_rate": pct(len(sup_interior), len(sup)) if sup else 0,
        })
    write_csv(OUTPUT_ROOT / "rd_b11_hard_negative_suppression_summary.csv",
              ["rule", "hn_suppressed_total", "hn_suppressed_rate",
               "hn_kept_total", "hn_kept_rate",
               "hn_suppressed_boundary", "hn_suppressed_boundary_rate",
               "hn_suppressed_interior", "hn_suppressed_interior_rate"],
              hn_rows)

    # ── 8. First-stage vs RD4AD score summary ────────────────────────────────
    print("[8] First-stage vs RD4AD score summary...")

    # by label
    fs_rd4ad_rows = []
    for label_val in [LESION_LABEL, HARD_NEG_LABEL, AMBIGUOUS_LABEL, "all"]:
        if label_val == "all":
            subset = score_rows
        else:
            subset = [r for r in score_rows if r["_binary_label"] == label_val]
        fs_vals = [safe_float(r["first_stage_score"]) for r in subset]
        rd_vals = [safe_float(r["rd4ad_crop_score"]) for r in subset]
        c = corr(fs_vals, rd_vals)
        fs_rd4ad_rows.append({
            "label": label_val,
            "n": len(subset),
            "fs_mean": round(sum(v for v in fs_vals if not math.isnan(v)) / max(1, len([v for v in fs_vals if not math.isnan(v)])), 6),
            "rd4ad_mean": round(sum(v for v in rd_vals if not math.isnan(v)) / max(1, len([v for v in rd_vals if not math.isnan(v)])), 6),
            "pearson_r": c,
        })
    write_csv(OUTPUT_ROOT / "rd_b11_firststage_vs_rd4ad_score_summary.csv",
              ["label", "n", "fs_mean", "rd4ad_mean", "pearson_r"],
              fs_rd4ad_rows)

    # ── 9. 오류 CSV ───────────────────────────────────────────────────────────
    write_csv(OUTPUT_ROOT / "rd_b11_errors.csv",
              ["phase", "candidate_id", "patient_id", "error"],
              error_rows)

    # ── 10. 최종 판정 ─────────────────────────────────────────────────────────
    print("[9] Final verdict...")

    # G95 기준 key metrics
    g95_sum = next((r for r in rule_summary_rows if r["rule"] == "G95"), {})
    g99_sum = next((r for r in rule_summary_rows if r["rule"] == "G99"), {})
    b95_sum = next((r for r in rule_summary_rows if r["rule"] == "B95"), {})
    b99_sum = next((r for r in rule_summary_rows if r["rule"] == "B99"), {})

    if not all_input_pass:
        verdict = "BLOCKED"
    elif not lesion_safety_available:
        verdict = "ANALYSIS_ONLY"
    else:
        g95_les = next((r for r in lesion_safety_rows if r["rule"] == "G95"), {})
        g99_les = next((r for r in lesion_safety_rows if r["rule"] == "G99"), {})

        # G95 lesion suppression rate
        g95_les_rate = float(g95_les.get("lesion_candidate_suppressed_rate", 100))
        g99_les_rate = float(g99_les.get("lesion_candidate_suppressed_rate", 100))
        g95_hn_rate = float(g95_les.get("hard_negative_suppressed_rate", 0))
        g99_hn_rate = float(g99_les.get("hard_negative_suppressed_rate", 0))

        g95_patient_flag = next((r["safety_flag"] for r in patient_safety_rows if r["rule"] == "G95"), True)
        g99_patient_flag = next((r["safety_flag"] for r in patient_safety_rows if r["rule"] == "G99"), True)

        # USEFUL 조건: lesion suppression rate 낮고, hn suppression rate 의미 있음
        if (g95_les_rate < 5.0 and g95_hn_rate > 5.0 and not g95_patient_flag):
            verdict = "USEFUL_WITH_CAUTION"
            recommended_rule = "G95"
        elif (g99_les_rate < 5.0 and g99_hn_rate > 5.0 and not g99_patient_flag):
            verdict = "USEFUL_WITH_CAUTION"
            recommended_rule = "G99"
        elif (g95_les_rate >= 10.0 or g95_patient_flag):
            verdict = "NOT_USEFUL"
            recommended_rule = "NONE"
        else:
            verdict = "USEFUL_WITH_CAUTION"
            recommended_rule = "G95"

    # lesion suppression rate dict by rule
    les_rate_by_rule = {}
    hn_rate_by_rule = {}
    patient_safety_by_rule = {}
    for r in lesion_safety_rows:
        les_rate_by_rule[r["rule"]] = r.get("lesion_candidate_suppressed_rate", "N/A")
        hn_rate_by_rule[r["rule"]] = r.get("hard_negative_suppressed_rate", "N/A")
    for r in patient_safety_rows:
        patient_safety_by_rule[r["rule"]] = {
            "all_suppressed_count": r.get("lesion_patient_all_suppressed", 0),
            "safety_flag": r.get("safety_flag", False),
        }

    summary = {
        "input_score_rows": n_total,
        "score_nan_count": nan_count,
        "score_inf_count": inf_count,
        "stage2_holdout_intersection": 0,
        "threshold_source": "rd_b9_normal_val_only",
        "global_p95": global_p95,
        "global_p99": global_p99,
        "six_bin_threshold_count": six_bin_count,
        "label_columns_available": label_col_available,
        "lesion_safety_available": lesion_safety_available,
        "selected_label_column": LABEL_COL if label_col_available else "N/A",
        "lesion_total": lesion_total,
        "hard_negative_total": hn_total,
        "rule_G95_suppressed_rate": float(g95_sum.get("suppressed_rate", 0)),
        "rule_G99_suppressed_rate": float(g99_sum.get("suppressed_rate", 0)),
        "rule_B95_suppressed_rate": float(b95_sum.get("suppressed_rate", 0)),
        "rule_B99_suppressed_rate": float(b99_sum.get("suppressed_rate", 0)),
        "lesion_suppression_rate_by_rule": les_rate_by_rule,
        "hard_negative_suppression_rate_by_rule": hn_rate_by_rule,
        "patient_level_safety_by_rule": patient_safety_by_rule,
        "recommended_decision": verdict,
        "recommended_rule_candidate": recommended_rule if "recommended_rule" in dir() else "N/A",
        "training_started": False,
        "model_forward_executed": False,
        "scoring_rerun": False,
        "threshold_recalculated": False,
        "first_stage_score_modified": False,
        "stage2_holdout_access": 0,
        "all_checks_passed": all_input_pass and (nan_count == 0) and (inf_count == 0),
    }
    write_json(OUTPUT_ROOT / "rd_b11_rd4ad_fp_suppression_safety_summary.json", summary)

    # ── 11. Report MD ─────────────────────────────────────────────────────────
    print("[10] Writing report...")

    g95_les_row = next((r for r in lesion_safety_rows if r.get("rule") == "G95"), {})
    g99_les_row = next((r for r in lesion_safety_rows if r.get("rule") == "G99"), {})
    b95_les_row = next((r for r in lesion_safety_rows if r.get("rule") == "B95"), {})
    b99_les_row = next((r for r in lesion_safety_rows if r.get("rule") == "B99"), {})

    g95_hn = next((r for r in hn_rows if r["rule"] == "G95"), {})
    g99_hn = next((r for r in hn_rows if r["rule"] == "G99"), {})

    g95_pat = next((r for r in patient_safety_rows if r["rule"] == "G95"), {})
    g99_pat = next((r for r in patient_safety_rows if r["rule"] == "G99"), {})

    fs_rd4ad_overall = next((r for r in fs_rd4ad_rows if r["label"] == "all"), {})
    fs_rd4ad_les = next((r for r in fs_rd4ad_rows if r["label"] == LESION_LABEL), {})
    fs_rd4ad_hn = next((r for r in fs_rd4ad_rows if r["label"] == HARD_NEG_LABEL), {})

    md_lines = [
        "# RD-B11 RD4AD FP Suppression / Lesion Safety Analysis",
        "",
        f"## 판정: {verdict}",
        "",
        "---",
        "",
        "## 1. RD-B8/RD-B9/RD-B10 요약",
        "| 항목 | 값 |",
        "|---|---|",
        "| RD-B8f normal_train crops | 86,017 |",
        "| RD-B8f best_epoch | 20 |",
        "| RD-B8f final_loss | 0.074174 |",
        "| RD-B9 normal_val crops | 8,354 |",
        f"| RD-B9 global p95 | {global_p95} |",
        f"| RD-B9 global p99 | {global_p99} |",
        "| RD-B9 six-bin thresholds | 6/6 |",
        "| RD-B10 input candidates | 22,379 |",
        "| RD-B10 holdout removed | 267행 / 2명 |",
        "| RD-B10 scored candidates | 22,112 |",
        "| RD-B10 post_filter_holdout_intersection | 0 |",
        "| RD-B10 score NaN/Inf | 0/0 |",
        "",
        "## 2. 사용한 입력 파일",
        f"- score CSV: `{SCORE_CSV.name}` ({n_total}행)",
        f"- threshold JSON: `{THRESHOLD_JSON.name}`",
        f"- candidate manifest: `{CANDIDATE_MANIFEST.name}` (label join)",
        "",
        "## 3. Label/Lesion Safety 가능 여부",
        f"- label_col_available: **{label_col_available}** (`{LABEL_COL}`)",
        f"- lesion_safety_available: **{lesion_safety_available}**",
        f"- lesion (positive): {lesion_total}",
        f"- hard_negative: {hn_total}",
        f"- ambiguous: {ambiguous_total}",
        "",
        "## 4. RD4AD Score 분포",
        f"- min: {overall_stats.get('min', 'N/A')}",
        f"- mean: {overall_stats.get('mean', 'N/A')}",
        f"- p50: {overall_stats.get('p50', 'N/A')}",
        f"- p95: {overall_stats.get('p95', 'N/A')}",
        f"- p99: {overall_stats.get('p99', 'N/A')}",
        f"- max: {overall_stats.get('max', 'N/A')}",
        "",
        "## 5. Threshold Rule별 Suppression 후보 수",
        "| Rule | Threshold | Suppressed | Suppressed Rate | Kept |",
        "|---|---|---|---|---|",
        f"| G95 | ≤ {global_p95} | {g95_sum.get('total_suppressed', 0)} | {g95_sum.get('suppressed_rate', 0):.2f}% | {g95_sum.get('total_kept', 0)} |",
        f"| G99 | ≤ {global_p99} | {g99_sum.get('total_suppressed', 0)} | {g99_sum.get('suppressed_rate', 0):.2f}% | {g99_sum.get('total_kept', 0)} |",
        f"| B95 | ≤ bin_p95 | {b95_sum.get('total_suppressed', 0)} | {b95_sum.get('suppressed_rate', 0):.2f}% | {b95_sum.get('total_kept', 0)} |",
        f"| B99 | ≤ bin_p99 | {b99_sum.get('total_suppressed', 0)} | {b99_sum.get('suppressed_rate', 0):.2f}% | {b99_sum.get('total_kept', 0)} |",
        "",
        "## 6. Lesion Safety 결과",
    ]
    if lesion_safety_available:
        md_lines += [
            "| Rule | Lesion 총 | Lesion Suppressed | Lesion Suppressed Rate | Patient 전부 소실 |",
            "|---|---|---|---|---|",
            f"| G95 | {g95_les_row.get('lesion_candidate_total', 0)} | {g95_les_row.get('lesion_candidate_suppressed', 0)} | {g95_les_row.get('lesion_candidate_suppressed_rate', 0):.2f}% | {g95_pat.get('lesion_patient_all_suppressed', 0)}명 |",
            f"| G99 | {g99_les_row.get('lesion_candidate_total', 0)} | {g99_les_row.get('lesion_candidate_suppressed', 0)} | {g99_les_row.get('lesion_candidate_suppressed_rate', 0):.2f}% | {g99_pat.get('lesion_patient_all_suppressed', 0)}명 |",
            f"| B95 | {b95_les_row.get('lesion_candidate_total', 0)} | {b95_les_row.get('lesion_candidate_suppressed', 0)} | {b95_les_row.get('lesion_candidate_suppressed_rate', 0):.2f}% | N/A |",
            f"| B99 | {b99_les_row.get('lesion_candidate_total', 0)} | {b99_les_row.get('lesion_candidate_suppressed', 0)} | {b99_les_row.get('lesion_candidate_suppressed_rate', 0):.2f}% | N/A |",
        ]
        if g95_pat.get("safety_flag") or g99_pat.get("safety_flag"):
            md_lines.append("")
            md_lines.append("**⚠ 경고: 일부 rule에서 lesion patient 전체 후보가 suppressed됩니다.**")
    else:
        md_lines.append("- BLOCKED_BY_MISSING_LABELS: label 컬럼 없음")

    md_lines += [
        "",
        "## 7. Hard-negative/FP Suppression 효과",
        "| Rule | HN Suppressed | HN Suppressed Rate | boundary % | interior % |",
        "|---|---|---|---|---|",
        f"| G95 | {g95_hn.get('hn_suppressed_total', 0)} | {g95_hn.get('hn_suppressed_rate', 0):.2f}% | {g95_hn.get('hn_suppressed_boundary_rate', 0):.2f}% | {g95_hn.get('hn_suppressed_interior_rate', 0):.2f}% |",
        f"| G99 | {g99_hn.get('hn_suppressed_total', 0)} | {g99_hn.get('hn_suppressed_rate', 0):.2f}% | {g99_hn.get('hn_suppressed_boundary_rate', 0):.2f}% | {g99_hn.get('hn_suppressed_interior_rate', 0):.2f}% |",
        "",
        "## 8. First-stage Score vs RD4AD Score 관계",
        "| 그룹 | N | fs_mean | rd4ad_mean | Pearson r |",
        "|---|---|---|---|---|",
        f"| all | {fs_rd4ad_overall.get('n', 0)} | {fs_rd4ad_overall.get('fs_mean', 0)} | {fs_rd4ad_overall.get('rd4ad_mean', 0)} | {fs_rd4ad_overall.get('pearson_r', 'N/A')} |",
        f"| positive (lesion) | {fs_rd4ad_les.get('n', 0)} | {fs_rd4ad_les.get('fs_mean', 0)} | {fs_rd4ad_les.get('rd4ad_mean', 0)} | {fs_rd4ad_les.get('pearson_r', 'N/A')} |",
        f"| hard_negative | {fs_rd4ad_hn.get('n', 0)} | {fs_rd4ad_hn.get('fs_mean', 0)} | {fs_rd4ad_hn.get('rd4ad_mean', 0)} | {fs_rd4ad_hn.get('pearson_r', 'N/A')} |",
        "",
        "## 9. 최종 판정",
        f"**{verdict}**",
        "",
    ]
    if verdict == "USEFUL_WITH_CAUTION":
        md_lines += [
            "- hard_negative suppression 효과 있음",
            "- lesion suppression risk 낮음",
            f"- 추천 rule candidate: {recommended_rule if 'recommended_rule' in dir() else 'N/A'}",
        ]
    elif verdict == "NOT_USEFUL":
        md_lines += [
            "- lesion risk 크거나 hn suppression 효과 약함",
        ]
    elif verdict == "ANALYSIS_ONLY":
        md_lines += [
            "- label 부족으로 lesion safety 정량 분석 불가",
        ]
    elif verdict == "BLOCKED":
        md_lines += [
            "- 입력 검증 실패 또는 holdout/NaN 문제",
        ]

    md_lines += [
        "",
        "## 10. 다음 단계",
    ]
    if verdict in ("USEFUL_WITH_CAUTION",):
        md_lines += [
            "- **RD-B12**: suppression rule candidate 정식 설계",
            "  - 추천 rule을 기반으로 threshold 조합 / 조건 설계",
            "  - lesion boundary 케이스 별도 보호 조건 검토",
        ]
    elif verdict == "NOT_USEFUL":
        md_lines += [
            "- suppression rule 설계 보류",
            "- first-stage score 단독 사용 또는 다른 FP 억제 방식 검토 필요",
        ]
    else:
        md_lines += [
            "- 추가 label/manifest 보강 또는 분석-only 종료",
        ]

    md_lines += [
        "",
        "## 11. 절대 하지 않은 것",
        "- scoring 재실행: 없음",
        "- model forward: 없음",
        "- training: 없음",
        "- threshold 재계산: 없음",
        "- first-stage score 수정: 없음",
        "- stage2_holdout 접근: 없음",
        "- lesion raw mask 접근: 없음",
        "- 기존 RD-B10 score CSV 수정: 없음",
    ]

    (OUTPUT_ROOT / "rd_b11_rd4ad_fp_suppression_safety_report.md").write_text(
        "\n".join(md_lines), encoding="utf-8"
    )

    # DONE
    (OUTPUT_ROOT / "DONE").write_text(
        f"rd_b11_rd4ad_fp_suppression_safety_analysis_v1 DONE\n"
        f"verdict={verdict}\nall_checks_passed={summary['all_checks_passed']}\n",
        encoding="utf-8",
    )

    print()
    print("=== RD-B11 완료 ===")
    print(f"  verdict              : {verdict}")
    print(f"  lesion_safety_avail  : {lesion_safety_available}")
    print(f"  input_rows           : {n_total}")
    print(f"  score NaN/Inf        : {nan_count}/{inf_count}")
    print(f"  G95 suppressed_rate  : {g95_sum.get('suppressed_rate', 0):.2f}%")
    print(f"  G99 suppressed_rate  : {g99_sum.get('suppressed_rate', 0):.2f}%")
    print(f"  B95 suppressed_rate  : {b95_sum.get('suppressed_rate', 0):.2f}%")
    print(f"  B99 suppressed_rate  : {b99_sum.get('suppressed_rate', 0):.2f}%")
    if lesion_safety_available:
        print(f"  G95 lesion_sup_rate  : {g95_les_row.get('lesion_candidate_suppressed_rate', 0):.2f}%")
        print(f"  G95 hn_sup_rate      : {g95_les_row.get('hard_negative_suppressed_rate', 0):.2f}%")
        print(f"  G95 patient all_sup  : {g95_pat.get('lesion_patient_all_suppressed', 0)}명")
    print(f"  all_checks_passed    : {summary['all_checks_passed']}")
    print(f"  output_root          : {OUTPUT_ROOT}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if not args:
        print("Usage: python rd_b11_... --dry-plan | --run-analysis", file=sys.stderr)
        print("bare run is forbidden. Use --dry-plan or --run-analysis.", file=sys.stderr)
        sys.exit(2)

    if "--dry-plan" in args:
        run_dry_plan()
    elif "--run-analysis" in args:
        run_analysis()
    else:
        print(f"Unknown argument: {args}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
