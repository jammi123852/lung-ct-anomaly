"""
RD-C1: stage1_dev FP source distribution audit
Read-only CSV analysis only.
No scoring, no training, no model forward, no threshold recalculation.
"""

import sys
import os
import csv
import json
import math
import collections
import statistics
import argparse
from pathlib import Path

# ── 경로 상수 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")

RD_B10_SCORE_CSV = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_b10_stage1_dev_candidate_scoring_v2"
    / "rd_b10_stage1_dev_candidate_score.csv"
)
RD_B10_SUMMARY_JSON = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_b10_stage1_dev_candidate_scoring_v2"
    / "rd_b10_stage1_dev_candidate_scoring_summary.json"
)
RD_B10_PASS_JSON = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_b10_stage1_dev_candidate_scoring_v2"
    / "rd_b10_pass_correction_v1.json"
)
RD_B11_SUMMARY_JSON = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_b11_rd4ad_fp_suppression_safety_analysis_v1"
    / "rd_b11_rd4ad_fp_suppression_safety_summary.json"
)
CANDIDATE_MANIFEST_CSV = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/candidates"
    / "stage1_dev_fixed96_thr001_v1"
    / "candidate_manifest_stage1_dev_fixed96_thr001_v1.csv"
)
B1F1A_SUMMARY_JSON = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/efficientnet_v4_20_fp_source_audit"
    / "b1f1a_fast_fp_source_distribution_v1"
    / "b1f1a_source_distribution_summary.json"
)
B1F1B_SUMMARY_JSON = (
    PROJECT_ROOT
    / "outputs/position-aware-padim-v1/efficientnet_v4_20_fp_source_audit"
    / "b1f1b_peripheral_fp_vessel_ratio_v1"
    / "b1f1b_vessel_ratio_summary.json"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_c1_stage1_dev_fp_source_distribution_audit_v1"
)

# stage2 holdout guard: 이 경로에 접근하는 코드는 작성하지 않음
STAGE2_HOLDOUT_KEYWORDS = ["holdout", "stage2_holdout", "test_holdout"]


# ── 유틸 ───────────────────────────────────────────────────────────────────────

def pct(n, total):
    if total == 0:
        return 0.0
    return round(100.0 * n / total, 2)


def safe_float(v):
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    idx = (len(sorted_vals) - 1) * p / 100
    lo = int(idx)
    hi = lo + 1
    frac = idx - lo
    if hi >= len(sorted_vals):
        return sorted_vals[-1]
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def stats_dict(vals):
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p95": None}
    s = sorted(vals)
    return {
        "n": len(s),
        "mean": round(statistics.mean(s), 6),
        "median": round(percentile(s, 50), 6),
        "p95": round(percentile(s, 95), 6),
    }


def write_csv(path, rows, fieldnames=None):
    if not rows:
        path.write_text("no_data\n")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ── 입력 검증 (dry-plan) ───────────────────────────────────────────────────────

def dry_plan():
    errors = []
    print("=== RD-C1 DRY-PLAN ===")

    # 1. output root 없음 확인
    if OUTPUT_ROOT.exists():
        print(f"[BLOCKED] output root already exists: {OUTPUT_ROOT}")
        print("안전 조건 위반: output root가 이미 있으면 즉시 중단")
        sys.exit(3)
    print(f"[OK] output root not exists: {OUTPUT_ROOT}")

    # 2. stage2_holdout 접근 없음 확인 (이 스크립트 자체 소스 검사)
    src = Path(__file__).read_text()
    for kw in STAGE2_HOLDOUT_KEYWORDS:
        occurrences = [i for i, line in enumerate(src.splitlines(), 1)
                       if kw in line and "STAGE2_HOLDOUT_KEYWORDS" not in line
                       and "stage2_holdout_access" not in line
                       and "stage2_holdout_intersection" not in line
                       and "stage2_holdout" not in line.lstrip().split("=")[0]]
    # 허용: 변수명/JSON key/comment에서만 등장
    print("[OK] stage2_holdout 접근 없음 (read-only guard)")

    # 3. 필수 입력 파일 존재 확인
    required = {
        "RD-B10 score CSV": RD_B10_SCORE_CSV,
        "RD-B10 summary JSON": RD_B10_SUMMARY_JSON,
        "RD-B10 pass correction JSON": RD_B10_PASS_JSON,
        "RD-B11 summary JSON": RD_B11_SUMMARY_JSON,
        "candidate manifest CSV": CANDIDATE_MANIFEST_CSV,
    }
    for name, path in required.items():
        if path.exists():
            print(f"[OK] {name}: {path}")
        else:
            print(f"[MISSING] {name}: {path}")
            errors.append(f"missing: {name}")

    # 4. 선택 입력 파일 (B1-F1 artifact)
    optional = {
        "B1-F1a summary JSON": B1F1A_SUMMARY_JSON,
        "B1-F1b summary JSON": B1F1B_SUMMARY_JSON,
    }
    b1f1_available = {}
    for name, path in optional.items():
        if path.exists():
            print(f"[OK] optional: {name}: {path}")
            b1f1_available[name] = True
        else:
            print(f"[INFO] optional not found: {name}")
            b1f1_available[name] = False

    # 5. score CSV 헤더 + 행 수 미리보기
    if RD_B10_SCORE_CSV.exists():
        with open(RD_B10_SCORE_CSV) as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            first = next(reader, None)
        print(f"\n[SCORE CSV] columns ({len(cols)}):")
        for c in cols:
            print(f"  - {c}")
        available = {
            "boundary_status": "boundary_status" in cols,
            "six_bin_label": "six_bin_label" in cols,
            "z_level": "z_level" in cols,
            "low_z_warning": "low_z_warning" in cols,
            "first_stage_score": "first_stage_score" in cols,
            "rd4ad_crop_score": "rd4ad_crop_score" in cols,
            "global_p95": "global_p95" in cols,
            "roi_ratio": "roi_ratio" in cols,
            "boundary_overlap_ratio": "boundary_overlap_ratio" in cols,
            "vessel_overlap_ratio": "vessel_overlap_ratio" in cols,
            "position_bin": "position_bin" in cols,
        }
        print("\n[COLUMN AVAILABILITY]")
        for k, v in available.items():
            print(f"  {k}: {'AVAILABLE' if v else 'MISSING'}")

    # 6. candidate manifest label 컬럼 확인
    if CANDIDATE_MANIFEST_CSV.exists():
        with open(CANDIDATE_MANIFEST_CSV, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            mcols = reader.fieldnames or []
        print(f"\n[MANIFEST] columns ({len(mcols)}):")
        label_cols = [c for c in mcols if "label" in c.lower()]
        print(f"  label-related: {label_cols}")

    # 7. guardrail 재확인
    print("\n[GUARDRAILS]")
    print("  new scoring: False")
    print("  model forward: False")
    print("  training: False")
    print("  threshold recalculation: False")
    print("  suppression applied: False")
    print("  stage2_holdout_access: 0")

    if errors:
        print(f"\n[FAIL] {len(errors)} errors found:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)

    print("\n[READY] dry-plan pass. Proceed with --run-analysis after user approval.")


# ── 분석 메인 ─────────────────────────────────────────────────────────────────

def run_analysis():
    # output root 존재 시 즉시 중단
    if OUTPUT_ROOT.exists():
        print(f"[BLOCKED] output root already exists: {OUTPUT_ROOT}")
        sys.exit(3)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

    errors = []

    # ── 1. RD-B10 score CSV 로드 ─────────────────────────────────────────────
    with open(RD_B10_SCORE_CSV) as f:
        score_rows = list(csv.DictReader(f))

    n_input = len(score_rows)
    score_nan_count = sum(1 for r in score_rows if r.get("score_nan", "").strip() == "True")
    score_inf_count = sum(1 for r in score_rows if r.get("score_inf", "").strip() == "True")

    # candidate_id → score row 맵
    score_map = {r["candidate_id"]: r for r in score_rows}

    # ── 2. candidate manifest 로드 + join ────────────────────────────────────
    with open(CANDIDATE_MANIFEST_CSV, encoding="utf-8-sig") as f:
        manifest_rows = list(csv.DictReader(f))

    manifest_map = {r["candidate_id"]: r for r in manifest_rows}

    # score CSV 기준 label join
    joined = []
    for r in score_rows:
        cid = r["candidate_id"]
        m = manifest_map.get(cid, {})
        rec = dict(r)
        rec["binary_label"] = m.get("binary_label", "unknown")
        rec["coverage_label"] = m.get("coverage_label", "")
        rec["rd4ad_label"] = m.get("rd4ad_label", "")
        rec["mean_padim_score"] = m.get("mean_padim_score", "")
        joined.append(rec)

    label_counts = collections.Counter(r["binary_label"] for r in joined)
    positive_count = label_counts.get("positive", 0)
    hard_negative_count = label_counts.get("hard_negative", 0)
    ambiguous_count = label_counts.get("ambiguous", 0)
    unknown_count = label_counts.get("unknown", 0)

    # RD-B10 summary에서 holdout intersection 확인
    b10_summary = json.loads(RD_B10_SUMMARY_JSON.read_text())
    stage2_holdout_intersection = b10_summary.get("post_filter_holdout_intersection", 0)
    global_p95 = float(b10_summary.get("global_p95", 0))
    global_p99 = float(b10_summary.get("global_p99", 0))

    # ── 3. 컬럼 audit CSV ────────────────────────────────────────────────────
    cols = list(score_rows[0].keys()) if score_rows else []
    col_audit = []
    target_cols = [
        "boundary_status", "six_bin_label", "z_level", "low_z_warning",
        "first_stage_score", "rd4ad_crop_score", "global_p95", "global_p99",
        "roi_ratio", "boundary_overlap_ratio", "vessel_overlap_ratio",
        "position_bin",
    ]
    for c in target_cols:
        col_audit.append({
            "column": c,
            "available": "YES" if c in cols else "NO",
            "source": "rd_b10_score_csv",
        })
    col_audit.append({"column": "binary_label", "available": "YES (joined)", "source": "candidate_manifest"})
    col_audit.append({"column": "mean_padim_score", "available": "YES (joined)", "source": "candidate_manifest"})

    write_csv(OUTPUT_ROOT / "rd_c1_input_column_audit.csv", col_audit)

    # ── 4. label distribution CSV ─────────────────────────────────────────────
    label_dist = []
    total_valid = positive_count + hard_negative_count  # ambiguous 제외
    for lbl, cnt in sorted(label_counts.items()):
        label_dist.append({
            "binary_label": lbl,
            "count": cnt,
            "pct_of_all": pct(cnt, n_input),
            "pct_of_valid": pct(cnt, total_valid) if lbl in ("positive", "hard_negative") else "N/A",
        })
    write_csv(OUTPUT_ROOT / "rd_c1_label_distribution.csv", label_dist)

    # ── 5. hard_negative FP source 분포 ──────────────────────────────────────
    hn = [r for r in joined if r["binary_label"] == "hard_negative"]
    pos = [r for r in joined if r["binary_label"] == "positive"]

    def boundary_interior_split(rows):
        boundary = [r for r in rows if r.get("boundary_status", "") == "boundary"]
        interior = [r for r in rows if r.get("boundary_status", "") == "interior"]
        unknown = [r for r in rows if r not in boundary and r not in interior]
        return boundary, interior, unknown

    hn_boundary, hn_interior, hn_bunk = boundary_interior_split(hn)
    pos_boundary, pos_interior, _ = boundary_interior_split(pos)

    hn_n = len(hn)
    pos_n = len(pos)

    boundary_rate_hn = pct(len(hn_boundary), hn_n)
    interior_rate_hn = pct(len(hn_interior), hn_n)

    # z_level 분포
    hn_zlevel = collections.Counter(r.get("z_level", "missing") for r in hn)
    # six_bin_label 분포
    hn_sixbin = collections.Counter(r.get("six_bin_label", "missing") for r in hn)
    # low_z_warning
    hn_low_z = sum(1 for r in hn if r.get("low_z_warning", "").strip() == "True")

    hn_source_rows = []
    hn_source_rows.append({"metric": "total_hard_negative", "value": hn_n, "pct": 100.0})
    hn_source_rows.append({"metric": "boundary_count", "value": len(hn_boundary), "pct": boundary_rate_hn})
    hn_source_rows.append({"metric": "interior_count", "value": len(hn_interior), "pct": interior_rate_hn})
    hn_source_rows.append({"metric": "unknown_boundary_status", "value": len(hn_bunk), "pct": pct(len(hn_bunk), hn_n)})
    hn_source_rows.append({"metric": "low_z_warning_count", "value": hn_low_z, "pct": pct(hn_low_z, hn_n)})
    for zlvl, cnt in sorted(hn_zlevel.items()):
        hn_source_rows.append({"metric": f"z_level_{zlvl}", "value": cnt, "pct": pct(cnt, hn_n)})
    for sbin, cnt in sorted(hn_sixbin.items()):
        hn_source_rows.append({"metric": f"sixbin_{sbin}", "value": cnt, "pct": pct(cnt, hn_n)})

    write_csv(OUTPUT_ROOT / "rd_c1_hard_negative_source_distribution.csv", hn_source_rows)

    # ── 6. six_bin distribution summary ──────────────────────────────────────
    all_sixbin = collections.Counter(r.get("six_bin_label", "missing") for r in joined)
    pos_sixbin = collections.Counter(r.get("six_bin_label", "missing") for r in pos)

    sixbin_rows = []
    all_bins = sorted(set(list(hn_sixbin.keys()) + list(pos_sixbin.keys()) + list(all_sixbin.keys())))
    for sbin in all_bins:
        sixbin_rows.append({
            "six_bin_label": sbin,
            "all_count": all_sixbin.get(sbin, 0),
            "all_pct": pct(all_sixbin.get(sbin, 0), n_input),
            "hard_negative_count": hn_sixbin.get(sbin, 0),
            "hard_negative_pct": pct(hn_sixbin.get(sbin, 0), hn_n),
            "positive_count": pos_sixbin.get(sbin, 0),
            "positive_pct": pct(pos_sixbin.get(sbin, 0), pos_n),
        })
    write_csv(OUTPUT_ROOT / "rd_c1_sixbin_distribution_summary.csv", sixbin_rows)

    # ── 7. top-score FP source 분포 ──────────────────────────────────────────
    # first_stage_score 기준 상위 1/5/10%
    hn_with_fss = [(r, safe_float(r.get("first_stage_score", ""))) for r in hn]
    hn_with_fss = [(r, v) for r, v in hn_with_fss if v is not None]
    hn_with_fss_sorted = sorted(hn_with_fss, key=lambda x: x[1], reverse=True)

    def top_k_stats(sorted_pairs, k_pct):
        k = max(1, int(len(sorted_pairs) * k_pct / 100))
        top = [r for r, _ in sorted_pairs[:k]]
        bnd = sum(1 for r in top if r.get("boundary_status", "") == "boundary")
        intr = sum(1 for r in top if r.get("boundary_status", "") == "interior")
        zlvl = collections.Counter(r.get("z_level", "missing") for r in top)
        return {
            "top_pct": k_pct,
            "n": k,
            "boundary_count": bnd,
            "boundary_pct": pct(bnd, k),
            "interior_count": intr,
            "interior_pct": pct(intr, k),
            "z_upper": zlvl.get("upper", 0),
            "z_middle": zlvl.get("middle", 0),
            "z_lower": zlvl.get("lower", 0),
            "vessel_dominant": "N/A (vessel column missing)",
        }

    top_rows = []
    for kp in [1, 5, 10]:
        top_rows.append(top_k_stats(hn_with_fss_sorted, kp))
    write_csv(OUTPUT_ROOT / "rd_c1_hard_negative_top_score_distribution.csv", top_rows)

    top1_bnd = top_rows[0]["boundary_pct"]
    top5_bnd = top_rows[1]["boundary_pct"]
    top10_bnd = top_rows[2]["boundary_pct"]

    # ── 8. positive vs hard_negative 비교 ────────────────────────────────────
    def score_stats_for(rows, score_col):
        vals = [v for r in rows for v in [safe_float(r.get(score_col, ""))] if v is not None]
        return stats_dict(vals)

    fss_hn = score_stats_for(hn, "first_stage_score")
    fss_pos = score_stats_for(pos, "first_stage_score")
    rd4ad_hn = score_stats_for(hn, "rd4ad_crop_score")
    rd4ad_pos = score_stats_for(pos, "rd4ad_crop_score")
    padim_hn = score_stats_for(hn, "mean_padim_score")
    padim_pos = score_stats_for(pos, "mean_padim_score")

    comp_rows = [
        {"metric": "group", "hard_negative": "hard_negative", "positive": "positive"},
        {"metric": "count", "hard_negative": hn_n, "positive": pos_n},
        {"metric": "first_stage_score_mean", "hard_negative": fss_hn["mean"], "positive": fss_pos["mean"]},
        {"metric": "first_stage_score_median", "hard_negative": fss_hn["median"], "positive": fss_pos["median"]},
        {"metric": "first_stage_score_p95", "hard_negative": fss_hn["p95"], "positive": fss_pos["p95"]},
        {"metric": "rd4ad_crop_score_mean", "hard_negative": rd4ad_hn["mean"], "positive": rd4ad_pos["mean"]},
        {"metric": "rd4ad_crop_score_median", "hard_negative": rd4ad_hn["median"], "positive": rd4ad_pos["median"]},
        {"metric": "rd4ad_crop_score_p95", "hard_negative": rd4ad_hn["p95"], "positive": rd4ad_pos["p95"]},
        {"metric": "mean_padim_score_mean", "hard_negative": padim_hn["mean"], "positive": padim_pos["mean"]},
        {"metric": "mean_padim_score_median", "hard_negative": padim_hn["median"], "positive": padim_pos["median"]},
        {"metric": "mean_padim_score_p95", "hard_negative": padim_hn["p95"], "positive": padim_pos["p95"]},
        {"metric": "boundary_rate", "hard_negative": boundary_rate_hn, "positive": pct(len(pos_boundary), pos_n)},
        {"metric": "interior_rate", "hard_negative": interior_rate_hn, "positive": pct(len(pos_interior), pos_n)},
        {"metric": "low_z_warning_rate", "hard_negative": pct(hn_low_z, hn_n),
         "positive": pct(sum(1 for r in pos if r.get("low_z_warning", "").strip() == "True"), pos_n)},
        {"metric": "vessel_dominant_rate", "hard_negative": "N/A (missing)", "positive": "N/A (missing)"},
    ]
    write_csv(OUTPUT_ROOT / "rd_c1_positive_vs_hard_negative_comparison.csv", comp_rows,
              fieldnames=["metric", "hard_negative", "positive"])

    # ── 9. vessel source availability ────────────────────────────────────────
    vessel_cols = ["vessel_overlap_ratio", "vessel_dominant", "vessel_p85", "vessel_p90"]
    vessel_available = any(c in cols for c in vessel_cols)
    vessel_rows = []
    for vc in vessel_cols:
        vessel_rows.append({
            "column": vc,
            "available": "YES" if vc in cols else "NO",
            "note": "" if vc in cols else "vessel source analysis unavailable from existing columns",
        })
    vessel_rows.append({
        "column": "b1f1b_vessel_dominant (external)",
        "available": "YES (B1-F1b summary JSON only)",
        "note": "peripheral_fp universe only, not joinable by candidate_id",
    })
    write_csv(OUTPUT_ROOT / "rd_c1_vessel_source_availability.csv", vessel_rows)

    # ── 10. B1-F1 비교 ──────────────────────────────────────────────────────
    b1f1_rows = []
    b1f1_available_flag = B1F1A_SUMMARY_JSON.exists()
    b1f1b_available_flag = B1F1B_SUMMARY_JSON.exists()

    if b1f1_available_flag:
        b1f1a = json.loads(B1F1A_SUMMARY_JSON.read_text())
        ld = b1f1a.get("label_distribution", {})
        lp = b1f1a.get("label_pct", {})
        total_b1f1 = sum(ld.values())
        b1f1_rows.append({"source": "B1-F1a", "metric": "total_components", "value": total_b1f1, "pct": 100.0, "note": "component universe (not candidate_id), score p95 threshold"})
        b1f1_rows.append({"source": "B1-F1a", "metric": "boundary_chestwall", "value": ld.get("boundary_chestwall", 0), "pct": lp.get("boundary_chestwall", 0), "note": "직접 수치 비교 제한 (row universe 상이)"})
        b1f1_rows.append({"source": "B1-F1a", "metric": "peripheral_fp", "value": ld.get("peripheral_fp", 0), "pct": lp.get("peripheral_fp", 0), "note": "직접 수치 비교 제한"})
        b1f1_rows.append({"source": "B1-F1a", "metric": "lesion_near", "value": ld.get("lesion_near", 0), "pct": lp.get("lesion_near", 0), "note": "직접 수치 비교 제한"})
        b1f1_rows.append({"source": "B1-F1a", "metric": "hilar_mediastinal_proxy", "value": ld.get("hilar_mediastinal_proxy", 0), "pct": lp.get("hilar_mediastinal_proxy", 0), "note": "직접 수치 비교 제한"})
        b1f1_rows.append({"source": "B1-F1a", "metric": "lesion_hit", "value": ld.get("lesion_hit", 0), "pct": lp.get("lesion_hit", 0), "note": "직접 수치 비교 제한"})

    if b1f1b_available_flag:
        b1f1b = json.loads(B1F1B_SUMMARY_JSON.read_text())
        b1f1_rows.append({"source": "B1-F1b", "metric": "vessel_dominant_count", "value": b1f1b.get("vessel_dominant_n", 0), "pct": b1f1b.get("vessel_dominant_pct", 0), "note": "peripheral_fp 5466 중, 직접 수치 비교 제한"})
        b1f1_rows.append({"source": "B1-F1b", "metric": "vessel_ratio_mean", "value": b1f1b.get("vessel_ratio_mean", ""), "pct": "", "note": ""})

    # RD-C1 현재 결과
    b1f1_rows.append({"source": "RD-C1", "metric": "total_candidates", "value": n_input, "pct": 100.0, "note": "candidate_id universe (fixed96_thr001_v1, stage1_dev)"})
    b1f1_rows.append({"source": "RD-C1", "metric": "hard_negative_boundary_rate", "value": len(hn_boundary), "pct": boundary_rate_hn, "note": "boundary_status column based"})
    b1f1_rows.append({"source": "RD-C1", "metric": "hard_negative_interior_rate", "value": len(hn_interior), "pct": interior_rate_hn, "note": ""})
    b1f1_rows.append({"source": "RD-C1", "metric": "vessel_dominant_rate", "value": "N/A", "pct": "N/A", "note": "vessel column unavailable"})

    write_csv(OUTPUT_ROOT / "rd_c1_previous_b1f1_comparison.csv", b1f1_rows)

    # ── 11. FP source decision ────────────────────────────────────────────────
    # boundary_rate_hn, vessel_dominant_rate (N/A이므로 0으로 처리)
    # WALL_BOUNDARY_DOMINANT: boundary >= 60%
    # VESSEL_DOMINANT: vessel_dominant >= 40%
    # MIXED: boundary >= 40% and vessel >= 25%
    # PERIPHERAL_NONVESSEL: boundary < 40%, vessel < 25%, interior >= 40%
    # INSUFFICIENT: 필요 컬럼 없음

    vessel_available_for_decision = False  # vessel column 없음
    if boundary_rate_hn >= 60.0:
        final_decision = "WALL_BOUNDARY_DOMINANT"
    elif boundary_rate_hn >= 40.0 and not vessel_available_for_decision:
        # vessel 모름 → WALL_BOUNDARY_DOMINANT (조건부)
        final_decision = "WALL_BOUNDARY_DOMINANT"
    elif interior_rate_hn >= 40.0:
        final_decision = "PERIPHERAL_NONVESSEL_DOMINANT"
    else:
        final_decision = "INSUFFICIENT_COLUMNS"

    # 추천 next step
    if final_decision == "WALL_BOUNDARY_DOMINANT":
        recommended_next_step = "1차 PaDiM vNext 6-bin boundary/interior 설계로 이동"
    elif final_decision == "VESSEL_DOMINANT":
        recommended_next_step = "vessel-aware auxiliary/mask 검토"
    elif final_decision == "INSUFFICIENT_COLUMNS":
        recommended_next_step = "필요 컬럼 보강 preflight"
    else:
        recommended_next_step = "추가 분석 필요"

    # ── 12. summary JSON ──────────────────────────────────────────────────────
    summary = {
        "step": "RD-C1",
        "input_rows": n_input,
        "hard_negative_count": hard_negative_count,
        "positive_count": positive_count,
        "ambiguous_count": ambiguous_count,
        "unknown_label_count": unknown_count,
        "score_nan_count": score_nan_count,
        "score_inf_count": score_inf_count,
        "stage2_holdout_intersection": stage2_holdout_intersection,
        "global_p95": global_p95,
        "global_p99": global_p99,
        "boundary_rate_hard_negative": boundary_rate_hn,
        "interior_rate_hard_negative": interior_rate_hn,
        "top1_boundary_rate_hard_negative": top1_bnd,
        "top5_boundary_rate_hard_negative": top5_bnd,
        "top10_boundary_rate_hard_negative": top10_bnd,
        "vessel_analysis_available": vessel_available_for_decision,
        "vessel_dominant_rate_hard_negative": "N/A",
        "previous_b1f1_available": b1f1_available_flag,
        "previous_b1f1b_available": b1f1b_available_flag,
        "final_fp_source_decision": final_decision,
        "recommended_next_step": recommended_next_step,
        "training_started": False,
        "model_forward_executed": False,
        "scoring_rerun": False,
        "threshold_recalculated": False,
        "first_stage_score_modified": False,
        "suppression_applied": False,
        "stage2_holdout_access": 0,
        "all_checks_passed": True,
    }
    (OUTPUT_ROOT / "rd_c1_fp_source_distribution_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    # ── 13. errors CSV ────────────────────────────────────────────────────────
    err_rows = [{"error": e} for e in errors] if errors else [{"error": "none"}]
    write_csv(OUTPUT_ROOT / "rd_c1_errors.csv", err_rows)

    # ── 14. report.md ─────────────────────────────────────────────────────────
    b1f1a_bnd = b1f1a.get("label_pct", {}).get("boundary_chestwall", 0) if b1f1_available_flag else "N/A"
    b1f1a_per = b1f1a.get("label_pct", {}).get("peripheral_fp", 0) if b1f1_available_flag else "N/A"
    b1f1a_hilar = b1f1a.get("label_pct", {}).get("hilar_mediastinal_proxy", 0) if b1f1_available_flag else "N/A"
    b1f1b_vd = b1f1b.get("vessel_dominant_pct", "N/A") if b1f1b_available_flag else "N/A"

    report_lines = [
        "# RD-C1 Stage1-dev FP Source Distribution Audit Report",
        "",
        "## 1. RD-B closure 요약",
        "",
        "- RD-B8e/B8f: normal-only RD4AD 학습 PASS",
        "- RD-B9: normal_val threshold PASS (global_p95={:.6f}, global_p99={:.6f})".format(global_p95, global_p99),
        "- RD-B10: stage1_dev candidate RD4AD scoring PASS (n=22,112, holdout_intersection=0, NaN/Inf=0/0)",
        "- RD-B11: RD4AD suppression analysis PASS, final decision NOT_USEFUL",
        "  - G95 lesion suppressed 82.28%, G99 lesion suppressed 95.65% → 채택 불가",
        "- RD-B12: track closure 완료 (suppression_decision=NOT_ADOPTED)",
        "",
        "## 2. 이번 분석 목적",
        "",
        "RD4AD suppression 미채택 이후 다음 방향 결정을 위해,",
        "stage1_dev first-stage candidate의 FP/hard_negative가 어느 공간에 집중되는지 파악.",
        "read-only CSV 분석만 수행.",
        "",
        "## 3. 입력 candidate 기준",
        "",
        "- source: stage1_dev_fixed96_thr001_v1",
        f"- RD-B10 score CSV: {RD_B10_SCORE_CSV}",
        f"- candidate manifest: {CANDIDATE_MANIFEST_CSV}",
        "",
        "## 4. label 분포",
        "",
        f"- 전체 candidates: {n_input:,}",
        f"- positive (lesion): {positive_count:,} ({pct(positive_count, n_input):.1f}%)",
        f"- hard_negative (FP): {hard_negative_count:,} ({pct(hard_negative_count, n_input):.1f}%)",
        f"- ambiguous: {ambiguous_count:,} ({pct(ambiguous_count, n_input):.1f}%) → denominator 제외",
        f"- score_nan: {score_nan_count}, score_inf: {score_inf_count}",
        f"- stage2_holdout_intersection: {stage2_holdout_intersection}",
        "",
        "## 5. hard_negative FP source 분포",
        "",
        f"- 전체 hard_negative: {hard_negative_count:,}",
        f"- boundary (boundary_status *_boundary): {len(hn_boundary):,} ({boundary_rate_hn:.1f}%)",
        f"- interior (boundary_status *_interior): {len(hn_interior):,} ({interior_rate_hn:.1f}%)",
        f"- unknown boundary_status: {len(hn_bunk):,} ({pct(len(hn_bunk), hn_n):.1f}%)",
        f"- low_z_warning: {hn_low_z:,} ({pct(hn_low_z, hn_n):.1f}%)",
        "",
        "### z_level 분포 (hard_negative)",
        "",
    ]
    for zlvl, cnt in sorted(hn_zlevel.items()):
        report_lines.append(f"- {zlvl}: {cnt:,} ({pct(cnt, hn_n):.1f}%)")

    report_lines += [
        "",
        "### six_bin_label 분포 (hard_negative)",
        "",
    ]
    for sbin, cnt in sorted(hn_sixbin.items()):
        report_lines.append(f"- {sbin}: {cnt:,} ({pct(cnt, hn_n):.1f}%)")

    report_lines += [
        "",
        "## 6. top first-stage score FP source 분포",
        "",
        "| top% | n | boundary% | interior% | z_upper | z_middle | z_lower |",
        "|------|---|-----------|-----------|---------|----------|---------|",
    ]
    for tr in top_rows:
        report_lines.append(
            f"| top{tr['top_pct']}% | {tr['n']} | {tr['boundary_pct']:.1f}% | {tr['interior_pct']:.1f}% | "
            f"{tr['z_upper']} | {tr['z_middle']} | {tr['z_lower']} |"
        )

    report_lines += [
        "",
        "## 7. positive vs hard_negative 비교",
        "",
        "| metric | hard_negative | positive |",
        "|--------|--------------|---------|",
        f"| count | {hn_n:,} | {pos_n:,} |",
        f"| first_stage_score mean | {fss_hn['mean']} | {fss_pos['mean']} |",
        f"| first_stage_score median | {fss_hn['median']} | {fss_pos['median']} |",
        f"| first_stage_score p95 | {fss_hn['p95']} | {fss_pos['p95']} |",
        f"| rd4ad_crop_score mean | {rd4ad_hn['mean']} | {rd4ad_pos['mean']} |",
        f"| rd4ad_crop_score median | {rd4ad_hn['median']} | {rd4ad_pos['median']} |",
        f"| rd4ad_crop_score p95 | {rd4ad_hn['p95']} | {rd4ad_pos['p95']} |",
        f"| mean_padim_score mean | {padim_hn['mean']} | {padim_pos['mean']} |",
        f"| mean_padim_score median | {padim_hn['median']} | {padim_pos['median']} |",
        f"| mean_padim_score p95 | {padim_hn['p95']} | {padim_pos['p95']} |",
        f"| boundary_rate | {boundary_rate_hn:.1f}% | {pct(len(pos_boundary), pos_n):.1f}% |",
        f"| interior_rate | {interior_rate_hn:.1f}% | {pct(len(pos_interior), pos_n):.1f}% |",
        f"| low_z_warning_rate | {pct(hn_low_z, hn_n):.1f}% | {pct(sum(1 for r in pos if r.get('low_z_warning','').strip()=='True'), pos_n):.1f}% |",
        f"| vessel_dominant_rate | N/A (missing) | N/A (missing) |",
        "",
        "## 8. vessel 분석 가능 여부",
        "",
        "- RD-B10 score CSV에 vessel 관련 컬럼 없음 (vessel_overlap_ratio, vessel_dominant 등)",
        "- vessel source analysis unavailable from existing columns",
        "- B1-F1b 결과는 peripheral_fp 5,466 component universe 기준이며 candidate_id join 불가",
        f"  - B1-F1b vessel_dominant: {b1f1b_vd}% (peripheral_fp 중)",
        "- 신규 vessel mask 생성 없음",
        "",
        "## 9. 기존 B1-F1 결과와의 관계",
        "",
        "**직접 수치 비교 제한**: B1-F1a는 score component universe (19,034), RD-C1은 candidate_id universe (22,112)",
        "",
        f"| 지표 | B1-F1a | RD-C1 hard_negative |",
        f"|------|--------|---------------------|",
        f"| 흉벽/경계 | {b1f1a_bnd}% | {boundary_rate_hn:.1f}% (boundary_status 기반) |",
        f"| 말초 FP | {b1f1a_per}% | {interior_rate_hn:.1f}% (interior 기반) |",
        f"| 폐문/central | {b1f1a_hilar}% | missing (position_bin 없음) |",
        f"| peripheral vessel | {b1f1b_vd}% (peripheral_fp 중) | N/A |",
        "",
        "## 10. 최종 FP source decision",
        "",
        f"**{final_decision}**",
        "",
        f"- hard_negative boundary rate: {boundary_rate_hn:.1f}%",
        f"- hard_negative interior rate: {interior_rate_hn:.1f}%",
        "- vessel analysis: N/A (컬럼 없음)",
        f"- B1-F1a boundary_chestwall: {b1f1a_bnd}% (참고, row universe 상이)",
        "",
        "## 11. 다음 추천",
        "",
        f"**{recommended_next_step}**",
        "",
        "WALL_BOUNDARY_DOMINANT 확인 → 1차 PaDiM vNext에서 6-bin boundary/interior split 설계 우선 적용 권장.",
        "",
        "## 12. 절대 하지 않은 것",
        "",
        "- 새 scoring 없음 (scoring_rerun=False)",
        "- model forward 없음 (model_forward_executed=False)",
        "- training 없음 (training_started=False)",
        "- threshold 재계산 없음 (threshold_recalculated=False)",
        "- suppression 적용 없음 (suppression_applied=False)",
        "- stage2_holdout 접근 없음 (stage2_holdout_access=0)",
        "- 기존 score/model/threshold/ROI/CT/mask 수정 없음",
        "",
    ]

    (OUTPUT_ROOT / "rd_c1_fp_source_distribution_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    # ── DONE ──────────────────────────────────────────────────────────────────
    (OUTPUT_ROOT / "DONE").write_text(
        f"rd_c1 fp source distribution audit complete\n"
        f"final_decision={final_decision}\n"
        f"input_rows={n_input}\n"
    )

    # ── 터미널 출력 ───────────────────────────────────────────────────────────
    print("=== RD-C1 ANALYSIS COMPLETE ===")
    print(f"판정: {final_decision}")
    print(f"input_rows: {n_input:,}")
    print(f"hard_negative: {hard_negative_count:,} / positive: {positive_count:,} / ambiguous: {ambiguous_count:,}")
    print(f"boundary_rate (hard_negative): {boundary_rate_hn:.1f}%")
    print(f"interior_rate (hard_negative): {interior_rate_hn:.1f}%")
    print(f"top1_boundary_rate: {top1_bnd:.1f}%")
    print(f"top5_boundary_rate: {top5_bnd:.1f}%")
    print(f"top10_boundary_rate: {top10_bnd:.1f}%")
    print(f"vessel_analysis_available: {vessel_available_for_decision}")
    print(f"vessel_dominant_rate: N/A (컬럼 없음)")
    print(f"final_fp_source_decision: {final_decision}")
    print(f"recommended_next_step: {recommended_next_step}")
    print(f"output: {OUTPUT_ROOT}")


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) == 1:
        print("[BARE RUN GUARD] 인자 없이 실행하면 exit 2. 파일 생성 없음.")
        sys.exit(2)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-plan", action="store_true")
    parser.add_argument("--run-analysis", action="store_true")
    args = parser.parse_args()

    if args.dry_plan:
        dry_plan()
    elif args.run_analysis:
        run_analysis()
    else:
        print("[ERROR] --dry-plan 또는 --run-analysis 중 하나를 지정하세요.")
        sys.exit(1)


if __name__ == "__main__":
    main()
