"""
RD4AD z-continuity group-level full scoring MERGE v1

shard_0~3 group scores 를 병합/검증하고, group-level top-k 평가 및
report/summary 를 생성한다.

실행 방식:
  bare run (인자 없음): exit 2 로 막음
  dry-run:      python <script> --dry-run
  merge-shards: python <script> --merge-shards --confirm-readonly --confirm-stage1dev-only

입력 (read-only):
  shards/shard_{id}/group_scores_shard_{id}.csv  (0~3)
  shards/shard_{id}/DONE.json
  shards/shard_{id}/shard_{id}_summary.json
  manifests/group_manifest_full.csv
  manifests/full_scoring_shard_plan.csv
  RD-D1s scalar score CSV (patch baseline 비교용)

출력 (생성):
  manifests/group_scores_full_merged.csv
  manifests/group_scores_full_patient_topk.csv
  manifests/group_scores_full_problem_patient_audit.csv
  reports/rd4ad_z_continuity_group_full_scoring_report.md
  reports/rd4ad_z_continuity_group_full_scoring_summary.json
  DONE.json   (※ 주의: preflight DONE.json 을 최종 merge 판정으로 덮어씀)

금지:
  파일 추가 model forward / training / stage2_holdout 접근 /
  입력 manifest(group_manifest_full/group_representative_manifest_full/shard_plan) 덮어쓰기 /
  preflight report·summary 덮어쓰기 / 기존 외부 artifact 수정.
"""
import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

# =============================================================================
# 경로 상수
# =============================================================================

PROJECT_ROOT    = Path("/home/jinhy/project/lung-ct-anomaly")
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments/rd4ad_z_continuity_group_full_scoring_v1"

RD_D1S_SCORE_CSV = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_d1s_medi3ch_true_rd4ad_shard_run_v1"
    / "rd_d1s_stage1dev_candidate_score.csv"
)

MANIFEST_DIR = EXPERIMENT_ROOT / "manifests"
REPORT_DIR   = EXPERIMENT_ROOT / "reports"
SHARDS_DIR   = EXPERIMENT_ROOT / "shards"

# 입력 manifest (read-only, 덮어쓰기 금지)
GROUP_MANIFEST_CSV      = MANIFEST_DIR / "group_manifest_full.csv"
GROUP_REPR_MANIFEST_CSV = MANIFEST_DIR / "group_representative_manifest_full.csv"
SHARD_PLAN_CSV          = MANIFEST_DIR / "full_scoring_shard_plan.csv"

# preflight 산출물 (read-only, 덮어쓰기 금지)
PREFLIGHT_REPORT_MD   = REPORT_DIR / "rd4ad_z_continuity_group_full_scoring_preflight_report.md"
PREFLIGHT_SUMMARY_JSON = REPORT_DIR / "rd4ad_z_continuity_group_full_scoring_preflight_summary.json"

# 출력
MERGED_CSV            = MANIFEST_DIR / "group_scores_full_merged.csv"
PATIENT_TOPK_CSV      = MANIFEST_DIR / "group_scores_full_patient_topk.csv"
PROBLEM_AUDIT_CSV     = MANIFEST_DIR / "group_scores_full_problem_patient_audit.csv"
REPORT_MD             = REPORT_DIR / "rd4ad_z_continuity_group_full_scoring_report.md"
SUMMARY_JSON          = REPORT_DIR / "rd4ad_z_continuity_group_full_scoring_summary.json"
DONE_JSON             = EXPERIMENT_ROOT / "DONE_FULL_MERGE.json"

SHARD_COUNT = 4
EXPECTED_TOTAL_GROUPS = 20216

TOPK_VALS = [1, 3, 5, 10, 20, 50]
SCORE_COLS = [
    "rd4ad_group_score_raw",   # primary
    "P1_times_roi_mean", "P2_times_sqrt_roi_mean",
    "P3_soft_alpha_0_2", "P4_soft_alpha_0_3",  # preview only
]
PRIMARY_SCORE_COL = "rd4ad_group_score_raw"

# boundary-heavy 진단 임계 (preview/진단 전용, 삭제 rule 아님)
BOUNDARY_HEAVY_THRESHOLD = 0.5

PROBLEM_PATIENT_IDS = ["LUNG1-086", "LUNG1-386", "LUNG1-399"]

# =============================================================================
# guardrail
# =============================================================================

GUARDRAILS = {
    "stage2_holdout_accessed":     False,
    "checkpoint_loaded":           False,
    "model_forward_executed":      False,
    "training_executed":           False,
    "backward_executed":           False,
    "optimizer_created":           False,
    "checkpoint_saved":            False,
    "crop_generation_executed":    False,
    "full_scoring_executed":       "merge_only",
    "threshold_recalculated":      False,
    "existing_artifact_modified":  False,
    "existing_script_modified":    False,
    "output_overwrite":            False,
    "raw_rd4ad_primary_score":     True,
    "adjusted_score_preview_only": True,
}

# =============================================================================
# 안전 경로 검사
# =============================================================================

def assert_path_safe(p: Path):
    s = str(p).lower()
    if "stage2_holdout" in s or ("stage2" in s and "holdout" in s):
        GUARDRAILS["stage2_holdout_accessed"] = True
        raise RuntimeError(f"[ABORT] stage2_holdout 경로 접근 차단: {p}")


_PROTECTED = {
    GROUP_MANIFEST_CSV.resolve(),
    GROUP_REPR_MANIFEST_CSV.resolve(),
    SHARD_PLAN_CSV.resolve(),
    PREFLIGHT_REPORT_MD.resolve(),
    PREFLIGHT_SUMMARY_JSON.resolve(),
}

_ALLOWED_OUT = {
    MERGED_CSV.resolve(), PATIENT_TOPK_CSV.resolve(), PROBLEM_AUDIT_CSV.resolve(),
    REPORT_MD.resolve(), SUMMARY_JSON.resolve(), DONE_JSON.resolve(),
}


def ensure_output_path_safe(p: Path):
    rp = Path(p).resolve()
    if rp in _PROTECTED:
        GUARDRAILS["existing_artifact_modified"] = True
        raise RuntimeError(f"[ABORT] 입력/preflight artifact 덮어쓰기 차단: {p}")
    if rp not in _ALLOWED_OUT:
        GUARDRAILS["existing_artifact_modified"] = True
        raise RuntimeError(f"[ABORT] 허용되지 않은 출력 경로: {p}")


# =============================================================================
# CSV 유틸
# =============================================================================

def read_csv(path: Path):
    assert_path_safe(path)
    rows = []
    with open(str(path), encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    return rows


def write_csv(path: Path, fieldnames, rows):
    ensure_output_path_safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  saved: {path} ({len(rows)} rows)")


def _to_float(v, default=None):
    if v is None or v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


# =============================================================================
# top-k 평가
# =============================================================================

def _patient_groups(merged_rows):
    pat = defaultdict(list)
    for r in merged_rows:
        pat[r["patient_id"]].append(r)
    return pat


def eval_topk(merged_rows, score_col):
    """positive-bearing 환자 기준 top-k 지표 집합 반환."""
    pat = _patient_groups(merged_rows)

    total_pos_groups = sum(1 for r in merged_rows if r.get("has_positive") == "True")
    total_pos_cands  = sum(int(_to_float(r.get("positive_count"), 0)) for r in merged_rows)

    out = {}
    for k in TOPK_VALS:
        n_pos_patients = 0
        n_hit = 0
        in_topk_pos_groups = 0
        in_topk_pos_cands  = 0
        boundary_heavy = 0
        topk_groups_examined = 0

        for pid, rows in pat.items():
            has_pos_patient = any(r.get("has_positive") == "True" for r in rows)
            scored = sorted(rows, key=lambda r: _to_float(r.get(score_col), -1e9), reverse=True)
            topk = scored[:k]
            # boundary-heavy 는 모든 환자 top-k 대상으로 집계
            for r in topk:
                topk_groups_examined += 1
                blr = _to_float(r.get("boundary_like_ratio_mean"), None)
                if blr is not None and blr >= BOUNDARY_HEAVY_THRESHOLD:
                    boundary_heavy += 1
            if not has_pos_patient:
                continue
            n_pos_patients += 1
            if any(r.get("has_positive") == "True" for r in topk):
                n_hit += 1
            for r in topk:
                if r.get("has_positive") == "True":
                    in_topk_pos_groups += 1
                    in_topk_pos_cands  += int(_to_float(r.get("positive_count"), 0))

        retention = round(n_hit / n_pos_patients, 4) if n_pos_patients else 0.0
        out[k] = {
            "lesion_group_retention":        retention,
            "n_patients_with_positive":      n_pos_patients,
            "positive_group_coverage":       round(in_topk_pos_groups / total_pos_groups, 4) if total_pos_groups else 0.0,
            "positive_candidate_coverage":   round(in_topk_pos_cands / total_pos_cands, 4) if total_pos_cands else 0.0,
            "boundary_heavy_group_count":    boundary_heavy,
            "boundary_heavy_group_ratio":    round(boundary_heavy / topk_groups_examined, 4) if topk_groups_examined else 0.0,
            "pat_all_sup_proxy":             round(1.0 - retention, 4),
        }
    return out


def patch_topk_baseline():
    """patch-level RD-D1s baseline top-k retention."""
    if not RD_D1S_SCORE_CSV.exists():
        return {}
    rows = read_csv(RD_D1S_SCORE_CSV)
    pat = defaultdict(list)
    for r in rows:
        if r.get("stage_split", "") != "stage1_dev":
            continue
        pat[r["patient_id"]].append(r)
    base = {}
    for k in TOPK_VALS:
        hit = total = 0
        for pid, rs in pat.items():
            if not any(r.get("label") == "positive" for r in rs):
                continue
            total += 1
            scored = sorted(rs, key=lambda r: _to_float(r.get("rd_d1s_medi3ch_rd4ad_score"), -1e9), reverse=True)
            if any(r.get("label") == "positive" for r in scored[:k]):
                hit += 1
        base[k] = round(hit / total, 4) if total else 0.0
    return base


# =============================================================================
# dry-run
# =============================================================================

def run_dry():
    print("=" * 70)
    print("[DRY-RUN] RD4AD z-continuity group-level full scoring MERGE v1")
    print("=" * 70)
    issues = []

    print("\n[1] shard 출력 / DONE 확인")
    total_expected = 0
    for sid in range(SHARD_COUNT):
        d = SHARDS_DIR / f"shard_{sid}"
        csv_p  = d / f"group_scores_shard_{sid}.csv"
        done_p = d / "DONE.json"
        summ_p = d / f"shard_{sid}_summary.json"
        ok_csv  = csv_p.exists()
        ok_done = done_p.exists()
        print(f"  shard {sid}: csv={'OK' if ok_csv else 'MISSING'} done={'OK' if ok_done else 'MISSING'}")
        if not ok_csv:
            issues.append(f"shard {sid} csv 없음")
        if not ok_done:
            issues.append(f"shard {sid} DONE.json 없음")
        if summ_p.exists():
            try:
                with open(str(summ_p), encoding="utf-8") as f:
                    s = json.load(f)
                exp = s.get("expected_group_count", 0)
                total_expected += int(exp)
                print(f"           expected={exp} scored={s.get('actual_scored_group_count')} "
                      f"failed={s.get('failed_group_count')} verdict={s.get('verdict')}")
            except Exception as e:
                issues.append(f"shard {sid} summary 읽기 실패: {e}")

    print(f"\n[2] expected group count 합계: {total_expected:,} (목표 {EXPECTED_TOTAL_GROUPS:,})")
    if total_expected and total_expected != EXPECTED_TOTAL_GROUPS:
        issues.append(f"expected 합계 {total_expected} != {EXPECTED_TOTAL_GROUPS}")

    print("\n[3] 출력 overwrite 위험")
    for p in [MERGED_CSV, PATIENT_TOPK_CSV, PROBLEM_AUDIT_CSV, REPORT_MD, SUMMARY_JSON]:
        print(f"  {'WARN exists' if p.exists() else 'OK'}: {p.name}")
    if DONE_JSON.exists():
        print(f"  [주의] {DONE_JSON.name} 존재(preflight) → merge 시 최종 판정으로 덮어씀")

    print("\n[4] 입력 manifest read-only 확인")
    for p in [GROUP_MANIFEST_CSV, SHARD_PLAN_CSV]:
        print(f"  {'OK' if p.exists() else 'MISSING'}: {p.name}")

    print("\n" + "=" * 70)
    if issues:
        print("[DRY-RUN] 이슈:")
        for it in issues:
            print(f"  - {it}")
        print("판정: NEEDS_FIX (shard 미완료 또는 불일치)")
    else:
        print("[DRY-RUN] shard 출력/DONE OK, merge 준비됨")
        print("판정: READY_TO_MERGE")
    print("=" * 70)


# =============================================================================
# merge-shards
# =============================================================================

def run_merge():
    print("=" * 70)
    print("[MERGE] RD4AD z-continuity group-level full scoring MERGE v1")
    print("=" * 70)
    t0 = time.perf_counter()
    fail_reasons = []

    # 1. shard CSV / summary 로드
    print("\n[1] shard CSV 로드")
    merged_rows = []
    shard_row_counts = {}
    shard_summaries = {}
    for sid in range(SHARD_COUNT):
        d = SHARDS_DIR / f"shard_{sid}"
        csv_p = d / f"group_scores_shard_{sid}.csv"
        done_p = d / "DONE.json"
        summ_p = d / f"shard_{sid}_summary.json"
        if not csv_p.exists():
            fail_reasons.append(f"shard {sid} csv 없음")
            continue
        if not done_p.exists():
            fail_reasons.append(f"shard {sid} DONE.json 없음")
        rows = read_csv(csv_p)
        shard_row_counts[sid] = len(rows)
        merged_rows.extend(rows)
        if summ_p.exists():
            with open(str(summ_p), encoding="utf-8") as f:
                shard_summaries[sid] = json.load(f)
        print(f"  shard {sid}: {len(rows):,} rows")

    if fail_reasons:
        print("  [FAIL] shard 입력 누락:")
        for r in fail_reasons:
            print(f"    - {r}")
        _finalize(merged_rows, {}, {}, {}, shard_row_counts, shard_summaries,
                  "FAIL", fail_reasons, time.perf_counter() - t0)
        sys.exit(1)

    merged_row_count = len(merged_rows)
    print(f"  merged rows: {merged_row_count:,}")

    # 2. 무결성 검증
    print("\n[2] 무결성 검증")
    gids = [r["group_id"] for r in merged_rows]
    gid_set = set(gids)
    duplicate_count = len(gids) - len(gid_set)

    plan_rows = read_csv(SHARD_PLAN_CSV)
    plan_gids = set(r["group_id"] for r in plan_rows)
    manifest_gids = set(r["group_id"] for r in read_csv(GROUP_MANIFEST_CSV))

    missing_vs_plan = plan_gids - gid_set
    missing_vs_manifest = manifest_gids - gid_set
    extra_vs_plan = gid_set - plan_gids

    # shard plan 별 기대 row count
    plan_shard_counts = defaultdict(int)
    for r in plan_rows:
        plan_shard_counts[int(r["shard_id"])] += 1
    shard_count_ok = all(shard_row_counts.get(s, 0) == plan_shard_counts.get(s, 0)
                         for s in range(SHARD_COUNT))

    # NaN/Inf
    nan_count = inf_count = 0
    for r in merged_rows:
        v = _to_float(r.get(PRIMARY_SCORE_COL), None)
        if v is None:
            continue
        if math.isnan(v):
            nan_count += 1
        if math.isinf(v):
            inf_count += 1

    print(f"  merged_row_count       : {merged_row_count:,}")
    print(f"  duplicate_group_id     : {duplicate_count}")
    print(f"  missing_vs_plan        : {len(missing_vs_plan)}")
    print(f"  missing_vs_manifest    : {len(missing_vs_manifest)}")
    print(f"  extra_vs_plan          : {len(extra_vs_plan)}")
    print(f"  shard_count_ok         : {shard_count_ok}")
    print(f"  score NaN / Inf        : {nan_count} / {inf_count}")

    if merged_row_count != EXPECTED_TOTAL_GROUPS:
        fail_reasons.append(f"merged row count {merged_row_count} != {EXPECTED_TOTAL_GROUPS}")
    if duplicate_count != 0:
        fail_reasons.append(f"duplicate group_id {duplicate_count}")
    if missing_vs_plan:
        fail_reasons.append(f"missing vs plan {len(missing_vs_plan)}")
    if missing_vs_manifest:
        fail_reasons.append(f"missing vs manifest {len(missing_vs_manifest)}")
    if extra_vs_plan:
        fail_reasons.append(f"extra vs plan {len(extra_vs_plan)}")
    if not shard_count_ok:
        fail_reasons.append("shard row count != shard plan")
    if nan_count or inf_count:
        fail_reasons.append(f"NaN/Inf score {nan_count}/{inf_count}")

    # shard summary failed/guardrail 집계
    total_failed = sum(int(s.get("failed_group_count", 0)) for s in shard_summaries.values())
    any_stage2 = any(s.get("stage2_holdout_accessed", False) for s in shard_summaries.values())
    any_existing_mod = any(s.get("existing_artifact_modified", False) for s in shard_summaries.values())
    if total_failed != 0:
        fail_reasons.append(f"shard failed_group_count 합계 {total_failed}")
    if any_stage2:
        fail_reasons.append("어떤 shard 에서 stage2_holdout 접근")
        GUARDRAILS["stage2_holdout_accessed"] = True
    if any_existing_mod:
        fail_reasons.append("어떤 shard 에서 existing_artifact_modified")

    # 3. top-k 평가
    print("\n[3] group-level top-k 평가")
    topk_by_col = {}
    for sc in SCORE_COLS:
        topk_by_col[sc] = eval_topk(merged_rows, sc)
    for k in TOPK_VALS:
        m = topk_by_col[PRIMARY_SCORE_COL][k]
        print(f"  raw top{k:2d}: retention={m['lesion_group_retention']:.4f} "
              f"pos_grp_cov={m['positive_group_coverage']:.4f} "
              f"pos_cand_cov={m['positive_candidate_coverage']:.4f} "
              f"bheavy={m['boundary_heavy_group_count']}")

    patch_base = patch_topk_baseline()
    if patch_base:
        for k in TOPK_VALS:
            print(f"  patch baseline top{k:2d}: {patch_base[k]:.4f}")

    # primary top10 이 patch baseline 보다 악화되면 FAIL
    raw_top10 = topk_by_col[PRIMARY_SCORE_COL].get(10, {}).get("lesion_group_retention", 0.0)
    patch_top10 = patch_base.get(10, 0.0)
    if patch_base and raw_top10 < patch_top10:
        fail_reasons.append(f"raw top10 {raw_top10} < patch baseline {patch_top10}")

    # 4. problem patient audit
    print("\n[4] problem patient audit")
    problem_rows = []
    problem_summary = {}
    pat = _patient_groups(merged_rows)
    for pid in PROBLEM_PATIENT_IDS:
        rows = pat.get(pid, [])
        n_groups = len(rows)
        n_pos_groups = sum(1 for r in rows if r.get("has_positive") == "True")
        scored = sorted(rows, key=lambda r: _to_float(r.get(PRIMARY_SCORE_COL), -1e9), reverse=True)
        for k in TOPK_VALS:
            has_pos = any(r.get("has_positive") == "True" for r in scored[:k])
            problem_rows.append({
                "patient_id": pid, "n_groups": n_groups, "n_pos_groups": n_pos_groups,
                "k": k, "topk_has_positive": str(has_pos),
            })
            problem_summary[f"{pid}_top{k}_has_positive"] = str(has_pos)
        print(f"  {pid}: groups={n_groups} pos_groups={n_pos_groups} "
              f"top20={problem_summary.get(f'{pid}_top20_has_positive')} "
              f"top50={problem_summary.get(f'{pid}_top50_has_positive')}")

    # 5. 판정
    verdict = "PASS"
    if fail_reasons:
        # missing/duplicate/NaN/Inf/stage2/악화 → FAIL ; failed group 만 → PARTIAL_PASS
        hard_fail_keys = ["merged row count", "duplicate", "missing", "extra",
                          "NaN/Inf", "stage2_holdout", "existing_artifact", "patch baseline",
                          "shard row count", "shard", "csv 없음", "DONE.json 없음"]
        is_hard = any(any(k in r for k in hard_fail_keys) for r in fail_reasons)
        verdict = "FAIL" if is_hard else "PARTIAL_PASS"

    _finalize(merged_rows, topk_by_col, patch_base, problem_summary,
              shard_row_counts, shard_summaries, verdict, fail_reasons,
              time.perf_counter() - t0,
              integrity={
                  "merged_row_count": merged_row_count,
                  "duplicate_group_id": duplicate_count,
                  "missing_vs_plan": len(missing_vs_plan),
                  "missing_vs_manifest": len(missing_vs_manifest),
                  "extra_vs_plan": len(extra_vs_plan),
                  "shard_count_ok": shard_count_ok,
                  "score_nan_count": nan_count,
                  "score_inf_count": inf_count,
                  "total_failed_group_count": total_failed,
              },
              problem_rows=problem_rows)


# =============================================================================
# 출력 작성
# =============================================================================

def _finalize(merged_rows, topk_by_col, patch_base, problem_summary,
              shard_row_counts, shard_summaries, verdict, fail_reasons, elapsed,
              integrity=None, problem_rows=None):
    integrity = integrity or {}
    problem_rows = problem_rows or []

    # merged CSV (입력 shard 컬럼 보존)
    print("\n[5] merged CSV 저장")
    if merged_rows:
        fields = list(merged_rows[0].keys())
        # group_id 정렬
        merged_sorted = sorted(merged_rows, key=lambda r: r["group_id"])
        write_csv(MERGED_CSV, fields, merged_sorted)

    # patient_topk CSV
    print("\n[6] patient top-k CSV 저장")
    topk_rows = []
    for sc, kmap in topk_by_col.items():
        for k in TOPK_VALS:
            m = kmap[k]
            topk_rows.append({
                "score_col": sc, "k": k, "is_primary": str(sc == PRIMARY_SCORE_COL),
                "lesion_group_retention": m["lesion_group_retention"],
                "n_patients_with_positive": m["n_patients_with_positive"],
                "positive_group_coverage": m["positive_group_coverage"],
                "positive_candidate_coverage": m["positive_candidate_coverage"],
                "boundary_heavy_group_count": m["boundary_heavy_group_count"],
                "boundary_heavy_group_ratio": m["boundary_heavy_group_ratio"],
                "pat_all_sup_proxy": m["pat_all_sup_proxy"],
                "patch_baseline_retention": patch_base.get(k, "") if sc == PRIMARY_SCORE_COL else "",
            })
    if topk_rows:
        write_csv(PATIENT_TOPK_CSV,
                  ["score_col", "k", "is_primary", "lesion_group_retention",
                   "n_patients_with_positive", "positive_group_coverage",
                   "positive_candidate_coverage", "boundary_heavy_group_count",
                   "boundary_heavy_group_ratio", "pat_all_sup_proxy",
                   "patch_baseline_retention"],
                  topk_rows)

    # problem patient audit CSV
    print("\n[7] problem patient audit CSV 저장")
    if problem_rows:
        write_csv(PROBLEM_AUDIT_CSV,
                  ["patient_id", "n_groups", "n_pos_groups", "k", "topk_has_positive"],
                  problem_rows)

    # summary JSON
    print("\n[8] summary JSON 저장")
    raw = topk_by_col.get(PRIMARY_SCORE_COL, {})
    summary = {
        "verdict": verdict,
        "fail_reasons": fail_reasons,
        "expected_total_groups": EXPECTED_TOTAL_GROUPS,
        "shard_row_counts": shard_row_counts,
        "integrity": integrity,
        "raw_topk": {f"top{k}": raw.get(k, {}) for k in TOPK_VALS},
        "patch_baseline_topk": {f"top{k}": patch_base.get(k) for k in TOPK_VALS},
        "problem_patient_topk": problem_summary,
        "preview_score_cols": SCORE_COLS[1:],
        "primary_score_col": PRIMARY_SCORE_COL,
        "elapsed_sec": round(elapsed, 1),
        "guardrails": GUARDRAILS,
        "stage2_holdout_accessed": GUARDRAILS["stage2_holdout_accessed"],
        "shard_summaries": shard_summaries,
    }
    ensure_output_path_safe(SUMMARY_JSON)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(str(SUMMARY_JSON), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  saved: {SUMMARY_JSON}")

    # report MD
    print("\n[9] report MD 저장")
    lines = [
        "# RD4AD z-continuity group-level full scoring report v1",
        "",
        f"**판정: {verdict}**",
        "",
        "## 무결성",
        "",
    ]
    for k, v in integrity.items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## raw RD4AD top-k lesion group retention", ""]
    for k in TOPK_VALS:
        m = raw.get(k, {})
        pb = patch_base.get(k, "N/A")
        lines.append(f"- top{k:2d}: retention={m.get('lesion_group_retention')} "
                     f"pos_grp_cov={m.get('positive_group_coverage')} "
                     f"pos_cand_cov={m.get('positive_candidate_coverage')} "
                     f"bheavy={m.get('boundary_heavy_group_count')} "
                     f"pat_all_sup_proxy={m.get('pat_all_sup_proxy')} "
                     f"patch_baseline={pb}")
    lines += ["", "## problem patient (top-k has_positive)", ""]
    for pid in PROBLEM_PATIENT_IDS:
        cells = [f"top{k}={problem_summary.get(f'{pid}_top{k}_has_positive', 'N/A')}" for k in TOPK_VALS]
        lines.append(f"- {pid}: " + " ".join(cells))
    lines += ["", "## fail reasons", ""]
    if fail_reasons:
        for r in fail_reasons:
            lines.append(f"- {r}")
    else:
        lines.append("- 없음")
    lines += ["", "## guardrails", ""]
    for k, v in GUARDRAILS.items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## 비고",
              "- P1~P4 는 preview only (primary 는 rd4ad_group_score_raw).",
              f"- boundary_heavy threshold = {BOUNDARY_HEAVY_THRESHOLD} (진단 전용, 삭제 rule 아님).",
              "- pat_all_sup_proxy = 1 - lesion_group_retention (top-k 완전 누락 환자 비율 proxy)."]
    ensure_output_path_safe(REPORT_MD)
    with open(str(REPORT_MD), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  saved: {REPORT_MD}")

    # DONE.json (preflight DONE 을 최종 merge 판정으로 덮어씀)
    ensure_output_path_safe(DONE_JSON)
    with open(str(DONE_JSON), "w", encoding="utf-8") as f:
        json.dump({"verdict": verdict, "stage": "full_scoring_merge",
                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, f, indent=2)
    print(f"  saved: {DONE_JSON}")

    print("\n" + "=" * 70)
    print(f"[MERGE] 완료 ({elapsed:.1f}s)  판정: {verdict}")
    print("=" * 70)


# =============================================================================
# main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RD4AD z-continuity group-level full scoring MERGE v1")
    parser.add_argument("--dry-run",                action="store_true")
    parser.add_argument("--merge-shards",           action="store_true")
    parser.add_argument("--confirm-readonly",       action="store_true")
    parser.add_argument("--confirm-stage1dev-only", action="store_true")
    args = parser.parse_args()

    if not any([args.dry_run, args.merge_shards]):
        print("[ABORT] bare run 차단. --dry-run / --merge-shards 사용.", file=sys.stderr)
        sys.exit(2)

    if args.dry_run:
        run_dry()
        return

    if args.merge_shards:
        if not args.confirm_readonly or not args.confirm_stage1dev_only:
            print("[ABORT] --confirm-readonly --confirm-stage1dev-only 필요", file=sys.stderr)
            sys.exit(2)
        run_merge()
        return


if __name__ == "__main__":
    main()
