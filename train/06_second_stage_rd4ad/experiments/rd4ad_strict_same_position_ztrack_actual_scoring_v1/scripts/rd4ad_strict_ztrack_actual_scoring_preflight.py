#!/usr/bin/env python3
"""
RD4AD strict same-position z-track actual scoring preflight v1

대상: T1_minrun2 survived candidate 전체 (92,342개)
      모든 survived candidate를 RD4AD scoring 대상으로 보존
      track 대표 1개 선택 금지, first_stage_score 삭제 금지

Usage:
    python script.py                                           # → exit 2 (bare run blocked)
    python script.py --dry-run                                 # 파일 존재 확인, 생성 없음
    python script.py --run-preflight \\
        --confirm-readonly \\
        --confirm-stage1dev-only                               # 실제 preflight 실행
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT    = Path("/home/jinhy/project/lung-ct-anomaly")
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments/rd4ad_strict_same_position_ztrack_actual_scoring_v1"

SURVIVAL_CSV = (
    PROJECT_ROOT
    / "experiments/rd4ad_strict_same_position_ztrack_survival_preflight_v1"
    / "manifests/ztrack_candidate_survival_minrun2.csv"
)
ZTRACK_MANIFEST = (
    PROJECT_ROOT
    / "experiments/rd4ad_strict_same_position_ztrack_survival_preflight_v1"
    / "manifests/ztrack_manifest_minrun2.csv"
)
ORIG_MANIFEST = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_c2_effb0_v420_candidate_rd4ad_retest_v1"
    / "rd_c2_effb0_v420_candidate_manifest.csv"
)
CKPT_PATH = (
    PROJECT_ROOT
    / "outputs/models/rd_d1s_true_rd4ad_resnet18_medi3ch_shard_v1"
    / "checkpoints/best_train_loss.pth"
)
RESNET_WEIGHT = Path("/home/jinhy/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
CT_ROOT = Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1"
    "/volumes_npy"
)

# output paths
OUT_MANIFEST      = EXPERIMENT_ROOT / "manifests/strict_ztrack_scoring_candidate_manifest.csv"
OUT_SHARD_PLAN    = EXPERIMENT_ROOT / "manifests/strict_ztrack_scoring_shard_plan.csv"
OUT_PROBLEM_AUDIT = EXPERIMENT_ROOT / "manifests/strict_ztrack_problem_patient_preflight_audit.csv"
OUT_ERRORS_CSV    = EXPERIMENT_ROOT / "logs/errors.csv"
OUT_REPORT        = EXPERIMENT_ROOT / "reports/rd4ad_strict_ztrack_actual_scoring_preflight_report.md"
OUT_SUMMARY_JSON  = EXPERIMENT_ROOT / "reports/rd4ad_strict_ztrack_actual_scoring_preflight_summary.json"
OUT_DONE          = EXPERIMENT_ROOT / "DONE.json"

# ── constants ──────────────────────────────────────────────────────────────────

GUARDRAILS = {
    "stage2_holdout_accessed":                  False,
    "model_forward_executed":                   False,
    "checkpoint_loaded":                        False,
    "crop_generation_executed":                 False,
    "training_executed":                        False,
    "backward_executed":                        False,
    "optimizer_created":                        False,
    "checkpoint_saved":                         False,
    "full_scoring_executed":                    False,
    "threshold_recalculated":                   False,
    "existing_artifact_modified":               False,
    "existing_script_modified":                 False,
    "output_overwrite":                         False,
    "first_stage_score_used_for_candidate_deletion": False,
    "label_used_for_evaluation_only":           True,
    "label_used_for_scoring_selection":         False,
    "xy_radius_grouping_used":                  False,
    "representative_only_scoring_used":         False,
    "all_survived_track_candidates_preserved":  True,
    "primary_min_run_len":                      2,
    "t2_minrun3_used_as_primary":               False,
}

PROBLEM_PATIENT_IDS = {"LUNG1-086", "LUNG1-386", "LUNG1-399"}
SHARD_COUNT         = 8
CROP_SIZE           = 96

# expected values from survival preflight summary
EXPECTED_SURVIVED       = 92342
EXPECTED_POS_RETENTION  = 0.9797
EXPECTED_POS_SLICE_COV  = 0.9941
EXPECTED_POS_PAT_COV    = 1.0
EXPECTED_COMPLETE_MISS  = 0

# runtime reference: (n_forward, elapsed_sec) from group full scoring shards
RUNTIME_REF = [(4913, 84.2), (4037, 85.7), (4843, 90.1), (6423, 106.2)]

# ── global error accumulator ──────────────────────────────────────────────────

_errors: list = []

def _add_error(check_name: str, error_type: str, message: str) -> None:
    _errors.append({
        "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        "check_name": check_name,
        "error_type": error_type,
        "message":    message,
    })
    print(f"  [ERROR] {check_name}: {message}", file=sys.stderr)

# ── helpers ───────────────────────────────────────────────────────────────────

def _assert_path_safe(p: Path) -> None:
    s = str(p).lower()
    if "stage2_holdout" in s or ("stage2" in s and "holdout" in s):
        GUARDRAILS["stage2_holdout_accessed"] = True
        raise RuntimeError(f"[ABORT] stage2_holdout 경로 접근 차단: {p}")


def _patient_shard(patient_id: str, n_shards: int = SHARD_COUNT) -> int:
    return int(hashlib.md5(patient_id.encode()).hexdigest(), 16) % n_shards


def _sec_per_forward() -> float:
    rates = [elapsed / n for n, elapsed in RUNTIME_REF]
    return sum(rates) / len(rates)


def _read_csv_rows(path: Path) -> list:
    _assert_path_safe(path)
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ── dry-run ───────────────────────────────────────────────────────────────────

def run_dry_run() -> None:
    print("=" * 70)
    print("DRY-RUN: 파일 존재 확인 (파일 생성 없음, model forward 없음)")
    print("=" * 70)

    required_inputs = [
        ("survival_csv",    SURVIVAL_CSV),
        ("ztrack_manifest", ZTRACK_MANIFEST),
        ("orig_manifest",   ORIG_MANIFEST),
        ("ckpt_path",       CKPT_PATH),
        ("resnet_weight",   RESNET_WEIGHT),
    ]
    ok = True
    for label, p in required_inputs:
        exists = p.exists()
        mark   = "OK" if exists else "MISSING"
        print(f"  [{mark}] {label}: {p}")
        if not exists:
            ok = False

    # CT root
    ct_ok = CT_ROOT.exists()
    print(f"  [{'OK' if ct_ok else 'MISSING'}] ct_root: {CT_ROOT}")
    if not ct_ok:
        ok = False

    # output overwrite check
    existing_outputs = [p for p in [
        OUT_MANIFEST, OUT_SHARD_PLAN, OUT_PROBLEM_AUDIT,
        OUT_ERRORS_CSV, OUT_REPORT, OUT_SUMMARY_JSON, OUT_DONE,
    ] if p.exists()]
    if existing_outputs:
        print("\n  [WARN] 기존 output 파일이 있음 (overwrite 위험):")
        for p in existing_outputs:
            print(f"         {p}")
    else:
        print("\n  [OK] output overwrite 위험 없음")

    # stage2_holdout check
    stage2_paths = [
        PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/crops_stage2_holdout_6ch_dedicated_v1",
    ]
    accessed = any(str(p).lower() in str(s).lower() for p in stage2_paths
                   for s in [SURVIVAL_CSV, ZTRACK_MANIFEST, ORIG_MANIFEST])
    print(f"\n  [{'FAIL' if accessed else 'OK'}] stage2_holdout 접근 없음")

    # guardrails snapshot
    print("\n  guardrails (dry-run):")
    for k, v in GUARDRAILS.items():
        print(f"    {k}: {v}")

    status = "PASS" if ok else "FAIL (missing inputs)"
    print(f"\ndry-run 결과: {status}")
    sys.exit(0 if ok else 1)


# ── preflight helpers ─────────────────────────────────────────────────────────

def _load_survival_csv():
    """Return (survived_rows, all_rows)."""
    rows = _read_csv_rows(SURVIVAL_CSV)
    survived = [r for r in rows if r.get("survived", "").strip().lower() in ("true", "1")]
    return survived, rows


def _load_orig_manifest():
    """Return dict: candidate_id → row."""
    rows = _read_csv_rows(ORIG_MANIFEST)
    d = {}
    for row in rows:
        cid = row["candidate_id"]
        if cid in d:
            _add_error("orig_manifest_load", "DUPLICATE_CANDIDATE_ID",
                       f"candidate_id 중복: {cid}")
        d[cid] = row
    return d


def _load_ztrack_manifest():
    """Return dict: (patient_id, pos_y0, pos_x0, pos_y1, pos_x1) → list[(track_id, z_start, z_end)]."""
    rows = _read_csv_rows(ZTRACK_MANIFEST)
    d = defaultdict(list)
    for row in rows:
        key = (
            row["patient_id"],
            row["pos_y0"],
            row["pos_x0"],
            row["pos_y1"],
            row["pos_x1"],
        )
        d[key].append((
            row["track_id"],
            int(row["z_start"]),
            int(row["z_end"]),
        ))
    return dict(d)


def _assign_track_id(candidate: dict, ztrack_dict: dict) -> str:
    key = (
        candidate["patient_id"],
        candidate["pos_y0"],
        candidate["pos_x0"],
        candidate["pos_y1"],
        candidate["pos_x1"],
    )
    local_z = int(candidate["local_z"])
    candidates_tracks = ztrack_dict.get(key, [])
    for track_id, z_start, z_end in candidates_tracks:
        if z_start <= local_z <= z_end:
            return track_id
    return ""  # no track found (should not happen for survived candidates)


def _compute_metrics(survived_rows: list, orig_dict: dict):
    """Compute survival metrics for verification."""
    pos_candidates    = [r for r in survived_rows if r.get("label", "") == "positive"]
    hn_candidates     = [r for r in survived_rows if r.get("label", "") != "positive"]

    # positive slice coverage
    orig_pos_slices  = set()
    surv_pos_slices  = set()
    for cid, row in orig_dict.items():
        if row.get("label", "") == "positive" and row.get("stage_split", "") == "stage1_dev":
            orig_pos_slices.add((row["patient_id"], row["local_z"]))
    for r in pos_candidates:
        surv_pos_slices.add((r["patient_id"], r["local_z"]))

    # positive patient coverage
    orig_pos_patients = set(
        row["patient_id"] for row in orig_dict.values()
        if row.get("label", "") == "positive" and row.get("stage_split", "") == "stage1_dev"
    )
    surv_pos_patients = set(r["patient_id"] for r in pos_candidates)
    complete_miss     = orig_pos_patients - surv_pos_patients

    pos_retention   = len(pos_candidates) / 35247 if 35247 > 0 else 0.0
    slice_coverage  = len(surv_pos_slices) / len(orig_pos_slices) if orig_pos_slices else 0.0
    pat_coverage    = len(surv_pos_patients) / len(orig_pos_patients) if orig_pos_patients else 0.0

    return {
        "survived_count":           len(survived_rows),
        "positive_count":           len(pos_candidates),
        "hard_negative_count":      len(hn_candidates),
        "orig_positive_count":      35247,
        "pos_retention":            round(pos_retention, 4),
        "orig_positive_slices":     len(orig_pos_slices),
        "surv_positive_slices":     len(surv_pos_slices),
        "pos_slice_coverage":       round(slice_coverage, 4),
        "orig_positive_patients":   len(orig_pos_patients),
        "surv_positive_patients":   len(surv_pos_patients),
        "pos_patient_coverage":     round(pat_coverage, 4),
        "complete_miss_patients":   sorted(complete_miss),
        "complete_miss_count":      len(complete_miss),
    }


def _compute_problem_patient_audit(survived_rows: list):
    """Per-patient audit for PROBLEM_PATIENT_IDS."""
    by_patient = defaultdict(list)
    for r in survived_rows:
        by_patient[r["patient_id"]].append(r)

    audit_rows = []
    for pid in sorted(PROBLEM_PATIENT_IDS):
        rows    = by_patient.get(pid, [])
        pos     = [r for r in rows if r.get("label", "") == "positive"]
        slices  = set(r["local_z"] for r in pos)
        audit_rows.append({
            "patient_id":               pid,
            "survived_candidate_count": len(rows),
            "survived_positive_count":  len(pos),
            "survived_positive_slices": len(slices),
            "positive_slice_list":      sorted(slices),
            "has_positive":             len(pos) > 0,
        })
    return audit_rows


def _build_shard_plan(scoring_manifest: list, spf: float):
    """Build shard plan with patient_id stable hash."""
    shard_candidates     = defaultdict(list)
    shard_patients       = defaultdict(set)
    shard_pos_patients   = defaultdict(set)
    shard_problem_pats   = defaultdict(set)

    for row in scoring_manifest:
        sid = _patient_shard(row["patient_id"])
        shard_candidates[sid].append(row)
        shard_patients[sid].add(row["patient_id"])
        if row["label"] == "positive":
            shard_pos_patients[sid].add(row["patient_id"])
        if row["patient_id"] in PROBLEM_PATIENT_IDS:
            shard_problem_pats[sid].add(row["patient_id"])

    shard_rows = []
    for sid in range(SHARD_COUNT):
        cands = shard_candidates.get(sid, [])
        n_pos = sum(1 for r in cands if r["label"] == "positive")
        n_hn  = sum(1 for r in cands if r["label"] != "positive")
        n_cand = len(cands)
        est_sec = round(n_cand * spf, 1)
        shard_rows.append({
            "shard_id":                      sid,
            "candidate_count":               n_cand,
            "positive_candidate_count":      n_pos,
            "hard_negative_candidate_count": n_hn,
            "patient_count":                 len(shard_patients.get(sid, set())),
            "positive_patient_count":        len(shard_pos_patients.get(sid, set())),
            "problem_patients_in_shard":     ",".join(sorted(shard_problem_pats.get(sid, set()))) or "none",
            "estimated_runtime_sec":         est_sec,
            "estimated_runtime_min":         round(est_sec / 60, 1),
        })
    return shard_rows


def _check_ct_readiness(scoring_manifest: list):
    """Sample CT readiness: check if safe_id/ct_hu.npy exists for sample patients."""
    # collect unique safe_ids per patient
    safe_id_by_patient = {}
    for row in scoring_manifest:
        pid = row["patient_id"]
        if pid not in safe_id_by_patient:
            safe_id_by_patient[pid] = row["safe_id"]

    # sample: problem patients first, then up to 10 total
    sample_pids = list(PROBLEM_PATIENT_IDS & set(safe_id_by_patient.keys()))
    all_pids    = list(safe_id_by_patient.keys())
    import random as _rand
    _rand.seed(42)
    extra       = [p for p in _rand.sample(all_pids, min(len(all_pids), 30)) if p not in sample_pids]
    sample_pids = (sample_pids + extra)[:10]

    results = {}
    ct_root_exists = CT_ROOT.exists()
    for pid in sample_pids:
        safe_id  = safe_id_by_patient[pid]
        npy_path = CT_ROOT / safe_id / "ct_hu.npy"
        _assert_path_safe(npy_path)
        exists = npy_path.exists()
        results[pid] = {
            "safe_id":    safe_id,
            "npy_path":   str(npy_path),
            "exists":     exists,
        }

    n_ok      = sum(1 for v in results.values() if v["exists"])
    n_missing = len(results) - n_ok
    return {
        "ct_root_exists":   ct_root_exists,
        "ct_root":          str(CT_ROOT),
        "sample_count":     len(results),
        "sample_ok":        n_ok,
        "sample_missing":   n_missing,
        "details":          results,
        "unique_patients":  len(safe_id_by_patient),
        "readiness_status": "OK" if (ct_root_exists and n_missing == 0) else
                            "WARN_MISSING" if (ct_root_exists and n_missing > 0) else
                            "CT_ROOT_MISSING",
    }


def _check_crop_coords(scoring_manifest: list) -> dict:
    """Verify crop size is 96×96 for all candidates."""
    bad = []
    for row in scoring_manifest:
        h = int(row["crop_y1"]) - int(row["crop_y0"])
        w = int(row["crop_x1"]) - int(row["crop_x0"])
        if h != CROP_SIZE or w != CROP_SIZE:
            bad.append({
                "candidate_id": row["candidate_id"],
                "crop_h":       h,
                "crop_w":       w,
            })
    return {
        "total_checked":        len(scoring_manifest),
        "non_96x96_count":      len(bad),
        "non_96x96_samples":    bad[:5],
        "crop_coord_ok":        len(bad) == 0,
    }


def _check_checkpoint_readiness() -> dict:
    ckpt_exists  = CKPT_PATH.exists()
    rn18_exists  = RESNET_WEIGHT.exists()
    ckpt_size_mb = round(CKPT_PATH.stat().st_size / 1024**2, 1) if ckpt_exists else None
    rn18_size_mb = round(RESNET_WEIGHT.stat().st_size / 1024**2, 1) if rn18_exists else None
    return {
        "ckpt_path":    str(CKPT_PATH),
        "ckpt_exists":  ckpt_exists,
        "ckpt_size_mb": ckpt_size_mb,
        "resnet18_weight_path":   str(RESNET_WEIGHT),
        "resnet18_weight_exists": rn18_exists,
        "resnet18_size_mb":       rn18_size_mb,
        "readiness_status":       "OK" if (ckpt_exists and rn18_exists) else "MISSING",
    }


def _write_errors_csv() -> None:
    EXPERIMENT_ROOT.joinpath("logs").mkdir(parents=True, exist_ok=True)
    with open(OUT_ERRORS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "check_name", "error_type", "message"])
        writer.writeheader()
        writer.writerows(_errors)


# ── main preflight ────────────────────────────────────────────────────────────

def run_preflight() -> None:
    t0 = time.time()
    print("=" * 70)
    print("PREFLIGHT: RD4AD strict same-position z-track actual scoring v1")
    print("=" * 70)

    # ── 1. load survival CSV ─────────────────────────────────────────────────
    print("\n[1/9] survival CSV 로딩...")
    survived_rows, all_survival_rows = _load_survival_csv()
    print(f"      survived=True: {len(survived_rows):,}")
    print(f"      total in csv:  {len(all_survival_rows):,}")

    if len(survived_rows) != EXPECTED_SURVIVED:
        _add_error("survived_count", "COUNT_MISMATCH",
                   f"expected {EXPECTED_SURVIVED}, got {len(survived_rows)}")
    else:
        print(f"      [OK] survived count = {EXPECTED_SURVIVED:,} 일치")

    # duplicate candidate_id check in survival CSV
    survival_cids = [r["candidate_id"] for r in survived_rows]
    dup_cids = len(survival_cids) - len(set(survival_cids))
    if dup_cids > 0:
        _add_error("candidate_id_dedup", "DUPLICATE_CANDIDATE_ID",
                   f"survived CSV에 duplicate candidate_id: {dup_cids}개")
    else:
        print(f"      [OK] duplicate candidate_id = 0")

    # ── 2. load original manifest ────────────────────────────────────────────
    print("\n[2/9] original candidate manifest 로딩...")
    orig_dict = _load_orig_manifest()
    print(f"      original candidates: {len(orig_dict):,}")

    # ── 3. load ztrack manifest ──────────────────────────────────────────────
    print("\n[3/9] ztrack manifest 로딩...")
    ztrack_dict = _load_ztrack_manifest()
    print(f"      unique position keys: {len(ztrack_dict):,}")

    # ── 4. build scoring manifest ────────────────────────────────────────────
    print("\n[4/9] scoring manifest 생성...")

    scoring_manifest = []
    missing_in_orig  = []
    missing_track    = []

    orig_cols = [
        "safe_id", "stage_split", "crop_y0", "crop_x0", "crop_y1", "crop_x1",
        "first_stage_score", "source_score_csv",
        "slice_index", "source_branch", "backbone", "roi_source",
        "threshold_p95", "threshold_p99",
        "candidate_label", "candidate_rule", "sampling_reason",
    ]

    for r in survived_rows:
        cid = r["candidate_id"]
        if cid not in orig_dict:
            missing_in_orig.append(cid)
            _add_error("manifest_join", "MISSING_IN_ORIG",
                       f"candidate_id not in original manifest: {cid}")
            continue

        orig = orig_dict[cid]

        # stage2_holdout guard
        if orig.get("stage_split", "") not in ("stage1_dev", ""):
            _add_error("stage_split_check", "STAGE2_VIOLATION",
                       f"stage_split={orig['stage_split']} for {cid}")
            GUARDRAILS["stage2_holdout_accessed"] = True
            continue

        # assign track_id
        track_id = _assign_track_id(r, ztrack_dict)
        if not track_id:
            missing_track.append(cid)

        row_out = {
            "candidate_id":       cid,
            "patient_id":         r["patient_id"],
            "safe_id":            orig.get("safe_id", ""),
            "local_z":            r["local_z"],
            "crop_y0":            orig.get("crop_y0", ""),
            "crop_x0":            orig.get("crop_x0", ""),
            "crop_y1":            orig.get("crop_y1", ""),
            "crop_x1":            orig.get("crop_x1", ""),
            "label":              r.get("label", orig.get("label", "")),
            "track_id":           track_id,
            "pos_y0":             r.get("pos_y0", ""),
            "pos_x0":             r.get("pos_x0", ""),
            "pos_y1":             r.get("pos_y1", ""),
            "pos_x1":             r.get("pos_x1", ""),
            "ztrack_min_run_len": "2",
            "survived":           "True",
            "first_stage_score":  orig.get("first_stage_score", ""),
            "source_score_csv":   orig.get("source_score_csv", ""),
            "stage_split":        orig.get("stage_split", ""),
            "slice_index":        orig.get("slice_index", ""),
            "source_branch":      orig.get("source_branch", ""),
            "backbone":           orig.get("backbone", ""),
            "roi_source":         orig.get("roi_source", ""),
            "threshold_p95":      orig.get("threshold_p95", ""),
            "threshold_p99":      orig.get("threshold_p99", ""),
            "candidate_label":    orig.get("candidate_label", ""),
            "candidate_rule":     orig.get("candidate_rule", ""),
            "sampling_reason":    orig.get("sampling_reason", ""),
        }
        scoring_manifest.append(row_out)

    print(f"      scoring manifest rows: {len(scoring_manifest):,}")
    print(f"      missing in orig:       {len(missing_in_orig):,}")
    print(f"      missing track_id:      {len(missing_track):,}")

    if missing_in_orig:
        _add_error("manifest_join", "JOIN_INCOMPLETE",
                   f"원본 manifest join 실패: {len(missing_in_orig)}개")
    else:
        print(f"      [OK] 원본 manifest join 100%")

    if missing_track:
        _add_error("track_id_assignment", "MISSING_TRACK",
                   f"track_id 미할당 candidate: {len(missing_track)}개 "
                   f"(샘플: {missing_track[:5]})")

    # ── 5. survival metrics 재현 검증 ────────────────────────────────────────
    print("\n[5/9] survival metrics 재현 검증...")
    metrics = _compute_metrics(scoring_manifest, orig_dict)

    checks = [
        ("survived_count",       metrics["survived_count"],       EXPECTED_SURVIVED,
         lambda a, b: a == b),
        ("pos_retention",        metrics["pos_retention"],         EXPECTED_POS_RETENTION,
         lambda a, b: abs(a - b) < 0.002),
        ("pos_slice_coverage",   metrics["pos_slice_coverage"],    EXPECTED_POS_SLICE_COV,
         lambda a, b: abs(a - b) < 0.002),
        ("pos_patient_coverage", metrics["pos_patient_coverage"],  EXPECTED_POS_PAT_COV,
         lambda a, b: abs(a - b) < 0.001),
        ("complete_miss_count",  metrics["complete_miss_count"],   EXPECTED_COMPLETE_MISS,
         lambda a, b: a == b),
    ]
    for name, got, expected, check_fn in checks:
        ok = check_fn(got, expected)
        mark = "OK" if ok else "FAIL"
        print(f"      [{mark}] {name}: {got} (expected {expected})")
        if not ok:
            _add_error("metrics_verify", "MISMATCH",
                       f"{name}: got={got}, expected={expected}")

    if metrics["complete_miss_patients"]:
        _add_error("metrics_verify", "COMPLETE_MISS",
                   f"complete miss patients: {metrics['complete_miss_patients']}")

    # ── 6. problem patient audit ─────────────────────────────────────────────
    print("\n[6/9] 문제 환자 3명 audit...")
    problem_audit = _compute_problem_patient_audit(scoring_manifest)
    for pa in problem_audit:
        ok    = pa["has_positive"]
        mark  = "OK" if ok else "FAIL"
        print(f"      [{mark}] {pa['patient_id']}: "
              f"pos_cand={pa['survived_positive_count']}, "
              f"pos_slice={pa['survived_positive_slices']}")
        if not ok:
            _add_error("problem_patient_audit", "COMPLETE_MISS_PROBLEM",
                       f"{pa['patient_id']} survived positive candidate = 0")

    # ── 7. crop 좌표 검증 (sample only, no crop generation) ─────────────────
    print("\n[7/9] crop 좌표 검증 (96×96 확인, crop 생성 없음)...")
    crop_check = _check_crop_coords(scoring_manifest)
    mark = "OK" if crop_check["crop_coord_ok"] else "WARN"
    print(f"      [{mark}] 96×96 아닌 crop: {crop_check['non_96x96_count']:,} / {crop_check['total_checked']:,}")

    # ── 8. CT readiness ──────────────────────────────────────────────────────
    print("\n[8/9] CT readiness 샘플 확인...")
    ct_readiness = _check_ct_readiness(scoring_manifest)
    mark = ct_readiness["readiness_status"]
    print(f"      ct_root_exists:  {ct_readiness['ct_root_exists']}")
    print(f"      sample checked:  {ct_readiness['sample_count']}")
    print(f"      sample OK:       {ct_readiness['sample_ok']}")
    print(f"      sample MISSING:  {ct_readiness['sample_missing']}")
    print(f"      status:          {mark}")
    for pid, detail in ct_readiness["details"].items():
        sym = "OK" if detail["exists"] else "MISSING"
        print(f"        [{sym}] {pid}: {detail['npy_path']}")
    if ct_readiness["sample_missing"] > 0:
        _add_error("ct_readiness", "CT_MISSING",
                   f"CT npy missing: {ct_readiness['sample_missing']}/{ct_readiness['sample_count']}")

    # ── 9. checkpoint readiness ──────────────────────────────────────────────
    ckpt_info = _check_checkpoint_readiness()
    print(f"\n      checkpoint:  [{ckpt_info['readiness_status']}]")
    print(f"        {CKPT_PATH} ({ckpt_info['ckpt_size_mb']} MB)")
    print(f"        resnet18:    {ckpt_info['resnet18_weight_exists']} ({ckpt_info['resnet18_size_mb']} MB)")
    if ckpt_info["readiness_status"] != "OK":
        _add_error("checkpoint_readiness", "MISSING", "checkpoint or resnet18 weight 없음")

    # ── runtime estimate ─────────────────────────────────────────────────────
    spf          = _sec_per_forward()
    total_est    = len(scoring_manifest) * spf
    shard_est    = total_est / SHARD_COUNT
    print(f"\n      runtime estimate:")
    print(f"        sec_per_forward:  {spf:.5f}")
    print(f"        total forward:    {len(scoring_manifest):,}")
    print(f"        total est:        {total_est:.1f} sec ({total_est/60:.1f} min)")
    print(f"        per shard ({SHARD_COUNT}):   {shard_est:.1f} sec ({shard_est/60:.1f} min)")

    # ── shard plan ───────────────────────────────────────────────────────────
    print(f"\n[/] shard plan 생성 (patient_id stable hash, {SHARD_COUNT} shards)...")
    shard_plan = _build_shard_plan(scoring_manifest, spf)
    for s in shard_plan:
        pp = s["problem_patients_in_shard"]
        print(f"      shard {s['shard_id']}: "
              f"cand={s['candidate_count']:,} pos={s['positive_candidate_count']:,} "
              f"pat={s['patient_count']} est={s['estimated_runtime_min']}min "
              f"problem=[{pp}]")

    shard_imbalance = max(s["candidate_count"] for s in shard_plan) / \
                      (min(s["candidate_count"] for s in shard_plan) + 1)
    print(f"      shard imbalance (max/min): {shard_imbalance:.2f}")

    # ── output overwrite check ───────────────────────────────────────────────
    overwrite_candidates = [p for p in [
        OUT_MANIFEST, OUT_SHARD_PLAN, OUT_PROBLEM_AUDIT,
        OUT_ERRORS_CSV, OUT_REPORT, OUT_SUMMARY_JSON, OUT_DONE,
    ] if p.exists()]
    if overwrite_candidates:
        GUARDRAILS["output_overwrite"] = True
        print(f"\n  [WARN] output overwrite 대상: {len(overwrite_candidates)}개 파일")
        for p in overwrite_candidates:
            print(f"         {p}")

    # ── write outputs ────────────────────────────────────────────────────────
    print("\n[writing outputs]")
    EXPERIMENT_ROOT.joinpath("manifests").mkdir(parents=True, exist_ok=True)
    EXPERIMENT_ROOT.joinpath("reports").mkdir(parents=True, exist_ok=True)
    EXPERIMENT_ROOT.joinpath("logs").mkdir(parents=True, exist_ok=True)

    # manifest CSV
    manifest_cols = [
        "candidate_id", "patient_id", "safe_id", "local_z",
        "crop_y0", "crop_x0", "crop_y1", "crop_x1",
        "label", "track_id",
        "pos_y0", "pos_x0", "pos_y1", "pos_x1",
        "ztrack_min_run_len", "survived",
        "first_stage_score", "source_score_csv", "stage_split",
        "slice_index", "source_branch", "backbone", "roi_source",
        "threshold_p95", "threshold_p99",
        "candidate_label", "candidate_rule", "sampling_reason",
    ]
    with open(OUT_MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scoring_manifest)
    print(f"  written: {OUT_MANIFEST}")

    # shard plan CSV
    shard_cols = [
        "shard_id", "candidate_count", "positive_candidate_count",
        "hard_negative_candidate_count", "patient_count", "positive_patient_count",
        "problem_patients_in_shard",
        "estimated_runtime_sec", "estimated_runtime_min",
    ]
    with open(OUT_SHARD_PLAN, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=shard_cols)
        writer.writeheader()
        writer.writerows(shard_plan)
    print(f"  written: {OUT_SHARD_PLAN}")

    # problem patient audit CSV
    problem_cols = [
        "patient_id", "survived_candidate_count", "survived_positive_count",
        "survived_positive_slices", "positive_slice_list", "has_positive",
    ]
    with open(OUT_PROBLEM_AUDIT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=problem_cols)
        writer.writeheader()
        for pa in problem_audit:
            row = dict(pa)
            row["positive_slice_list"] = "|".join(str(z) for z in row["positive_slice_list"])
            writer.writerow(row)
    print(f"  written: {OUT_PROBLEM_AUDIT}")

    # errors CSV
    _write_errors_csv()
    print(f"  written: {OUT_ERRORS_CSV}  ({len(_errors)} errors)")

    # ── determine verdict ────────────────────────────────────────────────────
    critical_errors = [e for e in _errors if e["error_type"] not in ("WARN",)]
    passed_all = (
        metrics["survived_count"]       == EXPECTED_SURVIVED
        and metrics["pos_patient_coverage"] >= 0.999
        and metrics["complete_miss_count"]  == 0
        and all(pa["has_positive"] for pa in problem_audit)
        and len(missing_in_orig)            == 0
        and ckpt_info["readiness_status"]   == "OK"
        and not GUARDRAILS["stage2_holdout_accessed"]
        and not GUARDRAILS["existing_artifact_modified"]
    )
    partial = (
        not passed_all
        and metrics["survived_count"]       == EXPECTED_SURVIVED
        and metrics["pos_patient_coverage"] >= 0.999
        and metrics["complete_miss_count"]  == 0
    )
    verdict = "PASS" if passed_all else ("PARTIAL_PASS" if partial else "FAIL")

    # ── summary JSON ─────────────────────────────────────────────────────────
    elapsed = round(time.time() - t0, 1)
    summary = {
        "verdict":                    verdict,
        "critical_error":             not passed_all and not partial,
        "elapsed_sec":                elapsed,
        "survived_candidate_count":   metrics["survived_count"],
        "candidate_reduction_rate":   round(1 - metrics["survived_count"] / 113447, 4),
        "positive_candidate_count":   metrics["positive_count"],
        "hard_negative_count":        metrics["hard_negative_count"],
        "positive_candidate_retention": metrics["pos_retention"],
        "positive_slice_coverage":    metrics["pos_slice_coverage"],
        "positive_patient_coverage":  metrics["pos_patient_coverage"],
        "complete_miss_patients":     metrics["complete_miss_patients"],
        "complete_miss_count":        metrics["complete_miss_count"],
        "problem_patient_audit":      {pa["patient_id"]: pa for pa in problem_audit},
        "manifest_join_success":      len(missing_in_orig) == 0,
        "missing_in_orig_count":      len(missing_in_orig),
        "missing_track_id_count":     len(missing_track),
        "duplicate_candidate_id":     dup_cids,
        "crop_coord_check":           crop_check,
        "ct_readiness":               ct_readiness,
        "checkpoint_readiness":       ckpt_info,
        "shard_plan": {
            "shard_count":          SHARD_COUNT,
            "shard_key":            "patient_id_stable_hash",
            "total_candidates":     sum(s["candidate_count"] for s in shard_plan),
            "imbalance_ratio":      round(shard_imbalance, 2),
            "shards":               shard_plan,
        },
        "runtime_estimate": {
            "sec_per_forward":      round(spf, 6),
            "total_forward":        len(scoring_manifest),
            "total_estimated_sec":  round(total_est, 1),
            "total_estimated_min":  round(total_est / 60, 1),
            "per_shard_estimated_sec": round(shard_est, 1),
            "per_shard_estimated_min": round(shard_est / 60, 1),
        },
        "error_count":                len(_errors),
        "errors":                     _errors,
        "guardrails":                 GUARDRAILS,
        "note": (
            "runtime estimate = sec_per_forward 평균(4-shard 실측 비례) × 92,342 candidate. "
            "actual scoring에서는 T1_minrun2 survived candidate 전부 forward. "
            "track 대표 1개 선택 금지."
        ),
    }

    with open(OUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  written: {OUT_SUMMARY_JSON}")

    # ── report markdown ──────────────────────────────────────────────────────
    _write_report(summary, shard_plan, problem_audit, ct_readiness, ckpt_info, metrics, spf, elapsed)
    print(f"  written: {OUT_REPORT}")

    # ── DONE.json ────────────────────────────────────────────────────────────
    if verdict in ("PASS", "PARTIAL_PASS"):
        done = {
            "verdict":                  verdict,
            "timestamp":                time.strftime("%Y-%m-%dT%H:%M:%S"),
            "survived_candidate_count": metrics["survived_count"],
            "shard_count":              SHARD_COUNT,
            "shard_key":                "patient_id_stable_hash",
            "scoring_manifest":         str(OUT_MANIFEST),
            "shard_plan":               str(OUT_SHARD_PLAN),
            "next_step":                "actual strict z-track RD4AD scoring run/merge script 작성",
            "guardrails":               GUARDRAILS,
        }
        with open(OUT_DONE, "w", encoding="utf-8") as f:
            json.dump(done, f, indent=2, ensure_ascii=False)
        print(f"  written: {OUT_DONE}")

    # ── final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"판정: {verdict}")
    print("=" * 70)
    print(f"  survived candidate count:   {metrics['survived_count']:,}")
    print(f"  reduction vs 113,447:       {summary['candidate_reduction_rate']:.1%}")
    print(f"  positive candidate count:   {metrics['positive_count']:,}")
    print(f"  positive candidate retention: {metrics['pos_retention']:.4f}")
    print(f"  positive slice coverage:    {metrics['pos_slice_coverage']:.4f}")
    print(f"  positive patient coverage:  {metrics['pos_patient_coverage']:.4f}")
    print(f"  complete miss patient:      {metrics['complete_miss_count']}")
    for pa in problem_audit:
        print(f"  {pa['patient_id']}:  pos_cand={pa['survived_positive_count']} "
              f"pos_slice={pa['survived_positive_slices']}")
    print(f"  shard count:                {SHARD_COUNT}")
    print(f"  estimated total runtime:    {round(total_est/60, 1)} min")
    print(f"  CT readiness:               {ct_readiness['readiness_status']}")
    print(f"  checkpoint readiness:       {ckpt_info['readiness_status']}")
    print(f"  output overwrite:           {GUARDRAILS['output_overwrite']}")
    print(f"  stage2_holdout accessed:    {GUARDRAILS['stage2_holdout_accessed']}")
    print(f"  model forward executed:     {GUARDRAILS['model_forward_executed']}")
    print(f"  errors:                     {len(_errors)}")
    print(f"  elapsed:                    {elapsed}s")
    print()
    if verdict == "PASS":
        print("다음 단계: actual strict z-track RD4AD scoring run/merge script 작성")
    elif verdict == "PARTIAL_PASS":
        print("PARTIAL_PASS: CT readiness 또는 shard balance 경고 확인 후 진행")
    else:
        print("FAIL: 오류 수정 후 재실행")

    sys.exit(0 if verdict in ("PASS", "PARTIAL_PASS") else 1)


def _write_report(summary, shard_plan, problem_audit, ct_readiness,
                  ckpt_info, metrics, spf, elapsed):
    lines = []
    lines.append("# RD4AD strict same-position z-track actual scoring preflight v1 report")
    lines.append("")
    lines.append(f"**판정: {summary['verdict']}**")
    lines.append("")
    lines.append(f"- elapsed: {elapsed}s")
    lines.append(f"- position_source: crop_coords")
    lines.append(f"- primary_min_run_len: 2")
    lines.append("")
    lines.append("## 핵심 survival 지표 재현")
    lines.append("")
    lines.append("| 항목 | 값 | 기준 | 판정 |")
    lines.append("|---|---|---|---|")
    checks_tbl = [
        ("survived_count",         metrics["survived_count"],         92342,  "=="),
        ("positive_retention",     metrics["pos_retention"],          0.9797, "~="),
        ("positive_slice_coverage",metrics["pos_slice_coverage"],     0.9941, "~="),
        ("positive_patient_coverage", metrics["pos_patient_coverage"],1.0,    "=="),
        ("complete_miss_count",    metrics["complete_miss_count"],    0,      "=="),
    ]
    for name, got, exp, op in checks_tbl:
        if op == "==":
            ok = got == exp
        else:
            ok = abs(got - exp) < 0.003
        verdict_cell = "PASS" if ok else "FAIL"
        lines.append(f"| {name} | {got} | {exp} | {verdict_cell} |")
    lines.append("")
    lines.append("## 문제 환자 3명")
    lines.append("")
    lines.append("| patient_id | survived_pos_cand | survived_pos_slice | has_positive |")
    lines.append("|---|---|---|---|")
    for pa in problem_audit:
        lines.append(f"| {pa['patient_id']} | {pa['survived_positive_count']} | "
                     f"{pa['survived_positive_slices']} | {pa['has_positive']} |")
    lines.append("")
    lines.append("## Shard plan (patient_id stable hash, 8 shards)")
    lines.append("")
    lines.append("| shard_id | candidates | pos_cand | hn_cand | patients | pos_patients | problem_pats | est_min |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in shard_plan:
        lines.append(
            f"| {s['shard_id']} | {s['candidate_count']:,} | {s['positive_candidate_count']:,} | "
            f"{s['hard_negative_candidate_count']:,} | {s['patient_count']} | "
            f"{s['positive_patient_count']} | {s['problem_patients_in_shard']} | "
            f"{s['estimated_runtime_min']} |"
        )
    lines.append("")
    lines.append("## Runtime estimate")
    lines.append("")
    lines.append(f"- sec_per_forward: {spf:.5f}")
    lines.append(f"- 기준 shards: {RUNTIME_REF}")
    lines.append(f"- total forward: {metrics['survived_count']:,}")
    lines.append(f"- total estimated: {round(metrics['survived_count'] * spf / 60, 1)} min")
    lines.append(f"- per shard ({SHARD_COUNT}): {round(metrics['survived_count'] * spf / SHARD_COUNT / 60, 1)} min")
    lines.append("")
    lines.append("## CT readiness")
    lines.append("")
    lines.append(f"- ct_root_exists: {ct_readiness['ct_root_exists']}")
    lines.append(f"- ct_root: {ct_readiness['ct_root']}")
    lines.append(f"- status: {ct_readiness['readiness_status']}")
    lines.append(f"- sample: {ct_readiness['sample_ok']}/{ct_readiness['sample_count']} OK")
    for pid, detail in ct_readiness["details"].items():
        sym = "OK" if detail["exists"] else "MISSING"
        lines.append(f"  - [{sym}] {pid}: {detail['npy_path']}")
    lines.append("")
    lines.append("## Checkpoint readiness")
    lines.append("")
    lines.append(f"- status: {ckpt_info['readiness_status']}")
    lines.append(f"- ckpt: {ckpt_info['ckpt_path']} ({ckpt_info['ckpt_size_mb']} MB)")
    lines.append(f"- resnet18: {ckpt_info['resnet18_weight_path']} ({ckpt_info['resnet18_size_mb']} MB)")
    lines.append("")
    lines.append("## Guardrails")
    lines.append("")
    for k, v in summary["guardrails"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Errors")
    lines.append("")
    if summary["error_count"] == 0:
        lines.append("- (없음)")
    else:
        for e in summary["errors"]:
            lines.append(f"- [{e['error_type']}] {e['check_name']}: {e['message']}")
    lines.append("")
    if summary["verdict"] in ("PASS", "PARTIAL_PASS"):
        lines.append("## 다음 단계")
        lines.append("")
        lines.append("actual strict z-track RD4AD scoring run/merge script 작성.")
        lines.append(f"T1_minrun2 survived candidate {metrics['survived_count']:,}개를 "
                     f"{SHARD_COUNT}개 shard로 나눠 RD4AD forward 수행.")
        lines.append("track 대표 1개 선택 금지. 모든 survived candidate forward.")

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RD4AD strict same-position z-track actual scoring preflight v1"
    )
    parser.add_argument("--dry-run",               action="store_true",
                        help="입력 파일 존재만 확인. 파일 생성 없음.")
    parser.add_argument("--run-preflight",          action="store_true",
                        help="실제 preflight 실행.")
    parser.add_argument("--confirm-readonly",       action="store_true",
                        help="read-only 확인 동의.")
    parser.add_argument("--confirm-stage1dev-only", action="store_true",
                        help="stage1_dev only 동의.")
    args = parser.parse_args()

    if not args.dry_run and not args.run_preflight:
        print(
            "[ERROR] bare run blocked.\n"
            "사용법:\n"
            "  python script.py --dry-run\n"
            "  python script.py --run-preflight --confirm-readonly --confirm-stage1dev-only",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.dry_run:
        run_dry_run()

    elif args.run_preflight:
        if not args.confirm_readonly:
            print("[ERROR] --confirm-readonly 필수", file=sys.stderr)
            sys.exit(2)
        if not args.confirm_stage1dev_only:
            print("[ERROR] --confirm-stage1dev-only 필수", file=sys.stderr)
            sys.exit(2)
        run_preflight()


if __name__ == "__main__":
    main()
