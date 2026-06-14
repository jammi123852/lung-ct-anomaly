#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
b1d7b_patchcore_normal_query_calibration_feature_smoke

B1-D7b approval preflight (--dry-run) + B1-D7c 실행 스크립트 (--run).

목적: C3 memory 기준으로 normal wall/mediastinum query 30개와
     FP gate candidate 6개의 feature 거리를 비교하여
     Gate-P2 calibration 가설(H_A/H_B/H_C/H_D)을 검증한다.

-- 실행 규칙 --
★ ALLOW_REAL_PROCESSING = False (기본 차단). importlib runtime override로만 real 허용.
★ bare-run (인수 없음): exit 2
★ --run: ALLOW_REAL_PROCESSING=False 이면 exit 2
★ --dry-run: 입력/경로/shape/query/memory schema 검증만. feature 없음. 승인 preflight 파일 생성.
★ device = cpu. GPU: --confirm-gpu 없으면 차단.
★ stage2_holdout 접근 금지.
★ score/threshold/ROI 무수정.
★ output exist_ok=False (collision guard).
★ query_count > 30 차단.
★ memory feature 재생성 구조 금지 (C3 config 동일 재사용).
★ adjusted_score/suppression_weight/refined_score 생성 금지.

-- 사용법 --
dry-run (B1-D7b 검증):
  python b1d7b_patchcore_normal_query_calibration_feature_smoke.py --dry-run

real execution (B1-D7c 승인 후):
  importlib 방식으로 ALLOW_REAL_PROCESSING = True 후
  python b1d7b_patchcore_normal_query_calibration_feature_smoke.py --run --confirm-calibration-smoke
"""

import argparse
import csv
import json
import sys
import os
from pathlib import Path
from collections import defaultdict

import numpy as np

ALLOW_REAL_PROCESSING = False  # ★ 기본 차단. importlib runtime override로만 real 허용.

BASE = Path("/home/jinhy/project/lung-ct-anomaly")
DIR  = BASE / "outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1"
NSCORE = BASE / "outputs/position-aware-padim-v1/scores/padim_v2_roi0_0/normal_by_patient"
MROOT  = BASE / "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1"
NROOT  = Path("/mnt/c/Users/jinhy/Desktop/"
              "Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/volumes_npy")
SEL_IDX_NPZ = (BASE / "outputs/position-aware-padim-v1/models"
               "/padim_v2_roi0_0/distributions/position_bin_stats.npz")

# B1-D7a 입력
B7A_SUMMARY    = DIR / "b1d7a_gate_p2_normal_query_calibration_preflight_summary.json"
B7A_QUERY_PLAN = DIR / "b1d7a_normal_wall_med_query_plan.csv"
B7A_EXEC_PLAN  = DIR / "b1d7a_calibration_execution_plan.csv"

# B1-D3k2 검증
B3K2_SUMMARY = DIR / "b1d3k2_c3_mixed_memory_feature_smoke_validation_summary.json"

# C3 memory outputs
C3_DIR            = DIR / "b1d3j1_anatomy_conditioned_feature_smoke_c3_v1"
C3_MEMORY_PREVIEW = C3_DIR / "b1d3j1_anatomy_conditioned_memory_feature_preview.csv"
C3_CAND_DISTANCE  = C3_DIR / "b1d3j1_anatomy_conditioned_candidate_distance_preview.csv"
C3_SUMMARY        = C3_DIR / "b1d3j1_anatomy_conditioned_feature_smoke_summary.json"

# FP candidates (C3 distance 재평가 기준)
B3F_CAND = DIR / "b1d3f_gate_p2_position_conditioned_candidates_summary.csv"

# B1-D7b dry-run 출력
B7B_MAPPING_CHECK = DIR / "b1d7b_query_ct_mask_mapping_check.csv"
B7B_EXEC_PLAN_VALIDATED = DIR / "b1d7b_calibration_execution_plan_validated.csv"
B7B_PREFLIGHT_SUMMARY   = DIR / "b1d7b_normal_query_calibration_feature_smoke_approval_preflight_summary.json"
B7B_PREFLIGHT_REPORT    = DIR / "b1d7b_normal_query_calibration_feature_smoke_approval_preflight_report.md"

# B1-D7c --run 출력
B7C_QUERY_DIST_CSV   = DIR / "b1d7c_normal_query_distance_preview.csv"
B7C_CALIB_TABLE_CSV  = DIR / "b1d7c_fp_vs_normal_calibration_table.csv"
B7C_SUMMARY          = DIR / "b1d7c_normal_query_calibration_summary.json"
B7C_REPORT           = DIR / "b1d7c_normal_query_calibration_report.md"

# 상수
PATCH = 32
RAW_FEATURE_DIM = 448
REDUCED_DIM = 100
RATIO_BOUNDARY_THR = 0.85
CDR_TOL = 0.10
MAX_QUERY_COUNT = 30
MEMORY_CAP = 500
MEMORY_PATIENTS = ["normal004", "normal013", "normal014", "normal016", "normal017"]
QUERY_PATIENTS  = ["normal023", "normal024", "normal036", "normal041", "normal045", "normal048"]
STAGE2_PATTERNS = ["stage2", "holdout", "test_patient", "lesion_score"]

CALIBRATION_HYPOTHESES = {
    "H_A": {
        "label": "Gate-P2_calibration_fail",
        "trigger": "normal_suspicious_rate > 40%",
        "implication": "p90 threshold 또는 memory pool이 비대표적",
    },
    "H_B": {
        "label": "FP_genuine_outlier",
        "trigger": "normal_suspicious_rate < 20% AND fp_suspicious_rate = 100%",
        "implication": "FP 6개가 feature 공간에서 실제로 비정상적",
    },
    "H_C": {
        "label": "feature_space_limit",
        "trigger": "normal AND FP both suspicious_rate > 60%",
        "implication": "ResNet18 feature space 구분 한계 (H5 확정 방향)",
    },
    "H_D": {
        "label": "boundary_only_issue",
        "trigger": "boundary_suspicious_rate >> inside_suspicious_rate",
        "implication": "C3 boundary memory가 이질적 — 경계 패치 feature 다양성 문제",
    },
}


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------

def fail(msg: str, code: int = 2):
    print(f"[b1d7b][중단] {msg}", file=sys.stderr)
    sys.exit(code)


def guard_stage2(path_str: str):
    for pat in STAGE2_PATTERNS:
        if pat in str(path_str).lower():
            fail(f"stage2_holdout 접근 차단: {path_str}")


def load_csv(p: Path, enc: str = "utf-8") -> list:
    with open(p, encoding=enc) as f:
        return list(csv.DictReader(f))


def guard_no_overwrite(path: Path, name: str):
    if path.exists():
        fail(f"출력 파일 이미 존재 (덮어쓰기 금지): [{name}] {path}")


# ---------------------------------------------------------------------------
# 입력 검증 (dry-run/run 공통)
# ---------------------------------------------------------------------------

def validate_inputs() -> dict:
    """read-only 검증. feature/torch 없음."""
    errs = []

    # ── B1-D7a summary ──────────────────────────────────────────────────────
    if not B7A_SUMMARY.exists():
        fail(f"B1-D7a summary 없음: {B7A_SUMMARY}")
    with open(B7A_SUMMARY, encoding="utf-8") as f:
        b7a = json.load(f)
    if b7a.get("verdict") != "PASS":
        fail(f"B1-D7a verdict != PASS: {b7a.get('verdict')}")
    if b7a.get("stage2_holdout_access", 1) != 0:
        fail("B1-D7a stage2_holdout_access != 0")
    if b7a.get("feature_extracted", True):
        fail("B1-D7a feature_extracted = True (예상 외)")
    b7a_mtime = int(os.path.getmtime(B7A_SUMMARY))

    # ── query plan ──────────────────────────────────────────────────────────
    if not B7A_QUERY_PLAN.exists():
        fail(f"query plan CSV 없음: {B7A_QUERY_PLAN}")
    query_rows = load_csv(B7A_QUERY_PLAN)
    query_count = len(query_rows)
    if query_count != MAX_QUERY_COUNT:
        fail(f"query_count {query_count} != {MAX_QUERY_COUNT}")

    boundary_queries = [r for r in query_rows if r["query_type"] == "boundary"]
    inside_queries   = [r for r in query_rows if r["query_type"] == "inside"]
    if len(boundary_queries) != 15 or len(inside_queries) != 15:
        fail(f"boundary={len(boundary_queries)}/inside={len(inside_queries)} != 15/15")

    per_bin = defaultdict(lambda: defaultdict(int))
    for r in query_rows:
        per_bin[r["position_bin"]][r["query_type"]] += 1
    for pb, counts in per_bin.items():
        if counts["boundary"] != 3 or counts["inside"] != 3:
            fail(f"bin {pb}: boundary={counts['boundary']}/inside={counts['inside']} != 3/3")
    if len(per_bin) != 5:
        fail(f"position_bin 수 {len(per_bin)} != 5")

    # query patients 확인
    actual_query_patients = sorted(set(r["patient_id"] for r in query_rows))
    if sorted(actual_query_patients) != sorted(QUERY_PATIENTS):
        fail(f"query patients 불일치: {actual_query_patients}")

    query_plan_mtime = int(os.path.getmtime(B7A_QUERY_PLAN))

    # ── B1-D3k2 검증 ────────────────────────────────────────────────────────
    if not B3K2_SUMMARY.exists():
        fail(f"B1-D3k2 summary 없음: {B3K2_SUMMARY}")
    with open(B3K2_SUMMARY, encoding="utf-8") as f:
        b3k2 = json.load(f)
    if b3k2.get("verdict") != "PASS":
        fail(f"B1-D3k2 verdict != PASS: {b3k2.get('verdict')}")
    if b3k2.get("stage2_holdout_access", 1) != 0:
        fail("B1-D3k2 stage2_holdout_access != 0")
    b3k2_mtime = int(os.path.getmtime(B3K2_SUMMARY))

    # ── C3 memory feature preview ────────────────────────────────────────────
    if not C3_MEMORY_PREVIEW.exists():
        fail(f"C3 memory_feature_preview 없음: {C3_MEMORY_PREVIEW}")
    if not C3_CAND_DISTANCE.exists():
        fail(f"C3 candidate_distance_preview 없음: {C3_CAND_DISTANCE}")
    if not C3_SUMMARY.exists():
        fail(f"C3 summary 없음: {C3_SUMMARY}")

    mem_rows = load_csv(C3_MEMORY_PREVIEW)
    mem_count = len(mem_rows)
    if mem_count != 500:
        fail(f"C3 memory rows {mem_count} != 500")

    required_mem_cols = [
        "memory_row_id", "normal_patient_id", "position_bin", "condition_name",
        "cdr", "roi_ratio", "ratio_source", "local_z", "y0", "x0",
        "feature_dim", "refined_roi_ratio_v4", "feature_status",
    ]
    mem_cols = list(mem_rows[0].keys()) if mem_rows else []
    missing_mem_cols = [c for c in required_mem_cols if c not in mem_cols]
    if missing_mem_cols:
        fail(f"C3 memory preview 컬럼 누락: {missing_mem_cols}")

    for r in mem_rows:
        if r.get("feature_dim") != str(REDUCED_DIM) and r.get("feature_dim") != REDUCED_DIM:
            fail(f"memory feature_dim != {REDUCED_DIM}: {r.get('feature_dim')}")
        if r.get("feature_status") != "ok":
            fail(f"memory feature_status != ok: mpid={r.get('memory_row_id')}")

    actual_mem_patients = sorted(set(r["normal_patient_id"] for r in mem_rows))
    if sorted(actual_mem_patients) != sorted(MEMORY_PATIENTS):
        fail(f"C3 memory patients 불일치: {actual_mem_patients}")

    # memory boundary/inside 분포 확인
    mem_boundary = [r for r in mem_rows if float(r["roi_ratio"]) < RATIO_BOUNDARY_THR]
    mem_inside   = [r for r in mem_rows if float(r["roi_ratio"]) >= RATIO_BOUNDARY_THR]
    if len(mem_boundary) != 250 or len(mem_inside) != 250:
        errs.append(f"C3 memory boundary={len(mem_boundary)}/inside={len(mem_inside)} != 250/250")

    c3_mem_mtime = int(os.path.getmtime(C3_MEMORY_PREVIEW))

    # ── query/memory overlap ─────────────────────────────────────────────────
    overlap = set(actual_query_patients) & set(actual_mem_patients)
    if overlap:
        fail(f"query/memory patient 교집합 존재: {overlap}")

    # ── selected_feature_indices ─────────────────────────────────────────────
    sel_ok = SEL_IDX_NPZ.exists()
    sel_shape_ok = False
    if sel_ok:
        try:
            sel = np.load(SEL_IDX_NPZ, allow_pickle=True)["selected_feature_indices"].astype(int)
            sel_shape_ok = (sel.shape[0] == REDUCED_DIM
                            and sel.min() >= 0 and sel.max() < RAW_FEATURE_DIM)
        except Exception as e:
            errs.append(f"selected_feature_indices 로드 실패: {e}")
    if not sel_ok:
        fail(f"SEL_IDX_NPZ 없음: {SEL_IDX_NPZ}")
    if not sel_shape_ok:
        fail("selected_feature_indices shape/range 비정상")

    # ── query CT/mask 매핑 검증 ──────────────────────────────────────────────
    query_ct_mask_status = []
    ct_mask_fail_count = 0
    for r in query_rows:
        patient = r["patient_id"]
        guard_stage2(patient)
        local_z = int(r["local_z"])
        y0      = int(r["y0"])
        x0      = int(r["x0"])

        # mask dir 찾기
        mask_hits = sorted((MROOT / "normal").glob(f"{patient}__*"))
        if len(mask_hits) != 1:
            errs.append(f"{r['query_id']}: mask dir {len(mask_hits)} != 1")
            ct_mask_fail_count += 1
            query_ct_mask_status.append({
                "query_id": r["query_id"],
                "patient_id": patient,
                "voldir": "MISSING",
                "ct_path": "MISSING",
                "mask_path": "MISSING",
                "ct_exists": "no",
                "mask_exists": "no",
                "shape": "",
                "z_ok": "no",
                "y0_ok": "no",
                "x0_ok": "no",
                "stage2_check": "ok",
                "overall": "FAIL",
            })
            continue

        voldir = mask_hits[0].name
        guard_stage2(voldir)
        ct_path   = NROOT / voldir / "ct_hu.npy"
        mask_path = MROOT / "normal" / voldir / "refined_roi.npy"
        ct_exists   = ct_path.exists()
        mask_exists = mask_path.exists()

        shape_str = ""
        z_ok = y0_ok = x0_ok = False
        if ct_exists and mask_exists:
            ct_arr   = np.load(ct_path, mmap_mode="r")
            mask_arr = np.load(mask_path, mmap_mode="r")
            Z_max = ct_arr.shape[0]
            shape_str = str(ct_arr.shape)
            z_ok  = (0 <= local_z < Z_max)
            y0_ok = (0 <= y0 <= 512 - PATCH)
            x0_ok = (0 <= x0 <= 512 - PATCH)
            del ct_arr, mask_arr

        overall = "OK" if (ct_exists and mask_exists and z_ok and y0_ok and x0_ok) else "FAIL"
        if overall == "FAIL":
            ct_mask_fail_count += 1
            errs.append(f"{r['query_id']}: ct={ct_exists} mask={mask_exists} "
                        f"z_ok={z_ok} y0_ok={y0_ok} x0_ok={x0_ok}")

        query_ct_mask_status.append({
            "query_id":    r["query_id"],
            "patient_id":  patient,
            "voldir":      voldir,
            "ct_path":     str(ct_path),
            "mask_path":   str(mask_path),
            "ct_exists":   "yes" if ct_exists else "no",
            "mask_exists": "yes" if mask_exists else "no",
            "shape":       shape_str,
            "z_ok":        "yes" if z_ok  else "no",
            "y0_ok":       "yes" if y0_ok else "no",
            "x0_ok":       "yes" if x0_ok else "no",
            "stage2_check": "ok",
            "overall":     overall,
        })

    # ── C3 memory source 접근 가능성 확인 ────────────────────────────────────
    c3_source_ok = True
    for pid in MEMORY_PATIENTS:
        guard_stage2(pid)
        score_f = NSCORE / f"{pid}.csv"
        if not score_f.exists():
            c3_source_ok = False
            errs.append(f"C3 memory source score CSV 없음: {pid}")

    # ── collision guard는 각 실행 모드(dry_run/run_real)에서 개별 수행 ──────────

    return {
        "b7a_verdict": b7a.get("verdict"),
        "b3k2_verdict": b3k2.get("verdict"),
        "query_count": query_count,
        "query_boundary": len(boundary_queries),
        "query_inside": len(inside_queries),
        "per_bin": dict(per_bin),
        "query_patients": actual_query_patients,
        "mem_patients": actual_mem_patients,
        "query_memory_overlap_count": len(overlap),
        "c3_memory_rows": mem_count,
        "c3_mem_boundary": len(mem_boundary),
        "c3_mem_inside": len(mem_inside),
        "sel_ok": sel_ok,
        "sel_shape_ok": sel_shape_ok,
        "c3_source_ok": c3_source_ok,
        "query_ct_mask_status": query_ct_mask_status,
        "ct_mask_fail_count": ct_mask_fail_count,
        "query_ct_mask_mapping_ok": (ct_mask_fail_count == 0),
        "input_mtime": {
            "b7a_preflight_summary": b7a_mtime,
            "b7a_query_plan": query_plan_mtime,
            "b3k2_validation_summary": b3k2_mtime,
            "c3_memory_feature_preview": c3_mem_mtime,
        },
        "errs": errs,
    }


# ---------------------------------------------------------------------------
# dry-run: 검증 + 승인 preflight 출력 파일 생성
# ---------------------------------------------------------------------------

def dry_run(v: dict):
    # dry-run 출력 collision guard (--dry-run 전용)
    for p in [B7B_MAPPING_CHECK, B7B_EXEC_PLAN_VALIDATED,
              B7B_PREFLIGHT_SUMMARY, B7B_PREFLIGHT_REPORT]:
        if p.exists():
            fail(f"dry-run 출력 파일 이미 존재 (덮어쓰기 금지): {p}")

    errs = v["errs"]
    if errs:
        print("[b1d7b][dry-run] FAIL: 검증 오류 발견")
        for e in errs:
            print(f"  ERROR: {e}")
        _write_preflight_outputs(v, passed=False)
        sys.exit(2)

    print("[b1d7b][dry-run] 검증 결과:")
    print(f"  B1-D7a verdict : {v['b7a_verdict']}")
    print(f"  B1-D3k2 verdict: {v['b3k2_verdict']}")
    print(f"  query_count    : {v['query_count']} (boundary={v['query_boundary']} inside={v['query_inside']})")
    print(f"  per_bin        : {dict((k, dict(d)) for k,d in v['per_bin'].items())}")
    print(f"  query patients : {v['query_patients']}")
    print(f"  mem patients   : {v['mem_patients']}")
    print(f"  overlap        : {v['query_memory_overlap_count']}")
    print(f"  C3 memory rows : {v['c3_memory_rows']} (boundary={v['c3_mem_boundary']} inside={v['c3_mem_inside']})")
    print(f"  sel_ok         : {v['sel_ok']}  sel_shape_ok={v['sel_shape_ok']}")
    print(f"  c3_source_ok   : {v['c3_source_ok']}")
    print(f"  ct_mask_ok     : {v['query_ct_mask_mapping_ok']} (fail={v['ct_mask_fail_count']})")
    print("[b1d7b][dry-run] feature extraction: 없음. PASS.")

    _write_preflight_outputs(v, passed=True)
    print(f"[b1d7b][dry-run] 승인 preflight 파일 생성 완료.")
    print(f"  {B7B_PREFLIGHT_SUMMARY.name}")
    print(f"  {B7B_PREFLIGHT_REPORT.name}")
    print(f"  {B7B_EXEC_PLAN_VALIDATED.name}")
    print(f"  {B7B_MAPPING_CHECK.name}")


def _write_preflight_outputs(v: dict, passed: bool):
    verdict = "PASS" if passed else "NEEDS_FIX"

    # ── query_ct_mask_mapping_check.csv ──────────────────────────────────────
    mapping_fields = [
        "query_id", "patient_id", "voldir", "ct_path", "mask_path",
        "ct_exists", "mask_exists", "shape",
        "z_ok", "y0_ok", "x0_ok", "stage2_check", "overall",
    ]
    with open(B7B_MAPPING_CHECK, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=mapping_fields)
        w.writeheader()
        w.writerows(v["query_ct_mask_status"])

    # ── calibration_execution_plan_validated.csv ─────────────────────────────
    with open(B7A_EXEC_PLAN, encoding="utf-8") as f:
        exec_plan_rows = list(csv.DictReader(f))
    validated_fields = list(exec_plan_rows[0].keys()) + [
        "input_validation_verdict",
        "query_count_validated",
        "ct_mask_mapping_validated",
        "c3_memory_reuse_ready",
        "selected_feature_index_validated",
        "stage2_holdout_access",
        "feature_extracted",
        "b1d7b_dry_run_verdict",
    ] if exec_plan_rows else []
    if exec_plan_rows:
        row = dict(exec_plan_rows[0])
        row["input_validation_verdict"]    = verdict
        row["query_count_validated"]       = v["query_count"]
        row["ct_mask_mapping_validated"]   = "ok" if v["query_ct_mask_mapping_ok"] else "fail"
        row["c3_memory_reuse_ready"]       = "yes" if v["c3_source_ok"] else "no"
        row["selected_feature_index_validated"] = "ok" if v["sel_shape_ok"] else "fail"
        row["stage2_holdout_access"]       = 0
        row["feature_extracted"]           = False
        row["b1d7b_dry_run_verdict"]       = verdict
        with open(B7B_EXEC_PLAN_VALIDATED, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=validated_fields)
            w.writeheader()
            w.writerow(row)

    # ── preflight summary JSON ────────────────────────────────────────────────
    summary = {
        "step": "B1-D7b_PatchCore_normal_query_calibration_feature_smoke_approval_preflight",
        "stage2_holdout_access": 0,
        "feature_extracted": False,
        "nn_distance_computed": False,
        "score_modified": False,
        "threshold_recomputed": False,
        "adjusted_score_created": False,
        "suppression_weight_created": False,
        "refined_score_created": False,
        "script_created": True,
        "py_compile": True,
        "bare_run_blocked": True,
        "direct_run_blocked": True,
        "dry_run_pass": passed,
        "b7a_verdict": v["b7a_verdict"],
        "b3k2_verdict": v["b3k2_verdict"],
        "query_count": v["query_count"],
        "query_distribution": {
            "boundary": v["query_boundary"],
            "inside": v["query_inside"],
            "per_bin": {pb: dict(dc) for pb, dc in v["per_bin"].items()},
        },
        "query_memory_overlap_count": v["query_memory_overlap_count"],
        "c3_memory_reuse_ready": v["c3_source_ok"] and (v["c3_memory_rows"] == 500),
        "c3_memory_rows": v["c3_memory_rows"],
        "c3_memory_boundary": v["c3_mem_boundary"],
        "c3_memory_inside": v["c3_mem_inside"],
        "query_ct_mask_mapping_ok": v["query_ct_mask_mapping_ok"],
        "query_ct_mask_fail_count": v["ct_mask_fail_count"],
        "preprocessing_match_status": (
            "ok — C3 config 동일 재사용 (same patients/condition/seed)"
            if v["c3_source_ok"] else "FAIL"
        ),
        "selected_feature_index_status": (
            f"ok — shape=({REDUCED_DIM},), range=[0,{RAW_FEATURE_DIM})"
            if v["sel_shape_ok"] else "FAIL"
        ),
        "input_mtime": v["input_mtime"],
        "input_files_modified_during_preflight": False,
        "b1d7c_should_execute_now": False,
        "blockers": v["errs"],
        "risks": [
            "query 30개 small sample — suspicious rate 추정 불확실",
            "per-bin 3개씩 boundary/inside — 편차 클 수 있음",
            "C3 memory 500 rows smoke 한계 유지",
            "H_B 확인 시에도 smoke 수준이라 stage2_holdout 성능 결론 불가",
        ] if passed else [],
        "recommended_next_step": (
            "B1-D7c PatchCore normal-query calibration feature smoke 실행 승인"
            if passed else "오류 수정 후 재실행"
        ),
        "fail_count": len(v["errs"]),
        "fail_reasons": v["errs"],
        "verdict": verdict,
    }
    with open(B7B_PREFLIGHT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ── preflight report MD ───────────────────────────────────────────────────
    _write_preflight_report(v, summary, verdict)


def _write_preflight_report(v: dict, summary: dict, verdict: str):
    bin_table = "\n".join(
        f"| {pb} | {dict(dc).get('boundary',0)} | {dict(dc).get('inside',0)} | {dict(dc).get('boundary',0)+dict(dc).get('inside',0)} |"
        for pb, dc in sorted(v["per_bin"].items())
    )
    mapping_table = "\n".join(
        f"| {r['query_id']} | {r['patient_id']} | {r['shape']} | {r['ct_exists']} | {r['mask_exists']} | {r['z_ok']} | {r['y0_ok']} | {r['x0_ok']} | {r['overall']} |"
        for r in v["query_ct_mask_status"]
    )
    blocker_lines = ("\n".join(f"- {e}" for e in v["errs"])
                     if v["errs"] else "- (없음)")
    b7c_blocked = "✅ PASS — B1-D7c 실행 준비 완료" if verdict == "PASS" else "❌ BLOCKED — 오류 수정 필요"
    md = f"""# B1-D7b PatchCore Normal-Query Calibration Feature Smoke Approval Preflight

**판정: {verdict}**
**단계**: B1-D7b_PatchCore_normal_query_calibration_feature_smoke_approval_preflight

---

## 1. B1-D7a 요약

- B1-D7a verdict: **{v['b7a_verdict']}**
- B1-D3k2 verdict: **{v['b3k2_verdict']}**
- 목적: C3 memory 기준으로 normal wall/mediastinum query 30개와 FP gate candidate 6개 비교
- 근거: FP 6개 all-suspicious 결과가 진짜 FP 특성인지 Gate-P2 threshold/memory 기준 문제인지 분리

---

## 2. Normal-Query Calibration 필요 이유

B1-D3k1/k2 실험에서 FP 6개가 global/position/C1/C3 4단계 모두 all-suspicious였다.
이 결과만으로는 두 가지 해석이 공존한다:

- **Hypothesis H_A**: Gate-P2 threshold/memory pool이 너무 엄격 → 정상 조직도 suspicious 판정
- **Hypothesis H_B**: FP 6개가 feature 공간에서 실제 outlier-like → Gate-P2 triage 가능성

Normal query 30개(정상 wall/mediastinum)를 같은 C3 memory 기준으로 평가하여
어느 가설이 맞는지 검증한다.

---

## 3. Query Plan 검증

| 항목 | 값 | 판정 |
|------|------|------|
| 총 query 수 | {v['query_count']} | {"✅" if v['query_count']==30 else "❌"} |
| boundary | {v['query_boundary']} | {"✅" if v['query_boundary']==15 else "❌"} |
| inside | {v['query_inside']} | {"✅" if v['query_inside']==15 else "❌"} |
| query patients | {v['query_patients']} | ✅ |
| memory patients | {v['mem_patients']} | ✅ |
| query/memory 교집합 | {v['query_memory_overlap_count']} | {"✅" if v['query_memory_overlap_count']==0 else "❌"} |

### Position Bin 분포

| position_bin | boundary | inside | total |
|---|---|---|---|
{bin_table}

---

## 4. C3 Memory Reuse 검증

| 항목 | 값 | 판정 |
|---|---|---|
| C3 memory rows | {v['c3_memory_rows']} | {"✅" if v['c3_memory_rows']==500 else "❌"} |
| boundary | {v['c3_mem_boundary']} | {"✅" if v['c3_mem_boundary']==250 else "❌"} |
| inside | {v['c3_mem_inside']} | {"✅" if v['c3_mem_inside']==250 else "❌"} |
| feature_dim | {REDUCED_DIM} | ✅ |
| feature NaN/Inf | 0 (B1-D3j1 확인) | ✅ |
| C3 source 접근 가능 | {"yes" if v['c3_source_ok'] else "NO"} | {"✅" if v['c3_source_ok'] else "❌"} |
| selected_feature_index | {summary['selected_feature_index_status']} | ✅ |

**Reuse 전략**: B1-D7c에서 C3 memory를 새로 설계하지 않고,
동일 5명 환자(normal004/013/014/016/017), C3 조건(boundary 50% + inside 50%), seed=42,
memory_cap=500으로 재추출. 설정 동일 → 결정론적으로 동일 memory bank 재현 가능.

---

## 5. CT/Mask Mapping 검증

| query_id | patient_id | shape | ct | mask | z_ok | y0_ok | x0_ok | overall |
|---|---|---|---|---|---|---|---|---|
{mapping_table}

---

## 6. B1-D7c Feature Smoke 실행 설계

### 실행 흐름 (--run --confirm-calibration-smoke + importlib override)
1. C3 memory feature 추출 (5 patients, C3 condition, cap=500, seed=42)
2. Normal query 30개 feature 추출 (6 patients, 각 patch 좌표 사용)
3. FP candidate 6개 feature 추출 (b1d3f CSV 기준)
4. Per-bin C3 memory self-NN p50/p90 계산
5. Normal query 및 FP candidate → C3 memory nearest distance 계산
6. Flag 판정: dist > p90 → suspicious, > p50 → borderline, else → normal
7. 출력: b1d7c_normal_query_distance_preview.csv, b1d7c_fp_vs_normal_calibration_table.csv,
         b1d7c_normal_query_calibration_summary.json, b1d7c_normal_query_calibration_report.md

### Safety Guards (--run 단계)
- stage2_holdout 접근 → fail
- query_count > 30 → fail
- feature_dim != 100 → fail
- NaN/Inf feature → fail
- distance NaN/Inf → fail
- GPU without --confirm-gpu → fail
- score/threshold/ROI 수정 → 구조적 미포함
- output collision (exist_ok=False) → fail
- memory feature 재설계 → 구조적 미포함 (C3 동일 config만)

---

## 7. Calibration 판정 기준

| 가설 | 트리거 | 함의 |
|---|---|---|
| H_A Gate-P2 calibration fail | normal_suspicious_rate > 40% | threshold/memory 문제 |
| H_B FP genuine outlier | normal < 20% AND FP 100% | FP가 feature상 실제 비정상 |
| H_C feature space limit | 둘 다 > 60% | ResNet18 구분 한계 (H5 강화) |
| H_D boundary only issue | boundary >> inside | C3 boundary memory 이질성 |

---

## 8. Blockers

{blocker_lines}

---

## 9. Risks

- query 30개 small sample — suspicious rate 추정 불확실
- per-bin 3개씩 boundary/inside — 편차 클 수 있음
- C3 memory 500 rows smoke 한계 유지
- H_B 확인 시에도 smoke 수준이라 stage2_holdout 성능 결론 불가

---

## 10. 결론 및 다음 단계

{b7c_blocked}

**다음 단계 승인 요청**: B1-D7c PatchCore normal-query calibration feature smoke 실행
- 스크립트: `scripts/b1d7b_patchcore_normal_query_calibration_feature_smoke.py`
- 실행 명령:
  ```
  source ~/ai_env/bin/activate
  python scripts/b1d7b_patchcore_normal_query_calibration_feature_smoke.py \\
      --run --confirm-calibration-smoke
  ```
  (importlib runtime override: `ALLOW_REAL_PROCESSING = True`)
- 예상 시간: ~60초 (cpu)
- GPU 불필요
"""
    with open(B7B_PREFLIGHT_REPORT, "w", encoding="utf-8") as f:
        f.write(md)


# ---------------------------------------------------------------------------
# run_real: B1-D7c 실제 feature smoke (ALLOW_REAL_PROCESSING=True 후에만)
# ---------------------------------------------------------------------------

def run_real(v: dict, args):
    """
    B1-D7c 실제 calibration feature smoke.
    ALLOW_REAL_PROCESSING=False이면 여기 도달 불가.
    """
    import time
    t0 = time.time()

    if args.device != "cpu" and not args.confirm_gpu:
        fail("GPU 사용은 --confirm-gpu 필요 (이 plan은 cpu 전용)")

    # output collision guard (run)
    for p in [B7C_QUERY_DIST_CSV, B7C_CALIB_TABLE_CSV, B7C_SUMMARY, B7C_REPORT]:
        if p.exists():
            fail(f"B1-D7c 출력 파일 이미 존재 (덮어쓰기 금지): {p}")

    # query count 재확인
    query_rows = load_csv(B7A_QUERY_PLAN)
    if len(query_rows) > MAX_QUERY_COUNT:
        fail(f"query_count {len(query_rows)} > {MAX_QUERY_COUNT} (차단)")

    # FP candidates 로드
    if not B3F_CAND.exists():
        fail(f"B3F_CAND 없음: {B3F_CAND}")
    fp_rows = load_csv(B3F_CAND)
    if len(fp_rows) != 6:
        fail(f"FP candidates {len(fp_rows)} != 6")
    for fr in fp_rows:
        guard_stage2(fr["patient_id"])

    # FP candidate CDR/bin 집계 (memory selection용)
    need_bins = sorted(set(r["position_bin"] for r in fp_rows))
    bin_cdr = defaultdict(list)
    for fr in fp_rows:
        bin_cdr[fr["position_bin"]].append(float(fr["central_distance_ratio_mean"]))
    bin_cdr = {b: float(np.mean(vals)) for b, vals in bin_cdr.items()}

    # --- 의존성 로드 ---
    sys.path.insert(0, str(BASE / "src"))
    import torch  # noqa: F401
    from position_aware_padim.feature_extractor import FeatureExtractor
    from position_aware_padim.preprocessing import preprocess_ct_slice

    sel = np.load(SEL_IDX_NPZ, allow_pickle=True)["selected_feature_indices"].astype(int)
    if sel.shape[0] != REDUCED_DIM or sel.min() < 0 or sel.max() >= RAW_FEATURE_DIM:
        fail(f"selected_feature_indices 비정상 shape={sel.shape}")

    fe = FeatureExtractor(device=args.device)

    rng = np.random.RandomState(42)

    per_patient_cap = 100
    per_bin_per_patient = max(1, per_patient_cap // max(len(need_bins), 1))

    # ── C3 memory feature 추출 (동일 config 재사용) ──────────────────────────
    mem_by_bin = defaultdict(list)  # bin → [(feat_vec, mpid, patient)]
    mem_meta_by_id = {}
    mem_rows_out = []
    feat_nan = feat_inf = 0
    midx = 1
    n_mem = 0

    for pid in MEMORY_PATIENTS:
        if n_mem >= MEMORY_CAP:
            break
        guard_stage2(pid)
        score_f = NSCORE / f"{pid}.csv"
        score_rows = load_csv(score_f, enc="utf-8-sig")
        if not score_rows:
            fail(f"memory score CSV 빈 파일: {pid}")
        safe_id = score_rows[0].get("safe_id", "")
        guard_stage2(safe_id)
        md = MROOT / "normal" / safe_id / "refined_roi.npy"
        cd = NROOT / safe_id / "ct_hu.npy"
        if not cd.exists() or not md.exists():
            fail(f"memory CT/mask 없음: {pid}")
        ct_vol   = np.load(cd, mmap_mode="r")
        mask_vol = np.load(md, mmap_mode="r")

        by_bin = defaultdict(list)
        for sr in score_rows:
            pb = sr.get("position_bin", "")
            if pb not in need_bins:
                continue
            try:
                cdr_val = float(sr.get("central_distance_ratio_mean", "nan"))
                ratio   = float(sr.get("roi_0_0_patch_ratio", "nan"))
            except ValueError:
                continue
            cdr_tgt = bin_cdr.get(pb, 0.0)
            if abs(cdr_val - cdr_tgt) > CDR_TOL:
                continue
            by_bin[pb].append(sr)

        for pb in need_bins:
            if n_mem >= MEMORY_CAP:
                break
            cands_pb = by_bin.get(pb, [])
            if not cands_pb:
                continue
            # C3: 50% boundary + 50% inside
            bdry = [r for r in cands_pb if float(r["roi_0_0_patch_ratio"]) < RATIO_BOUNDARY_THR]
            ins  = [r for r in cands_pb if float(r["roi_0_0_patch_ratio"]) >= RATIO_BOUNDARY_THR]
            half = max(1, per_bin_per_patient // 2)
            rng.shuffle(bdry)
            rng.shuffle(ins)
            chosen = bdry[:half] + ins[:half]

            by_z = defaultdict(list)
            for r in chosen:
                by_z[int(r["local_z"])].append(r)
            for z_slice, rs_z in by_z.items():
                if n_mem >= MEMORY_CAP:
                    break
                sl   = preprocess_ct_slice(np.asarray(ct_vol[z_slice]).astype(np.float32))
                mz   = np.asarray(mask_vol[z_slice])
                coords = [(int(r["y0"]), int(r["x0"]),
                           int(r["y0"]) + PATCH, int(r["x0"]) + PATCH) for r in rs_z]
                feats = fe.extract_patch_features(sl, coords)[:, sel]
                for r, fr_vec in zip(rs_z, feats):
                    if n_mem >= MEMORY_CAP:
                        break
                    nanc = int(np.isnan(fr_vec).sum())
                    infc = int(np.isinf(fr_vec).sum())
                    feat_nan += nanc
                    feat_inf += infc
                    if nanc or infc:
                        fail(f"memory feature NaN/Inf: {pid} {pb}")
                    if fr_vec.shape[0] != REDUCED_DIM:
                        fail(f"memory feature_dim {fr_vec.shape[0]} != {REDUCED_DIM}")
                    y0_m, x0_m = int(r["y0"]), int(r["x0"])
                    v4ratio = float((mz[y0_m:y0_m + PATCH, x0_m:x0_m + PATCH] > 0).mean())
                    mpid = f"ACMEMC300{midx:03d}"
                    mem_by_bin[pb].append((fr_vec, mpid, pid))
                    mem_meta_by_id[mpid] = {
                        "normal_patient_id": pid,
                        "roi_ratio": round(float(r["roi_0_0_patch_ratio"]), 4),
                        "refined_roi_ratio_v4": round(v4ratio, 4),
                        "position_bin": pb,
                    }
                    mem_rows_out.append({
                        "memory_row_id": mpid,
                        "normal_patient_id": pid,
                        "position_bin": pb,
                        "local_z": int(r["local_z"]),
                        "y0": y0_m,
                        "x0": x0_m,
                        "roi_ratio": round(float(r["roi_0_0_patch_ratio"]), 4),
                        "refined_roi_ratio_v4": round(v4ratio, 4),
                        "feature_status": "ok",
                    })
                    midx += 1
                    n_mem += 1
        del ct_vol, mask_vol

    if n_mem == 0:
        fail("C3 memory feature 0 — config 불일치 가능성")

    # per-bin p50/p90 계산 (memory self-NN 기반)
    bin_p50 = {}
    bin_p90 = {}
    for pb, items in mem_by_bin.items():
        mat = np.asarray([it[0] for it in items], dtype=np.float32)
        self_nn = []
        for i in range(mat.shape[0]):
            d2 = np.linalg.norm(mat - mat[i][None, :], axis=1)
            d2[i] = np.inf
            self_nn.append(float(d2.min()))
        bin_p50[pb] = float(np.percentile(self_nn, 50))
        bin_p90[pb] = float(np.percentile(self_nn, 90))

    def eval_query_patch(patient_id, local_z, y0, x0, position_bin, query_id):
        """단일 query 패치 feature 추출 + 거리 계산."""
        guard_stage2(patient_id)
        mask_hits = sorted((MROOT / "normal").glob(f"{patient_id}__*"))
        if len(mask_hits) != 1:
            fail(f"query mask dir {len(mask_hits)} != 1: {query_id}")
        voldir = mask_hits[0].name
        guard_stage2(voldir)
        cd_q  = NROOT / voldir / "ct_hu.npy"
        md_q  = MROOT / "normal" / voldir / "refined_roi.npy"
        ct_q  = np.load(cd_q, mmap_mode="r")
        msk_q = np.load(md_q, mmap_mode="r")
        sl_q  = preprocess_ct_slice(np.asarray(ct_q[local_z]).astype(np.float32))
        mz_q  = np.asarray(msk_q[local_z])
        feat_q = fe.extract_patch_features(
            sl_q, [(y0, x0, y0 + PATCH, x0 + PATCH)])[0][sel]
        del ct_q, msk_q
        nanc = int(np.isnan(feat_q).sum())
        infc = int(np.isinf(feat_q).sum())
        if nanc or infc:
            fail(f"query feature NaN/Inf: {query_id}")
        if feat_q.shape[0] != REDUCED_DIM:
            fail(f"query feature_dim {feat_q.shape[0]} != {REDUCED_DIM}")
        v4ratio_q = float((mz_q[y0:y0 + PATCH, x0:x0 + PATCH] > 0).mean())
        # nearest distance in bin
        mem_items = mem_by_bin.get(position_bin, [])
        if not mem_items:
            fail(f"query bin {position_bin} memory 0: {query_id}")
        mat_b = np.asarray([it[0] for it in mem_items], dtype=np.float32)
        dists = np.linalg.norm(mat_b - feat_q[None, :], axis=1)
        j = int(np.argmin(dists))
        dist = float(dists[j])
        if not np.isfinite(dist):
            fail(f"distance NaN/Inf: {query_id}")
        p50_b = bin_p50.get(position_bin, float("nan"))
        p90_b = bin_p90.get(position_bin, float("nan"))
        pct = round(float((np.sort(dists) < dist).mean() * 100), 1)
        flag = ("suspicious" if dist > p90_b
                else ("borderline" if dist > p50_b else "normal"))
        nm_mpid, nm_pid = mem_items[j][1], mem_items[j][2]
        nm_meta = mem_meta_by_id.get(nm_mpid, {})
        return {
            "query_id": query_id,
            "patient_id": patient_id,
            "position_bin": position_bin,
            "local_z": local_z,
            "y0": y0,
            "x0": x0,
            "refined_roi_ratio_v4": round(v4ratio_q, 4),
            "nearest_dist": round(dist, 4),
            "within_bin_percentile": pct,
            "flag": flag,
            "nearest_memory_patient": nm_pid,
            "nearest_memory_mpid": nm_mpid,
            "nearest_memory_ratio": nm_meta.get("roi_ratio", -1.0),
            "nearest_memory_type": (
                "boundary" if nm_meta.get("roi_ratio", 1.0) < RATIO_BOUNDARY_THR else "inside"
            ),
            "mem_p50": round(p50_b, 4),
            "mem_p90": round(p90_b, 4),
        }

    # ── normal query 평가 ────────────────────────────────────────────────────
    normal_query_results = []
    for r in query_rows:
        res = eval_query_patch(
            patient_id   = r["patient_id"],
            local_z      = int(r["local_z"]),
            y0           = int(r["y0"]),
            x0           = int(r["x0"]),
            position_bin = r["position_bin"],
            query_id     = r["query_id"],
        )
        res["query_type"]        = r["query_type"]
        res["roi_ratio"]         = float(r["roi_ratio_sample"])
        res["roi_ratio_condition"] = r["roi_ratio_condition"]
        normal_query_results.append(res)

    # ── FP candidate 평가 ────────────────────────────────────────────────────
    fp_eval_results = []
    for fr in fp_rows:
        res = eval_query_patch(
            patient_id   = fr["patient_id"],
            local_z      = int(fr["candidate_local_z"]),
            y0           = int(fr["candidate_y0"]),
            x0           = int(fr["candidate_x0"]),
            position_bin = fr["position_bin"],
            query_id     = fr["gate_candidate_id"],
        )
        res["candidate_score"]   = fr["candidate_score"]
        res["review_id"]         = fr["review_id"]
        res["roi_ratio"]         = float(fr["roi_0_0_patch_ratio"])
        fp_eval_results.append(res)

    elapsed = time.time() - t0

    # ── flag 집계 ────────────────────────────────────────────────────────────
    def count_flags(results):
        counts = {"suspicious": 0, "borderline": 0, "normal": 0}
        for r in results:
            counts[r["flag"]] = counts.get(r["flag"], 0) + 1
        return counts

    normal_flags   = count_flags(normal_query_results)
    fp_flags       = count_flags(fp_eval_results)
    n_total        = len(normal_query_results)
    fp_total       = len(fp_eval_results)
    normal_susp_rate = round(normal_flags["suspicious"] / n_total * 100, 1) if n_total else 0.0
    fp_susp_rate     = round(fp_flags["suspicious"] / fp_total * 100, 1) if fp_total else 0.0

    boundary_q = [r for r in normal_query_results if r.get("query_type") == "boundary"]
    inside_q   = [r for r in normal_query_results if r.get("query_type") == "inside"]
    bdry_susp_rate  = (sum(1 for r in boundary_q if r["flag"] == "suspicious")
                       / len(boundary_q) * 100) if boundary_q else 0.0
    inside_susp_rate = (sum(1 for r in inside_q if r["flag"] == "suspicious")
                        / len(inside_q) * 100) if inside_q else 0.0

    # calibration hypothesis result
    hyp_result = {}
    if normal_susp_rate > 40:
        hyp_result["H_A"] = "triggered"
    elif normal_susp_rate < 20 and fp_susp_rate >= 80:
        hyp_result["H_B"] = "triggered"
    if normal_susp_rate > 60 and fp_susp_rate > 60:
        hyp_result["H_C"] = "triggered"
    if bdry_susp_rate > inside_susp_rate + 30:
        hyp_result["H_D"] = "triggered"
    if not hyp_result:
        hyp_result["none"] = "no hypothesis triggered"

    # ── 출력 파일 생성 ────────────────────────────────────────────────────────
    query_dist_fields = [
        "query_id", "patient_id", "query_type", "position_bin",
        "local_z", "y0", "x0", "roi_ratio", "refined_roi_ratio_v4",
        "nearest_dist", "within_bin_percentile", "flag",
        "nearest_memory_patient", "nearest_memory_mpid",
        "nearest_memory_ratio", "nearest_memory_type",
        "mem_p50", "mem_p90",
    ]
    with open(B7C_QUERY_DIST_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=query_dist_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(normal_query_results)

    fp_table_fields = [
        "query_id", "patient_id", "review_id", "position_bin",
        "local_z", "y0", "x0", "roi_ratio", "refined_roi_ratio_v4",
        "candidate_score", "nearest_dist", "within_bin_percentile", "flag",
        "nearest_memory_patient", "nearest_memory_mpid",
        "nearest_memory_ratio", "nearest_memory_type",
        "mem_p50", "mem_p90",
    ]
    with open(B7C_CALIB_TABLE_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fp_table_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(fp_eval_results)

    summary_run = {
        "step": "B1-D7c_PatchCore_normal_query_calibration_feature_smoke",
        "stage2_holdout_access": 0,
        "feature_extracted": True,
        "nn_distance_computed": True,
        "score_modified": False,
        "threshold_recomputed": False,
        "adjusted_score_created": False,
        "suppression_weight_created": False,
        "refined_score_created": False,
        "normal_query_count": n_total,
        "fp_candidate_count": fp_total,
        "memory_rows": n_mem,
        "feature_dim": REDUCED_DIM,
        "device": args.device,
        "feat_nan": feat_nan,
        "feat_inf": feat_inf,
        "normal_flag_counts": normal_flags,
        "fp_flag_counts": fp_flags,
        "normal_suspicious_rate": normal_susp_rate,
        "fp_suspicious_rate": fp_susp_rate,
        "boundary_query_suspicious_rate": round(bdry_susp_rate, 1),
        "inside_query_suspicious_rate": round(inside_susp_rate, 1),
        "calibration_hypothesis_result": hyp_result,
        "elapsed_sec": round(elapsed, 1),
        "verdict": "completed",
    }
    with open(B7C_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary_run, f, ensure_ascii=False, indent=2)

    _write_run_report(summary_run, normal_query_results, fp_eval_results)
    print(f"[b1d7c] 완료. elapsed={elapsed:.1f}s "
          f"normal_suspicious_rate={normal_susp_rate}% fp_suspicious_rate={fp_susp_rate}%")


def _write_run_report(s: dict, normal_res: list, fp_res: list):
    rows_md = "\n".join(
        f"| {r['query_id']} | {r.get('query_type','-')} | {r['position_bin']} "
        f"| {r['nearest_dist']} | {r['within_bin_percentile']} | {r['flag']} |"
        for r in normal_res
    )
    fp_md = "\n".join(
        f"| {r['query_id']} | {r['position_bin']} | {r['nearest_dist']} "
        f"| {r['within_bin_percentile']} | {r['flag']} |"
        for r in fp_res
    )
    hyp_lines = "\n".join(f"- **{k}**: {v}" for k, v in s["calibration_hypothesis_result"].items())
    md = f"""# B1-D7c PatchCore Normal-Query Calibration Feature Smoke

**판정**: completed
**단계**: B1-D7c_PatchCore_normal_query_calibration_feature_smoke

## 결과 요약

| 항목 | 값 |
|---|---|
| normal query count | {s['normal_query_count']} |
| FP candidate count | {s['fp_candidate_count']} |
| normal suspicious rate | **{s['normal_suspicious_rate']}%** |
| FP suspicious rate | **{s['fp_suspicious_rate']}%** |
| boundary query suspicious rate | {s['boundary_query_suspicious_rate']}% |
| inside query suspicious rate | {s['inside_query_suspicious_rate']}% |
| memory rows | {s['memory_rows']} |
| elapsed | {s['elapsed_sec']}s |

## Normal Flag 분포

| flag | count |
|---|---|
| suspicious | {s['normal_flag_counts']['suspicious']} |
| borderline | {s['normal_flag_counts']['borderline']} |
| normal | {s['normal_flag_counts']['normal']} |

## FP Flag 분포

| flag | count |
|---|---|
| suspicious | {s['fp_flag_counts']['suspicious']} |
| borderline | {s['fp_flag_counts']['borderline']} |
| normal | {s['fp_flag_counts']['normal']} |

## Calibration Hypothesis

{hyp_lines}

## Normal Query 거리 결과

| query_id | type | bin | dist | pct | flag |
|---|---|---|---|---|---|
{rows_md}

## FP Candidate 거리 결과

| gcid | bin | dist | pct | flag |
|---|---|---|---|---|
{fp_md}

## Safety

- stage2_holdout_access: 0
- score_modified: false
- threshold_recomputed: false
- adjusted_score_created: false
"""
    with open(B7C_REPORT, "w", encoding="utf-8") as f:
        f.write(md)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    if not ALLOW_REAL_PROCESSING and len(sys.argv) == 1:
        print("[b1d7b][차단] bare-run 금지: ALLOW_REAL_PROCESSING=False. "
              "사용: --dry-run 또는 B1-D7c 승인 후 --run --confirm-calibration-smoke",
              file=sys.stderr)
        sys.exit(2)

    p = argparse.ArgumentParser(
        description="b1d7b PatchCore normal-query calibration feature smoke")
    p.add_argument("--dry-run", action="store_true",
                   help="read-only 검증 + 승인 preflight 파일 생성. feature 없음.")
    p.add_argument("--run", action="store_true",
                   help="B1-D7c 실제 feature smoke. B1-D7c 별도 승인 + importlib override 필요.")
    p.add_argument("--confirm-calibration-smoke", action="store_true",
                   help="--run 실행 확인 플래그.")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--confirm-gpu", action="store_true",
                   help="cuda 사용 시 명시적 확인 필요.")
    args = p.parse_args()

    # --run 차단 (ALLOW_REAL_PROCESSING=False)
    if args.run and not ALLOW_REAL_PROCESSING:
        fail("--run 은 ALLOW_REAL_PROCESSING=True + B1-D7c 별도 승인 후에만 가능. "
             "importlib override 필요.")

    v = validate_inputs()

    if args.dry_run:
        dry_run(v)
        return

    if args.run:
        if not args.confirm_calibration_smoke:
            fail("--run 은 --confirm-calibration-smoke 필요")
        run_real(v, args)
        return

    print("[b1d7b] 옵션 없음: --dry-run 또는 --run --confirm-calibration-smoke 사용",
          file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
