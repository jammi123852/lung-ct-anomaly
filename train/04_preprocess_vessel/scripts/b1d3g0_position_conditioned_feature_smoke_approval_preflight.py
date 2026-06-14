#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B1-D3g0_Gate_P2_position_conditioned_minimal_feature_smoke_approval_preflight

position-conditioned feature smoke(B1-D3g1) 실행 전 승인 전 preflight.
- feature/GPU/CUDA/memory bank/NN/distance 없음.
- smoke 스크립트(b1d3g_...)를 py_compile + bare-run(exit2) + --dry-run(feature0/파일0)로 검증.
- Plan-PC-S sampling plan(execution_plan.csv) 확정. 출력 4개 collision guard. 입력 mtime 무수정.
"""
import csv
import json
import subprocess
import sys
import py_compile
from pathlib import Path
from collections import defaultdict

BASE = Path("/home/jinhy/project/lung-ct-anomaly")
DIR = BASE / "outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1"
SMOKE = BASE / "scripts/b1d3g_gate_p2_position_conditioned_feature_smoke.py"
SEL_IDX_NPZ = BASE / "outputs/position-aware-padim-v1/models/padim_v2_roi0_0/distributions/position_bin_stats.npz"

IN = {
    "b3f_cand": DIR / "b1d3f_gate_p2_position_conditioned_candidates_summary.csv",
    "b3f_pool": DIR / "b1d3f_gate_p2_position_conditioned_memory_pool_preview.csv",
    "b3f_summary": DIR / "b1d3f_gate_p2_position_conditioned_memory_preflight_summary.json",
    "b3f_report": DIR / "b1d3f_gate_p2_position_conditioned_memory_preflight_report.md",
    "b3c_cand": DIR / "b1d3c_gate_p2_feature_preflight_candidates.csv",
    "b3d1_summary": DIR / "b1d3d1_gate_p2_minimal_feature_smoke_plan_s_cpu_v1/b1d3d1_gate_p2_minimal_feature_smoke_summary.json",
}

OUT_MD = DIR / "b1d3g0_position_conditioned_feature_smoke_approval_preflight_report.md"
OUT_JSON = DIR / "b1d3g0_position_conditioned_feature_smoke_approval_preflight_summary.json"
OUT_PLAN = DIR / "b1d3g0_position_conditioned_execution_plan.csv"

PY = sys.executable
MEM_PATIENTS = 5
PER_PATIENT_CAP = 100
TOTAL_CAP = 500
CAND_LIMIT = 6


def fail(msg):
    print(f"[B1-D3g0][중단] {msg}", file=sys.stderr)
    sys.exit(2)


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_rows(p, enc="utf-8"):
    with open(p, encoding=enc) as f:
        return list(csv.DictReader(f))


def main():
    for p in (OUT_MD, OUT_JSON, OUT_PLAN):
        if p.exists():
            fail(f"출력 파일이 이미 존재함(덮어쓰기 금지): {p}")

    input_mtimes = {}
    for k, p in IN.items():
        if not p.exists():
            fail(f"필수 입력 없음: {k} -> {p}")
        input_mtimes[k] = round(p.stat().st_mtime, 3)
    if not SMOKE.exists():
        fail(f"smoke 스크립트 없음: {SMOKE}")

    cands = load_rows(IN["b3f_cand"])
    if len(cands) != 6:
        fail(f"candidates {len(cands)} != 6")
    b3f = load_json(IN["b3f_summary"])
    if b3f.get("position_conditioning_ready") is not True:
        fail("position_conditioning_ready != True")
    if b3f.get("memory_bias_reduced_assessment", {}).get("reduced") is not True:
        fail("memory_bias_reduced != True")
    if "Plan-PC-S(권장)" not in b3f.get("recommended_next_plan", {}):
        fail("recommended_next_plan 에 Plan-PC-S 없음")
    if b3f.get("stage2_holdout_access") != 0:
        fail("b3f stage2_holdout_access != 0")
    stage2_holdout_access = 0

    b3d1 = load_json(IN["b3d1_summary"])
    preprocessing_match_status = b3d1.get("preprocessing_match_status", "")
    selected_feature_index_status = b3d1.get("selected_feature_index_status", "")
    if "match" not in preprocessing_match_status.lower():
        fail("preprocessing match 확인 실패")
    if "matched v2" not in selected_feature_index_status.lower():
        fail("selected feature index v2 일치 확인 실패")

    # bin → GC, z_level, cp
    bin_to_gc = defaultdict(list)
    bin_meta = {}
    for c in cands:
        pb = c["position_bin"]
        bin_to_gc[pb].append(c["gate_candidate_id"])
        bin_meta[pb] = (c["z_level_bin"], c["central_peripheral"])
    need_bins = sorted(bin_to_gc.keys())
    per_bin_per_patient = max(1, PER_PATIENT_CAP // len(need_bins))

    # memory 환자: b3f pool preview 환자 중 candidate 환자 제외, 앞 5
    cand_patients = set(c["patient_id"] for c in cands)
    pool = load_rows(IN["b3f_pool"])
    preview_patients = []
    for r in pool:
        if r["preview_patient_id"] not in preview_patients:
            preview_patients.append(r["preview_patient_id"])
    mem_patients = [p for p in preview_patients if p not in cand_patients][:MEM_PATIENTS]

    # ---- execution_plan.csv (환자 × bin) ----
    plan_rows = []
    pid_n = 1
    for pid in mem_patients:
        for pb in need_bins:
            zlv, cp = bin_meta[pb]
            plan_rows.append({
                "plan_id": f"PLAN{pid_n:03d}",
                "selected_memory_patient_id": pid,
                "matched_gate_candidate_id": ",".join(bin_to_gc[pb]),
                "position_bin": pb, "z_level": zlv, "central_peripheral": cp,
                "planned_patch_cap_for_patient": PER_PATIENT_CAP,
                "planned_patch_cap_for_candidate_condition": per_bin_per_patient * MEM_PATIENTS,
                "sampling_priority": "tight_cdr_within_position_bin",
                "include_in_b1d3g1": "true", "exclusion_reason": "",
            })
            pid_n += 1
    with open(OUT_PLAN, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(plan_rows[0].keys()))
        w.writeheader()
        w.writerows(plan_rows)

    # ---- smoke 스크립트 안전가드 검증 ----
    try:
        py_compile.compile(str(SMOKE), doraise=True)
        py_compile_pass = True
    except py_compile.PyCompileError:
        py_compile_pass = False
    bare = subprocess.run([PY, str(SMOKE)], capture_output=True, text=True)
    bare_run_exit_2 = (bare.returncode == 2)
    dry = subprocess.run([PY, str(SMOKE), "--dry-run"], capture_output=True, text=True)
    dry_result, dry_run_pass = {}, False
    for line in dry.stdout.splitlines():
        if line.startswith("DRYRUN_RESULT "):
            dry_result = json.loads(line[len("DRYRUN_RESULT "):])
            dry_run_pass = (dry.returncode == 0
                            and dry_result.get("feature_extracted") is False
                            and dry_result.get("gpu_used") is False
                            and dry_result.get("files_created") == 0
                            and dry_result.get("candidates") == 6
                            and dry_result.get("planned_memory_patients") == MEM_PATIENTS
                            and dry_result.get("total_patch_cap") == TOTAL_CAP
                            and dry_result.get("per_patient_patch_cap") == PER_PATIENT_CAP
                            and dry_result.get("position_coverage_complete") is True)
            break
    out_folder = DIR / "b1d3g1_gate_p2_position_conditioned_feature_smoke_plan_pc_s_cpu_v1"
    dry_no_folder = not out_folder.exists()
    position_coverage_complete = dry_result.get("position_coverage_complete", False)

    recommended_plan = "Plan-PC-S (CPU, GPU 불필요)"
    safety_abort_conditions = [
        "stage2_holdout 접근", "device가 cpu 아님", "CUDA 사용", "candidate rows 6 초과",
        "memory patients 5 초과", "per-patient cap 100 초과", "total cap 500 초과",
        "lesion patient memory 포함", "position coverage 누락", "selected index 100D 불일치",
        "preprocessing/window 불일치", "output folder 존재", "score write 시도",
        "adjusted/suppression/refined 생성 시도", "feature NaN/Inf", "distance NaN/Inf",
        "memory feature 0", "candidate feature 0",
    ]
    b1d3g1_output_schema = {
        "folder": "b1d3g1_gate_p2_position_conditioned_feature_smoke_plan_pc_s_cpu_v1/ (exist_ok=False)",
        "files": ["b1d3g1_position_conditioned_memory_feature_preview.csv",
                  "b1d3g1_position_conditioned_candidate_distance_preview.csv",
                  "b1d3g1_position_conditioned_feature_smoke_summary.json",
                  "b1d3g1_position_conditioned_feature_smoke_report.md"],
        "candidate_distance_columns": ["gate_candidate_id", "review_id", "patient_id", "position_bin",
                                       "candidate_score", "feature_status", "feature_dim",
                                       "nearest_distance", "nearest_memory_patient", "nearest_memory_patch_id",
                                       "matched_position_bin", "distance_percentile_within_position_pool",
                                       "gate_p2_flag", "flag_reason", "score_modified", "safety_note"],
        "memory_feature_columns": ["memory_patch_id", "memory_patient_id", "matched_gate_candidate_id",
                                   "position_bin", "z_level", "y0", "x0", "refined_roi_ratio",
                                   "feature_status", "feature_dim", "used_in_memory",
                                   "sampling_reason", "exclusion_reason"],
    }

    script_safety_pass = py_compile_pass and bare_run_exit_2 and dry_run_pass and dry_no_folder
    verdict = "PASS" if (script_safety_pass and position_coverage_complete) else "NEEDS_FIX"

    summary = {
        "step": "B1-D3g0_Gate_P2_position_conditioned_minimal_feature_smoke_approval_preflight",
        "verdict": verdict, "input_mtimes": input_mtimes,
        "stage2_holdout_access": stage2_holdout_access,
        "recommended_plan": recommended_plan,
        "gate_candidate_rows": len(cands),
        "candidate_position_bins": {c["gate_candidate_id"]: c["position_bin"] for c in cands},
        "planned_memory_patients": len(mem_patients),
        "selected_memory_patients": mem_patients,
        "planned_per_patient_patch_cap": PER_PATIENT_CAP,
        "planned_total_patch_cap": TOTAL_CAP,
        "planned_candidate_rows": len(cands),
        "per_bin_per_patient": per_bin_per_patient,
        "needed_position_bins": need_bins,
        "position_coverage_complete": position_coverage_complete,
        "patient_balance_expected": f"환자별 cap {PER_PATIENT_CAP} → 5명 균등(단일 환자 20%)",
        "preprocessing_match_status": preprocessing_match_status,
        "selected_feature_index_status": selected_feature_index_status,
        "selected_index_npz_exists": SEL_IDX_NPZ.exists(),
        "script_created": True, "script_path": str(SMOKE),
        "py_compile_pass": py_compile_pass, "bare_run_exit_2": bare_run_exit_2,
        "dry_run_pass": dry_run_pass, "dry_run_result": dry_result,
        "dry_run_feature_extracted": dry_result.get("feature_extracted"),
        "dry_run_gpu_used": dry_result.get("gpu_used"),
        "dry_run_memory_bank_created": dry_result.get("memory_bank_created"),
        "dry_run_nn_computed": dry_result.get("nearest_neighbor_computed"),
        "dry_run_score_modified": dry_result.get("score_modified"),
        "dry_run_no_output_folder": dry_no_folder,
        "safety_abort_conditions": safety_abort_conditions,
        "b1d3g1_output_schema": b1d3g1_output_schema,
        "minimal_feature_smoke_requires_user_approval": True,
        "gpu_required": False, "score_modified": False, "roi_modified": False,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    plan_tbl = "\n".join(
        f"| {r['plan_id']} | {r['selected_memory_patient_id']} | {r['position_bin']} | "
        f"{r['matched_gate_candidate_id']} | {r['planned_patch_cap_for_candidate_condition']} |"
        for r in plan_rows)

    md = f"""# B1-D3g0 Position-conditioned Feature Smoke — Approval Preflight

position-conditioned memory 기반 feature smoke(B1-D3g1) 실행 전 승인 전 preflight.
feature/GPU/CUDA/memory bank/NN/distance 없음. smoke 스크립트는 차단 상태로 안전가드만 검증.

## 0. 판정
**{verdict}**

## 1. B1-D3f 결과 요약
- position_conditioning_ready=True, memory_bias_reduced=True, candidate 6, preview 8명, pool 288.
- candidate position_bins: {summary['candidate_position_bins']}
- GC별 normal 후보 풍부(40k~80k), patient max_share 0.171.

## 2. 왜 B1-D3g0 이 필요한가
B1-D3d1 all-suspicious 가 memory mismatch 였으므로, 위치 정합 memory 로 다시 feature smoke 하기 전
범위·sampling·중단조건·스크립트 안전가드를 잠근다. 이번 단계 실행 0.

## 3. Plan-PC-S 실행 범위
- normal memory patients **{MEM_PATIENTS}** ({', '.join(mem_patients)})
- per-patient patch cap **{PER_PATIENT_CAP}**, total cap **{TOTAL_CAP}**, gate candidate **{len(cands)}**
- needed position_bins {need_bins} → per-bin-per-patient **{per_bin_per_patient}** (=cap/{len(need_bins)}bins)
- device cpu, feature_dim 100(v2 selected), preprocessing v2 동일, score 무수정

## 4. memory sampling plan (execution_plan.csv, {len(plan_rows)}행)
| plan_id | patient | position_bin | matched_GC | per-condition cap |
|---|---|---|---|---|
{plan_tbl}

## 5. position coverage
- dry-run position_coverage_complete = **{position_coverage_complete}** (모든 5환자 × {len(need_bins)}bin 에 ≥{per_bin_per_patient} 후보)
- coverage_gaps: {dry_result.get('coverage_gaps', [])}

## 6. script safety guard
- py_compile_pass **{py_compile_pass}**, bare_run_exit_2 **{bare_run_exit_2}**, dry_run_pass **{dry_run_pass}**, no_output_folder **{dry_no_folder}**
- ALLOW_REAL_PROCESSING=False / --real 은 --confirm-feature-smoke / device=cuda 는 --confirm-gpu / output exist_ok=False
- preprocessing_match: {preprocessing_match_status[:70]}
- selected_feature_index: {selected_feature_index_status[:70]}

## 7. 안전 중단 조건 (B1-D3g1)
{chr(10).join('- ' + s for s in safety_abort_conditions)}

## 8. B1-D3g1 출력 스키마
- 폴더: {b1d3g1_output_schema['folder']}
- 파일: {', '.join(b1d3g1_output_schema['files'])}

## 9. ★ 사용자 승인 필요
minimal_feature_smoke_requires_user_approval = **True**. GPU 불필요(**CPU Plan-PC-S, 과금 없음**).
- 승인 시: `--real --confirm-feature-smoke --device cpu --memory-patient-limit 5 --per-patient-patch-cap 100 --memory-patch-cap 500 --candidate-limit 6` + ALLOW_REAL_PROCESSING 런타임 override.
- 이 smoke 는 성능 개선 실험이 아니라 위치 정합 distance/flag preview.

## 10. 다음 단계
**B1-D3g1 Gate-P2 position-conditioned minimal feature smoke CPU Plan-PC-S 실행 승인/보류 결정**

---
feature_extracted=False, gpu_used=False, memory_bank_created=False, nearest_neighbor_computed=False, score_modified=False, stage2_holdout_access={stage2_holdout_access}
"""
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"[B1-D3g0] {verdict}")
    print(f"  py_compile={py_compile_pass}, bare_run_exit_2={bare_run_exit_2}, dry_run_pass={dry_run_pass}, no_folder={dry_no_folder}")
    print(f"  mem_patients={mem_patients}, per_bin_per_patient={per_bin_per_patient}, coverage_complete={position_coverage_complete}")
    print(f"  생성: {OUT_MD.name}, {OUT_JSON.name}, {OUT_PLAN.name}")


if __name__ == "__main__":
    main()
