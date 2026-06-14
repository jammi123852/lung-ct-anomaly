#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B1-D3d0_Gate_P2_minimal_feature_smoke_approval_preflight

minimal feature smoke 실행(B1-D3d1) 전, 비용/범위/안전조건을 확정하는 승인 전 preflight.
- feature 추출/GPU/CUDA/memory bank/NN/distance 일절 없음.
- smoke 스크립트(scripts/b1d3d_gate_p2_minimal_feature_smoke.py)를 py_compile + bare-run(exit 2)
  + --dry-run(feature 0/파일 0)로 안전가드만 검증.
- 출력 report/summary/execution_plan 이미 있으면 즉시 중단(덮어쓰기 금지). 입력 mtime 무수정.
"""
import csv
import json
import subprocess
import sys
import py_compile
from pathlib import Path

BASE = Path("/home/jinhy/project/lung-ct-anomaly")
DIR = BASE / "outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1"
SMOKE = BASE / "scripts/b1d3d_gate_p2_minimal_feature_smoke.py"

IN = {
    "cand": DIR / "b1d3c_gate_p2_feature_preflight_candidates.csv",
    "pool": DIR / "b1d3c_gate_p2_memory_pool_preview.csv",
    "b3c_summary": DIR / "b1d3c_gate_p2_feature_preflight_summary.json",
    "safety": DIR / "b1d3a_smoke_safety_manifest.csv",
    "b3b_summary": DIR / "b1d3b_rule_b3_dry_smoke_summary.json",
}

OUT_MD = DIR / "b1d3d0_gate_p2_minimal_feature_smoke_approval_preflight_report.md"
OUT_JSON = DIR / "b1d3d0_gate_p2_minimal_feature_smoke_approval_preflight_summary.json"
OUT_PLAN = DIR / "b1d3d0_gate_p2_execution_plan.csv"

FEATURE_DIM = 448
PY = sys.executable


def fail(msg):
    print(f"[B1-D3d0][중단] {msg}", file=sys.stderr)
    sys.exit(2)


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_rows(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt_mb(n_patch, d=FEATURE_DIM):
    return round(n_patch * d * 4 / (1024 * 1024), 3)


def main():
    # ---- collision guard ----
    for p in (OUT_MD, OUT_JSON, OUT_PLAN):
        if p.exists():
            fail(f"출력 파일이 이미 존재함(덮어쓰기 금지): {p}")

    # ---- 입력 검증 + mtime ----
    input_mtimes = {}
    for k, p in IN.items():
        if not p.exists():
            fail(f"필수 입력 없음: {k} -> {p}")
        input_mtimes[k] = round(p.stat().st_mtime, 3)
    if not SMOKE.exists():
        fail(f"smoke 스크립트 없음: {SMOKE}")

    cands = load_rows(IN["cand"])
    pool = load_rows(IN["pool"])
    if len(cands) != 6:
        fail(f"candidates row {len(cands)} != 6")
    if len(pool) != 6:
        fail(f"pool row {len(pool)} != 6")

    b3c = load_json(IN["b3c_summary"])
    for k in ["gpu_used", "feature_extracted", "memory_bank_created",
              "nearest_neighbor_computed", "score_modified"]:
        if b3c.get(k) is not False:
            fail(f"B1-D3c {k} != False")
    if b3c.get("stage2_holdout_access") != 0:
        fail("B1-D3c stage2_holdout_access != 0")
    stage2_holdout_access = 0

    # ---- smoke 스크립트 안전가드 검증 (py_compile / bare-run / dry-run) ----
    try:
        py_compile.compile(str(SMOKE), doraise=True)
        py_compile_pass = True
    except py_compile.PyCompileError:
        py_compile_pass = False

    bare = subprocess.run([PY, str(SMOKE)], capture_output=True, text=True)
    bare_run_exit_2 = (bare.returncode == 2)

    dry = subprocess.run([PY, str(SMOKE), "--dry-run"], capture_output=True, text=True)
    dry_run_pass, dry_result = False, {}
    for line in dry.stdout.splitlines():
        if line.startswith("DRYRUN_RESULT "):
            dry_result = json.loads(line[len("DRYRUN_RESULT "):])
            dry_run_pass = (dry.returncode == 0
                            and dry_result.get("feature_extracted") is False
                            and dry_result.get("files_created") == 0
                            and dry_result.get("gpu_used") is False)
            break
    # dry-run 후 output folder 미생성 확인
    out_folder = DIR / "b1d3d1_gate_p2_minimal_feature_smoke_v1"
    dry_no_folder = not out_folder.exists()

    # ---- Plan-S/M/L ----
    plan_s = {"plan": "Plan-S", "memory_patients": 3, "memory_patch_cap": 500,
              "gate_candidates": 6, "feature_matrix_mb": fmt_mb(500),
              "purpose": "코드/shape/거리 계산 smoke(최소)", "recommended": "true"}
    plan_m = {"plan": "Plan-M", "memory_patients": 5, "memory_patch_cap": 1500,
              "gate_candidates": 6, "feature_matrix_mb": fmt_mb(1500),
              "purpose": "최소 거리 분포 감 잡기(권장 상한)", "recommended": "secondary"}
    plan_l = {"plan": "Plan-L", "memory_patients": 10, "memory_patch_cap": 5000,
              "gate_candidates": 6, "feature_matrix_mb": fmt_mb(5000),
              "purpose": "smoke 단계엔 과함 — 비권장", "recommended": "false"}
    plans = [plan_s, plan_m, plan_l]
    recommended_plan = "Plan-S (CPU, GPU 불필요)"

    with open(OUT_PLAN, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(plan_s.keys()))
        w.writeheader()
        w.writerows(plans)

    # ---- feature extractor status ----
    feature_extractor_status = {
        "backbone": "resnet18 + ImageNet (ResNet18_Weights.IMAGENET1K_V1) — baseline 허용",
        "feature_layers": "layer1(64)+layer2(128)+layer3(256) concat = 448",
        "feature_dim_raw_concat": FEATURE_DIM,
        "padim_internal_reduced_dim": 100,
        "patch_method": "slice 전체 forward 후 patch center 좌표 feature-map indexing (per-patch CNN 아님)",
        "patch_coords_format": "(y0, x0, y1, x1), 32x32 patch",
        "input": "preprocess_ct_slice → (3,H,W) float32, slice (512,512)",
        "preprocessing": "HU window clip [hu_min=-1000, hu_max=200] (default) → 3채널 → ImageNet mean/std",
        "hu_window_confirm_needed": "★ 확인 필요: v2 scoring 이 default(-1000/200)를 썼는지 별도 hu 값을 썼는지 — smoke 해석 전 일치 확인 권장",
        "device": "cuda if available else cpu → CPU 실행 가능",
        "reproducible": "yes (동일 FeatureExtractor/preprocess_ct_slice 재사용)",
        "module_paths": ["src/position_aware_padim/feature_extractor.py",
                         "src/position_aware_padim/preprocessing.py"],
    }

    estimated_memory_usage = {
        "feature_matrix_plan_s_mb": fmt_mb(500),
        "feature_matrix_plan_m_mb": fmt_mb(1500),
        "candidate_features_kb": round(6 * FEATURE_DIM * 4 / 1024, 2),
        "distance_matrix": "candidates(6) × memory(cap) 448-dim Euclidean → 무시 가능(<0.1s)",
        "peak_ram_note": "ResNet18 파라미터(~45MB) + 512x512x3 float32 slice(~3MB) + feature map. 총 RAM < 1GB, OOM 위험 없음",
    }
    estimated_runtime_notes = {
        "forward_unit": "ResNet18 512x512 1회 forward: CPU ~0.3~1.0s (추정)",
        "plan_s_forwards": "memory ~3환자×5slice=15 + candidate 6 = ~21회",
        "plan_s_total_estimate": "~1~2분 (CPU, 모델로드 포함) — 추정",
        "plan_m_total_estimate": "~2~4분 (CPU) — 추정",
        "bottleneck": ["CT slice load(/mnt/c mmap)", "ResNet18 forward(CPU 지배적)",
                       "feature matrix/distance는 무시 가능"],
        "gpu_needed": "minimal smoke 규모에선 CPU로 충분 → GPU 불필요(권장). GPU는 승인 시에만, 속도만 향상.",
    }

    safety_abort_conditions = [
        "stage2_holdout 접근 감지",
        "candidates row 수 6 초과",
        "memory patient 수가 승인 범위 초과",
        "memory patch cap 초과",
        "lesion patient 가 memory pool 에 들어감(normal root 외 경로)",
        "score 파일 write 시도",
        "adjusted_score/suppression_weight/refined_score 생성 시도",
        "output folder 이미 존재(exist_ok=False)",
        "GPU 미승인 상태에서 CUDA 사용 감지(--confirm-gpu 없는 device=cuda 차단)",
        "NaN/Inf feature 발생",
        "feature dimension mismatch(≠448)",
        "distance 결과 NaN/Inf",
    ]

    b1d3d1_output_schema = {
        "folder": "b1d3d1_gate_p2_minimal_feature_smoke_v1/ (exist_ok=False)",
        "files": [
            "b1d3d1_gate_p2_minimal_feature_smoke_memory_preview.csv",
            "b1d3d1_gate_p2_minimal_feature_smoke_candidate_distances.csv",
            "b1d3d1_gate_p2_minimal_feature_smoke_summary.json",
            "b1d3d1_gate_p2_minimal_feature_smoke_report.md",
        ],
        "candidate_distances_columns": [
            "gate_candidate_id", "review_id", "patient_id", "candidate_score",
            "feature_status", "nearest_distance", "nearest_memory_patient",
            "distance_rank_or_percentile", "gate_p2_flag(normal_like/uncertain/suspicious)",
            "flag_reason", "score_modified(=false)", "safety_note",
        ],
    }

    # ---- 판정 ----
    script_safety_pass = (py_compile_pass and bare_run_exit_2 and dry_run_pass and dry_no_folder)
    inputs_pass = (len(cands) == 6 and len(pool) == 6 and stage2_holdout_access == 0)
    verdict = "PASS" if (inputs_pass and script_safety_pass) else "NEEDS_FIX"

    summary = {
        "step": "B1-D3d0_Gate_P2_minimal_feature_smoke_approval_preflight",
        "verdict": verdict,
        "input_mtimes": input_mtimes,
        "stage2_holdout_access": stage2_holdout_access,
        "gate_candidate_rows": len(cands),
        "memory_pool_preview_rows": len(pool),
        "recommended_plan": recommended_plan,
        "plan_s": plan_s, "plan_m": plan_m, "plan_l": plan_l,
        "feature_extractor_status": feature_extractor_status,
        "estimated_memory_usage": estimated_memory_usage,
        "estimated_runtime_notes": estimated_runtime_notes,
        "safety_abort_conditions": safety_abort_conditions,
        "b1d3d1_output_schema": b1d3d1_output_schema,
        "script_created": True,
        "script_path": str(SMOKE),
        "py_compile_pass": py_compile_pass,
        "bare_run_exit_2": bare_run_exit_2,
        "dry_run_pass": dry_run_pass,
        "dry_run_result": dry_result,
        "dry_run_no_output_folder": dry_no_folder,
        "gpu_used": False,
        "feature_extracted": False,
        "memory_bank_created": False,
        "nearest_neighbor_computed": False,
        "score_modified": False,
        "minimal_feature_smoke_requires_user_approval": True,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- report MD ----
    plan_tbl = "\n".join(
        f"| {p['plan']} | {p['memory_patients']} | {p['memory_patch_cap']} | {p['gate_candidates']} | "
        f"{p['feature_matrix_mb']} MB | {p['recommended']} | {p['purpose']} |" for p in plans)

    md = f"""# B1-D3d0 Gate-P2 Minimal Feature Smoke — Approval Preflight

minimal feature smoke 실행(B1-D3d1) 전 **승인 전 preflight**.
feature 추출/GPU/CUDA/memory bank/NN/distance 일절 없음. smoke 스크립트는 차단 상태로 안전가드만 검증.

## 0. 판정
**{verdict}**

## 1. B1-D3c 결과 요약
- gate candidate 6(GC001~006, 전부 AD_wall_med D_keep, CT/mask/shape 유효), memory pool 6명 usable.
- B1-D3c: gpu_used/feature_extracted/memory_bank_created/nearest_neighbor_computed/score_modified 전부 False, holdout 0.

## 2. 왜 B1-D3d0 이 필요한가
실제 feature 추출(B1-D3d1)은 backbone forward 가 들어가므로, **범위·비용·중단조건을 먼저 잠그고** 승인 여부를 결정한다.
이번 단계는 실행 0(스크립트 차단 검증만).

## 3. Plan-S/M/L 비교
| plan | memory_patients | memory_patch_cap | gate_cand | feature_matrix | recommended | 목적 |
|---|---|---|---|---|---|---|
{plan_tbl}

- **추천: {recommended_plan}** — minimal smoke 규모는 feature matrix < 1MB, CPU 로 충분. GPU 불필요.

## 4. Feature extractor 확인 (read-only, 실행 0)
- backbone: {feature_extractor_status['backbone']}
- feature: {feature_extractor_status['feature_layers']} (raw concat **{FEATURE_DIM}**, PaDiM 내부는 100차원 random 축소)
- patch: {feature_extractor_status['patch_method']}, 좌표 {feature_extractor_status['patch_coords_format']}
- preprocessing: {feature_extractor_status['preprocessing']}
- device: {feature_extractor_status['device']}
- 재현성: {feature_extractor_status['reproducible']}
- **{feature_extractor_status['hu_window_confirm_needed']}**

## 5. 메모리/시간 추정 (실행 없이)
- feature matrix: Plan-S {estimated_memory_usage['feature_matrix_plan_s_mb']}MB / Plan-M {estimated_memory_usage['feature_matrix_plan_m_mb']}MB, candidate {estimated_memory_usage['candidate_features_kb']}KB
- distance: {estimated_memory_usage['distance_matrix']}
- peak RAM: {estimated_memory_usage['peak_ram_note']}
- forward: {estimated_runtime_notes['forward_unit']}
- Plan-S forwards {estimated_runtime_notes['plan_s_forwards']} → **{estimated_runtime_notes['plan_s_total_estimate']}**, Plan-M {estimated_runtime_notes['plan_m_total_estimate']}
- 병목: {', '.join(estimated_runtime_notes['bottleneck'])}
- **GPU 필요성: {estimated_runtime_notes['gpu_needed']}**

## 6. 안전 중단 조건 (B1-D3d1 실행 시)
{chr(10).join('- ' + s for s in safety_abort_conditions)}

## 7. B1-D3d1 출력 스키마
- 폴더: {b1d3d1_output_schema['folder']}
- 파일: {', '.join(b1d3d1_output_schema['files'])}
- candidate_distances 컬럼: {', '.join(b1d3d1_output_schema['candidate_distances_columns'])}

## 8. smoke 스크립트 안전가드 검증 결과
- script_created: True (`{SMOKE.name}`)
- py_compile_pass: **{py_compile_pass}**
- bare_run_exit_2: **{bare_run_exit_2}**
- dry_run_pass: **{dry_run_pass}** (feature_extracted=False, files_created=0, gpu_used=False)
- dry_run_no_output_folder: **{dry_no_folder}**
- 가드: ALLOW_REAL_PROCESSING=False / --real 은 --confirm-feature-smoke 필요 / device=cuda 는 --confirm-gpu 필요 / output exist_ok=False

## 9. ★ 사용자 승인 필요 문구
minimal_feature_smoke_requires_user_approval = **True**.
B1-D3d1 real feature smoke 실행은 **사용자 승인 후에만** 진행한다.
- 권장: **Plan-S, device=cpu (GPU 불필요 → 과금 없음)**.
- GPU(device=cuda) 사용을 원하면 별도 `--confirm-gpu` 승인 + 과금 검토가 필요(이번 규모엔 불필요).
- 이 smoke 는 **성능 개선 실험이 아니라** distance/flag 분리 동작 확인용 preview 다.

## 10. 다음 단계
**B1-D3d1 Gate-P2 minimal feature smoke 실행 승인/보류 결정.**
- 승인 시: `--real --confirm-feature-smoke --device cpu --memory-patient-limit 3 --memory-patch-cap 500`(Plan-S) + ALLOW_REAL_PROCESSING 런타임 override.

---
gpu_used=False, feature_extracted=False, memory_bank_created=False, nearest_neighbor_computed=False, score_modified=False, stage2_holdout_access={stage2_holdout_access}
"""
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)

    # ---- 콘솔 ----
    print(f"[B1-D3d0] {verdict}")
    print(f"  py_compile={py_compile_pass}, bare_run_exit_2={bare_run_exit_2}, dry_run_pass={dry_run_pass}, no_folder={dry_no_folder}")
    print(f"  recommended_plan={recommended_plan}")
    print(f"  생성: {OUT_MD.name}, {OUT_JSON.name}, {OUT_PLAN.name}")


if __name__ == "__main__":
    main()
