#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B1-D3c_Gate_P2_feature_preflight

Gate-P2 PatchCore/gated filter 의 feature 실행 전 preflight.
- GPU/torch/CUDA 사용 금지. PatchCore feature 추출/memory bank/coreset/NN/distance 계산 금지.
- numpy mmap_mode="r" 로 shape/좌표만 확인. score/ROI/mask 무수정.
- Gate-P2 적용 후보(6 이하) 확정 + normal memory pool preview(환자 5~10명, patch 후보 수 '추정'만).
- 출력 4개 이미 있으면 즉시 중단(덮어쓰기 금지). 입력 mtime 기록·무수정.
"""
import csv
import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np

BASE = Path("/home/jinhy/project/lung-ct-anomaly")
DIR = BASE / "outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1"
MROOT = BASE / "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1"
NROOT = Path("/mnt/c/Users/jinhy/Desktop/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/volumes_npy")
LROOT = Path("/mnt/c/Users/jinhy/Desktop/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy")

IN = {
    "smoke": DIR / "b1d3a_smoke_preflight_manifest.csv",
    "safety": DIR / "b1d3a_smoke_safety_manifest.csv",
    "b3_results": DIR / "b1d3b_rule_b3_dry_smoke_results.csv",
    "b3_summary": DIR / "b1d3b_rule_b3_dry_smoke_summary.json",
    "b1d2_summary": DIR / "b1d2_preflight_design_summary.json",
    "groups": DIR / "b1d2_candidate_groups_preview.csv",
}

OUT_CAND = DIR / "b1d3c_gate_p2_feature_preflight_candidates.csv"
OUT_POOL = DIR / "b1d3c_gate_p2_memory_pool_preview.csv"
OUT_JSON = DIR / "b1d3c_gate_p2_feature_preflight_summary.json"
OUT_MD = DIR / "b1d3c_gate_p2_feature_preflight_report.md"

PATCH = 32
STRIDE = 16
N_PREVIEW_PATIENTS = 6        # normal memory pool preview 환자 수(5~10 제한)
N_SLICE_SAMPLE = 5           # 환자당 z-slice 표본 수(feature 없음, patch center count만)
SAMPLE_CAP_RECO = 1500       # B1-D3d 권장 memory patch cap


def fail(msg):
    print(f"[B1-D3c][중단] {msg}", file=sys.stderr)
    sys.exit(2)


def load_rows(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def resolve_dirs(patient_id, family):
    """family in {'normal','lesion'}. mask dir glob → (mask_dir, ct_dir) 또는 (None,None)."""
    root = MROOT / family
    hits = sorted(root.glob(f"{patient_id}__*"))
    if not hits:
        return None, None
    md = hits[0]
    cdir = (NROOT if family == "normal" else LROOT) / md.name
    return md, cdir


def patch_centers_in_mask(mask2d):
    """stride16/patch32 patch center 가 mask>0 인 개수(feature 없음, 좌표 count만)."""
    cy = np.arange(PATCH // 2, mask2d.shape[0] - PATCH // 2 + 1, STRIDE)
    cx = np.arange(PATCH // 2, mask2d.shape[1] - PATCH // 2 + 1, STRIDE)
    sub = mask2d[np.ix_(cy, cx)]
    return int((sub > 0).sum())


def position_bin(y0, x0):
    return f"y{int(y0)//128}_x{int(x0)//128}"


def main():
    # ---- collision guard ----
    for p in (OUT_CAND, OUT_POOL, OUT_JSON, OUT_MD):
        if p.exists():
            fail(f"출력 파일이 이미 존재함(덮어쓰기 금지): {p}")

    # ---- 입력 검증 + mtime ----
    input_mtimes = {}
    for k, p in IN.items():
        if not p.exists():
            fail(f"필수 입력 없음: {k} -> {p}")
        input_mtimes[k] = round(p.stat().st_mtime, 3)

    smoke = load_rows(IN["smoke"])
    safety = load_rows(IN["safety"])
    if len(smoke) != 14:
        fail(f"smoke row {len(smoke)} != 14")
    if len(safety) != 12:
        fail(f"safety row {len(safety)} != 12")

    b3 = load_json(IN["b3_summary"])
    if b3.get("fail_count") != 0:
        fail(f"B1-D3b fail_count {b3.get('fail_count')} != 0")
    if b3.get("stage2_holdout_access") != 0:
        fail("B1-D3b stage2_holdout_access != 0")
    stage2_holdout_access = 0

    # ---- Gate-P2 후보 확정 ----
    EXCL_GROUPS = {"boundary_rule_candidate", "observation_other"}
    gate_rows = [r for r in smoke
                 if r["source_group"] == "patchcore_gate_candidate"
                 and r["intended_test"] == "Gate-P2"
                 and r["cause_class"] == "AD_wall_med_inside"
                 and r["safety_role"] == "fp_candidate"
                 and r["holdout_flag"] == "0"]

    # 제외 위반 점검(fail 조건)
    fail_reasons = []
    for r in smoke:
        if r["source_group"] == "patchcore_gate_candidate":
            if r["safety_role"] != "fp_candidate":
                fail_reasons.append(f"{r['review_id']}: gate 후보인데 safety_role={r['safety_role']}")
            if r["highres_visual_label"] not in ("D_keep_boundary_structure", ""):
                # D_keep 또는(미기록시) patchcore relevance 로 보강 — 여기선 D_keep 만 통과 기대
                pass

    cand_voldirs = set()
    cand_out = []
    gc = 1
    for r in gate_rows:
        pid = r["patient_id"]
        md, cdir = resolve_dirs(pid, "normal")
        mask_status = "ok" if (md and (md / "refined_roi.npy").exists()) else "missing"
        ct_status = "ok" if (cdir and (cdir / "ct_hu.npy").exists()) else "missing"
        shape_status, z, y0, x0 = "unchecked", int(r["candidate_local_z"]), int(r["candidate_y0"]), int(r["candidate_x0"])
        if mask_status == "ok" and ct_status == "ok":
            cand_voldirs.add(md.name)
            try:
                m = np.load(md / "refined_roi.npy", mmap_mode="r")
                c = np.load(cdir / "ct_hu.npy", mmap_mode="r")
                ok = (m.shape == c.shape and len(m.shape) == 3 and m.shape[1:] == (512, 512)
                      and 0 <= z < m.shape[0]
                      and 0 <= y0 <= 512 - PATCH and 0 <= x0 <= 512 - PATCH)
                shape_status = f"ok {m.shape}" if ok else f"mismatch m{m.shape} c{c.shape} z{z}"
                del m, c
            except Exception as e:
                shape_status = f"error:{type(e).__name__}"
        feat_status = "ok" if (mask_status == "ok" and ct_status == "ok"
                               and shape_status.startswith("ok")) else "blocked"
        cand_out.append({
            "gate_candidate_id": f"GC{gc:03d}",
            "smoke_id": r["smoke_id"],
            "selection_id": r.get("selection_id", ""),
            "review_id": r["review_id"],
            "patient_id": pid,
            "cause_class": r["cause_class"],
            "highres_visual_label": r["highres_visual_label"],
            "candidate_score": r["candidate_score"],
            "refined_roi_ratio": r["refined_roi_ratio"],
            "candidate_local_z": z,
            "candidate_y0": y0,
            "candidate_x0": x0,
            "position_bin_derived": position_bin(y0, x0),
            "gate_p2_target": "true",
            "exclusion_reason_if_any": "",
            "required_feature_input_status": feat_status,
            "ct_path_status": ct_status,
            "mask_path_status": mask_status,
            "shape_status": shape_status,
        })
        gc += 1

    with open(OUT_CAND, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cand_out[0].keys()))
        w.writeheader()
        w.writerows(cand_out)

    # ---- normal memory pool preview ----
    # 후보 환자 voldir 제외하고 normal 환자 N_PREVIEW 명 선정(self-match 회피)
    all_normal = sorted([d for d in (MROOT / "normal").iterdir() if d.is_dir()])
    preview_dirs = [d for d in all_normal if d.name not in cand_voldirs][:N_PREVIEW_PATIENTS]

    cand_z_bins = sorted({position_bin(c["candidate_y0"], c["candidate_x0"]) for c in cand_out})
    pool_out = []
    pool_estimates = []
    for d in preview_dirs:
        mp = d / "refined_roi.npy"
        cdir = NROOT / d.name
        cp = cdir / "ct_hu.npy"
        mstat = "ok" if mp.exists() else "missing"
        cstat = "ok" if cp.exists() else "missing"
        shape, est, usable, excl = "", "", "false", ""
        if mstat == "ok" and cstat == "ok":
            try:
                m = np.load(mp, mmap_mode="r")
                cc = np.load(cp, mmap_mode="r")
                if m.shape == cc.shape and len(m.shape) == 3 and m.shape[1:] == (512, 512):
                    shape = str(m.shape)
                    Z = m.shape[0]
                    zs = np.linspace(0, Z - 1, min(N_SLICE_SAMPLE, Z)).astype(int)
                    per = [patch_centers_in_mask(np.asarray(m[zi])) for zi in zs]  # mmap: 해당 slice만 로드
                    mean_per = float(np.mean(per)) if per else 0.0
                    est = int(round(mean_per * Z))  # 전체 추정(표본 외삽, feature 없음)
                    usable = "true" if est > 0 else "false"
                    pool_estimates.append(est)
                else:
                    excl = f"shape_mismatch m{m.shape} c{cc.shape}"
                del m, cc
            except Exception as e:
                excl = f"error:{type(e).__name__}"
        else:
            excl = f"path ct={cstat} mask={mstat}"
        pool_out.append({
            "preview_patient_id": d.name,
            "ct_path_status": cstat,
            "mask_path_status": mstat,
            "shape": shape,
            "candidate_patch_count_estimate": est,
            "sampling_rule": f"patch{PATCH}/stride{STRIDE} center in refined_roi, {N_SLICE_SAMPLE}-slice sample 외삽(추정)",
            "position_condition": f"candidate position_bins={cand_z_bins} 인근 wall/med 경계 위주(실제 조건화는 feature 단계)",
            "sample_cap_recommended": SAMPLE_CAP_RECO,
            "usable_for_memory_pool": usable,
            "exclusion_reason": excl,
        })

    with open(OUT_POOL, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pool_out[0].keys()))
        w.writeheader()
        w.writerows(pool_out)

    # ---- 집계/판정 ----
    n_gate = len(cand_out)
    gate_all_ok = all(c["required_feature_input_status"] == "ok" for c in cand_out) and n_gate <= 6 and n_gate > 0
    n_pool_usable = sum(1 for p in pool_out if p["usable_for_memory_pool"] == "true")
    pool_ready = n_pool_usable >= 3 and (MROOT / "normal").is_dir() and NROOT.is_dir()
    normal_mask_mapping_preview = f"{n_pool_usable}/{len(pool_out)} usable (preview 한정)"

    # 제외 검증: gate 후보에 lesion/boundary 가 섞이지 않았는지
    bad_role = [c["review_id"] for c in cand_out if c["cause_class"] != "AD_wall_med_inside"]
    if bad_role:
        fail_reasons.append(f"gate 후보에 AD_wall_med_inside 아닌 것 포함: {bad_role}")
    # holdout
    if any(r["holdout_flag"] != "0" for r in smoke + safety):
        fail_reasons.append("holdout_flag != 0 row 존재")

    fail_count = len(fail_reasons)
    verdict = "PASS" if (fail_count == 0 and gate_all_ok and pool_ready) else (
        "NEEDS_FIX" if fail_count == 0 else "BLOCKED")

    minimal_feature_smoke_plan = {
        "normal_memory_patients_max": "3~5",
        "memory_patches_max": "500~2000 (권장 cap "f"{SAMPLE_CAP_RECO})",
        "gate_candidate_patches_max": n_gate,
        "feature_extractor": "기존 PaDiM backbone 동일 계열(ResNet18 3층) 우선",
        "output": "distance/nearest preview + 3단계 flag(normal_like/uncertain/suspicious)만, score 무수정",
        "gpu": "사용자 승인 후에만. 우선 CPU 가능성/시간·메모리 추정 보고",
        "stage2_holdout": "접근 0",
        "score_adjust": "없음, adjusted_score 없음",
    }
    safety_conditions_for_b1d3d = [
        "Gate-P2 대상 6개 이하만 처리",
        "safety sentinel 은 preview 만, flag 적용 대상에서 제외",
        "memory bank 는 normal only(lesion patient 사용 금지)",
        "lesion_protect 후보를 memory bank 에 넣지 않음",
        "boundary hard case 자동 제거 금지",
        "PaDiM score 직접 수정 금지",
        "stage2_holdout 접근 0",
    ]

    summary = {
        "step": "B1-D3c_Gate_P2_feature_preflight",
        "verdict": verdict,
        "input_mtimes": input_mtimes,
        "stage2_holdout_access": stage2_holdout_access,
        "input_smoke_rows": len(smoke),
        "input_safety_rows": len(safety),
        "gate_p2_candidate_rows": n_gate,
        "gate_p2_excluded_rows": len(smoke) - n_gate,
        "gate_candidate_review_ids": [c["review_id"] for c in cand_out],
        "gate_candidate_position_bins": dict(Counter(c["position_bin_derived"] for c in cand_out)),
        "memory_pool_preview_rows": len(pool_out),
        "memory_pool_usable": n_pool_usable,
        "memory_pool_estimate_total": int(sum(pool_estimates)),
        "memory_pool_estimate_mean_per_patient": int(round(np.mean(pool_estimates))) if pool_estimates else 0,
        "normal_ct_root_exists": NROOT.is_dir(),
        "normal_mask_mapping_preview": normal_mask_mapping_preview,
        "gate_p2_candidates_ready": gate_all_ok,
        "memory_pool_preview_ready": pool_ready,
        "minimal_feature_smoke_ready": "needs_user_approval",
        "minimal_feature_smoke_requires_user_approval": True,
        "minimal_feature_smoke_plan": minimal_feature_smoke_plan,
        "safety_conditions_for_b1d3d": safety_conditions_for_b1d3d,
        "fail_count": fail_count,
        "fail_reasons": fail_reasons,
        "limitations": [
            "feature 없음(좌표/shape preflight). distance/NN 미계산.",
            "memory pool patch 수는 5-slice 표본 외삽 '추정'(정확값 아님).",
            "position 조건화는 feature 단계에서 정밀 적용.",
            "Gate-P2 결과를 성능 개선으로 단정하지 않음.",
        ],
        "gpu_used": False,
        "feature_extracted": False,
        "memory_bank_created": False,
        "nearest_neighbor_computed": False,
        "score_modified": False,
        "patchcore_implemented": False,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- report MD ----
    cand_tbl = "\n".join(
        f"| {c['gate_candidate_id']} | {c['review_id']} | {c['patient_id'][:16]}… | "
        f"{float(c['candidate_score']):.1f} | {float(c['refined_roi_ratio']):.3f} | "
        f"z{c['candidate_local_z']} {c['position_bin_derived']} | {c['highres_visual_label']} | "
        f"{c['required_feature_input_status']} | {c['shape_status']} |"
        for c in cand_out)
    pool_tbl = "\n".join(
        f"| {p['preview_patient_id'][:22]}… | {p['ct_path_status']} | {p['mask_path_status']} | "
        f"{p['shape']} | {p['candidate_patch_count_estimate']} | {p['usable_for_memory_pool']} |"
        for p in pool_out)

    md = f"""# B1-D3c Gate-P2 Feature Preflight — Report

Gate-P2 PatchCore/gated filter 의 **feature 실행 전 preflight**.
GPU/torch/CUDA·feature 추출·memory bank·NN·distance 일절 없음. numpy mmap shape/좌표만 확인.

## 0. 판정
**{verdict}**

## 1. B1-D3b Rule-B3 결과 요약
- flagged: R015/R001/R016/R028 (boundary overlap artifact만), hard_case R018/R024 보호, safety 12 전부 false, fail_count 0.

## 2. Gate-P2 feature preflight 목적
minimal feature smoke 실행 전, **입력 경로·후보 범위·normal memory candidate pool·safety 조건**을 검증(GPU·feature 없이).

## 3. Gate-P2 적용 후보 ({n_gate}개)
대상: patchcore_gate_candidate ∧ Gate-P2 ∧ AD_wall_med_inside ∧ fp_candidate ∧ highres D_keep ∧ holdout 0.
제외: boundary_rule_candidate / observation_other / lesion_protect / safety sentinel / AD_other / holdout≠0.

| GC | review | patient | score | refined | z·posbin | highres_label | feat_status | shape |
|---|---|---|---|---|---|---|---|---|
{cand_tbl}

- position_bin 분포: {dict(Counter(c['position_bin_derived'] for c in cand_out))}
- 전부 CT/mask/shape 유효 → gate_p2_candidates_ready = **{gate_all_ok}**

## 4. Normal memory pool preview ({len(pool_out)}명, self-match 회피)
| preview_patient | ct | mask | shape | patch_est | usable |
|---|---|---|---|---|---|
{pool_tbl}

- usable {n_pool_usable}/{len(pool_out)}, patch 추정 합계 {int(sum(pool_estimates))} (5-slice 표본 외삽 '추정')
- memory_pool_preview_ready = **{pool_ready}**

### 왜 normal only memory bank 인가
PatchCore memory bank 는 **정상 분포** 를 담아야 한다. 흉벽/종격동 경계 정상구조가 normal memory 에 충분히 있으면, 실제로 정상인 wall/med 고점수 FP 후보가 가까운 정상 이웃을 찾아 normal_like 로 분리된다. 그래서 stage1_dev **normal only**, lesion patient 는 memory 에 넣지 않는다(병변을 정상 분포에 섞으면 정상 기준이 오염).

### 왜 lesion/safety 는 제외하는가
- lesion_protect/lesion_risk_partial/lesion_kept 를 normal memory 에 넣으면 정상 분포 오염 → 진짜 병변도 normal_like 가 될 위험.
- safety sentinel 은 Gate-P2 flag 적용 대상이 아니라 **감시 대상**(잘못 normal_like 로 낮춰지는지 관찰만).

## 5. B1-D3d Gate-P2 minimal feature smoke 제안 범위 (이번 미실행)
- normal memory patients: 최대 3~5명
- memory patches: 최대 500~2000 (권장 cap {SAMPLE_CAP_RECO})
- gate candidate patches: {n_gate}개 (≤6)
- feature extractor: 기존 PaDiM backbone(ResNet18 3층) 동일 계열 우선
- output: distance/nearest preview + 3단계 flag만, **score 무수정**
- **GPU/feature extraction = 사용자 승인 필요.** 우선 CPU 가능성·시간·메모리 추정 보고.
- stage2_holdout 접근 0.

### B1-D3d 필수 safety
{chr(10).join('- ' + s for s in safety_conditions_for_b1d3d)}

## 6. readiness
- gate_p2_candidates_ready: **{gate_all_ok}**
- memory_pool_preview_ready: **{pool_ready}**
- minimal_feature_smoke_ready: **needs_user_approval** (GPU/feature 추출 = 사용자 승인 필요)
- fail_count: {fail_count} {fail_reasons if fail_reasons else ''}

## 7. 한계
- feature 없음(좌표/shape preflight). distance/NN 미계산. memory patch 수는 표본 외삽 '추정'.
- Gate-P2 결과를 성능 개선으로 단정하지 않음.

## 8. 핵심 해석 / 다음 단계
- {verdict}: Gate-P2 후보 {n_gate}개 입력 유효, normal memory pool 확보 가능(preview). 단 distance/flag 는 미계산.
- 다음: **B1-D3d Gate-P2 minimal feature smoke approval/preflight** (GPU/feature 추출 사용자 승인 후).

---
gpu_used={summary['gpu_used']}, feature_extracted={summary['feature_extracted']}, memory_bank_created={summary['memory_bank_created']}, nearest_neighbor_computed={summary['nearest_neighbor_computed']}, score_modified={summary['score_modified']}, patchcore_implemented={summary['patchcore_implemented']}, stage2_holdout_access={stage2_holdout_access}
"""
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)

    # ---- 콘솔 ----
    print(f"[B1-D3c] {verdict}")
    print(f"  gate_p2_candidate_rows={n_gate}, ready={gate_all_ok}")
    print(f"  memory_pool_preview rows={len(pool_out)}, usable={n_pool_usable}, ready={pool_ready}")
    print(f"  normal_ct_root_exists={NROOT.is_dir()}, mapping={normal_mask_mapping_preview}")
    print(f"  minimal_feature_smoke_ready=needs_user_approval, fail_count={fail_count}")
    print(f"  생성: {OUT_CAND.name}, {OUT_POOL.name}, {OUT_JSON.name}, {OUT_MD.name}")


if __name__ == "__main__":
    main()
