#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B1-D3a_small_scope_smoke_test_preflight

B1-D2 preflight 결과를 read-only 로 받아, boundary rule(Rule-B3) / PatchCore gate(Gate-P2)
small-scope smoke test 의 입력/대상/출력/안전조건을 잠근다.

- PatchCore feature 추출 / memory bank 생성 / NN 계산 / score 조정 / boundary rule 실제 적용 없음.
- candidate subset 선정 + smoke manifest + safety manifest + 출력/ablation 스키마 설계만.
- 경로/shape/좌표 존재 여부만 read-only 확인(feature/model 실행 없음).
- 숫자 하드코딩 금지(CSV/JSON 재집계), highres 라벨 우선.
- 출력 4개 이미 있으면 즉시 중단(덮어쓰기 금지). 입력 mtime 기록·무수정.
- stage1_dev only. stage2_holdout 접근 0.
"""
import csv
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

BASE = Path("/home/jinhy/project/lung-ct-anomaly")
DIR = BASE / "outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1"

IN = {
    "b1d2_summary": DIR / "b1d2_preflight_design_summary.json",
    "groups_csv": DIR / "b1d2_candidate_groups_preview.csv",
    "safety_csv": DIR / "b1d2_safety_set_preview.csv",
    "cause_csv": DIR / "b1d1_fp_cause_diagnostic.csv",
    "hr_labels": DIR / "b1d1_highres_visual_recheck_labels.csv",
}

# Gate-P2 memory bank 소스 / mask root (read-only 존재 확인용)
NROOT = Path("/mnt/c/Users/jinhy/Desktop/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/volumes_npy")
LROOT = Path("/mnt/c/Users/jinhy/Desktop/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy")
MROOT = BASE / "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1"

OUT_MANIFEST = DIR / "b1d3a_smoke_preflight_manifest.csv"
OUT_SAFETY = DIR / "b1d3a_smoke_safety_manifest.csv"
OUT_JSON = DIR / "b1d3a_smoke_preflight_summary.json"
OUT_MD = DIR / "b1d3a_smoke_preflight_report.md"

GATE_TARGET_N = 6      # PatchCore gate candidate (D_keep, patient cap 2)
BOUND_TARGET_N = 6     # boundary rule candidate (overlap 4 + hard 2)
OBS_TARGET_N = 2       # observation_other (AD_other)
LESION_KEPT_SENTINEL_N = 3  # lesion_kept 고점수 sentinel
PATIENT_CAP = 2


def fail(msg):
    print(f"[B1-D3a][중단] {msg}", file=sys.stderr)
    sys.exit(2)


def load_rows(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def sc(r):
    return float(r["candidate_score"])


def main():
    # ---- collision guard ----
    for p in (OUT_MANIFEST, OUT_SAFETY, OUT_JSON, OUT_MD):
        if p.exists():
            fail(f"출력 파일이 이미 존재함(덮어쓰기 금지): {p}")

    # ---- 입력 검증 + mtime ----
    input_mtimes = {}
    for k, p in IN.items():
        if not p.exists():
            fail(f"필수 입력 없음: {k} -> {p}")
        input_mtimes[k] = round(p.stat().st_mtime, 3)

    b1d2 = load_json(IN["b1d2_summary"])
    if b1d2.get("stage2_holdout_access") != 0:
        fail("b1d2 stage2_holdout_access != 0")
    stage2_holdout_access = 0

    groups = load_rows(IN["groups_csv"])
    safety = load_rows(IN["safety_csv"])
    if len(groups) != 30:
        fail(f"candidate_groups_preview row {len(groups)} != 30")
    if len(safety) != 26:
        fail(f"safety_set_preview row {len(safety)} != 26")

    cause = {r["review_id"]: r for r in load_rows(IN["cause_csv"])}
    hr = {r["review_id"]: r for r in load_rows(IN["hr_labels"])}

    # ---- 경로 readiness (read-only, feature 실행 없음) ----
    path_readiness = {
        "normal_ct_root": {"path": str(NROOT), "exists": NROOT.is_dir(),
                           "n_voldir": len(list(NROOT.iterdir())) if NROOT.is_dir() else 0},
        "lesion_ct_root": {"path": str(LROOT), "exists": LROOT.is_dir(),
                           "n_voldir": len(list(LROOT.iterdir())) if LROOT.is_dir() else 0},
        "mask_normal_root": {"path": str(MROOT / "normal"), "exists": (MROOT / "normal").is_dir(),
                             "n": len(list((MROOT / "normal").iterdir())) if (MROOT / "normal").is_dir() else 0},
        "mask_lesion_root": {"path": str(MROOT / "lesion"), "exists": (MROOT / "lesion").is_dir(),
                             "n": len(list((MROOT / "lesion").iterdir())) if (MROOT / "lesion").is_dir() else 0},
    }
    if not path_readiness["normal_ct_root"]["exists"]:
        fail("Gate-P2 memory bank 소스(normal CT root) 없음")

    # ---- helper: previous/highres label split ----
    def prev_hr_label(rid, best_label, best_source):
        if rid in hr:
            return hr[rid]["previous_visual_label"], hr[rid]["highres_visual_label"]
        # highres 재검토 안 된 row: previous=overlay best(or not_reviewed), highres=n/a
        return (best_label if best_source != "not_reviewed" else "not_reviewed"), ""

    # ---- smoke subset 선정 ----
    by_group = defaultdict(list)
    for r in groups:
        by_group[r["b1d2_group"]].append(r)

    # A. PatchCore gate: subtype==D_keep, -score, patient cap 2, target 6
    gate_pool = [r for r in by_group["patchcore_gate_candidate"] if r["b1d2_subtype"] == "D_keep"]
    gate_pool.sort(key=lambda r: -sc(r))
    gate_sel, gcap = [], defaultdict(int)
    for r in gate_pool:
        if gcap[r["patient_id"]] >= PATIENT_CAP:
            continue
        gate_sel.append(r)
        gcap[r["patient_id"]] += 1
        if len(gate_sel) >= GATE_TARGET_N:
            break

    # B. boundary rule: overlap_artifact 전부(최대4) + hard_case 전부(2), target 6
    bound_overlap = sorted([r for r in by_group["boundary_rule_candidate"]
                            if r["b1d2_subtype"] == "overlap_artifact"], key=lambda r: -sc(r))
    bound_hard = sorted([r for r in by_group["boundary_rule_candidate"]
                         if r["b1d2_subtype"] == "hard_case"], key=lambda r: -sc(r))
    bound_sel = bound_overlap[:4] + bound_hard[:2]
    bound_sel = bound_sel[:BOUND_TARGET_N]

    # C. observation_other: AD_other 고점수 1~2개
    obs_pool = sorted(by_group["excluded_observation"], key=lambda r: -sc(r))
    obs_sel, ocap = [], defaultdict(int)
    for r in obs_pool:
        if ocap[r["patient_id"]] >= 1:
            continue
        obs_sel.append(r)
        ocap[r["patient_id"]] += 1
        if len(obs_sel) >= OBS_TARGET_N:
            break

    # ---- smoke manifest 작성 ----
    EXPECT = {
        "patchcore_gate_candidate": "Gate-P2가 normal_like/uncertain으로 분류 기대(정상 경계구조). suspicious면 재검토. score 무수정.",
        "boundary_overlap": "Rule-B3가 boundary_flag_candidate=true로 표시 기대(경계걸침 artifact). score 무수정.",
        "boundary_hard": "Rule-B3가 boundary_flag_candidate=false로 보호 기대(hard_case 제외). true면 FAIL.",
        "observation_other": "어떤 rule/gate도 적용 안 함(observe only). gate/rule 범위 포함 시 설계 오류.",
    }
    INTENDED = {
        "patchcore_gate_candidate": "Gate-P2",
        "boundary_rule_candidate": "Rule-B3",
        "observation_other": "observe_only",
    }
    smoke_cols = ["smoke_id", "source_group", "selection_id", "review_id", "patient_id",
                  "safety_role", "human_label", "cause_class", "previous_visual_label",
                  "highres_visual_label", "candidate_score", "roi_0_0_patch_ratio",
                  "refined_roi_ratio", "candidate_local_z", "candidate_y0", "candidate_x0",
                  "intended_test", "expected_behavior", "exclusion_reason_if_any",
                  "stage_split_check", "holdout_flag"]

    smoke_rows = []
    sid = 1

    def add_smoke(r, source_group, intended, expect, exclusion=""):
        nonlocal sid
        rid = r["review_id"]
        prev_l, hr_l = prev_hr_label(rid, r.get("best_visual_label", ""), r.get("best_label_source", ""))
        smoke_rows.append({
            "smoke_id": f"SMK{sid:03d}",
            "source_group": source_group,
            "selection_id": r.get("selection_id", ""),
            "review_id": rid,
            "patient_id": r["patient_id"],
            "safety_role": r["safety_role"],
            "human_label": r["human_label"],
            "cause_class": r["cause_class"],
            "previous_visual_label": prev_l,
            "highres_visual_label": hr_l,
            "candidate_score": r["candidate_score"],
            "roi_0_0_patch_ratio": r["roi_0_0_patch_ratio"],
            "refined_roi_ratio": r["refined_roi_ratio"],
            "candidate_local_z": r["candidate_local_z"],
            "candidate_y0": r["candidate_y0"],
            "candidate_x0": r["candidate_x0"],
            "intended_test": intended,
            "expected_behavior": expect,
            "exclusion_reason_if_any": exclusion,
            "stage_split_check": "stage1_dev",
            "holdout_flag": "0",
        })
        sid += 1

    for r in gate_sel:
        add_smoke(r, "patchcore_gate_candidate", INTENDED["patchcore_gate_candidate"],
                  EXPECT["patchcore_gate_candidate"])
    for r in bound_sel:
        if r["b1d2_subtype"] == "hard_case":
            add_smoke(r, "boundary_rule_candidate", INTENDED["boundary_rule_candidate"],
                      EXPECT["boundary_hard"], exclusion="hard_case=Rule-B3 제외대상(보호 검증)")
        else:
            add_smoke(r, "boundary_rule_candidate", INTENDED["boundary_rule_candidate"],
                      EXPECT["boundary_overlap"])
    for r in obs_sel:
        add_smoke(r, "observation_other", INTENDED["observation_other"],
                  EXPECT["observation_other"], exclusion="AD_other=gate/rule 비대상, 관찰만")

    with open(OUT_MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=smoke_cols)
        w.writeheader()
        w.writerows(smoke_rows)

    # ---- safety subset 선정 ----
    by_sentinel = defaultdict(list)
    for r in safety:
        by_sentinel[r["sentinel_type"]].append(r)

    lrp = by_sentinel["lesion_risk_partial_at_risk"]            # 7 전부
    hard = by_sentinel["boundary_hard_case_must_keep"]          # 2 전부
    kept = sorted(by_sentinel["lesion_kept_must_not_degrade"],
                  key=lambda r: -sc(r))[:LESION_KEPT_SENTINEL_N]  # 고점수 3

    SAFE_Q = {
        "lesion_risk_partial": "이 병변 부분커버 후보가 Rule-B3 boundary flag 또는 Gate-P2 normal_like로 잘못 처리되는가?",
        "lesion_kept": "이 보존 병변(고점수)이 Gate-P2 normal_like 또는 Rule-B3 flag로 억제될 위험이 보이는가?",
        "boundary_hard_case": "이 hard_case가 Rule-B3에서 boundary artifact처럼 제거 표시되는가?",
    }
    SAFE_FAIL = {
        "lesion_risk_partial": "boundary_flag_candidate=true 또는 patchcore_gate_flag=normal_like",
        "lesion_kept": "patchcore_gate_flag=normal_like 또는 boundary_flag_candidate=true",
        "boundary_hard_case": "boundary_flag_candidate=true",
    }
    SAFE_MNF = {
        "lesion_risk_partial": "both",
        "lesion_kept": "both",
        "boundary_hard_case": "Rule-B3",
    }
    safety_cols = ["safety_id", "safety_type", "selection_id", "review_id", "patient_id",
                   "candidate_score", "refined_roi_ratio", "candidate_local_z",
                   "candidate_y0", "candidate_x0", "safety_question", "fail_condition",
                   "must_not_flag_by", "holdout_flag"]

    def add_safety(rows_in, stype, out, counter):
        for r in rows_in:
            rid = r["review_id"]
            cz = cause.get(rid, {})
            out.append({
                "safety_id": f"SAF{counter[0]:03d}",
                "safety_type": stype,
                "selection_id": r.get("selection_id", ""),
                "review_id": rid,
                "patient_id": r["patient_id"],
                "candidate_score": r["candidate_score"],
                "refined_roi_ratio": r["refined_roi_ratio"],
                "candidate_local_z": cz.get("candidate_local_z", ""),
                "candidate_y0": cz.get("candidate_y0", ""),
                "candidate_x0": cz.get("candidate_x0", ""),
                "safety_question": SAFE_Q[stype],
                "fail_condition": SAFE_FAIL[stype],
                "must_not_flag_by": SAFE_MNF[stype],
                "holdout_flag": "0",
            })
            counter[0] += 1

    safety_out, ctr = [], [1]
    add_safety(lrp, "lesion_risk_partial", safety_out, ctr)
    add_safety(hard, "boundary_hard_case", safety_out, ctr)
    add_safety(kept, "lesion_kept", safety_out, ctr)

    with open(OUT_SAFETY, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=safety_cols)
        w.writeheader()
        w.writerows(safety_out)

    # ---- 집계 ----
    smoke_by_group = dict(Counter(r["source_group"] for r in smoke_rows))
    safety_by_type = dict(Counter(r["safety_type"] for r in safety_out))
    smoke_patients = sorted(set(r["patient_id"] for r in smoke_rows))
    safety_patients = sorted(set(r["patient_id"] for r in safety_out))
    all_patients = sorted(set(smoke_patients) | set(safety_patients))

    # holdout 전수 0 확인 (smoke + safety)
    if any(r["holdout_flag"] != "0" for r in smoke_rows + safety_out):
        fail("holdout_flag != 0 인 row 존재")

    # ---- readiness ----
    rule_b3_cols = ["cause_class", "human_label", "refined_roi_ratio", "center_in_refined_roi",
                    "best_visual_label", "safety_role"]
    rule_b3_ready = all(c in groups[0] for c in rule_b3_cols)
    gate_p2_feature_preflight_ready = path_readiness["normal_ct_root"]["exists"] and len(gate_sel) > 0
    gate_p2_minimal_smoke_ready = False  # GPU feature 추출 = 사용자 승인 필요(과금규칙)

    recommended_b1d3b_order = [
        "1) Rule-B3 dry smoke (feature 없음, rule flag만, safety sentinel flag 여부 확인, score 무수정) — 최우선",
        "2) Gate-P2 feature preflight (경로/대상/normal memory candidate 수만 확인, distance 계산 없음)",
        "3) Gate-P2 minimal feature smoke (★별도 승인 필요: GPU feature 추출, normal sample cap 매우 작게, gate 소수, flag/distance preview만, score 무수정)",
    ]

    safety_fail_conditions = [
        "safety sentinel이 Rule-B3에 의해 제거/flag됨",
        "lesion_risk_partial이 Gate-P2에서 normal_like로 잘못 낮춰질 위험이 보임",
        "hard_case가 boundary artifact처럼 처리됨",
        "stage2_holdout 접근 발생",
        "score 원본이 수정됨",
        "adjusted score가 의도치 않게 생성됨",
        "ablation 출력이 섞임",
    ]

    memory_bank_design = {
        "source": "stage1_dev normal only (Normal_LUNA16 ... volumes_npy)",
        "patch_scope": "wall/mediastinum boundary-like normal patches",
        "forbidden": "전체 patch 사용 금지, lesion patient 사용 금지, stage2_holdout 접근 0",
        "conditioning": "position-conditioned 또는 position-bin-aware",
        "size_control": "coreset 또는 sample cap 필요(minimal smoke 는 cap 매우 작게)",
        "this_step": "memory bank 생성 안 함. 경로/대상 조건만 확인.",
        "normal_ct_root_n_voldir": path_readiness["normal_ct_root"]["n_voldir"],
    }

    summary = {
        "step": "B1-D3a_small_scope_smoke_test_preflight",
        "verdict": "PASS",
        "input_mtimes": input_mtimes,
        "stage2_holdout_access": stage2_holdout_access,
        "n_candidate_input": len(groups),
        "n_safety_input": len(safety),
        "smoke_manifest_rows": len(smoke_rows),
        "safety_manifest_rows": len(safety_out),
        "smoke_by_source_group": smoke_by_group,
        "safety_by_type": safety_by_type,
        "selected_patients": {
            "smoke": smoke_patients,
            "safety": safety_patients,
            "union_n": len(all_patients),
        },
        "path_readiness": path_readiness,
        "memory_bank_design": memory_bank_design,
        "recommended_b1d3b_order": recommended_b1d3b_order,
        "safety_fail_conditions": safety_fail_conditions,
        "rule_b3_ready": rule_b3_ready,
        "gate_p2_feature_preflight_ready": gate_p2_feature_preflight_ready,
        "gate_p2_minimal_smoke_ready": gate_p2_minimal_smoke_ready,
        "gate_p2_minimal_smoke_blocker": "GPU feature 추출 = 사용자 승인 필요(과금 방지 규칙)",
        "patchcore_implemented": False,
        "boundary_rule_implemented": False,
        "score_modified": False,
        "roi_modified": False,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- report MD ----
    smoke_tbl = "\n".join(
        f"| {r['smoke_id']} | {r['source_group']} | {r['review_id']} | {r['patient_id'][:18]}… | "
        f"{r['cause_class']} | {r['previous_visual_label']}/{r['highres_visual_label'] or '-'} | "
        f"{float(r['candidate_score']):.1f} | {float(r['refined_roi_ratio']):.3f} | {r['intended_test']} |"
        for r in smoke_rows)
    safety_tbl = "\n".join(
        f"| {r['safety_id']} | {r['safety_type']} | {r['review_id']} | {r['patient_id'][:18]}… | "
        f"{float(r['candidate_score']):.1f} | {float(r['refined_roi_ratio']):.3f} | {r['must_not_flag_by']} |"
        for r in safety_out)

    md = f"""# B1-D3a Small-scope Smoke Test — Preflight

boundary rule(Rule-B3) / PatchCore gate(Gate-P2) candidate-level smoke test 의 입력/대상/출력/안전조건 잠금.
feature 추출·memory bank·NN·score 조정·boundary 적용 **없음**. 다음 단계(B1-D3b)에서 실행.

## 0. 판정
**PASS** — smoke {len(smoke_rows)}행 + safety {len(safety_out)}행 manifest 생성, ablation/안전조건/B1-D3b 순서 잠금.

## 1. B1-D2 요약
- primary: PatchCore/gated filter **Gate-P2**(3단계 flag-only), secondary: boundary **Rule-B3**(selective)
- Gate-P3 조합 보류, **global ratio threshold Rule-B1 = reject**
- score 원본 보존, adjusted score 금지, ablation 분리, lesion safety monitoring 필수

### 왜 global rule(Rule-B1)이 reject 인가
B_boundary refined_roi_ratio({b1d2['ratio_overlap_finding']['boundary_rule_refined_roi_ratio_min']}~{b1d2['ratio_overlap_finding']['boundary_rule_refined_roi_ratio_max']})와 LESION_RISK_partial({b1d2['ratio_overlap_finding']['lesion_risk_partial_refined_roi_ratio_min']}~{b1d2['ratio_overlap_finding']['lesion_risk_partial_refined_roi_ratio_max']})가 겹쳐(겹침대역 {b1d2['ratio_overlap_finding']['overlap_band']}), 단일 global ratio 컷이 병변 부분커버까지 함께 제거. 그래서 selective(Rule-B3) 만 smoke 대상.

## 2. Smoke subset 선정 이유
- **PatchCore gate {smoke_by_group.get('patchcore_gate_candidate',0)}개**: AD_wall_med D_keep 중 -score·환자 cap2. Gate-P2가 정상 경계구조를 flag-only로 구분하는지 확인.
- **boundary rule {smoke_by_group.get('boundary_rule_candidate',0)}개**: overlap_artifact 4 + hard_case 2. Rule-B3가 artifact만 표시하고 hard_case를 보호하는지 확인.
- **observation_other {smoke_by_group.get('observation_other',0)}개**: AD_other 고점수. gate/rule 비대상 — 범위에 안 들어가는지 관찰만.
- 환자 분산(cap2), stage2_holdout 0, AD_other는 최대 2개만 관찰.

### Smoke manifest ({len(smoke_rows)}행)
| smoke_id | source_group | review | patient | cause_class | prev/highres | score | refined | intended |
|---|---|---|---|---|---|---|---|---|
{smoke_tbl}

## 3. Rule-B3 smoke 설계 (적용 금지, 로직 문서화만)
- 적용 대상: B_boundary ∧ wall/mediastinum/boundary-like label ∧ refined_roi_ratio boundary range ∧ best_visual_label==B_patch_overlap_artifact
- 제외 대상: lesion_protect 전부, B_true_boundary_hard_case 전부, lesion safety sentinel
- 출력: `boundary_flag_candidate`(true/false) + `rule_reason`. **score 절대 미변경.**
- 이번 단계: 실제 계산 안 함. 필요 입력 컬럼 존재 확인 → **rule_b3_ready = {rule_b3_ready}** (cause_class/human_label/refined_roi_ratio/center_in_refined_roi/best_visual_label/safety_role 모두 존재)

## 4. Gate-P2 smoke 설계 (PatchCore 계산 금지, 설계만)
- 적용 대상: AD_wall_med_inside ∧ (highres D_keep 또는 patchcore_relevance yes) ∧ PaDiM high-score wall/med boundary-like
- 제외 대상: lesion_protect, AD_other_inside(기본 제외), safety sentinel
- 출력: `patchcore_gate_flag`(normal_like/uncertain/suspicious). distance/rank 는 다음 단계에서만. **score 절대 미변경.**

### memory bank 설계 원칙
- 소스: {memory_bank_design['source']} (root voldir {memory_bank_design['normal_ct_root_n_voldir']}개 존재)
- patch 범위: {memory_bank_design['patch_scope']}
- 금지: {memory_bank_design['forbidden']}
- 조건화: {memory_bank_design['conditioning']}, 크기제어: {memory_bank_design['size_control']}
- 이번 단계: {memory_bank_design['this_step']}

## 5. Safety sentinel 목록 ({len(safety_out)}행)
- lesion_risk_partial {safety_by_type.get('lesion_risk_partial',0)} (절단위험, both 금지)
- boundary_hard_case {safety_by_type.get('boundary_hard_case',0)} (Rule-B3 제거 금지)
- lesion_kept {safety_by_type.get('lesion_kept',0)} (고점수 baseline, normal_like 억제 금지)

| safety_id | type | review | patient | score | refined | must_not_flag_by |
|---|---|---|---|---|---|---|
{safety_tbl}

## 6. Ablation 출력 스키마 (분리 필수)
- Baseline: PaDiM score 원본(무수정)
- Ablation-1 (Rule-B3): `boundary_flag_candidate`, `rule_reason` — 별도 컬럼/파일
- Ablation-2 (Gate-P2): `patchcore_gate_flag` — 별도 컬럼/파일
- 두 출력은 **처음부터 혼합 금지**, 각각 독립 파일. score 컬럼은 어디서도 덮어쓰지 않음.

## 7. B1-D3b 권장 실행 순서
{chr(10).join('- ' + s for s in recommended_b1d3b_order)}

- rule_b3_ready: **{rule_b3_ready}**
- gate_p2_feature_preflight_ready: **{gate_p2_feature_preflight_ready}**
- gate_p2_minimal_smoke_ready: **{gate_p2_minimal_smoke_ready}** (사유: GPU feature 추출 = 사용자 승인 필요)

## 8. Safety FAIL 조건 (B1-D3b 이후 하나라도 발생 시 FAIL)
{chr(10).join('- ' + s for s in safety_fail_conditions)}

## 9. 다음 단계 프롬프트 방향
**B1-D3b Rule-B3 dry smoke** (feature 없는 최안전 단계) 우선 권장. 그 다음 **Gate-P2 feature preflight**(경로/대상/메모리 후보 수만). Gate-P2 minimal feature smoke 는 GPU 과금 승인 후 별도 진행.

---
patchcore_implemented={summary['patchcore_implemented']}, boundary_rule_implemented={summary['boundary_rule_implemented']}, score_modified={summary['score_modified']}, roi_modified={summary['roi_modified']}, stage2_holdout_access={stage2_holdout_access}
"""
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)

    # ---- 콘솔 ----
    print("[B1-D3a] PASS")
    print(f"  smoke_by_group={smoke_by_group}, safety_by_type={safety_by_type}")
    print(f"  smoke rows={len(smoke_rows)}, safety rows={len(safety_out)}, union patients={len(all_patients)}")
    print(f"  rule_b3_ready={rule_b3_ready}, gate_p2_feature_preflight_ready={gate_p2_feature_preflight_ready}, "
          f"gate_p2_minimal_smoke_ready={gate_p2_minimal_smoke_ready}")
    print(f"  생성: {OUT_MANIFEST.name}, {OUT_SAFETY.name}, {OUT_JSON.name}, {OUT_MD.name}")


if __name__ == "__main__":
    main()
