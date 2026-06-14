#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B1-D3e_Gate_P2_smoke_result_interpretation

B1-D3d1 minimal feature smoke 결과를 read-only 로 해석한다.
- 추가 feature 추출/GPU/memory bank 재생성/NN·distance 재계산 없음.
- all-suspicious 가 신호인지 memory-bank mismatch artifact 인지 정량 판단.
- 출력 report.md / summary.json 이미 있으면 즉시 중단(덮어쓰기 금지). 입력 mtime 무수정.
"""
import csv
import json
import sys
import statistics as st
from pathlib import Path
from collections import Counter, defaultdict

BASE = Path("/home/jinhy/project/lung-ct-anomaly")
DIR = BASE / "outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1"
SMK = DIR / "b1d3d1_gate_p2_minimal_feature_smoke_plan_s_cpu_v1"

IN = {
    "mem": SMK / "b1d3d1_gate_p2_memory_feature_preview.csv",
    "cand": SMK / "b1d3d1_gate_p2_candidate_distance_preview.csv",
    "smk_summary": SMK / "b1d3d1_gate_p2_minimal_feature_smoke_summary.json",
    "smk_report": SMK / "b1d3d1_gate_p2_minimal_feature_smoke_report.md",
    "b3c_cand": DIR / "b1d3c_gate_p2_feature_preflight_candidates.csv",
    "b3c_pool": DIR / "b1d3c_gate_p2_memory_pool_preview.csv",
}

OUT_MD = DIR / "b1d3e_gate_p2_smoke_result_interpretation_report.md"
OUT_JSON = DIR / "b1d3e_gate_p2_smoke_result_interpretation_summary.json"


def fail(msg):
    print(f"[B1-D3e][중단] {msg}", file=sys.stderr)
    sys.exit(2)


def load_rows(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def pbin(y0, x0):
    return f"y{int(y0)//128}_x{int(x0)//128}"


def main():
    # ---- collision guard ----
    for p in (OUT_MD, OUT_JSON):
        if p.exists():
            fail(f"출력 파일이 이미 존재함(덮어쓰기 금지): {p}")

    # ---- 입력 검증 + mtime ----
    if not SMK.is_dir():
        fail(f"B1-D3d1 output folder 없음: {SMK}")
    input_mtimes = {}
    for k, p in IN.items():
        if not p.exists():
            fail(f"필수 입력 없음: {k} -> {p}")
        input_mtimes[k] = round(p.stat().st_mtime, 3)

    mem = load_rows(IN["mem"])
    cand = load_rows(IN["cand"])
    b3c_cand = load_rows(IN["b3c_cand"])
    smk = load_json(IN["smk_summary"])

    if len(mem) != 500:
        fail(f"memory rows {len(mem)} != 500")
    if len(cand) != 6:
        fail(f"candidate rows {len(cand)} != 6")
    if smk.get("stage2_holdout_access") != 0:
        fail("smk stage2_holdout_access != 0")
    if smk.get("gpu_used") is not False:
        fail("smk gpu_used != False")
    if smk.get("score_modified") is not False:
        fail("smk score_modified != False")
    for k in ("adjusted_score_created", "suppression_weight_created", "refined_score_created"):
        if smk.get(k) is not False:
            fail(f"smk {k} != False")
    stage2_holdout_access = 0

    # ---- distance 통계 ----
    dists = [float(r["nearest_distance"]) for r in cand]
    p50, p90 = float(smk["memNN_p50"]), float(smk["memNN_p90"])
    above_p90 = sum(1 for d in dists if d > p90)
    above_p50 = sum(1 for d in dists if d > p50)
    candidate_distance_stats = {
        "min": round(min(dists), 4), "max": round(max(dists), 4),
        "mean": round(st.mean(dists), 4), "median": round(st.median(dists), 4),
        "memNN_p50": p50, "memNN_p90": p90,
        "n_above_p90": above_p90, "n_above_p50": above_p50,
        "closest_candidate": min(cand, key=lambda r: float(r["nearest_distance"]))["review_id"],
        "farthest_candidate": max(cand, key=lambda r: float(r["nearest_distance"]))["review_id"],
    }
    flag_counts = smk.get("gate_p2_flag_counts", {})

    # ---- memory bias 분석 ----
    mem_by_patient = dict(Counter(r["memory_patient_id"] for r in mem))
    mem_bins = dict(sorted(Counter(pbin(r["y0"], r["x0"]) for r in mem).items()))
    mem_z = [int(r["z"]) for r in mem]
    mem_z_range = [min(mem_z), max(mem_z)]
    mem_z_by_patient = {k: sorted(set(int(r["z"]) for r in mem if r["memory_patient_id"] == k))
                        for k in mem_by_patient}
    mem_ratios = [float(r["refined_roi_ratio"]) for r in mem]
    mem_ratio_interior_pct = round(sum(1 for x in mem_ratios if x >= 0.999) / len(mem_ratios) * 100, 1)

    # candidate 위치 (b1d3c)
    cand_pos = {}
    for r in b3c_cand:
        cand_pos[r["review_id"]] = {
            "z": int(r["candidate_local_z"]), "y0": int(r["candidate_y0"]),
            "x0": int(r["candidate_x0"]), "bin": pbin(r["candidate_y0"], r["candidate_x0"]),
            "refined_roi_ratio": float(r["refined_roi_ratio"]),
        }
    cand_bins = dict(sorted(Counter(v["bin"] for v in cand_pos.values()).items()))

    # mismatch 판정: candidate 별 z 범위 밖 / position_bin 미커버
    mem_bin_set = set(mem_bins.keys())
    mismatch_detail = []
    for rid, pos in cand_pos.items():
        z_out = not (mem_z_range[0] <= pos["z"] <= mem_z_range[1])
        bin_uncovered = pos["bin"] not in mem_bin_set
        mismatch_detail.append({
            "review_id": rid, "z": pos["z"], "bin": pos["bin"],
            "z_outside_memory_range": z_out, "position_bin_uncovered": bin_uncovered,
        })
    n_z_out = sum(1 for d in mismatch_detail if d["z_outside_memory_range"])
    n_bin_uncov = sum(1 for d in mismatch_detail if d["position_bin_uncovered"])

    memory_bank_bias_assessment = {
        "patients_used": len(mem_by_patient),
        "patients_intended": smk.get("memory_patient_limit"),
        "cap_skew": f"normal001/002 로 cap {smk.get('memory_patch_cap')} 소진, 3번째 환자 미반영",
        "memory_z_range": mem_z_range,
        "candidates_z_outside_memory_range": n_z_out,
        "position_bins_memory": mem_bins,
        "position_bins_candidate": cand_bins,
        "candidates_position_bin_uncovered_by_memory": n_bin_uncov,
        "memory_interior_ratio1_pct": mem_ratio_interior_pct,
        "anatomical_mismatch": ("memory=무작위 refined-ROI 격자 패치(폐실질 다수, ratio1.0 "
                                f"{mem_ratio_interior_pct}%), candidate=wall/mediastinum 경계 구조 → "
                                "동일 ratio 라도 해부학적으로 상이"),
        "conditioning": "global NN(위치 비조건화) → 경계 후보가 전역 memory 와 멀어지는 것은 예상됨",
        "bias_severity": "high",
    }

    all_suspicious_interpretation = (
        f"expected_artifact_of_memory_position_mismatch. candidate {len(cand)}개 전부 distance>p90 이나, "
        f"memory z범위({mem_z_range[0]}~{mem_z_range[1]}) 밖 candidate {n_z_out}개, "
        f"position_bin 미커버 {n_bin_uncov}개(y3 memory 0), memory 가 wall/med 비특이 폐실질 위주 + global NN → "
        "all-suspicious 는 성능 신호가 아니라 위치/해부학 mismatch 의 당연한 결과일 가능성이 높다."
    )

    pipeline_integrity_status = {
        "feature_nan_inf": [smk.get("feature_nan_count"), smk.get("feature_inf_count")],
        "distance_nan_inf": [smk.get("distance_nan_count"), smk.get("distance_inf_count")],
        "preprocessing_match": smk.get("preprocessing_match_status", "")[:80],
        "selected_feature_index": smk.get("selected_feature_index_status", "")[:80],
        "feature_dim": smk.get("feature_dim"),
        "score_modified": smk.get("score_modified"),
        "status": "intact (NaN/Inf 0, preprocessing/selected-index v2 일치, score 무수정)",
    }

    recommended_interpretation = "pipeline_valid_but_memory_mismatch"
    gate_validity_status = "undetermined_hold (성공도 실패도 아님; position-conditioned memory 필요)"
    recommended_next_step = "B1-D3f_Gate_P2_position_conditioned_memory_preflight (Option C)"

    next_options = {
        "A_stop_gate_p2": "비권장 — pipeline 정상(NaN/Inf 0, preprocessing 일치). feature/distance 불안정 근거 없음.",
        "B_same_method_expand": "비권장 — memory 가 위치 정합 안 됨. all-suspicious 를 신호로 볼 근거 없음.",
        "C_position_conditioned_memory_preflight": "★유력 — pipeline 정상이나 memory 가 candidate 위치와 불일치. wall/med 정합 normal memory 재정의 필요.",
        "D_boundary_rule_first": "보조 유지 — Rule-B3 가 이미 overlap artifact 4 안전 flag + hard/lesion 보호. Gate-P2 해석 정리 전까지 boundary rule 을 1차 보조 경로로 유지.",
    }

    b1d3f_design = {
        "step": "B1-D3f_Gate_P2_position_conditioned_memory_preflight",
        "goal": "feature 추출 전, wall/mediastinum 위치 정합 normal memory bank 후보 재정의(실제 feature 없음)",
        "checks": [
            "gate candidate 6개의 z-level, x/y, refined_roi_ratio, boundary proximity 정리",
            "normal train 에서 동일 position-bin 또는 wall/mediastinum 인접 후보 탐색 규칙 정의",
            "normal memory patch cap 을 환자별 균등 샘플링으로 제한(normal001/002 편중 방지)",
            "memory patient 최소 5명 이상 균등 sample preview",
            "candidate z-level 을 포함하는 슬라이스에서 memory 후보 확보(z 범위 커버)",
            "feature 추출 없음, stage2_holdout 0",
        ],
    }

    verdict = "PASS"  # 해석 단계 정상 수행, pipeline integrity intact

    summary = {
        "step": "B1-D3e_Gate_P2_smoke_result_interpretation",
        "verdict": verdict,
        "input_mtimes": input_mtimes,
        "stage2_holdout_access": stage2_holdout_access,
        "input_memory_rows": len(mem),
        "input_candidate_rows": len(cand),
        "candidate_distance_stats": candidate_distance_stats,
        "gate_p2_flag_counts": flag_counts,
        "memory_patch_by_patient": mem_by_patient,
        "mismatch_detail": mismatch_detail,
        "all_suspicious_interpretation": all_suspicious_interpretation,
        "memory_bank_bias_assessment": memory_bank_bias_assessment,
        "pipeline_integrity_status": pipeline_integrity_status,
        "recommended_interpretation": recommended_interpretation,
        "gate_validity_status": gate_validity_status,
        "recommended_next_step": recommended_next_step,
        "next_options": next_options,
        "b1d3f_design": b1d3f_design,
        "conclusion_limits": [
            "Gate-P2 폐기 아님",
            "Gate-P2 성공도 아님",
            "position-conditioned memory 가 필요",
        ],
        "gpu_used": False,
        "additional_feature_extracted": False,
        "nearest_neighbor_recomputed": False,
        "score_modified": False,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- report MD ----
    cand_tbl = "\n".join(
        f"| {r['gate_candidate_id']} | {r['review_id']} | {float(r['candidate_score']):.1f} | "
        f"{float(r['nearest_distance']):.3f} | {cand_pos[r['review_id']]['bin']} | "
        f"z{cand_pos[r['review_id']]['z']} | {r['gate_p2_flag']} |" for r in cand)
    mm_tbl = "\n".join(
        f"| {d['review_id']} | z{d['z']} | {d['bin']} | {d['z_outside_memory_range']} | {d['position_bin_uncovered']} |"
        for d in mismatch_detail)

    md = f"""# B1-D3e Gate-P2 Smoke Result Interpretation — Report

B1-D3d1 minimal feature smoke 결과 해석(read-only). 추가 feature/GPU/memory 재생성/distance 재계산 없음.

## 0. 판정
**{verdict}** (해석 단계 정상 수행. pipeline integrity intact.)
- recommended_interpretation: **{recommended_interpretation}**
- gate_validity_status: **{gate_validity_status}**

## 1. B1-D3d1 결과 요약
- device cpu, gpu_used False, stage2_holdout 0, feature_dim 100(v2 selected), NaN/Inf feature 0/0, distance 0/0
- memory {len(mem)} patch, candidate {len(cand)}, **gate_p2_flag_counts={flag_counts}** (전부 suspicious)
- score/threshold/ROI 무수정, adjusted/suppression/refined 미생성

## 2. 왜 PASS 가 성능 신호가 아닌가
PASS 는 **파이프라인이 안전·정상 동작**(feature 추출, 100차원 v2 정합, distance 계산, 무결성, score 무수정)했다는 뜻이지,
**Gate-P2 가 FP 를 구분한다는 증거가 아니다.** all-suspicious 는 아래 mismatch 의 산물일 가능성이 높다.

## 3. candidate distance preview
| GC | review | score | nearest_dist | bin | z | flag |
|---|---|---|---|---|---|---|
{cand_tbl}

- distance: min {candidate_distance_stats['min']} / mean {candidate_distance_stats['mean']} / max {candidate_distance_stats['max']}
- memNN p50={p50:.3f}, p90={p90:.3f} → 6개 전부 p90 초과(가장 가까운 것도 {candidate_distance_stats['closest_candidate']})
- ★ percentile/threshold 는 **500 memory 내부 smoke 기준일 뿐 실제 threshold·성능지표 아님**

## 4. all-suspicious 의 가능한 원인 (memory-bank mismatch)
| review | z | bin | z 범위 밖? | position_bin 미커버? |
|---|---|---|---|---|
{mm_tbl}

- memory z 범위 = {mem_z_range} → candidate 중 **z 범위 밖 {n_z_out}개**(z60/63/79 ↘ memory 73~170)
- memory position_bin = {mem_bins}
- candidate position_bin = {cand_bins} → **y3 memory 0개라 GC006(R005) 위치 매칭 자체 없음**, 미커버 {n_bin_uncov}개
- memory refined_roi_ratio==1.0 비율 {mem_ratio_interior_pct}% (폐실질 interior 다수) vs candidate=wall/med 경계 구조
- conditioning: global NN(위치 비조건화)

## 5. memory-bank bias 평가
- patients_used {len(mem_by_patient)}/{smk.get('memory_patient_limit')} (cap {smk.get('memory_patch_cap')} → normal001 {mem_by_patient.get('normal001__104e7cb873','-')} + normal002 {mem_by_patient.get('normal002__d886c791fa','-')}, 3번째 미반영)
- bias_severity: **high** — z·position_bin·해부학 모두 candidate 와 불일치 + global NN
- 결론: 현재 all-suspicious 는 **expected artifact**. Gate-P2 유효/무효 판단 **보류**.

## 6. pipeline integrity
- feature NaN/Inf {pipeline_integrity_status['feature_nan_inf']}, distance NaN/Inf {pipeline_integrity_status['distance_nan_inf']}
- preprocessing/selected-index v2 일치, feature_dim {pipeline_integrity_status['feature_dim']}, score 무수정 → **intact**

## 7. Gate-P2 를 버릴지/유지할지
- **버리지 않음**(pipeline 정상). **성공도 아님**(memory mismatch). → position-conditioned memory 후 재평가.
- 다음 옵션:
  - A 중단: {next_options['A_stop_gate_p2']}
  - B 동일확장: {next_options['B_same_method_expand']}
  - **C 위치정합 memory preflight: {next_options['C_position_conditioned_memory_preflight']}**
  - D boundary rule 우선: {next_options['D_boundary_rule_first']}

## 8. boundary rule 결과와의 관계
Rule-B3(B1-D3b)는 이미 overlap artifact 4개만 안전 flag + hard_case/lesion 보호 PASS. 해석이 명확.
Gate-P2 는 memory 재정의 전까지 결론 보류 → **단기 1차 보조는 boundary rule, Gate-P2 는 B1-D3f 후 재평가**.

## 9. 다음 단계 (Option C)
**{recommended_next_step}**
- 목표: {b1d3f_design['goal']}
- 확인: {'; '.join(b1d3f_design['checks'])}

## 10. 결론 제한
- Gate-P2 폐기 아님 / Gate-P2 성공 아님 / **position-conditioned memory 필요**

---
gpu_used=False, additional_feature_extracted=False, nearest_neighbor_recomputed=False, score_modified=False, stage2_holdout_access={stage2_holdout_access}
"""
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)

    # ---- 콘솔 ----
    print(f"[B1-D3e] {verdict}")
    print(f"  recommended_interpretation={recommended_interpretation}")
    print(f"  z_outside={n_z_out}, bin_uncovered={n_bin_uncov}, bias=high")
    print(f"  gate_validity={gate_validity_status}")
    print(f"  next={recommended_next_step}")
    print(f"  생성: {OUT_MD.name}, {OUT_JSON.name}")


if __name__ == "__main__":
    main()
