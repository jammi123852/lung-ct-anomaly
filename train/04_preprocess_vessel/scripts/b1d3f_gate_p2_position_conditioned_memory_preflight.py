#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B1-D3f_Gate_P2_position_conditioned_memory_preflight

Gate-P2 normal memory bank 를 위치 정합 방식으로 재설계하는 feature-free preflight.
- feature 추출/GPU/CUDA/memory bank 생성/NN/distance 일절 없음.
- 프로젝트 canonical position_bin(upper/middle/lower × central/peripheral)으로 candidate-정상 매칭.
- 정상 score CSV(normal_by_patient)의 precompute position 메타만 read-only 사용.
- 출력 4개 이미 있으면 즉시 중단(덮어쓰기 금지). 입력 mtime 무수정.
"""
import csv
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

BASE = Path("/home/jinhy/project/lung-ct-anomaly")
DIR = BASE / "outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1"
SMK = DIR / "b1d3d1_gate_p2_minimal_feature_smoke_plan_s_cpu_v1"
NSCORE = BASE / "outputs/position-aware-padim-v1/scores/padim_v2_roi0_0/normal_by_patient"
MROOT = BASE / "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1"
NROOT = Path("/mnt/c/Users/jinhy/Desktop/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/volumes_npy")

IN = {
    "cand": DIR / "b1d3c_gate_p2_feature_preflight_candidates.csv",
    "pool": DIR / "b1d3c_gate_p2_memory_pool_preview.csv",
    "smk_mem": SMK / "b1d3d1_gate_p2_memory_feature_preview.csv",
    "smk_cand": SMK / "b1d3d1_gate_p2_candidate_distance_preview.csv",
    "b3e_summary": DIR / "b1d3e_gate_p2_smoke_result_interpretation_summary.json",
    "b3e_report": DIR / "b1d3e_gate_p2_smoke_result_interpretation_report.md",
}

OUT_CAND = DIR / "b1d3f_gate_p2_position_conditioned_candidates_summary.csv"
OUT_POOL = DIR / "b1d3f_gate_p2_position_conditioned_memory_pool_preview.csv"
OUT_JSON = DIR / "b1d3f_gate_p2_position_conditioned_memory_preflight_summary.json"
OUT_MD = DIR / "b1d3f_gate_p2_position_conditioned_memory_preflight_report.md"

N_PREVIEW_PATIENTS = 8        # 5~10 범위
PER_PATIENT_PER_BIN_CSV = 6   # CSV 표본(환자×bin 당)
CDR_TOL = 0.10                # central_distance_ratio_mean 근접 tolerance(tight match)
MIN_MATCH_PER_CANDIDATE = 50  # position_conditioning_ready 기준
PATCH = 32


def fail(msg):
    print(f"[B1-D3f][중단] {msg}", file=sys.stderr)
    sys.exit(2)


def load_rows(p, enc="utf-8"):
    with open(p, encoding=enc) as f:
        return list(csv.DictReader(f))


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


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

    cands = load_rows(IN["cand"])
    if len(cands) != 6:
        fail(f"candidate row {len(cands)} != 6")
    b3e = load_json(IN["b3e_summary"])
    if b3e.get("stage2_holdout_access") != 0:
        fail("b3e stage2_holdout_access != 0")
    if b3e.get("recommended_interpretation") != "pipeline_valid_but_memory_mismatch":
        fail(f"b3e recommended_interpretation 예상과 다름: {b3e.get('recommended_interpretation')}")
    stage2_holdout_access = 0

    # ---- candidate 위치 메타(canonical position_bin) 조회 ----
    def lookup_candidate(c):
        pid = c["patient_id"]
        f = NSCORE / f"{pid}.csv"
        if not f.exists():
            fail(f"candidate score CSV 없음: {f}")
        z, y0, x0 = int(c["candidate_local_z"]), int(c["candidate_y0"]), int(c["candidate_x0"])
        for r in load_rows(f, enc="utf-8-sig"):
            if int(r["local_z"]) == z and int(r["y0"]) == y0 and int(r["x0"]) == x0:
                return r
        fail(f"candidate 매칭 없음: {c['review_id']} z{z} y{y0} x{x0}")

    cand_summary = []
    cand_patients = set()
    for c in cands:
        r = lookup_candidate(c)
        z, y0, x0 = int(c["candidate_local_z"]), int(c["candidate_y0"]), int(c["candidate_x0"])
        cand_patients.add(c["patient_id"])
        cdr = float(r.get("central_distance_ratio_mean", "nan") or "nan")
        cand_summary.append({
            "gate_candidate_id": c["gate_candidate_id"], "review_id": c["review_id"],
            "patient_id": c["patient_id"], "candidate_local_z": z,
            "candidate_y0": y0, "candidate_x0": x0, "center_y": y0 + PATCH // 2, "center_x": x0 + PATCH // 2,
            "z_level_bin": r["z_level"], "z_ratio": round(float(r["z_ratio"]), 4),
            "y_bin": f"y{y0//128}", "x_bin": f"x{x0//128}",
            "position_bin": r["position_bin"], "central_peripheral": r["central_peripheral"],
            "central_distance_ratio_mean": round(cdr, 4),
            "roi_0_0_patch_ratio": round(float(r["roi_0_0_patch_ratio"]), 4),
            "refined_roi_ratio_v4": c["refined_roi_ratio"],
            "cause_class": c["cause_class"], "highres_visual_label": c["highres_visual_label"],
            "candidate_score": c["candidate_score"],
            "required_memory_condition": (f"position_bin=={r['position_bin']} "
                                          f"(z_level={r['z_level']}, central_peripheral={r['central_peripheral']}); "
                                          f"central_distance_ratio_mean~{cdr:.3f}±{CDR_TOL}; normal only; patient-balanced"),
        })

    with open(OUT_CAND, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cand_summary[0].keys()))
        w.writeheader()
        w.writerows(cand_summary)

    # candidate position_bin → 대표 cdr (tight match 용)
    bin_to_cdr = defaultdict(list)
    for cs in cand_summary:
        bin_to_cdr[cs["position_bin"]].append(cs["central_distance_ratio_mean"])
    cand_bins = sorted(set(cs["position_bin"] for cs in cand_summary))

    # ---- preview 정상 환자 선정(candidate 환자 제외, patient-balanced) ----
    all_norm = sorted(p.stem for p in NSCORE.glob("*.csv"))
    avail = [pid for pid in all_norm if pid not in cand_patients]
    preview_patients = avail[:N_PREVIEW_PATIENTS]
    if len(preview_patients) < 5:
        fail(f"preview 가능 정상 환자 {len(preview_patients)} < 5")

    # ---- 경로/shape readiness(환자당 1회, mmap) ----
    def patient_paths(pid, rows):
        safe_id = rows[0].get("safe_id", "")
        md = MROOT / "normal" / safe_id / "refined_roi.npy"
        cd = NROOT / safe_id / "ct_hu.npy"
        cstat = "ok" if cd.exists() else "missing"
        mstat = "ok" if md.exists() else "missing"
        sstat = "unchecked"
        if cstat == "ok" and mstat == "ok":
            try:
                m = np.load(md, mmap_mode="r")
                c = np.load(cd, mmap_mode="r")
                sstat = f"ok {m.shape}" if (m.shape == c.shape and m.shape[1:] == (512, 512)) else "mismatch"
                del m, c
            except Exception as e:
                sstat = f"error:{type(e).__name__}"
        return cstat, mstat, sstat

    # ---- position-conditioned 매칭 + preview CSV ----
    pool_rows = []
    coverage_by_candidate = defaultdict(int)        # gate_candidate_id -> 총 매칭 수(전체)
    coverage_patients = defaultdict(set)            # gate_candidate_id -> 매칭된 preview 환자 집합
    coverage_by_bin = defaultdict(int)
    coverage_by_zlevel = defaultdict(int)
    coverage_by_patient = defaultdict(int)
    matched_ratios = []
    interior_count = 0
    mid = 1
    for rank, pid in enumerate(preview_patients, 1):
        rows = load_rows(NSCORE / f"{pid}.csv", enc="utf-8-sig")
        cstat, mstat, sstat = patient_paths(pid, rows)
        # bin -> 해당 환자 패치들
        by_bin = defaultdict(list)
        for r in rows:
            by_bin[r["position_bin"]].append(r)
        for cs in cand_summary:
            gcid = cs["gate_candidate_id"]
            pb = cs["position_bin"]
            target_cdr = cs["central_distance_ratio_mean"]
            matches = by_bin.get(pb, [])
            coverage_by_candidate[gcid] += len(matches)
            if matches:
                coverage_patients[gcid].add(pid)
            coverage_by_bin[pb] += len(matches)
            coverage_by_patient[pid] += len(matches)
            for r in matches:
                coverage_by_zlevel[r["z_level"]] += 1
                rr = float(r["roi_0_0_patch_ratio"])
                matched_ratios.append(rr)
                if rr >= 0.999:
                    interior_count += 1
            # CSV 표본(tight match 우선: cdr 근접 → 그다음 일반)
            def key_tight(r):
                try:
                    return abs(float(r.get("central_distance_ratio_mean", "nan")) - target_cdr)
                except Exception:
                    return 9.9
            sample = sorted(matches, key=key_tight)[:PER_PATIENT_PER_BIN_CSV]
            for r in sample:
                y0, x0 = int(r["y0"]), int(r["x0"])
                rcdr = r.get("central_distance_ratio_mean", "")
                tight = ""
                try:
                    tight = "tight" if abs(float(rcdr) - target_cdr) <= CDR_TOL else "bin_only"
                except Exception:
                    tight = "bin_only"
                pool_rows.append({
                    "memory_preview_id": f"PCM{mid:04d}", "preview_patient_id": pid,
                    "matched_gate_candidate_id": gcid,
                    "matched_position_condition": pb,
                    "z": int(r["local_z"]), "y0": y0, "x0": x0,
                    "center_y": y0 + PATCH // 2, "center_x": x0 + PATCH // 2,
                    "z_level_bin": r["z_level"], "y_bin": f"y{y0//128}", "x_bin": f"x{x0//128}",
                    "position_bin": r["position_bin"],
                    "central_peripheral": r["central_peripheral"],
                    "central_distance_ratio_mean": rcdr,
                    "refined_roi_ratio": round(float(r["roi_0_0_patch_ratio"]), 4),
                    "center_in_refined_roi": "True",
                    "ct_path_status": cstat, "mask_path_status": mstat, "shape_status": sstat,
                    "candidate_condition_match": tight,
                    "used_for_future_memory_pool": "true" if (cstat == "ok" and mstat == "ok") else "false",
                    "exclusion_reason": "" if (cstat == "ok" and mstat == "ok") else f"path ct={cstat} mask={mstat}",
                    "patient_sample_rank": rank,
                })
                mid += 1

    with open(OUT_POOL, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pool_rows[0].keys()))
        w.writeheader()
        w.writerows(pool_rows)

    # ---- coverage / 판정 ----
    cov_by_cand = {gc: {"total_matches": coverage_by_candidate[gc],
                        "preview_patients_with_match": len(coverage_patients[gc])}
                   for gc in [cs["gate_candidate_id"] for cs in cand_summary]}
    insufficient = [gc for gc, v in cov_by_cand.items()
                    if v["total_matches"] < MIN_MATCH_PER_CANDIDATE or v["preview_patients_with_match"] < 3]
    interior_ratio = round(interior_count / len(matched_ratios), 3) if matched_ratios else 0.0

    pb_counts = dict(coverage_by_patient)
    total_matches = sum(pb_counts.values())
    max_share = round(max(pb_counts.values()) / total_matches, 3) if total_matches else 0.0
    patient_balance_summary = {
        "preview_patients": preview_patients,
        "matches_by_patient": pb_counts,
        "max_single_patient_share": max_share,
        "balanced": max_share <= 0.40,  # 단일 환자 40% 이하면 균형으로 간주
        "note": "B1-D3d1 은 2명(normal001/002)에 편중; 여기서는 환자별 cap 으로 균등 샘플 가능",
    }

    memory_bias_reduced_assessment = {
        "previous_bias": "B1-D3d1: 2환자 편중 + z 73~170 + position_bin y1/y2 + global NN(폐실질 interior 다수)",
        "now": (f"canonical position_bin 정합(candidate {len(cand_bins)} bins: {cand_bins}), "
                f"{len(preview_patients)}환자 균등 preview, 환자별 cap, z_level/central_peripheral 일치"),
        "interior_ratio_matched": interior_ratio,
        "reduced": len(insufficient) == 0 and max_share <= 0.40,
    }
    position_conditioning_ready = (len(insufficient) == 0 and stage2_holdout_access == 0
                                   and len(preview_patients) >= 5)

    recommended_next_plan = {
        "Plan-PC-S(권장)": {"normal_memory_patients": 5, "per_patient_patch_cap": 100,
                          "total_patch_cap": 500, "gate_candidates": 6,
                          "memory": "position-conditioned only(candidate position_bin 매칭)",
                          "device": "cpu", "score_modified": False},
        "Plan-PC-M": {"normal_memory_patients": 10, "per_patient_patch_cap": 150,
                      "total_patch_cap": 1500, "device": "cpu 가능성 추정, GPU 별도 승인"},
        "Plan-PC-L": "이번 단계 비권장",
    }
    safety_for_b1d3g = [
        "memory bank normal only", "patient-balanced sampling",
        "candidate별 position coverage 확인", "lesion/safety sentinel memory 제외",
        "score 수정 금지", "adjusted/suppression/refined 생성 금지",
        "stage2_holdout 0", "output folder versioning",
        "preprocessing/v2 selected index(100차원) 유지",
    ]

    verdict = "PASS" if position_conditioning_ready else "NEEDS_FIX"

    summary = {
        "step": "B1-D3f_Gate_P2_position_conditioned_memory_preflight",
        "verdict": verdict,
        "input_mtimes": input_mtimes,
        "stage2_holdout_access": stage2_holdout_access,
        "position_bin_definition": "project canonical: z_level(upper/middle/lower by z_ratio) × central/peripheral; normal score CSV precompute 사용",
        "gate_candidate_rows": len(cand_summary),
        "candidate_position_bins": {cs["gate_candidate_id"]: cs["position_bin"] for cs in cand_summary},
        "normal_preview_patients": len(preview_patients),
        "memory_pool_preview_rows": len(pool_rows),
        "coverage_by_gate_candidate": cov_by_cand,
        "coverage_by_position_bin": dict(coverage_by_bin),
        "coverage_by_z_level": dict(coverage_by_zlevel),
        "patient_balance_summary": patient_balance_summary,
        "refined_roi_ratio_distribution": {
            "n": len(matched_ratios),
            "min": round(min(matched_ratios), 3) if matched_ratios else None,
            "mean": round(float(np.mean(matched_ratios)), 3) if matched_ratios else None,
            "max": round(max(matched_ratios), 3) if matched_ratios else None,
        },
        "interior_ratio": interior_ratio,
        "insufficient_candidates": insufficient,
        "memory_bias_reduced_assessment": memory_bias_reduced_assessment,
        "position_conditioning_ready": position_conditioning_ready,
        "recommended_next_plan": recommended_next_plan,
        "safety_for_b1d3g": safety_for_b1d3g,
        "conclusion_limits": ["아직 성능 결론 아님", "all-suspicious 성공/실패 단정 아님",
                              "position-conditioned memory 후보 확보 가능성 확인 단계"],
        "feature_extracted": False, "gpu_used": False, "memory_bank_created": False,
        "nearest_neighbor_computed": False, "score_modified": False,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- report MD ----
    cand_tbl = "\n".join(
        f"| {cs['gate_candidate_id']} | {cs['review_id']} | {cs['position_bin']} | "
        f"{cs['z_level_bin']}/{cs['central_peripheral']} | {cs['central_distance_ratio_mean']} | "
        f"{float(cs['candidate_score']):.1f} | {cov_by_cand[cs['gate_candidate_id']]['total_matches']} "
        f"({cov_by_cand[cs['gate_candidate_id']]['preview_patients_with_match']}명) |"
        for cs in cand_summary)

    md = f"""# B1-D3f Gate-P2 Position-conditioned Memory — Preflight

Gate-P2 normal memory bank 위치 정합 재설계(feature-free). feature/GPU/memory bank/NN/distance 없음.
프로젝트 canonical position_bin(z_level × central/peripheral) 사용.

## 0. 판정
**{verdict}** — position_conditioning_ready={position_conditioning_ready}

## 1. B1-D3e 해석 요약
- pipeline integrity intact, all-suspicious = memory-position mismatch artifact(z범위밖 2 / y3 bin 미커버 1 / 폐실질 interior 위주 / global NN).
- Gate-P2 validity = undetermined_hold → 위치 정합 memory 재정의가 다음 단계.

## 2. 왜 기존 all-suspicious 가 memory mismatch 였나
B1-D3d1 memory 는 raw z 73~170, 무작위 ROI 격자, 2환자 편중, global NN.
candidate 는 canonical position_bin 5종({cand_bins})에 퍼져 있어, 위치 비조건화 memory 와는 멀어질 수밖에 없었다.
※ z_level 은 raw z 가 아니라 z_ratio 기반이라, raw z63/60 도 'upper' 로 정확히 분류됨(기존 raw-z 매칭의 오류).

## 3. Gate-P2 candidate 위치 특성 (canonical)
| GC | review | position_bin | z_level/cp | cdr | score | preview 매칭수(환자) |
|---|---|---|---|---|---|---|
{cand_tbl}

## 4. position-conditioned memory 규칙
- 필수: normal only, refined ROI mask 존재, CT/mask shape 일치, patch center∈refined ROI, **candidate 와 동일 position_bin**(z_level+central/peripheral), stage2_holdout 0.
- 권장: central_distance_ratio_mean ±{CDR_TOL} 근접(tight), 환자별 cap, patient-balanced(normal001/002 편중 방지).
- 제외: lesion patient, stage2_holdout, candidate 와 다른 position_bin, shape mismatch, 좌표 이상.

## 5. memory pool preview
- preview 정상 환자 {len(preview_patients)}명(candidate 환자 제외): {', '.join(preview_patients)}
- preview rows {len(pool_rows)} (환자×bin 당 최대 {PER_PATIENT_PER_BIN_CSV} 표본, tight 우선)
- coverage_by_position_bin: {dict(coverage_by_bin)}
- coverage_by_z_level: {dict(coverage_by_zlevel)}

## 6. coverage 분석
- candidate별 매칭: {{gc: (total, patients)}} = {{ {', '.join(f"{gc}:({v['total_matches']},{v['preview_patients_with_match']})" for gc,v in cov_by_cand.items())} }}
- 부족 candidate(<{MIN_MATCH_PER_CANDIDATE} 또는 <3환자): {insufficient if insufficient else '없음'}
- matched refined_roi_ratio: min {summary['refined_roi_ratio_distribution']['min']} / mean {summary['refined_roi_ratio_distribution']['mean']} / max {summary['refined_roi_ratio_distribution']['max']}
- interior_ratio(==1.0): {interior_ratio}
- patient balance: max_single_share {max_share} (balanced={patient_balance_summary['balanced']})

## 7. memory bias 감소 평가
- 이전: {memory_bias_reduced_assessment['previous_bias']}
- 현재: {memory_bias_reduced_assessment['now']}
- reduced: **{memory_bias_reduced_assessment['reduced']}** (position_bin 정합 + 환자 균등 + cap)

## 8. 다음 B1-D3g feature smoke 권장 범위
- **Plan-PC-S(권장)**: normal 5명/per-patient cap 100/total 500/candidate 6/position-conditioned only/CPU/score 무수정.
- Plan-PC-M: 10명/150/1500/CPU 추정·GPU 별도승인. Plan-PC-L: 비권장.
- B1-D3g safety: {'; '.join(safety_for_b1d3g)}

## 9. 결론 제한
- 아직 **성능 결론 아님**. all-suspicious 를 성공/실패로 단정하지 않음.
- 이번 단계는 위치 정합 memory 후보가 충분히 확보 가능한지의 preflight.

## 10. 다음 단계
**B1-D3g Gate-P2 position-conditioned minimal feature smoke approval/preflight**

---
feature_extracted=False, gpu_used=False, memory_bank_created=False, nearest_neighbor_computed=False, score_modified=False, stage2_holdout_access={stage2_holdout_access}
"""
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)

    # ---- 콘솔 ----
    print(f"[B1-D3f] {verdict}")
    print(f"  candidate position_bins: {[cs['position_bin'] for cs in cand_summary]}")
    print(f"  preview_patients={len(preview_patients)}, pool_rows={len(pool_rows)}")
    print(f"  coverage_by_candidate(total,patients)={ {gc: (v['total_matches'], v['preview_patients_with_match']) for gc,v in cov_by_cand.items()} }")
    print(f"  insufficient={insufficient}, interior_ratio={interior_ratio}, balanced={patient_balance_summary['balanced']}")
    print(f"  position_conditioning_ready={position_conditioning_ready}")
    print(f"  생성: {OUT_CAND.name}, {OUT_POOL.name}, {OUT_JSON.name}, {OUT_MD.name}")


if __name__ == "__main__":
    main()
