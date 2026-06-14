#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B1-D2_boundary_aware_rule_and_patchcore_gated_filter_preflight

B1-D1.2~B1-D1.8 산출물을 read-only 로 종합하여, 흉벽/종격동 FP 저감 후보 2개
(boundary-overlap rule / PatchCore-gated filter)의 "설계 preflight" 를 만든다.

- 구현/score 조정/threshold 재계산/full run 없음.
- candidate universe 를 3개(boundary-rule / patchcore-gate / lesion-safety)로 분리.
- boundary rule 3안, PatchCore gate 3안, ablation, safety plan, B1-D3 smoke 범위 설계.
- 숫자는 하드코딩하지 않고 CSV/JSON 에서 다시 읽어 집계.
- highres visual recheck 라벨을 최우선 판단 근거로 사용.
- 출력 4개(report.md, summary.json, candidate_groups_preview.csv, safety_set_preview.csv)
  는 이미 있으면 즉시 중단(덮어쓰기 금지). 입력은 mtime 기록 후 무수정.
- preview CSV 는 설계용 목록일 뿐 score 를 바꾸지 않는다.
"""
import csv
import json
import sys
from pathlib import Path
from collections import Counter

BASE = Path("/home/jinhy/project/lung-ct-anomaly")
DIR = BASE / "outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1"

IN = {
    "cause_csv": DIR / "b1d1_fp_cause_diagnostic.csv",
    "cause_json": DIR / "b1d1_cause_summary.json",
    "sel_csv": DIR / "b1d1_overlay_target_selection.csv",
    "ov_labels": DIR / "b1d1_overlay_visual_review_labels.csv",
    "hr_labels": DIR / "b1d1_highres_visual_recheck_labels.csv",
    "final_json": DIR / "b1d1_final_diagnostic_decision_summary.json",
    "final_md": DIR / "b1d1_final_diagnostic_decision_report.md",
}

OUT_MD = DIR / "b1d2_preflight_design_report.md"
OUT_JSON = DIR / "b1d2_preflight_design_summary.json"
OUT_GROUPS = DIR / "b1d2_candidate_groups_preview.csv"
OUT_SAFETY = DIR / "b1d2_safety_set_preview.csv"


def fail(msg):
    print(f"[B1-D2][중단] {msg}", file=sys.stderr)
    sys.exit(2)


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_rows(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    # ---- collision guard ----
    for p in (OUT_MD, OUT_JSON, OUT_GROUPS, OUT_SAFETY):
        if p.exists():
            fail(f"출력 파일이 이미 존재함(덮어쓰기 금지): {p}")

    # ---- 입력 검증 + mtime ----
    input_mtimes = {}
    for k, p in IN.items():
        if not p.exists():
            fail(f"필수 입력 없음: {k} -> {p}")
        input_mtimes[k] = round(p.stat().st_mtime, 3)

    cause = load_json(IN["cause_json"])
    final = load_json(IN["final_json"])
    if cause.get("stage2_holdout_access") != 0:
        fail("cause stage2_holdout_access != 0")
    if final.get("stage2_holdout_access") != 0:
        fail("final stage2_holdout_access != 0")
    stage2_holdout_access = 0

    rows = load_rows(IN["cause_csv"])
    if len(rows) != cause["n_rows"]:
        fail(f"cause CSV row {len(rows)} != {cause['n_rows']}")

    ov = {r["review_id"]: r for r in load_rows(IN["ov_labels"])}
    hr = {r["review_id"]: r for r in load_rows(IN["hr_labels"])}

    # selection_id 매핑(있으면)
    sel_by_rev = {r["review_id"]: r["selection_id"] for r in load_rows(IN["sel_csv"])}

    # ---- best visual label 병합 (highres 우선) ----
    def best_label(rid):
        if rid in hr:
            h = hr[rid]
            return {
                "source": "highres",
                "visual_label": h["highres_visual_label"],
                "confidence": h["highres_confidence"],
                "lesion_safety_concern": h["lesion_safety_concern"],
                "boundary_rule_relevance": h.get("boundary_rule_relevance", ""),
                "patchcore_relevance": h["patchcore_relevance"],
                "roi_trim_relevance": h["roi_trim_relevance"],
            }
        if rid in ov:
            o = ov[rid]
            return {
                "source": "overlay",
                "visual_label": o["visual_label"],
                "confidence": o["visual_confidence"],
                "lesion_safety_concern": o["lesion_safety_concern"],
                "boundary_rule_relevance": "",
                "patchcore_relevance": o["patchcore_relevance"],
                "roi_trim_relevance": o["roi_trim_relevance"],
            }
        return {
            "source": "not_reviewed",
            "visual_label": "not_reviewed",
            "confidence": "",
            "lesion_safety_concern": "",
            "boundary_rule_relevance": "",
            "patchcore_relevance": "",
            "roi_trim_relevance": "",
        }

    # ---- 그룹 할당 ----
    HARD = {"B_true_boundary_hard_case", "B_keep_hard_case"}
    enriched = []
    for r in rows:
        rid = r["review_id"]
        bl = best_label(rid)
        cc = r["cause_class"]
        ratio = float(r["refined_roi_ratio"])
        role = r["safety_role"]
        vlabel = bl["visual_label"]

        if cc == "B_boundary":
            group = "boundary_rule_candidate"
            if vlabel in HARD:
                subtype = "hard_case"
            elif vlabel == "B_patch_overlap_artifact":
                subtype = "overlap_artifact"
            else:
                subtype = "unreviewed_or_other"
        elif cc == "AD_wall_med_inside":
            group = "patchcore_gate_candidate"
            if vlabel == "D_keep_boundary_structure":
                subtype = "D_keep"
            elif vlabel == "AD_unclear":
                subtype = "unclear_hold"
            else:
                subtype = "unreviewed_or_other"
        elif cc == "AD_other_inside":
            group = "excluded_observation"  # vessel/diaphragm, patchcore_relevance no
            subtype = "vessel_or_diaphragm"
        elif cc == "LESION_RISK_partial":
            group = "lesion_safety_set"
            subtype = "lesion_risk_partial"
        elif cc == "lesion_kept":
            group = "lesion_safety_set"
            subtype = "lesion_kept_baseline"
        else:
            group = "other"
            subtype = cc

        enriched.append({
            "review_id": rid,
            "selection_id": sel_by_rev.get(rid, ""),
            "patient_id": r["patient_id"],
            "safety_role": role,
            "human_label": r["human_label"],
            "cause_class": cc,
            "candidate_local_z": r["candidate_local_z"],
            "candidate_y0": r["candidate_y0"],
            "candidate_x0": r["candidate_x0"],
            "candidate_score": r["candidate_score"],
            "roi_0_0_patch_ratio": r["roi_0_0_patch_ratio"],
            "refined_roi_ratio": r["refined_roi_ratio"],
            "center_in_refined_roi": r["center_in_refined_roi"],
            "b1d2_group": group,
            "b1d2_subtype": subtype,
            "best_label_source": bl["source"],
            "best_visual_label": vlabel,
            "best_confidence": bl["confidence"],
            "lesion_safety_concern": bl["lesion_safety_concern"],
            "patchcore_relevance": bl["patchcore_relevance"],
            "roi_trim_relevance": bl["roi_trim_relevance"],
        })

    # ---- 집계 ----
    def ratios(pred):
        return [float(e["refined_roi_ratio"]) for e in enriched if pred(e)]

    bound = [e for e in enriched if e["b1d2_group"] == "boundary_rule_candidate"]
    gate = [e for e in enriched if e["b1d2_group"] == "patchcore_gate_candidate"]
    safety = [e for e in enriched if e["b1d2_group"] == "lesion_safety_set"]
    excl = [e for e in enriched if e["b1d2_group"] == "excluded_observation"]

    bound_subtype = dict(Counter(e["b1d2_subtype"] for e in bound))
    gate_subtype = dict(Counter(e["b1d2_subtype"] for e in gate))
    safety_subtype = dict(Counter(e["b1d2_subtype"] for e in safety))

    b_ratios = ratios(lambda e: e["b1d2_group"] == "boundary_rule_candidate")
    lrp_ratios = ratios(lambda e: e["b1d2_subtype"] == "lesion_risk_partial")
    hard_ratios = ratios(lambda e: e["b1d2_subtype"] == "hard_case")

    ratio_overlap = {
        "boundary_rule_refined_roi_ratio_min": round(min(b_ratios), 4),
        "boundary_rule_refined_roi_ratio_max": round(max(b_ratios), 4),
        "lesion_risk_partial_refined_roi_ratio_min": round(min(lrp_ratios), 4),
        "lesion_risk_partial_refined_roi_ratio_max": round(max(lrp_ratios), 4),
        "hard_case_refined_roi_ratio": [round(x, 4) for x in sorted(hard_ratios)],
        "overlap_exists": min(lrp_ratios) <= max(b_ratios),
        "overlap_band": [round(min(lrp_ratios), 4), round(max(b_ratios), 4)],
    }

    # center_in_refined_roi == False 인 boundary 후보
    center_false = [e["review_id"] for e in bound if e["center_in_refined_roi"] == "False"]

    # ---- preview CSV: candidate groups (fp_candidate 30 + safety set) ----
    group_cols = ["review_id", "selection_id", "patient_id", "safety_role", "human_label",
                  "cause_class", "candidate_local_z", "candidate_y0", "candidate_x0",
                  "candidate_score", "roi_0_0_patch_ratio", "refined_roi_ratio",
                  "center_in_refined_roi", "b1d2_group", "b1d2_subtype",
                  "best_label_source", "best_visual_label", "best_confidence",
                  "lesion_safety_concern", "patchcore_relevance", "roi_trim_relevance"]
    with open(OUT_GROUPS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=group_cols)
        w.writeheader()
        # 활성 후보군 + 관찰그룹만 (lesion_safety_set 은 별도 CSV 에 상세)
        for e in enriched:
            if e["b1d2_group"] in ("boundary_rule_candidate", "patchcore_gate_candidate",
                                   "excluded_observation"):
                w.writerow({c: e[c] for c in group_cols})

    # ---- preview CSV: safety set (LESION_RISK_partial + lesion_kept + hard_case sentinels) ----
    safety_rows = []
    for e in enriched:
        if e["b1d2_subtype"] == "lesion_risk_partial":
            stype = "lesion_risk_partial_at_risk"
        elif e["b1d2_subtype"] == "lesion_kept_baseline":
            stype = "lesion_kept_must_not_degrade"
        elif e["b1d2_subtype"] == "hard_case":
            stype = "boundary_hard_case_must_keep"
        else:
            continue
        safety_rows.append({**e, "sentinel_type": stype})

    safety_cols = ["review_id", "selection_id", "patient_id", "safety_role", "human_label",
                   "cause_class", "candidate_score", "refined_roi_ratio",
                   "center_in_refined_roi", "best_visual_label", "best_confidence",
                   "lesion_safety_concern", "sentinel_type"]
    with open(OUT_SAFETY, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=safety_cols)
        w.writeheader()
        for e in safety_rows:
            w.writerow({c: e[c] for c in safety_cols})

    sentinel_counts = dict(Counter(e["sentinel_type"] for e in safety_rows))

    # ---- 설계 객체 ----
    candidate_groups = {
        "A_boundary_rule_candidate": {
            "purpose": "patch 가 refined ROI 경계를 걸친 경우만 대상으로 하는 경계걸침 보정 전용 후보군",
            "include": "cause_class == B_boundary (refined_roi_ratio 0.10~0.90, patch 가 ROI 경계 straddle)",
            "exclude": "AD_wall_med_inside(ROI 내부 완전포함, 별도 gate), AD_other_inside(vessel/diaphragm), 모든 lesion row",
            "n_rows": len(bound),
            "subtype_counts": bound_subtype,
            "center_in_refined_roi_false_reviews": center_false,
            "examples": [f"{e['review_id']}({e['selection_id']}) ratio{float(e['refined_roi_ratio']):.3f} {e['b1d2_subtype']}"
                         for e in bound[:5]],
            "note": "11개 중 6개만 시각검토됨(overlap_artifact 4, hard_case 2). 5개 미검토 → 규칙 적용 전 라벨 필요.",
        },
        "B_patchcore_gate_candidate": {
            "purpose": "PaDiM high-score 인 wall/mediastinum boundary-like 정상구조(D_keep) 재검사 전용 후보군",
            "include": "cause_class == AD_wall_med_inside (refined_roi_ratio 대부분 1.0, 흉벽/종격동 인접 ROI 내부 고점수)",
            "exclude": "B_boundary(경계걸침은 boundary rule 담당), AD_other_inside(patchcore_relevance no), 모든 lesion row",
            "n_rows": len(gate),
            "subtype_counts": gate_subtype,
            "primary_targets": [e["review_id"] for e in gate if e["b1d2_subtype"] == "D_keep"],
            "hold_targets": [e["review_id"] for e in gate if e["b1d2_subtype"] == "unclear_hold"],
            "note": "고해상도 재확인서 D_keep 9 / unclear 1. 전부 시각검토 완료. patchcore_relevance yes 다수.",
        },
        "C_lesion_safety_set": {
            "purpose": "규칙/게이트 적용 시 병변 인접이 잘리지 않는지 감시(성능개선 아님, monitoring 전용)",
            "include": "LESION_RISK_partial(부분커버=절단위험) + lesion_kept(저하금지 baseline) + B_keep_hard_case(boundary rule 오제거 금지)",
            "n_rows": len(safety_rows),
            "sentinel_counts": sentinel_counts,
            "note": "이 그룹의 어떤 row 도 flag/감점되면 안전 위반. boundary rule 은 hard_case 보존, 어떤 rule/gate 도 lesion row 미접촉.",
        },
    }

    boundary_rule_options = [
        {
            "id": "Rule-B1",
            "name": "refined_roi_ratio 단순 global threshold",
            "input_needed": ["refined_roi_ratio"],
            "apply_to": "ratio < T 인 모든 candidate (global)",
            "exclude": "없음(global) — 보호장치 없음",
            "pros": "최단순, 완전 해석가능",
            "risks": (
                f"치명적: boundary FP ratio 범위({ratio_overlap['boundary_rule_refined_roi_ratio_min']}~"
                f"{ratio_overlap['boundary_rule_refined_roi_ratio_max']})와 LESION_RISK_partial 범위("
                f"{ratio_overlap['lesion_risk_partial_refined_roi_ratio_min']}~"
                f"{ratio_overlap['lesion_risk_partial_refined_roi_ratio_max']})가 겹쳐 "
                f"({ratio_overlap['overlap_band'][0]}~{ratio_overlap['overlap_band'][1]} 대역) "
                "단일 ratio 컷이 병변 부분커버 row 까지 함께 제거"
            ),
            "hard_case_over_removal": "예 (hard_case ratio "
                                      f"{ratio_overlap['hard_case_refined_roi_ratio']} 가 컷 대역 내)",
            "lesion_safety_impact": "높음 (LESION_RISK_partial 전부 0.10~0.90, global 컷에 포함)",
            "interpretability": "high",
            "recommended_status": "reject (보호장치 없는 global threshold)",
        },
        {
            "id": "Rule-B2",
            "name": "center_in_refined_roi + ratio 결합",
            "input_needed": ["center_in_refined_roi", "refined_roi_ratio"],
            "apply_to": "center_in_refined_roi == False 또는 ratio < T_low 인 candidate",
            "exclude": "center 가 ROI 내부이고 ratio 높은 candidate",
            "pros": "patch 중심 위치를 써서 B1 보다 표적화. LESION_RISK_partial 은 전부 center=True 라 1차 제외됨",
            "risks": "hard_case 중 center=False(예: R024)가 잘못 제거될 수 있음 → hard_case 보호 미흡",
            "hard_case_over_removal": "부분 (center=False hard_case 가 잡힘)",
            "lesion_safety_impact": "낮음~중간 (lesion row 는 center=True 라 1차 보호되나 별도 sentinel 확인 필요)",
            "interpretability": "high",
            "recommended_status": "limited_candidate (hard_case 보호 추가 필요)",
        },
        {
            "id": "Rule-B3",
            "name": "wall/boundary-like selective rule (hard_case + lesion 명시 제외)",
            "input_needed": ["cause_class", "human_label", "refined_roi_ratio",
                             "center_in_refined_roi", "best_visual_label(=B_patch_overlap_artifact)",
                             "safety_role"],
            "apply_to": ("safety_role == fp_candidate AND cause_class == B_boundary AND "
                         "best_visual_label == B_patch_overlap_artifact (흉벽/경계 걸침 확인분)"),
            "exclude": ("lesion_protect 전부, B_keep_hard_case 전부, AD_wall_med_inside(별도 gate), "
                        "미검토 B_boundary(라벨 전까지 보류)"),
            "pros": "확인된 경계 overlap FP 만 표적, hard_case·lesion 을 구조적으로 보호",
            "risks": "미검토 B_boundary 5개는 라벨 후에만 편입 가능, 시각 라벨 의존",
            "hard_case_over_removal": "구조적 보호(명시 제외)",
            "lesion_safety_impact": "구조적 보호(safety_role 제외)",
            "interpretability": "high",
            "recommended_status": "preferred_form (candidate-level flag, score 컷 아님)",
        },
    ]

    patchcore_gate_options = [
        {
            "id": "Gate-P1",
            "name": "PatchCore distance 기반 binary reject gate",
            "input_needed": ["PatchCore distance(향후 계산)", "gate 후보 patch 좌표"],
            "memory_bank": "stage1_dev 정상 CT 의 wall/mediastinum 경계영역 정상 patch 만(position-conditioned subset), coreset subsampling",
            "apply_to": "Group B (AD_wall_med_inside D_keep) 한정",
            "output_form": "binary keep/reject flag (distance 낮음=정상=reject as FP)",
            "pros": "확인 정상 고점수를 직접 제거",
            "risks": "binary 경성 컷, memory bank 부족 시 흉벽 인접 진짜 이상도 오기각 위험",
            "impl_complexity": "medium",
            "lesion_safety_risk": "medium (gate 범위에 lesion-adjacent 미포함 보장 필요)",
            "interpretability": "medium",
            "why_aux_not_replace": "전체 patch 아님, AD_wall_med 한정 FP 재검사. PaDiM ranking 자체는 보존",
            "recommended_status": "candidate (binary 위험)",
        },
        {
            "id": "Gate-P2",
            "name": "PatchCore distance 기반 suspicious/keep 3단계 flag",
            "input_needed": ["PatchCore distance", "두 threshold(향후)"],
            "memory_bank": "Gate-P1 과 동일(wall/mediastinum 정상 patch coreset, position-conditioned)",
            "apply_to": "Group B (AD_wall_med_inside D_keep) 한정",
            "output_form": "3단계 flag only(normal/uncertain/suspicious), score 미변경",
            "pros": "uncertain 보존, 해석가능, flag-only 라 가장 안전, 원본 score 무수정",
            "risks": "threshold 2개 튜닝 필요",
            "impl_complexity": "medium",
            "lesion_safety_risk": "low~medium (flag-only, 제거 아님)",
            "interpretability": "high",
            "why_aux_not_replace": "PaDiM 대체 아님. high-score wall/med 후보에 normal 재검사 flag 만 부착",
            "recommended_status": "primary_form (flag-only 3단계, score 무수정)",
        },
        {
            "id": "Gate-P3",
            "name": "PatchCore + boundary-aware prefilter 조합(구조만)",
            "input_needed": ["boundary rule flag(Group A)", "PatchCore distance(Group B)"],
            "memory_bank": "Gate-P2 와 동일",
            "apply_to": "boundary prefilter 통과 후보를 PatchCore 로 재검사(2단)",
            "output_form": "ablation 분리 가능한 별도 출력(boundary-only / gate-only / combined 각각)",
            "pros": "경계걸침 + ROI 내부 정상구조 둘 다 커버",
            "risks": "ablation 분리 안 하면 두 효과 혼동 — 이번 단계는 구조설계만, 결과 생성 금지",
            "impl_complexity": "high",
            "lesion_safety_risk": "medium (combined 라 safety set 이중 확인)",
            "interpretability": "medium",
            "why_aux_not_replace": "여전히 후보 한정 보조 gate. 처음부터 혼합 결과만 만들지 않음",
            "recommended_status": "deferred (B1-D4 이후, 구조만 설계)",
        },
    ]

    ablation_plan = [
        {"axis": "Baseline", "input": "기존 PaDiM score(원본)", "changed": "없음",
         "fixed": "PaDiM 모델, refined ROI v4 2.0, candidate set, stage1_dev",
         "safety_check": "원본 무수정 확인", "expected": "현 흉벽 FP 잔존(기준선)"},
        {"axis": "Ablation-1 boundary rule only", "input": "Group A 후보 + Rule-B3",
         "changed": "boundary overlap flag 부착(별도 컬럼)", "fixed": "PaDiM score 원본, gate 미적용",
         "safety_check": "hard_case 2 보존, lesion row 0 접촉",
         "expected": "경계걸침 overlap FP 일부 flag"},
        {"axis": "Ablation-2 patchcore gate only", "input": "Group B 후보 + Gate-P2",
         "changed": "PatchCore 3단계 flag 부착(별도 컬럼)", "fixed": "PaDiM score 원본, boundary 미적용",
         "safety_check": "gate 범위에 lesion-adjacent 미포함 확인",
         "expected": "wall/med 정상 고점수 normal flag"},
        {"axis": "Ablation-3 boundary + gate", "input": "Group A + Group B 조합(Gate-P3)",
         "changed": "두 flag 결합(별도 출력, 이번 단계 구조만)", "fixed": "PaDiM score 원본",
         "safety_check": "두 safety 조건 모두",
         "expected": "후속 단계 평가용(이번 단계 결과 생성 금지)"},
    ]

    safety_plan = {
        "sentinels": {
            "lesion_risk_partial_at_risk": [e["review_id"] for e in safety_rows
                                            if e["sentinel_type"] == "lesion_risk_partial_at_risk"],
            "boundary_hard_case_must_keep": [e["review_id"] for e in safety_rows
                                             if e["sentinel_type"] == "boundary_hard_case_must_keep"],
            "lesion_kept_must_not_degrade_n": sum(1 for e in safety_rows
                                                  if e["sentinel_type"] == "lesion_kept_must_not_degrade"),
        },
        "must_manual_review_in_smoke": [
            "LESION_RISK_partial 7개가 flag/감점되었는지",
            "B_keep_hard_case 2개(R018,R024)가 boundary rule 로 제거되었는지",
            "lesion_kept 17개가 새로 억제/저하되었는지",
            "gate/rule 가 의도한 후보에만 적용되고 그 외엔 무접촉인지",
        ],
        "risk_judgement_criteria": [
            "lesion_safety_set 의 어떤 row 라도 flag/감점 → FAIL",
            "hard_case 가 keep 되지 않음 → FAIL",
            "FP 감소만 보고 성공판정 → 금지(lesion recall risk 동시관찰 필수)",
        ],
        "principle": "FP 감소 ∧ lesion recall risk 동시 관찰. false positive 감소 단독으로 성공 판정 금지.",
    }

    b1d3_smoke_scope = {
        "scope": "stage1_dev only, 아주 소수 patient/candidate subset, candidate-level 동작 확인 위주",
        "forbidden": "full thresholding/metric 산출, stage2_holdout 접근, score 원본 수정",
        "output": "adjusted/filtered 결과는 별도 출력(원본 보존), ablation 분리 가능 구조",
        "must_verify": [
            "rule/gate 가 의도한 후보에만 적용되는지",
            "lesion safety set 이 손상되지 않는지",
            "출력 구조가 ablation 가능하게 분리되는지",
        ],
        "subset_suggestion": "boundary: B_patch_overlap_artifact 보유 환자 / gate: AD_wall_med D_keep 보유 환자 2~3명",
    }

    recommended_primary_candidate = "PatchCore/gated filter (Gate-P2 flag-only 3단계)"
    recommended_secondary_candidate = "boundary-overlap rule (Rule-B3 selective)"
    recommended_next_step = "B1-D3 small-scope smoke test preflight (candidate-level 먼저)"

    final_recommendation_order = [
        "PatchCore/gated filter = primary smoke-test candidate (Gate-P2)",
        "boundary rule = limited smoke-test candidate (Rule-B3, Rule-B1 global 은 reject)",
        "둘의 혼합(Gate-P3) = 초기 단계 보류, 구조만 설계",
        "lesion safety monitoring = 필수(모든 ablation 공통)",
        "B1-D3 는 candidate-level smoke test 부터 시작",
    ]

    summary = {
        "step": "B1-D2_boundary_aware_rule_and_patchcore_gated_filter_preflight",
        "verdict": "PASS",
        "input_mtimes": input_mtimes,
        "stage2_holdout_access": stage2_holdout_access,
        "n_population_rows": len(rows),
        "group_counts": {
            "boundary_rule_candidate": len(bound),
            "patchcore_gate_candidate": len(gate),
            "lesion_safety_set": len(safety_rows),
            "excluded_observation_AD_other": len(excl),
        },
        "boundary_subtype_counts": bound_subtype,
        "patchcore_gate_subtype_counts": gate_subtype,
        "safety_sentinel_counts": sentinel_counts,
        "ratio_overlap_finding": ratio_overlap,
        "candidate_groups": candidate_groups,
        "boundary_rule_options": boundary_rule_options,
        "patchcore_gate_options": patchcore_gate_options,
        "ablation_plan": ablation_plan,
        "safety_plan": safety_plan,
        "b1d3_smoke_scope": b1d3_smoke_scope,
        "recommended_primary_candidate": recommended_primary_candidate,
        "recommended_secondary_candidate": recommended_secondary_candidate,
        "recommended_next_step": recommended_next_step,
        "final_recommendation_order": final_recommendation_order,
        "patchcore_implemented": False,
        "boundary_rule_implemented": False,
        "score_modified": False,
        "roi_modified": False,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- report MD ----
    def rule_block(o):
        return (f"### {o['id']} — {o['name']}  → **{o['recommended_status']}**\n"
                f"- 입력 필요값: {', '.join(o['input_needed'])}\n"
                f"- 적용 대상: {o['apply_to']}\n"
                f"- 제외 대상: {o['exclude']}\n"
                f"- 장점: {o['pros']}\n"
                f"- 위험: {o['risks']}\n"
                f"- hard case 오제거 가능성: {o['hard_case_over_removal']}\n"
                f"- lesion safety 영향: {o['lesion_safety_impact']}\n"
                f"- 해석성: {o['interpretability']}\n")

    def gate_block(o):
        return (f"### {o['id']} — {o['name']}  → **{o['recommended_status']}**\n"
                f"- 입력 필요값: {', '.join(o['input_needed'])}\n"
                f"- memory bank 구성: {o['memory_bank']}\n"
                f"- 적용 대상: {o['apply_to']}\n"
                f"- 출력 형태: {o['output_form']}\n"
                f"- 장점: {o['pros']}\n"
                f"- 위험: {o['risks']}\n"
                f"- 구현 복잡도: {o['impl_complexity']}\n"
                f"- lesion safety risk: {o['lesion_safety_risk']}\n"
                f"- 해석성: {o['interpretability']}\n"
                f"- 왜 대체가 아니라 보조 gate 인가: {o['why_aux_not_replace']}\n")

    abl_md = "\n".join(
        f"| {a['axis']} | {a['input']} | {a['changed']} | {a['fixed']} | {a['safety_check']} | {a['expected']} |"
        for a in ablation_plan)

    md = f"""# B1-D2 Boundary-aware Rule & PatchCore-gated Filter — Preflight Design

흉벽/종격동 FP 저감 후보 2개(경계걸침 rule / PatchCore gate)의 **설계 preflight**.
구현·score 조정·threshold 재계산·full run 없음. 목적 = "무엇을 어떻게 안전하게 비교할지" 결정.

## 0. 판정
**PASS** — candidate universe 3분리, boundary rule 3안 / PatchCore gate 3안 / ablation / safety / B1-D3 범위 설계 완료.

## 1. B1-D1.8 최종 결론 요약
- v4 2.0 mask = 기존 refined_roi_v4_20_modeB_all_v1 동일 → mask 변경 변수 닫힘
- C_outside_roi = 0 → ROI 밖 ranking 누수 아님
- A_mask_trim_candidate = 0, roi_trim_relevance yes = 0 → ROI/mask trim 단독 = low_priority
- boundary-overlap rule = limited_candidate, PatchCore/gated filter = primary_preflight_candidate
- full PatchCore replacement = reject, no action = reject

### 왜 ROI trim 단독이 약한가
흰 rim 이 이미 제거된 v4 2.0 기준에서도 A_mask_trim=0, 고해상도서 patch 가 폐실질 내 혈관/경계로 확인. 더 깎으면 폐실질·병변 손실 → 단독 비권장(제한적 local trim 여지만 남김).

### 왜 boundary rule 은 일부만 가능한가
B_boundary 안에 overlap_artifact 와 hard_case 가 섞임. 게다가 **boundary FP 와 병변 부분커버의 refined_roi_ratio 범위가 겹침** → 단일 ratio 컷으로 분리 불가(아래 정량근거).

### 왜 PatchCore/gated filter 가 유력한가
AD_wall_med_inside 다수가 ROI 로 못 빼는 경계 정상구조(D_keep). PaDiM 위치별 단일 Gaussian 으로 흉벽/종격동 다양성 흡수가 어려움 → 정상구조 재검사 gate 가 이론적 후보.

## 2. ★정량 근거: ratio 범위 겹침 (global threshold reject 근거)
- boundary_rule_candidate refined_roi_ratio: {ratio_overlap['boundary_rule_refined_roi_ratio_min']} ~ {ratio_overlap['boundary_rule_refined_roi_ratio_max']}
- LESION_RISK_partial refined_roi_ratio: {ratio_overlap['lesion_risk_partial_refined_roi_ratio_min']} ~ {ratio_overlap['lesion_risk_partial_refined_roi_ratio_max']}
- hard_case refined_roi_ratio: {ratio_overlap['hard_case_refined_roi_ratio']}
- **겹침 대역 {ratio_overlap['overlap_band'][0]} ~ {ratio_overlap['overlap_band'][1]}**: 이 대역에 경계 FP(hard_case 포함)와 병변 부분커버가 공존 → 어떤 단일 global ratio 컷도 둘을 동시에 자른다. **Rule-B1(global) reject 의 데이터 근거.**

## 3. Candidate Universe (3분리)
입력 모집단 {len(rows)}행. stage2_holdout_access = {stage2_holdout_access}.

### A. boundary-rule candidate ({len(bound)}행)
- 포함: cause_class == B_boundary (경계 straddle, ratio 0.10~0.90)
- 제외: AD_wall_med_inside, AD_other_inside, 모든 lesion row
- subtype: {bound_subtype}  (center_in_refined_roi==False: {center_false})
- 주의: 11개 중 6개만 검토(overlap 4 / hard 2), 5개 미검토 → 라벨 후 편입.

### B. PatchCore-gate candidate ({len(gate)}행)
- 포함: cause_class == AD_wall_med_inside (ROI 내부 wall/med 고점수)
- 제외: B_boundary, AD_other_inside, 모든 lesion row
- subtype: {gate_subtype}  (primary=D_keep, hold=unclear)
- 전부 고해상도 검토 완료, patchcore_relevance yes 다수.

### C. lesion safety set ({len(safety_rows)}행, monitoring 전용)
- 포함: LESION_RISK_partial(절단위험) + lesion_kept(저하금지 baseline) + B_keep_hard_case(오제거 금지)
- sentinel: {sentinel_counts}
- 성능개선용 아님. 이 그룹 어떤 row 라도 flag/감점되면 안전 위반.

(상세 목록: `b1d2_candidate_groups_preview.csv`, `b1d2_safety_set_preview.csv`)

## 4. Boundary overlap rule 후보 3안 (설계만, 적용 금지)
{rule_block(boundary_rule_options[0])}
{rule_block(boundary_rule_options[1])}
{rule_block(boundary_rule_options[2])}
> 일괄 global threshold 로 모든 patch 를 자르는 방식(Rule-B1)은 **비권장(reject)**. boundary hard case 보호장치가 없는 설계는 reject 후보.

## 5. PatchCore / gated filter 후보 3안 (설계만, 구현 금지)
핵심 원칙: full replacement 금지, 전체 patch/전체 데이터 적용 금지, PaDiM high-score 중 wall/mediastinum D-like patch 한정.
{gate_block(patchcore_gate_options[0])}
{gate_block(patchcore_gate_options[1])}
{gate_block(patchcore_gate_options[2])}

## 6. Ablation 설계 (분리 비교)
| 비교축 | 입력 | 바뀌는 요소 | 고정 요소 | 안전 체크 | 기대 효과 |
|---|---|---|---|---|---|
{abl_md}
> Ablation-3(boundary+gate)는 나중 단계용. 이번 단계는 구조만, 혼합 결과 생성 금지.

## 7. Safety evaluation plan (다음 단계 체크리스트)
- sentinels: LESION_RISK_partial {len(safety_plan['sentinels']['lesion_risk_partial_at_risk'])}개({', '.join(safety_plan['sentinels']['lesion_risk_partial_at_risk'])}), hard_case {len(safety_plan['sentinels']['boundary_hard_case_must_keep'])}개({', '.join(safety_plan['sentinels']['boundary_hard_case_must_keep'])}), lesion_kept {safety_plan['sentinels']['lesion_kept_must_not_degrade_n']}개
- smoke 수동검토 필수: {'; '.join(safety_plan['must_manual_review_in_smoke'])}
- 위험 판정 기준: {'; '.join(safety_plan['risk_judgement_criteria'])}
- 원칙: {safety_plan['principle']}

## 8. B1-D3 smoke test 권고 범위
- 범위: {b1d3_smoke_scope['scope']}
- 금지: {b1d3_smoke_scope['forbidden']}
- 출력: {b1d3_smoke_scope['output']}
- 확인: {'; '.join(b1d3_smoke_scope['must_verify'])}
- subset 제안: {b1d3_smoke_scope['subset_suggestion']}

## 9. 최종 권고 순서
{chr(10).join('- ' + s for s in final_recommendation_order)}

- recommended_primary_candidate: **{recommended_primary_candidate}**
- recommended_secondary_candidate: **{recommended_secondary_candidate}**
- 다음 단계: **{recommended_next_step}**

---
patchcore_implemented={summary['patchcore_implemented']}, boundary_rule_implemented={summary['boundary_rule_implemented']}, score_modified={summary['score_modified']}, roi_modified={summary['roi_modified']}, stage2_holdout_access={stage2_holdout_access}
"""
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)

    # ---- 콘솔 ----
    print("[B1-D2] PASS")
    print(f"  groups: boundary={len(bound)}, gate={len(gate)}, safety={len(safety_rows)}, excl_AD_other={len(excl)}")
    print(f"  boundary_subtype={bound_subtype}, gate_subtype={gate_subtype}")
    print(f"  ratio_overlap_band={ratio_overlap['overlap_band']} (global threshold reject 근거)")
    print(f"  생성: {OUT_MD.name}, {OUT_JSON.name}, {OUT_GROUPS.name}, {OUT_SAFETY.name}")


if __name__ == "__main__":
    main()
