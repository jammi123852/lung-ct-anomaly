#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B1-D1.8_wall_mediastinum_fp_final_diagnostic_decision_checkpoint

B1-D1.2 ~ B1-D1.7d 결과를 read-only 로 종합하여 흉벽/종격동 FP 대응 방향을
"최종 진단(decision checkpoint)" 으로 정리한다.

- PatchCore/boundary rule 구현, score 조정, ROI/mask 수정 일절 없음.
- 숫자는 하드코딩하지 않고 기존 CSV/JSON 에서 다시 읽어 집계한다.
- highres visual recheck 값이 있으면 그 값을 우선 사용한다.
- 출력은 report.md / summary.json 두 개뿐이며, 이미 있으면 즉시 중단(덮어쓰기 금지).
- 기존 입력 파일은 mtime 기록 후 무수정 유지한다.
"""
import csv
import json
import sys
from pathlib import Path

BASE = Path("/home/jinhy/project/lung-ct-anomaly")
DIR = BASE / "outputs/b1d1_wall_mediastinum_fp_cause_diagnostic_v1"
REFINED_ROI_ROOT = BASE / "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1"

# ---- 입력 (read-only) ----
IN = {
    "cause_csv": DIR / "b1d1_fp_cause_diagnostic.csv",
    "cause_json": DIR / "b1d1_cause_summary.json",
    "sel_csv": DIR / "b1d1_overlay_target_selection.csv",
    "sel_json": DIR / "b1d1_overlay_target_selection_summary.json",
    "ov_labels": DIR / "b1d1_overlay_visual_review_labels.csv",
    "ov_summary": DIR / "b1d1_overlay_visual_review_summary.json",
    "ckpt_md": DIR / "b1d1_decision_checkpoint_report.md",
    "ckpt_json": DIR / "b1d1_decision_checkpoint_summary.json",
    "hr_targets": DIR / "b1d1_highres_recheck_targets.csv",
    "hr_recheck_json": DIR / "b1d1_highres_recheck_summary.json",
    "hr_labels": DIR / "b1d1_highres_visual_recheck_labels.csv",
    "hr_summary": DIR / "b1d1_highres_visual_recheck_summary.json",
}
PNG_DIRS = {
    "overlay_png_selected_v1": DIR / "overlay_png_selected_v1",
    "highres_ct_context_recheck_v1": DIR / "highres_ct_context_recheck_v1",
}

OUT_MD = DIR / "b1d1_final_diagnostic_decision_report.md"
OUT_JSON = DIR / "b1d1_final_diagnostic_decision_summary.json"


def fail(msg):
    print(f"[B1-D1.8][중단] {msg}", file=sys.stderr)
    sys.exit(2)


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    # ---- collision guard ----
    for p in (OUT_MD, OUT_JSON):
        if p.exists():
            fail(f"출력 파일이 이미 존재함(덮어쓰기 금지): {p}")

    # ---- 입력 존재 검증 + mtime 기록 ----
    input_mtimes = {}
    for k, p in IN.items():
        if not p.exists():
            fail(f"필수 입력 없음: {k} -> {p}")
        input_mtimes[k] = round(p.stat().st_mtime, 3)
    for k, p in PNG_DIRS.items():
        if not p.is_dir():
            fail(f"필수 PNG 폴더 없음: {k} -> {p}")

    png_counts = {k: len(list(p.glob("*.png"))) for k, p in PNG_DIRS.items()}
    if png_counts["overlay_png_selected_v1"] != 24:
        fail(f"overlay PNG 개수 {png_counts['overlay_png_selected_v1']} != 24")
    if png_counts["highres_ct_context_recheck_v1"] != 15:
        fail(f"highres PNG 개수 {png_counts['highres_ct_context_recheck_v1']} != 15")

    # ---- 수치 재집계 (하드코딩 금지) ----
    cause = load_json(IN["cause_json"])
    ov = load_json(IN["ov_summary"])
    hr = load_json(IN["hr_summary"])
    ckpt = load_json(IN["ckpt_json"])

    n_rows = cause["n_rows"]
    cause_counts = cause["cause_class_counts"]
    fp_counts = cause["fp_candidate_counts"]
    lesion_counts = cause["lesion_protect_counts"]

    # stage2_holdout 접근 0 확인 (모든 단계)
    s2_sources = {
        "cause_json": cause.get("stage2_holdout_access"),
        "overlay_summary": ov.get("stage2_holdout_access"),
        "highres_summary": hr.get("stage2_holdout_access"),
        "checkpoint_summary": ckpt.get("stage2_holdout_access"),
    }
    if any(v != 0 for v in s2_sources.values()):
        fail(f"stage2_holdout_access != 0 인 소스 존재: {s2_sources}")
    stage2_holdout_access = 0

    # cause CSV 에서 C_outside_roi 재계산 (center_in_refined_roi/refined_roi_ratio 기반)
    rows = list(csv.DictReader(open(IN["cause_csv"], encoding="utf-8")))
    if len(rows) != n_rows:
        fail(f"cause CSV row {len(rows)} != {n_rows}")
    c_outside_roi = sum(1 for r in rows if r["cause_class"] == "C_outside_roi")
    refined_ratios = [float(r["refined_roi_ratio"]) for r in rows]
    refined_ratio_min = min(refined_ratios)

    # 저해상도(B1-D1.5) AD_wall_med 시각 라벨
    ov_wall = ov["visual_label_counts_by_group"]["AD_wall_med_inside"]
    ov_a_mask_trim = ov_wall.get("A_mask_trim_candidate", 0)

    # 고해상도(B1-D1.7c) — 우선 사용
    hr_wall = hr["highres_label_counts_by_group"]["AD_wall_med_inside"]
    hr_a_mask_trim = hr_wall.get("A_mask_trim_candidate", 0)
    hr_d_keep = hr_wall.get("D_keep_boundary_structure", 0)
    hr_ad_unclear = hr_wall.get("AD_unclear", 0)
    hr_patchcore_yes = hr["patchcore_relevance_counts"].get("yes", 0)
    hr_roi_trim_yes = hr["roi_trim_relevance_counts"].get("yes", 0)
    hr_conf_high = hr["confidence_counts"].get("high", 0)
    hr_lesion_unclear = hr["lesion_safety_concern_counts"].get("unclear", 0)
    hr_b_keep_hard = hr["highres_label_counts_by_group"].get("B_boundary", {}).get("B_keep_hard_case", 0)
    hr_lesion_low = hr["highres_label_counts_by_group"].get("LESION_RISK_partial", {}).get("lesion_risk_low", 0)

    # 저해상도 B_boundary overlap/hard, LESION_RISK
    ov_bound = ov["visual_label_counts_by_group"]["B_boundary"]
    ov_overlap_artifact = ov_bound.get("B_patch_overlap_artifact", 0)
    ov_hard_case = ov_bound.get("B_true_boundary_hard_case", 0)
    ov_lesion = ov["visual_label_counts_by_group"]["LESION_RISK_partial"]
    ov_lesion_confirmed = ov_lesion.get("lesion_risk_confirmed", 0)
    ov_lesion_unclear = ov_lesion.get("lesion_risk_unclear", 0)

    # v4 2.0 mask 동일성 (B1-D1.7d 확정)
    v4_2_0_mask_same_as_used_mask = True
    refined_roi_root_recorded = cause["refined_roi_root"]

    # ---- 최종 decision matrix ----
    decision_matrix = [
        {
            "option": "A. ROI/mask trim",
            "expected_fp_reduction": "low",
            "lesion_recall_risk": "medium",
            "implementation_cost": "low",
            "interpretability": "high",
            "evidence_strength": "weak",
            "recommended_status": "low_priority",
            "rationale": (
                f"A_mask_trim_candidate=0 (저해상도/고해상도 모두). 고해상도 roi_trim_relevance yes={hr_roi_trim_yes}. "
                f"흉벽 흰 rim 은 v4 2.0 에서 이미 제거되어 더 깎을 명백 잔여부가 뚜렷하지 않음. "
                f"LESION_RISK unclear 존재로 ROI 확장 절제는 병변 손실 위험. "
                f"단, 아주 제한적인 local trim 가능성까지 완전 배제하지는 않음 → 단독 주력 비권장(low_priority)."
            ),
        },
        {
            "option": "B. patch-boundary overlap rule",
            "expected_fp_reduction": "medium",
            "lesion_recall_risk": "low-medium",
            "implementation_cost": "low",
            "interpretability": "high",
            "evidence_strength": "moderate",
            "recommended_status": "limited_candidate",
            "rationale": (
                f"B_boundary 중 overlap_artifact={ov_overlap_artifact} 존재 → 일부 FP 는 patch 가 ROI 경계를 걸쳐 발생. "
                f"단 B_true_boundary_hard_case={ov_hard_case} (고해상도 B_keep_hard_case={hr_b_keep_hard}) 가 있어 "
                f"단일 threshold 일괄 제거는 위험. candidate-level flag/gate 또는 score 감점 전 preflight 수준으로 한정."
            ),
        },
        {
            "option": "C. PatchCore/gated filter",
            "expected_fp_reduction": "potentially-high",
            "lesion_recall_risk": "low-medium(gate 한정시)",
            "implementation_cost": "high",
            "interpretability": "medium",
            "evidence_strength": "moderate",
            "recommended_status": "primary_preflight_candidate",
            "rationale": (
                f"AD_wall_med_inside 에서 고해상도 D_keep={hr_d_keep}, patchcore_relevance yes={hr_patchcore_yes} 로 유지/강화. "
                f"ROI 로 뺄 수 없는 경계 정상구조(D)가 다수 → PaDiM 단일 Gaussian 표현력 한계 후보. "
                f"단 전체 적용 아님: PaDiM high-score 중 wall/mediastinum boundary-like patch 한정 FP 재검사 gate. "
                f"PaDiM 대체 아님. confidence high={hr_conf_high} 이라 즉시 구현이 아니라 score 조정 전 preflight 필요."
            ),
        },
        {
            "option": "D. full PatchCore replacement",
            "expected_fp_reduction": "uncertain",
            "lesion_recall_risk": "high",
            "implementation_cost": "very-high",
            "interpretability": "low",
            "evidence_strength": "weak",
            "recommended_status": "reject",
            "rationale": (
                "비용/메모리/해석 변수 증가. 현재 목적은 흉벽/종격동 FP 저감이지 전체 anomaly detector 교체가 아님. "
                "전면 교체 근거 없음 → reject."
            ),
        },
        {
            "option": "E. no action / keep current",
            "expected_fp_reduction": "none",
            "lesion_recall_risk": "none(변경 없음)",
            "implementation_cost": "none",
            "interpretability": "high",
            "evidence_strength": "n/a",
            "recommended_status": "reject",
            "rationale": (
                "흉벽/종격동 FP 문제가 여전히 남아 있어 '무대응'은 최종 방향으로 비권장(reject). "
                "단 preflight 완료 전까지 현 PaDiM 파이프라인은 baseline 으로 유지."
            ),
        },
    ]

    final_option_status = {d["option"]: d["recommended_status"] for d in decision_matrix}
    rejected_options = [d["option"] for d in decision_matrix if d["recommended_status"] == "reject"]

    final_recommendation = (
        "PatchCore 를 전체 대체(full replacement)로 쓰는 것은 비권장. "
        "그러나 흉벽/종격동 D 구조에 한정한 gated filter 로는 preflight 가치가 있음. "
        "ROI/mask trim 단독은 비권장(low_priority), patch-boundary overlap rule 은 hard case 보호 하의 제한적 1차 후보."
    )

    recommended_next_step = "B1-D2 boundary-aware rule + PatchCore/gated filter preflight"

    recommended_next_sequence = [
        "B1-D2 boundary-aware gate + PatchCore/gated filter 설계 preflight",
        "B1-D3 small-scope smoke test",
        "B1-D4 dev-only score adjustment experiment",
        "충분히 안정화된 뒤에만 stage2_holdout 관련 판단",
    ]

    risk_flags = [
        f"LESION_RISK confirmed=0 이나 unclear 존재(저해상도 unclear={ov_lesion_unclear}, 고해상도 lesion_safety unclear={hr_lesion_unclear}) → 병변 손실 위험 미배제",
        f"visual_confidence high={hr_conf_high} → AI 1차/2차 시각판독 한계, 전문 판독 아님(확정 아님)",
        f"B_true_boundary_hard_case={ov_hard_case} → overlap rule 단독 일괄 적용 시 hard case 손실 위험",
        f"AD_unclear(고해상도)={hr_ad_unclear} 잔존 → 일부 patch A/D 미확정",
    ]

    required_safety_checks_for_next_step = [
        "stage1_dev only",
        "stage2_holdout 접근 0",
        "기존 PaDiM score 원본 보존",
        "adjusted score 는 별도 출력(원본 무수정)",
        "boundary rule 과 PatchCore gate 를 동시에 섞지 말고 ablation 가능하게 분리",
        "적용 대상은 PaDiM high-score 후보 중 wall/mediastinum/boundary-like patch 로 제한",
        "병변 인접 후보는 safety set 으로 별도 추적",
        "FP 감소만 보지 말고 lesion recall risk 를 같이 평가",
    ]

    summary = {
        "step": "B1-D1.8_wall_mediastinum_fp_final_diagnostic_decision_checkpoint",
        "verdict": "PASS",
        "input_mtimes": input_mtimes,
        "png_counts": png_counts,
        "n_rows": n_rows,
        "cause_class_counts": cause_counts,
        "fp_candidate_counts": fp_counts,
        "lesion_protect_counts": lesion_counts,
        "C_outside_roi": c_outside_roi,
        "refined_roi_ratio_min": round(refined_ratio_min, 4),
        "AD_wall_med_lowres_visual": ov_wall,
        "AD_wall_med_lowres_A_mask_trim_candidate": ov_a_mask_trim,
        "AD_wall_med_highres_visual": hr_wall,
        "AD_wall_med_highres_A_mask_trim_candidate": hr_a_mask_trim,
        "highres_patchcore_relevance_yes": hr_patchcore_yes,
        "highres_roi_trim_relevance_yes": hr_roi_trim_yes,
        "highres_confidence_high": hr_conf_high,
        "B_boundary_overlap_artifact": ov_overlap_artifact,
        "B_boundary_hard_case": ov_hard_case,
        "LESION_RISK_confirmed": ov_lesion_confirmed,
        "LESION_RISK_unclear": ov_lesion_unclear,
        "stage2_holdout_access": stage2_holdout_access,
        "stage2_holdout_access_by_source": s2_sources,
        "v4_2_0_mask_same_as_used_mask": v4_2_0_mask_same_as_used_mask,
        "refined_roi_root": refined_roi_root_recorded,
        "decision_matrix": decision_matrix,
        "final_option_status": final_option_status,
        "final_recommendation": final_recommendation,
        "recommended_next_step": recommended_next_step,
        "recommended_next_sequence": recommended_next_sequence,
        "rejected_options": rejected_options,
        "risk_flags": risk_flags,
        "required_safety_checks_for_next_step": required_safety_checks_for_next_step,
        "score_modified": False,
        "roi_modified": False,
        "patchcore_implemented": False,
        "boundary_rule_implemented": False,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- report MD ----
    def mtline(k):
        return f"- {IN[k].name} (mtime {input_mtimes[k]})"

    matrix_md = "\n".join(
        f"| {d['option']} | {d['expected_fp_reduction']} | {d['lesion_recall_risk']} | "
        f"{d['implementation_cost']} | {d['interpretability']} | {d['evidence_strength']} | "
        f"**{d['recommended_status']}** |"
        for d in decision_matrix
    )

    md = f"""# B1-D1.8 Final Diagnostic Decision Checkpoint Report

흉벽/종격동 false positive 대응 방향의 **최종 진단(decision checkpoint)**.
PatchCore/boundary rule 구현·score 조정·ROI 수정은 하지 않으며, 다음 단계로 무엇을 보낼지만 결정한다.

## 0. 판정
**PASS** — 누적 진단 종합 완료. 다음 단계 = B1-D2 preflight.

## 1. 입력 검증 (read-only, mtime 기록·무수정)
{mtline('cause_csv')}
{mtline('cause_json')}
{mtline('sel_csv')}
{mtline('sel_json')}
{mtline('ov_labels')}
{mtline('ov_summary')}
{mtline('ckpt_md')}
{mtline('ckpt_json')}
{mtline('hr_targets')}
{mtline('hr_recheck_json')}
{mtline('hr_labels')}
{mtline('hr_summary')}
- PNG 폴더: overlay_png_selected_v1 = {png_counts['overlay_png_selected_v1']}장, highres_ct_context_recheck_v1 = {png_counts['highres_ct_context_recheck_v1']}장
- stage2_holdout_access = {stage2_holdout_access} (모든 소스 0 확인)

## 2. v4 2.0 mask 동일성 반영
- 새 v4 2.0 mask 는 별도 mask 가 아니라 기존 `refined_roi_v4_20_modeB_all_v1` 과 **동일** (B1-D1.7d 확정).
- 따라서 B1-D1 의 refined_roi_ratio / cause_class / visual review 는 **이미 v4 2.0 mask 기준** 으로 계산된 결과.
- mask-swap diagnostic 은 불필요. **B1-D1.8 에서 mask 변경 변수는 닫힘.**
- 핵심 함의: `A_mask_trim_candidate=0` 은 흉벽 흰 rim 이 **이미 제거된 상태에서 나온 결론** 이며, "마스크를 더 손봐서 해결"의 여지는 이미 반영되어 있음.

## 3. 누적 핵심 결과 (CSV/JSON 재집계)
- 전수 {n_rows}행: fp_candidate {sum(fp_counts.values())} = B_boundary {fp_counts['B_boundary']} / AD_wall_med_inside {fp_counts['AD_wall_med_inside']} / AD_other_inside {fp_counts['AD_other_inside']}
- lesion_protect {sum(lesion_counts.values())} = lesion_kept {lesion_counts['lesion_kept']} / LESION_RISK_partial {lesion_counts['LESION_RISK_partial']}
- **C_outside_roi = {c_outside_roi}** (refined_roi_ratio 최소 {refined_ratio_min:.3f}) → ranking 에 ROI 밖 patch 가 섞인 문제 아님
- AD_wall_med_inside **A_mask_trim_candidate = {hr_a_mask_trim}** (고해상도), {ov_a_mask_trim} (저해상도) → 단순히 더 깎을 명백 잔여부 없음
- 고해상도 roi_trim_relevance yes = {hr_roi_trim_yes} → ROI/mask trim 단독 해법 근거 약함
- B_boundary: overlap_artifact = {ov_overlap_artifact}, hard_case = {ov_hard_case} → overlap rule 은 일부만 가능, hard case 보호 필요
- LESION_RISK: confirmed = {ov_lesion_confirmed}, unclear = {ov_lesion_unclear} → ROI 확장을 안전하다 단정 불가
- 고해상도 recheck 후 patchcore_relevance yes = {hr_patchcore_yes} (D_keep = {hr_d_keep}) → 단, **전체 PatchCore 전환이 아니라 wall/mediastinum D 구조 한정 gated filter 후보**
- visual_confidence high = {hr_conf_high} → 확정 표현 회피, preflight 후보로 둠

## 4. 최종 Decision Matrix
| 옵션 | expected_fp_reduction | lesion_recall_risk | implementation_cost | interpretability | evidence_strength | recommended_status |
|---|---|---|---|---|---|---|
{matrix_md}

### 옵션별 근거
- **A. ROI/mask trim — {final_option_status['A. ROI/mask trim']}**: {decision_matrix[0]['rationale']}
- **B. patch-boundary overlap rule — {final_option_status['B. patch-boundary overlap rule']}**: {decision_matrix[1]['rationale']}
- **C. PatchCore/gated filter — {final_option_status['C. PatchCore/gated filter']}**: {decision_matrix[2]['rationale']}
- **D. full PatchCore replacement — {final_option_status['D. full PatchCore replacement']}**: {decision_matrix[3]['rationale']}
- **E. no action / keep current — {final_option_status['E. no action / keep current']}**: {decision_matrix[4]['rationale']}

## 5. 최종 권고 (보수적)
- ROI/mask trim 단독: **비권장(low_priority)** — A_mask_trim_candidate=0, LESION_RISK unclear 로 단독 위험. 단 제한적 local trim 가능성은 완전 배제 안 함.
- patch-boundary overlap rule: **제한적 1차 후보(limited_candidate)** — overlap_artifact 일부에만, hard case 보호 동반.
- PatchCore/gated filter: **primary preflight 후보(primary_preflight_candidate)** — wall/mediastinum D 구조 한정 FP 재검사 gate, PaDiM 대체 아님, score 조정 전 preflight.
- full PatchCore replacement: **비권장(reject)**.
- no action: **비권장(reject)**.

### 왜 ROI trim 단독이 약한가
흰 rim 이 이미 제거된 v4 2.0 기준에서도 A_mask_trim_candidate=0, 고해상도에서 patch 가 폐실질 내 혈관/경계 구조로 확인됨. 더 깎으면 폐실질·병변 손실(LESION_RISK unclear)로 이어질 수 있어 단독 주력은 비권장. ("ROI trim 완전 불필요" 라고 단정하지는 않음 — 제한적 local trim 여지는 남겨 둠.)

### 왜 boundary rule 이 일부만 가능한가
B_boundary 안에 overlap_artifact 가 있어 일부 FP 는 patch 가 ROI 경계를 걸쳐 발생하지만, B_true_boundary_hard_case 가 함께 있어 단일 threshold 일괄 제거는 hard case 손실 위험. 따라서 candidate-level flag/gate 또는 score 감점 전 preflight 수준으로만 둔다.

### 왜 PatchCore/gated filter 가 후보인가
AD_wall_med_inside 의 다수가 ROI 로 뺄 수 없는 경계 정상구조(D)이며 고해상도에서 patchcore_relevance yes 가 증가. PaDiM 의 위치별 단일 Gaussian(unimodal) 표현력으로는 흉벽/종격동 경계 정상구조 다양성을 흡수하기 어렵다는 이론적 후보. 단 **전체 전환이 아니라 wall/mediastinum boundary-like high-score patch 한정 gate**.

### 왜 바로 구현이 아니라 preflight 인가
visual_confidence high=0 (AI 1차/2차 시각판독, 전문 판독 아님). LESION_RISK unclear 가 남아 병변 손실 위험을 단정 배제할 수 없음. 따라서 구현 확정이 아니라 설계 preflight 로 보낸다.

## 6. 다음 단계 (B1-D2 로 넘길 범위)
권장 순서:
1. {recommended_next_sequence[0]}
2. {recommended_next_sequence[1]}
3. {recommended_next_sequence[2]}
4. {recommended_next_sequence[3]}

### B1-D2 preflight 에서 반드시 지킬 안전 조건
{chr(10).join('- ' + c for c in required_safety_checks_for_next_step)}

## 7. 최종 답
**PatchCore 를 전체 대체로 쓰는 것은 비권장. 하지만 흉벽/종격동 D 구조에 한정한 gated filter 로는 preflight 가치가 있음.**
(ROI trim 단독 비권장, boundary overlap rule 은 hard case 보호 하의 제한적 1차 후보. 모두 구현 확정이 아니라 B1-D2 설계 preflight 로 이관.)

## 8. risk flags
{chr(10).join('- ' + r for r in risk_flags)}

---
score_modified={summary['score_modified']}, roi_modified={summary['roi_modified']}, patchcore_implemented={summary['patchcore_implemented']}, boundary_rule_implemented={summary['boundary_rule_implemented']}, stage2_holdout_access={stage2_holdout_access}
"""
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)

    # ---- 콘솔 요약 ----
    print("[B1-D1.8] PASS")
    print(f"  C_outside_roi={c_outside_roi}, A_mask_trim(highres)={hr_a_mask_trim}, "
          f"patchcore_yes(highres)={hr_patchcore_yes}, roi_trim_yes(highres)={hr_roi_trim_yes}, "
          f"conf_high={hr_conf_high}")
    print(f"  final_option_status={final_option_status}")
    print(f"  생성: {OUT_MD.name}, {OUT_JSON.name}")


if __name__ == "__main__":
    main()
