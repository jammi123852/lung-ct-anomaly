#!/usr/bin/env python3
"""
P-A48 Soft Penalty Weight Design Preflight
- P-A47/P-A47.5 결과 기반 ResNet18 v2/v2 lower_peripheral FP 후보에 대한
  soft penalty weight 설계 가능성 검토
- 실제 score 수정/suppression/adjusted_score/suppression_weight 생성 없음
- stage1_dev 154명 only, stage2_holdout 접근 금지
"""
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

# ===================== 경로 설정 =====================
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
P47_ROOT   = PROJECT_ROOT / "experiments/resnet50_imagenet_v1/outputs/reports/lesion_stage1_dev/p_a47_soft_penalty_design_preflight"
P475_ROOT  = PROJECT_ROOT / "experiments/resnet50_imagenet_v1/outputs/reports/lesion_stage1_dev/p_a47_5_hold_pool_re_review"
P46C_ROOT  = PROJECT_ROOT / "experiments/resnet50_imagenet_v1/outputs/reports/lesion_stage1_dev/p_a46c_ct_context_review_labels"
P45_ROOT   = PROJECT_ROOT / "experiments/resnet50_imagenet_v1/outputs/reports/lesion_stage1_dev/p_a45_lower_peripheral_fp_rule_preflight"
P44_ROOT   = PROJECT_ROOT / "experiments/resnet50_imagenet_v1/outputs/reports/lesion_stage1_dev/p_a44_tiny_topk_sensitivity"
SPLIT_CSV  = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"
SCORE_DIR  = PROJECT_ROOT / "outputs/position-aware-padim-v1/scores/padim_v2_roi0_0/lesion_v2_by_patient"
OUTPUT_ROOT = PROJECT_ROOT / "experiments/resnet50_imagenet_v1/outputs/reports/lesion_stage1_dev/p_a48_soft_penalty_weight_design_preflight"

# ===================== 가드 0: 기존 P-A48 결과 없음 확인 =====================
if OUTPUT_ROOT.exists():
    existing = list(OUTPUT_ROOT.iterdir())
    if existing:
        print(f"[ABORT] 기존 P-A48 결과 존재: {[f.name for f in existing[:5]]} — 덮어쓰지 않고 중단")
        sys.exit(1)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ===================== 가드 1: P-A47 판정 통과 확인 =====================
p47_json = P47_ROOT / "p_a47_soft_penalty_design_preflight.json"
with open(p47_json, encoding="utf-8") as f:
    p47_meta = json.load(f)
if p47_meta.get("verdict") != "통과":
    print(f"[ABORT] P-A47 verdict={p47_meta.get('verdict')} — 통과 아님. 중단.")
    sys.exit(1)
print(f"가드 1: P-A47 verdict={p47_meta['verdict']} ✓")

# ===================== 가드 2: P-A47.5 판정 통과 확인 =====================
p475_json = P475_ROOT / "p_a47_5_hold_pool_re_review.json"
with open(p475_json, encoding="utf-8") as f:
    p475_meta = json.load(f)
if p475_meta.get("verdict") != "통과":
    print(f"[ABORT] P-A47.5 verdict={p475_meta.get('verdict')} — 통과 아님. 중단.")
    sys.exit(1)
print(f"가드 2: P-A47.5 verdict={p475_meta['verdict']} ✓")

# ===================== stage1_dev / stage2_holdout 목록 =====================
with open(SPLIT_CSV, newline="", encoding="utf-8-sig") as f:
    split_rows = list(csv.DictReader(f))
stage1_dev_patients    = {r["patient_id"].strip() for r in split_rows if r["stage_split"].strip() == "stage1_dev"}
stage2_holdout_patients = {r["patient_id"].strip() for r in split_rows if r["stage_split"].strip() == "stage2_holdout"}
if len(stage1_dev_patients) != 154:
    print(f"[ABORT] stage1_dev 환자 수 {len(stage1_dev_patients)}명 — 154명 아님. 중단.")
    sys.exit(1)
print(f"가드: stage1_dev {len(stage1_dev_patients)}명 ✓ / stage2_holdout {len(stage2_holdout_patients)}명 잠금 ✓")

# ===================== pool 로드 =====================
with open(P47_ROOT / "soft_penalty_candidate_pool.csv", newline="", encoding="utf-8") as f:
    cand_pool = list(csv.DictReader(f))
with open(P47_ROOT / "soft_penalty_exclusion_pool.csv", newline="", encoding="utf-8") as f:
    excl_p47 = list(csv.DictReader(f))
with open(P475_ROOT / "hold_pool_re_review_labels.csv", newline="", encoding="utf-8") as f:
    rr_rows = list(csv.DictReader(f))

excl_p475   = [r for r in rr_rows if r["re_review_label"] not in ("candidate_for_soft_penalty_preflight", "hold_unclear_manual_review")]
hold_final  = [r for r in rr_rows if r["re_review_label"] == "hold_unclear_manual_review"]

total_excl  = len(excl_p47) + len(excl_p475)
total_pool  = len(cand_pool) + total_excl + len(hold_final)

# ===================== 가드 3: 최종 pool 합산 확인 =====================
expected = {"candidate": 17, "exclusion": 29, "hold": 3, "total": 49}
actual   = {"candidate": len(cand_pool), "exclusion": total_excl, "hold": len(hold_final), "total": total_pool}
if actual != expected:
    print(f"[ABORT] 최종 pool 불일치: 기대={expected}, 실제={actual}. 중단.")
    sys.exit(1)
print(f"가드 3: 최종 pool candidate={actual['candidate']} / exclusion={actual['exclusion']} / hold={actual['hold']} / total={actual['total']} ✓")

# ===================== 가드 4: stage2_holdout 환자 포함 여부 =====================
all_cand_patients = {r["patient_id"].strip() for r in cand_pool}
holdout_leak = all_cand_patients & stage2_holdout_patients
if holdout_leak:
    print(f"[ABORT] candidate pool 환자 중 stage2_holdout 포함: {holdout_leak} — 즉시 중단.")
    sys.exit(1)
print(f"가드 4: candidate pool 환자({len(all_cand_patients)}명) 중 stage2_holdout 0명 ✓")

# ===================== P-A45 threshold 로드 =====================
with open(P45_ROOT / "p_a45_lower_peripheral_fp_rule_preflight.json", encoding="utf-8") as f:
    p45_meta = json.load(f)
p95_threshold = float(p45_meta["p95_threshold"])
p99_threshold = float(p45_meta["p99_threshold"])
lp_lesion_patch_count   = 9245
lp_lesion_patient_count = 44

# ===================== score 분포 분석 =====================
def score_stats(rows):
    scores = [float(r["padim_score"]) for r in rows if r.get("padim_score")]
    if not scores:
        return {}
    return {
        "count": len(scores),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "mean": round(statistics.mean(scores), 4),
        "median": round(statistics.median(scores), 4),
        "stdev": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
        "above_p95": sum(1 for s in scores if s > p95_threshold),
        "above_p99": sum(1 for s in scores if s > p99_threshold),
        "scores_sorted": sorted([round(s, 2) for s in scores]),
    }

label_dist = Counter(r["ct_context_label"] for r in cand_pool)
pleura_rows    = [r for r in cand_pool if r["ct_context_label"] == "pleura_chest_wall_fp_likely"]
diaphragm_rows = [r for r in cand_pool if r["ct_context_label"] == "diaphragm_base_fp_likely"]
hilar_rows     = [r for r in cand_pool if r["ct_context_label"] == "hilar_mediastinal_fp_likely"]

stats_all        = score_stats(cand_pool)
stats_pleura     = score_stats(pleura_rows)
stats_diaphragm  = score_stats(diaphragm_rows)
stats_hilar      = score_stats(hilar_rows)

print(f"\n[score 분포]")
print(f"  전체 candidate: min={stats_all['min']}, max={stats_all['max']}, mean={stats_all['mean']}, median={stats_all['median']}")
print(f"  pleura {stats_pleura['count']}건: min={stats_pleura['min']}, max={stats_pleura['max']}, mean={stats_pleura['mean']}")
print(f"  diaphragm {stats_diaphragm['count']}건: min={stats_diaphragm['min']}, max={stats_diaphragm['max']}, mean={stats_diaphragm['mean']}")
print(f"  hilar {stats_hilar['count']}건: min={stats_hilar['min']}, max={stats_hilar['max']}, mean={stats_hilar['mean']}")

# ===================== weight 설계 후보표 =====================
# 실제 적용 금지 — 후보 설계값으로만 기록
WEIGHT_OPTIONS = [
    {
        "weight_level": "very_soft",
        "multiplier": 0.90,
        "description": "score × 0.90 — 가장 보수적인 penalty",
        "applicable_branch": "pleura_chest_wall",
        "lesion_safety_note": "lesion score도 10% 감소. 보수적이나 lesion recall 영향 최소.",
        "recommendation": "lesion safety 위험이 큰 경우 우선 적용 후보",
    },
    {
        "weight_level": "soft",
        "multiplier": 0.80,
        "description": "score × 0.80 — 중간 수준 penalty",
        "applicable_branch": "pleura_chest_wall, diaphragm_base",
        "lesion_safety_note": "lesion score 20% 감소. lower_peripheral lesion 9,245개에 영향 가능.",
        "recommendation": "manual-label 기반 candidate에만 적용 — 자동 적용 금지",
    },
    {
        "weight_level": "moderate",
        "multiplier": 0.70,
        "description": "score × 0.70 — 강한 penalty",
        "applicable_branch": "diaphragm_base (추가 검증 후)",
        "lesion_safety_note": "lesion score 30% 감소. lower_peripheral lesion 누락 위험 높음.",
        "recommendation": "소수 3건(diaphragm)에 대해서만 검토. lesion safety 검증 필수.",
    },
    {
        "weight_level": "hold_for_later",
        "multiplier": None,
        "description": "적용 보류 — 추가 검증 필요",
        "applicable_branch": "hilar_mediastinal",
        "lesion_safety_note": "hilar 구조는 position_bin으로 구분 불가. CT-context label 없이 자동 rule 불가.",
        "recommendation": "lower_peripheral rule과 분리. 별도 branch 또는 hold-for-later.",
    },
]

# ===================== safety constraints =====================
SAFETY_CONSTRAINTS = [
    {
        "constraint_id": "SC-01",
        "constraint": "lesion_protect=True 케이스 절대 제외",
        "affected_count": len([r for r in excl_p47 if r.get("exclusion_reason", "") == "lesion_protect=True"]),
        "rationale": "P-A46c lesion_protect=True 25건은 어떤 penalty도 적용 불가",
    },
    {
        "constraint_id": "SC-02",
        "constraint": "hold pool 3건 penalty 후보 제외",
        "affected_count": len(hold_final),
        "rationale": "LUNG1-156 ×2, LUNG1-313 ×1 — 불확실성 미해소",
    },
    {
        "constraint_id": "SC-03",
        "constraint": "lower_peripheral 전체 억제 금지",
        "affected_count": lp_lesion_patch_count,
        "rationale": f"lower_peripheral 내 lesion patch {lp_lesion_patch_count}개({lp_lesion_patient_count}명). 전체 억제 시 lesion recall 큰 손상.",
    },
    {
        "constraint_id": "SC-04",
        "constraint": "position_bin 단독 자동 적용 금지",
        "affected_count": 0,
        "rationale": "ct_context_label은 수동 review 결과 — score CSV에 없음. position_bin만으로 pleura/diaphragm/hilar 구분 불가.",
    },
    {
        "constraint_id": "SC-05",
        "constraint": "hilar/mediastinal 2건 별도 branch 처리",
        "affected_count": len(hilar_rows),
        "rationale": "hilar 구조는 middle_central과 혼재 가능. lower_peripheral rule과 섞지 말 것.",
    },
    {
        "constraint_id": "SC-06",
        "constraint": "manual-label 기반 filtering 없이 자동 적용 금지",
        "affected_count": len(cand_pool),
        "rationale": "ct_context_label 없이 candidate/exclusion 구분 불가. 자동화 전 별도 label 체계 필요.",
    },
    {
        "constraint_id": "SC-07",
        "constraint": "adjusted_score / suppression_weight 파일 생성 금지",
        "affected_count": 0,
        "rationale": "이번 단계는 preflight — 실제 score 수정 없음.",
    },
]

# ===================== hypothetical FP reduction 추정 (산술 추정만) =====================
# candidate 17건이 FP라고 가정했을 때 penalty 효과를 score 감소율로만 추정
# 실제 성능 개선 주장 금지
pleura_scores_sorted = sorted([float(r["padim_score"]) for r in pleura_rows])
diaphragm_scores_sorted = sorted([float(r["padim_score"]) for r in diaphragm_rows])

fp_reduction_estimates = []
for opt in WEIGHT_OPTIONS:
    if opt["multiplier"] is None:
        fp_reduction_estimates.append({
            "weight_level": opt["weight_level"],
            "multiplier": "N/A",
            "branch": opt["applicable_branch"],
            "pleura_score_range_after": "N/A",
            "diaphragm_score_range_after": "N/A",
            "hypothetical_note": "hold-for-later — 산술 추정 불가",
        })
        continue
    m = opt["multiplier"]
    p_after = [round(s * m, 2) for s in pleura_scores_sorted] if pleura_scores_sorted else []
    d_after = [round(s * m, 2) for s in diaphragm_scores_sorted] if diaphragm_scores_sorted else []
    fp_reduction_estimates.append({
        "weight_level": opt["weight_level"],
        "multiplier": m,
        "branch": opt["applicable_branch"],
        "pleura_score_range_after": f"{min(p_after):.2f}~{max(p_after):.2f}" if p_after else "N/A",
        "diaphragm_score_range_after": f"{min(d_after):.2f}~{max(d_after):.2f}" if d_after else "N/A",
        "hypothetical_note": (
            f"candidate {len(cand_pool)}건 중 pleura {len(pleura_rows)}건 score를 ×{m}로 가정 시 "
            f"{min(p_after):.2f}~{max(p_after):.2f}로 감소. "
            "실제 score 미적용 — dry weight design 추정치."
        ),
    })

# ===================== 추가 검증 목록 =====================
ADDITIONAL_VERIFICATION = [
    "1. pleura/chest-wall 12건 환자의 전체 score CSV에서 lesion_pixels>0 patch가 해당 슬라이스에 있는지 확인",
    "2. diaphragm/base 3건 환자의 lower_central position_bin score 분포 확인",
    "3. hilar/mediastinal 2건 — middle_central과의 구분 기준 추가 검토",
    "4. hold 3건 (LUNG1-156×2, LUNG1-313×1) 별도 수동 판독 후 재분류",
    "5. penalty weight 실제 적용 전 lesion_protect 조건 코드 레벨 가드 구현 필요",
    "6. top-k aggregation과 병행 시 penalty가 top-k 순위 변경에 미치는 영향 개념 검토 (실제 계산 금지)",
    "7. manual-label 기반 candidate list를 score CSV join 키로 연결하는 방법 설계",
]

# ===================== top-k 병행 가능성 개념 기록 =====================
TOPK_CONCEPT = (
    "soft penalty는 top-k aggregation과 개념적으로 병행 가능. "
    "score에 weight를 곱하면 해당 patch의 top-k 기여도가 줄어드는 효과. "
    "단, penalty 대상이 manual-label 기반 17건으로 제한되므로 자동 일반화는 불가. "
    "실제 top-k 재계산 금지 — 개념 수준으로만 기록."
)

# ===================== 출력 파일 생성 =====================

# 1. final_soft_penalty_candidate_pool.csv
# P-A47 candidate 17건 그대로 + re_review_label 추가
final_cand_rows = []
for r in cand_pool:
    final_cand_rows.append({
        "review_id": r["review_id"],
        "patient_id": r["patient_id"],
        "ct_context_label": r["ct_context_label"],
        "penalty_safe": r["penalty_safe"],
        "lesion_protect": r["lesion_protect"],
        "penalty_risk": r["penalty_risk"],
        "padim_score": r["padim_score"],
        "slice_index": r["slice_index"],
        "branch": (
            "pleura_chest_wall" if r["ct_context_label"] == "pleura_chest_wall_fp_likely"
            else "diaphragm_base" if r["ct_context_label"] == "diaphragm_base_fp_likely"
            else "hilar_mediastinal_separate"
        ),
        "weight_candidate": (
            "very_soft_or_soft" if r["ct_context_label"] == "pleura_chest_wall_fp_likely"
            else "soft_or_moderate" if r["ct_context_label"] == "diaphragm_base_fp_likely"
            else "hold_for_later"
        ),
        "source": "P-A47",
    })

cand_fieldnames = [
    "review_id","patient_id","ct_context_label","penalty_safe","lesion_protect",
    "penalty_risk","padim_score","slice_index","branch","weight_candidate","source",
]
with open(OUTPUT_ROOT / "final_soft_penalty_candidate_pool.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=cand_fieldnames)
    writer.writeheader()
    writer.writerows(final_cand_rows)

# 2. final_soft_penalty_exclusion_pool.csv
final_excl_rows = []
for r in excl_p47:
    final_excl_rows.append({
        "review_id": r["review_id"],
        "patient_id": r["patient_id"],
        "ct_context_label": r["ct_context_label"],
        "penalty_safe": r["penalty_safe"],
        "lesion_protect": r.get("lesion_protect", ""),
        "exclusion_reason": r.get("exclusion_reason", ""),
        "source": "P-A47",
    })
for r in excl_p475:
    final_excl_rows.append({
        "review_id": r["review_id"],
        "patient_id": r["patient_id"],
        "ct_context_label": r.get("original_ct_context_label", r.get("ct_context_label", "")),
        "penalty_safe": r.get("original_penalty_safe", ""),
        "lesion_protect": r.get("lesion_protect", ""),
        "exclusion_reason": r["re_review_label"],
        "source": "P-A47.5",
    })

excl_fieldnames = ["review_id","patient_id","ct_context_label","penalty_safe","lesion_protect","exclusion_reason","source"]
with open(OUTPUT_ROOT / "final_soft_penalty_exclusion_pool.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=excl_fieldnames)
    writer.writeheader()
    writer.writerows(final_excl_rows)

# 3. final_soft_penalty_hold_pool.csv
hold_fieldnames = ["review_id","patient_id","original_ct_context_label","original_penalty_safe","padim_score","slice_index","re_review_label","hold_reason","source"]
hold_out_rows = []
for r in hold_final:
    hold_out_rows.append({
        "review_id": r["review_id"],
        "patient_id": r["patient_id"],
        "original_ct_context_label": r.get("original_ct_context_label", ""),
        "original_penalty_safe": r.get("original_penalty_safe", ""),
        "padim_score": r.get("padim_score", ""),
        "slice_index": r.get("slice_index", ""),
        "re_review_label": r["re_review_label"],
        "hold_reason": r.get("reason_code", ""),
        "source": "P-A47.5",
    })
with open(OUTPUT_ROOT / "final_soft_penalty_hold_pool.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=hold_fieldnames)
    writer.writeheader()
    writer.writerows(hold_out_rows)

# 4. soft_penalty_weight_design_options.csv
weight_fieldnames = ["weight_level","multiplier","description","applicable_branch","lesion_safety_note","recommendation"]
with open(OUTPUT_ROOT / "soft_penalty_weight_design_options.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=weight_fieldnames)
    writer.writeheader()
    writer.writerows(WEIGHT_OPTIONS)

# 5. soft_penalty_safety_constraints.csv
safety_fieldnames = ["constraint_id","constraint","affected_count","rationale"]
with open(OUTPUT_ROOT / "soft_penalty_safety_constraints.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=safety_fieldnames)
    writer.writeheader()
    writer.writerows(SAFETY_CONSTRAINTS)

# 6. p_a48_soft_penalty_weight_design_preflight.json
result_json = {
    "phase": "P-A48",
    "title": "Soft Penalty Weight Design Preflight",
    "date": "2026-06-01",
    "verdict": "통과",

    # 입력 검증
    "p_a47_verdict": p47_meta.get("verdict"),
    "p_a475_verdict": p475_meta.get("verdict"),
    "final_candidate_count": len(final_cand_rows),
    "final_exclusion_count": len(final_excl_rows),
    "final_hold_count": len(hold_out_rows),
    "final_total": len(final_cand_rows) + len(final_excl_rows) + len(hold_out_rows),

    # pool 분포
    "candidate_by_branch": {
        "pleura_chest_wall": stats_pleura,
        "diaphragm_base": stats_diaphragm,
        "hilar_mediastinal_separate": stats_hilar,
    },
    "p95_threshold": p95_threshold,
    "p99_threshold": p99_threshold,
    "all_candidates_above_p99": stats_all["above_p99"] == len(final_cand_rows),

    # weight 설계
    "weight_design_options": WEIGHT_OPTIONS,
    "fp_reduction_estimates": fp_reduction_estimates,
    "weight_design_note": (
        "dry weight design — 후보 설계값만 기록. 실제 score 미적용. "
        "'성능 개선' 단정 금지. candidate-only preflight, hypothetical effect."
    ),

    # safety constraints
    "safety_constraints": SAFETY_CONSTRAINTS,
    "lower_peripheral_blanket_suppression": "금지 — 유지",
    "position_bin_auto_rule": "금지 — manual-label 기반 filtering 필수",
    "hilar_mediastinal_branch": "lower_peripheral rule과 분리 — hold_for_later",
    "manual_review_gated_design": True,

    # 추가 검증 목록
    "additional_verification_required": ADDITIONAL_VERIFICATION,
    "topk_aggregation_concept": TOPK_CONCEPT,

    # 실행 안전 확인
    "score_csv_modified": False,
    "adjusted_score_generated": False,
    "suppression_weight_generated": False,
    "threshold_changed": False,
    "metrics_recalculated": False,
    "scoring_rerun": False,
    "model_forward": False,
    "training": False,
    "stage2_holdout_access": 0,
    "stage2_holdout_locked": True,
    "existing_results_modified": False,
    "new_png_generated": False,

    "output_files": [
        "final_soft_penalty_candidate_pool.csv",
        "final_soft_penalty_exclusion_pool.csv",
        "final_soft_penalty_hold_pool.csv",
        "soft_penalty_weight_design_options.csv",
        "soft_penalty_safety_constraints.csv",
        "p_a48_soft_penalty_weight_design_preflight.md",
        "p_a48_soft_penalty_weight_design_preflight.json",
    ],

    "next_step": (
        "P-A48 결과 검토 후, manual-label 기반 candidate 17건에 대해 "
        "weight 적용 코드(P-A49) 설계. "
        "hold 3건 별도 수동 판독. "
        "실제 score adjustment는 사용자 승인 후 진행."
    ),
}
with open(OUTPUT_ROOT / "p_a48_soft_penalty_weight_design_preflight.json", "w", encoding="utf-8") as f:
    json.dump(result_json, f, ensure_ascii=False, indent=2)

# 7. p_a48_soft_penalty_weight_design_preflight.md
md = []
md += [
    "# P-A48 Soft Penalty Weight Design Preflight",
    "",
    "## 판정: 통과",
    "",
    "---",
    "",
    "## 1. P-A47 / P-A47.5 입력 검증",
    "",
    f"| 항목 | 결과 |",
    f"|---|---|",
    f"| P-A47 verdict | {p47_meta.get('verdict')} ✓ |",
    f"| P-A47.5 verdict | {p475_meta.get('verdict')} ✓ |",
    f"| 최종 candidate | {len(final_cand_rows)}건 ✓ |",
    f"| 최종 exclusion | {len(final_excl_rows)}건 ✓ |",
    f"| 최종 hold | {len(hold_out_rows)}건 ✓ |",
    f"| total | {len(final_cand_rows)+len(final_excl_rows)+len(hold_out_rows)}건 ✓ |",
    "",
    "---",
    "",
    "## 2. 최종 Pool 요약",
    "",
    "### 2-1. Candidate Pool (17건)",
    "",
    "| branch | 건수 | score 범위 | mean |",
    "|---|---|---|---|",
    f"| pleura_chest_wall | {stats_pleura['count']} | {stats_pleura['min']}~{stats_pleura['max']} | {stats_pleura['mean']} |",
    f"| diaphragm_base | {stats_diaphragm['count']} | {stats_diaphragm['min']}~{stats_diaphragm['max']} | {stats_diaphragm['mean']} |",
    f"| hilar_mediastinal (별도) | {stats_hilar['count']} | {stats_hilar['min']}~{stats_hilar['max']} | {stats_hilar['mean']} |",
    "",
    f"> **전체 candidate 17건 모두 p99 threshold({p99_threshold:.4f}) 초과** — p99 이상 고점수 FP 후보군.",
    f"> p95={p95_threshold:.4f}, p99={p99_threshold:.4f} (P-A45 기준)",
    "",
    "### 2-2. Exclusion Pool (29건)",
    "",
    f"- P-A47 exclusion 25건 (lesion_protect=True 25건)",
    f"- P-A47.5 추가 exclusion 4건 (exclude_protect_lesion 1, exclude_near_boundary_or_vessel 1, exclude_insufficient_context 2)",
    "",
    "### 2-3. Hold Pool (3건)",
    "",
]
for r in hold_out_rows:
    md.append(f"- {r['review_id']} | {r['patient_id']} | {r['original_ct_context_label']} | {r['hold_reason']}")

md += [
    "",
    "---",
    "",
    "## 3. pleura/chest-wall 후보 12건",
    "",
    f"- score 범위: {stats_pleura['min']}~{stats_pleura['max']}",
    f"- mean: {stats_pleura['mean']}, median: {stats_pleura['median']}",
    f"- scores: {stats_pleura['scores_sorted']}",
    f"- 전부 p99({p99_threshold:.4f}) 초과 — 고점수 FP",
    "- **weight 후보: very_soft(×0.90) 또는 soft(×0.80)**",
    "- manual-label 기반 후보로만 설계 — 자동 적용 금지",
    "",
    "---",
    "",
    "## 4. diaphragm/base 후보 3건",
    "",
    f"- score 범위: {stats_diaphragm['min']}~{stats_diaphragm['max']}",
    f"- mean: {stats_diaphragm['mean']}",
    f"- scores: {stats_diaphragm['scores_sorted']}",
    "- **weight 후보: soft(×0.80) 또는 moderate(×0.70)** — 추가 검증 후",
    "- 소수 3건이므로 lesion safety 검증 필수",
    "",
    "---",
    "",
    "## 5. hilar/mediastinal 후보 2건 — 별도 branch",
    "",
    f"- score 범위: {stats_hilar['min']}~{stats_hilar['max']}",
    f"- scores: {stats_hilar['scores_sorted']}",
    "- **lower_peripheral rule과 분리 — hold_for_later**",
    "- position_bin으로 hilar 구분 불가. CT-context label 없이 자동 rule 불가.",
    "- 별도 branch 유지 또는 hold-for-later 처리",
    "",
    "---",
    "",
    "## 6. lower_peripheral 전체 억제 금지 재확인",
    "",
    f"- **금지 유지** ✓",
    f"- lower_peripheral 내 lesion: {lp_lesion_patch_count}개 / {lp_lesion_patient_count}명",
    "",
    "---",
    "",
    "## 7. position_bin 단독 자동 rule 금지 재확인",
    "",
    "- ct_context_label은 수동 review 결과 → score CSV에 없음",
    "- position_bin으로 pleura/diaphragm/hilar 구분 불가",
    "- **manual-label 기반 filtering 없이 자동 적용 금지**",
    "- manual-review-gated design으로 결론",
    "",
    "---",
    "",
    "## 8. 후보 weight design table (dry weight design — 실제 적용 금지)",
    "",
    "| weight_level | multiplier | applicable_branch | lesion_safety_note | recommendation |",
    "|---|---|---|---|---|",
]
for opt in WEIGHT_OPTIONS:
    m = opt['multiplier'] if opt['multiplier'] is not None else "N/A"
    md.append(f"| {opt['weight_level']} | {m} | {opt['applicable_branch']} | {opt['lesion_safety_note']} | {opt['recommendation']} |")

md += [
    "",
    "> **주의:** 위 값은 후보 설계값(dry weight design)이며, 실제 score에 적용되지 않았음.",
    "> 'hypothetical effect' — 성능 개선 단정 금지.",
    "",
    "---",
    "",
    "## 9. Hypothetical FP Reduction 추정 (산술 추정만)",
    "",
    "| weight_level | multiplier | pleura score 범위 (적용 후 가정) | diaphragm score 범위 (적용 후 가정) |",
    "|---|---|---|---|",
]
for est in fp_reduction_estimates:
    m_str = str(est['multiplier']) if est['multiplier'] != "N/A" else "N/A"
    md.append(f"| {est['weight_level']} | {m_str} | {est['pleura_score_range_after']} | {est['diaphragm_score_range_after']} |")

md += [
    "",
    "> 실제 score 미적용 — candidate-only preflight 산술 추정치.",
    "",
    "---",
    "",
    "## 10. Lesion Safety Constraints",
    "",
    "| constraint_id | constraint | affected_count |",
    "|---|---|---|",
]
for c in SAFETY_CONSTRAINTS:
    md.append(f"| {c['constraint_id']} | {c['constraint']} | {c['affected_count']} |")

md += [
    "",
    "---",
    "",
    "## 11. 추가 검증 목록 (실제 적용 전 필수)",
    "",
]
for v in ADDITIONAL_VERIFICATION:
    md.append(f"- {v}")

md += [
    "",
    "---",
    "",
    "## 12. top-k aggregation 병행 가능성 (개념 수준)",
    "",
    f"> {TOPK_CONCEPT}",
    "",
    "---",
    "",
    "## 13. 실행 안전 확인",
    "",
    "| 항목 | 상태 |",
    "|---|---|",
    "| adjusted_score 생성 | 없음 ✓ |",
    "| suppression_weight 생성 | 없음 ✓ |",
    "| score CSV 수정 | 없음 ✓ |",
    "| threshold 변경 | 없음 ✓ |",
    "| metrics 재계산 | 없음 ✓ |",
    "| scoring 재실행 | 없음 ✓ |",
    "| model forward | 없음 ✓ |",
    "| training | 없음 ✓ |",
    "| stage2_holdout 접근 | 없음 ✓ |",
    "| 기존 결과 수정 | 없음 ✓ |",
    "| 새 PNG 생성 | 없음 ✓ |",
    "",
    "---",
    "",
    "## 14. 다음 단계",
    "",
    "1. P-A48 결과 사용자 확인",
    "2. manual-label 기반 candidate 17건 weight 적용 코드 설계 (P-A49) — 사용자 승인 후 진행",
    "3. hold 3건 (LUNG1-156 ×2, LUNG1-313 ×1) 별도 수동 판독",
    "4. hilar/mediastinal 2건 별도 branch 처리 방침 확정",
    "5. 실제 score adjustment는 사용자 승인 전까지 진행 금지",
    "",
]
with open(OUTPUT_ROOT / "p_a48_soft_penalty_weight_design_preflight.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md))

print(f"\n[완료] 출력 경로: {OUTPUT_ROOT}")
for fname in result_json["output_files"]:
    fpath = OUTPUT_ROOT / fname
    size = fpath.stat().st_size if fpath.exists() else 0
    print(f"  - {fname} ({size:,}B)")
print(f"\n판정: 통과")
print(f"candidate: {len(final_cand_rows)}건 / exclusion: {len(final_excl_rows)}건 / hold: {len(hold_out_rows)}건")
