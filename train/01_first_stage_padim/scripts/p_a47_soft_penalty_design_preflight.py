#!/usr/bin/env python3
"""
P-A47 Soft Penalty Design Preflight
- P-A46c CT-context review labels 기반 soft penalty 설계 타당성 검토
- 실제 score 수정/suppression/adjusted_score 생성 없음
- stage1_dev 154명 only, stage2_holdout 접근 금지
"""
import csv
import json
import os
import sys
from pathlib import Path

# ===================== 경로 설정 =====================
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
P46C_ROOT = PROJECT_ROOT / "experiments/resnet50_imagenet_v1/outputs/reports/lesion_stage1_dev/p_a46c_ct_context_review_labels"
P45_ROOT  = PROJECT_ROOT / "experiments/resnet50_imagenet_v1/outputs/reports/lesion_stage1_dev/p_a45_lower_peripheral_fp_rule_preflight"
P44_ROOT  = PROJECT_ROOT / "experiments/resnet50_imagenet_v1/outputs/reports/lesion_stage1_dev/p_a44_tiny_topk_sensitivity"
P42_ROOT  = PROJECT_ROOT / "experiments/resnet50_imagenet_v1/outputs/reports/lesion_stage1_dev/p_a42_fpfn_visual_review_labels"
SPLIT_CSV  = PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"
SCORE_DIR  = PROJECT_ROOT / "outputs/position-aware-padim-v1/scores/padim_v2_roi0_0/lesion_v2_by_patient"
OUTPUT_ROOT = PROJECT_ROOT / "experiments/resnet50_imagenet_v1/outputs/reports/lesion_stage1_dev/p_a47_soft_penalty_design_preflight"

# ===================== 가드 1: 기존 결과 없음 확인 =====================
if OUTPUT_ROOT.exists():
    existing = list(OUTPUT_ROOT.iterdir())
    if existing:
        print(f"[ABORT] 기존 P-A47 결과 존재: {[f.name for f in existing[:5]]} — 덮어쓰지 않고 중단")
        sys.exit(1)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
guard_log = []

# ===================== 가드 2: P-A46c 통과 상태 확인 =====================
p46c_json_path = P46C_ROOT / "p_a46c_ct_context_review_labels.json"
if not p46c_json_path.exists():
    print(f"[ABORT] P-A46c JSON 없음: {p46c_json_path}")
    sys.exit(1)

with open(p46c_json_path, encoding="utf-8") as f:
    p46c_meta = json.load(f)

if p46c_meta.get("verdict") != "통과":
    print(f"[ABORT] P-A46c verdict={p46c_meta.get('verdict')} — 통과 아님. 중단.")
    sys.exit(1)
guard_log.append(f"가드 2: P-A46c verdict={p46c_meta['verdict']} ✓")

if p46c_meta.get("stage2_holdout_access", 1) != 0:
    print(f"[ABORT] P-A46c stage2_holdout_access={p46c_meta.get('stage2_holdout_access')} — 잠금 위반. 중단.")
    sys.exit(1)
guard_log.append(f"가드 2b: stage2_holdout_access={p46c_meta['stage2_holdout_access']} ✓")

# ===================== 가드 3: P-A46c labels 49행 확인 =====================
p46c_labels_path = P46C_ROOT / "ct_context_review_labels_filled.csv"
if not p46c_labels_path.exists():
    print(f"[ABORT] P-A46c labels 없음: {p46c_labels_path}")
    sys.exit(1)

with open(p46c_labels_path, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    p46c_rows = list(reader)

if len(p46c_rows) != 49:
    print(f"[ABORT] P-A46c labels {len(p46c_rows)}행 — 49행이 아님. 중단.")
    sys.exit(1)
guard_log.append(f"가드 3: P-A46c labels {len(p46c_rows)}행 ✓")

# ===================== 가드 4: stage1_dev 154명 목록 확인 =====================
if not SPLIT_CSV.exists():
    print(f"[ABORT] split CSV 없음: {SPLIT_CSV}")
    sys.exit(1)

with open(SPLIT_CSV, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    split_rows = list(reader)

stage1_dev_patients = set()
stage2_holdout_patients = set()
for row in split_rows:
    stage = row.get("stage_split", "").strip()
    pid = row.get("patient_id", "").strip()
    if stage == "stage1_dev":
        stage1_dev_patients.add(pid)
    elif stage == "stage2_holdout":
        stage2_holdout_patients.add(pid)

if len(stage1_dev_patients) != 154:
    print(f"[ABORT] stage1_dev 환자 수 {len(stage1_dev_patients)}명 — 154명이 아님. 중단.")
    sys.exit(1)
guard_log.append(f"가드 4: stage1_dev {len(stage1_dev_patients)}명 확정 ✓")
guard_log.append(f"가드 4b: stage2_holdout {len(stage2_holdout_patients)}명 잠금 확인 ✓")

# ===================== 가드 5: P-A46c 대상 환자 stage2_holdout 포함 여부 =====================
p46c_patients = set(row["patient_id"].strip() for row in p46c_rows)
holdout_leak = p46c_patients & stage2_holdout_patients
if holdout_leak:
    print(f"[ABORT] P-A46c 환자 중 stage2_holdout 포함: {holdout_leak} — 즉시 중단.")
    sys.exit(1)
guard_log.append(f"가드 5: P-A46c 환자({len(p46c_patients)}명) 중 stage2_holdout 0명 ✓")

# ===================== P-A46c labels 분류 =====================
EXCLUDE_CONDITIONS = {
    "lesion_protect": ("True",),
    "ct_context_label": (
        "lesion_near_boundary_protect",
        "lesion_near_pleura_or_vessel_protect",
        "lower_peripheral_true_lesion_protect",
    ),
    "penalty_safe": ("False",),
}
HOLD_CONDITIONS_LABEL = ("patch_location_unclear", "roi_or_slice_context_unclear")

candidate_pool = []
exclusion_pool = []
hold_pool = []

for row in p46c_rows:
    review_id   = row["review_id"].strip()
    patient_id  = row["patient_id"].strip()
    category    = row.get("category", "").strip()
    ct_label    = row.get("ct_context_label", "").strip()
    penalty_safe = row.get("penalty_safe", "").strip()
    lesion_protect = row.get("lesion_protect", "").strip()
    penalty_risk = row.get("penalty_risk", "").strip()
    action_rec   = row.get("action_recommendation", "").strip()
    review_note  = row.get("review_note", "").strip()
    padim_score  = row.get("padim_score", "").strip()
    slice_index  = row.get("slice_index", "").strip()

    # lesion_protect=True → 무조건 exclusion
    if lesion_protect == "True":
        exclusion_pool.append({
            "review_id": review_id,
            "patient_id": patient_id,
            "ct_context_label": ct_label,
            "penalty_safe": penalty_safe,
            "lesion_protect": lesion_protect,
            "penalty_risk": penalty_risk,
            "exclusion_reason": "lesion_protect=True",
            "padim_score": padim_score,
            "slice_index": slice_index,
        })
        continue

    # penalty_safe=Unclear 또는 ct_context_label이 unclear 계열 → hold
    if penalty_safe == "Unclear" or ct_label in HOLD_CONDITIONS_LABEL:
        hold_pool.append({
            "review_id": review_id,
            "patient_id": patient_id,
            "ct_context_label": ct_label,
            "penalty_safe": penalty_safe,
            "lesion_protect": lesion_protect,
            "penalty_risk": penalty_risk,
            "hold_reason": "penalty_safe=Unclear 또는 ct_context_label unclear",
            "padim_score": padim_score,
            "slice_index": slice_index,
        })
        continue

    # penalty_safe=True → candidate
    if penalty_safe == "True":
        candidate_pool.append({
            "review_id": review_id,
            "patient_id": patient_id,
            "ct_context_label": ct_label,
            "penalty_safe": penalty_safe,
            "lesion_protect": lesion_protect,
            "penalty_risk": penalty_risk,
            "padim_score": padim_score,
            "slice_index": slice_index,
            "action_recommendation": action_rec,
        })
        continue

    # 나머지 (penalty_safe=False이면서 lesion_protect=False) → exclusion (비호출)
    exclusion_pool.append({
        "review_id": review_id,
        "patient_id": patient_id,
        "ct_context_label": ct_label,
        "penalty_safe": penalty_safe,
        "lesion_protect": lesion_protect,
        "penalty_risk": penalty_risk,
        "exclusion_reason": "penalty_safe=False",
        "padim_score": padim_score,
        "slice_index": slice_index,
    })

guard_log.append(f"분류: candidate={len(candidate_pool)}, exclusion={len(exclusion_pool)}, hold={len(hold_pool)}, 합계={len(candidate_pool)+len(exclusion_pool)+len(hold_pool)}")
print("\n".join(guard_log))

# ===================== candidate pool 내 category 분포 =====================
from collections import Counter
cand_label_dist = Counter(row["ct_context_label"] for row in candidate_pool)
cand_patient_dist = Counter(row["patient_id"] for row in candidate_pool)
hold_label_dist = Counter(row["ct_context_label"] for row in hold_pool)
excl_reason_dist = Counter(row.get("exclusion_reason", "unknown") for row in exclusion_pool)

print(f"\n[candidate pool] {len(candidate_pool)}건")
for k, v in cand_label_dist.most_common():
    print(f"  {k}: {v}")

print(f"\n[hold pool] {len(hold_pool)}건")
for k, v in hold_label_dist.most_common():
    print(f"  {k}: {v}")

print(f"\n[exclusion pool] {len(exclusion_pool)}건")
for k, v in excl_reason_dist.most_common():
    print(f"  {k}: {v}")

# ===================== score CSV 컬럼 확인 (stage1_dev 1개 read-only) =====================
score_csv_columns = []
score_csv_sample_patient = None
score_csv_files = list(SCORE_DIR.glob("*.csv"))
stage1_score_files = [f for f in score_csv_files if f.stem in stage1_dev_patients]

# stage2_holdout score CSV 접근 금지 검증
for sf in score_csv_files:
    pid = sf.stem
    if pid in stage2_holdout_patients:
        print(f"[WARN] stage2_holdout score CSV 감지: {sf.name} — 열지 않음")

if stage1_score_files:
    # 1개만 read-only 샘플링으로 컬럼 확인
    sample_file = stage1_score_files[0]
    score_csv_sample_patient = sample_file.stem
    with open(sample_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        score_csv_columns = [c.strip() for c in header]
    print(f"\n[score CSV 컬럼 확인] 샘플: {sample_file.name}")
    print(f"  컬럼: {score_csv_columns}")

# ===================== dry-rule 자동 적용 가능성 분석 =====================
# score CSV에 position_bin 컬럼이 있는지 확인
has_position_bin = "position_bin" in score_csv_columns
has_padim_score_col = "padim_score" in score_csv_columns or "score" in score_csv_columns

# ct_context_label은 수동 review 결과 → score CSV에는 없음
# 따라서 자동 rule은 position_bin 기반으로만 가능
AUTO_RULE_FEASIBILITY = {}

# 1. pleura/chest-wall
pleura_candidates = [r for r in candidate_pool if r["ct_context_label"] == "pleura_chest_wall_fp_likely"]
AUTO_RULE_FEASIBILITY["pleura_chest_wall"] = {
    "candidate_count": len(pleura_candidates),
    "auto_rule_possible": "조건부" if has_position_bin else "불가",
    "auto_condition_candidate": "position_bin in [lower_peripheral, upper_peripheral] AND score > threshold" if has_position_bin else "N/A",
    "risk": "lower_peripheral 내 lesion patch와 overlap 위험 → position_bin 단독 rule은 lesion safety 위험",
    "recommendation": "manual-label 기반 후보 filtering 우선. position_bin 단독 자동 적용 불가.",
    "patients": list(set(r["patient_id"] for r in pleura_candidates)),
}

# 2. diaphragm/base
diaphragm_candidates = [r for r in candidate_pool if r["ct_context_label"] == "diaphragm_base_fp_likely"]
AUTO_RULE_FEASIBILITY["diaphragm_base"] = {
    "candidate_count": len(diaphragm_candidates),
    "auto_rule_possible": "조건부" if has_position_bin else "불가",
    "auto_condition_candidate": "position_bin == lower_central AND score > threshold" if has_position_bin else "N/A",
    "risk": "lower_central 내 lesion 가능성 존재. P-A45 lesion safety check 필요.",
    "recommendation": "manual-label 기반 후보 filtering 우선. position_bin 조건 추가 검증 필요.",
    "patients": list(set(r["patient_id"] for r in diaphragm_candidates)),
}

# 3. hilar/mediastinal
hilar_candidates = [r for r in candidate_pool if r["ct_context_label"] == "hilar_mediastinal_fp_likely"]
AUTO_RULE_FEASIBILITY["hilar_mediastinal"] = {
    "candidate_count": len(hilar_candidates),
    "auto_rule_possible": "불가",
    "auto_condition_candidate": "position_bin 기준 hilar 구분 불가 — middle_central 포함 가능성",
    "risk": "hilar 구조는 position_bin으로 구분 어려움. CT-context label 없이 자동 적용 불가.",
    "recommendation": "lower_peripheral와 별도 branch 유지. manual-label 기반 후보로만 남길 것.",
    "patients": list(set(r["patient_id"] for r in hilar_candidates)),
    "separate_branch": True,
}

# ===================== lesion safety risk 계산 =====================
# P-A45: lower_peripheral 내 lesion patch 9245개, 44명
p45_lp_lesion_patch_count = 9245
p45_lp_lesion_patient_count = 44
# candidate pool 내 lower_peripheral 관련 환자 수 (pleura 후보가 lower_peripheral 포함 가능성)
candidate_patients = set(r["patient_id"] for r in candidate_pool)

# P-A42 lesion protect 환자와의 overlap
p42_json_path = P42_ROOT / "p_a42_fpfn_visual_review_labels.json"
p42_protect_patients = set()
if p42_json_path.exists():
    with open(p42_json_path, encoding="utf-8") as f:
        p42_meta = json.load(f)
    # P-A42에서 lesion protect 환자 목록 추출 시도
    if "lesion_protect_patients" in p42_meta:
        p42_protect_patients = set(p42_meta["lesion_protect_patients"])

lesion_safety_risk_count = len(candidate_patients)  # candidate pool 내 모든 환자

# ===================== p95/p99 FP burden 추정 (추정치만) =====================
p45_json_path = P45_ROOT / "p_a45_lower_peripheral_fp_rule_preflight.json"
p95_threshold = None
if p45_json_path.exists():
    with open(p45_json_path, encoding="utf-8") as f:
        p45_meta = json.load(f)
    p95_threshold = p45_meta.get("p95_threshold")

# candidate pool 내 padim_score 기반 추정
try:
    cand_scores = [float(r["padim_score"]) for r in candidate_pool if r["padim_score"]]
    above_p95 = [s for s in cand_scores if p95_threshold and s > p95_threshold]
    fp_reduction_estimate = f"candidate {len(candidate_pool)}건 중 p95({p95_threshold:.2f}) 초과 {len(above_p95)}건 — soft penalty 적용 시 potential FP reduction 추정치 (실제 score 미적용)"
except Exception:
    fp_reduction_estimate = "score 파싱 불가 — 추정 불가"

# ===================== 출력 파일 생성 =====================

# 1. soft_penalty_candidate_pool.csv
cand_fieldnames = ["review_id", "patient_id", "ct_context_label", "penalty_safe", "lesion_protect", "penalty_risk", "padim_score", "slice_index", "action_recommendation"]
with open(OUTPUT_ROOT / "soft_penalty_candidate_pool.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=cand_fieldnames)
    writer.writeheader()
    writer.writerows(candidate_pool)

# 2. soft_penalty_exclusion_pool.csv
excl_fieldnames = ["review_id", "patient_id", "ct_context_label", "penalty_safe", "lesion_protect", "penalty_risk", "exclusion_reason", "padim_score", "slice_index"]
with open(OUTPUT_ROOT / "soft_penalty_exclusion_pool.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=excl_fieldnames)
    writer.writeheader()
    writer.writerows(exclusion_pool)

# 3. soft_penalty_hold_pool.csv
hold_fieldnames = ["review_id", "patient_id", "ct_context_label", "penalty_safe", "lesion_protect", "penalty_risk", "hold_reason", "padim_score", "slice_index"]
with open(OUTPUT_ROOT / "soft_penalty_hold_pool.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=hold_fieldnames)
    writer.writeheader()
    writer.writerows(hold_pool)

# 4. dry_rule_design_summary.csv
dry_rule_rows = []
for branch, info in AUTO_RULE_FEASIBILITY.items():
    dry_rule_rows.append({
        "branch": branch,
        "candidate_count": info["candidate_count"],
        "auto_rule_possible": info["auto_rule_possible"],
        "auto_condition_candidate": info["auto_condition_candidate"],
        "risk_note": info["risk"],
        "recommendation": info["recommendation"],
        "separate_branch": info.get("separate_branch", False),
        "patient_count": len(info["patients"]),
        "patients": "|".join(info["patients"]),
    })
dry_rule_fieldnames = ["branch", "candidate_count", "auto_rule_possible", "auto_condition_candidate", "risk_note", "recommendation", "separate_branch", "patient_count", "patients"]
with open(OUTPUT_ROOT / "dry_rule_design_summary.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=dry_rule_fieldnames)
    writer.writeheader()
    writer.writerows(dry_rule_rows)

# ===================== JSON 보고서 =====================
result_json = {
    "phase": "P-A47",
    "title": "Soft Penalty Design Preflight",
    "date": "2026-06-01",
    "verdict": "통과",

    # 가드 확인
    "p_a46c_verdict": p46c_meta.get("verdict"),
    "p_a46c_labels_count": len(p46c_rows),
    "p_a46c_labels_expected": 49,
    "stage2_holdout_access": 0,
    "stage1_dev_patient_count": len(stage1_dev_patients),
    "stage2_holdout_patient_count": len(stage2_holdout_patients),
    "p46c_patients_in_holdout": list(holdout_leak),

    # 수치 불일치 기록 (설계 문서 18건 vs 실제 17건)
    "design_doc_expected_candidate_count": 18,
    "actual_candidate_count": len(candidate_pool),
    "candidate_count_discrepancy_note": (
        "설계 문서 pleura 13건 → 실제 CSV pleura_chest_wall_fp_likely+True=12건 (1건 차이). "
        "설계 문서 total 18건 → 실제 penalty_safe=True 17건. "
        "P-A46c summary JSON '18건'과 CSV 실측 17건 불일치 — 실제 CSV 기준으로 진행."
    ),

    # pool 분포
    "candidate_pool_count": len(candidate_pool),
    "exclusion_pool_count": len(exclusion_pool),
    "hold_pool_count": len(hold_pool),
    "total_reviewed": len(candidate_pool) + len(exclusion_pool) + len(hold_pool),
    "candidate_label_distribution": dict(cand_label_dist),
    "hold_label_distribution": dict(hold_label_dist),
    "exclusion_reason_distribution": dict(excl_reason_dist),

    # lower_peripheral 전체 억제 금지 재확인
    "lower_peripheral_blanket_suppression": "금지 — 적용 없음 확인",
    "lower_peripheral_lesion_patch_count": p45_lp_lesion_patch_count,
    "lower_peripheral_lesion_patient_count": p45_lp_lesion_patient_count,

    # dry-rule 설계 가능성
    "auto_rule_feasibility": AUTO_RULE_FEASIBILITY,
    "manual_label_based_only": True,
    "manual_label_reason": "ct_context_label은 수동 review 결과로 score CSV에 없음. position_bin 단독 자동 rule은 lesion safety 위험.",

    # lesion safety risk
    "lesion_safety_risk_candidate_patient_count": lesion_safety_risk_count,
    "lesion_safety_risk_note": (
        f"candidate {len(candidate_pool)}건, {lesion_safety_risk_count}명. "
        "pleura/chest-wall rule 적용 시 lower_peripheral 내 lesion 9,245 patch(44명)와 overlap 위험."
    ),

    # p95/p99 FP burden 추정 (추정치만)
    "p95_threshold": p95_threshold,
    "fp_reduction_estimate": fp_reduction_estimate,
    "fp_reduction_note": "추정치 — 실제 score 미적용. 성능 개선 단정 금지.",

    # 실행 안전 확인
    "score_csv_modified": False,
    "adjusted_score_generated": False,
    "suppression_weight_generated": False,
    "threshold_changed": False,
    "metrics_recalculated": False,
    "scoring_rerun": False,
    "model_forward": False,
    "training": False,
    "stage2_holdout_locked": True,
    "existing_results_modified": False,
    "new_png_generated": False,

    # 출력 파일
    "output_files": [
        "soft_penalty_candidate_pool.csv",
        "soft_penalty_exclusion_pool.csv",
        "soft_penalty_hold_pool.csv",
        "dry_rule_design_summary.csv",
        "p_a47_soft_penalty_design_preflight.md",
        "p_a47_soft_penalty_design_preflight.json",
    ],

    # 다음 단계 추천
    "next_step": (
        "P-A47 결과 검토 후, manual-label 기반 candidate 17건에 대해 "
        "position_bin/score 조건 추가 검증 진행 가능. "
        "hilar/mediastinal 2건은 별도 branch 유지. "
        "실제 penalty weight 설계(P-A48)는 사용자 승인 후 진행."
    ),
}

with open(OUTPUT_ROOT / "p_a47_soft_penalty_design_preflight.json", "w", encoding="utf-8") as f:
    json.dump(result_json, f, ensure_ascii=False, indent=2)

# ===================== MD 보고서 =====================
pleura_cand_rows = [r for r in candidate_pool if r["ct_context_label"] == "pleura_chest_wall_fp_likely"]
diaphragm_cand_rows = [r for r in candidate_pool if r["ct_context_label"] == "diaphragm_base_fp_likely"]
hilar_cand_rows = [r for r in candidate_pool if r["ct_context_label"] == "hilar_mediastinal_fp_likely"]

md_lines = [
    "# P-A47 Soft Penalty Design Preflight",
    "",
    "## 판정: 통과",
    "",
    "---",
    "",
    "## 1. P-A46c 입력 검증",
    "",
    f"- verdict: {p46c_meta.get('verdict')} ✓",
    f"- labels 행수: {len(p46c_rows)}행 (기준 49행) ✓",
    f"- stage2_holdout_access: {p46c_meta.get('stage2_holdout_access', 0)} ✓",
    f"- p_a47_penalty_design_ready: {p46c_meta.get('p_a47_penalty_design_ready')}",
    "",
    "---",
    "",
    "## 2. 수치 불일치 기록",
    "",
    "| 항목 | 설계 문서 | 실제 CSV |",
    "|---|---|---|",
    f"| pleura/chest-wall 후보 | 13건 | {len(pleura_cand_rows)}건 |",
    f"| diaphragm/base 후보 | 3건 | {len(diaphragm_cand_rows)}건 |",
    f"| hilar/mediastinal 후보 | 2건 | {len(hilar_cand_rows)}건 |",
    f"| **확정 candidate pool** | **18건** | **{len(candidate_pool)}건** |",
    f"| hold pool | 7건 | {len(hold_pool)}건 |",
    f"| lesion_protect 제외 | 25건 | {len([r for r in exclusion_pool if r.get('exclusion_reason','')=='lesion_protect=True'])}건 |",
    "",
    "> `pleura_chest_wall_fp_likely` 15건 중 3건이 `penalty_safe=Unclear` → hold 처리됨.",
    "> 실제 CSV 기준 17건으로 진행. P-A46c summary JSON '18건'과 1건 차이 기록.",
    "",
    "---",
    "",
    "## 3. Candidate Pool 요약 (17건)",
    "",
    "| ct_context_label | 건수 | 환자 수 |",
    "|---|---|---|",
    f"| pleura_chest_wall_fp_likely | {len(pleura_cand_rows)} | {len(set(r['patient_id'] for r in pleura_cand_rows))} |",
    f"| diaphragm_base_fp_likely | {len(diaphragm_cand_rows)} | {len(set(r['patient_id'] for r in diaphragm_cand_rows))} |",
    f"| hilar_mediastinal_fp_likely | {len(hilar_cand_rows)} | {len(set(r['patient_id'] for r in hilar_cand_rows))} |",
    f"| **합계** | **{len(candidate_pool)}** | **{len(set(r['patient_id'] for r in candidate_pool))}** |",
    "",
    "---",
    "",
    "## 4. Exclusion Pool 요약 (lesion_protect=True 25건 포함)",
    "",
    f"- lesion_protect=True: {len([r for r in exclusion_pool if r.get('exclusion_reason','')=='lesion_protect=True'])}건",
    f"- penalty_safe=False (비호출): {len([r for r in exclusion_pool if r.get('exclusion_reason','')=='penalty_safe=False'])}건",
    f"- 합계: {len(exclusion_pool)}건",
    "",
    "lesion_protect=True 케이스 제외 조건:",
    "- lesion_near_boundary_protect",
    "- lesion_near_pleura_or_vessel_protect",
    "- lower_peripheral_true_lesion_protect",
    "",
    "---",
    "",
    "## 5. Hold Pool 요약 (7건)",
    "",
]
for r in hold_pool:
    md_lines.append(f"- {r['review_id']} | {r['patient_id']} | {r['ct_context_label']} | penalty_safe={r['penalty_safe']}")
md_lines += [
    "",
    "> Unclear/needs_manual_review 7건은 수동 검토 후 별도 결정 필요.",
    "",
    "---",
    "",
    "## 6. lower_peripheral 전체 억제 금지 재확인",
    "",
    f"- 억제 적용: **없음** ✓",
    f"- lower_peripheral 내 lesion patch: {p45_lp_lesion_patch_count}개 / {p45_lp_lesion_patient_count}명",
    "- 전체 억제 금지 — soft penalty candidate 필터링 후에도 lesion 보호 우선",
    "",
    "---",
    "",
    "## 7. Dry-Rule 설계 가능성",
    "",
    "### 7-1. pleura/chest-wall soft penalty",
    f"- 후보: {len(pleura_cand_rows)}건",
    f"- 자동 rule 가능 여부: 조건부 (position_bin 기반 — lesion safety 위험 존재)",
    "- **결론: manual-label 기반 후보 filtering 우선**",
    "- position_bin 단독 자동 적용 불가 — lower_peripheral 내 lesion 9,245 patch overlap 위험",
    "",
    "### 7-2. diaphragm/base soft penalty",
    f"- 후보: {len(diaphragm_cand_rows)}건",
    f"- 자동 rule 가능 여부: 조건부 (lower_central position_bin 기반 — 추가 검증 필요)",
    "- **결론: manual-label 기반 후보 filtering 우선**",
    "",
    "### 7-3. hilar/mediastinal soft penalty",
    f"- 후보: {len(hilar_cand_rows)}건",
    "- 자동 rule 가능 여부: **불가** (position_bin으로 hilar 구분 어려움)",
    "- **결론: lower_peripheral와 별도 branch 유지. manual-label 기반 후보로만 남길 것**",
    "",
    "---",
    "",
    "## 8. Lesion Safety Risk",
    "",
    f"- candidate {len(candidate_pool)}건, {lesion_safety_risk_count}명",
    f"- lower_peripheral 내 lesion: {p45_lp_lesion_patch_count}개 / {p45_lp_lesion_patient_count}명 (P-A45 기준)",
    "- pleura/chest-wall penalty rule 적용 시 lower_peripheral lesion과 overlap 위험 존재",
    "- **실제 적용 전 lesion_protect 조건 확인 필수**",
    "",
    "---",
    "",
    "## 9. Expected FP Reduction (추정치 only)",
    "",
    f"- p95 threshold: {p95_threshold:.4f} (P-A45 기준)" if p95_threshold else "- p95 threshold: 확인 불가",
    f"- {fp_reduction_estimate}",
    "- **'성능 개선' 단정 금지 — potential FP reduction 추정치로만 기록**",
    "",
    "---",
    "",
    "## 10. 실행 안전 확인",
    "",
    "| 항목 | 상태 |",
    "|---|---|",
    "| score CSV 수정 | 없음 ✓ |",
    "| adjusted_score 생성 | 없음 ✓ |",
    "| suppression_weight 생성 | 없음 ✓ |",
    "| threshold 변경 | 없음 ✓ |",
    "| metrics 재계산 | 없음 ✓ |",
    "| scoring 재실행 | 없음 ✓ |",
    "| model forward | 없음 ✓ |",
    "| training | 없음 ✓ |",
    "| stage2_holdout 접근 | 없음 ✓ |",
    "| 기존 결과 수정 | 없음 ✓ |",
    "| PNG 신규 생성 | 없음 ✓ |",
    "",
    "---",
    "",
    "## 11. 다음 단계 추천",
    "",
    "1. P-A47 결과 검토 후, manual-label 기반 candidate 17건에 대해 position_bin/score 조건 추가 검증",
    "2. hilar/mediastinal 2건 별도 branch 유지",
    "3. hold pool 7건 수동 검토 후 candidate 또는 exclusion 분류",
    "4. 실제 penalty weight 설계(P-A48)는 사용자 승인 후 진행",
    "",
]

with open(OUTPUT_ROOT / "p_a47_soft_penalty_design_preflight.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print(f"\n[완료] 출력 경로: {OUTPUT_ROOT}")
print(f"  - soft_penalty_candidate_pool.csv  ({len(candidate_pool)}행)")
print(f"  - soft_penalty_exclusion_pool.csv  ({len(exclusion_pool)}행)")
print(f"  - soft_penalty_hold_pool.csv       ({len(hold_pool)}행)")
print(f"  - dry_rule_design_summary.csv      ({len(dry_rule_rows)}행)")
print(f"  - p_a47_soft_penalty_design_preflight.md")
print(f"  - p_a47_soft_penalty_design_preflight.json")
print(f"\n판정: 통과")
