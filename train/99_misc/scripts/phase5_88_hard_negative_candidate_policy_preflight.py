"""
Phase 5.88: Hard Negative Candidate Policy Preflight
목적: FP cause review 결과를 바탕으로 hard negative 후보 정책을 설계 전 점검한다.
- hard negative 최종 채택 없음
- training dataset / crop manifest 수정 없음
- model forward / score 재계산 없음
- threshold 확정 없음
- stage2_holdout / v2 접근 없음
"""

import json
import csv
import os

BASE_DIR = "/home/jinhy/project/lung-ct-anomaly"

INPUT_PHASE5_69_JSON = os.path.join(
    BASE_DIR,
    "outputs/second-stage-lesion-refiner-v1/review_annotations"
    "/hard_negative_top_score_qa_v1/phase5_69_expanded_fp_cause_summary_v1"
    "/phase5_69_expanded_fp_cause_summary_v1.json"
)
INPUT_PHASE5_86_JSON = os.path.join(
    BASE_DIR,
    "outputs/second-stage-lesion-refiner-v1/review_annotations"
    "/phase5_86_weak_3d_visual_review_label_acceptance_v1"
    "/phase5_86_weak_3d_visual_review_label_acceptance_summary_v1.json"
)
INPUT_PHASE5_87_JSON = os.path.join(
    BASE_DIR,
    "outputs/second-stage-lesion-refiner-v1/review_annotations"
    "/phase5_87_fp_cause_multi_pool_expanded_summary_v1"
    "/phase5_87_fp_cause_multi_pool_expanded_summary_v1.json"
)

OUTPUT_ROOT = os.path.join(
    BASE_DIR,
    "outputs/second-stage-lesion-refiner-v1/review_annotations"
    "/phase5_88_hard_negative_candidate_policy_preflight_v1"
)
OUTPUT_CSV = os.path.join(OUTPUT_ROOT, "phase5_88_hard_negative_candidate_policy_preflight_v1.csv")
OUTPUT_JSON = os.path.join(OUTPUT_ROOT, "phase5_88_hard_negative_candidate_policy_preflight_v1.json")
OUTPUT_MD = os.path.join(OUTPUT_ROOT, "phase5_88_hard_negative_candidate_policy_preflight_report_v1.md")


def load_and_verify_inputs():
    print("=== 입력 파일 로드 및 검증 ===")

    with open(INPUT_PHASE5_69_JSON) as f:
        d69 = json.load(f)
    print(f"[OK] Phase 5.69 JSON 로드 완료: {INPUT_PHASE5_69_JSON}")

    with open(INPUT_PHASE5_86_JSON) as f:
        d86 = json.load(f)
    print(f"[OK] Phase 5.86 JSON 로드 완료: {INPUT_PHASE5_86_JSON}")

    with open(INPUT_PHASE5_87_JSON) as f:
        d87 = json.load(f)
    print(f"[OK] Phase 5.87 JSON 로드 완료: {INPUT_PHASE5_87_JSON}")

    # Phase 5.69 count 검증
    n69 = d69.get("n_expanded_fp_pool", "?")
    label_summary_69 = d69.get("n_by_expanded_fp_cause_label", {})
    pw69 = label_summary_69.get("pleural_wall", "?")
    lb69 = label_summary_69.get("large_bbox_structure", "?")
    vb69 = label_summary_69.get("vessel_branch", "?")
    ba69 = label_summary_69.get("bronchus_air_boundary", "?")
    print(f"[VERIFY] Phase 5.69 n={n69}, pleural_wall={pw69}, large_bbox_structure={lb69}, vessel_branch={vb69}, bronchus_air_boundary={ba69}")
    assert n69 == 39, f"Phase 5.69 n 불일치: {n69}"
    assert pw69 == 23, f"pleural_wall 불일치: {pw69}"
    assert lb69 == 12, f"large_bbox_structure 불일치: {lb69}"
    assert vb69 == 3, f"vessel_branch 불일치: {vb69}"
    assert ba69 == 1, f"bronchus_air_boundary 불일치: {ba69}"
    print("[PASS] Phase 5.69 count 검증 완료")

    # Phase 5.86 count 검증
    n86 = d86.get("total_rows", "?")
    label_count_86 = d86.get("label_count", {})
    vb86 = label_count_86.get("vessel_branch", "?")
    lb86 = label_count_86.get("large_bbox_structure", "?")
    pw86 = label_count_86.get("pleural_wall", "?")
    ba86 = label_count_86.get("bronchus_air_boundary", "?")
    print(f"[VERIFY] Phase 5.86 n={n86}, vessel_branch={vb86}, large_bbox_structure={lb86}, pleural_wall={pw86}, bronchus_air_boundary={ba86}")
    assert n86 == 14, f"Phase 5.86 n 불일치: {n86}"
    assert vb86 == 6, f"vessel_branch 불일치: {vb86}"
    assert lb86 == 6, f"large_bbox_structure 불일치: {lb86}"
    assert pw86 == 1, f"pleural_wall 불일치: {pw86}"
    assert ba86 == 1, f"bronchus_air_boundary 불일치: {ba86}"
    print("[PASS] Phase 5.86 count 검증 완료")

    # Phase 5.87 not pooled prevalence 경고 확인
    ref_only = d87.get("reference_only_combined_counts", {})
    warning = ref_only.get("warning", "")
    assert "reference_only" in warning, f"Phase 5.87 not pooled warning 없음: {warning}"
    print(f"[PASS] Phase 5.87 not pooled prevalence warning 확인: {warning}")

    return d69, d86, d87


def build_section_a():
    """Section A: candidate class policy"""
    rows = [
        {
            "section": "A",
            "candidate_class": "pleural_wall_boundary_candidate",
            "included_labels": "pleural_wall",
            "source_evidence": "Phase 5.69 dominance: pleural_wall=23/39 (59.0%); Phase 5.86 minor: 1/14 (7.1%)",
            "recommended_status": "review_candidate_only",
            "allowed_next_use": "boundary/pleural FP robustness review; Phase 5.89 manifest 설계 참고",
            "forbidden_use": "hard negative final selection without approval; training dataset modification; threshold finalization",
            "rationale": "Phase 5.69 high-score FP pool에서 가장 지배적인 원인(59.0%). pleural wall 경계 구조와 관련된 FP robustness 검토 대상으로 우선 분류.",
            "limitation": "Phase 5.69와 5.86은 selection 기준이 달라 단순 비교 불가. not pooled prevalence. Phase 5.86에서는 소수(7.1%)."
        },
        {
            "section": "A",
            "candidate_class": "vessel_mediastinal_hilar_candidate",
            "included_labels": "vessel_branch,large_bbox_structure",
            "source_evidence": "Phase 5.86: vessel_branch+large_bbox_structure=12/14 (85.7%); Phase 5.69: large_bbox_structure=12/39 (30.8%), vessel_branch=3/39 (7.7%)",
            "recommended_status": "review_candidate_only",
            "allowed_next_use": "central vessel/mediastinal FP robustness review; Phase 5.89 manifest 설계 참고",
            "forbidden_use": "hard negative final selection without approval; training dataset modification; threshold finalization",
            "rationale": "Phase 5.86 weak 3D QA pool에서 지배적(85.7%). large_bbox_structure는 두 pool 모두에서 반복 확인. vessel_branch는 weak 3D visual QA에서 두드러짐.",
            "limitation": "두 pool은 selection 기준이 다름. not pooled prevalence. vessel_branch는 Phase 5.69에서는 소수."
        },
        {
            "section": "A",
            "candidate_class": "bronchus_air_boundary_candidate",
            "included_labels": "bronchus_air_boundary",
            "source_evidence": "Phase 5.69: 1/39 (2.6%); Phase 5.86: 1/14 (7.1%)",
            "recommended_status": "review_candidate_only",
            "allowed_next_use": "small reference group only; 두 pool 모두에서 반복 확인 참고",
            "forbidden_use": "hard negative final selection without approval; training dataset modification",
            "rationale": "두 pool 모두에서 소수이지만 반복 확인됨. 소수 reference group으로 유지.",
            "limitation": "수가 적어 통계적 판단 불가. not pooled prevalence."
        },
        {
            "section": "A",
            "candidate_class": "outside_roi_or_context_caution",
            "included_labels": "possible_lesion_overlap,outside_roi_artifact,unclear,z_overmerge_suspicious_overmerge",
            "source_evidence": "prior outside ROI and context overlap cautions; prior phase decisions",
            "recommended_status": "quarantine_pending",
            "allowed_next_use": "metadata 기록만 허용; 별도 승인 후 별도 단계에서 재검토 가능",
            "forbidden_use": "hard negative candidate 포함 금지; training dataset 포함 금지; 병변 subset 적용 금지",
            "rationale": "병변 overlap 가능성, outside ROI artifact, 불명확 label, suspicious overmerge는 hard negative로 사용 시 오염 위험. 별도 승인 전까지 quarantine.",
            "limitation": "이미 다른 phase에서 일부 검토됨. 향후 별도 승인 기반 재검토 대상."
        },
    ]
    return rows


def build_section_b():
    """Section B: label-level policy"""
    rows = [
        {
            "section": "B",
            "user_label": "pleural_wall",
            "phase5_69_count": 23,
            "phase5_69_ratio_percent": 59.0,
            "phase5_86_count": 1,
            "phase5_86_ratio_percent": 7.1,
            "recommended_status": "review_candidate_only",
            "hard_negative_policy_note": "Phase 5.69 dominant FP cause. boundary axis 대표. 별도 manifest 설계 후 검토 가능.",
            "required_approval_before_use": "yes"
        },
        {
            "section": "B",
            "user_label": "large_bbox_structure",
            "phase5_69_count": 12,
            "phase5_69_ratio_percent": 30.8,
            "phase5_86_count": 6,
            "phase5_86_ratio_percent": 42.9,
            "recommended_status": "review_candidate_only",
            "hard_negative_policy_note": "두 pool 모두에서 반복 확인됨. vessel/mediastinal/hilar axis 포함. 별도 manifest 설계 후 검토 가능.",
            "required_approval_before_use": "yes"
        },
        {
            "section": "B",
            "user_label": "vessel_branch",
            "phase5_69_count": 3,
            "phase5_69_ratio_percent": 7.7,
            "phase5_86_count": 6,
            "phase5_86_ratio_percent": 42.9,
            "recommended_status": "review_candidate_only",
            "hard_negative_policy_note": "Phase 5.86 weak 3D QA에서 두드러짐(42.9%). vessel/mediastinal/hilar axis 포함. 별도 manifest 설계 후 검토 가능.",
            "required_approval_before_use": "yes"
        },
        {
            "section": "B",
            "user_label": "bronchus_air_boundary",
            "phase5_69_count": 1,
            "phase5_69_ratio_percent": 2.6,
            "phase5_86_count": 1,
            "phase5_86_ratio_percent": 7.1,
            "recommended_status": "review_candidate_only",
            "hard_negative_policy_note": "두 pool 모두 소수. 소수 reference group 유지. 단독 large-scale 사용 금지.",
            "required_approval_before_use": "yes"
        },
        {
            "section": "B",
            "user_label": "possible_lesion_overlap",
            "phase5_69_count": "N/A",
            "phase5_69_ratio_percent": "N/A",
            "phase5_86_count": "N/A",
            "phase5_86_ratio_percent": "N/A",
            "recommended_status": "quarantine_pending",
            "hard_negative_policy_note": "병변 overlap 가능성. hard negative 포함 금지. 별도 승인 전까지 quarantine.",
            "required_approval_before_use": "yes - separate approval required"
        },
        {
            "section": "B",
            "user_label": "outside_roi_artifact",
            "phase5_69_count": "N/A",
            "phase5_69_ratio_percent": "N/A",
            "phase5_86_count": "N/A",
            "phase5_86_ratio_percent": "N/A",
            "recommended_status": "quarantine_pending",
            "hard_negative_policy_note": "ROI 외부 artifact. hard negative 포함 금지. 별도 승인 전까지 quarantine.",
            "required_approval_before_use": "yes - separate approval required"
        },
        {
            "section": "B",
            "user_label": "unclear",
            "phase5_69_count": "N/A",
            "phase5_69_ratio_percent": "N/A",
            "phase5_86_count": "N/A",
            "phase5_86_ratio_percent": "N/A",
            "recommended_status": "quarantine_pending",
            "hard_negative_policy_note": "label 불명확. hard negative 포함 금지. 별도 재검토 후 승인 필요.",
            "required_approval_before_use": "yes - separate approval required"
        },
        {
            "section": "B",
            "user_label": "z_overmerge_suspicious_overmerge",
            "phase5_69_count": "N/A",
            "phase5_69_ratio_percent": "N/A",
            "phase5_86_count": "N/A",
            "phase5_86_ratio_percent": "N/A",
            "recommended_status": "quarantine_pending",
            "hard_negative_policy_note": "suspicious overmerge. 오염 위험. hard negative 포함 금지. 별도 승인 전까지 quarantine.",
            "required_approval_before_use": "yes - separate approval required"
        },
        {
            "section": "B",
            "user_label": "z_overmerge_ok_continuous_structure",
            "phase5_69_count": "N/A",
            "phase5_69_ratio_percent": "N/A",
            "phase5_86_count": "N/A",
            "phase5_86_ratio_percent": "N/A",
            "recommended_status": "metadata_caution_only",
            "hard_negative_policy_note": "visually continuous structure로 확인됨. metadata caution만 적용. 자동 제외 금지이나 hard negative 사용 시 별도 승인 필요.",
            "required_approval_before_use": "yes"
        },
    ]
    return rows


def build_section_c():
    """Section C: exclusion/quarantine rules"""
    rows = [
        {
            "section": "C",
            "condition": "user_label == possible_lesion_overlap",
            "action": "quarantine_pending",
            "reason": "병변 overlap 가능성이 있는 crop을 hard negative로 사용 시 병변 신호가 포함될 수 있음. 별도 승인 전 hard negative pool 제외.",
            "approval_required": "yes"
        },
        {
            "section": "C",
            "condition": "user_label == outside_roi_artifact",
            "action": "quarantine_pending",
            "reason": "ROI 외부 artifact crop은 정상 분포 학습에 부적합. 별도 승인 전 hard negative pool 제외.",
            "approval_required": "yes"
        },
        {
            "section": "C",
            "condition": "user_label == unclear",
            "action": "quarantine_pending",
            "reason": "label이 불명확한 crop은 hard negative로 사용 시 noise 도입 위험. 별도 재검토 후 승인 필요.",
            "approval_required": "yes"
        },
        {
            "section": "C",
            "condition": "user_label == z_overmerge_suspicious_overmerge",
            "action": "quarantine_pending",
            "reason": "suspicious overmerge로 분류된 crop은 구조적 오염 위험. 별도 승인 전 hard negative pool 제외.",
            "approval_required": "yes"
        },
        {
            "section": "C",
            "condition": "user_label == z_overmerge_ok_continuous_structure AND hard_negative_use_intended",
            "action": "metadata_caution_only",
            "reason": "visually continuous structure로 확인됐으나 hard negative로 사용 시 별도 승인 필요. 자동 제외는 하지 않고 caution 메모만 유지.",
            "approval_required": "yes"
        },
        {
            "section": "C",
            "condition": "overmerge_flag == True AND label NOT in [z_overmerge_ok_continuous_structure]",
            "action": "metadata_caution_only",
            "reason": "overmerge_flag=True이나 visually continuous structure로 확인되지 않은 경우 caution 메모 유지. 자동 제외 아님.",
            "approval_required": "yes"
        },
    ]
    return rows


def build_section_d():
    """Section D: next-step options"""
    rows = [
        {
            "section": "D",
            "option_id": "A",
            "option_name": "Phase 5.89 hard negative candidate manifest 설계 preflight",
            "description": "Phase 5.88 policy preflight를 바탕으로 실제 hard negative candidate manifest 설계를 위한 preflight를 수행한다. crop_id 목록, 분류 근거, 사용 가능 여부를 정리하는 단계.",
            "risk": "manifest 생성 시 training dataset 수정 위험. 별도 승인 필요.",
            "approval_required": "yes"
        },
        {
            "section": "D",
            "option_id": "B",
            "option_name": "vessel/mediastinal/pleural rule candidate design",
            "description": "Phase 5.88에서 제안된 두 axis(pleural_wall boundary / vessel_mediastinal_hilar)에 대한 rule-based candidate 설계를 검토한다. crop 선택 기준, 구조 정의, 경계 조건을 설계 문서로 정리.",
            "risk": "rule 설계 후 적용 시 model forward / score 재계산 위험. 설계 단계에서는 read-only.",
            "approval_required": "yes"
        },
        {
            "section": "D",
            "option_id": "C",
            "option_name": "현재 결과 handoff 최신화",
            "description": "Phase 5.87 ~ 5.88 결과를 포함한 최신 handoff 문서를 업데이트한다. 다음 단계 진입 전 누적 결과 정리.",
            "risk": "low. read-only 결과 정리 작업.",
            "approval_required": "no"
        },
    ]
    return rows


def write_csv(rows_a, rows_b, rows_c, rows_d):
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    all_rows = rows_a + rows_b + rows_c + rows_d

    # 전체 column 목록 수집
    all_keys = []
    for row in all_rows:
        for k in row:
            if k not in all_keys:
                all_keys.append(k)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"[OK] CSV 생성 완료: {OUTPUT_CSV}")


def write_json(rows_a, rows_b, rows_c, rows_d):
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    output = {
        "phase": "5.88",
        "checkpoint_type": "hard_negative_candidate_policy_preflight",
        "version": "v1",
        "input_paths": {
            "phase5_69_json": INPUT_PHASE5_69_JSON,
            "phase5_86_json": INPUT_PHASE5_86_JSON,
            "phase5_87_json": INPUT_PHASE5_87_JSON,
        },
        "output_csv_path": OUTPUT_CSV,
        "candidate_classes": rows_a,
        "label_level_policy": rows_b,
        "exclusion_quarantine_rules": rows_c,
        "next_step_options": rows_d,
        "key_findings": [
            "Phase 5.69와 Phase 5.86은 selection 기준이 달라 단순 합산 prevalence로 해석하지 않는다.",
            "FP cause는 최소 두 axis로 관리해야 한다: (A) pleural_wall/lung wall/diaphragm boundary axis, (B) vessel/mediastinal/hilar large-structure axis.",
            "large_bbox_structure는 두 pool 모두에서 반복 확인됨 (Phase 5.69: 30.8%, Phase 5.86: 42.9%).",
            "vessel_branch는 Phase 5.86 weak 3D visual QA pool에서 두드러짐 (7.7% -> 42.9%).",
            "bronchus_air_boundary는 두 pool 모두에서 소수(2.6%, 7.1%). 소수 reference group으로 유지.",
            "possible_lesion_overlap, outside_roi_artifact, unclear, z_overmerge_suspicious_overmerge는 별도 승인 전 quarantine.",
        ],
        "limitations": [
            "Phase 5.69와 Phase 5.86의 selection 기준이 달라 단순 비교 불가.",
            "pooled prevalence를 주장하지 않는다 (reference_only_combined_counts는 참고용).",
            "이 단계는 policy preflight만 수행하며 hard negative 최종 채택을 하지 않는다.",
            "Phase 5.69는 n=39 (2D/ROI 재정렬 기반 high-score FP crop review).",
            "Phase 5.86은 n=14 (weak 3D top9 + overmerge_priority visual QA, user accepted).",
            "normal patient FP cause review 결과이므로 전체 FP 분포나 전체 모델 성능으로 일반화 금지.",
        ],
        "notes": {
            "policy_preflight_only": True,
            "no_hard_negative_final_selection": True,
            "no_training_dataset_modification": True,
            "no_crop_manifest_modification": True,
            "no_model_forward": True,
            "no_score_recalculation": True,
            "threshold_not_finalized": True,
            "lesion_conclusion_forbidden": True,
            "stage2_holdout_unused": True,
            "v2_unused": True,
            "existing_outputs_unmodified": True,
        }
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[OK] JSON 생성 완료: {OUTPUT_JSON}")
    return output


def write_md(output_json):
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    rows_a = output_json["candidate_classes"]
    rows_b = output_json["label_level_policy"]
    rows_c = output_json["exclusion_quarantine_rules"]
    rows_d = output_json["next_step_options"]

    lines = []
    lines.append("# Phase 5.88: Hard Negative Candidate Policy Preflight Report")
    lines.append("")

    # 1. 목적
    lines.append("## 1. 목적")
    lines.append("")
    lines.append("- 지금까지의 FP cause review 결과(Phase 5.69, 5.86, 5.87)를 바탕으로 hard negative 후보 정책을 **설계 전 점검**한다.")
    lines.append("- 실제 hard negative 최종 채택은 하지 않는다.")
    lines.append("- training dataset 수정, crop 생성, score 재계산, model forward, threshold 확정은 하지 않는다.")
    lines.append("- stage2_holdout / v2는 계속 봉인한다.")
    lines.append("")

    # 2. 입력 근거 요약
    lines.append("## 2. 입력 근거 요약")
    lines.append("")
    lines.append("### Phase 5.69 (n=39, 2D/ROI 재정렬 기반 high-score FP crop review)")
    lines.append("")
    lines.append("| label | count | ratio |")
    lines.append("|-------|-------|-------|")
    lines.append("| pleural_wall | 23 | 59.0% |")
    lines.append("| large_bbox_structure | 12 | 30.8% |")
    lines.append("| vessel_branch | 3 | 7.7% |")
    lines.append("| bronchus_air_boundary | 1 | 2.6% |")
    lines.append("")
    lines.append("### Phase 5.86 (n=14, weak 3D top9 + overmerge_priority visual QA, user accepted)")
    lines.append("")
    lines.append("| label | count | ratio |")
    lines.append("|-------|-------|-------|")
    lines.append("| vessel_branch | 6 | 42.9% |")
    lines.append("| large_bbox_structure | 6 | 42.9% |")
    lines.append("| large_bbox_structure + vessel_branch | 12 | 85.7% |")
    lines.append("| pleural_wall | 1 | 7.1% |")
    lines.append("| bronchus_air_boundary | 1 | 7.1% |")
    lines.append("")
    lines.append("### Phase 5.87 해석")
    lines.append("")
    lines.append("- FP cause는 최소 두 axis로 관리: (A) pleural_wall/lung wall/diaphragm boundary axis, (B) vessel/mediastinal/hilar large-structure axis")
    lines.append("- large_bbox_structure는 두 pool 모두에서 반복 확인")
    lines.append("- **중요: pooled prevalence를 주장하지 않는다.** Phase 5.69와 5.86은 selection 기준이 다르므로 단순 합산 해석 금지.")
    lines.append("")

    # 3. candidate class policy 표
    lines.append("## 3. Candidate Class Policy")
    lines.append("")
    lines.append("| candidate_class | included_labels | source_evidence | recommended_status | rationale |")
    lines.append("|-----------------|-----------------|-----------------|-------------------|-----------|")
    for r in rows_a:
        lines.append(f"| {r['candidate_class']} | {r['included_labels']} | {r['source_evidence']} | {r['recommended_status']} | {r['rationale']} |")
    lines.append("")

    # 4. label-level policy 표
    lines.append("## 4. Label-Level Policy")
    lines.append("")
    lines.append("| user_label | 5.69_count | 5.69_ratio | 5.86_count | 5.86_ratio | recommended_status | hard_negative_policy_note |")
    lines.append("|------------|------------|------------|------------|------------|-------------------|--------------------------|")
    for r in rows_b:
        lines.append(
            f"| {r['user_label']} | {r['phase5_69_count']} | {r['phase5_69_ratio_percent']} | {r['phase5_86_count']} | {r['phase5_86_ratio_percent']} | {r['recommended_status']} | {r['hard_negative_policy_note']} |"
        )
    lines.append("")

    # 5. exclusion/quarantine rule 표
    lines.append("## 5. Exclusion / Quarantine Rules")
    lines.append("")
    lines.append("| condition | action | reason | approval_required |")
    lines.append("|-----------|--------|--------|------------------|")
    for r in rows_c:
        lines.append(f"| {r['condition']} | {r['action']} | {r['reason']} | {r['approval_required']} |")
    lines.append("")

    # 6. 핵심 판단
    lines.append("## 6. 핵심 판단")
    lines.append("")
    lines.append("1. **pleural_wall boundary axis**와 **vessel/mediastinal/hilar axis**를 분리하여 관리한다.")
    lines.append("   - Phase 5.69: pleural_wall dominance (59.0%) → boundary axis 대표")
    lines.append("   - Phase 5.86: vessel_branch + large_bbox_structure dominance (85.7%) → vessel/hilar axis 대표")
    lines.append("2. **large_bbox_structure**는 두 pool 모두에서 반복 확인됨 (5.69: 30.8%, 5.86: 42.9%).")
    lines.append("3. **vessel_branch**는 Phase 5.86 weak 3D visual QA에서 중요 (7.7% → 42.9%).")
    lines.append("4. **bronchus_air_boundary**는 소수 reference group으로 유지 (두 pool 모두 ≤7.1%).")
    lines.append("")

    # 7. 해석 제한
    lines.append("## 7. 해석 제한")
    lines.append("")
    for lim in output_json["limitations"]:
        lines.append(f"- {lim}")
    lines.append("")

    # 8. 다음 단계 후보
    lines.append("## 8. 다음 단계 후보")
    lines.append("")
    lines.append("| option_id | option_name | description | risk | approval_required |")
    lines.append("|-----------|-------------|-------------|------|------------------|")
    for r in rows_d:
        lines.append(f"| {r['option_id']} | {r['option_name']} | {r['description']} | {r['risk']} | {r['approval_required']} |")
    lines.append("")

    # 9. 금지
    lines.append("## 9. 금지 사항")
    lines.append("")
    lines.append("- hard negative 최종 채택 금지")
    lines.append("- threshold 확정 금지")
    lines.append("- 병변 subset 적용 금지")
    lines.append("- training dataset / crop manifest 수정 금지")
    lines.append("- model forward 금지")
    lines.append("- score 재계산 금지")
    lines.append("- 병변 성능 결론 금지")
    lines.append("- stage2_holdout / v2 접근 금지")
    lines.append("- 입력 CSV/JSON/MD 수정 금지")
    lines.append("- PNG/HTML/ZIP 생성 금지")
    lines.append("- CT/ROI/mask npy 로드 금지")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*This phase only proposes a candidate handling policy.*")
    lines.append("*No hard negative has been finally selected.*")
    lines.append("*No train/val/test split has been modified.*")
    lines.append("*No crop manifest has been modified.*")
    lines.append("*No threshold has been finalized.*")
    lines.append("*No lesion performance conclusion is made.*")

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] MD report 생성 완료: {OUTPUT_MD}")


def main():
    print("=== Phase 5.88 Hard Negative Candidate Policy Preflight ===")

    d69, d86, d87 = load_and_verify_inputs()

    rows_a = build_section_a()
    rows_b = build_section_b()
    rows_c = build_section_c()
    rows_d = build_section_d()

    write_csv(rows_a, rows_b, rows_c, rows_d)
    output_json = write_json(rows_a, rows_b, rows_c, rows_d)
    write_md(output_json)

    print("")
    print("=== 생성 완료 ===")
    print(f"output root : {OUTPUT_ROOT}")
    print(f"CSV         : {OUTPUT_CSV}")
    print(f"JSON        : {OUTPUT_JSON}")
    print(f"MD          : {OUTPUT_MD}")
    print("")
    print("=== 금지사항 위반 확인 ===")
    print("[OK] hard negative 최종 채택 없음")
    print("[OK] training dataset / crop manifest 수정 없음")
    print("[OK] model forward 없음")
    print("[OK] score 재계산 없음")
    print("[OK] threshold 확정 없음")
    print("[OK] 병변 성능 결론 없음")
    print("[OK] stage2_holdout / v2 미접근")
    print("[OK] 입력 CSV/JSON/MD 수정 없음")
    print("[OK] PNG/HTML/ZIP 생성 없음")
    print("[OK] CT/ROI/mask npy 로드 없음")


if __name__ == "__main__":
    main()
