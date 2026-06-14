"""
Phase 5.89 Hard Negative Candidate Manifest Preflight
=====================================================
목적: Phase 5.86 accepted labels 14행을 바탕으로 hard negative candidate manifest
      preflight table 생성. 실제 training manifest가 아닌 preflight-only 결과물.

금지:
- CT/ROI/mask npy 로드 금지
- model forward 금지
- score 재계산 금지
- crop manifest 생성/수정 금지
- training dataset 수정 금지
- stage2_holdout / v2 접근 금지
"""

import json
import os
from pathlib import Path

import pandas as pd

# ===========================================================================
# 경로 설정
# ===========================================================================
BASE = Path("/home/jinhy/project/lung-ct-anomaly")

INPUT_P86_CSV = BASE / "outputs/second-stage-lesion-refiner-v1/review_annotations/phase5_86_weak_3d_visual_review_label_acceptance_v1/phase5_86_weak_3d_visual_review_user_accepted_labels_v1.csv"
INPUT_P86_JSON = BASE / "outputs/second-stage-lesion-refiner-v1/review_annotations/phase5_86_weak_3d_visual_review_label_acceptance_v1/phase5_86_weak_3d_visual_review_label_acceptance_summary_v1.json"
INPUT_P87_CSV = BASE / "outputs/second-stage-lesion-refiner-v1/review_annotations/phase5_87_fp_cause_multi_pool_expanded_summary_v1/phase5_87_fp_cause_multi_pool_expanded_summary_v1.csv"
INPUT_P87_JSON = BASE / "outputs/second-stage-lesion-refiner-v1/review_annotations/phase5_87_fp_cause_multi_pool_expanded_summary_v1/phase5_87_fp_cause_multi_pool_expanded_summary_v1.json"
INPUT_P88_CSV = BASE / "outputs/second-stage-lesion-refiner-v1/review_annotations/phase5_88_hard_negative_candidate_policy_preflight_v1/phase5_88_hard_negative_candidate_policy_preflight_v1.csv"
INPUT_P88_JSON = BASE / "outputs/second-stage-lesion-refiner-v1/review_annotations/phase5_88_hard_negative_candidate_policy_preflight_v1/phase5_88_hard_negative_candidate_policy_preflight_v1.json"
INPUT_P88_MD = BASE / "outputs/second-stage-lesion-refiner-v1/review_annotations/phase5_88_hard_negative_candidate_policy_preflight_v1/phase5_88_hard_negative_candidate_policy_preflight_report_v1.md"

OUTPUT_ROOT = BASE / "outputs/second-stage-lesion-refiner-v1/review_annotations/phase5_89_hard_negative_candidate_manifest_preflight_v1"
OUTPUT_CSV = OUTPUT_ROOT / "phase5_89_hard_negative_candidate_manifest_preflight_v1.csv"
OUTPUT_JSON = OUTPUT_ROOT / "phase5_89_hard_negative_candidate_manifest_preflight_v1.json"
OUTPUT_MD = OUTPUT_ROOT / "phase5_89_hard_negative_candidate_manifest_preflight_report_v1.md"

# ===========================================================================
# 매핑 테이블
# ===========================================================================
LABEL_TO_CANDIDATE_CLASS = {
    "pleural_wall": "pleural_wall_boundary_candidate",
    "large_bbox_structure": "vessel_mediastinal_hilar_candidate",
    "vessel_branch": "vessel_mediastinal_hilar_candidate",
    "bronchus_air_boundary": "bronchus_air_boundary_candidate",
}

LABEL_TO_CANDIDATE_STATUS = {
    "pleural_wall": "review_candidate_only",
    "large_bbox_structure": "review_candidate_only",
    "vessel_branch": "review_candidate_only",
    "bronchus_air_boundary": "review_candidate_only",
    "outside_roi_artifact": "quarantine_pending",
    "unclear": "quarantine_pending",
    "possible_lesion_overlap": "quarantine_pending",
    "z_overmerge_suspicious_overmerge": "quarantine_pending",
    "z_overmerge_ok_continuous_structure": "metadata_caution_only",
}

LABEL_TO_RISK_FLAG = {
    "pleural_wall": "pleural_boundary_fp_axis",
    "large_bbox_structure": "large_structure_fp_axis",
    "vessel_branch": "vessel_fp_axis",
    "bronchus_air_boundary": "airway_boundary_fp_axis",
}

ALLOWED_LABELS = set(LABEL_TO_CANDIDATE_CLASS.keys())


def check_inputs():
    """입력 파일 존재 확인 (read-only, 파일 수정 없음)."""
    required = [
        INPUT_P86_CSV, INPUT_P86_JSON,
        INPUT_P87_CSV, INPUT_P87_JSON,
        INPUT_P88_CSV, INPUT_P88_JSON, INPUT_P88_MD,
    ]
    for p in required:
        assert p.exists(), f"입력 파일 없음: {p}"
    print("[검증] 모든 입력 파일 존재 확인 완료")


def check_outputs_not_exist():
    """출력 파일/폴더가 이미 존재하면 즉시 중단 (overwrite 방지)."""
    if OUTPUT_ROOT.exists():
        raise SystemExit(f"[중단] OUTPUT_ROOT가 이미 존재합니다. 덮어쓰기 금지: {OUTPUT_ROOT}")
    for p in [OUTPUT_CSV, OUTPUT_JSON, OUTPUT_MD]:
        if p.exists():
            raise SystemExit(f"[중단] 출력 파일이 이미 존재합니다. 덮어쓰기 금지: {p}")
    print("[검증] 출력 파일/폴더 미존재 확인 완료")


def build_risk_note(row):
    """risk_note 생성: overmerge + label 기반 flag 조합."""
    flags = []
    label = row["user_label"]
    if row.get("overmerge_flag", False) is True or str(row.get("overmerge_flag", "")).lower() == "true":
        flags.append("z_overmerge_metadata_caution")
    if label in LABEL_TO_RISK_FLAG:
        flags.append(LABEL_TO_RISK_FLAG[label])
    return ";".join(flags) if flags else ""


def load_p88_lookup(p88_df):
    """
    Phase 5.88 CSV에서 두 가지 lookup 딕셔너리를 구성:
    - class_lookup: candidate_class → {section, allowed_next_use, forbidden_use, limitation}
    - label_approval_lookup: user_label → approval_required_before_use
    """
    # Section A: candidate_class 기준
    sec_a = p88_df[p88_df["section"] == "A"].dropna(subset=["candidate_class"])
    class_lookup = {}
    for _, r in sec_a.iterrows():
        key = str(r["candidate_class"]).strip()
        class_lookup[key] = {
            "section": str(r.get("section", "A")),
            "allowed_next_use": str(r.get("allowed_next_use", "")) if pd.notna(r.get("allowed_next_use")) else "",
            "forbidden_use": str(r.get("forbidden_use", "")) if pd.notna(r.get("forbidden_use")) else "",
            "limitation": str(r.get("limitation", "")) if pd.notna(r.get("limitation")) else "",
        }

    # Section B: user_label 기준
    sec_b = p88_df[p88_df["section"] == "B"].dropna(subset=["user_label"])
    label_approval_lookup = {}
    for _, r in sec_b.iterrows():
        key = str(r["user_label"]).strip()
        val = r.get("required_approval_before_use", "")
        label_approval_lookup[key] = str(val) if pd.notna(val) else "yes"

    return class_lookup, label_approval_lookup


def build_manifest(p86_df, class_lookup, label_approval_lookup):
    """Phase 5.86 14행 → Phase 5.89 manifest DataFrame 구성."""
    rows = []
    for _, r in p86_df.iterrows():
        label = str(r["user_label"]).strip()

        # 허용 label 검증
        assert label in ALLOWED_LABELS, f"허용되지 않은 label: {label}"

        candidate_class = LABEL_TO_CANDIDATE_CLASS[label]
        candidate_status = LABEL_TO_CANDIDATE_STATUS[label]
        candidate_use_scope = "requires_user_approval_before_manifest_use"

        # risk_note
        overmerge_raw = r.get("overmerge_flag", False)
        overmerge_bool = (str(overmerge_raw).strip().lower() in ("true", "1", "yes"))
        risk_flags = []
        if overmerge_bool:
            risk_flags.append("z_overmerge_metadata_caution")
        if label in LABEL_TO_RISK_FLAG:
            risk_flags.append(LABEL_TO_RISK_FLAG[label])
        risk_note = ";".join(risk_flags)

        # Phase 5.88 join
        cls_info = class_lookup.get(candidate_class, {})
        section = cls_info.get("section", "A")
        allowed_next_use = cls_info.get("allowed_next_use", "see_phase5_88_policy")
        forbidden_use = cls_info.get("forbidden_use", "see_phase5_88_policy")
        limitation = cls_info.get("limitation", "see_phase5_88_policy")
        approval_required_before_use = label_approval_lookup.get(label, "yes")

        rows.append({
            "section": section,
            "review_order": r["review_order"],
            "patient_id": r["patient_id"],
            "cluster3d_id": r["cluster3d_id"],
            "user_label": label,
            "user_note": r.get("user_note", ""),
            "candidate_class": candidate_class,
            "candidate_status": candidate_status,
            "candidate_use_scope": candidate_use_scope,
            "risk_note": risk_note,
            "source_phase": "phase5_86",
            "label_status": r.get("label_status", "user_accepted_for_phase5_86"),
            "review_candidate_flag": True,
            "overmerge_flag": overmerge_bool,
            "z_span": r.get("z_span", ""),
            "n_2d_clusters": r.get("n_2d_clusters", ""),
            "n_patches_total": r.get("n_patches_total", ""),
            "bbox_area": r.get("bbox_area", ""),
            "top3_mean_patch_score_3d": r.get("top3_mean_patch_score_3d", ""),
            "png_path": r.get("png_path", ""),
            "html_relative_png_path": r.get("html_relative_png_path", ""),
            "allowed_next_use": allowed_next_use,
            "forbidden_use": forbidden_use,
            "approval_required_before_use": approval_required_before_use,
            "limitation": limitation,
        })

    df = pd.DataFrame(rows)
    return df


def build_summary_stats(df):
    """요약 통계 dict 반환."""
    n_by_status = df["candidate_status"].value_counts().to_dict()
    n_by_class = df["candidate_class"].value_counts().to_dict()
    n_by_label = df["user_label"].value_counts().to_dict()
    n_by_patient = df["patient_id"].value_counts().to_dict()
    n_by_overmerge = df["overmerge_flag"].value_counts().to_dict()
    n_by_overmerge = {str(k): int(v) for k, v in n_by_overmerge.items()}

    n_review_candidate_only = int((df["candidate_status"] == "review_candidate_only").sum())
    n_quarantine_pending = int((df["candidate_status"] == "quarantine_pending").sum())
    n_metadata_caution_only = int((df["candidate_status"] == "metadata_caution_only").sum())
    n_training_ready = 0  # 항상 0

    return {
        "n_by_candidate_status": {k: int(v) for k, v in n_by_status.items()},
        "n_by_candidate_class": {k: int(v) for k, v in n_by_class.items()},
        "n_by_user_label": {k: int(v) for k, v in n_by_label.items()},
        "n_by_patient_id": {k: int(v) for k, v in n_by_patient.items()},
        "n_by_overmerge_flag": n_by_overmerge,
        "n_review_candidate_only": n_review_candidate_only,
        "n_quarantine_pending": n_quarantine_pending,
        "n_metadata_caution_only": n_metadata_caution_only,
        "n_training_ready": n_training_ready,
    }


def build_json_output(df, stats):
    """출력 JSON 구성."""
    return {
        "phase": "phase5_89",
        "preflight_only": True,
        "not_training_manifest": True,
        "no_hard_negative_final_selection": True,
        "no_training_dataset_modification": True,
        "no_crop_manifest_modification": True,
        "no_threshold_finalization": True,
        "no_lesion_overlap_assessment": True,
        "no_performance_conclusion": True,
        "stage2_holdout_unused": True,
        "v2_unused": True,
        "input_paths": {
            "phase5_86_csv": str(INPUT_P86_CSV),
            "phase5_86_json": str(INPUT_P86_JSON),
            "phase5_87_csv": str(INPUT_P87_CSV),
            "phase5_87_json": str(INPUT_P87_JSON),
            "phase5_88_csv": str(INPUT_P88_CSV),
            "phase5_88_json": str(INPUT_P88_JSON),
            "phase5_88_md": str(INPUT_P88_MD),
        },
        "output_csv_path": str(OUTPUT_CSV),
        "n_rows": int(len(df)),
        "n_by_candidate_status": stats["n_by_candidate_status"],
        "n_by_candidate_class": stats["n_by_candidate_class"],
        "n_by_user_label": stats["n_by_user_label"],
        "n_by_patient_id": stats["n_by_patient_id"],
        "n_by_overmerge_flag": stats["n_by_overmerge_flag"],
        "n_training_ready": stats["n_training_ready"],
        "hard_negative_finalized": False,
        "preflight_conclusion": (
            "14 rows are eligible as review_candidate_only / policy discussion candidates. "
            "0 rows are training-ready hard negatives. "
            "No row is finally selected as hard negative. "
            "A later phase must decide whether to create a true hard negative manifest "
            "and how to avoid bias/leakage. "
            "Any training use requires a separate approved design."
        ),
        "limitations": [
            "Phase 5.86 labels are normal patient FP cause review candidates, not lesion overlap assessments.",
            "Phase 5.87 multi-pool comparison is reference-only, not pooled prevalence.",
            "Phase 5.88 policy is preflight-only, not final hard negative selection.",
            "This table is preflight-only; no training manifest has been created.",
            "No threshold has been finalized.",
            "No lesion performance conclusion is drawn.",
        ],
        "notes": {
            "phase5_86_input_readonly": True,
            "phase5_87_input_readonly": True,
            "phase5_88_input_readonly": True,
            "existing_outputs_unmodified": True,
            "no_model_forward": True,
            "no_score_recalculation": True,
            "threshold_not_finalized": True,
            "lesion_conclusion_forbidden": True,
            "hard_negative_not_finalized": True,
            "training_dataset_unmodified": True,
            "crop_manifest_unmodified": True,
            "stage2_holdout_unused": True,
            "v2_unused": True,
        },
    }


def build_md_report(df, stats, output_data):
    """MD 보고서 작성."""
    n_by_status = stats["n_by_candidate_status"]
    n_by_class = stats["n_by_candidate_class"]
    n_by_label = stats["n_by_user_label"]
    n_overmerge = stats["n_by_overmerge_flag"]
    overmerge_true = n_overmerge.get("True", 0)
    overmerge_false = n_overmerge.get("False", 0)

    # overmerge × label 교차 집계
    overmerge_by_label = df.groupby(["user_label", "overmerge_flag"]).size().unstack(fill_value=0)

    status_table = "\n".join(
        f"| {k} | {v} |" for k, v in n_by_status.items()
    )
    class_table = "\n".join(
        f"| {k} | {v} |" for k, v in n_by_class.items()
    )
    label_table = "\n".join(
        f"| {k} | {v} |" for k, v in n_by_label.items()
    )

    overmerge_by_label_lines = []
    for lbl, row_s in overmerge_by_label.iterrows():
        t_val = int(row_s.get(True, 0))
        f_val = int(row_s.get(False, 0))
        overmerge_by_label_lines.append(f"| {lbl} | {t_val} | {f_val} |")
    overmerge_by_label_table = "\n".join(overmerge_by_label_lines)

    md = f"""# Phase 5.89 Hard Negative Candidate Manifest Preflight Report

> **주의**: 이 보고서는 preflight-only 결과입니다. 실제 training manifest가 아니며, hard negative 최종 채택 결과가 아닙니다.

---

## 1. Phase 5.89 목적

Phase 5.88 hard negative candidate policy preflight를 바탕으로, 실제 hard negative 후보 manifest를 만들기 전에 사전 후보표를 설계한다.

- 이번 단계는 manifest "preflight"다.
- 실제 hard negative training manifest를 생성하지 않는다.
- 실제 crop manifest를 생성하거나 수정하지 않는다.
- 후보 status만 정리한다.

---

## 2. 입력 파일

| 파일 | 경로 |
|------|------|
| Phase 5.86 CSV | `{INPUT_P86_CSV}` |
| Phase 5.86 JSON | `{INPUT_P86_JSON}` |
| Phase 5.87 CSV | `{INPUT_P87_CSV}` |
| Phase 5.87 JSON | `{INPUT_P87_JSON}` |
| Phase 5.88 CSV | `{INPUT_P88_CSV}` |
| Phase 5.88 JSON | `{INPUT_P88_JSON}` |
| Phase 5.88 MD | `{INPUT_P88_MD}` |

---

## 3. Candidate Status Rule

| user_label | candidate_status |
|------------|-----------------|
| pleural_wall | review_candidate_only |
| large_bbox_structure | review_candidate_only |
| vessel_branch | review_candidate_only |
| bronchus_air_boundary | review_candidate_only |
| outside_roi_artifact | quarantine_pending |
| unclear | quarantine_pending |
| possible_lesion_overlap | quarantine_pending |
| z_overmerge_suspicious_overmerge | quarantine_pending |
| z_overmerge_ok_continuous_structure | metadata_caution_only |

---

## 4. Candidate Class Rule

| user_label | candidate_class |
|------------|----------------|
| pleural_wall | pleural_wall_boundary_candidate |
| large_bbox_structure | vessel_mediastinal_hilar_candidate |
| vessel_branch | vessel_mediastinal_hilar_candidate |
| bronchus_air_boundary | bronchus_air_boundary_candidate |

---

## 5. Preflight Manifest Summary

- **총 행 수**: {len(df)}
- **입력 Phase 5.86 row 수**: 14
- **n_training_ready**: 0
- **hard_negative_finalized**: false

---

## 6. Candidate Status Count

| candidate_status | count |
|-----------------|-------|
{status_table}

---

## 7. Candidate Class Count

| candidate_class | count |
|----------------|-------|
{class_table}

---

## 8. User Label Count

| user_label | count |
|------------|-------|
{label_table}

---

## 9. Overmerge Caution 요약

- overmerge_flag=True: **{overmerge_true}행**
- overmerge_flag=False: **{overmerge_false}행**

### Label별 overmerge_flag 분포

| user_label | overmerge_flag=True | overmerge_flag=False |
|------------|--------------------|--------------------|
{overmerge_by_label_table}

- overmerge_flag=True인 행: risk_note에 `z_overmerge_metadata_caution` 포함

---

## 10. 핵심 결론

- **review_candidate_only 후보**: 14개 (모든 행)
- **training-ready hard negative**: 0개
- **hard negative 최종 채택**: 아님
- **실제 학습/manifest 사용**: 별도 승인 필요 (`requires_user_approval_before_manifest_use`)

---

## 11. 해석 제한

- Phase 5.86 14개 rows는 normal patient FP cause review 후보이며, 병변 overlap 평가가 아니다.
- Phase 5.87 multi-pool 비교는 reference-only이며 pooled prevalence가 아니다.
- Phase 5.88 policy는 hard negative 최종 채택이 아니라 policy preflight다.
- 이 CSV는 training manifest가 아니라 preflight table이다.
- threshold가 확정되지 않았다.
- 병변 성능 결론을 내릴 수 없다.

---

## 12. 다음 단계

**A. Phase 5.90** — True hard negative manifest design plan
- 이 preflight table을 입력으로 사용
- bias/leakage 방지 전략 설계
- 실제 training manifest 생성 기준 정의

**B. Vessel / Mediastinal / Pleural Rule Candidate Design**
- vessel_mediastinal_hilar_candidate (12행) 세부 분류 기준 수립
- pleural_wall_boundary_candidate (1행) 경계 판단 기준 수립

**C. 현재 결과 Handoff 최신화**
- Phase 5.89 preflight 결과를 handoff 문서에 반영

---

## 제한 사항 (Limitations)

- preflight_only: True
- not_training_manifest: True
- no_hard_negative_final_selection: True
- no_training_dataset_modification: True
- no_crop_manifest_modification: True
- no_threshold_finalization: True
- no_lesion_overlap_assessment: True
- no_performance_conclusion: True
- stage2_holdout_unused: True
- v2_unused: True
"""
    return md


def run_validations(df, p86_df):
    """검증 항목 실행."""
    assert len(p86_df) == 14, f"Phase 5.86 row 수 오류: {len(p86_df)}"
    assert (p86_df["label_status"] == "user_accepted_for_phase5_86").all(), "label_status 불일치"

    label_counts = p86_df["user_label"].value_counts().to_dict()
    assert label_counts.get("vessel_branch", 0) == 6, f"vessel_branch 수 오류: {label_counts}"
    assert label_counts.get("large_bbox_structure", 0) == 6, f"large_bbox_structure 수 오류"
    assert label_counts.get("pleural_wall", 0) == 1, f"pleural_wall 수 오류"
    assert label_counts.get("bronchus_air_boundary", 0) == 1, f"bronchus_air_boundary 수 오류"

    unexpected = set(p86_df["user_label"].unique()) - ALLOWED_LABELS
    assert len(unexpected) == 0, f"허용 label 외 값 존재: {unexpected}"

    assert df["candidate_status"].notna().all(), "candidate_status 빈칸 존재"
    assert df["candidate_class"].notna().all(), "candidate_class 빈칸 존재"
    assert (df["candidate_status"] != "").all(), "candidate_status 빈 문자열 존재"
    assert (df["candidate_class"] != "").all(), "candidate_class 빈 문자열 존재"

    n_training_ready = 0
    assert n_training_ready == 0, "n_training_ready 오류"

    print("[검증] 모든 검증 항목 통과")
    print(f"  - Phase 5.86 row 수: {len(p86_df)}")
    print(f"  - label_status=user_accepted_for_phase5_86: 전체")
    print(f"  - label count: vessel_branch={label_counts.get('vessel_branch',0)}, "
          f"large_bbox_structure={label_counts.get('large_bbox_structure',0)}, "
          f"pleural_wall={label_counts.get('pleural_wall',0)}, "
          f"bronchus_air_boundary={label_counts.get('bronchus_air_boundary',0)}")
    print(f"  - 허용 label 외 값: 0건")
    print(f"  - candidate_status 빈칸: 0건")
    print(f"  - candidate_class 빈칸: 0건")
    print(f"  - n_training_ready: 0")
    print(f"  - hard_negative_finalized: False")


def main():
    print("=" * 60)
    print("Phase 5.89 Hard Negative Candidate Manifest Preflight")
    print("=" * 60)

    # 1. 입력 파일 존재 확인
    check_inputs()

    # 2. 출력 파일/폴더 사전 guard (로드 전 빠른 실패)
    check_outputs_not_exist()

    # 3. Phase 5.86 로드
    p86_df = pd.read_csv(INPUT_P86_CSV)
    print(f"[로드] Phase 5.86 CSV: {len(p86_df)}행")

    # 3. Phase 5.88 로드 및 lookup 구성
    p88_df = pd.read_csv(INPUT_P88_CSV)
    print(f"[로드] Phase 5.88 CSV: {len(p88_df)}행")
    class_lookup, label_approval_lookup = load_p88_lookup(p88_df)
    print(f"  - class_lookup keys: {list(class_lookup.keys())}")
    print(f"  - label_approval_lookup keys: {list(label_approval_lookup.keys())}")

    # 4. manifest 구성
    df = build_manifest(p86_df, class_lookup, label_approval_lookup)
    print(f"[구성] manifest rows: {len(df)}")

    # 5. 검증
    run_validations(df, p86_df)

    # 6. 통계
    stats = build_summary_stats(df)

    # 8. 출력 폴더 생성 (모든 검증·구성 완료 후)
    # 저장 직전 출력 파일/폴더 재확인 guard
    if OUTPUT_ROOT.exists():
        raise SystemExit(f"[중단] OUTPUT_ROOT가 존재합니다. 덮어쓰기 금지: {OUTPUT_ROOT}")
    for p in [OUTPUT_CSV, OUTPUT_JSON, OUTPUT_MD]:
        if p.exists():
            raise SystemExit(f"[중단] 출력 파일이 존재합니다. 덮어쓰기 금지: {p}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

    # 9. CSV 저장
    col_order = [
        "section", "review_order", "patient_id", "cluster3d_id",
        "user_label", "user_note", "candidate_class", "candidate_status",
        "candidate_use_scope", "risk_note", "source_phase", "label_status",
        "review_candidate_flag", "overmerge_flag", "z_span", "n_2d_clusters",
        "n_patches_total", "bbox_area", "top3_mean_patch_score_3d",
        "png_path", "html_relative_png_path",
        "allowed_next_use", "forbidden_use", "approval_required_before_use", "limitation",
    ]
    df[col_order].to_csv(OUTPUT_CSV, index=False)
    print(f"[저장] CSV: {OUTPUT_CSV}")

    # 9. JSON 저장
    output_data = build_json_output(df, stats)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"[저장] JSON: {OUTPUT_JSON}")

    # 10. MD 저장
    md_text = build_md_report(df, stats, output_data)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"[저장] MD: {OUTPUT_MD}")

    # 11. 최종 보고
    print()
    print("=" * 60)
    print("최종 보고")
    print("=" * 60)
    print(f"출력 root: {OUTPUT_ROOT}")
    print(f"CSV: {OUTPUT_CSV.name}")
    print(f"JSON: {OUTPUT_JSON.name}")
    print(f"MD: {OUTPUT_MD.name}")
    print(f"Phase 5.86 row 수: {len(p86_df)}")
    print(f"candidate_status count: {stats['n_by_candidate_status']}")
    print(f"candidate_class count: {stats['n_by_candidate_class']}")
    print(f"user_label count: {stats['n_by_user_label']}")
    print(f"overmerge_flag count: {stats['n_by_overmerge_flag']}")
    print(f"n_training_ready: {stats['n_training_ready']}")
    print(f"hard_negative_finalized: False")
    print()
    print("preflight conclusion:")
    print("  - 14 rows are eligible as review_candidate_only")
    print("  - 0 rows are training-ready hard negatives")
    print("  - No row is finally selected as hard negative")
    print("  - Any training use requires a separate approved design")
    print()
    print("[완료] Phase 5.89 preflight 생성 완료")


if __name__ == "__main__":
    main()
