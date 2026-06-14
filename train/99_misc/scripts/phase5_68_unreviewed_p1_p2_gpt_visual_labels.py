"""
Phase 5.68 unreviewed P1/P2 ChatGPT 1차 visual QA 라벨 저장 스크립트.

- Phase 5.67 manifest read-only 사용
- 기존 파일 수정/삭제/덮어쓰기 금지
- model forward / score 재계산 / stage2_holdout 접근 금지
"""

import json
import pathlib
import datetime
import pandas as pd


def abort_if(condition, message):
    if condition:
        raise RuntimeError(f"[ABORT] {message}")


# ──────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────

BASE = pathlib.Path(
    "/home/jinhy/project/lung-ct-anomaly/outputs"
    "/second-stage-lesion-refiner-v1"
    "/review_annotations/hard_negative_top_score_qa_v1"
)

IN_MANIFEST = (
    BASE
    / "phase5_67_unreviewed_p1_p2_visual_review_pack_v1"
    / "phase5_67_unreviewed_p1_p2_visual_review_manifest_v1.csv"
)

OUT_ROOT = BASE / "phase5_68_unreviewed_p1_p2_gpt_visual_labels_v1"
OUT_CSV = OUT_ROOT / "phase5_68_unreviewed_p1_p2_gpt_visual_labels_v1.csv"
OUT_JSON = OUT_ROOT / "phase5_68_unreviewed_p1_p2_gpt_visual_labels_summary_v1.json"
OUT_MD = OUT_ROOT / "phase5_68_unreviewed_p1_p2_gpt_visual_labels_report_v1.md"


# ──────────────────────────────────────────────
# [Guard 1] output root 기존 존재 시 즉시 중단
# ──────────────────────────────────────────────

abort_if(OUT_ROOT.exists(), f"output root가 이미 존재함 (중단): {OUT_ROOT}")


# ──────────────────────────────────────────────
# [Guard 2] 입력 파일 존재 여부
# ──────────────────────────────────────────────

abort_if(not IN_MANIFEST.exists(), f"입력 파일 없음: {IN_MANIFEST}")


# ──────────────────────────────────────────────
# 입력 파일 로드
# ──────────────────────────────────────────────

manifest = pd.read_csv(IN_MANIFEST)


# ──────────────────────────────────────────────
# [Guard 3] 입력 row 수 / crop_id 검증
# ──────────────────────────────────────────────

abort_if(len(manifest) != 13, f"입력 row 수 오류: {len(manifest)} (기대 13)")


# ──────────────────────────────────────────────
# ChatGPT 1차 visual QA 라벨 (하드코딩)
# ──────────────────────────────────────────────

GPT_LABELS = {
    6306: {
        "user_label": "outside_roi_artifact",
        "user_confidence": "medium",
        "user_note": "ROI 비율이 매우 낮고 작은 고음영 구조가 폐 내부보다는 흉벽 또는 폐외부 영향처럼 보임",
    },
    6940: {
        "user_label": "pleural_wall",
        "user_confidence": "medium",
        "user_note": "횡격막 및 흉막 경계 영향 후보로 보임",
    },
    2254: {
        "user_label": "large_bbox_structure",
        "user_confidence": "medium",
        "user_note": "심장 또는 횡격막 인접 큰 구조물 영향 후보. patch_bbox_clipped 해석 주의",
    },
    2278: {
        "user_label": "outside_roi_artifact",
        "user_confidence": "medium",
        "user_note": "폐하부 경계와 폐외부 또는 흉벽 영향이 커 보임",
    },
    6573: {
        "user_label": "pleural_wall",
        "user_confidence": "medium",
        "user_note": "폐기저부 및 횡격막 경계 영향 후보. patch_bbox_clipped 해석 주의",
    },
    6576: {
        "user_label": "vessel_branch",
        "user_confidence": "medium",
        "user_note": "폐 내부 혈관 또는 혈관분기성 구조가 score에 기여한 후보로 보임",
    },
    6580: {
        "user_label": "pleural_wall",
        "user_confidence": "medium",
        "user_note": "폐하부 및 횡격막/흉막 경계 영향 후보로 보임",
    },
    7001: {
        "user_label": "large_bbox_structure",
        "user_confidence": "medium",
        "user_note": "폐문부 또는 큰 혈관·기관지성 구조 영향 후보. patch_bbox_clipped 해석 주의",
    },
    7018: {
        "user_label": "possible_lesion_overlap",
        "user_confidence": "high",
        "user_note": "context192_overlap_only로 빨간 lesion contour가 보여 FP pool에 바로 섞지 말고 caution으로 분리 필요",
    },
    7033: {
        "user_label": "pleural_wall",
        "user_confidence": "medium",
        "user_note": "폐경계 또는 흉막 인접 구조 영향으로 보임",
    },
    7059: {
        "user_label": "pleural_wall",
        "user_confidence": "medium",
        "user_note": "폐기저부 및 횡격막 경계 영향 후보. patch_bbox_clipped 해석 주의",
    },
    7100: {
        "user_label": "large_bbox_structure",
        "user_confidence": "medium",
        "user_note": "심장·종격동·기관지/공기층 인접 큰 구조 영향 후보로 보임",
    },
    7109: {
        "user_label": "large_bbox_structure",
        "user_confidence": "medium",
        "user_note": "대동맥·종격동·기관지 인접 큰 구조 영향 후보로 보임",
    },
}


# ──────────────────────────────────────────────
# [Guard 4] crop_id 13개 전부 라벨 매칭 확인
# ──────────────────────────────────────────────

manifest_crop_ids = set(manifest["crop_id"].tolist())
label_crop_ids = set(GPT_LABELS.keys())
abort_if(
    manifest_crop_ids != label_crop_ids,
    f"crop_id 불일치: manifest={sorted(manifest_crop_ids)}, labels={sorted(label_crop_ids)}",
)


# ──────────────────────────────────────────────
# recommended_next_action 규칙
# ──────────────────────────────────────────────

FP_CAUSE_LABELS = {"pleural_wall", "large_bbox_structure", "vessel_branch", "bronchus_air_boundary"}


def assign_next_action(user_label):
    if user_label == "possible_lesion_overlap":
        return "separate_context_overlap_caution"
    elif user_label == "outside_roi_artifact":
        return "separate_outside_roi_artifact"
    elif user_label in FP_CAUSE_LABELS:
        return "fp_cause_review_candidate"
    elif user_label == "unclear":
        return "manual_recheck_required"
    else:
        return "manual_recheck_required"


# ──────────────────────────────────────────────
# 출력 DataFrame 구성
# ──────────────────────────────────────────────

out_df = manifest.copy()
out_df["user_label"] = out_df["crop_id"].map(lambda x: GPT_LABELS[x]["user_label"])
out_df["user_confidence"] = out_df["crop_id"].map(lambda x: GPT_LABELS[x]["user_confidence"])
out_df["user_note"] = out_df["crop_id"].map(lambda x: GPT_LABELS[x]["user_note"])
out_df["reviewer"] = "ChatGPT"
out_df["reviewed_at"] = datetime.datetime.now().strftime("%Y-%m-%d")
out_df["gpt_review_status"] = "reviewed_by_chatgpt_first_pass"
out_df["gpt_review_source"] = "ChatGPT_phase5_67_visual_review"
out_df["gpt_review_limit"] = "visual_QA_only_not_medical_diagnosis"
out_df["recommended_next_action"] = out_df["user_label"].map(assign_next_action)


# ──────────────────────────────────────────────
# [Guard 5] 출력 검증
# ──────────────────────────────────────────────

abort_if(len(out_df) != 13, f"출력 row 수 오류: {len(out_df)} (기대 13)")
abort_if(out_df["user_label"].isna().any(), "user_label 누락 존재")
abort_if(out_df["recommended_next_action"].isna().any(), "recommended_next_action 누락 존재")


# ──────────────────────────────────────────────
# 저장 직전 Guard
# ──────────────────────────────────────────────

abort_if(OUT_ROOT.exists(), f"저장 직전 output root가 이미 존재함 (재검증): {OUT_ROOT}")


# ──────────────────────────────────────────────
# 출력 디렉토리 생성
# ──────────────────────────────────────────────

OUT_ROOT.mkdir(parents=True, exist_ok=False)


# ──────────────────────────────────────────────
# CSV 저장
# ──────────────────────────────────────────────

out_df.to_csv(OUT_CSV, index=False)


# ──────────────────────────────────────────────
# 집계
# ──────────────────────────────────────────────

n_by_user_label = out_df["user_label"].value_counts().to_dict()
n_by_next_action = out_df["recommended_next_action"].value_counts().to_dict()

p1_crop_ids = sorted(out_df.loc[out_df["review_priority"] == "P1_unreviewed_pass_both_p99", "crop_id"].tolist())
p2_crop_ids = sorted(out_df.loc[out_df["review_priority"] == "P2_unreviewed_pass_any_p99", "crop_id"].tolist())
possible_lesion_overlap_ids = sorted(out_df.loc[out_df["user_label"] == "possible_lesion_overlap", "crop_id"].tolist())
outside_roi_artifact_ids = sorted(out_df.loc[out_df["user_label"] == "outside_roi_artifact", "crop_id"].tolist())
fp_candidate_ids = sorted(out_df.loc[out_df["recommended_next_action"] == "fp_cause_review_candidate", "crop_id"].tolist())


# ──────────────────────────────────────────────
# JSON 저장
# ──────────────────────────────────────────────

summary = {
    "n_rows": len(out_df),
    "n_by_user_label": n_by_user_label,
    "n_by_recommended_next_action": n_by_next_action,
    "p1_crop_ids": p1_crop_ids,
    "p2_crop_ids": p2_crop_ids,
    "possible_lesion_overlap_crop_ids": possible_lesion_overlap_ids,
    "outside_roi_artifact_crop_ids": outside_roi_artifact_ids,
    "fp_cause_review_candidate_crop_ids": fp_candidate_ids,
    "notes": {
        "visual_QA_only": True,
        "not_medical_diagnosis": True,
        "threshold_not_finalized": True,
        "lesion_conclusion_forbidden": True,
        "hard_negative_not_finalized": True,
        "original_manifest_unmodified": True,
        "original_png_html_zip_unmodified": True,
        "stage2_holdout_unused": True,
        "v2_unused": True,
        "model_forward_not_run": True,
        "score_recalculation_not_run": True,
    },
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# MD report 생성
# ──────────────────────────────────────────────

now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

p1_rows = out_df[out_df["review_priority"] == "P1_unreviewed_pass_both_p99"]
p2_rows = out_df[out_df["review_priority"] == "P2_unreviewed_pass_any_p99"]

md_lines = [
    "# Phase 5.68 Unreviewed P1/P2 ChatGPT Visual QA Labels Report",
    "",
    f"생성일시: {now_str}",
    "",
    "---",
    "",
    "## 1. Phase 5.68 목적",
    "",
    "Phase 5.67에서 생성한 P1/P2 visual review pack 13개에 대해 ChatGPT 1차 visual QA 라벨을 기록한다. "
    "기존 Phase 5.67 manifest, PNG, HTML, ZIP은 수정하지 않으며, "
    "이 라벨은 의학적 확정이나 threshold 확정이 아닌 QA용 참고 라벨이다.",
    "",
    "---",
    "",
    "## 2. Phase 5.67 P1/P2 pack 요약",
    "",
    "| 항목 | 값 |",
    "|------|-----|",
    f"| 총 대상 | 13개 |",
    f"| P1 (pass_both_p99) | {len(p1_crop_ids)}개 |",
    f"| P2 (pass_any_p99) | {len(p2_crop_ids)}개 |",
    f"| P1 crop_id | {p1_crop_ids} |",
    f"| P2 crop_id | {p2_crop_ids} |",
    "",
    "---",
    "",
    "## 3. ChatGPT 1차 visual QA 라벨 요약",
    "",
    "| user_label | count |",
    "|------------|-------|",
]

for label, cnt in sorted(n_by_user_label.items(), key=lambda x: -x[1]):
    md_lines.append(f"| {label} | {cnt} |")

md_lines += [
    "",
    "| recommended_next_action | count |",
    "|-------------------------|-------|",
]

for action, cnt in sorted(n_by_next_action.items(), key=lambda x: -x[1]):
    md_lines.append(f"| {action} | {cnt} |")

md_lines += [
    "",
    "---",
    "",
    "## 4. P1 2개 판정",
    "",
]

for _, row in p1_rows.iterrows():
    md_lines.append(
        f"- **crop_id {row['crop_id']}** ({row['patient_id']}): "
        f"`{row['user_label']}` [{row['user_confidence']}] — {row['user_note']}"
    )
    md_lines.append(f"  → **{row['recommended_next_action']}**")
    md_lines.append("")

md_lines += [
    "---",
    "",
    "## 5. P2 11개 판정",
    "",
]

for _, row in p2_rows.iterrows():
    md_lines.append(
        f"- **crop_id {row['crop_id']}** ({row['patient_id']}): "
        f"`{row['user_label']}` [{row['user_confidence']}] — {row['user_note']}"
    )
    md_lines.append(f"  → **{row['recommended_next_action']}**")
    md_lines.append("")

md_lines += [
    "---",
    "",
    "## 6. possible_lesion_overlap 분리 필요",
    "",
    f"crop_id {possible_lesion_overlap_ids} — FP pool에 바로 섞지 말고 separate_context_overlap_caution으로 분리.",
    "",
    "---",
    "",
    "## 7. outside_roi_artifact 분리 필요",
    "",
    f"crop_id {outside_roi_artifact_ids} — separate_outside_roi_artifact로 분리.",
    "",
    "---",
    "",
    "## 8. fp_cause_review_candidate 목록",
    "",
    f"crop_id {fp_candidate_ids} ({len(fp_candidate_ids)}개) — Phase 5.58 기존 FP cause pool 29개와 합산 검토 대상.",
    "",
    "---",
    "",
    "## 9. 해석 제한",
    "",
    "- 이 라벨은 QA용 visual label이며 의학적 확정이 아니다.",
    "- threshold 확정이 아니다.",
    "- 병변 성능 결론이 아니다.",
    "- hard negative 최종 채택이 아니다.",
    "",
    "---",
    "",
    "## 10. 다음 단계 제안",
    "",
    "1. **Phase 5.69 검토**: Phase 5.58 기존 FP cause pool 29개 + Phase 5.68 신규 P1/P2 FP 후보를 합쳐 FP cause summary를 업데이트할지 검토.",
    "2. **possible_lesion_overlap (crop_id 7018)과 outside_roi_artifact (crop_id 6306, 2278)는 계속 분리 유지.**",
    "3. **stage2_holdout 계속 봉인** — threshold 확정 및 hard negative 최종 채택 이후에만 접근 검토.",
]

with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines) + "\n")


# ──────────────────────────────────────────────
# 최종 보고 출력
# ──────────────────────────────────────────────

print("=" * 60)
print("검토 판정: 전체 통과")
print("=" * 60)
print(f"output root    : {OUT_ROOT}")
print(f"생성 CSV       : {OUT_CSV}")
print(f"생성 JSON      : {OUT_JSON}")
print(f"생성 MD        : {OUT_MD}")
print(f"row 수         : {len(out_df)}")
print("-" * 60)
print("n_by_user_label:")
for k, v in sorted(n_by_user_label.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")
print("n_by_recommended_next_action:")
for k, v in sorted(n_by_next_action.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")
print("-" * 60)
print(f"possible_lesion_overlap crop_id  : {possible_lesion_overlap_ids}")
print(f"outside_roi_artifact crop_id     : {outside_roi_artifact_ids}")
print(f"fp_cause_review_candidate crop_id: {fp_candidate_ids}")
print("-" * 60)
print("기존 Phase 5.67 manifest 미수정 : True")
print("기존 PNG/HTML/ZIP 미수정        : True")
print("model forward 없음              : True")
print("score 재계산 없음               : True")
print("threshold 확정 없음             : True")
print("병변 성능 결론 없음             : True")
print("stage2_holdout/v2 미접근        : True")
print("=" * 60)
print()
print("[다음 단계 제안]")
print("1. Phase 5.69에서 Phase 5.58 FP cause pool 29개 + Phase 5.68 신규 P1/P2 FP 후보 합산 FP cause summary 업데이트 검토")
print("2. possible_lesion_overlap (crop 7018), outside_roi_artifact (crop 6306, 2278) 계속 분리 유지")
print("3. stage2_holdout 계속 봉인")
