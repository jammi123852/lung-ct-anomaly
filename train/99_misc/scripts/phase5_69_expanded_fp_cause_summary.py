"""
Phase 5.69 expanded FP cause summary 생성 스크립트.

- Phase 5.58 FP cause pool 29개 + Phase 5.68 신규 FP candidate 10개 합산
- possible_lesion_overlap 1개, outside_roi_artifact 2개는 분리 유지
- 기존 파일 수정/삭제/덮어쓰기 금지
- model forward / score 재계산 / stage2_holdout 접근 금지
"""

import json
import pathlib
import datetime
import io
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

IN_5_58_CSV = (
    BASE
    / "phase5_58_final_fp_cause_table_v1"
    / "phase5_58_final_fp_cause_table_v1.csv"
)
IN_5_68_CSV = (
    BASE
    / "phase5_68_unreviewed_p1_p2_gpt_visual_labels_v1"
    / "phase5_68_unreviewed_p1_p2_gpt_visual_labels_v1.csv"
)
IN_5_68_JSON = (
    BASE
    / "phase5_68_unreviewed_p1_p2_gpt_visual_labels_v1"
    / "phase5_68_unreviewed_p1_p2_gpt_visual_labels_summary_v1.json"
)

OUT_ROOT = BASE / "phase5_69_expanded_fp_cause_summary_v1"
OUT_CSV = OUT_ROOT / "phase5_69_expanded_fp_cause_summary_v1.csv"
OUT_JSON = OUT_ROOT / "phase5_69_expanded_fp_cause_summary_v1.json"
OUT_MD = OUT_ROOT / "phase5_69_expanded_fp_cause_summary_report_v1.md"

CAUSE_ORDER = ["pleural_wall", "large_bbox_structure", "vessel_branch", "bronchus_air_boundary"]

EXPECTED = {
    "pleural_wall": 23,
    "large_bbox_structure": 12,
    "vessel_branch": 3,
    "bronchus_air_boundary": 1,
}
EXPECTED_TOTAL = 39


# ──────────────────────────────────────────────
# [Guard 1] output root 기존 존재 시 즉시 중단
# ──────────────────────────────────────────────

abort_if(OUT_ROOT.exists(), f"output root가 이미 존재함 (중단): {OUT_ROOT}")


# ──────────────────────────────────────────────
# [Guard 2] 입력 파일 존재 여부
# ──────────────────────────────────────────────

for p in [IN_5_58_CSV, IN_5_68_CSV, IN_5_68_JSON]:
    abort_if(not p.exists(), f"입력 파일 없음: {p}")


# ──────────────────────────────────────────────
# Phase 5.58 SECTION_B 로드 (섹션 구조 파싱)
# ──────────────────────────────────────────────

with open(IN_5_58_CSV, encoding="utf-8") as f:
    raw_lines = f.readlines()

section_b_start = None
for i, line in enumerate(raw_lines):
    if line.startswith("# SECTION_B"):
        section_b_start = i
        break

abort_if(section_b_start is None, "Phase 5.58 CSV에 SECTION_B 없음")

df_58 = pd.read_csv(io.StringIO("".join(raw_lines[section_b_start + 1:])))


# ──────────────────────────────────────────────
# [Guard 3] Phase 5.58 row 수 / cause label 검증
# ──────────────────────────────────────────────

abort_if(len(df_58) != 29, f"Phase 5.58 FP pool row 수 오류: {len(df_58)} (기대 29)")
abort_if("final_fp_cause_label" not in df_58.columns, "Phase 5.58에 final_fp_cause_label 없음")


# ──────────────────────────────────────────────
# Phase 5.68 로드
# ──────────────────────────────────────────────

df_68 = pd.read_csv(IN_5_68_CSV)


# ──────────────────────────────────────────────
# [Guard 4] Phase 5.68 row 수 / 그룹 수 검증
# ──────────────────────────────────────────────

abort_if(len(df_68) != 13, f"Phase 5.68 row 수 오류: {len(df_68)} (기대 13)")

fp_candidates = df_68[df_68["recommended_next_action"] == "fp_cause_review_candidate"]
abort_if(len(fp_candidates) != 10, f"Phase 5.68 fp_cause_review_candidate 수 오류: {len(fp_candidates)} (기대 10)")

possible_lesion = df_68[df_68["user_label"] == "possible_lesion_overlap"]
abort_if(len(possible_lesion) != 1, f"Phase 5.68 possible_lesion_overlap 수 오류: {len(possible_lesion)} (기대 1)")

outside_roi = df_68[df_68["user_label"] == "outside_roi_artifact"]
abort_if(len(outside_roi) != 2, f"Phase 5.68 outside_roi_artifact 수 오류: {len(outside_roi)} (기대 2)")


# ──────────────────────────────────────────────
# Phase 5.58 detailed rows 구성 (출력 스키마 맞춤)
# ──────────────────────────────────────────────

rows_58 = []
for _, row in df_58.iterrows():
    rows_58.append({
        "source_phase": "phase5_58",
        "crop_id": row["crop_id"],
        "patient_id": row["patient_id"],
        "final_fp_cause_label": row["final_fp_cause_label"],
        "user_label": row["final_fp_cause_label"],
        "user_confidence": row.get("user_verified_confidence", ""),
        "user_note": row.get("user_verified_note", ""),
        "review_priority": "",
        "recommended_next_action": "fp_cause_review_candidate",
        "rd4ad_l1_roi_mean": "",
        "rd4ad_l1_lung_channel_roi_mean": "",
        "rd4ad_l1_roi_patch_mean": row.get("rd4ad_l1_roi_patch_mean", ""),
        "rd4ad_l1_outside_roi_mean": row.get("rd4ad_l1_outside_roi_mean", ""),
        "roi_ratio_fixed96": "",
        "patch_roi_ratio": "",
        "patch_bbox_clipped": row.get("patch_bbox_clipped", ""),
        "png_path": row.get("png_path", ""),
        "contact_sheet_path": row.get("contact_sheet_path", ""),
    })

df_58_detail = pd.DataFrame(rows_58)


# ──────────────────────────────────────────────
# Phase 5.68 FP candidate rows 구성
# ──────────────────────────────────────────────

rows_68 = []
for _, row in fp_candidates.iterrows():
    rows_68.append({
        "source_phase": "phase5_68",
        "crop_id": row["crop_id"],
        "patient_id": row["patient_id"],
        "final_fp_cause_label": row["user_label"],
        "user_label": row["user_label"],
        "user_confidence": row.get("user_confidence", ""),
        "user_note": row.get("user_note", ""),
        "review_priority": row.get("review_priority", ""),
        "recommended_next_action": row.get("recommended_next_action", ""),
        "rd4ad_l1_roi_mean": row.get("rd4ad_l1_roi_mean", ""),
        "rd4ad_l1_lung_channel_roi_mean": row.get("rd4ad_l1_lung_channel_roi_mean", ""),
        "rd4ad_l1_roi_patch_mean": row.get("rd4ad_l1_roi_patch_mean", ""),
        "rd4ad_l1_outside_roi_mean": row.get("rd4ad_l1_outside_roi_mean", ""),
        "roi_ratio_fixed96": row.get("roi_ratio_fixed96", ""),
        "patch_roi_ratio": row.get("patch_roi_ratio", ""),
        "patch_bbox_clipped": row.get("patch_bbox_clipped", ""),
        "png_path": row.get("original_png_path", ""),
        "contact_sheet_path": row.get("contact_sheet_path", ""),
    })

df_68_detail = pd.DataFrame(rows_68)


# ──────────────────────────────────────────────
# expanded detailed rows 합산
# ──────────────────────────────────────────────

expanded_detail = pd.concat([df_58_detail, df_68_detail], ignore_index=True)


# ──────────────────────────────────────────────
# [Guard 5] expanded row 수 / cause label count 검증
# ──────────────────────────────────────────────

abort_if(len(expanded_detail) != EXPECTED_TOTAL,
         f"expanded FP pool row 수 오류: {len(expanded_detail)} (기대 {EXPECTED_TOTAL})")

actual_counts = expanded_detail["final_fp_cause_label"].value_counts().to_dict()
for label, expected_count in EXPECTED.items():
    actual = actual_counts.get(label, 0)
    abort_if(actual != expected_count,
             f"{label} count 오류: {actual} (기대 {expected_count})")

abort_if(sum(actual_counts.values()) != EXPECTED_TOTAL,
         f"cause label count 합계 오류: {sum(actual_counts.values())} (기대 {EXPECTED_TOTAL})")


# ──────────────────────────────────────────────
# SECTION_A 확장 원인 요약표 구성
# ──────────────────────────────────────────────

INTERP = {
    "pleural_wall": "폐벽·흉막·횡격막 경계 기반 FP가 가장 큰 비중 차지",
    "large_bbox_structure": "심장·종격동·폐문부·큰 공기층 인접 구조 영향",
    "vessel_branch": "혈관 분기 구조 영향 소수 후보",
    "bronchus_air_boundary": "기관지/공기층 경계 영향 단일 후보",
}

summary_a_rows = []
for rank, label in enumerate(CAUSE_ORDER, start=1):
    cnt_58 = df_58["final_fp_cause_label"].value_counts().get(label, 0)
    cnt_68 = int(df_68_detail["final_fp_cause_label"].value_counts().get(label, 0))
    cnt_total = EXPECTED[label]
    ratio = round(cnt_total / EXPECTED_TOTAL * 100, 2)

    rep_ids = sorted(
        expanded_detail.loc[expanded_detail["final_fp_cause_label"] == label, "crop_id"]
        .head(3).tolist()
    )
    rep_str = ";".join(str(x) for x in rep_ids)

    summary_a_rows.append({
        "final_fp_cause_label": label,
        "count_phase5_58": cnt_58,
        "count_phase5_68_added": cnt_68,
        "count_expanded_total": cnt_total,
        "ratio_expanded_percent": ratio,
        "representative_crop_ids": rep_str,
        "interpretation_short": INTERP[label],
    })

df_summary_a = pd.DataFrame(summary_a_rows)


# ──────────────────────────────────────────────
# SECTION_C 분리 항목 구성
# ──────────────────────────────────────────────

separated_rows = []

for _, row in possible_lesion.iterrows():
    separated_rows.append({
        "crop_id": row["crop_id"],
        "patient_id": row["patient_id"],
        "user_label": row["user_label"],
        "recommended_next_action": row["recommended_next_action"],
        "separation_reason": "possible_lesion_overlap",
        "note": "FP cause pool에 합산 제외 — lesion contour 확인 필요",
        "source_phase": "phase5_68",
    })

for _, row in outside_roi.iterrows():
    separated_rows.append({
        "crop_id": row["crop_id"],
        "patient_id": row["patient_id"],
        "user_label": row["user_label"],
        "recommended_next_action": row["recommended_next_action"],
        "separation_reason": "outside_roi_artifact",
        "note": "FP cause pool에 합산 제외 — outside ROI 영향으로 분리",
        "source_phase": "phase5_68",
    })

df_separated = pd.DataFrame(separated_rows)


# ──────────────────────────────────────────────
# 저장 직전 Guard
# ──────────────────────────────────────────────

abort_if(OUT_ROOT.exists(), f"저장 직전 output root가 이미 존재함 (재검증): {OUT_ROOT}")


# ──────────────────────────────────────────────
# 출력 디렉토리 생성
# ──────────────────────────────────────────────

OUT_ROOT.mkdir(parents=True, exist_ok=False)


# ──────────────────────────────────────────────
# CSV 저장 (섹션 구조)
# ──────────────────────────────────────────────

with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
    f.write("# SECTION_A_EXPANDED_CAUSE_SUMMARY\n")
    df_summary_a.to_csv(f, index=False)
    f.write("\n# SECTION_B_EXPANDED_DETAILED_ROWS\n")
    expanded_detail.to_csv(f, index=False)
    f.write("\n# SECTION_C_SEPARATED_ROWS\n")
    df_separated.to_csv(f, index=False)


# ──────────────────────────────────────────────
# JSON 저장
# ──────────────────────────────────────────────

ratio_dict = {
    label: round(EXPECTED[label] / EXPECTED_TOTAL * 100, 2)
    for label in CAUSE_ORDER
}

added_crop_ids = sorted(fp_candidates["crop_id"].tolist())
excl_lesion_ids = sorted(possible_lesion["crop_id"].tolist())
excl_outside_ids = sorted(outside_roi["crop_id"].tolist())

json_summary = {
    "n_phase5_58_fp_pool": 29,
    "n_phase5_68_fp_candidates_added": 10,
    "n_expanded_fp_pool": EXPECTED_TOTAL,
    "n_by_expanded_fp_cause_label": {label: EXPECTED[label] for label in CAUSE_ORDER},
    "ratio_by_expanded_fp_cause_label": ratio_dict,
    "added_crop_ids": added_crop_ids,
    "excluded_possible_lesion_overlap_crop_ids": excl_lesion_ids,
    "excluded_outside_roi_artifact_crop_ids": excl_outside_ids,
    "notes": {
        "expanded_fp_cause_summary_only": True,
        "threshold_not_finalized": True,
        "hard_negative_not_finalized": True,
        "lesion_conclusion_forbidden": True,
        "possible_lesion_overlap_separated": True,
        "outside_roi_artifact_separated": True,
        "stage2_holdout_unused": True,
        "v2_unused": True,
        "model_forward_not_run": True,
        "score_recalculation_not_run": True,
        "original_files_unmodified": True,
    },
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(json_summary, f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# MD report 생성
# ──────────────────────────────────────────────

now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

fp_candidate_ids_list = sorted(fp_candidates["crop_id"].tolist())

md_lines = [
    "# Phase 5.69 Expanded FP Cause Summary Report",
    "",
    f"생성일시: {now_str}",
    "",
    "---",
    "",
    "## 1. Phase 5.69 목적",
    "",
    "Phase 5.58에서 visual QA를 완료한 기존 FP cause pool 29개와, "
    "Phase 5.68에서 ChatGPT 1차 visual QA를 거쳐 fp_cause_review_candidate로 분류된 "
    "신규 P1/P2 후보 10개를 합산하여 확장 FP 원인 요약을 생성한다. "
    "이 단계에서는 threshold 확정, 병변 성능 결론, hard negative 최종 채택을 하지 않는다.",
    "",
    "---",
    "",
    "## 2. 왜 29개에 신규 10개만 합산하는가",
    "",
    "Phase 5.68 대상 P1/P2 13개 중:",
    "",
    "- `fp_cause_review_candidate` 10개: FP 원인이 명확하여 기존 pool에 합산",
    "- `possible_lesion_overlap` 1개 (crop_id 7018): lesion contour가 확인되어 FP pool에 섞지 않고 분리",
    "- `outside_roi_artifact` 2개 (crop_id 2278, 6306): outside ROI 영향으로 별도 분리",
    "",
    "합산 대상은 FP 원인이 명확한 10개만이다.",
    "",
    "---",
    "",
    "## 3. 합산 제외 항목",
    "",
    "| 분리 항목 | crop_id | 이유 |",
    "|-----------|---------|------|",
    "| possible_lesion_overlap | 7018 | lesion contour 확인 — FP pool 합산 제외, caution 분리 |",
    "| outside_roi_artifact | 2278, 6306 | outside ROI 영향 — FP cause pool과 별도 분리 |",
    "",
    "---",
    "",
    "## 4. 확장 FP cause pool 39개 원인 비율표",
    "",
    "| 원인 | Phase 5.58 | Phase 5.68 추가 | 합계 | 비율 |",
    "|------|-----------|-----------------|------|------|",
]

for label in CAUSE_ORDER:
    cnt_58 = df_58["final_fp_cause_label"].value_counts().get(label, 0)
    cnt_68_add = EXPECTED[label] - cnt_58
    ratio = ratio_dict[label]
    md_lines.append(f"| {label} | {cnt_58} | {cnt_68_add} | {EXPECTED[label]} | {ratio}% |")

md_lines += [
    f"| **합계** | **29** | **10** | **{EXPECTED_TOTAL}** | **100%** |",
    "",
    "---",
    "",
    "## 5. 기존 Phase 5.58 29개 대비 변화",
    "",
    "| 원인 | 기존 29개 비율 | 확장 39개 비율 | 변화 |",
    "|------|--------------|--------------|------|",
]

old_counts = {"pleural_wall": 18, "large_bbox_structure": 8, "vessel_branch": 2, "bronchus_air_boundary": 1}
for label in CAUSE_ORDER:
    old_ratio = round(old_counts[label] / 29 * 100, 1)
    new_ratio = ratio_dict[label]
    direction = "→" if abs(new_ratio - old_ratio) < 0.5 else ("↑" if new_ratio > old_ratio else "↓")
    md_lines.append(f"| {label} | {old_ratio}% | {new_ratio}% | {direction} |")

md_lines += [
    "",
    "---",
    "",
    "## 6. 원인 해석",
    "",
    "**pleural_wall (23개, 59.0%)** — 폐벽·흉막·횡격막 경계 기반 FP가 여전히 1순위를 차지한다. "
    "신규 10개 중 5개가 pleural_wall로 분류되어 기존 경향이 유지되었다.",
    "",
    "**large_bbox_structure (12개, 30.8%)** — 심장·종격동·폐문부 인접 큰 구조물이 2순위이다. "
    "신규 10개 중 4개가 추가되어 비율이 27.6%에서 30.8%로 소폭 증가했다.",
    "",
    "**vessel_branch (3개, 7.7%)** — 혈관 분기 구조 영향은 여전히 소수이다. "
    "신규 1개가 추가되어 6.9%에서 7.7%로 소폭 증가했다.",
    "",
    "**bronchus_air_boundary (1개, 2.6%)** — 기관지/공기층 경계 영향은 단일 후보로 유지된다. "
    "신규 추가 없음.",
    "",
    "---",
    "",
    "## 7. 보고용 문장 초안",
    "",
    "**짧은 문장 1:**",
    "QA 후보 내 FP 원인 분포 분석 결과, 확장 pool 39개 중 폐벽·흉막·횡격막 경계 기반 구조가 59.0%로 1순위를 차지하였다.",
    "",
    "**짧은 문장 2:**",
    "심장·종격동·폐문부 인접 큰 구조물이 30.8%로 2순위이며, 혈관 분기는 7.7%로 소수 원인에 해당한다.",
    "",
    "**자세한 문장:**",
    "Phase 5.58 기존 FP cause pool 29개에 Phase 5.68 신규 P1/P2 fp_cause_review_candidate 10개를 합산한 "
    "확장 FP cause pool 39개를 분석한 결과, QA 후보 내 FP 원인 분포는 "
    "폐벽·흉막·횡격막 경계(pleural_wall) 59.0%, "
    "심장·종격동·폐문부 인접 큰 구조물(large_bbox_structure) 30.8%, "
    "혈관 분기(vessel_branch) 7.7%, "
    "기관지/공기층 경계(bronchus_air_boundary) 2.6% 순으로 나타났다. "
    "이 수치는 QA 후보 내 FP 원인 분포를 나타내며, threshold 확정이나 병변 성능 결론과 무관하다.",
    "",
    "---",
    "",
    "## 8. 해석 제한",
    "",
    "- 이 요약은 QA 후보 내 FP 원인 분포를 나타낸다.",
    "- threshold 확정이 아니다.",
    "- hard negative 최종 채택이 아니다.",
    "- 병변 성능 결론이 아니다.",
    "- stage2_holdout 검증이 아니다.",
    "",
    "---",
    "",
    "## 9. 다음 단계 제안",
    "",
    "1. **Phase 5.70 handoff 문서 최신화**: 확장 FP cause pool 39개 분포를 handoff 문서에 반영할지 검토.",
    "2. **P3 46개 확장 검토 여부**: P3 visual review pack 생성은 별도 승인 후 결정.",
    "3. **stage2_holdout 계속 봉인**: threshold 확정 및 hard negative 최종 채택 이후에만 접근 검토.",
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
print("-" * 60)
print(f"Phase 5.58 FP pool row 수  : {len(df_58)}")
print(f"Phase 5.68 row 수          : {len(df_68)}")
print(f"신규 추가 FP candidate crop_id: {added_crop_ids}")
print(f"합산 제외 possible_lesion_overlap: {excl_lesion_ids}")
print(f"합산 제외 outside_roi_artifact: {excl_outside_ids}")
print(f"expanded FP pool row 수    : {len(expanded_detail)}")
print("-" * 60)
print("expanded cause label별 count:")
for label in CAUSE_ORDER:
    print(f"  {label}: {EXPECTED[label]}")
print("expanded cause label별 ratio:")
for label in CAUSE_ORDER:
    print(f"  {label}: {ratio_dict[label]}%")
print("-" * 60)
print("기존 29개 대비 변화:")
for label in CAUSE_ORDER:
    old_r = round(old_counts[label] / 29 * 100, 1)
    new_r = ratio_dict[label]
    print(f"  {label}: {old_r}% → {new_r}%")
print("-" * 60)
print("[보고용 짧은 문장]")
print("1. QA 후보 내 FP 원인 분포 분석 결과, 확장 pool 39개 중 폐벽·흉막·횡격막 경계 기반 구조가 59.0%로 1순위를 차지하였다.")
print("2. 심장·종격동·폐문부 인접 큰 구조물이 30.8%로 2순위이며, 혈관 분기는 7.7%로 소수 원인에 해당한다.")
print("-" * 60)
print("기존 파일 미수정         : True")
print("model forward 없음       : True")
print("score 재계산 없음        : True")
print("threshold 확정 없음      : True")
print("병변 성능 결론 없음      : True")
print("stage2_holdout/v2 미접근 : True")
print("=" * 60)
print()
print("[다음 단계 제안]")
print("1. Phase 5.70 handoff 문서 최신화 검토")
print("2. P3 46개 확장 검토 여부는 별도 승인 후 결정")
print("3. stage2_holdout 계속 봉인")
