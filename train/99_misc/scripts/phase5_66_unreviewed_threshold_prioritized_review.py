"""
Phase 5.66 unreviewed_qa_candidate threshold-prioritized review plan 생성 스크립트.

- Phase 5.65 read-only application CSV 기반
- unreviewed_qa_candidate 110개에 P1~P5 우선순위 부여
- 기존 파일 수정/삭제/덮어쓰기 금지
- model forward / score 재계산 / stage2_holdout 접근 금지
"""

import json
import pathlib
import datetime
import pandas as pd


# ──────────────────────────────────────────────
# abort helper (python -O 에서도 동작)
# ──────────────────────────────────────────────

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

IN_CSV = (
    BASE
    / "phase5_65_roi_threshold_readonly_application_v1"
    / "phase5_65_roi_threshold_readonly_application_v1.csv"
)
IN_JSON = (
    BASE
    / "phase5_65_roi_threshold_readonly_application_v1"
    / "phase5_65_roi_threshold_readonly_application_summary_v1.json"
)
IN_MANIFEST = pathlib.Path(
    "/home/jinhy/project/lung-ct-anomaly/outputs"
    "/second-stage-lesion-refiner-v1"
    "/visualizations/hard_negative_top_score_context_overlay_qa_v1"
    "/manifest_png_index_context_overlay_v1.csv"
)
IN_FP_CAUSE = (
    BASE
    / "phase5_58_final_fp_cause_table_v1"
    / "phase5_58_final_fp_cause_table_v1.csv"
)

OUT_ROOT = (
    BASE / "phase5_66_unreviewed_threshold_prioritized_review_v1"
)
OUT_CSV = OUT_ROOT / "phase5_66_unreviewed_threshold_prioritized_review_v1.csv"
OUT_JSON = OUT_ROOT / "phase5_66_unreviewed_threshold_prioritized_review_summary_v1.json"
OUT_MD = OUT_ROOT / "phase5_66_unreviewed_threshold_prioritized_review_report_v1.md"


# ──────────────────────────────────────────────
# [Guard 1] output root 기존 존재 시 즉시 중단
# ──────────────────────────────────────────────

abort_if(OUT_ROOT.exists(), f"output root가 이미 존재함 (중단): {OUT_ROOT}")


# ──────────────────────────────────────────────
# [Guard 2] 입력 파일 존재 여부
# ──────────────────────────────────────────────

for p in [IN_CSV, IN_JSON, IN_MANIFEST, IN_FP_CAUSE]:
    abort_if(not p.exists(), f"입력 파일 없음: {p}")


# ──────────────────────────────────────────────
# 입력 파일 로드
# ──────────────────────────────────────────────

df = pd.read_csv(IN_CSV)
manifest = pd.read_csv(IN_MANIFEST)


# ──────────────────────────────────────────────
# [Guard 3] 입력 row 수 / 컬럼 / 그룹 수 검증
# ──────────────────────────────────────────────

abort_if(len(df) != 150, f"입력 row 수 오류: {len(df)} (기대 150)")
abort_if(
    "threshold_application_group" not in df.columns,
    "threshold_application_group 컬럼 없음",
)

n_unreviewed = (df["threshold_application_group"] == "unreviewed_qa_candidate").sum()
abort_if(n_unreviewed != 110, f"unreviewed_qa_candidate 수 오류: {n_unreviewed} (기대 110)")

n_fp_pool = (df["threshold_application_group"] == "strong_fp_cause_candidate").sum()
abort_if(n_fp_pool != 29, f"strong_fp_cause_candidate 수 오류: {n_fp_pool} (기대 29)")

n_lesion_excl = (df["threshold_application_group"] == "lesion_overlap_excluded").sum()
abort_if(n_lesion_excl != 9, f"lesion_overlap_excluded 수 오류: {n_lesion_excl} (기대 9)")

n_outside = (df["threshold_application_group"] == "outside_roi_artifact_separated").sum()
abort_if(n_outside != 2, f"outside_roi_artifact_separated 수 오류: {n_outside} (기대 2)")


# ──────────────────────────────────────────────
# unreviewed 110개 필터링
# ──────────────────────────────────────────────

unrev = df[df["threshold_application_group"] == "unreviewed_qa_candidate"].copy()


# ──────────────────────────────────────────────
# 우선순위 부여
# ──────────────────────────────────────────────

def assign_priority(row):
    if row["pass_both_p99"]:
        return "P1_unreviewed_pass_both_p99"
    elif row["pass_any_p99"] and not row["pass_both_p99"]:
        return "P2_unreviewed_pass_any_p99"
    elif row["pass_both_p95"] and not row["pass_any_p99"]:
        return "P3_unreviewed_pass_both_p95_only"
    elif row["pass_any_p95"] and not row["pass_both_p95"] and not row["pass_any_p99"]:
        return "P4_unreviewed_pass_any_p95_only"
    else:
        return "P5_unreviewed_below_p95"


unrev["review_priority"] = unrev.apply(assign_priority, axis=1)


# ──────────────────────────────────────────────
# suggested_review_action
# ──────────────────────────────────────────────

def assign_action(priority):
    if priority in ("P1_unreviewed_pass_both_p99", "P2_unreviewed_pass_any_p99"):
        return "review_now"
    elif priority == "P3_unreviewed_pass_both_p95_only":
        return "review_if_time"
    else:
        return "low_priority"


unrev["suggested_review_action"] = unrev["review_priority"].map(assign_action)


# ──────────────────────────────────────────────
# [Guard 4] manifest 필수 컬럼 검증
# ──────────────────────────────────────────────

for col in ["crop_id", "png_path", "contact_sheet_path"]:
    abort_if(col not in manifest.columns, f"manifest에 필수 컬럼 없음: {col}")

n_manifest_before_dedup = len(manifest)
manifest_subset = manifest[["crop_id", "png_path", "contact_sheet_path"]].drop_duplicates("crop_id")
n_manifest_after_dedup = len(manifest_subset)
print(f"manifest crop_id dedup: {n_manifest_before_dedup} → {n_manifest_after_dedup}")


# ──────────────────────────────────────────────
# [Guard 5] png_path suffix 방지: unrev에 기존 컬럼 있으면 제거 후 merge
# ──────────────────────────────────────────────

for col in ["png_path", "contact_sheet_path"]:
    if col in unrev.columns:
        unrev = unrev.drop(columns=[col])


# ──────────────────────────────────────────────
# manifest join (png_path, contact_sheet_path)
# ──────────────────────────────────────────────

unrev = unrev.merge(manifest_subset, on="crop_id", how="left")


# ──────────────────────────────────────────────
# [Guard 6] join 후 row 수 / 누락 검증
# ──────────────────────────────────────────────

abort_if(len(unrev) != 110, f"manifest join 후 row 수 오류: {len(unrev)} (기대 110)")

n_png_null = unrev["png_path"].isna().sum()
abort_if(n_png_null != 0, f"png_path 누락 {n_png_null}건 (기대 0)")

n_cs_null = unrev["contact_sheet_path"].isna().sum()
abort_if(n_cs_null != 0, f"contact_sheet_path 누락 {n_cs_null}건 (기대 0)")


# ──────────────────────────────────────────────
# [Guard 7] 실제 파일 존재 110/110 확인
# ──────────────────────────────────────────────

missing_png = [p for p in unrev["png_path"].tolist() if not pathlib.Path(p).exists()]
abort_if(
    len(missing_png) != 0,
    f"png_path 실제 파일 없음 {len(missing_png)}건: {missing_png[:5]}",
)

missing_cs = [p for p in unrev["contact_sheet_path"].tolist() if not pathlib.Path(p).exists()]
abort_if(
    len(missing_cs) != 0,
    f"contact_sheet_path 실제 파일 없음 {len(missing_cs)}건: {missing_cs[:5]}",
)


# ──────────────────────────────────────────────
# 빈 레이블 컬럼 추가
# ──────────────────────────────────────────────

for col in ["user_label", "user_confidence", "user_note", "reviewer", "reviewed_at"]:
    unrev[col] = ""


# ──────────────────────────────────────────────
# 출력 컬럼 순서 정렬
# ──────────────────────────────────────────────

OUTPUT_COLS = [
    "review_priority", "crop_id", "patient_id", "qa_group", "qa_priority",
    "lesion_overlap_class", "threshold_application_group",
    "pass_roi_mean_p95", "pass_lung_roi_mean_p95", "pass_any_p95", "pass_both_p95",
    "pass_roi_mean_p99", "pass_lung_roi_mean_p99", "pass_any_p99", "pass_both_p99",
    "rd4ad_l1_roi_mean", "rd4ad_l1_lung_channel_roi_mean", "rd4ad_l1_roi_patch_mean",
    "rd4ad_l1_outside_roi_mean", "roi_ratio_fixed96", "patch_roi_ratio", "patch_bbox_clipped",
    "png_path", "contact_sheet_path",
    "suggested_review_action",
    "user_label", "user_confidence", "user_note", "reviewer", "reviewed_at",
]

PNG_JOIN_COLS = {"png_path", "contact_sheet_path"}
for col in OUTPUT_COLS:
    if col not in unrev.columns:
        if col in PNG_JOIN_COLS:
            raise RuntimeError(f"[ABORT] join 후에도 {col} 컬럼 없음 — manifest join 실패")
        unrev[col] = ""

out_df = unrev[OUTPUT_COLS].copy()


# ──────────────────────────────────────────────
# [Guard 8] P1~P5 합계 검증
# ──────────────────────────────────────────────

p_labels = [
    "P1_unreviewed_pass_both_p99",
    "P2_unreviewed_pass_any_p99",
    "P3_unreviewed_pass_both_p95_only",
    "P4_unreviewed_pass_any_p95_only",
    "P5_unreviewed_below_p95",
]
priority_counts = out_df["review_priority"].value_counts().to_dict()
total_p = sum(priority_counts.get(p, 0) for p in p_labels)
abort_if(total_p != 110, f"P1~P5 합계 오류: {total_p} (기대 110)")

n_by_priority = {p: priority_counts.get(p, 0) for p in p_labels}
crop_ids_by_priority = {
    p: sorted(out_df.loc[out_df["review_priority"] == p, "crop_id"].tolist())
    for p in p_labels
}


# ──────────────────────────────────────────────
# [Guard 9] final output 검증
# ──────────────────────────────────────────────

abort_if(len(out_df) != 110, f"out_df row 수 오류: {len(out_df)} (기대 110)")
abort_if(out_df["review_priority"].isna().any(), "review_priority 누락 존재")
abort_if(out_df["suggested_review_action"].isna().any(), "suggested_review_action 누락 존재")
abort_if(out_df["png_path"].isna().any(), "out_df png_path 누락 존재")
abort_if(out_df["contact_sheet_path"].isna().any(), "out_df contact_sheet_path 누락 존재")


# ──────────────────────────────────────────────
# [Guard 10] 저장 직전 출력 파일 3개 재검증
# ──────────────────────────────────────────────

abort_if(OUT_ROOT.exists(), f"저장 직전 output root가 이미 존재함 (재검증): {OUT_ROOT}")
for out_path in [OUT_CSV, OUT_JSON, OUT_MD]:
    abort_if(out_path.exists(), f"출력 파일이 이미 존재함 (덮어쓰기 금지): {out_path}")


# ──────────────────────────────────────────────
# 출력 디렉토리 생성 (모든 검증 통과 후)
# ──────────────────────────────────────────────

OUT_ROOT.mkdir(parents=True, exist_ok=False)

# mkdir 직후 재검증
for out_path in [OUT_CSV, OUT_JSON, OUT_MD]:
    abort_if(out_path.exists(), f"mkdir 직후 출력 파일이 이미 존재함: {out_path}")


# ──────────────────────────────────────────────
# CSV 저장
# ──────────────────────────────────────────────

out_df.to_csv(OUT_CSV, index=False)


# ──────────────────────────────────────────────
# JSON 저장
# ──────────────────────────────────────────────

summary = {
    "n_total_qa": 150,
    "n_unreviewed": 110,
    "n_by_review_priority": n_by_priority,
    "crop_ids_by_review_priority": crop_ids_by_priority,
    "n_reviewed_fp_pool": 29,
    "n_excluded_lesion_overlap": 9,
    "n_outside_roi_artifact_separate": 2,
    "notes": {
        "threshold_not_finalized": True,
        "readonly_prioritization_only": True,
        "unreviewed_candidates_only": True,
        "reviewed_fp_pool_not_repeated": True,
        "hard_negative_not_finalized": True,
        "lesion_exclude_kept_separate": True,
        "outside_roi_artifact_kept_separate": True,
        "stage2_holdout_unused": True,
        "v2_unused": True,
        "model_forward_not_run": True,
        "score_recalculation_not_run": True,
        "original_files_unmodified": True,
    },
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# MD report 생성
# ──────────────────────────────────────────────

p1_ids = crop_ids_by_priority["P1_unreviewed_pass_both_p99"]
p2_ids = crop_ids_by_priority["P2_unreviewed_pass_any_p99"]
p3_ids = crop_ids_by_priority["P3_unreviewed_pass_both_p95_only"]
p4_ids = crop_ids_by_priority["P4_unreviewed_pass_any_p95_only"]
p5_ids = crop_ids_by_priority["P5_unreviewed_below_p95"]

n_p1 = n_by_priority["P1_unreviewed_pass_both_p99"]
n_p2 = n_by_priority["P2_unreviewed_pass_any_p99"]
n_p3 = n_by_priority["P3_unreviewed_pass_both_p95_only"]
n_p4 = n_by_priority["P4_unreviewed_pass_any_p95_only"]
n_p5 = n_by_priority["P5_unreviewed_below_p95"]

now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

md_lines = [
    "# Phase 5.66 Unreviewed QA Candidate Threshold-Prioritized Review Plan",
    "",
    f"생성일시: {now_str}",
    "",
    "---",
    "",
    "## 1. Phase 5.66 목적",
    "",
    "Phase 5.65에서 threshold를 read-only로 적용한 결과, 전체 QA 150개 중 아직 visual review를 완료하지 않은 "
    "unreviewed_qa_candidate 110개가 남아 있다. Phase 5.66은 이 110개에 대해 p99 기반 우선순위를 부여하고, "
    "reviewer가 어떤 후보부터 visual review를 진행해야 하는지 review 순서를 명시하는 것이 목적이다.",
    "",
    "이 단계에서는 threshold 확정, 병변 성능 결론, hard negative 최종 채택을 하지 않는다.",
    "",
    "---",
    "",
    "## 2. 왜 29개 visual pack 반복이 아니라 unreviewed 110개 분석이 필요한가",
    "",
    "Phase 5.58에서 fp_cause_review_pool 29개에 대해 visual QA와 원인 라벨링이 완료되었다. "
    "이 29개는 strong_fp_cause_candidate로 분류되어 있으며 threshold 기반으로 이미 pass_both_p95 통과 확인이 완료된 상태다.",
    "",
    "반면 unreviewed_qa_candidate 110개는 threshold 통과 여부는 계산됐으나 "
    "어떤 후보를 먼저 봐야 하는지 우선순위가 없다. "
    "p95 기준 통과율이 71.33%로 높기 때문에, 모든 110개를 동일 우선순위로 review하면 "
    "신뢰도가 낮은 후보(p95 only)에 review 자원이 낭비될 수 있다. "
    "p99 기반 고신뢰 후보를 먼저 식별하여 review 효율을 높이는 것이 이 단계의 이유다.",
    "",
    "---",
    "",
    "## 3. Phase 5.65 결과 요약",
    "",
    "| 항목 | 값 |",
    "|------|-----|",
    "| 전체 QA | 150개 |",
    "| pass_roi_mean_p95 | 107개 (71.33%) |",
    "| pass_lung_roi_mean_p95 | 95개 (63.33%) |",
    "| pass_any_p95 | 107개 (71.33%) |",
    "| pass_both_p95 | 95개 (63.33%) |",
    "| pass_any_p99 | 40개 (26.67%) |",
    "| pass_both_p99 | 21개 (14.00%) |",
    "| fp_cause_review_pool (strong_fp_cause_candidate) | 29개 (visual QA 완료) |",
    "| unreviewed_qa_candidate | 110개 (visual review 미완료) |",
    "| lesion_overlap_excluded | 9개 (제외 유지) |",
    "| outside_roi_artifact_separated | 2개 (제외 유지) |",
    "| threshold 확정 | 미확정 |",
    "| 병변 성능 결론 | 없음 |",
    "",
    "---",
    "",
    "## 4. Unreviewed 110개 우선순위표",
    "",
    "| 우선순위 | 조건 | count | crop_id 목록 |",
    "|----------|------|-------|--------------|",
    f"| P1 (pass_both_p99) | pass_both_p99=True | {n_p1} | {p1_ids} |",
    f"| P2 (pass_any_p99) | pass_any_p99=True, pass_both_p99=False | {n_p2} | {p2_ids} |",
    f"| P3 (pass_both_p95_only) | pass_both_p95=True, pass_any_p99=False | {n_p3} | {p3_ids} |",
    f"| P4 (pass_any_p95_only) | pass_any_p95=True, pass_both_p95=False, pass_any_p99=False | {n_p4} | {p4_ids} |",
    f"| P5 (below_p95) | pass_any_p95=False | {n_p5} | {p5_ids} |",
    "| **합계** | | **110** | |",
    "",
    "---",
    "",
    "## 5. P1~P5 설명",
    "",
    "**P1 — pass_both_p99 (고신뢰, 즉시 review)**",
    "두 score 채널(roi_mean, lung_channel_roi_mean) 모두 p99 threshold를 초과한 후보.",
    "가장 anomaly score가 높고, 두 측정 기준이 일치하므로 신뢰도가 가장 높다.",
    "",
    "**P2 — pass_any_p99 (고신뢰, 즉시 review)**",
    "두 채널 중 하나만 p99 초과. P1보다 신뢰도가 약간 낮지만 여전히 고신뢰 후보.",
    "",
    "**P3 — pass_both_p95_only (중간 신뢰, 시간이 있을 때 review)**",
    "두 채널 모두 p95 초과하나 p99 미달. p95는 기준이 넓어 FP 가능성이 P1/P2보다 높다.",
    "",
    "**P4 — pass_any_p95_only (낮은 신뢰, 후순위)**",
    "한 채널만 p95 초과. 두 기준이 불일치하고 p99 미달이므로 후순위 review 대상.",
    "",
    "**P5 — below_p95 (최저 신뢰, 후순위)**",
    "두 채널 모두 p95 미달. 현재 threshold 기준에서 통과하지 못한 후보.",
    "",
    "---",
    "",
    "## 6. p95가 넓은 기준이라는 해석",
    "",
    "pass_any_p95 기준으로 전체 QA의 71.33%인 107개가 통과한다. "
    "이는 p95 threshold가 상당히 낮게 설정되어 있음을 의미하며, "
    "FP 가능성이 높은 후보도 통과 범위에 포함될 수 있다. "
    "특히 pass_any_p95만 만족하고 pass_both_p95를 만족하지 못하는 P4 구간은 "
    "두 채널 간 불일치가 존재하므로 해석에 주의가 필요하다.",
    "",
    "---",
    "",
    "## 7. p99 후보를 먼저 보는 이유",
    "",
    "p99는 전체 정상 score 분포의 상위 1%에 해당하는 임계값으로, "
    "p95보다 훨씬 엄격한 기준이다. pass_both_p99는 전체 QA에서 14.00%(21개)만 통과하므로 "
    "review 자원을 집중하기에 적합하다. "
    "P1/P2(pass_any_p99) 후보를 먼저 review하면 "
    "가장 신뢰도 높은 이상 후보를 우선적으로 확인할 수 있고, "
    "threshold 확정 전 사전 검토로서의 의미도 가진다.",
    "",
    "---",
    "",
    "## 8. 다음 단계 제안",
    "",
    "1. **P1/P2 중심 visual review pack 생성 여부 검토**",
    "   - P1 + P2 = " + f"{n_p1 + n_p2}개에 대해 context overlay PNG contact sheet 생성 여부를 결정한다.",
    "   - 필요 시 P3까지 포함하여 확장한다.",
    "",
    "2. **P4/P5는 후순위**",
    "   - P4/P5는 threshold를 낮추지 않는 한 review 우선순위가 낮다.",
    "   - 별도 승인 없이 P4/P5 visual pack을 자동 생성하지 않는다.",
    "",
    "3. **병변/holdout 적용은 별도 승인 전 금지**",
    "   - 이 단계의 결과를 병변 성능 평가나 holdout 적용에 직접 사용하지 않는다.",
    "",
    "4. **stage2_holdout 계속 봉인**",
    "   - stage2_holdout/v2 경로 접근 금지 상태를 유지한다.",
    "   - threshold 확정 및 hard negative 최종 채택 이후에만 접근을 검토한다.",
    "",
    "---",
    "",
    "## 검증 결과",
    "",
    "| 항목 | 결과 |",
    "|------|------|",
    "| input QA row 수 | 150 ✓ |",
    "| unreviewed_qa_candidate | 110 ✓ |",
    f"| P1~P5 합계 | {total_p} ✓ |",
    "| strong_fp_cause_candidate 29개 제외 | ✓ |",
    "| lesion_overlap_excluded 9개 제외 | ✓ |",
    "| outside_roi_artifact_separated 2개 제외 | ✓ |",
    "| png_path 누락 0건 | ✓ |",
    "| contact_sheet_path 누락 0건 | ✓ |",
    "| png_path 실제 파일 존재 110/110 | ✓ |",
    "| contact_sheet_path 실제 파일 존재 110/110 | ✓ |",
    "| 기존 파일 미수정 | ✓ |",
    "| model forward 없음 | ✓ |",
    "| score 재계산 없음 | ✓ |",
    "| threshold 확정 없음 | ✓ |",
    "| 병변 성능 결론 없음 | ✓ |",
    "| stage2_holdout/v2 미접근 | ✓ |",
]

with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines) + "\n")


# ──────────────────────────────────────────────
# 최종 보고 출력
# ──────────────────────────────────────────────

print("=" * 60)
print("검토 판정: 전체 통과")
print("=" * 60)
print(f"output root         : {OUT_ROOT}")
print(f"생성 CSV            : {OUT_CSV}")
print(f"생성 JSON           : {OUT_JSON}")
print(f"생성 MD             : {OUT_MD}")
print("-" * 60)
print(f"input QA row 수     : {len(df)}")
print(f"unreviewed 수       : {n_unreviewed}")
print("-" * 60)
print("P1~P5 count:")
for p in p_labels:
    print(f"  {p}: {n_by_priority[p]}")
print("-" * 60)
print(f"P1 crop_id 목록: {p1_ids}")
print(f"P2 crop_id 목록: {p2_ids}")
print(f"P3 crop_id 목록: {p3_ids}")
print(f"P4 crop_id 목록: {p4_ids}")
print(f"P5 crop_id 목록: {p5_ids}")
print("-" * 60)
print("기존 파일 미수정         : True")
print("model forward 없음       : True")
print("score 재계산 없음        : True")
print("threshold 확정 없음      : True")
print("병변 성능 결론 없음      : True")
print("stage2_holdout/v2 미접근 : True")
print("=" * 60)
