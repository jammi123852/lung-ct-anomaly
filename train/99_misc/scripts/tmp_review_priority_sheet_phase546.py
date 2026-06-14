"""
Phase 5.46: Lesion Overlap 기반 Manual Review Priority Sheet 생성

입력:
  1. lesion overlap CSV (150행)
  2. context overlay PNG index (150행) - score 컬럼 포함
  3. (lesion_overlap_ratio_* / roi_ratio_patch,fixed96 컬럼은 lesion overlap CSV에 없음 -> 빈 문자열)

출력:
  1. review CSV: hard_negative_top_score_overlap_prioritized_review_sheet_v1.csv
  2. summary JSON: hard_negative_top_score_overlap_prioritized_review_summary_v1.json
  3. (guide MD는 별도 작성)

절대 금지:
  - PNG 재생성 / score 재계산 / training / model forward / checkpoint
  - threshold 확정 / 병변 성능 결론
  - stage2_holdout / v2 접근
  - volume source 접근
"""

import os
import sys
import json
import pandas as pd

# ──────────────────────────────────────────────
# 경로 정의
# ──────────────────────────────────────────────
PROJECT_ROOT = "/home/jinhy/project/lung-ct-anomaly"

LESION_OVERLAP_CSV = os.path.join(
    PROJECT_ROOT,
    "outputs/second-stage-lesion-refiner-v1/evaluation/"
    "rd4ad_2p5d_normal_mw_fixed96_v1/hard_negative_top_score_qa_v1/"
    "hard_negative_top_score_lesion_overlap_analysis_v1.csv"
)

CONTEXT_OVERLAY_INDEX_CSV = os.path.join(
    PROJECT_ROOT,
    "outputs/second-stage-lesion-refiner-v1/visualizations/"
    "hard_negative_top_score_context_overlay_qa_v1/"
    "manifest_png_index_context_overlay_v1.csv"
)

OUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs/second-stage-lesion-refiner-v1/review_annotations/"
    "hard_negative_top_score_qa_v1"
)

OUT_REVIEW_CSV = os.path.join(
    OUT_DIR,
    "hard_negative_top_score_overlap_prioritized_review_sheet_v1.csv"
)

OUT_SUMMARY_JSON = os.path.join(
    OUT_DIR,
    "hard_negative_top_score_overlap_prioritized_review_summary_v1.json"
)

# ──────────────────────────────────────────────
# STEP 1: 로드
# ──────────────────────────────────────────────
print("[1] lesion overlap CSV 로드")
df_overlap = pd.read_csv(LESION_OVERLAP_CSV)
print(f"    rows: {len(df_overlap)}")

print("[2] context overlay PNG index 로드")
df_png = pd.read_csv(CONTEXT_OVERLAY_INDEX_CSV)
print(f"    rows: {len(df_png)}")

# ──────────────────────────────────────────────
# STEP 2: crop_id 기준 join
# ──────────────────────────────────────────────
print("[3] crop_id 기준 join (score 컬럼: context overlay index에서 가져옴)")

# PNG index에서 가져올 컬럼 (score 관련 + png_path, contact_sheet_path)
PNG_COLS_TO_JOIN = [
    "crop_id",
    "png_path",
    "contact_sheet_path",
    "crop_score_l1_mean",
    "padim_score_mean",
    "padim_score_max",
    "threshold_exceed_val_p90",
    "threshold_exceed_val_p95",
    "threshold_exceed_val_p99",
]
df_png_sub = df_png[PNG_COLS_TO_JOIN].copy()

df = df_overlap.merge(df_png_sub, on="crop_id", how="left")

if len(df) != 150:
    print(f"[ERROR] join 후 row 수 {len(df)} != 150. 중복 crop_id 확인 필요.")
    sys.exit(1)
print(f"    join 후 rows: {len(df)} (기대: 150) -> OK")

# ──────────────────────────────────────────────
# STEP 3: 없는 컬럼 빈 문자열로 채우기
# ──────────────────────────────────────────────
# lesion_overlap_ratio_patch/fixed96/context192, roi_ratio_patch/fixed96 없음
for col in ["lesion_overlap_ratio_patch", "lesion_overlap_ratio_fixed96",
            "lesion_overlap_ratio_context192",
            "roi_ratio_patch", "roi_ratio_fixed96"]:
    if col not in df.columns:
        df[col] = ""

# roi_ratio_context192는 lesion overlap CSV에 있음 (30번째 컬럼)
# 이미 join 후 df에 있음

# ──────────────────────────────────────────────
# STEP 4: review_priority 분류
# ──────────────────────────────────────────────
print("[4] review_priority 분류")


def assign_review_priority(row):
    oc = row["lesion_overlap_class"]
    qg = str(row["qa_group"]) if pd.notna(row["qa_group"]) else ""

    if oc == "patch_overlap":
        return "P1_patch_overlap"
    elif oc == "context192_overlap_only":
        return "P2_context_overlap_only"
    elif oc == "no_lesion_overlap":
        # P3: HN-p95 계열 또는 top10-crops 계열
        is_p95 = "HN-p95" in qg
        is_top10crops = "HN-top10-crops" in qg
        if is_p95 or is_top10crops:
            return "P3_high_score_no_overlap"
        # P4: HN-large-bbox 계열 (p95 없이)
        is_large_bbox = "HN-large-bbox" in qg
        if is_large_bbox:
            return "P4_large_bbox_no_overlap"
        # P5: 나머지
        return "P5_other_no_overlap"
    else:
        # 알 수 없는 class -> P5
        return "P5_other_no_overlap"


df["review_priority"] = df.apply(assign_review_priority, axis=1)

priority_counts = df["review_priority"].value_counts().to_dict()
print(f"    review_priority 분포:")
for p in ["P1_patch_overlap", "P2_context_overlap_only",
          "P3_high_score_no_overlap", "P4_large_bbox_no_overlap",
          "P5_other_no_overlap"]:
    print(f"      {p}: {priority_counts.get(p, 0)}")

# ──────────────────────────────────────────────
# STEP 5: suggested_action 컬럼 추가
# ──────────────────────────────────────────────
print("[5] suggested_action 생성")


def assign_suggested_action(row):
    oc = row["lesion_overlap_class"]
    qg = str(row["qa_group"]) if pd.notna(row["qa_group"]) else ""
    rp = row["review_priority"]

    if oc == "patch_overlap":
        return "병변과 직접 겹침. hard negative/false positive label 금지. 병변 후보 또는 병변 포함 후보로 확인"
    elif oc == "context192_overlap_only":
        return "병변 주변 context에만 포함. 후보 patch와 병변 위치 관계 확인"
    elif oc == "no_lesion_overlap":
        if "HN-padim-high-rd4ad-low" in qg:
            return "1차 PaDiM high, 2차 low. 병변 무관 false positive 제거 후보 확인"
        elif "HN-large-bbox" in qg:
            return "큰 구조물 false positive 원인 확인"
        else:
            return "혈관/폐벽/기관지/공기층 경계 원인 분류"
    else:
        return "혈관/폐벽/기관지/공기층 경계 원인 분류"


df["suggested_action"] = df.apply(assign_suggested_action, axis=1)

# ──────────────────────────────────────────────
# STEP 6: 정렬
# ──────────────────────────────────────────────
print("[6] 정렬: review_priority -> lesion_pixels_patch 내림차순 -> crop_score_l1_mean 내림차순 -> patient_id -> crop_id")

PRIORITY_ORDER = {
    "P1_patch_overlap": 1,
    "P2_context_overlap_only": 2,
    "P3_high_score_no_overlap": 3,
    "P4_large_bbox_no_overlap": 4,
    "P5_other_no_overlap": 5,
}
df["_priority_rank"] = df["review_priority"].map(PRIORITY_ORDER)

# crop_score_l1_mean이 없을 경우 대비 (join 실패 시)
if "crop_score_l1_mean" not in df.columns:
    df["crop_score_l1_mean"] = float("nan")

df_sorted = df.sort_values(
    by=["_priority_rank", "lesion_pixels_patch", "crop_score_l1_mean", "patient_id", "crop_id"],
    ascending=[True, False, False, True, True]
).reset_index(drop=True)

df_sorted.drop(columns=["_priority_rank"], inplace=True)

# ──────────────────────────────────────────────
# STEP 7: review_order 컬럼 추가 (1~150)
# ──────────────────────────────────────────────
df_sorted["review_order"] = range(1, len(df_sorted) + 1)

# ──────────────────────────────────────────────
# STEP 8: manual 컬럼 추가 (빈 문자열)
# ──────────────────────────────────────────────
for col in ["manual_label", "manual_confidence", "manual_note", "reviewer", "reviewed_at"]:
    df_sorted[col] = ""

# ──────────────────────────────────────────────
# STEP 9: 출력 컬럼 순서 정의
# ──────────────────────────────────────────────
OUTPUT_COLS = [
    "review_order",
    "review_priority",
    "lesion_overlap_class",
    "patient_id",
    "crop_id",
    "qa_group",
    "qa_priority",
    "crop_score_l1_mean",
    "padim_score_mean",
    "padim_score_max",
    "threshold_exceed_val_p90",
    "threshold_exceed_val_p95",
    "threshold_exceed_val_p99",
    "lesion_pixels_patch",
    "lesion_pixels_fixed96",
    "lesion_pixels_context192",
    "lesion_overlap_ratio_patch",
    "lesion_overlap_ratio_fixed96",
    "lesion_overlap_ratio_context192",
    "roi_ratio_patch",
    "roi_ratio_fixed96",
    "roi_ratio_context192",
    "png_path",
    "contact_sheet_path",
    "suggested_action",
    "manual_label",
    "manual_confidence",
    "manual_note",
    "reviewer",
    "reviewed_at",
]

# 없는 컬럼은 빈 문자열로 채움
for col in OUTPUT_COLS:
    if col not in df_sorted.columns:
        df_sorted[col] = ""

df_out = df_sorted[OUTPUT_COLS].copy()

# ──────────────────────────────────────────────
# STEP 10: 출력 폴더 생성 및 CSV 저장
# ──────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
df_out.to_csv(OUT_REVIEW_CSV, index=False)
print(f"[7] review CSV 저장 완료: {OUT_REVIEW_CSV}")
print(f"    rows: {len(df_out)}")

# ──────────────────────────────────────────────
# STEP 11: summary JSON 생성
# ──────────────────────────────────────────────
print("[8] summary JSON 생성")

overlap_class_counts = df_sorted["lesion_overlap_class"].value_counts().to_dict()
priority_counts_final = df_sorted["review_priority"].value_counts().to_dict()

patch_overlap_ids = sorted(
    df_sorted.loc[df_sorted["lesion_overlap_class"] == "patch_overlap", "crop_id"].tolist()
)
context192_overlap_only_ids = sorted(
    df_sorted.loc[df_sorted["lesion_overlap_class"] == "context192_overlap_only", "crop_id"].tolist()
)

n_no_overlap = int(overlap_class_counts.get("no_lesion_overlap", 0))

# HN-p95 계열 (no_lesion_overlap 중)
no_over_df = df_sorted[df_sorted["lesion_overlap_class"] == "no_lesion_overlap"]
n_hn_p95_overlap = int(
    no_over_df["qa_group"].apply(lambda g: "HN-p95" in str(g) or "HN-top10-crops" in str(g)).sum()
)
n_hn_large_bbox_overlap = int(
    no_over_df["qa_group"].apply(
        lambda g: ("HN-large-bbox" in str(g)) and ("HN-p95" not in str(g)) and ("HN-top10-crops" not in str(g))
    ).sum()
)

n_patients = int(df_sorted["patient_id"].nunique())

summary = {
    "n_rows": len(df_sorted),
    "n_patients": n_patients,
    "n_by_lesion_overlap_class": {k: int(v) for k, v in overlap_class_counts.items()},
    "n_by_review_priority": {k: int(v) for k, v in priority_counts_final.items()},
    "patch_overlap_crop_ids": patch_overlap_ids,
    "context192_overlap_only_crop_ids": context192_overlap_only_ids,
    "n_no_lesion_overlap": n_no_overlap,
    "n_hn_p95_overlap": n_hn_p95_overlap,
    "n_hn_large_bbox_overlap": n_hn_large_bbox_overlap,
    "note": {
        "review_sheet_only": True,
        "threshold_not_finalized": True,
        "lesion_conclusion_forbidden": True,
        "patch_overlap_should_not_be_used_as_hard_negative": True,
        "stage2_holdout_unused": True,
        "v2_unused": True,
        "volume_source_not_accessed": True,
    },
}

with open(OUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"[9] summary JSON 저장 완료: {OUT_SUMMARY_JSON}")

print("\n[완료] 모든 출력 파일 생성 완료.")
print(f"  review CSV : {OUT_REVIEW_CSV}")
print(f"  summary JSON: {OUT_SUMMARY_JSON}")
