"""
P1/P2 lesion-overlap visual review helper
phase: 547

생성 파일 (4개):
  1. helper CSV
  2. MD guide
  3. HTML gallery
  4. summary JSON

입력 파일 (3개):
  1. hard_negative_top_score_overlap_prioritized_review_sheet_v1.csv
  2. hard_negative_top_score_overlap_prioritized_review_summary_v1.json
  3. manifest_png_index_context_overlay_v1.csv

금지 사항:
  - PNG 재생성/복사 금지
  - contact sheet 재생성 금지
  - score 재계산 금지
  - threshold 확정 금지
  - 병변 성능 결론 금지
  - stage2_holdout/v2 접근 금지
  - volume source 접근 금지
"""

import os
import json
import pandas as pd
from datetime import datetime

# ─── 경로 설정 ───────────────────────────────────────────────────────────────
ROOT = "/home/jinhy/project/lung-ct-anomaly"

INPUT_REVIEW_SHEET = os.path.join(
    ROOT,
    "outputs/second-stage-lesion-refiner-v1/review_annotations/hard_negative_top_score_qa_v1",
    "hard_negative_top_score_overlap_prioritized_review_sheet_v1.csv",
)
INPUT_REVIEW_SUMMARY = os.path.join(
    ROOT,
    "outputs/second-stage-lesion-refiner-v1/review_annotations/hard_negative_top_score_qa_v1",
    "hard_negative_top_score_overlap_prioritized_review_summary_v1.json",
)
INPUT_PNG_INDEX = os.path.join(
    ROOT,
    "outputs/second-stage-lesion-refiner-v1/visualizations/hard_negative_top_score_context_overlay_qa_v1",
    "manifest_png_index_context_overlay_v1.csv",
)

OUT_DIR = os.path.join(
    ROOT,
    "outputs/second-stage-lesion-refiner-v1/review_annotations/hard_negative_top_score_qa_v1",
)

OUT_CSV = os.path.join(OUT_DIR, "p1_p2_overlap_visual_review_helper_v1.csv")
OUT_MD = os.path.join(OUT_DIR, "p1_p2_overlap_visual_review_guide_v1.md")
OUT_HTML = os.path.join(OUT_DIR, "p1_p2_overlap_visual_review_gallery_v1.html")
OUT_JSON = os.path.join(OUT_DIR, "p1_p2_overlap_visual_review_summary_v1.json")


# ─── 입력 파일 존재 확인 ─────────────────────────────────────────────────────
def check_inputs():
    for path in [INPUT_REVIEW_SHEET, INPUT_REVIEW_SUMMARY, INPUT_PNG_INDEX]:
        assert os.path.exists(path), f"입력 파일 없음: {path}"
    print("[OK] 입력 파일 3개 존재 확인")


# ─── P1/P2 추출 및 CSV 생성 ──────────────────────────────────────────────────
OUTPUT_COLUMNS = [
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
    "lesion_pixels_patch",
    "lesion_pixels_context192",
    "lesion_overlap_ratio_patch",
    "lesion_overlap_ratio_context192",
    "png_path",
    "contact_sheet_path",
    "check_lesion_contour_on_candidate",
    "check_candidate_is_true_lesion",
    "check_candidate_is_vessel_or_wall",
    "check_hard_negative_exclude",
    "manual_label",
    "manual_confidence",
    "manual_note",
    "reviewer",
    "reviewed_at",
]


def build_helper_csv():
    df = pd.read_csv(INPUT_REVIEW_SHEET)

    p1 = df[df["review_priority"] == "P1_patch_overlap"].copy()
    p2 = df[df["review_priority"] == "P2_context_overlap_only"].copy()

    # P1: lesion_pixels_patch 내림차순 정렬
    p1 = p1.sort_values("lesion_pixels_patch", ascending=False).reset_index(drop=True)
    # P2: 순서 유지
    p2 = p2.reset_index(drop=True)

    merged = pd.concat([p1, p2], ignore_index=True)

    # review_order 재부여 (1~12)
    merged["review_order"] = range(1, len(merged) + 1)

    # 빈 컬럼 추가
    for col in [
        "check_lesion_contour_on_candidate",
        "check_candidate_is_true_lesion",
        "check_candidate_is_vessel_or_wall",
        "check_hard_negative_exclude",
        "manual_label",
        "manual_confidence",
        "manual_note",
        "reviewer",
        "reviewed_at",
    ]:
        if col not in merged.columns:
            merged[col] = ""
        else:
            # 기존 컬럼이 있으면 빈값으로 초기화 (기존 입력 없음)
            merged[col] = ""

    # 출력 컬럼만 선택
    out_df = merged[OUTPUT_COLUMNS].copy()
    out_df.to_csv(OUT_CSV, index=False)

    print(f"[OK] helper CSV 생성: {OUT_CSV}")
    print(f"     rows={len(out_df)}, P1={len(p1)}, P2={len(p2)}")
    return out_df


# ─── PNG 경로 존재 확인 ───────────────────────────────────────────────────────
def verify_png_paths(df):
    missing = []
    for _, row in df.iterrows():
        if not os.path.exists(str(row["png_path"])):
            missing.append(row["png_path"])
    if missing:
        print(f"[WARN] PNG 누락 {len(missing)}개:")
        for p in missing:
            print(f"  {p}")
        png_all_exist = False
    else:
        print(f"[OK] png_path {len(df)}/{len(df)} 존재 확인")
        png_all_exist = True
    return png_all_exist


# ─── HTML gallery 생성 ───────────────────────────────────────────────────────
def build_html_gallery(df):
    p1_rows = df[df["review_priority"] == "P1_patch_overlap"]
    p2_rows = df[df["review_priority"] == "P2_context_overlap_only"]

    def row_html(row, section_color):
        png_path = str(row["png_path"])
        # 절대경로는 file:// 프로토콜 적용
        if png_path.startswith("/"):
            img_src = f"file://{png_path}"
        else:
            img_src = png_path

        header_text = (
            f"[{row['review_priority']}] "
            f"patient_id={row['patient_id']} | "
            f"crop_id={row['crop_id']} | "
            f"qa_group={row['qa_group']} | "
            f"lesion_pixels_patch={row['lesion_pixels_patch']}"
        )

        return f"""
<div style="margin-bottom: 40px; padding: 16px; background-color: {section_color}; border-radius: 8px;">
  <h3 style="margin: 0 0 8px 0; font-size: 14px; color: #333;">#{row['review_order']} {header_text}</h3>
  <img src="{img_src}" style="width: 600px; display: block; border: 2px solid #888;" alt="crop_{row['crop_id']}" />
  <div style="margin-top: 12px; font-size: 13px; color: #444;">
    <b>체크리스트:</b>
    <ol>
      <li>빨간 lesion contour가 후보 중심에 걸리는가? (check_lesion_contour_on_candidate)</li>
      <li>병변처럼 보이는가, 혈관/폐벽/기관지처럼 보이는가? (check_candidate_is_true_lesion / check_candidate_is_vessel_or_wall)</li>
      <li>hard negative로 제외해야 하는가? (check_hard_negative_exclude)</li>
      <li>manual_label 후보는 무엇인가? (vessel_branch / elongated_vessel / pleural_wall / bronchus_air_boundary / nodule_suspect / large_bbox_structure / fragmented_small_objects / irregular_large_object / unclear)</li>
    </ol>
    <p style="font-size: 12px; color: #888;">
      crop_score_l1_mean={row['crop_score_l1_mean']:.6f} |
      padim_score_mean={row['padim_score_mean']:.3f} |
      padim_score_max={row['padim_score_max']:.3f} |
      lesion_pixels_patch={row['lesion_pixels_patch']} |
      lesion_pixels_context192={row['lesion_pixels_context192']}
    </p>
  </div>
</div>
"""

    p1_html = ""
    for _, row in p1_rows.iterrows():
        p1_html += row_html(row, "#fff0f0")

    p2_html = ""
    for _, row in p2_rows.iterrows():
        p2_html += row_html(row, "#fffbe0")

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>P1/P2 Lesion Overlap Visual Review Gallery v1</title>
  <style>
    body {{
      background: #ffffff;
      font-family: 'Segoe UI', sans-serif;
      margin: 32px;
      color: #222;
    }}
    h1 {{ font-size: 20px; color: #111; }}
    h2 {{ font-size: 16px; padding: 10px 16px; border-radius: 6px; margin-top: 40px; }}
    .p1-header {{ background-color: #ffcccc; color: #600; }}
    .p2-header {{ background-color: #fff176; color: #554400; }}
    .note {{
      background: #f0f0f0;
      border-left: 4px solid #999;
      padding: 10px 14px;
      font-size: 12px;
      margin-bottom: 24px;
    }}
  </style>
</head>
<body>
  <h1>P1/P2 Lesion Overlap Visual Review Gallery v1</h1>
  <div class="note">
    <b>주의:</b>
    threshold 확정 금지 &nbsp;|&nbsp;
    병변 성능 결론 금지 &nbsp;|&nbsp;
    stage2_holdout/v2 미접근 &nbsp;|&nbsp;
    PNG 복사 없이 원본 경로 참조 &nbsp;|&nbsp;
    P1은 hard negative 단정 금지
  </div>

  <h2 class="p1-header">P1: patch_overlap (n={len(p1_rows)}) — 병변과 직접 겹침, hard negative 단정 금지</h2>
  {p1_html}

  <h2 class="p2-header">P2: context192_overlap_only (n={len(p2_rows)}) — 병변 주변 구조물 여부 확인</h2>
  {p2_html}

</body>
</html>
"""

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] HTML gallery 생성: {OUT_HTML}")


# ─── summary JSON 생성 ────────────────────────────────────────────────────────
def build_summary_json(df, png_all_exist):
    p1_crop_ids = sorted(
        df[df["review_priority"] == "P1_patch_overlap"]["crop_id"].tolist()
    )
    p2_crop_ids = sorted(
        df[df["review_priority"] == "P2_context_overlap_only"]["crop_id"].tolist()
    )

    summary = {
        "n_p1": len(p1_crop_ids),
        "n_p2": len(p2_crop_ids),
        "n_total": len(df),
        "p1_crop_ids": p1_crop_ids,
        "p2_crop_ids": p2_crop_ids,
        "png_path_all_exist": png_all_exist,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": {
            "png_not_copied": True,
            "threshold_not_finalized": True,
            "lesion_conclusion_forbidden": True,
            "stage2_holdout_unused": True,
            "v2_unused": True,
            "volume_source_not_accessed": True,
        },
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] summary JSON 생성: {OUT_JSON}")
    return summary


# ─── MD guide 생성 ────────────────────────────────────────────────────────────
def build_md_guide(summary):
    p1_ids_str = ", ".join(str(x) for x in summary["p1_crop_ids"])
    p2_ids_str = ", ".join(str(x) for x in summary["p2_crop_ids"])

    md = f"""# P1/P2 Lesion Overlap Visual Review Guide v1

생성일: {summary['created_at']}

---

## 1. P1/P2만 먼저 보는 이유

전체 150개 hard negative 후보 중, lesion mask와 직접 또는 간접 겹치는 12개(P1=9, P2=3)를 우선 리뷰한다.

이유:
- P1/P2는 hard negative/false positive로 단정할 수 없는 후보다.
- 병변 contour가 실제로 걸려 있는지, 아니면 병변 주변 구조물인지 시각적으로 확인해야 한다.
- 이 12개를 먼저 분류해야 이후 전체 hard negative 정제 방향을 잡을 수 있다.

---

## 2. P1_patch_overlap (n=9): hard negative/false positive 단정 금지

crop_id 목록: {p1_ids_str}

- patch 영역 내 lesion pixel이 1개 이상 포함된 후보다.
- 병변의 일부가 후보 안에 들어와 있을 가능성이 있다.
- hard negative 또는 false positive로 바로 단정하지 않는다.
- 반드시 빨간 contour(lesion)가 후보 중심부에 걸리는지 시각적으로 확인한다.

---

## 3. P2_context_overlap_only (n=3): 병변 주변 구조물 여부 확인

crop_id 목록: {p2_ids_str}

- patch 영역에는 lesion pixel이 없지만, context192 영역에 lesion pixel이 포함된 후보다.
- 병변 바로 옆에 붙어 있는 혈관, 폐벽, 기관지 경계 구조물일 가능성이 높다.
- 병변과의 공간적 관계를 확인하여 manual_label을 부여한다.

---

## 4. 빨간 contour(lesion)와 초록 contour(ROI) 보는 법

- 빨간 contour: lesion mask 경계 (병변 위치)
- 초록 contour: 후보 patch 또는 ROI 경계 (탐지된 후보 위치)
- 두 contour가 겹치거나 매우 인접한 경우 → P1 또는 P2에 해당
- 초록 contour 중심이 빨간 contour 안에 들어와 있으면 병변 후보에 가깝다
- 초록 contour가 빨간 contour 바깥에 위치하면 병변 주변 구조물에 가깝다

---

## 5. manual_label 후보 목록

| 값 | 설명 |
|---|---|
| vessel_branch | 혈관 분지 |
| elongated_vessel | 길게 뻗은 혈관 |
| pleural_wall | 흉막/폐벽 경계 |
| bronchus_air_boundary | 기관지-공기 경계 |
| nodule_suspect | 결절 의심 (병변 후보) |
| large_bbox_structure | 큰 bbox에 포함된 구조물 |
| fragmented_small_objects | 작은 단편화 객체들 |
| irregular_large_object | 불규칙한 큰 객체 |
| unclear | 판단 불가 |

---

## 6. 절대 금지 사항

- **threshold 확정 금지**: 이 리뷰는 threshold 최종 결정을 위한 것이 아니다.
- **병변 성능 결론 금지**: P1/P2 분포를 보고 모델 성능을 판단하지 않는다.
- **stage2_holdout 접근 금지**: 이 리뷰는 dev set 범위 내에서만 진행한다.
- **v2 접근 금지**: 현재 v1 기준으로만 리뷰한다.
- **PNG 복사/재생성 금지**: HTML gallery는 원본 PNG 절대경로를 직접 참조한다.
- **score 재계산 금지**: 기존 CSV 수치를 그대로 사용한다.
- **volume source 접근 금지**: CT volume 원본 파일에 접근하지 않는다.

---

## 7. 리뷰 결과 기록 방법

helper CSV(`p1_p2_overlap_visual_review_helper_v1.csv`)의 빈 컬럼에 리뷰 결과를 기록한다:

| 컬럼 | 설명 |
|---|---|
| check_lesion_contour_on_candidate | 빨간 contour가 후보 중심에 걸리는가 (yes/no/partial) |
| check_candidate_is_true_lesion | 병변처럼 보이는가 (yes/no/unclear) |
| check_candidate_is_vessel_or_wall | 혈관/폐벽처럼 보이는가 (yes/no/unclear) |
| check_hard_negative_exclude | hard negative 제외 대상인가 (yes/no/hold) |
| manual_label | 위 목록 중 선택 |
| manual_confidence | 확신도 (high/medium/low) |
| manual_note | 자유 메모 |
| reviewer | 리뷰어 이름 |
| reviewed_at | 리뷰 날짜 (YYYY-MM-DD) |
"""

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[OK] MD guide 생성: {OUT_MD}")


# ─── 메인 실행 ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("P1/P2 visual review helper 생성 시작")
    print("=" * 60)

    check_inputs()

    df = build_helper_csv()
    png_all_exist = verify_png_paths(df)
    build_html_gallery(df)
    summary = build_summary_json(df, png_all_exist)
    build_md_guide(summary)

    print()
    print("=" * 60)
    print("생성 완료")
    print(f"  CSV  : {OUT_CSV}")
    print(f"  MD   : {OUT_MD}")
    print(f"  HTML : {OUT_HTML}")
    print(f"  JSON : {OUT_JSON}")
    print("=" * 60)


if __name__ == "__main__":
    main()
