"""
Phase 5.67 unreviewed P1/P2 visual review pack 생성 스크립트.

- Phase 5.66 read-only priority CSV 기반
- P1/P2 13개만 대상
- 기존 PNG 복사 (재생성 금지)
- manifest CSV / HTML gallery / guide MD / summary JSON / ZIP 생성
- 기존 파일 수정/삭제/덮어쓰기 금지
- model forward / score 재계산 / stage2_holdout 접근 금지
"""

import json
import pathlib
import shutil
import zipfile
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

IN_CSV = (
    BASE
    / "phase5_66_unreviewed_threshold_prioritized_review_v1"
    / "phase5_66_unreviewed_threshold_prioritized_review_v1.csv"
)
IN_JSON = (
    BASE
    / "phase5_66_unreviewed_threshold_prioritized_review_v1"
    / "phase5_66_unreviewed_threshold_prioritized_review_summary_v1.json"
)

OUT_ROOT = BASE / "phase5_67_unreviewed_p1_p2_visual_review_pack_v1"
OUT_IMAGES = OUT_ROOT / "images"
OUT_IMAGES_P1 = OUT_IMAGES / "P1_unreviewed_pass_both_p99"
OUT_IMAGES_P2 = OUT_IMAGES / "P2_unreviewed_pass_any_p99"
OUT_CSV = OUT_ROOT / "phase5_67_unreviewed_p1_p2_visual_review_manifest_v1.csv"
OUT_HTML = OUT_ROOT / "phase5_67_unreviewed_p1_p2_visual_review_gallery_v1.html"
OUT_MD = OUT_ROOT / "phase5_67_unreviewed_p1_p2_visual_review_guide_v1.md"
OUT_JSON = OUT_ROOT / "phase5_67_unreviewed_p1_p2_visual_review_summary_v1.json"
OUT_ZIP = OUT_ROOT / "phase5_67_unreviewed_p1_p2_visual_review_pack_v1.zip"

P1_LABEL = "P1_unreviewed_pass_both_p99"
P2_LABEL = "P2_unreviewed_pass_any_p99"


# ──────────────────────────────────────────────
# [Guard 1] output root 기존 존재 시 즉시 중단
# ──────────────────────────────────────────────

abort_if(OUT_ROOT.exists(), f"output root가 이미 존재함 (중단): {OUT_ROOT}")


# ──────────────────────────────────────────────
# [Guard 2] 입력 파일 존재 여부
# ──────────────────────────────────────────────

for p in [IN_CSV, IN_JSON]:
    abort_if(not p.exists(), f"입력 파일 없음: {p}")


# ──────────────────────────────────────────────
# 입력 파일 로드
# ──────────────────────────────────────────────

df = pd.read_csv(IN_CSV)


# ──────────────────────────────────────────────
# [Guard 3] 입력 row 수 / 컬럼 검증
# ──────────────────────────────────────────────

abort_if(len(df) != 110, f"입력 row 수 오류: {len(df)} (기대 110)")
abort_if("review_priority" not in df.columns, "review_priority 컬럼 없음")
abort_if("png_path" not in df.columns, "png_path 컬럼 없음")
abort_if("contact_sheet_path" not in df.columns, "contact_sheet_path 컬럼 없음")


# ──────────────────────────────────────────────
# P1/P2 필터링
# ──────────────────────────────────────────────

p1_df = df[df["review_priority"] == P1_LABEL].copy()
p2_df = df[df["review_priority"] == P2_LABEL].copy()

abort_if(len(p1_df) != 2, f"P1 count 오류: {len(p1_df)} (기대 2)")
abort_if(len(p2_df) != 11, f"P2 count 오류: {len(p2_df)} (기대 11)")

# crop_id 오름차순 정렬: P1 먼저, P2 다음
p1_df = p1_df.sort_values("crop_id").reset_index(drop=True)
p2_df = p2_df.sort_values("crop_id").reset_index(drop=True)

selected = pd.concat([p1_df, p2_df], ignore_index=True)
abort_if(len(selected) != 13, f"selected row 수 오류: {len(selected)} (기대 13)")

# review_order 부여 (1-indexed)
selected["review_order"] = range(1, len(selected) + 1)


# ──────────────────────────────────────────────
# [Guard 4] png_path 실제 파일 존재 13/13 확인
# ──────────────────────────────────────────────

missing_png = [p for p in selected["png_path"].tolist() if not pathlib.Path(p).exists()]
abort_if(len(missing_png) != 0, f"png_path 실제 파일 없음 {len(missing_png)}건: {missing_png[:5]}")

missing_cs = [p for p in selected["contact_sheet_path"].tolist() if not pathlib.Path(p).exists()]
abort_if(len(missing_cs) != 0, f"contact_sheet_path 실제 파일 없음 {len(missing_cs)}건: {missing_cs[:5]}")


# ──────────────────────────────────────────────
# priority 축약 레이블 (파일명용)
# ──────────────────────────────────────────────

PRIORITY_SHORT = {
    P1_LABEL: "P1",
    P2_LABEL: "P2",
}

PRIORITY_SUBDIR = {
    P1_LABEL: OUT_IMAGES_P1,
    P2_LABEL: OUT_IMAGES_P2,
}


# ──────────────────────────────────────────────
# 저장 직전 Guard
# ──────────────────────────────────────────────

abort_if(OUT_ROOT.exists(), f"저장 직전 output root가 이미 존재함 (재검증): {OUT_ROOT}")


# ──────────────────────────────────────────────
# 출력 디렉토리 생성 (모든 검증 통과 후)
# ──────────────────────────────────────────────

OUT_ROOT.mkdir(parents=True, exist_ok=False)
OUT_IMAGES_P1.mkdir(parents=True, exist_ok=True)
OUT_IMAGES_P2.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# PNG 복사 및 copied_image_path 기록
# ──────────────────────────────────────────────

copied_paths = []
for _, row in selected.iterrows():
    order = int(row["review_order"])
    priority = row["review_priority"]
    crop_id = row["crop_id"]
    patient_id = str(row["patient_id"])
    short = PRIORITY_SHORT[priority]
    subdir = PRIORITY_SUBDIR[priority]

    src = pathlib.Path(row["png_path"])
    dst_name = f"{order:03d}__{short}__crop{crop_id}__{patient_id}.png"
    dst = subdir / dst_name

    shutil.copy2(str(src), str(dst))
    copied_paths.append(str(dst))

selected["copied_image_path"] = copied_paths


# ──────────────────────────────────────────────
# [Guard 5] copied image 13/13 확인
# ──────────────────────────────────────────────

missing_copied = [p for p in copied_paths if not pathlib.Path(p).exists()]
abort_if(len(missing_copied) != 0, f"copied image 파일 없음 {len(missing_copied)}건: {missing_copied[:5]}")


# ──────────────────────────────────────────────
# manifest CSV 컬럼 정렬 및 저장
# ──────────────────────────────────────────────

MANIFEST_COLS = [
    "review_order", "review_priority", "crop_id", "patient_id",
    "qa_group", "qa_priority", "lesion_overlap_class", "threshold_application_group",
    "pass_roi_mean_p95", "pass_lung_roi_mean_p95", "pass_any_p95", "pass_both_p95",
    "pass_roi_mean_p99", "pass_lung_roi_mean_p99", "pass_any_p99", "pass_both_p99",
    "rd4ad_l1_roi_mean", "rd4ad_l1_lung_channel_roi_mean",
    "rd4ad_l1_roi_patch_mean", "rd4ad_l1_outside_roi_mean",
    "roi_ratio_fixed96", "patch_roi_ratio", "patch_bbox_clipped",
    "original_png_path", "copied_image_path", "contact_sheet_path",
    "user_label", "user_confidence", "user_note", "reviewer", "reviewed_at",
]

selected = selected.rename(columns={"png_path": "original_png_path"})

for col in MANIFEST_COLS:
    if col not in selected.columns:
        selected[col] = ""

manifest_df = selected[MANIFEST_COLS].copy()
manifest_df.to_csv(OUT_CSV, index=False)


# ──────────────────────────────────────────────
# HTML gallery 생성
# ──────────────────────────────────────────────

CHECKLIST_ITEMS = [
    "노란 patch bbox가 실제 폐 내부/경계 구조를 잡는가?",
    "빨간 lesion contour가 있다면 병변 overlap 후보는 아닌가?",
    "초록 ROI 내부 구조가 score 상승 원인으로 보이는가?",
    "폐벽/흉막/횡격막 경계인가?",
    "큰 구조물/종격동/심장/폐문부 인접 구조인가?",
    "혈관분기/기관지/공기층 경계인가?",
    "outside ROI 영향이 섞였는가?",
    "최종 label을 무엇으로 둘 것인가?",
]

LABEL_CANDIDATES = [
    "pleural_wall", "large_bbox_structure", "vessel_branch",
    "bronchus_air_boundary", "outside_roi_artifact", "fragmented_small_objects",
    "irregular_large_object", "unclear", "possible_lesion_overlap",
]

now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def priority_section_html(priority_label, rows_df, short_label, color):
    html = f'<h2 style="color:{color};">{priority_label} ({len(rows_df)}개)</h2>\n'
    for _, row in rows_df.iterrows():
        order = int(row["review_order"])
        crop_id = row["crop_id"]
        patient_id = row["patient_id"]
        priority = row["review_priority"]
        roi_mean = row["rd4ad_l1_roi_mean"]
        lung_roi_mean = row["rd4ad_l1_lung_channel_roi_mean"]
        pass_any_p99 = row["pass_any_p99"]
        pass_both_p99 = row["pass_both_p99"]
        pass_any_p95 = row["pass_any_p95"]
        pass_both_p95 = row["pass_both_p95"]

        # 상대경로 (OUT_ROOT 기준)
        abs_path = pathlib.Path(row["copied_image_path"])
        rel_path = abs_path.relative_to(OUT_ROOT)

        checklist_html = "<ol>\n"
        for item in CHECKLIST_ITEMS:
            checklist_html += f"  <li>{item}</li>\n"
        checklist_html += "</ol>\n"

        labels_html = " / ".join(LABEL_CANDIDATES)

        html += f"""
<div style="border:2px solid {color};margin:16px 0;padding:12px;border-radius:6px;">
  <div style="font-size:14px;margin-bottom:6px;">
    <b>#{order:03d}</b> &nbsp;|&nbsp;
    <b>crop_id:</b> {crop_id} &nbsp;|&nbsp;
    <b>patient_id:</b> {patient_id} &nbsp;|&nbsp;
    <b>priority:</b> {priority}<br/>
    <b>roi_mean:</b> {roi_mean:.6f} &nbsp;|&nbsp;
    <b>lung_roi_mean:</b> {lung_roi_mean:.6f}<br/>
    <b>pass_any_p95:</b> {pass_any_p95} &nbsp;|&nbsp;
    <b>pass_both_p95:</b> {pass_both_p95} &nbsp;|&nbsp;
    <b>pass_any_p99:</b> {pass_any_p99} &nbsp;|&nbsp;
    <b>pass_both_p99:</b> {pass_both_p99}
  </div>
  <img src="{rel_path}" style="max-width:800px;width:100%;border:1px solid #ccc;" alt="crop{crop_id}"/><br/>
  <div style="margin-top:8px;">
    <b>체크리스트:</b>
    {checklist_html}
    <b>manual label 후보:</b> {labels_html}
  </div>
</div>
"""
    return html


p1_rows = manifest_df[manifest_df["review_priority"] == P1_LABEL]
p2_rows = manifest_df[manifest_df["review_priority"] == P2_LABEL]

html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8"/>
<title>Phase 5.67 Unreviewed P1/P2 Visual Review Gallery</title>
<style>
body {{ font-family: sans-serif; max-width: 900px; margin: 40px auto; }}
h1 {{ color: #333; }}
h2 {{ margin-top: 32px; }}
</style>
</head>
<body>
<h1>Phase 5.67 Unreviewed P1/P2 Visual Review Gallery</h1>
<p>생성일시: {now_str}</p>
<p>총 {len(manifest_df)}개 (P1: {len(p1_rows)}개, P2: {len(p2_rows)}개)</p>
<hr/>
{priority_section_html(P1_LABEL, p1_rows, "P1", "#c0392b")}
<hr/>
{priority_section_html(P2_LABEL, p2_rows, "P2", "#2980b9")}
</body>
</html>
"""

with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html_content)


# ──────────────────────────────────────────────
# guide MD 생성
# ──────────────────────────────────────────────

md_lines = [
    "# Phase 5.67 Unreviewed P1/P2 Visual Review Guide",
    "",
    f"생성일시: {now_str}",
    "",
    "---",
    "",
    "## 1. 이 pack의 목적",
    "",
    "이 pack은 Phase 5.66에서 우선순위화된 unreviewed_qa_candidate 110개 중 "
    "p99 기준 고신뢰 후보인 P1/P2 13개만을 대상으로 하는 visual review pack이다. "
    "reviewer가 어떤 후보부터 확인해야 하는지를 명확히 하기 위해 생성되었다.",
    "",
    "---",
    "",
    "## 2. P1/P2 기준 설명",
    "",
    "**P1 — pass_both_p99 (2개)**",
    "roi_mean과 lung_channel_roi_mean 두 채널 모두 p99 threshold를 초과한 후보. "
    "두 측정 기준이 일치하므로 신뢰도가 가장 높다.",
    "",
    "**P2 — pass_any_p99 (11개)**",
    "두 채널 중 하나만 p99 초과. P1보다 신뢰도가 약간 낮지만 여전히 고신뢰 후보.",
    "",
    "---",
    "",
    "## 3. 이번 review 대상이 아닌 항목",
    "",
    "- P3/P4/P5: 이번 pack에 포함되지 않음",
    "- fp_cause_review_pool 29개: Phase 5.58에서 visual QA 완료, 반복하지 않음",
    "- lesion_overlap_excluded 9개: 제외 유지",
    "- outside_roi_artifact_separated 2개: 제외 유지",
    "",
    "---",
    "",
    "## 4. reviewer 작업 절차",
    "",
    "1. `phase5_67_unreviewed_p1_p2_visual_review_gallery_v1.html`을 브라우저에서 연다.",
    "2. P1부터 순서대로 이미지를 확인한다.",
    "3. 체크리스트 8개 항목을 검토한다.",
    "4. manual label 후보 중 하나를 선택한다.",
    "5. `phase5_67_unreviewed_p1_p2_visual_review_manifest_v1.csv`의 "
    "`user_label`, `user_confidence`, `user_note`, `reviewer`, `reviewed_at` 컬럼을 채운다.",
    "",
    "---",
    "",
    "## 5. 금지 항목",
    "",
    "- threshold 확정 금지",
    "- 병변 성능 결론 금지",
    "- hard negative 최종 채택 금지",
    "- stage2_holdout/v2 접근 금지",
    "- model forward 금지",
    "- score 재계산 금지",
    "- 기존 PNG 원본 수정 금지",
    "- 기존 CSV/JSON/MD 수정 금지",
    "",
    "---",
    "",
    "## 6. manual label 후보",
    "",
    "| label | 설명 |",
    "|-------|------|",
    "| pleural_wall | 폐벽/흉막/횡격막 경계 |",
    "| large_bbox_structure | 큰 구조물/종격동/심장/폐문부 인접 |",
    "| vessel_branch | 혈관분기 |",
    "| bronchus_air_boundary | 기관지/공기층 경계 |",
    "| outside_roi_artifact | outside ROI 영향 |",
    "| fragmented_small_objects | 파편화된 소형 구조물 |",
    "| irregular_large_object | 불규칙 대형 구조물 |",
    "| unclear | 판단 불가 |",
    "| possible_lesion_overlap | 병변 overlap 가능성 |",
]

with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines) + "\n")


# ──────────────────────────────────────────────
# summary JSON 생성
# ──────────────────────────────────────────────

p1_crop_ids = sorted(manifest_df.loc[manifest_df["review_priority"] == P1_LABEL, "crop_id"].tolist())
p2_crop_ids = sorted(manifest_df.loc[manifest_df["review_priority"] == P2_LABEL, "crop_id"].tolist())

n_p3 = (df["review_priority"] == "P3_unreviewed_pass_both_p95_only").sum()
n_p4 = (df["review_priority"] == "P4_unreviewed_pass_any_p95_only").sum()
n_p5 = (df["review_priority"] == "P5_unreviewed_below_p95").sum()

summary = {
    "n_total_input": len(df),
    "n_selected": len(manifest_df),
    "n_P1": len(p1_crop_ids),
    "n_P2": len(p2_crop_ids),
    "selected_crop_ids": sorted(manifest_df["crop_id"].tolist()),
    "P1_crop_ids": p1_crop_ids,
    "P2_crop_ids": p2_crop_ids,
    "copied_image_count": len(copied_paths),
    "zip_created": True,
    "excluded_P3_count": int(n_p3),
    "excluded_P4_count": int(n_p4),
    "excluded_P5_count": int(n_p5),
    "notes": {
        "threshold_not_finalized": True,
        "visual_review_pack_only": True,
        "unreviewed_p1_p2_only": True,
        "reviewed_fp_pool_not_repeated": True,
        "hard_negative_not_finalized": True,
        "lesion_conclusion_forbidden": True,
        "stage2_holdout_unused": True,
        "v2_unused": True,
        "model_forward_not_run": True,
        "score_recalculation_not_run": True,
        "png_regenerated_false": True,
        "original_files_unmodified": True,
    },
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# ZIP 생성 (OUT_ROOT 아래 모든 파일, ZIP 자체 제외)
# ──────────────────────────────────────────────

with zipfile.ZipFile(str(OUT_ZIP), "w", zipfile.ZIP_DEFLATED) as zf:
    for file_path in sorted(OUT_ROOT.rglob("*")):
        if file_path == OUT_ZIP:
            continue
        if file_path.is_file():
            arcname = file_path.relative_to(OUT_ROOT)
            zf.write(str(file_path), str(arcname))

zip_file_count = 0
with zipfile.ZipFile(str(OUT_ZIP), "r") as zf:
    zip_file_count = len(zf.namelist())


# ──────────────────────────────────────────────
# [Guard 6] HTML img src 상대경로 검증
# ──────────────────────────────────────────────

html_text = OUT_HTML.read_text(encoding="utf-8")
abort_if(
    "/home/jinhy" in html_text,
    "HTML에 절대경로가 포함됨 (images/ 상대경로여야 함)",
)


# ──────────────────────────────────────────────
# [Guard 7] manifest row 수 최종 검증
# ──────────────────────────────────────────────

final_manifest = pd.read_csv(OUT_CSV)
abort_if(len(final_manifest) != 13, f"최종 manifest row 수 오류: {len(final_manifest)} (기대 13)")


# ──────────────────────────────────────────────
# 최종 보고 출력
# ──────────────────────────────────────────────

print("=" * 60)
print("검토 판정: 전체 통과")
print("=" * 60)
print(f"output root         : {OUT_ROOT}")
print(f"ZIP                 : {OUT_ZIP}")
print(f"manifest CSV        : {OUT_CSV} ({len(final_manifest)}행)")
print(f"HTML gallery        : {OUT_HTML}")
print(f"guide MD            : {OUT_MD}")
print(f"summary JSON        : {OUT_JSON}")
print("-" * 60)
print(f"P1 count            : {len(p1_crop_ids)} / crop_id: {p1_crop_ids}")
print(f"P2 count            : {len(p2_crop_ids)} / crop_id: {p2_crop_ids}")
print(f"selected 총 row 수  : {len(manifest_df)}")
print(f"copied image 수     : {len(copied_paths)}")
print(f"ZIP 내부 파일 수    : {zip_file_count}")
print("-" * 60)
print(f"P3 제외 수          : {n_p3}")
print(f"P4 제외 수          : {n_p4}")
print(f"P5 제외 수          : {n_p5}")
print("P3/P4/P5 미포함     : True")
print("기존 Phase 5.66 파일 미수정 : True")
print("기존 PNG 원본 미수정        : True")
print("model forward 없음         : True")
print("score 재계산 없음          : True")
print("threshold 확정 없음        : True")
print("병변 성능 결론 없음        : True")
print("stage2_holdout/v2 미접근   : True")
print("=" * 60)
print()
print("[다음 단계 제안]")
print("1. HTML gallery를 브라우저에서 열어 P1/P2 13개 이미지를 visual review 진행")
print("2. manifest CSV의 user_label/user_note 컬럼을 채운 후 다음 phase로 이동")
print("3. P3 46개 review가 필요한 경우 별도 승인 후 Phase 5.68에서 처리")
print("4. threshold 확정 및 hard negative 최종 채택은 visual review 완료 후 별도 승인")
