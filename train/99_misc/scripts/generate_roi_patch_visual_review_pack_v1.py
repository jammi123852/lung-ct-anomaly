"""
Phase 5.53 - ROI/patch re-ranking 기반 visual review pack 생성

PNG 재생성 없음. model forward 없음. score 재계산 없음.
기존 Phase 5.52 결과 수정 없음.
threshold 확정 금지. 병변 성능 결론 금지.
"""

import json
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

# ──────────────────────────────────────────────────────────
# 입력 경로
# ──────────────────────────────────────────────────────────
REVIEW_V1_ROOT = (
    "outputs/second-stage-lesion-refiner-v1/review_annotations/"
    "hard_negative_top_score_qa_v1/roi_patch_reranking_review_v1"
)
INPUT_REVIEW_SHEET = f"{REVIEW_V1_ROOT}/roi_patch_reranking_review_sheet_v1.csv"

OUTPUT_ROOT = (
    "outputs/second-stage-lesion-refiner-v1/review_annotations/"
    "hard_negative_top_score_qa_v1/roi_patch_reranking_visual_review_pack_v1"
)
OUTPUT_MANIFEST_NAME = "roi_patch_reranking_visual_review_manifest_v1.csv"
OUTPUT_HTML_NAME = "roi_patch_reranking_visual_review_gallery_v1.html"
OUTPUT_GUIDE_MD_NAME = "roi_patch_reranking_visual_review_guide_v1.md"
OUTPUT_SUMMARY_JSON_NAME = "roi_patch_reranking_visual_review_summary_v1.json"
OUTPUT_ZIP_NAME = "roi_patch_reranking_visual_review_pack_v1.zip"

FORBIDDEN_PATH_KEYWORDS = ["stage2_holdout", "holdout", "v2"]

MANIFEST_COLS = [
    "review_order", "main_review_group", "crop_id", "patient_id",
    "qa_group", "qa_priority", "lesion_overlap_class", "diagnostic_class",
    "suggested_review_action",
    "rd4ad_l1_whole_crop_stored", "rd4ad_l1_roi_mean", "rd4ad_l1_outside_roi_mean",
    "rd4ad_l1_roi_patch_mean", "outside_to_roi_ratio", "whole_to_roi_patch_ratio",
    "rank_whole", "rank_roi_patch", "rank_shift_whole_to_roi_patch",
    "roi_ratio_fixed96", "patch_roi_ratio", "patch_bbox_clipped",
    "lesion_pixels_patch", "lesion_pixels_context192",
    "original_png_path", "copied_image_path", "contact_sheet_path",
    "gpt_or_user_label", "review_confidence", "review_note", "reviewer", "reviewed_at",
]

GROUP_PRIORITY = [
    "roi_patch_high_candidate",
    "outside_roi_driven_candidate",
    "lesion_overlap_exclude",
]
GROUP_DIR_MAP = {
    "roi_patch_high_candidate": "roi_patch_high",
    "outside_roi_driven_candidate": "outside_roi_driven",
    "lesion_overlap_exclude": "lesion_overlap_exclude",
    "overlap_between_groups": "overlap_between_groups",
}
GROUP_LABEL_MAP = {
    "roi_patch_high_candidate": "ROI∩patch high 후보",
    "outside_roi_driven_candidate": "outside ROI driven 후보",
    "lesion_overlap_exclude": "lesion overlap exclude 후보",
    "overlap_between_groups": "그룹 중복 후보",
    "clipped_patch_uncertain": "clipped patch caution 후보",
}


# ──────────────────────────────────────────────────────────
# Guard
# ──────────────────────────────────────────────────────────
def check_forbidden_path(path_str: str) -> None:
    p = str(path_str).lower()
    for kw in FORBIDDEN_PATH_KEYWORDS:
        if kw in p:
            raise RuntimeError(
                f"[GUARD] 금지된 경로 키워드 '{kw}' 감지: {path_str}"
            )


# ──────────────────────────────────────────────────────────
# main_review_group 결정
# ──────────────────────────────────────────────────────────
def get_main_group(diag_class: str) -> str:
    classes = [c.strip() for c in str(diag_class).split(";")]
    matched = [g for g in GROUP_PRIORITY if g in classes]
    if len(matched) >= 2:
        return "overlap_between_groups"
    if len(matched) == 1:
        return matched[0]
    return ""


# ──────────────────────────────────────────────────────────
# HTML 생성
# ──────────────────────────────────────────────────────────
CHECKLIST_ITEMS = [
    "노란 patch bbox와 초록 ROI 내부 구조가 실제로 보이는가?",
    "높은 error가 폐 안 구조 때문인가, ROI 밖/경계 때문인가?",
    "혈관/폐벽/기관지/큰 구조물 중 무엇에 가까운가?",
    "lesion_overlap_exclude라면 hard negative에서 제외해야 하는가?",
    "patch_bbox_clipped라면 patch score 해석이 가능한가?",
]


def _checklist_html() -> str:
    items = "".join(
        f'<li><input type="checkbox"> {item}</li>' for item in CHECKLIST_ITEMS
    )
    return f'<ul class="checklist">{items}</ul>'


def _img_card(row, use_relative: bool = True) -> str:
    src = str(row.get("copied_image_path", "")) if use_relative else ""
    if not src:
        src = str(row.get("original_png_path", ""))
    roi_patch = row.get("rd4ad_l1_roi_patch_mean", "N/A")
    outside_ratio = row.get("outside_to_roi_ratio", "N/A")
    roi_patch_str = (
        f"{roi_patch:.6f}" if isinstance(roi_patch, float) else str(roi_patch)
    )
    outside_ratio_str = (
        f"{outside_ratio:.3f}" if isinstance(outside_ratio, float) else str(outside_ratio)
    )
    return f"""
<div class="card">
  <div class="card-header">
    crop_id: {row['crop_id']} | patient: {row['patient_id']}<br>
    class: {row['diagnostic_class']}<br>
    roi_patch_score: {roi_patch_str} | outside_to_roi_ratio: {outside_ratio_str}
  </div>
  <img src="{src}" alt="crop_{row['crop_id']}" style="max-width:100%;max-height:400px;">
  {_checklist_html()}
</div>"""


def build_html(df_manifest: pd.DataFrame, df_clipped: pd.DataFrame) -> str:
    style = """
<style>
body { font-family: sans-serif; background: #f0f0f0; padding: 16px; }
.card { background: white; margin: 10px; padding: 12px; border-radius: 8px;
        display: inline-block; max-width: 480px; vertical-align: top; }
.card-header { font-size: 12px; color: #333; margin-bottom: 8px; }
.checklist { font-size: 11px; margin-top: 8px; }
h2 { margin-top: 32px; color: #222; }
.group { display: flex; flex-wrap: wrap; }
</style>"""

    sections = []
    for group_key in list(GROUP_DIR_MAP.keys()):
        sub = df_manifest[df_manifest["main_review_group"] == group_key]
        if len(sub) == 0:
            continue
        label = GROUP_LABEL_MAP.get(group_key, group_key)
        cards = "".join(
            _img_card(row, use_relative=True) for _, row in sub.iterrows()
        )
        sections.append(
            f'<h2>{label} ({len(sub)}개)</h2><div class="group">{cards}</div>'
        )

    # clipped_patch_uncertain 전용 섹션 (images/ 미복사 케이스만)
    manifest_crop_ids = set(df_manifest["crop_id"].astype(str).tolist())
    clipped_only = df_clipped[
        ~df_clipped["crop_id"].astype(str).isin(manifest_crop_ids)
    ]
    if len(clipped_only) > 0:
        label = GROUP_LABEL_MAP["clipped_patch_uncertain"]
        cards = "".join(
            _img_card(row, use_relative=False) for _, row in clipped_only.iterrows()
        )
        sections.append(
            f'<h2>{label} (복사 미포함, {len(clipped_only)}개)</h2>'
            f'<div class="group">{cards}</div>'
        )

    body = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Phase 5.53 ROI/Patch Visual Review</title>
{style}
</head>
<body>
<h1>Phase 5.53 ROI/Patch Score 기반 Visual Review</h1>
<p>threshold 확정 금지. 병변 성능 결론 금지. stage2_holdout/v2 접근 금지.</p>
{body}
</body>
</html>"""


# ──────────────────────────────────────────────────────────
# guide MD 생성
# ──────────────────────────────────────────────────────────
def build_guide_md() -> str:
    lines = [
        "# Phase 5.53 ROI/Patch Re-ranking Visual Review Pack 가이드",
        "",
        "## 왜 whole-crop 기준 대신 roi_patch 기준 후보를 보는가",
        "",
        "Phase 5.51에서 150개 hard negative 중 149개에서 outside_roi_mean > roi_mean 이 확인됐다.",
        "whole-crop score가 높은 케이스 대부분이 ROI 밖 오차로 구동되므로,",
        "whole-crop score만으로 폐 내부 false positive를 결론짓는 것은 위험하다.",
        "roi_patch 기준으로 다시 정렬한 후보를 시각 검토한다.",
        "",
        "## roi_patch_high_candidate",
        "",
        "ROI∩patch 내부 error가 높은 후보. 실제 폐 내부 false positive 원인 우선 검토 대상.",
        "",
        "## outside_roi_driven_candidate",
        "",
        "whole-crop 점수는 높지만 ROI 밖 error 영향이 큰 후보.",
        "outside_to_roi_ratio > 2이고 whole score가 상위권이지만 roi_patch score는 낮음.",
        "폐 안쪽 이상으로 해석 금지.",
        "",
        "## lesion_overlap_exclude",
        "",
        "병변 mask와 patch가 겹친 후보. hard negative 제외 후보로 유지.",
        "성능 결론에 사용 금지.",
        "",
        "## clipped_patch_uncertain",
        "",
        "patch bbox가 fixed96 경계에서 잘림. patch-local score 해석 주의.",
        "roi_patch score가 whole-crop score와 유사하게 나오면 patch-local 해석 아닌 crop-level에 가까움.",
        "",
        "## manual label 후보",
        "",
        "- vessel_branch",
        "- elongated_vessel",
        "- pleural_wall",
        "- bronchus_air_boundary",
        "- large_bbox_structure",
        "- fragmented_small_objects",
        "- irregular_large_object",
        "- nodule_suspect",
        "- outside_roi_artifact",
        "- unclear",
        "",
        "## 주의사항",
        "",
        "- threshold 확정 금지.",
        "- 병변 성능 결론 금지.",
        "- stage2_holdout/v2 접근 금지.",
        "- model forward, score 재계산 금지.",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────
def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    print(f"[INFO] project_root: {project_root}")
    run_ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    print(f"[INFO] run_timestamp: {run_ts}")

    # forbidden path guard
    check_forbidden_path(str(project_root / OUTPUT_ROOT))

    out_root = project_root / OUTPUT_ROOT
    if out_root.exists():
        sys.exit(f"[ABORT] output root 이미 존재함. 덮어쓰기 방지로 중단: {out_root}")

    # 입력 파일 존재 확인
    review_sheet_path = project_root / INPUT_REVIEW_SHEET
    check_forbidden_path(str(review_sheet_path))
    if not review_sheet_path.exists():
        sys.exit(f"[ABORT] review sheet 없음: {review_sheet_path}")

    # 입력 로드
    df = pd.read_csv(review_sheet_path)
    print(f"[INFO] review sheet rows: {len(df)}")

    # 입력 검증
    if len(df) != 150:
        sys.exit(f"[ABORT] row 수 불일치: {len(df)} (expected 150)")
    if "diagnostic_class" not in df.columns:
        sys.exit("[ABORT] diagnostic_class 컬럼 없음")
    if df["png_path"].isna().sum() > 0:
        sys.exit(f"[ABORT] png_path 누락: {df['png_path'].isna().sum()}건")

    n_roi_patch_high = int(
        df["diagnostic_class"].str.contains("roi_patch_high_candidate", na=False).sum()
    )
    n_outside_driven = int(
        df["diagnostic_class"].str.contains("outside_roi_driven_candidate", na=False).sum()
    )
    n_lesion_exclude = int(
        df["diagnostic_class"].str.contains("lesion_overlap_exclude", na=False).sum()
    )
    n_clipped = int(
        df["diagnostic_class"].str.contains("clipped_patch_uncertain", na=False).sum()
    )
    print(
        f"[CHECK] roi_patch_high={n_roi_patch_high}, outside_driven={n_outside_driven}, "
        f"lesion_exclude={n_lesion_exclude}, clipped={n_clipped}"
    )

    if n_roi_patch_high != 20:
        sys.exit(f"[ABORT] roi_patch_high_candidate 수 불일치: {n_roi_patch_high} (expected 20)")
    if n_outside_driven != 13:
        sys.exit(f"[ABORT] outside_roi_driven_candidate 수 불일치: {n_outside_driven} (expected 13)")
    if n_lesion_exclude != 9:
        sys.exit(f"[ABORT] lesion_overlap_exclude 수 불일치: {n_lesion_exclude} (expected 9)")

    # 대상 crop 선정 (3그룹 union, crop_id unique)
    mask_target = (
        df["diagnostic_class"].str.contains("roi_patch_high_candidate", na=False)
        | df["diagnostic_class"].str.contains("outside_roi_driven_candidate", na=False)
        | df["diagnostic_class"].str.contains("lesion_overlap_exclude", na=False)
    )
    df_target = df[mask_target].copy()
    df_target["crop_id"] = df_target["crop_id"].astype(str)
    df_target = df_target.drop_duplicates(subset="crop_id").reset_index(drop=True)
    n_unique = len(df_target)
    print(f"[INFO] unique crops (3그룹 union): {n_unique}")

    # clipped 전체 (HTML 섹션용)
    df_clipped = df[
        df["diagnostic_class"].str.contains("clipped_patch_uncertain", na=False)
    ].copy()
    df_clipped["crop_id"] = df_clipped["crop_id"].astype(str)

    # main_review_group 결정
    df_target["main_review_group"] = df_target["diagnostic_class"].apply(get_main_group)

    # 원본 PNG 존재 확인
    missing_png = [
        str(r["crop_id"])
        for _, r in df_target.iterrows()
        if not Path(str(r["png_path"])).exists()
    ]
    if missing_png:
        sys.exit(f"[ABORT] 원본 PNG 없음: {len(missing_png)}건 예: {missing_png[:3]}")

    # 모든 검증 통과 → output root 생성
    out_root.mkdir(parents=True, exist_ok=True)
    images_dir = out_root / "images"
    for subdir in GROUP_DIR_MAP.values():
        (images_dir / subdir).mkdir(parents=True, exist_ok=True)

    # review_order 부여 (그룹 우선순위 → roi_patch_mean 내림차순)
    group_order_list = list(GROUP_DIR_MAP.keys())

    def _group_sort_key(g: str) -> int:
        return group_order_list.index(g) if g in group_order_list else 99

    df_target["_group_sort"] = df_target["main_review_group"].apply(_group_sort_key)
    df_target = df_target.sort_values(
        ["_group_sort", "rd4ad_l1_roi_patch_mean"], ascending=[True, False]
    ).reset_index(drop=True)
    df_target["review_order"] = df_target.index + 1

    # PNG 복사 + copied_image_path 기록
    copied_paths = []
    for _, row in df_target.iterrows():
        grp = row["main_review_group"]
        subdir = GROUP_DIR_MAP.get(grp, "overlap_between_groups")
        src = Path(str(row["png_path"]))
        order_str = f"{int(row['review_order']):03d}"
        dest_name = f"{order_str}__{subdir}__{row['crop_id']}__{row['patient_id']}.png"
        dest = images_dir / subdir / dest_name
        shutil.copy2(str(src), str(dest))
        copied_paths.append(f"images/{subdir}/{dest_name}")
    df_target["copied_image_path"] = copied_paths

    # manifest 컬럼 정리
    df_target["original_png_path"] = df_target["png_path"]
    for col in ["gpt_or_user_label", "review_confidence", "review_note", "reviewer", "reviewed_at"]:
        df_target[col] = ""
    existing_manifest_cols = [c for c in MANIFEST_COLS if c in df_target.columns]
    df_manifest = df_target[existing_manifest_cols].copy()

    # 그룹별 crop_id 목록
    crop_ids_by_group = {}
    for g in GROUP_DIR_MAP.keys():
        crop_ids_by_group[g] = (
            df_target[df_target["main_review_group"] == g]["crop_id"].tolist()
        )
    n_overlap = len(crop_ids_by_group.get("overlap_between_groups", []))

    # 저장 전 검증
    if len(df_manifest) != n_unique:
        sys.exit(f"[ABORT] manifest row 수 불일치: {len(df_manifest)} != {n_unique}")
    if len(copied_paths) != n_unique:
        sys.exit(f"[ABORT] copied image 수 불일치: {len(copied_paths)} != {n_unique}")

    # HTML 생성 및 img src 검증
    html_content = build_html(df_manifest, df_clipped)
    if 'src="images/' not in html_content:
        sys.exit("[ABORT] HTML img src에 images/ 상대경로 없음")

    guide_md_content = build_guide_md()

    summary = {
        "run_timestamp": run_ts,
        "n_total_unique_crops": n_unique,
        "n_roi_patch_high": n_roi_patch_high,
        "n_outside_roi_driven": n_outside_driven,
        "n_lesion_overlap_exclude": n_lesion_exclude,
        "n_clipped_patch_uncertain": n_clipped,
        "n_overlap_between_groups": n_overlap,
        "crop_ids_by_group": crop_ids_by_group,
        "copied_image_count": len(copied_paths),
        "zip_created": False,
        "threshold_not_finalized": True,
        "lesion_conclusion_forbidden": True,
        "stage2_holdout_unused": True,
        "v2_unused": True,
        "volume_source_not_accessed": True,
        "png_regenerated": False,
    }

    # 저장
    manifest_path = out_root / OUTPUT_MANIFEST_NAME
    df_manifest.to_csv(str(manifest_path), index=False)
    print(f"[SAVE] manifest CSV: {manifest_path} ({len(df_manifest)} rows)")

    html_path = out_root / OUTPUT_HTML_NAME
    with open(str(html_path), "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[SAVE] HTML: {html_path}")

    guide_md_path = out_root / OUTPUT_GUIDE_MD_NAME
    with open(str(guide_md_path), "w", encoding="utf-8") as f:
        f.write(guide_md_content)
    print(f"[SAVE] Guide MD: {guide_md_path}")

    summary_json_path = out_root / OUTPUT_SUMMARY_JSON_NAME
    with open(str(summary_json_path), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] Summary JSON: {summary_json_path}")

    # ZIP 생성
    zip_path = out_root / OUTPUT_ZIP_NAME
    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(out_root.rglob("*")):
            if file_path.is_file() and file_path != zip_path:
                zf.write(str(file_path), arcname=str(file_path.relative_to(out_root)))
    summary["zip_created"] = True
    with open(str(summary_json_path), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] ZIP: {zip_path}")

    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zip_count = len(zf.namelist())
    print(f"[INFO] ZIP 내부 파일 수: {zip_count}")

    print(f"\n[DONE] Phase 5.53 완료. output root: {out_root}")
    print(f"\n=== 요약 ===")
    print(f"unique crops: {n_unique}")
    print(f"copied images: {len(copied_paths)}")
    print(f"roi_patch_high: {n_roi_patch_high}")
    print(f"outside_roi_driven: {n_outside_driven}")
    print(f"lesion_overlap_exclude: {n_lesion_exclude}")
    print(f"overlap_between_groups: {n_overlap}")
    print(f"zip 내부 파일 수: {zip_count}")


if __name__ == "__main__":
    main()
