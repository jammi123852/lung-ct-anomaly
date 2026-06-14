"""
Phase 5-4-5: Hard negative top-score context overlay 150개 후보
lesion mask overlap 분석 스크립트

생성 파일:
  - hard_negative_top_score_lesion_overlap_analysis_v1.csv
  - hard_negative_top_score_lesion_overlap_summary_v1.json
"""

import os
import gc
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
VOL_ROOT = Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1"
    "/volumes_npy"
)

QA_MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/evaluation"
    / "rd4ad_2p5d_normal_mw_fixed96_v1/hard_negative_top_score_qa_v1"
    / "hard_negative_top_score_qa_manifest_v1.csv"
)
CROP_MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/crops"
    / "rd4ad_train_2p5d_mw_fixed96_thr001_v1/manifests"
    / "crop_manifest_rd4ad_train_2p5d_mw_fixed96_thr001_v1.csv"
)
CTX_INDEX_PATH = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/visualizations"
    / "hard_negative_top_score_context_overlay_qa_v1"
    / "manifest_png_index_context_overlay_v1.csv"
)

OUT_DIR = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/evaluation"
    / "rd4ad_2p5d_normal_mw_fixed96_v1/hard_negative_top_score_qa_v1"
)
OUT_CSV = OUT_DIR / "hard_negative_top_score_lesion_overlap_analysis_v1.csv"
OUT_JSON = OUT_DIR / "hard_negative_top_score_lesion_overlap_summary_v1.json"


# ── helper functions ──────────────────────────────────────────────────────────

def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


def _count_lesion_pixels(mask_2d, y0, x0, y1, x1, H, W):
    """지정 bbox 내 lesion pixel 수 반환. bbox는 clamp 후 처리."""
    y0c = _clamp(int(y0), 0, H)
    y1c = _clamp(int(y1), 0, H)
    x0c = _clamp(int(x0), 0, W)
    x1c = _clamp(int(x1), 0, W)
    if y1c <= y0c or x1c <= x0c:
        return 0
    patch = mask_2d[y0c:y1c, x0c:x1c]
    return int(patch.sum())


def _count_roi_pixels(roi_2d, y0, x0, y1, x1, H, W):
    """ROI 내 pixel 수 반환."""
    y0c = _clamp(int(y0), 0, H)
    y1c = _clamp(int(y1), 0, H)
    x0c = _clamp(int(x0), 0, W)
    x1c = _clamp(int(x1), 0, W)
    if y1c <= y0c or x1c <= x0c:
        return 0
    patch = roi_2d[y0c:y1c, x0c:x1c]
    return int(patch.sum())


def _make_invalid_row(row, reason="unknown"):
    return {
        "crop_id": row.get("crop_id", None),
        "patient_id": row.get("patient_id", None),
        "safe_id": row.get("_safe_id", None),
        "qa_group": row.get("qa_group", None),
        "qa_priority": row.get("qa_priority", None),
        "z_center": row.get("z_center", None),
        "y0_patch": None, "x0_patch": None, "y1_patch": None, "x1_patch": None,
        "y0_fixed96": None, "x0_fixed96": None, "y1_fixed96": None, "x1_fixed96": None,
        "y_context_start": row.get("y_context_start", None),
        "y_context_end": row.get("y_context_end", None),
        "x_context_start": row.get("x_context_start", None),
        "x_context_end": row.get("x_context_end", None),
        "lesion_pixels_patch": None,
        "lesion_pixels_fixed96": None,
        "lesion_pixels_context192": None,
        "roi_pixels_patch": None,
        "roi_pixels_fixed96": None,
        "roi_pixels_context192": None,
        "has_lesion_overlap_patch": None,
        "has_lesion_overlap_fixed96": None,
        "has_lesion_overlap_context192": None,
        "lesion_overlap_class": "invalid",
        "invalid_reason": reason,
        "roi_ratio_context192": None,
    }


def _process_row(row, mask_arr, roi_arr, Z, H, W):
    z = int(row["z_center"])
    if z < 0 or z >= Z:
        return _make_invalid_row(row, reason=f"z_out_of_range:{z}")

    mask_2d = mask_arr[z]  # (H, W) uint8
    roi_2d = roi_arr[z] if roi_arr is not None else None

    # patch bbox (raw bbox: y0/x0/y1/x1 from crop manifest)
    py0 = row.get("y0", None)
    px0 = row.get("x0", None)
    py1 = row.get("y1", None)
    px1 = row.get("x1", None)

    # fixed96 bbox
    fy0 = row.get("y0_fixed_crop", None)
    fx0 = row.get("x0_fixed_crop", None)
    fy1 = row.get("y1_fixed_crop", None)
    fx1 = row.get("x1_fixed_crop", None)

    # context192 bbox
    cy0 = row.get("y_context_start", None)
    cy1 = row.get("y_context_end", None)
    cx0 = row.get("x_context_start", None)
    cx1 = row.get("x_context_end", None)

    def is_invalid_coord(v):
        return v is None or (isinstance(v, float) and np.isnan(v))

    def safe_count(y0_, x0_, y1_, x1_):
        if any(is_invalid_coord(v) for v in [y0_, x0_, y1_, x1_]):
            return None
        return _count_lesion_pixels(mask_2d, y0_, x0_, y1_, x1_, H, W)

    def safe_roi_count(y0_, x0_, y1_, x1_):
        if roi_2d is None:
            return None
        if any(is_invalid_coord(v) for v in [y0_, x0_, y1_, x1_]):
            return None
        return _count_roi_pixels(roi_2d, y0_, x0_, y1_, x1_, H, W)

    lp_patch = safe_count(py0, px0, py1, px1)
    lp_fixed96 = safe_count(fy0, fx0, fy1, fx1)
    lp_ctx = safe_count(cy0, cx0, cy1, cx1)

    rp_patch = safe_roi_count(py0, px0, py1, px1)
    rp_fixed96 = safe_roi_count(fy0, fx0, fy1, fx1)
    rp_ctx = safe_roi_count(cy0, cx0, cy1, cx1)

    has_patch = bool(lp_patch > 0) if lp_patch is not None else None
    has_fixed96 = bool(lp_fixed96 > 0) if lp_fixed96 is not None else None
    has_ctx = bool(lp_ctx > 0) if lp_ctx is not None else None

    # roi_ratio (lesion_pixels_context192 / roi_pixels_context192)
    if lp_ctx is not None and rp_ctx is not None:
        roi_ratio = (float(lp_ctx) / float(rp_ctx)) if rp_ctx > 0 else 0.0
    else:
        roi_ratio = None

    # overlap class 분류
    if has_patch is None and has_fixed96 is None and has_ctx is None:
        overlap_class = "invalid"
        invalid_reason = "all_bbox_none"
    elif has_patch:
        overlap_class = "patch_overlap"
        invalid_reason = ""
    elif has_fixed96:
        overlap_class = "fixed96_overlap_only"
        invalid_reason = ""
    elif has_ctx:
        overlap_class = "context192_overlap_only"
        invalid_reason = ""
    else:
        overlap_class = "no_lesion_overlap"
        invalid_reason = ""

    return {
        "crop_id": row["crop_id"],
        "patient_id": row.get("patient_id", None),
        "safe_id": row.get("_safe_id", None),
        "qa_group": row.get("qa_group", None),
        "qa_priority": row.get("qa_priority", None),
        "z_center": z,
        "y0_patch": py0, "x0_patch": px0, "y1_patch": py1, "x1_patch": px1,
        "y0_fixed96": fy0, "x0_fixed96": fx0, "y1_fixed96": fy1, "x1_fixed96": fx1,
        "y_context_start": cy0, "y_context_end": cy1,
        "x_context_start": cx0, "x_context_end": cx1,
        "lesion_pixels_patch": lp_patch,
        "lesion_pixels_fixed96": lp_fixed96,
        "lesion_pixels_context192": lp_ctx,
        "roi_pixels_patch": rp_patch,
        "roi_pixels_fixed96": rp_fixed96,
        "roi_pixels_context192": rp_ctx,
        "has_lesion_overlap_patch": has_patch,
        "has_lesion_overlap_fixed96": has_fixed96,
        "has_lesion_overlap_context192": has_ctx,
        "lesion_overlap_class": overlap_class,
        "invalid_reason": invalid_reason,
        "roi_ratio_context192": roi_ratio,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. 입력 파일 로드
    print("[1] 입력 파일 로드...")
    ctx_df = pd.read_csv(CTX_INDEX_PATH)
    crop_df = pd.read_csv(CROP_MANIFEST_PATH)
    qa_df = pd.read_csv(QA_MANIFEST_PATH)

    print(f"  context overlay rows: {len(ctx_df)}")
    print(f"  crop manifest rows:   {len(crop_df)}")
    print(f"  QA manifest rows:     {len(qa_df)}")

    # 2. crop_id 중복 확인
    print("[2] crop_id 중복 확인...")
    dup_count = ctx_df.duplicated(subset=["crop_id"]).sum()
    if dup_count > 0:
        print(f"  [ERROR] context overlay index에 crop_id 중복 {dup_count}건. 종료.")
        raise SystemExit(1)
    print(f"  crop_id 중복 없음 (unique={ctx_df['crop_id'].nunique()})")

    # 3. crop manifest join (index=crop_id)
    print("[3] crop manifest join (index=crop_id)...")
    crop_cols = [
        "candidate_id", "safe_id", "z_center",
        "y0", "x0", "y1", "x1",
        "y0_fixed_crop", "x0_fixed_crop", "y1_fixed_crop", "x1_fixed_crop",
    ]
    crop_sub = crop_df[crop_cols].copy()
    # crop_df.index = 0,1,...,7699 → crop_id와 일치함 (확인됨)
    ctx_crop_ids = ctx_df["crop_id"].tolist()
    missing_ids = [cid for cid in ctx_crop_ids if cid not in crop_df.index]
    if missing_ids:
        print(f"  [WARNING] crop manifest에 없는 crop_id {len(missing_ids)}건: {missing_ids[:5]}")

    # ctx_df에 crop manifest 컬럼 병합
    ctx_indexed = ctx_df.set_index("crop_id")
    crop_sub_renamed = crop_sub.rename(columns={
        "safe_id": "safe_id_crop",
        "z_center": "z_center_crop",
    })
    merged = ctx_indexed.join(crop_sub_renamed, how="left")
    merged = merged.reset_index()  # crop_id 다시 컬럼으로

    # 4. safe_id, patient_id 확정 (QA manifest 우선)
    print("[4] safe_id / patient_id 확정...")
    qa_safeid_map = qa_df.set_index("crop_id")["safe_id"].to_dict()
    qa_pid_map = qa_df.set_index("crop_id")["patient_id"].to_dict()
    merged["_safe_id"] = merged["crop_id"].map(qa_safeid_map)
    merged["patient_id"] = merged["crop_id"].map(qa_pid_map)

    nan_safe = merged["_safe_id"].isna().sum()
    print(f"  safe_id NaN 수: {nan_safe}")

    # 5. volume source 폴더 목록 구축
    print("[5] volume source 폴더 목록 구축...")
    vol_folders = {
        f: (VOL_ROOT / f)
        for f in os.listdir(VOL_ROOT)
        if (VOL_ROOT / f).is_dir()
    }

    def find_vol_dir(safe_id_str):
        if safe_id_str in vol_folders:
            return vol_folders[safe_id_str]
        for fname, fpath in vol_folders.items():
            if fname.startswith(safe_id_str) or safe_id_str in fname:
                return fpath
        return None

    # 6. 환자별 처리
    print("[6] 환자별 lesion mask overlap 계산 시작...")
    results = []

    valid_patients = merged["_safe_id"].dropna().unique()
    print(f"  유효 safe_id 환자 수: {len(valid_patients)}")

    for pat_safe_id in valid_patients:
        pat_rows = merged[merged["_safe_id"] == pat_safe_id].copy()

        vol_dir = find_vol_dir(str(pat_safe_id))
        mask_path = (vol_dir / "lesion_mask_roi_0_0.npy") if vol_dir else None
        roi_path = (vol_dir / "roi_0_0.npy") if vol_dir else None

        mask_missing = (
            vol_dir is None
            or mask_path is None
            or not mask_path.exists()
        )

        if mask_missing:
            print(f"  [WARNING] mask 없음: {pat_safe_id}")
            for _, row in pat_rows.iterrows():
                results.append(_make_invalid_row(row, reason="mask_not_found"))
            continue

        # 환자별 1회 로드
        try:
            mask_arr = np.load(str(mask_path), mmap_mode="r")
            if roi_path is not None and roi_path.exists():
                roi_arr = np.load(str(roi_path), mmap_mode="r")
            else:
                roi_arr = None
        except Exception as e:
            print(f"  [WARNING] 로드 실패: {pat_safe_id} - {e}")
            for _, row in pat_rows.iterrows():
                results.append(_make_invalid_row(row, reason=f"load_error:{e}"))
            continue

        Z, H, W = mask_arr.shape

        for _, row in pat_rows.iterrows():
            try:
                result = _process_row(row, mask_arr, roi_arr, Z, H, W)
            except Exception as e:
                result = _make_invalid_row(row, reason=f"process_error:{e}")
            results.append(result)

        del mask_arr
        if roi_arr is not None:
            del roi_arr
        gc.collect()

    # safe_id NaN인 행 처리
    nan_rows = merged[merged["_safe_id"].isna()]
    for _, row in nan_rows.iterrows():
        results.append(_make_invalid_row(row, reason="safe_id_missing"))

    print(f"  처리 완료: {len(results)}건")

    # 7. 결과 CSV 저장
    print("[7] 결과 CSV 저장...")
    result_df = pd.DataFrame(results)
    result_df.to_csv(OUT_CSV, index=False)
    print(f"  저장: {OUT_CSV}")
    print(f"  결과 행 수: {len(result_df)}")

    # 8. overlap class 분포 출력
    print("[8] overlap class 분포:")
    class_counts = result_df["lesion_overlap_class"].value_counts().to_dict()
    for k, v in class_counts.items():
        print(f"  {k}: {v}")

    # 9. summary JSON 생성
    print("[9] summary JSON 생성...")

    roi_ratios = result_df["roi_ratio_context192"].dropna()

    # qa_group별 overlap class 분포
    qa_group_counts = {}
    for grp, sub in result_df.groupby("qa_group", dropna=False):
        qa_group_counts[str(grp)] = sub["lesion_overlap_class"].value_counts().to_dict()

    # qa_priority별 overlap class 분포
    qa_priority_counts = {}
    for pri, sub in result_df.groupby("qa_priority", dropna=False):
        qa_priority_counts[str(pri)] = sub["lesion_overlap_class"].value_counts().to_dict()

    # 특정 계열별 overlap 여부
    def group_overlap_flag(df_, group_key_substr):
        sub = df_[df_["qa_group"].str.contains(group_key_substr, na=False)]
        if len(sub) == 0:
            return {"count": 0, "has_any_overlap": False, "overlap_count": 0}
        has_any = (
            (sub["lesion_overlap_class"] != "no_lesion_overlap")
            & (sub["lesion_overlap_class"] != "invalid")
        )
        return {
            "count": int(len(sub)),
            "has_any_overlap": bool(has_any.any()),
            "overlap_count": int(has_any.sum()),
        }

    summary = {
        "script": "tmp_lesion_overlap_analysis_phase545.py",
        "analysis_tag": "hard_negative_top_score_lesion_overlap_analysis_v1",
        "total_rows": int(len(result_df)),
        "patient_count": int(result_df["patient_id"].nunique()),
        "overlap_class_counts": {k: int(v) for k, v in class_counts.items()},
        "qa_group_overlap_counts": qa_group_counts,
        "qa_priority_overlap_counts": qa_priority_counts,
        "roi_ratio_context192_stats": {
            "mean": float(roi_ratios.mean()) if len(roi_ratios) > 0 else None,
            "min": float(roi_ratios.min()) if len(roi_ratios) > 0 else None,
            "max": float(roi_ratios.max()) if len(roi_ratios) > 0 else None,
            "count_non_null": int(len(roi_ratios)),
        },
        "group_HN_p95": group_overlap_flag(result_df, "HN-p95"),
        "group_HN_large_box": group_overlap_flag(result_df, "HN-large-bbox"),
        "group_HN_padim_high_rd4ad_low": group_overlap_flag(
            result_df, "HN-padim-high-rd4ad-low"
        ),
        "input_files": {
            "qa_manifest": str(QA_MANIFEST_PATH),
            "crop_manifest": str(CROP_MANIFEST_PATH),
            "context_overlay_index": str(CTX_INDEX_PATH),
        },
        "output_files": {
            "analysis_csv": str(OUT_CSV),
            "summary_json": str(OUT_JSON),
        },
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  저장: {OUT_JSON}")

    print("[완료]")
    print(f"  분석 row 수: {len(result_df)}")
    print(f"  환자 수: {result_df['patient_id'].nunique()}")
    for k, v in class_counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
