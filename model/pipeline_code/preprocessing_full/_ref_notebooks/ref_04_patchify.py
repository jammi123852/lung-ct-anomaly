# ============================================================
# Patch extraction pipeline - resume / memory safe version
# ------------------------------------------------------------
# 목적:
#   - 전처리 완료 결과에서 patch 추출
#   - 환자별 patch CSV로 바로 저장
#   - 이미 처리된 환자는 자동 skip
#   - 전체 patch를 메모리에 한 번에 쌓지 않음
#   - MemoryError 방지
# ============================================================

from pathlib import Path
import json
import gc
import time
from collections import Counter

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt


# ============================================================
# 1. CONFIG
# ============================================================

PATCH_CONFIG = {
    "preprocess_root": r"E:\jyp\ct_data_2d_preprocessed\NSCLC_MSD_preprocessed_roi0_0_ts_lung_raw_no_dilate_usable_only_v1",

    "patch_out_root": r"E:\jyp\ct_data_2d_preprocessed\NSCLC_MSD_patchsplit_roi0_0_ts_lung_raw_no_dilate_usable_only_v1",

    "patch_size": 32,
    "patch_stride": 16,

    "min_slice_roi_ratio": 0.005,
    "min_roi_pixels_per_slice": 50,
    "min_patch_roi_ratio": 0.50,

    "lesion_patch_ratio_threshold": 0.001,
    "central_distance_ratio_threshold": 0.50,

    "hist_bins": [
        -1000, -900, -800, -700, -600, -500, -400,
        -300, -200, -100, 0, 100, 200, 400
    ],

    "skip_existing_patient_csv": True,
    "feature_use_roi_pixels_only": True,
}

PRE_ROOT = Path(PATCH_CONFIG["preprocess_root"])
PATCH_OUT = Path(PATCH_CONFIG["patch_out_root"])
PATCH_BY_PATIENT_DIR = PATCH_OUT / "patches_by_patient"
PATCH_DONE_DIR = PATCH_OUT / "done_markers"

PATCH_OUT.mkdir(parents=True, exist_ok=True)
PATCH_BY_PATIENT_DIR.mkdir(parents=True, exist_ok=True)
PATCH_DONE_DIR.mkdir(parents=True, exist_ok=True)

with open(PATCH_OUT / "patch_config.json", "w", encoding="utf-8") as f:
    json.dump(PATCH_CONFIG, f, indent=2, ensure_ascii=False)

print("========== PATCH CONFIG ==========")
print("PRE_ROOT:", PRE_ROOT)
print("PATCH_OUT:", PATCH_OUT)
print("PATCH_BY_PATIENT_DIR:", PATCH_BY_PATIENT_DIR)


# ============================================================
# 2. helper functions
# ============================================================

def format_seconds(seconds: float) -> str:
    seconds = int(seconds)

    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h}시간 {m}분 {s}초"

    if m > 0:
        return f"{m}분 {s}초"

    return f"{s}초"


def read_nii_array(path: Path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return img, arr


def safe_patient_csv_name(patient_id: str) -> str:
    return (
        str(patient_id)
        .replace("\\", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
    )


def calc_patch_features(patch_hu, hist_bins):
    """
    patch HU 값으로 간단 feature 계산.
    patch_hu는 2D patch이거나, pure_lung 픽셀만 골라낸 1D 배열일 수 있음.
    """

    x = patch_hu.reshape(-1)

    if x.size == 0:
        feat = {
            "hu_mean": np.nan,
            "hu_std": np.nan,
            "hu_min": np.nan,
            "hu_max": np.nan,
            "hu_p05": np.nan,
            "hu_p25": np.nan,
            "hu_p50": np.nan,
            "hu_p75": np.nan,
            "hu_p95": np.nan,
        }

        for i in range(len(hist_bins) - 1):
            feat[f"hist_bin_{i:02d}"] = np.nan

        return feat

    x = x.astype(np.float32, copy=False)

    hist, _ = np.histogram(x, bins=hist_bins)
    hist = hist.astype(np.float32)

    hist_sum = hist.sum()

    if hist_sum > 0:
        hist = hist / hist_sum

    feat = {
        "hu_mean": float(np.mean(x)),
        "hu_std": float(np.std(x)),
        "hu_min": float(np.min(x)),
        "hu_max": float(np.max(x)),
        "hu_p05": float(np.percentile(x, 5)),
        "hu_p25": float(np.percentile(x, 25)),
        "hu_p50": float(np.percentile(x, 50)),
        "hu_p75": float(np.percentile(x, 75)),
        "hu_p95": float(np.percentile(x, 95)),
    }

    for i, v in enumerate(hist):
        feat[f"hist_bin_{i:02d}"] = float(v)

    return feat


def get_z_level(z, valid_z_min, valid_z_max):
    """
    lung range 안에서 z 위치를 lower / middle / upper로 나눔.
    """

    if valid_z_max <= valid_z_min:
        return "middle", 0.5

    z_ratio = (z - valid_z_min) / float(valid_z_max - valid_z_min)
    z_ratio = float(np.clip(z_ratio, 0.0, 1.0))

    if z_ratio < 1.0 / 3.0:
        return "lower", z_ratio

    if z_ratio < 2.0 / 3.0:
        return "middle", z_ratio

    return "upper", z_ratio


def get_central_peripheral(dist_patch_values, threshold=0.5):
    """
    patch 안 pure_lung 픽셀들의 distance ratio 평균으로 central/peripheral 결정.
    """

    if len(dist_patch_values) == 0:
        return "peripheral", 0.0

    mean_ratio = float(np.mean(dist_patch_values))

    if mean_ratio >= threshold:
        return "central", mean_ratio

    return "peripheral", mean_ratio

def get_lesion_zone_type_simple(
    lesion_ratio,
    lesion_patch_ratio_threshold,
):
    if lesion_ratio >= lesion_patch_ratio_threshold:
        return "lesion_in_roi_patch"

    return "no_lesion"



def get_left_right_by_patch_center(x0, patch_size, width):
    """
    좌우 정보는 일단 metadata로만 저장.
    image_left / image_right는 영상 좌표 기준.
    """

    cx = x0 + patch_size / 2.0

    if cx < width / 2.0:
        return "image_left"

    return "image_right"


# ============================================================
# 3. one patient patch extraction
# ============================================================

def extract_patches_one_patient(group_name, patient_id, patient_df, config):
    patch_size = int(config["patch_size"])
    patch_stride = int(config["patch_stride"])
    patch_area = patch_size * patch_size

    row0 = patient_df.iloc[0]

    ct_path = Path(row0["ct_1mm_lung_range_nii"])
    roi_path = Path(row0["roi_0_0_lung_range_nii"])
    lesion_path = Path(row0["lesion_mask_roi_0_0_lung_range_nii"])

    if not ct_path.exists():
        raise FileNotFoundError(f"CT lung range 없음: {ct_path}")

    if not roi_path.exists():
        raise FileNotFoundError(f"roi_0_0 lung range 없음: {roi_path}")
    if not lesion_path.exists():
        raise FileNotFoundError(f"lesion_mask_roi_0_0 lung range 없음: {lesion_path}")

    _, ct_arr = read_nii_array(ct_path)
    _, roi_arr = read_nii_array(roi_path)
    _, lesion_arr = read_nii_array(lesion_path)

    roi_arr = roi_arr > 0
    lesion_arr = lesion_arr > 0


    if ct_arr.shape != roi_arr.shape:
        raise RuntimeError(f"{group_name}/{patient_id}: CT와 roi_0_0 shape 다름")
    
    if ct_arr.shape != lesion_arr.shape:
        raise RuntimeError(f"{group_name}/{patient_id}: CT와 lesion_mask_roi_0_0 shape 다름")

    zdim, h, w = ct_arr.shape

    if "local_z" in patient_df.columns:
        patient_df = patient_df.sort_values("local_z").reset_index(drop=True)
    else:
        patient_df = patient_df.sort_values("slice_index").reset_index(drop=True)

    if len(patient_df) != zdim:
        print(
            f"[WARN] {group_name}/{patient_id}: metadata row 수와 CT zdim 다름 "
            f"metadata={len(patient_df)}, zdim={zdim}. local_z 기준으로 진행."
        )

    roi_area_per_slice = roi_arr.sum(axis=(1, 2))
    roi_ratio_per_slice = roi_area_per_slice / float(h * w)

    valid_z_indices = np.where(
        (roi_ratio_per_slice >= float(config["min_slice_roi_ratio"]))
        & (roi_area_per_slice >= int(config["min_roi_pixels_per_slice"]))
    )[0]

    if len(valid_z_indices) == 0:
        summary = {
            "group": group_name,
            "patient_id": patient_id,
            "status": "no_valid_z",
            "zdim": int(zdim),
            "height": int(h),
            "width": int(w),
            "valid_z_min": "",
            "valid_z_max": "",
            "valid_z_count": 0,
            "candidate_patch_count_before_filter": 0,
            "selected_patch_count": 0,
            "lesion_patch_count": 0,
        }

        del ct_arr, roi_arr, lesion_arr
        gc.collect()

        return [], summary

    valid_z_min = int(valid_z_indices.min())
    valid_z_max = int(valid_z_indices.max())

    patch_rows = []
    patient_candidate_count = 0
    patient_selected_count = 0
    patient_lesion_patch_count = 0

    for z in valid_z_indices:
        z = int(z)
        ct_slice = ct_arr[z]
        roi_slice = roi_arr[z]
        lesion_slice = lesion_arr[z]

        dist_map = distance_transform_edt(roi_slice)
        max_dist = float(dist_map.max())

        if max_dist <= 0:
            continue

        dist_ratio_map = dist_map / (max_dist + 1e-6)

        z_level, z_ratio = get_z_level(
            z=z,
            valid_z_min=valid_z_min,
            valid_z_max=valid_z_max,
        )

        slice_index_value = int(z)
        is_original_slice_value = -1

        if z < len(patient_df):
            row_z = patient_df.iloc[z]

            if "slice_index" in row_z.index:
                slice_index_value = int(row_z["slice_index"])

            if "is_original_slice" in row_z.index:
                is_original_slice_value = int(row_z["is_original_slice"])

        for y0 in range(0, h - patch_size + 1, patch_stride):
            for x0 in range(0, w - patch_size + 1, patch_stride):
                patient_candidate_count += 1

                y1 = y0 + patch_size
                x1 = x0 + patch_size

                roi_patch = roi_slice[y0:y1, x0:x1]
                lesion_patch = lesion_slice[y0:y1, x0:x1]

                roi_pixels = int(roi_patch.sum())
                lesion_pixels = int(lesion_patch.sum())

                roi_ratio = roi_pixels / float(patch_area)
                lesion_ratio = lesion_pixels / float(patch_area)

                # 테스트용 patch 선택 기준:
                # 1) model_roi가 충분히 포함된 patch는 저장
                # 2) model_roi 비율이 낮아도 lesion이 있으면 저장
                keep_by_roi = (
                    roi_ratio >= float(config["min_patch_roi_ratio"])
                )
                
                if not keep_by_roi:
                    continue

                ct_patch = ct_slice[y0:y1, x0:x1]

                dist_patch = dist_ratio_map[y0:y1, x0:x1]
                dist_values_inside_roi = dist_patch[roi_patch > 0]

                central_bin, central_ratio_mean = get_central_peripheral(
                    dist_patch_values=dist_values_inside_roi,
                    threshold=float(config["central_distance_ratio_threshold"]),
                )

                position_bin = f"{z_level}_{central_bin}"

                left_right_bin = get_left_right_by_patch_center(
                    x0=x0,
                    patch_size=patch_size,
                    width=w,
                )

                if bool(config["feature_use_roi_pixels_only"]):
                    feature_pixels = ct_patch[roi_patch > 0]
                else:
                    feature_pixels = ct_patch

                features = calc_patch_features(
                    patch_hu=feature_pixels,
                    hist_bins=config["hist_bins"],
                )

                has_lesion_patch = int(
                    lesion_ratio >= float(config["lesion_patch_ratio_threshold"])
                )


                lesion_zone_type = get_lesion_zone_type_simple(
                    lesion_ratio=lesion_ratio,
                    lesion_patch_ratio_threshold=float(config["lesion_patch_ratio_threshold"]),
                )

                if has_lesion_patch == 1:
                    patient_lesion_patch_count += 1

                row = {
                    "group": group_name,
                    "patient_id": patient_id,

                    "local_z": int(z),
                    "slice_index": int(slice_index_value),
                    "is_original_slice": int(is_original_slice_value),

                    "y0": int(y0),
                    "x0": int(x0),
                    "y1": int(y1),
                    "x1": int(x1),

                    "patch_size": int(patch_size),
                    "patch_stride": int(patch_stride),

                    "roi_0_0_pixels": int(roi_pixels),
                    "lesion_pixels": int(lesion_pixels),

                    "roi_0_0_patch_ratio": float(roi_ratio),
                    "lesion_patch_ratio": float(lesion_ratio),
                    "slice_roi_0_0_ratio": float(roi_ratio_per_slice[z]),

                    "has_lesion_patch": int(has_lesion_patch),
                    "lesion_zone_type": lesion_zone_type,
                    
                    "z_level": z_level,
                    "z_ratio": float(z_ratio),
                    "central_peripheral": central_bin,
                    "central_distance_ratio_mean": float(central_ratio_mean),
                    "position_bin": position_bin,
                    "left_right_metadata": left_right_bin,
                    
                    "ct_1mm_lung_range_nii": str(ct_path),
                    "roi_0_0_lung_range_nii": str(roi_path),
                    "lesion_mask_roi_0_0_lung_range_nii": str(lesion_path),
                }

                row.update(features)

                patch_rows.append(row)
                patient_selected_count += 1

    summary = {
        "group": group_name,
        "patient_id": patient_id,
        "status": "success",
        "zdim": int(zdim),
        "height": int(h),
        "width": int(w),
        "valid_z_min": int(valid_z_min),
        "valid_z_max": int(valid_z_max),
        "valid_z_count": int(len(valid_z_indices)),
        "candidate_patch_count_before_filter": int(patient_candidate_count),
        "selected_patch_count": int(patient_selected_count),
        "lesion_patch_count": int(patient_lesion_patch_count),
    }

    del ct_arr, roi_arr, lesion_arr
    gc.collect()

    return patch_rows, summary

# ============================================================
# 4. run patch extraction with resume
# ============================================================

metadata_path = PRE_ROOT / "metadata_slices.csv"

if not metadata_path.exists():
    raise FileNotFoundError(f"metadata_slices.csv 없음: {metadata_path}")

slice_df = pd.read_csv(metadata_path)

required_cols = [
    "group",
    "patient_id",
    "ct_1mm_lung_range_nii",
    "roi_0_0_lung_range_nii",
    "lesion_mask_roi_0_0_lung_range_nii",
]

for c in required_cols:
    if c not in slice_df.columns:
        raise RuntimeError(f"metadata_slices.csv에 필요한 컬럼 없음: {c}")

patient_keys = (
    slice_df[["group", "patient_id"]]
    .drop_duplicates()
    .sort_values(["group", "patient_id"])
    .reset_index(drop=True)
)

print("========== PATCH EXTRACTION START ==========")
print("patient count:", len(patient_keys))
print("PATCH_OUT:", PATCH_OUT)

summary_rows = []
error_rows = []

global_start = time.perf_counter()
total_cases = len(patient_keys)

for case_idx, key_row in enumerate(
    tqdm(
        patient_keys.to_dict("records"),
        desc="Patch extraction by patient",
        total=total_cases,
        ncols=100,
        ascii=True,
        leave=True,
        dynamic_ncols=False,
    ),
    start=1
):
    group_name = str(key_row["group"])
    patient_id = str(key_row["patient_id"])
    start = time.perf_counter()

    patient_safe = safe_patient_csv_name(f"{group_name}_{patient_id}")
    patient_csv = PATCH_BY_PATIENT_DIR / f"{patient_safe}.csv"
    patient_tmp_csv = PATCH_BY_PATIENT_DIR / f"{patient_safe}.tmp.csv"
    done_marker = PATCH_DONE_DIR / f"{patient_safe}.done"

    if (
        bool(PATCH_CONFIG["skip_existing_patient_csv"])
        and patient_csv.exists()
        and done_marker.exists()
    ):
        elapsed = time.perf_counter() - start

        summary_rows.append({
            "group": group_name,
            "patient_id": patient_id,
            "status": "skipped_existing",
            "patient_csv": str(patient_csv),
            "elapsed_seconds": round(elapsed, 2),
            "elapsed_readable": format_seconds(elapsed),
        })

        continue

    try:
        pdf = slice_df[
            (slice_df["group"].astype(str) == group_name)
            & (slice_df["patient_id"].astype(str) == patient_id)
        ].copy()
        
        if len(pdf) == 0:
            raise RuntimeError(f"{group_name}/{patient_id}: metadata_slices.csv에 환자 row 없음")
        
        if "local_z" in pdf.columns:
            pdf = pdf.sort_values("local_z").reset_index(drop=True)
        else:
            pdf = pdf.sort_values("slice_index").reset_index(drop=True)
        
        patch_rows, summary = extract_patches_one_patient(
            group_name=group_name,
            patient_id=patient_id,
            patient_df=pdf,
            config=PATCH_CONFIG,
        )

        patient_df = pd.DataFrame(patch_rows)

        # tmp 파일로 먼저 저장하고, 성공하면 최종 파일명으로 교체
        patient_df.to_csv(patient_tmp_csv, index=False, encoding="utf-8-sig")
        patient_tmp_csv.replace(patient_csv)

        done_marker.write_text("done", encoding="utf-8")

        elapsed = time.perf_counter() - start
        total_elapsed = time.perf_counter() - global_start
        avg_per_case = total_elapsed / case_idx
        remain = total_cases - case_idx
        eta = avg_per_case * remain

        summary.update({
            "patient_csv": str(patient_csv),
            "elapsed_seconds": round(elapsed, 2),
            "elapsed_readable": format_seconds(elapsed),
            "total_elapsed_seconds": round(total_elapsed, 2),
            "total_elapsed_readable": format_seconds(total_elapsed),
            "estimated_remaining_seconds": round(eta, 2),
            "estimated_remaining_readable": format_seconds(eta),
        })

        summary_rows.append(summary)

        pd.DataFrame(summary_rows).to_csv(
            PATCH_OUT / "patch_patient_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"[PATCH] {patient_id} 완료 "
            f"({case_idx}/{total_cases}) | "
            f"patch={len(patient_df)} | "
            f"이번 환자: {format_seconds(elapsed)} | "
            f"전체 경과: {format_seconds(total_elapsed)} | "
            f"예상 남은 시간: {format_seconds(eta)}"
        )

        del patch_rows, patient_df
        gc.collect()

    except Exception as e:
        elapsed = time.perf_counter() - start

        error_rows.append({
            "group": group_name,
            "patient_id": patient_id,
            "error": str(e),
            "elapsed_seconds": round(elapsed, 2),
            "elapsed_readable": format_seconds(elapsed),
        })

        pd.DataFrame(error_rows).to_csv(
            PATCH_OUT / "patch_errors.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print("[ERROR]", patient_id, str(e))
        gc.collect()


summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(
    PATCH_OUT / "patch_patient_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

if len(error_rows) > 0:
    pd.DataFrame(error_rows).to_csv(
        PATCH_OUT / "patch_errors.csv",
        index=False,
        encoding="utf-8-sig",
    )

print("\n========== PATCH EXTRACTION FINISHED ==========")
print("PATCH_OUT:", PATCH_OUT)
print("patient csv dir:", PATCH_BY_PATIENT_DIR)
print("summary:", PATCH_OUT / "patch_patient_summary.csv")
print("errors:", len(error_rows))


# ============================================================
# 5. position_bin count summary
# ------------------------------------------------------------
# 전체 patch를 한 번에 메모리에 올리지 않고,
# 환자별 CSV를 하나씩 읽어서 position count만 집계.
# ============================================================

position_counter = Counter()
patient_patch_counts = []

patient_csv_files = sorted(PATCH_BY_PATIENT_DIR.glob("*.csv"))

for csv_path in tqdm(
    patient_csv_files,
    desc="Summarize patch CSVs",
    total=len(patient_csv_files),
    ncols=100,
    ascii=True,
    leave=True,
    dynamic_ncols=False,
):
    if csv_path.name.endswith(".tmp.csv"):
        continue

    try:
        df = pd.read_csv(
            csv_path,
            usecols=[
                "group",
                "patient_id",
                "position_bin",
                "lesion_zone_type",
                "has_lesion_patch",
            ]
        )
    except (pd.errors.EmptyDataError, ValueError):
        continue

    if len(df) == 0:
        continue

    group_name = df["group"].iloc[0]
    patient_id = df["patient_id"].iloc[0]
    
    counts = df["position_bin"].value_counts()
    
    for k, v in counts.items():
        position_counter[k] += int(v)
    
    lesion_zone_counts = df["lesion_zone_type"].value_counts().to_dict()

    patient_patch_counts.append({
        "group": group_name,
        "patient_id": patient_id,
        "patch_count": int(len(df)),
        "lesion_patch_count": int(df["has_lesion_patch"].sum()),
        "no_lesion_patch_count": int(lesion_zone_counts.get("no_lesion", 0)),
        "lesion_in_roi_patch_count": int(lesion_zone_counts.get("lesion_in_roi_patch", 0)),
    })

position_df = pd.DataFrame([
    {"position_bin": k, "patch_count": v}
    for k, v in sorted(position_counter.items())
])

patient_count_df = pd.DataFrame(patient_patch_counts)

position_df.to_csv(
    PATCH_OUT / "patch_position_counts.csv",
    index=False,
    encoding="utf-8-sig",
)

patient_count_df.to_csv(
    PATCH_OUT / "patch_count_by_patient.csv",
    index=False,
    encoding="utf-8-sig",
)


lesion_zone_counter = Counter()

for csv_path in patient_csv_files:
    if csv_path.name.endswith(".tmp.csv"):
        continue

    try:
        df_zone = pd.read_csv(csv_path, usecols=["lesion_zone_type"])
    except (pd.errors.EmptyDataError, ValueError):
        continue

    if len(df_zone) == 0:
        continue

    for k, v in df_zone["lesion_zone_type"].value_counts().items():
        lesion_zone_counter[k] += int(v)

lesion_zone_df = pd.DataFrame([
    {"lesion_zone_type": k, "patch_count": v}
    for k, v in sorted(lesion_zone_counter.items())
])

lesion_zone_df.to_csv(
    PATCH_OUT / "patch_lesion_zone_counts.csv",
    index=False,
    encoding="utf-8-sig",
)
print("\n========== Position bin counts ==========")
display(position_df)

print("\n========== Patch count by patient summary ==========")

if len(patient_count_df) > 0:
    display(patient_count_df["patch_count"].describe())
else:
    print("patient_count_df 비어 있음")

print("\nSaved:")
print(PATCH_OUT / "patch_position_counts.csv")
print(PATCH_OUT / "patch_count_by_patient.csv")
print(PATCH_OUT / "patch_lesion_zone_counts.csv")

# %%==== CELL ====

