# ============================================================
# Build 0/0 ROI preprocess folder before patch extraction
# ------------------------------------------------------------
# 목적:
#   - 기존 usable-only 전처리 결과에서 읽기
#   - TotalSegmentator 폐엽 5개를 그대로 합쳐 0/0 ROI 생성
#   - 추가 dilation 없음
#   - lung_range 기준 roi_0_0 마스크 저장
#   - lesion_mask를 roi_0_0 안쪽으로 자른 lesion_mask_roi_0_0 저장
#   - 패치분할 코드가 읽을 새 metadata_slices.csv 생성
#   - 환자별 병변 포함률 CSV 저장
#
# 기존 파일 삭제/수정 없음.
# ============================================================

from pathlib import Path
import json
import gc

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

try:
    from scipy.ndimage import binary_dilation
except Exception as e:
    raise ImportError("scipy가 필요합니다. scipy 설치 상태를 확인하세요.") from e


# ============================================================
# 1. CONFIG
# ============================================================

SOURCE_PREPROCESS_ROOT = Path(
    r"E:\jyp\ct_data_2d_preprocessed\NSCLC_MSD_model_roi_final_usable_only_v1"
)

ROI_VERSION_NAME = "roi0_0_ts_lung_raw_no_dilate_usable_only_v1"

ROI_PREPROCESS_ROOT = SOURCE_PREPROCESS_ROOT.parent / (
    f"NSCLC_MSD_preprocessed_{ROI_VERSION_NAME}"
)

QA_OUT_DIR = SOURCE_PREPROCESS_ROOT.parent / (
    f"qa_before_patchsplit_{ROI_VERSION_NAME}"
)

ROI_LUNG_RANGE_DIR = ROI_PREPROCESS_ROOT / "roi_0_0_lung_range"
LESION_ROI_LUNG_RANGE_DIR = ROI_PREPROCESS_ROOT / "lesion_mask_roi_0_0_lung_range"

ROI_PREPROCESS_ROOT.mkdir(parents=True, exist_ok=True)
QA_OUT_DIR.mkdir(parents=True, exist_ok=True)
ROI_LUNG_RANGE_DIR.mkdir(parents=True, exist_ok=True)
LESION_ROI_LUNG_RANGE_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_METADATA_PATH = SOURCE_PREPROCESS_ROOT / "metadata_slices.csv"
SOURCE_ORGANS_PATH = SOURCE_PREPROCESS_ROOT / "metadata_organs.csv"

OUTPUT_METADATA_PATH = ROI_PREPROCESS_ROOT / "metadata_slices.csv"

# 0/0 기준
TS_LUNG_DILATE_ITER = 0
ROI_EXTRA_DILATE_ITER = 0

# True면 혹시 폐엽 마스크와 장기 마스크가 겹친 부분을 마지막에 제거
# TotalSeg 폐엽 그대로만 쓰고 싶으면 False로 바꾸면 됨.
USE_BODY_GUARD = False
USE_ORGAN_EXCLUSION = True
ORGAN_EXCLUSION_DILATE_ITER = 1

OVERWRITE_EXISTING_MASKS = False

LUNG_LOBE_NAMES = [
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
]

ORGAN_EXCLUSION_NAMES = [
    "heart",
    "aorta",
    "trachea",
    "esophagus",
    "liver",
    "stomach",
    "spleen",
    "pancreas",
]

FLAG_WARN_RATIO = 0.90
FLAG_FAIL_RATIO = 0.80

print("========== BUILD 0/0 ROI PREPROCESS ==========")
print("SOURCE_PREPROCESS_ROOT:", SOURCE_PREPROCESS_ROOT)
print("ROI_PREPROCESS_ROOT:", ROI_PREPROCESS_ROOT)
print("QA_OUT_DIR:", QA_OUT_DIR)
print("SOURCE_METADATA_PATH:", SOURCE_METADATA_PATH)
print("SOURCE_ORGANS_PATH:", SOURCE_ORGANS_PATH)
print("OUTPUT_METADATA_PATH:", OUTPUT_METADATA_PATH)

if not SOURCE_METADATA_PATH.exists():
    raise FileNotFoundError(f"metadata_slices.csv 없음: {SOURCE_METADATA_PATH}")

if not SOURCE_ORGANS_PATH.exists():
    raise FileNotFoundError(f"metadata_organs.csv 없음: {SOURCE_ORGANS_PATH}")


# ============================================================
# 2. helper functions
# ============================================================

def safe_name(x):
    return (
        str(x)
        .replace("\\", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
        .replace(" ", "_")
    )


def read_nii_image_and_array(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"NIfTI 없음: {path}")

    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return img, arr


def read_bool_nii(path):
    _, arr = read_nii_image_and_array(path)
    return arr > 0


def write_mask_like_reference(mask_arr, reference_img, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mask_img = sitk.GetImageFromArray(mask_arr.astype(np.uint8))
    mask_img.CopyInformation(reference_img)

    sitk.WriteImage(mask_img, str(out_path))


def valid_path_value(v):
    if v is None:
        return False

    if isinstance(v, float) and np.isnan(v):
        return False

    s = str(v).strip()

    if s == "" or s.lower() in ["nan", "none"]:
        return False

    return True


def resolve_path(v, root):
    if not valid_path_value(v):
        return None

    p = Path(str(v))

    if p.exists():
        return p

    p2 = root / p

    if p2.exists():
        return p2

    return p


def first_existing_path_from_row(row, candidate_cols, root):
    for col in candidate_cols:
        if col not in row.index:
            continue

        p = resolve_path(row[col], root)

        if p is not None and p.exists():
            return p

    return None


def safe_ratio(num, den):
    if den == 0:
        return np.nan

    return float(num) / float(den)


def dilate_mask(mask, iterations):
    if iterations <= 0:
        return mask.copy()

    return binary_dilation(
        mask,
        structure=np.ones((3, 3, 3), dtype=bool),
        iterations=int(iterations),
    )


def load_union_mask_from_organs(organs_df, group, patient_id, organ_names):
    organ_rows = organs_df[
        (organs_df["group"].astype(str) == str(group))
        & (organs_df["patient_id"].astype(str) == str(patient_id))
    ].copy()

    union_mask = None
    found_names = []
    missing_names = []

    for name in organ_names:
        sub = organ_rows[organ_rows["organ_name"].astype(str) == name]

        if len(sub) == 0:
            missing_names.append(name)
            continue

        p = resolve_path(sub.iloc[0]["organ_mask_1mm_nii"], SOURCE_PREPROCESS_ROOT)

        if p is None or not p.exists():
            missing_names.append(name)
            continue

        arr = read_bool_nii(p)

        if union_mask is None:
            union_mask = np.zeros_like(arr, dtype=bool)

        union_mask |= arr
        found_names.append(name)

    return union_mask, found_names, missing_names


def classify_flag(ratio):
    if pd.isna(ratio):
        return "NO_LESION_OR_NAN"

    if ratio < FLAG_FAIL_RATIO:
        return "FAIL_UNDER_0.80"

    if ratio < FLAG_WARN_RATIO:
        return "WARN_0.80_TO_0.90"

    return "PASS_OVER_0.90"


def select_crop_indices_by_geometry(crop_img, full_ref_img, crop_zdim, full_zdim):
    """
    ct_lung_range처럼 z축 crop된 image가
    full 1mm mask/lesion volume의 어느 z slice에 해당하는지 image geometry로 찾음.

    이유:
    - metadata_slices.csv는 full z축 전체 row를 가질 수 있음
    - ct_1mm_lung_range_nii는 폐 범위로 crop되어 zdim이 더 작을 수 있음
    - 그래서 metadata row 수와 ct_lung_range zdim을 같다고 보면 안 됨
    """

    crop_size = crop_img.GetSize()  # SimpleITK: (x, y, z)
    x_center = int(crop_size[0] // 2)
    y_center = int(crop_size[1] // 2)

    slice_indices = []

    for local_z in range(int(crop_zdim)):
        # crop image의 local_z 중심 좌표를 실제 물리 좌표로 변환
        physical_point = crop_img.TransformIndexToPhysicalPoint(
            (x_center, y_center, int(local_z))
        )

        # 그 물리 좌표가 full reference image에서 몇 번째 z인지 계산
        full_cont_idx = full_ref_img.TransformPhysicalPointToContinuousIndex(
            physical_point
        )

        full_z = int(round(full_cont_idx[2]))

        if full_z < 0 or full_z >= int(full_zdim):
            raise RuntimeError(
                f"crop local_z가 full z 범위를 벗어남: "
                f"local_z={local_z}, full_z={full_z}, full_zdim={full_zdim}"
            )

        slice_indices.append(full_z)

    slice_indices = np.asarray(slice_indices, dtype=int)

    if len(slice_indices) != int(crop_zdim):
        raise RuntimeError(
            f"계산된 slice_indices 길이와 crop_zdim 다름: "
            f"indices={len(slice_indices)}, crop_zdim={crop_zdim}"
        )

    if len(np.unique(slice_indices)) != len(slice_indices):
        raise RuntimeError(
            "계산된 slice_indices에 중복이 있음. "
            "crop/full image spacing 또는 origin 확인 필요."
        )

    return slice_indices


def build_roi_0_0(raw_ts_lung, body_guard=None, organ_exclusion=None):
    """
    0/0 ROI:
    - TotalSegmentator 폐엽 5개 합침
    - 폐 쪽 dilation 없음
    - 추가 ROI dilation 없음
    - body_guard 선택 적용
    - organ_exclusion 선택 적용
    """

    roi = raw_ts_lung.copy()

    if USE_BODY_GUARD and body_guard is not None:
        roi = roi & body_guard

    if USE_ORGAN_EXCLUSION and organ_exclusion is not None:
        roi = roi & (~organ_exclusion)

    return roi


# ============================================================
# 3. metadata load
# ============================================================

meta = pd.read_csv(
    SOURCE_METADATA_PATH,
    dtype={
        "group": str,
        "patient_id": str,
    },
)

organs = pd.read_csv(
    SOURCE_ORGANS_PATH,
    dtype={
        "group": str,
        "patient_id": str,
        "organ_name": str,
    },
)

required_source_cols = [
    "group",
    "patient_id",
    "slice_index",
    "ct_1mm_lung_range_nii",
]

for col in required_source_cols:
    if col not in meta.columns:
        raise RuntimeError(f"source metadata_slices.csv에 필요한 컬럼 없음: {col}")

required_organ_cols = [
    "group",
    "patient_id",
    "organ_name",
    "organ_mask_1mm_nii",
]

for col in required_organ_cols:
    if col not in organs.columns:
        raise RuntimeError(f"metadata_organs.csv에 필요한 컬럼 없음: {col}")

patient_keys = (
    meta[["group", "patient_id"]]
    .drop_duplicates()
    .sort_values(["group", "patient_id"])
    .reset_index(drop=True)
)

print("metadata_slices rows:", len(meta))
print("metadata_organs rows:", len(organs))
print("patient count:", len(patient_keys))


# ============================================================
# 4. build ROI 0/0 files by patient
# ============================================================

new_meta_parts = []
coverage_rows = []
error_rows = []

for _, key in tqdm(
    patient_keys.iterrows(),
    total=len(patient_keys),
    desc="Build roi_0_0 by patient",
    ncols=100,
):
    group = str(key["group"])
    patient_id = str(key["patient_id"])

    patient_df = meta[
        (meta["group"].astype(str) == group)
        & (meta["patient_id"].astype(str) == patient_id)
    ].copy()

    patient_df = patient_df.sort_values("slice_index").reset_index(drop=True)

    try:
        row0 = patient_df.iloc[0]

        ct_lung_range_path = first_existing_path_from_row(
            row0,
            ["ct_1mm_lung_range_nii"],
            SOURCE_PREPROCESS_ROOT,
        )

        if ct_lung_range_path is None:
            raise FileNotFoundError(f"ct_1mm_lung_range_nii 없음: {group}/{patient_id}")

        ct_lung_img, ct_lung_arr = read_nii_image_and_array(ct_lung_range_path)
        crop_zdim = int(ct_lung_arr.shape[0])

        raw_ts_lung_full, found_lung_names, missing_lung_names = load_union_mask_from_organs(
            organs_df=organs,
            group=group,
            patient_id=patient_id,
            organ_names=LUNG_LOBE_NAMES,
        )

        if raw_ts_lung_full is None:
            raise RuntimeError(f"폐엽 마스크를 하나도 못 찾음: {group}/{patient_id}")

        # 폐 쪽 dilation 0
        raw_ts_lung_full = dilate_mask(
            raw_ts_lung_full,
            iterations=TS_LUNG_DILATE_ITER,
        )

        body_guard_full = None

        if USE_BODY_GUARD:
            body_guard_path = first_existing_path_from_row(
                row0,
                ["body_guard_1mm_nii", "body_guard_nii"],
                SOURCE_PREPROCESS_ROOT,
            )

            if body_guard_path is not None:
                body_guard_full = read_bool_nii(body_guard_path)

        organ_exclusion_full = None
        found_exclusion_names = []
        missing_exclusion_names = []

        if USE_ORGAN_EXCLUSION:
            organ_exclusion_raw, found_exclusion_names, missing_exclusion_names = load_union_mask_from_organs(
                organs_df=organs,
                group=group,
                patient_id=patient_id,
                organ_names=ORGAN_EXCLUSION_NAMES,
            )

            if organ_exclusion_raw is not None:
                organ_exclusion_full = dilate_mask(
                    organ_exclusion_raw,
                    iterations=ORGAN_EXCLUSION_DILATE_ITER,
                )

        roi_0_0_full = build_roi_0_0(
            raw_ts_lung=raw_ts_lung_full,
            body_guard=body_guard_full,
            organ_exclusion=organ_exclusion_full,
        )

        # lesion full mask
        lesion_path = first_existing_path_from_row(
            row0,
            ["lesion_mask_1mm_nii", "lesion_mask_nii"],
            SOURCE_PREPROCESS_ROOT,
        )

        if lesion_path is None:
            raise FileNotFoundError(f"원본 lesion_mask_1mm_nii 없음: {group}/{patient_id}")

        lesion_img, lesion_arr = read_nii_image_and_array(lesion_path)
        lesion_full = lesion_arr > 0

        if roi_0_0_full.shape != lesion_full.shape:
            raise RuntimeError(
                f"roi와 lesion shape 다름: roi={roi_0_0_full.shape}, lesion={lesion_full.shape}"
            )

        if roi_0_0_full.shape[1:] != ct_lung_arr.shape[1:]:
            raise RuntimeError(
                f"roi와 ct lung_range y/x shape 다름: roi={roi_0_0_full.shape}, ct={ct_lung_arr.shape}"
            )

        slice_indices = select_crop_indices_by_geometry(
            crop_img=ct_lung_img,
            full_ref_img=lesion_img,
            crop_zdim=crop_zdim,
            full_zdim=roi_0_0_full.shape[0],
        )

        roi_0_0_crop = roi_0_0_full[slice_indices, :, :]
        lesion_crop = lesion_full[slice_indices, :, :]
        lesion_roi_0_0_crop = lesion_crop & roi_0_0_crop

        if roi_0_0_crop.shape != ct_lung_arr.shape:
            raise RuntimeError(
                f"roi crop과 ct_lung_range shape 다름: roi={roi_0_0_crop.shape}, ct={ct_lung_arr.shape}"
            )

        patient_safe = safe_name(f"{group}_{patient_id}")

        roi_out_path = (
            ROI_LUNG_RANGE_DIR
            / safe_name(group)
            / f"{patient_safe}_roi_0_0_lung_range.nii.gz"
        )

        lesion_roi_out_path = (
            LESION_ROI_LUNG_RANGE_DIR
            / safe_name(group)
            / f"{patient_safe}_lesion_mask_roi_0_0_lung_range.nii.gz"
        )

        if OVERWRITE_EXISTING_MASKS or (not roi_out_path.exists()):
            write_mask_like_reference(
                mask_arr=roi_0_0_crop,
                reference_img=ct_lung_img,
                out_path=roi_out_path,
            )

        if OVERWRITE_EXISTING_MASKS or (not lesion_roi_out_path.exists()):
            write_mask_like_reference(
                mask_arr=lesion_roi_0_0_crop,
                reference_img=ct_lung_img,
                out_path=lesion_roi_out_path,
            )

        lesion_total_full = int(lesion_full.sum())
        lesion_total_lung_range = int(lesion_crop.sum())
        lesion_inside_roi = int(lesion_roi_0_0_crop.sum())
        lesion_outside_roi = int((lesion_crop & (~roi_0_0_crop)).sum())

        inside_ratio_full = safe_ratio(lesion_inside_roi, lesion_total_full)
        inside_ratio_lung_range = safe_ratio(lesion_inside_roi, lesion_total_lung_range)

        qa_flag = classify_flag(inside_ratio_full)
        qa_flag_lung_range = classify_flag(inside_ratio_lung_range)
        slice_indices_list = [int(v) for v in slice_indices]
        
        source_by_slice = (
            patient_df
            .drop_duplicates(subset=["slice_index"])
            .set_index("slice_index", drop=False)
        )
        
        missing_meta_slices = [
            z for z in slice_indices_list
            if z not in source_by_slice.index
        ]
        
        if len(missing_meta_slices) > 0:
            raise RuntimeError(
                f"metadata_slices.csv에 lung_range slice_index 일부가 없음: "
                f"missing_count={len(missing_meta_slices)}, "
                f"examples={missing_meta_slices[:10]}"
            )
        
        tmp_meta = source_by_slice.loc[slice_indices_list].copy().reset_index(drop=True)
        
        # 패치분할 코드에서 local_z와 metadata row가 1:1로 맞도록 저장
        tmp_meta["local_z"] = np.arange(len(tmp_meta), dtype=int)
        
        tmp_meta["ct_1mm_lung_range_nii"] = str(ct_lung_range_path)
        tmp_meta["roi_0_0_lung_range_nii"] = str(roi_out_path)
        tmp_meta["lesion_mask_roi_0_0_lung_range_nii"] = str(lesion_roi_out_path)

        tmp_meta["roi_0_0_voxels_lung_range"] = int(roi_0_0_crop.sum())
        tmp_meta["lesion_total_voxels_full"] = lesion_total_full
        tmp_meta["lesion_total_voxels_lung_range"] = lesion_total_lung_range
        tmp_meta["lesion_inside_roi_0_0_voxels"] = lesion_inside_roi
        tmp_meta["lesion_inside_roi_0_0_ratio_full"] = inside_ratio_full
        tmp_meta["lesion_inside_roi_0_0_ratio_lung_range"] = inside_ratio_lung_range
        tmp_meta["lesion_outside_roi_0_0_voxels_lung_range"] = lesion_outside_roi
        tmp_meta["lesion_outside_roi_0_0_ratio_lung_range"] = safe_ratio(
            lesion_outside_roi,
            lesion_total_lung_range,
        )
        tmp_meta["roi_0_0_qa_flag"] = qa_flag
        tmp_meta["roi_0_0_qa_flag_lung_range"] = qa_flag_lung_range
        new_meta_parts.append(tmp_meta)

        coverage_rows.append({
            "group": group,
            "patient_id": patient_id,
            "status": "OK",

            "ts_lung_dilate_iter": TS_LUNG_DILATE_ITER,
            "roi_extra_dilate_iter": ROI_EXTRA_DILATE_ITER,
            "use_body_guard": bool(USE_BODY_GUARD),
            "use_organ_exclusion": bool(USE_ORGAN_EXCLUSION),
            "organ_exclusion_dilate_iter": ORGAN_EXCLUSION_DILATE_ITER,

            "ct_1mm_lung_range_nii": str(ct_lung_range_path),
            "roi_0_0_lung_range_nii": str(roi_out_path),
            "lesion_mask_roi_0_0_lung_range_nii": str(lesion_roi_out_path),

            "roi_0_0_voxels_lung_range": int(roi_0_0_crop.sum()),

            "lesion_total_voxels_full": lesion_total_full,
            "lesion_total_voxels_lung_range": lesion_total_lung_range,

            "lesion_inside_roi_0_0_voxels": lesion_inside_roi,
            "lesion_inside_roi_0_0_ratio_full": inside_ratio_full,
            "lesion_inside_roi_0_0_ratio_lung_range": inside_ratio_lung_range,

            "lesion_outside_roi_0_0_voxels_lung_range": lesion_outside_roi,
            "lesion_outside_roi_0_0_ratio_lung_range": safe_ratio(
                lesion_outside_roi,
                lesion_total_lung_range,
            ),

            "qa_flag": qa_flag,
            "qa_flag_lung_range": qa_flag_lung_range,
            "found_lung_names": "|".join(found_lung_names),
            "missing_lung_names": "|".join(missing_lung_names),
            "found_exclusion_names": "|".join(found_exclusion_names),
            "missing_exclusion_names": "|".join(missing_exclusion_names),
            "lesion_mask_1mm_nii": str(lesion_path),
        })

        del (
            ct_lung_img,
            ct_lung_arr,
            raw_ts_lung_full,
            roi_0_0_full,
            roi_0_0_crop,
            lesion_img,
            lesion_full,
            lesion_crop,
            lesion_roi_0_0_crop,
        )

        if body_guard_full is not None:
            del body_guard_full

        if organ_exclusion_full is not None:
            del organ_exclusion_full

        gc.collect()

    except Exception as e:
        error_rows.append({
            "group": group,
            "patient_id": patient_id,
            "error": str(e),
        })

        print("[ERROR]", group, patient_id, str(e))
        gc.collect()


# ============================================================
# 5. save metadata / QA CSV
# ============================================================

if len(new_meta_parts) == 0:
    raise RuntimeError("생성된 metadata row가 없음. error CSV를 확인하세요.")

new_meta = pd.concat(new_meta_parts, ignore_index=True)

# 패치분할 코드가 필요한 핵심 컬럼을 앞쪽으로 정렬
front_cols = [
    "group",
    "patient_id",
    "slice_index",
]

if "is_original_slice" in new_meta.columns:
    front_cols.append("is_original_slice")

front_cols.extend([
    "ct_1mm_lung_range_nii",
    "roi_0_0_lung_range_nii",
    "lesion_mask_roi_0_0_lung_range_nii",
    "roi_0_0_qa_flag",
    "lesion_inside_roi_0_0_ratio_full",
    "lesion_inside_roi_0_0_ratio_lung_range",
])

front_cols = [c for c in front_cols if c in new_meta.columns]
other_cols = [c for c in new_meta.columns if c not in front_cols]

new_meta = new_meta[front_cols + other_cols]

new_meta.to_csv(
    OUTPUT_METADATA_PATH,
    index=False,
    encoding="utf-8-sig",
)

coverage_df = pd.DataFrame(coverage_rows)
coverage_csv = QA_OUT_DIR / "before_patchsplit_roi_0_0_lesion_coverage_by_patient.csv"
coverage_df.to_csv(
    coverage_csv,
    index=False,
    encoding="utf-8-sig",
)

if len(error_rows) > 0:
    error_df = pd.DataFrame(error_rows)
else:
    error_df = pd.DataFrame(columns=["group", "patient_id", "error"])

error_csv = QA_OUT_DIR / "build_roi_0_0_errors.csv"
error_df.to_csv(
    error_csv,
    index=False,
    encoding="utf-8-sig",
)

# summary
summary_rows = []

ok_df = coverage_df[coverage_df["status"] == "OK"].copy()

if len(ok_df) > 0:
    for group, gdf in ok_df.groupby("group"):
        ratios = pd.to_numeric(
            gdf["lesion_inside_roi_0_0_ratio_full"],
            errors="coerce",
        )

        summary_rows.append({
            "group": group,
            "num_patients": len(gdf),
            "mean_lesion_inside_roi_0_0_ratio_full": ratios.mean(),
            "min_lesion_inside_roi_0_0_ratio_full": ratios.min(),
            "max_lesion_inside_roi_0_0_ratio_full": ratios.max(),
            "num_pass_over_0.90": int((gdf["qa_flag_lung_range"] == "PASS_OVER_0.90").sum()),
            "num_warn_0.80_to_0.90": int((gdf["qa_flag_lung_range"] == "WARN_0.80_TO_0.90").sum()),
            "num_fail_under_0.80": int((gdf["qa_flag_lung_range"] == "FAIL_UNDER_0.80").sum()),
        })

    all_ratios = pd.to_numeric(
        ok_df["lesion_inside_roi_0_0_ratio_full"],
        errors="coerce",
    )

    summary_rows.append({
        "group": "ALL",
        "num_patients": len(ok_df),
        "mean_lesion_inside_roi_0_0_ratio_full": all_ratios.mean(),
        "min_lesion_inside_roi_0_0_ratio_full": all_ratios.min(),
        "max_lesion_inside_roi_0_0_ratio_full": all_ratios.max(),
        "num_pass_over_0.90": int((ok_df["qa_flag"] == "PASS_OVER_0.90").sum()),
        "num_warn_0.80_to_0.90": int((ok_df["qa_flag"] == "WARN_0.80_TO_0.90").sum()),
        "num_fail_under_0.80": int((ok_df["qa_flag"] == "FAIL_UNDER_0.80").sum()),
    })

summary_df = pd.DataFrame(summary_rows)
summary_csv = QA_OUT_DIR / "before_patchsplit_roi_0_0_lesion_coverage_summary.csv"
summary_df.to_csv(
    summary_csv,
    index=False,
    encoding="utf-8-sig",
)

flag_df = coverage_df[
    coverage_df["qa_flag_lung_range"].isin([
        "WARN_0.80_TO_0.90",
        "FAIL_UNDER_0.80",
        "NO_LESION_OR_NAN",
    ])
].copy()

flag_csv = QA_OUT_DIR / "before_patchsplit_roi_0_0_lesion_coverage_flag_cases.csv"
flag_df.to_csv(
    flag_csv,
    index=False,
    encoding="utf-8-sig",
)

config_json = ROI_PREPROCESS_ROOT / "roi_0_0_build_config.json"

with open(config_json, "w", encoding="utf-8") as f:
    json.dump(
        {
            "source_preprocess_root": str(SOURCE_PREPROCESS_ROOT),
            "roi_preprocess_root": str(ROI_PREPROCESS_ROOT),
            "qa_out_dir": str(QA_OUT_DIR),
            "roi_version_name": ROI_VERSION_NAME,
            "ts_lung_dilate_iter": TS_LUNG_DILATE_ITER,
            "roi_extra_dilate_iter": ROI_EXTRA_DILATE_ITER,
            "use_body_guard": bool(USE_BODY_GUARD),
            "use_organ_exclusion": bool(USE_ORGAN_EXCLUSION),
            "organ_exclusion_dilate_iter": ORGAN_EXCLUSION_DILATE_ITER,
            "overwrite_existing_masks": bool(OVERWRITE_EXISTING_MASKS),
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

print("\n========== 0/0 ROI BUILD FINISHED ==========")
print("ROI_PREPROCESS_ROOT:", ROI_PREPROCESS_ROOT)
print("metadata:", OUTPUT_METADATA_PATH)
print("coverage_csv:", coverage_csv)
print("summary_csv:", summary_csv)
print("flag_csv:", flag_csv)
print("error_csv:", error_csv)
print("config_json:", config_json)

print("\n========== Coverage summary ==========")
display(summary_df)

print("\n========== Flag cases ==========")
display(flag_df)

# %%==== CELL ====

