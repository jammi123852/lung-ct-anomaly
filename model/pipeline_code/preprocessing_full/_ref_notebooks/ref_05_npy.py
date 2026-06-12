# ============================================================
# Build final training-ready dataset v2
# ------------------------------------------------------------
# 목적:
#   - 기존 전처리 결과와 patch CSV를 학습용 최종 폴더로 정리
#   - 환자별 full volume을 .npy로 저장
#   - CT는 int16 유지
#   - pure_lung / organ_exclusion mask 저장
#   - patch CSV에 학습용 .npy 경로 추가
#   - 빈 patch CSV / patch 없는 환자도 안전 처리
#   - train / val / test 환자 단위 split 생성
#   - 이미 만든 환자는 재실행 시 skip 가능
# ============================================================

from pathlib import Path
import json
import shutil
import hashlib
import random
import time
import gc

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from pandas.errors import EmptyDataError


# ============================================================
# 1. CONFIG
# ============================================================

CONFIG = {
    # 1번 코드가 만든 0/0 ROI 전처리 최종 폴더
    "preprocess_root": r"E:\jyp\ct_data_2d_preprocessed\NSCLC_MSD_preprocessed_roi0_0_ts_lung_raw_no_dilate_usable_only_v1",

    # 2번 코드가 만든 패치분할 결과 폴더
    "patch_root": r"E:\jyp\ct_data_2d_preprocessed\NSCLC_MSD_patchsplit_roi0_0_ts_lung_raw_no_dilate_usable_only_v1",

    # 3번 코드가 만들 NPY test-ready 폴더
    "training_ready_root": r"E:\jyp\ct_data_2d_preprocessed\NSCLC_MSD_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1",

    "ct_dtype": "int16",
    "mask_dtype": "uint8",

    "skip_existing": True,
    "overwrite_existing": False,

    "require_patch_csv": True,
    "drop_source_nii_columns_in_patch_csv": False,

    "default_label": "lesion_test",

    "make_train_val_test_split": False,
    "split_seed": 42,
    "train_ratio": 0.80,
    "val_ratio": 0.10,
    "test_ratio": 0.10,
}


PRE_ROOT = Path(CONFIG["preprocess_root"])
PATCH_ROOT = Path(CONFIG["patch_root"])
PATCH_BY_PATIENT_DIR = PATCH_ROOT / "patches_by_patient"

OUT_ROOT = Path(CONFIG["training_ready_root"])
VOLUME_ROOT = OUT_ROOT / "volumes_npy"
PATCH_INDEX_ROOT = OUT_ROOT / "patch_index_by_patient"
MANIFEST_DIR = OUT_ROOT / "manifests"
CONFIG_DIR = OUT_ROOT / "configs"
LOG_DIR = OUT_ROOT / "logs"

for d in [OUT_ROOT, VOLUME_ROOT, PATCH_INDEX_ROOT, MANIFEST_DIR, CONFIG_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

with open(CONFIG_DIR / "training_ready_config.json", "w", encoding="utf-8") as f:
    json.dump(CONFIG, f, indent=2, ensure_ascii=False)

print("========== CONFIG ==========")
print("PRE_ROOT:", PRE_ROOT)
print("PATCH_ROOT:", PATCH_ROOT)
print("PATCH_BY_PATIENT_DIR:", PATCH_BY_PATIENT_DIR)
print("OUT_ROOT:", OUT_ROOT)
print("VOLUME_ROOT:", VOLUME_ROOT)
print("PATCH_INDEX_ROOT:", PATCH_INDEX_ROOT)
print("ct_dtype:", CONFIG["ct_dtype"])


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


def safe_id_from_patient_id(patient_id: str, max_prefix_len: int = 80) -> str:
    """
    긴 LUNA16 UID를 Windows 경로에서 안전하게 쓰기 위한 짧은 ID 생성.
    원래 patient_id는 manifest에 그대로 보존함.
    """

    patient_id = str(patient_id)

    cleaned = (
        patient_id
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
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
    )

    digest = hashlib.md5(patient_id.encode("utf-8")).hexdigest()[:10]

    if len(cleaned) > max_prefix_len:
        cleaned = cleaned[:max_prefix_len]

    return f"{cleaned}__{digest}"


def safe_patient_csv_name(patient_id: str) -> str:
    """
    patch_all_v2_resume 쪽 환자별 CSV 이름과 맞추기 위한 safe name.
    """

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


def read_sitk_array(path: Path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return img, arr


def convert_ct_array_for_save(ct_arr: np.ndarray, dtype_name: str):
    """
    CT HU를 학습용 cache로 저장.
    현재 기본은 int16.
    sitkLinear 보간 때문에 float HU가 있을 수 있으므로 int16 저장 시 반올림됨.
    """

    dtype_name = str(dtype_name).lower()

    if dtype_name == "int16":
        if np.issubdtype(ct_arr.dtype, np.floating):
            out = np.rint(ct_arr)
            out = np.clip(out, np.iinfo(np.int16).min, np.iinfo(np.int16).max)
            return out.astype(np.int16)

        return ct_arr.astype(np.int16, copy=False)

    if dtype_name == "float32":
        return ct_arr.astype(np.float32, copy=False)

    raise ValueError(f"지원하지 않는 ct_dtype: {dtype_name}")


def convert_mask_array_for_save(mask_arr: np.ndarray, dtype_name: str):
    """
    mask는 0/1 형태로 저장.
    """

    dtype_name = str(dtype_name).lower()

    if dtype_name == "uint8":
        return (mask_arr > 0).astype(np.uint8)

    if dtype_name == "bool":
        return mask_arr > 0

    raise ValueError(f"지원하지 않는 mask_dtype: {dtype_name}")


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json_if_exists(path: Path):
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def copy_config_if_exists(src: Path, dst: Path):
    if src.exists():
        shutil.copy2(str(src), str(dst))
        return True

    return False


def read_csv_safe(path: Path):
    """
    빈 CSV도 에러 없이 읽기.
    """

    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def count_rows_csv_safe(path: Path):
    """
    patch CSV row 수 계산.
    빈 CSV면 0 반환.
    """

    if not path.exists():
        return 0

    try:
        df = pd.read_csv(path, usecols=["patient_id"])
        return int(len(df))
    except EmptyDataError:
        return 0
    except Exception:
        try:
            df = pd.read_csv(path)
            return int(len(df))
        except Exception:
            return 0


def make_empty_patch_df():
    """
    patch가 없는 환자도 빈 CSV에 최소 header를 남김.
    0/0 ROI 테스트셋용.
    """

    return pd.DataFrame(columns=[
        "group",
        "patient_id",
        "safe_id",
        "label",

        "local_z",
        "slice_index",
        "is_original_slice",
        "y0",
        "x0",
        "y1",
        "x1",

        "patch_size",
        "patch_stride",

        "roi_0_0_pixels",
        "lesion_pixels",

        "roi_0_0_patch_ratio",
        "lesion_patch_ratio",
        "slice_roi_0_0_ratio",

        "has_lesion_patch",
        "lesion_zone_type",

        "position_bin",
        "z_level",
        "z_ratio",
        "central_peripheral",
        "central_distance_ratio_mean",
        "left_right_metadata",

        "volume_dir",
        "ct_hu_npy",
        "roi_0_0_npy",
        "lesion_mask_roi_0_0_npy",
        "volume_meta_json",

        "rel_volume_dir",
        "rel_ct_hu_npy",
        "rel_roi_0_0_npy",
        "rel_lesion_mask_roi_0_0_npy",
        "rel_volume_meta_json",
    ])
def find_existing_patch_csv(group_name: str, patient_id: str, patch_summary_df=None):
    """
    테스트 patch 결과에서 환자별 patch csv 찾기.

    1순위:
        patch_patient_summary.csv의 group + patient_id + patient_csv

    2순위:
        patches_by_patient 폴더에서 group_patient_id safe name 직접 검색
    """

    group_name = str(group_name)
    patient_id = str(patient_id)

    if patch_summary_df is not None and len(patch_summary_df) > 0:
        if (
            "group" in patch_summary_df.columns
            and "patient_id" in patch_summary_df.columns
            and "patient_csv" in patch_summary_df.columns
        ):
            rows = patch_summary_df[
                (patch_summary_df["group"].astype(str) == group_name)
                & (patch_summary_df["patient_id"].astype(str) == patient_id)
            ]

            if len(rows) > 0:
                for p in rows["patient_csv"].dropna().tolist():
                    p = Path(str(p))

                    if p.exists():
                        return p

        if "patient_id" in patch_summary_df.columns and "patient_csv" in patch_summary_df.columns:
            rows = patch_summary_df[
                patch_summary_df["patient_id"].astype(str) == patient_id
            ]

            if len(rows) > 0:
                for p in rows["patient_csv"].dropna().tolist():
                    p = Path(str(p))

                    if p.exists():
                        return p

    direct_name = safe_patient_csv_name(f"{group_name}_{patient_id}")
    direct_path = PATCH_BY_PATIENT_DIR / f"{direct_name}.csv"

    if direct_path.exists():
        return direct_path

    fallback_name = safe_patient_csv_name(patient_id)
    fallback_path = PATCH_BY_PATIENT_DIR / f"{fallback_name}.csv"

    if fallback_path.exists():
        return fallback_path

    return None


# ============================================================
# 3. load metadata
# ============================================================

metadata_slices_path = PRE_ROOT / "metadata_slices.csv"
metadata_organs_path = PRE_ROOT / "metadata_organs.csv"
patch_summary_path = PATCH_ROOT / "patch_patient_summary.csv"
patch_config_path = PATCH_ROOT / "patch_config.json"
preprocess_config_path = PRE_ROOT / "final_preprocess_config.json"

if not metadata_slices_path.exists():
    raise FileNotFoundError(f"metadata_slices.csv 없음: {metadata_slices_path}")

if not PATCH_BY_PATIENT_DIR.exists():
    raise FileNotFoundError(f"patches_by_patient 폴더 없음: {PATCH_BY_PATIENT_DIR}")

slice_df = pd.read_csv(metadata_slices_path)

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

if patch_summary_path.exists():
    patch_summary_df = pd.read_csv(patch_summary_path)
else:
    patch_summary_df = pd.DataFrame()

patient_keys = (
    slice_df[["group", "patient_id"]]
    .drop_duplicates()
    .sort_values(["group", "patient_id"])
    .reset_index(drop=True)
)

print("========== INPUT CHECK ==========")
print("patient count:", len(patient_keys))
print("metadata_slices:", metadata_slices_path)
print("patch_summary exists:", patch_summary_path.exists())
print("patch csv dir:", PATCH_BY_PATIENT_DIR)

copy_config_if_exists(preprocess_config_path, CONFIG_DIR / "preprocess_config.json")
copy_config_if_exists(patch_config_path, CONFIG_DIR / "patch_config.json")


# ============================================================
# 4. build one patient training-ready files
# ============================================================
def build_one_patient(group_name: str, patient_id: str, row0, patch_summary_df):
    group_name = str(group_name)
    patient_id = str(patient_id)

    safe_id = safe_id_from_patient_id(f"{group_name}_{patient_id}")

    patient_volume_dir = VOLUME_ROOT / safe_id
    patient_patch_csv_out = PATCH_INDEX_ROOT / f"{safe_id}.csv"

    ct_npy = patient_volume_dir / "ct_hu.npy"
    roi_0_0_npy = patient_volume_dir / "roi_0_0.npy"
    lesion_npy = patient_volume_dir / "lesion_mask_roi_0_0.npy"
    meta_json = patient_volume_dir / "meta.json"
    done_marker = patient_volume_dir / ".done"

    if bool(CONFIG["overwrite_existing"]):
        if patient_volume_dir.exists():
            shutil.rmtree(patient_volume_dir)

        if patient_patch_csv_out.exists():
            patient_patch_csv_out.unlink()

    # --------------------------------------------------------
    # skip branch
    # --------------------------------------------------------
    if (
        bool(CONFIG["skip_existing"])
        and done_marker.exists()
        and ct_npy.exists()
        and roi_0_0_npy.exists()
        and lesion_npy.exists()
        and patient_patch_csv_out.exists()
    ):
        meta = read_json_if_exists(meta_json)
        patch_count = count_rows_csv_safe(patient_patch_csv_out)

        return {
            "group": group_name,
            "patient_id": patient_id,
            "safe_id": safe_id,
            "status": "skipped_existing",
            "label": meta.get("label", CONFIG["default_label"]),
        
            "volume_dir": str(patient_volume_dir),
            "ct_hu_npy": str(ct_npy),
            "roi_0_0_npy": str(roi_0_0_npy),
            "lesion_mask_roi_0_0_npy": str(lesion_npy),
            "meta_json": str(meta_json),
        
            "source_ct_1mm_lung_range_nii": meta.get("source_ct_1mm_lung_range_nii", ""),
            "source_roi_0_0_lung_range_nii": meta.get("source_roi_0_0_lung_range_nii", ""),
            "source_lesion_mask_roi_0_0_lung_range_nii": meta.get("source_lesion_mask_roi_0_0_lung_range_nii", ""),
        
            "source_patch_csv": meta.get("source_patch_csv", ""),
            "patch_csv": str(patient_patch_csv_out),
            "patch_count": int(patch_count),
        
            "lesion_voxel_count": int(meta.get("lesion_voxel_count", 0)),
            "roi_0_0_voxel_count": int(meta.get("roi_0_0_voxel_count", 0)),
        
            "shape_zyx": str(tuple(meta.get("shape_zyx", []))),
            "spacing_xyz": str(tuple(meta.get("spacing_xyz", []))),
        }

    patient_volume_dir.mkdir(parents=True, exist_ok=True)

    ct_path = Path(row0["ct_1mm_lung_range_nii"])
    roi_0_0_path = Path(row0["roi_0_0_lung_range_nii"])
    lesion_path = Path(row0["lesion_mask_roi_0_0_lung_range_nii"])

    if not ct_path.exists():
        raise FileNotFoundError(f"CT lung range 없음: {ct_path}")
    
    if not roi_0_0_path.exists():
        raise FileNotFoundError(f"roi_0_0 lung range 없음: {roi_0_0_path}")
    
    if not lesion_path.exists():
        raise FileNotFoundError(f"lesion_mask_roi_0_0 lung range 없음: {lesion_path}")

    source_patch_csv = find_existing_patch_csv(
        group_name=group_name,
        patient_id=patient_id,
        patch_summary_df=patch_summary_df,
    )

    if source_patch_csv is None and bool(CONFIG["require_patch_csv"]):
        raise FileNotFoundError(f"환자 patch CSV 없음: {group_name}/{patient_id}")

    # --------------------------------------------------------
    # read NIfTI
    # --------------------------------------------------------
    ct_img, ct_arr = read_sitk_array(ct_path)
    _, roi_0_0_arr = read_sitk_array(roi_0_0_path)
    _, lesion_arr = read_sitk_array(lesion_path)

    if ct_arr.shape != roi_0_0_arr.shape:
        raise RuntimeError(f"{group_name}/{patient_id}: CT와 roi_0_0 shape 다름")
    
    if ct_arr.shape != lesion_arr.shape:
        raise RuntimeError(f"{group_name}/{patient_id}: CT와 lesion_mask_roi_0_0 shape 다름")

    # --------------------------------------------------------
    # save arrays
    # --------------------------------------------------------
    ct_save = convert_ct_array_for_save(ct_arr, CONFIG["ct_dtype"])

    roi_0_0_bool = roi_0_0_arr > 0
    lesion_bool = lesion_arr > 0

    roi_0_0_save = convert_mask_array_for_save(roi_0_0_bool, CONFIG["mask_dtype"])
    lesion_save = convert_mask_array_for_save(lesion_bool, CONFIG["mask_dtype"])

    shape_zyx = tuple(map(int, ct_save.shape))
    spacing_xyz = tuple(map(float, ct_img.GetSpacing()))
    origin_xyz = tuple(map(float, ct_img.GetOrigin()))
    direction = tuple(map(float, ct_img.GetDirection()))
    np.save(ct_npy, ct_save)
    np.save(roi_0_0_npy, roi_0_0_save)
    np.save(lesion_npy, lesion_save)

    lesion_voxel_count = int(lesion_bool.sum())
    roi_0_0_voxel_count = int(roi_0_0_bool.sum())
    # --------------------------------------------------------
    # patch CSV 정리
    # --------------------------------------------------------
    patch_count = 0

    if source_patch_csv is not None and Path(source_patch_csv).exists():
        patch_df = read_csv_safe(Path(source_patch_csv))

        if len(patch_df) == 0:
            patch_df = make_empty_patch_df()
            patch_count = 0
        else:
            patch_count = int(len(patch_df))

            patch_df["group"] = group_name
            patch_df["safe_id"] = safe_id
            patch_df["label"] = CONFIG["default_label"]

            patch_df["volume_dir"] = str(patient_volume_dir)
            patch_df["ct_hu_npy"] = str(ct_npy)
            patch_df["roi_0_0_npy"] = str(roi_0_0_npy)
            patch_df["lesion_mask_roi_0_0_npy"] = str(lesion_npy)
            patch_df["volume_meta_json"] = str(meta_json)

            patch_df["rel_volume_dir"] = str(patient_volume_dir.relative_to(OUT_ROOT))
            patch_df["rel_ct_hu_npy"] = str(ct_npy.relative_to(OUT_ROOT))


            
            patch_df["rel_roi_0_0_npy"] = str(roi_0_0_npy.relative_to(OUT_ROOT))
            patch_df["rel_lesion_mask_roi_0_0_npy"] = str(lesion_npy.relative_to(OUT_ROOT))
            patch_df["rel_volume_meta_json"] = str(meta_json.relative_to(OUT_ROOT))

            if bool(CONFIG["drop_source_nii_columns_in_patch_csv"]):
                drop_cols = [
                    "ct_1mm_lung_range_nii",
                    "roi_0_0_lung_range_nii",
                    "lesion_mask_roi_0_0_lung_range_nii",
                ]

                patch_df = patch_df.drop(
                    columns=[c for c in drop_cols if c in patch_df.columns],
                    errors="ignore",
                )
    else:
        patch_df = make_empty_patch_df()
        patch_count = 0

    patch_df.to_csv(
        patient_patch_csv_out,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # meta 저장
    # --------------------------------------------------------
    meta = {
        "group": group_name,
        "patient_id": patient_id,
        "safe_id": safe_id,
        "label": CONFIG["default_label"],
    
        "shape_zyx": list(shape_zyx),
        "ct_dtype": str(ct_save.dtype),
        "mask_dtype": str(roi_0_0_save.dtype),
    
        "spacing_xyz": list(spacing_xyz),
        "origin_xyz": list(origin_xyz),
        "direction": list(direction),
    
        "source_ct_1mm_lung_range_nii": str(ct_path),
        "source_roi_0_0_lung_range_nii": str(roi_0_0_path),
        "source_lesion_mask_roi_0_0_lung_range_nii": str(lesion_path),
        "source_patch_csv": str(source_patch_csv) if source_patch_csv is not None else "",
    
        "ct_hu_npy": str(ct_npy),
        "roi_0_0_npy": str(roi_0_0_npy),
        "lesion_mask_roi_0_0_npy": str(lesion_npy),
    
        "patch_csv": str(patient_patch_csv_out),
        "patch_count": int(patch_count),
    
        "lesion_voxel_count": int(lesion_voxel_count),
        "roi_0_0_voxel_count": int(roi_0_0_voxel_count),
    
        "ct_dtype_save_policy": "int16 유지. sitkLinear 보간 소수점 HU는 반올림되어 저장됨.",
    }
    
    save_json(meta, meta_json)
    done_marker.write_text("done", encoding="utf-8")

    del ct_img, ct_arr, roi_0_0_arr, lesion_arr
    del ct_save, roi_0_0_save, lesion_save
    del roi_0_0_bool, lesion_bool
    del patch_df
    gc.collect()

    return {
        "group": group_name,
        "patient_id": patient_id,
        "safe_id": safe_id,
        "status": "success",
        "label": CONFIG["default_label"],

        "volume_dir": str(patient_volume_dir),
        "ct_hu_npy": str(ct_npy),
        "roi_0_0_npy": str(roi_0_0_npy),
        "lesion_mask_roi_0_0_npy": str(lesion_npy),
        "meta_json": str(meta_json),

        "source_ct_1mm_lung_range_nii": str(ct_path),
        "source_roi_0_0_lung_range_nii": str(roi_0_0_path),
        "source_lesion_mask_roi_0_0_lung_range_nii": str(lesion_path),
        "source_patch_csv": str(source_patch_csv) if source_patch_csv is not None else "",

        "patch_csv": str(patient_patch_csv_out),
        "patch_count": int(patch_count),

        "lesion_voxel_count": int(lesion_voxel_count),
        "roi_0_0_voxel_count": int(roi_0_0_voxel_count),

        "shape_zyx": str(shape_zyx),
        "spacing_xyz": str(spacing_xyz),
    }

# ============================================================
# 5. run build
# ============================================================

summary_rows = []
error_rows = []

global_start = time.perf_counter()
total_cases = len(patient_keys)

print("\n========== BUILD TEST READY START ==========")

for case_idx, key_row in enumerate(
    tqdm(
        patient_keys.to_dict("records"),
        desc="Build test-ready dataset",
        total=total_cases,
        ncols=100,
        ascii=True,
        leave=True,
        dynamic_ncols=False,
    ),
    start=1,
):
    group_name = str(key_row["group"])
    patient_id = str(key_row["patient_id"])

    start = time.perf_counter()

    try:
        pdf = slice_df[
            (slice_df["group"].astype(str) == group_name)
            & (slice_df["patient_id"].astype(str) == patient_id)
        ].copy()

        if len(pdf) == 0:
            raise RuntimeError(f"metadata_slices.csv에 환자 row 없음: {group_name}/{patient_id}")

        pdf = pdf.sort_values("slice_index").reset_index(drop=True)
        row0 = pdf.iloc[0]

        row = build_one_patient(
            group_name=group_name,
            patient_id=patient_id,
            row0=row0,
            patch_summary_df=patch_summary_df,
        )

        elapsed = time.perf_counter() - start
        total_elapsed = time.perf_counter() - global_start
        avg = total_elapsed / case_idx
        remain = total_cases - case_idx
        eta = avg * remain

        row.update({
            "case_index": int(case_idx),
            "total_cases": int(total_cases),
            "elapsed_seconds": round(elapsed, 2),
            "elapsed_readable": format_seconds(elapsed),
            "total_elapsed_seconds": round(total_elapsed, 2),
            "total_elapsed_readable": format_seconds(total_elapsed),
            "estimated_remaining_seconds": round(eta, 2),
            "estimated_remaining_readable": format_seconds(eta),
        })

        summary_rows.append(row)

        pd.DataFrame(summary_rows).to_csv(
            LOG_DIR / "build_test_ready_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"[READY] {group_name}/{patient_id} 완료 "
            f"({case_idx}/{total_cases}) | "
            f"status={row['status']} | "
            f"patch={row.get('patch_count', 0)} | "
            f"lesion_voxels={row.get('lesion_voxel_count', 0)} | "
            f"이번 환자: {format_seconds(elapsed)} | "
            f"예상 남은 시간: {format_seconds(eta)}"
        )

    except Exception as e:
        elapsed = time.perf_counter() - start

        err = {
            "group": group_name,
            "patient_id": patient_id,
            "error": str(e),
            "elapsed_seconds": round(elapsed, 2),
            "elapsed_readable": format_seconds(elapsed),
        }

        error_rows.append(err)

        pd.DataFrame(error_rows).to_csv(
            LOG_DIR / "build_test_ready_errors.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print("[ERROR]", group_name, patient_id, str(e))

    gc.collect()


summary_df = pd.DataFrame(summary_rows)
error_df = pd.DataFrame(error_rows)

summary_df.to_csv(
    LOG_DIR / "build_test_ready_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

if len(error_df) > 0:
    error_df.to_csv(
        LOG_DIR / "build_test_ready_errors.csv",
        index=False,
        encoding="utf-8-sig",
    )

print("\n========== BUILD TEST READY FINISHED ==========")
print("OUT_ROOT:", OUT_ROOT)
print("success / skipped rows:", len(summary_df))
print("errors:", len(error_df))


# ============================================================
# 6. create patient manifest
# ============================================================

if len(summary_df) == 0:
    raise RuntimeError("성공한 환자가 없음. error csv 확인 필요.")

patient_manifest = summary_df.copy()

front_cols = [
    "group",
    "patient_id",
    "safe_id",
    "status",
    "label",
    "patch_count",
    "lesion_voxel_count",
    "roi_0_0_voxel_count",
    "shape_zyx",
    "spacing_xyz",
    "volume_dir",
    "ct_hu_npy",
    "roi_0_0_npy",
    "lesion_mask_roi_0_0_npy",
    "meta_json",
    "patch_csv",
]

front_cols = [c for c in front_cols if c in patient_manifest.columns]
other_cols = [c for c in patient_manifest.columns if c not in front_cols]
patient_manifest = patient_manifest[front_cols + other_cols]

patient_manifest.to_csv(
    MANIFEST_DIR / "patient_manifest.csv",
    index=False,
    encoding="utf-8-sig",
)

# source patch summary 복사
if patch_summary_path.exists():
    shutil.copy2(
        str(patch_summary_path),
        str(MANIFEST_DIR / "source_patch_patient_summary.csv"),
    )

if (PATCH_ROOT / "patch_position_counts.csv").exists():
    shutil.copy2(
        str(PATCH_ROOT / "patch_position_counts.csv"),
        str(MANIFEST_DIR / "source_patch_position_counts.csv"),
    )

if (PATCH_ROOT / "patch_count_by_patient.csv").exists():
    shutil.copy2(
        str(PATCH_ROOT / "patch_count_by_patient.csv"),
        str(MANIFEST_DIR / "source_patch_count_by_patient.csv"),
    )

if (PATCH_ROOT / "patch_lesion_zone_counts.csv").exists():
    shutil.copy2(
        str(PATCH_ROOT / "patch_lesion_zone_counts.csv"),
        str(MANIFEST_DIR / "source_patch_lesion_zone_counts.csv"),
    )

# ============================================================
# 7. summarize final patch CSVs
# ============================================================

patient_position_rows = []
patch_count_rows = []
patient_lesion_zone_rows = []

for _, row in tqdm(
    patient_manifest.iterrows(),
    total=len(patient_manifest),
    desc="Summarize final patch indexes",
    ncols=100,
    ascii=True,
):
    group_name = row.get("group", "")
    patient_id = row["patient_id"]
    safe_id = row["safe_id"]
    patch_csv = Path(row["patch_csv"])

    if not patch_csv.exists():
        patch_count_rows.append({
            "group": group_name,
            "patient_id": patient_id,
            "safe_id": safe_id,
            "patch_count": 0,
            "lesion_patch_count": 0,
            "no_lesion_patch_count": 0,
            "lesion_in_roi_patch_count": 0,
        })
        continue

    try:
        df = pd.read_csv(
            patch_csv,
            usecols=[
                "group",
                "patient_id",
                "position_bin",
                "has_lesion_patch",
                "lesion_zone_type",
            ],
        )
    except EmptyDataError:
        df = pd.DataFrame(columns=[
            "group",
            "patient_id",
            "position_bin",
            "has_lesion_patch",
            "lesion_zone_type",
        ])
    except ValueError:
        df = pd.DataFrame(columns=[
            "group",
            "patient_id",
            "position_bin",
            "has_lesion_patch",
            "lesion_zone_type",
        ])

    patch_count = int(len(df))
    lesion_patch_count = int(df["has_lesion_patch"].sum()) if "has_lesion_patch" in df.columns else 0

    if len(df) > 0 and "lesion_zone_type" in df.columns:
        lesion_zone_counts = df["lesion_zone_type"].value_counts().to_dict()
    else:
        lesion_zone_counts = {}

    patch_count_rows.append({
        "group": group_name,
        "patient_id": patient_id,
        "safe_id": safe_id,
        "patch_count": int(patch_count),
        "lesion_patch_count": int(lesion_patch_count),
        "no_lesion_patch_count": int(lesion_zone_counts.get("no_lesion", 0)),
        "lesion_in_roi_patch_count": int(lesion_zone_counts.get("lesion_in_roi_patch", 0)),
    })

    if len(df) == 0:
        continue

    if "position_bin" in df.columns:
        for position_bin, count in df["position_bin"].value_counts().items():
            patient_position_rows.append({
                "group": group_name,
                "patient_id": patient_id,
                "safe_id": safe_id,
                "position_bin": position_bin,
                "patch_count": int(count),
            })

    if "lesion_zone_type" in df.columns:
        for lesion_zone_type, count in df["lesion_zone_type"].value_counts().items():
            patient_lesion_zone_rows.append({
                "group": group_name,
                "patient_id": patient_id,
                "safe_id": safe_id,
                "lesion_zone_type": lesion_zone_type,
                "patch_count": int(count),
            })


patient_position_df = pd.DataFrame(patient_position_rows)
patient_lesion_zone_df = pd.DataFrame(patient_lesion_zone_rows)
patch_count_df = pd.DataFrame(patch_count_rows)

if len(patient_position_df) > 0:
    patch_position_counts = (
        patient_position_df
        .groupby("position_bin", as_index=False)["patch_count"]
        .sum()
        .sort_values("position_bin")
    )
else:
    patch_position_counts = pd.DataFrame(columns=["position_bin", "patch_count"])

if len(patient_lesion_zone_df) > 0:
    patch_lesion_zone_counts = (
        patient_lesion_zone_df
        .groupby("lesion_zone_type", as_index=False)["patch_count"]
        .sum()
        .sort_values("lesion_zone_type")
    )
else:
    patch_lesion_zone_counts = pd.DataFrame(columns=["lesion_zone_type", "patch_count"])

patient_position_df.to_csv(
    MANIFEST_DIR / "patient_position_counts.csv",
    index=False,
    encoding="utf-8-sig",
)

patch_position_counts.to_csv(
    MANIFEST_DIR / "patch_position_counts.csv",
    index=False,
    encoding="utf-8-sig",
)

patient_lesion_zone_df.to_csv(
    MANIFEST_DIR / "patient_lesion_zone_counts.csv",
    index=False,
    encoding="utf-8-sig",
)

patch_lesion_zone_counts.to_csv(
    MANIFEST_DIR / "patch_lesion_zone_counts.csv",
    index=False,
    encoding="utf-8-sig",
)

patch_count_df.to_csv(
    MANIFEST_DIR / "patch_count_by_patient.csv",
    index=False,
    encoding="utf-8-sig",
)

# ============================================================
# 8. train / val / test split
# ============================================================

if bool(CONFIG["make_train_val_test_split"]):
    split_base = patient_manifest.copy()

    split_base["patch_count"] = split_base["patch_count"].fillna(0).astype(int)

    # patch_count 0인 환자는 split 제외
    split_df = split_base[
        split_base["patch_count"] > 0
    ][[
        "patient_id",
        "safe_id",
        "patch_count",
        "patch_csv",
        "volume_dir",
        "ct_hu_npy",
        "roi_0_0_npy",
        "lesion_mask_roi_0_0_npy",
    ]].copy()

    rng = random.Random(int(CONFIG["split_seed"]))
    ids = split_df["patient_id"].tolist()
    rng.shuffle(ids)

    n = len(ids)

    n_train = int(round(n * float(CONFIG["train_ratio"])))
    n_val = int(round(n * float(CONFIG["val_ratio"])))

    # 비율 반올림 때문에 전체를 넘지 않게 보정
    if n_train + n_val > n:
        n_val = max(0, n - n_train)

    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train + n_val])
    test_ids = set(ids[n_train + n_val:])

    def assign_split(pid):
        if pid in train_ids:
            return "train"

        if pid in val_ids:
            return "val"

        return "test"

    split_df["split"] = split_df["patient_id"].apply(assign_split)

    split_df.to_csv(
        MANIFEST_DIR / "train_val_test_split.csv",
        index=False,
        encoding="utf-8-sig",
    )

    split_df[split_df["split"] == "train"].to_csv(
        MANIFEST_DIR / "train_patients.csv",
        index=False,
        encoding="utf-8-sig",
    )

    split_df[split_df["split"] == "val"].to_csv(
        MANIFEST_DIR / "val_patients.csv",
        index=False,
        encoding="utf-8-sig",
    )

    split_df[split_df["split"] == "test"].to_csv(
        MANIFEST_DIR / "test_patients.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n========== SPLIT COUNT ==========")
    print(split_df["split"].value_counts())

    # --------------------------------------------------------
    # split별 position_bin 분포 확인
    # --------------------------------------------------------
    split_position_rows = []

    split_map = dict(zip(split_df["patient_id"], split_df["split"]))

    if len(patient_position_df) > 0:
        for _, r in patient_position_df.iterrows():
            pid = r["patient_id"]

            if pid not in split_map:
                continue

            split_position_rows.append({
                "split": split_map[pid],
                "patient_id": pid,
                "safe_id": r["safe_id"],
                "position_bin": r["position_bin"],
                "patch_count": int(r["patch_count"]),
            })

    split_position_df = pd.DataFrame(split_position_rows)

    if len(split_position_df) > 0:
        split_position_summary = (
            split_position_df
            .groupby(["split", "position_bin"], as_index=False)["patch_count"]
            .sum()
            .sort_values(["split", "position_bin"])
        )
    else:
        split_position_summary = pd.DataFrame(columns=["split", "position_bin", "patch_count"])

    split_position_df.to_csv(
        MANIFEST_DIR / "split_patient_position_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    split_position_summary.to_csv(
        MANIFEST_DIR / "split_position_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n========== SPLIT POSITION COUNTS ==========")
    display(split_position_summary)


# ============================================================
# 9. final check
# ============================================================

print("\n========== FINAL OUTPUT ==========")
print("OUT_ROOT:", OUT_ROOT)
print("volumes_npy:", VOLUME_ROOT)
print("patch_index_by_patient:", PATCH_INDEX_ROOT)
print("manifests:", MANIFEST_DIR)
print("configs:", CONFIG_DIR)
print("logs:", LOG_DIR)

print("\n========== FILES ==========")
print("patient_manifest:", MANIFEST_DIR / "patient_manifest.csv")
print("patch_position_counts:", MANIFEST_DIR / "patch_position_counts.csv")
print("patch_lesion_zone_counts:", MANIFEST_DIR / "patch_lesion_zone_counts.csv")
print("patch_count_by_patient:", MANIFEST_DIR / "patch_count_by_patient.csv")
print("patient_position_counts:", MANIFEST_DIR / "patient_position_counts.csv")
print("patient_lesion_zone_counts:", MANIFEST_DIR / "patient_lesion_zone_counts.csv")

if bool(CONFIG["make_train_val_test_split"]):
    print("train_val_test_split:", MANIFEST_DIR / "train_val_test_split.csv")
    print("split_position_counts:", MANIFEST_DIR / "split_position_counts.csv")
print("\n========== Position bin counts ==========")
display(patch_position_counts)


print("\n========== Lesion zone type counts ==========")
display(patch_lesion_zone_counts)

print("\n========== Patch count by patient summary ==========")
if len(patch_count_df) > 0:
    display(patch_count_df["patch_count"].describe())
else:
    print("patch_count_df 비어 있음")

print("\n========== Lesion patch count by patient summary ==========")
if len(patch_count_df) > 0 and "lesion_patch_count" in patch_count_df.columns:
    display(patch_count_df["lesion_patch_count"].describe())
else:
    print("lesion_patch_count 없음")

print("\n========== Error count ==========")
print(len(error_df))

if len(error_df) > 0:
    print("error file:", LOG_DIR / "build_test_ready_errors.csv")

# %%==== CELL ====

