# ============================================================
# Final preprocessing pipeline
# ------------------------------------------------------------
# 목적:
#   - 5명 랜덤 선택
#   - 원본 spacing full volume에서 TotalSegmentator 장기 mask 생성
#   - CT를 z축 1mm로 resampling
#   - 장기 mask를 1mm CT grid로 nearest-neighbor 정렬
#   - 1mm CT에서 세밀 폐 mask 생성
#   - 장기 제외 mask 생성
#   - pure lung mask 생성
#   - slice별 PNG / overlay 저장
#
# 제외:
#   - patch 추출
#   - 1mm CT에서 TotalSegmentator 재실행
#   - Dice 비교
# ============================================================

import os
import json
import random
import shutil
import subprocess
import traceback
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from scipy.ndimage import binary_dilation


# ============================================================
# 1. CONFIG
# ============================================================

CONFIG = {
    # ========================================================
    # 입력 / 출력 경로
    # ========================================================

    "src_roots": [
        r"E:\jyp\ct_data_2d\Normal Cases",
        r"E:\jyp\ct_data_2d\LUNA16_Normal_Candidates_287",
    ],

    # 흉곽 후보 제거 버전
    "out_root": r"E:\jyp\ct_data_2d_preprocessed\Normal_LUNA16_final_all_nonfast_v2_tslungguard_nochest",

    "group_name": "Normal",

    # ========================================================
    # 처리 범위
    # ========================================================

    "process_all_cases": True,

    # process_all_cases=False일 때만 사용
    "num_random_patients": 1,
    "random_seed": 42,

    # ========================================================
    # 방향 / spacing
    # ========================================================

    "orientation": "LPS",
    "target_z": 1.0,

    # ========================================================
    # CT window
    # ========================================================

    "hu_min": -1000,
    "hu_max": 400,

    # body 외곽 확인용 threshold
    "body_guard_hu_threshold": -500,

    # ========================================================
    # TotalSegmentator 실행 설정
    # ========================================================

    # False = non-fast 정밀 모드
    "use_fast_totalseg": False,

    # False면 기존 결과 재사용
    # 단, 새 out_root라면 처음부터 13개 ROI만 생성됨
    "overwrite_totalseg": False,

    "print_totalseg_tail": False,

    # ========================================================
    # TotalSegmentator ROI 이름 묶음
    # ========================================================

    # TS 폐 mask는 최종 폐 mask로 직접 쓰지 않음
    # 기존 HU 기반 폐 추출 결과를 제한하는 guard로만 사용
    "ts_lung_roi_names": [
        "lung_upper_lobe_left",
        "lung_lower_lobe_left",
        "lung_upper_lobe_right",
        "lung_middle_lobe_right",
        "lung_lower_lobe_right",
    ],

    # pure_lung에서 제외할 장기
    "organ_exclusion_roi_names": [
        "heart",
        "aorta",
        "trachea",
        "esophagus",
        "liver",
        "stomach",
        "spleen",
        "pancreas",
    ],

    # 실제 TotalSegmentator에 요청할 전체 ROI subset
    # 흉곽 후보 제거 후 총 13개
    "organ_roi_subset": [
        # TS 폐 guard용 5개
        "lung_upper_lobe_left",
        "lung_lower_lobe_left",
        "lung_upper_lobe_right",
        "lung_middle_lobe_right",
        "lung_lower_lobe_right",

        # pure_lung 제외용 장기 8개
        "heart",
        "aorta",
        "trachea",
        "esophagus",
        "liver",
        "stomach",
        "spleen",
        "pancreas",
    ],

    # 기존 build_organ_exclusion_mask 함수가 사용하는 이름
    "organ_exclusion_names": [
        "heart",
        "aorta",
        "trachea",
        "esophagus",
        "liver",
        "stomach",
        "spleen",
        "pancreas",
    ],

    "organ_exclusion_dilate_iter": 1,

    # ========================================================
    # TS lung guard 설정
    # ========================================================

    "use_ts_lung_guard": True,
    "ts_lung_guard_dilate_iter": 2,
    "strict_ts_lung_guard": True,

    # 흉곽 후보는 이번 버전에서 생성하지 않음
    "save_chestwall_candidate_mask": False,

    # body guard는 TotalSegmentator가 아니라 HU 기준 직접 생성이므로 유지
    "save_body_guard_mask": True,

    # ========================================================
    # Lung crop size 분석 설정
    # ========================================================

    "save_lung_crop_size_stats": True,
    "lung_crop_margin_pixels": 16,
    "lung_crop_size_multiple": 32,

    # ========================================================
    # PNG 저장 설정
    # ========================================================

    "save_png": False,

    # ========================================================
    # 폐 z-range 저장 설정
    # ========================================================

    "save_lung_range_volume": True,
    "lung_range_min_pure_lung_area_ratio": 0.01,
    "lung_range_margin_slices": 5,
    "lung_range_max_gap_slices": 5,
    "lung_range_min_segment_slices": 10,

    # ========================================================
    # 출력 제어
    # ========================================================

    "verbose": False,
    "max_missing_raw_warnings": 10,
}
args = SimpleNamespace(**CONFIG)

print("========== CONFIG ==========")
for k, v in CONFIG.items():
    print(f"{k}: {v}")


# ============================================================
# 2. 기본 함수
# ============================================================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_name(name: str) -> str:
    return (
        str(name)
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
    )


def find_totalseg_executable():
    for cmd in ["TotalSegmentator", "totalsegmentator"]:
        if shutil.which(cmd) is not None:
            return cmd

    raise RuntimeError(
        "TotalSegmentator 실행 파일을 찾지 못했음. "
        "Jupyter 커널이 Python (lungprep_ts)인지 확인해야 함."
    )


def list_dicom_patient_dirs(src_root: Path):
    if not src_root.exists():
        raise FileNotFoundError(f"src_root 없음: {src_root}")

    patient_dirs = sorted([p for p in src_root.iterdir() if p.is_dir()])
    rows = []

    for p in patient_dirs:
        dcm_count = len(list(p.rglob("*.dcm")))

        if dcm_count > 0:
            rows.append({
                "patient_id": safe_name(p.name),
                "patient_dir": p,
                "dcm_count": dcm_count,
            })

    return rows

def find_raw_from_mhd_header(mhd_path: Path) -> Path | None:
    """
    .mhd 파일 안의 ElementDataFile 항목을 읽어서 연결된 .raw 파일을 찾음.
    """

    text = mhd_path.read_text(encoding="utf-8", errors="ignore")

    raw_name = None

    for line in text.splitlines():
        line = line.strip()

        if line.startswith("ElementDataFile"):
            parts = line.split("=")

            if len(parts) >= 2:
                raw_name = parts[1].strip()
            break

    if raw_name is None or raw_name.upper() == "LOCAL":
        candidate = mhd_path.with_suffix(".raw")

        if candidate.exists():
            return candidate

        return None

    raw_path = mhd_path.parent / raw_name

    if raw_path.exists():
        return raw_path

    candidate = mhd_path.with_suffix(".raw")

    if candidate.exists():
        return candidate

    return None


def list_input_cases(src_roots, args):
    """
    DICOM 환자 폴더와 .mhd + .raw 쌍을 함께 찾음.
    .mhd와 .raw가 쌍으로 없으면 skip.
    """

    if isinstance(src_roots, (str, Path)):
        src_roots = [src_roots]

    rows = []
    missing_raw_list = []

    for root in src_roots:
        root = Path(root)

        if not root.exists():
            print(f"[WARN] src_root 없음: {root}")
            continue

        log_print(args, f"[SCAN] {root}", force=True)

        # -----------------------------
        # DICOM 폴더 찾기
        # -----------------------------
        for p in sorted([x for x in root.iterdir() if x.is_dir()]):
            dcm_files = list(p.rglob("*.dcm"))

            if len(dcm_files) > 0:
                rows.append({
                    "input_type": "dicom",
                    "patient_id": safe_name(p.name),
                    "patient_dir": p,
                    "input_path": p,
                    "raw_path": "",
                    "dcm_count": len(dcm_files),
                    "mhd_count": 0,
                    "source_root": root,
                })

        # -----------------------------
        # MHD + RAW 쌍 찾기
        # -----------------------------
        mhd_files = sorted(root.rglob("*.mhd"))

        for mhd_path in mhd_files:
            raw_path = find_raw_from_mhd_header(mhd_path)

            if raw_path is None:
                missing_raw_list.append(str(mhd_path))
                continue

            subset_name = safe_name(mhd_path.parent.name)
            case_name = safe_name(mhd_path.stem)

            patient_id = safe_name(f"{subset_name}_{case_name}")

            rows.append({
                "input_type": "mhd",
                "patient_id": patient_id,
                "patient_dir": mhd_path.parent,
                "input_path": mhd_path,
                "raw_path": raw_path,
                "dcm_count": 0,
                "mhd_count": 1,
                "source_root": root,
            })

    if len(missing_raw_list) > 0:
        print(f"[WARN] raw 파일이 없어 skip된 mhd 개수: {len(missing_raw_list)}")

        max_show = int(getattr(args, "max_missing_raw_warnings", 10))

        for p in missing_raw_list[:max_show]:
            print("  skip:", p)

        if len(missing_raw_list) > max_show:
            print(f"  ... 나머지 {len(missing_raw_list) - max_show}개 생략")

    if len(rows) == 0:
        raise RuntimeError("DICOM 또는 MHD/RAW 입력 케이스를 하나도 찾지 못함.")

    # patient_id 중복 방지
    seen = {}
    unique_rows = []

    for row in rows:
        pid = row["patient_id"]

        if pid not in seen:
            seen[pid] = 0
            unique_rows.append(row)
        else:
            seen[pid] += 1
            row = dict(row)
            row["patient_id"] = safe_name(f"{pid}_{seen[pid]}")
            unique_rows.append(row)

    print("[SCAN SUMMARY]")
    print("total input cases:", len(unique_rows))
    print("dicom cases:", sum(1 for r in unique_rows if r["input_type"] == "dicom"))
    print("mhd/raw cases:", sum(1 for r in unique_rows if r["input_type"] == "mhd"))

    return unique_rows

import time


def log_print(args, message, force=False):
    """
    verbose=False일 때는 중요한 출력만 보여줌.
    force=True면 항상 출력.
    """
    if force or bool(getattr(args, "verbose", True)):
        print(message)


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
# ============================================================
# 3. DICOM load / orientation / resampling
# ============================================================

def load_dicom_series_from_folder(series_dir: Path) -> sitk.Image:
    """
    환자 폴더 안 DICOM series 로드.
    여러 series가 있으면 slice 수가 가장 많은 series 선택.
    """

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(series_dir))

    if not series_ids:
        raise RuntimeError(f"DICOM series를 찾지 못함: {series_dir}")

    best_files = None
    best_series_id = None
    best_count = -1

    for sid in series_ids:
        files = reader.GetGDCMSeriesFileNames(str(series_dir), sid)

        if len(files) > best_count:
            best_files = files
            best_series_id = sid
            best_count = len(files)

    if best_files is None or len(best_files) == 0:
        raise RuntimeError(f"DICOM file 없음: {series_dir}")

    log_print(args, f"[DICOM] selected series id: {best_series_id}")
    log_print(args, f"[DICOM] number of slices: {len(best_files)}")

    reader.SetFileNames(best_files)
    return reader.Execute()
def load_mhd_raw_pair(mhd_path: Path) -> sitk.Image:
    """
    .mhd 파일을 읽음.
    .mhd 안에 ElementDataFile로 .raw가 연결되어 있으면
    SimpleITK가 자동으로 .raw까지 같이 읽음.
    """

    mhd_path = Path(mhd_path)

    if not mhd_path.exists():
        raise FileNotFoundError(f"MHD 파일 없음: {mhd_path}")

    raw_path = find_raw_from_mhd_header(mhd_path)

    if raw_path is None:
        raise FileNotFoundError(f"MHD와 연결된 RAW 파일을 찾지 못함: {mhd_path}")

    log_print(args, f"[MHD] mhd path: {mhd_path}")
    log_print(args, f"[MHD] raw path: {raw_path}")

    img = sitk.ReadImage(str(mhd_path))

    return img


def load_input_volume(case_info) -> sitk.Image:
    """
    input_type에 따라 DICOM 또는 MHD volume을 읽음.
    """

    input_type = case_info.get("input_type", "dicom")

    if input_type == "dicom":
        return load_dicom_series_from_folder(Path(case_info["input_path"]))

    if input_type == "mhd":
        return load_mhd_raw_pair(Path(case_info["input_path"]))

    raise ValueError(f"지원하지 않는 input_type: {input_type}")

def orient_image(img: sitk.Image, orientation: str) -> sitk.Image:
    return sitk.DICOMOrient(img, orientation)


def resample_z_only(img: sitk.Image, target_z: float, interpolator, default_value: float = 0.0) -> sitk.Image:
    """
    x/y spacing은 유지하고 z spacing만 target_z로 변경.
    CT는 sitkLinear 사용.
    """

    original_spacing = img.GetSpacing()
    original_size = img.GetSize()

    new_spacing = (
        float(original_spacing[0]),
        float(original_spacing[1]),
        float(target_z),
    )

    new_size_z = int(round(original_size[2] * original_spacing[2] / target_z))
    new_size_z = max(new_size_z, 1)

    new_size = (
        int(original_size[0]),
        int(original_size[1]),
        int(new_size_z),
    )

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)

    return resampler.Execute(img)


def resample_to_reference(moving: sitk.Image, reference: sitk.Image, interpolator, default_value: float = 0.0) -> sitk.Image:
    """
    moving image를 reference image grid에 맞춤.
    mask는 반드시 sitkNearestNeighbor 사용.
    """

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)

    return resampler.Execute(moving)


def build_original_slice_lookup(original_img: sitk.Image, resampled_img: sitk.Image):
    """
    1mm slice가 실제 원본 slice 위치인지,
    원본 slice 사이에서 보간된 slice인지 기록.
    """

    original_size = original_img.GetSize()
    resampled_size = resampled_img.GetSize()

    original_z_phys = []

    for oz in range(original_size[2]):
        p = original_img.TransformIndexToPhysicalPoint((0, 0, int(oz)))
        original_z_phys.append(float(p[2]))

    original_z_phys = np.array(original_z_phys, dtype=np.float32)

    lookup = {}

    for rz in range(resampled_size[2]):
        p = resampled_img.TransformIndexToPhysicalPoint((0, 0, int(rz)))
        rz_phys = float(p[2])

        nearest_idx = int(np.argmin(np.abs(original_z_phys - rz_phys)))
        nearest_z = float(original_z_phys[nearest_idx])
        dist_mm = float(abs(nearest_z - rz_phys))

        lookup[int(rz)] = {
            "nearest_original_slice_index": nearest_idx,
            "nearest_original_z_physical": nearest_z,
            "distance_to_original_slice_mm": dist_mm,
            "is_original_slice": int(dist_mm <= 0.05),
        }

    return lookup


def array_to_sitk_like(mask_arr: np.ndarray, reference_img: sitk.Image) -> sitk.Image:
    out = sitk.GetImageFromArray(mask_arr.astype(np.uint8))
    out.CopyInformation(reference_img)
    return out
# ============================================================
# Lung z-range crop functions
# ============================================================

def fill_small_false_gaps(valid_mask: np.ndarray, max_gap: int) -> np.ndarray:
    """
    valid_mask 안에서 짧게 끊긴 False 구간을 True로 메움.
    예:
        True True False False True True
        max_gap=2이면 중간 False False를 True로 바꿈.

    목적:
        폐가 이어지는 중간 slice가 잠깐 기준 아래로 떨어져도
        그 slice만 갑자기 빠지지 않게 하기 위함.
    """

    valid_mask = valid_mask.astype(bool).copy()

    if max_gap <= 0:
        return valid_mask

    n = len(valid_mask)
    i = 0

    while i < n:
        if valid_mask[i]:
            i += 1
            continue

        start = i

        while i < n and not valid_mask[i]:
            i += 1

        end = i - 1
        gap_len = end - start + 1

        has_true_before = start > 0 and valid_mask[start - 1]
        has_true_after = i < n and valid_mask[i]

        if has_true_before and has_true_after and gap_len <= max_gap:
            valid_mask[start:end + 1] = True

    return valid_mask


def find_true_segments(valid_mask: np.ndarray):
    """
    True가 연속되는 구간들을 (start, end)로 반환.
    """

    valid_mask = valid_mask.astype(bool)
    segments = []

    in_segment = False
    start = None

    for i, v in enumerate(valid_mask):
        if v and not in_segment:
            start = i
            in_segment = True

        if (not v) and in_segment:
            end = i - 1
            segments.append((start, end))
            in_segment = False

    if in_segment:
        segments.append((start, len(valid_mask) - 1))

    return segments


def find_lung_z_range_from_pure_lung(
    pure_lung: np.ndarray,
    min_area_ratio: float,
    margin_slices: int,
    max_gap_slices: int,
    min_segment_slices: int,
):
    """
    pure_lung mask를 기준으로 폐가 있는 z축 연속 구간을 찾음.

    중요한 점:
        중간 slice를 띄엄띄엄 제거하지 않음.
        최종 결과는 반드시 z_start ~ z_end까지 연속 구간임.
    """

    zdim, h, w = pure_lung.shape

    pure_area = pure_lung.sum(axis=(1, 2))
    pure_area_ratio = pure_area / float(h * w)

    valid = pure_area_ratio >= float(min_area_ratio)

    # 중간에 잠깐 끊긴 작은 gap 메움
    valid_filled = fill_small_false_gaps(
        valid_mask=valid,
        max_gap=int(max_gap_slices),
    )

    segments = find_true_segments(valid_filled)

    # 너무 짧은 구간 제거
    segments = [
        (s, e)
        for s, e in segments
        if (e - s + 1) >= int(min_segment_slices)
    ]

    if len(segments) == 0:
        # 폐 구간을 못 찾으면 전체 volume 유지
        return {
            "z_start": 0,
            "z_end": zdim - 1,
            "found_lung_range": 0,
            "reason": "no_valid_lung_segment_found",
            "pure_lung_area_ratio_per_slice": pure_area_ratio,
        }

    # 가장 중요한 폐 구간 선택
    # 기준: 해당 구간의 pure lung 총 면적이 가장 큰 구간
    best_segment = None
    best_score = -1

    for s, e in segments:
        score = float(pure_area[s:e + 1].sum())

        if score > best_score:
            best_score = score
            best_segment = (s, e)

    z_start, z_end = best_segment

    # 앞뒤 margin 추가
    z_start = max(0, int(z_start) - int(margin_slices))
    z_end = min(zdim - 1, int(z_end) + int(margin_slices))

    return {
        "z_start": int(z_start),
        "z_end": int(z_end),
        "found_lung_range": 1,
        "reason": "ok",
        "pure_lung_area_ratio_per_slice": pure_area_ratio,
    }


def crop_sitk_z_range(img: sitk.Image, z_start: int, z_end: int) -> sitk.Image:
    """
    SimpleITK image를 z_start~z_end 범위로 crop.
    x/y는 그대로 유지.
    """

    size = list(img.GetSize())  # x, y, z
    index = [0, 0, int(z_start)]

    crop_size = [
        int(size[0]),
        int(size[1]),
        int(z_end - z_start + 1),
    ]

    cropped = sitk.RegionOfInterest(
        img,
        size=crop_size,
        index=index,
    )

    return cropped

# ============================================================
# 4. TotalSegmentator native CT 실행
# ============================================================
def has_required_totalseg_masks(out_dir: Path, required_names):
    """
    TotalSegmentator 결과 폴더에 필요한 ROI mask가 모두 있는지 확인.
    기존 결과 재사용 시, 일부 mask만 있는 폴더를 잘못 재사용하지 않기 위함.
    """

    out_dir = Path(out_dir)

    if not out_dir.exists():
        return False

    for name in required_names:
        if not (out_dir / f"{name}.nii.gz").exists():
            return False

    return True
def run_totalsegmentator_native(ct_native_path: Path, out_dir: Path, args, patient_log_dir: Path):
    """
    원본 spacing CT에서 TotalSegmentator 실행.
    """

    if out_dir.exists() and not args.overwrite_totalseg:
        required_names = list(getattr(args, "organ_roi_subset", []))
    
        if has_required_totalseg_masks(out_dir, required_names):
            print(f"[TotalSegmentator] 기존 결과 재사용: {out_dir}")
            return
    
        print(f"[TotalSegmentator] 기존 결과가 있지만 필요한 ROI가 부족해서 다시 실행: {out_dir}")
        shutil.rmtree(out_dir)

    if out_dir.exists() and args.overwrite_totalseg:
        shutil.rmtree(out_dir)

    ensure_dir(out_dir)
    ensure_dir(patient_log_dir)

    exe = find_totalseg_executable()

    cmd = [
        exe,
        "-i", str(ct_native_path),
        "-o", str(out_dir),
        "--nr_thr_resamp", "1",
        "--nr_thr_saving", "1",
    ]

    if args.use_fast_totalseg:
        cmd.append("-f")

    if len(args.organ_roi_subset) > 0:
        cmd += ["--roi_subset"] + list(args.organ_roi_subset)

    log_print(args, "[TotalSegmentator] command")
    log_print(args, " ".join(cmd)) 

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    (patient_log_dir / "totalseg_stdout.txt").write_text(
        result.stdout or "",
        encoding="utf-8",
        errors="replace",
    )

    (patient_log_dir / "totalseg_stderr.txt").write_text(
        result.stderr or "",
        encoding="utf-8",
        errors="replace",
    )

    log_print(args, f"[TotalSegmentator] returncode: {result.returncode}")
    if bool(getattr(args, "print_totalseg_tail", False)):
        print("[stdout tail]")
        print((result.stdout or "")[-1500:])
        print("[stderr tail]")
        print((result.stderr or "")[-1500:])

    if result.returncode != 0:
        raise RuntimeError("TotalSegmentator 실행 실패")


def resample_organ_masks_to_1mm(totalseg_dir: Path, ct_1mm: sitk.Image, out_dir: Path):
    """
    원본 spacing TotalSegmentator mask를 1mm CT grid로 맞춰 저장.
    장기 하나당 3D NIfTI 하나.
    """

    ensure_dir(out_dir)

    mask_files = sorted(totalseg_dir.glob("*.nii.gz"))

    if len(mask_files) == 0:
        raise RuntimeError(f"TotalSegmentator 결과 mask 없음: {totalseg_dir}")

    rows = []

    for mask_path in mask_files:
        organ_name = mask_path.name.replace(".nii.gz", "")

        mask_native = sitk.ReadImage(str(mask_path))

        mask_1mm = resample_to_reference(
            moving=mask_native,
            reference=ct_1mm,
            interpolator=sitk.sitkNearestNeighbor,
            default_value=0,
        )

        out_path = out_dir / f"{organ_name}.nii.gz"
        sitk.WriteImage(mask_1mm, str(out_path))

        arr = sitk.GetArrayFromImage(mask_1mm) > 0

        rows.append({
            "organ_name": organ_name,
            "organ_mask_1mm_nii": str(out_path),
            "voxel_count": int(arr.sum()),
        })

    return rows


# ============================================================
# 5. HU / PNG
# ============================================================

def hu_to_uint8(slice_hu: np.ndarray, hu_min: int, hu_max: int) -> np.ndarray:
    x = np.clip(slice_hu.astype(np.float32), hu_min, hu_max)
    x = (x - hu_min) / float(hu_max - hu_min)
    x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return x


def save_gray_png(arr_uint8: np.ndarray, path: Path):
    ensure_dir(path.parent)
    Image.fromarray(arr_uint8).save(path)


def save_mask_png(mask_bool: np.ndarray, path: Path):
    ensure_dir(path.parent)
    Image.fromarray((mask_bool.astype(np.uint8) * 255)).save(path)


def save_overlay_png(ct_uint8: np.ndarray, refined_lung: np.ndarray, organ_exclusion: np.ndarray, pure_lung: np.ndarray, path: Path):
    """
    색상:
    - refined lung: 초록
    - organ exclusion: 빨강
    - pure lung: 노랑
    """

    ensure_dir(path.parent)

    rgb = np.stack([ct_uint8, ct_uint8, ct_uint8], axis=-1).astype(np.uint8)

    lung = refined_lung > 0
    organ = organ_exclusion > 0
    pure = pure_lung > 0

    rgb[lung, 1] = 220

    rgb[organ, 0] = 255
    rgb[organ, 1] = 40
    rgb[organ, 2] = 40

    rgb[pure, 0] = 255
    rgb[pure, 1] = 255
    rgb[pure, 2] = 40

    Image.fromarray(rgb).save(path)


# ============================================================
# 6. 세밀 폐 mask 생성
# ============================================================

def keep_largest_component(mask):
    mask = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    keep_id = int(np.argmax(areas) + 1)

    return (labels == keep_id).astype(np.uint8)


def keep_largest_two_components(mask):
    mask = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]

    if len(areas) == 1:
        keep = [1]
    else:
        keep = list(np.argsort(areas)[-2:] + 1)

    return np.isin(labels, keep).astype(np.uint8)


def fill_holes(mask):
    mask = (mask > 0).astype(np.uint8)

    if mask.sum() == 0:
        return mask

    flood = (mask * 255).copy()
    h, w = flood.shape
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(flood, ff_mask, (0, 0), 255)

    flood_inv = cv2.bitwise_not(flood)
    filled = (mask * 255) | flood_inv

    return (filled > 0).astype(np.uint8)


def remove_border_connected(mask):
    mask = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    h, w = mask.shape
    out = np.zeros_like(mask)

    for i in range(1, num):
        x, y, ww, hh, area = stats[i]

        touches_border = (
            x == 0
            or y == 0
            or x + ww >= w
            or y + hh >= h
        )

        if not touches_border:
            out[labels == i] = 1

    return out


def refined_lung_mask_2d_from_hu(slice_hu: np.ndarray, hu_min: int, hu_max: int) -> np.ndarray:
    """
    HU slice에서 폐 영역을 세밀하게 추출.
    TotalSegmentator 폐 mask를 쓰지 않고, HU 기반으로 폐 후보를 만듦.
    """

    arr = hu_to_uint8(slice_hu, hu_min, hu_max)

    blur = cv2.GaussianBlur(arr, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(blur)

    # body mask
    body = (arr > 5).astype(np.uint8)
    body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    body = cv2.morphologyEx(body, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    body = keep_largest_component(body)
    body = fill_holes(body)

    vals = eq[body > 0]

    if vals.size < 50:
        lung_thr = 90
    else:
        lung_thr, _ = cv2.threshold(
            vals.reshape(-1, 1).astype(np.uint8),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        lung_thr = int(np.clip(lung_thr, 55, 115))

    lung = ((eq <= lung_thr) & (body > 0)).astype(np.uint8)

    lung = remove_border_connected(lung)
    lung = cv2.morphologyEx(lung, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    lung = cv2.morphologyEx(lung, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    lung = fill_holes(lung)
    lung = keep_largest_two_components(lung)
    lung = fill_holes(lung)

    return lung.astype(bool)


def build_refined_lung_mask_3d(ct_arr: np.ndarray, args):
    zdim, h, w = ct_arr.shape
    lung = np.zeros((zdim, h, w), dtype=bool)

    iterator = tqdm(
        range(zdim),
        desc="Build refined lung mask",
        ncols=100,
        ascii=True,
        leave=False,
        disable=not bool(getattr(args, "verbose", False)),
    )
    
    for z in iterator:
        lung[z] = refined_lung_mask_2d_from_hu(
            ct_arr[z],
            args.hu_min,
            args.hu_max,
        )

    return lung


# ============================================================
# 7. organ exclusion / pure lung
# ============================================================

def build_organ_exclusion_mask(organ_rows, args):
    exclusion = None
    used_organs = []

    for row in organ_rows:
        organ_name = row["organ_name"]

        if organ_name not in args.organ_exclusion_names:
            continue

        mask_img = sitk.ReadImage(row["organ_mask_1mm_nii"])
        mask_arr = sitk.GetArrayFromImage(mask_img) > 0

        if exclusion is None:
            exclusion = np.zeros_like(mask_arr, dtype=bool)

        exclusion |= mask_arr
        used_organs.append(organ_name)

    if exclusion is None:
        raise RuntimeError("organ exclusion에 사용할 장기 mask가 없음.")

    if int(args.organ_exclusion_dilate_iter) > 0:
        exclusion = binary_dilation(
            exclusion,
            structure=np.ones((3, 3, 3), dtype=bool),
            iterations=int(args.organ_exclusion_dilate_iter),
        )

    return exclusion.astype(bool), used_organs
def build_union_mask_from_organ_rows(
    organ_rows,
    target_names,
    reference_shape=None,
    dilate_iter=0,
):
    """
    organ_rows에서 target_names에 해당하는 mask들을 합쳐서 하나의 union mask 생성.

    예:
        TS lung lobe 5개 합치기
        ribs/sternum/vertebrae 흉곽 후보 합치기
    """

    target_names = set(list(target_names))
    union = None
    used_names = []

    for row in organ_rows:
        organ_name = row["organ_name"]

        if organ_name not in target_names:
            continue

        mask_img = sitk.ReadImage(row["organ_mask_1mm_nii"])
        mask_arr = sitk.GetArrayFromImage(mask_img) > 0

        if union is None:
            union = np.zeros_like(mask_arr, dtype=bool)

        union |= mask_arr
        used_names.append(organ_name)

    if union is None:
        if reference_shape is None:
            return None, []

        union = np.zeros(reference_shape, dtype=bool)

    if int(dilate_iter) > 0 and union.sum() > 0:
        union = binary_dilation(
            union,
            structure=np.ones((3, 3, 3), dtype=bool),
            iterations=int(dilate_iter),
        )

    return union.astype(bool), used_names


def build_body_guard_mask_3d(ct_arr: np.ndarray, args):
    """
    HU 기준 body outer contour 생성.
    이건 정확한 흉벽 mask가 아니라, 몸통 외곽 확인/저장용.
    최종 pure_lung 제한은 TS lung guard가 담당.
    """

    zdim, h, w = ct_arr.shape
    body_3d = np.zeros((zdim, h, w), dtype=bool)

    hu_thr = float(getattr(args, "body_guard_hu_threshold", -500))

    for z in range(zdim):
        body = (ct_arr[z] > hu_thr).astype(np.uint8)

        body = cv2.morphologyEx(
            body,
            cv2.MORPH_CLOSE,
            np.ones((9, 9), np.uint8),
        )

        body = cv2.morphologyEx(
            body,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
        )

        body = keep_largest_component(body)
        body = fill_holes(body)

        body_3d[z] = body > 0

    return body_3d
def get_2d_bbox_from_mask(mask_2d: np.ndarray):
    """
    2D mask에서 bounding box를 구함.

    반환:
        has_mask, y_min, y_max, x_min, x_max, bbox_h, bbox_w
    """

    mask_2d = mask_2d > 0

    ys, xs = np.where(mask_2d)

    if len(xs) == 0 or len(ys) == 0:
        return {
            "has_mask": 0,
            "y_min": -1,
            "y_max": -1,
            "x_min": -1,
            "x_max": -1,
            "bbox_h": 0,
            "bbox_w": 0,
        }

    y_min = int(ys.min())
    y_max = int(ys.max())
    x_min = int(xs.min())
    x_max = int(xs.max())

    bbox_h = int(y_max - y_min + 1)
    bbox_w = int(x_max - x_min + 1)

    return {
        "has_mask": 1,
        "y_min": y_min,
        "y_max": y_max,
        "x_min": x_min,
        "x_max": x_max,
        "bbox_h": bbox_h,
        "bbox_w": bbox_w,
    }


def round_up_to_multiple(value: int, multiple: int = 32):
    """
    value를 multiple 배수로 올림.
    예:
        257 -> 288
        321 -> 352
    """

    value = int(value)
    multiple = int(multiple)

    if value <= 0:
        return 0

    return int(np.ceil(value / multiple) * multiple)


def choose_standard_square_size(value: int):
    """
    모델 입력으로 쓰기 좋은 정사각형 크기 후보 중 하나 선택.

    기준:
        128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512

    value보다 크거나 같은 가장 작은 크기를 선택.
    """

    candidates = [
        128,
        160,
        192,
        224,
        256,
        288,
        320,
        352,
        384,
        416,
        448,
        480,
        512,
    ]

    value = int(value)

    for s in candidates:
        if value <= s:
            return int(s)

    return int(round_up_to_multiple(value, 32))


def compute_lung_square_crop_stats_3d(
    mask_3d: np.ndarray,
    margin_pixels: int = 16,
    multiple: int = 32,
):
    """
    3D lung mask에서 slice별 폐 bounding box와 정사각형 crop 크기 계산.

    기준:
        - mask가 있는 slice만 계산
        - bbox_h, bbox_w 중 큰 값을 square_size로 사용
        - margin_pixels를 양쪽 여유로 추가
        - 32 배수로 올림
        - 모델용 표준 크기 후보도 같이 저장

    반환:
        slice_stats_df, summary
    """

    mask_3d = mask_3d > 0
    zdim, h, w = mask_3d.shape

    rows = []

    for z in range(zdim):
        bbox = get_2d_bbox_from_mask(mask_3d[z])

        bbox_h = int(bbox["bbox_h"])
        bbox_w = int(bbox["bbox_w"])

        raw_square = int(max(bbox_h, bbox_w))

        if bbox["has_mask"]:
            square_with_margin = int(raw_square + 2 * int(margin_pixels))
            square_with_margin = min(square_with_margin, max(h, w))
            square_32 = int(round_up_to_multiple(square_with_margin, multiple))
            square_32 = min(square_32, max(h, w))
            standard_square = int(choose_standard_square_size(square_with_margin))
            standard_square = min(standard_square, max(h, w))
        else:
            square_with_margin = 0
            square_32 = 0
            standard_square = 0

        rows.append({
            "slice_index": int(z),
            "has_lung_mask": int(bbox["has_mask"]),

            "lung_bbox_y_min": int(bbox["y_min"]),
            "lung_bbox_y_max": int(bbox["y_max"]),
            "lung_bbox_x_min": int(bbox["x_min"]),
            "lung_bbox_x_max": int(bbox["x_max"]),

            "lung_bbox_h": int(bbox_h),
            "lung_bbox_w": int(bbox_w),
            "lung_bbox_square_raw": int(raw_square),

            "lung_square_margin_pixels": int(margin_pixels),
            "lung_bbox_square_with_margin": int(square_with_margin),
            "lung_bbox_square_32": int(square_32),
            "lung_bbox_standard_square": int(standard_square),
        })

    df = pd.DataFrame(rows)

    valid_df = df[df["has_lung_mask"] == 1].copy()

    if len(valid_df) == 0:
        summary = {
            "lung_bbox_valid_slice_count": 0,
            "lung_bbox_max_h": 0,
            "lung_bbox_max_w": 0,
            "lung_bbox_max_square_raw": 0,
            "lung_bbox_max_square_with_margin": 0,
            "lung_bbox_max_square_32": 0,
            "lung_bbox_recommended_standard_square": 0,
            "lung_bbox_margin_pixels": int(margin_pixels),
        }

        return df, summary

    max_h = int(valid_df["lung_bbox_h"].max())
    max_w = int(valid_df["lung_bbox_w"].max())
    max_square_raw = int(valid_df["lung_bbox_square_raw"].max())
    max_square_with_margin = int(valid_df["lung_bbox_square_with_margin"].max())
    max_square_32 = int(valid_df["lung_bbox_square_32"].max())
    recommended_standard_square = int(
        choose_standard_square_size(max_square_with_margin)
    )

    recommended_standard_square = min(
        recommended_standard_square,
        int(max(h, w)),
    )

    summary = {
        "lung_bbox_valid_slice_count": int(len(valid_df)),
        "lung_bbox_max_h": int(max_h),
        "lung_bbox_max_w": int(max_w),
        "lung_bbox_max_square_raw": int(max_square_raw),
        "lung_bbox_max_square_with_margin": int(max_square_with_margin),
        "lung_bbox_max_square_32": int(max_square_32),
        "lung_bbox_recommended_standard_square": int(recommended_standard_square),
        "lung_bbox_margin_pixels": int(margin_pixels),
    }

    return df, summary
# ============================================================
# 8. output dirs
# ============================================================


def build_out_dirs(out_root: Path):
    dirs = {
        "ct_native_lps": out_root / "ct_native_lps",
        "ct_1mm_lps": out_root / "ct_1mm_lps",
        "totalseg_native": out_root / "totalseg_native",
        "organ_masks_1mm": out_root / "organ_masks_1mm",

        "lung_masks_1mm": out_root / "lung_masks_refined_1mm",
        "organ_exclusion_1mm": out_root / "organ_exclusion_1mm",
        "pure_lung_1mm": out_root / "pure_lung_1mm",

        "ct_1mm_lung_range": out_root / "ct_1mm_lung_range",
        "lung_masks_lung_range": out_root / "lung_masks_lung_range",
        "organ_exclusion_lung_range": out_root / "organ_exclusion_lung_range",
        "pure_lung_lung_range": out_root / "pure_lung_lung_range",

        "slices_png": out_root / "slices_png",
        "lung_png": out_root / "lung_png",
        "organ_exclusion_png": out_root / "organ_exclusion_png",
        "pure_lung_png": out_root / "pure_lung_png",
        "overlay_png": out_root / "overlay_png",

        "logs": out_root / "logs",

        "ts_lung_guard_1mm": out_root / "ts_lung_guard_1mm",
        "body_guard_1mm": out_root / "body_guard_1mm",

        "lung_crop_stats": out_root / "lung_crop_stats",
    }

    for p in dirs.values():
        ensure_dir(p)

    return dirs

# ============================================================
# 9. process one patient
# ============================================================
def process_one_patient(patient_info, args, out_dirs):
    patient_id = patient_info["patient_id"]
    patient_dir = Path(patient_info["patient_dir"])
    group = args.group_name

    log_print(args, "\n" + "=" * 100)
    log_print(args, f"[Patient] {patient_id}")
    log_print(args, f"[Folder] {patient_dir}")

    patient_log_dir = out_dirs["logs"] / group / patient_id
    ensure_dir(patient_log_dir)

    # ========================================================
    # 1. Input volume load
    # ========================================================

    input_type = patient_info.get("input_type", "dicom")
    input_path = Path(patient_info.get("input_path", patient_dir))

    log_print(args, "[INPUT]")
    log_print(args, f"input_type: {input_type}")
    log_print(args, f"input_path: {input_path}")

    ct_raw = load_input_volume(patient_info)

    raw_size = ct_raw.GetSize()
    raw_spacing = ct_raw.GetSpacing()

    log_print(args, "[RAW]")
    log_print(args, f"size: {raw_size}")
    log_print(args, f"spacing: {raw_spacing}")

    # ========================================================
    # 2. LPS orientation
    # ========================================================

    ct_native = orient_image(ct_raw, args.orientation)

    log_print(args, "[NATIVE LPS]")
    log_print(args, f"size: {ct_native.GetSize()}")
    log_print(args, f"spacing: {ct_native.GetSpacing()}")

    # ========================================================
    # 3. save native CT
    # ========================================================

    ct_native_path = out_dirs["ct_native_lps"] / group / f"{patient_id}_native_lps.nii.gz"
    ensure_dir(ct_native_path.parent)
    sitk.WriteImage(ct_native, str(ct_native_path))

    # ========================================================
    # 4. TotalSegmentator on native CT
    # ========================================================

    totalseg_dir = out_dirs["totalseg_native"] / group / patient_id

    run_totalsegmentator_native(
        ct_native_path=ct_native_path,
        out_dir=totalseg_dir,
        args=args,
        patient_log_dir=patient_log_dir,
    )

    # ========================================================
    # 5. CT z-only 1mm resampling
    # ========================================================

    ct_1mm = resample_z_only(
        img=ct_native,
        target_z=args.target_z,
        interpolator=sitk.sitkLinear,
        default_value=-1024,
    )

    log_print(args, "[1MM CT]")
    log_print(args, f"size: {ct_1mm.GetSize()}")
    log_print(args, f"spacing: {ct_1mm.GetSpacing()}")

    ct_1mm_path = out_dirs["ct_1mm_lps"] / group / f"{patient_id}_1mm_lps.nii.gz"
    ensure_dir(ct_1mm_path.parent)
    sitk.WriteImage(ct_1mm, str(ct_1mm_path))

    slice_lookup = build_original_slice_lookup(ct_native, ct_1mm)

    # ========================================================
    # 6. TotalSegmentator masks native -> 1mm
    # ========================================================

    organ_mask_dir = out_dirs["organ_masks_1mm"] / group / patient_id

    organ_rows = resample_organ_masks_to_1mm(
        totalseg_dir=totalseg_dir,
        ct_1mm=ct_1mm,
        out_dir=organ_mask_dir,
    )

    # ========================================================
    # 7. CT array
    # ========================================================

    ct_arr = sitk.GetArrayFromImage(ct_1mm).astype(np.float32)
    zdim, h, w = ct_arr.shape

    # ========================================================
    # 7-1. TS lung guard 생성
    # ========================================================

    ts_lung_guard, used_ts_lung_names = build_union_mask_from_organ_rows(
        organ_rows=organ_rows,
        target_names=args.ts_lung_roi_names,
        reference_shape=ct_arr.shape,
        dilate_iter=int(args.ts_lung_guard_dilate_iter),
    )

    if ts_lung_guard is None or ts_lung_guard.sum() == 0:
        message = f"TS lung guard 생성 실패: {patient_id}"

        if bool(args.strict_ts_lung_guard):
            raise RuntimeError(message)

        print("[WARN]", message)
        ts_lung_guard = np.ones(ct_arr.shape, dtype=bool)
        used_ts_lung_names = []

    # ========================================================
    # 7-2. body guard 생성
    # --------------------------------------------------------
    # 흉곽 후보는 만들지 않음.
    # body_guard는 TotalSegmentator가 아니라 CT HU 값으로 직접 생성.
    # ========================================================

    body_guard = build_body_guard_mask_3d(ct_arr, args)

    # ========================================================
    # 7-3. 기존 HU 기반 세밀 폐 추출
    # ========================================================

    refined_lung_raw = build_refined_lung_mask_3d(ct_arr, args)

    # ========================================================
    # 7-4. 핵심 변경
    # TS 폐 mask 자체를 최종 폐로 쓰지 않음.
    # 기존 HU 폐 추출 결과를 TS lung guard 안쪽으로만 제한.
    # ========================================================

    if bool(args.use_ts_lung_guard):
        refined_lung = refined_lung_raw & ts_lung_guard
    else:
        refined_lung = refined_lung_raw

    # ========================================================
    # 8. organ exclusion and pure lung
    # ========================================================

    organ_exclusion, used_organs = build_organ_exclusion_mask(
        organ_rows=organ_rows,
        args=args,
    )

    pure_lung = refined_lung & (~organ_exclusion)

    # ========================================================
    # 8-1. Lung crop size stats
    # --------------------------------------------------------
    # 3개 기준으로 crop 크기 통계 저장
    # 1) ts_lung_guard
    # 2) refined_lung
    # 3) pure_lung
    # ========================================================

    if bool(getattr(args, "save_lung_crop_size_stats", True)):
        crop_stats_dir = out_dirs["lung_crop_stats"] / group / patient_id
        ensure_dir(crop_stats_dir)

        crop_stats = {}

        crop_mask_dict = {
            "ts_lung_guard": ts_lung_guard,
            "refined_lung": refined_lung,
            "pure_lung": pure_lung,
        }

        for crop_name, crop_mask in crop_mask_dict.items():
            crop_slice_csv = crop_stats_dir / f"{crop_name}_crop_slice_stats.csv"
            crop_summary_json = crop_stats_dir / f"{crop_name}_crop_summary.json"

            crop_slice_df, crop_summary = compute_lung_square_crop_stats_3d(
                mask_3d=crop_mask,
                margin_pixels=int(getattr(args, "lung_crop_margin_pixels", 16)),
                multiple=int(getattr(args, "lung_crop_size_multiple", 32)),
            )

            crop_slice_df.to_csv(
                crop_slice_csv,
                index=False,
                encoding="utf-8-sig",
            )

            crop_summary["patient_id"] = patient_id
            crop_summary["crop_slice_csv"] = str(crop_slice_csv)
            crop_summary["mask_used_for_crop_stats"] = crop_name

            with open(crop_summary_json, "w", encoding="utf-8") as f:
                json.dump(crop_summary, f, indent=2, ensure_ascii=False)

            crop_stats[crop_name] = {
                "slice_df": crop_slice_df,
                "summary": crop_summary,
                "slice_csv": crop_slice_csv,
                "summary_json": crop_summary_json,
            }

        ts_crop_slice_df = crop_stats["ts_lung_guard"]["slice_df"]
        refined_crop_slice_df = crop_stats["refined_lung"]["slice_df"]
        pure_crop_slice_df = crop_stats["pure_lung"]["slice_df"]

        ts_crop_summary = crop_stats["ts_lung_guard"]["summary"]
        refined_crop_summary = crop_stats["refined_lung"]["summary"]
        pure_crop_summary = crop_stats["pure_lung"]["summary"]

        ts_crop_slice_csv = crop_stats["ts_lung_guard"]["slice_csv"]
        refined_crop_slice_csv = crop_stats["refined_lung"]["slice_csv"]
        pure_crop_slice_csv = crop_stats["pure_lung"]["slice_csv"]

        ts_crop_summary_json = crop_stats["ts_lung_guard"]["summary_json"]
        refined_crop_summary_json = crop_stats["refined_lung"]["summary_json"]
        pure_crop_summary_json = crop_stats["pure_lung"]["summary_json"]

    else:
        empty_crop_summary = {
            "lung_bbox_valid_slice_count": 0,
            "lung_bbox_max_h": 0,
            "lung_bbox_max_w": 0,
            "lung_bbox_max_square_raw": 0,
            "lung_bbox_max_square_with_margin": 0,
            "lung_bbox_max_square_32": 0,
            "lung_bbox_recommended_standard_square": 0,
            "lung_bbox_margin_pixels": int(getattr(args, "lung_crop_margin_pixels", 16)),
        }

        ts_crop_slice_df = None
        refined_crop_slice_df = None
        pure_crop_slice_df = None

        ts_crop_summary = dict(empty_crop_summary)
        refined_crop_summary = dict(empty_crop_summary)
        pure_crop_summary = dict(empty_crop_summary)

        ts_crop_slice_csv = ""
        refined_crop_slice_csv = ""
        pure_crop_slice_csv = ""

        ts_crop_summary_json = ""
        refined_crop_summary_json = ""
        pure_crop_summary_json = ""

    # ========================================================
    # 9. save masks NIfTI
    # ========================================================

    lung_dir = out_dirs["lung_masks_1mm"] / group / patient_id
    exclusion_dir = out_dirs["organ_exclusion_1mm"] / group / patient_id
    pure_dir = out_dirs["pure_lung_1mm"] / group / patient_id
    ts_lung_guard_dir = out_dirs["ts_lung_guard_1mm"] / group / patient_id
    body_guard_dir = out_dirs["body_guard_1mm"] / group / patient_id

    ensure_dir(lung_dir)
    ensure_dir(exclusion_dir)
    ensure_dir(pure_dir)
    ensure_dir(ts_lung_guard_dir)
    ensure_dir(body_guard_dir)

    lung_nii = lung_dir / "lung_refined_1mm.nii.gz"
    exclusion_nii = exclusion_dir / "organ_exclusion_1mm.nii.gz"
    pure_nii = pure_dir / "pure_lung_1mm.nii.gz"
    ts_lung_guard_nii = ts_lung_guard_dir / "ts_lung_guard_1mm.nii.gz"
    body_guard_nii = body_guard_dir / "body_guard_1mm.nii.gz"

    sitk.WriteImage(array_to_sitk_like(refined_lung, ct_1mm), str(lung_nii))
    sitk.WriteImage(array_to_sitk_like(organ_exclusion, ct_1mm), str(exclusion_nii))
    sitk.WriteImage(array_to_sitk_like(pure_lung, ct_1mm), str(pure_nii))
    sitk.WriteImage(array_to_sitk_like(ts_lung_guard, ct_1mm), str(ts_lung_guard_nii))
    sitk.WriteImage(array_to_sitk_like(body_guard, ct_1mm), str(body_guard_nii))

    # ========================================================
    # 9-1. 폐가 있는 z축 연속 구간만 모델용으로 따로 저장
    # ========================================================

    lung_range_info = find_lung_z_range_from_pure_lung(
        pure_lung=pure_lung,
        min_area_ratio=args.lung_range_min_pure_lung_area_ratio,
        margin_slices=args.lung_range_margin_slices,
        max_gap_slices=args.lung_range_max_gap_slices,
        min_segment_slices=args.lung_range_min_segment_slices,
    )

    lung_z_start = int(lung_range_info["z_start"])
    lung_z_end = int(lung_range_info["z_end"])

    ct_lung_range_dir = out_dirs["ct_1mm_lung_range"] / group
    lung_lung_range_dir = out_dirs["lung_masks_lung_range"] / group / patient_id
    exclusion_lung_range_dir = out_dirs["organ_exclusion_lung_range"] / group / patient_id
    pure_lung_range_dir = out_dirs["pure_lung_lung_range"] / group / patient_id

    ensure_dir(ct_lung_range_dir)
    ensure_dir(lung_lung_range_dir)
    ensure_dir(exclusion_lung_range_dir)
    ensure_dir(pure_lung_range_dir)

    ct_lung_range_nii = ct_lung_range_dir / f"{patient_id}_1mm_lung_range_lps.nii.gz"
    lung_lung_range_nii = lung_lung_range_dir / "lung_refined_1mm_lung_range.nii.gz"
    exclusion_lung_range_nii = exclusion_lung_range_dir / "organ_exclusion_1mm_lung_range.nii.gz"
    pure_lung_range_nii = pure_lung_range_dir / "pure_lung_1mm_lung_range.nii.gz"
    ts_lung_guard_lung_range_nii = ts_lung_guard_dir / "ts_lung_guard_1mm_lung_range.nii.gz"
    body_guard_lung_range_nii = body_guard_dir / "body_guard_1mm_lung_range.nii.gz"

    if bool(args.save_lung_range_volume):
        ct_lung_range_img = crop_sitk_z_range(
            img=ct_1mm,
            z_start=lung_z_start,
            z_end=lung_z_end,
        )

        lung_lung_range_img = crop_sitk_z_range(
            img=array_to_sitk_like(refined_lung, ct_1mm),
            z_start=lung_z_start,
            z_end=lung_z_end,
        )

        exclusion_lung_range_img = crop_sitk_z_range(
            img=array_to_sitk_like(organ_exclusion, ct_1mm),
            z_start=lung_z_start,
            z_end=lung_z_end,
        )

        pure_lung_range_img = crop_sitk_z_range(
            img=array_to_sitk_like(pure_lung, ct_1mm),
            z_start=lung_z_start,
            z_end=lung_z_end,
        )

        ts_lung_guard_lung_range_img = crop_sitk_z_range(
            img=array_to_sitk_like(ts_lung_guard, ct_1mm),
            z_start=lung_z_start,
            z_end=lung_z_end,
        )

        body_guard_lung_range_img = crop_sitk_z_range(
            img=array_to_sitk_like(body_guard, ct_1mm),
            z_start=lung_z_start,
            z_end=lung_z_end,
        )

        sitk.WriteImage(ct_lung_range_img, str(ct_lung_range_nii))
        sitk.WriteImage(lung_lung_range_img, str(lung_lung_range_nii))
        sitk.WriteImage(exclusion_lung_range_img, str(exclusion_lung_range_nii))
        sitk.WriteImage(pure_lung_range_img, str(pure_lung_range_nii))
        sitk.WriteImage(ts_lung_guard_lung_range_img, str(ts_lung_guard_lung_range_nii))
        sitk.WriteImage(body_guard_lung_range_img, str(body_guard_lung_range_nii))

    log_print(args, "[LUNG RANGE]")
    log_print(args, f"z_start: {lung_z_start}")
    log_print(args, f"z_end: {lung_z_end}")
    log_print(args, f"num_slices: {lung_z_end - lung_z_start + 1}")
    log_print(args, f"found: {lung_range_info['found_lung_range']}")
    log_print(args, f"reason: {lung_range_info['reason']}")

    # ========================================================
    # 10. save slice metadata / optional PNGs
    # ========================================================

    slice_rows = []

    use_crop_stats = bool(getattr(args, "save_lung_crop_size_stats", True))

    for z in range(zdim):
        z = int(z)
        info = slice_lookup[z]

        if args.save_png:
            slice_png = out_dirs["slices_png"] / group / patient_id / f"slice_{z:04d}.png"
            lung_png = out_dirs["lung_png"] / group / patient_id / f"slice_{z:04d}_lung.png"
            exclusion_png = out_dirs["organ_exclusion_png"] / group / patient_id / f"slice_{z:04d}_organ_exclusion.png"
            pure_png = out_dirs["pure_lung_png"] / group / patient_id / f"slice_{z:04d}_pure_lung.png"
            overlay_png = out_dirs["overlay_png"] / group / patient_id / f"slice_{z:04d}_overlay.png"

            ct_uint8 = hu_to_uint8(ct_arr[z], args.hu_min, args.hu_max)

            save_gray_png(ct_uint8, slice_png)
            save_mask_png(refined_lung[z], lung_png)
            save_mask_png(organ_exclusion[z], exclusion_png)
            save_mask_png(pure_lung[z], pure_png)
            save_overlay_png(
                ct_uint8,
                refined_lung[z],
                organ_exclusion[z],
                pure_lung[z],
                overlay_png,
            )

            slice_png = str(slice_png)
            lung_png = str(lung_png)
            exclusion_png = str(exclusion_png)
            pure_png = str(pure_png)
            overlay_png = str(overlay_png)
        else:
            slice_png = ""
            lung_png = ""
            exclusion_png = ""
            pure_png = ""
            overlay_png = ""

        row_dict = {
            "group": group,
            "patient_id": patient_id,
            "patient_dir": str(patient_dir),

            "slice_index": int(z),
            "slice_png": str(slice_png),
            "lung_png": str(lung_png),
            "organ_exclusion_png": str(exclusion_png),
            "pure_lung_png": str(pure_png),
            "overlay_png": str(overlay_png),

            "refined_lung_area": int(refined_lung[z].sum()),
            "organ_exclusion_area": int(organ_exclusion[z].sum()),
            "pure_lung_area": int(pure_lung[z].sum()),

            "refined_lung_raw_area": int(refined_lung_raw[z].sum()),
            "ts_lung_guard_area": int(ts_lung_guard[z].sum()),
            "body_guard_area": int(body_guard[z].sum()),

            "refined_lung_area_ratio": float(refined_lung[z].sum() / (h * w)),
            "refined_lung_raw_area_ratio": float(refined_lung_raw[z].sum() / (h * w)),
            "organ_exclusion_area_ratio": float(organ_exclusion[z].sum() / (h * w)),
            "pure_lung_area_ratio": float(pure_lung[z].sum() / (h * w)),
            "ts_lung_guard_area_ratio": float(ts_lung_guard[z].sum() / (h * w)),
            "body_guard_area_ratio": float(body_guard[z].sum() / (h * w)),

            "is_lung_range_slice": int(lung_z_start <= z <= lung_z_end),
            "lung_range_z_start": int(lung_z_start),
            "lung_range_z_end": int(lung_z_end),
            "lung_range_num_slices": int(lung_z_end - lung_z_start + 1),
            "lung_range_found": int(lung_range_info["found_lung_range"]),
            "lung_range_reason": lung_range_info["reason"],
            "lung_range_min_pure_lung_area_ratio": float(args.lung_range_min_pure_lung_area_ratio),
            "lung_range_margin_slices": int(args.lung_range_margin_slices),

            "ts_lung_bbox_h": int(ts_crop_slice_df.loc[z, "lung_bbox_h"]) if use_crop_stats else 0,
            "ts_lung_bbox_w": int(ts_crop_slice_df.loc[z, "lung_bbox_w"]) if use_crop_stats else 0,
            "ts_lung_bbox_square_raw": int(ts_crop_slice_df.loc[z, "lung_bbox_square_raw"]) if use_crop_stats else 0,
            "ts_lung_bbox_square_with_margin": int(ts_crop_slice_df.loc[z, "lung_bbox_square_with_margin"]) if use_crop_stats else 0,
            "ts_lung_bbox_square_32": int(ts_crop_slice_df.loc[z, "lung_bbox_square_32"]) if use_crop_stats else 0,
            "ts_lung_bbox_standard_square": int(ts_crop_slice_df.loc[z, "lung_bbox_standard_square"]) if use_crop_stats else 0,

            "patient_ts_lung_bbox_max_h": int(ts_crop_summary["lung_bbox_max_h"]),
            "patient_ts_lung_bbox_max_w": int(ts_crop_summary["lung_bbox_max_w"]),
            "patient_ts_lung_bbox_max_square_raw": int(ts_crop_summary["lung_bbox_max_square_raw"]),
            "patient_ts_lung_bbox_max_square_with_margin": int(ts_crop_summary["lung_bbox_max_square_with_margin"]),
            "patient_ts_lung_bbox_max_square_32": int(ts_crop_summary["lung_bbox_max_square_32"]),
            "patient_ts_lung_bbox_recommended_standard_square": int(ts_crop_summary["lung_bbox_recommended_standard_square"]),

            "ts_lung_crop_slice_stats_csv": str(ts_crop_slice_csv),
            "ts_lung_crop_summary_json": str(ts_crop_summary_json),

            "refined_lung_bbox_h": int(refined_crop_slice_df.loc[z, "lung_bbox_h"]) if use_crop_stats else 0,
            "refined_lung_bbox_w": int(refined_crop_slice_df.loc[z, "lung_bbox_w"]) if use_crop_stats else 0,
            "refined_lung_bbox_square_raw": int(refined_crop_slice_df.loc[z, "lung_bbox_square_raw"]) if use_crop_stats else 0,
            "refined_lung_bbox_square_with_margin": int(refined_crop_slice_df.loc[z, "lung_bbox_square_with_margin"]) if use_crop_stats else 0,
            "refined_lung_bbox_square_32": int(refined_crop_slice_df.loc[z, "lung_bbox_square_32"]) if use_crop_stats else 0,
            "refined_lung_bbox_standard_square": int(refined_crop_slice_df.loc[z, "lung_bbox_standard_square"]) if use_crop_stats else 0,

            "patient_refined_lung_bbox_max_h": int(refined_crop_summary["lung_bbox_max_h"]),
            "patient_refined_lung_bbox_max_w": int(refined_crop_summary["lung_bbox_max_w"]),
            "patient_refined_lung_bbox_max_square_raw": int(refined_crop_summary["lung_bbox_max_square_raw"]),
            "patient_refined_lung_bbox_max_square_with_margin": int(refined_crop_summary["lung_bbox_max_square_with_margin"]),
            "patient_refined_lung_bbox_max_square_32": int(refined_crop_summary["lung_bbox_max_square_32"]),
            "patient_refined_lung_bbox_recommended_standard_square": int(refined_crop_summary["lung_bbox_recommended_standard_square"]),

            "refined_lung_crop_slice_stats_csv": str(refined_crop_slice_csv),
            "refined_lung_crop_summary_json": str(refined_crop_summary_json),

            "pure_lung_bbox_h": int(pure_crop_slice_df.loc[z, "lung_bbox_h"]) if use_crop_stats else 0,
            "pure_lung_bbox_w": int(pure_crop_slice_df.loc[z, "lung_bbox_w"]) if use_crop_stats else 0,
            "pure_lung_bbox_square_raw": int(pure_crop_slice_df.loc[z, "lung_bbox_square_raw"]) if use_crop_stats else 0,
            "pure_lung_bbox_square_with_margin": int(pure_crop_slice_df.loc[z, "lung_bbox_square_with_margin"]) if use_crop_stats else 0,
            "pure_lung_bbox_square_32": int(pure_crop_slice_df.loc[z, "lung_bbox_square_32"]) if use_crop_stats else 0,
            "pure_lung_bbox_standard_square": int(pure_crop_slice_df.loc[z, "lung_bbox_standard_square"]) if use_crop_stats else 0,

            "patient_pure_lung_bbox_max_h": int(pure_crop_summary["lung_bbox_max_h"]),
            "patient_pure_lung_bbox_max_w": int(pure_crop_summary["lung_bbox_max_w"]),
            "patient_pure_lung_bbox_max_square_raw": int(pure_crop_summary["lung_bbox_max_square_raw"]),
            "patient_pure_lung_bbox_max_square_with_margin": int(pure_crop_summary["lung_bbox_max_square_with_margin"]),
            "patient_pure_lung_bbox_max_square_32": int(pure_crop_summary["lung_bbox_max_square_32"]),
            "patient_pure_lung_bbox_recommended_standard_square": int(pure_crop_summary["lung_bbox_recommended_standard_square"]),

            "pure_lung_crop_slice_stats_csv": str(pure_crop_slice_csv),
            "pure_lung_crop_summary_json": str(pure_crop_summary_json),

            "nearest_original_slice_index": info["nearest_original_slice_index"],
            "nearest_original_z_physical": info["nearest_original_z_physical"],
            "distance_to_original_slice_mm": info["distance_to_original_slice_mm"],
            "is_original_slice": info["is_original_slice"],

            "ct_native_lps_nii": str(ct_native_path),
            "ct_1mm_lps_nii": str(ct_1mm_path),

            "lung_refined_1mm_nii": str(lung_nii),
            "organ_exclusion_1mm_nii": str(exclusion_nii),
            "pure_lung_1mm_nii": str(pure_nii),

            "ts_lung_guard_1mm_nii": str(ts_lung_guard_nii),
            "body_guard_1mm_nii": str(body_guard_nii),

            "ct_1mm_lung_range_nii": str(ct_lung_range_nii),
            "lung_refined_lung_range_nii": str(lung_lung_range_nii),
            "organ_exclusion_lung_range_nii": str(exclusion_lung_range_nii),
            "pure_lung_lung_range_nii": str(pure_lung_range_nii),
            "ts_lung_guard_lung_range_nii": str(ts_lung_guard_lung_range_nii),
            "body_guard_lung_range_nii": str(body_guard_lung_range_nii),

            "organ_mask_dir": str(organ_mask_dir),
            "used_organs_for_exclusion": "|".join(used_organs),
            "used_ts_lung_names": "|".join(used_ts_lung_names),

            "raw_size": str(raw_size),
            "raw_spacing": str(raw_spacing),
            "one_mm_size": str(ct_1mm.GetSize()),
            "one_mm_spacing": str(ct_1mm.GetSpacing()),
        }

        slice_rows.append(row_dict)

    for row in organ_rows:
        row["group"] = group
        row["patient_id"] = patient_id
        row["patient_dir"] = str(patient_dir)

    log_print(args, f"[DONE] {patient_id}")
    log_print(args, f"slices: {len(slice_rows)}")
    log_print(args, f"organs: {len(organ_rows)}")

    return slice_rows, organ_rows


# ============================================================
# 10. run
# ============================================================
src_roots = getattr(args, "src_roots", None)

if src_roots is None:
    src_roots = [args.src_root]

out_root = Path(args.out_root)

ensure_dir(out_root)
out_dirs = build_out_dirs(out_root)

with open(out_root / "final_preprocess_config.json", "w", encoding="utf-8") as f:
    json.dump(CONFIG, f, indent=2, ensure_ascii=False)

all_patients = list_input_cases(src_roots, args)

print("\n========== Patient pool ==========")
print("valid patient count:", len(all_patients))

if getattr(args, "process_all_cases", False):
    selected_patients = all_patients
else:
    if len(all_patients) < args.num_random_patients:
        raise RuntimeError("선택할 환자 수가 부족함.")

    random.seed(args.random_seed)
    selected_patients = random.sample(all_patients, args.num_random_patients)

print("\n========== Selected patients ==========")
print("selected count:", len(selected_patients))
print("dicom selected:", sum(1 for p in selected_patients if p["input_type"] == "dicom"))
print("mhd selected:", sum(1 for p in selected_patients if p["input_type"] == "mhd"))

selected_df = pd.DataFrame(selected_patients)

for col in ["patient_dir", "input_path", "raw_path", "source_root"]:
    if col in selected_df.columns:
        selected_df[col] = selected_df[col].astype(str)

selected_df.to_csv(
    out_root / "selected_patients.csv",
    index=False,
    encoding="utf-8-sig",
)

all_slice_rows = []
all_organ_rows = []
error_rows = []

global_start_time = time.perf_counter()
runtime_rows = []

total_cases = len(selected_patients)

for case_idx, p in enumerate(
    tqdm(
        selected_patients,
        desc="Final preprocessing",
        total=total_cases,
        ncols=100,
        ascii=True,
        leave=True,
        dynamic_ncols=False,
    ),
    start=1
):
    patient_start_time = time.perf_counter()

    try:
        slice_rows, organ_rows = process_one_patient(
            patient_info=p,
            args=args,
            out_dirs=out_dirs,
        )
        all_slice_rows.extend(slice_rows)
        all_organ_rows.extend(organ_rows)
        
        pd.DataFrame(all_slice_rows).to_csv(
            out_root / "metadata_slices.csv",
            index=False,
            encoding="utf-8-sig",
        )
        
        pd.DataFrame(all_organ_rows).to_csv(
            out_root / "metadata_organs.csv",
            index=False,
            encoding="utf-8-sig",
        )
        patient_elapsed = time.perf_counter() - patient_start_time
        total_elapsed = time.perf_counter() - global_start_time

        avg_per_case = total_elapsed / case_idx
        remain_cases = total_cases - case_idx
        eta_seconds = avg_per_case * remain_cases

        runtime_rows.append({
            "patient_id": p["patient_id"],
            "input_type": p.get("input_type", "unknown"),
            "case_index": case_idx,
            "total_cases": total_cases,
            "status": "success",
            "elapsed_seconds": round(patient_elapsed, 2),
            "elapsed_readable": format_seconds(patient_elapsed),
            "total_elapsed_seconds": round(total_elapsed, 2),
            "total_elapsed_readable": format_seconds(total_elapsed),
            "estimated_remaining_seconds": round(eta_seconds, 2),
            "estimated_remaining_readable": format_seconds(eta_seconds),
            "slice_count": len(slice_rows),
            "organ_count": len(organ_rows),
        })

        print(
            f"[TIME] {p['patient_id']} 완료 "
            f"({case_idx}/{total_cases}) | "
            f"이번 환자: {format_seconds(patient_elapsed)} | "
            f"전체 경과: {format_seconds(total_elapsed)} | "
            f"예상 남은 시간: {format_seconds(eta_seconds)}"
        )

        pd.DataFrame(runtime_rows).to_csv(
            out_root / "runtime_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    except Exception as e:
        print("\n[ERROR]")
        print("patient:", p)
        print("error:", str(e))
        traceback.print_exc()

        error_rows.append({
            "patient_id": p["patient_id"],
            "patient_dir": str(p["patient_dir"]),
            "error": str(e),
        })

        pd.DataFrame(error_rows).to_csv(
            out_root / "preprocess_errors.csv",
            index=False,
            encoding="utf-8-sig",
        )
        patient_elapsed = time.perf_counter() - patient_start_time
        total_elapsed = time.perf_counter() - global_start_time

        runtime_rows.append({
            "patient_id": p["patient_id"],
            "input_type": p.get("input_type", "unknown"),
            "case_index": case_idx,
            "total_cases": total_cases,
            "status": "error",
            "elapsed_seconds": round(patient_elapsed, 2),
            "elapsed_readable": format_seconds(patient_elapsed),
            "total_elapsed_seconds": round(total_elapsed, 2),
            "total_elapsed_readable": format_seconds(total_elapsed),
            "estimated_remaining_seconds": "",
            "estimated_remaining_readable": "",
            "slice_count": 0,
            "organ_count": 0,
        })

        pd.DataFrame(runtime_rows).to_csv(
            out_root / "runtime_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

slice_df = pd.DataFrame(all_slice_rows)
organ_df = pd.DataFrame(all_organ_rows)

slice_df.to_csv(out_root / "metadata_slices.csv", index=False, encoding="utf-8-sig")
organ_df.to_csv(out_root / "metadata_organs.csv", index=False, encoding="utf-8-sig")

if len(error_rows) > 0:
    pd.DataFrame(error_rows).to_csv(out_root / "preprocess_errors.csv", index=False, encoding="utf-8-sig")

print("\n========== FINISHED ==========")
print("out_root:", out_root)
print("slice rows:", len(slice_df))
print("organ rows:", len(organ_df))
print("errors:", len(error_rows))

print("\nmetadata_slices:", out_root / "metadata_slices.csv")
print("metadata_organs:", out_root / "metadata_organs.csv")

if len(slice_df) > 0:
    print("\nSlice count by patient")
    display(slice_df.groupby("patient_id").size())

    print("\nOriginal/interpolated slice count")
    display(slice_df["is_original_slice"].value_counts())

    print("\nArea ratio summary")
    display(slice_df[[
        "refined_lung_area_ratio",
        "organ_exclusion_area_ratio",
        "pure_lung_area_ratio",
    ]].describe())

if len(organ_df) > 0:
    print("\nOrgan mask count by patient")
    display(organ_df.groupby("patient_id").size())

# %%==== CELL ====

# ============================================================
# Final preprocessing QA check + patient overlay preview
# ------------------------------------------------------------
# 목적:
#   1. 전처리 결과 기본 파일 확인
#   2. metadata 필수 컬럼 확인
#   3. NIfTI shape 일치 확인
#   4. 환자별 대표 slice 1장 선택
#   5. CT + pure_lung + ts_lung_guard + body_guard overlay 저장
#
# 색상:
#   - 노랑: pure_lung
#   - 하늘색: ts_lung_guard
#   - 연두색: body_guard
#
# 중요:
#   - no-chest 버전에서는 chestwall_candidate를 확인하지 않음
#   - 전처리 본체에 섞지 말고, 전처리 완료 후 별도 셀로 실행
# ============================================================

from pathlib import Path
import json
import math

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image, ImageDraw

try:
    from IPython.display import display, Image as IPyImage
    CAN_DISPLAY = True
except Exception:
    CAN_DISPLAY = False


# ============================================================
# 1. CONFIG
# ============================================================

# 둘 중 실제 존재하는 폴더를 자동 선택함.
# 최종 no-chest 버전이면 첫 번째가 선택되는 게 정상.
CANDIDATE_OUT_ROOTS = [
    Path(r"E:\jyp\ct_data_2d_preprocessed\Normal_LUNA16_final_all_nonfast_v2_tslungguard_nochest"),
    Path(r"E:\jyp\ct_data_2d_preprocessed\Normal_LUNA16_final_all_nonfast_v2_tslungguard"),
]

# overlay 저장 폴더 이름
QA_OUT_DIR_NAME = "qa_overlay_pure_lung_ts_guard"

# Jupyter에서 바로 보여줄 이미지 개수
# 환자 362명 전체를 한 번에 display하면 너무 많을 수 있음.
# 전부 보고 싶으면 None으로 바꾸면 됨.
DISPLAY_LIMIT = 20

# CT window
HU_MIN = -1000
HU_MAX = 400

# overlay 투명도
MASK_ALPHA = 0.35
CONTOUR_THICKNESS = 2

# 환자별 대표 slice 선택 기준
# "max_pure_lung_area" 권장
SLICE_SELECT_MODE = "max_pure_lung_area"


# ============================================================
# 2. helper functions
# ============================================================

def pick_existing_out_root(candidate_roots):
    existing = [p for p in candidate_roots if p.exists()]

    if len(existing) == 0:
        print("확인한 후보 폴더:")
        for p in candidate_roots:
            print(" ", p, "exists:", p.exists())

        raise FileNotFoundError(
            "전처리 결과 OUT_ROOT를 찾지 못함. "
            "CANDIDATE_OUT_ROOTS에 실제 결과 폴더를 넣어야 함."
        )

    return existing[0]


def read_nii_array(path):
    path = Path(str(path))

    if not path.exists():
        raise FileNotFoundError(f"NIfTI 파일 없음: {path}")

    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)

    return img, arr


def hu_to_uint8(slice_hu, hu_min=-1000, hu_max=400):
    x = slice_hu.astype(np.float32)
    x = np.clip(x, hu_min, hu_max)
    x = (x - hu_min) / float(hu_max - hu_min)
    x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return x


def draw_mask_fill(rgb, mask, color, alpha=0.35):
    """
    mask 내부를 반투명 색으로 칠함.
    """
    mask = mask > 0

    if mask.sum() == 0:
        return rgb

    color_arr = np.array(color, dtype=np.float32)
    rgb_float = rgb.astype(np.float32)

    rgb_float[mask] = (
        rgb_float[mask] * (1.0 - alpha)
        + color_arr * alpha
    )

    return np.clip(rgb_float, 0, 255).astype(np.uint8)


def draw_mask_contour(rgb, mask, color, thickness=2):
    """
    mask 외곽선을 그림.
    """
    mask_u8 = (mask > 0).astype(np.uint8) * 255

    if mask_u8.sum() == 0:
        return rgb

    contours, _ = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    out = rgb.copy()

    cv2.drawContours(
        out,
        contours,
        contourIdx=-1,
        color=tuple(int(c) for c in color),
        thickness=int(thickness),
    )

    return out


def add_text_bar(rgb, lines):
    """
    이미지 위에 검은 bar를 만들고 설명 텍스트를 넣음.
    """
    h, w = rgb.shape[:2]
    bar_h = 58

    canvas = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    canvas[bar_h:, :, :] = rgb

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)

    y = 6

    for line in lines:
        draw.text((8, y), str(line), fill=(255, 255, 255))
        y += 20

    return np.array(img)


def make_overlay_image(
    ct_slice,
    pure_slice,
    guard_slice,
    body_slice=None,
    patient_id="",
    z_index=None,
    out_path=None,
):
    """
    CT 위에 pure_lung, ts_lung_guard, body_guard를 올림.
    """
    ct_u8 = hu_to_uint8(ct_slice, HU_MIN, HU_MAX)
    rgb = np.stack([ct_u8, ct_u8, ct_u8], axis=-1)

    # 색상
    pure_color = (255, 255, 0)      # 노랑
    guard_color = (0, 220, 255)     # 하늘색
    body_color = (120, 255, 80)     # 연두색

    # pure_lung은 내부를 살짝 칠하고 외곽도 그림
    rgb = draw_mask_fill(
        rgb=rgb,
        mask=pure_slice,
        color=pure_color,
        alpha=MASK_ALPHA,
    )
    rgb = draw_mask_contour(
        rgb=rgb,
        mask=pure_slice,
        color=pure_color,
        thickness=CONTOUR_THICKNESS,
    )

    # ts_lung_guard는 외곽선 중심
    rgb = draw_mask_contour(
        rgb=rgb,
        mask=guard_slice,
        color=guard_color,
        thickness=CONTOUR_THICKNESS,
    )

    # body_guard는 있으면 외곽선만 표시
    if body_slice is not None:
        rgb = draw_mask_contour(
            rgb=rgb,
            mask=body_slice,
            color=body_color,
            thickness=1,
        )

    text_lines = [
        f"patient_id: {patient_id}",
        f"slice_index: {z_index} | yellow=pure_lung | cyan=ts_lung_guard | lime=body_guard",
    ]

    rgb = add_text_bar(rgb, text_lines)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(out_path)

    return rgb


def choose_representative_slice(patient_df):
    """
    환자별 대표 slice 1개 선택.
    기본은 pure_lung_area가 가장 큰 slice.
    """
    df = patient_df.copy()

    if SLICE_SELECT_MODE == "max_pure_lung_area":
        if "pure_lung_area" in df.columns:
            df = df.sort_values(
                ["pure_lung_area", "ts_lung_guard_area"],
                ascending=False,
            )
            return df.iloc[0]

    # fallback
    if "is_lung_range_slice" in df.columns:
        lung_range_df = df[df["is_lung_range_slice"] == 1].copy()

        if len(lung_range_df) > 0:
            mid_idx = len(lung_range_df) // 2
            return lung_range_df.iloc[mid_idx]

    mid_idx = len(df) // 2
    return df.iloc[mid_idx]


def check_path_column_exists(meta, col):
    if col not in meta.columns:
        return False

    values = meta[col].dropna().astype(str)

    if len(values) == 0:
        return False

    first_value = values.iloc[0]

    if first_value.strip() == "":
        return False

    return Path(first_value).exists()


# ============================================================
# 3. OUT_ROOT and required files check
# ============================================================

OUT_ROOT = pick_existing_out_root(CANDIDATE_OUT_ROOTS)
QA_OUT_DIR = OUT_ROOT / QA_OUT_DIR_NAME

print("========== OUT ROOT ==========")
print("OUT_ROOT:", OUT_ROOT)
print("OUT_ROOT exists:", OUT_ROOT.exists())
print("QA_OUT_DIR:", QA_OUT_DIR)

required_files = [
    OUT_ROOT / "metadata_slices.csv",
    OUT_ROOT / "metadata_organs.csv",
    OUT_ROOT / "runtime_summary.csv",
    OUT_ROOT / "selected_patients.csv",
    OUT_ROOT / "final_preprocess_config.json",
]

print("\n========== FILE CHECK ==========")

missing_files = []

for p in required_files:
    exists = p.exists()
    print(p.name, ":", exists)

    if not exists:
        missing_files.append(str(p))

if len(missing_files) > 0:
    raise FileNotFoundError(
        "필수 파일이 없음:\n" + "\n".join(missing_files)
    )


# ============================================================
# 4. load csv
# ============================================================

meta = pd.read_csv(OUT_ROOT / "metadata_slices.csv")
organs = pd.read_csv(OUT_ROOT / "metadata_organs.csv")
runtime = pd.read_csv(OUT_ROOT / "runtime_summary.csv")

print("\n========== BASIC CHECK ==========")
print("metadata rows:", len(meta))
print("organ rows:", len(organs))
print("runtime rows:", len(runtime))
print("patients:", meta["patient_id"].nunique())

print("\nruntime status counts:")
display(runtime["status"].value_counts())


# ============================================================
# 5. config check
# ============================================================

config_path = OUT_ROOT / "final_preprocess_config.json"

with open(config_path, "r", encoding="utf-8") as f:
    saved_config = json.load(f)

print("\n========== CONFIG CHECK ==========")
print("saved out_root:", saved_config.get("out_root", ""))
print("use_fast_totalseg:", saved_config.get("use_fast_totalseg", ""))
print("save_png:", saved_config.get("save_png", ""))
print("use_ts_lung_guard:", saved_config.get("use_ts_lung_guard", ""))
print("ts_lung_guard_dilate_iter:", saved_config.get("ts_lung_guard_dilate_iter", ""))
print("save_chestwall_candidate_mask:", saved_config.get("save_chestwall_candidate_mask", ""))
print("save_body_guard_mask:", saved_config.get("save_body_guard_mask", ""))


# ============================================================
# 6. required column check
# ============================================================

print("\n========== REQUIRED COLUMN CHECK ==========")

required_cols = [
    "patient_id",
    "slice_index",

    "ts_lung_guard_area",
    "refined_lung_area",
    "pure_lung_area",
    "body_guard_area",

    "refined_lung_raw_area_ratio",
    "ts_lung_guard_area_ratio",
    "refined_lung_area_ratio",
    "organ_exclusion_area_ratio",
    "pure_lung_area_ratio",
    "body_guard_area_ratio",

    "ts_lung_bbox_square_with_margin",
    "refined_lung_bbox_square_with_margin",
    "pure_lung_bbox_square_with_margin",

    "patient_ts_lung_bbox_max_square_with_margin",
    "patient_refined_lung_bbox_max_square_with_margin",
    "patient_pure_lung_bbox_max_square_with_margin",

    "patient_ts_lung_bbox_recommended_standard_square",
    "patient_refined_lung_bbox_recommended_standard_square",
    "patient_pure_lung_bbox_recommended_standard_square",

    "ts_lung_crop_slice_stats_csv",
    "refined_lung_crop_slice_stats_csv",
    "pure_lung_crop_slice_stats_csv",

    "ts_lung_crop_summary_json",
    "refined_lung_crop_summary_json",
    "pure_lung_crop_summary_json",

    "ct_1mm_lps_nii",
    "lung_refined_1mm_nii",
    "organ_exclusion_1mm_nii",
    "pure_lung_1mm_nii",
    "ts_lung_guard_1mm_nii",
    "body_guard_1mm_nii",

    "ct_1mm_lung_range_nii",
    "lung_refined_lung_range_nii",
    "organ_exclusion_lung_range_nii",
    "pure_lung_lung_range_nii",
    "ts_lung_guard_lung_range_nii",
    "body_guard_lung_range_nii",
]

missing_cols = [c for c in required_cols if c not in meta.columns]

if len(missing_cols) == 0:
    print("필수 컬럼 있음: OK")
else:
    print("필수 컬럼 누락:")
    for c in missing_cols:
        print(" ", c)


# ============================================================
# 7. area summary
# ============================================================

print("\n========== AREA SUMMARY ==========")

area_cols = [
    "refined_lung_raw_area_ratio",
    "ts_lung_guard_area_ratio",
    "refined_lung_area_ratio",
    "organ_exclusion_area_ratio",
    "pure_lung_area_ratio",
    "body_guard_area_ratio",
]

existing_area_cols = [c for c in area_cols if c in meta.columns]

display(meta[existing_area_cols].describe())


# ============================================================
# 8. crop size summary
# ============================================================

print("\n========== CROP SIZE SUMMARY ==========")

crop_summary_cols = {
    "patient_ts_lung_bbox_max_square_with_margin": "max",
    "patient_refined_lung_bbox_max_square_with_margin": "max",
    "patient_pure_lung_bbox_max_square_with_margin": "max",
    "patient_ts_lung_bbox_recommended_standard_square": "max",
    "patient_refined_lung_bbox_recommended_standard_square": "max",
    "patient_pure_lung_bbox_recommended_standard_square": "max",
}

existing_crop_summary_cols = {
    k: v
    for k, v in crop_summary_cols.items()
    if k in meta.columns
}

if len(existing_crop_summary_cols) > 0:
    patient_crop = (
        meta
        .groupby("patient_id")
        .agg(existing_crop_summary_cols)
        .reset_index()
    )

    display(patient_crop)
else:
    patient_crop = pd.DataFrame()
    print("crop summary 컬럼이 없음")


# ============================================================
# 9. crop file check
# ============================================================

print("\n========== CROP FILE CHECK ==========")

crop_file_cols = [
    "ts_lung_crop_slice_stats_csv",
    "refined_lung_crop_slice_stats_csv",
    "pure_lung_crop_slice_stats_csv",
    "ts_lung_crop_summary_json",
    "refined_lung_crop_summary_json",
    "pure_lung_crop_summary_json",
]

crop_file_missing = []

for col in crop_file_cols:
    if col not in meta.columns:
        print(col, ": 컬럼 없음")
        crop_file_missing.append(col)
        continue

    paths = meta[col].dropna().astype(str).unique()
    print(col, "unique:", len(paths))

    checked_count = 0

    for p in paths[:3]:
        exists = Path(p).exists()
        print(" ", exists, p)

        if not exists:
            crop_file_missing.append(p)

        checked_count += 1


# ============================================================
# 10. NIfTI shape check
# ============================================================

print("\n========== NIFTI SHAPE CHECK ==========")

row = meta.iloc[0]

nii_cols = [
    "ct_1mm_lps_nii",
    "lung_refined_1mm_nii",
    "organ_exclusion_1mm_nii",
    "pure_lung_1mm_nii",
    "ts_lung_guard_1mm_nii",
    "body_guard_1mm_nii",
]

# no-chest가 아닌 예전 결과라면 있을 수도 있으므로, 있으면 추가 확인
optional_nii_cols = [
    "chestwall_candidate_1mm_nii",
]

for c in optional_nii_cols:
    if check_path_column_exists(meta, c):
        nii_cols.append(c)

shapes = {}
shape_errors = []

for col in nii_cols:
    if col not in meta.columns:
        print(col, ": 컬럼 없음 → skip")
        shape_errors.append(f"missing column: {col}")
        continue

    p = Path(str(row[col]))

    if not p.exists():
        print(col, ": 파일 없음 →", p)
        shape_errors.append(f"missing file: {p}")
        continue

    try:
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(p)))
        shapes[col] = arr.shape
        print(col, arr.shape, arr.dtype)
    except Exception as e:
        print(col, ": 읽기 실패 →", str(e))
        shape_errors.append(f"read error: {col}: {e}")

same_shape = len(set(shapes.values())) == 1 if len(shapes) > 0 else False
print("all same shape:", same_shape)


# ============================================================
# 11. organ count check
# ============================================================

print("\n========== ORGAN COUNT CHECK ==========")

if "organ_count" in runtime.columns:
    print("runtime organ_count summary:")
    display(runtime["organ_count"].describe())

    print("organ_count value counts:")
    display(runtime["organ_count"].value_counts().sort_index())

if "organ_name" in organs.columns:
    print("organ names:")
    display(sorted(organs["organ_name"].dropna().unique().tolist()))


# ============================================================
# 12. make one overlay per patient
# ============================================================

print("\n========== MAKE QA OVERLAY ==========")

QA_OUT_DIR.mkdir(parents=True, exist_ok=True)

overlay_rows = []

patient_ids = sorted(meta["patient_id"].dropna().astype(str).unique())

for idx, patient_id in enumerate(patient_ids, start=1):
    patient_df = meta[meta["patient_id"].astype(str) == patient_id].copy()

    if len(patient_df) == 0:
        continue

    selected_row = choose_representative_slice(patient_df)
    z = int(selected_row["slice_index"])

    ct_path = Path(str(selected_row["ct_1mm_lps_nii"]))
    pure_path = Path(str(selected_row["pure_lung_1mm_nii"]))
    guard_path = Path(str(selected_row["ts_lung_guard_1mm_nii"]))
    body_path = Path(str(selected_row["body_guard_1mm_nii"]))

    try:
        _, ct_arr = read_nii_array(ct_path)
        _, pure_arr = read_nii_array(pure_path)
        _, guard_arr = read_nii_array(guard_path)

        body_arr = None

        if body_path.exists():
            _, body_arr = read_nii_array(body_path)

        if ct_arr.shape != pure_arr.shape:
            raise RuntimeError("CT와 pure_lung shape 다름")

        if ct_arr.shape != guard_arr.shape:
            raise RuntimeError("CT와 ts_lung_guard shape 다름")

        if body_arr is not None and ct_arr.shape != body_arr.shape:
            raise RuntimeError("CT와 body_guard shape 다름")

        out_png = QA_OUT_DIR / f"{idx:04d}_{patient_id}_slice_{z:04d}_overlay.png"

        make_overlay_image(
            ct_slice=ct_arr[z],
            pure_slice=pure_arr[z],
            guard_slice=guard_arr[z],
            body_slice=body_arr[z] if body_arr is not None else None,
            patient_id=patient_id,
            z_index=z,
            out_path=out_png,
        )

        overlay_rows.append({
            "patient_id": patient_id,
            "slice_index": z,
            "overlay_png": str(out_png),
            "pure_lung_area": int(selected_row.get("pure_lung_area", -1)),
            "ts_lung_guard_area": int(selected_row.get("ts_lung_guard_area", -1)),
            "body_guard_area": int(selected_row.get("body_guard_area", -1)),
            "status": "success",
            "error": "",
        })

        print(f"[{idx}/{len(patient_ids)}] saved:", out_png)

    except Exception as e:
        overlay_rows.append({
            "patient_id": patient_id,
            "slice_index": z,
            "overlay_png": "",
            "pure_lung_area": int(selected_row.get("pure_lung_area", -1)),
            "ts_lung_guard_area": int(selected_row.get("ts_lung_guard_area", -1)),
            "body_guard_area": int(selected_row.get("body_guard_area", -1)),
            "status": "error",
            "error": str(e),
        })

        print(f"[ERROR] {patient_id}:", str(e))


overlay_df = pd.DataFrame(overlay_rows)

overlay_summary_csv = QA_OUT_DIR / "qa_overlay_summary.csv"
overlay_df.to_csv(
    overlay_summary_csv,
    index=False,
    encoding="utf-8-sig",
)

print("\nSaved overlay summary:", overlay_summary_csv)
print("overlay success:", int((overlay_df["status"] == "success").sum()))
print("overlay error:", int((overlay_df["status"] == "error").sum()))


# ============================================================
# 13. display overlay images in notebook
# ============================================================

print("\n========== DISPLAY QA OVERLAY ==========")

success_overlay_df = overlay_df[overlay_df["status"] == "success"].copy()

if DISPLAY_LIMIT is None:
    display_df = success_overlay_df
else:
    display_df = success_overlay_df.head(int(DISPLAY_LIMIT))

print("display count:", len(display_df))
print("saved overlay dir:", QA_OUT_DIR)

if CAN_DISPLAY:
    for _, r in display_df.iterrows():
        print("\npatient:", r["patient_id"], "| slice:", r["slice_index"])
        display(IPyImage(filename=r["overlay_png"]))
else:
    print("IPython display 사용 불가. PNG 파일을 직접 열어서 확인해야 함.")


# ============================================================
# 14. final quick judgment
# ============================================================

print("\n========== FINAL QUICK JUDGMENT ==========")

success_count = int((runtime["status"] == "success").sum()) if "status" in runtime.columns else 0
error_count = int((runtime["status"] == "error").sum()) if "status" in runtime.columns else -1

final_ok = (
    len(missing_cols) == 0
    and same_shape
    and success_count >= 1
    and int((overlay_df["status"] == "success").sum()) >= 1
)

print("missing required columns:", len(missing_cols))
print("same shape:", same_shape)
print("runtime success count:", success_count)
print("runtime error count:", error_count)
print("overlay success count:", int((overlay_df["status"] == "success").sum()))
print("overlay error count:", int((overlay_df["status"] == "error").sum()))

if final_ok:
    print("테스트 통과 가능성 높음")
else:
    print("확인 필요")

# %%==== CELL ====

from pathlib import Path
import pandas as pd

OUT_ROOT = Path(r"E:\jyp\ct_data_2d_preprocessed\Normal_LUNA16_final_all_nonfast_v2_tslungguard_nochest")

meta = pd.read_csv(OUT_ROOT / "metadata_slices.csv")

patient_crop = meta.groupby("patient_id").agg({
    "patient_ts_lung_bbox_max_square_with_margin": "max",
    "patient_refined_lung_bbox_max_square_with_margin": "max",
    "patient_pure_lung_bbox_max_square_with_margin": "max",
    "patient_ts_lung_bbox_recommended_standard_square": "max",
    "patient_refined_lung_bbox_recommended_standard_square": "max",
    "patient_pure_lung_bbox_recommended_standard_square": "max",
}).reset_index()

display(patient_crop)

print("\n========== 전체 환자 기준 최대값 ==========")
print("TS lung guard max with margin:",
      patient_crop["patient_ts_lung_bbox_max_square_with_margin"].max())

print("Refined lung max with margin:",
      patient_crop["patient_refined_lung_bbox_max_square_with_margin"].max())

print("Pure lung max with margin:",
      patient_crop["patient_pure_lung_bbox_max_square_with_margin"].max())

print("\n========== 추천 표준 입력 크기 ==========")
print("TS lung guard recommended:",
      patient_crop["patient_ts_lung_bbox_recommended_standard_square"].max())

print("Refined lung recommended:",
      patient_crop["patient_refined_lung_bbox_recommended_standard_square"].max())

print("Pure lung recommended:",
      patient_crop["patient_pure_lung_bbox_recommended_standard_square"].max())

# %%==== CELL ====

