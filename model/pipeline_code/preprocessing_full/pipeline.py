# ============================================================
# pipeline.py
# ------------------------------------------------------------
# ref_01_preprocess.py + ref_02_roi_0_0.py 함수 포팅
# 함수명은 노트북 원본 유지.
# 디버깅용 PNG 저장, QA overlay, NIfTI 중간파일 저장은 포함하지 않음.
# 추론에 필요한 array 파이프라인만 포함.
# ============================================================

import os
import sys
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation
from tqdm import tqdm

from . import config as C


# ============================================================
# 기본 유틸
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


def log_print(message: str, verbose: bool = False, force: bool = False):
    if force or verbose:
        print(message)


# ============================================================
# 입력 로드 — DICOM / MHD+RAW
# ============================================================

def find_raw_from_mhd_header(mhd_path: Path):
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


def load_dicom_series_from_folder(series_dir: Path, verbose: bool = False) -> sitk.Image:
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

    log_print(f"[DICOM] selected series id: {best_series_id}", verbose)
    log_print(f"[DICOM] number of slices: {len(best_files)}", verbose)

    reader.SetFileNames(best_files)
    return reader.Execute()


def load_mhd_raw_pair(mhd_path: Path, verbose: bool = False) -> sitk.Image:
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

    log_print(f"[MHD] mhd path: {mhd_path}", verbose)
    log_print(f"[MHD] raw path: {raw_path}", verbose)

    img = sitk.ReadImage(str(mhd_path))
    return img


def load_input_volume(input_path, verbose: bool = False) -> sitk.Image:
    """
    input_path가 디렉토리이면 DICOM, .mhd 파일이면 MHD+RAW로 로드.
    """
    input_path = Path(input_path)

    if input_path.is_dir():
        # DICOM 폴더
        return load_dicom_series_from_folder(input_path, verbose=verbose)
    elif input_path.suffix.lower() == ".mhd":
        return load_mhd_raw_pair(input_path, verbose=verbose)
    else:
        raise ValueError(
            f"input_path는 DICOM 폴더 또는 .mhd 파일이어야 함: {input_path}"
        )


# ============================================================
# Orient / Resample
# ============================================================

def orient_image(img: sitk.Image, orientation: str = C.ORIENTATION) -> sitk.Image:
    """LPS orientation으로 변환."""
    return sitk.DICOMOrient(img, orientation)


def resample_z_only(
    img: sitk.Image,
    target_z: float = C.TARGET_Z,
    interpolator=sitk.sitkLinear,
    default_value: float = -1024.0,
) -> sitk.Image:
    """
    x/y spacing은 유지하고 z spacing만 target_z(1.0mm)로 변경.
    CT는 sitkLinear 사용.
    default_value=-1024 (노트북 process_one_patient 호출 기준).
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


def resample_to_reference(
    moving: sitk.Image,
    reference: sitk.Image,
    interpolator=sitk.sitkNearestNeighbor,
    default_value: float = 0.0,
) -> sitk.Image:
    """
    moving image를 reference image grid에 맞춤.
    mask는 반드시 sitkNearestNeighbor 사용.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    return resampler.Execute(moving)


def array_to_sitk_like(mask_arr: np.ndarray, reference_img: sitk.Image) -> sitk.Image:
    out = sitk.GetImageFromArray(mask_arr.astype(np.uint8))
    out.CopyInformation(reference_img)
    return out


# ============================================================
# TotalSegmentator 실행
# ============================================================

def find_totalseg_executable() -> str:
    # 1) PATH 탐색 (운영: start.bat가 lunar_env\Scripts를 PATH에 추가)
    for cmd in ["TotalSegmentator", "totalsegmentator"]:
        if shutil.which(cmd) is not None:
            return cmd
    # 2) 실행 중 python 옆 Scripts/ 폴백 (lunar_env 임베디드 환경: PATH 미설정 시)
    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    for cand in [
        os.path.join(exe_dir, "Scripts", "TotalSegmentator.exe"),
        os.path.join(exe_dir, "TotalSegmentator.exe"),
        os.path.join(exe_dir, "Scripts", "TotalSegmentator"),
        os.path.join(exe_dir, "TotalSegmentator"),
    ]:
        if os.path.isfile(cand):
            return cand
    raise RuntimeError(
        "TotalSegmentator 실행 파일을 찾지 못했음. "
        "환경에 TotalSegmentator가 설치되어 있는지 확인해야 함."
    )


def has_required_totalseg_masks(out_dir: Path, required_names) -> bool:
    """
    TotalSegmentator 결과 폴더에 필요한 ROI mask가 모두 있는지 확인.
    """
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return False
    for name in required_names:
        if not (out_dir / f"{name}.nii.gz").exists():
            return False
    return True


def run_totalsegmentator_native(
    ct_native_path: Path,
    out_dir: Path,
    totalseg_kwargs: dict = None,
    verbose: bool = False,
) -> None:
    """
    원본 spacing CT에서 TotalSegmentator 실행.
    totalseg_kwargs 기본값은 config.py 기준.

    totalseg_kwargs 허용 키:
        overwrite (bool): 기존 결과 덮어쓰기 여부, 기본 False
        use_fast (bool): fast 모드 여부, 기본 False (non-fast 정밀 모드)
        organ_roi_subset (list): 요청할 ROI 이름 목록
        log_dir (Path or None): stdout/stderr 저장 디렉토리, None이면 저장 안 함
    """
    if totalseg_kwargs is None:
        totalseg_kwargs = {}

    overwrite = bool(totalseg_kwargs.get("overwrite", C.OVERWRITE_TOTALSEG))
    use_fast = bool(totalseg_kwargs.get("use_fast", C.USE_FAST_TOTALSEG))
    organ_roi_subset = list(totalseg_kwargs.get("organ_roi_subset", C.ORGAN_ROI_SUBSET))
    log_dir = totalseg_kwargs.get("log_dir", None)

    if out_dir.exists() and not overwrite:
        if has_required_totalseg_masks(out_dir, organ_roi_subset):
            log_print(f"[TotalSegmentator] 기존 결과 재사용: {out_dir}", verbose, force=True)
            return
        log_print(
            f"[TotalSegmentator] 기존 결과가 있지만 필요한 ROI가 부족해서 다시 실행: {out_dir}",
            verbose, force=True,
        )
        shutil.rmtree(out_dir)

    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)

    ensure_dir(out_dir)

    exe = find_totalseg_executable()

    cmd = [
        exe,
        "-i", str(ct_native_path),
        "-o", str(out_dir),
        "--nr_thr_resamp", "1",
        "--nr_thr_saving", "1",
    ]

    if use_fast:
        cmd.append("-f")

    if len(organ_roi_subset) > 0:
        cmd += ["--roi_subset"] + list(organ_roi_subset)

    log_print(f"[TotalSegmentator] command: {' '.join(cmd)}", verbose)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if log_dir is not None:
        log_dir = Path(log_dir)
        ensure_dir(log_dir)
        (log_dir / "totalseg_stdout.txt").write_text(
            result.stdout or "", encoding="utf-8", errors="replace"
        )
        (log_dir / "totalseg_stderr.txt").write_text(
            result.stderr or "", encoding="utf-8", errors="replace"
        )

    log_print(f"[TotalSegmentator] returncode: {result.returncode}", verbose)

    if result.returncode != 0:
        raise RuntimeError(
            f"TotalSegmentator 실행 실패 (returncode={result.returncode})\n"
            + (result.stderr or "")[-2000:]
        )


def resample_organ_masks_to_1mm(
    totalseg_dir: Path,
    ct_1mm: sitk.Image,
) -> list:
    """
    원본 spacing TotalSegmentator mask를 1mm CT grid로 맞춤.
    NIfTI로 저장하지 않고 {organ_name: np.ndarray(bool)} dict 반환.
    """
    totalseg_dir = Path(totalseg_dir)
    mask_files = sorted(totalseg_dir.glob("*.nii.gz"))

    if len(mask_files) == 0:
        raise RuntimeError(f"TotalSegmentator 결과 mask 없음: {totalseg_dir}")

    organ_arrays = {}

    for mask_path in mask_files:
        organ_name = mask_path.name.replace(".nii.gz", "")
        mask_native = sitk.ReadImage(str(mask_path))

        mask_1mm = resample_to_reference(
            moving=mask_native,
            reference=ct_1mm,
            interpolator=sitk.sitkNearestNeighbor,
            default_value=0,
        )

        arr = sitk.GetArrayFromImage(mask_1mm) > 0
        organ_arrays[organ_name] = arr

    return organ_arrays


# ============================================================
# HU 유틸
# ============================================================

def hu_to_uint8(
    slice_hu: np.ndarray,
    hu_min: int = C.HU_MIN,
    hu_max: int = C.HU_MAX,
) -> np.ndarray:
    x = np.clip(slice_hu.astype(np.float32), hu_min, hu_max)
    x = (x - hu_min) / float(hu_max - hu_min)
    x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return x


# ============================================================
# 세밀 폐 mask 생성 (HU 기반)
# ============================================================

def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep_id = int(np.argmax(areas) + 1)
    return (labels == keep_id).astype(np.uint8)


def keep_largest_two_components(mask: np.ndarray) -> np.ndarray:
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


def fill_holes(mask: np.ndarray) -> np.ndarray:
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


def remove_border_connected(mask: np.ndarray) -> np.ndarray:
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


def refined_lung_mask_2d_from_hu(
    slice_hu: np.ndarray,
    hu_min: int = C.HU_MIN,
    hu_max: int = C.HU_MAX,
) -> np.ndarray:
    """
    HU slice에서 폐 영역을 세밀하게 추출.
    TotalSegmentator 폐 mask를 쓰지 않고, HU 기반으로 폐 후보를 만듦.
    임계값·연산 순서 노트북 원본과 동일.
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


def build_refined_lung_mask_3d(
    ct_arr: np.ndarray,
    hu_min: int = C.HU_MIN,
    hu_max: int = C.HU_MAX,
    verbose: bool = False,
) -> np.ndarray:
    zdim, h, w = ct_arr.shape
    lung = np.zeros((zdim, h, w), dtype=bool)

    iterator = tqdm(
        range(zdim),
        desc="Build refined lung mask",
        ncols=100,
        ascii=True,
        leave=False,
        disable=not verbose,
    )

    for z in iterator:
        lung[z] = refined_lung_mask_2d_from_hu(ct_arr[z], hu_min, hu_max)

    return lung


# ============================================================
# organ exclusion / TS lung guard / body guard
# ============================================================

def build_union_mask_from_organ_arrays(
    organ_arrays: dict,
    target_names: list,
    reference_shape=None,
    dilate_iter: int = 0,
):
    """
    organ_arrays (dict: organ_name -> np.ndarray bool)에서
    target_names에 해당하는 mask들을 합쳐서 하나의 union mask 생성.

    ref_01의 build_union_mask_from_organ_rows에 대응.
    NIfTI 재로드 없이 이미 array로 넘겨받은 버전.
    """
    target_names = set(list(target_names))
    union = None
    used_names = []

    for organ_name, mask_arr in organ_arrays.items():
        if organ_name not in target_names:
            continue
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


def build_organ_exclusion_mask(
    organ_arrays: dict,
    organ_exclusion_names: list = None,
    dilate_iter: int = C.ORGAN_EXCLUSION_DILATE_ITER,
):
    """
    pure_lung에서 제외할 장기 union mask 생성.
    ref_01의 build_organ_exclusion_mask에 대응.
    """
    if organ_exclusion_names is None:
        organ_exclusion_names = C.ORGAN_EXCLUSION_ROI_NAMES

    exclusion, used_organs = build_union_mask_from_organ_arrays(
        organ_arrays=organ_arrays,
        target_names=organ_exclusion_names,
        dilate_iter=dilate_iter,
    )

    if exclusion is None:
        raise RuntimeError("organ exclusion에 사용할 장기 mask가 없음.")

    return exclusion.astype(bool), used_organs


def build_body_guard_mask_3d(
    ct_arr: np.ndarray,
    hu_threshold: float = C.BODY_GUARD_HU_THRESHOLD,
) -> np.ndarray:
    """
    HU 기준 body outer contour 생성.
    TotalSegmentator가 아니라 CT HU 값으로 직접 생성.
    """
    zdim, h, w = ct_arr.shape
    body_3d = np.zeros((zdim, h, w), dtype=bool)

    for z in range(zdim):
        body = (ct_arr[z] > hu_threshold).astype(np.uint8)
        body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        body = cv2.morphologyEx(body, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        body = keep_largest_component(body)
        body = fill_holes(body)
        body_3d[z] = body > 0

    return body_3d


# ============================================================
# Lung z-range crop
# ============================================================

def fill_small_false_gaps(valid_mask: np.ndarray, max_gap: int) -> np.ndarray:
    """
    valid_mask 안에서 짧게 끊긴 False 구간을 True로 메움.
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
    """True가 연속되는 구간들을 (start, end)로 반환."""
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
    min_area_ratio: float = C.LUNG_RANGE_MIN_PURE_LUNG_AREA_RATIO,
    margin_slices: int = C.LUNG_RANGE_MARGIN_SLICES,
    max_gap_slices: int = C.LUNG_RANGE_MAX_GAP_SLICES,
    min_segment_slices: int = C.LUNG_RANGE_MIN_SEGMENT_SLICES,
) -> dict:
    """
    pure_lung mask를 기준으로 폐가 있는 z축 연속 구간을 찾음.
    최종 결과는 반드시 z_start ~ z_end까지 연속 구간.
    파라미터 기본값은 학습 계약 고정값.
    """
    zdim, h, w = pure_lung.shape

    pure_area = pure_lung.sum(axis=(1, 2))
    pure_area_ratio = pure_area / float(h * w)

    valid = pure_area_ratio >= float(min_area_ratio)

    valid_filled = fill_small_false_gaps(
        valid_mask=valid,
        max_gap=int(max_gap_slices),
    )

    segments = find_true_segments(valid_filled)

    segments = [
        (s, e)
        for s, e in segments
        if (e - s + 1) >= int(min_segment_slices)
    ]

    if len(segments) == 0:
        return {
            "z_start": 0,
            "z_end": zdim - 1,
            "found_lung_range": 0,
            "reason": "no_valid_lung_segment_found",
            "pure_lung_area_ratio_per_slice": pure_area_ratio,
        }

    # 가장 큰 단일 구간 (로깅/참고용)
    best_segment = None
    best_score = -1
    for s, e in segments:
        score = float(pure_area[s:e + 1].sum())
        if score > best_score:
            best_score = score
            best_segment = (s, e)

    select_mode = getattr(C, "LUNG_RANGE_SELECT_MODE", "full_span")
    if select_mode == "largest_segment":
        # 노트북 원본 동작: 가장 큰 단일 구간만
        z_start, z_end = best_segment
    else:
        # full_span: 폐가 중간 dip으로 쪼개져도 첫 구간 시작 ~ 마지막 구간 끝까지 모두 포함
        # → 실제 폐 슬라이스 손실(과다 crop) 방지. 단일 구간이면 largest와 동일.
        z_start = min(s for s, _ in segments)
        z_end = max(e for _, e in segments)

    # 진단 로그: 폐 구간이 여러 개로 쪼개졌는지/얼마나 잘렸는지 확인용
    print(f"[LUNG_RANGE] mode={select_mode} n_segments={len(segments)} "
          f"segments={segments} largest={best_segment} chosen=({z_start},{z_end}) "
          f"area_ratio[min={float(pure_area_ratio.min()):.4f},max={float(pure_area_ratio.max()):.4f}]",
          flush=True)
    if len(segments) > 1 and select_mode != "largest_segment":
        print(f"[LUNG_RANGE] ⚠ 폐가 {len(segments)}개 구간으로 분리됨 → full_span으로 전체 포함 "
              f"(largest만 쓰면 폐 손실)", flush=True)

    z_start = max(0, int(z_start) - int(margin_slices))
    z_end = min(zdim - 1, int(z_end) + int(margin_slices))

    return {
        "z_start": int(z_start),
        "z_end": int(z_end),
        "found_lung_range": 1,
        "reason": "ok",
        "n_segments": len(segments),
        "segments": [(int(s), int(e)) for s, e in segments],
        "largest_segment": (int(best_segment[0]), int(best_segment[1])),
        "select_mode": select_mode,
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
    return sitk.RegionOfInterest(img, size=crop_size, index=index)


# ============================================================
# build_roi_0_0
# ref_02_roi_0_0.py 기준 (no_dilate 버전)
# ============================================================

def dilate_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    """3D binary dilation. iterations=0이면 copy만 반환."""
    if iterations <= 0:
        return mask.copy()
    return binary_dilation(
        mask,
        structure=np.ones((3, 3, 3), dtype=bool),
        iterations=int(iterations),
    )


def build_roi_0_0(
    raw_ts_lung: np.ndarray,
    body_guard=None,
    organ_exclusion=None,
    use_body_guard: bool = C.ROI_USE_BODY_GUARD,
    use_organ_exclusion: bool = C.ROI_USE_ORGAN_EXCLUSION,
) -> np.ndarray:
    """
    0/0 ROI 생성 (no_dilate 버전):
    - TotalSegmentator 폐엽 5개 합침 (dilation 없음)
    - body_guard 미적용 (use_body_guard=False 고정)
    - organ_exclusion 적용 (use_organ_exclusion=True 고정)

    ref_02의 build_roi_0_0 그대로 포팅.
    """
    roi = raw_ts_lung.copy()

    if use_body_guard and body_guard is not None:
        roi = roi & body_guard

    if use_organ_exclusion and organ_exclusion is not None:
        roi = roi & (~organ_exclusion)

    return roi
