import os
import sys
import numpy as np
import pydicom
import io
import cv2
import SimpleITK as sitk
from scipy.ndimage import binary_dilation
import subprocess
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from position_aware_padim.padim_model import PaDiMModel
from position_aware_padim.feature_extractor_effnet_b0_scaffold import FeatureExtractorEffNetB0
from position_aware_padim.preprocessing import preprocess_ct_slice

# ── 가중치 경로 ────────────────────────────────────────────────────────────────
_BACKEND_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR   = os.path.join(_BACKEND_DIR, "weights")
NPZ_PATH      = os.path.join(WEIGHTS_DIR, "position_bin_stats.npz")
INDICES_PATH  = os.path.join(WEIGHTS_DIR, "selected_feature_indices.npy")

# ── TotalSegmentator 번들 가중치 경로 ─────────────────────────────────────────
_BUNDLED_TS_DIR     = os.path.join(_BACKEND_DIR, "totalseg_data")
_BUNDLED_TS_WEIGHTS = os.path.join(_BUNDLED_TS_DIR, "nnunet", "results")
if os.path.isdir(_BUNDLED_TS_WEIGHTS):
    os.environ.setdefault("TOTALSEG_HOME_DIR",    _BUNDLED_TS_DIR)
    os.environ.setdefault("TOTALSEG_WEIGHTS_PATH", _BUNDLED_TS_WEIGHTS)

# ── TotalSegmentator 설정 ──────────────────────────────────────────────────────
TS_LUNG_LOBE_NAMES = [
    "lung_upper_lobe_left", "lung_lower_lobe_left",
    "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
]
ORGAN_EXCLUSION_NAMES = [
    "heart", "aorta", "trachea", "esophagus",
    "liver", "stomach", "spleen", "pancreas",
]
TS_LUNG_DILATE_ITER = 2
ORGAN_DILATE_ITER   = 1

# ── PaDiM 모델 로드 (서버 시작 시 1회) ────────────────────────────────────────
_padim_model = PaDiMModel(
    selected_feature_indices_path=INDICES_PATH,
    feature_dim=100,
    eps=1e-5,
)
_padim_model.load(NPZ_PATH)
_feature_extractor = FeatureExtractorEffNetB0()

# ── threshold (팀원 확정값 고정) ───────────────────────────────────────────────
PADIM_THRESHOLD_P90 = 12.20   # 정상 36명 1,490,012 패치 p90 (최강 파이프라인 기준)

# =============================================================================
# 헬퍼: HU → uint8
# =============================================================================

def _hu_to_uint8(hu: np.ndarray, hu_min=-1000, hu_max=400) -> np.ndarray:
    x = np.clip(hu.astype(np.float32), hu_min, hu_max)
    x = (x - hu_min) / float(hu_max - hu_min)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def _keep_largest(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    keep = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)
    return (labels == keep).astype(np.uint8)


def _keep_largest_two(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = list(np.argsort(areas)[-2:] + 1) if len(areas) >= 2 else [1]
    return np.isin(labels, keep).astype(np.uint8)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return mask
    flood = (mask * 255).copy()
    h, w  = flood.shape
    cv2.floodFill(flood, np.zeros((h+2, w+2), np.uint8), (0, 0), 255)
    filled = (mask * 255) | cv2.bitwise_not(flood)
    return (filled > 0).astype(np.uint8)


def _remove_border(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = mask.shape
    out  = np.zeros_like(mask)
    for i in range(1, n):
        x, y, ww, hh, _ = stats[i]
        if not (x == 0 or y == 0 or x+ww >= w or y+hh >= h):
            out[labels == i] = 1
    return out


def _hu_lung_mask(hu: np.ndarray) -> np.ndarray:
    """v1 GaussianBlur+CLAHE+Otsu 기반 폐 마스크"""
    arr  = _hu_to_uint8(hu)
    blur = cv2.GaussianBlur(arr, (5, 5), 0)
    eq   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(blur)

    body = (arr > 5).astype(np.uint8)
    body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    body = cv2.morphologyEx(body, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8))
    body = _keep_largest(body)
    body = _fill_holes(body)

    vals = eq[body > 0]
    if vals.size < 50:
        thr = 90
    else:
        thr, _ = cv2.threshold(
            vals.reshape(-1,1).astype(np.uint8), 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = int(np.clip(thr, 55, 115))

    lung = ((eq <= thr) & (body > 0)).astype(np.uint8)
    lung = _remove_border(lung)
    lung = cv2.morphologyEx(lung, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8))
    lung = cv2.morphologyEx(lung, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    lung = _fill_holes(lung)
    lung = _keep_largest_two(lung)
    lung = _fill_holes(lung)
    return lung.astype(bool)

# =============================================================================
# TotalSegmentator
# =============================================================================

def _run_ts(sitk_img: sitk.Image, tmp_dir: str) -> dict:
    ct_path = os.path.join(tmp_dir, "ct.nii.gz")
    seg_dir = os.path.join(tmp_dir, "seg")
    os.makedirs(seg_dir, exist_ok=True)
    sitk.WriteImage(sitk_img, ct_path)

    target_roi = TS_LUNG_LOBE_NAMES + ORGAN_EXCLUSION_NAMES
    import torch
    device_flag = ["--device", "gpu"] if torch.cuda.is_available() else ["--device", "cpu"]
    cmd = ["TotalSegmentator", "-i", ct_path, "-o", seg_dir,
            "--fast", *device_flag, "--roi_subset", *target_roi]
    subprocess.run(cmd, check=True, timeout=600)

    masks = {}
    for name in target_roi:
        p = os.path.join(seg_dir, f"{name}.nii.gz")
        if os.path.exists(p):
            masks[name] = sitk.GetArrayFromImage(sitk.ReadImage(p)) > 0
    return masks


def _ts_lung_guard(masks: dict, shape: tuple) -> np.ndarray:
    guard = np.zeros(shape, dtype=bool)
    for n in TS_LUNG_LOBE_NAMES:
        if n in masks:
            guard |= masks[n]
    if TS_LUNG_DILATE_ITER > 0:
        guard = binary_dilation(guard,
            structure=np.ones((3,3,3), dtype=bool),
            iterations=TS_LUNG_DILATE_ITER)
    return guard


def _organ_excl(masks: dict, shape: tuple) -> np.ndarray:
    excl = np.zeros(shape, dtype=bool)
    for n in ORGAN_EXCLUSION_NAMES:
        if n in masks:
            excl |= masks[n]
    if ORGAN_DILATE_ITER > 0:
        excl = binary_dilation(excl,
            structure=np.ones((3,3,3), dtype=bool),
            iterations=ORGAN_DILATE_ITER)
    return excl

# =============================================================================
# position bin
# 원본 position_bin_stats.npz 키 확인 결과:
#   upper_central, upper_peripheral, middle_central,
#   middle_peripheral, lower_central, lower_peripheral
# → 아래 함수 출력 형식과 일치, 변경 없음
# =============================================================================

def assign_position_bin(y: float, x: float, H: float, W: float) -> str:
    zone   = "upper" if y < H/3 else ("middle" if y < 2*H/3 else "lower")
    region = "central" if abs(x - W/2) < W * 0.2 else "peripheral"
    return f"{zone}_{region}"

# =============================================================================
# RD4AD 입력용 medi3ch crop 생성
# 원본: build_medi3ch_crop (rd_d1s_medi3ch_shard_optimized_rd4ad.py)
# ── HU clip: mediastinal window [-160, 240]
# ── 정규화: [0, 1]
# ── 3채널: z-1, z, z+1 슬라이스
# ── OOB: reflect padding (valid_h/w <= 1이면 edge fallback)
# ── shape 불일치 시 RuntimeError (원본과 동일, cv2.resize 강제 변환 금지)
# ── NaN/Inf 포함 시 RuntimeError (원본과 동일)
# =============================================================================

MEDI_HU_MIN = -160.0
MEDI_HU_MAX =  240.0
CROP_SIZE   = 96

def make_medi3ch_crop(hu_volume: np.ndarray, z: int,
                      y0: int, x0: int) -> np.ndarray:
    """
    hu_volume: (Z, H, W)
    z        : 현재 슬라이스 인덱스
    반환: (3, 96, 96) float32  [0, 1]
    원본 build_medi3ch_crop과 동일한 로직.
    """
    Z, H, W = hu_volume.shape
    z  = int(z)
    zm = max(z - 1, 0)
    zp = min(z + 1, Z - 1)
    y0, x0 = int(y0), int(x0)
    y1 = y0 + CROP_SIZE
    x1 = x0 + CROP_SIZE

    pad_top    = max(0, -y0)
    pad_bottom = max(0, y1 - H)
    pad_left   = max(0, -x0)
    pad_right  = max(0, x1 - W)
    needs_pad  = (pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0)

    cy0 = max(0, y0)
    cy1 = min(H, y1)
    cx0 = max(0, x0)
    cx1 = min(W, x1)
    valid_h = cy1 - cy0
    valid_w = cx1 - cx0

    if needs_pad:
        can_reflect = (valid_h > 1) and (valid_w > 1)
        pad_mode    = "reflect" if can_reflect else "edge"

    def _win(sl):
        c = np.clip(sl.astype(np.float32), MEDI_HU_MIN, MEDI_HU_MAX)
        return (c - MEDI_HU_MIN) / (MEDI_HU_MAX - MEDI_HU_MIN)

    def _build_ch(z_idx):
        normed = _win(hu_volume[z_idx, cy0:cy1, cx0:cx1])
        if needs_pad:
            normed = np.pad(normed,
                            ((pad_top, pad_bottom), (pad_left, pad_right)),
                            mode=pad_mode)
        return normed

    crop = np.stack([_build_ch(zm), _build_ch(z), _build_ch(zp)], axis=0)

    # 원본과 동일: shape 불일치 시 RuntimeError (cv2.resize 강제 변환 금지)
    if crop.shape != (3, CROP_SIZE, CROP_SIZE):
        raise RuntimeError(
            f"[ABORT] crop shape {crop.shape} != (3,{CROP_SIZE},{CROP_SIZE})  "
            f"y0={y0} x0={x0} y1={y1} x1={x1} H={H} W={W} "
            f"pad=({pad_top},{pad_bottom},{pad_left},{pad_right})"
        )
    # 원본과 동일: NaN/Inf 포함 시 RuntimeError
    if not np.isfinite(crop).all():
        raise RuntimeError(
            f"[ABORT] crop contains NaN/Inf  y0={y0} x0={x0} y1={y1} x1={x1}"
        )

    return crop.astype(np.float32)

# =============================================================================
# 1차 PaDiM 추론
# =============================================================================

def run_padim(dicom_bytes: bytes, hu_volume: np.ndarray = None,
              slice_z: int = 0) -> dict:
    """
    dicom_bytes : 단일 슬라이스 DICOM
    hu_volume   : (Z,H,W) 전체 볼륨 (2차 RD4AD medi3ch 입력용)
                  None이면 단일 슬라이스 3채널 복제 사용
    slice_z     : 현재 슬라이스가 볼륨 내 몇 번째인지
    """
    tmp_dir = tempfile.mkdtemp(prefix="lunar_ts_")
    try:
        # 1. DICOM → HU
        ds        = pydicom.dcmread(io.BytesIO(dicom_bytes))
        pixel     = ds.pixel_array.astype(np.float32)
        slope     = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", -1024))
        hu_slice  = pixel * slope + intercept
        H, W      = hu_slice.shape

        # 2. TotalSegmentator (v2 전처리) 시도
        try:
            vol_3d    = hu_slice[np.newaxis, :, :].astype(np.float32)
            sitk_img  = sitk.GetImageFromArray(vol_3d)
            sitk_img.SetSpacing((1.0, 1.0, 1.0))
            masks     = _run_ts(sitk_img, tmp_dir)
            ts_guard  = _ts_lung_guard(masks, (1, H, W))[0]
            organ_exc = _organ_excl(masks, (1, H, W))[0]
            hu_lung   = _hu_lung_mask(hu_slice)
            refined   = hu_lung & ts_guard
            pure_lung = refined & ~organ_exc
        except Exception as e:
            print(f"[WARN] TotalSegmentator 실패 → HU 폴백: {e}")
            hu_lung   = _hu_lung_mask(hu_slice)
            med       = np.zeros((H, W), dtype=bool)
            med[int(H*0.2):int(H*0.8), int(W*0.35):int(W*0.65)] = True
            org       = binary_dilation(med & (hu_slice > -100), iterations=3)
            pure_lung = hu_lung & ~org

        # 3. 비폐 슬라이스 스킵
        if pure_lung.sum() < 100:
            return {"candidate_patches": []}

        # 4. 전처리 (3채널, EfficientNet-B0 입력 형식)
        preprocessed = preprocess_ct_slice(hu_slice)

        # 5. 패치 추출 (pure_lung 중심 + 면적 50%)
        patch_size, stride = 32, 16
        patch_coords = []
        for y0 in range(0, H - patch_size + 1, stride):
            for x0 in range(0, W - patch_size + 1, stride):
                cy = (y0 * 2 + patch_size) // 2
                cx = (x0 * 2 + patch_size) // 2
                if not pure_lung[cy, cx]:
                    continue
                pm = pure_lung[y0:y0+patch_size, x0:x0+patch_size]
                if pm.sum() >= patch_size * patch_size * 0.5:
                    patch_coords.append((y0, x0, y0+patch_size, x0+patch_size))

        if not patch_coords:
            return {"candidate_patches": []}

        # 6. feature 추출 (EfficientNet-B0, 144차원 → 100차원)
        features_144 = _feature_extractor.extract_patch_features(preprocessed, patch_coords)
        features_100 = features_144[:, _padim_model.selected_feature_indices]

        # 7. PaDiM score (원본 p_b9 p95 threshold 기준)
        candidate_patches = []
        for i, (y0, x0, y1, x1) in enumerate(patch_coords):
            cy = (y0 + y1) / 2
            cx = (x0 + x1) / 2
            pb = assign_position_bin(cy, cx, H, W)
            try:
                score = _padim_model.score_patch(features_100[i], pb)
            except Exception:
                continue
            if score > PADIM_THRESHOLD_P90:
                if hu_volume is not None:
                    try:
                        img = make_medi3ch_crop(hu_volume, slice_z, y0, x0).tolist()
                    except RuntimeError as e:
                        print(f"[WARN] make_medi3ch_crop 실패, 단일 슬라이스 복제 사용: {e}")
                        img = _make_rd4ad_input_single(hu_slice, y0, x0)
                else:
                    img = _make_rd4ad_input_single(hu_slice, y0, x0)
                candidate_patches.append({
                    "position": {"y0": y0, "x0": x0, "y1": y1, "x1": x1},
                    "padim_score": float(score),
                    "position_bin": pb,
                    "image": img,
                })

        return {"candidate_patches": candidate_patches}

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _make_rd4ad_input_single(hu: np.ndarray, y0: int, x0: int) -> list:
    """볼륨 없을 때 단일 슬라이스 mediastinal window 3채널 복제"""
    y0, x0 = int(y0), int(x0)
    y1 = y0 + CROP_SIZE
    x1 = x0 + CROP_SIZE

    pad_top    = max(0, -y0)
    pad_bottom = max(0, y1 - hu.shape[0])
    pad_left   = max(0, -x0)
    pad_right  = max(0, x1 - hu.shape[1])
    cy0 = max(0, y0)
    cy1 = min(hu.shape[0], y1)
    cx0 = max(0, x0)
    cx1 = min(hu.shape[1], x1)

    c = np.clip(hu[cy0:cy1, cx0:cx1].astype(np.float32), MEDI_HU_MIN, MEDI_HU_MAX)
    c = (c - MEDI_HU_MIN) / (MEDI_HU_MAX - MEDI_HU_MIN)

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        valid_h = cy1 - cy0
        valid_w = cx1 - cx0
        can_reflect = (valid_h > 1) and (valid_w > 1)
        pad_mode = "reflect" if can_reflect else "edge"
        c = np.pad(c, ((pad_top, pad_bottom), (pad_left, pad_right)), mode=pad_mode)

    return np.stack([c, c, c], axis=0).tolist()
