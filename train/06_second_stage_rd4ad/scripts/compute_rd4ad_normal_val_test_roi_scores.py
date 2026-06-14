"""
Phase 5.60 - Normal val/test ROI-masked score diagnostic script

normal val/test crop에 대해 ROI-masked RD4AD score 계산.
기존 whole-crop score 보존 + ROI-masked diagnostic score 컬럼 추가.
threshold 산출 없음. test split은 threshold 결정에 사용 금지.
stage2_holdout / holdout / v2 접근 금지. training 금지. optimizer 금지.
"""

import argparse
import ast
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────
# 경로 상수
# ──────────────────────────────────────────────────────────
DEFAULT_CROP_MANIFEST = (
    "outputs/second-stage-lesion-refiner-v1/crops_normal/"
    "normal_rd4ad_2p5d_mw_fixed96_v1/manifests/"
    "crop_manifest_normal_rd4ad_2p5d_mw_fixed96_v1.csv"
)
DEFAULT_CHECKPOINT = (
    "outputs/second-stage-lesion-refiner-v1/models/"
    "rd4ad_2p5d_normal_mw_fixed96_v1/checkpoints/best_val_loss.pt"
)
DEFAULT_EXISTING_SCORE_DIR = (
    "outputs/second-stage-lesion-refiner-v1/evaluation/"
    "rd4ad_2p5d_normal_mw_fixed96_v1/normal_val_test_scores_v1/"
)
DEFAULT_VOLUME_ROOT = (
    "/mnt/c/Users/jinhy/Desktop/"
    "Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1"
)
DEFAULT_OUTPUT_ROOT = (
    "outputs/second-stage-lesion-refiner-v1/evaluation/"
    "rd4ad_2p5d_normal_mw_fixed96_v1/normal_val_test_roi_scores_v1/"
)
OUTPUT_CSV_NAME = "normal_val_test_roi_scores_v1.csv"
OUTPUT_JSON_NAME = "normal_val_test_roi_scores_summary_v1.json"
OUTPUT_MD_NAME = "phase5_60_normal_val_test_roi_score_diagnostic_v1.md"

EXPECTED_TOTAL_MANIFEST = 18100
EXPECTED_VAL = 1800
EXPECTED_TEST = 1800
LOW_ROI_RATIO_THRESHOLD = 0.05

# ──────────────────────────────────────────────────────────
# Guard 상수
# ──────────────────────────────────────────────────────────
FORBIDDEN_PATH_KEYWORDS = ["stage2_holdout", "holdout", "v2"]
FORBIDDEN_VOLUME_EXTENSIONS = [".csv", ".manifest", ".split", ".json"]
FORBIDDEN_VOLUME_FILENAMES = ["manifest", "score", "split", "eval", "patch_score"]

# path segment 기준 lesion/hard_negative 차단 목록
# "lesion" 단독 substring 사용 금지 (second-stage-lesion-refiner-v1 오탐 방지)
LESION_HARD_NEG_FORBIDDEN_SEGMENTS = [
    "hard_negative",
    "crops_lesion",
    "lesion_by_patient",
    "lesion_subset",
    "lesion_v2_by_patient",
    "nsclc",
    "msd_lung",
    "nsclc_msd",
]


# ──────────────────────────────────────────────────────────
# Guard 함수
# ──────────────────────────────────────────────────────────
def check_forbidden_path(path_str: str) -> None:
    p = str(path_str).lower()
    for kw in FORBIDDEN_PATH_KEYWORDS:
        if kw in p:
            raise RuntimeError(
                f"[GUARD] 금지된 경로 키워드 '{kw}' 감지: {path_str}\n"
                "stage2_holdout/holdout/v2 경로 접근 금지."
            )


def check_forbidden_volume_file(path_str: str) -> None:
    p = Path(path_str)
    ext = p.suffix.lower()
    name_lower = p.name.lower()
    if ext in FORBIDDEN_VOLUME_EXTENSIONS:
        raise RuntimeError(
            f"[GUARD] volume source 내 금지 확장자 파일 접근 시도: {path_str}"
        )
    for kw in FORBIDDEN_VOLUME_FILENAMES:
        if kw in name_lower:
            raise RuntimeError(
                f"[GUARD] volume source 내 금지 파일명 키워드 '{kw}' 감지: {path_str}"
            )


def check_not_train_split(split: str) -> None:
    if split == "train":
        raise RuntimeError(
            "[GUARD] train split은 이 스크립트에서 처리 금지.\n"
            "val 또는 test만 허용."
        )


def check_no_lesion_hard_negative_in_path(path_str: str, label: str) -> None:
    # path segment 기준으로 검사 (second-stage-lesion-refiner-v1 오탐 방지)
    parts = [part.lower() for part in Path(path_str).parts]
    for part in parts:
        for kw in LESION_HARD_NEG_FORBIDDEN_SEGMENTS:
            if kw in part:
                raise RuntimeError(
                    f"[GUARD] {label}에 lesion/hard_negative 경로 감지: {path_str}\n"
                    "threshold 결정에 lesion/hard_negative 데이터 혼입 금지."
                )


# ──────────────────────────────────────────────────────────
# 모델 클래스 (train 스크립트와 동일)
# ──────────────────────────────────────────────────────────
class ConvAutoencoder2p5D(nn.Module):
    """
    Minimal reconstruction baseline for 2.5D normal crop verifier.
    NOT a full RD4AD teacher-student implementation.

    input_channels=6  (lung z-1/z/z+1, mediastinal z-1/z/z+1)
    output_channels=6 (reconstruction target == input)
    """

    def __init__(self, input_channels: int = 6, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, c, 3, padding=1),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(c, c * 2, 3, padding=1),
            nn.BatchNorm2d(c * 2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(c * 2, c * 4, 3, padding=1),
            nn.BatchNorm2d(c * 4), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(c * 4, c * 8, 3, padding=1),
            nn.BatchNorm2d(c * 8), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2),
            nn.BatchNorm2d(c * 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2),
            nn.BatchNorm2d(c * 2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 2, c, 2, stride=2),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, input_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ──────────────────────────────────────────────────────────
# 모델 로드
# ──────────────────────────────────────────────────────────
def load_model(ckpt_path: str, device: torch.device) -> ConvAutoencoder2p5D:
    check_forbidden_path(ckpt_path)
    ckpt_p = Path(ckpt_path)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"[ERROR] checkpoint 없음: {ckpt_p}")
    ckpt = torch.load(str(ckpt_p), map_location="cpu", weights_only=False)
    required_keys = ["epoch", "model_state_dict", "best_val_loss", "config"]
    missing = [k for k in required_keys if k not in ckpt]
    if missing:
        raise RuntimeError(f"[ERROR] checkpoint 누락 key: {missing}")
    model = ConvAutoencoder2p5D(input_channels=6, base_channels=32)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def get_ckpt_epoch(ckpt_path: str) -> int:
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        return int(ckpt.get("epoch", -1))
    except Exception:
        return -1


# ──────────────────────────────────────────────────────────
# ROI mask 로드
# ──────────────────────────────────────────────────────────
def load_roi_mask(safe_id: str, volume_root: str) -> np.ndarray:
    """roi_0_0.npy 로드. 반환: (Z, H, W) bool ndarray."""
    vol_dir = Path(volume_root) / "volumes_npy" / safe_id
    roi_path = vol_dir / "roi_0_0.npy"
    check_forbidden_volume_file(str(roi_path))
    check_forbidden_path(str(roi_path))
    if not roi_path.exists():
        raise FileNotFoundError(f"[ERROR] roi_0_0.npy 없음: {roi_path}")
    roi = np.load(str(roi_path))
    if roi.ndim != 3:
        raise ValueError(f"[ERROR] roi_0_0.npy shape이 3D가 아님: {roi.shape}")
    s0, s1, s2 = roi.shape
    if s0 < s1 or s0 < s2:
        roi_zwh = roi
    else:
        roi_zwh = np.transpose(roi, (2, 0, 1))
    return roi_zwh.astype(bool)


# ──────────────────────────────────────────────────────────
# ROI crop 추출 (6채널)
# ──────────────────────────────────────────────────────────
def extract_roi_crop(
    roi_volume: np.ndarray,
    z_center: int,
    y0_f96: int,
    x0_f96: int,
    y1_f96: int,
    x1_f96: int,
    z_lo: int = None,
    z_hi: int = None,
) -> tuple:
    """
    roi_volume: (Z, H, W) bool
    반환: (roi_mask_6ch (6,96,96) bool, clipped_flag bool)
    ch0,1,2 = lung (z-1,z,z+1), ch3,4,5 = mediastinal (z-1,z,z+1)
    z 범위 밖은 edge clamp.
    """
    Z = roi_volume.shape[0]
    z_c = max(0, min(z_center, Z - 1))
    z_m = max(0, z_c - 1) if z_lo is None else max(0, min(z_lo, Z - 1))
    z_p = min(Z - 1, z_c + 1) if z_hi is None else max(0, min(z_hi, Z - 1))

    roi_zm = roi_volume[z_m, y0_f96:y1_f96, x0_f96:x1_f96]
    roi_zc = roi_volume[z_c, y0_f96:y1_f96, x0_f96:x1_f96]
    roi_zp = roi_volume[z_p, y0_f96:y1_f96, x0_f96:x1_f96]

    def _ensure_96(arr):
        h, w = arr.shape
        if h == 96 and w == 96:
            return arr, False
        out = np.zeros((96, 96), dtype=bool)
        h_use = min(h, 96)
        w_use = min(w, 96)
        out[:h_use, :w_use] = arr[:h_use, :w_use]
        return out, True

    roi_zm, clipped_m = _ensure_96(roi_zm)
    roi_zc, clipped_c = _ensure_96(roi_zc)
    roi_zp, clipped_p = _ensure_96(roi_zp)
    clipped_flag = clipped_m or clipped_c or clipped_p

    roi_mask_6ch = np.stack(
        [roi_zm, roi_zc, roi_zp, roi_zm, roi_zc, roi_zp], axis=0
    ).astype(bool)
    return roi_mask_6ch, clipped_flag


# ──────────────────────────────────────────────────────────
# z_indices_used 파싱
# ──────────────────────────────────────────────────────────
def _parse_z_indices(z_indices_str) -> list:
    if pd.isna(z_indices_str) or z_indices_str == "":
        return None
    try:
        val = ast.literal_eval(str(z_indices_str))
        if isinstance(val, (list, tuple)) and len(val) >= 3:
            return [int(v) for v in val]
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────────────
# ROI score 계산
# ──────────────────────────────────────────────────────────
def compute_roi_scores(
    error: np.ndarray,
    roi_mask_6ch: np.ndarray,
    stored_l1_mean: float,
) -> dict:
    """
    error: (6,96,96) float32 = |recon - input|
    roi_mask_6ch: (6,96,96) bool
    반환: score 컬럼 dict
    """
    error_flat = error.flatten()
    roi_flat = roi_mask_6ch.flatten()

    normal_rd4ad_l1_whole_crop_recomputed = float(error.mean())
    normal_rd4ad_l1_whole_crop_stored = float(stored_l1_mean)
    normal_rd4ad_l1_whole_crop_match = (
        abs(normal_rd4ad_l1_whole_crop_recomputed - normal_rd4ad_l1_whole_crop_stored) < 1e-4
    )

    if roi_flat.sum() > 0:
        normal_rd4ad_l1_roi_mean = float(error_flat[roi_flat].mean())
    else:
        normal_rd4ad_l1_roi_mean = float("nan")

    outside_roi_flat = ~roi_flat
    if outside_roi_flat.sum() > 0:
        normal_rd4ad_l1_outside_roi_mean = float(error_flat[outside_roi_flat].mean())
    else:
        normal_rd4ad_l1_outside_roi_mean = float("nan")

    lung_roi = roi_mask_6ch[0:3].flatten()
    lung_error = error[0:3].flatten()
    if lung_roi.sum() > 0:
        normal_rd4ad_l1_lung_channel_roi_mean = float(lung_error[lung_roi].mean())
    else:
        normal_rd4ad_l1_lung_channel_roi_mean = float("nan")

    med_roi = roi_mask_6ch[3:6].flatten()
    med_error = error[3:6].flatten()
    if med_roi.sum() > 0:
        normal_rd4ad_l1_med_channel_roi_mean = float(med_error[med_roi].mean())
    else:
        normal_rd4ad_l1_med_channel_roi_mean = float("nan")

    roi_z_center = roi_mask_6ch[1]
    roi_pixels_fixed96 = int(roi_z_center.sum())
    roi_ratio_fixed96 = roi_pixels_fixed96 / (96 * 96)
    outside_roi_pixels_fixed96 = (96 * 96) - roi_pixels_fixed96
    low_roi_ratio_flag = roi_ratio_fixed96 < LOW_ROI_RATIO_THRESHOLD

    return {
        "normal_rd4ad_l1_whole_crop_recomputed": normal_rd4ad_l1_whole_crop_recomputed,
        "normal_rd4ad_l1_whole_crop_stored": normal_rd4ad_l1_whole_crop_stored,
        "normal_rd4ad_l1_whole_crop_match": normal_rd4ad_l1_whole_crop_match,
        "normal_rd4ad_l1_roi_mean": normal_rd4ad_l1_roi_mean,
        "normal_rd4ad_l1_outside_roi_mean": normal_rd4ad_l1_outside_roi_mean,
        "normal_rd4ad_l1_lung_channel_roi_mean": normal_rd4ad_l1_lung_channel_roi_mean,
        "normal_rd4ad_l1_med_channel_roi_mean": normal_rd4ad_l1_med_channel_roi_mean,
        "roi_pixels_fixed96": roi_pixels_fixed96,
        "roi_ratio_fixed96": roi_ratio_fixed96,
        "outside_roi_pixels_fixed96": outside_roi_pixels_fixed96,
        "low_roi_ratio_flag": low_roi_ratio_flag,
    }


# ──────────────────────────────────────────────────────────
# crop 1개 처리
# ──────────────────────────────────────────────────────────
def process_one_crop(
    row: pd.Series,
    stored_score_map: dict,
    model: ConvAutoencoder2p5D,
    device: torch.device,
    volume_root: str,
) -> dict:
    """단일 normal val/test crop 처리. 반환: result dict."""
    crop_id = str(row["crop_id"])
    patient_id = str(row["patient_id"])
    safe_id = str(row.get("safe_id", patient_id))
    crop_path = str(row["crop_path"])
    split = str(row.get("normal_split", ""))

    result_base = {
        "patient_id": patient_id,
        "safe_id": safe_id,
        "split": split,
        "crop_id": crop_id,
        "crop_path": crop_path,
        "error_reason": None,
        "roi_crop_clipped_flag": False,
    }

    # guard
    try:
        check_forbidden_path(crop_path)
        check_not_train_split(split)
        check_no_lesion_hard_negative_in_path(crop_path, "crop_path")
    except RuntimeError as e:
        result_base["error_reason"] = str(e)
        return result_base

    # 1. crop npz 로드
    crop_p = Path(crop_path)
    if not crop_p.exists():
        result_base["error_reason"] = f"crop npz 없음: {crop_path}"
        return result_base
    try:
        data = np.load(str(crop_p))
    except Exception as e:
        result_base["error_reason"] = f"crop npz 로드 실패: {e}"
        return result_base
    if "image" not in data:
        result_base["error_reason"] = "crop npz에 'image' key 없음"
        return result_base
    image = data["image"].astype(np.float32)
    if image.shape != (6, 96, 96):
        result_base["error_reason"] = f"image shape 불일치: {image.shape}"
        return result_base

    # stored score 조회 (None이면 NaN 사용)
    stored_l1_mean = stored_score_map.get(crop_id, float("nan"))

    # 2. model forward
    inp = torch.from_numpy(image).unsqueeze(0).to(device)
    try:
        with torch.no_grad():
            recon = model(inp)
    except Exception as e:
        result_base["error_reason"] = f"model forward 실패: {e}"
        return result_base
    recon_np = recon.squeeze(0).cpu().numpy().astype(np.float32)
    error = np.abs(recon_np - image)

    if np.any(np.isnan(error)) or np.any(np.isinf(error)):
        result_base["error_reason"] = "error map에 NaN/Inf 감지"

    # 3. ROI mask 로드
    row_volume_root = str(row.get("volume_root", volume_root)) if volume_root == "" else volume_root
    if row_volume_root == "":
        row_volume_root = DEFAULT_VOLUME_ROOT
    z_center = int(row["z_center"])
    crop_y0 = int(row["crop_y0"])
    crop_x0 = int(row["crop_x0"])
    crop_y1 = int(row["crop_y1"])
    crop_x1 = int(row["crop_x1"])

    z_lo_val = None
    z_hi_val = None
    if "z_lo" in row.index:
        z_lo_val = int(row["z_lo"])
    if "z_hi" in row.index:
        z_hi_val = int(row["z_hi"])
    # z_indices_used 우선 사용
    if "z_indices_used" in row.index:
        parsed = _parse_z_indices(row["z_indices_used"])
        if parsed is not None and len(parsed) >= 3:
            z_lo_val = parsed[0]
            z_hi_val = parsed[2]

    roi_mask_6ch = None
    try:
        roi_volume = load_roi_mask(safe_id, row_volume_root)
        roi_mask_6ch, clipped_flag = extract_roi_crop(
            roi_volume, z_center, crop_y0, crop_x0, crop_y1, crop_x1,
            z_lo=z_lo_val, z_hi=z_hi_val,
        )
        result_base["roi_crop_clipped_flag"] = clipped_flag
    except Exception as e:
        err_msg = f"ROI 로드 실패: {e}"
        result_base["error_reason"] = (
            (result_base["error_reason"] + " | " + err_msg)
            if result_base["error_reason"]
            else err_msg
        )
        roi_mask_6ch = np.zeros((6, 96, 96), dtype=bool)

    # 4. ROI score 계산
    scores = compute_roi_scores(error, roi_mask_6ch, stored_l1_mean)
    result_base.update(scores)

    return result_base


# ──────────────────────────────────────────────────────────
# split 필터
# ──────────────────────────────────────────────────────────
def filter_splits(df: pd.DataFrame, split_arg: str) -> pd.DataFrame:
    """--split val/test/all 처리. train 포함 시 강제 제외 후 경고."""
    split_col = "normal_split"
    if split_col not in df.columns:
        raise RuntimeError(f"[ERROR] manifest에 '{split_col}' 컬럼 없음.")

    if "train" in df[split_col].unique():
        n_train = int((df[split_col] == "train").sum())
        print(f"[WARN] train split {n_train}개 감지 → 강제 제외. full-run 대상 아님.")
        df = df[df[split_col] != "train"].reset_index(drop=True)

    if split_arg == "val":
        result = df[df[split_col] == "val"].reset_index(drop=True)
    elif split_arg == "test":
        result = df[df[split_col] == "test"].reset_index(drop=True)
    else:  # "all" = val + test
        result = df[df[split_col].isin(["val", "test"])].reset_index(drop=True)

    if len(result) == 0:
        raise RuntimeError(f"[ERROR] split='{split_arg}' 처리 대상 crop 없음.")
    return result


# ──────────────────────────────────────────────────────────
# device 해석
# ──────────────────────────────────────────────────────────
def resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda_if_available":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("[ERROR] --device cuda 지정했지만 CUDA 없음.")
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# ──────────────────────────────────────────────────────────
# 통계 dict 유틸
# ──────────────────────────────────────────────────────────
def _stat_dict(values: list) -> dict:
    arr = np.array(
        [v for v in values if v is not None and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))],
        dtype=float,
    )
    if len(arr) == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None,
                "p50": None, "p90": None, "p95": None, "p99": None}
    return {
        "n": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


# ──────────────────────────────────────────────────────────
# Summary JSON 생성
# ──────────────────────────────────────────────────────────
def generate_summary_json(
    rows: list,
    args,
    n_success: int,
    n_error: int,
    run_ts: str,
    ckpt_epoch: int,
) -> dict:
    df = pd.DataFrame(rows)

    def _col(col):
        if col not in df.columns:
            return []
        return df[col].tolist()

    def _split_col(split_val, col):
        if col not in df.columns or "split" not in df.columns:
            return []
        return df.loc[df["split"] == split_val, col].tolist()

    whole_crop_match_count = int(df["normal_rd4ad_l1_whole_crop_match"].sum()) \
        if "normal_rd4ad_l1_whole_crop_match" in df.columns else None
    low_roi_ratio_count = int(df["low_roi_ratio_flag"].sum()) \
        if "low_roi_ratio_flag" in df.columns else None
    clipped_count = int(df["roi_crop_clipped_flag"].sum()) \
        if "roi_crop_clipped_flag" in df.columns else None
    n_val = int((df["split"] == "val").sum()) if "split" in df.columns else 0
    n_test = int((df["split"] == "test").sum()) if "split" in df.columns else 0

    summary = {
        "script": "compute_rd4ad_normal_val_test_roi_scores.py",
        "phase": "5.60",
        "run_timestamp": run_ts,
        "checkpoint_epoch": ckpt_epoch,
        "output_tag": args.output_tag,
        "split_arg": args.split,
        "n_total": len(df),
        "n_val": n_val,
        "n_test": n_test,
        "n_success": n_success,
        "n_error": n_error,
        "whole_crop_score_match_count": whole_crop_match_count,
        "low_roi_ratio_count": low_roi_ratio_count,
        "roi_crop_clipped_count": clipped_count,
        "roi_ratio_fixed96": {
            "val": _stat_dict(_split_col("val", "roi_ratio_fixed96")),
            "test": _stat_dict(_split_col("test", "roi_ratio_fixed96")),
        },
        "normal_rd4ad_l1_roi_mean": {
            "val": _stat_dict(_split_col("val", "normal_rd4ad_l1_roi_mean")),
            "test": _stat_dict(_split_col("test", "normal_rd4ad_l1_roi_mean")),
        },
        "normal_rd4ad_l1_lung_channel_roi_mean": {
            "val": _stat_dict(_split_col("val", "normal_rd4ad_l1_lung_channel_roi_mean")),
            "test": _stat_dict(_split_col("test", "normal_rd4ad_l1_lung_channel_roi_mean")),
        },
        "normal_rd4ad_l1_med_channel_roi_mean": {
            "val": _stat_dict(_split_col("val", "normal_rd4ad_l1_med_channel_roi_mean")),
            "test": _stat_dict(_split_col("test", "normal_rd4ad_l1_med_channel_roi_mean")),
        },
        "normal_rd4ad_l1_outside_roi_mean": {
            "val": _stat_dict(_split_col("val", "normal_rd4ad_l1_outside_roi_mean")),
            "test": _stat_dict(_split_col("test", "normal_rd4ad_l1_outside_roi_mean")),
        },
        "normal_rd4ad_l1_whole_crop_recomputed": {
            "val": _stat_dict(_split_col("val", "normal_rd4ad_l1_whole_crop_recomputed")),
            "test": _stat_dict(_split_col("test", "normal_rd4ad_l1_whole_crop_recomputed")),
        },
        "limitation": (
            "기존 모델은 whole-crop L1로 학습됨. "
            "이 분석은 diagnostic ROI score 추가이며, threshold 산출은 이 스크립트에서 하지 않음. "
            "test는 threshold tuning에 쓰면 안 됨. "
            "stage2_holdout 미사용. "
            "이 결과는 QA 후보 내 분석이며 전체 모델 성능 결론 아님."
        ),
    }
    return summary


# ──────────────────────────────────────────────────────────
# MD report 생성
# ──────────────────────────────────────────────────────────
def generate_md_report(rows: list, summary: dict, run_ts: str) -> str:
    df = pd.DataFrame(rows)
    lines = [
        "# Phase 5.60 Normal Val/Test ROI-Masked Score Diagnostic Report",
        "",
        f"- run_timestamp: {run_ts}",
        f"- checkpoint_epoch: {summary.get('checkpoint_epoch')}",
        f"- output_tag: {summary.get('output_tag')}",
        f"- split_arg: {summary.get('split_arg')}",
        "",
        "## 요약",
        f"- n_total: {summary['n_total']}",
        f"- n_val: {summary['n_val']}",
        f"- n_test: {summary['n_test']}",
        f"- n_success: {summary['n_success']}",
        f"- n_error: {summary['n_error']}",
        f"- whole_crop_score_match_count: {summary['whole_crop_score_match_count']}",
        f"- low_roi_ratio_count (roi_ratio < {LOW_ROI_RATIO_THRESHOLD}): {summary['low_roi_ratio_count']}",
        f"- roi_crop_clipped_count: {summary['roi_crop_clipped_count']}",
        "",
        "## Split별 ROI Ratio 통계",
    ]
    for sp in ["val", "test"]:
        st = summary["roi_ratio_fixed96"].get(sp, {})
        if st and st.get("mean") is not None:
            lines.append(
                f"- {sp}: mean={st['mean']:.4f}, std={st['std']:.4f}, "
                f"min={st['min']:.4f}, max={st['max']:.4f}, "
                f"p50={st['p50']:.4f}, p90={st['p90']:.4f}"
            )
    lines += [
        "",
        "## Split별 ROI Score 통계",
    ]
    for score_key in [
        "normal_rd4ad_l1_roi_mean",
        "normal_rd4ad_l1_lung_channel_roi_mean",
        "normal_rd4ad_l1_med_channel_roi_mean",
        "normal_rd4ad_l1_outside_roi_mean",
        "normal_rd4ad_l1_whole_crop_recomputed",
    ]:
        lines.append(f"### {score_key}")
        for sp in ["val", "test"]:
            st = summary.get(score_key, {}).get(sp, {})
            if st and st.get("mean") is not None:
                lines.append(
                    f"- {sp}: mean={st['mean']:.6f}, std={st['std']:.6f}, "
                    f"p50={st['p50']:.6f}, p90={st['p90']:.6f}, "
                    f"p95={st['p95']:.6f}, p99={st['p99']:.6f}"
                )
            else:
                lines.append(f"- {sp}: (데이터 없음)")
    lines += [
        "",
        "## Limitation",
        summary.get("limitation", ""),
        "",
        "## Note — threshold 산출 금지",
        "- 이 보고서는 score 분포 기록만 포함.",
        "- threshold 후보 산출은 Phase 5.63에서 별도 수행.",
        "- normal test는 threshold tuning에 사용 금지.",
        "- stage2_holdout 미사용.",
        "",
        "---",
        "*이 보고서는 자동 생성된 diagnostic 문서임. 임상 결론에 사용 금지.*",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────
# preflight
# ──────────────────────────────────────────────────────────
def run_preflight(args, project_root: Path) -> None:
    print("=" * 60)
    print("[PREFLIGHT] Phase 5.60 normal val/test ROI score 입력 검증")
    print("=" * 60)
    print("[PREFLIGHT] model forward 없음. CSV/JSON/MD 저장 없음.")

    errors = []
    warnings = []
    ok_items = []

    # 1. forbidden path guard — 주요 경로 점검
    paths_to_guard = [
        args.checkpoint, args.crop_manifest, args.output_root,
        args.existing_score_dir, args.volume_root,
    ]
    any_forbidden = False
    for fp in paths_to_guard:
        for kw in FORBIDDEN_PATH_KEYWORDS:
            if kw in str(fp).lower():
                errors.append(f"[ERROR] 금지 키워드 '{kw}' 경로 감지: {fp}")
                any_forbidden = True
    if not any_forbidden:
        ok_items.append("[OK] 금지 경로 키워드 없음 (stage2_holdout/holdout/v2)")

    # 2. manifest 존재 및 row 수
    mf_p = project_root / args.crop_manifest
    if mf_p.exists():
        df_mf = pd.read_csv(mf_p)
        total = len(df_mf)
        ok_items.append(f"[OK] manifest 존재: {mf_p} ({total} rows)")
        if total != EXPECTED_TOTAL_MANIFEST:
            warnings.append(f"[WARN] manifest total rows={total}, expected={EXPECTED_TOTAL_MANIFEST}")
        else:
            ok_items.append(f"[OK] manifest total rows: {total}")

        # val/test row 수
        if "normal_split" in df_mf.columns:
            n_val = int((df_mf["normal_split"] == "val").sum())
            n_test = int((df_mf["normal_split"] == "test").sum())
            n_train = int((df_mf["normal_split"] == "train").sum())
            ok_items.append(f"[OK] val rows: {n_val}, test rows: {n_test}")
            if n_val != EXPECTED_VAL:
                warnings.append(f"[WARN] val rows={n_val}, expected={EXPECTED_VAL}")
            if n_test != EXPECTED_TEST:
                warnings.append(f"[WARN] test rows={n_test}, expected={EXPECTED_TEST}")
            if n_train > 0:
                ok_items.append(f"[OK] train rows={n_train} → full-run 시 자동 제외됨")
        else:
            errors.append("[ERROR] manifest에 'normal_split' 컬럼 없음")

        # 필수 좌표 컬럼 확인
        required_coord_cols = ["z_center", "crop_y0", "crop_x0", "crop_y1", "crop_x1",
                               "safe_id", "patient_id", "crop_path", "crop_id"]
        missing_cols = [c for c in required_coord_cols if c not in df_mf.columns]
        if missing_cols:
            errors.append(f"[ERROR] manifest 누락 컬럼: {missing_cols}")
        else:
            ok_items.append("[OK] manifest 필수 컬럼 모두 존재")

        # sample crop npz 확인 (val/test 각 1개)
        for sp in ["val", "test"]:
            if "normal_split" not in df_mf.columns:
                break
            sp_df = df_mf[df_mf["normal_split"] == sp]
            if len(sp_df) == 0:
                warnings.append(f"[WARN] split={sp} row 없음")
                continue
            sample_row = sp_df.iloc[0]
            cp = str(sample_row.get("crop_path", ""))
            if not Path(cp).exists():
                errors.append(f"[ERROR] split={sp} sample crop 없음: {cp}")
            else:
                try:
                    d = np.load(cp)
                    keys = list(d.keys())
                    if "image" not in keys:
                        errors.append(f"[ERROR] split={sp} sample npz: 'image' key 없음 (keys={keys})")
                    elif d["image"].shape != (6, 96, 96):
                        errors.append(f"[ERROR] split={sp} sample npz shape={d['image'].shape}")
                    elif d["image"].dtype != np.float32:
                        warnings.append(f"[WARN] split={sp} sample npz dtype={d['image'].dtype}")
                    else:
                        ok_items.append(f"[OK] split={sp} sample npz: keys={keys}, shape={d['image'].shape}")
                except Exception as e:
                    errors.append(f"[ERROR] split={sp} sample npz 로드 실패: {e}")
    else:
        errors.append(f"[ERROR] manifest 없음: {mf_p}")
        df_mf = None

    # 3. checkpoint 존재 및 key 확인
    ckpt_p = project_root / args.checkpoint
    if ckpt_p.exists():
        try:
            ckpt = torch.load(str(ckpt_p), map_location="cpu", weights_only=False)
            req_keys = ["epoch", "model_state_dict", "best_val_loss", "config"]
            missing_k = [k for k in req_keys if k not in ckpt]
            if missing_k:
                errors.append(f"[ERROR] checkpoint 누락 key: {missing_k}")
            else:
                ok_items.append(f"[OK] checkpoint 존재 및 key 확인: epoch={ckpt['epoch']}")
        except Exception as e:
            errors.append(f"[ERROR] checkpoint 로드 실패: {e}")
    else:
        errors.append(f"[ERROR] checkpoint 없음: {ckpt_p}")

    # 4. existing score CSV 존재
    score_dir = project_root / args.existing_score_dir
    for sp in ["val", "test"]:
        csv_p = score_dir / f"normal_{sp}_scores_v1.csv"
        if csv_p.exists():
            ok_items.append(f"[OK] existing {sp} score CSV: {csv_p}")
        else:
            warnings.append(f"[WARN] existing {sp} score CSV 없음 (stored_score는 NaN으로 처리됨): {csv_p}")

    # 5. volume_root 및 sample roi_0_0.npy
    vol_root = Path(args.volume_root)
    if vol_root.exists():
        ok_items.append(f"[OK] volume_root 접근 가능: {vol_root}")
        # sample 3명 roi_0_0.npy 확인
        if df_mf is not None and "safe_id" in df_mf.columns:
            sample_ids = df_mf[df_mf.get("normal_split", pd.Series()) != "train"]["safe_id"].unique()[:3] \
                if "normal_split" in df_mf.columns else df_mf["safe_id"].unique()[:3]
            for sid in sample_ids:
                roi_p = vol_root / "volumes_npy" / sid / "roi_0_0.npy"
                if roi_p.exists():
                    ok_items.append(f"[OK] roi_0_0.npy 존재: {sid}")
                else:
                    errors.append(f"[ERROR] roi_0_0.npy 없음: {roi_p}")
    else:
        errors.append(f"[ERROR] volume_root 없음: {vol_root}")

    # 6. output root 충돌 확인
    out_root = project_root / args.output_root
    if out_root.exists():
        if not args.force:
            warnings.append(
                f"[WARN] output root 이미 존재: {out_root}\n"
                "       full-run 시 --force 없으면 RuntimeError 발생"
            )
        else:
            warnings.append(f"[WARN] --force 지정됨. output root 덮어쓰기 허용: {out_root}")
    else:
        ok_items.append(f"[OK] output root 없음 (신규 생성 예정)")

    # 7. train split이 target에 포함되지 않는지
    ok_items.append("[OK] train split 제외 guard 활성화됨 (split_arg로 val/test/all만 허용)")

    # 8. threshold 산출 금지 note
    ok_items.append("[OK] threshold 산출 없음 (Phase 5.63에서 별도 수행)")
    ok_items.append("[OK] test split은 threshold tuning에 사용 금지 (참고 FP rate 확인용만 허용)")

    # 결과 출력
    print("\n[PREFLIGHT] 정상 항목:")
    for item in ok_items:
        print(f"  {item}")
    if warnings:
        print("\n[PREFLIGHT] 경고 항목:")
        for w in warnings:
            print(f"  {w}")
    if errors:
        print("\n[PREFLIGHT] 오류 항목:")
        for e in errors:
            print(f"  {e}")
        print(f"\n[PREFLIGHT] 결과: {len(errors)}개 오류, {len(warnings)}개 경고 → 수정 후 재실행 필요")
        sys.exit(1)
    else:
        print(f"\n[PREFLIGHT] 결과: OK (오류 0, 경고 {len(warnings)})")
        print("[PREFLIGHT] dry-run 또는 full-run 진행 가능.")


# ──────────────────────────────────────────────────────────
# existing score CSV 로드 → crop_id: stored_score map 구성
# ──────────────────────────────────────────────────────────
def load_stored_score_map(score_dir: Path, splits: list) -> dict:
    """normal_val/test_scores_v1.csv에서 crop_id → crop_score_l1_mean 매핑."""
    result = {}
    for sp in splits:
        csv_p = score_dir / f"normal_{sp}_scores_v1.csv"
        if not csv_p.exists():
            print(f"[WARN] stored score CSV 없음 (stored는 NaN 처리): {csv_p}")
            continue
        df = pd.read_csv(csv_p, usecols=["crop_id", "crop_score_l1_mean"])
        for _, row in df.iterrows():
            result[str(row["crop_id"])] = float(row["crop_score_l1_mean"])
    return result


# ──────────────────────────────────────────────────────────
# dry-run
# ──────────────────────────────────────────────────────────
def run_dry_run(args, project_root: Path) -> None:
    print("=" * 60)
    print("[DRY-RUN] Phase 5.60 normal val/test ROI score diagnostic")
    print("=" * 60)
    print("[DRY-RUN] CSV/JSON/MD 저장 금지. 기존 파일 미수정.")

    device = resolve_device(args.device)
    mf_p = project_root / args.crop_manifest
    df_mf = pd.read_csv(mf_p)
    df_target = filter_splits(df_mf, args.split)

    if args.target_crop_ids:
        ids = set(str(x) for x in args.target_crop_ids.split(","))
        df_target = df_target[df_target["crop_id"].astype(str).isin(ids)].reset_index(drop=True)
        if len(df_target) == 0:
            print(f"[DRY-RUN][WARN] target_crop_ids에 해당하는 row 없음")
            return
    else:
        limit = args.limit if args.limit is not None else 3
        df_target = df_target.head(limit).reset_index(drop=True)

    print(f"[DRY-RUN] 처리 예정: {len(df_target)}개")

    model = load_model(str(project_root / args.checkpoint), device)
    print(f"[DRY-RUN] 모델 로드 완료")

    stored_score_map = load_stored_score_map(
        project_root / args.existing_score_dir, ["val", "test"]
    )
    vol_root = args.volume_root

    for i, row in df_target.iterrows():
        crop_id_str = str(row["crop_id"])
        print(f"\n[DRY-RUN] crop {i+1}/{len(df_target)}: crop_id={crop_id_str}")
        result = process_one_crop(row, stored_score_map, model, device, vol_root)

        print_keys = [
            "patient_id", "split", "crop_id",
            "normal_rd4ad_l1_whole_crop_stored",
            "normal_rd4ad_l1_whole_crop_recomputed",
            "normal_rd4ad_l1_whole_crop_match",
            "normal_rd4ad_l1_roi_mean",
            "normal_rd4ad_l1_outside_roi_mean",
            "normal_rd4ad_l1_lung_channel_roi_mean",
            "normal_rd4ad_l1_med_channel_roi_mean",
            "roi_pixels_fixed96", "roi_ratio_fixed96",
            "outside_roi_pixels_fixed96", "low_roi_ratio_flag",
            "roi_crop_clipped_flag", "error_reason",
        ]
        for k in print_keys:
            v = result.get(k, "N/A")
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

        for k, v in result.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                print(f"  [WARN] NaN/Inf: {k}={v}")

    print("\n[DRY-RUN] 완료. CSV/JSON/MD 미저장.")


# ──────────────────────────────────────────────────────────
# full-run
# ──────────────────────────────────────────────────────────
def run_full_run(args, project_root: Path) -> None:
    print("=" * 60)
    print("[FULL-RUN] Phase 5.60 normal val/test ROI score diagnostic")
    print("=" * 60)

    out_root = project_root / args.output_root
    check_forbidden_path(str(out_root))
    if out_root.exists() and not args.force:
        raise RuntimeError(
            f"[GUARD] output root 이미 존재: {out_root}\n"
            "--force 없이 full-run 금지."
        )
    if out_root.exists() and args.force:
        print(f"[WARN] --force 지정됨. 기존 output root 덮어쓰기: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    mf_p = project_root / args.crop_manifest
    df_mf = pd.read_csv(mf_p)
    df_target = filter_splits(df_mf, args.split)

    if args.limit is not None:
        df_target = df_target.head(args.limit).reset_index(drop=True)

    print(f"[FULL-RUN] 처리 대상: {len(df_target)}개 crops")
    n_val_target = int((df_target["normal_split"] == "val").sum()) if "normal_split" in df_target.columns else 0
    n_test_target = int((df_target["normal_split"] == "test").sum()) if "normal_split" in df_target.columns else 0
    print(f"[FULL-RUN] val: {n_val_target}, test: {n_test_target}")

    model = load_model(str(project_root / args.checkpoint), device)
    ckpt_epoch = get_ckpt_epoch(str(project_root / args.checkpoint))
    print(f"[FULL-RUN] 모델 로드 완료: epoch={ckpt_epoch}")

    stored_score_map = load_stored_score_map(
        project_root / args.existing_score_dir, ["val", "test"]
    )
    vol_root = args.volume_root

    all_results = []
    n_success = 0
    n_error = 0

    for i, row in df_target.iterrows():
        crop_id_str = str(row["crop_id"])
        if (i + 1) % 100 == 0 or i == 0:
            print(f"[FULL-RUN] {i+1}/{len(df_target)}: crop_id={crop_id_str}")
        try:
            result = process_one_crop(row, stored_score_map, model, device, vol_root)
        except Exception as e:
            result = {
                "patient_id": str(row.get("patient_id", "")),
                "safe_id": str(row.get("safe_id", "")),
                "split": str(row.get("normal_split", "")),
                "crop_id": crop_id_str,
                "crop_path": str(row.get("crop_path", "")),
                "error_reason": f"process_one_crop 예외: {e}",
            }
        if result.get("error_reason"):
            n_error += 1
        else:
            n_success += 1
        all_results.append(result)

    print(f"\n[FULL-RUN] 완료: n_success={n_success}, n_error={n_error}")

    run_ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    df_result = pd.DataFrame(all_results)
    csv_path = out_root / OUTPUT_CSV_NAME
    df_result.to_csv(str(csv_path), index=False)
    print(f"[SAVE] CSV: {csv_path} ({len(df_result)} rows)")

    summary = generate_summary_json(all_results, args, n_success, n_error, run_ts, ckpt_epoch)
    json_path = out_root / OUTPUT_JSON_NAME
    with open(str(json_path), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] JSON: {json_path}")

    md_text = generate_md_report(all_results, summary, run_ts)
    md_path = out_root / OUTPUT_MD_NAME
    with open(str(md_path), "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"[SAVE] MD: {md_path}")
    print(f"\n[FULL-RUN] 결과 저장 완료: {out_root}")
    print("[NOTE] threshold 산출 없음. Phase 5.63에서 별도 수행.")
    print("[NOTE] test는 threshold tuning에 사용 금지. 참고 FP rate 확인용만.")


# ──────────────────────────────────────────────────────────
# argparse
# ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 5.60 - Normal val/test ROI-masked score diagnostic"
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--preflight-only", action="store_true",
        help="입력 검증만. model forward 없음. CSV/JSON/MD 저장 없음.",
    )
    mode_group.add_argument(
        "--dry-run", action="store_true",
        help="소량 crop model forward + 콘솔 출력. 결과 저장 금지.",
    )
    mode_group.add_argument(
        "--full-run", action="store_true",
        help="val/test 전체 처리 + CSV/JSON/MD 저장.",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test", "all"],
        default="all",
        help="처리 대상 split. val/test/all(=val+test). train 포함 시 강제 제외.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="처리 최대 crop 수 (dry-run 기본 3, full-run 기본 전체).",
    )
    parser.add_argument(
        "--target-crop-ids", type=str, default=None,
        help="쉼표 구분 crop_id 목록 (dry-run 전용).",
    )
    parser.add_argument(
        "--device",
        choices=["cuda_if_available", "cuda", "cpu"],
        default="cuda_if_available",
        help="실행 device (기본: cuda_if_available).",
    )
    parser.add_argument(
        "--output-tag", type=str, default="normal_val_test_roi_scores_v1",
        help="출력 태그.",
    )
    parser.add_argument(
        "--volume-root", type=str, default=DEFAULT_VOLUME_ROOT,
        help="normal volume root (roi_0_0.npy 포함 폴더).",
    )
    parser.add_argument(
        "--crop-manifest", type=str, default=DEFAULT_CROP_MANIFEST,
        help="normal crop manifest CSV 경로.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
        help="checkpoint 경로.",
    )
    parser.add_argument(
        "--existing-score-dir", type=str, default=DEFAULT_EXISTING_SCORE_DIR,
        help="기존 whole-crop score CSV가 있는 폴더.",
    )
    parser.add_argument(
        "--output-root", type=str, default=DEFAULT_OUTPUT_ROOT,
        help="결과 저장 root 폴더.",
    )
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="output root 덮어쓰기 허용 (기본 False).",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────
def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    print(f"[INFO] project_root: {project_root}")
    print(f"[INFO] mode: {'preflight-only' if args.preflight_only else 'dry-run' if args.dry_run else 'full-run'}")
    print(f"[INFO] split: {args.split}")
    print(f"[INFO] run_timestamp: {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")

    # guard
    check_forbidden_path(str(project_root / args.output_root))
    check_forbidden_path(str(project_root / args.checkpoint))
    check_forbidden_path(str(project_root / args.crop_manifest))
    check_no_lesion_hard_negative_in_path(args.volume_root, "--volume-root")

    if args.preflight_only:
        run_preflight(args, project_root)
    elif args.dry_run:
        run_dry_run(args, project_root)
    elif args.full_run:
        run_full_run(args, project_root)
    else:
        print("[ERROR] 모드 지정 필요: --preflight-only / --dry-run / --full-run")
        sys.exit(1)


if __name__ == "__main__":
    main()
