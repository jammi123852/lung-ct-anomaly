"""
Phase 5.51 - RD4AD ROI/patch-masked score diagnostic script

기존 whole-crop L1 score 보존 + ROI/patch-masked diagnostic score 컬럼 추가 계산.
모델 재학습 없음. 추론(forward pass)만 수행.
이 스크립트는 diagnostic 전용이며, threshold 재확정 / 병변 성능 결론 금지.

실행 금지: full-run 외 자동 실행 없음.
py_compile 통과만 확인.
"""

import argparse
import ast
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────
# 입력 경로 상수 (하드코딩)
# ──────────────────────────────────────────────────────────
BASE_EVAL = (
    "outputs/second-stage-lesion-refiner-v1/evaluation/"
    "rd4ad_2p5d_normal_mw_fixed96_v1"
)
QA_DIR = f"{BASE_EVAL}/hard_negative_top_score_qa_v1"

QA_MANIFEST_PATH = (
    f"{QA_DIR}/hard_negative_top_score_qa_manifest_v1.csv"
)
OVERLAP_ANALYSIS_PATH = (
    f"{QA_DIR}/hard_negative_top_score_lesion_overlap_analysis_v1.csv"
)
CROP_MANIFEST_PATH = (
    "outputs/second-stage-lesion-refiner-v1/crops/"
    "rd4ad_train_2p5d_mw_fixed96_thr001_v1/manifests/"
    "crop_manifest_rd4ad_train_2p5d_mw_fixed96_thr001_v1.csv"
)
CHECKPOINT_PATH = (
    "outputs/second-stage-lesion-refiner-v1/models/"
    "rd4ad_2p5d_normal_mw_fixed96_v1/checkpoints/best_val_loss.pt"
)
NORMAL_SCORE_SUMMARY_PATH = (
    f"{BASE_EVAL}/normal_val_test_scores_v1/"
    "normal_val_test_score_summary_v1.json"
)
VOLUME_SOURCE_ROOT = (
    "/mnt/c/Users/jinhy/Desktop/"
    "NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1"
)

OUTPUT_ROOT = (
    f"{QA_DIR}/roi_patch_masked_score_diagnostic_v1"
)

OUTPUT_CSV_NAME = "rd4ad_roi_patch_masked_scores_v1.csv"
OUTPUT_JSON_NAME = "rd4ad_roi_patch_masked_scores_summary_v1.json"
OUTPUT_MD_NAME = "phase5_51_rd4ad_roi_patch_masked_score_diagnostic_v1.md"

QA_EXPECTED_ROWS = 150

# ──────────────────────────────────────────────────────────
# Guard 상수
# ──────────────────────────────────────────────────────────
FORBIDDEN_PATH_KEYWORDS = ["stage2_holdout", "holdout", "v2"]
FORBIDDEN_VOLUME_EXTENSIONS = [".csv", ".manifest", ".split", ".json"]
FORBIDDEN_VOLUME_FILENAMES = ["manifest", "score", "split", "eval", "patch_score"]


# ──────────────────────────────────────────────────────────
# 모델 클래스 (train 스크립트에서 동일하게 복사)
# ──────────────────────────────────────────────────────────
class ConvAutoencoder2p5D(nn.Module):
    """
    Minimal reconstruction baseline for 2.5D normal crop verifier.
    NOT a full RD4AD teacher-student implementation.

    input_channels=6  (lung z-1/z/z+1, mediastinal z-1/z/z+1)
    output_channels=6 (reconstruction target == input)
    Encoder: 96->48->24->12 (3x MaxPool2d)
    Decoder: 12->24->48->96 (3x ConvTranspose2d) + Sigmoid
    """

    def __init__(self, input_channels=6, base_channels=32):
        super().__init__()
        c = base_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c, c * 2, 3, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c * 2, c * 4, 3, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c * 4, c * 8, 3, padding=1),
            nn.BatchNorm2d(c * 8),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 2, c, 2, stride=2),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, input_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ──────────────────────────────────────────────────────────
# Guard 함수
# ──────────────────────────────────────────────────────────
def check_forbidden_path(path_str: str) -> None:
    """경로에 금지 키워드가 포함되면 RuntimeError."""
    p = str(path_str).lower()
    for kw in FORBIDDEN_PATH_KEYWORDS:
        if kw in p:
            raise RuntimeError(
                f"[GUARD] 금지된 경로 키워드 '{kw}' 감지: {path_str}\n"
                "stage2_holdout/holdout/v2 경로 접근 금지."
            )


def check_forbidden_volume_file(path_str: str) -> None:
    """volume source 내부에서 금지 파일 접근 시 RuntimeError."""
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


# ──────────────────────────────────────────────────────────
# 모델 로드
# ──────────────────────────────────────────────────────────
def load_model(ckpt_path: str, device: torch.device) -> ConvAutoencoder2p5D:
    """checkpoint에서 모델 로드. model.eval() 상태 반환. torch.save 없음."""
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
    model.eval()  # model.train() 금지, eval()만 사용
    return model


# ──────────────────────────────────────────────────────────
# ROI mask 로드
# ──────────────────────────────────────────────────────────
def load_roi_mask(safe_id: str, volume_root: str) -> np.ndarray:
    """
    roi_0_0.npy 로드. 반환: (Z, H, W) bool ndarray.
    shape[0] < shape[1] or shape[2] -> (Z, H, W) 가정
    그렇지 않으면 (H, W, Z) 가정하고 transpose -> (Z, H, W)
    """
    vol_dir = Path(volume_root) / "volumes_npy" / safe_id
    roi_path = vol_dir / "roi_0_0.npy"

    check_forbidden_volume_file(str(roi_path))
    check_forbidden_path(str(roi_path))

    if not roi_path.exists():
        raise FileNotFoundError(f"[ERROR] roi_0_0.npy 없음: {roi_path}")

    roi = np.load(str(roi_path))

    # shape axis 자동 판단
    if roi.ndim != 3:
        raise ValueError(f"[ERROR] roi_0_0.npy shape이 3D가 아님: {roi.shape}")

    s0, s1, s2 = roi.shape
    if s0 < s1 or s0 < s2:
        # (Z, H, W) 가정
        roi_zwh = roi
    else:
        # (H, W, Z) 가정 -> transpose -> (Z, H, W)
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
    z_indices_used=None,
) -> np.ndarray:
    """
    roi_volume: (Z, H, W) bool
    z_indices_used: list of 3 z indices [z_lo, z_center, z_hi] (파싱 가능하면 우선)
    반환: roi_mask_6ch shape (6, 96, 96) bool
    """
    Z = roi_volume.shape[0]

    # z_indices_used 우선 사용
    if z_indices_used is not None and len(z_indices_used) >= 3:
        z_lo = int(z_indices_used[0])
        z_c = int(z_indices_used[1])
        z_hi = int(z_indices_used[2])
        # clamp
        z_lo = max(0, min(z_lo, Z - 1))
        z_c = max(0, min(z_c, Z - 1))
        z_hi = max(0, min(z_hi, Z - 1))
    else:
        z_lo = max(0, z_center - 1)
        z_c = z_center
        z_hi = min(Z - 1, z_center + 1)

    # fixed96 crop 범위로 슬라이싱
    roi_zm = roi_volume[z_lo, y0_f96:y1_f96, x0_f96:x1_f96]
    roi_zc = roi_volume[z_c, y0_f96:y1_f96, x0_f96:x1_f96]
    roi_zp = roi_volume[z_hi, y0_f96:y1_f96, x0_f96:x1_f96]

    # shape 보정: 96x96이 아닌 경우 pad/crop (edge case)
    def _ensure_96(arr):
        h, w = arr.shape
        if h == 96 and w == 96:
            return arr
        out = np.zeros((96, 96), dtype=bool)
        h_use = min(h, 96)
        w_use = min(w, 96)
        out[:h_use, :w_use] = arr[:h_use, :w_use]
        return out

    roi_zm = _ensure_96(roi_zm)
    roi_zc = _ensure_96(roi_zc)
    roi_zp = _ensure_96(roi_zp)

    # ch0,1,2 = lung (z-1, z, z+1), ch3,4,5 = mediastinal (z-1, z, z+1)
    roi_mask_6ch = np.stack(
        [roi_zm, roi_zc, roi_zp, roi_zm, roi_zc, roi_zp], axis=0
    ).astype(bool)  # (6, 96, 96)

    return roi_mask_6ch


# ──────────────────────────────────────────────────────────
# Patch mask 구성
# ──────────────────────────────────────────────────────────
def build_patch_mask(
    y0_patch: int,
    x0_patch: int,
    y1_patch: int,
    x1_patch: int,
    y0_fixed96: int,
    x0_fixed96: int,
    y1_fixed96: int,
    x1_fixed96: int,
) -> tuple:
    """
    patch bbox를 fixed96 local 좌표로 변환하여 2D mask 생성.
    반환: (patch_mask_2d (96,96) bool, patch_bbox_clipped bool)
    """
    # absolute -> local 좌표
    raw_ly0 = y0_patch - y0_fixed96
    raw_lx0 = x0_patch - x0_fixed96
    raw_ly1 = y1_patch - y0_fixed96
    raw_lx1 = x1_patch - x0_fixed96

    local_y0 = int(np.clip(raw_ly0, 0, 96))
    local_x0 = int(np.clip(raw_lx0, 0, 96))
    local_y1 = int(np.clip(raw_ly1, 0, 96))
    local_x1 = int(np.clip(raw_lx1, 0, 96))

    patch_bbox_clipped = (
        local_y0 != raw_ly0
        or local_x0 != raw_lx0
        or local_y1 != raw_ly1
        or local_x1 != raw_lx1
    )

    patch_mask_2d = np.zeros((96, 96), dtype=bool)
    if local_y1 > local_y0 and local_x1 > local_x0:
        patch_mask_2d[local_y0:local_y1, local_x0:local_x1] = True

    return patch_mask_2d, patch_bbox_clipped


# ──────────────────────────────────────────────────────────
# 점수 계산
# ──────────────────────────────────────────────────────────
def compute_scores(
    error: np.ndarray,
    roi_mask_6ch: np.ndarray,
    patch_mask_2d: np.ndarray,
    stored_l1_mean: float,
) -> dict:
    """
    error: (6, 96, 96) float32 = |recon - input|
    roi_mask_6ch: (6, 96, 96) bool
    patch_mask_2d: (96, 96) bool

    반환: 모든 score 컬럼 dict
    """
    patch_mask_6ch = np.stack([patch_mask_2d] * 6, axis=0)  # (6, 96, 96)
    roi_z_center = roi_mask_6ch[1]  # z_center 슬라이스의 ROI mask (ch1)

    # ── Whole-crop ──────────────────────────────────────────
    rd4ad_l1_whole_crop_recomputed = float(error.mean())
    rd4ad_l1_whole_crop_stored = float(stored_l1_mean)
    rd4ad_l1_whole_crop_match = (
        abs(rd4ad_l1_whole_crop_recomputed - rd4ad_l1_whole_crop_stored) < 1e-4
    )

    # ── ROI 관련 ────────────────────────────────────────────
    roi_flat = roi_mask_6ch.flatten()
    error_flat = error.flatten()

    if roi_flat.sum() > 0:
        rd4ad_l1_roi_mean = float(error_flat[roi_flat].mean())
    else:
        rd4ad_l1_roi_mean = float("nan")

    outside_roi_flat = ~roi_flat
    if outside_roi_flat.sum() > 0:
        rd4ad_l1_outside_roi_mean = float(error_flat[outside_roi_flat].mean())
    else:
        rd4ad_l1_outside_roi_mean = float("nan")

    # lung channel (ch0,1,2) ROI 내부
    lung_roi = roi_mask_6ch[0:3].flatten()
    lung_error = error[0:3].flatten()
    if lung_roi.sum() > 0:
        rd4ad_l1_lung_channel_roi_mean = float(lung_error[lung_roi].mean())
    else:
        rd4ad_l1_lung_channel_roi_mean = float("nan")

    # mediastinal channel (ch3,4,5) ROI 내부
    med_roi = roi_mask_6ch[3:6].flatten()
    med_error = error[3:6].flatten()
    if med_roi.sum() > 0:
        rd4ad_l1_med_channel_roi_mean = float(med_error[med_roi].mean())
    else:
        rd4ad_l1_med_channel_roi_mean = float("nan")

    # ── Patch 관련 ──────────────────────────────────────────
    patch_flat = patch_mask_6ch.flatten()
    if patch_flat.sum() > 0:
        rd4ad_l1_patch_mean = float(error_flat[patch_flat].mean())
    else:
        rd4ad_l1_patch_mean = float("nan")

    # ROI ∩ patch (z_center 기준 ROI + patch)
    roi_patch_mask = patch_mask_6ch & roi_mask_6ch
    roi_patch_flat = roi_patch_mask.flatten()
    if roi_patch_flat.sum() > 0:
        rd4ad_l1_roi_patch_mean = float(error_flat[roi_patch_flat].mean())
    else:
        rd4ad_l1_roi_patch_mean = float("nan")

    # patch bbox 안이지만 ROI 밖
    patch_outside_roi_mask = patch_mask_6ch & (~roi_mask_6ch)
    patch_outside_roi_flat = patch_outside_roi_mask.flatten()
    if patch_outside_roi_flat.sum() > 0:
        rd4ad_l1_patch_outside_roi_mean = float(
            error_flat[patch_outside_roi_flat].mean()
        )
    else:
        rd4ad_l1_patch_outside_roi_mean = float("nan")

    # ── 마스크/면적 ─────────────────────────────────────────
    roi_pixels_fixed96 = int(roi_z_center.sum())
    roi_ratio_fixed96 = roi_pixels_fixed96 / (96 * 96)
    outside_roi_pixels_fixed96 = (96 * 96) - roi_pixels_fixed96

    # patch_pixels: patch_mask_2d 내 픽셀 수 (클리핑 후)
    patch_pixels = int(patch_mask_2d.sum())

    # patch ∩ ROI (z_center 기준)
    patch_roi_mask_2d = patch_mask_2d & roi_z_center
    patch_roi_pixels = int(patch_roi_mask_2d.sum())
    patch_roi_ratio = patch_roi_pixels / patch_pixels if patch_pixels > 0 else 0.0

    return {
        "rd4ad_l1_whole_crop_recomputed": rd4ad_l1_whole_crop_recomputed,
        "rd4ad_l1_whole_crop_stored": rd4ad_l1_whole_crop_stored,
        "rd4ad_l1_whole_crop_match": rd4ad_l1_whole_crop_match,
        "rd4ad_l1_roi_mean": rd4ad_l1_roi_mean,
        "rd4ad_l1_outside_roi_mean": rd4ad_l1_outside_roi_mean,
        "rd4ad_l1_lung_channel_roi_mean": rd4ad_l1_lung_channel_roi_mean,
        "rd4ad_l1_med_channel_roi_mean": rd4ad_l1_med_channel_roi_mean,
        "rd4ad_l1_patch_mean": rd4ad_l1_patch_mean,
        "rd4ad_l1_roi_patch_mean": rd4ad_l1_roi_patch_mean,
        "rd4ad_l1_patch_outside_roi_mean": rd4ad_l1_patch_outside_roi_mean,
        "roi_pixels_fixed96": roi_pixels_fixed96,
        "roi_ratio_fixed96": roi_ratio_fixed96,
        "outside_roi_pixels_fixed96": outside_roi_pixels_fixed96,
        "patch_pixels": patch_pixels,
        "patch_roi_pixels": patch_roi_pixels,
        "patch_roi_ratio": patch_roi_ratio,
    }


# ──────────────────────────────────────────────────────────
# z_indices_used 파싱
# ──────────────────────────────────────────────────────────
def _parse_z_indices(z_indices_str) -> list:
    """z_indices_used 컬럼 파싱. 실패 시 None 반환."""
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
# crop 1개 처리
# ──────────────────────────────────────────────────────────
def process_one_crop(
    row_qa: pd.Series,
    row_overlap: pd.Series,
    row_manifest,  # None 가능
    model: ConvAutoencoder2p5D,
    device: torch.device,
    volume_root: str,
) -> dict:
    """
    단일 crop 처리.
    반환: 모든 결과 컬럼 dict (오류 시 error_reason 포함)
    """
    crop_id = str(row_qa["crop_id"])
    safe_id = str(row_qa["safe_id"])
    crop_path = str(row_qa["crop_path"])
    stored_l1_mean = float(row_qa["crop_score_l1_mean"])

    result_base = {
        "crop_id": crop_id,
        "patient_id": str(row_qa["patient_id"]),
        "safe_id": safe_id,
        "qa_group": str(row_qa["qa_group"]),
        "qa_priority": str(row_qa["qa_priority"]),
        "rd4ad_label": str(row_qa["rd4ad_label"]),
        "binary_label": str(row_qa["binary_label"]),
        "crop_path": crop_path,
        "rd4ad_l1_whole_crop_stored": stored_l1_mean,
        # lesion 관련 (overlap CSV에서)
        "lesion_overlap_class": str(row_overlap.get("lesion_overlap_class", "")),
        "lesion_pixels_patch": row_overlap.get("lesion_pixels_patch", None),
        "lesion_pixels_context192": row_overlap.get("lesion_pixels_context192", None),
        "has_lesion_overlap_patch": row_overlap.get("has_lesion_overlap_patch", None),
        "has_lesion_overlap_context192": row_overlap.get(
            "has_lesion_overlap_context192", None
        ),
        "error_reason": None,
        "patch_bbox_clipped": None,
    }

    # 1. crop npz 로드
    check_forbidden_path(crop_path)
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

    # 2. model forward
    inp = torch.from_numpy(image).unsqueeze(0).to(device)  # (1,6,96,96)
    try:
        with torch.no_grad():  # model.train() 금지, no_grad + eval()만
            recon = model(inp)
    except Exception as e:
        result_base["error_reason"] = f"model forward 실패: {e}"
        return result_base

    recon_np = recon.squeeze(0).cpu().numpy().astype(np.float32)  # (6,96,96)
    error = np.abs(recon_np - image)  # (6,96,96)

    # NaN/Inf 검출
    if np.any(np.isnan(error)) or np.any(np.isinf(error)):
        result_base["error_reason"] = "error map에 NaN/Inf 감지"

    # 3. ROI mask 로드
    z_center = int(row_overlap["z_center"])
    y0_f96 = int(row_overlap["y0_fixed96"])
    x0_f96 = int(row_overlap["x0_fixed96"])
    y1_f96 = int(row_overlap["y1_fixed96"])
    x1_f96 = int(row_overlap["x1_fixed96"])

    z_indices_used = None
    if row_manifest is not None and "z_indices_used" in row_manifest.index:
        z_indices_used = _parse_z_indices(row_manifest["z_indices_used"])

    try:
        roi_volume = load_roi_mask(safe_id, volume_root)
        roi_mask_6ch = extract_roi_crop(
            roi_volume, z_center, y0_f96, x0_f96, y1_f96, x1_f96, z_indices_used
        )
    except Exception as e:
        result_base["error_reason"] = (result_base["error_reason"] or "") + f" | ROI 로드 실패: {e}"
        roi_mask_6ch = np.zeros((6, 96, 96), dtype=bool)

    # 4. Patch mask 구성
    y0_patch = int(row_overlap["y0_patch"])
    x0_patch = int(row_overlap["x0_patch"])
    y1_patch = int(row_overlap["y1_patch"])
    x1_patch = int(row_overlap["x1_patch"])

    patch_mask_2d, patch_bbox_clipped = build_patch_mask(
        y0_patch, x0_patch, y1_patch, x1_patch,
        y0_f96, x0_f96, y1_f96, x1_f96,
    )
    result_base["patch_bbox_clipped"] = patch_bbox_clipped

    # 5. score 계산
    scores = compute_scores(error, roi_mask_6ch, patch_mask_2d, stored_l1_mean)
    result_base.update(scores)

    # whole_crop_stored는 이미 result_base에 있으므로 중복 제거
    result_base.pop("rd4ad_l1_whole_crop_stored", None)
    result_base.update(scores)  # scores 포함 (stored 값 덮어쓰기 방지 위해 다시 업데이트)

    return result_base


# ──────────────────────────────────────────────────────────
# 데이터 로드 헬퍼
# ──────────────────────────────────────────────────────────
def load_input_dataframes(project_root: Path):
    """
    QA manifest, overlap analysis, crop manifest 로드 및 merge 준비.
    반환: (df_qa, df_overlap, df_crop_manifest)
    """
    qa_p = project_root / QA_MANIFEST_PATH
    overlap_p = project_root / OVERLAP_ANALYSIS_PATH
    crop_manifest_p = project_root / CROP_MANIFEST_PATH

    for label, p in [
        ("QA manifest", qa_p),
        ("overlap analysis", overlap_p),
        ("crop manifest", crop_manifest_p),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"[ERROR] {label} 없음: {p}")

    df_qa = pd.read_csv(qa_p)
    df_overlap = pd.read_csv(overlap_p)
    df_crop = pd.read_csv(crop_manifest_p)

    return df_qa, df_overlap, df_crop


# ──────────────────────────────────────────────────────────
# preflight
# ──────────────────────────────────────────────────────────
def run_preflight(args, project_root: Path) -> None:
    """입력 파일 검증만. model forward 없음."""
    print("=" * 60)
    print("[PREFLIGHT] Phase 5.51 입력 검증 시작")
    print("=" * 60)

    errors = []
    warnings = []
    ok_items = []

    # 1. 입력 CSV 3개 존재
    qa_p = project_root / QA_MANIFEST_PATH
    overlap_p = project_root / OVERLAP_ANALYSIS_PATH
    crop_manifest_p = project_root / CROP_MANIFEST_PATH

    for label, p in [
        ("QA manifest", qa_p),
        ("overlap analysis", overlap_p),
        ("crop manifest", crop_manifest_p),
    ]:
        if p.exists():
            ok_items.append(f"[OK] {label} 존재: {p}")
        else:
            errors.append(f"[ERROR] {label} 없음: {p}")

    # 2. QA manifest row 수 == 150
    if qa_p.exists():
        df_qa = pd.read_csv(qa_p)
        n_rows = len(df_qa)
        if n_rows == QA_EXPECTED_ROWS:
            ok_items.append(f"[OK] QA manifest row 수: {n_rows}")
        else:
            errors.append(f"[ERROR] QA manifest row 수 불일치: {n_rows} (expected {QA_EXPECTED_ROWS})")

        # 3. crop_id 중복 없음
        if df_qa["crop_id"].duplicated().any():
            errors.append("[ERROR] QA manifest crop_id 중복 있음")
        else:
            ok_items.append("[OK] QA manifest crop_id 중복 없음")
    else:
        df_qa = None

    # 4. crop manifest join 가능
    if crop_manifest_p.exists() and df_qa is not None:
        df_crop = pd.read_csv(crop_manifest_p)
        # crop manifest의 crop_id 컬럼 확인 (candidate_id가 기준)
        # crop manifest에는 crop_path 컬럼 있음, crop_id 직접 없음 -> candidate_id 기반
        # QA manifest crop_id는 숫자, crop manifest는 candidate_id
        # join은 crop_path 기준으로 가능
        qa_paths = set(df_qa["crop_path"].tolist())
        crop_paths = set(df_crop["crop_path"].tolist()) if "crop_path" in df_crop.columns else set()
        overlap_count = len(qa_paths & crop_paths)
        if overlap_count > 0:
            ok_items.append(f"[OK] crop manifest crop_path join 가능: {overlap_count}개 매칭")
        else:
            warnings.append("[WARN] crop manifest와 crop_path 매칭 0개 (crop_id 직접 join 불가)")
    else:
        df_crop = None

    # 5. overlap analysis join 가능
    if overlap_p.exists() and df_qa is not None:
        df_overlap = pd.read_csv(overlap_p)
        qa_crop_ids = set(df_qa["crop_id"].astype(str).tolist())
        ol_crop_ids = set(df_overlap["crop_id"].astype(str).tolist())
        match_count = len(qa_crop_ids & ol_crop_ids)
        if match_count == QA_EXPECTED_ROWS:
            ok_items.append(f"[OK] overlap analysis crop_id join: {match_count}개 전체 매칭")
        else:
            errors.append(
                f"[ERROR] overlap analysis crop_id join: {match_count}/{QA_EXPECTED_ROWS} 매칭"
            )
    else:
        df_overlap = None

    # 6. crop_path 150개 파일 존재
    if df_qa is not None:
        missing_crops = []
        for _, row in df_qa.iterrows():
            if not Path(row["crop_path"]).exists():
                missing_crops.append(row["crop_path"])
        if not missing_crops:
            ok_items.append(f"[OK] crop npz {len(df_qa)}개 모두 존재")
        else:
            errors.append(f"[ERROR] crop npz {len(missing_crops)}개 없음")
            for cp in missing_crops[:5]:
                errors.append(f"       없음: {cp}")

    # 7. crop npz sample 3개 검증
    if df_qa is not None:
        sample_rows = df_qa.head(3)
        for _, srow in sample_rows.iterrows():
            cp = Path(srow["crop_path"])
            if cp.exists():
                try:
                    d = np.load(str(cp))
                    if "image" not in d:
                        errors.append(f"[ERROR] {cp.name}: 'image' key 없음")
                    elif d["image"].shape != (6, 96, 96):
                        errors.append(f"[ERROR] {cp.name}: shape={d['image'].shape}")
                    elif d["image"].dtype != np.float32:
                        errors.append(f"[ERROR] {cp.name}: dtype={d['image'].dtype}")
                    elif d["image"].min() < -0.01 or d["image"].max() > 1.01:
                        warnings.append(f"[WARN] {cp.name}: range [{d['image'].min():.4f}, {d['image'].max():.4f}]")
                    else:
                        ok_items.append(f"[OK] sample crop: {cp.name} shape/dtype/range OK")
                except Exception as e:
                    errors.append(f"[ERROR] {cp.name} 로드 실패: {e}")

    # 8. checkpoint 존재 및 torch.load 가능
    ckpt_p = project_root / CHECKPOINT_PATH
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
            errors.append(f"[ERROR] checkpoint torch.load 실패: {e}")
    else:
        errors.append(f"[ERROR] checkpoint 없음: {ckpt_p}")

    # 9. volume source root 존재
    vol_root = Path(VOLUME_SOURCE_ROOT)
    if vol_root.exists():
        ok_items.append(f"[OK] volume source root 존재: {vol_root}")
    else:
        errors.append(f"[ERROR] volume source root 없음: {vol_root}")

    # 10. sample patient folder 매칭 (3개 safe_id)
    if df_qa is not None and vol_root.exists():
        sample_safe_ids = df_qa["safe_id"].unique()[:3]
        for sid in sample_safe_ids:
            sid_dir = vol_root / "volumes_npy" / sid
            if sid_dir.exists():
                ok_items.append(f"[OK] patient folder 존재: {sid}")
            else:
                errors.append(f"[ERROR] patient folder 없음: {sid_dir}")

    # 11. sample roi_0_0.npy 존재 (3개)
    if df_qa is not None and vol_root.exists():
        sample_safe_ids = df_qa["safe_id"].unique()[:3]
        for sid in sample_safe_ids:
            roi_p = vol_root / "volumes_npy" / sid / "roi_0_0.npy"
            if roi_p.exists():
                ok_items.append(f"[OK] roi_0_0.npy 존재: {sid}")
            else:
                errors.append(f"[ERROR] roi_0_0.npy 없음: {roi_p}")

    # 12-14. overlap analysis 컬럼 확인
    if df_overlap is not None:
        required_overlap_cols = [
            "y0_fixed96", "x0_fixed96", "y1_fixed96", "x1_fixed96",
            "y0_patch", "x0_patch", "y1_patch", "x1_patch",
            "z_center",
        ]
        missing_cols = [c for c in required_overlap_cols if c not in df_overlap.columns]
        if missing_cols:
            errors.append(f"[ERROR] overlap analysis 누락 컬럼: {missing_cols}")
        else:
            ok_items.append("[OK] overlap analysis 필수 컬럼 모두 존재")

    # 15. output root 충돌 여부
    out_root = project_root / OUTPUT_ROOT
    if out_root.exists():
        if not args.force:
            warnings.append(
                f"[WARN] output root 이미 존재: {out_root}\n"
                "       full-run 시 --force 없으면 RuntimeError 발생"
            )
        else:
            warnings.append(f"[WARN] --force 지정됨. output root 덮어쓰기 허용: {out_root}")
    else:
        ok_items.append(f"[OK] output root 없음 (신규 생성 예정): {out_root}")

    # 16. stage2_holdout/v2 금지 경로 차단 확인
    forbidden_check_paths = [
        QA_MANIFEST_PATH, OVERLAP_ANALYSIS_PATH, CROP_MANIFEST_PATH,
        CHECKPOINT_PATH, OUTPUT_ROOT,
    ]
    any_forbidden = False
    for fp in forbidden_check_paths:
        for kw in FORBIDDEN_PATH_KEYWORDS:
            if kw in str(fp).lower():
                errors.append(f"[ERROR] 금지 키워드 '{kw}' 경로 감지: {fp}")
                any_forbidden = True
    if not any_forbidden:
        ok_items.append("[OK] 금지 경로 키워드 없음 (stage2_holdout/holdout/v2)")

    # 17. 결과 출력
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
        print(f"\n[PREFLIGHT] 결과: {len(errors)}개 오류, {len(warnings)}개 경고 -> 수정 후 재실행 필요")
        sys.exit(1)
    else:
        print(f"\n[PREFLIGHT] 결과: OK (오류 0, 경고 {len(warnings)})")
        print("[PREFLIGHT] dry-run 또는 full-run 진행 가능.")


# ──────────────────────────────────────────────────────────
# dry-run
# ──────────────────────────────────────────────────────────
def run_dry_run(args, project_root: Path) -> None:
    """소량 crop model forward + 콘솔 출력. CSV/JSON/MD 저장 금지."""
    print("=" * 60)
    print("[DRY-RUN] Phase 5.51 ROI/patch-masked score diagnostic")
    print("=" * 60)
    print("[DRY-RUN] CSV/JSON/MD 저장 금지. 기존 파일 미수정.")

    device = _resolve_device(args.device)
    df_qa, df_overlap, df_crop = load_input_dataframes(project_root)

    # target-crop-ids 또는 limit
    if args.target_crop_ids:
        target_ids = set(str(x) for x in args.target_crop_ids.split(","))
        df_qa_sub = df_qa[df_qa["crop_id"].astype(str).isin(target_ids)].reset_index(drop=True)
        if len(df_qa_sub) == 0:
            print(f"[DRY-RUN][WARN] target_crop_ids={args.target_crop_ids}에 해당하는 row 없음")
            return
    else:
        limit = args.limit if args.limit is not None else 3
        df_qa_sub = df_qa.head(limit).reset_index(drop=True)

    print(f"[DRY-RUN] 처리 예정: {len(df_qa_sub)}개 (full-run 기준 {len(df_qa)}개)")

    # 모델 로드
    ckpt_path = project_root / CHECKPOINT_PATH
    model = load_model(str(ckpt_path), device)
    print(f"[DRY-RUN] 모델 로드 완료: {ckpt_path}")

    # overlap, crop manifest index
    df_overlap_idx = df_overlap.set_index(df_overlap["crop_id"].astype(str))
    df_crop_path_idx = df_crop.set_index("crop_path") if "crop_path" in df_crop.columns else None

    vol_root = VOLUME_SOURCE_ROOT

    for i, row_qa in df_qa_sub.iterrows():
        crop_id_str = str(row_qa["crop_id"])
        print(f"\n[DRY-RUN] crop {i+1}/{len(df_qa_sub)}: crop_id={crop_id_str}")

        if crop_id_str not in df_overlap_idx.index:
            print(f"  [WARN] overlap analysis에 crop_id={crop_id_str} 없음. 스킵.")
            continue
        row_overlap = df_overlap_idx.loc[crop_id_str]
        if isinstance(row_overlap, pd.DataFrame):
            row_overlap = row_overlap.iloc[0]

        row_manifest = None
        if df_crop_path_idx is not None:
            cp = row_qa["crop_path"]
            if cp in df_crop_path_idx.index:
                row_manifest = df_crop_path_idx.loc[cp]
                if isinstance(row_manifest, pd.DataFrame):
                    row_manifest = row_manifest.iloc[0]

        result = process_one_crop(
            row_qa, row_overlap, row_manifest, model, device, vol_root
        )

        # 콘솔 pretty-print
        _pretty_print_result(result)

        # NaN/Inf 경고
        for key, val in result.items():
            if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                print(f"  [WARN] NaN/Inf 감지: {key}={val}")

    print("\n[DRY-RUN] 완료. CSV/JSON/MD 미저장. 기존 파일 미수정.")


def _pretty_print_result(result: dict) -> None:
    """결과 dict 콘솔 출력."""
    print(f"  crop_id: {result.get('crop_id')}")
    score_keys = [
        "rd4ad_l1_whole_crop_stored",
        "rd4ad_l1_whole_crop_recomputed",
        "rd4ad_l1_whole_crop_match",
        "rd4ad_l1_roi_mean",
        "rd4ad_l1_outside_roi_mean",
        "rd4ad_l1_lung_channel_roi_mean",
        "rd4ad_l1_med_channel_roi_mean",
        "rd4ad_l1_patch_mean",
        "rd4ad_l1_roi_patch_mean",
        "rd4ad_l1_patch_outside_roi_mean",
        "roi_pixels_fixed96",
        "roi_ratio_fixed96",
        "patch_pixels",
        "patch_roi_pixels",
        "patch_roi_ratio",
        "patch_bbox_clipped",
        "lesion_overlap_class",
        "error_reason",
    ]
    for k in score_keys:
        v = result.get(k, "N/A")
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")


# ──────────────────────────────────────────────────────────
# full-run
# ──────────────────────────────────────────────────────────
def run_full_run(args, project_root: Path) -> None:
    """150개 전체 처리. output root 신규 생성."""
    print("=" * 60)
    print("[FULL-RUN] Phase 5.51 ROI/patch-masked score diagnostic")
    print("=" * 60)

    out_root = project_root / OUTPUT_ROOT
    check_forbidden_path(str(out_root))

    if out_root.exists() and not args.force:
        raise RuntimeError(
            f"[GUARD] output root 이미 존재: {out_root}\n"
            "--force 없이 full-run 금지."
        )
    if out_root.exists() and args.force:
        print(f"[WARN] --force 지정됨. 기존 output root 덮어쓰기: {out_root}")

    out_root.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    df_qa, df_overlap, df_crop = load_input_dataframes(project_root)

    limit = args.limit if args.limit is not None else len(df_qa)
    df_qa_sub = df_qa.head(limit).reset_index(drop=True)
    print(f"[FULL-RUN] 처리 대상: {len(df_qa_sub)}개")

    # 모델 로드
    ckpt_path = project_root / CHECKPOINT_PATH
    model = load_model(str(ckpt_path), device)
    print(f"[FULL-RUN] 모델 로드 완료: epoch={_get_ckpt_epoch(project_root / CHECKPOINT_PATH)}")

    df_overlap_idx = df_overlap.set_index(df_overlap["crop_id"].astype(str))
    df_crop_path_idx = df_crop.set_index("crop_path") if "crop_path" in df_crop.columns else None
    vol_root = VOLUME_SOURCE_ROOT

    all_results = []
    n_success = 0
    n_error = 0

    for i, row_qa in df_qa_sub.iterrows():
        crop_id_str = str(row_qa["crop_id"])
        if (i + 1) % 10 == 0 or i == 0:
            print(f"[FULL-RUN] {i+1}/{len(df_qa_sub)} crop_id={crop_id_str}")

        if crop_id_str not in df_overlap_idx.index:
            r = {
                "crop_id": crop_id_str,
                "patient_id": str(row_qa.get("patient_id", "")),
                "safe_id": str(row_qa.get("safe_id", "")),
                "error_reason": f"overlap analysis에 crop_id={crop_id_str} 없음",
            }
            all_results.append(r)
            n_error += 1
            continue

        row_overlap = df_overlap_idx.loc[crop_id_str]
        if isinstance(row_overlap, pd.DataFrame):
            row_overlap = row_overlap.iloc[0]

        row_manifest = None
        if df_crop_path_idx is not None:
            cp = row_qa["crop_path"]
            if cp in df_crop_path_idx.index:
                row_manifest = df_crop_path_idx.loc[cp]
                if isinstance(row_manifest, pd.DataFrame):
                    row_manifest = row_manifest.iloc[0]

        try:
            result = process_one_crop(
                row_qa, row_overlap, row_manifest, model, device, vol_root
            )
        except Exception as e:
            result = {
                "crop_id": crop_id_str,
                "patient_id": str(row_qa.get("patient_id", "")),
                "safe_id": str(row_qa.get("safe_id", "")),
                "error_reason": f"process_one_crop 예외: {e}",
            }

        if result.get("error_reason"):
            n_error += 1
        else:
            n_success += 1

        all_results.append(result)

    print(f"\n[FULL-RUN] 처리 완료: n_success={n_success}, n_error={n_error}")

    # 결과 저장
    save_results(all_results, out_root, args, n_success, n_error, df_qa_sub)
    print(f"[FULL-RUN] 결과 저장 완료: {out_root}")


def _get_ckpt_epoch(ckpt_path: Path) -> int:
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        return ckpt.get("epoch", -1)
    except Exception:
        return -1


# ──────────────────────────────────────────────────────────
# 결과 저장
# ──────────────────────────────────────────────────────────
def save_results(
    rows: list,
    output_root: Path,
    args,
    n_success: int,
    n_error: int,
    df_qa_sub: pd.DataFrame,
) -> None:
    """CSV, JSON, MD 저장."""
    run_ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # CSV
    df_result = pd.DataFrame(rows)
    csv_path = output_root / OUTPUT_CSV_NAME
    df_result.to_csv(str(csv_path), index=False)
    print(f"[SAVE] CSV: {csv_path} ({len(df_result)} rows)")

    # summary JSON
    summary = generate_summary_json(rows, args, n_success, n_error, run_ts)
    json_path = output_root / OUTPUT_JSON_NAME
    with open(str(json_path), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] JSON: {json_path}")

    # MD report
    md_text = generate_md_report(rows, summary, args, run_ts)
    md_path = output_root / OUTPUT_MD_NAME
    with open(str(md_path), "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"[SAVE] MD: {md_path}")


# ──────────────────────────────────────────────────────────
# 요약 JSON 생성
# ──────────────────────────────────────────────────────────
def _stat_dict(values: list) -> dict:
    """기초 통계 dict."""
    arr = np.array([v for v in values if v is not None and not np.isnan(v) and not np.isinf(v)], dtype=float)
    if len(arr) == 0:
        return {"mean": None, "std": None, "min": None, "max": None,
                "p25": None, "p50": None, "p75": None, "p90": None,
                "p95": None, "p99": None}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def generate_summary_json(
    rows: list,
    args,
    n_success: int,
    n_error: int,
    run_ts: str,
) -> dict:
    df = pd.DataFrame(rows)
    n_rows = len(df)
    n_patients = df["patient_id"].nunique() if "patient_id" in df.columns else 0

    def _col_vals(col):
        if col not in df.columns:
            return []
        return [v for v in df[col].tolist() if v is not None]

    def _bool_count(col):
        if col not in df.columns:
            return None
        return int(df[col].sum()) if col in df.columns else None

    whole_crop_match_count = _bool_count("rd4ad_l1_whole_crop_match")

    # qa_group별 평균
    qa_group_avg = {}
    if "qa_group" in df.columns:
        for grp, sub in df.groupby("qa_group"):
            qa_group_avg[str(grp)] = {
                "n": len(sub),
                "rd4ad_l1_whole_crop_stored_mean": float(sub["rd4ad_l1_whole_crop_stored"].mean())
                if "rd4ad_l1_whole_crop_stored" in sub.columns else None,
                "rd4ad_l1_roi_mean_mean": float(sub["rd4ad_l1_roi_mean"].mean())
                if "rd4ad_l1_roi_mean" in sub.columns else None,
                "rd4ad_l1_roi_patch_mean_mean": float(sub["rd4ad_l1_roi_patch_mean"].mean())
                if "rd4ad_l1_roi_patch_mean" in sub.columns else None,
            }

    # lesion_overlap_class별 평균
    lesion_class_avg = {}
    if "lesion_overlap_class" in df.columns:
        for cls, sub in df.groupby("lesion_overlap_class"):
            lesion_class_avg[str(cls)] = {
                "n": len(sub),
                "rd4ad_l1_whole_crop_stored_mean": float(sub["rd4ad_l1_whole_crop_stored"].mean())
                if "rd4ad_l1_whole_crop_stored" in sub.columns else None,
                "rd4ad_l1_roi_patch_mean_mean": float(sub["rd4ad_l1_roi_patch_mean"].mean())
                if "rd4ad_l1_roi_patch_mean" in sub.columns else None,
            }

    # top10 whole_crop vs roi_patch
    top10_whole = []
    top10_roi_patch = []
    if "rd4ad_l1_whole_crop_stored" in df.columns:
        top10_whole = df.nlargest(10, "rd4ad_l1_whole_crop_stored")["crop_id"].astype(str).tolist()
    if "rd4ad_l1_roi_patch_mean" in df.columns:
        df_valid = df[df["rd4ad_l1_roi_patch_mean"].notna()]
        top10_roi_patch = df_valid.nlargest(10, "rd4ad_l1_roi_patch_mean")["crop_id"].astype(str).tolist()

    # outside_roi > roi_mean 케이스
    outside_roi_higher = []
    if "rd4ad_l1_outside_roi_mean" in df.columns and "rd4ad_l1_roi_mean" in df.columns:
        mask = (
            df["rd4ad_l1_outside_roi_mean"].notna()
            & df["rd4ad_l1_roi_mean"].notna()
            & (df["rd4ad_l1_outside_roi_mean"] > df["rd4ad_l1_roi_mean"])
        )
        outside_roi_higher = df[mask]["crop_id"].astype(str).tolist()

    summary = {
        "script_run_mode": "full-run",
        "output_tag": args.output_tag,
        "run_timestamp": run_ts,
        "n_rows": n_rows,
        "n_patients": n_patients,
        "n_success": n_success,
        "n_error": n_error,
        "whole_crop_score_match_count": whole_crop_match_count,
        "roi_ratio_fixed96": _stat_dict(_col_vals("roi_ratio_fixed96")),
        "patch_roi_ratio": _stat_dict(_col_vals("patch_roi_ratio")),
        "rd4ad_l1_whole_crop_stored": _stat_dict(_col_vals("rd4ad_l1_whole_crop_stored")),
        "rd4ad_l1_roi_mean": _stat_dict(_col_vals("rd4ad_l1_roi_mean")),
        "rd4ad_l1_patch_mean": _stat_dict(_col_vals("rd4ad_l1_patch_mean")),
        "rd4ad_l1_roi_patch_mean": _stat_dict(_col_vals("rd4ad_l1_roi_patch_mean")),
        "qa_group_avg": qa_group_avg,
        "lesion_overlap_class_avg": lesion_class_avg,
        "top10_by_whole_crop_score": top10_whole,
        "top10_by_roi_patch_score": top10_roi_patch,
        "outside_roi_mean_higher_than_roi_mean_crop_ids": outside_roi_higher,
        "limitation": (
            "이 결과는 diagnostic 전용임. "
            "모델 재학습 없음. threshold 재확정 아님. 병변 성능 결론 아님. "
            "ROI mask는 roi_0_0.npy 기반. 이 스크립트는 재현 가능한 분석 도구임."
        ),
    }
    return summary


# ──────────────────────────────────────────────────────────
# MD report 생성
# ──────────────────────────────────────────────────────────
def generate_md_report(rows: list, summary: dict, args, run_ts: str) -> str:
    df = pd.DataFrame(rows)
    lines = []
    lines.append("# Phase 5.51 RD4AD ROI/Patch-Masked Score Diagnostic Report")
    lines.append("")
    lines.append(f"- run_timestamp: {run_ts}")
    lines.append(f"- script_run_mode: {summary.get('script_run_mode', 'full-run')}")
    lines.append(f"- output_tag: {summary.get('output_tag', '')}")
    lines.append("")
    lines.append("## 요약")
    lines.append(f"- n_rows: {summary['n_rows']}")
    lines.append(f"- n_patients: {summary['n_patients']}")
    lines.append(f"- n_success: {summary['n_success']}")
    lines.append(f"- n_error: {summary['n_error']}")
    lines.append(f"- whole_crop_score_match_count: {summary['whole_crop_score_match_count']}")
    lines.append("")
    lines.append("## ROI/Patch 비율 통계")
    for key in ["roi_ratio_fixed96", "patch_roi_ratio"]:
        st = summary.get(key, {})
        if st:
            lines.append(f"### {key}")
            lines.append(f"- mean: {st.get('mean')}, std: {st.get('std')}, min: {st.get('min')}, max: {st.get('max')}")
    lines.append("")
    lines.append("## Score 통계")
    for key in ["rd4ad_l1_whole_crop_stored", "rd4ad_l1_roi_mean",
                "rd4ad_l1_patch_mean", "rd4ad_l1_roi_patch_mean"]:
        st = summary.get(key, {})
        if st:
            lines.append(f"### {key}")
            lines.append(
                f"- mean: {st.get('mean'):.6f}, std: {st.get('std'):.6f}, "
                f"p50: {st.get('p50'):.6f}, p90: {st.get('p90'):.6f}, "
                f"p95: {st.get('p95'):.6f}, p99: {st.get('p99'):.6f}"
                if st.get("mean") is not None else "- (데이터 없음)"
            )
    lines.append("")
    lines.append("## QA Group별 평균")
    for grp, vals in summary.get("qa_group_avg", {}).items():
        lines.append(f"- {grp}: n={vals['n']}, whole_crop_stored_mean={vals.get('rd4ad_l1_whole_crop_stored_mean')}, roi_patch_mean={vals.get('rd4ad_l1_roi_patch_mean_mean')}")
    lines.append("")
    lines.append("## Lesion Overlap Class별 평균")
    for cls, vals in summary.get("lesion_overlap_class_avg", {}).items():
        lines.append(f"- {cls}: n={vals['n']}, whole_crop_mean={vals.get('rd4ad_l1_whole_crop_stored_mean')}, roi_patch_mean={vals.get('rd4ad_l1_roi_patch_mean_mean')}")
    lines.append("")
    lines.append("## Top-10 비교")
    lines.append("### whole_crop 기준 top10 crop_id")
    lines.append(", ".join(summary.get("top10_by_whole_crop_score", [])))
    lines.append("### roi_patch 기준 top10 crop_id")
    lines.append(", ".join(summary.get("top10_by_roi_patch_score", [])))
    lines.append("")
    lines.append("## ROI 밖 error > ROI 안 error 케이스")
    outside_ids = summary.get("outside_roi_mean_higher_than_roi_mean_crop_ids", [])
    lines.append(f"총 {len(outside_ids)}개")
    if outside_ids:
        lines.append(", ".join(outside_ids[:20]))
    lines.append("")
    lines.append("## Limitation")
    lines.append(summary.get("limitation", ""))
    lines.append("")
    lines.append("---")
    lines.append("*이 보고서는 자동 생성된 diagnostic 문서임. 임상 결론에 사용 금지.*")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────
# device 해석
# ──────────────────────────────────────────────────────────
def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda_if_available":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("[ERROR] --device cuda 지정했지만 CUDA 없음.")
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# ──────────────────────────────────────────────────────────
# argparse
# ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 5.51 - RD4AD ROI/patch-masked score diagnostic"
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--preflight-only",
        action="store_true",
        help="입력 검증만. model forward 없음.",
    )
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="소량 crop model forward + 콘솔 출력. CSV/JSON/MD 저장 금지.",
    )
    mode_group.add_argument(
        "--full-run",
        action="store_true",
        help="150개 전체 처리 + 결과 저장.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="처리 최대 crop 수 (dry-run 기본 3, full-run 기본 150).",
    )
    parser.add_argument(
        "--target-crop-ids",
        type=str,
        default=None,
        help="쉼표 구분 crop_id 목록 (예: '6454,6600'). dry-run 전용.",
    )
    parser.add_argument(
        "--device",
        choices=["cuda_if_available", "cuda", "cpu"],
        default="cuda_if_available",
        help="실행 device (기본: cuda_if_available).",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default="roi_patch_masked_v1",
        help="출력 태그 (기본: roi_patch_masked_v1).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="output root 덮어쓰기 허용 (기본 False).",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # 프로젝트 root 결정 (이 스크립트 위치 기준)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent  # scripts/ -> project root

    print(f"[INFO] project_root: {project_root}")
    print(f"[INFO] output_tag: {args.output_tag}")
    print(f"[INFO] run_timestamp: {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")

    # guard: output root 경로
    check_forbidden_path(str(project_root / OUTPUT_ROOT))

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
