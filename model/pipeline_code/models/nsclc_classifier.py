"""
nsclc_classifier.py

Normal-like vs NSCLC-lesion-like 보조 분류기 (P-C-NORMAL24j).
EfficientNet-B0 (1280-dim) + scalar branch (lung_z_percentile, crop_lung_roi_ratio)

학습 조건:
  - 체크포인트: p_c_normal24j_fix_balanced_w1_best_val_auc
  - Patient AUROC=0.9898, sensitivity=0.9918, specificity=0.4167
  - Shortcut risk (SR-HU, SR-CONTEXT) 미제거 → 독립 evidence 전용
  - 진단 아님, NSCLC 확률 아님

입력:
  - 3-channel MIP HU crop (raw HU float32, NOT windowed)
  - lung_z_percentile: z / max(Z-1, 1)
  - crop_lung_roi_ratio: pure_lung[y0:y1, x0:x1].mean()
"""
import os
import numpy as np
import torch
import torch.nn as nn

WEIGHTS_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
WEIGHT_PATH  = os.path.join(WEIGHTS_DIR, "nsclc_classifier.pt")

_CROP_SIZE   = 96
HU_MIN       = -1000.0
HU_MAX       =  200.0
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── 모델 아키텍처 ──────────────────────────────────────────────────────────────

class _ScalarFusionModel(nn.Module):
    """EfficientNet-B0 (1280-dim) + 2-scalar branch → binary head"""

    def __init__(self):
        super().__init__()
        from torchvision.models import efficientnet_b0
        backbone = efficientnet_b0(weights=None)
        self.img_features  = backbone.features
        self.img_avgpool   = backbone.avgpool
        self.scalar_branch = nn.Sequential(
            nn.Linear(2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
        )
        self.fusion_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280 + 16, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1),
        )

    def forward(self, img, scalar):
        x = self.img_features(img)
        x = self.img_avgpool(x)
        x = torch.flatten(x, 1)
        s = self.scalar_branch(scalar)
        return self.fusion_head(torch.cat([x, s], dim=1))


# ── 모델 로드 (서버 시작 시 1회) ───────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_ckpt = torch.load(WEIGHT_PATH, map_location=device, weights_only=False)
_scalar_norm_stats = _ckpt["scalar_norm_stats"]   # 체크포인트에 내장

_nsclc_model = _ScalarFusionModel()
_nsclc_model.load_state_dict(_ckpt["model_state_dict"])
_nsclc_model.to(device)
_nsclc_model.eval()

# ── 전처리 헬퍼 ───────────────────────────────────────────────────────────────

def _normalize_scalar(lung_z_pct: float, crop_roi_ratio: float) -> np.ndarray:
    """scalar stats z-score 정규화"""
    z_mean  = _scalar_norm_stats["lung_z_percentile"]["mean"]
    z_std   = _scalar_norm_stats["lung_z_percentile"]["std"]
    r_mean  = _scalar_norm_stats["crop_lung_roi_ratio"]["mean"]
    r_std   = _scalar_norm_stats["crop_lung_roi_ratio"]["std"]
    return np.array([
        (lung_z_pct    - z_mean) / z_std,
        (crop_roi_ratio - r_mean) / r_std,
    ], dtype=np.float32)


def build_nsclc_hu_crop(hu_volume: np.ndarray, z: int, y0: int, x0: int) -> np.ndarray:
    """(Z,H,W) HU → (3,96,96) raw HU float32. 윈도우 없음.
    ch0=MIP(z-3,z-2,z-1), ch1=MIP(z-1,z,z+1), ch2=MIP(z+1,z+2,z+3) — 학습 조건과 동일.
    NSCLC 분류기 / Grad-CAM 공용 (HU[-1000,200] 클립은 추론 시 적용).
    """
    Z, H, W = hu_volume.shape
    z  = int(z)
    y1 = y0 + _CROP_SIZE
    x1 = x0 + _CROP_SIZE
    pad_top    = max(0, -y0);    pad_bottom = max(0, y1 - H)
    pad_left   = max(0, -x0);   pad_right  = max(0, x1 - W)
    needs_pad  = pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0
    cy0 = max(0, y0);  cy1 = min(H, y1)
    cx0 = max(0, x0);  cx1 = min(W, x1)
    pad_mode   = "reflect" if (cy1 - cy0 > 1) and (cx1 - cx0 > 1) else "edge"

    def _sl(zi):
        zi = int(np.clip(zi, 0, Z - 1))
        s  = hu_volume[zi, cy0:cy1, cx0:cx1].astype(np.float32)
        if needs_pad:
            s = np.pad(s, ((pad_top, pad_bottom), (pad_left, pad_right)), mode=pad_mode)
        return s

    def _mip(z_list):
        return np.stack([_sl(zi) for zi in z_list], axis=0).max(axis=0)

    ch0 = _mip([z - 3, z - 2, z - 1])
    ch1 = _mip([z - 1, z,     z + 1])
    ch2 = _mip([z + 1, z + 2, z + 3])
    return np.stack([ch0, ch1, ch2], axis=0)


def _preprocess_hu_crop(hu_crop: np.ndarray) -> np.ndarray:
    """(3,96,96) raw HU → ImageNet 정규화된 float32 tensor 입력용"""
    arr = np.clip(hu_crop, HU_MIN, HU_MAX)
    arr = (arr - HU_MIN) / (HU_MAX - HU_MIN)          # [0, 1]
    arr = (arr - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return arr.astype(np.float32)


# ── 단일 패치 추론 ─────────────────────────────────────────────────────────────

def run_nsclc(
    hu_volume: np.ndarray,
    z: int,
    y0: int,
    x0: int,
    lung_z_pct: float,
    crop_lung_roi_ratio: float,
) -> dict:
    """단일 패치 NSCLC 보조 점수 계산.

    반환:
      {"prob": float, "logit": float, "label": "NSCLC-like" | "Normal-like"}
    """
    hu_crop = build_nsclc_hu_crop(hu_volume, z, y0, x0)
    img     = _preprocess_hu_crop(hu_crop)
    scalar  = _normalize_scalar(lung_z_pct, crop_lung_roi_ratio)

    img_t    = torch.tensor(img,    dtype=torch.float32).unsqueeze(0).to(device)
    scalar_t = torch.tensor(scalar, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = _nsclc_model(img_t, scalar_t).squeeze().item()
        prob  = float(torch.sigmoid(torch.tensor(logit)).item())

    return {
        "prob":  round(prob,  4),
        "logit": round(logit, 4),
        "label": "NSCLC-like" if prob >= 0.5 else "Normal-like",
    }


def run_nsclc_batch(
    hu_volume: np.ndarray,
    candidates: list,           # list of {"z", "y0", "x0", "lung_z_pct", "crop_lung_roi_ratio"}
    batch_size: int = 32,
) -> list:
    """배치 NSCLC 추론. candidates 순서대로 결과 반환.

    각 결과: {"prob": float, "logit": float, "label": str}
    """
    if not candidates:
        return []

    crops   = []
    scalars = []
    for c in candidates:
        hu_crop = build_nsclc_hu_crop(hu_volume, c["z"], c["y0"], c["x0"])
        crops.append(_preprocess_hu_crop(hu_crop))
        scalars.append(_normalize_scalar(c["lung_z_pct"], c["crop_lung_roi_ratio"]))

    results = []
    for i in range(0, len(crops), batch_size):
        b_crops   = np.stack(crops[i:i + batch_size], axis=0)
        b_scalars = np.stack(scalars[i:i + batch_size], axis=0)
        img_t    = torch.tensor(b_crops,   dtype=torch.float32).to(device)
        scalar_t = torch.tensor(b_scalars, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = _nsclc_model(img_t, scalar_t).squeeze(1)
            probs  = torch.sigmoid(logits)
        for j in range(len(b_crops)):
            p = float(probs[j].item())
            l = float(logits[j].item())
            if i == 0 and j == 0:
                print(f"[NSCLC DEBUG] logit={l:.4f} prob={p:.4f} scalar={b_scalars[j]}")
            results.append({
                "prob":  round(p, 4),
                "logit": round(l, 4),
                "label": "NSCLC-like" if p >= 0.5 else "Normal-like",
            })

    return results
