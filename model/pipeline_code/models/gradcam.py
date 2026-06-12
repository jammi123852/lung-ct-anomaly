"""
gradcam.py — P-C-NORMAL30b Grad-CAM 실시간 추론

img_features[7] 기반 Grad-CAM, mask-inside-only YlOrRd colormap.
오프라인 기준: outputs/end/gradcam_masked_input_inference_v1/gradcam_run.py (37g)

체크포인트: weights/gradcam_model.pt (P-C-NORMAL30b, epoch=6, val_auc=0.9999)
- mask_applied_to_image_only=True → 추론 시 lung mask 적용
- scalar: lung_z_percentile, crop_lung_roi_ratio
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.cm as mpl_cm
from io import BytesIO
import base64
from PIL import Image

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
GRADCAM_WEIGHT_PATH = os.path.join(WEIGHTS_DIR, "gradcam_model.pt")

HU_MIN  = -1000.0
HU_MAX  =  200.0
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CROP_SIZE     = 96
ALPHA_GAMMA   = 0.55
ALPHA_MAX     = 0.88

_YLRD = mpl_cm.get_cmap("YlOrRd")


# ── 모델 (P-C-NORMAL30b, 동일 아키텍처) ───────────────────────────────────────
class _GradCamModel(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import efficientnet_b0
        backbone = efficientnet_b0(weights=None)
        self.img_features  = backbone.features
        self.img_avgpool   = backbone.avgpool
        self.scalar_branch = nn.Sequential(
            nn.Linear(2, 32), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Linear(32, 16), nn.ReLU(inplace=True),
        )
        self.fusion_head = nn.Sequential(
            nn.Dropout(p=0.2), nn.Linear(1280 + 16, 64),
            nn.ReLU(inplace=True), nn.Dropout(p=0.2), nn.Linear(64, 1),
        )

    def forward(self, img, scalar):
        x = self.img_features(img)
        x = self.img_avgpool(x)
        x = torch.flatten(x, 1)
        s = self.scalar_branch(scalar)
        return self.fusion_head(torch.cat([x, s], dim=1))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_gradcam_model = None
_gradcam_scalar_stats = None  # P-C-NORMAL30b는 24j stats와 동일한 분포 사용

def _load_gradcam_model():
    global _gradcam_model, _gradcam_scalar_stats
    if _gradcam_model is not None:
        return

    # 24j scalar stats 재사용 (동일 scalar 피처, 유사 분포)
    from models.nsclc_classifier import _ckpt as _nsclc_ckpt
    _gradcam_scalar_stats = _nsclc_ckpt["scalar_norm_stats"]

    ckpt = torch.load(GRADCAM_WEIGHT_PATH, map_location=device, weights_only=False)
    _gradcam_model = _GradCamModel()
    _gradcam_model.load_state_dict(ckpt["model_state_dict"])
    _gradcam_model.to(device)
    _gradcam_model.eval()
    epoch = ckpt.get("epoch", "?")
    val_auc = ckpt.get("val_auc", "?")
    print(f"[GRADCAM] P-C-NORMAL30b loaded: epoch={epoch}, val_auc={val_auc}")


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────
def _normalize_scalar(lung_z_pct: float, crop_roi_ratio: float) -> np.ndarray:
    stats = _gradcam_scalar_stats
    z_mean = stats["lung_z_percentile"]["mean"]
    z_std  = stats["lung_z_percentile"]["std"]
    r_mean = stats["crop_lung_roi_ratio"]["mean"]
    r_std  = stats["crop_lung_roi_ratio"]["std"]
    return np.array([
        (lung_z_pct      - z_mean) / z_std,
        (crop_roi_ratio  - r_mean) / r_std,
    ], dtype=np.float32)


def _mask_inside_norm(cam_raw, mask_96):
    masked = cam_raw * mask_96
    vals = masked[mask_96 > 0.5]
    if len(vals) == 0 or (vals.max() - vals.min()) < 1e-8:
        return np.zeros_like(masked)
    vmin, vmax = float(vals.min()), float(vals.max())
    return np.where(mask_96 > 0.5, (masked - vmin) / (vmax - vmin), 0.0).astype(np.float32)


def _compute_gradcam_raw(img_t, sc_t):
    """img_features[7] Grad-CAM. no_grad 없이 실행."""
    act_store = {}; grad_store = {}

    def fwd(m, i, o): act_store["v"] = o
    def bwd(m, gi, go): grad_store["v"] = go[0]

    tgt = _gradcam_model.img_features[7]
    fh  = tgt.register_forward_hook(fwd)
    bh  = tgt.register_full_backward_hook(bwd)

    _gradcam_model.zero_grad()
    logit = _gradcam_model(img_t, sc_t)
    logit.backward()

    fh.remove(); bh.remove()

    act  = act_store["v"]; grad = grad_store["v"]
    w    = grad.mean(dim=(2, 3), keepdim=True)
    cam  = torch.relu((act * w).sum(dim=1, keepdim=True))
    cam_up = F.interpolate(cam, size=(CROP_SIZE, CROP_SIZE), mode="bilinear", align_corners=False)
    cam_np = cam_up[0, 0].detach().cpu().numpy().astype(np.float32)

    cmin, cmax = float(cam_np.min()), float(cam_np.max())
    cam_np = (cam_np - cmin) / (cmax - cmin) if cmax - cmin > 1e-8 else np.zeros_like(cam_np)
    return cam_np, float(logit.item()), float(torch.sigmoid(logit).item())


# ── 외부 진입점 ────────────────────────────────────────────────────────────────
def compute_gradcam_base64(
    hu_volume: np.ndarray,
    z: int,
    y0: int,
    x0: int,
    lung_z_pct: float,
    crop_lung_roi_ratio: float,
    ts_guard_vol: np.ndarray | None = None,
) -> str | None:
    """
    (Z,H,W) HU volume → Grad-CAM RGBA PNG base64 (96×96 YlOrRd)
    ts_guard_vol: TotalSegmentator 폐 마스크 (bool Z×H×W). 없으면 HU > -950 fallback.
    실패 시 None 반환.
    """
    try:
        _load_gradcam_model()

        from models.nsclc_classifier import build_nsclc_hu_crop
        hu_crop = build_nsclc_hu_crop(hu_volume, z, y0, x0)  # (3,96,96) raw HU

        # lung mask: TotalSegmentator 캐시 우선, 없으면 HU > -950 fallback
        if ts_guard_vol is not None:
            Z_, H_, W_ = hu_volume.shape
            cy0 = max(0, y0); cy1 = min(H_, y0 + 96)
            cx0 = max(0, x0); cx1 = min(W_, x0 + 96)
            z_c = max(0, min(z, Z_ - 1))
            m = ts_guard_vol[z_c, cy0:cy1, cx0:cx1].astype(np.float32)
            if m.shape != (96, 96):
                ph = 96 - m.shape[0]; pw = 96 - m.shape[1]
                m = np.pad(m, ((0, ph), (0, pw)), mode="edge")
            mask_96 = m
        else:
            mask_96 = (hu_crop[1] > -950).astype(np.float32)

        # masked input (P-C-NORMAL30b는 mask_applied_to_image_only=True)
        arr = (np.clip(hu_crop, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)
        masked_arr = arr * mask_96[None, :, :]
        img_arr = (masked_arr - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]

        img_t = torch.tensor(img_arr, dtype=torch.float32).unsqueeze(0).to(device)
        sc_arr = _normalize_scalar(lung_z_pct, crop_lung_roi_ratio)
        sc_t   = torch.tensor(sc_arr, dtype=torch.float32).unsqueeze(0).to(device)

        cam_raw, logit, prob = _compute_gradcam_raw(img_t, sc_t)
        _gradcam_model.zero_grad()
        print(f"[GRADCAM] logit={logit:.4f} prob={prob:.4f}")

        cam_norm = _mask_inside_norm(cam_raw, mask_96)

        # RGBA PNG: YlOrRd + alpha = cam^0.55 * 0.88
        rgba = _YLRD(cam_norm).astype(np.float32)
        rgba[..., 3] = np.power(cam_norm, ALPHA_GAMMA) * ALPHA_MAX
        rgba_uint8 = (rgba * 255).clip(0, 255).astype(np.uint8)
        img_pil = Image.fromarray(rgba_uint8, "RGBA")
        buf = BytesIO()
        img_pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[GRADCAM] 오류: {e}")
        return None
