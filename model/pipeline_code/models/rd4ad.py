"""
RD4AD E2 (EfficientNet-B0 Teacher-Student, lung3ch)
  - Teacher: EfficientNet-B0 (frozen, ImageNet pretrained)
  - Student Decoder: de_late(80->80) -> de_mid(upsample + 80->40) -> de_early(upsample + 40->24)
  - 입력: (3, 96, 96) lung window HU [-1000, 600], lung3ch: ch0=z-1, ch1=z, ch2=z+1
  - 이상 점수: teacher-student cosine distance 합산 평균 (late/mid/early)
  - 가중치: best_train_loss.pth (student decoder, E2)
  - 모델 구조 변경 금지
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
WEIGHT_PATH = os.path.join(WEIGHTS_DIR, "best_train_loss.pth")

_EFFB0_FILENAME      = "efficientnet_b0_rwightman-7f5810bc.pth"
_EFFB0_PROJECT_PATH  = Path(WEIGHTS_DIR) / _EFFB0_FILENAME
_EFFB0_CACHE_PATH    = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / _EFFB0_FILENAME


def _get_effb0_path():
    if _EFFB0_PROJECT_PATH.exists():
        return _EFFB0_PROJECT_PATH
    if _EFFB0_CACHE_PATH.exists():
        return _EFFB0_CACHE_PATH
    return None


# =============================================================================
# Teacher: EfficientNet-B0 (frozen, ImageNet)
# =============================================================================

def _build_teacher():
    import torchvision.models as models
    weight_path = _get_effb0_path()
    if weight_path is None:
        raise RuntimeError(
            f"EfficientNet-B0 weight 파일이 없습니다.\n"
            f"다음 위치 중 하나에 파일을 복사하세요:\n"
            f"  (1) {_EFFB0_PROJECT_PATH}\n"
            f"  (2) {_EFFB0_CACHE_PATH}\n"
            f"파일명: {_EFFB0_FILENAME}"
        )
    effnet = models.efficientnet_b0(weights=None)
    effnet.load_state_dict(
        torch.load(str(weight_path), map_location="cpu", weights_only=True)
    )
    effnet.eval()
    effnet.requires_grad_(False)
    return effnet


# =============================================================================
# Student Decoder E2
# de_late(80->80) -> de_mid(upsample x2 + 80->40) -> de_early(upsample x2 + 40->24)
# =============================================================================

class _StudentDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.de_late = nn.Sequential(
            nn.Conv2d(80, 80, 3, 1, 1),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True),
        )
        self.de_mid = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(80, 40, 3, 1, 1),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True),
        )
        self.de_early = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(40, 24, 3, 1, 1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )

    def forward(self, late_feat):
        x    = self.de_late(late_feat);  de_l = x
        x    = self.de_mid(x);           de_m = x
        x    = self.de_early(x);         de_e = x
        return de_l, de_m, de_e


# =============================================================================
# 모델 로드 (서버 시작 시 1회)
# =============================================================================

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_teacher = _build_teacher().to(device)
_student = _StudentDecoder().to(device)

# teacher feature hook: features[2]=early(24ch), features[3]=mid(40ch), features[4]=late(80ch)
_teacher_feats = {}

def _make_hook(name):
    def _h(module, inp, output):
        _teacher_feats[name] = output
    return _h

_teacher.features[2].register_forward_hook(_make_hook("early"))
_teacher.features[3].register_forward_hook(_make_hook("mid"))
_teacher.features[4].register_forward_hook(_make_hook("late"))

# student 가중치 로드
ckpt  = torch.load(WEIGHT_PATH, map_location=device, weights_only=False)
state = ckpt.get("student_state_dict", ckpt.get("model_state_dict", ckpt))
_student.load_state_dict(state)
_student.eval()

# =============================================================================
# lung3ch crop builder (E2 학습 조건과 동일)
# =============================================================================

HU_MIN     = -1000.0
HU_MAX     =   600.0
_CROP_SIZE = 96


def build_lung3ch_crop(hu_volume: np.ndarray, z: int, y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
    """(Z,H,W) HU -> (3,96,96) float32 [0,1].
    lung window [-1000,600], lung3ch: ch0=z-1, ch1=z, ch2=z+1.
    """
    Z, H, W = hu_volume.shape
    z  = int(z);  y0 = int(y0);  x0 = int(x0);  y1 = int(y1);  x1 = int(x1)
    zm = max(z - 1, 0);  zp = min(z + 1, Z - 1)
    pad_top = max(0, -y0);  pad_bot = max(0, y1 - H)
    pad_lft = max(0, -x0);  pad_rgt = max(0, x1 - W)
    cy0, cy1 = max(0, y0), min(H, y1)
    cx0, cx1 = max(0, x0), min(W, x1)
    needs_pad = pad_top > 0 or pad_bot > 0 or pad_lft > 0 or pad_rgt > 0

    def _sl(zi):
        s = np.asarray(hu_volume[zi, cy0:cy1, cx0:cx1], dtype="float32")
        s = (s.clip(HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)
        if needs_pad:
            mode = "reflect" if (cy1 - cy0 > 1 and cx1 - cx0 > 1) else "edge"
            s = np.pad(s, ((pad_top, pad_bot), (pad_lft, pad_rgt)), mode=mode)
        return s

    crop = np.stack([_sl(zm), _sl(z), _sl(zp)], axis=0).astype("float32")
    if crop.shape != (3, _CROP_SIZE, _CROP_SIZE):
        raise RuntimeError(f"[ABORT] crop shape {crop.shape} != (3,{_CROP_SIZE},{_CROP_SIZE})")
    if not np.isfinite(crop).all():
        raise RuntimeError(f"[ABORT] crop contains NaN/Inf  y0={y0} x0={x0}")
    return crop


# =============================================================================
# 배치 추론 (main.py에서 candidate_patches 받아서 실행)
# =============================================================================

def run_rd4ad(padim_result: dict) -> dict:
    patches = padim_result.get("candidate_patches", [])
    if not patches:
        return {"score": 0.0, "patches": []}

    BATCH_SIZE = 64
    scored = []
    imgs   = [np.array(p["image"], dtype=np.float32) for p in patches]

    for batch_start in range(0, len(imgs), BATCH_SIZE):
        batch_imgs    = imgs[batch_start:batch_start + BATCH_SIZE]
        batch_patches = patches[batch_start:batch_start + BATCH_SIZE]
        batch_tensor  = torch.tensor(
            np.stack(batch_imgs, axis=0), dtype=torch.float32
        ).to(device)

        with torch.no_grad():
            _teacher(batch_tensor)
            tf_late  = _teacher_feats["late"]
            tf_mid   = _teacher_feats["mid"]
            tf_early = _teacher_feats["early"]
            de_l, de_m, de_e = _student(tf_late)

            s_l = (1 - F.cosine_similarity(de_l,  tf_late,  dim=1)).mean(dim=(1, 2))
            s_m = (1 - F.cosine_similarity(de_m,  tf_mid,   dim=1)).mean(dim=(1, 2))
            s_e = (1 - F.cosine_similarity(de_e,  tf_early, dim=1)).mean(dim=(1, 2))
            batch_scores = ((s_l + s_m + s_e) / 3).cpu().numpy()

        for j in range(len(batch_imgs)):
            scored.append({
                "score":    round(float(batch_scores[j]), 6),
                "position": batch_patches[j]["position"],
            })

    if not scored:
        return {"score": 0.0, "patches": []}

    max_score = max(p["score"] for p in scored)
    return {
        "score":   round(max_score, 4),
        "patches": sorted(scored, key=lambda x: x["score"], reverse=True),
    }


# =============================================================================
# E2 spatial map → base64 PNG (RD4AD 히트맵, cam_prob < 0.5 fallback)
# 원본: build_dynamic_ref_card_any_candidate_rd4ad_v1_panel4_clinical_compact.py
#       run_rd4ad_e2_spatial_map() + 렌더링 else 분기
# =============================================================================

def run_rd4ad_e2_spatial_map_base64(hu_volume: np.ndarray, z: int, y0: int, x0: int,
                                    ts_guard_vol: "np.ndarray | None" = None) -> "str | None":
    """96x96 crop → E2 cosine distance spatial map → YlOrRd+CT blend RGBA base64 PNG. 실패 시 None.
    ts_guard_vol: TotalSegmentator 폐 마스크 (bool Z×H×W). 있으면 폐 내부만 표시.
    """
    try:
        import base64
        from io import BytesIO
        import matplotlib
        from PIL import Image

        crop = build_lung3ch_crop(hu_volume, z, y0, x0, y0 + _CROP_SIZE, x0 + _CROP_SIZE)
        inp  = torch.tensor(crop, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            _teacher(inp)
            tf_late  = _teacher_feats["late"]
            tf_mid   = _teacher_feats["mid"]
            tf_early = _teacher_feats["early"]
            de_l, de_m, de_e = _student(tf_late)

            s_l = F.interpolate(
                (1 - F.cosine_similarity(de_l, tf_late,  dim=1, eps=1e-8)).unsqueeze(1),
                size=(_CROP_SIZE, _CROP_SIZE), mode="bilinear", align_corners=False)
            s_m = F.interpolate(
                (1 - F.cosine_similarity(de_m, tf_mid,   dim=1, eps=1e-8)).unsqueeze(1),
                size=(_CROP_SIZE, _CROP_SIZE), mode="bilinear", align_corners=False)
            s_e = F.interpolate(
                (1 - F.cosine_similarity(de_e, tf_early, dim=1, eps=1e-8)).unsqueeze(1),
                size=(_CROP_SIZE, _CROP_SIZE), mode="bilinear", align_corners=False)
            smap = ((s_l + s_m + s_e) / 3)[0, 0].cpu().numpy().astype("float32")

        # 폐 마스크 적용 (있으면 TotalSeg, 없으면 HU > -950 fallback)
        if ts_guard_vol is not None:
            Z_, H_, W_ = hu_volume.shape
            cy0 = max(0, y0); cy1 = min(H_, y0 + _CROP_SIZE)
            cx0 = max(0, x0); cx1 = min(W_, x0 + _CROP_SIZE)
            z_c = max(0, min(z, Z_ - 1))
            mask_96 = ts_guard_vol[z_c, cy0:cy1, cx0:cx1].astype(np.float32)
            if mask_96.shape != (_CROP_SIZE, _CROP_SIZE):
                ph = _CROP_SIZE - mask_96.shape[0]; pw = _CROP_SIZE - mask_96.shape[1]
                mask_96 = np.pad(mask_96, ((0, ph), (0, pw)), mode="edge")
        else:
            mask_96 = (crop[1] > ((-950.0 - HU_MIN) / (HU_MAX - HU_MIN))).astype(np.float32)

        # 마스크 내부만 min-max normalize
        vals = smap[mask_96 > 0.5]
        if len(vals) > 0 and (vals.max() - vals.min()) > 1e-8:
            smap = np.where(mask_96 > 0.5, (smap - vals.min()) / (vals.max() - vals.min()), 0.0).astype("float32")
        else:
            smap = np.zeros_like(smap)

        # YlOrRd 컬러맵 + alpha (원본 GRADCAM_ALPHA_GAMMA=0.55, GRADCAM_ALPHA_MAX=0.88)
        _ylrd = matplotlib.colormaps["YlOrRd"]
        rgba_f = _ylrd(smap).astype("float32")
        rgba_f[..., 3] = (smap ** 0.55) * 0.88

        # CT 슬라이스 배경 블렌딩
        Z_, H_, W_ = hu_volume.shape
        cy0 = max(0, y0); cy1 = min(H_, y0 + _CROP_SIZE)
        cx0 = max(0, x0); cx1 = min(W_, x0 + _CROP_SIZE)
        bg = np.asarray(hu_volume[max(0, min(z, Z_ - 1)), cy0:cy1, cx0:cx1], dtype="float32")
        bg = np.clip((bg - HU_MIN) / (HU_MAX - HU_MIN), 0, 1) * 255
        if bg.shape != (_CROP_SIZE, _CROP_SIZE):
            ph = _CROP_SIZE - bg.shape[0]; pw = _CROP_SIZE - bg.shape[1]
            bg = np.pad(bg, ((0, ph), (0, pw)), mode="edge")
        bg_rgb = np.stack([bg, bg, bg], axis=-1).astype("float32")
        a3  = rgba_f[..., 3:4]
        out = np.clip(bg_rgb * (1 - a3) + rgba_f[..., :3] * 255 * a3, 0, 255).astype("uint8")

        buf = BytesIO()
        Image.fromarray(out, "RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"[RD4AD] spatial map 실패: {e}")
        return None
