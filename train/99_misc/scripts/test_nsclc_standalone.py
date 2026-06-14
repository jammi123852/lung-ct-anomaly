"""
NSCLC 분류기 단독 테스트 스크립트
- DICOM 시리즈 로드 → HU 변환 → NSCLC 모델 추론
- 이미지 피처 기여 / scalar 기여 / 합산 logit 각각 출력
- 진단: 어느 쪽이 prob≈0 원인인지 확인
사용:
  source ~/ai_env/bin/activate
  python scripts/test_nsclc_standalone.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

# ─────────────────────────────────────────────
# 경로 설정: 실제 경로에 맞게 수정
# ─────────────────────────────────────────────
DICOM_DIR   = "/mnt/c/Users/jinhy/Downloads/test_dicom_nsclc_lung1_007"
WEIGHT_PATH = "/mnt/c/Users/jinhy/Downloads/model/lunar_backend/weights/nsclc_classifier.pt"

# 테스트할 z 슬라이스 목록 (백엔드 로그에서 이상 검출된 슬라이스)
TEST_ZS = [128, 129, 130, 131, 132, 133, 134, 135]

# z=133에서 테스트할 패치 중심 좌표 (백엔드 로그 기준, 없으면 여러 위치 격자 탐색)
TEST_PATCHES = [
    # (z, center_y, center_x, note)
    (133, 200, 200, "center-ish"),
    (133, 150, 300, "upper-right"),
    (133, 300, 150, "lower-left"),
    (133, 256, 256, "image center"),
    (133, 180, 350, "upper-right2"),
    (133, 350, 200, "lower-center"),
]

# ─────────────────────────────────────────────
# 모델 상수
# ─────────────────────────────────────────────
CROP_SIZE    = 96
HU_MIN       = -1000.0
HU_MAX       =  200.0
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# 모델 아키텍처 (nsclc_classifier.py와 동일)
# ─────────────────────────────────────────────
class ScalarFusionModel(nn.Module):
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

    def forward_debug(self, img, scalar):
        """이미지 피처 단독 / scalar=0 / 합산 logit 반환"""
        x = self.img_features(img)
        x = self.img_avgpool(x)
        x = torch.flatten(x, 1)
        s = self.scalar_branch(scalar)

        # 합산 logit
        logit_full   = self.fusion_head(torch.cat([x, s],             dim=1))
        # 이미지만 (scalar 기여 0)
        s_zero = torch.zeros_like(s)
        logit_imgonly = self.fusion_head(torch.cat([x, s_zero],       dim=1))
        # scalar만 (이미지 기여 0)
        x_zero = torch.zeros_like(x)
        logit_scalaronly = self.fusion_head(torch.cat([x_zero, s],    dim=1))

        return {
            "logit_full":       logit_full.squeeze().item(),
            "logit_img_only":   logit_imgonly.squeeze().item(),
            "logit_scalar_only":logit_scalaronly.squeeze().item(),
            "img_feat_norm":    float(x.norm(dim=1).mean().item()),
            "scalar_feat_norm": float(s.norm(dim=1).mean().item()),
        }


# ─────────────────────────────────────────────
# DICOM 로드 → HU 볼륨
# ─────────────────────────────────────────────
def load_dicom_volume(dcm_dir):
    try:
        import pydicom
    except ImportError:
        print("[ERROR] pydicom 없음: pip install pydicom")
        sys.exit(1)

    files = sorted(
        [os.path.join(dcm_dir, f) for f in os.listdir(dcm_dir) if f.endswith(".dcm")],
        key=lambda p: os.path.basename(p)
    )
    print(f"[INFO] DICOM 파일 수: {len(files)}")
    if not files:
        print("[ERROR] .dcm 파일 없음")
        sys.exit(1)

    # 첫 슬라이스로 shape 확인
    ds0 = pydicom.dcmread(files[0])
    H, W = int(ds0.Rows), int(ds0.Columns)
    Z = len(files)
    print(f"[INFO] 볼륨 shape: ({Z}, {H}, {W})")

    hu_vol = np.zeros((Z, H, W), dtype=np.float32)
    for i, f in enumerate(files):
        ds = pydicom.dcmread(f)
        slope  = float(getattr(ds, "RescaleSlope",     1.0))
        interc = float(getattr(ds, "RescaleIntercept", 0.0))
        px = ds.pixel_array.astype(np.float32)
        hu_vol[i] = px * slope + interc

    print(f"[INFO] HU 범위: [{hu_vol.min():.1f}, {hu_vol.max():.1f}]")
    return hu_vol


# ─────────────────────────────────────────────
# HU crop → 3-channel MIP
# ─────────────────────────────────────────────
def build_crop(hu_vol, z, cy, cx):
    """(cy, cx) 중심으로 96×96 3-channel MIP crop 반환"""
    Z, H, W = hu_vol.shape
    y0 = cy - CROP_SIZE // 2
    x0 = cx - CROP_SIZE // 2
    y1 = y0 + CROP_SIZE
    x1 = x0 + CROP_SIZE

    pad_top    = max(0, -y0)
    pad_bottom = max(0, y1 - H)
    pad_left   = max(0, -x0)
    pad_right  = max(0, x1 - W)
    cy0 = max(0, y0); cy1 = min(H, y1)
    cx0 = max(0, x0); cx1 = min(W, x1)

    def _sl(zi):
        zi = int(np.clip(zi, 0, Z - 1))
        s = hu_vol[zi, cy0:cy1, cx0:cx1].astype(np.float32)
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            s = np.pad(s, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")
        return s

    ch0 = np.stack([_sl(z-3), _sl(z-2), _sl(z-1)], 0).max(0)
    ch1 = np.stack([_sl(z-1), _sl(z  ), _sl(z+1)], 0).max(0)
    ch2 = np.stack([_sl(z+1), _sl(z+2), _sl(z+3)], 0).max(0)
    return np.stack([ch0, ch1, ch2], 0)  # (3, 96, 96)


def preprocess(hu_crop):
    arr = np.clip(hu_crop, HU_MIN, HU_MAX)
    arr = (arr - HU_MIN) / (HU_MAX - HU_MIN)
    arr = (arr - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return arr.astype(np.float32)


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────
def main():
    print(f"[INFO] device: {device}")
    print(f"[INFO] 가중치 로드: {WEIGHT_PATH}")

    ckpt = torch.load(WEIGHT_PATH, map_location=device, weights_only=False)
    stats = ckpt["scalar_norm_stats"]
    print(f"[INFO] scalar_norm_stats:")
    print(f"       lung_z_percentile  mean={stats['lung_z_percentile']['mean']:.4f}  std={stats['lung_z_percentile']['std']:.4f}")
    print(f"       crop_lung_roi_ratio mean={stats['crop_lung_roi_ratio']['mean']:.4f}  std={stats['crop_lung_roi_ratio']['std']:.4f}")

    model = ScalarFusionModel()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print("[INFO] 모델 로드 완료\n")

    hu_vol = load_dicom_volume(DICOM_DIR)
    Z = hu_vol.shape[0]

    z_mean = stats["lung_z_percentile"]["mean"]
    z_std  = stats["lung_z_percentile"]["std"]
    r_mean = stats["crop_lung_roi_ratio"]["mean"]
    r_std  = stats["crop_lung_roi_ratio"]["std"]

    print("=" * 70)
    print("1. 다중 위치 격자 탐색 (z=133)")
    print("=" * 70)
    for (z, cy, cx, note) in TEST_PATCHES:
        lz_pct = z / max(Z - 1, 1)
        # crop 내 roi: HU > -950 비율 (중심 MIP)
        y0c = max(0, cy - 48); y1c = min(hu_vol.shape[1], cy + 48)
        x0c = max(0, cx - 48); x1c = min(hu_vol.shape[2], cx + 48)
        z0 = max(0, z-1); z1 = z; z2 = min(Z-1, z+1)
        mip_c = np.maximum(np.maximum(hu_vol[z0], hu_vol[z1]), hu_vol[z2])
        roi = float((mip_c[y0c:y1c, x0c:x1c] > -950).mean())

        s0 = (lz_pct - z_mean) / z_std
        s1 = (roi   - r_mean) / r_std

        hu_crop = build_crop(hu_vol, z, cy, cx)
        img_arr = preprocess(hu_crop)
        img_t  = torch.tensor(img_arr, dtype=torch.float32).unsqueeze(0).to(device)
        sc_t   = torch.tensor([[s0, s1]], dtype=torch.float32).to(device)

        with torch.no_grad():
            dbg = model.forward_debug(img_t, sc_t)

        prob_full  = float(torch.sigmoid(torch.tensor(dbg["logit_full"])).item())
        prob_img   = float(torch.sigmoid(torch.tensor(dbg["logit_img_only"])).item())
        prob_scl   = float(torch.sigmoid(torch.tensor(dbg["logit_scalar_only"])).item())

        print(f"z={z} cy={cy:3d} cx={cx:3d} [{note}]")
        print(f"  lz_pct={lz_pct:.3f} roi={roi:.3f}  scalar=[{s0:.3f}, {s1:.3f}]")
        print(f"  logit_full={dbg['logit_full']:7.3f}  prob={prob_full:.4f}")
        print(f"  logit_img_only={dbg['logit_img_only']:7.3f}  prob={prob_img:.4f}  ← 이미지 단독")
        print(f"  logit_scalar_only={dbg['logit_scalar_only']:7.3f}  prob={prob_scl:.4f}  ← scalar 단독")
        print(f"  img_feat_norm={dbg['img_feat_norm']:.2f}  scalar_feat_norm={dbg['scalar_feat_norm']:.2f}")
        print()

    print("=" * 70)
    print("2. 전체 슬라이스 격자 탐색 (256x256 center 고정)")
    print("=" * 70)
    for z in TEST_ZS:
        lz_pct = z / max(Z - 1, 1)
        cy, cx = 256, 256
        y0c = max(0, cy - 48); y1c = min(hu_vol.shape[1], cy + 48)
        x0c = max(0, cx - 48); x1c = min(hu_vol.shape[2], cx + 48)
        z0 = max(0, z-1); z1 = z; z2 = min(Z-1, z+1)
        mip_c = np.maximum(np.maximum(hu_vol[z0], hu_vol[z1]), hu_vol[z2])
        roi = float((mip_c[y0c:y1c, x0c:x1c] > -950).mean())
        s0 = (lz_pct - z_mean) / z_std
        s1 = (roi   - r_mean) / r_std

        hu_crop = build_crop(hu_vol, z, cy, cx)
        img_arr = preprocess(hu_crop)
        img_t  = torch.tensor(img_arr, dtype=torch.float32).unsqueeze(0).to(device)
        sc_t   = torch.tensor([[s0, s1]], dtype=torch.float32).to(device)

        with torch.no_grad():
            dbg = model.forward_debug(img_t, sc_t)
        prob_full = float(torch.sigmoid(torch.tensor(dbg["logit_full"])).item())

        print(f"z={z:3d}  lz_pct={lz_pct:.3f}  roi={roi:.3f}  "
              f"logit={dbg['logit_full']:7.3f}  prob={prob_full:.4f}  "
              f"[img={dbg['logit_img_only']:6.3f} scl={dbg['logit_scalar_only']:6.3f}]")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
