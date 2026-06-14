"""
End-to-End Pipeline Test: PaDiM → z-track → RD4AD E2 → NSCLC → XAI
대상: LUNG1-007 (235 slices, NSCLC 환자)
평가: z=80~109 병변 GT 기준 각 단계별 hit rate

사용:
  source ~/ai_env/bin/activate
  cd /home/jinhy/project/lung-ct-anomaly
  python scripts/pipeline_e2e_test.py

출력: scripts/pipeline_output/ 폴더에 XAI 이미지 + 성능 요약
"""

import sys, os, csv, json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

DICOM_DIR  = "/mnt/c/Users/jinhy/Downloads/test_dicom_nsclc_lung1_007"
LESION_CSV = "/home/jinhy/project/lung-ct-anomaly/outputs/p_c_normal24k_fix_final_test_prediction_export/p_c_normal24k_balanced_w1_final_test_predictions.csv"
OUTPUT_DIR = "/home/jinhy/project/lung-ct-anomaly/scripts/pipeline_output"
BACKEND    = "/mnt/c/Users/jinhy/Downloads/model/lunar_backend/models"

sys.path.insert(0, "/home/jinhy/project/lung-ct-anomaly/src")
sys.path.insert(0, BACKEND)

os.makedirs(OUTPUT_DIR, exist_ok=True)

PADIM_P90   = 12.20        # Stage 1 평가 기준 (기존 유지)
P95_THRESHOLD = 14.092    # 원본 V2V2_P95_THRESHOLD (GS2 G0 기준)
SLICE_TOP_N   = 30        # 원본 slice_top30 기준
Z_TRACK_MIN = 2
RD4AD_CROP  = 96

GRADCAM_PTH   = "/home/jinhy/project/lung-ct-anomaly/outputs/p_c_normal30b_masked_input_full_train/checkpoints/p_c_normal30b_best_val_auc_checkpoint.pt"
GRADCAM_STATS = "/home/jinhy/project/lung-ct-anomaly/outputs/reports/p_c_normal24h_fix_zroi_scalar_fusion_drycheck/p_c_normal24h_fix_scalar_normalization_stats.json"

# Grad-CAM 렌더링 상수 (gradcam_run.py 동일)
ALPHA_GAMMA = 0.55
ALPHA_MAX   = 0.88


# ═══════════════════════════════════════════════════════════════════════════════
# 0. DICOM 로드
# ═══════════════════════════════════════════════════════════════════════════════
def load_dicom_volume(dcm_dir):
    import pydicom
    files = sorted(f for f in os.listdir(dcm_dir) if f.endswith(".dcm"))
    print(f"[DICOM] {len(files)} slices")
    ds0 = pydicom.dcmread(os.path.join(dcm_dir, files[0]))
    H, W = int(ds0.Rows), int(ds0.Columns)
    vol = np.zeros((len(files), H, W), dtype=np.float32)
    for i, fname in enumerate(files):
        ds = pydicom.dcmread(os.path.join(dcm_dir, fname))
        slope  = float(getattr(ds, "RescaleSlope",     1.0))
        interc = float(getattr(ds, "RescaleIntercept", 0.0))
        vol[i] = ds.pixel_array.astype(np.float32) * slope + interc
    print(f"[DICOM] shape={vol.shape}, HU=[{vol.min():.0f}, {vol.max():.0f}]")
    return vol


# ═══════════════════════════════════════════════════════════════════════════════
# 1. STAGE 1: PaDiM (백엔드 모듈 내부 함수 활용)
# ═══════════════════════════════════════════════════════════════════════════════
def load_padim_backend():
    import padim as _padim
    print(f"[PaDiM] 백엔드 모듈 로드 완료 (threshold P90={PADIM_P90})")
    return _padim


def run_padim_on_slice(hu_slice, padim_mod):
    from scipy.ndimage import binary_dilation
    from position_aware_padim.preprocessing import preprocess_ct_slice

    H, W = hu_slice.shape
    patch_size, stride = 32, 16

    hu_lung = padim_mod._hu_lung_mask(hu_slice)
    med = np.zeros((H, W), dtype=bool)
    med[int(H*0.2):int(H*0.8), int(W*0.35):int(W*0.65)] = True
    org = binary_dilation(med & (hu_slice > -100), iterations=3)
    pure_lung = hu_lung & ~org

    if pure_lung.sum() < 100:
        return []

    preprocessed = preprocess_ct_slice(hu_slice)

    patch_coords = []
    for y0 in range(0, H - patch_size + 1, stride):
        for x0 in range(0, W - patch_size + 1, stride):
            cy = (y0 * 2 + patch_size) // 2
            cx = (x0 * 2 + patch_size) // 2
            if not pure_lung[cy, cx]:
                continue
            if pure_lung[y0:y0+patch_size, x0:x0+patch_size].sum() < patch_size*patch_size*0.5:
                continue
            patch_coords.append((y0, x0, y0+patch_size, x0+patch_size))

    if not patch_coords:
        return []

    features_144 = padim_mod._feature_extractor.extract_patch_features(preprocessed, patch_coords)
    features_100 = features_144[:, padim_mod._padim_model.selected_feature_indices]

    hits = []
    for i, (y0, x0, y1, x1) in enumerate(patch_coords):
        cy = (y0 + y1) / 2; cx = (x0 + x1) / 2
        pb = padim_mod.assign_position_bin(cy, cx, H, W)
        try:
            score = padim_mod._padim_model.score_patch(features_100[i], pb)
        except Exception:
            continue
        if score > PADIM_P90:
            hits.append({"y0": int(y0), "x0": int(x0), "y1": int(y1), "x1": int(x1),
                         "padim_score": float(score), "position_bin": pb})
    return hits


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Z-TRACK 필터
# ═══════════════════════════════════════════════════════════════════════════════
def apply_ztrack(hits_by_z, total_z):
    track = {}
    result = {}
    for z in range(total_z):
        if z not in hits_by_z:
            continue
        for p in hits_by_z[z]:
            key = (p["y0"], p["x0"])
            st  = track.get(key, {"last_z": -99, "run_len": 0})
            run = (st["run_len"] + 1) if z - st["last_z"] == 1 else 1
            track[key] = {"last_z": z, "run_len": run}

        for p in hits_by_z[z]:
            key = (p["y0"], p["x0"])
            run = track[key]["run_len"]
            if run >= Z_TRACK_MIN:
                result.setdefault(z, []).append({**p, "run_len": run})

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 2-B. GS2 FILTER (원본 phase8_2e 선별 로직)
# ═══════════════════════════════════════════════════════════════════════════════
def apply_gs2_filter(hits_by_z, hu_vol):
    """원본 GS2 pool: G0 (padim_score >= P95) OR slice_top30 by score_v950.
    score_v950 = padim_score × roi_ratio (HU>-950 비율).
    """
    Z, H, W = hu_vol.shape
    out = {}
    for z, patches in hits_by_z.items():
        z0 = max(0, z - 1); z2 = min(Z - 1, z + 1)
        mip = np.maximum(np.maximum(hu_vol[z0], hu_vol[z]), hu_vol[z2])

        for p in patches:
            ry0 = max(0, p["y0"]); ry1 = min(H, p["y1"])
            rx0 = max(0, p["x0"]); rx1 = min(W, p["x1"])
            roi = float((mip[ry0:ry1, rx0:rx1] > -950).mean()) if ry1 > ry0 and rx1 > rx0 else 0.5
            p["roi_ratio"]  = roi
            p["score_v950"] = p["padim_score"] * roi

        # G0: 절대 threshold 이상
        # slice_top30: 슬라이스 내 score_v950 상위 30개
        top30_indices = set(
            sorted(range(len(patches)), key=lambda i: -patches[i]["score_v950"])[:SLICE_TOP_N]
        )
        passed = [
            p for i, p in enumerate(patches)
            if p["padim_score"] >= P95_THRESHOLD or i in top30_indices
        ]
        if passed:
            out[z] = passed
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 3. STAGE 2: RD4AD E2
# ═══════════════════════════════════════════════════════════════════════════════
def load_rd4ad_backend():
    import rd4ad as _rd4ad
    print("[RD4AD] 백엔드 모듈 로드 완료")
    return _rd4ad


def score_rd4ad_slice(hu_vol, z, candidates, rd4ad_mod):
    """슬라이스별 raw cosine + roi_ratio 계산 (P5는 전체 완료 후 compute_p5_tracks에서 집계)"""
    if not candidates:
        return []

    device = rd4ad_mod.device
    Z, H, W = hu_vol.shape
    z0 = max(0, z-1); z2 = min(Z-1, z+1)
    mip = np.maximum(np.maximum(hu_vol[z0], hu_vol[z]), hu_vol[z2])

    crops, rois, crop_coords = [], [], []
    HALF = RD4AD_CROP // 2   # 48
    for c in candidates:
        # 원본과 동일: PaDiM 패치 center 기준 96×96 crop
        cy = (c["y0"] + c["y1"]) // 2
        cx = (c["x0"] + c["x1"]) // 2
        cy0 = cy - HALF; cx0 = cx - HALF
        cy1 = cy + HALF; cx1 = cx + HALF
        crop_coords.append((cy0, cx0, cy1, cx1))
        try:
            crop = rd4ad_mod.build_lung3ch_crop(hu_vol, z, cy0, cx0, cy1, cx1)
            crops.append(crop)
        except RuntimeError:
            crops.append(np.zeros((3, RD4AD_CROP, RD4AD_CROP), dtype=np.float32))
        # roi: GS2에서 계산된 값 재사용, 없으면 center 기준으로 재계산
        if "roi_ratio" in c:
            rois.append(c["roi_ratio"])
        else:
            ry0 = max(0, cy0); ry1 = min(H, cy1)
            rx0 = max(0, cx0); rx1 = min(W, cx1)
            roi = float((mip[ry0:ry1, rx0:rx1] > -950).mean()) if ry1 > ry0 and rx1 > rx0 else 0.5
            rois.append(roi)

    batch = torch.tensor(np.stack(crops, 0), dtype=torch.float32).to(device)
    with torch.no_grad():
        rd4ad_mod._teacher(batch)
        tf_late  = rd4ad_mod._teacher_feats["late"]
        tf_mid   = rd4ad_mod._teacher_feats["mid"]
        tf_early = rd4ad_mod._teacher_feats["early"]
        de_l, de_m, de_e = rd4ad_mod._student(tf_late)

        s_l = (1 - F.cosine_similarity(de_l,  tf_late,  dim=1)).mean(dim=(1,2))
        s_m = (1 - F.cosine_similarity(de_m,  tf_mid,   dim=1)).mean(dim=(1,2))
        s_e = (1 - F.cosine_similarity(de_e,  tf_early, dim=1)).mean(dim=(1,2))
        raw_scores = ((s_l + s_m + s_e) / 3).cpu().numpy()

    out = []
    for i, c in enumerate(candidates):
        out.append({**c, "z": z, "rd4ad_cosine": float(raw_scores[i]), "roi_ratio": rois[i]})
    return out


def compute_p5_tracks(all_rd4ad_patches):
    """원본 stage2_eval P5_len_norm_track_top3_mean 수식 구현.
    P1_times_roi = cosine * roi_ratio
    P1_len_norm  = P1_times_roi * (track_len / 3)
    P5           = track 내 P1_len_norm 상위 3개 평균
    반환: track 대표 패치 목록 (각 track에서 P5 최고 슬라이스 기준)
    """
    from collections import defaultdict

    # (y0,x0) 기준으로 그룹핑
    pos_groups = defaultdict(list)
    for p in all_rd4ad_patches:
        pos_groups[(p["y0"], p["x0"])].append(p)

    track_results = []
    for (y0, x0), patches in pos_groups.items():
        # z 순서 정렬 후 연속 구간 분리 (같은 위치라도 비연속이면 다른 track)
        patches.sort(key=lambda p: p["z"])
        groups = []
        group  = [patches[0]]
        for p in patches[1:]:
            if p["z"] == group[-1]["z"] + 1:
                group.append(p)
            else:
                groups.append(group)
                group = [p]
        groups.append(group)

        for g in groups:
            if len(g) < Z_TRACK_MIN:
                continue
            track_len  = len(g)
            p1_ln_vals = [p["rd4ad_cosine"] * p["roi_ratio"] * (track_len / 3.0) for p in g]
            top3_mean  = sum(sorted(p1_ln_vals, reverse=True)[:3]) / min(3, len(p1_ln_vals))
            best_patch = max(g, key=lambda p: p["rd4ad_cosine"])
            track_results.append({
                **best_patch,
                "track_len": track_len,
                "score":     round(top3_mean, 6),   # P5 = P5_len_norm_track_top3_mean
            })

    return track_results


# ═══════════════════════════════════════════════════════════════════════════════
# 4. STAGE 3: NSCLC
# ═══════════════════════════════════════════════════════════════════════════════
def load_nsclc_backend():
    import nsclc_classifier as _nc
    print("[NSCLC] 백엔드 모듈 로드 완료")
    return _nc


def score_nsclc(hu_vol, z, candidates, nsclc_mod):
    Z, H, W = hu_vol.shape
    z0 = max(0, z-1); z2 = min(Z-1, z+1)
    mip = np.maximum(np.maximum(hu_vol[z0], hu_vol[z]), hu_vol[z2])
    lz_pct = z / max(Z-1, 1)

    inputs = []
    for c in candidates:
        cy = (c["y0"] + c["y1"]) // 2; cx = (c["x0"] + c["x1"]) // 2
        y0n = cy - 48; x0n = cx - 48
        ry0 = max(0, y0n); ry1 = min(H, y0n+96)
        rx0 = max(0, x0n); rx1 = min(W, x0n+96)
        roi = float((mip[ry0:ry1, rx0:rx1] > -950).mean()) if ry1>ry0 and rx1>rx0 else 0.5
        inputs.append({"z": z, "y0": y0n, "x0": x0n,
                       "lung_z_pct": lz_pct, "crop_lung_roi_ratio": roi})

    results = nsclc_mod.run_nsclc_batch(hu_vol, inputs)
    return [{**c, "nsclc_prob": r["prob"], "nsclc_label": r["label"]}
            for c, r in zip(candidates, results)]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. XAI: Grad-CAM (원본 gradcam_run.py 로직)
# ═══════════════════════════════════════════════════════════════════════════════
def load_gradcam_model():
    """P-C-NORMAL30b masked-input Grad-CAM 모델 로드"""
    import torch.nn as nn
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

    class _GradCamModel(nn.Module):
        def __init__(self):
            super().__init__()
            backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.img_features  = backbone.features
            self.img_avgpool   = backbone.avgpool
            self.scalar_branch = nn.Sequential(
                nn.Linear(2, 32), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
                nn.Linear(32, 16), nn.ReLU(inplace=True),
            )
            self.fusion_head = nn.Sequential(
                nn.Dropout(p=0.2), nn.Linear(1280+16, 64),
                nn.ReLU(inplace=True), nn.Dropout(p=0.2), nn.Linear(64, 1),
            )
        def forward(self, img, scalar):
            x = self.img_features(img)
            x = self.img_avgpool(x)
            x = torch.flatten(x, 1)
            return self.fusion_head(torch.cat([x, self.scalar_branch(scalar)], 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(GRADCAM_PTH, map_location=device, weights_only=False)
    model  = _GradCamModel()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"[GradCAM] P-C-NORMAL30b 로드 완료 (epoch={ckpt.get('epoch','?')})")

    with open(GRADCAM_STATS) as f:
        stats = json.load(f)["features"]

    return model, stats, device


def _compute_gradcam(model, img_t, sc_t):
    """gradcam_run.py의 compute_gradcam 그대로 이식"""
    act_store = {}; grad_store = {}

    def fwd(m, i, o): act_store["v"] = o
    def bwd(m, gi, go): grad_store["v"] = go[0]

    tgt = model.img_features[7]
    fh  = tgt.register_forward_hook(fwd)
    bh  = tgt.register_full_backward_hook(bwd)

    model.zero_grad()
    logit = model(img_t, sc_t)
    logit.backward()
    fh.remove(); bh.remove()

    act  = act_store["v"]
    grad = grad_store["v"]
    w    = grad.mean(dim=(2, 3), keepdim=True)
    cam  = torch.relu((act * w).sum(dim=1, keepdim=True))
    cam_up = F.interpolate(cam, size=(96, 96), mode="bilinear", align_corners=False)
    cam_np = cam_up[0, 0].detach().cpu().numpy().astype(np.float32)

    cmin, cmax = float(cam_np.min()), float(cam_np.max())
    cam_np = (cam_np - cmin) / (cmax - cmin) if cmax - cmin > 1e-8 else np.zeros_like(cam_np)
    return cam_np, float(logit.item()), float(torch.sigmoid(logit).item())


def _mask_inside_norm(cam_raw, mask_96):
    """gradcam_run.py의 mask_inside_norm 그대로"""
    masked = cam_raw * mask_96
    vals   = masked[mask_96 > 0.5]
    if len(vals) == 0 or (vals.max() - vals.min()) < 1e-8:
        return np.zeros_like(masked)
    vmin, vmax = float(vals.min()), float(vals.max())
    return np.where(mask_96 > 0.5, (masked - vmin) / (vmax - vmin), 0.0).astype(np.float32)


def save_xai_image(hu_vol, z, patch, nsclc_prob, rd4ad_mod, gradcam_model_pack, out_path):
    try:
        if nsclc_prob >= 0.5:
            _save_gradcam(hu_vol, z, patch, gradcam_model_pack, out_path)
            return "gradcam"
        else:
            b64 = rd4ad_mod.run_rd4ad_e2_spatial_map_base64(hu_vol, z, patch["y0"], patch["x0"])
            if b64:
                import base64
                from PIL import Image
                from io import BytesIO
                Image.open(BytesIO(base64.b64decode(b64))).resize((128, 128)).save(out_path)
                return "rd4ad"
    except Exception as e:
        print(f"  [WARN] XAI 실패 z={z}: {e}")
    return None


def _save_gradcam(hu_vol, z, patch, gradcam_model_pack, out_path):
    """원본 gradcam_run.py 로직: masked input → YlOrRd RGBA → CT blend"""
    import matplotlib.cm as mpl_cm
    from PIL import Image
    from nsclc_classifier import build_nsclc_hu_crop

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    HU_MIN, HU_MAX = -1000.0, 200.0
    _YLRD = mpl_cm.get_cmap("YlOrRd")

    model, stats, device = gradcam_model_pack
    Z, H, W = hu_vol.shape
    cy = (patch["y0"] + patch["y1"]) // 2
    cx = (patch["x0"] + patch["x1"]) // 2
    y0n = cy - 48; x0n = cx - 48
    lz_pct = z / max(Z-1, 1)

    # roi ratio
    z0 = max(0,z-1); z2 = min(Z-1,z+1)
    mip = np.maximum(np.maximum(hu_vol[z0], hu_vol[z]), hu_vol[z2])
    ry0 = max(0,y0n); ry1 = min(H,y0n+96)
    rx0 = max(0,x0n); rx1 = min(W,x0n+96)
    roi = float((mip[ry0:ry1, rx0:rx1] > -950).mean()) if ry1>ry0 and rx1>rx0 else 0.5

    # scalar 정규화
    lz_n  = (lz_pct - stats["lung_z_percentile"]["mean"])  / stats["lung_z_percentile"]["std"]
    roi_n = (roi     - stats["crop_lung_roi_ratio"]["mean"]) / stats["crop_lung_roi_ratio"]["std"]
    sc_t  = torch.tensor([[lz_n, roi_n]], dtype=torch.float32).to(device)

    # HU crop → mask_96 (HU fallback, center ch > -950)
    hu_crop = build_nsclc_hu_crop(hu_vol, z, y0n, x0n)  # (3,96,96) raw HU
    mask_96 = (hu_crop[1] > -950).astype(np.float32)     # center ch 기준

    # masked input (mask_applied_to_image_only=True, 학습 조건 동일)
    arr        = (np.clip(hu_crop, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)
    masked_arr = arr * mask_96[None, :, :]
    img_arr    = (masked_arr - IMAGENET_MEAN[:,None,None]) / IMAGENET_STD[:,None,None]
    img_t      = torch.tensor(img_arr[None], dtype=torch.float32).to(device)

    # Grad-CAM 계산 (원본 compute_gradcam)
    cam_raw, logit, prob = _compute_gradcam(model, img_t, sc_t)
    model.zero_grad()

    # mask 내부 normalize (원본 mask_inside_norm)
    cam_norm = _mask_inside_norm(cam_raw, mask_96)

    # YlOrRd RGBA (alpha = cam^0.55 * 0.88)
    rgba       = _YLRD(cam_norm).astype(np.float32)       # (96,96,4)
    rgba[...,3] = np.power(cam_norm, ALPHA_GAMMA) * ALPHA_MAX

    # CT 배경 (lung window) + RGBA overlay blend
    ct_disp = (np.clip(hu_crop[1], -1350.0, 150.0) - (-1350.0)) / (150.0 - (-1350.0))
    ct_rgb  = np.stack([ct_disp]*3, axis=-1).astype(np.float32)   # (96,96,3)
    alpha3  = rgba[..., 3:4]
    blended = (ct_rgb * (1 - alpha3) + rgba[..., :3] * alpha3)
    blended = (blended * 255).clip(0, 255).astype(np.uint8)

    print(f"    [GradCAM] logit={logit:.3f} prob={prob:.3f}")
    Image.fromarray(blended).resize((128, 128)).save(out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 병변 GT 로드 (평가 전용)
# ═══════════════════════════════════════════════════════════════════════════════
def load_lesion_gt(patient_id="LUNG1-007"):
    gt = {}
    with open(LESION_CSV) as f:
        for row in csv.DictReader(f):
            if row["patient_id"] != patient_id or row["label"] != "1":
                continue
            z  = int(float(row["canonical_volume_z"]))
            cy = int(float(row["center_y"]))
            cx = int(float(row["center_x"]))
            gt.setdefault(z, []).append((cy, cx))
    print(f"[GT] LUNG1-007 병변 z={min(gt)}~{max(gt)}, {sum(len(v) for v in gt.values())} patches")
    return gt


def patch_hits_gt(patch, gt_z):
    for cy, cx in gt_z:
        if patch["y0"] <= cy <= patch["y1"] and patch["x0"] <= cx <= patch["x1"]:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# 7. 성능 집계
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate(patches_by_z, gt, label):
    gt_zs = set(gt.keys())
    total = sum(len(v) for v in patches_by_z.values())

    hit_zs, tp = set(), 0
    for z, patches in patches_by_z.items():
        if z not in gt:
            continue
        for p in patches:
            if patch_hits_gt(p, gt[z]):
                hit_zs.add(z)
                tp += 1

    hr   = len(hit_zs) / len(gt_zs) if gt_zs else 0
    prec = tp / total if total else 0

    print(f"\n[{label}]")
    print(f"  탐지 슬라이스: {len(patches_by_z):3d} | 총 패치: {total:5d}")
    print(f"  병변 슬라이스 hit rate: {len(hit_zs)}/{len(gt_zs)} = {hr:.3f}")
    print(f"  패치 precision: {tp}/{total} = {prec:.3f}")
    return {"label": label, "det_slices": len(patches_by_z), "total_patches": total,
            "hit_zs": len(hit_zs), "gt_zs": len(gt_zs), "hit_rate_z": hr,
            "tp_patches": tp, "precision": prec}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Pipeline: PaDiM → z-track → RD4AD E2 → NSCLC → XAI")
    print("=" * 60)

    hu_vol = load_dicom_volume(DICOM_DIR)
    Z = hu_vol.shape[0]
    gt = load_lesion_gt("LUNG1-007")

    # ── Stage 1: PaDiM ────────────────────────────────────────────
    print("\n[Stage 1] PaDiM 이상 탐지 중...")
    padim_mod = load_padim_backend()

    s1_by_z = {}
    for z in range(Z):
        hits = run_padim_on_slice(hu_vol[z], padim_mod)
        if hits:
            s1_by_z[z] = hits
        if (z+1) % 20 == 0 or z == Z-1:
            print(f"  진행: {z+1}/{Z} | 히트 슬라이스: {len(s1_by_z)}")

    perf_s1 = evaluate(s1_by_z, gt, "Stage1: PaDiM (P90+)")

    # ── GS2 Filter: 원본 선별 로직 ────────────────────────────────
    print("\n[GS2 Filter] G0(P95+) OR slice_top30 선별 중...")
    gs2_by_z = apply_gs2_filter(s1_by_z, hu_vol)
    gs2_total = sum(len(v) for v in gs2_by_z.values())
    print(f"  GS2 통과: {len(gs2_by_z)} 슬라이스, {gs2_total} 패치")
    evaluate(gs2_by_z, gt, "GS2 Filter (P95+ OR slice_top30)")

    # ── Z-Track 필터 ───────────────────────────────────────────────
    print("\n[Z-Track] 연속 2+ 슬라이스 필터 중...")
    zt_by_z = apply_ztrack(gs2_by_z, Z)
    zt_total = sum(len(v) for v in zt_by_z.values())
    print(f"  z-track 통과: {len(zt_by_z)} 슬라이스, {zt_total} 패치")

    # ── Stage 2: RD4AD E2 (전체 슬라이스 raw score 수집) ─────────────
    print("\n[Stage 2] RD4AD E2 raw cosine 스코어링 중...")
    rd4ad_mod = load_rd4ad_backend()

    all_rd4ad = []   # 전체 z-track 통과 패치의 raw score 목록
    for z, cands in zt_by_z.items():
        scored = score_rd4ad_slice(hu_vol, z, cands, rd4ad_mod)
        all_rd4ad.extend(scored)
    print(f"  raw score 수집 완료: {len(all_rd4ad)} 패치")

    # P5 = P1_len_norm_track_top3_mean (원본 stage2_eval 수식)
    track_cands = compute_p5_tracks(all_rd4ad)
    track_cands.sort(key=lambda c: -c["score"])
    top10_cands = track_cands[:10]   # P5 상위 10개만 Stage 3으로
    print(f"  P5 트랙 집계 완료: {len(track_cands)} 트랙 → 상위 10개 선택")

    # 평가용: z별로 재구성
    s2_by_z = {}
    for c in top10_cands:
        s2_by_z.setdefault(c["z"], []).append(c)

    perf_s2 = evaluate(s2_by_z, gt, "Stage2: RD4AD E2 P5 Top10")

    # ── Stage 3: NSCLC (prob >= 0.5 필터, 원본 기준) ──────────────
    print("\n[Stage 3] NSCLC 분류기 추론 중...")
    nsclc_mod = load_nsclc_backend()

    s3_by_z = {}
    for z, cands in s2_by_z.items():
        classified = score_nsclc(hu_vol, z, cands, nsclc_mod)
        passed = [c for c in classified if c["nsclc_prob"] >= 0.5]
        if passed:
            s3_by_z[z] = passed

    perf_s3 = evaluate(s3_by_z, gt, "Stage3: NSCLC prob>=0.5")

    # 상위 후보 목록 (P5 내림차순)
    all_cands = [(c["z"], c) for cands in s3_by_z.values() for c in cands]
    all_cands.sort(key=lambda x: -x[1]["score"])

    print(f"\n  Stage2 Top10 → NSCLC 통과: {len(all_cands)}개")
    print(f"\n  [Stage2 Top10 전체 NSCLC 결과]")
    print(f"  {'z':>4}  {'P5':>7}  {'NSCLC%':>7}  {'통과':>4}  {'GT?':>5}")
    for c in top10_cands:
        z = c["z"]
        passed = c.get("nsclc_prob", 0) >= 0.5 if "nsclc_prob" in c else "?"
        # NSCLC 결과 찾기
        nsclc_c = next((x for x in (s3_by_z.get(z, []) + [c]) if x.get("y0") == c["y0"]), c)
        prob = nsclc_c.get("nsclc_prob", float("nan"))
        in_gt = "✓ GT" if (z in gt and patch_hits_gt(c, gt.get(z, []))) else "-"
        flag = "✓" if prob >= 0.5 else "✗"
        print(f"  {z:>4}  {c['score']:>7.4f}  {prob*100:>6.1f}%  {flag:>4}  {in_gt:>5}")

    # ── XAI 이미지 생성 ────────────────────────────────────────────
    print(f"\n[XAI] Grad-CAM 모델 로드 중...")
    gradcam_model_pack = load_gradcam_model()

    print(f"\n[XAI] NSCLC 통과 후보 이미지 생성 중...")
    xai_saved = []
    for rank, (z, c) in enumerate(all_cands):
        in_gt = z in gt and patch_hits_gt(c, gt.get(z, []))
        fname = (f"rank{rank+1:02d}_z{z:03d}_p5{c['score']:.3f}_"
                 f"nsclc{c['nsclc_prob']:.2f}_{'GT' if in_gt else 'FP'}.png")
        fpath = os.path.join(OUTPUT_DIR, fname)
        xai_type = save_xai_image(hu_vol, z, c, c["nsclc_prob"], rd4ad_mod, gradcam_model_pack, fpath)
        if xai_type:
            xai_saved.append({"rank": rank+1, "z": z, "score": round(c["score"], 4),
                               "nsclc_prob": c["nsclc_prob"], "in_gt": bool(in_gt),
                               "file": fname, "xai_type": xai_type})
            print(f"  [{rank+1:02d}] z={z:03d} P5={c['score']:.3f} NSCLC={c['nsclc_prob']:.2f}"
                  f"  {'[GT]' if in_gt else '[FP]'}  → {fname}")

    # ── 성능 요약 저장 ─────────────────────────────────────────────
    summary = {
        "patient": "LUNG1-007",
        "total_slices": Z,
        "lesion_z_range": f"{min(gt)}~{max(gt)}",
        "stage1_padim":    perf_s1,
        "stage2_rd4ad_e2": perf_s2,
        "stage3_nsclc":    perf_s3,
        "xai_images": xai_saved,
    }
    out_json = os.path.join(OUTPUT_DIR, "performance_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== 최종 성능 요약 ===")
    for key in ["stage1_padim", "stage2_rd4ad_e2", "stage3_nsclc"]:
        p = summary[key]
        print(f"  {p['label']}: hit_rate={p['hit_rate_z']:.3f}  "
              f"precision={p['precision']:.3f}  patches={p['total_patches']}")
    print(f"\n[완료] {out_json}")


if __name__ == "__main__":
    main()
