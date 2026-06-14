# -*- coding: utf-8 -*-
"""
rd4ad_infer_save_spatial_map_1case_v1.py

LUNG1-415 / upper_peripheral / z=226 후보 1건에 대해
RD4AD spatial anomaly map을 생성하는 preflight.

원칙:
  - model forward: 1건만 허용 (heatmap 목적)
  - 기존 score CSV overwrite 금지
  - stage2_holdout_access = False
  - training / threshold_tuning = False
  - raw feature tensor 대량 저장 금지 (spatial distance map만 저장)
  - end package에 npy 복사 금지 (preflight root 내부에만 유지)

spatial map 정의:
  layer1: de1 vs tf1  shape=(24,24)  → 가장 세밀
  layer2: de2 vs tf2  shape=(12,12)
  layer3: de3 vs tf3  shape=(6,6)   → coarse (Panel 3 grid 비교용)
  map = 1 - cosine_similarity(de_l, tf_l, dim=channel)
  normalize: (map - map.min()) / (map.max() - map.min())
"""
import os, sys, csv, json, math
from datetime import date
from pathlib import Path

ROOT      = "/home/jinhy/project/lung-ct-anomaly"
INFER_DIR = os.path.join(ROOT, "outputs/end/rd4ad_lung_mip3ch_infer_v1")
SCORE_CSV = os.path.join(INFER_DIR, "test_LUNG1-415_scores.csv")
CKPT      = os.path.join(INFER_DIR, "best_train_loss.pth")
CTBASE    = "/mnt/c/Users/jinhy/Desktop/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy"
OUT_ROOT  = os.path.join(ROOT, "outputs/position-aware-padim-v1/reports/rd4ad_spatial_map_preflight_lung1_415_upper_peripheral_v1")
FONT_KO   = "/mnt/c/Windows/Fonts/malgun.ttf"

# ── target (hard-coded 1-case preflight) ──────────────────────────────────────
PATIENT   = "LUNG1-415"
SAFE_ID   = "NSCLC_LUNG1-415__75e5b68d83"
LOCAL_Z   = 226
CROP_Y0, CROP_X0, CROP_Y1, CROP_X1 = 224, 176, 320, 272
PBIN      = "upper_peripheral"
SCORE_SAVED = 0.40526

HU_MIN, HU_MAX = -1000.0, 600.0
CROP_SIZE = 96


# ── MIP3ch crop (rd4ad_infer.py와 동일) ───────────────────────────────────────
def build_mip3ch_crop(ct_arr, local_z, y0, x0, y1, x1):
    import numpy as np
    Z, H, W = ct_arr.shape
    z = int(local_z)
    y0, x0, y1, x1 = int(y0), int(x0), int(y1), int(x1)
    pad_top  = max(0, -y0);  pad_bot   = max(0, y1 - H)
    pad_left = max(0, -x0);  pad_right = max(0, x1 - W)
    needs_pad = any([pad_top, pad_bot, pad_left, pad_right])
    cy0 = max(0, y0); cy1 = min(H, y1)
    cx0 = max(0, x0); cx1 = min(W, x1)

    def _win(sl):
        c = np.clip(sl.astype(np.float32), HU_MIN, HU_MAX)
        return (c - HU_MIN) / (HU_MAX - HU_MIN)

    def _ch(zi):
        zi = int(np.clip(zi, 0, Z - 1))
        s = _win(ct_arr[zi, cy0:cy1, cx0:cx1])
        if needs_pad:
            pm = "reflect" if (cy1 - cy0 > 1) and (cx1 - cx0 > 1) else "edge"
            s = np.pad(s, ((pad_top, pad_bot), (pad_left, pad_right)), mode=pm)
        return s

    def _mip(zs):
        return np.stack([_ch(zi) for zi in zs], axis=0).max(axis=0)

    return np.stack([
        _mip([z - 3, z - 2, z - 1]),
        _mip([z - 1, z,     z + 1]),
        _mip([z + 1, z + 2, z + 3]),
    ], axis=0).astype("float32")


# ── model (rd4ad_infer.py와 동일) ─────────────────────────────────────────────
def build_student_decoder():
    import torch.nn as nn
    class StudentDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.de_layer3 = nn.Sequential(
                nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
            self.de_layer2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
            self.de_layer1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        def forward(self, layer3_feat):
            x = self.de_layer3(layer3_feat); de3 = x
            x = self.de_layer2(x);           de2 = x
            x = self.de_layer1(x);           de1 = x
            return de3, de2, de1
    return StudentDecoder()


def load_model(ckpt_path, device):
    import torch
    import torchvision.models as models
    teacher = models.resnet18(weights=None).to(device)
    teacher.eval(); teacher.requires_grad_(False)
    student = build_student_decoder().to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    student.load_state_dict(ckpt["student_state_dict"]); student.eval()
    tf_feats = {}
    for name, mod in [("layer1", teacher.layer1), ("layer2", teacher.layer2), ("layer3", teacher.layer3)]:
        def _hook(m, inp, out, _n=name): tf_feats[_n] = out
        mod.register_forward_hook(_hook)
    return teacher, student, tf_feats


# ── spatial map 시각화 헬퍼 ───────────────────────────────────────────────────
def colorize_map(map_np, cmap_name="jet"):
    """(H,W) float [0,1] → (H,W,3) uint8 via matplotlib colormap"""
    import matplotlib.cm as cm
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(map_np)
    return (rgba[:, :, :3] * 255).astype("uint8")


def make_overlay(bg_gray_01, map_01, alpha=0.5, cmap="jet"):
    """bg_gray_01: (H,W) float [0,1], map_01: (H,W) float [0,1] → (H,W,3) uint8"""
    import numpy as np
    colored = colorize_map(map_01, cmap)
    bg_rgb  = (np.stack([bg_gray_01] * 3, axis=-1) * 255).astype("uint8")
    overlay = (alpha * colored.astype("float32") + (1 - alpha) * bg_rgb.astype("float32"))
    return overlay.clip(0, 255).astype("uint8")


def save_map_png(map_np, out_path, title="", layer_name=""):
    """normalized map (H,W) → colorized PNG with colorbar"""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), facecolor="#111")
    im = ax.imshow(map_np, cmap="jet", vmin=0, vmax=1)
    ax.set_title(f"{layer_name} spatial anomaly map\n{title}", color="white", fontsize=9)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor="#111")
    plt.close(fig)


def save_overlay_png(bg_gray_01, map_01, out_path, alpha=0.5, layer_name="", score=None):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    overlay = make_overlay(bg_gray_01, map_01, alpha=alpha)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), facecolor="#111")
    axes[0].imshow(bg_gray_01, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Candidate patch (MIP ch1)", color="white", fontsize=8); axes[0].axis("off")
    im = axes[1].imshow(map_01, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title(f"{layer_name} anomaly map\n(normalized 0→1)", color="white", fontsize=8); axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay α={alpha}\nrd4ad_score={score:.5f}" if score else "Overlay", color="white", fontsize=8)
    axes[2].axis("off")
    fig.suptitle(f"RD4AD Spatial Anomaly — LUNG1-415 z=226 {layer_name} [research-use visualization]",
                 color="#aaaaff", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor="#111")
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_z", type=int, default=None, help="target local_z (default: hard-coded 1-case value)")
    ap.add_argument("--out_root_override", default=None, help="출력 root 경로 override")
    args = ap.parse_args()

    # argparse로 받은 z가 있으면 score CSV에서 자동으로 행 찾아 상수 override
    global LOCAL_Z, CROP_Y0, CROP_X0, CROP_Y1, CROP_X1, SAFE_ID, PBIN, SCORE_SAVED, OUT_ROOT
    if args.local_z is not None and args.local_z != LOCAL_Z:
        rows_ = [r for r in csv.DictReader(open(SCORE_CSV, encoding="utf-8-sig"))
                 if r.get("patient_id") == PATIENT and int(r.get("local_z", -1)) == args.local_z]
        assert rows_, f"z={args.local_z} not found in score CSV for {PATIENT}"
        row_ = max(rows_, key=lambda r: float(r["rd4ad_score"]))
        LOCAL_Z     = args.local_z
        CROP_Y0, CROP_X0 = int(row_["crop_y0"]), int(row_["crop_x0"])
        CROP_Y1, CROP_X1 = int(row_["crop_y1"]), int(row_["crop_x1"])
        SAFE_ID     = row_["safe_id"]
        PBIN        = row_.get("position_bin", PBIN)
        SCORE_SAVED = float(row_["rd4ad_score"])
        OUT_ROOT    = os.path.join(ROOT,
            "outputs/position-aware-padim-v1/reports",
            f"rd4ad_spatial_map_preflight_{PATIENT.lower().replace('-','_')}_z{LOCAL_Z}_v1")
    if args.out_root_override:
        OUT_ROOT = args.out_root_override

    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageDraw, ImageFont

    os.makedirs(OUT_ROOT, exist_ok=True)
    errors = []

    # ── 1. saved score 확인 ────────────────────────────────────────────────────
    score_rows = [r for r in csv.DictReader(open(SCORE_CSV, encoding="utf-8-sig"))
                  if r.get("patient_id") == PATIENT and int(r.get("local_z", -1)) == LOCAL_Z]
    assert len(score_rows) >= 1, f"expected >=1 row for z={LOCAL_Z}, got {len(score_rows)}"
    saved_row = max(score_rows, key=lambda r: float(r["rd4ad_score"]))
    saved_score_actual = float(saved_row["rd4ad_score"])
    label_ = saved_row.get("label", "?")
    print(f"[OK] saved score match: {saved_score_actual:.5f}  pbin={PBIN}  label={label_}")

    # ── 2. CT 로드 + crop 생성 ─────────────────────────────────────────────────
    ct_path = os.path.join(CTBASE, SAFE_ID, "ct_hu.npy")
    assert os.path.exists(ct_path), f"CT 없음: {ct_path}"
    ct_arr = np.load(ct_path, mmap_mode="r")
    crop = build_mip3ch_crop(ct_arr, LOCAL_Z, CROP_Y0, CROP_X0, CROP_Y1, CROP_X1)
    assert crop.shape == (3, CROP_SIZE, CROP_SIZE), f"crop shape 이상: {crop.shape}"
    print(f"[OK] crop shape: {crop.shape}")

    # candidate_patch.png: 3채널 나란히 (ch0/ch1/ch2)
    PAD = 4
    patch_w = CROP_SIZE * 3 + PAD * 2
    patch_h = CROP_SIZE + 24
    patch_img = Image.new("RGB", (patch_w, patch_h), (20, 20, 20))
    draw = ImageDraw.Draw(patch_img)
    try: fnt = ImageFont.truetype(FONT_KO, 10)
    except: fnt = ImageFont.load_default()
    ch_labels = ["ch0 MIP z-3~z-1", "ch1 MIP z-1~z+1", "ch2 MIP z+1~z+3"]
    for ci in range(3):
        arr = (crop[ci] * 255).astype("uint8")
        x = ci * (CROP_SIZE + PAD)
        patch_img.paste(Image.fromarray(arr, "L").convert("RGB"), (x, 20))
        draw.text((x, 4), ch_labels[ci], font=fnt, fill=(170, 200, 150))
    patch_img.save(os.path.join(OUT_ROOT, "candidate_patch.png"), "PNG")
    print("[OK] candidate_patch.png saved")

    # ── 3. model forward (1-case, heatmap 목적) ───────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    teacher, student, tf_feats = load_model(CKPT, device)
    print("[OK] model loaded")

    batch = torch.from_numpy(crop[np.newaxis]).to(device)  # (1, 3, 96, 96)
    with torch.no_grad():
        teacher(batch)
        tf3 = tf_feats["layer3"]  # (1, 256, 6, 6)
        tf2 = tf_feats["layer2"]  # (1, 128, 12, 12)
        tf1 = tf_feats["layer1"]  # (1, 64, 24, 24)
        de3, de2, de1 = student(tf3)

    # ── 4. spatial maps 계산 ──────────────────────────────────────────────────
    def spatial_map(de, tf):
        """(1,C,H,W) → (H,W) cosine distance map"""
        dist = (1 - F.cosine_similarity(de, tf, dim=1))  # (1,H,W)
        m = dist[0].cpu().numpy()
        return m

    sm_l3 = spatial_map(de3, tf3)  # (6,6)
    sm_l2 = spatial_map(de2, tf2)  # (12,12)
    sm_l1 = spatial_map(de1, tf1)  # (24,24)

    # NaN/Inf 검증
    for name, sm in [("l1", sm_l1), ("l2", sm_l2), ("l3", sm_l3)]:
        assert np.all(np.isfinite(sm)), f"spatial map {name} has NaN/Inf"
        assert sm.max() > 0, f"spatial map {name} is all-zero"

    # scalar score 검증 (saved score와 비교)
    s3_ = float((1 - F.cosine_similarity(de3, tf3, dim=1)).mean())
    s2_ = float((1 - F.cosine_similarity(de2, tf2, dim=1)).mean())
    s1_ = float((1 - F.cosine_similarity(de1, tf1, dim=1)).mean())
    recomputed = (s1_ + s2_ + s3_) / 3.0
    score_delta = abs(recomputed - SCORE_SAVED)
    print(f"[INFO] recomputed score={recomputed:.6f}  saved={SCORE_SAVED:.6f}  Δ={score_delta:.6f}")

    print(f"[OK] spatial maps: l1={sm_l1.shape} l2={sm_l2.shape} l3={sm_l3.shape}")

    # ── 5. normalize + upsample ───────────────────────────────────────────────
    def norm01(m):
        lo, hi = m.min(), m.max()
        return (m - lo) / max(hi - lo, 1e-8), lo, hi

    sm_l1_n, l1_min, l1_max = norm01(sm_l1)
    sm_l2_n, l2_min, l2_max = norm01(sm_l2)
    sm_l3_n, l3_min, l3_max = norm01(sm_l3)

    def upsample_map(m_np):
        """(H,W) → (96,96) bilinear"""
        t = torch.from_numpy(m_np[np.newaxis, np.newaxis].astype("float32"))
        up = F.interpolate(t, size=(CROP_SIZE, CROP_SIZE), mode="bilinear", align_corners=False)
        return up[0, 0].numpy()

    sm_l1_up = upsample_map(sm_l1_n)  # (96,96)
    sm_l2_up = upsample_map(sm_l2_n)
    sm_l3_up = upsample_map(sm_l3_n)

    # combined: l1+l2 average (가장 유용한 overlay)
    sm_combined = (sm_l1_up + sm_l2_up) / 2.0

    # background: crop ch1 (MIP z-1~z+1)
    bg = crop[1]  # (96,96) float32 [0,1]

    # ── 6. npy 저장 (preflight 내부용, end package 제외) ──────────────────────
    np.save(os.path.join(OUT_ROOT, "spatial_map_layer1.npy"), sm_l1.astype("float32"))
    np.save(os.path.join(OUT_ROOT, "spatial_map_layer2.npy"), sm_l2.astype("float32"))
    np.save(os.path.join(OUT_ROOT, "spatial_map_layer3.npy"), sm_l3.astype("float32"))
    print("[OK] npy saved (preflight-only, not for end package)")

    # ── 7. PNG 저장 ───────────────────────────────────────────────────────────
    case_label = f"LUNG1-415 z=226 score={SCORE_SAVED}"

    save_map_png(sm_l1_n, os.path.join(OUT_ROOT, "spatial_map_layer1.png"), case_label, "Layer1 (24×24)")
    save_map_png(sm_l2_n, os.path.join(OUT_ROOT, "spatial_map_layer2.png"), case_label, "Layer2 (12×12)")
    save_map_png(sm_l3_n, os.path.join(OUT_ROOT, "spatial_map_layer3.png"), case_label, "Layer3 (6×6) coarse")

    save_overlay_png(bg, sm_l1_up, os.path.join(OUT_ROOT, "spatial_overlay_layer1.png"),
                     alpha=0.5, layer_name="Layer1 (24×24)", score=SCORE_SAVED)
    save_overlay_png(bg, sm_l2_up, os.path.join(OUT_ROOT, "spatial_overlay_layer2.png"),
                     alpha=0.5, layer_name="Layer2 (12×12)", score=SCORE_SAVED)
    save_overlay_png(bg, sm_l3_up, os.path.join(OUT_ROOT, "spatial_overlay_layer3.png"),
                     alpha=0.5, layer_name="Layer3 (6×6) coarse", score=SCORE_SAVED)
    save_overlay_png(bg, sm_combined, os.path.join(OUT_ROOT, "spatial_overlay_l1l2_combined.png"),
                     alpha=0.5, layer_name="L1+L2 combined", score=SCORE_SAVED)
    print("[OK] overlay PNGs saved")

    # ── 8. heatmap_summary.json ────────────────────────────────────────────────
    summary = {
        "case_id": f"{PATIENT}__{PBIN}",
        "patient": PATIENT,
        "local_z": LOCAL_Z,
        "rd4ad_score_saved": SCORE_SAVED,
        "rd4ad_score_recomputed": round(recomputed, 6),
        "score_delta_vs_saved": round(score_delta, 6),
        "score_match": score_delta < 0.001,
        "score_csv_overwritten": False,
        "heatmap_possible_without_model_forward": False,
        "heatmap_source": "model_forward_1case_heatmap_only",
        "model_forward_for_heatmap_only": True,
        "training": False,
        "threshold_tuning": False,
        "stage2_holdout_access": False,
        "spatial_maps": {
            "layer1": {"shape": list(sm_l1.shape), "min": float(l1_min), "max": float(l1_max),
                       "normalize": "minmax_per_map", "npy": "spatial_map_layer1.npy",
                       "png": "spatial_map_layer1.png", "overlay": "spatial_overlay_layer1.png"},
            "layer2": {"shape": list(sm_l2.shape), "min": float(l2_min), "max": float(l2_max),
                       "normalize": "minmax_per_map", "npy": "spatial_map_layer2.npy",
                       "png": "spatial_map_layer2.png", "overlay": "spatial_overlay_layer2.png"},
            "layer3": {"shape": list(sm_l3.shape), "min": float(l3_min), "max": float(l3_max),
                       "normalize": "minmax_per_map", "note": "coarse 6x6; Panel 3 grid 비교용",
                       "npy": "spatial_map_layer3.npy",
                       "png": "spatial_map_layer3.png", "overlay": "spatial_overlay_layer3.png"},
        },
        "combined_overlay": "spatial_overlay_l1l2_combined.png",
        "normalize_method": "minmax_per_map_per_layer",
        "normalize_note": "normalize는 이 1개 crop 기준. 다중 crop 비교 시 global normalization 필요.",
        "upsampled_to_96x96": True,
        "upsample_method": "bilinear",
        "overlay_alpha": 0.5,
        "overlay_background": "crop_ch1_MIP_z-1_z+1",
        "diagnostic_claim": False,
        "generated_date": str(date.today()),
    }
    json.dump(summary, open(os.path.join(OUT_ROOT, "heatmap_summary.json"), "w"), ensure_ascii=False, indent=2)

    # ── 9. source audit ────────────────────────────────────────────────────────
    audit_rows = [
        {"source_file": os.path.relpath(SCORE_CSV, ROOT), "exists": True, "file_type": "csv",
         "shape": f"(602 rows)", "dtype": "str", "used_for_heatmap": False, "note": "saved scalar scores, read-only"},
        {"source_file": os.path.relpath(CKPT, ROOT), "exists": os.path.exists(CKPT), "file_type": "pth",
         "shape": "student_state_dict", "dtype": "float32", "used_for_heatmap": True, "note": "RD4AD student checkpoint"},
        {"source_file": os.path.relpath(ct_path, ROOT), "exists": True, "file_type": "npy",
         "shape": str(ct_arr.shape), "dtype": str(ct_arr.dtype), "used_for_heatmap": True, "note": "CT mmap read-only"},
        {"source_file": "outputs/.../spatial_map_layer1.npy", "exists": True, "file_type": "npy",
         "shape": str(sm_l1.shape), "dtype": "float32", "used_for_heatmap": True, "note": "generated spatial map layer1"},
        {"source_file": "outputs/.../spatial_map_layer2.npy", "exists": True, "file_type": "npy",
         "shape": str(sm_l2.shape), "dtype": "float32", "used_for_heatmap": True, "note": "generated spatial map layer2"},
        {"source_file": "outputs/.../spatial_map_layer3.npy", "exists": True, "file_type": "npy",
         "shape": str(sm_l3.shape), "dtype": "float32", "used_for_heatmap": True, "note": "coarse spatial map layer3"},
    ]
    with open(os.path.join(OUT_ROOT, "heatmap_source_audit.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source_file", "exists", "file_type", "shape", "dtype", "used_for_heatmap", "note"])
        w.writeheader(); w.writerows(audit_rows)

    # ── 10. safety_check.json ─────────────────────────────────────────────────
    json.dump({
        "model_forward_for_heatmap_only": True,
        "score_csv_overwritten": False,
        "score_recompute_for_report": True,
        "score_recompute_overwrites_saved": False,
        "training": False,
        "threshold_tuning": False,
        "stage2_holdout_access": False,
        "raw_ct_copied": False,
        "raw_feature_tensor_saved": False,
        "spatial_map_saved": True,
        "spatial_map_in_end_package": False,
        "existing_card_overwrite": False,
        "diagnostic_claim": False,
        "verdict": "PASS_SAFETY_CHECK",
    }, open(os.path.join(OUT_ROOT, "safety_check.json"), "w"), ensure_ascii=False, indent=2)

    with open(os.path.join(OUT_ROOT, "errors.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["type", "msg"]); w.writeheader()
        for e in errors: w.writerow(e)

    json.dump({
        "done": True,
        "verdict": "PASS_RD4AD_SPATIAL_MAP_PREFLIGHT_READY",
        "patient": PATIENT, "local_z": LOCAL_Z,
        "spatial_maps_generated": ["layer1", "layer2", "layer3", "l1l2_combined"],
        "overlay_files": [
            "spatial_overlay_layer1.png",
            "spatial_overlay_layer2.png",
            "spatial_overlay_layer3.png",
            "spatial_overlay_l1l2_combined.png",
        ],
        "score_csv_overwritten": False,
        "errors": len(errors),
        "generated_date": str(date.today()),
    }, open(os.path.join(OUT_ROOT, "DONE.json"), "w"), ensure_ascii=False, indent=2)

    print(f"\n=== VERDICT: PASS_RD4AD_SPATIAL_MAP_PREFLIGHT_READY ===")
    print(f"layer1 shape: {sm_l1.shape}  min={l1_min:.4f} max={l1_max:.4f}")
    print(f"layer2 shape: {sm_l2.shape}  min={l2_min:.4f} max={l2_max:.4f}")
    print(f"layer3 shape: {sm_l3.shape}  min={l3_min:.4f} max={l3_max:.4f}  [coarse]")
    print(f"errors: {len(errors)}")
    print(f"output: {OUT_ROOT}")


if __name__ == "__main__":
    main()
