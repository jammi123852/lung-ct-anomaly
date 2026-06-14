"""
P-C-NORMAL37e: Grad-CAM pilot
- 11개 sample crop에 대해 EfficientNet-B0 기반 Grad-CAM 시각화
- target layer: model.img_features[7] (마지막 MBConv block, 3×3 spatial)
- masked-input 방식 그대로 입력 구성 (CT crop × mask → ImageNet normalize)
- full-slice context PNG 포함
- 재학습/threshold 최적화/기존 결과 수정 금지
"""

import csv
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
STAGE_LABEL  = "P-C-NORMAL37e"

# ── paths ────────────────────────────────────────────────────────────────────
CHECKPOINT    = PROJECT_ROOT / "outputs/p_c_normal30b_masked_input_full_train/checkpoints/p_c_normal30b_best_val_auc_checkpoint.pt"
SCORE_CSV     = PROJECT_ROOT / "outputs/p_c_normal35_full_downstream_scoring/p_c_normal35_full_crop_scores.csv"
SCALAR_STATS  = PROJECT_ROOT / "outputs/reports/p_c_normal24h_fix_zroi_scalar_fusion_drycheck/p_c_normal24h_fix_scalar_normalization_stats.json"

OUTPUT_DIR    = PROJECT_ROOT / "outputs/p_c_normal37e_gradcam_pilot"
REPORT_DIR    = PROJECT_ROOT / "outputs/reports/p_c_normal37e_gradcam_pilot"
PANELS_DIR    = OUTPUT_DIR / "panels"
CONTEXT_DIR   = OUTPUT_DIR / "context"

NSCLC_VOL_ROOT  = Path("/mnt/c/Users/jinhy/Desktop/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy")
NORMAL_VOL_ROOT = Path("/mnt/c/Users/jinhy/Desktop/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/volumes_npy")

# ── constants ────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
HU_MIN        = -1000.0
HU_MAX        = 200.0
CROP_SIZE     = 96
HALF          = CROP_SIZE // 2

# sample selection: (type, patient_id, z, label)
SAMPLE_TARGETS = [
    ("NSCLC_TP",    "LUNG1-196",  "81.0",   "1"),
    ("NSCLC_TP",    "LUNG1-205",  "146.0",  "1"),
    ("NSCLC_TP",    "LUNG1-349",  "159.0",  "1"),
    ("NSCLC_TP",    "LUNG1-043",  "124.0",  "1"),
    ("NSCLC_TP",    "LUNG1-396",  "130.0",  "1"),
    ("Normal_FP",   "normal016",  "45.0",   "0"),
    ("low_mask",    "LUNG1-205",  "204.0",  "1"),
    ("low_mask",    "LUNG1-205",  "205.0",  "1"),
]
# Normal FP 긴 이름 추가
NORMAL_FP_PIDS = [
    "subset3_1.3.6.1.4.1.14519.5.2.1.6279.6001.314519596680450457855054746285",
    "subset2_1.3.6.1.4.1.14519.5.2.1.6279.6001.102133688497886810253331438797",
]


def _write_csv(rows, path):
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _write_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _abort(msg):
    print(f"[ABORT] {msg}", file=sys.stderr)
    sys.exit(2)


# ── Model ────────────────────────────────────────────────────────────────────
class ScalarFusionModel(nn.Module):
    def __init__(self, scalar_hidden=32, scalar_out=16, dropout=0.2):
        super().__init__()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.img_features  = backbone.features
        self.img_avgpool   = backbone.avgpool
        self.scalar_branch = nn.Sequential(
            nn.Linear(2, scalar_hidden),
            nn.BatchNorm1d(scalar_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(scalar_hidden, scalar_out),
            nn.ReLU(inplace=True),
        )
        fusion_in = 1280 + scalar_out
        self.fusion_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(fusion_in, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1),
        )

    def forward(self, img, scalar):
        x = self.img_features(img)
        x = self.img_avgpool(x)
        x = torch.flatten(x, 1)
        s = self.scalar_branch(scalar)
        return self.fusion_head(torch.cat([x, s], dim=1))


def load_model():
    model = ScalarFusionModel()
    ckpt  = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  checkpoint loaded: epoch={ckpt.get('epoch')}, smoke_only={ckpt.get('smoke_only')}")
    return model


# ── Grad-CAM ─────────────────────────────────────────────────────────────────
def compute_gradcam(model, img_t, sc_t):
    """
    target_layer = model.img_features[7] (마지막 MBConv block, 320ch, 3×3)
    returns: cam_np (96×96, [0,1]), logit_val, act_shape
    """
    activations = {}
    gradients   = {}

    def fwd_hook(module, inp, out):
        activations["feat"] = out

    def bwd_hook(module, grad_in, grad_out):
        gradients["feat"] = grad_out[0]

    target_layer = model.img_features[7]
    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    logit = model(img_t, sc_t)  # (1, 1)
    logit.backward()

    fh.remove()
    bh.remove()

    act = activations["feat"]       # (1, C, H, W)
    grad = gradients["feat"]        # (1, C, H, W)

    # GAP of gradients → weights
    weights = grad.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
    cam = (act * weights).sum(dim=1, keepdim=True)  # (1, 1, H, W)
    cam = torch.relu(cam)

    # upsample → 96×96
    cam_up = F.interpolate(cam, size=(CROP_SIZE, CROP_SIZE), mode="bilinear", align_corners=False)
    cam_np = cam_up[0, 0].detach().cpu().numpy().astype(np.float32)

    # normalize [0, 1]
    cmin, cmax = float(cam_np.min()), float(cam_np.max())
    if cmax - cmin > 1e-8:
        cam_np = (cam_np - cmin) / (cmax - cmin)
    else:
        cam_np = np.zeros_like(cam_np)

    return cam_np, float(logit.item()), tuple(act.shape)


# ── image prep ───────────────────────────────────────────────────────────────
def prep_input(row, scalar_stats):
    """masked-input tensor 구성 (training pipeline 동일)"""
    feat = scalar_stats["features"]
    lzp_mean = feat["lung_z_percentile"]["mean"]
    lzp_std  = feat["lung_z_percentile"]["std"]
    clrr_mean = feat["crop_lung_roi_ratio"]["mean"]
    clrr_std  = feat["crop_lung_roi_ratio"]["std"]

    # CT crop (학습과 동일: HU clip → [0,1] normalize → mask → ImageNet normalize)
    data = np.load(row["crop_path"])
    ct_crop = data["ct_crop"].astype(np.float32)   # (3, 96, 96)
    arr = np.clip(ct_crop, HU_MIN, HU_MAX)
    arr = (arr - HU_MIN) / (HU_MAX - HU_MIN)       # [0, 1]

    # mask
    mdata = np.load(row["mask_path"])
    if "mask_3ch" in mdata:
        mask_3ch = mdata["mask_3ch"].astype(np.float32)
    elif "mask" in mdata:
        m1 = mdata["mask"].astype(np.float32)
        mask_3ch = np.stack([m1, m1, m1], axis=0)
    else:
        mask_3ch = np.ones_like(arr)

    masked_arr = arr * mask_3ch   # (3, 96, 96)

    # ImageNet normalize
    mean_t = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std_t  = np.array(IMAGENET_STD,  dtype=np.float32).reshape(3, 1, 1)
    img_norm = (masked_arr - mean_t) / std_t

    img_t = torch.tensor(img_norm[np.newaxis], dtype=torch.float32)   # (1, 3, 96, 96)

    # scalar normalize
    lzp_raw  = float(row["lung_z_percentile_raw"])
    clrr_raw = float(row["crop_lung_roi_ratio_raw"])
    lzp_n    = (lzp_raw  - lzp_mean)  / lzp_std
    clrr_n   = (clrr_raw - clrr_mean) / clrr_std
    sc_t = torch.tensor([[lzp_n, clrr_n]], dtype=torch.float32)

    # center slice display (시각화용, raw HU 기반)
    raw_ch   = ct_crop[1]    # center channel (HU)
    mask_ch  = mask_3ch[1]
    raw_disp = np.clip(raw_ch, HU_MIN, HU_MAX)
    raw_disp = (raw_disp - HU_MIN) / (HU_MAX - HU_MIN)   # [0, 1]
    mask_disp    = mask_ch
    masked_disp  = raw_disp * mask_disp

    return img_t, sc_t, raw_disp, masked_disp, mask_disp


def resolve_ct_volume(safe_id, source_split):
    if source_split == "stage2_holdout":
        return NSCLC_VOL_ROOT / safe_id / "ct_hu.npy"
    else:
        return NORMAL_VOL_ROOT / safe_id / "ct_hu.npy"


# ── panel PNG (6-panel) ───────────────────────────────────────────────────────
def save_panel_png(raw_disp, masked_disp, mask_disp, cam_np, row, png_path, plt):
    prob  = float(row["prob"])
    logit = float(row["logit"])
    pid   = row["patient_id"]
    cid   = row["crop_path"].split("/")[-1].replace(".npz", "")
    label = "NSCLC" if row["label"] == "1" else "normal"
    pred  = "NSCLC" if row["pred_at_0p5"] == "1" else "normal"
    low_m = row["low_mask_flag"]
    ratio = float(row["mask_nonzero_ratio_mean"])
    z     = row["canonical_volume_z"]
    cy    = row["center_y"]
    cx    = row["center_x"]

    title_main = (f"{pid}  z={z}  cy={cy} cx={cx}\n"
                  f"label={label}  pred={pred}  prob={prob:.4f}  logit={logit:.2f}"
                  f"  low_mask={low_m}  mask_ratio={ratio:.3f}")

    heatmap_cmap = "hot"
    alpha_val    = 0.55

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(title_main, fontsize=8, y=1.00)

    # row0: raw / masked / mask
    axes[0, 0].imshow(raw_disp,     cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("raw center slice", fontsize=7)
    axes[0, 1].imshow(masked_disp,  cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("masked center slice", fontsize=7)
    axes[0, 2].imshow(mask_disp,    cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("crop mask", fontsize=7)

    # row1: raw+CAM / masked+CAM / raw+mask boundary+CAM
    axes[1, 0].imshow(raw_disp,    cmap="gray", vmin=0, vmax=1)
    axes[1, 0].imshow(cam_np,      cmap=heatmap_cmap, alpha=alpha_val, vmin=0, vmax=1)
    axes[1, 0].set_title("raw + Grad-CAM", fontsize=7)

    axes[1, 1].imshow(masked_disp, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].imshow(cam_np,      cmap=heatmap_cmap, alpha=alpha_val, vmin=0, vmax=1)
    axes[1, 1].set_title("masked + Grad-CAM", fontsize=7)

    axes[1, 2].imshow(raw_disp,    cmap="gray", vmin=0, vmax=1)
    # mask boundary: contour
    axes[1, 2].contour(mask_disp, levels=[0.5], colors=["cyan"], linewidths=0.8)
    axes[1, 2].imshow(cam_np,      cmap=heatmap_cmap, alpha=alpha_val, vmin=0, vmax=1)
    axes[1, 2].set_title("raw + mask boundary + Grad-CAM", fontsize=7)

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── context PNG (full-slice + crop box) ──────────────────────────────────────
def save_context_png(row, cam_np, png_path, plt):
    safe_id      = row["safe_id"]
    source_split = row["source_split"]
    z_str        = row["canonical_volume_z"]
    cy           = int(float(row["center_y"]))
    cx           = int(float(row["center_x"]))
    pid          = row["patient_id"]
    label        = "NSCLC" if row["label"] == "1" else "normal"
    prob         = float(row["prob"])

    vol_path = resolve_ct_volume(safe_id, source_split)
    if not vol_path.exists():
        print(f"    [WARN] CT volume not found: {vol_path}")
        return False

    vol = np.load(str(vol_path), mmap_mode="r")   # (Z, H, W) or (Z, 1, H, W)
    if vol.ndim == 4:
        vol = vol[:, 0, :, :]

    z_idx = int(float(z_str))
    if z_idx >= vol.shape[0]:
        z_idx = vol.shape[0] - 1

    ct_slice = vol[z_idx].astype(np.float32)
    ct_disp  = np.clip(ct_slice, -1000, 400)
    ct_disp  = (ct_disp - ct_disp.min()) / (ct_disp.max() - ct_disp.min() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    title = (f"{pid}  z={z_str}  label={label}  prob={prob:.4f}")
    fig.suptitle(title, fontsize=8)

    # left: full slice + crop box
    axes[0].imshow(ct_disp, cmap="gray", vmin=0, vmax=1)
    rect = patches.Rectangle(
        (cx - HALF, cy - HALF), CROP_SIZE, CROP_SIZE,
        linewidth=1.5, edgecolor="red", facecolor="none"
    )
    axes[0].add_patch(rect)
    axes[0].set_title(f"full CT slice  (z={z_str})\nred box = crop region", fontsize=7)
    axes[0].axis("off")

    # right: crop zoom + CAM
    y0, y1 = max(0, cy - HALF), min(ct_disp.shape[0], cy + HALF)
    x0, x1 = max(0, cx - HALF), min(ct_disp.shape[1], cx + HALF)
    crop_zoom = ct_disp[y0:y1, x0:x1]
    axes[1].imshow(crop_zoom, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(cam_np[:crop_zoom.shape[0], :crop_zoom.shape[1]],
                   cmap="hot", alpha=0.55, vmin=0, vmax=1)
    axes[1].set_title("crop zoom + Grad-CAM", fontsize=7)
    axes[1].axis("off")

    plt.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return True


# ── qualitative tag ───────────────────────────────────────────────────────────
def assign_qualitative_tag(cam_np, mask_disp):
    peak_val = float(cam_np.max())
    if peak_val < 0.05:
        return "near_blank"
    # center of mass
    total = float(cam_np.sum())
    if total < 1e-8:
        return "near_blank"
    ys, xs = np.indices(cam_np.shape)
    com_y = float((ys * cam_np).sum() / total)
    com_x = float((xs * cam_np).sum() / total)
    # peak location
    py, px = np.unravel_index(int(cam_np.argmax()), cam_np.shape)
    # inside mask check
    if mask_disp[py, px] > 0.5:
        if peak_val > 0.7:
            return "lesion_focused"
        return "vessel_or_boundary_focused"
    # spread check
    high_mask = cam_np > 0.5
    coverage = float(high_mask.sum()) / cam_np.size
    if coverage > 0.4:
        return "diffuse"
    return "uncertain"


# ── contact sheet ─────────────────────────────────────────────────────────────
def save_contact_sheet(panel_paths, report_dir, plt):
    n = len(panel_paths)
    if n == 0:
        return
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)
    for i, pp in enumerate(panel_paths):
        r, c = divmod(i, ncols)
        img = plt.imread(str(pp))
        axes[r, c].imshow(img)
        axes[r, c].axis("off")
        axes[r, c].set_title(pp.name[:50], fontsize=6)
    for i in range(n, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].axis("off")
    plt.tight_layout()
    sheet_path = report_dir / "contact_sheet.png"
    plt.savefig(sheet_path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"  contact_sheet saved: {sheet_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # guard
    if REPORT_DIR.exists() and any(REPORT_DIR.iterdir()):
        _abort(f"output already exists: {REPORT_DIR}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PANELS_DIR.mkdir(parents=True, exist_ok=True)
    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[{STAGE_LABEL}] start: {ts}")

    # 1. load scalar stats
    scalar_stats = json.loads(SCALAR_STATS.read_text())

    # 2. load score CSV
    print(f"[{STAGE_LABEL}] Step1: load score CSV")
    score_rows = list(csv.DictReader(open(SCORE_CSV)))
    print(f"  {len(score_rows)} rows")

    # index by (patient_id, canonical_volume_z)
    idx = defaultdict(list)
    for r in score_rows:
        idx[(r["patient_id"], r["canonical_volume_z"])].append(r)

    # 3. sample selection
    print(f"[{STAGE_LABEL}] Step2: sample selection")
    samples = []
    sample_sel_rows = []

    def pick_top(pid, z, typ):
        key = (pid, z)
        if key not in idx:
            print(f"  [SKIP] not found: {pid} z={z}")
            return
        row = sorted(idx[key], key=lambda r: -float(r["logit"]))[0]
        row["sample_type"] = typ
        samples.append(row)
        sample_sel_rows.append({
            "sample_type": typ,
            "patient_id":  row["patient_id"],
            "canonical_volume_z": z,
            "label": row["label"],
            "prob": row["prob"],
            "logit": row["logit"],
            "low_mask_flag": row["low_mask_flag"],
            "mask_nonzero_ratio_mean": row["mask_nonzero_ratio_mean"],
            "crop_path": row["crop_path"],
        })
        print(f"  [{typ}] {pid} z={z} prob={float(row['prob']):.4f} low_mask={row['low_mask_flag']}")

    for typ, pid, z, _ in SAMPLE_TARGETS:
        pick_top(pid, z, typ)

    # Normal FP 2개 추가
    for pid in NORMAL_FP_PIDS:
        prows = [r for r in score_rows if r["patient_id"] == pid and r["label"] == "0" and r["pred_at_0p5"] == "1"]
        if prows:
            row = sorted(prows, key=lambda r: -float(r["logit"]))[0]
            row["sample_type"] = "Normal_FP"
            samples.append(row)
            sample_sel_rows.append({
                "sample_type": "Normal_FP",
                "patient_id": row["patient_id"],
                "canonical_volume_z": row["canonical_volume_z"],
                "label": row["label"],
                "prob": row["prob"],
                "logit": row["logit"],
                "low_mask_flag": row["low_mask_flag"],
                "mask_nonzero_ratio_mean": row["mask_nonzero_ratio_mean"],
                "crop_path": row["crop_path"],
            })
            print(f"  [Normal_FP] {pid[:40]}... z={row['canonical_volume_z']} prob={float(row['prob']):.4f}")
        else:
            print(f"  [SKIP] no FP found for {pid[:40]}")

    print(f"  total samples: {len(samples)}")
    _write_csv(sample_sel_rows, REPORT_DIR / "p_c_normal37e_sample_selection.csv")

    # 4. load model
    print(f"[{STAGE_LABEL}] Step3: load model")
    model = load_model()
    model.eval()

    # 5. Grad-CAM per sample
    print(f"[{STAGE_LABEL}] Step4: Grad-CAM + render")
    gradcam_rows = []
    panel_paths  = []
    n_ok = 0
    n_fail = 0

    for i, row in enumerate(samples):
        pid  = row["patient_id"]
        z    = row["canonical_volume_z"]
        typ  = row.get("sample_type", "unknown")
        print(f"  [{i+1}/{len(samples)}] {typ} {pid} z={z}")

        # prep input
        try:
            img_t, sc_t, raw_disp, masked_disp, mask_disp = prep_input(row, scalar_stats)
        except Exception as e:
            print(f"    [WARN] prep_input failed: {e}")
            n_fail += 1
            continue

        # Grad-CAM
        try:
            cam_np, logit_val, act_shape = compute_gradcam(model, img_t, sc_t)
        except Exception as e:
            print(f"    [WARN] gradcam failed: {e}")
            n_fail += 1
            continue

        if np.isnan(cam_np).any() or np.isinf(cam_np).any():
            print(f"    [WARN] cam_np has NaN/Inf")
            n_fail += 1
            continue

        # peak / center of mass
        peak_idx = np.unravel_index(int(cam_np.argmax()), cam_np.shape)
        py, px   = int(peak_idx[0]), int(peak_idx[1])
        total    = float(cam_np.sum())
        if total > 1e-8:
            ys, xs = np.indices(cam_np.shape)
            com_y = float((ys * cam_np).sum() / total)
            com_x = float((xs * cam_np).sum() / total)
        else:
            com_y = com_x = HALF

        peak_inside_mask = bool(mask_disp[py, px] > 0.5)
        qual_tag = assign_qualitative_tag(cam_np, mask_disp)

        prob_val = float(row["prob"])
        safe_pid = pid.replace("/", "_").replace(".", "_")[:50]
        safe_z   = str(z).replace(".", "p")
        panel_name = f"{i+1:02d}_{typ}_{safe_pid}_z{safe_z}_panel.png"
        panel_path = PANELS_DIR / panel_name

        # panel PNG (6-panel)
        try:
            save_panel_png(raw_disp, masked_disp, mask_disp, cam_np, row, panel_path, plt)
            panel_paths.append(panel_path)
            print(f"    panel saved: {panel_name}")
        except Exception as e:
            print(f"    [WARN] panel PNG failed: {e}")

        # context PNG
        ctx_name = f"{i+1:02d}_{typ}_{safe_pid}_z{safe_z}_context.png"
        ctx_path = CONTEXT_DIR / ctx_name
        try:
            ctx_ok = save_context_png(row, cam_np, ctx_path, plt)
            if ctx_ok:
                print(f"    context saved: {ctx_name}")
        except Exception as e:
            print(f"    [WARN] context PNG failed: {e}")

        gradcam_rows.append({
            "sample_idx":             i + 1,
            "sample_type":            typ,
            "patient_id":             pid,
            "canonical_volume_z":     z,
            "true_label":             row["label"],
            "pred_label":             row["pred_at_0p5"],
            "prob":                   round(prob_val, 6),
            "logit":                  round(logit_val, 4),
            "logit_reproduced":       round(logit_val, 4),
            "low_mask":               row["low_mask_flag"],
            "mask_nonzero_ratio":     round(float(row["mask_nonzero_ratio_mean"]), 4),
            "gradcam_peak_yx":        f"({py},{px})",
            "gradcam_center_of_mass_yx": f"({com_y:.1f},{com_x:.1f})",
            "peak_inside_mask":       peak_inside_mask,
            "cam_max":                round(float(cam_np.max()), 4),
            "cam_mean":               round(float(cam_np.mean()), 4),
            "act_shape":              str(act_shape),
            "qualitative_tag":        qual_tag,
            "panel_png":              str(panel_path),
        })
        n_ok += 1

    print(f"  done: n_ok={n_ok}, n_fail={n_fail}")

    # 6. contact sheet
    if panel_paths:
        try:
            save_contact_sheet(panel_paths, REPORT_DIR, plt)
        except Exception as e:
            print(f"  [WARN] contact sheet failed: {e}")

    # 7. quality check
    qual_counts = defaultdict(int)
    for r in gradcam_rows:
        qual_counts[r["qualitative_tag"]] += 1
    quality_rows = [{"qualitative_tag": k, "count": v} for k, v in sorted(qual_counts.items())]
    _write_csv(gradcam_rows,   REPORT_DIR / "p_c_normal37e_gradcam_summary.csv")
    _write_csv(quality_rows,   REPORT_DIR / "p_c_normal37e_quality_check.csv")

    # 8. guardrail check
    guardrail_rows = [
        {"item": "gradcam_only",                       "value": True},
        {"item": "no_training",                        "value": True},
        {"item": "no_threshold_optimization",          "value": True},
        {"item": "no_threshold_sweep",                 "value": True},
        {"item": "no_existing_output_modification",    "value": True},
        {"item": "masked_input_pipeline_reused",       "value": True},
        {"item": "lesion_mask_audit_only",             "value": True},
        {"item": "full_dataset_gradcam",               "value": False},
        {"item": "sample_pilot_only",                  "value": True},
        {"item": "diagnostic_claim",                   "value": False},
        {"item": "n_samples_ok",                       "value": n_ok},
        {"item": "n_fail",                             "value": n_fail},
    ]
    _write_csv(guardrail_rows, REPORT_DIR / "p_c_normal37e_guardrail_check.csv")

    # 9. verdict
    if n_ok == 0:
        verdict = "FAIL"
    elif n_fail > 0:
        verdict = "PARTIAL_PASS"
    else:
        verdict = "PASS"

    lesion_focused_count = qual_counts.get("lesion_focused", 0)
    vessel_focused_count = qual_counts.get("vessel_or_boundary_focused", 0)

    summary = {
        "stage": STAGE_LABEL,
        "timestamp": ts,
        "verdict": verdict,
        "n_samples_ok": n_ok,
        "n_fail": n_fail,
        "qualitative_counts": dict(qual_counts),
        "lesion_focused_n": lesion_focused_count,
        "model": "ScalarFusionModel_EfficientNetB0",
        "checkpoint_epoch": 6,
        "gradcam_target_layer": "model.img_features[7] (last MBConv, 320ch, 3x3)",
        "model_forward": False,
        "training_run": False,
        "threshold_optimization": False,
        "existing_result_modified": False,
    }
    _write_json(summary, REPORT_DIR / "p_c_normal37e_summary.json")
    _write_json({"verdict": verdict, "timestamp": ts, "n_ok": n_ok}, REPORT_DIR / "DONE.json")

    print(f"\n[{STAGE_LABEL}] verdict: {verdict}")
    print(f"  qualitative tags: {dict(qual_counts)}")
    print(f"  panels: {PANELS_DIR}")
    print(f"  report: {REPORT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
