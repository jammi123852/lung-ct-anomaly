"""
Dual-stage overlay review visualization script
Phase 5.43d - P1/P2 candidates (12건) dual-stage visualization

Grad-CAM 제한 이유:
1. 1차 PaDiM: feature-mean Mahalanobis distance 기반 scoring
   - 분류형 backbone이 아니므로 표준 Grad-CAM 적용 불가
   - 대안: patch scan region bbox + score text + bbox+score text approach
   - 한계: 개별 32x32 patch score map은 v2에만 있어 재사용 금지 (v2 score reuse forbidden)
2. 2차 RD4AD (minimal reconstruction baseline):
   - 분류 logit 없어 표준 Grad-CAM 적용 불가
   - 대안: |recon - input| error map + gradient saliency of l1_mean w.r.t. input
   - saliency = grad(mean(|recon(x) - x|)) / dx  (구현 가능)

Forbidden:
- stage2_holdout, v2 경로 접근 금지
- v2 score/eval/metrics 재사용 금지
- 기존 CSV/JSON/PNG/checkpoint 덮어쓰기 금지
- 학습/scoring 재실행 금지
- output 경로 이외의 기존 결과 수정 금지
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────

FORBIDDEN_PATH_PATTERNS = ["stage2_holdout", "v2"]
VOLUME_SOURCE_ROOT = Path(
    "/mnt/c/Users/jinhy/Desktop/"
    "NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1"
    "/volumes_npy"
)

PROJECT_ROOT = Path(".")

# Target crop IDs (from current_review_2.md)
P1_CROP_IDS = [6454, 6600, 6751, 6763, 6800, 7005, 7006, 7256, 7352]
P2_CROP_IDS = [3065, 6567, 7018]
ALL_TARGET_CROP_IDS = P1_CROP_IDS + P2_CROP_IDS
DRYRUN_CROP_IDS = [6454, 6600]

# Window settings
LUNG_WINDOW = (-1350.0, 150.0)
MED_WINDOW = (-160.0, 240.0)
CHANNEL_NAMES = ["lung_z-1", "lung_z", "lung_z+1", "med_z-1", "med_z", "med_z+1"]

# Input sources (read-only)
OVERLAP_ANALYSIS_CSV = (
    "outputs/second-stage-lesion-refiner-v1/evaluation/"
    "rd4ad_2p5d_normal_mw_fixed96_v1/hard_negative_top_score_qa_v1/"
    "hard_negative_top_score_lesion_overlap_analysis_v1.csv"
)
QA_MANIFEST_CSV = (
    "outputs/second-stage-lesion-refiner-v1/evaluation/"
    "rd4ad_2p5d_normal_mw_fixed96_v1/hard_negative_top_score_qa_v1/"
    "hard_negative_top_score_qa_manifest_v1.csv"
)
CHECKPOINT_PATH = (
    "outputs/second-stage-lesion-refiner-v1/models/"
    "rd4ad_2p5d_normal_mw_fixed96_v1/checkpoints/best_val_loss.pt"
)
NORMAL_SCORE_SUMMARY = (
    "outputs/second-stage-lesion-refiner-v1/evaluation/"
    "rd4ad_2p5d_normal_mw_fixed96_v1/normal_val_test_scores_v1/"
    "normal_val_test_score_summary_v1.json"
)

# Output root (new, separate from all existing outputs)
OUTPUT_VIZ_ROOT = (
    "outputs/second-stage-lesion-refiner-v1/visualizations/"
    "p1_p2_dual_stage_overlay_review_v1"
)

# Fallback thresholds (from normal_val_test_score_summary_v1.json)
FALLBACK_VAL_P90 = 0.07003
FALLBACK_VAL_P95 = 0.08635
FALLBACK_VAL_P99 = 0.11331

# Heatmap proxy limitation note
HEATMAP_PROXY_LIMITATION = (
    "1차 PaDiM pixel-level heatmap: v1 score CSV에 target patient 없음, "
    "v2 score 재사용 금지. bbox+score text만 표시."
)


# ──────────────────────────────────────────────────────────
# Forbidden path guard
# ──────────────────────────────────────────────────────────

def check_forbidden(path_str: str, label: str = "") -> None:
    for pattern in FORBIDDEN_PATH_PATTERNS:
        if pattern in str(path_str):
            print(f"[ERROR] Forbidden path pattern '{pattern}' found in {label}: {path_str}")
            sys.exit(1)


def check_volume_source_guard(path_str: str) -> None:
    """Volume source must be NSCLC_MSD_padim_test_ready_roi0_0 only."""
    p = str(path_str)
    # Allow only VOLUME_SOURCE_ROOT directory for volume/mask reads
    vol_root_str = str(VOLUME_SOURCE_ROOT)
    # Also allow the crop NPZ paths (hard negative crops)
    crop_root_allowed = "rd4ad_train_2p5d_mw_fixed96_thr001"
    if vol_root_str not in p and crop_root_allowed not in p:
        if "ct_hu.npy" in p or "lesion_mask" in p or "roi_0_0.npy" in p:
            print(f"[ERROR] Volume read from forbidden path: {p}")
            print(f"        Only allowed: {vol_root_str}")
            sys.exit(1)


# ──────────────────────────────────────────────────────────
# Model definition (identical to train_rd4ad_2p5d_normal.py)
# ──────────────────────────────────────────────────────────

def build_model():
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("[ERROR] torch not available.")
        sys.exit(1)

    class ConvAutoencoder2p5D(nn.Module):
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

    return ConvAutoencoder2p5D


# ──────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────

def apply_window(hu, hu_min, hu_max):
    clipped = np.clip(hu.astype(np.float32), hu_min, hu_max)
    return ((clipped - hu_min) / (hu_max - hu_min) * 255.0).astype(np.uint8)


def extract_context_patch(arr_2d, y_start, y_end, x_start, x_end, expected_size=192):
    """Extract context region from a 2D array. Pad with zeros if out of bounds."""
    H, W = arr_2d.shape[:2]
    h = y_end - y_start
    w = x_end - x_start
    if arr_2d.ndim == 2:
        patch = np.zeros((h, w), dtype=arr_2d.dtype)
    else:
        patch = np.zeros((h, w, arr_2d.shape[2]), dtype=arr_2d.dtype)
    y0c = max(0, y_start)
    y1c = min(H, y_end)
    x0c = max(0, x_start)
    x1c = min(W, x_end)
    if y1c > y0c and x1c > x0c:
        dst_y0 = y0c - y_start
        dst_y1 = dst_y0 + (y1c - y0c)
        dst_x0 = x0c - x_start
        dst_x1 = dst_x0 + (x1c - x0c)
        patch[dst_y0:dst_y1, dst_x0:dst_x1] = arr_2d[y0c:y1c, x0c:x1c]
    return patch


def mask_to_contour_overlay(mask_2d, color_rgb, alpha=0.6):
    """Convert binary mask to RGBA overlay for contour visualization."""
    from skimage import measure  # optional, fall back to edge if missing
    result = np.zeros((*mask_2d.shape, 4), dtype=np.uint8)
    if mask_2d.sum() == 0:
        return result
    contours = measure.find_contours(mask_2d.astype(float), 0.5)
    for contour in contours:
        for r, c in contour.astype(int):
            if 0 <= r < mask_2d.shape[0] and 0 <= c < mask_2d.shape[1]:
                result[r, c, 0] = color_rgb[0]
                result[r, c, 1] = color_rgb[1]
                result[r, c, 2] = color_rgb[2]
                result[r, c, 3] = int(255 * alpha)
    return result


def load_thresholds(summary_path: str):
    p = Path(summary_path)
    if p.exists():
        check_forbidden(str(p), "normal_score_summary")
        with open(p) as f:
            data = json.load(f)
        return {
            "val_p90": data.get("val_score_p90", FALLBACK_VAL_P90),
            "val_p95": data.get("val_score_p95", FALLBACK_VAL_P95),
            "val_p99": data.get("val_score_p99", FALLBACK_VAL_P99),
        }
    return {"val_p90": FALLBACK_VAL_P90, "val_p95": FALLBACK_VAL_P95, "val_p99": FALLBACK_VAL_P99}


# ──────────────────────────────────────────────────────────
# Model forward (reconstruction + saliency)
# ──────────────────────────────────────────────────────────

def run_model_forward(model, crop_img: np.ndarray, device, compute_saliency: bool = True):
    """
    Forward pass through ConvAutoencoder2p5D.
    Returns dict: recon (6,96,96), error_map (6,96,96), saliency (96,96) or None.
    crop_img: float32 array [6, 96, 96] normalized [0,1].
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    crop_tensor = torch.from_numpy(crop_img[np.newaxis]).float().to(device)  # [1,6,96,96]

    if compute_saliency:
        crop_tensor_grad = crop_tensor.clone().requires_grad_(True)
        recon = model(crop_tensor_grad)
        l1_loss = F.l1_loss(recon, crop_tensor_grad)
        l1_loss.backward()
        if crop_tensor_grad.grad is not None:
            saliency = crop_tensor_grad.grad.squeeze(0).abs().cpu().numpy()  # [6,96,96]
            saliency_map = saliency.mean(axis=0)  # [96,96]
            saliency_max = saliency_map.max()
            if saliency_max > 0:
                saliency_map = saliency_map / saliency_max
        else:
            saliency_map = None
        recon_np = recon.detach().squeeze(0).cpu().numpy()  # [6,96,96]
    else:
        with torch.no_grad():
            recon = model(crop_tensor)
        recon_np = recon.squeeze(0).cpu().numpy()
        saliency_map = None

    error_map = np.abs(recon_np - crop_img)  # [6,96,96]
    crop_score_l1 = float(error_map.mean())

    return {
        "recon": recon_np,
        "error_map": error_map,
        "saliency_map": saliency_map,
        "crop_score_l1_mean_computed": crop_score_l1,
    }


# ──────────────────────────────────────────────────────────
# Stage 1 overlay panel
# ──────────────────────────────────────────────────────────

def render_stage1_overlay(
    ct_hu: np.ndarray,          # (Z, H, W) int16
    lesion_mask: np.ndarray,    # (Z, H, W)
    roi_mask: np.ndarray,       # (Z, H, W)
    row: dict,                  # merged row from overlap_analysis + qa_manifest
    thresholds: dict,
    save_path: Path,
    dry_run: bool = False,
) -> dict:
    """
    Render context 192x192 overlay with:
    - lung window (left) + mediastinal window (right)
    - roi_0_0 contour (green)
    - lesion contour (red)
    - patch scan region bbox (yellow dashed)
    - fixed96 crop bbox (blue solid)
    Returns info dict.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    z = int(row["z_center"])
    y0_ctx = int(row["y_context_start"])
    y1_ctx = int(row["y_context_end"])
    x0_ctx = int(row["x_context_start"])
    x1_ctx = int(row["x_context_end"])
    y0_patch = int(row["y0_patch"])
    x0_patch = int(row["x0_patch"])
    y1_patch = int(row["y1_patch"])
    x1_patch = int(row["x1_patch"])
    y0_96 = int(row["y0_fixed96"])
    x0_96 = int(row["x0_fixed96"])
    y1_96 = int(row["y1_fixed96"])
    x1_96 = int(row["x1_fixed96"])

    ctx_h = y1_ctx - y0_ctx
    ctx_w = x1_ctx - x0_ctx

    slice_hu = ct_hu[z]
    lung_slice = apply_window(slice_hu, *LUNG_WINDOW)
    med_slice = apply_window(slice_hu, *MED_WINDOW)

    lung_ctx = extract_context_patch(lung_slice, y0_ctx, y1_ctx, x0_ctx, x1_ctx)
    med_ctx = extract_context_patch(med_slice, y0_ctx, y1_ctx, x0_ctx, x1_ctx)

    lesion_ctx = extract_context_patch(lesion_mask[z], y0_ctx, y1_ctx, x0_ctx, x1_ctx)
    roi_ctx = extract_context_patch(roi_mask[z], y0_ctx, y1_ctx, x0_ctx, x1_ctx)

    # Coordinate offsets (absolute → relative to context)
    def to_ctx(y, x):
        return y - y0_ctx, x - x0_ctx

    patch_rel_y0, patch_rel_x0 = to_ctx(y0_patch, x0_patch)
    patch_rel_y1, patch_rel_x1 = to_ctx(y1_patch, x1_patch)
    crop96_rel_y0, crop96_rel_x0 = to_ctx(y0_96, x0_96)
    crop96_rel_y1, crop96_rel_x1 = to_ctx(y1_96, x1_96)

    crop_id = int(row["crop_id"])
    patient_id = str(row["patient_id"])
    qa_group = str(row.get("qa_group", ""))
    qa_priority = str(row.get("qa_priority_x", row.get("qa_priority", "")))
    padim_mean = float(row.get("padim_score_mean", 0.0))
    crop_score = float(row.get("crop_score_l1_mean", 0.0))
    thr_p90 = bool(row.get("threshold_exceed_val_p90", False))
    thr_p95 = bool(row.get("threshold_exceed_val_p95", False))
    overlap_class = str(row.get("lesion_overlap_class", ""))
    lesion_pixels_patch = int(row.get("lesion_pixels_patch", 0))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))
    fig.patch.set_facecolor("#1a1a1a")

    windows = [("Lung", lung_ctx), ("Mediastinal", med_ctx)]
    for ax, (win_name, img) in zip(axes, windows):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, origin="upper")
        ax.set_facecolor("#1a1a1a")

        # ROI contour (green)
        if roi_ctx.sum() > 0:
            ax.contour(roi_ctx, levels=[0.5], colors=["#00ff44"], linewidths=1.0, alpha=0.8)

        # Lesion contour (red)
        if lesion_ctx.sum() > 0:
            ax.contour(lesion_ctx, levels=[0.5], colors=["#ff3333"], linewidths=1.5, alpha=0.9)

        # Patch scan region bbox (yellow dashed)
        patch_rect = mpatches.Rectangle(
            (patch_rel_x0, patch_rel_y0),
            patch_rel_x1 - patch_rel_x0,
            patch_rel_y1 - patch_rel_y0,
            linewidth=1.2,
            edgecolor="yellow",
            facecolor="none",
            linestyle="--",
            alpha=0.85,
            label="Stage1 scan region",
        )
        ax.add_patch(patch_rect)

        # Fixed96 crop bbox (blue solid)
        crop96_rect = mpatches.Rectangle(
            (crop96_rel_x0, crop96_rel_y0),
            crop96_rel_x1 - crop96_rel_x0,
            crop96_rel_y1 - crop96_rel_y0,
            linewidth=2.0,
            edgecolor="#4499ff",
            facecolor="none",
            linestyle="-",
            alpha=0.9,
            label="Stage2 fixed96",
        )
        ax.add_patch(crop96_rect)

        ax.set_title(f"{win_name}", fontsize=9, color="white", pad=2)
        ax.set_xlim(0, ctx_w)
        ax.set_ylim(ctx_h, 0)
        ax.axis("off")

    # Legend
    handles = [
        mpatches.Patch(color="yellow", label="Stage1 scan region (bbox)", linestyle="--"),
        mpatches.Patch(color="#4499ff", label="Stage2 fixed96 crop"),
        mpatches.Patch(color="#00ff44", label="roi_0_0 contour"),
        mpatches.Patch(color="#ff3333", label="Lesion contour"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        fontsize=7,
        facecolor="#2a2a2a",
        labelcolor="white",
        framealpha=0.8,
        bbox_to_anchor=(0.5, -0.01),
    )

    # Title / annotation
    thr_str = f"p90:{int(thr_p90)} p95:{int(thr_p95)}"
    overlap_color = "#ff6666" if "patch_overlap" in overlap_class else "#ffaa33"
    title_str = (
        f"crop_id={crop_id}  patient={patient_id}  z={z}\n"
        f"qa_group={qa_group}  priority={qa_priority}\n"
        f"padim_mean={padim_mean:.2f}  rd4ad_l1={crop_score:.4f}  thr_exceed={thr_str}\n"
        f"overlap={overlap_class}  lesion_px_patch={lesion_pixels_patch}"
    )
    fig.suptitle(
        title_str,
        fontsize=8,
        color="white",
        y=1.01,
        ha="center",
    )

    note = HEATMAP_PROXY_LIMITATION
    fig.text(
        0.5, -0.06, f"[NOTE] {note}",
        ha="center", fontsize=6, color="#aaaaaa",
        wrap=True,
    )

    plt.tight_layout(pad=0.5)

    if not dry_run:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.exists():
            print(f"[ERROR] Output file already exists (overwrite guard): {save_path}")
            sys.exit(1)
        fig.savefig(str(save_path), dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())

    plt.close(fig)
    return {
        "crop_id": crop_id,
        "saved_path": str(save_path) if not dry_run else None,
        "window_size": f"{ctx_h}x{ctx_w}",
        "has_lesion": bool(lesion_ctx.sum() > 0),
        "has_roi": bool(roi_ctx.sum() > 0),
    }


# ──────────────────────────────────────────────────────────
# Stage 2 explanation panel
# ──────────────────────────────────────────────────────────

def render_stage2_explanation(
    model_result: dict,         # from run_model_forward()
    crop_img: np.ndarray,       # [6, 96, 96] float32
    row: dict,
    thresholds: dict,
    save_path: Path,
    dry_run: bool = False,
) -> dict:
    """
    Render stage2 explanation panel:
    Row 1 (lung): input_lung | recon_lung | error_lung
    Row 2 (med): input_med | recon_med | error_med
    Row 3: saliency_map | score text
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    recon = model_result["recon"]        # [6,96,96]
    error_map = model_result["error_map"]  # [6,96,96]
    saliency = model_result.get("saliency_map")  # [96,96] or None
    computed_score = model_result["crop_score_l1_mean_computed"]

    # Channel indices: lung_z (ch1=index 1), med_z (ch4=index 4)
    lung_idx = 1
    med_idx = 4

    crop_id = int(row["crop_id"])
    padim_mean = float(row.get("padim_score_mean", 0.0))
    stored_score = float(row.get("crop_score_l1_mean", 0.0))
    thr_p90 = bool(row.get("threshold_exceed_val_p90", False))
    thr_p95 = bool(row.get("threshold_exceed_val_p95", False))
    qa_group = str(row.get("qa_group", ""))
    patient_id = str(row["patient_id"])

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.patch.set_facecolor("#1a1a1a")
    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor("#1a1a1a")
            ax.axis("off")

    def show_img(ax, img_2d, title, cmap="gray", vmin=0, vmax=1):
        ax.imshow(img_2d, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        ax.set_title(title, fontsize=7, color="white", pad=2)
        ax.axis("off")

    # Row 0: lung channel
    show_img(axes[0][0], crop_img[lung_idx], "Input (lung_z)")
    show_img(axes[0][1], recon[lung_idx], "Recon (lung_z)")
    err_l = error_map[lung_idx]
    show_img(axes[0][2], err_l, f"|Error| lung max={err_l.max():.3f}", cmap="hot", vmin=0, vmax=err_l.max() + 1e-8)

    # Row 1: mediastinal channel
    show_img(axes[1][0], crop_img[med_idx], "Input (med_z)")
    show_img(axes[1][1], recon[med_idx], "Recon (med_z)")
    err_m = error_map[med_idx]
    show_img(axes[1][2], err_m, f"|Error| med max={err_m.max():.3f}", cmap="hot", vmin=0, vmax=err_m.max() + 1e-8)

    # Row 2: saliency + score text
    if saliency is not None:
        show_img(axes[2][0], saliency, "Saliency (grad|dl1/dx|)", cmap="inferno", vmin=0, vmax=1)
        saliency_note = "grad(l1_mean)/d_input, abs, mean over channels, normalized"
    else:
        axes[2][0].text(
            0.5, 0.5, "Saliency\nN/A",
            ha="center", va="center", transform=axes[2][0].transAxes,
            color="gray", fontsize=8,
        )
        saliency_note = "Saliency computation failed"

    # Score text panel
    val_p90 = thresholds.get("val_p90", FALLBACK_VAL_P90)
    val_p95 = thresholds.get("val_p95", FALLBACK_VAL_P95)
    score_text = (
        f"crop_id: {crop_id}\n"
        f"patient: {patient_id}\n"
        f"qa_group: {qa_group}\n\n"
        f"crop_score_l1_mean (stored): {stored_score:.5f}\n"
        f"crop_score_l1_mean (computed): {computed_score:.5f}\n\n"
        f"val_p90: {val_p90:.5f}  exceed={int(thr_p90)}\n"
        f"val_p95: {val_p95:.5f}  exceed={int(thr_p95)}\n\n"
        f"padim_score_mean: {padim_mean:.3f}\n\n"
        f"Saliency: {saliency_note}"
    )
    axes[2][1].text(
        0.05, 0.95, score_text,
        ha="left", va="top", transform=axes[2][1].transAxes,
        color="white", fontsize=7, fontfamily="monospace",
        wrap=True,
    )
    axes[2][1].set_xlim(0, 1)
    axes[2][1].set_ylim(0, 1)

    # Hide unused 3rd cell in row 2
    axes[2][2].set_visible(False)

    # Column labels
    for ax, label in zip(axes[0], ["Input", "Reconstruction", "|Error| (L1)"]):
        ax.set_title(label, fontsize=8, color="white", pad=2)

    fig.suptitle(
        f"Stage2 Explanation — crop_id={crop_id}  patient={patient_id}",
        fontsize=9, color="white", y=1.01,
    )
    plt.tight_layout(pad=0.3)

    if not dry_run:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.exists():
            print(f"[ERROR] Output file already exists (overwrite guard): {save_path}")
            sys.exit(1)
        fig.savefig(str(save_path), dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())

    plt.close(fig)
    return {
        "crop_id": crop_id,
        "saved_path": str(save_path) if not dry_run else None,
        "computed_l1": computed_score,
        "stored_l1": stored_score,
        "score_match": abs(computed_score - stored_score) < 0.002,
    }


# ──────────────────────────────────────────────────────────
# Combined panel
# ──────────────────────────────────────────────────────────

def render_combined_panel(
    ct_hu: np.ndarray,
    lesion_mask: np.ndarray,
    roi_mask: np.ndarray,
    model_result: dict,
    crop_img: np.ndarray,
    row: dict,
    thresholds: dict,
    save_path: Path,
    dry_run: bool = False,
) -> dict:
    """
    Combined panel: stage1 context overlay (lung) | stage2 explanation
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec

    z = int(row["z_center"])
    y0_ctx = int(row["y_context_start"])
    y1_ctx = int(row["y_context_end"])
    x0_ctx = int(row["x_context_start"])
    x1_ctx = int(row["x_context_end"])
    y0_patch = int(row["y0_patch"])
    x0_patch = int(row["x0_patch"])
    y1_patch = int(row["y1_patch"])
    x1_patch = int(row["x1_patch"])
    y0_96 = int(row["y0_fixed96"])
    x0_96 = int(row["x0_fixed96"])
    y1_96 = int(row["y1_fixed96"])
    x1_96 = int(row["x1_fixed96"])
    ctx_h = y1_ctx - y0_ctx
    ctx_w = x1_ctx - x0_ctx

    slice_hu = ct_hu[z]
    lung_ctx = extract_context_patch(apply_window(slice_hu, *LUNG_WINDOW), y0_ctx, y1_ctx, x0_ctx, x1_ctx)
    lesion_ctx = extract_context_patch(lesion_mask[z], y0_ctx, y1_ctx, x0_ctx, x1_ctx)
    roi_ctx = extract_context_patch(roi_mask[z], y0_ctx, y1_ctx, x0_ctx, x1_ctx)

    def to_ctx(y, x):
        return y - y0_ctx, x - x0_ctx

    recon = model_result["recon"]
    error_map = model_result["error_map"]
    saliency = model_result.get("saliency_map")
    lung_idx, med_idx = 1, 4

    crop_id = int(row["crop_id"])
    patient_id = str(row["patient_id"])
    padim_mean = float(row.get("padim_score_mean", 0.0))
    stored_score = float(row.get("crop_score_l1_mean", 0.0))
    thr_p90 = bool(row.get("threshold_exceed_val_p90", False))
    thr_p95 = bool(row.get("threshold_exceed_val_p95", False))
    overlap_class = str(row.get("lesion_overlap_class", ""))
    qa_group = str(row.get("qa_group", ""))

    fig = plt.figure(figsize=(16, 7), facecolor="#1a1a1a")
    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.35, wspace=0.3)

    # Left: Stage1 context overlay (lung, spans 2 rows × 2 cols)
    ax_ctx = fig.add_subplot(gs[:, :2])
    ax_ctx.set_facecolor("#1a1a1a")
    ax_ctx.imshow(lung_ctx, cmap="gray", vmin=0, vmax=255, origin="upper")
    if roi_ctx.sum() > 0:
        ax_ctx.contour(roi_ctx, levels=[0.5], colors=["#00ff44"], linewidths=1.0, alpha=0.8)
    if lesion_ctx.sum() > 0:
        ax_ctx.contour(lesion_ctx, levels=[0.5], colors=["#ff3333"], linewidths=1.5, alpha=0.9)
    pr_y0, pr_x0 = to_ctx(y0_patch, x0_patch)
    pr_y1, pr_x1 = to_ctx(y1_patch, x1_patch)
    ax_ctx.add_patch(mpatches.Rectangle(
        (pr_x0, pr_y0), pr_x1 - pr_x0, pr_y1 - pr_y0,
        linewidth=1.2, edgecolor="yellow", facecolor="none", linestyle="--", alpha=0.85,
    ))
    cr_y0, cr_x0 = to_ctx(y0_96, x0_96)
    cr_y1, cr_x1 = to_ctx(y1_96, x1_96)
    ax_ctx.add_patch(mpatches.Rectangle(
        (cr_x0, cr_y0), cr_x1 - cr_x0, cr_y1 - cr_y0,
        linewidth=2.0, edgecolor="#4499ff", facecolor="none", linestyle="-", alpha=0.9,
    ))
    ax_ctx.set_xlim(0, ctx_w)
    ax_ctx.set_ylim(ctx_h, 0)
    ax_ctx.axis("off")
    ax_ctx.set_title(
        f"Stage1 (Lung)\ncrop_id={crop_id} z={z} padim={padim_mean:.2f} overlap={overlap_class}",
        fontsize=8, color="white", pad=3,
    )

    # Right: Stage2 — row 0
    ax_in_l = fig.add_subplot(gs[0, 2])
    ax_in_l.imshow(crop_img[lung_idx], cmap="gray", vmin=0, vmax=1, origin="upper")
    ax_in_l.set_title("Input (lung_z)", fontsize=7, color="white", pad=2)
    ax_in_l.axis("off")

    ax_rc_l = fig.add_subplot(gs[0, 3])
    ax_rc_l.imshow(recon[lung_idx], cmap="gray", vmin=0, vmax=1, origin="upper")
    ax_rc_l.set_title("Recon (lung_z)", fontsize=7, color="white", pad=2)
    ax_rc_l.axis("off")

    ax_er_l = fig.add_subplot(gs[0, 4])
    err_l = error_map[lung_idx]
    ax_er_l.imshow(err_l, cmap="hot", vmin=0, vmax=max(err_l.max(), 1e-8), origin="upper")
    ax_er_l.set_title(f"|Err| lung\nmax={err_l.max():.3f}", fontsize=7, color="white", pad=2)
    ax_er_l.axis("off")

    # Right: Stage2 — row 1
    ax_in_m = fig.add_subplot(gs[1, 2])
    ax_in_m.imshow(crop_img[med_idx], cmap="gray", vmin=0, vmax=1, origin="upper")
    ax_in_m.set_title("Input (med_z)", fontsize=7, color="white", pad=2)
    ax_in_m.axis("off")

    ax_rc_m = fig.add_subplot(gs[1, 3])
    ax_rc_m.imshow(recon[med_idx], cmap="gray", vmin=0, vmax=1, origin="upper")
    ax_rc_m.set_title("Recon (med_z)", fontsize=7, color="white", pad=2)
    ax_rc_m.axis("off")

    ax_er_m = fig.add_subplot(gs[1, 4])
    if saliency is not None:
        ax_er_m.imshow(saliency, cmap="inferno", vmin=0, vmax=1, origin="upper")
        ax_er_m.set_title("Saliency\n(grad l1/dx)", fontsize=7, color="white", pad=2)
    else:
        err_m = error_map[med_idx]
        ax_er_m.imshow(err_m, cmap="hot", vmin=0, vmax=max(err_m.max(), 1e-8), origin="upper")
        ax_er_m.set_title(f"|Err| med\nmax={err_m.max():.3f}", fontsize=7, color="white", pad=2)
    ax_er_m.axis("off")

    val_p95 = thresholds.get("val_p95", FALLBACK_VAL_P95)
    fig.suptitle(
        f"Combined Review — crop_id={crop_id}  patient={patient_id}\n"
        f"qa_group={qa_group}  rd4ad_l1={stored_score:.4f}  val_p95={val_p95:.4f}  "
        f"p90={int(thr_p90)} p95={int(thr_p95)}",
        fontsize=9, color="white", y=1.01,
    )

    if not dry_run:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.exists():
            print(f"[ERROR] Output file already exists (overwrite guard): {save_path}")
            sys.exit(1)
        fig.savefig(str(save_path), dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())

    plt.close(fig)
    return {"crop_id": crop_id, "saved_path": str(save_path) if not dry_run else None}


# ──────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────

def load_input_dataframes(project_root: Path):
    """Load overlap analysis and QA manifest CSVs. Return merged dict keyed by crop_id."""
    import pandas as pd

    overlap_path = project_root / OVERLAP_ANALYSIS_CSV
    qa_path = project_root / QA_MANIFEST_CSV

    check_forbidden(str(overlap_path), "overlap_analysis_csv")
    check_forbidden(str(qa_path), "qa_manifest_csv")

    if not overlap_path.exists():
        print(f"[ERROR] Overlap analysis CSV not found: {overlap_path}")
        sys.exit(1)
    if not qa_path.exists():
        print(f"[ERROR] QA manifest CSV not found: {qa_path}")
        sys.exit(1)

    df_overlap = pd.read_csv(overlap_path)
    df_qa = pd.read_csv(qa_path)

    # Merge on crop_id
    df = df_overlap.merge(df_qa[["crop_id", "crop_path", "crop_score_l1_mean",
                                  "padim_score_mean", "padim_score_max",
                                  "threshold_exceed_val_p90", "threshold_exceed_val_p95",
                                  "qa_priority"]], on="crop_id", how="left", suffixes=("", "_qa"))
    return df


def get_volume_dir(safe_id: str) -> Path:
    """Return volume directory path for a given safe_id."""
    vol_dir = VOLUME_SOURCE_ROOT / safe_id
    check_volume_source_guard(str(vol_dir))
    check_forbidden(str(vol_dir), f"volume_dir for {safe_id}")
    return vol_dir


def load_volume(vol_dir: Path):
    """Load ct_hu, lesion_mask, roi_0_0 arrays."""
    ct_path = vol_dir / "ct_hu.npy"
    lesion_path = vol_dir / "lesion_mask_roi_0_0.npy"
    roi_path = vol_dir / "roi_0_0.npy"

    check_volume_source_guard(str(ct_path))
    check_volume_source_guard(str(lesion_path))
    check_volume_source_guard(str(roi_path))

    if not ct_path.exists():
        raise FileNotFoundError(f"ct_hu.npy not found: {ct_path}")
    if not lesion_path.exists():
        raise FileNotFoundError(f"lesion_mask_roi_0_0.npy not found: {lesion_path}")
    if not roi_path.exists():
        raise FileNotFoundError(f"roi_0_0.npy not found: {roi_path}")

    ct_hu = np.load(str(ct_path), mmap_mode="r")
    lesion_mask = np.load(str(lesion_path), mmap_mode="r")
    roi_mask = np.load(str(roi_path), mmap_mode="r")
    return ct_hu, lesion_mask, roi_mask


def load_crop_npz(crop_path_str: str, project_root: Path) -> np.ndarray:
    """Load crop image from NPZ file. Returns float32 [6, 96, 96]."""
    p = Path(crop_path_str)
    if not p.is_absolute():
        p = project_root / p
    check_forbidden(str(p), "crop_npz")
    if not p.exists():
        raise FileNotFoundError(f"Crop NPZ not found: {p}")
    data = np.load(str(p))
    return data["image"].astype(np.float32)


def load_model(project_root: Path, device):
    """Load ConvAutoencoder2p5D from checkpoint."""
    import torch

    ckpt_path = project_root / CHECKPOINT_PATH
    check_forbidden(str(ckpt_path), "checkpoint")
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    ModelClass = build_model()
    model = ModelClass(input_channels=6)
    state = torch.load(str(ckpt_path), map_location=device)
    # state may be full checkpoint dict or just state_dict
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"[INFO] Model loaded from {ckpt_path}")
    return model


# ──────────────────────────────────────────────────────────
# Preflight checks
# ──────────────────────────────────────────────────────────

def run_preflight(project_root: Path, crop_ids: list) -> dict:
    """Check all inputs without running model forward or generating PNG."""
    import pandas as pd

    print("\n" + "=" * 60)
    print("PREFLIGHT CHECK")
    print("=" * 60)

    issues = []
    warnings = []

    # 1. Input CSV existence
    overlap_path = project_root / OVERLAP_ANALYSIS_CSV
    qa_path = project_root / QA_MANIFEST_CSV
    ckpt_path = project_root / CHECKPOINT_PATH
    score_summary_path = project_root / NORMAL_SCORE_SUMMARY

    for label, p in [
        ("overlap_analysis_csv", overlap_path),
        ("qa_manifest_csv", qa_path),
        ("checkpoint", ckpt_path),
        ("normal_score_summary", score_summary_path),
    ]:
        check_forbidden(str(p), label)
        if p.exists():
            print(f"  [OK] {label}: {p}")
        else:
            print(f"  [MISSING] {label}: {p}")
            issues.append(f"Missing: {label}")

    # 2. Volume source root
    if VOLUME_SOURCE_ROOT.exists():
        print(f"  [OK] volume_source_root: {VOLUME_SOURCE_ROOT}")
    else:
        print(f"  [MISSING] volume_source_root: {VOLUME_SOURCE_ROOT}")
        issues.append("Volume source root not accessible")

    # 3. Load dataframes and check target crop IDs
    if overlap_path.exists() and qa_path.exists():
        df_overlap = pd.read_csv(overlap_path)
        df_qa = pd.read_csv(qa_path)

        found_overlap = set(df_overlap["crop_id"].tolist())
        found_qa = set(df_qa["crop_id"].tolist())
        for cid in crop_ids:
            ok_ov = cid in found_overlap
            ok_qa = cid in found_qa
            status = "OK" if (ok_ov and ok_qa) else "PARTIAL" if (ok_ov or ok_qa) else "MISSING"
            print(f"  [{status}] crop_id={cid}  overlap={ok_ov}  qa={ok_qa}")
            if not (ok_ov and ok_qa):
                issues.append(f"crop_id={cid} not found in both CSVs")

    # 4. Crop NPZ existence per crop_id
    if qa_path.exists():
        df_qa = pd.read_csv(qa_path)
        df_target = df_qa[df_qa["crop_id"].isin(crop_ids)]
        for _, r in df_target.iterrows():
            crop_path_str = str(r["crop_path"])
            p = Path(crop_path_str)
            if not p.is_absolute():
                p = project_root / p
            check_forbidden(str(p), f"crop_npz crop_id={r['crop_id']}")
            if p.exists():
                print(f"  [OK] crop NPZ crop_id={r['crop_id']}: {p.name}")
            else:
                print(f"  [MISSING] crop NPZ crop_id={r['crop_id']}: {p}")
                issues.append(f"Missing crop NPZ for crop_id={r['crop_id']}")

    # 5. Volume files per safe_id
    if overlap_path.exists():
        df_overlap = pd.read_csv(overlap_path)
        df_target = df_overlap[df_overlap["crop_id"].isin(crop_ids)]
        for _, r in df_target.iterrows():
            safe_id = str(r["safe_id"])
            vol_dir = VOLUME_SOURCE_ROOT / safe_id
            check_volume_source_guard(str(vol_dir))
            check_forbidden(str(vol_dir), f"vol_dir {safe_id}")
            for fname in ["ct_hu.npy", "lesion_mask_roi_0_0.npy", "roi_0_0.npy"]:
                fpath = vol_dir / fname
                if fpath.exists():
                    print(f"  [OK] {safe_id}/{fname}")
                else:
                    print(f"  [MISSING] {safe_id}/{fname}")
                    issues.append(f"Missing volume file: {safe_id}/{fname}")

    # 6. Output directory (should NOT exist yet, or be empty)
    out_root = project_root / OUTPUT_VIZ_ROOT
    if out_root.exists():
        existing = list(out_root.rglob("*.png"))
        if existing:
            print(f"  [WARN] Output dir exists with {len(existing)} PNG files: {out_root}")
            warnings.append(f"Output dir has existing PNGs: {len(existing)}")
        else:
            print(f"  [WARN] Output dir already exists (empty): {out_root}")
            warnings.append("Output dir already exists (no PNGs)")
    else:
        print(f"  [OK] Output dir does not exist (will be created): {out_root}")

    # 7. Forbidden path double check
    forbidden_dirs = [
        project_root / "outputs/second-stage-lesion-refiner-v1/stage2_holdout",
        project_root / "outputs/second-stage-lesion-refiner-v1/evaluation/v2",
    ]
    for d in forbidden_dirs:
        if d.exists():
            print(f"  [WARN] Forbidden dir exists (will NOT be read): {d}")
            warnings.append(f"Forbidden dir exists: {d}")
        else:
            print(f"  [OK] Forbidden dir absent: {d}")

    print("=" * 60)
    if issues:
        print(f"PREFLIGHT: {len(issues)} issue(s) found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("PREFLIGHT: All checks passed.")
    if warnings:
        print(f"PREFLIGHT: {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  - {w}")

    return {"issues": issues, "warnings": warnings, "passed": len(issues) == 0}


# ──────────────────────────────────────────────────────────
# Main visualization loop
# ──────────────────────────────────────────────────────────

def run_visualization(
    project_root: Path,
    crop_ids: list,
    device,
    skip_saliency: bool = False,
    dry_run: bool = False,
) -> dict:
    import pandas as pd
    import torch

    out_root = project_root / OUTPUT_VIZ_ROOT
    stage1_dir = out_root / "stage1_overlay"
    stage2_dir = out_root / "stage2_explanation"
    combined_dir = out_root / "combined"

    if not dry_run:
        for d in [stage1_dir, stage2_dir, combined_dir]:
            d.mkdir(parents=True, exist_ok=True)

    thresholds = load_thresholds(str(project_root / NORMAL_SCORE_SUMMARY))
    df = load_input_dataframes(project_root)
    model = load_model(project_root, device)

    results = []
    errors = []

    df_target = df[df["crop_id"].isin(crop_ids)].copy()
    if len(df_target) != len(crop_ids):
        found = set(df_target["crop_id"].tolist())
        missing = [c for c in crop_ids if c not in found]
        print(f"[WARN] Some crop_ids not found in merged dataframe: {missing}")

    for _, row in df_target.iterrows():
        crop_id = int(row["crop_id"])
        patient_id = str(row["patient_id"])
        safe_id = str(row["safe_id"])
        crop_path_str = str(row["crop_path"])

        print(f"\n[INFO] Processing crop_id={crop_id}  patient={patient_id}")

        try:
            # Load volume
            vol_dir = get_volume_dir(safe_id)
            ct_hu, lesion_mask, roi_mask = load_volume(vol_dir)

            # Load crop NPZ
            crop_img = load_crop_npz(crop_path_str, project_root)

            # Run model
            model_result = run_model_forward(model, crop_img, device, compute_saliency=not skip_saliency)
            print(f"  l1_stored={row.get('crop_score_l1_mean', 'N/A'):.5f}  "
                  f"l1_computed={model_result['crop_score_l1_mean_computed']:.5f}")

            # Stage1 overlay
            s1_path = stage1_dir / f"{crop_id:05d}_stage1.png"
            s1_info = render_stage1_overlay(ct_hu, lesion_mask, roi_mask, row.to_dict(),
                                            thresholds, s1_path, dry_run=dry_run)

            # Stage2 explanation
            s2_path = stage2_dir / f"{crop_id:05d}_stage2.png"
            s2_info = render_stage2_explanation(model_result, crop_img, row.to_dict(),
                                                thresholds, s2_path, dry_run=dry_run)

            # Combined
            comb_path = combined_dir / f"{crop_id:05d}_combined.png"
            comb_info = render_combined_panel(ct_hu, lesion_mask, roi_mask, model_result,
                                              crop_img, row.to_dict(), thresholds, comb_path,
                                              dry_run=dry_run)

            results.append({
                "crop_id": crop_id,
                "patient_id": patient_id,
                "safe_id": safe_id,
                "qa_group": str(row.get("qa_group", "")),
                "qa_priority": str(row.get("qa_priority_x", row.get("qa_priority", ""))),
                "padim_score_mean": float(row.get("padim_score_mean", 0.0)),
                "crop_score_l1_mean_stored": float(row.get("crop_score_l1_mean", 0.0)),
                "crop_score_l1_mean_computed": model_result["crop_score_l1_mean_computed"],
                "threshold_exceed_val_p90": bool(row.get("threshold_exceed_val_p90", False)),
                "threshold_exceed_val_p95": bool(row.get("threshold_exceed_val_p95", False)),
                "lesion_overlap_class": str(row.get("lesion_overlap_class", "")),
                "has_lesion_in_context": s1_info["has_lesion"],
                "stage1_overlay": str(s1_path) if not dry_run else "dry_run",
                "stage2_explanation": str(s2_path) if not dry_run else "dry_run",
                "combined": str(comb_path) if not dry_run else "dry_run",
                "score_match": s2_info["score_match"],
                "status": "ok",
            })
            print(f"  [OK] crop_id={crop_id}")

        except Exception as e:
            import traceback
            print(f"  [ERROR] crop_id={crop_id}: {e}")
            traceback.print_exc()
            errors.append({"crop_id": crop_id, "error": str(e)})

    return {"results": results, "errors": errors}


# ──────────────────────────────────────────────────────────
# Save manifest + summary + guide
# ──────────────────────────────────────────────────────────

def save_outputs(run_result: dict, project_root: Path, mode: str) -> None:
    import pandas as pd

    out_root = project_root / OUTPUT_VIZ_ROOT
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "manifest_dual_stage_review_v1.csv"
    summary_path = out_root / "summary_dual_stage_review_v1.json"
    guide_path = out_root / "review_guide_dual_stage_review_v1.md"

    # Overwrite guard
    for p in [manifest_path, summary_path, guide_path]:
        if p.exists():
            print(f"[WARN] Output file already exists, will not overwrite: {p}")
            return

    results = run_result.get("results", [])
    errors = run_result.get("errors", [])

    if results:
        df_out = pd.DataFrame(results)
        df_out.to_csv(manifest_path, index=False)
        print(f"[INFO] Manifest saved: {manifest_path}")

    summary = {
        "mode": mode,
        "timestamp": datetime.now().isoformat(),
        "n_target": len(P1_CROP_IDS) + len(P2_CROP_IDS),
        "n_processed": len(results),
        "n_errors": len(errors),
        "errors": errors,
        "val_p90": FALLBACK_VAL_P90,
        "val_p95": FALLBACK_VAL_P95,
        "stage1_method": "patch_scan_region_bbox + score_text (no pixel heatmap)",
        "stage2_method": "error_map + gradient_saliency",
        "grad_cam_note": (
            "1차 PaDiM: Mahalanobis distance 기반, Grad-CAM 불가. "
            "2차 RD4AD: reconstruction baseline, Grad-CAM 불가. "
            "대체: bbox+error_map+saliency 사용."
        ),
        "heatmap_limitation": HEATMAP_PROXY_LIMITATION,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Summary saved: {summary_path}")

    guide_content = f"""# Dual-Stage Overlay Review Guide

Generated: {datetime.now().isoformat()}
Mode: {mode}

## 시각화 구성

### A. Stage1 Overlay Panel (`stage1_overlay/`)
- Context 192x192 (lung window)
- Context 192x192 (mediastinal window)
- ROI contour (green)
- Lesion contour (red)
- Stage1 patch scan region bbox (yellow dashed)
- Stage2 fixed96 crop bbox (blue solid)

### B. Stage2 Explanation Panel (`stage2_explanation/`)
- Input crop (lung_z, med_z)
- Reconstruction (lung_z, med_z)
- |Error| heatmap (lung_z, med_z)
- Gradient saliency (if available)
- Score text

### C. Combined Panel (`combined/`)
- Stage1 context (lung) + Stage2 explanation side by side

## Grad-CAM 제한 이유

1. **1차 PaDiM**: Mahalanobis distance 기반, 분류 logit 없음
   - 표준 Grad-CAM 불가
   - 대체: bbox + score text

2. **2차 RD4AD (minimal reconstruction baseline)**:
   - 분류 logit 없음
   - 대체: reconstruction error map + gradient saliency

## 한계
- {HEATMAP_PROXY_LIMITATION}

## 대상 crop_ids
- P1 (patch_overlap): {P1_CROP_IDS}
- P2 (context_overlap_only): {P2_CROP_IDS}
"""
    with open(guide_path, "w", encoding="utf-8") as f:
        f.write(guide_content)
    print(f"[INFO] Review guide saved: {guide_path}")


# ──────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Dual-stage overlay review visualization (P1/P2, 12 crops)"
    )
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="Run preflight checks only. No model forward, no PNG generation.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process DRYRUN_CROP_IDS (2 crops). No PNG saved to disk.",
    )
    parser.add_argument(
        "--full-run", action="store_true",
        help="Process all 12 target crop IDs. Requires explicit flag.",
    )
    parser.add_argument(
        "--skip-saliency", action="store_true",
        help="Skip gradient saliency computation (faster).",
    )
    parser.add_argument(
        "--device", default="cuda_if_available",
        choices=["cuda", "cpu", "cuda_if_available"],
    )
    parser.add_argument(
        "--project-root", default=".",
        help="Project root directory.",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────

def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    # Resolve device
    if args.device == "cuda_if_available":
        try:
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            device = "cpu"
    else:
        try:
            import torch
            device = torch.device(args.device)
        except ImportError:
            device = args.device
    print(f"[INFO] Device: {device}")

    if args.preflight_only:
        print("[MODE] preflight-only")
        result = run_preflight(project_root, ALL_TARGET_CROP_IDS)
        if result["passed"]:
            print("\n[PREFLIGHT PASSED] Ready for dry-run approval.")
        else:
            print("\n[PREFLIGHT FAILED] Fix issues before proceeding.")
            sys.exit(1)
        return

    if args.dry_run:
        print(f"[MODE] dry-run (crop_ids: {DRYRUN_CROP_IDS})")
        preflight = run_preflight(project_root, DRYRUN_CROP_IDS)
        if not preflight["passed"]:
            print("[ERROR] Preflight failed. Cannot proceed with dry-run.")
            sys.exit(1)
        run_result = run_visualization(
            project_root, DRYRUN_CROP_IDS, device,
            skip_saliency=args.skip_saliency, dry_run=True,
        )
        print(f"\n[DRY-RUN COMPLETE] processed={len(run_result['results'])} errors={len(run_result['errors'])}")
        for r in run_result["results"]:
            print(f"  crop_id={r['crop_id']} score_match={r['score_match']} has_lesion={r['has_lesion_in_context']}")
        for e in run_result["errors"]:
            print(f"  [ERROR] crop_id={e['crop_id']}: {e['error']}")
        return

    if args.full_run:
        print(f"[MODE] full-run (crop_ids: {ALL_TARGET_CROP_IDS})")
        preflight = run_preflight(project_root, ALL_TARGET_CROP_IDS)
        if not preflight["passed"]:
            print("[ERROR] Preflight failed. Cannot proceed with full-run.")
            sys.exit(1)
        run_result = run_visualization(
            project_root, ALL_TARGET_CROP_IDS, device,
            skip_saliency=args.skip_saliency, dry_run=False,
        )
        save_outputs(run_result, project_root, mode="full_run")
        print(f"\n[FULL-RUN COMPLETE] processed={len(run_result['results'])} errors={len(run_result['errors'])}")
        for e in run_result["errors"]:
            print(f"  [ERROR] crop_id={e['crop_id']}: {e['error']}")
        return

    print("[ERROR] No mode specified. Use --preflight-only, --dry-run, or --full-run.")
    sys.exit(1)


if __name__ == "__main__":
    main()
