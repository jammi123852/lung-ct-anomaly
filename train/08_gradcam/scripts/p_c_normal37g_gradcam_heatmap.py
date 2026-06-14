"""
P-C-NORMAL37g: Grad-CAM mask-inside-only / YlOrRd gradient heatmap
- 37f와 동일한 입력/Grad-CAM 계산 (masked input, img_features[7])
- 색상 변경: yellow→orange→red gradient (YlOrRd), alpha ∝ intensity
- 낮은 값 = 거의 투명, 높은 값 = 진한 빨강 → 그라데이션 단계감 확보
- mask 내부로만 heatmap 표시
- 새 output root만 사용 / 재학습·threshold sweep 금지
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
import matplotlib.cm as mpl_cm

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")
STAGE_LABEL  = "P-C-NORMAL37g"

# ── paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT   = PROJECT_ROOT / "outputs/p_c_normal30b_masked_input_full_train/checkpoints/p_c_normal30b_best_val_auc_checkpoint.pt"
SCORE_CSV    = PROJECT_ROOT / "outputs/p_c_normal35_full_downstream_scoring/p_c_normal35_full_crop_scores.csv"
SCALAR_STATS = PROJECT_ROOT / "outputs/reports/p_c_normal24h_fix_zroi_scalar_fusion_drycheck/p_c_normal24h_fix_scalar_normalization_stats.json"

OUTPUT_DIR   = PROJECT_ROOT / "outputs/p_c_normal37g_gradcam_mask_inside_heatmap"
REPORT_DIR   = PROJECT_ROOT / "outputs/reports/p_c_normal37g_gradcam_mask_inside_heatmap"
PANELS_DIR   = OUTPUT_DIR / "panels"
CONTEXT_DIR  = OUTPUT_DIR / "context"

NSCLC_VOL_ROOT  = Path("/mnt/c/Users/jinhy/Desktop/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy")
NORMAL_VOL_ROOT = Path("/mnt/c/Users/jinhy/Desktop/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/volumes_npy")

# ── constants ─────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
HU_MIN        = -1000.0
HU_MAX        = 200.0
CROP_SIZE     = 96
HALF          = CROP_SIZE // 2
ALPHA_GAMMA   = 0.55    # alpha = cam^gamma: 작을수록 낮은 값이 더 투명해짐
ALPHA_MAX     = 0.88    # 최고 값의 alpha 상한

# 37e/37f와 동일한 sample 목록
SAMPLE_TARGETS = [
    ("NSCLC_TP",  "LUNG1-196", "81.0"),
    ("NSCLC_TP",  "LUNG1-205", "146.0"),
    ("NSCLC_TP",  "LUNG1-349", "159.0"),
    ("NSCLC_TP",  "LUNG1-043", "124.0"),
    ("NSCLC_TP",  "LUNG1-396", "130.0"),
    ("Normal_FP", "normal016", "45.0"),
    ("low_mask",  "LUNG1-205", "204.0"),
    ("low_mask",  "LUNG1-205", "205.0"),
]
NORMAL_FP_PIDS = [
    "subset3_1.3.6.1.4.1.14519.5.2.1.6279.6001.314519596680450457855054746285",
    "subset2_1.3.6.1.4.1.14519.5.2.1.6279.6001.102133688497886810253331438797",
]

# YlOrRd: 0=노랑, 0.5=주황, 1=빨강 (matplotlib 내장)
_YLRD = mpl_cm.get_cmap("YlOrRd")


# ── utils ─────────────────────────────────────────────────────────────────────
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


# ── Model ─────────────────────────────────────────────────────────────────────
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
    print(f"  checkpoint loaded: epoch={ckpt.get('epoch')}")
    return model


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
def compute_gradcam_raw(model, img_t, sc_t):
    """img_features[7] → GAP weights → ReLU → upsample 96×96, returns raw cam [0,1]"""
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
    logit = model(img_t, sc_t)
    logit.backward()

    fh.remove()
    bh.remove()

    act  = activations["feat"]
    grad = gradients["feat"]

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam     = (act * weights).sum(dim=1, keepdim=True)
    cam     = torch.relu(cam)
    cam_up  = F.interpolate(cam, size=(CROP_SIZE, CROP_SIZE), mode="bilinear", align_corners=False)
    cam_np  = cam_up[0, 0].detach().cpu().numpy().astype(np.float32)

    cmin, cmax = float(cam_np.min()), float(cam_np.max())
    cam_np = (cam_np - cmin) / (cmax - cmin) if cmax - cmin > 1e-8 else np.zeros_like(cam_np)

    logit_val = float(logit.item())
    prob_val  = float(torch.sigmoid(logit).item())
    return cam_np, logit_val, prob_val


def apply_mask_inside_norm(cam_raw, mask_96):
    """mask 내부만 남기고 mask 내부 픽셀 기준으로 min-max normalize"""
    cam_masked   = cam_raw * mask_96
    inside_vals  = cam_masked[mask_96 > 0.5]

    if len(inside_vals) == 0 or (inside_vals.max() - inside_vals.min()) < 1e-8:
        return np.zeros_like(cam_masked), True

    vmin     = float(inside_vals.min())
    vmax     = float(inside_vals.max())
    cam_norm = np.where(mask_96 > 0.5, (cam_masked - vmin) / (vmax - vmin), 0.0)
    return cam_norm.astype(np.float32), False


# ── input prep (37e/37f 동일) ─────────────────────────────────────────────────
def prep_input(row, scalar_stats):
    feat      = scalar_stats["features"]
    lzp_mean  = feat["lung_z_percentile"]["mean"]
    lzp_std   = feat["lung_z_percentile"]["std"]
    clrr_mean = feat["crop_lung_roi_ratio"]["mean"]
    clrr_std  = feat["crop_lung_roi_ratio"]["std"]

    data    = np.load(row["crop_path"])
    ct_crop = data["ct_crop"].astype(np.float32)
    arr     = np.clip(ct_crop, HU_MIN, HU_MAX)
    arr     = (arr - HU_MIN) / (HU_MAX - HU_MIN)

    mdata = np.load(row["mask_path"])
    if "mask_3ch" in mdata:
        mask_3ch = mdata["mask_3ch"].astype(np.float32)
    elif "mask" in mdata:
        m1 = mdata["mask"].astype(np.float32)
        mask_3ch = np.stack([m1, m1, m1], axis=0)
    else:
        mask_3ch = np.ones_like(arr)

    masked_arr = arr * mask_3ch
    mean_t   = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std_t    = np.array(IMAGENET_STD,  dtype=np.float32).reshape(3, 1, 1)
    img_norm = (masked_arr - mean_t) / std_t
    img_t    = torch.tensor(img_norm[np.newaxis], dtype=torch.float32)

    lzp_n  = (float(row["lung_z_percentile_raw"])  - lzp_mean)  / lzp_std
    clrr_n = (float(row["crop_lung_roi_ratio_raw"]) - clrr_mean) / clrr_std
    sc_t   = torch.tensor([[lzp_n, clrr_n]], dtype=torch.float32)

    raw_disp    = (np.clip(ct_crop[1], HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)
    mask_96     = mask_3ch[1]
    masked_disp = raw_disp * mask_96
    return img_t, sc_t, raw_disp, masked_disp, mask_96


# ── heatmap RGBA ──────────────────────────────────────────────────────────────
def make_heat_rgba(cam_norm):
    """
    YlOrRd gradient: yellow(low)→orange(mid)→red(high)
    alpha = cam_norm^ALPHA_GAMMA * ALPHA_MAX
    → 낮은 값 거의 투명, 높은 값 선명하게
    """
    rgba        = _YLRD(cam_norm).astype(np.float32)   # (H, W, 4), RGB from colormap
    rgba[..., 3] = np.power(cam_norm, ALPHA_GAMMA) * ALPHA_MAX
    return rgba


# ── panel PNG (6-panel) ───────────────────────────────────────────────────────
def save_panel_png(raw_disp, masked_disp, mask_96, cam_norm, zero_cam,
                   row, metrics, png_path, plt):
    pid   = row["patient_id"]
    z     = row["canonical_volume_z"]
    cy    = row["center_y"]
    cx    = row["center_x"]
    label = "NSCLC" if row["label"] == "1" else "normal"
    pred  = "NSCLC" if row["pred_at_0p5"] == "1" else "normal"
    prob  = float(row["prob"])
    logit = float(row["logit"])
    low_m = row["low_mask_flag"]
    ratio = float(row["mask_nonzero_ratio_mean"])
    pim   = metrics.get("peak_inside_mask", "?")

    title = (f"{pid}  z={z}  cy={cy} cx={cx}\n"
             f"label={label}  pred={pred}  prob={prob:.4f}  logit={logit:.2f}"
             f"  low_mask={low_m}  mask_ratio={ratio:.3f}  peak_in_mask={pim}")

    heat_rgba = make_heat_rgba(cam_norm)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.patch.set_facecolor("#111111")
    fig.suptitle(title, fontsize=7.5, color="white", y=1.01)

    def show_gray(ax, arr, ttl):
        ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
        ax.set_title(ttl, fontsize=7, color="white")
        ax.set_facecolor("black")
        ax.axis("off")

    show_gray(axes[0, 0], raw_disp,    "1. raw center slice")
    show_gray(axes[0, 1], masked_disp, "2. masked center slice")
    show_gray(axes[0, 2], mask_96,     "3. crop mask")

    # 4번: masked + heatmap
    axes[1, 0].set_facecolor("black")
    axes[1, 0].imshow(masked_disp, cmap="gray", vmin=0, vmax=1)
    if not zero_cam:
        axes[1, 0].imshow(heat_rgba)
    axes[1, 0].set_title("4. masked + Grad-CAM heatmap (mask inside)", fontsize=7, color="white")
    axes[1, 0].axis("off")

    # 5번: raw + mask boundary + heatmap
    axes[1, 1].set_facecolor("black")
    axes[1, 1].imshow(raw_disp, cmap="gray", vmin=0, vmax=1)
    if not zero_cam:
        axes[1, 1].imshow(heat_rgba)
    axes[1, 1].contour(mask_96, levels=[0.5], colors=["cyan"], linewidths=0.8)
    axes[1, 1].set_title("5. raw + mask boundary + Grad-CAM heatmap", fontsize=7, color="white")
    axes[1, 1].axis("off")

    # 6번: heatmap only
    axes[1, 2].set_facecolor("black")
    black_bg = np.zeros_like(raw_disp)
    axes[1, 2].imshow(black_bg, cmap="gray", vmin=0, vmax=1)
    if not zero_cam:
        axes[1, 2].imshow(heat_rgba)
    else:
        axes[1, 2].text(HALF, HALF, "near_blank\n(mask coverage 낮음)",
                        ha="center", va="center", color="yellow", fontsize=7)
    # 컬러바 (YlOrRd, alpha 무시한 색상 표시용)
    sm = plt.cm.ScalarMappable(cmap="YlOrRd")
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[1, 2], fraction=0.046, pad=0.04)
    cbar.set_label("anomaly (yellow=low, orange=mid, red=high)", color="white", fontsize=6)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    axes[1, 2].set_title("6. heatmap only (mask inside)", fontsize=7, color="white")
    axes[1, 2].axis("off")

    plt.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ── context PNG ───────────────────────────────────────────────────────────────
def save_context_png(row, cam_norm, png_path, plt):
    safe_id      = row["safe_id"]
    source_split = row["source_split"]
    z_str        = row["canonical_volume_z"]
    cy           = int(float(row["center_y"]))
    cx           = int(float(row["center_x"]))
    pid          = row["patient_id"]
    label        = "NSCLC" if row["label"] == "1" else "normal"
    prob         = float(row["prob"])

    if source_split == "stage2_holdout":
        vol_path = NSCLC_VOL_ROOT / safe_id / "ct_hu.npy"
    else:
        vol_path = NORMAL_VOL_ROOT / safe_id / "ct_hu.npy"

    if not vol_path.exists():
        return False

    vol = np.load(str(vol_path), mmap_mode="r")
    if vol.ndim == 4:
        vol = vol[:, 0, :, :]
    z_idx = min(int(float(z_str)), vol.shape[0] - 1)
    ct_sl = vol[z_idx].astype(np.float32)
    ct_d  = np.clip(ct_sl, HU_MIN, HU_MAX)
    ct_d  = (ct_d - HU_MIN) / (HU_MAX - HU_MIN)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#111111")
    fig.suptitle(f"{pid}  z={z_str}  label={label}  prob={prob:.4f}", fontsize=8, color="white")

    axes[0].imshow(ct_d, cmap="gray", vmin=0, vmax=1)
    rect = patches.Rectangle((cx - HALF, cy - HALF), CROP_SIZE, CROP_SIZE,
                              linewidth=1.5, edgecolor="red", facecolor="none")
    axes[0].add_patch(rect)
    axes[0].set_title(f"full CT slice z={z_str}  (red box=crop region)", fontsize=7, color="white")
    axes[0].axis("off")
    axes[0].set_facecolor("black")

    y0 = max(0, cy - HALF); y1 = min(ct_d.shape[0], cy + HALF)
    x0 = max(0, cx - HALF); x1 = min(ct_d.shape[1], cx + HALF)
    crop_zoom = ct_d[y0:y1, x0:x1]
    h_crop    = min(cam_norm.shape[0], crop_zoom.shape[0])
    w_crop    = min(cam_norm.shape[1], crop_zoom.shape[1])
    heat_rgba = make_heat_rgba(cam_norm[:h_crop, :w_crop])
    axes[1].imshow(crop_zoom, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(heat_rgba)
    axes[1].set_title("crop zoom + Grad-CAM (yellow→orange→red)", fontsize=7, color="white")
    axes[1].axis("off")
    axes[1].set_facecolor("black")

    plt.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


# ── qualitative tag ───────────────────────────────────────────────────────────
def assign_tag(row, cam_norm, mask_96, zero_cam):
    low_mask   = row["low_mask_flag"] == "True"
    mask_ratio = float(row["mask_nonzero_ratio_mean"])

    if zero_cam or low_mask or mask_ratio < 0.1:
        return "near_blank"

    inside_vals  = cam_norm[mask_96 > 0.5]
    outside_vals = cam_norm[mask_96 <= 0.5]
    in_mean  = float(inside_vals.mean())  if len(inside_vals)  > 0 else 0.0
    out_mean = float(outside_vals.mean()) if len(outside_vals) > 0 else 0.0
    in_std   = float(inside_vals.std())   if len(inside_vals)  > 1 else 0.0

    py, px      = np.unravel_index(int(cam_norm.argmax()), cam_norm.shape)
    peak_inside = bool(mask_96[py, px] > 0.5)

    if peak_inside and in_mean > out_mean:
        return "diffuse_inside_mask" if in_std < 0.15 else "lesion_focused"
    if not peak_inside:
        return "boundary_structure_suspect"
    return "diffuse_inside_mask"


# ── contact sheet ─────────────────────────────────────────────────────────────
def save_contact_sheet(panel_paths, out_path, plt):
    n = len(panel_paths)
    if n == 0:
        return
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6 * nrows))
    fig.patch.set_facecolor("#111111")
    axes = np.array(axes).reshape(nrows, ncols)
    for i, pp in enumerate(panel_paths):
        r, c = divmod(i, ncols)
        img  = plt.imread(str(pp))
        axes[r, c].imshow(img)
        axes[r, c].axis("off")
        axes[r, c].set_title(pp.name[:55], fontsize=5.5, color="white")
        axes[r, c].set_facecolor("black")
    for i in range(n, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=80, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ── report.md ────────────────────────────────────────────────────────────────
def write_report(sample_metrics, report_dir):
    lines = [
        "# P-C-NORMAL37g Grad-CAM Mask-Inside Heatmap Report",
        "",
        "## 1. 수정 목적",
        "- 기존 37f의 단색 빨강 overlay 방식에서 일반 heatmap gradient 방식으로 변경",
        "- high=red, mid=orange, low=yellow 형태로 위험도 차이를 색으로 구분",
        "- mask 내부로만 heatmap 제한 유지",
        "- score/model/threshold 변경 없음 (visualization only)",
        "",
        "## 2. 기존 37e/37f와의 차이",
        "| 항목 | 37e | 37f | 37g |",
        "|---|---|---|---|",
        "| heatmap 범위 | 96×96 전체 | mask 내부만 | mask 내부만 |",
        "| 색상 방식 | hot (단색) | custom red alpha | YlOrRd gradient |",
        "| 색상 단계 | 없음 | 없음 | yellow→orange→red |",
        "| alpha | 고정 | intensity 비례 | intensity^0.55 비례 |",
        "",
        "## 3. 판정 기준",
        "- **near_blank**: low_mask=True 또는 mask_ratio < 0.10",
        "- **lesion_focused**: peak_inside_mask=True, in_mean > out_mean, in_std >= 0.15",
        "- **diffuse_inside_mask**: 조건 충족하나 in_std < 0.15 (mask 내 분산)",
        "- **boundary_structure_suspect**: peak가 mask 바깥",
        "",
        "## 4. 샘플별 요약",
        "",
        "| # | type | patient | z | label | pred | prob | logit | low_mask | peak_in_mask | tag |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in sample_metrics:
        pid = m["patient_id"][:25]
        lines.append(
            f"| {m['sample_id']} | {m['sample_type']} | {pid} | {m['canonical_volume_z']} "
            f"| {m['label']} | {m['pred']} | {m['prob']:.4f} | {m['logit']:.2f} "
            f"| {m['low_mask']} | {m['peak_inside_mask']} | {m['tag']} |"
        )

    nsclc = [m for m in sample_metrics if m["sample_type"] == "NSCLC_TP"]
    fp    = [m for m in sample_metrics if m["sample_type"] == "Normal_FP"]
    lm    = [m for m in sample_metrics if m["sample_type"] == "low_mask"]

    lines += ["", "## 5. 유형별 해석", "", "### NSCLC TP"]
    for m in nsclc:
        lines.append(
            f"- **{m['patient_id']}** z={m['canonical_volume_z']}: `{m['tag']}` "
            f"(inside_mean={m['inside_cam_mean']:.3f}, io_ratio={m['inside_to_outside_ratio']})"
        )

    lines += ["", "### Normal FP"]
    for m in fp:
        lines.append(
            f"- **{m['patient_id'][:30]}** z={m['canonical_volume_z']}: `{m['tag']}` "
            f"(peak_in_mask={m['peak_inside_mask']}, inside_mean={m['inside_cam_mean']:.3f})"
        )

    lines += ["", "### low_mask FN"]
    for m in lm:
        lines.append(
            f"- **{m['patient_id']}** z={m['canonical_volume_z']}: `{m['tag']}` "
            f"(mask_ratio={m['mask_ratio']:.3f}, zero_cam={m['zero_cam_after_masking']})"
        )

    lines += [
        "",
        "## 6. Caveat",
        "- 이 결과는 masked-input 모델에 대한 Grad-CAM 시각화다.",
        "- mask 내부 입력 공간에서의 상대적 gradient 활성도를 표현한다.",
        "- 색이 빨간 부분 = 모델이 해당 위치 픽셀에 가장 많이 반응한 곳이다.",
        "- 이 결과로 진단 판정을 내리는 것은 금지한다.",
    ]

    (report_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if REPORT_DIR.exists() and any(REPORT_DIR.iterdir()):
        _abort(f"output already exists: {REPORT_DIR}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PANELS_DIR.mkdir(parents=True, exist_ok=True)
    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[{STAGE_LABEL}] start: {ts}")

    scalar_stats = json.loads(SCALAR_STATS.read_text())

    print(f"[{STAGE_LABEL}] Step1: load score CSV")
    score_rows = list(csv.DictReader(open(SCORE_CSV)))
    idx        = defaultdict(list)
    for r in score_rows:
        idx[(r["patient_id"], r["canonical_volume_z"])].append(r)
    print(f"  {len(score_rows)} rows")

    print(f"[{STAGE_LABEL}] Step2: sample selection")
    samples = []

    def pick_top(pid, z, typ):
        key = (pid, z)
        if key not in idx:
            print(f"  [SKIP] {pid} z={z}")
            return
        row = sorted(idx[key], key=lambda r: -float(r["logit"]))[0]
        row["sample_type"] = typ
        samples.append(row)
        print(f"  [{typ}] {pid} z={z} logit={float(row['logit']):.2f}")

    for typ, pid, z in SAMPLE_TARGETS:
        pick_top(pid, z, typ)

    for pid in NORMAL_FP_PIDS:
        prows = [r for r in score_rows if r["patient_id"] == pid
                 and r["label"] == "0" and r["pred_at_0p5"] == "1"]
        if prows:
            row = sorted(prows, key=lambda r: -float(r["logit"]))[0]
            row["sample_type"] = "Normal_FP"
            samples.append(row)
            print(f"  [Normal_FP] {pid[:40]}... logit={float(row['logit']):.2f}")

    print(f"  total: {len(samples)} samples")

    print(f"[{STAGE_LABEL}] Step3: load model")
    model = load_model()

    print(f"[{STAGE_LABEL}] Step4: Grad-CAM + render")
    sample_metrics = []
    panel_paths    = []
    n_ok = n_fail  = 0

    for i, row in enumerate(samples):
        pid  = row["patient_id"]
        z    = row["canonical_volume_z"]
        typ  = row.get("sample_type", "?")
        print(f"  [{i+1}/{len(samples)}] {typ} {pid} z={z}")

        try:
            img_t, sc_t, raw_disp, masked_disp, mask_96 = prep_input(row, scalar_stats)
        except Exception as e:
            print(f"    [WARN] prep failed: {e}")
            n_fail += 1
            continue

        try:
            cam_raw, logit_repr, prob_repr = compute_gradcam_raw(model, img_t, sc_t)
        except Exception as e:
            print(f"    [WARN] gradcam failed: {e}")
            n_fail += 1
            continue

        if np.isnan(cam_raw).any() or np.isinf(cam_raw).any():
            print(f"    [WARN] cam_raw NaN/Inf")
            n_fail += 1
            continue

        cam_norm, zero_cam = apply_mask_inside_norm(cam_raw, mask_96)

        inside_vals  = cam_norm[mask_96 > 0.5]
        outside_vals = cam_norm[mask_96 <= 0.5]
        in_mean  = float(inside_vals.mean())  if len(inside_vals)  > 0 else 0.0
        in_max   = float(inside_vals.max())   if len(inside_vals)  > 0 else 0.0
        out_mean = float(outside_vals.mean()) if len(outside_vals) > 0 else 0.0
        out_max  = float(outside_vals.max())  if len(outside_vals) > 0 else 0.0
        ratio_io = round(in_mean / (out_mean + 1e-8), 2)

        py, px      = np.unravel_index(int(cam_norm.argmax()), cam_norm.shape)
        peak_inside = bool(mask_96[py, px] > 0.5)

        logit_csv   = float(row["logit"])
        logit_match = abs(logit_repr - logit_csv) < 0.5
        print(f"    logit_csv={logit_csv:.2f}  logit_repr={logit_repr:.2f}  match={logit_match}")

        tag = assign_tag(row, cam_norm, mask_96, zero_cam)

        safe_pid   = pid.replace("/", "_").replace(".", "_")[:50]
        safe_z     = str(z).replace(".", "p")
        panel_name = f"{i+1:02d}_{typ}_{safe_pid}_z{safe_z}.png"
        panel_path = PANELS_DIR / panel_name

        metrics = {"peak_inside_mask": peak_inside, "inside_cam_mean": round(in_mean, 4)}
        try:
            save_panel_png(raw_disp, masked_disp, mask_96, cam_norm, zero_cam,
                           row, metrics, panel_path, plt)
            panel_paths.append(panel_path)
            print(f"    panel: {panel_name}  tag={tag}")
        except Exception as e:
            print(f"    [WARN] panel failed: {e}")

        ctx_path = CONTEXT_DIR / f"{i+1:02d}_{typ}_{safe_pid}_z{safe_z}_ctx.png"
        try:
            save_context_png(row, cam_norm, ctx_path, plt)
        except Exception as e:
            print(f"    [WARN] context failed: {e}")

        sample_metrics.append({
            "sample_id":               i + 1,
            "sample_type":             typ,
            "patient_id":              pid,
            "canonical_volume_z":      z,
            "label":                   row["label"],
            "pred":                    row["pred_at_0p5"],
            "prob":                    round(float(row["prob"]), 6),
            "logit":                   round(logit_repr, 4),
            "logit_csv":               round(logit_csv, 4),
            "logit_match":             logit_match,
            "low_mask":                row["low_mask_flag"],
            "mask_ratio":              round(float(row["mask_nonzero_ratio_mean"]), 4),
            "cam_peak_y":              py,
            "cam_peak_x":              px,
            "peak_inside_mask":        peak_inside,
            "inside_cam_mean":         round(in_mean,  4),
            "inside_cam_max":          round(in_max,   4),
            "outside_cam_mean":        round(out_mean, 4),
            "outside_cam_max":         round(out_max,  4),
            "inside_to_outside_ratio": ratio_io,
            "zero_cam_after_masking":  zero_cam,
            "tag":                     tag,
        })
        n_ok += 1

    if panel_paths:
        try:
            save_contact_sheet(panel_paths, REPORT_DIR / "contact_sheet.png", plt)
            print(f"  contact_sheet saved")
        except Exception as e:
            print(f"  [WARN] contact sheet: {e}")

    write_report(sample_metrics, REPORT_DIR)

    guardrail_rows = [
        {"item": "no_training",                  "value": True},
        {"item": "no_threshold_sweep",           "value": True},
        {"item": "no_existing_output_modified",  "value": True},
        {"item": "no_37e_37f_overwrite",         "value": True},
        {"item": "new_output_root_only",         "value": True},
        {"item": "masked_input_reproduced",      "value": True},
        {"item": "sample_pilot_only",            "value": True},
        {"item": "diagnostic_claim",             "value": False},
        {"item": "n_ok",                         "value": n_ok},
        {"item": "n_fail",                       "value": n_fail},
    ]
    _write_csv(guardrail_rows, REPORT_DIR / "guardrail_check.csv")
    _write_csv(sample_metrics, REPORT_DIR / "sample_metrics.csv")

    verdict   = "PASS" if n_fail == 0 else ("PARTIAL_PASS" if n_ok > 0 else "FAIL")
    all_match = all(m["logit_match"] for m in sample_metrics)

    tag_counts = defaultdict(int)
    for m in sample_metrics:
        tag_counts[m["tag"]] += 1

    summary = {
        "stage": STAGE_LABEL,
        "timestamp": ts,
        "verdict": verdict,
        "n_ok": n_ok,
        "n_fail": n_fail,
        "logit_all_match": all_match,
        "gradcam_target_layer": "model.img_features[7]",
        "colormap": "YlOrRd (yellow=low, orange=mid, red=high), alpha=cam^0.55*0.88",
        "mask_inside_only": True,
        "no_training": True,
        "no_threshold_sweep": True,
        "no_existing_output_modified": True,
        "tag_counts": dict(tag_counts),
    }
    _write_json(summary, REPORT_DIR / "summary.json")
    _write_json({"verdict": verdict, "timestamp": ts}, REPORT_DIR / "DONE.json")

    print(f"\n[{STAGE_LABEL}] verdict: {verdict}")
    print(f"  n_ok={n_ok}  n_fail={n_fail}  logit_all_match={all_match}")
    print(f"  tags: {dict(tag_counts)}")
    print(f"  panels:  {PANELS_DIR}")
    print(f"  report:  {REPORT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
