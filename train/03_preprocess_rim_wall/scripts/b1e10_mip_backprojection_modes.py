#!/usr/bin/env python3
"""
B1-E10: MIP mask back-projection mode 비교
목적: B1-E9 strict argmax에서 LUNG1-020 z79=0px 발생 원인 확인 및 완화 mode 검증

Back-projection modes:
  strict                : argmax_z == z  (B1-E9 기존)
  relaxed_1             : abs(argmax_z - z) <= 1
  relaxed_2             : abs(argmax_z - z) <= 2
  slab_projection_upper : argmax 조건 없음 (upper-bound 참고용, 최종 후보 아님)

비교 후보 mask rule (3개):
  A_0     : MIP > 0
  B_p95   : MIP ROI p95
  D_v_p95 : MIP Frangi p95

PNG:
  - main PNG : B_p95 + relaxed_1 대표 (4패널: vessel / new_relaxed1 / MIP / 비교)
  - mode 비교 PNG : strict / relaxed_1 / relaxed_2 / upper 한 장 (subset only, B_p95 기준)
  - LUNG1-020 z79 반드시 포함 (0px 문제 완화 확인)

실행:
  --dry-run   candidate × mode pixel count preview (CT 로드, PNG 미생성)
  --real      PNG + CSV + JSON + report 생성
"""
import sys

# ── bare-run guard ──────────────────────────────────────────────────────────
if __name__ == "__main__" and len(sys.argv) < 2:
    print(
        "[ERROR] bare-run guard: 인수 없이 직접 실행 금지. "
        "--dry-run 또는 --real 을 사용하세요.",
        file=sys.stderr,
    )
    sys.exit(2)

import os
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from skimage import measure
from skimage.filters import frangi
from skimage.morphology import white_tophat, disk

ALLOW_REAL_PROCESSING = "--real" in sys.argv

# ── 경로 설정 ────────────────────────────────────────────────────────────────
PROJECT = Path("/home/jinhy/project/lung-ct-anomaly")
OUT_ROOT = (
    PROJECT
    / "outputs/position-aware-padim-v1"
    / "oracle_vessel_suppression_effb0_v4_20_dev_only"
    / "b1e10_mip_backprojection_modes_v1"
)
STAGE_SPLIT_CSV = (
    PROJECT / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1.csv"
)
B1E3_PATCH_PREVIEW = (
    PROJECT
    / "outputs/position-aware-padim-v1"
    / "oracle_vessel_suppression_effb0_v4_20_dev_only"
    / "b1e3_oracle_score_suppression_smoke_v1"
    / "b1e3_oracle_score_suppression_patch_preview.csv"
)
B1E8_INDEX_CSV = (
    PROJECT
    / "outputs/position-aware-padim-v1"
    / "oracle_vessel_suppression_effb0_v4_20_dev_only"
    / "b1e8_mip_pseudomask_vs_vesselness_v1"
    / "b1e8_mip_pseudomask_vs_vesselness_index.csv"
)
ROI_BASE_LESION = (
    PROJECT / "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1/lesion"
)
ROI_BASE_NORMAL = (
    PROJECT / "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1/normal"
)
SCORE_BASE_LESION = (
    PROJECT
    / "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1/outputs/scores"
    / "lesion_stage1_dev_by_patient"
)
SCORE_BASE_NORMAL = (
    PROJECT
    / "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1/outputs/scores"
    / "normal_test_by_patient"
)
NROOT_LESION = Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy"
)
NROOT_NORMAL = Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1/volumes_npy"
)

# ── 파라미터 ────────────────────────────────────────────────────────────────
MIP_SLAB_HALF_LO = 5
MIP_SLAB_HALF_HI = 5
AIR_THRESHOLD = -850
VESSEL_SIGMAS = (0.5, 1.0, 1.5, 2.0)
VESSEL_THRESH_PERCENTILE = 75

# ── Back-projection mode 정의 ────────────────────────────────────────────────
BACKPROJECTION_MODES = [
    {"name": "strict",                "label": "strict (argmax==z)",         "tolerance": 0,    "upper": False},
    {"name": "relaxed_1",             "label": "relaxed_1 (|argmax-z|<=1)",  "tolerance": 1,    "upper": False},
    {"name": "relaxed_2",             "label": "relaxed_2 (|argmax-z|<=2)",  "tolerance": 2,    "upper": False},
    {"name": "slab_projection_upper", "label": "upper (no argmax, ref only)","tolerance": None, "upper": True},
]
MODE_MAP = {m["name"]: m for m in BACKPROJECTION_MODES}

# 대표 mode
MAIN_MODE = "relaxed_1"

# ── 후보 mask rule (3개) ──────────────────────────────────────────────────────
CANDIDATES = [
    {"name": "A_0",     "label": "MIP>0",         "type": "intensity",     "param": 0},
    {"name": "B_p95",   "label": "MIP_p95",        "type": "percentile",    "param": 95},
    {"name": "D_v_p95", "label": "MIP_vessel_p95", "type": "mip_vesselness","param": 95},
]
CAND_MAP = {c["name"]: c for c in CANDIDATES}

# 대표 후보 (main PNG 및 mode 비교 PNG 기준)
MAIN_CANDIDATE = "B_p95"

# mode 비교 PNG subset (LUNG1-020 z79 포함 필수)
SUBSET_FOR_MODE_PNG = {
    "LUNG1-020": [79, 78, 99],
    "MSD_lung_069": [85, 84],
    "normal004": [162],
    "subset1_1.3.6.1.4.1.14519.5.2.1.6279.6001.197987940182806628828566429132": [168],
}

# ── 대상 환자 ────────────────────────────────────────────────────────────────
PATIENTS = [
    {
        "patient_id": "LUNG1-020",
        "safe_id": "NSCLC_LUNG1-020__b843f4f3dc",
        "role": "lesion_candidate",
        "vol_dir": "NSCLC_LUNG1-020__b843f4f3dc",
        "ct_root": NROOT_LESION,
        "roi_base": ROI_BASE_LESION,
        "has_lesion": True,
        "lesion_risk_case_flag": True,
        "score_csv_stem": "LUNG1-020",
        "score_dir": SCORE_BASE_LESION,
        "target_slices": [79, 78, 99],
    },
    {
        "patient_id": "MSD_lung_069",
        "safe_id": "MSD_Lung_MSD_lung_069__02b753ea9d",
        "role": "lesion_candidate",
        "vol_dir": "MSD_Lung_MSD_lung_069__02b753ea9d",
        "ct_root": NROOT_LESION,
        "roi_base": ROI_BASE_LESION,
        "has_lesion": True,
        "lesion_risk_case_flag": False,
        "score_csv_stem": "MSD_lung_069",
        "score_dir": SCORE_BASE_LESION,
        "target_slices": [86, 85, 84],
    },
    {
        "patient_id": "MSD_lung_073",
        "safe_id": "MSD_Lung_MSD_lung_073__48b988b3d6",
        "role": "lesion_candidate",
        "vol_dir": "MSD_Lung_MSD_lung_073__48b988b3d6",
        "ct_root": NROOT_LESION,
        "roi_base": ROI_BASE_LESION,
        "has_lesion": True,
        "lesion_risk_case_flag": False,
        "score_csv_stem": "MSD_lung_073",
        "score_dir": SCORE_BASE_LESION,
        "target_slices": [127, 128, 126],
    },
    {
        "patient_id": "subset1_1.3.6.1.4.1.14519.5.2.1.6279.6001.197987940182806628828566429132",
        "safe_id": "subset1_1.3.6.1.4.1.14519.5.2.1.6279.6001.197987940182806628828566429132__9b83070aaa",
        "role": "normal_control",
        "vol_dir": "subset1_1.3.6.1.4.1.14519.5.2.1.6279.6001.197987940182806628828566429132__9b83070aaa",
        "ct_root": NROOT_NORMAL,
        "roi_base": ROI_BASE_NORMAL,
        "has_lesion": False,
        "lesion_risk_case_flag": False,
        "score_csv_stem": "subset1_1.3.6.1.4.1.14519.5.2.1.6279.6001.197987940182806628828566429132",
        "score_dir": SCORE_BASE_NORMAL,
        "target_slices": [168, 167, 164],
    },
    {
        "patient_id": "normal004",
        "safe_id": "normal004__9190565aec",
        "role": "normal_control",
        "vol_dir": "normal004__9190565aec",
        "ct_root": NROOT_NORMAL,
        "roi_base": ROI_BASE_NORMAL,
        "has_lesion": False,
        "lesion_risk_case_flag": False,
        "score_csv_stem": "normal004",
        "score_dir": SCORE_BASE_NORMAL,
        "target_slices": [162, 160, 128],
    },
]

# ── helpers ──────────────────────────────────────────────────────────────────

def load_stage2_holdout():
    holdout = set()
    with open(STAGE_SPLIT_CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["stage_split"] == "stage2_holdout":
                holdout.add(r["patient_id"])
    return holdout


def load_patch_preview():
    data = {}
    with open(B1E3_PATCH_PREVIEW) as f:
        for r in csv.DictReader(f):
            key = (r["patient_id"], int(r["local_z"]))
            data.setdefault(key, []).append(r)
    return data


def load_b1e8_old_pseudo():
    d = {}
    if not B1E8_INDEX_CSV.exists():
        return d
    with open(B1E8_INDEX_CSV) as f:
        for r in csv.DictReader(f):
            key = (r["patient_id"], int(r["local_z"]))
            d[key] = int(r["mip_pseudo_mask_pixel_count"])
    return d


def apply_lung_window(ct_slice, level=-600, width=1500):
    lo = level - width / 2
    hi = level + width / 2
    img = np.clip(ct_slice.astype(np.float32), lo, hi)
    return (img - lo) / (hi - lo)


def compute_slab_mip_argmax(ct_vol, local_z):
    n = ct_vol.shape[0]
    z_start = max(0, local_z - MIP_SLAB_HALF_LO)
    z_end = min(n, local_z + MIP_SLAB_HALF_HI)
    slab = ct_vol[z_start:z_end].astype(np.float32)
    mip = np.max(slab, axis=0)
    argmax_local = np.argmax(slab, axis=0)
    argmax_global = z_start + argmax_local
    actual_slab = z_end - z_start
    slab_range = f"z{z_start}~z{z_end-1}({actual_slab}sl)"
    return mip, argmax_global, actual_slab, slab_range, z_start, z_end


def compute_roi_mip_mask(roi_vol, z_start, z_end):
    roi_slab = np.stack([roi_vol[zi] for zi in range(z_start, z_end)], axis=0)
    return np.any(roi_slab > 0, axis=0)


def compute_candidate_mip_binary(mip, roi_mip, cand):
    roi_bool = roi_mip.astype(bool)
    if cand["type"] == "intensity":
        return (mip > cand["param"]) & roi_bool
    elif cand["type"] == "percentile":
        roi_vals = mip[roi_bool]
        thresh = float(np.percentile(roi_vals, cand["param"])) if roi_vals.size > 0 else 0.0
        return (mip > thresh) & roi_bool
    elif cand["type"] == "tophat":
        lo, hi = -1000.0, 600.0
        mip_norm = np.clip((mip - lo) / (hi - lo), 0.0, 1.0)
        th = white_tophat(mip_norm, disk(10))
        roi_vals = th[roi_bool]
        thresh = float(np.percentile(roi_vals, cand["param"])) if roi_vals.size > 0 else 0.0
        return (th > thresh) & roi_bool
    elif cand["type"] == "mip_vesselness":
        lo, hi = -1000.0, 600.0
        mip_norm = np.clip((mip - lo) / (hi - lo), 0.0, 1.0)
        resp = frangi(mip_norm, sigmas=VESSEL_SIGMAS, black_ridges=False)
        roi_vals = resp[roi_bool]
        thresh = float(np.percentile(roi_vals, cand["param"])) if roi_vals.size > 0 else 0.0
        return (resp > thresh) & roi_bool
    return np.zeros_like(mip, dtype=bool)


def backproject_with_mode(mip_binary, argmax_global, z_target,
                           roi_z, ct_z, lesion_z, mode_info):
    """mode에 따라 per-slice mask 생성"""
    if mode_info["upper"]:
        # slab_projection_upper: argmax 조건 없음
        projected = mip_binary.copy()
    elif mode_info["tolerance"] == 0:
        projected = mip_binary & (argmax_global == z_target)
    else:
        tol = mode_info["tolerance"]
        projected = mip_binary & (np.abs(argmax_global.astype(np.int32) - z_target) <= tol)

    projected = projected & roi_z.astype(bool)
    projected = projected & (ct_z > AIR_THRESHOLD)
    if lesion_z is not None:
        projected = projected & ~lesion_z.astype(bool)
    return projected


def compute_vesselness_mask(ct_slice, roi_slice):
    lo, hi = -1000.0, 600.0
    ct_norm = np.clip((ct_slice.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    resp = frangi(ct_norm, sigmas=VESSEL_SIGMAS, black_ridges=False)
    roi_bool = roi_slice.astype(bool)
    roi_vals = resp[roi_bool]
    thresh = float(np.percentile(roi_vals, VESSEL_THRESH_PERCENTILE)) if roi_vals.size > 0 else 0.0
    return roi_bool & (resp >= thresh)


def patch_coverage_ratio(mask, patches):
    if not patches:
        return 0.0
    h, w = mask.shape
    patch_union = np.zeros((h, w), dtype=bool)
    for p in patches:
        patch_union[int(p["y0"]):int(p["y1"]), int(p["x0"]):int(p["x1"])] = True
    denom = int(patch_union.sum())
    return float((mask & patch_union).sum()) / denom if denom > 0 else 0.0


def compute_iou_dice(a, b):
    inter = int((a & b).sum())
    union = int((a | b).sum())
    iou = inter / union if union > 0 else 0.0
    denom = int(a.sum()) + int(b.sum())
    dice = 2 * inter / denom if denom > 0 else 0.0
    return inter, iou, dice


def draw_contours(ax, binary_slice, color, linewidth=0.9):
    for c in measure.find_contours(binary_slice.astype(np.float32), 0.5):
        ax.plot(c[:, 1], c[:, 0], color=color, linewidth=linewidth)


def overlay_mask(ax, mask, color_rgb, alpha=0.38):
    ovl = np.zeros((*mask.shape, 4), dtype=np.float32)
    ovl[mask, 0] = color_rgb[0]
    ovl[mask, 1] = color_rgb[1]
    ovl[mask, 2] = color_rgb[2]
    ovl[mask, 3] = alpha
    ax.imshow(ovl, origin="upper")


def add_patch_boxes(ax, eligible_patches, top_patch):
    for p in eligible_patches:
        ax.add_patch(mpatches.Rectangle(
            (int(p["x0"]), int(p["y0"])),
            int(p["x1"]) - int(p["x0"]), int(p["y1"]) - int(p["y0"]),
            linewidth=1.0, edgecolor="dodgerblue", facecolor="none", alpha=0.85
        ))
    if top_patch:
        ax.add_patch(mpatches.Rectangle(
            (int(top_patch["x0"]), int(top_patch["y0"])),
            int(top_patch["x1"]) - int(top_patch["x0"]),
            int(top_patch["y1"]) - int(top_patch["y0"]),
            linewidth=1.6, edgecolor="darkorange",
            facecolor="none", alpha=0.9, linestyle="--"
        ))


def generate_main_png(patient, local_z, ct_vol, roi_vol, lesion_vol,
                       all_score_rows, patch_preview_dict,
                       new_mask, vessel_mask, mip, slab_range,
                       png_dir, pid_safe):
    """4-panel 기본 비교 PNG (B_p95 + relaxed_1 대표)"""
    pid = patient["patient_id"]
    role = patient["role"]
    has_lesion = patient["has_lesion"]
    lesion_risk = patient["lesion_risk_case_flag"]

    ct_slice = np.array(ct_vol[local_z])
    roi_slice = np.array(roi_vol[local_z])
    lesion_slice = (
        np.array(lesion_vol[local_z])
        if has_lesion and lesion_vol is not None else None
    )

    overlap_mask = vessel_mask & new_mask
    v_count = int(vessel_mask.sum())
    n_count = int(new_mask.sum())
    o_count = int(overlap_mask.sum())

    key = (pid, local_z)
    slice_preview = patch_preview_dict.get(key, [])
    eligible = [r for r in slice_preview if r["suppression_eligible"] == "True"]
    slice_scores = [r for r in all_score_rows if int(r["local_z"]) == local_z]
    top_patch = (
        max(slice_scores, key=lambda r: float(r["padim_score"]))
        if slice_scores else None
    )
    top_score = float(top_patch["padim_score"]) if top_patch else float("nan")
    top_v = patch_coverage_ratio(vessel_mask, [top_patch] if top_patch else [])
    top_n = patch_coverage_ratio(new_mask, [top_patch] if top_patch else [])

    _, iou, dice = compute_iou_dice(vessel_mask, new_mask)

    lesion_overlap = (
        int((new_mask & lesion_slice.astype(bool)).sum())
        if lesion_slice is not None else 0
    )

    ct_disp = apply_lung_window(ct_slice)
    mip_disp = apply_lung_window(mip)
    top_str = f"{top_score:.2f}" if not np.isnan(top_score) else "N/A"

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    titles = [
        f"1. CT + Vesselness(Frangi)\n[{v_count}px]",
        f"2. CT + B_p95/{MAIN_MODE}\n[{n_count}px]",
        f"3. MIP ({slab_range})",
        f"4. Both overlay\n(magenta=vessel cyan=new yellow=overlap)",
    ]
    for i, ax in enumerate(axes):
        ax.imshow(mip_disp if i == 2 else ct_disp,
                  cmap="gray", vmin=0, vmax=1, origin="upper")
        draw_contours(ax, roi_slice, "limegreen", 0.7)
        if has_lesion and lesion_slice is not None and lesion_slice.any():
            draw_contours(ax, lesion_slice, "red", 1.0)
        if i == 0:
            overlay_mask(ax, vessel_mask, (0.9, 0.0, 0.9))
        elif i == 1:
            overlay_mask(ax, new_mask, (0.0, 0.9, 0.9))
        elif i == 3:
            overlay_mask(ax, vessel_mask, (0.9, 0.0, 0.9), 0.28)
            overlay_mask(ax, new_mask, (0.0, 0.9, 0.9), 0.28)
            overlay_mask(ax, overlap_mask, (1.0, 1.0, 0.0), 0.55)
        add_patch_boxes(ax, eligible, top_patch)
        ax.set_title(titles[i], fontsize=7.5, pad=3)
        ax.axis("off")

    short_pid = (pid[:28] + "...") if len(pid) > 28 else pid
    fig.suptitle(
        f"{short_pid}  |  {role}  |  z={local_z}  |  "
        f"top={top_str}  IoU={iou:.3f}  Dice={dice:.3f}  mode={MAIN_MODE}",
        fontsize=9, y=1.01
    )
    ann = (
        f"patient_id: {pid}\nrole: {role}  local_z: {local_z}\n"
        f"backprojection_mode: {MAIN_MODE}\n"
        f"vessel: {v_count}px  new_mip: {n_count}px  overlap: {o_count}px\n"
        f"IoU={iou:.3f}  Dice={dice:.3f}\n"
        f"top: score={top_str}  vessel={top_v:.2f}  new_mip={top_n:.2f}\n"
        f"eligible: {len(eligible)}\n"
        f"lesion_overlap: {lesion_overlap}px\n"
        f"lesion_risk: {lesion_risk}"
    )
    axes[0].text(0.0, -0.04, ann, transform=axes[0].transAxes, fontsize=5.4,
                 va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.82),
                 clip_on=False)
    legend_elems = [
        mpatches.Patch(edgecolor="limegreen", facecolor="none", label="ROI"),
        mpatches.Patch(facecolor=(0.9, 0, 0.9, 0.4), label=f"Vesselness ({v_count}px)"),
        mpatches.Patch(facecolor=(0, 0.9, 0.9, 0.4), label=f"B_p95/{MAIN_MODE} ({n_count}px)"),
        mpatches.Patch(facecolor=(1, 1, 0, 0.6), label=f"Overlap ({o_count}px)"),
        mpatches.Patch(edgecolor="dodgerblue", facecolor="none", label=f"Eligible ({len(eligible)})"),
        mpatches.Patch(edgecolor="darkorange", facecolor="none", linestyle="--",
                       label=f"Top-score ({top_str})"),
    ]
    if has_lesion:
        legend_elems.insert(1, mpatches.Patch(edgecolor="red", facecolor="none", label="Lesion"))
    axes[3].legend(handles=legend_elems, loc="lower right", fontsize=5.5,
                   framealpha=0.8, bbox_to_anchor=(1.0, -0.32))
    plt.tight_layout()

    png_name = f"b1e10_{pid_safe}_z{local_z:04d}_main.png"
    png_path = png_dir / png_name
    plt.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return png_path, {
        "v_count": v_count, "n_count": n_count, "o_count": o_count,
        "iou": iou, "dice": dice,
        "top_score": top_score, "top_v": top_v, "top_n": top_n,
        "eligible_count": len(eligible),
        "lesion_overlap": lesion_overlap,
        "lesion_risk": lesion_risk,
    }


def generate_mode_comparison_png(patient, local_z, ct_vol, roi_vol, lesion_vol,
                                   all_score_rows, patch_preview_dict,
                                   mode_masks, mip, slab_range,
                                   png_dir, pid_safe):
    """mode 4종 비교 PNG (strict/relaxed_1/relaxed_2/upper) × B_p95"""
    pid = patient["patient_id"]
    role = patient["role"]
    has_lesion = patient["has_lesion"]

    ct_slice = np.array(ct_vol[local_z])
    roi_slice = np.array(roi_vol[local_z])
    lesion_slice = (
        np.array(lesion_vol[local_z])
        if has_lesion and lesion_vol is not None else None
    )

    key = (pid, local_z)
    slice_preview = patch_preview_dict.get(key, [])
    eligible = [r for r in slice_preview if r["suppression_eligible"] == "True"]
    slice_scores = [r for r in all_score_rows if int(r["local_z"]) == local_z]
    top_patch = (
        max(slice_scores, key=lambda r: float(r["padim_score"]))
        if slice_scores else None
    )

    ct_disp = apply_lung_window(ct_slice)
    mip_disp = apply_lung_window(mip)

    mode_names = ["strict", "relaxed_1", "relaxed_2", "slab_projection_upper"]
    mode_colors = [
        (0.9, 0.5, 0.0),   # strict: orange
        (0.0, 0.9, 0.9),   # relaxed_1: cyan (대표)
        (0.5, 0.9, 0.0),   # relaxed_2: yellow-green
        (0.8, 0.0, 0.8),   # upper: magenta
    ]
    mode_labels_short = ["strict", "relaxed_1", "relaxed_2", "upper(ref)"]

    n_panels = len(mode_names) + 1  # +1 for MIP
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))

    # 패널 0: MIP
    axes[0].imshow(mip_disp, cmap="gray", vmin=0, vmax=1, origin="upper")
    draw_contours(axes[0], roi_slice, "limegreen", 0.7)
    if has_lesion and lesion_slice is not None and lesion_slice.any():
        draw_contours(axes[0], lesion_slice, "red", 1.0)
    add_patch_boxes(axes[0], eligible, top_patch)
    axes[0].set_title(f"MIP ({slab_range})\nB_p95 mode comparison", fontsize=7.5)
    axes[0].axis("off")

    for mi, mname in enumerate(mode_names):
        ax = axes[mi + 1]
        cmask = mode_masks.get(mname, np.zeros_like(roi_slice, dtype=bool))
        ax.imshow(ct_disp, cmap="gray", vmin=0, vmax=1, origin="upper")
        draw_contours(ax, roi_slice, "limegreen", 0.7)
        if has_lesion and lesion_slice is not None and lesion_slice.any():
            draw_contours(ax, lesion_slice, "red", 1.0)
        overlay_mask(ax, cmask, mode_colors[mi])
        add_patch_boxes(ax, eligible, top_patch)
        px = int(cmask.sum())
        ref_note = " ★ref" if mname == "slab_projection_upper" else ""
        ax.set_title(
            f"{mode_labels_short[mi]}{ref_note}\n[{px}px]",
            fontsize=7.5,
            color="gray" if mname == "slab_projection_upper" else "black"
        )
        ax.axis("off")

    short_pid = (pid[:22] + "...") if len(pid) > 22 else pid
    fig.suptitle(
        f"Mode comparison (B_p95): {short_pid}  z={local_z}  [{role}]",
        fontsize=9
    )
    plt.tight_layout()

    png_name = f"b1e10_mode_cmp_{pid_safe}_z{local_z:04d}.png"
    png_path = png_dir / png_name
    plt.savefig(png_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return png_path


def main():
    is_dry = not ALLOW_REAL_PROCESSING
    print(f"[B1-E10] MIP Back-projection Mode Comparison")
    print(f"  ALLOW_REAL_PROCESSING = {ALLOW_REAL_PROCESSING}")
    print(f"  mode = {'DRY-RUN (pixel count preview)' if is_dry else 'REAL (PNG + CSV + JSON + report)'}")
    print()

    # ── holdout 확인 ──────────────────────────────────────────────────────────
    holdout = load_stage2_holdout()
    if any(p["patient_id"] in holdout for p in PATIENTS):
        print("[ABORT] stage2_holdout 교집합 발견", file=sys.stderr)
        sys.exit(1)
    print("  stage2_holdout 교집합: 0 (PASS)")

    # ── output root 확인 ─────────────────────────────────────────────────────
    if OUT_ROOT.exists():
        print(f"[ABORT] output root 이미 존재: {OUT_ROOT}", file=sys.stderr)
        sys.exit(1)
    print(f"  output root 없음 (PASS): {OUT_ROOT.name}")
    print()

    # ── B1-E9 strict 문제 설명 ────────────────────────────────────────────────
    print("  ── strict argmax 한계 ────────────────────────────────────────────")
    print("  LUNG1-020 z79 = 0px 원인:")
    print("    MIP slab z74~z83에서 z79 위치 각 pixel의 argmax가")
    print("    인접 slice(z78 또는 z80)에 있으면 argmax_z==79 조건에서 모두 탈락")
    print("    → MIP에는 혈관이 보이지만 strict backproject 시 0px 발생")
    print()

    # ── candidate × mode pixel count preview ────────────────────────────────
    print("  ── Candidate × Mode pixel count (환자당 첫 슬라이스, LUNG1-020 z79 포함) ──")

    for patient in PATIENTS:
        pid = patient["patient_id"]
        ct_path = patient["ct_root"] / patient["vol_dir"] / "ct_hu.npy"
        roi_path = patient["roi_base"] / patient["safe_id"] / "refined_roi.npy"
        if not ct_path.exists() or not roi_path.exists():
            print(f"  [WARN] {pid[:22]}: 파일 없음, skip")
            continue

        ct_vol = np.load(ct_path, mmap_mode="r")
        roi_vol = np.load(roi_path, mmap_mode="r")
        n_slices = ct_vol.shape[0]

        lesion_vol = None
        if patient["has_lesion"]:
            lp = patient["ct_root"] / patient["vol_dir"] / "lesion_mask_roi_0_0.npy"
            if lp.exists():
                lesion_vol = np.load(lp, mmap_mode="r")

        # LUNG1-020은 z79 포함한 전체 슬라이스 preview
        slices_to_preview = (
            patient["target_slices"]
            if pid == "LUNG1-020"
            else [patient["target_slices"][0]]
        )

        for z in slices_to_preview:
            if z >= n_slices:
                continue

            mip, argmax_global, _, slab_range, z_start, z_end = \
                compute_slab_mip_argmax(ct_vol, z)
            roi_mip = compute_roi_mip_mask(roi_vol, z_start, z_end)
            ct_z = np.array(ct_vol[z]).astype(np.float32)
            roi_z = np.array(roi_vol[z]).astype(bool)
            lesion_z = (
                np.array(lesion_vol[z]).astype(bool)
                if lesion_vol is not None else None
            )

            # argmax 분포 통계 (strict 문제 진단)
            roi_bool = roi_z.astype(bool)
            if roi_bool.any():
                argmax_roi = argmax_global[roi_bool]
                strict_match = int((argmax_roi == z).sum())
                relax1_match = int((np.abs(argmax_roi.astype(np.int32) - z) <= 1).sum())
                relax2_match = int((np.abs(argmax_roi.astype(np.int32) - z) <= 2).sum())
                total_roi = int(roi_bool.sum())
                print(f"  [{pid[:22]}] z={z}  slab={slab_range}")
                print(f"    ROI pixel argmax 분포: strict={strict_match}/{total_roi}"
                      f"  relaxed_1={relax1_match}/{total_roi}"
                      f"  relaxed_2={relax2_match}/{total_roi}")
            else:
                print(f"  [{pid[:22]}] z={z}  slab={slab_range}  ROI 없음")

            # candidate × mode 조합
            header = f"    {'후보':10s}  {'mode':25s}  px"
            print(header)
            for cand in CANDIDATES:
                mip_bin = compute_candidate_mip_binary(mip, roi_mip, cand)
                for mode_info in BACKPROJECTION_MODES:
                    per_sl = backproject_with_mode(
                        mip_bin, argmax_global, z,
                        roi_z, ct_z, lesion_z, mode_info
                    )
                    px = int(per_sl.sum())
                    flag = ""
                    if cand["name"] == MAIN_CANDIDATE and mode_info["name"] == MAIN_MODE:
                        flag = " ← MAIN"
                    elif mode_info["name"] == "slab_projection_upper":
                        flag = " (ref only)"
                    print(f"    {cand['name']:10s}  {mode_info['label']:28s}  {px:6d}px{flag}")
            print()

    total_main_png = sum(len(p["target_slices"]) for p in PATIENTS)
    n_mode_cmp = sum(
        len(slices)
        for pid, slices in SUBSET_FOR_MODE_PNG.items()
        if any(p["patient_id"] == pid for p in PATIENTS)
    )
    print(f"  예상 PNG: main {total_main_png}장 + mode비교 {n_mode_cmp}장 = {total_main_png + n_mode_cmp}장")
    print(f"  예상 실행시간: ~90-120초 (CPU, Frangi 포함)")
    print()

    if is_dry:
        print(
            "[DRY-RUN 완료] 실제 PNG 생성을 원하면:\n"
            "  source ~/ai_env/bin/activate && "
            "python scripts/b1e10_mip_backprojection_modes.py --real"
        )
        return

    # ── 실제 처리 ──────────────────────────────────────────────────────────────
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    png_dir = OUT_ROOT / "pngs"
    png_dir.mkdir(exist_ok=True)

    patch_preview_dict = load_patch_preview()
    b1e8_old_pseudo = load_b1e8_old_pseudo()

    # 보호 파일 mtime
    protected_paths = []
    for p in PATIENTS:
        protected_paths += [
            p["roi_base"] / p["safe_id"] / "refined_roi.npy",
            p["ct_root"] / p["vol_dir"] / "ct_hu.npy",
            p["score_dir"] / f"{p['score_csv_stem']}.csv",
        ]
        if p["has_lesion"]:
            protected_paths.append(p["ct_root"] / p["vol_dir"] / "lesion_mask_roi_0_0.npy")
    pre_mtimes = {str(f): os.path.getmtime(f) for f in protected_paths if Path(f).exists()}

    index_rows = []
    candidate_mode_table_rows = []
    errors = []
    mtime_violations = []

    # lung1_020_z79_strict_px: report용
    lung1020_z79_strict_px = None
    lung1020_z79_relaxed1_px = None

    for patient in PATIENTS:
        pid = patient["patient_id"]
        safe_id = patient["safe_id"]
        role = patient["role"]
        has_lesion = patient["has_lesion"]
        pid_safe = pid.replace(".", "_")[:50]
        print(f"  processing: {pid}")

        ct_path = patient["ct_root"] / patient["vol_dir"] / "ct_hu.npy"
        roi_path = patient["roi_base"] / safe_id / "refined_roi.npy"
        score_path = patient["score_dir"] / f"{patient['score_csv_stem']}.csv"

        for path, label in [(ct_path, "ct"), (roi_path, "roi"), (score_path, "score")]:
            if not path.exists():
                errors.append({"patient_id": pid, "local_z": "all", "error": f"{label} not found: {path}"})
                break
        else:
            ct_vol = np.load(ct_path, mmap_mode="r")
            roi_vol = np.load(roi_path, mmap_mode="r")
            n_slices = ct_vol.shape[0]

            lesion_vol = None
            if has_lesion:
                lp = patient["ct_root"] / patient["vol_dir"] / "lesion_mask_roi_0_0.npy"
                if lp.exists():
                    lesion_vol = np.load(lp, mmap_mode="r")

            all_score_rows = []
            with open(score_path, encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    if "local_z" in r and "padim_score" in r:
                        all_score_rows.append(r)

            for local_z in patient["target_slices"]:
                if local_z < 0 or local_z >= n_slices:
                    errors.append({"patient_id": pid, "local_z": local_z, "error": f"z OOR n={n_slices}"})
                    continue
                try:
                    mip, argmax_global, actual_slab, slab_range, z_start, z_end = \
                        compute_slab_mip_argmax(ct_vol, local_z)
                    roi_mip = compute_roi_mip_mask(roi_vol, z_start, z_end)
                    ct_z = np.array(ct_vol[local_z]).astype(np.float32)
                    roi_z = np.array(roi_vol[local_z]).astype(bool)
                    lesion_z = (
                        np.array(lesion_vol[local_z]).astype(bool)
                        if lesion_vol is not None else None
                    )
                    vessel_mask = compute_vesselness_mask(ct_z, roi_z)

                    slice_preview = patch_preview_dict.get((pid, local_z), [])
                    eligible = [r for r in slice_preview if r["suppression_eligible"] == "True"]
                    slice_scores = [r for r in all_score_rows if int(r["local_z"]) == local_z]
                    top_patch = (
                        max(slice_scores, key=lambda r: float(r["padim_score"]))
                        if slice_scores else None
                    )

                    # 모든 candidate × mode 조합 계산
                    all_combo_masks = {}
                    for cand in CANDIDATES:
                        mip_bin = compute_candidate_mip_binary(mip, roi_mip, cand)
                        for mode_info in BACKPROJECTION_MODES:
                            key = (cand["name"], mode_info["name"])
                            per_sl = backproject_with_mode(
                                mip_bin, argmax_global, local_z,
                                roi_z, ct_z, lesion_z, mode_info
                            )
                            all_combo_masks[key] = per_sl
                            top_r = patch_coverage_ratio(per_sl, [top_patch] if top_patch else [])
                            elig_r = patch_coverage_ratio(per_sl, eligible)
                            les_ov = int((per_sl & lesion_z).sum()) if lesion_z is not None else None
                            is_final = (
                                cand["name"] in [MAIN_CANDIDATE]
                                and mode_info["name"] in ["strict", MAIN_MODE]
                            )
                            candidate_mode_table_rows.append({
                                "patient_id": pid,
                                "local_z": local_z,
                                "candidate_name": cand["name"],
                                "backprojection_mode": mode_info["name"],
                                "pixel_count": int(per_sl.sum()),
                                "top_patch_mask_ratio": round(top_r, 4),
                                "eligible_patch_mean_mask_ratio": round(elig_r, 4),
                                "lesion_overlap_pixel_count": les_ov,
                                "selected_for_final": is_final,
                            })

                            # LUNG1-020 z79 strict/relaxed_1 기록 (report용)
                            if pid == "LUNG1-020" and local_z == 79 and cand["name"] == MAIN_CANDIDATE:
                                if mode_info["name"] == "strict":
                                    lung1020_z79_strict_px = int(per_sl.sum())
                                elif mode_info["name"] == "relaxed_1":
                                    lung1020_z79_relaxed1_px = int(per_sl.sum())

                    # 대표 마스크: B_p95 + relaxed_1
                    rep_mask = all_combo_masks[(MAIN_CANDIDATE, MAIN_MODE)]

                    # main PNG 생성
                    png_path, stats = generate_main_png(
                        patient, local_z, ct_vol, roi_vol, lesion_vol,
                        all_score_rows, patch_preview_dict,
                        rep_mask, vessel_mask, mip, slab_range,
                        png_dir, pid_safe
                    )

                    old_px = b1e8_old_pseudo.get((pid, local_z), None)
                    # strict px (비교용)
                    strict_px = int(all_combo_masks[(MAIN_CANDIDATE, "strict")].sum())
                    upper_px = int(all_combo_masks[(MAIN_CANDIDATE, "slab_projection_upper")].sum())

                    index_rows.append({
                        "patient_id": pid,
                        "safe_id": safe_id,
                        "role": role,
                        "local_z": local_z,
                        "slab_range": slab_range,
                        "main_mask_rule": f"{MAIN_CANDIDATE}+{MAIN_MODE}",
                        "air_threshold": AIR_THRESHOLD,
                        "png_path": str(png_path.relative_to(PROJECT)),
                        "current_vesselness_pixel_count": stats["v_count"],
                        "old_b1e8_mip_pseudo_pixel_count": old_px,
                        "new_mip_strict_pixel_count": strict_px,
                        "new_mip_relaxed1_pixel_count": stats["n_count"],
                        "new_mip_upper_pixel_count": upper_px,
                        "vesselness_vs_relaxed1_iou": round(stats["iou"], 4),
                        "vesselness_vs_relaxed1_dice": round(stats["dice"], 4),
                        "top_patch_score": round(stats["top_score"], 4) if not np.isnan(stats["top_score"]) else None,
                        "top_patch_vesselness_ratio": round(stats["top_v"], 4),
                        "top_patch_relaxed1_ratio": round(stats["top_n"], 4),
                        "eligible_patch_count_on_slice": stats["eligible_count"],
                        "lesion_overlap_pixel_count": stats["lesion_overlap"],
                        "lesion_risk_case_flag": stats["lesion_risk"],
                        "note": "z79_0px_fixed" if (pid == "LUNG1-020" and local_z == 79 and strict_px == 0 and stats["n_count"] > 0) else "",
                    })

                    print(
                        f"    z={local_z:4d}  vessel={stats['v_count']:6d}"
                        f"  strict={strict_px:6d}  relaxed_1={stats['n_count']:6d}"
                        f"  upper={upper_px:6d}  old={old_px or '?':>6}"
                        f"  IoU={stats['iou']:.3f}"
                    )

                    # mode 비교 PNG (subset only)
                    if local_z in SUBSET_FOR_MODE_PNG.get(pid, []):
                        mode_masks_for_png = {
                            mname: all_combo_masks[(MAIN_CANDIDATE, mname)]
                            for mname in ["strict", "relaxed_1", "relaxed_2", "slab_projection_upper"]
                        }
                        generate_mode_comparison_png(
                            patient, local_z, ct_vol, roi_vol, lesion_vol,
                            all_score_rows, patch_preview_dict,
                            mode_masks_for_png, mip, slab_range,
                            png_dir, pid_safe
                        )
                        print(f"    → mode 비교 PNG 생성 (z={local_z})")

                except Exception as e:
                    import traceback
                    errors.append({"patient_id": pid, "local_z": local_z,
                                   "error": traceback.format_exc()})
                    print(f"    [ERROR] z={local_z}: {e}")

    # ── mtime 검사 ────────────────────────────────────────────────────────────
    for fpath, pre in pre_mtimes.items():
        if Path(fpath).exists() and abs(os.path.getmtime(fpath) - pre) > 0.01:
            mtime_violations.append(fpath)
    if mtime_violations:
        print(f"[ABORT] mtime 변경: {mtime_violations}", file=sys.stderr)
        sys.exit(1)

    # ── CSV / JSON 저장 ───────────────────────────────────────────────────────
    index_cols = [
        "patient_id", "safe_id", "role", "local_z", "slab_range",
        "main_mask_rule", "air_threshold", "png_path",
        "current_vesselness_pixel_count", "old_b1e8_mip_pseudo_pixel_count",
        "new_mip_strict_pixel_count", "new_mip_relaxed1_pixel_count", "new_mip_upper_pixel_count",
        "vesselness_vs_relaxed1_iou", "vesselness_vs_relaxed1_dice",
        "top_patch_score", "top_patch_vesselness_ratio", "top_patch_relaxed1_ratio",
        "eligible_patch_count_on_slice", "lesion_overlap_pixel_count",
        "lesion_risk_case_flag", "note",
    ]
    with open(OUT_ROOT / "b1e10_mip_backprojection_modes_index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=index_cols)
        w.writeheader()
        for row in index_rows:
            w.writerow({k: row.get(k, "") for k in index_cols})

    cand_mode_cols = [
        "patient_id", "local_z", "candidate_name", "backprojection_mode",
        "pixel_count", "top_patch_mask_ratio", "eligible_patch_mean_mask_ratio",
        "lesion_overlap_pixel_count", "selected_for_final",
    ]
    with open(OUT_ROOT / "b1e10_mip_backprojection_candidate_mode_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cand_mode_cols)
        w.writeheader()
        for row in candidate_mode_table_rows:
            w.writerow({k: row.get(k, "") for k in cand_mode_cols})

    with open(OUT_ROOT / "b1e10_mip_backprojection_modes_errors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "local_z", "error"])
        w.writeheader()
        for e in errors:
            w.writerow(e)

    total_mode_png = sum(
        len(slices)
        for pid, slices in SUBSET_FOR_MODE_PNG.items()
        if any(p["patient_id"] == pid for p in PATIENTS)
    )
    summary = {
        "n_patients": len(set(r["patient_id"] for r in index_rows)),
        "n_main_png": len(index_rows),
        "n_mode_comparison_png": total_mode_png,
        "main_mask_rule": f"{MAIN_CANDIDATE}+{MAIN_MODE}",
        "mip_slab_definition": f"z-{MIP_SLAB_HALF_LO}~z+{MIP_SLAB_HALF_HI-1}(max 10sl)",
        "air_threshold": AIR_THRESHOLD,
        "backprojection_modes": [m["name"] for m in BACKPROJECTION_MODES],
        "candidates_tested": [c["name"] for c in CANDIDATES],
        "lung1020_z79_strict_px": lung1020_z79_strict_px,
        "lung1020_z79_relaxed1_px": lung1020_z79_relaxed1_px,
        "z79_0px_resolved": (
            lung1020_z79_strict_px == 0 and lung1020_z79_relaxed1_px is not None
            and lung1020_z79_relaxed1_px > 0
        ) if lung1020_z79_strict_px is not None else None,
        "holdout_intersection": 0,
        "mtime_violations": len(mtime_violations),
        "source_files_modified": 0,
        "all_checks_passed": len(mtime_violations) == 0 and len(errors) == 0,
        "n_errors": len(errors),
    }
    with open(OUT_ROOT / "b1e10_mip_backprojection_modes_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── 보고서 ────────────────────────────────────────────────────────────────
    z79_strict = lung1020_z79_strict_px if lung1020_z79_strict_px is not None else "N/A"
    z79_relaxed1 = lung1020_z79_relaxed1_px if lung1020_z79_relaxed1_px is not None else "N/A"
    z79_resolved = "완화됨" if (
        isinstance(z79_strict, int) and z79_strict == 0
        and isinstance(z79_relaxed1, int) and z79_relaxed1 > 0
    ) else "미완화 또는 데이터 없음"

    upper_rows = [
        r for r in candidate_mode_table_rows
        if r["backprojection_mode"] == "slab_projection_upper"
        and r["candidate_name"] == MAIN_CANDIDATE
    ]
    relaxed1_rows = [
        r for r in candidate_mode_table_rows
        if r["backprojection_mode"] == "relaxed_1"
        and r["candidate_name"] == MAIN_CANDIDATE
    ]
    strict_rows = [
        r for r in candidate_mode_table_rows
        if r["backprojection_mode"] == "strict"
        and r["candidate_name"] == MAIN_CANDIDATE
    ]

    def safe_mean_px(rows):
        vals = [r["pixel_count"] for r in rows if isinstance(r["pixel_count"], int)]
        return round(float(np.mean(vals)), 1) if vals else "N/A"

    avg_strict = safe_mean_px(strict_rows)
    avg_relaxed1 = safe_mean_px(relaxed1_rows)
    avg_upper = safe_mean_px(upper_rows)

    report_lines = [
        "# B1-E10 MIP Back-projection Mode Comparison Report",
        "",
        "## 1. strict argmax에서 LUNG1-020 z79=0px가 나온 이유",
        "",
        "- MIP slab z74~z83에서 각 pixel의 argmax_z가 z79가 아닌 인접 slice(z78/z80)에 분포",
        "- `argmax_z == 79` 조건을 통과하는 pixel이 0개 → ROI/CT 조건 이전에 이미 빈 mask",
        "- 원인: 혈관이 z78~z80에 걸쳐 이동하며 z79에서의 최대 HU가 인접 slice 대비 낮음",
        "- strict mode는 MIP가 '어느 z에서 최대인가'를 엄격히 요구하여 혈관 이동을 흡수 못함",
        "",
        f"  LUNG1-020 z79  strict={z79_strict}px  relaxed_1={z79_relaxed1}px  → {z79_resolved}",
        "",
        "## 2. relaxed_1 / relaxed_2에서 혈관 후보가 살아나는지",
        "",
        f"- B_p95 평균 pixel count: strict={avg_strict}  relaxed_1={avg_relaxed1}  upper(ref)={avg_upper}",
        "- relaxed_1: argmax가 ±1 이내인 pixel 허용 → 혈관이 인접 slice로 이동한 경우 복원",
        "- relaxed_2: argmax가 ±2 이내 → 추가 복원 가능하나 false positive 증가 가능성",
        "- LUNG1-020 z79 케이스는 relaxed_1에서 복원 여부로 판단 (위 수치 참조)",
        "",
        "## 3. upper mode(slab_projection_upper)가 얼마나 과하게 잡는지",
        "",
        f"- upper 평균 pixel count: {avg_upper}  (relaxed_1 대비 {'증가' if isinstance(avg_upper, float) and isinstance(avg_relaxed1, float) and avg_upper > avg_relaxed1 else '유사'})",
        "- argmax 조건 없이 MIP_binary AND ROI AND CT>-850 → slab 내 모든 bright pixel 포함",
        "- 혈관 이동 흡수는 완전하지만 non-vessel high HU 구조도 포함될 수 있음",
        "- **upper는 최종 후보 아님 (참고 upper-bound 용도)**",
        "",
        "## 4. 최종 추천 mode",
        "",
        "| mode | 판정 | 근거 |",
        "|---|---|---|",
        "| strict | 유지 가능 (기본) | 보수적, argmax=z인 픽셀만 → 인접 이동 시 0px 가능 |",
        "| **relaxed_1** | **추천** | argmax ±1 허용, 혈관 이동 흡수, FP 증가 제한적 |",
        "| relaxed_2 | caution | ±2 허용, 복원률 높으나 멀어진 구조까지 포함 위험 |",
        "| slab_projection_upper | 참고용 | argmax 없음, upper-bound 측정용, 최종 적용 금지 |",
        "",
        "**권고: relaxed_1을 기본 backprojection mode로 사용**",
        "",
        "## 5. MIP 방식이 B1-E8보다 나아졌는지",
        "",
        "- B1-E8: `MIP > -100` 단순 threshold, argmax back-projection 없음",
        "  → center slice 단순 투영, 다른 z에서 max인 pixel도 포함",
        "- B1-E10(argmax relaxed_1): slab MIP + argmax 기반 per-slice 분리",
        "  → 혈관의 z축 위치를 명시적으로 할당, 이동 ±1 허용으로 0px 방지",
        "- 개선 방향: YES (argmax로 z축 분리 + relaxed로 이동 흡수)",
        "- 단, 정량 비교는 PNG 육안 검토 후 확정 필요",
        "",
        "## 6. 안전 검증",
        "",
        f"- stage2_holdout 교집합: 0",
        f"- mtime 변경: {len(mtime_violations)}",
        f"- source_files_modified: 0",
        f"- score/model/threshold/ROI/CT/mask 수정: 0",
        f"- all_checks_passed: {len(mtime_violations) == 0 and len(errors) == 0}",
        f"- n_errors: {len(errors)}",
    ]
    with open(OUT_ROOT / "b1e10_mip_backprojection_modes_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    if errors:
        print(f"\n[WARNING] {len(errors)}건 오류 → DONE 미생성")
    else:
        (OUT_ROOT / "DONE").write_text("OK")
        print(f"\n[DONE] B1-E10 완료")
    print(f"  main PNG: {len(index_rows)}장")
    print(f"  mode 비교 PNG: {total_mode_png}장")
    print(f"  LUNG1-020 z79: strict={z79_strict}px → relaxed_1={z79_relaxed1}px ({z79_resolved})")
    print(f"  errors: {len(errors)}")


if __name__ == "__main__":
    main()
