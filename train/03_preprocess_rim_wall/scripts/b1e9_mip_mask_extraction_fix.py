#!/usr/bin/env python3
"""
B1-E9: MIP mask extraction fix visual check
목적: B1-E8 MIP pseudo-mask 개선 (argmax back-projection + 다양한 threshold 후보)
실행:
  --dry-run   후보 pixel count preview (CT 로드, PNG 미생성)
  --real      ALLOW_REAL_PROCESSING=True 시에만 PNG 생성

B1-E8 문제:
1. MIP > -100 threshold: center slice 단순 투영 (argmax 없음)
2. 비조영 CT 혈관 누락 가능성

새 방식:
- argmax_z map: slab 내 각 pixel에서 max값이 나온 실제 z
- per_slice_mask[z] = MIP_binary AND argmax_z==z AND roi AND ct>air_thresh AND ~lesion
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
    / "b1e9_mip_mask_extraction_fix_v1"
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
MIP_SLAB_HALF_LO = 5     # z-5 (B1-E8 동일)
MIP_SLAB_HALF_HI = 5     # z+4 (exclusive z+5)
AIR_THRESHOLD = -850     # per-slice CT HU 하한

# current vesselness (B1-E7/E8 동일)
VESSEL_SIGMAS = (0.5, 1.0, 1.5, 2.0)
VESSEL_THRESH_PERCENTILE = 75

# B1-E8 old pseudo (비교 기준)
OLD_MIP_HU_THRESH = -100

# ── 후보 정의 ──────────────────────────────────────────────────────────────
# type: intensity / percentile / tophat / mip_vesselness
CANDIDATES = [
    # Candidate A: intensity threshold
    {"name": "A_neg200", "label": "MIP>-200",       "type": "intensity",    "param": -200},
    {"name": "A_neg100", "label": "MIP>-100",        "type": "intensity",    "param": -100},  # B1-E8 기존
    {"name": "A_0",      "label": "MIP>0",            "type": "intensity",    "param": 0},
    {"name": "A_50",     "label": "MIP>50",           "type": "intensity",    "param": 50},
    {"name": "A_100",    "label": "MIP>100",          "type": "intensity",    "param": 100},
    # Candidate B: percentile
    {"name": "B_p90",    "label": "MIP_p90",          "type": "percentile",   "param": 90},
    {"name": "B_p95",    "label": "MIP_p95",          "type": "percentile",   "param": 95},
    {"name": "B_p97",    "label": "MIP_p97.5",        "type": "percentile",   "param": 97.5},
    {"name": "B_p99",    "label": "MIP_p99",          "type": "percentile",   "param": 99},
    # Candidate C: top-hat
    {"name": "C_th_p95", "label": "TopHat_p95",       "type": "tophat",       "param": 95},
    {"name": "C_th_p97", "label": "TopHat_p97.5",     "type": "tophat",       "param": 97.5},
    # Candidate D: MIP 2D vesselness
    {"name": "D_v_p90",  "label": "MIP_vessel_p90",   "type": "mip_vesselness","param": 90},
    {"name": "D_v_p95",  "label": "MIP_vessel_p95",   "type": "mip_vesselness","param": 95},
    {"name": "D_v_p97",  "label": "MIP_vessel_p97.5", "type": "mip_vesselness","param": 97.5},
    {"name": "D_v_p99",  "label": "MIP_vessel_p99",   "type": "mip_vesselness","param": 99},
]

# 대표 3개 (dry-run preview 후 확정; 기본값은 초기 추천)
FINAL_CANDIDATES = ["A_0", "B_p95", "D_v_p95"]

# subset 비교 PNG 대상 (후보 3개 비교 패널)
SUBSET_FOR_CANDIDATE_PNG = {
    "LUNG1-020": [78, 99],
    "MSD_lung_069": [84, 85],
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
    data: dict = {}
    with open(B1E3_PATCH_PREVIEW) as f:
        for r in csv.DictReader(f):
            key = (r["patient_id"], int(r["local_z"]))
            data.setdefault(key, []).append(r)
    return data


def load_b1e8_old_pseudo():
    """B1-E8 old pseudo pixel count 참조용 (read-only)"""
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
    argmax_local = np.argmax(slab, axis=0)   # 0 ~ slab_size-1
    argmax_global = z_start + argmax_local   # 실제 z 인덱스
    actual_slab = z_end - z_start
    slab_range = f"z{z_start}~z{z_end-1}({actual_slab}sl)"
    return mip, argmax_global, actual_slab, slab_range, z_start, z_end


def compute_roi_mip_mask(roi_vol, z_start, z_end):
    """slab 범위에서 ROI OR projection"""
    roi_slab = np.stack([roi_vol[zi] for zi in range(z_start, z_end)], axis=0)
    return np.any(roi_slab > 0, axis=0)


def compute_candidate_mip_binary(mip, roi_mip, cand):
    """MIP slice에서 후보 binary mask 계산 (ROI 내부 기준)"""
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


def argmax_backproject(mip_binary, argmax_global, z_target,
                        roi_z, ct_z, lesion_z=None):
    """argmax_z == z_target인 pixel만 남겨 per-slice mask 생성"""
    projected = mip_binary & (argmax_global == z_target)
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
                       selected_new_mask, mip, slab_range,
                       vessel_mask, png_dir, pid_safe):
    """4-panel 기본 비교 PNG"""
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

    overlap_mask = vessel_mask & selected_new_mask
    v_count = int(vessel_mask.sum())
    n_count = int(selected_new_mask.sum())
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
    top_n = patch_coverage_ratio(selected_new_mask, [top_patch] if top_patch else [])

    _, iou, dice = compute_iou_dice(vessel_mask, selected_new_mask)

    lesion_overlap = int((selected_new_mask & lesion_slice.astype(bool)).sum()) if lesion_slice is not None else 0
    lesion_ratio = lesion_overlap / max(1, int(lesion_slice.sum())) if lesion_slice is not None else 0.0

    ct_disp = apply_lung_window(ct_slice)
    mip_disp = apply_lung_window(mip)
    top_str = f"{top_score:.2f}" if not np.isnan(top_score) else "N/A"

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    titles = [
        f"1. CT + Vesselness(Frangi)\n[{v_count}px]",
        f"2. CT + New MIP-argmax mask\n[{n_count}px]",
        f"3. MIP image ({slab_range})",
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
            overlay_mask(ax, selected_new_mask, (0.0, 0.9, 0.9))
        elif i == 3:
            overlay_mask(ax, vessel_mask, (0.9, 0.0, 0.9), 0.28)
            overlay_mask(ax, selected_new_mask, (0.0, 0.9, 0.9), 0.28)
            overlay_mask(ax, overlap_mask, (1.0, 1.0, 0.0), 0.55)
        add_patch_boxes(ax, eligible, top_patch)
        ax.set_title(titles[i], fontsize=7.5, pad=3)
        ax.axis("off")

    short_pid = (pid[:28] + "...") if len(pid) > 28 else pid
    fig.suptitle(
        f"{short_pid}  |  {role}  |  z={local_z}  |  "
        f"top={top_str}  IoU={iou:.3f}  Dice={dice:.3f}",
        fontsize=9, y=1.01
    )
    ann = (
        f"patient_id: {pid}\nrole: {role}  local_z: {local_z}\n"
        f"vessel: {v_count}px  new_mip: {n_count}px  overlap: {o_count}px\n"
        f"IoU={iou:.3f}  Dice={dice:.3f}\n"
        f"top: score={top_str}  vessel={top_v:.2f}  new_mip={top_n:.2f}\n"
        f"eligible: {len(eligible)}\n"
        f"lesion_overlap: {lesion_overlap}px  ratio={lesion_ratio:.3f}\n"
        f"lesion_risk: {lesion_risk}"
    )
    axes[0].text(0.0, -0.04, ann, transform=axes[0].transAxes, fontsize=5.6,
                 va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.82),
                 clip_on=False)
    legend_elems = [
        mpatches.Patch(edgecolor="limegreen", facecolor="none", label="ROI"),
        mpatches.Patch(facecolor=(0.9, 0, 0.9, 0.4), label=f"Vesselness ({v_count}px)"),
        mpatches.Patch(facecolor=(0, 0.9, 0.9, 0.4), label=f"New MIP-argmax ({n_count}px)"),
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

    png_name = f"b1e9_{pid_safe}_z{local_z:04d}.png"
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


def generate_candidate_comparison_png(patient, local_z, ct_vol, roi_vol, lesion_vol,
                                        all_score_rows, patch_preview_dict,
                                        candidate_masks, mip, slab_range,
                                        png_dir, pid_safe):
    """후보 3개 비교 PNG (subset only)"""
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

    n_cands = len(candidate_masks)
    fig, axes = plt.subplots(1, n_cands + 1, figsize=(6 * (n_cands + 1), 6))
    # 패널 0: MIP image
    axes[0].imshow(mip_disp, cmap="gray", vmin=0, vmax=1, origin="upper")
    draw_contours(axes[0], roi_slice, "limegreen", 0.7)
    if has_lesion and lesion_slice is not None and lesion_slice.any():
        draw_contours(axes[0], lesion_slice, "red", 1.0)
    add_patch_boxes(axes[0], eligible, top_patch)
    axes[0].set_title(f"MIP ({slab_range})", fontsize=7.5)
    axes[0].axis("off")

    colors = [(0.0, 0.9, 0.9), (0.9, 0.5, 0.0), (0.5, 0.0, 0.9)]
    for ci, (cname, cmask) in enumerate(candidate_masks.items()):
        ax = axes[ci + 1]
        ax.imshow(ct_disp, cmap="gray", vmin=0, vmax=1, origin="upper")
        draw_contours(ax, roi_slice, "limegreen", 0.7)
        if has_lesion and lesion_slice is not None and lesion_slice.any():
            draw_contours(ax, lesion_slice, "red", 1.0)
        overlay_mask(ax, cmask, colors[ci % len(colors)])
        add_patch_boxes(ax, eligible, top_patch)
        ax.set_title(f"{cname}\n[{int(cmask.sum())}px]", fontsize=7.5)
        ax.axis("off")

    short_pid = (pid[:22] + "...") if len(pid) > 22 else pid
    fig.suptitle(f"Candidate comparison: {short_pid} z={local_z}", fontsize=9)
    plt.tight_layout()

    png_name = f"b1e9_cand_{pid_safe}_z{local_z:04d}.png"
    png_path = png_dir / png_name
    plt.savefig(png_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return png_path


def main():
    is_dry = not ALLOW_REAL_PROCESSING
    print(f"[B1-E9] MIP Mask Extraction Fix")
    print(f"  ALLOW_REAL_PROCESSING = {ALLOW_REAL_PROCESSING}")
    print(f"  mode = {'DRY-RUN (candidate preview)' if is_dry else 'REAL (PNG 생성)'}")
    print()

    # ── holdout 확인 ──────────────────────────────────────────────────────
    holdout = load_stage2_holdout()
    if any(p["patient_id"] in holdout for p in PATIENTS):
        print("[ABORT] stage2_holdout 교집합 발견", file=sys.stderr)
        sys.exit(1)
    print("  stage2_holdout 교집합: 0 (PASS)")

    # ── output root 확인 ─────────────────────────────────────────────────
    if OUT_ROOT.exists():
        print(f"[ABORT] output root 이미 존재: {OUT_ROOT}", file=sys.stderr)
        sys.exit(1)
    print(f"  output root 없음 (PASS): {OUT_ROOT.name}")
    print()

    # ── B1-E8 문제 정리 ────────────────────────────────────────────────────
    print("  ── B1-E8 문제 원인 ───────────────────────────────────────────")
    print("  1. MIP > -100: argmax back-projection 없이 center slice 단순 투영")
    print("  2. 실제 max HU가 다른 z에서 나와도 center z mask에 포함됨")
    print("  3. CT > -850 조건이 비조영 혈관 일부를 가릴 수 있음")
    print()

    # ── candidate preview (dry-run/real 공통: CT read-only) ──────────────
    print("  ── Candidate preview (환자당 첫 슬라이스 샘플) ──────────────")
    preview_rows = []
    cand_map = {c["name"]: c for c in CANDIDATES}

    for patient in PATIENTS:
        pid = patient["patient_id"]
        ct_path = patient["ct_root"] / patient["vol_dir"] / "ct_hu.npy"
        roi_path = patient["roi_base"] / patient["safe_id"] / "refined_roi.npy"
        if not ct_path.exists() or not roi_path.exists():
            print(f"  [WARN] {pid[:20]}: 파일 없음, skip")
            continue

        ct_vol = np.load(ct_path, mmap_mode="r")
        roi_vol = np.load(roi_path, mmap_mode="r")
        n_slices = ct_vol.shape[0]

        lesion_vol = None
        if patient["has_lesion"]:
            lp = patient["ct_root"] / patient["vol_dir"] / "lesion_mask_roi_0_0.npy"
            if lp.exists():
                lesion_vol = np.load(lp, mmap_mode="r")

        z = patient["target_slices"][0]
        if z >= n_slices:
            continue

        mip, argmax_global, _, slab_range, z_start, z_end = compute_slab_mip_argmax(ct_vol, z)
        roi_mip = compute_roi_mip_mask(roi_vol, z_start, z_end)
        ct_z = np.array(ct_vol[z]).astype(np.float32)
        roi_z = np.array(roi_vol[z]).astype(bool)
        lesion_z = np.array(lesion_vol[z]).astype(bool) if lesion_vol is not None else None

        print(f"  [{pid[:22]}] z={z}  slab={slab_range}")
        for cand in CANDIDATES:
            mip_bin = compute_candidate_mip_binary(mip, roi_mip, cand)
            per_slice = argmax_backproject(mip_bin, argmax_global, z, roi_z, ct_z, lesion_z)
            px = int(per_slice.sum())
            sel = " ← FINAL" if cand["name"] in FINAL_CANDIDATES else ""
            print(f"    {cand['name']:15s}  {cand['label']:20s}  {px:6d}px{sel}")
            preview_rows.append({
                "patient_id": pid,
                "local_z": z,
                "candidate_name": cand["name"],
                "mip_rule": cand["label"],
                "air_threshold": AIR_THRESHOLD,
                "pixel_count": px,
                "top_patch_mask_ratio": None,
                "eligible_patch_mean_mask_ratio": None,
                "lesion_overlap_pixel_count": None,
                "visual_priority_score": None,
                "selected_for_final": cand["name"] in FINAL_CANDIDATES,
            })
        print()

    print(f"  ── 대표 후보 3개 (기본 추천) ────────────────────────────────")
    for fn in FINAL_CANDIDATES:
        print(f"    {fn}: {cand_map[fn]['label']}")
    print()
    total_main = sum(len(p["target_slices"]) for p in PATIENTS)
    n_subset = sum(len(v) for v in SUBSET_FOR_CANDIDATE_PNG.values())
    print(f"  예상 PNG: 기본 비교 {total_main}장 + 후보 비교 {n_subset}장 = {total_main + n_subset}장")
    print(f"  예상 실행시간: ~60-90초 (CPU, top-hat + MIP vesselness 포함)")
    print()

    if is_dry:
        print(
            "[DRY-RUN 완료] 실제 PNG 생성을 원하면:\n"
            "  source ~/ai_env/bin/activate && "
            "python scripts/b1e9_mip_mask_extraction_fix.py --real"
        )
        return

    # ── 실제 처리 ──────────────────────────────────────────────────────────
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    png_dir = OUT_ROOT / "pngs"
    png_dir.mkdir(exist_ok=True)

    patch_preview_dict = load_patch_preview()
    b1e8_old_pseudo = load_b1e8_old_pseudo()

    # 보호 파일 mtime 기록
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
    candidate_table_rows = []
    errors = []
    mtime_violations = []
    cand_map = {c["name"]: c for c in CANDIDATES}
    final_cand_list = [cand_map[n] for n in FINAL_CANDIDATES]

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

                    # 모든 후보 계산 (candidate_table용)
                    all_candidate_masks = {}
                    slice_preview = patch_preview_dict.get((pid, local_z), [])
                    eligible = [r for r in slice_preview if r["suppression_eligible"] == "True"]
                    slice_scores = [r for r in all_score_rows if int(r["local_z"]) == local_z]
                    top_patch = (
                        max(slice_scores, key=lambda r: float(r["padim_score"]))
                        if slice_scores else None
                    )

                    for cand in CANDIDATES:
                        mip_bin = compute_candidate_mip_binary(mip, roi_mip, cand)
                        per_sl = argmax_backproject(mip_bin, argmax_global, local_z, roi_z, ct_z, lesion_z)
                        all_candidate_masks[cand["name"]] = per_sl
                        top_r = patch_coverage_ratio(per_sl, [top_patch] if top_patch else [])
                        elig_r = patch_coverage_ratio(per_sl, eligible)
                        les_ov = int((per_sl & lesion_z).sum()) if lesion_z is not None else None
                        candidate_table_rows.append({
                            "patient_id": pid,
                            "local_z": local_z,
                            "candidate_name": cand["name"],
                            "mip_rule": cand["label"],
                            "air_threshold": AIR_THRESHOLD,
                            "pixel_count": int(per_sl.sum()),
                            "top_patch_mask_ratio": round(top_r, 4),
                            "eligible_patch_mean_mask_ratio": round(elig_r, 4),
                            "lesion_overlap_pixel_count": les_ov,
                            "visual_priority_score": None,
                            "selected_for_final": cand["name"] in FINAL_CANDIDATES,
                        })

                    # 대표 마스크 (FINAL_CANDIDATES[1] = B_p95)
                    selected_new_mask = all_candidate_masks[FINAL_CANDIDATES[1]]

                    # 기본 비교 PNG
                    png_path, stats = generate_main_png(
                        patient, local_z, ct_vol, roi_vol, lesion_vol,
                        all_score_rows, patch_preview_dict,
                        selected_new_mask, mip, slab_range,
                        vessel_mask, png_dir, pid_safe
                    )

                    old_px = b1e8_old_pseudo.get((pid, local_z), None)
                    index_rows.append({
                        "patient_id": pid,
                        "safe_id": safe_id,
                        "role": role,
                        "local_z": local_z,
                        "slab_range": slab_range,
                        "selected_mip_mask_rule": f"{FINAL_CANDIDATES[1]}({cand_map[FINAL_CANDIDATES[1]]['label']})",
                        "air_threshold": AIR_THRESHOLD,
                        "png_path": str(png_path.relative_to(PROJECT)),
                        "current_vesselness_pixel_count": stats["v_count"],
                        "old_b1e8_mip_pseudo_pixel_count": old_px,
                        "new_mip_argmax_mask_pixel_count": stats["n_count"],
                        "vesselness_vs_new_iou": round(stats["iou"], 4),
                        "vesselness_vs_new_dice": round(stats["dice"], 4),
                        "top_patch_score": round(stats["top_score"], 4) if not np.isnan(stats["top_score"]) else None,
                        "top_patch_vesselness_ratio": round(stats["top_v"], 4),
                        "top_patch_old_mip_ratio": None,
                        "top_patch_new_mip_ratio": round(stats["top_n"], 4),
                        "eligible_patch_count_on_slice": stats["eligible_count"],
                        "lesion_overlap_pixel_count": stats["lesion_overlap"],
                        "lesion_risk_case_flag": stats["lesion_risk"],
                        "note": "",
                    })

                    print(
                        f"    z={local_z:4d}  vessel={stats['v_count']:6d}"
                        f"  new(B_p95)={stats['n_count']:6d}"
                        f"  old={old_px or '?':>6}  IoU={stats['iou']:.3f}"
                        f"  top_v={stats['top_v']:.2f}  top_n={stats['top_n']:.2f}"
                    )

                    # 후보 비교 PNG (subset)
                    if local_z in SUBSET_FOR_CANDIDATE_PNG.get(pid, []):
                        cand_masks_sel = {
                            n: all_candidate_masks[n] for n in FINAL_CANDIDATES
                        }
                        generate_candidate_comparison_png(
                            patient, local_z, ct_vol, roi_vol, lesion_vol,
                            all_score_rows, patch_preview_dict,
                            cand_masks_sel, mip, slab_range, png_dir, pid_safe
                        )
                        print(f"    → candidate comparison PNG 생성 (z={local_z})")

                except Exception as e:
                    import traceback
                    errors.append({"patient_id": pid, "local_z": local_z,
                                   "error": traceback.format_exc()})
                    print(f"    [ERROR] z={local_z}: {e}")

    # ── mtime 검사 ────────────────────────────────────────────────────────
    for fpath, pre in pre_mtimes.items():
        if Path(fpath).exists() and abs(os.path.getmtime(fpath) - pre) > 0.01:
            mtime_violations.append(fpath)
    if mtime_violations:
        print(f"[ABORT] mtime 변경: {mtime_violations}", file=sys.stderr)
        sys.exit(1)

    # ── CSV / JSON / report 저장 ──────────────────────────────────────────
    index_cols = [
        "patient_id", "safe_id", "role", "local_z", "slab_range",
        "selected_mip_mask_rule", "air_threshold", "png_path",
        "current_vesselness_pixel_count", "old_b1e8_mip_pseudo_pixel_count",
        "new_mip_argmax_mask_pixel_count", "vesselness_vs_new_iou", "vesselness_vs_new_dice",
        "top_patch_score", "top_patch_vesselness_ratio",
        "top_patch_old_mip_ratio", "top_patch_new_mip_ratio",
        "eligible_patch_count_on_slice", "lesion_overlap_pixel_count",
        "lesion_risk_case_flag", "note",
    ]
    with open(OUT_ROOT / "b1e9_mip_mask_extraction_fix_index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=index_cols)
        w.writeheader()
        for row in index_rows:
            w.writerow({k: row.get(k, "") for k in index_cols})

    cand_cols = [
        "patient_id", "local_z", "candidate_name", "mip_rule", "air_threshold",
        "pixel_count", "top_patch_mask_ratio", "eligible_patch_mean_mask_ratio",
        "lesion_overlap_pixel_count", "visual_priority_score", "selected_for_final",
    ]
    with open(OUT_ROOT / "b1e9_mip_mask_extraction_fix_candidate_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cand_cols)
        w.writeheader()
        for row in candidate_table_rows:
            w.writerow({k: row.get(k, "") for k in cand_cols})

    with open(OUT_ROOT / "b1e9_mip_mask_extraction_fix_errors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "local_z", "error"])
        w.writeheader()
        for e in errors:
            w.writerow(e)

    patient_list = sorted(set(r["patient_id"] for r in index_rows))
    summary = {
        "n_patients": len(patient_list),
        "n_png": len(index_rows) + sum(
            len(v) for k, v in SUBSET_FOR_CANDIDATE_PNG.items()
            if k in [p["patient_id"] for p in PATIENTS]
        ),
        "selected_mip_mask_rule": FINAL_CANDIDATES[1],
        "mip_slab_definition": f"z-{MIP_SLAB_HALF_LO}~z+{MIP_SLAB_HALF_HI-1}(max 10sl)",
        "air_threshold": AIR_THRESHOLD,
        "candidates_tested": [c["name"] for c in CANDIDATES],
        "final_candidates": FINAL_CANDIDATES,
        "holdout_intersection": 0,
        "mtime_violations": len(mtime_violations),
        "source_files_modified": 0,
        "all_checks_passed": len(mtime_violations) == 0 and len(errors) == 0,
        "n_errors": len(errors),
    }
    with open(OUT_ROOT / "b1e9_mip_mask_extraction_fix_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    iou_vals = [r["vesselness_vs_new_iou"] for r in index_rows if r["vesselness_vs_new_iou"] != ""]
    new_px = [r["new_mip_argmax_mask_pixel_count"] for r in index_rows if r["new_mip_argmax_mask_pixel_count"] != ""]

    report_lines = [
        "# B1-E9 MIP Mask Extraction Fix Report",
        "",
        "## 1. B1-E8 MIP pseudo-mask 문제 원인",
        "",
        "- **MIP binary 생성**: `MIP > -100` 단순 threshold, argmax back-projection 없음",
        "- **center slice 단순 투영**: slab 내 다른 z에서 max가 나온 픽셀도 center slice mask에 포함",
        "- **결과**: normal004 1200~1500px로 과소 포함, 혈관 미스 가능성",
        "",
        "## 2. 새 방식",
        "",
        "- slab MIP + argmax_z map 동시 계산",
        "- `per_slice_mask[z] = MIP_binary AND argmax_z==z AND ROI AND CT>air AND ~lesion`",
        "- argmax back-projection으로 '이 z에서 실제로 max인' 픽셀만 포함",
        "",
        f"## 3. 후보 비교 (대표 3개: {FINAL_CANDIDATES})",
        "",
        "| 후보 | 규칙 | 특성 |",
        "|---|---|---|",
        "| A_0 | MIP>0 | intensity 기준, 혈관+고밀도구조 |",
        "| B_p95 | MIP ROI p95 | 상대적 밝기 기준, 환자별 자동 조정 |",
        "| D_v_p95 | MIP Frangi p95 | tubular shape 기반 |",
        "",
        f"## 4. 정량 요약 (대표: B_p95)",
        "",
        f"- 평균 IoU (vessel vs new): {round(float(np.mean(iou_vals)), 4) if iou_vals else 'N/A'}",
        f"- 평균 new_mip_argmax px: {round(float(np.mean(new_px)), 1) if new_px else 'N/A'}",
        "",
        "## 5. 최종 판정 (PNG 육안 검토 후)",
        "",
        "- `GO`: 새 MIP argmax mask가 혈관을 더 그럴듯하게 잡음",
        "- `CAUTION`: 개선은 있으나 과소/과포함 존재",
        "- `NO_GO`: MIP mask extraction 개선해도 부적합",
        "",
        "**현재 판정: (PNG 육안 검토 필요)**",
        "",
        "## 6. 안전 검증",
        "",
        f"- stage2_holdout 교집합: 0",
        f"- mtime 변경: {len(mtime_violations)}",
        f"- source_files_modified: 0",
        f"- all_checks_passed: {len(mtime_violations) == 0 and len(errors) == 0}",
    ]
    with open(OUT_ROOT / "b1e9_mip_mask_extraction_fix_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    if errors:
        print(f"\n[WARNING] {len(errors)}건 오류 → DONE 미생성")
    else:
        (OUT_ROOT / "DONE").write_text("OK")
        print(f"\n[DONE] B1-E9 완료")
    print(f"  PNG: {len(index_rows)}장(기본) + subset 후보비교")
    print(f"  errors: {len(errors)}")


if __name__ == "__main__":
    main()
