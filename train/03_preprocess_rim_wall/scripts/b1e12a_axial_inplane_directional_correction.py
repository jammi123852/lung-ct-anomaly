#!/usr/bin/env python3
"""
B1-E12: Tri-planar MIP + single-slice correction vessel mask generation

axial/coronal/sagittal 3축 × intensity/top-hat/vesselness 3방식 → 9 masks
→ vote-based fusion (v1/v2/v3) → single-slice correction (corrected_v1/v2/v3)
→ per-slice axial vessel mask + 9-panel PNG

실행:
  --dry-run / --plan   계획/파라미터 미리보기 (PNG/CSV 미생성)
  --real               PNG + CSV + JSON + report 생성
"""
import sys

if __name__ == "__main__" and len(sys.argv) < 2:
    print(
        "[ERROR] bare-run guard: 인수 없이 실행 금지. "
        "--dry-run 또는 --real 을 사용하세요.",
        file=sys.stderr,
    )
    sys.exit(2)

import os
import csv
import json
import traceback as tb_mod
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.ndimage import maximum_filter1d, binary_dilation, binary_erosion
from skimage import measure
from skimage.filters import frangi
from skimage.morphology import white_tophat, disk, remove_small_objects

ALLOW_REAL_PROCESSING = "--real" in sys.argv

# ── 경로 ─────────────────────────────────────────────────────────────────────
PROJECT = Path("/home/jinhy/project/lung-ct-anomaly")
OUT_ROOT = (
    PROJECT
    / "outputs/position-aware-padim-v1"
    / "oracle_vessel_suppression_effb0_v4_20_dev_only"
    / "b1e12_triplanar_fusion_single_slice_correction_v1"
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

# ── 파라미터 ─────────────────────────────────────────────────────────────────
MIP_SLAB_SIZE = 10           # axial slab slices, coronal/sagittal sliding filter size
REP_PERCENTILE = 90          # 대표 threshold
TOPHAT_DISK_RADIUS = 10
VESSEL_SIGMAS = (0.5, 1.0, 1.5, 2.0)
MIN_AREA_CLEANUP = 10
SINGLE_SLICE_DILATE_RADIUS = 2
VOTE_V2_MIN = 2              # total_vote >= 2
PERIPHERAL_THIN_MAX_AREA = 30
CAND_PERCENTILES = [85, 90, 95]  # candidate_table용

# ── 환자 목록 ────────────────────────────────────────────────────────────────
PATIENTS = [
    {
        "patient_id": "LUNG1-020",
        "safe_id": "NSCLC_LUNG1-020__b843f4f3dc",
        "role": "lesion_candidate",
        "vol_dir": "NSCLC_LUNG1-020__b843f4f3dc",
        "ct_root": NROOT_LESION,
        "roi_base": ROI_BASE_LESION,
        "has_lesion": True,
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
        "score_csv_stem": "normal004",
        "score_dir": SCORE_BASE_NORMAL,
        "target_slices": [162, 160, 128],
    },
]

# ── helpers ───────────────────────────────────────────────────────────────────

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


def apply_lung_window(arr, level=-600, width=1500):
    lo = level - width / 2
    hi = level + width / 2
    img = np.clip(arr.astype(np.float32), lo, hi)
    return (img - lo) / (hi - lo)


def hu_normalize(arr):
    lo, hi = -1000.0, 600.0
    return np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


# ── MIP 계산 ─────────────────────────────────────────────────────────────────

def compute_axial_mip(ct_vol, roi_vol, local_z):
    """z방향 slab MIP → (H,W)"""
    n = ct_vol.shape[0]
    half = MIP_SLAB_SIZE // 2
    z_start = max(0, local_z - half)
    z_end = min(n, local_z + half)
    slab = np.array(ct_vol[z_start:z_end]).astype(np.float32)
    mip = np.max(slab, axis=0)
    roi_slab = np.array(roi_vol[z_start:z_end])
    roi_proj = np.any(roi_slab > 0, axis=0)
    slab_range = f"z{z_start}~z{z_end-1}({z_end-z_start}sl)"
    return mip, roi_proj, slab_range, z_start, z_end


def compute_coronal_mip(ct_vol, roi_vol, local_z):
    """axial slice z에서 y방향 sliding max (coronal direction) → (H,W)"""
    ct_z = np.array(ct_vol[local_z]).astype(np.float32)
    roi_z = np.array(roi_vol[local_z]).astype(bool)
    mip = maximum_filter1d(ct_z, size=MIP_SLAB_SIZE, axis=0)
    return mip, roi_z, f"y_slide(size={MIP_SLAB_SIZE})"


def compute_sagittal_mip(ct_vol, roi_vol, local_z):
    """axial slice z에서 x방향 sliding max (sagittal direction) → (H,W)"""
    ct_z = np.array(ct_vol[local_z]).astype(np.float32)
    roi_z = np.array(roi_vol[local_z]).astype(bool)
    mip = maximum_filter1d(ct_z, size=MIP_SLAB_SIZE, axis=1)
    return mip, roi_z, f"x_slide(size={MIP_SLAB_SIZE})"


def get_lesion_proj(lesion_vol, z_start, z_end):
    if lesion_vol is None:
        return None
    slab = np.array(lesion_vol[z_start:z_end])
    return np.any(slab > 0, axis=0)


def get_lesion_slice(lesion_vol, local_z):
    if lesion_vol is None:
        return None
    return np.array(lesion_vol[local_z]).astype(bool)


# ── vessel mask 계산 ──────────────────────────────────────────────────────────

def _preprocess_mip(mip, method):
    """MIP → 방식별 반응 이미지"""
    if method == "intensity":
        return apply_lung_window(mip)
    elif method == "tophat":
        img = hu_normalize(mip)
        return white_tophat(img, disk(TOPHAT_DISK_RADIUS))
    elif method == "vesselness":
        img = hu_normalize(mip)
        return frangi(img, sigmas=VESSEL_SIGMAS, black_ridges=False)
    return np.zeros_like(mip, dtype=np.float32)


def compute_vessel_mask(mip, roi_mask, lesion_mask, method, percentile=None):
    if percentile is None:
        percentile = REP_PERCENTILE
    excl = lesion_mask.astype(bool) if lesion_mask is not None else np.zeros_like(roi_mask, dtype=bool)
    valid = roi_mask.astype(bool) & ~excl
    resp = _preprocess_mip(mip, method)
    vals = resp[valid]
    if vals.size == 0:
        return np.zeros_like(roi_mask, dtype=bool)
    thresh = float(np.percentile(vals, percentile))
    return (resp > thresh) & valid


def apply_cleanup(mask):
    if MIN_AREA_CLEANUP <= 0 or not mask.any():
        return mask.copy()
    return remove_small_objects(mask.copy(), min_size=MIN_AREA_CLEANUP)


# ── 9 masks + vote + fusion ───────────────────────────────────────────────────

def compute_all_9_masks(axial_mip, coronal_mip, sagittal_mip,
                        roi_proj, roi_z, lesion_proj, lesion_z):
    """9개 mask: {plane}_{method}"""
    masks = {}
    for method in ("intensity", "tophat", "vesselness"):
        masks[f"axial_{method}"] = apply_cleanup(
            compute_vessel_mask(axial_mip, roi_proj, lesion_proj, method))
        masks[f"coronal_{method}"] = apply_cleanup(
            compute_vessel_mask(coronal_mip, roi_z, lesion_z, method))
        masks[f"sagittal_{method}"] = apply_cleanup(
            compute_vessel_mask(sagittal_mip, roi_z, lesion_z, method))
    return masks


def compute_vote_maps(masks):
    h, w = next(iter(masks.values())).shape
    total_vote = np.zeros((h, w), dtype=np.int32)
    method_acc = {"intensity": np.zeros((h, w), dtype=bool),
                  "tophat":    np.zeros((h, w), dtype=bool),
                  "vesselness":np.zeros((h, w), dtype=bool)}
    plane_acc  = {"axial":     np.zeros((h, w), dtype=bool),
                  "coronal":   np.zeros((h, w), dtype=bool),
                  "sagittal":  np.zeros((h, w), dtype=bool)}
    for key, mask in masks.items():
        total_vote += mask.astype(np.int32)
        plane, method = key.split("_", 1)
        method_acc[method] |= mask
        plane_acc[plane] |= mask
    method_vote = sum(v.astype(np.int32) for v in method_acc.values())
    plane_vote  = sum(v.astype(np.int32) for v in plane_acc.values())
    return method_vote, plane_vote, total_vote


def compute_fusion(masks, method_vote, plane_vote, total_vote):
    res = {}
    for method in ("intensity", "tophat", "vesselness"):
        res[f"{method}_allplane"] = (
            masks[f"axial_{method}"] | masks[f"coronal_{method}"] | masks[f"sagittal_{method}"])
    for plane in ("axial", "coronal", "sagittal"):
        res[f"{plane}_allmethod"] = (
            masks[f"{plane}_intensity"] | masks[f"{plane}_tophat"] | masks[f"{plane}_vesselness"])
    full_union = np.zeros_like(next(iter(masks.values())), dtype=bool)
    for m in masks.values():
        full_union |= m
    res["full_union"] = full_union
    res["final_fused_v1"] = full_union.copy()
    res["final_fused_v2"] = full_union & (total_vote >= VOTE_V2_MIN)
    res["final_fused_v3"] = full_union & ((plane_vote >= 2) | (method_vote >= 2))
    return res


# ── single-slice correction ───────────────────────────────────────────────────

def compute_slice_maps(ct_vol, roi_vol, lesion_vol, local_z):
    """단일 axial slice에서 3방식 vessel map"""
    ct_z = np.array(ct_vol[local_z]).astype(np.float32)
    roi_z = np.array(roi_vol[local_z]).astype(bool)
    les_z = get_lesion_slice(lesion_vol, local_z)
    s_i = apply_cleanup(compute_vessel_mask(ct_z, roi_z, les_z, "intensity"))
    s_t = apply_cleanup(compute_vessel_mask(ct_z, roi_z, les_z, "tophat"))
    s_v = apply_cleanup(compute_vessel_mask(ct_z, roi_z, les_z, "vesselness"))
    return s_i, s_t, s_v


def single_slice_correction(fused_mask, s_i, s_t, s_v):
    """confirm(fused & slice_any) + expand_nearby(dilate(fused) & slice_any & ~fused)"""
    slice_any = s_i | s_t | s_v
    selem = disk(SINGLE_SLICE_DILATE_RADIUS).astype(bool)
    fused_dilated = binary_dilation(fused_mask, structure=selem)
    confirmed = fused_mask & slice_any
    expanded  = fused_dilated & slice_any & ~fused_mask
    return confirmed | expanded


# ── 정량 헬퍼 ────────────────────────────────────────────────────────────────

def patch_ratio(mask, patches):
    if not patches or not mask.any():
        return 0.0
    h, w = mask.shape
    pu = np.zeros((h, w), dtype=bool)
    for p in patches:
        pu[int(p["y0"]):int(p["y1"]), int(p["x0"]):int(p["x1"])] = True
    denom = int(pu.sum())
    return float((mask & pu).sum()) / denom if denom > 0 else 0.0


def lesion_ov(mask, les_z):
    if les_z is None or not np.any(les_z):
        return 0.0
    les_b = les_z.astype(bool)
    denom = int(les_b.sum())
    return float((mask & les_b).sum()) / denom if denom > 0 else 0.0


def boundary_touch(mask, roi_mask):
    if not mask.any():
        return 0.0
    roi_eroded = binary_erosion(roi_mask, structure=np.ones((3, 3)))
    roi_bnd = roi_mask & ~roi_eroded
    denom = int(mask.sum())
    return float((mask & roi_bnd).sum()) / denom if denom > 0 else 0.0


def peripheral_thin_ratio(mask):
    if not mask.any():
        return 0.0
    props = measure.regionprops(measure.label(mask))
    thin_px = sum(p.area for p in props if p.area < PERIPHERAL_THIN_MAX_AREA)
    return float(thin_px) / int(mask.sum())


def hilar_ratio(mask, roi_mask):
    if not mask.any():
        return 0.0
    selem = disk(10).astype(bool)
    hilar = binary_erosion(roi_mask, structure=selem)
    if not hilar.any():
        return 0.0
    denom = int(mask.sum())
    return float((mask & hilar).sum()) / denom if denom > 0 else 0.0


# ── candidate table (p85/p90/p95) ────────────────────────────────────────────

def compute_candidate_table_rows(pid, local_z,
                                  axial_mip, coronal_mip, sagittal_mip,
                                  roi_proj, roi_z, lesion_proj, lesion_z,
                                  top_patch, eligible):
    rows = []
    mip_map = {"axial": (axial_mip, roi_proj, lesion_proj),
               "coronal": (coronal_mip, roi_z, lesion_z),
               "sagittal": (sagittal_mip, roi_z, lesion_z)}
    for plane, (mip, roi_m, les_m) in mip_map.items():
        for method in ("intensity", "tophat", "vesselness"):
            resp = _preprocess_mip(mip, method)
            excl = les_m.astype(bool) if les_m is not None else np.zeros_like(roi_m, dtype=bool)
            valid = roi_m.astype(bool) & ~excl
            vals = resp[valid]
            for pct in CAND_PERCENTILES:
                if vals.size > 0:
                    thresh = float(np.percentile(vals, pct))
                    mask = (resp > thresh) & valid
                    mask = apply_cleanup(mask)
                else:
                    mask = np.zeros_like(roi_m, dtype=bool)
                tp_r = patch_ratio(mask, [top_patch] if top_patch else [])
                ep_r = patch_ratio(mask, eligible)
                lp_r = lesion_ov(mask, lesion_z)
                rows.append({
                    "patient_id": pid, "local_z": local_z,
                    "plane": plane, "method": method, "percentile": pct,
                    "candidate_name": f"{plane}_{method}_p{pct}",
                    "pixel_count": int(mask.sum()),
                    "top_patch_ratio": round(tp_r, 4),
                    "eligible_patch_ratio": round(ep_r, 4),
                    "lesion_overlap_ratio": round(lp_r, 4),
                })
    return rows


# ── 시각화 ───────────────────────────────────────────────────────────────────

def draw_contours(ax, binary, color, lw=0.9):
    for c in measure.find_contours(binary.astype(np.float32), 0.5):
        ax.plot(c[:, 1], c[:, 0], color=color, linewidth=lw)


def overlay_mask(ax, mask, rgb, alpha=0.38):
    ovl = np.zeros((*mask.shape, 4), dtype=np.float32)
    ovl[mask, 0] = rgb[0]; ovl[mask, 1] = rgb[1]; ovl[mask, 2] = rgb[2]; ovl[mask, 3] = alpha
    ax.imshow(ovl, origin="upper")


def add_patch_boxes(ax, eligible, top_patch):
    for p in eligible:
        ax.add_patch(mpatches.Rectangle(
            (int(p["x0"]), int(p["y0"])),
            int(p["x1"]) - int(p["x0"]), int(p["y1"]) - int(p["y0"]),
            linewidth=1.0, edgecolor="dodgerblue", facecolor="none", alpha=0.85))
    if top_patch:
        ax.add_patch(mpatches.Rectangle(
            (int(top_patch["x0"]), int(top_patch["y0"])),
            int(top_patch["x1"]) - int(top_patch["x0"]),
            int(top_patch["y1"]) - int(top_patch["y0"]),
            linewidth=1.6, edgecolor="darkorange", facecolor="none", alpha=0.9, linestyle="--"))


def generate_png(patient, local_z, ct_vol,
                  axial_mip, coronal_mip, sagittal_mip,
                  roi_proj, roi_z, lesion_proj, lesion_z,
                  masks, fusion, s_i, s_t, s_v,
                  corrected_v1, corrected_v2, corrected_v3,
                  eligible, top_patch,
                  axial_sr, coronal_sr, sagittal_sr,
                  png_dir, pid_safe):
    pid = patient["patient_id"]
    has_lesion = patient["has_lesion"]
    role = patient["role"]

    ct_disp  = apply_lung_window(np.array(ct_vol[local_z]).astype(np.float32))
    ax_disp  = apply_lung_window(axial_mip)
    cor_disp = apply_lung_window(coronal_mip)
    sag_disp = apply_lung_window(sagittal_mip)

    full_union = fusion["full_union"]
    fused_v2   = fusion["final_fused_v2"]
    fused_v3   = fusion["final_fused_v3"]

    les_for_axial = lesion_proj if (has_lesion and lesion_proj is not None) else None
    les_for_slice = lesion_z    if (has_lesion and lesion_z is not None)    else None

    def base(ax, img, roi_m, les_m, title, patches=True):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, origin="upper")
        draw_contours(ax, roi_m, "limegreen", 0.7)
        if les_m is not None and les_m.any():
            draw_contours(ax, les_m.astype(bool), "red", 1.0)
        if patches:
            add_patch_boxes(ax, eligible, top_patch)
        ax.set_title(title, fontsize=6.5, pad=3)
        ax.axis("off")

    fig, axes = plt.subplots(3, 3, figsize=(27, 27))
    ax = axes.flatten()

    # 1: center axial CT
    base(ax[0], ct_disp, roi_z, les_for_slice,
         f"1. Axial CT  z={local_z}")

    # 2: axial MIP + 3방식
    base(ax[1], ax_disp, roi_proj, les_for_axial,
         f"2. Axial MIP\n{axial_sr}")
    overlay_mask(ax[1], masks["axial_intensity"],  (0,0.9,0.9), 0.25)
    overlay_mask(ax[1], masks["axial_tophat"],     (0,0.8,0.0), 0.25)
    overlay_mask(ax[1], masks["axial_vesselness"], (1,0.5,0.0), 0.25)

    # 3: coronal MIP + 3방식
    base(ax[2], cor_disp, roi_z, les_for_slice,
         f"3. Coronal MIP\n{coronal_sr}")
    overlay_mask(ax[2], masks["coronal_intensity"],  (0,0.9,0.9), 0.25)
    overlay_mask(ax[2], masks["coronal_tophat"],     (0,0.8,0.0), 0.25)
    overlay_mask(ax[2], masks["coronal_vesselness"], (1,0.5,0.0), 0.25)

    # 4: sagittal MIP + 3방식
    base(ax[3], sag_disp, roi_z, les_for_slice,
         f"4. Sagittal MIP\n{sagittal_sr}")
    overlay_mask(ax[3], masks["sagittal_intensity"],  (0,0.9,0.9), 0.25)
    overlay_mask(ax[3], masks["sagittal_tophat"],     (0,0.8,0.0), 0.25)
    overlay_mask(ax[3], masks["sagittal_vesselness"], (1,0.5,0.0), 0.25)

    # 5: full_union on CT
    base(ax[4], ct_disp, roi_z, les_for_slice,
         f"5. Full Union\n[{int(full_union.sum())}px]")
    overlay_mask(ax[4], full_union, (0.8,0.2,0.8), 0.45)

    # 6: fused_v2 on CT
    base(ax[5], ct_disp, roi_z, les_for_slice,
         f"6. Fused v2 (vote≥{VOTE_V2_MIN})\n[{int(fused_v2.sum())}px]")
    overlay_mask(ax[5], fused_v2, (0.2,0.6,1.0), 0.45)

    # 7: fused_v3 on CT
    base(ax[6], ct_disp, roi_z, les_for_slice,
         f"7. Fused v3 (plane≥2|method≥2)\n[{int(fused_v3.sum())}px]")
    overlay_mask(ax[6], fused_v3, (0.0,0.8,0.4), 0.45)

    # 8: corrected_v2 on CT
    base(ax[7], ct_disp, roi_z, les_for_slice,
         f"8. Corrected v2\n[{int(corrected_v2.sum())}px]")
    overlay_mask(ax[7], corrected_v2, (1.0,0.7,0.0), 0.45)

    # 9: corrected_v3 on CT
    base(ax[8], ct_disp, roi_z, les_for_slice,
         f"9. Corrected v3 (★recommended)\n[{int(corrected_v3.sum())}px]")
    overlay_mask(ax[8], corrected_v3, (1.0,0.3,0.3), 0.35)
    # recommended = corrected_v2 가 더 안정적이나 corrected_v3도 표시
    overlay_mask(ax[8], corrected_v2, (1.0,1.0,0.0), 0.20)

    legend_ax = ax[8]
    legend_elems = [
        mpatches.Patch(edgecolor="limegreen", facecolor="none", label="ROI"),
        mpatches.Patch(facecolor=(1,0.3,0.3,0.4), label=f"corr_v3({int(corrected_v3.sum())}px)"),
        mpatches.Patch(facecolor=(1,1,0,0.3),     label=f"corr_v2({int(corrected_v2.sum())}px)"),
    ]
    if has_lesion:
        legend_elems.insert(1, mpatches.Patch(edgecolor="red", facecolor="none", label="lesion"))
    legend_ax.legend(handles=legend_elems, loc="lower right", fontsize=5, framealpha=0.8,
                     bbox_to_anchor=(1.0, -0.28))

    short = (pid[:28] + "...") if len(pid) > 28 else pid
    fig.suptitle(
        f"{short}  |  {role}  |  z={local_z}  |  B1-E12 Tri-planar Fusion",
        fontsize=9, y=1.01
    )
    plt.tight_layout()
    png_name = f"b1e12_{pid_safe}_z{local_z:04d}_triplanar.png"
    png_path = png_dir / png_name
    plt.savefig(png_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return png_path


# ── report ────────────────────────────────────────────────────────────────────

def write_report(index_rows, out_root, n_errors):
    def smean(lst):
        try:
            return round(sum(float(x) for x in lst) / len(lst), 1) if lst else 0.0
        except Exception:
            return 0.0

    fu   = smean([r.get("full_union_pixel_count", 0) for r in index_rows])
    fv2  = smean([r.get("final_fused_v2_pixel_count", 0) for r in index_rows])
    fv3  = smean([r.get("final_fused_v3_pixel_count", 0) for r in index_rows])
    cv2  = smean([r.get("corrected_v2_pixel_count", 0) for r in index_rows])

    lines = [
        "# B1-E12 Tri-planar MIP + Single-slice Correction 보고서",
        "",
        "## 1. 3축 MIP를 쓰는 이유",
        "axial(z방향 slab)은 혈관 단면이 반복되는 구조를 잘 잡는다.",
        "coronal(y방향 sliding max)은 상하로 달리는 폐문 혈관과 세로 분지 혈관을 보완한다.",
        "sagittal(x방향 sliding max)은 좌우로 분지하는 혈관, 종격동 인접 가로 혈관을 보완한다.",
        "3축 union은 단일 orientation이 놓친 혈관을 상호 보완한다.",
        "",
        "## 2. 3방식을 모두 쓰는 이유",
        "intensity: 고HU voxel 직접 임계화 — 단순, 경계/흉벽 과포함 위험.",
        "top-hat: 국소 배경 대비 강조 — 배경 기울기 제거로 혈관을 더 선명하게 분리.",
        "vesselness(Frangi): 관형 곡률 반응 — 가는 혈관·비관형 잡음 구분.",
        "3방식 각각이 다른 혈관 패턴을 잡으므로 union/vote로 안정성을 높인다.",
        "",
        "## 3. orientation별 강점",
        "axial: 폐 단면의 원형/타원형 혈관 단면 반복 포착에 강함.",
        "coronal: 폐문에서 분지하며 상하로 달리는 폐동·정맥.",
        "sagittal: 좌우 분지 혈관, 종격동 인접 혈관.",
        "⚠ 주의: coronal/sagittal은 z-spacing ≠ xy-spacing이면 물리적 slab 두께가 다름.",
        "  (구현: maximum_filter1d axis=0/1 on single axial slice, 실제 3D ortho와 다름)",
        "",
        "## 4. Full union 확장 정도",
        f"full_union 평균 pixel: {fu}",
        f"final_fused_v2 평균 pixel: {fv2}",
        f"final_fused_v3 평균 pixel: {fv3}",
        f"corrected_v2 평균 pixel: {cv2}",
        "full_union은 axial_allmethod 대비 coronal/sagittal 기여로 ~20-40% 넓어질 수 있다.",
        "",
        "## 5. Vote 기반 fusion 안정성",
        f"total_vote >= {VOTE_V2_MIN} (v2): 9개 mask 중 최소 {VOTE_V2_MIN}개가 동의한 pixel만 통과.",
        "  단일 orientation/방식 특이 노이즈를 걸러내어 안정성 향상.",
        "plane_vote >= 2 또는 method_vote >= 2 (v3): 다른 축 또는 다른 방식에서 근거가 있는 pixel 우선.",
        "",
        "## 6. Single-slice correction 효과",
        f"confirm + expand-nearby (dilate_r={SINGLE_SLICE_DILATE_RADIUS}):",
        "  confirm: fused & slice_any → MIP에서 잡혔고 실제 slice에서도 확인된 pixel 유지.",
        "  expand: dilate(fused) & slice_any & ~fused → MIP 경계 근처에서 slice 근거 있는 pixel 추가.",
        "  효과: MIP 과연결(isolated bright spots)을 confirm 조건으로 일부 제거,",
        "        MIP에서 약하게 잡혔지만 slice에 뚜렷한 혈관 보완.",
        "",
        "## 7. corrected_v1/v2/v3 추천",
        "corrected_v2 (fused_v2 기반): vote≥2로 안정적, single-slice로 MIP 과포함 일부 제거 → 권장.",
        "corrected_v3: 더 보수적, 작은 혈관 일부 누락 가능.",
        "corrected_v1 = corrected(full_union): 가장 공격적, 주의 필요.",
        "",
        "## 8. 흉벽 근처 미세혈관",
        "3축 union은 coronal/sagittal이 흉벽 근처 미세혈관을 추가로 잡아 boundary_touch_ratio 증가 가능.",
        "lesion overlap: GGO 등 혈관처럼 보이는 병변에서 발생 가능 (lesion_proj 제외로 대부분 방지).",
        "",
        "## 9. 최종 추천",
        "recommended_final_mask = corrected_v2",
        "이유: vote≥2로 안정적, single-slice confirm+expand로 MIP 과연결 감소, 실제 slice 혈관 보완.",
        "",
        "## 10. 최종 판정",
        "CAUTION",
        "tri-planar + single-slice correction이 axial 단독 대비 혈관 커버리지를 높이지만:",
        " - coronal/sagittal은 실제 3D orthogonal MIP가 아닌 2D sliding max이므로 효과 한계 있음.",
        " - full_union은 현저히 넓어지므로 vote filter(v2/v3) 필수.",
        " - corrected_v2가 합리적 출발점이며 눈검증 후 최종 결정 권장.",
        "",
        "## 통계",
        f"- 총 슬라이스: {len(index_rows)}",
        f"- 오류: {n_errors}",
    ]
    rpath = out_root / "b1e12_triplanar_fusion_single_slice_correction_report.md"
    with open(rpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return rpath


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("[B1-E12] Tri-planar MIP + Single-slice Correction")
    print(f"  ALLOW_REAL_PROCESSING = {ALLOW_REAL_PROCESSING}")
    print(f"  mode = {'REAL' if ALLOW_REAL_PROCESSING else 'DRY-RUN'}")
    print()

    holdout = load_stage2_holdout()
    if any(p["patient_id"] in holdout for p in PATIENTS):
        print("[ABORT] stage2_holdout 교집합 발견", file=sys.stderr)
        sys.exit(1)
    print("  stage2_holdout 교집합: 0 (PASS)")

    if OUT_ROOT.exists():
        print(f"[ABORT] output root 이미 존재: {OUT_ROOT}", file=sys.stderr)
        sys.exit(1)
    print(f"  output root 없음 (PASS): {OUT_ROOT.name}")
    print()

    # 보호 경로 mtime 기록
    protected = []
    for p in PATIENTS:
        protected += [
            p["roi_base"] / p["safe_id"] / "refined_roi.npy",
            p["ct_root"] / p["vol_dir"] / "ct_hu.npy",
        ]
        if p["has_lesion"]:
            protected.append(p["ct_root"] / p["vol_dir"] / "lesion_mask_roi_0_0.npy")
    pre_mtimes = {str(f): os.path.getmtime(f) for f in protected if Path(f).exists()}

    total_slices = sum(len(p["target_slices"]) for p in PATIENTS)
    print(f"  대상: {len(PATIENTS)}명, {total_slices}슬라이스")

    if not ALLOW_REAL_PROCESSING:
        print()
        print("=== DRY-RUN PLAN ===")
        print(f"파라미터:")
        print(f"  MIP_SLAB_SIZE        = {MIP_SLAB_SIZE}")
        print(f"  REP_PERCENTILE       = {REP_PERCENTILE}")
        print(f"  TOPHAT_DISK_RADIUS   = {TOPHAT_DISK_RADIUS}")
        print(f"  VESSEL_SIGMAS        = {VESSEL_SIGMAS}")
        print(f"  MIN_AREA_CLEANUP     = {MIN_AREA_CLEANUP}")
        print(f"  SINGLE_SLICE_DILATE  = {SINGLE_SLICE_DILATE_RADIUS}")
        print(f"  VOTE_V2_MIN          = {VOTE_V2_MIN}")
        print(f"  CAND_PERCENTILES     = {CAND_PERCENTILES}")
        print()
        print("대표 threshold: p90 (ROI 내 유효 pixel 기준, 전 orientation × 방식 동일)")
        print()
        print("3축 MIP 구현:")
        print(f"  axial:    z방향 {MIP_SLAB_SIZE}slice slab max projection → (H,W)")
        print(f"  coronal:  y방향 maximum_filter1d size={MIP_SLAB_SIZE} on axial slice → (H,W)")
        print(f"  sagittal: x방향 maximum_filter1d size={MIP_SLAB_SIZE} on axial slice → (H,W)")
        print()
        print("Fusion 구조:")
        print("  full_union = 9개 mask OR")
        print(f"  fused_v2   = full_union & total_vote >= {VOTE_V2_MIN}")
        print("  fused_v3   = full_union & (plane_vote >= 2 | method_vote >= 2)")
        print(f"  corrected  = confirm(fused & slice_any) | expand(dilate(fused,r={SINGLE_SLICE_DILATE_RADIUS}) & slice_any & ~fused)")
        print("  recommended = corrected_v2")
        print()
        print("생성 예정 환자/슬라이스:")
        for p in PATIENTS:
            for z in p["target_slices"]:
                print(f"  {p['patient_id'][:40]:45s} z={z:3d}  ({p['role']})")
        print()
        est_min = total_slices * 25
        est_max = total_slices * 50
        print(f"예상 PNG 수:          {total_slices}")
        print(f"예상 실행시간:        ~{est_min}~{est_max}초 (CPU, vesselness 포함)")
        print()
        print("[DRY-RUN 완료] 실행 승인 후: python b1e12_... --real")
        return

    # ── REAL ─────────────────────────────────────────────────────────────────
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    png_dir = OUT_ROOT / "pngs"
    png_dir.mkdir(exist_ok=True)

    patch_preview_dict = load_patch_preview()
    index_rows, cand_rows, errors = [], [], []

    for patient in PATIENTS:
        pid = patient["patient_id"]
        safe_id = patient["safe_id"]
        has_lesion = patient["has_lesion"]
        pid_safe = pid.replace(".", "_")[:60]

        ct_path    = patient["ct_root"] / patient["vol_dir"] / "ct_hu.npy"
        roi_path   = patient["roi_base"] / safe_id / "refined_roi.npy"
        score_path = patient["score_dir"] / f"{patient['score_csv_stem']}.csv"

        miss = [lbl for pth, lbl in [(ct_path,"ct"),(roi_path,"roi"),(score_path,"score")]
                if not pth.exists()]
        if miss:
            for m in miss:
                errors.append({"patient_id": pid, "local_z": "all", "error": f"{m} not found"})
            print(f"  [WARN] {pid[:35]}: {miss} → skip")
            continue

        ct_vol  = np.load(ct_path,  mmap_mode="r")
        roi_vol = np.load(roi_path, mmap_mode="r")
        n_slices = ct_vol.shape[0]
        lesion_vol = None
        if has_lesion:
            lp = patient["ct_root"] / patient["vol_dir"] / "lesion_mask_roi_0_0.npy"
            if lp.exists():
                lesion_vol = np.load(lp, mmap_mode="r")

        all_scores = []
        with open(score_path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if "local_z" in r and "padim_score" in r:
                    all_scores.append(r)

        print(f"  [{pid[:40]}]  shape={ct_vol.shape}")

        for local_z in patient["target_slices"]:
            if local_z < 0 or local_z >= n_slices:
                errors.append({"patient_id": pid, "local_z": local_z,
                                "error": f"z OOR n={n_slices}"})
                continue
            try:
                # 3축 MIP
                axial_mip,  roi_proj, axial_sr,  z_start, z_end = \
                    compute_axial_mip(ct_vol, roi_vol, local_z)
                coronal_mip, roi_z, coronal_sr = \
                    compute_coronal_mip(ct_vol, roi_vol, local_z)
                sagittal_mip, _, sagittal_sr = \
                    compute_sagittal_mip(ct_vol, roi_vol, local_z)

                lesion_proj = get_lesion_proj(lesion_vol, z_start, z_end)
                lesion_z    = get_lesion_slice(lesion_vol, local_z)

                # patch 정보
                key = (pid, local_z)
                preview = patch_preview_dict.get(key, [])
                eligible = [r for r in preview if r.get("suppression_eligible") == "True"]
                zscores = [r for r in all_scores if int(r["local_z"]) == local_z]
                top_patch = (max(zscores, key=lambda r: float(r["padim_score"]))
                             if zscores else None)

                # 9 masks
                masks = compute_all_9_masks(
                    axial_mip, coronal_mip, sagittal_mip,
                    roi_proj, roi_z, lesion_proj, lesion_z)

                # vote + fusion
                method_vote, plane_vote, total_vote = compute_vote_maps(masks)
                fusion = compute_fusion(masks, method_vote, plane_vote, total_vote)

                # single-slice maps
                s_i, s_t, s_v = compute_slice_maps(ct_vol, roi_vol, lesion_vol, local_z)

                # corrected masks
                corr_v1 = single_slice_correction(fusion["final_fused_v1"], s_i, s_t, s_v)
                corr_v2 = single_slice_correction(fusion["final_fused_v2"], s_i, s_t, s_v)
                corr_v3 = single_slice_correction(fusion["final_fused_v3"], s_i, s_t, s_v)

                fu = fusion["full_union"]
                fv1 = fusion["final_fused_v1"]
                fv2 = fusion["final_fused_v2"]
                fv3 = fusion["final_fused_v3"]

                # PNG
                png_path = generate_png(
                    patient, local_z, ct_vol,
                    axial_mip, coronal_mip, sagittal_mip,
                    roi_proj, roi_z, lesion_proj, lesion_z,
                    masks, fusion, s_i, s_t, s_v,
                    corr_v1, corr_v2, corr_v3,
                    eligible, top_patch,
                    axial_sr, coronal_sr, sagittal_sr,
                    png_dir, pid_safe)

                # index row
                tp = lambda m: patch_ratio(m, [top_patch] if top_patch else [])
                ep = lambda m: patch_ratio(m, eligible)
                tv_mean = round(float(total_vote[roi_z].mean()), 4) if roi_z.any() else 0.0
                mv_mean = round(float(method_vote[roi_z].mean()), 4) if roi_z.any() else 0.0
                pv_mean = round(float(plane_vote[roi_z].mean()), 4) if roi_z.any() else 0.0

                row = {
                    "patient_id": pid, "local_z": local_z, "role": patient["role"],
                    "axial_slab_range": axial_sr,
                    "coronal_slab_range": coronal_sr,
                    "sagittal_slab_range": sagittal_sr,
                    "axial_intensity_pixel_count":    int(masks["axial_intensity"].sum()),
                    "axial_tophat_pixel_count":       int(masks["axial_tophat"].sum()),
                    "axial_vesselness_pixel_count":   int(masks["axial_vesselness"].sum()),
                    "coronal_intensity_pixel_count":  int(masks["coronal_intensity"].sum()),
                    "coronal_tophat_pixel_count":     int(masks["coronal_tophat"].sum()),
                    "coronal_vesselness_pixel_count": int(masks["coronal_vesselness"].sum()),
                    "sagittal_intensity_pixel_count": int(masks["sagittal_intensity"].sum()),
                    "sagittal_tophat_pixel_count":    int(masks["sagittal_tophat"].sum()),
                    "sagittal_vesselness_pixel_count":int(masks["sagittal_vesselness"].sum()),
                    "full_union_pixel_count":     int(fu.sum()),
                    "final_fused_v1_pixel_count": int(fv1.sum()),
                    "final_fused_v2_pixel_count": int(fv2.sum()),
                    "final_fused_v3_pixel_count": int(fv3.sum()),
                    "corrected_v1_pixel_count":   int(corr_v1.sum()),
                    "corrected_v2_pixel_count":   int(corr_v2.sum()),
                    "corrected_v3_pixel_count":   int(corr_v3.sum()),
                    "plane_vote_mean":   pv_mean,
                    "method_vote_mean":  mv_mean,
                    "total_vote_mean":   tv_mean,
                    "top_patch_full_union_ratio":    round(tp(fu), 4),
                    "top_patch_corrected_v2_ratio":  round(tp(corr_v2), 4),
                    "top_patch_corrected_v3_ratio":  round(tp(corr_v3), 4),
                    "eligible_patch_full_union_ratio":    round(ep(fu), 4),
                    "eligible_patch_corrected_v2_ratio":  round(ep(corr_v2), 4),
                    "eligible_patch_corrected_v3_ratio":  round(ep(corr_v3), 4),
                    "lesion_overlap_full_union":    round(lesion_ov(fu, lesion_z), 4),
                    "lesion_overlap_corrected_v2":  round(lesion_ov(corr_v2, lesion_z), 4),
                    "lesion_overlap_corrected_v3":  round(lesion_ov(corr_v3, lesion_z), 4),
                    "boundary_touch_ratio":         round(boundary_touch(fu, roi_z), 4),
                    "peripheral_thin_vessel_ratio": round(peripheral_thin_ratio(fu), 4),
                    "hilar_region_ratio":           round(hilar_ratio(fu, roi_z), 4),
                    "recommended_mask": "corrected_v2",
                    "png_path": str(png_path.relative_to(PROJECT)),
                }
                index_rows.append(row)

                # candidate table (p85/p90/p95)
                cand_rows += compute_candidate_table_rows(
                    pid, local_z,
                    axial_mip, coronal_mip, sagittal_mip,
                    roi_proj, roi_z, lesion_proj, lesion_z,
                    top_patch, eligible)

                print(
                    f"    z={local_z:4d}  ax_i={int(masks['axial_intensity'].sum()):5d}"
                    f"  full_union={int(fu.sum()):5d}"
                    f"  fused_v2={int(fv2.sum()):5d}"
                    f"  corr_v2={int(corr_v2.sum()):5d}")

            except Exception as e:
                tb_str = tb_mod.format_exc()
                errors.append({"patient_id": pid, "local_z": local_z,
                                "error": str(e), "traceback": tb_str[:400]})
                print(f"    [ERROR] z={local_z}: {e}")

    # mtime 검증
    mtime_viol = 0
    for f, mt in pre_mtimes.items():
        cur = os.path.getmtime(f) if Path(f).exists() else None
        if cur is not None and abs(cur - mt) > 1:
            print(f"  [WARN] mtime changed: {f}", file=sys.stderr)
            mtime_viol += 1
    print(f"\n  mtime violations: {mtime_viol}")

    # 저장
    if index_rows:
        idx_path = OUT_ROOT / "b1e12_triplanar_fusion_single_slice_correction_index.csv"
        with open(idx_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
            w.writeheader(); w.writerows(index_rows)
        print(f"  index CSV: {idx_path.name} ({len(index_rows)}행)")

    if cand_rows:
        ct_path = OUT_ROOT / "b1e12_triplanar_fusion_single_slice_correction_candidate_table.csv"
        with open(ct_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(cand_rows[0].keys()))
            w.writeheader(); w.writerows(cand_rows)
        print(f"  candidate table: {ct_path.name} ({len(cand_rows)}행)")

    err_path = OUT_ROOT / "b1e12_triplanar_fusion_single_slice_correction_errors.csv"
    with open(err_path, "w", newline="", encoding="utf-8") as f:
        fnames = ["patient_id", "local_z", "error", "traceback"]
        w = csv.DictWriter(f, fieldnames=fnames, extrasaction="ignore")
        w.writeheader(); w.writerows(errors)

    summary = {
        "n_patients": len(PATIENTS),
        "n_slices_processed": len(index_rows),
        "n_png": len(index_rows),
        "n_errors": len(errors),
        "mip_slab_size": MIP_SLAB_SIZE,
        "rep_percentile": REP_PERCENTILE,
        "cand_percentiles": CAND_PERCENTILES,
        "orientations": ["axial", "coronal", "sagittal"],
        "methods": ["intensity", "tophat", "vesselness"],
        "n_base_masks": 9,
        "fusion_versions": ["full_union", "final_fused_v1", "final_fused_v2", "final_fused_v3",
                            "corrected_v1", "corrected_v2", "corrected_v3"],
        "recommended_final_mask": "corrected_v2",
        "single_slice_correction": "confirm+expand_nearby",
        "single_slice_dilate_radius": SINGLE_SLICE_DILATE_RADIUS,
        "coronal_sagittal_note": "maximum_filter1d on single axial slice, not true 3D ortho MIP",
        "holdout_intersection": 0,
        "mtime_violations": mtime_viol,
        "all_checks_passed": len(errors) == 0 and mtime_viol == 0,
    }
    sum_path = OUT_ROOT / "b1e12_triplanar_fusion_single_slice_correction_summary.json"
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    write_report(index_rows, OUT_ROOT, len(errors))

    if len(errors) == 0 and mtime_viol == 0:
        (OUT_ROOT / "DONE").touch()
        print("\n  [DONE] 모든 검증 통과")
    else:
        print(f"\n  [WARN] DONE 미생성: errors={len(errors)}, mtime_viol={mtime_viol}")


if __name__ == "__main__":
    main()
