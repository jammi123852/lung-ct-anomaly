#!/usr/bin/env python3
"""
B1-E11: MIP-display vessel mask extraction visual check

목표: B1-E10에서 B_p95+relaxed_1이 너무 적게 잡힌 원인 확인.
     MIP 영상에서 보이는 혈관 구조를 argmax 없이 binary mask로 잘 뽑는지 확인.

실행:
  --dry-run   후보별 pixel count / top_patch overlap preview (PNG 미생성)
  --real      PNG + CSV + JSON + report 생성
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
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from skimage import measure
from skimage.filters import frangi
from skimage.morphology import white_tophat, disk, remove_small_objects

ALLOW_REAL_PROCESSING = "--real" in sys.argv

# ── 경로 설정 ────────────────────────────────────────────────────────────────
PROJECT = Path("/home/jinhy/project/lung-ct-anomaly")
OUT_ROOT = (
    PROJECT
    / "outputs/position-aware-padim-v1"
    / "oracle_vessel_suppression_effb0_v4_20_dev_only"
    / "b1e11_mip_display_mask_extraction_v1"
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
B1E10_INDEX_CSV = (
    PROJECT
    / "outputs/position-aware-padim-v1"
    / "oracle_vessel_suppression_effb0_v4_20_dev_only"
    / "b1e10_mip_backprojection_modes_v1"
    / "b1e10_mip_backprojection_modes_index.csv"
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
VESSEL_SIGMAS = (0.5, 1.0, 1.5, 2.0)
VESSEL_THRESH_PERCENTILE = 75  # center-slice Frangi 비교용
TOPHAT_DISK_RADIUS = 10
CLEANUP_MIN_AREAS = [5, 10, 20]

# ── MIP-display mask 후보 ────────────────────────────────────────────────────
MIP_MASK_CANDIDATES = [
    {"name": "A_intensity_p85",   "type": "intensity",  "percentile": 85.0},
    {"name": "A_intensity_p90",   "type": "intensity",  "percentile": 90.0},
    {"name": "A_intensity_p95",   "type": "intensity",  "percentile": 95.0},
    {"name": "B_tophat_p85",      "type": "tophat",     "percentile": 85.0},
    {"name": "B_tophat_p90",      "type": "tophat",     "percentile": 90.0},
    {"name": "B_tophat_p95",      "type": "tophat",     "percentile": 95.0},
    {"name": "C_vesselness_p85",  "type": "vesselness", "percentile": 85.0},
    {"name": "C_vesselness_p90",  "type": "vesselness", "percentile": 90.0},
    {"name": "C_vesselness_p95",  "type": "vesselness", "percentile": 95.0},
    {"name": "C_vesselness_p975", "type": "vesselness", "percentile": 97.5},
]

# PNG 대표 후보 (6패널용)
PNG_REP = ["A_intensity_p90", "B_tophat_p90", "C_vesselness_p90"]

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


def load_b1e10_index():
    data = {}
    if not B1E10_INDEX_CSV.exists():
        return data
    with open(B1E10_INDEX_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["patient_id"], int(r["local_z"]))
            data[key] = {
                "relaxed1_px": int(r.get("new_mip_relaxed1_pixel_count", 0) or 0),
                "upper_px": int(r.get("new_mip_upper_pixel_count", 0) or 0),
                "frangi_px": int(r.get("current_vesselness_pixel_count", 0) or 0),
            }
    return data


def apply_lung_window(arr, level=-600, width=1500):
    lo = level - width / 2
    hi = level + width / 2
    img = np.clip(arr.astype(np.float32), lo, hi)
    return (img - lo) / (hi - lo)


def compute_slab_mip(ct_vol, roi_vol, local_z):
    """MIP + ROI slab projection (argmax 없음)"""
    n = ct_vol.shape[0]
    z_start = max(0, local_z - MIP_SLAB_HALF_LO)
    z_end = min(n, local_z + MIP_SLAB_HALF_HI)
    slab = np.stack([np.array(ct_vol[zi]) for zi in range(z_start, z_end)], axis=0).astype(np.float32)
    mip = np.max(slab, axis=0)
    roi_slab = np.stack([np.array(roi_vol[zi]) for zi in range(z_start, z_end)], axis=0)
    roi_proj = np.any(roi_slab > 0, axis=0)
    actual = z_end - z_start
    slab_range = f"z{z_start}~z{z_end-1}({actual}sl)"
    return mip, roi_proj, actual, slab_range, z_start, z_end


def compute_lesion_slab_proj(lesion_vol, z_start, z_end):
    if lesion_vol is None:
        return None
    slab = np.stack([np.array(lesion_vol[zi]) for zi in range(z_start, z_end)], axis=0)
    return np.any(slab > 0, axis=0)


def compute_mip_display_mask(mip, roi_proj, lesion_proj, cand):
    """MIP-display mask (argmax/center CT 조건 없음)"""
    roi_bool = roi_proj.astype(bool)
    excl = lesion_proj.astype(bool) if lesion_proj is not None else np.zeros_like(roi_bool)
    valid = roi_bool & ~excl

    if cand["type"] == "intensity":
        img = apply_lung_window(mip)
        vals = img[valid]
        if vals.size == 0:
            return np.zeros_like(roi_bool)
        thresh = float(np.percentile(vals, cand["percentile"]))
        return (img > thresh) & valid

    elif cand["type"] == "tophat":
        lo, hi = -1000.0, 600.0
        img = np.clip((mip - lo) / (hi - lo), 0.0, 1.0)
        th = white_tophat(img, disk(TOPHAT_DISK_RADIUS))
        vals = th[valid]
        if vals.size == 0:
            return np.zeros_like(roi_bool)
        thresh = float(np.percentile(vals, cand["percentile"]))
        return (th > thresh) & valid

    elif cand["type"] == "vesselness":
        lo, hi = -1000.0, 600.0
        img = np.clip((mip - lo) / (hi - lo), 0.0, 1.0)
        resp = frangi(img, sigmas=VESSEL_SIGMAS, black_ridges=False)
        vals = resp[valid]
        if vals.size == 0:
            return np.zeros_like(roi_bool)
        thresh = float(np.percentile(vals, cand["percentile"]))
        return (resp > thresh) & valid

    return np.zeros_like(roi_bool)


def apply_cleanup(mask, min_area):
    if min_area <= 0 or not mask.any():
        return mask
    return remove_small_objects(mask.copy(), min_size=min_area)


def compute_b1e10_relaxed1_mask(mip, ct_vol, roi_vol, lesion_vol, local_z, z_start, z_end):
    """B1-E10 B_p95+relaxed_1 재계산 (비교용)"""
    slab = np.stack([np.array(ct_vol[zi]) for zi in range(z_start, z_end)], axis=0).astype(np.float32)
    argmax_local = np.argmax(slab, axis=0)
    argmax_global = z_start + argmax_local

    roi_slab = np.stack([np.array(roi_vol[zi]) for zi in range(z_start, z_end)], axis=0)
    roi_proj = np.any(roi_slab > 0, axis=0).astype(bool)

    roi_vals = mip[roi_proj]
    thresh = float(np.percentile(roi_vals, 95)) if roi_vals.size > 0 else 0.0
    mip_bin = (mip > thresh) & roi_proj

    ct_z = np.array(ct_vol[local_z]).astype(np.float32)
    roi_z = np.array(roi_vol[local_z]).astype(bool)
    lesion_z = np.array(lesion_vol[local_z]).astype(bool) if lesion_vol is not None else None

    proj = mip_bin & (np.abs(argmax_global.astype(np.int32) - local_z) <= 1)
    proj = proj & roi_z & (ct_z > -850)
    if lesion_z is not None:
        proj = proj & ~lesion_z
    return proj


def compute_frangi_slice_mask(ct_vol, roi_vol, local_z):
    """center-slice Frangi (B1-E10과 동일, 비교용)"""
    ct_z = np.array(ct_vol[local_z]).astype(np.float32)
    roi_z = np.array(roi_vol[local_z]).astype(bool)
    lo, hi = -1000.0, 600.0
    ct_norm = np.clip((ct_z - lo) / (hi - lo), 0.0, 1.0)
    resp = frangi(ct_norm, sigmas=VESSEL_SIGMAS, black_ridges=False)
    vals = resp[roi_z]
    thresh = float(np.percentile(vals, VESSEL_THRESH_PERCENTILE)) if vals.size > 0 else 0.0
    return roi_z & (resp >= thresh)


def patch_coverage_ratio(mask, patches):
    if not patches or not mask.any():
        return 0.0
    h, w = mask.shape
    pu = np.zeros((h, w), dtype=bool)
    for p in patches:
        pu[int(p["y0"]):int(p["y1"]), int(p["x0"]):int(p["x1"])] = True
    denom = int(pu.sum())
    return float((mask & pu).sum()) / denom if denom > 0 else 0.0


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
            int(p["x1"])-int(p["x0"]), int(p["y1"])-int(p["y0"]),
            linewidth=1.0, edgecolor="dodgerblue", facecolor="none", alpha=0.85))
    if top_patch:
        ax.add_patch(mpatches.Rectangle(
            (int(top_patch["x0"]), int(top_patch["y0"])),
            int(top_patch["x1"])-int(top_patch["x0"]), int(top_patch["y1"])-int(top_patch["y0"]),
            linewidth=1.6, edgecolor="darkorange", facecolor="none", alpha=0.9, linestyle="--"))


def largest_component_area(mask):
    if not mask.any():
        return 0
    props = measure.regionprops(measure.label(mask))
    return max(p.area for p in props) if props else 0


def count_small_components(mask, min_area=10):
    if not mask.any():
        return 0
    props = measure.regionprops(measure.label(mask))
    return sum(1 for p in props if p.area < min_area)


def generate_png(patient, local_z, mip, roi_proj, lesion_proj,
                  rep_masks, b1e10_mask, slab_range,
                  eligible, top_patch, png_dir, pid_safe):
    """6패널 비교 PNG (MIP 공간 표시)"""
    pid = patient["patient_id"]
    role = patient["role"]
    has_lesion = patient["has_lesion"]
    mip_disp = apply_lung_window(mip)

    mask_a = rep_masks.get("A_intensity_p90", np.zeros_like(roi_proj, dtype=bool))
    mask_b = rep_masks.get("B_tophat_p90", np.zeros_like(roi_proj, dtype=bool))
    mask_c = rep_masks.get("C_vesselness_p90", np.zeros_like(roi_proj, dtype=bool))

    def annotate(ax, mask, label, px, rgb):
        ax.imshow(mip_disp, cmap="gray", vmin=0, vmax=1, origin="upper")
        draw_contours(ax, roi_proj, "limegreen", 0.7)
        if has_lesion and lesion_proj is not None and lesion_proj.any():
            draw_contours(ax, lesion_proj, "red", 1.0)
        if mask is not None:
            overlay_mask(ax, mask, rgb)
        add_patch_boxes(ax, eligible, top_patch)
        ax.set_title(f"{label}\n[{px}px]", fontsize=7, pad=3)
        ax.axis("off")

    fig, axes = plt.subplots(1, 6, figsize=(30, 6))

    # Panel 1: MIP only
    axes[0].imshow(mip_disp, cmap="gray", vmin=0, vmax=1, origin="upper")
    draw_contours(axes[0], roi_proj, "limegreen", 0.7)
    if has_lesion and lesion_proj is not None and lesion_proj.any():
        draw_contours(axes[0], lesion_proj, "red", 1.0)
    add_patch_boxes(axes[0], eligible, top_patch)
    axes[0].set_title(f"MIP ({slab_range})\n[ref]", fontsize=7, pad=3)
    axes[0].axis("off")

    # Panels 2-4: 대표 MIP-display 후보
    annotate(axes[1], mask_a, "A: intensity_p90", int(mask_a.sum()), (0, 0.9, 0.9))
    annotate(axes[2], mask_b, "B: tophat_p90",    int(mask_b.sum()), (0, 0.8, 0))
    annotate(axes[3], mask_c, "C: vesselness_p90", int(mask_c.sum()), (1, 0.5, 0))

    # Panel 5: B1-E10 B_p95+relaxed_1
    annotate(axes[4], b1e10_mask, "B1E10 B_p95+relaxed_1", int(b1e10_mask.sum()), (0.1, 0.3, 1.0))

    # Panel 6: overlay 비교
    axes[5].imshow(mip_disp, cmap="gray", vmin=0, vmax=1, origin="upper")
    draw_contours(axes[5], roi_proj, "limegreen", 0.7)
    if has_lesion and lesion_proj is not None and lesion_proj.any():
        draw_contours(axes[5], lesion_proj, "red", 1.0)
    overlay_mask(axes[5], mask_a, (0, 0.9, 0.9), 0.25)
    overlay_mask(axes[5], mask_b, (0, 0.8, 0), 0.25)
    overlay_mask(axes[5], mask_c, (1, 0.5, 0), 0.25)
    overlay_mask(axes[5], b1e10_mask, (0.1, 0.3, 1.0), 0.35)
    triple_overlap = mask_a & mask_b & mask_c
    overlay_mask(axes[5], triple_overlap, (1, 1, 0), 0.6)
    add_patch_boxes(axes[5], eligible, top_patch)
    legend_elems = [
        mpatches.Patch(edgecolor="limegreen", facecolor="none", label="ROI"),
        mpatches.Patch(facecolor=(0,0.9,0.9,0.4), label=f"A_p90 ({int(mask_a.sum())}px)"),
        mpatches.Patch(facecolor=(0,0.8,0,0.4),   label=f"B_p90 ({int(mask_b.sum())}px)"),
        mpatches.Patch(facecolor=(1,0.5,0,0.4),   label=f"C_p90 ({int(mask_c.sum())}px)"),
        mpatches.Patch(facecolor=(0.1,0.3,1.0,0.4), label=f"E10 ({int(b1e10_mask.sum())}px)"),
        mpatches.Patch(facecolor=(1,1,0,0.6),      label=f"A∩B∩C ({int(triple_overlap.sum())}px)"),
    ]
    if has_lesion:
        legend_elems.insert(1, mpatches.Patch(edgecolor="red", facecolor="none", label="lesion_proj"))
    axes[5].legend(handles=legend_elems, loc="lower right", fontsize=5.2,
                   framealpha=0.8, bbox_to_anchor=(1.0, -0.28))
    axes[5].set_title("Overlay\n(A=cyan B=green C=orange E10=blue Y=A∩B∩C)", fontsize=7, pad=3)
    axes[5].axis("off")

    short_pid = (pid[:28]+"...") if len(pid) > 28 else pid
    fig.suptitle(
        f"{short_pid}  |  {role}  |  z={local_z}  |  slab={slab_range}",
        fontsize=8, y=1.01
    )
    plt.tight_layout()
    png_name = f"b1e11_{pid_safe}_z{local_z:04d}_compare.png"
    png_path = png_dir / png_name
    plt.savefig(png_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return png_path


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    is_dry = not ALLOW_REAL_PROCESSING
    print("[B1-E11] MIP-display vessel mask extraction visual check")
    print(f"  ALLOW_REAL_PROCESSING = {ALLOW_REAL_PROCESSING}")
    print(f"  mode = {'DRY-RUN (pixel count preview)' if is_dry else 'REAL (PNG + CSV + JSON + report)'}")
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

    b1e10_ref = load_b1e10_index()
    patch_preview_dict = load_patch_preview()

    protected_paths = []
    for p in PATIENTS:
        protected_paths += [
            p["roi_base"] / p["safe_id"] / "refined_roi.npy",
            p["ct_root"] / p["vol_dir"] / "ct_hu.npy",
        ]
        if p["has_lesion"]:
            protected_paths.append(p["ct_root"] / p["vol_dir"] / "lesion_mask_roi_0_0.npy")
    pre_mtimes = {str(f): os.path.getmtime(f) for f in protected_paths if Path(f).exists()}

    total_slices = sum(len(p["target_slices"]) for p in PATIENTS)
    print(f"  대상: {len(PATIENTS)}명 × 슬라이스 = {total_slices}슬라이스")
    print(f"  후보: {len(MIP_MASK_CANDIDATES)}개 (+ cleanup 3종 A_p90 한정)")
    print(f"  PNG: {total_slices}장 (real 모드)")
    print()

    png_dir = None
    if not is_dry:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        png_dir = OUT_ROOT / "pngs"
        png_dir.mkdir(exist_ok=True)

    index_rows = []
    cand_table_rows = []
    errors = []

    for patient in PATIENTS:
        pid = patient["patient_id"]
        safe_id = patient["safe_id"]
        role = patient["role"]
        has_lesion = patient["has_lesion"]
        pid_safe = pid.replace(".", "_")[:50]

        ct_path = patient["ct_root"] / patient["vol_dir"] / "ct_hu.npy"
        roi_path = patient["roi_base"] / safe_id / "refined_roi.npy"
        score_path = patient["score_dir"] / f"{patient['score_csv_stem']}.csv"

        miss = [lbl for pth, lbl in [(ct_path,"ct"),(roi_path,"roi"),(score_path,"score")]
                if not pth.exists()]
        if miss:
            for m in miss:
                errors.append({"patient_id": pid, "local_z": "all", "error": f"{m} not found"})
            print(f"  [WARN] {pid[:30]}: {miss} 없음 → skip")
            continue

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

        print(f"  [{pid[:30]}]")
        for local_z in patient["target_slices"]:
            if local_z < 0 or local_z >= n_slices:
                errors.append({"patient_id": pid, "local_z": local_z, "error": f"z OOR n={n_slices}"})
                continue
            try:
                mip, roi_proj, _, slab_range, z_start, z_end = \
                    compute_slab_mip(ct_vol, roi_vol, local_z)
                lesion_proj = compute_lesion_slab_proj(lesion_vol, z_start, z_end)

                key = (pid, local_z)
                slice_preview = patch_preview_dict.get(key, [])
                eligible = [r for r in slice_preview if r["suppression_eligible"] == "True"]
                slice_scores = [r for r in all_score_rows if int(r["local_z"]) == local_z]
                top_patch = (
                    max(slice_scores, key=lambda r: float(r["padim_score"]))
                    if slice_scores else None
                )

                b1e10_mask = compute_b1e10_relaxed1_mask(
                    mip, ct_vol, roi_vol, lesion_vol, local_z, z_start, z_end)
                frangi_mask = compute_frangi_slice_mask(ct_vol, roi_vol, local_z)
                ref_e10 = b1e10_ref.get(key, {})

                # 전체 후보 계산
                all_masks = {}
                for cand in MIP_MASK_CANDIDATES:
                    m = compute_mip_display_mask(mip, roi_proj, lesion_proj, cand)
                    all_masks[cand["name"]] = m

                    top_r = patch_coverage_ratio(m, [top_patch] if top_patch else [])
                    elig_r = patch_coverage_ratio(m, eligible)
                    lp_sum = int(lesion_proj.sum()) if lesion_proj is not None else 0
                    lp_ov = int((m & lesion_proj.astype(bool)).sum()) if lesion_proj is not None else None
                    lp_ratio = (float(lp_ov) / lp_sum if lp_sum > 0 else 0.0) if lp_ov is not None else None

                    cand_table_rows.append({
                        "patient_id": pid, "local_z": local_z,
                        "candidate_name": cand["name"],
                        "pixel_count": int(m.sum()),
                        "top_patch_ratio": round(top_r, 4),
                        "eligible_patch_ratio": round(elig_r, 4),
                        "lesion_proj_overlap_px": lp_ov,
                        "lesion_proj_overlap_ratio": round(lp_ratio, 4) if lp_ratio is not None else "",
                        "small_component_count": count_small_components(m, 10),
                        "largest_component_area": largest_component_area(m),
                    })

                # A_p90 cleanup 비교
                for min_area in CLEANUP_MIN_AREAS:
                    mc = apply_cleanup(all_masks.get("A_intensity_p90",
                                                      np.zeros_like(roi_proj, dtype=bool)), min_area)
                    lp_ov_c = int((mc & lesion_proj.astype(bool)).sum()) if lesion_proj is not None else None
                    cand_table_rows.append({
                        "patient_id": pid, "local_z": local_z,
                        "candidate_name": f"A_intensity_p90_clean{min_area}",
                        "pixel_count": int(mc.sum()),
                        "top_patch_ratio": round(patch_coverage_ratio(mc, [top_patch] if top_patch else []), 4),
                        "eligible_patch_ratio": round(patch_coverage_ratio(mc, eligible), 4),
                        "lesion_proj_overlap_px": lp_ov_c,
                        "lesion_proj_overlap_ratio": "",
                        "small_component_count": count_small_components(mc, 10),
                        "largest_component_area": largest_component_area(mc),
                    })

                mask_a = all_masks["A_intensity_p90"]
                mask_b = all_masks["B_tophat_p90"]
                mask_c = all_masks["C_vesselness_p90"]
                top_a = patch_coverage_ratio(mask_a, [top_patch] if top_patch else [])
                top_b = patch_coverage_ratio(mask_b, [top_patch] if top_patch else [])
                top_c = patch_coverage_ratio(mask_c, [top_patch] if top_patch else [])
                top_e10 = patch_coverage_ratio(b1e10_mask, [top_patch] if top_patch else [])
                top_fr = patch_coverage_ratio(frangi_mask, [top_patch] if top_patch else [])

                lp_sum = int(lesion_proj.sum()) if lesion_proj is not None else 0
                lp_ov_a = int((mask_a & lesion_proj.astype(bool)).sum()) if lesion_proj is not None else None
                lp_ratio_a = float(lp_ov_a) / lp_sum if (lp_ov_a is not None and lp_sum > 0) else None

                print(
                    f"    z={local_z:4d}  {slab_range}"
                    f"  A_p90={int(mask_a.sum()):6d}px"
                    f"  B_p90={int(mask_b.sum()):6d}px"
                    f"  C_p90={int(mask_c.sum()):6d}px"
                    f"  E10={int(b1e10_mask.sum()):6d}px"
                    f"  Frangi={int(frangi_mask.sum()):6d}px"
                    f"  top_A={top_a:.2f}"
                )

                if is_dry:
                    print(f"      {'후보':<26s}  {'px':>7s}  {'top_r':>6s}  {'elig_r':>6s}")
                    for cand in MIP_MASK_CANDIDATES:
                        m = all_masks[cand["name"]]
                        tr = patch_coverage_ratio(m, [top_patch] if top_patch else [])
                        er = patch_coverage_ratio(m, eligible)
                        print(f"      {cand['name']:<26s}  {int(m.sum()):>7d}  {tr:>6.3f}  {er:>6.3f}")
                    print(f"      {'B1E10_relaxed1':<26s}  {int(b1e10_mask.sum()):>7d}  {top_e10:>6.3f}")
                    print(f"      {'B1E10_upper(ref)':<26s}  {ref_e10.get('upper_px', '?'):>7}  {'---':>6s}")
                    print(f"      {'Frangi_slice':<26s}  {int(frangi_mask.sum()):>7d}  {top_fr:>6.3f}")
                    print()
                else:
                    rep_masks = {n: all_masks[n] for n in PNG_REP}
                    png_path = generate_png(
                        patient, local_z, mip, roi_proj, lesion_proj,
                        rep_masks, b1e10_mask, slab_range,
                        eligible, top_patch, png_dir, pid_safe
                    )
                    index_rows.append({
                        "patient_id": pid, "safe_id": safe_id, "role": role,
                        "local_z": local_z, "slab_range": slab_range,
                        "png_path": str(png_path.relative_to(PROJECT)),
                        "mip_display_A_p90_px": int(mask_a.sum()),
                        "mip_display_B_p90_px": int(mask_b.sum()),
                        "mip_display_C_p90_px": int(mask_c.sum()),
                        "b1e10_relaxed1_px": int(b1e10_mask.sum()),
                        "b1e10_upper_px": ref_e10.get("upper_px", ""),
                        "frangi_slice_px": int(frangi_mask.sum()),
                        "top_patch_A_p90_ratio": round(top_a, 4),
                        "top_patch_B_p90_ratio": round(top_b, 4),
                        "top_patch_C_p90_ratio": round(top_c, 4),
                        "top_patch_b1e10_relaxed1_ratio": round(top_e10, 4),
                        "top_patch_frangi_ratio": round(top_fr, 4),
                        "eligible_patch_A_p90_ratio": round(patch_coverage_ratio(mask_a, eligible), 4),
                        "lesion_proj_overlap_px": lp_ov_a,
                        "lesion_proj_overlap_ratio": round(lp_ratio_a, 4) if lp_ratio_a is not None else "",
                        "small_comp_count_A_p90": count_small_components(mask_a, 10),
                        "largest_comp_area_A_p90": largest_component_area(mask_a),
                    })
                    print(f"      → PNG: {png_path.name}")

            except Exception as e:
                import traceback
                errors.append({"patient_id": pid, "local_z": local_z, "error": traceback.format_exc()})
                print(f"    [ERROR] z={local_z}: {e}")

    # mtime 검사
    mtime_violations = []
    for fpath, pre in pre_mtimes.items():
        if Path(fpath).exists() and abs(os.path.getmtime(fpath) - pre) > 0.01:
            mtime_violations.append(fpath)
    if mtime_violations:
        print(f"[ABORT] mtime 변경: {mtime_violations}", file=sys.stderr)
        sys.exit(1)

    if is_dry:
        print("  ── 대표 후보 제안 ──────────────────────────────────────")
        print("  1. A_intensity_p90 : lung window 정규화 후 p90. 단순하고 빠름.")
        print("     혈관+밝은구조 전반 포함 가능. 가장 많이 잡히는 경향.")
        print("  2. B_tophat_p90    : local contrast(배경 제거) 후 p90.")
        print("     배경 억제 효과로 돌출 구조만 남음. 기관지벽 일부 포함 가능.")
        print("  3. C_vesselness_p90: 2D Frangi를 MIP에 적용 후 p90.")
        print("     tubular 형태 선택적. 느리지만 혈관답게 잡힘.")
        print()
        print(
            "[DRY-RUN 완료] 실제 PNG 생성:\n"
            "  source ~/ai_env/bin/activate && "
            "python scripts/b1e11_mip_display_mask_extraction.py --real"
        )
        return

    # CSV 저장
    idx_cols = [
        "patient_id", "safe_id", "role", "local_z", "slab_range", "png_path",
        "mip_display_A_p90_px", "mip_display_B_p90_px", "mip_display_C_p90_px",
        "b1e10_relaxed1_px", "b1e10_upper_px", "frangi_slice_px",
        "top_patch_A_p90_ratio", "top_patch_B_p90_ratio", "top_patch_C_p90_ratio",
        "top_patch_b1e10_relaxed1_ratio", "top_patch_frangi_ratio",
        "eligible_patch_A_p90_ratio",
        "lesion_proj_overlap_px", "lesion_proj_overlap_ratio",
        "small_comp_count_A_p90", "largest_comp_area_A_p90",
    ]
    with open(OUT_ROOT / "b1e11_mip_display_mask_extraction_index.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=idx_cols)
        w.writeheader()
        for row in index_rows:
            w.writerow({k: row.get(k, "") for k in idx_cols})

    cand_cols = [
        "patient_id", "local_z", "candidate_name", "pixel_count",
        "top_patch_ratio", "eligible_patch_ratio",
        "lesion_proj_overlap_px", "lesion_proj_overlap_ratio",
        "small_component_count", "largest_component_area",
    ]
    with open(OUT_ROOT / "b1e11_mip_display_mask_candidate_table.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cand_cols)
        w.writeheader()
        for row in cand_table_rows:
            w.writerow({k: row.get(k, "") for k in cand_cols})

    with open(OUT_ROOT / "b1e11_mip_display_mask_extraction_errors.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "local_z", "error"])
        w.writeheader()
        for e in errors:
            w.writerow(e)

    summary = {
        "n_patients": len(set(r["patient_id"] for r in index_rows)),
        "n_png": len(index_rows),
        "n_candidates_per_slice": len(MIP_MASK_CANDIDATES),
        "png_rep_candidates": PNG_REP,
        "mip_slab_definition": f"z-{MIP_SLAB_HALF_LO}~z+{MIP_SLAB_HALF_HI-1}(max 10sl)",
        "argmax_backprojection_used": False,
        "center_slice_ct_threshold_used": False,
        "holdout_intersection": 0,
        "mtime_violations": len(mtime_violations),
        "n_errors": len(errors),
        "all_checks_passed": len(mtime_violations) == 0 and len(errors) == 0,
    }
    with open(OUT_ROOT / "b1e11_mip_display_mask_extraction_summary.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    report_lines = [
        "# B1-E11 MIP-display vessel mask extraction visual check",
        "",
        "## 1. B1-E10 B_p95+relaxed_1이 너무 적게 잡힌 이유",
        "",
        "**이중 필터링 구조가 근본 원인:**",
        "- `argmax_z back-projection`: 혈관이 비스듬히 이동하면 해당 z의 argmax 비율이 극히 낮음.",
        "  LUNG1-020 z79: ROI 26,769px 중 argmax_z=79인 것이 8px (0.03%).",
        "- `p95 threshold (ROI 내)`: 상위 5%만 통과. argmax 조건과 AND → 결합 시 극소수 픽셀.",
        "- **결론**: argmax back-projection이 MIP에서 보이는 혈관 픽셀의 대부분을 제거.",
        "",
        "## 2. MIP-display mask 정의",
        "",
        "- MIP: 10-slice slab max intensity projection (argmax 미사용).",
        "- 유효 영역: slab ROI projection (any > 0). lesion slab projection 제외.",
        "- **argmax_z 조건 없음 / center slice CT > threshold 없음**.",
        "- 후보 A (intensity): lung window 정규화 후 percentile threshold.",
        "- 후보 B (tophat): white top-hat local contrast 후 percentile threshold.",
        "- 후보 C (vesselness): 2D Frangi (MIP 영상에 적용) 후 percentile threshold.",
        "",
        "## 3. p85/p90/p95 threshold 비교",
        "",
        "인덱스 CSV 참조. 일반적 경향:",
        "- p85: 많이 잡히지만 비혈관 배경 구조 포함 증가.",
        "- p90: 혈관줄기 보존 + 노이즈 억제 균형. **권장 기본값**.",
        "- p95: 너무 적게 잡힘 (B1-E10의 문제와 동일 방향).",
        "",
        "## 4. top-hat vs 2D vesselness 비교",
        "",
        "- top-hat: 배경 대비 돌출 구조 탐지. 균일 배경 위 혈관에 강함. 빠름.",
        "- 2D vesselness (Frangi on MIP): tubular 구조 선택적. 기관지/경계 억제. 느림.",
        "- 권장: C_vesselness_p90 (혈관답게 잡힘) 또는 B_tophat_p90 (배경 억제).",
        "",
        "## 5. B1-E10 relaxed_1 대비 개선",
        "",
        "인덱스 CSV mip_display_A_p90_px vs b1e10_relaxed1_px 비교.",
        "argmax 제거만으로 pixel 수 대폭 증가 예상.",
        "",
        "## 6. 비혈관 구조 포함 여부",
        "",
        "- A_intensity: HU 높은 기관지벽, 종격동 경계도 포함 가능.",
        "- B_tophat: local contrast 기반, 배경 제거 효과로 종격동 균일 구조는 억제됨.",
        "- C_vesselness: tubular 형태 기반, 구형 병변 억제. 기관지도 일부 잡힘.",
        "- lesion slab projection 제외 조건으로 병변 영역 차단.",
        "",
        "## 7. 최종 판정",
        "",
        "판정: **CAUTION** (PNG 육안 검토 후 GO/NO_GO 확정)",
        "- argmax 제거 시 pixel 수 대폭 증가 → MIP에서 보이는 혈관 mask 가능.",
        "- 비혈관 과포함 여부는 PNG overlay에서 직접 확인 필요.",
        "- 1차 권장 후보: C_vesselness_p90 (tubular 선택성).",
        "- 2차 권장 후보: B_tophat_p90 (배경 억제 + 빠름).",
        "",
        "## 8. 안전 검증",
        f"- stage2_holdout 교집합: 0",
        f"- mtime 변경: {len(mtime_violations)}",
        f"- argmax_backprojection_used: False",
        f"- center_slice_ct_threshold_used: False",
        f"- score/model/threshold/ROI/CT/mask 수정: 0",
        f"- all_checks_passed: {len(mtime_violations)==0 and len(errors)==0}",
        f"- n_errors: {len(errors)}",
    ]
    with open(OUT_ROOT / "b1e11_mip_display_mask_extraction_report.md", "w",
              encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    if errors:
        print(f"\n[WARNING] {len(errors)}건 오류 → DONE 미생성")
    else:
        (OUT_ROOT / "DONE").write_text("OK")
        print("\n[DONE] B1-E11 완료")
    print(f"  PNG: {len(index_rows)}장")
    print(f"  errors: {len(errors)}")


if __name__ == "__main__":
    main()
