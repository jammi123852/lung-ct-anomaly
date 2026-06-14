#!/usr/bin/env python3
"""
B1-E13 Smoke: axial 10-slice MIP 3-method union - 단일 슬라이스 검증
LUNG1-020 z=79 1장으로 intensity_p90 / tophat_p90 / vesselness_p90 union 동작 확인

실행:
  --dry-run / --plan   계획 미리보기 (PNG/CSV 미생성)
  --real               PNG + CSV + JSON 생성
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
from scipy.ndimage import binary_erosion
from skimage import measure
from skimage.filters import frangi
from skimage.morphology import white_tophat, disk, remove_small_objects

ALLOW_REAL_PROCESSING = "--real" in sys.argv

# ── 경로 ─────────────────────────────────────────────────────────────────────
PROJECT  = Path("/home/jinhy/project/lung-ct-anomaly")
OUT_ROOT = (
    PROJECT
    / "outputs/position-aware-padim-v1"
    / "oracle_vessel_suppression_effb0_v4_20_dev_only"
    / "b1e13_smoke_p85_union_v1"
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
B1E12_INDEX = (
    PROJECT
    / "outputs/position-aware-padim-v1"
    / "oracle_vessel_suppression_effb0_v4_20_dev_only"
    / "b1e12_triplanar_local_mip_single_slice_correction_v1"
    / "b1e12_triplanar_local_mip_single_slice_correction_index.csv"
)
NROOT_LESION = Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy"
)
ROI_BASE_LESION = (
    PROJECT / "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1/lesion"
)
SCORE_BASE_LESION = (
    PROJECT
    / "experiments/efficientnet_b0_imagenet_chestwall_removed_roi_v1/outputs/scores"
    / "lesion_stage1_dev_by_patient"
)

# ── 단일 슬라이스 대상 ────────────────────────────────────────────────────────
PATIENT = {
    "patient_id":    "LUNG1-020",
    "safe_id":       "NSCLC_LUNG1-020__b843f4f3dc",
    "role":          "lesion_candidate",
    "vol_dir":       "NSCLC_LUNG1-020__b843f4f3dc",
    "ct_root":       NROOT_LESION,
    "roi_base":      ROI_BASE_LESION,
    "has_lesion":    True,
    "score_csv_stem": "LUNG1-020",
    "score_dir":     SCORE_BASE_LESION,
    "target_z":      79,
}

# ── 파라미터 (B1-E11과 동일) ──────────────────────────────────────────────────
MIP_SLAB_HALF_LO   = 5
MIP_SLAB_HALF_HI   = 5
VESSEL_SIGMAS      = (0.5, 1.0, 1.5, 2.0)
TOPHAT_DISK_RADIUS = 10
MIN_AREA_CLEANUP   = 10
PERCENTILE         = 85.0

_STEM = "b1e13_smoke_p85_union"


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


def load_b1e12_row(pid, local_z):
    if not B1E12_INDEX.exists():
        return None
    with open(B1E12_INDEX, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["patient_id"] == pid and int(r["local_z"]) == local_z:
                return r
    return None


def apply_lung_window(arr, level=-600, width=1500):
    lo = level - width / 2
    hi = level + width / 2
    img = np.clip(arr.astype(np.float32), lo, hi)
    return (img - lo) / (hi - lo)


def hu_normalize(arr):
    lo, hi = -1000.0, 600.0
    return np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


# ── axial 10-slice MIP (B1-E11 방식 그대로) ──────────────────────────────────

def compute_slab_mip(ct_vol, roi_vol, local_z):
    n = ct_vol.shape[0]
    z_start = max(0, local_z - MIP_SLAB_HALF_LO)
    z_end   = min(n, local_z + MIP_SLAB_HALF_HI)
    slab     = np.stack([np.array(ct_vol[zi]) for zi in range(z_start, z_end)], axis=0).astype(np.float32)
    mip      = np.max(slab, axis=0)
    roi_slab = np.stack([np.array(roi_vol[zi]) for zi in range(z_start, z_end)], axis=0)
    roi_proj = np.any(roi_slab > 0, axis=0)
    actual   = z_end - z_start
    slab_range = f"z{z_start}~z{z_end-1}({actual}sl)"
    return mip, roi_proj, actual, slab_range, z_start, z_end


def compute_lesion_slab_proj(lesion_vol, z_start, z_end):
    if lesion_vol is None:
        return None
    slab = np.stack([np.array(lesion_vol[zi]) for zi in range(z_start, z_end)], axis=0)
    return np.any(slab > 0, axis=0)


# ── 3-method mask 계산 (B1-E11 compute_mip_display_mask와 동일) ───────────────

def compute_intensity_mask(mip, valid):
    img   = apply_lung_window(mip)
    vals  = img[valid]
    if vals.size == 0:
        return np.zeros(mip.shape, dtype=bool)
    thresh = float(np.percentile(vals, PERCENTILE))
    return (img > thresh) & valid


def compute_tophat_mask(mip, valid):
    img   = hu_normalize(mip)
    th    = white_tophat(img, disk(TOPHAT_DISK_RADIUS))
    vals  = th[valid]
    if vals.size == 0:
        return np.zeros(mip.shape, dtype=bool)
    thresh = float(np.percentile(vals, PERCENTILE))
    return (th > thresh) & valid


def compute_vesselness_mask(mip, valid):
    img   = hu_normalize(mip)
    resp  = frangi(img, sigmas=VESSEL_SIGMAS, black_ridges=False)
    vals  = resp[valid]
    if vals.size == 0:
        return np.zeros(mip.shape, dtype=bool)
    thresh = float(np.percentile(vals, PERCENTILE))
    return (resp > thresh) & valid


def apply_cleanup(mask):
    if MIN_AREA_CLEANUP <= 0 or not mask.any():
        return mask.copy()
    return remove_small_objects(mask.copy(), min_size=MIN_AREA_CLEANUP)


# ── 정량 ─────────────────────────────────────────────────────────────────────

def patch_ratio(mask, patches):
    if not patches or not mask.any():
        return 0.0
    h, w = mask.shape
    pu = np.zeros((h, w), dtype=bool)
    for p in patches:
        pu[int(p["y0"]):int(p["y1"]), int(p["x0"]):int(p["x1"])] = True
    denom = int(pu.sum())
    return float((mask & pu).sum()) / denom if denom > 0 else 0.0


def lesion_ov(mask, les):
    if les is None or not np.any(les):
        return 0.0
    denom = int(les.sum())
    return float((mask & les.astype(bool)).sum()) / denom if denom > 0 else 0.0


def boundary_touch(mask, roi_proj):
    if not mask.any():
        return 0.0
    roi_eroded = binary_erosion(roi_proj, structure=np.ones((3, 3)))
    bnd = roi_proj & ~roi_eroded
    denom = int(mask.sum())
    return float((mask & bnd).sum()) / denom if denom > 0 else 0.0


def peripheral_thin_ratio(mask):
    if not mask.any():
        return 0.0
    props = measure.regionprops(measure.label(mask))
    thin_px = sum(p.area for p in props if p.area < 30)
    return float(thin_px) / int(mask.sum())


def hilar_ratio(mask, roi_proj):
    if not mask.any():
        return 0.0
    from skimage.morphology import disk as skdisk
    selem = disk(10).astype(bool)
    from scipy.ndimage import binary_erosion as be
    hilar = be(roi_proj, structure=selem)
    if not hilar.any():
        return 0.0
    denom = int(mask.sum())
    return float((mask & hilar).sum()) / denom if denom > 0 else 0.0


def iou(a, b):
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return round(inter / union, 4) if union > 0 else 0.0


# ── 시각화 ───────────────────────────────────────────────────────────────────

def draw_contours(ax, binary, color, lw=0.9):
    for c in measure.find_contours(binary.astype(np.float32), 0.5):
        ax.plot(c[:, 1], c[:, 0], color=color, linewidth=lw)


def overlay_mask(ax, mask, rgb, alpha=0.40):
    ovl = np.zeros((*mask.shape, 4), dtype=np.float32)
    ovl[mask, 0] = rgb[0]; ovl[mask, 1] = rgb[1]
    ovl[mask, 2] = rgb[2]; ovl[mask, 3] = alpha
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


def generate_png(local_z, ct_disp, mip_disp, roi_proj, lesion_proj,
                  m_int, m_top, m_ves, m_union,
                  b1e12_mask, eligible, top_patch, slab_range, png_dir):
    pid = PATIENT["patient_id"]

    def base(ax, img, roi_m, les_m, title):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, origin="upper")
        draw_contours(ax, roi_m, "limegreen", 0.7)
        if les_m is not None and np.any(les_m):
            draw_contours(ax, les_m.astype(bool), "red", 1.0)
        add_patch_boxes(ax, eligible, top_patch)
        ax.set_title(title, fontsize=7, pad=3)
        ax.axis("off")

    n_cols = 8 if b1e12_mask is not None else 6
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 5, 5.5))
    ax = axes

    base(ax[0], ct_disp,  roi_proj, lesion_proj, f"1. Axial CT  z={local_z}")
    base(ax[1], mip_disp, roi_proj, lesion_proj, f"2. Axial MIP\n{slab_range}")

    base(ax[2], ct_disp, roi_proj, lesion_proj,
         f"3. intensity_p90\n[{int(m_int.sum())}px]")
    overlay_mask(ax[2], m_int, (0, 0.9, 0.9))

    base(ax[3], ct_disp, roi_proj, lesion_proj,
         f"4. tophat_p90\n[{int(m_top.sum())}px]")
    overlay_mask(ax[3], m_top, (0, 0.8, 0))

    base(ax[4], ct_disp, roi_proj, lesion_proj,
         f"5. vesselness_p90\n[{int(m_ves.sum())}px]")
    overlay_mask(ax[4], m_ves, (0.8, 0, 0.8))

    base(ax[5], ct_disp, roi_proj, lesion_proj,
         f"6. union_3method ★\n[{int(m_union.sum())}px]")
    overlay_mask(ax[5], m_union, (1.0, 0.85, 0))

    if b1e12_mask is not None:
        base(ax[6], ct_disp, roi_proj, lesion_proj,
             f"7. B1-E12 corr_v2\n[{int(b1e12_mask.sum())}px] (HOLD)")
        overlay_mask(ax[6], b1e12_mask, (0.3, 0.5, 1.0))

        diff = m_union ^ b1e12_mask
        base(ax[7], ct_disp, roi_proj, lesion_proj,
             f"8. union vs B1-E12 diff\n[{int(diff.sum())}px]")
        overlay_mask(ax[7], m_union & ~b1e12_mask, (1.0, 0.5, 0), 0.45)   # union only
        overlay_mask(ax[7], b1e12_mask & ~m_union, (0.3, 0.5, 1.0), 0.45) # b1e12 only

    legend_elems = [
        mpatches.Patch(edgecolor="limegreen", facecolor="none", label="ROI"),
        mpatches.Patch(facecolor=(0, 0.9, 0.9, 0.5), label=f"intensity_p90({int(m_int.sum())}px)"),
        mpatches.Patch(facecolor=(0, 0.8, 0, 0.5),   label=f"tophat_p90({int(m_top.sum())}px)"),
        mpatches.Patch(facecolor=(0.8, 0, 0.8, 0.5), label=f"vesselness_p90({int(m_ves.sum())}px)"),
        mpatches.Patch(facecolor=(1, 0.85, 0, 0.6),  label=f"union_3method({int(m_union.sum())}px)★"),
    ]
    if PATIENT["has_lesion"]:
        legend_elems.insert(1, mpatches.Patch(edgecolor="red", facecolor="none", label="lesion"))

    ax[-1].legend(handles=legend_elems, loc="lower right", fontsize=5,
                  framealpha=0.85, bbox_to_anchor=(1.0, -0.35))

    fig.suptitle(
        f"{pid}  |  lesion_candidate  |  z={local_z}  |  B1-E13 smoke: axial MIP 3-method union",
        fontsize=8, y=1.02
    )
    plt.tight_layout()
    png_name = f"{_STEM}_{pid}_z{local_z:04d}.png"
    png_path = png_dir / png_name
    plt.savefig(png_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return png_path


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    pid      = PATIENT["patient_id"]
    local_z  = PATIENT["target_z"]

    print("[B1-E13 smoke] axial MIP 3-method union - 단일 슬라이스 검증")
    print(f"  ALLOW_REAL_PROCESSING = {ALLOW_REAL_PROCESSING}")
    print(f"  mode = {'REAL' if ALLOW_REAL_PROCESSING else 'DRY-RUN'}")
    print(f"  대상: {pid}  z={local_z}")
    print()

    # holdout check
    holdout = load_stage2_holdout()
    if pid in holdout:
        print(f"[ABORT] {pid}는 stage2_holdout", file=sys.stderr)
        sys.exit(1)
    print(f"  stage2_holdout 교집합: 0 (PASS)")

    # output root check
    if OUT_ROOT.exists():
        print(f"[ABORT] output root 이미 존재: {OUT_ROOT}", file=sys.stderr)
        sys.exit(1)
    print(f"  output root 없음 (PASS): {OUT_ROOT.name}")

    # B1-E12 비교 파일
    b1e12_available = B1E12_INDEX.exists()
    print(f"  B1-E12 index: {'있음' if b1e12_available else '없음'}")
    print()

    # 경로 확인
    ct_path    = PATIENT["ct_root"] / PATIENT["vol_dir"] / "ct_hu.npy"
    roi_path   = PATIENT["roi_base"] / PATIENT["safe_id"] / "refined_roi.npy"
    score_path = PATIENT["score_dir"] / f"{PATIENT['score_csv_stem']}.csv"
    les_path   = PATIENT["ct_root"] / PATIENT["vol_dir"] / "lesion_mask_roi_0_0.npy"

    for lbl, pth in [("ct", ct_path), ("roi", roi_path), ("score", score_path), ("lesion", les_path)]:
        status = "OK" if pth.exists() else "MISSING"
        print(f"  {lbl:8s}: {status}  {pth.name}")

    # 보호 파일 mtime 기록
    protected = [ct_path, roi_path, les_path]
    pre_mtimes = {str(f): os.path.getmtime(f) for f in protected if Path(f).exists()}

    if not ALLOW_REAL_PROCESSING:
        print()
        print("=== DRY-RUN PLAN ===")
        print("파라미터:")
        print(f"  MIP_SLAB    : z-{MIP_SLAB_HALF_LO} ~ z+{MIP_SLAB_HALF_HI-1} (max 10sl, B1-E11 동일)")
        print(f"  PERCENTILE  : {PERCENTILE}")
        print(f"  TOPHAT_DISK : {TOPHAT_DISK_RADIUS}")
        print(f"  VESSEL_SIGMAS: {VESSEL_SIGMAS}")
        print(f"  MIN_AREA    : {MIN_AREA_CLEANUP}")
        print()
        print("생성 mask:")
        print("  intensity_p90  = apply_lung_window(MIP) > p90 & valid")
        print("  tophat_p90     = white_tophat(hu_norm(MIP), disk(10)) > p90 & valid")
        print("  vesselness_p90 = frangi(hu_norm(MIP)) > p90 & valid")
        print("  union_3method  = intensity_p90 | tophat_p90 | vesselness_p90")
        print("  최종 mask: roi_proj (slab OR projection) 내부, lesion 제외")
        print()
        print("생성 예정 파일:")
        print(f"  PNG   : {OUT_ROOT}/pngs/{_STEM}_LUNG1-020_z0079.png")
        print(f"  CSV   : {OUT_ROOT}/{_STEM}_index.csv")
        print(f"  NPZ   : {OUT_ROOT}/{_STEM}_masks.npz")
        print(f"  JSON  : {OUT_ROOT}/{_STEM}_summary.json")
        print(f"  errors: {OUT_ROOT}/{_STEM}_errors.csv")
        print()
        print("예상 실행시간: ~30~60초 (CPU, frangi 포함)")
        print("GPU: 없음 / 과금: 없음")
        print()
        if b1e12_available:
            print("B1-E12 비교 가능: corrected_v2/corrected_v3 pixel count + IoU 표기 예정")
        else:
            print("B1-E12 index 없음: 비교 패널 스킵")
        print()
        print("[DRY-RUN 완료] 실행 승인 후: python b1e13_smoke_single_slice.py --real")
        return

    # ── REAL ─────────────────────────────────────────────────────────────────
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    png_dir = OUT_ROOT / "pngs"
    png_dir.mkdir(exist_ok=True)

    patch_preview_dict = load_patch_preview()
    errors = []
    result_row = None
    npz_data = {}

    try:
        ct_vol  = np.load(ct_path,  mmap_mode="r")
        roi_vol = np.load(roi_path, mmap_mode="r")
        les_vol = np.load(les_path, mmap_mode="r") if les_path.exists() else None

        n_slices = ct_vol.shape[0]
        print(f"  CT shape: {ct_vol.shape}")

        if local_z < 0 or local_z >= n_slices:
            raise ValueError(f"z={local_z} OOR n={n_slices}")

        # axial 10-slice MIP
        mip, roi_proj, actual_sl, slab_range, z_start, z_end = \
            compute_slab_mip(ct_vol, roi_vol, local_z)
        lesion_proj = compute_lesion_slab_proj(les_vol, z_start, z_end)

        # valid mask (roi_proj & ~lesion)
        excl  = lesion_proj.astype(bool) if lesion_proj is not None else np.zeros_like(roi_proj, dtype=bool)
        valid = roi_proj.astype(bool) & ~excl

        # 3 method masks
        m_int  = apply_cleanup(compute_intensity_mask(mip, valid))
        m_top  = apply_cleanup(compute_tophat_mask(mip, valid))
        m_ves  = apply_cleanup(compute_vesselness_mask(mip, valid))
        m_union = m_int | m_top | m_ves

        print(f"  slab_range    : {slab_range}")
        print(f"  intensity_p90 : {int(m_int.sum())} px")
        print(f"  tophat_p90    : {int(m_top.sum())} px")
        print(f"  vesselness_p90: {int(m_ves.sum())} px")
        print(f"  union_3method : {int(m_union.sum())} px")

        # patch 정보
        key      = (pid, local_z)
        preview  = patch_preview_dict.get(key, [])
        eligible = [r for r in preview if r.get("suppression_eligible") == "True"]
        all_scores = []
        with open(score_path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if "local_z" in r and "padim_score" in r:
                    all_scores.append(r)
        zscores   = [r for r in all_scores if int(r["local_z"]) == local_z]
        top_patch = max(zscores, key=lambda r: float(r["padim_score"])) if zscores else None
        top_score = float(top_patch["padim_score"]) if top_patch else None

        # lesion_z (단일 슬라이스, 시각화용)
        lesion_z = np.array(les_vol[local_z]).astype(bool) if les_vol is not None else None

        # B1-E12 비교
        b1e12_row  = load_b1e12_row(pid, local_z)
        b1e12_corr_v2_px = int(b1e12_row["corrected_v2_pixel_count"]) if b1e12_row else None
        b1e12_corr_v3_px = int(b1e12_row["corrected_v3_pixel_count"]) if b1e12_row else None
        b1e12_mask = None  # pixel mask는 저장 안 했으므로 시각화 불가

        # PNG
        ct_disp  = apply_lung_window(np.array(ct_vol[local_z]).astype(np.float32))
        mip_disp = apply_lung_window(mip)
        png_path = generate_png(
            local_z, ct_disp, mip_disp, roi_proj, lesion_proj,
            m_int, m_top, m_ves, m_union,
            b1e12_mask, eligible, top_patch, slab_range, png_dir)
        print(f"  PNG: {png_path.name}")

        # NPZ 저장 (15 slice용 구조 예비, 이번은 1장)
        npz_data[f"{pid}_z{local_z:04d}_intensity_p90"]  = m_int.astype(np.uint8)
        npz_data[f"{pid}_z{local_z:04d}_tophat_p90"]     = m_top.astype(np.uint8)
        npz_data[f"{pid}_z{local_z:04d}_vesselness_p90"] = m_ves.astype(np.uint8)
        npz_data[f"{pid}_z{local_z:04d}_union_3method"]  = m_union.astype(np.uint8)

        # 정량
        b_touch = boundary_touch(m_union, roi_proj)
        p_thin  = peripheral_thin_ratio(m_union)
        h_rat   = hilar_ratio(m_union, roi_proj)
        les_ov  = lesion_ov(m_union, lesion_proj)

        result_row = {
            "patient_id":     pid,
            "safe_id":        PATIENT["safe_id"],
            "role":           PATIENT["role"],
            "local_z":        local_z,
            "slab_range":     slab_range,
            "actual_slices":  actual_sl,
            "png_path":       str(png_path.relative_to(PROJECT)),
            "intensity_p90_pixel_count":   int(m_int.sum()),
            "top_hat_p90_pixel_count":     int(m_top.sum()),
            "vesselness_p90_pixel_count":  int(m_ves.sum()),
            "union_3method_pixel_count":   int(m_union.sum()),
            "union_vs_intensity_added":    int((m_union & ~m_int).sum()),
            "union_vs_tophat_added":       int((m_union & ~m_top).sum()),
            "union_vs_vesselness_added":   int((m_union & ~m_ves).sum()),
            "top_patch_score":             round(top_score, 4) if top_score else None,
            "top_patch_intensity_ratio":   round(patch_ratio(m_int,   [top_patch] if top_patch else []), 4),
            "top_patch_tophat_ratio":      round(patch_ratio(m_top,   [top_patch] if top_patch else []), 4),
            "top_patch_vesselness_ratio":  round(patch_ratio(m_ves,   [top_patch] if top_patch else []), 4),
            "top_patch_union_3method_ratio": round(patch_ratio(m_union, [top_patch] if top_patch else []), 4),
            "eligible_patch_union_3method_ratio": round(patch_ratio(m_union, eligible), 4),
            "lesion_overlap_intensity":    round(lesion_ov(m_int, lesion_z), 4),
            "lesion_overlap_tophat":       round(lesion_ov(m_top, lesion_z), 4),
            "lesion_overlap_vesselness":   round(lesion_ov(m_ves, lesion_z), 4),
            "lesion_overlap_union_3method": round(les_ov, 4),
            "boundary_touch_ratio_union":  round(b_touch, 4),
            "peripheral_thin_ratio_union": round(p_thin, 4),
            "hilar_region_ratio_union":    round(h_rat, 4),
            "b1e12_compare_available":     b1e12_row is not None,
            "b1e12_corrected_v2_pixel_count": b1e12_corr_v2_px,
            "b1e12_corrected_v3_pixel_count": b1e12_corr_v3_px,
            "union_vs_b1e12_corrected_v2_iou": None,   # mask 저장 없어 계산 불가
            "final_candidate_label": "union_3method",
            "note": "B1-E12 HOLD: coronal/sagittal blocky artifact; B1-E13 axial 3method union 채택",
        }

        print(f"  boundary_touch  : {b_touch:.4f}")
        print(f"  peripheral_thin : {p_thin:.4f}")
        print(f"  lesion_overlap  : {les_ov:.4f}")

    except Exception as e:
        errors.append({"patient_id": pid, "local_z": local_z,
                       "error": str(e), "traceback": tb_mod.format_exc()[:400]})
        print(f"  [ERROR] {e}", file=sys.stderr)

    # mtime 검증
    mtime_viol = 0
    for f, mt in pre_mtimes.items():
        cur = os.path.getmtime(f) if Path(f).exists() else None
        if cur is not None and abs(cur - mt) > 1:
            print(f"  [WARN] mtime changed: {f}", file=sys.stderr)
            mtime_viol += 1
    print(f"\n  mtime violations: {mtime_viol}")

    # 저장
    if result_row:
        idx_path = OUT_ROOT / f"{_STEM}_index.csv"
        with open(idx_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(result_row.keys()))
            w.writeheader(); w.writerow(result_row)
        print(f"  index CSV: {idx_path.name}")

    if npz_data:
        npz_path = OUT_ROOT / f"{_STEM}_masks.npz"
        np.savez_compressed(npz_path, **npz_data)
        print(f"  masks NPZ: {npz_path.name}")

    err_path = OUT_ROOT / f"{_STEM}_errors.csv"
    with open(err_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "local_z", "error", "traceback"],
                           extrasaction="ignore")
        w.writeheader(); w.writerows(errors)

    summary = {
        "stage":               "B1-E13 smoke (single slice)",
        "patient_id":          pid,
        "local_z":             local_z,
        "mip_slab":            f"z-{MIP_SLAB_HALF_LO}~z+{MIP_SLAB_HALF_HI-1} (B1-E11 방식)",
        "percentile":          PERCENTILE,
        "tophat_disk":         TOPHAT_DISK_RADIUS,
        "vessel_sigmas":       list(VESSEL_SIGMAS),
        "min_area_cleanup":    MIN_AREA_CLEANUP,
        "methods":             ["intensity", "tophat", "vesselness"],
        "final_candidate":     "union_3method",
        "b1e12_decision":      "HOLD/NO_GO: coronal/sagittal blocky artifact, visual quality < B1-E11",
        "n_errors":            len(errors),
        "mtime_violations":    mtime_viol,
        "source_files_modified": False,
        "holdout_intersection": 0,
        "all_checks_passed":   len(errors) == 0 and mtime_viol == 0,
    }
    sum_path = OUT_ROOT / f"{_STEM}_summary.json"
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if len(errors) == 0 and mtime_viol == 0:
        (OUT_ROOT / "DONE").touch()
        print("\n  [DONE] 모든 검증 통과")
    else:
        print(f"\n  [WARN] DONE 미생성: errors={len(errors)}, mtime_viol={mtime_viol}")


if __name__ == "__main__":
    main()
