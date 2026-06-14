#!/usr/bin/env python3
"""
B1-E7: Thin-slab MIP + vesselness visual smoke
목적: HU threshold 기반 oracle/broad mask vs thin-slab MIP vs vesselness(Frangi)를
      동일 slice 위에 5-panel 비교해서 혈관 선택성 육안 검증
실행:
  --dry-run   계획만 출력 (기본값, PNG 미생성)
  --real      ALLOW_REAL_PROCESSING=True 시에만 PNG 생성
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

ALLOW_REAL_PROCESSING = "--real" in sys.argv

# ── 경로 설정 ────────────────────────────────────────────────────────────────
PROJECT = Path("/home/jinhy/project/lung-ct-anomaly")
OUT_ROOT = (
    PROJECT
    / "outputs/position-aware-padim-v1"
    / "oracle_vessel_suppression_effb0_v4_20_dev_only"
    / "b1e7_mip_vesselness_visual_smoke_v1"
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

# ── MIP / vesselness 설정 ───────────────────────────────────────────────────
MIP_SLAB_HALF = 2          # z±2 → 5-slice slab (center ± 2)
MIP_SLAB_THICKNESS = MIP_SLAB_HALF * 2 + 1  # = 5

VESSELNESS_METHOD = "Frangi"
# sigmas: 혈관 두께 추정 범위 (pixel 단위)
VESSELNESS_SIGMAS = (0.5, 1.0, 1.5, 2.0)
# black_ridges=False → 밝은(HU 높은) tubular structure 탐지
VESSELNESS_BLACK_RIDGES = False

# threshold 후보: p50 / p75 / p90 (ROI 내부 vesselness response 기준)
VESSELNESS_THRESH_PERCENTILES = [50, 75, 90]
VESSELNESS_THRESH_REPR = 75   # PNG에 표시할 대표 threshold percentile

# broad mask HU threshold (B1-E6에서 "가장 현실적"으로 본 -850)
BROAD_HU_THRESH = -850

# ── 대상 환자 (B1-E5/E6 동일) ────────────────────────────────────────────────
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
        "selection_reason": "top eligible(z=79/78), high oracle_ratio eligible(z=99,ratio=0.216), lesion_risk_case",
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
        "selection_reason": "top eligible cluster(z=84~86), oracle_ratio 0.18~0.21",
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
        "selection_reason": "top eligible cluster(z=126~129), oracle_ratio 0.10~0.18",
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
        "selection_reason": "normal with highest oracle_ratio(0.37/0.35/0.12), high-score normal",
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
        "selection_reason": "normal with lowest oracle_ratio(0.001393 overall), low-score normal",
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


def apply_lung_window(ct_slice, level=-600, width=1500):
    lo = level - width / 2
    hi = level + width / 2
    img = np.clip(ct_slice.astype(np.float32), lo, hi)
    return (img - lo) / (hi - lo)


def compute_oracle_mask(ct_slice, roi_slice, lesion_slice=None):
    in_roi = roi_slice.astype(bool)
    bright = ct_slice >= 0
    mask = in_roi & bright
    if lesion_slice is not None:
        mask = mask & ~lesion_slice.astype(bool)
    return mask


def compute_broad_mask(ct_slice, roi_slice, lesion_slice=None, hu_thresh=BROAD_HU_THRESH):
    in_roi = roi_slice.astype(bool)
    non_air = ct_slice > hu_thresh
    mask = in_roi & non_air
    if lesion_slice is not None:
        mask = mask & ~lesion_slice.astype(bool)
    return mask


def compute_mip_slice(ct_vol, local_z, slab_half=MIP_SLAB_HALF):
    n_slices = ct_vol.shape[0]
    z_start = max(0, local_z - slab_half)
    z_end = min(n_slices, local_z + slab_half + 1)
    slab = ct_vol[z_start:z_end].astype(np.float32)
    return np.max(slab, axis=0), z_end - z_start


def compute_vesselness(ct_slice, roi_slice, sigmas=VESSELNESS_SIGMAS,
                       black_ridges=VESSELNESS_BLACK_RIDGES):
    # HU → 정규화 (Frangi는 0~1 입력 권장)
    ct_f = ct_slice.astype(np.float32)
    lo, hi = -1000.0, 600.0
    ct_norm = np.clip((ct_f - lo) / (hi - lo), 0.0, 1.0)

    resp = frangi(ct_norm, sigmas=sigmas, black_ridges=black_ridges)

    # ROI 내부 response 기준으로 percentile threshold 산출
    roi_mask = roi_slice.astype(bool)
    roi_resp = resp[roi_mask]
    thresholds = {}
    for pct in VESSELNESS_THRESH_PERCENTILES:
        thresholds[pct] = float(np.percentile(roi_resp, pct)) if roi_resp.size > 0 else 0.0

    # 대표 threshold 기반 binary mask
    repr_thresh = thresholds[VESSELNESS_THRESH_REPR]
    vessel_mask = roi_mask & (resp >= repr_thresh)

    return resp, vessel_mask, thresholds


def draw_contours(ax, binary_slice, color, linewidth=0.9):
    contours = measure.find_contours(binary_slice.astype(np.float32), 0.5)
    for c in contours:
        ax.plot(c[:, 1], c[:, 0], color=color, linewidth=linewidth)


def overlay_mask(ax, mask, color_rgb, alpha=0.35):
    ovl = np.zeros((*mask.shape, 4), dtype=np.float32)
    ovl[mask, 0] = color_rgb[0]
    ovl[mask, 1] = color_rgb[1]
    ovl[mask, 2] = color_rgb[2]
    ovl[mask, 3] = alpha
    ax.imshow(ovl, origin="upper")


def add_patch_boxes(ax, eligible_patches, top_patch):
    for p in eligible_patches:
        y0, x0 = int(p["y0"]), int(p["x0"])
        h = int(p["y1"]) - y0
        w = int(p["x1"]) - x0
        rect = mpatches.Rectangle(
            (x0, y0), w, h, linewidth=1.0, edgecolor="dodgerblue",
            facecolor="none", alpha=0.85
        )
        ax.add_patch(rect)
    if top_patch is not None:
        ty0, tx0 = int(top_patch["y0"]), int(top_patch["x0"])
        th = int(top_patch["y1"]) - ty0
        tw = int(top_patch["x1"]) - tx0
        rect = mpatches.Rectangle(
            (tx0, ty0), tw, th, linewidth=1.6, edgecolor="darkorange",
            facecolor="none", alpha=0.9, linestyle="--"
        )
        ax.add_patch(rect)


def generate_png(patient, local_z, ct_vol, roi_vol, lesion_vol,
                 all_score_rows, patch_preview_dict, png_dir):
    pid = patient["patient_id"]
    safe_id = patient["safe_id"]
    role = patient["role"]
    has_lesion = patient["has_lesion"]
    lesion_risk = patient["lesion_risk_case_flag"]

    ct_slice = np.array(ct_vol[local_z])
    roi_slice = np.array(roi_vol[local_z])
    lesion_slice = (
        np.array(lesion_vol[local_z])
        if has_lesion and lesion_vol is not None
        else None
    )

    # ── 각 마스크 계산 ────────────────────────────────────────────────────
    oracle_mask = compute_oracle_mask(ct_slice, roi_slice, lesion_slice)
    broad_mask = compute_broad_mask(ct_slice, roi_slice, lesion_slice,
                                    hu_thresh=BROAD_HU_THRESH)
    mip_slice, actual_slab = compute_mip_slice(ct_vol, local_z, MIP_SLAB_HALF)
    vesselness_resp, vessel_mask, v_thresholds = compute_vesselness(
        ct_slice, roi_slice
    )

    oracle_count = int(oracle_mask.sum())
    broad_count = int(broad_mask.sum())
    vessel_count = int(vessel_mask.sum())

    # ── patch 정보 ────────────────────────────────────────────────────────
    key = (pid, local_z)
    slice_preview = patch_preview_dict.get(key, [])
    eligible_patches = [r for r in slice_preview if r["suppression_eligible"] == "True"]
    eligible_count = len(eligible_patches)

    slice_score_rows = [r for r in all_score_rows if int(r["local_z"]) == local_z]
    top_patch = (
        max(slice_score_rows, key=lambda r: float(r["padim_score"]))
        if slice_score_rows else None
    )
    top_score = float(top_patch["padim_score"]) if top_patch else float("nan")

    # ── PNG 5-panel ───────────────────────────────────────────────────────
    ct_disp = apply_lung_window(ct_slice)
    mip_disp = apply_lung_window(mip_slice)

    fig, axes = plt.subplots(1, 5, figsize=(28, 6))
    panel_titles = [
        "1. Original CT",
        "2. Oracle mask\n(HU≥0, lesion excl.)",
        f"3. Broad mask\n(HU>{BROAD_HU_THRESH}, lesion excl.)",
        f"4. Thin-slab MIP\n(z±{MIP_SLAB_HALF}, {actual_slab}-slice)",
        f"5. Vesselness\n({VESSELNESS_METHOD}, p{VESSELNESS_THRESH_REPR})",
    ]
    panel_colors = [None, (1.0, 1.0, 0.0), (0.0, 0.9, 0.9), None, (0.9, 0.0, 0.9)]

    for i, ax in enumerate(axes):
        if i == 3:
            ax.imshow(mip_disp, cmap="gray", vmin=0, vmax=1, origin="upper")
        else:
            ax.imshow(ct_disp, cmap="gray", vmin=0, vmax=1, origin="upper")

        # ROI contour (초록) - 패널 1,2,3,5에만 (MIP엔 선택적)
        draw_contours(ax, roi_slice, color="limegreen", linewidth=0.7)

        # lesion contour (빨강)
        if has_lesion and lesion_slice is not None and lesion_slice.any():
            draw_contours(ax, lesion_slice, color="red", linewidth=1.0)

        # 마스크 overlay
        if i == 1:
            overlay_mask(ax, oracle_mask, (1.0, 1.0, 0.0))
        elif i == 2:
            overlay_mask(ax, broad_mask, (0.0, 0.9, 0.9))
        elif i == 3:
            # MIP: ROI contour만 (overlay 없음 - MIP 자체가 비교 대상)
            pass
        elif i == 4:
            overlay_mask(ax, vessel_mask, (0.9, 0.0, 0.9))

        # patch bbox는 모든 패널에 표시
        add_patch_boxes(ax, eligible_patches, top_patch)

        ax.set_title(panel_titles[i], fontsize=7.5, pad=3)
        ax.axis("off")

    # 전체 제목
    short_pid = (pid[:30] + "...") if len(pid) > 30 else pid
    top_score_str = f"{top_score:.2f}" if not np.isnan(top_score) else "N/A"
    fig.suptitle(
        f"{short_pid}  |  {role}  |  z={local_z}  |  top_score={top_score_str}",
        fontsize=9, y=1.01
    )

    # 하단 annotation (좌측 패널 아래)
    v_thresh_str = " / ".join(
        f"p{p}={v_thresholds[p]:.4f}" for p in VESSELNESS_THRESH_PERCENTILES
    )
    ann = (
        f"patient_id: {pid}\n"
        f"role: {role}  |  local_z: {local_z}\n"
        f"oracle_px: {oracle_count}  |  broad_px (HU>{BROAD_HU_THRESH}): {broad_count}\n"
        f"vesselness_px (p{VESSELNESS_THRESH_REPR}): {vessel_count}\n"
        f"vesselness thresholds: {v_thresh_str}\n"
        f"mip_slab: {actual_slab}-slice (z±{MIP_SLAB_HALF})\n"
        f"eligible_patches: {eligible_count}  |  top_score: {top_score_str}\n"
        f"lesion_risk: {lesion_risk}"
    )
    axes[0].text(
        0.0, -0.04, ann, transform=axes[0].transAxes, fontsize=6.0,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.82),
        clip_on=False,
    )

    # 범례 (마지막 패널 아래)
    legend_elems = [
        mpatches.Patch(edgecolor="limegreen", facecolor="none", label="ROI contour"),
        mpatches.Patch(facecolor=(1, 1, 0, 0.4), label=f"Oracle (HU≥0): {oracle_count}px"),
        mpatches.Patch(facecolor=(0, 0.9, 0.9, 0.4), label=f"Broad (HU>{BROAD_HU_THRESH}): {broad_count}px"),
        mpatches.Patch(facecolor=(0.9, 0, 0.9, 0.4), label=f"Vesselness p{VESSELNESS_THRESH_REPR}: {vessel_count}px"),
        mpatches.Patch(edgecolor="dodgerblue", facecolor="none", label=f"Eligible patches ({eligible_count})"),
        mpatches.Patch(edgecolor="darkorange", facecolor="none", linestyle="dashed",
                       label=f"Top-score patch ({top_score_str})"),
    ]
    if has_lesion:
        legend_elems.insert(1, mpatches.Patch(edgecolor="red", facecolor="none",
                                               label="Lesion contour"))
    axes[4].legend(handles=legend_elems, loc="lower right", fontsize=5.5,
                   framealpha=0.8, bbox_to_anchor=(1.0, -0.28))

    plt.tight_layout()

    pid_safe = pid.replace(".", "_")[:50]
    png_name = f"b1e7_{pid_safe}_z{local_z:04d}.png"
    png_path = png_dir / png_name
    plt.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return {
        "patient_id": pid,
        "safe_id": safe_id,
        "role": role,
        "local_z": local_z,
        "mip_slab_thickness": actual_slab,
        "vesselness_method": VESSELNESS_METHOD,
        "vesselness_threshold": round(v_thresholds[VESSELNESS_THRESH_REPR], 6),
        "png_path": str(png_path.relative_to(PROJECT)),
        "oracle_mask_pixel_count": oracle_count,
        "broad_mask_pixel_count": broad_count,
        "vesselness_mask_pixel_count": vessel_count,
        "eligible_patch_count_on_slice": eligible_count,
        "top_patch_score": (round(top_score, 4) if not np.isnan(top_score) else None),
        "lesion_risk_case_flag": lesion_risk,
        "note": patient["selection_reason"],
    }


def main():
    is_dry = not ALLOW_REAL_PROCESSING
    mode_str = "DRY-RUN (plan only)" if is_dry else "REAL (PNG 생성)"
    print(f"[B1-E7] Thin-slab MIP + Vesselness Visual Smoke")
    print(f"  ALLOW_REAL_PROCESSING = {ALLOW_REAL_PROCESSING}")
    print(f"  mode = {mode_str}")
    print()

    # ── holdout denylist 확인 ──────────────────────────────────────────────
    holdout = load_stage2_holdout()
    holdout_intersection = [p["patient_id"] for p in PATIENTS if p["patient_id"] in holdout]
    if holdout_intersection:
        print(f"[ABORT] stage2_holdout 교집합 발견: {holdout_intersection}", file=sys.stderr)
        sys.exit(1)
    print(f"  stage2_holdout 교집합: 0 (PASS)")

    # ── output root 존재 여부 ──────────────────────────────────────────────
    if OUT_ROOT.exists():
        print(f"[ABORT] output root 이미 존재 → 즉시 중단: {OUT_ROOT}", file=sys.stderr)
        sys.exit(1)
    print(f"  output root 없음 (PASS): {OUT_ROOT.name}")

    # ── 계획 출력 ──────────────────────────────────────────────────────────
    total_slices = sum(len(p["target_slices"]) for p in PATIENTS)
    print()
    print(f"  ── 설계 파라미터 ──────────────────────────────────────────")
    print(f"  MIP slab: {MIP_SLAB_THICKNESS}-slice (center z±{MIP_SLAB_HALF})")
    print(f"  broad mask HU threshold: HU > {BROAD_HU_THRESH}")
    print(f"  vesselness method: {VESSELNESS_METHOD}")
    print(f"  vesselness sigmas: {VESSELNESS_SIGMAS}")
    print(f"  vesselness black_ridges: {VESSELNESS_BLACK_RIDGES}")
    print(f"  vesselness threshold 후보: p{VESSELNESS_THRESH_PERCENTILES}")
    print(f"  PNG 대표 threshold: p{VESSELNESS_THRESH_REPR}")
    print()
    print(f"  ── 대상 환자/슬라이스 ────────────────────────────────────")
    print(f"  대상 환자 수: {len(PATIENTS)} (lesion 3, normal 2)")
    print(f"  총 예상 PNG 수: {total_slices}")
    print()
    for p in PATIENTS:
        print(f"  [{p['role']}] {p['patient_id']}")
        print(f"    slices: {p['target_slices']}")
        print(f"    reason: {p['selection_reason']}")
        print()

    print(f"  ── 생성 예정 파일 ──────────────────────────────────────────")
    print(f"  {OUT_ROOT}/pngs/b1e7_*.png  ×{total_slices}")
    print(f"  {OUT_ROOT}/b1e7_mip_vesselness_visual_smoke_index.csv")
    print(f"  {OUT_ROOT}/b1e7_mip_vesselness_visual_smoke_summary.json")
    print(f"  {OUT_ROOT}/b1e7_mip_vesselness_visual_smoke_report.md")
    print(f"  {OUT_ROOT}/b1e7_mip_vesselness_visual_smoke_errors.csv")
    print(f"  {OUT_ROOT}/DONE")
    print()

    if is_dry:
        print(
            "[DRY-RUN 완료] 실제 PNG 생성을 원하면:\n"
            "  source ~/ai_env/bin/activate && "
            "python scripts/b1e7_mip_vesselness_visual_smoke.py --real"
        )
        return

    # ── 실제 처리 ──────────────────────────────────────────────────────────
    if not ALLOW_REAL_PROCESSING:
        print("[ERROR] ALLOW_REAL_PROCESSING=False. --real 없이 실행 금지.", file=sys.stderr)
        sys.exit(1)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    png_dir = OUT_ROOT / "pngs"
    png_dir.mkdir(exist_ok=True)

    patch_preview_dict = load_patch_preview()

    # 보호 파일 mtime 기록
    protected_paths = []
    for p in PATIENTS:
        safe = p["safe_id"]
        roi_path = p["roi_base"] / safe / "refined_roi.npy"
        ct_path = p["ct_root"] / p["vol_dir"] / "ct_hu.npy"
        score_path = p["score_dir"] / f"{p['score_csv_stem']}.csv"
        protected_paths += [roi_path, ct_path, score_path]
        if p["has_lesion"]:
            les_path = p["ct_root"] / p["vol_dir"] / "lesion_mask_roi_0_0.npy"
            protected_paths.append(les_path)
    pre_mtimes = {
        str(f): os.path.getmtime(f)
        for f in protected_paths if Path(f).exists()
    }

    index_rows = []
    errors = []
    mtime_violations = []

    for patient in PATIENTS:
        pid = patient["patient_id"]
        safe_id = patient["safe_id"]
        print(f"  processing: {pid}")

        # 경로 확인
        ct_path = patient["ct_root"] / patient["vol_dir"] / "ct_hu.npy"
        roi_path = patient["roi_base"] / safe_id / "refined_roi.npy"
        score_path = patient["score_dir"] / f"{patient['score_csv_stem']}.csv"

        if not ct_path.exists():
            msg = f"ct_hu.npy not found: {ct_path}"
            print(f"    [ERROR] {msg}")
            errors.append({"patient_id": pid, "local_z": "all", "error": msg})
            continue
        if not roi_path.exists():
            msg = f"refined_roi.npy not found: {roi_path}"
            print(f"    [ERROR] {msg}")
            errors.append({"patient_id": pid, "local_z": "all", "error": msg})
            continue
        if not score_path.exists():
            msg = f"score csv not found: {score_path}"
            print(f"    [ERROR] {msg}")
            errors.append({"patient_id": pid, "local_z": "all", "error": msg})
            continue

        # CT / ROI 로드
        ct_vol = np.load(ct_path, mmap_mode="r")
        roi_vol = np.load(roi_path, mmap_mode="r")
        n_slices = ct_vol.shape[0]

        # lesion mask
        lesion_vol = None
        if patient["has_lesion"]:
            les_path = patient["ct_root"] / patient["vol_dir"] / "lesion_mask_roi_0_0.npy"
            if les_path.exists():
                lesion_vol = np.load(les_path, mmap_mode="r")
            else:
                print(f"    [WARN] lesion_mask_roi_0_0.npy not found: {les_path}")

        # score 로드
        all_score_rows = []
        with open(score_path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if "local_z" in r and "padim_score" in r:
                    all_score_rows.append(r)

        for local_z in patient["target_slices"]:
            if local_z < 0 or local_z >= n_slices:
                msg = f"local_z={local_z} out of range (n={n_slices})"
                print(f"    [ERROR] {msg}")
                errors.append({"patient_id": pid, "local_z": local_z, "error": msg})
                continue
            try:
                row = generate_png(
                    patient, local_z, ct_vol, roi_vol, lesion_vol,
                    all_score_rows, patch_preview_dict, png_dir
                )
                index_rows.append(row)
                print(
                    f"    z={local_z:4d}  oracle={row['oracle_mask_pixel_count']:5d}"
                    f"  broad={row['broad_mask_pixel_count']:6d}"
                    f"  vessel(p{VESSELNESS_THRESH_REPR})={row['vesselness_mask_pixel_count']:5d}"
                    f"  eligible={row['eligible_patch_count_on_slice']:3d}"
                )
            except Exception as e:
                import traceback
                msg = traceback.format_exc()
                print(f"    [ERROR] z={local_z}: {e}")
                errors.append({"patient_id": pid, "local_z": local_z, "error": str(msg)})

    # ── mtime 검사 ────────────────────────────────────────────────────────
    for fpath, pre in pre_mtimes.items():
        if Path(fpath).exists():
            post = os.path.getmtime(fpath)
            if abs(post - pre) > 0.01:
                mtime_violations.append(fpath)
    if mtime_violations:
        print(f"[ABORT] mtime 변경 감지: {mtime_violations}", file=sys.stderr)
        sys.exit(1)

    # ── index CSV 저장 ────────────────────────────────────────────────────
    index_cols = [
        "patient_id", "safe_id", "role", "local_z",
        "mip_slab_thickness", "vesselness_method", "vesselness_threshold",
        "png_path", "oracle_mask_pixel_count", "broad_mask_pixel_count",
        "vesselness_mask_pixel_count", "eligible_patch_count_on_slice",
        "top_patch_score", "lesion_risk_case_flag", "note",
    ]
    index_csv = OUT_ROOT / "b1e7_mip_vesselness_visual_smoke_index.csv"
    with open(index_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=index_cols)
        w.writeheader()
        for row in index_rows:
            w.writerow({k: row.get(k, "") for k in index_cols})

    # ── summary JSON ──────────────────────────────────────────────────────
    patient_list = sorted(set(r["patient_id"] for r in index_rows))
    summary = {
        "n_patients": len(patient_list),
        "n_png": len(index_rows),
        "patient_list": patient_list,
        "mip_slab_thickness": MIP_SLAB_THICKNESS,
        "vesselness_method": VESSELNESS_METHOD,
        "vesselness_sigmas": list(VESSELNESS_SIGMAS),
        "vesselness_thresholds_tested": VESSELNESS_THRESH_PERCENTILES,
        "vesselness_threshold_repr": VESSELNESS_THRESH_REPR,
        "broad_hu_thresh": BROAD_HU_THRESH,
        "holdout_intersection": 0,
        "mtime_violations": len(mtime_violations),
        "source_files_modified": 0,
        "all_checks_passed": len(mtime_violations) == 0 and len(errors) == 0,
        "n_errors": len(errors),
    }
    summary_json = OUT_ROOT / "b1e7_mip_vesselness_visual_smoke_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── errors CSV ────────────────────────────────────────────────────────
    errors_csv = OUT_ROOT / "b1e7_mip_vesselness_visual_smoke_errors.csv"
    with open(errors_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "local_z", "error"])
        w.writeheader()
        for e in errors:
            w.writerow(e)

    # ── report.md ────────────────────────────────────────────────────────
    report_md = OUT_ROOT / "b1e7_mip_vesselness_visual_smoke_report.md"
    report_lines = [
        "# B1-E7 Thin-slab MIP + Vesselness Visual Smoke Report",
        "",
        "## 단계 정의",
        "",
        "- **이 단계는 visual smoke / preflight 성격이다.**",
        "  suppression 적용 실험이 아니며, 성능 평가도 아니다.",
        "- 목적: HU threshold 방식(oracle / broad)과 thin-slab MIP / vesselness 방식이",
        "  동일 slice에서 혈관 구조를 얼마나 다르게 드러내는지 시각적으로 비교한다.",
        "",
        "## 설계 파라미터",
        "",
        f"| 항목 | 값 |",
        f"|---|---|",
        f"| MIP slab | {MIP_SLAB_THICKNESS}-slice (center z±{MIP_SLAB_HALF}) |",
        f"| broad mask HU threshold | HU > {BROAD_HU_THRESH} |",
        f"| vesselness method | {VESSELNESS_METHOD} (skimage.filters.frangi) |",
        f"| vesselness sigmas | {VESSELNESS_SIGMAS} |",
        f"| vesselness black_ridges | {VESSELNESS_BLACK_RIDGES} (밝은 tubular 구조 탐지) |",
        f"| vesselness threshold 후보 | p{VESSELNESS_THRESH_PERCENTILES} (ROI 내부 response 기준) |",
        f"| PNG 대표 threshold | p{VESSELNESS_THRESH_REPR} |",
        "",
        "## 방법론 설명",
        "",
        "### Thin-slab MIP",
        "- Center slice 단독이 아니라 **주변 ±2 slice를 포함한 5-slice slab의 maximum intensity projection**이다.",
        "- 혈관이 slice 사이에 불연속적으로 나타나는 경우, MIP는 해당 위치의 최대 HU를 투영해",
        "  혈관 구조를 단일 slice보다 더 연속적으로 드러낸다.",
        "- MIP는 overlay mask가 아닌 **CT 영상 자체**로 비교된다 (패널 4).",
        "",
        "### Vesselness (Frangi filter)",
        "- Intensity threshold가 아니라 **tubular structure의 기하학적 형태(Hessian eigenvalue 기반)**",
        "  를 강조하는 filter이다.",
        "- multi-scale (sigmas) 적용으로 다양한 두께의 혈관에 반응한다.",
        "- 출력은 vesselness response를 ROI 내부 percentile로 threshold한 binary candidate mask이다.",
        "",
        "## 비교 패널 구성",
        "",
        "| 패널 | 내용 |",
        "|---|---|",
        "| 1. Original CT | center slice 원본 (lung window) |",
        "| 2. Oracle mask | HU≥0, ROI 내부, lesion 제외 |",
        f"| 3. Broad mask | HU>{BROAD_HU_THRESH}, ROI 내부, lesion 제외 |",
        f"| 4. Thin-slab MIP | {MIP_SLAB_THICKNESS}-slice MIP 영상 (overlay 없음) |",
        f"| 5. Vesselness | Frangi p{VESSELNESS_THRESH_REPR} candidate mask overlay |",
        "",
        "## 비교 평가 항목",
        "",
        "각 환자/slice PNG를 보면서 아래를 확인한다:",
        "",
        "- **혈관 선택성**: oracle / broad / vesselness 중 가는 혈관을 더 선택적으로 잡는가?",
        "- **기관지벽 반응**: 기관지벽이 과도하게 포함되는가?",
        "- **폐문부/종격동 반응**: 폐문부나 종격동 구조까지 같이 잡히는가?",
        "- **경계부 과포함**: ROI 경계부에서 비혈관 구조가 포함되는가?",
        "- **top-score patch와의 겹침**: 각 mask가 top-score patch와 실제로 겹치는가?",
        "",
        "## 산출물",
        "",
        f"- PNG: {len(index_rows)}장",
        f"- patients: {len(patient_list)}명",
        f"- errors: {len(errors)}건",
        "",
        "## 1차 결론 (결과 확인 후 채워야 함)",
        "",
        "아래는 PNG 육안 검토 후 채워야 할 항목이다.",
        "",
        "- 혈관 선택성 비교: (확인 필요)",
        "- 기관지벽/폐문부 반응: (확인 필요)",
        "- top-score patch 겹침: (확인 필요)",
        "",
        "### 후속 단계 권장",
        "",
        "- `GO`: vesselness/MIP가 확실히 더 선택적",
        "- `CAUTION`: 일부 개선 있으나 비혈관 구조 반응 큼",
        "- `NO_GO`: HU threshold와 큰 차이 없음",
        "",
        "**현재 판정: (PNG 육안 검토 필요)**",
        "",
        "## 안전 검증",
        "",
        f"- stage2_holdout 교집합: 0",
        f"- mtime 변경: {len(mtime_violations)}",
        f"- source_files_modified: 0",
        f"- all_checks_passed: {len(mtime_violations) == 0 and len(errors) == 0}",
    ]
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    # ── 완료 ──────────────────────────────────────────────────────────────
    if errors:
        print(f"\n[WARNING] {len(errors)}건 오류 발생 → DONE 미생성")
        print(f"  errors: {errors_csv}")
    else:
        (OUT_ROOT / "DONE").write_text("OK")
        print(f"\n[DONE] B1-E7 완료")

    print(f"  PNG: {len(index_rows)}장")
    print(f"  index: {index_csv.name}")
    print(f"  summary: {summary_json.name}")
    print(f"  report: {report_md.name}")


if __name__ == "__main__":
    main()
