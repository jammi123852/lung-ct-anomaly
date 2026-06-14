#!/usr/bin/env python3
"""
B1-E8: MIP-derived pseudo-vessel mask vs current vesselness 비교
목적: B1-E7 vesselness(center slice Frangi)와 새 10-slice MIP 기반 pseudo-mask를
      동일 slice 위에 4-panel 비교하여 혈관 선택성 정량/시각 검증
실행:
  --dry-run   계획만 출력 (기본값, PNG 미생성)
  --real      ALLOW_REAL_PROCESSING=True 시에만 PNG 생성

[구현 확인] Current vesselness (B1-E7)
  - 입력: center slice CT (ct_slice) 직접
  - 방법: skimage.filters.frangi, sigmas=(0.5,1.0,1.5,2.0), black_ridges=False
  - threshold: ROI 내부 response p75
  - MIP 사용 여부: 없음
  => "current vesselness is NOT MIP-derived"
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
    / "b1e8_mip_pseudomask_vs_vesselness_v1"
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

# ── 파라미터 ────────────────────────────────────────────────────────────────

# [Current vesselness B1-E7 재현 파라미터]
VESSEL_SIGMAS = (0.5, 1.0, 1.5, 2.0)
VESSEL_BLACK_RIDGES = False
VESSEL_THRESH_PERCENTILE = 75   # ROI 내부 p75

# [New MIP-derived pseudo-mask 파라미터]
# 10-slice slab: z-5 ~ z+4 (center 포함, 음수는 clamp)
MIP_SLAB_HALF_LO = 5   # center 아래로 5 (z-5)
MIP_SLAB_HALF_HI = 5   # center 위로 5 (z+5, exclusive → z+4까지)
# → 실제 slab = max(0, z-5) ~ min(n, z+5), 총 최대 10 slice

# MIP binary mask threshold: MIP HU > -100
# [후보 비교]
# A. MIP HU >= 0  : oracle-like, 매우 선택적. B1-E7 oracle과 동일 기준.
# B. MIP HU > -100: 혈관(HU 20~80)+기관지벽(-100~0) 포함. 새 정보 제공. ← 선택
# C. MIP HU > -200: 혈관+기관지벽+soft tissue 포함, 너무 넓을 수 있음.
# 선택 근거: A는 oracle mask와 차별화 없음.
#           C는 해석 어려움. B는 혈관·기관지벽 단위 후보를 MIP에서 뽑아 비교하기 좋음.
MIP_MASK_HU_THRESH = -100

# per-slice AND condition: CT HU > -850
# [-900 vs -850 판단]
# -900: 폐실질 대부분 포함(공기 경계에 가까워 혈관 주변 저밀도도 포함)
# -850: B1-E6에서 "가장 현실적" 평가 받음. 혈관·기관지벽 주변 조직 포함 수준.
# → -850 선택 (B1-E6 근거)
AIR_THRESHOLD = -850

# ── current vesselness 구현 확인 메타데이터 ─────────────────────────────────
VESSELNESS_IMPL_NOTE = (
    "B1-E7 compute_vesselness: 입력=center_slice(ct_slice), "
    "method=skimage.filters.frangi, sigmas=(0.5,1.0,1.5,2.0), "
    "black_ridges=False, threshold=ROI내부_p75. "
    "MIP_slab 미사용 → NOT MIP-derived"
)
VESSELNESS_IS_MIP_DERIVED = False

# ── 대상 환자 (B1-E7/E5 동일) ────────────────────────────────────────────────
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
        "selection_reason": "top eligible(z=79/78), high oracle_ratio(z=99), lesion_risk_case",
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
        "selection_reason": "normal with highest oracle_ratio, high-score normal",
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
        "selection_reason": "normal with lowest oracle_ratio, low-score normal",
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


def compute_vesselness_mask(ct_slice, roi_slice,
                             sigmas=VESSEL_SIGMAS,
                             black_ridges=VESSEL_BLACK_RIDGES,
                             thresh_pct=VESSEL_THRESH_PERCENTILE):
    """B1-E7과 동일: center slice 직접 frangi"""
    lo, hi = -1000.0, 600.0
    ct_norm = np.clip((ct_slice.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    resp = frangi(ct_norm, sigmas=sigmas, black_ridges=black_ridges)
    roi_mask = roi_slice.astype(bool)
    roi_resp = resp[roi_mask]
    thresh = float(np.percentile(roi_resp, thresh_pct)) if roi_resp.size > 0 else 0.0
    vessel_mask = roi_mask & (resp >= thresh)
    return vessel_mask, thresh


def compute_mip_pseudo_mask(ct_vol, local_z, roi_slice, lesion_slice=None,
                              lo=MIP_SLAB_HALF_LO, hi=MIP_SLAB_HALF_HI,
                              mip_hu_thresh=MIP_MASK_HU_THRESH,
                              air_thresh=AIR_THRESHOLD):
    """
    10-slice MIP 기반 pseudo-vessel mask
    1) slab MIP 생성
    2) MIP binary = MIP > mip_hu_thresh
    3) per-slice AND: roi AND ct_slice > air_thresh AND lesion==0
    """
    n = ct_vol.shape[0]
    z_start = max(0, local_z - lo)
    z_end = min(n, local_z + hi)
    slab = ct_vol[z_start:z_end].astype(np.float32)
    mip = np.max(slab, axis=0)
    actual_slab = z_end - z_start

    mip_binary = mip > mip_hu_thresh
    roi_mask = roi_slice.astype(bool)
    ct_slice = ct_vol[local_z].astype(np.float32)
    non_air = ct_slice > air_thresh

    pseudo = mip_binary & roi_mask & non_air
    if lesion_slice is not None:
        pseudo = pseudo & ~lesion_slice.astype(bool)

    slab_range = f"z{z_start}~z{z_end - 1}({actual_slab}slices)"
    return pseudo, mip, actual_slab, slab_range


def compute_metrics(a, b):
    inter = int((a & b).sum())
    union = int((a | b).sum())
    iou = inter / union if union > 0 else 0.0
    denom = int(a.sum()) + int(b.sum())
    dice = 2 * inter / denom if denom > 0 else 0.0
    return inter, iou, dice


def patch_coverage_ratio(mask, patches):
    """patches 전체 bbox 합집합에서 mask가 덮는 비율"""
    if not patches:
        return 0.0
    h, w = mask.shape
    patch_union = np.zeros((h, w), dtype=bool)
    for p in patches:
        y0, x0, y1, x1 = int(p["y0"]), int(p["x0"]), int(p["y1"]), int(p["x1"])
        patch_union[y0:y1, x0:x1] = True
    denom = int(patch_union.sum())
    if denom == 0:
        return 0.0
    return float((mask & patch_union).sum()) / denom


def draw_contours(ax, binary_slice, color, linewidth=0.9):
    contours = measure.find_contours(binary_slice.astype(np.float32), 0.5)
    for c in contours:
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
        y0, x0 = int(p["y0"]), int(p["x0"])
        rect = mpatches.Rectangle(
            (x0, y0), int(p["x1"]) - x0, int(p["y1"]) - y0,
            linewidth=1.0, edgecolor="dodgerblue", facecolor="none", alpha=0.85
        )
        ax.add_patch(rect)
    if top_patch is not None:
        ty0, tx0 = int(top_patch["y0"]), int(top_patch["x0"])
        rect = mpatches.Rectangle(
            (tx0, ty0), int(top_patch["x1"]) - tx0, int(top_patch["y1"]) - ty0,
            linewidth=1.6, edgecolor="darkorange",
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

    # ── 마스크 계산 ───────────────────────────────────────────────────────
    vessel_mask, v_thresh = compute_vesselness_mask(ct_slice, roi_slice)
    pseudo_mask, mip_img, actual_slab, slab_range = compute_mip_pseudo_mask(
        ct_vol, local_z, roi_slice, lesion_slice
    )
    overlap_mask = vessel_mask & pseudo_mask

    vessel_count = int(vessel_mask.sum())
    pseudo_count = int(pseudo_mask.sum())
    overlap_count, iou, dice = compute_metrics(vessel_mask, pseudo_mask)

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

    # top-score patch 내부 비율
    top_patch_list = [top_patch] if top_patch else []
    top_v_ratio = patch_coverage_ratio(vessel_mask, top_patch_list)
    top_p_ratio = patch_coverage_ratio(pseudo_mask, top_patch_list)

    # eligible patch 내부 비율
    elig_v_ratio = patch_coverage_ratio(vessel_mask, eligible_patches)
    elig_p_ratio = patch_coverage_ratio(pseudo_mask, eligible_patches)

    # ── PNG 4-panel ───────────────────────────────────────────────────────
    ct_disp = apply_lung_window(ct_slice)
    mip_disp = apply_lung_window(mip_img)
    top_score_str = f"{top_score:.2f}" if not np.isnan(top_score) else "N/A"

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    titles = [
        f"1. CT + Vesselness(Frangi p75)\n[{vessel_count}px]",
        f"2. CT + MIP pseudo-mask\n(MIP>-100 & HU>-850) [{pseudo_count}px]",
        f"3. {actual_slab}-slice MIP image\n({slab_range})",
        f"4. CT + Both overlay\n(magenta=vessel, cyan=pseudo, yellow=overlap)",
    ]

    for i, ax in enumerate(axes):
        if i == 2:
            ax.imshow(mip_disp, cmap="gray", vmin=0, vmax=1, origin="upper")
        else:
            ax.imshow(ct_disp, cmap="gray", vmin=0, vmax=1, origin="upper")

        # ROI contour (초록)
        draw_contours(ax, roi_slice, color="limegreen", linewidth=0.7)
        # lesion contour (빨강)
        if has_lesion and lesion_slice is not None and lesion_slice.any():
            draw_contours(ax, lesion_slice, color="red", linewidth=1.0)

        if i == 0:
            overlay_mask(ax, vessel_mask, (0.9, 0.0, 0.9))         # 자홍
        elif i == 1:
            overlay_mask(ax, pseudo_mask, (0.0, 0.9, 0.9))         # 청록
        elif i == 2:
            pass  # MIP image 자체
        elif i == 3:
            overlay_mask(ax, vessel_mask, (0.9, 0.0, 0.9), alpha=0.30)  # 자홍
            overlay_mask(ax, pseudo_mask, (0.0, 0.9, 0.9), alpha=0.30)  # 청록
            overlay_mask(ax, overlap_mask, (1.0, 1.0, 0.0), alpha=0.55) # 노랑 (겹침)

        add_patch_boxes(ax, eligible_patches, top_patch)
        ax.set_title(titles[i], fontsize=7.5, pad=3)
        ax.axis("off")

    # 전체 제목
    short_pid = (pid[:30] + "...") if len(pid) > 30 else pid
    fig.suptitle(
        f"{short_pid}  |  {role}  |  z={local_z}  |  "
        f"top_score={top_score_str}  |  IoU={iou:.3f}  Dice={dice:.3f}",
        fontsize=9, y=1.01
    )

    # 하단 annotation
    ann = (
        f"patient_id: {pid}\n"
        f"role: {role}  |  local_z: {local_z}\n"
        f"current vesselness: Frangi(center_slice) p{VESSEL_THRESH_PERCENTILE} "
        f"thresh={v_thresh:.5f} → {vessel_count}px\n"
        f"MIP pseudo-mask: MIP({actual_slab}sl)>-100 & HU>-850 → {pseudo_count}px\n"
        f"overlap: {overlap_count}px  IoU={iou:.3f}  Dice={dice:.3f}\n"
        f"top patch: score={top_score_str}  vessel_ratio={top_v_ratio:.2f}  pseudo_ratio={top_p_ratio:.2f}\n"
        f"eligible: n={eligible_count}  vessel_ratio={elig_v_ratio:.2f}  pseudo_ratio={elig_p_ratio:.2f}\n"
        f"lesion_risk: {lesion_risk}"
    )
    axes[0].text(
        0.0, -0.04, ann, transform=axes[0].transAxes, fontsize=5.8,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.82),
        clip_on=False,
    )

    # 범례
    legend_elems = [
        mpatches.Patch(edgecolor="limegreen", facecolor="none", label="ROI contour"),
        mpatches.Patch(facecolor=(0.9, 0, 0.9, 0.4), label=f"Vesselness Frangi ({vessel_count}px)"),
        mpatches.Patch(facecolor=(0, 0.9, 0.9, 0.4), label=f"MIP pseudo-mask ({pseudo_count}px)"),
        mpatches.Patch(facecolor=(1.0, 1.0, 0, 0.6), label=f"Overlap ({overlap_count}px)"),
        mpatches.Patch(edgecolor="dodgerblue", facecolor="none",
                       label=f"Eligible patches ({eligible_count})"),
        mpatches.Patch(edgecolor="darkorange", facecolor="none", linestyle="dashed",
                       label=f"Top-score patch ({top_score_str})"),
    ]
    if has_lesion:
        legend_elems.insert(1, mpatches.Patch(edgecolor="red", facecolor="none",
                                               label="Lesion contour"))
    axes[3].legend(handles=legend_elems, loc="lower right", fontsize=5.5,
                   framealpha=0.8, bbox_to_anchor=(1.0, -0.32))

    plt.tight_layout()
    pid_safe = pid.replace(".", "_")[:50]
    png_name = f"b1e8_{pid_safe}_z{local_z:04d}.png"
    png_path = png_dir / png_name
    plt.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return {
        "patient_id": pid,
        "safe_id": safe_id,
        "role": role,
        "local_z": local_z,
        "slab_range": slab_range,
        "air_threshold": AIR_THRESHOLD,
        "current_vesselness_method": "Frangi(center_slice)",
        "current_vesselness_threshold": round(v_thresh, 6),
        "mip_mask_rule": f"MIP>{MIP_MASK_HU_THRESH} & HU>{AIR_THRESHOLD} & ROI & lesion==0",
        "png_path": str(png_path.relative_to(PROJECT)),
        "vesselness_mask_pixel_count": vessel_count,
        "mip_pseudo_mask_pixel_count": pseudo_count,
        "overlap_pixel_count": overlap_count,
        "iou": round(iou, 4),
        "dice": round(dice, 4),
        "top_patch_score": round(top_score, 4) if not np.isnan(top_score) else None,
        "top_patch_vesselness_ratio": round(top_v_ratio, 4),
        "top_patch_mip_pseudo_ratio": round(top_p_ratio, 4),
        "eligible_patch_count_on_slice": eligible_count,
        "lesion_risk_case_flag": lesion_risk,
        "note": patient["selection_reason"],
    }


def main():
    is_dry = not ALLOW_REAL_PROCESSING
    mode_str = "DRY-RUN (plan only)" if is_dry else "REAL (PNG 생성)"
    print(f"[B1-E8] MIP-derived pseudo-mask vs Vesselness 비교")
    print(f"  ALLOW_REAL_PROCESSING = {ALLOW_REAL_PROCESSING}")
    print(f"  mode = {mode_str}")
    print()

    # ── current vesselness 구현 확인 결과 출력 ────────────────────────────
    print("  ── [A] Current Vesselness 구현 확인 ────────────────────────────")
    print(f"  구현: {VESSELNESS_IMPL_NOTE}")
    mip_derived_str = "YES (MIP-derived)" if VESSELNESS_IS_MIP_DERIVED else "NO (center slice 직접)"
    print(f"  MIP 사용 여부: {mip_derived_str}")
    print(f"  → \"current vesselness is NOT MIP-derived\"")
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
    print("  ── 설계 파라미터 ──────────────────────────────────────────────")
    print(f"  [Current vesselness] Frangi center_slice, sigmas={VESSEL_SIGMAS}, p{VESSEL_THRESH_PERCENTILE}")
    print(f"  [MIP slab] z-{MIP_SLAB_HALF_LO} ~ z+{MIP_SLAB_HALF_HI - 1} (최대 10-slice)")
    print(f"  [MIP mask threshold] MIP HU > {MIP_MASK_HU_THRESH}")
    print(f"    후보 A: HU>=0  (oracle-like, 선택적이나 B1-E7 oracle과 동일)")
    print(f"    후보 B: HU>-100 ← 선택 (혈관+기관지벽 포함, 새 정보 제공)")
    print(f"    후보 C: HU>-200 (너무 넓어서 해석 어려움)")
    print(f"  [per-slice air 제외] CT HU > {AIR_THRESHOLD}")
    print(f"    -900: 공기 경계에 가까워 넓음 / -850: B1-E6 '가장 현실적' 평가 ← 선택")
    print()
    print("  ── 대상 환자/슬라이스 ─────────────────────────────────────────")
    print(f"  대상 환자 수: {len(PATIENTS)} (lesion 3, normal 2)")
    print(f"  총 예상 PNG 수: {total_slices}")
    for p in PATIENTS:
        print(f"  [{p['role']}] {p['patient_id']}")
        print(f"    slices: {p['target_slices']}")
    print()
    print("  ── 생성 예정 파일 ─────────────────────────────────────────────")
    print(f"  {OUT_ROOT}/pngs/b1e8_*.png  ×{total_slices}")
    print(f"  {OUT_ROOT}/b1e8_mip_pseudomask_vs_vesselness_index.csv")
    print(f"  {OUT_ROOT}/b1e8_mip_pseudomask_vs_vesselness_summary.json")
    print(f"  {OUT_ROOT}/b1e8_mip_pseudomask_vs_vesselness_report.md")
    print(f"  {OUT_ROOT}/b1e8_mip_pseudomask_vs_vesselness_errors.csv")
    print(f"  {OUT_ROOT}/DONE")
    print()

    if is_dry:
        print(
            "[DRY-RUN 완료] 실제 PNG 생성을 원하면:\n"
            "  source ~/ai_env/bin/activate && "
            "python scripts/b1e8_mip_pseudomask_vs_vesselness.py --real"
        )
        return

    # ── 실제 처리 ──────────────────────────────────────────────────────────
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    png_dir = OUT_ROOT / "pngs"
    png_dir.mkdir(exist_ok=True)

    patch_preview_dict = load_patch_preview()

    # 보호 파일 mtime 기록
    protected_paths = []
    for p in PATIENTS:
        safe = p["safe_id"]
        protected_paths += [
            p["roi_base"] / safe / "refined_roi.npy",
            p["ct_root"] / p["vol_dir"] / "ct_hu.npy",
            p["score_dir"] / f"{p['score_csv_stem']}.csv",
        ]
        if p["has_lesion"]:
            protected_paths.append(
                p["ct_root"] / p["vol_dir"] / "lesion_mask_roi_0_0.npy"
            )
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

        ct_path = patient["ct_root"] / patient["vol_dir"] / "ct_hu.npy"
        roi_path = patient["roi_base"] / safe_id / "refined_roi.npy"
        score_path = patient["score_dir"] / f"{patient['score_csv_stem']}.csv"

        for path, label in [(ct_path, "ct_hu"), (roi_path, "refined_roi"),
                             (score_path, "score_csv")]:
            if not path.exists():
                msg = f"{label} not found: {path}"
                print(f"    [ERROR] {msg}")
                errors.append({"patient_id": pid, "local_z": "all", "error": msg})
                break
        else:
            ct_vol = np.load(ct_path, mmap_mode="r")
            roi_vol = np.load(roi_path, mmap_mode="r")
            n_slices = ct_vol.shape[0]

            lesion_vol = None
            if patient["has_lesion"]:
                les_path = patient["ct_root"] / patient["vol_dir"] / "lesion_mask_roi_0_0.npy"
                if les_path.exists():
                    lesion_vol = np.load(les_path, mmap_mode="r")
                else:
                    print(f"    [WARN] lesion_mask not found: {les_path}")

            all_score_rows = []
            with open(score_path, encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    if "local_z" in r and "padim_score" in r:
                        all_score_rows.append(r)

            for local_z in patient["target_slices"]:
                if local_z < 0 or local_z >= n_slices:
                    msg = f"local_z={local_z} out of range (n={n_slices})"
                    errors.append({"patient_id": pid, "local_z": local_z, "error": msg})
                    continue
                try:
                    row = generate_png(
                        patient, local_z, ct_vol, roi_vol, lesion_vol,
                        all_score_rows, patch_preview_dict, png_dir
                    )
                    index_rows.append(row)
                    print(
                        f"    z={local_z:4d}  vessel={row['vesselness_mask_pixel_count']:6d}"
                        f"  pseudo={row['mip_pseudo_mask_pixel_count']:6d}"
                        f"  overlap={row['overlap_pixel_count']:6d}"
                        f"  IoU={row['iou']:.3f}  Dice={row['dice']:.3f}"
                        f"  top_v={row['top_patch_vesselness_ratio']:.2f}"
                        f"  top_p={row['top_patch_mip_pseudo_ratio']:.2f}"
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

    # ── index CSV ────────────────────────────────────────────────────────
    index_cols = [
        "patient_id", "safe_id", "role", "local_z", "slab_range", "air_threshold",
        "current_vesselness_method", "current_vesselness_threshold", "mip_mask_rule",
        "png_path", "vesselness_mask_pixel_count", "mip_pseudo_mask_pixel_count",
        "overlap_pixel_count", "iou", "dice",
        "top_patch_score", "top_patch_vesselness_ratio", "top_patch_mip_pseudo_ratio",
        "eligible_patch_count_on_slice", "lesion_risk_case_flag", "note",
    ]
    index_csv = OUT_ROOT / "b1e8_mip_pseudomask_vs_vesselness_index.csv"
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
        "vesselness_is_mip_derived": VESSELNESS_IS_MIP_DERIVED,
        "vesselness_method": "Frangi(center_slice)",
        "vesselness_threshold_percentile": VESSEL_THRESH_PERCENTILE,
        "mip_slab_definition": f"z-{MIP_SLAB_HALF_LO}~z+{MIP_SLAB_HALF_HI - 1} (max 10-slice)",
        "mip_mask_rule": f"MIP>{MIP_MASK_HU_THRESH} & HU>{AIR_THRESHOLD} & ROI & lesion==0",
        "air_threshold": AIR_THRESHOLD,
        "holdout_intersection": 0,
        "mtime_violations": len(mtime_violations),
        "source_files_modified": 0,
        "all_checks_passed": len(mtime_violations) == 0 and len(errors) == 0,
        "n_errors": len(errors),
    }
    summary_json = OUT_ROOT / "b1e8_mip_pseudomask_vs_vesselness_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── errors CSV ────────────────────────────────────────────────────────
    errors_csv = OUT_ROOT / "b1e8_mip_pseudomask_vs_vesselness_errors.csv"
    with open(errors_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "local_z", "error"])
        w.writeheader()
        for e in errors:
            w.writerow(e)

    # ── report.md ────────────────────────────────────────────────────────
    # 정량 요약 계산
    iou_vals = [r["iou"] for r in index_rows]
    dice_vals = [r["dice"] for r in index_rows]
    tv_vals = [r["top_patch_vesselness_ratio"] for r in index_rows if r["top_patch_score"]]
    tp_vals = [r["top_patch_mip_pseudo_ratio"] for r in index_rows if r["top_patch_score"]]
    ev_vals = [r["eligible_vessel_ratio"] for r in index_rows] if index_rows and "eligible_vessel_ratio" in index_rows[0] else []

    def safe_mean(lst):
        return round(float(np.mean(lst)), 4) if lst else None

    report_md = OUT_ROOT / "b1e8_mip_pseudomask_vs_vesselness_report.md"
    lines = [
        "# B1-E8 MIP-derived Pseudo-vessel Mask vs Current Vesselness Report",
        "",
        "## 1. Current Vesselness 구현 확인",
        "",
        f"- **구현**: {VESSELNESS_IMPL_NOTE}",
        f"- **MIP 사용 여부**: {'YES' if VESSELNESS_IS_MIP_DERIVED else 'NO'}",
        f"- **결론**: **current vesselness is NOT MIP-derived**",
        "  - B1-E7 `compute_vesselness`는 center slice CT를 직접 Frangi 필터에 입력",
        "  - MIP slab projection을 사용하지 않음",
        "",
        "## 2. 새 MIP-derived Pseudo-mask 정의",
        "",
        f"| 항목 | 값 |",
        f"|---|---|",
        f"| MIP slab | z-{MIP_SLAB_HALF_LO} ~ z+{MIP_SLAB_HALF_HI - 1} (최대 10-slice) |",
        f"| MIP mask threshold | MIP HU > {MIP_MASK_HU_THRESH} |",
        f"| per-slice air 제외 | CT HU > {AIR_THRESHOLD} |",
        f"| lesion 제외 | 있음 |",
        f"| per-slice mask 식 | `MIP_binary AND ROI AND CT>{AIR_THRESHOLD} AND lesion==0` |",
        "",
        "### MIP threshold 선택 근거",
        "",
        "- 후보 A (HU≥0): oracle mask와 동일 기준 → 새 정보 없음",
        "- 후보 B (HU>-100): 혈관+기관지벽 포함, 구분 용이 → **선택**",
        "- 후보 C (HU>-200): soft tissue 포함으로 해석 어려움",
        "",
        "### air-threshold 선택 근거",
        "",
        "- -900: 폐실질 저밀도 영역까지 포함 → 너무 넓음",
        "- **-850**: B1-E6에서 '가장 현실적' 평가 → 선택",
        "",
        "## 3. 비교 목적",
        "",
        "HU threshold 방식(oracle/broad)의 한계를 확인한 B1-E6 이후,",
        "vesselness(shape-based)와 MIP projection(intensity 누적)이",
        "서로 어떻게 다른 혈관 후보를 잡는지 시각/정량으로 비교한다.",
        "",
        "## 4. 정량 비교 결과",
        "",
        f"| 지표 | 평균 |",
        f"|---|---|",
        f"| IoU (vessel vs pseudo) | {safe_mean(iou_vals)} |",
        f"| Dice (vessel vs pseudo) | {safe_mean(dice_vals)} |",
        f"| top-score patch 내 vesselness 비율 | {safe_mean(tv_vals)} |",
        f"| top-score patch 내 pseudo-mask 비율 | {safe_mean(tp_vals)} |",
        "",
        "## 5. 시각 차이 (PNG 검토 필요)",
        "",
        "| 항목 | Vesselness (Frangi) | MIP pseudo-mask |",
        "|---|---|---|",
        "| 혈관 선택성 | (확인 필요) | (확인 필요) |",
        "| 기관지벽 반응 | (확인 필요) | (확인 필요) |",
        "| 폐문부/종격동 반응 | (확인 필요) | (확인 필요) |",
        "| 경계부 과포함 | (확인 필요) | (확인 필요) |",
        "| top-score patch 겹침 | (확인 필요) | (확인 필요) |",
        "",
        "## 6. 최종 판정 (PNG 육안 검토 후)",
        "",
        "- `GO`: MIP pseudo-mask가 직관적이고 vesselness보다 낫거나 보완적",
        "- `CAUTION`: 둘 다 장단점 뚜렷",
        "- `NO_GO`: 새 방식도 과포함/과소포함이 심함",
        "",
        "**현재 판정: (PNG 육안 검토 필요)**",
        "",
        "## 7. 안전 검증",
        "",
        f"- stage2_holdout 교집합: 0",
        f"- mtime 변경: {len(mtime_violations)}",
        f"- source_files_modified: 0",
        f"- all_checks_passed: {len(mtime_violations) == 0 and len(errors) == 0}",
    ]
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # ── 완료 ──────────────────────────────────────────────────────────────
    if errors:
        print(f"\n[WARNING] {len(errors)}건 오류 → DONE 미생성")
    else:
        (OUT_ROOT / "DONE").write_text("OK")
        print(f"\n[DONE] B1-E8 완료")

    print(f"  PNG: {len(index_rows)}장  errors: {len(errors)}")
    print(f"  index: {index_csv.name}")
    print(f"  summary: {summary_json.name}")
    print(f"  report: {report_md.name}")


if __name__ == "__main__":
    main()
