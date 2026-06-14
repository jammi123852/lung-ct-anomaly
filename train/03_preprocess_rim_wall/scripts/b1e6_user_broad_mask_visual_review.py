#!/usr/bin/env python3
"""
B1-E6: User-defined broad non-air mask visual review
목적: "ROI 내부 & 공기층 제외 & 병변 제외" broad surrogate mask를 CT slice에 overlay해서
      기존 oracle-like mask(HU>=0)와 비교 확인
마스크 정의:
  lesion 환자: refined_roi 내부 & HU > air_threshold & ~lesion_mask
  normal 환자: refined_roi 내부 & HU > air_threshold
이 마스크는 true vessel mask가 아니라 user-defined broad non-air surrogate mask이다.

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

ALLOW_REAL_PROCESSING = "--real" in sys.argv

# ── 경로 설정 ────────────────────────────────────────────────────────────────
PROJECT = Path("/home/jinhy/project/lung-ct-anomaly")
OUT_ROOT = (
    PROJECT
    / "outputs/position-aware-padim-v1"
    / "oracle_vessel_suppression_effb0_v4_20_dev_only"
    / "b1e6_user_broad_mask_visual_review_v1"
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

# ── air threshold 후보 ───────────────────────────────────────────────────────
# 프로젝트 기존 코드에서 -950 기준을 사용 중 (air_ratio_950)
# 3종 비교: -950, -900, -850
AIR_THRESHOLDS = [-950, -900, -850]

# ── 대상 환자 (B1-E5와 동일) ─────────────────────────────────────────────────
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
        "selection_reason": "B1-E5 동일 slice 재사용, lesion_risk_case",
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
        "selection_reason": "B1-E5 동일 slice 재사용, top eligible cluster",
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
        "selection_reason": "B1-E5 동일 slice 재사용",
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
        "selection_reason": "B1-E5 동일 slice 재사용, high oracle normal",
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
        "selection_reason": "B1-E5 동일 slice 재사용, low oracle normal",
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
    """oracle-like: HU >= 0 & ROI & ~lesion"""
    mask = roi_slice.astype(bool) & (ct_slice >= 0)
    if lesion_slice is not None:
        mask = mask & ~lesion_slice.astype(bool)
    return mask


def compute_broad_mask(ct_slice, roi_slice, air_threshold, lesion_slice=None):
    """user-defined broad non-air: HU > threshold & ROI & ~lesion"""
    mask = roi_slice.astype(bool) & (ct_slice > air_threshold)
    if lesion_slice is not None:
        mask = mask & ~lesion_slice.astype(bool)
    return mask


def draw_contours(ax, binary_slice, color, linewidth=0.9):
    contours = measure.find_contours(binary_slice.astype(np.float32), 0.5)
    for c in contours:
        ax.plot(c[:, 1], c[:, 0], color=color, linewidth=linewidth)


def draw_overlay(ax, mask, rgba):
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    for i, v in enumerate(rgba):
        overlay[mask, i] = v
    ax.imshow(overlay, origin="upper")


def draw_patches(ax, patches_list, edgecolor, linewidth, linestyle="solid", alpha=0.9):
    for p in patches_list:
        y0, x0 = int(p["y0"]), int(p["x0"])
        h = int(p["y1"]) - y0
        w = int(p["x1"]) - x0
        rect = mpatches.Rectangle(
            (x0, y0), w, h, linewidth=linewidth, edgecolor=edgecolor,
            facecolor="none", alpha=alpha, linestyle=linestyle,
        )
        ax.add_patch(rect)


def generate_png(patient, local_z, ct_vol, roi_vol, lesion_vol, all_score_rows,
                 patch_preview_dict, png_dir):
    """1 PNG = 3 subplot (threshold 별 비교)"""
    pid = patient["patient_id"]
    safe_id = patient["safe_id"]
    role = patient["role"]
    has_lesion = patient["has_lesion"]
    lesion_risk = patient["lesion_risk_case_flag"]

    ct_slice = np.array(ct_vol[local_z])
    roi_slice = np.array(roi_vol[local_z])
    lesion_slice = np.array(lesion_vol[local_z]) if has_lesion and lesion_vol is not None else None
    ct_disp = apply_lung_window(ct_slice)

    oracle_mask = compute_oracle_mask(ct_slice, roi_slice, lesion_slice)
    oracle_px = int(oracle_mask.sum())

    key = (pid, local_z)
    eligible_patches = [r for r in patch_preview_dict.get(key, []) if r["suppression_eligible"] == "True"]
    all_slice_patches = patch_preview_dict.get(key, [])

    score_rows_slice = [r for r in all_score_rows if int(r["local_z"]) == local_z]
    top_patch = max(score_rows_slice, key=lambda r: float(r["padim_score"])) if score_rows_slice else None
    top_score = float(top_patch["padim_score"]) if top_patch else float("nan")
    top_elig_score = (
        max(float(r["original_score"]) for r in eligible_patches)
        if eligible_patches else float("nan")
    )

    n_thresh = len(AIR_THRESHOLDS)
    fig, axes = plt.subplots(1, n_thresh, figsize=(10 * n_thresh, 10))
    if n_thresh == 1:
        axes = [axes]

    index_rows_for_slice = []

    for ax, thr in zip(axes, AIR_THRESHOLDS):
        broad_mask = compute_broad_mask(ct_slice, roi_slice, thr, lesion_slice)
        broad_px = int(broad_mask.sum())

        ax.imshow(ct_disp, cmap="gray", vmin=0, vmax=1, origin="upper")

        # ROI contour (초록)
        draw_contours(ax, roi_slice, color="limegreen", linewidth=0.8)

        # lesion contour (빨강)
        if has_lesion and lesion_slice is not None and lesion_slice.any():
            draw_contours(ax, lesion_slice, color="red", linewidth=1.2)

        # broad non-air mask (청록 반투명) — 먼저 깔기
        draw_overlay(ax, broad_mask, (0.0, 0.9, 0.9, 0.25))

        # oracle mask (노랑 반투명) — 위에 덧그리기
        draw_overlay(ax, oracle_mask, (1.0, 1.0, 0.0, 0.40))

        # eligible patch bbox (파랑)
        draw_patches(ax, eligible_patches, edgecolor="dodgerblue", linewidth=1.2)

        # top-score patch bbox (주황 점선)
        if top_patch:
            draw_patches(ax, [top_patch], edgecolor="darkorange", linewidth=1.8, linestyle="dashed")

        # 범례
        legend_elems = [
            mpatches.Patch(edgecolor="limegreen", facecolor="none", label="ROI contour"),
            mpatches.Patch(facecolor=(0.0, 0.9, 0.9, 0.3), label=f"Broad non-air (HU>{thr})"),
            mpatches.Patch(facecolor=(1, 1, 0, 0.5), label="Oracle-like (HU≥0)"),
            mpatches.Patch(edgecolor="dodgerblue", facecolor="none",
                           label=f"Eligible ({len(eligible_patches)})"),
            mpatches.Patch(edgecolor="darkorange", facecolor="none", linestyle="dashed",
                           label=f"Top-score ({top_score:.1f})"),
        ]
        if has_lesion:
            legend_elems.insert(1, mpatches.Patch(edgecolor="red", facecolor="none",
                                                   label="Lesion contour"))
        ax.legend(handles=legend_elems, loc="upper right", fontsize=7, framealpha=0.75)

        # subplot 제목
        short_pid = (pid[:22] + "...") if len(pid) > 22 else pid
        ax.set_title(
            f"{short_pid}  |  {role}  |  z={local_z}  |  HU>{thr}",
            fontsize=9, pad=4,
        )

        # annotation
        ts_str = f"{top_score:.3f}" if not np.isnan(top_score) else "N/A"
        te_str = f"{top_elig_score:.3f}" if not np.isnan(top_elig_score) else "N/A"
        ann = (
            f"patient_id: {pid}\n"
            f"safe_id: {safe_id}\n"
            f"role: {role}  |  z: {local_z}\n"
            f"air_threshold: HU>{thr}\n"
            f"broad_mask_px: {broad_px}\n"
            f"oracle_mask_px: {oracle_px}\n"
            f"eligible_patches: {len(eligible_patches)}\n"
            f"top_patch_score: {ts_str}\n"
            f"top_eligible_score: {te_str}\n"
            f"lesion_risk: {lesion_risk}"
        )
        ax.text(
            0.01, 0.01, ann, transform=ax.transAxes, fontsize=6,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.78),
        )
        ax.axis("off")

        index_rows_for_slice.append({
            "patient_id": pid,
            "safe_id": safe_id,
            "role": role,
            "local_z": local_z,
            "air_threshold": thr,
            "oracle_mask_pixel_count": oracle_px,
            "broad_mask_pixel_count": broad_px,
            "eligible_patch_count_on_slice": len(eligible_patches),
            "top_patch_score": round(top_score, 4) if not np.isnan(top_score) else None,
            "top_eligible_patch_score": round(top_elig_score, 4) if not np.isnan(top_elig_score) else None,
            "lesion_risk_case_flag": lesion_risk,
            "note": patient["selection_reason"],
            "png_path": "",  # 아래에서 채움
        })

    plt.suptitle(
        f"{pid[:40]}  |  {role}  |  z={local_z}  |  threshold compare: {AIR_THRESHOLDS}",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()

    pid_safe = pid.replace(".", "_")[:50]
    png_name = f"b1e6_{pid_safe}_z{local_z:04d}_thresh_compare.png"
    png_path = png_dir / png_name
    plt.savefig(png_path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    rel_path = str(png_path.relative_to(PROJECT))
    for row in index_rows_for_slice:
        row["png_path"] = rel_path

    return index_rows_for_slice


def main():
    is_dry = not ALLOW_REAL_PROCESSING
    mode_str = "DRY-RUN (plan only)" if is_dry else "REAL (PNG 생성)"
    print(f"[B1-E6] User-defined Broad Non-Air Mask Visual Review")
    print(f"  ALLOW_REAL_PROCESSING = {ALLOW_REAL_PROCESSING}")
    print(f"  mode = {mode_str}")

    # ── holdout denylist ──────────────────────────────────────────────────────
    holdout = load_stage2_holdout()
    holdout_intersection = [p["patient_id"] for p in PATIENTS if p["patient_id"] in holdout]
    if holdout_intersection:
        print(f"[ABORT] stage2_holdout 교집합: {holdout_intersection}", file=sys.stderr)
        sys.exit(1)
    print(f"  stage2_holdout 교집합: 0 (PASS)")

    # ── output root 존재 여부 ─────────────────────────────────────────────────
    if OUT_ROOT.exists():
        print(f"[ABORT] output root 이미 존재: {OUT_ROOT}", file=sys.stderr)
        sys.exit(1)

    # ── 계획 출력 ─────────────────────────────────────────────────────────────
    total_png = sum(len(p["target_slices"]) for p in PATIENTS)
    total_index_rows = total_png * len(AIR_THRESHOLDS)
    print(f"\n  air threshold 후보: {AIR_THRESHOLDS}")
    print(f"  기존 프로젝트 기준: -950 (air_ratio_950 변수로 사용 중)")
    print(f"  대상 환자 수: {len(PATIENTS)}")
    print(f"  총 예상 PNG 수: {total_png} (1 PNG = 3 threshold subplot 비교)")
    print(f"  총 index 행 수: {total_index_rows} (PNG당 3행)")
    print()
    for p in PATIENTS:
        print(f"  [{p['role']}] {p['patient_id']}")
        print(f"    slices: {p['target_slices']}")
        print(f"    reason: {p['selection_reason']}")
    print()

    if is_dry:
        print(
            "[DRY-RUN 완료] 실제 PNG 생성을 원하면:\n"
            "  source ~/ai_env/bin/activate && "
            "python scripts/b1e6_user_broad_mask_visual_review.py --real"
        )
        return

    # ── 실제 처리 ─────────────────────────────────────────────────────────────
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
        protected_paths += [
            p["ct_root"] / p["vol_dir"] / "ct_hu.npy",
            p["roi_base"] / p["safe_id"] / "refined_roi.npy",
            p["score_dir"] / f"{p['score_csv_stem']}.csv",
        ]
        if p["has_lesion"]:
            protected_paths.append(p["ct_root"] / p["vol_dir"] / "lesion_mask_roi_0_0.npy")
    pre_mtimes = {str(f): os.path.getmtime(f) for f in protected_paths if Path(f).exists()}

    index_rows = []
    errors = []

    for patient in PATIENTS:
        pid = patient["patient_id"]
        safe_id = patient["safe_id"]
        has_lesion = patient["has_lesion"]

        ct_path = patient["ct_root"] / patient["vol_dir"] / "ct_hu.npy"
        roi_path = patient["roi_base"] / safe_id / "refined_roi.npy"
        lesion_path = (
            patient["ct_root"] / patient["vol_dir"] / "lesion_mask_roi_0_0.npy"
            if has_lesion else None
        )
        score_csv_path = patient["score_dir"] / f"{patient['score_csv_stem']}.csv"

        try:
            ct_vol = np.load(ct_path, mmap_mode="r")
            roi_vol = np.load(roi_path, mmap_mode="r")
            lesion_vol = np.load(lesion_path, mmap_mode="r") if lesion_path else None

            with open(score_csv_path) as f:
                all_score_rows = list(csv.DictReader(f))

            for lz in patient["target_slices"]:
                try:
                    rows = generate_png(
                        patient, lz, ct_vol, roi_vol, lesion_vol,
                        all_score_rows, patch_preview_dict, png_dir,
                    )
                    index_rows.extend(rows)
                    broad_pxs = [r["broad_mask_pixel_count"] for r in rows]
                    print(
                        f"  [OK] {pid} z={lz} "
                        f"oracle_px={rows[0]['oracle_mask_pixel_count']} "
                        f"broad_px={broad_pxs} eligible={rows[0]['eligible_patch_count_on_slice']}"
                    )
                except Exception as e:
                    errors.append({"patient_id": pid, "local_z": lz, "error": str(e)})
                    print(f"  [ERR] {pid} z={lz}: {e}", file=sys.stderr)
        except Exception as e:
            for lz in patient["target_slices"]:
                errors.append({"patient_id": pid, "local_z": lz, "error": str(e)})
            print(f"  [ERR] {pid}: {e}", file=sys.stderr)

    # mtime 변조 체크
    mtime_violations = []
    for f_str, mt_before in pre_mtimes.items():
        if Path(f_str).exists() and abs(os.path.getmtime(f_str) - mt_before) > 1.0:
            mtime_violations.append(f_str)

    # index CSV
    index_csv_path = OUT_ROOT / "b1e6_user_broad_mask_visual_review_index.csv"
    with open(index_csv_path, "w", newline="") as f:
        if index_rows:
            w = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
            w.writeheader()
            w.writerows(index_rows)

    # errors CSV
    errors_csv_path = OUT_ROOT / "b1e6_user_broad_mask_visual_review_errors.csv"
    with open(errors_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "local_z", "error"])
        w.writeheader()
        w.writerows(errors)

    # summary JSON
    n_png = len({r["png_path"] for r in index_rows})
    all_checks_passed = (
        len(errors) == 0 and len(mtime_violations) == 0 and len(holdout_intersection) == 0
    )
    summary = {
        "step": "B1-E6",
        "created": "2026-06-05",
        "branch": "efficientnet_b0_imagenet_chestwall_removed_roi_v1",
        "roi_source": "refined_roi_v4_20_modeB_all_v1",
        "n_patients": len(PATIENTS),
        "n_png": n_png,
        "thresholds_tested": AIR_THRESHOLDS,
        "project_air_threshold_reference": "-950 (기존 코드 air_ratio_950 변수)",
        "patient_list": [p["patient_id"] for p in PATIENTS],
        "lesion_count": sum(1 for p in PATIENTS if p["has_lesion"]),
        "normal_count": sum(1 for p in PATIENTS if not p["has_lesion"]),
        "holdout_intersection": 0,
        "mtime_violations": len(mtime_violations),
        "mtime_violations_list": mtime_violations,
        "n_errors": len(errors),
        "source_files_modified": False,
        "score_modified": False,
        "suppression_applied": False,
        "gpu_used": False,
        "all_checks_passed": all_checks_passed,
    }
    summary_path = OUT_ROOT / "b1e6_user_broad_mask_visual_review_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # report.md
    report_path = OUT_ROOT / "b1e6_user_broad_mask_visual_review_report.md"
    with open(report_path, "w") as f:
        f.write("# B1-E6 User-defined Broad Non-Air Mask Visual Review Report\n\n")
        f.write(f"**생성일:** 2026-06-05  \n")
        f.write(f"**브랜치:** efficientnet_b0_imagenet_chestwall_removed_roi_v1  \n")
        f.write(f"**ROI:** refined_roi_v4_20_modeB_all_v1  \n\n")

        f.write("## 1. 목적\n\n")
        f.write(
            "이번 단계는 user-defined broad non-air mask visual review이다. "
            "성능 평가가 아니라 마스크가 실제로 어떤 구조를 포함하는지 육안으로 확인하는 것이 목적이다.\n\n"
        )

        f.write("## 2. 마스크 정의\n\n")
        f.write("이 마스크는 true vessel mask가 아니라 **user-defined broad non-air surrogate mask**이다.\n\n")
        f.write("- **lesion 환자:** `refined_roi 내부 & HU > threshold & ~lesion_mask`\n")
        f.write("- **normal 환자:** `refined_roi 내부 & HU > threshold`\n\n")
        f.write("**비교 대상 (oracle-like, B1-E5 기준):** `refined_roi 내부 & HU >= 0 & ~lesion_mask`\n\n")

        f.write("## 3. Air Threshold 비교\n\n")
        f.write("| threshold | 의미 |\n|---|---|\n")
        f.write("| HU > -950 | 기존 프로젝트 air_ratio_950 기준. 폐포 공기 대부분 제외. |\n")
        f.write("| HU > -900 | 중간 기준. 부분 공기 포함 가능. |\n")
        f.write("| HU > -850 | 가장 좁은 기준. 공기 경계부 일부 포함될 수 있음. |\n\n")

        f.write("## 4. 환자/슬라이스 선정\n\n")
        for p in PATIENTS:
            f.write(f"- **{p['patient_id']}** ({p['role']}): z={p['target_slices']} — {p['selection_reason']}\n")
        f.write("\n")

        f.write("## 5. 비교 포인트 (PNG 확인 후 기입)\n\n")
        obs_items = [
            "broad non-air mask가 실제로 혈관만 잡는지, 폐문/종격동/기관지벽/경계까지 포함되는지",
            "기존 oracle-like mask(HU>=0, 노랑)보다 broad mask(청록)가 얼마나 더 넓게 퍼지는지",
            "top-score patch(주황)가 broad mask 안에 대부분 포함되는지",
            "threshold -950/-900/-850 간 실제 차이가 시각적으로 어느 정도인지",
            "normal004(low oracle)에서 broad mask가 어떻게 분포하는지",
            "LUNG1-020 lesion risk case에서 broad mask와 lesion contour의 공간 관계",
        ]
        for item in obs_items:
            f.write(f"- {item}\n")
        f.write("\n(PNG 확인 후 기입)\n\n")

        f.write("## 6. 생성된 PNG 목록\n\n")
        seen_pngs = set()
        for row in index_rows:
            if row["png_path"] not in seen_pngs:
                seen_pngs.add(row["png_path"])
                f.write(f"- `{row['png_path']}`  z={row['local_z']}\n")
        f.write("\n")

        f.write("## 7. 안전 검증\n\n")
        f.write(f"- stage2_holdout 교집합: {len(holdout_intersection)}\n")
        f.write(f"- mtime 변조: {len(mtime_violations)}\n")
        f.write(f"- 오류: {len(errors)}\n")
        f.write(f"- score 수정: false\n")
        f.write(f"- suppression 적용: false\n")
        f.write(f"- GPU 사용: false\n")
        f.write(f"- all_checks_passed: {all_checks_passed}\n")

    # DONE
    if all_checks_passed:
        (OUT_ROOT / "DONE").touch()
        print(f"\n[PASS] 완료. DONE 생성. PNG {n_png}개.")
    else:
        print(
            f"\n[WARN] 오류 {len(errors)}개, mtime 위반 {len(mtime_violations)}개. "
            "DONE 생성 안 함.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
