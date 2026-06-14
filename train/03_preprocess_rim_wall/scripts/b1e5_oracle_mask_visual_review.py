#!/usr/bin/env python3
"""
B1-E5: Oracle-like vessel mask visual review
목적: HU>=0 oracle-like vessel mask, suppression eligible patch, top-score patch를
      CT slice 위에 overlay해서 육안으로 검증
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
    / "b1e5_oracle_mask_visual_review_v1"
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

# ── 대상 환자 설정 (B1-E1 targets.csv + B1-E3 patch_preview 기반) ────────────
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
    """(patient_id, local_z) → list of row dicts"""
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
    oracle = in_roi & bright
    if lesion_slice is not None:
        oracle = oracle & ~lesion_slice.astype(bool)
    return oracle


def draw_contours(ax, binary_slice, color, linewidth=0.9):
    contours = measure.find_contours(binary_slice.astype(np.float32), 0.5)
    for c in contours:
        ax.plot(c[:, 1], c[:, 0], color=color, linewidth=linewidth)


def generate_png(patient, local_z, ct_vol, roi_vol, lesion_vol, all_score_rows,
                 patch_preview_dict, png_dir):
    pid = patient["patient_id"]
    safe_id = patient["safe_id"]
    role = patient["role"]
    has_lesion = patient["has_lesion"]
    lesion_risk = patient["lesion_risk_case_flag"]

    ct_slice = np.array(ct_vol[local_z])
    roi_slice = np.array(roi_vol[local_z])
    lesion_slice = np.array(lesion_vol[local_z]) if has_lesion and lesion_vol is not None else None

    oracle_mask = compute_oracle_mask(ct_slice, roi_slice, lesion_slice)
    oracle_pixel_count = int(oracle_mask.sum())

    ct_disp = apply_lung_window(ct_slice)

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
    top_eligible_score = (
        max(float(r["original_score"]) for r in eligible_patches)
        if eligible_patches else float("nan")
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(ct_disp, cmap="gray", vmin=0, vmax=1, origin="upper")

    # ROI contour (초록)
    draw_contours(ax, roi_slice, color="limegreen", linewidth=0.8)

    # lesion contour (빨강)
    if has_lesion and lesion_slice is not None and lesion_slice.any():
        draw_contours(ax, lesion_slice, color="red", linewidth=1.2)

    # oracle mask overlay (노랑 반투명)
    overlay = np.zeros((*oracle_mask.shape, 4), dtype=np.float32)
    overlay[oracle_mask, 0] = 1.0
    overlay[oracle_mask, 1] = 1.0
    overlay[oracle_mask, 2] = 0.0
    overlay[oracle_mask, 3] = 0.35
    ax.imshow(overlay, origin="upper")

    # suppression eligible patch bbox (파랑)
    for p in eligible_patches:
        y0, x0 = int(p["y0"]), int(p["x0"])
        h = int(p["y1"]) - y0
        w = int(p["x1"]) - x0
        rect = mpatches.Rectangle(
            (x0, y0), w, h, linewidth=1.2, edgecolor="dodgerblue",
            facecolor="none", alpha=0.9
        )
        ax.add_patch(rect)

    # top-score patch bbox (주황 점선)
    if top_patch is not None:
        ty0, tx0 = int(top_patch["y0"]), int(top_patch["x0"])
        th = int(top_patch["y1"]) - ty0
        tw = int(top_patch["x1"]) - tx0
        rect = mpatches.Rectangle(
            (tx0, ty0), tw, th, linewidth=1.8, edgecolor="darkorange",
            facecolor="none", alpha=0.9, linestyle="--"
        )
        ax.add_patch(rect)

    # 범례
    legend_elems = [
        mpatches.Patch(edgecolor="limegreen", facecolor="none", label="ROI contour"),
        mpatches.Patch(facecolor=(1, 1, 0, 0.4), label="Oracle-like vessel mask (HU≥0)"),
        mpatches.Patch(edgecolor="dodgerblue", facecolor="none",
                       label=f"Suppression eligible ({eligible_count})"),
        mpatches.Patch(edgecolor="darkorange", facecolor="none", linestyle="dashed",
                       label=f"Top-score patch (score={top_score:.2f})"),
    ]
    if has_lesion:
        legend_elems.insert(1, mpatches.Patch(edgecolor="red", facecolor="none",
                                               label="Lesion contour"))
    ax.legend(handles=legend_elems, loc="upper right", fontsize=7.5, framealpha=0.75)

    # 제목
    short_pid = (pid[:28] + "...") if len(pid) > 28 else pid
    ax.set_title(f"{short_pid}  |  {role}  |  z={local_z}", fontsize=10, pad=6)

    # 하단 annotation
    top_elig_str = f"{top_eligible_score:.3f}" if not np.isnan(top_eligible_score) else "N/A"
    top_score_str = f"{top_score:.3f}" if not np.isnan(top_score) else "N/A"
    ann = (
        f"patient_id: {pid}\n"
        f"safe_id: {safe_id}\n"
        f"role: {role}  |  local_z: {local_z}\n"
        f"oracle_pixels_on_slice: {oracle_pixel_count}\n"
        f"eligible_patches_on_slice: {eligible_count}\n"
        f"top_patch_score: {top_score_str}\n"
        f"top_eligible_patch_score: {top_elig_str}\n"
        f"lesion_risk_case: {lesion_risk}"
    )
    ax.text(
        0.01, 0.01, ann, transform=ax.transAxes, fontsize=6.5,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.78),
    )
    ax.axis("off")
    plt.tight_layout()

    pid_safe = pid.replace(".", "_")[:50]
    png_name = f"b1e5_{pid_safe}_z{local_z:04d}.png"
    png_path = png_dir / png_name
    plt.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return {
        "patient_id": pid,
        "safe_id": safe_id,
        "role": role,
        "local_z": local_z,
        "png_path": str(png_path.relative_to(PROJECT)),
        "oracle_pixel_count_on_slice": oracle_pixel_count,
        "eligible_patch_count_on_slice": eligible_count,
        "top_patch_score": (
            round(top_score, 4) if not np.isnan(top_score) else None
        ),
        "top_eligible_patch_score": (
            round(top_eligible_score, 4) if not np.isnan(top_eligible_score) else None
        ),
        "lesion_risk_case_flag": lesion_risk,
        "note": patient["selection_reason"],
    }


def main():
    is_dry = not ALLOW_REAL_PROCESSING
    mode_str = "DRY-RUN (plan only)" if is_dry else "REAL (PNG 생성)"
    print(f"[B1-E5] Oracle Mask Visual Review")
    print(f"  ALLOW_REAL_PROCESSING = {ALLOW_REAL_PROCESSING}")
    print(f"  mode = {mode_str}")

    # ── holdout denylist 확인 ──────────────────────────────────────────────
    holdout = load_stage2_holdout()
    holdout_intersection = [p["patient_id"] for p in PATIENTS if p["patient_id"] in holdout]
    if holdout_intersection:
        print(f"[ABORT] stage2_holdout 교집합 발견: {holdout_intersection}", file=sys.stderr)
        sys.exit(1)
    print(f"  stage2_holdout 교집합: 0 (PASS)")

    # ── output root 존재 여부 ──────────────────────────────────────────────
    if OUT_ROOT.exists():
        print(f"[ABORT] output root 이미 존재: {OUT_ROOT}", file=sys.stderr)
        sys.exit(1)

    # ── 계획 출력 ──────────────────────────────────────────────────────────
    total_slices = sum(len(p["target_slices"]) for p in PATIENTS)
    print(f"\n  대상 환자 수: {len(PATIENTS)} (lesion 3, normal 2)")
    print(f"  총 예상 PNG 수: {total_slices}")
    print()
    for p in PATIENTS:
        print(f"  [{p['role']}] {p['patient_id']}")
        print(f"    slices: {p['target_slices']}")
        print(f"    reason: {p['selection_reason']}")
        print()

    if is_dry:
        print(
            "[DRY-RUN 완료] 실제 PNG 생성을 원하면:\n"
            "  source ~/ai_env/bin/activate && python scripts/b1e5_oracle_mask_visual_review.py --real"
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
        str(f): os.path.getmtime(f) for f in protected_paths if Path(f).exists()
    }

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

            all_score_rows = []
            with open(score_csv_path) as f:
                all_score_rows = list(csv.DictReader(f))

            for lz in patient["target_slices"]:
                try:
                    row = generate_png(
                        patient, lz, ct_vol, roi_vol, lesion_vol,
                        all_score_rows, patch_preview_dict, png_dir,
                    )
                    index_rows.append(row)
                    print(
                        f"  [OK] {pid} z={lz} "
                        f"oracle_px={row['oracle_pixel_count_on_slice']} "
                        f"eligible={row['eligible_patch_count_on_slice']}"
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
        if Path(f_str).exists():
            mt_after = os.path.getmtime(f_str)
            if abs(mt_after - mt_before) > 1.0:
                mtime_violations.append(f_str)

    # index CSV
    index_csv_path = OUT_ROOT / "b1e5_oracle_mask_visual_review_index.csv"
    with open(index_csv_path, "w", newline="") as f:
        if index_rows:
            w = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
            w.writeheader()
            w.writerows(index_rows)

    # errors CSV
    errors_csv_path = OUT_ROOT / "b1e5_oracle_mask_visual_review_errors.csv"
    with open(errors_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "local_z", "error"])
        w.writeheader()
        w.writerows(errors)

    # summary JSON
    all_checks_passed = (
        len(errors) == 0
        and len(mtime_violations) == 0
        and len(holdout_intersection) == 0
    )
    summary = {
        "step": "B1-E5",
        "created": "2026-06-05",
        "branch": "efficientnet_b0_imagenet_chestwall_removed_roi_v1",
        "roi_source": "refined_roi_v4_20_modeB_all_v1",
        "n_patients": len(PATIENTS),
        "n_png": len(index_rows),
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
    summary_path = OUT_ROOT / "b1e5_oracle_mask_visual_review_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # report.md
    report_path = OUT_ROOT / "b1e5_oracle_mask_visual_review_report.md"
    with open(report_path, "w") as f:
        f.write("# B1-E5 Oracle Mask Visual Review Report\n\n")
        f.write(f"**생성일:** 2026-06-05  \n")
        f.write(f"**브랜치:** efficientnet_b0_imagenet_chestwall_removed_roi_v1  \n")
        f.write(f"**ROI:** refined_roi_v4_20_modeB_all_v1  \n\n")

        f.write("## 1. 목적\n\n")
        f.write(
            "이번 단계는 visual review 전용이며 suppression 재실행이 아니다. "
            "B1-E1~B1-E3에서 사용한 oracle-like vessel mask가 실제로 어디를 잡고 있는지 "
            "CT slice 위에 overlay해서 육안으로 확인하기 위한 단계이다.\n\n"
        )

        f.write("## 2. Oracle-like Vessel Mask 정의\n\n")
        f.write("- **lesion 환자:** refined ROI 내부에서 HU >= 0 이고 lesion mask가 아닌 voxel\n")
        f.write("- **normal 환자:** refined ROI 내부에서 HU >= 0 voxel\n")
        f.write(
            "- **주의:** true vessel GT가 아니라 oracle-like bright vessel candidate mask이다. "
            "석회화, 종격동, partial-volume 등 HU >= 0인 비혈관 구조도 포함될 수 있다.\n\n"
        )

        f.write("## 3. 환자/슬라이스 선정 기준\n\n")
        for p in PATIENTS:
            f.write(f"- **{p['patient_id']}** ({p['role']}): z={p['target_slices']} — {p['selection_reason']}\n")
        f.write("\n")

        f.write("## 4. 생성된 PNG 목록\n\n")
        for row in index_rows:
            ts = row["top_patch_score"]
            te = row["top_eligible_patch_score"]
            f.write(
                f"- `{row['png_path']}` "
                f"z={row['local_z']} oracle_px={row['oracle_pixel_count_on_slice']} "
                f"eligible={row['eligible_patch_count_on_slice']} "
                f"top_score={ts} top_elig={te}\n"
            )
        f.write("\n")

        f.write("## 5. 시각적 관찰 항목 (확인 후 기입)\n\n")
        obs_items = [
            "oracle mask가 주로 혈관을 잡는지, 폐문/종격동 구조가 더 큰지",
            "경계/partial-volume 아티팩트 비율",
            "suppression eligible patch가 high-score 위치와 실제로 얼마나 겹치는지",
            "LUNG1-020 lesion risk case에서 oracle mask와 lesion contour의 공간 관계",
            "B1-E3에서 효과가 약했던 원인에 대한 시각적 힌트",
            "normal(subset1 vs normal004)에서 oracle mask 분포 차이",
        ]
        for item in obs_items:
            f.write(f"- {item}\n")
        f.write("\n(PNG 확인 후 기입)\n\n")

        f.write("## 6. LUNG1-020 별도 해석\n\n")
        f.write(
            "LUNG1-020은 B1-E3에서 lesion_risk_case_flag=True로 분류된 케이스이다. "
            "oracle mask가 lesion 근처를 잡고 있는지, "
            "suppression eligible patch와 lesion contour가 얼마나 겹치는지 확인이 필요하다.\n\n"
        )

        f.write("## 7. 1차 결론 (확인 후 기입)\n\n")
        f.write(
            "- 실제로 혈관 위주인지\n"
            "- 경계/종격동 구조가 더 큰지\n"
            "- B1-E3 효과 약함의 시각적 원인\n\n"
            "(PNG 확인 후 기입)\n\n"
        )

        f.write("## 8. 안전 검증\n\n")
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
        print(f"\n[PASS] 완료. DONE 생성. PNG {len(index_rows)}개.")
    else:
        print(
            f"\n[WARN] 오류 {len(errors)}개, mtime 위반 {len(mtime_violations)}개. "
            "DONE 생성 안 함.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
