"""
RD-A2 Normal Train Crop Visual Verification v1

목적:
  RD-A1에서 분류된 normal train crop의 subtype별 시각 검증.
  실제 CT 슬라이스 위에 ROI 경계 + crop bbox를 오버레이하고,
  crop 중앙 슬라이스를 나란히 보여주는 그리드 PNG를 생성한다.

안전 조건:
  - 기존 파일 수정 금지
  - output root 이미 존재 시 즉시 중단
  - stage2_holdout 접근 금지
  - 모델 forward / GPU 사용 금지
  - score / metric 생성 금지

실행:
  python scripts/rd_a2_normal_train_crop_visual_verify.py --dry
  python scripts/rd_a2_normal_train_crop_visual_verify.py --real
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")

A1_OUTPUT = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_a1_normal_train_crop_subtype_audit_v1"
)
AUDIT_CSV = A1_OUTPUT / "rd_a1_normal_crop_subtype_audit.csv"

CROP_ROOT = (
    PROJECT_ROOT
    / "outputs/second-stage-lesion-refiner-v1/crops_normal"
    / "normal_rd4ad_2p5d_mw_fixed96_v1/crops/train"
)

CT_ROOT = Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1"
    "/volumes_npy"
)

ROI_ROOT = (
    PROJECT_ROOT
    / "outputs/mip-postprocess-research-v1/masks"
    / "refined_roi_v4_20_modeB_all_v1/normal"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_a2_normal_train_crop_visual_verify_v1"
)

SUBTYPES = [
    "normal_boundary",
    "normal_hilar_or_central",
    "normal_vessel",
    "normal_vessel_boundary_mixed",
]

SAMPLES_PER_SUBTYPE = 16  # 환자 분산 샘플링

# crop npz image: (6, 96, 96) — 중앙 슬라이스 인덱스
CROP_CENTER_SLICE = 2  # 0-indexed: 5슬라이스 중 3번째

# CT HU window
HU_MIN, HU_MAX = -1000, 400


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _parse_crop_id(crop_id: str):
    """crop_id → (patient_dir, slice_index, y0, x0)
    형식: {patient_dir}__{local_z}_{y0}_{x0}  ← 언더스코어 2개로 patient_dir 분리
    예: normal001__104e7cb873_11_336_144
    """
    parts = crop_id.rsplit("_", 3)
    # parts[-3]=local_z, parts[-2]=y0, parts[-1]=x0
    # parts[0] = patient_dir (double-underscore 포함)
    patient_dir = parts[0]
    local_z = int(parts[1])
    y0 = int(parts[2])
    x0 = int(parts[3])
    return patient_dir, local_z, y0, x0


def _load_ct_slice(patient_dir: str, z_idx: int):
    """CT 볼륨에서 z_idx 슬라이스 반환 (HU int16)."""
    ct_path = CT_ROOT / patient_dir / "ct_hu.npy"
    if not ct_path.exists():
        return None
    vol = np.load(ct_path, mmap_mode="r")
    return vol[z_idx].astype(np.float32)


def _load_roi_slice(patient_dir: str, z_idx: int):
    """ROI 마스크에서 z_idx 슬라이스 반환."""
    roi_path = ROI_ROOT / patient_dir / "refined_roi.npy"
    if not roi_path.exists():
        return None
    roi = np.load(roi_path, mmap_mode="r")
    return roi[z_idx].astype(np.uint8)


def _load_crop_center(crop_id: str, patient_dir: str):
    """crop npz에서 중앙 슬라이스 반환."""
    npz_path = CROP_ROOT / patient_dir / f"{crop_id}.npz"
    if not npz_path.exists():
        return None
    f = np.load(npz_path)
    img = f["image"]  # (6, 96, 96) float32 [0,1]
    return img[CROP_CENTER_SLICE]


def _ct_to_uint8(ct_slice: np.ndarray) -> np.ndarray:
    """HU → [0,255] uint8."""
    clipped = np.clip(ct_slice, HU_MIN, HU_MAX)
    normed = (clipped - HU_MIN) / (HU_MAX - HU_MIN)
    return (normed * 255).astype(np.uint8)


def _roi_contour(roi_slice: np.ndarray) -> np.ndarray:
    """ROI 경계 픽셀 마스크 반환 (간단한 erosion 차이)."""
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(roi_slice > 0)
    return (roi_slice > 0) & (~eroded)


def _patient_balanced_sample(df_sub: pd.DataFrame, n: int) -> pd.DataFrame:
    """환자 분산 라운드로빈 샘플링."""
    patients = df_sub["patient_id"].unique().tolist()
    sampled = []
    patient_rows = {p: df_sub[df_sub["patient_id"] == p].sample(frac=1, random_state=42) for p in patients}
    iters = {p: iter(patient_rows[p].itertuples()) for p in patients}
    active = list(patients)
    while len(sampled) < n and active:
        for p in list(active):
            try:
                row = next(iters[p])
                sampled.append(row)
                if len(sampled) >= n:
                    break
            except StopIteration:
                active.remove(p)
    return sampled  # list of namedtuples


# ─────────────────────────────────────────────────────────
# Grid rendering
# ─────────────────────────────────────────────────────────

SUBTYPE_COLORS = {
    "normal_boundary": "#FF6B35",
    "normal_hilar_or_central": "#4ECDC4",
    "normal_vessel": "#45B7D1",
    "normal_vessel_boundary_mixed": "#96CEB4",
}


def _render_subtype_grid(rows, subtype: str, out_path: Path):
    """rows(list of namedtuples) → 그리드 PNG."""
    n = len(rows)
    if n == 0:
        return

    ncols = 4  # 4쌍 (CT overlay + crop) = 8 axes per row
    # 각 샘플 = 2 패널 (CT overlay | crop center)
    cols_per_sample = 2
    n_cols_total = ncols * cols_per_sample
    n_rows_total = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        n_rows_total, n_cols_total,
        figsize=(n_cols_total * 2.2, n_rows_total * 2.5),
        squeeze=False,
    )
    fig.suptitle(
        f"RD-A2 Visual Verify — {subtype}\n(n={n}, center_slice={CROP_CENTER_SLICE})",
        fontsize=11, y=1.01,
    )

    color = SUBTYPE_COLORS.get(subtype, "#888888")

    for idx, row in enumerate(rows):
        grid_r = idx // ncols
        grid_c = idx % ncols
        ax_ct = axes[grid_r][grid_c * 2]
        ax_crop = axes[grid_r][grid_c * 2 + 1]

        crop_id = row.crop_id
        patient_dir, local_z, y0, x0 = _parse_crop_id(crop_id)
        crop_size = 96

        # ── CT overlay ──
        ct_slice = _load_ct_slice(patient_dir, local_z)
        roi_slice = _load_roi_slice(patient_dir, local_z)

        if ct_slice is not None:
            ct_u8 = _ct_to_uint8(ct_slice)
            ax_ct.imshow(ct_u8, cmap="gray", interpolation="nearest")

            if roi_slice is not None:
                contour = _roi_contour(roi_slice)
                ax_ct.contour(roi_slice > 0, levels=[0.5], colors=["cyan"], linewidths=0.5, alpha=0.7)

            # crop bbox
            rect = mpatches.Rectangle(
                (x0, y0), crop_size, crop_size,
                linewidth=1.5, edgecolor=color, facecolor="none",
            )
            ax_ct.add_patch(rect)
            ax_ct.set_xlim(max(0, x0 - 20), min(ct_slice.shape[1], x0 + crop_size + 20))
            ax_ct.set_ylim(min(ct_slice.shape[0], y0 + crop_size + 20), max(0, y0 - 20))
        else:
            ax_ct.text(0.5, 0.5, "CT missing", ha="center", va="center", transform=ax_ct.transAxes, fontsize=7)

        ax_ct.set_title(f"{patient_dir.split('__')[0]}\nz={local_z}", fontsize=6, pad=2)
        ax_ct.axis("off")

        # ── Crop center ──
        crop_img = _load_crop_center(crop_id, patient_dir)
        if crop_img is not None:
            ax_crop.imshow(crop_img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax_crop.set_title(
                f"{row.normal_subtype_label.replace('normal_','')}\nbin={row.position_bin}",
                fontsize=6, pad=2, color=color,
            )
        else:
            ax_crop.text(0.5, 0.5, "npz missing", ha="center", va="center", transform=ax_crop.transAxes, fontsize=7)
        ax_crop.axis("off")

    # 빈 셀 숨기기
    for idx in range(n, n_rows_total * ncols):
        grid_r = idx // ncols
        grid_c = idx % ncols
        axes[grid_r][grid_c * 2].axis("off")
        axes[grid_r][grid_c * 2 + 1].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PNG] {out_path.name}")


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true", help="dry run: 후보 확인만")
    parser.add_argument("--real", action="store_true", help="실제 시각화 실행")
    args = parser.parse_args()

    if not args.dry and not args.real:
        print("[ERROR] --dry 또는 --real 중 하나를 지정하세요.")
        sys.exit(1)

    # ── 전제 조건 확인 ──
    if not AUDIT_CSV.exists():
        print(f"[ERROR] A1 audit CSV 없음: {AUDIT_CSV}")
        sys.exit(1)

    if not A1_OUTPUT.joinpath("DONE").exists():
        print(f"[ERROR] A1 DONE marker 없음. RD-A1을 먼저 완료하세요.")
        sys.exit(1)

    df = pd.read_csv(AUDIT_CSV)
    df_train = df[df["split"] == "train"].copy()
    print(f"[INFO] train crops: {len(df_train)}")

    for st in SUBTYPES:
        sub = df_train[df_train["normal_subtype_label"] == st]
        print(f"  {st}: {len(sub)}")

    if args.dry:
        print("\n[DRY] 예상 출력:")
        for st in SUBTYPES:
            sub = df_train[df_train["normal_subtype_label"] == st]
            n = min(SAMPLES_PER_SUBTYPE, len(sub))
            print(f"  rd_a2_{st}_grid.png  ({n} samples)")
        print(f"\n  output: {OUTPUT_ROOT}")
        print("[DRY] 실제 실행하려면 --real 사용")
        return

    # ── real 실행 ──
    if OUTPUT_ROOT.exists():
        print(f"[ERROR] output root 이미 존재: {OUTPUT_ROOT}")
        print("  삭제 후 재실행하세요.")
        sys.exit(1)

    OUTPUT_ROOT.mkdir(parents=True)
    print(f"[INFO] output root 생성: {OUTPUT_ROOT}")

    t_start = time.time()

    for st in SUBTYPES:
        sub = df_train[df_train["normal_subtype_label"] == st]
        if len(sub) == 0:
            print(f"[SKIP] {st}: 샘플 없음")
            continue

        rows = _patient_balanced_sample(sub, SAMPLES_PER_SUBTYPE)
        out_png = OUTPUT_ROOT / f"rd_a2_{st}_grid.png"
        print(f"[RENDER] {st} ({len(rows)} samples)...")
        _render_subtype_grid(rows, st, out_png)

    elapsed = time.time() - t_start
    print(f"\n[COMPLETE] elapsed: {elapsed:.1f}s")
    print(f"  output: {OUTPUT_ROOT}")

    # DONE marker
    (OUTPUT_ROOT / "DONE").write_text("RD-A2 visual verify complete\n")
    print("[DONE] DONE marker")


if __name__ == "__main__":
    main()
