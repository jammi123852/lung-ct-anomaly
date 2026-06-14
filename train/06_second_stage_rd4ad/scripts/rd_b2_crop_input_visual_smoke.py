#!/usr/bin/env python3
"""
RD-B2: 6-bin balanced normal crop input visual smoke test.

안전 조건:
- crop NPZ 저장 없음 / 학습 없음 / scoring 없음
- stage2_holdout 접근 없음 / GPU 사용 없음
- 기존 파일 수정/삭제 없음
- output root가 이미 있으면 즉시 중단

실행:
  python rd_b2_crop_input_visual_smoke.py --dry-run   # 샘플링 plan 확인
  python rd_b2_crop_input_visual_smoke.py --real      # 실제 PNG 생성
"""

ALLOW_REAL_PROCESSING = False  # --real 플래그로만 True로 변경

import argparse
import json
import sys
import time
import traceback
from collections import Counter
from itertools import groupby
from operator import itemgetter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import distance_transform_edt

# ══════════════════════════════════════════════════════════════════════════════
# 경로 상수
# ══════════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path("/home/jinhy/project/lung-ct-anomaly")

RD_B1_ROOT = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_b1_6bin_balanced_manifest_preflight_v1"
)
MANIFEST_CSV = RD_B1_ROOT / "rd_b1_6bin_balanced_normal_train_coordinate_manifest.csv"

META_ROOT = Path(
    "/mnt/c/Users/jinhy/Desktop"
    "/Normal_LUNA16_padim_training_ready_roi0_0_ts_lung_raw_no_dilate_v1"
    "/volumes_npy"
)

V4_20_ROI_ROOT = (
    PROJECT_ROOT
    / "outputs/mip-postprocess-research-v1/masks"
    / "refined_roi_v4_20_modeB_all_v1/normal"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/normal_based_stage2_verifier_audit"
    / "rd_b2_crop_input_visual_smoke_v1"
)

# ══════════════════════════════════════════════════════════════════════════════
# 파라미터
# ══════════════════════════════════════════════════════════════════════════════
SIX_BINS = [
    "upper_boundary", "upper_interior",
    "middle_boundary", "middle_interior",
    "lower_boundary", "lower_interior",
]
SAMPLES_PER_BIN = 6
CROP_SIZE = 96
Z_SPACING_MM = 1.0
MIP_SLAB_SLICES = 3        # 3mm / 1.0mm = 3 slices per slab
SAMPLING_SEED = 42
PADDING_POLICY = "edge_clamp"
EROSION_PX = 5             # boundary ring 시각화 두께

# 정규화 윈도우
LUNG_WIN_MIN, LUNG_WIN_MAX = -1350, 150
RD_CLIP_MIN, RD_CLIP_MAX   = -1000, 600

FORBIDDEN_PREFIXES = [
    str(
        (PROJECT_ROOT / "outputs/second-stage-lesion-refiner-v1/datasets").resolve()
    ),
]

# ══════════════════════════════════════════════════════════════════════════════
# 안전 함수
# ══════════════════════════════════════════════════════════════════════════════
def assert_safe_path(p: Path) -> None:
    resolved = str(Path(p).resolve())
    for fp in FORBIDDEN_PREFIXES:
        if resolved.startswith(fp):
            raise RuntimeError(f"FORBIDDEN path access: {p}")


def assert_not_exists(p: Path, label: str) -> None:
    if p.exists():
        print(f"[ABORT] {label} already exists: {p}", flush=True)
        print("[ABORT] output root가 이미 있습니다. 즉시 중단합니다.", flush=True)
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 데이터 로딩
# ══════════════════════════════════════════════════════════════════════════════
def load_ct(safe_id: str) -> np.ndarray:
    p = META_ROOT / safe_id / "ct_hu.npy"
    assert_safe_path(p)
    if not p.exists():
        raise FileNotFoundError(f"CT not found: {p}")
    return np.load(p, mmap_mode="r")


def load_roi(safe_id: str) -> np.ndarray:
    p = V4_20_ROI_ROOT / safe_id / "refined_roi.npy"
    assert_safe_path(p)
    if not p.exists():
        raise FileNotFoundError(f"ROI not found: {p}")
    return np.load(p, mmap_mode="r")


def load_meta(safe_id: str) -> dict:
    p = META_ROOT / safe_id / "meta.json"
    assert_safe_path(p)
    if not p.exists():
        raise FileNotFoundError(f"meta.json not found: {p}")
    with open(p) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# 유틸 – crop / MIP / normalization
# ══════════════════════════════════════════════════════════════════════════════
def clamp_z(idx: int, z_max: int) -> int:
    return int(max(0, min(idx, z_max - 1)))


def crop_slice(vol: np.ndarray, z_idx: int, y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
    """vol에서 z_idx 슬라이스의 crop 영역을 float32로 추출. edge clamp + 제로패딩."""
    Z, H, W = vol.shape
    iz = clamp_z(z_idx, Z)
    py0, py1 = max(0, y0), min(H, y1)
    px0, px1 = max(0, x0), min(W, x1)
    sub = vol[iz, py0:py1, px0:px1].astype(np.float32)
    ch, cw = y1 - y0, x1 - x0
    if sub.shape == (ch, cw):
        return sub
    out = np.zeros((ch, cw), dtype=np.float32)
    out[: sub.shape[0], : sub.shape[1]] = sub
    return out


def crop_full_slice(vol: np.ndarray, z_idx: int) -> np.ndarray:
    Z = vol.shape[0]
    return vol[clamp_z(z_idx, Z)].astype(np.float32)


def compute_mip_crop(vol: np.ndarray, center_z: int, offsets: list,
                     y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
    """crop 영역에 대해 주어진 offset들의 MIP 계산 (edge clamp)."""
    result: np.ndarray = None
    for o in offsets:
        sl = crop_slice(vol, center_z + o, y0, x0, y1, x1)
        result = sl if result is None else np.maximum(result, sl)
    return result


def norm_lung(arr: np.ndarray) -> np.ndarray:
    """old_AE_style lung window: HU [-1350, 150] → [0, 1]"""
    a = np.clip(arr, LUNG_WIN_MIN, LUNG_WIN_MAX).astype(np.float32)
    return (a - LUNG_WIN_MIN) / (LUNG_WIN_MAX - LUNG_WIN_MIN)


def norm_rd(arr: np.ndarray) -> np.ndarray:
    """new_RD_style: clip HU [-1000, 600] → (x+1000)/1600 → [0, 1]"""
    a = np.clip(arr, RD_CLIP_MIN, RD_CLIP_MAX).astype(np.float32)
    return (a - RD_CLIP_MIN) / (RD_CLIP_MAX - RD_CLIP_MIN)


# ══════════════════════════════════════════════════════════════════════════════
# 샘플링
# ══════════════════════════════════════════════════════════════════════════════
def sample_previews(manifest_df: pd.DataFrame, rng: np.random.Generator) -> list:
    """6-bin별 SAMPLES_PER_BIN개 샘플 선택."""
    selected = []

    for bin_label in SIX_BINS:
        bin_df = manifest_df[manifest_df["six_bin_label"] == bin_label].copy()
        if len(bin_df) == 0:
            print(f"[WARN] bin {bin_label}: 0 rows → skip", flush=True)
            continue

        is_boundary = "boundary" in bin_label

        if is_boundary:
            # boundary_overlap_ratio 삼분위 high/mid/low 각 2개
            sorted_df = bin_df.sort_values("boundary_overlap_ratio").reset_index(drop=True)
            n = len(sorted_df)
            t1, t2 = max(1, n // 3), max(2, 2 * n // 3)
            parts = [
                sorted_df.iloc[:t1],      # low
                sorted_df.iloc[t1:t2],    # mid
                sorted_df.iloc[t2:],      # high
            ]
            candidates = []
            for part in parts:
                if len(part) == 0:
                    continue
                uniq = part.drop_duplicates("patient_id")
                n_pick = min(2, len(uniq))
                picked = uniq.sample(n_pick, random_state=int(rng.integers(0, 2**31)))
                candidates.append(picked)

            result = pd.concat(candidates) if candidates else pd.DataFrame()
            # 부족하면 나머지에서 보충 (patient 다양성 유지)
            if len(result) < SAMPLES_PER_BIN:
                used_idx = set(result.index)
                remaining = bin_df[~bin_df.index.isin(used_idx)].drop_duplicates("patient_id")
                need = SAMPLES_PER_BIN - len(result)
                if len(remaining) > 0:
                    extra = remaining.sample(min(need, len(remaining)),
                                             random_state=int(rng.integers(0, 2**31)))
                    result = pd.concat([result, extra])

        else:
            # interior: refined_roi_ratio 높은 순, patient 다양성, z 위치 분산
            sorted_df = bin_df.sort_values("refined_roi_ratio", ascending=False)
            deduped = sorted_df.drop_duplicates("patient_id")
            n_target = SAMPLES_PER_BIN

            if len(deduped) >= n_target:
                # z_ratio 기준으로 고르게 spread
                sorted_by_z = deduped.sort_values("z_ratio")
                indices = np.linspace(0, len(sorted_by_z) - 1, n_target, dtype=int)
                result = sorted_by_z.iloc[indices]
            else:
                result = deduped

        result = result.head(SAMPLES_PER_BIN)
        for _, row in result.iterrows():
            selected.append(row.to_dict())

    return selected


# ══════════════════════════════════════════════════════════════════════════════
# PNG 생성 – 8-panel (5행 × 3열)
# ══════════════════════════════════════════════════════════════════════════════
def make_preview_png(
    preview_id: str,
    row: dict,
    ct: np.ndarray,
    roi: np.ndarray,
    png_path: Path,
    meta: dict,
) -> dict:
    """단일 crop에 대한 PNG 생성. NPZ 저장 없음."""
    lz = int(row["local_z"])
    y0, x0 = int(row["crop_y0"]), int(row["crop_x0"])
    y1, x1 = int(row["crop_y1"]), int(row["crop_x1"])

    roi_Z = roi.shape[0]
    roi_slice_idx = clamp_z(lz, roi_Z)

    # ── 전체 슬라이스 (Panel 1용)
    ct_full = crop_full_slice(ct, lz)
    roi_full = roi[roi_slice_idx].astype(np.uint8)

    # ── crop 추출
    ct_crop = crop_slice(ct, lz, y0, x0, y1, x1)
    roi_crop_raw = roi_full[max(0, y0):min(roi_full.shape[0], y1),
                             max(0, x0):min(roi_full.shape[1], x1)]
    if roi_crop_raw.shape != (y1 - y0, x1 - x0):
        tmp = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        tmp[:roi_crop_raw.shape[0], :roi_crop_raw.shape[1]] = roi_crop_raw
        roi_crop = tmp
    else:
        roi_crop = roi_crop_raw

    # ── MIP crops (3mm slab = 3 slices, z_spacing=1.0mm)
    lower_mip  = compute_mip_crop(ct, lz, [-3, -2, -1], y0, x0, y1, x1)
    center_mip = compute_mip_crop(ct, lz, [-1,  0,  1], y0, x0, y1, x1)
    upper_mip  = compute_mip_crop(ct, lz, [ 1,  2,  3], y0, x0, y1, x1)

    # ── 3ch 입력 후보
    baseline_3ch = [
        crop_slice(ct, lz - 1, y0, x0, y1, x1),  # z-1
        ct_crop,                                   # z
        crop_slice(ct, lz + 1, y0, x0, y1, x1),  # z+1
    ]
    mip_3ch   = [lower_mip, center_mip, upper_mip]
    mixed_3ch = [ct_crop, lower_mip, upper_mip]

    # ── 정규화 (center crop)
    old_norm = norm_lung(ct_crop)
    new_norm  = norm_rd(ct_crop)

    # ── ROI boundary ring (distance_transform_edt on full ROI slice)
    dist_full = distance_transform_edt(roi_full > 0)
    bnd_ring_full = ((roi_full > 0) & (dist_full <= EROSION_PX)).astype(np.float32)
    bnd_ring_raw  = bnd_ring_full[max(0, y0):min(bnd_ring_full.shape[0], y1),
                                   max(0, x0):min(bnd_ring_full.shape[1], x1)]
    if bnd_ring_raw.shape != (y1 - y0, x1 - x0):
        tmp2 = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
        tmp2[:bnd_ring_raw.shape[0], :bnd_ring_raw.shape[1]] = bnd_ring_raw
        bnd_ring_crop = tmp2
    else:
        bnd_ring_crop = bnd_ring_raw

    # ── 통계
    def arr_stats(arrs):
        combined = np.concatenate([a.ravel() for a in arrs])
        return float(combined.min()), float(combined.max())

    b_min, b_max = arr_stats(baseline_3ch)
    m_min, m_max = arr_stats(mip_3ch)
    x_min, x_max = arr_stats(mixed_3ch)
    o_min, o_max = float(old_norm.min()), float(old_norm.max())
    n_min, n_max = float(new_norm.min()), float(new_norm.max())

    # ── 레이아웃: subplot_mosaic 5행 × 3열
    mosaic = [
        ["full",  "raw",  "boundary"],
        ["b0",    "b1",   "b2"],
        ["m0",    "m1",   "m2"],
        ["x0",    "x1",   "x2"],
        ["old",   "new",  "txt"],
    ]
    fig, axes = plt.subplot_mosaic(
        mosaic, figsize=(18, 24), constrained_layout=True
    )

    z_sp = meta.get("spacing_xyz", [None, None, 1.0])[2]
    title = (
        f"{preview_id}  |  {row['patient_id']} ({row['safe_id']})\n"
        f"bin={row['six_bin_label']}  z={lz}  z_ratio={row['z_ratio']:.3f}  "
        f"bnd_ratio={row['boundary_overlap_ratio']:.3f}  roi_ratio={row['refined_roi_ratio']:.3f}"
    )
    fig.suptitle(title, fontsize=9)

    # ── Panel 0: 전체 슬라이스 + overlays
    ax = axes["full"]
    ax.imshow(norm_lung(ct_full), cmap="gray", origin="upper")
    ax.contour(roi_full, levels=[0.5], colors=["cyan"], linewidths=0.8)
    ax.contour(bnd_ring_full, levels=[0.5], colors=["yellow"], linewidths=0.5)
    rect = mpatches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        linewidth=2, edgecolor="orange", facecolor="none",
    )
    ax.add_patch(rect)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    ax.plot(cx, cy, "+", color="orange", markersize=8, markeredgewidth=1.5)
    ax.set_title("Full slice | ROI=cyan | bbox=orange | bnd_ring=yellow", fontsize=7)
    ax.axis("off")

    # ── Panel 1: crop raw HU
    ax = axes["raw"]
    ax.imshow(ct_crop, cmap="gray", vmin=-1000, vmax=300, origin="upper")
    ax.contour(roi_crop.astype(float), levels=[0.5], colors=["cyan"], linewidths=0.8)
    ax.set_title("Crop raw HU vmin=-1000 vmax=300", fontsize=7)
    ax.axis("off")

    # ── Panel 2: boundary ring overlay
    ax = axes["boundary"]
    ax.imshow(norm_lung(ct_crop), cmap="gray", origin="upper")
    rgba = np.zeros((*ct_crop.shape, 4), dtype=np.float32)
    rgba[..., 0] = bnd_ring_crop          # red channel
    rgba[..., 3] = bnd_ring_crop * 0.75   # alpha
    ax.imshow(rgba, origin="upper")
    ax.contour(roi_crop.astype(float), levels=[0.5], colors=["cyan"], linewidths=0.8)
    ax.set_title("Crop | boundary=red | ROI=cyan", fontsize=7)
    ax.axis("off")

    # ── Panel 3-5: baseline_2p5d_3ch
    b_labels = ["z-1", "z (center)", "z+1"]
    for key, ch, lbl in zip(["b0", "b1", "b2"], baseline_3ch, b_labels):
        ax = axes[key]
        ax.imshow(norm_lung(ch), cmap="gray", origin="upper")
        ax.set_title(f"baseline_2p5d | {lbl}", fontsize=7)
        ax.axis("off")
    axes["b0"].text(-0.12, 0.5, "baseline\n2p5d", transform=axes["b0"].transAxes,
                    fontsize=7, va="center", rotation=90)

    # ── Panel 6-8: mip_context_3ch
    m_labels = ["lower MIP (z-3~z-1)", "center MIP (z-1~z+1)", "upper MIP (z+1~z+3)"]
    for key, ch, lbl in zip(["m0", "m1", "m2"], mip_3ch, m_labels):
        ax = axes[key]
        ax.imshow(norm_lung(ch), cmap="gray", origin="upper")
        ax.set_title(f"mip_ctx | {lbl}", fontsize=7)
        ax.axis("off")
    axes["m0"].text(-0.12, 0.5, "mip_ctx\n3ch", transform=axes["m0"].transAxes,
                    fontsize=7, va="center", rotation=90)

    # ── Panel 9-11: mixed_3ch
    x_labels = ["CT center", "lower 3mm MIP", "upper 3mm MIP"]
    for key, ch, lbl in zip(["x0", "x1", "x2"], mixed_3ch, x_labels):
        ax = axes[key]
        ax.imshow(norm_lung(ch), cmap="gray", origin="upper")
        ax.set_title(f"mixed_3ch | {lbl}", fontsize=7)
        ax.axis("off")
    axes["x0"].text(-0.12, 0.5, "mixed\n3ch", transform=axes["x0"].transAxes,
                    fontsize=7, va="center", rotation=90)

    # ── Panel 12: old_AE_style
    ax = axes["old"]
    ax.imshow(old_norm, cmap="gray", vmin=0, vmax=1, origin="upper")
    ax.set_title(f"old_AE_style | HU[{LUNG_WIN_MIN},{LUNG_WIN_MAX}]→[0,1]", fontsize=7)
    ax.axis("off")

    # ── Panel 13: new_RD_style
    ax = axes["new"]
    ax.imshow(new_norm, cmap="gray", vmin=0, vmax=1, origin="upper")
    ax.set_title(f"new_RD_style | HU[{RD_CLIP_MIN},{RD_CLIP_MAX}]→[0,1]", fontsize=7)
    ax.axis("off")

    # ── Panel 14: stats text
    ax = axes["txt"]
    ax.axis("off")
    stats_text = (
        f"padding: {PADDING_POLICY}\n"
        f"z_spacing: {z_sp:.1f}mm  slab={MIP_SLAB_SLICES}slices\n\n"
        f"baseline 3ch HU:\n  [{b_min:.0f}, {b_max:.0f}]\n\n"
        f"mip_ctx 3ch HU:\n  [{m_min:.0f}, {m_max:.0f}]\n\n"
        f"mixed 3ch HU:\n  [{x_min:.0f}, {x_max:.0f}]\n\n"
        f"old_norm: [{o_min:.3f}, {o_max:.3f}]\n"
        f"new_norm: [{n_min:.3f}, {n_max:.3f}]\n\n"
        f"roi_bnd_dist_min:\n  {row['roi_boundary_distance_min']:.1f}px\n"
        f"refined_roi_ratio: {row['refined_roi_ratio']:.3f}\n"
        f"bnd_overlap: {row['boundary_overlap_ratio']:.3f}"
    )
    ax.text(
        0.05, 0.97, stats_text,
        transform=ax.transAxes, fontsize=7, va="top", ha="left",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return {
        "baseline_2p5d_min": b_min, "baseline_2p5d_max": b_max,
        "mip_context_min":   m_min, "mip_context_max":   m_max,
        "mixed_3ch_min":     x_min, "mixed_3ch_max":     x_max,
        "old_norm_min":      o_min, "old_norm_max":      o_max,
        "new_norm_min":      n_min, "new_norm_max":      n_max,
    }


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    global ALLOW_REAL_PROCESSING

    parser = argparse.ArgumentParser(description="RD-B2 crop input visual smoke")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="샘플링 plan 확인만")
    group.add_argument("--real",    action="store_true", help="실제 PNG 생성")
    args = parser.parse_args()

    if args.real:
        ALLOW_REAL_PROCESSING = True

    t0 = time.time()
    errors = []

    mode_label = "REAL" if ALLOW_REAL_PROCESSING else "DRY-RUN"
    print(f"[RD-B2] {mode_label} mode start", flush=True)
    print(f"[RD-B2] output root: {OUTPUT_ROOT}", flush=True)

    # ── 출력 root 존재 확인 (REAL 모드에서만 실제 체크)
    if ALLOW_REAL_PROCESSING:
        assert_not_exists(OUTPUT_ROOT, "OUTPUT_ROOT")

    # ── manifest 로드
    if not MANIFEST_CSV.exists():
        print(f"[ABORT] manifest not found: {MANIFEST_CSV}", flush=True)
        sys.exit(1)
    print(f"[RD-B2] Loading manifest: {MANIFEST_CSV}", flush=True)
    manifest_df = pd.read_csv(MANIFEST_CSV, encoding="utf-8-sig", low_memory=False)
    print(f"[RD-B2] Manifest: {len(manifest_df)} rows", flush=True)

    # RD-B1에서 stage2_holdout_intersection=0 확인됨
    stage2_holdout_intersection = 0

    # ── 샘플링
    rng = np.random.default_rng(SAMPLING_SEED)
    selected = sample_previews(manifest_df, rng)
    n_total = len(selected)

    bin_counts = Counter(r["six_bin_label"] for r in selected)
    print(f"[RD-B2] Selected {n_total} crops:", flush=True)
    for b in SIX_BINS:
        print(f"  {b:25s}: {bin_counts.get(b, 0)}", flush=True)

    unique_patients = len({r["safe_id"] for r in selected})
    print(f"[RD-B2] Unique patients: {unique_patients}", flush=True)

    # ── DRY-RUN 모드: plan 출력 후 종료
    if not ALLOW_REAL_PROCESSING:
        print(f"\n[DRY-RUN] Preview 목록 ({n_total}개):", flush=True)
        print(f"  {'#':>3}  {'bin':25s}  {'patient':12s}  {'z':>5}  {'bnd':>6}  {'roi':>6}", flush=True)
        for i, r in enumerate(selected):
            print(
                f"  [{i:02d}] {r['six_bin_label']:25s}  "
                f"{r['patient_id']:12s}  z={int(r['local_z']):4d}  "
                f"bnd={r['boundary_overlap_ratio']:.3f}  "
                f"roi={r['refined_roi_ratio']:.3f}",
                flush=True,
            )
        print(f"\n[DRY-RUN] 예상 PNG: {n_total}장", flush=True)
        print(f"[DRY-RUN] 예상 실행시간: ~{n_total * 3 // 60 + 1}–{n_total * 5 // 60 + 2}분 (CT 로딩 포함)", flush=True)
        print(f"[DRY-RUN] 입력 후보: baseline_2p5d_3ch / mip_context_3ch / mixed_3ch", flush=True)
        print(f"[DRY-RUN] 정규화 후보: old_AE_style / new_RD_style", flush=True)
        print(f"[DRY-RUN] crop NPZ 생성: 없음 / 학습: 없음 / scoring: 없음", flush=True)
        print(f"\n[DRY-RUN] 실제 실행: --real 플래그를 사용하세요.", flush=True)
        return

    # ══════════════════════════════════════════════════════════════════════════
    # REAL 모드
    # ══════════════════════════════════════════════════════════════════════════
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    (OUTPUT_ROOT / "pngs").mkdir()
    print(f"[RD-B2] Output root created: {OUTPUT_ROOT}", flush=True)

    index_rows = []
    n_ok = 0
    preview_counter = 0

    # patient별 그룹화 → CT/ROI를 환자당 1회만 로드
    selected_sorted = sorted(selected, key=itemgetter("safe_id"))
    grouped = {k: list(v) for k, v in groupby(selected_sorted, key=itemgetter("safe_id"))}

    for safe_id, rows_for_patient in grouped.items():
        print(f"\n[RD-B2] Patient {safe_id} ({len(rows_for_patient)} crops) ...", flush=True)
        try:
            ct   = load_ct(safe_id)
            roi  = load_roi(safe_id)
            meta = load_meta(safe_id)
        except Exception as e:
            err_msg = str(e)
            for row in rows_for_patient:
                pid_str = f"rd_b2_preview_{preview_counter:07d}"
                errors.append({
                    "preview_id": pid_str,
                    "safe_id": safe_id,
                    "stage": "load",
                    "error": err_msg,
                })
                print(f"  [ERROR] load {safe_id}: {err_msg}", flush=True)
                preview_counter += 1
            continue

        z_sp = meta.get("spacing_xyz", [None, None, 1.0])[2]

        for row in rows_for_patient:
            preview_id = f"rd_b2_preview_{preview_counter:07d}"
            png_path = OUTPUT_ROOT / "pngs" / f"{preview_id}.png"

            try:
                stats = make_preview_png(preview_id, row, ct, roi, png_path, meta)
                n_ok += 1
                print(f"  [{preview_counter:02d}] {preview_id} OK  bin={row['six_bin_label']:20s}  z={int(row['local_z']):4d}", flush=True)

                notes = []
                if row["boundary_overlap_ratio"] > 0.20:
                    notes.append("high_bnd_ratio")
                elif row["boundary_overlap_ratio"] < 0.05:
                    notes.append("low_bnd_ratio")
                if row["refined_roi_ratio"] > 0.95:
                    notes.append("deep_interior")
                visual_note = ";".join(notes) if notes else "normal"

                index_rows.append({
                    "preview_id":              preview_id,
                    "patient_id":              row["patient_id"],
                    "safe_id":                 row["safe_id"],
                    "six_bin_label":           row["six_bin_label"],
                    "z_level":                 row["z_level"],
                    "boundary_status":         row["boundary_status"],
                    "local_z":                 int(row["local_z"]),
                    "crop_y0":                 int(row["crop_y0"]),
                    "crop_x0":                 int(row["crop_x0"]),
                    "crop_y1":                 int(row["crop_y1"]),
                    "crop_x1":                 int(row["crop_x1"]),
                    "boundary_overlap_ratio":  row["boundary_overlap_ratio"],
                    "refined_roi_ratio":       row["refined_roi_ratio"],
                    "roi_boundary_distance_min": row["roi_boundary_distance_min"],
                    "z_spacing":               z_sp,
                    **stats,
                    "png_path":   str(png_path.relative_to(OUTPUT_ROOT)),
                    "visual_note": visual_note,
                })

            except Exception as e:
                errors.append({
                    "preview_id": preview_id,
                    "safe_id":    safe_id,
                    "stage":      "png",
                    "error":      str(e),
                    "tb":         traceback.format_exc(),
                })
                print(f"  [ERROR] {preview_id}: {e}", flush=True)

            preview_counter += 1

    elapsed = time.time() - t0

    # ── index CSV
    index_df = pd.DataFrame(index_rows)
    index_csv = OUTPUT_ROOT / "rd_b2_crop_visual_smoke_index.csv"
    index_df.to_csv(index_csv, index=False)
    print(f"\n[RD-B2] Index CSV: {index_csv}", flush=True)

    # ── errors CSV
    error_df = pd.DataFrame(errors)
    error_csv = OUTPUT_ROOT / "rd_b2_errors.csv"
    error_df.to_csv(error_csv, index=False)

    # ── summary JSON
    z_spacings_found = sorted({r["z_spacing"] for r in index_rows}) if index_rows else []
    summary = {
        "version": "rd_b2_v1",
        "n_preview_crops": n_ok,
        "n_bins": len(SIX_BINS),
        "samples_per_bin": SAMPLES_PER_BIN,
        "input_candidates_tested": [
            "baseline_2p5d_3ch",
            "mip_context_3ch",
            "mixed_3ch",
        ],
        "normalization_candidates_tested": ["old_AE_style", "new_RD_style"],
        "z_spacing_summary": z_spacings_found,
        "padding_policy": PADDING_POLICY,
        "mip_slab_definition": {
            "lower":  "z-3 ~ z-1 (edge_clamp)",
            "center": "z-1 ~ z+1 (edge_clamp)",
            "upper":  "z+1 ~ z+3 (edge_clamp)",
        },
        "crop_npz_generated": False,
        "training_started": False,
        "scoring_started": False,
        "stage2_holdout_intersection": stage2_holdout_intersection,
        "source_files_modified": False,
        "n_errors": len(errors),
        "elapsed_seconds": round(elapsed, 1),
        "all_checks_passed": len(errors) == 0,
        "absolute_not_done": [
            "crop NPZ 생성 없음",
            "학습 없음",
            "scoring 없음",
            "stage2_holdout 접근 없음",
            "기존 파일 수정 없음",
            "모델 forward 없음",
            "GPU 사용 없음",
            "threshold 재계산 없음",
            "score 재계산 없음",
        ],
    }
    summary_json = OUTPUT_ROOT / "rd_b2_crop_input_visual_smoke_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── report MD
    report_lines = [
        "# RD-B2 Crop Input Visual Smoke Report",
        "",
        "## 1. RD-B1 결과 요약",
        "- actual_selected_total: 86,017",
        "- 6-bin: upper/middle/lower × boundary/interior",
        "- erosion_px=5, boundary_threshold=0.05, cap=50/bin/patient",
        "- crop_size=96, n_normal_train_patients=290",
        "- full_cap_patient_count: 272, partial: 18, zero_bin: 0",
        "",
        "## 2. RD-B2 목적",
        "- RD-B1 manifest가 실제로 올바른 crop을 뽑는지 소수 샘플로 시각 검증",
        "- 입력 후보 3종 비교: baseline_2p5d_3ch / mip_context_3ch / mixed_3ch",
        "- 정규화 2종 비교: old_AE_style / new_RD_style",
        "- crop NPZ 생성 없음 / 학습 없음 / scoring 없음",
        "",
        f"## 3. 시각 검증 샘플",
        f"- 총 {n_ok}개 PNG 생성",
        f"- bin별 {SAMPLES_PER_BIN}개",
        f"- 고유 환자: {unique_patients}명",
        "",
        "## 4. 입력 후보 비교",
        "",
        "### A. baseline_2p5d_3ch (z-1, z, z+1)",
        "- 전후 슬라이스의 spatial context를 직접 제공",
        "- 혈관/기관지 구조가 z축 연속성으로 자연스럽게 표현",
        "- 단점: 고립 병변의 경우 z±1 슬라이스에 차이 없음",
        "",
        "### B. mip_context_3ch (lower/center/upper 3mm MIP)",
        "- 혈관/기관지를 더 강조 (MIP 특성상 밝은 구조 부각)",
        "- 폐 내부 fine structure가 선명하게 보일 수 있음",
        "- 단점: 혈관 과강조 가능성, 정상 기준이 달라질 수 있음",
        "",
        "### C. mixed_3ch (CT center, lower MIP, upper MIP)",
        "- center slice의 원본 정보를 보존하면서 MIP context 추가",
        "- 병변 HU 정보와 구조 context 동시 제공",
        "",
        "## 5. 정규화 비교",
        "",
        "### old_AE_style",
        f"- lung window HU [{LUNG_WIN_MIN}, {LUNG_WIN_MAX}] → [0, 1]",
        "- 폐 실질 및 기도 구조에 최적화",
        "",
        "### new_RD_style",
        f"- clip HU [{RD_CLIP_MIN}, {RD_CLIP_MAX}] → (x+{abs(RD_CLIP_MIN)})/{RD_CLIP_MAX - RD_CLIP_MIN} → [0, 1]",
        "- 더 넓은 HU 범위 커버 (흉벽·종격동 구조 포함 가능)",
        "",
        "## 6. 추천 입력 후보",
        "- [ ] baseline 유지",
        "- [ ] MIP-context 채택",
        "- [ ] mixed_3ch 채택",
        "- [ ] 추가 검토 필요",
        "- **→ 시각 검증 후 결정 필요**",
        "",
        "## 7. 추천 정규화",
        "- [ ] old_AE_style 유지",
        "- [ ] new_RD_style 채택",
        "- [ ] 추가 비교 필요",
        "- **→ 시각 검증 후 결정 필요**",
        "",
        "## 8. padding 정책",
        f"- padding_policy: **{PADDING_POLICY}**",
        f"- lower MIP slab: z-3 ~ z-1",
        f"- center MIP slab: z-1 ~ z+1",
        f"- upper MIP slab: z+1 ~ z+3",
        f"- z_spacing 기준: {Z_SPACING_MM}mm (3mm slab = 3 slices)",
        "",
        "## 9. 발견된 문제",
        "- [ ] crop 좌표 어긋남: 확인 필요",
        "- [ ] MIP 과강조: 확인 필요",
        "- [ ] boundary/interior 오분류: 확인 필요",
        "- [ ] normalization 문제: 확인 필요",
        "",
        "## 10. 다음 단계",
        "- **RD-B3**: true RD4AD teacher-student architecture preflight",
        "- 또는 **RD-B2b**: input design refinement (시각 검증 결과에 따라)",
        "",
        "## 11. 절대 하지 않은 것",
        "- crop NPZ 생성 없음",
        "- 학습 없음",
        "- scoring 없음",
        "- stage2_holdout 접근 없음",
        "- 기존 파일 수정 없음",
        "- 모델 forward 없음",
        "- GPU 사용 없음",
        "- threshold 재계산 없음",
        "- score 재계산 없음",
        "",
        "---",
        f"elapsed: {elapsed:.1f}s | n_ok: {n_ok} | n_errors: {len(errors)}",
    ]

    report_md = OUTPUT_ROOT / "rd_b2_crop_input_visual_smoke_report.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # ── DONE marker
    (OUTPUT_ROOT / "DONE").touch()

    print(f"\n[RD-B2] Done: {n_ok} PNGs, {len(errors)} errors, {elapsed:.1f}s", flush=True)
    print(f"[RD-B2] Output: {OUTPUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
