"""
p_c_normal24h_audit_crop_lung_roi_ratio.py

P-C-NORMAL24h: crop_lung_roi_ratio 분모 mismatch audit
- 목적: 24g에서 분모를 항상 96×96(=9216)으로 고정했는데,
        bbox가 32×32인 경우 roi_sum 최대값은 1024 → ratio 최대 0.1111로 제한됨
- 검증: CT crop + ROI overlay, current vs corrected ratio 비교
- 금지: 기존 24g/24h 결과 수정 금지, model training/forward/metrics/threshold 금지,
        final_test 사용 금지, vessel feature 사용 금지

판정 기준:
  PASS_FIX_NEEDED : corrected ratio가 시각적으로 맞고, current가 1/9로 축소 확인
  PASS_NO_FIX     : current ratio가 시각적으로 맞으면 24g 유지
  FAIL            : 좌표계 불일치, bbox/ROI alignment 미확인
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 경로 ──────────────────────────────────────────────────────────────────────
BRANCH_ROOT  = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BRANCH_ROOT.parents[1]

MANIFEST_DIR = PROJECT_ROOT / "outputs/manifests/p_c_normal24g_zroi_only_feature_manifest"
ROI_DIR      = PROJECT_ROOT / "outputs/mip-postprocess-research-v1/masks/refined_roi_v4_20_modeB_all_v1"
REPORT_OUT   = PROJECT_ROOT / "outputs/reports/p_c_normal24h_audit_crop_lung_roi_ratio"

CROP_SIZE = 96
N_SAMPLE  = 20  # low/mid/high 각 20개

# ── ROI map 빌드 ──────────────────────────────────────────────────────────────
def build_roi_map() -> dict:
    roi_map = {}
    for grp in ["normal", "lesion"]:
        grp_dir = ROI_DIR / grp
        if not grp_dir.exists():
            continue
        for sid in grp_dir.iterdir():
            p = sid / "refined_roi.npy"
            if p.exists():
                roi_map[sid.name] = str(p)
    return roi_map


def load_roi(safe_id: str, roi_map: dict):
    if safe_id not in roi_map:
        return None
    return np.load(roi_map[safe_id])


# ── ROI crop 합계 계산 ────────────────────────────────────────────────────────
def get_roi_sum_and_crop(roi, z: int, y0: int, x0: int, y1: int, x1: int):
    """roi[z, y0:y1, x0:x1] 합계와 crop 배열 반환 (boundary clip 포함)"""
    nz, ny, nx = roi.shape
    if z < 0 or z >= nz:
        return None, None
    y0c = max(0, y0); y1c = min(ny, y1)
    x0c = max(0, x0); x1c = min(nx, x1)
    crop = roi[z, y0c:y1c, x0c:x1c]
    return float(crop.sum()), crop


# ── CT crop 로드 ─────────────────────────────────────────────────────────────
def load_ct_crop(crop_path: str):
    """ct_crop (3, 96, 96) → 중앙 채널 (96, 96)"""
    try:
        d = np.load(crop_path)
        ct = d["ct_crop"]
        if ct.ndim == 3:
            return ct[ct.shape[0] // 2].astype(float)
        return ct.astype(float)
    except Exception:
        return None


# ── 샘플 선정 ─────────────────────────────────────────────────────────────────
def select_samples(df: pd.DataFrame, n: int) -> dict:
    """low / mid / high ratio 구간에서 n개씩 선정"""
    cr = df["crop_lung_roi_ratio"].dropna()
    low_thr  = cr.quantile(0.15)
    high_thr = cr.quantile(0.85)
    mid_lo   = cr.quantile(0.40)
    mid_hi   = cr.quantile(0.60)

    low_df  = df[df["crop_lung_roi_ratio"] <= low_thr].sample(min(n, len(df[df["crop_lung_roi_ratio"] <= low_thr])), random_state=42)
    mid_df  = df[(df["crop_lung_roi_ratio"] >= mid_lo) & (df["crop_lung_roi_ratio"] <= mid_hi)].sample(min(n, len(df[(df["crop_lung_roi_ratio"] >= mid_lo) & (df["crop_lung_roi_ratio"] <= mid_hi)])), random_state=42)
    high_df = df[df["crop_lung_roi_ratio"] >= high_thr].sample(min(n, len(df[df["crop_lung_roi_ratio"] >= high_thr])), random_state=42)

    return {"low": low_df, "mid": mid_df, "high": high_df}


# ── 단일 샘플 시각화 ─────────────────────────────────────────────────────────
def visualize_sample(ax_ct, ax_roi, ax_info, row, roi_map: dict):
    """한 행에 대해 3개의 axes에 시각화: CT crop / ROI slice+bbox / info"""
    safe_id = str(row["safe_id"])
    z = int(row["canonical_volume_z"])
    y0 = int(row["y0"]); x0 = int(row["x0"])
    y1 = int(row["y1"]); x1 = int(row["x1"])
    bbox_h = y1 - y0
    bbox_w = x1 - x0
    current_ratio = float(row["crop_lung_roi_ratio"])

    # corrected ratio 계산
    roi = load_roi(safe_id, roi_map)
    if roi is not None:
        roi_sum, roi_crop = get_roi_sum_and_crop(roi, z, y0, x0, y1, x1)
        corrected_ratio = roi_sum / (bbox_h * bbox_w) if roi_sum is not None else float("nan")
    else:
        roi_sum = None
        corrected_ratio = float("nan")
        roi_crop = None

    # CT crop
    ct = load_ct_crop(str(row["crop_path"]))
    if ct is not None:
        ax_ct.imshow(ct, cmap="gray", aspect="equal",
                     vmin=np.percentile(ct, 1), vmax=np.percentile(ct, 99))
        # bbox 위치를 CT에 오버레이 (CT는 96×96이나 roi bbox는 512 좌표계 → 표시만)
        ax_ct.set_title(f"CT crop\n{Path(row['crop_path']).name[:20]}", fontsize=6)
    else:
        ax_ct.text(0.5, 0.5, "load fail", ha="center", va="center", transform=ax_ct.transAxes, fontsize=7)
        ax_ct.set_title("CT crop (fail)", fontsize=6)
    ax_ct.axis("off")

    # ROI slice + bbox
    if roi is not None and z < roi.shape[0]:
        roi_slice = roi[z]  # (512, 512)
        ax_roi.imshow(roi_slice, cmap="Blues", aspect="equal", vmin=0, vmax=1)
        rect = mpatches.Rectangle((x0, y0), bbox_w, bbox_h,
                                   linewidth=1.5, edgecolor="red", facecolor="none")
        ax_roi.add_patch(rect)
        ax_roi.set_xlim(max(0, x0 - 50), min(roi_slice.shape[1], x1 + 50))
        ax_roi.set_ylim(min(roi_slice.shape[0], y1 + 50), max(0, y0 - 50))
        ax_roi.set_title(f"ROI z={z} (bbox {bbox_h}×{bbox_w})", fontsize=6)
    else:
        ax_roi.text(0.5, 0.5, "no ROI", ha="center", va="center", transform=ax_roi.transAxes, fontsize=7)
        ax_roi.set_title("ROI (none)", fontsize=6)
    ax_roi.axis("off")

    # info text
    info_lines = [
        f"safe_id: {safe_id[:20]}",
        f"z={z}  bbox={bbox_h}×{bbox_w}",
        f"roi_sum={roi_sum:.1f}" if roi_sum is not None else "roi_sum=N/A",
        f"",
        f"current_ratio:",
        f"  {roi_sum:.0f} / {CROP_SIZE*CROP_SIZE} = {current_ratio:.4f}" if roi_sum is not None else f"  {current_ratio:.4f}",
        f"",
        f"corrected_ratio:",
        f"  {roi_sum:.0f} / ({bbox_h}×{bbox_w}) = {corrected_ratio:.4f}" if roi_sum is not None else "  N/A",
        f"",
        f"scale_factor: {corrected_ratio/current_ratio:.2f}x" if (not np.isnan(corrected_ratio) and current_ratio > 0) else "",
    ]
    ax_info.text(0.05, 0.95, "\n".join(info_lines),
                 transform=ax_info.transAxes, fontsize=6,
                 verticalalignment="top", fontfamily="monospace")
    ax_info.axis("off")

    return {
        "safe_id": safe_id,
        "z": z,
        "bbox_h": bbox_h,
        "bbox_w": bbox_w,
        "roi_sum": float(roi_sum) if roi_sum is not None else float("nan"),
        "current_denominator": CROP_SIZE * CROP_SIZE,
        "corrected_denominator": bbox_h * bbox_w,
        "current_ratio": current_ratio,
        "corrected_ratio": float(corrected_ratio),
        "scale_factor": float(corrected_ratio / current_ratio) if (not np.isnan(corrected_ratio) and current_ratio > 0) else float("nan"),
    }


# ── contact sheet 생성 ────────────────────────────────────────────────────────
def make_contact_sheet(samples_df: pd.DataFrame, tier: str,
                       roi_map: dict, out_dir: Path) -> list:
    """N개 샘플의 contact sheet PNG 생성. row당 CT/ROI/info 3개 cols."""
    n = len(samples_df)
    ncols = 3
    nrows = n

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f"P-C-NORMAL24h audit: {tier.upper()} ratio tier (n={n})\n"
                 f"col1=CT crop, col2=ROI+bbox(red), col3=ratio info",
                 fontsize=9, y=1.01)

    sample_stats = []
    for i, (_, row) in enumerate(samples_df.iterrows()):
        stat = visualize_sample(axes[i, 0], axes[i, 1], axes[i, 2], row, roi_map)
        stat["tier"] = tier
        stat["row_index"] = i
        sample_stats.append(stat)

    plt.tight_layout()
    out_path = out_dir / f"p_c_normal24h_contact_sheet_{tier}.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")
    return sample_stats


# ── 분포 비교 플롯 ────────────────────────────────────────────────────────────
def make_distribution_plot(df: pd.DataFrame, out_dir: Path):
    """current vs corrected ratio 분포 비교 히스토그램"""
    df = df.copy()
    df["bbox_h"] = df["y1"] - df["y0"]
    df["bbox_w"] = df["x1"] - df["x0"]
    df["roi_sum_est"] = df["crop_lung_roi_ratio"] * (CROP_SIZE * CROP_SIZE)
    df["corrected_ratio"] = df["roi_sum_est"] / (df["bbox_h"] * df["bbox_w"])
    df["corrected_ratio_clipped"] = df["corrected_ratio"].clip(0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1) current ratio
    cr = df["crop_lung_roi_ratio"].dropna()
    axes[0].hist(cr, bins=50, color="steelblue", alpha=0.7)
    axes[0].axvline(1/9, color="red", linestyle="--", linewidth=1.5, label="1/9=0.111 (32x32 max)")
    axes[0].set_title(f"Current ratio (denom=96×96)\nmax={cr.max():.4f}  mean={cr.mean():.4f}")
    axes[0].set_xlabel("crop_lung_roi_ratio")
    axes[0].legend(fontsize=7)

    # 2) corrected ratio (raw, may exceed 1)
    cr2 = df["corrected_ratio"].dropna()
    axes[1].hist(cr2, bins=50, color="orange", alpha=0.7)
    axes[1].axvline(1.0, color="red", linestyle="--", linewidth=1.5, label="max=1.0")
    axes[1].set_title(f"Corrected ratio (denom=bbox_h×bbox_w)\nmax={cr2.max():.4f}  mean={cr2.mean():.4f}")
    axes[1].set_xlabel("corrected_ratio")
    axes[1].legend(fontsize=7)

    # 3) bbox 크기 분포
    df["bbox_area"] = df["bbox_h"] * df["bbox_w"]
    axes[2].hist(df["bbox_area"], bins=30, color="green", alpha=0.7)
    bbox_vc = df["bbox_h"].value_counts().to_dict()
    axes[2].set_title(f"BBox area distribution\nbbox_h vc={bbox_vc}")
    axes[2].set_xlabel("bbox_h × bbox_w")

    fig.suptitle("P-C-NORMAL24h: crop_lung_roi_ratio 분모 mismatch audit (train+val usable)",
                 fontsize=11)
    plt.tight_layout()
    out_path = out_dir / "p_c_normal24h_ratio_distribution.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── output collision guard ────────────────────────────────────────────────
    if REPORT_OUT.exists() and any(REPORT_OUT.iterdir()):
        print("[ABORT] REPORT_OUT already exists and is not empty.")
        print(f"  {REPORT_OUT}")
        sys.exit(2)

    REPORT_OUT.mkdir(parents=True, exist_ok=True)

    # ROI map
    print("ROI map 빌드 중...")
    roi_map = build_roi_map()
    print(f"  roi_map size: {len(roi_map)}")

    # train+val usable manifest 로드
    print("Manifest 로드 중...")
    train_df = pd.read_csv(MANIFEST_DIR / "p_c_normal24g_train_feature_manifest_usable.csv")
    val_df   = pd.read_csv(MANIFEST_DIR / "p_c_normal24g_val_feature_manifest_usable.csv")
    df = pd.concat([train_df, val_df], ignore_index=True)
    print(f"  total train+val usable: {len(df)}")

    # bbox 크기 분포 통계
    df["bbox_h"] = df["y1"] - df["y0"]
    df["bbox_w"] = df["x1"] - df["x0"]
    bbox_h_vc = df["bbox_h"].value_counts().to_dict()
    print(f"  bbox_h distribution: {bbox_h_vc}")

    # current ratio 통계
    cr = df["crop_lung_roi_ratio"].dropna()
    print(f"  current ratio: min={cr.min():.6f}, mean={cr.mean():.6f}, max={cr.max():.6f}")

    # corrected ratio 계산 (roi_sum = current_ratio * 9216)
    df["roi_sum_est"] = df["crop_lung_roi_ratio"] * (CROP_SIZE * CROP_SIZE)
    df["corrected_ratio"] = df["roi_sum_est"] / (df["bbox_h"] * df["bbox_w"])
    cr2 = df["corrected_ratio"].dropna()
    print(f"  corrected ratio: min={cr2.min():.6f}, mean={cr2.mean():.6f}, max={cr2.max():.6f}")

    # 분포 플롯
    print("분포 플롯 생성 중...")
    make_distribution_plot(df, REPORT_OUT)

    # low/mid/high 샘플 선정
    print("샘플 선정 중...")
    samples = select_samples(df, N_SAMPLE)
    for tier, sdf in samples.items():
        print(f"  {tier}: {len(sdf)}개 (ratio range {sdf['crop_lung_roi_ratio'].min():.4f}~{sdf['crop_lung_roi_ratio'].max():.4f})")

    # contact sheet 생성
    all_stats = []
    for tier, sdf in samples.items():
        print(f"\n{tier.upper()} contact sheet 생성 중...")
        stats = make_contact_sheet(sdf, tier, roi_map, REPORT_OUT)
        all_stats.extend(stats)

    # contact sheet별 corrected ratio 범위
    for tier in ["low", "mid", "high"]:
        tier_stats = [s for s in all_stats if s["tier"] == tier]
        if tier_stats:
            corr_vals = [s["corrected_ratio"] for s in tier_stats if not np.isnan(s["corrected_ratio"])]
            curr_vals = [s["current_ratio"] for s in tier_stats]
            scale_vals = [s["scale_factor"] for s in tier_stats if not np.isnan(s["scale_factor"])]
            print(f"  {tier}: corrected={min(corr_vals):.4f}~{max(corr_vals):.4f}, "
                  f"scale={min(scale_vals):.2f}~{max(scale_vals):.2f}x" if scale_vals else f"  {tier}: no scale info")

    # 32×32 케이스 통계
    df32 = df[df["bbox_h"] == 32]
    df96 = df[df["bbox_h"] == 96]
    cr32 = df32["crop_lung_roi_ratio"].dropna()
    cr96 = df96["crop_lung_roi_ratio"].dropna()

    # 판정 로직
    max_32bbox_ratio = cr32.max() if len(cr32) > 0 else 0
    has_32bbox_over_threshold = max_32bbox_ratio > (1/9 + 0.001)  # 0.1111 초과 여부

    # 32×32 bbox에서 0.1111 초과하는 케이스가 있으면 좌표계 문제
    # 32×32 bbox에서 max가 ~0.1111이면 분모 mismatch 확인됨
    if len(df32) > 0 and max_32bbox_ratio <= (1/9 + 0.005):
        verdict = "PASS_FIX_NEEDED"
        verdict_reason = (
            f"32×32 bbox (n={len(df32)}) current ratio max={max_32bbox_ratio:.6f} ≈ 1/9 (0.1111). "
            f"분모가 9216으로 고정되어 1/9 스케일 축소 확인됨. 24g 계산식 수정 필요."
        )
    elif len(df32) > 0 and has_32bbox_over_threshold:
        verdict = "FAIL"
        verdict_reason = (
            f"32×32 bbox에서 ratio>{1/9:.4f}인 케이스가 존재 (max={max_32bbox_ratio:.6f}). "
            f"bbox 좌표계 또는 ROI 좌표계 추가 확인 필요."
        )
    else:
        verdict = "PASS_NO_FIX"
        verdict_reason = "32×32 bbox가 없거나 ratio 분포가 올바름."

    # summary JSON 생성
    summary = {
        "branch": "P-C-NORMAL24h-audit",
        "step": "crop_lung_roi_ratio_denominator_mismatch_audit",
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "timestamp": ts,
        "dataset": {
            "train_usable": len(train_df),
            "val_usable": len(val_df),
            "total": len(df),
        },
        "bbox_size_distribution": {str(k): int(v) for k, v in bbox_h_vc.items()},
        "current_denominator": CROP_SIZE * CROP_SIZE,
        "corrected_denominator_formula": "bbox_h × bbox_w",
        "current_ratio_stats": {
            "min": float(cr.min()),
            "mean": float(cr.mean()),
            "max": float(cr.max()),
            "theoretical_max_32bbox": float(32 * 32 / (CROP_SIZE * CROP_SIZE)),
        },
        "corrected_ratio_stats": {
            "min": float(cr2.min()),
            "mean": float(cr2.mean()),
            "max": float(cr2.max()),
        },
        "by_bbox_size": {
            "32x32": {
                "count": int(len(df32)),
                "current_ratio_min": float(cr32.min()) if len(cr32) > 0 else None,
                "current_ratio_mean": float(cr32.mean()) if len(cr32) > 0 else None,
                "current_ratio_max": float(cr32.max()) if len(cr32) > 0 else None,
                "corrected_denominator": 32 * 32,
                "expected_max_current_ratio": float(32 * 32 / (CROP_SIZE * CROP_SIZE)),
            },
            "96x96": {
                "count": int(len(df96)),
                "current_ratio_min": float(cr96.min()) if len(cr96) > 0 else None,
                "current_ratio_mean": float(cr96.mean()) if len(cr96) > 0 else None,
                "current_ratio_max": float(cr96.max()) if len(cr96) > 0 else None,
                "corrected_denominator": 96 * 96,
                "note": "96x96 bbox: current denominator matches, no mismatch",
            },
        },
        "contact_sheets": [
            f"p_c_normal24h_contact_sheet_low.png",
            f"p_c_normal24h_contact_sheet_mid.png",
            f"p_c_normal24h_contact_sheet_high.png",
        ],
        "distribution_plot": "p_c_normal24h_ratio_distribution.png",
        "sample_detail": all_stats,
        "next_step_blocked": True,
        "next_step_note": (
            "24i smoke training 보류: 사용자 승인 필요. "
            "24g 재생성 여부도 사용자 결정 필요."
        ),
        "guardrails": {
            "existing_24g_results_modified": False,
            "existing_24h_results_modified": False,
            "model_training_run": False,
            "model_forward_run": False,
            "metrics_computed": False,
            "threshold_computed": False,
            "final_test_used": False,
            "vessel_feature_used": False,
        },
    }

    out_json = REPORT_OUT / "p_c_normal24h_audit_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  saved: {out_json}")

    # 간단한 sample_stats CSV
    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(REPORT_OUT / "p_c_normal24h_sample_stats.csv", index=False)

    # DONE.json
    with open(REPORT_OUT / "DONE.json", "w", encoding="utf-8") as f:
        json.dump({
            "step": "p_c_normal24h_audit_crop_lung_roi_ratio",
            "verdict": verdict,
            "timestamp": ts,
            "next_step_blocked": True,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"판정: {verdict}")
    print(f"  {verdict_reason}")
    print(f"bbox_h 분포: {bbox_h_vc}")
    print(f"current ratio: min={cr.min():.6f}, mean={cr.mean():.6f}, max={cr.max():.6f}")
    print(f"corrected ratio: min={cr2.min():.6f}, mean={cr2.mean():.6f}, max={cr2.max():.6f}")
    print(f"보고서: {REPORT_OUT}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
