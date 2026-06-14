"""
visualize_lesion_screening.py: 1차 스크리닝 관점 샘플 환자 시각화.

- per_patient_screening.csv 기반으로 NSCLC 3명 + MSD_Lung 3명을 선정한다
  (잘 잡힌 예 / 과탐 많은 예 / 놓친(약한) 예).
- 각 환자의 대표 lesion slice 1~3장을 PIL로 시각화한다:
    좌 패널: CT(폐 window) + lesion_mask(red) + positive patch(blue box) overlay
    우 패널: patch score heatmap (PIL 간이; matplotlib 미설치 환경)
- positive = padim_score >= threshold(p95).

안전 원칙:
- score 재계산 / 모델 실행 없음. score CSV·npy는 read-only.
- 기존 결과 수정·삭제 없음. 신규 PNG / manifest만 생성.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import argparse
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
LESION_SCORE_DIR_V1 = REPO_ROOT / "outputs" / "position-aware-padim-v1" / "scores" / "padim_v1" / "lesion_by_patient"
LESION_SCORE_DIR_V2 = REPO_ROOT / "outputs" / "position-aware-padim-v1" / "scores" / "padim_v1" / "lesion_v2_by_patient"
EVAL_DIR_V1 = REPO_ROOT / "outputs" / "position-aware-padim-v1" / "evaluation" / "lesion_subset"
EVAL_DIR_V2 = REPO_ROOT / "outputs" / "position-aware-padim-v1" / "evaluation" / "lesion_subset_v2"
VIZ_DIR_V1 = REPO_ROOT / "outputs" / "position-aware-padim-v1" / "visualizations" / "lesion_subset_screening_review"
VIZ_DIR_V2 = REPO_ROOT / "outputs" / "position-aware-padim-v1" / "visualizations" / "lesion_subset_v2_screening_review"

WL, WW = -600.0, 1500.0          # 폐 window
MAX_SLICES_PER_PATIENT = 3
MIN_LESION_PATCH_FOR_SELECT = 50  # selection 후보 최소 lesion patch 수


def hu_to_rgb(slice_hu: np.ndarray) -> np.ndarray:
    lo, hi = WL - WW / 2.0, WL + WW / 2.0
    v = np.clip(slice_hu.astype(np.float32), lo, hi)
    v = (v - lo) / (hi - lo) * 255.0
    u8 = v.astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)


def blend_mask(rgb: np.ndarray, mask: np.ndarray, color, alpha=0.40) -> np.ndarray:
    out = rgb.astype(np.float32)
    m = mask > 0
    if m.any():
        c = np.array(color, dtype=np.float32)
        out[m] = out[m] * (1 - alpha) + c * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def load_threshold_p95(eval_dir: Path, v2_prefix: str = "") -> float:
    fname = f"lesion_eval_{v2_prefix}p95_fast_summary.json"
    with open(eval_dir / fname, encoding="utf-8") as f:
        return float(json.load(f)["threshold_value"])


def select_cases(pp: pd.DataFrame) -> list:
    """그룹별(NSCLC/MSD_Lung) 잘잡힘/과탐/놓침 1명씩 선정."""
    selected = []
    for grp in ["NSCLC", "MSD_Lung"]:
        sub = pp[(pp["group"] == grp) & (pp["lesion_patch_total"] >= MIN_LESION_PATCH_FOR_SELECT)].copy()
        if len(sub) == 0:
            sub = pp[pp["group"] == grp].copy()
        if len(sub) == 0:
            continue
        used = set()

        # 잘 잡힌 예: recall_p95 최대
        well = sub.sort_values("recall_p95", ascending=False).iloc[0]
        selected.append((well["patient_id"], grp, "well_detected_high_recall", well))
        used.add(well["patient_id"])

        # 과탐 많은 예: n_fp_p95 최대 (used 제외)
        sub2 = sub[~sub["patient_id"].isin(used)]
        if len(sub2) > 0:
            over = sub2.sort_values("n_fp_p95", ascending=False).iloc[0]
            selected.append((over["patient_id"], grp, "many_false_positive", over))
            used.add(over["patient_id"])

        # 놓친/약한 예: recall_p95 최소 (used 제외)
        sub3 = sub[~sub["patient_id"].isin(used)]
        if len(sub3) > 0:
            miss = sub3.sort_values("recall_p95", ascending=True).iloc[0]
            selected.append((miss["patient_id"], grp, "missed_or_weak_low_recall", miss))
            used.add(miss["patient_id"])

    return selected


def make_panels(ct_z, lesion_z, patches_z, thr) -> Image.Image:
    """좌: overlay(CT+lesion red+positive patch blue), 우: score heatmap. 가로 concat."""
    h, w = ct_z.shape
    base = hu_to_rgb(ct_z)

    # --- 좌: overlay ---
    over = blend_mask(base, lesion_z, (220, 30, 30), alpha=0.40)  # lesion = red
    over_img = Image.fromarray(over, "RGB")
    draw = ImageDraw.Draw(over_img)
    for r in patches_z.itertuples(index=False):
        if r.padim_score >= thr:
            # positive patch = blue box
            draw.rectangle([int(r.x0), int(r.y0), int(r.x1) - 1, int(r.y1) - 1],
                           outline=(40, 120, 255), width=1)

    # --- 우: score heatmap (PIL 간이) ---
    heat = np.zeros((h, w), dtype=np.float32)
    s = patches_z["padim_score"].values.astype(np.float32)
    if len(s) > 0:
        smin, smax = float(np.nanmin(s)), float(np.nanmax(s))
        rng = (smax - smin) if smax > smin else 1.0
        for r in patches_z.itertuples(index=False):
            val = (float(r.padim_score) - smin) / rng
            y0, x0, y1, x1 = int(r.y0), int(r.x0), int(r.y1), int(r.x1)
            heat[y0:y1, x0:x1] = np.maximum(heat[y0:y1, x0:x1], val)
    # grayscale 위에 빨강 강도로 blend
    heat_rgb = base.astype(np.float32)
    hv = (heat * 255.0)
    heat_rgb[..., 0] = np.maximum(heat_rgb[..., 0], hv)        # R 강조
    heat_rgb[..., 1] = heat_rgb[..., 1] * (1 - heat * 0.6)
    heat_rgb[..., 2] = heat_rgb[..., 2] * (1 - heat * 0.6)
    heat_img = Image.fromarray(np.clip(heat_rgb, 0, 255).astype(np.uint8), "RGB")

    # concat (가로)
    canvas = Image.new("RGB", (w * 2 + 8, h), (0, 0, 0))
    canvas.paste(over_img, (0, 0))
    canvas.paste(heat_img, (w + 8, 0))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="lesion 스크리닝 시각화")
    parser.add_argument(
        "--dataset-profile",
        type=str,
        default="v1_model_roi",
        choices=["v1_model_roi", "v2_roi_0_0"],
        help="시각화할 데이터셋 profile (기본값: v1_model_roi).",
    )
    args = parser.parse_args()
    is_v2 = (args.dataset_profile == "v2_roi_0_0")

    LESION_SCORE_DIR = LESION_SCORE_DIR_V2 if is_v2 else LESION_SCORE_DIR_V1
    EVAL_DIR = EVAL_DIR_V2 if is_v2 else EVAL_DIR_V1
    VIZ_DIR = VIZ_DIR_V2 if is_v2 else VIZ_DIR_V1
    v2_prefix = "v2_" if is_v2 else ""
    lesion_mask_npy = "lesion_mask_roi_0_0.npy" if is_v2 else "lesion_mask_model_roi.npy"

    # DATA_ROOT: config에서 읽음
    cfg_path = REPO_ROOT / "configs" / "paths.local.yaml"
    with open(cfg_path, encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f) or {}
    cfg_key = "nsclc_msd_usable_only_v2" if is_v2 else "nsclc_msd_usable_only"
    LESION_DATA_ROOT = Path((cfg.get(cfg_key) or "").strip())

    thr = load_threshold_p95(EVAL_DIR, v2_prefix)
    print(f"[viz] dataset_profile={args.dataset_profile}")
    print(f"[viz] p95 threshold = {thr:.4f}")

    pp = pd.read_csv(EVAL_DIR / "per_patient_screening.csv", encoding="utf-8-sig")
    cases = select_cases(pp)
    print(f"[viz] 선정 케이스 {len(cases)}명")

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for pid, grp, reason, row in cases:
        safe_id = str(row["safe_id"])
        ct_path = LESION_DATA_ROOT / "volumes_npy" / safe_id / "ct_hu.npy"
        lm_path = LESION_DATA_ROOT / "volumes_npy" / safe_id / lesion_mask_npy
        score_csv = LESION_SCORE_DIR / f"{pid}.csv"
        if not (ct_path.exists() and lm_path.exists() and score_csv.exists()):
            print(f"  [SKIP] {pid}: 파일 없음")
            continue

        df = pd.read_csv(score_csv, encoding="utf-8-sig",
                         usecols=["local_z", "y0", "x0", "y1", "x1", "padim_score", "patch_label"])
        df = df[~df["padim_score"].isna()]

        # 대표 lesion slice: lesion patch 많은 local_z top N
        lesion_df = df[df["patch_label"] == 1]
        if len(lesion_df) == 0:
            print(f"  [SKIP] {pid}: lesion patch 없음")
            continue
        top_slices = (lesion_df.groupby("local_z").size()
                      .sort_values(ascending=False).head(MAX_SLICES_PER_PATIENT).index.tolist())

        ct = np.load(str(ct_path), mmap_mode="r")
        lm = np.load(str(lm_path), mmap_mode="r")

        shown = []
        for z in top_slices:
            z = int(z)
            if z < 0 or z >= ct.shape[0]:
                continue
            patches_z = df[df["local_z"] == z]
            canvas = make_panels(np.array(ct[z]), np.array(lm[z]), patches_z, thr)
            out_name = f"{grp}_{pid}_{reason}_z{z:03d}.png"
            canvas.save(str(VIZ_DIR / out_name))
            shown.append(z)
            print(f"  [OK] {out_name}")

        manifest_rows.append({
            "patient_id": pid, "safe_id": safe_id, "group": grp,
            "selection_reason": reason,
            "lesion_patch_total": int(row["lesion_patch_total"]),
            "recall_p95": round(float(row["recall_p95"]), 4),
            "n_positive_p95": int(row["n_positive_p95"]),
            "n_fp_p95": int(row["n_fp_p95"]),
            "slice_cov_p95": round(float(row["slice_cov_p95"]), 4),
            "slices_shown": ";".join(str(z) for z in shown),
        })

    manifest_csv = VIZ_DIR / "sample_cases_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False, encoding="utf-8-sig")
    print(f"\n[viz] manifest: {manifest_csv}")
    print(f"[viz] 시각화 폴더: {VIZ_DIR}")
    print("[viz] 좌=CT+lesion(red)+positive patch(blue) / 우=score heatmap. positive=score>=p95 threshold.")


if __name__ == "__main__":
    main()
