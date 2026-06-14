"""
visualize_lesion_v2_review.py: v2 roi_0_0 1차 스크리닝 리뷰용 시각화.

- v2 no-hit 환자 2명 강제 포함 (LUNG1-156, LUNG1-415)
- weak/low-recall 2명, well-detected 2명, FP-heavy 2명 추가 선정 (총 8명 내외)
- 환자당 대표 slice 3장 이하 PNG 생성 (2패널: overlay / score heatmap)
- 기존 v1/v2 score CSV·npy 수정 없음. 신규 PNG·CSV만 생성.
- 출력 폴더가 이미 있으면 중단.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]

LESION_SCORE_DIR = REPO_ROOT / "outputs" / "position-aware-padim-v1" / "scores" / "padim_v1" / "lesion_v2_by_patient"
EVAL_DIR = REPO_ROOT / "outputs" / "position-aware-padim-v1" / "evaluation" / "lesion_subset_v2"
REPORTS_DIR = REPO_ROOT / "outputs" / "position-aware-padim-v1" / "reports"
VIZ_DIR = REPO_ROOT / "outputs" / "position-aware-padim-v1" / "visualizations" / "lesion_subset_v2_review"
STAGE_SPLIT_CSV = REPO_ROOT / "outputs" / "second-stage-lesion-refiner-v1" / "splits" / "lesion_stage_split_v1_balanced.csv"

MANIFEST_CSV = VIZ_DIR / "sample_cases_manifest_v2.csv"
NOHIT_LOWRECALL_CSV = REPORTS_DIR / "v2_nohit_and_lowrecall_cases.csv"

WL, WW = -600.0, 1500.0
MAX_SLICES_PER_PATIENT = 3

FORCE_NOHIT = [
    ("LUNG1-156", "NSCLC", "v2_no_hit"),
    ("LUNG1-415", "NSCLC", "v2_no_hit"),
]


# ── 시각화 헬퍼 ──────────────────────────────────────────────────────────────

def hu_to_rgb(slice_hu: np.ndarray) -> np.ndarray:
    lo, hi = WL - WW / 2.0, WL + WW / 2.0
    v = np.clip(slice_hu.astype(np.float32), lo, hi)
    v = (v - lo) / (hi - lo) * 255.0
    return np.stack([v.astype(np.uint8)] * 3, axis=-1)


def blend_mask(rgb: np.ndarray, mask: np.ndarray, color, alpha: float = 0.40) -> np.ndarray:
    out = rgb.astype(np.float32)
    m = mask > 0
    if m.any():
        c = np.array(color, dtype=np.float32)
        out[m] = out[m] * (1 - alpha) + c * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def make_panels(ct_z: np.ndarray, lesion_z: np.ndarray,
                patches_z: pd.DataFrame, thr: float) -> Image.Image:
    """좌: CT+lesion(red)+positive patch(blue box) / 우: score heatmap"""
    h, w = ct_z.shape
    base = hu_to_rgb(ct_z)

    # 좌 패널
    over = blend_mask(base.copy(), lesion_z, (220, 30, 30), alpha=0.40)
    over_img = Image.fromarray(over, "RGB")
    draw = ImageDraw.Draw(over_img)
    for r in patches_z.itertuples(index=False):
        if r.padim_score >= thr:
            draw.rectangle([int(r.x0), int(r.y0), int(r.x1) - 1, int(r.y1) - 1],
                           outline=(40, 120, 255), width=1)

    # 우 패널 (score heatmap)
    heat = np.zeros((h, w), dtype=np.float32)
    s = patches_z["padim_score"].values.astype(np.float32)
    if len(s) > 0:
        smin, smax = float(np.nanmin(s)), float(np.nanmax(s))
        rng = (smax - smin) if smax > smin else 1.0
        for r in patches_z.itertuples(index=False):
            val = (float(r.padim_score) - smin) / rng
            heat[int(r.y0):int(r.y1), int(r.x0):int(r.x1)] = np.maximum(
                heat[int(r.y0):int(r.y1), int(r.x0):int(r.x1)], val)
    heat_rgb = base.astype(np.float32)
    hv = heat * 255.0
    heat_rgb[..., 0] = np.maximum(heat_rgb[..., 0], hv)
    heat_rgb[..., 1] = heat_rgb[..., 1] * (1 - heat * 0.6)
    heat_rgb[..., 2] = heat_rgb[..., 2] * (1 - heat * 0.6)
    heat_img = Image.fromarray(np.clip(heat_rgb, 0, 255).astype(np.uint8), "RGB")

    canvas = Image.new("RGB", (w * 2 + 8, h), (0, 0, 0))
    canvas.paste(over_img, (0, 0))
    canvas.paste(heat_img, (w + 8, 0))
    return canvas


# ── 사례 선정 ─────────────────────────────────────────────────────────────────

def select_cases(pp: pd.DataFrame, hop: pd.DataFrame) -> list[tuple]:
    """
    Returns list of (patient_id, group, selection_reason, pp_row, hop_row)
    고정: no-hit 2명
    추가: weak 2명, good 2명, fp-heavy 2명
    """
    force_ids = {pid for pid, _, _ in FORCE_NOHIT}
    rest = pp[~pp["patient_id"].isin(force_ids)].copy()
    rest_hop = hop[~hop["patient_id"].isin(force_ids)].copy()
    rest = rest.merge(rest_hop[["patient_id", "stage_split", "patient_patch_recall",
                                "continuous_hit_ratio", "missed_lesion_slice_count"]],
                      on="patient_id", how="left")

    selected = []

    # 1) no-hit 강제 포함
    for pid, grp, reason in FORCE_NOHIT:
        pr = pp[pp["patient_id"] == pid]
        hr = hop[hop["patient_id"] == pid]
        if len(pr) == 0:
            print(f"  [WARN] {pid} not found in per_patient_screening")
            continue
        selected.append((pid, grp, reason, pr.iloc[0], hr.iloc[0] if len(hr) > 0 else None))

    used = force_ids.copy()

    def pick(df, sort_col, ascending, reason, n=2, hit_only=False):
        sub = df[~df["patient_id"].isin(used)].copy()
        if hit_only:
            # patient_hit 컬럼 없으면 recall_p95 > 0으로 대체
            sub = sub[sub["recall_p95"] > 0]
        sub = sub.sort_values(sort_col, ascending=ascending)
        picked = []
        for _, row in sub.iterrows():
            if len(picked) >= n:
                break
            pid = row["patient_id"]
            hr = hop[hop["patient_id"] == pid]
            selected.append((pid, row["group"], reason, row,
                             hr.iloc[0] if len(hr) > 0 else None))
            used.add(pid)
            picked.append(pid)

    # 2) weak/low-recall 2명 (hit 있음)
    pick(rest, "recall_p95", ascending=True, reason="weak_low_recall", n=2, hit_only=True)
    # 3) well-detected 2명
    pick(rest, "recall_p95", ascending=False, reason="well_detected_high_recall", n=2)
    # 4) FP-heavy 2명
    pick(rest, "n_positive_p95", ascending=False, reason="many_false_positive", n=2)

    return selected


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # 출력 폴더/CSV 존재 체크
    blocking = [p for p in (VIZ_DIR, MANIFEST_CSV, NOHIT_LOWRECALL_CSV) if Path(p).exists()]
    if blocking:
        print("[ERROR] 출력 파일/폴더가 이미 존재합니다. 덮어쓰기 금지 — 중단합니다:")
        for b in blocking:
            print(f"  {b}")
        sys.exit(1)

    # threshold
    with open(EVAL_DIR / "lesion_eval_v2_p95_fast_summary.json", encoding="utf-8") as f:
        thr = float(json.load(f)["threshold_value"])
    print(f"[viz_v2] p95 threshold = {thr:.4f}")

    # 데이터 로드
    pp = pd.read_csv(EVAL_DIR / "per_patient_screening.csv", encoding="utf-8-sig")
    hop = pd.read_csv(REPORTS_DIR / "lesion_hit_overlap_by_patient_v2.csv", encoding="utf-8-sig")

    # stage_split 보완 (hop에 없을 경우 stage_split CSV에서 읽음)
    if "stage_split" not in hop.columns and STAGE_SPLIT_CSV.exists():
        ss = pd.read_csv(STAGE_SPLIT_CSV, encoding="utf-8-sig",
                         usecols=["patient_id", "stage_split"])
        hop = hop.merge(ss, on="patient_id", how="left")

    # 데이터 루트
    cfg_path = REPO_ROOT / "configs" / "paths.local.yaml"
    with open(cfg_path, encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f) or {}
    DATA_ROOT = Path((cfg.get("nsclc_msd_usable_only_v2") or "").strip())

    # 사례 선정
    cases = select_cases(pp, hop)
    print(f"[viz_v2] 선정 케이스: {len(cases)}명")

    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    nohit_lr_rows = []

    for pid, grp, reason, prow, hrow in cases:
        safe_id = str(prow.get("safe_id", pid))
        ct_path = DATA_ROOT / "volumes_npy" / safe_id / "ct_hu.npy"
        lm_path = DATA_ROOT / "volumes_npy" / safe_id / "lesion_mask_roi_0_0.npy"
        score_csv = LESION_SCORE_DIR / f"{pid}.csv"

        if not (ct_path.exists() and lm_path.exists() and score_csv.exists()):
            print(f"  [SKIP] {pid}: 파일 없음 (ct={ct_path.exists()}, lm={lm_path.exists()}, score={score_csv.exists()})")
            continue

        df = pd.read_csv(score_csv, encoding="utf-8-sig",
                         usecols=["local_z", "y0", "x0", "y1", "x1", "padim_score", "patch_label"])
        df = df[~df["padim_score"].isna()]

        # 대표 slice 선정
        # no-hit: lesion patch가 많은 z (positive와 무관하게 병변이 보이는 slice)
        # 그 외: lesion patch 많은 z
        lesion_df = df[df["patch_label"] == 1]
        if len(lesion_df) == 0:
            print(f"  [SKIP] {pid}: lesion patch 없음")
            continue
        top_slices = (lesion_df.groupby("local_z").size()
                      .sort_values(ascending=False)
                      .head(MAX_SLICES_PER_PATIENT).index.tolist())

        ct = np.load(str(ct_path), mmap_mode="r")
        lm = np.load(str(lm_path), mmap_mode="r")

        shown = []
        png_paths = []
        for z in top_slices:
            z = int(z)
            if z < 0 or z >= ct.shape[0]:
                continue
            patches_z = df[df["local_z"] == z]
            canvas = make_panels(np.array(ct[z]), np.array(lm[z]), patches_z, thr)
            out_name = f"{grp}_{pid}_{reason}_z{z:03d}.png"
            out_path = VIZ_DIR / out_name
            canvas.save(str(out_path))
            shown.append(z)
            png_paths.append(str(out_path))
            print(f"  [OK] {out_name}")

        # hit_overlap 컬럼
        stage_split = hrow["stage_split"] if hrow is not None and "stage_split" in hrow.index else None
        patch_recall = hrow["patient_patch_recall"] if hrow is not None else float("nan")
        cont_ratio = hrow["continuous_hit_ratio"] if hrow is not None else float("nan")
        missed_slices = hrow["missed_lesion_slice_count"] if hrow is not None else float("nan")

        manifest_rows.append({
            "patient_id": pid,
            "group": grp,
            "stage_split": stage_split,
            "selection_reason": reason,
            "patient_patch_recall": round(float(patch_recall), 4) if pd.notna(patch_recall) else None,
            "continuous_hit_ratio": round(float(cont_ratio), 4) if pd.notna(cont_ratio) else None,
            "lesion_patch_count": int(prow.get("lesion_patch_total", 0)),
            "lesion_slice_count": int(prow.get("lesion_slice_total", 0)),
            "missed_lesion_slice_count": int(missed_slices) if pd.notna(missed_slices) else None,
            "n_positive_p95": int(prow.get("n_positive_p95", 0)),
            "representative_slices": ";".join(str(z) for z in shown),
            "output_png_paths": ";".join(png_paths),
        })

        # no-hit / low-recall CSV
        if reason in ("v2_no_hit", "weak_low_recall"):
            nohit_lr_rows.append({
                "patient_id": pid,
                "group": grp,
                "stage_split": stage_split,
                "selection_reason": reason,
                "patient_patch_recall": round(float(patch_recall), 4) if pd.notna(patch_recall) else None,
                "continuous_hit_ratio": round(float(cont_ratio), 4) if pd.notna(cont_ratio) else None,
                "lesion_patch_count": int(prow.get("lesion_patch_total", 0)),
                "lesion_slice_count": int(prow.get("lesion_slice_total", 0)),
                "missed_lesion_slice_count": int(missed_slices) if pd.notna(missed_slices) else None,
                "n_positive_p95": int(prow.get("n_positive_p95", 0)),
                "recall_p95": round(float(prow.get("recall_p95", 0)), 4),
                "slice_cov_p95": round(float(prow.get("slice_cov_p95", 0)), 4),
            })

    # CSV 저장
    pd.DataFrame(manifest_rows).to_csv(MANIFEST_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(nohit_lr_rows).to_csv(NOHIT_LOWRECALL_CSV, index=False, encoding="utf-8-sig")

    n_png = sum(len(r["output_png_paths"].split(";")) for r in manifest_rows if r["output_png_paths"])
    print(f"\n[viz_v2] PNG {n_png}개 생성")
    print(f"[viz_v2] manifest: {MANIFEST_CSV}")
    print(f"[viz_v2] no-hit/low-recall CSV: {NOHIT_LOWRECALL_CSV}")
    print(f"[viz_v2] 시각화 폴더: {VIZ_DIR}")
    print("[viz_v2] 좌=CT+lesion(red)+positive(blue) / 우=score heatmap / positive=score>=p95")


if __name__ == "__main__":
    main()
