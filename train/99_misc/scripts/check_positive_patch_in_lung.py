"""
check_positive_patch_in_lung.py
positive patch(padim_score >= p95)가 폐 마스크(model_roi / pure_lung) 안에 있는지 검증한다.

- score CSV는 read-only. score 재계산 / 모델 실행 없음.
- 308명 positive patch의 model_roi_patch_ratio / pure_lung_patch_ratio 분포 집계.
- sample 6명은 model_roi.npy / pure_lung.npy를 직접 열어 patch 영역 overlap을 재계산,
  CSV 기록 ratio와 일치하는지 검증한다. (폐 밖/경계 의심 patch 우선 선택)
- 출력 파일이 이미 있으면 실행을 중단한다(기존 report 덮어쓰기 금지).
"""
from __future__ import annotations
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
LESION_SCORE_DIR_V1 = REPO_ROOT / "outputs/position-aware-padim-v1/scores/padim_v1/lesion_by_patient"
LESION_SCORE_DIR_V2 = REPO_ROOT / "outputs/position-aware-padim-v1/scores/padim_v1/lesion_v2_by_patient"
EVAL_DIR_V1 = REPO_ROOT / "outputs/position-aware-padim-v1/evaluation/lesion_subset"
EVAL_DIR_V2 = REPO_ROOT / "outputs/position-aware-padim-v1/evaluation/lesion_subset_v2"
VIZ_DIR_V1 = REPO_ROOT / "outputs/position-aware-padim-v1/visualizations/lesion_subset_screening_review"
VIZ_DIR_V2 = REPO_ROOT / "outputs/position-aware-padim-v1/visualizations/lesion_subset_v2_screening_review"
REPORTS_DIR = REPO_ROOT / "outputs/position-aware-padim-v1/reports"

# v1 고정 컬럼
USECOLS_V1 = ["patient_id", "safe_id", "local_z", "y0", "x0", "y1", "x1",
              "padim_score", "model_roi_patch_ratio", "pure_lung_patch_ratio"]
# v2 고정 컬럼 (pure_lung 없음, roi_0_0 사용)
USECOLS_V2 = ["patient_id", "safe_id", "local_z", "y0", "x0", "y1", "x1",
              "padim_score", "roi_0_0_patch_ratio"]

SAMPLE_RECHECK_PER_PATIENT = 50  # 샘플 6명에서 재계산할 positive patch 수
EPS = 1e-12                      # ratio zero 판정 tolerance


def load_p95(eval_dir: Path, v2_prefix: str = "") -> float:
    fname = f"lesion_eval_{v2_prefix}p95_fast_summary.json"
    with open(eval_dir / fname, encoding="utf-8") as f:
        return float(json.load(f)["threshold_value"])


def ratio_bin(v: float) -> str:
    if v <= EPS:
        return "==0"
    if v < 0.1:
        return "0~0.1"
    if v < 0.5:
        return "0.1~0.5"
    return ">=0.5"


def main() -> None:
    parser = argparse.ArgumentParser(description="positive patch ROI 포함 여부 검증")
    parser.add_argument(
        "--dataset-profile",
        type=str,
        default="v1_model_roi",
        choices=["v1_model_roi", "v2_roi_0_0"],
        help="검증할 데이터셋 profile (기본값: v1_model_roi).",
    )
    args = parser.parse_args()
    is_v2 = (args.dataset_profile == "v2_roi_0_0")

    LESION_SCORE_DIR = LESION_SCORE_DIR_V2 if is_v2 else LESION_SCORE_DIR_V1
    EVAL_DIR = EVAL_DIR_V2 if is_v2 else EVAL_DIR_V1
    VIZ_DIR = VIZ_DIR_V2 if is_v2 else VIZ_DIR_V1
    v2_prefix = "v2_" if is_v2 else ""
    USECOLS = USECOLS_V2 if is_v2 else USECOLS_V1
    roi_col = "roi_0_0_patch_ratio" if is_v2 else "model_roi_patch_ratio"

    # DATA_ROOT: config에서 읽음
    cfg_path = REPO_ROOT / "configs" / "paths.local.yaml"
    with open(cfg_path, encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f) or {}
    cfg_key = "nsclc_msd_usable_only_v2" if is_v2 else "nsclc_msd_usable_only"
    DATA_ROOT = Path((cfg.get(cfg_key) or "").strip())

    # v2 ROI 파일명: roi_0_0.npy (pure_lung 없음)
    roi_npy = "roi_0_0.npy" if is_v2 else "model_roi.npy"

    OUT_CSV = REPORTS_DIR / (f"lesion_positive_patch_roi_check_v2.csv" if is_v2
                             else "lesion_positive_patch_roi_check.csv")
    OUT_JSON = REPORTS_DIR / (f"lesion_positive_patch_roi_check_summary_v2.json" if is_v2
                              else "lesion_positive_patch_roi_check_summary.json")
    OUT_RECHECK_CSV = REPORTS_DIR / (f"lesion_positive_patch_roi_recheck_samples_v2.csv" if is_v2
                                     else "lesion_positive_patch_roi_recheck_samples.csv")

    print(f"[roi_check] dataset_profile={args.dataset_profile}, roi_col={roi_col}")

    # --- 출력 파일 덮어쓰기 방지: 하나라도 있으면 중단 ---
    existing = [str(p) for p in (OUT_CSV, OUT_JSON, OUT_RECHECK_CSV) if p.exists()]
    if existing:
        print("[ERROR] 출력 파일이 이미 존재합니다. 덮어쓰기 금지 — 실행을 중단합니다:")
        for e in existing:
            print(f"  - {e}")
        sys.exit(1)

    thr = load_p95(EVAL_DIR, v2_prefix)
    print(f"[roi_check] p95 threshold = {thr:.4f}")

    csv_paths = sorted(glob.glob(str(LESION_SCORE_DIR / "*.csv")))

    mr_bins = {"==0": 0, "0~0.1": 0, "0.1~0.5": 0, ">=0.5": 0}
    pl_bins = {"==0": 0, "0~0.1": 0, "0.1~0.5": 0, ">=0.5": 0}
    mr_zero = pl_zero = pos_total = patch_total = 0
    all_mr_sum = all_pl_sum = all_n = 0.0
    pos_mr_sum = pos_pl_sum = 0.0
    per_patient_rows = []

    for p in csv_paths:
        df = pd.read_csv(p, encoding="utf-8-sig", usecols=USECOLS)
        df = df[~df["padim_score"].isna()]
        patch_total += len(df)
        all_mr_sum += df[roi_col].sum()
        if not is_v2 and "pure_lung_patch_ratio" in df.columns:
            all_pl_sum += df["pure_lung_patch_ratio"].sum()
        all_n += len(df)

        pos = df[df["padim_score"] >= thr]
        pos_total += len(pos)
        mr = pos[roi_col].values
        mr_zero += int((mr <= EPS).sum())
        pos_mr_sum += mr.sum()
        for v in mr:
            mr_bins[ratio_bin(float(v))] += 1

        row_data = {
            "patient_id": str(df["patient_id"].iloc[0]),
            "safe_id": str(df["safe_id"].iloc[0]),
            "n_positive": int(len(pos)),
            f"n_pos_{roi_col}_zero": int((mr <= EPS).sum()),
            f"pos_{roi_col}_mean": float(mr.mean()) if len(mr) else None,
        }
        if not is_v2 and "pure_lung_patch_ratio" in pos.columns:
            pl = pos["pure_lung_patch_ratio"].values
            pl_zero += int((pl <= EPS).sum())
            pos_pl_sum += pl.sum()
            for v in pl:
                pl_bins[ratio_bin(float(v))] += 1
            row_data["n_pos_pure_lung_zero"] = int((pl <= EPS).sum())
            row_data["pos_pure_lung_ratio_mean"] = float(pl.mean()) if len(pl) else None
        per_patient_rows.append(row_data)

    # ---- sample 6명: 폐 밖/경계 의심 patch 우선 선택 후 mask로 overlap 재계산 ----
    sample_manifest_path = VIZ_DIR / "sample_cases_manifest.csv"
    recheck_rows = []
    if sample_manifest_path.exists():
        manifest = pd.read_csv(sample_manifest_path, encoding="utf-8-sig")
        for _, m in manifest.iterrows():
            pid, safe_id = str(m["patient_id"]), str(m["safe_id"])
            roi_path = DATA_ROOT / "volumes_npy" / safe_id / roi_npy
            score_csv = LESION_SCORE_DIR / f"{pid}.csv"
            if not (roi_path.exists() and score_csv.exists()):
                continue
            roi_vol = np.load(str(roi_path), mmap_mode="r")
            df = pd.read_csv(score_csv, encoding="utf-8-sig", usecols=USECOLS)
            df = df[(~df["padim_score"].isna()) & (df["padim_score"] >= thr)]
            df = df.sort_values([roi_col, "padim_score"], ascending=[True, False])
            sample = df.head(SAMPLE_RECHECK_PER_PATIENT)
            for r in sample.itertuples(index=False):
                z, y0, x0, y1, x1 = int(r.local_z), int(r.y0), int(r.x0), int(r.y1), int(r.x1)
                area = max((y1 - y0) * (x1 - x0), 1)
                roi_recompute = float((np.array(roi_vol[z, y0:y1, x0:x1]) > 0).sum()) / area
                csv_roi = float(getattr(r, roi_col.replace(".", "_"), 0.0))
                recheck_row = {
                    "patient_id": pid, "local_z": z, "y0": y0, "x0": x0, "y1": y1, "x1": x1,
                    "padim_score": float(r.padim_score),
                    f"csv_{roi_col}": csv_roi,
                    f"recompute_{roi_col}": round(roi_recompute, 6),
                    f"{roi_col}_diff": round(abs(csv_roi - roi_recompute), 6),
                }
                recheck_rows.append(recheck_row)
    else:
        print(f"[roi_check] sample_cases_manifest.csv 없음 — recheck 건너뜀: {sample_manifest_path}")

    verdict = (f"ROI_밖_positive_없음" if mr_zero == 0
               else "ROI_밖_positive_존재_patch생성기준확인필요")

    recheck_df = pd.DataFrame(recheck_rows)
    diff_col = f"{roi_col}_diff"
    summary = {
        "dataset_profile": args.dataset_profile,
        "roi_col": roi_col,
        "p95_threshold": thr,
        "eps": EPS,
        "patch_total": patch_total,
        "positive_total": pos_total,
        f"positive_{roi_col}_zero": mr_zero,
        f"positive_{roi_col}_bins": mr_bins,
        f"all_patch_{roi_col}_mean": (all_mr_sum / all_n) if all_n else None,
        f"positive_{roi_col}_mean": (pos_mr_sum / pos_total) if pos_total else None,
        "sample_recheck_max_roi_diff": (
            float(recheck_df[diff_col].max()) if len(recheck_df) > 0 and diff_col in recheck_df.columns else None),
        "verdict": verdict,
        "note": "read-only 검증. score 재계산 없음. sample은 ROI 밖/경계 의심 우선 표본(대표성 아님). 성능 결론 아님. ROI 밖 patch 포함 여부 검증 목적.",
    }
    if not is_v2:
        summary["positive_pure_lung_ratio_zero"] = pl_zero
        summary["positive_pure_lung_ratio_bins"] = pl_bins
        summary["all_patch_pure_lung_ratio_mean"] = (all_pl_sum / all_n) if all_n else None
        summary["positive_pure_lung_ratio_mean"] = (pos_pl_sum / pos_total) if pos_total else None

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_patient_rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    recheck_df.to_csv(OUT_RECHECK_CSV, index=False, encoding="utf-8-sig")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
