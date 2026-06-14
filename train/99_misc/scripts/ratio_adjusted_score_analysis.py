"""
ratio_adjusted_score_analysis.py
(ChatGPT 검토/승인 후에만 실행)

ROI/공기층 기반 ratio-adjusted score를 v1/v2, v2/v2 score CSV에 적용하고,
original vs 보정 방식 비교 분석 결과를 생성한다.

실행 방법:
  --sample   : sample 15명 처리 (기본 실행 모드)
  --full-run : 전체 308명 처리 (별도 승인 후에만 사용)

기본 실행(옵션 없음)은 abort됨.

제약:
- 기존 score CSV는 read-only. 수정/덮어쓰기 절대 금지.
- scoring/모델 학습 재실행 없음.
- pixel-level score map 없음 → patch-level ratio-adjusted score로 제한.
- pure_lung.npy 없음 → roi_0_0 기준 variant만 계산.
- npy 파일은 환자별 순차 로드 후 즉시 해제 (OOM 방지).
- near_wall boundary distance 계산 미구현 → position_bin/central_peripheral proxy 사용.
- global top-k(top10/30/50)는 전체 patch 전역 rank 기준이므로 해석 제한 명시.
- lesion_size_bin은 patch-level lesion occupancy bin이며 patient-level 병변 크기가 아님.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]

V1V2_SCORE_DIR = REPO_ROOT / "outputs/position-aware-padim-v1/scores/padim_v1/lesion_v2_by_patient"
V2V2_SCORE_DIR = REPO_ROOT / "outputs/position-aware-padim-v1/scores/padim_v2_roi0_0/lesion_v2_by_patient"
NPY_BASE = Path("/mnt/c/Users/jinhy/Desktop/NSCLC_MSD_padim_test_ready_roi0_0_ts_lung_raw_no_dilate_usable_only_v1/volumes_npy")
OUT_DIR = REPO_ROOT / "outputs/position-aware-padim-v1/reports_v2_roi0_0_lesion"

# per_patient_screening CSV (read-only, patient-level lesion size join용)
# v1/v2: v2 병변 테스트셋 기준 → lesion_subset_v2 사용 (lesion_subset은 v1/v1 기준)
PER_PATIENT_SCREEN_V1V2 = REPO_ROOT / "outputs/position-aware-padim-v1/evaluation/lesion_subset_v2/per_patient_screening.csv"
PER_PATIENT_SCREEN_V2V2 = REPO_ROOT / "outputs/position-aware-padim-v1/evaluation/lesion_subset_v2_model_v2/per_patient_screening.csv"

# ---------------------------------------------------------------------------
# Threshold (사전 확인 완료)
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "v1v2": {"p95": 14.377350028011772, "p99": 18.672782302362954},
    "v2v2": {"p95": 14.092057666455288, "p99": 17.763281310708145},
}

# ---------------------------------------------------------------------------
# Score variant 목록
# ---------------------------------------------------------------------------
SCORE_VARIANTS = [
    "score_original",
    "score_roi_weighted",
    "score_valid950_weighted",
    "score_valid970_weighted",
    "score_valid950_pow025",
    "score_valid950_floor025",
    "score_valid950_soft",
]

# ---------------------------------------------------------------------------
# Weak patient 목록
# ---------------------------------------------------------------------------
WEAK_PATIENTS = ["LUNG1-156", "LUNG1-415", "MSD_lung_071", "MSD_lung_096", "MSD_lung_079"]

# ---------------------------------------------------------------------------
# Sample 환자 목록 (seed=42, weak 5명 + NSCLC 5명 + MSD 5명 = 15명)
# ---------------------------------------------------------------------------
_SAMPLE_NSCLC = ["LUNG1-194", "LUNG1-252", "LUNG1-270", "LUNG1-278", "LUNG1-321"]
_SAMPLE_MSD = ["MSD_lung_003", "MSD_lung_034", "MSD_lung_037", "MSD_lung_054", "MSD_lung_069"]
SAMPLE_PATIENTS = WEAK_PATIENTS + _SAMPLE_NSCLC + _SAMPLE_MSD  # 15명

# ---------------------------------------------------------------------------
# 출력 파일명 결정
# ---------------------------------------------------------------------------
def get_out_files(is_sample: bool) -> dict[str, Path]:
    prefix = "sample" if is_sample else "full"
    return {
        "csv": OUT_DIR / f"ratio_adjusted_score_{prefix}_diagnostic.csv",
        "json": OUT_DIR / f"ratio_adjusted_score_{prefix}_diagnostic.json",
        "md": OUT_DIR / f"ratio_adjusted_score_{prefix}_diagnostic.md",
        "strat": OUT_DIR / f"ratio_adjusted_score_{prefix}_stratified_summary.csv",
        "weak": OUT_DIR / f"ratio_adjusted_score_{prefix}_weak_patient_summary.csv",
    }


def check_output_conflicts(out_files: dict[str, Path]) -> None:
    for key, f in out_files.items():
        if f.exists():
            print(f"[중단] 이미 존재: {f}")
            print("  기존 파일을 삭제하거나 이름을 바꾼 후 재실행하세요.")
            sys.exit(1)


# ---------------------------------------------------------------------------
# Step 1: score CSV 로드 (sample 모드에서는 sample 환자 파일만 읽음)
# ---------------------------------------------------------------------------
def load_scores(score_dir: Path, model_type: str, patient_ids: list[str] | None = None) -> pd.DataFrame:
    if patient_ids is not None:
        csv_files = []
        for pid in patient_ids:
            f = score_dir / f"{pid}.csv"
            if f.exists():
                csv_files.append(f)
            else:
                print(f"  [경고] CSV 없음: {f.name}")
        print(f"[{model_type}] sample CSV 파일 수: {len(csv_files)}")
    else:
        csv_files = sorted(score_dir.glob("*.csv"))
        print(f"[{model_type}] CSV 파일 수: {len(csv_files)}")

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
            dfs.append(df)
        except Exception as e:
            print(f"  [오류] {f.name}: {e}")

    if not dfs:
        print(f"[{model_type}] 로드된 CSV 없음 — 중단")
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True)
    combined["model_type"] = model_type
    print(f"[{model_type}] 전체 row: {len(combined):,}")
    return combined


# ---------------------------------------------------------------------------
# Step 2: npy 폴더 매핑 빌드
# ---------------------------------------------------------------------------
def build_npy_mapping(npy_base: Path) -> dict[str, Path]:
    """safe_id → npy 폴더 경로 매핑"""
    return {folder.name: folder for folder in npy_base.iterdir() if folder.is_dir()}


# ---------------------------------------------------------------------------
# Step 2-2: 환자별 ratio 계산
# ---------------------------------------------------------------------------
def compute_patient_ratios(df_patient: pd.DataFrame, npy_folder: Path) -> pd.DataFrame:
    ct_path = npy_folder / "ct_hu.npy"
    roi_path = npy_folder / "roi_0_0.npy"

    ct_hu = np.load(ct_path, mmap_mode="r")
    roi_mask = np.load(roi_path, mmap_mode="r")

    n_slices, H, W = ct_hu.shape

    n = len(df_patient)
    roi_inside_ratios = np.zeros(n, dtype=np.float32)
    air_ratio_950 = np.zeros(n, dtype=np.float32)
    air_ratio_970 = np.zeros(n, dtype=np.float32)
    valid_roi_air950 = np.zeros(n, dtype=np.float32)
    valid_roi_air970 = np.zeros(n, dtype=np.float32)
    soft_tissue_ratio = np.zeros(n, dtype=np.float32)

    rows = df_patient[["local_z", "y0", "x0", "y1", "x1"]].values

    for i, (local_z, y0, x0, y1, x1) in enumerate(rows):
        local_z = int(local_z)
        y0, x0, y1, x1 = int(y0), int(x0), int(y1), int(x1)

        if local_z < 0 or local_z >= n_slices:
            continue

        y0c, y1c = max(0, y0), min(H, y1)
        x0c, x1c = max(0, x0), min(W, x1)
        n_pixels = (y1 - y0) * (x1 - x0)
        if n_pixels <= 0:
            continue

        ct_patch = ct_hu[local_z, y0c:y1c, x0c:x1c].astype(np.float32)
        roi_patch = roi_mask[local_z, y0c:y1c, x0c:x1c]

        roi_inside_ratios[i] = roi_patch.sum() / n_pixels
        air_ratio_950[i] = (ct_patch < -950).sum() / n_pixels
        air_ratio_970[i] = (ct_patch < -970).sum() / n_pixels
        valid_roi_air950[i] = ((roi_patch > 0) & (ct_patch >= -950)).sum() / n_pixels
        valid_roi_air970[i] = ((roi_patch > 0) & (ct_patch >= -970)).sum() / n_pixels
        soft_tissue_ratio[i] = (ct_patch > -300).sum() / n_pixels

    result = df_patient.copy()
    result["roi_inside_ratio"] = roi_inside_ratios
    result["air_ratio_950"] = air_ratio_950
    result["air_ratio_970"] = air_ratio_970
    result["valid_ratio_roi_air950"] = valid_roi_air950
    result["valid_ratio_roi_air970"] = valid_roi_air970
    result["soft_tissue_ratio"] = soft_tissue_ratio
    return result


# ---------------------------------------------------------------------------
# Step 2-3: 전체 환자 순차 처리
# ---------------------------------------------------------------------------
def compute_all_ratios(df: pd.DataFrame, npy_mapping: dict[str, Path]) -> pd.DataFrame:
    safe_ids = df["safe_id"].unique()
    total = len(safe_ids)
    results = []
    error_patients = []

    for idx, safe_id in enumerate(safe_ids):
        npy_folder = npy_mapping.get(safe_id)
        if npy_folder is None:
            print(f"  [경고] npy 폴더 없음: {safe_id}")
            sub = df[df["safe_id"] == safe_id].copy()
            for col in ["roi_inside_ratio", "air_ratio_950", "air_ratio_970",
                        "valid_ratio_roi_air950", "valid_ratio_roi_air970", "soft_tissue_ratio"]:
                sub[col] = np.nan
            results.append(sub)
            continue

        if idx % 5 == 0 or idx == total - 1:
            print(f"  ratio 계산 중... {idx+1}/{total} ({safe_id})")

        try:
            sub = df[df["safe_id"] == safe_id]
            processed = compute_patient_ratios(sub, npy_folder)
            results.append(processed)
        except Exception as e:
            print(f"  [오류] {safe_id}: {e}")
            traceback.print_exc()
            error_patients.append(safe_id)
            sub = df[df["safe_id"] == safe_id].copy()
            for col in ["roi_inside_ratio", "air_ratio_950", "air_ratio_970",
                        "valid_ratio_roi_air950", "valid_ratio_roi_air970", "soft_tissue_ratio"]:
                sub[col] = np.nan
            results.append(sub)
        finally:
            gc.collect()

    if error_patients:
        print(f"\n[오류 환자 목록] ({len(error_patients)}개)")
        for p in error_patients:
            print(f"  - {p}")

    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# Step 3: score variant 계산
# ---------------------------------------------------------------------------
def compute_score_variants(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    v950 = df["valid_ratio_roi_air950"].clip(0, 1)
    df["score_original"] = df["padim_score"]
    df["score_roi_weighted"] = df["padim_score"] * np.sqrt(df["roi_inside_ratio"].clip(0, 1))
    df["score_valid950_weighted"] = df["padim_score"] * np.sqrt(v950)
    df["score_valid970_weighted"] = df["padim_score"] * np.sqrt(df["valid_ratio_roi_air970"].clip(0, 1))
    df["score_valid950_pow025"] = df["padim_score"] * (v950 ** 0.25)
    df["score_valid950_floor025"] = df["padim_score"] * np.sqrt(v950.clip(lower=0.25))
    df["score_valid950_soft"] = df["padim_score"] * (0.7 + 0.3 * np.sqrt(v950))
    return df


# ---------------------------------------------------------------------------
# Step 4: 평가 지표 계산
# ---------------------------------------------------------------------------
def is_lesion_patch(df: pd.DataFrame) -> pd.Series:
    if "patch_label" in df.columns:
        return df["patch_label"] == 1
    return df["lesion_patch_ratio"] > 0


def compute_patient_topk_metrics(df: pd.DataFrame, score_col: str, ks: list[int]) -> dict:
    """환자별 top-k 기준 hit rate 계산"""
    patient_ids = df["patient_id"].unique()
    hit_counts = {k: 0 for k in ks}
    best_ranks = []
    best_percentiles = []
    n_patients_with_lesion = 0

    for pid in patient_ids:
        sub_p = df[df["patient_id"] == pid]
        lesion_flag = is_lesion_patch(sub_p)
        if lesion_flag.sum() == 0:
            continue
        n_patients_with_lesion += 1
        n_total_p = len(sub_p)
        ranked = sub_p[score_col].rank(ascending=False, method="first")
        lesion_ranks = ranked[lesion_flag]
        best_rank = int(lesion_ranks.min())
        best_pct = float(best_rank / n_total_p * 100)
        best_ranks.append(best_rank)
        best_percentiles.append(best_pct)
        for k in ks:
            if best_rank <= k:
                hit_counts[k] += 1

    result = {}
    for k in ks:
        result[f"patient_top{k}_hit_rate"] = (
            float(hit_counts[k] / n_patients_with_lesion) if n_patients_with_lesion > 0 else 0.0
        )
    result["patient_lesion_best_rank_mean"] = float(np.mean(best_ranks)) if best_ranks else 0.0
    result["patient_lesion_best_percentile_mean"] = float(np.mean(best_percentiles)) if best_percentiles else 0.0
    return result


def compute_metrics(df: pd.DataFrame, score_col: str, threshold: float,
                    threshold_type: str, n_topn: int | None = None) -> dict:
    lesion_flag = is_lesion_patch(df)

    if n_topn is None:
        selected = df[score_col] >= threshold
    else:
        cutoff = df[score_col].nlargest(n_topn).min() if n_topn > 0 else df[score_col].max() + 1
        selected = df[score_col] >= cutoff

    n_total = len(df)
    n_selected = selected.sum()
    n_lesion_total = lesion_flag.sum()
    n_selected_lesion = (selected & lesion_flag).sum()
    n_fp = n_selected - n_selected_lesion

    lesion_patch_recall = n_selected_lesion / n_lesion_total if n_lesion_total > 0 else 0.0

    # lesion slice recall
    lesion_slices = set(df[lesion_flag][["patient_id", "local_z"]].apply(tuple, axis=1))
    selected_slices = set(df[selected][["patient_id", "local_z"]].apply(tuple, axis=1))
    hit_lesion_slices = lesion_slices & selected_slices
    lesion_slice_recall = len(hit_lesion_slices) / len(lesion_slices) if lesion_slices else 0.0

    # patient hit rate
    lesion_patients = df[lesion_flag]["patient_id"].unique()
    hit_patients = df[selected & lesion_flag]["patient_id"].unique()
    no_hit_patients = [p for p in lesion_patients if p not in hit_patients]
    patient_hit_rate = len(hit_patients) / len(lesion_patients) if len(lesion_patients) > 0 else 0.0

    # global top-k (해석 제한 주의: 전체 patch 전역 rank 기준)
    df_sorted = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    df_sorted["_rank"] = df_sorted.index + 1
    lesion_ranks_global = df_sorted[is_lesion_patch(df_sorted)]["_rank"]
    global_top10 = int((lesion_ranks_global <= 10).sum())
    global_top30 = int((lesion_ranks_global <= 30).sum())
    global_top50 = int((lesion_ranks_global <= 50).sum())

    # lesion best rank/percentile (global)
    if len(lesion_ranks_global) > 0:
        lesion_best_rank = int(lesion_ranks_global.min())
        lesion_best_percentile = float(lesion_best_rank / n_total * 100)
    else:
        lesion_best_rank = -1
        lesion_best_percentile = -1.0

    # 환자별 top-k
    patient_topk = compute_patient_topk_metrics(df, score_col, ks=[10, 30, 50, 100, 500])

    # patient별 selected 수 통계
    per_patient_sel = df[selected].groupby("patient_id").size()
    sel_stats = {
        "min": int(per_patient_sel.min()) if len(per_patient_sel) > 0 else 0,
        "median": float(per_patient_sel.median()) if len(per_patient_sel) > 0 else 0.0,
        "mean": float(per_patient_sel.mean()) if len(per_patient_sel) > 0 else 0.0,
        "max": int(per_patient_sel.max()) if len(per_patient_sel) > 0 else 0,
    }

    return {
        "score_col": score_col,
        "threshold_type": threshold_type,
        "threshold": threshold if n_topn is None else None,
        "n_topn": n_topn,
        "n_total": n_total,
        "n_selected": int(n_selected),
        "selected_ratio": float(n_selected / n_total) if n_total > 0 else 0.0,
        "n_lesion_total": int(n_lesion_total),
        "n_selected_lesion": int(n_selected_lesion),
        "n_fp": int(n_fp),
        "positive_ratio": float(n_selected_lesion / n_selected) if n_selected > 0 else 0.0,
        "fp_ratio": float(n_fp / n_selected) if n_selected > 0 else 0.0,
        "lesion_patch_recall": float(lesion_patch_recall),
        "lesion_slice_recall": float(lesion_slice_recall),
        "patient_hit_rate": float(patient_hit_rate),
        "n_nohit_patients": len(no_hit_patients),
        "nohit_patients": no_hit_patients,
        "global_top10_note": "전체 patch 전역 rank 기준. 해석 제한.",
        "global_top10_lesion": global_top10,
        "global_top30_lesion": global_top30,
        "global_top50_lesion": global_top50,
        "lesion_best_rank_global": lesion_best_rank,
        "lesion_best_percentile_global": float(lesion_best_percentile),
        **patient_topk,
        "per_patient_selected_stats": sel_stats,
    }


def run_all_metrics(df: pd.DataFrame) -> list[dict]:
    records = []
    for model_type in df["model_type"].unique():
        sub = df[df["model_type"] == model_type].copy()
        thr = THRESHOLDS[model_type]

        orig_p95_sel = int((sub["score_original"] >= thr["p95"]).sum())
        orig_p99_sel = int((sub["score_original"] >= thr["p99"]).sum())

        for sv in SCORE_VARIANTS:
            for thr_type, thr_val in [("p95", thr["p95"]), ("p99", thr["p99"])]:
                m = compute_metrics(sub, sv, thr_val, thr_type)
                m["model_type"] = model_type
                records.append(m)

            for thr_type, n_sel in [("topN_p95", orig_p95_sel), ("topN_p99", orig_p99_sel)]:
                m = compute_metrics(sub, sv, 0.0, thr_type, n_topn=n_sel)
                m["model_type"] = model_type
                records.append(m)

    return records


# ---------------------------------------------------------------------------
# Step 5: stratified 분석
# ---------------------------------------------------------------------------
def load_patient_lesion_size(model_type: str) -> pd.DataFrame | None:
    """per_patient_screening.csv에서 patient-level lesion_patch_total 로드"""
    csv_path = PER_PATIENT_SCREEN_V1V2 if model_type == "v1v2" else PER_PATIENT_SCREEN_V2V2
    if not csv_path.exists():
        print(f"  [경고] per_patient_screening 없음: {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if "lesion_patch_total" not in df.columns:
            print(f"  [경고] lesion_patch_total 컬럼 없음: {csv_path}")
            return None
        return df[["patient_id", "lesion_patch_total"]].copy()
    except Exception as e:
        print(f"  [경고] per_patient_screening 로드 오류: {e}")
        return None


def assign_patient_lesion_size_bin(df: pd.DataFrame, model_type: str) -> pd.DataFrame:
    """
    patient-level lesion_size_bin을 per_patient_screening.csv 기반으로 추가.
    불가능하면 patch-level bin 제한으로 명시.
    """
    screen_df = load_patient_lesion_size(model_type)
    if screen_df is None:
        df["patient_lesion_size_bin"] = "unavailable"
        return df

    # patient-level lesion_patch_total 기준 bin
    p25 = screen_df["lesion_patch_total"].quantile(0.25)
    p75 = screen_df["lesion_patch_total"].quantile(0.75)
    median = screen_df["lesion_patch_total"].median()

    def assign_bin(val):
        if val <= p25:
            return "patient_tiny"
        elif val <= median:
            return "patient_small"
        elif val <= p75:
            return "patient_medium"
        else:
            return "patient_large"

    screen_df["patient_lesion_size_bin"] = screen_df["lesion_patch_total"].apply(assign_bin)
    df = df.merge(screen_df[["patient_id", "patient_lesion_size_bin"]], on="patient_id", how="left")
    df["patient_lesion_size_bin"] = df["patient_lesion_size_bin"].fillna("no_lesion_info")
    return df


def assign_patch_lesion_size_bin(df: pd.DataFrame) -> pd.Series:
    """patch-level lesion occupancy bin (patient-level 병변 크기가 아님)"""
    ratio = df["lesion_patch_ratio"]
    bins = pd.Series("non-lesion", index=df.index)
    bins[ratio > 0] = "patch_tiny"
    bins[ratio >= 0.1] = "patch_small"
    bins[ratio >= 0.3] = "patch_medium"
    bins[ratio >= 0.6] = "patch_large"
    return bins


def assign_air_bin(series: pd.Series, label: str = "air") -> pd.Series:
    bins = pd.cut(series, bins=[-0.001, 0.3, 0.7, 1.001],
                  labels=[f"{label}_low", f"{label}_mid", f"{label}_high"])
    return bins.astype(str)


def assign_valid_bin(series: pd.Series) -> pd.Series:
    bins = pd.cut(series, bins=[-0.001, 0.05, 0.3, 1.001],
                  labels=["valid_low", "valid_mid", "valid_high"])
    return bins.astype(str)


def simple_group_metrics(sub: pd.DataFrame, score_col: str, threshold: float) -> dict:
    lesion_flag = is_lesion_patch(sub)
    selected = sub[score_col] >= threshold
    n_total = len(sub)
    n_selected = selected.sum()
    n_lesion = lesion_flag.sum()
    n_sel_lesion = (selected & lesion_flag).sum()
    n_fp = n_selected - n_sel_lesion
    return {
        "n_total": n_total,
        "n_selected": int(n_selected),
        "n_lesion": int(n_lesion),
        "n_selected_lesion": int(n_sel_lesion),
        "n_fp": int(n_fp),
        "selected_ratio": float(n_selected / n_total) if n_total > 0 else 0.0,
        "positive_ratio": float(n_sel_lesion / n_selected) if n_selected > 0 else 0.0,
        "fp_ratio": float(n_fp / n_selected) if n_selected > 0 else 0.0,
        "lesion_recall": float(n_sel_lesion / n_lesion) if n_lesion > 0 else 0.0,
        "score_mean": float(sub[score_col].mean()),
        "score_median": float(sub[score_col].median()),
        "score_p95": float(sub[score_col].quantile(0.95)),
        "score_p99": float(sub[score_col].quantile(0.99)),
    }


def run_stratified_analysis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["patch_lesion_size_bin"] = assign_patch_lesion_size_bin(df)
    df["air_bin_950"] = assign_air_bin(df["air_ratio_950"], "air950")
    df["valid_bin_950"] = assign_valid_bin(df["valid_ratio_roi_air950"])
    df["vessel_like"] = (df["soft_tissue_ratio"] > 0.3).map(
        {True: "vessel_like_high", False: "vessel_like_low"}
    )

    # patient-level lesion size bin (model_type별 join)
    df_parts = []
    for model_type in df["model_type"].unique():
        sub = df[df["model_type"] == model_type].copy()
        sub = assign_patient_lesion_size_bin(sub, model_type)
        df_parts.append(sub)
    df = pd.concat(df_parts, ignore_index=True)

    # near_wall: 미구현 → position_bin/central_peripheral proxy 사용
    # boundary distance 계산이 없으므로 near_wall 분석은 이번 버전에서 제외.

    strat_cols = {
        "patch_lesion_size_bin": "patch-level lesion occupancy (patient-level 병변 크기 아님)",
        "patient_lesion_size_bin": "patient-level 병변 크기 (per_patient_screening join 기준)",
        "position_bin": "폐 위치 (near_wall proxy 포함)",
        "z_level": "z 위치",
        "central_peripheral": "central/peripheral (near_wall proxy)",
        "air_bin_950": "공기층 비율 (-950 기준)",
        "valid_bin_950": "유효 조직 비율",
        "vessel_like": "고음영 구조물 proxy (혈관 확정 아님 - 혈관 segmentation 없음)",
    }

    records = []
    for model_type in df["model_type"].unique():
        sub_model = df[df["model_type"] == model_type]
        thr = THRESHOLDS[model_type]

        for strat_col, strat_label in strat_cols.items():
            if strat_col not in sub_model.columns:
                continue
            for group_val in sorted(sub_model[strat_col].dropna().unique()):
                sub_group = sub_model[sub_model[strat_col] == group_val]
                for sv in SCORE_VARIANTS:
                    for thr_type, thr_val in [("p95", thr["p95"]), ("p99", thr["p99"])]:
                        m = simple_group_metrics(sub_group, sv, thr_val)
                        row = {
                            "model_type": model_type,
                            "strat_col": strat_col,
                            "strat_label": strat_label,
                            "group": str(group_val),
                            "score_col": sv,
                            "threshold_type": thr_type,
                        }
                        row.update(m)
                        records.append(row)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Step 6: weak patient 분석
# ---------------------------------------------------------------------------
def run_weak_patient_analysis(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for model_type in df["model_type"].unique():
        sub_model = df[df["model_type"] == model_type]
        thr = THRESHOLDS[model_type]

        for pid in WEAK_PATIENTS:
            sub_p = sub_model[sub_model["patient_id"] == pid]
            if sub_p.empty:
                print(f"  [경고] weak patient 없음: {pid} (model={model_type})")
                continue

            lesion_flag = is_lesion_patch(sub_p)
            sub_lesion = sub_p[lesion_flag]
            n_total = len(sub_p)

            # original baseline 계산
            orig_ranked = sub_p.sort_values("score_original", ascending=False).reset_index(drop=True)
            orig_ranked["_rank"] = orig_ranked.index + 1
            orig_lesion_ranks = orig_ranked[is_lesion_patch(orig_ranked)]["_rank"]
            orig_best_rank = int(orig_lesion_ranks.min()) if len(orig_lesion_ranks) > 0 else None
            orig_lesion_sel_p95 = int(((sub_p["score_original"] >= thr["p95"]) & lesion_flag).sum())

            for sv in SCORE_VARIANTS:
                ranked = sub_p.sort_values(sv, ascending=False).reset_index(drop=True)
                ranked["_rank"] = ranked.index + 1
                lesion_ranks = ranked[is_lesion_patch(ranked)]["_rank"]

                adj_lesion_sel_p95 = int(((sub_p[sv] >= thr["p95"]) & lesion_flag).sum())

                row = {
                    "model_type": model_type,
                    "patient_id": pid,
                    "score_col": sv,
                    "n_patch_total": n_total,
                    "n_lesion_patch": int(lesion_flag.sum()),
                    "lesion_max_score": float(sub_lesion[sv].max()) if not sub_lesion.empty else None,
                    "lesion_best_rank": int(lesion_ranks.min()) if len(lesion_ranks) > 0 else None,
                    "lesion_best_percentile": float(lesion_ranks.min() / n_total * 100) if len(lesion_ranks) > 0 else None,
                    "selected_p95": int((sub_p[sv] >= thr["p95"]).sum()),
                    "selected_p99": int((sub_p[sv] >= thr["p99"]).sum()),
                    "lesion_sel_p95": adj_lesion_sel_p95,
                    "lesion_sel_p99": int(((sub_p[sv] >= thr["p99"]) & lesion_flag).sum()),
                    "lesion_in_top500": int((lesion_ranks <= 500).sum()) if len(lesion_ranks) > 0 else 0,
                    "lesion_in_top1000": int((lesion_ranks <= 1000).sum()) if len(lesion_ranks) > 0 else 0,
                    "orig_best_rank": orig_best_rank,
                    "rank_change": (
                        int(lesion_ranks.min()) - orig_best_rank
                        if (len(lesion_ranks) > 0 and orig_best_rank is not None)
                        else None
                    ),
                    "lesion_recovered": (orig_lesion_sel_p95 == 0 and adj_lesion_sel_p95 > 0),
                    "lesion_lost": (orig_lesion_sel_p95 > 0 and adj_lesion_sel_p95 == 0),
                }
                records.append(row)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Step 7: union/intersection 분석
# ---------------------------------------------------------------------------
def run_union_intersection(df: pd.DataFrame) -> dict:
    results = {}
    for model_type in df["model_type"].unique():
        sub = df[df["model_type"] == model_type]
        thr = THRESHOLDS[model_type]
        results[model_type] = {}

        for thr_type, thr_val in [("p95", thr["p95"]), ("p99", thr["p99"])]:
            orig_sel = sub["score_original"] >= thr_val
            lesion_flag = is_lesion_patch(sub)
            entry = {"threshold_type": thr_type, "threshold": thr_val}

            for sv in SCORE_VARIANTS:
                if sv == "score_original":
                    continue
                adj_sel = sub[sv] >= thr_val
                union_sel = orig_sel | adj_sel
                inter_sel = orig_sel & adj_sel
                down_sel = orig_sel & ~adj_sel
                up_sel = ~orig_sel & adj_sel

                n_lesion = lesion_flag.sum()
                entry[sv] = {
                    "orig_selected": int(orig_sel.sum()),
                    "adj_selected": int(adj_sel.sum()),
                    "union_selected": int(union_sel.sum()),
                    "inter_selected": int(inter_sel.sum()),
                    "down_selected": int(down_sel.sum()),
                    "up_selected": int(up_sel.sum()),
                    "orig_lesion_recall": float((orig_sel & lesion_flag).sum() / n_lesion) if n_lesion > 0 else 0.0,
                    "adj_lesion_recall": float((adj_sel & lesion_flag).sum() / n_lesion) if n_lesion > 0 else 0.0,
                    "union_lesion_recall": float((union_sel & lesion_flag).sum() / n_lesion) if n_lesion > 0 else 0.0,
                    "inter_lesion_recall": float((inter_sel & lesion_flag).sum() / n_lesion) if n_lesion > 0 else 0.0,
                    "down_lesion_count": int((down_sel & lesion_flag).sum()),
                    "up_lesion_count": int((up_sel & lesion_flag).sum()),
                }

            results[model_type][thr_type] = entry

    return results


# ---------------------------------------------------------------------------
# Step 8: 보고서 MD 생성
# ---------------------------------------------------------------------------
INTERPRETATION_NOTES = """
## 해석 주의사항

- ratio-adjusted score는 ROI 밖 영역과 확실한 공기층이 많은 patch를 낮추는 신뢰도 보정 점수이다.
- 작은 병변과 큰 병변의 크기 차이를 직접 보정하는 점수가 아니므로, original 후보와 adjusted 후보의 union 방식으로 병변 누락 위험을 줄인다.
- pixel-level score map이 없으면 patch 내부의 병변 중심부만 따로 재계산할 수 없으므로, 이번 분석은 patch-level 후처리 보정으로 제한한다.
- 혈관 segmentation이 없는 경우 vessel_like 결과는 혈관 확정이 아니라 고음영 구조물 proxy로만 해석한다.
- **global top-k (top10/30/50)는 전체 patch 전역 rank 기준이므로 해석 제한.** 전체 587만+ patch 중 상위 k개라서 강한 기준. 환자별 top-k (patient_topk_hit_rate)를 함께 확인할 것.
- **lesion_size_bin (patch-level)**: patch 안 lesion 점유 비율 기준이며, patient-level 병변 크기가 아님. patient-level 분석은 patient_lesion_size_bin (per_patient_screening join) 사용.
- **near_wall 분석**: boundary distance 계산 미구현. position_bin과 central_peripheral을 proxy로 사용함.
"""


def build_md_report(metrics_records: list[dict], union_results: dict,
                    df: pd.DataFrame, is_sample: bool) -> str:
    mode_tag = "[SAMPLE 실행]" if is_sample else "[FULL 실행]"
    lines = [f"# Ratio-Adjusted Score 전체 진단 보고서 {mode_tag}\n"]
    if is_sample:
        lines.append(f"- **sample 환자 수**: {df['patient_id'].nunique()}명")
        lines.append(f"- **sample 환자 목록**: {sorted(df['patient_id'].unique())}\n")
    lines.append(INTERPRETATION_NOTES)

    lines.append("\n## 데이터 개요\n")
    for model_type in df["model_type"].unique():
        sub = df[df["model_type"] == model_type]
        lesion_cnt = is_lesion_patch(sub).sum()
        lines.append(f"- **{model_type}**: 전체 {len(sub):,} patch, lesion {lesion_cnt:,} patch")

    lines.append("\n## Score Variant별 평가 결과 (threshold 기준)\n")
    lines.append("| model | score_col | threshold | n_selected | lesion_recall | slice_recall | hit_rate | fp_ratio | pat_top50_hit |")
    lines.append("|-------|-----------|-----------|------------|---------------|--------------|----------|----------|----------------|")

    for r in metrics_records:
        if r.get("n_topn") is not None:
            continue
        lines.append(
            f"| {r['model_type']} | {r['score_col']} | {r['threshold_type']} "
            f"| {r['n_selected']:,} | {r['lesion_patch_recall']:.4f} "
            f"| {r['lesion_slice_recall']:.4f} | {r['patient_hit_rate']:.4f} "
            f"| {r['fp_ratio']:.4f} | {r.get('patient_top50_hit_rate', 0.0):.4f} |"
        )

    lines.append("\n## 환자별 Top-k Hit Rate (patient_topk_hit_rate)\n")
    lines.append("| model | score_col | threshold | top10 | top30 | top50 | top100 | top500 | best_rank_mean |")
    lines.append("|-------|-----------|-----------|-------|-------|-------|--------|--------|----------------|")
    for r in metrics_records:
        if r.get("n_topn") is not None:
            continue
        lines.append(
            f"| {r['model_type']} | {r['score_col']} | {r['threshold_type']} "
            f"| {r.get('patient_top10_hit_rate',0):.4f} | {r.get('patient_top30_hit_rate',0):.4f} "
            f"| {r.get('patient_top50_hit_rate',0):.4f} | {r.get('patient_top100_hit_rate',0):.4f} "
            f"| {r.get('patient_top500_hit_rate',0):.4f} | {r.get('patient_lesion_best_rank_mean',0):.1f} |"
        )

    lines.append("\n## Union/Intersection 비교\n")
    for model_type, thr_data in union_results.items():
        lines.append(f"\n### {model_type}")
        for thr_type, entry in thr_data.items():
            lines.append(f"\n#### {thr_type}")
            lines.append("| variant | orig_sel | adj_sel | union_sel | union_recall | adj_recall | down_lesion | up_lesion |")
            lines.append("|---------|----------|---------|-----------|--------------|------------|-------------|-----------|")
            for sv, v in entry.items():
                if sv in ("threshold_type", "threshold"):
                    continue
                lines.append(
                    f"| {sv} | {v['orig_selected']:,} | {v['adj_selected']:,} "
                    f"| {v['union_selected']:,} | {v['union_lesion_recall']:.4f} "
                    f"| {v['adj_lesion_recall']:.4f} | {v['down_lesion_count']} | {v['up_lesion_count']} |"
                )

    lines.append("\n## Weak Patient 확인 방법\n")
    lines.append(f"- 대상: {', '.join(WEAK_PATIENTS)}")
    lines.append("- rank_change < 0: 순위 개선 / > 0: 순위 하락")
    lines.append("- lesion_recovered=True: original에서 놓쳤으나 adjusted에서 살아남")
    lines.append("- lesion_lost=True: original에서 잡았으나 adjusted에서 누락")

    lines.append("\n## 판정 기준\n")
    lines.append("**효과 있음**: FP 비율 감소, patient hit rate 유지, lesion slice recall 유지, non-lesion high-score 후보 하락, union에서 hit 증가/유지")
    lines.append("\n**위험**: no-hit 증가, lesion slice recall 감소, tiny/small 병변 recall 감소, weak 환자 rank 악화, peripheral/near_wall 병변 하락")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ratio-adjusted score 분석. --sample 또는 --full-run 중 하나 필수."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", action="store_true",
                       help="Sample 15명만 처리 (weak 5 + NSCLC 5 + MSD 5)")
    group.add_argument("--full-run", dest="full_run", action="store_true",
                       help="전체 308명 처리 (별도 승인 후에만 사용)")
    return parser.parse_args()


def main():
    args = parse_args()
    is_sample = args.sample
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 출력 파일 충돌 확인
    out_files = get_out_files(is_sample)
    check_output_conflicts(out_files)

    # full 실행 시 full 파일 추가 보호
    if is_sample:
        full_files = get_out_files(False)
        for key, f in full_files.items():
            if f.exists():
                print(f"[경고] full 출력 파일 이미 존재 (sample 실행이므로 진행): {f}")

    # 처리 환자 목록
    patient_ids = SAMPLE_PATIENTS if is_sample else None
    print(f"\n실행 모드: {'SAMPLE (' + str(len(SAMPLE_PATIENTS)) + '명)' if is_sample else 'FULL (308명)'}")
    if is_sample:
        print(f"Sample 환자: {sorted(SAMPLE_PATIENTS)}")

    # Step 1: CSV 로드
    print("\n=== Step 1: Score CSV 로드 ===")
    df_v1v2 = load_scores(V1V2_SCORE_DIR, "v1v2", patient_ids)
    df_v2v2 = load_scores(V2V2_SCORE_DIR, "v2v2", patient_ids)
    df_all = pd.concat([df_v1v2, df_v2v2], ignore_index=True)
    del df_v1v2, df_v2v2
    gc.collect()
    print(f"전체 합계: {len(df_all):,} row")

    # Step 2: npy 매핑
    print("\n=== Step 2: npy 폴더 매핑 ===")
    npy_mapping = build_npy_mapping(NPY_BASE)
    print(f"npy 폴더 수: {len(npy_mapping)}")

    # Step 2-3: ratio 계산
    print("\n=== Step 2: ratio 계산 (환자별 순차) ===")
    df_all = compute_all_ratios(df_all, npy_mapping)
    gc.collect()

    # Step 3: score variant 계산
    print("\n=== Step 3: Score variant 계산 ===")
    df_all = compute_score_variants(df_all)

    # Step 4: 전체 평가 지표
    print("\n=== Step 4: 평가 지표 계산 ===")
    metrics_records = run_all_metrics(df_all)
    print(f"지표 조합 수: {len(metrics_records)}")

    # Step 5: stratified 분석
    print("\n=== Step 5: Stratified 분석 ===")
    df_strat = run_stratified_analysis(df_all)
    print(f"stratified 행 수: {len(df_strat):,}")

    # Step 6: weak patient 분석
    print("\n=== Step 6: Weak patient 분석 ===")
    df_weak = run_weak_patient_analysis(df_all)
    print(f"weak patient 행 수: {len(df_weak)}")

    # Step 7: union/intersection
    print("\n=== Step 7: Union/Intersection 분석 ===")
    union_results = run_union_intersection(df_all)

    # Step 8: 출력 파일 저장
    print("\n=== Step 8: 출력 파일 저장 ===")

    ratio_cols = ["roi_inside_ratio", "air_ratio_950", "air_ratio_970",
                  "valid_ratio_roi_air950", "valid_ratio_roi_air970", "soft_tissue_ratio"]

    # full_diagnostic.csv
    base_cols = list(df_all.columns[:df_all.columns.get_loc("padim_score") + 1])
    save_cols = [c for c in base_cols + ratio_cols + SCORE_VARIANTS + ["model_type"]
                 if c in df_all.columns]
    df_all[save_cols].to_csv(out_files["csv"], index=False)
    print(f"저장: {out_files['csv']}")

    # full_diagnostic.json
    def safe_json(obj):
        if isinstance(obj, dict):
            return {k: safe_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [safe_json(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj

    json_out = {
        "mode": "sample" if is_sample else "full",
        "sample_patients": SAMPLE_PATIENTS if is_sample else None,
        "metrics": safe_json(metrics_records),
        "union_intersection": safe_json(union_results),
        "interpretation": {
            "ratio_adjusted_score": "ROI 밖 영역과 확실한 공기층이 많은 patch를 낮추는 신뢰도 보정 점수",
            "lesion_size_correction": "작은 병변과 큰 병변의 크기 차이를 직접 보정하는 점수가 아님",
            "pixel_level_limit": "pixel-level score map 없음 - patch-level 후처리 보정으로 제한",
            "vessel_like_note": "vessel_like 결과는 혈관 확정이 아니라 고음영 구조물 proxy로만 해석",
            "global_topk_note": "global top-k는 전체 patch 전역 rank 기준으로 해석 제한. patient_topk_hit_rate 사용 권장.",
            "lesion_size_bin_note": "patch_lesion_size_bin은 patch-level lesion occupancy bin. patient-level 분석은 patient_lesion_size_bin 사용.",
            "near_wall_note": "boundary distance 계산 미구현. position_bin/central_peripheral을 proxy로 사용.",
        },
    }
    with open(out_files["json"], "w", encoding="utf-8") as f:
        json.dump(json_out, f, ensure_ascii=False, indent=2)
    print(f"저장: {out_files['json']}")

    # full_diagnostic.md
    md_text = build_md_report(metrics_records, union_results, df_all, is_sample)
    out_files["md"].write_text(md_text, encoding="utf-8")
    print(f"저장: {out_files['md']}")

    # stratified_summary.csv
    df_strat.to_csv(out_files["strat"], index=False)
    print(f"저장: {out_files['strat']}")

    # weak_patient_summary.csv
    df_weak.to_csv(out_files["weak"], index=False)
    print(f"저장: {out_files['weak']}")

    print(f"\n=== 완료 ({'SAMPLE' if is_sample else 'FULL'}) ===")
    print(f"출력 디렉토리: {OUT_DIR}")


if __name__ == "__main__":
    main()
