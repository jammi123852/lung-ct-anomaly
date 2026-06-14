"""
rule_g_rank_based_candidate_diagnostic.py

Rule G: label-free rank-based candidate scoring 후보 variant를 manifest-only로 비교하는 진단 스크립트.
crop/npz/PNG 생성 없음. summary CSV/JSON/MD만 출력.

실행:
  --run 없이: preflight 체크만 수행 후 종료
  --run: 실제 분석 실행

절대 금지:
- crop/npz/PNG 생성 금지
- 모델 학습 / scoring 재실행 금지
- 기존 score/candidate/evaluation/crop 파일 수정 금지
- stage2_holdout 환자 분석 금지
- weak 환자 전용 예외 로직 금지 (weak 환자 목록은 평가용만)
- lesion local_z 직접 후보 추가 금지
- 병변 mask를 후보 생성에 사용 금지
- pip/conda install 금지
- variant manifest 파일 저장 금지 (summary만)
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]

DIAG_CSV = REPO_ROOT / "outputs/position-aware-padim-v1/reports_v2_roi0_0_lesion/ratio_adjusted_score_full_diagnostic.csv"
STAGE_SPLIT_CSV = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1_balanced.csv"
RULE_F_JSON = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/reports/rule_f_candidate_diagnostic_summary.json"
RULE_F_CSV = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/reports/rule_f_candidate_diagnostic_summary.csv"

OUT_DIR = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/reports"
OUT_CSV = OUT_DIR / "rule_g_rank_based_candidate_summary.csv"
OUT_JSON = OUT_DIR / "rule_g_rank_based_candidate_summary.json"
OUT_MD = OUT_DIR / "rule_g_rank_based_candidate_summary.md"

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
V2V2_P95_THRESHOLD = 14.092057666455288
DEDUP_KEYS = ["patient_id", "local_z", "y0", "x0", "y1", "x1"]
WEAK_PATIENTS = ["LUNG1-156", "LUNG1-415", "MSD_lung_071", "MSD_lung_096", "MSD_lung_079"]
CHUNKSIZE = 200_000
CROP_FEASIBLE_THRESHOLD = 50_000

REQUIRED_COLS = [
    "patient_id", "local_z", "y0", "x0", "y1", "x1",
    "model_type",
    "score_original", "score_valid950_weighted", "score_valid950_pow025", "score_valid950_soft",
    "lesion_patch_ratio", "position_bin", "z_level", "central_peripheral",
]
OPTIONAL_COLS = [
    "roi_inside_ratio", "air_ratio_950", "air_ratio_970",
    "valid_ratio_roi_air950", "valid_ratio_roi_air970",
]

# ---------------------------------------------------------------------------
# Step 0: Guard 체크
# ---------------------------------------------------------------------------
def guard_check() -> None:
    for f in [OUT_CSV, OUT_JSON, OUT_MD]:
        if f.exists():
            print(f"[중단] 출력 파일 이미 존재: {f}")
            print("  기존 파일을 삭제하거나 이름을 바꾼 후 재실행하세요.")
            sys.exit(1)

    missing = []
    for f in [DIAG_CSV, STAGE_SPLIT_CSV, RULE_F_JSON, RULE_F_CSV]:
        if not f.exists():
            missing.append(str(f))
    if missing:
        print("[중단] 입력 파일 없음:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    print("[Guard] 모든 입력 파일 존재 확인 완료")

# ---------------------------------------------------------------------------
# Step 1: stage split 로드
# ---------------------------------------------------------------------------
def load_stage_split() -> tuple[set[str], set[str]]:
    df = pd.read_csv(STAGE_SPLIT_CSV, encoding="utf-8-sig")
    if "stage_split" not in df.columns:
        print(f"[중단] stage_split 컬럼 없음. 실제 컬럼: {list(df.columns)}")
        sys.exit(1)

    dev = set(df[df["stage_split"] == "stage1_dev"]["patient_id"].tolist())
    holdout = set(df[df["stage_split"] == "stage2_holdout"]["patient_id"].tolist())

    print(f"[Stage Split] stage1_dev: {len(dev)}명, stage2_holdout: {len(holdout)}명")
    if len(dev) != 154:
        print(f"  [경고] stage1_dev 기대 154명, 실제 {len(dev)}명")

    return dev, holdout

# ---------------------------------------------------------------------------
# Step 2: ratio_adjusted_score_full_diagnostic.csv chunk 로드
# ---------------------------------------------------------------------------
def load_diag_filtered(dev_patients: set[str], holdout_patients: set[str]) -> pd.DataFrame:
    print(f"\n[Step 2] {DIAG_CSV.name} chunk 로드 시작 (chunksize={CHUNKSIZE:,})")

    # 첫 chunk로 컬럼 확인
    first_chunk = pd.read_csv(DIAG_CSV, nrows=1, encoding="utf-8-sig")
    missing_cols = [c for c in REQUIRED_COLS if c not in first_chunk.columns]
    if missing_cols:
        print(f"[중단] 필수 컬럼 누락: {missing_cols}")
        sys.exit(1)
    print(f"  필수 컬럼 확인 완료")

    # 실제 존재하는 optional 컬럼만 선택
    actual_optional = [c for c in OPTIONAL_COLS if c in first_chunk.columns]
    load_cols = REQUIRED_COLS + actual_optional
    print(f"  로드할 optional 컬럼: {actual_optional}")

    chunks = []
    total_read = 0
    total_filtered = 0

    for i, chunk in enumerate(pd.read_csv(DIAG_CSV, chunksize=CHUNKSIZE, encoding="utf-8-sig", low_memory=False)):
        total_read += len(chunk)
        filtered = chunk[
            (chunk["model_type"] == "v2v2") &
            (chunk["patient_id"].isin(dev_patients))
        ]
        if len(filtered) > 0:
            avail_cols = [c for c in load_cols if c in filtered.columns]
            chunks.append(filtered[avail_cols])
            total_filtered += len(filtered)
        if (i + 1) % 10 == 0:
            print(f"  chunk {i+1} 처리 중... 읽은 행: {total_read:,}, 필터된 행: {total_filtered:,}")

    print(f"  전체 읽은 행: {total_read:,}, 필터 후: {total_filtered:,}")

    if not chunks:
        print("[중단] 필터 후 데이터 없음")
        sys.exit(1)

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    # 중복 제거
    before = len(df)
    df = df.drop_duplicates(subset=DEDUP_KEYS)
    after = len(df)
    print(f"  중복 제거: {before:,} → {after:,} (제거 {before - after:,})")

    # stage2_holdout 침투 확인
    holdout_in_data = set(df["patient_id"].unique()) & holdout_patients
    if holdout_in_data:
        print(f"[중단] stage2_holdout 환자가 데이터에 포함됨: {holdout_in_data}")
        sys.exit(1)
    print(f"  stage2_holdout 봉인 확인 완료")

    if after == 0:
        print("[중단] 중복 제거 후 0행")
        sys.exit(1)

    # 최종 로드된 optional 컬럼 목록 출력
    loaded_optional = [c for c in actual_optional if c in df.columns]
    print(f"  최종 로드된 optional 컬럼: {loaded_optional}")
    print(f"  환자 수: {df['patient_id'].nunique()}")
    print(f"  model_type 값: {df['model_type'].unique().tolist()}")

    return df

# ---------------------------------------------------------------------------
# Step 3: Rank score 계산
# ---------------------------------------------------------------------------
def compute_rank_scores(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Step 3] Rank score 계산 시작")

    def rank_percentile(series: pd.Series) -> pd.Series:
        r = series.rank(method="min", ascending=False)
        n = len(series)
        if n <= 1:
            return pd.Series(1.0, index=series.index)
        return 1.0 - (r - 1) / (n - 1)

    # patient_rank_original: 환자별 score_original 내림차순 percentile
    df["patient_rank_original"] = df.groupby("patient_id")["score_original"].transform(rank_percentile)

    # slice_rank_original: (patient_id, local_z) 단위 score_original 내림차순 percentile
    df["slice_rank_original"] = df.groupby(["patient_id", "local_z"])["score_original"].transform(rank_percentile)

    # patient_rank_valid950: 환자별 score_valid950_weighted 내림차순 percentile
    df["patient_rank_valid950"] = df.groupby("patient_id")["score_valid950_weighted"].transform(rank_percentile)

    # slice_rank_valid950: (patient_id, local_z) 단위 score_valid950_weighted 내림차순 percentile
    df["slice_rank_valid950"] = df.groupby(["patient_id", "local_z"])["score_valid950_weighted"].transform(rank_percentile)

    # patient_rank_soft: 환자별 score_valid950_soft 내림차순 percentile
    df["patient_rank_soft"] = df.groupby("patient_id")["score_valid950_soft"].transform(rank_percentile)

    # slice_rank_soft: (patient_id, local_z) 단위 score_valid950_soft 내림차순 percentile
    df["slice_rank_soft"] = df.groupby(["patient_id", "local_z"])["score_valid950_soft"].transform(rank_percentile)

    print("  rank 컬럼 6개 계산 완료")

    # Composite score 4개
    df["composite_rank_v1"] = (
        0.5 * df["patient_rank_original"]
        + 0.3 * df["slice_rank_original"]
        + 0.2 * df["patient_rank_valid950"]
    )
    df["composite_rank_v2"] = (
        0.4 * df["patient_rank_original"]
        + 0.3 * df["slice_rank_original"]
        + 0.2 * df["patient_rank_valid950"]
        + 0.1 * df["slice_rank_valid950"]
    )
    df["composite_rank_v3"] = df[[
        "patient_rank_original",
        "slice_rank_original",
        "patient_rank_valid950",
        "slice_rank_valid950",
    ]].max(axis=1)
    df["composite_rank_v4"] = (
        0.4 * df["patient_rank_original"]
        + 0.2 * df["slice_rank_original"]
        + 0.2 * df["patient_rank_valid950"]
        + 0.2 * df["patient_rank_soft"]
    )

    print("  composite rank 4개 계산 완료")
    return df

# ---------------------------------------------------------------------------
# Step 4: topN/topK 마스크 헬퍼 함수
# ---------------------------------------------------------------------------
def get_topn_mask(df: pd.DataFrame, score_col: str, n: int) -> pd.Series:
    if n <= 0:
        return pd.Series(False, index=df.index)
    cutoff = df[score_col].nlargest(n).min()
    return df[score_col] >= cutoff


def get_patient_topk_mask(df: pd.DataFrame, score_col: str, k: int) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for pid, sub in df.groupby("patient_id"):
        top_idx = sub[score_col].nlargest(min(k, len(sub))).index
        mask.loc[top_idx] = True
    return mask


def get_slice_topk_mask(df: pd.DataFrame, score_col: str, k: int) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for (pid, lz), sub in df.groupby(["patient_id", "local_z"]):
        top_idx = sub[score_col].nlargest(min(k, len(sub))).index
        mask.loc[top_idx] = True
    return mask

# ---------------------------------------------------------------------------
# Step 5: Variant 후보 생성
# ---------------------------------------------------------------------------
def build_variants(df: pd.DataFrame) -> dict[str, pd.Series]:
    g0_mask = df["score_original"] >= V2V2_P95_THRESHOLD
    n_baseline = int(g0_mask.sum())
    print(f"\n[Variants] G0 baseline 후보 수: {n_baseline:,}")

    variant_masks = {}

    # G0 baseline
    variant_masks["G0_original_p95"] = g0_mask

    # G1 ~ G6: G0 union topN 마스크
    variant_masks["G1_patient_rank_topN"] = g0_mask | get_topn_mask(df, "patient_rank_original", n_baseline)
    variant_masks["G2_slice_rank_topN"] = g0_mask | get_topn_mask(df, "slice_rank_original", n_baseline)
    variant_masks["G3_patient_slice_rank_topN"] = g0_mask | get_topn_mask(df, "composite_rank_v1", n_baseline)
    variant_masks["G4_patient_slice_valid950_rank_topN"] = g0_mask | get_topn_mask(df, "composite_rank_v2", n_baseline)
    variant_masks["G5_max_rank_topN"] = g0_mask | get_topn_mask(df, "composite_rank_v3", n_baseline)
    variant_masks["G6_soft_safe_rank_topN"] = g0_mask | get_topn_mask(df, "composite_rank_v4", n_baseline)

    # GP 계열: patient topK
    print("  GP 계열 patient topK 마스크 계산 중...")
    variant_masks["GP1_patient_top100"] = g0_mask | get_patient_topk_mask(df, "composite_rank_v2", 100)
    variant_masks["GP2_patient_top300"] = g0_mask | get_patient_topk_mask(df, "composite_rank_v2", 300)
    variant_masks["GP3_patient_top500"] = g0_mask | get_patient_topk_mask(df, "composite_rank_v2", 500)

    # GS 계열: slice topK (mask만 반환)
    print("  GS 계열 slice topK 마스크 계산 중...")
    variant_masks["GS1_slice_top10"] = g0_mask | get_slice_topk_mask(df, "composite_rank_v2", 10)
    variant_masks["GS2_slice_top30"] = g0_mask | get_slice_topk_mask(df, "composite_rank_v2", 30)
    variant_masks["GS3_slice_top50"] = g0_mask | get_slice_topk_mask(df, "composite_rank_v2", 50)

    for name, mask in variant_masks.items():
        print(f"  {name}: {int(mask.sum()):,}행 (mask만 저장)")

    return variant_masks

# ---------------------------------------------------------------------------
# Step 6: 평가 지표 계산
# ---------------------------------------------------------------------------
def is_lesion_series(df: pd.DataFrame) -> pd.Series:
    flag = pd.Series(False, index=df.index)
    if "lesion_patch_ratio" in df.columns:
        flag = flag | (df["lesion_patch_ratio"].fillna(0) > 0)
    if "patch_label" in df.columns:
        flag = flag | (df["patch_label"].fillna(0) == 1)
    if "lesion_overlap" in df.columns:
        flag = flag | df["lesion_overlap"].fillna(False).astype(bool)
    if "sampling_label" in df.columns:
        flag = flag | (df["sampling_label"].fillna("") == "positive")
    return flag


def compute_variant_metrics(
    df_variant: pd.DataFrame,
    df_full: pd.DataFrame,
    variant_name: str,
    f_comparison_data: dict,
) -> dict:
    # df_full 기준 lesion 집합 계산
    lesion_flag_full = df_full["lesion_patch_ratio"] > 0
    n_lesion_total = int(lesion_flag_full.sum())
    lesion_slices_full = set(
        map(tuple, df_full[lesion_flag_full][["patient_id", "local_z"]].values.tolist())
    )
    lesion_patients_full = set(df_full[lesion_flag_full]["patient_id"].unique())

    # df_variant 기준
    lesion_flag_v = is_lesion_series(df_variant)

    n_candidates = len(df_variant)
    n_patients = df_variant["patient_id"].nunique()

    per_patient = df_variant.groupby("patient_id").size()
    per_patient_stats = {
        "min": int(per_patient.min()) if len(per_patient) > 0 else 0,
        "median": float(per_patient.median()) if len(per_patient) > 0 else 0.0,
        "mean": float(per_patient.mean()) if len(per_patient) > 0 else 0.0,
        "max": int(per_patient.max()) if len(per_patient) > 0 else 0,
    }

    n_positive = int(lesion_flag_v.sum())
    n_fp = int((~lesion_flag_v).sum())
    positive_ratio = float(n_positive / n_candidates) if n_candidates > 0 else 0.0
    fp_ratio = float(n_fp / n_candidates) if n_candidates > 0 else 0.0

    lesion_patch_recall = float(n_positive / n_lesion_total) if n_lesion_total > 0 else 0.0

    # lesion slice recall
    v_slices = set(
        map(tuple, df_variant[lesion_flag_v][["patient_id", "local_z"]].values.tolist())
    )
    hit_lesion_slices = lesion_slices_full & v_slices
    lesion_slice_recall = float(len(hit_lesion_slices) / len(lesion_slices_full)) if lesion_slices_full else 0.0

    # patient hit rate
    hit_patients = set(df_variant[lesion_flag_v]["patient_id"].unique())
    nohit_patients = sorted(lesion_patients_full - hit_patients)
    patient_hit_rate = float(len(hit_patients) / len(lesion_patients_full)) if lesion_patients_full else 0.0

    # weak patients
    weak_hit = {}
    for wp in WEAK_PATIENTS:
        wp_lesion = df_full[(df_full["patient_id"] == wp) & (df_full["lesion_patch_ratio"] > 0)]
        wp_v = df_variant[df_variant["patient_id"] == wp]
        wp_lesion_hits = int(is_lesion_series(wp_v).sum())
        weak_hit[wp] = bool(wp_lesion_hits > 0) if len(wp_lesion) > 0 else None

    # patient lesion size group 기준 recall
    patient_lesion_counts = df_full[df_full["lesion_patch_ratio"] > 0].groupby("patient_id").size()
    if len(patient_lesion_counts) >= 4:
        q25 = patient_lesion_counts.quantile(0.25)
        q50 = patient_lesion_counts.quantile(0.50)
        q75 = patient_lesion_counts.quantile(0.75)

        def size_bin(val):
            if val <= q25:
                return "tiny"
            elif val <= q50:
                return "small"
            elif val <= q75:
                return "medium"
            else:
                return "large"
        patient_size_bin = patient_lesion_counts.apply(size_bin)
    else:
        patient_size_bin = pd.Series(dtype=str)

    size_recall = {}
    for bin_name in ["tiny", "small", "medium", "large"]:
        bin_patients = set(patient_size_bin[patient_size_bin == bin_name].index)
        bin_lesion = df_full[(df_full["patient_id"].isin(bin_patients)) & (df_full["lesion_patch_ratio"] > 0)]
        bin_v_sub = df_variant[df_variant["patient_id"].isin(bin_patients)]
        n_bin_lesion = len(bin_lesion)
        n_bin_hit = int(is_lesion_series(bin_v_sub).sum())
        size_recall[bin_name] = float(n_bin_hit / n_bin_lesion) if n_bin_lesion > 0 else None

    # central/peripheral recall
    cp_recall = {}
    for cp_val in ["central", "peripheral"]:
        sub_full = df_full[(df_full["central_peripheral"] == cp_val) & (df_full["lesion_patch_ratio"] > 0)]
        if "central_peripheral" in df_variant.columns:
            sub_v = df_variant[df_variant["central_peripheral"] == cp_val]
        else:
            sub_v = df_variant.iloc[0:0]
        n_total_cp = len(sub_full)
        n_hit_cp = int(is_lesion_series(sub_v).sum())
        cp_recall[cp_val] = float(n_hit_cp / n_total_cp) if n_total_cp > 0 else None

    # position_bin recall
    pos_recall = {}
    for pb in sorted(df_full["position_bin"].dropna().unique()):
        sub_full = df_full[(df_full["position_bin"] == pb) & (df_full["lesion_patch_ratio"] > 0)]
        if "position_bin" in df_variant.columns:
            sub_v = df_variant[df_variant["position_bin"] == pb]
        else:
            sub_v = df_variant.iloc[0:0]
        n_total_pb = len(sub_full)
        n_hit_pb = int(is_lesion_series(sub_v).sum())
        pos_recall[str(pb)] = float(n_hit_pb / n_total_pb) if n_total_pb > 0 else None

    # patient_topk_hit_rate 계산
    patient_topk_hit = {}
    for k in [10, 30, 50, 100, 500]:
        score_col = "composite_rank_v2" if "composite_rank_v2" in df_variant.columns else "score_original"
        hit_count = 0
        for pid, sub in df_variant.groupby("patient_id"):
            top_sub = sub.nlargest(min(k, len(sub)), score_col)
            if (top_sub["lesion_patch_ratio"].fillna(0) > 0).any():
                hit_count += 1
        patient_topk_hit[f"top{k}"] = float(hit_count / len(lesion_patients_full)) if lesion_patients_full else 0.0

    # F 대비 비교
    f5_n = f_comparison_data.get("f5_n", 0)
    f5_recall = f_comparison_data.get("f5_recall", 0.0)
    f1_recall = f_comparison_data.get("f1_recall", 0.0)
    f0_recall = f_comparison_data.get("f0_recall", 0.0)
    delta_n_vs_f5 = n_candidates - f5_n
    delta_recall_vs_f5 = lesion_patch_recall - f5_recall
    delta_recall_vs_f1 = lesion_patch_recall - f1_recall

    crop_feasible = bool(n_candidates < CROP_FEASIBLE_THRESHOLD)

    return {
        "variant": variant_name,
        "n_candidates": n_candidates,
        "n_patients": n_patients,
        "per_patient_stats": per_patient_stats,
        "n_positive": n_positive,
        "n_fp": n_fp,
        "positive_ratio": round(positive_ratio, 6),
        "fp_ratio": round(fp_ratio, 6),
        "lesion_patch_recall": round(lesion_patch_recall, 6),
        "lesion_slice_recall": round(lesion_slice_recall, 6),
        "patient_hit_rate": round(patient_hit_rate, 6),
        "n_nohit": len(nohit_patients),
        "nohit_patients": nohit_patients,
        "weak_patients_hit": weak_hit,
        "patient_lesion_size_recall": size_recall,
        "central_peripheral_recall": cp_recall,
        "position_bin_recall": pos_recall,
        "patient_topk_hit_rate": patient_topk_hit,
        "f_comparison": {
            "f5_n_candidates": f5_n,
            "f5_lesion_patch_recall": f5_recall,
            "f1_lesion_patch_recall": f1_recall,
            "f0_lesion_patch_recall": f0_recall,
            "delta_n_vs_f5": delta_n_vs_f5,
            "delta_recall_vs_f5": round(delta_recall_vs_f5, 6),
            "delta_recall_vs_f1": round(delta_recall_vs_f1, 6),
        },
        "crop_feasible": crop_feasible,
    }

# ---------------------------------------------------------------------------
# Step 7: Rule F 비교값 로드
# ---------------------------------------------------------------------------
def load_f_comparison() -> dict:
    with open(RULE_F_JSON, encoding="utf-8") as f:
        data = json.load(f)
    lookup = {m["variant"]: m for m in data}
    f0 = lookup.get("F0_original_p95", {})
    f1 = lookup.get("F1_original_p95_plus_valid950_topN", {})
    f5 = lookup.get("F5_F3_plus_D2_grid", {})
    return {
        "f0_n": f0.get("n_candidates", 0),
        "f0_recall": f0.get("lesion_patch_recall", 0.0),
        "f1_n": f1.get("n_candidates", 0),
        "f1_recall": f1.get("lesion_patch_recall", 0.0),
        "f5_n": f5.get("n_candidates", 0),
        "f5_recall": f5.get("lesion_patch_recall", 0.0),
    }

# ---------------------------------------------------------------------------
# Step 8: Preflight 보고
# ---------------------------------------------------------------------------
def preflight_report(dev_patients: set[str]) -> None:
    print("\n=== Preflight 보고 ===")
    diag_size_gb = DIAG_CSV.stat().st_size / 1e9
    print(f"입력 파일: {DIAG_CSV.name} ({diag_size_gb:.1f}GB)")
    print(f"stage1_dev 환자 수: {len(dev_patients)}명")

    # 예상 처리 행 수 (첫 chunk 샘플 비율로 추정)
    sample = next(pd.read_csv(DIAG_CSV, chunksize=CHUNKSIZE, encoding="utf-8-sig", low_memory=False))
    sample_filtered = sample[(sample["model_type"] == "v2v2") & (sample["patient_id"].isin(dev_patients))]
    sample_rate = len(sample_filtered) / len(sample) if len(sample) > 0 else 0
    # Rule F에서 확인된 전체 행 수 기준값
    total_rows_est = 11_749_900
    est_filtered = int(total_rows_est * sample_rate)
    est_chunks = int(total_rows_est / CHUNKSIZE) + 1
    est_mem_gb = est_filtered * 40 / 1e9

    print(f"예상 필터 후 행 수: ~{est_filtered:,}행 (샘플 비율 {sample_rate:.3f})")
    print(f"예상 청크 수: ~{est_chunks}개")
    print(f"예상 소요 시간: rank 계산 포함 10~30분 (groupby transform 포함)")
    print(f"메모리 위험: 중간")
    print(f"  - df 전체 + rank 컬럼은 메모리에 올라감. 예상 ~{est_mem_gb:.1f}GB")
    print(f"  - variant는 mask만 저장하므로 variant DataFrame 복사 위험 없음")
    print(f"\n출력 예정 파일:")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_JSON}")
    print(f"  {OUT_MD}")
    print(f"\n실행 명령 초안:")
    print(f"  /home/jinhy/ai_env/bin/python3 scripts/rule_g_rank_based_candidate_diagnostic.py --run")
    print(f"\n[Preflight 완료] 실제 실행은 --run 플래그 추가 후 진행하세요.")

# ---------------------------------------------------------------------------
# Step 9: 출력 저장
# ---------------------------------------------------------------------------
def save_outputs(all_metrics: list[dict], f_cmp: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # CSV (flat)
    flat_rows = []
    for m in all_metrics:
        fc = m.get("f_comparison", {})
        pp = m.get("per_patient_stats", {})
        topk = m.get("patient_topk_hit_rate", {})
        row = {
            "variant": m["variant"],
            "n_candidates": m["n_candidates"],
            "n_patients": m["n_patients"],
            "per_patient_min": pp.get("min"),
            "per_patient_median": pp.get("median"),
            "per_patient_mean": pp.get("mean"),
            "per_patient_max": pp.get("max"),
            "n_positive": m["n_positive"],
            "n_fp": m["n_fp"],
            "positive_ratio": m["positive_ratio"],
            "fp_ratio": m["fp_ratio"],
            "lesion_patch_recall": m["lesion_patch_recall"],
            "lesion_slice_recall": m["lesion_slice_recall"],
            "patient_hit_rate": m["patient_hit_rate"],
            "n_nohit": m["n_nohit"],
        }
        for wp in WEAK_PATIENTS:
            row[f"weak_hit_{wp}"] = m["weak_patients_hit"].get(wp)
        for size in ["tiny", "small", "medium", "large"]:
            row[f"size_recall_{size}"] = m["patient_lesion_size_recall"].get(size)
        for cp in ["central", "peripheral"]:
            row[f"cp_recall_{cp}"] = m["central_peripheral_recall"].get(cp)
        for k_label in ["top10", "top30", "top50", "top100", "top500"]:
            row[f"patient_topk_hit_{k_label}"] = topk.get(k_label)
        row["delta_n_vs_f5"] = fc.get("delta_n_vs_f5")
        row["delta_recall_vs_f5"] = fc.get("delta_recall_vs_f5")
        row["delta_recall_vs_f1"] = fc.get("delta_recall_vs_f1")
        row["crop_feasible"] = m["crop_feasible"]
        flat_rows.append(row)

    pd.DataFrame(flat_rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT_CSV}")

    # JSON safe 직렬화
    def safe_json(obj):
        if isinstance(obj, dict):
            return {k: safe_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [safe_json(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(safe_json(all_metrics), f, ensure_ascii=False, indent=2)
    print(f"저장: {OUT_JSON}")

    # MD
    f0_recall = f_cmp.get("f0_recall", 0.0)
    f1_recall = f_cmp.get("f1_recall", 0.0)
    f5_recall = f_cmp.get("f5_recall", 0.0)
    f0_n = f_cmp.get("f0_n", 0)
    f1_n = f_cmp.get("f1_n", 0)
    f5_n = f_cmp.get("f5_n", 0)

    lines = ["# Rule G 후보 Variant 진단 보고서\n"]

    lines.append("## 해석 주의사항\n")
    lines.append("- Rule G는 label-free rank 기반 설계이다. score의 절댓값이 아닌 환자 내 / slice 내 상대 순위를 사용한다.")
    lines.append("- 동일 데이터에서 rank를 계산하므로 G0 후보(threshold 기반)와 rank 기반 후보는 부분 중복이 발생한다.")
    lines.append("- F5와 recall 비교 시 주의: F5의 recall은 D2 label(lesion_overlap_ratio) 기반으로 보강된 값이다.")
    lines.append("  Rule G는 score_original + rank 보조만 사용하므로 F5와 직접 recall 비교는 방법론 차이가 있음을 인지해야 한다.")
    lines.append("- weak 환자(LUNG1-156, LUNG1-415, MSD_lung_071, MSD_lung_096, MSD_lung_079)는 평가 참고용이며 이 환자를 위한 예외 로직 없음.")
    lines.append("- stage2_holdout 환자는 이 분석에 포함되지 않는다.\n")

    lines.append("## F 비교 기준값\n")
    lines.append("| variant | n_candidates | lesion_patch_recall |")
    lines.append("|---------|-------------|---------------------|")
    lines.append(f"| F0_original_p95 | {f0_n:,} | {f0_recall:.6f} |")
    lines.append(f"| F1_original_p95_plus_valid950_topN | {f1_n:,} | {f1_recall:.6f} |")
    lines.append(f"| F5_F3_plus_D2_grid | {f5_n:,} | {f5_recall:.6f} |\n")

    lines.append("## Variant 요약\n")
    lines.append("| variant | n_candidates | lesion_recall | slice_recall | hit_rate | nohit | crop_feasible | delta_n_vs_f5 | delta_recall_vs_f5 |")
    lines.append("|---------|-------------|---------------|--------------|----------|-------|---------------|---------------|--------------------|")
    for m in all_metrics:
        fc = m.get("f_comparison", {})
        dn = fc.get("delta_n_vs_f5")
        dr = fc.get("delta_recall_vs_f5")
        dn_str = f"{int(dn):+,}" if isinstance(dn, (int, float)) else "N/A"
        dr_str = f"{float(dr):+.4f}" if isinstance(dr, (int, float)) else "N/A"
        lines.append(
            f"| {m['variant']} | {m['n_candidates']:,} "
            f"| {m['lesion_patch_recall']:.4f} | {m['lesion_slice_recall']:.4f} "
            f"| {m['patient_hit_rate']:.4f} | {m['n_nohit']} "
            f"| {m['crop_feasible']} | {dn_str} | {dr_str} |"
        )

    lines.append("\n## Weak Patient 회수 테이블\n")
    lines.append("| variant | " + " | ".join(WEAK_PATIENTS) + " |")
    lines.append("|---------|" + "|".join(["---"] * len(WEAK_PATIENTS)) + "|")
    for m in all_metrics:
        hits = [str(m["weak_patients_hit"].get(wp)) for wp in WEAK_PATIENTS]
        lines.append(f"| {m['variant']} | " + " | ".join(hits) + " |")

    lines.append("\n## Patient topk hit rate\n")
    lines.append("> patient_topk_hit_rate는 variant별 원래 selection score가 아니라 composite_rank_v2 기준 재랭킹 후 top-k hit rate이다. 따라서 후보 pool을 같은 재랭킹 점수로 정렬했을 때의 비교로 해석한다.\n")
    lines.append("| variant | top10 | top30 | top50 | top100 | top500 |")
    lines.append("|---------|-------|-------|-------|--------|--------|")
    for m in all_metrics:
        topk = m.get("patient_topk_hit_rate", {})
        lines.append(
            f"| {m['variant']} "
            f"| {topk.get('top10', 'N/A')} "
            f"| {topk.get('top30', 'N/A')} "
            f"| {topk.get('top50', 'N/A')} "
            f"| {topk.get('top100', 'N/A')} "
            f"| {topk.get('top500', 'N/A')} |"
        )

    lines.append("\n## 병변 크기별 Recall\n")
    lines.append("| variant | tiny | small | medium | large |")
    lines.append("|---------|------|-------|--------|-------|")
    for m in all_metrics:
        sr = m["patient_lesion_size_recall"]
        lines.append(
            f"| {m['variant']} "
            f"| {sr.get('tiny', 'N/A')} "
            f"| {sr.get('small', 'N/A')} "
            f"| {sr.get('medium', 'N/A')} "
            f"| {sr.get('large', 'N/A')} |"
        )

    lines.append("\n## Central / Peripheral Recall\n")
    lines.append("| variant | central | peripheral |")
    lines.append("|---------|---------|------------|")
    for m in all_metrics:
        cp = m["central_peripheral_recall"]
        lines.append(
            f"| {m['variant']} "
            f"| {cp.get('central', 'N/A')} "
            f"| {cp.get('peripheral', 'N/A')} |"
        )

    lines.append("\n## GC1 설계 설명 (이번 단계 실행 제외)\n")
    lines.append(
        "GC1_component_seed: G0 union GP2(composite_rank_v2 patient top300) 후보에서 "
        "같은 환자 내 local_z가 인접(|dz|<=2)하고 patch bbox가 겹치거나 가까운 후보끼리 "
        "연결 성분(component)으로 묶기. "
        "각 component에서 max composite_rank_v2 / mean score_original / 후보 수 계산 후 "
        "component별 대표 후보 top-k 선택. "
        "이번 단계에서는 실행 제외."
    )

    lines.append("\n## 최종 추천 variant 선택 안내\n")
    lines.append("- 이 단계에서 variant manifest 파일은 저장하지 않았습니다.")
    lines.append("- variant manifest 생성은 사용자 승인 후 별도 진행합니다.")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"저장: {OUT_MD}")

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rule G rank-based 후보 진단 스크립트 (--run 없으면 preflight만)")
    parser.add_argument("--run", action="store_true", help="실제 분석 실행")
    return parser.parse_args()


def main():
    args = parse_args()
    guard_check()
    dev_patients, holdout_patients = load_stage_split()

    if not args.run:
        preflight_report(dev_patients)
        return

    df = load_diag_filtered(dev_patients, holdout_patients)
    df = compute_rank_scores(df)
    gc.collect()

    variant_masks = build_variants(df)
    f_cmp = load_f_comparison()

    all_metrics = []
    for name, mask in variant_masks.items():
        print(f"  {name} 계산 중...")
        try:
            df_v = df.loc[mask]
            m = compute_variant_metrics(df_v, df, name, f_cmp)
            all_metrics.append(m)
            del df_v
            gc.collect()
        except Exception as e:
            print(f"  [오류] {name}: {e}")
            traceback.print_exc()

    save_outputs(all_metrics, f_cmp)
    print(f"\n=== 완료 ===\n출력 디렉토리: {OUT_DIR}")


if __name__ == "__main__":
    main()
