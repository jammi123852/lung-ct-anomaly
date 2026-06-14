"""
rule_f_candidate_diagnostic.py

Rule F 후보 variant를 manifest-only로 비교하는 진단 스크립트.
crop/npz/PNG 생성 없음. summary CSV/JSON/MD만 출력.

실행:
  --run 없이: preflight 체크만 수행 후 종료
  --run: 실제 분석 실행

절대 금지:
- crop/npz/PNG 생성 금지
- 모델 학습 / scoring 재실행 금지
- 기존 score/candidate/evaluation/crop 파일 수정 금지
- stage2_holdout 환자 분석 금지
- weak 환자 전용 예외 로직 금지
- lesion local_z 직접 후보 추가 금지
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
S4D2_MANIFEST = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/candidates/s4_plus_d2_union_stage1_dev_candidate_manifest_dryrun.csv"
RULED_MANIFEST = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/candidates/rule_d_stage1_dev_candidate_manifest_dryrun.csv"

OUT_DIR = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/reports"
OUT_CSV = OUT_DIR / "rule_f_candidate_diagnostic_summary.csv"
OUT_JSON = OUT_DIR / "rule_f_candidate_diagnostic_summary.json"
OUT_MD = OUT_DIR / "rule_f_candidate_diagnostic_summary.md"

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
V2V2_P95_THRESHOLD = 14.092057666455288
DEDUP_KEYS = ["patient_id", "local_z", "y0", "x0", "y1", "x1"]
WEAK_PATIENTS = ["LUNG1-156", "LUNG1-415", "MSD_lung_071", "MSD_lung_096", "MSD_lung_079"]
REQUIRED_COLS = [
    "patient_id", "local_z", "y0", "x0", "y1", "x1",
    "model_type", "padim_score",
    "score_original", "score_valid950_weighted",
    "score_valid950_pow025", "score_valid950_soft",
    "lesion_patch_ratio", "position_bin", "z_level", "central_peripheral",
]
CHUNKSIZE = 200_000
CROP_FEASIBLE_THRESHOLD = 50_000

# ---------------------------------------------------------------------------
# Step 0: Guard 체크
# ---------------------------------------------------------------------------
def guard_check(run_mode: bool) -> None:
    # 출력 파일 충돌 확인
    for f in [OUT_CSV, OUT_JSON, OUT_MD]:
        if f.exists():
            print(f"[중단] 출력 파일 이미 존재: {f}")
            print("  기존 파일을 삭제하거나 이름을 바꾼 후 재실행하세요.")
            sys.exit(1)

    # 입력 파일 존재 확인
    missing = []
    for f in [DIAG_CSV, STAGE_SPLIT_CSV, S4D2_MANIFEST, RULED_MANIFEST]:
        if not f.exists():
            missing.append(str(f))
    if missing:
        print("[중단] 입력 파일 없음:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    print("[Guard] 모든 입력 파일 존재 확인 ✓")

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
    print(f"  필수 컬럼 확인 ✓")

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
            chunks.append(filtered[REQUIRED_COLS])
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

    # stage2_holdout 침투 확인
    holdout_in_data = set(df["patient_id"].unique()) & holdout_patients
    if holdout_in_data:
        print(f"[중단] stage2_holdout 환자가 데이터에 포함됨: {holdout_in_data}")
        sys.exit(1)
    print(f"  stage2_holdout 봉인 확인 ✓")

    # 중복 제거
    before = len(df)
    df = df.drop_duplicates(subset=DEDUP_KEYS)
    after = len(df)
    print(f"  중복 제거: {before:,} → {after:,} (제거 {before - after:,})")

    if after == 0:
        print("[중단] 중복 제거 후 0행")
        sys.exit(1)

    print(f"  환자 수: {df['patient_id'].nunique()}")
    print(f"  model_type 값: {df['model_type'].unique().tolist()}")

    return df

# ---------------------------------------------------------------------------
# Step 3: S4+D2 union manifest 로드
# ---------------------------------------------------------------------------
def load_s4d2_manifest(dev_patients: set[str]) -> pd.DataFrame:
    df = pd.read_csv(S4D2_MANIFEST, encoding="utf-8-sig")
    total_all = len(df)
    print(f"[S4+D2] manifest 전체 행 수: {total_all:,} (기대: 111,176)")
    if total_all != 111_176:
        print(f"  [경고] 기대값 111,176과 다름: {total_all:,}")
    if "stage_split" in df.columns:
        df = df[df["stage_split"] == "stage1_dev"]
    df = df[df["patient_id"].isin(dev_patients)]
    df = df.drop_duplicates(subset=["patient_id", "local_z", "y0", "x0", "y1", "x1"])
    print(f"[S4+D2] 로드: {len(df):,}행, {df['patient_id'].nunique()}명")
    print(f"  실제 컬럼: {list(df.columns)}")
    avail_eval = [c for c in ["sampling_label", "source"] if c in df.columns]
    print(f"  평가 가능 컬럼: {avail_eval}")
    return df

# ---------------------------------------------------------------------------
# Step 4: Rule D D2 manifest 로드
# ---------------------------------------------------------------------------
def load_d2_manifest(dev_patients: set[str]) -> pd.DataFrame:
    df = pd.read_csv(RULED_MANIFEST, encoding="utf-8-sig")
    all_d2 = df[df["rule_d_variant"] == "D2_grid4x4_all_suspicious_slices"]
    print(f"[Rule D D2] D2_grid4x4_all_suspicious_slices 전체: {len(all_d2):,} (기대: 62,765)")
    if len(all_d2) != 62_765:
        print(f"  [경고] 기대값 62,765와 다름: {len(all_d2):,}")
    df = all_d2[all_d2["patient_id"].isin(dev_patients)].copy()
    df = df.drop_duplicates(subset=["patient_id", "local_z", "y0", "x0", "y1", "x1"])
    print(f"[Rule D D2] 로드: {len(df):,}행, {df['patient_id'].nunique()}명")
    print(f"  실제 컬럼: {list(df.columns)}")
    avail_eval = [c for c in ["sampling_label", "lesion_overlap_ratio", "lesion_pixel_count"] if c in df.columns]
    print(f"  평가 가능 컬럼: {avail_eval}")
    return df

# ---------------------------------------------------------------------------
# Step 4.5: lesion hit 판단 함수
# ---------------------------------------------------------------------------
def is_lesion_series(df: pd.DataFrame) -> pd.Series:
    """
    lesion hit 여부 벡터화 판단.
    우선순위: lesion_patch_ratio > 0 → patch_label == 1 → lesion_overlap == True → sampling_label == "positive"
    """
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


# ---------------------------------------------------------------------------
# Step 5: variant 후보 인덱스 생성
# ---------------------------------------------------------------------------
def get_topn_mask(df: pd.DataFrame, score_col: str, n: int) -> pd.Series:
    if n <= 0:
        return pd.Series(False, index=df.index)
    cutoff = df[score_col].nlargest(n).min()
    return df[score_col] >= cutoff


def get_patient_topk_mask(df: pd.DataFrame, score_col: str, k: int) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for pid, sub in df.groupby("patient_id"):
        if len(sub) == 0:
            continue
        top_idx = sub[score_col].nlargest(min(k, len(sub))).index
        mask.loc[top_idx] = True
    return mask


def build_variants(df: pd.DataFrame, df_s4d2: pd.DataFrame, df_d2: pd.DataFrame) -> dict[str, pd.DataFrame]:
    thr = V2V2_P95_THRESHOLD

    f0_mask = df["score_original"] >= thr
    n_baseline = int(f0_mask.sum())
    print(f"\n[Variants] F0 baseline 후보 수: {n_baseline:,}")

    # topN 마스크
    v950_topn = get_topn_mask(df, "score_valid950_weighted", n_baseline)
    pow025_topn = get_topn_mask(df, "score_valid950_pow025", n_baseline)
    soft_topn = get_topn_mask(df, "score_valid950_soft", n_baseline)

    f0 = df[f0_mask].copy()
    f1 = df[f0_mask | v950_topn].copy()
    f2 = df[f0_mask | soft_topn].copy()
    f3 = df[f0_mask | v950_topn | soft_topn].copy()
    f4 = df[f0_mask | v950_topn | pow025_topn | soft_topn].copy()

    # F5 = F3 ∪ D2
    f3_keys = set(map(tuple, f3[["patient_id", "local_z", "y0", "x0", "y1", "x1"]].values.tolist()))
    d2_keys = set(map(tuple, df_d2[["patient_id", "local_z", "y0", "x0", "y1", "x1"]].values.tolist()))
    d2_new_keys = d2_keys - f3_keys
    # D2 추가 행: key 컬럼 + 평가 컬럼 보존
    D2_EVAL_COLS = ["sampling_label", "lesion_overlap_ratio", "lesion_pixel_count"]
    d2_keep = DEDUP_KEYS + [c for c in D2_EVAL_COLS if c in df_d2.columns]
    f5_df_part = df_d2[
        df_d2.apply(lambda r: (r["patient_id"], r["local_z"], r["y0"], r["x0"], r["y1"], r["x1"]) in d2_new_keys, axis=1)
    ][d2_keep].copy()
    # lesion_overlap_ratio → lesion_patch_ratio 직접 매핑 (patch_area 없으므로 ratio 그대로 사용)
    if "lesion_overlap_ratio" in f5_df_part.columns:
        f5_df_part["lesion_patch_ratio"] = f5_df_part["lesion_overlap_ratio"]
        print(f"  F5 D2 추가: lesion_overlap_ratio→lesion_patch_ratio 변환 적용")
    elif "sampling_label" in f5_df_part.columns:
        f5_df_part["lesion_patch_ratio"] = f5_df_part["sampling_label"].map({"positive": 1.0}).fillna(0.0)
        print(f"  F5 D2 추가: sampling_label→lesion_patch_ratio 임시 변환 적용 (positive→1.0, 그 외→0.0)")
    n_d2_label_unknown = int(f5_df_part[["lesion_patch_ratio", "sampling_label"]].isna().all(axis=1).sum()) if "sampling_label" in f5_df_part.columns else int(f5_df_part["lesion_patch_ratio"].isna().sum() if "lesion_patch_ratio" in f5_df_part.columns else len(f5_df_part))
    print(f"  F5 D2 추가 행: {len(f5_df_part):,}행, label 판단 불가: {n_d2_label_unknown}행")
    for col in f3.columns:
        if col not in f5_df_part.columns:
            f5_df_part[col] = np.nan
    f5 = pd.concat([f3, f5_df_part], ignore_index=True).drop_duplicates(subset=DEDUP_KEYS)

    # F6 = F3 ∪ S4D2 union
    s4d2_keys = set(map(tuple, df_s4d2[["patient_id", "local_z", "y0", "x0", "y1", "x1"]].values.tolist()))
    f3_keys_for_f6 = set(map(tuple, f3[["patient_id", "local_z", "y0", "x0", "y1", "x1"]].values.tolist()))
    s4d2_new_keys = s4d2_keys - f3_keys_for_f6
    # S4D2 추가 행: key 컬럼 + 평가 컬럼 보존
    S4D2_EVAL_COLS = ["sampling_label", "source"]
    s4d2_keep = DEDUP_KEYS + [c for c in S4D2_EVAL_COLS if c in df_s4d2.columns]
    s4d2_new_part = df_s4d2[
        df_s4d2.apply(lambda r: (r["patient_id"], r["local_z"], r["y0"], r["x0"], r["y1"], r["x1"]) in s4d2_new_keys, axis=1)
    ][s4d2_keep].copy()
    # sampling_label → lesion_patch_ratio 임시 변환 (summary에 명시)
    if "sampling_label" in s4d2_new_part.columns:
        s4d2_new_part["lesion_patch_ratio"] = s4d2_new_part["sampling_label"].map({"positive": 1.0}).fillna(0.0)
        print(f"  F6 S4D2 추가: sampling_label→lesion_patch_ratio 임시 변환 적용 (positive→1.0, 그 외→0.0)")
    n_s4d2_label_unknown = int(s4d2_new_part["lesion_patch_ratio"].isna().sum()) if "lesion_patch_ratio" in s4d2_new_part.columns else int(len(s4d2_new_part))
    print(f"  F6 S4D2 추가 행: {len(s4d2_new_part):,}행, label 판단 불가: {n_s4d2_label_unknown}행")
    for col in f3.columns:
        if col not in s4d2_new_part.columns:
            s4d2_new_part[col] = np.nan
    f6 = pd.concat([f3, s4d2_new_part], ignore_index=True).drop_duplicates(subset=DEDUP_KEYS)

    # Patient-wise variants
    p1_mask = get_patient_topk_mask(df, "score_valid950_weighted", 50)
    p2_mask = get_patient_topk_mask(df, "score_valid950_weighted", 100)
    p3_mask = get_patient_topk_mask(df, "score_valid950_soft", 100)

    p1 = df[f0_mask | p1_mask].copy()
    p2 = df[f0_mask | p2_mask].copy()
    p3 = df[f0_mask | p3_mask].copy()

    variants = {
        "F0_original_p95": f0,
        "F1_original_p95_plus_valid950_topN": f1,
        "F2_original_p95_plus_soft_topN": f2,
        "F3_original_p95_plus_valid950_soft_topN": f3,
        "F4_original_p95_plus_valid950_pow025_soft_topN": f4,
        "F5_F3_plus_D2_grid": f5,
        "F6_F3_plus_S4D2_union": f6,
        "P1_original_p95_plus_valid950_patient_top50": p1,
        "P2_original_p95_plus_valid950_patient_top100": p2,
        "P3_original_p95_plus_soft_patient_top100": p3,
    }
    for name, v in variants.items():
        print(f"  {name}: {len(v):,}행")
    return variants

# ---------------------------------------------------------------------------
# Step 6: 평가 지표 계산
# ---------------------------------------------------------------------------
def compute_variant_metrics(
    df_variant: pd.DataFrame,
    df_full: pd.DataFrame,
    variant_name: str,
    df_s4d2: pd.DataFrame,
) -> dict:
    lesion_flag_full = df_full["lesion_patch_ratio"] > 0
    n_lesion_total = int(lesion_flag_full.sum())
    lesion_slices_full = set(
        map(tuple, df_full[lesion_flag_full][["patient_id", "local_z"]].values.tolist())
    )
    lesion_patients_full = set(df_full[lesion_flag_full]["patient_id"].unique())

    # label 판단 불가 행 수 계산
    label_cols = ["lesion_patch_ratio", "patch_label", "lesion_overlap", "sampling_label"]
    avail_label_cols = [c for c in label_cols if c in df_variant.columns]
    if avail_label_cols:
        n_label_unknown = int(df_variant[avail_label_cols].isna().all(axis=1).sum())
    else:
        n_label_unknown = len(df_variant)

    # is_lesion_series 기반으로 전체 행 평가 (dropna 제거)
    df_v_scored = df_variant
    lesion_flag_v = is_lesion_series(df_v_scored)

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
        map(tuple, df_v_scored[lesion_flag_v][["patient_id", "local_z"]].values.tolist())
    )
    hit_lesion_slices = lesion_slices_full & v_slices
    lesion_slice_recall = float(len(hit_lesion_slices) / len(lesion_slices_full)) if lesion_slices_full else 0.0

    # patient hit rate
    hit_patients = set(df_v_scored[lesion_flag_v]["patient_id"].unique())
    nohit_patients = sorted(lesion_patients_full - hit_patients)
    patient_hit_rate = float(len(hit_patients) / len(lesion_patients_full)) if lesion_patients_full else 0.0

    # weak patients
    weak_hit = {}
    for wp in WEAK_PATIENTS:
        wp_lesion = df_full[(df_full["patient_id"] == wp) & (df_full["lesion_patch_ratio"] > 0)]
        wp_v = df_v_scored[df_v_scored["patient_id"] == wp]
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
        bin_v_sub = df_v_scored[df_v_scored["patient_id"].isin(bin_patients)]
        n_bin_lesion = len(bin_lesion)
        n_bin_hit = int(is_lesion_series(bin_v_sub).sum())
        size_recall[bin_name] = float(n_bin_hit / n_bin_lesion) if n_bin_lesion > 0 else None

    # central/peripheral recall
    cp_recall = {}
    for cp_val in ["central", "peripheral"]:
        sub_full = df_full[(df_full["central_peripheral"] == cp_val) & (df_full["lesion_patch_ratio"] > 0)]
        if "central_peripheral" in df_v_scored.columns:
            sub_v = df_v_scored[df_v_scored["central_peripheral"] == cp_val]
        else:
            sub_v = df_v_scored.iloc[0:0]
        n_total_cp = len(sub_full)
        n_hit_cp = int(is_lesion_series(sub_v).sum())
        cp_recall[cp_val] = float(n_hit_cp / n_total_cp) if n_total_cp > 0 else None

    # position_bin recall
    pos_recall = {}
    for pb in sorted(df_full["position_bin"].dropna().unique()):
        sub_full = df_full[(df_full["position_bin"] == pb) & (df_full["lesion_patch_ratio"] > 0)]
        if "position_bin" in df_v_scored.columns:
            sub_v = df_v_scored[df_v_scored["position_bin"] == pb]
        else:
            sub_v = df_v_scored.iloc[0:0]
        n_total_pb = len(sub_full)
        n_hit_pb = int(is_lesion_series(sub_v).sum())
        pos_recall[str(pb)] = float(n_hit_pb / n_total_pb) if n_total_pb > 0 else None

    # S4+D2 비교: S4D2 manifest 자체의 label 사용
    s4d2_n = len(df_s4d2)
    if "sampling_label" in df_s4d2.columns:
        s4d2_positive_count = int((df_s4d2["sampling_label"] == "positive").sum())
        s4d2_patch_recall = float(s4d2_positive_count / n_lesion_total) if n_lesion_total > 0 else 0.0
        s4d2_recall_source = "s4d2_manifest_sampling_label"
    else:
        s4d2_lesion = df_full.merge(
            df_s4d2[["patient_id", "local_z", "y0", "x0", "y1", "x1"]],
            on=["patient_id", "local_z", "y0", "x0", "y1", "x1"], how="inner"
        )
        s4d2_positive_count = int((s4d2_lesion["lesion_patch_ratio"] > 0).sum())
        s4d2_patch_recall = float(s4d2_positive_count / n_lesion_total) if n_lesion_total > 0 else 0.0
        s4d2_recall_source = "merge_with_df_full"
    s4d2_comparison = {
        "s4d2_n_candidates": s4d2_n,
        "s4d2_lesion_patch_recall": s4d2_patch_recall,
        "s4d2_recall_source": s4d2_recall_source,
        "delta_n_candidates": n_candidates - s4d2_n,
        "delta_lesion_patch_recall": round(lesion_patch_recall - s4d2_patch_recall, 6),
    }

    crop_feasible = bool(n_candidates < CROP_FEASIBLE_THRESHOLD)

    return {
        "variant": variant_name,
        "n_candidates": n_candidates,
        "n_patients": n_patients,
        "n_label_unknown": n_label_unknown,
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
        "s4d2_comparison": s4d2_comparison,
        "crop_feasible": crop_feasible,
    }

# ---------------------------------------------------------------------------
# Step 7: 출력 저장
# ---------------------------------------------------------------------------
def save_outputs(all_metrics: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # CSV (flat)
    flat_rows = []
    for m in all_metrics:
        row = {
            "variant": m["variant"],
            "n_candidates": m["n_candidates"],
            "n_patients": m["n_patients"],
            "per_patient_min": m["per_patient_stats"]["min"],
            "per_patient_median": m["per_patient_stats"]["median"],
            "per_patient_mean": m["per_patient_stats"]["mean"],
            "per_patient_max": m["per_patient_stats"]["max"],
            "n_positive": m["n_positive"],
            "n_fp": m["n_fp"],
            "positive_ratio": m["positive_ratio"],
            "fp_ratio": m["fp_ratio"],
            "lesion_patch_recall": m["lesion_patch_recall"],
            "lesion_slice_recall": m["lesion_slice_recall"],
            "patient_hit_rate": m["patient_hit_rate"],
            "n_nohit": m["n_nohit"],
            "n_label_unknown": m.get("n_label_unknown", 0),
            "crop_feasible": m["crop_feasible"],
            "delta_n_vs_s4d2": m["s4d2_comparison"]["delta_n_candidates"],
            "delta_recall_vs_s4d2": m["s4d2_comparison"]["delta_lesion_patch_recall"],
            "s4d2_recall_source": m["s4d2_comparison"].get("s4d2_recall_source", ""),
        }
        for wp in WEAK_PATIENTS:
            row[f"weak_hit_{wp}"] = m["weak_patients_hit"].get(wp)
        for size in ["tiny", "small", "medium", "large"]:
            row[f"size_recall_{size}"] = m["patient_lesion_size_recall"].get(size)
        for cp in ["central", "peripheral"]:
            row[f"cp_recall_{cp}"] = m["central_peripheral_recall"].get(cp)
        flat_rows.append(row)

    pd.DataFrame(flat_rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT_CSV}")

    # JSON
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
    lines = ["# Rule F 후보 Variant 진단 보고서\n"]
    lines.append("## 해석 주의사항\n")
    lines.append("- adjusted score는 후보 삭제용으로 쓰지 않음. original p95 후보는 항상 유지.")
    lines.append("- adjusted score는 topN 보조 후보 또는 재랭킹 후보로만 사용.")
    lines.append("- D2는 위치 miss 보완용 grid fallback으로만 사용.")
    lines.append("- weak 환자 전용 예외 로직 없음.")
    lines.append("- lesion local_z 직접 후보 추가 없음.")
    lines.append("- stage2_holdout 봉인. stage1_dev만 분석.")
    lines.append("- F5 D2 추가 행: lesion_overlap_ratio → lesion_patch_ratio 직접 사용 (patch_area 없으므로 ratio 그대로 매핑).")
    lines.append("- F6 S4D2 추가 행: sampling_label → lesion_patch_ratio 임시 변환 (positive→1.0, 그 외→0.0). lesion_patch_ratio가 원본에 없으므로 임시값이며 해석 제한 적용.")
    lines.append("- n_label_unknown: label 판단 가능 컬럼이 모두 NaN인 행 수. 이 행 수만큼 positive/recall 지표 과소 평가 가능성 있음.\n")

    lines.append("## Variant 요약\n")
    lines.append("| variant | n_candidates | lesion_recall | slice_recall | hit_rate | nohit | crop_feasible | delta_n_vs_s4d2 |")
    lines.append("|---------|-------------|---------------|--------------|----------|-------|---------------|-----------------|")
    for m in all_metrics:
        lines.append(
            f"| {m['variant']} | {m['n_candidates']:,} "
            f"| {m['lesion_patch_recall']:.4f} | {m['lesion_slice_recall']:.4f} "
            f"| {m['patient_hit_rate']:.4f} | {m['n_nohit']} "
            f"| {m['crop_feasible']} | {m['s4d2_comparison']['delta_n_candidates']:+,} |"
        )

    lines.append("\n## Weak Patient 회수 여부\n")
    lines.append("| variant | " + " | ".join(WEAK_PATIENTS) + " |")
    lines.append("|---------|" + "|".join(["---"] * len(WEAK_PATIENTS)) + "|")
    for m in all_metrics:
        hits = [str(m["weak_patients_hit"].get(wp)) for wp in WEAK_PATIENTS]
        lines.append(f"| {m['variant']} | " + " | ".join(hits) + " |")

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

    lines.append("\n## S4+D2 union 대비 변화\n")
    lines.append("| variant | s4d2_n | s4d2_recall | delta_n | delta_recall |")
    lines.append("|---------|--------|-------------|---------|--------------|")
    for m in all_metrics:
        c = m["s4d2_comparison"]
        lines.append(
            f"| {m['variant']} "
            f"| {c['s4d2_n_candidates']:,} "
            f"| {c['s4d2_lesion_patch_recall']:.4f} "
            f"| {c['delta_n_candidates']:+,} "
            f"| {c['delta_lesion_patch_recall']:+.4f} |"
        )

    lines.append("\n## 최종 추천 variant manifest 생성")
    lines.append("- 이 단계에서 variant manifest 파일은 저장하지 않았습니다.")
    lines.append("- 최종 추천 variant 1개에 대해 manifest를 생성하려면 **사용자 승인 후 별도 진행**이 필요합니다.")
    lines.append(f"- 후보 manifest 예정 경로: `outputs/second-stage-lesion-refiner-v1/candidates/rule_f_selected_candidate_manifest_dryrun.csv`")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"저장: {OUT_MD}")

# ---------------------------------------------------------------------------
# Preflight 전용 보고
# ---------------------------------------------------------------------------
def preflight_report(dev_patients: set[str]) -> None:
    print("\n=== Preflight 보고 ===")
    print(f"입력 파일: ratio_adjusted_score_full_diagnostic.csv ({DIAG_CSV.stat().st_size / 1e9:.1f}GB)")
    print(f"stage1_dev 환자 수: {len(dev_patients)}명")

    # 예상 행 수 추정 (첫 chunk 샘플)
    sample = next(pd.read_csv(DIAG_CSV, chunksize=CHUNKSIZE, encoding="utf-8-sig", low_memory=False))
    sample_filtered = sample[(sample["model_type"] == "v2v2") & (sample["patient_id"].isin(dev_patients))]
    sample_rate = len(sample_filtered) / len(sample) if len(sample) > 0 else 0
    total_rows_est = 11_749_900
    est_filtered = int(total_rows_est * sample_rate)
    print(f"예상 필터 후 행 수: ~{est_filtered:,}행 (샘플 비율 {sample_rate:.3f})")
    print(f"예상 청크 수: ~{int(total_rows_est / CHUNKSIZE) + 1}개")
    print(f"예상 소요 시간: 5~15분 (npy 없음, CSV 읽기만)")
    print(f"메모리 위험: 낮음 (chunk 처리, 약 {est_filtered * 40 / 1e9:.1f}GB 예상)")
    print(f"\n출력 예정 파일:")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_JSON}")
    print(f"  {OUT_MD}")
    print(f"\n실행 명령 초안:")
    print(f"  /home/jinhy/ai_env/bin/python3 scripts/rule_f_candidate_diagnostic.py --run")
    print(f"\n[Preflight 완료] 실제 실행은 --run 플래그 추가 후 진행하세요.")

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rule F 후보 진단 스크립트 (--run 없으면 preflight만)")
    parser.add_argument("--run", action="store_true", help="실제 분석 실행")
    return parser.parse_args()


def main():
    args = parse_args()

    guard_check(run_mode=args.run)

    dev_patients, holdout_patients = load_stage_split()

    if not args.run:
        preflight_report(dev_patients)
        return

    # 실제 실행
    print("\n=== 실제 분석 실행 ===")

    df_full = load_diag_filtered(dev_patients, holdout_patients)
    gc.collect()

    df_s4d2 = load_s4d2_manifest(dev_patients)
    df_d2 = load_d2_manifest(dev_patients)

    variants = build_variants(df_full, df_s4d2, df_d2)

    print("\n=== 지표 계산 ===")
    all_metrics = []
    for name, df_v in variants.items():
        print(f"  {name} 계산 중...")
        try:
            m = compute_variant_metrics(df_v, df_full, name, df_s4d2)
            all_metrics.append(m)
        except Exception as e:
            print(f"  [오류] {name}: {e}")
            traceback.print_exc()

    print("\n=== 출력 저장 ===")
    save_outputs(all_metrics)

    print("\n=== 완료 ===")
    print(f"출력 디렉토리: {OUT_DIR}")


if __name__ == "__main__":
    main()
