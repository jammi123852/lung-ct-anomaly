"""
rule_s6_gs2_sampling_design.py

GS2_slice_top30 후보 pool에서 학습용 sampling variant를 비교하는 설계 스크립트.
summary CSV/JSON/MD만 출력. manifest 저장 없음. crop/npz/PNG 생성 없음.

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
- 병변 mask를 후보 pool 생성에 사용 금지
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
RULE_G_JSON = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/reports/rule_g_rank_based_candidate_summary.json"
RULE_G_CSV = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/reports/rule_g_rank_based_candidate_summary.csv"
RULE_F_JSON = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/reports/rule_f_candidate_diagnostic_summary.json"

OUT_DIR = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/reports"
OUT_CSV = OUT_DIR / "rule_s6_gs2_sampling_design_summary.csv"
OUT_JSON = OUT_DIR / "rule_s6_gs2_sampling_design_summary.json"
OUT_MD = OUT_DIR / "rule_s6_gs2_sampling_design_summary.md"

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
V2V2_P95_THRESHOLD = 14.092057666455288
DEDUP_KEYS = ["patient_id", "local_z", "y0", "x0", "y1", "x1"]
WEAK_PATIENTS = ["LUNG1-156", "LUNG1-415", "MSD_lung_071", "MSD_lung_096", "MSD_lung_079"]
CHUNKSIZE = 200_000
CROP_FEASIBLE_THRESHOLD = 50_000
EXPLOSION_THRESHOLD = 2000

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
    for f in [DIAG_CSV, STAGE_SPLIT_CSV, RULE_G_JSON, RULE_G_CSV, RULE_F_JSON]:
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

    first_chunk = pd.read_csv(DIAG_CSV, nrows=1, encoding="utf-8-sig")
    missing_cols = [c for c in REQUIRED_COLS if c not in first_chunk.columns]
    if missing_cols:
        print(f"[중단] 필수 컬럼 누락: {missing_cols}")
        sys.exit(1)
    print("  필수 컬럼 확인 완료")

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

    before = len(df)
    df = df.drop_duplicates(subset=DEDUP_KEYS)
    after = len(df)
    print(f"  중복 제거: {before:,} → {after:,} (제거 {before - after:,})")

    holdout_in_data = set(df["patient_id"].unique()) & holdout_patients
    if holdout_in_data:
        print(f"[중단] stage2_holdout 환자가 데이터에 포함됨: {holdout_in_data}")
        sys.exit(1)
    print("  stage2_holdout 봉인 확인 완료")

    if after == 0:
        print("[중단] 중복 제거 후 0행")
        sys.exit(1)

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

    df["patient_rank_original"] = df.groupby("patient_id")["score_original"].transform(rank_percentile)
    df["slice_rank_original"] = df.groupby(["patient_id", "local_z"])["score_original"].transform(rank_percentile)
    df["patient_rank_valid950"] = df.groupby("patient_id")["score_valid950_weighted"].transform(rank_percentile)
    df["slice_rank_valid950"] = df.groupby(["patient_id", "local_z"])["score_valid950_weighted"].transform(rank_percentile)

    df["composite_rank_v2"] = (
        0.4 * df["patient_rank_original"]
        + 0.3 * df["slice_rank_original"]
        + 0.2 * df["patient_rank_valid950"]
        + 0.1 * df["slice_rank_valid950"]
    )

    print("  rank 컬럼 4개 + composite_rank_v2 계산 완료")
    return df

# ---------------------------------------------------------------------------
# Step 4: GS2 pool 구성 (후보 pool 생성에 병변 label 미사용)
# ---------------------------------------------------------------------------
def build_gs2_mask(df: pd.DataFrame) -> pd.Series:
    print("\n[Step 4] GS2 pool 구성 시작")

    g0_mask = df["score_original"] >= V2V2_P95_THRESHOLD
    n_g0 = int(g0_mask.sum())
    print(f"  G0_original_p95 후보 수: {n_g0:,}")

    slice_top30_mask = pd.Series(False, index=df.index)
    for (pid, lz), sub in df.groupby(["patient_id", "local_z"]):
        top_idx = sub["composite_rank_v2"].nlargest(min(30, len(sub))).index
        slice_top30_mask.loc[top_idx] = True

    gs2_mask = g0_mask | slice_top30_mask
    n_gs2 = int(gs2_mask.sum())
    print(f"  GS2 pool 후보 수: {n_gs2:,} (G0 union slice top30)")
    return gs2_mask

# ---------------------------------------------------------------------------
# Step 5: Positive/hard_negative 판정 (sampling 단계에서만 사용)
# ---------------------------------------------------------------------------
def is_positive(df_sub: pd.DataFrame) -> pd.Series:
    flag = pd.Series(False, index=df_sub.index)
    if "lesion_patch_ratio" in df_sub.columns:
        flag = flag | (df_sub["lesion_patch_ratio"].fillna(0) > 0)
    if "patch_label" in df_sub.columns:
        flag = flag | (df_sub["patch_label"].fillna(0) == 1)
    if "lesion_overlap" in df_sub.columns:
        flag = flag | df_sub["lesion_overlap"].fillna(False).astype(bool)
    if "sampling_label" in df_sub.columns:
        flag = flag | (df_sub["sampling_label"].fillna("") == "positive")
    return flag

# ---------------------------------------------------------------------------
# Step 6: Sampling variant 함수들
# ---------------------------------------------------------------------------

def sample_ratio_hn(df_pool: pd.DataFrame, hn_ratio: float, patient_hn_cap: int) -> pd.Series:
    """S6-A/B: positive 전부 + hard_negative ratio배, 환자별 cap 적용."""
    pos_mask = is_positive(df_pool)
    pos_idx = set(df_pool[pos_mask].index.tolist())
    n_total_pos = len(pos_idx)
    target_total_hn = int(n_total_pos * hn_ratio)

    hn_df = df_pool[~pos_mask]

    # 환자별 cap 적용 후 전체에서 composite_rank_v2 상위 target_total_hn개 선택
    per_patient_hn = []
    for pid, sub in hn_df.groupby("patient_id"):
        top_hn = sub.sort_values("composite_rank_v2", ascending=False).head(patient_hn_cap)
        per_patient_hn.append(top_hn)

    if per_patient_hn:
        all_hn = pd.concat(per_patient_hn)
        selected_hn_idx = set(
            all_hn.sort_values("composite_rank_v2", ascending=False)
            .head(target_total_hn)
            .index.tolist()
        )
    else:
        selected_hn_idx = set()

    selected_idx = pos_idx | selected_hn_idx
    return df_pool.index.isin(selected_idx)


def sample_patient_cap(df_pool: pd.DataFrame, cap: int) -> pd.Series:
    """S6-C/D: 환자별 cap, positive 우선, 나머지는 hard_negative (position_bin 다양성)."""
    pos_mask = is_positive(df_pool)
    selected_idx = []

    for pid, sub in df_pool.groupby("patient_id"):
        pos_sub = sub[pos_mask.loc[sub.index]]
        hn_sub = sub[~pos_mask.loc[sub.index]].sort_values("composite_rank_v2", ascending=False)

        pos_idx = pos_sub.index.tolist()
        remaining = cap - len(pos_idx)

        if remaining <= 0:
            selected_idx.extend(pos_idx[:cap])
            continue

        if "position_bin" in hn_sub.columns and len(hn_sub) > 0:
            bins = hn_sub["position_bin"].dropna().unique()
            if len(bins) > 0:
                # position_bin별 비례 배분 후 composite_rank_v2 상위로 보충
                per_bin = max(1, remaining // len(bins))
                hn_selected_set: set = set()
                for b in bins:
                    bin_sub = hn_sub[hn_sub["position_bin"] == b]
                    hn_selected_set.update(bin_sub.index[:per_bin].tolist())
                # 부족분 composite_rank_v2 상위로 보충
                for idx in hn_sub.index:
                    if len(hn_selected_set) >= remaining:
                        break
                    hn_selected_set.add(idx)
                hn_idx = list(hn_selected_set)[:remaining]
            else:
                hn_idx = hn_sub.index[:remaining].tolist()
        else:
            hn_idx = hn_sub.index[:remaining].tolist()

        selected_idx.extend(pos_idx + hn_idx)

    return df_pool.index.isin(selected_idx)


def sample_slice_balanced(df_pool: pd.DataFrame, max_per_slice: int) -> pd.Series:
    """S6-E1/E2: slice별 max_per_slice개 제한, positive 우선, 나머지 composite_rank_v2 상위."""
    pos_mask = is_positive(df_pool)
    selected_idx = []

    for (pid, lz), sub in df_pool.groupby(["patient_id", "local_z"]):
        pos_sub = sub[pos_mask.loc[sub.index]]
        hn_sub = sub[~pos_mask.loc[sub.index]].sort_values("composite_rank_v2", ascending=False)

        pos_idx = pos_sub.index.tolist()
        remaining = max_per_slice - len(pos_idx)
        hn_idx = hn_sub.index[:max(0, remaining)].tolist()

        selected_idx.extend(pos_idx + hn_idx)

    return df_pool.index.isin(selected_idx)


def sample_size_sensitive_safe(df_pool: pd.DataFrame, df_full: pd.DataFrame) -> pd.Series:
    """S6-F: positive 전부 + tiny/small은 positive slice ±2 hn 우선, cap은 크기별로 다름."""
    pos_mask_full = df_full["lesion_patch_ratio"].fillna(0) > 0
    patient_lesion_counts = df_full[pos_mask_full].groupby("patient_id").size()

    if len(patient_lesion_counts) >= 4:
        q25 = patient_lesion_counts.quantile(0.25)
        q50 = patient_lesion_counts.quantile(0.50)
        q75 = patient_lesion_counts.quantile(0.75)

        def size_bin(val: float) -> str:
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

    pos_mask = is_positive(df_pool)
    selected_idx = []

    for pid, sub in df_pool.groupby("patient_id"):
        pos_sub = sub[pos_mask.loc[sub.index]]
        hn_sub = sub[~pos_mask.loc[sub.index]]

        size = patient_size_bin.get(pid, "medium")
        cap = 800 if size in ("tiny", "small") else 500

        pos_idx = pos_sub.index.tolist()
        remaining = cap - len(pos_idx)

        if remaining <= 0:
            selected_idx.extend(pos_idx[:cap])
            continue

        if size in ("tiny", "small") and len(pos_sub) > 0:
            # GS2 pool 내 positive 후보의 local_z ±2 인접 slice의 hard_negative 우선 보존
            pos_local_z = set(pos_sub["local_z"].unique())
            neighbor_z: set = set()
            for lz in pos_local_z:
                for dz in range(-2, 3):
                    neighbor_z.add(lz + dz)
            neighbor_hn = hn_sub[hn_sub["local_z"].isin(neighbor_z)].sort_values("composite_rank_v2", ascending=False)
            other_hn = hn_sub[~hn_sub["local_z"].isin(neighbor_z)].sort_values("composite_rank_v2", ascending=False)
            hn_sorted = pd.concat([neighbor_hn, other_hn])
        else:
            hn_sorted = hn_sub.sort_values("composite_rank_v2", ascending=False)

        hn_idx = hn_sorted.index[:remaining].tolist()
        selected_idx.extend(pos_idx + hn_idx)

    return df_pool.index.isin(selected_idx)

# ---------------------------------------------------------------------------
# Step 7: 평가 지표 계산
# ---------------------------------------------------------------------------
def get_patient_size_bin(df_full: pd.DataFrame) -> pd.Series:
    pos_mask = df_full["lesion_patch_ratio"].fillna(0) > 0
    patient_lesion_counts = df_full[pos_mask].groupby("patient_id").size()
    if len(patient_lesion_counts) < 4:
        return pd.Series(dtype=str)
    q25 = patient_lesion_counts.quantile(0.25)
    q50 = patient_lesion_counts.quantile(0.50)
    q75 = patient_lesion_counts.quantile(0.75)
    def size_bin(val: float) -> str:
        if val <= q25:
            return "tiny"
        elif val <= q50:
            return "small"
        elif val <= q75:
            return "medium"
        else:
            return "large"
    return patient_lesion_counts.apply(size_bin)


def compute_sampling_metrics(
    df_sampled: pd.DataFrame,
    df_full: pd.DataFrame,
    variant_name: str,
    f_comparison: dict,
    gs2_stats: dict,
    patient_size_bin: pd.Series,
) -> dict:
    pos_mask_full = df_full["lesion_patch_ratio"].fillna(0) > 0
    n_lesion_total = int(pos_mask_full.sum())
    lesion_slices_full = set(
        map(tuple, df_full[pos_mask_full][["patient_id", "local_z"]].values.tolist())
    )
    lesion_patients_full = set(df_full[pos_mask_full]["patient_id"].unique())

    pos_mask_s = is_positive(df_sampled)
    n_candidates = len(df_sampled)
    n_patients = df_sampled["patient_id"].nunique()

    per_patient = df_sampled.groupby("patient_id").size()
    per_patient_stats = {
        "min": int(per_patient.min()) if len(per_patient) > 0 else 0,
        "median": float(per_patient.median()) if len(per_patient) > 0 else 0.0,
        "mean": float(per_patient.mean()) if len(per_patient) > 0 else 0.0,
        "max": int(per_patient.max()) if len(per_patient) > 0 else 0,
    }
    explosion_patients = per_patient[per_patient > EXPLOSION_THRESHOLD].index.tolist()

    n_positive = int(pos_mask_s.sum())
    n_hn = int((~pos_mask_s).sum())
    positive_ratio = float(n_positive / n_candidates) if n_candidates > 0 else 0.0
    fp_ratio = float(n_hn / n_candidates) if n_candidates > 0 else 0.0

    lesion_patch_recall = float(n_positive / n_lesion_total) if n_lesion_total > 0 else 0.0

    s_slices = set(
        map(tuple, df_sampled[pos_mask_s][["patient_id", "local_z"]].values.tolist())
    )
    hit_lesion_slices = lesion_slices_full & s_slices
    lesion_slice_recall = float(len(hit_lesion_slices) / len(lesion_slices_full)) if lesion_slices_full else 0.0

    hit_patients = set(df_sampled[pos_mask_s]["patient_id"].unique())
    nohit_patients = sorted(lesion_patients_full - hit_patients)
    patient_hit_rate = float(len(hit_patients) / len(lesion_patients_full)) if lesion_patients_full else 0.0

    weak_hit = {}
    for wp in WEAK_PATIENTS:
        wp_lesion = df_full[(df_full["patient_id"] == wp) & (df_full["lesion_patch_ratio"].fillna(0) > 0)]
        wp_s = df_sampled[df_sampled["patient_id"] == wp]
        wp_hits = int(is_positive(wp_s).sum())
        weak_hit[wp] = bool(wp_hits > 0) if len(wp_lesion) > 0 else None

    size_recall: dict = {}
    for bin_name in ["tiny", "small", "medium", "large"]:
        bin_patients = set(patient_size_bin[patient_size_bin == bin_name].index)
        bin_lesion = df_full[(df_full["patient_id"].isin(bin_patients)) & (df_full["lesion_patch_ratio"].fillna(0) > 0)]
        bin_s = df_sampled[df_sampled["patient_id"].isin(bin_patients)]
        n_bin_lesion = len(bin_lesion)
        n_bin_hit = int(is_positive(bin_s).sum())
        size_recall[bin_name] = float(n_bin_hit / n_bin_lesion) if n_bin_lesion > 0 else None

    cp_recall: dict = {}
    for cp_val in ["central", "peripheral"]:
        if "central_peripheral" in df_full.columns:
            sub_full = df_full[(df_full["central_peripheral"] == cp_val) & (df_full["lesion_patch_ratio"].fillna(0) > 0)]
        else:
            sub_full = df_full.iloc[0:0]
        if "central_peripheral" in df_sampled.columns:
            sub_s = df_sampled[df_sampled["central_peripheral"] == cp_val]
        else:
            sub_s = df_sampled.iloc[0:0]
        n_total_cp = len(sub_full)
        n_hit_cp = int(is_positive(sub_s).sum())
        cp_recall[cp_val] = float(n_hit_cp / n_total_cp) if n_total_cp > 0 else None

    pos_recall: dict = {}
    if "position_bin" in df_full.columns:
        for pb in sorted(df_full["position_bin"].dropna().unique()):
            sub_full = df_full[(df_full["position_bin"] == pb) & (df_full["lesion_patch_ratio"].fillna(0) > 0)]
            if "position_bin" in df_sampled.columns:
                sub_s = df_sampled[df_sampled["position_bin"] == pb]
            else:
                sub_s = df_sampled.iloc[0:0]
            n_total_pb = len(sub_full)
            n_hit_pb = int(is_positive(sub_s).sum())
            pos_recall[str(pb)] = float(n_hit_pb / n_total_pb) if n_total_pb > 0 else None

    gs2_n = gs2_stats.get("n_candidates", 0)
    gs2_recall = gs2_stats.get("lesion_patch_recall", 0.0)
    reduction_rate_vs_gs2 = float((gs2_n - n_candidates) / gs2_n) if gs2_n > 0 else 0.0
    delta_recall_vs_gs2 = lesion_patch_recall - gs2_recall

    f5_n = f_comparison.get("f5_n", 0)
    f5_recall = f_comparison.get("f5_recall", 0.0)
    delta_n_vs_f5 = n_candidates - f5_n
    delta_recall_vs_f5 = lesion_patch_recall - f5_recall

    crop_feasible = bool(n_candidates < CROP_FEASIBLE_THRESHOLD)

    return {
        "variant": variant_name,
        "n_candidates": n_candidates,
        "n_patients": n_patients,
        "per_patient_stats": per_patient_stats,
        "explosion_patients": explosion_patients,
        "n_positive": n_positive,
        "n_hard_negative": n_hn,
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
        "gs2_comparison": {
            "gs2_n_candidates": gs2_n,
            "gs2_lesion_patch_recall": gs2_recall,
            "delta_n_vs_gs2": n_candidates - gs2_n,
            "reduction_rate_vs_gs2": round(reduction_rate_vs_gs2, 4),
            "delta_recall_vs_gs2": round(delta_recall_vs_gs2, 6),
        },
        "f5_comparison": {
            "f5_n_candidates": f5_n,
            "f5_lesion_patch_recall": f5_recall,
            "delta_n_vs_f5": delta_n_vs_f5,
            "delta_recall_vs_f5": round(delta_recall_vs_f5, 6),
        },
        "crop_feasible": crop_feasible,
    }

# ---------------------------------------------------------------------------
# Step 8: F/GS2 비교값 로드
# ---------------------------------------------------------------------------
def load_f_comparison() -> dict:
    with open(RULE_F_JSON, encoding="utf-8") as f:
        data = json.load(f)
    lookup = {m["variant"]: m for m in data}
    f5 = lookup.get("F5_F3_plus_D2_grid", {})
    return {
        "f5_n": f5.get("n_candidates", 0),
        "f5_recall": f5.get("lesion_patch_recall", 0.0),
    }


def load_gs2_stats() -> dict:
    with open(RULE_G_JSON, encoding="utf-8") as f:
        data = json.load(f)
    lookup = {m["variant"]: m for m in data}
    gs2 = lookup.get("GS2_slice_top30", {})
    return {
        "n_candidates": gs2.get("n_candidates", 0),
        "lesion_patch_recall": gs2.get("lesion_patch_recall", 0.0),
        "lesion_slice_recall": gs2.get("lesion_slice_recall", 0.0),
        "patient_hit_rate": gs2.get("patient_hit_rate", 0.0),
    }

# ---------------------------------------------------------------------------
# Step 9: Preflight 보고
# ---------------------------------------------------------------------------
def preflight_report(dev_patients: set[str]) -> None:
    print("\n=== Preflight 보고 ===")
    diag_size_gb = DIAG_CSV.stat().st_size / 1e9
    print(f"입력 파일: {DIAG_CSV.name} ({diag_size_gb:.1f}GB)")
    print(f"stage1_dev 환자 수: {len(dev_patients)}명")

    sample = next(pd.read_csv(DIAG_CSV, chunksize=CHUNKSIZE, encoding="utf-8-sig", low_memory=False))
    sample_filtered = sample[(sample["model_type"] == "v2v2") & (sample["patient_id"].isin(dev_patients))]
    sample_rate = len(sample_filtered) / len(sample) if len(sample) > 0 else 0
    total_rows_est = 11_749_900
    est_filtered = int(total_rows_est * sample_rate)
    est_chunks = int(total_rows_est / CHUNKSIZE) + 1
    est_mem_gb = est_filtered * 40 / 1e9

    print(f"예상 필터 후 행 수: ~{est_filtered:,}행 (샘플 비율 {sample_rate:.3f})")
    print(f"예상 청크 수: ~{est_chunks}개")
    print(f"예상 소요 시간: rank 계산 + GS2 mask + S6 sampling 포함 15~40분")
    print(f"메모리 위험: 중간")
    print(f"  - df 전체 + rank 컬럼은 메모리에 올라감. 예상 ~{est_mem_gb:.1f}GB")
    print(f"  - GS2 pool은 mask 기반, S6 variant는 metrics만 저장")
    print(f"\nS6 variant 목록 (총 7개):")
    for v in [
        "S6-A_positive_all_hn_ratio2  (positive 전부 + hn×2, 환자별 cap=600)",
        "S6-B_positive_all_hn_ratio3  (positive 전부 + hn×3, 환자별 cap=900)",
        "S6-C_patient_cap500          (환자별 cap=500, positive 우선, position_bin 다양성)",
        "S6-D_patient_cap800          (환자별 cap=800, positive 우선, position_bin 다양성)",
        "S6-E1_slice_balanced_max10   (slice별 max=10, positive 우선)",
        "S6-E2_slice_balanced_max15   (slice별 max=15, positive 우선)",
        "S6-F_size_sensitive_safe     (tiny/small cap=800, medium/large cap=500, ±2 slice 우선)",
    ]:
        print(f"  {v}")
    print(f"\n출력 예정 파일:")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_JSON}")
    print(f"  {OUT_MD}")
    print(f"\n실행 명령 초안:")
    print(f"  /home/jinhy/ai_env/bin/python3 scripts/rule_s6_gs2_sampling_design.py --run")
    print(f"\n[Preflight 완료] 실제 실행은 --run 플래그 추가 후 진행하세요.")

# ---------------------------------------------------------------------------
# Step 10: 출력 저장
# ---------------------------------------------------------------------------
def save_outputs(all_metrics: list[dict], f_cmp: dict, gs2_stats: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    flat_rows = []
    for m in all_metrics:
        gs2c = m.get("gs2_comparison", {})
        f5c = m.get("f5_comparison", {})
        pp = m.get("per_patient_stats", {})
        row = {
            "variant": m["variant"],
            "n_candidates": m["n_candidates"],
            "n_patients": m["n_patients"],
            "per_patient_min": pp.get("min"),
            "per_patient_median": pp.get("median"),
            "per_patient_mean": pp.get("mean"),
            "per_patient_max": pp.get("max"),
            "n_positive": m["n_positive"],
            "n_hard_negative": m["n_hard_negative"],
            "positive_ratio": m["positive_ratio"],
            "fp_ratio": m["fp_ratio"],
            "lesion_patch_recall": m["lesion_patch_recall"],
            "lesion_slice_recall": m["lesion_slice_recall"],
            "patient_hit_rate": m["patient_hit_rate"],
            "n_nohit": m["n_nohit"],
            "explosion_count": len(m.get("explosion_patients", [])),
        }
        for wp in WEAK_PATIENTS:
            row[f"weak_hit_{wp}"] = m["weak_patients_hit"].get(wp)
        for size in ["tiny", "small", "medium", "large"]:
            row[f"size_recall_{size}"] = m["patient_lesion_size_recall"].get(size)
        for cp in ["central", "peripheral"]:
            row[f"cp_recall_{cp}"] = m["central_peripheral_recall"].get(cp)
        row["reduction_rate_vs_gs2"] = gs2c.get("reduction_rate_vs_gs2")
        row["delta_recall_vs_gs2"] = gs2c.get("delta_recall_vs_gs2")
        row["delta_n_vs_f5"] = f5c.get("delta_n_vs_f5")
        row["delta_recall_vs_f5"] = f5c.get("delta_recall_vs_f5")
        row["crop_feasible"] = m["crop_feasible"]
        flat_rows.append(row)

    pd.DataFrame(flat_rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT_CSV}")

    def safe_json(obj: object) -> object:
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

    gs2_n = gs2_stats.get("n_candidates", 0)
    gs2_recall = gs2_stats.get("lesion_patch_recall", 0.0)
    f5_n = f_cmp.get("f5_n", 0)
    f5_recall = f_cmp.get("f5_recall", 0.0)

    lines = ["# Rule S6 GS2 Sampling Design 비교 보고서\n"]

    lines.append("## 해석 주의사항\n")
    lines.append("- GS2_slice_top30 pool (986,701개)에서 학습용으로 줄이는 sampling variant 비교이다.")
    lines.append("- positive/hard_negative 구분은 sampling 단계에서만 사용한다. 후보 pool 생성에는 병변 label 미사용.")
    lines.append("- patient_hit_rate 1.000 유지가 최우선 기준이다.")
    lines.append("- LUNG1-415, LUNG1-156 회수 유지 여부를 반드시 확인해야 한다.")
    lines.append("- stage2_holdout 환자는 이 분석에 포함되지 않는다.\n")

    lines.append("## 비교 기준값\n")
    lines.append("| 기준 | n_candidates | lesion_patch_recall |")
    lines.append("|------|-------------|---------------------|")
    lines.append(f"| GS2_slice_top30 | {gs2_n:,} | {gs2_recall:.6f} |")
    lines.append(f"| F5_F3_plus_D2_grid | {f5_n:,} | {f5_recall:.6f} |\n")

    lines.append("## S6 Variant 요약\n")
    lines.append("| variant | n_cand | n_pos | n_hn | pos_ratio | lp_recall | ls_recall | hit_rate | nohit | crop_ok | reduc_gs2 | Δrecall_gs2 | Δn_f5 | Δrecall_f5 |")
    lines.append("|---------|--------|-------|------|-----------|-----------|-----------|----------|-------|---------|-----------|-------------|-------|------------|")
    for m in all_metrics:
        gs2c = m.get("gs2_comparison", {})
        f5c = m.get("f5_comparison", {})
        rrgs2 = gs2c.get("reduction_rate_vs_gs2", 0.0)
        drgs2 = gs2c.get("delta_recall_vs_gs2", 0.0)
        dnf5 = f5c.get("delta_n_vs_f5", 0)
        drf5 = f5c.get("delta_recall_vs_f5", 0.0)
        lines.append(
            f"| {m['variant']} "
            f"| {m['n_candidates']:,} "
            f"| {m['n_positive']:,} "
            f"| {m['n_hard_negative']:,} "
            f"| {m['positive_ratio']:.4f} "
            f"| {m['lesion_patch_recall']:.4f} "
            f"| {m['lesion_slice_recall']:.4f} "
            f"| {m['patient_hit_rate']:.4f} "
            f"| {m['n_nohit']} "
            f"| {m['crop_feasible']} "
            f"| {rrgs2:.3f} "
            f"| {drgs2:+.4f} "
            f"| {int(dnf5):+,} "
            f"| {float(drf5):+.4f} |"
        )

    lines.append("\n## Weak Patient 회수 테이블\n")
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

        def fmt(v: object) -> str:
            return f"{float(v):.4f}" if v is not None else "N/A"

        lines.append(
            f"| {m['variant']} "
            f"| {fmt(sr.get('tiny'))} "
            f"| {fmt(sr.get('small'))} "
            f"| {fmt(sr.get('medium'))} "
            f"| {fmt(sr.get('large'))} |"
        )

    lines.append("\n## Central / Peripheral Recall\n")
    lines.append("| variant | central | peripheral |")
    lines.append("|---------|---------|------------|")
    for m in all_metrics:
        cp = m["central_peripheral_recall"]

        def fmt(v: object) -> str:
            return f"{float(v):.4f}" if v is not None else "N/A"

        lines.append(
            f"| {m['variant']} "
            f"| {fmt(cp.get('central'))} "
            f"| {fmt(cp.get('peripheral'))} |"
        )

    lines.append("\n## 환자별 후보 수 폭주 현황 (>2000개)\n")
    lines.append("| variant | explosion_count | explosion_patients |")
    lines.append("|---------|----------------|-------------------|")
    for m in all_metrics:
        ep = m.get("explosion_patients", [])
        lines.append(f"| {m['variant']} | {len(ep)} | {ep if ep else 'None'} |")

    lines.append("\n## 최종 추천 variant 선택 안내\n")
    lines.append("- 이 단계에서 variant manifest 파일은 저장하지 않았습니다.")
    lines.append("- manifest 생성은 사용자 승인 후 별도 진행합니다.")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"저장: {OUT_MD}")

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rule S6 GS2 Sampling Design (--run 없으면 preflight만)")
    parser.add_argument("--run", action="store_true", help="실제 분석 실행")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    guard_check()
    dev_patients, holdout_patients = load_stage_split()

    if not args.run:
        preflight_report(dev_patients)
        return

    df = load_diag_filtered(dev_patients, holdout_patients)
    df = compute_rank_scores(df)
    gc.collect()

    gs2_mask = build_gs2_mask(df)
    df_gs2 = df.loc[gs2_mask].copy()
    print(f"\n[GS2 pool] 총 {len(df_gs2):,}행, {df_gs2['patient_id'].nunique()}명")

    f_cmp = load_f_comparison()
    gs2_stats = load_gs2_stats()
    patient_size_bin = get_patient_size_bin(df)

    variant_configs = [
        ("S6-A_positive_all_hn_ratio2", lambda pool: sample_ratio_hn(pool, hn_ratio=2.0, patient_hn_cap=600)),
        ("S6-B_positive_all_hn_ratio3", lambda pool: sample_ratio_hn(pool, hn_ratio=3.0, patient_hn_cap=900)),
        ("S6-C_patient_cap500", lambda pool: sample_patient_cap(pool, cap=500)),
        ("S6-D_patient_cap800", lambda pool: sample_patient_cap(pool, cap=800)),
        ("S6-E1_slice_balanced_max10", lambda pool: sample_slice_balanced(pool, max_per_slice=10)),
        ("S6-E2_slice_balanced_max15", lambda pool: sample_slice_balanced(pool, max_per_slice=15)),
        ("S6-F_size_sensitive_safe", lambda pool: sample_size_sensitive_safe(pool, df)),
    ]

    all_metrics = []
    for name, sample_fn in variant_configs:
        print(f"\n  [{name}] sampling 중...")
        try:
            sampled_mask = sample_fn(df_gs2)
            df_sampled = df_gs2.loc[sampled_mask]
            print(f"    → {len(df_sampled):,}행")
            m = compute_sampling_metrics(df_sampled, df, name, f_cmp, gs2_stats, patient_size_bin)
            all_metrics.append(m)
            del df_sampled
            gc.collect()
        except Exception as e:
            print(f"  [오류] {name}: {e}")
            traceback.print_exc()

    del df_gs2
    gc.collect()

    save_outputs(all_metrics, f_cmp, gs2_stats)
    print(f"\n=== 완료 ===\n출력 디렉토리: {OUT_DIR}")


if __name__ == "__main__":
    main()
