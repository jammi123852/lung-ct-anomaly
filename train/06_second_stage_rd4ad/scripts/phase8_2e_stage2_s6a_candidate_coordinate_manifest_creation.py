"""
phase8_2e_stage2_s6a_candidate_coordinate_manifest_creation.py

stage2_holdout 환자 154명에 대해 기존 S6-A rule을 그대로 적용하여
candidate coordinate manifest를 생성한다.

실행:
  --run 없이          : dry-run 보고만 출력 후 종료
  --run --confirm-run : 실제 manifest 생성 실행

절대 금지:
- stage2_holdout 데이터 내용으로 threshold/rule 수정 금지
- 기존 DIAG_CSV 수정 금지
- 기존 Phase 6/7/8 output 수정/삭제/이동 금지
- lesion_mask 로드 금지
- npy/npz 로드 금지
- model forward 금지
- scoring 재계산 금지
- metric 계산 금지
- 새 coordinate 생성 금지 (y0/x0/y1/x1은 DIAG_CSV 값 그대로 사용)
- v1v2 row 사용 금지
- stage1_dev row 사용 금지
- pip/conda install 금지
"""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]

DIAG_CSV = REPO_ROOT / "outputs/position-aware-padim-v1/reports_v2_roi0_0_lesion/ratio_adjusted_score_full_diagnostic.csv"
SPLIT_CSV = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1.csv"

OUT_ROOT = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/review_annotations/phase8_2e_stage2_s6a_candidate_coordinate_manifest_creation_v1"
OUT_MANIFEST_FINAL = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/datasets/s6a_stage2_holdout_candidate_coordinate_manifest_v1.csv"
OUT_MANIFEST_TMP = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/datasets/s6a_stage2_holdout_candidate_coordinate_manifest_v1.csv.tmp"
OUT_REPORT = OUT_ROOT / "phase8_2e_stage2_s6a_candidate_coordinate_manifest_creation_report_v1.md"
OUT_SUMMARY = OUT_ROOT / "phase8_2e_stage2_s6a_candidate_coordinate_manifest_creation_summary_v1.json"
OUT_ERRORS = OUT_ROOT / "phase8_2e_stage2_s6a_candidate_coordinate_manifest_creation_errors_v1.csv"
OUT_RUNTIME = OUT_ROOT / "phase8_2e_stage2_s6a_candidate_coordinate_manifest_creation_runtime_summary_v1.csv"
OUT_DONE = OUT_ROOT / "phase8_2e_stage2_s6a_candidate_coordinate_manifest_creation_DONE.json"

# ---------------------------------------------------------------------------
# 상수 (기존 S6-A rule 그대로 재현)
# ---------------------------------------------------------------------------
V2V2_P95_THRESHOLD = 14.092057666455288
DEDUP_KEYS = ["patient_id", "local_z", "y0", "x0", "y1", "x1"]
CHUNKSIZE = 200_000

HN_RATIO = 2.0
PATIENT_HN_CAP = 600

# contamination 대상 환자
CONTAMINATION_PATIENTS = {"LUNG1-295", "LUNG1-415"}

# 출력 manifest 컬럼 순서 (스키마 정의)
MANIFEST_SCHEMA_COLS = [
    "row_id", "patient_id", "safe_id", "local_z", "y0", "x0", "y1", "x1",
    "label", "sampling_label", "stage_split", "model_type",
    "score_original", "score_valid950_weighted", "lesion_patch_ratio", "composite_rank_v2",
    "source_diag_csv", "asset_scope", "coordinate_source", "coordinate_rule", "sampling_rule",
    "contamination_check_status", "approval_required_before_crop_generation",
    "manifest_status", "issue", "note",
]

REQUIRED_DIAG_COLS = [
    "patient_id", "local_z", "y0", "x0", "y1", "x1",
    "model_type",
    "score_original", "score_valid950_weighted",
    "lesion_patch_ratio",
]

OPTIONAL_DIAG_COLS = [
    "safe_id",
    "slice_index",
    "group",
    "patch_label", "lesion_overlap", "sampling_label",
    "position_bin", "z_level", "central_peripheral",
    "roi_inside_ratio", "air_ratio_950", "air_ratio_970",
    "valid_ratio_roi_air950", "valid_ratio_roi_air970",
    "soft_tissue_ratio", "score_roi_weighted", "score_valid950_pow025",
    "score_valid950_floor025", "score_valid950_soft",
    "score_valid970_weighted",
    "composite_rank_v2",
    "patient_rank_original", "slice_rank_original",
    "patient_rank_valid950", "slice_rank_valid950",
]


# ---------------------------------------------------------------------------
# Guard: 출력 파일/디렉토리 존재 검사
# ---------------------------------------------------------------------------
def guard_check() -> None:
    check_targets = [
        OUT_ROOT, OUT_MANIFEST_FINAL, OUT_MANIFEST_TMP,
        OUT_REPORT, OUT_SUMMARY, OUT_ERRORS, OUT_RUNTIME, OUT_DONE,
    ]
    for target in check_targets:
        if target.exists():
            print(f"[중단] 출력 대상 이미 존재: {target}")
            print("  기존 파일/디렉토리를 삭제하거나 이름을 바꾼 후 재실행하세요.")
            sys.exit(1)

    # 입력 파일 존재 확인
    missing = []
    for f in [DIAG_CSV, SPLIT_CSV]:
        if not f.exists():
            missing.append(str(f))
    if missing:
        print("[중단] 입력 파일 없음:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    print("[Guard] 출력 파일 미존재 확인 완료")
    print("[Guard] 입력 파일 존재 확인 완료")


# ---------------------------------------------------------------------------
# Step 1: stage split 로드 → stage2_holdout 환자 + safe_id_map 반환
# ---------------------------------------------------------------------------
def load_stage_split() -> tuple[set[str], dict[str, str]]:
    df = pd.read_csv(SPLIT_CSV, encoding="utf-8-sig")
    if "stage_split" not in df.columns:
        print(f"[중단] stage_split 컬럼 없음. 실제 컬럼: {list(df.columns)}")
        sys.exit(1)
    if "patient_id" not in df.columns:
        print(f"[중단] patient_id 컬럼 없음. 실제 컬럼: {list(df.columns)}")
        sys.exit(1)

    holdout_df = df[df["stage_split"] == "stage2_holdout"]
    holdout_patients = set(holdout_df["patient_id"].tolist())

    n_dev = int((df["stage_split"] == "stage1_dev").sum())
    n_holdout = len(holdout_patients)

    print(f"[Stage Split] stage1_dev: {n_dev}명, stage2_holdout: {n_holdout}명")
    if n_holdout != 154:
        print(f"[중단] stage2_holdout 기대 154명, 실제 {n_holdout}명")
        sys.exit(1)

    # safe_id_map: stage2_holdout 환자 patient_id → safe_id (컬럼 있을 때만)
    safe_id_map: dict[str, str] = {}
    if "safe_id" in holdout_df.columns:
        for _, row in holdout_df.iterrows():
            pid = row["patient_id"]
            sid = row["safe_id"]
            if pd.notna(sid) and str(sid).strip() != "":
                safe_id_map[pid] = str(sid).strip()
        print(f"  safe_id_map 구성: {len(safe_id_map)}명 (split CSV 기준)")
    else:
        print("  SPLIT_CSV에 safe_id 컬럼 없음 → safe_id_map 비어 있음")

    return holdout_patients, safe_id_map


# ---------------------------------------------------------------------------
# Dry-run 보고
# ---------------------------------------------------------------------------
def dry_run_report(holdout_patients: set[str]) -> None:
    print("\n=== Dry-run 보고 (실제 실행 아님) ===")
    diag_size_gb = DIAG_CSV.stat().st_size / 1e9
    print(f"입력 DIAG_CSV : {DIAG_CSV.name} ({diag_size_gb:.1f} GB)")
    print(f"입력 SPLIT_CSV: {SPLIT_CSV.name}")
    print(f"stage2_holdout 환자 수: {len(holdout_patients)}명")

    # DIAG_CSV header 확인 (composite_rank_v2 존재 여부)
    first_row = pd.read_csv(DIAG_CSV, nrows=1, encoding="utf-8-sig")
    composite_rank_in_csv = "composite_rank_v2" in first_row.columns
    actual_optional_preview = [c for c in OPTIONAL_DIAG_COLS if c in first_row.columns]
    load_cols_preview = list(dict.fromkeys(REQUIRED_DIAG_COLS + actual_optional_preview))
    print(f"\n[DIAG_CSV header 확인]")
    print(f"  composite_rank_v2 존재: {composite_rank_in_csv}")
    print(f"  load_cols에 composite_rank_v2 포함: {'composite_rank_v2' in load_cols_preview}")
    if composite_rank_in_csv:
        print(f"  → has_existing_rank = True (기존값 사용, 재계산 생략)")
    else:
        print(f"  → has_existing_rank = False (DIAG_CSV에 없음, fallback 재계산 수행)")
    print(f"  rank 관련 optional 컬럼 (존재 시 로드): {[c for c in actual_optional_preview if 'rank' in c]}")
    print(f"\n생성 예정 파일:")
    print(f"  manifest final  : {OUT_MANIFEST_FINAL}")
    print(f"  manifest tmp    : {OUT_MANIFEST_TMP}")
    print(f"  report          : {OUT_REPORT}")
    print(f"  summary JSON    : {OUT_SUMMARY}")
    print(f"  errors CSV      : {OUT_ERRORS}")
    print(f"  runtime summary : {OUT_RUNTIME}")
    print(f"  DONE marker     : {OUT_DONE}")
    print(f"\nS6-A rule (기존 그대로 재현):")
    print(f"  GS2 pool : G0 (score_original >= {V2V2_P95_THRESHOLD}) | slice top30 per (patient_id, local_z)")
    print(f"  sampling : positive 전부 + hn x{HN_RATIO}, patient_hn_cap={PATIENT_HN_CAP}")
    print(f"  threshold: V2V2_P95_THRESHOLD = {V2V2_P95_THRESHOLD} (변경 없음)")
    print(f"\n금지사항 확인:")
    print(f"  - DIAG_CSV 수정 없음")
    print(f"  - npy/npz/mask 로드 없음")
    print(f"  - model forward 없음")
    print(f"  - scoring 재계산 없음")
    print(f"  - v1v2 row 사용 안함")
    print(f"  - stage1_dev row 사용 안함")
    print(f"\ncontamination 특이 환자: {sorted(CONTAMINATION_PATIENTS)}")
    print(f"\n실행 명령:")
    print(f"  source ~/ai_env/bin/activate && python scripts/phase8_2e_stage2_s6a_candidate_coordinate_manifest_creation.py --run --confirm-run")
    print(f"\n[Dry-run 완료] 실제 실행은 --run --confirm-run 둘 다 추가 후 진행하세요.")


# ---------------------------------------------------------------------------
# Step 4: DIAG_CSV chunk 로드 → stage2_holdout + v2v2 필터
# ---------------------------------------------------------------------------
def load_diag_filtered_stage2(holdout_patients: set[str], safe_id_map: dict[str, str]) -> pd.DataFrame:
    print(f"\n[Step 4] {DIAG_CSV.name} chunk 로드 시작 (chunksize={CHUNKSIZE:,})")

    # 첫 행으로 컬럼 존재 확인
    first_chunk = pd.read_csv(DIAG_CSV, nrows=1, encoding="utf-8-sig")
    missing_cols = [c for c in REQUIRED_DIAG_COLS if c not in first_chunk.columns]
    if missing_cols:
        print(f"[중단] 필수 컬럼 누락: {missing_cols}")
        sys.exit(1)
    print("  필수 컬럼 확인 완료")

    actual_optional = [c for c in OPTIONAL_DIAG_COLS if c in first_chunk.columns]
    load_cols = list(dict.fromkeys(REQUIRED_DIAG_COLS + actual_optional))
    composite_rank_in_csv = "composite_rank_v2" in first_chunk.columns
    print(f"  DIAG_CSV header composite_rank_v2 존재: {composite_rank_in_csv}")
    print(f"  load_cols에 composite_rank_v2 포함: {'composite_rank_v2' in load_cols}")
    if composite_rank_in_csv:
        print(f"  → has_existing_rank = True 예정 (재계산 생략)")
    else:
        print(f"  → has_existing_rank = False 예정 (fallback 재계산 수행)")
    print(f"  로드할 optional 컬럼: {actual_optional}")

    chunks = []
    total_read = 0
    total_filtered = 0

    for i, chunk in enumerate(
        pd.read_csv(DIAG_CSV, chunksize=CHUNKSIZE, encoding="utf-8-sig", low_memory=False)
    ):
        total_read += len(chunk)
        filtered = chunk[
            (chunk["model_type"] == "v2v2") &
            (chunk["patient_id"].isin(holdout_patients))
        ]
        if len(filtered) > 0:
            avail_cols = [c for c in load_cols if c in filtered.columns]
            chunks.append(filtered[avail_cols].copy())
            total_filtered += len(filtered)
        if (i + 1) % 10 == 0:
            print(f"  chunk {i+1} 처리 중... 읽은 행: {total_read:,}, 필터된 행: {total_filtered:,}")

    print(f"  전체 읽은 행: {total_read:,}, stage2_holdout+v2v2 필터 후: {total_filtered:,}")

    if not chunks:
        print("[중단] 필터 후 데이터 없음 (stage2_holdout v2v2 row가 DIAG_CSV에 없음)")
        sys.exit(1)

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    # stage1_dev row 포함 여부 확인 (금지사항)
    stage1_in = df[~df["patient_id"].isin(holdout_patients)]
    if len(stage1_in) > 0:
        print(f"[중단] stage1_dev row {len(stage1_in):,}행이 필터에서 누락 없이 포함됨")
        sys.exit(1)
    print("  stage1_dev row 미포함 확인 완료")

    # v1v2 row 포함 여부 확인 (금지사항)
    v1v2_in = df[df["model_type"] != "v2v2"]
    if len(v1v2_in) > 0:
        print(f"[중단] v1v2 row {len(v1v2_in):,}행이 포함됨")
        sys.exit(1)
    print("  v1v2 row 미포함 확인 완료")

    before_dedup = len(df)
    df = df.drop_duplicates(subset=DEDUP_KEYS)
    after_dedup = len(df)
    print(f"  중복 제거: {before_dedup:,} → {after_dedup:,} (제거 {before_dedup - after_dedup:,})")

    if after_dedup == 0:
        print("[중단] 중복 제거 후 0행")
        sys.exit(1)

    # stage_split 컬럼 부여
    df["stage_split"] = "stage2_holdout"

    # safe_id join: DIAG_CSV safe_id 없거나 empty → safe_id_map에서 채움
    # DIAG_CSV safe_id 있으면 safe_id_map과 일치 여부 확인
    if "safe_id" not in df.columns:
        df["safe_id"] = ""
    else:
        df["safe_id"] = df["safe_id"].fillna("").astype(str).str.strip()

    n_overwrite = 0
    n_filled = 0
    n_mismatch = 0
    for pid, sid_map in safe_id_map.items():
        mask = df["patient_id"] == pid
        existing = df.loc[mask, "safe_id"]
        if existing.empty:
            continue
        existing_vals = existing.unique()
        # 모두 비어 있으면 채움
        if len(existing_vals) == 1 and existing_vals[0] == "":
            df.loc[mask, "safe_id"] = sid_map
            n_filled += 1
        else:
            # 비어 있지 않은 값과 safe_id_map 불일치 확인
            non_empty = [v for v in existing_vals if v != ""]
            mismatched = [v for v in non_empty if v != sid_map]
            if mismatched:
                print(f"  [경고] safe_id 불일치: patient_id={pid}, diag={mismatched}, map={sid_map} → map 값으로 덮어씀")
                df.loc[mask, "safe_id"] = sid_map
                n_mismatch += 1
                n_overwrite += 1
            else:
                # 일치 or 일부 비어 있는 경우 map 값으로 통일
                df.loc[mask, "safe_id"] = sid_map

    print(f"  safe_id join: 신규 채움 {n_filled}명, 불일치 덮어씀 {n_mismatch}명")

    patient_count = df["patient_id"].nunique()
    print(f"  환자 수: {patient_count}명")
    print(f"  model_type 값: {sorted(df['model_type'].unique().tolist())}")

    return df


# ---------------------------------------------------------------------------
# Step 5: Rank score 계산 (기존 S6-A와 동일)
# ---------------------------------------------------------------------------
def compute_rank_scores(df: pd.DataFrame, has_existing_rank: bool) -> pd.DataFrame:
    print("\n[Step 5] Rank score 계산 시작")

    if has_existing_rank:
        print("  composite_rank_v2가 DIAG_CSV에 이미 존재 → 재계산 생략")
        return df

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
# Step 6: GS2 pool 구성 (기존 S6-A와 동일)
# ---------------------------------------------------------------------------
def build_gs2_mask(df: pd.DataFrame) -> pd.Series:
    print("\n[Step 6] GS2 pool 구성 시작")

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
# Positive 판정 (기존 S6-A와 동일)
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
# Step 7: S6-A Sampling (기존과 동일)
# ---------------------------------------------------------------------------
def sample_s6a_stage2(df_pool: pd.DataFrame) -> pd.Series:
    """S6-A: positive 전부 + hard_negative ratio 2배, 환자별 cap=600."""
    pos_mask = is_positive(df_pool)
    pos_idx = set(df_pool[pos_mask].index.tolist())
    n_total_pos = len(pos_idx)
    target_total_hn = int(n_total_pos * HN_RATIO)

    print(f"  S6-A sampling: positive {n_total_pos:,}개, target_hn {target_total_hn:,}개 (ratio={HN_RATIO}, cap={PATIENT_HN_CAP})")

    hn_df = df_pool[~pos_mask]

    per_patient_hn = []
    for pid, sub in hn_df.groupby("patient_id"):
        top_hn = sub.sort_values("composite_rank_v2", ascending=False).head(PATIENT_HN_CAP)
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


# ---------------------------------------------------------------------------
# Step 8: sampling_label 부여
# ---------------------------------------------------------------------------
def assign_sampling_label(df_sampled: pd.DataFrame) -> pd.DataFrame:
    pos_mask = is_positive(df_sampled)
    df_out = df_sampled.copy()
    df_out["sampling_label"] = "hard_negative"
    df_out.loc[pos_mask, "sampling_label"] = "positive"
    return df_out


# ---------------------------------------------------------------------------
# Step 9: manifest 스키마 컬럼 부여
# ---------------------------------------------------------------------------
def build_manifest(df_sampled: pd.DataFrame) -> pd.DataFrame:
    df_out = df_sampled.copy()

    # label: is_positive()와 동일 기준 (sampling_label과 정합성 보장)
    pos_mask = is_positive(df_out)
    df_out["label"] = pos_mask.astype(int)
    # sampling_label도 동일 pos_mask 기준으로 부여 (assign_sampling_label() 결과 덮어씀)
    df_out["sampling_label"] = "hard_negative"
    df_out.loc[pos_mask, "sampling_label"] = "positive"

    # contamination_check_status
    df_out["contamination_check_status"] = df_out["patient_id"].apply(
        lambda pid: (
            "coordinate_from_existing_stage2_diag_after_prior_crop_contamination"
            if pid in CONTAMINATION_PATIENTS
            else "coordinate_from_existing_stage2_diag"
        )
    )

    # 고정 컬럼
    df_out["stage_split"] = "stage2_holdout"
    df_out["model_type"] = "v2v2"
    df_out["asset_scope"] = "dedicated_stage2_holdout_candidate_coordinate_manifest"
    df_out["coordinate_source"] = "ratio_adjusted_score_full_diagnostic_csv_existing_stage2_v2v2_rows"
    df_out["coordinate_rule"] = "existing_diag_csv_coordinates_reused_without_change"
    df_out["sampling_rule"] = "existing_S6A_GS2_positive_all_hn_ratio2_reused_without_change"
    df_out["approval_required_before_crop_generation"] = True
    df_out["manifest_status"] = "created_after_phase8_2e_run"
    df_out["source_diag_csv"] = str(DIAG_CSV)
    df_out["issue"] = ""
    df_out["note"] = ""

    # safe_id: DIAG_CSV에 있으면 사용, 없으면 빈 문자열
    if "safe_id" not in df_out.columns:
        df_out["safe_id"] = ""

    # row_id: 0-based integer index
    df_out = df_out.reset_index(drop=True)
    df_out["row_id"] = df_out.index

    # 스키마 순서로 컬럼 정렬 (없는 컬럼은 건너뜀)
    final_cols = [c for c in MANIFEST_SCHEMA_COLS if c in df_out.columns]
    # 스키마에 없는 추가 컬럼은 뒤에 붙임
    extra_cols = [c for c in df_out.columns if c not in MANIFEST_SCHEMA_COLS]
    df_out = df_out[final_cols + extra_cols]

    return df_out


# ---------------------------------------------------------------------------
# Step 10: 검증
# ---------------------------------------------------------------------------
def validate_manifest(df_out: pd.DataFrame, error_records: list[dict]) -> bool:
    """20개 검증 항목 확인. 치명적 오류 발생 시 False 반환."""
    print("\n[Step 10] 검증 시작")
    is_fatal = False

    def add_error(check_id: str, message: str, fatal: bool = True) -> None:
        nonlocal is_fatal
        print(f"  [{'FATAL' if fatal else 'WARNING'}] {check_id}: {message}")
        error_records.append({
            "check_id": check_id,
            "fatal": fatal,
            "message": message,
        })
        if fatal:
            is_fatal = True

    # 1. stage2_holdout patient count == 154
    n_patients = df_out["patient_id"].nunique()
    if n_patients != 154:
        add_error("V01_patient_count", f"stage2_holdout 환자 수 기대 154, 실제 {n_patients}")
    else:
        print(f"  [OK] V01: stage2_holdout 환자 수 = {n_patients}")

    # 2. stage1_dev rows == 0
    n_dev_rows = int((df_out["stage_split"] == "stage1_dev").sum())
    if n_dev_rows > 0:
        add_error("V02_no_stage1_dev", f"stage1_dev row {n_dev_rows}행 포함")
    else:
        print(f"  [OK] V02: stage1_dev row 없음")

    # 3. v1v2 rows == 0
    n_v1v2 = int((df_out["model_type"] == "v1v2").sum())
    if n_v1v2 > 0:
        add_error("V03_no_v1v2", f"v1v2 row {n_v1v2}행 포함")
    else:
        print(f"  [OK] V03: v1v2 row 없음")

    # 4. model_type unique == {"v2v2"}
    mt_unique = set(df_out["model_type"].unique().tolist())
    if mt_unique != {"v2v2"}:
        add_error("V04_model_type", f"model_type unique = {mt_unique}, 기대 = {{'v2v2'}}")
    else:
        print(f"  [OK] V04: model_type = {mt_unique}")

    # 5. stage_split unique == {"stage2_holdout"}
    ss_unique = set(df_out["stage_split"].unique().tolist())
    if ss_unique != {"stage2_holdout"}:
        add_error("V05_stage_split", f"stage_split unique = {ss_unique}")
    else:
        print(f"  [OK] V05: stage_split = {ss_unique}")

    # 6. y0/x0/y1/x1 not null
    for col in ["y0", "x0", "y1", "x1"]:
        n_null = int(df_out[col].isna().sum())
        if n_null > 0:
            add_error(f"V06_coord_null_{col}", f"{col} null {n_null}행")
        else:
            print(f"  [OK] V06: {col} null 없음")

    # 7. y1 > y0 and x1 > x0
    bad_y = int((df_out["y1"] <= df_out["y0"]).sum())
    bad_x = int((df_out["x1"] <= df_out["x0"]).sum())
    if bad_y > 0:
        add_error("V07_coord_order_y", f"y1 <= y0인 행 {bad_y}개")
    else:
        print(f"  [OK] V07: y1 > y0 전부 만족")
    if bad_x > 0:
        add_error("V07_coord_order_x", f"x1 <= x0인 행 {bad_x}개")
    else:
        print(f"  [OK] V07: x1 > x0 전부 만족")

    # 8. (y1 - y0) == 32 and (x1 - x0) == 32 (patch_size 검증)
    bad_patch_h = int(((df_out["y1"] - df_out["y0"]) != 32).sum())
    bad_patch_w = int(((df_out["x1"] - df_out["x0"]) != 32).sum())
    if bad_patch_h > 0:
        add_error("V08_patch_size_h", f"(y1-y0) != 32인 행 {bad_patch_h}개")
    else:
        print(f"  [OK] V08: patch height == 32 전부 만족")
    if bad_patch_w > 0:
        add_error("V08_patch_size_w", f"(x1-x0) != 32인 행 {bad_patch_w}개")
    else:
        print(f"  [OK] V08: patch width == 32 전부 만족")

    # 9. local_z not null
    n_lz_null = int(df_out["local_z"].isna().sum())
    if n_lz_null > 0:
        add_error("V09_local_z_null", f"local_z null {n_lz_null}행")
    else:
        print(f"  [OK] V09: local_z null 없음")

    # 10. label values in {0, 1}
    label_unique = set(df_out["label"].unique().tolist())
    if not label_unique.issubset({0, 1}):
        add_error("V10_label_values", f"label 값 = {label_unique}, 기대 subset of {{0, 1}}")
    else:
        print(f"  [OK] V10: label 값 = {label_unique}")

    # 11. sampling_label values in {"positive", "hard_negative"}
    sl_unique = set(df_out["sampling_label"].unique().tolist())
    if not sl_unique.issubset({"positive", "hard_negative"}):
        add_error("V11_sampling_label", f"sampling_label 값 = {sl_unique}")
    else:
        print(f"  [OK] V11: sampling_label 값 = {sl_unique}")

    # 12. positive count > 0
    n_pos = int((df_out["sampling_label"] == "positive").sum())
    if n_pos == 0:
        add_error("V12_positive_count", "positive 행이 0개")
    else:
        print(f"  [OK] V12: positive count = {n_pos:,}")

    # 13. hard_negative count > 0
    n_hn = int((df_out["sampling_label"] == "hard_negative").sum())
    if n_hn == 0:
        add_error("V13_hn_count", "hard_negative 행이 0개")
    else:
        print(f"  [OK] V13: hard_negative count = {n_hn:,}")

    # 14. hard_negative count <= positive count × 2 (warning만)
    if n_pos > 0 and n_hn > n_pos * 2:
        add_error("V14_hn_ratio", f"hard_negative {n_hn:,} > positive×2 {n_pos*2:,}", fatal=False)
    else:
        print(f"  [OK] V14: hn ratio 확인 (hn={n_hn:,}, pos×2={n_pos*2:,})")

    # 15. patient hard_negative cap <= 600
    hn_df = df_out[df_out["sampling_label"] == "hard_negative"]
    if len(hn_df) > 0:
        per_patient_hn = hn_df.groupby("patient_id").size()
        max_hn_per_patient = int(per_patient_hn.max())
        over_cap = per_patient_hn[per_patient_hn > PATIENT_HN_CAP]
        if len(over_cap) > 0:
            add_error("V15_hn_cap", f"환자별 hn cap {PATIENT_HN_CAP} 초과 환자 {len(over_cap)}명: {sorted(over_cap.index.tolist())}")
        else:
            print(f"  [OK] V15: 환자별 hn cap <= {PATIENT_HN_CAP} (max={max_hn_per_patient})")
    else:
        print(f"  [OK] V15: hard_negative 행 없음 (skip)")

    # 16. LUNG1-295 포함 확인
    if "LUNG1-295" not in set(df_out["patient_id"].unique()):
        add_error("V16_lung1_295", "LUNG1-295가 manifest에 없음")
    else:
        print(f"  [OK] V16: LUNG1-295 포함")

    # 17. LUNG1-415 포함 확인
    if "LUNG1-415" not in set(df_out["patient_id"].unique()):
        add_error("V17_lung1_415", "LUNG1-415가 manifest에 없음")
    else:
        print(f"  [OK] V17: LUNG1-415 포함")

    # 18. approval_required_before_crop_generation == True (모든 행)
    if "approval_required_before_crop_generation" in df_out.columns:
        n_not_true = int((df_out["approval_required_before_crop_generation"] != True).sum())
        if n_not_true > 0:
            add_error("V18_approval_flag", f"approval_required_before_crop_generation != True인 행 {n_not_true}개")
        else:
            print(f"  [OK] V18: approval_required_before_crop_generation == True 전부")
    else:
        add_error("V18_approval_flag", "approval_required_before_crop_generation 컬럼 없음")

    # 19. duplicate row_id 없음
    n_dup_row_id = int(df_out["row_id"].duplicated().sum())
    if n_dup_row_id > 0:
        add_error("V19_dup_row_id", f"duplicate row_id {n_dup_row_id}개")
    else:
        print(f"  [OK] V19: duplicate row_id 없음")

    # 20. duplicate (patient_id, local_z, y0, x0, y1, x1) 보고
    dup_coord = df_out.duplicated(subset=["patient_id", "local_z", "y0", "x0", "y1", "x1"]).sum()
    print(f"  [INFO] V20: duplicate coordinate 수 = {int(dup_coord)}")
    if dup_coord > 0:
        add_error("V20_dup_coord", f"duplicate (patient_id, local_z, y0, x0, y1, x1) {int(dup_coord)}개", fatal=False)

    # 21. safe_id null 또는 empty string인 행 수 → fatal error
    if "safe_id" in df_out.columns:
        n_safe_id_empty = int(
            df_out["safe_id"].isna().sum() + (df_out["safe_id"].fillna("").astype(str).str.strip() == "").sum()
        )
        if n_safe_id_empty > 0:
            add_error("V21_safe_id_empty", f"safe_id null 또는 empty인 행 {n_safe_id_empty}개")
        else:
            print(f"  [OK] V21: safe_id empty 없음")
    else:
        add_error("V21_safe_id_empty", "safe_id 컬럼 없음")

    # 22. LUNG1-295 safe_id non-empty 확인
    lung295 = df_out[df_out["patient_id"] == "LUNG1-295"]
    if len(lung295) == 0:
        add_error("V22_lung1_295_safe_id", "LUNG1-295가 manifest에 없어 safe_id 확인 불가")
    else:
        n_empty_295 = int(
            lung295["safe_id"].isna().sum() + (lung295["safe_id"].fillna("").astype(str).str.strip() == "").sum()
        )
        if n_empty_295 > 0:
            add_error("V22_lung1_295_safe_id", f"LUNG1-295 safe_id empty/null 행 {n_empty_295}개")
        else:
            print(f"  [OK] V22: LUNG1-295 safe_id non-empty")

    # 23. LUNG1-415 safe_id non-empty 확인
    lung415 = df_out[df_out["patient_id"] == "LUNG1-415"]
    if len(lung415) == 0:
        add_error("V23_lung1_415_safe_id", "LUNG1-415가 manifest에 없어 safe_id 확인 불가")
    else:
        n_empty_415 = int(
            lung415["safe_id"].isna().sum() + (lung415["safe_id"].fillna("").astype(str).str.strip() == "").sum()
        )
        if n_empty_415 > 0:
            add_error("V23_lung1_415_safe_id", f"LUNG1-415 safe_id empty/null 행 {n_empty_415}개")
        else:
            print(f"  [OK] V23: LUNG1-415 safe_id non-empty")

    # 24. stage2_holdout 154명 전원 safe_id non-empty 확인
    if "safe_id" in df_out.columns:
        per_patient_safe = df_out.groupby("patient_id")["safe_id"].apply(
            lambda s: (s.isna() | (s.fillna("").astype(str).str.strip() == "")).any()
        )
        patients_with_empty_safe_id = per_patient_safe[per_patient_safe].index.tolist()
        if patients_with_empty_safe_id:
            add_error(
                "V24_all_patients_safe_id",
                f"safe_id empty/null 환자 {len(patients_with_empty_safe_id)}명: {sorted(patients_with_empty_safe_id)[:10]}..."
                if len(patients_with_empty_safe_id) > 10
                else f"safe_id empty/null 환자 {len(patients_with_empty_safe_id)}명: {sorted(patients_with_empty_safe_id)}",
            )
        else:
            print(f"  [OK] V24: stage2_holdout 전원 safe_id non-empty")
    else:
        add_error("V24_all_patients_safe_id", "safe_id 컬럼 없음")

    # V_LABEL_POS: sampling_label == "positive"이면 label == 1 전부 만족
    if "sampling_label" in df_out.columns and "label" in df_out.columns:
        pos_rows = df_out[df_out["sampling_label"] == "positive"]
        n_label_mismatch_pos = int((pos_rows["label"] != 1).sum())
        if n_label_mismatch_pos > 0:
            add_error("V_LABEL_POS", f"sampling_label==positive인데 label!=1인 행 {n_label_mismatch_pos}개")
        else:
            print(f"  [OK] V_LABEL_POS: positive → label==1 전부 만족")

        # V_LABEL_HN: sampling_label == "hard_negative"이면 label == 0 전부 만족
        hn_rows = df_out[df_out["sampling_label"] == "hard_negative"]
        n_label_mismatch_hn = int((hn_rows["label"] != 0).sum())
        if n_label_mismatch_hn > 0:
            add_error("V_LABEL_HN", f"sampling_label==hard_negative인데 label!=0인 행 {n_label_mismatch_hn}개")
        else:
            print(f"  [OK] V_LABEL_HN: hard_negative → label==0 전부 만족")

    return not is_fatal


# ---------------------------------------------------------------------------
# 저장 함수들
# ---------------------------------------------------------------------------
def safe_json_serialize(obj: object) -> object:
    if isinstance(obj, dict):
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json_serialize(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, bool):
        return obj
    return obj


def save_report(df_out: pd.DataFrame, error_records: list[dict], start_time: float, composite_rank_source: str = "") -> None:
    n_total = len(df_out)
    n_patients = df_out["patient_id"].nunique()
    n_pos = int((df_out["sampling_label"] == "positive").sum())
    n_hn = int((df_out["sampling_label"] == "hard_negative").sum())
    elapsed = time.time() - start_time

    lines = ["# Phase 8.2E Stage2 S6-A Candidate Coordinate Manifest 생성 보고서\n"]
    lines.append(f"생성 일시: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("## 기본 정보\n")
    lines.append(f"- stage_split: stage2_holdout")
    lines.append(f"- model_type: v2v2")
    lines.append(f"- 적용 rule: S6-A (기존 그대로 재현)")
    lines.append(f"- V2V2_P95_THRESHOLD: {V2V2_P95_THRESHOLD}")
    lines.append(f"- HN_RATIO: {HN_RATIO}, PATIENT_HN_CAP: {PATIENT_HN_CAP}")
    lines.append(f"- composite_rank_source: {composite_rank_source}\n")

    lines.append("## 결과 요약\n")
    lines.append(f"| 항목 | 값 |")
    lines.append(f"|------|-----|")
    lines.append(f"| 총 manifest 행 수 | {n_total:,} |")
    lines.append(f"| 환자 수 | {n_patients} |")
    lines.append(f"| positive 행 수 | {n_pos:,} |")
    lines.append(f"| hard_negative 행 수 | {n_hn:,} |")
    lines.append(f"| 소요 시간 | {elapsed:.1f}초 |\n")

    lines.append("## 검증 오류 목록\n")
    if error_records:
        lines.append(f"총 {len(error_records)}개 오류/경고\n")
        lines.append("| check_id | fatal | message |")
        lines.append("|----------|-------|---------|")
        for e in error_records:
            lines.append(f"| {e['check_id']} | {e['fatal']} | {e['message']} |")
    else:
        lines.append("검증 오류 없음 (전체 통과)")

    lines.append("\n## contamination 특이 환자\n")
    lines.append(f"- LUNG1-295: coordinate_from_existing_stage2_diag_after_prior_crop_contamination")
    lines.append(f"- LUNG1-415: coordinate_from_existing_stage2_diag_after_prior_crop_contamination")

    lines.append("\n## 출력 파일\n")
    lines.append(f"- manifest: {OUT_MANIFEST_FINAL}")
    lines.append(f"- report: {OUT_REPORT}")
    lines.append(f"- summary: {OUT_SUMMARY}")
    lines.append(f"- errors: {OUT_ERRORS}")
    lines.append(f"- runtime: {OUT_RUNTIME}")
    lines.append(f"- DONE: {OUT_DONE}")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"  저장: {OUT_REPORT}")


def save_summary(df_out: pd.DataFrame, error_records: list[dict], start_time: float, composite_rank_source: str = "") -> None:
    n_total = len(df_out)
    n_patients = df_out["patient_id"].nunique()
    n_pos = int((df_out["sampling_label"] == "positive").sum())
    n_hn = int((df_out["sampling_label"] == "hard_negative").sum())
    elapsed = time.time() - start_time

    per_patient = df_out.groupby("patient_id").size()
    summary = {
        "script": "phase8_2e_stage2_s6a_candidate_coordinate_manifest_creation.py",
        "created_at": datetime.datetime.now().isoformat(),
        "stage_split": "stage2_holdout",
        "model_type": "v2v2",
        "sampling_rule": "existing_S6A_GS2_positive_all_hn_ratio2_reused_without_change",
        "V2V2_P95_THRESHOLD": V2V2_P95_THRESHOLD,
        "HN_RATIO": HN_RATIO,
        "PATIENT_HN_CAP": PATIENT_HN_CAP,
        "composite_rank_source": composite_rank_source,
        "n_total_rows": n_total,
        "n_patients": n_patients,
        "n_positive": n_pos,
        "n_hard_negative": n_hn,
        "hn_ratio_actual": round(n_hn / n_pos, 6) if n_pos > 0 else None,
        "per_patient_min": int(per_patient.min()) if len(per_patient) > 0 else 0,
        "per_patient_median": float(per_patient.median()) if len(per_patient) > 0 else 0.0,
        "per_patient_mean": float(per_patient.mean()) if len(per_patient) > 0 else 0.0,
        "per_patient_max": int(per_patient.max()) if len(per_patient) > 0 else 0,
        "contamination_patients": sorted(CONTAMINATION_PATIENTS),
        "validation_errors": error_records,
        "n_validation_errors": len(error_records),
        "elapsed_seconds": round(elapsed, 2),
        "manifest_path": str(OUT_MANIFEST_FINAL),
        "approval_required_before_crop_generation": True,
    }

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(safe_json_serialize(summary), f, ensure_ascii=False, indent=2)
    print(f"  저장: {OUT_SUMMARY}")


def save_errors(error_records: list[dict]) -> None:
    if error_records:
        pd.DataFrame(error_records).to_csv(OUT_ERRORS, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["check_id", "fatal", "message"]).to_csv(OUT_ERRORS, index=False, encoding="utf-8-sig")
    print(f"  저장: {OUT_ERRORS} ({len(error_records)}건)")


def save_runtime_summary(start_time: float, n_total: int) -> None:
    elapsed = time.time() - start_time
    df_rt = pd.DataFrame([{
        "script": "phase8_2e_stage2_s6a_candidate_coordinate_manifest_creation.py",
        "start_time": datetime.datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "n_output_rows": n_total,
        "status": "DONE",
    }])
    df_rt.to_csv(OUT_RUNTIME, index=False, encoding="utf-8-sig")
    print(f"  저장: {OUT_RUNTIME}")


def save_done_marker(start_time: float, n_total: int) -> None:
    done_info = {
        "script": "phase8_2e_stage2_s6a_candidate_coordinate_manifest_creation.py",
        "completed_at": datetime.datetime.now().isoformat(),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "n_output_rows": n_total,
        "manifest_path": str(OUT_MANIFEST_FINAL),
        "status": "DONE",
    }
    with open(OUT_DONE, "w", encoding="utf-8") as f:
        json.dump(done_info, f, ensure_ascii=False, indent=2)
    print(f"  저장: {OUT_DONE}")


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 8.2E: stage2_holdout S6-A candidate coordinate manifest 생성"
    )
    parser.add_argument("--run", action="store_true", help="실행 플래그 (--confirm-run도 필요)")
    parser.add_argument("--confirm-run", action="store_true", help="실행 확인 플래그 (--run도 필요)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    start_time = time.time()

    # Guard: 출력 파일 존재 검사
    guard_check()

    # Stage split 로드
    holdout_patients, safe_id_map = load_stage_split()

    # --run AND --confirm-run 없으면 dry-run만
    if not (args.run and args.confirm_run):
        dry_run_report(holdout_patients)
        return

    print("\n=== Phase 8.2E 실제 실행 시작 ===")

    # Step 4: DIAG_CSV 로드 (stage2_holdout + v2v2 필터)
    df = load_diag_filtered_stage2(holdout_patients, safe_id_map)

    # Step 5: Rank score 계산 (DIAG_CSV에 composite_rank_v2가 있으면 재계산 생략)
    has_existing_rank = "composite_rank_v2" in df.columns
    composite_rank_source = (
        "existing_diag_csv_column" if has_existing_rank
        else "recomputed_by_existing_formula_because_missing"
    )
    print(f"  composite_rank_source: {composite_rank_source}")
    df = compute_rank_scores(df, has_existing_rank=has_existing_rank)
    gc.collect()

    # Step 6: GS2 pool 구성
    gs2_mask = build_gs2_mask(df)
    df_gs2 = df.loc[gs2_mask].copy()
    print(f"\n[GS2 pool] 총 {len(df_gs2):,}행, {df_gs2['patient_id'].nunique()}명")

    # Step 7: S6-A sampling
    print("\n[Step 7] S6-A sampling 시작")
    sampled_mask = sample_s6a_stage2(df_gs2)
    df_sampled = df_gs2.loc[sampled_mask].copy()
    del df_gs2
    gc.collect()
    print(f"  sampled: {len(df_sampled):,}행")

    # Step 8: sampling_label 부여
    df_sampled = assign_sampling_label(df_sampled)

    # Step 9: manifest 스키마 컬럼 부여
    print("\n[Step 9] Manifest 스키마 적용 시작")
    df_out = build_manifest(df_sampled)
    del df_sampled, df
    gc.collect()
    print(f"  manifest 행 수: {len(df_out):,}")

    # Step 10: 검증
    error_records: list[dict] = []
    validation_passed = validate_manifest(df_out, error_records)

    # OUT_ROOT 생성 (존재하면 에러 → guard_check에서 이미 확인)
    OUT_ROOT.mkdir(parents=True, exist_ok=False)
    datasets_dir = OUT_MANIFEST_FINAL.parent
    datasets_dir.mkdir(parents=True, exist_ok=True)

    if not validation_passed:
        print(f"\n[중단] 치명적 검증 오류 {sum(1 for e in error_records if e['fatal'])}건 발생")
        print("  오류 내역:")
        for e in error_records:
            if e["fatal"]:
                print(f"    - {e['check_id']}: {e['message']}")
        # errors CSV 저장 후 종료 (manifest는 생성하지 않음)
        save_errors(error_records)
        save_runtime_summary(start_time, 0)
        sys.exit(1)

    # 저장: tmp 먼저
    print(f"\n[저장] tmp manifest 저장 중...")
    df_out.to_csv(OUT_MANIFEST_TMP, index=False, encoding="utf-8-sig")
    print(f"  tmp 저장 완료: {OUT_MANIFEST_TMP} ({len(df_out):,}행)")

    # 검증 통과 → tmp → final rename
    OUT_MANIFEST_TMP.rename(OUT_MANIFEST_FINAL)
    print(f"  rename 완료: {OUT_MANIFEST_TMP.name} → {OUT_MANIFEST_FINAL.name}")

    # 보고서, summary, errors, runtime 저장
    print(f"\n[저장] 보조 파일 저장 중...")
    save_report(df_out, error_records, start_time, composite_rank_source=composite_rank_source)
    save_summary(df_out, error_records, start_time, composite_rank_source=composite_rank_source)
    save_errors(error_records)
    save_runtime_summary(start_time, len(df_out))

    # DONE marker 마지막에 저장
    save_done_marker(start_time, len(df_out))

    elapsed = time.time() - start_time
    print(f"\n=== Phase 8.2E 완료 ===")
    print(f"  총 소요 시간: {elapsed:.1f}초")
    print(f"  manifest: {OUT_MANIFEST_FINAL} ({len(df_out):,}행)")
    print(f"  출력 디렉토리: {OUT_ROOT}")


if __name__ == "__main__":
    main()
