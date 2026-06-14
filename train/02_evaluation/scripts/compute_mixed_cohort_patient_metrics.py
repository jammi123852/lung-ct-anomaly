"""
compute_mixed_cohort_patient_metrics.py

목적:
  Mixed cohort (정상 negative + 병변 positive) 기반 patient-level metric 계산.
  기존 score CSV를 read-only 전제로 사용. 신규 scoring/model forward/training 금지.

비교 대상:
  - v1_v1: v1 model + v1 lesion test. 정상 score는 padim_v1/by_patient/.
  - v1_v2: v1 model + v2 lesion test. 정상 score는 padim_v1/by_patient/ (동일 경로 — 공정 비교).
  - v2_v2: v2 model + v2 lesion test. 정상 score는 padim_v2_roi0_0/normal_by_patient/.
  [핵심 비교축] v1_v2 vs v2_v2: 같은 v2 lesion test 기준으로 model 차이 비교.
  [참고 비교축] v1_v1 vs v1_v2 vs v2_v2: 3자 참고.

Cohort 정책:
  - dev_safe (기본): 정상 test 36명 + 병변 stage1_dev 154명 = 190명.
    stage2_holdout 발견 시 즉시 abort.
    use_scope = development_only
  - full_retrospective (기본 차단): 정상 test 36명 + 병변 전체 308명 = 344명.
    --confirm-full-retrospective 없으면 abort.
    use_scope = quarantine_reference_only

실행 금지:
  이번 단계(코드 작성 단계)에서 스크립트 실행 금지.
  허용 검증: python -m py_compile scripts/compute_mixed_cohort_patient_metrics.py

이 스크립트 실행 시에는 반드시 사용자 승인 후 실행하고,
결과 저장 시 --confirm-run 플래그가 필요하다.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# AUROC / AUPRC helper — 기존 검증 구현 기반
# 원본: scripts/compute_lesion_metrics_fast.py (fast_auroc / fast_auprc)
# sklearn 미사용. numpy만 사용.
# ---------------------------------------------------------------------------

def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mann-Whitney U 기반 AUROC. tie는 average rank로 처리. trapz ROC와 동일값.
    기존 검증 구현 기반 — 원본: scripts/compute_lesion_metrics_fast.py fast_auroc"""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # average rank (tie 평균) — np.unique로 완전 벡터화
    _, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    start = cum - counts            # 0-based 시작 위치
    avg_rank_per_uniq = (start + 1 + cum) / 2.0   # 1-based 평균 rank
    ranks = avg_rank_per_uniq[inv]
    sum_pos = ranks[y_true == 1].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """내림차순 누적 기반 AUPRC (trapz). O(N log N).
    기존 검증 구현 기반 — 원본: scripts/compute_lesion_metrics_fast.py fast_auprc"""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    yt = (y_true[order] == 1).astype(np.int64)
    tp = np.cumsum(yt)
    fp = np.cumsum(1 - yt)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.trapz(precision, recall))

# ---------------------------------------------------------------------------
# 상수 — repo 내 known path
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

# 정상 score 경로
NORMAL_SCORE_DIR_V1 = REPO_ROOT / "outputs/position-aware-padim-v1/scores/padim_v1/by_patient"
NORMAL_SCORE_DIR_V2 = REPO_ROOT / "outputs/position-aware-padim-v1/scores/padim_v2_roi0_0/normal_by_patient"

# 병변 score 경로
LESION_SCORE_DIR_V1V1 = REPO_ROOT / "outputs/position-aware-padim-v1/scores/padim_v1/lesion_by_patient"
LESION_SCORE_DIR_V1V2 = REPO_ROOT / "outputs/position-aware-padim-v1/scores/padim_v1/lesion_v2_by_patient"
LESION_SCORE_DIR_V2V2 = REPO_ROOT / "outputs/position-aware-padim-v1/scores/padim_v2_roi0_0/lesion_v2_by_patient"

# Split CSV
NORMAL_SPLIT_CSV = REPO_ROOT / "data/normal_training_ready/manifests/train_val_test_split.csv"
LESION_STAGE_SPLIT_CSV = REPO_ROOT / "outputs/second-stage-lesion-refiner-v1/splits/lesion_stage_split_v1.csv"

# Threshold JSON 경로 — 파일 없으면 NEEDS_THRESHOLD_CONFIRMATION 처리
THRESHOLD_JSON_V1 = REPO_ROOT / "outputs/position-aware-padim-v1/evaluation/lesion_subset/lesion_eval_p95_fast_summary.json"
THRESHOLD_JSON_V2 = REPO_ROOT / "outputs/position-aware-padim-v1/evaluation/normal_v2_roi0_0/normal_v2_threshold.json"

# v1 p99 키가 lesion_eval_p95_fast_summary.json에 존재하지 않음(확인됨).
# v1 p99는 별도 파일에서 확인 필요. 아래 상수로 명시하고 코드에서 처리.
V1_THRESHOLD_P99_KEY = "NEEDS_THRESHOLD_CONFIRMATION"
# v1 p95 키
V1_THRESHOLD_P95_KEY = "threshold_value"

# v2 threshold JSON 키
V2_THRESHOLD_P95_KEY = "threshold_p95"
V2_THRESHOLD_P99_KEY = "threshold_p99"

# score 컬럼명 — 없으면 abort
SCORE_COL = "padim_score"

# slice 구분자 컬럼 — 없으면 patient_max_slice_score = UNAVAILABLE
SLICE_COL_CANDIDATES = ["local_z", "slice_index"]

# ---------------------------------------------------------------------------
# Threshold 로드 함수
# ---------------------------------------------------------------------------

def load_v1_thresholds():
    """
    v1 모델 threshold 로드.
    Returns: dict with keys 'p95' (float or None), 'p99' (float or None),
             'p95_status' (str), 'p99_status' (str), 'source_path' (str)
    """
    result = {
        "p95": None,
        "p99": None,
        "p95_status": "NEEDS_THRESHOLD_CONFIRMATION",
        "p99_status": "NEEDS_THRESHOLD_CONFIRMATION",
        "source_path": str(THRESHOLD_JSON_V1),
    }
    if not THRESHOLD_JSON_V1.exists():
        result["p95_status"] = f"NEEDS_THRESHOLD_CONFIRMATION: file not found: {THRESHOLD_JSON_V1}"
        result["p99_status"] = f"NEEDS_THRESHOLD_CONFIRMATION: file not found: {THRESHOLD_JSON_V1}"
        return result
    with open(THRESHOLD_JSON_V1, "r", encoding="utf-8") as f:
        data = json.load(f)
    # p95
    if V1_THRESHOLD_P95_KEY in data:
        result["p95"] = float(data[V1_THRESHOLD_P95_KEY])
        result["p95_status"] = "confirmed"
    else:
        result["p95_status"] = (
            f"NEEDS_THRESHOLD_CONFIRMATION: key '{V1_THRESHOLD_P95_KEY}' "
            f"not found in {THRESHOLD_JSON_V1}"
        )
    # p99 — 이 파일에 p99 키 없음(설계 문서 확인). 명시적으로 NEEDS_THRESHOLD_CONFIRMATION.
    result["p99_status"] = (
        "NEEDS_THRESHOLD_CONFIRMATION: v1 p99 not present in "
        f"{THRESHOLD_JSON_V1}. Confirm source file for v1 p99 threshold."
    )
    return result


def load_v2_thresholds():
    """
    v2 모델 threshold 로드.
    Returns: dict with keys 'p95', 'p99', 'p95_status', 'p99_status', 'source_path'
    """
    result = {
        "p95": None,
        "p99": None,
        "p95_status": "NEEDS_THRESHOLD_CONFIRMATION",
        "p99_status": "NEEDS_THRESHOLD_CONFIRMATION",
        "source_path": str(THRESHOLD_JSON_V2),
    }
    if not THRESHOLD_JSON_V2.exists():
        result["p95_status"] = f"NEEDS_THRESHOLD_CONFIRMATION: file not found: {THRESHOLD_JSON_V2}"
        result["p99_status"] = f"NEEDS_THRESHOLD_CONFIRMATION: file not found: {THRESHOLD_JSON_V2}"
        return result
    with open(THRESHOLD_JSON_V2, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key, level in [(V2_THRESHOLD_P95_KEY, "p95"), (V2_THRESHOLD_P99_KEY, "p99")]:
        if key in data:
            result[level] = float(data[key])
            result[f"{level}_status"] = "confirmed"
        else:
            result[f"{level}_status"] = (
                f"NEEDS_THRESHOLD_CONFIRMATION: key '{key}' not found in {THRESHOLD_JSON_V2}"
            )
    return result


# ---------------------------------------------------------------------------
# Split 로드 함수
# ---------------------------------------------------------------------------

def load_normal_test_patient_ids():
    """
    정상 test split 환자 ID 목록 로드.
    Returns: list of str (patient_id)
    """
    if not NORMAL_SPLIT_CSV.exists():
        raise FileNotFoundError(
            f"[ABORT] 정상 split CSV를 찾을 수 없음: {NORMAL_SPLIT_CSV}\n"
            "NEEDS_PATH_CONFIRMATION: NORMAL_SPLIT_CSV 경로를 확인하세요."
        )
    df = pd.read_csv(NORMAL_SPLIT_CSV)
    df.columns = df.columns.str.lstrip("﻿")  # BOM 제거
    if "patient_id" not in df.columns:
        raise ValueError(
            f"[ABORT] 'patient_id' 컬럼 없음: {NORMAL_SPLIT_CSV} (컬럼: {list(df.columns)})"
        )
    if "split" not in df.columns:
        raise ValueError(
            f"[ABORT] 'split' 컬럼 없음: {NORMAL_SPLIT_CSV} (컬럼: {list(df.columns)})"
        )
    test_ids = df.loc[df["split"] == "test", "patient_id"].astype(str).tolist()
    return test_ids


def load_lesion_stage_patient_ids(include_stage2_holdout: bool = False):
    """
    병변 stage split CSV 로드.
    Returns:
        stage1_dev_ids: list of str
        stage2_holdout_ids: list of str
    """
    if not LESION_STAGE_SPLIT_CSV.exists():
        raise FileNotFoundError(
            f"[ABORT] 병변 stage split CSV를 찾을 수 없음: {LESION_STAGE_SPLIT_CSV}\n"
            "NEEDS_PATH_CONFIRMATION: LESION_STAGE_SPLIT_CSV 경로를 확인하세요."
        )
    df = pd.read_csv(LESION_STAGE_SPLIT_CSV)
    df.columns = df.columns.str.lstrip("﻿")  # BOM 제거
    for col in ["patient_id", "stage_split"]:
        if col not in df.columns:
            raise ValueError(
                f"[ABORT] '{col}' 컬럼 없음: {LESION_STAGE_SPLIT_CSV} (컬럼: {list(df.columns)})"
            )
    stage1_ids = df.loc[df["stage_split"] == "stage1_dev", "patient_id"].astype(str).tolist()
    stage2_ids = df.loc[df["stage_split"] == "stage2_holdout", "patient_id"].astype(str).tolist()
    return stage1_ids, stage2_ids


# ---------------------------------------------------------------------------
# Leakage guard
# ---------------------------------------------------------------------------

def check_no_stage2_in_list(patient_ids: list, stage2_ids: list, context: str):
    """
    patient_ids에 stage2_holdout id가 포함되어 있으면 즉시 abort.
    """
    stage2_set = set(stage2_ids)
    found = [pid for pid in patient_ids if pid in stage2_set]
    if found:
        raise ValueError(
            f"[ABORT] stage2_holdout 환자가 {context}에 포함됨 (count={len(found)}): {found[:5]}...\n"
            "leakage guard: dev_safe cohort에 stage2_holdout 포함 불가."
        )


def check_no_overlap_normal_lesion(normal_ids: list, lesion_ids: list):
    """
    정상 환자 ID와 병변 환자 ID가 겹치면 즉시 abort.
    """
    normal_set = set(normal_ids)
    lesion_set = set(lesion_ids)
    overlap = normal_set & lesion_set
    if overlap:
        raise ValueError(
            f"[ABORT] 정상/병변 patient_id 충돌 (count={len(overlap)}): {list(overlap)[:5]}...\n"
            "label 정의 충돌: 동일 환자가 정상(label=0)과 병변(label=1)에 중복됨."
        )


def check_no_duplicate_ids(ids: list, group_name: str):
    """
    중복 patient_id 있으면 abort.
    """
    seen = set()
    dupes = []
    for pid in ids:
        if pid in seen:
            dupes.append(pid)
        seen.add(pid)
    if dupes:
        raise ValueError(
            f"[ABORT] {group_name} 내 중복 patient_id (count={len(dupes)}): {dupes[:5]}...\n"
            "중복 patient_id는 aggregation 결과를 신뢰할 수 없음."
        )


# ---------------------------------------------------------------------------
# Score 파일 로드 함수
# ---------------------------------------------------------------------------

def find_score_file(score_dir: Path, patient_id: str) -> Path:
    """
    score 디렉토리에서 patient_id에 해당하는 CSV 파일을 찾는다.
    파일명 = {patient_id}.csv
    없으면 FileNotFoundError.
    """
    expected = score_dir / f"{patient_id}.csv"
    if expected.exists():
        return expected
    raise FileNotFoundError(
        f"[MISSING] score 파일 없음: {expected}\n"
        f"  score_dir={score_dir}, patient_id={patient_id}"
    )


def load_patient_score_df(score_dir: Path, patient_id: str) -> pd.DataFrame:
    """
    환자 score CSV 로드. padim_score 컬럼 없으면 abort.
    NaN/Inf는 count 후 abort.
    Returns: pd.DataFrame
    """
    fpath = find_score_file(score_dir, patient_id)
    df = pd.read_csv(fpath)
    df.columns = df.columns.str.lstrip("﻿")
    if SCORE_COL not in df.columns:
        raise ValueError(
            f"[ABORT] '{SCORE_COL}' 컬럼 없음: {fpath}\n"
            f"  실제 컬럼: {list(df.columns)}"
        )
    nan_count = df[SCORE_COL].isna().sum()
    inf_count = np.isinf(df[SCORE_COL]).sum()
    if nan_count > 0 or inf_count > 0:
        raise ValueError(
            f"[ABORT] NaN/Inf 발견: {fpath}\n"
            f"  patient_id={patient_id}, NaN={nan_count}, Inf={inf_count}\n"
            "NaN/Inf가 있는 환자는 aggregation 결과를 신뢰할 수 없음."
        )
    return df


# ---------------------------------------------------------------------------
# Patient-level aggregation 함수
# ---------------------------------------------------------------------------

def detect_slice_col(df: pd.DataFrame) -> str | None:
    """slice 구분자 컬럼 탐지. SLICE_COL_CANDIDATES 순서로 확인."""
    for col in SLICE_COL_CANDIDATES:
        if col in df.columns:
            return col
    return None


def aggregate_patient(
    df: pd.DataFrame,
    patient_id: str,
) -> dict:
    """
    단일 환자 patch score를 집계하여 patient-level score dict 반환.

    Returns:
        dict with keys:
          patient_id, n_patches,
          patient_p95_patch_score    (Primary)
          patient_top1pct_mean_patch_score  (Secondary)
          patient_top10_mean_patch_score    (Secondary)
          patient_max_patch_score    (Exploratory)
          patient_max_slice_score    (Exploratory; float or "UNAVAILABLE")
          slice_col_used             (str or None)
          top1pct_n_used             (int)
          top10_n_used               (int)
    """
    scores = df[SCORE_COL].values
    n = len(scores)

    # Primary
    p95 = float(np.percentile(scores, 95))

    # Secondary: top1% mean (최소 1개 patch 포함)
    top1pct_k = max(1, int(np.ceil(n * 0.01)))
    top1pct_idx = np.argpartition(scores, -top1pct_k)[-top1pct_k:]
    top1pct_mean = float(np.mean(scores[top1pct_idx]))

    # Secondary: top10 mean (patch < 10이면 가능한 개수만 사용 — metadata 기록)
    top10_k = min(10, n)
    if top10_k < 10:
        # patch 수 < 10: 가능한 개수만 사용, 결과 metadata에 기록
        pass
    top10_idx = np.argpartition(scores, -top10_k)[-top10_k:]
    top10_mean = float(np.mean(scores[top10_idx]))

    # Exploratory: max patch
    max_patch = float(np.max(scores))

    # Exploratory: max slice score
    slice_col = detect_slice_col(df)
    if slice_col is not None:
        slice_max = df.groupby(slice_col)[SCORE_COL].max()
        max_slice = float(slice_max.max())
    else:
        max_slice = "UNAVAILABLE"

    return {
        "patient_id": patient_id,
        "n_patches": n,
        "patient_p95_patch_score": p95,
        "patient_top1pct_mean_patch_score": top1pct_mean,
        "patient_top10_mean_patch_score": top10_mean,
        "patient_max_patch_score": max_patch,
        "patient_max_slice_score": max_slice,
        "slice_col_used": slice_col,
        "top1pct_n_used": top1pct_k,
        "top10_n_used": top10_k,
    }


# ---------------------------------------------------------------------------
# Metric 계산 함수
# ---------------------------------------------------------------------------

def compute_metrics_for_aggregation(
    scores_arr: np.ndarray,
    labels_arr: np.ndarray,
    agg_name: str,
    threshold_dict: dict,
    comparison_tag: str,
) -> dict:
    """
    patient-level aggregation score + label → metric 계산.

    threshold_dict: {'p95': float or None, 'p95_status': str,
                     'p99': float or None, 'p99_status': str, ...}
    comparison_tag: 'v1_v1' | 'v1_v2' | 'v2_v2'

    Returns: dict of metrics
    """
    result = {
        "aggregation": agg_name,
        "comparison": comparison_tag,
        "n_patients": int(len(labels_arr)),
        "n_positive": int(np.sum(labels_arr == 1)),
        "n_negative": int(np.sum(labels_arr == 0)),
    }

    # AUROC
    if len(np.unique(labels_arr)) < 2:
        result["patient_auroc"] = "NOT_COMPUTABLE_SINGLE_CLASS"
    else:
        result["patient_auroc"] = _auroc(labels_arr, scores_arr)

    # AUPRC
    if len(np.unique(labels_arr)) < 2:
        result["patient_auprc"] = "NOT_COMPUTABLE_SINGLE_CLASS"
    else:
        result["patient_auprc"] = _auprc(labels_arr, scores_arr)

    # specificity/sensitivity/confusion matrix at p95 and p99
    for level in ["p95", "p99"]:
        thr_val = threshold_dict.get(level)
        thr_status = threshold_dict.get(f"{level}_status", "NEEDS_THRESHOLD_CONFIRMATION")
        if thr_val is None or "NEEDS_THRESHOLD_CONFIRMATION" in str(thr_status):
            result[f"specificity_at_{level}"] = "NEEDS_THRESHOLD_CONFIRMATION"
            result[f"sensitivity_at_{level}"] = "NEEDS_THRESHOLD_CONFIRMATION"
            result[f"confusion_matrix_at_{level}"] = "NEEDS_THRESHOLD_CONFIRMATION"
            result[f"threshold_{level}_used"] = thr_status
        else:
            pred = (scores_arr >= thr_val).astype(int)
            tp = int(np.sum((pred == 1) & (labels_arr == 1)))
            fp = int(np.sum((pred == 1) & (labels_arr == 0)))
            tn = int(np.sum((pred == 0) & (labels_arr == 0)))
            fn = int(np.sum((pred == 0) & (labels_arr == 1)))
            specificity = tn / (tn + fp) if (tn + fp) > 0 else "UNDEFINED"
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else "UNDEFINED"
            result[f"specificity_at_{level}"] = specificity
            result[f"sensitivity_at_{level}"] = sensitivity
            result[f"confusion_matrix_at_{level}"] = {
                "tp": tp, "fp": fp, "tn": tn, "fn": fn
            }
            result[f"threshold_{level}_used"] = thr_val
            result[f"threshold_{level}_source"] = threshold_dict.get("source_path", "unknown")

    return result


# ---------------------------------------------------------------------------
# Cohort 구성 함수
# ---------------------------------------------------------------------------

def build_cohort(
    cohort: str,
    normal_score_dir: Path,
    lesion_score_dir: Path,
    comparison_tag: str,
    stage1_ids: list,
    stage2_ids: list,
    normal_test_ids: list,
) -> tuple:
    """
    cohort 구성 + leakage guard + aggregation.

    Returns:
        patient_rows: list of dict (patient 집계 결과)
        cohort_meta: dict (환자 수, label 분포, stage2_holdout_count 등)
    """
    if cohort == "dev_safe":
        lesion_ids = stage1_ids
        # guard: stage2_holdout 포함 여부 검사
        check_no_stage2_in_list(lesion_ids, stage2_ids, context=f"{comparison_tag} dev_safe lesion_ids")
        check_no_stage2_in_list(normal_test_ids, stage2_ids, context=f"{comparison_tag} dev_safe normal_ids")
        use_scope = "development_only"
        stage2_holdout_count = 0
    elif cohort == "full_retrospective":
        # QUARANTINE: 병변 전체 308명 사용 (stage1_dev + stage2_holdout)
        lesion_ids = stage1_ids + stage2_ids
        use_scope = "quarantine_reference_only"
        stage2_holdout_count = len(stage2_ids)
    else:
        raise ValueError(f"[ABORT] 알 수 없는 cohort: {cohort}. 'dev_safe' 또는 'full_retrospective'만 허용.")

    check_no_duplicate_ids(normal_test_ids, "normal_test")
    check_no_duplicate_ids(lesion_ids, f"{comparison_tag} lesion")
    check_no_overlap_normal_lesion(normal_test_ids, lesion_ids)

    patient_rows = []
    missing_files = []

    # 정상 환자 집계 (label=0)
    for pid in normal_test_ids:
        try:
            df = load_patient_score_df(normal_score_dir, pid)
            row = aggregate_patient(df, pid)
            row["label"] = 0
            row["group"] = "normal"
            patient_rows.append(row)
        except (FileNotFoundError, ValueError) as e:
            missing_files.append({"patient_id": pid, "group": "normal", "error": str(e)})

    # 병변 환자 집계 (label=1)
    for pid in lesion_ids:
        try:
            df = load_patient_score_df(lesion_score_dir, pid)
            row = aggregate_patient(df, pid)
            row["label"] = 1
            row["group"] = "lesion"
            patient_rows.append(row)
        except (FileNotFoundError, ValueError) as e:
            missing_files.append({"patient_id": pid, "group": "lesion", "error": str(e)})

    if missing_files:
        msg = "\n".join([
            f"  [{r['group']}] {r['patient_id']}: {r['error']}"
            for r in missing_files[:10]
        ])
        raise FileNotFoundError(
            f"[ABORT] score 파일 누락 또는 컬럼 오류 ({len(missing_files)}건):\n{msg}"
        )

    cohort_meta = {
        "cohort": cohort,
        "comparison": comparison_tag,
        "use_scope": use_scope,
        "n_normal": len(normal_test_ids),
        "n_lesion_stage1_dev": len(stage1_ids),
        "n_lesion_total": len(lesion_ids),
        "n_patients_total": len(patient_rows),
        "stage2_holdout_count": stage2_holdout_count,
        "normal_score_dir": str(normal_score_dir),
        "lesion_score_dir": str(lesion_score_dir),
    }
    return patient_rows, cohort_meta


# ---------------------------------------------------------------------------
# Dry-run 함수
# ---------------------------------------------------------------------------

def run_dry_run(
    cohort: str,
    comparisons: list,
    output_dir: Path,
    run_tag: str,
    stage1_ids: list,
    stage2_ids: list,
    normal_test_ids: list,
):
    """
    dry-run: 입력 경로 존재, score CSV 파일 수, 컬럼 샘플, 예상 환자 수,
    label 분포, output 충돌, stage2_holdout 포함 여부 확인.
    metric 계산/결과 저장 금지.
    """
    print("\n[DRY-RUN] ======== dry-run 시작 ========")
    print(f"  cohort       : {cohort}")
    print(f"  comparisons  : {comparisons}")
    print(f"  output_dir   : {output_dir}")
    print(f"  run_tag      : {run_tag}")

    # 정상 split CSV
    print(f"\n[DRY-RUN] 정상 split CSV: {NORMAL_SPLIT_CSV}")
    print(f"  존재: {NORMAL_SPLIT_CSV.exists()}")
    print(f"  test 환자 수: {len(normal_test_ids)}")

    # 병변 split CSV
    print(f"\n[DRY-RUN] 병변 stage split CSV: {LESION_STAGE_SPLIT_CSV}")
    print(f"  존재: {LESION_STAGE_SPLIT_CSV.exists()}")
    print(f"  stage1_dev 수: {len(stage1_ids)}")
    print(f"  stage2_holdout 수: {len(stage2_ids)}")

    # leakage guard 사전 확인
    if cohort == "dev_safe":
        stage2_set = set(stage2_ids)
        found_in_normal = [pid for pid in normal_test_ids if pid in stage2_set]
        found_in_lesion = [pid for pid in stage1_ids if pid in stage2_set]
        print(f"\n[DRY-RUN] stage2_holdout leakage guard:")
        print(f"  normal_test에 stage2_holdout 포함: {len(found_in_normal)}")
        print(f"  stage1_dev에 stage2_holdout 포함: {len(found_in_lesion)}")
        if found_in_normal or found_in_lesion:
            print("  [WARNING] stage2_holdout 포함 감지됨 — 실행 시 abort 예정")
        else:
            print("  [OK] stage2_holdout 포함 없음")

    # cohort 구성 예상
    if cohort == "dev_safe":
        n_lesion = len(stage1_ids)
    else:
        n_lesion = len(stage1_ids) + len(stage2_ids)
    n_total = len(normal_test_ids) + n_lesion
    print(f"\n[DRY-RUN] 예상 cohort 구성:")
    print(f"  normal (label=0): {len(normal_test_ids)}")
    print(f"  lesion (label=1): {n_lesion}")
    print(f"  합계            : {n_total}")

    # score 디렉토리별 파일 수 + 컬럼 샘플 확인
    score_dirs = {
        "v1_v1 lesion": LESION_SCORE_DIR_V1V1,
        "v1_v2 lesion": LESION_SCORE_DIR_V1V2,
        "v2_v2 lesion": LESION_SCORE_DIR_V2V2,
        "v1 normal": NORMAL_SCORE_DIR_V1,
        "v2 normal": NORMAL_SCORE_DIR_V2,
    }
    print(f"\n[DRY-RUN] score 디렉토리 확인:")
    for label, d in score_dirs.items():
        exists = d.exists()
        if exists:
            csvs = list(d.glob("*.csv"))
            n_files = len(csvs)
            # 컬럼 샘플 (첫 번째 파일의 컬럼만)
            if csvs:
                sample_df = pd.read_csv(csvs[0], nrows=0)
                sample_df.columns = sample_df.columns.str.lstrip("﻿")
                has_score_col = SCORE_COL in sample_df.columns
                slice_col = detect_slice_col(sample_df)
            else:
                has_score_col = False
                slice_col = None
        else:
            n_files = 0
            has_score_col = False
            slice_col = None
        print(
            f"  [{label}] 존재={exists}, 파일수={n_files}, "
            f"{SCORE_COL}={has_score_col}, slice_col={slice_col}"
        )

    # output collision 확인
    print(f"\n[DRY-RUN] output collision 확인 (output_dir={output_dir}):")
    suffixes = ["_patient_scores.csv", "_metrics.csv", "_summary.json", "_summary.md"]
    for suffix in suffixes:
        fpath = output_dir / f"{run_tag}{suffix}"
        exists = fpath.exists()
        status = "COLLISION — 실행 시 abort 예정" if exists else "없음"
        print(f"  {fpath.name}: {status}")

    print("\n[DRY-RUN] ======== dry-run 완료 — metric 계산/결과 저장 없음 ========\n")


# ---------------------------------------------------------------------------
# Output guard
# ---------------------------------------------------------------------------

def check_output_collision(output_dir: Path, run_tag: str):
    """
    출력 파일이 이미 존재하면 abort. --overwrite 옵션 없음(설계 의도).
    """
    suffixes = ["_patient_scores.csv", "_metrics.csv", "_summary.json", "_summary.md"]
    collisions = []
    for suffix in suffixes:
        fpath = output_dir / f"{run_tag}{suffix}"
        if fpath.exists():
            collisions.append(str(fpath))
    if collisions:
        raise FileExistsError(
            f"[ABORT] output 파일이 이미 존재함 (덮어쓰기 금지):\n"
            + "\n".join(f"  {p}" for p in collisions)
            + "\nrun_tag를 변경하거나 기존 파일을 수동으로 처리하세요."
        )


# ---------------------------------------------------------------------------
# 결과 저장 함수
# ---------------------------------------------------------------------------

def save_results(
    output_dir: Path,
    run_tag: str,
    patient_rows: list,
    all_metrics: list,
    summary_meta: dict,
):
    """
    결과 CSV / JSON / MD 저장.
    실제 실행 시에만 호출 (--confirm-run 필수).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) patient_scores.csv
    scores_path = output_dir / f"{run_tag}_patient_scores.csv"
    scores_df = pd.DataFrame(patient_rows)
    scores_df.to_csv(scores_path, index=False)
    print(f"[SAVED] {scores_path}")

    # 2) metrics.csv
    metrics_path = output_dir / f"{run_tag}_metrics.csv"
    flat_metrics = []
    for m in all_metrics:
        row = {k: str(v) if isinstance(v, dict) else v for k, v in m.items()}
        flat_metrics.append(row)
    metrics_df = pd.DataFrame(flat_metrics)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[SAVED] {metrics_path}")

    # 3) summary.json
    summary_path = output_dir / f"{run_tag}_summary.json"
    full_summary = {
        "meta": summary_meta,
        "metrics": all_metrics,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"[SAVED] {summary_path}")

    # 4) summary.md
    md_path = output_dir / f"{run_tag}_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Mixed Cohort Patient Metric Summary\n\n")
        f.write(f"**run_tag**: {run_tag}  \n")
        f.write(f"**cohort**: {summary_meta.get('cohort')}  \n")
        f.write(f"**use_scope**: {summary_meta.get('use_scope')}  \n")
        f.write(f"**stage2_holdout_count**: {summary_meta.get('stage2_holdout_count')}  \n\n")
        f.write("## Metrics\n\n")
        for m in all_metrics:
            f.write(f"### {m.get('comparison')} / {m.get('aggregation')}\n")
            for k, v in m.items():
                if k not in ("comparison", "aggregation"):
                    f.write(f"- **{k}**: {v}\n")
            f.write("\n")
    print(f"[SAVED] {md_path}")


# ---------------------------------------------------------------------------
# 메인 실행
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Mixed cohort patient-level metric 계산 스크립트 (read-only score CSV 사용)"
    )
    parser.add_argument(
        "--cohort",
        choices=["dev_safe", "full_retrospective"],
        default="dev_safe",
        help="사용할 cohort. 기본값: dev_safe (정상 36 + stage1_dev 154). "
             "full_retrospective는 --confirm-full-retrospective 없으면 abort.",
    )
    parser.add_argument(
        "--confirm-full-retrospective",
        action="store_true",
        default=False,
        help="full_retrospective cohort 실행 시 필수. QUARANTINE 데이터 접근 동의.",
    )
    parser.add_argument(
        "--comparison",
        choices=["v1_v1", "v1_v2", "v2_v2", "all"],
        default="all",
        help="비교 설정. 기본값: all (세 설정 모두). "
             "[핵심] v1_v2 vs v2_v2 = 같은 v2 lesion test 기준 model 차이 비교. "
             "[참고] v1_v1 vs v1_v2 vs v2_v2 3자 참고.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation/mixed_cohort_patient_metrics",
        help="결과 저장 디렉토리. 기본: evaluation/mixed_cohort_patient_metrics",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="dev_safe_v1",
        help="출력 파일명 prefix. 기본: dev_safe_v1",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="입력 경로/환자수/label 분포/output 충돌만 확인. metric 계산/저장 없음.",
    )
    parser.add_argument(
        "--confirm-run",
        action="store_true",
        default=False,
        help="실제 metric 저장 시 필수. 이 플래그 없으면 결과 파일 저장 안 함.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # -----------------------------------------------------------------------
    # cohort guard: full_retrospective 기본 차단
    # -----------------------------------------------------------------------
    if args.cohort == "full_retrospective" and not args.confirm_full_retrospective:
        print(
            "[ABORT] full_retrospective cohort는 기본 차단 상태입니다.\n"
            "  stage2_holdout 154명이 포함되어 있어 개발 판단 근거 사용 금지.\n"
            "  use_scope = quarantine_reference_only\n"
            "  진행하려면 --confirm-full-retrospective 플래그를 명시하세요.\n"
            "  단, 이 cohort는 stage2_holdout 봉인 해제 전까지 개발 결정에 사용 금지입니다.",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = REPO_ROOT / args.output_dir

    # -----------------------------------------------------------------------
    # split CSV 로드
    # -----------------------------------------------------------------------
    try:
        normal_test_ids = load_normal_test_patient_ids()
    except (FileNotFoundError, ValueError) as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    try:
        stage1_ids, stage2_ids = load_lesion_stage_patient_ids()
    except (FileNotFoundError, ValueError) as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # dry-run 분기
    # -----------------------------------------------------------------------
    if args.dry_run:
        comparisons = (
            ["v1_v1", "v1_v2", "v2_v2"] if args.comparison == "all" else [args.comparison]
        )
        run_dry_run(
            cohort=args.cohort,
            comparisons=comparisons,
            output_dir=output_dir,
            run_tag=args.run_tag,
            stage1_ids=stage1_ids,
            stage2_ids=stage2_ids,
            normal_test_ids=normal_test_ids,
        )
        return

    # -----------------------------------------------------------------------
    # 실제 metric 계산 — 여기부터는 --confirm-run 필수
    # --confirm-run 없이는 이 구간에 절대 진입하지 않는다.
    # -----------------------------------------------------------------------
    if not args.confirm_run:
        print(
            "[ABORT] 실제 metric 계산은 --confirm-run 없이는 실행할 수 없음.\n"
            "  dry-run만 하려면 --dry-run 플래그를 사용하세요.\n"
            "  metric 계산 및 결과 저장은 별도 승인 후 --confirm-run 플래그로 실행하세요.",
            file=sys.stderr,
        )
        sys.exit(1)

    # output collision guard
    try:
        check_output_collision(output_dir, args.run_tag)
    except FileExistsError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    # threshold 로드
    v1_thr = load_v1_thresholds()
    v2_thr = load_v2_thresholds()

    # 비교 설정별 경로 매핑
    # [핵심 비교축] v1_v2 vs v2_v2: 같은 v2 lesion test 기준 model 차이
    # [참고 비교축] v1_v1 vs v1_v2 vs v2_v2 3자
    comparison_configs = {
        "v1_v1": {
            "normal_score_dir": NORMAL_SCORE_DIR_V1,
            "lesion_score_dir": LESION_SCORE_DIR_V1V1,
            "threshold": v1_thr,
            "note": "v1 model + v1 lesion test. threshold: v1 normal-val p95/p99.",
            "comparison_axis": "reference",
        },
        "v1_v2": {
            "normal_score_dir": NORMAL_SCORE_DIR_V1,
            "lesion_score_dir": LESION_SCORE_DIR_V1V2,
            "threshold": v1_thr,
            "note": "v1 model + v2 lesion test. threshold: v1 normal-val p95/p99. [핵심 비교축: v1_v2 vs v2_v2]",
            "comparison_axis": "core",
        },
        "v2_v2": {
            "normal_score_dir": NORMAL_SCORE_DIR_V2,
            "lesion_score_dir": LESION_SCORE_DIR_V2V2,
            "threshold": v2_thr,
            "note": "v2 model + v2 lesion test. threshold: v2 normal-val p95/p99. [핵심 비교축: v1_v2 vs v2_v2]",
            "comparison_axis": "core",
        },
    }

    comparisons = (
        ["v1_v1", "v1_v2", "v2_v2"] if args.comparison == "all" else [args.comparison]
    )

    all_patient_rows = []
    all_metrics = []
    summary_meta_list = []

    for comp_tag in comparisons:
        cfg = comparison_configs[comp_tag]
        print(f"\n[INFO] === {comp_tag} 처리 중 ===")
        print(f"  {cfg['note']}")

        try:
            patient_rows, cohort_meta = build_cohort(
                cohort=args.cohort,
                normal_score_dir=cfg["normal_score_dir"],
                lesion_score_dir=cfg["lesion_score_dir"],
                comparison_tag=comp_tag,
                stage1_ids=stage1_ids,
                stage2_ids=stage2_ids,
                normal_test_ids=normal_test_ids,
            )
        except (FileNotFoundError, ValueError) as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)

        print(
            f"  [INFO] 환자 수 로드 완료: "
            f"normal={cohort_meta['n_normal']}, "
            f"lesion={cohort_meta['n_lesion_total']}, "
            f"total={cohort_meta['n_patients_total']}, "
            f"stage2_holdout_count={cohort_meta['stage2_holdout_count']}"
        )

        # patient-level scores array
        df_pat = pd.DataFrame(patient_rows)
        labels_arr = df_pat["label"].values.astype(int)

        # aggregation 목록 — 집계 컬럼 이름 → aggregation 이름
        agg_cols = [
            ("patient_p95_patch_score", "p95"),
            ("patient_top1pct_mean_patch_score", "top1pct_mean"),
            ("patient_top10_mean_patch_score", "top10_mean"),
            ("patient_max_patch_score", "max_patch"),
        ]
        # max_slice는 UNAVAILABLE일 수 있음
        slice_available = any(
            isinstance(r.get("patient_max_slice_score"), float) for r in patient_rows
        )
        if slice_available:
            agg_cols.append(("patient_max_slice_score", "max_slice"))
        else:
            print(f"  [INFO] patient_max_slice_score: UNAVAILABLE — slice 컬럼 없음. 건너뜀.")

        for score_col_name, agg_name in agg_cols:
            scores_arr = df_pat[score_col_name].values.astype(float)
            metrics = compute_metrics_for_aggregation(
                scores_arr=scores_arr,
                labels_arr=labels_arr,
                agg_name=agg_name,
                threshold_dict=cfg["threshold"],
                comparison_tag=comp_tag,
            )
            metrics["comparison_axis_note"] = cfg["comparison_axis"]
            all_metrics.append(metrics)

        cohort_meta["comparison_axis"] = cfg["comparison_axis"]
        cohort_meta["note"] = cfg["note"]
        summary_meta_list.append(cohort_meta)

        # 환자 행에 comparison 태그 추가
        for r in patient_rows:
            r["comparison"] = comp_tag
        all_patient_rows.extend(patient_rows)

    # -----------------------------------------------------------------------
    # 결과 저장 (--confirm-run 없으면 저장 안 함)
    # -----------------------------------------------------------------------
    summary_meta = {
        "run_tag": args.run_tag,
        "cohort": args.cohort,
        "use_scope": (
            "development_only" if args.cohort == "dev_safe" else "quarantine_reference_only"
        ),
        "stage2_holdout_count": sum(m["stage2_holdout_count"] for m in summary_meta_list),
        "comparisons": comparisons,
        "comparison_axis_note": (
            "핵심 비교축: v1_v2 vs v2_v2 (같은 v2 lesion test 기준 model 차이). "
            "참고 비교축: v1_v1 vs v1_v2 vs v2_v2 3자."
        ),
        "details": summary_meta_list,
    }

    if args.confirm_run:
        try:
            save_results(
                output_dir=output_dir,
                run_tag=args.run_tag,
                patient_rows=all_patient_rows,
                all_metrics=all_metrics,
                summary_meta=summary_meta,
            )
        except Exception as e:
            print(f"[ABORT] 결과 저장 중 오류:\n{e}", file=sys.stderr)
            sys.exit(1)
    # --confirm-run 없이는 이 else 블록에 도달하지 않음 (위 abort guard에서 차단됨)

    print("\n[DONE] compute_mixed_cohort_patient_metrics.py 완료.")


if __name__ == "__main__":
    main()
