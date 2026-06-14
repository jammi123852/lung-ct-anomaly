"""
compute_lesion_metrics_fast.py: 이미 저장된 lesion score CSV로부터 평가 지표를 빠르게 산출한다.

배경:
- evaluate_lesion_subset.py의 score 계산(환자별 CSV)은 끝났으나,
  Evaluator.compute_auroc/auprc가 unique threshold마다 전체 배열을 순회하는 O(T*N) 구조라
  724만 patch 규모에서 사실상 완료되지 않는다.
- 이 스크립트는 score CSV를 read-only로 읽어 정렬 기반 O(N log N) AUROC/AUPRC로 지표만 다시 계산한다.

안전 원칙:
- 실행 중인 evaluate_lesion_subset.py 프로세스를 건드리지 않는다 (이 스크립트는 별도 프로세스).
- 기존 score CSV / evaluation 결과 파일을 수정·삭제하지 않는다 (score CSV는 read-only).
- 결과는 {output_tag}_fast_* 로 분리 저장하여 기존 lesion_eval_* 파일과 겹치지 않게 한다.

지표 정의는 evaluate_lesion_subset.py와 동일하게 맞춘다:
- patch_label: score CSV에 이미 저장된 값(label_mode=any_pixel)을 그대로 사용.
- slice_label: 환자×local_z 그룹에 양성 patch가 1개라도 있으면 1, slice score = 그룹 max padim_score.
- patient_label: lesion-only 전부 양성 → AUROC 계산 불가 → not_applicable_positive_only.
- threshold: normal_val_p95 / normal_val_p99 (정상 val 환자 by_patient score percentile).
- patch_dice / patch_iou: patch_label(true) vs (padim_score >= threshold)(pred). patch-level (pixel-level 아님).
"""

from __future__ import annotations

import csv
import glob
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from position_aware_padim.patient_splitter import PatientSplitter

NORMAL_SCORE_DIR = REPO_ROOT / "outputs" / "position-aware-padim-v1" / "scores" / "padim_v1" / "by_patient"
LESION_SCORE_DIR_V1 = REPO_ROOT / "outputs" / "position-aware-padim-v1" / "scores" / "padim_v1" / "lesion_by_patient"
LESION_SCORE_DIR_V2 = REPO_ROOT / "outputs" / "position-aware-padim-v1" / "scores" / "padim_v1" / "lesion_v2_by_patient"
EVAL_DIR_V1 = REPO_ROOT / "outputs" / "position-aware-padim-v1" / "evaluation" / "lesion_subset"
EVAL_DIR_V2 = REPO_ROOT / "outputs" / "position-aware-padim-v1" / "evaluation" / "lesion_subset_v2"
REPORTS_DIR = REPO_ROOT / "outputs" / "position-aware-padim-v1" / "reports"
RUNTIME_CSV = REPORTS_DIR / "runtime_summary.csv"
RUNTIME_COLUMNS = ["timestamp", "script", "metric", "value"]
SCRIPT_NAME = "compute_lesion_metrics_fast.py"


# ---------------------------------------------------------------------------
# 정렬 기반 효율 지표 (numpy, O(N log N))
# ---------------------------------------------------------------------------

def fast_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mann-Whitney U 기반 AUROC. tie는 average rank로 처리. trapz ROC와 동일값."""
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


def fast_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """내림차순 누적 기반 AUPRC (trapz). O(N log N)."""
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


def patch_dice(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    denom = int((y_true == 1).sum()) + int((y_pred == 1).sum())
    return float(2 * tp / denom) if denom > 0 else float("nan")


def patch_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    inter = int(((y_pred == 1) & (y_true == 1)).sum())
    union = int(((y_pred == 1) | (y_true == 1)).sum())
    return float(inter / union) if union > 0 else float("nan")


def compute_threshold_percentile(pct: int, threshold_json: str | None = None):
    """정상 val 환자 by_patient score의 percentile을 threshold로 계산. (threshold, info).

    threshold_json이 있으면 JSON에서 직접 읽어 NORMAL_SCORE_DIR 대신 사용 (v2 전용).
    """
    if threshold_json is not None:
        try:
            with open(threshold_json, encoding="utf-8") as f:
                data = json.load(f)
            thr = float(data[f"threshold_p{pct}"])
            return thr, {"source": f"threshold_json_p{pct}", "status": "applied", "threshold_json": threshold_json}
        except Exception as exc:
            return None, {"status": "확인 필요", "reason": f"threshold_json 읽기 실패: {exc}"}

    info = {"source": f"normal_val_p{pct}"}
    try:
        split = PatientSplitter(str(REPO_ROOT)).load_split()
        val_pids = list(split.val)
    except Exception as exc:
        return None, {"status": "확인 필요", "reason": f"val 구분 실패: {exc}"}
    if not val_pids:
        return None, {"status": "확인 필요", "reason": "val 목록 비어 있음"}
    scores_list = []
    n_used = 0
    for pid in val_pids:
        p = NORMAL_SCORE_DIR / f"{pid}.csv"
        if not p.exists():
            continue
        try:
            col = pd.read_csv(p, encoding="utf-8-sig", usecols=["padim_score"])["padim_score"]
        except Exception:
            continue
        col = col[~col.isna()]
        if len(col) > 0:
            scores_list.append(col.values)
            n_used += 1
    if not scores_list:
        return None, {"status": "확인 필요", "reason": "val score 수집 실패"}
    all_val = np.concatenate(scores_list)
    thr = float(np.percentile(all_val, pct))
    info.update({"status": "applied", "n_val_patients_used": n_used, "n_val_scores": int(len(all_val))})
    return thr, info


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="lesion score 기반 빠른 지표 산출")
    parser.add_argument(
        "--dataset-profile",
        type=str,
        default="v1_model_roi",
        choices=["v1_model_roi", "v2_roi_0_0"],
        help=(
            "평가할 데이터셋 profile. "
            "v1_model_roi: 기존 lesion_by_patient score 사용 (기본값). "
            "v2_roi_0_0: lesion_v2_by_patient score 사용 (출력도 lesion_subset_v2)."
        ),
    )
    parser.add_argument(
        "--score-dir",
        type=str,
        default=None,
        help="lesion score 경로 오버라이드 (기본: dataset-profile 기반 자동 결정)",
    )
    parser.add_argument(
        "--evaluation-dir",
        type=str,
        default=None,
        help="evaluation 출력 경로 오버라이드 (기본: dataset-profile 기반 자동 결정)",
    )
    parser.add_argument(
        "--threshold-json",
        type=str,
        default=None,
        help="normal threshold JSON 경로 (v2 threshold 사용 시. 예: evaluation/normal_v2_roi0_0/normal_v2_threshold.json)",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=None,
        help="reports 경로 오버라이드 (기본: reports)",
    )
    args = parser.parse_args()
    is_v2 = (args.dataset_profile == "v2_roi_0_0")
    LESION_SCORE_DIR = LESION_SCORE_DIR_V2 if is_v2 else LESION_SCORE_DIR_V1
    EVAL_DIR = EVAL_DIR_V2 if is_v2 else EVAL_DIR_V1

    # CLI 오버라이드
    if args.score_dir is not None:
        LESION_SCORE_DIR = REPO_ROOT / args.score_dir
    if args.evaluation_dir is not None:
        EVAL_DIR = REPO_ROOT / args.evaluation_dir
    if args.reports_dir is not None:
        global REPORTS_DIR, RUNTIME_CSV
        REPORTS_DIR = REPO_ROOT / args.reports_dir
        RUNTIME_CSV = REPORTS_DIR / "runtime_summary.csv"
    v2_prefix = "v2_" if is_v2 else ""

    start = time.time()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(glob.glob(str(LESION_SCORE_DIR / "*.csv")))
    n_files = len(csv_paths)
    if n_files == 0:
        print(f"[ERROR] lesion score CSV가 없습니다: {LESION_SCORE_DIR}")
        sys.exit(1)
    print(f"[fast_metrics] dataset_profile={args.dataset_profile}")
    print(f"[fast_metrics] lesion score CSV {n_files}개 read-only 집계 시작")

    # 환자별로 읽어 patch/slice/patient 단위 누적 (메모리 안전: padim_score+patch_label만)
    patch_scores_parts = []
    patch_labels_parts = []
    slice_scores_parts = []
    slice_labels_parts = []
    patient_scores = []
    patient_labels = []
    n_nan_total = 0
    n_inf_total = 0
    group_counts = {"NSCLC": 0, "MSD_Lung": 0, "OTHER": 0}

    for i, p in enumerate(csv_paths):
        try:
            df = pd.read_csv(p, encoding="utf-8-sig",
                             usecols=["local_z", "padim_score", "patch_label", "group"])
        except ValueError:
            # group 컬럼이 없을 수 있으니 fallback
            df = pd.read_csv(p, encoding="utf-8-sig",
                             usecols=["local_z", "padim_score", "patch_label"])
            df["group"] = "OTHER"

        g0 = str(df["group"].iloc[0]) if "group" in df.columns and len(df) > 0 else "OTHER"
        if g0 in group_counts:
            group_counts[g0] += 1
        else:
            group_counts["OTHER"] += 1

        n_inf_total += int(np.isinf(df["padim_score"].values).sum())
        n_nan_total += int(df["padim_score"].isna().sum())
        df = df[~df["padim_score"].isna()]
        if len(df) == 0:
            continue

        sc = df["padim_score"].values.astype(np.float64)
        lb = (df["patch_label"].values == 1).astype(np.int8)
        patch_scores_parts.append(sc)
        patch_labels_parts.append(lb)

        # slice: local_z 그룹 max score / any positive
        gb = df.groupby("local_z")
        s_score = gb["padim_score"].max().values.astype(np.float64)
        s_label = (gb["patch_label"].max().values >= 1).astype(np.int8)
        slice_scores_parts.append(s_score)
        slice_labels_parts.append(s_label)

        # patient: max score / any positive
        patient_scores.append(float(df["padim_score"].max()))
        patient_labels.append(int(df["patch_label"].max() >= 1))

        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{n_files} 파일 집계")

    patch_scores = np.concatenate(patch_scores_parts)
    patch_labels = np.concatenate(patch_labels_parts)
    slice_scores = np.concatenate(slice_scores_parts)
    slice_labels = np.concatenate(slice_labels_parts)
    patient_scores = np.asarray(patient_scores, dtype=np.float64)
    patient_labels = np.asarray(patient_labels, dtype=np.int8)

    print(f"[fast_metrics] patch={len(patch_scores):,}, slice={len(slice_scores):,}, "
          f"patient={len(patient_scores)}, NaN={n_nan_total}, Inf={n_inf_total}")

    # patch / slice AUROC·AUPRC (threshold 무관)
    patch_auroc = fast_auroc(patch_labels, patch_scores)
    patch_auprc = fast_auprc(patch_labels, patch_scores)
    slice_auroc = fast_auroc(slice_labels, slice_scores)
    slice_auprc = fast_auprc(slice_labels, slice_scores)

    patient_pos = int((patient_labels == 1).sum())
    patient_total = int(len(patient_labels))
    patient_status = ("not_applicable_positive_only"
                      if patient_pos == patient_total else "computed")

    base_summary = {
        "patch_total": int(len(patch_scores)),
        "patch_positive": int((patch_labels == 1).sum()),
        "patch_negative": int((patch_labels == 0).sum()),
        "patch_score_nan": n_nan_total,
        "patch_score_inf": n_inf_total,
        "slice_total": int(len(slice_scores)),
        "slice_positive": int((slice_labels == 1).sum()),
        "slice_negative": int((slice_labels == 0).sum()),
        "patient_total": patient_total,
        "patient_positive": patient_pos,
        "patient_auroc_status": patient_status,
        "group_counts": group_counts,
        "n_score_csv": n_files,
        "patch_auroc": patch_auroc,
        "patch_auprc": patch_auprc,
        "slice_auroc": slice_auroc,
        "slice_auprc": slice_auprc,
        "note": "score CSV 기반 재산출(fast). 병변 성능 결론 아님. patch_dice/iou는 patch-level (pixel-level 아님).",
    }

    # threshold별(p95/p99) patch_dice/iou + metrics/summary 저장
    for pct, tag in [(95, "p95"), (99, "p99")]:
        thr, thr_info = compute_threshold_percentile(pct, threshold_json=args.threshold_json)
        pdice = piou = None
        if thr is not None:
            pred = (patch_scores >= thr).astype(np.int8)
            pdice = patch_dice(patch_labels, pred)
            piou = patch_iou(patch_labels, pred)

        metrics_rows = [
            {"level": "patch", "auroc": patch_auroc, "auprc": patch_auprc,
             "n": int(len(patch_scores)), "n_pos": int((patch_labels == 1).sum()),
             "patch_dice": pdice, "patch_iou": piou, "threshold": thr, "note": ""},
            {"level": "slice", "auroc": slice_auroc, "auprc": slice_auprc,
             "n": int(len(slice_scores)), "n_pos": int((slice_labels == 1).sum()),
             "patch_dice": None, "patch_iou": None, "threshold": None, "note": ""},
            {"level": "patient", "auroc": None, "auprc": None,
             "n": patient_total, "n_pos": patient_pos,
             "patch_dice": None, "patch_iou": None, "threshold": None,
             "note": patient_status},
        ]
        metrics_df = pd.DataFrame(
            metrics_rows,
            columns=["level", "auroc", "auprc", "n", "n_pos",
                     "patch_dice", "patch_iou", "threshold", "note"],
        )
        out_tag = f"lesion_eval_{v2_prefix}{tag}_fast"
        metrics_csv = EVAL_DIR / f"{out_tag}_metrics.csv"
        summary_json = EVAL_DIR / f"{out_tag}_summary.json"
        metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")

        summary = dict(base_summary)
        summary.update({
            "output_tag": out_tag,
            "threshold_mode": f"normal_val_p{pct}",
            "threshold_value": thr,
            "threshold_info": thr_info,
            "patch_dice": pdice,
            "patch_iou": piou,
        })
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"[fast_metrics] [{tag}] threshold={thr} | patch AUROC={patch_auroc:.4f} "
              f"AUPRC={patch_auprc:.4f} | slice AUROC={slice_auroc:.4f} AUPRC={slice_auprc:.4f} "
              f"| patch_dice={pdice} patch_iou={piou}")
        print(f"          → {metrics_csv}")

    elapsed = time.time() - start
    print(f"[fast_metrics] 완료: {elapsed:.1f}s")

    # runtime_summary 기록 (append, 4컬럼)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RUNTIME_CSV.exists() or RUNTIME_CSV.stat().st_size == 0
    ts = datetime.now().isoformat(timespec="seconds")
    with open(RUNTIME_CSV, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUNTIME_COLUMNS)
        if write_header:
            writer.writeheader()
        for metric, value in [
            ("n_score_csv", n_files),
            ("patch_total", int(len(patch_scores))),
            ("patch_auroc", patch_auroc),
            ("patch_auprc", patch_auprc),
            ("slice_auroc", slice_auroc),
            ("slice_auprc", slice_auprc),
            ("patient_auroc_status", patient_status),
            ("elapsed_seconds", round(elapsed, 2)),
        ]:
            writer.writerow({"timestamp": ts, "script": SCRIPT_NAME, "metric": metric, "value": value})
    print(f"[fast_metrics] runtime_summary.csv 기록 완료")


if __name__ == "__main__":
    main()
