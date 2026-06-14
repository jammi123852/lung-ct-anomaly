"""
Phase 8.5B Metric Calculation (crop-level only)

목적:
  stage2_holdout 143,735 crops에 대해 score column별
  crop-level AUROC/AUPRC를 numpy 직접 구현으로 계산한다.

금지 사항:
  - sklearn 사용 금지 (미설치)
  - threshold/p95/p99/hit-rate/recall 계산 금지
  - threshold sweep 금지
  - patient-level label 임의 정의 금지
  - patient-level metric 계산 금지 (label 정의 미확정 → NEEDS_REVIEW)
  - model forward / training / backward / optimizer step 금지
  - checkpoint 생성 금지
  - score CSV 수정/재생성 금지
  - 기존 Phase 6/7/8 output 수정/삭제/덮어쓰기 금지

실행 guard:
  --run-metric과 --confirm-run 둘 다 없으면 dry-run 보고만 수행.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────
EXPECTED_ROW_COUNT = 143735
EXPECTED_POSITIVE_COUNT = 51335
EXPECTED_NEGATIVE_COUNT = 92400

SCORE_COLUMNS = [
    "crop_score_l1_mean",
    "crop_score_l1_max",
    "crop_score_mse_mean",
    "lung_channels_l1_mean",
    "mediastinal_channels_l1_mean",
]

DEFAULT_SCORE_CSV = (
    "outputs/second-stage-lesion-refiner-v1/scores/"
    "phase8_4_stage2_full_scoring_v1/"
    "phase8_4_stage2_full_scoring_v1.csv"
)
DEFAULT_PREFLIGHT_SUMMARY = (
    "outputs/second-stage-lesion-refiner-v1/review_annotations/"
    "phase8_5a_metric_calculation_preflight_v1/"
    "phase8_5a_metric_calculation_preflight_summary.json"
)
DEFAULT_OUTPUT_ROOT = (
    "outputs/second-stage-lesion-refiner-v1/review_annotations/"
    "phase8_5b_metric_calculation_v1/"
)

OUTPUT_CROP_CSV = "phase8_5b_metric_results_crop_level.csv"
OUTPUT_PATIENT_CSV = "phase8_5b_metric_results_patient_level.csv"
OUTPUT_JSON = "phase8_5b_metric_calculation_summary.json"
OUTPUT_MD = "phase8_5b_metric_calculation_report.md"


# ──────────────────────────────────────────────────────────
# numpy AUROC (rank-based, tie-aware)
# ──────────────────────────────────────────────────────────
def compute_auroc(y_true: np.ndarray, y_score: np.ndarray):
    """
    AUROC: rank-based tie-aware 방식.
    AUROC = (R_pos - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    positive/negative 중 한 class가 없으면 None 반환.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    # tie-aware rank: 오름차순 정렬, 동점은 평균 rank
    order = np.argsort(y_score, kind="stable")
    ranks = np.empty(len(y_score), dtype=np.float64)
    i = 0
    n = len(y_score)
    sorted_scores = y_score[order]
    while i < n:
        j = i
        while j < n - 1 and sorted_scores[j] == sorted_scores[j + 1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    rank_sum_pos = float(ranks[y_true == 1].sum())
    U = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(U / (n_pos * n_neg))


# ──────────────────────────────────────────────────────────
# numpy AUPRC (average precision, score descending)
# ──────────────────────────────────────────────────────────
def compute_auprc(y_true: np.ndarray, y_score: np.ndarray):
    """
    AUPRC: average precision 방식.
    score 내림차순 정렬 → cumulative TP/FP → precision-recall curve.
    positive class가 없으면 None 반환.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return None

    order = np.argsort(-y_score, kind="stable")
    y_sorted = y_true[order]

    tp = np.cumsum(y_sorted).astype(np.float64)
    fp = np.cumsum(1 - y_sorted).astype(np.float64)

    precision = tp / (tp + fp)
    recall = tp / float(n_pos)

    recall_prev = np.concatenate([[0.0], recall[:-1]])
    delta_recall = recall - recall_prev
    return float(np.sum(precision * delta_recall))


# ──────────────────────────────────────────────────────────
# self-check (toy array만 사용, 실제 데이터와 분리)
# ──────────────────────────────────────────────────────────
def run_self_check():
    """
    구현 정확성 self-check.
    완전 분리 / 역순 / 단일 class 케이스 검증.
    """
    # 완전 분리: AUROC=1.0
    yt = np.array([1, 1, 1, 0, 0, 0])
    ys = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    auroc_perfect = compute_auroc(yt, ys)
    assert auroc_perfect is not None and abs(auroc_perfect - 1.0) < 1e-9, \
        f"완전 분리 AUROC 실패: {auroc_perfect}"

    auprc_perfect = compute_auprc(yt, ys)
    assert auprc_perfect is not None and abs(auprc_perfect - 1.0) < 1e-6, \
        f"완전 분리 AUPRC 실패: {auprc_perfect}"

    # 역순: AUROC=0.0
    ys_rev = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    auroc_worst = compute_auroc(yt, ys_rev)
    assert auroc_worst is not None and abs(auroc_worst - 0.0) < 1e-9, \
        f"역순 AUROC 실패: {auroc_worst}"

    # 단일 class (positive only): NOT_APPLICABLE
    yt_pos = np.array([1, 1, 1])
    ys_pos = np.array([0.9, 0.8, 0.7])
    assert compute_auroc(yt_pos, ys_pos) is None, "positive only AUROC은 None이어야 함"

    # 단일 class (negative only): NOT_APPLICABLE
    yt_neg = np.array([0, 0, 0])
    assert compute_auroc(yt_neg, ys_pos) is None, "negative only AUROC은 None이어야 함"
    assert compute_auprc(yt_neg, ys_pos) is None, "negative only AUPRC은 None이어야 함"

    # tie 처리: 동점 score 케이스 (AUROC=0.5 기대)
    yt_tie = np.array([1, 0])
    ys_tie = np.array([0.5, 0.5])
    auroc_tie = compute_auroc(yt_tie, ys_tie)
    assert auroc_tie is not None and abs(auroc_tie - 0.5) < 1e-9, \
        f"동점 AUROC 실패: {auroc_tie}"

    print("[INFO] numpy metric self-check: 5개 케이스 PASS")


# ──────────────────────────────────────────────────────────
# output guard
# ──────────────────────────────────────────────────────────
def check_output_guard(output_root: Path) -> None:
    if output_root.exists():
        print(f"[ERROR] output root가 이미 존재합니다: {output_root}")
        print("[ERROR] 기존 파일 덮어쓰기 금지. 즉시 중단합니다.")
        sys.exit(1)


# ──────────────────────────────────────────────────────────
# Argument Parser
# ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Phase 8.5B crop-level metric calculation (numpy, sklearn 미사용). "
            "--run-metric과 --confirm-run 둘 다 없으면 dry-run만 수행합니다."
        )
    )
    parser.add_argument("--run-metric", action="store_true",
                        help="[필수] metric 계산 실행 플래그 1.")
    parser.add_argument("--confirm-run", action="store_true",
                        help="[필수] metric 계산 실행 플래그 2.")
    parser.add_argument("--score-csv", default=DEFAULT_SCORE_CSV)
    parser.add_argument("--preflight-summary", default=DEFAULT_PREFLIGHT_SUMMARY)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


# ──────────────────────────────────────────────────────────
# MD 보고서 생성
# ──────────────────────────────────────────────────────────
def generate_md_report(crop_results: list, summary: dict, timestamp: str) -> str:
    final_status = summary.get("final_status", "UNKNOWN")
    status_line = "PASS_WITH_PATIENT_LEVEL_NEEDS_REVIEW" == final_status

    header = f"""# Phase 8.5B Metric Calculation Report

생성 시각: {timestamp}

---

## 최종 판정: {final_status}

- crop-level metric 계산: 완료
- patient-level metric 계산: NEEDS_REVIEW (patient label 정의 미확정)
- threshold/p95/p99/hit-rate/recall: 계산 안 함
- sklearn 사용: False (numpy 직접 구현)

---

## 입력

- score CSV: `{summary.get('input_score_csv_path', '')}`
- crop row count: {summary.get('crop_row_count', '')}
- crop positive: {summary.get('crop_positive_count', '')}
- crop negative: {summary.get('crop_negative_count', '')}

---

## binary label 정의

- sampling_label == "positive" → 1
- sampling_label == "hard_negative" → 0

---

## numpy metric 구현 self-check

5개 케이스 모두 통과: 완전분리(AUROC=1.0), 역순(AUROC=0.0), positive-only(None), negative-only(None), 동점(AUROC=0.5)

---

## Crop-level Metric 결과

| score_column | AUROC | AUPRC |
|---|---|---|
"""
    rows = []
    for r in crop_results:
        auroc_str = f"{r['auroc']:.6f}" if r["auroc"] is not None else "NOT_APPLICABLE"
        auprc_str = f"{r['auprc']:.6f}" if r["auprc"] is not None else "NOT_APPLICABLE"
        rows.append(f"| {r['score_column']} | {auroc_str} | {auprc_str} |")

    patient_section = """
---

## Patient-level Metric

**NEEDS_REVIEW**

- 이유: patient-level label 정의 미명시 (Phase 8.5A preflight에서 확정되지 않음)
- blocker: missing_patient_label_definition
- "positive crop 하나라도 있으면 patient positive" 기준 사용 금지
- patient-level AUROC/AUPRC 계산 수행 안 함

---

## threshold / hit-rate 미수행 확인

- threshold_calculated: False
- p95_calculated: False
- p99_calculated: False
- hit_rate_calculated: False
- recall_calculated: False
- threshold_sweep_executed: False

---

## 다음 단계

patient-level label 정의 확정 후 Phase 8.5C patient-level metric 계산 진행.
"""
    return header + "\n".join(rows) + patient_section


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def main():
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    args = parse_args()

    # ── 이중 플래그 guard
    if not (args.run_metric and args.confirm_run):
        print("[DRY-RUN] --run-metric과 --confirm-run 둘 다 필요합니다.")
        print("[DRY-RUN] 이번 실행은 dry-run 보고만 수행합니다.")
        print("[DRY-RUN] 실제 metric 계산은 수행되지 않았습니다.")
        print(
            "[DRY-RUN] 실행 명령: "
            "python scripts/phase8_5b_metric_calculation.py "
            "--run-metric --confirm-run"
        )
        sys.exit(0)

    project_root = Path(__file__).resolve().parent.parent

    score_csv_path = (
        Path(args.score_csv) if Path(args.score_csv).is_absolute()
        else project_root / args.score_csv
    )
    preflight_summary_path = (
        Path(args.preflight_summary) if Path(args.preflight_summary).is_absolute()
        else project_root / args.preflight_summary
    )
    output_root = (
        Path(args.output_root) if Path(args.output_root).is_absolute()
        else project_root / args.output_root
    )

    print(f"[INFO] Phase 8.5B metric calculation 시작: {timestamp}")
    print(f"[INFO] score_csv: {score_csv_path}")
    print(f"[INFO] output_root: {output_root}")

    # ── output guard
    check_output_guard(output_root)

    # ── preflight summary 확인
    if not preflight_summary_path.exists():
        print(f"[ERROR] preflight summary 없음: {preflight_summary_path}")
        sys.exit(1)
    with open(str(preflight_summary_path)) as f:
        preflight = json.load(f)
    if not preflight.get("phase_8_5b_allowed"):
        print("[ERROR] phase_8_5b_allowed=False. 실행 불가.")
        sys.exit(1)
    print(f"[INFO] preflight PASS 확인: phase_8_5b_allowed=True")

    # ── score CSV 로드
    if not score_csv_path.exists():
        print(f"[ERROR] score CSV 없음: {score_csv_path}")
        sys.exit(1)
    df = pd.read_csv(str(score_csv_path))
    print(f"[INFO] score CSV 로드: {len(df)} rows")

    # ── row count 확인
    if len(df) != EXPECTED_ROW_COUNT:
        print(f"[ERROR] row count mismatch: {len(df)} != {EXPECTED_ROW_COUNT}")
        sys.exit(1)

    # ── binary label 생성
    def to_binary(v):
        s = str(v).strip().lower()
        if s == "positive":
            return 1
        if s == "hard_negative":
            return 0
        return -1

    df["_binary_label"] = df["sampling_label"].apply(to_binary)
    invalid = int((df["_binary_label"] == -1).sum())
    if invalid > 0:
        print(f"[ERROR] binary label 변환 실패: {invalid} rows")
        sys.exit(1)

    # ── class balance 확인
    pos_cnt = int((df["_binary_label"] == 1).sum())
    neg_cnt = int((df["_binary_label"] == 0).sum())
    print(f"[INFO] class balance: positive={pos_cnt}, negative={neg_cnt}")
    if pos_cnt != EXPECTED_POSITIVE_COUNT or neg_cnt != EXPECTED_NEGATIVE_COUNT:
        print(f"[WARN] class balance 기대값과 다름 "
              f"(expected pos={EXPECTED_POSITIVE_COUNT}, neg={EXPECTED_NEGATIVE_COUNT})")

    # ── numpy metric self-check
    print("[INFO] numpy metric self-check 실행...")
    run_self_check()

    # ── crop-level metric 계산
    y_true = df["_binary_label"].values
    crop_results = []
    print("[INFO] crop-level metric 계산 시작...")
    for col in SCORE_COLUMNS:
        if col not in df.columns:
            print(f"[WARN] score column 없음: {col}")
            crop_results.append({
                "score_column": col,
                "auroc": None,
                "auprc": None,
                "auroc_status": "MISSING_COLUMN",
                "auprc_status": "MISSING_COLUMN",
                "n_total": len(df),
                "n_positive": pos_cnt,
                "n_negative": neg_cnt,
            })
            continue

        y_score = df[col].values.astype(np.float64)
        auroc = compute_auroc(y_true, y_score)
        auprc = compute_auprc(y_true, y_score)

        auroc_status = "OK" if auroc is not None else "NOT_APPLICABLE"
        auprc_status = "OK" if auprc is not None else "NOT_APPLICABLE"

        auroc_str = f"{auroc:.6f}" if auroc is not None else "NOT_APPLICABLE"
        auprc_str = f"{auprc:.6f}" if auprc is not None else "NOT_APPLICABLE"
        print(f"[INFO]   {col}: AUROC={auroc_str}, AUPRC={auprc_str}")

        crop_results.append({
            "score_column": col,
            "auroc": auroc,
            "auprc": auprc,
            "auroc_status": auroc_status,
            "auprc_status": auprc_status,
            "n_total": len(df),
            "n_positive": pos_cnt,
            "n_negative": neg_cnt,
        })

    # ── patient-level: NEEDS_REVIEW 기록
    patient_rows = []
    for col in SCORE_COLUMNS:
        patient_rows.append({
            "score_column": col,
            "aggregation": "NEEDS_REVIEW",
            "auroc": "NEEDS_REVIEW",
            "auprc": "NEEDS_REVIEW",
            "status": "NEEDS_REVIEW",
            "reason": "missing_patient_label_definition",
            "note": (
                "patient-level label 정의 미확정. "
                "'positive crop 하나라도 있으면 patient positive' 기준 사용 금지. "
                "Phase 8.5C에서 label 정의 확정 후 계산 예정."
            ),
        })

    # ── output root 생성 (계산 완료 후)
    output_root.mkdir(parents=True, exist_ok=False)
    print(f"[INFO] output_root 생성: {output_root}")

    # ── crop-level CSV 저장
    crop_df = pd.DataFrame(crop_results)
    crop_csv_path = output_root / OUTPUT_CROP_CSV
    crop_df.to_csv(str(crop_csv_path), index=False)
    print(f"[INFO] crop-level CSV 저장: {crop_csv_path}")

    # ── patient-level CSV 저장 (NEEDS_REVIEW)
    patient_df = pd.DataFrame(patient_rows)
    patient_csv_path = output_root / OUTPUT_PATIENT_CSV
    patient_df.to_csv(str(patient_csv_path), index=False)
    print(f"[INFO] patient-level CSV 저장 (NEEDS_REVIEW): {patient_csv_path}")

    # ── summary JSON 저장
    crop_metric_dict = {
        r["score_column"]: {
            "auroc": r["auroc"],
            "auprc": r["auprc"],
            "auroc_status": r["auroc_status"],
            "auprc_status": r["auprc_status"],
        }
        for r in crop_results
    }
    summary = {
        "phase": "8.5B",
        "timestamp": timestamp,
        "metric_calculation_executed": True,
        "crop_level_metric_calculation_executed": True,
        "patient_level_metric_calculation_executed": False,
        "patient_level_status": "NEEDS_REVIEW",
        "patient_level_blocker": "missing_patient_label_definition",
        "sklearn_used": False,
        "sklearn_available": False,
        "numpy_metric_implementation_used": True,
        "numpy_self_check_pass": True,
        "threshold_calculated": False,
        "hit_rate_calculated": False,
        "recall_calculated": False,
        "threshold_sweep_executed": False,
        "training_executed": False,
        "model_forward_executed": False,
        "checkpoint_created": False,
        "input_score_csv_path": str(score_csv_path),
        "crop_row_count": len(df),
        "crop_positive_count": pos_cnt,
        "crop_negative_count": neg_cnt,
        "score_columns": SCORE_COLUMNS,
        "crop_level_metrics": crop_metric_dict,
        "patient_aggregations": "NEEDS_REVIEW",
        "output_files": {
            "crop_level_csv": str(crop_csv_path),
            "patient_level_csv": str(patient_csv_path),
            "summary_json": str(output_root / OUTPUT_JSON),
            "md_report": str(output_root / OUTPUT_MD),
        },
        "final_status": "PASS_WITH_PATIENT_LEVEL_NEEDS_REVIEW",
    }

    json_path = output_root / OUTPUT_JSON
    with open(str(json_path), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"[INFO] summary JSON 저장: {json_path}")

    # ── MD 보고서 저장
    md_text = generate_md_report(crop_results, summary, timestamp)
    md_path = output_root / OUTPUT_MD
    with open(str(md_path), "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"[INFO] MD 보고서 저장: {md_path}")

    # ── 최종 요약 출력
    print(f"\n[DONE] Phase 8.5B metric calculation 완료")
    print(f"  final_status: {summary['final_status']}")
    print(f"  crop-level metric:")
    for r in crop_results:
        auroc_str = f"{r['auroc']:.6f}" if r["auroc"] is not None else "NOT_APPLICABLE"
        auprc_str = f"{r['auprc']:.6f}" if r["auprc"] is not None else "NOT_APPLICABLE"
        print(f"    {r['score_column']}: AUROC={auroc_str}, AUPRC={auprc_str}")
    print(f"  patient-level: NEEDS_REVIEW (missing_patient_label_definition)")
    print(f"  threshold/p95/p99/hit-rate/recall: 계산 안 함")
    print(f"  sklearn_used: False")


if __name__ == "__main__":
    main()
