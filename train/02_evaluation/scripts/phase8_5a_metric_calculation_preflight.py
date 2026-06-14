"""
Phase 8.5A Metric Calculation Preflight

목적:
  Phase 8.5B metric 계산 전 입력/정의/출력 guard 검증.
  metric 계산 자체는 수행하지 않는다.

금지 사항:
  - AUROC/AUPRC/threshold/p95/p99/hit-rate 계산
  - model forward / training / backward / optimizer step
  - checkpoint 생성
  - score CSV 수정/재생성
  - 기존 Phase 6/7/8 output 수정/삭제/덮어쓰기
  - v2/v2v2 접근
  - adjusted score 생성
  - Phase 8.5B 실행

실행 guard:
  --run-preflight와 --confirm-run 둘 다 없으면 dry-run 보고만 수행.
  실제 preflight 검증 + 출력 생성은 두 플래그가 모두 있을 때만 수행.
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────
EXPECTED_ROW_COUNT = 143735
EXPECTED_PATIENT_COUNT = 154
EXPECTED_POSITIVE_COUNT = 51335
EXPECTED_HARD_NEGATIVE_COUNT = 92400

METRIC_SCORE_COLUMNS = [
    "crop_score_l1_mean",
    "crop_score_l1_max",
    "crop_score_mse_mean",
    "lung_channels_l1_mean",
    "mediastinal_channels_l1_mean",
]

PATIENT_AGGREGATION_CANDIDATES = [
    "patient_mean",
    "patient_top10pct_mean",
    "patient_top5pct_mean",
    "patient_top1pct_mean",
    "patient_max",
]

CROP_LEVEL_METRIC_CANDIDATES = ["AUROC", "AUPRC"]
PATIENT_LEVEL_METRIC_CANDIDATES = ["AUROC", "AUPRC"]

PHASE_8_5B_ARTIFACTS = [
    "phase8_5b_metric_results_crop_level.csv",
    "phase8_5b_metric_results_patient_level.csv",
    "phase8_5b_metric_calculation_summary.json",
    "phase8_5b_metric_calculation_report.md",
]

DEFAULT_SCORE_CSV = (
    "outputs/second-stage-lesion-refiner-v1/scores/"
    "phase8_4_stage2_full_scoring_v1/"
    "phase8_4_stage2_full_scoring_v1.csv"
)
DEFAULT_VALIDATION_SUMMARY = (
    "outputs/second-stage-lesion-refiner-v1/review_annotations/"
    "phase8_4_stage2_full_scoring_v1/"
    "phase8_4c_artifact_validation_summary.json"
)
DEFAULT_OUTPUT_ROOT = (
    "outputs/second-stage-lesion-refiner-v1/review_annotations/"
    "phase8_5a_metric_calculation_preflight_v1/"
)

OUTPUT_MD_NAME = "phase8_5a_metric_calculation_preflight_report.md"
OUTPUT_JSON_NAME = "phase8_5a_metric_calculation_preflight_summary.json"


# ──────────────────────────────────────────────────────────
# Argument Parser
# ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Phase 8.5A metric calculation preflight. "
            "--run-preflight과 --confirm-run 둘 다 없으면 dry-run만 수행합니다."
        )
    )
    parser.add_argument(
        "--run-preflight",
        action="store_true",
        help="[필수] preflight 실행 플래그 1. --confirm-run과 함께 사용해야 합니다.",
    )
    parser.add_argument(
        "--confirm-run",
        action="store_true",
        help="[필수] preflight 실행 플래그 2. --run-preflight과 함께 사용해야 합니다.",
    )
    parser.add_argument(
        "--score-csv",
        default=DEFAULT_SCORE_CSV,
        help="Phase 8.4 score CSV 경로",
    )
    parser.add_argument(
        "--validation-summary",
        default=DEFAULT_VALIDATION_SUMMARY,
        help="Phase 8.4C artifact validation summary JSON 경로",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="preflight 출력 root 디렉토리",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────
# label 정규화
# ──────────────────────────────────────────────────────────
def _normalize_label(val) -> str:
    s = str(val).strip()
    try:
        f = float(s)
        if f == 1.0:
            return "positive"
        if f == 0.0:
            return "hard_negative"
    except (ValueError, TypeError):
        pass
    sl = s.lower()
    if sl in ("positive", "hard_negative"):
        return sl
    return "unknown"


# ──────────────────────────────────────────────────────────
# output guard
# ──────────────────────────────────────────────────────────
def check_output_guard(output_root: Path) -> None:
    if output_root.exists():
        print(f"[ERROR] output root가 이미 존재합니다: {output_root}")
        print("[ERROR] 기존 파일 덮어쓰기 금지. 즉시 중단합니다.")
        sys.exit(1)
    for fname in [OUTPUT_MD_NAME, OUTPUT_JSON_NAME]:
        fpath = output_root / fname
        if fpath.exists():
            print(f"[ERROR] 출력 파일이 이미 존재합니다: {fpath}")
            sys.exit(1)


# ──────────────────────────────────────────────────────────
# preflight 검증
# ──────────────────────────────────────────────────────────
def run_preflight_checks(
    score_csv_path: Path,
    validation_summary_path: Path,
) -> dict:
    """
    20개 preflight 항목 검증.
    결과 dict 반환: {item_key: {"pass": bool, ...}}
    """
    results = {}

    # 1. Phase 8.4C validation summary 존재
    exists_vs = validation_summary_path.exists()
    results["01_validation_summary_exists"] = {"pass": exists_vs}
    if exists_vs:
        print(f"[INFO] Phase 8.4C validation summary 존재: {validation_summary_path}")
    else:
        print(f"[ERROR] Phase 8.4C validation summary 없음: {validation_summary_path}")

    # 2. Phase 8.4C 최종 판정 PASS
    phase84c_pass = False
    if exists_vs:
        with open(str(validation_summary_path)) as f:
            vs = json.load(f)
        phase84c_pass = bool(vs.get("overall_pass") is True or vs.get("verdict") == "PASS")
        results["02_validation_summary_pass"] = {"actual": vs.get("verdict"), "pass": phase84c_pass}
        print(f"[INFO] Phase 8.4C 최종 판정: {vs.get('verdict')} → pass={phase84c_pass}")
    else:
        results["02_validation_summary_pass"] = {"pass": False, "note": "summary 없음"}

    # 3. score CSV 존재
    exists_csv = score_csv_path.exists()
    results["03_score_csv_exists"] = {"pass": bool(exists_csv)}
    if exists_csv:
        print(f"[INFO] score CSV 존재: {score_csv_path}")
    else:
        print(f"[ERROR] score CSV 없음: {score_csv_path}")
        # 이후 항목은 모두 FAIL 처리
        for k in [f"{i:02d}_" for i in range(4, 21)]:
            results[k + "skipped"] = {"pass": False, "note": "score CSV 없음"}
        return results

    df = pd.read_csv(str(score_csv_path))
    print(f"[INFO] score CSV 로드 완료: {len(df)} rows")

    # 4. row count
    row_ok = len(df) == EXPECTED_ROW_COUNT
    results["04_row_count"] = {"actual": len(df), "expected": EXPECTED_ROW_COUNT, "pass": row_ok}
    print(f"[INFO] row count: {len(df)} (expected={EXPECTED_ROW_COUNT}) → pass={row_ok}")

    # 5. patient_id unique
    pat_unique = int(df["patient_id"].nunique())
    pat_ok = pat_unique == EXPECTED_PATIENT_COUNT
    results["05_patient_id_unique"] = {"actual": pat_unique, "expected": EXPECTED_PATIENT_COUNT, "pass": pat_ok}
    print(f"[INFO] patient_id unique: {pat_unique} → pass={pat_ok}")

    # 6. sampling_label counts
    pos_cnt = int((df["sampling_label"] == "positive").sum())
    neg_cnt = int((df["sampling_label"] == "hard_negative").sum())
    sl_ok = pos_cnt == EXPECTED_POSITIVE_COUNT and neg_cnt == EXPECTED_HARD_NEGATIVE_COUNT
    results["06_sampling_label_counts"] = {
        "positive": pos_cnt,
        "hard_negative": neg_cnt,
        "pass": sl_ok,
    }
    print(f"[INFO] sampling_label: positive={pos_cnt}, hard_negative={neg_cnt} → pass={sl_ok}")

    # 7. label/sampling_label 정합성
    nl = df["label"].apply(_normalize_label)
    ns = df["sampling_label"].apply(lambda v: str(v).strip().lower())
    mismatch = int((nl != ns).sum())
    lc_ok = mismatch == 0
    results["07_label_consistency"] = {"mismatch_count": mismatch, "pass": lc_ok}
    print(f"[INFO] label/sampling_label mismatch: {mismatch} → pass={lc_ok}")

    # 8. has_nan sum
    nan_sum = int(df["has_nan"].sum())
    results["08_has_nan_sum"] = {"actual": nan_sum, "pass": nan_sum == 0}
    print(f"[INFO] has_nan sum: {nan_sum} → pass={nan_sum==0}")

    # 9. has_inf sum
    inf_sum = int(df["has_inf"].sum())
    results["09_has_inf_sum"] = {"actual": inf_sum, "pass": inf_sum == 0}
    print(f"[INFO] has_inf sum: {inf_sum} → pass={inf_sum==0}")

    # 10. scoring_status 전부 PASS
    non_pass = int((df["scoring_status"] != "PASS").sum())
    results["10_scoring_status_all_pass"] = {"non_pass_count": non_pass, "pass": non_pass == 0}
    print(f"[INFO] scoring_status non-PASS: {non_pass} → pass={non_pass==0}")

    # 11. metric score column 후보 존재 확인
    missing_cols = [c for c in METRIC_SCORE_COLUMNS if c not in df.columns]
    col_ok = len(missing_cols) == 0
    results["11_metric_score_columns_exist"] = {
        "columns": METRIC_SCORE_COLUMNS,
        "missing": missing_cols,
        "pass": col_ok,
    }
    print(f"[INFO] metric score columns 존재: missing={missing_cols} → pass={col_ok}")

    # 12. score columns NaN/Inf 없음
    present_cols = [c for c in METRIC_SCORE_COLUMNS if c in df.columns]
    def is_inf(x):
        try:
            return math.isinf(float(x))
        except Exception:
            return False

    sc_nan = int(df[present_cols].isnull().sum().sum())
    sc_inf = int(sum(is_inf(x) for col in present_cols for x in df[col]))
    sc_ok = sc_nan == 0 and sc_inf == 0
    results["12_score_columns_no_nan_inf"] = {
        "nan_count": sc_nan,
        "inf_count": sc_inf,
        "pass": sc_ok,
    }
    print(f"[INFO] score columns NaN={sc_nan}, Inf={sc_inf} → pass={sc_ok}")

    # 13. binary label 정의 확정 (문서화 only)
    results["13_binary_label_definition"] = {
        "positive": 1,
        "hard_negative": 0,
        "note": "positive=1, hard_negative=0으로 고정. 이번 preflight에서 실제 변환 수행 안 함.",
        "pass": True,
    }
    print("[INFO] binary label 정의 확정: positive=1, hard_negative=0")

    # 14. patient-level aggregation 후보 정의 (문서화 only)
    results["14_patient_aggregation_candidates"] = {
        "candidates": PATIENT_AGGREGATION_CANDIDATES,
        "note": "Phase 8.5B에서 계산 예정. 이번 preflight에서 계산 수행 안 함.",
        "pass": True,
    }
    print(f"[INFO] patient aggregation 후보: {PATIENT_AGGREGATION_CANDIDATES}")

    # 15. crop-level metric 후보 정의 (문서화 only)
    results["15_crop_level_metric_candidates"] = {
        "candidates": CROP_LEVEL_METRIC_CANDIDATES,
        "note": "Phase 8.5B에서 계산 예정. 이번 preflight에서 계산 수행 안 함.",
        "pass": True,
    }
    print(f"[INFO] crop-level metric 후보: {CROP_LEVEL_METRIC_CANDIDATES}")

    # 16. patient-level metric 후보 정의 (문서화 only)
    results["16_patient_level_metric_candidates"] = {
        "candidates": PATIENT_LEVEL_METRIC_CANDIDATES,
        "note": "Phase 8.5B에서 계산 예정. 이번 preflight에서 계산 수행 안 함.",
        "pass": True,
    }
    print(f"[INFO] patient-level metric 후보: {PATIENT_LEVEL_METRIC_CANDIDATES}")

    # 17. threshold-based metric 제외 명시
    results["17_threshold_metrics_excluded"] = {
        "excluded": ["p95", "p99", "hit-rate", "recall"],
        "note": "이번 preflight 및 Phase 8.5B 초기 계산에서 threshold-based metric 제외.",
        "pass": True,
    }
    print("[INFO] threshold-based metric (p95/p99/hit-rate/recall) 제외 확인")

    # 18. output root guard (이미 check_output_guard에서 처리)
    results["18_output_root_guard"] = {
        "note": "output root 신규 생성 예정. 기존 파일 덮어쓰기 금지.",
        "pass": True,
    }

    # 19. Phase 8.5B 산출물 이름 문서화
    results["19_phase85b_artifact_names"] = {
        "artifacts": PHASE_8_5B_ARTIFACTS,
        "note": "Phase 8.5B에서 생성 예정. 이번 8.5A에서는 생성 안 함.",
        "pass": True,
    }
    print(f"[INFO] Phase 8.5B 예정 산출물: {PHASE_8_5B_ARTIFACTS}")

    # 20. 8.5B 산출물 미생성 확인
    results["20_phase85b_artifacts_not_created"] = {
        "artifacts_created": [],
        "metric_calculation_executed": False,
        "pass": True,
    }
    print("[INFO] Phase 8.5B 산출물 미생성 확인 OK")

    return results


# ──────────────────────────────────────────────────────────
# MD 보고서 생성
# ──────────────────────────────────────────────────────────
def generate_md_report(
    results: dict,
    preflight_pass: bool,
    score_csv_path: str,
    validation_summary_path: str,
    timestamp: str,
) -> str:
    pass_str = "PASS" if preflight_pass else "FAIL"

    rows = []
    for k, v in results.items():
        p = v.get("pass", False)
        status = "PASS" if p else "FAIL"
        note = ""
        if "actual" in v and "expected" in v:
            note = f"actual={v['actual']}, expected={v['expected']}"
        elif "actual" in v:
            note = f"actual={v['actual']}"
        elif "mismatch_count" in v:
            note = f"mismatch={v['mismatch_count']}"
        elif "missing" in v:
            note = f"missing={v['missing']}"
        elif "nan_count" in v:
            note = f"nan={v['nan_count']}, inf={v['inf_count']}"
        elif "candidates" in v:
            note = ", ".join(v["candidates"])
        elif "note" in v:
            note = v["note"]
        rows.append(f"| {k} | {status} | {note} |")

    table = "\n".join(rows)

    md = f"""# Phase 8.5A Metric Calculation Preflight Report

생성 시각: {timestamp}

---

## 최종 판정: {pass_str}

Phase 8.5B metric calculation 실행 가능 여부: {"**가능**" if preflight_pass else "**불가 — FAIL 항목 확인 필요**"}

---

## 입력

- score CSV: `{score_csv_path}`
- Phase 8.4C validation summary: `{validation_summary_path}`

---

## 검증 항목별 결과

| 항목 | 판정 | 상세 |
|------|------|------|
{table}

---

## metric 계산 기준

### binary label 정의
- positive = 1 (sampling_label == "positive")
- hard_negative = 0 (sampling_label == "hard_negative")

### score column 후보 (crop-level input)
| 컬럼명 | 정의 |
|--------|------|
| crop_score_l1_mean | mean(|input - recon|), 6채널 전체 |
| crop_score_l1_max | max(|input - recon|), 6채널 전체 |
| crop_score_mse_mean | mean((input-recon)^2), 6채널 전체 |
| lung_channels_l1_mean | channels 0~2 mean(|input - recon|) |
| mediastinal_channels_l1_mean | channels 3~5 mean(|input - recon|) |

### patient-level aggregation 후보
- patient_mean: 환자 내 crop score 평균
- patient_top10pct_mean: 상위 10% crop score 평균
- patient_top5pct_mean: 상위 5% crop score 평균
- patient_top1pct_mean: 상위 1% crop score 평균
- patient_max: 환자 내 crop score 최댓값

### crop-level metric 후보
- AUROC, AUPRC

### patient-level metric 후보
- AUROC, AUPRC

### 제외 항목
- p95 / p99 / hit-rate / recall: 이번 Phase 8.5B 초기 계산에서 제외

---

## Phase 8.5B 예정 산출물

| 파일명 | 설명 |
|--------|------|
| phase8_5b_metric_results_crop_level.csv | crop-level AUROC/AUPRC 결과 |
| phase8_5b_metric_results_patient_level.csv | patient-level aggregation별 AUROC/AUPRC 결과 |
| phase8_5b_metric_calculation_summary.json | 계산 요약 JSON |
| phase8_5b_metric_calculation_report.md | 보고서 |

**이번 8.5A에서는 위 산출물을 생성하지 않았음.**

---

## metric 미수행 확인

- metric_calculation_executed: False
- auroc_calculated: False
- auprc_calculated: False
- threshold_calculated: False
- training_executed: False

---

## 다음 단계

{"Phase 8.5B metric calculation 사용자 승인 요청 후 진행." if preflight_pass else "FAIL 항목 해소 후 preflight 재실행 필요."}
"""
    return md


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def main():
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    args = parse_args()

    # ── 이중 플래그 guard
    if not (args.run_preflight and args.confirm_run):
        print("[DRY-RUN] --run-preflight과 --confirm-run 둘 다 필요합니다.")
        print("[DRY-RUN] 이번 실행은 dry-run 보고만 수행합니다.")
        print("[DRY-RUN] 실제 preflight 검증 및 출력 생성은 수행되지 않았습니다.")
        print(
            "[DRY-RUN] 실행 명령: "
            "python scripts/phase8_5a_metric_calculation_preflight.py "
            "--run-preflight --confirm-run"
        )
        sys.exit(0)

    project_root = Path(__file__).resolve().parent.parent

    score_csv_path = (
        Path(args.score_csv) if Path(args.score_csv).is_absolute()
        else project_root / args.score_csv
    )
    validation_summary_path = (
        Path(args.validation_summary) if Path(args.validation_summary).is_absolute()
        else project_root / args.validation_summary
    )
    output_root = (
        Path(args.output_root) if Path(args.output_root).is_absolute()
        else project_root / args.output_root
    )

    print(f"[INFO] Phase 8.5A metric calculation preflight 시작: {timestamp}")
    print(f"[INFO] score_csv: {score_csv_path}")
    print(f"[INFO] validation_summary: {validation_summary_path}")
    print(f"[INFO] output_root: {output_root}")

    # ── output guard
    check_output_guard(output_root)

    # ── preflight 검증
    print("[INFO] preflight 검증 시작...")
    results = run_preflight_checks(score_csv_path, validation_summary_path)

    preflight_pass = all(v.get("pass", False) for v in results.values())
    blockers = [k for k, v in results.items() if not v.get("pass", False)]
    print(f"[INFO] preflight 판정: {'PASS' if preflight_pass else 'FAIL'}")
    if blockers:
        for b in blockers:
            print(f"[ERROR] blocker: {b}")

    # ── 출력 root 생성
    output_root.mkdir(parents=True, exist_ok=False)
    print(f"[INFO] output_root 생성: {output_root}")

    # ── summary JSON 저장
    summary = {
        "phase": "8.5A",
        "timestamp": timestamp,
        "score_csv_path": str(score_csv_path),
        "validation_summary_path": str(validation_summary_path),
        "preflight_pass": preflight_pass,
        "verdict": "PASS" if preflight_pass else "FAIL",
        "phase_8_5b_allowed": preflight_pass,
        "total_items_checked": len(results),
        "total_pass": sum(1 for v in results.values() if v.get("pass", False)),
        "total_fail": sum(1 for v in results.values() if not v.get("pass", False)),
        "blockers": blockers,
        "metric_calculation_executed": False,
        "auroc_calculated": False,
        "auprc_calculated": False,
        "threshold_calculated": False,
        "training_executed": False,
        "backward_executed": False,
        "optimizer_step_executed": False,
        "checkpoint_created": False,
        "phase85b_artifacts_created": [],
        "results": results,
    }

    json_path = output_root / OUTPUT_JSON_NAME
    with open(str(json_path), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"[INFO] summary JSON 저장: {json_path}")

    # ── MD 보고서 저장
    md_text = generate_md_report(
        results=results,
        preflight_pass=preflight_pass,
        score_csv_path=str(score_csv_path),
        validation_summary_path=str(validation_summary_path),
        timestamp=timestamp,
    )
    md_path = output_root / OUTPUT_MD_NAME
    with open(str(md_path), "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"[INFO] MD 보고서 저장: {md_path}")

    print(f"\n[DONE] Phase 8.5A preflight 완료")
    print(f"  verdict: {'PASS' if preflight_pass else 'FAIL'}")
    print(f"  phase_8_5b_allowed: {preflight_pass}")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")


if __name__ == "__main__":
    main()
