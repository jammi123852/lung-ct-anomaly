"""
analyze_lesion_screening.py: 병변 평가 결과를 1차 스크리닝 관점에서 추가 분석한다.

목적:
- 이미 저장된 lesion score CSV(read-only)로 recall/coverage 중심 지표를 계산한다.
- 1차 스크리닝 = "병변 의심 부위를 최대한 놓치지 않는 것"이 목적이므로 recall/coverage를 우선한다.

안전 원칙:
- score 재계산 / FeatureExtractor / score_patient 재실행 없음. score CSV는 read-only.
- 기존 결과 파일을 수정·삭제하지 않는다. 신규 분석 파일만 생성한다.

지표 정의 (positive = padim_score >= threshold):
- lesion patch recall = (patch_label=1 & positive) / (patch_label=1)            [micro]
- lesion slice recall = lesion slice 중 그 slice에서 lesion patch가 positive로 잡힌 비율
                        (lesion slice = local_z 그룹에 patch_label=1 존재)
- patient coverage    = 환자별 [잡은 lesion slice / 전체 lesion slice]의 평균
- patient hit rate    = lesion patch가 하나라도 positive인 환자 / 전체 환자
- top-k coverage      = 환자별 score 상위 k patch에 lesion patch가 포함되는 환자 / 전체 환자 (k=10/30/50)
"""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
LESION_SCORE_DIR_V1 = REPO_ROOT / "outputs" / "position-aware-padim-v1" / "scores" / "padim_v1" / "lesion_by_patient"
LESION_SCORE_DIR_V2 = REPO_ROOT / "outputs" / "position-aware-padim-v1" / "scores" / "padim_v1" / "lesion_v2_by_patient"
EVAL_DIR_V1 = REPO_ROOT / "outputs" / "position-aware-padim-v1" / "evaluation" / "lesion_subset"
EVAL_DIR_V2 = REPO_ROOT / "outputs" / "position-aware-padim-v1" / "evaluation" / "lesion_subset_v2"

TOPK = [10, 30, 50]
USECOLS = ["patient_id", "safe_id", "group", "local_z", "padim_score", "patch_label"]


def load_thresholds(eval_dir: Path, v2_prefix: str = "") -> dict:
    """fast summary json에서 p95/p99 threshold를 read-only로 읽는다."""
    thr = {}
    for tag in ["p95", "p99"]:
        p = eval_dir / f"lesion_eval_{v2_prefix}{tag}_fast_summary.json"
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        thr[tag] = float(d["threshold_value"])
    return thr


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="lesion 스크리닝 분석")
    parser.add_argument(
        "--dataset-profile",
        type=str,
        default="v1_model_roi",
        choices=["v1_model_roi", "v2_roi_0_0"],
        help="분석할 데이터셋 profile (기본값: v1_model_roi).",
    )
    parser.add_argument(
        "--score-dir",
        type=str,
        default=None,
        help="lesion score 경로 오버라이드 (기본: dataset-profile 기반 자동 결정).",
    )
    parser.add_argument(
        "--evaluation-dir",
        type=str,
        default=None,
        help="evaluation 출력 경로 오버라이드 (기본: dataset-profile 기반 자동 결정).",
    )
    parser.add_argument(
        "--threshold-json",
        type=str,
        default=None,
        help="normal threshold JSON 경로 직접 지정 (이 옵션이 있으면 fast_summary.json 대신 사용).",
    )
    args = parser.parse_args()
    is_v2 = (args.dataset_profile == "v2_roi_0_0")
    LESION_SCORE_DIR = LESION_SCORE_DIR_V2 if is_v2 else LESION_SCORE_DIR_V1
    EVAL_DIR = EVAL_DIR_V2 if is_v2 else EVAL_DIR_V1
    v2_prefix = "v2_" if is_v2 else ""

    if args.score_dir is not None:
        LESION_SCORE_DIR = REPO_ROOT / args.score_dir
    if args.evaluation_dir is not None:
        EVAL_DIR = REPO_ROOT / args.evaluation_dir

    if args.threshold_json is not None:
        with open(args.threshold_json, encoding="utf-8") as f:
            _tj = json.load(f)
        thresholds = {"p95": float(_tj["threshold_p95"]), "p99": float(_tj["threshold_p99"])}
    else:
        thresholds = load_thresholds(EVAL_DIR, v2_prefix)
    print(f"[screening] dataset_profile={args.dataset_profile}")
    print(f"[screening] thresholds: {thresholds}")

    csv_paths = sorted(glob.glob(str(LESION_SCORE_DIR / "*.csv")))
    n_files = len(csv_paths)
    if n_files == 0:
        raise SystemExit(f"[ERROR] lesion score CSV 없음: {LESION_SCORE_DIR}")
    print(f"[screening] lesion score CSV {n_files}개 read-only 분석")

    agg = {t: {"lp_total": 0, "lp_det": 0, "ls_total": 0, "ls_det": 0,
               "hit": 0, "cov_sum": 0.0, "cov_n": 0,
               "topk_hit": {k: 0 for k in TOPK}} for t in thresholds}
    per_patient_rows = []
    n_patients = 0

    for i, p in enumerate(csv_paths):
        df = pd.read_csv(p, encoding="utf-8-sig", usecols=USECOLS)
        df = df[~df["padim_score"].isna()]
        if len(df) == 0:
            continue
        n_patients += 1

        pid = str(df["patient_id"].iloc[0])
        safe = str(df["safe_id"].iloc[0])
        grp = str(df["group"].iloc[0])
        score = df["padim_score"].values.astype(np.float64)
        is_lesion = (df["patch_label"].values == 1)
        lz = df["local_z"].values
        order = np.argsort(-score)

        lp_total = int(is_lesion.sum())
        lesion_slices = set(lz[is_lesion].tolist())

        rec = {
            "patient_id": pid, "safe_id": safe, "group": grp,
            "n_patch": int(len(df)),
            "lesion_patch_total": lp_total,
            "lesion_slice_total": len(lesion_slices),
        }

        for t, thr in thresholds.items():
            pos = score >= thr
            lp_det = int((is_lesion & pos).sum())
            det_slices = set(lz[is_lesion & pos].tolist())
            n_pos = int(pos.sum())
            n_fp = int((pos & ~is_lesion).sum())

            agg[t]["lp_total"] += lp_total
            agg[t]["lp_det"] += lp_det
            agg[t]["ls_total"] += len(lesion_slices)
            agg[t]["ls_det"] += len(det_slices)
            if lp_det >= 1:
                agg[t]["hit"] += 1
            if lesion_slices:
                cov = len(det_slices) / len(lesion_slices)
                agg[t]["cov_sum"] += cov
                agg[t]["cov_n"] += 1
            else:
                cov = float("nan")
            for k in TOPK:
                if is_lesion[order[:k]].any():
                    agg[t]["topk_hit"][k] += 1

            rec[f"recall_{t}"] = (lp_det / lp_total) if lp_total > 0 else float("nan")
            rec[f"slice_cov_{t}"] = cov
            rec[f"n_positive_{t}"] = n_pos
            rec[f"n_lesion_detected_{t}"] = lp_det
            rec[f"n_fp_{t}"] = n_fp
            rec[f"fp_ratio_{t}"] = (n_fp / n_pos) if n_pos > 0 else float("nan")

        per_patient_rows.append(rec)
        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{n_files}")

    # 전체 지표 요약
    summary_metrics = []
    for t in thresholds:
        a = agg[t]
        summary_metrics.append({
            "threshold_mode": f"normal_val_{t}",
            "threshold_value": thresholds[t],
            "n_patients": n_patients,
            "lesion_patch_recall": (a["lp_det"] / a["lp_total"]) if a["lp_total"] > 0 else None,
            "lesion_slice_recall": (a["ls_det"] / a["ls_total"]) if a["ls_total"] > 0 else None,
            "patient_coverage_mean": (a["cov_sum"] / a["cov_n"]) if a["cov_n"] > 0 else None,
            "patient_hit_rate": a["hit"] / n_patients if n_patients > 0 else None,
            "topk10_coverage": a["topk_hit"][10] / n_patients if n_patients > 0 else None,
            "topk30_coverage": a["topk_hit"][30] / n_patients if n_patients > 0 else None,
            "topk50_coverage": a["topk_hit"][50] / n_patients if n_patients > 0 else None,
            "lesion_patch_total": a["lp_total"],
            "lesion_patch_detected": a["lp_det"],
            "lesion_slice_total": a["ls_total"],
            "lesion_slice_detected": a["ls_det"],
        })

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_metrics)
    summary_csv = EVAL_DIR / "screening_analysis_summary.csv"
    summary_json = EVAL_DIR / "screening_analysis_summary.json"
    per_patient_csv = EVAL_DIR / "per_patient_screening.csv"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(per_patient_rows).to_csv(per_patient_csv, index=False, encoding="utf-8-sig")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({
            "purpose": "1차 스크리닝(병변 의심 부위를 놓치지 않는 것) 관점. recall/coverage 우선, precision은 참고.",
            "positive_definition": "padim_score >= threshold",
            "thresholds": thresholds,
            "metrics": summary_metrics,
            "note": "score CSV read-only 분석. 재계산 없음. 성능 최종 결론 아님.",
        }, f, ensure_ascii=False, indent=2)

    # 콘솔 요약
    print("\n[screening] === 1차 스크리닝 지표 (recall/coverage 우선) ===")
    for m in summary_metrics:
        print(f"  [{m['threshold_mode']}] thr={m['threshold_value']:.4f}")
        print(f"     lesion_patch_recall  = {m['lesion_patch_recall']:.4f}")
        print(f"     lesion_slice_recall  = {m['lesion_slice_recall']:.4f}")
        print(f"     patient_coverage_mean= {m['patient_coverage_mean']:.4f}")
        print(f"     patient_hit_rate     = {m['patient_hit_rate']:.4f}")
        print(f"     topk coverage 10/30/50 = {m['topk10_coverage']:.4f} / {m['topk30_coverage']:.4f} / {m['topk50_coverage']:.4f}")
    print(f"\n[screening] 저장: {summary_csv}")
    print(f"[screening] 저장: {summary_json}")
    print(f"[screening] 저장: {per_patient_csv}")


if __name__ == "__main__":
    main()
