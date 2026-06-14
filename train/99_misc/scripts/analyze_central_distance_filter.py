"""
analyze_central_distance_filter.py: central_distance_ratio_mean 기반 후보 필터
효과 분석 (read-only).

목표:
1. central_distance_ratio_mean 의미 방향성(0=중심/폐문 vs 외곽) 자동 판정.
2. threshold sweep (0.1~0.5)로 top50/top10 분포·FP 제거 효과 측정.
3. tie-breaker(percentile_raw_tiebreak, percentile_central_bonus) 효과 비교.

출력:
- outputs/.../reports/candidate_analysis/central_distance_filter_analysis.csv
- outputs/.../reports/candidate_analysis/central_distance_filter_summary.json

금지:
- 학습/스코어링/시각화 재실행
- 기존 score/candidate CSV 수정
- pip install
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from position_aware_padim.score_aggregator import ScoreAggregator


REPORTS_DIR = REPO_ROOT / "outputs" / "position-aware-padim-v1" / "reports"
ERROR_CSV = REPORTS_DIR / "error.csv"
RUNTIME_CSV = REPORTS_DIR / "runtime_summary.csv"

ERROR_COLUMNS = ["patient_id", "error_type", "error_msg", "file_logical"]
RUNTIME_COLUMNS = ["timestamp", "script", "metric", "value"]

SCRIPT_NAME = "analyze_central_distance_filter.py"


def record_error(patient_id: str, error_type: str, error_msg: str, file_logical: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not ERROR_CSV.exists() or ERROR_CSV.stat().st_size == 0
    with open(ERROR_CSV, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ERROR_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "patient_id": patient_id,
            "error_type": error_type,
            "error_msg": error_msg,
            "file_logical": file_logical,
        })


def record_runtime_rows(rows: list[dict]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RUNTIME_CSV.exists() or RUNTIME_CSV.stat().st_size == 0
    with open(RUNTIME_CSV, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUNTIME_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def stat_dict(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "p50": None, "p95": None, "max": None}
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "min": float(s.min()),
        "p50": float(s.quantile(0.50)),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max()),
    }


def is_upper(b) -> bool:
    return isinstance(b, str) and b.startswith("upper")


def top10_summary(top_df: pd.DataFrame, score_col: str) -> dict:
    if len(top_df) == 0:
        return {
            "selected": 0,
            "top10_patient_counts": "",
            "top10_position_bin_counts": "",
            "top10_score_min": None,
            "top10_score_max": None,
            "top10_upper_ratio": None,
        }
    top10 = top_df.head(10)
    pcnt = Counter(top10["patient_id"].astype(str))
    bcnt = Counter(top10["position_bin"].astype(str))
    upper = sum(1 for b in top10["position_bin"] if is_upper(b))
    return {
        "selected": int(len(top_df)),
        "top10_patient_counts": "; ".join(f"{c}x{p[:18]}" for p, c in pcnt.most_common()),
        "top10_position_bin_counts": "; ".join(f"{c}x{b}" for b, c in bcnt.most_common()),
        "top10_score_min": float(top10[score_col].min()),
        "top10_score_max": float(top10[score_col].max()),
        "top10_upper_ratio": round(upper / len(top10), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="central_distance_ratio_mean 필터 효과 분석")
    parser.add_argument("--model", default="padim_v1")
    parser.add_argument("--score-col", default="padim_score", dest="score_col")
    args = parser.parse_args()

    model = args.model
    score_col = args.score_col
    central_col = "central_distance_ratio_mean"

    score_dir = REPO_ROOT / "outputs" / "position-aware-padim-v1" / "scores" / model / "by_patient"
    candidates_dir = REPO_ROOT / "outputs" / "position-aware-padim-v1" / "candidates" / model

    out_dir = REPORTS_DIR / "candidate_analysis"
    out_csv = out_dir / "central_distance_filter_analysis.csv"
    out_json = out_dir / "central_distance_filter_summary.json"

    # 안전 가드
    if out_csv.exists() or out_json.exists():
        print(f"[ERROR] 기존 분석 출력 파일이 존재합니다:")
        if out_csv.exists(): print(f"  {out_csv}")
        if out_json.exists(): print(f"  {out_json}")
        sys.exit(1)

    if not score_dir.exists():
        print(f"[ERROR] score 디렉토리가 없습니다: {score_dir}")
        sys.exit(1)

    print(f"[analyze_cdr] model={model}, score_col={score_col}")

    start_time = time.time()

    # ----------------------------------------------------------------
    # 1. 합본 로드 + 검증
    # ----------------------------------------------------------------
    aggregator = ScoreAggregator(score_col=score_col)
    csvs = sorted(score_dir.glob("*.csv"))
    n_input = len(csvs)
    dfs: list[pd.DataFrame] = []
    n_total_nan = 0
    n_total_inf = 0
    n_total_rows = 0
    for p in csvs:
        df = aggregator.load_csv(str(p))
        col = df[score_col]
        n_total_nan += int(col.isna().sum())
        n_total_inf += int(np.isinf(col.to_numpy(dtype=float)).sum())
        n_total_rows += len(df)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    print(f"[analyze_cdr] 입력 CSV={n_input}, 합본 rows={n_total_rows}, NaN={n_total_nan}, inf={n_total_inf}")
    if n_total_nan > 0 or n_total_inf > 0:
        raise ValueError(f"NaN/inf 포함: NaN={n_total_nan}, inf={n_total_inf}")

    if central_col not in combined.columns:
        raise ValueError(f"합본에 {central_col} 컬럼이 없습니다.")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # 2. 기존 candidate CSV 3개 로드 (top50 분포 비교용)
    # ----------------------------------------------------------------
    cand_files = {
        "raw_top50": candidates_dir / "patch_topk.csv",
        "diverse_top50": candidates_dir / "patch_topk_diverse.csv",
        "bin_percentile_top50": candidates_dir / "patch_topk_bin_percentile_filtered.csv",
    }
    cand_dfs: dict[str, pd.DataFrame] = {}
    for k, p in cand_files.items():
        if p.exists():
            cand_dfs[k] = pd.read_csv(p, encoding="utf-8-sig")

    # ----------------------------------------------------------------
    # A. 분포 표 (전체 / top50 raw / top50 diverse / top50 bin_percentile / position_bin별)
    # ----------------------------------------------------------------
    print("[analyze_cdr] A. 분포 분석 ...")
    rows = []
    rows.append({"scope": "all", **stat_dict(combined[central_col])})
    for k, df in cand_dfs.items():
        if central_col in df.columns:
            rows.append({"scope": k, **stat_dict(df[central_col].head(50))})
    for bin_, grp in combined.groupby("position_bin"):
        rows.append({"scope": f"position_bin={bin_}", **stat_dict(grp[central_col])})

    # 기존 QA FP top10의 central_distance_ratio_mean (raw + diverse + bin_percentile 각 top10)
    qa_fp_rows = []
    for k, df in cand_dfs.items():
        top10 = df.head(10)
        for _, r in top10.iterrows():
            qa_fp_rows.append({
                "scope": f"{k}_top10",
                "rank": int(r["rank"]),
                "patient_id": str(r["patient_id"]),
                "local_z": int(r["local_z"]),
                "y0": int(r["y0"]), "x0": int(r["x0"]),
                "position_bin": str(r["position_bin"]),
                "central_distance_ratio_mean": float(r[central_col]) if central_col in r else None,
                "padim_score": float(r[score_col]) if score_col in r else None,
            })

    rows_df = pd.DataFrame(rows)
    rows_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # ----------------------------------------------------------------
    # B. 방향성 자동 판정
    # central_distance_ratio_mean이 작을수록 중심(폐문)인지 큰값이 중심인지 확인.
    # top50의 mean이 전체 mean보다 작으면 → 0이 중심에 가깝다는 의미 (FP 후보가 중심 인접).
    # 그 경우 `>= threshold` 필터로 중심 인접 후보 제거 가능.
    # ----------------------------------------------------------------
    all_mean = float(combined[central_col].mean())
    top50_raw_mean = (float(combined.sort_values(score_col, ascending=False).head(50)[central_col].mean())
                      if score_col in combined.columns else None)
    direction = "low_is_center" if (top50_raw_mean is not None and top50_raw_mean < all_mean) else "uncertain"
    filter_direction_ok = direction == "low_is_center"
    print(f"[analyze_cdr] 방향성: all_mean={all_mean:.4f}, top50_raw_mean={top50_raw_mean:.4f} → {direction}")

    # ----------------------------------------------------------------
    # C. threshold sweep (0.1~0.5) + tie-breaker variant 3가지
    # ----------------------------------------------------------------
    print("[analyze_cdr] C. threshold sweep ...")
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    sweep_records = []

    # 기존 FP top10 set (raw + diverse + bin_percentile 합집합)
    qa_fp_keys: set = set()
    for k, df in cand_dfs.items():
        for _, r in df.head(10).iterrows():
            qa_fp_keys.add((str(r["patient_id"]), int(r["local_z"]), int(r["y0"]), int(r["x0"])))

    # 합본에 ranking_score 계산용 사전 작업
    combined = combined.copy()
    combined["bin_percentile"] = combined.groupby("position_bin")[score_col].rank(pct=True)
    g_max = float(combined[score_col].max())
    if g_max == 0:
        g_max = 1.0
    combined["normalized_raw"] = combined[score_col] / g_max

    def compute_ranking(df: pd.DataFrame, mode: str,
                        tiny_weight: float = 1e-6,
                        central_weight: float = 1e-4) -> pd.Series:
        if mode == "percentile_only":
            return df["bin_percentile"]
        if mode == "percentile_raw_tiebreak":
            return df["bin_percentile"] + tiny_weight * df["normalized_raw"]
        if mode == "percentile_central_bonus":
            return (df["bin_percentile"]
                    + tiny_weight * df["normalized_raw"]
                    + central_weight * df[central_col])
        raise ValueError(mode)

    def top_k_with_caps(df: pd.DataFrame, rank_col: str, top_k: int,
                        max_per_patient: int = 2, max_per_position_bin: int = 10) -> pd.DataFrame:
        sorted_df = df.sort_values(
            by=[rank_col, "patient_id", "local_z", "y0", "x0"],
            ascending=[False, True, True, True, True], kind="stable",
        )
        selected = []
        p_count: dict = {}
        b_count: dict = {}
        for _, row in sorted_df.iterrows():
            pid = str(row["patient_id"])
            bin_ = str(row["position_bin"])
            if p_count.get(pid, 0) >= max_per_patient:
                continue
            if b_count.get(bin_, 0) >= max_per_position_bin:
                continue
            selected.append(row)
            p_count[pid] = p_count.get(pid, 0) + 1
            b_count[bin_] = b_count.get(bin_, 0) + 1
            if len(selected) >= top_k:
                break
        if not selected:
            return sorted_df.iloc[0:0].copy()
        return pd.DataFrame(selected).reset_index(drop=True)

    # slice_pure_lung_ratio >= 0.1 기본 적용
    base = combined[combined["slice_pure_lung_ratio"] >= 0.1].reset_index(drop=True)
    print(f"[analyze_cdr] base (slice_pure>=0.1) rows={len(base)}")

    for threshold in thresholds:
        if filter_direction_ok:
            filtered = base[base[central_col] >= threshold].reset_index(drop=True)
        else:
            filtered = base[base[central_col] <= threshold].reset_index(drop=True)

        n_after = int(len(filtered))
        if n_after == 0:
            sweep_records.append({
                "threshold": threshold,
                "tie_breaker": "n/a",
                "n_after_filter": 0,
                "n_selected": 0,
                "shortage": 1,
                "top10_patient_counts": "",
                "top10_position_bin_counts": "",
                "top10_score_min": None,
                "top10_score_max": None,
                "top10_upper_ratio": None,
                "removed_qa_fp_count": None,
            })
            continue

        for tie_mode in ["percentile_only", "percentile_raw_tiebreak", "percentile_central_bonus"]:
            tmp = filtered.copy()
            tmp["__ranking__"] = compute_ranking(tmp, tie_mode)
            top_df = top_k_with_caps(tmp, "__ranking__", top_k=50)
            s = top10_summary(top_df, score_col)

            # QA FP 중 top10에서 제거된 개수 = QA FP set 중 새 top10에 안 들어간 후보 (단 그 환자/슬라이스/좌표가 base에 존재해야 의미)
            new_top10_keys = set(
                (str(r["patient_id"]), int(r["local_z"]), int(r["y0"]), int(r["x0"]))
                for _, r in top_df.head(10).iterrows()
            )
            removed_qa_fp_count = int(len(qa_fp_keys - new_top10_keys))

            sweep_records.append({
                "threshold": threshold,
                "tie_breaker": tie_mode,
                "n_after_filter": n_after,
                "n_selected": s["selected"],
                "shortage": int(s["selected"] < 50),
                "top10_patient_counts": s["top10_patient_counts"],
                "top10_position_bin_counts": s["top10_position_bin_counts"],
                "top10_score_min": s["top10_score_min"],
                "top10_score_max": s["top10_score_max"],
                "top10_upper_ratio": s["top10_upper_ratio"],
                "removed_qa_fp_count": removed_qa_fp_count,
            })

    sweep_df = pd.DataFrame(sweep_records)

    # ----------------------------------------------------------------
    # D. 가장 안전한 threshold + tie-breaker 자동 추천
    # 기준: shortage=0, removed_qa_fp_count >= 2, upper_ratio 가장 낮은 것 → 동률이면 threshold 가장 낮은 것
    # ----------------------------------------------------------------
    candidates_safe = sweep_df[(sweep_df["shortage"] == 0)
                               & (sweep_df["removed_qa_fp_count"] >= 2)].copy()
    if len(candidates_safe) > 0:
        # upper_ratio 오름차순 → threshold 오름차순 → removed_qa_fp_count 내림차순
        candidates_safe = candidates_safe.sort_values(
            by=["top10_upper_ratio", "threshold", "removed_qa_fp_count"],
            ascending=[True, True, False], kind="stable",
        )
        rec = candidates_safe.iloc[0]
        recommended = {
            "threshold": float(rec["threshold"]),
            "tie_breaker": str(rec["tie_breaker"]),
            "rationale": "shortage=0, removed_qa_fp_count>=2 중 upper_ratio 최소, threshold 최저",
            "top10_upper_ratio": rec["top10_upper_ratio"],
            "removed_qa_fp_count": int(rec["removed_qa_fp_count"]),
        }
    else:
        recommended = None

    # ----------------------------------------------------------------
    # summary JSON
    # ----------------------------------------------------------------
    elapsed = time.time() - start_time
    summary = {
        "script": SCRIPT_NAME,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "score_col": score_col,
        "central_col": central_col,
        "input_csv_count": n_input,
        "input_total_rows": n_total_rows,
        "n_nan_total": n_total_nan,
        "n_inf_total": n_total_inf,
        "all_mean": all_mean,
        "top50_raw_mean": top50_raw_mean,
        "direction_judgment": direction,
        "filter_direction_ok": filter_direction_ok,
        "qa_fp_top10_records": qa_fp_rows,
        "sweep_records": sweep_records,
        "recommended": recommended,
        "elapsed_seconds": round(elapsed, 2),
        "output_csv": str(out_csv),
        "output_json": str(out_json),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # runtime_summary 4컬럼 기록
    ts = datetime.now().isoformat(timespec="seconds")
    record_runtime_rows([
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_input_csvs", "value": n_input},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_total_rows_input", "value": n_total_rows},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_sweep_records", "value": len(sweep_records)},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "direction_judgment", "value": direction},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "elapsed_seconds", "value": round(elapsed, 2)},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "output_csv", "value": str(out_csv)},
    ])

    print(f"[analyze_cdr] 완료 ({elapsed:.1f}s)")
    print(f"  CSV : {out_csv}")
    print(f"  JSON: {out_json}")
    if recommended:
        print(f"  RECOMMENDED: threshold={recommended['threshold']}, tie_breaker={recommended['tie_breaker']}, "
              f"upper_ratio={recommended['top10_upper_ratio']}, removed_fp={recommended['removed_qa_fp_count']}")
    else:
        print("  RECOMMENDED: 없음 — 기존 마스크 단계 개입 필요")


if __name__ == "__main__":
    main()
