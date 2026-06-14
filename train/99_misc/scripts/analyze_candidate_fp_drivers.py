"""
analyze_candidate_fp_drivers.py: PaDiM top 후보 false positive 원인 진단.

학습/스코어링/시각화 재실행 없이 기존 score CSV만 사용한 read-only 분석.

입력:
- outputs/.../scores/{model}/by_patient/*.csv (72개)
- outputs/.../candidates/{model}/patch_topk.csv
- outputs/.../candidates/{model}/patch_topk_diverse.csv

출력 (outputs/.../reports/candidate_analysis/):
1. position_bin_score_summary.csv     - position_bin별 score 분포 + top50/top200 점유
2. top_candidates_feature_summary.csv - top 후보의 patch 품질 컬럼 요약
3. candidate_filter_sweep.csv         - 독립 sweep + 최종 추천 조합
4. candidate_ranking_variants.csv     - 3가지 normalization variant의 top50 효과
5. candidate_analysis_summary.json    - 메타 + rank8 마스크 의심 후보 데이터 + 다음 단계 추천

금지:
- 기존 score CSV / candidate CSV / visualization 폴더 수정
- 학습 / 스코어링 / 시각화 재실행
- 병변 데이터 접근
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

SCRIPT_NAME = "analyze_candidate_fp_drivers.py"


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


# ------------------------------------------------------------------
# 분석 헬퍼
# ------------------------------------------------------------------

def is_upper_bin(b: str) -> bool:
    return isinstance(b, str) and b.startswith("upper")


def apply_filter(
    df: pd.DataFrame,
    pure_lung_min: float | None = None,
    organ_max: float | None = None,
    slice_pure_min: float | None = None,
) -> pd.DataFrame:
    out = df
    if pure_lung_min is not None and "pure_lung_patch_ratio" in out.columns:
        out = out[out["pure_lung_patch_ratio"] >= pure_lung_min]
    if organ_max is not None and "organ_patch_ratio" in out.columns:
        out = out[out["organ_patch_ratio"] <= organ_max]
    if slice_pure_min is not None and "slice_pure_lung_ratio" in out.columns:
        out = out[out["slice_pure_lung_ratio"] >= slice_pure_min]
    return out


def top_k_with_caps(
    df: pd.DataFrame,
    score_col: str,
    top_k: int,
    max_per_patient: int | None = None,
    max_per_position_bin: int | None = None,
) -> pd.DataFrame:
    sorted_df = df.sort_values(
        by=[score_col, "patient_id", "local_z", "y0", "x0"],
        ascending=[False, True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    if max_per_patient is None and max_per_position_bin is None:
        return sorted_df.head(top_k).reset_index(drop=True)

    selected = []
    p_count: dict = {}
    b_count: dict = {}
    for _, row in sorted_df.iterrows():
        pid = str(row["patient_id"])
        bin_ = str(row["position_bin"])
        if max_per_patient is not None and p_count.get(pid, 0) >= max_per_patient:
            continue
        if max_per_position_bin is not None and b_count.get(bin_, 0) >= max_per_position_bin:
            continue
        selected.append(row)
        p_count[pid] = p_count.get(pid, 0) + 1
        b_count[bin_] = b_count.get(bin_, 0) + 1
        if len(selected) >= top_k:
            break
    if not selected:
        return sorted_df.iloc[0:0].copy()
    return pd.DataFrame(selected).reset_index(drop=True)


def summarize_top10(top_df: pd.DataFrame, score_col: str) -> dict:
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
    upper = sum(1 for b in top10["position_bin"] if is_upper_bin(b))
    return {
        "selected": int(len(top_df)),
        "top10_patient_counts": "; ".join(f"{c}x{p[:20]}" for p, c in pcnt.most_common()),
        "top10_position_bin_counts": "; ".join(f"{c}x{b}" for b, c in bcnt.most_common()),
        "top10_score_min": float(top10[score_col].min()),
        "top10_score_max": float(top10[score_col].max()),
        "top10_upper_ratio": round(upper / len(top10), 3),
    }


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PaDiM top 후보 FP 원인 진단 (read-only)")
    parser.add_argument("--model", default="padim_v1")
    parser.add_argument("--score-col", default="padim_score", dest="score_col")
    args = parser.parse_args()

    model = args.model
    score_col = args.score_col

    score_dir = (
        REPO_ROOT / "outputs" / "position-aware-padim-v1"
        / "scores" / model / "by_patient"
    )
    candidates_dir = (
        REPO_ROOT / "outputs" / "position-aware-padim-v1"
        / "candidates" / model
    )
    patch_topk_path = candidates_dir / "patch_topk.csv"
    patch_topk_diverse_path = candidates_dir / "patch_topk_diverse.csv"

    out_dir = REPORTS_DIR / "candidate_analysis"
    out_files = {
        "position_bin": out_dir / "position_bin_score_summary.csv",
        "feature": out_dir / "top_candidates_feature_summary.csv",
        "sweep": out_dir / "candidate_filter_sweep.csv",
        "variants": out_dir / "candidate_ranking_variants.csv",
        "summary": out_dir / "candidate_analysis_summary.json",
    }

    start_time = time.time()

    # ----------------------------------------------------------------
    # 안전 가드: 출력 파일 중복 시 중단
    # ----------------------------------------------------------------
    existing = [str(p) for p in out_files.values() if p.exists()]
    if existing:
        print(
            "[ERROR] 기존 분석 출력 파일이 존재합니다. archive 또는 삭제 후 다시 실행하세요:"
        )
        for p in existing:
            print(f"  {p}")
        sys.exit(1)

    if not score_dir.exists():
        print(f"[ERROR] score 디렉토리가 없습니다: {score_dir}")
        sys.exit(1)
    if not patch_topk_path.exists():
        print(f"[ERROR] patch_topk.csv가 없습니다: {patch_topk_path}")
        sys.exit(1)
    if not patch_topk_diverse_path.exists():
        print(f"[ERROR] patch_topk_diverse.csv가 없습니다: {patch_topk_diverse_path}")
        sys.exit(1)

    print(f"[analyze] model={model}, score_col={score_col}")
    print(f"[analyze] score_dir={score_dir}")
    print(f"[analyze] out_dir={out_dir}")
    print()

    # ----------------------------------------------------------------
    # 1. 72개 score CSV 로드 + concat (ScoreAggregator가 NaN/inf 검증)
    # ----------------------------------------------------------------
    aggregator = ScoreAggregator(score_col=score_col)
    csvs = sorted(score_dir.glob("*.csv"))
    n_input = len(csvs)
    dfs: list[pd.DataFrame] = []
    n_total_nan = 0
    n_total_inf = 0
    n_total_rows = 0
    for p in csvs:
        try:
            df = aggregator.load_csv(str(p))
        except Exception as exc:
            record_error(p.stem, "load_or_validate_error", str(exc), str(p))
            print(f"  [FAIL] {p.stem}: {exc}")
            raise
        col = df[score_col]
        n_total_nan += int(col.isna().sum())
        n_total_inf += int(np.isinf(col.to_numpy(dtype=float)).sum())
        n_total_rows += len(df)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)

    print(f"[analyze] 입력 CSV={n_input}, 합본 rows={n_total_rows}, NaN={n_total_nan}, inf={n_total_inf}")
    if n_total_nan > 0 or n_total_inf > 0:
        raise ValueError(f"NaN/inf 포함: NaN={n_total_nan}, inf={n_total_inf}")
    if len(combined) != n_total_rows:
        raise ValueError(f"concat 후 rows 불일치: {len(combined)} vs {n_total_rows}")

    # 출력 폴더 생성 (검증 모두 통과 후)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # A. position_bin별 score 분포
    # ----------------------------------------------------------------
    print("[analyze] A. position_bin별 score 분포 ...")
    top50_global = combined.sort_values(score_col, ascending=False, kind="stable").head(50)
    top200_global = combined.sort_values(score_col, ascending=False, kind="stable").head(200)
    top50_bin_cnt = Counter(top50_global["position_bin"].astype(str))
    top200_bin_cnt = Counter(top200_global["position_bin"].astype(str))

    rows = []
    for bin_, grp in combined.groupby("position_bin"):
        s = grp[score_col]
        rows.append({
            "position_bin": str(bin_),
            "count": int(len(grp)),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            "p50": float(s.quantile(0.50)),
            "p95": float(s.quantile(0.95)),
            "p99": float(s.quantile(0.99)),
            "max": float(s.max()),
            "in_top50": int(top50_bin_cnt.get(str(bin_), 0)),
            "in_top200": int(top200_bin_cnt.get(str(bin_), 0)),
        })
    bin_summary = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
    bin_summary.to_csv(out_files["position_bin"], index=False, encoding="utf-8-sig")

    # ----------------------------------------------------------------
    # B. top 후보의 patch 품질 컬럼 분석 (top50 / top200 + 전체 비교)
    # ----------------------------------------------------------------
    print("[analyze] B. top 후보 patch 품질 컬럼 분석 ...")
    quality_cols = [
        "pure_lung_patch_ratio",
        "organ_patch_ratio",
        "slice_pure_lung_ratio",
        "central_distance_ratio_mean",
        "z_ratio",
        "local_z",
    ]
    quality_cols_present = [c for c in quality_cols if c in combined.columns]

    def col_summary(df: pd.DataFrame, scope: str) -> list[dict]:
        rows = []
        for c in quality_cols_present:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s) == 0:
                continue
            rows.append({
                "scope": scope,
                "column": c,
                "n": int(len(s)),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                "min": float(s.min()),
                "p50": float(s.quantile(0.50)),
                "p95": float(s.quantile(0.95)),
                "max": float(s.max()),
            })
        # 범주형 central_peripheral
        if "central_peripheral" in df.columns:
            vc = df["central_peripheral"].astype(str).value_counts()
            for v, c in vc.items():
                rows.append({
                    "scope": scope,
                    "column": f"central_peripheral={v}",
                    "n": int(c),
                    "mean": None, "std": None, "min": None,
                    "p50": None, "p95": None, "max": None,
                })
        # position_bin 분포
        if "position_bin" in df.columns:
            vc = df["position_bin"].astype(str).value_counts()
            for v, c in vc.items():
                rows.append({
                    "scope": scope,
                    "column": f"position_bin={v}",
                    "n": int(c),
                    "mean": None, "std": None, "min": None,
                    "p50": None, "p95": None, "max": None,
                })
        return rows

    feat_rows: list[dict] = []
    feat_rows += col_summary(combined, "all")
    feat_rows += col_summary(top200_global, "top200")
    feat_rows += col_summary(top50_global, "top50")
    feat_df = pd.DataFrame(feat_rows)
    feat_df.to_csv(out_files["feature"], index=False, encoding="utf-8-sig")

    # ----------------------------------------------------------------
    # C. filter sweep — 독립 sweep + 최종 추천 조합 1개
    # ----------------------------------------------------------------
    print("[analyze] C. filter sweep ...")
    sweep_rows: list[dict] = []

    def add_sweep(label: str, **filters_and_caps):
        pure = filters_and_caps.get("pure_lung_min")
        organ = filters_and_caps.get("organ_max")
        slice_pure = filters_and_caps.get("slice_pure_min")
        max_pp = filters_and_caps.get("max_per_patient")
        max_pb = filters_and_caps.get("max_per_position_bin")
        filtered = apply_filter(
            combined, pure_lung_min=pure, organ_max=organ, slice_pure_min=slice_pure
        )
        n_after_filter = int(len(filtered))
        top_df = top_k_with_caps(
            filtered, score_col=score_col, top_k=50,
            max_per_patient=max_pp, max_per_position_bin=max_pb,
        )
        s = summarize_top10(top_df, score_col)
        sweep_rows.append({
            "label": label,
            "pure_lung_patch_ratio_min": pure,
            "organ_patch_ratio_max": organ,
            "slice_pure_lung_ratio_min": slice_pure,
            "max_per_patient": max_pp,
            "max_per_position_bin": max_pb,
            "n_after_filter": n_after_filter,
            "n_selected": s["selected"],
            "shortage": int(s["selected"] < 50),
            "top10_patient_counts": s["top10_patient_counts"],
            "top10_position_bin_counts": s["top10_position_bin_counts"],
            "top10_score_min": s["top10_score_min"],
            "top10_score_max": s["top10_score_max"],
            "top10_upper_ratio": s["top10_upper_ratio"],
        })

    # 독립 sweep (14행): 각 차원별 단독 변형
    for v in [0.5, 0.7, 0.9]:
        add_sweep(f"pure_lung_min={v}", pure_lung_min=v)
    for v in [0.0, 0.01, 0.05]:
        add_sweep(f"organ_max={v}", organ_max=v)
    for v in [0.05, 0.1, 0.2]:
        add_sweep(f"slice_pure_min={v}", slice_pure_min=v)
    for v in [5, 10, 20]:
        add_sweep(f"max_per_position_bin={v}", max_per_position_bin=v)
    for v in [1, 2]:
        add_sweep(f"max_per_patient={v}", max_per_patient=v)

    # 최종 추천 조합 1개 (휴리스틱: 가장 안전한 보수적 디폴트)
    rec_filters = dict(
        pure_lung_min=0.7,
        organ_max=0.01,
        slice_pure_min=0.1,
        max_per_position_bin=5,
        max_per_patient=2,
    )
    add_sweep(
        "RECOMMENDED_COMBO("
        "pure_lung>=0.7, organ<=0.01, slice_pure>=0.1, "
        "max_per_bin=5, max_per_patient=2)",
        **rec_filters,
    )

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(out_files["sweep"], index=False, encoding="utf-8-sig")

    # ----------------------------------------------------------------
    # D. score normalization 실험
    # ----------------------------------------------------------------
    print("[analyze] D. score normalization 실험 ...")
    # bin_z_score
    bin_mean = combined.groupby("position_bin")[score_col].transform("mean")
    bin_std = combined.groupby("position_bin")[score_col].transform("std")
    combined["bin_z_score"] = (combined[score_col] - bin_mean) / bin_std.replace(0, np.nan)
    combined["bin_z_score"] = combined["bin_z_score"].fillna(0.0)

    # bin_percentile_score
    combined["bin_percentile_score"] = combined.groupby("position_bin")[score_col].rank(pct=True)

    # patient_z_score
    pat_mean = combined.groupby("patient_id")[score_col].transform("mean")
    pat_std = combined.groupby("patient_id")[score_col].transform("std")
    combined["patient_z_score"] = (combined[score_col] - pat_mean) / pat_std.replace(0, np.nan)
    combined["patient_z_score"] = combined["patient_z_score"].fillna(0.0)

    # 원본 top10 (비교용)
    orig_top10 = combined.sort_values(score_col, ascending=False, kind="stable").head(10)
    orig_keys = set(
        (str(r["patient_id"]), int(r["local_z"]), int(r["y0"]), int(r["x0"]))
        for _, r in orig_top10.iterrows()
    )

    variants_rows = []
    for variant in ["padim_score", "bin_z_score", "bin_percentile_score", "patient_z_score"]:
        top50 = combined.sort_values(variant, ascending=False, kind="stable").head(50)
        s = summarize_top10(top50, variant)
        var_top10 = top50.head(10)
        var_keys = set(
            (str(r["patient_id"]), int(r["local_z"]), int(r["y0"]), int(r["x0"]))
            for _, r in var_top10.iterrows()
        )
        overlap_with_original = len(orig_keys & var_keys)
        variants_rows.append({
            "variant": variant,
            "top10_patient_counts": s["top10_patient_counts"],
            "top10_position_bin_counts": s["top10_position_bin_counts"],
            "top10_score_min": s["top10_score_min"],
            "top10_score_max": s["top10_score_max"],
            "top10_upper_ratio": s["top10_upper_ratio"],
            "overlap_with_padim_score_top10": overlap_with_original,
        })
    variants_df = pd.DataFrame(variants_rows)
    variants_df.to_csv(out_files["variants"], index=False, encoding="utf-8-sig")

    # ----------------------------------------------------------------
    # E. rank8 마스크 의심 후보 별도 확인
    # ----------------------------------------------------------------
    print("[analyze] E. rank8 마스크 의심 후보 확인 ...")
    diverse_df = pd.read_csv(patch_topk_diverse_path, encoding="utf-8-sig")
    rank8_row = diverse_df[diverse_df["rank"] == 8]
    if len(rank8_row) == 0:
        rank8_data = {"warning": "patch_topk_diverse.csv에 rank=8 행이 없습니다."}
    else:
        r = rank8_row.iloc[0]
        keys = [
            "patient_id", "local_z", "y0", "x0", "y1", "x1",
            "pure_lung_patch_ratio", "organ_patch_ratio", "slice_pure_lung_ratio",
            "position_bin", "z_ratio", score_col,
        ]
        rank8_data = {k: (None if k not in r else (
            float(r[k]) if isinstance(r[k], (int, float, np.integer, np.floating)) and k not in ("patient_id", "position_bin") else
            (int(r[k]) if k in ("local_z", "y0", "x0", "y1", "x1") else str(r[k]))
        )) for k in keys}

    # ----------------------------------------------------------------
    # candidate_analysis_summary.json
    # ----------------------------------------------------------------
    print("[analyze] summary JSON 작성 ...")
    elapsed = time.time() - start_time

    # 추천 다음 단계 (휴리스틱 텍스트)
    rec_combo_row = sweep_rows[-1]
    next_steps = []
    if rec_combo_row["top10_upper_ratio"] is not None and rec_combo_row["top10_upper_ratio"] >= 0.9:
        next_steps.append(
            "필터 적용 후에도 upper 비율이 90% 이상이면 score normalization "
            "(특히 bin_z_score 또는 bin_percentile_score) 적용을 고려"
        )
    next_steps.append(
        "rank_candidates.py에 --max-per-position-bin 옵션 추가 검토"
    )
    next_steps.append(
        "pure_lung_patch_ratio / slice_pure_lung_ratio 임계 필터를 rank_candidates 단계로 통합 검토"
    )

    summary = {
        "script": SCRIPT_NAME,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "score_col": score_col,
        "input_csv_count": n_input,
        "input_total_rows": n_total_rows,
        "n_nan_total": n_total_nan,
        "n_inf_total": n_total_inf,
        "elapsed_seconds": round(elapsed, 2),
        "output_files": {k: str(v) for k, v in out_files.items()},
        "rank8_mask_suspect_candidate": rank8_data,
        "recommended_filter_combo": rec_filters,
        "recommended_combo_summary": {
            "n_selected": rec_combo_row["n_selected"],
            "top10_upper_ratio": rec_combo_row["top10_upper_ratio"],
            "top10_position_bin_counts": rec_combo_row["top10_position_bin_counts"],
            "top10_patient_counts": rec_combo_row["top10_patient_counts"],
        },
        "next_step_recommendations": next_steps,
    }
    with open(out_files["summary"], "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print()
    print(f"[analyze] 완료 ({elapsed:.1f}s). 생성 파일:")
    for k, p in out_files.items():
        print(f"  {k}: {p}")

    # ----------------------------------------------------------------
    # runtime_summary.csv 4컬럼 기록
    # ----------------------------------------------------------------
    ts = datetime.now().isoformat(timespec="seconds")
    rows = [
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_input_csvs", "value": n_input},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_total_rows_input", "value": n_total_rows},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_output_files", "value": len(out_files)},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "elapsed_seconds", "value": round(elapsed, 2)},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "model", "value": model},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "score_col", "value": score_col},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "output_dir", "value": str(out_dir)},
    ]
    record_runtime_rows(rows)
    print(f"[analyze] runtime_summary.csv 기록 완료: {RUNTIME_CSV}")


if __name__ == "__main__":
    main()
