"""
summarize_normal_scores.py: v2 normal score 기반 threshold/summary 산출 스크립트.

- val split 36명 score만 사용하여 p95/p99 threshold 산출
- test split 36명은 threshold 산출에 사용 금지, 검증 요약용으로만 사용
- 출력 경로는 v2 전용으로 분리 (기존 v1 결과 수정 금지)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_SPLIT_JSON = (
    REPO_ROOT / "outputs" / "position-aware-padim-v1" / "splits" / "normal_v1.json"
)
DEFAULT_SCORE_DIR = (
    REPO_ROOT
    / "outputs"
    / "position-aware-padim-v1"
    / "scores"
    / "padim_v2_roi0_0"
    / "normal_by_patient"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "position-aware-padim-v1"
    / "evaluation"
    / "normal_v2_roi0_0"
)


def load_split(split_json: Path) -> tuple[list[str], list[str]]:
    with open(split_json, encoding="utf-8") as f:
        data = json.load(f)
    return data["val"], data["test"]


def load_scores(score_dir: Path, patient_ids: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """patient_ids에 해당하는 score CSV를 로드하여 합친다. 없는 환자는 missing 목록에 추가."""
    dfs = []
    missing = []
    for pid in patient_ids:
        p = score_dir / f"{pid}.csv"
        if not p.exists():
            missing.append(pid)
            continue
        df = pd.read_csv(p, encoding="utf-8-sig")
        if "padim_score" not in df.columns:
            print(f"[ERROR] padim_score 컬럼 없음: {p}")
            sys.exit(1)
        df["patient_id"] = pid
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return combined, missing


def compute_stats(scores: np.ndarray, label: str) -> dict:
    return {
        f"{label}_n_patches": int(len(scores)),
        f"{label}_nan": int(np.isnan(scores).sum()),
        f"{label}_inf": int(np.isinf(scores).sum()),
        f"{label}_min": float(np.nanmin(scores)),
        f"{label}_max": float(np.nanmax(scores)),
        f"{label}_mean": float(np.nanmean(scores)),
        f"{label}_median": float(np.nanmedian(scores)),
        f"{label}_p95": float(np.nanpercentile(scores, 95)),
        f"{label}_p99": float(np.nanpercentile(scores, 99)),
    }


def per_patient_summary(df: pd.DataFrame, split_label: str) -> pd.DataFrame:
    rows = []
    for pid, grp in df.groupby("patient_id"):
        scores = grp["padim_score"].values
        rows.append({
            "patient_id": pid,
            "split": split_label,
            "n_patches": len(scores),
            "nan": int(np.isnan(scores).sum()),
            "inf": int(np.isinf(scores).sum()),
            "min": float(np.nanmin(scores)),
            "max": float(np.nanmax(scores)),
            "mean": float(np.nanmean(scores)),
            "median": float(np.nanmedian(scores)),
            "p95": float(np.nanpercentile(scores, 95)),
            "p99": float(np.nanpercentile(scores, 99)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="v2 normal score threshold/summary 산출")
    parser.add_argument(
        "--score-dir",
        type=str,
        default=str(DEFAULT_SCORE_DIR),
        help="v2 normal score CSV 폴더",
    )
    parser.add_argument(
        "--split-json",
        type=str,
        default=str(DEFAULT_SPLIT_JSON),
        help="val/test split JSON (normal_v1.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="v2 전용 출력 폴더",
    )
    args = parser.parse_args()

    score_dir = Path(args.score_dir)
    split_json = Path(args.split_json)
    output_dir = Path(args.output_dir)

    # ----------------------------------------------------------------
    # 입력 경로 확인
    # ----------------------------------------------------------------
    if not score_dir.exists():
        print(f"[ERROR] score-dir 없음: {score_dir}")
        sys.exit(1)
    if not split_json.exists():
        print(f"[ERROR] split-json 없음: {split_json}")
        sys.exit(1)

    val_ids, test_ids = load_split(split_json)
    print(f"[summarize] val: {len(val_ids)}명, test: {len(test_ids)}명")

    # ----------------------------------------------------------------
    # score CSV 로드
    # ----------------------------------------------------------------
    val_df, val_missing = load_scores(score_dir, val_ids)
    test_df, test_missing = load_scores(score_dir, test_ids)

    if val_missing:
        print(f"[ERROR] val 누락 환자 {len(val_missing)}명 — threshold 산출 중단: {val_missing}")
        sys.exit(1)
    if test_missing:
        print(f"[ERROR] test 누락 환자 {len(test_missing)}명 — summary 산출 중단: {test_missing}")
        sys.exit(1)

    if val_df.empty:
        print("[ERROR] val score 없음 — threshold 산출 불가")
        sys.exit(1)

    val_scores = val_df["padim_score"].values
    test_scores = test_df["padim_score"].values if not test_df.empty else np.array([])

    # ----------------------------------------------------------------
    # NaN / Inf guard
    # ----------------------------------------------------------------
    n_val_nan = int(np.isnan(val_scores).sum())
    n_val_inf = int(np.isinf(val_scores).sum())
    if n_val_nan > 0 or n_val_inf > 0:
        print(f"[ERROR] val padim_score NaN={n_val_nan}, Inf={n_val_inf} — threshold 산출 중단")
        sys.exit(1)
    if len(test_scores) > 0:
        n_test_nan = int(np.isnan(test_scores).sum())
        n_test_inf = int(np.isinf(test_scores).sum())
        if n_test_nan > 0 or n_test_inf > 0:
            print(f"[ERROR] test padim_score NaN={n_test_nan}, Inf={n_test_inf} — summary 산출 중단")
            sys.exit(1)

    # ----------------------------------------------------------------
    # threshold 산출 (val 전용)
    # ----------------------------------------------------------------
    threshold_p95 = float(np.nanpercentile(val_scores, 95))
    threshold_p99 = float(np.nanpercentile(val_scores, 99))
    print(f"[summarize] val p95 threshold: {threshold_p95:.6f}")
    print(f"[summarize] val p99 threshold: {threshold_p99:.6f}")

    # ----------------------------------------------------------------
    # 통계 계산
    # ----------------------------------------------------------------
    val_stats = compute_stats(val_scores, "val")
    test_stats = compute_stats(test_scores, "test") if len(test_scores) > 0 else {}

    # test threshold 초과 patch 비율
    if len(test_scores) > 0:
        test_exceed_p95 = float((test_scores > threshold_p95).mean())
        test_exceed_p99 = float((test_scores > threshold_p99).mean())
    else:
        test_exceed_p95 = float("nan")
        test_exceed_p99 = float("nan")

    # ----------------------------------------------------------------
    # 출력 저장
    # ----------------------------------------------------------------
    threshold_path = output_dir / "normal_v2_threshold.json"
    per_patient_path = output_dir / "per_patient_score_summary.csv"
    existing = [p for p in [threshold_path, per_patient_path] if p.exists()]
    if existing:
        print("[ERROR] 출력 파일이 이미 존재합니다 — 덮어쓰기 방지:")
        for p in existing:
            print(f"  {p}")
        print("삭제 후 다시 실행하거나 --output-dir를 변경하세요.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # threshold JSON
    threshold_data = {
        "threshold_source": "val_split",
        "n_val_patients": len(val_ids) - len(val_missing),
        "n_val_patches": int(len(val_scores)),
        "threshold_p95": threshold_p95,
        "threshold_p99": threshold_p99,
        "val_stats": val_stats,
        "test_stats": test_stats,
        "test_exceed_p95_ratio": test_exceed_p95,
        "test_exceed_p99_ratio": test_exceed_p99,
        "score_dir": str(score_dir),
        "split_json": str(split_json),
        "val_missing": val_missing,
        "test_missing": test_missing,
    }
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump(threshold_data, f, indent=2, ensure_ascii=False)
    print(f"[summarize] threshold JSON 저장: {threshold_path}")

    # 환자별 summary CSV
    per_patient_rows = []
    if not val_df.empty:
        per_patient_rows.append(per_patient_summary(val_df, "val"))
    if not test_df.empty:
        per_patient_rows.append(per_patient_summary(test_df, "test"))

    if per_patient_rows:
        per_patient_df = pd.concat(per_patient_rows, ignore_index=True)
        per_patient_df.to_csv(per_patient_path, index=False, encoding="utf-8-sig")
        print(f"[summarize] 환자별 summary CSV 저장: {per_patient_path}")

    print()
    print("[summarize] 완료")
    print(f"  val p95 threshold : {threshold_p95:.6f}")
    print(f"  val p99 threshold : {threshold_p99:.6f}")
    print(f"  test exceed p95   : {test_exceed_p95:.4%}")
    print(f"  test exceed p99   : {test_exceed_p99:.4%}")
    print(f"  출력 경로          : {output_dir}")


if __name__ == "__main__":
    main()
