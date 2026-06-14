"""
rank_candidates.py: PaDiM patch-level top-k 후보 CSV 생성 스크립트.

- outputs/.../scores/{model}/by_patient/*.csv 전체를 읽어 합본을 만든다.
- ScoreAggregator.load_csv로 환자별 NaN/inf 가드 + 필수 컬럼 검증을 수행한다.
- CandidateRanker(score_col).rank_patches(df, top_k=N)로 patch-level 후보를 정렬한다.
- 결과 CSV에 rank, model, score_col 컬럼을 추가하고 원본 patch 컬럼 전체를 보존한다.
- 별도 metadata json도 함께 저장한다.
- 기존 patch_topk.csv가 있으면 안전을 위해 중단한다 (사용자가 archive/삭제 처리).
- 실행 정보를 runtime_summary.csv(4컬럼)와 error.csv(4컬럼)에 기록한다.

금지:
- slice-level / patient-level 후보 정렬
- 이미지 저장 / 시각화 생성
- 기존 score CSV 수정 / 삭제
- 학습 또는 스코어링 재실행
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# 프로젝트 루트를 sys.path에 추가 (src 하위 패키지 import용)
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from position_aware_padim.candidate_ranker import CandidateRanker
from position_aware_padim.score_aggregator import ScoreAggregator


REPORTS_DIR = REPO_ROOT / "outputs" / "position-aware-padim-v1" / "reports"
ERROR_CSV = REPORTS_DIR / "error.csv"
RUNTIME_CSV = REPORTS_DIR / "runtime_summary.csv"

ERROR_COLUMNS = ["patient_id", "error_type", "error_msg", "file_logical"]
RUNTIME_COLUMNS = ["timestamp", "script", "metric", "value"]
RUNTIME_SCHEMA_HEADER = "timestamp,script,metric,value"

SCRIPT_NAME = "rank_candidates.py"


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


def check_and_archive_runtime_csv() -> None:
    """runtime_summary.csv 헤더가 공통 4컬럼 스키마와 다르면 archive로 백업 후 새로 생성한다."""
    if not RUNTIME_CSV.exists():
        return
    if RUNTIME_CSV.stat().st_size == 0:
        return

    with open(RUNTIME_CSV, encoding="utf-8-sig", newline="") as f:
        first_line = f.readline().rstrip("\r\n")

    first_line_clean = first_line.lstrip("﻿")
    if first_line_clean == RUNTIME_SCHEMA_HEADER:
        return

    archive_dir = REPORTS_DIR / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"runtime_summary_{ts}.csv"
    shutil.move(str(RUNTIME_CSV), str(archive_path))
    print(f"[rank_candidates] 기존 runtime_summary.csv 스키마 불일치 → 백업: {archive_path}")


def record_runtime_rows(rows: list[dict]) -> None:
    """4컬럼 공통 스키마(timestamp,script,metric,value)로 여러 행을 기록한다."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RUNTIME_CSV.exists() or RUNTIME_CSV.stat().st_size == 0
    with open(RUNTIME_CSV, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUNTIME_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def select_diverse_candidates(
    combined: pd.DataFrame,
    score_col: str,
    top_k: int,
    max_per_patient: int,
    z_nms_window: int,
    coord_nms_distance: float,
    ranking_col: str | None = None,
    max_per_position_bin: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """patch_score 내림차순에서 환자 제한 + (z, 좌표) NMS로 다양성 제약 후보를 선택한다.

    선택 규칙:
    - ranking_col이 주어지면 그 컬럼 기준 내림차순 정렬, 없으면 score_col 사용.
    - 같은 patient_id에서 이미 max_per_patient 개 선택됐으면 skip.
    - max_per_position_bin이 주어지면 position_bin별 후보 수도 제한.
    - 같은 patient_id에서 |Δz| <= z_nms_window AND
      patch 중심 거리 <= coord_nms_distance 이면 skip.
    - top_k 개 선택될 때까지 반복.
    - 부족하면 부분 결과 반환 (호출부에서 경고 처리).

    Returns
    -------
    (diverse_df, stats)
    """
    sort_col = ranking_col or score_col
    sorted_df = combined.sort_values(
        by=[sort_col, "patient_id", "local_z", "y0", "x0"],
        ascending=[False, True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)

    selected_rows: list = []
    per_patient_count: dict = {}
    per_patient_picks: dict = {}  # {pid: [(local_z, cy, cx), ...]}
    per_bin_count: dict = {}
    n_scanned = 0
    n_skipped_patient_limit = 0
    n_skipped_z_coord_nms = 0
    n_skipped_position_bin_limit = 0

    for _, row in sorted_df.iterrows():
        n_scanned += 1
        pid = str(row["patient_id"])
        bin_ = str(row["position_bin"])
        z = int(row["local_z"])
        cy = (int(row["y0"]) + int(row["y1"])) / 2.0
        cx = (int(row["x0"]) + int(row["x1"])) / 2.0

        # 환자별 max 후보 제한
        if per_patient_count.get(pid, 0) >= max_per_patient:
            n_skipped_patient_limit += 1
            continue

        # position_bin별 max 후보 제한
        if max_per_position_bin is not None and per_bin_count.get(bin_, 0) >= max_per_position_bin:
            n_skipped_position_bin_limit += 1
            continue

        # z + coord NMS (같은 환자 내 이미 선택된 후보와 비교)
        skip_nms = False
        for (pz, pcy, pcx) in per_patient_picks.get(pid, []):
            if abs(z - pz) <= z_nms_window:
                d = ((cy - pcy) ** 2 + (cx - pcx) ** 2) ** 0.5
                if d <= coord_nms_distance:
                    skip_nms = True
                    break
        if skip_nms:
            n_skipped_z_coord_nms += 1
            continue

        selected_rows.append(row)
        per_patient_count[pid] = per_patient_count.get(pid, 0) + 1
        per_bin_count[bin_] = per_bin_count.get(bin_, 0) + 1
        per_patient_picks.setdefault(pid, []).append((z, cy, cx))
        if len(selected_rows) >= top_k:
            break

    if selected_rows:
        diverse_df = pd.DataFrame(selected_rows).reset_index(drop=True)
    else:
        diverse_df = sorted_df.iloc[0:0].copy()

    stats = {
        "n_candidates_scanned": n_scanned,
        "n_candidates_selected": int(len(diverse_df)),
        "n_candidates_skipped_patient_limit": n_skipped_patient_limit,
        "n_candidates_skipped_z_coord_nms": n_skipped_z_coord_nms,
        "n_candidates_skipped_position_bin_limit": n_skipped_position_bin_limit,
    }
    return diverse_df, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="PaDiM patch-level top-k 후보 CSV 생성")
    parser.add_argument(
        "--model",
        default="padim_v1",
        help="모델 이름 (기본: padim_v1)",
    )
    parser.add_argument(
        "--score-col",
        default="padim_score",
        dest="score_col",
        help="정렬 기준 점수 컬럼명 (기본: padim_score)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        dest="top_k",
        help="추출할 patch-level 후보 수 (기본: 50)",
    )
    parser.add_argument(
        "--diverse",
        action="store_true",
        default=False,
        help="환자/슬라이스/좌표 NMS 기반 다양성 제약 모드 활성화",
    )
    parser.add_argument(
        "--max-per-patient",
        type=int,
        default=2,
        dest="max_per_patient",
        help="(diverse) 환자당 최대 후보 수 (기본: 2)",
    )
    parser.add_argument(
        "--z-nms-window",
        type=int,
        default=5,
        dest="z_nms_window",
        help="(diverse) z(slice) NMS 윈도우. |Δz|<=이 값이면 NMS 후보 (기본: 5)",
    )
    parser.add_argument(
        "--coord-nms-distance",
        type=float,
        default=40.0,
        dest="coord_nms_distance",
        help="(diverse) patch 중심 좌표 NMS 거리(pixel, 기본: 40)",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        dest="output_name",
        help="출력 CSV 파일명. 미지정 시 모드별 기본값 사용 (non-diverse: patch_topk.csv, diverse: patch_topk_diverse.csv)",
    )
    parser.add_argument(
        "--score-mode",
        choices=["raw", "bin_z", "bin_percentile", "patient_z"],
        default="raw",
        dest="score_mode",
        help="정렬 기준 점수 변환 모드 (raw: 원본, bin_z: position_bin별 z-score, "
             "bin_percentile: position_bin별 percentile rank, "
             "patient_z: patient_id별 z-score). 기본값: raw",
    )
    parser.add_argument(
        "--min-slice-pure-lung-ratio",
        type=float,
        default=None,
        dest="min_slice_pure_lung_ratio",
        help="후보 필터: slice_pure_lung_ratio >= 이 값. 기본값: None (필터 미적용)",
    )
    parser.add_argument(
        "--max-per-position-bin",
        type=int,
        default=None,
        dest="max_per_position_bin",
        help="(diverse 모드 전용) position_bin별 최대 후보 수. 기본값: None (제한 없음)",
    )
    parser.add_argument(
        "--min-central-distance-ratio-mean",
        type=float,
        default=None,
        dest="min_central_distance_ratio_mean",
        help="후보 필터: central_distance_ratio_mean >= 이 값 (폐 외곽 우선). 기본값: None",
    )
    parser.add_argument(
        "--tie-breaker",
        choices=["none", "percentile_raw_tiebreak", "percentile_central_bonus"],
        default="none",
        dest="tie_breaker",
        help="(score_mode=bin_percentile 전용) ranking_score 동률 해결 방식. 기본값: none",
    )
    parser.add_argument(
        "--tiny-weight",
        type=float,
        default=1e-6,
        dest="tiny_weight",
        help="(tie_breaker 활성 시) normalized_raw_score 가중치. 기본값: 1e-6",
    )
    parser.add_argument(
        "--central-weight",
        type=float,
        default=1e-4,
        dest="central_weight",
        help="(tie_breaker=percentile_central_bonus 전용) central_distance_ratio_mean 가중치. 기본값: 1e-4",
    )
    args = parser.parse_args()

    model = args.model
    score_col = args.score_col
    top_k = args.top_k
    diverse = args.diverse
    max_per_patient = args.max_per_patient
    z_nms_window = args.z_nms_window
    coord_nms_distance = args.coord_nms_distance
    score_mode = args.score_mode
    min_slice_pure_lung_ratio = args.min_slice_pure_lung_ratio
    max_per_position_bin = args.max_per_position_bin
    min_central_distance_ratio_mean = args.min_central_distance_ratio_mean
    tie_breaker = args.tie_breaker
    tiny_weight = args.tiny_weight
    central_weight = args.central_weight

    if top_k <= 0:
        print(f"[ERROR] --top-k는 양의 정수여야 합니다: {top_k}")
        sys.exit(1)

    # diverse 모드 안전 가드
    if diverse:
        if max_per_patient <= 0:
            print(f"[ERROR] --max-per-patient는 양의 정수여야 합니다: {max_per_patient}")
            sys.exit(1)
        if z_nms_window < 0:
            print(f"[ERROR] --z-nms-window는 0 이상이어야 합니다: {z_nms_window}")
            sys.exit(1)
        if coord_nms_distance < 0:
            print(f"[ERROR] --coord-nms-distance는 0 이상이어야 합니다: {coord_nms_distance}")
            sys.exit(1)

    # min-slice-pure-lung-ratio 가드
    if min_slice_pure_lung_ratio is not None:
        if min_slice_pure_lung_ratio < 0.0 or min_slice_pure_lung_ratio > 1.0:
            print(f"[ERROR] --min-slice-pure-lung-ratio는 0.0~1.0 범위여야 합니다: {min_slice_pure_lung_ratio}")
            sys.exit(1)

    # max-per-position-bin 가드 + 모드 제한
    if max_per_position_bin is not None:
        if max_per_position_bin <= 0:
            print(f"[ERROR] --max-per-position-bin은 양의 정수여야 합니다: {max_per_position_bin}")
            sys.exit(1)
        if not diverse:
            print(
                "[ERROR] --max-per-position-bin은 --diverse 모드와 함께만 사용할 수 있습니다.\n"
                "        diverse 모드 활성화 후 다시 시도하세요."
            )
            sys.exit(1)

    # min-central-distance-ratio-mean 가드
    if min_central_distance_ratio_mean is not None:
        if min_central_distance_ratio_mean < 0.0 or min_central_distance_ratio_mean > 1.0:
            print(f"[ERROR] --min-central-distance-ratio-mean은 0.0~1.0 범위여야 합니다: {min_central_distance_ratio_mean}")
            sys.exit(1)

    # tie-breaker 가드: score_mode=bin_percentile일 때만 의미 있음
    if tie_breaker != "none" and score_mode != "bin_percentile":
        print(
            f"[ERROR] --tie-breaker는 --score-mode bin_percentile일 때만 사용할 수 있습니다.\n"
            f"        현재 score_mode={score_mode}"
        )
        sys.exit(1)

    score_dir = (
        REPO_ROOT
        / "outputs"
        / "position-aware-padim-v1"
        / "scores"
        / model
        / "by_patient"
    )
    out_dir = (
        REPO_ROOT
        / "outputs"
        / "position-aware-padim-v1"
        / "candidates"
        / model
    )
    # 출력 파일명 결정 (기존 non-diverse 모드의 기본값 보존)
    if args.output_name is not None:
        output_name = args.output_name
    elif diverse:
        output_name = "patch_topk_diverse.csv"
    else:
        output_name = "patch_topk.csv"

    if not output_name.endswith(".csv"):
        print(f"[ERROR] --output-name은 .csv로 끝나야 합니다: {output_name}")
        sys.exit(1)

    out_csv = out_dir / output_name
    out_meta = out_dir / output_name.replace(".csv", "_metadata.json")

    start_time = time.time()

    # ----------------------------------------------------------------
    # runtime_summary.csv 스키마 점검
    # ----------------------------------------------------------------
    check_and_archive_runtime_csv()

    # ----------------------------------------------------------------
    # 안전 가드: 기존 출력 파일이 있으면 중단
    # ----------------------------------------------------------------
    if out_csv.exists():
        print(
            f"[ERROR] 기존 출력 CSV가 이미 존재합니다: {out_csv}\n"
            "기존 파일을 archive로 이동하거나 삭제한 뒤 다시 실행하세요."
        )
        sys.exit(1)

    # ----------------------------------------------------------------
    # 입력 디렉토리 / CSV 목록 확인
    # ----------------------------------------------------------------
    if not score_dir.exists():
        print(f"[ERROR] score 디렉토리가 없습니다: {score_dir}")
        sys.exit(1)

    csvs = sorted(score_dir.glob("*.csv"))
    n_input = len(csvs)

    print(f"[rank_candidates] model       : {model}")
    print(f"[rank_candidates] score_col   : {score_col}")
    print(f"[rank_candidates] top_k       : {top_k}")
    print(f"[rank_candidates] 입력 디렉토리: {score_dir}")
    print(f"[rank_candidates] 입력 CSV 수 : {n_input}")
    print(f"[rank_candidates] 출력 CSV    : {out_csv}")
    print(f"[rank_candidates] 출력 메타   : {out_meta}")
    print()

    if n_input == 0:
        msg = f"입력 CSV가 없습니다: {score_dir}"
        record_error("__concat__", "no_input_csv", msg, str(score_dir))
        print(f"[ERROR] {msg}")
        sys.exit(1)

    # ----------------------------------------------------------------
    # 각 CSV 로드 + NaN/inf 가드 + concat
    # ScoreAggregator.load_csv가 필수 컬럼/score_col/NaN/inf를 검증한다.
    # ----------------------------------------------------------------
    aggregator = ScoreAggregator(score_col=score_col)

    dfs: list[pd.DataFrame] = []
    n_total_rows = 0
    n_total_nan = 0
    n_total_inf = 0

    for csv_path in csvs:
        patient_id = csv_path.stem
        try:
            df = aggregator.load_csv(str(csv_path))
        except Exception as exc:
            record_error(patient_id, "load_or_validate_error", str(exc), str(csv_path))
            print(f"  [FAIL] {patient_id}: {exc}")
            raise

        col = df[score_col]
        n_nan = int(col.isna().sum())
        n_inf = int(np.isinf(col.to_numpy(dtype=float)).sum())
        n_total_nan += n_nan
        n_total_inf += n_inf
        n_total_rows += len(df)
        dfs.append(df)

        print(
            f"  [OK]   {patient_id}: rows={len(df)}, NaN={n_nan}, inf={n_inf}"
        )

    print()
    print(
        f"[rank_candidates] 합본 rows={n_total_rows}, "
        f"NaN={n_total_nan}, inf={n_total_inf}"
    )

    if n_total_nan > 0 or n_total_inf > 0:
        msg = f"score_col '{score_col}'에 NaN/inf 포함: NaN={n_total_nan}, inf={n_total_inf}"
        record_error("__concat__", "nan_or_inf", msg, str(score_dir))
        raise ValueError(msg)

    # ----------------------------------------------------------------
    # concat
    # ----------------------------------------------------------------
    combined = pd.concat(dfs, ignore_index=True)
    if len(combined) != n_total_rows:
        msg = f"concat 후 rows 불일치: 기대={n_total_rows}, 실제={len(combined)}"
        record_error("__concat__", "concat_mismatch", msg, str(score_dir))
        raise ValueError(msg)

    # ----------------------------------------------------------------
    # ranking_score 계산 (score_mode 별)
    # 합본(필터 전) 기반으로 정상 분포 의미 보존.
    # ----------------------------------------------------------------
    if score_mode == "raw":
        combined["ranking_score"] = combined[score_col].astype(float)
    elif score_mode == "bin_z":
        g = combined.groupby("position_bin")[score_col]
        bin_mean = g.transform("mean")
        bin_std = g.transform("std").replace(0, np.nan)
        combined["ranking_score"] = ((combined[score_col] - bin_mean) / bin_std).fillna(0.0)
    elif score_mode == "bin_percentile":
        bin_pct = combined.groupby("position_bin")[score_col].rank(pct=True)
        if tie_breaker == "none":
            combined["ranking_score"] = bin_pct
        else:
            g_max = float(combined[score_col].max())
            if g_max == 0:
                g_max = 1.0
            normalized_raw = combined[score_col].astype(float) / g_max
            if tie_breaker == "percentile_raw_tiebreak":
                combined["ranking_score"] = bin_pct + tiny_weight * normalized_raw
            elif tie_breaker == "percentile_central_bonus":
                if "central_distance_ratio_mean" not in combined.columns:
                    msg = "합본에 central_distance_ratio_mean 컬럼이 없어 percentile_central_bonus를 적용할 수 없습니다."
                    record_error("__tie_breaker__", "missing_column", msg, str(score_dir))
                    raise ValueError(msg)
                combined["ranking_score"] = (
                    bin_pct
                    + tiny_weight * normalized_raw
                    + central_weight * combined["central_distance_ratio_mean"].astype(float)
                )
            else:
                raise ValueError(f"unknown tie_breaker: {tie_breaker}")
    elif score_mode == "patient_z":
        g = combined.groupby("patient_id")[score_col]
        pat_mean = g.transform("mean")
        pat_std = g.transform("std").replace(0, np.nan)
        combined["ranking_score"] = ((combined[score_col] - pat_mean) / pat_std).fillna(0.0)
    else:
        raise ValueError(f"unknown score_mode: {score_mode}")

    print(
        f"[rank_candidates] score_mode={score_mode}, "
        f"ranking_score: min={combined['ranking_score'].min():.6f}, "
        f"max={combined['ranking_score'].max():.6f}"
    )

    # ----------------------------------------------------------------
    # slice_pure_lung_ratio 필터 (정렬·선택 직전 적용)
    # ----------------------------------------------------------------
    n_before_slice_filter = len(combined)
    if min_slice_pure_lung_ratio is not None:
        if "slice_pure_lung_ratio" not in combined.columns:
            msg = "합본에 slice_pure_lung_ratio 컬럼이 없어 필터를 적용할 수 없습니다."
            record_error("__filter__", "missing_column", msg, str(score_dir))
            raise ValueError(msg)
        combined = combined[combined["slice_pure_lung_ratio"] >= min_slice_pure_lung_ratio].reset_index(drop=True)
    n_after_slice_filter = len(combined)
    print(
        f"[rank_candidates] slice_pure_lung_ratio>={min_slice_pure_lung_ratio} → "
        f"{n_before_slice_filter} → {n_after_slice_filter} rows"
    )

    # central_distance_ratio_mean 필터 (정렬·선택 직전 적용)
    if min_central_distance_ratio_mean is not None:
        if "central_distance_ratio_mean" not in combined.columns:
            msg = "합본에 central_distance_ratio_mean 컬럼이 없어 필터를 적용할 수 없습니다."
            record_error("__filter__", "missing_column", msg, str(score_dir))
            raise ValueError(msg)
        n_before_central_filter = len(combined)
        combined = combined[
            combined["central_distance_ratio_mean"] >= min_central_distance_ratio_mean
        ].reset_index(drop=True)
        print(
            f"[rank_candidates] central_distance_ratio_mean>={min_central_distance_ratio_mean} → "
            f"{n_before_central_filter} → {len(combined)} rows"
        )

    n_after_filters = len(combined)
    if n_after_filters == 0:
        msg = "필터 후 후보 0개 — 임계값을 낮추세요."
        record_error("__filter__", "empty_after_filter", msg, str(score_dir))
        raise ValueError(msg)

    # ----------------------------------------------------------------
    # 정렬·선택: ranking_score 기준
    # ----------------------------------------------------------------
    if top_k > len(combined):
        print(
            f"[rank_candidates] [WARN] top_k({top_k}) > 합본 rows({len(combined)}) "
            f"→ 합본 rows 전체를 반환합니다."
        )

    diverse_stats = None
    if diverse:
        print(
            f"[rank_candidates] diverse 모드: "
            f"max_per_patient={max_per_patient}, "
            f"z_nms_window={z_nms_window}, "
            f"coord_nms_distance={coord_nms_distance}"
        )
        top_df, diverse_stats = select_diverse_candidates(
            combined,
            score_col=score_col,
            top_k=top_k,
            max_per_patient=max_per_patient,
            z_nms_window=z_nms_window,
            coord_nms_distance=coord_nms_distance,
            ranking_col="ranking_score",
            max_per_position_bin=max_per_position_bin,
        )
        top_df = top_df.copy()
        if len(top_df) < top_k:
            print(
                f"[rank_candidates] [WARN] diverse 모드에서 top_k({top_k}) 만큼 "
                f"채우지 못했습니다 (선택={len(top_df)}). 부분 결과로 저장합니다."
            )
        print(
            f"[rank_candidates] diverse 후보 선택 완료 "
            f"(rows={len(top_df)}, scanned={diverse_stats['n_candidates_scanned']}, "
            f"skipped_patient={diverse_stats['n_candidates_skipped_patient_limit']}, "
            f"skipped_bin={diverse_stats['n_candidates_skipped_position_bin_limit']}, "
            f"skipped_nms={diverse_stats['n_candidates_skipped_z_coord_nms']})"
        )
    else:
        # non-diverse: ranking_score 기준 단순 top-k
        top_df = combined.sort_values(
            by=["ranking_score", "patient_id", "local_z", "y0", "x0"],
            ascending=[False, True, True, True, True],
            kind="stable",
        ).head(top_k).copy()
        print(f"[rank_candidates] top_{top_k} 후보 선택 완료 (rows={len(top_df)})")

    # ----------------------------------------------------------------
    # rank / model / score_col / score_mode / ranking_score 컬럼 추가
    # 원본 padim_score(=score_col) 컬럼은 그대로 보존됨.
    # ----------------------------------------------------------------
    top_df.insert(0, "rank", range(1, len(top_df) + 1))
    top_df["model"] = model
    top_df["score_col"] = score_col
    top_df["score_mode"] = score_mode

    # ----------------------------------------------------------------
    # 출력 디렉토리 생성 및 CSV 저장
    # ----------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    top_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[rank_candidates] CSV 저장 완료: {out_csv}")

    # ----------------------------------------------------------------
    # metadata json 저장
    # ----------------------------------------------------------------
    score_min = float(top_df[score_col].min())
    score_max = float(top_df[score_col].max())
    elapsed = time.time() - start_time

    metadata = {
        "script": SCRIPT_NAME,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "score_col": score_col,
        "top_k": top_k,
        "ranking_level": "patch",
        "sort_order": f"{score_col} descending",
        "input_csv_count": n_input,
        "input_total_rows": n_total_rows,
        "n_output_rows": int(len(top_df)),
        "score_min": score_min,
        "score_max": score_max,
        "output_csv": str(out_csv),
        "input_dir": str(score_dir),
        "elapsed_seconds": round(elapsed, 2),
    }
    if diverse:
        metadata.update({
            "diverse": True,
            "max_per_patient": max_per_patient,
            "z_nms_window": z_nms_window,
            "coord_nms_distance": coord_nms_distance,
            "max_per_position_bin": max_per_position_bin,
            "n_candidates_scanned": diverse_stats["n_candidates_scanned"],
            "n_candidates_selected": diverse_stats["n_candidates_selected"],
            "n_candidates_skipped_patient_limit": diverse_stats["n_candidates_skipped_patient_limit"],
            "n_candidates_skipped_z_coord_nms": diverse_stats["n_candidates_skipped_z_coord_nms"],
            "n_candidates_skipped_position_bin_limit": diverse_stats["n_candidates_skipped_position_bin_limit"],
        })

    # score_mode / filter / top10 분포 메타 (모드 무관 공통)
    from collections import Counter
    top10 = top_df.head(10)
    bin_counter = Counter(top10["position_bin"].astype(str))
    pat_counter = Counter(top10["patient_id"].astype(str))
    metadata.update({
        "score_mode": score_mode,
        "ranking_score_col": "ranking_score",
        "min_slice_pure_lung_ratio": min_slice_pure_lung_ratio,
        "min_central_distance_ratio_mean": min_central_distance_ratio_mean,
        "max_per_position_bin": max_per_position_bin,
        "tie_breaker": tie_breaker,
        "tiny_weight": tiny_weight if tie_breaker != "none" else None,
        "central_weight": central_weight if tie_breaker == "percentile_central_bonus" else None,
        "n_candidates_after_slice_filter": int(n_after_slice_filter),
        "n_candidates_after_filters": int(n_after_filters),
        "n_candidates_selected_final": int(len(top_df)),
        "top10_position_bin_distribution": dict(bin_counter),
        "top10_patient_count": dict(pat_counter),
    })
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"[rank_candidates] metadata 저장 완료: {out_meta}")

    # ----------------------------------------------------------------
    # top1 후보 정보 출력
    # ----------------------------------------------------------------
    top1 = top_df.iloc[0]
    print()
    print(
        f"[rank_candidates] top1: patient_id={top1['patient_id']}, "
        f"local_z={int(top1['local_z'])}, "
        f"{score_col}={float(top1[score_col]):.6f}"
    )

    # ----------------------------------------------------------------
    # runtime_summary.csv 기록 (4컬럼 스키마)
    # ----------------------------------------------------------------
    ts = datetime.now().isoformat(timespec="seconds")
    rows = [
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_input_csvs", "value": n_input},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_total_rows_input", "value": n_total_rows},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "top_k", "value": top_k},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_output_rows", "value": int(len(top_df))},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "score_min", "value": round(score_min, 6)},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "score_max", "value": round(score_max, 6)},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "elapsed_seconds", "value": round(elapsed, 2)},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "model", "value": model},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "score_col", "value": score_col},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "output_path", "value": str(out_csv)},
    ]
    record_runtime_rows(rows)
    print(f"[rank_candidates] runtime_summary.csv 기록 완료: {RUNTIME_CSV}")


if __name__ == "__main__":
    main()
