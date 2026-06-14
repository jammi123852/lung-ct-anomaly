"""
score_hu_stat.py: HU Stat Baseline 스코어링 스크립트.

- normal_v1.json의 val + test split을 사용한다 (train 제외).
- HU stat 모델(position_bin_stats.npz)을 로드한다.
- DataLoader를 통해 환자 단위로 스트리밍하며 메모리에 전체 데이터를 올리지 않는다.
- pure_lung mask 기준으로 patch-level z-score를 계산한다.
- outputs/position-aware-padim-v1/scores/hu_stat/by_patient/{patient_id}.csv에 저장한다.
- 기존 score 파일이 있으면 skip하는 resume 기능을 지원한다.
- --limit 옵션으로 처리 환자 수를 제한한다.
- 오류 발생 시 error.csv에 기록한다.
- 실행 정보를 runtime_summary.csv에 기록한다.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (src 하위 패키지 import용)
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from position_aware_padim.config_manager import ConfigManager
from position_aware_padim.data_loader import DataLoader
from position_aware_padim.hu_stat_baseline import HUStatBaseline
from position_aware_padim.path_resolver import PathResolver
from position_aware_padim.patient_splitter import PatientSplitter


REPORTS_DIR = REPO_ROOT / "outputs" / "position-aware-padim-v1" / "reports"
ERROR_CSV = REPORTS_DIR / "error.csv"
RUNTIME_CSV = REPORTS_DIR / "runtime_summary.csv"
MODEL_NPZ = (
    REPO_ROOT
    / "outputs"
    / "position-aware-padim-v1"
    / "models"
    / "hu_stat"
    / "position_bin_stats.npz"
)
SCORE_DIR = (
    REPO_ROOT
    / "outputs"
    / "position-aware-padim-v1"
    / "scores"
    / "hu_stat"
    / "by_patient"
)

ERROR_COLUMNS = ["patient_id", "error_type", "error_msg", "file_logical"]
RUNTIME_COLUMNS = ["timestamp", "script", "metric", "value"]
RUNTIME_SCHEMA_HEADER = "timestamp,script,metric,value"


def record_error(patient_id: str, error_type: str, error_msg: str, file_logical: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not ERROR_CSV.exists() or ERROR_CSV.stat().st_size == 0
    with open(ERROR_CSV, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ERROR_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "patient_id": patient_id,
                "error_type": error_type,
                "error_msg": error_msg,
                "file_logical": file_logical,
            }
        )


def check_and_archive_runtime_csv() -> None:
    """runtime_summary.csv 헤더가 공통 4컬럼 스키마와 다르면 archive로 백업 후 새로 생성한다."""
    if not RUNTIME_CSV.exists():
        return
    if RUNTIME_CSV.stat().st_size == 0:
        return

    with open(RUNTIME_CSV, encoding="utf-8-sig", newline="") as f:
        first_line = f.readline().rstrip("\r\n")

    # BOM 제거 후 비교
    first_line_clean = first_line.lstrip("﻿")
    if first_line_clean == RUNTIME_SCHEMA_HEADER:
        return  # 스키마 일치 → 그대로 append

    # 스키마 불일치 → archive로 이동
    archive_dir = REPORTS_DIR / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"runtime_summary_{ts}.csv"
    shutil.move(str(RUNTIME_CSV), str(archive_path))
    print(f"[score_hu_stat] 기존 runtime_summary.csv 스키마 불일치 → 백업: {archive_path}")


def record_runtime(
    n_requested: int,
    n_scored: int,
    n_skipped_resume: int,
    n_failed: int,
    elapsed: float,
    limit: int | None,
    split_used: str,
) -> None:
    """4컬럼 공통 스키마(timestamp,script,metric,value)로 여러 행을 기록한다."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RUNTIME_CSV.exists() or RUNTIME_CSV.stat().st_size == 0
    ts = datetime.now().isoformat(timespec="seconds")
    script = "score_hu_stat.py"

    rows = [
        {"timestamp": ts, "script": script, "metric": "n_patients_requested", "value": n_requested},
        {"timestamp": ts, "script": script, "metric": "n_patients_scored", "value": n_scored},
        {"timestamp": ts, "script": script, "metric": "n_patients_skipped_resume", "value": n_skipped_resume},
        {"timestamp": ts, "script": script, "metric": "n_patients_failed", "value": n_failed},
        {"timestamp": ts, "script": script, "metric": "elapsed_seconds", "value": round(elapsed, 2)},
        {"timestamp": ts, "script": script, "metric": "limit", "value": str(limit) if limit is not None else "None"},
        {"timestamp": ts, "script": script, "metric": "split_used", "value": split_used},
    ]

    with open(RUNTIME_CSV, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUNTIME_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="HU Stat Baseline 스코어링")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="처리할 최대 환자 수",
    )
    # --full-run: Phase 10 또는 사용자 명시 승인 후에만 사용한다
    parser.add_argument(
        "--full-run",
        action="store_true",
        default=False,
        help="전체 val+test 환자 스코어링 (Phase 10 또는 사용자 명시 승인 후에만 사용)",
    )
    args = parser.parse_args()

    # 안전 가드: --limit 또는 --full-run 중 하나를 반드시 명시해야 한다
    if args.limit is None and not args.full_run:
        print(
            "[ERROR] 안전을 위해 --limit N 또는 --full-run 중 하나를 명시해야 합니다.\n"
            "예: python scripts/score_hu_stat.py --limit 5\n"
            "     python scripts/score_hu_stat.py --full-run"
        )
        sys.exit(1)

    start_time = time.time()

    # ----------------------------------------------------------------
    # runtime_summary.csv 스키마 점검 (오염됐으면 archive 후 재생성)
    # ----------------------------------------------------------------
    check_and_archive_runtime_csv()

    # ----------------------------------------------------------------
    # 설정 로드
    # ----------------------------------------------------------------
    cfg = ConfigManager(str(REPO_ROOT))
    cfg.load_config()

    normal_training_ready: str = cfg.get("paths", "normal_training_ready", "")
    if not normal_training_ready:
        raise ValueError(
            "configs/paths.local.yaml에 'normal_training_ready' 경로가 설정되어 있지 않습니다."
        )

    manifest_path = Path(normal_training_ready) / "manifests" / "patient_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"patient_manifest.csv를 찾을 수 없습니다: {manifest_path}")

    # ----------------------------------------------------------------
    # HU stat 모델 로드
    # ----------------------------------------------------------------
    if not MODEL_NPZ.exists():
        raise FileNotFoundError(
            f"HU stat 모델 파일을 찾을 수 없습니다: {MODEL_NPZ}\n"
            "train_hu_stat.py를 먼저 실행하세요."
        )

    baseline = HUStatBaseline()
    baseline.load(str(MODEL_NPZ))
    print(f"[score_hu_stat] HU stat 모델 로드 완료: {MODEL_NPZ}")
    print(f"[score_hu_stat] position_bin 수: {len(baseline.stats)}")

    # ----------------------------------------------------------------
    # Split 로드 (val + test만 사용)
    # ----------------------------------------------------------------
    splitter = PatientSplitter(str(REPO_ROOT))
    patient_split = splitter.load_split()

    val_patients = list(patient_split.val)
    test_patients = list(patient_split.test)
    target_patients = val_patients + test_patients
    split_used = f"val({len(val_patients)})+test({len(test_patients)})"

    n_total_target = len(target_patients)
    if args.limit is not None:
        target_patients = target_patients[: args.limit]

    print(f"[score_hu_stat] val 환자 수: {len(val_patients)}명")
    print(f"[score_hu_stat] test 환자 수: {len(test_patients)}명")
    print(f"[score_hu_stat] 전체 대상: {n_total_target}명, 이번 실행: {len(target_patients)}명")
    print(f"[score_hu_stat] --limit: {args.limit}")
    print(f"[score_hu_stat] manifest: {manifest_path}")
    print(f"[score_hu_stat] score 저장 위치: {SCORE_DIR}")
    print()

    # ----------------------------------------------------------------
    # PathResolver / DataLoader 생성
    # ----------------------------------------------------------------
    path_resolver = PathResolver(str(manifest_path), normal_training_ready)
    loader = DataLoader(str(manifest_path), path_resolver, str(ERROR_CSV))

    # ----------------------------------------------------------------
    # score 저장 폴더 생성
    # ----------------------------------------------------------------
    SCORE_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # 환자별 스트리밍 스코어링
    # ----------------------------------------------------------------
    n_scored = 0
    n_skipped_resume = 0
    n_failed = 0

    for pid in target_patients:
        score_path = SCORE_DIR / f"{pid}.csv"

        # resume: 기존 파일 있으면 skip
        if score_path.exists():
            n_skipped_resume += 1
            print(f"  [SKIP] {pid}: 이미 존재 (resume)")
            continue

        # 환자 데이터 로드
        data = loader.load_patient_data(pid, mask_type="pure_lung")
        if data is None:
            n_failed += 1
            print(f"  [FAIL] {pid}: 로드 실패 (error.csv 기록됨)")
            continue

        # 스코어링
        try:
            scored_df = baseline.score_patient(data)
        except Exception as exc:
            n_failed += 1
            record_error(pid, "score_error", str(exc), "hu_stat_baseline.score_patient")
            print(f"  [FAIL] {pid}: 스코어링 오류 — {exc}")
            continue

        # CSV 저장
        try:
            scored_df.to_csv(score_path, index=False, encoding="utf-8-sig")
            n_scored += 1
            n_patches = len(scored_df)
            n_nan = int(scored_df["hu_z_score"].isna().sum())
            print(
                f"  [OK]   {pid}: {n_patches}개 patch, NaN={n_nan}, "
                f"저장={score_path.name}"
            )
        except Exception as exc:
            n_failed += 1
            record_error(pid, "save_error", str(exc), str(score_path))
            print(f"  [FAIL] {pid}: CSV 저장 오류 — {exc}")
            continue

    elapsed = time.time() - start_time

    # ----------------------------------------------------------------
    # 결과 요약 출력
    # ----------------------------------------------------------------
    print()
    print(f"[score_hu_stat] 완료: {n_scored}명 스코어링, {n_skipped_resume}명 skip(resume), {n_failed}명 실패, {elapsed:.1f}초")
    print(f"[score_hu_stat] score 파일 위치: {SCORE_DIR}")

    # ----------------------------------------------------------------
    # runtime_summary.csv 기록
    # ----------------------------------------------------------------
    record_runtime(
        n_requested=len(target_patients),
        n_scored=n_scored,
        n_skipped_resume=n_skipped_resume,
        n_failed=n_failed,
        elapsed=elapsed,
        limit=args.limit,
        split_used=split_used,
    )
    print(f"[score_hu_stat] runtime_summary.csv 기록 완료: {RUNTIME_CSV}")


if __name__ == "__main__":
    main()
