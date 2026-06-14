"""
train_hu_stat.py: HU Stat Baseline 학습 스크립트.

- normal_v1.json의 train split만 사용한다.
- DataLoader를 통해 환자 단위로 스트리밍하며 메모리에 전체 데이터를 올리지 않는다.
- pure_lung mask 내부 픽셀만 사용한다.
- position_bin별 HU mean/std를 계산하여 position_bin_stats.npz에 저장한다.
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
OUTPUT_NPZ = (
    REPO_ROOT
    / "outputs"
    / "position-aware-padim-v1"
    / "models"
    / "hu_stat"
    / "position_bin_stats.npz"
)

ERROR_COLUMNS = ["patient_id", "error_type", "error_msg", "file_logical"]
# create_split.py 등 다른 스크립트와 공통으로 사용하는 4컬럼 스키마
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
    print(f"[train_hu_stat] 기존 runtime_summary.csv 스키마 불일치 → 백업: {archive_path}")


def record_runtime(
    n_requested: int,
    n_processed: int,
    n_failed: int,
    elapsed: float,
    limit: int | None,
    n_patches_skipped_mask: int = 0,
) -> None:
    """4컬럼 공통 스키마(timestamp,script,metric,value)로 여러 행을 기록한다."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RUNTIME_CSV.exists() or RUNTIME_CSV.stat().st_size == 0
    ts = datetime.now().isoformat(timespec="seconds")
    script = "train_hu_stat.py"

    rows = [
        {"timestamp": ts, "script": script, "metric": "n_patients_requested", "value": n_requested},
        {"timestamp": ts, "script": script, "metric": "n_patients_processed", "value": n_processed},
        {"timestamp": ts, "script": script, "metric": "n_patients_failed", "value": n_failed},
        {"timestamp": ts, "script": script, "metric": "elapsed_seconds", "value": round(elapsed, 2)},
        {"timestamp": ts, "script": script, "metric": "output_path", "value": str(OUTPUT_NPZ)},
        {"timestamp": ts, "script": script, "metric": "limit", "value": str(limit) if limit is not None else "None"},
        {"timestamp": ts, "script": script, "metric": "n_patches_skipped_mask", "value": n_patches_skipped_mask},
    ]

    with open(RUNTIME_CSV, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUNTIME_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="HU Stat Baseline 학습")
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
        help="전체 train 환자 학습 (Phase 10 또는 사용자 명시 승인 후에만 사용)",
    )
    args = parser.parse_args()

    # 안전 가드: --limit 또는 --full-run 중 하나를 반드시 명시해야 한다
    if args.limit is None and not args.full_run:
        print(
            "[ERROR] 안전을 위해 --limit N 또는 --full-run 중 하나를 명시해야 합니다.\n"
            "예: python scripts/train_hu_stat.py --limit 5\n"
            "     python scripts/train_hu_stat.py --full-run"
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
    # Split 로드 (normal_v1.json의 train split만 사용)
    # ----------------------------------------------------------------
    splitter = PatientSplitter(str(REPO_ROOT))
    patient_split = splitter.load_split()
    train_patients = list(patient_split.train)

    n_total_train = len(train_patients)
    if args.limit is not None:
        train_patients = train_patients[: args.limit]

    print(f"[train_hu_stat] train 환자 수: 전체 {n_total_train}명, 이번 실행 {len(train_patients)}명")
    print(f"[train_hu_stat] --limit: {args.limit}")
    print(f"[train_hu_stat] manifest: {manifest_path}")
    print(f"[train_hu_stat] output: {OUTPUT_NPZ}")
    print()

    # ----------------------------------------------------------------
    # PathResolver / DataLoader 생성
    # ----------------------------------------------------------------
    path_resolver = PathResolver(str(manifest_path), normal_training_ready)
    loader = DataLoader(str(manifest_path), path_resolver, str(ERROR_CSV))

    # ----------------------------------------------------------------
    # 환자별 스트리밍 generator
    # ----------------------------------------------------------------
    n_processed = 0
    n_failed = 0

    def patient_stream():
        nonlocal n_processed, n_failed
        for pid in train_patients:
            data = loader.load_patient_data(pid, mask_type="pure_lung")
            if data is None:
                n_failed += 1
                print(f"  [SKIP] {pid}: 로드 실패 (error.csv 기록됨)")
                continue
            n_processed += 1
            print(f"  [OK]   {pid}: ct_hu={data['ct_hu'].shape}, patches={len(data['patch_df'])}")
            yield data

    # ----------------------------------------------------------------
    # HUStatBaseline 학습
    # ----------------------------------------------------------------
    baseline = HUStatBaseline()
    print("[train_hu_stat] 학습 시작...")
    try:
        baseline.train(patient_stream())
    except Exception as exc:
        record_error("__train__", "train_error", str(exc), "train_hu_stat")
        raise

    elapsed = time.time() - start_time
    print(f"\n[train_hu_stat] 학습 완료: {n_processed}명 처리, {n_failed}명 실패, {elapsed:.1f}초")

    if not baseline.stats:
        record_error("__train__", "empty_stats", "학습 후 stats가 비어 있습니다.", "hu_stat_baseline")
        raise RuntimeError("학습 결과가 비어 있습니다. 데이터 경로와 mask를 확인하세요.")

    # ----------------------------------------------------------------
    # 저장
    # ----------------------------------------------------------------
    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    baseline.save(str(OUTPUT_NPZ))
    print(f"[train_hu_stat] 저장 완료: {OUTPUT_NPZ}")

    # ----------------------------------------------------------------
    # position_bin별 요약 출력
    # ----------------------------------------------------------------
    print("\n[position_bin 통계 요약]")
    print(f"{'position_bin':<25} {'count':>10} {'mean':>10} {'std':>10}")
    print("-" * 60)
    for pos_bin in sorted(baseline.stats.keys()):
        s = baseline.stats[pos_bin]
        print(f"{pos_bin:<25} {s['count']:>10,} {s['mean']:>10.2f} {s['std']:>10.2f}")

    # ----------------------------------------------------------------
    # runtime_summary.csv 기록
    # ----------------------------------------------------------------
    record_runtime(
        n_requested=len(train_patients),
        n_processed=n_processed,
        n_failed=n_failed,
        elapsed=elapsed,
        limit=args.limit,
        n_patches_skipped_mask=baseline.train_summary.get("n_patches_skipped_mask", 0),
    )
    print(f"\n[train_hu_stat] runtime_summary.csv 기록 완료: {RUNTIME_CSV}")


if __name__ == "__main__":
    main()
