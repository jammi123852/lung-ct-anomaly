"""
train_padim.py: Position-Aware PaDiM 분포 학습 스크립트 (Task 6.4).

- normal_v1.json의 train split만 사용한다.
- DataLoader를 통해 환자 단위로 스트리밍하며 메모리에 전체 데이터를 올리지 않는다.
- FeatureExtractor(ResNet18)를 통해 slice 단위 feature를 추출한다.
- PaDiMModel.train으로 position_bin별 분포를 streaming 누적 방식으로 학습한다.
- feature를 list/array에 append해서 RAM에 쌓는 방식은 PaDiMModel 내부에서 금지된다.
- --limit 옵션으로 처리 환자 수를 제한한다.
- 오류 발생 시 error.csv에 기록한다.
- 실행 정보를 runtime_summary.csv에 기록한다.
- 결과: outputs/position-aware-padim-v1/models/padim_v1/distributions/position_bin_stats.npz
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
from position_aware_padim.feature_extractor import FeatureExtractor
from position_aware_padim.padim_model import PaDiMModel
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
    / "padim_v1"
    / "distributions"
    / "position_bin_stats.npz"
)
SELECTED_INDICES_PATH = (
    REPO_ROOT
    / "outputs"
    / "position-aware-padim-v1"
    / "models"
    / "padim_v1"
    / "distributions"
    / "selected_feature_indices.npy"
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

    first_line_clean = first_line.lstrip("﻿")
    if first_line_clean == RUNTIME_SCHEMA_HEADER:
        return

    archive_dir = REPORTS_DIR / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"runtime_summary_{ts}.csv"
    shutil.move(str(RUNTIME_CSV), str(archive_path))
    print(f"[train_padim] 기존 runtime_summary.csv 스키마 불일치 → 백업: {archive_path}")


def record_runtime(
    n_requested: int,
    n_processed: int,
    n_failed: int,
    elapsed: float,
    limit: int | None,
    n_patches_used: int = 0,
    n_patches_skipped: int = 0,
) -> None:
    """4컬럼 공통 스키마(timestamp,script,metric,value)로 여러 행을 기록한다."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RUNTIME_CSV.exists() or RUNTIME_CSV.stat().st_size == 0
    ts = datetime.now().isoformat(timespec="seconds")
    script = "train_padim.py"

    rows = [
        {"timestamp": ts, "script": script, "metric": "n_patients_requested", "value": n_requested},
        {"timestamp": ts, "script": script, "metric": "n_patients_processed", "value": n_processed},
        {"timestamp": ts, "script": script, "metric": "n_patients_failed", "value": n_failed},
        {"timestamp": ts, "script": script, "metric": "elapsed_seconds", "value": round(elapsed, 2)},
        {"timestamp": ts, "script": script, "metric": "output_path", "value": str(OUTPUT_NPZ)},
        {"timestamp": ts, "script": script, "metric": "limit", "value": str(limit) if limit is not None else "None"},
        {"timestamp": ts, "script": script, "metric": "n_patches_used", "value": n_patches_used},
        {"timestamp": ts, "script": script, "metric": "n_patches_skipped", "value": n_patches_skipped},
    ]

    with open(RUNTIME_CSV, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUNTIME_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Position-Aware PaDiM 분포 학습")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="처리할 최대 환자 수",
    )
    parser.add_argument(
        "--full-run",
        action="store_true",
        default=False,
        help="전체 train 환자 학습 (사용자 명시 승인 후에만 사용)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="기존 position_bin_stats.npz를 덮어쓴다 (사용자 명시 승인 후에만 사용)",
    )
    parser.add_argument(
        "--archive-existing",
        action="store_true",
        default=False,
        help=(
            "기존 position_bin_stats.npz를 "
            "archive/position_bin_stats_<timestamp>.npz로 이동 후 새로 저장한다"
        ),
    )
    parser.add_argument(
        "--paths-config",
        type=str,
        default="configs/paths.local.yaml",
        help="paths config yaml 파일 경로 (기본: configs/paths.local.yaml, v2: configs/paths.local.v2_roi0_0.yaml)",
    )
    parser.add_argument(
        "--mask-type",
        type=str,
        default="pure_lung",
        choices=["pure_lung", "roi_0_0"],
        help="학습에 사용할 mask 종류 (기본: pure_lung, v2: roi_0_0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/position-aware-padim-v1/models/padim_v1",
        help="출력 디렉터리 (기본: padim_v1, v2 학습: outputs/position-aware-padim-v1/models/padim_v2_roi0_0)",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="outputs/position-aware-padim-v1/reports",
        help="reports 디렉터리 (기본: v1 reports, v2 학습: outputs/position-aware-padim-v1/reports_v2_roi0_0)",
    )
    args = parser.parse_args()

    # --output-dir 기반으로 OUTPUT_NPZ 경로 결정 (기본값은 기존 v1 경로)
    global OUTPUT_NPZ, REPORTS_DIR, ERROR_CSV, RUNTIME_CSV
    OUTPUT_NPZ = REPO_ROOT / args.output_dir / "distributions" / "position_bin_stats.npz"
    REPORTS_DIR = REPO_ROOT / args.reports_dir
    ERROR_CSV = REPORTS_DIR / "error.csv"
    RUNTIME_CSV = REPORTS_DIR / "runtime_summary.csv"

    # 안전 가드: --limit 또는 --full-run 중 하나를 반드시 명시해야 한다
    if args.limit is None and not args.full_run:
        print(
            "[ERROR] 안전을 위해 --limit N 또는 --full-run 중 하나를 명시해야 합니다.\n"
            "예: python scripts/train_padim.py --limit 1\n"
            "     python scripts/train_padim.py --full-run"
        )
        sys.exit(1)

    # 덮어쓰기 보호 가드: 기존 npz가 있을 때 --overwrite 또는 --archive-existing 없으면 중단
    if OUTPUT_NPZ.exists() and not args.overwrite and not args.archive_existing:
        print(
            f"[ERROR] 기존 position_bin_stats.npz가 이미 존재합니다: {OUTPUT_NPZ}\n"
            "기본 동작은 중단입니다. 아래 옵션 중 하나를 명시하세요:\n"
            "  --overwrite         : 기존 파일을 덮어씁니다\n"
            "  --archive-existing  : 기존 파일을 archive/ 로 이동 후 새로 저장합니다"
        )
        sys.exit(1)

    # --archive-existing: 기존 npz를 archive/ 로 이동
    if OUTPUT_NPZ.exists() and args.archive_existing:
        archive_dir = OUTPUT_NPZ.parent / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = archive_dir / f"position_bin_stats_{ts}.npz"
        shutil.move(str(OUTPUT_NPZ), str(archive_path))
        print(f"[train_padim] 기존 position_bin_stats.npz → 아카이브: {archive_path}")

    start_time = time.time()

    # ----------------------------------------------------------------
    # runtime_summary.csv 스키마 점검
    # ----------------------------------------------------------------
    check_and_archive_runtime_csv()

    # ----------------------------------------------------------------
    # 설정 로드
    # ----------------------------------------------------------------
    cfg = ConfigManager(str(REPO_ROOT))
    cfg.load_config(paths_yaml=Path(args.paths_config).name)

    normal_training_ready: str = cfg.get("paths", "normal_training_ready", "")
    if not normal_training_ready:
        raise ValueError(
            "configs/paths.local.yaml에 'normal_training_ready' 경로가 설정되어 있지 않습니다."
        )

    manifest_path = Path(normal_training_ready) / "manifests" / "patient_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"patient_manifest.csv를 찾을 수 없습니다: {manifest_path}")

    # ----------------------------------------------------------------
    # selected_feature_indices.npy 절대경로 확인
    # ----------------------------------------------------------------
    if not SELECTED_INDICES_PATH.exists():
        raise FileNotFoundError(
            f"selected_feature_indices.npy를 찾을 수 없습니다: {SELECTED_INDICES_PATH}"
        )

    # ----------------------------------------------------------------
    # Split 로드 (normal_v1.json의 train split만 사용)
    # ----------------------------------------------------------------
    splitter = PatientSplitter(str(REPO_ROOT))
    patient_split = splitter.load_split()
    train_patients = list(patient_split.train)

    n_total_train = len(train_patients)
    if args.limit is not None:
        train_patients = train_patients[: args.limit]

    print(f"[train_padim] train 환자 수: 전체 {n_total_train}명, 이번 실행 {len(train_patients)}명")
    print(f"[train_padim] --limit: {args.limit}")
    print(f"[train_padim] manifest: {manifest_path}")
    print(f"[train_padim] selected_feature_indices: {SELECTED_INDICES_PATH}")
    print(f"[train_padim] output: {OUTPUT_NPZ}")
    print()

    # ----------------------------------------------------------------
    # PathResolver / DataLoader 생성
    # ----------------------------------------------------------------
    path_resolver = PathResolver(str(manifest_path), normal_training_ready)
    loader = DataLoader(str(manifest_path), path_resolver, str(ERROR_CSV), use_mmap=True)

    # ----------------------------------------------------------------
    # FeatureExtractor 생성 (ResNet18, ImageNet1K V1 weights)
    # ----------------------------------------------------------------
    print("[train_padim] FeatureExtractor 초기화 중...")
    feature_extractor = FeatureExtractor()
    print(f"[train_padim] FeatureExtractor device: {feature_extractor.device}")

    # ----------------------------------------------------------------
    # PaDiMModel 생성 (selected_feature_indices 절대경로 필수)
    # ----------------------------------------------------------------
    model = PaDiMModel(
        selected_feature_indices_path=str(SELECTED_INDICES_PATH),
        feature_dim=100,
        eps=1e-5,
    )
    print(f"[train_padim] PaDiMModel 초기화 완료: feature_dim={model.feature_dim}")

    # ----------------------------------------------------------------
    # 환자별 스트리밍 generator
    # ----------------------------------------------------------------
    n_processed = 0
    n_failed = 0

    def patient_stream():
        nonlocal n_processed, n_failed
        for pid in train_patients:
            data = loader.load_patient_data(pid, mask_type=args.mask_type)
            if data is None:
                n_failed += 1
                record_error(pid, "load_failed", "DataLoader.load_patient_data returned None", "patient_data")
                print(f"  [SKIP] {pid}: 로드 실패 (error.csv 기록됨)")
                continue
            n_processed += 1
            print(f"  [OK]   {pid}: ct_hu={data['ct_hu'].shape}, patches={len(data['patch_df'])}")
            yield data

    # ----------------------------------------------------------------
    # PaDiMModel 학습 (split=None → patient_stream이 이미 limit 적용)
    # ----------------------------------------------------------------
    print("[train_padim] 학습 시작...")
    try:
        model.train(feature_extractor, patient_stream(), split=None)
    except Exception as exc:
        record_error("__train__", "train_error", str(exc), "train_padim")
        raise

    elapsed = time.time() - start_time
    summary = model.train_summary
    print(
        f"\n[train_padim] 학습 완료: "
        f"{summary['n_patients_success']}명 성공, "
        f"{summary['n_patients_failed']}명 실패, "
        f"{elapsed:.1f}초"
    )
    print(f"[train_padim] 사용 patch 수: {summary['n_patches_used']:,}")
    print(f"[train_padim] 스킵 patch 수: {summary['n_patches_skipped']:,}")

    if not model.stats:
        record_error("__train__", "empty_stats", "학습 후 stats가 비어 있습니다.", "padim_model")
        raise RuntimeError("학습 결과가 비어 있습니다. 데이터 경로와 mask를 확인하세요.")

    # ----------------------------------------------------------------
    # 저장
    # ----------------------------------------------------------------
    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(OUTPUT_NPZ))
    print(f"[train_padim] 저장 완료: {OUTPUT_NPZ}")

    # ----------------------------------------------------------------
    # position_bin별 요약 출력
    # ----------------------------------------------------------------
    import numpy as np

    print("\n[position_bin 통계 요약]")
    print(f"{'key':<25} {'count':>10} {'mean[0]':>12} {'mean[-1]':>12}")
    print("-" * 65)
    for key in model._all_keys():
        s = model.stats.get(key, {})
        count = s.get("count", 0)
        if count > 0 and "mean" in s:
            mean = s["mean"]
            print(f"{key:<25} {count:>10,} {mean[0]:>12.6f} {mean[-1]:>12.6f}")
            nan_count = int(np.isnan(mean).sum())
            inf_count = int(np.isinf(mean).sum())
            if nan_count > 0 or inf_count > 0:
                print(f"  [WARN] mean에 NaN={nan_count}, inf={inf_count} 포함!")
        else:
            print(f"{key:<25} {'insufficient':>10}")

    # ----------------------------------------------------------------
    # 실패한 환자 요약
    # ----------------------------------------------------------------
    patient_failures = summary.get("patient_failures", [])
    if patient_failures:
        print(f"\n[train_padim] 실패 환자 {len(patient_failures)}명:")
        for f in patient_failures:
            print(f"  {f['patient_id']}: {f['error_type']} — {f['error_msg'][:80]}")

    # ----------------------------------------------------------------
    # runtime_summary.csv 기록
    # ----------------------------------------------------------------
    record_runtime(
        n_requested=len(train_patients),
        n_processed=summary["n_patients_success"],
        n_failed=summary["n_patients_failed"],
        elapsed=elapsed,
        limit=args.limit,
        n_patches_used=summary["n_patches_used"],
        n_patches_skipped=summary["n_patches_skipped"],
    )
    print(f"\n[train_padim] runtime_summary.csv 기록 완료: {RUNTIME_CSV}")


if __name__ == "__main__":
    main()
