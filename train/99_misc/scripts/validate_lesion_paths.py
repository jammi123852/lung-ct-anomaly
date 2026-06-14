"""
validate_lesion_paths.py: lesion 테스트 데이터셋의 Phase 8 평가 전 구조 검증 게이트.

- configs/paths.local.yaml의 nsclc_msd_usable_only 경로를 읽는다.
- lesion 데이터셋용 PathResolver를 생성하고
  DataValidator.validate_lesion_paths()로 구조 무결성을 검증한다.
- 결과를 reports에 저장한다.

이 스크립트는 병변 평가 / 스코어링 / 학습을 실행하지 않는다. 구조 검증만 한다.
정상 학습셋용 PathResolver / DataValidator 기존 동작은 건드리지 않는다.
"""

from __future__ import annotations

import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

# 프로젝트 루트를 sys.path에 추가 (src 하위 패키지 import용)
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import argparse

from position_aware_padim.path_resolver import PathResolver
from position_aware_padim.data_validator import DataValidator

REPORTS_DIR = REPO_ROOT / "outputs" / "position-aware-padim-v1" / "reports"
SUMMARY_CSV_V1 = REPORTS_DIR / "lesion_path_validation_summary.csv"
SUMMARY_CSV_V2 = REPORTS_DIR / "lesion_path_validation_summary_v2.csv"
ERROR_CSV = REPORTS_DIR / "error.csv"
RUNTIME_CSV = REPORTS_DIR / "runtime_summary.csv"

ERROR_COLUMNS = ["patient_id", "error_type", "error_msg", "file_logical"]
RUNTIME_COLUMNS = ["timestamp", "script", "metric", "value"]

SCRIPT_NAME = "validate_lesion_paths.py"


def append_errors(error_df) -> None:
    """error_df를 기존 error.csv(4컬럼)에 append. 형식 유지."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not ERROR_CSV.exists() or ERROR_CSV.stat().st_size == 0
    with open(ERROR_CSV, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ERROR_COLUMNS)
        if write_header:
            writer.writeheader()
        for _, r in error_df.iterrows():
            writer.writerow({c: r.get(c, "") for c in ERROR_COLUMNS})


def record_runtime_rows(rows) -> None:
    """runtime_summary.csv(4컬럼)에 append. 형식 유지."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RUNTIME_CSV.exists() or RUNTIME_CSV.stat().st_size == 0
    with open(RUNTIME_CSV, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUNTIME_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="lesion 데이터셋 구조 검증 게이트")
    parser.add_argument(
        "--dataset-profile",
        type=str,
        default="v1_model_roi",
        choices=["v1_model_roi", "v2_roi_0_0"],
        help=(
            "검증할 데이터셋 profile. "
            "v1_model_roi: 기존 model_roi 기반 데이터셋 (기본값). "
            "v2_roi_0_0: roi_0_0 기반 신규 데이터셋."
        ),
    )
    args = parser.parse_args()
    dataset_profile = args.dataset_profile
    is_v2 = (dataset_profile == "v2_roi_0_0")
    SUMMARY_CSV = SUMMARY_CSV_V2 if is_v2 else SUMMARY_CSV_V1

    # --- 1~3. config 읽기 및 경로 유효성 검사 ---
    cfg_path = REPO_ROOT / "configs" / "paths.local.yaml"
    if not cfg_path.exists():
        print(f"[ERROR] config 없음: {cfg_path}")
        sys.exit(1)
    with open(cfg_path, encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f) or {}

    cfg_key = "nsclc_msd_usable_only_v2" if is_v2 else "nsclc_msd_usable_only"
    base = (cfg.get(cfg_key) or "").strip()
    if not base:
        print(f"[ERROR] configs/paths.local.yaml의 {cfg_key}가 비어 있습니다.")
        print("        병변 테스트 데이터 경로를 먼저 설정하세요.")
        sys.exit(1)

    base_path = Path(base)
    if not base_path.exists():
        print(f"[ERROR] nsclc_msd_usable_only 경로가 존재하지 않습니다: {base_path}")
        sys.exit(1)

    manifest_path = base_path / "manifests" / "patient_manifest.csv"
    if not manifest_path.exists():
        print(f"[ERROR] patient_manifest.csv 없음: {manifest_path}")
        sys.exit(1)

    # anchor override 키가 있으면 사용, 없으면 PathResolver가 base 폴더명을 자동 사용
    anchor = (cfg.get("nsclc_msd_usable_only_anchor") or "").strip() or None

    print(f"[validate_lesion_paths] dataset_profile = {dataset_profile}")
    print(f"[validate_lesion_paths] base_path = {base_path}")
    print(f"[validate_lesion_paths] manifest  = {manifest_path}")
    print(f"[validate_lesion_paths] anchor    = {anchor if anchor else f'(자동: {base_path.name})'}")
    print()

    # --- lesion PathResolver + DataValidator 실행 ---
    start = time.time()
    resolver = PathResolver(str(manifest_path), str(base_path), anchor=anchor)
    validator = DataValidator(resolver)
    summary_df, error_df, stats = validator.validate_lesion_paths(
        sample_n=5, dataset_profile=dataset_profile
    )
    elapsed = time.time() - start

    # --- 결과 저장 ---
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    n_errors = len(error_df)
    if n_errors > 0:
        append_errors(error_df)

    # --- 게이트 판정 ---
    folders_ok = (
        stats["manifests_dir_exists"]
        and stats["volumes_npy_dir_exists"]
        and stats["patch_index_dir_exists"]
        and stats["manifest_csv_exists"]
    )
    gate_pass = (
        stats["root_exists"]
        and folders_ok
        and bool(stats["count_match"])
        and bool(stats["sample_shape_match_all"])
        and bool(stats["patch_required_ok"])
        and n_errors == 0
    )
    verdict = "통과" if gate_pass else "미통과 / 확인 필요"

    # --- 콘솔 요약 ---
    print(f"[validate_lesion_paths] 루트 존재          : {stats['root_exists']}")
    print(f"[validate_lesion_paths] 필수 폴더 존재      : {folders_ok}")
    print(f"[validate_lesion_paths] volumes_npy 폴더 수 : {stats['volumes_npy_count']}")
    print(f"[validate_lesion_paths] patch CSV 수        : {stats['patch_csv_count']}")
    print(f"[validate_lesion_paths] manifest 케이스 수  : {stats['manifest_case_count']}")
    print(f"[validate_lesion_paths] 개수 일치(폴더=CSV) : {stats['count_match']}")
    print(f"[validate_lesion_paths] group 집계          : {stats['group_counts']}")
    print(f"[validate_lesion_paths] 샘플 shape 일치     : {stats['sample_shape_match_all']} (n={stats['sample_n']})")
    print(f"[validate_lesion_paths] patch 필수 컬럼 OK  : {stats['patch_required_ok']}")
    if stats["patch_required_missing"]:
        print(f"[validate_lesion_paths]   누락 필수 컬럼    : {stats['patch_required_missing']}")
    print(f"[validate_lesion_paths] patch 권장 컬럼 존재: {stats['patch_recommended_present']}")
    if stats["patch_recommended_missing"]:
        print(f"[validate_lesion_paths]   미존재 권장 컬럼  : {stats['patch_recommended_missing']}")
    print(f"[validate_lesion_paths] error 건수          : {n_errors}")
    print()
    print(f"[validate_lesion_paths] 검증 게이트 판정    : {verdict}")
    print(f"[validate_lesion_paths] summary 저장        : {SUMMARY_CSV}")
    print(f"[validate_lesion_paths] 소요 시간           : {elapsed:.1f}s")

    # --- runtime_summary 기록 ---
    ts = datetime.now().isoformat(timespec="seconds")
    record_runtime_rows([
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "base_path", "value": str(base_path)},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "volumes_npy_count", "value": stats["volumes_npy_count"]},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "patch_csv_count", "value": stats["patch_csv_count"]},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "manifest_case_count", "value": stats["manifest_case_count"]},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "count_match", "value": str(stats["count_match"])},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "group_counts", "value": str(stats["group_counts"])},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "sample_shape_match_all", "value": str(stats["sample_shape_match_all"])},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "patch_required_ok", "value": str(stats["patch_required_ok"])},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "n_errors", "value": n_errors},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "gate_verdict", "value": verdict},
        {"timestamp": ts, "script": SCRIPT_NAME, "metric": "elapsed_seconds", "value": round(elapsed, 2)},
    ])
    print(f"[validate_lesion_paths] runtime_summary.csv 기록 완료")


if __name__ == "__main__":
    main()
