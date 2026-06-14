"""
create_split.py: 정상 데이터 train/val/test split을 생성 또는 변환한다.

우선순위:
  1. manifests/train_val_test_split.csv가 존재하면 load_from_csv → convert_csv_to_json
  2. CSV가 없을 때만 사용자 명시 승인 후 create_split 호출 (자동 생성 금지)

출력:
  outputs/position-aware-padim-v1/splits/normal_v1.json
  outputs/position-aware-padim-v1/reports/runtime_summary.csv
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from position_aware_padim.config_manager import ConfigManager
from position_aware_padim.patient_splitter import PatientSplitter

REPORTS_DIR = _REPO_ROOT / "outputs" / "position-aware-padim-v1" / "reports"
RUNTIME_SUMMARY_CSV = REPORTS_DIR / "runtime_summary.csv"
SCRIPT_NAME = "create_split.py"


def append_runtime_summary(rows: list[dict]) -> None:
    """runtime_summary.csv에 행을 추가한다."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RUNTIME_SUMMARY_CSV.exists() or RUNTIME_SUMMARY_CSV.stat().st_size == 0
    with open(RUNTIME_SUMMARY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "script", "metric", "value"])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 설정 로드
    cfg_manager = ConfigManager(repo_root=str(_REPO_ROOT))
    try:
        cfg_manager.load_config()
        cfg_manager.validate_config()
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] 설정 로드 실패: {exc}")
        sys.exit(1)

    splitter = PatientSplitter(repo_root=str(_REPO_ROOT))

    # normal_v1.json 사전 확인
    if splitter.normal_v1_json.exists():
        print(f"[INFO] normal_v1.json이 이미 존재합니다: {splitter.normal_v1_json}")
        print("[INFO] 덮어쓰기는 사용자 명시 승인 없이 진행하지 않습니다.")
        print("[INFO] 기존 파일을 유지합니다. 덮어쓰려면 수동으로 삭제 후 재실행하세요.")

        # 기존 파일을 load하여 통계만 출력
        try:
            existing_split = splitter.load_split()
            stats = splitter.validate_split(existing_split)
            _print_stats(stats, source="기존 normal_v1.json")
            _write_summary(now_str, stats, note="existing_json_kept")
        except Exception as exc:
            print(f"[WARN] 기존 normal_v1.json 읽기 실패: {exc}")
        sys.exit(0)

    # 우선순위 1: train_val_test_split.csv 존재 시
    if splitter.split_csv_path.exists():
        print(f"[INFO] train_val_test_split.csv 발견: {splitter.split_csv_path}")
        print("[INFO] load_from_csv → convert_csv_to_json 실행 중...")

        try:
            patient_split = splitter.load_from_csv()
        except (FileNotFoundError, ValueError) as exc:
            print(f"[ERROR] load_from_csv 실패: {exc}")
            sys.exit(1)

        try:
            out_path = splitter.convert_csv_to_json(patient_split, overwrite=False)
        except FileExistsError as exc:
            print(f"[ERROR] {exc}")
            sys.exit(1)

        stats = splitter.validate_split(patient_split)
        _print_stats(stats, source=str(splitter.split_csv_path))
        _write_summary(now_str, stats, note="loaded_from_csv")

        print(f"\n[OK] normal_v1.json 생성 완료: {out_path}")

    # 우선순위 2: CSV 없을 때 — 자동 생성 금지
    else:
        print(f"[INFO] train_val_test_split.csv를 찾을 수 없습니다: {splitter.split_csv_path}")
        print("[INFO] 자동 새 split 생성은 금지되어 있습니다.")
        print("[INFO] 사용자 명시 승인 후에만 create_split을 사용할 수 있습니다.")
        print("[INFO] CSV 파일을 준비하거나 수동으로 PatientSplitter.create_split을 호출하세요.")

        _write_summary(
            now_str,
            {"n_train": 0, "n_val": 0, "n_test": 0, "n_total": 0, "has_duplicates": False},
            note="csv_not_found_no_split_created",
        )
        sys.exit(1)


def _print_stats(stats: dict, source: str) -> None:
    print()
    print("=" * 50)
    print(f"source          : {source}")
    print(f"train 환자 수   : {stats['n_train']}")
    print(f"val 환자 수     : {stats['n_val']}")
    print(f"test 환자 수    : {stats['n_test']}")
    print(f"전체 환자 수    : {stats['n_total']}")
    print(f"중복 patient_id : {'있음 ' + str(stats.get('duplicates', [])) if stats['has_duplicates'] else '없음'}")
    print("=" * 50)


def _write_summary(now_str: str, stats: dict, note: str) -> None:
    rows = [
        {"timestamp": now_str, "script": SCRIPT_NAME, "metric": "n_train",       "value": stats.get("n_train", 0)},
        {"timestamp": now_str, "script": SCRIPT_NAME, "metric": "n_val",         "value": stats.get("n_val", 0)},
        {"timestamp": now_str, "script": SCRIPT_NAME, "metric": "n_test",        "value": stats.get("n_test", 0)},
        {"timestamp": now_str, "script": SCRIPT_NAME, "metric": "n_total",       "value": stats.get("n_total", 0)},
        {"timestamp": now_str, "script": SCRIPT_NAME, "metric": "has_duplicates","value": stats.get("has_duplicates", False)},
        {"timestamp": now_str, "script": SCRIPT_NAME, "metric": "note",          "value": note},
    ]
    append_runtime_summary(rows)
    print(f"[INFO] runtime_summary.csv 기록 완료: {RUNTIME_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
