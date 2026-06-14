"""
validate_data.py: Training-ready 데이터 구조 검증 스크립트.

사용법:
    python scripts/validate_data.py --limit 5
    python scripts/validate_data.py          # 전체 (사용자 승인 후 실행)

출력:
    outputs/position-aware-padim-v1/reports/data_validation_summary.csv
    outputs/position-aware-padim-v1/reports/error.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# repo_root 기준으로 src 경로 추가
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from position_aware_padim.config_manager import ConfigManager
from position_aware_padim.path_resolver import PathResolver
from position_aware_padim.data_validator import DataValidator, ERROR_COLUMNS
from position_aware_padim.csv_validator import CSVValidator


REPORTS_DIR = _REPO_ROOT / "outputs" / "position-aware-padim-v1" / "reports"
SUMMARY_CSV = REPORTS_DIR / "data_validation_summary.csv"
ERROR_CSV = REPORTS_DIR / "error.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training-ready 데이터 검증")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="검증할 최대 환자 수",
    )
    # --full-run: Phase 10 또는 사용자 명시 승인 후에만 사용한다
    parser.add_argument(
        "--full-run",
        action="store_true",
        default=False,
        help="전체 환자 검증 (Phase 10 또는 사용자 명시 승인 후에만 사용)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 안전 가드: --limit 또는 --full-run 중 하나를 반드시 명시해야 한다
    if args.limit is None and not args.full_run:
        print(
            "[ERROR] 안전을 위해 --limit N 또는 --full-run 중 하나를 명시해야 합니다.\n"
            "예: python scripts/validate_data.py --limit 5\n"
            "     python scripts/validate_data.py --full-run"
        )
        sys.exit(1)

    # 설정 로드
    cfg_manager = ConfigManager(repo_root=str(_REPO_ROOT))
    cfg_manager.load_config()

    base_path: str = cfg_manager.get("paths", "normal_training_ready", "")
    if not base_path:
        print("[ERROR] configs/paths.local.yaml에 normal_training_ready 경로가 없습니다.")
        sys.exit(1)

    anchor: str | None = cfg_manager.get("paths", "normal_training_ready_anchor", None)
    manifest_path = str(Path(base_path) / "manifests" / "patient_manifest.csv")

    # PathResolver 생성
    resolver = PathResolver(
        manifest_path=manifest_path,
        base_path=base_path,
        anchor=anchor,
    )

    # DataValidator 실행
    print(f"[INFO] DataValidator 실행 (limit={args.limit})")
    validator = DataValidator(path_resolver=resolver)
    summary_df, error_df = validator.validate_structure(limit=args.limit)

    # CSVValidator: 각 환자 patch CSV 검증
    print("[INFO] CSVValidator 실행 중...")
    csv_validator = CSVValidator()
    csv_error_rows = []
    csv_fail_count = 0

    for _, row in summary_df.iterrows():
        pid = row["patient_id"]
        try:
            patch_csv_path = resolver.resolve(pid, "patch_index")
        except (KeyError, ValueError) as exc:
            csv_error_rows.append(
                {
                    "patient_id": pid,
                    "error_type": "patch_csv_path_error",
                    "error_msg": str(exc),
                    "file_logical": "patch_index",
                }
            )
            csv_fail_count += 1
            continue

        result = csv_validator.validate_patch_csv(patch_csv_path)
        if not result["is_valid"]:
            csv_fail_count += 1
            detail_parts = []
            if result["missing_columns"]:
                detail_parts.append(f"missing_cols={result['missing_columns']}")
            if result["coord_errors"]:
                detail_parts.append(f"coord_errors={result['coord_errors']}")
            if result["invalid_position_bins"]:
                detail_parts.append(f"invalid_bins={result['invalid_position_bins']}")
            csv_error_rows.append(
                {
                    "patient_id": pid,
                    "error_type": "csv_validation_failed",
                    "error_msg": "; ".join(detail_parts),
                    "file_logical": "patch_index",
                }
            )

    # CSV 오류를 error_df에 병합
    if csv_error_rows:
        csv_err_df = pd.DataFrame(csv_error_rows, columns=ERROR_COLUMNS)
        error_df = pd.concat([error_df, csv_err_df], ignore_index=True)

    # reports 폴더 생성 (없을 경우 대비)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 결과 저장
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    error_df.to_csv(ERROR_CSV, index=False, encoding="utf-8-sig")

    # 통계 출력
    total = len(summary_df)
    ct_ok = int(summary_df["ct_hu_exists"].sum())
    lung_ok = int(summary_df["pure_lung_exists"].sum())
    meta_ok = int(summary_df["meta_exists"].sum())
    patch_ok = int(summary_df["patch_csv_exists"].sum())
    shape_ok = int((summary_df["shape_match"] == True).sum())
    shape_mismatch = int((summary_df["shape_match"] == False).sum())

    print()
    print("=" * 50)
    print(f"검증된 환자 수       : {total}")
    print(f"ct_hu.npy 존재       : {ct_ok} / {total}")
    print(f"pure_lung.npy 존재   : {lung_ok} / {total}")
    print(f"meta.json 존재       : {meta_ok} / {total}")
    print(f"patch_csv 존재       : {patch_ok} / {total}")
    print(f"shape 일치           : {shape_ok} / {total}")
    print(f"shape 불일치         : {shape_mismatch}")
    print(f"CSV 검증 실패        : {csv_fail_count}")
    print(f"error.csv 행 수      : {len(error_df)}")
    print(f"summary 저장 위치    : {SUMMARY_CSV}")
    print(f"error 저장 위치      : {ERROR_CSV}")
    print("=" * 50)


if __name__ == "__main__":
    main()
