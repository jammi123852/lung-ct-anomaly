"""
Task 2.0 검증 스크립트: PathResolver sample_check(5) 결과를 출력한다.

실행 방법:
    python scripts/check_path_resolver.py

완료 조건:
    5건 모두 file_exists=True + 사용자 명시 승인 → PathResolver 확정
    확정 전까지 DataValidator/DataLoader bulk 실행 금지
"""

from __future__ import annotations

import sys
from pathlib import Path

# 저장소 루트를 sys.path에 추가
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import yaml

from position_aware_padim.path_resolver import PathResolver


def load_paths_config(repo_root: Path) -> dict:
    config_path = repo_root / "configs" / "paths.local.yaml"
    if not config_path.exists():
        print(f"[ERROR] configs/paths.local.yaml 없음: {config_path}")
        sys.exit(1)
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    repo_root = _REPO_ROOT
    paths_cfg = load_paths_config(repo_root)

    base_path: str = paths_cfg.get("normal_training_ready", "")
    if not base_path:
        print("[ERROR] configs/paths.local.yaml에 normal_training_ready 키가 없거나 비어있습니다.")
        sys.exit(1)

    # anchor: normal_training_ready_anchor가 있으면 override, 없으면 None → PathResolver 내부에서 Path(base_path).name 사용
    anchor: str | None = paths_cfg.get("normal_training_ready_anchor") or None

    manifest_path = Path(base_path) / "manifests" / "patient_manifest.csv"
    if not manifest_path.exists():
        print(f"[ERROR] patient_manifest.csv 없음: {manifest_path}")
        sys.exit(1)

    print(f"base_path : {base_path}")
    print(f"anchor    : {anchor if anchor else f'(자동) {Path(base_path).name}'}")
    print(f"manifest  : {manifest_path}")
    print()

    resolver = PathResolver(
        manifest_path=str(manifest_path),
        base_path=base_path,
        anchor=anchor,
    )

    results = resolver.sample_check(n=5)

    all_exist = all(exists for *_, exists in results)

    print(f"{'#':<3}  {'patient_id':<30}  {'file_exists'}")
    print("-" * 70)
    for i, (pid, original, relative, resolved, exists) in enumerate(results, 1):
        status = "✅ True" if exists else "❌ False"
        print(f"{i:<3}  {pid:<30}  {status}")
        print(f"     original : {original}")
        print(f"     relative : {relative}")
        print(f"     resolved : {resolved}")
        print()

    print("=" * 70)
    if all_exist:
        print("결과: 5건 모두 file_exists=True")
        print()
        print("PathResolver 변환 규칙을 확인하고 '변환 규칙 확정' 승인을 해주세요.")
        print("승인 전에는 DataValidator/DataLoader bulk 실행을 진행하지 않습니다.")
    else:
        failed = [r[0] for r in results if not r[4]]
        print(f"결과: file_exists=False 항목 있음 → {failed}")
        print("base_path 또는 anchor 설정을 확인하세요.")
        sys.exit(1)


if __name__ == "__main__":
    main()
