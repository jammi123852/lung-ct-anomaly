"""
PathResolver: patient_manifest 기반 경로 매핑 + base join 방식으로 실제 경로를 해석한다.

설계 원칙:
- 사용 컬럼 8개: patient_id, safe_id, volume_dir, ct_hu_npy, pure_lung_npy, roi_0_0_npy, meta_json, patch_csv
  - pure_lung_npy: v1(model_roi 기반) 데이터셋 전용. v2에서는 컬럼 없음 → resolve 시 ValueError.
  - roi_0_0_npy: v2(roi_0_0 기반) 데이터셋 전용. v1에서는 컬럼 없음 → resolve 시 ValueError.
- source_* 컬럼은 hardlink 원본 추적용이므로 경로 해석에 사용하지 않는다.
- E:\\... 드라이브 직매핑은 기본 전략이 아니다.
- anchor는 하드코딩하지 않고 Path(base_path).name을 기본값으로 사용한다.
  configs/paths.local.yaml의 normal_training_ready_anchor 키가 있으면 그 값으로 override한다.
- DataLoader/DataValidator bulk 실행 전 sample_check(5)로 사용자 확인을 받아야 한다.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple


USED_COLUMNS: List[str] = [
    "patient_id",
    "safe_id",
    "volume_dir",
    "ct_hu_npy",
    "pure_lung_npy",
    "roi_0_0_npy",
    "meta_json",
    "patch_csv",
]

LOGICAL_TO_COLUMN: Dict[str, str] = {
    "ct_hu": "ct_hu_npy",
    "pure_lung": "pure_lung_npy",
    "roi_0_0": "roi_0_0_npy",
    "meta": "meta_json",
    "patch_index": "patch_csv",
    "volume_dir": "volume_dir",
}


class PathResolver:
    """patient_manifest + base join 전략으로 실제 파일 경로를 해석한다."""

    def __init__(
        self,
        manifest_path: str,
        base_path: str,
        anchor: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        manifest_path : str
            manifests/patient_manifest.csv 경로. encoding='utf-8-sig'로 읽는다.
        base_path : str
            configs/paths.local.yaml의 normal_training_ready 값.
        anchor : str | None
            manifest 경로에서 상대경로 추출 기준 디렉터리명.
            None이면 Path(base_path).name을 자동 사용.
            configs/paths.local.yaml의 normal_training_ready_anchor 키가 있으면
            호출 측에서 그 값을 전달한다. 코드 내부에 anchor 문자열을 하드코딩하지 않는다.
        """
        self.base_path = Path(base_path)
        self.anchor: str = anchor if anchor else self.base_path.name
        self._manifest: Dict[str, Dict[str, str]] = {}
        self._load_manifest(manifest_path)

    # ------------------------------------------------------------------
    # 내부 초기화
    # ------------------------------------------------------------------

    def _load_manifest(self, manifest_path: str) -> None:
        """patient_manifest.csv를 읽어 patient_id → 7컬럼 dict로 저장한다."""
        with open(manifest_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = (row.get("patient_id") or "").strip()
                if not pid:
                    continue
                self._manifest[pid] = {col: (row.get(col) or "").strip() for col in USED_COLUMNS}

    # ------------------------------------------------------------------
    # 핵심 메서드
    # ------------------------------------------------------------------

    def resolve(self, patient_id: str, file_logical: str) -> str:
        """
        patient_id + logical key → 실제 절대경로(str).

        file_logical 값:
            'ct_hu'       → ct_hu_npy 컬럼
            'pure_lung'   → pure_lung_npy 컬럼
            'meta'        → meta_json 컬럼
            'patch_index' → patch_csv 컬럼
            'volume_dir'  → volume_dir 컬럼

        실패 시 ValueError/KeyError를 발생시킨다. 조용히 넘어가지 않는다.
        """
        row = self._manifest.get(patient_id)
        if row is None:
            raise KeyError(f"patient_id '{patient_id}'가 manifest에 없습니다.")

        col = LOGICAL_TO_COLUMN.get(file_logical)
        if col is None:
            raise ValueError(
                f"알 수 없는 file_logical: '{file_logical}'. "
                f"허용값: {list(LOGICAL_TO_COLUMN.keys())}"
            )

        original = row.get(col, "")
        if not original:
            raise ValueError(
                f"manifest의 '{col}' 컬럼이 비어있습니다 (patient_id={patient_id})"
            )

        relative = self.extract_relative(original)
        resolved = self.normalize_path(str(self.base_path / relative))
        return resolved

    def extract_relative(self, manifest_path_str: str) -> str:
        """
        manifest 경로에서 anchor 이후 상대경로를 추출한다.

        anchor는 기본적으로 Path(base_path).name이며,
        configs/paths.local.yaml의 normal_training_ready_anchor 키가 있으면 그 값으로 override한다.
        anchor 문자열을 코드에 직접 하드코딩하지 않는다.

        예)
            manifest_path_str = 'E:\\...\\Normal_LUNA16_...\\volumes_npy\\p001\\ct_hu.npy'
            anchor            = 'Normal_LUNA16_...'
            → 'volumes_npy/p001/ct_hu.npy'
        """
        normalized = manifest_path_str.replace("\\", "/")
        idx = normalized.find(self.anchor)
        if idx == -1:
            raise ValueError(
                f"anchor '{self.anchor}'를 경로에서 찾을 수 없습니다: {manifest_path_str!r}\n"
                f"configs/paths.local.yaml의 normal_training_ready 또는 "
                f"normal_training_ready_anchor를 확인하세요."
            )
        relative = normalized[idx + len(self.anchor):]
        return relative.lstrip("/")

    def normalize_path(self, path: str) -> str:
        """base join 산출물의 경로 구분자와 중복 구분자를 정리한다."""
        return str(Path(path))

    def is_windows_path(self, p: str) -> bool:
        """드라이브 문자로 시작하는 Windows 절대경로 여부."""
        return len(p) >= 2 and p[1] == ":"

    # ------------------------------------------------------------------
    # 샘플 검증 (bulk 실행 전 필수)
    # ------------------------------------------------------------------

    def sample_check(self, n: int = 5) -> List[Tuple[str, str, str, str, bool]]:
        """
        manifest에서 n건 샘플을 뽑아 base join 변환 결과를 반환한다.

        Returns
        -------
        list of (patient_id, original, relative, resolved, file_exists)

        Confirmation 조건:
            모든 n건 file_exists=True 이고 사용자가 명시적으로 승인해야 확정.
            이 조건 전까지 DataValidator/DataLoader bulk 실행 금지.
        """
        results: List[Tuple[str, str, str, str, bool]] = []
        for pid, row in list(self._manifest.items())[:n]:
            original = row.get("ct_hu_npy", "")
            try:
                relative = self.extract_relative(original)
                resolved = self.normalize_path(str(self.base_path / relative))
                exists = Path(resolved).exists()
            except Exception as exc:
                relative = f"ERROR: {exc}"
                resolved = ""
                exists = False
            results.append((pid, original, relative, resolved, exists))
        return results
