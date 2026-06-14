"""
DataLoader: patient_manifest 기반으로 CT 데이터를 로드한다.

설계 원칙:
- PathResolver 인스턴스를 생성자 인자로 받는다 (내부 생성 금지).
- {patient_id} 문자열 직접 조립 금지. 모든 경로는 PathResolver.resolve() 경유.
- patient_manifest.csv는 encoding='utf-8-sig'로 읽는다.
- source_* 컬럼은 사용하지 않는다. 7개 사용 컬럼만 활용한다.
- 메모리에 전체 데이터를 올리지 않도록 환자 단위로만 로드한다.
- 경로 해석 또는 파일 open 실패 시 error.csv에 기록하고 None을 반환한다.
- model_roi / lesion_mask 파일명은 결정 5에서 확정 예정이므로 후보 처리한다.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .path_resolver import PathResolver, USED_COLUMNS


ERROR_COLUMNS = [
    "patient_id",
    "error_type",
    "error_msg",
    "file_logical",
]


class DataLoader:
    """patient_manifest + PathResolver 기반으로 환자 단위 CT 데이터를 로드한다."""

    # model_roi 후보 파일명 목록 (v1 데이터셋)
    _MODEL_ROI_CANDIDATES: List[str] = [
        "model_roi.npy",
        "model_roi_lung_range.npy",
    ]

    # roi_0_0 후보 파일명 목록 (v2 데이터셋)
    _ROI_0_0_CANDIDATES: List[str] = [
        "roi_0_0.npy",
    ]

    # lesion_mask 후보 파일명 목록 (v1/v2 공통 — 존재하는 첫 번째를 사용)
    _LESION_MASK_CANDIDATES: List[str] = [
        "lesion_mask_roi_0_0.npy",
        "lesion_mask.npy",
        "lesion_mask_model_roi.npy",
        "lesion.npy",
    ]

    def __init__(
        self,
        manifest_path: str,
        path_resolver: PathResolver,
        error_csv_path: Optional[str] = None,
        use_mmap: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        manifest_path : str
            patient_manifest.csv 경로. encoding='utf-8-sig'로 읽는다.
        path_resolver : PathResolver
            sample_check 완료 후 확정된 PathResolver 인스턴스.
        error_csv_path : str | None
            경로 해석 또는 로드 실패를 기록할 error.csv 경로.
            None이면 에러 기록을 건너뛴다.
        use_mmap : bool, default True
            True이면 ct_hu / pure_lung / model_roi / lesion_mask npy 로드 시
            np.load(..., mmap_mode='r')을 사용하여 디스크 매핑한다.
            False이면 기존 방식(전체 RAM 적재)을 사용한다.
            mmap 배열은 read-only이므로 호출부에서 in-place 수정을 하지 않는 경우에만 True 권장.
        """
        self.path_resolver = path_resolver
        self.error_csv_path = Path(error_csv_path) if error_csv_path else None
        self.use_mmap = use_mmap
        self._manifest: Dict[str, Dict[str, str]] = {}
        self._load_manifest(manifest_path)

    # ------------------------------------------------------------------
    # mmap 적용 헬퍼
    # ------------------------------------------------------------------

    def _np_load(self, path: str) -> np.ndarray:
        """
        use_mmap=True이면 mmap_mode='r'로, False이면 기본 방식으로 npy를 로드한다.
        meta.json/patch_csv는 이 함수를 사용하지 않는다.
        """
        if self.use_mmap:
            return np.load(path, mmap_mode="r")
        return np.load(path)

    # ------------------------------------------------------------------
    # 내부 초기화
    # ------------------------------------------------------------------

    def _load_manifest(self, manifest_path: str) -> None:
        """patient_manifest.csv를 읽어 patient_id → 7컬럼 dict로 저장한다."""
        with open(manifest_path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = (row.get("patient_id") or "").strip()
                if not pid:
                    continue
                self._manifest[pid] = {
                    col: (row.get(col) or "").strip() for col in USED_COLUMNS
                }

    def _record_error(
        self,
        patient_id: str,
        error_type: str,
        error_msg: str,
        file_logical: str,
    ) -> None:
        """error.csv에 오류를 추가한다. error_csv_path가 None이면 건너뛴다."""
        if self.error_csv_path is None:
            return
        self.error_csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = (
            not self.error_csv_path.exists()
            or self.error_csv_path.stat().st_size == 0
        )
        with open(self.error_csv_path, "a", encoding="utf-8-sig", newline="") as f:
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

    # ------------------------------------------------------------------
    # 핵심 메서드
    # ------------------------------------------------------------------

    def load_patient_data(
        self,
        patient_id: str,
        mask_type: str = "pure_lung",
    ) -> Optional[Dict]:
        """
        환자 CT 데이터를 로드한다.

        Parameters
        ----------
        patient_id : str
            로드할 환자 ID.
        mask_type : str
            'pure_lung': pure_lung.npy 로드 (기본값, normal 학습용, v1 전용).
            'model_roi': model_roi.npy 로드 (v1 lesion 평가용; 파일명 후보 처리).
            'roi_0_0':   roi_0_0.npy 로드 (v2 lesion 평가용).

        Returns
        -------
        dict | None
            {
                'patient_id': str,
                'safe_id': str,
                'ct_hu': np.ndarray,              # shape (slices, H, W)
                'mask': np.ndarray,               # shape (slices, H, W)
                'meta': dict,                     # meta.json 파싱 결과
                'patch_df': pd.DataFrame,         # patch_csv 로드 결과
                'lesion_mask': np.ndarray | None, # 후보 파일 없으면 None
            }
            경로 해석 또는 파일 로드 실패 시 None 반환.
        """
        row = self._manifest.get(patient_id)
        if row is None:
            self._record_error(
                patient_id,
                "patient_not_in_manifest",
                f"patient_id '{patient_id}'가 manifest에 없습니다.",
                "",
            )
            return None

        safe_id = row.get("safe_id", "")

        # ct_hu.npy 로드 (use_mmap=True이면 mmap_mode='r')
        try:
            ct_path = self.path_resolver.resolve(patient_id, "ct_hu")
            ct_hu = self._np_load(ct_path)
        except Exception as exc:
            self._record_error(patient_id, "load_error", str(exc), "ct_hu")
            return None

        # mask 로드 (use_mmap=True이면 mmap_mode='r')
        if mask_type == "pure_lung":
            try:
                mask_path = self.path_resolver.resolve(patient_id, "pure_lung")
                mask = self._np_load(mask_path)
            except Exception as exc:
                self._record_error(patient_id, "load_error", str(exc), "pure_lung")
                return None
        elif mask_type == "model_roi":
            mask = self._load_model_roi_candidate(patient_id)
            if mask is None:
                return None
        elif mask_type == "roi_0_0":
            mask = self._load_roi_0_0_candidate(patient_id)
            if mask is None:
                return None
        else:
            self._record_error(
                patient_id,
                "unknown_mask_type",
                f"알 수 없는 mask_type: '{mask_type}'. 허용값: pure_lung, model_roi, roi_0_0",
                "mask",
            )
            return None

        # meta.json 로드
        try:
            meta_path = self.path_resolver.resolve(patient_id, "meta")
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as exc:
            self._record_error(patient_id, "load_error", str(exc), "meta")
            return None

        # patch_csv 로드
        try:
            patch_csv_path = self.path_resolver.resolve(patient_id, "patch_index")
            patch_df = pd.read_csv(patch_csv_path, encoding="utf-8-sig")
        except Exception as exc:
            self._record_error(patient_id, "load_error", str(exc), "patch_index")
            return None

        # lesion_mask: 후보 처리 (없으면 None이어도 정상 반환)
        lesion_mask = self._load_lesion_mask_candidate(patient_id)

        return {
            "patient_id": patient_id,
            "safe_id": safe_id,
            "ct_hu": ct_hu,
            "mask": mask,
            "meta": meta,
            "patch_df": patch_df,
            "lesion_mask": lesion_mask,
        }

    def load_slice(
        self,
        ct_volume: np.ndarray,
        slice_idx: int,
    ) -> np.ndarray:
        """
        3D CT 볼륨에서 단일 slice를 로드하고 1-channel → 3-channel로 변환한다.

        Parameters
        ----------
        ct_volume : np.ndarray
            shape (slices, H, W)인 3D CT 볼륨.
        slice_idx : int
            로드할 slice 인덱스.

        Returns
        -------
        np.ndarray
            shape (H, W, 3) - 1-channel을 3-channel로 복제.
        """
        slice_2d = ct_volume[slice_idx]  # (H, W)
        return np.stack([slice_2d, slice_2d, slice_2d], axis=-1)  # (H, W, 3)

    # ------------------------------------------------------------------
    # 미확정 파일명 후보 처리 (결정 5 영역)
    # ------------------------------------------------------------------

    def _load_roi_0_0_candidate(self, patient_id: str) -> Optional[np.ndarray]:
        """
        roi_0_0 후보 파일 중 존재하는 첫 번째를 로드한다 (v2 데이터셋 전용).
        존재하는 후보가 없으면 error.csv에 기록하고 None 반환.
        """
        try:
            volume_dir = self.path_resolver.resolve(patient_id, "volume_dir")
        except Exception as exc:
            self._record_error(
                patient_id, "load_error", str(exc), "roi_0_0(volume_dir)"
            )
            return None

        for candidate in self._ROI_0_0_CANDIDATES:
            candidate_path = Path(volume_dir) / candidate
            if candidate_path.exists():
                try:
                    return self._np_load(str(candidate_path))
                except Exception as exc:
                    self._record_error(
                        patient_id,
                        "load_error",
                        str(exc),
                        f"roi_0_0({candidate})",
                    )
                    return None

        self._record_error(
            patient_id,
            "roi_0_0_not_found",
            f"roi_0_0 후보 파일을 찾을 수 없습니다: {self._ROI_0_0_CANDIDATES}",
            "roi_0_0",
        )
        return None

    def _load_model_roi_candidate(self, patient_id: str) -> Optional[np.ndarray]:
        """
        model_roi 후보 파일 중 존재하는 첫 번째를 로드한다.
        존재하는 후보가 없으면 error.csv에 기록하고 None 반환.
        """
        try:
            volume_dir = self.path_resolver.resolve(patient_id, "volume_dir")
        except Exception as exc:
            self._record_error(
                patient_id, "load_error", str(exc), "model_roi(volume_dir)"
            )
            return None

        for candidate in self._MODEL_ROI_CANDIDATES:
            candidate_path = Path(volume_dir) / candidate
            if candidate_path.exists():
                try:
                    return self._np_load(str(candidate_path))
                except Exception as exc:
                    self._record_error(
                        patient_id,
                        "load_error",
                        str(exc),
                        f"model_roi({candidate})",
                    )
                    return None

        self._record_error(
            patient_id,
            "model_roi_not_found",
            f"model_roi 후보 파일을 찾을 수 없습니다: {self._MODEL_ROI_CANDIDATES}",
            "model_roi",
        )
        return None

    def _load_lesion_mask_candidate(self, patient_id: str) -> Optional[np.ndarray]:
        """
        lesion_mask 후보 파일 중 존재하는 첫 번째를 로드한다.
        정상 데이터에선 없는 것이 정상이므로 없으면 None 반환 (error 기록 없음).
        """
        try:
            volume_dir = self.path_resolver.resolve(patient_id, "volume_dir")
        except Exception:
            return None

        for candidate in self._LESION_MASK_CANDIDATES:
            candidate_path = Path(volume_dir) / candidate
            if candidate_path.exists():
                try:
                    return self._np_load(str(candidate_path))
                except Exception:
                    return None

        return None
