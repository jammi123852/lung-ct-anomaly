"""
PaDiMModel: Position-Aware PaDiM 모델의 기본 구조 (Task 6.1).

설계 원칙 (design.md Section 7 "Training Memory Safety Rules" 강제 규칙):
- patch feature를 list나 array에 append하여 RAM에 쌓는 구조 금지.
- position_bin / z_level / global 분포 모두 streaming 누적만 허용.
  누적 단위: (sum_vec, sum_outer, count).
- mean = sum_vec / count
- cov = sum_outer / count - np.outer(mean, mean) + eps * I
- selected_feature_indices.npy는 REPO_ROOT 기준 절대경로로만 전달.
  None 또는 상대경로 입력은 ValueError로 차단.

이 파일은 Task 6.1 "기본 구조"에 해당한다.
- update / finalize: streaming 누적 보조 메서드 (Task 6.1에서 구현, 본 클래스 진입점)
- train / score_patch / score_patient / save / load: 시그니처만 두고
  NotImplementedError 발생 (Task 6.2 / Task 6.3에서 구현)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np


class PaDiMModel:
    """
    Position-Aware PaDiM 분포 모델 (기본 구조 — Task 6.1).

    Parameters
    ----------
    selected_feature_indices_path : str
        selected_feature_indices.npy 절대경로. None 또는 상대경로 입력 시 ValueError.
        파일이 실제 존재하지 않아도 일단 None 허용 (Task 6.2 train 시점에 강제).
    feature_dim : int, default 100
        reduced patch feature 차원.
    eps : float, default 1e-5
        covariance diagonal 안정화 상수.
    """

    # ------------------------------------------------------------------
    # position_bin / z_level / global 키 정의
    # ------------------------------------------------------------------

    # 6개 position_bin (hu_stat_baseline.py와 동일 규약)
    POSITION_BINS: Tuple[str, ...] = (
        "upper_central",
        "upper_peripheral",
        "middle_central",
        "middle_peripheral",
        "lower_central",
        "lower_peripheral",
    )

    # 3개 z_level 통합 분포 키 (fallback level 1 — Task 6.3에서 사용)
    Z_LEVEL_KEYS: Tuple[str, ...] = (
        "upper_all",
        "middle_all",
        "lower_all",
    )

    # global pure_lung 분포 키 (fallback level 2 — Task 6.3에서 사용)
    GLOBAL_KEY: str = "global_pure_lung"

    # ------------------------------------------------------------------
    # 초기화
    # ------------------------------------------------------------------

    def __init__(
        self,
        selected_feature_indices_path: Optional[str] = None,
        feature_dim: int = 100,
        eps: float = 1e-5,
    ) -> None:
        # selected_feature_indices_path 검증 — 절대경로만 허용 (None 허용은 init 시점만)
        if selected_feature_indices_path is not None:
            if not isinstance(selected_feature_indices_path, str):
                raise ValueError(
                    "selected_feature_indices_path는 str이어야 합니다. "
                    f"받은 타입: {type(selected_feature_indices_path).__name__}"
                )
            if not os.path.isabs(selected_feature_indices_path):
                raise ValueError(
                    "selected_feature_indices_path는 절대경로여야 합니다. "
                    "상대경로는 허용되지 않습니다. "
                    f"받은 값: '{selected_feature_indices_path}'"
                )

        self.selected_feature_indices_path: Optional[str] = selected_feature_indices_path
        self.feature_dim: int = int(feature_dim)
        self.eps: float = float(eps)

        # selected_feature_indices.npy 로드 시도 — 파일 없으면 None 유지 (Task 6.2 train 시 강제)
        # 파일이 존재하면 다음 항목을 모두 검증한다:
        #   - shape == (feature_dim,)
        #   - dtype 정수형 (np.integer)
        #   - 중복 index 없음
        #   - 범위 0 <= idx < 448 (입력 차원 448 기준)
        self.selected_feature_indices: Optional[np.ndarray] = None
        if (
            self.selected_feature_indices_path is not None
            and os.path.exists(self.selected_feature_indices_path)
        ):
            indices = np.load(self.selected_feature_indices_path)

            # shape 검증
            if indices.ndim != 1 or indices.shape[0] != self.feature_dim:
                raise ValueError(
                    f"selected_feature_indices shape이 ({self.feature_dim},)여야 합니다. "
                    f"받은 shape: {indices.shape}"
                )

            # dtype 검증 — 정수형만 허용
            if not np.issubdtype(indices.dtype, np.integer):
                raise ValueError(
                    f"selected_feature_indices dtype은 정수형이어야 합니다. "
                    f"받은 dtype: {indices.dtype}"
                )

            # 중복 검증
            if len(np.unique(indices)) != len(indices):
                raise ValueError(
                    "selected_feature_indices에 중복 index가 있습니다."
                )

            # 범위 검증 (입력 raw feature 차원은 448)
            if (indices < 0).any() or (indices >= 448).any():
                bad = indices[(indices < 0) | (indices >= 448)]
                raise ValueError(
                    f"selected_feature_indices는 0 <= idx < 448 범위여야 합니다. "
                    f"위반 index 예: {bad[:5].tolist()}"
                )

            self.selected_feature_indices = indices

        # streaming 누적 자료구조 초기화
        # 절대로 patch feature를 list/array에 append하지 않는다.
        # accum[key] = {"sum_vec": (feature_dim,), "sum_outer": (feature_dim, feature_dim), "count": int}
        self.accum: Dict[str, Dict[str, np.ndarray]] = {}
        for key in self._all_keys():
            self.accum[key] = self._make_empty_accum()

        # finalize() 결과 — Task 6.3 score 단계에서 참조
        # stats[key] = {"mean": (feature_dim,), "cov": (feature_dim, feature_dim), "count": int}
        #              count == 0이면 {"status": "insufficient", "count": 0}
        self.stats: Dict[str, Dict] = {}

        # train_summary 스키마 — Task 6.2에서 정의
        # train 진행 중 누적 갱신되며, save 시 npz로 기록되지 않고 메모리 상에만 유지된다.
        self.train_summary: Dict = {
            "n_patients_seen": 0,
            "n_patients_success": 0,
            "n_patients_failed": 0,
            "n_patches_used": 0,
            "n_patches_skipped": 0,
            "position_bin_counts": {k: 0 for k in self._all_keys()},
            "fallback_counts": {},  # Task 6.3 score 단계에서 채움
            "feature_dim": self.feature_dim,
            "eps": self.eps,
            "selected_feature_indices_path": self.selected_feature_indices_path,
        }

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _all_keys(self) -> List[str]:
        """position_bin + z_level + global 모든 누적 키 목록."""
        keys: List[str] = list(self.POSITION_BINS)
        keys.extend(self.Z_LEVEL_KEYS)
        keys.append(self.GLOBAL_KEY)
        return keys

    def _make_empty_accum(self) -> Dict[str, np.ndarray]:
        """빈 streaming 누적 dict를 생성한다. patch feature 자체는 저장하지 않는다."""
        return {
            "sum_vec": np.zeros(self.feature_dim, dtype=np.float64),
            "sum_outer": np.zeros((self.feature_dim, self.feature_dim), dtype=np.float64),
            "count": 0,
        }

    # ------------------------------------------------------------------
    # streaming 누적 메서드 (Task 6.1 진입점)
    # ------------------------------------------------------------------

    def update(self, features: np.ndarray, position_bin: str) -> None:
        """
        position_bin에 대한 patch feature를 streaming으로 누적한다.

        Parameters
        ----------
        features : np.ndarray
            shape (M, feature_dim), dtype float. patch feature 모음.
            list/array에 보관하지 않고 즉시 sum/outer-sum/count에 반영한 뒤 폐기 가정.
        position_bin : str
            6개 POSITION_BINS 중 하나.

        Raises
        ------
        ValueError
            position_bin이 6개 키에 없을 때.
            features.shape[-1] != feature_dim 일 때.
            NaN/inf 포함 시.
        """
        if position_bin not in self.POSITION_BINS:
            raise ValueError(
                f"position_bin '{position_bin}'은 허용된 6개 키에 없습니다. "
                f"허용: {list(self.POSITION_BINS)}"
            )

        if not isinstance(features, np.ndarray):
            raise ValueError(
                f"features는 numpy ndarray여야 합니다. 받은 타입: {type(features).__name__}"
            )

        if features.ndim != 2:
            raise ValueError(
                f"features는 2D array여야 합니다. 받은 ndim: {features.ndim}, shape: {features.shape}"
            )

        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"features 마지막 차원이 feature_dim({self.feature_dim})과 달라야 합니다. "
                f"받은 shape: {features.shape}"
            )

        if not np.isfinite(features).all():
            n_nan = int(np.isnan(features).sum())
            n_inf = int(np.isinf(features).sum())
            raise ValueError(
                f"features에 NaN/inf가 포함되어 있습니다. NaN={n_nan}, inf={n_inf}"
            )

        if features.shape[0] == 0:
            return  # 빈 배열은 skip

        # streaming 누적 (전체 feature 자체를 보관하지 않는다)
        feats64 = features.astype(np.float64, copy=False)
        self.accum[position_bin]["sum_vec"] += feats64.sum(axis=0)
        self.accum[position_bin]["sum_outer"] += feats64.T @ feats64
        self.accum[position_bin]["count"] += int(feats64.shape[0])

        # z_level 통합 분포에도 동일 누적 (fallback level 1)
        z_key = self._position_bin_to_z_level(position_bin)
        self.accum[z_key]["sum_vec"] += feats64.sum(axis=0)
        self.accum[z_key]["sum_outer"] += feats64.T @ feats64
        self.accum[z_key]["count"] += int(feats64.shape[0])

        # global 분포에도 동일 누적 (fallback level 2)
        self.accum[self.GLOBAL_KEY]["sum_vec"] += feats64.sum(axis=0)
        self.accum[self.GLOBAL_KEY]["sum_outer"] += feats64.T @ feats64
        self.accum[self.GLOBAL_KEY]["count"] += int(feats64.shape[0])

    @staticmethod
    def _position_bin_to_z_level(position_bin: str) -> str:
        """position_bin 문자열에서 z_level 키를 추출한다."""
        if position_bin.startswith("upper_"):
            return "upper_all"
        if position_bin.startswith("middle_"):
            return "middle_all"
        if position_bin.startswith("lower_"):
            return "lower_all"
        raise ValueError(f"알 수 없는 position_bin prefix: '{position_bin}'")

    def finalize(self) -> None:
        """
        누적된 (sum_vec, sum_outer, count)로부터 모든 키의 mean/cov를 계산한다.

        - mean = sum_vec / count
        - cov  = sum_outer / count - outer(mean, mean) + eps * I
        - count == 0인 키는 self.stats[key] = {"status": "insufficient", "count": 0}로 표시
          (TODO: Task 6.3 fallback strategy 적용 시점에 처리)
        """
        self.stats = {}
        for key in self._all_keys():
            a = self.accum[key]
            n = int(a["count"])
            if n == 0:
                # TODO(Task 6.3): position_bin 누락/sample 부족 시 z_level → global fallback 적용
                self.stats[key] = {"status": "insufficient", "count": 0}
                continue

            mean = a["sum_vec"] / n  # shape (feature_dim,)
            cov = a["sum_outer"] / n - np.outer(mean, mean)
            cov = cov + self.eps * np.eye(self.feature_dim, dtype=np.float64)

            self.stats[key] = {
                "mean": mean.astype(np.float64),
                "cov": cov.astype(np.float64),
                "count": n,
            }

    # ------------------------------------------------------------------
    # 본 학습/스코어/저장 메서드 시그니처 — Task 6.2 / 6.3에서 구현
    # ------------------------------------------------------------------

    def train(
        self,
        feature_extractor,
        training_data,
        split,
    ) -> None:
        """
        환자별 streaming 학습.

        설계 (design.md Section 7 "Training Memory Safety Rules" 강제):
        - training_data를 환자별로 streaming 순회.
        - 각 slice별로 FeatureExtractor.extract_patch_features 호출하여 (M, 448) feature 추출.
        - selected_feature_indices로 (M, 100) reduced feature 생성.
        - position_bin별로 self.update(...) 호출 (z_level / global 동시 누적은 update 내부).
        - patch feature는 list/array에 보관하지 않는다.
        - 환자 1명 처리 후 ct_hu, mask, patch_df, feature 변수 명시적 del + gc.collect()
          + (가능하면) torch.cuda.empty_cache().
        - 모든 환자 처리 후 finalize() 호출.
        - 개별 환자 처리 중 예외 발생 시 n_patients_failed 증가하고 다음 환자 진행 (학습 중단 금지).
        - 분포 파일 저장은 save()에 위임한다.

        Parameters
        ----------
        feature_extractor : FeatureExtractor
            ResNet18 기반 feature 추출기. None 입력 시 ValueError.
        training_data : Iterable[dict]
            환자별 PatientData (dict with keys: patient_id, ct_hu, mask, patch_df, ...).
        split : PatientSplit
            split.train (set 또는 list)을 가진 객체. None이면 모든 환자 처리.
        """
        import gc
        from collections import defaultdict

        # 학습 시점에는 selected_feature_indices가 반드시 로드되어 있어야 한다
        if self.selected_feature_indices is None:
            raise RuntimeError(
                "selected_feature_indices가 로드되지 않았습니다. "
                "PaDiMModel(selected_feature_indices_path=<절대경로>)로 인스턴스를 생성하거나, "
                "load()로 분포 파일을 로드하세요."
            )
        if feature_extractor is None:
            raise ValueError("feature_extractor는 None일 수 없습니다.")

        # split.train 환자 set 구성 (없으면 모든 환자 처리)
        train_patients: Optional[set] = None
        if split is not None and hasattr(split, "train") and split.train is not None:
            train_patients = set(split.train)

        # 전처리 함수 lazy import (preprocessing 모듈이 없는 환경에서 import 자체가 실패하지 않도록)
        from .preprocessing import preprocess_ct_slice

        indices = self.selected_feature_indices  # (100,)
        required_cols = {"position_bin", "local_z", "y0", "x0", "y1", "x1"}

        for patient_data in training_data:
            self.train_summary["n_patients_seen"] += 1

            ct_hu = None
            mask = None
            patch_df = None
            # pid를 try 밖에서 초기화 — except 블록에서 실패 사유 기록 시 참조 가능
            pid = (
                patient_data.get("patient_id", "")
                if isinstance(patient_data, dict)
                else getattr(patient_data, "patient_id", "")
            )
            try:
                # split 필터링
                if train_patients is not None and pid not in train_patients:
                    # split 외 환자는 success/failed 어느쪽도 아님 — 단순 skip
                    continue

                # dict와 dataclass/object 입력 모두 지원
                # 필수 필드 존재 여부를 먼저 확인 — 누락 시 명확한 ValueError 발생
                if isinstance(patient_data, dict):
                    missing_fields = [
                        k for k in ("ct_hu", "mask", "patch_df") if k not in patient_data
                    ]
                    if missing_fields:
                        raise ValueError(
                            f"patient_data(dict)에 필수 필드가 없습니다. "
                            f"patient_id='{pid}', missing={missing_fields}"
                        )
                    ct_hu = patient_data["ct_hu"]
                    mask = patient_data["mask"]
                    patch_df = patient_data["patch_df"]
                else:
                    missing_fields = [
                        k for k in ("ct_hu", "mask", "patch_df")
                        if not hasattr(patient_data, k)
                    ]
                    if missing_fields:
                        raise ValueError(
                            f"patient_data(object)에 필수 속성이 없습니다. "
                            f"patient_id='{pid}', missing={missing_fields}"
                        )
                    ct_hu = getattr(patient_data, "ct_hu")
                    mask = getattr(patient_data, "mask")
                    patch_df = getattr(patient_data, "patch_df")

                missing = required_cols - set(patch_df.columns)
                if missing:
                    raise ValueError(
                        f"patch_df에 필수 컬럼이 없습니다. "
                        f"patient_id='{pid}', missing={sorted(missing)}"
                    )

                # slice 단위 batch 처리 (slice별로 그룹화)
                for z_value, group in patch_df.groupby("local_z"):
                    z = int(z_value)
                    if z < 0 or z >= ct_hu.shape[0]:
                        self.train_summary["n_patches_skipped"] += len(group)
                        continue

                    # patch_coords / position_bins 구성
                    patch_coords = [
                        (int(r.y0), int(r.x0), int(r.y1), int(r.x1))
                        for r in group.itertuples(index=False)
                    ]
                    position_bins = [
                        str(r.position_bin)
                        for r in group.itertuples(index=False)
                    ]

                    # slice 전처리 → (3, H, W)
                    slice_2d = np.asarray(ct_hu[z], dtype=np.float32)  # mmap이면 사본 생성 + int16→float32 변환
                    preprocessed = preprocess_ct_slice(slice_2d)

                    # 448차원 feature → 100차원 reduce
                    features_448 = feature_extractor.extract_patch_features(
                        preprocessed, patch_coords
                    )  # (M, 448)
                    features_100 = features_448[:, indices].astype(np.float32)  # (M, 100)

                    # position_bin별 그룹화하여 update 호출 (효율)
                    bin_to_indices: Dict[str, List[int]] = defaultdict(list)
                    for i, pb in enumerate(position_bins):
                        bin_to_indices[pb].append(i)

                    for pb, idx_list in bin_to_indices.items():
                        if pb not in self.POSITION_BINS:
                            self.train_summary["n_patches_skipped"] += len(idx_list)
                            continue
                        sub_features = features_100[idx_list]
                        if sub_features.shape[0] == 0:
                            continue
                        self.update(sub_features, pb)
                        m = sub_features.shape[0]
                        self.train_summary["n_patches_used"] += m
                        # position_bin_counts: pb/z_level/global 3중 누적 (의도된 동작)
                        # 1개 patch가 pb·z_level·global 3개 키에 동시 반영 → n_patches_used와 1:1 비교 불가
                        self.train_summary["position_bin_counts"][pb] += m
                        z_key = self._position_bin_to_z_level(pb)
                        self.train_summary["position_bin_counts"][z_key] += m
                        self.train_summary["position_bin_counts"][self.GLOBAL_KEY] += m

                    # slice 단위 임시 변수 해제
                    del slice_2d, preprocessed, features_448, features_100

                self.train_summary["n_patients_success"] += 1

            except Exception as e:
                # 학습 중단 금지 — 다음 환자로 진행
                self.train_summary["n_patients_failed"] += 1
                # 실패 사유 보관 (error.csv 직접 기록은 Task 6.4 train_padim.py에서 처리)
                self.train_summary.setdefault("patient_failures", []).append({
                    "patient_id": pid,
                    "error_type": type(e).__name__,
                    "error_msg": str(e),
                })
            finally:
                # 환자 1명 처리 후 명시적 해제
                if ct_hu is not None:
                    del ct_hu
                if mask is not None:
                    del mask
                if patch_df is not None:
                    del patch_df
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                except Exception:
                    pass

        # 모든 환자 처리 후 분포 finalize
        self.finalize()

    def _get_stat_for_scoring(self, position_bin: str):
        """
        position_bin 기반 fallback 우선순위로 분포 stat을 반환한다.

        우선순위:
            1) position_bin (count >= 2)
            2) 같은 z_level 통합 분포 (count >= 2)
            3) global_pure_lung (count >= 2)

        Returns
        -------
        (stat, used_key) : (dict, str)

        Raises
        ------
        RuntimeError
            모든 fallback 분포가 insufficient(count < 2)인 경우.
        """
        # 1순위: position_bin
        s = self.stats.get(position_bin, {})
        if s.get("count", 0) >= 2:
            return s, position_bin

        # 2순위: z_level 통합 분포
        z_key = self._position_bin_to_z_level(position_bin)
        s = self.stats.get(z_key, {})
        if s.get("count", 0) >= 2:
            return s, z_key

        # 3순위: global_pure_lung
        s = self.stats.get(self.GLOBAL_KEY, {})
        if s.get("count", 0) >= 2:
            return s, self.GLOBAL_KEY

        raise RuntimeError(
            f"position_bin='{position_bin}'에 대해 충분한 분포 데이터가 없습니다. "
            f"position_bin / z_level({z_key}) / {self.GLOBAL_KEY} 모두 insufficient (count < 2)."
        )

    def score_patch(
        self,
        patch_feature: np.ndarray,
        position_bin: str,
    ) -> float:
        """
        Mahalanobis distance 기반 patch 스코어.

        Parameters
        ----------
        patch_feature : np.ndarray
            shape (feature_dim,). 100차원 reduced feature.
        position_bin : str
            6개 POSITION_BINS 중 하나.

        Returns
        -------
        float
            Mahalanobis distance. 0 이상.

        Raises
        ------
        RuntimeError
            stats가 비어 있을 때, 또는 모든 fallback이 insufficient일 때.
        ValueError
            position_bin 범위 오류, feature shape/NaN 오류.
        """
        if not self.stats:
            raise RuntimeError(
                "stats가 비어 있습니다. finalize() 또는 load()를 먼저 호출하세요."
            )
        if position_bin not in self.POSITION_BINS:
            raise ValueError(
                f"position_bin '{position_bin}'은 허용된 6개 키에 없습니다. "
                f"허용: {list(self.POSITION_BINS)}"
            )
        if not isinstance(patch_feature, np.ndarray):
            raise ValueError(
                f"patch_feature는 numpy ndarray여야 합니다. "
                f"받은 타입: {type(patch_feature).__name__}"
            )
        if patch_feature.shape != (self.feature_dim,):
            raise ValueError(
                f"patch_feature shape이 ({self.feature_dim},)여야 합니다. "
                f"받은 shape: {patch_feature.shape}"
            )
        if not np.isfinite(patch_feature).all():
            raise ValueError("patch_feature에 NaN/inf가 포함되어 있습니다.")

        # fallback 우선순위로 분포 조회
        stat, used_key = self._get_stat_for_scoring(position_bin)

        # fallback 사용 시 train_summary["fallback_counts"]에 기록
        if used_key != position_bin:
            fb_key = f"{position_bin}->{used_key}"
            self.train_summary["fallback_counts"][fb_key] = (
                self.train_summary["fallback_counts"].get(fb_key, 0) + 1
            )

        # Mahalanobis distance 계산 (numpy 기반)
        mean = stat["mean"]   # (feature_dim,)
        cov = stat["cov"]     # (feature_dim, feature_dim)
        diff = patch_feature.astype(np.float64) - mean

        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            # singular matrix → pseudo-inverse fallback
            cov_inv = np.linalg.pinv(cov)
            self.train_summary["fallback_counts"]["pinv_used"] = (
                self.train_summary["fallback_counts"].get("pinv_used", 0) + 1
            )

        dist_sq = float(diff @ cov_inv @ diff)
        return float(np.sqrt(max(0.0, dist_sq)))

    def score_patient(
        self,
        feature_extractor,
        patient_data,
    ):
        """
        환자 단위 스코어링.

        patch_df 원본 컬럼을 모두 보존하고 padim_score 컬럼을 추가하여 반환한다.
        z 범위 밖이거나 score 실패 시 padim_score = NaN.
        patch feature를 list/array에 누적하지 않는다. slice 단위 처리 후 즉시 폐기.

        Parameters
        ----------
        feature_extractor : FeatureExtractor
            ResNet18 기반 feature 추출기. None 입력 시 ValueError.
        patient_data : dict 또는 dataclass/object
            dict: {"patient_id", "ct_hu", "mask", "patch_df", ...}
            object: .patient_id, .ct_hu, .mask, .patch_df 속성.

        Returns
        -------
        pd.DataFrame
            patch_df 원본 컬럼 + "padim_score" 컬럼.

        Raises
        ------
        RuntimeError
            stats가 비어 있거나 selected_feature_indices가 없는 경우.
            환자 내 모든 patch 스코어링이 실패(전체 NaN)인 경우.
        ValueError
            feature_extractor가 None이거나 필수 필드 누락인 경우.
        """
        import gc
        import pandas as pd

        if not self.stats:
            raise RuntimeError(
                "stats가 비어 있습니다. finalize() 또는 load()를 먼저 호출하세요."
            )
        if self.selected_feature_indices is None:
            raise RuntimeError(
                "selected_feature_indices가 없습니다. "
                "PaDiMModel(selected_feature_indices_path=<절대경로>) 또는 load()로 로드하세요."
            )
        if feature_extractor is None:
            raise ValueError("feature_extractor는 None일 수 없습니다.")

        from .preprocessing import preprocess_ct_slice

        pid = (
            patient_data.get("patient_id", "")
            if isinstance(patient_data, dict)
            else getattr(patient_data, "patient_id", "")
        )

        ct_hu = None
        patch_df = None
        result_df = None
        # 패치 스코어링 결과 요약 — NaN 개수/에러 타입 추적
        _nan_error_types: list = []
        try:
            if isinstance(patient_data, dict):
                missing_fields = [
                    k for k in ("ct_hu", "mask", "patch_df") if k not in patient_data
                ]
                if missing_fields:
                    raise ValueError(
                        f"patient_data(dict)에 필수 필드가 없습니다. "
                        f"patient_id='{pid}', missing={missing_fields}"
                    )
                ct_hu = patient_data["ct_hu"]
                patch_df = patient_data["patch_df"]
            else:
                missing_fields = [
                    k for k in ("ct_hu", "mask", "patch_df") if not hasattr(patient_data, k)
                ]
                if missing_fields:
                    raise ValueError(
                        f"patient_data(object)에 필수 속성이 없습니다. "
                        f"patient_id='{pid}', missing={missing_fields}"
                    )
                ct_hu = getattr(patient_data, "ct_hu")
                patch_df = getattr(patient_data, "patch_df")

            required_cols = {"position_bin", "local_z", "y0", "x0", "y1", "x1"}
            missing_cols = required_cols - set(patch_df.columns)
            if missing_cols:
                raise ValueError(
                    f"patch_df에 필수 컬럼이 없습니다. "
                    f"patient_id='{pid}', missing={sorted(missing_cols)}"
                )

            indices = self.selected_feature_indices  # (feature_dim,)
            # reset_index로 iloc 위치 = index 값이 되도록 확보
            patch_df_r = patch_df.reset_index(drop=True)
            scores = np.full(len(patch_df_r), float("nan"))

            for z_value, group in patch_df_r.groupby("local_z"):
                z = int(z_value)
                if z < 0 or z >= ct_hu.shape[0]:
                    continue

                group_positions = list(group.index)  # iloc 위치
                patch_coords = [
                    (int(r.y0), int(r.x0), int(r.y1), int(r.x1))
                    for r in group.itertuples(index=False)
                ]

                slice_2d = np.asarray(ct_hu[z], dtype=np.float32)  # int16→float32 변환
                preprocessed = preprocess_ct_slice(slice_2d)
                features_448 = feature_extractor.extract_patch_features(
                    preprocessed, patch_coords
                )  # (M, 448)
                features_100 = features_448[:, indices].astype(np.float32)  # (M, feature_dim)
                del features_448

                for local_i, df_pos in enumerate(group_positions):
                    pb = str(group.iloc[local_i]["position_bin"])
                    try:
                        scores[df_pos] = self.score_patch(features_100[local_i], pb)
                    except (RuntimeError, ValueError) as e:
                        scores[df_pos] = float("nan")
                        _nan_error_types.append(type(e).__name__)
                    except Exception as e:
                        scores[df_pos] = float("nan")
                        _nan_error_types.append(type(e).__name__)

                del slice_2d, preprocessed, features_100

            n_total = len(scores)
            n_nan = int(np.isnan(scores).sum())
            n_scored = n_total - n_nan

            # score_summary 기록 (train_summary에 통합)
            score_summary = {
                "patient_id": pid,
                "n_patches_total": n_total,
                "n_patches_scored": n_scored,
                "n_patches_nan": n_nan,
                "nan_error_types": _nan_error_types,
            }
            self.train_summary.setdefault("score_summaries", []).append(score_summary)

            # 전체 patch NaN → 스코어링 자체가 무의미 → RuntimeError
            if n_scored == 0 and n_total > 0:
                raise RuntimeError(
                    f"환자 '{pid}'의 모든 patch 스코어링이 실패했습니다. "
                    f"n_patches_total={n_total}, n_patches_nan={n_nan}. "
                    f"nan_error_types={list(set(_nan_error_types))}"
                )

            result_df = patch_df.copy()
            result_df["padim_score"] = scores

        finally:
            if ct_hu is not None:
                del ct_hu
            if patch_df is not None:
                del patch_df
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            except Exception:
                pass

        return result_df

    # ------------------------------------------------------------------
    # 저장 / 로드 — Task 6.2에서 구현
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        position_bin_stats.npz로 저장 (design.md Section 7 "Distribution Storage Format").

        Saved keys (10개 모든 키에 대해):
            {key}_mean   : shape (feature_dim,)
            {key}_cov    : shape (feature_dim, feature_dim)
            {key}_count  : int (0이면 insufficient)
        추가 메타:
            covariance_epsilon, feature_dim, position_bins, z_level_keys, global_key,
            selected_feature_indices (있을 때만)

        insufficient 키는 mean/cov를 zeros로 저장하고 count=0으로 마킹한다.

        Parameters
        ----------
        path : str
            절대경로. 상대경로 입력 시 ValueError.

        Raises
        ------
        RuntimeError
            finalize() 이전에 호출된 경우.
        ValueError
            path가 None / 비-str / 상대경로일 때.
        """
        if not self.stats:
            raise RuntimeError(
                "self.stats가 비어 있습니다. finalize()를 먼저 호출하세요."
            )
        if not isinstance(path, str):
            raise ValueError(
                f"path는 str이어야 합니다. 받은 타입: {type(path).__name__}"
            )
        if not os.path.isabs(path):
            raise ValueError(
                f"path는 절대경로여야 합니다. 상대경로는 허용되지 않습니다. 받은 값: '{path}'"
            )

        save_dir = os.path.dirname(path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        save_dict: Dict[str, np.ndarray] = {}
        for key in self._all_keys():
            s = self.stats.get(key, {"status": "insufficient", "count": 0})
            if s.get("status") == "insufficient" or s.get("count", 0) == 0:
                save_dict[f"{key}_mean"] = np.zeros(self.feature_dim, dtype=np.float64)
                save_dict[f"{key}_cov"] = np.zeros(
                    (self.feature_dim, self.feature_dim), dtype=np.float64
                )
                save_dict[f"{key}_count"] = np.int64(0)
            else:
                save_dict[f"{key}_mean"] = np.asarray(s["mean"], dtype=np.float64)
                save_dict[f"{key}_cov"] = np.asarray(s["cov"], dtype=np.float64)
                save_dict[f"{key}_count"] = np.int64(s["count"])

        save_dict["covariance_epsilon"] = np.float64(self.eps)
        save_dict["feature_dim"] = np.int64(self.feature_dim)
        save_dict["position_bins"] = np.array(list(self.POSITION_BINS), dtype=object)
        save_dict["z_level_keys"] = np.array(list(self.Z_LEVEL_KEYS), dtype=object)
        save_dict["global_key"] = np.array(self.GLOBAL_KEY, dtype=object)
        if self.selected_feature_indices is not None:
            save_dict["selected_feature_indices"] = np.asarray(
                self.selected_feature_indices
            )
        if self.selected_feature_indices_path is not None:
            save_dict["selected_feature_indices_path"] = np.array(
                self.selected_feature_indices_path, dtype=object
            )

        np.savez(path, **save_dict)

    def load(self, path: str) -> None:
        """
        position_bin_stats.npz에서 self.stats / selected_feature_indices / eps 복원.

        - count > 0인 키: self.stats[key] = {"mean", "cov", "count"}
        - count == 0인 키: self.stats[key] = {"status": "insufficient", "count": 0}

        Parameters
        ----------
        path : str
            절대경로. 상대경로 입력 시 ValueError.

        Raises
        ------
        ValueError
            path가 None / 비-str / 상대경로이거나 feature_dim 불일치 / shape 불일치인 경우.
        FileNotFoundError
            파일이 존재하지 않을 때.
        """
        if not isinstance(path, str):
            raise ValueError(
                f"path는 str이어야 합니다. 받은 타입: {type(path).__name__}"
            )
        if not os.path.isabs(path):
            raise ValueError(
                f"path는 절대경로여야 합니다. 상대경로는 허용되지 않습니다. 받은 값: '{path}'"
            )
        if not os.path.exists(path):
            raise FileNotFoundError(f"분포 파일을 찾을 수 없습니다: {path}")

        data = np.load(path, allow_pickle=True)
        files = set(data.files)

        # feature_dim 일치성 검증 (npz에 저장되어 있을 때만)
        if "feature_dim" in files:
            loaded_dim = int(data["feature_dim"])
            if loaded_dim != self.feature_dim:
                raise ValueError(
                    f"feature_dim 불일치: 로드된 값={loaded_dim}, 현재={self.feature_dim}"
                )

        # eps 복원
        if "covariance_epsilon" in files:
            self.eps = float(data["covariance_epsilon"])

        # stats 복원
        self.stats = {}
        for key in self._all_keys():
            count_key = f"{key}_count"
            mean_key = f"{key}_mean"
            cov_key = f"{key}_cov"

            if count_key not in files:
                # 키가 빠져 있으면 insufficient로 표시
                self.stats[key] = {"status": "insufficient", "count": 0}
                continue

            count = int(data[count_key])
            if count == 0:
                self.stats[key] = {"status": "insufficient", "count": 0}
                continue

            mean = data[mean_key]
            cov = data[cov_key]

            if mean.shape != (self.feature_dim,):
                raise ValueError(
                    f"{mean_key} shape이 ({self.feature_dim},)여야 합니다. 받은 shape: {mean.shape}"
                )
            if cov.shape != (self.feature_dim, self.feature_dim):
                raise ValueError(
                    f"{cov_key} shape이 ({self.feature_dim}, {self.feature_dim})여야 합니다. "
                    f"받은 shape: {cov.shape}"
                )

            self.stats[key] = {
                "mean": np.asarray(mean, dtype=np.float64),
                "cov": np.asarray(cov, dtype=np.float64),
                "count": count,
            }

        # selected_feature_indices 복원
        if "selected_feature_indices" in files:
            self.selected_feature_indices = np.asarray(data["selected_feature_indices"])

        # selected_feature_indices_path 복원: npz에 저장된 경우 복원, 없으면 None으로 초기화
        if "selected_feature_indices_path" in files:
            raw_path = str(data["selected_feature_indices_path"])
            self.selected_feature_indices_path = (
                raw_path if raw_path and raw_path != "None" else None
            )
        else:
            self.selected_feature_indices_path = None
        self.train_summary["selected_feature_indices_path"] = self.selected_feature_indices_path
