"""
ResNet50 scaffold 전용 PaDiM 모델 (G3).

원본 padim_model.PaDiMModel 을 수정하지 않고 상속한다.
원본의 selected_feature_indices 범위 검증이 `idx < 448`로 하드코딩되어 있어
ResNet50(raw_feature_dim=1792)에서는 사용할 수 없으므로, 이 scaffold 클래스는
`__init__`만 재정의하여 `idx < raw_feature_dim` 으로 검증한다.

그 외 update/finalize/fit/score/save/load 등 모든 메서드는 원본을 그대로 상속한다.
(원본 메서드는 self.feature_dim(=100)과 self.selected_feature_indices만 사용하므로
 raw_feature_dim 차이와 무관하게 동작한다.)

G3 결정 사항(요청서):
- 생성자는 PaDiMModel(feature_dim=100, raw_feature_dim=448) 형태.
- 기본값 raw_feature_dim=448 로 기존 ResNet18 동작을 최대한 유지.
- ResNet50 경로에서만 raw_feature_dim=1792 명시.
- 자동검출 방식은 사용하지 않는다 (명시 인자만).
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np

from .padim_model import PaDiMModel


class PaDiMModelResNet50Scaffold(PaDiMModel):
    """
    raw_feature_dim 명시 인자를 받는 PaDiM scaffold 모델.

    Parameters
    ----------
    selected_feature_indices_path : str, optional
        selected_feature_indices.npy 절대경로. None 또는 상대경로 시 ValueError.
        (이번 단계에서는 ResNet50 index 파일을 생성하지 않으므로 보통 미존재)
    feature_dim : int, default 100
        reduce 후 차원 (backbone 무관, 기존 유지).
    raw_feature_dim : int, default 448
        reduce 입력 차원. ResNet18=448, ResNet50=1792. 명시 인자(자동검출 금지).
    eps : float, default 1e-5
    """

    def __init__(
        self,
        selected_feature_indices_path: Optional[str] = None,
        feature_dim: int = 100,
        raw_feature_dim: int = 448,
        eps: float = 1e-5,
    ) -> None:
        # selected_feature_indices_path 검증 (원본과 동일 규약)
        if selected_feature_indices_path is not None:
            if not isinstance(selected_feature_indices_path, str):
                raise ValueError(
                    "selected_feature_indices_path는 str이어야 합니다. "
                    f"받은 타입: {type(selected_feature_indices_path).__name__}"
                )
            if not os.path.isabs(selected_feature_indices_path):
                raise ValueError(
                    "selected_feature_indices_path는 절대경로여야 합니다. "
                    f"받은 값: '{selected_feature_indices_path}'"
                )

        self.selected_feature_indices_path: Optional[str] = selected_feature_indices_path
        self.feature_dim: int = int(feature_dim)
        self.raw_feature_dim: int = int(raw_feature_dim)  # G3: 명시 인자
        self.eps: float = float(eps)

        # selected_feature_indices.npy 로드 시도 — 파일 없으면 None 유지
        #   범위 검증: 0 <= idx < raw_feature_dim (원본의 하드코딩 448 대체)
        self.selected_feature_indices: Optional[np.ndarray] = None
        if (
            self.selected_feature_indices_path is not None
            and os.path.exists(self.selected_feature_indices_path)
        ):
            indices = np.load(self.selected_feature_indices_path)

            if indices.ndim != 1 or indices.shape[0] != self.feature_dim:
                raise ValueError(
                    f"selected_feature_indices shape이 ({self.feature_dim},)여야 합니다. "
                    f"받은 shape: {indices.shape}"
                )
            if not np.issubdtype(indices.dtype, np.integer):
                raise ValueError(
                    f"selected_feature_indices dtype은 정수형이어야 합니다. "
                    f"받은 dtype: {indices.dtype}"
                )
            if len(np.unique(indices)) != len(indices):
                raise ValueError("selected_feature_indices에 중복 index가 있습니다.")

            # G3 핵심: 하드코딩 448 대신 raw_feature_dim 기준 범위 검증
            if (indices < 0).any() or (indices >= self.raw_feature_dim).any():
                bad = indices[(indices < 0) | (indices >= self.raw_feature_dim)]
                raise ValueError(
                    f"selected_feature_indices는 0 <= idx < {self.raw_feature_dim} "
                    f"범위여야 합니다. 위반 index 예: {bad[:5].tolist()}"
                )

            self.selected_feature_indices = indices

        # streaming 누적 자료구조 초기화 (원본과 동일)
        self.accum: Dict[str, Dict[str, np.ndarray]] = {}
        for key in self._all_keys():
            self.accum[key] = self._make_empty_accum()

        self.stats: Dict[str, Dict] = {}

        # train_summary 스키마 (원본과 동일 + raw_feature_dim 기록)
        self.train_summary: Dict = {
            "n_patients_seen": 0,
            "n_patients_success": 0,
            "n_patients_failed": 0,
            "n_patches_used": 0,
            "n_patches_skipped": 0,
            "position_bin_counts": {k: 0 for k in self._all_keys()},
            "fallback_counts": {},
            "feature_dim": self.feature_dim,
            "raw_feature_dim": self.raw_feature_dim,
            "eps": self.eps,
            "selected_feature_indices_path": self.selected_feature_indices_path,
        }

        # GPU scoring 캐시 (bin별 mean/cov_inv 텐서). score_patient 최초 호출 시 1회 빌드.
        self._score_cache: Optional[Dict[str, Dict]] = None
        self._score_device = None

    # ------------------------------------------------------------------
    # GPU 가속 scoring (원본 padim_model.score_patient 수치 동등 override)
    #   - cov_inv 는 bin당 1회만 numpy(inv/pinv)로 계산해 캐시 (patch마다 재계산 제거).
    #   - Mahalanobis 거리계산은 patch 배치를 GPU float64 텐서로 묶어 처리.
    #   - fallback 우선순위/카운트, NaN/insufficient 처리, score_summary, 결과 컬럼은
    #     원본과 동일하게 유지한다. 역행렬 자체는 원본과 같은 numpy 결과를 사용하므로
    #     score 값은 float64 수준에서 원본과 동등(GPU matmul 오차 ~1e-9).
    # ------------------------------------------------------------------
    def _build_score_cache(self, device):
        """count>=2 인 모든 stat 키에 대해 (mean, cov_inv)를 device 텐서로 1회 캐시."""
        import torch

        cache: Dict[str, Dict] = {}
        for key, s in self.stats.items():
            if not isinstance(s, dict) or s.get("count", 0) < 2:
                continue
            if "mean" not in s or "cov" not in s:
                continue
            mean = np.asarray(s["mean"], dtype=np.float64)
            cov = np.asarray(s["cov"], dtype=np.float64)
            used_pinv = False
            try:
                cov_inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                cov_inv = np.linalg.pinv(cov)
                used_pinv = True
            cache[key] = {
                "mean": torch.from_numpy(np.ascontiguousarray(mean)).to(device),
                "cov_inv": torch.from_numpy(np.ascontiguousarray(cov_inv)).to(device),
                "used_pinv": used_pinv,
            }
        return cache

    def score_patient(self, feature_extractor, patient_data):
        """
        환자 단위 스코어링 (GPU 배치 버전). 원본 PaDiMModel.score_patient 와
        동일한 반환/예외 규약을 따른다. patch_df 원본 컬럼 + "padim_score" 반환.
        """
        import gc

        import pandas as pd
        import torch

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

        # scoring device: feature_extractor 와 동일 device 사용 (없으면 cuda 우선)
        device = getattr(feature_extractor, "device", None)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self._score_cache is None or self._score_device != device:
            self._score_cache = self._build_score_cache(device)
            self._score_device = device

        pid = (
            patient_data.get("patient_id", "")
            if isinstance(patient_data, dict)
            else getattr(patient_data, "patient_id", "")
        )

        ct_hu = None
        patch_df = None
        result_df = None
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
            patch_df_r = patch_df.reset_index(drop=True)
            scores = np.full(len(patch_df_r), float("nan"))

            for z_value, group in patch_df_r.groupby("local_z"):
                z = int(z_value)
                if z < 0 or z >= ct_hu.shape[0]:
                    continue

                group_positions = np.asarray(list(group.index))  # scores 내 위치
                patch_coords = [
                    (int(r.y0), int(r.x0), int(r.y1), int(r.x1))
                    for r in group.itertuples(index=False)
                ]
                bins_arr = group["position_bin"].astype(str).to_numpy()

                slice_2d = np.asarray(ct_hu[z], dtype=np.float32)
                preprocessed = preprocess_ct_slice(slice_2d)
                features_raw = feature_extractor.extract_patch_features(
                    preprocessed, patch_coords
                )  # (M, raw_feature_dim)
                features_sel = features_raw[:, indices].astype(np.float32)  # (M, feature_dim)
                del features_raw

                # position_bin 별로 묶어 GPU 배치 scoring (fallback/카운트 원본과 동일)
                for pb in np.unique(bins_arr):
                    pb = str(pb)
                    local_idx = np.where(bins_arr == pb)[0]
                    target = group_positions[local_idx]

                    # 1) position_bin 유효성 (원본 score_patch ValueError 경로)
                    if pb not in self.POSITION_BINS:
                        _nan_error_types.extend(["ValueError"] * len(target))
                        continue

                    # 2) fallback 우선순위로 stat 조회 (insufficient → RuntimeError 경로)
                    try:
                        stat, used_key = self._get_stat_for_scoring(pb)
                    except RuntimeError:
                        _nan_error_types.extend(["RuntimeError"] * len(target))
                        continue
                    if used_key != pb:
                        fb_key = f"{pb}->{used_key}"
                        self.train_summary["fallback_counts"][fb_key] = (
                            self.train_summary["fallback_counts"].get(fb_key, 0) + len(target)
                        )

                    entry = self._score_cache.get(used_key)
                    if entry is None:
                        _nan_error_types.extend(["RuntimeError"] * len(target))
                        continue
                    if entry["used_pinv"]:
                        self.train_summary["fallback_counts"]["pinv_used"] = (
                            self.train_summary["fallback_counts"].get("pinv_used", 0) + len(target)
                        )

                    Xb = features_sel[local_idx]  # (m, feature_dim) float32
                    finite_mask = np.isfinite(Xb).all(axis=1)
                    # NaN/inf feature → 원본 ValueError 경로 → score NaN
                    n_bad = int((~finite_mask).sum())
                    if n_bad:
                        _nan_error_types.extend(["ValueError"] * n_bad)
                    if not finite_mask.any():
                        continue

                    Xt = torch.from_numpy(
                        np.ascontiguousarray(Xb[finite_mask].astype(np.float64))
                    ).to(device)
                    diff = Xt - entry["mean"]
                    dist_sq = (diff @ entry["cov_inv"] * diff).sum(dim=1)
                    dist = torch.sqrt(torch.clamp(dist_sq, min=0.0))
                    scores[target[finite_mask]] = dist.detach().cpu().numpy()

                del slice_2d, preprocessed, features_sel

            n_total = len(scores)
            n_nan = int(np.isnan(scores).sum())
            n_scored = n_total - n_nan

            score_summary = {
                "patient_id": pid,
                "n_patches_total": n_total,
                "n_patches_scored": n_scored,
                "n_patches_nan": n_nan,
                "nan_error_types": _nan_error_types,
            }
            self.train_summary.setdefault("score_summaries", []).append(score_summary)

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
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        return result_df
