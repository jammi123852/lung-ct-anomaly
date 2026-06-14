"""
HUStatBaseline: position_bin별 HU 통계(mean/std) 기반 baseline 모델.

설계 원칙:
- 전체 데이터를 메모리에 올리지 않는다 (환자별 스트리밍).
- mask 내부(mask==1) 픽셀만 사용한다.
- streaming 방식으로 sum/sum_sq/count를 누적하여 mean/std를 계산한다.
- std가 0이면 1.0으로 대체한다 (division by zero 방지).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Generator, Iterator, Optional

import numpy as np
import pandas as pd


class HUStatBaseline:
    """position_bin별 HU mean/std 기반 baseline 모델."""

    # patch 내 mask 내부 픽셀이 이 값 미만이면 skip
    MIN_MASK_PIXELS: int = 1

    def __init__(self) -> None:
        # {bin: {'mean': float, 'std': float, 'count': int}}
        self.stats: Dict[str, Dict[str, float]] = {}
        # train 후 집계 요약 (train_hu_stat.py에서 runtime/error 기록에 활용)
        self.train_summary: Dict[str, int] = {
            "n_patches_total": 0,
            "n_patches_skipped_mask": 0,
        }

    # ------------------------------------------------------------------
    # 학습
    # ------------------------------------------------------------------

    def train(self, patient_data_iter: Iterator[Dict]) -> None:
        """
        환자별 data dict를 스트리밍하며 position_bin별 mean/std를 계산한다.

        Parameters
        ----------
        patient_data_iter : Iterator[dict]
            DataLoader.load_patient_data() 반환값의 iterator.
            각 dict: {
                'ct_hu':    np.ndarray (slices, H, W),
                'mask':     np.ndarray (slices, H, W),  # pure_lung mask
                'patch_df': pd.DataFrame,
                ...
            }
        """
        # {bin: {'sum': float, 'sum_sq': float, 'count': int}}
        # count = patch 개수 (pixel 개수 아님)
        accum: Dict[str, Dict[str, float]] = {}

        self.train_summary = {"n_patches_total": 0, "n_patches_skipped_mask": 0}

        for patient_data in patient_data_iter:
            ct_hu: np.ndarray = patient_data["ct_hu"]
            mask: np.ndarray = patient_data["mask"]
            patch_df: pd.DataFrame = patient_data["patch_df"]

            required_cols = {"position_bin", "local_z", "y0", "x0", "y1", "x1"}
            missing = required_cols - set(patch_df.columns)
            if missing:
                continue

            for _, row in patch_df.iterrows():
                self.train_summary["n_patches_total"] += 1

                pos_bin = str(row["position_bin"])
                z = int(row["local_z"])
                y0 = int(row["y0"])
                x0 = int(row["x0"])
                y1 = int(row["y1"])
                x1 = int(row["x1"])

                if z < 0 or z >= ct_hu.shape[0]:
                    self.train_summary["n_patches_skipped_mask"] += 1
                    continue

                patch_hu = ct_hu[z, y0:y1, x0:x1]       # (pH, pW)
                patch_mask = mask[z, y0:y1, x0:x1]       # (pH, pW)

                inside = patch_hu[patch_mask == 1]
                if inside.size < self.MIN_MASK_PIXELS:
                    self.train_summary["n_patches_skipped_mask"] += 1
                    continue

                if pos_bin not in accum:
                    accum[pos_bin] = {"sum": 0.0, "sum_sq": 0.0, "count": 0}

                # patch-level 누적: 이 patch의 mask 내부 픽셀 평균 HU를 단일 값으로 사용
                patch_mean_hu = float(inside.mean())
                accum[pos_bin]["sum"] += patch_mean_hu
                accum[pos_bin]["sum_sq"] += patch_mean_hu ** 2
                accum[pos_bin]["count"] += 1  # patch 개수

        # mean/std 계산
        self.stats = {}
        for pos_bin, a in accum.items():
            n = a["count"]
            if n == 0:
                self.stats[pos_bin] = {"mean": 0.0, "std": 1.0, "count": 0}
                continue
            mean = a["sum"] / n
            var = a["sum_sq"] / n - mean ** 2
            var = max(var, 0.0)
            std = math.sqrt(var)
            if std == 0.0:
                std = 1.0
            self.stats[pos_bin] = {"mean": mean, "std": std, "count": n}

    # ------------------------------------------------------------------
    # 저장 / 로드
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """position_bin_stats.npz로 저장한다. (bins, means, stds, counts)"""
        if not self.stats:
            raise RuntimeError("학습 결과가 없습니다. train()을 먼저 호출하세요.")

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        bins = np.array(list(self.stats.keys()), dtype=object)
        means = np.array([self.stats[b]["mean"] for b in bins], dtype=np.float64)
        stds = np.array([self.stats[b]["std"] for b in bins], dtype=np.float64)
        counts = np.array([self.stats[b]["count"] for b in bins], dtype=np.int64)

        np.savez(str(out), bins=bins, means=means, stds=stds, counts=counts)

    def load(self, path: str) -> None:
        """npz에서 로드하여 self.stats를 복원한다."""
        data = np.load(path, allow_pickle=True)
        bins = data["bins"]
        means = data["means"]
        stds = data["stds"]
        counts = data["counts"]

        self.stats = {}
        for b, m, s, c in zip(bins, means, stds, counts):
            self.stats[str(b)] = {
                "mean": float(m),
                "std": float(s),
                "count": int(c),
            }

    # ------------------------------------------------------------------
    # 스코어링
    # ------------------------------------------------------------------

    def score_patch(
        self,
        hu_values: np.ndarray,
        position_bin: str,
    ) -> float:
        """
        patch의 mask 내부 HU값에 대한 z-score를 반환한다.

        Parameters
        ----------
        hu_values : np.ndarray
            mask 내부 픽셀 HU값 배열.
        position_bin : str
            position_bin 문자열.

        Returns
        -------
        float
            z_score = |mean(hu_values) - bin_mean| / bin_std.
            bin이 없거나 hu_values가 비어있으면 nan 반환.
        """
        if hu_values.size == 0:
            return float("nan")

        stat = self.stats.get(position_bin)
        if stat is None:
            return float("nan")

        patch_mean = float(hu_values.mean())
        z_score = abs(patch_mean - stat["mean"]) / stat["std"]
        return z_score

    def score_patient(self, patient_data: Dict) -> pd.DataFrame:
        """
        환자의 모든 patch에 z-score를 계산한다.

        Parameters
        ----------
        patient_data : dict
            DataLoader.load_patient_data() 반환값.

        Returns
        -------
        pd.DataFrame
            patch_df 컬럼 + 'hu_z_score' 컬럼 추가.
        """
        ct_hu: np.ndarray = patient_data["ct_hu"]
        mask: np.ndarray = patient_data["mask"]
        patch_df: pd.DataFrame = patient_data["patch_df"].copy()

        scores = []
        for _, row in patch_df.iterrows():
            pos_bin = str(row.get("position_bin", ""))
            z = int(row.get("local_z", -1))
            y0 = int(row.get("y0", 0))
            x0 = int(row.get("x0", 0))
            y1 = int(row.get("y1", 0))
            x1 = int(row.get("x1", 0))

            if z < 0 or z >= ct_hu.shape[0]:
                scores.append(float("nan"))
                continue

            patch_hu = ct_hu[z, y0:y1, x0:x1]
            patch_mask = mask[z, y0:y1, x0:x1]
            inside = patch_hu[patch_mask == 1]

            scores.append(self.score_patch(inside, pos_bin))

        patch_df["hu_z_score"] = scores
        return patch_df
