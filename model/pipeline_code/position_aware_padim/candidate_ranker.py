"""
CandidateRanker: patch score DataFrame을 받아 patch / slice / patient 레벨 후보를 정렬한다.

설계 원칙:
- score_col 파라미터로 padim_score / hu_z_score를 독립 처리한다.
- PaDiM과 HU Stat은 join하지 않는다. 각자 다른 인스턴스로 사용한다.
- CSV 로드는 호출부에서 encoding='utf-8-sig'로 처리한다. 이 클래스는 DataFrame만 받는다.
- 파일 저장은 이 클래스에서 수행하지 않는다.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "patient_id",
    "safe_id",
    "local_z",
    "y0",
    "x0",
    "y1",
    "x1",
    "position_bin",
]

VALID_AGG = ("max", "mean", "p95")


class CandidateRanker:
    """patch score DataFrame → patch / slice / patient 레벨 후보 정렬기."""

    def __init__(self, score_col: str) -> None:
        """
        Parameters
        ----------
        score_col : str
            정렬 기준 점수 컬럼명. 예: 'padim_score', 'hu_z_score'.
        """
        self.score_col = score_col

    # ------------------------------------------------------------------
    # 입력 검증
    # ------------------------------------------------------------------

    def _validate_df(self, df: pd.DataFrame) -> None:
        """필수 컬럼 존재와 score_col NaN/inf 여부를 검증한다."""
        required = REQUIRED_COLUMNS + [self.score_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"필수 컬럼이 없습니다: {missing}")

        col = df[self.score_col]
        if col.isna().any():
            raise ValueError(
                f"score_col '{self.score_col}'에 NaN이 포함되어 있습니다. "
                f"NaN 행 수: {col.isna().sum()}"
            )
        if np.isinf(col.to_numpy(dtype=float)).any():
            raise ValueError(
                f"score_col '{self.score_col}'에 inf가 포함되어 있습니다. "
                f"inf 행 수: {np.isinf(col.to_numpy(dtype=float)).sum()}"
            )

    # ------------------------------------------------------------------
    # patch-level 정렬
    # ------------------------------------------------------------------

    def rank_patches(
        self,
        df: pd.DataFrame,
        top_k: Optional[int] = 10,
    ) -> pd.DataFrame:
        """
        patch를 score_col 내림차순으로 정렬하고 top_k를 반환한다.

        Parameters
        ----------
        df : pd.DataFrame
            patch score DataFrame. 원본 컬럼 전체를 보존한다.
        top_k : int | None
            반환할 상위 patch 수. None이면 전체 반환.

        Returns
        -------
        pd.DataFrame
            score_col 내림차순 정렬된 DataFrame. 원본 컬럼 전체 보존.
        """
        self._validate_df(df)

        sorted_df = df.sort_values(
            by=[self.score_col, "patient_id", "local_z", "y0", "x0"],
            ascending=[False, True, True, True, True],
            kind="stable",
        ).reset_index(drop=True)

        if top_k is not None:
            return sorted_df.head(top_k)
        return sorted_df

    # ------------------------------------------------------------------
    # slice-level 정렬
    # ------------------------------------------------------------------

    def rank_slices(
        self,
        df: pd.DataFrame,
        top_k: Optional[int] = 10,
        agg: str = "max",
    ) -> pd.DataFrame:
        """
        local_z 기준으로 score_col을 집계하고 내림차순 정렬한다.

        Parameters
        ----------
        df : pd.DataFrame
            patch score DataFrame.
        top_k : int | None
            반환할 상위 slice 수. None이면 전체 반환.
        agg : str
            집계 방식. 'max', 'mean', 'p95' 중 하나.

        Returns
        -------
        pd.DataFrame
            컬럼: local_z, {score_col}_{agg}, patch_count.
        """
        if agg not in VALID_AGG:
            raise ValueError(
                f"agg는 {VALID_AGG} 중 하나이어야 합니다: {agg!r}"
            )
        self._validate_df(df)

        col = self.score_col
        agg_col = f"{col}_{agg}"

        if agg == "p95":
            grouped = (
                df.groupby("local_z")[col]
                .agg(
                    **{agg_col: lambda x: x.quantile(0.95)},
                )
                .reset_index()
            )
        else:
            grouped = (
                df.groupby("local_z")[col]
                .agg(**{agg_col: agg})
                .reset_index()
            )

        patch_count = df.groupby("local_z").size().rename("patch_count").reset_index()
        result = grouped.merge(patch_count, on="local_z")

        result = result.sort_values(
            by=[agg_col, "local_z"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)

        if top_k is not None:
            return result.head(top_k)
        return result

    # ------------------------------------------------------------------
    # patient-level 정렬
    # ------------------------------------------------------------------

    def rank_patients(
        self,
        df: pd.DataFrame,
        agg: str = "max",
    ) -> pd.DataFrame:
        """
        patient_id 기준으로 score_col을 집계하고 내림차순 정렬한다.

        Parameters
        ----------
        df : pd.DataFrame
            patch score DataFrame.
        agg : str
            집계 방식. 'max', 'mean', 'p95' 중 하나.

        Returns
        -------
        pd.DataFrame
            컬럼: patient_id, {score_col}_{agg}, patch_count.
        """
        if agg not in VALID_AGG:
            raise ValueError(
                f"agg는 {VALID_AGG} 중 하나이어야 합니다: {agg!r}"
            )
        self._validate_df(df)

        col = self.score_col
        agg_col = f"{col}_{agg}"

        if agg == "p95":
            grouped = (
                df.groupby("patient_id")[col]
                .agg(**{agg_col: lambda x: x.quantile(0.95)})
                .reset_index()
            )
        else:
            grouped = (
                df.groupby("patient_id")[col]
                .agg(**{agg_col: agg})
                .reset_index()
            )

        patch_count = df.groupby("patient_id").size().rename("patch_count").reset_index()
        result = grouped.merge(patch_count, on="patient_id")

        result = result.sort_values(
            by=[agg_col, "patient_id"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)

        return result
