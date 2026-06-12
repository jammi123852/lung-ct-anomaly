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


class ScoreAggregator:
    def __init__(self, score_col: str):
        self.score_col = score_col

    def load_csv(self, path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {path}")

        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"필수 공통 컬럼 누락: {missing_cols}")

        if self.score_col not in df.columns:
            raise ValueError(
                f"score_col '{self.score_col}'이 CSV에 없습니다. "
                f"존재하는 컬럼: {list(df.columns)}"
            )

        col = df[self.score_col]
        if col.isna().any():
            raise ValueError(
                f"score_col '{self.score_col}'에 NaN이 포함되어 있습니다. "
                f"NaN 행 수: {col.isna().sum()}"
            )
        if np.isinf(col).any():
            raise ValueError(
                f"score_col '{self.score_col}'에 inf가 포함되어 있습니다. "
                f"inf 행 수: {np.isinf(col).sum()}"
            )

        return df

    def aggregate_slice_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        col = self.score_col
        agg = (
            df.groupby("local_z")[col]
            .agg(
                max="max",
                mean="mean",
                p95=lambda x: x.quantile(0.95),
                count="count",
            )
            .reset_index()
        )
        agg.columns = ["local_z", f"{col}_max", f"{col}_mean", f"{col}_p95", "count"]
        return agg

    def aggregate_patient_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        col = self.score_col
        agg = (
            df.groupby("patient_id")[col]
            .agg(
                max="max",
                mean="mean",
                p95=lambda x: x.quantile(0.95),
                count="count",
            )
            .reset_index()
        )
        agg.columns = [
            "patient_id",
            f"{col}_max",
            f"{col}_mean",
            f"{col}_p95",
            "count",
        ]
        return agg

    def aggregate_position_bin_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        col = self.score_col
        agg = (
            df.groupby("position_bin")[col]
            .agg(
                max="max",
                mean="mean",
                p95=lambda x: x.quantile(0.95),
                count="count",
            )
            .reset_index()
        )
        agg.columns = [
            "position_bin",
            f"{col}_max",
            f"{col}_mean",
            f"{col}_p95",
            "count",
        ]
        return agg

    def aggregate_all(self, path: str) -> dict:
        df = self.load_csv(path)
        return {
            "patches": df,
            "slice_agg": self.aggregate_slice_scores(df),
            "patient_agg": self.aggregate_patient_scores(df),
            "position_bin_agg": self.aggregate_position_bin_scores(df),
        }
