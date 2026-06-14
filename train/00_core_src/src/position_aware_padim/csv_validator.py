"""
CSVValidator: Patch CSV 파일의 형식과 내용을 검증한다.

검증 항목:
- 필수 컬럼 존재 여부
- 좌표 범위 (0 <= x0 < x1 <= 512, 0 <= y0 < y1 <= 512)
- position_bin 값이 6개 정의값 중 하나인지 확인
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


VALID_POSITION_BINS = {
    "upper_central",
    "upper_peripheral",
    "middle_central",
    "middle_peripheral",
    "lower_central",
    "lower_peripheral",
}

# patient_id 또는 safe_id 중 하나 이상 필수
_ID_COLUMNS = {"patient_id", "safe_id"}

# 나머지 필수 컬럼
_REQUIRED_COLUMNS = {"local_z", "y0", "x0", "y1", "x1", "position_bin"}


class CSVValidator:
    """Patch CSV 파일의 형식과 내용을 검증한다."""

    def validate_patch_csv(self, csv_path: str) -> Dict:
        """
        Patch CSV 파일을 검증한다.

        Parameters
        ----------
        csv_path : str
            검증할 CSV 파일 경로.

        Returns
        -------
        dict with keys:
            is_valid            : bool
            missing_columns     : list  (누락된 필수 컬럼명)
            coord_errors        : int   (좌표 범위 위반 행 수)
            invalid_position_bins : list (발견된 비정상 position_bin 값, unique)
            row_count           : int
        """
        result: Dict = {
            "is_valid": False,
            "missing_columns": [],
            "coord_errors": 0,
            "invalid_position_bins": [],
            "row_count": 0,
        }

        if not Path(csv_path).exists():
            result["missing_columns"] = ["<file_not_found>"]
            return result

        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        result["row_count"] = len(df)
        actual_cols = set(df.columns.tolist())

        # ID 컬럼: patient_id 또는 safe_id 중 하나 이상 있어야 함
        has_id = bool(_ID_COLUMNS & actual_cols)
        missing: List[str] = []
        if not has_id:
            missing.append("patient_id or safe_id")

        # 나머지 필수 컬럼 확인
        for col in sorted(_REQUIRED_COLUMNS):
            if col not in actual_cols:
                missing.append(col)

        result["missing_columns"] = missing
        if missing:
            return result

        # 좌표 범위 확인
        coord_mask = (
            (df["x0"] < 0) | (df["x0"] >= df["x1"]) | (df["x1"] > 512)
            | (df["y0"] < 0) | (df["y0"] >= df["y1"]) | (df["y1"] > 512)
        )
        result["coord_errors"] = int(coord_mask.sum())

        # position_bin 값 확인
        invalid_bins = (
            df["position_bin"]
            .dropna()
            .loc[~df["position_bin"].isin(VALID_POSITION_BINS)]
            .unique()
            .tolist()
        )
        result["invalid_position_bins"] = [str(v) for v in invalid_bins]

        result["is_valid"] = (
            result["coord_errors"] == 0
            and len(result["invalid_position_bins"]) == 0
        )

        return result
