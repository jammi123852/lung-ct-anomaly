"""
HeatmapGenerator: patch score CSV와 CT slice를 입력받아 anomaly heatmap과 overlay를 생성한다.

설계 원칙:
- cv2 / matplotlib 의존 없음. PIL + numpy만 사용한다.
- 파일 저장은 이 클래스에서 수행하지 않는다. 호출부에서 저장한다.
- lesion_mask 경계선 그리기는 optional (None이면 건너뜀).
- score_col 파라미터로 padim_score / hu_z_score를 독립 처리한다.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image  # noqa: F401  overlay 합성용 (현재는 numpy 방식 사용, PIL import 유지)


REQUIRED_PATCH_COLUMNS = ["local_z", "y0", "x0", "y1", "x1"]


def _make_hot_lut() -> np.ndarray:
    """matplotlib.cm.hot과 유사한 256-entry RGB LUT를 생성한다."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    # 0~84: R 0→255, G=0, B=0
    for i in range(85):
        lut[i] = [int(i * 255 / 84), 0, 0]
    # 85~169: R=255, G 0→255, B=0
    for i in range(85, 170):
        lut[i] = [255, int((i - 85) * 255 / 84), 0]
    # 170~255: R=255, G=255, B 0→255
    for i in range(170, 256):
        lut[i] = [255, 255, int((i - 170) * 255 / 85)]
    return lut


_HOT_LUT: np.ndarray = _make_hot_lut()


class HeatmapGenerator:
    """patch score → 2D anomaly heatmap / CT overlay 생성기."""

    def __init__(self, score_col: str, overlap_mode: str = "max") -> None:
        """
        Parameters
        ----------
        score_col : str
            heatmap 생성에 사용할 점수 컬럼명. 예: 'padim_score', 'hu_z_score'.
        overlap_mode : str
            같은 pixel에 여러 patch가 겹칠 때 집계 방식.
            'max'  : 겹치는 patch 중 최댓값 사용 (기본값).
            'mean' : 겹치는 patch의 평균값 사용.
        """
        if overlap_mode not in ("max", "mean"):
            raise ValueError(
                f"overlap_mode는 'max' 또는 'mean'이어야 합니다: {overlap_mode!r}"
            )
        self.score_col = score_col
        self.overlap_mode = overlap_mode

    # ------------------------------------------------------------------
    # 입력 검증
    # ------------------------------------------------------------------

    def _validate_patch_df(self, df: pd.DataFrame) -> None:
        """patch DataFrame의 필수 컬럼과 score_col NaN/inf 여부를 검증한다."""
        missing_patch_cols = [c for c in REQUIRED_PATCH_COLUMNS if c not in df.columns]
        if missing_patch_cols:
            raise ValueError(
                f"patch DataFrame에 필수 컬럼이 없습니다: {missing_patch_cols}"
            )

        if self.score_col not in df.columns:
            raise ValueError(
                f"score_col '{self.score_col}'이 DataFrame에 없습니다. "
                f"존재하는 컬럼: {list(df.columns)}"
            )

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
    # heatmap 생성
    # ------------------------------------------------------------------

    def build_heatmap(
        self, df_slice: pd.DataFrame, ct_h: int, ct_w: int
    ) -> np.ndarray:
        """
        한 slice의 patch score를 2D heatmap으로 변환한다.

        Parameters
        ----------
        df_slice : pd.DataFrame
            단일 local_z에 해당하는 patch rows.
        ct_h, ct_w : int
            CT slice의 height, width.

        Returns
        -------
        np.ndarray
            shape (ct_h, ct_w), dtype float32.
            patch가 없는 위치는 0.0.
        """
        self._validate_patch_df(df_slice)

        if self.overlap_mode == "max":
            score_map = np.full((ct_h, ct_w), -np.inf, dtype=np.float32)
            for _, row in df_slice.iterrows():
                y0 = int(row["y0"])
                x0 = int(row["x0"])
                y1 = int(row["y1"])
                x1 = int(row["x1"])
                score = float(row[self.score_col])
                y0c = max(0, y0)
                y1c = min(ct_h, y1)
                x0c = max(0, x0)
                x1c = min(ct_w, x1)
                if y1c > y0c and x1c > x0c:
                    score_map[y0c:y1c, x0c:x1c] = np.maximum(
                        score_map[y0c:y1c, x0c:x1c], score
                    )
            heatmap = np.zeros((ct_h, ct_w), dtype=np.float32)
            valid = score_map > -np.inf
            heatmap[valid] = score_map[valid]

        else:  # mean
            sum_map = np.zeros((ct_h, ct_w), dtype=np.float64)
            cnt_map = np.zeros((ct_h, ct_w), dtype=np.int32)
            for _, row in df_slice.iterrows():
                y0 = int(row["y0"])
                x0 = int(row["x0"])
                y1 = int(row["y1"])
                x1 = int(row["x1"])
                score = float(row[self.score_col])
                y0c = max(0, y0)
                y1c = min(ct_h, y1)
                x0c = max(0, x0)
                x1c = min(ct_w, x1)
                if y1c > y0c and x1c > x0c:
                    sum_map[y0c:y1c, x0c:x1c] += score
                    cnt_map[y0c:y1c, x0c:x1c] += 1
            heatmap = np.zeros((ct_h, ct_w), dtype=np.float32)
            valid = cnt_map > 0
            heatmap[valid] = (sum_map[valid] / cnt_map[valid]).astype(np.float32)

        return heatmap

    def normalize_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        """
        min-max normalization으로 [0, 1]로 스케일링한다.
        score가 모두 같으면 zeros_like를 반환한다.

        Parameters
        ----------
        heatmap : np.ndarray
            shape (H, W), dtype float32.

        Returns
        -------
        np.ndarray
            shape (H, W), dtype float32, [0, 1].
        """
        h_min = float(heatmap.min())
        h_max = float(heatmap.max())
        if h_max - h_min < 1e-8:
            return np.zeros_like(heatmap)
        return ((heatmap - h_min) / (h_max - h_min)).astype(np.float32)

    # ------------------------------------------------------------------
    # CT slice 변환
    # ------------------------------------------------------------------

    def window_ct_slice(
        self,
        ct_slice: np.ndarray,
        hu_min: int = -1000,
        hu_max: int = 200,
    ) -> np.ndarray:
        """
        HU windowing 후 0~255 uint8로 변환한다.

        Parameters
        ----------
        ct_slice : np.ndarray
            shape (H, W), float 또는 int HU 값. 2D이어야 한다.
        hu_min, hu_max : int
            windowing 범위.

        Returns
        -------
        np.ndarray
            shape (H, W), dtype uint8, [0, 255].
        """
        if ct_slice.ndim != 2:
            raise ValueError(
                f"CT slice는 2D (H, W)이어야 합니다. 실제 shape: {ct_slice.shape}"
            )
        clipped = np.clip(ct_slice.astype(np.float32), hu_min, hu_max)
        normalized = (clipped - hu_min) / float(hu_max - hu_min)
        return (normalized * 255).clip(0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # colormap
    # ------------------------------------------------------------------

    def _apply_colormap(self, norm_heatmap: np.ndarray) -> np.ndarray:
        """
        normalize된 heatmap에 hot colormap LUT를 적용한다.

        Parameters
        ----------
        norm_heatmap : np.ndarray
            shape (H, W), float32 [0, 1].

        Returns
        -------
        np.ndarray
            shape (H, W, 3), dtype uint8, RGB.
        """
        idx = (norm_heatmap * 255).clip(0, 255).astype(np.uint8)
        return _HOT_LUT[idx]

    # ------------------------------------------------------------------
    # lesion boundary (optional)
    # ------------------------------------------------------------------

    def _draw_lesion_boundary(
        self,
        overlay_rgb: np.ndarray,
        lesion_mask_2d: np.ndarray,
        boundary_color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """
        lesion_mask의 경계선을 overlay에 그린다. scipy 없이 numpy만 사용한다.

        Parameters
        ----------
        overlay_rgb : np.ndarray
            shape (H, W, 3) uint8.
        lesion_mask_2d : np.ndarray
            shape (H, W) bool 또는 0/1.
        boundary_color : tuple
            RGB 색상 (기본: 녹색 (0, 255, 0)).

        Returns
        -------
        np.ndarray
            shape (H, W, 3) uint8, 경계선이 그려진 overlay.
        """
        mask = lesion_mask_2d.astype(bool)
        # 4방향 shift AND 연산으로 erosion 계산
        eroded = (
            np.roll(mask, 1, axis=0)
            & np.roll(mask, -1, axis=0)
            & np.roll(mask, 1, axis=1)
            & np.roll(mask, -1, axis=1)
            & mask
        )
        boundary = mask & ~eroded
        result = overlay_rgb.copy()
        result[boundary] = list(boundary_color)
        return result

    # ------------------------------------------------------------------
    # overlay 생성 (파일 저장 없음)
    # ------------------------------------------------------------------

    def create_overlay(
        self,
        ct_slice_hu: np.ndarray,
        df_slice: pd.DataFrame,
        alpha: float = 0.5,
        lesion_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        CT slice에 heatmap을 alpha blending하여 overlay를 생성한다.

        Parameters
        ----------
        ct_slice_hu : np.ndarray
            shape (H, W), HU 값. 2D이어야 한다.
        df_slice : pd.DataFrame
            단일 local_z에 해당하는 patch rows.
        alpha : float
            heatmap 투명도 (0: CT만, 1: heatmap만). 기본값 0.5.
        lesion_mask : np.ndarray | None
            shape (H, W) 또는 (slices, H, W).
            None이면 boundary 그리기를 건너뛴다.

        Returns
        -------
        np.ndarray
            shape (H, W, 3), dtype uint8, RGB ndarray. 파일 저장 없음.
        """
        if ct_slice_hu.ndim != 2:
            raise ValueError(
                f"CT slice는 2D (H, W)이어야 합니다. 실제 shape: {ct_slice_hu.shape}"
            )
        ct_h, ct_w = ct_slice_hu.shape

        # CT windowing → grayscale uint8 → RGB
        ct_uint8 = self.window_ct_slice(ct_slice_hu)
        ct_rgb = np.stack([ct_uint8, ct_uint8, ct_uint8], axis=-1)

        # heatmap 생성 → normalize → colormap 적용
        heatmap = self.build_heatmap(df_slice, ct_h, ct_w)
        norm_hm = self.normalize_heatmap(heatmap)
        heatmap_rgb = self._apply_colormap(norm_hm)

        # heatmap 크기와 CT slice 크기 일치 확인
        if heatmap_rgb.shape[:2] != (ct_h, ct_w):
            raise ValueError(
                f"heatmap 크기 {heatmap_rgb.shape[:2]}가 "
                f"CT slice 크기 {(ct_h, ct_w)}와 다릅니다."
            )

        # alpha blending (float32 연산 후 uint8 clip)
        overlay = (
            (1.0 - alpha) * ct_rgb.astype(np.float32)
            + alpha * heatmap_rgb.astype(np.float32)
        ).clip(0, 255).astype(np.uint8)

        # lesion_mask boundary 그리기 (optional)
        if lesion_mask is not None:
            if lesion_mask.ndim == 3:
                local_z = int(df_slice["local_z"].iloc[0])
                lesion_2d = lesion_mask[local_z]
            else:
                lesion_2d = lesion_mask
            overlay = self._draw_lesion_boundary(overlay, lesion_2d)

        return overlay

    def get_heatmap_array(
        self,
        ct_slice_hu: np.ndarray,
        df_slice: pd.DataFrame,
    ) -> np.ndarray:
        """
        normalize된 heatmap array를 반환한다. 파일 저장은 하지 않는다.

        Parameters
        ----------
        ct_slice_hu : np.ndarray
            shape (H, W), HU 값. 2D이어야 한다.
        df_slice : pd.DataFrame
            단일 local_z에 해당하는 patch rows.

        Returns
        -------
        np.ndarray
            shape (H, W), dtype float32, [0, 1].
        """
        if ct_slice_hu.ndim != 2:
            raise ValueError(
                f"CT slice는 2D (H, W)이어야 합니다. 실제 shape: {ct_slice_hu.shape}"
            )
        ct_h, ct_w = ct_slice_hu.shape
        heatmap = self.build_heatmap(df_slice, ct_h, ct_w)
        return self.normalize_heatmap(heatmap)
