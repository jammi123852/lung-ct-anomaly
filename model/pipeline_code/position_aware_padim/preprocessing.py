"""
CT slice 전처리 모듈.
preprocess_ct_slice 함수를 통해 2D CT HU slice를 ResNet18 입력 형식으로 변환한다.
"""

import numpy as np


# ImageNet normalization 상수
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_ct_slice(
    slice_2d: np.ndarray,
    hu_min: float = -1000.0,
    hu_max: float = 200.0,
) -> np.ndarray:
    """
    2D CT HU slice를 ResNet18 입력 형식으로 전처리한다.

    처리 순서:
        1. 입력 검증 (shape, dtype, NaN/inf)
        2. HU windowing: [hu_min, hu_max] 범위로 clip
        3. 0-1 normalization
        4. 1-channel → 3-channel 복제
        5. ImageNet normalization (mean/std)

    Parameters
    ----------
    slice_2d : np.ndarray
        2D CT HU slice. shape: (H, W), dtype: float32 또는 float64.
        값 범위는 HU 단위 (예: -1000 ~ 200).
    hu_min : float
        HU windowing 하한값. 기본값 -1000.
    hu_max : float
        HU windowing 상한값. 기본값 200.

    Returns
    -------
    np.ndarray
        전처리된 배열. shape: (3, H, W), dtype: float32.

    Raises
    ------
    ValueError
        입력 shape이 2D가 아닌 경우.
        입력 dtype이 부동소수점이 아닌 경우.
        입력에 NaN 또는 inf가 포함된 경우.
        hu_min >= hu_max 인 경우.
    """
    # --- 1. 입력 검증 ---
    if not isinstance(slice_2d, np.ndarray):
        raise ValueError(
            f"입력은 numpy ndarray여야 합니다. 받은 타입: {type(slice_2d)}"
        )

    if slice_2d.ndim != 2:
        raise ValueError(
            f"입력 shape이 2D여야 합니다. 받은 shape: {slice_2d.shape}"
        )

    if not np.issubdtype(slice_2d.dtype, np.floating):
        raise ValueError(
            f"입력 dtype이 부동소수점(float32, float64 등)이어야 합니다. "
            f"받은 dtype: {slice_2d.dtype}"
        )

    if np.any(np.isnan(slice_2d)):
        raise ValueError("입력 배열에 NaN이 포함되어 있습니다.")

    if np.any(np.isinf(slice_2d)):
        raise ValueError("입력 배열에 inf가 포함되어 있습니다.")

    if hu_min >= hu_max:
        raise ValueError(
            f"hu_min({hu_min})은 hu_max({hu_max})보다 작아야 합니다."
        )

    # --- 2. HU windowing: [hu_min, hu_max]로 clip ---
    windowed = np.clip(slice_2d, hu_min, hu_max)

    # --- 3. 0-1 normalization ---
    normalized = (windowed - hu_min) / (hu_max - hu_min)
    normalized = normalized.astype(np.float32)

    # --- 4. 1-channel → 3-channel 복제: shape (H, W) → (3, H, W) ---
    three_channel = np.stack([normalized, normalized, normalized], axis=0)

    # --- 5. ImageNet normalization: (C, H, W) 기준으로 적용 ---
    # mean, std shape: (3, 1, 1) 로 브로드캐스팅
    mean = _IMAGENET_MEAN[:, np.newaxis, np.newaxis]
    std = _IMAGENET_STD[:, np.newaxis, np.newaxis]
    result = (three_channel - mean) / std

    return result.astype(np.float32)
