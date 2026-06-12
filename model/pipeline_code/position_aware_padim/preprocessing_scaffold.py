"""
ResNet50 scaffold 전용 normalization profile 분기 전처리 래퍼 (G5).

핵심 보장:
- normalization_profile='imagenet'(기본)일 때, 기존 preprocessing.preprocess_ct_slice
  를 그대로 호출하므로 기존 ImageNet mean/std 결과와 "완전히 동일"하다.
  (별도 상수를 다시 정의하지 않고 원본 함수에 위임 → 값 drift 불가)
- normalization_profile='radimagenet_recommended'는 아직 미확정 placeholder이며
  호출 시 NotImplementedError를 발생시킨다 (이번 단계 사용 금지).

원본 preprocessing.py 는 수정하지 않고 import만 한다 (read-only 참조).
이 모듈은 신규 scaffold 전용이며 model forward / weight 로드와 무관하다.
"""

from __future__ import annotations

import numpy as np

from .backbone_registry_scaffold import (
    DEFAULT_NORMALIZATION_PROFILE,
    validate_normalization_profile,
)
# 원본 ImageNet 전처리 함수와 상수를 read-only로 참조 (수정하지 않음)
from .preprocessing import (
    preprocess_ct_slice as _original_preprocess_ct_slice,
    _IMAGENET_MEAN,
    _IMAGENET_STD,
)

# scaffold가 노출하는 ImageNet 상수는 원본과 동일 객체를 그대로 가리킨다 (G5 동일성 보장)
SCAFFOLD_IMAGENET_MEAN = _IMAGENET_MEAN
SCAFFOLD_IMAGENET_STD = _IMAGENET_STD


def imagenet_constants_match_original() -> bool:
    """
    scaffold ImageNet 상수가 원본 preprocessing 상수와 완전히 동일한지 확인 (G5 dry-check).
    원본 함수에 위임하는 구조라 항상 True여야 한다.
    """
    return bool(
        np.array_equal(SCAFFOLD_IMAGENET_MEAN, _IMAGENET_MEAN)
        and np.array_equal(SCAFFOLD_IMAGENET_STD, _IMAGENET_STD)
    )


def preprocess_ct_slice_scaffold(
    slice_2d: np.ndarray,
    normalization_profile: str = DEFAULT_NORMALIZATION_PROFILE,
    hu_min: float = -1000.0,
    hu_max: float = 200.0,
) -> np.ndarray:
    """
    normalization_profile 분기 전처리 (G5).

    Parameters
    ----------
    slice_2d : np.ndarray
        2D CT HU slice. shape (H, W), 부동소수점.
    normalization_profile : str
        'imagenet'(기본) → 원본 preprocess_ct_slice 그대로 호출 (동일 결과 보장).
        'radimagenet_recommended' → NotImplementedError (미확정 placeholder, 사용 금지).
    hu_min, hu_max : float
        HU windowing 범위. 원본과 동일 기본값.

    Returns
    -------
    np.ndarray
        shape (3, H, W), dtype float32. (imagenet profile일 때 원본과 byte-identical)
    """
    # placeholder 사용 차단 (allow_placeholder=False)
    validate_normalization_profile(normalization_profile, allow_placeholder=False)

    if normalization_profile == "imagenet":
        # 원본 함수에 그대로 위임 → 기존 ImageNet 결과와 완전히 동일
        return _original_preprocess_ct_slice(slice_2d, hu_min=hu_min, hu_max=hu_max)

    # validate_normalization_profile가 placeholder를 이미 차단하므로 도달하지 않음.
    # 방어적으로 명시 (RadImageNet 정규화는 P 단계 승인 후 확정).
    raise NotImplementedError(
        "RadImageNet 권장 정규화는 아직 확정되지 않았습니다 (placeholder). "
        "이번 단계에서 사용 금지."
    )
