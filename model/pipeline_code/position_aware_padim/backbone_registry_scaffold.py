"""
ResNet50 backbone scaffold 전용 순수 registry 모듈 (P1 scaffold 단계).

목적 (G2 / G4):
- torch / torchvision import 없이, weight 로드 없이, forward 없이
  backbone별 raw_feature_dim, 유효 (backbone, pretrain_source) 조합,
  normalization profile, run tag 를 확인할 수 있도록 순수 상수/헬퍼만 제공한다.

원본 파일(feature_extractor.py, padim_model.py 등)은 일절 수정하지 않는다.
이 모듈은 신규 scaffold 전용이며 기존 ResNet18/v2/v2 실행 경로에 영향이 없다.
"""

from __future__ import annotations

from typing import Dict, Tuple


# ------------------------------------------------------------------
# G2: backbone별 raw_feature_dim (layer1/2/3 channel 합산)
#   resnet18: 64 + 128 + 256  = 448
#   resnet50: 256 + 512 + 1024 = 1792
# weight 로드/forward 없이 이 dict 만으로 차원을 확인할 수 있다.
# ------------------------------------------------------------------
RAW_FEATURE_DIM: Dict[str, int] = {
    "resnet18": 448,
    "resnet50": 1792,
}

# backbone별 layer 채널 (참고용 — 합산이 RAW_FEATURE_DIM과 일치하는지 dry-check에서 검증)
BACKBONE_LAYER_CHANNELS: Dict[str, Tuple[int, int, int]] = {
    "resnet18": (64, 128, 256),
    "resnet50": (256, 512, 1024),
}

# 사용 feature layer 및 stride (resnet18/resnet50 동일)
FEATURE_LAYERS: Tuple[str, str, str] = ("layer1", "layer2", "layer3")
FEATURE_STRIDES: Tuple[int, int, int] = (4, 8, 16)

# reduce 후 최종 feature_dim (backbone 무관, 기존 유지)
REDUCED_FEATURE_DIM: int = 100


# ------------------------------------------------------------------
# G4: 유효 (backbone, pretrain_source) 조합
#   resnet18 + imagenet         : 기존 baseline (허용)
#   resnet50 + imagenet         : 단계 A (허용)
#   resnet50 + radimagenet      : 단계 B (허용, 단 이번 단계 weight 로드 금지)
#   resnet18 + radimagenet      : 금지 (RadImageNet resnet18 미제공 가정)
# ------------------------------------------------------------------
VALID_COMBOS: Tuple[Tuple[str, str], ...] = (
    ("resnet18", "imagenet"),
    ("resnet50", "imagenet"),
    ("resnet50", "radimagenet"),
)

SUPPORTED_BACKBONES: Tuple[str, ...] = ("resnet18", "resnet50")
SUPPORTED_PRETRAIN_SOURCES: Tuple[str, ...] = ("imagenet", "radimagenet")


# ------------------------------------------------------------------
# G5: normalization profile
#   imagenet              : 기존 ImageNet mean/std와 동일 (기본값)
#   radimagenet_recommended : placeholder. 아직 확정 금지, 기본 사용 금지.
# ------------------------------------------------------------------
NORMALIZATION_PROFILES: Tuple[str, ...] = ("imagenet", "radimagenet_recommended")
DEFAULT_NORMALIZATION_PROFILE: str = "imagenet"


# ------------------------------------------------------------------
# G6: run tag (기존 padim_v2_roi0_0 와 섞이지 않는 신규 tag)
# ------------------------------------------------------------------
SCAFFOLD_RUN_TAGS: Tuple[str, str] = (
    "padim_v2_resnet50_imagenet",
    "padim_v2_resnet50_radimagenet",
)

# run tag → (backbone, pretrain_source) 매핑
RUN_TAG_TO_COMBO: Dict[str, Tuple[str, str]] = {
    "padim_v2_resnet50_imagenet": ("resnet50", "imagenet"),
    "padim_v2_resnet50_radimagenet": ("resnet50", "radimagenet"),
}


def get_raw_feature_dim(backbone: str) -> int:
    """backbone 이름으로 raw_feature_dim을 반환한다. (forward/weight 로드 없음)"""
    if backbone not in RAW_FEATURE_DIM:
        raise ValueError(
            f"지원하지 않는 backbone: '{backbone}'. "
            f"지원: {sorted(RAW_FEATURE_DIM.keys())}"
        )
    return RAW_FEATURE_DIM[backbone]


def validate_backbone_pretrain(backbone: str, pretrain_source: str) -> None:
    """
    (backbone, pretrain_source) 조합 유효성 검증 (G4).

    Raises
    ------
    ValueError
        backbone/pretrain_source 자체가 미지원이거나, 조합이 VALID_COMBOS에 없을 때.
        예: ('resnet18', 'radimagenet') 는 금지.
    """
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"지원하지 않는 backbone: '{backbone}'. 지원: {list(SUPPORTED_BACKBONES)}"
        )
    if pretrain_source not in SUPPORTED_PRETRAIN_SOURCES:
        raise ValueError(
            f"지원하지 않는 pretrain_source: '{pretrain_source}'. "
            f"지원: {list(SUPPORTED_PRETRAIN_SOURCES)}"
        )
    if (backbone, pretrain_source) not in VALID_COMBOS:
        raise ValueError(
            f"무효 조합: ({backbone}, {pretrain_source}). "
            f"허용 조합: {list(VALID_COMBOS)}. "
            "예: resnet18 + radimagenet 은 현재 금지."
        )


def validate_normalization_profile(profile: str, allow_placeholder: bool = False) -> None:
    """
    normalization profile 유효성 검증 (G5).

    Parameters
    ----------
    profile : str
        'imagenet' 또는 'radimagenet_recommended'.
    allow_placeholder : bool
        False(기본)이면 'radimagenet_recommended'(placeholder) 사용을 차단한다.
        이번 P1 단계에서는 항상 False로 호출해 placeholder 기본 사용을 막는다.
    """
    if profile not in NORMALIZATION_PROFILES:
        raise ValueError(
            f"지원하지 않는 normalization_profile: '{profile}'. "
            f"지원: {list(NORMALIZATION_PROFILES)}"
        )
    if profile == "radimagenet_recommended" and not allow_placeholder:
        raise ValueError(
            "normalization_profile='radimagenet_recommended'는 아직 확정되지 않은 "
            "placeholder입니다. 기본 사용이 금지되어 있습니다. "
            "RadImageNet 권장 정규화 확정 후 allow_placeholder=True로만 사용하세요."
        )
