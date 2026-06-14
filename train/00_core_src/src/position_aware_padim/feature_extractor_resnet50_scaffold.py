"""
ResNet18 / ResNet50 backbone 분기 feature 추출 scaffold (G1).

원본 feature_extractor.py 를 수정하지 않고, backbone(resnet18/resnet50) 과
pretrain_source(imagenet/radimagenet) 를 인자로 받는 신규 추출기를 제공한다.

이번 P1 scaffold 단계 규칙:
- weight 자동 다운로드 금지 (캐시 없으면 RuntimeError).
- RadImageNet weight 는 실제 로드하지 않는다 (경로/키만 준비, 로드 시 NotImplementedError).
- dry-check 는 이 클래스를 "인스턴스화하지 않고" backbone_registry_scaffold 의
  순수 헬퍼만으로 차원/조합을 확인한다 (forward/weight 로드 없음).

좌표 변환 / extract 로직은 원본과 동일 규약을 따른다 (stride 4/8/16, 채널 합산은
backbone에 따라 자동 결정 — resnet18=448, resnet50=1792).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .backbone_registry_scaffold import (
    FEATURE_STRIDES,
    get_raw_feature_dim,
    validate_backbone_pretrain,
)

# RadImageNet ResNet50 weight 후보 경로 (이번 단계 로드 금지 — placeholder 준비만)
RADIMAGENET_RESNET50_WEIGHT_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "weights" / "radimagenet_resnet50.pth"
)


class FeatureExtractorScaffold:
    """
    backbone / pretrain_source 분기 feature 추출기 (scaffold).

    Parameters
    ----------
    backbone : str
        'resnet18' 또는 'resnet50'.
    pretrain_source : str
        'imagenet' 또는 'radimagenet'. (resnet18+radimagenet 조합은 금지)
    device : str, optional
        'cuda' 또는 'cpu'. 기본은 CUDA 가용 시 'cuda'.

    Notes
    -----
    - __init__ 에서 ImageNet weight 캐시 존재를 확인하며, 없으면 RuntimeError.
      (자동 다운로드 금지)
    - pretrain_source='radimagenet' 인 경우 이번 단계에서는 NotImplementedError.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrain_source: str = "imagenet",
        device: str = None,
    ):
        # G4: 조합 검증 (resnet18+radimagenet 등 무효 조합 차단)
        validate_backbone_pretrain(backbone, pretrain_source)

        self.backbone_name = backbone
        self.pretrain_source = pretrain_source
        self.raw_feature_dim = get_raw_feature_dim(backbone)  # 448 / 1792

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if pretrain_source == "radimagenet":
            # 이번 P1 단계에서는 RadImageNet weight 로드 금지 (placeholder만 준비)
            raise NotImplementedError(
                "RadImageNet weight 로드는 이번 scaffold 단계에서 금지되어 있습니다. "
                f"준비된 후보 경로: {RADIMAGENET_RESNET50_WEIGHT_PATH} "
                "(P1 사용자 승인 후 다운로드/로드)."
            )

        # ImageNet pretrain 경로 (resnet18 / resnet50)
        backbone_obj, expected_cache = self._build_imagenet_backbone(backbone)

        # weight 캐시 존재 확인 — 없으면 자동 다운로드 차단
        if not expected_cache.exists():
            raise RuntimeError(
                f"{backbone} ImageNet weight 캐시 파일이 없습니다: {expected_cache}\n"
                "자동 다운로드는 허용되지 않습니다. 사용자 승인(P1) 후 다운로드 필요."
            )

        backbone_obj.eval()
        self._stem = nn.Sequential(
            backbone_obj.conv1,
            backbone_obj.bn1,
            backbone_obj.relu,
            backbone_obj.maxpool,
        )
        self._layer1 = backbone_obj.layer1
        self._layer2 = backbone_obj.layer2
        self._layer3 = backbone_obj.layer3

        for module in (self._stem, self._layer1, self._layer2, self._layer3):
            module.to(self.device)
            for param in module.parameters():
                param.requires_grad_(False)

    @staticmethod
    def _build_imagenet_backbone(backbone: str):
        """
        backbone 객체와 기대 weight 캐시 경로를 반환한다.
        weights enum 의 url 에서 캐시 파일명을 도출하므로 hash 하드코딩이 없다.
        (url 문자열 읽기는 다운로드를 유발하지 않는다)

        cache 위치는 torch.hub.get_dir()(=$TORCH_HOME/hub)를 따른다.
        워크스페이스에서 TORCH_HOME을 설정하면 weight를 워크스페이스 내부에서 찾는다.
        """
        cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
        if backbone == "resnet18":
            from torchvision.models import resnet18, ResNet18_Weights

            weights = ResNet18_Weights.IMAGENET1K_V1
            model = resnet18(weights=weights)
        elif backbone == "resnet50":
            from torchvision.models import resnet50, ResNet50_Weights

            weights = ResNet50_Weights.IMAGENET1K_V1
            model = resnet50(weights=weights)
        else:
            raise ValueError(f"지원하지 않는 backbone: '{backbone}'")

        expected_cache = cache_dir / os.path.basename(weights.url)
        return model, expected_cache

    def _to_tensor(self, slice_array: np.ndarray) -> torch.Tensor:
        if not isinstance(slice_array, np.ndarray):
            raise ValueError(
                f"입력은 numpy ndarray여야 합니다. 받은 타입: {type(slice_array)}"
            )
        if slice_array.ndim != 3 or slice_array.shape[0] != 3:
            raise ValueError(
                f"입력 shape이 (3, H, W) 여야 합니다. 받은 shape: {slice_array.shape}"
            )
        if not np.issubdtype(slice_array.dtype, np.floating):
            raise ValueError(
                f"입력 dtype이 float32여야 합니다. 받은 dtype: {slice_array.dtype}"
            )
        tensor = torch.from_numpy(slice_array).unsqueeze(0)
        return tensor.to(self.device)

    def _forward(
        self, tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            x = self._stem(tensor)
            feat1 = self._layer1(x)
            feat2 = self._layer2(feat1)
            feat3 = self._layer3(feat2)
        return feat1, feat2, feat3

    def extract_slice_features(self, slice_array: np.ndarray) -> Dict[str, np.ndarray]:
        tensor = self._to_tensor(slice_array)
        feat1, feat2, feat3 = self._forward(tensor)
        return {
            "layer1": feat1.squeeze(0).cpu().numpy(),
            "layer2": feat2.squeeze(0).cpu().numpy(),
            "layer3": feat3.squeeze(0).cpu().numpy(),
        }

    def extract_patch_features(
        self,
        slice_array: np.ndarray,
        patch_coords: List[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        """
        patch 좌표별 feature vector 추출. concat 차원 = self.raw_feature_dim
        (resnet18=448, resnet50=1792). 좌표 변환은 원본과 동일 규약.
        """
        if len(patch_coords) == 0:
            return np.zeros((0, self.raw_feature_dim), dtype=np.float32)

        tensor = self._to_tensor(slice_array)
        feat1, feat2, feat3 = self._forward(tensor)

        f1 = feat1.squeeze(0).cpu().numpy()
        f2 = feat2.squeeze(0).cpu().numpy()
        f3 = feat3.squeeze(0).cpu().numpy()

        _, Hf1, Wf1 = f1.shape
        _, Hf2, Wf2 = f2.shape
        _, Hf3, Wf3 = f3.shape

        strides = list(FEATURE_STRIDES)  # (4, 8, 16)
        feat_maps = [f1, f2, f3]
        feat_sizes = [(Hf1, Wf1), (Hf2, Wf2), (Hf3, Wf3)]

        patch_features = []
        for (y0, x0, y1, x1) in patch_coords:
            cy = (y0 + y1) / 2.0
            cx = (x0 + x1) / 2.0
            vectors = []
            for fmap, stride, (Hf, Wf) in zip(feat_maps, strides, feat_sizes):
                fy = int(math.floor(cy / stride))
                fx = int(math.floor(cx / stride))
                fy = max(0, min(fy, Hf - 1))
                fx = max(0, min(fx, Wf - 1))
                vectors.append(fmap[:, fy, fx])
            concat_vec = np.concatenate(vectors, axis=0)
            patch_features.append(concat_vec)

        result = np.stack(patch_features, axis=0).astype(np.float32)
        return result

    def reduce_dimensions(
        self,
        features: np.ndarray,
        save_path: str = None,
        n_components: int = 100,
        input_dim: int = None,
        random_seed: int = 42,
    ) -> np.ndarray:
        """
        random feature selection (원본과 동일 규약). input_dim 기본값을
        self.raw_feature_dim(backbone별 448/1792)로 자동 사용한다.

        selected_feature_indices 는 save_path(절대경로)에 저장/로드된다.
        이번 단계에서는 이 메서드를 실행하지 않는다 (index 생성 금지).
        """
        if input_dim is None:
            input_dim = self.raw_feature_dim

        if save_path is None:
            raise ValueError("save_path는 반드시 절대경로를 전달해야 합니다.")
        if not os.path.isabs(save_path):
            raise ValueError(
                f"save_path는 절대경로여야 합니다. 받은 값: '{save_path}'"
            )
        if features.ndim != 2:
            raise ValueError(
                f"입력 features는 2D array여야 합니다. 받은 shape: {features.shape}"
            )
        if features.shape[-1] != input_dim:
            raise ValueError(
                f"입력 feature 마지막 차원이 {input_dim}이어야 합니다. "
                f"받은 shape: {features.shape}"
            )

        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)

        if os.path.exists(save_path):
            selected_indices = np.load(save_path)
        else:
            rng = np.random.RandomState(random_seed)
            selected_indices = rng.choice(input_dim, n_components, replace=False)
            selected_indices = np.sort(selected_indices)
            np.save(save_path, selected_indices)

        reduced = features[:, selected_indices].astype(np.float32)
        if reduced.shape[-1] != n_components:
            raise ValueError(
                f"출력 feature 마지막 차원이 {n_components}이어야 합니다. "
                f"실제 shape: {reduced.shape}"
            )
        if not np.isfinite(reduced).all():
            raise ValueError("출력 feature에 NaN 또는 inf가 포함되어 있습니다.")
        return reduced
