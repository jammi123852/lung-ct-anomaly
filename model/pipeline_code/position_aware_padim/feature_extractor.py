"""
ResNet18 기반 feature 추출 모듈.
FeatureExtractor 클래스를 통해 CT slice 또는 patch 좌표로부터
layer1 / layer2 / layer3 feature map을 추출한다.
"""

import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# ResNet18 ImageNet1K V1 weight 캐시 경로
_RESNET18_CACHE_PATH = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "resnet18-f37072fd.pth"


class FeatureExtractor:
    """
    ResNet18 backbone 기반 feature 추출기.

    전체 slice를 ResNet18에 넣고 중간 레이어 feature map을 추출한다.
    patch를 직접 CNN에 넣지 않고, slice 전체 feature map에서
    patch 중심 좌표를 feature map 좌표로 변환하여 indexing한다.

    사용 레이어:
        layer1: stride 4  → feature map 크기 = H/4 × W/4
        layer2: stride 8  → feature map 크기 = H/8 × W/8
        layer3: stride 16 → feature map 크기 = H/16 × W/16

    캐시 확인 정책:
        ~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth 가 없으면
        자동 다운로드하지 않고 RuntimeError를 발생시킨다.
        사용자 승인 후 직접 다운로드해야 한다.

    Parameters
    ----------
    device : str, optional
        'cuda' 또는 'cpu'. 기본값은 CUDA 사용 가능 시 'cuda', 아니면 'cpu'.
    """

    def __init__(self, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # ResNet18 weight 캐시 존재 여부 확인 — 없으면 자동 다운로드 차단
        if not _RESNET18_CACHE_PATH.exists():
            raise RuntimeError(
                f"ResNet18 weight 캐시 파일이 없습니다: {_RESNET18_CACHE_PATH}\n"
                "자동 다운로드는 허용되지 않습니다. 사용자 승인 후 다운로드 필요.\n"
                "수동 다운로드 방법:\n"
                "  python -c \"from torchvision.models import resnet18, ResNet18_Weights; "
                "resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)\"\n"
                "위 명령 실행 전 반드시 사용자 승인을 받으세요."
            )

        # ResNet18 backbone 로드 (weights API 사용, ImageNet1K V1, 캐시 확인 완료)
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.eval()

        # 필요한 레이어만 분리하여 보관
        self._stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self._layer1 = backbone.layer1
        self._layer2 = backbone.layer2
        self._layer3 = backbone.layer3

        # 모든 파라미터를 device로 이동하고 grad 비활성화
        self._stem.to(self.device)
        self._layer1.to(self.device)
        self._layer2.to(self.device)
        self._layer3.to(self.device)

        for param in self._stem.parameters():
            param.requires_grad_(False)
        for param in self._layer1.parameters():
            param.requires_grad_(False)
        for param in self._layer2.parameters():
            param.requires_grad_(False)
        for param in self._layer3.parameters():
            param.requires_grad_(False)

    def _to_tensor(self, slice_array: np.ndarray) -> torch.Tensor:
        """
        (3, H, W) numpy float32 배열을 (1, 3, H, W) torch.Tensor로 변환한다.

        Parameters
        ----------
        slice_array : np.ndarray
            preprocess_ct_slice 출력. shape: (3, H, W), dtype: float32.

        Returns
        -------
        torch.Tensor
            shape: (1, 3, H, W), device: self.device
        """
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
        tensor = torch.from_numpy(slice_array).unsqueeze(0)  # (1, 3, H, W)
        return tensor.to(self.device)

    def _forward(
        self, tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        stem → layer1 → layer2 → layer3 순서로 forward.

        Parameters
        ----------
        tensor : torch.Tensor
            shape: (1, 3, H, W)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (feat1, feat2, feat3)
            feat1: shape (1, 64, H/4, W/4)
            feat2: shape (1, 128, H/8, W/8)
            feat3: shape (1, 256, H/16, W/16)
        """
        with torch.no_grad():
            x = self._stem(tensor)
            feat1 = self._layer1(x)
            feat2 = self._layer2(feat1)
            feat3 = self._layer3(feat2)
        return feat1, feat2, feat3

    def extract_slice_features(
        self, slice_array: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        CT slice 전체의 feature map을 추출한다.

        Parameters
        ----------
        slice_array : np.ndarray
            preprocess_ct_slice 결과. shape: (3, H, W), dtype: float32.

        Returns
        -------
        dict
            {
                'layer1': np.ndarray, shape (64, H/4, W/4),
                'layer2': np.ndarray, shape (128, H/8, W/8),
                'layer3': np.ndarray, shape (256, H/16, W/16),
            }
        """
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
        patch 좌표 목록에 대한 feature vector를 추출한다.

        전체 slice를 ResNet18에 넣고 feature map에서 patch 중심 좌표를
        feature map 좌표로 변환하여 indexing한다.
        patch를 직접 CNN에 넣지 않는다.

        layer1 / layer2 / layer3 feature를 channel 방향으로 concat하여 반환한다.
        concat 결과 feature_dim = 64 + 128 + 256 = 448.

        좌표 변환:
            각 layer의 stride는 layer1=4, layer2=8, layer3=16.
            patch 중심점 (cy, cx) = ((y0 + y1) / 2, (x0 + x1) / 2).
            feature map 좌표 = floor(중심점 / stride), clamp(0, feat_size - 1).

        Parameters
        ----------
        slice_array : np.ndarray
            preprocess_ct_slice 결과. shape: (3, H, W), dtype: float32.
        patch_coords : list of tuple
            패치 좌표 리스트. 각 원소는 (y0, x0, y1, x1).

        Returns
        -------
        np.ndarray
            shape: (패치 수, feature_dim).
            feature_dim = 64 + 128 + 256 = 448.
            dtype: float32.
        """
        if len(patch_coords) == 0:
            # 패치가 없으면 빈 배열 반환
            return np.zeros((0, 448), dtype=np.float32)

        tensor = self._to_tensor(slice_array)
        feat1, feat2, feat3 = self._forward(tensor)

        # (1, C, Hf, Wf) → (C, Hf, Wf), CPU numpy
        f1 = feat1.squeeze(0).cpu().numpy()  # (64, Hf1, Wf1)
        f2 = feat2.squeeze(0).cpu().numpy()  # (128, Hf2, Wf2)
        f3 = feat3.squeeze(0).cpu().numpy()  # (256, Hf3, Wf3)

        _, Hf1, Wf1 = f1.shape
        _, Hf2, Wf2 = f2.shape
        _, Hf3, Wf3 = f3.shape

        # stride: layer1=4, layer2=8, layer3=16
        strides = [4, 8, 16]
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
                # clamp: feature map 범위 내로 제한
                fy = max(0, min(fy, Hf - 1))
                fx = max(0, min(fx, Wf - 1))
                vectors.append(fmap[:, fy, fx])  # shape: (C,)

            concat_vec = np.concatenate(vectors, axis=0)  # (448,)
            patch_features.append(concat_vec)

        result = np.stack(patch_features, axis=0).astype(np.float32)  # (N, 448)
        return result

    def reduce_dimensions(
        self,
        features: np.ndarray,
        save_path: str = None,
        n_components: int = 100,
        input_dim: int = 448,
        random_seed: int = 42,
    ) -> np.ndarray:
        """
        random feature selection 방식으로 feature 차원을 축소한다.

        448차원 feature vector에서 random seed=42 기반으로 100차원을 선택한다.
        선택된 index는 save_path에 저장되며, 이미 존재하면 로드하여 재사용한다.

        Parameters
        ----------
        features : np.ndarray
            입력 feature. shape: (N, 448), dtype: float32.
        save_path : str
            선택된 feature index 저장 경로. 반드시 절대경로를 전달해야 한다.
            None이면 ValueError. 상대경로이면 ValueError.
            예: str(REPO_ROOT / "outputs/.../selected_feature_indices.npy")
        n_components : int
            선택할 feature 차원 수. 기본값: 100.
        input_dim : int
            기대 입력 차원 수. 기본값: 448.
        random_seed : int
            random seed. 기본값: 42.

        Returns
        -------
        np.ndarray
            shape: (N, 100), dtype: float32.

        Raises
        ------
        ValueError
            save_path가 None이거나 상대경로인 경우.
            입력 shape이 2D가 아니거나, 마지막 차원이 input_dim(448)과 다를 때.
            출력 shape의 마지막 차원이 n_components(100)가 아닐 때.
            NaN 또는 inf가 포함된 경우.
        """
        import os

        # save_path 검증 — None 또는 상대경로 차단
        if save_path is None:
            raise ValueError(
                "save_path는 반드시 절대경로를 전달해야 합니다. "
                "예: str(REPO_ROOT / 'outputs/.../selected_feature_indices.npy')"
            )
        if not os.path.isabs(save_path):
            raise ValueError(
                f"save_path는 절대경로여야 합니다. 상대경로는 허용되지 않습니다. "
                f"받은 값: '{save_path}'"
            )

        # 입력 shape 검증
        if features.ndim != 2:
            raise ValueError(
                f"입력 features는 2D array여야 합니다. 받은 ndim: {features.ndim}, shape: {features.shape}"
            )
        if features.shape[-1] != input_dim:
            raise ValueError(
                f"입력 feature 마지막 차원이 {input_dim}이어야 합니다. "
                f"받은 shape: {features.shape}"
            )

        # 저장 경로 디렉토리 자동 생성
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)

        abs_save_path = save_path

        # 이미 저장된 index가 있으면 로드, 없으면 새로 생성
        if os.path.exists(abs_save_path):
            selected_indices = np.load(abs_save_path)
            print(f"[reduce_dimensions] 기존 selected_feature_indices 로드: {abs_save_path}")
            print(f"[reduce_dimensions] 로드된 index 개수: {len(selected_indices)}")
        else:
            rng = np.random.RandomState(random_seed)
            selected_indices = rng.choice(input_dim, n_components, replace=False)
            selected_indices = np.sort(selected_indices)
            np.save(abs_save_path, selected_indices)
            print(f"[reduce_dimensions] 새 selected_feature_indices 생성 및 저장: {abs_save_path}")
            print(f"[reduce_dimensions] 생성된 index 개수: {len(selected_indices)}")

        # feature 선택
        reduced = features[:, selected_indices].astype(np.float32)  # (N, 100)

        # 출력 shape 검증
        if reduced.shape[-1] != n_components:
            raise ValueError(
                f"출력 feature 마지막 차원이 {n_components}이어야 합니다. "
                f"실제 출력 shape: {reduced.shape}"
            )

        # NaN/inf 검증
        if not np.isfinite(reduced).all():
            nan_count = np.sum(np.isnan(reduced))
            inf_count = np.sum(np.isinf(reduced))
            raise ValueError(
                f"출력 feature에 NaN 또는 inf가 포함되어 있습니다. "
                f"NaN 수: {nan_count}, inf 수: {inf_count}"
            )

        return reduced
