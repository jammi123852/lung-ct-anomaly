"""
EfficientNet-B0 ImageNet backbone feature 추출기 scaffold (P-A67a).

기존 feature_extractor.py / feature_extractor_resnet50_scaffold.py 를 수정하지 않고
EfficientNet-B0 전용 신규 추출기를 분리 파일로 제공한다.

Feature tap points (224×224 input 기준):
    features[0:3] → output: 24ch, stride=4  (H/4 × W/4)  ← layer1 대응
    features[3:4] → output: 40ch, stride=8  (H/8 × W/8)  ← layer2 대응
    features[4:5] → output: 80ch, stride=16 (H/16 × W/16) ← layer3 대응
    raw_feature_dim = 24 + 40 + 80 = 144

P-A67a scaffold 단계 규칙:
    - weight 캐시가 없으면 RuntimeError (자동 다운로드 차단).
    - P-A67a 단계에서는 이 클래스를 인스턴스화하지 않는다. 코드 초안만 작성.
    - model forward, training, scoring, selected_feature_indices 생성 금지.
    - P-A67b 이후 사용자 승인 후에만 인스턴스화/forward 허용.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


# ------------------------------------------------------------------
# EfficientNet-B0 tap point 설계 (weight 로드 없이 순수 상수)
# ------------------------------------------------------------------

# tap layer별 output channel (torchvision EfficientNet-B0 기준, width_mult=1.0)
#   features[2]: MBConv6 stride=2, 16→24ch → cumulative stride 4 from 224 input
#   features[3]: MBConv6 stride=2, 24→40ch → cumulative stride 8
#   features[4]: MBConv6 stride=2, 40→80ch → cumulative stride 16
EFFNET_B0_LAYER_CHANNELS: Tuple[int, int, int] = (24, 40, 80)
EFFNET_B0_STRIDES: Tuple[int, int, int] = (4, 8, 16)

RAW_FEATURE_DIM: int = sum(EFFNET_B0_LAYER_CHANNELS)  # 144

REDUCED_FEATURE_DIM: int = 100  # reduce 후 차원 (PaDiM 기존 정책 유지)

# weight cache 경로 (torchvision 0.20.1 기준)
_EFFNET_B0_CACHE_FILENAME = "efficientnet_b0_rwightman-7f5810bc.pth"
_EFFNET_B0_CACHE_PATH = (
    Path(torch.hub.get_dir()) / "checkpoints" / _EFFNET_B0_CACHE_FILENAME
)


class FeatureExtractorEffNetB0:
    """
    EfficientNet-B0 ImageNet backbone feature 추출기.

    torchvision.models.efficientnet_b0 의 features[2], features[3], features[4]를
    tap point로 사용한다. stride 4/8/16 구조가 ResNet18과 동일하므로
    기존 patch 좌표 변환 로직을 그대로 재사용할 수 있다.

    Parameters
    ----------
    device : str, optional
        'cuda' 또는 'cpu'. 기본값은 CUDA 가용 시 'cuda', 아니면 'cpu'.

    Notes
    -----
    - __init__ 에서 weight 캐시 존재를 확인하며, 없으면 RuntimeError.
      자동 다운로드 차단. P-A67b 단계에서 사용자 승인 후 다운로드.
    - P-A67a 단계에서는 인스턴스화 금지.
    """

    def __init__(self, device: str = None) -> None:
        # weight 캐시 존재 확인 — 없으면 자동 다운로드 차단
        if not _EFFNET_B0_CACHE_PATH.exists():
            raise RuntimeError(
                f"EfficientNet-B0 weight 캐시 파일이 없습니다: {_EFFNET_B0_CACHE_PATH}\n"
                "자동 다운로드는 허용되지 않습니다. P-A67b 사용자 승인 후 다운로드 필요.\n"
                "다운로드 방법 (승인 후에만 실행):\n"
                "  python -c \"from torchvision.models import efficientnet_b0, "
                "EfficientNet_B0_Weights; efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)\""
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.raw_feature_dim: int = RAW_FEATURE_DIM  # 144

        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        backbone.eval()

        # tap points: features 슬라이싱으로 cumulative stride 확보
        #   self._early : features[0~2] → 24ch, stride 4
        #   self._mid   : features[3]   → 40ch, stride 8
        #   self._late  : features[4]   → 80ch, stride 16
        self._early = nn.Sequential(*list(backbone.features[:3]))
        self._mid   = nn.Sequential(*list(backbone.features[3:4]))
        self._late  = nn.Sequential(*list(backbone.features[4:5]))

        for module in (self._early, self._mid, self._late):
            module.to(self.device)
            for param in module.parameters():
                param.requires_grad_(False)

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
                f"입력 dtype이 float여야 합니다. 받은 dtype: {slice_array.dtype}"
            )
        return torch.from_numpy(slice_array).unsqueeze(0).to(self.device)

    def _forward(
        self, tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            f_early = self._early(tensor)   # (1, 24, H/4, W/4)
            f_mid   = self._mid(f_early)    # (1, 40, H/8, W/8)
            f_late  = self._late(f_mid)     # (1, 80, H/16, W/16)
        return f_early, f_mid, f_late

    def extract_slice_features(self, slice_array: np.ndarray) -> Dict[str, np.ndarray]:
        """
        slice 전체의 feature map을 layer별로 반환한다.

        Returns
        -------
        dict with keys 'early' (24ch), 'mid' (40ch), 'late' (80ch).
        """
        tensor = self._to_tensor(slice_array)
        f_early, f_mid, f_late = self._forward(tensor)
        return {
            "early": f_early.squeeze(0).cpu().numpy(),  # (24, H/4, W/4)
            "mid":   f_mid.squeeze(0).cpu().numpy(),    # (40, H/8, W/8)
            "late":  f_late.squeeze(0).cpu().numpy(),   # (80, H/16, W/16)
        }

    def extract_patch_features(
        self,
        slice_array: np.ndarray,
        patch_coords: List[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        """
        patch 좌표별 feature vector 추출.
        concat 차원 = RAW_FEATURE_DIM = 144.

        patch 좌표 변환 규약은 ResNet18 feature_extractor.py와 동일.
        (center 좌표 → feature map 좌표, stride 4/8/16)

        Parameters
        ----------
        slice_array : np.ndarray, shape (3, H, W), float
        patch_coords : list of (y0, x0, y1, x1)

        Returns
        -------
        np.ndarray, shape (M, 144), float32
        """
        if len(patch_coords) == 0:
            return np.zeros((0, self.raw_feature_dim), dtype=np.float32)

        tensor = self._to_tensor(slice_array)
        f_early, f_mid, f_late = self._forward(tensor)

        fe = f_early.squeeze(0).cpu().numpy()  # (24, H/4, W/4)
        fm = f_mid.squeeze(0).cpu().numpy()    # (40, H/8, W/8)
        fl = f_late.squeeze(0).cpu().numpy()   # (80, H/16, W/16)

        _, Hfe, Wfe = fe.shape
        _, Hfm, Wfm = fm.shape
        _, Hfl, Wfl = fl.shape

        feat_maps = [fe, fm, fl]
        strides    = list(EFFNET_B0_STRIDES)  # (4, 8, 16)
        feat_sizes = [(Hfe, Wfe), (Hfm, Wfm), (Hfl, Wfl)]

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
            concat_vec = np.concatenate(vectors, axis=0)  # (144,)
            patch_features.append(concat_vec)

        return np.stack(patch_features, axis=0).astype(np.float32)

    def reduce_dimensions(
        self,
        features: np.ndarray,
        save_path: str = None,
        n_components: int = REDUCED_FEATURE_DIM,
        input_dim: int = None,
        random_seed: int = 42,
    ) -> np.ndarray:
        """
        random feature selection (기존 정책과 동일).
        P-A67a 단계에서는 selected_feature_indices 생성 금지.

        input_dim 기본값은 self.raw_feature_dim(=144).
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
