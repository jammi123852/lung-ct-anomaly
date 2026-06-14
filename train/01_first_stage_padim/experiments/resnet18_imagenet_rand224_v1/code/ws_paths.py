"""
ResNet18 ImageNet random224 실험 워크스페이스 경로 단일 정의.

모든 산출물은 experiments/resnet18_imagenet_rand224_v1/outputs/ 아래에만 저장한다.
기존 outputs/position-aware-padim-v1/ 결과 트리에는 절대 쓰지 않는다.
"""

from __future__ import annotations

from pathlib import Path

# repo 루트: 이 파일 기준 experiments/resnet18_imagenet_rand224_v1/code/ → 3단계 상위
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"

# 격리 워크스페이스 루트
EXP_ROOT = Path(__file__).resolve().parents[1]  # experiments/resnet18_imagenet_rand224_v1
OUTPUTS = EXP_ROOT / "outputs"

# backbone 고정값
BACKBONE = "resnet18"
PRETRAIN_SOURCE = "imagenet"
RAW_FEATURE_DIM = 448          # layer1=64 + layer2=128 + layer3=256
REDUCED_FEATURE_DIM = 224      # rand224
NORMALIZATION_PROFILE = "imagenet"

MASK_TYPE = "roi_0_0"
PATHS_CONFIG = "configs/paths.local.v2_roi0_0.yaml"

# selected_feature_indices (P-A53 생성, read-only 이후 사용)
SELECTED_INDICES_PATH = OUTPUTS / "models" / "distributions" / "selected_feature_indices.npy"

# full train 산출물
MODELS_DIR = OUTPUTS / "models"
DISTRIBUTIONS_DIR = MODELS_DIR / "distributions"
MODEL_NPZ = DISTRIBUTIONS_DIR / "position_bin_stats.npz"

REPORTS_DIR = OUTPUTS / "reports"
REPORTS_FULL_DIR = REPORTS_DIR / "full"
REPORTS_SMOKE_DIR = REPORTS_DIR / "smoke"

SMOKE_ROOT = OUTPUTS / "smoke"

ERROR_CSV = REPORTS_DIR / "error.csv"
RUNTIME_CSV = REPORTS_DIR / "runtime_summary.csv"


def resolve_train_output(full_run: bool, limit):
    """
    train 출력 경로를 mode별로 분리한다. smoke와 full이 같은 파일을 쓰지 않게 한다.

    Returns dict: {mode, tag, model_npz, reports_dir}
    - full_run=True  → full : MODEL_NPZ, reports/full/
    - full_run=False → smoke: smoke/train_limit{N}/position_bin_stats.npz, reports/smoke/
    """
    if full_run:
        return {
            "mode": "full",
            "tag": "full",
            "model_npz": MODEL_NPZ,
            "reports_dir": REPORTS_FULL_DIR,
        }
    if limit is None:
        raise ValueError("smoke 모드는 limit 값이 필요합니다.")
    tag = f"train_limit{limit}"
    base = SMOKE_ROOT / tag
    return {
        "mode": "smoke",
        "tag": tag,
        "model_npz": base / "position_bin_stats.npz",
        "reports_dir": REPORTS_SMOKE_DIR,
    }
