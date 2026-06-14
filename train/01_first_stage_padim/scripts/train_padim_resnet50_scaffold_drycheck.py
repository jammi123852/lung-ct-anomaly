"""
ResNet50 scaffold train 배선 dry-check (P1 단계).

train 경로가 backbone/raw_feature_dim/경로 분리까지 일관되게 연결되는지
"실행 없이" 확인한다.

금지/미실행 (확인만):
  - weight 로드 / model forward / feature extraction 없음
    (FeatureExtractorScaffold 는 인스턴스화하지 않는다 = weight 로드 회피)
  - PaDiM training / scoring 없음
  - selected_feature_indices 생성 없음 (경로만 도출)

확인 항목:
  - PaDiMModelResNet50Scaffold(feature_dim=100, raw_feature_dim=1792) 생성자 배선
    (selected_feature_indices 파일 미존재 → 검증/forward 없이 구조만 확인)
  - run_tag_paths 로 train 출력 경로(model_npz, selected_feature_indices, distributions) 도출
  - FeatureExtractorScaffold 클래스/시그니처가 backbone 인자를 받는지 (import 수준 확인)

실행:
  source ~/ai_env/bin/activate && python scripts/train_padim_resnet50_scaffold_drycheck.py
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from position_aware_padim.backbone_registry_scaffold import get_raw_feature_dim  # noqa: E402
from position_aware_padim.padim_model_resnet50_scaffold import (  # noqa: E402
    PaDiMModelResNet50Scaffold,
)
from position_aware_padim.run_tag_paths_scaffold import resolve_run_tag_paths  # noqa: E402

RUN_TAG = "padim_v2_resnet50_imagenet"


def main() -> int:
    print("[train-drycheck] ResNet50 scaffold train 배선 확인 시작 (실행 없음)")

    raw_dim = get_raw_feature_dim("resnet50")
    assert raw_dim == 1792

    paths = resolve_run_tag_paths(RUN_TAG, repo_root=str(REPO_ROOT))
    print(f"[train-drycheck] model_npz 경로: {paths['model_npz']}")
    print(f"[train-drycheck] selected_feature_indices 경로(미생성): "
          f"{paths['selected_feature_indices']}")
    assert RUN_TAG in paths["model_npz"]
    assert RUN_TAG in paths["selected_feature_indices"]

    # PaDiM scaffold 생성자 배선 (파일 미존재 → 검증/forward 없이 구조만)
    model = PaDiMModelResNet50Scaffold(
        selected_feature_indices_path=paths["selected_feature_indices"],
        feature_dim=100,
        raw_feature_dim=raw_dim,
        eps=1e-5,
    )
    assert model.feature_dim == 100
    assert model.raw_feature_dim == 1792
    assert model.selected_feature_indices is None  # 파일 미존재 → None
    assert model.train_summary["raw_feature_dim"] == 1792
    print("[train-drycheck] PaDiMModelResNet50Scaffold 배선 OK "
          f"(feature_dim=100, raw_feature_dim={model.raw_feature_dim})")

    # FeatureExtractorScaffold 는 인스턴스화하지 않고 시그니처만 확인 (weight 로드 회피)
    from position_aware_padim.feature_extractor_resnet50_scaffold import (
        FeatureExtractorScaffold,
    )
    sig = inspect.signature(FeatureExtractorScaffold.__init__)
    assert "backbone" in sig.parameters
    assert "pretrain_source" in sig.parameters
    print("[train-drycheck] FeatureExtractorScaffold 시그니처 OK "
          f"(params={list(sig.parameters)})")

    print("[train-drycheck] PASS (weight 로드/forward/training 미실행)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
