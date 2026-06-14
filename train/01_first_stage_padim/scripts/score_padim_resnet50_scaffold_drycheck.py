"""
ResNet50 scaffold score 배선 dry-check (P1 단계).

score 경로가 model_npz / score_dir / raw_feature_dim 까지 일관되게 연결되는지
"실행 없이" 확인한다.

금지/미실행 (확인만):
  - model_npz 로드 / model forward / lesion scoring 없음
  - threshold 계산 없음
  - FeatureExtractorScaffold 인스턴스화 없음 (weight 로드 회피)

확인 항목:
  - run_tag_paths 로 score 입력(model_npz)/출력(scores_dir) 경로 도출 및 분리
  - PaDiMModelResNet50Scaffold(raw_feature_dim=1792) 생성자 배선
  - 기존 padim_v2_roi0_0 경로와 미충돌

실행:
  source ~/ai_env/bin/activate && python scripts/score_padim_resnet50_scaffold_drycheck.py
"""

from __future__ import annotations

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
PROTECTED_SUBSTR = "padim_v2_roi0_0"


def main() -> int:
    print("[score-drycheck] ResNet50 scaffold score 배선 확인 시작 (실행 없음)")

    raw_dim = get_raw_feature_dim("resnet50")
    paths = resolve_run_tag_paths(RUN_TAG, repo_root=str(REPO_ROOT))

    print(f"[score-drycheck] 입력 model_npz: {paths['model_npz']}")
    print(f"[score-drycheck] 출력 scores_dir: {paths['scores_dir']}")
    for key in ("model_npz", "scores_dir", "evaluation_dir"):
        assert RUN_TAG in paths[key]
        assert PROTECTED_SUBSTR not in paths[key]

    # PaDiM scaffold 생성자 배선만 확인 (npz 로드/forward/scoring 없음)
    model = PaDiMModelResNet50Scaffold(
        selected_feature_indices_path=paths["selected_feature_indices"],
        feature_dim=100,
        raw_feature_dim=raw_dim,
        eps=1e-5,
    )
    assert model.raw_feature_dim == 1792
    print("[score-drycheck] PaDiMModelResNet50Scaffold 배선 OK "
          f"(raw_feature_dim={model.raw_feature_dim})")

    print("[score-drycheck] PASS (npz 로드/forward/scoring/threshold 미실행)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
