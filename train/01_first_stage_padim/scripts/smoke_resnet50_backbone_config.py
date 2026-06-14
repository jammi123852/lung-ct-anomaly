"""
ResNet50 backbone scaffold 중앙 dry-check (P1 단계).

확인 항목 (forward / weight 로드 / scoring / training 전혀 없음):
  - G2: backbone별 raw_feature_dim(448/1792), layer 채널 합산 일치
  - G4: (backbone, pretrain_source) 무효 조합 검증 (resnet18+radimagenet 차단)
  - G5: normalization_profile=imagenet 기본값이 원본 ImageNet 결과와 동일,
        radimagenet_recommended placeholder 기본 사용 차단
  - G6: run tag 기반 경로 단일 도출, 기존 padim_v2_roi0_0 과 미충돌,
        ResNet50 selected_feature_indices 경로는 아직 미존재(생성 금지)

이 스크립트는 FeatureExtractorScaffold 를 인스턴스화하지 않는다(=weight 로드 없음).
torch 도 import 하지 않는다. 순수 registry/경로/numpy 전처리만 사용한다.

실행:
  source ~/ai_env/bin/activate && python scripts/smoke_resnet50_backbone_config.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from position_aware_padim.backbone_registry_scaffold import (  # noqa: E402
    BACKBONE_LAYER_CHANNELS,
    RAW_FEATURE_DIM,
    SCAFFOLD_RUN_TAGS,
    get_raw_feature_dim,
    validate_backbone_pretrain,
    validate_normalization_profile,
)
from position_aware_padim.preprocessing import preprocess_ct_slice  # noqa: E402
from position_aware_padim.preprocessing_scaffold import (  # noqa: E402
    imagenet_constants_match_original,
    preprocess_ct_slice_scaffold,
)
from position_aware_padim.run_tag_paths_scaffold import (  # noqa: E402
    resolve_run_tag_paths,
    selected_indices_exists,
)

PROTECTED_SUBSTR = "padim_v2_roi0_0"


def check_dims() -> dict:
    assert get_raw_feature_dim("resnet18") == 448
    assert get_raw_feature_dim("resnet50") == 1792
    for bb, dim in RAW_FEATURE_DIM.items():
        assert sum(BACKBONE_LAYER_CHANNELS[bb]) == dim, (bb, dim)
    return {"resnet18": 448, "resnet50": 1792, "channel_sum_match": True}


def check_combos() -> dict:
    # 허용 조합
    validate_backbone_pretrain("resnet18", "imagenet")
    validate_backbone_pretrain("resnet50", "imagenet")
    validate_backbone_pretrain("resnet50", "radimagenet")
    # 무효 조합 차단 확인
    blocked = False
    try:
        validate_backbone_pretrain("resnet18", "radimagenet")
    except ValueError:
        blocked = True
    assert blocked, "resnet18+radimagenet 조합이 차단되지 않음"
    return {"valid_pass": True, "resnet18_radimagenet_blocked": True}


def check_normalization() -> dict:
    # 상수 동일성
    assert imagenet_constants_match_original() is True
    # 합성 2D array 로 원본과 byte-identical 확인 (모델/forward 아님, 순수 numpy 전처리)
    rng = np.random.RandomState(0)
    synth = (rng.rand(8, 8).astype(np.float32) * 1200.0 - 1000.0)  # HU 유사 범위
    out_orig = preprocess_ct_slice(synth)
    out_scaf = preprocess_ct_slice_scaffold(synth, normalization_profile="imagenet")
    assert np.array_equal(out_orig, out_scaf), "imagenet profile 결과가 원본과 다름"
    # placeholder 기본 사용 차단
    blocked = False
    try:
        validate_normalization_profile("radimagenet_recommended", allow_placeholder=False)
    except ValueError:
        blocked = True
    assert blocked, "radimagenet_recommended placeholder 가 차단되지 않음"
    return {
        "imagenet_constants_match": True,
        "imagenet_output_identical": True,
        "radimagenet_placeholder_blocked": True,
    }


def check_paths() -> dict:
    result = {}
    for tag in SCAFFOLD_RUN_TAGS:
        paths = resolve_run_tag_paths(tag, repo_root=str(REPO_ROOT))
        # 모든 경로에 run_tag 포함, 기존 padim_v2_roi0_0 과 미충돌
        for key in ("models_dir", "scores_dir", "evaluation_dir", "reports_dir",
                    "model_npz", "selected_feature_indices"):
            assert tag in paths[key], (tag, key, paths[key])
            assert PROTECTED_SUBSTR not in paths[key], (tag, key, paths[key])
        # selected_feature_indices 는 아직 미존재여야 함 (생성 금지)
        exists = selected_indices_exists(tag, repo_root=str(REPO_ROOT))
        assert exists is False, f"{tag} selected_feature_indices 가 이미 존재함"
        result[tag] = {
            "no_collision_with_padim_v2_roi0_0": True,
            "selected_indices_exists": exists,
        }
    return result


def main() -> int:
    print("[smoke] ResNet50 backbone scaffold dry-check 시작 (forward/weight 로드 없음)")
    dims = check_dims()
    print(f"[smoke] G2 dims: {dims}")
    combos = check_combos()
    print(f"[smoke] G4 combos: {combos}")
    norm = check_normalization()
    print(f"[smoke] G5 normalization: {norm}")
    paths = check_paths()
    print(f"[smoke] G6 paths: {paths}")
    print("[smoke] 전체 dry-check PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
