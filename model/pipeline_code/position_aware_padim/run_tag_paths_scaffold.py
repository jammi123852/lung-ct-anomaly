"""
ResNet50 scaffold run tag → 출력 경로 단일 도출 헬퍼 (G6).

목적:
- models / scores / evaluation / reports / distributions / selected_feature_indices
  경로를 run tag 한 곳에서 일관되게 도출해 train/score 간 경로 drift를 막는다.
- 기존 padim_v2_roi0_0 (ResNet18) 경로와 절대 섞이지 않는 신규 tag만 생성한다.
- selected_feature_indices 는 ResNet50 전용 경로만 "준비"하며, 실제 index 파일은
  이 모듈에서 생성하지 않는다 (생성 금지). 존재 여부만 확인할 수 있다.

원본 파일은 일절 수정하지 않는다. 이 모듈은 신규 scaffold 전용이다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from .backbone_registry_scaffold import RUN_TAG_TO_COMBO, SCAFFOLD_RUN_TAGS

# 기존 ResNet18 baseline run tag (보호 대상 — 신규 tag가 이와 겹치면 안 됨)
PROTECTED_EXISTING_TAGS = ("padim_v1", "padim_v2_roi0_0")

# 출력 루트 (기존과 동일 루트 사용, tag 하위로만 분기)
_OUTPUT_ROOT_PARTS = ("outputs", "position-aware-padim-v1")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_run_tag_paths(run_tag: str, repo_root: str | None = None) -> Dict[str, str]:
    """
    run tag 하나로 models/scores/evaluation/reports 및 산출 파일 경로를 도출한다.

    Parameters
    ----------
    run_tag : str
        'padim_v2_resnet50_imagenet' 또는 'padim_v2_resnet50_radimagenet'.
    repo_root : str, optional
        repo 루트. 미지정 시 이 파일 기준 2단계 상위.

    Returns
    -------
    dict
        모든 값은 절대경로 문자열. 키:
        run_tag, backbone, pretrain_source,
        models_dir, distributions_dir, model_npz, selected_feature_indices,
        scores_dir, evaluation_dir, reports_dir

    Raises
    ------
    ValueError
        run_tag 가 지원 scaffold tag가 아니거나, 보호 대상 기존 tag와 충돌할 때.
    """
    if run_tag in PROTECTED_EXISTING_TAGS:
        raise ValueError(
            f"run_tag '{run_tag}' 는 기존 ResNet18 baseline 경로입니다. "
            "scaffold는 기존 경로를 재사용/덮어쓰기 할 수 없습니다."
        )
    if run_tag not in SCAFFOLD_RUN_TAGS:
        raise ValueError(
            f"지원하지 않는 scaffold run_tag: '{run_tag}'. "
            f"지원: {list(SCAFFOLD_RUN_TAGS)}"
        )

    backbone, pretrain_source = RUN_TAG_TO_COMBO[run_tag]

    root = Path(repo_root) if repo_root is not None else _repo_root()
    base = root.joinpath(*_OUTPUT_ROOT_PARTS)

    models_dir = base / "models" / run_tag
    distributions_dir = models_dir / "distributions"
    scores_dir = base / "scores" / run_tag / "by_patient"
    evaluation_dir = base / "evaluation" / run_tag
    # 기존 v2 명명 관례(reports_v2_roi0_0 등)와 동일하게 reports_<tag> 형태로 분리
    reports_dir = base / ("reports_" + run_tag)

    return {
        "run_tag": run_tag,
        "backbone": backbone,
        "pretrain_source": pretrain_source,
        "models_dir": str(models_dir),
        "distributions_dir": str(distributions_dir),
        "model_npz": str(distributions_dir / "position_bin_stats.npz"),
        "selected_feature_indices": str(distributions_dir / "selected_feature_indices.npy"),
        "scores_dir": str(scores_dir),
        "evaluation_dir": str(evaluation_dir),
        "reports_dir": str(reports_dir),
    }


def selected_indices_exists(run_tag: str, repo_root: str | None = None) -> bool:
    """ResNet50 전용 selected_feature_indices 파일 존재 여부만 확인 (생성하지 않음)."""
    paths = resolve_run_tag_paths(run_tag, repo_root=repo_root)
    return Path(paths["selected_feature_indices"]).exists()
