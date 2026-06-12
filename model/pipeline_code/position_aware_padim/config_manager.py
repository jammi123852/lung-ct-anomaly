import shutil
from pathlib import Path
from typing import Any

import yaml


REQUIRED_KEYS = {
    "paths": ["normal_training_ready"],
    "model": ["backbone", "feature_layers", "position_bins"],
    "scoring": ["slice_aggregation", "covariance_epsilon", "fallback_strategy"],
    "output": ["top_n_candidates", "full_export_heatmaps", "cache_features"],
}


class ConfigManager:
    def __init__(self, repo_root: str | None = None):
        if repo_root is None:
            repo_root = str(Path(__file__).resolve().parents[2])
        self.repo_root = Path(repo_root)
        self.configs_dir = self.repo_root / "configs"
        self.config: dict[str, Any] = {}

    def load_config(self, paths_yaml: str = "paths.local.yaml") -> dict[str, Any]:
        file_map = {
            "paths": paths_yaml,
            "model": "model.yaml",
            "scoring": "scoring.yaml",
            "output": "output.yaml",
        }
        for section, filename in file_map.items():
            path = self.configs_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"설정 파일 없음: {path}")
            with open(path, encoding="utf-8") as fp:
                self.config[section] = yaml.safe_load(fp) or {}
        return self.config

    def validate_config(self) -> None:
        if not self.config:
            raise RuntimeError("load_config()를 먼저 호출하세요.")
        missing = {}
        for section, keys in REQUIRED_KEYS.items():
            section_data = self.config.get(section, {}) or {}
            absent = [k for k in keys if k not in section_data]
            if absent:
                missing[section] = absent
        if missing:
            lines = [f"  [{sec}]: {', '.join(keys)}" for sec, keys in missing.items()]
            raise ValueError("필수 파라미터 누락:\n" + "\n".join(lines))

    def save_snapshot(self, snapshot_dir: str | None = None) -> None:
        """실행 시작 시점 config snapshot을 outputs/.../configs/ 로 복사한다."""
        if snapshot_dir is None:
            snapshot_dir = str(
                self.repo_root / "outputs" / "position-aware-padim-v1" / "configs"
            )
        dst = Path(snapshot_dir)
        dst.mkdir(parents=True, exist_ok=True)
        for yaml_file in self.configs_dir.glob("*.yaml"):
            shutil.copy2(yaml_file, dst / yaml_file.name)

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return (self.config.get(section) or {}).get(key, default)
