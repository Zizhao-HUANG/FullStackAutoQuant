"""Configuration loading module.

Provides YAML and env-var based config reading, defaults to `config/settings.yaml`
and `config/env.template` (env var template).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values


class ConfigError(Exception):
    """Configuration loading error."""


class ConfigLoader:
    """Read and merge YAML config with environment variables."""

    def __init__(self, base_dir: Path | str | None = None) -> None:
        self.base_dir = Path(base_dir or Path(__file__).resolve().parents[1])
        self.config_path = self.base_dir / "config" / "settings.yaml"
        self.env_path = self.base_dir / ".env"

    def load(self) -> dict[str, Any]:
        base_config = self._load_yaml(self.config_path, required=False)
        env_values = self._load_env(self.env_path)
        merged = self._merge(base_config, env_values)
        return merged

    def _load_yaml(self, path: Path, required: bool = True) -> dict[str, Any]:
        if not path.exists():
            if required:
                raise ConfigError(f"Missing config file: {path}")
            return {}
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    def _load_env(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        data = dotenv_values(path)
        return {k: v for k, v in data.items() if v is not None}

    def _merge(self, config: dict[str, Any], env: dict[str, Any]) -> dict[str, Any]:
        merged = dict(config)
        if "tushare" not in merged:
            merged["tushare"] = {}
        if "TUSHARE" in env:
            merged["tushare"]["token"] = env["TUSHARE"]
        merged.setdefault("paths", {})
        paths = merged["paths"]
        base_paths = {
            "database": "../data/trade.db",
            "logs_dir": "../data/logs",
            "targets_dir": "../../trading/logs",
            "trading_dir": "../trading",
            "trading_config": "../trading/config.local.yaml",
            "ranked_csv": "../../ModelInferenceBundle/ranked_scores_AUTO_via_qlib.csv",
            "manual_logs_dir": "../trading/logs/manual",
        }
        for key, value in base_paths.items():
            paths.setdefault(key, value)
        merged["env"] = env
        return merged


def ensure_default_config(base_dir: Path | str | None = None) -> None:
    """If no config file exists, copy default from template."""

    base_dir = Path(base_dir or Path(__file__).resolve().parents[1])
    settings_path = base_dir / "config" / "settings.yaml"
    env_path = base_dir / ".env"

    template_settings = base_dir / "config" / "settings.template.yaml"
    template_env = base_dir / "config" / "env.template"

    if not settings_path.exists() and template_settings.exists():
        settings_path.write_text(template_settings.read_text(encoding="utf-8"), encoding="utf-8")

    if not env_path.exists() and template_env.exists():
        env_path.write_text(template_env.read_text(encoding="utf-8"), encoding="utf-8")
