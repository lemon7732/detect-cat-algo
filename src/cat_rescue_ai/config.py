"""Configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from cat_rescue_ai.exceptions import ConfigError


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config must be a mapping: {config_path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    config = load_yaml(path)
    parent = config.get("global_config")
    if parent:
        base = load_yaml(parent)
        config = deep_merge(base, config)
    return config


def resolve_path(config: dict[str, Any], *keys: str, default: str | None = None) -> Path:
    cursor: Any = config
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            if default is None:
                raise ConfigError(f"Missing config path: {'/'.join(keys)}")
            return Path(default)
        cursor = cursor[key]
    return Path(str(cursor))
