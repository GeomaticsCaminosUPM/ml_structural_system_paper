"""YAML config loading with an ``extends:`` inheritance mechanism."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .exceptions import ConfigError


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` on top of ``base``."""
    result = dict(base)
    for key, value in override.items():
        base_value = result.get(key)
        if key in result and isinstance(base_value, dict) and isinstance(value, dict):
            result[key] = deep_merge(base_value, value)
        elif (
            isinstance(base_value, list)
            and isinstance(value, dict)
            and set(value.keys()) <= {"exclude"}
        ):
            exclude = set(value.get("exclude") or [])
            result[key] = [item for item in base_value if item not in exclude]
        else:
            result[key] = value
    return result


def load_yaml_with_extends(config_path: str | Path, _chain: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load a YAML file, resolving any ``extends:`` chain into a single merged dict."""
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")
    if config_path in _chain:
        cycle = " -> ".join(str(p) for p in (*_chain, config_path))
        raise ConfigError(f"Circular 'extends' chain: {cycle}")

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    extends = raw.pop("extends", None)
    if not extends:
        return raw

    base_path = (config_path.parent / extends).resolve()
    base = load_yaml_with_extends(base_path, _chain=(*_chain, config_path))
    return deep_merge(base, raw)
