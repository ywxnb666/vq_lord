from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _expand_env_value(value: Any) -> Any:
    if isinstance(value, str):
        if _ENV_PATTERN.fullmatch(value.strip()):
            match = _ENV_PATTERN.fullmatch(value.strip())
            name, default = match.groups()
            return os.environ.get(name, default or "")

        def replace(match: re.Match[str]) -> str:
            name, default = match.groups()
            return os.environ.get(name, default or "")

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand_env_value(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _expand_env_value(v) for k, v in value.items()}
    return value


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a JSON object: {config_path}")
    data = _expand_env_value(data)
    data["_config_path"] = str(config_path)
    data["_config_dir"] = str(config_path.resolve().parent)
    return data


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def env_or_config(env_name: str, config: Dict[str, Any], key: str, default: Any = None) -> Any:
    value = os.environ.get(env_name)
    if value is not None and value != "":
        return value
    return config.get(key, default)


def resolve_path(path: str, base_dir: Optional[str] = None) -> Path:
    value = Path(str(path)).expanduser()
    if value.is_absolute() or not base_dir:
        return value
    return Path(base_dir) / value
