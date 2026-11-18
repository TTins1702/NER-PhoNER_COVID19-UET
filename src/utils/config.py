import re
from pathlib import Path
from typing import Any

import yaml

_SCIENTIFIC_FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)[eE][+-]?\d+$")


def _parse_override_value(raw_value: str) -> Any:
    """Parse a CLI override string into bool, int, float, or str."""
    parsed = yaml.safe_load(raw_value)
    # PyYAML implements YAML 1.1, which parses '3e-5' (without a decimal point) as str.
    if isinstance(parsed, str) and _SCIENTIFIC_FLOAT_RE.match(parsed.strip()):
        return float(parsed)
    return parsed


def _set_nested_value(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a nested dictionary key specified in dot notation (e.g. 'model.use_crf')."""
    keys = [part.strip() for part in dotted_key.split(".") if part.strip()]
    if not keys:
        raise ValueError(f"Invalid override key: {dotted_key!r}")

    target = config
    for key in keys[:-1]:
        node = target.get(key)
        if node is None:
            node = {}
            target[key] = node
        elif not isinstance(node, dict):
            raise ValueError(f"Cannot traverse non-dict key {key!r} in override {dotted_key!r}")
        target = node

    target[keys[-1]] = value


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML configuration file and apply optional 'a.b=val' CLI overrides."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Configuration file {path} must contain a YAML mapping")

    if overrides:
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"Override must follow 'key.path=value' format, got {item!r}")
            key_path, raw_val = item.split("=", 1)
            _set_nested_value(config, key_path, _parse_override_value(raw_val))

    return config
