from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    def merge(base, override):
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                merge(base[key], value)
            else:
                base[key] = value
        return base

    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("extends", None)
    base = load_config(path.parent / parent) if parent else {}
    return merge(base, raw)
