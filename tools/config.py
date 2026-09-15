from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "link.yaml"

_cache: dict[str, Any] | None = None


def load_config(path: str | Path | None = None, reload: bool = False) -> dict[str, Any]:
    global _cache
    if _cache is not None and not reload and path is None:
        return _cache

    p = Path(path) if path else DEFAULT_CONFIG
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if path is None and not reload:
        _cache = cfg
    return cfg


def get(section: str, key: str | None = None, default=None):
    cfg = load_config()
    sec = cfg.get(section, {})
    if key is None:
        return sec
    return sec.get(key, default)


def record_dir() -> Path:
    d = ROOT / get("paths", "record_dir", "data/recordings")
    d.mkdir(parents=True, exist_ok=True)
    return d
