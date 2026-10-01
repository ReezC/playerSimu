from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "link.yaml"
LIVE_CONFIG = ROOT / "config" / "live.yaml"   # 实时预览参数（独立文件，避免重写 link.yaml 丢注释）

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


def load_live() -> dict[str, Any]:
    """读实时预览参数（config/live.yaml），不存在返回空 dict。"""
    if LIVE_CONFIG.exists():
        with open(LIVE_CONFIG, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_live(cfg: dict[str, Any]):
    """写实时预览参数到 config/live.yaml（**整文件覆盖**）。"""
    with open(LIVE_CONFIG, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def update_live(**kw) -> dict[str, Any]:
    """改 `config/live.yaml` 里的**几个键**，其余原样保留。

    **为什么要有它**：`save_live()` 是整文件覆盖 —— 直接
    `save_live({...几个键})` 会把别处写进去的键（`perf_log`、
    `perf_keepalive`）**一起抹掉**，而且不报错。现象很绕：
    「设置里明明开着，重启后文件里没了、选项又变回默认」。
    凡是要改这份配置，一律走这里。
    """
    cfg = load_live()
    cfg.update(kw)
    save_live(cfg)
    return cfg


def record_dir() -> Path:
    d = ROOT / get("paths", "record_dir", "data/recordings")
    d.mkdir(parents=True, exist_ok=True)
    return d
