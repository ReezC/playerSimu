from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "link.yaml"
LIVE_CONFIG = ROOT / "config" / "live.yaml"   # 实时预览参数（独立文件，避免重写 link.yaml 丢注释）
LINK_CONFIG = ROOT / "config" / "link.yaml"   # 链路配置（注释密集 ⇒ 只许**逐行**改，见 update_link ✓）

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


def load_link(path: str | Path | None = None) -> dict[str, Any]:
    """**重新读一遍** `config/link.yaml`（不吃 `load_config()` 那份进程缓存 ✓）。

    ⚠ 为什么单独一个函数 ✗：`load_config()` 是**惰性缓存**（第一次读进来就留着 ✓）——
      设置窗口刚改完文件，再 `get()` 读到的还是**旧值** ✗（"改了不生效、重启才对"
      就是这么来的 ✓）。
    """
    p = Path(path) if path else LINK_CONFIG
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _fmt_yaml_scalar(v) -> str:
    """把一个 Python 值写成 YAML 标量（够 `link.yaml` 用 ✓ 不追求通用 ✗）。"""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    s = str(v)
    if s == "":
        return '""'
    # 带引号最省心（值里可能有 `:` / `#` / 首尾空格 ✓ 不引会被 YAML 解释成别的 ✗）
    return '"%s"' % s.replace("\\", "\\\\").replace('"', '\\"')


def update_link(path: str | Path | None = None, **kw) -> dict[str, Any]:
    """改 `config/link.yaml` 里的**几个键** ⇒ **注释与排版一个字节都不动** ✓。

    键写成 `"section.key"`（如 `stream.width` / `kbd.serial_local` ✓）；顶层标量直接写键名
    （`a_host` / `b_host` ✓）。

    ⚠⚠ 为什么**不能**像 `live.yaml` 那样"读进来 → 整文件 `safe_dump`" ✗：
      `link.yaml` 是**注释密集的说明书**（每一格都写着"为什么这么填" ✓ 比如
      "cell 必须 ≥ 16px…" / "A 机屏幕 → 流画面的缩放比…" ✓）—— 整写一次，**注释全没了** ✗✗
      （`LIVE_CONFIG` 之所以单独成文件，就是为了躲这件事 ✓ 见它上面那行注释 ✓）。
    ⇒ 走**逐行外科手术**：只替换 `section: key` 那一行的**值** ✓；行尾注释、缩进、
      其它行**原样保留** ✓；值没变 ⇒ **不写文件** ✓（免得白白改 mtime ✓）。
    """
    p = Path(path) if path else LINK_CONFIG
    lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
    changed: dict[str, Any] = {}

    def _split(line: str):
        """`"  key: value   # 注释\\n"` ⇒ `(indent, key, value, tail) | None`。"""
        body = line.rstrip("\r\n")
        eol = line[len(body):]
        if not body.strip() or body.lstrip().startswith("#"):
            return None
        cut = body.find("#")
        head = body[:cut] if cut >= 0 else body
        # ⚠⚠ **值后面那个空格必须留着** ✗：YAML 里 `#` 只有在"前面是空白"时才算注释 ⇒
        #   写成 `"COM9"# 说明` 会把**注释并进值里** ✗（读回来变成 `COM9"# 说明` 之类 ✓
        #   2026-10-09 实测踩到：`samples` 读回来是 `50# 采样次数` ✗）。
        tail = (((" " if cut > 0 else "") + body[cut:]) if cut >= 0 else "") + eol
        if ":" not in head:
            return None
        indent = head[:len(head) - len(head.lstrip())]
        k, _, v = head.strip().partition(":")
        return indent, k.strip(), v.strip(), tail

    for dotted, new in kw.items():
        sec, _, key = dotted.partition(".")
        key = key or sec                       # 顶层标量：键名就是自己 ✓
        _in_sec = (sec != key)                 # False = 顶层标量 ✓
        _hit = False
        _sec_at = -1
        for i, line in enumerate(lines):
            parsed = _split(line)
            if parsed is None:
                continue
            indent, k, old_v, tail = parsed
            if _in_sec:
                # 段落头：**顶层**、无值的那一行（`stream:` ✓）
                if indent == "" and old_v == "" and k == sec:
                    _sec_at = i
                    continue
                if _sec_at < 0 or i < _sec_at:
                    continue
                # 出了这一段（又遇到顶层的行）⇒ 收工 ✓
                if indent == "" and i > _sec_at and old_v != "":
                    break
                if indent == "" and old_v == "" and k != sec:
                    break
                if indent == "" and old_v == "" and k == sec:
                    continue
                if k != key or indent == "":
                    continue
            else:
                if indent != "" or k != key:
                    continue
            _hit = True
            if old_v == _fmt_yaml_scalar(new) or old_v == str(new):
                break                          # 值没变 ⇒ 一个字都不动 ✓
            lines[i] = ("%s%s: %s%s" % (indent, k, _fmt_yaml_scalar(new), tail))
            changed[dotted] = new
            break
        if not _hit:
            # 没找到 ⇒ 追加：有段落就补在该段落末尾 ✓，顶层标量就补在文件末尾 ✓
            add = "%s%s: %s\n" % ("  " if _in_sec else "", key, _fmt_yaml_scalar(new))
            if _in_sec and _sec_at >= 0:
                j = _sec_at + 1
                while j < len(lines) and _split(lines[j]) is not None \
                        and _split(lines[j])[0] != "":
                    j += 1
                lines.insert(j, add)
            else:
                if lines and not lines[-1].endswith("\n"):
                    lines[-1] += "\n"
                lines.append(add)
            changed[dotted] = new

    if changed:
        p.write_text("".join(lines), encoding="utf-8")
    # ⚠ **顺手把进程缓存刷掉**（`load_config` 是惰性的 ✓）：不刷的话，文件改了、
    #   可程序里 `get()` 还是旧值 ⇒ "改了不生效、重启才对" ✗（2026-10-09 ✓）。
    if p == LINK_CONFIG:
        load_config(reload=True)
    cfg = load_link(p)
    return cfg


def record_dir() -> Path:
    d = ROOT / get("paths", "record_dir", "data/recordings")
    d.mkdir(parents=True, exist_ok=True)
    return d
