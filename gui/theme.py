"""界面主题：目前只管字号。

**为什么需要这一层**
    字号原本散落在十几处**内联样式**里（写死像素值的那种）。
    内联样式的优先级高于 QApplication 的全局字体，不清理掉的话
    setFont() 改完看起来毫无反应 —— 这正是"想调大字体"最难搞的地方。

    所以策略是：
      1. 内联样式**只保留颜色**，字号交给样式表
      2. 字号由这里按基准值统一推导（缩放时改用 QApplication.setFont，
         因为所有控件都继承它）

**为什么用基准值 + setFont 而不是直接写 QSS**
    setFont 会让所有未指定字号的控件立即跟随，无需重建界面，
    改完当场可见。
"""

from pathlib import Path

import yaml

FAMILY = "Microsoft YaHei UI"

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / "config" / "ui.yaml"

DEFAULT_SIZE = 11
MIN_SIZE = 9
MAX_SIZE = 20


def clamp(v):
    try:
        v = int(v)
    except (TypeError, ValueError):
        return DEFAULT_SIZE
    return max(MIN_SIZE, min(MAX_SIZE, v))


# 可视化默认值（颜色用 RGB hex；vision_width 是视野线宽，像素）
VIS_DEFAULTS = {
    "mob_color": "#34a853",       # 怪物框 绿
    "player_color": "#4285f4",    # 玩家框 蓝
    "lock_color": "#ff0000",      # 锁定目标框 红
    "attack_color": "#f9ab00",    # 最大攻击距离线 黄
    "min_attack_color": "#ffa500",  # 最小攻击距离 / 规避范围线 橙
    "vision_color": "#5f6368",    # 视野线 深灰
    "vision_width": 1,            # 视野线宽
}

_VIS_COLOR_KEYS = ("mob_color", "player_color", "lock_color",
                   "attack_color", "min_attack_color", "vision_color")


def _load():
    try:
        with open(CFG, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _save(data):
    try:
        CFG.parent.mkdir(parents=True, exist_ok=True)
        with open(CFG, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True)
    except Exception:
        pass


def load_size():
    """读配置里的字号；文件不存在或损坏时退回默认值。"""
    return clamp(_load().get("font_size", DEFAULT_SIZE))


def save_size(v):
    """保存字号，返回实际生效的值（可能被夹到允许范围内）。"""
    v = clamp(v)
    data = _load()
    data["font_size"] = v
    _save(data)
    return v


def _valid_color(v):
    return (isinstance(v, str) and v.startswith("#") and len(v) == 7
            and all(c in "0123456789abcdefABCDEF" for c in v[1:]))


def load_vis():
    """读可视化配置，返回 dict（颜色 + 线宽，含默认值兜底）。"""
    vis = _load().get("vis") or {}
    out = dict(VIS_DEFAULTS)
    for k in _VIS_COLOR_KEYS:
        if _valid_color(vis.get(k)):
            out[k] = vis[k]
    try:
        out["vision_width"] = max(1, min(10, int(vis.get("vision_width", 1))))
    except (TypeError, ValueError):
        out["vision_width"] = 1
    return out


def save_vis(cfg):
    """保存可视化配置。cfg 为 {key: value} 的子集。"""
    data = _load()
    vis = dict((data.get("vis") or {}))
    vis.update(cfg)
    data["vis"] = vis
    _save(data)


def hex_to_bgr(hexstr):
    """'#rrggbb' → (b, g, r)，供 cv2 绘制用。"""
    hexstr = hexstr.lstrip("#")
    return (int(hexstr[4:6], 16), int(hexstr[2:4], 16), int(hexstr[0:2], 16))


def apply(app, size=None):
    """把字号应用到整个程序。

    app  : QApplication 实例
    size : 字号；None 表示读配置
    """
    from PyQt5.QtGui import QFont

    size = clamp(size if size is not None else load_size())
    app.setFont(QFont(FAMILY, size))
    return size
