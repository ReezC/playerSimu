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

from perception import classes   # 类别表（唯一定义处）：框颜色按它生成

FAMILY = "Microsoft YaHei UI"

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / "config" / "ui.yaml"

DEFAULT_SIZE = 11
MIN_SIZE = 9
MAX_SIZE = 20

#: 设置弹窗的页签栏样式。**两个界面共用这一份**（工作台 gui/settings_dialog.py、
#: 部署台 deploy/settings_dialog.py）。
#:
#: 为什么放这里：主窗口那份 QSS 是 `setStyleSheet()` 打在**主窗口**上的，对话框不
#: 继承 —— 所以每个对话框都得自己带一份。而各写一份必然漂移，最后表现为"两个界面
#: 的设置弹窗长得不一样"，那是最容易让人以为装错版本的那种差异。
TAB_QSS = """
QTabWidget::pane { border: 1px solid #e2e5ea; border-radius: 6px;
                   background: #ffffff; top: -1px; }
QTabBar::tab { background: #f0f2f5; color: #5f6368; padding: 6px 14px;
               border: 1px solid #e2e5ea; border-bottom: none;
               border-top-left-radius: 6px; border-top-right-radius: 6px;
               margin-right: 2px; }
QTabBar::tab:selected { background: #ffffff; color: #202124; font-weight: 600; }
QTabBar::tab:hover { color: #202124; }
"""


def clamp(v):
    try:
        v = int(v)
    except (TypeError, ValueError):
        return DEFAULT_SIZE
    return max(MIN_SIZE, min(MAX_SIZE, v))


# ---- 框颜色：**每个类别一条**，键名 = 类别英文名 + "_color" ----
#
# 直接从类别表（perception/classes.py）生成，所以「设置里能改的颜色」永远等于
# 「存在的类别」—— 以后加一个类别，设置里自动多一行，不会漏掉某一类的框色。
CLASS_COLOR_KEYS = tuple(c[1] + "_color" for c in classes.CLASSES)


def _class_color_defaults():
    """类别框颜色的默认值（RGB hex）—— 就是类别表（perception/classes.py）的最后一列。"""
    return {en: classes.bgr_to_hex(bgr)
            for _cid, en, _zh, bgr in classes.CLASSES if bgr}


# 可视化默认值（颜色用 RGB hex；vision_width 是视野线宽，像素）
VIS_DEFAULTS = dict(
    {"%s_color" % en: v for en, v in _class_color_defaults().items()},
    lock_color="#ff0000",         # 锁定目标框 红
    attack_color="#f9ab00",       # 最大攻击距离线 黄
    min_attack_color="#ffa500",   # 最小攻击距离 / 规避范围线 橙
    vision_color="#5f6368",       # 视野线 深灰
    vision_width=1,               # 视野线宽
)

#: 参与「颜色合法性校验」的键：类别框色 + 辅助线色（vision_width 是数值，不在内）
_VIS_COLOR_KEYS = CLASS_COLOR_KEYS + ("lock_color", "attack_color",
                                      "min_attack_color", "vision_color")


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


def class_colors():
    """{类别 id: (b, g, r)} —— 画框用。**每个类别**的颜色都能在设置里改。

    load_vis() 保证类别框色的键都存在（缺的用默认值补），所以这里直接取。
    """
    vis = load_vis()
    return {cid: hex_to_bgr(vis["%s_color" % en])
            for cid, en, _zh, _bgr in classes.CLASSES}


def class_color(cls):
    """单个类别的框色 (b, g, r)。给命令行工具这类只画一种框的调用方。"""
    return class_colors().get(cls, (255, 255, 255))


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
