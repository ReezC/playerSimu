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


def load_size():
    """读配置里的字号；文件不存在或损坏时退回默认值。"""
    try:
        with open(CFG, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return clamp(data.get("font_size", DEFAULT_SIZE))
    except Exception:
        return DEFAULT_SIZE


def save_size(v):
    """保存字号，返回实际生效的值（可能被夹到允许范围内）。"""
    v = clamp(v)
    try:
        CFG.parent.mkdir(parents=True, exist_ok=True)
        with open(CFG, "w", encoding="utf-8") as f:
            yaml.safe_dump({"font_size": v}, f, allow_unicode=True)
    except Exception:
        pass
    return v


def apply(app, size=None):
    """把字号应用到整个程序。

    app  : QApplication 实例
    size : 字号；None 表示读配置
    """
    from PyQt5.QtGui import QFont

    size = clamp(size if size is not None else load_size())
    app.setFont(QFont(FAMILY, size))
    return size
