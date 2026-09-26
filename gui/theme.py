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
    # 「当前任务 / 定时任务」那几行的字色（贴在实时画面上，**不铺底色**，只有描边）——
    # 2026-09-26 用户要求"文字颜色在设置里配"。默认取亮黄：它要压在游戏画面上，
    # 暗色读不清（这里没有黑底可衬）。
    timer_color="#ffeb3b",
)

#: 参与「颜色合法性校验」的键：类别框色 + 辅助线色（vision_width 是数值，不在内）
_VIS_COLOR_KEYS = CLASS_COLOR_KEYS + ("lock_color", "attack_color",
                                      "min_attack_color", "vision_color",
                                      "timer_color")


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


# ---- foothold 集合编辑器的线条宽度（gui/zone_editor.py）----
#
# 单位是**屏幕像素**，不是场景单位：编辑器的缩放跨度很大（整图 fit 时约 0.35 倍、
# 放大看细节能到 8 倍），按场景单位给宽度的话"同一个值"在两种视图下差 20 多倍 ——
# 要么整图时看不见线、要么放大时线糊成一片。所以这里定的是"屏幕上多粗"，
# 编辑器每次重绘都按当前缩放折算（见 gui/zone_editor._ZoneView.set_line_width）。
FOOTHOLD_W_DEFAULT = 1
FOOTHOLD_W_MIN = 1
FOOTHOLD_W_MAX = 8


def load_foothold_width():
    """读编辑器的线条宽度（屏幕像素，1~8）；文件坏了退回默认。"""
    try:
        v = int(_load().get("foothold_line_w", FOOTHOLD_W_DEFAULT))
    except (TypeError, ValueError):
        v = FOOTHOLD_W_DEFAULT
    return max(FOOTHOLD_W_MIN, min(FOOTHOLD_W_MAX, v))


def save_foothold_width(v):
    """保存编辑器的线条宽度，返回实际生效的值（夹到范围内）。"""
    try:
        v = int(v)
    except (TypeError, ValueError):
        v = FOOTHOLD_W_DEFAULT
    v = max(FOOTHOLD_W_MIN, min(FOOTHOLD_W_MAX, v))
    data = _load()
    data["foothold_line_w"] = v
    _save(data)
    return v


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


def repolish(w):
    """让 `w` 及其子控件按**当前**全局字号重新解析一遍样式。

    **为什么非要有这一步**（实测踩到的坑，2026-09-25）：光调 `app.setFont()`，
    **打过样式表的控件完全不跟**。
      · 没打过样式表：`setFont` 立刻跟随（实测 11 → 18 ✓）
      · 打过样式表（部署台/工作台的主窗口都打了一整份 QSS）：**一动不动** ✗
        实测：父窗口一份 QSS、**里面一个字号都没写**，子控件在 setFont 之后 11 → 11。
    根因是 `QStyleSheetStyle` 在 polish 时会把自己解析出的字体 `setFont` 到控件上 ——
    等于给控件标了"显式字体"，此后全局字体再变它就不理了。

    修法是**把同一个样式表字符串再设一遍**：`setStyleSheet` 会递归 repolish，
    字体就按新的全局字号重新解析了（实测 11 → 18 ✓）。
    它不像"遍历控件逐个 setFont"那样把**有意设过**的字体一起冲掉
    （日志正文那份等宽字体就是有意的，见 deploy/app.py 的 LogPane.apply_font）。
    """
    from PyQt5.QtWidgets import QWidget

    try:
        if w.styleSheet():
            w.setStyleSheet(w.styleSheet())
        for c in w.findChildren(QWidget):
            if c.styleSheet():
                c.setStyleSheet(c.styleSheet())
    except Exception:                    # noqa: BLE001
        pass                             # 刷新失败不该把设置弹窗弄崩


def apply(app, size=None):
    """把字号应用到整个程序 —— **包括已经存在的控件**。

    app  : QApplication 实例
    size : 字号；None 表示读配置

    两步缺一不可：`setFont` 管"新建的控件"和"没打过样式表的控件"，
    `repolish` 管"已经存在的、打过样式表的控件"（见 repolish 的说明）。
    """
    from PyQt5.QtGui import QFont

    size = clamp(size if size is not None else load_size())
    app.setFont(QFont(FAMILY, size))
    # 样式表也可能挂在 QApplication 上，那 topLevelWidgets 里就没有带 QSS 的了
    try:
        if app.styleSheet():
            app.setStyleSheet(app.styleSheet())
    except Exception:                    # noqa: BLE001
        pass
    for w in app.topLevelWidgets():
        repolish(w)
    return size
