"""**A 机屏幕 / DPI 的唯一算法**（用户 2026-10-17 ✓「分辨率设置」卡片 ✓）。

用户原话（这是整件事的口径）：
    「填写 A 机的分辨率、DPI 缩放比例，**需要用到的地方（包括 B 机如果需要这些数据的话）
      一律走取这个配置的自计算**」✓

⇒ 数字只有一份来源，**别再手填派生值** ✗：
    · 配置：`config/link.yaml` 的 `[screen]`（`width` / `height` / `dpi_scale` ✓
      **双机共用** ✓ A、B 读同一份 ✓）；
    · **逻辑分辨率** = 物理 ÷ `dpi_scale`（Windows 显示缩放 150% ⇒ ÷1.5 ✓）；
    · **输出缩放** = **流分辨率宽 ÷ 逻辑分辨率宽** ✓
      —— 这个数以前是手填 `probe.out_scale` ✗（0.8004 ✓ 改分辨率就忘 ✓），
      现在**一律现算** ✓（探针卡上只显示结果、不许改 ✗）。

⚠ 谁要用这几个数 ⇒ **import 这个模块** ✗ 不要各自再算一遍
  （`deploy` / `probe_gen` / 界面显示各算一遍 ⇒ 迟早不一致 ✓ —— 用户点名要"自计算"✓）。
⚠ 全部**纯函数 + 读配置**：B 机不跑部署台也能用（它读 `link.yaml` 就够了 ✓）。
"""
from __future__ import annotations

from .config import load_link

#: 配置缺失时的回落（= 现在这台 A 机的真实值 ✓ 改机器就改 `link.yaml [screen]` ✓）
DEFAULTS = {"width": 2560.0, "height": 1440.0, "dpi_scale": 1.5}


def _num(value, default):
    try:
        got = float(value)
    except (TypeError, ValueError):
        return float(default)
    return got


def cfg():
    """`[screen]` 的三个数 ⇒ `{"width", "height", "dpi_scale"}` ✓（缺什么都补默认 ✓）。"""
    try:
        sec = (load_link() or {}).get("screen") or {}
    except Exception:                          # noqa: BLE001 —— 配置坏了也别炸 ✓
        sec = {}
    dpi = _num(sec.get("dpi_scale"), DEFAULTS["dpi_scale"])
    return {"width": _num(sec.get("width"), DEFAULTS["width"]),
            "height": _num(sec.get("height"), DEFAULTS["height"]),
            "dpi_scale": dpi if dpi > 0 else 1.0}


def logical(width=None, height=None, dpi=None):
    """**逻辑分辨率** `(w, h)`（= 物理 ÷ dpi_scale ✓ 取整 ✓）。"""
    s = cfg()
    w = _num(width if width is not None else s["width"], DEFAULTS["width"])
    h = _num(height if height is not None else s["height"], DEFAULTS["height"])
    d = _num(dpi if dpi is not None else s["dpi_scale"], 1.0)
    d = d if d > 0 else 1.0
    return max(1, int(round(w / d))), max(1, int(round(h / d)))


def stream_size():
    """流分辨率 `(w, h)`（读 `link.yaml [stream]` ✓ 没有就 1366×768 ✓）。"""
    try:
        sec = (load_link() or {}).get("stream") or {}
    except Exception:                          # noqa: BLE001
        sec = {}
    return (int(_num(sec.get("width"), 1366)), int(_num(sec.get("height"), 768)))


def out_scale(stream_w=None, width=None, height=None, dpi=None):
    """**输出缩放** = 流宽 ÷ 逻辑宽 ✓（探针画图 / 任何"流↔屏幕"换算都走这里 ✓）。

    ⚠ 宽、高两边的比**必须一致**（同一块画面的等比缩放 ✓）：不一致说明推流宽高比
      和屏幕宽高比对不上（比如推流被裁过 ✓）⇒ 以**宽**为准（探针沿 x 排 ✓
      高度那边差一点点只会让方块略扁，不影响解码 ✓）。
    """
    w = int(_num(stream_w if stream_w is not None else stream_size()[0], 1366))
    s = cfg()
    pw = _num(width if width is not None else s["width"], DEFAULTS["width"])
    d = _num(dpi if dpi is not None else s["dpi_scale"], 1.0)
    d = d if d > 0 else 1.0
    # ⚠ **别用取整后的逻辑宽** ✗：2560/1.5 = 1706.67 ⇒ 取整 1707 ⇒ 0.8002，
    #   而历史标定值一直是 **0.8004**（= 1366 / 1706.67 ✓）。差 0.02% 会让探针整体偏
    #   零点几像素 ✓（解得出但没必要 ✓）⇒ 这里用**浮点逻辑宽** ✓ 与老口径一致 ✓。
    lw = pw / d
    if lw <= 0 or w <= 0:
        return 1.0
    return round(float(w) / float(lw), 4)


def summary():
    """一行给人和界面看的说明 ✓（"逻辑 1706×960 ⇒ 输出缩放 0.8004"✓）。"""
    s = cfg()
    lw, lh = logical()
    sw, _sh = stream_size()
    return ("屏幕 %.0f×%.0f @ %.0f%% ⇒ 逻辑 %d×%d，流 %d×%d ⇒ 输出缩放 **%.4f**"
            % (s["width"], s["height"], s["dpi_scale"] * 100.0, lw, lh, sw,
               stream_size()[1], out_scale()))
