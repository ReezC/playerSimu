# -*- coding: utf-8 -*-
"""鼠标**绝对定位**（断线重连「点服务器 / 点频道」用的那一层）。

分工（与 `decision/reconnect.py` 的模块头同一条纪律 —— **一处实现**）：

    `decision/reconnect.py`   什么时候点、点**画面上的哪个比例位置**（策略）
    `decision/mouse_aim.py`   ① 比例 → 像素 → 指令单位 的**换算**；② 撞角归零 + 走位 + 左键（发输入）
    `decision/input.py`       真正把 `MOVE` / `CLICK` 发到 Pro Micro

**两个出口、同一个换算**（2026-10-09 ✓）：`click_ratio` = 低频"点一下"（每发都撞角归零 ✓）；
`aim_to` + `aim_tick` = 高频"指着目标"（**设目标 + 逐拍小步插值** ⇒ 平滑 ✓ 见那边的说明 ✓）。

**为什么必须有这一层**（`docs/断线重连设计.md` §6.1）：鼠标通道**只有相对位移**
（`input.mouse_move(dx, dy)` ✓ —— 固件是相对 HID 鼠标 ✗ 不是绝对定位设备），
没有"移动到某坐标"这种指令。而选服务器 / 选频道**不能用键盘**（用户实测 ✓），
所以只能：**先撞到已知的角（绝对原点）→ 再按标定好的比例走位 → 点击**。

    1) `MOVE -4000 -4000` ⇒ 光标被 OS 夹到屏幕左上角 **(0, 0)**（已知绝对原点 ✓）
    2) 目标画面像素 P ⇒ 指令单位 = P / gain，发 `MOVE dx dy`
    3) `CLICK LEFT`

**三条前提 / 纪律**（缺一个坐标就是错的，所以宁可不动手 ✗）：

  1. **游戏画面左上角 = 屏幕左上角**（全屏 / 无边框窗口，且在主显示器上）。
     撞角归零落在**屏幕** (0,0)，而目标比例是相对**画面**算的 ⇒ 两者不同源就没法用。
     不是这个情况 ⇒ 这一层**不报错、也不猜**（用户要自己保证，见 tooltip ✓）。
  2. **gain（= 画面像素 / 指令单位）必须标定过**，且必须在**同一路画面**上量的
     （标定走的哪份画面，用的时候就得是那份 —— 流画面被缩放过的话，
     物理像素与流像素差一个比例，混用就差一个比例 ✗）。
     没标定 ⇒ **返回失败 + 人话**，绝不退回 1.0 硬点 ✗（宁可不做，也不乱点 ✓）。
  3. **指针加速必须关**（Windows "提高指针精确度" ✗）：开了之后 count→像素是**非线性**
     的，任何单一 gain 都不成立（标定工具会顺带量线性度 ✓ 见 §6.1）。

**gain 存在哪 / 怎么量**：`config/mouse_gain.json`（`{"gain_x": px/单位, "gain_y": ...}` ✓）
—— 这是**本机（游戏机侧）**的属性（指针速度 + 有没有加速 + 流缩放），与"哪个项目"无关 ⇒
放机器级配置 ✓ 不进 `project.yaml` ✗。写它的工具是 `tools/mouse_aim_calib.py`
（`tools/mouse_gain_calib.py` 的同款"帧差量光标位移"手法，但它能直接从**流**取帧 ✓
—— 本仓库的实时画面来自推流，`live.yaml` 里没有窗口 `rect` ⇒ 老工具那条路走不通 ✗）。

⚠ 与 `perception/lie_controller.load_gain()` **不是同一个读者**：那个是测谎闭环用的，
坏文件时**退回 1.0**（闭环能自己纠 ✓）；这一层是**开环绝对定位**，退回 1.0 就是
点错地方 ✗ ⇒ 这里**缺文件一律返回 None**（判据不同，故意不合并 ✓）。
两个读者读的是**同一个文件、同一套单位**（"px/单位" ✓）。
"""

import threading
import time
from pathlib import Path

from decision import input as dinput

ROOT = Path(__file__).resolve().parent.parent
GAIN_PATH = ROOT / "config" / "mouse_gain.json"

#: 撞角那一下发多大（单位 = 指令 count）。**必须远大于任何屏幕的边长**：
#: OS 会把光标夹在屏幕里，多出来的部分自然被吃掉（发 4000 与发 9999 结果一样 ✓）。
#: ⚠ 固件对 `MOVE` 是**分块发送**的（`mouseMoveChunked`，每块 ≤120 ✓）——
#: 直接把 4000 塞给 `Mouse.move()` 会被截成 signed char（4000 → -96，**方向都反了** ✗），
#: 那个坑已经在固件里修掉了（见 `remote_kbd/pro_micro/pro_micro.ino` ✓）。
CORNER_STEP = 4000


#: 标定文件形状（**v2 ✓ 2026-10-10 起**）：
#:     {"v": 2,
#:      "gain_x": …, "gain_y": …, "ref_w": …, "ref_h": …,      ← **最新那一份**照旧平铺 ✓
#:      "by_frame": {"1366x768": {gain_x, gain_y, ref_w, ref_h, …}, "1918x1079": {…}}}
#:   ⚠ 平铺那层**必须留着** ✗：`perception/lie_controller.load_gain()` 也读这个文件 ✓
#:     （测谎闭环那份 ✓ 它只认 `gain_x`/`gain_y` ✓ 见本模块开头"两个读者"那段 ✓）。
#:   老形状（只有平铺 ✓ 没有 `by_frame`）仍读得出 ✓ ⇒ 当"就只有那一份"✓ 不报错 ✓。
GAIN_V2 = 2

#: 找"这个尺寸的那一份"时允许的**相对差**（2026-10-11 ✓ 现场定案）。
#: ⚠⚠ 为什么必须有它：本机窗口那条量出来是 **1918×1079** ✓，而后来画面是 **1920×1080** ✗
#:   —— **同一路画面、差两个像素** ✓，可我第一版要求**精确相等** ✗ ⇒ 判成"没量过"
#:   ⇒ **一个字节都不点** ⇒ 用户看到的正是「**本地的断线重连不选频道了**」✓
#:   （日志原话：`⛔ 先不点：这个画面尺寸（1920×1080）还没有量过鼠标` ✓）。
#: ⇒ 改成"**最接近的那一份 + 两轴都在这个比例内**"✓：
#:     1920 vs 1918 = **0.1%** ⇒ 命中 ✓；1366 vs 1920 = **28%** ⇒ **不命中** ✓（正确 ✓）。
#: ⚠ **绝不许"随便就近取一个"** ✗ —— 那是拿别的来源的几何来用，正是本仓库明令禁止的猜 ✓。
GAIN_SIZE_TOL = 0.02


def gain_frame_key(frame_shape):
    """帧尺寸 → 那一份标定的键 `"1366x768"` ✓（`frame_shape` = `(h, w, …)` ✓）。"""
    try:
        h, w = int(frame_shape[0]), int(frame_shape[1])
    except (TypeError, ValueError, IndexError):
        return ""
    if w <= 0 or h <= 0:
        return ""
    return "%dx%d" % (w, h)


#: 增益文件的小缓存：`((mtime_ns, size), (flat, by))` ✓ —— **实时回路每帧都在查它** ✗
#: ⚠⚠ 为什么必须缓存（用户 2026-10-17 ✓ 原话：「**自动测谎特别卡 鼠标跟丢了**」✗）：
#:   瞄准那条路**每帧**都要问一次增益（`_aim_geo` → `load_gain` ✓），而一帧里
#:   `load_gain` / `frame_rect` / `screen_gain` 会各读一遍 ⇒ 每秒上百次
#:   **读盘 + JSON 解析** ✗（每条 0.1~0.3ms ✓ 叠起来就是可感的卡顿 ✓）。
#:   ⇒ 按"文件改没改"（mtime + 大小 ✓）缓存 ✓ ⇒ 平时纯内存 ✓ 改了立刻生效 ✓
#:   （不按时间过期 ✗ —— 那样标定刚写完还得等一会儿才生效 ✓ 更容易误判"标定没用"✗）
_GAIN_CACHE = {"key": None, "val": ({}, {})}


def _read_gain_file():
    """⇒ `(平铺那份, {尺寸键: 记录})`（读不出来 ⇒ `({}, {})` ✓）。**按文件状态缓存** ✓。"""
    import json
    try:
        st = GAIN_PATH.stat()
        key = (int(st.st_mtime_ns), int(st.st_size))
    except Exception:                            # noqa: BLE001 —— 没有文件 ⇒ 就是"没标定过" ✓
        return {}, {}
    if _GAIN_CACHE["key"] == key:
        return _GAIN_CACHE["val"]
    try:
        d = json.loads(GAIN_PATH.read_text(encoding="utf-8"))
    except Exception:                            # noqa: BLE001 —— 坏文件当"没标定过" ✓
        return {}, {}
    if not isinstance(d, dict):
        return {}, {}
    by = d.get("by_frame")
    by = dict(by) if isinstance(by, dict) else {}
    flat = dict(d)
    if not by and flat.get("ref_w") and flat.get("ref_h"):
        # 老格式（只有平铺 ✓ 只有一份几何 ✓）⇒ 归到它自己那个尺寸下 ✓
        try:
            by["%dx%d" % (int(flat["ref_w"]), int(flat["ref_h"]))] = flat
        except (TypeError, ValueError):
            pass
    _GAIN_CACHE["key"] = key
    _GAIN_CACHE["val"] = (flat, by)
    return flat, by


def _ok_entry(e):
    """这份记录的增益**可用**吗 ⇒ `(gx, gy)` / `None` ✓。

    ⚠ 两个分量都必须是**正数** ✓：非正说明方向反了或量歪了 ⇒ 当没标定过 ✓
      （`tools/mouse_aim_calib.py` 也是这么判的 ✓）⇒ **绝不退回 1.0** ✗。
    """
    try:
        gx, gy = float(e.get("gain_x")), float(e.get("gain_y"))
    except (TypeError, ValueError, AttributeError):
        return None
    return (gx, gy) if (gx > 0 and gy > 0) else None


def load_gain(frame_shape=None):
    """这一帧尺寸对应的标定 ⇒ `(gain_x, gain_y)`；**这个尺寸没量过 ⇒ None** ✓（绝不退 1.0 ✗）。

    ⚠⚠ 为什么改成「**按帧尺寸存、按尺寸取**」（2026-10-10 ✓ 用户原话："**这个过程不能每次都
      发现错然后跑命令吧 太麻烦了 不能自动吗？**" ✓）：gain 的单位是「**帧像素** / 指令单位」✓
      （见 `counts_for` ✓）⇒ **只要帧尺寸一样，这份标定就一直成立** ✓ ⇒ 那就**每种尺寸各存
      一份** ✓ ⇒ 每条来源**只量一次** ✓，此后在窗口(1918)/收流(1366) 之间来回切**全自动** ✓
      —— 原来**只有一份** ✗ ⇒ 换一种尺寸就把前一份顶掉了 ✓（正是"每次都要重跑命令"的由来 ✓）。
    ⚠ 不给 `frame_shape` ⇒ 给"**最新量的那份**"（平铺那份 ✓）：`aim_available()` 那种只问
      "**量过没有**"的地方用它 ✓（它手上没有帧 ✓）。
    """
    flat, by = _read_gain_file()
    if frame_shape is None:
        for e in (flat, *by.values()):
            g = _ok_entry(e)
            if g:
                return g
        return None
    key = gain_frame_key(frame_shape)
    if not key:
        return None
    if key in by:
        return _ok_entry(by[key])
    # ⚠ 精确相等找不到 ⇒ 找**最接近**的那一份（**两轴都要**落在 `GAIN_SIZE_TOL` 内 ✓）
    #   见 `GAIN_SIZE_TOL` 那段：1918×1079 ↔ 1920×1080 是**同一路画面** ✓ 该命中 ✓；
    #   1366×768 ↔ 1920×1080 差 28% ⇒ **不许命中** ✗（那会拿收流的几何去点本机 ✓）。
    try:
        h, w = int(frame_shape[0]), int(frame_shape[1])
    except (TypeError, ValueError, IndexError):
        return None
    if w <= 0 or h <= 0:
        return None
    best, best_d = None, None
    for e in by.values():
        g = _ok_entry(e)
        if not g:
            continue
        try:
            ew, eh = int(e.get("ref_w") or 0), int(e.get("ref_h") or 0)
        except (TypeError, ValueError):
            continue
        if ew <= 0 or eh <= 0:
            continue
        dw = abs(ew - w) / float(w)
        dh = abs(eh - h) / float(h)
        if dw > GAIN_SIZE_TOL or dh > GAIN_SIZE_TOL:
            continue
        _d = dw + dh
        if best_d is None or _d < best_d:
            best, best_d = g, _d
    return best


def gain_sizes():
    """**量过哪些帧尺寸** ⇒ `["1918x1079", "1366x768"]`（给人话用 ✓ 见 `gain_mismatch` ✓）。"""
    _flat, by = _read_gain_file()
    return list(by.keys())


def gain_ref_frame(frame_shape=None):
    """那一份标定是在**多大**的画面上量的 ⇒ `(w, h)`；没记 / 没量过 ⇒ `None` ✓。

    （给了 `frame_shape` ⇒ 看**那一份**✓；不给 ⇒ 看最新那份 ✓。）
    """
    flat, by = _read_gain_file()
    e = flat if frame_shape is None else (by.get(gain_frame_key(frame_shape)) or {})
    try:
        w, h = e.get("ref_w"), e.get("ref_h")
        return (int(w), int(h)) if w and h else None
    except (TypeError, ValueError):
        return None


def save_gain(gx, gy, frame_shape=None, extra=None):
    """写一份标定 ⇒ 路径 ✓（**按帧尺寸存 ✓ 别的尺寸的那几份原样保留 ✓**）。

    ⚠ 平铺那层一起更新（= 最新那份 ✓ 给"只问量过没有"的老读者 / `lie_controller` ✓
      见上面文件形状那段 ✓）。⚠ 写盘失败**往上抛** ✓（标定写不进 = 白量一次 ✓ 得让人知道 ✓）。
    """
    import json
    body = {"gain_x": round(float(gx), 6), "gain_y": round(float(gy), 6)}
    if extra:
        body.update(extra)
    if frame_shape is not None:
        try:
            h, w = int(frame_shape[0]), int(frame_shape[1])
        except (TypeError, ValueError, IndexError):
            h = w = 0
        if w > 0 and h > 0:
            body["ref_w"], body["ref_h"] = w, h
    flat, by = _read_gain_file()
    out = dict(flat)
    out.pop("by_frame", None)
    out["v"] = GAIN_V2
    out["gain_x"], out["gain_y"] = body["gain_x"], body["gain_y"]
    if "ref_w" in body:
        out["ref_w"], out["ref_h"] = body["ref_w"], body["ref_h"]
    key = gain_frame_key(frame_shape) if frame_shape is not None else ""
    if key:
        by[key] = dict(body)
    out["by_frame"] = by
    GAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GAIN_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return GAIN_PATH


#: 「旧格式标定」那句提醒**每个进程只说一次** ✓（见 `gain_warn` ✓ —— 重连时点击是每拍一发 ✗）
_GAIN_OLD_WARNED = False


#: 帧尺寸与标定基准差**超过这个比例** ⇒ **拒绝点**（不是只提醒 ✗ 见 `gain_mismatch` ✓）。
#:  ⚠ 为什么是"拒绝"而不是"提醒"（2026-10-10 ✓ 用户现场："**现在远程收流的选频道界面鼠标
#:    又点歪了**" ✓）：歪的点击**本身就可能点坏东西** ✗（这套 UI 上「删除角色 / 回到首页」
#:    离频道格并不远 ✓ 同 §9 那条"越界就直接判失败、不夹取"的口径 ✓）⇒ **宁可不点** ✓，
#:    也别给一个"点了但没中"的假动作 ✗（那会让人以为"游戏卡了"✓ 现场就被这个坑过 ✓）。
GAIN_MISMATCH_PCT = 0.05
#: 「为什么这一下没点」每个进程只说一次 ✓（重连点击是每拍一发 ✗ 别刷屏 ✓）。
_MISMATCH_SAID = False


def gain_mismatch(frame_shape):
    """这一帧尺寸**有没有**标定 ⇒ 没有 ⇒ `(True, 人话)`；量过 ⇒ `(False, "")` ✓。

    ⚠ 判据是「**按尺寸找**」✗ 不是"差多少 %"（2026-10-10 改 ✓）：上一版拿"帧宽差 > 5%"
      判 ✓，可它**默认两份画面只差一个缩放** ✗ —— 而「本地窗口（裁切）↔ 收流（整屏缩放）」
      之间根本没这个关系 ✗ ⇒ 比出来的是个**没意义的数** ✓（现场正差在这一点上 ✓）。
      现在直接问"**这个尺寸量过没有**"✓：量过 ⇒ 那份就是准的 ✓；没量过 ⇒ **别点** ✓。
    """
    if frame_shape is None:
        return False, ""
    key = gain_frame_key(frame_shape)
    if not key:
        return False, ""
    if load_gain(frame_shape):
        return False, ""
    _known = gain_sizes()
    return True, ("这个画面尺寸（**%s**）还没有量过鼠标 ⇒ gain 是「帧像素 / 指令单位」口径 ✓"
                  "拿别的尺寸那份来用只会点歪 ✗%s"
                  % (key.replace("x", "×"),
                     ("（这台量过的是：%s ✓）"
                      % "、".join(k.replace("x", "×") for k in _known)) if _known else ""))


def gain_mismatch_block(frame_shape):
    """尺寸没量过 ⇒ `(True, 人话)`（**这一下不要点** ✗）；量过 / 说不准 ⇒ `(False, "")` ✓。"""
    global _MISMATCH_SAID
    bad, why = gain_mismatch(frame_shape)
    if not bad:
        return False, ""
    _known = gain_sizes()
    msg = ("⛔ **先不点**：%s。\n"
           "　修法（**一次性** ✓ 量过之后这个尺寸就一直有效 ✓ 来回切来源都自动 ✓）："
           "**先把实时预览停掉**，在**跑工作台这台机器**上按你现在这条来源量一次 ——\n"
           "　　收流：`python -X utf8 -m tools.mouse_aim_calib --source stream`\n"
           "　　本地窗口：`python -X utf8 -m tools.mouse_aim_calib --source live`%s"
           % (why,
              ("\n　（⚠ 切回**已经量过**的那个尺寸 ⇒ **不用重量** ✓ 会自动用上那一份 ✓）"
               if _known else "")))
    if not _MISMATCH_SAID:
        _MISMATCH_SAID = True
        try:
            from core import behavior              # ⚠ 是 `core.behavior` ✓（别写裸 import ✓）
            behavior.event("mouse_gain_missing", frame=str(gain_frame_key(frame_shape)),
                           known=",".join(_known))
        except Exception:                          # noqa: BLE001 —— 打点坏了别影响判断 ✓
            pass
    return True, msg


#: 自动量一次时**走多少单位**（够大 ⇒ 抗噪声 ✓；够小 ⇒ 稳在画面里、不跑飞 ✓）。
AUTO_STEP = 300
#: 同一帧尺寸**多久内不再自动重试** ✓（失败多半是"看不见光标"✓
#: 每拍都试 = 每拍都挪鼠标 ✗）。
AUTO_RETRY_S = 60.0
#: 帧尺寸键 → 上次自动量的时刻（冷却用 ✓ 见 `AUTO_RETRY_S` ✓）。
_AUTO_LAST = {}


def _restore_cursor(pre, p0, gx_screen=None, gy_screen=None):
    """量完**把光标放回撞角之前那儿** ✓（放不了就算了 ✓ 它是"体贴"不是"正确性" ✓）。

    ⚠⚠ 为什么必须有（2026-10-10 ✓ 用户**两次**报"**A 机鼠标停在左上角不动了**"✗）：
      量一次必然要"撞角归零"✓ —— 量完若**后面那一拍没接着点**（重连已经放弃 / 这一拍被
      别的门挡了 ✓）⇒ 光标就**留在左上角** ✓ ⇒ 人看到的就是"鼠标卡在左上角"✓。
      ⇒ 现在量完主动把它送回原处 ✓（`pre − p0` 两个读数都是现成的 ✓ 不用再问一次 ✓）。
    ⚠⚠ **位移要换成"指令单位"** ✗：`pre − p0` 是**屏幕像素** ✓ 而 `mouse_move` 要的是
      **单位** ✓ ⇒ 必须除以刚量到的**屏幕增益**（`gx_screen/gy_screen` ✓）——
      ⚠ 我第一版直接拿像素当单位发 ✗（1:1 那条路看不出来 ✓ 有加速的机器上就会跑过头 ✓）。
    ⚠ 没量到增益（撞角后就看不见光标了 ✓）⇒ **放弃送回** ✓（宁可不送，也别乱送 ✗）。
    """
    if not pre or not p0 or not gx_screen or not gy_screen:
        return
    try:
        ux = int(round((int(pre[0]) - int(p0[0])) / float(gx_screen)))
        uy = int(round((int(pre[1]) - int(p0[1])) / float(gy_screen)))
        if ux or uy:
            dinput.mouse_move(ux, uy)
    except Exception:                            # noqa: BLE001 —— 放不回去只是不体贴 ✓
        pass


def _probe_cursor():
    """读一次"光标在哪 + 屏幕信息" ⇒ 见 `decision/input.cursor_probe` ✓（本地读 / 远端问 ✓）。"""
    f = getattr(dinput, "cursor_probe", None)
    try:
        return f() if f else None
    except Exception:                            # noqa: BLE001 —— 问不到就当量不了 ✓
        return None


def auto_measure(frame_shape, force=False):
    """**自己把这一帧尺寸的 gain 量出来** ⇒ `(ok, 人话)` ✓（用户 2026-10-10："**两边一起**"✓）。

    三步，全是**现成动作** ✓（不用工具、不用人 ✓）：
      ① `corner_zero()`（撞到**那台机器**虚拟屏左上角 ✓ 见它说明 ✓）⇒ 读光标 `p0` ✓；
      ② 走 `AUTO_STEP` 单位 ⇒ 再读 `p1` ⇒ `gain_screen = (p1 − p0) / 单位` ✓（两轴各算 ✓）；
      ③ 折成「**帧像素** / 单位」：`折算比 = 帧尺寸 ÷ 那台的抓取区域尺寸` ✓ ——
         ⚠ 这两步都必须用**真数** ✗（relay 从 A 机 `config/deploy.json` 读来的抓取区域 ✓
           或者帧尺寸正好等于屏幕尺寸 = 1:1 ✓）；**没有这个数就不写** ✗
           （**拿猜当准**是本仓库明令禁止的 ✓ 见 `gain_mismatch` 那段 ✓）。
    `force=True` ⇒ 无视冷却（给工具 / 用例用 ✓）。
    ⚠ 它会**动鼠标**（撞角 + 走一步 + 走回来 ✓，**不点** ✓）⇒ 只在"这个尺寸没量过"时叫它 ✓
      并且按 `AUTO_RETRY_S` 冷却 ✓（失败别每拍重试 ✗）。
    """
    if frame_shape is None or len(frame_shape) < 2:
        return False, "拿不到画面尺寸"
    key = gain_frame_key(frame_shape)
    _now = time.monotonic()
    if not force and (_now - _AUTO_LAST.get(key, 0.0)) < AUTO_RETRY_S:
        return False, ("刚自动量过（%.0f 秒内不重复试 ✓ —— 免得一直挪鼠标 ✓）" % AUTO_RETRY_S)
    _AUTO_LAST[key] = _now
    if not dinput.mouse_available():
        return False, "鼠标通道不可用 ⇒ 自动量不了（本地 SendInput 模式没有硬件鼠标 ✓）"
    pre = _probe_cursor()
    if pre is None:
        return False, ("**看不见光标** ⇒ 自动量不了 ✗（鼠标在被控机上 ⇒ 要 A 机的 relay "
                       "回 `CURSOR?` ✓ 见 `remote_kbd/relay.py`；relay 是旧版就更新它 ✓）"
                       "⇒ 这个尺寸手工量一次 ✓")
    h, w = int(frame_shape[0]), int(frame_shape[1])

    def _wait_cursor(pred, timeout=0.8):
        """等"读到的位置**满足 `pred`**" ✓ ⇒ 返回那一次读数 ✓（到点还没有 ⇒ 返回最后一次 ✓ 照实报 ✓）。

        ⚠⚠⚠ **为什么判据不能是"两次读数相同 / 和上一次不同"** ✗✗（2026-10-15 ✓ 这里连栽两版 ✓）：
          慢 relay 的形态是"**读到的永远是几拍之前的位置**" ✓ ⇒
            · 判"**和上一次不同**" ⇒ 会**立刻返回那个旧位置** ✗（现场 `Δ=0` ✓）；
            · 判"**两次读数相同**" ⇒ **照样立刻返回** ✗✗ —— 因为在这个形态下**连续两次读数本来
              就相同** ✓（都在读同一条旧尾巴 ✓）⇒ 它给了个"看着很稳"的**假基准** ✓
              （**比不判更糟** ✗ —— 这一版就是被 `t_mouse_aim_measure_settle` 那根桩当场逮住的 ✓
              报的正是 `Δ=-900,-700`：基准还是撞角之前那个点 ✓）。
          ⇒ 判据只能是"**这一拍该在的那个点**" ✓：
            · 撞角之后 ⇒ 该在**虚拟屏左上角**（`(vx, vy)` ✓ 字段序见 `cursor_reply_text` ✓）；
            · 走一步之后 ⇒ 该**离开**那个点 ✓（走多少**正是要量的** ⇒ 不预设 ✓ 只要求"变了" ✓）。
        ⚠ 到点还没满足 ⇒ 返回最后一次读数 ✓（让调用方**照实**按"没动 / 不可信"报 ✓ ——
          宁可不写盘，也不给自己一个假数 ✗ 同 `save_gain` 的口径 ✓）。
        ⚠ 它**只许**在 `auto_measure` 里（也就只在**后台线程**里 ✓ 见 `auto_measure_async` /
          「立即量一次」的 `_ForceMeasure` ✓）：实时回路那条路上**一个都不许睡** ✗
          —— `selftest_decision` 里那条源码钉把这条口径钉着 ✓（本意：睡 = 卡住整条实时回路 ✓）。
        """
        _last, _t0 = None, time.monotonic()
        while True:
            _last = _probe_cursor()
            if _last is not None and pred(_last):
                return _last
            if (time.monotonic() - _t0) >= timeout:
                return _last
            time.sleep(0.05)

    # 撞角的目标 = **虚拟屏左上角** ✓（`pre` 的字段序：`x y vx vy vw vh [cap]` ✓
    #   见 `cursor_reply_text` ✓）；老格式只有三个数 ⇒ 退化成 (0, 0) 那台机器的原点 ✓
    try:
        _vx, _vy = int(pre[2]), int(pre[3])
    except (TypeError, ValueError, IndexError):
        _vx, _vy = 0, 0
    corner_zero()                                # ① 到已知原点（那台机器的虚拟屏左上角 ✓）
    # ⭐ **等它真的到角上**再当基准 ✓（撞角是大位移 ✓ 慢 relay 下要过几拍才读得到 ✓）
    p0 = _wait_cursor(lambda _p: abs(int(_p[0]) - _vx) <= 4 and abs(int(_p[1]) - _vy) <= 4)
    if p0 is None:
        _restore_cursor(pre, None)               # ⚠ 撞角了就得负责放回去 ✗
        return False, "撞角之后读不到光标 ⇒ 自动量不了 ✗"
    dinput.mouse_move(AUTO_STEP, AUTO_STEP)      # ② 走一步（两个轴一起走，一次量两轴 ✓）
    # ⭐ **等它离开那个点**（走多少正是要量的 ⇒ 不预设 ✓ 只要"变了" ✓ 见 `_wait_cursor` 那段 ✓）
    p1 = _wait_cursor(lambda _p: (int(_p[0]), int(_p[1])) != (int(p0[0]), int(p0[1])))
    dinput.mouse_move(-AUTO_STEP, -AUTO_STEP)    # ⚠ 先挪回撞角处（别把光标留在别处 ✓）
    dx_px = dy_px = None
    if p1 is not None:
        dx_px, dy_px = int(p1[0]) - int(p0[0]), int(p1[1]) - int(p0[1])
    # ⚠⚠ 再**放回撞角之前那儿** ✓（见 `_restore_cursor` ✓ —— 要除以刚量到的**屏幕**增益 ✓
    #   ⇒ 所以先把它算出来再送 ✓；量不到 ⇒ 它自己会放弃 ✓）
    _restore_cursor(pre, p0,
                    (dx_px / float(AUTO_STEP)) if dx_px else None,
                    (dy_px / float(AUTO_STEP)) if dy_px else None)
    if p1 is None:
        return False, "走完那一步读不到光标 ⇒ 自动量不了 ✗"
    if dx_px <= 0 or dy_px <= 0:
        return False, ("走 %d 单位之后光标没动（Δ=%d,%d）⇒ 这一步量不可信 ✗"
                       "（检查 A 机的指针速度 / 加速 ✓）" % (AUTO_STEP, dx_px, dy_px))
    vw, vh, cap = int(p0[4]), int(p0[5]), p0[6] if len(p0) > 6 else None
    # ③ 折算比：**优先 relay 报的抓取区域**（真数 ✓）；退而求其次才是"帧就是整屏"（1:1 ✓）
    if cap and int(cap[0]) > 0 and int(cap[1]) > 0:
        sx, sy = w / float(cap[0]), h / float(cap[1])
    elif vw > 0 and vh > 0 and abs(w - vw) <= max(4, 0.02 * vw) and abs(h - vh) <= max(4, 0.02 * vh):
        sx, sy = 1.0, 1.0
    else:
        return False, (
            "能量到「一单位走多少**屏幕**像素」✓，但这一帧（%dx%d）跟那台机器的屏幕"
            "**不是 1:1**（%dx%d）✗、relay 也没报**抓取区域**✗ ⇒ 折算比算不出来 ✓"
            "（**不猜** ✓ 拿猜当准只会点歪 ✓）⇒ 这个尺寸手工量一次 ✓"
            % (w, h, vw, vh))
    gx = (dx_px / float(AUTO_STEP)) * sx
    gy = (dy_px / float(AUTO_STEP)) * sy
    if not (0.01 <= gx <= 50.0 and 0.01 <= gy <= 50.0):
        return False, ("自动量出来的是 %.4f / %.4f px/单位 ⇒ 不像话（正常 0.1~5 ✓）"
                       "⇒ 不写盘 ✗（宁可不做，也别乱点 ✓）" % (gx, gy))
    save_gain(gx, gy, frame_shape=frame_shape,
              extra={"probe": {"source": "auto", "step": AUTO_STEP,
                               "screen_px": [dx_px, dy_px],
                               "scale": [round(sx, 6), round(sy, 6)],
                               "screen": [vw, vh],
                               "cap": [int(cap[0]), int(cap[1])] if cap else None}})
    return True, ("自动量好了：一单位走 **%.4f / %.4f 帧像素**"
                  "（屏幕基准 %.4f / %.4f × 折算 %.4f ✓）⇒ 已按帧尺寸 %s 存下 ✓"
                  % (gx, gy, dx_px / float(AUTO_STEP), dy_px / float(AUTO_STEP), sx, key))


#: ⭐⭐ 「**实时回路正收到的那个帧**」的尺寸（`(h, w)` ✓）—— 由实时回路每拍更新 ✓。
#: ⚠⚠ 为什么必须有它（2026-10-15 现场 ✓）：设置里那个按钮原来只问"**设置对话框所属那一台
#:   工作台**的 `live_panel._last_stats["size"]`" ✗ ⇒ 现场连着两次留痕都是 `size=[0, 0]`
#:   （`remote_measure_click` ✓）⇒ 一个字节都没发 ✗ ⇒ 用户看到的就是"**点了没反应**" ✓。
#:   而**开两个工作台**时，设置对话框属于哪一台、预览跑在哪一台，本来就可能**不是同一台** ✗
#:   ⇒ 那条路不可靠 ✓ ⇒ 让**真正在收帧的那条回路**把尺寸报上来 ✓（与工作台实例无关 ✓）。
_LAST_FRAME_SHAPE = None


def note_frame_shape(frame_shape):
    """实时回路每拍调一下 ✓：记下"我正在收的那个帧"的尺寸 ✓（`(h, w)` 口径 ✓）。

    ⚠ 只在拿到**有效**尺寸时覆盖 ✗（没有帧的那几拍不许把上一次的抹成 `None` ✗ ——
      否则按钮会在"刚好那一拍没帧"时误报"拿不到画面尺寸" ✓）。
    """
    global _LAST_FRAME_SHAPE
    try:
        if frame_shape is not None and len(frame_shape) >= 2:
            _h, _w = int(frame_shape[0]), int(frame_shape[1])
            if _w > 0 and _h > 0:
                _LAST_FRAME_SHAPE = (_h, _w)
    except (TypeError, ValueError, IndexError):
        pass


def last_frame_shape():
    """`(h, w)` = 实时回路**最近**一帧的尺寸；**从没有过帧 ⇒ `None`** ✓（界面据它说清 ✓）。"""
    return _LAST_FRAME_SHAPE


def gain_record(frame_shape=None):
    """这一尺寸的**整份记录**（含探针真数 ✓）⇒ dict / None ✓（用户 2026-10-15 要的"看得见"✓）。

    ⚠ 为什么必须补它 ✗（本轮踩的坑 ✓）：`load_gain` 只回 `(gx, gy)` ✓ ⇒ 界面只知道
      "**量过没有**"，**看不见"折算比是多少 / relay 报的屏幕与抓取区域是多少"** ✗ ——
      而这次"重量了也没用"恰恰就出在这一步（`probe.scale` / `probe.cap` ✓）⇒
      那三个数必须**能一眼看到** ✓（`calib_status` 现在就把它们写进状态行 ✓）。
    """
    flat, by = _read_gain_file()
    if frame_shape is None:
        return flat or None
    key = gain_frame_key(frame_shape)
    if key and key in by:
        return by[key]
    return None


def _log_probe(ok, frame_shape, why, source=None):
    """把一次测量的**真数**写进 `behavior.log` ✓（**成功失败都写** ✓ —— 失败那次尤其重要 ✗）。

    ⚠⚠ 为什么非要有它（2026-10-15 现场 ✓）：标定失败原来只写在**设置对话框那行小字**里 ✗
      ⇒ 关掉设置就没了 ⇒ 事后只能靠"连 A 机 relay 问一遍"（还被工作台占着连不上 ✗）或
      "反推文件改动时间"来猜 ✓ —— 这次整整绕了两轮 ✗ ⇒ 记下来：**帧尺寸 / relay 报的屏幕 /
      抓取区域 / 折算比 / 量出来的增益 / 人话原因** ✓。
    ⚠ 它自己炸了**绝不许**影响测量 ✗（留痕是附带的 ✓ 不是主流程 ✓）。
    """
    try:
        from core import behavior
        _r = gain_record(frame_shape) or {}
        _pr = dict(_r.get("probe") or {})
        _w, _h = 0, 0
        try:
            _h, _w = int(frame_shape[0]), int(frame_shape[1])
        except Exception:                            # noqa: BLE001
            pass
        behavior.event("mouse_gain_probe", ok=bool(ok), frame=[_w, _h],
                       source=source or _pr.get("frame_source") or _pr.get("source"),
                       screen=_pr.get("screen"), cap=_pr.get("cap"),
                       scale=_pr.get("scale"), gain=[_r.get("gain_x"), _r.get("gain_y")],
                       why=str(why)[:180])
    except Exception:                                # noqa: BLE001
        pass


_raw_auto_measure = auto_measure


def auto_measure(frame_shape, force=False, source=None):     # noqa: F811
    """同 `_raw_auto_measure` ✓，外面包一层**留痕** ✓（成功失败都记 ✓ 见 `_log_probe` ✓）。

    ⚠ 为什么不直接在原函数体里加日志 ✗：它有十来处 `return False, …`（每处一句人话 ✓）
      ⇒ 逐个加等于把那份说明撕成十几段 ✓；**在出口包一层**只动一处 ✓，而且**所有调用方**
      （`auto_measure_async` 那条自动路 ✓、设置里「立即量一次」那个按钮 ✓）自动全带上 ✓。
    ⚠ 参数与老的一字不差 ✓（多的 `source` 只喂给日志 ✓ —— 调用方知道"这一帧是哪一路"
      就传 ✓ 不知道就不传 ✓ 绝不影响测量本身 ✓）。
    """
    ok, why = _raw_auto_measure(frame_shape, force=force)
    _log_probe(ok, frame_shape, why, source=source)
    return ok, why


#: 自动量**正在后台跑**吗 ✓（跑的时候**谁都不许再动鼠标** ✗ —— 两步插在一起就白量了 ✓）。
_AUTO_BUSY = False
#: 忙标记是**什么时候**立起来的（见 `AUTO_BUSY_MAX_S` ✓）。
_AUTO_BUSY_AT = 0.0
#: ⭐⭐ 忙标记最多算"忙"这么久（秒 ✓）—— **自愈** ✓。
#: ⚠⚠ 为什么必须有它（2026-10-10 ✓ 用户："**进入选频道界面后 A 机的鼠标没有被操控**" ✓）：
#:   只要忙标记**被永久卡住**（那条线程出意外没走到 `finally` ✓）⇒ 三个鼠标出口
#:   （点 / 瞄 / 滚 ✓）就**永远返回"正在后台自动量"** ✗ ⇒ 表现就是"**鼠标完全没被操控**"✓
#:   —— 正好是用户看到的那一幕 ✓。⇒ 超时就当它**没在忙** ✓（宁可偶尔让两步插一下，
#:   也绝不把鼠标**永久锁死** ✗ 那等于这个功能废了 ✓）。
AUTO_BUSY_MAX_S = 30.0


def _auto_busy():
    """现在**真的**在量吗 ✓（超过 `AUTO_BUSY_MAX_S` 没动静 ⇒ 当它没在忙 ✓ 见那段说明 ✓）。"""
    if not _AUTO_BUSY:
        return False
    if (time.monotonic() - _AUTO_BUSY_AT) > AUTO_BUSY_MAX_S:
        return False
    return True
#: 自动量的串行锁（同一时刻只允许一条量 ✓）。
_AUTO_LOCK = threading.Lock()
#: ⭐⭐ 失败记忆：帧尺寸键 → `(时刻, 人话)` ⇒ **冷却期内不再重试** ✓（见 `AUTO_RETRY_S` ✓）。
#: ⚠⚠⚠ 为什么它才是治"**A 机鼠标停在左上角不动了**"的那一刀（2026-10-10 ✓ 用户**报了两次**
#:   才查清 ✗ 这是我引入的第三处 ✗）：第一版 `auto_measure_async` 里调的是
#:   `auto_measure(…, force=True)` ✗ —— **`force` 会跳过冷却** ✗ ⇒ 只要这个尺寸一直量不成
#:   （量出来被判"不像话" / 看不见光标 / 折算比算不出 ✓）⇒ **每一拍都起一条新的后台量** ✗
#:   ⇒ 每 ~100 毫秒"撞角归零"一次 ⇒ **光标被反复撞到左上角 ⇒ 看上去就是卡在那儿一动不动** ✓✓
#:   （现场现象一字不差 ✓）。
#: ⇒ 现在：**失败了就记住** ✓（冷却 `AUTO_RETRY_S` ✓ + 最多试 `AUTO_MAX_TRIES` 次 ✓
#:   ⇒ 试完就认了、让人手工量一次 ✓ 绝不再折腾机器 ✓）。
_AUTO_FAIL = {}
#: 同一个尺寸**最多自动试几次** ✓（试完就认 ✓）。
AUTO_MAX_TRIES = 3
_AUTO_TRIES = {}


#: ⭐⭐⭐ **自动量的总开关 —— 默认【关】** ✓✓（2026-10-10 第三次现场之后定的 ✓）。
#:   用户原话（**同一句报了三次**）："**A 机鼠标停在左上角不动了**" ✓
#:   ⇒ 猜了两次都没到根 ✗ ⇒ 那就**先把这个会动鼠标的新功能关掉** ✓（它一关，
#:     行为就回到"老样子"：没量过 ⇒ **说清 + 不点** ✓ 鼠标绝不会自己跑 ✓）
#:   ⇒ 想开：把 `config/mouse_gain.json` 里的 `"auto_measure": true` 写上 ✓
#:     （找不到这个键 / 读不出来 ⇒ **当关** ✓ 安全的一侧做默认 ✓）。
#:   ⚠ 开着它才可能动鼠标 ✓；关着的时候 `auto_measure_async` **一个字节都不发** ✗。
AUTO_MEASURE_DEFAULT = False


def auto_measure_enabled():
    """自动量开着吗 ⇒ 读 `config/mouse_gain.json` 的 `"auto_measure"` ✓（缺省 / 读不出来 ⇒ **关** ✓）。"""
    import json
    try:
        d = json.loads(GAIN_PATH.read_text(encoding="utf-8"))
        return bool(d.get("auto_measure", AUTO_MEASURE_DEFAULT))
    except Exception:                            # noqa: BLE001 —— 读不出来 ⇒ 当关 ✓
        return AUTO_MEASURE_DEFAULT


def set_auto_measure(on):
    """**写**「自动量」这个开关 ⇒ `(ok, 人话)` ✓（用户 2026-10-15：「设置 → 链路 → 远程操控」要一个勾选框 ✓）。

    ⚠⚠ 为什么必须由**这里**写、不许界面自己 `json.dump` ✗：这个键和标定**同住一个文件** ✓
      （`config/mouse_gain.json` ✓ 见 `_read_gain_file` 的文件形状那段 ✓）⇒ 界面整写会把
      `by_frame` 里那几份**抹掉** ✗ —— 那是"一种尺寸量一次"攒下来的 ✓（`load_gain` 的说明里
      写着为什么按尺寸存 ✓）抹了就得**每种尺寸重量一次** ✓。
    ⇒ 这里**只改这一个键** ✓，读得出来的部分**一个字都不丢** ✓（文件不在 / 坏了 ⇒ 从空开始 ✓）。
    """
    import json
    try:
        try:
            d = json.loads(GAIN_PATH.read_text(encoding="utf-8"))
            if not isinstance(d, dict):
                d = {}
        except Exception:                        # noqa: BLE001 —— 没有 / 坏了 ⇒ 从空开始 ✓
            d = {}
        d["auto_measure"] = bool(on)
        GAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GAIN_PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:                       # noqa: BLE001 —— 写盘失败 ⇒ 说清 ✓
        return False, "写不进 `%s`：%s" % (GAIN_PATH.name, e)
    return True, ("已**开启**自动量 ✓（这个尺寸没量过时它会自己量一次 ⇒ **会动鼠标** ✓）"
                  if on else "已**关闭**自动量 ✓（没量过的尺寸 ⇒ 只提示、绝不自己动鼠标 ✓）")


def calib_status():
    """给「远程操控」那一组用的一行状态 ⇒ **人话**（用户 2026-10-15 ✓）。

    ⚠ 只报**文件里有什么** ✓、不猜"现在收到的是多大" ✗ —— 那是实时回路的账 ✓（判据见
      `gain_mismatch`：它问的就是"**这个尺寸量过没有**"✓），界面这一侧拿不到帧 ✓。
    """
    _sizes = gain_sizes()
    if not _sizes:
        return "还没有量过任何画面尺寸 ⇒ 鼠标**不会点**（点服务器 / 点频道 / 测谎瞄准都靠它 ✗）"
    _lines = ["已量过的画面尺寸：%s　｜　自动量：%s" % (
        " / ".join(sorted(_sizes)), "开" if auto_measure_enabled() else "关")]
    # ⚠⚠ **折算比必须看得见**（2026-10-15 ✓ 这一条是拿一次真事故换来的 ✗）：`load_gain` 只回
    #   两个增益 ✗ ⇒ 界面看不见"它是拿**什么几何**算出来的" ✗ —— 而"重量了也没用"这次恰恰就是
    #   **relay 报的抓取区域不对** ✓（`scale` = 这一帧 ÷ 抓取区域 ✓）⇒ 每个尺寸给一行：
    #   **来源 / 折算比 / relay 报的屏幕 / 抓取区域** ✓。
    #   ⭐ 自证判据：`scale` 不等于"这一帧 ÷ **你 A 机的真实屏幕**"⇒ 一眼就是 relay 那边报错了 ✓
    #     （例：帧 1366×768 / A 机 1920×1080 ⇒ 对的折算比是 **0.7115**；写 0.8 ⇒ 它以为屏幕是
    #      1707×960 ✗ ⇒ 点击会**按比例**落在 80% 处 ✗ —— 这正是用户看到的现象 ✓）。
    _flat, _by = _read_gain_file()
    for _k in sorted(_by):
        _e = _by[_k] or {}
        _pr = dict(_e.get("probe") or {})
        _bits = [str(_k)]
        _src = _pr.get("frame_source") or _pr.get("source")
        if _src:
            _bits.append(str(_src))
        _sc = _pr.get("scale")
        try:
            if _sc:
                _bits.append("折算 %s" % "/".join("%.4f" % float(v) for v in _sc))
        except (TypeError, ValueError):
            pass
        _cap = _pr.get("cap")
        try:
            if _cap:
                _bits.append("抓取区域 %dx%d" % (int(_cap[0]), int(_cap[1])))
        except (TypeError, ValueError, IndexError):
            pass
        _scr = _pr.get("screen")
        try:
            if _scr:
                _bits.append("relay 报的屏幕 %dx%d" % (int(_scr[0]), int(_scr[1])))
        except (TypeError, ValueError, IndexError):
            pass
        _lines.append("　· " + "　".join(_bits))
    return "\n".join(_lines)


def auto_measure_async(frame_shape):
    """⭐ **后台**量一次 ✓（**实时回路绝不许阻塞** ✗）⇒ `(是否已起, 人话)` ✓。

    ⚠⚠ 为什么必须有它（2026-10-10 ✓ 现场："**A 机鼠标停在左上角不动了**" ✓ 查出来的
      第二处 ✗ 也是我引入的）：量一次要**读两次光标** ✓，远端那两次是**等 relay 回包** ✓
      （`KbdClient.cursor` 各等最多 0.8 秒 ✓）⇒ 直接在**实时回路**那条线程里做 ⇒
      它被卡住一两秒 ✗ —— 而本仓库的硬规矩是「**实时回路不许 sleep / 不许阻塞**」✗
      （见 `click_ratio` 的说明 ✓：睡一下就是丢帧 / 积压 ✓）。
      ⇒ 挪到后台 ✓；代价是**这一拍点不了** ✓（返回 False 并说清"正在量"✓）
      ⇒ 量完存下 ✓ ⇒ **下一拍**（重连状态机本来就会重试 ✓）就正常点了 ✓。

    ⚠⚠⚠ **失败不许当没发生** ✗（那是"每拍都撞一次角"的根 ✓ 见 `_AUTO_FAIL` ✓）：
      记下失败 + 冷却 + 最多 `AUTO_MAX_TRIES` 次 ✓ ⇒ 试完就**明说"请手工量一次"** ✓。
    """
    global _AUTO_BUSY, _AUTO_BUSY_AT
    key = gain_frame_key(frame_shape)
    if not key:
        return False, "拿不到画面尺寸"
    if not auto_measure_enabled():
        # ⭐⭐⭐ **默认关** ✓（见 `AUTO_MEASURE_DEFAULT` ✓）—— 关着就**一个字节都不发** ✗
        #   用户 2026-10-10 同一句报了三次 ⇒ 先把"会动鼠标"的新功能退出去 ✓
        #   （退出去之后：没量过 ⇒ 老样子「说清 + 不点」✓ 鼠标绝不会自己跑 ✓）。
        return False, ("自动量鼠标标定**当前是关的** ✓（怕再出现「A 机鼠标卡在左上角」✓）"
                       "⇒ 这个尺寸**手工量一次**："
                       "`python -X utf8 -m tools.mouse_aim_calib --source stream` ✓"
                       "（要开自动量：在 `config/mouse_gain.json` 里加一行 "
                       "`\"auto_measure\": true` ✓）")
    if _auto_busy():
        return False, "正在**后台自动量**标定 ✓ ⇒ 这一拍不动鼠标 ✓（量完下一拍就能点 ✓）"
    _t, _why = _AUTO_FAIL.get(key, (0.0, ""))
    if _why:
        _n = _AUTO_TRIES.get(key, 0)
        if _n >= AUTO_MAX_TRIES:
            return False, ("自动量**已经试过 %d 次都没成**（%s）⇒ 不再自动试了 ✓"
                           "这个尺寸**手工量一次** ✓"
                           "（`python -X utf8 -m tools.mouse_aim_calib --source stream` ✓）"
                           % (_n, _why))
        if (time.monotonic() - _t) < AUTO_RETRY_S:
            return False, ("刚自动量过没成（%s）⇒ **%.0f 秒内不再试** ✓（免得一直挪鼠标 ✓）"
                           % (_why, AUTO_RETRY_S))
    _AUTO_BUSY = True
    _AUTO_BUSY_AT = time.monotonic()          # ⚠ 记时刻 ⇒ 超时自愈 ✓
    _AUTO_TRIES[key] = _AUTO_TRIES.get(key, 0) + 1

    def _run():
        global _AUTO_BUSY, _AUTO_BUSY_AT
        try:
            with _AUTO_LOCK:
                _ok, _w = auto_measure(frame_shape, force=True)
            if not _ok:
                _AUTO_FAIL[key] = (time.monotonic(), str(_w)[:120])
        except Exception as e:                   # noqa: BLE001 —— 后台线程不许把谁炸了 ✗
            _AUTO_FAIL[key] = (time.monotonic(), "%s: %s" % (type(e).__name__, e))
        finally:
            _AUTO_BUSY = False
            _AUTO_BUSY_AT = 0.0

    threading.Thread(target=_run, daemon=True, name="mouse-auto-measure").start()
    return True, ("正在**后台**自动量这个尺寸的鼠标标定 ✓（这一拍先不动鼠标 ✓"
                  "量完下一拍就能点 ✓）")


def gain_warn(frame_w=None):
    """**这一帧要不要提醒"标定不是在这路画面上量的"** ⇒ 人话；不用提醒 ⇒ "" ✓。

    ⚠⚠ 为什么非要有（2026-10-10 ✓ 用户："**操作不对，鼠标点歪了**" ✗ 定案）：
      现场是**收流（1366 宽）量、本地窗口（1918 宽）用** ✗ ⇒ 走位长了约 **40%**
      ⇒ 「点服务器」10 次全停在同一界面 ✗。而这一步**全程静默** ✗（没有任何地方说
      "标定口径跟当前画面不一样" ✓）⇒ 只能靠人看出"点歪" ✓。
    ⇒ 判据：帧宽与标定基准差 **> 5%** ⇒ 返回一句**能直接进日志**的人话 ✓（见两个调用点 ✓）。
    """
    if not frame_w:
        return ""
    ref = gain_ref_frame()
    if not ref or not ref[0]:
        # ⚠ **老标定文件没记基准帧**（2026-10-10 之前写的都没有 ✓）⇒ 没法算差值 ✗，
        #   但**恰恰最可疑**（现场那份就是这种 ✓）⇒ 每个进程只提醒**一次** ✓ 别刷屏 ✗。
        global _GAIN_OLD_WARNED
        if not _GAIN_OLD_WARNED:
            _GAIN_OLD_WARNED = True
            return ("⚠ 这份标定**没记基准帧尺寸**（旧格式）⇒ 换过画面来源 / 帧尺寸的话"
                    "**落点会差一个比例** ✗ ⇒ 建议重量一次："
                    "`python -X utf8 -m tools.mouse_aim_calib --source live` ✓")
        return ""
    w = float(frame_w)
    if abs(w / float(ref[0]) - 1.0) <= 0.05:
        return ""
    return ("⚠ 标定是在 **%d 宽**的画面上量的，现在这帧是 **%d 宽** ⇒ "
            "gain 是「帧像素 / 指令单位」口径 ⇒ **落点会差 %.0f%%** ✗ "
            "（重新标定：`python -X utf8 -m tools.mouse_aim_calib --source %s` ✓）"
            % (ref[0], int(w), (w / float(ref[0]) - 1.0) * 100.0,
               "live" if abs(w - ref[0]) > 0.05 * ref[0] else "stream"))


def target_px(frame_w, frame_h, xr, yr):
    """画面比例 (0~1) → 画面像素。判不出来 ⇒ `(None, why)`。

    比例越界**不夹取**、直接判失败：目标写错了就该**停下来报**，
    而不是"顺手"点到边上（那可能正好是「回到首页」/「删除角色」✗ 见设计文档 §9）。
    """
    try:
        w = float(frame_w)
        h = float(frame_h)
        x = float(xr)
        y = float(yr)
    except (TypeError, ValueError):
        return None, "目标比例不是数字"
    if w <= 0 or h <= 0:
        return None, "拿不到画面尺寸"
    if not (0.0 <= x <= 1.0) or not (0.0 <= y <= 1.0):
        return None, "目标比例超出画面（要看 0~1：x=%s y=%s）" % (xr, yr)
    return (x * w, y * h), ""


def _virtual_origin():
    """`corner_zero()` 撞角后光标**实际落在哪** ⇒ `(vx, vy)`（整套绝对定位的原点 ✓）。

    ⚠⚠ **只在"鼠标打本机"时才是本机的虚拟屏左上角** ✗（2026-10-10 ✓ 用户原话：
      **"我可能会把游戏窗口随意拖动，并且游戏窗口可能会被拖到副显示屏"** ✓ 定案）：
      · 后端 = **本地串口 ProMicro**（`SerialKbd` ✓ = 鼠标动在**本机** ✓）
        ⇒ 原点 = 本机虚拟屏左上角 ✓（本机实测 `(−2560, 0)` ✓ ⇒ **副屏（x<0）也够得着** ✓）；
      · 后端 = **网络 relay**（`KbdClient` ✓ = 鼠标动在**被控机** ✓）
        ⇒ 那台的屏幕布局**这边问不到** ✗ ⇒ 只能按 `(0, 0)` 算 ✓（= 老行为一字不变 ✓；
        被控机是单显示器时本来就对 ✓，多显示器时这台机器上量不出它的原点 ⇒ **别硬猜** ✗）；
      · 没有后端（本地 SendInput ✗ 根本没有硬件鼠标 ✓）⇒ `(0, 0)` ✓。
    """
    try:
        from decision import input as _in
        if type(getattr(_in, "_remote", None)).__name__ != "SerialKbd":
            return 0, 0
    except Exception:                          # noqa: BLE001
        return 0, 0
    try:
        from core import wincap                 # 一处实现 ✓（别自己调 GetSystemMetrics ✗）
        vx, vy, _w, _h = wincap.virtual_screen()
        return int(vx), int(vy)
    except Exception:                          # noqa: BLE001 —— 问不到就按 (0,0) ✓
        return 0, 0


#: `frame_rect` 的**离谱门**：抓取区域/帧尺寸 的比例必须在这个范围里 ✓
#: （本机实测 1.4054 ✓ 收流整屏那种可能到 3.7 ✓；超出这个范围 ⇒ 记的东西多半是错的 ✓
#:  ⇒ 当没量过 ✓ 宁可退回老行为，也别按一条离谱记录去点 ✗）。
RECT_SANITY = 4.0


def frame_rect(frame_shape=None):
    """这一帧对应的**抓取区域** ⇒ `(ox, oy, cw, ch)`（**屏幕像素** ✓）；没量过 ⇒ `None` ✓。

    ⚠⚠ **为什么非要它**（2026-10-17 ✓ 用户现场两句话：
      「**选频道界面鼠标操作完全歪了**」✗ ＋「**刚跑了标定 还是没用**」✗）：
      · `to_screen()` 老版本**只加原点、不折算比例** ✗，而帧是**推流**来的、被缩过 ✗ ——
        本机实测：抓取区域 **1920×1080** ✓ 而帧是 **1366×768** ✓ ⇒ 比例 **k = 1.4054** ✗
        ⇒ 目标点整体偏 k 倍 ✓；再叠上桌面 **5120×1440** ✓ / 虚拟原点 **(−2560, 0)** ✓ 的
        原点折返 ✓ ⇒ 现场算出 **4317** 单位（正确值约 **700** ✓）⇒ 差 6 倍 = "完全歪了" ✓。
      · ⚠ 标定工具**拒绝猜**这个折算比是对的 ✓（当时报的是"这一帧跟那台机器的屏幕不是
        1:1（5120x1440）✗、relay 也没报**抓取区域** ✗ ⇒ 算不出来 ✓（**不猜** ✓）"✓）
        ⇒ 只能**实测** ✓（`tools/mouse_aim_rect.py` ✓）。
    ⚠ **没量到 ⇒ `None`** ⇒ `to_screen` / `screen_gain` **一字退回老行为** ✓
      ⇒ 这套东西**加了也不会把现在能用的场景弄坏** ✓（没量过就与今天一样 ✓）。
    ⚠ 校验（宁缺毋滥 ✓）：四个数得是数、`cw/ch` 为正 ✓、比例落在 `RECT_SANITY` 内 ✓。
    """
    if frame_shape is None:
        return None                      # 手上没帧 ⇒ 不折算 ✓（宁可不动 ✓）
    try:
        fh, fw = int(frame_shape[0]), int(frame_shape[1])
    except (TypeError, ValueError, IndexError):
        return None
    if fw <= 0 or fh <= 0:
        return None
    _flat, by = _read_gain_file()
    key = gain_frame_key(frame_shape)
    e = by.get(key) if key else None
    if not isinstance(e, dict):
        # 精确找不到 ⇒ 找**最接近**那份（与 `load_gain` 同一条规矩 ✓ 2% 以内 ✓）
        best, best_d = None, None
        for cand in by.values():
            if not isinstance(cand, dict):
                continue
            try:
                ew, eh = int(cand.get("ref_w") or 0), int(cand.get("ref_h") or 0)
            except (TypeError, ValueError):
                continue
            if ew <= 0 or eh <= 0:
                continue
            dw = abs(ew - fw) / float(fw)
            dh = abs(eh - fh) / float(fh)
            if dw > GAIN_SIZE_TOL or dh > GAIN_SIZE_TOL:
                continue
            if best_d is None or dw + dh < best_d:
                best, best_d = cand, dw + dh
        e = best
    if not isinstance(e, dict):
        return None
    try:
        ox, oy, cw, ch = (int(round(float(v))) for v in e.get("frame_rect"))
    except (TypeError, ValueError):
        return None
    if cw <= 0 or ch <= 0:
        return None
    kx, ky = cw / float(fw), ch / float(fh)
    if not (1.0 / RECT_SANITY <= kx <= RECT_SANITY
            and 1.0 / RECT_SANITY <= ky <= RECT_SANITY):
        return None                      # 差得离谱 ⇒ 当没量过 ✓（按它点就是飞 ✓）
    return ox, oy, cw, ch


def roi_px(roi, frame_shape, xr, yr):
    """比例（**相对"我们看到的这块画面"**）⇒ **整帧**像素 ✓（部署台框选游戏区域用 ✓）。

    ⭐⭐⭐ 为什么要有它（用户 2026-10-17 ✓ 原话：
      「**现在我需要部署台也能像本地流程一样能够自由选择游戏窗口区域**
       注意**断线重连、自动测谎的鼠标适配**」✓）：
      · 部署台（B 机）自己**不抓屏** ✗，吃的是 A 机推来的整帧 ✓；
      · 框了"游戏窗口区域"之后，我们看到的画面就是**整帧里的一块** ✓；
      · 而断线重连 / 自动测谎给的比例是**相对它看到的那块**算的 ✓
        ⇒ 直接拿它去乘**整帧**尺寸就会**整体偏移一块**（点歪 ✓ 而且偏得很有规律 ✓）。
    ⇒ 口径：`整帧像素 = roi 原点 + 比例 × 这块的尺寸` ✓（**一块实现**，两处调用共用 ✓）。
    ⚠ 比例越界**不夹取**、直接失败（同 `target_px` ✓ —— 写错了就该停下报 ✗）。
    """
    px, why = target_px(frame_shape[1], frame_shape[0], xr, yr)
    if px is None:
        return None, why
    try:
        x, y, w, h = (float(v) for v in roi)
    except (TypeError, ValueError, IndexError):
        return None, "框选区域不是四个数（%r）" % (roi,)
    if w <= 0 or h <= 0:
        return None, "框选区域无效（宽高该 > 0：%r）" % (roi,)
    return (x + px[0], y + px[1]), ""


def to_screen(px, py, origin=None, frame_shape=None):
    """**画面像素** → **屏幕像素**。

    ⚠⚠ 为什么必须有这一步（2026-10-10 ✓ 用户原话：**"我可能会把游戏窗口随意拖动，并且
      游戏窗口可能会被拖到副显示屏"** ✓）：帧是**从你框选那块区域裁出来的** ✗ ⇒ 帧里的
      `(0, 0)` 是**那块区域的左上角** ✓、不是屏幕原点 ✗。窗口拖到副屏（x<0 ✓）时，
      "帧像素"与"屏幕像素"差的正是这个 `origin` ✓ —— 不补上 ⇒ 点击落点整体偏一整块区域 ✗
      （老版本在副屏上则是**根本够不着** ✓ 见 `counts_for` 的说明 ✓）。

    ⚠⚠ **两件事，缺一不可**（2026-10-17 ✓ 现场"鼠标完全歪了"就是漏了第二件 ✗）：
      ① 加**原点**（那块区域的左上角在屏幕哪 ✓ —— 老版本就有 ✓）；
      ② 乘**比例**（抓取区域尺寸 / 帧尺寸 ✓ —— 老版本**没有** ✗）：
         ⚠ 帧是**推流**来的、被缩过 ✗（本机 1920×1080 的区域 → 1366×768 的帧 ✓
         ⇒ 帧里的 497px 其实是屏幕上的 **698px** ✓）⇒ 只加原点不加比例 ⇒
         落点整体偏 **k 倍** ✓（正是"完全歪了"✓）。
      ② 的比例与①的原点**一起**来自 `frame_rect()` ✓（**量过才有** ✓）⇒
      没量过 ⇒ ①②都不做 ✓（= 老行为**一字不变** ✓）。
    `origin` 给了、而**没量过** `frame_rect` ⇒ 按调用方给的（老行为：只加、不折算 ✓）；
    量过 `frame_rect` ⇒ **一律折算**（它才是真值 ✓ 见下面那条教训 ✗）。

    ⚠⚠ **2026-10-17 现场（断线重连「选区/点服务器」那一步鼠标点歪 ✗）**：
      原来 `if origin: return 只加原点` **排在折算之前** ✗ —— 而断线重连 / 自动测谎
      **每次都传 origin**（`live_thread` 那条 `_org` 恒有值 ✓ `(0, 0)` 也算有 ✓）
      ⇒ **永远走不到折算那一段** ✗ ⇒ 推流缩放（1366×768 帧 ← 1920×1080 区域，k=1.4054）
      被整个忽略 ✓；而 `screen_gain()` **是折算的** ✓ ⇒ 分子没乘 k、分母乘了 k
      ⇒ 落点整体**短 k 倍**（朝左上偏）✓ = 用户说的「**断线重连选区阶段鼠标点歪了**」✓
      而且**每次都歪、歪得一模一样** ✓（不是偶发 ✗）。
      ⇒ 口径改成：**量过就以 `frame_rect` 为准**（它同时给了真原点与真比例 ✓），
        `origin` 只在"没量过"时兜底 ✓。
    """
    r = frame_rect(frame_shape) if frame_shape is not None else None
    if r:
        try:
            fh, fw = int(frame_shape[0]), int(frame_shape[1])
        except (TypeError, ValueError, IndexError):
            return float(px), float(py)
        return (float(px) * r[2] / float(fw) + r[0],
                float(py) * r[3] / float(fh) + r[1])
    # ⛔ **"没实测就按 link.yaml 折 k"那一版已撤回**（2026-10-17 ✓ 实机反证）：
    #   它只折了**分子**（目标像素）没折**分母**（`screen_gain` 的增益 ✓ 那支只在有
    #   `frame_rect` 时才折算 ✓）⇒ 送的指令数多出 k 倍 ⇒ 光标**偏右下越过目标** ✓
    #   （用户原话：「**偏右下，差一整块（越过了目标）**」✗）。
    #   ⇒ 回到"要么两边都折、要么都不折" ✓ 现在这条路上两边都**不折**：
    #     目标不折（497）＋ 增益不折（1.0）⇒ 落点与改之前**一模一样** ✓
    #     —— 真正的偏差是**增益本身**（实测约 1.15 px/单位 ✓ 见 `mouse_tap_probe`）。
    if origin:
        return float(px) + int(origin[0]), float(py) + int(origin[1])
    return float(px), float(py)


def screen_gain(gain, frame_shape=None):
    """「**帧像素** / 指令单位」的 gain ⇒ 折成「**屏幕像素** / 指令单位」✓（`counts_for` 要的正是后者 ✓）。

    ⚠⚠ **这就是现场那条"尺度被算了两遍"的病根**（2026-10-17 ✗）：
      `to_screen()` 已经把帧像素折成屏幕像素 ✓，而 `counts_for()` 又拿**帧像素口径**的 gain
      除了第二遍 ✗（`load_gain` 的注释写"帧像素"✗、`counts_for` 的 docstring 写"屏幕像素"✓
      ⇒ **同一个文件里两处口径打架** ✗）⇒ 于是**多乘了一个 k** ✓。
      ⭐ 折算过来是**自洽**的 ✓：`0.7115 × 1.4054 = 1.0` ✓ —— 正好等于标定探针量到的
      「**1 单位走 1 个屏幕像素**」✓✓（`probe.step=300 / screen_px=[300,300]` ✓）
      ⇒ 这也反过来印证 ①（比例 k）该乘上 ✓。
    ⚠ 没量过 `frame_rect` ⇒ **原样返回** ✓（老行为一字不变 ✓）。
    """
    try:
        gx, gy = float(gain[0]), float(gain[1])
    except (TypeError, ValueError, IndexError):
        return gain
    r = frame_rect(frame_shape) if frame_shape is not None else None
    if not r:
        return (gx, gy)
    try:
        fh, fw = int(frame_shape[0]), int(frame_shape[1])
    except (TypeError, ValueError, IndexError):
        return (gx, gy)
    return (gx * r[2] / float(fw), gy * r[3] / float(fh))


def counts_for(px, py, gain_x, gain_y):
    """**屏幕像素** → 「从**虚拟屏左上角**（`corner_zero` 的落点 ✓）要走多少指令单位」。

    返回 `((dx, dy), why)`：算不出来时第一个是 `None`、`why` 是人话。
    `gain` = 屏幕像素 / 指令单位（见 `tools/mouse_aim_calib.py` ✓）。

    ⚠⚠ 口径（2026-10-10 ✓ 用户："**游戏窗口可能会被拖到副显示屏**" ✓ 定案）：
      · 撞角归零把光标撞在**虚拟屏**左上角 = `(vx, vy)` ✓（本机 `(−2560, 0)` ✓ 实测 ✓）；
      · ⇒ 从**那儿**算起，指令才是**非负**的 ✓：主屏目标 `(700, 220)` ⇒ `(3260, 220)` ✓；
        副屏目标 `(−1200, 300)` ⇒ `(1360, 300)` ✓ —— **两块屏都够得着** ✓；
      · ⚠ 老版本按 `(0, 0)` 算 ✗ ⇒ **副屏（x<0）根本走不到** ✗（算式只会给出非负数 ✓
        而副屏在原点**左边** ✗）⇒ 窗口一拖到左边那台就点不中 ✓。
    """
    try:
        gx = float(gain_x)
        gy = float(gain_y)
    except (TypeError, ValueError):
        return None, "鼠标增益不是数字"
    if gx <= 0 or gy <= 0:
        return None, "鼠标还没标定（增益非正：%s / %s）" % (gain_x, gain_y)
    vx, vy = _virtual_origin()
    return (int(round((float(px) - vx) / gx)),
            int(round((float(py) - vy) / gy))), ""


def aim_available():
    """这一层现在**能不能真的点**：鼠标通道在 + 标定过了。返回 `(ok, why)`。

    为什么要单独一个"能不能"：断线重连的**状态机**要在"发出点击动作"之前
    就知道该不该动（没标定就一点都不该动 ✓ 而且要说清缺什么 ✓），
    而不是发一条注定失败的点击再回退 ✗。
    """
    if not dinput.mouse_available():
        return False, ("鼠标通道不可用（本地 SendInput 模式没有硬件鼠标）"
                       "⇒ 先把「被控机部署台」的 ProMicro 后端连上")
    if load_gain() is None:
        # ⚠ 报"在哪跑"要说准：这个工具跑在**工作机（跑工作台那台）**上 ——
        #   鼠标指令经 relay 打到游戏机 ✓（见 `tools/mouse_aim_calib.py` 的模块头 ✓）。
        #   写成"在游戏机上跑"会把人引到 A 机上去找仓库 ✗（2026-10-07 用户正是这么问的 ✓）。
        return False, ("鼠标没标定（缺 config/mouse_gain.json）"
                       "⇒ 先在**跑工作台这台机器**上跑 tools/mouse_aim_calib.py"
                       "（指令经 relay 打到游戏机 ✓，且要先停掉实时预览 ✓）")
    return True, ""


def corner_zero():
    """把光标撞到**虚拟屏左上角** —— 那就是整套绝对定位的**原点** ✓（见 `counts_for` ✓）。

    ⚠⚠ **不要再往右下补到 (0, 0)** ✗（2026-10-10 ✓ 两次现场合起来才看清 ✓）：
      · `mouse_move(-4000, -4000)` 会被 OS **夹在虚拟屏边界** ✓ ⇒ 稳稳停在 `(vx, vy)` ✓
        （本机 `(−2560, 0)` ✓ 实测 ✓）；
      · 上一版（293 条）为了让"主屏基准"对上，撞完又补了一下、落到 **(0, 0)** ✗ ——
        主屏是准了 ✓，但**副屏（x<0）就再也够不着了** ✗（`counts_for` 只会给出非负指令 ✓
        而副屏在原点**左边** ✗）。用户 2026-10-10 原话：**"我可能会把游戏窗口随意拖动，
        并且游戏窗口可能会被拖到副显示屏"** ✓ ⇒ 两个都要能用 ✓。
      · 正确做法：**原点就用虚拟屏左上角** ✓（撞角的天然落点 ✓）、由 `counts_for`
        从这个原点算 ✓ ⇒ 两块屏都够得着 ✓，而且**一条指令都不多发** ✓。
    ⚠ 不读光标位置 ✓（点击那条路上不许 sleep / 不许等往返 ✓ 见 `click_ratio` ✓）。
    ⚠ 坐标口径：`wincap.virtual_screen()` 用的是 `GetSystemMetrics(76/77)`（**物理**坐标 ✓
      与鼠标指令同一个坐标系 ✓ —— 实测本机虚拟屏 = `(−2560, 0, 5120, 1440)` ✓）。
    """
    dinput.mouse_move(-CORNER_STEP, -CORNER_STEP)


#: 撞角「到位」的容差（像素 ✓）—— 实测撞角落点**是精确的** `(−2560, 0)` ✓，
#: 留几像素只为吸收 OS 的取整 ✓；**不许放大** ✗（放大了就把"没归零"当"归零了"✓）。
CORNER_TOL = 8


def _cursor_pos():
    """**系统光标现在在哪** ⇒ `(x, y)`；读不到 ⇒ `None` ✓（瞬时 ✓ **绝不 sleep** ✗）。

    ⚠⚠ 为什么要读它（用户 2026-10-17 ✓ 现场：断线重连卡在"点服务器"上 ✗、
      说"**鼠标都不在游戏窗口内 一直在屏幕最右边**"✗）：
      `click_ratio` 原来是「`corner_zero()` → `mouse_move(dx, dy)` → `click`」**一口气发完** ✗
      —— 设计前提是"固件按行顺序执行、所以按钮按下时光标已经在目标上 ✓"（见它的 docstring ✓）。
      ⚠ 但那条前提**漏了一件事**：撞角那一步是 `−4000,−4000` ✓，而固件对 `MOVE` 是
      **分块发送**的（`mouseMoveChunked`，每块 ≤120 ✓ 见本文件 L57-58 ✓）
      ⇒ 那是 **~34 块/轴** 的一串报文 ✓；紧接着就发 `mouse_move(3072, 203)` ✗
      ⇒ 只要撞角那串**还没发完/被后来的行顶掉** ✗，"角"就不是角 ✗
      ⇒ `+3072` 从错误的位置算起 ⇒ 落点整体偏 ✗，最典型就是被**屏幕右缘夹住** ✓
      （`0 + 3072 > 2559` ⇒ 停在 **2559** ✓ = 用户看到的"屏幕最右边"✓✓ 数字正好对上 ✓）。
      ⚠ 旁证：我**空着串口**实测时是"撞角 + `sleep 0.5`"⇒ 落点精确 ✓；差别就在这 ✗。
    ⚠ 读光标是**瞬时**的 ✓（`GetCursorPos` 系统调用 ✓）⇒ 不违反"实时回路不许 sleep"那条硬规矩 ✓
      （那条钉子见 `selftest_decision` ✓）。
    """
    try:
        import win32api
        return tuple(int(v) for v in win32api.GetCursorPos())
    except Exception:                           # noqa: BLE001 —— 读不到就照老路走 ✓
        return None


def _sync_aim_cur(frame_shape, gain_pair, origin):
    """把**瞄准的「当前位置」基准**同步成**真光标** ⇒ 同步到了返回 `True` ✓（瞬时 ✓ **不 sleep** ✗）。

    ⭐⭐⭐ 为什么非有它不可（用户 2026-10-17 ✓ 原话：
      「**鳄鱼潭项目1 自动测谎：鼠标没有指着准心，而是在屏幕很靠右的地方动**」✗）：
      `aim_tick` 在"位置不可信"时的老写法是「`corner_zero()` ＋ **立刻**当光标已在原点
      `(0,0)`」✗ —— 而撞角是 `−4000,−4000`、固件对 `MOVE` 是**分块发**的（~34 块/轴 ✓
      见本文件头 L57-58 ✓）⇒ **16ms 后**那一拍就已经按 `(0,0)` 算差量了 ✗
      ⇒ 撞角那串还没走完 / 被打断 ⇒ **基准整体偏** ✗ ⇒ **瞄准整体偏** ✓
      —— 症状就是「没指着准心、在很靠右的地方动」✓。
      （⚠ 与 `click_ratio` 411/412 那两轮是**同一类病** ✓ 同一个修法 ✓。）
    ⚠ 只在"**要指挥的就是这只光标**"时才算数 —— 这层由 `_own_cursor()` 判掉 ✓
      （网络 relay ⇒ 返回 `None` ⇒ 这里**什么都不做** ⇒ 老行为一字不变 ✓）。
    """
    cur = _own_cursor()
    if cur is None:
       # ⭐ **收流（relay）那条**：本机读不到那只光标 ✓ ⇒ 用后台采到的**真光标位置**
       #   （`_CUR_SCREEN` ✓ 见 `set_cursor_screen` ✓）折成指令空间备着 ✓
       #   ⇒ `aim_tick` 起算时就不必先撞角 ✓（用户 2026-10-17 ✓
       #   「**自动测谎时为什么鼠标要先到左上角再移过去 不能从当前位置**」✓）。
       #   ⚠ 只是**备好**，用不用由 `aim_tick` 判（采样要比上次移动更新 ✓）。
       try:
           _gx, _gy = screen_gain(tuple(gain_pair), frame_shape)
           _refresh_cur_hint(_gx, _gy)
       except Exception:                    # noqa: BLE001 —— 算不出来就当没有 ✓
           pass
       return False
    try:
        gx, gy = screen_gain(tuple(gain_pair), frame_shape)
        vx, vy = _virtual_origin()
        global _AIM_CUR, _AIM_AT
        _AIM_CUR = (int(round((cur[0] - vx) / gx)),
                    int(round((cur[1] - vy) / gy)))
        _AIM_AT = time.monotonic()
        return True
    except Exception:                        # noqa: BLE001 —— 同步不了就照老路 ✓
        return False


def _own_cursor():
    """**我们要指挥的那只光标**现在在哪 ⇒ `(x, y)`；问不到 / 不是这只 ⇒ `None` ✓（瞬时 ✓）。

    ⚠⚠ 为什么不能直接用 `_cursor_pos()`（2026-10-17 第二轮 ✓）：它读的是**本机**光标 ✓，
      只有"鼠标确实动在**本机**"时它才是**要指挥的那只** ✓：
        · **本地串口 ProMicro**（`SerialKbd` ✓）/ 本机 SendInput ⇒ 是 ✓
          （`SerialKbd` **没有** `cursor()` 方法 ✓ ⇒ 这里就能认出来 ✓）；
        · **网络 relay**（`KbdClient` ✓）⇒ 鼠标动在**被控机** ✗ ⇒ 本机读到的是**另一只** ✗
          ⇒ 拿它算差量会**点飞** ✗ ⇒ 返回 `None`，让调用方退回"撞角 + 绝对走位" ✓。
      ⚠ 与 `decision/input.cursor_probe()` 的分工：那条**能问 A 机**（relay 也对 ✓）
        但要走一次网络 ✗ ⇒ 只适合**后台线程** ✓；**实时回路线程**上只能读本机的
        （瞬时 ✓ 不阻塞 ✓ 不违反"不许 sleep"那条硬规矩 ✓）。
      ⚠ 判据用 `hasattr(_remote, "cursor")`（= relay 那个客户端 ✓ `input.cursor_probe`
        也是这么分工的 ✓）—— **别**拿类型名去猜 ✗。
    """
    try:
        _r = getattr(dinput, "_remote", None)
        if _r is not None and hasattr(_r, "cursor"):
            return None                      # relay：本机读到的不是要指挥的那只 ✗
        return _cursor_pos()
    except Exception:                        # noqa: BLE001
        return None


def repeat_click():
    """**再点一下**（不重新走位 ✓ —— 给「双击」的第二下用 ✓）。返回 `(ok, why)`。

    ⚠⚠ 为什么第二下要单独一个出口、为什么**不**在 `click_ratio` 里点两下：
      `click_ratio` 跑在**实时回路线程**里 —— 那里**不许 sleep** ✗（睡一下就是丢帧 / 积压，
      见它的说明 ✓）。而"双击"要求两下有间隔 ✓ ⇒ 只能由**状态机**在**下一拍**补
      （`decision/reconnect.py` 的 `DOUBLE_CLICK_UIS` / `DOUBLE_CLICK_GAP` ✓）。
      所以这一层只负责"**发一条左键**" ✓ —— 光标还在原地（我们没动过它 ✓）⇒ 不用再撞角走位 ✓。

    ⚠ 2026-10-07 真机验出来的用法：**频道面板**单击只是"选中" ✗ ⇒ 必须双击才进频道 ✓
      （服务器行单击就行 ✓ 别搞混 ✗ 见设计文档 §11 ✓）。
    """
    if not dinput.mouse_available():
        return False, "鼠标通道不可用（本地 SendInput 模式没有硬件鼠标）"
    dinput.mouse_click("left")
    return True, "补第二下左键（双击 ✓ 光标没动 ⇒ 不用重新走位 ✓）"


def scroll(up=0, down=0, frame_shape=None, xr=None, yr=None, origin=None):
    """滚轮：**先把光标放到要滚的地方**、再"向上 `up` 格 → 向下 `down` 格" ✓。

    ⚠⚠ **为什么必须带位置**（2026-10-09 真机修的 ✗ 用户原话："**我选的 22 频道，怎么断线重连
      去 2 了？**"）：滚轮**只作用在光标底下那个控件上** ✗。原来这条**直接发 `SCROLL`**，
      而那时光标还停在**上一步"点服务器"**的位置（画面上方 (~497,159) ⇒ 根本不在频道列表里 ✗）
      ⇒ **列表一格都没滚** ⇒ 接着照"第 1 屏第 1 行第 2 列"去点 ⇒ **进了频道 2** ✗✗
      （真机日志 23:16 那一段就是这么走的 ✓ 滚动那条 `ok=True` 骗过了所有断言 ✗ ——
      它只证明"指令发出去了"，证明不了"列表滚了" ✓）。
    ⇒ 现在走**同一套**：撞角归零 → 走到目标格 → 再滚 ✓ —— 三步都是**入队指令**，
      固件按顺序执行 ✓（同 `click_ratio` 的说明 ✓），所以这里**照旧不 sleep** ✓
      （滚完等生效仍由**状态机下一拍**负责 ✓ `reconnect.SCROLL_SETTLE` ✓）。

    `frame_shape` / `xr` / `yr`：不给 ⇒ 按**老行为**只发滚轮（老调用方 / 单测不受影响 ✓）；
      给了就必须**标定过**，否则**一个字节都不发** ✓（宁可不滚，也不在错的地方滚 ✓）。
    ⚠ 滚轮归这一层，别在 `live_thread` 里直接摸 `dinput` ✗（那边没有这个名字 ✓
      会当场 `NameError` ✓ —— 2026-10-08 真栽过，被 live_panel 的静态检查逮住 ✓）。
    """
    if not dinput.mouse_available():
        return False, "鼠标通道不可用（滚轮也发不出去）"
    up, down = int(up), int(abs(down))
    moved = ""
    if frame_shape is not None and xr is not None and yr is not None:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        if _auto_busy():
            # ⚠ 后台正在量标定 ⇒ **连光标都不许挪** ✗（见 `_AUTO_BUSY` ✓）
            return False, ("正在**后台自动量**鼠标标定 ✓ ⇒ 这一拍先不动 ✓"
                           "（量完下一拍就能点 ✓）")
        gain = load_gain(frame_shape)     # ⚠ 按**帧尺寸**取那一份 ✓
        if not gain:
            # ⭐⭐ 同上：这个尺寸没量过 ⇒ **起后台线程自己量** ✓（见 `auto_measure_async` ✓）
            _am_ok, _am_why = auto_measure_async(frame_shape)
            if _am_ok:
                return False, (_am_why
                               or "正在**后台自动量**标定 ✓ ⇒ 这一拍不动鼠标 ✓")
            gain = load_gain(frame_shape)
        if not gain:
            # 量不了 ⇒ **连光标都不挪** ✗（歪着滚 = 在别的地方滚 ✓）
            _blk, _blk_why = gain_mismatch_block(frame_shape)
            return False, ((_blk_why
                            or "鼠标没标定（缺 config/mouse_gain.json）⇒ 不在画面上乱滚")
                           + (("\n　（自动量这次没成：%s）" % _am_why) if _am_why else ""))
        px, why = target_px(w, h, xr, yr)
        if px is None:
            return False, why
        # ⚠ 先把**帧像素**折成**屏幕像素**（帧是裁出来的 ⇒ 差一个 `origin` ✓ 见 `to_screen` ✓）
        _sx, _sy = to_screen(px[0], px[1], origin, frame_shape)   # ①原点 ②比例（见它说明 ✓）
        _g = screen_gain(gain, frame_shape)                       # 帧像素口径 → 屏幕像素口径 ✓
        d, why = counts_for(_sx, _sy, _g[0], _g[1])
        if d is None:
            return False, why
        corner_zero()                    # ① 到已知绝对原点
        dinput.mouse_move(d[0], d[1])    # ② 走到要滚的那个控件上（否则滚轮打不到它 ✗）
        moved = ("撞角归零 → 走 (%d, %d) 指令单位（画面 %.0f, %.0f 像素）→ "
                 % (d[0], d[1], px[0], px[1]))
    if up:
        dinput.mouse_scroll(up)          # 正数 = 向上 ✓（见 `input.mouse_scroll` ✓）
    if down:
        dinput.mouse_scroll(-down)
    # ⚠ 文案要说清"实际发了什么" ✗：`up=0`（现在频道那一步就是这样 ✓ 见
    #   `reconnect.CHANNEL_WHEEL_*` 那段 ✓）⇒ **别再说"先往上滚 0 格到顶"** ✗（人会看不懂 ✓）。
    if up and down:
        _what = "先往上滚 %d 格、再往下滚 %d 格" % (up, down)
    elif up:
        _what = "往上滚 %d 格" % up
    else:
        _what = "往下滚 %d 格" % down
    _gw = gain_warn(frame_shape[1] if frame_shape is not None and len(frame_shape) > 1
                    else None)
    return True, ("%s%s ✓（点击在下一拍 ✓）%s"
                  % (moved, _what, ("　" + _gw) if _gw else ""))


#: 「瞄准」用的**上一次命令到的位置**（指令单位，从屏幕左上角算 ✓）—— `None` = 不知道
#: （下一拍要**先撞角归零一次**把原点建起来 ✓）。
#: ⚠ 这是**我们命令过的**位置，不是"OS 报回来的"（固件只有相对移动 ⇒ 没有那条路 ✗）⇒
#:   它只在"我们一直在动它"的那段时间可信 ✓；停顿久了 / 人碰过鼠标 / 游戏重设过光标 ⇒
#:   一律靠 `AIM_FRESH_S` 判"不可信"再重来 ✓（见 `aim_tick` ✓）。
_AIM_CUR = None
_AIM_AT = 0.0
#: 两拍之间**超过这么久 ⇒ 位置当作不可信**（人可能碰过鼠标 / 游戏重设过光标 ✓）。
AIM_FRESH_S = 1.0
#: 增量离谱（超过这个指令数）⇒ 说明位置漂了 ⇒ 重新撞角归零一次 ✓。
_AIM_MAX_STEP = 4000
#: **当前瞄准目标**（指令单位 ✓）与它是**什么时候**设的 —— 目标过期就不再追 ✓。
_AIM_TGT = None
_AIM_TGT_AT = 0.0
#: 目标多久没刷新 ⇒ **不再追**（人/那一轮已经不需要了 ✓ 同 SDK"点不新鲜就不补发"的精神 ✓）。
AIM_TGT_TTL_S = 0.25

#: ⭐⭐ **后台采到的「A 机光标现在在哪」**（屏幕像素 + 它自报的虚拟屏原点 ✓）与采集时刻 ✓。
#: 用户 2026-10-17 ✓ 原话：「**自动测谎时为什么鼠标要先到左上角再移过去 不能从当前位置**」✓
#: ⚠ 为什么以前必须撞角：固件只有**相对移动** ⇒ 要走到"绝对某点"就先得有个已知起点 ✓，
#:   而收流那条路**看不见 A 机光标**（`cursor_visible()` 恒 False ✓）⇒ 只能撞角建原点 ✗。
#: ⇒ 现在 relay 能答 `CURSOR?`（`decision/input.cursor_probe` ✓）——
#:   **后台线程**每秒问几次（见 `gui/live_thread._cur_loop` ✓）⇒ 起算点直接用**真光标位置** ✓
#:   ⇒ 瞄准**不再先跳左上角** ✓（用户要的"从当前位置移过去"✓）。
_CUR_SCREEN = None          # (sx, sy, vx, vy)
_CUR_SCREEN_AT = 0.0
#: 指令空间的同一件事（`aim_to` 拿得到增益时算好 ✓ 见 `_refresh_cur_hint` ✓）
_CUR_HINT = None            # (cx, cy)
_CUR_HINT_AT = 0.0
#: 采样比这更旧 ⇒ **不信**（这期间可能有人碰过鼠标 / 我们自己动过 ✓）
CUR_HINT_FRESH_S = 0.45


def set_cursor_screen(sx, sy, vx=0, vy=0, now=None):
    """后台线程喂进来：**A 机光标此刻在屏幕哪儿**（像素 ✓ 不阻塞 ✓ 纯赋值 ✓）。"""
    global _CUR_SCREEN, _CUR_SCREEN_AT
    _CUR_SCREEN = (float(sx), float(sy), int(vx), int(vy))
    _CUR_SCREEN_AT = time.monotonic() if now is None else float(now)


def _refresh_cur_hint(gx, gy):
    """把采样折成**指令空间**（`_AIM_CUR` 的单位 ✓）备用 —— 由 `aim_to` 顺手调 ✓。"""
    global _CUR_HINT, _CUR_HINT_AT
    if _CUR_SCREEN is None:
        _CUR_HINT, _CUR_HINT_AT = None, 0.0
        return
    sx, sy, vx, vy = _CUR_SCREEN
    _CUR_HINT = (int(round((sx - vx) / (float(gx) or 1.0))),
                 int(round((sy - vy) / (float(gy) or 1.0))))
    _CUR_HINT_AT = _CUR_SCREEN_AT        # ⚠ 记**采样时刻** ✗ 不是"现在"（否则等于假新鲜 ✓）


def aim_reset():
    """忘掉"目标 + 我知道光标在哪" ⇒ 下一拍瞄准会**先撞角归零一次**（见 `aim_tick` ✓）。

    什么时候该调：换了一轮 / 成功结束 / 人手动碰过鼠标之后（调用方知道而这里不知道的时刻 ✓）。
    """
    global _AIM_CUR, _AIM_AT, _AIM_TGT, _AIM_TGT_AT
    _AIM_CUR, _AIM_AT = None, 0.0
    _AIM_TGT, _AIM_TGT_AT = None, 0.0


def aim_to(frame_shape, xr, yr, gain=None, origin=None,
           roi=None, full_shape=None):
    """**设一个瞄准目标**（**不直接动鼠标** ✓）—— 换算仍是老那一处（`target_px` / `counts_for` ✓）。

    为什么是"设目标"而不是"一次发到位"（2026-10-09 现场 ✓ 用户原话："鼠标还是一顿一顿跳动的，
    不是顺滑的" ✗）：固件只有相对位移 ⇒ 一次发到位、下一拍才更新画面 ⇒ 光标只能**一跳一跳** ✓。
    要顺滑就必须**高频小步插值**（跟踪包的生产端也是这么做的：**8ms 插值定时器** ✓
    见 `docs/INTEGRATION.md:101` 那句"本适配器没有生产端 8ms 插值定时器" ✓）
    —— 所以这里**只设目标**，真正的走位交给 `aim_tick` ✓。

    返回 `(ok, why)`：闸与以前一样（标定 / 通道 / 比例 ✓），只是**这一下不动手** ✓。
    """
    global _AIM_TGT, _AIM_TGT_AT
    if frame_shape is None or len(frame_shape) < 2:
        return False, "拿不到画面尺寸"
    if _auto_busy():
        # ⚠⚠ 后台正在量标定（那条线程在动鼠标 ✓）⇒ 这里**一个字都不许发** ✗
        #   （两步插在一起 ⇒ 量出来的数是错的 ✓ 而且点也点不准 ✓）
        return False, ("正在**后台自动量**鼠标标定 ✓ ⇒ 这一拍先不动鼠标 ✓"
                       "（量完下一拍就能点 ✓）")
    _fs = full_shape if (full_shape is not None and len(full_shape) >= 2) else frame_shape
    h, w = int(_fs[0]), int(_fs[1])
    if gain is None:
        gain, _gwhy = _gain_or_why(_fs)
        if not gain:
            return False, _gwhy
    gx, gy = gain
    px, why = (roi_px(roi, frame_shape, xr, yr) if roi else target_px(w, h, xr, yr))
    if px is None:
        return False, why
    # ⚠ 先把**帧像素**折成**屏幕像素**（见 `to_screen` ✓ —— 副屏/挪过的窗口全靠这一步 ✓）
    _sx, _sy = to_screen(px[0], px[1], origin, _fs)
    _gx, _gy = screen_gain((gx, gy), _fs)             # 帧像素口径 → 屏幕像素口径 ✓
    d, why = counts_for(_sx, _sy, _gx, _gy)
    if d is None:
        return False, why
    if not dinput.mouse_available():
        return False, "鼠标通道不可用（本地 SendInput 模式没有硬件鼠标）"
    _AIM_TGT = d
    _AIM_TGT_AT = time.monotonic()
    # ⭐⭐⭐ **顺手把基准同步成真光标**（本地 ✓ 瞬时 ✓ 不 sleep ✗ —— 见 `_sync_aim_cur` 那段 ✓）：
    #   ⛔ 老口径是"等 `aim_tick` 自己撞角建原点"，而那一步**假设撞角瞬间到位** ✗
    #     ⇒ 瞄准整体偏 ✓（用户 2026-10-17：「**没指着准心、在很靠右的地方动**」✗）。
    #   ⚠ 它自己会判"这是不是要指挥的那只光标"（relay ⇒ 不动 ✓），失败也不影响下面 ✓。
    _sync_aim_cur(_fs, (_gx, _gy), origin)
    _gw = gain_warn(w)
    return True, ("瞄准目标 (%.0f, %.0f) 像素 ⇒ 指令 %r%s"
                  % (px[0], px[1], d, ("　" + _gw) if _gw else ""))


def aim_tick(tick_ms=16.0, settle_ms=120.0, now=None):
    """**朝目标走一小步**（高频调：例 16ms 一拍 ⇒ 60Hz ✓）⇒ 返回这一拍**有没有真发指令**。

    走法 = "这一拍该走的那一份"：`step = (目标 - 当前位置) * tick_ms / settle_ms` ✓
    ⇒ 大约 `settle_ms` 之后到点 ✓、中途**每一拍都发**（顺滑的来源 ✓）。

    ⚠ 目标超过 `AIM_TGT_TTL_S` 没刷新 ⇒ **停手**（不补发 ✓ —— 补发会让光标在过期点上乱晃 ✓）。
    ⚠ 位置不可信（第一拍 / 停久了）⇒ 先 `corner_zero()` 建原点一次 ✓（同 `click_ratio` 的前提 ✓）。
    """
    global _AIM_CUR, _AIM_AT
    now = time.monotonic() if now is None else float(now)
    if _AIM_TGT is None or (now - _AIM_TGT_AT) > AIM_TGT_TTL_S:
        return False
    if not dinput.mouse_available():
        return False
    if (_AIM_CUR is None) or (now - _AIM_AT > AIM_FRESH_S):
        # ⭐⭐ **能用真光标起算就用它**（用户 2026-10-17 ✓「为什么鼠标要先到左上角再移过去
        #    不能从当前位置」✓）——判据三重，缺一不用：
        #    ① 有采样 ✓；② 采样**比我们上一次动鼠标更新** ✓（`_CUR_HINT_AT >= _AIM_AT`，
        #       这样不会"用旧位置当现在"⇒ 不会往回跳 ✓）；
        #    ③ 采样不超过 `CUR_HINT_FRESH_S` ✓（这期间可能有人碰过鼠标 ✓）。
        #    ⇒ 起算点 = 光标**现在**在哪 ⇒ 直接朝目标走 ✓ **不再跳左上角** ✓；
        #    有一条不满足 ⇒ 老老实实撞角（宁可多跳一下，不能按错位置点 ✗）。
        _hint_ok = (_CUR_HINT is not None
                    and _CUR_HINT_AT >= _AIM_AT
                    and (now - _CUR_HINT_AT) <= CUR_HINT_FRESH_S)
        if _hint_ok:
            _AIM_CUR = _CUR_HINT
            _AIM_AT = now                   # ⚠ "我现在知道它在哪" ⇒ 下一步别再重复 seed ✓
        else:
            corner_zero()                   # 建原点（只在位置不可信时做 ✓）
            _AIM_CUR = (0, 0)
    dx = _AIM_TGT[0] - _AIM_CUR[0]
    dy = _AIM_TGT[1] - _AIM_CUR[1]
    if dx == 0 and dy == 0:
        _AIM_AT = now
        return False
    frac = max(0.05, min(1.0, float(tick_ms) / max(1.0, float(settle_ms))))
    sx = 0 if dx == 0 else (int(round(dx * frac)) or (1 if dx > 0 else -1))
    sy = 0 if dy == 0 else (int(round(dy * frac)) or (1 if dy > 0 else -1))
    sx = max(-abs(dx), min(abs(dx), sx)) if dx else 0
    sy = max(-abs(dy), min(abs(dy), sy)) if dy else 0
    dinput.mouse_move(sx, sy)
    _AIM_CUR = (_AIM_CUR[0] + sx, _AIM_CUR[1] + sy)
    _AIM_AT = now
    return True


# ⛔ `aim_ratio`（"一发到位"的那版）已**删除** ✗（2026-10-09 当天的事）：它虽然不撞角了，但
#   一次发到位、下一拍才更新 ⇒ 光标还是"跳" ✓ ⇒ 被 `aim_to` + `aim_tick`（高频插值）取代 ✓。


def cursor_visible():
    """**这只要指挥的光标看不看得见** ⇒ `True` / `False` / **`None`（说不准）** ✓
    （瞬时 ✓ 不走网络 ✓ **不 sleep** ✗）。

    ⚠ 干嘛用（用户 2026-10-17 ✓「**地铁二号线 无法选中频道**」✗）：
      · **看得见**（本地 ✓）⇒ `click_ratio` 走"**只发差量**"✓ ⇒ **根本不用撞角** ✗
        ⇒ 也不需要"撞角单独一拍"那道手续 ✓；
      · **看不见**（收流 relay ✓）⇒ 只能"撞角 + 绝对走位"✓ ⇒ **必须分两拍** ✓
        （否则撞角那串没走完就走位 ⇒ 点歪 ✓ 见 `corner_only` ✓）。
      ⚠ 这里**不**问 A 机 ✗（`KbdClient.cursor` 是**网络往返** ✓ 实时回路不许等 ✓）——
        relay 就是"看不见" ✓，这正是那一步要分拍的判据 ✓。

    ⚠⚠ **三态**（`True` 看得见 / `False` **确定**看不见 / `None` 说不准）：
      只有**确定看不见**（relay ✓）才值得多加"只归零"那一拍 ✓；
      **说不准**（读光标那一步本身失败 ✓ 比如没装 pywin32 的机器 ✓）⇒ **保持老行为** ✗
      —— 宁可不动，也不要因为"读不到"就给别人多插一拍 ✓（那会让**本地**那条路
      平白多一次撞角 + 多等一拍 ✓，还会把既有用例全逼红 ✓ 现场真栽过 ✓）。
    """
    try:
        _r = getattr(dinput, "_remote", None)
        if _r is not None and hasattr(_r, "cursor"):
            return False                     # 网络 relay：鼠标动在被控机 ⇒ 本机这只不是它 ✓
        return True if _cursor_pos() is not None else None
    except Exception:                        # noqa: BLE001 —— 说不准 ✓ 别改行为 ✗
        return None


def corner_only(frame_shape=None):
    """只**撞角归零**、不点 ⇒ `(ok, why)`（给"撞角和走位必须**分拍**"那条用 ✓ 见下 ✓）。

    ⚠⚠ 为什么要它（用户 2026-10-17 ✓ 原话：「**地铁二号线 无法选中频道**」✗）：
      **收流（relay）**那条路**读不到光标** ✗ ⇒ `click_ratio` 只能"撞角 + 绝对走位" ✓，
      而老写法是**同一拍**把两条一起发出去 ✗ —— `remote_kbd/A_relay_trace.log` 实证：
      ```
      10:18:15.022  tcp->serial MOVE -4000 -4000|   ← 撞角（固件要分 ~34 块/轴 才走完 ✗）
      10:18:15.022  tcp->serial MOVE 2625 1805|     ← 走位**同一毫秒**就排上去 ✗
      10:18:15.023  tcp->serial SCROLL -1|
      ```
      ⇒ **起点根本不是角** ✗ ⇒ 落点整体偏 ⇒ 频道格点不中 ⇒「无法选中频道」✓。
    ⇒ 正解（同 `reconnect` 里滚轮那条"**一格一拍**"✓）：**这一拍只撞角** ✓、
      **下一拍再走位 + 点** ✓（`click_ratio(..., homed=True)` ✓ 跳到"不撞角"那支 ✓）。
    ⚠ 守卫与 `click_ratio` **同一套** ✓（自动量标定在忙 ⇒ 一个字都不发 ✗；没标定 ⇒ 不点 ✗）。
    """
    if _auto_busy():
        return False, ("正在**后台自动量**鼠标标定 ✓ ⇒ 这一拍先不动鼠标 ✓")
    if frame_shape is not None and not load_gain(frame_shape):
        _am_ok, _am_why = auto_measure_async(frame_shape)
        if _am_ok or _auto_busy():
            return False, (_am_why or "正在**后台自动量**标定 ✓ ⇒ 这一拍不动鼠标 ✓")
        return False, "鼠标没标定（缺 config/mouse_gain.json）⇒ 先不撞角"
    if not dinput.mouse_available():
        return False, "鼠标通道不可用（本地 SendInput 模式没有硬件鼠标）"
    corner_zero()
    return True, "撞角归零（这一拍**只归零** ✓ 下一拍再走位 + 点 ✓ 撞角那串要走完 ✓）"


def _gain_or_why(frame_shape):
    """取**这个整帧尺寸**的标定；没有就按老规矩走「后台自动量」那条路 ✓。

    ⭐⭐⭐ 为什么收成一处（2026-10-17 ✓ 用户要求「**如果功能有耦合，优化代码结构**」✓）：
      `aim_to`（瞄准）与 `click_ratio`（撞角+走位+点）原来**各自抄了一份**这段兜底 ✗
      ⇒ 改一处漏一处 ⇒ 上一轮接 `roi` 时就漏了瞄准那一路 ✓（光标停在偏一块的地方再点 ✓）。
    ⭐ 返回 `(gain, why)`：`why` 非空 = **这一拍别动鼠标** ✓（原因人话 ✓ 直接
      `return False, why` ✓）；`gain` 非空 = 可以往下算 ✓。
    ⚠ **绝不在实时回路上等** ✗（`auto_measure_async` 是后台线程 ✓ 等一次最多 1.6 秒 ✗
      ⇒ 那条线程被卡住 = 丢帧 / 指令积压 ✓ 本仓库硬规矩 ✓）。
    """
    gain = load_gain(frame_shape)         # ⚠ 按**整帧尺寸**取那一份 ✓（见 `load_gain` ✓）
    if gain:
        return gain, ""
    _am_ok, _am_why = auto_measure_async(frame_shape)
    if _am_ok or _auto_busy():
        return None, (_am_why or "正在**后台自动量**标定 ✓ ⇒ 这一拍不动鼠标 ✓")
    gain = load_gain(frame_shape)
    if gain:
        return gain, ""
    _blk, _blk_why = gain_mismatch_block(frame_shape)
    return None, ((_blk_why or "鼠标没标定（缺 config/mouse_gain.json）")
                  + (("\n　（自动量这次没成：%s）" % _am_why) if _am_why else ""))


def click_ratio(frame_shape, xr, yr, gain=None, origin=None, homed=False,
                roi=None, full_shape=None):
    """**撞角归零 → 走位 → 左键点击**（这一层的唯一出口）。

    `homed=True`：**上一拍已经撞过角**（`corner_only()` ✓）⇒ 这一拍**不再撞角** ✗、
    直接走位 + 点 ✓（撞角与走位必须**分拍**，见 `corner_only` 的说明 ✓）。

    `frame_shape`：`frame.shape`（只要前两维 ✓）。
    `gain`：不给就现读 `config/mouse_gain.json`（`(gx, gy)` 或 None ✓）。

    返回 `(ok, why)`：`why` 一定是人话（成功也说清"走多少、点哪儿"✓ ——
    事后复盘只靠 `behavior.log`，看不到画面 ✓）。

    ⚠ **按顺序发三条指令**（`MOVE → MOVE → CLICK`）：固件按行顺序执行、
    OS 按到达顺序处理 HID 报文 ⇒ 按钮按下时系统里的光标**已经在目标上**了 ✓
    （`WM_LBUTTONDOWN` 带的是当时的坐标 ✓）。**所以这里不需要 sleep** ✗
    —— 这条函数跑在**实时回路线程**里，睡一下就是丢帧 / 积压（实测过代价 ✓）。
    """
    if frame_shape is None or len(frame_shape) < 2:
        return False, "拿不到画面尺寸"
    if _auto_busy():
        # ⚠⚠ 后台正在量标定（那条线程在动鼠标 ✓）⇒ 这里**一个字都不许发** ✗
        #   （两步插在一起 ⇒ 量出来的数是错的 ✓ 而且点也点不准 ✓）
        return False, ("正在**后台自动量**鼠标标定 ✓ ⇒ 这一拍先不动鼠标 ✓"
                       "（量完下一拍就能点 ✓）")
    _fs = full_shape if (full_shape is not None and len(full_shape) >= 2) else frame_shape
    h, w = int(_fs[0]), int(_fs[1])
    if gain is None:
        gain, _gwhy = _gain_or_why(_fs)
        if not gain:
            return False, _gwhy
    gx, gy = gain
    px, why = (roi_px(roi, frame_shape, xr, yr) if roi else target_px(w, h, xr, yr))
    if px is None:
        return False, why
    # ⚠ 先把**帧像素**折成**屏幕像素**（见 `to_screen` ✓ —— 副屏/挪过的窗口全靠这一步 ✓）
    # ⛔ 「没实测就按 link.yaml 折 k」那一版**已撤回**（只折一边 ⇒ 偏右下越过 ✓
    #   见 `to_screen` 里那段说明 ✓）
    _sx, _sy = to_screen(px[0], px[1], origin, _fs)
    _gx, _gy = screen_gain((gx, gy), _fs)             # 帧像素口径 → 屏幕像素口径 ✓
    d, why = counts_for(_sx, _sy, _gx, _gy)
    if d is None:
        return False, why
    if not dinput.mouse_available():
        return False, "鼠标通道不可用（本地 SendInput 模式没有硬件鼠标）"
    dx, dy = d
    _vx, _vy = _virtual_origin()
    # ⭐⭐⭐ **光标位置拿得到 ⇒ 只发"差量"，根本不用撞角**（2026-10-17 第二轮 ✓）
    #   ⛔ 为什么把 411 那版（"先只归零、下一拍再走位"✗）**撤掉**：它**吃重试次数** ✗ ——
    #     "点服务器"那一步每发一次动作、等满 `reconnect_step_timeout_ms` 才算一次 ✓
    #     ⇒ "只归零"那一拍等于**白吃一次** ✗ ⇒ 3 次用光 ⇒ **重连放弃** ✓
    #     ⇒ 光标被留在角上 ✓✓ = 用户 2026-10-17 现场原话
    #     「**现在鼠标停在屏幕左上角一动不动**」✗（一字不差 ✓ 而这也正是本文件
    #     L567-588 记着的那个老症状 ✓ 别再犯第三次 ✗）。
    #   ⭐ 正解：**拿绝对读到的光标当基准** ✓（本地读本机 `GetCursorPos` ✓ **瞬时** ✓
    #     **绝不 sleep** ✗）⇒ 差量通常很小 ⇒ 既不会被"撞角那串还没发完"咬 ✓
    #     也不吃重试 ✓ 更不会把光标留在角上 ✓。
    #   ⚠ 只在"**要指挥的就是这只光标**"时才能这么算 ✓（见 `_own_cursor` ✓）——
    #     网络 relay 时本机读到的是**另一台**的 ✗ ⇒ 用它算差量会点飞 ✗ ⇒ 那种情况退回老路 ✓。
    _cur = _own_cursor()
    if _cur is not None:
        _ddx = int(round(dx - (_cur[0] - _vx) / _gx))
        _ddy = int(round(dy - (_cur[1] - _vy) / _gy))
        if _ddx or _ddy:
            dinput.mouse_move(_ddx, _ddy)   # ① 从**现在的位置**走差量（小步 ✓）
        dinput.mouse_click("left")          # ② 点
        return True, ("按现在光标 (%d, %d) 走差量 (%d, %d) 指令单位 → 左键点击"
                      "（画面 (%.0f, %.0f) 像素 ⇒ 屏幕 (%.0f, %.0f)）%s"
                      % (_cur[0], _cur[1], _ddx, _ddy, px[0], px[1], _sx, _sy,
                         ("　" + _w) if (_w := gain_warn(w)) else ""))
    # ⚠ 拿不到"我们要指挥的那只光标"（网络 relay ✓ / 极少数读不到 ✓）⇒ **一字退回老路** ✓
    #   （撞角 + 从虚拟原点走绝对量 ✓；老路那个"撞角那串还没发完就走位"的毛病照旧留着 ✗ ——
    #    但它**不会**把重连的重试次数吃光 ✓ 也不会把光标留在角上 ✓，比 411 那版安全 ✓）
    if not homed:
        # ⭐⭐⭐ 撞角**只在"这一拍还没归过零"时发**（用户 2026-10-17 ✓ 原话：
        #   「**地铁二号线 无法选中频道**」✗）—— 撞角那串固件要分块发完（~34 块/轴 ✗），
        #   同一拍接着走位 ⇒ **起点不是角** ⇒ 落点整体偏 ⇒ 频道格点不中 ✓
        #   （`remote_kbd/A_relay_trace.log` 实证：两条 `MOVE` 同一毫秒 ✓）。
        #   `homed=True` = 上一拍已经 `corner_only()` 归过了 ✓ ⇒ 这一拍直接走位 ✓。
        corner_zero()                   # ① 到已知绝对原点
    dinput.mouse_move(dx, dy)           # ② 走到目标（x 用 gx、y 用 gy ✓）
    dinput.mouse_click("left")          # ③ 点
    # ⭐ **口径提醒**（2026-10-10 ✓）：标定那一帧的尺寸跟现在这帧差得多 ⇒
    #   把话说在回执里 ✓（它会进 `behavior.log` 的状态行 ✓）—— 别再让"点歪"静默发生 ✗。
    return True, ("撞角归零 → 走 (%d, %d) 指令单位 → 左键点击"
                  "（画面 (%.0f, %.0f) 像素 ⇒ 屏幕 (%.0f, %.0f)，比例 %.3f, %.3f）%s"
                  % (dx, dy, px[0], px[1], _sx, _sy, xr, yr,
                     ("　" + _w) if (_w := gain_warn(w)) else ""))
