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


def gain_frame_key(frame_shape):
    """帧尺寸 → 那一份标定的键 `"1366x768"` ✓（`frame_shape` = `(h, w, …)` ✓）。"""
    try:
        h, w = int(frame_shape[0]), int(frame_shape[1])
    except (TypeError, ValueError, IndexError):
        return ""
    if w <= 0 or h <= 0:
        return ""
    return "%dx%d" % (w, h)


def _read_gain_file():
    """⇒ `(平铺那份, {尺寸键: 记录})`（读不出来 ⇒ `({}, {})` ✓）。"""
    import json
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
    return _ok_entry(by.get(key) or {})


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
    corner_zero()                                # ① 到已知原点（那台机器的虚拟屏左上角 ✓）
    p0 = _probe_cursor()
    if p0 is None:
        _restore_cursor(pre, None)               # ⚠ 撞角了就得负责放回去 ✗
        return False, "撞角之后读不到光标 ⇒ 自动量不了 ✗"
    dinput.mouse_move(AUTO_STEP, AUTO_STEP)      # ② 走一步（两个轴一起走，一次量两轴 ✓）
    p1 = _probe_cursor()
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


#: 自动量**正在后台跑**吗 ✓（跑的时候**谁都不许再动鼠标** ✗ —— 两步插在一起就白量了 ✓）。
_AUTO_BUSY = False
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
    global _AUTO_BUSY
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
    if _AUTO_BUSY:
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
    _AUTO_TRIES[key] = _AUTO_TRIES.get(key, 0) + 1

    def _run():
        global _AUTO_BUSY
        try:
            with _AUTO_LOCK:
                _ok, _w = auto_measure(frame_shape, force=True)
            if not _ok:
                _AUTO_FAIL[key] = (time.monotonic(), str(_w)[:120])
        except Exception as e:                   # noqa: BLE001 —— 后台线程不许把谁炸了 ✗
            _AUTO_FAIL[key] = (time.monotonic(), "%s: %s" % (type(e).__name__, e))
        finally:
            _AUTO_BUSY = False

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


def to_screen(px, py, origin=None):
    """**画面像素** → **屏幕像素**（`origin` = 这一帧在屏幕上的左上角 ✓）。

    ⚠⚠ 为什么必须有这一步（2026-10-10 ✓ 用户原话：**"我可能会把游戏窗口随意拖动，并且
      游戏窗口可能会被拖到副显示屏"** ✓）：帧是**从你框选那块区域裁出来的** ✗ ⇒ 帧里的
      `(0, 0)` 是**那块区域的左上角** ✓、不是屏幕原点 ✗。窗口拖到副屏（x<0 ✓）时，
      "帧像素"与"屏幕像素"差的正是这个 `origin` ✓ —— 不补上 ⇒ 点击落点整体偏一整块区域 ✗
      （老版本在副屏上则是**根本够不着** ✓ 见 `counts_for` 的说明 ✓）。
    `origin` 不给 ⇒ 当 `(0, 0)`（帧就在主屏原点 ⇒ 与老行为一字不差 ✓）。
    """
    ox, oy = (0, 0) if not origin else (int(origin[0]), int(origin[1]))
    return float(px) + ox, float(py) + oy


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
        if _AUTO_BUSY:
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
        _sx, _sy = to_screen(px[0], px[1], origin)
        d, why = counts_for(_sx, _sy, gain[0], gain[1])
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


def aim_reset():
    """忘掉"目标 + 我知道光标在哪" ⇒ 下一拍瞄准会**先撞角归零一次**（见 `aim_tick` ✓）。

    什么时候该调：换了一轮 / 成功结束 / 人手动碰过鼠标之后（调用方知道而这里不知道的时刻 ✓）。
    """
    global _AIM_CUR, _AIM_AT, _AIM_TGT, _AIM_TGT_AT
    _AIM_CUR, _AIM_AT = None, 0.0
    _AIM_TGT, _AIM_TGT_AT = None, 0.0


def aim_to(frame_shape, xr, yr, gain=None, origin=None):
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
    if _AUTO_BUSY:
        # ⚠⚠ 后台正在量标定（那条线程在动鼠标 ✓）⇒ 这里**一个字都不许发** ✗
        #   （两步插在一起 ⇒ 量出来的数是错的 ✓ 而且点也点不准 ✓）
        return False, ("正在**后台自动量**鼠标标定 ✓ ⇒ 这一拍先不动鼠标 ✓"
                       "（量完下一拍就能点 ✓）")
    h, w = int(frame_shape[0]), int(frame_shape[1])
    if gain is None:
        gain = load_gain(frame_shape)     # ⚠ 按**帧尺寸**取那一份 ✓（见 `load_gain` ✓）
    if not gain:
        # ⭐⭐ **这个尺寸没量过 ⇒ 起一条后台线程自己量** ✓（用户 2026-10-10："**不能自动吗？**"✓
        #   见 `auto_measure_async` ✓）：⚠ **绝不在实时回路上等** ✗（等一次最多 1.6 秒 ✗
        #   ⇒ 那条线程被卡住 = 丢帧 / 指令积压 ✓ 本仓库硬规矩 ✓）
        #   ⇒ 这一拍先不动鼠标、**下一拍量好了就接着点** ✓（状态机本来就会重试 ✓）；
        #   只有"**起不来**"（冷却中 / 看不见光标 ✓）才回落到"说清 + 不点" ✓。
        _am_ok, _am_why = auto_measure_async(frame_shape)
        if _am_ok or _AUTO_BUSY:
            return False, (_am_why or "正在**后台自动量**标定 ✓ ⇒ 这一拍不动鼠标 ✓")
        gain = load_gain(frame_shape)
        if not gain:
            _blk, _blk_why = gain_mismatch_block(frame_shape)
            return False, ((_blk_why or "鼠标没标定（缺 config/mouse_gain.json）")
                           + (("\n　（自动量这次没成：%s）" % _am_why) if _am_why else ""))
    gx, gy = gain
    px, why = target_px(w, h, xr, yr)
    if px is None:
        return False, why
    # ⚠ 先把**帧像素**折成**屏幕像素**（见 `to_screen` ✓ —— 副屏/挪过的窗口全靠这一步 ✓）
    _sx, _sy = to_screen(px[0], px[1], origin)
    d, why = counts_for(_sx, _sy, gx, gy)
    if d is None:
        return False, why
    if not dinput.mouse_available():
        return False, "鼠标通道不可用（本地 SendInput 模式没有硬件鼠标）"
    _AIM_TGT = d
    _AIM_TGT_AT = time.monotonic()
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
        corner_zero()                       # 建原点（只在位置不可信时做 ✓）
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


def click_ratio(frame_shape, xr, yr, gain=None, origin=None):
    """**撞角归零 → 走位 → 左键点击**（这一层的唯一出口）。

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
    if _AUTO_BUSY:
        # ⚠⚠ 后台正在量标定（那条线程在动鼠标 ✓）⇒ 这里**一个字都不许发** ✗
        #   （两步插在一起 ⇒ 量出来的数是错的 ✓ 而且点也点不准 ✓）
        return False, ("正在**后台自动量**鼠标标定 ✓ ⇒ 这一拍先不动鼠标 ✓"
                       "（量完下一拍就能点 ✓）")
    h, w = int(frame_shape[0]), int(frame_shape[1])
    if gain is None:
        gain = load_gain(frame_shape)     # ⚠ 按**帧尺寸**取那一份 ✓（见 `load_gain` ✓）
    if not gain:
        # ⭐⭐ **这个尺寸没量过 ⇒ 起一条后台线程自己量** ✓（用户 2026-10-10："**不能自动吗？**"✓
        #   见 `auto_measure_async` ✓）：⚠ **绝不在实时回路上等** ✗（等一次最多 1.6 秒 ✗
        #   ⇒ 那条线程被卡住 = 丢帧 / 指令积压 ✓ 本仓库硬规矩 ✓）
        #   ⇒ 这一拍先不动鼠标、**下一拍量好了就接着点** ✓（状态机本来就会重试 ✓）；
        #   只有"**起不来**"（冷却中 / 看不见光标 ✓）才回落到"说清 + 不点" ✓。
        _am_ok, _am_why = auto_measure_async(frame_shape)
        if _am_ok or _AUTO_BUSY:
            return False, (_am_why or "正在**后台自动量**标定 ✓ ⇒ 这一拍不动鼠标 ✓")
        gain = load_gain(frame_shape)
        if not gain:
            _blk, _blk_why = gain_mismatch_block(frame_shape)
            return False, ((_blk_why or "鼠标没标定（缺 config/mouse_gain.json）")
                           + (("\n　（自动量这次没成：%s）" % _am_why) if _am_why else ""))
    gx, gy = gain
    px, why = target_px(w, h, xr, yr)
    if px is None:
        return False, why
    # ⚠ 先把**帧像素**折成**屏幕像素**（见 `to_screen` ✓ —— 副屏/挪过的窗口全靠这一步 ✓）
    _sx, _sy = to_screen(px[0], px[1], origin)
    d, why = counts_for(_sx, _sy, gx, gy)
    if d is None:
        return False, why
    if not dinput.mouse_available():
        return False, "鼠标通道不可用（本地 SendInput 模式没有硬件鼠标）"
    dx, dy = d
    corner_zero()                       # ① 到已知绝对原点
    dinput.mouse_move(dx, dy)           # ② 走到目标（x 用 gx、y 用 gy ✓）
    dinput.mouse_click("left")          # ③ 点
    # ⭐ **口径提醒**（2026-10-10 ✓）：标定那一帧的尺寸跟现在这帧差得多 ⇒
    #   把话说在回执里 ✓（它会进 `behavior.log` 的状态行 ✓）—— 别再让"点歪"静默发生 ✗。
    return True, ("撞角归零 → 走 (%d, %d) 指令单位 → 左键点击"
                  "（画面 (%.0f, %.0f) 像素 ⇒ 屏幕 (%.0f, %.0f)，比例 %.3f, %.3f）%s"
                  % (dx, dy, px[0], px[1], _sx, _sy, xr, yr,
                     ("　" + _w) if (_w := gain_warn(w)) else ""))
