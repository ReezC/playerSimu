# -*- coding: utf-8 -*-
"""鼠标**绝对定位**（断线重连「点服务器 / 点频道」用的那一层）。

分工（与 `decision/reconnect.py` 的模块头同一条纪律 —— **一处实现**）：

    `decision/reconnect.py`   什么时候点、点**画面上的哪个比例位置**（策略）
    `execution/mouse_aim.py`   ① 比例 → 像素 → 指令单位 的**换算**；② 撞角归零 + 走位 + 左键（发输入）
    `execution/input.py`       真正把 `MOVE` / `CLICK` 发到 Pro Micro

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

⚠ 与 `liedetect/lie_controller.load_gain()` **不是同一个读者**：那个是测谎闭环用的，
坏文件时**退回 1.0**（闭环能自己纠 ✓）；这一层是**开环绝对定位**，退回 1.0 就是
点错地方 ✗ ⇒ 这里**缺文件一律返回 None**（判据不同，故意不合并 ✓）。
两个读者读的是**同一个文件、同一套单位**（"px/单位" ✓）。
"""

from execution import remote as _remote_mod
import threading
import time
from pathlib import Path

from execution import input as dinput
from execution import aim_convert as conv


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
           _gx, _gy = conv.screen_gain(tuple(gain_pair), frame_shape)
           _refresh_cur_hint(_gx, _gy)
       except Exception:                    # noqa: BLE001 —— 算不出来就当没有 ✓
           pass
       return False
    try:
        gx, gy = conv.screen_gain(tuple(gain_pair), frame_shape)
        vx, vy = conv._virtual_origin()
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
      ⚠ 与 `execution/input.cursor_probe()` 的分工：那条**能问 A 机**（relay 也对 ✓）
        但要走一次网络 ✗ ⇒ 只适合**后台线程** ✓；**实时回路线程**上只能读本机的
        （瞬时 ✓ 不阻塞 ✓ 不违反"不许 sleep"那条硬规矩 ✓）。
      ⚠ 判据用 `hasattr(_remote, "cursor")`（= relay 那个客户端 ✓ `input.cursor_probe`
        也是这么分工的 ✓）—— **别**拿类型名去猜 ✗。
    """
    try:
        _r = getattr(_remote_mod, "_remote", None)
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
    ⇒ 现在走**同一套**：**走到目标格 → 再滚** ✓ —— 都是**入队指令**，
      固件按顺序执行 ✓（同 `click_ratio` 的说明 ✓），所以这里**照旧不 sleep** ✓
      （滚完等生效仍由**状态机下一拍**负责 ✓ `reconnect.SCROLL_SETTLE` ✓）。

    ⚠⚠ **每一步都不许"先砸左上角"**（用户 2026-10-20 ✓ 原话："选频道界面鼠标乱飞"✗）：
      光标位置问得到（`_own_cursor` ✓）⇒ 从**现在的位置**走差量 ✓（第 1 格之后差量都是 0
      ⇒ **只发滚轮** ✓）；问不到（网络 relay ✓）才退回"撞角 + 绝对走位" ✓
      —— 那一版每格横穿全屏一次（3236 指令单位 ×9 格）⇒ 视觉上就是"乱飞"✓
      而且九次横穿里只要有一次光标没落到列表上，那一格滚轮就**白滚**（面板截图逐像素相同 ✓）。

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
        if conv._auto_busy():
            # ⚠ 后台正在量标定 ⇒ **连光标都不许挪** ✗（见 `_AUTO_BUSY` ✓）
            return False, ("正在**后台自动量**鼠标标定 ✓ ⇒ 这一拍先不动 ✓"
                           "（量完下一拍就能点 ✓）")
        gain = conv.load_gain(frame_shape)     # ⚠ 按**帧尺寸**取那一份 ✓
        if not gain:
            # ⭐⭐ 同上：这个尺寸没量过 ⇒ **起后台线程自己量** ✓（见 `auto_measure_async` ✓）
            _am_ok, _am_why = conv.auto_measure_async(frame_shape)
            if _am_ok:
                return False, (_am_why
                               or "正在**后台自动量**标定 ✓ ⇒ 这一拍不动鼠标 ✓")
            gain = conv.load_gain(frame_shape)
        if not gain:
            # 量不了 ⇒ **连光标都不挪** ✗（歪着滚 = 在别的地方滚 ✓）
            _blk, _blk_why = conv.gain_mismatch_block(frame_shape)
            return False, ((_blk_why
                            or "鼠标没标定（缺 config/mouse_gain.json）⇒ 不在画面上乱滚")
                           + (("\n　（自动量这次没成：%s）" % _am_why) if _am_why else ""))
        px, why = conv.target_px(w, h, xr, yr)
        if px is None:
            return False, why
        # ⚠ 先把**帧像素**折成**屏幕像素**（帧是裁出来的 ⇒ 差一个 `origin` ✓ 见 `to_screen` ✓）
        _sx, _sy = conv.to_screen(px[0], px[1], origin, frame_shape)   # ①原点 ②比例（见它说明 ✓）
        _g = conv.screen_gain(gain, frame_shape)                       # 帧像素口径 → 屏幕像素口径 ✓
        d, why = conv.counts_for(_sx, _sy, _g[0], _g[1])
        if d is None:
            return False, why
        # ⭐⭐⭐ **光标位置拿得到 ⇒ 只走差量，绝不撞角**（用户 2026-10-20 ✓ 原话：
        #   "**选频道界面鼠标乱飞**"✗）
        #   病根：这条原来**每一格**都 `corner_zero()` + 走绝对量 ✗ ⇒ 频道 54 要滚 9 格
        #   ⇒ **9 次"砸左上角 → 横穿整个屏幕飞回来"**（每次 3000+ 指令单位 ✓ 见
        #   `behavior.log` 08:57 那串 "撞角归零 → 走 (3236, 547)" ✓）⇒ 看着就是"乱飞"✓。
        #   而且每次横穿都给了客户端一次**把滚轮吃掉**的机会（08:57:42 vs 08:57:49 两张
        #   面板截图**逐像素相同** ⇒ 那 9 格一格都没滚 ✗ ⇒ 接着点到错的行 ✗）。
        #   ⇒ 与 `click_ratio` **同一套判据**（`_own_cursor` ✓ 见它的说明 ✓）：
        #     第 1 格才需要"走过去"，后面几格光标已经在原地 ⇒ 差量 (0,0) ⇒ **只发滚轮** ✓
        #     ⇒ 不飞了 ✓ 滚轮也稳了（一直停在列表上 ✓）。
        _cur = _own_cursor()
        _ddx = _ddy = 0
        if _cur is not None:
            _vx, _vy = conv._virtual_origin()
            _ddx = int(round(d[0] - (_cur[0] - _vx) / _g[0]))
            _ddy = int(round(d[1] - (_cur[1] - _vy) / _g[1]))
            if abs(_ddx) > _AIM_MAX_STEP or abs(_ddy) > _AIM_MAX_STEP:
                _cur = None               # 差量离谱 ⇒ 位置不可信 ⇒ 老老实实撞角 ✓
        if _cur is None:
            # ⚠ 拿不到"要指挥的那只光标"（网络 relay ✓）⇒ 一字退回老路 ✓
            conv.corner_zero()                # ① 到已知绝对原点
            dinput.mouse_move(d[0], d[1])     # ② 走到要滚的那个控件上（否则滚轮打不到它 ✗）
            moved = ("撞角归零 → 走 (%d, %d) 指令单位（画面 %.0f, %.0f 像素）→ "
                     % (d[0], d[1], px[0], px[1]))
        else:
            if _ddx or _ddy:
                dinput.mouse_move(_ddx, _ddy)
            moved = ("按现在光标 (%d, %d) 走差量 (%d, %d) 指令单位（画面 %.0f, %.0f 像素）→ "
                     % (_cur[0], _cur[1], _ddx, _ddy, px[0], px[1]))
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
    _gw = conv.gain_warn(frame_shape[1] if frame_shape is not None and len(frame_shape) > 1
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
#: ⇒ 现在 relay 能答 `CURSOR?`（`execution/input.cursor_probe` ✓）——
#:   **后台线程**每秒问几次（见 `gui/live_thread._cur_loop` ✓）⇒ 起算点直接用**真光标位置** ✓
#:   ⇒ 瞄准**不再先跳左上角** ✓（用户要的"从当前位置移过去"✓）。
_CUR_SCREEN = None          # (sx, sy, vx, vy)
_CUR_SCREEN_AT = 0.0
#: 指令空间的同一件事（`aim_to` 拿得到增益时算好 ✓ 见 `_refresh_cur_hint` ✓）
_CUR_HINT = None            # (cx, cy)
_CUR_HINT_AT = 0.0
#: 采样比这更旧 ⇒ **不信**（这期间可能有人碰过鼠标 / 我们自己动过 ✓）
CUR_HINT_FRESH_S = 0.45
#: ⭐⭐ **「撞角归零」的账**（用户 2026-10-20 ✓ 用户这套是**远程**：画面收流、键鼠走 relay ✓）：
#:   `_HOME_N` 累计次数 · `_HOME_WHY` 最后一次为什么 · `_HOME_AT` 最后一次的时刻。
#:   为什么要记（现场那句"**鼠标没有跟上准心的位置**"✗）：
#:     每次撞角 ⇒ 光标**先飞左上角再走回来**（用户 2026-10-17 就问过"为什么先到左上角"✓）⇒
#:     如果 relay 不给 `CURSOR?` 采样，**每次目标过期（`AIM_TGT_TTL_S`）+ 位置过期（`AIM_FRESH_S`）
#:     都会撞一次角** ⇒ 看着就是"光标乱跑、不跟准心"✓。
#:   ⇒ 现在三个数都留账：是"relay 不给采样"、"采样过期"，还是"从来没采到" —— 一眼分得清 ✓
#:     （日志见 `gui/live_thread._vt_aim_loop` 的 `vt_aim_home` ✓）。
_HOME_N = 0
_HOME_WHY = ""
_HOME_AT = 0.0


def aim_health():
    """瞄准基准的健康状况 ⇒ 纯读字典（日志 / 状态行用 ✓）。

    · `homes`     累计**撞角归零**次数（涨得快 = 光标在乱飞 ✓）；
    · `home_why`  最后一次为什么撞角（人话 ✓）；
    · `cur_age`   A 机光标采样有多旧（`None` = **从来没采到** ⇒ relay 的 `CURSOR?` 没通 ✓）；
    · `hint_age`  折成指令空间那份采样有多旧 ✓。
    """
    _now = time.monotonic()
    return {"homes": int(_HOME_N),
            "home_why": str(_HOME_WHY),
            "cur_age": (round(_now - float(_CUR_SCREEN_AT), 2) if _CUR_SCREEN else None),
            "hint_age": (round(_now - float(_CUR_HINT_AT), 2) if _CUR_HINT else None)}


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
    if conv._auto_busy():
        # ⚠⚠ 后台正在量标定（那条线程在动鼠标 ✓）⇒ 这里**一个字都不许发** ✗
        #   （两步插在一起 ⇒ 量出来的数是错的 ✓ 而且点也点不准 ✓）
        return False, ("正在**后台自动量**鼠标标定 ✓ ⇒ 这一拍先不动鼠标 ✓"
                       "（量完下一拍就能点 ✓）")
    _fs = full_shape if (full_shape is not None and len(full_shape) >= 2) else frame_shape
    h, w = int(_fs[0]), int(_fs[1])
    if gain is None:
        gain, _gwhy = conv._gain_or_why(_fs)
        if not gain:
            return False, _gwhy
    gx, gy = gain
    px, why = (conv.roi_px(roi, frame_shape, xr, yr) if roi else conv.target_px(w, h, xr, yr))
    if px is None:
        return False, why
    # ⚠ 先把**帧像素**折成**屏幕像素**（见 `to_screen` ✓ —— 副屏/挪过的窗口全靠这一步 ✓）
    _sx, _sy = conv.to_screen(px[0], px[1], origin, _fs)
    _gx, _gy = conv.screen_gain((gx, gy), _fs)             # 帧像素口径 → 屏幕像素口径 ✓
    d, why = conv.counts_for(_sx, _sy, _gx, _gy)
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
    _gw = conv.gain_warn(w)
    return True, ("瞄准目标 (%.0f, %.0f) 像素 ⇒ 指令 %r%s"
                  % (px[0], px[1], d, ("　" + _gw) if _gw else ""))


def aim_tick(tick_ms=16.0, settle_ms=120.0, now=None):
    """**朝目标走一小步**（高频调：例 16ms 一拍 ⇒ 60Hz ✓）⇒ 返回这一拍**有没有真发指令**。

    走法 = "这一拍该走的那一份"：`step = (目标 - 当前位置) * tick_ms / settle_ms` ✓
    ⇒ 大约 `settle_ms` 之后到点 ✓、中途**每一拍都发**（顺滑的来源 ✓）。

    ⚠ 目标超过 `AIM_TGT_TTL_S` 没刷新 ⇒ **停手**（不补发 ✓ —— 补发会让光标在过期点上乱晃 ✓）。
    ⚠ 位置不可信（第一拍 / 停久了）⇒ 先 `corner_zero()` 建原点一次 ✓（同 `click_ratio` 的前提 ✓）。
    """
    global _AIM_CUR, _AIM_AT, _HOME_N, _HOME_WHY, _HOME_AT
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
            # ⭐ 撞角要**留账**（用户 2026-10-20 ✓ 远程这套"鼠标没跟上准心"就靠它定位病因 ✓）：
            #   每次撞角光标都会**先飞左上角再走回来** ✗ ⇒ 现场看着就是"乱跑"✓；
            #   三个原因分开写，才能一眼分出是 relay 的事还是我们自己的事 ✓ 见 `aim_health` ✓。
            _HOME_N += 1
            if _CUR_HINT is None:
                _HOME_WHY = ("从来没有 A 机光标采样（relay 的 `CURSOR?` 没回 / 后端不是 relay）"
                             if _CUR_SCREEN is None else
                             "光标采样还没折成指令空间（增益没加载 ✓）")
            elif _CUR_HINT_AT < _AIM_AT:
                _HOME_WHY = ("光标采样比我们上次动鼠标还旧（%.2fs 前录的）⇒ 不敢拿它当现在"
                             % max(0.0, now - float(_CUR_HINT_AT)))
            else:
                _HOME_WHY = ("光标采样过期（%.2fs > %.2fs）⇒ 这期间可能有人碰过鼠标"
                             % (now - float(_CUR_HINT_AT), CUR_HINT_FRESH_S))
            _HOME_AT = now
            conv.corner_zero()                   # 建原点（只在位置不可信时做 ✓）
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
        _r = getattr(_remote_mod, "_remote", None)
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
    if conv._auto_busy():
        return False, ("正在**后台自动量**鼠标标定 ✓ ⇒ 这一拍先不动鼠标 ✓")
    if frame_shape is not None and not conv.load_gain(frame_shape):
        _am_ok, _am_why = conv.auto_measure_async(frame_shape)
        if _am_ok or conv._auto_busy():
            return False, (_am_why or "正在**后台自动量**标定 ✓ ⇒ 这一拍不动鼠标 ✓")
        return False, "鼠标没标定（缺 config/mouse_gain.json）⇒ 先不撞角"
    if not dinput.mouse_available():
        return False, "鼠标通道不可用（本地 SendInput 模式没有硬件鼠标）"
    conv.corner_zero()
    return True, "撞角归零（这一拍**只归零** ✓ 下一拍再走位 + 点 ✓ 撞角那串要走完 ✓）"


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
    if conv._auto_busy():
        # ⚠⚠ 后台正在量标定（那条线程在动鼠标 ✓）⇒ 这里**一个字都不许发** ✗
        #   （两步插在一起 ⇒ 量出来的数是错的 ✓ 而且点也点不准 ✓）
        return False, ("正在**后台自动量**鼠标标定 ✓ ⇒ 这一拍先不动鼠标 ✓"
                       "（量完下一拍就能点 ✓）")
    _fs = full_shape if (full_shape is not None and len(full_shape) >= 2) else frame_shape
    h, w = int(_fs[0]), int(_fs[1])
    if gain is None:
        gain, _gwhy = conv._gain_or_why(_fs)
        if not gain:
            return False, _gwhy
    gx, gy = gain
    px, why = (conv.roi_px(roi, frame_shape, xr, yr) if roi else conv.target_px(w, h, xr, yr))
    if px is None:
        return False, why
    # ⚠ 先把**帧像素**折成**屏幕像素**（见 `to_screen` ✓ —— 副屏/挪过的窗口全靠这一步 ✓）
    # ⛔ 「没实测就按 link.yaml 折 k」那一版**已撤回**（只折一边 ⇒ 偏右下越过 ✓
    #   见 `to_screen` 里那段说明 ✓）
    _sx, _sy = conv.to_screen(px[0], px[1], origin, _fs)
    _gx, _gy = conv.screen_gain((gx, gy), _fs)             # 帧像素口径 → 屏幕像素口径 ✓
    d, why = conv.counts_for(_sx, _sy, _gx, _gy)
    if d is None:
        return False, why
    if not dinput.mouse_available():
        return False, "鼠标通道不可用（本地 SendInput 模式没有硬件鼠标）"
    dx, dy = d
    _vx, _vy = conv._virtual_origin()
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
                         ("　" + _w) if (_w := conv.gain_warn(w)) else ""))
    # ⚠ 拿不到"我们要指挥的那只光标"（网络 relay ✓ / 极少数读不到 ✓）⇒ **一字退回老路** ✓
    #   （撞角 + 从虚拟原点走绝对量 ✓；老路那个"撞角那串还没发完就走位"的毛病照旧留着 ✗ ——
    #    但它**不会**把重连的重试次数吃光 ✓ 也不会把光标留在角上 ✓，比 411 那版安全 ✓）
    if not homed:
        # ⭐⭐⭐ 撞角**只在"这一拍还没归过零"时发**（用户 2026-10-17 ✓ 原话：
        #   「**地铁二号线 无法选中频道**」✗）—— 撞角那串固件要分块发完（~34 块/轴 ✗），
        #   同一拍接着走位 ⇒ **起点不是角** ⇒ 落点整体偏 ⇒ 频道格点不中 ✓
        #   （`remote_kbd/A_relay_trace.log` 实证：两条 `MOVE` 同一毫秒 ✓）。
        #   `homed=True` = 上一拍已经 `corner_only()` 归过了 ✓ ⇒ 这一拍直接走位 ✓。
        conv.corner_zero()                   # ① 到已知绝对原点
    dinput.mouse_move(dx, dy)           # ② 走到目标（x 用 gx、y 用 gy ✓）
    dinput.mouse_click("left")          # ③ 点
    # ⭐ **口径提醒**（2026-10-10 ✓）：标定那一帧的尺寸跟现在这帧差得多 ⇒
    #   把话说在回执里 ✓（它会进 `behavior.log` 的状态行 ✓）—— 别再让"点歪"静默发生 ✗。
    return True, ("撞角归零 → 走 (%d, %d) 指令单位 → 左键点击"
                  "（画面 (%.0f, %.0f) 像素 ⇒ 屏幕 (%.0f, %.0f)，比例 %.3f, %.3f）%s"
                  % (dx, dy, px[0], px[1], _sx, _sy, xr, yr,
                     ("　" + _w) if (_w := conv.gain_warn(w)) else ""))
