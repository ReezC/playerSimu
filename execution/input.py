"""键盘模拟：键名映射 + SendInput 底层发送。

游戏通常读 GetAsyncKeyState / DirectInput。SendInput 的虚拟键码对
GetAsyncKeyState 有效（大多数 ARPG）；个别用 DirectInput 读扫描码的
游戏才需要 KEYEVENTF_SCANCODE，这里先走虚拟键码，遇到不生效再换。

键名用可读字符串（"left"/"ctrl"/"f10"），UI 层只管键名，不碰 VK 码。
"""

import ctypes
import os
import threading
import time
from ctypes import wintypes

from core import perf
from execution.keys import (DIRECTION_KEYS, LOCAL_ONLY, is_local_only,
                            resolve_vk)
from execution import remote as _link
from execution.win32 import (INPUT, INPUT_KEYBOARD, KEYBDINPUT, KEYEVENTF_KEYUP,
                             user32 as _user32)


# ---- ⭐⭐ 「离屏（自检 / 冒烟）**绝不往本机桌面发键**」的闸（用户 2026-10-17 现场 ✓）----
#: 显式"我就是要真发"（少数确实要按真键盘的场合，例如 `tools/selftest_link.py`）
_ALLOW_LOCAL_ENV = "PSIMU_ALLOW_LOCAL"


def local_allowed():
    """这条进程**允不允许真的往本机桌面发键**（`SendInput` 那条路）。

    ⭐⭐ 为什么要有它（用户 2026-10-17 ✓ 原话："**为什么你跑某些进程时会动到我的方向键？**"）：
      `_link.net_allowed()` 只封"**连不连被控机**" ✓ —— 本地 `SendInput` 那条路它**不管** ✗
      ⇒ 自检跑在 `QT_QPA_PLATFORM=offscreen` 下时，`_link._remote` 是 None、`_link._blocked` 是 False
      ⇒ `key_down("left")` 真的 `SendInput` 到**操作员当前那个窗口**（IDE / 游戏 / 浏览器 ✗）。
      实测（2026-10-17 用计数器版跑 `selftest_decision`）：
        `keydown/keyup left|right|up|down` 共 **47 次**、`ctrl`/`enter`/`z` 各 1 次
      ⇒ 用户看到的就是"跑自检的时候方向键自己在动" ✓。

    判据（与 `_link.net_allowed` 同一套信号 ✓ 先显式、后推断 ✓）：
      · `PSIMU_ALLOW_LOCAL=1` ⇒ **一律允许**（真要按真键盘的场合 ✓）；
      · 否则 `QT_QPA_PLATFORM=offscreen` ⇒ **不允许**（离屏 = 自检 / 冒烟 ⇒ 发出去只会
        打到人正在用的桌面上 ✗）；
      · 其余（正常工作台 / 部署台 / 有界面的工具）⇒ 允许 ✓。

    ⚠ 只管**键盘**这条（`_send` ✓）。鼠标那条本来就只在 `_link._remote` 连着时才发
      （见 `mouse_move` ✓）⇒ 离屏下天然不会动你的鼠标 ✓。
    """
    if str(os.environ.get(_ALLOW_LOCAL_ENV, "")).strip().lower() in ("1", "true", "yes", "on"):
        return True
    return str(os.environ.get("QT_QPA_PLATFORM", "")).strip().lower() != "offscreen"


def _send(vk, up=False):
    # ⚠ 闸在最里面（所有发送路径都经过这里 ✓ 见 `key_down/key_up/tap/tap_directions`）
    if not local_allowed():
        _link._note_net_skip("local")             # 留痕：不是静默丢弃 ✓（打点 `net_skip_local`）
        return
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.union.ki = KEYBDINPUT(vk, 0, KEYEVENTF_KEYUP if up else 0, 0, 0)
    _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def key_down(name):
    if is_local_only(name):
        return      # 本机操作键（F10~F12）：永不发给游戏，见 LOCAL_ONLY
    if _link._remote is not None:
        _link._send_remote("PRESS " + _link._remote_key(name))
        return
    if _link._blocked:
        return      # 通道断了又没接上：宁可什么都不发，也别打到控制机上（见 _link._blocked）
    vk = resolve_vk(name)
    if vk is not None:
        _send(vk, False)


def key_up(name):
    if is_local_only(name):
        return      # 同上：本机操作键的松开也不发（否则游戏侧会收到一个孤儿 RELEASE）
    if _link._remote is not None:
        _link._send_remote("RELEASE " + _link._remote_key(name))
        return
    if _link._blocked:
        return
    vk = resolve_vk(name)
    if vk is not None:
        _send(vk, True)


def tap(name, duration=0.06):
    """点按一下（按下 → 松开）。给「开关自动」这类瞬发键用。"""
    if is_local_only(name):
        return      # 同上：本机操作键不点
    if _link._remote is not None:
        _link._send_remote("TAP %s %d" % (_link._remote_key(name), int(duration * 1000)))
        return
    if _link._blocked:
        return
    vk = resolve_vk(name)
    if vk is None:
        return
    _send(vk, False)
    time.sleep(duration)
    _send(vk, True)


def tap_directions(keymap, duration=0.06, only_hardware=False):
    """**把所有方向键各点按一遍**（按下 → 松开 ✓）—— 「清键 / 重置通道」前的第 ⓪ 步 ✓。

    用户 2026-10-01（手动「重置指令通道」）＋ 2026-10-11（明确要求**定时清键也要**这一步 ✓
    原话："检查定时清键 和 重置指令通道有没有 ① 把所有方向键全按一遍 ② 再清楚指令通道"✓）
    —— 两处**同一段实现** ✓（一条口径 ✓ 别各写一遍 ✗）。

    为什么非要先来这一下：重连 / RELEASEALL **只清得了固件侧**按住的键 ✗ ——
    若某一条 RELEASE 在链路里丢了，**游戏侧**会一直以为那个方向键还按着
    （角色朝一个方向一直走、按啥都不停 ✓），而那条"卡住的键"**根本不经过固件**了 ✗
    ⇒ 只有往游戏里**补一个「按下 → 松开」**，才能把游戏侧刷回"松开" ✓。
    ⇒ 顺序**必须先补按、后清通道** ✓：先补的这四下刷游戏侧 ✓、随后 RELEASEALL / 重连刷固件侧 ✓
      ⇒ 两边都不留卡键 ✓。

    ⚠ `only_hardware=True` ⇒ **本地(仅测试)后端直接跳过** ✓：那几下会打进**操作员的桌面** ✗，
      不该由**自动**动作（定时清键 / 通道自愈）去按 ✓；手动点按钮时照发 ✓（那是人明确要求的 ✓）。
    ⚠ 通道断了这几下会**静默失败** ✓（`tap` 自己不抛 ✓）⇒ 不影响后面的重连 / RELEASEALL ✓。
    ⚠ 返回值 = **真的补了几下**（给打点/用例看 ✓ 不猜 ✓）。
    """
    if only_hardware:
        try:
            if str(_link.link_health().get("backend") or "") == "local":
                return 0
        except Exception:                       # noqa: BLE001 —— 判不了就照补 ✓（宁可多按一下）
            pass
    n = 0
    for name in DIRECTION_KEYS:
        k = (keymap or {}).get(name)
        if not k:
            continue
        try:
            tap(k, duration)
            n += 1
        except Exception:                       # noqa: BLE001 —— 一下发不出去不该拖住后面 ✓
            pass
    return n


class KeyState:
    """跟踪当前按下的键，只在集合变化时发 key_up/key_down。

    移动键要「按住不放」，攻击键也要持续 —— 不能每帧都重新按下/松开，
    否则游戏里表现为断断续续。这个类记着上次按了哪些，对比本次目标集合，
    只对差异发按键。
    """

    def __init__(self):
        self._pressed = set()
        #: ⭐ **瞬发键**（输出序列 / 攻击 tap ✓ 2026-09-30）：它们不能走 `set()`
        #:   —— 决策拍是**全量重算** ✗ 会把序列正按着的键当场顶掉 ⇒ 单独登记、
        #:   `pressed()` 取**并集** ✓（流画面左下的「按键帽」据此点亮输出键 ✓）。
        self._momentary = set()

    def pressed(self):
        """当前按着的键（**只读快照** ✓）—— 给"现在按着 ↑/↓ 吗"这类**许可条件**用 ✓。
        ⭐ 2026-09-30 起含**瞬发键**（序列/攻击 ✓ 见 `_momentary` ✓）—— 流画面
        的按键帽据此点亮 ✓；`holding_vertical` 等许可判定不受影响 ✓（瞬发的攻击键
        本来就不是攀爬许可的输入 ✗ 语义不冲突 ✓）。

        用户 2026-09-27 要求："位置状态判定优化：除非按住了 ↑ 或 ↓，不能主动判定为在绳梯上"
        ⇒ `agent.holding_vertical()` 读它，再由感知层决定要不要写 `player.ladder_id` ✓。
        """
        return set(self._pressed) | set(self._momentary)

    def set(self, keys):
        """keys：本次应该按下的键名集合。"""
        keys = set(k for k in keys if k)
        for k in self._pressed - keys:
            key_up(k)
        for k in keys - self._pressed:
            key_down(k)
        self._pressed = keys

    def momentary_mark(self, name):
        """**瞬发键记账**（只记账、**不发键** ✓ —— 键已由调用方发出，再发就是双按
        ✗ 实测输出 CD 用例当场炸 ✓）。⚠ 该键已被**决策**按着（∈ `_pressed`）⇒
        不记账（决策侧本来就在 `pressed()` 里 ✓ 摘账时也不能发松开 ✓）。"""
        if name and name not in self._momentary and name not in self._pressed:
            self._momentary.add(name)

    def momentary_discard(self, name):
        """**瞬发键摘账**（只摘账、**不发松开** ✓ 同上 ✓）。"""
        self._momentary.discard(name)

    def held_by_decision(self, name):
        """这个键**这一拍正被决策侧按着**吗（∈ `_pressed`，**不含**瞬发账 ✓）。

        谁问：输出序列里那个 `up` 步（用户 2026-10-03 ✓ —— "按键挤压"的另一半）。
        为什么需要它：序列 `down` 时若这个键**已经被决策按着**，`momentary_mark` **故意
        不登记**（决策侧本来就在 `pressed()` 里 ✓）；到 `up` 那一步要是照发 `key_up`，
        松掉的其实是**决策侧**按着的那个键 ✗ —— 而决策侧 `keys.set` 因为"键集没变"
        **不会补发** PRESS ⇒ 这个键**静默丢掉**（按住的动作莫名断一下 = 现场说的挤压 ✓）。
        ⚠ 与 `_tap` 用的 `pressed()`（**并集** ✓）**不是一回事，别合并** ✗：
          点按不能碰任何"别人按着的"键（并集 ✓）；而"松开"只该让开**决策侧**的键 ——
          序列**自己**按下去的那个必须照松（不然序列自己就卡住了 ✗）。
        """
        return str(name) in self._pressed

    def release_all(self):
        self.set(set())

    def clear(self):
        """只清空按键状态、不发 key_up。配合固件 RELEASEALL 使用：
        固件侧已一次性释放所有键，本地只需同步状态，下一帧重新按需要的键。"""
        self._pressed = set()
_link._CMD_KEY.update({"f%d" % i: "F%d" % i for i in range(1, 13)})


# ---- 鼠标（远程模式：由 Pro Micro 固件作为硬件 HID 鼠标输出，同键盘一样走固件）----
#: 上一条 `mouse_move` 的时刻（算「命令间隔」用 ✓ 见 `mouse_move`）。
_last_move_t = None


def mouse_move(dx, dy):
    """相对移动鼠标：dx/dy 像素（正数向右/向下）。

    ⭐ **打点**（2026-10-03 ✓ 用户现场："看起来现象是在 A 机上一段一段一顿一顿的指令
    汇报很离散，B 机卡不卡其实没关系"）：
    鼠标这条链以前**没有任何专用计数** ✗（`MOVE` 和键盘混在 `send` 里 ✓）⇒ 「实际命令率
    多少 / 间隔匀不匀 / 每步几像素」全都量不出来、只能猜 ✓。现在补三个：

      · `mouse_send`   条数 ⇒ 命令率（触控板正常 60~125/s；远低于它 = 在攒着发 ✗）
      · `move_gap_ms`  相邻两条 MOVE 的间隔 ⇒ 中位小而 p95 大 = **一撮一撮** ✗
      · `move_px`      `|dx|+|dy|` ⇒ **中位 1~2 = 量化台阶**（`_on_pad_moved` 的 `int()`
                        + 余数累积：慢速时几个事件才凑够 1 像素 ✓）

    ⚠ 另一半在 **A 侧**（`remote_kbd/relay.py` 的 `MOVE 节拍` 那行 ✓）⇒ 两边一对比就能
      定案「是 B **发**得就不匀」还是「A **转**得不匀」✓（这正是本轮只加打点、不先改代码
      的原因 ✓）。
    """
    global _last_move_t
    if _link._remote is not None:
        _now = time.perf_counter()
        if _last_move_t is not None:
            perf.sample("move_gap_ms", (_now - _last_move_t) * 1000.0)
        _last_move_t = _now
        perf.count("mouse_send")
        perf.sample("move_px", abs(int(dx)) + abs(int(dy)))
        _link._send_remote("MOVE %d %d" % (int(dx), int(dy)))


def mouse_click(btn="left"):
    """点击鼠标：left / right / middle。"""
    if _link._remote is not None:
        _link._send_remote("CLICK " + str(btn).upper())


def mouse_press(btn="left"):
    """按住鼠标按钮不松开。"""
    if _link._remote is not None:
        _link._send_remote("PRESSM " + str(btn).upper())


def mouse_release(btn="left"):
    """松开鼠标按钮。"""
    if _link._remote is not None:
        _link._send_remote("RELEASEM " + str(btn).upper())


def mouse_scroll(n):
    """滚轮：正数向上、负数向下。"""
    if _link._remote is not None:
        _link._send_remote("SCROLL %d" % int(n))


def mouse_drag(dx, dy, btn="left", steps=8):
    """拖拽：按下 -> 分段移动 -> 松开。**交给固件一条指令原子完成**。

    **为什么不在这里发 PRESSM / MOVE×n / RELEASEM 三条**：中途串口或 TLS 断一次，
    那条 RELEASEM 就永远到不了 —— 左键会一直按着（只能拔板子，或者关掉界面让
    relay 替我们松）。合成一条发过去，最坏后果只是「这次拖拽没发生」，
    不会留下按住的键。

    steps：移动分几段（固件侧 1~40，段间隔几毫秒）。**必须分段** ——
    瞬移式的拖拽游戏看不到中间位置，不会被当成拖动。

    **发送失败补一发 RELEASEM**：`_link._send_remote` 返回 False 说明这条没发出去，
    但它也可能是「发出去了一半才断」，固件那边已经按下了。宁可多松一次
    （松一个没按的键无害），也别留一个按住的左键。

    注：这是给**程序化拖拽**（脚本 / 重连流程 / 一次性的拖到位）用的。
    触控板上那种跟着手指走的拖拽是**人机交互**，必须按住期间连续发 MOVE，
    所以那边仍然用 mouse_press / mouse_move / mouse_release 三件套。
    """
    if _link._remote is None:
        return
    n = max(1, min(40, int(steps)))
    if not _link._send_remote("DRAG %s %d %d %d"
                        % (str(btn).upper(), int(dx), int(dy), n)):
        mouse_release(btn)


def mouse_available():
    """鼠标控制是否可用：只有 ProMicro 后端（远程 / 本地串口）能输出硬件鼠标。

    本地 SendInput 模式不实现鼠标 —— 本机鼠标正用于操作界面，
    再让程序模拟鼠标会和人手打架。
    """
    return _link._remote is not None