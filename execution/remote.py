# -*- coding: utf-8 -*-
"""链路：远端 relay（TLS）/ 本地串口 的**连接、断开、重连、健康**（**不发键** ✓）。

从 `execution/input.py` 拆出来的（2026-10-17 ✓ 第 1 波第 3 刀 ✓）：
那个文件原来 766 行，一半在管"怎么把键变成指令"、另一半在管"这条通道活着没" ✗ ——
而**通道状态**有 4 个全局量（`_remote`/`_cfg`/`_blocked`/`_connect_err`）、
还被 14 个文件从外面摸（`dinput._remote = 假的` 之类）⇒ 混在一起时，
"发键"和"换链路"互相看不见边界 ✗。

⚠ **方向**：`input`（发键/鼠标） → `remote`（本模块）✓，反方向不存在 ✓
  —— 本模块里 `reconnect_remote`/`use_*` **一个键都不按**（只开/关 socket 或串口 ✓），
  按真键是 `input` 的事 ✓。
⚠ **状态只有一个真值**（就在本模块 ✓）：谁要判断"远端连着没"，就**当场**读
  `remote._remote`（不许 `from ... import _remote` ✗ 那是快照，重连后就过期了 ✗）。
"""
import ctypes
import os
import threading
import time
from ctypes import wintypes

from core import perf




# ---- 远程后端（agent 跑在控制机，通过 TLS 把按键发到游戏机的 Pro Micro）----
# 默认 _remote=None 走本地 SendInput；调用 use_network() 后切到远程硬件键盘。
_remote = None


# 连接参数记下来，「重连通道」要用（见 reconnect_remote）
_cfg = None


# 通道断了又接不上时置 True：这时**绝不能**退化成本机 SendInput ——
# 控制机正是人在用的机器，把游戏按键打到本机上是会出事的（比如往别的窗口里打字）。
# 断开期间宁可什么都不发，等重连成功再恢复。
_blocked = False



#: 最近一次**连后端失败**的原因（人话 ✓；空串 = 没失败过 ✓）—— 给界面 / `behavior.log` 用。
#: ⚠⚠ 为什么必须留（2026-10-10 ✓ 用户报「**本地ProMicro连接失败**」✗ 而界面只有
#:   「Pro Micro 连接失败，已回退本地」八股 ✓）：`connect_async` 把异常**整个吞掉**了 ✗
#:   （`except Exception: status = "fail"` ✓）⇒ **为什么失败**在哪儿都查不到 ✓ 人只能猜 ✓
#:   —— 现场原因是 `PermissionError(13, 拒绝访问)` = **串口被另一个工作台占着** ✓
#:   （一个串口只能被一个进程独占 ✓ 而当时有**两个工作台**在跑 ✓）。
_connect_err = ""



# 换后端（连 / 断 / 重连）**必须串行**：这几件事都是「先关旧的、再建新的」。
# 两个线程同时做就会出现「一边正在 TLS 握手，另一边把那个 socket 关掉」——
# 表现是 relay 侧刷「客户端在 TLS 握手阶段断开」（实测踩过：手动的「重置指令通道」
# 和界面上新加的自动重连撞在一起）。用 RLock：reconnect_remote 内部要调
# use_network（同一线程重入），普通 Lock 会把自己锁死。
_link_rlock = threading.RLock()



# ---- ⭐⭐ 「离屏（自检 / 冒烟）**绝不连被控机**」的唯一闸（用户 2026-10-05 ✓ 治根）----
#: 显式"我就是要连"（专门的链路自检 `tools/selftest_link.py`、无头部署那种场合 ✓）
_ALLOW_NET_ENV = "PSIMU_ALLOW_NET"


#: 闸关着时的留痕标记（同一 kind 只记一次，别刷爆日志 ✗ 见 `_note_net_skip`）
_net_skip_kind = None




def net_allowed():
    """这条进程**允不允许**去连被控机（远端 relay / 串口硬件键盘）。

    ⭐⭐ 为什么要有它（用户 2026-10-05 ✓ 原话："**从源头彻底封死**"）：
      病根是**打桩打在了用例里、而且只打了一个套件** ✗ —— 原来只有
      `tools/selftest_main_window._win()` 在 monkeypatch 面板方法 ✓，而
      `gui_smoke` / `selftest_minimap` / `selftest_screen_state` / `selftest_decision`
      里那几个 `PlayerPanel()` **都没打** ✗ ⇒ 只要项目配置是 `input_device: remote`
      （`config/decision.json` 与好几个项目就是 ✓）它们照样起那条**后台连接线程** ✗
      ⇒ 线程在**几秒后**才超时返回，而那时用例早跑完、Qt 对象已销毁 ⇒
      **偶发原生崩 `0xC0000005`** ✗（时机性 ⇒ "多跑几次"不算验证 ✗）。

    判据（先显式、后推断 ✓；**每次都读环境**，不缓存 ✓ 用例要能当场摆判据）：
      · `PSIMU_ALLOW_NET=1` ⇒ **一律允许**（`tools/selftest_link.py` 这种"就是要连"的 ✓）；
      · 否则 `QT_QPA_PLATFORM=offscreen` ⇒ **不允许**：离屏 = 没人看得见画面，此时去连被控机
        **只可能**是自检 / 冒烟在跑（~20 个套件 + `gui_smoke` 都用它 ✓）；
      · 其余（正常工作台 / **部署台**，真平台）⇒ 允许 ✓ —— `deploy/` 侧不设 offscreen（已核 ✓）。

    ⚠ 只管一件事：**连不连被控机**。本地 SendInput 那条路不归它管 ✓。
    """
    if str(os.environ.get(_ALLOW_NET_ENV, "")).strip().lower() in ("1", "true", "yes", "on"):
        return True
    return str(os.environ.get("QT_QPA_PLATFORM", "")).strip().lower() != "offscreen"




def _note_net_skip(kind):
    """留痕：闸关着、**没去连**（不许静默 ✗）—— 同一 kind 只记一次 ✓。"""
    global _net_skip_kind
    if _net_skip_kind != kind:
        _net_skip_kind = kind
        try:
            perf.count("net_skip_" + str(kind or "?"))
        except Exception:                       # noqa: BLE001 —— 打点坏了别影响行为 ✗
            pass




def use_blocked(why=""):
    """切到「**什么都不发**」的后端（`_blocked` ✓）：连不上被控机时**绝不能**退化成把按键
    打到控制机上（控制机正是人在用的机器 ✗ —— 老口径见 `_blocked` 的说明 ✓）。

    ⭐ 新用途（用户 2026-10-05 ✓）：闸关着（离屏自检 / 显式关闸 ✓）时走这儿 ——
      **不连、也不发**（比"退化成本地"安全 ✓ 也比"静默"诚实 ✓）。
    """
    global _blocked
    with _link_rlock:
        _drop_remote()
        _blocked = True
        _note_net_skip(why or "blocked")




def last_connect_err():
    """最近一次**连后端失败**的原因（人话 ✓；空串 = 没失败过 ✓）—— 见 `_connect_err` ✓。"""
    return _connect_err




def cursor_probe():
    """**现在光标在哪**（＋ 那台机器的屏幕信息）⇒ `(x, y, vx, vy, vw, vh, cap)`；问不到 ⇒ None ✓。

    · **本地后端**（本机 SendInput / 串口 ProMicro ✓）⇒ 本机 `GetCursorPos` ✓（`cap = None` ✓）；
    · **网络 relay**（鼠标动在**被控机** ✓）⇒ 问 A 机（`KbdClient.cursor` ✓ 见 relay 那边 ✓）。

    ⚠ 这一条就是"**能不能自己量鼠标标定**"的闸 ✓（`execution/mouse_aim.auto_measure` ✓）：
      问不到 ⇒ 只能请人**手工量一次** ✓（当帧尺寸变了的时候 ✓ 见 `gain_mismatch_block` ✓）。
    ⚠ 为什么非要能"看见光标"：gain 是「帧像素 / 指令单位」✓ ⇒ 只能**发已知单位、看走了多少像素**
      量出来 ✓ ⇒ 看不见光标 ⇒ 量不出来 ✓（这是用户 2026-10-10："**不能自动吗？**" 的那道坎 ✓）。
    """
    if _remote is not None and hasattr(_remote, "cursor"):
        # 网络后端（`KbdClient` ✓）：**问 A 机** ✓ 见 `remote_kbd/relay.py` 的 `CURSOR_QUERY` ✓
        try:
            return _remote.cursor()
        except Exception:                        # noqa: BLE001 —— 问不到就当我们量不了 ✓
            return None
    try:
        import ctypes
        from ctypes import wintypes
        u = ctypes.windll.user32
        pt = wintypes.POINT()
        if not u.GetCursorPos(ctypes.byref(pt)):
            return None
        return (int(pt.x), int(pt.y),
                int(u.GetSystemMetrics(76)), int(u.GetSystemMetrics(77)),
                int(u.GetSystemMetrics(78)), int(u.GetSystemMetrics(79)), None)
    except Exception:                            # noqa: BLE001
        return None




def window_probe():
    """**被控机上游戏窗口在哪** ⇒ `(x, y, w, h)`（客户区 / 屏幕坐标）；问不到 ⇒ `None` ✓。

    ⭐⭐⭐ 为什么要有它（用户 2026-10-17 ✓ 原话：
      「**A 机游戏窗口是会移动的，你保证你的方案能适配吗？**」✓）：
      · 断线重连 / 测谎的点击比例是**相对画面**算的 ✓，而画面 = A 机抓的那块屏幕 ✓；
      · 要落成**屏幕坐标**就必须知道"那块在屏幕的哪儿" ✓ —— 老实现把它当 `(0, 0)` ✗
        ⇒ **窗口一挪就全偏** ✓（用户说的"会移动"正是这条 ✗）；
      · 现在每次点之前问一次 A 机（relay 的 `WINDOW?` ✓ 读 `deploy.json` 的
        `push.region_win` ✓ 跟随窗口时就是**窗口客户区** ✓）⇒ 挪了也自动跟上 ✓
        **不用改任何配置** ✓。
    ⚠ 本地后端（本机 SendInput / 串口）⇒ 返回 `None` ✓（那条路的 `origin` 另有来源 ✓
      见 `gui/live_thread` 的 `origin_ref` ✓）；问不到也 `None` ⇒ 调用方退回老路 ✓ 不会更糟 ✓。
    """
    if _remote is not None and hasattr(_remote, "window"):
        try:
            return _remote.window()
        except Exception:                        # noqa: BLE001 —— 问不到就当不知道 ✓
            return None
    return None




def _note_connect_fail(kind, why):
    """留痕：连不上时**把原因记下来**（日志 + 打点 ✓ 不许静默 ✗ 见 `_connect_err` ✓）。"""
    try:
        # ⚠ 是 `core.behavior` ✗（原来写成裸 `import behavior` ⇒ 这里**每次都抛** ✓
        #   被下面那句 except 吃掉 ⇒ **这条日志一直没写出去** ✓ 2026-10-10 顺手修 ✓）。
        from core import behavior
        behavior.event("connect_fail", kind=str(kind or "?"), why=str(why)[:160])
    except Exception:                           # noqa: BLE001 —— 打点坏了别影响连接 ✗
        pass
    try:
        perf.count("connect_fail_" + str(kind or "?"))
    except Exception:                           # noqa: BLE001
        pass




def connect_async(kind, on_done=None, **kw):
    """按 `kind` 起一条**后台连接线程**（GUI 用，`PlayerPanel._apply_input_device` 调它）。

    ⚠⭐ **全仓库唯一**去连被控机的地方（原来那两处 `threading.Thread(...)` 在面板里 ✗）——
      用户 2026-10-05 要求"从源头彻底封死"：**线程的出生地和闸放同一处** ✓ ⇒ 谁调都绕不过去 ✓。

    闸关着（离屏自检 / `PSIMU_ALLOW_NET` 没开 ✓ 见 `net_allowed`）⇒ **连线程都不起** ✓
    （**不是**"起一条马上返回"✗ —— 那样它仍可能在解释器收尾时活着 ✗ 正是要治的病 ✓）
    ＋ 转 `use_blocked`（什么都不发 ✓）。

    返回 `True` = 起了线程；`False` = **没起**（`on_done` 也不会被叫 ✓ 调用方自己给提示 ✓）。

    `on_done(status)` 在**后台线程**里被叫：`"ok"` / `"fail"` —— 回调里碰 Qt 对象必须自己兜
    `RuntimeError`（窗口可能已经销毁 ✓ 见 `PlayerPanel._on_device_connected` ✓）。
    """
    if not net_allowed():
        use_blocked(str(kind))
        return False

    def _run():
        global _connect_err
        status = "ok"
        try:
            if kind == "serial":
                use_serial(kw.get("port"))
            else:
                use_network(kw.get("host"), kw.get("port"), kw.get("cert"))
        except Exception as e:                  # noqa: BLE001 —— 连不上是常态（对面没开机 ✓）
            status = "fail"
            # ⚠⚠ **不许静默**（2026-10-10 ✓ 用户现场）：原来这里把异常整个吞掉 ✗ ⇒ 界面只剩
            #   "Pro Micro 连接失败" ✗、`behavior.log` 里也没线索 ⇒ 只能靠人猜 ✓
            #   （实际原因是**串口被另一个工作台占着** ✓ 见 `_connect_err` ✓）。
            if not _connect_err:
                _connect_err = "%s: %s" % (type(e).__name__, e)
            _note_connect_fail(kind, _connect_err)
        if on_done is not None:
            on_done(status)

    threading.Thread(target=_run, daemon=True,
                     name="input-connect-%s" % kind).start()
    return True




_CMD_KEY = {
    "left": "LEFT", "right": "RIGHT", "up": "UP", "down": "DOWN",
    "ctrl": "CTRL", "alt": "ALT", "shift": "SHIFT",
    "enter": "ENTER", "space": "SPACE", "esc": "ESC", "tab": "TAB",
    "backspace": "BACKSPACE",
    "del": "DEL", "insert": "INSERT", "home": "HOME", "end": "END",
    "pageup": "PGUP", "pagedown": "PGDN",
    "grave": "GRAVE",
}




def _remote_key(name):
    """input.py 键名 → 固件键名。"""
    n = str(name).lower()
    return _CMD_KEY.get(n, n)




def _send_remote(line):
    """发一条远程指令。返回 True/False（False 或不返回都表示这条没发出去）。

    以前这里 `except: pass` 把失败全吞了 —— 通道死了上层一点感觉都没有，
    还在不停发。表现就是「停自动没用、手动也没用，只能关 GUI」。
    """
    if _remote is None:
        return False
    _t = time.perf_counter()
    try:
        ok = bool(_remote.send(line))
    except Exception:
        ok = False
    # 打点：一次发键在通道上花了多久（远程是 socket / 串口写，链路卡住时这里最先变大）
    perf.ms("send_ms", _t)
    if ok:
        perf.count("send")
    else:
        perf.count("send_fail")
        # 失败按指令名分开计（fail_PRESS / fail_RELEASEALL / fail_TAP…）：
        # 「所有指令都在丢」和「某一条特定指令在丢」是完全不同的问题
        # —— 后者往往是固件正卡在某个 delay 里没读串口。
        head = (str(line).split() or ["?"])[0][:10]
        perf.count("fail_" + head)
    return ok




# 发过指令后，超过这么久收不到固件回执就判定链路卡死（秒）。
# 固件对每条指令都会回 DONE/ERR，所以「静默」是可靠的死链信号。
REPLY_SILENCE_LIMIT = 5.0




def link_health():
    """指令通道当前状态：{"backend", "ok", "err", "fails", "silent"}。

    backend: "local"（本机 SendInput）/ "remote"（网络 Pro Micro）/
             "serial"（直连）/ "blocked"（断了又接不上，已停止发送）。

    ok 同时对「发送有没有报错」和「固件还有没有回执」两件事下判断 ——
    后者能抓到 relay 卡在串口写上这种「发送成功、实际什么都没发生」的情况。
    """
    if _remote is None:
        # ⚠ 连不上时 `_remote` 就是 None ✗ ⇒ 这里**必须**把 `_connect_err` 带出去 ✓
        #   （不然界面 / 日志只有"连接失败"四个字 ✓ 见 `_connect_err` 的说明 ✓）。
        return {"backend": "blocked" if _blocked else "local",
                "ok": not _blocked, "err": _connect_err, "fails": 0, "silent": 0.0}
    kind = _cfg[0] if _cfg else "remote"
    send_ok = bool(getattr(_remote, "ok", True))
    # ⭐⭐ **同机多实例：relay 只让第一个连上的人控制**（用户 2026-10-17 ✓「多实例」）
    #   ⇒ 后连上的那个是**只读**（能问光标/窗口 ✓ 发指令一律 `ERR busy` ✓）。
    #   ⚠ 光看 `ERR` 分不清"固件不认这条"还是"控制位被别人占了" ✗
    #     ⇒ relay 握手后先发 `ROLE readonly` ✓（`kbd_client.role` ✓）这里如实说 ✓
    #     ⇒ 上层（agent 自检 / 状态栏）能立刻看到真原因 ✓ 而不是"卡住不动"✓。
    _role = str(getattr(_remote, "role", "") or "")
    _readonly = (_role == "readonly")
    if _readonly:
        send_ok = False               # ⚠ 只读 ⇒ 一律当"发不出去"（上层别继续对着它发 ✓）
    silent = 0.0
    try:
        silent = float(_remote.silent_for())
    except Exception:
        pass
    quiet_ok = silent < REPLY_SILENCE_LIMIT
    err = str(getattr(_remote, "last_err", ""))
    if not quiet_ok and not err:
        err = "发指令后 %.0f 秒没收到固件回执（链路卡死）" % silent
    if _readonly:
        # ⚠ 这句**必须写在这之后**（✗ 否则会被上面那句 `err = last_err` 冲掉 ✓）
        err = ("控制位被另一个工作台实例占用（本实例只读）——"
               "同一台机器只能有一个实例控它：把其中一个实例的输入设备改成「本机」，"
               "或让它连另一台机器")
    return {"backend": kind,
            "ok": send_ok and quiet_ok,
            "err": err,
            "fails": int(getattr(_remote, "fails", 0)),
            "silent": silent,
            #: `control` / `readonly` / `""`（老 relay 不说 ✓ 当控制端用 ✓）
            "role": _role}




def reconnect_remote():
    """丢掉当前连接重开。返回 True = 重开后通道可用。

    **为什么重连能解开卡键**：relay 在客户端断开时会往串口直接写一条 RELEASEALL
    （`remote_kbd/relay.py` 的 bridge() 里的 finally）。也就是说「断开」这个动作
    本身就是一次硬复位 —— 而且它**不依赖我们这边的 TCP 通道是否还通**。
    这正是上次「关掉 GUI」能让卡住的键松开的原因：那一下是 relay 替我们松的。

    重连失败**不会**退化成本机 SendInput（那会把按键打到控制机上），而是进入
    _blocked：什么都不发，等下次重连成功。

    ⚠ 闸关着（离屏自检 / `PSIMU_ALLOW_NET` 没开 ✓ 见 `net_allowed`）⇒ **不重连**、
      直接转"什么都不发"并返回 `False`（"通道不可用"是**实话** ✓ 别报 True 骗上层 ✗）。
    """
    global _remote, _blocked
    if not net_allowed():
        use_blocked("retry")
        return False
    with _link_rlock:               # 与其它换后端的调用串行（见 _link_rlock 说明）
        if _remote is None or _cfg is None:
            _blocked = False
            return True
        kind = _cfg[0]
        try:
            if kind == "remote":
                use_network(_cfg[1], _cfg[2], _cfg[3])
            else:
                use_serial(_cfg[1])
            return True
        except Exception:
            _drop_remote()
            _blocked = True
            return False




def release_all_remote():
    """发 RELEASEALL 到固件，一次性清空固件侧所有按住的键（防长时间运行卡键）。"""
    if _remote is not None:
        _send_remote("RELEASEALL")




def _drop_remote():
    """关闭当前远程/串口后端（socket / 串口不泄漏），切到别的后端前调用。"""
    global _remote
    if _remote is not None:
        try:
            _remote.close()
        except Exception:
            pass
    _remote = None




def use_network(host, port, cafile):
    """切到网络后端（agent 在控制机时，启动阶段调用一次）。

    拿锁期间会做完整的 TLS 握手（可能要几秒）—— 这是故意的：握手和「关旧连接」
    必须互斥，否则会互相打断（见 _link_rlock 的说明）。

    ⚠⭐ 闸关着（离屏自检 / `PSIMU_ALLOW_NET` 没开 ✓ 见 `net_allowed`）⇒ **一个 socket
      都不建**、直接转"什么都不发"（`use_blocked` ✓）—— 这条是**兜底**：正常 GUI 走
      `connect_async`（它自己先问闸 ✓，连线程都不起 ✓）。
    """
    global _remote, _cfg, _blocked, _connect_err
    if not net_allowed():
        use_blocked("remote")
        _connect_err = "闸关着（离屏自检 / `PSIMU_ALLOW_NET` 没开）⇒ 没去连被控机"
        return
    with _link_rlock:
        _drop_remote()
        from remote_kbd.kbd_client import KbdClient
        _remote = KbdClient(host, port, cafile)
        _cfg = ("remote", host, port, cafile)   # 记下来给 reconnect_remote 用
        _blocked = False
        _connect_err = ""                       # 连上了 ⇒ 清掉上次的失败原因 ✓




def _serial_fail_text(port, exc, found=True):
    """把「串口打不开」翻译成**能照着做的人话**（见 `_connect_err` 的说明 ✓）。

    ⚠ 为什么要分这么细（2026-10-10 ✓ 用户现场「本地ProMicro连接失败」✗）：
      **设备在不在**和**能不能打开**是两件事 ✗ —— 当时设备好好的（`COM8 Arduino Leonardo`
      ✓ 自动找口也找得到 ✓），可 `serial.Serial(...)` 抛
      `PermissionError(13, 拒绝访问)` ✗ = **串口被别的进程占着** ✓
      （Windows 串口是**独占**的 ✓ —— 当时有**两个工作台**在跑 ✓）。
      这两种病要给**完全相反**的建议（一个"拔插 USB"、一个"关掉另一个工作台" ✓）
      ⇒ 混成一句"连不上"等于没说 ✓。
    """
    s = ("%s: %s" % (type(exc).__name__, exc)) if exc is not None else ""
    low = str(exc).lower() if exc is not None else ""
    if found is None:
        return ("找不到 Pro Micro 串口（USB 插着吗？也可以看看 `config/link.yaml` 的 "
                "`serial_local` ✓ 换 USB 口后串口号会变、这儿会自动找 ✓）")
    busy = (isinstance(exc, PermissionError) or "拒绝访问" in s or "access is denied" in low
            or "permissionerror" in low or "busy" in low or "占用" in s)
    if busy:
        # ⚠ 只说**这一种病**该怎么做 ✓ —— 别把"拔插 USB"混进来 ✗：那是"插都没插"那一种的
        #   建议 ✓ 两种病建议**相反** ✓（混着说是把人往反方向推 ✓ 见上面那段说明 ✓）。
        return ("串口 %s 被**别的进程占用**了（%s）⇒ 一个串口同时只能被一个进程打开 ✓："
                "**多半是另一个工作台 / relay 还开着** ✓ 关掉多余的那个、再点连接 ✓"
                % (port, s))
    return ("串口 %s 打不开（%s）⇒ 拔插一次 USB、或换个 USB 口试试 ✓" % (port, s))




def use_serial(port):
    """切到本地直连 Pro Micro（USB CDC 串口），无需 relay。

    port 为空或打开失败时，自动扫描 Arduino/SparkFun 串口（换 USB 口不用改配置）。

    ⚠⭐ 闸关着（离屏自检 / `PSIMU_ALLOW_NET` 没开 ✓ 见 `net_allowed`）⇒ **串口也不开**、
      直接转"什么都不发"（`use_blocked` ✓）—— 同 `use_network`：谁直接调都绕不过去 ✓。

    ⚠⚠ 连不上时**抛出的那句话就是给人看的**（见 `_serial_fail_text` ✓）—— 别把它简化成
      "连接失败" ✗（那正是 2026-10-10 现场"查不出原因"的由来 ✓）。
    """
    global _remote, _cfg, _blocked, _connect_err
    if not net_allowed():
        use_blocked("serial")
        _connect_err = "闸关着（离屏自检 / `PSIMU_ALLOW_NET` 没开）⇒ 没去连串口"
        return
    with _link_rlock:
        _drop_remote()
        from remote_kbd.serial_kbd import SerialKbd, find_pro_micro_port
        if port:
            try:
                _remote = SerialKbd(port)
                _cfg = ("serial", port)
                _blocked = False
                _connect_err = ""
                return
            except Exception as e:              # noqa: BLE001 —— 记下**第一手**原因 ✓（下面要说清 ✓）
                _first = e
        else:
            _first = None
        found = find_pro_micro_port()
        if found is None:
            _connect_err = _serial_fail_text(port, _first, found=None)
            raise RuntimeError(_connect_err)
        try:
            _remote = SerialKbd(found)
        except Exception as e:                  # noqa: BLE001 —— 找到了却打不开（多半是被占用 ✓）
            _connect_err = _serial_fail_text(found, e)
            raise RuntimeError(_connect_err)
        _cfg = ("serial", found)
        _blocked = False
        _connect_err = ""




def use_local():
    """切回本地 SendInput，并关闭远程连接（socket / 串口不泄漏）。

    ⚠⚠ **不清 `_connect_err`** ✗（2026-10-10 ✓ 特意留的）：连不上时 `PlayerPanel` 会调
      它**回退本地** ✓，接着就要把"为什么没连上"显示出来 ✓ ⇒ 这里清掉的话
      界面又只剩"连接失败"四个字 ✗（那正是要治的病 ✓ 见 `_connect_err` ✓）。
      真正该清的地方是**连接成功**那两条路（`use_serial` / `use_network` ✓）。
    """
    global _blocked
    with _link_rlock:
        _drop_remote()
        _blocked = False




def shutdown():
    """关闭远程键盘连接（如果存在），停止指令传输。GUI 关闭时调用。"""
    use_local()
