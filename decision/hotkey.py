"""系统级全局热键（RegisterHotKey + WM_HOTKEY），**外加一条键盘钩子后路**。

让「开启自动」的 F11 在**任何窗口聚焦时**都能触发 —— 普通 QShortcut 只在
工作台 GUI 有焦点时有效，而自动打怪时焦点在游戏窗口（MapleNecrocer 的
Playmode），必须用系统级热键 RegisterHotKey 才能跨窗口捕获。

⚠⚠ **两条路的区别（2026-10-10 ✓ 用户："我需要 F11 全局生效"）**：
  · `RegisterHotKey` 是**独占**的 ✗ —— 同一个键**只允许一个进程**注册 ✓
    现场就是 **F11 已被别的进程占着**（探针实测：`RegisterHotKey(…, VK_F11)` 返回 0 ✓
    `F12` 也占着 ✓ 只有 `F10` 能注册 ✓）⇒ 本进程只能降级成"本窗口聚焦时"✗
    ⇒ 用户把焦点切到游戏后按 F11 **一点反应都没有** ✓（他要的正是这个 ✗）；
  · `WH_KEYBOARD_LL`（低级键盘钩子）是**观察型** ✓ —— 谁都能装、互相不冲突 ✓
    ⇒ 就算别人占着 F11，我们照样收得到 ✓。
⇒ 首选仍是 `RegisterHotKey` ✓（系统级、最省事 ✓、且它**不注入**任何东西 ✓），
  **它失败才装钩子** ✓（最小改动 ✓ 老行为一字不变 ✓）。

⚠ 钩子的两条纪律（都写在 `install_hook` 的说明里 ✓）：
  · 回调对象**必须留在模块级** ✗（Python 侧一回收 ⇒ 原生回调变野指针 ⇒ 崩 ✓）；
  · 回调里**只许做极轻的事** ✗（它在系统钩子链里 ⇒ 慢了会拖全系统的按键 ✓）
    ⇒ 真正的动作由调用方丢回事件循环（`QTimer.singleShot(0, …)` ✓）。
"""

import ctypes
from ctypes import wintypes

MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312

VK_F11 = 0x7A

# ---- 低级键盘钩子（后路 ✓ 不独占 ✓）----
WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100
WM_SYSKEYDOWN = 0x0104


class MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("message", wintypes.UINT),
        ("wParam", wintypes.WPARAM),
        ("lParam", wintypes.LPARAM),
        ("time", wintypes.DWORD),
        ("pt", ctypes.c_long * 2),
    ]


class KBDLLHOOKSTRUCT(ctypes.Structure):
    """钩子回调拿到的按键信息（`WH_KEYBOARD_LL` ✓）—— 我们只读 `vkCode` ✓。"""
    _fields_ = [
        ("vkCode", wintypes.DWORD),
        ("scanCode", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


_user32 = ctypes.windll.user32

_LRESULT = ctypes.c_ssize_t                      # 64 位下必须是 LONG_PTR ✗（写 c_long 会截断 ✓）
_HOOKPROC = ctypes.WINFUNCTYPE(_LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
try:
    _user32.SetWindowsHookExW.restype = ctypes.c_void_p
    _user32.SetWindowsHookExW.argtypes = [ctypes.c_int, _HOOKPROC, ctypes.c_void_p,
                                          wintypes.DWORD]
    _user32.CallNextHookEx.restype = _LRESULT
    _user32.CallNextHookEx.argtypes = [ctypes.c_void_p, ctypes.c_int, wintypes.WPARAM,
                                       wintypes.LPARAM]
    _user32.UnhookWindowsHookEx.restype = wintypes.BOOL
    _user32.UnhookWindowsHookEx.argtypes = [ctypes.c_void_p]
except Exception:                                # noqa: BLE001 —— 非 Windows / 极老系统 ✓
    pass

#: ⚠⚠ 钩子的**全部状态都留在模块级** ✗ —— 尤其是 `_hook_proc`（那个 ctypes 回调对象 ✓）：
#:   它一被回收，系统再调进来就是**野指针** ⇒ 进程直接崩 ✓（而且崩在系统调用里、栈很难看 ✓）。
_hook_proc = None
_hook_handle = None
_hook_vk = None
_hook_cb = None


def register(hwnd, hotkey_id, vk, modifiers=MOD_NOREPEAT):
    """注册全局热键。成功返回 True，失败（**被别的进程占用** ✓）返回 False。"""
    try:
        return bool(_user32.RegisterHotKey(int(hwnd), int(hotkey_id),
                                           int(modifiers), int(vk)))
    except Exception:
        return False


def unregister(hwnd, hotkey_id):
    try:
        _user32.UnregisterHotKey(int(hwnd), int(hotkey_id))
    except Exception:
        pass


def is_hotkey(message_ptr, hotkey_id):
    """nativeEvent 的 message 指针 → 是不是指定的 WM_HOTKEY。"""
    try:
        msg = MSG.from_address(int(message_ptr))
        return msg.message == WM_HOTKEY and msg.wParam == int(hotkey_id)
    except Exception:
        return False


def hook_installed():
    """现在挂着键盘钩子吗（诊断 / 用例用 ✓）。"""
    return _hook_handle is not None


def install_hook(vk, on_fire):
    """装**全局键盘钩子**：`vk` 被按下时叫 `on_fire()`。返回 `(ok, why)` ✓。

    ⚠ 它**不独占** ✓（这正是要有它的原因 ✓ 见模块头那段 ✓）——别的程序（或另一个
      工作台实例）占着同一个键，我们**照样收得到** ✓。
    ⚠⚠ **一定不吞键** ✗：回调末尾无条件 `CallNextHookEx` ✓ ⇒ 那个键该给谁还给谁 ✓
      （游戏里 F11 原本干什么，照旧干什么 ✓ —— 我们只是"顺便知道了"✓）。
    ⚠ 重复调用会**先卸掉旧的** ✓（幂等 ✓ 别叠着挂一串 ✓）。
    ⚠ `on_fire` 在**系统钩子链里**被叫 ✗ ⇒ 里面**只许做极轻的事** ✓
      （调用方用 `QTimer.singleShot(0, …)` 丢回事件循环 ✓ 见 `main_window` ✓）。
    """
    global _hook_proc, _hook_handle, _hook_vk, _hook_cb
    remove_hook()
    _hook_vk = int(vk)
    _hook_cb = on_fire

    def _proc(ncode, wparam, lparam):
        try:
            if int(ncode) == 0 and int(wparam) in (WM_KEYDOWN, WM_SYSKEYDOWN):
                _kb = ctypes.cast(int(lparam), ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
                if int(_kb.vkCode) == _hook_vk and _hook_cb is not None:
                    _hook_cb()
        except Exception:                        # noqa: BLE001 —— 回调里**绝不许**抛 ✓（吞掉 ✓）
            pass
        # ⚠⚠ **必须原样传下去** ✗（不吞键 ✓ 见上面说明 ✓）
        return _user32.CallNextHookEx(None, int(ncode), int(wparam), int(lparam))

    # ⚠⚠ **多实例要提前打招呼**（用户 2026-10-17 ✓「多实例」）：
    #   `RegisterHotKey` 是**独占**的 ✓（只有一个实例拿得到 ✓），可钩子这条路**不独占** ✗
    #   ⇒ 同机两个工作台都走钩子时，按一下 F11 **两边一起切** ✓
    #   （一个控远程、一个控本机 ⇒ 两个客户端同时开始/停止自动 ✓ 这不是"看着乱"✗）。
    #   ⇒ 这里只做**如实记账**（不擅自关掉功能 ✗：多数人只开一个 ✓ 关掉反而是坏的 ✓）；
    #     要彻底避开 ⇒ 给两个实例配**不同的热键**（热键是项目级 ✓ 两个项目各配一个 ✓）。
    try:
        from core import instance as _inst
        if _inst.slug():
            from core import behavior as _bh
            _bh.event("hotkey", "多实例：全局热键不独占", ok=True,
                      inst=_inst.label(),
                      why="两个实例都会收到同一个键；请给两边配不同热键")
    except Exception:                        # noqa: BLE001 —— 记账不许影响装钩子 ✓
        pass

    _hook_proc = _HOOKPROC(_proc)                # ⚠ 留模块级 ✓（回收 = 崩 ✓）
    h = _user32.SetWindowsHookExW(WH_KEYBOARD_LL, _hook_proc, None, 0)
    if not h:
        _hook_proc, _hook_vk, _hook_cb = None, None, None
        return False, ("装键盘钩子失败（`SetWindowsHookExW` 返回 0 ✓ 常见于"
                       "被安全软件拦 / 装它的线程没有消息循环 ✓）")
    _hook_handle = h
    return True, ""


def _pid_elevated(pid):
    """那个进程是不是**提权**（管理员）跑的 ⇒ `True` / `False` / `None`（问不到 ✓）。"""
    try:
        import ctypes
        from ctypes import wintypes
        k32, adv = ctypes.windll.kernel32, ctypes.windll.advapi32
        h = k32.OpenProcess(0x1000, False, int(pid))       # QUERY_LIMITED_INFORMATION
        if not h:
            return None
        try:
            tok = wintypes.HANDLE()
            if not adv.OpenProcessToken(h, 0x0008, ctypes.byref(tok)):   # TOKEN_QUERY
                return None
            try:
                val, n = wintypes.DWORD(), wintypes.DWORD()
                if not adv.GetTokenInformation(tok, 20, ctypes.byref(val),
                                               ctypes.sizeof(val), ctypes.byref(n)):
                    return None                                 # 20 = TokenElevation
                return bool(val.value)
            finally:
                k32.CloseHandle(tok)
        finally:
            k32.CloseHandle(h)
    except Exception:                                    # noqa: BLE001
        return None


def self_elevated():
    """**我们自己**是不是提权跑的 ✓（`None` = 问不到 ✓）。"""
    try:
        import ctypes
        return _pid_elevated(ctypes.windll.kernel32.GetCurrentProcessId())
    except Exception:                                    # noqa: BLE001
        return None


def elevated_visible_titles(limit=3):
    """现在**看得见**的顶层窗口里，属于**提权**进程的那几个标题 ✓（取不到就跳过 ✓）。

    ⚠ 只看**看得见**的（游戏/播放器等 ✓）—— 系统里本来就有几个常驻的提权窗口，
      全列出来会把提示冲淡 ✓。
    """
    out = []
    try:
        import ctypes
        from ctypes import wintypes
        u = ctypes.windll.user32
        titles = []

        @ctypes.WINFUNCTYPE(ctypes.c_int, wintypes.HWND, wintypes.LPARAM)
        def _cb(hwnd, _lp):
            try:
                if not u.IsWindowVisible(hwnd):
                    return 1
                n = u.GetWindowTextLengthW(hwnd)
                if n <= 0:
                    return 1
                buf = ctypes.create_unicode_buffer(n + 1)
                u.GetWindowTextW(hwnd, buf, n + 1)
                pid = wintypes.DWORD()
                u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                titles.append((pid.value, buf.value))
            except Exception:                            # noqa: BLE001
                pass
            return 1

        u.EnumWindows(_cb, 0)
        for pid, t in titles:
            if pid and _pid_elevated(pid) is True:
                out.append(t.strip()[:40])
            if len(out) >= limit:
                break
    except Exception:                                    # noqa: BLE001
        return []
    return out


def elevation_hint():
    """⇒ 一句**给人看**的话（不用提醒 ⇒ `""` ✓）：**别人提权、我们没提** ⇒ 热键收不到 ✓。

    ⚠⚠⚠ 为什么必须有它（2026-10-11 ✓ 用户："**本地实时聚焦游戏窗口时按 F11 无法暂停自动**"✓）：
      · **低级键盘钩子 / `RegisterHotKey` 都收不到"发给更高权限窗口"的按键** ✗（UIPI ✓）；
      · 现场实测：`Maplestory_Classic.exe` **提权 = True** ✓，而工作台 `pythonw.exe`
        **提权 = False** ✗ ⇒ **游戏一聚焦，我们什么都收不到** ✓
        （⚠ 这正是"工作台自己聚焦时能用、切到游戏就没反应"的成因 ✓ —— 前者走的是
          **窗口内快捷键**那条降级路 ✓）。
      ⇒ 治法只有两条（都得人来决定 ✓）：**工作台也用管理员身份启动** ✓，或
        **把游戏改回不提权启动** ✓（游戏快捷方式 → 属性 → 取消"以管理员身份运行"✓）。
    """
    if self_elevated() is not False:
        return ""                                        # 我们本来就是提权 / 问不到 ⇒ 不啰嗦 ✓
    _names = elevated_visible_titles()
    if not _names:
        return ""
    return ("⚠ 现在这些窗口是**以管理员身份**跑的：%s ⇒ 我们的全局热键 `F11` 在它们聚焦时"
            "**收不到** ✗（Windows 的 UIPI：低权限收不到发给高权限窗口的按键 ✓）。"
            "治法：**用管理员身份启动本工作台** ✓，或把它们的「以管理员身份运行」去掉 ✓"
            % "、".join("「%s」" % n for n in _names))


def remove_hook():
    """卸掉键盘钩子（**窗口关闭时必须调** ✓ 不然它会一直挂在系统钩子链上 ✓）。"""
    global _hook_proc, _hook_handle, _hook_vk, _hook_cb
    if _hook_handle is not None:
        try:
            _user32.UnhookWindowsHookEx(_hook_handle)
        except Exception:                        # noqa: BLE001
            pass
    _hook_proc, _hook_handle, _hook_vk, _hook_cb = None, None, None, None
