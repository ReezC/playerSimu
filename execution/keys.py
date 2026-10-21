# -*- coding: utf-8 -*-
"""键位表与键名翻译：**纯数据 + 纯函数** ✓（零副作用、零依赖 ✓）。

为什么独立成文件：`input.py` 里"发键"和"键位数据"混在一起 ✗ ⇒
改一张键位表要动 983 行的传输文件 ✗。这里不许 import 任何本仓模块 ✓
（连 `core` 都不要 ✓ 它得能被随便谁安全导入 ✓）。
"""



# ---- 键名 → 虚拟键码 ----
_VK = {
    "left": 0x25, "right": 0x27, "up": 0x26, "down": 0x28,
    "ctrl": 0x11, "alt": 0x12, "shift": 0x10,
    "enter": 0x0D, "space": 0x20, "esc": 0x1B, "tab": 0x09,
    "backspace": 0x08, "del": 0x2E, "insert": 0x2D, "home": 0x24, "end": 0x23,
    "pageup": 0x21, "pagedown": 0x22,
    "grave": 0xC0,   # ` / ~（反引号键，VK_OEM_3）
    # 标点符号键（VK_OEM_*，手动输入 / 自定义按键要用）
    ",": 0xBC, ".": 0xBE, "/": 0xBF, ";": 0xBA, "'": 0xDE,
    "[": 0xDB, "]": 0xDD, "\\": 0xDC, "-": 0xBD, "=": 0xBB,
}
_VK.update({"f%d" % i: 0x70 + i - 1 for i in range(1, 13)})
_VK.update({chr(c): c for c in range(ord("A"), ord("Z") + 1)})   # a~z
_VK.update({str(i): ord("0") + i for i in range(10)})            # 0~9


# 键名 → 给 UI 显示的友好名（下拉框里用）
DISPLAY = {
    "left": "←", "right": "→", "up": "↑", "down": "↓",
    "ctrl": "Ctrl", "alt": "Alt", "shift": "Shift",
    "space": "空格", "enter": "回车", "esc": "Esc", "tab": "Tab",
    "del": "Del", "insert": "Ins", "home": "Home", "end": "End",
    "pageup": "PgUp", "pagedown": "PgDn",
    "grave": "~",
}
DISPLAY.update({"f%d" % i: "F%d" % i for i in range(1, 13)})
DISPLAY.update({chr(c): chr(c).upper() for c in range(ord("A"), ord("Z") + 1)})
DISPLAY.update({str(i): str(i) for i in range(10)})


# 键盘映射里可以选的键（按常用度排序）
CHOICES = ["left", "right", "up", "down", "ctrl", "alt", "shift", "space"] + \
          ["f%d" % i for i in range(1, 13)] + \
          [chr(c) for c in range(ord("A"), ord("Z") + 1)] + \
          [str(i) for i in range(10)] + \
          [",", ".", "/", ";", "'", "[", "]", "\\", "-", "=", "grave"]



# 本机操作键：**硬编码，永不发给游戏**。
#   F10 = 触控板触控模式开关（gui/touchpad.py）
#   F11 = 开关自动（默认热键；全局热键注册以外的地方也不该发它）
#   F12 = 预留（调试 / 截图之类）
# 拦在所有发送路径的最里面 —— 决策层、手动输入、断线重连都从 key_down/key_up/tap
# 出去，所以在这里拦一次就够。不拦的话：按一下本地开关，游戏里会跟着触发一次
# （最典型的是「按住 Ctrl 用触控板」——Ctrl 是默认攻击键，一按就打出去了）。
LOCAL_ONLY = ("f10", "f11", "f12")




def is_local_only(name):
    """是本机操作键吗（是的话，任何发送路径都不该把它发出去）。"""
    return str(name).lower() in LOCAL_ONLY




# 默认键位
DEFAULT_KEYMAP = {
    "auto": "f11",      # 开关自动
    "left": "left",     # 移动←
    "right": "right",   # 移动→
    "up": "up",         # 移动↑
    "down": "down",     # 移动↓
    "attack": "ctrl",   # 输出（攻击）
    "jump": "alt",      # 跳跃
    #: ⭐ 「拾取」（用户 2026-10-06 ✓）：**只给「战斗参数 → 自动拾取」用** ✓（那个开关开了
    #:   才会被点按 ✓）。默认 `"z"` = 这类游戏的常见拾取键 ✓ —— 不对就在
    #:   「键盘映射」组里点一下那个按钮、再按你游戏里的拾取键（Esc 取消 ✓）。
    "pickup": "z",      # 拾取（自动拾取）
    "hp_pot": None,     # 补血（自动喝药）
    "mp_pot": None,     # 补蓝（自动喝药）
    "feed_pet": None,   # 喂宠
    "shop": None,       # 商城
    "enter": "enter",   # 回车
}




def display_name(key):
    return DISPLAY.get(key, key)




def resolve_vk(key):
    """键名 → VK 码；不认识返回 None。"""
    if isinstance(key, int):
        return key
    n = str(key).lower()
    vk = _VK.get(n)
    if vk is None:
        # 字母键在 _VK 里存的是大写（'A'~'Z'），小写查不到时回退大写
        vk = _VK.get(n.upper())
    return vk




#: ⭐⭐ 「清通道」前的第 ⓪ 步要补按的键（**名义名** ✓ 与 `keymap` 同一套 ✓）。
#: 顺序照手动那个按钮的既有顺序（用户 2026-10-01 定的 ✓ 不新造 ✗）。
#: ⛔⛔ **不带 `up`**（用户 2026-10-11 ✓ 原话："这两个 **排除↑键把 只按←↓→**"✓）：
#:   这游戏里 **↑ = 抓绳 / 上梯**（同源口径见 `agent` 里那段"输出行为CD 只摘左右"的说明 ✓：
#:   "上下是**爬绳 / 上下跳** …… 绝不动 ↑/↓"✗）—— 补按本来是"刷掉卡住的键"✓，
#:   可它自己会**把角色按上绳子** ✗ ⇒ 清键动作反倒造出一个新卡点 ✓ ⇒ ↑ **一律不点** ✓。
#:   ⚠ **↓ 留着**（用户明确要 ←↓→ ✓）：它是"往下"那条的刷新手段 ✓（下跳那条链就靠 ↓ 触发 ✓），
#:     一次 `tap` 只有 60ms ✓ 不会把人按趴住 ✓。
DIRECTION_KEYS = ("left", "right", "down")
