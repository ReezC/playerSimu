"""一帧的第 3 段：**可视化配置热读**（阶段 1「编排收口」第二刀 ✓ 2026-10-20 ✓）。

搬自 `gui/live_thread.py::_run`（**行为零改动** ✓）：原来那句
`if time.perf_counter() - _vis_refresh_last >= 1.0:` 的节流 ＋ 后面 20 来个
颜色 / 线宽 / 画不画的换算，整段搬过来 ✓（顺序、默认值、回退口径一字不改 ✓）。

⚠⚠ **为什么这些值挂在线程上（`th._vis_ui`）而不是进 `FrameCtx`**（用户 2026-10-20 选 B ✓）：
  它们是**跨帧活着**的东西 —— 1 秒才重读一次，中间每一拍用的都是**上一份** ✓。
  按 `FrameCtx` 自己的判据"**下一帧还会用吗？**"⇒ 会 ⇒ **不进盒子** ✗
  （进了就成了"跨帧状态藏在一个看着像仅本帧的名字里"✓ 那种 bug 最难查 ✓）。
  ⇒ 所以这里两样东西都留在线程上：`th._vis_ui`（这份配置）＋ `th._vis_refresh_last`（上次重读时刻 ✓）。

⚠ `theme` 必须**由调用方传**（`gui/theme.py` 在 gui 层 ✓ 而 `runtime` **不许 import gui** ✗
  —— 见 `selftest_arch` 那条"`runtime` 里出现指向 gui/tools 的 import"✓）。
  同理 `player_class` / `mob_class` 也是调用方给（模型加载后会**按名字重对齐** ✓ 见 `_run` ✓）。

判据 `selftest_live_panel::t_frame_vis_stage`：①首次必算 ②1 秒内不重算（同一份对象）
③到点重算并换新对象 ④类号变了、下一次重算就跟着变 ⑤口径（默认值/回退）与原来一致 ✓。
"""

import time

from perception.detect.classes import CLASS_MOB, CLASS_PLAYER
#: ⚠ 名字对齐 `gui/live_thread.py` 那边：它 import 的是 `ZH_NAMES as CLASS_NAMES` ✓
#:   （直接 `import CLASS_NAMES` 会 **ImportError** —— 这个模块里没有这个名字 ✗）
from perception.detect.classes import ZH_NAMES as CLASS_NAMES

#: 重读间隔（秒）—— 用户 2026-10-05 ✓："改完 1 秒内生效"✓（原代码就是 `>= 1.0` ✓）
REFRESH_S = 1.0


class VisUI:
    """一份可视化配置：**画框要用的颜色 / 线宽 / 开关**（一处算清 ✓ 绘制端只读 ✓）。

    字段名 = `gui/live_thread.py` 里原来那些局部名去掉前导下划线 ✓（`_p_color` → `p_color`），
    这样搬动时"哪个值对哪个名字"一眼看得住 ✓。
    """

    __slots__ = ("cfg", "cls_colors", "p_color", "m_color", "p_label", "m_label",
                 "lock_color", "attack_color", "min_attack_color", "chase_jump_color",
                 "jump_attack_color", "vision_color", "vision_width",
                 "lock_width", "attack_width", "min_attack_width",
                 "jump_attack_width", "chase_jump_width",
                 "on_lock", "on_attack", "on_blind", "on_chase",
                 "on_jump_attack", "on_vision")

    def __init__(self, cfg, cls_colors, p_color, m_color, p_label, m_label,
                 lock_color, attack_color, min_attack_color, chase_jump_color,
                 jump_attack_color, vision_color, vision_width,
                 lock_width, attack_width, min_attack_width,
                 jump_attack_width, chase_jump_width,
                 on_lock, on_attack, on_blind, on_chase, on_jump_attack, on_vision):
        self.cfg = cfg
        self.cls_colors = cls_colors
        self.p_color = p_color
        self.m_color = m_color
        self.p_label = p_label
        self.m_label = m_label
        self.lock_color = lock_color
        self.attack_color = attack_color
        self.min_attack_color = min_attack_color
        self.chase_jump_color = chase_jump_color
        self.jump_attack_color = jump_attack_color
        self.vision_color = vision_color
        self.vision_width = vision_width
        self.lock_width = lock_width
        self.attack_width = attack_width
        self.min_attack_width = min_attack_width
        self.jump_attack_width = jump_attack_width
        self.chase_jump_width = chase_jump_width
        self.on_lock = on_lock
        self.on_attack = on_attack
        self.on_blind = on_blind
        self.on_chase = on_chase
        self.on_jump_attack = on_jump_attack
        self.on_vision = on_vision

    def __repr__(self):
        return "<VisUI lock=%s attack=%s vision=%s>" % (
            self.on_lock, self.on_attack, self.on_vision)


def step(th, player_class, mob_class, theme, now=None):
    """到点就重读一次可视化配置 ⇒ 返回**当前这份** `VisUI`（没到点返回上一份 ✓）。

    :param th: `LiveThread` —— 本段读写的跨帧状态只有 `th._vis_ui` / `th._vis_refresh_last` ✓
    :param player_class: 玩家**实际解析出来的**类号（模型可能别处训的、顺序不同 ✓）
    :param mob_class: 怪物类号（同上）
    :param theme: `gui.theme`（**由调用方传** ✓ 见文件开头那条 ⚠）
    :param now: 时刻（用例要控时间 ✓ 缺省 `time.perf_counter()` ✓）
    """
    _now = time.perf_counter() if now is None else float(now)
    _ui = getattr(th, "_vis_ui", None)
    _last = float(getattr(th, "_vis_refresh_last", 0.0) or 0.0)
    # 未到点 ⇒ 沿用上一份（首拍 `_last = 0.0` ⇒ 必算 ✓ 和老代码一致 ✓）
    if _ui is not None and (_now - _last) < REFRESH_S:
        return _ui
    th._vis_refresh_last = _now
    cfg = theme.load_vis()
    cls_colors = theme.class_colors()
    # 画框/标签按**实际解析出来的**类 id 取（权重可能在别处训、顺序和我们不同），
    # 取不到就回退到类别表里的默认项。
    _ui = VisUI(
        cfg=cfg,
        cls_colors=cls_colors,
        p_color=cls_colors.get(player_class, cls_colors[CLASS_PLAYER]),
        m_color=cls_colors.get(mob_class, cls_colors[CLASS_MOB]),
        p_label=CLASS_NAMES.get(player_class, CLASS_NAMES[CLASS_PLAYER]),
        m_label=CLASS_NAMES.get(mob_class, CLASS_NAMES[CLASS_MOB]),
        lock_color=theme.hex_to_bgr(cfg["lock_color"]),
        attack_color=theme.hex_to_bgra(cfg["attack_color"]),
        min_attack_color=theme.hex_to_bgra(cfg["min_attack_color"]),
        chase_jump_color=theme.hex_to_bgra(cfg["chase_jump_color"]),
        # ⭐ 「跳跃攻击范围框」（用户 2026-10-12 ✓）
        jump_attack_color=theme.hex_to_bgra(cfg["jump_attack_color"]),
        vision_color=theme.hex_to_bgra(cfg["vision_color"]),
        vision_width=int(cfg["vision_width"]),
        # ⭐⭐ **各框线宽**（用户 2026-09-28 ✓"给其他的粗细也加配置"）—— 就在
        #   「界面 → 辅助线与标记」里**色块右边那一格**调 ✓；⚠ **默认值 = 原来
        #   在下面写死的那个数**（锁定框 3、其余 2 ✓ 观感一字不变 ✓）。
        lock_width=int(cfg.get("lock_width", 3) or 3),
        attack_width=int(cfg.get("attack_width", 2) or 2),
        min_attack_width=int(cfg.get("min_attack_width", 2) or 2),
        jump_attack_width=int(cfg.get("jump_attack_width", 2) or 2),
        chase_jump_width=int(cfg.get("chase_jump_width", 2) or 2),
        on_lock=bool(cfg.get("lock_on", True)),
        on_attack=bool(cfg.get("attack_on", True)),
        on_blind=bool(cfg.get("min_attack_on", True)),
        on_chase=bool(cfg.get("chase_jump_on", True)),
        on_jump_attack=bool(cfg.get("jump_attack_on", True)),
        on_vision=bool(cfg.get("vision_on", True)),
    )
    th._vis_ui = _ui
    return _ui
