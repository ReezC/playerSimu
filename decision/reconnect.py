"""断线自动重连：判断当前停在哪个界面，按步骤走回游戏。

分工：

    perception/ui_state.py   判别界面（模板锚点）
    decision/reconnect.py    状态机：什么时候停自动、下一步做什么、什么时候放弃
    decision/input.py        真正发键（Enter）/ 点鼠标

为什么单独一层：状态机能脱离抓屏和 Qt 单独测（喂界面 id 就行），而且
「什么时候动手」这条策略最容易出事，值得有个独立、看得见的地方。

**三条安全绳**（少一条都可能乱点）：

    1. 只在「玩家框丢了够久」之后才去探界面 —— 只是被树挡住、走出视野不算断线；
    2. 探到的必须是**已知的断线界面**才动手；判不出界面时绝不动
       （判不出还乱按，就可能在游戏里狂发 Enter 打开一堆窗口）；
    3. 界面变了才立刻做下一步；界面没变就等超时重试，重试到上限**放弃并报警**。

**为什么每步都要先确认界面**：选角界面上「开始游戏 / 创建角色 / 删除角色」三个
按钮紧挨着，盲点一下就可能把角色删了。所以本模块只做「判到 A 界面 → 做 A 动作」，
判不出就什么都不做。

鼠标点击（选服务器 / 选频道）**故意还没接**：鼠标只有相对位移，要先做
「撞角归零 + counts/px 标定」（见 docs/断线重连设计.md 6.1）。没标定就点，
落到哪全凭运气 —— 宁可不做，也不乱点。
"""

import time

from perception import ui_state

# 界面 -> (动作, 说明)。动作只有三种：
#     "enter"  按一次回车
#     "click"  鼠标点击（还需要标定，见模块文档）
#     "wait"   等它自己过去（排队）
STEPS = {
    ui_state.UI_LOGIN_ERR: ("enter", "消掉断线提示框"),
    ui_state.UI_LOGIN: ("enter", "点「连接」"),
    ui_state.UI_CHANNEL_LIST: ("click", "点服务器"),
    ui_state.UI_CHANNEL_PANEL: ("click", "点频道"),
    ui_state.UI_QUEUE: ("wait", "排队中"),
    ui_state.UI_CHAR_SELECT: ("enter", "进游戏"),
}

# 判到这些界面就认为「人已经不在游戏里」→ 停自动 + 开始重连
RECONNECT_UIS = tuple(STEPS)

# 探界面的最小间隔（秒）。detect() 实测 ~4ms（窗口匹配 + 0.5 缩放），
# 1 秒一次足够跟上界面切换，又不跟推理抢时间。
PROBE_INTERVAL = 1.0

# 状态文字挂多久（秒）。挂太久会让人以为「现在还在重连」；但**不能一帧就清** ——
# 「已放弃，原因是 X」这种结论正是要给人看的。
NOTE_KEEP = 30.0


class Reconnector:
    """断线重连状态机。调用方每帧调一次 update()。

    它自己不做输入，只返回「该做什么」；发键/点鼠标由调用方执行 ——
    这样发键路径只有一处，状态机可以单独测。

    返回的动作：
        {"act": "tap", "key": "enter", "desc": "..."}   按一下某键
        None                                            什么都不做
    """

    def __init__(self, settings):
        self.s = settings
        self.active = False        # 是否在重连流程里
        self.ui = None             # 最近一次判到的界面
        self.note = ""             # 给界面显示的状态文字
        self._lost_since = None    # 玩家框开始丢失的时刻
        self._probe_at = 0.0       # 上次探界面的时刻
        self._scene = None         # 上次动作所在界面（界面变了才算上一步生效）
        self._scene_at = 0.0       # 进入当前界面的时刻
        self._tries = 0            # 当前界面上已动作次数
        self._auto_was_on = False  # 断线前自动是不是开着（决定要不要恢复）
        self._note_until = 0.0     # 状态文字保留到什么时候（见 NOTE_KEEP）
        self._now = 0.0            # 最近一次 update 的时间（_say 算过期用）

    # ---------------- 唯一入口 ----------------

    def update(self, frame, player_found, now):
        """每帧调一次。frame 是当前画面（BGR），player_found 是玩家框有没有。

        返回要执行的动作 dict 或 None（见类文档）。
        """
        s = self.s
        self._now = now

        # 关闭开关：清干净，不干预
        if not s.reconnect_enabled:
            if self.active:
                self._reset()
            self._say("", now)
            return None

        # 玩家框回来了 → 已经回到游戏
        if player_found:
            self._lost_since = None
            self._probe_at = 0.0
            self.ui = None
            if self.active:
                return self._finish(now)
            self._decay_note(now)
            return None

        # ---- 玩家框丢失 ----
        if self._lost_since is None:
            self._lost_since = now

        if not self.active:
            # 没在重连：必须「自动开着」才参与 —— 自动本来就是关的，
            # 说明人在自己操作（比如手动重连），别去插手。
            if not s.enabled:
                self._decay_note(now)
                return None
            # 丢一会儿再说：被树挡住/走出视野也会丢框，先等够时间
            wait = max(0.0, float(s.reconnect_probe_after_lost_sec))
            if now - self._lost_since < wait:
                self._decay_note(now)
                return None

        # 探界面（限流）。判不出界面时保留上一次结果 —— 界面切换中间会有
        # 一两帧判不出来，立刻清掉会让状态机误以为「界面变了」而重复动作。
        if now - self._probe_at >= PROBE_INTERVAL:
            self._probe_at = now
            self.ui = ui_state.detect(frame)
        ui = self.ui

        if ui not in RECONNECT_UIS:
            if self.active:
                self._say("等待画面…（当前判不出界面）")
            return None

        # 断线判断成立 → 先停自动，再开始走流程
        if not self.active:
            self.active = True
            self._scene = None
            self._tries = 0
            self._scene_at = now
            self._auto_was_on = True      # 能走到这里说明 s.enabled 是开的
            s.enabled = False
            self._say("检测到%s —— 已停止自动" % ui_state.UI_NAMES.get(ui, ui))

        act, desc = STEPS[ui]

        # 界面变了 = 上一步生效了 → 立刻做这一步
        if ui != self._scene:
            self._scene = ui
            self._scene_at = now
            self._tries = 0
            return self._do(act, desc, now)

        # 界面没变：等超时再重试
        if act == "wait":
            limit = max(1.0, float(s.reconnect_queue_timeout_ms) / 1000.0)
            if now - self._scene_at >= limit:
                self._give_up("排队超过 %.0f 秒" % limit)
            else:
                self._say("排队中…（已等 %.0f 秒）" % (now - self._scene_at))
            return None

        step = max(0.1, float(s.reconnect_step_timeout_ms) / 1000.0)
        if now - self._scene_at < step:
            self._say("%s：等界面变化…" % desc)
            return None
        if self._tries >= max(1, int(s.reconnect_max_retry)):
            self._give_up("%s 重试 %d 次仍停在同一界面" % (desc, self._tries))
            return None
        self._scene_at = now
        return self._do(act, desc, now)

    # ---------------- 内部 ----------------

    def _do(self, act, desc, now):
        """生成一个动作。注意：**每个动作前都已经确认过界面**。"""
        s = self.s
        if act == "enter":
            self._tries += 1
            key = s.keymap.get("enter") or "enter"
            self._say("%s：按 %s" % (desc, key))
            return {"act": "tap", "key": key, "desc": desc}
        if act == "click":
            # 鼠标点击还没接（要先把鼠标标定做掉，见模块文档）
            self._say("%s：需要鼠标点击，但还没做鼠标标定 —— 先不动手" % desc)
            return None
        # "wait"：界面没变时上面已经处理，这里只是「刚切到排队界面」
        self._say("排队中…")
        return None

    def _finish(self, now):
        """回到游戏：结束重连，按需恢复自动。"""
        was = self._auto_was_on and bool(self.s.reconnect_resume_auto)
        self.active = False
        self._scene = None
        self._tries = 0
        if was:
            self.s.enabled = True
            self._say("已回到游戏 —— 恢复自动")
        else:
            self._say("已回到游戏")
        return None

    def _give_up(self, why):
        """放弃：停在这里并报警 —— 绝不带着错误状态继续打怪。"""
        self.active = False
        self._scene = None
        self.s.enabled = False
        self._say("重连放弃：%s（自动保持停止）" % why)

    def _reset(self):
        self.active = False
        self._scene = None
        self._tries = 0
        self.ui = None

    def _say(self, text, now=None):
        """更新状态文字。非空的会在 NOTE_KEEP 秒后过期（见 _decay_note）。"""
        self.note = text
        if text:
            self._note_until = ((self._now if now is None else now) + NOTE_KEEP)
        try:
            self.s.reconnect_note = text
        except Exception:
            pass

    def _decay_note(self, now):
        """状态文字挂够时间就清掉 —— 免得几小时前的旧结论还挂在界面上。

        只在这两种「没在重连」的情况下调用：玩家框回来了、或自动本来就关着。
        放弃/结束的文字因此能停留一会儿给人看见，而不是下一帧就没了。
        """
        if self.note and now >= self._note_until:
            self._say("", now)
