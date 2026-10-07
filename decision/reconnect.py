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

鼠标点击（选服务器 / 选频道）**已接**（2026-10-06 ✓）：鼠标只有相对位移，所以走
「撞角归零 + 走位 + 左键」那一条（换算与发指令的唯一实现在 `decision/mouse_aim.py` ✓）。
⚠ **没标定就一点都不点**（`mouse_aim.aim_available()` ✓）：不是"点了再回退" ✗ ——
落到哪全凭运气，而选角界面上「删除角色」就在同一片区域（见设计文档 §9 ✓）。

**「停自动」与「自动重连」是两件事**（2026-09-30 解耦 ✓ 用户原话："刚刚判定到
断线了，在断线的时候停止自动吧"）：判到断线界面 ⇒ **必停自动** ✓（安全行为，
常开、不看开关）；`reconnect_enabled` 只管"要不要自动按回车/走回游戏" ✓。
（原来开关把两件事连坐 ✗ 默认 False ⇒ 断线了自动还在瞎跑 ✗。）

**「恢复自动」要过界面那份体检**（2026-10-07 ✓ 用户在三选一里选了"**体检**"）：
回到游戏**不直接**开自动 ✗ —— 只置 `settings.reconnect_resume_pending`（旗子 ✓），
由 `gui/player_panel._poll_reconnect_resume`（界面线程 ✓ 500ms）用
`_precheck_problems`（= "开自动前体检"的**同一份**判据 ✓）过一遍 ⇒ 通过才开 ✓；
**不过 ⇒ 不恢复 + 把原因写在状态栏** ✓。为什么不能在这儿体检：`decision/` 读不了
项目 / 标定 / 地形文件（那是界面的事 ✓），而且实时线程里**不许**弹模态框（会卡住
整条回路 ✗）⇒ 只置旗子，跑在 500ms 那趟界面的车上 ✓（同 `rest_request` 的做法 ✓）。
"""

import time

from core import behavior
from decision import mouse_aim
from perception import ui_state

# 界面 -> (动作, 说明)。动作只有三种：
#     "enter"  按一次回车
#     "click"  鼠标点击（撞角归零 + 走位 + 左键，见 mouse_aim）
#     "wait"   等它自己过去（排队）
STEPS = {
    ui_state.UI_LOGIN_ERR: ("enter", "消掉断线提示框"),
    ui_state.UI_LOGIN: ("enter", "点「连接」"),
    ui_state.UI_CHANNEL_LIST: ("click", "点服务器"),
    ui_state.UI_CHANNEL_PANEL: ("click", "点频道"),
    ui_state.UI_QUEUE: ("wait", "排队中"),
    ui_state.UI_CHAR_SELECT: ("enter", "进游戏"),
}

#: 「click」那两步点**画面的哪个比例位置**（0~1，**不是像素** ✓ —— 换分辨率 / 窗口大小
#: 自动适配，同 HP/MP 条那套思路 ✓ 见设计文档 §7）。
#:
#: 值是 2026-10-06 **从素材里量出来的**（不是目测 ✗）：`tools/_probe_disc_frames.py`
#: 的 `--bars` / 木色掩模在 1920×1080 的素材帧上量出 —— 服务器 5 格的中心在
#: x = 698 / 849 / 999 / 1149 / 1300、y ≈ 245 ⇒ 第 1 格 (698, 245)；
#: 频道面板 4 列 × 5 行、格子中心 x = 788 / 919 / 1049 / 1180、y = 550 / 593 / 636 /
#: 680 / 724 ⇒ 第 1 格 (788, 550)。（设计文档里那几条"估算值"已按实测替换 ✓。）
CLICK_TARGETS = {
    ui_state.UI_CHANNEL_LIST: ("reconnect_server_x", "reconnect_server_y", "服务器（第 1 格）"),
    ui_state.UI_CHANNEL_PANEL: ("reconnect_channel_x", "reconnect_channel_y", "频道（面板第 1 格）"),
}

#: ⭐⭐ **频道面板的格子布局**（把「频道（第几格）」折成画面比例 ✓ —— 用户 2026-10-07 要的
#: 那个正经参数 ✓：挂机保护 → 断线重连 → 「频道」（整数）✓）。
#:
#: ⚠ 数从哪来：2026-10-06 从断线素材里**量出来的格子中心**（@1920×1080 ✓ 见上面那段 ✓）：
#:   横向 788 / 919 / 1049 / 1180 ⇒ 格距 ≈131px ⇒ 比例 **131/1920 ≈ 0.0682** ✓；
#:   纵向 550 / 593 / 636 / 680 / 724 ⇒ 格距 ≈43px ⇒ 比例 **43/1080 ≈ 0.0398** ✓。
#: ⇒ 第 N 格（**从左到右、从上到下**数 ✓ 一行 `COLS` 个 ✓）：
#: ⭐ **这个顺序已经真机验过** ✓（2026-10-07 ✓ 用户把「频道」设成 **19** 跑了一遍，
#:   点中、进频道、走完 ✓ 并确认"19 就是我要的" ✓）⇒ 不是猜的 ✓（见设计文档 §11 ✓）。
#:     x = `reconnect_channel_x` + ((N−1) % COLS) × DX ✓
#:     y = `reconnect_channel_y` + ((N−1) // COLS) × DY ✓
#: ⭐ 存的是**比例** ⇒ 换分辨率 / 窗口大小**不用重配** ✓（同 HP/MP 条那套 ✓）。
#: ⚠ 想改"第几格 = 哪个频道"的对应关系（万一客户端编号不是从左到右 ✓）⇒ **只改这一段** ✓
#:   （一处实现 ✓ 别在界面里再写一遍算式 ✗）。
CHANNEL_GRID_COLS = 4
CHANNEL_GRID_ROWS = 5
CHANNEL_GRID_DX = 131.0 / 1920.0     # ≈ 0.0682（横向格距，比例 ✓）
CHANNEL_GRID_DY = 43.0 / 1080.0      # ≈ 0.0398（纵向格距，比例 ✓）
CHANNEL_GRID_MAX = CHANNEL_GRID_COLS * CHANNEL_GRID_ROWS      # 20 ✓

# 判到这些界面就认为「人已经不在游戏里」→ 停自动 + 开始重连
RECONNECT_UIS = tuple(STEPS)

# 探界面的最小间隔（秒）。detect() 实测 ~4ms（窗口匹配 + 0.5 缩放），
# 1 秒一次足够跟上界面切换，又不跟推理抢时间。
PROBE_INTERVAL = 1.0

# 状态文字挂多久（秒）。挂太久会让人以为「现在还在重连」；但**不能一帧就清** ——
# 「已放弃，原因是 X」这种结论正是要给人看的。
NOTE_KEEP = 30.0

#: ⭐⭐ **要点「双击」的界面**（2026-10-07 ✓ 真机验出来的，别改回去 ✗）：
#: 频道面板**单击只是"选中"** ✗ —— 真机实测**单击三次全无反应** ✓，而**双击那一下直接
#: 进到「选择角色」** ✓（`data/_accept_click2.py --double` ✓ 见设计文档 §11 ✓）。
#: ⚠ 服务器行（`UI_CHANNEL_LIST`）**单击就行** ✓（真机验过 ✓）⇒ **别把它也双击** ✗。
DOUBLE_CLICK_UIS = (ui_state.UI_CHANNEL_PANEL,)

#: 双击两下之间的间隔（秒 ✓）。
#: ⚠⚠ 为什么**不**在 `mouse_aim` 里 `sleep` 一下再点第二下 ✗：那条路跑在**实时回路线程**里，
#:   睡一下就是丢帧 / 积压（见 `mouse_aim.click_ratio` 的模块头 ✓ 那里有一条专门的纪律 ✓）
#:   ⇒ 第二下交给**状态机**在**下一拍**补 ✓（`update()` 每帧都跑 ✓，这里只判"到点了没" ✓）。
#: 实测 60ms 就能被认出来 ✓；这里给 **120ms** 留余量 ✓ —— 再大就有被认成"两次单击"的风险 ✗
#: （系统双击阈值 ~500ms，但游戏自己那套口径更短 ✓ 别试边缘值 ✗）。
DOUBLE_CLICK_GAP = 0.12

#: ⭐⭐ **会出现"假玩家框"的界面**（2026-10-07 ✓ 真机日志逮到的）：**选角界面**里那个角色
#: 立绘会被玩家检测当成"玩家框" ✗ ⇒ 状态机误判"人回来了"⇒ 提前收工 ⇒ 下一拍框又没了 ⇒
#: 白跑一轮 ✓（日志：03:45:48「已回到游戏」→ 03:45:49「检测到选择角色 → 又走一轮」✓）。
#: ⇒ 只有这几个界面里 "玩家框出现" **不算回到游戏** ✓。
#:
#: ⚠⚠ **只放"真会画出角色"的那个界面** ✗ —— **别扩成"所有断线系界面"**（`RECONNECT_UIS` ✗）：
#:   那个口径的副作用是"**万一界面误判/卡住，人已经在游戏里了，状态机却还在按 Enter**" ✗✗
#:   （游戏里 Enter 会开聊天/关弹窗 ✓ 那是真会捣乱的动作 ✗）；而这条窄口径的失效方式只是
#:   "多走一轮" ✓（真机验过：无害 ✓ 第二轮按下 Enter 就真的进去了 ✓）。
#:   ⭐ **失效方式轻的那个口径才是对的口径** ✓。
FAKE_PLAYER_UIS = (ui_state.UI_CHAR_SELECT,)


class Reconnector:
    """断线重连状态机。调用方每帧调一次 update()。

    它自己不做输入，只返回「该做什么」；发键/点鼠标由调用方执行 ——
    这样发键路径只有一处，状态机可以单独测。

    返回的动作（**都由调用方执行** ✓）：
        {"act": "tap",   "key": "enter", "desc": "..."}            按一下某键
        {"act": "click", "x": 0.36, "y": 0.23, "desc": "..."}      **画面比例**坐标，
          调用方走 `mouse_aim.click_ratio(frame.shape, x, y)` 落地（换算 + 发指令只有那一处 ✓）
        {"act": "click2", "desc": "..."}                           **再点一下左键**（双击的第二下 ✓
          —— 只给 `DOUBLE_CLICK_UIS` 里那些界面用 ✓，调用方走 `mouse_aim.repeat_click()` ✓）
        None                                                       什么都不做
    ⚠ `click` 执行完**必须**调 `feedback(ok, why)` 回填（成败只有发完才知道 ✓ 见那个方法 ✓）。
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
        #: 最近一次**动作执行失败**的原因（调用方经 `feedback` 回填 ✓）。
        #: 只为一件事：放弃时把"为什么没成"一起说出来（否则只有"重试 N 次"✗）。
        self._last_fail = ""
        #: 上一拍**没出手**的原因（条件不具备：没标定 / 比例没配 ✓）。
        #: 只为一件事：界面没变那几拍**别改口说"等界面变化…"** ✗（见 `update` 里那段 ✓）。
        self._declined = ""
        #: 最近一条**已经写进 behavior.log** 的状态文字（`_say` 判变化用 ✓ —— 防刷屏）。
        self._logged = None
        #: ⭐⭐ 「双击」用（见 `DOUBLE_CLICK_UIS` / `DOUBLE_CLICK_GAP` ✓）：
        #: `_click_at` = 第一次点击那一拍的时刻（0 = 还没点 ✓）；
        #: `_doubled` = 这一轮里第二下**已经补过**（别每拍都补 ✗）。
        self._click_at = 0.0
        self._doubled = False

    # ---------------- 唯一入口 ----------------

    def update(self, frame, player_found, now):
        """每帧调一次。frame 是当前画面（BGR），player_found 是玩家框有没有。

        返回要执行的动作 dict 或 None（见类文档）。
        """
        s = self.s
        self._now = now
        # ⭐ **「停自动」与「自动重连」解耦**（2026-09-30 用户要求 ✓ 原话："刚刚
        #   判定到断线了，在断线的时候停止自动吧"）—— 原来这个早退把两件事连坐
        #   ✗（`reconnect_enabled` 默认 False ⇒ 断线了自动还在瞎跑 ✗）：现在
        #   **判到断线界面必停自动** ✓（安全行为、常开 ✓）；开关只管重连动作 ✓。
        reconnect_on = bool(s.reconnect_enabled)

        # 玩家框回来了 —— ⚠⚠ **但这不等于"回到游戏"** ✗（2026-10-07 ✓ 真机日志逮到的）：
        #   **选角界面**里那个角色会被玩家检测当成"玩家框" ✓ ⇒ 那一拍就误判"人回来了"⇒
        #   提前收工 ⇒ 下一拍框又没了 ⇒ 又开一轮重连 ✗（真机日志：03:45:48「已回到游戏」→
        #   03:45:49「检测到选择角色 → 又走一轮」→ 03:45:56 才真进游戏 ✓
        #   —— 结果无害，但白跑一轮、状态栏还闪两下 ✓）。
        #   ⇒ 口径：**只有"当前不是断线系界面"时，"玩家框出现"才算回到游戏** ✓
        #     （判不出界面 / 判成别的 ⇒ 算回到游戏 ✓ 与老行为一致 ✓）。
        #   ⚠ `self._probe_at = 0.0` = **催下一拍立刻重探界面** ✓：判据要用**刚判的**界面 ✓，
        #     不是几百毫秒前那份旧结果 ✗（回来时少等一拍 ✓）。
        if player_found:
            self._lost_since = None
            self._probe_at = 0.0
            if self.ui not in FAKE_PLAYER_UIS:
                self.ui = None
                if self.active:
                    return self._finish(now)
                self._decay_note(now)
                return None
            # ⭐ 还停在**会画出角色**的那个界面（选角 ✓）⇒ **不当成回到游戏** ✓
            #   落到下面继续走流程 ✓（选角那一步照旧按 Enter 进游戏 ✓ 那才是真的"回到游戏" ✓）。
            if self.active:
                self._say("画面里有像玩家框的东西，但还停在%s —— 继续走重连 ✓"
                          % ui_state.UI_NAMES.get(self.ui, self.ui))

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
            self._last_fail = ""
            self._declined = ""
            self._scene_at = now
            self._auto_was_on = True      # 能走到这里说明 s.enabled 是开的
            s.enabled = False
            if reconnect_on:
                self._say("检测到%s —— 已停止自动，开始重连"
                          % ui_state.UI_NAMES.get(ui, ui))
            else:
                self._say("检测到%s —— 已停止自动（重连未开启：不会自动按键 ✓）"
                          % ui_state.UI_NAMES.get(ui, ui))

        if not reconnect_on:
            return None                   # 只停自动 ✓：按 Enter 等重连动作不做 ✓

        act, desc = STEPS[ui]

        # 界面变了 = 上一步生效了 → 立刻做这一步
        if ui != self._scene:
            self._scene = ui
            self._scene_at = now
            self._tries = 0
            self._click_at = 0.0            # 换界面 ⇒ 双击那一轮作废 ✓
            self._doubled = False
            return self._do(act, desc, now)

        # ⭐⭐ **「频道格要双击」的第二下**（2026-10-07 ✓ 真机验出来：单击只是"选中" ✗）：
        #   第一下由 `_do` 发（撞角归零 + 走位 + 左键 ✓）；**隔一小会儿**再补一条**光左键** ✓
        #   （光标没动 ✓ 不用重新走位 ✓）。⚠ 补的这一下**不吃 `_tries`** ✓ —— 它不是"重试"，
        #   是**同一次动作的第二次点击** ✗（否则一次双击就等于两次重试 ⇒ 白少一次机会 ✗）。
        #   ⚠ 必须排在下面"等超时重试"之前 ✓（间隔 120ms 本来也远小于步骤超时 ✓）。
        if (ui in DOUBLE_CLICK_UIS and self._click_at
                and not self._doubled
                and now - self._click_at >= DOUBLE_CLICK_GAP):
            self._doubled = True
            self._declined = ""
            self._say("%s：补第二下（这个界面要双击 ✓）" % desc)
            return {"act": "click2", "desc": desc}

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
            # ⭐ 上一拍**根本没出手**（条件不具备：没标定 / 比例没配 ✗）⇒ 这里别
            #   说"等界面变化…" —— 那句话在骗人：界面**不会**因为我们没动手而变 ✓
            #   （实测：两条消息每 3 秒互相盖，界面上看着像在正常工作 ✗）。
            if self._declined:
                self._say("%s：暂时不动手 —— %s（已停在这里 %.0f 秒）"
                          % (desc, self._declined, max(0.0, now - self._scene_at)))
            else:
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
            self._declined = ""
            key = s.keymap.get("enter") or "enter"
            self._say("%s：按 %s" % (desc, key))
            return {"act": "tap", "key": key, "desc": desc}
        if act == "click":
            tgt, why = self._click_target()
            if tgt is None:
                # ⚠ 这里**不是"点了再回退"**：算不出目标 / 鼠标不可用 / 没标定 ⇒
                #   一个字节都不该发出去（落在哪全凭运气 ✗ 见模块文档）。
                #   ⚠ 也不吃 `_tries`：这不是"重试失败"，是"条件不具备" ——
                #     条件具备了（用户去标定）下一拍就能点上 ✓ 不该被判成放弃 ✗。
                #     代价：会一直停在这儿等着 ✓ —— 所以**把等了多久一起说出来** ✓
                #     （否则界面上挂着一句不变的话，看不出它是"卡住了"还是"在等"）。
                #     记进 `_declined`：界面没变那几拍别改口说"等界面变化…" ✗。
                self._declined = why
                self._say("%s：暂时不动手 —— %s（已停在这里 %.0f 秒）"
                          % (desc, why, max(0.0, now - self._scene_at)))
                return None
            xr, yr, name = tgt
            self._tries += 1
            self._declined = ""
            # ⭐ 这一步要不要**双击**（`DOUBLE_CLICK_UIS` ✓）：要 ⇒ 记下这一拍的时刻，
            #   下一拍 `update()` 到点会补第二下（`click2` ✓ 见那边 ✓）。
            if self.ui in DOUBLE_CLICK_UIS:
                self._click_at = now
                self._doubled = False
            self._say("%s：撞角归零后点%s（画面比例 %.3f, %.3f）%s"
                      % (desc, name, xr, yr,
                         "（这个界面要双击 ✓）" if self.ui in DOUBLE_CLICK_UIS else ""))
            return {"act": "click", "x": xr, "y": yr, "desc": desc,
                    "double": self.ui in DOUBLE_CLICK_UIS}
        # "wait"：界面没变时上面已经处理，这里只是「刚切到排队界面」
        self._say("排队中…")
        return None

    def _click_target(self):
        """这一步该点画面的哪个比例位置 → `((x比例, y比例, 人话), why)`。

        判据只有两处：① 这个界面在 `CLICK_TARGETS` 里有没有登记；② 比例是不是
        合法的 0~1 数、且**鼠标这一层真的可用**（`mouse_aim.aim_available` ✓）。
        任何一条不成立 ⇒ `(None, 人话)`，调用方**一律不动手** ✓。
        """
        ui = self.ui
        spec = CLICK_TARGETS.get(ui)
        if spec is None:
            return None, "这个界面（%s）没登记点击位置" % ui_state.UI_NAMES.get(ui, ui)
        xf, yf, name = spec
        try:
            x = float(getattr(self.s, xf))
            y = float(getattr(self.s, yf))
        except (TypeError, ValueError):
            return None, "点击位置没配 / 不是数字（%s / %s）" % (xf, yf)
        if not (0.0 <= x <= 1.0) or not (0.0 <= y <= 1.0):
            return None, "点击位置不在画面里（要看 0~1：%s=%s %s=%s）" % (xf, x, yf, y)
        # ⭐⭐ 「想去第几个频道」：把格号折算成落点（见 `CHANNEL_GRID_*` ✓ 一处实现 ✓）。
        #   ⚠ 只对**频道面板**这一步生效 ✓（服务器行永远是第 1 格 ✓ 真机验过 ✓ 别顺手也挪 ✗）。
        if ui == ui_state.UI_CHANNEL_PANEL:
            try:
                # ⚠ **别写 `or 1`** ✗（反向验证时正是这么漏掉的 ✓）：0 会被悄悄当成 1 ⇒
                #   用户填错了却"看着正常在点" ✗ —— 越界就要**当场拒绝 + 说清** ✓。
                n = int(getattr(self.s, "reconnect_channel", 1))
            except (TypeError, ValueError):
                return None, "「频道」不是整数（去「挂机保护 → 断线重连 → 频道」改 ✓）"
            if not (1 <= n <= CHANNEL_GRID_MAX):
                return None, ("「频道」填的是 %d，超出面板格数（1~%d ✓ %d 列 × %d 行）"
                              "⇒ 不点（改小一点 ✓）"
                              % (n, CHANNEL_GRID_MAX, CHANNEL_GRID_COLS, CHANNEL_GRID_ROWS))
            x += ((n - 1) % CHANNEL_GRID_COLS) * CHANNEL_GRID_DX
            y += ((n - 1) // CHANNEL_GRID_COLS) * CHANNEL_GRID_DY
            name = "%s（第 %d 格）" % (name, n)
            if not (0.0 <= x <= 1.0) or not (0.0 <= y <= 1.0):
                # 格号合法但算出来出界（配置把"第 1 格"配到画面右下角之类 ✓）⇒ 照样不点 ✓
                return None, ("第 %d 格算出来跑到画面外了（%.3f, %.3f）—— 先把「频道 X/Y 比例」"
                              "配成第 1 格的中心 ✓" % (n, x, y))
        ok, why = mouse_aim.aim_available()
        if not ok:
            return None, why
        return (x, y, name), ""

    def feedback(self, ok, why):
        """调用方**执行完一个动作**后回填结果（`ok` + 人话）⇒ 写进状态文字 / 日志。

        为什么要有这条回填：`click` 的成败只有在**发完输入**之后才知道
        （撞角归零走位可能要失败：通道没连上 / 固件没回执 ✓）。状态机自己
        不发输入（那是 `decision/mouse_aim.py` 的活 ✓），但它得**把结果说出来**
        —— 否则界面上只剩上一句"准备点…"，人根本不知道点没点上 ✗。
        """
        if not self.active:
            return
        if ok:
            self._say(why)                 # 成功也留一句（含走了多少、点了哪 ✓）
        else:
            self._say("鼠标点击没成功：%s" % why)
            self._last_fail = str(why or "")
        # 成败单独记一条：`_say` 那句是**人话**（会随界面变），复盘时要能一眼数出
        # "点了几次、成了几次" ⇒ 结构化一条更好查 ✓（与 `mob_goto_ok/_fail` 同款 ✓）。
        behavior.event("reconnect_click", ok=bool(ok), why=str(why or "")[:120])

    def _finish(self, now):
        """回到游戏：结束重连，按需恢复自动。"""
        # ⭐ 只有**重连流程**走回来的才恢复自动 ✓ —— 开关没开时停自动是用户要的
        #   唯一动作 ⇒ 回到游戏也**保持停止** ✓（人自己开 ✓）。
        was = (self._auto_was_on and bool(self.s.reconnect_resume_auto)
               and bool(self.s.reconnect_enabled))
        self.active = False
        self._scene = None
        self._tries = 0
        self._last_fail = ""
        self._declined = ""
        if was:
            # ⭐⭐ **只"请求"恢复自动，不直接开**（用户 2026-10-07 ✓ 三选一里选了"体检"）：
            #   真正开自动前要过 `player_panel._precheck_problems` 那份体检（**同一份判据** ✓）。
            #   ⛔ 这里**不许**自己去开自动（≈ 把 `enabled` 置真 ✗ = 绕过体检）：
            #     · 实时线程里读不了项目 / 标定 / 地形文件（那是界面的事 ✓）；
            #     · 也不许在实时线程弹模态框（会卡住整条回路 ✗）；
            #     · 本仓库的纪律是"跨线程不许直接改状态，走请求/旗子"（同 `rest_request` ✓）。
            #   ⚠ 绕过体检的后果正是 2026-09-29 查了半天的那个病根：标定只做了一半 /
            #     没有地形图 ⇒ 自动照样被打开 ⇒ "开了却只站着不打、过几分钟自己停" ✗。
            #   ⚠ **没有界面侧消费**（headless / 别的宿主）⇒ 旗子一直挂着、自动不会自己开 ✓
            #     （宁可不开 ✓ 状态文字会说清在等什么 ✓）。
            self.s.reconnect_resume_pending = True
            self._say("已回到游戏 —— 已请求恢复自动（等界面体检…）")
        else:
            self._say("已回到游戏（自动保持停止 ✓）")
        return None

    def _give_up(self, why):
        """放弃：停在这里并报警 —— 绝不带着错误状态继续打怪。"""
        self.active = False
        self._scene = None
        self._last_fail, fail = "", self._last_fail
        self._declined = ""
        self.s.enabled = False
        if fail:
            # 把"动作本身为什么没成"一起说出来：只说"重试 3 次"，
            # 人看到的是"它试过了"，而实际上可能是"它一次都没发出键" ✗
            # （`feedback` 回填进来的原因正是这件事 ✓）。
            self._say("重连放弃：%s（最近一次动作失败：%s）（自动保持停止）"
                      % (why, fail))
        else:
            self._say("重连放弃：%s（自动保持停止）" % why)

    def _reset(self):
        self.active = False
        self._scene = None
        self._tries = 0
        self.ui = None
        self._last_fail = ""
        self._declined = ""

    def _say(self, text, now=None):
        """更新状态文字。非空的会在 NOTE_KEEP 秒后过期（见 _decay_note）。

        ⭐ **同时进 `behavior.log`**（2026-10-06 ✓ 交付要求"每一步都要能在 behavior.log
        里复盘"）：状态栏那行是给**当场看**的，事后要复盘只能靠日志 ✓。
        ⚠ **只在文字变了才记一条** ✗：`_say` 每拍都会被调（"排队中…（已等 N 秒）"
        这种一秒一变 ✓）⇒ 不判变化就是拿日志刷屏 ✓（本项目为这个坑专门改过好几处 ✓）。
        """
        self.note = text
        if text:
            self._note_until = ((self._now if now is None else now) + NOTE_KEEP)
        if text and text != self._logged:
            self._logged = text
            behavior.event("reconnect", note=text)
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
