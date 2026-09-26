"""定点上绳（climb）的执行器 —— P4「固定路径执行器」的第一块。

用户 2026-09-26 定的流程（照抄，别自己发明顺序）：

    ① **移动对齐**目标绳梯的 x：偏差 ≤ 「坐标对齐误差范围」，并且**保持**「坐标对齐
       误差时间」那么久，才算对齐成功（两个数都在 设置 → 判定参数 里）；
    ② **按住跳** + 按住 **↑**（向上爬）或 **↓**（向下爬）；
    ③ **判定到达**目标集合的 foothold ⇒ 结束。

**为什么先做「定点上绳」**：它是"贴着绳按上去"，**对起跳时机不敏感** —— 不必先做
跳跃标定（`docs/寻路设计.md` §6 的纪律：标定前不把视觉预测变成按键）。
"移动跳上绳"要在跑动中卡时机，属于标定之后的事；用户也说了最终会让 Agent 在两种
之间**随机挑**，所以这里留一个 `mode` 的位置（现在只有 `"定点"` 一种）。

**为什么这个模块不碰按键、不读画面**：
    · 它只回答"现在该**往哪边挪**、要不要**按跳**、**到了没有**"（`update()` 的回值）；
    · 按键下发仍走 `decision/agent.py` 那条路（`KeyState.set`），画面仍由感知层读；
    · 好处是它**能被自检用假坐标跑完整个流程**（`tools/selftest_route.py`），
      不用开游戏、不用按键 —— 爬绳最容易错的是"什么时候算对齐好了"，那正是它的活。

⚠ 世界坐标可能是**上一帧的**（小地图认不出黄点时沿用，见 `perception/minimap.py`）：
`update()` 收的是**调用方给的值**，"这帧的值新不新"由调用方负责 —— 别让它自己猜
（老代码里 `.held` 的坑就是这么来的）。
"""

MODE_FIXED = "定点"          # 唯一已实现的模式；将来加 "移动" 时在这里多一个取值

# ⚠ 这里原来有个 `ARRIVE_PAD = 14.0`（"离绳顶还差 14 像素就算到达"）——
# **2026-09-26 用户指出：他从没提过这个需求，那个数也没有任何数据支持**，已删除。
# 后果是实测角色停在 -204（平台面是 -208，差 4px 没上去）。判据现在只用**能指出
# 来源的数据**（集合 / 目标平台的面），见 `ClimbJob._arrived`。要再加容差，先说明
# 来历并征得用户同意（见 README「工程约定：不许拍脑袋补数」）。

#: 一次上绳的总超时（秒）。对齐不了 / 卡在绳上不动，都不许无限等（P4 要求"超时报警"）。
DEFAULT_TIMEOUT_S = 12.0

#: 不在绳上多久算「掉下来了」（秒）。**必须留容错期**：按下跳之后人有几拍还在空中，
#: `ladder_id` 那时是 None —— 一离开就判失败会当场误杀。
OFF_LADDER_GRACE_S = 2.0

#: 离绳的 x 还差这么多像素时就改**点按**（2026-09-26 用户要求，先硬编码 20px）。
#: 为什么：按住方向键在游戏里就是"一直走"，快到位时一下就冲过头 ⇒ 来回抖（用户实测）。
NEAR_PX = 20.0

#: 点按的"按住"时长 / 点按周期（秒）。15fps 下 TAP_ON_S=0.06 大约是"按 1 帧、停 2 帧"。
TAP_ON_S = 0.06
TAP_PERIOD_S = 0.18

#: **偏离绳的 x** 多少算偏出去了（像素，硬编码 —— 用户 2026-09-26 要求）。
#: 绳在数据里是"宽度 0"的一条线，`mapdata.LADDER_DX` 是 10 ⇒ 超过 18 基本就是没贴在绳上。
OFF_X_PX = 18.0

#: 偏出去持续多久算失败（秒）。留一点时间：上绳那一下本来就会歪，能自己蹭回来就不算失败。
OFF_X_GRACE_S = 0.6

#: 失败后最多重来几次（含第一次）。**不能无限重来**：数据写错（比如绳的 x 记错了）会
#: 变成"一直重试"，看着像卡死；到上限就放弃并记日志（`perf.count("climb_giveup")`）。
DEFAULT_MAX_ATTEMPTS = 3

#: 下跳：**先按住 ↓ 多久再补跳**（秒）。用户 2026-09-26 说的顺序是"按住 ↓ 再按跳"
#: —— 同一次 tick 里两个键一起发，在游戏里未必算"先↓后跳"，所以分两步发。
ARM_HOLD_S = 0.25

#: 下跳的几何兜底：y 比出发时**增大**这么多就算落地了（y 向下增大）。
#: 只在"落地平台没被圈成集合"时用（没有集合信息也不能认定"永远没到"）。
DROP_DY_PX = 40.0

#: 「爬上去了但**不动了**」的判据（秒）：上绳阶段里 y 连续这么久没有变好 ⇒ 判失败，
#: 并把**两个数**（卡在哪个 y / 离目标面还差多少）写进 note。
#:
#: 为什么必须有（2026-09-26 用户报"角色在 Y=-170 就停了"，而这已经是**第二次**报同一个
#: 现象）：原来的行为是**默默超时** —— 30 秒里一直按住跳键，然后重试 3 次、放弃，界面上
#: 只看到"角色停住"，一个字的线索都没有 ⇒ 只能靠猜（是坐标口径不对？还是这段绳到不了顶？）
#: 而这两种原因的**修法完全相反**。现在让任务自己把话说清楚。
STALL_S = 3.0

#: y 要变好**这么多**才算"还在往上爬"（像素）。防抖：小地图读数本来就在抖几个像素，
#: 拿它当"还在动"会让卡死永远判不出来。
STALL_GAIN_PX = 2.0


class ClimbJob:
    """一次「定点上绳」。`update()` 是纯的：喂当前状态，回下一步该干什么。

    用法（调用方每拍喂一次）：

        job = ClimbJob("L2", x=701.0, y1=-115.0, y2=231.0, direction=+1,
                       dst_set="上下过渡平台", tol_px=6, hold_ms=250)
        out = job.update(now, px=ws.player.x, py=ws.player.y,
                         ladder_id=ws.player.ladder_id,      # 现在在哪根绳上（可 None）
                         here_sets=ws.player.here_sets)      # 脚下属于哪些集合（可 None）
        if out["move"]:  ...按左右...
        if out["jump"]:  ...按住跳 + 按 out["dir"]...

    回值：`{"phase", "move", "jump", "dir", "note", "done", "failed"}`
        move  -1/0/+1 往哪边挪（0 = 站住别动）
        jump  True = **按住**跳键（不是点按）
        dir   -1/0/+1 对应 ↓ / 不按 / ↑（向上爬时是 ↑）
    """

    ALIGN = "align"          # ① 对齐绳的 x
    CLIMB = "climb"          # ② 按跳 + 方向
    DONE = "done"            # ③ 到了
    FAILED = "failed"        # 超时 / 中途掉下来

    def __init__(self, ladder_id, x, y1, y2, direction, dst_set="",
                 dst_y=None,
                 mode=MODE_FIXED, tol_px=6, hold_ms=250,
                 timeout_s=DEFAULT_TIMEOUT_S, max_attempts=DEFAULT_MAX_ATTEMPTS):
        self.mode = mode
        self.ladder_id = str(ladder_id)
        self.x = float(x)
        self.y_top = float(min(y1, y2))
        self.y_bot = float(max(y1, y2))
        self.dir = 1 if direction >= 0 else -1     # +1 向上 / -1 向下
        self.dst_set = str(dst_set or "")
        #: **目标平台那条 foothold 的面**（世界坐标 y；None = 不知道，退回几何兜底）。
        #: 2026-09-26 用户实测：爬上去站着读数是 -209，而绳端连接的那条 foothold 的 y 是
        #: -208 ⇒ **至少 y ≤ -208 就该算到达**。拿"绳的上端 + 容差"（-191）当目标是不对的：
        #: 绳的顶端常常在平台面下面一截 ⇒ "到绳顶了还要求再往上" ⇒ 超时 ⇒ 重试 ⇒ 放弃
        #: （现象就是"爬到某个 y 就不动了"）。
        self.dst_y = None if dst_y is None else float(dst_y)
        self.tol_px = max(1, int(tol_px))
        self.hold_ms = max(0, int(hold_ms))
        self.timeout_s = float(timeout_s)
        self.max_attempts = max(1, int(max_attempts))
        self.attempt = 1                    # 第几次尝试（失败重来会 +1，见 retry）
        self.phase = self.ALIGN
        self._off_x_since = None            # 从哪一刻起横向偏出去了
        self.note = "对齐到 %s 的 x=%.0f（容许 ±%d px，要稳 %d ms）" % (
            self.ladder_id, self.x, self.tol_px, self.hold_ms)
        self._t0 = None
        self._in_tol_since = None
        self._off_since = None             # 从哪一刻起不在绳上（判"掉下来了"，带容错期）
        #: 上绳阶段"爬到过的最好 y" + 它发生的时刻（判"不动了"，见 STALL_S）。
        #: 只在**有 y 读数**时用；`None` = 还没有可用于判断的读数（那就别判，见下面）。
        self._best_y = None
        self._best_at = None
        #: 有没有某一拍读到过"脚下已经是目标集合"。**只做提示/日志**，不作到达依据
        #:（到达是纯几何的，见 `_arrived`）。
        self._saw_dst_set = False
        #: 第一次判到达的时刻 + 那一刻定下的"再按住方向键多久"（毫秒）+ 到达那句话。
        #: 到达≠立刻松手（用户 2026-09-26 要求），见 update 里那段。
        self._arrived_at = None
        self._arrived_hold = 0
        self._arrived_why = ""

    # ---- 对外 ----

    def update(self, now, px, py=None, ladder_id=None, here_sets=None):
        """喂一拍 → 下一步该干什么。`now` 用 `time.monotonic()` 那种秒。

        ⚠ `px` 是**世界坐标 x**（`WorldState.Player.world_x`），不是画面坐标！
        2026-09-26 踩过：调用方一直喂的是画面检测框中心，而 `self.x` 是世界坐标 ⇒
        `dx` 永远是大数 ⇒ 角色朝一个方向一直走、来回抖（用户报的"卡在 -300~-380"）。
        `py` 从头到尾就是用 `world_y` 的 —— 一个画面一个世界，本身就是自相矛盾的。
        """
        if self._t0 is None:
            self._t0 = now
        if self.phase in (self.DONE, self.FAILED):
            return self._out(0, False)
        if now - self._t0 > self.timeout_s:
            return self._fail("超时 %.0fs：没能在 %s 上完成上绳" % (
                self.timeout_s, self.ladder_id))

        if self.phase == self.ALIGN:
            dx = self.x - float(px)
            if abs(dx) > self.tol_px:
                self._in_tol_since = None
                sign = 1 if dx > 0 else -1
                if abs(dx) <= NEAR_PX:
                    # **快到了 ⇒ 一下一下地点按**（用户 2026-09-26 要求 2）：
                    # 按住方向键 = 一直走，差二十几像素那一下必然冲过头、然后往回走，
                    # 表现就是"在绳两边来回抖"。这里按 TAP_PERIOD_S 的周期只按住
                    # TAP_ON_S（≈ 一帧），其余那几拍**什么都不按**（松开 = 站住）。
                    if (float(now) % TAP_PERIOD_S) < TAP_ON_S:
                        self.note = "微调：差 %.0f px（点按 %s）" % (
                            dx, "→" if sign > 0 else "←")
                        return self._out(sign, False)
                    self.note = "微调：差 %.0f px（松开这一拍，别冲过头）" % dx
                    return self._out(0, False)
                self.note = "对齐中：差 %.0f px（容许 %d）" % (dx, self.tol_px)
                return self._out(sign, False)
            if self._in_tol_since is None:
                self._in_tol_since = now
                self.note = "进误差范围了（差 %.0f px）—— 站住等 %d ms" % (
                    dx, self.hold_ms)
                if self.hold_ms > 0:
                    return self._out(0, False)      # 要等就这一拍先站住
                # 不要等（hold=0）⇒ **同一拍就上绳**，别白拖一拍
            if (now - self._in_tol_since) * 1000.0 < self.hold_ms:
                return self._out(0, False)          # 保持中，别动（动就把对齐弄丢了）
            self.phase = self.CLIMB
            self.note = ("对齐好了 ⇒ 按住跳贴上去，上去之后只按 %s"
                         % ("↑" if self.dir > 0 else "↓"))

        # ---- ② 上绳 ----
        # **偏离保护**（用户 2026-09-26 要求）：爬绳时横向偏出去（|px − 绳.x| > OFF_X_PX）
        # 并**持续** OFF_X_GRACE_S ⇒ 判失败，交给上层"重新激活"（见 retry / agent）。
        # 为什么留宽限期：上绳那一下本来就会歪一点，能自己蹭回来就不该算失败。
        dx_off = abs(float(px) - self.x)
        if dx_off > OFF_X_PX:
            if self._off_x_since is None:
                self._off_x_since = now
            elif (now - self._off_x_since) >= OFF_X_GRACE_S:
                return self._fail("偏离绳梯：横向差 %.0f px（阈值 %.0f）已 %.1fs"
                                  % (dx_off, OFF_X_PX, OFF_X_GRACE_S))
        else:
            self._off_x_since = None
        # ---- 到达之后：**再按住方向键一会儿才松**（用户 2026-09-26 要求：
        #      "↑ 需要延迟『坐标对齐误差时间』（设置里那个）再松开"）----
        # 为什么：读数到目标面 ≠ 人已经稳稳站上去 —— 读数本身滞后一个端到端延迟，而游戏里
        # "迈上平台"那一步也要按着 ↑ 才走得完 ⇒ 立刻松手就是**差最后一点点**（"在 -170 /
        # -208 附近停下"里就有这一份）。等待时长 = 和"上绳前"同一个保持窗口（设置里的误差
        # 时间 + 当前端到端延迟，由 `agent._climb_tick` 写进 `hold_ms`），**判到到达那一拍
        # 就定死**，免得等待期间 e2e 抖动让目标时刻一直往后挪。
        # ⚠ 这段放在 `_arrived` **之前**：一旦判到到达，等待就自己走完 —— 中途定位掉一拍
        #（py=None）不该把"再按住一会儿"打断 ✗（那时人其实已经站在上面了）。
        if self._arrived_at is not None:
            if (now - self._arrived_at) * 1000.0 < self._arrived_hold:
                return self._out(0, False, hold_dir=True)
            self.phase = self.DONE
            self.note = self._arrived_why
            return self._out(0, False)
        why = self._arrived(py, here_sets)
        if why:
            self._arrived_at = now
            self._arrived_why = why
            # ⚠ 不能 `max(1, …)`：误差时间设成 0 = **要立刻松手**（老行为），
            # 硬留 1 ms 会让"这一拍就该收工"的用例（和用户自己设 0 的情况）卡住 ✗。
            self._arrived_hold = max(0, int(self.hold_ms))
            if self._arrived_hold <= 0:
                self.phase = self.DONE
                self.note = why
                return self._out(0, False)
            self.note = ("%s（再按住 %s %d ms 再松）"
                         % (why, "↑" if self.dir > 0 else "↓", self._arrived_hold))
            return self._out(0, False, hold_dir=True)
        # **爬不动了就别再默默按住跳键**（用户 2026-09-26 第二次报"卡在某个 y"）：
        # y 连续 STALL_S 秒没变好 ⇒ 判失败，并把两个数说清楚 —— 因为"读数与目标面差一截"
        # 和"这段绳到不了顶"这两种原因的修法**正好相反**，必须让人一眼看出是哪种。
        stall = self._stalled_by(now, py)
        if stall:
            return self._fail(stall)
        # 还没上绳 / 中途掉下来：**给一段容错期**再判失败 —— 按跳那几拍人本来就还在
        # 空中（`ladder_id` 是 None），一离开就判失败会当场误杀。
        if ladder_id is not None and str(ladder_id) == self.ladder_id:
            self._off_since = None
            # **已经上绳 ⇒ 只按方向键（↑），不再按跳**（用户 2026-09-26 定的规则：
            # "按 ↑ 直到玩家的真实世界坐标 ≤ 绳梯上端连接的 foothold 的 y"）。
            # 跳键**只用来贴上绳**：一直按住它，人会卡在绳上/绳顶不往上走
            #（用户实测：命令前往左上平台时"还在绳上、离平台还差一截"）。
            # 注意这里要 `hold_dir`：不按跳也得把 ↑ 按住，否则这一拍什么都不按=站着不动。
            self.note = "已上绳：按住 %s 往上爬（到 y ≤ %.0f 算到）" % (
                "↑" if self.dir > 0 else "↓",
                self.dst_y if self.dst_y is not None else
                (self.y_top if self.dir > 0 else self.y_bot))
            return self._out(0, False, hold_dir=True)
        if self._off_since is None:
            self._off_since = now
        elif (now - self._off_since) >= OFF_LADDER_GRACE_S:
            return self._fail("从 %s 上掉下来了（%.0fs 没回到绳上）" % (
                self.ladder_id, OFF_LADDER_GRACE_S))
        return self._out(0, True)

    def retry(self):
        """**失败后重新激活**（用户 2026-09-26：「失败触发后重新激活就行」）。

        回到第一步"重新对齐"，把超时 / 保持 / 偏离 / 离绳那几个计时全清掉，`attempt` +1。
        超上限由上层放弃（`agent._climb_tick`）—— 不然数据写错时会一直重来，看着像卡死。
        """
        self.attempt += 1
        self.phase = self.ALIGN
        self._t0 = None
        self._in_tol_since = None
        self._off_since = None
        self._off_x_since = None
        # "爬到过的最好 y"也要清：重新对齐可能把起点挪了，留着上一次的极值会让
        # "不动了"判据一上来就误判（那是上一次尝试的读数）。
        self._best_y = None
        self._best_at = None
        # 到达后的"再按住一会儿"同理要清：留着上一次的时刻会让新一轮一上来就收工 ✗。
        self._arrived_at = None
        self._arrived_hold = 0
        self._arrived_why = ""
        self.note = "第 %d/%d 次尝试：重新对齐 %s" % (self.attempt,
                                                    self.max_attempts,
                                                    self.ladder_id)
        return self._out(0, False)

    def cancel(self, why="取消"):
        """外部叫停（比如切了目标/手动关自动）—— 状态收干净，别再按键。"""
        if self.phase not in (self.DONE, self.FAILED):
            self.phase = self.FAILED
            self.note = str(why)
        return self._out(0, False)

    # ---- 内部 ----

    def _stalled_by(self, now, py):
        """爬到一半**不动了**？→ 一句说清原因的失败话（没卡住给空串）。

        只有**有 y 读数**时才判 —— 定位没输出的时候（`py is None`）"没动"是**观测**问题，
        不是"爬不动"，那属于 `agent._climb_tick` 里"拿不到世界坐标"那条路，别在这里乱说。

        话里必须同时给出**卡住的 y**和**离目标面还差多少** —— 这两个数直接指向两种相反的
        原因（读数口径差一截 / 这段绳到不了顶），是这道判据存在的全部意义。
        """
        if py is None or self.dst_y is None:
            return ""
        pyv = float(py)
        better = (pyv < self._best_y - STALL_GAIN_PX) if (self._best_y is not None
                                                          and self.dir > 0) else (
            pyv > (self._best_y or 0.0) + STALL_GAIN_PX if self._best_y is not None
            else True)
        if better or self._best_y is None:
            self._best_y, self._best_at = pyv, now
            return ""
        if self._best_at is None or (now - self._best_at) < STALL_S:
            return ""
        return ("爬到 y=%.0f 就不再%s了（%.1fs 没变好）：离目标面 %.0f 还差 %.0f px。"
                "先核对「小地图 → 坐标系偏移」（读数加过它，判据用的是地形原始坐标），"
                "再看这截绳到不到得了平台"
                % (self._best_y, "升" if self.dir > 0 else "降", STALL_S, self.dst_y,
                   abs(self._best_y - self.dst_y)))

    def _arrived(self, py, here_sets):
        """到了没有 —— **纯几何**，判据就是用户定的那条（2026-09-26）：

            「按 ↑ 直到玩家的**真实世界坐标** ≤ **绳梯上端连接的 foothold 的 y**」

        即 `dst_y`（那条 foothold 的面，已换算到真值坐标系）+ 没有它时退回"越过绳端"。

        ⚠ **这里不看 `here_sets` 了**（原来先查集合 ✗）：集合是"脚踩在哪条 foothold"，
        而定位读数会抖动 / 有几拍的滞留值，`-170`（离目标还差 5px）那种位置也可能被
        报成"已在目标集合" ⇒ 提前收工（用户实测：**在 -170 就松开了 ↑** ✗）。
        集合判据没丢，它挪到了它该在的地方：**失败之后 "已经到了就别再重试"**
        （见 `agent._climb_tick`）—— 那才是用户当初要它的意思（"结束重新激活的任务"）。
        """
        if here_sets is not None and self.dst_set and self.dst_set in set(here_sets):
            # 只记一笔给日志/提示用，**不作为到达依据**（到达只认下面的几何判据）
            self._saw_dst_set = True
        # 判据①（比"绳端"准，2026-09-26 用户要求）：**够到目标平台的面**就算到。
        # 用户实测：爬上去站着读数是 -209，而绳端连接的那条 foothold 的 y 是 -208
        # ⇒ 至少 `y ≤ -208` 就该判到达。**不加容差**：平台面就是"站上去的高度"，
        # 加 14px 等于"还没踩上去就算到了"（那正是老写法"到绳顶还要求再往上"的反面）。
        if self.dst_y is not None:
            if py is not None:
                pyv = float(py)
                if self.dir > 0 and pyv <= self.dst_y:
                    return "到达：y=%.0f 已到目标平台的面（%.0f）" % (pyv, self.dst_y)
                if self.dir < 0 and pyv >= self.dst_y:
                    return "到达：y=%.0f 已到目标平台的面（%.0f）" % (pyv, self.dst_y)
            # **有平台面就不再拿"绳端"当目标**：绳顶常常比平台面低一截，拿它当目标是错的
            #（用户 2026-09-26 定的口径：至少 `y ≤ dst_y` 才算到）。
            return ""
        # ⚠ 方向别搞反：y 向下增大（`y_top` 是 min）。向上爬 ⇒ y **减小到**绳上端附近；
        # 所以判据是"够到绳端 ± 容差"，不是"越过绳端之外"（第一版写成 ± 反了，
        # 兜底判据就**永远不会触发** —— 人到了绳顶还一直按着跳）。
        if py is not None:
            pyv = float(py)
            # 够到绳的**那一端**（不加容差）：这条只在**算不出目标平台面**时兜底
            #（比如那条边的目标集合还没圈 foothold）。**不许再加 pad** —— 见文件头
            # 关于 ARRIVE_PAD 的说明与 README 的「不许拍脑袋补数」。
            if self.dir > 0 and pyv <= self.y_top:
                return "到达：y=%.0f 已到 %s 上端（%.0f）" % (pyv, self.ladder_id,
                                                              self.y_top)
            if self.dir < 0 and pyv >= self.y_bot:
                return "到达：y=%.0f 已到 %s 下端（%.0f）" % (pyv, self.ladder_id,
                                                              self.y_bot)
        return ""

    def _fail(self, why):
        self.phase = self.FAILED
        self.note = why
        return self._out(0, False)

    def _out(self, move, jump, hold_dir=False):
        """`hold_dir`：**不按跳也要按住方向键** —— 下跳要求"先按住 ↓，再按跳"。

        （爬绳用不到它：那边只有"按住跳 + ↑/↓"一步。）
        """
        return {"phase": self.phase, "move": int(move), "jump": bool(jump),
                "dir": (self.dir if (jump or hold_dir) else 0), "note": self.note,
                "done": self.phase == self.DONE,
                "failed": self.phase == self.FAILED}


class DropJob:
    """一次「下跳」（`drop`）：对齐到某个**可下跳 foothold** → 按住 ↓ 再按跳 → 落地。

    用户 2026-09-26 定的动作：**按住 ↓ 再按跳**（和「跳(jump)」= 在 foothold 边缘按跳
    是两回事，别混）。可下跳的 foothold 来自边上的 `footholds` 那一格
    （不填 = 起点集合的全部，见 `zones.drop_footholds`）—— 一条边上常常有好几个能下去
    的点，**就近挑一个**，省得为了那一个点横穿平台。

    接口和 `ClimbJob` **故意做成一样**（`update/retry/cancel/attempt/max_attempts/phase`）：
    agent 那边就只有一条执行路径（`_climb_tick`），不用为每种通行方式各写一遍
    —— 也保证两种的失败保护/重来行为一致。
    """

    ALIGN = "align"          # ① 就近平齐到某个可下跳 foothold
    ARMED = "armed"          # ② **先按住 ↓**（这一刻还不跳）
    DROP = "drop"            # ③ ↓ 按住不放 + 按跳
    DONE = "done"
    FAILED = "failed"

    def __init__(self, src_set, dst_set, spots, tol_px=6, hold_ms=250,
                 timeout_s=DEFAULT_TIMEOUT_S, max_attempts=DEFAULT_MAX_ATTEMPTS):
        self.src_set = str(src_set or "")
        self.dst_set = str(dst_set or "")
        #: [(中心 x, 左, 右, foothold id)] —— 可下跳的那几条（就近挑）
        self.spots = [(float(c), float(a), float(b), str(i))
                      for c, a, b, i in (spots or [])]
        if not self.spots:
            raise ValueError("下跳任务没有可下跳的 foothold")
        self.tol_px = max(1, int(tol_px))
        self.hold_ms = max(0, int(hold_ms))
        self.timeout_s = float(timeout_s)
        self.max_attempts = max(1, int(max_attempts))
        self.attempt = 1
        self.dir = -1                       # 下跳永远是"往下"（按住 ↓）
        self.phase = self.ALIGN
        self.note = "就近平齐到一个可下跳 foothold"
        self._t0 = None
        self._in_tol_since = None
        self._armed_at = None
        self._y0 = None
        self._pick = None

    def update(self, now, px, py=None, ladder_id=None, here_sets=None):
        if self._t0 is None:
            self._t0 = now
        if self.phase in (self.DONE, self.FAILED):
            return self._out(0, False)
        if now - self._t0 > self.timeout_s:
            return self._fail("超时 %.0fs：没能从「%s」下跳到「%s」"
                              % (self.timeout_s, self.src_set, self.dst_set))

        if self.phase == self.ALIGN:
            if self._pick is None:
                self._pick = min(self.spots, key=lambda s: abs(s[0] - float(px)))
                self.note = "就近平齐到 fh %s（x=%.0f）" % (self._pick[3], self._pick[0])
            dx = self._pick[0] - float(px)
            if abs(dx) > self.tol_px:
                self._in_tol_since = None
                self.note = "对齐中：差 %.0f px（容许 %d）" % (dx, self.tol_px)
                return self._out(1 if dx > 0 else -1, False)
            if self._in_tol_since is None:
                self._in_tol_since = now
                self.note = "进误差范围了（差 %.0f px）—— 站住等 %d ms" % (
                    dx, self.hold_ms)
                if self.hold_ms > 0:
                    return self._out(0, False)
            if (now - self._in_tol_since) * 1000.0 < self.hold_ms:
                return self._out(0, False)
            self.phase = self.ARMED
            self._armed_at = now
            self._y0 = float(py) if py is not None else None
            self.note = "按住 ↓（%d ms）再按跳" % int(ARM_HOLD_S * 1000)

        if self.phase == self.ARMED:
            # **先按住 ↓ 一小会儿**（用户说的顺序：按住↓ → 按跳）。
            # 这一刻只发方向键、不发跳（agent 那边靠 `hold_dir` 知道该怎么办）。
            #
            # ⚠ 出发 y 要**一直更新到起跳前**：世界坐标不是每拍都有（认不出黄点时是 None），
            # 只在"进 ARMED 那一拍"记一次的话，那拍没坐标 ⇒ 出发 y 是 None ⇒ 落地兜底
            # 判据（y 掉下去一段）**永远不会成立**（踩过）。
            if py is not None:
                self._y0 = float(py)
            if (now - self._armed_at) >= ARM_HOLD_S:
                self.phase = self.DROP
                self.note = "按住 ↓ + 跳"
            else:
                return self._out(0, False, hold_dir=True)

        # ---- ③ 下跳中：到了没有 ----
        why = self._arrived(py, here_sets)
        if why:
            self.phase = self.DONE
            self.note = why
            return self._out(0, False)
        return self._out(0, True, hold_dir=True)

    def retry(self):
        """失败后**重新激活**：回到对齐那一步重来（和 ClimbJob 同一套保护）。"""
        self.attempt += 1
        self.phase = self.ALIGN
        self._t0 = None
        self._in_tol_since = None
        self._armed_at = None
        self._y0 = None
        self.note = "第 %d/%d 次尝试：重新对齐（下跳）" % (self.attempt,
                                                        self.max_attempts)
        return self._out(0, False)

    def cancel(self, why="取消"):
        if self.phase not in (self.DONE, self.FAILED):
            self.phase = self.FAILED
            self.note = str(why)
        return self._out(0, False)

    def _arrived(self, py, here_sets):
        """到了没有：**先集合后几何**（和下跳那边同一套口径）。

        几何兜底是"y 比出发时**大了**一段"（y 向下增大 ⇒ 往下掉了）—— 落地平台要是
        没被圈成集合，就靠它兜住，不能因此认定"永远没到"。
        """
        if here_sets and self.dst_set and self.dst_set in set(here_sets):
            return "到达：脚下已经是「%s」" % self.dst_set
        if py is not None and self._y0 is not None:
            if float(py) >= self._y0 + DROP_DY_PX:
                return "到达：y 从 %.0f 掉到 %.0f（≥%.0f px）" % (self._y0, float(py),
                                                                DROP_DY_PX)
        return ""

    def _fail(self, why):
        self.phase = self.FAILED
        self.note = why
        return self._out(0, False)

    def _out(self, move, jump, hold_dir=False):
        return {"phase": self.phase, "move": int(move), "jump": bool(jump),
                "dir": (self.dir if (jump or hold_dir) else 0), "note": self.note,
                "done": self.phase == self.DONE,
                "failed": self.phase == self.FAILED}


def drop_job_for_edge(terrain, z, edge, **kw):
    """从一条「下跳」边造出任务：可下跳 foothold 取边上的那一格（不填 = 起点集合的全部）。

    抛 ValueError 的两种情况：一个可下跳的都没有、或者那些 id 在本图里找不到
    （换了地图/地形重导过）—— **不许猜**，让上层报警或跳过。

    坐标口径同 `job_for_edge`：地形数据本来就在世界坐标系里，玩家读数与它**直接比** ✓。
    """
    from core import zones

    if (edge.get("kind") or "") != "drop":
        raise ValueError("不是「下跳」边：%s" % edge.get("kind"))
    ids = zones.drop_footholds(z, edge)
    if not ids:
        raise ValueError("这条下跳边一个可下跳 foothold 都没有：%s → %s"
                         % (edge.get("from"), edge.get("to")))
    known = {str(f.fid): f for f in terrain.footholds}
    spots = []
    for fid in ids:
        f = known.get(str(fid))
        if f is None or f.is_wall:
            continue
        spots.append(((f.left + f.right) / 2.0, f.left, f.right, str(fid)))
    if not spots:
        raise ValueError("可下跳 foothold 在本图里都找不到（或都是墙）：%s" % (ids,))
    return DropJob(edge.get("from"), edge.get("to"), spots, **kw)


#: 走（walk）的到达容差：与目标 foothold 的 x 范围差 ≤ 它就算到（像素）。
#: 为什么给容差：走是"走到那块**地面**上"，不是"走到某个像素" —— 人站上那条 foothold
#: 的范围就该算到（集合判据优先，见 `WalkJob._arrived`）。
WALK_TOL_PX = 12.0

#: 几何兜底判据允许的高度差（像素）：站上去以后 y 和那条 foothold 的面差这么多以内
#: 才算"站在它上面"（防止 x 对了、其实人在上一层/下一层的平台上）。
WALK_Y_BAND_PX = 30.0


class WalkJob:
    """一次「走过去」：沿地面走到目标集合的某条 foothold 上。

    对外形状和 `ClimbJob` / `DropJob` **完全一样**（`update()` 是纯函数 + `retry()` /
    `cancel()`）⇒ `agent._climb_tick` 里那套"任务优先 / 有怪先打 / 失败延迟重来 /
    结束原因挂在画面上"**一行都不用改**就能用（2026-09-26 用户要求"把 walk 的执行器做了"）。

    到达判据（**先集合后几何**，和别的任务同一口径）：
      · 脚下集合已经有目标集合 ⇒ 到了（人确实站在那块地上了 ✓ 走是唯一"站上去就算到"
        的任务 —— 上绳不行，因为定位抖一下会把差 38px 的地方报成"已在集合"，见
        `ClimbJob._arrived` 的说明）；
      · 否则几何兜底：站在目标集合某条 foothold 的 x 范围 ±`WALK_TOL_PX`、且 y 和它的面
        差 ≤`WALK_Y_BAND_PX`。

    动作：`move` 给 -1/0/+1（左右），**不按跳、不按上下** —— 走就是走；差得远按住走，
    快到了（≤`NEAR_PX`）改成点按，别冲过头（和上绳对齐同一个做法）。
    卡住（朝目标那侧的距离 `stall_s` 秒没变好）⇒ 失败并说清卡在哪个 x（被墙/台阶挡住
    时人要看得懂，而不是干等到超时 ✗）。
    """

    WALK = "walk"
    DONE = "done"
    FAILED = "failed"

    # ⚠ 这里**曾经**有 `MODE_CENTER / MODE_LEFT / MODE_RIGHT` 和一个 `mode` 参数
    #（还摆进了设置里的「走的方向」下拉）。2026-09-26 用户明确否掉：
    # **走只有一种走法 —— 朝目标集合的 x 中点**，设置里不许有这个参数。
    # 将来若某一步真需要"只按 ← 或 →"，那是**逐边**的事：在
    # foothold 编辑器 →「可到达」窗口里配，而不是全局设置。

    def __init__(self, dst_set, spots, tol_px=WALK_TOL_PX, hold_ms=None,
                 timeout_s=DEFAULT_TIMEOUT_S, max_attempts=DEFAULT_MAX_ATTEMPTS,
                 stall_s=STALL_S):
        #: ⚠ `hold_ms` 收下但**不用**：走没有"到达后再按住一会儿"这回事（那是上绳需要的，
        #: 因为读数滞后 + 游戏里迈上平台那一步要按着 ↑）。之所以要有这个参数：三个任务
        #: 走的是**同一个调用入口**（`route_panel._command_first_step` 会把对齐参数一起
        #: 传进来），少一个参数就是 `TypeError` —— 实测就是这么炸的 ✗。
        #: 目标集合名（`agent.current_goto_set()` 用它显示"前往：X"）。
        self.dst_set = str(dst_set or "")
        #: 候选落点 `[(中心x, 左, 右, 面y, foothold id)]` —— 由 `walk_job_for_edge` 给。
        self.spots = [tuple(s) for s in spots]
        self.tol_px = max(1.0, float(tol_px))
        self.timeout_s = float(timeout_s)
        self.max_attempts = max(1, int(max_attempts))
        self.stall_s = max(0.5, float(stall_s))
        self.attempt = 1
        self.phase = self.WALK
        self.note = "走过去：「%s」（%d 个落点，就近）" % (self.dst_set, len(self.spots))
        self._t0 = None
        self._best_d = None            # 离"最近那个目标点"最好走到过多近
        self._best_at = None

    # ---- 对外 ----

    def update(self, now, px, py=None, ladder_id=None, here_sets=None):
        """喂一拍 → 下一步该干什么。

        ⚠ `px` 必须是**世界坐标 x**（`WorldState.Player.world_x`），不是画面坐标
        —— 和 `ClimbJob.update` 同一个坑（喂错了会往反方向走、来回抖）。
        `ladder_id` 收下但不用（走不看绳）。
        """
        if self._t0 is None:
            self._t0 = now
        if self.phase in (self.DONE, self.FAILED):
            return self._out(0)
        if now - self._t0 > self.timeout_s:
            return self._fail("超时 %.0fs：没能走到「%s」" % (self.timeout_s,
                                                          self.dst_set))
        why = self._arrived(px, py, here_sets)
        if why:
            self.phase = self.DONE
            self.note = why
            return self._out(0)
        if px is None:
            # 定不了位就别乱走（往哪边走都可能是错的）—— 如实说，让人去查小地图
            self.note = "走出去：拿不到世界坐标（小地图定位没有输出）"
            return self._out(0)
        # 目标 x = 集合的**中点**（所有落点 x 的中点）—— 走只有这一种走法（2026-09-26 用户定；
        # "仅向左 / 仅向右"那个选项已删，见 `__init__` 上面的说明）。
        target_x = (min(float(s[0]) for s in self.spots)
                    + max(float(s[0]) for s in self.spots)) / 2.0
        #: 提示里那句话（写清"目标是集合中点"）
        tgt_txt = "集合中心 x=%.0f" % target_x
        dx = target_x - float(px)
        d = abs(dx)
        # **卡住检测**：朝目标那侧的距离一直不变好 ⇒ 被挡住了（墙/台阶/逆风），
        # 报出来比干等到超时有用得多（上绳那边同一套判据，见 STALL_S）。
        # ⚠ 进度用**到"集合区间"的距离**，不用上面那个 `d`（`d` 是到"中点"的距离）：
        # 站在中点附近、还没踏进集合时它会一直不变 ⇒ 走得好好的也会被判"走不动" ✗
        # 到区间的距离才是单调下降的 ✓（这里原本还替"仅向左/仅向右"兜了一道，
        # 那个模式 2026-09-26 已按用户要求删掉，但这条判据本身仍然更对）。
        _lo = min(float(s[1]) for s in self.spots)
        _hi = max(float(s[2]) for s in self.spots)
        _prog = (0.0 if _lo <= float(px) <= _hi
                 else min(abs(float(px) - _lo), abs(float(px) - _hi)))
        if self._best_d is None or _prog < self._best_d - 1.0:
            self._best_d, self._best_at = _prog, now
        elif self._best_at is not None and (now - self._best_at) >= self.stall_s:
            return self._fail("走不动了：卡在 x=%.0f（目标在%s，还差 %.0f px）"
                              "—— 可能被墙/台阶挡住，或者这条边的落点/方向写错了"
                              % (float(px), tgt_txt, d))
        if d <= self.tol_px:
            self.note = "走到落点附近了（差 %.0f px）—— 站住，等判到达" % d
            return self._out(0)
        sign = 1 if dx > 0 else -1
        if d <= NEAR_PX:
            # 快到了 ⇒ **点按**，别冲过头（和上绳对齐同一个做法，见 ClimbJob）
            if (float(now) % TAP_PERIOD_S) < TAP_ON_S:
                self.note = "微调：差 %.0f px（点按 %s）" % (d, "→" if sign > 0 else "←")
                return self._out(sign)
            self.note = "微调：差 %.0f px（松开这一拍，别冲过头）" % d
            return self._out(0)
        self.note = "走过去：目标在%s，还差 %.0f px（%s）" % (
            tgt_txt, d, "→" if sign > 0 else "←")
        return self._out(sign)

    def retry(self):
        """失败后**重新激活**（和上绳/下跳同一套：回到起始状态重来一遍）。"""
        self.attempt += 1
        self.phase = self.WALK
        self._t0 = None
        self._best_d = None
        self._best_at = None
        self.note = "第 %d/%d 次尝试：重新走过去「%s」" % (self.attempt,
                                                       self.max_attempts,
                                                       self.dst_set)
        return self._out(0)

    def cancel(self, why="取消"):
        if self.phase not in (self.DONE, self.FAILED):
            self.phase = self.FAILED
            self.note = str(why)
        return self._out(0)

    # ---- 内部 ----

    def _arrived(self, px, py, here_sets):
        """到了没有（先集合后几何 —— 走是唯一"站上去就算到"的任务，理由见类注释）。"""
        if here_sets and self.dst_set and self.dst_set in set(here_sets):
            return "到达：脚下已经是「%s」" % self.dst_set
        if px is None or py is None:
            # ⚠ 几何兜底**必须有 y 读数**：只看 x 会误判"换层"（同一 x 上下两层平台很常见，
            # 实测被自己的用例逮到过：x 对了、人在上一层也判到达 ✗）。没有 y 就等下一拍
            #（定位每拍都在给；一直给不出来会走到超时并如实报警）。
            return ""
        for _xc, left, right, y, _fid in self.spots:
            if (left - self.tol_px) <= float(px) <= (right + self.tol_px):
                if abs(float(py) - float(y)) <= WALK_Y_BAND_PX:
                    return ("到达：走到「%s」那条 foothold 上"
                            "（x=%.0f，面 y=%.0f）" % (self.dst_set, float(px), float(y)))
        return ""

    def _fail(self, why):
        self.phase = self.FAILED
        self.note = why
        return self._out(0)

    def _out(self, move):
        """走不按跳、也不按上下（`dir` 恒为 0）—— 和另外两个任务的回值形状一致。"""
        return {"phase": self.phase, "move": int(move), "jump": False, "dir": 0,
                "note": self.note, "done": self.phase == self.DONE,
                "failed": self.phase == self.FAILED}


def walk_job_for_edge(terrain, z, edge, **kw):
    """从一条「走」边造出任务：落点 = 目标集合里的非墙 foothold（**不许猜**）。

    抛 ValueError 的两种情况：目标集合里一条 foothold 都没有、或者那些 id 在本图里
    找不到（换了地图/地形重导过）—— 让上层如实报警，别下一条"不知道往哪走"的命令 ✗。
    """
    from core import zones                                          # noqa: F401

    if (edge.get("kind") or "") != "walk":
        raise ValueError("不是「走」边：%s" % edge.get("kind"))
    ids = set(str(v) for v in
              ((z.sets.get(edge.get("to")) or {}).get("footholds") or []))
    if not ids:
        raise ValueError("目标集合「%s」里一条 foothold 都没有：%s → %s"
                         % (edge.get("to"), edge.get("from"), edge.get("to")))
    spots = []
    for f in terrain.footholds:
        if f.is_wall or str(f.fid) not in ids:
            continue
        spots.append(((f.left + f.right) / 2.0, float(f.left), float(f.right),
                      float(f.y_at((f.left + f.right) / 2.0)), str(f.fid)))
    if not spots:
        raise ValueError("目标集合「%s」的 foothold 在本图里都找不到（或都是墙）：%s"
                         % (edge.get("to"), sorted(ids)))
    return WalkJob(edge.get("to"), spots, **kw)


def plan_jobs(terrain, z, src, dst, tol_px=6, hold_ms=250):
    """「从 `src` 走到 `dst`」→ `{"path", "jobs", "why", "here"}`（**一处实现，两边共用**）。

    谁在用：
      · `gui/route_panel._command_first_step`（「命令前往」下整条路径）；
      · `gui/live_thread` 给 agent 注入的**路径解析器**（「定点休息」要 agent 自己走过去）。
    ⚠ 两边**必须同一套判据** —— 各写一份迟早分叉（出现"面板能下发、休息却走不过去"
    这种最难查的怪事 ✗），所以判据只放这里。

    判据（就是「命令前往」原来那套 ✓）：
      · `zones.find_path` 找不到路 ⇒ `jobs=[]`，`why` 说清（它会把"从起点能走到的全部
        集合"列出来 —— 那就是缺边的位置，比一句"没有路径"有用得多）；
      · 路径里出现「跳 / 传送门」⇒ **整条都不造**并说清是第几步、靠什么
       （半途停在半空中比不下更糟 ✗）；
      · 任何一段造任务抛 `ValueError`（绳找不到 / 说不清上下 / 落点找不到）⇒ **抛给
        上层**，让它如实报警或跳过，**不许猜**。
    `src == dst` ⇒ `here=True`（人已经在目标上，不用走 ✓）。
    返回的 `jobs` 直接喂 `agent.start_route`。
    """
    from core import zones as zones_mod

    src, dst = str(src or ""), str(dst or "")
    path, why = zones_mod.find_path(z, src, dst)
    if not path:
        return {"path": [], "jobs": [], "why": why, "here": False}
    if len(path) < 2:
        return {"path": path, "jobs": [],
                "why": "已经在「%s」上了，不用走" % dst, "here": True}
    kinds = []
    for i, (a, b) in enumerate(zip(path, path[1:]), 1):
        e = zones_mod.edge_between(z, a, b)
        if e is None:
            return {"path": path, "jobs": [],
                    "why": "%s → %s 之间找不到那条可达（回「编辑集合…」看看）" % (a, b),
                    "here": False}
        k = e.get("kind") or ""
        if k not in ("walk", "climb", "drop"):
            return {"path": path, "jobs": [],
                    "why": ("第 %d 步 %s → %s 靠「%s」过去 —— 这种执行器还没做，"
                            "现在只做到「走」「爬」「下跳」"
                            % (i, a, b, zones_mod.kind_label(k))),
                    "here": False}
        kinds.append(e)
    jobs = [job_for_edge(terrain, z, e, tol_px=int(tol_px), hold_ms=int(hold_ms))
            for e in kinds]
    return {"path": path, "jobs": jobs, "why": "", "here": False}


def job_for_edge(terrain, z, edge, **kw):
    """从一条「爬」边造出上绳任务 —— **上下由目标那边在绳的上端还是下端判**。

    这条把三样东西接起来（都走现成设施，不自己发明）：
        · `zones.ladder_ids` 找绳（与编辑器画在绳上的编号同一套）；
        · `zones.climb_direction` 判上下（用户 2026-09-26 的规则）；
        · `ClimbJob` 执行。
    判不出来（绳找不到 / 目标不在绳的两端）就抛 ValueError —— 调用方要么报警要么跳过，
    **不许猜**（猜错就是"往反方向爬"，比不做更糟）。

    ⚠ **坐标口径**（2026-09-26 用户澄清，这里曾经改错过一次）：`L.x / L.y1 / L.y2 /
    dst_y` 全来自地形数据，而**地形数据本来就在游戏的世界坐标系里**（例：105090600 的
    fh44 地形 y=-208，它的世界 y 就是 -208）⇒ 玩家读数（= 定位 + 那个"坐标系偏移" ✓）
    与它们**直接比大小**就是对的 ✓。**不要**给这些数再加偏移 —— 加了会把"到达目标平台面"
    从 -208 放宽成 -175（差 33px），看起来像"提前算到了" ✗。
    """
    from core import zones

    kind = edge.get("kind") or ""
    if kind == "drop":
        return drop_job_for_edge(terrain, z, edge, **kw)
    if kind == "walk":
        return walk_job_for_edge(terrain, z, edge, **kw)
    if kind != "climb":
        raise ValueError("这个模块只认「走」「爬」「下跳」，收到：%s" % kind)
    lids = zones.ladder_ids(terrain)
    want = str(edge.get("ladder") or "")
    L = next((x for x in terrain.ladders if lids.get(id(x)) == want), None)
    if L is None:
        raise ValueError("这张图上找不到绳 %s" % (want or "(空)"))
    d = zones.climb_direction(terrain, z, L, edge.get("from"), edge.get("to"))
    if d is None:
        raise ValueError(
            "说不清 %s →(爬 %s)→ %s 是向上还是向下"
            "（终点或起点不在绳的两端，或者一头圈进了两个集合）—— 回编辑器看一下。"
            % (edge.get("from"), want, edge.get("to")))
    dst_y = dst_surface_y(terrain, z, edge.get("to"), L, d)
    return ClimbJob(want, L.x, L.y1, L.y2, d, dst_set=edge.get("to"),
                    dst_y=dst_y, **kw)


def dst_surface_y(terrain, z, dst_set, L, direction=1):
    """目标集合里**爬上去会踩上的那一层**的面（y）——"到哪儿算到了"。

    挑选（**按方向**，2026-09-26 用户实测定的口径）：
      · 向上爬 ⇒ 只要**在绳上端之上（或齐平）**的那些面（`y <= y_top`），取**最靠下**
        的那个（也就是最先踩上的那层）；
      · 向下爬 ⇒ 对称：只要 `y >= y_bot` 的，取最靠上的那个。
      两组都空（绳顶悬在平台面**下面**这种事很常见）⇒ 退回"离绳端最近的那个面"。

    ⚠ 为什么不能只取"离绳端最近"（第一版就是这么写的，当场挑错了）：同一个集合里
    可能有**只差几像素的两层**（实测「左上平台」里既有一条 -204、又有一条 -208），
    "最近"会挑中绳端**下方**那条 ⇒ 判据比用户要的松 ✗。用户要的是 `y <= -208` ✓。

    找不到（集合里没有 foothold / 本图没有这些 id）⇒ None ⇒ 调用方退回几何兜底。
    """
    ids = set(str(v) for v in ((z.sets.get(dst_set) or {}).get("footholds") or []))
    if not ids:
        return None
    y_top, y_bot = min(L.y1, L.y2), max(L.y1, L.y2)
    cand = []
    for f in terrain.footholds:
        if f.is_wall or str(f.fid) not in ids:
            continue
        cand.append(float(f.y_at((f.x1 + f.x2) / 2.0)))
    if not cand:
        return None
    if direction >= 0:                            # 向上：绳端之上的那些面，取最靠下的
        up = [y for y in cand if y <= y_top]
        return max(up) if up else min(cand, key=lambda y: abs(y - y_top))
    dn = [y for y in cand if y >= y_bot]          # 向下：对称
    return min(dn) if dn else min(cand, key=lambda y: abs(y - y_bot))
