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
            self.note = "对齐好了 ⇒ 按住跳 + %s" % ("↑" if self.dir > 0 else "↓")

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
        why = self._arrived(py, here_sets)
        if why:
            self.phase = self.DONE
            self.note = why
            return self._out(0, False)
        # 还没上绳 / 中途掉下来：**给一段容错期**再判失败 —— 按跳那几拍人本来就还在
        # 空中（`ladder_id` 是 None），一离开就判失败会当场误杀。
        if ladder_id is not None and str(ladder_id) == self.ladder_id:
            self._off_since = None
        else:
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

    def _arrived(self, py, here_sets):
        """到了没有。两个判据，**先集合后几何**（集合是人工圈的，最贴近"到了哪块平台"）。

        判据二（y 越过绳的另一端）是**没有集合信息时的兜底** —— 先做定点上绳时可能
        只有坐标（小地图那行的 foothold/集合还没接上），不能因此就认定"永远没到"。
        """
        if here_sets and self.dst_set and self.dst_set in set(here_sets):
            return "到达：脚下已经是「%s」" % self.dst_set
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


def job_for_edge(terrain, z, edge, **kw):
    """从一条「爬」边造出上绳任务 —— **上下由目标那边在绳的上端还是下端判**。

    这条把三样东西接起来（都走现成设施，不自己发明）：
        · `zones.ladder_ids` 找绳（与编辑器画在绳上的编号同一套）；
        · `zones.climb_direction` 判上下（用户 2026-09-26 的规则）；
        · `ClimbJob` 执行。
    判不出来（绳找不到 / 目标不在绳的两端）就抛 ValueError —— 调用方要么报警要么跳过，
    **不许猜**（猜错就是"往反方向爬"，比不做更糟）。
    """
    from core import zones

    kind = edge.get("kind") or ""
    if kind == "drop":
        return drop_job_for_edge(terrain, z, edge, **kw)
    if kind != "climb":
        raise ValueError("这个模块只认「爬」和「下跳」，收到：%s" % kind)
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
    return ClimbJob(want, L.x, L.y1, L.y2, d, dst_set=edge.get("to"),
                    dst_y=dst_surface_y(terrain, z, edge.get("to"), L, d), **kw)


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
