"""跳跃执行器 `JumpJob`（从 `decision/route.py` 切出）。"""

from decision.jobs.base import _PausableJob
from decision.jobs.params import (DEFAULT_TIMEOUT_S, JUMP_START_PX, STALL_S,
                                  TAP_ON_S, WALK_TOL_PX)


class JumpJob(_PausableJob):
    """一次「跳」（`jump`）：走到目标平台最近的那条 foothold 边 → **按跳** → 落到目标集合。

    用户 2026-09-27 定的流程（照抄流程图，每个节点都在 `update` 里能指到）：

    ```
    入口（被打断了，还原时从这开始）
      ├─ 角色 x 已经在**目标 foothold 集某条的 x 范围**里 ⇒ **直接按跳**
      └─ 否则 ⇒ 朝**最近的那条目标 foothold**走，进了「起跳距离」⇒ 按跳
    等角色位于**目标 foothold 集合** ⇒ 成功（这一步结束，交给下一段）
    按跳之后「卡住判定时长」还没到目标集合 ⇒ **回入口**（重走、再跳）
    ```

    ⚠ 四条口径（都是用户定的，别自己发明 ✗）：
      · 「最近」= **到那条 foothold 的 x 范围的距离**最近（= 到它**最近端点**的距离 ✓），
        范围内算 0 ✓；
      · 「起跳距离」`jump_start_px` 是**临时参数**：默认 0 ⇒ 进了范围才跳 ✓；
      · **不设"最多试几次"**（用户 2026-09-27）：回入口可以一直重来，整天由 agent 的
        「**寻路超时时间**」兜底 ⇒ 本执行器**自己不判失败** ✓（只在"拿不到世界坐标"时
        如实说，并等着 ✓）；
      · 到达**只认集合**（`here_sets` 里有目标集合 ✓）—— **不用几何兜底**：人还在空中时
        x 早就落在目标范围里了，几何会**误判"到了"** ✗（等跳跃标定 + 落点预测做完，
        才有资格判"落哪儿了" ✓）。

    **跳过之后方向键一直按着**（用户 2026-09-27："当进入起跳距离后按跳这一步，方向键不能
    松开，要等成功或超时判定才能松"✓）—— 方向就是起跳前那一下在走的方向（朝最近那条
    foothold ✓），按到**到了**（`_arrived` ⇒ 松 ✓）或**卡住判定**到点（回入口 ⇒ 松 ✓）为止；
    跳键本身是**点按**（`TAP_ON_S` 之后松手，和「跳下绳梯」同一个节奏，不新造时长常数 ✓）。

    对外形状与走 / 爬 / 下跳**完全一样**（`update/retry/cancel/attempt/phase` +
    `interrupted/_resume`）⇒ `agent._climb_tick` **一行都不用改** ✓。
    """

    ENTRY = "entry"          # 入口：判"要不要按跳"，不要就朝最近的 foothold 走
    JUMP = "jump"            # 已按跳：点按 + 等落到目标集合（超 `stall_s` ⇒ 回入口）
    DONE = "done"
    FAILED = "failed"

    def __init__(self, dst_set, spots, src_set="", tol_px=WALK_TOL_PX, hold_ms=None,
                 stall_s=STALL_S, jump_start_px=JUMP_START_PX,
                 timeout_s=DEFAULT_TIMEOUT_S):
        self.dst_set = str(dst_set or "")
        self.src_set = str(src_set or "")
        #: 候选落点 `[(左, 右, foothold id)]` —— 由 `jump_job_for_edge` 给（目标集合的全部 ✓）
        self.spots = [(float(a), float(b), str(i)) for a, b, i in (spots or [])]
        if not self.spots:
            raise ValueError("跳跃任务没有候选 foothold")
        #: 收下但**不用**：入口只有一条（`job_for_edge` 会把对齐那套参数一起传进来 ✓，
        #: 少一个参数就是 `TypeError`，和 `WalkJob.hold_ms` 同一个理由 ✓）
        self.tol_px = max(1.0, float(tol_px))
        self.hold_ms = hold_ms
        #: 「卡住判定时长(s)」：**按跳之后**这么久还没进目标集合 ⇒ 回入口 ✓
        #: （用户 2026-09-27 同意把它从"只管攀爬"扩成"攀爬 + 跳跃" ✓）
        self.stall_s = max(0.5, float(stall_s))
        #: 「起跳距离」（世界像素；0 = 进了目标 foothold 范围才跳 ✓，见 `JUMP_START_PX`）
        self.jump_start_px = max(0.0, float(jump_start_px))
        self.timeout_s = float(timeout_s)
        #: 第几次尝试 —— ⚠ **不是上限**（用户 2026-09-27 起**四个执行器都不设**"最多试
        #: 几次"了 ⇒ 整段由 agent 的「寻路超时时间」收口 ✓，见文件头那段）。
        self.attempt = 1
        self.dir = 0                       # 跳**不带上下**（不走 ↑/↓ 那一套 ✓）
        self.phase = self.ENTRY
        self.note = "跳过去：「%s」（%d 个候选落点，就近）" % (self.dst_set, len(self.spots))
        self._t0 = None
        #: 这一次按跳的时刻（= 「卡住判定时长」的起点 ✓；None = 还没按 ✓）
        self._jump_at = None
        #: 起跳那一刻**手上的方向**（-1/0/+1，0 = 起跳前本来就没在走 —— 人已经在目标 fh
        #: 范围里 ⇒ `JUMP` 那一相不发方向键 ✓）。用户 2026-09-27："当进入起跳距离后按跳
        #: 这一步，方向键不能松开，要等成功或超时判定才能松" ⇒ 这个值**冻结在起跳那一刻**
        #: （跳出去以后不再跟着 px 变 —— 那时人已经在空中，读数拿它做方向没意义 ✓）。
        self._jump_dir = 0
        #: 被打断时要"跳过时长"的锚点（见 `_PausableJob`）：不扣的话，打完架回来第一拍
        #: 就会 `now - _jump_at ≥ stall_s` ⇒ 当场判"跳过去没到"✗（其实刚按下去）。
        self._ANCHORS = ("_jump_at",)

    #: ⭐ 本执行器**会按跳**（见 `_PausableJob.NEEDS_JUMP` ✓）⇒ agent 会为「跳跃CD」
    #:   把"连跳时的第 2 段"**推迟** ✓（用户 2026-10-13 ✓）。
    NEEDS_JUMP = True

    def defer_jump(self, now):
        """⭐⭐ **被「跳跃CD」挡住 ⇒ 把"跳锚"推到现在**（由 agent 调 ✓ 用户 2026-10-13 ✓
        见 `_PausableJob.defer_jump` ✓）。

        为什么非要推它：`_jump_at` 同时是"跳过去没到 ⇒ 判失败"那道钟（`now - _jump_at >= stall_s`
        ✓ 见下面 JUMP 相 ✓）—— 不推的话，**等 CD 的那几百毫秒**会被算成"跳了没效果" ✗
        ⇒ 用户点名要的"**第 2 段寻路任务推迟**"就变成"**第 2 段直接失败**"了 ✗✗。
        推完之后：CD 一过、真跳出去那一刻钟才重新起算 ✓（= 只是**推迟** ✓）。
        """
        self._jump_at = float(now)

    # ---- 对外 ----

    def update(self, now, px, py=None, ladder_id=None, here_sets=None, at_top=None,
               at_bottom=None, ground_y=None, on_rope_pos=None,
               climb_failed=None, climb_stalled=None):
        """喂一拍 → 下一步该干什么。

        ⚠ `px` 必须是**世界坐标 x**（`WorldState.Player.world_x`），不是画面坐标
        —— 和其它三个执行器同一个坑（喂错了方向会反、来回抖 ✗）。
        `py` / `ladder_id` 收下但不用（跳不看高度、不看绳 ✓）。
        """
        self._resume(now)
        if self._t0 is None:
            self._t0 = now
        if self.phase in (self.DONE, self.FAILED):
            return self._out(0, False)
        if now - self._t0 > self.timeout_s:
            # ⚠ 实际跑起来 `timeout_s` 被 agent 设成 1e9（「寻路超时时间」统一在那边管 ✓）
            #   ⇒ 这里只是执行器单独用（写用例）时的兜底 ✓
            return self._fail("超时 %.0fs：没能从「%s」跳到「%s」"
                              % (self.timeout_s, self.src_set, self.dst_set))

        # ---- 节点①：到了没有（每一拍都判：站在地上、跳在空中都算 ✓）----
        why = self._arrived(here_sets)
        if why:
            self.phase = self.DONE
            self.note = why
            return self._out(0, False)

        # ---- 节点②：已经按跳 ⇒ 点按完就等落地；「卡住判定时长」到点还没到 ⇒ 回入口 ----
        if self.phase == self.JUMP:
            if (now - self._jump_at) >= self.stall_s:
                self.phase = self.ENTRY
                self._jump_at = None
                self._jump_dir = 0      # 「超时判定」到点 ⇒ 这里才松方向（用户 2026-09-27 ✓）
                self.attempt += 1
                self.note = ("跳过去 %.1fs 还没到「%s」⇒ **回入口**重来（第 %d 次；"
                             "整体由「寻路超时时间」兜底）"
                             % (self.stall_s, self.dst_set, self.attempt))
                # 这一拍**只记这一笔**（什么都不按）：下一拍才重新判"要不要再跳 / 要不要
                # 走近" ✓。为什么要单独占一拍：不占的话这句"回入口"会**当场被下面那句
                # 走位说明盖掉**，日志里就再也看不出"刚才那次跳失败了"✗（实测踩过）。
                return self._out(0, False)
            # ⚠ 方向键**一直按着**（`self._jump_dir`，起跳那一刻冻结的）：用户 2026-09-27
            #   "当进入起跳距离后按跳这一步，方向键不能松开，要等成功或超时判定才能松" ✓
            #   —— 跳键点按、方向键长按，正是"跑跳"的手感（松手就变成原地跳 ✗）。
            if (now - self._jump_at) < TAP_ON_S:
                return self._out(self._jump_dir, True)   # 点按还没够 ⇒ 跳键也继续按着 ✓
            else:
                self.note = "跳了：方向键继续按着（%s），等落到「%s」（%.1fs 没到就回入口）" % (
                    "←" if self._jump_dir < 0 else ("→" if self._jump_dir > 0 else "无"),
                    self.dst_set, self.stall_s)
                return self._out(self._jump_dir, False)  # 松跳、**方向不松** ✓

        # ---- 节点③：入口 —— x 在目标 foothold 范围里就"直接按跳"，否则先走近 ----
        if px is None:
            # 定不了位就别乱跳（跳出去落在哪完全不可知 ✗）—— 如实说，让人去查小地图 ✓
            self.note = "跳出去：拿不到世界坐标（小地图定位没有输出）"
            return self._out(0, False)
        px = float(px)
        near = min(self.spots, key=lambda s: self._span_dist(px, s))
        d = self._span_dist(px, near)
        if d <= self.jump_start_px:
            self.phase = self.JUMP
            self._jump_at = now
            # 记下起跳前**手上的方向**（= 朝最近那条 foothold 那一边）——`JUMP` 那一相不松手 ✓
            # （用户 2026-09-27："进入起跳距离后按跳这一步，方向键不能松开"✓）。
            # 已经在范围里（d ≤ 0）说明本来就没在走 ⇒ 0（那一相也就不发方向键 ✓）。
            self._jump_dir = (1 if near[0] > px else (-1 if near[1] < px else 0))
            self.note = ("已经在目标 foothold（fh %s）的 x 范围里 ⇒ **直接按跳**" % near[2]
                         if d <= 0 else
                         "进了起跳距离（离 fh %s 的范围还差 %.0f px ≤ %.0f）⇒ **按跳**，"
                         "方向键（%s）继续按着"
                         % (near[2], d, self.jump_start_px,
                            "←" if self._jump_dir < 0 else "→"))
            return self._out(self._jump_dir, True)
        self.phase = self.ENTRY
        self.note = ("走向目标 foothold：fh %s 的范围是 %.0f~%.0f，还差 %.0f px"
                     "（起跳距离 %.0f）" % (near[2], near[0], near[1], d,
                                          self.jump_start_px))
        return self._out(1 if near[0] > px else -1, False)

    def retry(self):
        """失败后重新激活：**回到入口**（和流程图一致 ✓）；本执行器一般不判失败 ✓。"""
        self.attempt += 1
        self.phase = self.ENTRY
        self._t0 = None
        self._jump_at = None
        self._jump_dir = 0
        self.note = "第 %d 次尝试：回入口（跳）" % self.attempt
        return self._out(0, False)

    def cancel(self, why="取消"):
        # ⚠ **运行时字段一起收干净**（用户 2026-10-02 #8 ✓ 见 `_cancel_common` ✓）——
        #   原来这里只置相位，带着脏锚被复用就是隐患 ✗。
        self._cancel_common(why)
        return self._out(0, False)

    # ---- 内部 ----

    @staticmethod
    def _span_dist(px, spot):
        """`px` 到那条 foothold 的 **x 范围**的距离：在范围里 = 0，否则＝到**最近端点**的距离 ✓。"""
        left, right, _fid = spot
        if px < left:
            return left - px
        if px > right:
            return px - right
        return 0.0

    def _arrived(self, here_sets):
        """到了没有：**只认集合**（用户 2026-09-27 口径 ✓，见类说明里为什么不用几何 ✗）。"""
        if here_sets and self.dst_set and self.dst_set in set(here_sets):
            return "到达：脚下已经是「%s」" % self.dst_set
        return ""

    def _fail(self, why):
        self.phase = self.FAILED
        self.note = why
        return self._out(0, False)

    def _out(self, move, jump):
        """跳的执行器输出：**不带 ↑/↓**（`dir` 恒 0）✓。"""
        return self._result(move, jump)
