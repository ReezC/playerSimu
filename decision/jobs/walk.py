"""走 / 走过去对齐执行器 `WalkJob`（从 `decision/route.py` 切出）。"""

from decision.jobs.base import _PausableJob
from decision.jobs.params import (DEFAULT_TIMEOUT_S, NEAR_PX, STALL_S, TAP_ON_S,
                                  TAP_PERIOD_S, WALK_TOL_PX)


#: 几何兜底判据允许的高度差（像素）：站上去以后 y 和那条 foothold 的面差这么多以内
#: 才算"站在它上面"（防止 x 对了、其实人在上一层/下一层的平台上）。
WALK_Y_BAND_PX = 30.0


class WalkJob(_PausableJob):
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
    ⚠ 这条边配了 `dir`（仅向左 / 仅向右）时**没有"快到了"这一说**：一直按那一边，直到
    判到达、或者"一直不动"被判失败（见 `update` 里那一段与 `walk_dir` 的说明）。
    卡住（朝目标那侧的距离 `stall_s` 秒没变好）⇒ 失败并说清卡在哪个 x（被墙/台阶挡住
    时人要看得懂，而不是干等到超时 ✗）。
    """

    WALK = "walk"
    DONE = "done"
    FAILED = "failed"

    # ⚠ 这里**曾经**有 `MODE_CENTER / MODE_LEFT / MODE_RIGHT` 和一个 `mode` 参数
    #（还摆进了设置里的「走的方向」下拉）。2026-09-26 用户明确否掉：
    # **默认只有一种走法 —— 朝目标集合的 x 中点**，设置里不许有这个参数。
    # 并留了一句："将来若某一步真需要只按 ← 或 →，那是**逐边**的事：在
    # foothold 编辑器 →「可到达」窗口里配，而不是全局设置。"
    #
    # ⚠ 那个"将来"就是现在（2026-09-27 用户报"我配的通行方式是向左走，可它根本没向左"）：
    # **逐边**的 `dir` 真的接进来了 —— `walk_dir` 参数 ← `walk_job_for_edge` ← 边的 `dir` ✓。
    # **全局设置那条仍然禁止**（`t_walk_only_center` 反向钉着）：它只能来自某一条边 ✓。

    def __init__(self, dst_set, spots, tol_px=WALK_TOL_PX, hold_ms=None,
                 timeout_s=DEFAULT_TIMEOUT_S,
                 stall_s=STALL_S, walk_dir="", stall_jump=True):
        #: ⚠ `stall_s`（"走不动了"的判定时长）**故意只认这个常量默认值** ——
        #: 设置里那格「卡住判定时长(s)」**只管攀爬**（用户 2026-09-27 明确："移除对走的
        #: 相关判定，走用超时配置兜底"⇒ `job_for_edge` 会把那一格从走的参数里摘掉 ✓）。
        #: 走这边的卡住由「寻路超时时间」（`goto_timeout_s`）兜底 ✓。**别把它接回设置** ✗。
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
        self.stall_s = max(0.5, float(stall_s))
        #: 第几次尝试 —— ⚠ **不是上限**（没有"最多试几次"了，见文件头那段 ✓）
        self.attempt = 1
        self.phase = self.WALK
        #: 这条边配的走法 —— **逐边**配的（foothold 编辑器 →「可到达」窗口），
        #: 取值见 `core.zones.WALK_DIRS`：
        #:   ""      = 朝**目标集合的 x 中点**走（老行为，2026-09-26 用户定的唯一走法 ✓）
        #:   "left"  = **只按 ←**（"仅向左"）
        #:   "right" = **只按 →**（"仅向右"）
        #:
        #: 为什么现在才做它：`dir` 原来只是**空占位**（`core/zones.py` 里写着"逻辑等后面
        #: 做"）⇒ 用户 2026-09-27 现场报"配了「仅向左」根本没向左"就是踩在这上面：
        #: 「左上平台 → 左上」目标集合是**整片**「左上」（x 从 -387 到 657、中点 135），
        #: 人站在平台上（x≈2~87）⇒ 中点还在**右边** ⇒ 老代码一路按 → ✗，而且"到集合区间
        #: 的距离"恒为 0 ⇒ 3 秒后判「走不动了」= 卡住 ✗（复现见 tools 里那次探针）。
        #:
        #: ⚠ **全局设置里不许**有这个参数（2026-09-26 用户明确否掉，见本类注释）；它只能
        #:   来自**这一条边** ✓。
        self.walk_dir = (str(walk_dir or "") if str(walk_dir or "") in ("left", "right")
                         else "")
        self.note = ("走过去：「%s」（这条边配了**只能往%s**，一直按到踏上它）"
                     % (self.dst_set, "左" if self.walk_dir == "left" else "右")
                     if self.walk_dir else
                     "走过去：「%s」（%d 个落点，就近）" % (self.dst_set, len(self.spots)))
        self._t0 = None
        self._best_d = None            # 离"最近那个目标点"最好走到过多近
        self._best_at = None
        #: ⭐「走不动之后**单点跳一下**」的两个锚（用户 2026-09-28 要求："walk 执行器判定
        #: 走不动了之后，发个单点跳键"）：
        #:   `_hopped` = 这一轮（一次 `retry` 周期内）跳过没 —— **一轮只跳一次** ✓（跳过
        #:     还走不动才 `_fail`，别无限跳 ✗）；`_hop_at` = 跳键按下去的时刻（点按
        #:     `TAP_ON_S`，None = 没在跳）—— 走不动被小台阶挡住时跳一下就过去了 ✓。
        #:   ⚠ 这两个**不进 `_ANCHORS`**：跳进行中只有 `TAP_ON_S`(30ms) 那么短，而且战斗
        #:     打断回来也不该接着跳（keys 已经清空，重新走就行 ✓）。
        self._hopped = False
        self._hop_at = None
        #: ⭐⭐⭐ 跳这一下**朝哪边按方向键**（-1/0/+1 ✓ 用户 2026-10-17 现场：
        #:   「**走不动跳一下 没有搭配方向键**」✗ —— 原地蹦根本过不去 ✓
        #:   小台阶/小坎要**带着方向跳**才上得去 ✓ 见 `_hop_or_fail` ✓）
        self._hop_dir = 0
        #: ⭐⭐ 「**走不动按跳**」（用户 2026-10-06 ✓ 见 `DecisionSettings.walk_stall_jump`）：
        #:   **关** ⇒ 走不动**直接判失败**（= 加这个单点跳之前的老行为 ✓）。
        #:   ⚠ 由 `agent.start_climb` **一处**灌进来（所有走任务都过那儿 ✓ 见那边的说明 ✓）。
        self.stall_jump = bool(stall_jump)
        #: 被战斗打断时要"跳过时长"的锚点（见 `_PausableJob`）：不扣的话，打完架回来
        #: 第一拍就 `now - _best_at ≥ stall_s` ⇒ 当场报"走不动了" ✗。
        #: （`_t0` **不在里面** —— 它由 `interrupted()` 直接**重打**：每次进战斗超时重算 ✓。）
        self._ANCHORS = ("_best_at",)

    # ---- 对外 ----

    def update(self, now, px, py=None, ladder_id=None, here_sets=None, at_top=None,
               at_bottom=None, ground_y=None, on_rope_pos=None,
               climb_failed=None, climb_stalled=None):
        """喂一拍 → 下一步该干什么。

        ⚠ `px` 必须是**世界坐标 x**（`WorldState.Player.world_x`），不是画面坐标
        —— 和 `ClimbJob.update` 同一个坑（喂错了会往反方向走、来回抖）。
        `ladder_id` 收下但不用（走不看绳）。
        """
        # 被战斗打断过 ⇒ 把"挨打那几秒"从计时锚点上扣掉，并把 `_paused_at` 收掉。
        # ⚠ 必须收掉：不收的话，下一段战斗就**不会再**重打超时了 ✗
        #（`interrupted()` 只在 `_paused_at is None` 时才动手 ✓）。
        self._resume(now)
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
        # ---- 单点跳进行中（走不动后那一跳）：按够 `TAP_ON_S` 就松，回到正常走 ----
        # 用户 2026-09-28：走不动先"单点跳一下"跳过小台阶，别直接判失败（见 `_hop_or_fail`）。
        # 跳键是**点按**（按 `TAP_ON_S` ≈ 30ms 就松，和「跳下绳梯」同一份节奏，不新造时长 ✓）。
        if self._hop_at is not None:
            if (now - self._hop_at) < TAP_ON_S:
                # ⭐⭐ **跳的时候要带着方向键**（用户 2026-10-17 ✓ 现场反证：
                #   「**走不动跳一下 没有搭配方向键**」✗ —— 原地蹦过不去小台阶 ✓）：
                #   老写法 `_result(0, jump=True)` 只发跳键 ⇒ 人原地弹一下、位移为 0
                #   ⇒ 30ms 后接着走还是被同一级台阶挡住 ✓ ⇒ 一轮跳完照样判失败 ✓。
                #   现在跟 `_hop_dir`（跳之前朝目标那一边 ✓）一起发 ✓ ——
                #   跳完那一拍就回到正常走，方向键仍然是同一边（连续，不会回摆 ✓）。
                _hd = int(getattr(self, "_hop_dir", 0))
                self.note = ("跳一下：按住%s + 跳（走不动了，跳过小台阶 ✓）"
                             % ("←" if _hd < 0 else ("→" if _hd > 0 else "（原地）")))
                return self._result(_hd, jump=True)
            self._hop_at = None                      # 跳完了 ⇒ 下面接着走
        # ---- 「仅向左 / 仅向右」（逐边配的 `dir`，2026-09-27 起真的生效）----
        # 只按那一个方向键、**不看集合中点**：这种边说的是"朝那个方向一直走，直到踏上
        # 目标集合"（典型：平台边缘往左走出去、掉到下面那层 ✓）。
        # 进度用**位移**本身（左 = px 变小、右 = px 变大）—— ⚠ 不能用上面那种"到集合
        # x 区间的距离"：目标集合整片盖住人的 x 时那个距离**恒为 0** ⇒ 走得好好的也会被
        # 判「走不动了」✗（用户 2026-09-27 现场就是这么卡住的：人站在 x≈2~87、「左上」
        # 的区间是 -387~657 ⇒ 距离恒 0 ⇒ 3 秒后必报卡住）。
        if self.walk_dir:
            sign = -1 if self.walk_dir == "left" else 1
            _prog = float(px) if sign < 0 else -float(px)
            if self._best_d is None or _prog < self._best_d - 1.0:
                self._best_d, self._best_at = _prog, now
            elif self._best_at is not None and (now - self._best_at) >= self.stall_s:
                return self._hop_or_fail(
                    now,
                    "走不动了：这条边是「仅向%s」，x 卡在 %.0f、%.0fs 没再%s"
                    " —— 可能被墙/台阶挡住，或者这条边配的方向反了（目标「%s」在另一头？）"
                    % ("左" if sign < 0 else "右", float(px), self.stall_s,
                       "变小" if sign < 0 else "变大", self.dst_set),
                    move=sign)
            self.note = "走过去：这条边是「仅向%s」⇒ 只按 %s（走到「%s」上才算到；现在 x=%.0f）" % (
                "左" if sign < 0 else "右", "←" if sign < 0 else "→",
                self.dst_set, float(px))
            return self._out(sign)
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
        self._best_d, self._best_at, _stalled = self._stalled_since(
            now, _prog, self._best_d, self._best_at, self.stall_s,
            lambda v, b: v < b - 1.0)             # 走路：离落点越近越好（缩小 ≥1px 才算进步 ✓）
        if _stalled:
            return self._hop_or_fail(
                now,
                "走不动了：卡在 x=%.0f（目标在%s，还差 %.0f px）"
                "—— 可能被墙/台阶挡住，或者这条边的落点/方向写错了"
                % (float(px), tgt_txt, d),
                move=(1 if dx > 0 else -1))
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
        # ⚠ 单点跳也重新武装：新的一轮允许再"走不动 → 跳一下"一次 ✓
        self._hopped = False
        self._hop_at = None
        self._hop_dir = 0
        # ⚠ 不写分母（同 `ClimbJob.retry`：没有"最多试几次"了 ✓）
        self.note = "第 %d 次尝试：重新走过去「%s」" % (self.attempt, self.dst_set)
        return self._out(0)

    def defer_jump(self, now):
        """⭐⭐ **被「跳跃CD」挡住 ⇒ 这一下跳根本没发出去 ⇒ 把"跳过一次"的记号撤回来** ✓
        （用户 2026-10-17 ✓ 原话：「**现在"走不动时跳一下"的功能失效了**」✗）

        ⚠ 为什么非要撤（不然就是"跳一下"**静默失效** ✗）：agent 在 `_jump_cd_block` 命中时
          **不会把跳键加进去**（见 `agent.py` 那段 ✓ 用户 2026-10-13「如果跳在CD就不允许按跳」✓）
          ⇒ 这一下跳等于**没发生** ✓，可 `_hopped` 已经被 `_hop_or_fail` 置真 ✗
          ⇒ 下一次"走不动"直接 `_fail`（note = "跳了一下还是没过去" ✗）
          ⇒ 观感就是"**这一下跳从来没出现过**"✓。
        ⚠ `_best_at / _best_d` 也一起清 ✓：**等 CD 的这几百毫秒不该被算成"你卡住了"** ✗
          （同 `JumpJob.defer_jump` 的理由 ✓ 见那边 ✓ —— 那个是"推迟"，这个是"撤回再重来" ✓）。
        """
        self._hopped = False
        self._hop_at = None
        self._hop_dir = 0
        self._best_d = None
        self._best_at = None

    def cancel(self, why="取消"):
        # ⚠ **运行时字段一起收干净**（用户 2026-10-02 #8 ✓ 见 `_cancel_common` ✓）
        self._cancel_common(why)
        return self._out(0)          # ★ 走只有 move 一个位置参数 ✓

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

    def _hop_or_fail(self, now, why, move=0):
        """走不动了：**朝目标按住方向 + 跳一下**（跳过小台阶/小坎），跳完还走不动才判失败。

        用户 2026-09-28 要求："walk 执行器判定走不动了之后，发个单点跳键"。
        用户 2026-10-17 ✓ 现场更正："**走不动跳一下 没有搭配方向键**" ✗ ——
          ⛔ 老写法 `move=0`（"只跳、不横挪，跳起来挪会挪歪"）**在小台阶上根本过不去** ✓：
            原地弹一下、位移为 0 ⇒ 30ms 后接着走还是被同一级挡住 ⇒ 一轮跳完照样失败 ✓。
          ⭐ 现在：`move` = **跳之前朝目标那一边**（±1 ✓ 由调用方给：两条走路分支各自
            知道自己按的是哪边 ✓，不在这里重算 ✗）⇒ 带着方向跳，才上得去台阶 ✓；
            跳完那一拍立刻回到正常走，方向还是同一边 ⇒ **不会回摆** ✓。
        · 一轮只跳一次（`_hopped` ✓）：跳完清掉 stall 锚点重新计时，再走不动才 `_fail`
          （被真墙挡住时不会无限跳 ✗）；
        · 跳键是**点按**（按 `TAP_ON_S` ≈ 30ms 就松，见 `update` 里那段"跳进行中"），
          不新造时长常数 ✓。
        """
        # ⭐⭐ **开关**（用户 2026-10-06 ✓ "走不动按跳"）：关掉 ⇒ 走不动**直接判失败** ✓
        #   （= 加这个单点跳之前的老行为 ✓ 一字不差：`note` 还是原来那句"走不动了…" ✓）。
        if not self.stall_jump:
            return self._fail(why)
        if not self._hopped:
            self._hopped = True
            self._hop_at = now
            self._hop_dir = int(1 if float(move) > 0 else (-1 if float(move) < 0 else 0))
            self._best_d = None            # 跳之前那段"没动"不算数，跳完重新计时 ✓
            self._best_at = None
            self.note = ("%s ⇒ **按住%s + 跳一下**，跳过小台阶"
                         % (why, "←" if self._hop_dir < 0
                            else ("→" if self._hop_dir > 0 else "（原地）")))
            return self._result(self._hop_dir, jump=True)
        return self._fail("%s（带着方向跳了一下还是没过去）" % why)

    def _out(self, move):
        """走不按跳、也不按上下（`dir` 恒 0）—— 回值走 `_PausableJob._result` 的统一形状 ✓。"""
        return self._result(move)
