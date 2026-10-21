"""执行器基类：暂停/继续 + 卡住判定的公共壳（从 `decision/route.py` 切出）。"""

from core.timing import stall_s_from_retry_ms
from decision.jobs.params import PAUSE_MIN_S


class _PausableJob:
    """三个执行器（走 / 爬 / 下跳）共用的「被战斗打断」契约。

    为什么需要它：「攻击范围内有怪」是 tick 里的**第一条总判据**（见 `docs/寻路设计.md`
    的"攻击优先"一节）⇒ 那一拍**整套任务都不跑**（`agent._climb_tick` 根本不被调用 ✓）。
    可任务内部全是**绝对时刻**锚点：`_best_at`（"不动了"）、`_off_since` / `_off_x_since`
    （"掉下来了 / 偏出去了"的宽限）、`_in_tol_since`（"站住保持"）—— 而 `now` 一直在走
    ⇒ 打完架回来，它会把**站着挨打的那几秒**当成"爬不动了 / 走不动了" ✗
    （用户 2026-09-26 报的"爬到 y=-170 就不再升了（3.0s 没变好）"就是这个形状 ✗）。

    契约两条（agent 只管叫一声，怎么继续由各执行器自己定 ✓）：
      · `interrupted(now)` —— "这一拍我没跑你"（幂等：只记第一次的时刻 ✓）；
      · `update()` 开头调 `_resume(now)` —— 回来时把上面那些锚点**整体往后挪**
        （等于"任务的时间停了一会儿" ✓），并返回被打断了几秒，各执行器据此决定从哪继续。

    ⚠ **`_t0`（整体超时）= 每次「进入战斗」就重新计时**（用户 2026-09-26 要求）：
      战斗中挨打不是"这段路卡住了"✗ ⇒ 打怪的时间不许算进寻路超时里 ✓。所以
      `interrupted()` 一被叫到（= 这一拍进了 attack ✓）就把 `_t0` 重打到当前时刻 ✓
      （同一段战斗只重打**一次** —— 每拍都重打就等于"超时永远不触发"✗）。
      代价说清楚：只要一直有怪，这一段路就**不会**被超时收掉 ✓（这正是用户要的：
      超时管的是"卡住"，不是"路上有怪"✓）；要结束只能靠走完 / 手动取消 / 战斗停了
      之后又卡住 ✓。（防"无限重来"的是 **agent 的「寻路超时时间」** ✓ —— 2026-09-27 起
      执行器自己**没有**"最多试几次"了，见文件头那段说明 ✓。）
    """

    #: ⚠ **与模块常量 `PAUSE_MIN_S` 同一个值**（用例 `t_pause_min_s_single_source` 钉着两边相等 ✓）。
    #: 为什么**同时**挂成类属性（2026-10-09 ✓）：`agent` 那层要判"这次打断够久吗"（见
    #: `_climb_tick` 里"回来就重算目的地"那一段 ✓）—— 可 `agent.py` 里**没有 `route` 这个
    #: 模块名** ✗（route 是**反过来** import agent 的这边 ✓ 见 `start_climb` 那段血的教训 ✓）
    #: ⇒ 挂到**实例/类**上，就能 `getattr(job, "PAUSE_MIN_S")` 读到 ✓ 一处口径 ✓ 不用抄一份 ✗。
    PAUSE_MIN_S = PAUSE_MIN_S

    #: 需要"跳过打断时长"的锚点名字（子类在 `__init__` 里列全）。
    #: 漏一个，那条判据就会把打架的那几秒算进去 ✗。
    _ANCHORS = ()

    #: ⭐ **这一步会不会按跳**（用户 2026-10-13 ✓ 见 `DecisionAgent.start_climb` 里那段 ✓）：
    #: `agent` 拿它决定"要不要为「跳跃CD」把这一步**推迟**" ✓ —— 用**类属性**而不是
    #: `isinstance(JumpJob)` ✗：`agent.py` **不许** import `route`（route 反过来 import agent ✓
    #: 会成环 ✓ 见 `start_climb` 那段血的教训 ✓）⇒ 靠这个开关"打字" ✓。
    NEEDS_JUMP = False

    def interrupted(self, now):
        """agent 在"这一拍没跑任务"时叫一声（同一段战斗里每拍都会叫，幂等 ✓）。

        **进场即重打超时**（用户 2026-09-26 要求："每次进入 attack，寻路超时要重新计时"）：
        战斗不是"这段路卡住了" ✗ ⇒ 挨打的时间不算超时 ✓。只在**这一段战斗的第一次**
        （`_paused_at is None` ✓）重打 —— 每拍都重打就等于"超时永不触发" ✗。
        ⚠ 重打的是**整体超时**（`_t0`，"这一段走了多久"的总账）；"爬不动了 / 走不动了"
        那几条细判据仍然只看**真正在走的那段时间**（由 `_resume` 把打断时长扣掉 ✓）。
        """
        if getattr(self, "_paused_at", None) is None:
            self._paused_at = float(now)
            self._t0 = float(now)          # ← 超时从"进战斗这一刻"重算（见上面说明）

    def _resume(self, now):
        """被打断后**回来那一拍**：把打断时长从进度/宽限锚点上扣掉 ⇒ 返回被打断了几秒。"""
        p = getattr(self, "_paused_at", None)
        if p is None:
            return 0.0
        self._paused_at = None
        dt = max(0.0, float(now) - p)
        for name in self._ANCHORS:
            v = getattr(self, name, None)
            if v is not None:
                setattr(self, name, v + dt)
        return dt

    def defer_jump(self, now):
        """**「跳跃CD」还没到 ⇒ 这一下跳被拦下了**（由 `agent._climb_tick` 调 ✓ 用户 2026-10-13 ✓）。

        契约：**把"我要跳"的超时锚点推到现在** ✓ —— 不这么做，`stall_s` 那道"跳过去没到 ⇒
        判失败"的钟会把**等 CD 的这几百毫秒**算进去 ✗ ⇒ 用户要的"连续 2 段寻路时第 2 段**推迟**"
        就变成"第 2 段**直接失败**"了 ✓。
        ⚠ 默认**什么都不做** ✓（只有"真会按跳、且拿 `now - 锚点` 判超时"的执行器才需要覆盖 ✓
          现在只有 `JumpJob` ✓ 见那边 ✓；将来谁加了新的跳类执行器，自己覆盖一份 ✓）。
        """
        return None

    # ⭐ 执行器回值的**统一形状**（2026-09-27 结构整理 ✓）：四个 `_out` 原来各拼一份一模一样的
    #   dict ✗ —— 现在**一处拼** ✓，形状与以前逐字一致 ✓。
    def _result(self, move, jump=False, dir=0, reassert=None, replan=False):
        """`reassert=None` = **不带**这个键（走 / 跳从不重发按键 ✗，带上反而误导 ✓）；
        `False/True` = 带上（爬 / 下跳会重发 ✓）。
        `replan=True` = **位置状态返回非目标**（上了别的绳 / 站在别的集合）⇒ 通知上层重启寻路
        （用户 2026-10-02 ✓ 治本：align 阶段按住 ↑ 是设计，但被非目标绳吸住时该立刻重启，不卡住）。"""
        out = {"phase": self.phase, "move": int(move), "jump": bool(jump),
               "dir": dir, "note": self.note,
               "done": self.phase == self.DONE,
               "failed": self.phase == self.FAILED}
        if reassert is not None:
            out["reassert"] = bool(reassert)
        if replan:
            out["replan"] = True
        return out

    def _cancel_common(self, why):
        """外部叫停的**统一收尾**（用户 2026-10-02 #8 ✓）—— 四个 `cancel()` 共用这一处。

        ⚠ 为什么必须这么做（原风险）：`cancel()` 原来**只置 `FAILED` + note**，**一个运行时
          字段都不清** ✗；而 `retry()` 要清的字段一大把（ClimbJob ~28 个）⇒ 将来谁复用同一个
          job 实例、或者新加字段忘了同步，就会带着**上一个任务的脏锚**（历史踩过：
          `_diag_at` / `_climb_sig` / `_rope_seen` ✓）。⇒ 让 `cancel` **复用 `retry` 的清单**
          （一处口径 ✓：以后加字段只改 `retry`✓），再盖成 `FAILED` + 原因 ✓。
        ⚠ `attempt` 会被 `retry()` +1：取消后这个对象就作废了，无副作用 ✓（用例只看相位/note ✓）。
        ⚠ **这里不碰 `_out`**：四个子类的 `_out` 形状不同（走是 `_out(move)`、跳是 `_out(move, jump)`
          ⇒ 没有共同的调用形状 ✗）—— 回值由各自的 `cancel()` 自己拼 ✓（见那四处 ✓）。
        """
        self.retry()
        self.phase = self.FAILED
        self.note = str(why)

    # ⭐ 共享的「进步跟踪」骨架（2026-09-27 结构整理 ✓）：ClimbJob 的"爬不动"与 WalkJob 的
    #   "走不动"原来**各写一遍** `best + best_at + 卡住时长` 的比较 ✗ —— 现在一处 ✓。
    #   ⚠ 另两个"卡住"（DropJob 的 DROP 相、JumpJob 的跳后超时）是**一次性计时**（"离某个
    #   时刻过了多久"），不是"best 值有没有变好"，语义不同 ⇒ 不硬塞进来（硬塞 = 增加耦合 ✗）。
    def _stalled_since(self, now, value, best, best_at, stall_s, better):
        """`better(value, best)` = 这一拍有没有**朝目标**变好（调用方给 ✓，方向各不同：
        上爬 = y 越小越好、下爬 = y 越大越好、走路 = 离落点越近越好 ✓）。
        变好 ⇒ 记新 best / 时刻、不算卡；否则 `now - best_at ≥ stall_s` ⇒ **卡住** ✓。
        返回 `(best, best_at, stalled)` —— 调用方把前两个存回自己的锚点 ✓。
        """
        if best is None or better(value, best):
            return value, now, False
        stalled = (best_at is not None) and (now - best_at) >= stall_s
        return best, best_at, stalled


# ⚠⚠ 2026-10-21：换算**已搬进 `core/timing.py`**（那个 0.5 秒下限是**两层共用**的 ✓）。
#   为什么搬：感知那侧（`perception/world/pos_state.py` 判「爬不动」）原来**自己又算了一遍**
#   `ms / 1000`（**没有下限** ✗）⇒ 天空项目 350 ms 时：执行器按 0.5 秒、广播按 0.35 秒 ⇒
#   广播**先**报"爬不动" ⇒ 执行器松 ↑ ⇒ 人往下滑 ⇒ 用户报的「3→2 在 L1 上不按 ↑ 了」✗。
#   这里**保留同名再导出**（三个既有调用点不用改 ✓），实现只留在 `core` 一份 ✓。
#
#   ```python
#   from decision.jobs.base import stall_s_from_retry_ms   # ← 老写法照旧可用 ✓
#   ```
