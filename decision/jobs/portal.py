"""传送门执行器 `PortalJob`（从 `decision/route.py` 切出）。"""

from decision.jobs.align import align_x_decide
from decision.jobs.base import _PausableJob
from decision.jobs.params import (ALIGN_TOL_PX, DEFAULT_TIMEOUT_S, NEAR_PX,
                                  STALL_S, TAP_ON_S, TAP_PERIOD_S)


#: 传送点**到达**判据的容差：抄 `core.mapdata.Terrain.PORTAL_DX/PORTAL_DY`（`portal_at` 同一口径 ✓
#: —— 那边判"人在某扇门的判定框里"用的就是这两个数 ✓ 别各写一份 ✗）。
#: ⚠ 它管的是"**我已经站在出口门上了**"这件事 ✓；**对齐**（走多准才按 ↑）用的是
#:   「坐标对齐误差范围」(`tol_px` ✓ 用户 2026-10-06 ✓) —— 两件事别混 ✗。
PORTAL_TOL_X, PORTAL_TOL_Y = 15.0, 12.0


class PortalJob(_PausableJob):
    """一次「走传送点」：**走到门前 ⇒ 点按一下 ↑** ⇒ 等到达（用户 2026-10-06 ✓）。

    用户原话："操作方式是走到该点位按↑" ＋ "点一下↑" ✓（按 `TAP_ON_S` ≈30ms 就松 ✓ 不拍新常数 ✓）。
    对外形状同 `WalkJob` / `ClimbJob` / `DropJob`（`update()` 纯函数 + `retry` / `cancel` ✓）
    ⇒ `agent._climb_tick` 那一套（任务优先 / 有怪先打 / 失败延迟重来 / 原因挂画面）**一行都不用改** ✓。

    三段（`phase`）：
      WALK —— 朝**门**的 x 走（进步跟踪借 `_stalled_since` ✓ 卡住如实报 ✓ 同走路那条 ✓）；
      TAP  —— 人**进了「坐标对齐误差范围」**(`tol_px` ✓) 且 y 也对 ⇒ **点按一下 ↑**（`dir=+1` ✓）；
      WAIT —— 松手之后等"真到了那边"；**位置状态没回执 ⇒ 补按 ↑**（见下 ✓）。

    ⭐⭐ **对齐与补按：与攀爬执行器**同一套**（用户 2026-10-06 ✓ 原话："去传送门的对齐需要调用
      攀爬执行器对齐那个方法：在「坐标误差范围」内就可以按 ↑ ＋ **位置状态没回执补按 ↑**"✓）：
      · **对齐判据用 `tol_px`**（= 设置里的「**坐标对齐误差范围**」✓ 与 `ClimbJob.tol_px`
        是**同一把尺** ✓ 由 `plan_jobs` 一处灌进来 ✓）—— 不再自己拍一个"门的框"当对齐线 ✗；
      · **没回执 ⇒ 补按 ↑**：在 WAIT 相里，"已经到了"就是**回执** ✓；还没到 ⇒ 每
        「**移动操作尝试间隔**」(`retry_ms` ✓ 与爬绳 / 下跳同一个参数 ✓) **再发一次 PRESS ↑**
        （`reassert=True` ✓ 由 `agent._climb_tick` 统一执行 —— `KeyState` 只在键集变化时才发键，
        对面把键丢了本机不知道 ✗，见那里那段说明 ✓）。
      ⚠ **不再自己判"按了 ↑ 多久还没到"** ✗：原来那个 `PORTAL_WAIT_S`(2.5s) 会在**第一次补按
        还没到点**时就把任务判死 ✗（补按间隔默认 3s > 2.5s，等于**永远补不上**✗✓）。
        现在与爬绳同一口径：**真过不去由「寻路超时时间」兜底**（用户 2026-09-29 定的 ✓
        见 `agent._climb_tick` 那道总闸 ✓）。

    ⚠ **判到达优先看集合**（与 `WalkJob` 同一套口径 ✓）：传送的落点由**出口门**决定 ✓
      （`zones.portal_exit` 按 `tn` 解析 ✓ —— 这正是 2026-09-27 移除这个通行方式时缺的"落点" ✓），
      而"目标集合"由用户在编辑器里指 ✓（通常就是出口门所在那个集合 ✓）。
    ⚠ 拿不到坐标 / 门 / 落点 ⇒ **如实说、不乱走**（宁可失败，也别瞎按 ↑ 把自己送错地方 ✓）。
    ⚠ **这里不接「走不动按跳」** ✗：那个开关管的是**走路**执行器（见 `WalkJob._hop_or_fail` ✓）；
      传送点这一段走不动就如实报 ✓（两件事别混 ✓）。
    ⚠ **寻路代价**（用户 2026-10-06 定的口径 ✓）：代价 = **当前 → 起点门** + **出口门 → 目标**
      ⇒ 前一半由"走到门口"这段本身体现 ✓、后一半由 `exit_portal` 的坐标给 ✓
      （规划那边取代价时按这个算 ✓ 见 `zones.portal_exit` 的说明 ✓）。
    """

    # ⚠ `DONE` / `FAILED` **必须自己定义** ✗ —— 它们**不在基类上**（每个任务各一份 ✓，
    #   `_PausableJob._result` 里 `self.phase == self.DONE` 就是这么判的 ✓）：
    #   实测漏定义 ⇒ `AttributeError: 'PortalJob' object has no attribute 'DONE'` ✓，
    #   而且**第一拍就炸** ✓（因为它一进 `update` 就要拼回值 ✓）。取值与走/爬/下跳那三个一致 ✓。
    WALK, TAP, WAIT = "walk", "tap", "wait"
    DONE, FAILED = "done", "failed"

    def __init__(self, dst_set, portal, exit_portal=None, tol_px=ALIGN_TOL_PX,
                 hold_ms=250, near_px=NEAR_PX, align_gap_ms=None, retry_ms=0,
                 timeout_s=DEFAULT_TIMEOUT_S, stall_s=STALL_S):
        self.dst_set = str(dst_set or "")
        self.portal = portal
        self.exit_portal = exit_portal          # 出口门（`tn` 解析出来的 ✓ 可能是 None ✓）
        #: 「**坐标对齐误差范围**」（世界像素 ✓）—— 与 `ClimbJob.tol_px` **同一把尺** ✓
        #: （`plan_jobs` 一处灌进来 ✓ 用户 2026-10-06："在「坐标误差范围」内就可以按 ↑"✓）。
        self.tol_px = max(1, int(tol_px))
        #: 「**坐标对齐误差时间**」(ms)：进了容差还要**稳这么久**才按 ↑ ✓（与爬绳同一把尺 ✓
        #: 用户 2026-10-06："整套照搬对齐式爬绳梯的逻辑"✓）—— 免得"路过一下"就按 ↑ ✗。
        self.hold_ms = max(0, int(hold_ms))
        #: 「**开始对齐…的距离(px)**」（`near_px` ✓）：差得比它远 ⇒ **一口气按住走**；
        #: 进到它以内 ⇒ **改成点按**（一下一下，受 `align_gap_ms` 约束 ✓）—— 同 `ClimbJob` ✓。
        self.near_px = max(1.0, float(NEAR_PX if near_px is None else near_px))
        #: 「**对齐绳梯移动延迟(ms)**」：两次按下方向键之间至少隔这么久 ✓（0 = 不限制 ✓）。
        #: 默认 = `TAP_PERIOD_S`（= 原来那个点按周期 ✓ 老行为 ✓）；由 `agent.start_climb`
        #: 一处灌进来 ✓（它对**每个**任务都灌 ✓ 所以这里只需给个默认 ✓）。
        self.align_gap_ms = (int(TAP_PERIOD_S * 1000.0) if align_gap_ms is None
                             else max(0, int(align_gap_ms)))
        #: 「**移动操作尝试间隔**」(ms) ⇒ `_iv` 秒没回执就**补按一次 ↑** ✓（0 = 不补 ✓ 老行为 ✓）。
        #: ⚠ `agent.start_climb` 会给每个任务灌 `job.retry_ms`（它读设置 ✓ 一处换算 ✓）——
        #:   `DropJob` 用毫秒、这里也用毫秒 ✓（爬绳那边是秒，见 `ClimbJob.reassert_s` ✓）。
        self.retry_ms = max(0, int(retry_ms))
        self.timeout_s = float(timeout_s)
        self.stall_s = max(0.2, float(stall_s))
        self.phase = self.WALK
        self.attempt = 0
        self.note = ("走传送点：「%s」—— 走到门 %s（对齐到 ±%d px）、按一下 ↑"
                     % (self.dst_set, getattr(portal, "pn", "?"), self.tol_px))
        self._t0 = None
        self._best_d = None
        self._best_at = None
        self._tap_at = None
        self._wait_at = None
        #: **上一次发 ↑ 的时刻**（补按的节拍锚 ✓ 见 WAIT 相 ✓）。
        self._press_at = None
        #: 补按了几次（只给 note / 排查看 ✓）。
        self._retap_n = 0
        # ---- 对齐相的记账（`align_x_decide` 的 `state` 就存这几个 ✓ 一处写、一处读 ✓）----
        #: 现在按着哪边（0 = 没按 ✓）—— 也是"这一次按下"的判据（同 `ClimbJob._press_dir` ✓）。
        self._alx_dir = 0
        #: 这一次按下的时刻（⚠ 只有"新的一次按下"才写 ✓ 见 `align_x_decide` ✓）。
        self._alx_press_at = None
        #: 上一次"真的靠近了"的时刻（**补按方向**的判据 ✓）。
        self._alx_progress_at = None
        #: 目前为止最小的 `|dx|`（进步记账 ✓）。
        self._alx_best_dx = None
        #: 对齐相补按了几次（note / 排查 ✓）。
        self._alx_reasserts = 0
        #: **进容差的时刻**（稳 `hold_ms` 用 ✓）。
        self._in_tol_since = None
        # ---- ⭐ **对齐期间"点按 ↑"的记账**（用户 2026-10-06 起 ✓ 传送点**专属** ✓）----
        # 「**增加传送点的特殊对齐逻辑**：对齐过程开始时就可以按 ↑，并以「移动操作尝试间隔」
        #   连续点按 ↑」 —— 为什么要它：门是**按 ↑ 才开**的 ✓，而"先对齐到 ±10px 再按 ↑"
        #   等于把人**钉在门口**等对齐 ✗ ⇒ 站着一路试才有可能直接开门 ✓。
        # ⭐⭐ **2026-10-09 用户修正"起拍点"** ✗（原话："传送点执行器，**对齐期间搁很远就按 ↑
        #   是不对的**，需要是要在**首次到达「坐标对齐误差范围」内才开始连按 ↑**"✓）：
        #     · 原来（2026-10-06）：进**对齐相**第一拍就打 ⇒ 实测"还差 40px 就一路 ↑" ✗；
        #     · 现在：**先进 `tol_px`**（= 设置里那把尺「坐标对齐误差范围」✓ 与对齐判据**同一个
        #       数** ✓ 不新造容差 ✓）才起拍；而且**只认第一次**（`_up_armed` 一置上就不回退 ✓）
        #       —— 之后哪怕过冲又出去了，↑ **照旧连着打** ✓（"已经在门口试上了就别断" ✓）。
        #: 「**曾经**进过「坐标对齐误差范围」」—— `False` ⇒ **一个 ↑ 都不许打** ✗。
        #: ⚠ 它是**开关**、不是时刻 ⇒ **不许**塞进 `_ANCHORS` ✗（那套是按"战斗打断了多久"
        #:   整体**平移时刻**的 ✓ 平移一个布尔会把它翻成真 ✗）。一轮任务里只前进不回退 ✓。
        self._up_armed = False
        #: 这一下 ↑ 是从哪一刻按下的（`None` = 现在没按着 ✓）—— 按满 `TAP_ON_S` 就松 ✓
        #:   （与后面 `TAP` 相**同一个时长口径** ✓ 不新造时长 ✓）。
        self._up_tap_at = None
        #: 下一按该在什么时候（`None` = **还没打过** ⇒ 一"武装"就立刻起拍 ✓）。
        self._up_next_at = None
        #: ⭐ **连续点按的节拍 = 全项目那份「点按周期」`TAP_PERIOD_S`(180ms)**（用户 2026-10-06
        #:   第二轮 ✓ 原话"**取消传送点通行方式对齐的时候 ↑ 以「移动操作尝试间隔」按，直接连续
        #:   发点按**"✓）—— ⚠ **不再有 `_up_repeat` 这个开关** ✗：原来"`retry_ms=0` ⇒ 只打开头
        #:   那一下"是按「移动操作尝试间隔」那个口径来的 ✓ 现在一律连续 ✓ 与它无关 ✓
        #:   （`retry_ms` 只再管进容差之后 `WAIT` 相的"补按 ↑"✓ 那条一个字没动 ✓）。
        #: 打了多少下（note / 排查 ✓，同 `_retap_n` 的用法 ✓）。
        self._up_taps = 0
        # 被战斗打断时要把这几个时刻整体往后挪（`_PausableJob` ✓ 同 WalkJob 的取舍 ✓）
        # ⚠ `_press_at` 也要挪 ✗：不挪的话"打完架回来"会拿打架前那个锚算 ⇒ 一回来就立刻补按 ✗。
        # ⚠ 对齐相的 `_alx_press_at` / `_alx_progress_at` / `_in_tol_since` **同样要挪** ✗ ——
        #   不挪就是"打完架回来立刻补按一次方向 / 立刻算稳够 hold_ms"✓（假的 ✓）。
        # ⚠ 那里↑点按的两个锚点也一样（`_up_tap_at` / `_up_next_at`）✗：不挪 ⇒ 打完架回来
        #   立刻又打一下 ↑（假的 ✓ 而且那一下本该在打架前就打完 ✓）。
        self._ANCHORS = ("_best_at", "_tap_at", "_wait_at", "_press_at",
                         "_alx_press_at", "_alx_progress_at", "_in_tol_since",
                         "_up_tap_at", "_up_next_at")

    # ---- 内部 ----

    def _arrived(self, px, py, here_sets):
        """到了没有：**集合优先、出口门兜底**（见类说明 ✓）。"""
        if here_sets and self.dst_set and self.dst_set in set(here_sets):
            return "到达：脚下已经是「%s」" % self.dst_set
        ep = self.exit_portal
        if (self.phase == self.WAIT and ep is not None
                and px is not None and py is not None
                and abs(float(px) - float(ep.x)) <= PORTAL_TOL_X
                and abs(float(py) - float(ep.y)) <= PORTAL_TOL_Y):
            return "到达：已经站在出口门 %s 上" % getattr(ep, "pn", "?")
        return ""

    def _fail(self, why):
        self.phase = self.FAILED
        self.note = why
        return self._out(0)

    def _out(self, move):
        """走/等的时候不按跳、不按上下（`dir` 恒 0 ✓）—— 回值走统一形状 ✓。"""
        return self._result(move)

    def _up_tap_step(self, now):
        """⭐ 对齐期间**点按 ↑** 的节拍（用户 2026-10-06 传送点专属口径 ✓）—— True = 这一拍按 ↑。

        口径（2026-10-06 立、**2026-10-09 修正起拍点** ✓）：
          · ⭐⭐ **先"武装"才起拍**（`_up_armed` ✓ 用户 2026-10-09 原话："对齐期间搁很远就按 ↑
            是不对的，需要是要在**首次到达「坐标对齐误差范围」内才开始连按 ↑**"✓）：
            `False` ⇒ 直接 `return False`（**一个 ↑ 都不打** ✗）。武装由调用方置
            （`update` 里"x 进 `tol_px` 那一刻"✓ —— 判据就是**同一把尺** ✓ 不新造容差 ✓）；
          · **武装的下一拍立刻打**（`_up_next_at is None` ⇒ 开打 ✓）—— 所以"首次进容差"那一拍
            就按上了 ✓（要的就是这个：**人一站到门口就开始试门** ✓ 别空等对齐完 ✗）；
          · ⭐ 之后一"下"接一"下"**直接连续发点按**（用户 2026-10-06 第二轮 ✓ 原话：
            "**取消传送点通行方式对齐的时候 ↑ 以「移动操作尝试间隔」按，直接连续发点按**"✓）：
            两"下"起点之间隔 `TAP_PERIOD_S`(180ms ✓ = 全项目那份点按周期 ✓ 不新造时长 ✓)，
            每"下"按 `TAP_ON_S` 就松 ✓（与 `TAP` 相**同一个时长口径** ✓）；
          · ⚠ **与 `retry_ms`「移动操作尝试间隔」无关** ✗（那个默认 3 秒 ⇒ 太稀 ✓ 用户实测 ✓）——
            它只再管进容差之后 `WAIT` 相的"补按 ↑" ✓（那条口径一个字没动 ✓）。
        ⚠ 与横向对齐**互不干扰** ✓：`update` 把两路并进**同一份回值**（`move` ＋ `dir` ✓）——
          消费端 `agent._climb_tick` 本来就是分开装配的（`_steer` 管横向、`dir` 管竖直 ✓
          同一拍可以都有 ✓ = 爬绳"朝绳走 + 按住 ↑"那条路 ✓）。
        ⚠ 这几下 ↑ **不算**「爬绳 / 下跳的执行器通知」✓：`agent.climbing_vertical()` 只认
          `ClimbJob` / `DropJob` ✗ ⇒ 走传送点时**不会**打开"能上绳梯"的口 ✓ —— **正是要这样**
          （用户 2026-09-27 定的"碰到绳子不许判在绳上"✓；走传送点更不该被判成在爬绳 ✗）。
        """
        if not self._up_armed:
            return False                                 # ⭐ 还没进过「坐标对齐误差范围」⇒ 不许按 ✗
        now = float(now)
        if self._up_tap_at is not None and (now - float(self._up_tap_at)) < TAP_ON_S:
            return True                                  # 这一下还按着 ✓
        self._up_tap_at = None                           # 按够 `TAP_ON_S` ⇒ 松手 ✓
        # ⭐ **一"下"接一"下"，直接连续发点按**（用户 2026-10-06 第二轮 ✓ 原话："**取消传送点
        #   通行方式对齐的时候 ↑ 以「移动操作尝试间隔」按，直接连续发点按**"✓）：
        #   两"下"的**起点**之间隔 `TAP_PERIOD_S`(180ms ✓) —— 全项目**同一份点按周期** ✓
        #   不新造时长 ✓（同爬绳对齐 / 走路微调那套节奏 ✓）。
        #   ⚠ **不再看 `retry_ms`** ✗：那是「移动操作尝试间隔」（默认 3 秒 ✓ 用户实测"太稀"✓）；
        #     它仍然管**进容差之后 `WAIT` 相的"补按 ↑"** ✓（那条口径一个字没动 ✓）。
        #   ⚠ "开头那一下"仍然**不等对齐好** ✓（`_up_next_at is None` ⇒ 立刻打 ✓）。
        if self._up_next_at is None or now >= float(self._up_next_at):
            self._up_tap_at = now
            self._up_next_at = now + TAP_PERIOD_S
            self._up_taps += 1
            return True
        return False

    # ---- 对外 ----

    def update(self, now, px, py=None, ladder_id=None, here_sets=None, at_top=None,
               at_bottom=None, ground_y=None, on_rope_pos=None, **kw):
        if self._t0 is None:
            self._t0 = float(now)
        if here_sets is None:
            here_sets = ()
        # ① 超时（口径同走路：这一段跑了多久 ✓）
        if self.timeout_s > 0 and (float(now) - self._t0) >= self.timeout_s:
            return self._fail("走传送点超时：这一段已经跑了 %.0f 秒（门 %s）"
                              % (self.timeout_s, getattr(self.portal, "pn", "?")))
        # ② 到达（集合优先 ✓ 出口门兜底 ✓）
        why = self._arrived(px, py, here_sets)
        if why:
            self.phase = self.DONE
            self.note = why
            return self._out(0)
        p = self.portal
        if p is None:
            return self._fail("这条传送点边没指名哪扇门（数据坏了 ⇒ 回编辑器选一个 ✓）")
        if px is None:
            # 定不了位就别乱走、更别按 ↑（按了可能把自己送到别处 ✗）
            self.note = "走传送点：拿不到世界坐标（小地图定位没有输出）"
            return self._out(0)

        dx = float(p.x) - float(px)
        # ⭐ **按 ↑ 只看 x 对齐**（⚠ 不再拿 `dy_ok` 当门 ✗）：这是**照搬爬绳那条口径** ✓ ——
        #   用户 2026-09-27 在爬绳上定的："**对齐了就按跳开始爬啊，不需要判定别的**"✓
        #   （原话，见 `ClimbJob._align_step` 里"按跳的许可条件"那段 ✓）⇒ 传送点同理：
        #   **x 对齐 + 稳够 `hold_ms` ⇒ 按 ↑** ✓，不再额外要求"人正在门的判定框里"✗。
        #   ⚠ y 差多少**照样写进 note** ✓：人在别的那层时按了也不动 ⇒ **必须从日志看得出来** ✗
        #     （用户 2026-10-06 就是盯着日志发现"左右晃却不按 ↑"的 ✓ 别再让他猜 ✓）。
        _dy = None if py is None else abs(float(p.y) - float(py))
        _y_tip = ("" if (_dy is None or _dy <= PORTAL_TOL_Y)
                  else "　⚠ 人 y=%.0f 与门 y=%.0f 差 %.0f px（> %d）：按 ↑ 只看 x ✓"
                       " 但这一层未必是门那层 ⇒ 按了可能不动 ✓"
                       % (float(py), float(p.y), _dy, int(PORTAL_TOL_Y)))

        # ③ ↑ 正按着（TAP 相）⇒ 按够 `TAP_ON_S` 就松手、进 WAIT ✓
        # ⚠⚠ **`dir` 的正负号是"↑ = +1"** ✗（权威在 `agent._climb_tick` 的消费点：
        #   `dk = km.get("up") if out["dir"] > 0 else km.get("down")` ✓ —— 与 `ClimbJob`
        #   的"向上爬 d=+1"同一套 ✓）。
        #   ⛔ 这里原来写的是 **`dir=-1`** ✗ ⇒ **一直按的是 ↓** ⇒ 用户报的"**站在传送点门口
        #     不按↑**" ✓（现场：走到门口就停住、什么都不发生 ✓ 因为发出去的是"↓"、
        #     而人正站在地上，看着就是"没按"✗）。**自检也没抓住** ✗ —— 因为用例里的期望
        #     （`o["dir"] == -1`）是**照着实现写的** ✗（把 bug 钉成了"期望"✓ 教训同 SKILL **273**）⇒
        #     现在用例改成断言 **`+1`** ✓（照**消费点**的口径，而不是照实现 ✗）。
        #   ⚠ 这里**只看"按够 `TAP_ON_S` 没有"**、不看"人还在不在容差里" ✓：走到一半滑出去也把
        #     这一下按完 ✓（老版下半段那条 ✓ 同一件事 ✓ 别拆成两处 ✗）。
        if self.phase == self.TAP:
            if (float(now) - float(self._tap_at or now)) < TAP_ON_S:
                return self._result(0, dir=1)      # 只按 ↑、不横挪 ✓
            self._tap_at = None
            self.phase = self.WAIT
            self._wait_at = float(now)
            self.note = ("已按 ↑（门 %s）—— 等到达「%s」%s"
                         % (getattr(p, "pn", "?"), self.dst_set, _y_tip))
            return self._out(0)

        # ④ 按过 ↑ 之后的等待窗：**位置状态还没回执 ⇒ 补按 ↑**（用户 2026-10-06 ✓ 同爬绳那套 ✓）
        #   "回执" = **已经到了那边**（上面 `_arrived` 已经判过 ✓）⇒ 走到这儿就是**还没到** ✓。
        #   ⚠ **不自己判"按了多久还没到"** ✗：老那版 `PORTAL_WAIT_S`(2.5s) 会在**第一次补按
        #     还没到点**时就把任务判死 ✗（补按间隔默认 3s > 2.5s ⇒ 等于**永远补不上** ✗✓）；
        #     现在与爬绳同一口径 —— **真过不去由「寻路超时时间」兜底** ✓（见类说明 ✓）。
        #   ⚠ 节拍 = 「**移动操作尝试间隔**」(`retry_ms` ✓ 用户 2026-10-06 追问确认 ✓)。
        if self.phase == self.WAIT:
            waited = float(now) - float(self._wait_at or now)
            _iv = self.retry_ms / 1000.0
            _last = float(self._press_at if self._press_at is not None
                          else (self._wait_at or now))
            if _iv > 0 and (float(now) - _last) >= _iv:
                self._press_at = float(now)
                self._retap_n += 1
                self.note = ("按过 ↑ 之后 %.1f 秒还没到「%s」⇒ **补按一次 ↑**"
                             "（第 %d 次；对面把键丢了也能救回来 ✓）%s"
                             % (waited, self.dst_set, self._retap_n, _y_tip))
                return self._result(0, dir=1, reassert=True)
            self.note = ("已经按过 ↑（门 %s），等到达「%s」（%.1fs%s）%s"
                         % (getattr(p, "pn", "?"), self.dst_set, waited,
                            "，补按过 %d 次" % self._retap_n if self._retap_n else "",
                            _y_tip))
            return self._out(0)

        # ⑤ **对齐 x** —— ⭐ **与爬绳同一处实现**（`align_x_decide` ✓）。
        #   用户 2026-10-06 原话："我们应该**整套照搬对齐式爬绳梯的逻辑**，**还要补按方向**"✓ ——
        #   现场症状："在从 5 到 4 的 h010 传送门**左右来回晃**、**但是没有按 ↑**"✗。
        #   三个要点（都在那个函数里，逐条有注释 ✓）：
        #     · 远（> `near_px`）⇒ 一口气按住；近 ⇒ 一下一下**点按**（按够 `TAP_ON_S` 就松 ✓）；
        #     · ⭐ **两次按下之间隔「对齐绳梯移动延迟」**（默认 180ms ✓）—— **过冲之后往回按
        #       也照样等** ⇒ **不会在目标两边来回晃** ✗✓（老版这里是"这一拍按一下、下一拍
        #       看方向又按回来"✗ ⇒ 人就在门口抖 ✓）；
        #     · ⭐ **补按方向**（"还要补按方向"✓）：按着某个方向却一直没进步 ⇒
        #       `reassert=True` ⇒ 调用方**再发一次 PRESS** ✓（对面把键丢了也能救 ✓）。
        #   ⭐⭐ **另一半：对齐期间**同时**"点按 ↑"**（用户 2026-10-06 追加的**传送点专属**口径 ✓
        #   原话："**增加传送点的特殊对齐逻辑：对齐过程开始时就可以按 ↑，并以「移动操作尝试
        #   间隔」连续点按 ↑**"✓）—— 门是**按 ↑ 才开**的 ✓，而原来"先对齐到 ±`tol_px`
        #   再按 ↑"等于把人**钉在门口**等对齐 ✗ ⇒ 现在**一进对齐就开打** ✓、之后每
        #   「移动操作尝试间隔」再打一下 ✓（节拍口径全在 `_up_tap_step` ✓）。
        #   两路（横向 / ↑）**同一拍一起发** ✓ —— 消费端分开装配（`_steer` / `dir` ✓）。
        kind, move, _re, why, st = align_x_decide(
            now, dx, press_dir=self._alx_dir, press_at=self._alx_press_at,
            progress_at=self._alx_progress_at, best_dx=self._alx_best_dx,
            gap_ms=self.align_gap_ms, near_px=self.near_px, tol_px=self.tol_px,
            retry_ms=self.retry_ms)
        self._alx_dir = st["press_dir"]
        self._alx_press_at = st["press_at"]
        self._alx_progress_at = st["progress_at"]
        self._alx_best_dx = st["best_dx"]
        self._alx_reasserts = st["reasserts"]
        # ⭐⭐ **"武装" ↑ 的点按**（用户 2026-10-09 ✓ 见 `__init__` 那段原话）：
        #   **对齐判据自己说"进容差"了**（`kind == "in_tol"` ✓）才起拍 —— 用的是**同一把尺**
        #   `tol_px`（= 设置里的「坐标对齐误差范围」✓），**不另判一个数** ✗；
        #   置上之后**本轮不再回退** ✓（过冲出了容差 ↑ 照旧连着打 ✓ —— "已经在门口试上了别断"✓）。
        if kind == "in_tol":
            self._up_armed = True
        _up = self._up_tap_step(now)        # ⭐ 这一拍要不要点按 ↑（见方法说明 ✓）
        if kind == "in_tol":
            # 对齐好了 ⇒ **站住稳 `hold_ms`** 再按那一下「正式的 ↑」（照搬爬绳"站住等 hold_ms
            #   ⇒ 按跳"✓）：⚠ 不做这一道就是"路过一下也按 ↑"✗（过冲经过门口那一瞬也按 ✓）。
            # ⚠⚠ 但**稳住的这几拍不再把 ↑ 停下** ✗（用户 2026-10-06 新口径："**对齐过程**
            #   开始时就可以按 ↑" ✓ —— 稳住也是对齐过程的一部分 ✓）：继续按 `_up_tap_step`
            #   的节拍点按 ✓；门真在这一下开了 ⇒ 上面 ② 的到达判据当场接走 ✓。
            if self._in_tol_since is None:
                self._in_tol_since = float(now)
                self.note = ("%s（站住稳 %d ms；对齐期间一直在点按 ↑，已打 %d 下）%s"
                             % (why, self.hold_ms, self._up_taps, _y_tip))
                if self.hold_ms > 0:
                    return self._result(0, dir=1) if _up else self._out(0)
            elif (float(now) - float(self._in_tol_since)) * 1000.0 < self.hold_ms:
                self.note = ("对齐好了，保持中（差 %.0f px；对齐期间点按 ↑ 已打 %d 下）%s"
                             % (dx, self._up_taps, _y_tip))
                return self._result(0, dir=1) if _up else self._out(0)
            self._tap_at = float(now)
            self._press_at = float(now)            # 补按 ↑ 的节拍从这一下起算 ✓
            self.phase = self.TAP
            self._best_d = self._best_at = None
            self.note = ("走到门 %s 了（差 %.0f px ≤ 容许 %d，已稳 %d ms）⇒ 点按一下 ↑%s"
                         % (getattr(p, "pn", "?"), abs(dx), self.tol_px,
                            self.hold_ms, _y_tip))
            return self._result(0, dir=1)
        # ---- 还没对齐：站住等的那几拍**别把"进容差"那次记账留着** ✗（不然下次进容差秒按 ✓）----
        self._in_tol_since = None
        # 走不动就如实报（被墙/台阶挡住、或者门不在这层 ✓ —— 老版那条兜底，留着 ✓）
        self._best_d, self._best_at, _stalled = self._stalled_since(
            now, abs(dx), self._best_d, self._best_at, self.stall_s,
            lambda v, b: v < b - 1.0)
        if _stalled:
            return self._fail(
                "走传送点走不动了：卡在 x=%.0f（门 %s 在 x=%d，还差 %.0f px）"
                "—— 可能被墙/台阶挡住，或者这扇门不在你站的这层%s"
                % (float(px), getattr(p, "pn", "?"), int(p.x), abs(dx), _y_tip))
        # ⚠ 说明里**报的是「点按周期」**（用户 2026-10-06 第二轮 ✓）：现在是**连续发点按** ✓
        #   —— 节拍与「移动操作尝试间隔」**无关** ✓（那个只再管 `WAIT` 相的补按 ↑ ✓）。
        _cad = "每 %d ms 一下（连续发点按）" % int(TAP_PERIOD_S * 1000.0)
        self.note = ("%s（对齐期间点按 ↑：%s，已打 %d 下）%s"
                     % (why, _cad, self._up_taps, _y_tip))
        if kind == "press":
            # ⭐ 横向「press」与 ↑ 的点按**同一拍一起发** ✓（`dir=1` 叠加在 `move` 上 ✓）
            return (self._result(move, dir=1, reassert=_re) if _up
                    else self._result(move, reassert=_re))
        # "wait"（横向这一拍不按：点按松开 / 等移动延迟 ✓）—— ↑ 的点按**照旧**可能要发 ✓
        return self._result(0, dir=1) if _up else self._out(0)

    def retry(self):
        """失败后重新激活（同上绳/下跳：回到起始状态重来一遍 ✓）。"""
        self.attempt += 1
        self.phase = self.WALK
        self._t0 = None
        self._best_d = self._best_at = None
        self._tap_at = self._wait_at = None
        self._press_at = None            # 补按的节拍锚也清掉 ✓（新一轮从头算 ✓）
        self._retap_n = 0
        # 对齐相的记账**全部**清掉 ✓（新一轮从头对齐 ✓ —— 留着"上一轮已经按着哪边/稳了多久"
        # 就是脏状态：会一上来就判"早按够了"⇒ 立刻按 ↑ ✗）
        self._alx_dir = 0
        self._alx_press_at = None
        self._alx_progress_at = None
        self._alx_best_dx = None
        self._alx_reasserts = 0
        self._in_tol_since = None
        # 对齐期间"点按 ↑"的记账也清干净 ✓ —— ⚠ **`_up_armed` 必须一起复位** ✗：
        #   新一轮要**重新走到容差里**才起拍（用户 2026-10-09 的口径 ✓ 见 `__init__` 那段）——
        #   留着旧的 True ⇒ 新一轮在远处又一路按 ↑ ✗（正是今天要修的那个毛病 ✓）。
        self._up_armed = False
        self._up_tap_at = None
        self._up_next_at = None
        self._up_taps = 0
        self.note = ("第 %d 次尝试：走传送点「%s」（门 %s）"
                     % (self.attempt, self.dst_set, getattr(self.portal, "pn", "?")))
        return self._out(0)

    def cancel(self, why="取消"):
        self.phase = self.FAILED
        self.note = why
        return self._out(0)
