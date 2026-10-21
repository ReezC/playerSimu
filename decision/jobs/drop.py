"""下跳执行器 `DropJob`（从 `decision/route.py` 切出）。"""

from decision.jobs.base import _PausableJob
from decision.jobs.params import (ALIGN_TOL_PX, ARM_HOLD_S, DEFAULT_TIMEOUT_S,
                                  PAUSE_MIN_S, STALL_S, TAP_ON_S, TAP_PERIOD_S)


#: 「人**站在那块面上**吗」的判据：`|玩家 y − 脚下那块面的 y|` 超过它就说明**人悬着**
#: （挂在绳上 / 在空中 ✗），而不是站在面上 ✓ —— 下跳的「被绳吸住」判据用它 ✓。
#: 取值来历：**与 `foothold_below` 的 `tol` 默认值 40 同口径** ✓（那个 40 的语义就是
#: "黄点画的是玩家中心、而地面在脚底"的那点偏差 + 几像素读数抖动 ✓，见
#: `core/mapdata.Terrain.find_below` 的说明 ✓）⇒ 不新造数 ✓。
FOOT_GAP_PX = 40.0


class DropJob(_PausableJob):
    """一次「下跳」（`drop`）：**就近平齐到某个可下跳 foothold 的中心** → 站住 →
    按住 ↓ + **点按跳** → Y 开始变化就**松开 ↓** → 落到目标算成功。

    用户 2026-09-27 定的流程（照抄，别自己发明顺序）：

      ① 找**最近**的允许下跳的 foothold，走到它的**中心**；
      ② 判定**位于该 foothold**（x 落在它的范围里）持续 ≥「坐标对齐误差时间」⇒ 松开行走键；
      ③ **按住 ↓ + 点按跳**；
      ④ **Y 向下变化超过「坐标对齐误差范围」⇒ 松开 ↓**（松手，等落地 ✓）——
        用户 2026-09-27 明确："**当 Y 变化超过「坐标对齐误差范围」才能松开 ↓**"：
        **向上抖不算**（那是读数噪声，不是"开始下落"✗），而且**没有 y 读数就不按跳**
        （基准都拿不到 ⇒ 判不出下落 ⇒ 只能一路按到超时 ✗）；
      ⑤ 落到目标 foothold ⇒ 成功。

    ⭐⭐ **③ 之后三条出口（用户 2026-10-06 的流程图定稿 ✓，照抄别自己发明）**：

      · **位置状态说上绳了** ⇒ 松开 ↓、按方向 + 跳 **脱离**（→ `DETACH` 相 ✓，
        方向 = 有锁定目标朝目标、否则任意 ✓）；
      · **正常下落**（Y 真的往下走了 ✓）⇒ 离开"触发"这一环，转 `LANDING`/到达判定 ✓；
      · **「移动操作尝试间隔」内没反应**（既没下落、也没上绳）⇒ **回入口**：
        重新判"**当前这块 foothold 还能不能下跳**"（= 回到 `ALIGN` 相 ✓，
        `_to_entry` 那一步；判下来还是这块 ⇒ 重新"按住 ↓ + 跳" ✓，
        不是这块了 ⇒ 就近挑一块走过去 ✓）。

    ⛔ **"按键补按（重发 PRESS）"那套已删**（2026-10-06 ✓ 用户原话："**下跳 drop 执行器的
      按键补按逻辑移除**，改为当在「移动操作尝试间隔」内没反应就从最开始的判断走"✓）：
      `_reassert_at` / `_reassert_n` / `_hold_reassert()` / `out["reassert"]` 全部不在下跳里 ✗
      —— **爬绳那边的「补按 ↑」照旧保留**（那是另一条口径 ✓）。
    ⚠ 回入口那一拍**一个键都不按**（`_out(0, False)` ⇒ ↓ 真的松开一拍 ✓）：`KeyState` 只在
      **键集变化**时才发键 ⇒ 不松这一下、下一轮那个"按住 ↓"对面**根本收不到新的按下** ✗
      （人**不趴** ⇒ 后面每一下跳都只是**原地跳** ✓）—— 这正是原来 `_hold_reassert` 要解决的
      同一个坑，现在改用"**真的松一下 → 真的一次新按下**"来解（**不靠重发** ✓）。
    ⚠ 「点按跳」而不是"按住跳"：用户 2026-09-26 定的动作是**两次**按下（先 ↓ 再跳），
      一直按住跳会被游戏当成"还挂在上面"✗（`_jump_down_beat` 那套也是点按 ✓）。

    可下跳的 foothold 来自边上的 `footholds` 那一格（不填 = 起点集合的全部，见
    `zones.drop_footholds`）—— 一条边上常常有好几个能下去的点，**就近挑一个** ✓。

    接口和 `ClimbJob` **故意做成一样**（`update/retry/cancel/attempt/phase`）：
    agent 那边就只有一条执行路径（`_climb_tick`），不用为每种通行方式各写一遍
    —— 也保证两种的失败保护/重来行为一致。
    """

    ALIGN = "align"          # ① 就近平齐到某个可下跳 foothold（要**站在它上面**够久）
    ARMED = "armed"          # ② **先按住 ↓**（这一刻还不跳）
    DROP = "drop"            # ③ ↓ 按住不放 + **点按跳**（每次尝试一个"尝试间隔"的窗口）
    LANDING = "landing"      # ④ **松开 ↓**，等落到目标
    #: ⓪' **被绳吸住 ⇒ 脱离**（用户 2026-09-27）：下跳途中吸附到"通往下层的绳"上 ⇒
    #: **按住一个水平方向（= 松手）+ 连按跳**，把 ↓ **松开**（按 ↓ = 还挂在绳上 ✗）。
    #: 见 `_detach_step`。
    DETACH = "detach"
    DONE = "done"
    FAILED = "failed"

    def __init__(self, src_set, dst_set, spots, tol_px=ALIGN_TOL_PX, hold_ms=250,
                 y_tol_px=ALIGN_TOL_PX, retry_ms=3000, stall_s=STALL_S,
                 timeout_s=DEFAULT_TIMEOUT_S, dst_ys=None):
        self.src_set = str(src_set or "")
        self.dst_set = str(dst_set or "")
        #: ⭐ **目标平台的"面 y"**（世界坐标 y 的列表；空 = 不知道 ✓）—— 给"到了没有"的
        #: **几何兜底**用：**脚下这块面就是目标平台的面** ✓（读位置状态广播的 `ground_y` 判 ✓）。
        #: 用户 2026-09-28 明确要求换掉旧判据（原话："把『y 掉 40px』换成『脚下这块面 ==
        #: 目标面』"）：旧的那条**太松** —— 人只要往下掉 40px 就报"到了"，哪怕还在半空 /
        #: 落到了别处 ✗（实测：追击反复下发同一条下跳任务，就是它误报"到了" ⇒ 人其实没
        #: 下去 ⇒ 下一拍又判"怪不在我这块平台" ⇒ 又下 ✗）。
        #: ⚠ **目标集合里每条非墙 foothold 的面 y 都收进来**：运行时落点取决于从哪个下跳点
        #:   跳，这里**不预先绑定**那一条 ✓（命中任意一条即算 ✓）。
        self.dst_ys = [float(v) for v in (dst_ys or [])]
        #: [(中心 x, 左, 右, **面 y**, foothold id)] —— 可下跳的那几条（就近挑）。
        #: 形状**统一成 5 元组**（与 `WalkJob` 的"落点五元组"同款 ✓）：面 y 是
        #: `f.y_at(中心 x)`（与 `_set_spans` 同一口径 ✓），给 ALIGN 的"**人是不是站在
        #: 这一块上**"用 —— 用户 2026-10-02 现场（下跳 ALIGN **只看 x** ⇒ 人掉到别的层
        #: 也被判成"已经在下跳点上" ⇒ 从错误的层按 ↓+跳 ⇒ 一路穿到最底层 ✗）。
        #: ⚠ 传**老的 4 元组**（手写用例 / 老调用方）⇒ 面 y 记 `None` ⇒ 那一条**不参与
        #:   y 校验**（"判不出来不拦"，与 `ground_y is None` 那条**同一个口径** ✓
        #:   老行为一字不变 ✓）；形状既不是 4 也不是 5 ⇒ 报错（别猜 ✓）。
        #: ⭐⭐⭐ **"按 ↓ 被绳吸走"过的下跳点 id**（用户 2026-10-10 ✓ 原话："**现在下跳（drop）
        #:   每次都会被卡在绳子上**"✗）—— ALIGN 挑点时**先挑别的点** ✓，全都被记过才回来用它 ✓。
        #:   为什么这么修（现场复盘 ✓ `behavior.log` 07:21:12 那一趟 ✓）：
        #:     人**规规矩矩站在下跳点上**（`y` 就在平台面上 ✓）⇒ 进 ARMED 按 ↓ ⇒
        #:     **那根绳就在手边** ⇒ 这游戏把 ↓ 当成"**抓绳**"✗ ⇒ 人被吸上绳 ⇒ 执行器走
        #:     ①档 `_hanging` ⇒ DETACH 脱绳 ✓ ⇒ 脱完**又回到同一个点**（它离得最近 ✓）
        #:     ⇒ **再按 ↓ ⇒ 再次被吸** ✗✗ ⇒ 肉眼就是"每次下跳都卡在绳子上" ✓。
        #:   ⚠ **为什么不去动"吸住判据"**（把"在绳段里 + ARMED/DROP"也判成吸住 ✗）：那条
        #:     2026-10-06 **试过又撤了** ✓ —— 分不出"人被吸在绳上"和"人站在绳底平台上" ✓
        #:     （我们**自己按着的 ↓** 会把"在绳上"这个信号打开 ✗ 见 `_hanging` 那段 ✓）。
        #:     换个下跳点是**不需要新信号**就能治本的一招 ✓（那个点的几何本来就不好 ✓）。
        #:   ⚠ 只记**这一趟任务**（运行时账、不持久化 ✓ 同 `_mob_h_samples` ✓）：绳/地形的
        #:     几何是"这一段任务"的事 ✓，而且换个平台后同一个 fid 不一定还贴着绳 ✓。
        self._bad_spots = set()
        self.spots = []
        for _s in (spots or []):
            _s = tuple(_s)
            if len(_s) == 4:
                _c, _a, _b, _i = _s
                self.spots.append((float(_c), float(_a), float(_b), None, str(_i)))
            elif len(_s) == 5:
                _c, _a, _b, _y, _i = _s
                self.spots.append((float(_c), float(_a), float(_b),
                                   None if _y is None else float(_y), str(_i)))
            else:
                raise ValueError(
                    "可下跳点的形状该是 (中心x, 左, 右, 面y, id)（或老的 4 元组）：%r" % (_s,))
        if not self.spots:
            raise ValueError("下跳任务没有可下跳的 foothold")
        #: 站在 foothold 范围两侧允许的**读数余量**（世界像素，见 `update` 的 ALIGN ✓）
        self.tol_px = max(1.0, float(tol_px))
        #: 「坐标对齐误差时间」（ms）：判定**位于该 foothold** 要持续这么久才松手 ✓
        self.hold_ms = max(0, int(hold_ms))
        #: 「坐标对齐误差范围」（世界像素）：Y 变化超过它才算**真的开始下落** ⇒ 松开 ↓ ✓
        self.y_tol_px = max(1.0, float(y_tol_px))
        #: ⭐ **下落过程中「补按跳」**的锚（用户 2026-09-28 口径 ✓ 原话："下落过程中也按照
        #:   『移动操作尝试间隔』补按 **按住 ↓ + 跳**"）—— 见 `_falling_jump` ✓。
        #: ⛔ **曾经还有一个 `_reassert_at`（"按下 ↓"的重发观察窗 ✓）—— 2026-10-06 删掉** ✗：
        #:   用户改口径了（原话："**按键补按逻辑移除**，改为当在「移动操作尝试间隔」内没反应
        #:   就**从最开始的判断走**（当前 foothold 是否可下跳）"✓）⇒ "没反应"的处置**不再是
        #:   重发一次 PRESS**，而是**回入口重判**（`_to_entry` ✓）。见 `update` 的 DROP 相 ✓。
        self._jump_reassert_at = None
        #: **已经在往下掉了**（Y 变化越过误差范围 ✓）：从这一刻起**不再点按跳**，
        #: 但 **↓ 一直按着** —— 直到 `_arrived` 说"位置状态到目的地了"才松 ✓
        #: （用户 2026-09-27："在 drop 时，除非位置状态到了目的地，否则 ↓ 是不能松开的" ✓）。
        self._falling = False
        #: 「移动操作尝试间隔」（ms）：③ 之后这么久 Y 还没动 ⇒ 补发一个"松开 ↓"重来 ✓
        #: 由 `agent.start_climb()` 从设置灌进来（0 = 不重试 ⇒ 老行为 ✓）。
        #: ⚠ 2026-09-27 起它还兼一个角色：**松 ↓ 之后隔多久再开始下一轮**（见 `_next_round_at` ✓）。
        self.retry_ms = max(0, int(retry_ms))
        #: 「**卡住判定时长(秒)**」（设置 → 判定参数 ✓）：一轮"按住 ↓ + **连按跳**"最长按多久 ——
        #: 到点 Y 还没动就算卡住 ⇒ 松开 ↓ 起身，隔「移动操作尝试间隔」再连按 ✓。
        #: 为什么用它：用户 2026-09-27 原话"按住↓，下一拍再按跳，**可以用「卡住判定时长」连按**" ✓。
        #: 下限 0.5：再小就等于"刚按一下就判卡住"✗（同 `ClimbJob.stall_s` 的钳法 ✓）。
        self.stall_s = max(0.5, float(stall_s))
        self.timeout_s = float(timeout_s)
        #: 第几次尝试 —— ⚠ **不是上限**（没有"最多试几次"了，见文件头那段 ✓）
        self.attempt = 1
        self.dir = -1                       # 下跳永远是"往下"（按住 ↓）
        #: 上一拍 `_out` 请他按的**竖直键**（-1 = ↓ / 0 = 没按 ✓）—— 给 ladder_id 许可的
        #: 「执行器通知」用（同 `ClimbJob._last_vert` ✓，用户 2026-09-28 诉求 1：drop 按 ↓ 时
        #: 也要给位置状态机发通知，让它能进"能上绳梯"的入口，直到 drop 收工再关上 ✓）。
        self._last_vert = 0
        self.phase = self.ALIGN
        self.note = "就近平齐到一个可下跳 foothold"
        self._t0 = None
        #: 被战斗打断时，`_PausableJob.interrupted/_resume` 要把**这些计时锚点**整体往后挪
        #: （= "任务的时间停了一会儿" ✓）—— 挨打那几秒不许算进"站住/按住↓/卡住"里 ✗。
        #: ⚠ 2026-09-27 之前 DropJob **没继承** `_PausableJob`（所以它从不跳过打架时长 ✗，
        #:   还吃不到 `_result` 那套统一回值 ✗ —— 本轮补齐 ✓）。`_y0/_y_base` 是 **y 值**不是
        #:   时刻 ⇒ 不在里面 ✓。
        #: ⛔ **`_reassert_n`（补按「按住 ↓」发了几次）也一起删了**（2026-10-06 ✓ 补按没了，
        #:   这个计数就没有来源了 ✗；日志里 `drop_reassert` 那条 `sample` 同步删 ✓）。
        self._ANCHORS = ("_in_tol_since", "_armed_at", "_attempt_at", "_jump_at",
                         "_landing_at", "_next_round_at",
                         "_jump_reassert_at", "_detach_tap_at")
        self._in_tol_since = None
        self._armed_at = None
        self._y0 = None
        self._pick = None
        #: 本次尝试（一次"按住 ↓ + 点按跳"）的起点时刻 / 起点 y / 跳按下的时刻 / 第几次
        self._attempt_at = None
        self._y_base = None
        self._jump_at = None
        self._tap_n = 1
        #: 松开 ↓ 的那一刻（"松了却没落下去"的退路计时用：超 `retry_ms` ⇒ 回入口 ✓）
        self._landing_at = None
        #: **下一轮**最早什么时候开始（= 松 ↓ 那一刻 + 「移动操作尝试间隔」✓）—— 不是 None 就说明
        #: 现在正"松着 ↓ 等间隔"（那几拍**一个键都不按**，让人真的站直 ✓）。
        self._next_round_at = None
        #: 「**被绳吸住、正在脱离**」那一下跳是什么时候按下去的（到 `TAP_PERIOD_S` 再按一下 ✓
        #: —— 和 ③ 那份点按节奏同一份，不新造时长 ✓）。见 `_detach_step`。
        self._detach_tap_at = None
        #: ⭐⭐⭐ **"跳下绳"的第一步（方向键先单独占一拍）发过没** —— 用户 2026-10-10 ✓
        #:   原话："**应该是需要再判定上绳梯后执行跳下绳的操作**"✓。
        #:   口径**照抄 `ClimbJob._jump_down_beat`** ✓（那儿写着理由：同一次 tick 里把"方向"和
        #:   "跳"一起发，**游戏里未必算"按住方向再跳"** ✗ —— `KeyState` 一次下发多个键
        #:   **不保证顺序** ✓）⇒ 必须分两步发 ✓。
        self._detach_pressed = False
        #: 「点按跳」的**上升沿**计数 + 上一拍跳的开关（`tap_count()` 用 ✓）。
        #: 用户 2026-09-28 报"**趴在那不动了也不按跳**"：光看 `drop_phase=3` 分不清"跳没发"
        #: 还是"发了游戏不响应" ✗ ⇒ `behavior.log` 里的 `drop_taps` 一眼就能判 ✓。
        self._taps = 0
        self._jump_on = False

    def update(self, now, px, py=None, ladder_id=None, here_sets=None, at_top=None,
               at_bottom=None, ground_y=None, on_rope_pos=None,
               climb_failed=None, climb_stalled=None):
        _paused = self._resume(now)      # 被战斗打断过 ⇒ 把挨打那几秒从计时锚点上扣掉 ✓
        if self._t0 is None:
            self._t0 = now
        if self.phase in (self.DONE, self.FAILED):
            return self._out(0, False)
        if now - self._t0 > self.timeout_s:
            return self._fail("超时 %.0fs：没能从「%s」下跳到「%s」"
                              % (self.timeout_s, self.src_set, self.dst_set))

        # ---- ⭐⭐ 被战斗打断过 ⇒ **从最开始的判断重来**（用户 2026-10-08 ✓ 原话：
        #   "下跳 drop 执行器在需要进 attack 状态时，需要被切断，等 attack 结束后再重头开始
        #    （最开始的是否可下跳判断）"）----
        # 口径（三条 ✓）：
        #   · **切断**：进 attack 那几拍 `agent.tick` 根本不跑任务 ⇒ 它本来就不发键 ✓
        #     （`_PausableJob` 那套 ✓ 见文件头 / `interrupted()` ✓）；
        #   · **重头**：回来这一拍**不接着上次跑** ✗ ⇒ 走 `_to_entry` ✓ = 相位打回 `ALIGN`
        #     （就是流程图顶部那个「**当前 foothold 是否可下跳**」✓）+ 运行时状态全清
        #     （`_pick = None` ⇒ **重新就近挑**一块可下跳的 foothold ✓ 那可是关键一步：
        #      打架这几秒里人可能被打走位、原来那块已经不是最优/已经不可下跳了 ✗）；
        #   · **那一拍一个键都不按** ✓（`_to_entry` 的返回值就是 `_out(0, False)` ✓
        #     —— 刚从战斗里出来先把人站稳，下一拍再按流程走 ✓）。
        # ⚠ 阈值与 `ClimbJob` **同一条口径**（`PAUSE_MIN_S` ✓ 比帧间隔大一个数量级 ⇒
        #   正常拍间隔不算打断 ✓），但**落点必须分开写** ✗：爬那边是"回 `ALIGN` + `_reset_round`"
        #   （它要保住"已经在绳上"那条判定 ✓），下跳这边是**回 `_to_entry`**（要**重新挑** ✓）。
        if _paused >= PAUSE_MIN_S:
            return self._to_entry(
                "被战斗打断 %.1fs ⇒ 回入口：重判「当前 foothold 是否可下跳」" % _paused)

        # ---- ⓪ **吸附在绳上 ⇒ 先脱离**（用户 2026-09-27 要求）----
        # 原话："现在在下跳执行器中可能会吸附到通往下层的绳子上，这种情况应该执行
        #   「**任意方向按住 + 连按跳**」脱离（**世界状态应该是能知道的**）"。
        # ⭐ 判据 = **两份"只看位置"的客观事实**（都是世界状态 ✓、**都与按键许可无关** ✓）：
        #   · `on_rope_pos` —— 位置就落在某根绳的**绳段**里（`pos_state._ladder_at` 那套
        #     几何 ✓，感知层同一次算出来的 ✓）；
        #   · `ground_y is None` —— **脚下没有面**（悬空 ✓）。
        #   两件同时成立 = 人**挂在绳上**（"被绳吸住"的字面事实 ✓）。
        # ⛔ **2026-09-28 修：原来只判「带按键许可的 `ladder_id`」⇒ 两头都错** ✗
        #   （用户现场："**下跳 drop 结果实际爬上了向下爬的绳子**"✗）：
        #     · **误报**：`ARMED` / `DROP` 相**自己按 ↓** ⇒ `climbing_vertical()` 为真 ⇒ 许可开
        #       ⇒ 人只要站在下跳点（那根"通往下层的绳"就在旁边；`LADDER_DX`=**24** /
        #       `LADDER_PAD`=**12** 一把**宽**尺盖得住 ✓）⇒ `ladder_id` **当场非空** ⇒ 进
        #       `DETACH`（现场：`align → armed → detach` 只隔 **28 ms** ✗）；而 `DETACH` 相
        #       **不按 ↓**（`vert=0` ✗）⇒ 下一拍许可关 ⇒ "又没在绳上" ⇒ 回 `ALIGN` ⟲；
        #     · **漏报**：人**真挂在绳上**时，`ALIGN` / `DETACH` 相**不按竖直键** ⇒ 许可关
        #       ⇒ `ladder_id` 恒 `None` ⇒ **判不出来** ✗ ⇒ 人一直挂在绳上（`behavior.log`：
        #       `climb_y=-90.7` 恒定 + `climb_on_ladder=0` 近 **9 秒** ✗）⇒ `ALIGN` 走不到
        #       下跳点中心 ⇒ `align→armed` 再也不发生（那 9 秒里**一条 `route_phase` 都没有**
        #       ✗）⇒ 10 秒「寻路超时」⇒ 重下 ⇒ 再超时 ⟲。
        #   ⚠ 这两份事实**本来就有**（`update` 的参数里都带着 ✓）—— 不新造感知 ✗、
        #     也不加"人到底在不在绳上"的第三份判断 ✓。
        # ⚠ 放在**所有相之前**：被吸住时"往下穿"这件事根本做不了（人挂在绳上 ✗），
        #   不论当时在哪一相（走去下跳点 / 站住 / 按压跳）都得先松手 ✓。
        # ⚠ **"脚下那块面离我有多远"才是"悬空"的判据**（2026-09-28 修 ✓）：
        #   ⛔ 原来写的是 `ground_y is None` ✗ —— 实测**不成立**：人挂在绳上时，感知照样给得出
        #     `ground_y`（`foothold_below` 只对"头顶上方"卡 40px、**下方不卡** ⇒ 绳下方那层
        #     的面会被认领 ✗）。实测（图 `106010105` 绳 L2）：人挂在绳正中 `y=-291`、离下面
        #     那块面还有 **109px**，`ground_y` 仍是 `-182` ✗ ⇒ `_hanging` 永远 False ⇒
        #     从不脱离（用户现场："drop 执行器没有从位置状态机拿到上绳通知而去执行跳开绳子的
        #     逻辑"✗）。
        #   ⇒ 改成"**位置在绳段里** 且 **脚下那块面离我超过容差**" ✓：人真站在平台上时
        #     `|py - ground_y|` 只有几像素（就是"黄点画中心 vs 地面在脚底"那点偏差 ✓
        #     `FOOT_GAP_PX` 与 `foothold_below` 的 40 同口径 ✓）。
        _foot_gap = (None if ground_y is None or py is None
                     else abs(float(py) - float(ground_y)))
        # ⭐⭐ **"被绳吸住"要分两档判**（用户 2026-10-06 ✓ 现场原话："**drop 触发 → 人上绳梯 →
        #   位置状态显示到了 fh：6 → 然后就不动了**"✓ —— 我按他的现象回查日志，完全对上 ✓）：
        #   ① 老那档（2026-09-28 那条 ✓ 不动它）：**在绳段里 且 脚下那块面离我超过
        #      `FOOT_GAP_PX`** ✓ = 人悬在绳中 ✓；
        #   ② ⭐ **新那档**：**在绳段里 且 相位已经是"要往下跳"那几相**（ARMED / DROP /
        #      LANDING ✓）—— 这几相的意图就是**往下跳**，人却在绳段里 ⇒ 说明**游戏把按住的 ↓
        #      当成"顺绳往下爬"、把他吸在绳上了** ✗。
        #      ⚠ 为什么非加这一档（实测证据 ✓）：人挂在**绳梯下部**时，脚下那块面就在 40px 内
        #      ⇒ ① 那档**判不出来** ✗ ⇒ 从不脱离 ⇒ 现场 `drop_phase=5`（DETACH）**一次都没进过** ✓
        #      （与日志完全吻合 ✓）。
        #   ⚠ **只在这几相**放宽 ✗：ALIGN 相（走去下跳点）路过绳段不算被吸住 ✓ 不许乱脱 ✗。
        #   ⚠⛔ **"在绳段里 + 那几相"这一档试过、又撤了**（2026-10-06 ✓ 如实记）：`on_rope_pos`
        #     和 `ladder_id` **两种信号都分不出**"人挂在绳梯上"与"人站在绳底平台上" ✗ ——
        #     **下跳点本来就常紧贴绳底**（`on_rope_pos` 真 ✓），而 ARMED/DROP 那几拍我们**正按着 ↓**
        #     ⇒ 许可开着 ⇒ 站在绳底平台上时 `ladder_id` **也报得出绳号** ✗ ⇒ 加这一档会把
        #     "规规矩矩站在下跳点上、准备往下跳"判成"被绳吸住" ⇒ 自己按的 ↓ 把自己判死 ✗
        #     （既有用例 `t_drop_detach_ladder` 当场按住 ✓ 它钉的就是这条 ✓）。
        #   ⇒ **要治"人被吸在绳梯上"，先得有一个能分辨这两者的信号** ✗（见开发日志里那条待办 ✓）。
        # ⭐⭐ 2026-10-16 **ARMED 那一档（用户口径"乙 + Y"✓）**：几何符合 **且** 我们**正按住 ↓**
        #   **且「按住之后 Y 真的在变」** ⇒ 才算"被绳吸住"✓。
        #   · 为什么要这一档（用户 2026-10-16 原话："**在 ARMED 刚开始时，就在几何上允许被判
        #     被绳吸住**"✓）：人一进 ARMED 本来就是"站在下跳点上、绳在旁边"✓ —— 若**只**按几何判，
        #     就会重现当年那次死循环 ✗（钉子 `t_drop_detach_ladder` 的 ⑦ 原文 + 现场：
        #     `align → armed → detach` **只隔 28 ms** ✗，成因"**自己按的 ↓ 把自己判死**"✓）；
        #   · ⇒ **"Y 在变"才是"人真被绳带着往下走"的客观证据** ✓（人**站着**时 Y 一动不动 ✓）；
        #     `_y0` 是 ARMED 进入那一刻记下的出发 y ✓（`py` 每拍刷新 ✓ 见上面那段 ✓），
        #     容差复用 **「坐标对齐误差范围」** ✓（`self.y_tol_px` ✓ —— 不新造尺子 ✗）；
        #   · ⚠ 只在 **ARMED** 这一相放宽 ✗：`ALIGN` 相（走去下跳点）**只是路过绳段**不算被吸住 ✓
        #     （同上面那段 ⚠ 的口径 ✓ 不许乱脱 ✓）；`DROP` / `LANDING` 相按下面 ① 的老档走 ✓。
        _dy_armed = None
        if py is not None and self._y0 is not None:
            _dy_armed = float(py) - float(self._y0)
        _hanging_armed = bool(
            on_rope_pos) and self.phase == self.ARMED and _dy_armed is not None \
            and abs(_dy_armed) > float(self.y_tol_px)
        _hanging = (bool(on_rope_pos) and (_foot_gap is None or _foot_gap > FOOT_GAP_PX)) \
            or _hanging_armed
        if _hanging:
            if self.phase != self.DETACH:
                self.phase = self.DETACH
                self._detach_tap_at = None
                self._detach_pressed = False      # "跳下绳"的第一步重新开始 ✓
                # ⭐ 记下"**这个下跳点按 ↓ 会被吸绳**"⇒ ALIGN 下次**先挑别的点** ✓
                #   （用户 2026-10-10 ✓ 治"每次下跳都卡在同一个点上"那个循环 ✓ 见
                #    `_bad_spots` 那段说明 ✓）
                self._mark_pick_bad()
                self.note = "在下跳中被绳吸住了（%s）⇒ 按住方向 + 连按跳 脱离" % on_rope_pos
            return self._detach_step(now, px)
        if self.phase == self.DETACH:
            # 已经不在绳上了（出了绳段 **或** 脚下有面 = 落回平台 ✓）⇒ 回入口重走一遍下跳
            #（重新就近对齐 → 站住 → 按压跳 ✓）
            self.phase = self.ALIGN
            self._in_tol_since = None
            self._pick = None          # 重挑下跳点：人可能已经被绳子带到别处了 ✓
            self._armed_at = None
            self._attempt_at = None
            self._y_base = None
            self._jump_at = None
            self._landing_at = None
            self._next_round_at = None
            self._detach_tap_at = None
            self._detach_pressed = False
            self.note = "已经从绳上松开 ⇒ 回入口重走下跳"

        # ---- ⭐⭐⭐ **位置状态说人站在一块「跟本任务无关」的面上 ⇒ 回入口**（用户 2026-10-11 ✓
        #   原话："**下跳 drop 过程中，如果位置状态返回了非预期的 fh 变化，那么需要从执行器
        #   入口开始重启 drop 任务**"✓；现场："**从 2→1 过程中意外到了底层，然后就一直趴着**"✗）
        # 为什么原来会"一直趴着"（复盘 ✓）：人半路掉到**别的层**（底层 ✓）之后，`DROP` 相里
        #   `_falling` 那条分支**只做两件事**（`_arrived` 判"到了" ✓ / 按「移动操作尝试间隔」
        #   补按跳 ✓）⇒ 而"到了"永远不成立（脚下那层既不是起跳、也不是目标 ✗）⇒ **一拍都不回
        #   入口** ✗ ⇒ ↓ 一直按着 = 人一直趴着 ✓ —— 只剩「寻路超时」那道总闸能救 ✓
        #   （日志里那一段只有 `drop_taps` 在涨、`drop_phase` 一直 3 ✓）。
        # 判据见 `_strayed`（三条同时成立 ✓ 全用现成的广播 ✓ **不新造几何** ✗ 同 SKILL 77 ✓）。
        # ⚠ **ALIGN 相不管** ✗（它自己有 2026-10-02 那两条：到目标 ⇒ 收工 ✓ / 在别的集合 ⇒
        #   如实失败让上层按现在的位置重算 ✓）—— 在这儿抢它的活会变成"回入口 ⇒ 又回入口"的
        #   死循环 ✗（ALIGN 每一拍都会再命中同一条判据 ✓）。这里只补"**已经按过 ↓ 之后**"
        #   那几相（ARMED / DROP / LANDING / DETACH ✓）—— 正是"过程中"这三个字 ✓。
        # ⚠ 回入口那一拍 `_to_entry` **一个键都不按** ✓ ⇒ ↓ 立刻松开、人**当场起身** ✓
        #   （这就是现场要的那一下：不再趴着 ✓）；下一拍照入口流程重判 ✓。
        if self.phase != self.ALIGN:
            _stray = self._strayed(py, here_sets, ground_y)
            if _stray:
                return self._to_entry(_stray)

        if self.phase == self.ALIGN:
            # ⭐⭐ **ALIGN 改成"当前是否可下跳"的判 + 走**（用户 2026-09-28 定稿 ✓ 原话：
            #   "问题出在 ALIGN 上"；流程图：入口先判「**当前是否可下跳**」⇒ **是** 就直接
            #   ARMED/DROP/LANDING；**否** 就"**就近挑一块『可下跳 foothold』并前往**" ⇒
            #   **进入范围** ⇒ **回入口再判** ✓）。
            #   ⭐ 2026-10-06：③''' 那条"「移动操作尝试间隔」内没反应"**就是回到这里** ✓
            #   （`_to_entry()` 把相位打回 `ALIGN` ✓）—— 所以本相**必须**保持"先判再决定"
            #   这个形状，别改成"直接从 ARMED 续上" ✗（那样就绕过了入口判断 ✓）。
            #
            # ⚠⚠ **改之前的病**（现场 2026-09-28 18:17 那次「三楼 → 二楼」超时 ✗）：
            #   老判据是"人的 x 落进**某一小块** foothold 的 `[left-tol, right+tol]`，
            #   **并站够 `hold_ms`**"（= 设置里「坐标对齐误差时间」，那个项目设的 **1000ms** ✗）。
            #   而**位置读数本身会摆**：实测那 10 秒里人的世界 x 在
            #   `1165 → 1159 → 1173 → 1185 → 1159 → 1134 → 1173` 之间跳，`drop_dx` 随之在
            #   **4.7 ⇄ 30.1** 来回摆 ✓ ⇒ 只要有一拍跳出那 ±10px 余量，`_in_tol_since` 就
            #   **清零重来** ⇒ 10 秒总闸内**永远凑不满一个连续的 1 秒** ⇒ 连"按住 ↓"都没进过
            #   ⇒ 超时 ✗（`drop_phase` 全程 `1.0` ✓、`drop_taps` 一直是 0 ✓）。
            #   ⚠ 而"走位"恰恰是**执行器自己**发方向键造成的：朝中心走一步就冲出范围、再反向
            #     ⇒ **振荡**（`drop_dx` 4.7 ⇄ 30.1 就是它 ✓）。
            #
            # ⇒ 现在只做两件事，**不站、不等** ✓：
            #   · **可下跳** = 人的 x 落在**任意一块**「可下跳 foothold」的范围里（`tol_px`
            #     那点余量保留 —— 它是给读数抖动的 ✓）⇒ **立刻**进 ARMED ✓
            #     （这一步就是流程图里"进入范围 ⇒ 回入口再判"的落点 ✓）；
            #   · 都不在 ⇒ **就近挑一块、朝它走**（图里"否"那条 ✓）—— 到了下一拍自然命中 ✓。
            # ⚠ **不再做"下方是不是目标集合"的二审**：那几块 foothold 是**你在编辑器里圈给
            #   这条下跳边**的（`zones.drop_footholds` ✓）⇒ "从这儿跳能到目标"是**你配的意图**
            #   ✓，执行器**不必也不该**再验一遍（同"编辑器编的是意图、连不连通归执行器自洽"✓）。
            # ---- ⓪ **别在错的层上做下跳**（用户 2026-10-02 现场 ✓，判据**只问广播** ✓）----
            #   2026-10-02「龙族打猎场卡 40 秒」的根因：这一段原来**只看 x** ⇒ 人从「上层」
            #   滑落平台、掉到「二楼」时，那个 x 正好落进**上层**某块可下跳 foothold 的
            #   ±`tol_px` 里 ⇒ 被判成"已经在下跳点上" ⇒ 从**错误的层**按 ↓+跳（在二楼穿平台）
            #   ⇒ 一路掉到最底层，而任务还在"等落到中层"⇒ 卡死 ✗✗。
            #   日志铁证：`align→armed` 那一拍 `pos sets=二楼 x=456.7 y=855.5`，而 spots 全是
            #   上层的面（y≈305）✓。
            #   ⇒ 两条闸（都只用广播，**执行器不自己算几何** ✓ 同 SKILL 77）：
            #     · **已经到目标层**（`here_sets` 命中 `dst_set`）⇒ **收工** ✓ —— 他可能是
            #       掉下来正好落到目标，那是"到"、不是"失败"（用现成的 `_arrived` ✓ 一处口径）；
            #     · **既不在起跳平台、也不在目标层**（`here_sets` 非空却两边都不沾）⇒
            #       **这一段做不了 ⇒ 如实失败**，让上层**按现在的位置重算**路线 ✓
            #       （现场就是掉到二楼 ⇒ 失败 ⇒ 追击重规划 ⇒ 从二楼出发 ✓）。
            #   ⚠ 脚下**没圈进任何集合**（`here_sets` 空 = 判不出来 / 半空中）⇒ **不拦** ✓
            #     （继续走流程，"站在对的面上/超时"那两条会兜住 ✓）。
            #   ⚠ `src_set` 空（手工建的任务 / 老调用方）⇒ 换层那条**不判** ✓。
            _why = self._arrived(py, here_sets, ground_y=ground_y)
            if _why:
                self.phase = self.DONE
                self.note = _why
                return self._out(0, False)
            _here = set(str(v) for v in (here_sets or ()) if str(v))
            if (_here and self.src_set and self.dst_set
                    and self.src_set not in _here and self.dst_set not in _here):
                return self._fail(
                    "人已经不在起跳的「%s」上了（脚下是「%s」）⇒ 这段下跳做不了，"
                    "交给上层按现在的位置重算"
                    % (self.src_set, "／".join(sorted(_here))))
            # ---- ① "就在下跳点上" = x 命中 **且** 脚下那块面就是它的面（y 判据 ✓）----
            #   ⚠ y 那把尺**只问广播的 `ground_y`**（= 位置状态给的"脚下那块面的 y"✓，
            #     与"是不是悬着"那条（`_hanging`）**同一把尺** `FOOT_GAP_PX` ✓ 不新造 ✓）。
            #   `ground_y` / 该点的面 y 任一拍拿不到（None）⇒ **退回只看 x** ✓
            #   —— "判不出来不拦"，老用例 / 老调用方行为一字不变 ✓。
            _gy = None if ground_y is None else float(ground_y)
            _hit = None
            # ⭐⭐⭐ **先挑"没被记过"的下跳点**（`_bad_spots` ✓ 用户 2026-10-10 ✓ 见它那段现场 ✓）：
            #   被记过的那些点**按 ↓ 会被旁边的绳吸走** ✗ ⇒ 有人换过就换一个 ✓；
            #   **全都被记过**才回来用它们 ✓（宁可再慢一轮，也不许"这里下不去"✗）。
            #   ⚠ 两趟走**同一个判据**（x 命中 + 脚下那块面够近 ✓）⇒ 只是"先看谁"不同 ✓，
            #     不在这一层引入任何新几何 ✗。
            for _pass in (0, 1):
                for _c, _a, _b, _fy, _fid in self.spots:
                    if _pass == 0 and str(_fid) in self._bad_spots:
                        continue
                    if _a - self.tol_px <= float(px) <= _b + self.tol_px:
                        if _gy is None or _fy is None or abs(_gy - float(_fy)) <= FOOT_GAP_PX:
                            _hit = (_c, _a, _b, _fy, _fid)
                            break
                if _hit is not None:
                    break
            if _hit is not None:
                self._pick = _hit
                self._in_tol_since = None    # 这套"站住计时"已经不用了（字段留给 DETACH 清 ✓）
                self.note = ("已经在可下跳的 fh %s 上（x 落在 %.0f~%.0f%s）⇒ 直接按住 ↓"
                             % (_hit[-1], _hit[1], _hit[2],
                                "" if (_gy is None or _hit[3] is None)
                                else "、脚下那块面差 %.0f px" % abs(_gy - float(_hit[3]))))
                self.phase = self.ARMED
            else:
                # ⚠ 这里**不做"重挑点"**（用户 2026-10-02 ② 的另一半，实测是空操作 ✓ 说明如下）：
                #   `spots` **全部来自起跳集合**（`zones.drop_footholds` 给的就是 `from` 集合
                #   那几块面 ✓）⇒ 人已经不在这一层时，重挑出来的还是同层的点 ⇒ 朝它走也走不到
                #   ✗。而"人到底在哪一层"这件事**只有广播说得清** ⇒ 交给上面 ⓪ 的两条闸：
                #   到了目标层 ⇒ 收工 ✓ / 在别的已知集合 ⇒ 失败让上层重算 ✓；**判不出来**
                #   （集合空）⇒ 就是下面这条：不许 ARMED，照旧走流程等它落定 ✓。
                if self._pick is None:
                    self._pick = min(self.spots, key=lambda s: abs(s[0] - float(px)))
                    self.note = ("就近平齐到 fh %s 的中心（x=%.0f，该 fh 范围 %.0f~%.0f）"
                                 % (self._pick[-1], self._pick[0], self._pick[1], self._pick[2]))
                _pc, _pl, _pr = self._pick[0], self._pick[1], self._pick[2]
                self.note = "对齐中：朝中心 x=%.0f 走（差 %.0f px）" % (_pc, _pc - float(px))
                return self._out(1 if _pc > float(px) else -1, False)
            self._armed_at = now
            self._y0 = float(py) if py is not None else None
            self.note = "松开行走键，按住 ↓（%d ms）再点按跳" % int(ARM_HOLD_S * 1000)

        if self.phase == self.ARMED:
            # **先按住 ↓ 一小会儿**（用户说的顺序：按住↓ → 点按跳）。
            # 这一刻只发方向键、不发跳（agent 那边靠 `hold_dir` 知道该怎么办）。
            #
            # ⚠ 出发 y 要**一直更新到起跳前**：世界坐标不是每拍都有（认不出黄点时是 None），
            # 只在"进 ARMED 那一拍"记一次的话，那拍没坐标 ⇒ 出发 y 是 None ⇒ 落地兜底
            # 判据（y 掉下去一段）**永远不会成立**（踩过）。
            if py is not None:
                self._y0 = float(py)
            if (now - self._armed_at) >= ARM_HOLD_S:
                self.phase = self.DROP
                self._attempt_at = None      # 下一拍开始第一次尝试 ✓
                self.note = "按住 ↓ + 点按跳"
            else:
                # ⚠ 这里就是流程图那个"**通知位置状态：允许判为在绳梯上**"的落点 ✓ ——
                #   按住 ↓（`hold_dir=True` ⇒ `_last_vert = -1` ✓）⇒ `agent.climbing_vertical()`
                #   为真 ⇒ 位置状态机**才**允许判"在绳上" ✓（2026-09-28 诉求 1 ✓）。
                #   **不另加一个"通知"开关** ✗：那会有第二份真相（同 `_last_vert` 打架 ✓）。
                return self._out(0, False, hold_dir=True)

        # ---- ④⑤ 到了没有（下跳中 / 已松开 ↓ 等落地，都要判）----
        if self.phase in (self.DROP, self.LANDING):
            # ⭐ 几何兜底要**位置状态广播的 `ground_y`**（"脚下那块面的 y" ✓）——
            #   用户 2026-09-28 定的新判据见 `_arrived` ✓。
            why = self._arrived(py, here_sets, ground_y=ground_y,
                                on_rope_pos=on_rope_pos)
            if why:
                self.phase = self.DONE
                self.note = why
                return self._out(0, False)      # 落地即松手（↓ 也不再按）✓

        if self.phase == self.DROP:
            # ③ 一轮 = **按住 ↓ + 连按跳**（用户 2026-09-27 现场口径，照抄）：
            #   原话："**按住↓，下一拍再按跳，可以用「卡住判定时长」连按**" +
            #   "按住↓就会趴，**可能还没趴你就按跳所以无效**" ⇒ 下跳的触发条件是
            #   **趴着时按跳** ✓ ⇒ 原来"只点按**一次**跳"太死 ✗：那一下要是还没趴下去，
            #   就白按了（现场就是"趴着不跳、起身才跳"✗）。
            #   所以：↓ **全程按着**（松了就起身 ⇒ 根本不触发 ✗），跳**按点按周期一遍遍连按** ✓
            #   （每下 `TAP_ON_S`、间隔 `TAP_PERIOD_S` —— 和「跳下绳梯」同一份节奏，不新造 ✓）。
            #   一轮窗口 = 「**卡住判定时长**」(`stall_s` ✓ 用户指定用它连按)：窗口内一直连按；
            #   到点 Y 还没动 ⇒ **松开 ↓ 一拍**（起身）⇒ 隔「移动操作尝试间隔」(`retry_ms`)
            #   再按住 ↓ 连按（下一轮 ✓；`0` = **不重试** ⇒ 一直按住连按，交给「寻路超时」✓）。
            if self._attempt_at is None:                  # 刚进 DROP（第一拍 ✓）
                self._falling = False                     # "已经在掉"从零开始 ✓
                self._jump_reassert_at = None             # （下行补按跳那笔账也重新记 ✓）
                # ⛔ **2026-09-28 删掉**：这里原来有"松着 ↓ 等「移动操作尝试间隔」到点再按住"
                #    那几拍（`_next_round_at` ✓）—— 用户口径是"**进了 DROP 就一路按住 ↓，
                #    只有判定 drop 上绳梯了才松开 ↓，其他只有补按**"✓ ⇒ **不许有空等的几拍** ✗
                #    （那几拍人是站直的 ⇒ 又要重新趴 ⇒ 永远趴不下去 ✓）。
                self._next_round_at = None
                self._attempt_at = now
                self._y_base = None                       # 基准 = "跳之前"的 y（下面那一小段拿 ✓）
                self._jump_at = None
                self.note = ("按住 ↓ + 连按跳（第 %d 轮，%.1f s 内看 Y 动不动）"
                             % (self._tap_n, self.stall_s))
            # ⚠ **没有 y 基准就先别按跳**（用户 2026-09-27："当 Y 变化超过「坐标对齐误差范围」
            #   **才能**松开 ↓"）：基准拿不到 ⇒ "有没有开始下落"根本判不出来 ⇒ 按了跳也只能
            #   一路按到超时 ✗。所以先按住 ↓ 等一拍 y 读数（通常下一拍就有 ✓）再按跳 ✓ ——
            #   这样"按下跳"和"记下基准"一定在同一拍上（判据才站得住 ✓）。
            if self._y_base is None:
                if py is None:
                    self.note = "按住 ↓：等一个 y 读数（没读数就没法判「开始下落了没有」）"
                    return self._out(0, False, hold_dir=True)
                self._y_base = float(py)
                self.note = ("按住 ↓ + 连按跳（第 %d 轮，%.1f s 内看 Y 动不动）"
                             % (self._tap_n, self.stall_s))
            # ④ **Y 真的开始下落 ⇒ 停掉「点按跳」，但 ↓ 一直按着** ✓
            #    用户 2026-09-27 明确（原话）："在 drop 时，**除非位置状态到了目的地，否则 ↓ 是
            #    不能松开的**" ✓ —— 它盖掉了当天早上那条"Y 一变就松 ↓"✗：
            #    早松会让人在半空 / 刚落地那几拍被迫起身（又去抓绳 / 站住不动 ✗）。
            #    ⇒ 到位只由 `_arrived(py, here_sets)` 判（上面那段 ✓，判到就 DONE 并松手 ✓）。
            if self._y_changed(py):
                if not self._falling:
                    self._falling = True
                    self.note = ("Y 从 %.0f 掉到 %.0f（向下 ≥%.0f px）⇒ **↓ 继续按着**，"
                                 "等落到「%s」再松"
                                 % (self._y_base, float(py), self.y_tol_px, self.dst_set))
                # ⭐⭐ **掉的过程中：↓ 一路按着 + 按「移动操作尝试间隔」补按跳**
                #   （用户 2026-09-28 口径 ✓ 原话："下落过程中也按照『移动操作尝试间隔』补按
                #    **按住 ↓ + 跳**"）。
                #   ⚠ 原来是"只按住 ↓、**不再点按跳**"✗ —— 病根是"跳是用来触发的、已经在掉了"
                #     这条**假设**：可下落途中**也可能又卡住**（被绳吸住 / 卡在半空的绳上 ——
                #     `_detach_step` 那条"被绳吸住"正是靠按跳脱离 ✓）⇒ 只按 ↓ 会**一路干等到
                #     超时** ✗（现场就是"按住 ↓ 不按跳、也不补按，直到超时"✗）。
                #   ⚠ 节奏照抄「移动操作尝试间隔」（`retry_ms`，与 `_hold_reassert` 同一份 ✓，
                #     不新造时长 ✓）⇒ `retry_ms <= 0` 仍**不补**（尊重设置 ✓）。
                #   ⚠ ↓ 仍由 `hold_dir=True` 保证**全程按着**（用户 2026-09-27："除非位置状态
                #     到了目的地，否则 ↓ 不能松开" ✓）—— 这里只决定"跳"按不按 ✓。
                jp = self._falling_jump(now)
                return self._out(0, jp, hold_dir=True)
            # ⭐⭐ **③'''  「移动操作尝试间隔」内没反应 ⇒ 回入口**（用户 2026-10-06 流程图 ✓
            #   原话："**当在「移动操作尝试间隔」内没反应就从最开始的判断走**
            #   （**当前 foothold 是否可下跳**）"）。
            #   · "**没反应**" = 既**没有正常下落**（`_y_changed` 上面那条没命中 ✓）、
            #     也**没被位置状态判成上绳**（那条更早、在 `update` 最前面 ✓）⇒ 走到这儿
            #     就是"按住 ↓ + 跳了半天，游戏**一点动静都没有**" ✓。
            #   · 处置 = `_to_entry()`（相位打回 `ALIGN` = 流程图顶部那个判断 ✓）⇒
            #     下一拍重新判"**当前这块 foothold 还能不能下跳**"：**是** ⇒ 重新"按住 ↓ + 跳"
            #     ✓；**否** ⇒ 就近挑一块、走过去 ✓。
            #   ⚠ **窗口从"进 DROP 那一刻"算**（`_attempt_at` ✓，进相时打的点 ✓），
            #     与流程图里"按住↓+跳 ⇒ 间隔到点还没反应"那一支对齐 ✓。
            #   ⚠ `retry_ms <= 0` ⇒ **不回头**（用户在设置里把重试关掉 = 老行为：一路按住 ↓
            #     连按，交给「寻路超时」✓ —— 尊重设置 ✓）。
            if (self.retry_ms > 0 and self._attempt_at is not None
                    and (float(now) - float(self._attempt_at)) * 1000.0 >= self.retry_ms):
                # ⚠ `_to_entry` 返回的那一拍**一个键都不按**（= 真的松开一拍 ✓）——
                #   那正是"下一轮按住 ↓ 是**新的一次按下**"的保证 ✓（见它的 docstring ✓）。
                return self._to_entry(
                    "按住 ↓ + 跳了 %.0f ms（「移动操作尝试间隔」）既**没下落**、也没**上绳** "
                    "⇒ 回入口，重新判「当前这块 foothold 还能不能下跳」"
                    % float(self.retry_ms))
            # ⛔⛔ **2026-09-28 整段删掉**：这里原来有"**一轮窗口（「卡住判定时长」`stall_s`）
            #   到点、Y 还没动 ⇒ 松开 ↓ 起身** ⇒ 隔「移动操作尝试间隔」再来一轮"✗。
            #   用户口径（2026-09-28 ✓ 原话）："**到点 Y 还没动，也不应该松开 ↓ 起身，只要进了
            #   DROP 就一路按住 ↓**"／"**流程中，只有判定 drop 上绳梯了才松开 ↓，其他只有补按**"✓
            #   ⇒ 那段"松手起身"**正是永远趴不下去的根因** ✗：每一轮只活 ~0.35 s 就站直一次
            #     （日志里 `drop_round` 3→9 ✓），游戏里**根本来不及趴** ⇒ 每一下跳都是**站着
            #     原地跳** ✓ —— 现场（用户 2026-09-28 报"又卡了、在原地跳"）就是这个 ✓。
            #   ⇒ 现在这一相**只剩三件事**：
            #     ① `_falling`（真开始落了 ⇒ 继续补按跳 ✓，见上面那段）；
            #     ② ⭐ **③'''「移动操作尝试间隔」内没反应 ⇒ 回入口**（2026-10-06 加 ✓）；
            #     ③ ③''「连按跳」（按 `TAP_ON_S` / `TAP_PERIOD_S` 的节奏 ✓）。
            #   ⚠ **↓ 松手的时机只剩三处**（与用户流程图**一致** ✓）：
            #     · 落地 `DONE` / `FAILED`（`_arrive_step` / `_fail` ✓ 该松了 ✓）；
            #     · `_detach_step`（**判定上绳梯** ⇒ 脱离，那几拍传 `vert=0` **主动松 ↓** ✓）——
            #       这正是用户说的"**只有判定 drop 上绳梯了才松开 ↓**"✓；
            #     · ⭐ **回入口那一拍**（③''' ✓ 一个键都不按 —— **必须**松一下，理由见下 ✓）。
            #   ⚠ `stall_s` / `_tap_n` 从这一刻起**不再参与决策**（留着字段只为日志/用例 ✗
            #     不删定义：设置里那两项还在，别让别的调用方当场炸 ✗）。
            # ③'' **连按**：到「点按周期」就**再按一下**（`_jump_at` = 这一下的起点 ✓）——
            #     用户口径的重点：下跳要"**趴着时按跳**"才触发，而趴下去要一点点时间 ⇒
            #     那一下要是按早了就白按 ⇒ **一遍遍连按**，总有一下落在趴着的时候 ✓。
            # ⛔ **2026-09-28 修：这一段的顺序反了** ✗（用户当天报"**趴在那不动了也不按跳**"）。
            #   原来"该**按**"写在前面、靠 `now - _jump_at >= TAP_PERIOD_S` 判 ⇒ **拍间隔一旦
            #   大于「点按周期」`TAP_PERIOD_S`(180ms)**，每拍都会命中"重新按"那条 ⇒
            #   **整段跳过"松窗"**（`[TAP_ON_S, TAP_PERIOD_S)`）⇒ 退化成**按住跳** ✗
            #   （实测：拍间隔 200ms ⇒ 输出序列全是"按住"、10 秒只算出 1 下 ✗；
            #    拍间隔 25ms ⇒ 正常"按 3 拍、松 5 拍"✓）⇒ 而**"一直按住跳会被游戏当成还挂在
            #   上面"**（用户 2026-09-26 口径 ✗）⇒ **下跳根本不触发** ✓ —— 现场"趴着不动"就是它 ✓。
            #   ⇒ 改成**先判"该松"**（按 `_jump_on` 这个**状态**判，不靠窗口落点 ✓）：
            #     跳按下去之后，只要过了 `TAP_ON_S` 就**必须松**；松了之后才轮到"该按" ✓。
            #     这样最差也只是"按一拍、松一拍"（周期 = 2 拍 ⇒ 慢拍下约 2.5 下/秒 ✓），
            #     **绝不会**再退化成按住 ✓。
            if self._jump_on and (float(now) - self._jump_at) >= TAP_ON_S:
                return self._out(0, False, hold_dir=True)
            if (self._jump_at is None
                    or (float(now) - self._jump_at) >= TAP_PERIOD_S):
                self._jump_at = now
                self.note = ("按住 ↓ + 连按跳（第 %d 轮，%.1f s 内看 Y 动不动）"
                             % (self._tap_n, self.stall_s))
                #                    ↓ 按住 + 这一下跳按下 ✓
                # ⛔ 原来这里带 `reassert=`（"顺手把 ↓ 也重发一次"✓）—— 2026-10-06 删 ✗：
                #   用户要求"**按键补按逻辑移除**"。对面要是把 ↓ 丢了，靠的是**③''' 那条
                #   回入口**（回入口那一拍把键全松开 ⇒ 下一轮的"按住 ↓"是**真正的新按下** ✓），
                #   不再靠"键集没变也重发一遍" ✗。
                return self._out(0, True)
            if (float(now) - self._jump_at) < TAP_ON_S:
                # 这一下还没够时长 ⇒ 继续按着 ✓
                return self._out(0, True)
            return self._out(0, False, hold_dir=True)   # ↓ 继续按着，等下一 下 ✓

        # ---- ④ 等落地：↓ 已经松开（用户口径"Y 变化超过误差范围**才能**松开 ↓"）⇒ 什么都不按 ----
        # ⚠ **"松了却没落下去"必须有退路**（否则它会一直"什么都不按"地挂到「寻路超时」✗）：
        #   超过「移动操作尝试间隔」（`retry_ms`，= 用户定的"移动操作尝试间隔"）还没到 ⇒
        #   **回入口**：重新对齐、再按跳 ✓。走到这儿通常意味着那次点按跳没生效
        #   （对面把键丢了 / 按早按晚了 / 人根本没离地 ✓）。
        if (self.retry_ms > 0 and self._landing_at is not None
                and (now - self._landing_at) * 1000.0 >= self.retry_ms):
            return self._to_entry("松开 ↓ 后 %.0f ms 还没落地 ⇒ **回入口**重来" % self.retry_ms)
        return self._out(0, False)

    def _to_entry(self, why):
        """**回入口**：相位打回 `ALIGN`（= 流程图顶部那个"**当前 foothold 是否可下跳**" ✓）
        + 把这一轮的运行时状态**全部清干净**。

        两处调用它（**一处实现** ✓，别各写一遍 ✗）：
          · ⭐ ③'''「**移动操作尝试间隔**」内没反应（用户 2026-10-06 ✓ 本轮改的口径）；
          · ④ 「松开 ↓ 之后 `retry_ms` 还没落地」（2026-09-27 的退路 ✓）。

        ⚠ `_pick = None` ⇒ 回到 `ALIGN` 会**重新就近挑**一块可下跳 foothold（流程图"否"那条 ✓）；
          ⚠ `_falling = False`：走到这儿就说明"这一轮没能落下去"，下一轮从头看 ✓；
          ⭐ **它的返回值就是"回入口那一拍"**（`_out(0, False)` ⇒ **一个键都不按** ✓）：
            调用方一律 `return self._to_entry(...)` ✓ —— 这一拍"真的松开"是**故意的**，
            因为 `KeyState` 只在**键集变化**时才发键 ⇒ 不松这一下，下一轮那个"按住 ↓"
            对面**收不到新的按下** ⇒ 人**不趴** ⇒ 以后每一下跳都只是**原地跳** ✗
            （原来靠"补按"绕开，现在靠"真的松一下 ⇒ 真的一次新按下"解 ✓ 见类 docstring ✓）。
        """
        self.attempt += 1
        self.phase = self.ALIGN
        self._in_tol_since = None
        self._pick = None
        self._armed_at = None
        self._attempt_at = None
        self._y_base = None
        self._jump_at = None
        self._landing_at = None
        self._next_round_at = None
        self._falling = False
        self._jump_reassert_at = None
        self.note = ("%s（第 %d 次；整体由「寻路超时时间」兜底）" % (why, self.attempt))
        return self._out(0, False)

    def _falling_jump(self, now):
        """**下落过程中**这一拍要不要按跳 —— 按「移动操作尝试间隔」补按（用户 2026-09-28 ✓）。

        口径（用户原话）："下落过程中也按照『移动操作尝试间隔』补按 **按住 ↓ + 跳**" ✓。

        ⚠ 为什么需要：原来这一段（`_falling`）是"只按住 ↓、**不再点按跳**"✗，理由是"跳是用来
          **触发**下落的、已经在掉了" —— 可下落途中**自己也会卡住**：**被绳吸住 / 卡在半空的
          绳上**（`_detach_step` 那条"被绳吸住"正是靠按跳脱离 ✓）⇒ 只按 ↓ 会**一路干等到
          寻路超时** ✗（现场就是"按住 ↓ 时没有按跳，也没有补按，直到超时"✗）。
        ⚠ 节奏**照抄**「移动操作尝试间隔」`retry_ms`（**同一份**节拍 ✓，不新造时长 ✓）：
          到一个间隔按一下跳、按满一小段就松开，如此反复 ✓；
          `retry_ms <= 0` ⇒ **不补**（用户在设置里关掉重试 ⇒ 尊重设置 ✓）。
        ⚠⚠ **只有这一处在"补按"了**（2026-10-06 ✓）—— "按住 ↓ 的重发"那套（`_hold_reassert`）
          已按用户要求删掉 ✗；这里的"补按跳"是**另一件事**（下落途中跳一下也救"被绳吸住"✓），
          且它是**真的重新按下跳**（键集变了 ⇒ 对面收得到 ✓），不是"重发一个已经在按的键"✓。
        ⚠ **第一下立刻按**（进"下落"那一拍就按 ✓）：下跳的触发条件是"**趴着时按跳**" ✓，
          早按不亏，还能救"被绳吸住"那种 ✓。
        ⚠ 这段**只判"跳"**：↓ 由调用方 `hold_dir=True` 保证**全程按着** ✓（用户 2026-09-27：
          "除非位置状态到了目的地，否则 ↓ 不能松开" ✓）。
        ⚠ ⚠ **别退化成"一直按住跳"**：`retry_ms` 若比「点按时长」还短 ⇒ 把这一下的按满时长
          钳到**间隔的一半**（保证至少松一拍 ✓）—— 用户 2026-09-26 口径："一直按住跳会被游戏
          当成**还挂在上面**" ⇒ 下跳根本不触发 ✗。
        ⚠ 点数会进 `drop_taps` 探针（`_out` 数上升沿 ✓）⇒ 日志里能看出"补按在跑" ✓。
        """
        if self.retry_ms <= 0:
            return False
        on_s = min(TAP_ON_S, (self.retry_ms / 1000.0) * 0.5)
        if self._jump_reassert_at is None:
            self._jump_reassert_at = float(now)
            self.note = ("下落中：↓ 一路按着 + 按「移动操作尝试间隔」（%.0f ms）补按跳"
                         "（等落到「%s」再松 ↓）" % (self.retry_ms, self.dst_set))
            return True
        dt = float(now) - self._jump_reassert_at
        if dt < on_s:
            return True                     # 这一下还没按满 ⇒ 继续按着 ✓
        if dt * 1000.0 >= self.retry_ms:
            self._jump_reassert_at = float(now)     # 又到了一个间隔 ⇒ 再按一下 ✓
            return True
        return False                        # 间隔期 ⇒ 跳松开（↓ 仍按着 ✓）

    def _y_changed(self, py):
        """**Y 是不是真的开始下落了**：相对本次尝试的基准**向下**变化 ≥「坐标对齐误差范围」。

        用户 2026-09-27 的口径（下跳第 ④ 条）："**当 Y 变化超过「坐标对齐误差范围」才能
        松开 ↓**" ⇒ 这就是**松开 ↓ 的唯一正常理由** ✓（另一处是第 ③' 条的"尝试间隔内没动
        ⇒ 补发一个松开 ↓ 再重来"，那也是用户定的 ✓）。

        ⚠ **只认向下**（y 变大 = 真的掉下去了 ✓）：**向上抖不算** ✗。以前这里写的是
          `abs(...)`，于是人还站在平台上、读数向上抖一下（>误差范围）就被判"开始下落"、
          把 ↓ 松掉 ⇒ 那一次下跳当场作废，而且之后停在"等落地"什么都不按 ✗
          （现在就算误松也有回入口兜底 ✓，但判据本身就该只认下落 ✓）。
        ⚠ 没有 y 读数（定位掉帧）⇒ **不判**（观测问题，不是"没动" ✗）；`_y_base is None`
          ⇒ 也不判（DROP 里已经保证"没基准就不按跳" ⇒ 正常走不到 ✓）。
        """
        if py is None or self._y_base is None:
            return False
        return (float(py) - self._y_base) >= self.y_tol_px

    def retry(self):
        """失败后**重新激活**：回到对齐那一步重来（和 ClimbJob 同一套保护）。"""
        self.attempt += 1
        self.phase = self.ALIGN
        self._t0 = None
        self._in_tol_since = None
        self._armed_at = None
        self._y0 = None
        self._attempt_at = None
        self._y_base = None
        self._jump_at = None
        self._landing_at = None
        self._tap_n = 1
        self._next_round_at = None      # "松着 ↓ 等间隔"也收干净（新的一轮从对齐开始 ✓）
        self._jump_reassert_at = None   # "下落中补按跳"那笔账同理（下次下落重新计 ✓）
        self._falling = False           # "已经在掉"同样收干净 ✓
        self._detach_tap_at = None      # "被绳吸住、正在脱离"同理收干净 ✓
        self._detach_pressed = False    # "跳下绳"第一步（方向单独一拍）也重新开始 ✓
        # ⚠ 不写分母（同 `ClimbJob.retry`：没有"最多试几次"了 ✓）
        self.note = "第 %d 次尝试：重新对齐（下跳）" % self.attempt
        return self._out(0, False)

    def cancel(self, why="取消"):
        # ⚠ **运行时字段一起收干净**（用户 2026-10-02 #8 ✓ 见 `_cancel_common` ✓）——
        #   原来这里只置相位，带着脏锚被复用就是隐患 ✗。
        self._cancel_common(why)
        return self._out(0, False)

    def _arrived(self, py, here_sets, ground_y=None, on_rope_pos=None):
        """到了没有：**先集合后几何**（和下跳那边同一套口径）。

        ⛔ **2026-09-28 加一条总闸：人在绳上就不算到**（用户现场："从**顶层到三楼**的下跳期间
        **上绳子了**，但是 drop 执行器没有从位置状态机拿到上绳通知而去执行跳开绳子的逻辑"✗）。
        根因：**感知的"脚下"会被绳段下方的平台骗**（实测图 `106010105` 的绳 L2：人挂在
        y=-291（绳正中）、离下面那块面还有 109px，`pos_state` 仍给
        `ground_y=-182` + `here_sets=['三楼']` ✗ —— 因为 `foothold_below` 只对"头顶上方"
        卡 40px 容差、**下方不卡** ✗）⇒ 于是上面两条判据**双双误报"到了"**：
        · 集合那条 ⇒ 直接 `done`（日志：`prev=drop to=done  note=到达：脚下已经是「三楼」`
          ✗ 而人还在绳上 ✗）⇒ 任务提前收工 ⇒ **根本轮不到"跳开绳子"** ✗；
        · 面 y 那条 ⇒ 绳段下方的面正好等于目标面时同样误报 ✗。
        ⇒ 判据：**`on_rope_pos` 非空（位置就落在某根绳的绳段里）⇒ 一律不算到达** ✓
        （`on_rope_pos` 与按键许可无关 ✓，正是"人挂在绳上"这个客观事实 ✓）。
        ⚠ 只在**下跳**这儿收这道闸（用户 2026-09-28 明确："不动 `foothold_below`"✓）——
        不动那个共用判据，避免波及集合归属 / 段 / 起点判定那些消费方 ✓。

        下面两条判据的细节：

        ⭐ **几何兜底 = 「脚下这块面就是目标平台的面」**（用户 2026-09-28 明确要求 ✓）：
        读**位置状态广播**的 `ground_y`（"脚下那块面的 y"，按玩家 x 取 ✓），命中目标集合
        里任意一条 foothold 的面 y（± `y_tol_px` ＝「坐标对齐误差范围」✓）就算落地 ✓。
        ⛔ **旧判据已删**（"y 比出发时大了 `DROP_DY_PX`(40px)"✗）：太松 —— 人只要往下掉
        40px 就报"到了"，哪怕还在半空、或者落到了**别的**平台上 ✗（实测：追击反复下发同一条
        下跳任务，就是它误报"到了" ⇒ 人其实没下去 ⇒ 下一拍又判"怪不在我这块平台" ⇒ 又下 ✗）。
        ⚠ 目标面算不出来（`dst_ys` 空：老调用方 / 目标集合没圈 foothold）⇒ **退回只看集合** ✓
        —— **不许**再拿"掉了多少像素"凑一个数 ✗（那正是这次要拆掉的东西 ✓）。
        """
        if on_rope_pos:
            # 人正挂在绳段里 ⇒ 上面的"脚下"读数**不可信**（绳段横跨好几层 ⇒ 报出来的
            # 常常是**绳下方那层**的面 ✗）⇒ 一律不算到 ✓（见 docstring 里那条实测 ✗）。
            return ""
        if here_sets and self.dst_set and self.dst_set in set(here_sets):
            return "到达：脚下已经是「%s」" % self.dst_set
        if ground_y is not None and self.dst_ys:
            _g = float(ground_y)
            for _y in self.dst_ys:
                if abs(_g - float(_y)) <= self.y_tol_px:
                    return ("到达：位置状态广播「脚下那块面 y=%.0f 就是目标平台的面（%.0f）」"
                            % (_g, float(_y)))
        return ""

    def _strayed(self, py, here_sets, ground_y):
        """人是不是**站在一块「跟本任务无关」的面上**了（用户 2026-10-11 ✓ 原话见 `update` ✓）。

        返回 "" = **没有**（照常走流程 ✓）；否则是一句**给人看的原因**（喂给 `_to_entry` ✓）。

        三条**同时**成立才算（都用现成的广播 ✓ 不新造几何 ✗）：
          ① 广播给了集合（`here_sets` 非空 ✓）—— **判不出来就不拦** ✓
             （与 ALIGN ⓪ 那条**同一口径** ✓：半空中 / 没圈进集合 ⇒ 一律放过 ✓）；
          ② 既不是**起跳集合**、也不是**目标集合**（`src_set` / `dst_set` 都得有值才算 ✓
             —— 手工建的任务 / 老调用方没填 ⇒ 不判 ✓ 老行为一字不变 ✓）；
          ③ **人真的是站在那块面上**（`|py − ground_y| ≤ FOOT_GAP_PX` ✓ 与"被绳吸住"那条
             **同一把尺** ✓ 不新造容差 ✓）。
        ⚠⚠ **第 ③ 条是防误报的要害** ✗：**半空中**广播照样给得出"下面那层的面"（实测过 ✓
          见 `_hanging` 那段：人挂在绳中、离下面那块面 109px，`ground_y` 照给 ✗）⇒ 不要求
          "站住"的话，正常下落**路过**别的层就会被判成"跑到别处了" ⇒ 一次正常的 2→1 会被
          自己打断 ✗✗。⇒ 只有"脚下那块面就在脚底"才算"**站在那儿**" ✓。
        ⚠ 为什么判据落在**集合**这一层（而不是某个 fh 号 ✗）：广播里**只有集合**（`here_sets`
          ✓ 见 `perception/pos_state.PosSnapshot` ✓ 它没有逐条 fh 号）⇒ 拿集合判是"用现有的
          唯一那份事实" ✓，也正好盖住用户说的那种意外（底层 / 别的层 ✓ 都是"别的集合"✓）。
        ⚠ 目标层**到了**不算 stray（`dst_set in here` ⇒ 上面第 ② 条就挡掉了 ✓）—— "到了"
          由 `_arrived` 判 ✓（**一处口径** ✓ 别在这儿重复实现一遍 ✗）。
        """
        if not (self.src_set and self.dst_set):
            return ""                     # 判据不全（手工任务 / 老调用方）⇒ 不判 ✓
        _here = set(str(v) for v in (here_sets or ()) if str(v))
        if not _here:
            return ""                     # ① 判不出来 ⇒ 不拦 ✓
        if self.src_set in _here or self.dst_set in _here:
            return ""                     # ② 还在起跳 / 已经到目标 ⇒ 正常 ✓
        if ground_y is None or py is None:
            return ""                     # ③ 面 / 位置缺一边 ⇒ 判不出来 ⇒ 不拦 ✓
        if abs(float(py) - float(ground_y)) > FOOT_GAP_PX:
            return ""                     # ③ 悬在空中（只是路过那层）⇒ 不算"站在别处" ✓
        return ("位置状态说人已经站在「%s」上了（既不是起跳的「%s」、也不是目标「%s」）"
                "⇒ 回入口重判「当前 foothold 是否可下跳」"
                % ("／".join(sorted(_here)), self.src_set, self.dst_set))

    def _fail(self, why):
        self.phase = self.FAILED
        self.note = why
        return self._out(0, False)

    def _detach_dir(self, px):
        """脱离时按住哪个**水平方向**（±1）。

        ⭐ **用户 2026-09-28 按流程图定的口径**（原话："跳下绳子：按**任意方向**+跳（**若有锁定
        目标则是锁定目标方向**）"）：
          · **有锁定目标** ⇒ **朝它**（读注入的 `_target_dir_fn()` ✓：`+1` 右 / `-1` 左 / `0`
            没有目标或同列 ⇒ 落到下面那条 ✓）；
          · **其余情况** ⇒ **任意方向**（取一个**确定性**的：**背离下跳点中心** ✓ —— 顺手离它
            远一点，别刚松手又被吸回去 ✓）；还没挑定下跳点（或 x 正好在中心）⇒ 向右 ✓。
        ⚠ 注入位用 `getattr` 兜底：**没有注入 / 注入的回调抛异常** ⇒ 都当成"没有目标" ✓
          （**绝不许**因为拿不到目标就报错或停住 ✗ —— 脱离那几拍按不出方向就等于卡死在绳上）。
        """
        _td = 0
        _fn = getattr(self, "_target_dir_fn", None)
        if _fn is not None:
            try:
                _td = int(_fn() or 0)
            except Exception:                 # noqa: BLE001 —— 回调坏了 ⇒ 退回"任意方向" ✓
                _td = 0
        if _td > 0:
            return 1                          # ⭐ 锁定目标在右边 ⇒ 朝它 ✓
        if _td < 0:
            return -1                         # ⭐ 锁定目标在左边 ⇒ 朝它 ✓
        if self._pick is None:
            return 1
        return -1 if float(px) > float(self._pick[0]) else 1

    def _mark_pick_bad(self):
        """把**这一轮挑中的那个下跳点**记成"按 ↓ 会被绳吸走" ⇒ ALIGN 下次**先挑别的点** ✓。

        用户 2026-10-10 ✓（原话："现在下跳（drop）每次都会被卡在绳子上"✗）：那个下跳点
        紧贴着绳 ⇒ 按 ↓ 就被当成"抓绳"⇒ 脱完绳又就近挑回**同一个点** ⇒ 再吸 ⇒ 死循环 ✓。
        记一笔之后 ALIGN 会**先把别的点挑一遍** ✓（全被记过才回到它 ✓ —— 宁可再吸一次，
        也不许"这里下不去"✗）。见 `_bad_spots` 那段说明 ✓。
        """
        if self._pick is None:
            return
        try:
            self._bad_spots.add(str(self._pick[-1]))
        except Exception:                       # noqa: BLE001 —— 记账失败不该弄坏下跳 ✓
            pass

    def _detach_step(self, now, px):
        """「**被绳吸住 ⇒ 按住方向 + 连按跳**」的一拍（用户 2026-09-27 口径，照抄）。

        ⚠ **绝不按 ↓**（`vert=0`）：在绳上按 **↓** 在游戏里是"**还挂在绳上**"（同一条口径见
          `ClimbJob._out` 的 `vert` 说明 ✓）⇒ 按 ↓ 等于把自己按回去 ✗；
          而按**左右**是"松手"（同一条口径的另一面 ✓）—— 所以这几拍只发"**左右 + 跳**" ✓。
        ⚠ 跳按**同一份点按节奏**（`TAP_ON_S` / `TAP_PERIOD_S` ✓，不新造时长 ✓）：
          和下面 ③ 的"连按跳"是同一个理由（那一下可能还没到"能起效"的时机 ⇒ 一遍遍按 ✓）。
        ⭐⭐⭐ **第一步：方向键单独占一拍**（用户 2026-10-10 ✓ 原话："**应该是需要再判定上绳梯后
          执行跳下绳的操作**"✓）—— 口径**照抄 `ClimbJob._jump_down_beat`** ✓，理由也在那儿：
          同一次 tick 里把"方向"和"跳"一起发，**游戏里未必算"按住方向再跳"** ✗（`KeyState`
          一次下发多个键**不保证顺序** ✓）⇒ 必须分成两拍发 ✓。
          ⚠ 这就是"卡在绳子上"的真正修法 ✓ —— 老版**方向+跳同拍发** ✗ ⇒ 游戏不认这个
            "跳下绳"动作 ⇒ 人一直挂在绳上 ✓（现场 `behavior.log` 07:21:13 那一趟 ✓）。
        """
        step = self._detach_dir(px)
        if not self._detach_pressed:
            # 第一拍：**只按方向**（不按跳、不按 ↓ ✓）—— 让游戏先认下"按住方向" ✓
            self._detach_pressed = True
            self.note = ("在下跳中被绳吸住 ⇒ 先按住 %s（单独一拍）⇒ 再按跳脱离"
                         "（**不按 ↓**：在绳上按 ↓ = 还挂在绳上 ✗）"
                         % ("→" if step > 0 else "←"))
            return self._out(step, False, vert=0)
        if (self._detach_tap_at is None
                or (float(now) - self._detach_tap_at) >= TAP_PERIOD_S):
            self._detach_tap_at = now
        jump = (float(now) - self._detach_tap_at) < TAP_ON_S
        self.note = ("在下跳中被绳吸住了 ⇒ 按住 %s %s 脱离"
                     "（**不按 ↓**：在绳上按 ↓ = 还挂在绳上 ✗）"
                     % ("→" if step > 0 else "←", "并连按跳" if jump else "、等下一 下跳"))
        return self._out(step, jump, vert=0)

    #: 相 → 编码（**给 `behavior.log` 的 `drop_phase` 用** ✓）。
    #: 为什么非要有它（2026-09-28 现场）：上绳那套 `climb_phase` **只认 `ClimbJob` 的相名**
    #: （`CLIMB` / `JUMP_DOWN`）⇒ **下跳任务在它眼里恒 0** ✗ ⇒ 卡住的那近 9 秒里日志只剩一排
    #: `climb_phase=0.0`，**看不出卡在哪一相** ✗（用户当场只能报"实际爬上了向下爬的绳子"）。
    #: ⚠ 加新相时**两处一起改**（这里 + `update` 里的相名 ✓）。
    _PHASE_CODES = {ALIGN: 1, ARMED: 2, DROP: 3, LANDING: 4, DETACH: 5,
                    DONE: 6, FAILED: 7}

    def phase_code(self):
        """这一拍在哪一相（数字，见 `_PHASE_CODES` ✓）—— 日志/界面读它 ✓。"""
        return self._PHASE_CODES.get(self.phase, 0)

    def align_target_x(self):
        """这一轮挑中的**下跳点中心 x**（还没挑定 ⇒ `None` ✓）—— 日志里看"还差多远" ✓。"""
        return float(self._pick[0]) if self._pick else None

    def tap_count(self):
        """已经**点按跳**多少下（**上升沿**计数 ✓）。

        用户 2026-09-28 报"**趴在那不动了也不按跳**"：光看 `drop_phase=3`（在 DROP 相）**分不清**
        "跳压根没发出去"和"发了但游戏没响应" ✗ —— `drop_taps`（每秒记一次）数字在涨 ⇒ 跳在发 ✓；
        不动 ⇒ 见 `drop_round`（轮次有没有在转 ✓）。
        """
        return int(self._taps)

    def round_n(self):
        """现在是第几轮「按住 ↓ + 连按跳」（1 起 ✓）—— 看轮次**有没有在转** ✓。

        ⚠ 2026-10-06 起"轮"的含义变了：一轮 = **一次"按住 ↓ + 连按跳"的尝试**，
        轮与轮之间是 **③''' 回入口**（相位打回 `ALIGN` ⇒ 重新判"当前这块 fh 还能不能下跳" ✓），
        不再是"松 ↓ 空等「移动操作尝试间隔」"（那套早删了 ✗）。
        """
        return int(self._tap_n)

    # ⛔ **`reassert_count()` 已删**（2026-10-06 ✓ 用户要求"按键补按逻辑移除"）：
    #   它数的是"重发了几次「按下 ↓」"，而这个机制本身没了 ✗ ⇒ 没有来源 ✓
    #   （`behavior.log` 里那条 `drop_reassert` 探针同步删 ✓）。"有没有反应"现在看
    #   **`drop_round`（轮次转不转）** + `route_phase`（回没回 `ALIGN`）✓。

    def _out(self, move, jump, hold_dir=False, vert=None, replan=False):
        """`vert`：这一拍要不要发 ↓（`None` = 老规则：按跳或 `hold_dir` 时发 `self.dir`=↓ ✓）。

        ⚠ 「**被绳吸住要脱离**」那几拍必须传 `vert=0`：在绳上按 ↓ 是"还挂在绳上"（口径同
          `ClimbJob._out` ✓）⇒ 按 ↓ 等于把自己按回去 ✗；那几拍只按**左右 + 跳** ✓。

        ⛔ **没有 `reassert` 这个参数了**（2026-10-06 ✓ 用户要求"按键补按逻辑移除"）——
          `ClimbJob._out` 那边**照旧有**（爬绳的「补按 ↑」是另一条口径 ✓）。下跳不再"重发
          已经在按的键"，改成"**回入口**那一拍真的松开全部键"（见 `_to_entry` ✓）。
        """
        d = (self.dir if (jump or hold_dir) else 0) if vert is None else int(vert)
        # ⭐ **点按跳的"上升沿"计数**（见 `tap_count` ✓）：`jump` 会连着几拍为真（要按满
        #   `TAP_ON_S` ✓），只数"从没按到按"那一下 ⇒ 这个数 = **真发出去的"跳按下"次数** ✓
        #   （用户 2026-09-28："趴在那不动了也不按跳" —— 有它才能分清是"没发"还是"不响应" ✗）。
        if jump and not self._jump_on:
            self._taps += 1
        self._jump_on = bool(jump)
        # ⭐ 记下这一拍请他按的竖直键（给 ladder_id 许可的「执行器通知」用，同 ClimbJob ✓）
        self._last_vert = int(d)
        return self._result(move, jump, d, replan=replan)
