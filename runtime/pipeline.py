"""一帧的**编排**（阶段 1「编排收口」✓ 2026-10-13 起 ✓）。

这一层要收的三样东西（用户拍板的清单 ✓）：

    ① **一帧的阶段顺序** —— 哪几段、按什么顺序、哪一段能早退（⏳ 还没收完 ✓）；
    ② **限速闸** —— 每秒最多跑多少拍重活（✅ 已收，见 `HeavyGate` ✓）；
    ③ **接管状态** —— 界面切换 / 测谎 / 换图的接管与还权（⏳ 还没收 ✓）。

⚠ 这三样**都不是"算法"**，是**调度**：它们坏掉的样子都是"看着能跑、但节奏/归属错了" ✗
（跳拍不喂时间拍 ⇒ 攻击节奏漂移 ✗；接管状态记错 ⇒ 操作权还给错的模块 ✗）—— 所以每搬一样，
都要有一条**不用开游戏就能跑**的用例钉住 ✓（见 `tools/selftest_live_panel.py` ✓）。
"""
import time

from core import perf


class HeavyGate:
    """**重活闸**（限速）：给"每秒的活量"封顶，不是"等机器空出来" ✓。

    用户 2026-10-10 ✓ 原话："**我就是想在挂游戏+浏览器+音乐+steam+玩大型游戏的时候，
    实时还能流畅跑**" ✓。

    为什么需要它（实测 ✓）：`pipe_ms` 里**推理占 66~70%**（`infer_ms` 13ms / `pipe_ms` 20ms ✓），
    而输入推流是 **60fps** ⇒ 天棚只比输入高一点点 ⇒ 机器上再有人抢 GPU/CPU ⇒ **必然积压** ✗。
    砍掉一半的拍 ⇒ 每秒的 GPU/CPU 活量减半 ✓，而**延迟不受影响** ✗（永远取最新帧 ✓
    —— 跳过的帧直接丢，同 `_LatestSlot` 的口径 ✓）。

    到点才跑「推理＋追踪＋决策＋画框」那条重链；**不到点的帧**：

      · **只喂时间拍**（`agent.tick_timing` ✓ = 输出CD / delay / 跳间隔那条序列 ✓）
        —— 它走**本机绝对时钟** ✓ ⇒ 跳拍**不影响精确度** ✓（同"暂时没新帧"那条 ✓）；
      · 这一帧**直接丢** ⇒ 我们永远只处理**最新**帧 ✓ ⇒ **延迟不升** ✓。

    ⚠⚠ **跳过的拍必须照喂时间拍**（上面那条 ✓）—— 漏了就是"限了速 ⇒ 攻击节奏变慢 /
      延迟漂移" ✗（很难查 ✓ 用例钉着 ✓）。
    ⚠ `0` = **不限制** ⇒ 这一整段等于不存在 ✓（**老行为一字不变** ✓ 默认值必须是 0 ✓）。

    搬自 `gui/live_thread.py::_heavy_gate`（2026-10-13 抽成方法 ✓ 2026-10-15 搬进本包 ✓
    **行为零改动** ✓）：计时与计数打点（`timing_ms` / `timing_calls` / `skip_heavy` ✓）口径一字不动 ✓。
    """

    def __init__(self, fps_limit=0.0):
        #: 每秒最多跑多少拍重活；`0` = 不限制（**老行为** ✓）
        self.fps_limit = 0.0
        #: 上一次**放行**（跑了重活）的时刻；0.0 = 从没跑过 ⇒ 第一拍必放行 ✓
        self.last_pass_at = 0.0
        self.fps_limit = float(fps_limit or 0.0)

    def skip_now(self, now, active, agent, ws):
        """这一拍**该不该跳过重活**？⇒ True = 跳过（调用方 `continue` ✓）。

        :param now: 当前时刻（`time.perf_counter()` ✓ 与 `last_pass_at` 同一时钟 ✓）
        :param active: 现在跑不跑重活（主循环传的是"有没有在推理" `_infer.is_set()` ✓）
        :param agent: 只用来喂**时间拍**（`tick_timing` ✓）—— 它不要知道"闸"的存在 ✓
        :param ws: 一起喂进去的那份世界快照（**冻结过的** ✓ 见 `WorldState._freeze` ✓）

        ⚠ 返回 True 时**副作用三件**已经做完（喂时间拍 + 打点 + 记计数 ✓）——
          调用方**只负责 `continue`** ✓，别再自己补喂一次 ✗（会双倍喂 ⇒ 节奏偏快 ✗）。
        """
        if self.fps_limit <= 0.0 or not active:
            return False
        if now - self.last_pass_at < 1.0 / self.fps_limit:
            _t_t = time.perf_counter()
            try:
                agent.tick_timing(ws)
            except Exception:              # noqa: BLE001 —— 时间拍出事也不许炸回路 ✗
                pass
            perf.ms("timing_ms", _t_t)
            perf.count("timing_calls")
            perf.count("skip_heavy")
            return True
        self.last_pass_at = now
        return False


class FrameCtx:
    """**一帧的上下文**：阶段之间唯一的"传递物"（口径 B ✓ 用户 2026-10-15 拍板 ✓）。

    它解决的是那件把 `_run` 撑到 1651 行的事 ✗：一帧要用到的东西（画面、检出、玩家框…）
    原先**生在循环体里、被几十处代码读改写**，想把某一段搬出去就得"把几十个变量传进去"✗
    —— 已有的例子就摆在眼前：`_decide_frame_on_infer` 一抽出来就是 **19 个参数** ✗，而
    `pot_last` / `e2e_ref` 那几个还得**拿列表当"传值出口"**（`pot_last[0] = …` ✓）。

    ⇒ 立这个盒子之后，每段只收 `(ctx)` ✓：加字段不动签名 ✓、抽出来的段能**拿一个假盒子单测** ✓。

    ⚠⚠ **什么进盒子、什么不进**（这条最要紧 ✓ 判断依据如下）：

      · **进**：**只属于"这一帧"**的东西 —— 这一帧的图像 / 原生副本 / 时间戳 / 这一帧算出
        的检出与玩家框 / 这一帧要喂给 Agent 的那份世界快照 ✓；
      · **不进**：**跨帧活着**的东西 ✗ —— 帧计数、上一帧时刻、最近 60 个帧间隔的窗口、
        配置（`_pipe_fps` 之类）、`size_ref` / `cap_ref` 这种"回填用的小列表"✓。
        它们**留在 `LiveThread` 自己身上** ✓（各阶段是方法，够得着 `self.` ✓）。
        判据一句话：**"下一帧还会用吗？"** —— 会 ⇒ 不进盒子 ✗（进了就变成跨帧状态藏在
        一个看着像"仅本帧"的名字里 ✓ 那种 bug 最难查 ✓）。

    ⚠⚠ **三条纪律**（`selftest_live_panel.t_frame_ctx_discipline` 钉着 ✓）：
      ① **字段写死**：用 `__slots__` ⇒ 写一个没声明的字段**当场 `AttributeError`** ✓
         —— 不然它就是个"谁都能往里塞东西的袋子" ✗（= 换了个名字的全局变量 ✓ 比现在更难查 ✓）；
      ② **一帧一个**（**不跨帧复用** ✗）：复用省的那点分配，跟"读到上一帧的怪框"这种
         静默 bug 比不值 ✓（这条和 `ws._freeze()` 是同一个思路：宁可多花一点，也别留下
         "上一帧的东西还在这"的可能 ✓）；
      ③ 它是**可变**的 ✓（阶段就是要往里写结果的 ✓）⇒ 谁写什么，靠**阶段边界**约束 ✓：
         一个阶段只许写它自己那份产出 ✗（各阶段自己的文档里写明 ✓）。
    """

    __slots__ = ("f", "vis", "raw", "t_take", "t_pipe", "ws")

    def __init__(self, f=None, vis=None, raw=None, t_take=None, t_pipe=None, ws=None):
        #: 这一帧（`_LatestSlot.take()` 交出来的那个 ✓：`.image` / `.frame_id` /
        #: `.t_recv_mono` ✓）；"这一拍没帧"的那条路**不建盒子** ✓（根本进不了阶段 ✓）
        self.f = f
        #: BGR 画面（`f.image` ✓）—— ⚠ 下面会往它上面**就地**画框 ⇒ 画完只剩"画过的" ✗
        self.vis = vis
        #: 原生帧副本（**画之前**拷的那份 ✓ 取帧做测量的几个功能量的是像素 ✓ 见 `_run` 里那段）
        self.raw = raw
        #: 取帧那一刻（`time.perf_counter` ✓ 算槽等待 / 端到端都要它 ✓）
        self.t_take = t_take
        #: 「收到这一帧 → 决策完」那段计时的**起点** ✓
        self.t_pipe = t_pipe
        #: 喂 Agent 的那份世界快照（**冻结过的** ✓ 见 `perception/world_state.py` 的 `_freeze` ✓）
        self.ws = ws

    def __repr__(self):
        """日志里能一眼看出"这一帧是哪一帧" ✓（别让它显示成一串内存地址 ✗）。"""
        _fid = getattr(self.f, "frame_id", None)
        _size = (self.vis.shape[:2] if getattr(self, "vis", None) is not None else None)
        return "<FrameCtx frame=%r size=%r>" % (_fid, _size)
