"""一帧的第 7 段：**玩家定位**（阶段 1「编排收口」第二刀 ✓ 2026-10-20 ✓）。

搬自 `gui/live_thread.py::_run`（**行为零改动** ✓ —— 函数体原样搬过来，只把 `self.` 换成 `th.`
并把这些量变成参数 ✓ 注释一个字没动 ✓）：

  · **脚底偏移**灌进 `world_state.FOOT_OFFSET_PX`（那边不做反向 import ✓ 见它的说明）；
  · **框面积闸**（用户 2026-09-28）：上一拍锁定的框比"过去 N 秒基线"小太多 ⇒ 这一拍
    **不做查询**（`_loc = None` ⇒ 世界坐标**沿用上一拍** = "不定位"，而不是"定到错的地方"✗）；
  · **小地图那份世界坐标**（`_loc`）：来源=收流 ⇒ 取高频回路的最新一份（`_loc_slot_take` ✓）；
    来源=live ⇒ 每拍现算（`_locate_latest` ✓）；
  · **`PlayerTracker.update(player_cands, world=_world_now, low=player_low)`** ⇒ `player_box`；
  · **可信相机**（`camera()`）＋ 质量指标（`cam_valid` / `cam_rejected` / `cam_resid`）。

⚠⚠ **顺序不许换**：黄点那份世界坐标必须**先算**、再拿去挑框 ——
  `PlayerTracker` 用的是"**本帧黄点 + 上一帧的框/相机**"；同一帧的 world 与 box 相减会退化成
  "离上一帧框多远"（= 老判据）⇒ 等于白加 ✗（用户 2026-09-27 提的就是这个 ✓）。
⚠ 算不出来时 `world_now` 是 `None` ⇒ 追踪器**自动退回老判据** ✓（不猜 ✗）。
⚠ 这一段读的 `_last_player` / `_area_hist` / `_first_cam_t` 都是**跨帧**的容器（列表 / 单元素盒 ✓
  由调用方持有 ✓ 本段只读写里面的内容 ✓）；`player_tracker` 也是**跨帧**对象 ✓ 由调用方传。
⚠ 本模块**不 import gui** ✓（`map_locator` / `world_state` 都在 perception ✓ 由 `th._xxx` 那些方法
  自己懒导入 ✓ 这里只调它们）。

判据 `selftest_live_panel::t_frame_locate_stage`：脚底偏移灌进去 · 面积闸两种关法 ·
收流取"最新一份" / live 现算 · 世界坐标喂给 tracker / 算不出给 None · 相机的三个指标 ✓。
"""

import threading
import time

from core import behavior, perf


class LocOut:
    """这一段的产出（**本帧**数据 ✓）。

    · `player_box`  YOLO 挑出来的"我"（`None` = 这帧没找到 ✓ **不是**跟丢兜底那份 ✗）
    · `cam_now`     可信相机（`None` = 还没建立 ⇒ 下游退回老口径 ✓ 不许当 0 ✗）
    · `loc`         本拍取用的小地图定位结论（`None` = 面积闸拦住 / 这拍没算 ✓）
    · `world_now`   由 `loc` 解出的世界坐标（`None` ⇒ tracker 退回老判据 ✓）
    · `foot_off`    脚底偏移（画「玩家位置」箭头要用 ✓ 见 `_draw_show_frame` ✓）
    """

    __slots__ = ("player_box", "cam_now", "loc", "world_now", "foot_off")

    def __init__(self, player_box, cam_now, loc, world_now, foot_off):
        self.player_box = player_box
        self.cam_now = cam_now
        self.loc = loc
        self.world_now = world_now
        self.foot_off = foot_off

    def __repr__(self):
        return "<LocOut box=%r cam=%r loc=%s>" % (
            None if self.player_box is None else "yes",
            None if self.cam_now is None else "yes",
            "yes" if self.loc is not None else "None")


def step(th, player_tracker, player_cands, player_low, decision_settings, agent,
         _last_player, _area_hist, _first_cam_t, t_start):
    """跑这一拍的定位 ⇒ `LocOut`。

    :param player_tracker: 跨帧的 `PlayerTracker`（挑"我"、给相机 ✓）
    :param _last_player: 单元素盒 `[_last_player[0]]` —— **上一拍**的框（面积闸拿它算基线 ✓
                         ⚠ 本段只读，写完由调用方做 ✓）
    :param _area_hist: 框面积历史 `[(时刻, 面积)]`（跨帧 ✓ 时间窗筛 ✓ 2000 条内存上限）
    :param _first_cam_t: 单元素盒 `[None]` —— 第一次拿到可信相机的时刻（只写一次 ✓）
    :param t_start: 实时线程启动时刻（算 `first_cam_s` 用 ✓）
    """

    # ---- 玩家定位：YOLO 候选 → PlayerTracker 跟住「我」 ----
    # 用位置连续性（离预测位置最近）挑出操作者的框，避免「画面里多个
    # 玩家时取置信度最高的、把别人当成自己」；漏检用速度外推补框，
    # 连续跟丢才解锁重锁。max_jump 就是「玩家追踪阈值」参数。
    # ---- 小地图定位**先算一遍**（在挑玩家框之前）----
    # 用户 2026-09-27 提的："用小地图的玩家世界坐标辅助玩家框防抖" ⇒ 黄点是
    # **独立于 YOLO 的第二个传感器**（一个认画面、一个读小地图 ✓）⇒ 挑框时
    # 可以拿"**本帧我到底在哪**"当尺子 ✓（`PlayerTracker.update(world=…)` ✓）。
    # ⚠ 必须"**本帧黄点 + 上一帧的框/相机**"才成立：同一帧的 world 与 box 相减
    #   会退化成"离上一帧框多远"（= 老判据），等于没加 ✗ —— 所以定位得先跑 ✓。
    # 算不出来时 `_world_now` 就是 None ⇒ 追踪器**自动退回老判据** ✓（不猜 ✗）。
    # ---- ⭐⭐ **「玩家位置」组：每帧现读**（用户 2026-09-28 ✓）----
    #   ① **脚底偏移**灌进 `world_state` 的镜像（那边不做反向 import，说明在
    #      `perception/world/world_state.py` 的 `FOOT_OFFSET_PX` 上 ✓）；
    _ploc_fo = float(getattr(decision_settings,
                                                      "player_foot_offset_px", 0) or 0)
    try:
            from perception.world import world_state as _ws_mod
            _ws_mod.FOOT_OFFSET_PX = _ploc_fo
    except Exception:
            pass
    #   ②③④ **框面积闸**（用户原话："当前框面积 ≤ 近期滚动基线 × (1 − 容差%)
    #      ⇒ 这一拍推迟 / 不做查询 ✓（**拦在查询之前**，而不是靠放宽挑面 ✗）"）。
    #      ⚠ 只能拿**上一拍锁定的框**说事：本帧的框要等 `_locate_mmap` 的出参才
    #        挑得出来（黄点是"辅助挑框"的尺子 ✓ 见上面那段）⇒ 这一拍先判、再查 ✓。
    #      ⚠ 两个"关"的口子：**容差 0** / **基线还没攒够 3 拍** ⇒ 照常查询 ✓
    #        （老行为一字不变 ✓）；**框面积最小占比 0** ⇒ 那条也关 ✓。
    _gate_closed = False
    _area_prev = 0.0
    _pb_prev = _last_player[0]
    if _pb_prev is not None and len(_pb_prev) >= 6:
            _area_prev = float(_pb_prev[4]) * float(_pb_prev[5])
    if _area_prev > 0.0:
            # ⭐⭐ **基线窗口改成"过去 N 秒"**（用户 2026-09-28 ✓ 原话："『面积基线
            #   窗口』，这个『拍』是什么？是多久？**需要可量化的描述**"✓）——
            #   存 `(时刻, 面积)` 并按**时间窗**筛 ✓：**拍数依赖帧率** ✗
            #   （30 拍在 30fps 是 1 秒、在 15fps 是 2 秒 ⇒ 说不清多久 ✗），秒才可量化 ✓。
            _t_now = time.monotonic()
            _area_hist.append((_t_now, _area_prev))
            if len(_area_hist) > 2000:      # 只当内存上限（真筛选靠时间窗 ✓）
                    del _area_hist[:1000]
            _pm = float(getattr(decision_settings,
                                                    "player_box_min_area_pct", 0.0) or 0.0)
            _pt = float(getattr(decision_settings,
                                                    "player_box_area_tol_pct", 0.0) or 0.0)
            _bs = max(0.5, float(getattr(
                    decision_settings, "player_box_area_base_s", 3.0) or 3.0))
            _win = [_a for _t, _a in _area_hist if _t_now - _t <= _bs]
            _base = (sum(_win) / float(len(_win))) if _win else 0.0
            # ⭐⭐ **「框面积最小占比」改成"占基线"**（用户 2026-09-28 ✓ 原话：
            #   "『框面积最小占比』应该是占**过去 n 秒内平均面积**的比例"✓）——
            #   原来跟**画面面积**比 ✗（不同地图 / 分辨率下人物框本来就不一样大 ⇒
            #   跟画面比**没有可比性** ✗）。
            if _pm > 0.0 and _base > 0.0:
                    if _area_prev < _base * _pm / 100.0:
                            _gate_closed = True      # 比基线小太多 ⇒ 这一帧**检测无效** ✓
                            perf.count("player_box_too_small")
            if not _gate_closed and _pt > 0.0 and len(_win) >= 3:
                    if _base > 0.0 and _area_prev <= _base * (1.0 - _pt / 100.0):
                            _gate_closed = True      # 比近期基线小 ⇒ 这一拍不查
                            perf.count("player_area_below_base")
    #      ⚠ 拦住时 `_loc = None` ⇒ 下面 `apply_to_player` 不跑 ⇒ **世界坐标沿
    #        用上一拍** = 这一拍"**不定位**"✓（而不是"定到错的地方"✗）。
    # ⭐⭐ **高频定位回路**（用户 2026-10-03 ✓ 原话："小地图作为权威世界坐标，
    #   应该尽可能用较高的帧率去更新"）：来源=收流 时才起 —— 它**每收到一帧
    #   小地图就定位一次**，主回路只**取最新一份**（见 `_mmap_loc_loop` ✓）。
    #   以前定位绑在**推理拍**上（≈30 次/秒）⇒ A 机推 60fps 也有一半被丢 ✗；
    #   定位只要 ~0.5ms ⇒ 现在更新率 = 推流帧率 ✓。
    if (th._mmap_src != "live" and th._mmap_mid
                    and th._loc_thread is None):
            th._loc_thread = threading.Thread(
                    target=th._mmap_loc_loop, daemon=True, name="mmap-locate")
            th._loc_thread.start()
            perf.note("mmap_loc_loop", "on")
    # ⚠ `_gate_closed`（框面积闸）**照旧拦住这一拍的消费** ✓ 行为一字不变 ✓：
    #   那条闸说的是"**这一拍检测无效**"（框太小 ⇒ 别拿它挑框 ✗），不是
    #   "小地图不可信" ⇒ 拦住的这一拍不取用；**但回路照旧在跑** ✓ ⇒ 下一拍
    #   拿到的坐标只会更新鲜 ✓（收益就在这儿 ✓）。
    if _gate_closed:
            _loc = None
    elif th._mmap_src == "live" or th._loc_thread is None:
            _loc = th._locate_latest()      # 没有独立帧源 ⇒ 照旧每拍现算 ✓
    else:
            _loc = th._loc_slot_take()      # ⭐ 回路给的最新一份 ✓
    _world_now = None
    if _loc is not None:
            _wx0, _wy0 = _loc.get("world_x"), _loc.get("world_y")
            if _wx0 is not None and _wy0 is not None:
                    _world_now = (float(_wx0), float(_wy0))

    player_tracker.max_jump = max(
            1.0, float(decision_settings.player_track_jump))
    # ⭐ **把"此刻按着哪个方向"喂给追踪器**（用户 2026-10-01 ✓
    #   「输入感知的运动预测」）：角色是被按键驱动的 ⇒ 漏检那几帧照着
    #   **意图**推演（按着右就在加速、松手就在地面摩擦下减速），比"拿上一帧
    #   位移线性外推"准得多 —— 玩家中途变向 / 停住时，老口径会朝错的方向
    #   一路跑 ⇒ 挑候选挑错、补出来的框也错 ✗（两处共用同一个预测位置）。
    #   模型与它的边界（"意图 ≠ 事实"）都写在 `PlayerTracker.set_input` ✓。
    # ⚠ 取的是**按键名快照**：`keys.pressed()` 返回 set 拷贝、跨线程读安全 ✓
    #   （画按键帽那处也是这么取的 ✓）。拿不到（替身 / 老环境）⇒ 不喂
    #   ⇒ 追踪器 `_input_aware` 保持假 ⇒ **老口径一字不变** ✓。
    # ⚠ 键名是**映射后的名字**（`keymap` 里就是 left/right/up/down ✓）。
    try:
            _pk = agent.keys.pressed()
    except Exception:                       # noqa: BLE001
            _pk = ()
    player_tracker.set_input(
            1 if "right" in _pk else (-1 if "left" in _pk else 0),
            1 if "up" in _pk else (-1 if "down" in _pk else 0))
    _t_trk = time.perf_counter()
    # ⭐ `player_low` = 玩家的**低分档**（2026-10-11 ✓）：它能不能被用由
    #   tracker 说了算（**必须已锁定 + 黄点对得上** ✓ 见 `PlayerTracker.update` ✓
    #   —— 玩家低分框很可能是**别人** ✗ 所以比怪多两道前提 ✓）。
    player_box = player_tracker.update(player_cands, world=_world_now,
                                                                        low=player_low)
    # ⭐ **追踪+相机对账阶段耗时**（同上：`pipe_ms` 减掉 `infer_ms`/`agent_ms`
    #   和这一段，剩下的才是"候选整理 + 状态机填充"那种零碎）
    perf.ms("track_ms", _t_trk)
    # ⭐ **可信相机**（用户 2026-09-29 任务 2 ✓）：追踪器拿黄点的**权威世界
    #   坐标**核对过玩家框，只把"对得上"那几拍采纳成相机 ✓ ⇒ 下游所有
    #   "画面 → 世界"都用它（`screen_to_world` / 怪的集合解析器 ✓），
    #   玩家框抖 Δ 时不再把 Δ 原样灌进每只怪的世界坐标 ✓。
    #   ⚠ `None` = 还没建立 ⇒ 下游退回老口径现算，行为一字不变 ✓（不许当 0 ✗）。
    _cam_now = player_tracker.camera()
    # ⭐ **质量指标**（参考 Maple_xfeat 的 `valid_camera_fraction`）：耗时低
    #   ≠ 质量好。每帧记 1.0/0.0 ⇒ `perf.log` 的分位数天然就是"有效比例" ✓。
    #   · `cam_valid`    = 这一拍有没有可信相机（p50 = 有效比例）
    #   · `cam_rejected` = 这一拍"框与黄点对不上"（p50 = 拒绝比例）
    perf.sample("cam_valid", 1.0 if _cam_now is not None else 0.0)
    perf.sample("cam_rejected", 1.0 if player_tracker.cam_rejected else 0.0)
    if _cam_now is not None and _first_cam_t[0] is None:
            _first_cam_t[0] = time.perf_counter()
            perf.note("first_cam_s", round(_first_cam_t[0] - t_start, 2))
    if player_tracker.cam_rejected:
            # 这一拍"框与黄点对不上" ⇒ 相机冻着用旧的（**打点**，排查要用：
            # 一直涨说明玩家框锁错了 / 标定偏了 / 黄点在骗人 ✓）
            perf.count("cam_reject")
    # ⭐ 相机与"两个传感器差多少"也写进**行为日志**（用户 2026-09-29 任务 2）：
    #   `cam_resid`（世界像素）是"玩家框那一路稳不稳"唯一的数；"世界坐标
    #   整体偏"那类排查要看的就是它（差几十像素 ⇒ 就是它把怪推了一层 ✓）。
    if _cam_now is not None:
            behavior.sample("cam_x", float(_cam_now[0]), min_gap=1.0)
            behavior.sample("cam_y", float(_cam_now[1]), min_gap=1.0)
    if player_tracker.cam_resid is not None:
            behavior.sample("cam_resid", float(player_tracker.cam_resid),
                                            min_gap=1.0)
    #: ⚠ `n_boxes = k` 留**在调用方**（`k` 是第 6 段产出的怪框数 ✓ 本段不产它 ✓）
    return LocOut(player_box=player_box, cam_now=_cam_now, loc=_loc,
                  world_now=_world_now, foot_off=_ploc_fo)
