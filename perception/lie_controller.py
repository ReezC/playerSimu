# -*- coding: utf-8 -*-
"""M2c：测谎小游戏的**鼠标闭环控制律**（纯逻辑，不碰硬件 ✓ 好测 ✓）。

契约（docs/测谎设计.md §M2c）：
  · 控制律 = **P 控制**（目标质心 − 光标）× 增益 + **目标速度前馈**（抵消一帧延迟）；
  · **丢失保护**：丢 N 帧 ⇒ 光标**原地停住**（绝不乱飞 ✗），持续丢失 ⇒ 记事件，
    由成功/失败弹窗兜底 ✓；
  · 鼠标只有**相对移动**（固件 HID ✓ `decision/input.py::mouse_move(dx,dy)`）——
    所以控制器输出的是「这一步发多少」(dx, dy)，不是绝对坐标 ✓。

光标位置从哪来：`cursor`（游戏画面像素 ✓）——实机由**光标检测**给（M2c 实跑时接
`live_thread` 的原始帧 + 帧差/模板），离线仿真用模拟光标给 ✓；拿不到就退**航位推算**
（`cursor=None` ⇒ 用自己发过的指令累加 ✓ 标定准才可用，见 `tools/lie_closed_loop.py` ✓）。
"""
import json
from pathlib import Path

#: 增益标定文件的默认位置（`tools/mouse_gain_calib.py` 写 ✓）
GAIN_PATH = Path(__file__).resolve().parent.parent / "config" / "mouse_gain.json"

# ---- 参数（集中在头部 ✓ 实跑调这里）----
_DEADZONE = 3.0           # 死区（游戏像素）：|误差| 小于它不发指令（防抖/防过冲 ✓）
_MAX_STEP = 200.0         # 单步限幅（指令单位）：防"一帧飞很远"（丢帧后误差巨大 ✓）
_FOLLOW_GAIN = 1.0        # ⭐⭐⭐ **「鼠标跟随效率倍率」默认值**（用户 2026-10-03 ✓ 原话："能否
                          #   开放一个系数配置，调追踪器跟上圆心的效率倍率？" ✓ + "**我不想影响
                          #   位置计算**，只调追踪器（较大的白描边圈绿圆）" ✓）：P 控制每拍把
                          #   误差（光标 → 目标点）**消掉这么多倍** ✓ —— 作用在**指令输出**上，
                          #   ⇒ 只改"**模拟鼠标往圆心走多快**" ✗ **不碰 `LieTracker` 的任何位置
                          #   计算** ✓（演示窗里那个"白圈 + 绿实心点"的大点 = 控制器输出 ✓）。
                          #   · **1.0（默认）= 全量**（一拍尽量贴上去 = 老行为一字不变 ✓，
                          #     实际仍受 `_MAX_STEP` 限幅与整数取整 ✓）；
                          #   · **< 1** ⇒ 每拍只走剩余误差的一部分 ⇒ 光标**渐进**贴上圆心 ✓
                          #     （拟人 ✓ ⚠ 太小会因整数取整停在几 px 的残差上 ✗）；
                          #   · **> 1** ⇒ 过冲（在"一拍延迟 + 前馈"的闭环里能提前压住滞后 ✓
                          #     代价是抖 ✓）。
                          #   ⚠ 与 `gain`（**标定**：1 指令单位 ⇒ 多少游戏像素 ✓）不是一回事 ✗
                          #     —— gain 是"换算尺"、本项是"走多快" ✓。
#: ⭐⭐⭐⭐⭐ **前馈提前量** —— ⚠⚠⚠ **单位是「拍」，不是「秒」** ✗✗（用户 2026-10-04 ✓
#:   他点了"A" ✓ 见下面那段单位订正 ✓）。
#:   · 取 **1.0** = **补整整一拍**（= `目标点 = 报告位置 + 目标每拍位移` ✓）
#:     —— 这才对应"**指令晚一拍落地**"那个延迟 ✓（闭环里就是这个延迟 ✓）。
#:   ⚠⚠ **原来写的是 `_FF_LEAD_S = 0.10` 并声称单位是「秒」** ✗✗ —— 而喂进来的 `vel`
#:     其实是 **px/拍**（= `MotionTracker.vel` = "每拍位移"的 EMA ✓ 由 `_moves` 算 ✓）
#:     ⇒ 实际只补了 **0.10 拍** ✗ ⇒ **前馈几乎等于没有** ✓
#:     （实测 `10月1日`：`|vel|` 中位 **6.7** vs「报告位置每拍真走」**6.5** ⇒ 比值 **1.03**
#:      ⇒ 单位确实是 px/拍 ✓✓；6.7px/拍 下那 0.10 拍只多挪 **0.67px** ✗ 等于摆设 ✓）
#:   ⚠ 调法：**> 1 会过冲**（目标急停/拐弯时鼠标越过绿圈 ✗）；**< 1 压不住滞后** ✓
#:     （先按 1.0 ✓ 看"急停那几拍鼠标会不会冲出去"再定 ✓）。
_FF_LEAD = 1.0
#: ⭐ **前馈速度限幅**（单位 **px/拍** ✗ 不是 px/s ✗）—— 防"速度估计被跳变污染"✓。
#:   ⚠⚠ **原值 300 是按「px/s」写的** ✗ ⇒ 换算到 px/拍 相当于 **5 拍/s × 300** 的天量
#:     ⇒ **永远不会触发** ✗（限幅形同虚设 ✓）。实测：典型 **6.7 px/拍** ✓、混战时也就
#:     20~40 px/拍 ✓ ⇒ 取 **60**（留 ~1.5 倍余量 ✓）：**真跳变**（一拍窜上百 px ✓）会被削掉 ✓。
_FF_VMAX = 60.0
_LOST_HOLD = 8            # 连续丢这么多帧 ⇒ 判"持续丢失"（记事件 ✓ 同 tracker 口径 ✓）


def load_gain(path=None):
    """读增益标定 ⇒ `(gx, gy)`；没有/坏了 ⇒ `(1.0, 1.0)`（退 1:1，不拦 ✓）。"""
    p = Path(path) if path else GAIN_PATH
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        gx, gy = float(d.get("gain_x", 1.0)), float(d.get("gain_y", 1.0))
        if gx <= 0 or gy <= 0:
            return 1.0, 1.0
        return gx, gy
    except Exception:                     # noqa: BLE001 —— 坏文件当没标定过 ✓
        return 1.0, 1.0


class LieMouseController:
    """P 控制 + 速度前馈 + 丢失保护（状态：光标估计 / 丢失计数 / 事件）。

    用法（实机）：
        c = LieMouseController()
        out = tracker.process(frame, ts=t)          # → pos / state
        dx, dy = c.step(out["pos"], ts=t, cursor=cursor_px)
        if dx or dy:
            dinput.mouse_move(dx, dy)
        if c.lost_event:                            # 持续丢失 ⇒ 交给弹窗兜底 ✓
            ...
    """

    def __init__(self, gain=None, deadzone=_DEADZONE, max_step=_MAX_STEP,
                 ff_lead=_FF_LEAD, lost_hold=_LOST_HOLD, assume=None,
                 follow_gain=None):
        self.gain = tuple(gain) if gain else load_gain()
        self.deadzone = float(deadzone)
        self.max_step = float(max_step)
        # ⚠ 参数名**从 `ff_lead_s` 改成 `ff_lead`** ✗（`_s` 那个后缀一直在暗示"秒" ✗
        #   而它**从来就不是秒** ✓ 名字本身在骗人 ⇒ 一起改掉 ✓ 见模块头 `_FF_LEAD` ✓）。
        self.ff_lead = float(ff_lead)
        self.lost_hold = int(lost_hold)
        # ⭐⭐⭐ **鼠标跟随效率倍率**（用户 2026-10-03 ✓ 见 `_FOLLOW_GAIN` ✓）：每拍把"光标 →
        #   目标点"的误差消掉这么多倍 ✓。⚠ **0 会被夹成一个极小值**（= 光标几乎不动 ⇒
        #   永不收敛 ✗ 无实际意义 ✓）；上界 5 防手输离谱值 ✓；`None` ⇒ 模块默认 1.0 ✓。
        self.follow_gain = (float(_FOLLOW_GAIN) if follow_gain is None
                            else max(0.01, min(5.0, float(follow_gain))))
        # ⚠ **起点假设**（无光标反馈时的航位推算起点 ✓）：不给就 (0,0)——
        #   ⛔ 千万别默认成"光标就在目标上"（误差恒 0 ⇒ 永远不发指令 ✗ 踩过思路陷阱 ✓）。
        #   实机建议给**弹窗视口中心**（光标通常就近 ✓ 见 M3 的视口矩形）。
        self.assume = assume
        self.cursor = None        # 光标估计（游戏像素 ✓ 有反馈就吃反馈 ✓）
        self.lost_n = 0           # 连续丢失帧数
        self.lost_event = False   # ⭐ 持续丢失事件（调用方读一次自己清 ✓）
        self.held = False         # 当前是否处于"停住"保护
        self.last_cmd = (0, 0)
        self._last = None         # 上一帧 (pos, ts)（差分估速度 ✓）

    # ---- 主入口 ----
    def step(self, pos, ts=None, vel=None, cursor=None):
        """一步：给目标位置 ⇒ 返回本步要发的 `(dx, dy)`（**整数指令单位** ✓）。

        `pos`：目标质心（游戏像素；`None` = 本帧丢了 ✓）；
        `vel`：目标速度（px/s，前馈用；不给就用位置差分自己估 ✓）；
        `cursor`：光标的**实测**位置（游戏像素；不给 = 航位推算 ✓）。
        """
        # ① 反馈优先（实测光标 ✓）——喂进来就同步估计（标定误差靠它自校正 ✓）
        if cursor is not None:
            self.cursor = (float(cursor[0]), float(cursor[1]))

        # ② 丢失保护：**原地停住**（不发指令 ⇒ 光标不动 ✓ 绝不乱飞 ✗）
        if pos is None:
            self.lost_n += 1
            self.held = True
            self.last_cmd = (0, 0)
            if self.lost_n >= self.lost_hold:
                self.lost_event = True        # 持续丢失 ⇒ 记事件（弹窗兜底 ✓）
            return (0, 0)
        self.lost_n = 0
        self.held = False

        # ③ ⭐⭐⭐⭐ **目标点 = 质心 + 速度前馈 × 一拍**（补"指令晚一拍落地"那个延迟 ✓）——
        #   ⚠⚠ **`vel` 的契约 = 「px/拍」** ✗✗（**不是 px/s** ✗ 见模块头 `_FF_LEAD` 那段 ✓）：
        #     喂进来的是 `MotionTracker.vel`（= "每拍位移"的 EMA ✓ 见 `lie_motion._moves` ✓）
        #     —— 实测比值 1.03 ⇒ 确认是 px/拍 ✓✓。
        #   ⚠⚠ **没给 `vel` ⇒ 一律"不前瞻"** ✗✗（原来退回 `self._velocity` ✓ 而那个是 **px/s** ✗
        #     ⇒ **两个不同单位混进同一个乘法** ✗✗ ⇒ 宁可**不前馈** ✓ 也不能甩鼠标 ✓。
        #     这条路径只在"轨迹刚出生、还没配上过一次"时走到 ✓ —— 那几拍**本来就没有可信速度** ✓
        #     ⇒ 不前馈反而是对的 ✓。（⚠ `_velocity` **保留不删** ✓ 要回退随时能接 ✓。）
        if vel is not None:
            _vx, _vy = float(vel[0]), float(vel[1])
            _m = max(abs(_vx), abs(_vy))
            if _m > _FF_VMAX:                      # 限幅（px/拍 ✓ 见 `_FF_VMAX` ✓）
                _k = _FF_VMAX / _m
                _vx, _vy = _vx * _k, _vy * _k
            tx = float(pos[0]) + _vx * self.ff_lead
            ty = float(pos[1]) + _vy * self.ff_lead
        else:
            tx = float(pos[0])
            ty = float(pos[1])

        # ④ 没有光标反馈 ⇒ 航位推算（起点按 `assume` ✓ 缺省 (0,0)；**不许**默认在目标上 ✗）
        if self.cursor is None:
            self.cursor = tuple(self.assume) if self.assume else (0.0, 0.0)
        ex = tx - self.cursor[0]
        ey = ty - self.cursor[1]

        # ⑤ 死区（防抖 ✓）
        if abs(ex) < self.deadzone and abs(ey) < self.deadzone:
            self.last_cmd = (0, 0)
            return (0, 0)

        # ⑥ P 控制 + 限幅（游戏像素误差 ⇒ 指令单位 ✓）
        #   ⭐⭐⭐ **「鼠标跟随效率倍率」`follow_gain`**（用户 2026-10-03 ✓ 见 `_FOLLOW_GAIN` ✓）：
        #     每拍消掉误差的**多少倍** ✓ —— 1.0 = 全量（老行为 ✓）；< 1 ⇒ 渐进贴上 ✓；
        #     > 1 ⇒ 过冲（对抗一拍延迟 ✓）。⚠ 只作用在**指令输出**上 ⇒ 不碰 `LieTracker` ✓。
        gx, gy = self.gain
        dx = ex / gx * self.follow_gain
        dy = ey / gy * self.follow_gain
        dx, dy = self._clamp_step(dx, dy)
        self.last_cmd = (int(round(dx)), int(round(dy)))
        # 航位推算：**自己发过的指令**累回去（有反馈时会被 ① 覆盖 ✓ 不会漂 ✗）
        self.cursor = (self.cursor[0] + self.last_cmd[0] * gx,
                       self.cursor[1] + self.last_cmd[1] * gy)
        return self.last_cmd

    # ---- 工具 ----
    def _clamp_step(self, dx, dy):
        m = max(abs(dx), abs(dy))
        if m > self.max_step:
            k = self.max_step / m
            dx, dy = dx * k, dy * k
        return dx, dy

    def _velocity(self, pos, ts):
        """位置差分估速度（**px/s** ✗）；没有历史/时间为 0 ⇒ 0。

        ⚠⚠ **`step` 现在不用它了** ✗（用户 2026-10-04 ✓ 原话"**A**" ✓ 见模块头 `_FF_LEAD` ✓）——
          它回的是 **px/s** ✗，而 `step` 吃的 `vel` 是 **px/拍** ✗ ⇒ 两个单位混在一个乘法里
          就是灾难 ✓ ⇒ 现在"**没给 `vel` 就不前瞻**" ✓（宁可少补一拍 ✓ 也不甩鼠标 ✗）。
        ⚠ **保留不删** ✓：要回退、或将来有"真按 px/s 喂"的调用方，随时能接上 ✓。
        """
        if self._last is None or ts is None or self._last[1] is None:
            self._last = (pos, ts)
            return (0.0, 0.0)
        dt = ts - self._last[1]
        prev = self._last[0]
        self._last = (pos, ts)
        if not dt or dt <= 0:
            return (0.0, 0.0)
        return ((pos[0] - prev[0]) / dt, (pos[1] - prev[1]) / dt)

    def sync_cursor(self, xy):
        """外部（光标检测 ✓）给绝对位置时同步一下（防止推算漂 ✓）。"""
        self.cursor = (float(xy[0]), float(xy[1]))

    def reset(self):
        self.cursor = None
        self.lost_n = 0
        self.held = False
        self.lost_event = False
        self.last_cmd = (0, 0)
        self._last = None
