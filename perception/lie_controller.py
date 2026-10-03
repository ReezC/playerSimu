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
_FF_LEAD_S = 0.10         # 前馈提前量：目标速度 × 它（≈ 传输+处理一拍延迟 ✓）
_FF_VMAX = 300.0          # ⭐ 前馈速度限幅（px/s）：正常目标几十 px/s —— 超过它说明
                          #   速度估计被**跳变**污染，前馈会把指令甩飞 ✗（闭环实测：
                          #   跳变 → 裸差分 560px/s → 前馈甩出 85px ⇒ 命中率掉 13% ✗）
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
                 ff_lead_s=_FF_LEAD_S, lost_hold=_LOST_HOLD, assume=None,
                 follow_gain=None):
        self.gain = tuple(gain) if gain else load_gain()
        self.deadzone = float(deadzone)
        self.max_step = float(max_step)
        self.ff_lead_s = float(ff_lead_s)
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

        # ③ 目标点 = 质心 + 速度前馈（补一拍延迟 ✓）；速度**限幅**（防跳变污染 ✓）
        v = vel if vel is not None else self._velocity(pos, ts)
        m = max(abs(v[0]), abs(v[1]))
        if m > _FF_VMAX:
            k = _FF_VMAX / m
            v = (v[0] * k, v[1] * k)
        tx = float(pos[0]) + v[0] * self.ff_lead_s
        ty = float(pos[1]) + v[1] * self.ff_lead_s

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
        """位置差分估速度（px/s ✓）；没有历史/时间为 0 ⇒ 0。"""
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
