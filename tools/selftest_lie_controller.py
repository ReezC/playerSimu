# -*- coding: utf-8 -*-
"""M2c 鼠标闭环控制律自检（**纯逻辑 + 模拟光标**，不碰硬件 ✓）。

模拟闭环世界（知道答案 ✓）：
  · 目标沿已知轨迹匀速运动（~55px/s ✓ 同真实量级）；
  · 光标 = 真位置，吃指令后 `cursor += cmd × gain_true`（+ 量化取整 ✓ + 一帧反馈延迟 ✓）；
  · 控制器只看**上一帧的反馈光标**（= 真实系统里"光标检测"给的东西 ✓）。

钉五件：
  ① P 控制收敛：从任意起点几步内贴上目标（残余误差 ≤ 死区 + 一拍滞后 ✓）；
  ② 死区不发指令（目标静止时**一步都不发** ✓ 防抖 ✓）；
  ③ 单步限幅（起点极远时**一步不超过 _MAX_STEP** ✓ 防"一帧飞很远" ✗）；
  ④ **丢失保护**：目标丢 ⇒ 一步不发（光标**原地不动** ✓）、持续丢 ⇒ `lost_event` ✓；
  ⑤ **增益标定错也能收敛**（真增益 1.6× 配置 ⇒ 靠反馈自校正 ✓；反向验证：**掐掉
     反馈**（纯航位推算）⇒ 同一场景系统性偏掉 ⇒ 证明"标定要同状态做"和小偏差
     都由反馈兜住 ✓）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perception.lie_controller import LieMouseController  # noqa: E402

_FAILED = []


def check(cond, msg):
    if cond:
        print("  [OK] %s" % msg)
    else:
        print("  [NG] %s" % msg)
        _FAILED.append(msg)


class Sim:
    """模拟光标世界：目标匀速直线 + 光标吃指令（增益/量化/一帧延迟 ✓）。"""

    def __init__(self, gain_true=(1.0, 1.0), start=(40.0, 300.0),
                 tgt0=(600.0, 120.0), vel=(55.0, 0.0), delay=1):
        self.gain_true = gain_true
        self.cursor = start          # 光标真位置（游戏像素 ✓）
        self.cursor_delay = start    # 控制器能看到的（延迟一拍 ✓）
        self.t = 0.0
        self.tgt0 = tgt0
        self.vel = vel
        self.delay = delay
        self.pending = []            # 待生效的指令（延迟队列 ✓）
        self.moves = 0               # 发过几步（含非零指令 ✓）

    def target(self):
        return (self.tgt0[0] + self.vel[0] * self.t, self.tgt0[1] + self.vel[1] * self.t)

    def apply(self, cmd):
        """把控制器这一步的指令排进延迟队列（下一次 advance 生效 ✓）。"""
        if cmd[0] or cmd[1]:
            self.moves += 1
        self.pending.append((self.t + self.delay * 0.14, cmd))

    def advance(self, dt=0.14):
        self.t += dt
        # 到期的指令生效（分块 = 固件 ±127 的行为 ✓ 这里只做整数分割 ✓）
        rest = []
        for due, cmd in self.pending:
            if due <= self.t:
                for k in range(2):
                    step = int(cmd[k])
                    while step:
                        chunk = max(-120, min(120, step))
                        self.cursor = (self.cursor[0] + (chunk * self.gain_true[0]
                                                         if k == 0 else 0.0),
                                       self.cursor[1] + (chunk * self.gain_true[1]
                                                         if k == 1 else 0.0))
                        step -= chunk
            else:
                rest.append((due, cmd))
        self.pending = rest
        # 量化：光标位置按整数像素（游戏里就是整数 ✓）
        self.cursor = (round(self.cursor[0]), round(self.cursor[1]))
        self.cursor_delay = self.cursor     # 简化：延迟一拍用"上一帧值"模拟 ✓
        return self.cursor


def t_converge():
    """① 收敛：贴住移动目标（残差 ≤ 死区 + 速度×一拍 ✓）。"""
    sim = Sim()
    c = LieMouseController(gain=(1.0, 1.0), deadzone=3.0, max_step=200.0)
    errs = []
    for i in range(60):
        tgt = sim.target()
        cmd = c.step(tgt, ts=i * 0.14, cursor=sim.cursor_delay)
        sim.apply(cmd)
        sim.advance()
        if i >= 10:                          # 给几步起飞
            errs.append(abs(c.cursor[0] - sim.target()[0]))
    worst = max(errs) if errs else 999.0
    check(worst <= 55.0 * 0.14 + 6.0,
          "稳态误差 %.0f px 超「目标速度×一拍 + 死区」= %.0f px（贴不住 ✗）"
          % (worst, 55.0 * 0.14 + 6.0))


def t_deadzone():
    """② 目标静止 ⇒ 到了之后**一步都不再发**（防抖 ✓）。"""
    sim = Sim(vel=(0.0, 0.0), tgt0=(300.0, 300.0), start=(300.0, 300.0))
    c = LieMouseController(gain=(1.0, 1.0), deadzone=3.0)
    moves_after = None
    for i in range(30):
        tgt = sim.target()
        cmd = c.step(tgt, ts=i * 0.14, cursor=sim.cursor_delay)
        sim.apply(cmd)
        sim.advance()
        if moves_after is None and abs(sim.cursor[0] - tgt[0]) < 2.0:
            moves_after = sim.moves
    check(moves_after is not None,
          "光标压根没贴到静止目标（用例本身没造出收敛局面 ✗）")
    check(sim.moves - (moves_after or 0) <= 1,
          "贴住之后还在发指令（死区失效 ⇒ 抖 ✗）：%d 步"
          % (sim.moves - (moves_after or 0)))


def t_clamp():
    """③ 远起点一步不超过限幅。"""
    sim = Sim(start=(0.0, 0.0), tgt0=(700.0, 480.0), vel=(0.0, 0.0))
    c = LieMouseController(gain=(1.0, 1.0), max_step=200.0)
    cmd = c.step(sim.target(), ts=0.0, cursor=(0.0, 0.0))
    check(max(abs(cmd[0]), abs(cmd[1])) <= 200,
          "单步 %s 超限幅 200（会一帧飞很远 ✗）" % (cmd,))


def t_lost_hold():
    """④ 丢失保护：一步不发 + 持续丢记事件 ✓。"""
    sim = Sim()
    c = LieMouseController(gain=(1.0, 1.0), lost_hold=8)
    for i in range(4):                       # 正常跟几步
        tgt = sim.target()
        sim.apply(c.step(tgt, ts=i * 0.14, cursor=sim.cursor_delay))
        sim.advance()
    cursor_before = sim.cursor
    n_before = sim.moves
    for i in range(10):                      # 之后目标丢了（pos=None ✓）
        sim.apply(c.step(None, ts=(4 + i) * 0.14, cursor=sim.cursor_delay))
        sim.advance()
    check(sim.cursor == cursor_before and sim.moves == n_before,
          "丢失期间光标动了或发了指令（该**原地停住** ✗）：%s → %s / 步数 %d → %d"
          % (cursor_before, sim.cursor, n_before, sim.moves))
    check(c.lost_event and c.held,
          "持续丢失没记事件/没进保护态（弹窗兜底就无从触发 ✗）")
    check(c.step(sim.target(), ts=2.0, cursor=sim.cursor_delay) != (0, 0),
          "目标回来了却不恢复输出（保护态没退出 ✗）")


def t_gain_error():
    """⑤ 增益标定错 ⇒ 反馈自校正；掐掉反馈 ⇒ 系统性偏掉 ✓。"""
    def run(with_feedback, gain_cfg):
        sim = Sim(gain_true=(1.6, 1.6))       # 真增益 = 配置的 1.6 倍（标定错 60% ✓）
        c = LieMouseController(gain=gain_cfg, deadzone=3.0, max_step=400.0)
        errs = []
        for i in range(80):
            tgt = sim.target()
            cmd = c.step(tgt, ts=i * 0.14,
                         cursor=sim.cursor_delay if with_feedback else None)
            sim.apply(cmd)
            sim.advance()
            if i >= 40:                      # ⚠ 量**模拟世界的真光标**（不是控制器的
                errs.append(abs(sim.cursor[0] - sim.target()[0]))   # 自洽估计 ✗ 踩过）
        return max(errs) if errs else 999.0

    e_fb = run(True, (1.0, 1.0))
    e_open = run(False, (1.0, 1.0))
    check(e_fb <= 20.0,
          "有反馈时增益错 60%% 也收敛不了（P 控制该自校正 ✓）：残差 %.0f px" % e_fb)
    check(e_open >= 3 * max(1.0, e_fb),
          "无反馈（纯航位推算）居然也不偏（用例没造出错增益的后果 ✗）：%.0f px" % e_open)


def main():
    print("测谎鼠标闭环控制律自检：")
    t_converge()
    t_deadzone()
    t_clamp()
    t_lost_hold()
    t_gain_error()
    if _FAILED:
        print("自检：%d 条失败" % len(_FAILED))
        return 1
    print("测谎控制律自检全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
