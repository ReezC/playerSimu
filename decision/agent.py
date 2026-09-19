"""决策状态机：在「一个平台」上找怪打。

输入 WorldState（玩家位置 + 怪物列表），输出按键动作。

状态（最简 HFSM）：
    idle   —— 没怪 / 没开自动，松开所有键
    chase  —— 有怪但超出攻击距离，朝怪水平移动
    attack —— 怪进入攻击距离，停下攻击

「一个平台」的含义：横版 ARPG 先只做水平方向找怪打，不跨平台跳跃。
垂直（怪在头顶/脚下）留到下一版，先让「找到 → 靠近 → 打」这条闭环跑通。

**线程模型**：agent 跑在实时推理线程里，settings 由 GUI 主线程写。
CPython 下对 float/bool/引用 的简单赋值是原子的（GIL 保证不崩），
读到旧值顶多延迟一帧生效，可接受 —— 不为此上锁。
"""

import json
import random
import time
from pathlib import Path

from decision.input import DEFAULT_KEYMAP, KeyState, tap

_SETTINGS_FILE = Path(__file__).resolve().parent.parent / "config" / "decision.json"


class DecisionSettings:
    """决策参数（UI 写，决策线程读），改动后持久化到 config/decision.json。"""

    def __init__(self):
        self.enabled = False
        self.attack_dist = 80.0     # 攻击距离（画面像素），小于它就打
        self.keymap = dict(DEFAULT_KEYMAP)
        self.target_cd = [500, 1000]  # 目标切换 CD [min, max]（毫秒）

    def set_key(self, name, key):
        self.keymap[name] = key

    def save(self):
        """落盘。enabled 不存 —— 自动开关是运行时状态，重启后总是关闭。"""
        try:
            data = {"attack_dist": self.attack_dist, "keymap": self.keymap,
                    "target_cd": self.target_cd}
            _SETTINGS_FILE.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def load(self):
        """从 config/decision.json 读回上次的决策参数。"""
        try:
            data = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
            self.attack_dist = float(data.get("attack_dist", 80.0))
            km = data.get("keymap") or {}
            for k, v in km.items():
                if k in self.keymap and isinstance(v, str):
                    self.keymap[k] = v
            cd = data.get("target_cd")
            if isinstance(cd, (list, tuple)) and len(cd) == 2:
                self.target_cd = [int(cd[0]), int(cd[1])]
        except Exception:
            pass


class CombatAgent:
    def __init__(self, settings):
        self.settings = settings
        self.keys = KeyState()
        self.state = "idle"
        self.facing = 1             # 朝向：+1 右（默认）/ -1 左，由最后按的方向键决定
        self._deadzone = 6.0        # |dx| 小于它就停，避免左右抖
        self._next_attack = 0.0     # 下次攻击的时刻（time.monotonic）
        self._attack_duration = 0.03  # 每次点按攻击键的按住时长（秒）
        self._target_id = None      # 当前锁定的目标 id
        self._target_until = 0.0    # 锁定到期时间（monotonic）

    @staticmethod
    def _dist(m, px, py):
        return ((m.x - px) ** 2 + (m.y - py) ** 2) ** 0.5

    def _nearest(self, mobs, px, py):
        """找最近的怪（欧氏距离），返回 (target, dist)；没怪返回 (None, None)。"""
        target, best = None, None
        for m in mobs:
            d = self._dist(m, px, py)
            if best is None or d < best:
                best = d
                target = m
        return target, best

    def _random_target_cd(self):
        """目标切换 CD（秒）：从 [min, max] 毫秒区间随机取。"""
        lo, hi = self.settings.target_cd
        lo = max(0.0, float(lo)) / 1000.0
        hi = max(lo, float(hi)) / 1000.0
        return random.uniform(lo, hi)

    def tick(self, ws):
        """跑一帧决策。ws: WorldState。返回动作描述 dict（调试/展示）。"""
        s = self.settings

        if not s.enabled or not ws.player.found:
            self.keys.release_all()
            self.state = "idle"
            return {"state": "idle", "reason": "未开启" if not s.enabled else "未定位玩家"}

        px, py = ws.player.x, ws.player.y
        attack_dist = s.attack_dist

        # 目标锁定 + 切换 CD：
        #   · 锁定目标还在、且未到期 → 保持锁定（即使别的怪更近也不切）
        #   · 锁定目标消失 → CD 清零，立即重新找最近
        #   · CD 到期 → 重新找最近（可能切换），随机新 CD
        now = time.monotonic()
        target = None
        if self._target_id is not None:
            for m in ws.mobs:
                if m.id == self._target_id:
                    target = m
                    break

        if target is not None and now < self._target_until:
            best = self._dist(target, px, py)      # CD 内保持锁定
        else:
            target, best = self._nearest(ws.mobs, px, py)
            if target is not None:
                self._target_id = target.id
                self._target_until = now + self._random_target_cd()
            else:
                self._target_id = None
                self._target_until = 0.0

        if target is None:
            self.keys.release_all()
            self.state = "idle"
            return {"state": "idle", "reason": "无怪"}

        dx = target.x - px
        keys = set()

        # 怪在朝向侧（前面）才算「能打到」：dx 和 facing 同号（dx=0 也算前面）。
        # 怪在背后时不攻击，而是转身去追 —— 否则会在背后狂按输出。
        in_front = dx * self.facing >= 0

        if best <= attack_dist and in_front:
            # 前面 + 攻击距离内：攻击（连按，单独处理，不放进 keys 按住）
            self.state = "attack"
        elif in_front and abs(dx) <= self._deadzone:
            # 怪在前面但水平很近、又不在攻击距离内（多半在头顶/脚下平台）：
            # 水平移动没用，停住别左右晃。
            self.state = "chase"
        elif dx > 0:
            # 怪在右边（前方追击，或背后转身）
            self.state = "chase"
            keys.add(s.keymap["right"])
            self.facing = 1
        elif dx < 0:
            # 怪在左边
            self.state = "chase"
            keys.add(s.keymap["left"])
            self.facing = -1
        else:
            # dx == 0（怪在正头顶/脚下），不动
            self.state = "chase"

        self.keys.set(keys)

        # 攻击：连按输出键，两次点按间隔随机 0.07~0.13 秒。
        if self.state == "attack":
            now = time.monotonic()
            if now >= self._next_attack:
                tap(self.settings.keymap["attack"], self._attack_duration)
                self._next_attack = now + random.uniform(0.07, 0.13)
        else:
            self._next_attack = 0.0     # 离开攻击状态，下次进入立刻打

        return {"state": self.state, "target": target.id,
                "dx": round(dx, 1), "dist": round(best, 1),
                "keys": sorted(keys), "facing": self.facing}

    def shutdown(self):
        self.keys.release_all()
        self.state = "idle"


# 全局决策设置（单例）：GUI 主线程写，实时推理线程读。
# 用单例让「决策参数」页签和「实时」页共享同一份配置，不用层层传引用。
settings = DecisionSettings()
settings.load()     # 启动时读回上次保存的决策参数
