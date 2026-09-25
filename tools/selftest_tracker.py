"""怪追踪自检：把"移动目标沿轨迹排出一串框"这几类现象固化成断言。

**为什么要有它**
    追踪器里那三条规则（运动补偿匹配 / 同体合并 / 平滑+幽灵外推）都是冲着
    **实测现象**去的，而这类 bug 的特点是：**不报错、只在画面上看得出来** ——
    一串框、框在抖、框粘在原地。等下次改检测/改帧率把它们弄坏，靠肉眼发现
    要重新走一遍"进游戏看半天"的流程。所以拿合成序列在离屏里验：

        同一个怪在走 → 每帧只该有 **一个** 框、**一个** id

    时钟是假的（`tracker.time` 被换成手动推进的），所以防抖时间也能确定性验。

跑法：
    python -m tools.selftest_tracker       # 全过返回 0，有失败返回 1
"""

import random
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import perception.tracker as trk                             # noqa: E402

FRAME_MS = 1000.0 / 15.0     # 主循环节拍（capture_fps=15），时序都量化到这个粒度
BOX_W, BOX_H = 40.0, 60.0


class Clock:
    """替身时钟：测试自己推进时间，不动真实 monotonic。"""

    def __init__(self, t=1000.0):
        self.t = float(t)

    def __call__(self):
        return self.t

    def tick(self, ms=FRAME_MS):
        self.t += ms / 1000.0
        return self.t


def det(cx, cy=300.0, w=BOX_W, h=BOX_H, conf=0.9):
    """按中心造一个 YOLO 检测框 (x1, y1, x2, y2, conf)。"""
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0, float(conf))


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


class Feed:
    """喂帧的小工具：每次 update 都推进假时钟。"""

    def __init__(self, **kw):
        self.clock = Clock()
        self.t = trk.MobTracker(**kw)
        self._p = mock.patch.object(trk, "time",
                                    SimpleNamespace(monotonic=self.clock))
        self._p.start()

    def stop(self):
        self._p.stop()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.stop()

    def step(self, *dets, ms=FRAME_MS):
        self.clock.tick(ms)
        return self.t.update(list(dets))

    def run(self, dets_per_frame, ms=FRAME_MS):
        return [self.step(*d, ms=ms) for d in dets_per_frame]

    def idle(self, n=1, ms=FRAME_MS):
        """空帧（本帧什么都没检测到）。"""
        return [self.step(ms=ms) for _ in range(n)]


# ---------------------------------------------------------------- 静止 / 移动

def t_static_one_box():
    """静止目标：每帧恰好一个框、同一个 id。"""
    with Feed() as f:
        for _ in range(10):
            out = f.step(det(500))
            check(len(out) == 1, "静止目标出现了 %d 个框" % len(out))
            check(out[0].id == 1, "id 变成了 %s" % out[0].id)
        check(len(f.t._tracks) == 1, "轨迹数 = %d，应该只有 1" % len(f.t._tracks))


def t_moving_one_box():
    """中速移动（8 px/帧）：不许在轨迹上排出多个框，id 要稳住、位置要跟上。"""
    with Feed() as f:
        x = 300.0
        for _ in range(25):
            out = f.step(det(x))
            check(len(out) == 1, "移动目标这一帧出了 %d 个框（一串框的回归）" % len(out))
            check(out[0].id == 1, "移动中 id 变成了 %s" % out[0].id)
            x += 8.0
        check(len(f.t._tracks) == 1, "轨迹数 = %d，应该只有 1" % len(f.t._tracks))
        check(abs(out[0].x - (x - 8.0)) < 5.0,
              "跟上误差 %.1f px 太大" % abs(out[0].x - (x - 8.0)))


def t_fast_moving_one_box():
    """快速移动（30 px/帧 ≈ 0.75 个框宽）：**旧实现就在这里崩**（IoU 掉到 0.3 以下
    就每帧新起轨迹）—— 现在靠「离预测位置够近」那条认领回来，仍然只有一个框。"""
    with Feed() as f:
        x = 300.0
        ids = set()
        for _ in range(30):
            out = f.step(det(x))
            check(len(out) == 1, "快速移动这一帧出了 %d 个框" % len(out))
            ids.add(out[0].id)
            x += 30.0
        check(ids == {1}, "快速移动中 id 变过：%s" % ids)
        check(len(f.t._tracks) == 1, "轨迹数 = %d，应该只有 1" % len(f.t._tracks))
        # 走得快时位置几乎不平滑（自适应系数降到 0），不该拖在后面
        check(abs(out[0].x - (x - 30.0)) < 6.0,
              "快速移动跟不上：差 %.1f px" % abs(out[0].x - (x - 30.0)))


def t_size_flicker_one_box():
    """框忽大忽小（YOLO 只框到半个身子）：仍是同一个 id、一个框，宽度不跟着跳。"""
    with Feed() as f:
        widths = [40, 20, 45, 22, 40, 18, 42, 24, 40, 21]
        got = []
        for w in widths:
            out = f.step(det(500, w=w))
            check(len(out) == 1, "尺寸抖动时出了 %d 个框" % len(out))
            check(out[0].id == 1, "尺寸抖动时 id 变成了 %s" % out[0].id)
            got.append(out[0].w)
        check(len(f.t._tracks) == 1, "轨迹数 = %d，应该只有 1" % len(f.t._tracks))
        check(max(got) - min(got) < 18,
              "输出的框宽度还在跟着抖：%s" % [round(v) for v in got])


def t_two_far_mobs_stay_two():
    """两只离得远的怪：两个 id、互不合并（不能矫枉过正把真怪吞了）。"""
    with Feed() as f:
        for _ in range(8):
            out = f.step(det(200), det(700))
            check(len(out) == 2, "两只怪只剩 %d 个框" % len(out))
        check(len({m.id for m in out}) == 2, "两只怪的 id 撞了")
        check(len(f.t._tracks) == 2, "轨迹数 = %d，应该 2" % len(f.t._tracks))


def t_jitter_smoothed():
    """静止 + ±4 px 噪声：输出框的帧间抖动要明显小于输入（"框在抖"）。"""
    rnd = random.Random(7)
    with Feed() as f:
        f.step(det(500))                    # 先建立轨迹
        inp, outp = [], []
        prev_in = prev_out = None
        for _ in range(40):
            x = 500 + rnd.uniform(-4, 4)
            out = f.step(det(x))[0]
            if prev_in is not None:
                inp.append(abs(x - prev_in))
                outp.append(abs(out.x - prev_out))
            prev_in, prev_out = x, out.x
        mi = sum(inp) / len(inp)
        mo = sum(outp) / len(outp)
        check(mo < mi * 0.6,
              "抖动没压下去：输入平均 %.2f px，输出 %.2f px" % (mi, mo))


# ---------------------------------------------------------------- 防抖 / 幽灵

def t_debounce_window():
    """防抖：框消失后按 debounce_ms 保留（missed>0），到期不再输出，再过 500ms 清理。

    时间自造：一帧 66.7ms，所以"到 300ms 为止"是第 4 帧、第 5 帧必须停 ——
    按**累计毫秒**推，别写死帧数（帧率一改就错）。
    """
    with Feed() as f:
        f.t.debounce_conf = 0.5
        f.t.debounce_ms = 300.0
        f.step(det(500))                    # 最后一次看见它

        elapsed = 0.0
        while elapsed + FRAME_MS <= 300.0:  # 窗口内：一直留着，且是幽灵框
            elapsed += FRAME_MS
            out = f.idle(1)[0]
            check(len(out) == 1, "防抖期内（%.0f ms）不该丢框" % elapsed)
            check(out[0].missed > 0,
                  "防抖期内的框应当带 missed>0（决策层靠它防空放技能）")

        elapsed += FRAME_MS                 # 越过 300ms：必须停
        out = f.idle(1)[0]
        check(len(out) == 0,
              "过了防抖时间（%.0f ms > 300）还在输出" % elapsed)

        # 再过了缓冲（+500ms）：轨迹本身被清掉（不然一直躺着占内存）
        f.idle(10)
        check(len(f.t._tracks) == 0, "该清理的轨迹还在：%s" % f.t._tracks)


def t_debounce_conf_gate():
    """置信度低于防抖置信度的框：消失就消失，不留幽灵框。"""
    with Feed() as f:
        f.t.debounce_conf = 0.8
        f.t.debounce_ms = 1000.0
        f.step(det(500, conf=0.4))          # 低置信度
        out = f.idle(1)[0]
        check(len(out) == 0, "低置信度的框不该被保留：%d 个" % len(out))


def t_ghost_drift():
    """幽灵框按（衰减的）速度继续走，且有总位移上限 —— 不许冻在原地，也不许飘走。"""
    with Feed() as f:
        f.t.debounce_conf = 0.5
        f.t.debounce_ms = 3000.0
        x = 500.0
        for _ in range(12):                 # 稳定地以 10 px/帧向右走
            out = f.step(det(x))
            x += 10.0
        last_x = out[0].x

        steps = []
        prev = last_x
        for _ in range(10):
            out = f.idle(1)[0]
            check(len(out) == 1, "防抖期内应当还有这个框")
            steps.append(out[0].x - prev)
            prev = out[0].x

        check(steps[0] > 2.0, "幽灵框冻在原地了（第一帧只走了 %.2f px）" % steps[0])
        check(all(s >= -1e-6 for s in steps), "幽灵框往回走了：%s"
              % [round(s, 2) for s in steps])
        check(steps[0] >= steps[1] >= steps[2],
              "幽灵框的速度没有衰减：%s" % [round(s, 2) for s in steps[:4]])
        total = prev - last_x
        check(total <= 0.6 * BOX_H + 6.0,
              "幽灵框飘太远：总共走了 %.1f px" % total)


def t_kill():
    """kill(id) 立即消除（幽灵框被攻击时，live_thread 会调它）。"""
    with Feed() as f:
        f.t.debounce_conf = 0.5
        f.t.debounce_ms = 3000.0
        f.step(det(500))
        f.t.kill(1)
        out = f.idle(1)[0]
        check(len(out) == 0, "kill 之后还有 %d 个框" % len(out))


def t_ids_unique():
    """同一帧里输出的 id 必须唯一 —— 重复 id 会让决策层的目标锁定彻底乱掉。"""
    with Feed() as f:
        f.t.debounce_conf = 0.5
        f.t.debounce_ms = 1000.0
        f.step(det(300), det(700))
        f.step(det(305), det(705))
        for out in (f.step(det(310)), f.idle(1)[0], f.step(det(315), det(715))):
            ids = [m.id for m in out]
            check(len(ids) == len(set(ids)), "同一帧里 id 重复了：%s" % ids)


TESTS = (
    ("静止目标：一帧一个框、一个 id", t_static_one_box),
    ("中速移动：不在轨迹上排一串框", t_moving_one_box),
    ("快速移动：老实现会每帧新起轨迹的那个场景", t_fast_moving_one_box),
    ("框忽大忽小：仍是同一条轨迹", t_size_flicker_one_box),
    ("两只离得远的怪：不合并", t_two_far_mobs_stay_two),
    ("静止 + 噪声：抖动被压下去", t_jitter_smoothed),
    ("防抖窗口：保留 / 到期 / 清理", t_debounce_window),
    ("置信度不够就不留幽灵框", t_debounce_conf_gate),
    ("幽灵框继续走、速度衰减、位移有上限", t_ghost_drift),
    ("kill 立即消除", t_kill),
    ("同一帧 id 唯一", t_ids_unique),
)


def main() -> int:
    failed = 0
    for name, fn in TESTS:
        try:
            fn()
        except Exception as e:
            failed += 1
            print("[FAIL] %s\n       %s: %s" % (name, type(e).__name__, e))
        else:
            print("[ OK ] %s" % name)
    print("\n%d/%d 通过" % (len(TESTS) - failed, len(TESTS)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
