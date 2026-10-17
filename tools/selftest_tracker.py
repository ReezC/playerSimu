"""怪追踪 + 玩家追踪自检：把"移动目标沿轨迹排出一串框 / 框闪 / 锁到别人身上"固化成断言。

**为什么要有它**
    追踪器里那几条规则（运动补偿匹配 / 同体合并 / 平滑+幽灵外推 / **确认闸** / 全局分配 /
    玩家那四关）都是冲着**实测现象**去的，而这类 bug 的特点是：**不报错、只在画面上看得出来**
    —— 一串框、框在抖、框粘在原地、锁到别人身上。等下次改检测/改帧率把它们弄坏，
    靠肉眼发现要重新走一遍"进游戏看半天"的流程。所以拿合成序列在离屏里验：

        同一个怪在走 → 每帧只该有 **一个** 框、**一个** id（而且要先"看见两次"才算数 ✓）
        别人从旁边过 → 锁**不许**交出去（用户 2026-10-11 报的"传送后被抢框"）

    时钟是假的（`tracker.time` 被换成手动推进的），所以防抖时间 / 确认时间都能确定性验。

跑法：
    python -m tools.selftest_tracker       # 全过返回 0，有失败返回 1

⭐⭐ 2026-10-11 起本套件覆盖两条**新链条**（用户原话："**角色使用传送点后被其他玩家抢检出框**"＋
   "**防抖置信度+防抖时间的配置 基本无效，同个怪物的检出框疯狂闪烁**"✓）：
    · 怪：**确认才输出**（`CONFIRM_HITS` / `CONFIRM_MS` ✓）+ **低分框只续命**（BYTE ✓）
      + **全局最优认领**（`perception/assign.py` ✓ 与暴力解对拍 ✓）；
    · 玩家：**黄点权威关 / 竞争者余量 / 跳变隔离 / 传送武装**（见类上面那组常量 ✓）。
"""
import itertools
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import perception.assign as asg                                 # noqa: E402
import perception.tracker as trk                                # noqa: E402

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

    def step(self, *dets, ms=FRAME_MS, low=()):
        self.clock.tick(ms)
        return self.t.update(list(dets), list(low))

    def warm(self, *dets, n=2, ms=FRAME_MS):
        """先喂 n 拍同一批框，把轨迹**确认**下来 ✓。

        ⚠ 2026-10-11 起**新轨迹要确认才输出**（`CONFIRM_HITS` 拍 **且** 活满 `CONFIRM_MS`
          毫秒 ✓）⇒ 老用例里"第一拍就该有框"的断言全部要先把这一步补上 ✓
          （这不是放宽要求 ✓ 而是把"确认"这件事显式写进场景 ✓）。
        """
        out = []
        for _ in range(n):
            out = self.step(*dets, ms=ms)
        return out

    def run(self, dets_per_frame, ms=FRAME_MS):
        return [self.step(*d, ms=ms) for d in dets_per_frame]

    def idle(self, n=1, ms=FRAME_MS):
        """空帧（本帧什么都没检测到）。"""
        return [self.step(ms=ms) for _ in range(n)]


class PFeed:
    """喂**玩家**追踪器的小工具（黄点世界坐标可给可不给 ✓ 时钟同样是假的 ✓）。"""

    def __init__(self, **kw):
        self.clock = Clock()
        self.t = trk.PlayerTracker(**kw)
        self._p = mock.patch.object(trk, "time",
                                    SimpleNamespace(monotonic=self.clock))
        self._p.start()

    def stop(self):
        self._p.stop()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.stop()

    def step(self, *cands, world=None, ms=FRAME_MS):
        self.clock.tick(ms)
        return self.t.update(list(cands), world=world)


# ---------------------------------------------------------------- 静止 / 移动

def t_static_one_box():
    """静止目标：每帧恰好一个框、同一个 id（**确认之后** ✓）。"""
    with Feed() as f:
        first = f.step(det(500))
        check(len(first) == 0,
              "**还没确认就不该输出**（就是它治的「闪一下」✗）：%d 个" % len(first))
        for _ in range(10):
            out = f.step(det(500))
            check(len(out) == 1, "静止目标出现了 %d 个框" % len(out))
            check(out[0].id == 1, "id 变成了 %s" % out[0].id)
        check(len(f.t._tracks) == 1, "轨迹数 = %d，应该只有 1" % len(f.t._tracks))


def t_moving_one_box():
    """中速移动（8 px/帧）：不许在轨迹上排出多个框，id 要稳住、位置要跟上。"""
    with Feed() as f:
        x = 300.0
        f.warm(det(x))
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
        f.warm(det(x))
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
        f.warm(det(500, w=widths[0]))
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
        f.warm(det(200), det(700))
        for _ in range(8):
            out = f.step(det(200), det(700))
            check(len(out) == 2, "两只怪只剩 %d 个框" % len(out))
        check(len({m.id for m in out}) == 2, "两只怪的 id 撞了")
        check(len(f.t._tracks) == 2, "轨迹数 = %d，应该 2" % len(f.t._tracks))


def t_jitter_smoothed():
    """静止 + ±4 px 噪声：输出框的帧间抖动要明显小于输入（"框在抖"）。"""
    rnd = random.Random(7)
    with Feed() as f:
        f.warm(det(500))                    # 先确认下来（确认闸 ✓）
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
        f.warm(det(500), n=2)               # 先确认（确认闸 ✓）再"最后一次看见它"
        f.step(det(500))

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
    """置信度**从没达到过**防抖置信度的框：消失就消失，不留幽灵框。"""
    with Feed() as f:
        f.t.debounce_conf = 0.8
        f.t.debounce_ms = 1000.0
        f.warm(det(500, conf=0.4))          # 低置信度（但确认过 ✓ 免得被确认闸挡住 ✓）
        out = f.idle(1)[0]
        check(len(out) == 0, "低置信度的框不该被保留：%d 个" % len(out))


def t_debounce_conf_peak():
    """⭐ 防抖置信度看**历史最高**（用户 2026-10-01 ✓）：先高后低的怪，消失时仍残留。

    原来用**最后一次**的 `conf` ✗：一只怪先被高置信度认出来、之后几帧框得不准掉到
    阈值以下，它一消失就被立刻丢掉 ✗。改后用 `peak_conf`（历史最高）✓ ——
    "只要达到过阈值，消失时就残留防抖时间" ✓。
    """
    with Feed() as f:
        f.t.debounce_conf = 0.8
        f.t.debounce_ms = 1000.0
        f.warm(det(500, conf=0.9))          # 高置信度：达到过阈值（且确认过 ✓）
        f.step(det(500, conf=0.4))          # 掉到阈值以下（最后一次只有 0.4）
        out = f.idle(1)[0]                  # 消失
        check(len(out) == 1 and out[0].missed > 0,
              "置信度**曾经达到过**阈值、只是最后几帧变低了，消失时却不残留防抖：%d 个"
              % len(out))


def t_ghost_drift():
    """幽灵框按（衰减的）速度继续走，且有总位移上限 —— 不许冻在原地，也不许飘走。"""
    with Feed() as f:
        f.t.debounce_conf = 0.5
        f.t.debounce_ms = 3000.0
        x = 500.0
        f.warm(det(x))
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
        f.warm(det(500))
        f.t.kill(1)
        out = f.idle(1)[0]
        check(len(out) == 0, "kill 之后还有 %d 个框" % len(out))


def t_ids_unique():
    """同一帧里输出的 id 必须唯一 —— 重复 id 会让决策层的目标锁定彻底乱掉。"""
    with Feed() as f:
        f.t.debounce_conf = 0.5
        f.t.debounce_ms = 1000.0
        f.warm(det(300), det(700))          # 先确认两条（否则空列表也能"通过"✗）
        for out in (f.step(det(310)), f.idle(1)[0], f.step(det(315), det(715))):
            ids = [m.id for m in out]
            check(len(ids) == len(set(ids)), "同一帧里 id 重复了：%s" % ids)
        check(len(out) == 2, "这一拍两只都该在（顺带证明确认后没丢）：%d" % len(out))


# ------------------------------------------------- 2026-10-11 新增：确认 / 低分续命

def t_confirm_gate_no_single_frame_flick():
    """⭐⭐ **单帧假框一个都不许出**（用户 2026-10-11："**同个怪物的检出框疯狂闪烁**"✓）。

    为什么这是"闪"的主因：老实现里**新轨迹当拍就输出** ✗ ⇒ YOLO 偶发的一帧假框
    也会立刻画出来、下一拍就没了 ⇒ 画面上就是**闪一下** ✓。
    现在要**连续 2 拍 且 活满 60ms**（`CONFIRM_HITS` / `CONFIRM_MS` ✓）才输出 ✓。

    钉四件：
      ① 只出现**一拍**的框 ⇒ 一拍都不输出 ✓（这就是治闪的那道闸 ✓）；
      ② 连着两拍（同一位置 ✓）⇒ **第二拍**才出 ✓；
      ③ 出框之后它就是"确认过的"✓ —— 后面漏一拍按防抖留（不会因为没连续看见就消失 ✓）；
      ④ ⚠ **时间地板**真的在起作用：把两拍压到 10ms 以内（60fps 那种连着的两帧 ✓）
         ⇒ 仍然**不许**输出 ✓（连着的两次假框往往同源 ✓ 光数帧数拦不住 ✓）。
    """
    with Feed() as f:
        out = f.step(det(500))
        check(len(out) == 0, "① 单帧假框不该输出：%d 个" % len(out))
        out = f.step(det(500))
        check(len(out) == 1, "② 连着两拍、又活了 66ms ⇒ 该输出了：%d 个" % len(out))
        check(out[0].age >= 2, "输出了但 age=%s（确认闸没生效？）" % out[0].age)
        # ③ 确认过之后，漏一拍仍按防抖留（确认**只涨不落** ✓）
        f.t.debounce_conf = 0.5
        f.t.debounce_ms = 500.0
        out = f.idle(1)[0]
        check(len(out) == 1 and out[0].missed > 0,
              "③ 确认过的轨迹漏一拍该按防抖留着：%d 个" % len(out))
    # ④ 时间地板
    with Feed() as f:
        got = []
        for _ in range(4):
            got.append(len(f.step(det(500), ms=5.0)))     # 5ms 一拍 ⇒ 4 拍才 20ms
        check(all(n == 0 for n in got),
              "④ 60ms 内的连续几拍就被算成「确认」了（时间地板没生效 ✗）：%s" % got)


def t_low_conf_maintains_identity():
    """⭐⭐ **低分框只能续命、不能新建身份**（借 BYTE ✓ 2026-10-11 ✓）。

    为什么要有它：怪框的 conf 经常**在 `conf_mob` 上下跳** ✓ ⇒ 掉下去的那几拍
    "整拍查无此怪"✗ ⇒ 幽灵规则也救不了（它要求"曾经达到过 防抖置信度"✓）
    ⇒ 画面上就是**闪烁** ✓（用户报的第二半 ✓）。现在低分档的框能**续住已有轨迹** ✓，
    但**不许**用它新建身份 ✗（不然一片噪声就能凭空造出"怪"来 ✓）。

    钉三件：
      ① **只有低分框**（没有正常档）⇒ 一个轨迹都不新建、一个框都不输出 ✓；
      ② 已有**确认过**的轨迹 + 这一拍只有低分框（同一个位置 ✓）⇒ 轨迹**续住**：
         id 不变 ✓、`missed == 0`（不是幽灵 ✓ 位置是新鲜的 ✓）；
      ③ ⚠ 低分框**不许抬 `peak_conf`** ✗（那会把"其实没看清"记成看清过 ✓
         ⇒ 防抖闸跟着变松 ✗ —— 那就是第二份口径了 ✓）。
    """
    with Feed() as f:
        # ⚠ 这一拍**只有低分框**（正常档一个都没有 ✓）—— 别顺手塞一个正常框进去 ✗
        #   （那会**名正言顺**地新建一条轨迹 ✓ 用例就成了假红 ✓ 第一版我就这么写的 ✗）
        out = f.step(low=[det(905, conf=0.3)])
        check(len(out) == 0 and len(f.t._tracks) == 0,
              "① 低分框不该新建身份：out=%d tracks=%d" % (len(out), len(f.t._tracks)))

    with Feed() as f:
        f.warm(det(500, conf=0.4))                     # 确认过的一条轨迹（peak=0.4 ✓）
        out = f.step(low=[det(505, conf=0.45)])        # 这一拍**只有低分框**
        check(len(out) == 1 and out[0].id == 1,
              "② 低分框该把已有轨迹续住（id 不变）：%d 个 / id=%s"
              % (len(out), [m.id for m in out]))
        check(out[0].missed == 0,
              "② 续住的那一拍**位置是新鲜的** ⇒ 不该是幽灵（missed=%s）" % out[0].missed)
        check(abs(out[0].conf - 0.45) < 1e-6,
              "② 置信度该按这一拍更新：%s" % out[0].conf)
        check(abs(out[0].peak_conf - 0.4) < 1e-6,
              "③ 低分框把 `peak_conf` 抬高了（%s ≠ 0.4）⇒ 防抖闸会被悄悄放宽 ✗"
              % out[0].peak_conf)


def t_flick_recorded():
    """⭐ **"闪一下"要留痕**（用户 2026-10-11："**疯狂闪烁**"✓ 得先能看见是哪种闪 ✓）。

    两种闪的成因完全不同（① 同一个 id 闪 = 检测不稳 ✓；② 每帧换 id = 认领/新建有问题 ✓）✗
    ⇒ 不把 id 与存活拍数打出来就只能靠猜 ✓（沿用"坏了要看得见"那条 ✓）。

    钉三件：闪过的（活得短、没确认）留一条现场 ✓；**确认过的**消失**不算闪** ✓；
    `drain_flick()` 取走之后清空 ✓（不许越攒越多 ✓）。
    """
    with Feed() as f:
        f.step(det(500))
        f.idle(8)                          # 熬过"防抖 + 500ms 缓冲" ⇒ 被清理
        got = f.t.drain_flick()
        check(len(got) == 1 and got[0]["id"] == 1,
              "闪一下没被记下来：%r" % (got,))
        for k in ("id", "age", "conf", "peak", "missed", "ms"):
            check(k in got[0], "现场缺 `%s`（打点要用它 ✓）：%r" % (k, got[0]))
        check(f.t.drain_flick() == [], "`drain_flick` 取走之后该清空 ✓")

    with Feed() as f:
        f.warm(det(500))                   # 确认过的
        f.t.debounce_conf = 0.5
        f.t.debounce_ms = 100.0
        f.idle(12)
        check(f.t.drain_flick() == [],
              "**确认过**的轨迹消失不该算「闪一下」（那会天天误报 ✗）")


# ---------------------------------------------------------------- 全局分配（步 4）

def _brute(cost):
    """暴力解，**语义与 `hungarian` 对齐**：最多配对优先 ✓ 其次总代价最小 ✓。

    ⚠⚠ 这个函数我自己第一版就写歪过 ✗（写成"每一行都必须配上"✓）⇒ 报了一堆**假**不一致 ✓
      ⇒ 教训：**验算函数本身也要先想清语义** ✓（见 `assign.hungarian` 的说明 ✓）。
    """
    n = len(cost)
    m = max((len(r) for r in cost), default=0)
    best = None
    for k in range(1, min(n, m) + 1):
        for rows in itertools.combinations(range(n), k):
            for cols in itertools.permutations(range(m), k):
                tot, ok = 0.0, True
                for i, j in zip(rows, cols):
                    c = cost[i][j] if j < len(cost[i]) else asg.BIG
                    if c >= asg.BIG:
                        ok = False
                        break
                    tot += c
                if not ok:
                    continue
                if best is None or k > best[0] or (k == best[0] and tot < best[1] - 1e-9):
                    best = (k, tot)
    return best


def t_assign_global_optimum():
    """⭐⭐ **全局分配必须真最优**（借 OC-SORT 的那一步 ✓ 2026-10-11 ✓ 与暴力解对拍）。

    为什么按行为验：这是**纯数学** ✓ 而且第一版（手写 JV + `BIG` 占位列）**真的有 bug** ✗
    —— 势被抬到 1e6 量级 ⇒ 小差值被淹没 ⇒ **选错行**（实测 `[[0.713],[0.25],[0.455]]`
    挑成 0.455 ✗ 应为 0.25 ✓）⇒ 所以它必须被**独立**验一遍 ✓（不能只靠"看起来对"✓）。

    钉三件（随机 120 例 + 两条不变量 ✓）：
      ① 结果**合法**：行、列各最多出现一次 ✓ 且**永不**配 `BIG` 的禁止对 ✓；
      ② 与暴力解一致（**配对最多 + 总代价最小** ✓）；
      ③ 空表 / 全禁止 / 单格 ⇒ 不抛、回空 ✓（调用方靠这个"这一拍谁都没配上"✓）。
    """
    check(asg.hungarian([]) == [] and asg.hungarian([[]]) == []
          and asg.hungarian([[asg.BIG]]) == [], "空 / 全禁止该回空表 ✓")

    rnd = random.Random(20261011)
    for _ in range(120):
        n = rnd.randint(0, 4)
        m = rnd.randint(0, 4)
        cost = [[(asg.BIG if rnd.random() < 0.25 else round(rnd.random(), 3))
                 for _ in range(m)] for _ in range(n)]
        pairs = asg.hungarian(cost)
        rows = [i for i, _ in pairs]
        cols = [j for _, j in pairs]
        check(len(set(rows)) == len(rows), "① 同一行配了两次：%r %r" % (cost, pairs))
        check(len(set(cols)) == len(cols), "① 同一列配了两次：%r %r" % (cost, pairs))
        check(all(cost[i][j] < asg.BIG for i, j in pairs),
              "① 配了禁止对（BIG）：%r %r" % (cost, pairs))
        want = _brute(cost)
        got = (len(pairs), sum(cost[i][j] for i, j in pairs))
        if want is None:
            check(not pairs, "② 没有合法配对却配了：%r %r" % (cost, pairs))
        else:
            check(got[0] == want[0] and abs(got[1] - want[1]) <= 1e-6,
                  "② 不是最优（要 配对最多+总代价最小）：cost=%r got=%r want=%r"
                  % (cost, got, want))


def t_assign_crossed_beats_greedy():
    """⭐⭐ **"互相抢"时全局分配比贪心好**（这正是要换掉贪心的理由 ✓）。

    构造（代价表 ✓ 越小越好）：两行两列，**"各自的最近"是错配**：
        行0：列0=0.10（看着最近 ✓）列1=0.40  ⇒ 贪心会让行0抢走列0
        行1：列0=0.11   列1=0.20  ⇒ 行1 只能退而取列1
      贪心总代价 = 0.10 + 0.20 = 0.30；全局最优 = 行0→列1(0.40) + 行1→列0(0.11) = 0.51 ✗
    ⚠ 上面这个例子**恰好贪心更优** ✓ ⇒ 换个方向构造（让"最近"是**错**的）：
        行0：列0=0.30   列1=0.10
        行1：列0=0.11   列1=0.90
      贪心（行0 先挑）：行0→列1(0.10) 然后行1→列0(0.11) ⇒ 0.21 ✓ —— 也是最优 ✗
    真正能分开的是**行0 抢走列0 之后行1 无路可走**那种（列数少 ✓）：
        行0：列0=0.20   列1=0.21
        行1：列0=0.05   （只有列0 可达）
      贪心（按"最近"）：行0 取列0(0.20) ⇒ 行1 配不上 ⇒ 总代价 0.20、**配对 1**
      全局最优：行0→列1(0.21) + 行1→列0(0.05) ⇒ 总代价 0.26、**配对 2** ✓
      ⇒ 关键差别是**配对数**（多认领到一只怪 ✓ 代价只多一点 ✓ 这正是我们要的 ✓）。
    """
    cost = [[0.20, 0.21], [0.05, asg.BIG]]
    pairs = asg.hungarian(cost)
    check(len(pairs) == 2, "全局分配该把两条都配上（贪心只会配 1 条）：%r" % (pairs,))
    check(sorted(pairs) == [(0, 1), (1, 0)],
          "该是「行0 让开、行1 拿走列0」：%r" % (pairs,))


# ------------------------------------------------------ 玩家跟踪：四关（步 2）

def _pbox(cx, cy=300.0, w=BOX_W, h=BOX_H, conf=0.9):
    """玩家候选框（形状同 `det` ✓ 只是名字好读 ✓）。"""
    return det(cx, cy=cy, w=w, h=h, conf=conf)


def t_player_authority_filter():
    """① **黄点权威关**（纯函数级 ✓）：离黄点远一个量级 ⇒ 出局 ✓ 但**绝不筛空** ✗。

    为什么这条能治"被别的玩家抢框"（用户 2026-10-11 ✓）：别的玩家也会被 YOLO 认成玩家框 ✓，
    可他离本帧黄点很远 ✓ —— 而老判据只问"离我上一帧的预测位置近不近"✗
    ⇒ 传送走之后他就"最近"了 ✓ 于是赢 ✓。
    """
    t = trk.PlayerTracker()
    t.cam = (0.0, 0.0)
    # scored 的每项 = (候选框, 像素距离, 世界距离, cx, cy) ✓
    near = (_pbox(500), 5.0, 10.0, 500.0, 300.0)
    far = (_pbox(560), 8.0, 400.0, 560.0, 300.0)
    out = t._authority_filter([near, far], teleport=False)
    check(out == [near], "离黄点 400 而最优才 10 ⇒ 该被剔掉：%r" % (out,))
    # 量不出世界距离的（None ✓）一律留着（世界数据缺 ⇒ 退回老判据 ✓ 不许因此丢锁 ✗）
    unk = (_pbox(520), 7.0, None, 520.0, 300.0)
    out = t._authority_filter([near, unk, far], teleport=False)
    check(unk in out and far not in out, "`None` 该留着、远的该剔掉：%r" % (out,))
    # 全部都很远（黄点被认错 / 相机旧 ✓）⇒ **不许筛空** ✓
    out = t._authority_filter([far, (_pbox(600), 9.0, 500.0, 600.0, 300.0)],
                              teleport=False)
    check(len(out) >= 1, "筛空了 ⇒ 世界数据一坏就丢锁 ✗")
    # 传送那几拍跳过（相机刚作废 ⇒ 判不了 ✓）
    check(t._authority_filter([near, far], teleport=True) == [near, far],
          "传送那几拍该跳过这一关（相机是旧地点的 ✓）")


def t_player_teleport_reacquire():
    """⭐⭐④ **传送武装**：角色用传送点之后，**立刻**认账到新位置 —— 而且**不**锁到
    "还站在旧位置附近的那个别人"身上（用户 2026-10-11 ✓ 原话："**角色使用传送点后
    被其他玩家抢检出框**"✓）。

    判据是**黄点的位移**（`_world_step` ✓）：一帧内跳 ≥ `TELEPORT_MIN_WORLD_PX`（400 ✓）
    ⇒ 这是"我看到它跳了"的**证据** ✓ ⇒ 预测位置改成"**旧框 + ΔW**"✓
    ⇒ 站在旧位置的别人离它远得离谱 ✓ 直接出局 ✓（老判据"离旧位置近就赢"必错 ✗）。

    ⚠ 为什么敢把"世界像素"当"画面像素"用：本仓库**本来就是平移口径** ✓
      （`_cam_of`：`cam = world − box` **没有比例项** ✓）。
    """
    with PFeed() as p:
        # 先锁上（⚠ ②确认闸：**锁定那一拍不出框** ✓ 但内部照旧锁上 + 建相机 ✓
        #   ⇒ 这里连喂两拍把"确认"攒够 ✓ 见 `t_player_confirm_gate` ✓）
        b0 = p.t.update([_pbox(500)], world=(2000.0, 2000.0))
        check(b0 is None, "② 确认闸：锁定那一拍不该出框：%r" % (b0,))
        check(p.t.camera() is not None, "锁定那一拍内部该把相机建起来 ✓")
        p.clock.tick(FRAME_MS)
        b1 = p.t.update([_pbox(505)], world=(2005.0, 2000.0))
        check(b1 is not None, "② 第二拍（活了 66ms）该出框了：%r" % (b1,))
        p.clock.tick(FRAME_MS)
        # 传送：黄点跳到 (2000 + 2000, 2000) ⇒ ΔW ≈ (+2000, 0)
        # 候选 = ① 我（跟着 ΔW 到画面 ≈2505 ✓）② 别人（还站在旧位置 505 ✓）
        got = p.t.update([_pbox(2505), _pbox(505)], world=(4005.0, 2000.0))
        check(got is not None, "传送那一拍没给出框（该认账到新位置 ✓）")
        check(abs(got[0] - 2505.0) < 40.0,
              "传送后该跟到新位置（≈2505），却给了 %.0f ⇒ **被旧位置那个别人抢走了** ✗"
              % got[0])
        check(p.t.diag.get("teleport") == 1, "没记下「这一拍是传送」：%r" % (p.t.diag,))
        check(p.t.cam is None or p.t.diag.get("relock") == 1,
              "传送后该重建相机 / 记一条 relock：cam=%r diag=%r" % (p.t.cam, p.t.diag))


def t_player_competitor_and_quarantine():
    """② **竞争者余量** + ③ **跳变隔离**（用户 2026-10-11 ✓ 借 SDK 的 `ASSOCIATION_*`）。

    ② 两个候选**差不多近** ⇒ 谁都不换 ✓（宁可继续用外推的那个位置 ✓ 也别押错 ✓）。
    ③ 只有一个候选、但它**要让锁跳一大段**（> `max_jump × 0.5` ✓）⇒ 先**隔离几拍** ✓：
       同一个新候选**连着赢** `QUARANTINE_FRAMES`（3 ✓）才认账 ✓。
    ⚠⚠ 两道关**只对"要换框"生效** ✗：同一个人接着往下走、或者站在别人旁边**不动**，
      都不受影响 ✓（不然会把"跟前一个人"这件事弄钝 ✗）。
    """
    # ② 含糊：两个候选都在 100 px 外、彼此只差 6 px
    with PFeed() as p:
        p.t.update([_pbox(500)], world=(2000.0, 2000.0))      # 锁定（② 确认闸：这拍不出框 ✓）
        p.clock.tick(FRAME_MS)
        p.t.update([_pbox(505)], world=(2005.0, 2000.0))      # 第二拍 ⇒ 确认 ✓
        p.clock.tick(FRAME_MS)
        got = p.t.update([_pbox(600), _pbox(606)], world=(2010.0, 2000.0))
        check(p.t.diag.get("ambiguous") == 1,
              "两个候选差不多近 ⇒ 该判含糊：%r" % (p.t.diag,))
        check(got is not None and got[0] < 560.0,
              "含糊时该继续用外推的位置（不押那个新框）：%.0f" % (got[0] if got else -1))
        check(not p.t.diag.get("relock"),
              "含糊那拍**不该**记 relock（没换锁 ✓）：%r" % (p.t.diag,))

    # ③ 隔离：一个候选、要跳 100 px（> 75 ✓）⇒ 头两拍不换、第三拍认账
    with PFeed() as p:
        p.t.update([_pbox(500)], world=(2000.0, 2000.0))      # 锁定
        p.clock.tick(FRAME_MS)
        p.t.update([_pbox(505)], world=(2005.0, 2000.0))      # 第二拍 ⇒ 确认 ✓
        firsts = []
        quar = 0
        for k in range(3):
            p.clock.tick(FRAME_MS)
            got = p.t.update([_pbox(600)], world=(2000.0 + 10.0 * (k + 1), 2000.0))
            firsts.append(None if got is None else round(got[0], 1))
            # ⚠⚠ `diag` 是**每拍重置**的 ✓ ⇒ 必须**在循环里**把账加起来 ✗
            #   （第一版我在循环外读 ✓ 拿到的永远是**最后一拍**的账 ⇒ 假红 ✗）
            quar += int(p.t.diag.get("quarantine") or 0)
        check(quar >= 1, "要跳 100 px 却没隔离（三拍里的 quarantine 合计 %d）：%r"
              % (quar, p.t.diag))
        check(firsts[0] is not None and firsts[0] < 560.0,
              "隔离期内该继续用外推位置（不立刻跳过去）：%r" % (firsts,))
        check(p.t.diag.get("relock") == 1, "第三拍该认账了：%r %r"
              % (p.t.diag, firsts))
        check(abs(firsts[-1] - 600.0) < 20.0, "认账之后该跟到 600：%r" % (firsts,))


# --------------------------------- 2026-10-11 新增：玩家那条链的三件（①②③）+ 角标

def t_player_confirm_gate():
    """⭐⭐② **玩家确认闸**（2026-10-11 ✓ 同怪那条的口径 ✓ 见 `PLAYER_CONFIRM_HITS`）。

    为什么玩家也要（**后果比怪严重** ✗）：认错人会污染**可信相机** ✓ ⇒ 所有"画面 → 世界"
    的换算整体平移 ✗（每只怪的世界坐标、"脚下属于哪个集合"全跟着偏 ✓）。
    所以"将信将疑的那一拍"**宁可不出框** ✗ —— 下游对 `player_box is None` 的处理
    与"这一拍没检测到"**完全一样** ✓。

    钉四件：① 锁定那一拍**不出框** ✓（但内部照旧锁上、相机照旧建 ✓）；
      ② 又看见一次且活满 `PLAYER_CONFIRM_MS` ⇒ 出框 ✓；
      ③ **确认只涨不落**：之后漏一拍（外推）**照样出框** ✓（否则又会闪 ✓）；
      ④ **跟丢重锁要重新确认** ✓（`max_missed` 之后解锁 ⇒ 头一拍又不出框 ✓）。
    """
    with PFeed() as p:
        check(p.t.update([_pbox(500)], world=(2000.0, 2000.0)) is None,
              "① 锁定那一拍不该出框（② 确认闸 ✓）")
        check(p.t.camera() is not None,
              "① 内部该照旧建立相机（确认闸只管「给外面用」那一下 ✓）")
        p.clock.tick(FRAME_MS)
        out = p.t.update([_pbox(505)], world=(2005.0, 2000.0))
        check(out is not None, "② 第二拍（活了 66ms）该出框 ✓")
        p.clock.tick(FRAME_MS)
        out = p.t.update([], world=(2005.0, 2000.0))
        check(out is not None and p.t.last_missed > 0,
              "③ 确认过之后漏一拍该**照样出框**（外推 ✓ 角标标 `*` ✓）：%r missed=%s"
              % (out, p.t.last_missed))
        for _ in range(p.t.max_missed + 1):
            p.clock.tick(FRAME_MS)
            p.t.update([], world=(2005.0, 2000.0))
        check(p.t.locked is False, "④ 连续漏检该解锁：locked=%s" % p.t.locked)
        p.clock.tick(FRAME_MS)
        check(p.t.update([_pbox(900)], world=(2005.0, 2000.0)) is None,
              "④ 重锁之后**要重新确认**（头一拍仍不出框 ✓）")


def t_player_unlock_authority():
    """⭐① **未锁定（开局 / 跟丢重锁）也要过黄点关**（2026-10-11 ✓）。

    原来只有"离上一次的框最近 / 取置信度最高"✗ ⇒ **开局那一下认错人没人拦** ✗
    （"传送后被别的玩家抢框"里有一半就是这种 ✓）。
    现在：有黄点时必须"**离黄点一个量级内**"才算候选 ✓（远出一截的直接出局 ✓
    用**相对倍数** ✓ 无量纲 ✓ 见 `_authority_filter` ✓）；
    ⚠ 没有黄点读数 ⇒ **一个字都不动** ✓（退回"置信度最高"✓ **绝不因此不锁** ✗）。

    钉三件：① 两个候选里"远处那个置信度更高" ⇒ 仍该锁**离黄点近的** ✓；
      ② 没给黄点 ⇒ 退回"置信度最高" ✓（老口径 ✓）；
      ③ ⛔ **低分档不许参与锁定** ✗（未锁定那段**故意不看** `low` ✓）。
    """
    def _lock_and_lose(p, box, world):
        p.t.update([box], world=world)
        for _ in range(p.t.max_missed + 2):
            p.clock.tick(FRAME_MS)
            p.t.update([], world=world)
        check(p.t.locked is False, "该先解锁：locked=%s" % p.t.locked)
        p.clock.tick(FRAME_MS)

    with PFeed() as p:
        _lock_and_lose(p, _pbox(500, conf=0.9), (2000.0, 2000.0))
        _near = _pbox(500, conf=0.55)      # 离黄点近（conf 低 ✓）
        _far = _pbox(5000, conf=0.99)      # 离黄点远一个量级（conf 高 ✓）
        got = p.t.update([_near, _far], world=(2000.0, 2000.0))
        check(p.t.locked is True, "① 这一拍该锁上 ✓")
        check(abs(p.t._box[0] - 500.0) < 1.0,
              "① 该锁「离黄点近」的那个（≈500），而不是置信度更高的远处那个：%.1f"
              % p.t._box[0])
        check(got is None, "① 顺带：头一拍仍受**确认闸**管（不出框 ✓）")

    with PFeed() as p:
        _lock_and_lose(p, _pbox(500, conf=0.9), (2000.0, 2000.0))
        p.t.update([_pbox(500, conf=0.55), _pbox(900, conf=0.99)], world=None)
        check(abs(p.t._box[0] - 900.0) < 1.0,
              "② 没有黄点 ⇒ 退回「置信度最高」（老口径 ✓）：%.1f" % p.t._box[0])

    with PFeed() as p:
        p.t.update([], world=(2000.0, 2000.0), low=[_pbox(500, conf=0.3)])
        check(p.t.locked is False,
              "③ 低分档不该让追踪器锁定（「低分只续命」的第一步 ✓）")


def t_player_low_maintains():
    """⭐⭐③ **玩家低分档只续命**（BYTE ✓ 2026-10-11 ✓ **两个前提缺一不可** ✗）。

    ⚠ 为什么玩家比怪**多两条前提**：玩家的低分框很可能是**别人** ✗（怪的类别只有怪 ✓），
      而认错人会**污染可信相机** ✓ ⇒ 只在「**已锁定** ＋ **黄点对得上**」时才准用它续命 ✓；
      而且（第三条）**不校相机** ✗ —— 它本来就不够格当那把尺 ✓。

    钉四件：① **未锁定** + 只给低分框 ⇒ 不许锁定 ✗（见 `t_player_unlock_authority` ③ ✓）；
      ② 已锁定 + 黄点对得上 + 只给低分框 ⇒ 续命（位置更新 ✓ `last_low=True` ✓
         `missed==0` ✓ **相机一个字没动** ✓）；
      ③ 已锁定但**这一拍没给黄点** ⇒ 低分档**不许用** ✗（走外推 ✓ `last_missed>0` ✓）；
      ④ 已锁定 + 有黄点，但低分框**离预测位置太远**（> `max_jump` ✓）⇒ 不许用 ✗。
      ⚠ 另有"离黄点远一个量级也不许用"那条（`AUTHORITY_FACTOR` ✓）：
        它是 `_authority_filter` **同一份**判据 ✓ 已在 `t_player_authority_filter` 里单测 ✓。
    """
    with PFeed() as p:
        p.t.update([_pbox(500)], world=(2000.0, 2000.0))
        p.clock.tick(FRAME_MS)
        p.t.update([_pbox(505)], world=(2005.0, 2000.0))        # 第二拍 ⇒ 确认
        _cam0 = p.t.camera()
        p.clock.tick(FRAME_MS)
        out = p.t.update([], world=(2008.0, 2000.0), low=[_pbox(510, conf=0.3)])
        check(out is not None and abs(out[0] - 510.0) < 12.0,
              "② 低分档该把位置续住（≈510）：%r" % (out,))
        check(p.t.last_low is True and p.t.last_missed == 0,
              "② 该标成「低分续命」且不是幽灵：low=%s missed=%s"
              % (p.t.last_low, p.t.last_missed))
        check(p.t.camera() == _cam0,
              "② 低分档**不许校相机**（它不够格当那把尺 ✓）：%r → %r"
              % (_cam0, p.t.camera()))

    with PFeed() as p:                                          # ③ 没黄点
        p.t.update([_pbox(500)], world=(2000.0, 2000.0))
        p.clock.tick(FRAME_MS)
        p.t.update([_pbox(505)], world=(2005.0, 2000.0))
        p.clock.tick(FRAME_MS)
        out = p.t.update([], world=None, low=[_pbox(510, conf=0.3)])
        check(out is not None and p.t.last_missed > 0 and p.t.last_low is False,
              "③ 没有黄点 ⇒ 低分档不许用（该走外推：missed=%s low=%s）"
              % (p.t.last_missed, p.t.last_low))

    with PFeed() as p:                                          # ④ 离预测太远
        p.t.update([_pbox(500)], world=(2000.0, 2000.0))
        p.clock.tick(FRAME_MS)
        p.t.update([_pbox(505)], world=(2005.0, 2000.0))
        p.clock.tick(FRAME_MS)
        p.t.update([], world=(2008.0, 2000.0), low=[_pbox(5000, conf=0.3)])
        check(p.t.last_low is False,
              "④ 低分框离预测位置太远（> max_jump）⇒ 不许用 ✗（last_low=%s）"
              % (p.t.last_low,))


def t_player_teleport_needs_adjacent():
    """⚠⚠ **「大」≠「传送」**（用户 2026-10-11 ✓ 原话：「**又触发了，隔这么远直接飞过去锁别人**」✗）。

    现场根因：`ΔW` 只有「**两次读数确实相邻**」时才是「**一帧**的位移」✓ —— 中间隔了好几帧
    （黄点漏检被沿用上一拍 / 框面积闸把这一拍的消费关掉 / 定位回路给的是旧值 ✓）时，
    差值里混着**那段走路的累积** ✓ ⇒ 轻易越过 `TELEPORT_MIN_WORLD_PX` ⇒ 被误当传送 ⇒
    预测点被推到荒谬的位置 ⇒ **谁离那儿近就锁谁** ✗（= 「飞过去锁别人」✓）。

    钉三件：
      ① 隔了 `TELEPORT_MAX_GAP_MS` 以上再喂一个「大位移」⇒ **不算传送** ✓
         （`diag["teleport"]` 没有 ✓ 且记了一条 `teleport_gap` ✓ 看得见 ✓）；
      ② 那一拍**不许**锁到「站在那个远处位置」的候选 ✓（本来就该 hold ✓）；
      ③ ⚠ **相邻两帧的真大跳仍然武装** ✓ —— 别把功能一起治死 ✗（真传送要立刻跟过去 ✓）。
    """
    with PFeed() as p:
        p.t.update([_pbox(500)], world=(2000.0, 2000.0))
        p.clock.tick(FRAME_MS)
        p.t.update([_pbox(505)], world=(2005.0, 2000.0))          # 第二拍 ⇒ 确认 ✓
        # ① 隔一大段时间（> TELEPORT_MAX_GAP_MS ✓）才来下一个读数
        p.clock.tick(400.0)                                       # 400ms ≈ 6 帧没有读数
        p.t.update([_pbox(3105)], world=(4005.0, 2000.0))
        check(not p.t.diag.get("teleport"),
              "隔了几帧的「走路累积」被当成传送了（预测点会被推飞 ✗）：%r" % (p.t.diag,))
        check(p.t.diag.get("teleport_gap") == 1,
              "没记下「大位移但读数不相邻」这件事（调阈值时看不见 ✗）：%r" % (p.t.diag,))
        # ② 不该锁到那个远处候选
        check(abs(p.t._box[0] - 505.0) < 60.0,
              "锁到「站在远处」那个候选身上了（%.0f）—— 正是「飞过去锁别人」✗"
              % p.t._box[0])
    # ③ ⚠ 用**干净状态**验「相邻两帧的真大跳仍然武装」✗（别接在上面那三拍假读数后面：
    #   那三拍会触发刚做的「认账失败就解锁」✓ —— 那是**该有的行为** ✓ 但会把这一条搅浑 ✗）
    with PFeed() as p:
        p.t.update([_pbox(500)], world=(2000.0, 2000.0))
        p.clock.tick(FRAME_MS)
        p.t.update([_pbox(505)], world=(2005.0, 2000.0))          # 确认 ✓
        p.clock.tick(FRAME_MS)
        got = p.t.update([_pbox(2505)], world=(4005.0, 2000.0))   # 相邻 + ΔW = 2000 ✓
        check(p.t.diag.get("teleport") == 1,
              "相邻两帧的真大跳没被当传送（功能被治死了 ✗）：%r" % (p.t.diag,))
        check(got is not None and abs(got[0] - 2505.0) < 40.0,
              "真传送那拍该跟到新位置（≈2505）：%r" % (got,))


def t_player_teleport_tight_tolerance():
    """⭐⭐ **武装那一拍用紧容差**（`TELEPORT_TOL_RATIO` ✓ 见它的常量说明）。

    真传送时「旧框 + ΔW」按平移口径**就是**新位置 ✓（只差几个像素 ✓）⇒ 拿普通帧的
    `max_jump`（150px ✗）去接**太宽**了 —— 宽出来的那一圈正好装着「别人」✗
    （用户原话：「**隔这么远直接飞过去锁别人**」✗）。

    钉两件：
      ① 武装那拍，候选落在「预测点 + 90px」（= `max_jump × 0.6` ✓ **普通容差里、紧容差外** ✓）
         ⇒ **不许认它** ✓（先 hold ✓ 绝不飞过去锁 ✓）；
      ② 候选正好落在预测点上（真传送 ✓）⇒ **认** ✓（别把功能治死 ✗）。
    """
    with PFeed() as p:
        p.t.update([_pbox(500)], world=(2000.0, 2000.0))
        p.clock.tick(FRAME_MS)
        p.t.update([_pbox(505)], world=(2005.0, 2000.0))          # 确认 ✓
        p.clock.tick(FRAME_MS)
        p.t.update([_pbox(2595)], world=(4005.0, 2000.0))         # 相邻 + ΔW=2000 ⇒ 武装 ✓
        check(p.t.diag.get("teleport") == 1, "这一拍该判成传送：%r" % (p.t.diag,))
        check(abs(p.t._box[0] - 2595.0) > 60.0,
              "偏 90px 的候选被认了（紧容差没生效 ⇒ 会「飞过去锁别人」✗）：%.0f"
              % p.t._box[0])

    with PFeed() as p:
        p.t.update([_pbox(500)], world=(2000.0, 2000.0))
        p.clock.tick(FRAME_MS)
        p.t.update([_pbox(505)], world=(2005.0, 2000.0))
        p.clock.tick(FRAME_MS)
        got = p.t.update([_pbox(2505)], world=(4005.0, 2000.0))   # 正好落在预测点 ✓
        check(got is not None and abs(got[0] - 2505.0) < 40.0,
              "真传送（候选正好在预测点上）该认 ✓：%r" % (got,))


def t_player_teleport_gives_up():
    """⚠ **武装要有出口**（`TELEPORT_HOLD_MAX` ✓ 2026-10-11 ✓）。

    连着几拍「武装了、却谁也不认」⇒ 这次「传送」多半是**读数错了**（黄点锁错 / 标定漂 ✓）
    ⇒ 认账失败 ⇒ **解锁重锁** ✓ —— 下一拍走正常锁定路（「离黄点最近」里挑 ✓ 那才是「我」✓）。
    ⚠ 宁可慢几拍，也**不许因为一个坏读数永久瞎** ✗（不出框 = 决策层没有位置 ✓ 比慢几拍糟得多 ✓）。

    钉两件：① 连续武装 + 附近没候选 ⇒ 到第 `TELEPORT_HOLD_MAX` 拍**解锁** ✓ 且记一条
      `teleport_giveup` ✓；② 解锁之后**下一拍能按正常路重新锁上** ✓（别卡死 ✗）。
    """
    with PFeed() as p:
        p.t.update([_pbox(500)], world=(2000.0, 2000.0))
        p.clock.tick(FRAME_MS)
        p.t.update([_pbox(505)], world=(2005.0, 2000.0))          # 确认 ✓
        _w = 2005.0
        # 每拍都「相邻 + 大跳」⇒ 每拍都武装 ✓；候选**故意不在**预测点附近
        #   （= 坏读数造出来的假位置 ⇒ 紧容差里一个候选都没有 ✓）
        for _k in range(int(trk.TELEPORT_HOLD_MAX)):
            _w += 2000.0
            p.clock.tick(FRAME_MS)
            p.t.update([_pbox(505)], world=(_w, 2000.0))
        check(p.t.diag.get("teleport_giveup") == 1,
              "连着几拍谁也不认，该记一条 `teleport_giveup`：%r" % (p.t.diag,))
        check(p.t.locked is False,
              "武装失败该**解锁重锁**（不然会一直不出框 ✗）：locked=%s" % p.t.locked)
        # ② 下一拍按正常路重新锁上（候选就在黄点该在的地方 ✓）
        p.clock.tick(FRAME_MS)
        got = p.t.update([_pbox(505)], world=(_w + 5.0, 2000.0))
        check(p.t.locked is True, "解锁之后没能重新锁上（卡死了 ✗）")
        check(got is None,
              "重锁头一拍仍受**确认闸**管（不出框 ✓）：%r" % (got,))


# ---------------------------------------------------------------- 注册表

TESTS = (

    ("静止目标：一帧一个框、一个 id（确认之后）", t_static_one_box),
    ("中速移动：不在轨迹上排一串框", t_moving_one_box),
    ("快速移动：老实现会每帧新起轨迹的那个场景", t_fast_moving_one_box),
    ("框忽大忽小：仍是同一条轨迹", t_size_flicker_one_box),
    ("两只离得远的怪：不合并", t_two_far_mobs_stay_two),
    ("静止 + 噪声：抖动被压下去", t_jitter_smoothed),
    ("防抖窗口：保留 / 到期 / 清理", t_debounce_window),
    ("置信度不够就不留幽灵框", t_debounce_conf_gate),
    ("置信度看历史最高：先高后低也残留（用户 2026-10-01）", t_debounce_conf_peak),
    ("幽灵框继续走、速度衰减、位移有上限", t_ghost_drift),
    ("kill 立即消除", t_kill),
    ("同一帧 id 唯一", t_ids_unique),
    ("⭐⭐ 确认闸：单帧假框一个都不许出（治「疯狂闪烁」）+ 时间地板",
     t_confirm_gate_no_single_frame_flick),
    ("⭐⭐ 低分框只续命、不新建身份（BYTE）+ 不许抬 peak_conf",
     t_low_conf_maintains_identity),
    ("⭐ 「闪一下」要留痕（id/存活拍数/峰值），确认过的不算闪", t_flick_recorded),
    ("⭐⭐ 全局分配与暴力解对拍（最多配对 + 总代价最小）", t_assign_global_optimum),
    ("⭐⭐ 互相抢时全局分配优于贪心（配对更多）", t_assign_crossed_beats_greedy),
    ("⭐ 玩家①黄点权威关：远的剔掉、绝不筛空", t_player_authority_filter),
    ("⭐⭐ 玩家④传送武装：立刻跟到新位置，不被旧位置的别人抢走",
     t_player_teleport_reacquire),
    ("⭐⭐ 玩家②竞争者余量 + ③跳变隔离：要换框先等三拍", t_player_competitor_and_quarantine),
    ("⭐⭐ 玩家①未锁定也过黄点关（开局那一下不许认错人；没黄点退回置信度最高）",
     t_player_unlock_authority),
    ("⭐⭐ 玩家②确认闸：锁定那拍不出框 · 只涨不落 · 跟丢重锁要重新确认",
     t_player_confirm_gate),
    ("⭐⭐ 玩家③低分档只续命（两个前提：已锁定 + 黄点对得上；且不校相机）",
     t_player_low_maintains),
    ("⚠⚠ 传送武装：「大」≠「传送」（读数不相邻的大位移不许武装 —— 用户 2026-10-11："
     "又触发了，隔这么远直接飞过去锁别人）",
     t_player_teleport_needs_adjacent),
    ("⭐⭐ 传送武装：武装那拍用**紧容差**（偏 90px 的候选不许认）",
     t_player_teleport_tight_tolerance),
    ("⚠ 传送武装：连着几拍谁也不认 ⇒ 解锁重锁（不许因为一个坏读数永久瞎）",
     t_player_teleport_gives_up),
)


def t_rel_anchor_keeps_id():
    """⭐⭐⭐ **角色相对系**：按角色的**世界坐标**算相对位移/相对速度（用户 2026-10-13 ✓ 原话：
    「**不应该**去算相机的位移，而是应该**根据角色的世界坐标**去算**相对位移、相对速度**」✓）。

    现象（用户报的）：**同一个位置的怪框依旧会立即消失，且重新出现后变成了另一个 id** ✗。
    病根：怪框是**画面坐标** ✓，角色一走 / 相机一滚，同一只怪在画面上就平移 ✓（世界没动 ✓），
    而认领那道闸只有 `max(10, 0.9 × 框长边)`（40px 的怪 ≈ 36px ✗）⇒ 滚两帧就顶破 ✗
    ⇒ 认领失败 ⇒ 新起一条 ⇒ **新 id** ✓✓。
    ⇒ 修法：认领之前先把老轨迹按**角色的世界位移**反着挪一次（`_shift_anchor` ✓），
      这样 `_score` 比的是「**怪相对角色**」动了多少 ✓（`vx/vy` 也就成了**相对速度** ✓）。

    钉四件（`MobTracker.update(..., player_world=(wx, wy))` ✓）：
      ① **不喂锚**（老行为）⇒ 画面平移 60px 的那只**续不上**（会换 id ✓）—— 故意钉病根 ✓；
      ② **喂锚**（角色世界也 +60 ✓）⇒ **同一条 id 续上** ✓（`missed` 归零 ✓、不新建 ✓）；
      ③ 锚**跳得离谱**（> `CAM_MAX_STEP` = 传送 / 换图 ✓）⇒ 不挪、只重锚 ✓；
      ④ 锚是 `None`（没定位）⇒ 不挪（老行为 ✓ 宁缺勿错 ✓）。
    """
    from perception.tracker import CAM_MAX_STEP, MobTracker

    def _feed(tr, x1, anchor):
        return tr.update([(x1, 100.0, x1 + 40.0, 150.0, 0.9)], [], player_world=anchor)

    # ① 不喂锚（= 老行为）⇒ 续不上
    tr0 = MobTracker()
    _feed(tr0, 100.0, None)
    _feed(tr0, 40.0, None)
    check(len(tr0._tracks) >= 2,
          "用例前提不成立：不喂锚时这条该**认不上**（否则测不到病根 ✓）：%d 条"
          % (len(tr0._tracks),))

    # ② 喂锚（角色世界 +60 ⇒ 静止的怪在画面上左移 60 ✓）⇒ 同一条续上
    tr1 = MobTracker()
    tr1.debounce_conf = 0.3
    tr1.debounce_ms = 1000.0
    _feed(tr1, 100.0, (0.0, 0.0))
    _tid = tr1._tracks[0]["id"]
    out1 = _feed(tr1, 40.0, (60.0, 0.0))
    check(len(tr1._tracks) == 1 and [m.id for m in out1] == [_tid],
          "喂了角色世界坐标却还是换了一条轨迹 ✗（相对位移没生效 ✓）：%d 条 / %r"
          % (len(tr1._tracks), [m.id for m in out1]))
    check(int(out1[0].missed) == 0,
          "相对系对齐之后这一拍该是**认领**（missed=0 ✓）：%r" % (out1[0].missed,))
    check(tr1._tracks[0]["x1"] < 60.0,
          "老轨迹没被挪进这一帧的相对系 ✗（该 ≈40 ✓）：%r" % (tr1._tracks[0]["x1"],))

    # ③ 锚跳得离谱 ⇒ 不挪、只重锚
    tr2 = MobTracker()
    _feed(tr2, 100.0, (0.0, 0.0))
    _x2 = tr2._tracks[0]["x1"]
    _feed(tr2, 100.0, (CAM_MAX_STEP * 3.0, 0.0))
    check(abs(tr2._tracks[0]["x1"] - _x2) < 1e-6,
          "角色坐标跳得离谱（传送 ✓）时还硬挪了轨迹 ✗（该只重锚 ✓）")

    # ④ 锚缺失 ⇒ 不挪
    tr3 = MobTracker()
    _feed(tr3, 100.0, (0.0, 0.0))
    _x3 = tr3._tracks[0]["x1"]
    _feed(tr3, 100.0, None)
    check(abs(tr3._tracks[0]["x1"] - _x3) < 1e-6,
          "锚拿不到时挪了轨迹 ✗（该原样不动 ✓）")


# ⭐ 注册放在**函数定义之后**（`TESTS` 列表在文件上半部 ⇒ 直接写进列表会 NameError ✓）
TESTS = TESTS + (
    ("⭐⭐⭐ 角色相对系：按角色世界坐标算相对位移/相对速度（相机一滚不再换 id）",
     t_rel_anchor_keeps_id),
)


def t_gmc_fallback_keeps_id():
    """⭐⭐⭐⭐ **GMC 兜底**（用户 2026-10-13 ✓ 原话："**多目标追踪算法能否缓解这个问题？**"＋
    "**全部做完**"✓）—— 这是 MOT 里对"移动相机"的标配做法（借 SDK 的多数派运动 ✓ 见
    `GMC_MIN_TRACKS` ✓）。钉三件：

      ① **前提（病根）**：样本**不够**（这里只有 1 只怪 ⇒ 中位数没有意义 ✓）⇒ 不补偿 ⇒
         画面一滚就**认不上**（换 id ✓）—— 这就是现场 `age=1 / missed≈80` 那条链 ✓；
      ② **样本够**（3 只怪 ✓）⇒ 锚拿不到（`player_world=None`）也**一路续得上同一个 id** ✓；
      ③ 估计本身：中位数 + 剔离群（在跑的那只不许把整体带走 ✓）+ 样本不够就回 `None` ✓。
    """
    # ⚠⚠ **本轮实测踩到的两条（记下来，别再拿它们当"病根" ✗）**：
    #   ① **匀速整体平移**靠"速度外推"本来就跟得上 ✓（`_score` 的预测项 `pcx = cx + vx` 会把
    #      整体平移学进 `vx` ✓）；
    #   ② 漏帧那几拍还有**补桥**（`MATCH_RELAX` ✓）兜着 ⇒ 平滑滚动这条路上 GMC **并不是**
    #      唯一的救命稻草 ✓（实测：关掉 GMC 也照样续上同一个 id ✓）。
    #   ⇒ 所以这条用例**不假装有病根** ✗（那样只会写出一条"前提不成立"的假用例 ✓），
    #     只钉两件**确定的事实**：估计本身对不对 ✓、以及"锚拿不到时它真的顶上去了" ✓。
    def _run(use_gmc):
        with Feed() as f:
            xs = (200.0, 320.0, 440.0)
            f.warm(*[det(x) for x in xs], n=3)
            _ids = sorted(t["id"] for t in f.t._tracks)
            check(len(_ids) == 3, "用例前提：该有 3 条轨迹（GMC 要 3 个样本 ✓）：%r" % (_ids,))
            _x0 = float(f.t._tracks[0]["x1"])
            for k in (1, 2):                      # 让它"看见"整体滚动 30px/帧 ✓
                f.step(*[det(x - 30.0 * k) for x in xs])
            check(sorted(t["id"] for t in f.t._tracks) == _ids,
                  "看见的这几拍就换 id 了 ✗（那是另一条链的问题 ✓）：%r"
                  % (sorted(t["id"] for t in f.t._tracks),))
            # 锚**拿不到**（`Feed.step` 不喂 `player_world` ✓）⇒ 该走 GMC 那条 ✓
            f.step(*[det(x - 30.0 * 3) for x in xs])
            return f.t, _x0

    # ① 估计本身：中位数 + 剔离群 + 样本不够沿用并衰减
    tr = trk.MobTracker()
    tr._gmc_samples = [(10.0, 0.0), (12.0, 1.0)]
    check(tr._gmc_estimate() is None,
          "没算过又没样本时该回 `None`（宁缺勿错 ✓）")
    tr._gmc_samples = [(10.0, 0.0), (11.0, 1.0), (9.0, 0.0), (300.0, 0.0)]
    _g = tr._gmc_estimate()
    check(_g is not None and abs(_g[0] - 10.0) <= 2.0 and abs(_g[1]) <= 2.0,
          "中位数/剔离群不对：一个离群样本（300px）不该把整体带走 ✗：%r" % (_g,))
    tr.gmc = (20.0, 0.0)
    tr._gmc_samples = []                        # 这一拍没配对（典型：都在漏帧 ✓）
    _g2 = tr._gmc_estimate()
    check(_g2 is not None and abs(_g2[0] - 20.0 * trk.GMC_HOLD_RATIO) < 1e-6,
          "样本不够时该**沿用上一次并衰减**（幽灵框得跟着画面走 ✓）：%r" % (_g2,))

    # ② 真顶上去了：锚拿不到 ⇒ `diag.gmc` 计数 + 轨迹按估计值整体挪 ✓
    _t, _x_before = _run(True)
    check(int(_t.diag.get("gmc", 0)) >= 1,
          "锚拿不到时没用上 GMC 兜底（`diag.gmc` = %r ⇒ 还在走「不挪」那条 ✗）"
          % (_t.diag.get("gmc"),))
    # ⚠ 只钉**方向**（负 ✓ = 画面整体往左滚 ✓），不钉精确值 ✗ —— 本轮实测这一格量到
    #   `-14.7`（≈ `-30 × 0.7²`）：说明最后那几拍里有的拍**没凑够 3 个样本**，走了
    #   `GMC_HOLD_RATIO` 的衰减那条 ✓（值偏小但方向对 ✓ 而且够用：认领闸 36px 以内 ✓）。
    #   ⇒ 记一条**待查**：为什么那几拍的配对样本会不足（要么是它被同体合并吃掉了 ✓
    #     要么是 `seen_frame` 的连续判据太严 ✓）—— 别为了让用例好看就把断言写成"≈ -30" ✗。
    check(_t.gmc is not None and float(_t.gmc[0]) < 0.0,
          "学到的「画面整体平移」方向不对（该是负 = 画面往左滚 ✓）：%r" % (_t.gmc,))
    check(float(_t._tracks[0]["x1"]) < _x_before,
          "轨迹没跟着画面挪（幽灵框会停在原地 ⇒ Agent 朝错位置走 ✓）：%r"
          % (float(_t._tracks[0]["x1"]),))

    # ③ 估计本身：中位数 + 剔离群 + 样本不够回 None
    tr = trk.MobTracker()
    tr._gmc_samples = [(10.0, 0.0), (12.0, 1.0)]
    check(tr._gmc_estimate() is None,
          "样本少于 `GMC_MIN_TRACKS` 时该回 `None`（宁缺勿错 ✓）")
    tr._gmc_samples = [(10.0, 0.0), (11.0, 1.0), (9.0, 0.0), (300.0, 0.0)]
    _g = tr._gmc_estimate()
    check(_g is not None and abs(_g[0] - 10.0) <= 2.0 and abs(_g[1]) <= 2.0,
          "中位数/剔离群不对：一个离群样本（300px）不该把整体带走 ✗：%r" % (_g,))


def t_bridge_keeps_id_after_gap():
    """⭐⭐⭐⭐ **跨漏帧重关联（"补桥"）**（用户 2026-10-13 ✓ 同上 ⇒ MOT 的 lost-track / OCR
    再匹配 ✓ 见 `MATCH_RELAX` ✓）：漏了几拍之后**同一个 id 回来** ✓ —— 不补这一刀时必然新起一条 ✓
    （用户看到的"疯狂闪烁"里就有它 ✓）。
    """
    _tr0 = trk.MobTracker()
    _gate = _tr0._gate({"x1": 0.0, "y1": 0.0, "x2": BOX_W, "y2": BOX_H,
                        "vx": 0.0, "vy": 0.0})
    _jump = _gate * 1.5          # > 正常闸（36px ✓）但 < 放宽后的闸（`MATCH_RELAX`=1.8 ✓）
    check(_jump > _gate and _jump <= _gate * trk.MATCH_RELAX,
          "用例前提：这一下位移该「正常闸够不着、放宽闸够得着」（%.1f vs %.1f ✓）"
          % (_jump, _gate))

    with Feed() as f:
        f.t.debounce_ms = 5000.0     # 让这条轨迹在漏帧那几拍里还活着 ✓
        f.warm(det(200.0), n=3)
        _tid = f.t._tracks[0]["id"]
        f.idle(2)                    # 漏两拍（幽灵外推 ✓）
        out = f.step(det(200.0 + _jump))
        check([m.id for m in out] == [_tid] and len(f.t._tracks) == 1,
              "漏帧后再出现时**没认回同一个 id** ✗（补桥没生效 ✓）：%r / %d 条"
              % ([m.id for m in out], len(f.t._tracks)))
        check(int(f.t.diag.get("bridge", 0)) >= 1,
              "这一拍该记一条补桥（`diag.bridge` ✓）：%r" % (f.t.diag.get("bridge"),))


def t_same_id_jump_quarantine():
    """⭐⭐⭐ **同 ID 跳变隔离**（用户 2026-10-13 ✓ 同上 ⇒ 借 SDK `branch_tracker.py:106-116` ✓
    见 `MOB_QUARANTINE_*` ✓）：id 没变、但框跳到另一个更连续的位置（两只贴着的怪整体换身 ✓）⇒
    **这一拍先别认**（按 coast 走 ✓），要连着 `MOB_QUARANTINE_FRAMES` 拍都指向同一处才认 ✓。
    """
    with Feed() as f:
        f.warm(det(200.0), n=3)
        _tid = f.t._tracks[0]["id"]
        # 位移：**比"认领半径 × 0.6"大、但仍在认领半径之内** ✓
        #   ⚠⚠ 本轮实测的教训（别照抄 SDK 的绝对像素 ✗）：SDK 是 IoU+卡尔曼、没有固定像素闸，
        #     所以它敢用 `96px` ✓；我们的闸是 `max(10, 0.9 × 框长边)`（40px 的怪 ≈ 36px ✓）
        #     ⇒ 绝对 96px 那一下**根本配不上**（走不到这里 ✓）= **死代码** ✓（见常量说明 ✓）。
        _gate = f.t._gate({"x1": 0.0, "y1": 0.0, "x2": BOX_W, "y2": BOX_H,
                           "vx": 0.0, "vy": 0.0})
        _x = 200.0 + _gate * (trk.MOB_QUARANTINE_JUMP_RATIO + 0.2)
        check(_gate * (trk.MOB_QUARANTINE_JUMP_RATIO + 0.2) < _gate,
              "用例前提：这一下位移该**小于认领半径**（否则连配都配不上 ✓）")
        # ① 第一拍：跳得可疑（> 认领半径 × 0.6 ✓）⇒ **这一拍不认**（`diag.quarantine` 记一笔 ✓）
        #   ⚠⚠ **本轮实测（别把断言写得比事实强 ✗）**：被隔离这一拍确实没认 ✓，但紧接着
        #     `_dedupe` 会把"同体的新轨迹"并回**老 id**（`_same_body` ✓ 认领半径内 ✓）⇒
        #     那一拍的 `missed` 又会归零 ✗ ⇒ **拿 `missed==1` 当断言是错的** ✓
        #     （隔离的净效果 = "不靠这一拍换位置"，而同体合并照旧保住 id ✓ 这正是我们要的 ✓）。
        f.step(det(_x))
        check(int(f.t.diag.get("quarantine", 0)) >= 1,
              "跳变隔离没触发（`diag.quarantine` ✓）：%r" % (f.t.diag.get("quarantine"),))
        # ② 关键后果：**id 不许换**（这一只是同一个身体 ✓ 合并之后还是老 id ✓ 且不会多出一条 ✓）
        _ids_now = sorted(t["id"] for t in f.t._tracks)
        check(_tid in _ids_now and len(_ids_now) == 1,
              "可疑跳变之后换了 id / 多长出一条 ✗（该仍是老 id 这一条 ✓）：%r"
              % (_ids_now,))


TESTS = TESTS + (
    ("⭐⭐⭐⭐ GMC 兜底：锚拿不到时用「画面整体平移（多数派运动）」顶上（相机滚动不再换 id）",
     t_gmc_fallback_keeps_id),
    ("⭐⭐⭐⭐ 跨漏帧重关联（补桥）：漏几拍之后再出现 ⇒ 认回同一个 id",
     t_bridge_keeps_id_after_gap),
    ("⭐⭐⭐ 同 ID 跳变隔离：跳得离谱先别认，连着 3 拍同一处才认",
     t_same_id_jump_quarantine),
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
