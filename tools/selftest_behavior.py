"""玩家行为打点（`core/behavior.py`）自检：**一行一个事件、人直接读得懂**。

为什么单列一条：这份日志的全部价值就是"事后不用工具就能读"（用户 2026-09-27 要求单开
一份行为打点，和 `perf.log` 的性能统计分开）。所以这里钉的是**格式与健壮性**，不是业务
事件本身（业务事件在 `selftest_decision.t_behavior_task_events` 里走真入口钉 ✓）：

  ① 关着的时候**一个字都不写**（开销只剩一次 bool 判断 ✓）；
  ② 开着的时候：时间戳 + 事件名 + `key=value`，**一行一个**、值与值两个空格 ✓；
  ③ 缺的字段**不写**（`None`/空串不许变成 `secs=None` 这种噪音 ✗）、值里的换行被压平
     （**多出来的行会把日志读乱** ✗）；
  ④ 写不进去（路径坏掉）**绝不许抛** —— 打点把主流程搞挂是最不能接受的 ✗；
  ⑤ 文件超上限时砍掉前半截、**留下最近的**（别把最新的也砍了 ✗）。
"""

import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import behavior                                    # noqa: E402

#: 每行开头：日期 时间.毫秒 + **两个空格**（值和字段的分隔符，也是"人能读"的关键 ✓）
_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}  ")


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _tmp():
    return Path(tempfile.mkdtemp(prefix="behavior_")) / "behavior.log"


def t_off_writes_nothing():
    """① 关掉之后：一个字节都不写（常关不能有任何代价）✓。"""
    logf = _tmp()
    old_log, old_en = behavior.LOG, behavior.ENABLED
    try:
        behavior.configure(False, log=logf)
        behavior.event("task_start", why="命令前往：左上平台", steps=3)
        check(not logf.exists(), "关着打点却写了文件：%s" % logf)
    finally:
        behavior.configure(old_en)
        behavior.LOG = old_log


def t_one_line_per_event():
    """② 一行一个事件：时间戳 + 名字 + `key=value`，值与值两个空格 ✓。"""
    logf = _tmp()
    old_log, old_en = behavior.LOG, behavior.ENABLED
    try:
        behavior.configure(True, log=logf)
        behavior.event("task_start", why="命令前往：左上平台", steps=3)
        behavior.event("task_step", step="1/3", secs=15.6, note="到了「右下休息平台」")
        lines = logf.read_text(encoding="utf-8").splitlines()
        check(len(lines) == 2, "两个事件该正好两行，实际 %d 行：%r" % (len(lines), lines))
        for ln in lines:
            check(_TS_RE.match(ln),
                  "行首该是「2026-09-27 10:12:03.412」+ 两个空格：%r" % ln)
        check("task_start" in lines[0] and "why=命令前往：左上平台" in lines[0]
              and "steps=3" in lines[0],
              "事件名 / 字段没按 key=value 写：%r" % lines[0])
        check("  why=" in lines[0] and "  steps=3" in lines[0],
              "字段之间该是两个空格（好读、也好 grep）：%r" % lines[0])
        check("secs=15.6" in lines[1] and "step=1/3" in lines[1],
              "数字字段写丢了：%r" % lines[1])
    finally:
        behavior.configure(old_en)
        behavior.LOG = old_log


def t_missing_and_multiline():
    """③ 缺的字段不写；值里的换行压平（多出来的行会把日志读乱 ✗）。"""
    logf = _tmp()
    old_log, old_en = behavior.LOG, behavior.ENABLED
    try:
        behavior.configure(True, log=logf)
        behavior.event("task_fail", step="2/3", secs=None, left=1,
                       reason="爬到 y=100 就不再升了\n离到达目标还差 200 px")
        lines = logf.read_text(encoding="utf-8").splitlines()
        check(len(lines) == 1, "值里的换行没压平（多出 %d 行）：%r" % (len(lines), lines))
        check("secs=" not in lines[0], "缺的字段被写成了空值/None（噪音 ✗）：%r" % lines[0])
        check("reason=爬到 y=100 就不再升了 离到达目标还差 200 px" in lines[0],
              "多行文本该压成一行（空格连接）：%r" % lines[0])
    finally:
        behavior.configure(old_en)
        behavior.LOG = old_log


def t_never_raises_on_bad_path():
    """④ 写不进去（路径是个目录）⇒ 吞掉，绝不抛（打点不许把主流程搞挂 ✗）。"""
    d = Path(tempfile.mkdtemp(prefix="behavior_bad_"))
    old_log, old_en = behavior.LOG, behavior.ENABLED
    try:
        behavior.configure(True, log=d)          # ← 目标是**目录**，open(...,"a") 必失败
        behavior.event("task_start", why="试一下", steps=1)
    finally:
        behavior.configure(old_en)
        behavior.LOG = old_log


def t_trim_keeps_recent():
    """⑤ 超上限时砍前半截，**最近的必须留下** ✓。"""
    logf = _tmp()
    old_log, old_en, old_max = behavior.LOG, behavior.ENABLED, behavior.MAX_BYTES
    try:
        behavior.MAX_BYTES = 300                 # 缩小上限，几条就能触发（跑得快 ✓）
        behavior.configure(True, log=logf)
        for i in range(40):
            behavior.event("task_step", step="%d/40" % (i + 1), note="x" * 30)
        txt = logf.read_text(encoding="utf-8")
        check(len(txt.encode("utf-8")) <= behavior.MAX_BYTES * 2,
              "裁剪没生效（%d 字节，上限 %d）" % (len(txt.encode("utf-8")), behavior.MAX_BYTES))
        check("step=40/40" in txt, "裁剪把**最新**的那条也砍掉了 ✗")
        check("step=1/40" not in txt, "裁剪没砍掉最早的那些（文件会一直长）")
    finally:
        behavior.MAX_BYTES = old_max
        behavior.configure(old_en)
        behavior.LOG = old_log


def t_sample_min_gap_is_fixed_rate():
    """`sample` 的 **`min_gap` = 「定频」** —— 连续量（距离 / 坐标）**每拍都在变**，
    只靠"值变了就记"等于**不节流** ✗（实测 `chase_dist` / `climb_y` / `climb_dx` 把
    `behavior.log` 刷到 760 KB ✗，用户 2026-09-28 要求改「定频」✓）。

    钉三件：
      ① `min_gap` 之内、**哪怕值一直在变** ⇒ 一条都不记（这才是"定频"✓）；
      ② 过了 `min_gap` ⇒ 记一条（值没变也记 ✓，1 秒一个点看趋势够用 ✓）；
      ③ 不给 `min_gap`（**离散状态**那一路：在不在绳上…）⇒ **值变了立刻记** ✓
         （跳变最要紧 ✓，行为一个字没改 ✓）。
    """
    logf = _tmp()
    old_log, old_en = behavior.LOG, behavior.ENABLED
    try:
        behavior.configure(True, log=logf)
        # ① 连续量：连着 20 拍、值一直在变 ⇒ **只有第一拍**落盘（定频 1s ✓）
        #    ⚠ 第一拍**必须**记：不然曲线就没有起点了 ✓（判据是"两条之间至少隔 min_gap"✓）
        for i in range(20):
            behavior.sample("chase_dist", 100.0 + i, min_gap=1.0)
        _l1 = logf.read_text(encoding="utf-8").splitlines() if logf.exists() else []
        check(len(_l1) == 1 and "v=100.0" in _l1[0],
              "`min_gap` 之内不该每拍都记（**只有第一拍**落盘 ✓）：%r" % _l1)
        # ② 把"上次记录时刻"往前挪 2 秒 ⇒ 再采一次就该记 ✓
        _ts, _val = behavior._last_sample["chase_dist"]
        behavior._last_sample["chase_dist"] = (_ts - 2.0, _val)
        behavior.sample("chase_dist", 123.0, min_gap=1.0)
        txt = logf.read_text(encoding="utf-8")
        check("chase_dist" in txt and "v=123.0" in txt,
              "过了定频间隔该记一条（1 秒一个点 ✓）：\n%s" % txt)
        # ③ 离散状态（不给 min_gap）⇒ 值变了**立刻**记 ✓
        logf2 = _tmp()
        behavior.configure(True, log=logf2)
        behavior.sample("climb_on_ladder", 0)
        behavior.sample("climb_on_ladder", 1)
        txt2 = logf2.read_text(encoding="utf-8")
        check("v=0" in txt2 and "v=1" in txt2,
              "离散状态该「值变了立刻记」（0→1 的跳变最要紧 ✗）：\n%s" % txt2)
    finally:
        behavior.configure(old_en)
        behavior.LOG = old_log


def t_task_quick_merges_fast_tasks():
    """**瞬时任务合并成一条**（用户 2026-09-28 要求："`secs < 1s` 的『下一条、完成一条』
    合成一行（`task_quick`）"—— 实测这类占 `task_done` 的 ~45%、连上 `task_start` 一共
    吃掉一半多的日志量 ✗）。

    钉五件：
      ① `task_open` 之后**不马上落盘**（暂存 ✓）—— 这是"合得起来"的前提 ✓；
      ② 立刻 `task_settle(secs=0.2)` ⇒ 只出一条 **`task_quick`**（带起跑那句 `why` ✓）；
      ③ `task_settle(secs=30)`（跑得久）⇒ **分开两条**：`task_start` + 收工那条 ✓；
      ④ `merge=False`（**失败 / 取消**）⇒ **绝不合并** ✗（那不是"走完了"✓）；
      ⑤ 暂存挂够 `QUICK_S` ⇒ 下一次打点**顺手补写** ✓（长任务不会一直不落盘 ✗）。
    """
    old_log, old_en = behavior.LOG, behavior.ENABLED
    try:
        # ①② 瞬时任务 ⇒ 一条 task_quick
        logf = _tmp()
        behavior.configure(True, log=logf)
        behavior.task_open("task_start", why="命令前往：乙平台", steps=1)
        check(not logf.exists(),
              "`task_open` 该先**暂存**（不落盘 ✗）—— 不然后面合不起来")
        behavior.task_settle("task_done", steps=1, secs=0.2, note="到了")
        lines = logf.read_text(encoding="utf-8").splitlines()
        check(len(lines) == 1 and "task_quick" in lines[0],
              "瞬时任务该**合成一条** `task_quick`（不是 start + done 两条 ✗）：%r" % lines)
        check("why=命令前往：乙平台" in lines[0] and "secs=0.2" in lines[0],
              "合并出来的那条该带上起跑的 `why` + 用时：%r" % lines[0])

        # ③ 长任务 ⇒ 分两条（起跑那条**按原时间戳**补上 ✓）
        logf2 = _tmp()
        behavior.configure(True, log=logf2)
        behavior.task_open("task_start", why="命令前往：顶层", steps=4)
        behavior.task_settle("task_done", steps=4, secs=30.0, note="到了")
        lines2 = logf2.read_text(encoding="utf-8").splitlines()
        check(len(lines2) == 2, "长任务该分两条（start + done）：%r" % lines2)
        check("task_start" in lines2[0] and "why=命令前往：顶层" in lines2[0],
              "起跑那条该照写：%r" % lines2[0])
        check("task_done" in lines2[1] and "secs=30.0" in lines2[1],
              "收工那条该照写：%r" % lines2[1])

        # ④ merge=False（失败 / 取消）⇒ 绝不合并
        logf3 = _tmp()
        behavior.configure(True, log=logf3)
        behavior.task_open("task_start", why="命令前往：乙平台", steps=1)
        behavior.task_settle("task_fail", secs=0.2, merge=False, reason="卡住了")
        txt3 = logf3.read_text(encoding="utf-8")
        check("task_quick" not in txt3 and "task_start" in txt3 and "task_fail" in txt3,
              "失败 / 取消**不是「走完了」** ⇒ 不许合并成 `task_quick` ✗：\n%s" % txt3)

        # ⑤ 暂存挂够 QUICK_S ⇒ 下一次打点顺手补写（长任务不会一直不落盘 ✓）
        logf4 = _tmp()
        behavior.configure(True, log=logf4)
        behavior.task_open("task_start", why="命令前往：顶层", steps=1)
        _p = behavior._pending
        behavior._pending = (_p[0] - behavior.QUICK_S - 1.0,) + _p[1:]
        behavior.event("route_phase", note="随便一条")
        txt4 = logf4.read_text(encoding="utf-8")
        check("task_start" in txt4 and "why=命令前往：顶层" in txt4,
              "暂存挂过 `QUICK_S` 之后该被下一次打点**补写**（长任务不许一直不落盘 ✗）：\n%s"
              % txt4)
        check(txt4.index("task_start") < txt4.index("route_phase"),
              "补写那条该排在**当前这条之前**（它的时间戳是起跑那一刻 ✓）：\n%s" % txt4)
    finally:
        behavior.configure(old_en)
        behavior.LOG = old_log


TESTS = (
    ("关着打点：一个字节都不写", t_off_writes_nothing),
    ("一行一个事件：时间戳 + key=value", t_one_line_per_event),
    ("缺字段不写、多行压平", t_missing_and_multiline),
    ("写不进去也不抛（打点不许搞挂主流程）", t_never_raises_on_bad_path),
    ("超上限砍前半截、留最近的", t_trim_keeps_recent),
    ("`sample` 的 `min_gap` = 定频（连续量不许每拍都记）", t_sample_min_gap_is_fixed_rate),
    ("瞬时任务合并成一条 `task_quick`", t_task_quick_merges_fast_tasks),
)


def main():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    bad = 0
    for name, fn in TESTS:
        try:
            fn()
            print("[ OK ] %s" % name)
        except Exception as e:                   # noqa: BLE001
            bad += 1
            print("[FAIL] %s\n       %s: %s" % (name, type(e).__name__, e))
    print("\n%d/%d 通过" % (len(TESTS) - bad, len(TESTS)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
