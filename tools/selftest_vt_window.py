"""**测谎自动跟踪 · 独立测试窗**的自检（用户 2026-10-17 ✓ 原话：
「**我们不用需要授权的版本，我们只用它的功能做一个测试版的自己的窗口**」✓）。

**它钉什么**（都是"看错了会出事"的地方 ✓）
  ① **默认不许动鼠标** —— 一上车就抢光标是最不能接受的事故 ✓
     （开窗、跑一拍，`SetCursorPos` 一条都不许发 ✓）；
  ② **坐标要对**：SDK 给的目标框是**帧坐标**，而 `roi` 是内容区在帧里的偏移
     ⇒ 画到全帧上必须加偏移 ✓（差这一下，框就整体偏一块 ✓ 看着像"跟踪飘了"✗）；
  ③ **人工接管要让路**：勾了"移动鼠标"之后，**按住左键**必须立刻停手 ✓
     （不然人想接管时会被它扯回去 ✓）；
  ④ **本轮结束要真停**：`SUCCESS` / `ENDED` ⇒ 循环停 ✓（不然继续对着弹窗发坐标 ✗）；
  ⑤ 源码级：**DPI 感知必须声明**（否则物理坐标被系统再缩一次 ⇒ "框对了鼠标却差一截"✗）、
     `SetCursorPos` **只许出现在 `_move_cursor` 一处** ✓（散着写就漏掉左键让路 ✓）。

跑法（离屏、不需要游戏窗口、不会动你的鼠标 ✓）：

    python -m tools.selftest_vt_window          # 全过返回 0
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                              # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


class _Box:
    x0, y0, x1, y1 = 11, 22, 55, 66


class _Res:
    target_bbox = _Box()


class _Obs:
    phase = "LOCKED"
    roi = (5, 6, 100, 80)
    tracking_frame = np.zeros((80, 100, 3), np.uint8)
    result = _Res()
    aim_frame = (30, 40)
    aim_screen = (1030.0, 1040.0)


#: ⚠⚠ **必须留一个强引用**（踩过一次 ✓）：`QApplication.instance() or QApplication([])`
#: 装进局部变量的话，函数一返回就没人引用它了 ⇒ PyQt 把 QApplication 回收
#: ⇒ **它下面的控件全被删掉** ✓ ⇒ 后面一碰复选框就是
#: `RuntimeError: wrapped C/C++ object of type QCheckBox has been deleted` ✓（看着像用例写错 ✗）。
_APP = None


def _win(rect=(100, 200, 640, 480)):
    global _APP
    from PyQt5.QtWidgets import QApplication
    _APP = QApplication.instance() or QApplication([])
    from tools import vt_test_window as M
    w = M.VtTestWindow(rect=rect)
    # 不走真 SDK、不抓真屏幕（这套**不许碰**用户桌面 ✓）
    w.vt.feed = lambda frame, ts, origin: _Obs()
    w.runner._grab = lambda: (np.full((480, 640, 3), 60, np.uint8), rect[:2], "screen")
    return M, w


def t_default_never_moves_mouse():
    """① 开窗 + 跑一拍 ⇒ **一条鼠标指令都不许发**（默认关 ✓）。"""
    M, w = _win()
    check(not w.chk_move.isChecked(), "「移动鼠标」默认被勾上了 ✗（一开窗就抢光标 ✓）")
    check(w.runner.allow_move is False, "runner 默认就允许移动鼠标 ✗")

    calls = []
    M._move_cursor = lambda x, y: (calls.append((x, y)) or True)
    w.runner.running = True
    w.runner.tick()
    check(w.runner.moved == 0, "默认状态下发出去 %d 条鼠标指令 ✗" % w.runner.moved)
    check(not calls, "默认状态下调用了移动函数：%r ✗" % (calls,))


def t_frame_coords_offset():
    """② 目标框是**帧坐标** ⇒ 画到全帧上要**加 roi 偏移**（差这一下就整体偏 ✗）。"""
    M, w = _win()
    snap = w.runner._pack(np.zeros((480, 640, 3), np.uint8), _Obs(), 0.0)
    check(snap["box"] == (16, 28, 60, 72),
          "目标框没加 roi 偏移（得到 %r，应为 (16,28,60,72)）⇒ 框会整体偏一块 ✗"
          % (snap["box"],))
    check(snap["aim"] == (30, 40), "瞄点（帧坐标）不对：%r" % (snap["aim"],))
    check(snap["aim_screen"] == (1030.0, 1040.0),
          "瞄点（屏幕坐标）丢了 ⇒ 没法移动鼠标 ✗：%r" % (snap["aim_screen"],))
    check(snap["roi"] == (5, 6, 100, 80), "ROI 没带出来：%r" % (snap["roi"],))


def t_manual_takeover_yields():
    """③ 勾了"移动鼠标"之后，**左键按住 = 人工接管** ⇒ 必须停手 ✓。"""
    M, w = _win()
    calls = []
    M._move_cursor = lambda x, y: (calls.append((x, y)) or True)
    M._left_down = lambda: True                    # 模拟"人按住左键"
    w.chk_move.setChecked(True)
    w.runner.running = True
    w.runner.tick()
    check(w.runner.moved == 0,
          "人按住左键时仍然发了 %d 条鼠标指令 ✗（会把光标从人手里扯回去 ✓）"
          % w.runner.moved)
    check(w.runner.moved_blocked >= 1,
          "让路了却没记账（`moved_blocked` 还是 %d）⇒ 现场排查时看不出是谁在抢 ✗"
          % w.runner.moved_blocked)
    check(not calls, "让路了却还调用了移动函数 ✗")

    # 松手 ⇒ 恢复发送（否则一按过左键就永久不动了 ✗）
    M._left_down = lambda: False
    w.runner.tick()
    check(w.runner.moved == 1 and calls, "松手后没恢复发送 ⇒ 之后再也不跟了 ✗")


def t_end_of_round_stops():
    """④ `SUCCESS` / `ENDED` ⇒ **本轮真停**（不然继续对着成功弹窗发坐标 ✗）。"""
    M, w = _win()
    obs = _Obs()
    obs.phase = "SUCCESS"
    w.vt.feed = lambda frame, ts, origin: obs
    w.runner.running = True
    msg = w.runner.tick()
    check("SUCCESS" in msg, "结束原因没说清（得到 %r）" % (msg,))
    check(w.runner.running is False, "阶段已经是 SUCCESS 了循环还在跑 ✗")


def t_worker_thread_and_cheap_paint():
    """⑥ 抓帧/推理在**工作线程**、`_paint` 只画（用户 2026-10-17 ✓「它好像特别的慢」）。

    现场症状是"整体慢"，实际是两件事叠在一起：
      · SDK 单帧 37ms（p90 172ms）**串在界面线程**上 ⇒ 这期间窗口收不到事件 ⇒ 拖不动 ✗；
      · 每帧对 1080p 做 `QPixmap.scaled(Smooth)` ⇒ 再花 30~60ms ✗。
    钉四条：① 有 `TickThread` 且 `_toggle` 真启动了它；② `_paint` 里**不许**再 `runner.tick()`；
           ③ `_paint` 里**不许**再有 `QPixmap.scaled`；④ `_compose` 要真把图缩到控件大小。
    """
    M, w = _win()
    check(hasattr(M, "TickThread"), "没有 `TickThread` ⇒ 抓帧+推理还串在界面线程上（界面会卡 ✗）")
    src = (ROOT / "tools" / "vt_test_window.py").read_text(encoding="utf-8")
    _i = src.find("def _paint(")
    _j = src.find("def _compose(")
    check(0 <= _i < _j, "找不到 `_paint` / `_compose`（结构变了，这条钉要跟着改 ✓）")
    _seg = src[_i:_j]
    # ⚠ 只看**代码行**：注释里写着"不再 QPixmap.scaled"也会命中 ✗（第一版就这么假红过 ✓）
    _code = "\n".join(ln for ln in _seg.splitlines()
                      if not ln.strip().startswith("#"))
    check("runner.tick(" not in _code,
          "`_paint` 里还在调 `runner.tick()` ⇒ 界面线程又被推理堵住（拖动/按钮会卡 ✗）")
    check("QPixmap.scaled" not in _code,
          "`_paint` 里还在对整幅图 `QPixmap.scaled` ⇒ 每帧多花 30~60ms ✗（该在 `_compose` 里先缩 ✓）")
    check("TickThread(self.runner" in src, "`_toggle` 没启动 `TickThread` ⇒ 还是在界面线程里跑 ✗")

    # ④ 小视图 ⇒ `_compose` 出来的图必须真的变小（不然等于没优化 ✗）
    w.view.setFixedSize(320, 240)
    snap = w.runner._pack(np.zeros((480, 640, 3), np.uint8), _Obs(), 0.0)
    rgb = w._compose(snap)
    check(rgb.shape[1] <= 320 and rgb.shape[0] <= 240,
          "`_compose` 没把图缩到控件大小（得到 %r）⇒ 每帧还在传 1080p ✗" % (rgb.shape,))
    check(rgb.shape[2] == 3, "`_compose` 出来的不是 3 通道 RGB：%r" % (rgb.shape,))

    # 缩了之后叠加坐标也要跟着缩（框不能还在原坐标上 ✗）
    _hit = int((rgb > 0).sum())
    check(_hit > 0, "缩完之后连框都没画上（`_draw` 没吃缩放系数 k ✗）")


def t_source_pins():
    """⑤ 源码级：DPI 感知要声明 · `SetCursorPos` 只许出现在 `_move_cursor` 一处 ✓。"""
    src = (ROOT / "tools" / "vt_test_window.py").read_text(encoding="utf-8")
    check("SetProcessDpiAwareness" in src,
          "没声明 DPI 感知 ⇒ 物理坐标会被系统再缩一次（\"框对了、鼠标却差一截\"✗）")
    check("_set_dpi_aware()" in src, "写了 DPI 感知却没在主入口调用 ✗")
    # ⚠ 数的是**调用**（`SetCursorPos(` ✓）：注释里提到这个名字不算 ✗（否则钉的是注释 ✗）
    check(src.count("SetCursorPos(") == 1,
          "`SetCursorPos(...)` 出现 %d 处 ⇒ 有绕开 `_move_cursor` 直接动鼠标的地方 ✗"
          "（那就绕开了\"左键让路\"✓）" % src.count("SetCursorPos("))
    check("_left_down()" in src, "移动前没查左键 ⇒ 人会跟它抢光标 ✗")


TESTS = (
    ("① 开窗默认**不动鼠标**：勾选框不勾 · 跑一拍一条指令都不发", t_default_never_moves_mouse),
    ("② 目标框/瞄点坐标：帧坐标 + roi 偏移 ⇒ 画到全帧上不许偏", t_frame_coords_offset),
    ("③ 人工接管：按住左键立刻让路 · 并记账 · 松手恢复", t_manual_takeover_yields),
    ("④ 本轮结束（SUCCESS/ENDED）必须真停", t_end_of_round_stops),
    ("⑤ 源码钉：DPI 感知 · SetCursorPos 只在 `_move_cursor` 一处", t_source_pins),
    ("⑥ 抓帧/推理放工作线程 · 界面只画缩略图（用户 2026-10-17「它好像特别的慢」）",
     t_worker_thread_and_cheap_paint),
)


def main():
    failed = 0
    print("=" * 76)
    print("测谎自动跟踪 · 测试窗自检")
    print("=" * 76)
    for i, (title, fn) in enumerate(TESTS, 1):
        print("\n[%d/%d] %s" % (i, len(TESTS), title))
        try:
            fn()
        except AssertionError as e:
            failed += 1
            print("   [FAIL] %s" % e)
        except Exception as e:                                 # noqa: BLE001
            failed += 1
            import traceback
            print("   [FAIL] %s: %s" % (type(e).__name__, e))
            traceback.print_exc()
        else:
            print("   [ OK ]")
    print("\n" + "=" * 76)
    if failed:
        print("%d/%d 失败" % (failed, len(TESTS)))
        return 1
    print("%d/%d 通过" % (len(TESTS), len(TESTS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
