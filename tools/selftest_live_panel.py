"""实时预览面板自检：**绘制合并**（主线程慢的时候不许把帧排成队）。

**为什么要它**
    面板每帧的重活 —— `QImage` + `QPixmap` 两次整幅拷贝、再加一次缩放，
    1080p 大约 10~20 ms —— 跑在 **GUI 主线程**上，而收流线程按 30 fps 推信号。
    主线程只要慢一点（失焦被系统降级、窗口正在缩放、机器同时在跑训练），
    信号队列就**越堆越长**：画面越来越滞后、点一下半天才响应，而工作线程侧的
    统计（输入/处理/丢帧）一切正常 —— 体感和数字对不上，最难查的就是这一种。

    合并的做法：**只留最新一帧**，画之前又来新帧就覆盖；面板不可见时根本不画。

跑法（离屏，不连流、不建 YOLO）：

    python -m tools.selftest_live_panel      # 全过返回 0，有失败返回 1
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                          # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _frame(n):
    """造一张"一眼能认出是第几帧"的图（左上角像素的蓝通道 = n）。"""
    img = np.zeros((60, 80, 3), np.uint8)
    img[:, :, 0] = n % 256
    return img


def _panel():
    from PyQt5.QtWidgets import QApplication
    from gui import live_panel as lp

    app = QApplication.instance() or QApplication([])
    p = lp.LivePanel()
    p.resize(240, 180)
    p.show()
    app.processEvents()

    made = {"n": 0}
    orig = lp._bgr_to_pixmap

    def spy(img):
        made["n"] += 1
        return orig(img)

    lp._bgr_to_pixmap = spy                 # _draw_pending 里是模块级调用 → 换得掉
    return app, lp, p, made, orig


def t_coalesce():
    """连推 10 帧：只画**一次**（最新那帧），中间 9 帧被合并掉，绝不排队。"""
    app, lp, p, made, orig = _panel()
    try:
        last = None
        for i in range(10):
            last = _frame(i)
            p._on_frame(last)
        check(p._disp_merged == 9,
              "连推 10 帧应当合并掉 9 帧，实际合并 %d 帧" % p._disp_merged)
        check(made["n"] == 0,
              "事件循环还没跑就画了 %d 次（说明没合并，直接每帧都画）" % made["n"])
        check(p.current_frame() is last,
              "current_frame() 必须是**最新**那帧（探针标定/血条框选靠它）")

        app.processEvents()
        check(made["n"] == 1, "合并之后应当只画 1 次，实际 %d 次" % made["n"])
        check(p._disp_drawn == 1, "画出来的帧数记错了：%d" % p._disp_drawn)
        check(p._draw_ms >= 0.0, "没记下绘制耗时")
    finally:
        lp._bgr_to_pixmap = orig
        p.close()


def t_hidden_skips_draw():
    """面板不可见（切到别的页签/窗口被藏起来）：**一帧都不画**，但帧仍是最新的。

    "没人看就不画"是这次改动的重点之一：画面没人看的时候，省下的主线程时间
    要给决策回路 —— 而且这时**绝不能**反过来让收流变慢（决策不依赖这个画面）。
    """
    app, lp, p, made, orig = _panel()
    try:
        p._on_frame(_frame(1))
        app.processEvents()
        drew = p._disp_drawn
        p.hide()
        app.processEvents()
        for _ in range(3):
            p._on_frame(_frame(11))
            app.processEvents()
        check(p._disp_drawn == drew,
              "不可见时不该画，实际又画了 %d 帧" % (p._disp_drawn - drew))
        check(p._disp_skipped == 3,
              "不可见时应当记下 3 帧「跳过」，实际 %d" % p._disp_skipped)
        cur = p.current_frame()
        check(cur is not None and int(cur[0, 0, 0]) == 11,
              "不可见时 current_frame() 也必须是**最新**帧（不然框选会拿到旧画面）")
    finally:
        lp._bgr_to_pixmap = orig
        p.close()


def t_stats_show_draw():
    """状态行必须露出「绘制 x ms / 合并丢弃 n」—— 数字看得见才谈得上排查。"""
    src = (ROOT / "gui" / "live_panel.py").read_text(encoding="utf-8")
    check("绘制 %4.1f ms" in src and "合并丢弃" in src,
          "实时预览状态行没报绘制耗时/合并丢帧数（失焦卡顿就又没有数字可看了）")
    check("_disp_merged" in src and "_draw_ms" in src,
          "绘制记账字段没接进面板")
    # 源码约定：_on_frame 里不许直接转 QPixmap（那就等于每帧都画，合并失效）
    on_frame = src.split("def _on_frame", 1)[1].split("def _draw_pending", 1)[0]
    check("_bgr_to_pixmap" not in on_frame,
          "_on_frame 里又直接转 QPixmap 了 —— 绘制合并会被绕过（每帧都画）")


TESTS = (
    ("连推 10 帧只画最新那帧（合并，不排队）", t_coalesce),
    ("不可见时一帧都不画，但帧仍是最新的", t_hidden_skips_draw),
    ("状态行露出「绘制 ms / 合并丢弃」（源码约定）", t_stats_show_draw),
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
