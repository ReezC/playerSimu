"""质检台自检：**←/→ 切帧**（用户 2026-09-27 要求："聚焦质检台时，希望能按←→切换帧"）。

**为什么要单开一套**
    "聚焦质检台就能用方向键切帧"这件事，靠的是 `QShortcut(..., WidgetWithChildrenShortcut)`
    的**焦点范围** —— 而焦点范围是最容易改坏、坏了又**不会立刻被发现**的东西：
    · 换成 `keyPressEvent` ⇒ 只有焦点在面板**本身**时收得到，而这一页是多控件面板
      （画布 / 下拉 / 滑块）⇒ 实际表现是"有时候能切、有时候不能" ✗；
    · 换成 `WindowShortcut` / `ApplicationShortcut` ⇒ 主窗口别的页签（实时 / 路线识别…）
      里按 ←→ 也会切帧 ✗ —— 那就是"抢键"，比没有更糟。
    这两种错法都不会报错、不会崩，只会让人以为"键盘坏了" ⇒ 必须有用例钉住 ✓。

跑法（离屏，不需要项目、不碰任何标注文件）：

    python -m tools.selftest_review      # 全过返回 0，有失败返回 1
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt5.QtCore import Qt                              # noqa: E402
from PyQt5.QtWidgets import (QApplication, QLabel,        # noqa: E402
                             QPushButton, QShortcut)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _panel(items=None):
    """建一个质检台面板，并把它"钉"在假帧列表上（**不碰磁盘**）。

    `step()` 正常会 `_save_current()` / `_load_current()`（读写 frames/labels）——
    这里两个都换成空操作：用例只管"索引有没有跟着走" ✓，一个字节都不写 ✓。
    """
    import unittest.mock as mock

    from gui.review import ReviewPanel

    app = QApplication.instance() or QApplication([])
    p = ReviewPanel()
    p.resize(800, 600)
    p.items = list(items if items is not None else [("f%03d" % i, i % 3, False, -1)
                                                    for i in range(5)])
    p.index = 0
    p._stub_save = mock.patch.object(p, "_save_current", lambda: None)
    p._stub_load = mock.patch.object(p, "_load_current", lambda: None)
    p._stub_save.start()
    p._stub_load.start()
    p._update_pos()
    return app, p


def _shortcuts(p):
    """面板里的 QShortcut → `{"Left": sc, "Right": sc}`（按键名索引 ✓）。"""
    out = {}
    for sc in p.findChildren(QShortcut):
        out[sc.key().toString()] = sc
    return out


def t_left_right_switch_frames():
    """←/→ 真的切帧，而且**到头就停住**（不许越界、不许崩）。

    触发走的是快捷键自己的 `activated`（= 真实链路那一下 ✓），按键映射另有一条
    （`key().toString()` 必须是 Left / Right ✓）—— 两条合起来才说明"按这个键会发生这件事" ✓。
    """
    app, p = _panel()
    try:
        sc = _shortcuts(p)
        check(set(sc) == {"Left", "Right"},
              "质检台里没有 ← / → 两个快捷键（用户 2026-09-27 要求）：%s" % sorted(sc))

        check(p.index == 0, "起始帧不是第 0 帧：%d" % p.index)
        sc["Right"].activated.emit()
        check(p.index == 1, "按 → 没走到下一帧（index=%d，该是 1）" % p.index)
        sc["Right"].activated.emit()
        check(p.index == 2, "按 → 只走了一格就停了（index=%d，该是 2）" % p.index)
        sc["Left"].activated.emit()
        check(p.index == 1, "按 ← 没回到上一帧（index=%d，该是 1）" % p.index)

        # 到头就停：第一帧再往左、最后一帧再往右，都**不许越界**（越界会去读不存在的文件 ✗）
        p.index = 0
        sc["Left"].activated.emit()
        check(p.index == 0, "在第一帧按 ← 越界了（index=%d）" % p.index)
        p.index = len(p.items) - 1
        sc["Right"].activated.emit()
        check(p.index == len(p.items) - 1,
              "在最后一帧按 → 越界了（index=%d，共 %d 帧）" % (p.index, len(p.items)))

        # 空列表（筛出来一帧都没有）⇒ 不许崩、也不许动索引
        #   （`reload()` 在这种情况会把 index 归 0 并清空列表 ✓；这里直接把列表清掉是为了
        #    造出"没有帧"这一态，所以只要求"按了没反应、没异常" ✓）
        p.items = []
        p._update_pos()
        before = p.index
        for k in ("Left", "Right"):
            sc[k].activated.emit()
        check(p.index == before,
              "没有帧的时候按方向键把索引改了（%d → %d）" % (before, p.index))
    finally:
        p._stub_save.stop()
        p._stub_load.stop()
        p.close()


def t_shortcut_scope_is_review_panel_only():
    """焦点范围：**只在本面板（或其子控件）里生效** —— 这是这条需求的关键，也是最容易改坏的。

    钉三件：
      ① `context` 必须是 `WidgetWithChildrenShortcut`（`Window` / `Application` 会在别页抢键 ✗）；
      ② 面板自己 `focusPolicy` 是 `StrongFocus`（否则点空白处焦点留在别的页签 ⇒ "聚焦了却没用" ✗）；
      ③ 画布（子控件）本身可聚焦 ⇒ 点图片后按 ←→ 同样生效 ✓（QGraphicsView 默认就该是）。
    """
    app, p = _panel()
    try:
        for name, sc in _shortcuts(p).items():
            check(sc.context() == Qt.WidgetWithChildrenShortcut,
                  "%s 的快捷键范围不是「本面板及其子控件」（context=%d）—— "
                  "Window/Application 会在别的页签里也抢 ←→ ✗"
                  % (name, sc.context()))
        check(p.focusPolicy() == Qt.StrongFocus,
              "质检台不是 StrongFocus（点空白处不算聚焦 ⇒ 按 ←→ 没反应 ✗）：%s" % p.focusPolicy())
        check(p.canvas.focusPolicy() & Qt.ClickFocus,
              "画布不可聚焦 —— 点一下图片再按 ←→ 就不生效了 ✗")
    finally:
        p._stub_save.stop()
        p._stub_load.stop()
        p.close()


def t_hint_mentions_arrows():
    """可发现性：提示行 + ⏴⏵ 按钮 tooltip 都要写出 ←/→（没写出来等于没这个功能 ✓）。"""
    app, p = _panel()
    try:
        texts = [l.text() for l in p.findChildren(QLabel)]
        check(any("←" in t and "→" in t for t in texts),
              "操作提示行没写「←/→=上一帧/下一帧」：%s" % [t for t in texts if t.strip()])
        tips = [b.toolTip() for b in p.findChildren(QPushButton)]
        check(any("←" in t for t in tips) and any("→" in t for t in tips),
              "⏴⏵ 两个按钮的 tooltip 没写快捷键：%s" % tips)
    finally:
        p._stub_save.stop()
        p._stub_load.stop()
        p.close()


def t_editor_card_entry_button_style():
    """⑤ 卡片的「**打开质检台**」是**入口按钮**角色（用户 2026-09-27 要求："按钮颜色换一下"）。

    本仓库的配色是**按角色**分的（见 `gui/theme.py` 的 `ENTRY_BTN_QSS` 注释）：
    蓝 = 运行 / 主按钮、**靛蓝 = 打开一个干活的地方**、绿 = 通过、橙 = 提醒、红 = 错误。
    「打开质检台」和「寻路编辑器」是同一个角色 ⇒ 用**同一份**色值 ✓。

    钉四件：
      ① 文案还是「打开质检台」（它是质检台的主入口 ✓）；
      ② 样式**就是** `theme.ENTRY_BTN_QSS`（自己再写一套颜色 ⇒ 两份迟早不一致 ✗）；
      ③ 这张卡上**不该有**「运行」按钮（人工环节没有可跑的任务：挂个点了没反应的按钮，
         只会让人以为程序坏了 —— 见 `EditorCard` 的 docstring ✓）；
      ④ 两个入口按钮都从 `theme` 取色，**源码里不许再出现那两个字面色值** ✗。
    """
    from PyQt5.QtWidgets import QApplication

    from gui import theme
    from gui.steps.cards import EditorCard

    root = Path(__file__).resolve().parent.parent
    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")

    c = EditorCard()
    try:
        check(c.btn_view.text() == "打开质检台",
              "⑤ 卡的按钮文案不是「打开质检台」：%r" % c.btn_view.text())
        check(c.btn_view.styleSheet() == theme.ENTRY_BTN_QSS,
              "「打开质检台」不是「入口按钮」那一份样式（颜色要跟「寻路编辑器」一致 ✓）：%r"
              % c.btn_view.styleSheet())
        check(c.btn_run.isHidden(),
              "⑤ 卡上冒出了「运行」按钮（人工环节没有可跑的任务 ✗）")
    finally:
        c.close()

    for f in ("gui/route_panel.py", "gui/steps/cards.py"):
        src = (root / f).read_text(encoding="utf-8")
        check("theme.ENTRY_BTN_QSS" in src,
              "%s 没走 `theme.ENTRY_BTN_QSS`（入口按钮的色值只许有一处 ✗）" % f)
        for hard in ("#e8eaf6", "#c5cae9", "#283593", "#9fa8da"):
            check(hard not in src,
                  "%s 里又硬写了一遍入口按钮的色值（%s）—— 该从 theme 取 ✗" % (f, hard))


TESTS = (
    ("⑤ 卡「打开质检台」= 入口按钮样式（与寻路编辑器同一份色值）",
     t_editor_card_entry_button_style),
    ("质检台 ←/→ 切帧：真的换帧、到头停住、空列表不崩", t_left_right_switch_frames),
    ("快捷键范围只有质检台（不许在别的页签抢 ←→）+ 面板/画布聚焦", t_shortcut_scope_is_review_panel_only),
    ("提示行与 ⏴⏵ tooltip 写了 ←/→", t_hint_mentions_arrows),
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
