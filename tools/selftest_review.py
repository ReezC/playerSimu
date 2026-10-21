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
                             QPushButton, QShortcut)          # noqa: E402

from gui import labelio                                    # noqa: E402


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
        # ⚠ 用**包含**而不是相等：后来加了 Q/W（操作模式 ✓ 用户 2026-10-04 第 3 条 ✓）
        #   —— 写死成 `== {"Left","Right"}` 就会在"加了新快捷键"时假红 ✗（本轮就是这么红的 ✓）。
        check({"Left", "Right"} <= set(sc),
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


def t_dataset_empty_frame_is_negative():
    """⭐⭐ **「标过但空」的帧能当负样本收进来**（用户 2026-09-28 ✓ 原话："frame_00381 画面里
    没有东西，他对训练**有益无害**就加"）。

    钉三件：
      ① **没有标注文件** ⇒ `_merge_label_lines` 给 `None`（= 这帧**没标注过** ✗ 该跳过 ✓）；
      ② ⭐ **文件在、但内容为空** ⇒ 给 `[]`（= **标过、确认没东西** ⇒ 负样本 ✓）；
      ③ ⭐ `_collect_pairs` + `min_boxes=0` ⇒ **收下它** ✓ ——
         ⚠ 原来这两种都返回 `None` ✗ ⇒ 空帧**永远进不来** ⇒ `prepare_dataset` 里那句
         "`0 = 保留空帧（当负样本）`"**从来没生效过**（真 bug ✓ 用户当天就撞上了 ✓）。
    ⚠ `_merge_label_lines` 全仓**只有一个调用点**（`_collect_pairs` ✓）⇒ 改这语义是安全的 ✓。
    """
    import tempfile
    from pathlib import Path

    from perception import prepare_dataset as P

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fr = root / "frames"
        lb = root / "labels"
        fr.mkdir()
        lb.mkdir()
        for _stem in ("f_none", "f_empty", "f_box"):
            (fr / (_stem + ".png")).write_bytes(b"")
        # ① `f_none`：**不建**标注文件（= 没标注过 ✓）
        # ② `f_empty`：**建**一个 0 字节的（= 标过、确认没东西 ✓）
        (lb / "f_empty.txt").write_text("", encoding="utf-8")
        # ③ `f_box`：有一个真框
        (lb / "f_box.txt").write_text("1 0.5 0.5 0.1 0.1\n", encoding="utf-8")

        check(P._merge_label_lines("f_none", [str(lb)], ()) is None,
              "「**没有**标注文件」该给 `None`（= 没标注过 ⇒ 跳过 ✓）：%r"
              % (P._merge_label_lines("f_none", [str(lb)], ()),))
        _e = P._merge_label_lines("f_empty", [str(lb)], ())
        check(_e == [],
              "「**标过但空**」该给 `[]`（负样本 ✓）—— 给 `None` 就会把它当「没标注」丢掉，"
              "于是 `min_boxes=0` 也永远收不到（真 bug，用户 2026-09-28 撞上的 ✗）：%r" % (_e,))

        _p1, _nl1, _em1 = P._collect_pairs(str(fr), [str(lb)], 1, ())
        check(len(_p1) == 1 and _nl1 == 1 and _em1 == 1,
              "`min_boxes=1` 该只收那 1 帧有框的（没标注 1 / 框数不足 1）：%r / %r / %r"
              % (len(_p1), _nl1, _em1))
        _p0, _nl0, _em0 = P._collect_pairs(str(fr), [str(lb)], 0, ())
        check(len(_p0) == 2 and _nl0 == 1 and _em0 == 0,
              "`min_boxes=0` 该**多收那帧空帧**（当负样本 ✓）—— 这正是用户要的「就加」："
              "%r 帧 / 没标注 %r" % (len(_p0), _nl0))


def t_mask_and_outline_box():
    """⭐ **蒙版与描边框**（2026-09-29 用户紧急反馈两件 ✓）：

      ① **设置弹窗能建**（第一版把局部函数 `_on_mask` 误写成 `self._on_mask` ⇒
         打开即 AttributeError ✗ —— "设置打不开"就是这么来的 ✓）；
      ② **框只有描边、不填充**（填充 14% 蓝色蒙在框中间，看"框和目标对不对"费眼 ✗）；
      ③ **框在蒙版上面**（box z=10 > mask z=0 ✓ —— 描边不被灰蒙版压暗 ✓）；
      ④ `set_mask_alpha` 生效且**截到 [0,1]** ✓。
    """
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QPixmap
    from PyQt5.QtWidgets import QApplication

    # ⚠ 本用例排在最前 ⇒ QApplication 可能还没建（QWidget 没有它就是 0xC0000409 ✗）
    _ = QApplication.instance() or QApplication(sys.argv)

    # ②③④ 画布：蒙版层 + 描边框
    # （① "设置弹窗能建"的检查 → `tools/selftest_yolo_wb.py`：本套件离屏环境建
    #   SettingsDialog 会 0xC0000409 ✗ —— 同 selftest_decision 的已知问题 ✓）
    from gui.canvas import ImageCanvas
    from gui.theme import review_mask_alpha, set_review_mask_alpha

    _old = review_mask_alpha()
    c = None
    try:
        set_review_mask_alpha(0.5)
        c = ImageCanvas()
        c.set_mask_alpha(0.5)
        c.load(QPixmap(120, 90), [])
        c.add_box(10, 10, 50, 40, 0, False)
        box = c.boxes[0]
        assert c._mask_item is not None and c._mask_item.opacity() == 0.5, \
            "蒙版层没建 / 浓淡不对 ✗：%r" % (
                c._mask_item.opacity() if c._mask_item else None,)
        assert box.zValue() > c._mask_item.zValue(), \
            "框不在蒙版上面（描边会被灰蒙版压暗 ✗）：%r / %r" % (
                box.zValue(), c._mask_item.zValue())
        assert box.brush().style() == Qt.NoBrush, \
            "框内部还有填充（用户：**只有描边** ✗）：%r" % (box.brush().style(),)
        c.set_mask_alpha(7.0)
        assert c._mask_alpha == 1.0, "蒙版浓度没截到 [0,1] ✗：%r" % (c._mask_alpha,)
    finally:
        set_review_mask_alpha(_old)
        if c is not None:
            c.deleteLater()


# ---------------------------------------------------------------- 掉落物类 + 筛选 + 操作模式

#: 造用例画面用的尺寸（**真 PNG** 才读得出来 ✓）
_IMG_W, _IMG_H = 400, 300


def _send(widget, kind, pt, btn=Qt.LeftButton, mod=Qt.NoModifier):
    """往画布的 **viewport** 发一个鼠标事件。

    ⚠⚠ **别用 `QTest.mouseMove`** ✗：离屏下它**根本不投递**（实测 `_rubber.rect()` 一直是空的
      ⇒ 拖拽在用例里"没发生" ⇒ 会写出"建不了框 / 选不中"这种**假红** ✓）。自己发事件就对了 ✓。
    """
    from PyQt5.QtCore import QEvent, QPointF
    from PyQt5.QtGui import QMouseEvent
    from PyQt5.QtWidgets import QApplication

    QApplication.sendEvent(widget, QMouseEvent(kind, QPointF(pt), btn, btn, mod))


def _drag(canvas, a, b, mod=Qt.NoModifier):
    """在画布上从场景坐标 a 拖到 b（按下 → 移动 → 松手 ✓）。"""
    from PyQt5.QtCore import QEvent

    va = canvas.mapFromScene(*a)
    vb = canvas.mapFromScene(*b)
    _send(canvas.viewport(), QEvent.MouseButtonPress, va, Qt.LeftButton, mod)
    _send(canvas.viewport(), QEvent.MouseMove, vb, Qt.NoButton, mod)
    _send(canvas.viewport(), QEvent.MouseButtonRelease, vb, Qt.LeftButton, mod)


def _mk_project(td, frames):
    """造一个**真项目**：`frames/<stem>.png` + 各自标注 ✓。

    `frames` = `[(stem, [(cls, x, y, w, h, manual), ...] 或 None), ...]`
    （`None` = **不写**标注文件 = 这帧没标注过 ✓）。
    """
    from PyQt5.QtGui import QPixmap
    from gui.project import Project

    proj = Project.create(Path(td), name="用例", map_id="105090600")
    proj.frames.mkdir(parents=True, exist_ok=True)
    for stem, boxes in frames:
        QPixmap(_IMG_W, _IMG_H).save(str(proj.frames / (stem + ".png")))
        if boxes is not None:
            labelio.save_boxes(proj, stem, boxes, _IMG_W, _IMG_H)
    return proj


def t_pet_filter():
    """⭐⭐ 筛选「只看有宠物」（用户 2026-10-05 ✓ 原话："增加质检台筛选策略『只看有宠物』"）。

    宠物是**干扰类**（class 5 ✓ 见 `perception/detect/classes.py`）—— 它最烦的错误是"宠物被误检成怪 /
    怪被误检成宠物" ✓ ⇒ 质检台要能一键把"标过宠物的帧"挑出来**逐帧核对** ✓（不用在几百帧里翻 ✓）。
    这条与「只看有掉落物」（用户 2026-10-04 ✓）**同款** ✓。

    钉四件：
      ① 下拉里真有「只看有宠物」这条（文案就是用户说的那六个字 ✓）、筛选码是 `has_pet` ✓
         （`reload` 靠它分派 ✓ 别用会跟别的模式撞的字符串 ✗）；
      ② 选中后**只剩有宠物框的帧** ✓，且**自动（`labels_auto/`）/ 人工（`labels/`）两侧都算** ✓；
      ③ 切回「全部」能复原（证明②不是把列表弄空了 ✓）；
      ④ **宠物框不算"空帧"** ✓ —— 「只看空帧（疑似漏检）」里不许出现它 ✗
         （否则人会把"已经标好的帧"当成漏检去补 ✓ 那是白费功夫 ✓）。
    """
    import shutil
    import tempfile

    from PyQt5.QtWidgets import QApplication

    from gui.review import ReviewPanel

    td = tempfile.mkdtemp(prefix="revpet_")
    try:
        # ⚠⚠ **QApplication 要建在 `_mk_project` 之前** ✗：它里面就 `QPixmap(...).save()`
        #   （造帧图 ✓）—— 没有 app 时会 `qFatal("Cannot create a QPixmap without QApplication")`
        #   ⇒ 进程 **`0xC0000409`、没有 traceback** ✓（实测踩过一次 ✓ 同 SKILL 260 里那个坑 ✓）。
        #   ⚠ 而且引用要**接住**（`app = ...` ✓ 写成不接住的表达式会被当场析构 ✓）。
        app = QApplication.instance() or QApplication([])
        proj = _mk_project(td, [
            ("f_mob", [(labelio.CLASS_MOB, 10, 10, 20, 20, False)]),        # 只有怪
            ("f_pet_auto", [(labelio.CLASS_PET, 10, 10, 20, 20, False)]),   # 宠物（自动那侧 ✓）
            ("f_pet_manual", [(labelio.CLASS_PET, 12, 12, 18, 18, True)]),  # 宠物（人工那侧 ✓）
            ("f_both", [(labelio.CLASS_MOB, 5, 5, 20, 20, False),
                        (labelio.CLASS_PET, 30, 30, 16, 16, True)]),
            ("f_none", None),
        ])
        p = ReviewPanel()
        p.bind(proj)
        app.processEvents()

        i_f = p.cmb_filter.findData("has_pet")
        check(i_f >= 0, "筛选里没有「只看有宠物」（用户点名要的 ✗）：%r"
              % ([p.cmb_filter.itemText(i) for i in range(p.cmb_filter.count())],))
        check(p.cmb_filter.itemText(i_f) == "只看有宠物",
              "那一条的文案不是「只看有宠物」：%r" % p.cmb_filter.itemText(i_f))
        p.cmb_filter.setCurrentIndex(i_f)
        app.processEvents()
        got = sorted(it[0] for it in p.items)
        check(got == ["f_both", "f_pet_auto", "f_pet_manual"],
              "「只看有宠物」筛出来的不对（自动 + 人工两侧都该算 ✓，别的帧不该进来 ✓）：%r"
              % (got,))

        # ③ 切回「全部」⇒ 五帧都在（不是把列表弄空了 ✓）
        p.cmb_filter.setCurrentIndex(p.cmb_filter.findData("all"))
        app.processEvents()
        check(len(p.items) == 5, "切回「全部」没有 5 帧：%r" % ([it[0] for it in p.items],))

        # ④ 宠物框**不算空帧**：只看空帧里只能有 f_none ✓
        p.cmb_filter.setCurrentIndex(p.cmb_filter.findData("zero"))
        app.processEvents()
        check([it[0] for it in p.items] == ["f_none"],
              "「只看空帧」把有宠物框的帧也算进去了 ✗（宠物框也是框 ✓ 那不算漏检 ✓）：%r"
              % ([it[0] for it in p.items],))
    finally:
        shutil.rmtree(td, ignore_errors=True)


def t_frame_line_shows_resolution():
    """⭐⭐ 质检台**帧名后面显示分辨率**（用户 2026-10-05 ✓ 原话："在质检台帧名后面显示分辨率"）。

    为什么值得放在这一行：**同一批里混进别的分辨率**真会发生 ✓，而且后果隐蔽 ✗ ——
    实测现场（阳光沙滩那个项目）：前 158 帧 1920×1080、后 28 帧 **1280×720**（重采时窗口变小 ✓）
    ⇒ 标定是按 1080p 做的（`scale_at` ✓）⇒ 720p 那批里目标小 1.5 倍、模板大 1.5 倍
    ⇒ **那批帧一个都检不出** ✗（日志只说"scale 不对" ✓ 人会被带偏 ✓）。

    钉四件：① 帧名后面**就有**分辨率 ✓；② 与**标定分辨率一致**的帧**不报警** ✓
    （天天报警等于没报警 ✗）；③ 与标定**不一致**的帧当场点出来（"与标定不符" ✓）；
      ④ 提示（tooltip）里写清**后果与处理办法**（分辨率对不上 ⇒ 模板匹配一个都检不出 ✓）。
    """
    import shutil
    import tempfile

    from PyQt5.QtGui import QPixmap
    from PyQt5.QtWidgets import QApplication

    from gui.project import Project
    from gui.review import ReviewPanel

    td = tempfile.mkdtemp(prefix="revres_")
    try:
        # ⚠ app 要先建（下面就是 `QPixmap(...).save()` ✓ 没 app 会 `qFatal` ⇒ 0xC0000409 ✓）
        app = QApplication.instance() or QApplication([])
        proj = Project.create(Path(td), name="用例", map_id="105090600")
        proj.frames.mkdir(parents=True, exist_ok=True)
        QPixmap(64, 48).save(str(proj.frames / "frame_00001.png"))     # 与标定一致 ✓
        QPixmap(32, 24).save(str(proj.frames / "frame_00002.png"))     # 换过窗口的那批 ✓
        proj.set("scale_at", {"width": 64, "height": 48, "confidence": "manual"})
        proj.save()

        p = ReviewPanel()
        p.bind(proj)
        app.processEvents()

        t1 = p.lbl_frame.text()
        check("64×48" in t1, "① 帧名后面没显示分辨率（用户点名要的 ✗）：%r" % t1)
        check("frame_00001" in t1, "① 帧名本身丢了：%r" % t1)
        check("与标定不符" not in t1,
              "② 与标定**一致**的帧不该报警（天天报警等于没报警 ✗）：%r" % t1)

        p.step(1)
        app.processEvents()
        t2 = p.lbl_frame.text()
        check("32×24" in t2, "① 第二帧也没显示分辨率：%r" % t2)
        check("与标定不符" in t2 and "64×48" in t2,
              "③ 分辨率与标定不同的帧没当场点出来（那正是「一个都检不出」的原因 ✓）：%r" % t2)
        tip = p.lbl_frame.toolTip()
        check("检不出" in tip and "重标" in tip,
              "④ 提示里没说清后果与处理办法（只说「不一样」没用 ✓）：%r" % tip)
    finally:
        shutil.rmtree(td, ignore_errors=True)


def t_drop_class_and_filter():
    """⭐⭐ 质检台**能手动标掉落物** + 筛选里能**只看有掉落物**（用户 2026-10-04 ✓ 第 1、2 条：
    "质检台现在需要能手动标注掉落物类" / "筛选策略里增加『只看有掉落物』"）。

    掉落物这条线里，**自动标注只产 玩家/怪物**（见 `tools/label_drops` ✓），掉落框基本靠人工补
    ⇒ 质检台这个入口是必需的 ✓。

    钉六件：
      ① 「新建框」下拉里有**掉落物**（`CLASS_DROP = 2` ✓），且选中它之后
         `canvas.current_cls` 跟着变 ✓；
      ② 用它真拉一个框 ⇒ 落盘/取值都是 **class 2**、且记为**人工框** ✓；
      ③ 筛选里多了「只看有掉落物」✓，选中后**只剩有掉落框的帧** ✓
         （人工 `labels/` 和自动 `labels_auto/` **两边都算** ✓ —— 取的是合并结果 ✓）；
      ④ 「全部」下三帧都在 ✓（对照 ✓ 证明③不是"把列表弄空了"）；
      ⑤ 老筛选（只看空帧）没坏 ✓；
      ⑥ 「适应有掉落物」... 即：筛选码是 `has_drop` ✓（别用会跟别的模式撞的字符串 ✓）。
    """
    import shutil
    import tempfile

    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtGui import QKeyEvent
    from PyQt5.QtWidgets import QApplication

    from gui.review import ReviewPanel

    app = QApplication.instance() or QApplication([])
    td = tempfile.mkdtemp(prefix="rv_drop_")
    try:
        proj = _mk_project(td, (
            ("f_mob", [(labelio.CLASS_MOB, 10, 10, 40, 40, True)]),
            ("f_drop", [(labelio.CLASS_MOB, 10, 10, 40, 40, True),
                        (labelio.CLASS_DROP, 100, 100, 20, 20, True)]),
            # 掉落框在**自动**那一侧（labels_auto）也要算进来 ✓ —— 将来自动标注产掉落物时同理 ✓
            ("f_drop_auto", [(labelio.CLASS_DROP, 50, 50, 24, 24, False)]),
            ("f_empty", []),
        ))
        p = ReviewPanel()
        p.resize(900, 620)
        try:
            # ① 下拉里有掉落物
            items = [p.cmb_cls.itemText(i) for i in range(p.cmb_cls.count())]
            i_drop = p.cmb_cls.findData(labelio.CLASS_DROP)
            check(i_drop >= 0 and "掉落" in items[i_drop],
                  "「新建框」下拉里没有掉落物类（用户第 1 条 ✗）：%r" % (items,))

            p.bind(proj)
            check(len(p.items) == 4, "「全部」下不是 4 帧：%r" % ([it[0] for it in p.items],))

            # ② 选中掉落物类 ⇒ 真拉一个框 ⇒ class 2 + 人工
            p.cmb_cls.setCurrentIndex(i_drop)
            check(p.canvas.current_cls == labelio.CLASS_DROP,
                  "选了掉落物，画布的新建框类别没跟着变：%r" % (p.canvas.current_cls,))
            n0 = p.canvas.count()
            _drag(p.canvas, (200, 200), (260, 260))
            app.processEvents()
            check(p.canvas.count() == n0 + 1,
                  "空白拖没建出框（掉落物标注就走不通了 ✗）：%d -> %d"
                  % (n0, p.canvas.count()))
            cls, _x, _y, _w, _h, manual = p.canvas.get_boxes()[-1]
            check(cls == labelio.CLASS_DROP and manual,
                  "新建的框不是「掉落物 + 人工」：cls=%r manual=%r" % (cls, manual))

            # ③④ 筛选：只看有掉落物 ⇒ 剩两帧（人工一帧 + 自动一帧 ✓）
            i_f = p.cmb_filter.findData("has_drop")
            check(i_f >= 0,
                  "筛选里没有「只看有掉落物」（用户第 2 条 ✗）：%r"
                  % ([p.cmb_filter.itemText(i) for i in range(p.cmb_filter.count())],))
            p.cmb_filter.setCurrentIndex(i_f)
            app.processEvents()
            got = sorted(it[0] for it in p.items)
            check(got == ["f_drop", "f_drop_auto"],
                  "「只看有掉落物」筛出来的不对（人工 + 自动两侧都该算 ✓）：%r" % (got,))
            check("has_drop" == str(p.cmb_filter.currentData()),
                  "筛选码不对（`reload` 里靠它分派 ✗）：%r" % (p.cmb_filter.currentData(),))

            # 换个筛选项再换回来，随后「全部」要能复原（证明③不是把列表弄空了 ✓）
            p.cmb_filter.setCurrentIndex(p.cmb_filter.findData("all"))
            app.processEvents()
            check(len(p.items) == 4, "切回「全部」没有 4 帧：%r" % ([it[0] for it in p.items],))
            # ⑤ 老筛选没坏
            p.cmb_filter.setCurrentIndex(p.cmb_filter.findData("zero"))
            app.processEvents()
            check([it[0] for it in p.items] == ["f_empty"],
                  "「只看空帧」坏了（老功能回归 ✗）：%r" % ([it[0] for it in p.items],))
        finally:
            p.close()
    finally:
        shutil.rmtree(td, ignore_errors=True)


def t_op_modes_and_band_select():
    """⭐⭐⭐ 操作模式：**Q 标注 / W 选择**，选择模式的框选是**黑色虚线**、**不建框**、
    **选中被框住的框**（用户 2026-10-04 ✓ 第 3 条）。

    原文："质检台右上角的『适应窗口』改为操作模式的下拉列表，默认是『标注模式』，可以改为
    『选择模式』，该模式选中后框选不能创建标注框且显示的框选范围是黑色虚线，而是选中被框住的
    标注框（方便批量删除）。增加快捷键：Q 切换到标注模式，W 切换到选择模式。"

    钉七件：
      ① 下拉替掉了那个「适应窗口」按钮 ✓（它没消失：**双击**仍是适应窗口 ✓ 提示行里也写着 ✓）；
      ② 默认 = **标注模式** ✓，空白拖 ⇒ **建框** ✓（老行为没变 ✓）；
      ③ 切选择模式 ⇒ 空白拖**不建框** ✓、橡皮筋是**黑色 + 虚线** ✓；
      ④ 松手 ⇒ **选中被框住的框** ✓（没被框住的不选 ✓）；
      ⑤ ⚠ **选中不算修改** ✓（不许发 `boxes_changed` ✗ —— 发了就会把"只是框选看了一眼"的帧
         标成人工改过 ✓ 见 `ImageCanvas._select_in_rect` 里那段说明 ✓）；
      ⑥ 框选之后**能一次删多个**（`remove_selected` ✓ 正是这条需求的目的 ✓）；
      ⑦ 快捷键 Q / W 真的切成对应模式，**而且下拉框跟着变**（不然人会以为快捷键没生效 ✗）。
    """
    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtGui import QKeyEvent, QPixmap
    from PyQt5.QtWidgets import QApplication, QShortcut

    from gui.canvas import MODE_LABEL, MODE_SELECT, ImageCanvas
    from gui.review import ReviewPanel

    app = QApplication.instance() or QApplication([])
    _ = QApplication.instance() or QApplication(sys.argv)

    # ① 源码钉：按钮让位给下拉（"适应窗口"这四个字还在注释/提示里 ⇒ 只钉那个**按钮** ✓）
    _src = (Path(__file__).resolve().parents[1] / "gui" / "review.py").read_text(encoding="utf-8")
    check('QPushButton("适应窗口")' not in _src,
          "右上角还留着「适应窗口」按钮（用户明确要它让位给操作模式下拉 ✗）")
    check("cmb_mode" in _src, "没有操作模式下拉 ✗")

    # ② 默认标注模式 + 空白拖建框
    c = ImageCanvas()
    c.resize(520, 420)
    c.show()
    app.processEvents()
    c.load(QPixmap(_IMG_W, _IMG_H),
           [(labelio.CLASS_MOB, 20, 20, 40, 40, True),
            (labelio.CLASS_MOB, 220, 200, 50, 50, True)], editable=True, fit=False)
    app.processEvents()
    check(c.mode == MODE_LABEL, "画布默认不是标注模式（用户要默认标注模式 ✗）：%r" % (c.mode,))
    changed = []
    c.boxes_changed.connect(lambda: changed.append(1))
    n0 = c.count()
    _drag(c, (5, 5), (70, 70))
    app.processEvents()
    check(c.count() == n0 + 1, "标注模式下空白拖没建框（老行为被改坏了 ✗）：%d -> %d"
          % (n0, c.count()))

    # ③④⑤ 选择模式：黑虚线框选、不建框、选中框住的、不算修改
    c.set_mode(MODE_SELECT)
    # ⚠ 光标也换了（箭头 = 挑框 ✓ 十字 = 画框 ✓）—— 拿不到就算了，别让它把用例弄红 ✓
    check(c.mode == MODE_SELECT, "set_mode 没切成选择模式：%r" % (c.mode,))
    c.clear_selection()
    n1, ch1 = c.count(), len(changed)
    # ⚠ 框选**只盖住那一个已知的框**（220,200,50,50 ✓）—— 别用 (0,0)-(150,150)：
    #   上面标注模式那一步已经在 (5,5) 建了一个框，一起被盖住 ⇒ 该是 2 个 ✗
    #   （第一版就是这么写出假红的 ✓ 记一笔 ✓）。
    _send(c.viewport(), QEvent.MouseButtonPress, c.mapFromScene(200, 190))
    check(c._rubber is not None, "按下后没有橡皮筋（框选看不见 ✗）")
    pen = c._rubber.pen()
    check(pen.color().name() == "#000000" and pen.style() == Qt.DashLine,
          "框选范围不是**黑色虚线**（用户明确要求 ✗）：%r / %r"
          % (pen.color().name(), pen.style()))
    _send(c.viewport(), QEvent.MouseMove, c.mapFromScene(290, 270), Qt.NoButton)
    _send(c.viewport(), QEvent.MouseButtonRelease, c.mapFromScene(290, 270))
    app.processEvents()
    check(c.count() == n1,
          "选择模式的框选**建了框**（用户明确说「不能创建标注框」✗）：%d -> %d"
          % (n1, c.count()))
    check(c.selected_count() == 1,
          "框选没选中被框住的那个框（该 1 个 ✓）：%r" % (c.selected_count(),))
    check(len(changed) == ch1,
          "框选发了 `boxes_changed` ⇒「只是看了一眼」的帧会被当成人工改过 ✗：%d 次"
          % (len(changed) - ch1))

    # ⑥ 框选 + Del 一次删多个
    c.clear_selection()
    _send(c.viewport(), QEvent.MouseButtonPress, c.mapFromScene(0, 0))
    _send(c.viewport(), QEvent.MouseMove, c.mapFromScene(380, 280), Qt.NoButton)
    _send(c.viewport(), QEvent.MouseButtonRelease, c.mapFromScene(380, 280))
    app.processEvents()
    check(c.selected_count() == c.count() and c.count() > 1,
          "全选框选没把框都选上：%r / %r" % (c.selected_count(), c.count()))
    c.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier))
    app.processEvents()
    check(c.count() == 0, "框选之后按 Del 没一次删掉（这条需求的目的 ✗）：剩 %d 个" % c.count())

    # ⑦ 面板：默认 + 快捷键 Q/W（**下拉也要跟着变** ✓）
    p = ReviewPanel()
    p.resize(900, 620)
    try:
        scs = {sc.key().toString(): sc for sc in p.findChildren(QShortcut)}
        check({"Q", "W"} <= set(scs),
              "没有 Q / W 两个快捷键（用户第 3 条 ✗）：%r" % (sorted(scs),))
        check(p.cmb_mode.currentData() == MODE_LABEL and p.canvas.mode == MODE_LABEL,
              "面板默认不是标注模式：%r / %r" % (p.cmb_mode.currentData(), p.canvas.mode))
        scs["W"].activated.emit()
        check(p.canvas.mode == MODE_SELECT and p.cmb_mode.currentData() == MODE_SELECT,
              "按 W 没切成选择模式 / 下拉没跟着变：%r / %r"
              % (p.canvas.mode, p.cmb_mode.currentData()))
        scs["Q"].activated.emit()
        check(p.canvas.mode == MODE_LABEL and p.cmb_mode.currentData() == MODE_LABEL,
              "按 Q 没切回标注模式 / 下拉没跟着变：%r / %r"
              % (p.canvas.mode, p.cmb_mode.currentData()))
        # ★ 下拉自己切也要生效（不是只有快捷键那条路 ✓）
        p.cmb_mode.setCurrentIndex(p.cmb_mode.findData(MODE_SELECT))
        check(p.canvas.mode == MODE_SELECT,
              "直接改下拉没生效：%r" % (p.canvas.mode,))
    finally:
        p.close()
        c.deleteLater()



def t_region_auto_label_mode():
    """⭐⭐⭐ 「区域自动标注」操作模式：框一块 ⇒ **弹窗**（用户 2026-10-11 ✓ 原话：
    "给质检台新增「区域自动标注」操作模式，在该模式，框选一个区域后出现弹窗…"）。

    钉五件：
      ① 下拉里真有这一项（`MODE_REGION` ✓）且默认不是它 ✗（默认仍是标注模式 ✓ 老行为不变 ✓）；
      ② 画布那一侧：**只有这个模式**松手才发 `region_picked` ✓，而且**不建框、不动选中** ✗；
      ③ 面板那一侧：框完真的去开弹窗 ✓，**参数对**（当前帧 + 那个矩形 ✓）；
      ④ 框太小 / 没项目 / 没帧 ⇒ **不开弹窗**，只把人话写在状态行 ✓（别弹个没用的窗 ✗）；
      ⑤ ~~模态（`exec_`）~~ ⇒ ⭐⭐⭐ **2026-10-16 口径变更**（用户原话："**区域自动标注 弹窗
         打开时需要能交互主窗口**"✓）：改判"**非模态 + 目标帧跟着当前帧走**"✓ ——
         原来那条风险（"背后切帧 ⇒ 以为标的是新那一帧"✗）**换了挡法**：让弹窗标的永远等于
         你正看着的那一帧（`set_target` + `_load_current` 里那次推送 ✓）⇒ 风险从根上不成立 ✓。
    """
    import inspect

    from gui.canvas import MODE_LABEL, MODE_REGION, ImageCanvas
    from gui.review import ReviewPanel

    _root = Path(__file__).resolve().parents[1]
    _src = (_root / "gui" / "review.py").read_text(encoding="utf-8")
    _csrc = (_root / "gui" / "canvas.py").read_text(encoding="utf-8")

    # ---- ①② 源码级：模式存在 / 画布只在它下面发信号 / 不建框 ----
    check("MODE_REGION" in _src and "区域自动标注" in _src,
          "质检台下拉里没有「区域自动标注」这一项 ✗（用户 2026-10-11 点名要它 ✓）")
    check("region_picked" in _csrc and "MODE_REGION" in _csrc,
          "画布没有「区域自动标注」模式 / 没有 `region_picked` 信号 ✗")
    _branch = _csrc.split("if self._region_band:", 1)[-1].split("if band:", 1)[0]
    check("region_picked.emit" in _branch,
          "那个模式松手后**没把矩形交出去** ✗ ⇒ 弹窗打不开 ✓")
    # ⚠ 只看**真调用**（`self.xxx.emit` / `self._select_in_rect(` ✓）—— 那段注释里就写着
    #   "不建框、不发 boxes_changed" ✓，拿**字符串**去查会把自己的注释当违规 ✗（踩过一次 ✓）。
    check("boxes_changed.emit" not in _branch and "_select_in_rect(" not in _branch,
          "区域框选**改动了标注**（建框 / 选中）✗ —— 它只该「圈一块地方」✓")
    _mw = inspect.getsource(ReviewPanel._on_region_picked)
    # ⭐⭐⭐ **口径变更（2026-10-16 ✓ 用户原话："区域自动标注 弹窗打开时需要能交互主窗口"✓）**：
    #   本条原样是与它**相反**的一条 —— `check("exec_()" in _mw, "弹窗不是模态的 ✗ ⇒
    #   背后还能切帧 ⇒ 人会以为标的是新那一帧 ✓")` ✓（当年靠**模态**挡那个风险 ✓）。
    #   ⚠ 现在**不是把风险当不存在** ✗，而是**换挡法** ✓：**目标帧跟着当前帧走** ✓
    #   ⇒ "以为标的是新那一帧"从根上不成立 ✓（弹窗上还常显「目标帧：xxx」✓ 看得见才敢按 ✓）。
    #   ⇒ 判据随之从"必须 `exec_`" 改成"**必须非模态，且那条链在**"✓。
    check("exec_()" not in _mw and "setModal(False)" in _mw,
          "弹窗还是模态的 ✗ ⇒ 主窗口点不动 ✓（用户 2026-10-16 要求能交互 ✓）")
    check("set_target" in _mw and "set_target" in _src,
          "目标帧没跟着当前帧走 ✗ ⇒ 换了帧再按按钮会以为标的是新帧 ✓"
          "（这条替代了原来「靠模态挡住」那条 ✓）")
    check("_region_dlg" in _mw and "_on_region_closed" in _mw,
          "非模态的结果没人收 / 会开出第二个窗 ✗（要存引用 + 从信号收结果 ✓）")

    # ---- ③④ 行为级：框完开弹窗，参数对；太小/没帧 ⇒ 只写状态 ----
    import gui.region_label_dialog as _rld
    from PyQt5.QtCore import QRectF

    app = QApplication.instance() or QApplication([])
    panel = ReviewPanel()
    panel.resize(640, 480)
    panel.show()
    app.processEvents()

    _opened = []

    class _Sig:
        """假信号（够 `connect` 用 ✓）——非模态之后结果是从信号收的 ✓。"""

        def connect(self, *_a, **_k):
            return None

    class _FakeDlg:
        def __init__(self, parent, project, stem, region, *a, **k):
            _opened.append((stem, region))
            self.stem = stem
            self._done_once = False
            self.finished = _Sig()          # ⭐ 非模态：面板会连它 ✓（见 `_on_region_closed` ✓）
            # ⭐ **跑完一趟就报**那条（用户 2026-10-16 ✓ "显示完成但没看到框"✓）——
            #   质检台靠它**每跑完一次就重载** ✓ 见 `review._on_region_labeled` ✓。
            self.labeled = _Sig()
            self.modal_calls = []
            self.targets = []

        # ---- 非模态所需的那几个接口（用户 2026-10-16 ✓ 见 `_on_region_picked` ✓）----
        def setModal(self, on):             # noqa: N802 —— 照 Qt 的驼峰 ✓
            self.modal_calls.append(bool(on))

        def setWindowModality(self, _m):    # noqa: N802
            return None

        def show(self):
            return None

        def isVisible(self):                # noqa: N802
            return False                    # 当"已经关掉"⇒ 后面几次调用各开各的 ✓

        def set_target(self, stem):
            self.targets.append(stem)

        def raise_(self):
            return None

        def activateWindow(self):           # noqa: N802
            return None

        def exec_(self):
            return 0

    # ⚠ `_on_region_picked` 里是**调用时**才 `from gui.region_label_dialog import …` ✓
    #   ⇒ 把**那个模块上的名字**换掉就能拦住它 ✓（不用碰 import 语句 ✓）
    _o_dlg = _rld.RegionLabelDialog
    try:
        _rld.RegionLabelDialog = _FakeDlg
        # 面板里塞一份"假的当前帧"，模拟已经打开项目、选好帧 ✓
        panel.project = object()
        panel.items = [("frame_00042", 3, False)]
        panel.index = 0
        panel._on_region_picked(QRectF(10, 20, 300, 200))
        check(_opened == [("frame_00042", "10,20,300,200")],
              "框完没按「当前帧 + 那个矩形」开弹窗 ✗：%r" % (_opened,))
        # ⭐⭐ **非模态**（用户 2026-10-16 ✓ 口径变更 ✓）：面板必须**关掉模态**再 `show` ✓
        #   —— 只 `show()` 不 `setModal(False)` 在某些风格下照样挡主窗口 ✗（两件一起做才稳 ✓）。
        _d0 = panel._region_dlg
        check(getattr(_d0, "modal_calls", None) == [False],
              "没把弹窗设成非模态 ✗ ⇒ 主窗口点不动 ✓（用户的第 1 条 ✓）")
        # ⭐ **换帧 ⇒ 目标帧跟着走** ✓（这就是替代"靠模态挡住"的那条链 ✓）
        #   ⚠ 走**源码级**判据（同本文件惯例 ✓）：行为级要真跑 `_load_current` ⇒
        #     得造一个够用的 `project`（`frames` / `dir_of` / `labelio.*` 一串 ✓）——
        #     那会把本条用例变成"测双份" ✗，不值 ✓（推送那一句就在这个方法里 ✓）。
        check("set_target" in inspect.getsource(ReviewPanel._load_current),
              "换帧时没把新帧名推给弹窗 ✗ ⇒ 人会以为标的是新那一帧 ✓（原来的风险又来了 ✗）")
        _opened.clear()
        panel._on_region_picked(QRectF(10, 20, 1, 1))          # 太小
        check(not _opened, "框太小也弹窗 ✗ —— 该只写一句人话 ✓")
        check("太小" in panel.lbl_status.text(),
              "框太小时状态行没说清 ✗：%r" % panel.lbl_status.text())
        panel.items = []
        panel._on_region_picked(QRectF(10, 20, 300, 200))      # 没帧
        check(not _opened, "没选帧也弹窗 ✗")
        check("先打开项目" in panel.lbl_status.text(),
              "没帧时该说「先打开项目、选一帧」✗：%r" % panel.lbl_status.text())
    finally:
        _rld.RegionLabelDialog = _o_dlg
        panel.close()

    # ---- 列表面板里的那一项：默认仍得是标注模式 ✓ ----
    _p2 = ReviewPanel()
    check(_p2.cmb_mode.currentData() == MODE_LABEL,
          "默认模式变了 ✗（该仍是标注模式 ✓ 老习惯不能改 ✓）")
    _i = _p2.cmb_mode.findData(MODE_REGION)
    check(_i >= 0, "下拉里找不到 MODE_REGION ✗")
    _p2.cmb_mode.setCurrentIndex(_i)
    check(_p2.canvas.mode == MODE_REGION, "选了这一项，画布模式没跟上 ✗")
    check("青框" in _p2.lbl_status.text() or "框" in _p2.lbl_status.text(),
          "切到这个模式后状态行该说「拖一个框」✗：%r" % _p2.lbl_status.text())
    _p2.close()


def t_region_dialog_limits_scope():
    """⭐⭐⭐ 弹窗必须**只标当前帧 + 只标框选区域**（用户 2026-10-11 ✓ "点击各个标注按钮，
    **只会在当前帧框选的区域内**进行自动标注"）。

    钉五件：
      ① 四块**真的是卡片原件**搬进来的（`parent()` 是弹窗里那一节 ✓）—— "直接照抄"就该是
         **同一份控件** ✗ 不是抄一份代码 ✓（抄一份必然越用越歪 ✓）；
      ② YOLO 权重下拉也是**原件**（选中的模型卡片自己读得到 ✓ 一处口径 ✓）；
      ③ 勾选项 ⇒ 那一节显隐（`isHidden()` 翻 ✓）；
      ④ 卡片被改成"**就这一帧**"（`_pick_frames` 覆盖 ✓ 不弹选帧窗 ✓）；
      ⑤ ⚠⚠ **走真入口跑时** `only=[当前帧]` + `region=那个矩形` **四个目标全都要对** ✗
         —— 这是这一屏的全部意义 ✓（少一个 ⇒ 那个按钮偷偷标了整份画面 ✓）。
    """
    from PyQt5.QtWidgets import QApplication

    import gui.region_label_dialog as RLD
    from gui.project import last_opened
    from gui.region_label_dialog import TARGETS

    app = QApplication.instance() or QApplication([])
    proj = last_opened()
    check(proj is not None, "这台机器上没有最近打开过的项目 ⇒ 这条用例没法验真数据 ✗")

    _spec = "10,20,300,200"
    dlg = RLD.RegionLabelDialog(None, proj, "frame_00001", _spec)
    try:
        # ---- ①② 原件搬进来了 ----
        for key, label, attr in TARGETS:
            box = getattr(dlg.card, attr, None)
            check(box is not None, "卡片里没有 `%s` ✗（弹窗没得搬 ✓）" % attr)
            check(box.parent() is dlg.sections[attr],
                  "「%s」那块**不是卡片原件**（没搬进弹窗 ✓）✗ ⇒ 参数会越抄越歪 ✓" % label)
        check(dlg.card.cmb_yolo.parent() is dlg,
              "YOLO 权重下拉没搬进弹窗 ✗ ⇒ 弹窗里选的模型卡片读不到 ✓（两处口径 ✗）")

        # ---- ③ 勾选 ⇒ 显隐 ----
        _sec = dlg.sections["drop_box"]
        check(_sec.isHidden(),
              "一开始该收着 ✓（用户：「勾选项被勾选时，在该勾选项下面显示出详细的配置内容」✓）")
        dlg.checks["drop"].setChecked(True)
        check(not _sec.isHidden(), "勾上了却没收起来/没展开 ✗")
        dlg.checks["drop"].setChecked(False)
        check(_sec.isHidden(), "取消勾选后该收回去 ✗")

        # ---- ④ 只这一帧（不弹选帧窗 ✓）----
        check(dlg.card._pick_frames("drop") == (True, ["frame_00001"]),
              "卡片没被改成「就这一帧」✗ ⇒ 点按钮会弹选帧窗（用户要的是当前帧 ✓）")
        check(dlg.card._region == _spec or dlg.card._region == RLD.region_mod.to_spec(_spec),
              "卡片没被设成「只标这块区域」✗：%r" % (dlg.card._region,))

        # ---- ⑤ 走真入口：四个目标都要 only + region ----
        _seen = {}

        class _FakeTask:
            def __init__(self, fn, params, parent=None):
                _seen["fn"], _seen["params"] = fn, params

                class _S:
                    def connect(self, *_a, **_k):
                        pass

                self.sig_log = self.sig_progress = self.sig_done = _S()
                self.finished = _S()

            def start(self):
                _seen["started"] = True

            def cancel(self):
                pass

        _o_task, _o_save = RLD.TaskThread, proj.save
        try:
            RLD.TaskThread = _FakeTask
            proj.save = lambda: None                     # ⚠ 用例不许改真项目 ✓
            for _t in ("drop", "pet", "mob", "player"):
                dlg.task = None                          # 上一轮的替身没收尾（那条闸是对的 ✓）
                dlg.card._label_target = _t
                dlg.card._drop_mode = dlg.card._pet_mode = "template"
                _seen.clear()
                dlg._on_run(dlg.card)
                _p = _seen.get("params") or {}
                _p = _p.get("detect") or _p
                check(_seen.get("started") is True,
                      "「%s」那条没起成任务 ✗（`_on_run` 该把它跑起来 ✓）" % _t)
                check(_p.get("only") == ["frame_00001"],
                      "「%s」的任务**没被钉成「只当前帧」** ✗ ⇒ 会重标整份画面 ✓：%r"
                      % (_t, _p.get("only")))
                check(_p.get("region"),
                      "「%s」的任务里**没有 region** ✗ ⇒ 那个按钮会标全图 ✓：%r"
                      % (_t, _p.get("region")))
        finally:
            RLD.TaskThread, proj.save = _o_task, _o_save
    finally:
        dlg.close()


def t_region_forwarded_in_chains():
    """⭐⭐⭐ 「区域」必须**真的传到标注内核**（用户 2026-10-11 ✓ "只会在当前帧框选的区域内
    进行自动标注"）。

    为什么单钉这一条：弹窗那一条只验到"任务参数里有 `region`" ✓ —— 而**中间那几层**
    （`label_drops` / `label_pets` / `yolo_augment` / `detect_player` ✓）要是**没往下转**，
    表现就是"弹窗里框了区域、结果还是标了整幅画面" ✗✗ —— 那比没有这个功能**更坏** ✓
    （人会以为已经限定了 ✓）。所以每一层都钉：
      · `label_drops` / `label_pets`：两处都要转（模板匹配那条 + YOLO 那条 ✓）；
      · `yolo_augment`：从 params 里读出来 + **按中心点筛掉区域外的预测** ✓；
      · `detect_player`：cfg 里带上 + `work()` 里筛 ✓；
      · 内核 `perceive/region.py`：纯函数的四种形态 ✓（空 ⇒ **一律算"在区域内"** ✓ = 老行为 ✓）。
    """
    _root = Path(__file__).resolve().parents[1]

    def _src(p):
        return (_root / p).read_text(encoding="utf-8")

    # ---- ① 两条 label_*：模板匹配 + YOLO 都要转 ----
    for _f, _n in (("tools/label_drops.py", 2), ("tools/label_pets.py", 2)):
        _t = _src(_f)
        _cnt = _t.count('"region": params.get("region")')
        check(_cnt >= _n,
              "%s 里只转了 %d 处 region ✗（该 ≥ %d：模板匹配那条 + YOLO 那条 ✓）"
              "⇒ 有一类按钮会**偷偷标整幅画面** ✓" % (_f, _cnt, _n))

    # ---- ② yolo_augment：读 + 筛 ----
    _y = _src("tools/yolo_augment.py")
    check("region = region_mod.parse(params.get" in _y,
          "`yolo_augment` 没从 params 里读 region ✗ ⇒ YOLO 那几类照样标全图 ✓")
    check("region_mod.center_inside(" in _y,
          "`yolo_augment` 没**筛**区域外的预测 ✗（判据要跟别的链同一套 ✓ 见 `region.py` ✓）")
    check("boxes_outside" in _y,
          "被区域筛掉的框数没报出来 ✗ ⇒ 「框 0 个」时分不清是模型没检出还是全在框外 ✓")

    # ---- ③ detect_player：cfg 里带 + work 里筛 ----
    _d = _src("tools/detect_player.py")
    check('"region": params.get("region"),' in _d,
          "`detect_player` 的参数里没带 region ✗ ⇒ 玩家那条照样标全图 ✓")
    check("region_mod.center_inside(cfg.get(\"region\")" in _d,
          "`detect_player.work()` 没按区域筛玩家框 ✗")

    # ---- ④ 内核纯函数（一处口径 ✓）----
    from perception.world import region as R

    check(R.to_spec((10, 20, 300, 200)) == "10,20,300,200", "`to_spec` 元组形态不对 ✓")
    check(R.to_spec(None) == "" and R.to_spec((0, 0, 0, 0)) == "",
          "空框该给空串 ✓（面板靠它判「太小」✓）")
    check(R.parse("10,20,300,200") == (10, 20, 300, 200), "`parse` 字符串形态不对 ✓")
    check(R.parse("") is None and R.parse("坏") is None and R.parse("1,2,3") is None,
          "坏输入该给 None ✓（不许抛 ✗）")
    check(R.center_inside("", 99999, 99999) is True,
          "**没框区域时**该一律算「在区域内」✗（那才是老行为 ✓ 一个字节都不改 ✓）")
    check(R.center_inside("0,0,100,100", 50, 50) is True
          and R.center_inside("0,0,100,100", 150, 50) is False,
          "中心点判据不对 ✓（区域外的必须 False ✓）")
    # 归一化框：第 0 条中心 = (500,500)、第 1 条 = (50,50) ✓
    _rows = [(1, 0.5, 0.5, 0.1, 0.1), (0, 0.05, 0.05, 0.1, 0.1)]
    # ⇒ 只框住 (400,400)-(700,700) 那块 ⇒ 只有第 0 条在区域内 ✓
    check(R.filter_norm("400,400,300,300", _rows, 1000, 1000) == [_rows[0]],
          "`filter_norm` 没把区域外的归一化框筛掉 ✗（YOLO/玩家那两条要用它 ✓）")
    check(R.filter_norm("", _rows, 1000, 1000) == _rows,
          "没区域时 `filter_norm` 改了内容 ✗（老行为必须一字不变 ✓）")


def t_region_merge_keeps_outside_boxes():
    """⭐⭐⭐ **区域模式只许改区域内**：区域外的旧框必须留住（2026-10-16 事故 ✓ 用户原话：
    「**并不是只标注我框选的区域，而是把整帧其他的标注框全删掉了**」✓）。

    钉**两条链**（同一处病、两套写法 ✓）：
      · `detect_mobs.merge_class_lines` —— 怪物 / 掉落物 / 宠物 三条链共用 ✓；
      · `detect_player.keep_old_player_lines` —— 玩家 ✓
        （⚠ 它**有意整帧重写 class 0** ✓ ⇒ "区域内不保留、区域外保留" ✓ 与那边不是一个语义 ✗ 别混 ✓）。
    """
    from tools.detect_mobs import merge_class_lines
    from tools.detect_player import keep_old_player_lines

    IN = "1 0.100000 0.100000 0.050000 0.050000"        # 中心 (20,20) ⇒ 区域里 ✓
    OUT = "1 0.800000 0.800000 0.050000 0.050000"       # 中心 (160,160) ⇒ 区域外 ✓
    NEW = "1 0.200000 0.200000 0.050000 0.050000"
    other = "0 0.500000 0.500000 0.050000 0.050000"     # 别的类 ✓
    REG = "0,0,100,100"

    lines, _p = merge_class_lines("\n".join([IN, OUT, other]), [NEW], 1, REG, 200, 200)
    check(OUT in lines, "区域外的**本类**旧框被删了 ✗（= 2026-10-16 那次事故原样复发 ✓）")
    check(NEW in lines, "这一趟的新框没写进去 ✗")
    check(IN not in lines, "区域内的本类旧框没被替掉 ✗（那就不叫「只重标这一块」✓）")
    check(other in lines, "别的类的框被动了 ✗")

    lines2, _p2 = merge_class_lines("\n".join([IN, OUT]), [NEW], 1, None, 200, 200)
    check(OUT not in lines2 and IN not in lines2,
          "**没框区域**时不再是整帧重写 ✗（那是改尺度 / 阈值重标的语义 ✓ 不许变 ✓）")

    lines3, prot = merge_class_lines("\n".join([IN, OUT]), [], 1, REG, 200, 200)
    check(IN in lines3 and OUT in lines3 and prot == 1,
          "区域内 0 检出的**保命**没了 ✗（用户 2026-10-05 那个老坑 ✓）")

    pl_in = "0 0.100000 0.100000 0.050000 0.050000"
    pl_out = "0 0.800000 0.800000 0.050000 0.050000"
    mob = "1 0.500000 0.500000 0.050000 0.050000"
    k = keep_old_player_lines("\n".join([pl_in, pl_out, mob]), REG, 200, 200)
    check(pl_out in k, "区域外的**玩家**旧框被删了 ✗（玩家那条是同一处病 ✓）")
    check(pl_in not in k, "区域内的玩家旧框没被替掉 ✗（玩家那条**有意**整帧重写 class 0 ✓）")
    check(mob in k, "玩家那条把怪物框一起清了 ✗")

    k2 = keep_old_player_lines("\n".join([pl_in, pl_out, mob]), None, 200, 200)
    check(pl_in not in k2 and pl_out not in k2 and mob in k2,
          "没框区域时玩家那条不再是「整帧重写 class 0」✗（那是**有意**的语义 ✓ 不许变 ✓）")


def t_marks_keep_screen_size():
    """⭐⭐⭐ 地形图上的标记（准心 / 圈）**不随缩放变小**（2026-10-18 用户现场 ✓）。

    用户原话：「地形图里的准心现在特别小 根本看不清」✓ ——
    地形图 1312×3488 缩到窗口里只有 **0.14 倍**；标记原来按**画布单位**画（r=14 画布像素）
    ⇒ 屏幕上只剩 **2 px** ✗。改成 `ItemIgnoresTransformations` 之后
    设备变换矩阵恒为 1 ⇒ 屏幕上恒是 2r 像素 ✓（位置仍跟地形走 ✓，那是 `setPos` 的事）。
    """
    from PyQt5.QtGui import QPixmap
    from PyQt5.QtWidgets import QApplication

    from gui.canvas import ImageCanvas

    app = QApplication.instance() or QApplication([])      # noqa: F841
    c = ImageCanvas()
    c.load(QPixmap(1312, 3488), [], editable=False, fit=False)
    n = c.set_markers([(656.0, 1744.0, "玩家 世界(0, 0)", "#ffd400", 14.0)])
    check(n == 1, "标记没画上去（%d 个）" % n)

    grp = c._mark_items[0]
    check(bool(grp.flags() & grp.ItemIgnoresTransformations),
          "标记没有「忽略缩放」⇒ 地形图一缩小就变成 2px、根本看不见 ✗（用户 2026-10-18）")
    check(abs(grp.pos().x() - 656.0) < 1e-6 and abs(grp.pos().y() - 1744.0) < 1e-6,
          "标记的位置没落在画布坐标上（%r）—— 忽略缩放只该管**大小**、不许挪位置 ✗"
          % (grp.pos(),))

    c.scale(0.14, 0.14)                                    # 现场那个缩放
    dev = grp.deviceTransform(c.viewportTransform())
    check(abs(dev.m11() - 1.0) < 1e-6 and abs(dev.m22() - 1.0) < 1e-6,
          "缩到 0.14 倍后标记还是被缩了（设备变换 %r）⇒ 屏幕上依旧看不清 ✗" % (dev.m11(),))


def t_preview_box_is_dashed():
    """⭐⭐ **辅助预览框必须是虚线**（用户 2026-10-17 ✓ 原话：
    「地形图里有一个粉色的框，这是哪里来的？能不能去掉？」+「一直框住了左下鳄鱼平台
    整个 foothold 集合，像是一种选中框」✗）。

    ⚠ 那根本不是检出框、也不是我们画错 —— 是**「选择平台」选中的那个集合的预览框**
      （`route_panel._preview_boxes` ✓）。它原来走通用 `BBoxItem`：
      **实线 + 类别色**（`cls=1` ⇒ 品红）+ 标签只写平台名
      ⇒ 跟"检出结果"**长得一模一样** ✓ 谁都认错 ✓
      （2026-10-06 还因为标签写着"怪物"被当成检出框问过一次 ✓ 同一个根因 ✓）。

    钉四条：
      ① `preview=True` ⇒ 虚线 + 青蓝（不是类别色）✓；
      ② 普通检出框**仍是实线** ✓（别把检出语义改坏 ✗）；
      ③ `route_panel` 真给 `load` 传了 `preview=True` ✓（只在 canvas 加参数没人用 = 白改 ✗）；
      ④ 预览框标签带「平台：」前缀 ✓（只写名字跟类名同一种长相 ✗）。
    """
    from PyQt5.QtCore import Qt as _Qt
    from PyQt5.QtGui import QPixmap

    from gui.canvas import ImageCanvas, PREVIEW_BOX_COLOR

    app = QApplication.instance() or QApplication([])      # noqa: F841
    pm = QPixmap(200, 100)
    pm.fill(_Qt.black)

    cv = ImageCanvas()
    cv.load(pm, [(1, 20, 20, 80, 20, False)], editable=False, fit=False,
            preview=True)
    check(len(cv.boxes) == 1, "预览框没建出来（%d 个）" % len(cv.boxes))
    _pen = cv.boxes[0].pen()
    check(_pen.style() == _Qt.DashLine,
          "预览框还是**实线** ⇒ 跟检出框分不开（用户 2026-10-17 就是这么认错的 ✗）")
    check(_pen.color().name().lower() == PREVIEW_BOX_COLOR.lower(),
          "预览框没用青蓝（还在用类别色 ⇒ 看着像一个品红检出框 ✗）：%s"
          % (_pen.color().name(),))

    cv2 = ImageCanvas()
    cv2.load(pm, [(1, 20, 20, 80, 20, False)], editable=False, fit=False)
    check(cv2.boxes[0].pen().style() == _Qt.SolidLine,
          "**普通检出框**被改成虚线了 ✗（虚线在这套 UI 里专指辅助层 ✓）")

    src = (Path(__file__).resolve().parent.parent / "gui" / "route_panel.py"
           ).read_text(encoding="utf-8")
    check("preview=bool(_pv)" in src,
          "`route_panel` 没给 `load` 传 `preview=True` ⇒ 预览框还是实线品红 ✗")
    check('"平台：%s"' in src,
          "预览框标签没带「平台：」前缀 ⇒ 跟检出框的类名同一种长相 ✗")


TESTS = (
    ("⭐⭐⭐ 地形图标记（准心）尺寸**不随缩放**（用户 2026-10-18：缩到 0.14 倍只剩 2px）",
     t_marks_keep_screen_size),
    ("⭐⭐ 辅助预览框（「选择平台」那块）必须是青蓝虚线、标签带「平台：」"
     "（用户 2026-10-17 认成检出框/选中框）", t_preview_box_is_dashed),
    ("⭐⭐⭐ 区域模式只许改区域内：**区域外的旧框必须留住**（怪物/掉落/宠物/玩家 四条链 ✓）",
     t_region_merge_keeps_outside_boxes),
    ("⭐⭐⭐ 「区域自动标注」模式：框一块 ⇒ 弹窗（模态）；太小/没帧 ⇒ 只写状态、不弹窗"
     "（用户 2026-10-11）",
     t_region_auto_label_mode),
    ("⭐⭐⭐ 区域弹窗：四块是**卡片原件** + 只标当前帧 + 只标框选区域（四个目标全都要对）",
     t_region_dialog_limits_scope),
    ("⭐⭐⭐ 区域**真的传到标注内核**：四条链各自转发/筛选 + 内核纯函数（空区域 = 老行为）",
     t_region_forwarded_in_chains),
    ("⭐⭐ 数据集：「标过但空」的帧能当负样本收（`min_boxes=0`）—— 空文件给 `[]`、"
     "没文件给 `None`（用户 2026-09-28：frame_00381 画面没东西，有益无害就加）",
     t_dataset_empty_frame_is_negative),
    ("⭐ 蒙版与描边框：设置弹窗能建 / 框只有描边 / 框在蒙版上面（2026-09-29 紧急反馈）",
     t_mask_and_outline_box),
    ("⑤ 卡「打开质检台」= 入口按钮样式（与寻路编辑器同一份色值）",
     t_editor_card_entry_button_style),
    ("质检台 ←/→ 切帧：真的换帧、到头停住、空列表不崩", t_left_right_switch_frames),
    ("快捷键范围只有质检台（不许在别的页签抢 ←→）+ 面板/画布聚焦", t_shortcut_scope_is_review_panel_only),
    ("提示行与 ⏴⏵ tooltip 写了 ←/→", t_hint_mentions_arrows),
    ("⭐⭐ 质检台能手动标掉落物 + 筛选「只看有掉落物」（人工/自动两侧都算）",
     t_drop_class_and_filter),
    ("⭐⭐ 筛选「只看有宠物」（用户 2026-10-05）：文案/筛选码 / 人工+自动两侧都算 / "
     "切回全部能复原 / 宠物框不算空帧",
     t_pet_filter),
    ("⭐⭐ 质检台帧名后面显示分辨率 + 与标定不符时当场点出来（用户 2026-10-05）",
     t_frame_line_shows_resolution),
    ("⭐⭐⭐ 操作模式：Q 标注 / W 选择；选择模式黑虚线框选、不建框、选中被框住的框（Del 批量删）",
     t_op_modes_and_band_select),
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
