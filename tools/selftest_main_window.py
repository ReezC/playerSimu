"""主窗口自检：**主视区页签的开合流程**（用户 2026-09-27 要求："主视区也能分页签，
只要会占用主视区的都应该走打开、关闭页签流程"）。

**为什么要盯这个**
    主视区从 `QStackedWidget`（固定页号 `PAGE_LIVE` 那种）换成 `QTabWidget`（页签）之后，
    "占用主视区"只剩**一个入口**：`MainWindow.open_view(widget, 标题)` /
    `close_view(widget)` ✓。
    而"自己 `viewer.setCurrentIndex(...)` / `addTab(...)`"这种旁路**写下去就能跑**，
    不会报错、不会崩 —— 只是等哪天页号一变（多开/关了一个页签）就跳错页，
    表现是"点了查看没反应"✗，而**页号没人会去记** ⇒ 必须有源码级的钉子 ✓。

跑法（离屏，不连流、不碰项目文件）：

    python -m tools.selftest_main_window     # 全过返回 0，有失败返回 1
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _win():
    """建主窗口（离屏）。**只碰界面**：不打开项目、不连流 ✓、**也不连远端键盘** ✓。

    ⭐⭐ 2026-10-05（用户："**从源头彻底封死**"✓）—— **本函数不再打桩** ✗：
      原来那两条 monkeypatch（`decision.input.use_network` / `PlayerPanel._apply_input_device`）
      只护住了**本套件** ✗，而 `gui_smoke` / `selftest_minimap` / `selftest_screen_state` /
      `selftest_decision` 里那几个 `PlayerPanel()` **都没打** ✗ ⇒ 只要项目配置是
      `input_device: remote`（`config/decision.json` 与好几个项目就是 ✓）就照样起那条
      **后台连被控机**的线程 ✗ ⇒ 它几秒后才超时返回，而那时用例早跑完、Qt 对象已销毁 ⇒
      **偶发原生段错误 `0xC0000005`** ✗（时机性 ⇒ "多跑几次"不算验证 ✗）。
      ⇒ 现在闸在**真源**：`decision/input.py::net_allowed` ＋ 线程的**唯一出生地**
      `input.connect_async` ✓ ⇒ 离屏（`QT_QPA_PLATFORM=offscreen`）**连线程都不起** ✓。
      本套件**不再打桩**（打桩反而会掩盖"闸到底有没有生效" ✗），改成**当场断言闸关着** ✓。
    """
    from PyQt5.QtWidgets import QApplication

    from decision import input as _dinput
    from gui import main_window as _mw
    from gui.main_window import MainWindow
    from gui.player_panel import PlayerPanel as _PP

    app = QApplication.instance() or QApplication([])
    # ⚠⚠ **当场自检闸**（不打桩了就得靠它 ✓ 免得哪天判据被改坏、又偷偷去连被控机 ✗）
    assert not _dinput.net_allowed(), \
        "离屏（offscreen）下闸没关 ⇒ 会起那条连被控机的线程（= 段错误 0xC0000005 的真凶 ✗）"
    assert _PP._apply_input_device.__name__ == "_apply_input_device", \
        "面板那条起线程的路又被打了桩（本套件不再打桩 ✓ 打桩会掩盖闸有没有生效 ✗）"
    # ⚠⚠ 2026-10-04：`MainWindow()` 现在会**自动打开"上次那个项目"**（见
    #   `_open_last_project_at_startup` ✓ 用户现场："参数每次重开都要重新填" ⇒ 启动就把
    #   上次那个开上 ✓）。那会让**每条用例**都去开**用户真项目** ✗（慢 + 随
    #   `config/session.json` 变 ⇒ 结果不确定 ✗）⇒ 这里把「最近打开的项目」钉成 None ✓
    #   （= 老行为 ✓ 各条用例照旧"没有项目"）。**专门验"会打开"的那条**自己再打桩 ✓。
    if not hasattr(_mw, "_selftest_orig_last_opened"):
        _mw._selftest_orig_last_opened = _mw.last_opened
        _mw.last_opened = lambda: None
    assert _mw.last_opened() is None, "「最近打开的项目」的打桩没生效（自检要立刻炸 ✗）"
    w = MainWindow()
    w.resize(1200, 800)
    return app, w


def t_view_open_close_flow():
    """开合流程：开一次出现在主视区、再开不重复、关掉不销毁、起始页关不掉。"""
    app, w = _win()
    try:
        v = w.viewer
        # ⭐⭐ **2026-09-28 改口径**：一开始就是 **3 个常驻页签**（用户原话："起始页签、实时页签、
        #    质检台页签现在是 **3 个常驻页签**"✓）—— 它们**启动就都在** ✓
        #    （以前「实时」「质检台」是**按需打开**的 ✗，所以老断言是"一开始只有「起始」"✗）。
        check(v.count() == 3 and [v.tabText(i) for i in range(3)] == ["起始", "实时", "质检台"],
              "主视区一开始该有 3 个常驻页签「起始 / 实时 / 质检台」（实际 %d 个：%s）"
              % (v.count(), [v.tabText(i) for i in range(v.count())]))

        # ① 打开**已经在的**页签 ⇒ **不重复开**、只切过去 ✓（`open_view` 的老契约 ✓）
        _n0 = v.count()
        w.open_view(w.live_panel, "实时")
        check(v.count() == _n0 and v.currentWidget() is w.live_panel,
              "open_view 没把「实时」切到最前面（或重复开了页签）：%d → %d 个 / 当前 %r"
              % (_n0, v.count(), v.currentWidget()))
        check(v.tabText(v.indexOf(w.live_panel)) == "实时",
              "页签标题不对：%r" % v.tabText(v.indexOf(w.live_panel)))

        # ② 再开同一个 = 只切过去，**不许**开第二个
        n = v.count()
        w.open_view(w.live_panel, "实时")
        check(v.count() == n, "同一个控件被开了两个页签（%d → %d）" % (n, v.count()))

        w.open_view(w.review, "质检台")
        check(v.currentWidget() is w.review, "open_view 没切到新开的页签")

        # ③ 关掉 = 从主视区摘下来，**控件不销毁**（下次打开还是原来那个）
        w.close_view(w.review)
        check(v.indexOf(w.review) < 0 and v.currentWidget() is not w.review,
              "close_view 没把页签关掉")
        w.open_view(w.review, "质检台")
        check(v.widget(v.indexOf(w.review)) is w.review,
              "关掉再打开时换了控件（滚动位置/状态都会丢 ✗）")

        # ④ 起始页关不掉（工作页全关光之后总得有个落脚处）
        n = v.count()
        w._on_view_close(v.indexOf(w.home_view))
        check(v.count() == n,
              "「起始」页签被关掉了（主视区就变成一块空白 ✗）：%d → %d" % (n, v.count()))
    finally:
        w.close()


def t_closing_live_tab_says_it_keeps_running():
    """⭐ **「实时」是常驻页签 ⇒ 关不掉**（用户 2026-09-28 改的口径 ✓ 原话："起始页签、实时页签、
    质检台页签现在是 **3 个常驻页签，不需要关闭按钮**"）。

    ⚠ 老用例是"关掉「实时」⇒ 要说清收流/推理还在跑"✗ —— 那个前提**已经不成立**：
      现在它**根本关不掉** ✓（页签栏上连「×」都不显示 ✓ 见 `_refresh_tab_close_buttons`）
      ⇒ 那句"还在跑"的提醒也随"可关"一起去掉了 ✓（**不是漏了，是不需要了** ✓；
        跑不跑仍然由「实时」页里那个「开始 / 停止」管 ✓）。
    """
    app, w = _win()
    # ⭐⭐ **启动时三个常驻页签就都在**（用户 2026-09-28 ✓ 当场问过"**实时页签呢？质检台页签呢？**"✗）
    #   ⚠ 这正是我第一版漏掉的：只做了"关不掉"+摘「×」✗，而这俩页签以前是**按需打开**的
    #     （「实时」靠**顶栏那个按钮** ✓）⇒ 按钮一删就**没入口**了 ✗ ⇒ **"常驻"必须启动就全建** ✓。
    _names = [w.viewer.tabText(i) for i in range(w.viewer.count())]
    check(_names == ["起始", "实时", "质检台"],
          "启动时三个常驻页签**没有都在**（用户当场问过「实时页签呢？质检台页签呢？」✗）：%s"
          % (_names,))
    said = []
    try:
        w.open_view(w.live_panel, "实时")
        w.log = lambda msg, level="info": said.append(str(msg))     # 只截获这一条
        w._on_view_close(w.viewer.indexOf(w.live_panel))
        check(w.viewer.indexOf(w.live_panel) >= 0,
              "「实时」是常驻页签，却**被关掉了**（用户 2026-09-28：3 个常驻页签关不掉 ✗）")
        check(any("常驻" in s for s in said),
              "关「实时」时没说明「它是常驻页签、不能关」：%s" % said)
        # ⭐ 常驻页签上**不该有「×」**（用户："不需要关闭按钮"✓）
        from PyQt5.QtWidgets import QTabBar

        _i = w.viewer.indexOf(w.live_panel)
        check(w.viewer.tabBar().tabButton(_i, QTabBar.RightSide) is None,
              "常驻页签「实时」上还挂着「×」关闭按钮（用户要求不需要它 ✗）")
    finally:
        w.close()


def t_clear_pages_keeps_work_tabs():
    """换项目清详情页：**只清步骤详情页**，四个常驻工作页和起始页都留着 ✓。"""
    from gui.main_window import InfoPage

    app, w = _win()
    try:
        w.open_view(w.live_panel, "实时")
        w.open_view(w.review, "质检台")
        page = InfoPage("② 抽帧", "")
        w.view_widgets["frames"] = page
        w.open_view(page, w._view_title(type("C", (), {"num": 2, "title": "抽帧"})()))

        w._clear_pages()
        # ⚠⚠ **这里绝不能再碰 `page`**（踩过：整条用例段错误 `0xC0000005` ✗）：`_clear_pages()`
        #   里对它调了 `deleteLater()`，而 Qt 的 `deleteLater` 是"**事件循环里真删**"（不是
        #   立即）—— 同一个事件循环里再拿它去 `viewer.indexOf(page)` = **访问已销毁的 C++
        #   对象** ⇒ 段错误 ✗。⚠ 阴的地方是：Python 侧那个变量**还在、还能打印**，看着完全
        #   像"对象还活着"✗ ⇒ 这类崩溃最难查 ✓。⇒ 只查"它**被摘掉了**"这件事就够 ✓：
        check(not w.view_widgets,
              "换项目后步骤详情页没清掉（`view_widgets` 该空）：%r" % (w.view_widgets,))
        _tabs = [w.viewer.tabText(i) for i in range(w.viewer.count())]
        check(all("抽帧" not in str(t) for t in _tabs),
              "那个步骤详情页的页签还留在主视区（该摘掉 ✗）：%s" % _tabs)
        check(w.viewer.indexOf(w.home_view) >= 0
              and w.viewer.currentWidget() is w.home_view,
              "清完之后「起始」页不在 / 没切回去：%s" % _tabs)
        # ⚠⚠ 收尾必须把 Qt 的**延迟删除队列跑干净**（踩过：整条用例段错误 `0xC0000005` ✗）：
        #   `_clear_pages()` 里对那些步骤详情页调的是 `deleteLater()` —— 它是"**事件循环里
        #   才真删**"，而**离屏自检几乎不跑事件循环** ⇒ 这些控件一直挂在删除队列里 ⇒ 等到
        #   **解释器退出**（那时 `QApplication` 已经在拆）才轮到它们 ⇒ 段错误 ✗。
        #   ⚠ **别在这一条用例里手动 `processEvents()`**（2026-09-28 又踩一次 ✗）：那次
        #     实测把它加在用例内 ⇒ `selftest_main_window` **2/4 崩**；而把它移出用例、
        #     统一放到 `main()` 的"每条之后"（见下面那段 ✓）⇒ 就稳了 ✓。
        #     原因：在**用例中途**跑事件循环，会顺带把**别人**挂起的事件（定时器 / 队列里的
        #     `deleteLater`）一起处理掉 ⇒ 那些对象本来是"跑完这条再清"的 ✗。
        #   ⇒ 收尾统一交给 `main()` ✓（一处管全部 ✓ 比散在各用例里可控 ✓）。
        for wdg, name in ((w.home_view, "起始"), (w.live_panel, "实时"),
                          (w.review, "质检台")):
            check(w.viewer.indexOf(wdg) >= 0,
                  "换项目把常驻页「%s」也关掉了（正看着它的人会被抽走 ✗）" % name)
        check(w.viewer.currentWidget() is w.home_view, "清完之后没回到「起始」页")
    finally:
        w.close()


def t_viewer_has_single_entry():
    """**源码钉子**：主视区的开合只许走 `open_view` / `close_view`（别的写法一律算旁路 ✗）。

    做法：把那四个函数体从源码里剪掉，剩下的部分**不许**再出现
    `viewer.setCurrentIndex` / `viewer.addTab` / `viewer.removeTab`，
    也不许再有"固定页号"那套东西（`PAGE_XXX` / `viewer_pages`）。
    """
    import re

    src = (ROOT / "gui" / "main_window.py").read_text(encoding="utf-8")

    # ⚠ 按 **4 空格缩进的 `def`** 逐个方法切，别用"从 A 剪到下一个 def B"那种写法：
    #   那个写法跨过的是**整段中间区域**（第一次写就踩了：`_build_viewer` 一路剪到
    #   `open_view`，把中间的 `_show_live` 一起吞掉 ⇒ 里面写的旁路测不出来 ✗✗）。
    parts = re.split(r"^    def (\w+)\(", src, flags=re.M)
    names, bodies = parts[1::2], parts[2::2]
    #: 只有这几个方法**可以**碰主视区的页签 API（骨架 / 开 / 关 / × 槽 ✓）
    allowed = {"_build_viewer", "open_view", "close_view", "_on_view_close"}
    for name, body in zip(names, bodies):
        if name in allowed:
            continue
        for bad in ("viewer.setCurrentIndex", "viewer.addTab", "viewer.removeTab",
                    "viewer.addWidget", "self.viewer_pages"):
            check(bad not in body,
                  "%s() 里直接动了主视区的页签（%s）—— 占用主视区一律走 "
                  "`open_view` / `close_view` ✗" % (name, bad))

    # ⚠ 固定页号常量只找**定义**（`^PAGE_X = …`）：说明文字里就写着"那四个常量删掉了"，
    #   拿裸名字找会把**解释**一起判红（这个形状踩过好几次 ✗）
    left = re.findall(r"^PAGE_[A-Z_]+ *=", src, re.M)
    check(not left,
          "「固定页号」那套东西又回来了（%s）—— 页号会随开合变化，拿它定位迟早错位 ✗"
          % left)

    check("setTabsClosable(True)" in src,
          "主视区不是可关的页签（用户要的「关闭流程」没了 ✗）")


def _redirect_ui_store():
    """⛔ 自检**绝不许**改用户的 `config/ui.yaml`（全仓库规矩 ✓）—— 把 theme 的落点指到临时文件。

    ⚠ **这一句在别的套件里是"顺手加的"，在这里是"必须加的"**（2026-09-29 量出来的）：
    本套件里 `hide()`/`show()` 那套恢复是**延后一拍**做的（`QTimer.singleShot(0, …)` ✓），
    而 `_fake_store()` 的 `with patcher:` 早就退出了 ⇒ 延后那一拍**写的是真文件** ✗ ——
    实测本套件会往用户的 `ui.yaml` 里留下 `windows.t_bind` / `views.t_view2`
    （那两个键就是这么来的：它们本来只在用例的假 store 里 ✓）。
    `_fake_store` 继续留着（它管"用例自己那几次存取走内存" ✓），两者不冲突 ✓。
    """
    import tempfile

    from gui import theme

    theme.CFG = Path(tempfile.mkdtemp(prefix="psimu_ui_")) / "ui.yaml"


_redirect_ui_store()


def _fake_store():
    """把 `theme._load/_save` 换成内存里的字典 ⇒ **一个字节都不写真的 ui.yaml** ✓。

    （自检不许改用户文件 —— 这是全仓库的规矩，见其它套件里 mock `update_live` 那些 ✓）
    """
    import copy
    import unittest.mock as mock

    from gui import theme

    store = {}

    def _load():
        return copy.deepcopy(store)

    def _save(d):
        store.clear()
        store.update(copy.deepcopy(d))

    return theme, store, mock.patch.multiple(theme, _load=_load, _save=_save)


def t_window_state_roundtrip():
    """弹窗几何的存取（**按客户端一份**）：能存能取，坏值一律当"没存过"。

    坏值必须当"没存过"的理由：`config/ui.yaml` 是**手也能改**的文件 —— 少写一个键、
    把宽高写成 10 ⇒ 窗口会缩成一个点，人只会以为程序坏了 ✗（用例把它钉住）。
    """
    theme, store, patcher = _fake_store()
    with patcher:
        check(theme.load_window("没存过") is None, "没存过的键该给 None")
        theme.save_window("t_round", (100, 120, 900, 700))
        check(theme.load_window("t_round") == (100, 120, 900, 700),
              "存进去的几何没取回来：%r" % (theme.load_window("t_round"),))
        check(store.get("windows", {}).get("t_round"),
              "没写进 config/ui.yaml 的 windows: 段（那是「按客户端一份」的落点 ✗）")
        # 坏值：缺键 / 类型不对 / 太小
        store["windows"]["b1"] = {"x": 0, "y": 0}
        store["windows"]["b2"] = {"x": "啊", "y": 0, "w": 900, "h": 700}
        store["windows"]["b3"] = {"x": 0, "y": 0, "w": 20, "h": 20}
        for k, why in (("b1", "缺键"), ("b2", "类型不对"), ("b3", "太小")):
            check(theme.load_window(k) is None,
                  "%s 的几何该当「没存过」（否则窗口会缩成一个点 ✗）：%r"
                  % (why, theme.load_window(k)))


def t_window_state_save_restore():
    """装上之后：**hide 就存、show 就恢复**；**屏幕外的坐标夹回屏幕内**（拔了外接屏之后）。

    最后一条是这条需求最容易写错、后果又最难受的地方：坐标悬空时照样 `setGeometry`，
    窗口会开在看不见的地方（**标题栏在屏幕外、抓不到也拖不动** ✗ —— 用户 2026-09-27 报的
    就是这个 ✓）⇒ 现在恢复时把矩形**夹进那块屏的可用区**（尺寸超过屏幕才缩 ✓）。
    """
    from PyQt5.QtWidgets import QApplication, QDialog

    theme, store, patcher = _fake_store()
    app = QApplication.instance() or QApplication([])
    d = QDialog()
    d.resize(600, 500)
    try:
        with patcher:
            theme.bind_window_state(d, "t_bind")
            d.resize(880, 660)
            d.show()
            app.processEvents()
            d.hide()                       # ⇒ 该存下来
            got = theme.load_window("t_bind")
            check(got is not None and got[2] == 880 and got[3] == 660,
                  "hide 之后没记住大小：%r" % (got,))

            # 屏幕外（拔了外接屏 / 换分辨率）⇒ **夹回屏幕内**（用户 2026-09-27 报：
            # "每个弹窗打开时，其顶部的栏在**屏幕外面**，我都无法拖动"✗
            # ⇒ 现在保证标题栏一定露在屏幕里 ✓，尺寸尽量保留 ✓）
            store["windows"]["t_bind"] = {"x": -50000, "y": -50000,
                                          "w": 880, "h": 660}
            d.resize(600, 500)
            d.show()
            app.processEvents()
            scr = QApplication.primaryScreen()
            check(scr is not None, "用例前提：得有一块屏")
            av = scr.availableGeometry()
            check(av.contains(d.geometry().topLeft()),
                  "屏幕外的旧坐标没被夹回屏幕里（标题栏会落在屏外、抓不到 ✗）：%s ／ 屏 %s"
                  % (d.geometry(), av))
            check(d.width() == 880 and d.height() == 660,
                  "夹回屏幕时把尺寸也改了（**只夹位置、尺寸不动** ✓）：%dx%d"
                  % (d.width(), d.height()))

            # 屏幕内 ⇒ 恢复
            # ⚠ 顺序要紧：`hide()` 自己会**存一次**（那是它的职责 ✓）⇒ 注入的值必须写在
            #   **hide 之后**，否则当场被覆盖（第一版就是这么写错的：测出来 600×500 ✗）
            d.hide()
            store["windows"]["t_bind"] = {"x": 40, "y": 40, "w": 820, "h": 640}
            d.show()
            app.processEvents()
            check(d.width() == 820 and d.height() == 640,
                  "屏幕内的旧尺寸没恢复（尺寸原样 ✓）：%dx%d" % (d.width(), d.height()))
    finally:
        d.close()


def t_view_zoom_state():
    """画布的**缩放**也要记住（用户 2026-09-27 报，原话："我每次打开弹窗都要重新调缩放，
    之前的存缩放数据功能没做？"）。

    ⚠ 和"窗口几何"是**两件事**：那边记的是窗口矩形（`bind_window_state` ✓，**早就做了** ✓），
    这边记的是**画布里的 zoom**（`QGraphicsView` 的 transform ✓，以前**根本没存** ✗ ——
    画布每次新建 / 换图都会 `fit()` 适应窗口 ⇒ 不记就等于回默认 ✗）。

    钉四件：
      ① 存取往返 ✓；坏值（缺键 / 非数 / 超范围）一律当"没存过" ✓；
      ② 真画布：`hide` 记下来、`show` 按回去 ✓（恢复是**延后一拍**做的 ⇒ 用例得 `processEvents` ✓）；
      ③ 没存过 ⇒ **什么也不做**（不许猜一个倍数 ✗）；
      ④ 恢复发生在画布自己的 `fit()` **之后**（否则会被 `fit` 盖掉 ✗ —— 这就是为什么要
         `QTimer.singleShot(0, …)` ✓）。
    """
    from PyQt5.QtWidgets import QApplication

    from gui.canvas import ImageCanvas

    theme, store, patcher = _fake_store()
    store.setdefault("views", {})
    app = QApplication.instance() or QApplication([])
    v = ImageCanvas()
    try:
        with patcher:
            # ① 存取往返 + 坏值
            theme.save_view_zoom("t_view", 2.5)
            check(abs(float(theme.load_view_zoom("t_view")) - 2.5) < 1e-6,
                  "存了却读不回来：%r" % (theme.load_view_zoom("t_view"),))
            for bad in ({"scale": "x"}, {"scale": 0.0}, {"scale": 999.0}, {}):
                store["views"]["t_view"] = bad
                check(theme.load_view_zoom("t_view") is None,
                      "坏值没当「没存过」：%r" % (bad,))
            store["views"].pop("t_view", None)
            check(theme.load_view_zoom("t_view") is None,
                  "没存过却给了一个值（不许猜 ✗）")

            # ③ 没存过 ⇒ 不动缩放
            theme.bind_view_zoom(v, "t_view2")
            v.show()
            app.processEvents()
            check(abs(v.transform().m11() - 1.0) < 1e-6,
                  "没存过却动了缩放（不猜 ✓）：%r" % (v.transform().m11(),))

            # ② hide 存 / show 恢复（延后一拍 ⇒ processEvents 让它跑完 ✓）
            v.scale(2.0, 2.0)
            v.hide()
            app.processEvents()
            _saved = theme.load_view_zoom("t_view2")
            check(_saved is not None and abs(float(_saved) - 2.0) < 1e-3,
                  "hide 之后没记住画布缩放：%r" % (_saved,))
            v.resetTransform()
            v.show()
            app.processEvents()
            check(abs(v.transform().m11() - 2.0) < 1e-3,
                  "show 之后没把缩放按回去（用户「每次都要重调」的就是这件事 ✗）：%r"
                  % (v.transform().m11(),))
    finally:
        v.close()


def t_window_state_covers_editors():
    """**规范条**（用户 2026-09-27 要求进规范）：能拉大拉小的弹窗都要接上，键名唯一。

    钉三件：
      ① 每个弹窗的源码里都有 `theme.bind_window_state(self, "<键>")`，且**键名全局唯一**
         （键重了 ⇒ 两个窗口互相覆盖大小 ✗）；
      ② 这些文件都从 `gui` 引了 theme（不许在自己文件里另写一份存取 ✗）；
      ③ 那句必须落在**它自己那个类的 `__init__`** 里、**每个窗一次**（插进别的方法
         = 永远不生效 ✗）。
    漏一个的后果是"这个窗口每次都要重新拉"——不报错、不崩，所以只能靠这条钉子 ✓。

    ⚠ **一个文件可以有多个弹窗**（2026-09-28 补 `player_panel`：它里面**两个**「战斗区域」
      弹窗都要接 ✓）。⚠ 补之前这里**没列它** ⇒ 规范条把它俩整个漏掉了 —— 于是那两个窗
      既没接几何、也没人发现 ✗（用户当天正好报了其中一个："**编辑战斗区域子窗口也太大了，
      还不让缩小**"✓）。⇒ 教训：**规范条本身就是检查清单，漏列 = 没这根钉子** ✗。
    """
    import re

    #: 文件 → ((几何键, 那个弹窗的类名), …)
    plan = {"zone_editor": (("zone_editor", "ZoneEditorDialog"),
                            ("add_reach", "AddReachDialog")),
            "seq_editor": (("seq_editor", "SeqEditorDialog"),
                           ("timer_edit", "TimerEditDialog")),
            "minimap_calib": (("minimap_calib", "MinimapCalibDialog"),),
            "two_point_calib": (("two_point_calib", "TwoPointCalibDialog"),),
            "calib_manual": (("calib_manual", "CalibManualDialog"),),
            "settings_dialog": (("settings", "SettingsDialog"),),
            "train_report": (("train_report", "TrainReportDialog"),),
            "export_dialog": (("export", "ExportDialog"),),
            "frame_picker": (("frame_picker", "FramePickDialog"),),
            "mob_picker": (("mob_picker", "MobPickDialog"),),
            "player_panel": (("battle_zone", "BattleZoneDialog"),
                             ("battle_zone_list", "BattleZoneListDialog")),
            # ⚠ 2026-10-04 补：`live_panel` 一直**没列**（同 player_panel 那次的教训 ——
            #   规范条本身就是检查清单，漏列 = 没这根钉子 ✗）。它里面那个「导出引擎」进度窗
            #   是能拉大拉小的（见 `EngineExportDialog` ✓）⇒ 必须接几何 ✓。
            "live_panel": (("engine_export", "EngineExportDialog"),),
            # ⚠ 2026-10-05 补：新增的「地区选择」通用弹窗（`gui/element_picker.py` ✓
            #   —— 「站桩地点」/「拾取掉落地区」都用它 ✓）也必须接上几何 ✓。
            "element_picker": (("element_picker", "ElementPickerDialog"),)}
    seen = []
    for f, pairs in plan.items():
        src = (ROOT / "gui" / ("%s.py" % f)).read_text(encoding="utf-8")
        check(re.search(r"^from gui import .*\btheme\b|^from gui import theme",
                        src, re.M),
              "gui/%s.py 没从 gui 引 theme（不许自己再写一份存取 ✗）" % f)
        for key, cls in pairs:
            call = 'bind_window_state(self, "%s")' % key
            check(call in src,
                  "gui/%s.py 没接上「记住窗口几何」（用户要求所有编辑器弹窗都接 ✓）：%s"
                  % (f, call))
            # ⚠ 必须在**它自己那个类**的 `__init__` 里：插进别的方法 = 永远不生效，
            #   而它不报错、不崩 —— 只能靠这条钉子 ✓
            m = re.search(r"\nclass %s\b.*?\n(    def __init__\(self.*?)(?=\n    def |\Z)"
                          % cls, src, re.S)
            check(m is not None and "bind_window_state(" in m.group(1),
                  "gui/%s.py 里那句不在 %s.__init__ 里（插进别的方法 = 永远不生效 ✗）"
                  % (f, cls))
            check(m.group(1).count("bind_window_state(") == 1,
                  "gui/%s.py 的 %s.__init__ 里 `bind_window_state` 出现了 %d 次"
                  "（每个窗一次就够 ✗）"
                  % (f, cls, m.group(1).count("bind_window_state(")))
            seen.append(key)
        # 全文件总数 = 本文件弹窗数（多一处/少一处都说明接错了地方 ✗）
        check(src.count("bind_window_state(") == len(pairs),
              "gui/%s.py 里 `bind_window_state` 出现了 %d 次（本文件有 %d 个弹窗 ⇒ 该 %d 次 ✗）"
              % (f, src.count("bind_window_state("), len(pairs), len(pairs)))
    check(len(seen) == len(set(seen)), "窗口几何的键名重了：%s" % seen)


def t_startup_opens_last_project_and_says_where_params_go():
    """启动**自动打开"上次那个项目"** + 参数页顶部那行"参数存到哪儿"（用户 2026-10-04 ✓ 现场：
    "决策参数页签→战斗参数→追击起跳需要的冲刺时间，没有保存数据，每次重开 gui 都要重新填"）。

    为什么要盯**这两头**（它们是一条链 ✓）：
      · 决策参数**只按项目存** ✓ ⇒ **没打开项目时 `settings.save()` 一个字节都不写** ✗
        （见 `decision/agent.py` 的 `set_save_hook` ✓）⇒ 人在"还没开项目"时改的参数，
        一重开就回默认 ✓ —— 这就是用户那句的原形 ✓。光看"参数自己存/读对不对"是看不出来的 ✗
        （`chase_jump_dash_ms` 在 `to_dict`/`from_dict` 里都有 ✓ 实测往返也对 ✓）⇒
        必须**启动就把上次那个项目开上** ✓ 改动才有归属 ✓；
      · 而"现在这份存到哪儿"必须**看得见** ✗ 否则人只会得出"没保存"这个结论 ✓。
    """
    import atexit
    import shutil
    import tempfile
    from pathlib import Path as _P

    from PyQt5.QtWidgets import QApplication

    from gui import main_window as _mw
    from gui.main_window import MainWindow
    from gui.project import Project

    app = QApplication.instance() or QApplication([])
    tmp = _P(tempfile.mkdtemp(prefix="startup_proj_"))
    # ⚠⚠ **别在用例里当场删临时项目** ✗（2026-10-04 踩过 ✓）：这条用例会把窗口连同
    #   「当前项目」一起留着（其它用例也这样 ✓ 它们不 `close()` ✓），而目录一删，
    #   后面的用例里只要有人再碰那个项目 ⇒ **`0xC0000005`** ✗（症状极迷惑：报的是**下一条**
    #   用例崩 ✓）。⇒ 交给 `atexit` 在**进程退出**时删 ✓。
    atexit.register(shutil.rmtree, tmp, True)
    orig = _mw.last_opened
    try:
        # ⚠ 目录名跟项目名**一致** ✓：那行提示显示的是 `project.root.name`（目录名 ✓）
        proj = Project.create(tmp / "用例项目", name="用例项目", map_id="105090600")
        # ⭐ 这条参数的**非默认值**（默认 0 ✓）—— 用例就靠它认出"到底有没有回填" ✓
        # ⚠⚠ 必须 `save=True` ✗：下面 `last_opened` 是 `Project.open(...)` ⇒ **从磁盘重读** ✓
        #   （只写内存的话，主窗口读到的是一份**没有 decision 段**的项目 ⇒ 参数落到默认 0 ✗
        #     —— 本轮就这么红过一次 ✓）
        proj.set("decision", {"chase_jump_enabled": True, "chase_jump_min": -10,
                              "chase_jump_max": 5, "chase_jump_dash_ms": 137}, save=True)
        _mw.last_opened = lambda: Project.open(proj.root)

        # ① 启动就该把它打开 ✓（不打桩的话这里会是 None ✗ = 用户看到的"没保存"那套 ✓）
        w = MainWindow()
        try:
            check(w.project is not None and w.project.root == proj.root,
                  "启动没自动打开「上次那个项目」⇒ 改参数没有归属 ⇒ 重开就丢 ✗")

            # ② 打开之后，那条参数**真的回填进控件**（用户看的是控件 ✓）
            sp = w.player_panel.sp_chase_dash
            check(int(sp.value()) == 137,
                  "启动打开项目后，冲刺时间控件没回填（该 137）：%r" % sp.value())

            # ③ 顶部那行：有项目 ⇒ 说清"按项目保存 + 是哪个项目"✓
            lbl = w.player_panel.lbl_save_hint
            txt = lbl.text()
            check("用例项目" in txt and "保存" in txt,
                  "有项目时没在参数页说明「参数按项目保存」：%r" % txt)

            # ④ 没有项目时 ⇒ 必须**明说"不会保存"** ✗（不然用户就是会以为存了 ✓）
            w.project = None
            w.player_panel.bind(None)
            txt2 = w.player_panel.lbl_save_hint.text()
            check("不会保存" in txt2,
                  "没打开项目时没提示「改了不会保存」⇒ 人以为存了 ✗：%r" % txt2)
        finally:
            # ⚠⚠ 收尾要**一次做干净**（2026-10-04 踩过两轮 ✓）：
            #   ① 窗口**不 `close()`** ✗（那会顺手拆一堆东西 ⇒ 报的是**下一条**用例崩 ✓）；
            #   ② 「当前项目」放回 None ✓（别留下指向"待删目录"的引用 ✗）；
            #   ③ 然后**显式 `deleteLater()` + 跑一次事件循环** ✓ —— 照本套件 `main()` 里
            #      那条纪律（`deleteLater` 是"事件循环里才真删" ⇒ 攒到解释器退出一起还账
            #      就是 `0xC0000005` ✗）。
            w.project = None
            w.player_panel.bind(None)
            w.deleteLater()
            app.processEvents()
    finally:
        _mw.last_opened = orig


def t_offscreen_never_connects_device():
    """⭐⭐ 离屏 ⇒ **绝不连被控机**：闸在真源、连线程都不起（用户 2026-10-05 ✓ 治根）。

    治的就是那条老病（`0xC0000005`）：`PlayerPanel._apply_input_device("remote")` 会起一条
    **后台线程**去 TCP 连被控机，而它**几秒后**才超时返回 —— 那时用例早跑完、Qt 对象已销毁
    ⇒ 线程回来碰它 ⇒ **偶发原生段错误** ✗（时机性 ⇒ "多跑几次"不算验证 ✗）。
    原来只在**本套件**用例里打桩 ✗ ⇒ `gui_smoke` / `minimap` / `screen_state` / `decision`
    里那几个 `PlayerPanel()` 照样连 ✗ ⇒ 现在闸放在 `decision/input.py::net_allowed`
    （唯一真源 ✓）＋ 线程的唯一出生地 `connect_async` ✓。

    钉四件：
      ① 判据三条：offscreen ⇒ 不许连；显式 `PSIMU_ALLOW_NET=1` ⇒ 允许（`selftest_link`
         那种"就是要连"的 ✓）；真平台 ⇒ 允许 ✓；
      ② 运行时：闸关着时 `use_network` / `use_serial` **一个 socket 都不建**、
         `reconnect_remote()` 如实回 `False`，且转「什么都不发」（`_blocked` ✓ ——
         不许退化成把按键打到控制机上 ✗）；
      ③ **面板真调一次**：`_apply_input_device("remote")` ⇒ 线程数不变 + 那行说清「已跳过」
         + **不写** `settings.input_device`（别把"跳过"说成"连接失败"、也别写脏配置 ✗）；
      ④ 源码钉：`gui/` 里**没有人**直接调 `use_network` / `use_serial`（出生地只有
         `input.connect_async` ✓）、闸判在 `start()` **之前** ✓、三个入口都有闸 ✓。
    """
    import os
    import re
    import shutil
    import socket
    import subprocess
    import tempfile
    import unittest.mock as mock

    from decision import input as dinput

    _saved_allow = os.environ.pop("PSIMU_ALLOW_NET", None)
    try:
        # ① 判据三条（**每次都读环境** ⇒ 当场摆得动 ✓）
        with mock.patch.dict(os.environ, {"QT_QPA_PLATFORM": "offscreen"}):
            check(dinput.net_allowed() is False,
                  "offscreen 下闸没关（会去连被控机 ⇒ 段错误 0xC0000005 ✗）")
            os.environ["PSIMU_ALLOW_NET"] = "1"
            check(dinput.net_allowed() is True,
                  "显式 `PSIMU_ALLOW_NET=1` 没放开（`tools/selftest_link.py` 那种要连的会连不上 ✗）")
            del os.environ["PSIMU_ALLOW_NET"]
            os.environ["QT_QPA_PLATFORM"] = "windows"
            check(dinput.net_allowed() is True, "真平台（windows）下也不许连设备了 ✗")
            os.environ["QT_QPA_PLATFORM"] = "offscreen"

            # ② 运行时：闸关着 ⇒ 一个 socket 都不建（真建了这条 canary 会炸 ⇒ 用例红 ✓）
            with mock.patch.object(socket, "create_connection",
                                   side_effect=AssertionError("闸关着还去建连接 ✗")):
                dinput.use_network("127.0.0.1", 9, "不存在的证书.pem")
                dinput.use_serial("COM_不存在")
                check(dinput.reconnect_remote() is False,
                      "闸关着重连却回了 True（对上层说谎：通道其实不可用 ✗）")
            _h = dinput.link_health()
            check(_h.get("backend") == "blocked",
                  "闸关着却没进「什么都不发」（会退化成把按键打到控制机上 ✗）：%r" % _h)

            # ③ 面板**真调一次** —— 放**子进程**里做（本仓库既有手法 ✓ 见
            #   `selftest_live_panel` 那条"接线改由子进程验" ✓）：
            #   ⚠⚠ 本进程里**一个窗口都不建**（2026-10-05 实测：本套件里多建一个窗口，
            #     后面的 `t_clear_pages_keeps_work_tabs` 就**必崩 `0xC0000005`** ✗）；
            #     而且离屏**裸建 `PlayerPanel()`** 本身就会原生崩（`0xC0000409` ✗）
            #     ⇒ 子进程里用 `MainWindow().player_panel`（实测稳 ✓），收尾用 `os._exit`
            #     跳过 Qt 析构 ✓。
            _body = (
                "import os, sys, threading, tempfile\n"
                "sys.stdout.reconfigure(encoding='utf-8')\n"
                "os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')\n"
                "sys.path.insert(0, %r)\n"
                "from pathlib import Path\n"
                "from gui import theme, main_window as _mw\n"
                "theme.CFG = Path(tempfile.mkdtemp(prefix='probe_ui_')) / 'ui.yaml'\n"
                "_mw.last_opened = lambda: None\n"          # 别去开用户的真项目 ✓
                "from decision import input as dinput\n"
                "from PyQt5.QtWidgets import QApplication\n"
                "app = QApplication.instance() or QApplication([])\n"
                "print('ALLOWED=%%d' %% int(dinput.net_allowed()))\n"
                "from gui.main_window import MainWindow\n"
                "from gui.player_panel import settings as _ps\n"
                "w = MainWindow()\n"
                "dev0 = _ps.input_device\n"
                "n0 = threading.active_count()\n"
                "w.player_panel._apply_input_device('remote')\n"
                "print('THREADS=%%d,%%d' %% (n0, threading.active_count()))\n"
                "print('LABEL=%%s' %% w.player_panel.lbl_device_state.text())\n"
                "print('DEV=%%s,%%s' %% (dev0, _ps.input_device))\n"
                "os._exit(0)\n"
            ) % str(ROOT)
            _dir = Path(tempfile.mkdtemp(prefix="offscreen_net_"))
            _pf = _dir / "probe.py"
            _pf.write_text(_body, encoding="utf-8")
            try:
                _env = dict(os.environ)
                _env["QT_QPA_PLATFORM"] = "offscreen"
                _env.pop("PSIMU_ALLOW_NET", None)
                _r = subprocess.run([sys.executable, "-X", "utf8", str(_pf)],
                                    cwd=str(ROOT), capture_output=True, text=True,
                                    timeout=300, env=_env)
                _out = (_r.stdout or "") + (_r.stderr or "")
                check(_r.returncode == 0 and "THREADS=" in _out,
                      "子进程里「离屏切设备」没跑成（rc=%s）：\n%s"
                      % (_r.returncode, _out.strip()[-600:]))
                _m = re.search(r"ALLOWED=(\d)", _out)
                check(_m is not None and _m.group(1) == "0",
                      "子进程里闸没关：%s" % _out.strip()[-300:])
                _m = re.search(r"THREADS=(\d+),(\d+)", _out)
                check(_m is not None and _m.group(1) == _m.group(2),
                      "离屏下 `_apply_input_device('remote')` 起了后台线程"
                      "（正是段错误 0xC0000005 的真凶 ✗）：%s" % _out.strip()[-300:])
                check("LABEL=" in _out and "跳过" in _out,
                      "离屏跳过没在界面上说清（人只会以为「连不上」✗）：%s"
                      % _out.strip()[-300:])
                _m = re.search(r"DEV=(\S+),(\S+)", _out)
                check(_m is not None and _m.group(1) == _m.group(2),
                      "离屏跳过却改了 `settings.input_device`"
                      "（该当「跳过」而不是「失败」✗）：%s" % _out.strip()[-300:])
            finally:
                shutil.rmtree(_dir, ignore_errors=True)

        # ④ 源码钉：**出生地唯一** + 闸在起线程之前 + 三个入口都有闸
        _bad = [f.name for f in sorted((ROOT / "gui").glob("*.py"))
                if re.search(r"\b(use_network|use_serial)\s*\(", f.read_text(encoding="utf-8"))]
        check(not _bad,
              "gui/ 里还有文件直接调 use_network / use_serial"
              "（该走 `input.connect_async` = 唯一出生地 ✗）：%s" % _bad)

        _src = (ROOT / "decision" / "input.py").read_text(encoding="utf-8")

        def _body(fn):
            i = _src.index("def %s(" % fn)
            j = _src.find("\ndef ", i + 1)
            return _src[i:j if j > 0 else len(_src)]

        _ca = _body("connect_async")
        check("net_allowed()" in _ca, "`connect_async` 里没有闸（离屏会照样起那条线程 ✗）")
        check(_ca.index("net_allowed()") < _ca.index(".start()"),
              "`connect_async` 里闸判在起线程**之后**了（线程照样会活着 ⇒ 病还在 ✗）")
        for _fn in ("use_network", "use_serial", "reconnect_remote"):
            check("net_allowed()" in _body(_fn),
                  "`%s` 里没有闸（直接调它的人绕过去了 ✗）" % _fn)
    finally:
        if _saved_allow is not None:
            os.environ["PSIMU_ALLOW_NET"] = _saved_allow


def t_watchdog_guarantees_exit():
    """⭐⭐ 「关了必须走 / 卡住不许烧核」：`gui/app.py` 的两道看门狗（用户 2026-10-05 ✓
    原话："保证以后不要关了该占进程就行"）。

    **为什么要它**（现场实测 ✓）：有 **4 个工作台进程卡在启动里、各烧满一个核 25.7 小时** ✗
    （`py-spy` 抓栈 = `theme._row_title_of` 那个已修的死循环 ✓：窗口从没出来、进程活着、
    没有任何异常 ⇒ 用户看到的是"打不开 + 机器变卡"✓）。这类 bug **不报错、只是卡** ✗
    ⇒ 只能靠"到点自己留栈 + 走人"兜底 ✓。

    钉四件（都在**子进程**里验 ✓ —— 它会 `os._exit`，不能在本进程里跑 ✗）：
      ① 卡住 ⇒ 退出码 **3**（启动那段）+ `crash.log` 里有「看门狗触发」+ **一份栈** ✓
         （栈就是下次定位用的证据 ✓ 这次那 4 个僵尸就是靠它认出来的 ✓）；
      ② **正常路径绝不被误杀**：立刻 `done.set()` ⇒ 退出码 0 ✓ 且没有「看门狗触发」✓；
      ③ 源码钉：`main()` 里启动那道是**在 `MainWindow()` 之前** arm ✓（那段才死过 ✓）、
         收尾那道是**在 `app.exec_()` 之后** arm ✓、两处 `done.set()` 都在 ✓；
      ④ 常量齐（`WATCHDOG_STARTUP_S` / `WATCHDOG_EXIT_S`）且退出码 3/4 分开 ✓。
    """
    import shutil
    import subprocess
    import tempfile

    _src = (ROOT / "gui" / "app.py").read_text(encoding="utf-8")
    check("WATCHDOG_STARTUP_S" in _src and "WATCHDOG_EXIT_S" in _src,
          "看门狗的时长常量没了 ✗")
    _main = _src[_src.index("def main():"):]
    _i_boot = _main.find('_arm_watchdog(WATCHDOG_STARTUP_S, "启动", _boot_done, 3)')
    _i_win = _main.find("win = MainWindow()")
    _i_exec = _main.find("rc = app.exec_()")
    _i_exit = _main.find('_arm_watchdog(WATCHDOG_EXIT_S, "收尾", _exit_done, 4)')
    check(0 <= _i_boot < _i_win, "启动看门狗没 arm 在 `MainWindow()` **之前**"
          "（那段正是卡死过的那一段 ✗）")
    check(_i_exec < _i_exit, "收尾看门狗没 arm 在 `app.exec_()` **之后** ✗")
    check("_boot_done.set()" in _main and "_exit_done.set()" in _main,
          "正常路径没有收工信号 ⇒ 会把好进程也杀掉 ✗")
    check("os._exit(int(code))" in _src and "dump_traceback" in _src,
          "看门狗该「先留栈、再强退」（没栈 ⇒ 下次还是查不出来 ✗）")

    _dir = Path(tempfile.mkdtemp(prefix="watchdog_"))
    _child = _dir / "child.py"
    _log = _dir / "crash.log"
    _child.write_text(
        "import sys, time, threading\n"
        "sys.path.insert(0, %r)\n"
        "from pathlib import Path\n"
        "import gui.app as app\n"
        "app.CRASH_LOG = Path(%r)          # ⚠ 别写用户的 crash.log ✗\n"
        "done = threading.Event()\n"
        "app._arm_watchdog(0.5, '启动', done, 3)\n"
        "if sys.argv[1] == 'ok':\n"
        "    done.set()\n"
        "    print('NORMAL', flush=True)\n"
        "else:\n"
        "    time.sleep(8)                 # 模拟「卡住」（真实那次是死循环 ✓）\n"
        % (str(ROOT), str(_log)), encoding="utf-8")
    try:
        for mode, want_rc, want_log in (("ok", 0, False), ("hang", 3, True)):
            r = subprocess.run([sys.executable, "-X", "utf8", str(_child), mode],
                               cwd=str(ROOT), capture_output=True, text=True,
                               timeout=90)
            txt = _log.read_text(encoding="utf-8") if _log.exists() else ""
            check(r.returncode == want_rc,
                  "%s 的退出码该是 %d，实得 %s（stderr=%s）"
                  % (mode, want_rc, r.returncode, (r.stderr or "")[-200:]))
            check(("看门狗触发" in txt) == want_log,
                  "%s：crash.log 里「看门狗触发」对不对（卡住必须留痕 ✗）：\n%s"
                  % (mode, txt[-300:]))
            if want_log:
                # ⚠ 判据是**子进程脚本名**出现在栈里（faulthandler 打的是"文件 + 行号 + 函数名"，
                #   没有 `sleep` 这种字面 ✓）：有它才叫"留下了卡在哪一行的证据" ✓
                check("强制退出" in txt and "child.py" in txt,
                      "没留下「卡在哪」的栈（下次照样查不出来 ✗）：\n%s" % txt[-300:])
            _log.unlink(missing_ok=True)
    finally:
        shutil.rmtree(_dir, ignore_errors=True)


def t_run_export_uses_workbench_bar():
    """⭐⭐ **卡片之外的小任务也走工作台那条读条**（用户 2026-10-05 ✓ 原话："如果没有导出，
    在添加的带装备的宠物、关闭弹窗后，在数据集工作台显式读条导出"）。

    为什么非要有这一条（只钉源码不够 ✓）：`run_export` 是给**卡片之外**用的入口 ✓ ——
      `_start` 原来句句都当"有卡片"（`card.set_state` / `CIRCLED[card.num - 1]` ✗）
      ⇒ 传 None 当场炸 ✓；还要验"读条真变忙式"、"日志真写进工作台"、"结算真回调发起方"
      （卡片靠它重刷宠物列表 ✓）—— 这几件只有**真起一次任务**才看得到 ✓。

    钉五件：① 起得来、且读条**立刻**变忙式（不许还显示"空闲"✗）；② 标题进了工作台日志；
      ③ 任务里的日志真回到工作台；④ 结算时 `on_done(ok, summary)` 调**一次**、summary 是任务
      回的那份 ✓；⑤ **正忙时不许再起**（回 False ✓ 调用方据此如实提示 ✓ 不许静默 ✗）。
    """
    import time

    app, w = _win()
    seen = []

    def _fn(params, ctx):
        ctx.log("小任务：%s" % params.get("tag"))
        ctx.progress(3, 10, "补导中")
        return {"summary": "补导 2 项"}

    def _pump(sec):
        t0 = time.time()
        while time.time() - t0 < sec:
            app.processEvents()
            time.sleep(0.01)

    check(w.run_export(_fn, {"tag": "pet"}, "补导宠物图库（2 项）",
                       on_done=lambda ok, sm: seen.append((ok, sm))) is True,
          "`run_export` 没起得来（卡片之外的小任务这条路 ✗）")
    check(w.progress.maximum() == 0,
          "起任务之后读条没变**忙式**（range 还是 0..%d ⇒ 人看着像「空闲」✗）"
          % w.progress.maximum())

    _pump(3.0)
    check(len(seen) == 1 and seen[0] == (True, "补导 2 项"),
          "结算没回调发起方（卡片靠它重刷宠物列表 ✓）/ 回调得不对：%r" % (seen,))
    txt = w.txt_log.toPlainText()
    check("补导宠物图库（2 项）" in txt and "小任务：pet" in txt,
          "小任务的标题/日志没进工作台（那就成了「偷偷跑」✗）：\n%s" % txt[-300:])
    _pump(0.5)
    check(w.task is None, "任务结束了但 `self.task` 还挂着（下一条任务会被挡 ✗）")

    # ⑤ 正忙时不许再起（调用方据此如实说"有别的任务在跑"✓ 别静默 ✗）
    seen2 = []

    def _slow(params, ctx):
        time.sleep(0.6)
        return {"summary": "慢"}

    check(w.run_export(_slow, {}, "慢任务", on_done=lambda ok, sm: seen2.append(ok)) is True,
          "第二条起不来（第一条收尾没把状态清干净 ✗）")
    check(w.run_export(_fn, {}, "不该起来", on_done=None) is False,
          "正忙时又答应起了一条 ⇒ 两条一起跑（读条/取消都会乱 ✗）")
    _pump(2.5)
    check(seen2 == [True], "慢任务没正常结算：%r" % (seen2,))


TESTS = (
    ("⭐⭐ 卡片之外的小任务**也走工作台那条读条**（补导宠物图库用；用户 2026-10-05）",
     t_run_export_uses_workbench_bar),
    ("⭐⭐ 「关了必须走 / 卡住不许烧核」：两道看门狗（留栈 + 退出码 3/4；用户 2026-10-05）",
     t_watchdog_guarantees_exit),
    ("⭐⭐ 离屏 ⇒ **绝不连被控机**：闸在真源（`net_allowed`）+ 连线程都不起（用户 2026-10-05）",
     t_offscreen_never_connects_device),
    ("启动自动打开上次项目 + 参数页说明「参数存到哪儿」（用户 2026-10-04）",
     t_startup_opens_last_project_and_says_where_params_go),
    ("主视区页签开合：开一次/不重复/关掉不销毁/起始页关不掉", t_view_open_close_flow),
    ("「实时」是常驻页签：关不掉、且页签上没有「×」（用户 2026-09-28）",
     t_closing_live_tab_says_it_keeps_running),
    ("换项目只清步骤详情页，常驻工作页与起始页留着", t_clear_pages_keeps_work_tabs),
    ("源码：主视区开合只有 open_view / close_view 一条路", t_viewer_has_single_entry),
    ("弹窗几何：按客户端一份存取；坏值当没存过", t_window_state_roundtrip),
    ("弹窗几何：hide 就存、show 就恢复；屏幕外的坐标夹回屏幕内（标题栏一定抓得到）",
     t_window_state_save_restore),
    ("规范：所有能拉大拉小的弹窗都接了窗口几何（键名唯一）", t_window_state_covers_editors),
    ("画布缩放也记住：hide 存 / show 按回去（用户报的「每次都要重调缩放」）",
     t_view_zoom_state),
)


def main() -> int:
    failed = 0
    for name, fn in TESTS:
        # ⚠ `flush=True` + 先打"开始"（**故意**）：离屏自检里万一又出原生崩溃（段错误 ⇒
        #   连 `print` 的缓冲都可能丢 ✗），有这两行就能从输出里看出**崩在哪一条** ✓。
        print("==> %s" % name, flush=True)
        try:
            fn()
        except Exception as e:
            failed += 1
            print("[FAIL] %s\n       %s: %s" % (name, type(e).__name__, e), flush=True)
        else:
            print("[ OK ] %s" % name, flush=True)
        # ⚠⚠ **每条之后都要把 Qt 的延迟删除队列跑干净**（2026-09-28 修 ✓）：
        #   用例里那些 `deleteLater()`（`_clear_pages` 等）是"**事件循环里才真删**"，而离屏
        #   自检几乎不跑事件循环 ⇒ 它们**跨用例累积** ⇒ 等**解释器退出**时（`QApplication`
        #   已在拆）一起还账 ⇒ **段错误 `0xC0000005`** ✗（症状极迷惑：所有断言都过了、
        #   `faulthandler` 显示 `<no Python frame>` ✗）。⇒ 在这儿统一清 ✓（一处管全部 ✓）。
        try:
            from PyQt5.QtWidgets import QApplication
            _app = QApplication.instance()
            if _app is not None:
                _app.processEvents()
        except Exception:
            pass
    print("\n%d/%d 通过" % (len(TESTS) - failed, len(TESTS)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
