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
    """建主窗口（离屏）。**只碰界面**：不打开项目、不连流 ✓、**也不连远端键盘** ✓。"""
    from PyQt5.QtWidgets import QApplication

    from decision import agent as _ag
    from gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    # ⚠⚠ **必须先把输入设备摁成 `local`**（2026-09-28 修 ✓ 踩过）：
    #   项目配置里 `input_device` 常常是 `remote` ⇒ `PlayerPanel` 构造时会**自动**
    #   `_do_connect` ⇒ 起一条**后台线程**去 TCP 连被控机（`remote_kbd` ✓）⇒ 那条线程
    #   **没人管**，本用例跑完、主线程一退出 ⇒ 它就变成"**在别的线程里摸已经拆掉的
    #   解释器 / Qt**" ⇒ **段错误 `0xC0000005`** ✗。
    #   ⚠ 症状极迷惑（这次又踩）：`faulthandler` 打出来是**另一条线程**的栈
    #     （`socket.create_connection` ← `kbd_client` ← `input.use_network` ← `_do_connect`），
    #     而**当前线程**只是"正在跑用例" ⇒ 一眼看着像"用例自己崩了" ✗；
    #     而且它是**时机性**的：对面恰好拒绝连接时那条线程早早异常退出 ⇒ 碰巧不崩 ✗
    #     （所以之前有几轮是 8/8 通的 ✓ 不是修好了，是没赶上 ✗）。
    #   ⚠ **光改 `settings.input_device` 不管用**（踩过）：`MainWindow()` 构造时会**重载
    #     项目配置**（`from_dict` 里那个 `input_device: remote` ✓）⇒ 当场改回 `remote` ✓。
    #   ⚠ **只改 `decision.input.use_network` 也不够**（又踩一次）：起线程那一步在
    #     `PlayerPanel._apply_input_device` 里 ✓ ⇒ 从**源头**掐掉它最干净 ✓
    #     （离屏自检本来就不该真发按键、更不该连被控机 ✓）。
    _ag.settings.input_device = "local"
    # ⚠⚠ **离屏自检绝不连被控机**（2026-09-28 ✓ 这是"段错误 `0xC0000005`"的真凶）：
    #   `MainWindow()` 构造时会**重载项目配置**（`input_device: remote` ✓）⇒ 光设 `settings`
    #   会被当场盖回去 ✗ ⇒ `PlayerPanel._apply_input_device` 起一条**后台线程**去 TCP 连
    #   被控机（`remote_kbd` ✓）⇒ 它在**几秒后才超时**、而那时用例早跑完、`PlayerPanel`
    #   的 C++ 对象**已经销毁** ⇒ 线程回来碰它 ⇒ 崩 ✗（`faulthandler` 会指到
    #   `socket.create_connection` ← `kbd_client` ← `input.use_network` ← `_do_connect` ✓）。
    #   ⚠ 它是**时机性**的（对面秒拒就碰巧不崩 ✗）⇒ 所以"多跑几次"不算验证，
    #     这里**从源头掐掉** ✓ 并且**当场自检**（`_assert_net_disabled` ✓ 免得哪天又没生效 ✗）。
    from decision import input as _dinput
    if not hasattr(_dinput, "_selftest_orig_use_network"):
        _dinput._selftest_orig_use_network = _dinput.use_network
        _dinput.use_network = lambda *a, **k: None      # 连远端 ⇒ 直接当成功返回 ✓
    from gui.player_panel import PlayerPanel as _PP
    if not hasattr(_PP, "_selftest_orig_apply_input_device"):
        _PP._selftest_orig_apply_input_device = _PP._apply_input_device
        _PP._apply_input_device = lambda self, dev=None: None   # 不起那条连网线程 ✓
    assert _dinput.use_network.__name__ == "<lambda>", "打桩没生效（自检要立刻炸 ✗）"
    assert _PP._apply_input_device.__name__ == "<lambda>", "打桩没生效（自检要立刻炸 ✗）"
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
                             ("battle_zone_list", "BattleZoneListDialog"))}
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


TESTS = (
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
