"""框选自检：放大镜摆位 / 坐标换算 / 倍数 / 藏调用方窗口 / 唯一实现（UI 规范 §8）。

**为什么要有它**
    框选看着是个小工具，但差一两个像素的后果都是**下游直接错**：探针错一个方块
    宽度整条时间码错位、血条错一像素采样到背景、小地图框大了 B 机「底图对不上」
    （现象看着像寻路坏了）。所以框选按**功能**对待，这里把它那几条规矩钉住：

    · 放大镜**不压住光标**、**不越界**（拖拽时还要挪到框外侧）；
    · `<4px` 的框是误点，不返回结果；Esc 取消；
    · 结果坐标 = 图像坐标 + origin（屏幕框选时要加屏幕原点）；
    · 抓屏前把调用方窗口藏起来（不藏就框到自己的界面）；
    · 全仓库只有**一份**实现，且它只依赖 PyQt5（A 机部署台也要用，那边没 numpy）。

跑法：
    python -m tools.selftest_region       # 全过返回 0，有失败返回 1
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt, QTimer      # noqa: E402
from PyQt5.QtGui import QColor, QKeyEvent, QMouseEvent, QPixmap   # noqa: E402
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow    # noqa: E402

import gui.region_picker as rp                                    # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
_APP = None


def app():
    global _APP
    _APP = _APP or QApplication.instance() or QApplication([])
    return _APP


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def make_pix(w=400, h=300):
    """造一张能区分的图（画几条线，方便看放大镜取的是不是同一块）。"""
    from PyQt5.QtGui import QPainter, QPen
    pix = QPixmap(w, h)
    pix.fill(QColor("#2b2f36"))
    p = QPainter(pix)
    p.setPen(QPen(QColor("#e8eaed"), 1))
    for x in range(0, w, 17):
        p.drawLine(x, 0, x, h)
    p.end()
    return pix


def find_dialog():
    for w in QApplication.topLevelWidgets():
        if isinstance(w, QDialog) and w.isVisible():
            return w
    return None


class Driver:
    """替人操作那个框选窗口：**等它真的出现**再发假鼠标/键盘事件。

    为什么要轮询等（而不是固定 120ms 后发一次）：抓屏那条路会先"藏起调用方
    窗口 + 等 250ms"，窗口出现得比固定延时晚 —— 抢跑的话事件发给空气，
    对话框一直开着，测试看起来像"框选没返回"（本文件第一版就是这么错的）。
    另外每个 Driver 自带看门狗，真出问题时也只会测试失败，不会把自检挂死。
    """

    def __init__(self, x1, y1, x2, y2, esc=False, timeout_ms=4000):
        self.esc = esc
        self.pts = (x1, y1, x2, y2)
        self._t = QTimer()
        self._t.setInterval(30)
        self._t.timeout.connect(self._tick)
        self._wd = QTimer()
        self._wd.setSingleShot(True)
        self._wd.setInterval(timeout_ms)
        self._wd.timeout.connect(self._bang)

    def start(self):
        self._t.start()
        self._wd.start()
        return self

    def stop(self):
        self._t.stop()
        self._wd.stop()

    def _tick(self):
        dlg = find_dialog()
        if dlg is None:
            return
        self._t.stop()
        if self.esc:
            QApplication.sendEvent(dlg, QKeyEvent(QEvent.KeyPress, Qt.Key_Escape,
                                                  Qt.NoModifier))
            return
        x1, y1, x2, y2 = self.pts
        for ev in (QMouseEvent(QEvent.MouseButtonPress, QPointF(x1, y1),
                               Qt.LeftButton, Qt.LeftButton, Qt.NoModifier),
                   QMouseEvent(QEvent.MouseMove, QPointF(x2, y2),
                               Qt.NoButton, Qt.LeftButton, Qt.NoModifier),
                   QMouseEvent(QEvent.MouseButtonRelease, QPointF(x2, y2),
                               Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)):
            QApplication.sendEvent(dlg, ev)

    def _bang(self):
        dlg = find_dialog()
        if dlg is not None:
            dlg.close()


def drag_and_get(x1, y1, x2, y2, fn, *a, **kw):
    """替人拉一次框，返回框选函数的结果。"""
    d = Driver(x1, y1, x2, y2).start()
    try:
        return fn(*a, **kw)
    finally:
        d.stop()


def esc_and_get(fn, *a, **kw):
    """在框选窗口里按 Esc → 取消。"""
    d = Driver(0, 0, 0, 0, esc=True).start()
    try:
        return fn(*a, **kw)
    finally:
        d.stop()


# ---------------------------------------------------------------- 放大镜摆位

def t_loupe_placement():
    """放大镜不压住光标、整体不越窗口（拖拽时还要挪到框外侧）。"""
    app()
    sel = rp.LoupeSelector(make_pix(400, 300))
    check(sel.width() == 400 and sel.height() == 300,
          "窗口尺寸应当等于源图：%dx%d" % (sel.width(), sel.height()))

    # 光标在正中：放大镜不盖住光标，且落在窗口内
    sel._cursor = QPoint(200, 150)
    parts = sel._loupe_parts()
    check(parts is not None, "正中位置应当算得出放大镜")
    box, img, src, mark, cap = parts
    check(not (img.left() <= 200 <= img.right() and
               img.top() <= 150 <= img.bottom()),
          "放大镜压住了光标：img=%s" % (img,))
    check(box.left() >= 0 and box.top() >= 0
          and box.right() < sel.width() and box.bottom() < sel.height(),
          "放大镜越界了：box=%s  窗口 %dx%d"
          % (box, sel.width(), sel.height()))

    # 准心必须落在取到的源图块里，且指着光标那个像素
    check(src.contains(src.x() + (mark.x() - img.left()) // sel._zoom,
                       src.y() + (mark.y() - img.top()) // sel._zoom),
          "准心指的不是源图块里的像素")
    check(src.x() + (mark.x() - img.left()) // sel._zoom == 200
          and src.y() + (mark.y() - img.top()) // sel._zoom == 150,
          "准心没指着光标那一个像素")

    # 贴右下角：整体必须被夹回窗口内
    sel._cursor = QPoint(399, 299)
    box2, img2, _s2, _m2, _c2 = sel._loupe_parts()
    check(box2.right() < sel.width() and box2.bottom() < sel.height(),
          "贴角时越界了：box=%s" % (box2,))

    # 拖拽中：放框的外侧（往右拖 → 放大镜在左边）
    sel._origin = QPoint(100, 100)
    sel._cursor = QPoint(300, 200)
    _b, img3, _s, _m, _c = sel._loupe_parts()
    check(img3.right() <= 300, "往右拖时放大镜该在框的外侧（左侧），实际 img=%s" % (img3,))


def t_zoom_keys():
    """`+`/`-` 调倍数并夹在 4~16（滚轮刻意不动，见 UI 规范 §1）。"""
    app()
    sel = rp.LoupeSelector(make_pix(400, 300), zoom=8)

    def press(key):
        sel.keyPressEvent(QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier))

    press(Qt.Key_Plus)
    check(sel._zoom == 9, "+ 之后倍数应当是 9，实际 %d" % sel._zoom)
    for _ in range(20):
        press(Qt.Key_Plus)
    check(sel._zoom == rp.ZOOM_MAX, "倍数没夹在上限：%d" % sel._zoom)
    for _ in range(30):
        press(Qt.Key_Minus)
    check(sel._zoom == rp.ZOOM_MIN, "倍数没夹在下限：%d" % sel._zoom)
    # 起始倍数由调用方给（HP/MP 条那种细长目标要给大一点）
    check(rp.LoupeSelector(make_pix(400, 300), zoom=99)._zoom == rp.ZOOM_MAX,
          "起始倍数没被夹住")


def t_coords_and_origin():
    """结果坐标 = 图像坐标 + origin；尺寸是 QRect 含两端的约定（101×91 这种）。"""
    app()
    sel = rp.LoupeSelector(make_pix(400, 300), origin=(1920, 1080))
    sel._origin = QPoint(100, 40)
    sel._current = QPoint(200, 130)
    check(sel.result_rect() == (2020, 1120, 101, 91),
          "带 origin 的换算不对：%s" % (sel.result_rect(),))

    plain = rp.LoupeSelector(make_pix(400, 300))
    plain._origin = QPoint(100, 40)
    plain._current = QPoint(300, 190)
    check(plain.result_rect() == (100, 40, 201, 151),
          "屏幕框选的尺寸约定不对：%s" % (plain.result_rect(),))


def t_modal_small_and_esc():
    """真跑一次模态框：正常框返回、太小的框当误点、Esc 取消。"""
    app()
    pix = make_pix(400, 300)

    got = drag_and_get(100, 40, 300, 190, rp.select_on_pixmap, pix)
    check(got == (100, 40, 201, 151), "正常框选返回 %s" % (got,))

    got = drag_and_get(10, 10, 12, 12, rp.select_on_pixmap, pix)   # 3×3 = 误点
    check(got is None, "太小的框应当当误点（返回 None），实际 %s" % (got,))

    got = esc_and_get(rp.select_on_pixmap, pix)
    check(got is None, "Esc 应当取消，实际 %s" % (got,))


def t_hide_owner():
    """抓屏前把调用方窗口藏起来；抓完放回来；抓的那一刻它是不可见的。"""
    app()
    win = QMainWindow()
    win.resize(300, 200)
    win.show()
    seen = {}

    def fake_grab():
        seen["visible"] = win.isVisible()
        return make_pix(400, 300), (0, 0)

    try:
        got = drag_and_get(20, 20, 120, 100, rp.select_screen_region, win,
                           hide_owner=True, grab=fake_grab)
        check(seen.get("visible") is False, "抓屏那一刻调用方窗口应当是不可见的")
        check(win.isVisible(), "抓完要把窗口放回来")
        check(got == (20, 20, 101, 81), "抓屏框选的结果不对：%s" % (got,))
    finally:
        win.close()


def t_b_side_entry():
    """B 机入口（在实时画面上框）：坐标要和共用实现一致（含 numpy→QPixmap）。"""
    app()
    import numpy as np
    from gui.region_selector import select_region_on_image
    img = np.zeros((300, 400, 3), np.uint8)
    img[:] = (40, 40, 40)
    img[40:190, 100:300] = (200, 200, 200)      # 框内留个亮块，便于肉眼核对
    got = drag_and_get(100, 40, 300, 190, select_region_on_image, img)
    check(got == (100, 40, 201, 151),
          "B 机入口的坐标不对：%s（应当和共用实现一致）" % (got,))


# ---------------------------------------------------------------- 规范

def t_grab_follows_cursor_screen():
    """⭐⭐⭐⭐ **抓屏必须跟着光标那块屏走，origin 必须是那块屏的原点**（用户 2026-10-17 ✓
    原话：「**远程选区又坏了 / 怎么每次都坏？**」✗）。

    ⚠ 真病根（不是偶发 ✓ 是**每次都错** ✓）：
      · 老 `grab_primary_pixmap` 写死 `primaryScreen()` ✗ 且 origin 写死 `(0, 0)` ✗；
      · 多显示器时主屏未必是游戏那块（本机实测：主屏 `(0,0)`、副屏 `(-2560,0)` ✓）；
      · ⇒ 在副屏上框：显示的是**主屏画面** ✓、返回的 x 还少了 2560 ✓
        ⇒ 存进 `deploy.json` 的推流区域 / 小地图区域**整体偏到屏幕外** ✓
        ⇒ 下游现象就是"推流黑屏 / 底图对不上 / 黄点认不出" ✓ 用户眼里"选区坏了"✓。
    判据（四条 ✓）：
      ① 默认抓屏**跟光标**（源码里有 `screenAt` ✓）；
      ② **origin == 那块屏 `geometry().topLeft()`**（行为：换屏 ⇒ origin 跟着变 ✓）；
      ③ 抓到的是那块屏的**逻辑尺寸**（DPI 已归一 ✓ 窗口不会超出屏幕 ✓）；
      ④ **全黑要说清楚**（抛错 + 文案里给原因 ✓ 不再"点了没反应"✓）。
    """
    from PyQt5.QtGui import QGuiApplication

    src = (ROOT / "gui" / "region_picker.py").read_text(encoding="utf-8")
    seg = src[src.index("def grab_screen_pixmap"):]
    seg = seg[:seg.index("\ndef ", 10)]
    check("screenAt" in src and "geometry().topLeft()" in seg,
          "① 默认抓屏必须**跟光标所在屏**、origin 用该屏 `geometry().topLeft()` ✗ "
          "—— 写死主屏 / 写死 (0,0) ⇒ 副屏上框出来的区域每次全错 ✓（用户现场就是这个 ✓）")
    check("_looks_black" in seg and "raise RuntimeError" in seg,
          "④ 抓到**全黑**必须抛错说清（远程断开 / 锁屏 / 休眠）✗ —— 不能让人对着黑图框 ✓")

    app()
    from PyQt5.QtCore import QRect

    class _FakeScreen:
        """假屏：**不真抓屏**（自检跑在 offscreen 平台 ✓ 真屏抓不到 ✗），
        但把"origin / DPI 归一"这两件事原样走一遍 ✓。"""

        def __init__(self, rect, pix):
            self._rect, self._pix = rect, pix

        def name(self):
            return "FAKE"

        def geometry(self):
            return self._rect

        def grabWindow(self, _wid=0):
            return self._pix

    def _shot(w, h, dpr=1.0, color="#808080"):
        p = QPixmap(int(w * dpr), int(h * dpr))
        p.fill(QColor(color))
        if dpr != 1.0:
            p.setDevicePixelRatio(dpr)
        return p

    _orig = rp._pick_screen
    try:
        # ⭐ 副屏（原点 -2560）—— 现场就是这块屏上的游戏框不准 ✓
        _scr = _FakeScreen(QRect(-2560, 0, 1600, 900), _shot(1600, 900))
        rp._pick_screen = lambda: _scr
        _pix, _origin = rp.grab_screen_pixmap()
        check(_origin == (-2560, 0),
              "② 副屏 origin 该是 (-2560, 0)，实测 %r ✗ —— 写死 (0,0) 时每个坐标少 2560 ✓"
              % (_origin,))
        check((_pix.width(), _pix.height()) == (1600, 900),
              "③ 抓到的该是那块屏的逻辑尺寸 1600x900，实测 %dx%d"
              % (_pix.width(), _pix.height()))

        # DPI 缩放 2.0：抓回来是设备像素 3200x1800 ⇒ 必须缩回 1600x900 ✓
        _hidpi = _FakeScreen(QRect(0, 0, 1600, 900), _shot(1600, 900, dpr=2.0))
        rp._pick_screen = lambda: _hidpi
        _pix2, _origin2 = rp.grab_screen_pixmap()
        check((_pix2.width(), _pix2.height()) == (1600, 900),
              "③ DPI=2.0 的设备像素图必须缩回逻辑尺寸（实测 %dx%d ✗ ⇒ 选区窗会超出屏幕 ✓）"
              % (_pix2.width(), _pix2.height()))
        check(abs(float(rp.last_scale) - 2.0) < 1e-6,
              "③ `last_scale` 该记下 2.0（排查推流偏移要用 ✓）：%r" % (rp.last_scale,))

        # ⑤ **每次抓屏都要落一份诊断**（用户 2026-10-17：A 机单屏、现象只有"每次都坏"，
        #    光靠猜分不出"缩放 / 黑图 / 窗口盖住" ✓ ⇒ 让它自己写参数 ✓）
        _diag = ROOT / "data" / "debug" / "region_grab.txt"
        check(_diag.exists(),
              "⑤ 抓屏没落诊断文件 `data/debug/region_grab.txt` ✗"
              "（A 机排查「每次都坏」就靠它 ✓）")
        _txt = _diag.read_text(encoding="utf-8")
        for _k in ("ok =", "detail =", "last_scale =", "geometry =",
                   "screens =", "platform =", "high_dpi_scaling ="):
            check(_k in _txt, "⑤ 诊断里缺 `%s` ✗（这几项就是分病因用的 ✓）：%r" % (_k, _txt))
        check("dpr=2.000" in _txt,
              "⑤ 诊断里要写清这一次的 dpr（缩放那条病根全靠它 ✓）：%r" % (_txt,))

        # 全黑 ⇒ 必须抛错（调用方会弹「框选失败」✓ 不再静默 ✓）
        _dark = _FakeScreen(QRect(0, 0, 800, 600), _shot(800, 600, color="#000000"))
        rp._pick_screen = lambda: _dark
        try:
            rp.grab_screen_pixmap()
            _raised = False
        except RuntimeError as e:
            _raised = "全黑" in str(e)
        check(_raised, "④ 抓到全黑必须抛 RuntimeError 说清（远程断开 / 锁屏 / 休眠）✗")
    finally:
        rp._pick_screen = _orig

    _dim = QPixmap(320, 240)
    _dim.fill(QColor("#101010"))
    check(rp._looks_black(_dim) is False,
          "④ 深色画面（亮度 16）不许误报成黑屏 ✗（游戏夜景会被判死 ✓）")


def t_single_implementation():
    """只有一份实现；它只依赖 PyQt5；各处都委派过来（UI 规范 §8）。"""
    src = (ROOT / "gui" / "region_picker.py").read_text(encoding="utf-8")
    # 只看**代码行**（顶格写的 import）：文档里会提到"不能 import numpy"，
    # 那是解释，不该被当成违规（第一版自检就这么误报了一次）
    code = [ln.strip() for ln in src.splitlines()]
    for bad in ("import numpy", "import cv2", "from core import wincap",
                "from core.wincap import", "import wincap"):
        check(not any(ln.startswith(bad) for ln in code),
              "region_picker 里出现了 %r —— A 机部署台没装它，会 import 失败" % bad)
    check("def select_on_pixmap(" in src and "def select_screen_region(" in src,
          "region_picker 没提供两个入口")

    # 部署台那边的框选不再自己写一份：不许有本地的框选窗口类/自绘放大镜
    push = (ROOT / "tools" / "minimap_push.py").read_text(encoding="utf-8")
    check("class Pick" not in push and "paintEvent" not in push,
          "minimap_push 里又自己写了一份框选窗口")
    check("select_screen_region" in push, "部署台框选没走共用实现")

    # B 机入口也是委派
    sel = (ROOT / "gui" / "region_selector.py").read_text(encoding="utf-8")
    check("from gui.region_picker import" in sel and "paintEvent" not in sel,
          "region_selector 又自己实现了一遍")

    # HP/MP 走共用实现，且给了起始倍数
    panel = (ROOT / "gui" / "player_panel.py").read_text(encoding="utf-8")
    check("select_region_on_image(frame, self, zoom=" in panel,
          "HP/MP 框选没给起始倍数（细长目标该给大一点）")

    # 规范里写明「默认都要放大」
    spec = (ROOT / "docs" / "UI规范.md").read_text(encoding="utf-8")
    check("框选一律带放大镜" in spec, "UI 规范里没写这条")
    check("region_picker" in spec, "UI 规范里没指出唯一实现是哪个模块")


def t_import_without_numpy():
    """模拟 A 机（部署台那台只装了 PyQt5）：挡住 numpy/cv2/wincap 也要能 import。

    这条比源码那几条更硬：以后谁给 region_picker 加了个 numpy 依赖，部署台
    会在 A 机上报「No module named 'numpy'」—— 而那边根本没法装（A 机不需要
    也不该装这套）。开子进程跑，免得污染本进程的 sys.modules。
    """
    import subprocess
    code = (
        "import sys\n"
        "for name in ('numpy', 'cv2', 'core.wincap', 'core'):\n"
        "    sys.modules[name] = None\n"      # None = 再 import 直接 ImportError
        "import gui.region_picker as rp\n"
        "assert rp.ZOOM_DEFAULT == 8\n"
        "assert callable(rp.select_screen_region)\n"
        "print('ok')\n"
    )
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen",
               PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, encoding="utf-8", errors="replace", timeout=60)
    check(r.returncode == 0 and "ok" in (r.stdout or ""),
          "没有 numpy 的环境下 import 失败（A 机部署台就是这么跑的）：\n%s"
          % (r.stdout or "")[-600:])


TESTS = (
    ("放大镜摆位：不压光标 / 不越界 / 拖拽时挪到框外", t_loupe_placement),
    ("倍数：+/- 调节并夹在 4~16", t_zoom_keys),
    ("坐标换算：图像坐标 + origin，尺寸含两端", t_coords_and_origin),
    ("模态框：正常 / 误点 / Esc", t_modal_small_and_esc),
    ("抓屏前藏起调用方窗口", t_hide_owner),
    ("B 机入口（实时画面上框）", t_b_side_entry),
    ("⭐⭐⭐⭐ 抓屏跟光标那块屏 + origin 正确 + DPI 归一 + 全黑要报（选区\"每次都坏\"·用户 2026-10-17）",
     t_grab_follows_cursor_screen),
    ("唯一实现 + 规范写死", t_single_implementation),
    ("没有 numpy 也能 import（模拟 A 机）", t_import_without_numpy),
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
