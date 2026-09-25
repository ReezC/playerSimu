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


def t_load_warn_on_status():
    """本机负载告警要压在状态行**最前面**，且负载恢复后**自己消失**。

    为什么不写成源码匹配：这条的判据是「人看得见」—— 源码里有没有那几个字
    说明不了这件事，得真喂一份统计进去看标签上是什么。做法同 t_coalesce：
    直接调 `_on_stats`（它不碰网络、不碰线程，只排版）。

    消失那一半同样要测：一条常驻的告警等于没有告警（人会学会无视它），
    而它恰恰是「推理为什么慢」的唯一解释。
    """
    app, lp, p, made, orig = _panel()
    try:
        base = {"size": (1366, 768), "recv_fps": 48.0, "proc_fps": 30.0,
                "dropped": 2, "show_fps": 30.0, "boxes": 3, "probe_on": False}
        p._on_stats(dict(base, infer_ms=23.7,
                         load_warn="负载告警：内存 2.2/31.8GB、并行子进程 11 个(11.8GB)",
                         load_detail="可用物理内存 2.2 / 31.8 GB\n并行子进程 11 个"))
        txt = p.lbl_stats.text()
        check("负载告警" in txt, "状态行没显示负载告警：%r" % txt)
        check(txt.lstrip().startswith("【负载告警"),
              "负载告警没被顶到最前面（它是那些数字为什么变差的解释）：%r" % txt[:60])
        check("23.7 ms" in txt, "告警把原来的推理耗时挤掉了：%r" % txt)
        check("1366" in txt, "告警把分辨率挤掉了：%r" % txt)
        check("并行子进程 11 个" in p.lbl_stats.toolTip(),
              "明细没进 tooltip：%r" % p.lbl_stats.toolTip())

        p._on_stats(dict(base, infer_ms=10.2))
        check("负载告警" not in p.lbl_stats.text(),
              "负载恢复后状态行还挂着告警：%r" % p.lbl_stats.text())
        check("负载告警" not in p.lbl_stats.toolTip(),
              "负载恢复后 tooltip 还挂着明细：%r" % p.lbl_stats.toolTip())
    finally:
        lp._bgr_to_pixmap = orig
        p.close()


def t_minimap_overlay_is_display_only():
    """小地图叠图：只准画在显示层、**帧数据一个字节都不许改**，位置要落在画面坐标上。

    **为什么这条最要紧**：实时画面那一帧（`current_frame()`）还要喂探针标定、
    HP/MP 条框选，以及小地图「从实时画面框选」的标定弹窗（它拿这帧裁出面板去和
    底图做模板匹配）。叠图一旦烘进帧，标定弹窗就会**拿叠图和它自己匹配** ——
    匹配分虚高，框选还会框到画上去的东西。而且烘进帧实测要 2.7 ms/帧，
    显示层只要 0.06 ms。

    所以判据是两条，缺一不可：
      · `current_frame()` 必须没变（同一个对象、像素逐点相同）；
      · 画面上**该出现的那块必须真有叠加色** —— 否则就是"勾了没反应"。
    """
    from PyQt5.QtGui import QColor, QPixmap

    app, lp, p, made, orig = _panel()
    try:
        fw, fh = 600, 400
        frame = np.zeros((fh, fw, 3), np.uint8)
        frame[:, :, 2] = 200            # BGR 全红，方便一眼看出哪儿被盖住了
        before = frame.copy()

        p._on_frame(frame)
        app.processEvents()
        pm = p.view.pixmap()
        check(pm is not None and not pm.isNull() and pm.width() > 40
              and pm.height() > 30,
              "画面没画出来（拿不到 pixmap），后面的判据没法验")
        sx, sy = pm.width() / float(fw), pm.height() / float(fh)

        # 叠一块纯绿进去（alpha=1 = 完全不透明，判据才好写）
        rect = (150, 100, 200, 150)                       # 画面坐标 (x, y, w, h)
        green = QPixmap(8, 8)
        green.fill(QColor(0, 255, 0))
        p.set_minimap_overlay(green, None, rect, 1.0)
        img = p.view.pixmap().toImage()

        cx = int((rect[0] + rect[2] / 2.0) * sx)
        cy = int((rect[1] + rect[3] / 2.0) * sy)
        c = img.pixelColor(cx, cy)
        check(c.green() > 200 and c.red() < 80,
              "叠加没画到该在的位置（画面 %s → 截图 (%d, %d) 取到 rgb(%d, %d, %d)）"
              % (rect, cx, cy, c.red(), c.green(), c.blue()))

        # 叠加区**外面**还是原来的红底（说明只糊了那一块，没整幅盖住）
        out = img.pixelColor(3, 3)
        check(out.red() > 150 and out.green() < 80,
              "叠加把整幅画面都盖了：角落取到 rgb(%d, %d, %d)"
              % (out.red(), out.green(), out.blue()))

        # ★ 帧数据必须原封不动：它还要喂探针/血条/小地图标定弹窗
        check(p.current_frame() is frame,
              "current_frame() 换对象了 —— 叠图不许掺进这条链路")
        check((frame == before).all(),
              "叠图改到帧数据了！标定弹窗会拿叠图和它自己匹配（匹配分虚高）")

        # 卸掉之后要能自己擦干净（别留一层画上去的）
        p.set_minimap_overlay(None)
        img2 = p.view.pixmap().toImage()
        c2 = img2.pixelColor(cx, cy)
        check(c2.red() > 150 and c2.green() < 80,
              "卸掉叠图后那块还留着颜色：rgb(%d, %d, %d)"
              % (c2.red(), c2.green(), c2.blue()))

        # 整块在画面外（换了分辨率还没重框）：不许崩，也不该画
        p.set_minimap_overlay(green, None, (fw + 50, fh + 50, 40, 40), 1.0)
        check(p.view.pixmap() is not None, "画面外的叠图把绘制弄崩了")
        p.set_minimap_overlay(None)
    finally:
        lp._bgr_to_pixmap = orig
        p.close()


def t_overlay_source_out_of_image():
    """源矩形超出叠加图时（面板框得比底图还高就会有）：画出来的范围必须正确。

    实测遇到过这种框法：面板 134×109，而底图只有 101 高 —— 源矩形比图还高，
    多出来的那几行图里本来就没有。

    这条钉的是**结果**、不是某个实现：实测 Qt 的 `drawPixmap` 自己就会剪掉源、
    并把目标矩形按比例调好（和"手工求交集再按比例裁目标"逐像素一致），所以这里
    不要求谁去手写裁剪 —— 但谁要是把源**铺满**整个目标矩形（那就是整层叠图偏
    一截、看着像"标定偏了"），这条会红。

    判据的关键在**该空着的地方必须还是原色**：只查"该有叠图的地方有没有"的话，
    两种错法都能过。
    """
    from PyQt5.QtGui import QColor, QPixmap

    app, lp, p, made, orig = _panel()
    try:
        fw, fh = 600, 400
        frame = np.zeros((fh, fw, 3), np.uint8)
        frame[:, :, 2] = 200
        p._on_frame(frame)
        app.processEvents()
        pm = p.view.pixmap()
        sx, sy = pm.width() / float(fw), pm.height() / float(fh)

        green = QPixmap(40, 40)
        green.fill(QColor(0, 255, 0))
        # 目标 200×200；源声称 80×80（-20 起），可图只有 40×40
        # → 只有「源 ∩ 图」= (0,0,40,40) 那一半该被画出来
        p.set_minimap_overlay(green, (-20, -20, 60, 60), (200, 100, 200, 200), 1.0)
        img = p.view.pixmap().toImage()

        def at(fx, fy):
            c = img.pixelColor(int(fx * sx), int(fy * sy))
            return (c.red(), c.green(), c.blue())

        ins = at(300, 200)          # 有内容的那一半
        check(ins[1] > 200 and ins[0] < 80,
              "该有叠图的地方没画上：rgb%s" % (ins,))
        gap = at(212, 112)          # 源在图外的那一半 → 必须还是原色
        check(gap[0] > 150 and gap[1] < 80,
              "源在图外的那一块也被糊上了（源被铺满了整个目标矩形，"
              "整层叠图会偏一截）：rgb%s" % (gap,))
        out = at(500, 350)
        check(out[0] > 150 and out[1] < 80,
              "画到目标矩形外面去了：rgb%s" % (out,))
        # 整块都在图外：不许崩、也不该画
        p.set_minimap_overlay(green, (900, 900, 40, 40), (200, 100, 200, 200), 1.0)
        img2 = p.view.pixmap().toImage()
        c2 = img2.pixelColor(int(300 * sx), int(200 * sy))
        check(c2.red() > 150 and c2.green() < 80,
              "源整块在图外却还画了东西：rgb(%d,%d,%d)"
              % (c2.red(), c2.green(), c2.blue()))
        p.set_minimap_overlay(None)
    finally:
        lp._bgr_to_pixmap = orig
        p.close()


def t_probe_box_overlay():
    """实时画面上那条采样框：**颜色跟着判据变、每个采样点都画、且只画在副本上**。

    实测背景（2026-09-25）：几何和屏幕上真正画的码对不上时延迟读数是一坨乱数，
    而界面上一片正常 —— 人能一眼看出来的只有"绿框没框全"。所以把采样框直接画出来、
    颜色直接绑到判据结论上，这是最快的现场判据（也是这次事故唯一的现场证据）。

    另外钉一条**写错了很难查**的：只许画在副本上。`_on_frame` 里 `_pending` 与
    `_last_bgr` 是同一个数组，`current_frame()` 把它交给探针框选/HP 条框选用 ——
    在原图上画框线，线会落进方块改变灰度均值，自己污染自己的采样。
    """
    import unittest.mock as mock

    import cv2 as _cv2
    import numpy as _np

    from tools import probe_codec as pc
    from tools.selftest_probe_tune import synth as _synth

    app, lp, p, made, orig = _panel()
    try:
        geo = {"x": 40.0, "y": 30.0, "cell": 16.0, "gap": 2.25}
        gray = _synth(geo, pc.now_ms(), 40)
        bgr = _cv2.cvtColor(gray, _cv2.COLOR_GRAY2BGR)
        before = bgr.copy()
        fake = (40.0, 30.0, 16.0, 2.25, 40, "测试")
        with mock.patch.object(p, "_probe_geo_px", lambda _shape: fake):
            # ① 判据说可疑 → 红框；且**输入数组一个像素都不许动**
            p._last_stats = {"probe_on": True, "probe_samples": 20,
                             "probe_mono": False, "probe_jumpy": 5}
            out = p._draw_probe_overlay(bgr)
            check(_np.array_equal(bgr, before),
                  "采样框画在原图上了 —— current_frame() 的采样会被自己的框线污染")
            edge_bad = out[28, 80]              # 框上边（y-2=28 那一行）
            check(edge_bad[2] > 150 and edge_bad[0] < 120,
                  "判据可疑时框线不是红的：BGR=%s" % (edge_bad.tolist(),))

            # ② 判据 OK + 有延迟 → 绿框
            p._last_stats = {"probe_on": True, "probe_samples": 20,
                             "probe_mono": True, "probe_jumpy": 0,
                             "probe_value_ok": True, "delay_ms": 121.0}
            out2 = p._draw_probe_overlay(bgr.copy())
            edge_ok = out2[28, 80]
            check(edge_ok[1] > 150 and edge_ok[0] < 120,
                  "判据 OK 时框线不是绿的：BGR=%s" % (edge_ok.tolist(),))

            # ③ 采样点画出来了（白块=黄、黑块=蓝，总要出现）
            cols = {tuple(int(v) for v in c) for c in out2.reshape(-1, 3)}
            check((0, 255, 255) in cols, "没看到白块采样点（黄十字）")
            check((255, 120, 0) in cols, "没看到黑块采样点（蓝十字）")

            # ④ 探针关掉 → 灰框，而且不该画采样点
            p._last_stats = {"probe_on": False}
            out3 = p._draw_probe_overlay(bgr.copy())
            check(out3[28, 80][0] > 100 and out3[28, 80][2] > 100,
                  "探针未启用时框不是灰的：%s" % (out3[28, 80].tolist(),))
    finally:
        lp._bgr_to_pixmap = orig
        p.close()


def t_probe_mono_gate():
    """探针**单调性闸**：几何可疑时不许显示延迟；框选保存前要用实时流验单调性。

    **为什么必须有它**：2026-09-25 查「端到端延迟今天 300~400+」花了半天，最后发现
    几何比实际画的小 2.8px/块（42 块累积偏 118px）→ 低位全采错 → 解出的时间戳是假的，
    而**所有现成判据都放行了**：`ts_plausible` 只问"接近现在吗"，错位凑出来的值照样
    落在那 10 分钟窗口里；框选那条路还会试几百个候选，总能撞上一个"合法"的。
    后果是延迟被报成一坨乱数（p95 4.7s，紧贴自己的 --max-delay），人却去查链路和
    编码参数（四组参数改完中位与 p95 一动没动 —— 因为那个数本来就不是延迟）。

    所以这里钉两条：**状态行不许把假延迟摆出来**、**保存前的闸真的会拦住错几何**。
    """
    import unittest.mock as mock

    import cv2 as _cv2

    from tools import probe_codec as pc
    from tools.selftest_probe_tune import synth as _synth

    app, lp, p, made, orig = _panel()
    try:
        base = {"size": (1366, 768), "recv_fps": 48.0, "proc_fps": 45.0,
                "dropped": 1, "show_fps": 30.0, "boxes": 2, "probe_on": True}

        # ① 时间戳在乱跳 → 状态行报「几何可疑」，而且**不许**把那个延迟数摆出来
        p._on_stats(dict(base, delay_ms=281.0, probe_mono=False, probe_jumpy=7,
                         probe_samples=20))
        txt = p.lbl_stats.text()
        check("几何可疑" in txt, "乱跳时状态行没报几何可疑：%r" % txt)
        check("281" not in txt,
              "几何可疑时还把假延迟（281ms）摆在状态行上 —— 那正是要拦的：%r" % txt)
        tip = p.lbl_stats.toolTip()
        check("乱跳" in tip and "probe_tune" in tip,
              "tooltip 没说清怎么修：%r" % tip[:90])

        # ② 单调正常 → 正常显示延迟（别把好几何也拦了）
        p._on_stats(dict(base, delay_ms=142.0, probe_mono=True, probe_jumpy=0,
                         probe_samples=20))
        txt2 = p.lbl_stats.text()
        check("142" in txt2 and "几何可疑" not in txt2,
              "单调正常时没正常显示延迟：%r" % txt2)

        # ③ 框选保存前那道闸：拿实时流跑几帧验单调性（画对了才放行）
        geo_ok = {"x": 20.0, "y": 20.0, "cell": 16.0, "gap": 2.0}
        geo_drift = dict(geo_ok, gap=1.5)          # 采样式偏 0.5px/块（实测那类偏差）
        bits = 40
        now = pc.now_ms()

        def _frames(n=8):
            """按**正确**几何画 8 帧码带（时间戳每帧 +33ms）→ BGR 帧列表。"""
            out = []
            for k in range(n):
                g = _synth(geo_ok, now - 200 + int(k * 33), bits)
                out.append(_cv2.cvtColor(g, _cv2.COLOR_GRAY2BGR))
            return out

        for sample_geo, want_ok, tag in ((geo_ok, True, "采样几何正确"),
                                         (geo_drift, False, "采样偏 0.5px/块")):
            seq = _frames()
            it = iter(seq)
            with mock.patch.object(p, "current_frame",
                                   lambda: next(it, seq[-1])):
                ok, why = p._verify_geo_live(sample_geo, bits, seconds=1.0, want=6)
            check(ok == want_ok,
                  "%s 时判据结论不对（ok=%s）：%s" % (tag, ok, why))

        # ④ 帧太少时**不拦**（预览刚起就框选，不能因此存不下标定）
        one = iter(_frames(1))
        with mock.patch.object(p, "current_frame", lambda: next(one, None)):
            ok4, why4 = p._verify_geo_live(geo_ok, bits, seconds=0.05, want=6)
        check(ok4, "帧不够时把保存拦了（预览刚起就框不了）：%s" % why4)
    finally:
        lp._bgr_to_pixmap = orig
        p.close()


#: pyflakes 报的哪几类才当"会炸"处理。**只留 undefined name / syntax error**：
#: 上面那两次事故（局部 import 遮蔽、整行被吃掉）pyflakes 报的都是 `undefined name`
#: （拿最小样本实测过）。而"未使用 import/变量"是整洁问题、仓库里也有历史遗留，
#: 混进来会让人习惯性无视这条检查。
_FATAL_PAT = ("undefined name", "syntax error", "referenced before assignment")


def t_static_check_no_crash_classes():
    """静态检查：**会当场炸**的那几类名字错误（undefined / 先用后赋值…）。

    这条是拿两次真实事故换来的（都发生在 `LiveThread._run()` 里，而自检从来不跑
    它 —— 它要一条真流）：
      ① `from tools import probe_codec` 写在函数里，而新代码在它前面一百多行就用了
         → `UnboundLocalError`；
      ② 一次编辑把 `_clock = [0.0, offset_ms]` 整行吃掉 → `NameError`。
    两次都是"一开预览就报错、而自检全绿"。所以这里用 pyflakes 兜住这一类：
    只挑会让程序当场挂的几类，忽略"未使用 import"那种整洁问题。
    没装 pyflakes 就跳过（不阻塞 —— 这是附加防线，不是唯一防线）。
    """
    import io
    import subprocess
    import sys as _sys

    files = []
    for sub in ("gui", "tools", "deploy"):
        files += [str(p) for p in (ROOT / sub).glob("*.py")]
    try:
        r = subprocess.run([_sys.executable, "-m", "pyflakes"] + files,
                           capture_output=True, text=True, timeout=180)
    except Exception as e:                       # noqa: BLE001
        print("      （pyflakes 不可用，跳过：%s）" % type(e).__name__)
        return
    if r.returncode not in (0, 1):
        print("      （pyflakes 跑不起来，跳过）")
        return
    bad = [ln for ln in io.StringIO(r.stdout).read().splitlines()
           if any(pat in ln for pat in _FATAL_PAT)]
    check(not bad, "静态检查发现会当场炸的名字错误（自检跑不到这些路径）：\n       %s"
          % "\n       ".join(bad[:6]))


TESTS = (
    ("连推 10 帧只画最新那帧（合并，不排队）", t_coalesce),
    ("不可见时一帧都不画，但帧仍是最新的", t_hidden_skips_draw),
    ("状态行露出「绘制 ms / 合并丢弃」（源码约定）", t_stats_show_draw),
    ("负载告警顶在状态行最前面，恢复后自己消失", t_load_warn_on_status),
    ("小地图叠图：只画显示层、帧数据不许改", t_minimap_overlay_is_display_only),
    ("小地图叠图：源超出叠加图时画出来的范围要对", t_overlay_source_out_of_image),
    ("探针单调性闸：几何可疑不报延迟、保存前拦错几何", t_probe_mono_gate),
    ("实时画面上的采样框：颜色跟判据、只画在副本上", t_probe_box_overlay),
    ("静态检查：会当场炸的名字错误（pyflakes）", t_static_check_no_crash_classes),
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
