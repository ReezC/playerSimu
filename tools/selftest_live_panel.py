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

from execution import aim_convert as ma_c
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                          # noqa: E402

# ⭐⭐ **整个套件只建一次 `QApplication`，而且用模块级名字一直握着**（2026-10-10 ✓）：
#   各条用例原来都写 `app = QApplication.instance() or QApplication([])` 并赋给**局部变量** ✗
#   ⇒ 那个 app 一出作用域就被回收，而它名下建的控件还活着 ✗ ⇒ 到后面**再建控件**时 Qt 直接
#   fail-fast（`0xC0000409` ✓ **连 Python traceback 都没有** ✗）—— 本套件 2026-10-10 就在
#   新加的一条用例上撞到（跑到那条**静默死掉、连通过数都不打** ✓）。
#   ⚠ 同族的坑本仓库踩过多次（`selftest_decision` / `selftest_main_window` 都是这么修的 ✓
#     见它们的 `_APP` ✓）⇒ 这里也用**模块级**握着 ✓（各条用例里的 `QApplication.instance()`
#     照样拿到它 ✓ 不用改 ✓）。
from PyQt5.QtWidgets import QApplication as _QApp             # noqa: E402

_APP = _QApp.instance() or _QApp([])                          # noqa: F841

ROOT = Path(__file__).resolve().parent.parent


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _frame(n):
    """造一张"一眼能认出是第几帧"的图（左上角像素的蓝通道 = n）。"""
    img = np.zeros((60, 80, 3), np.uint8)
    img[:, :, 0] = n % 256
    return img


def _feed(p, img, raw=None):
    """按**真线程的顺序**喂一帧：先**原生帧**、再**显示帧**（两个信号，见 live_thread）。

    真线程里是两次队列信号（`raw_frame_ready` → `frame_ready`）；测试直接调槽 ✓。
    `raw=None` ⇒ 用 `img` 自己（线程里"没画东西"时就是同一个对象：收画面阶段 ✓）。
    """
    p._on_raw_frame(img if raw is None else raw)
    p._on_frame(img)


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
            # ⚠ 原生帧故意用**另一个**像素值（10 + i ⇒ 和显示帧 i 分得开）：
            #   `current_frame()` 必须是**原生**那份，不能被显示帧盖掉 ✓
            _feed(p, last, raw=_frame(10 + i))
        check(p._disp_merged == 9,
              "连推 10 帧应当合并掉 9 帧，实际合并 %d 帧" % p._disp_merged)
        check(made["n"] == 0,
              "事件循环还没跑就画了 %d 次（说明没合并，直接每帧都画）" % made["n"])
        cur = p.current_frame()
        check(cur is not None and int(cur[0, 0, 0]) == 19,
              "current_frame() 必须是**最新那帧的原生帧**（探针标定/血条框选靠它）："
              "拿到的是 %r（19 = 原生、9 = 显示帧的行号值）"
              % (None if cur is None else int(cur[0, 0, 0])))

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
        _feed(p, _frame(1))
        app.processEvents()
        drew = p._disp_drawn
        p.hide()
        app.processEvents()
        for _ in range(3):
            _feed(p, _frame(11))
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


def t_lag_watchdog_rules():
    """积压锁死看门狗：**持续**高才报、回落自己消失、迟滞区不闪也不乱计时。

    实测（2026-09-26）：B 机读线程跟不上 A 的推流速率时，积压堆在 ffmpeg/内核里长，
    端到端延迟稳在 2.6~3.9 秒**且自己回不来**（perf.log 连续 6 段都是这样，
    而机器并不忙：avail 5.7GB、mp_workers 0）。所以判据必须是"持续"而不是"高过一下"。

    为什么不用 `recv_fps < 名义帧率`：那是**间接**证据，会天天误报 —— 健康段 recv
    也是 117~121，而容器名义帧率是 144（A 机本来就没发满）。
    """
    from gui.live_thread import (LAG_RECOVER_MS, LAG_WARN_MS, LAG_WARN_SEC,
                                lag_watchdog)

    st = [None, None, False]
    # ① 只是抖一下（3 秒）→ 不许报
    check(lag_watchdog(900.0, 100.0, st)[0] == "", "刚高一下（首拍）就报警")
    check(lag_watchdog(900.0, 103.0, st)[0] == "",
          "不到 %.0f 秒就报警了" % LAG_WARN_SEC)
    # ② 持续够久 → 报，并且只"刚刚判定"一次（perf 记一次，不刷屏）
    txt, fresh = lag_watchdog(900.0, 100.0 + LAG_WARN_SEC + 0.1, st)
    check(txt and "锁死" in txt, "持续高却没报警：%r" % txt)
    check(fresh is True, "第一次判定没标成'刚刚'（perf 记不到那一下）")
    txt2, fresh2 = lag_watchdog(950.0, 100.0 + LAG_WARN_SEC + 0.6, st)
    check(txt2 and fresh2 is False, "判定被重复算'刚刚'（perf 会刷屏）")
    check("峰值" in txt2, "告警里没有峰值：%r" % txt2)
    # ③ 迟滞区（回落到 300ms）→ 已判定过就**继续挂着**，不闪
    check(lag_watchdog(LAG_RECOVER_MS + 100.0, 120.0, st)[0] != "",
          "迟滞区里告警闪掉了（会一闪一闪）")
    # ④ 真回落 → 清空状态，下次重新计时
    check(lag_watchdog(90.0, 200.0, st)[0] == "", "回落了还挂着告警")
    check(st[0] is None and st[2] is False, "回落没清状态：%s" % st)
    check(lag_watchdog(900.0, 201.0, st)[0] == "", "清空后没重新计时")
    # ⑤ 迟滞区**且没判定过**：不许靠"在 400ms 上耗 20 秒"骗过判据
    st2 = [None, None, False]
    check(lag_watchdog(LAG_RECOVER_MS + 200.0, 300.0, st2)[0] == "", "400ms 就报警")
    check(lag_watchdog(LAG_RECOVER_MS + 200.0, 340.0, st2)[0] == "",
          "在迟滞区耗久了也被判成锁死：%s" % st2)
    check(st2[0] is None, "迟滞区还在计时（判据边界没守住）：%s" % st2)
    # ⑥ 探针没开 / 拿不到延迟 → 不报（宁可不报，别误报）
    check(lag_watchdog(None, 999.0, [None, None, False])[0] == "",
          "拿不到延迟时乱报")
    # 边界自证：三个阈值必须是有序的，改坏了上面这些区间就全乱
    check(0 < LAG_RECOVER_MS < LAG_WARN_MS and LAG_WARN_SEC > 0,
          "阈值不合法：恢复 %s / 告警 %s / 时长 %s"
          % (LAG_RECOVER_MS, LAG_WARN_MS, LAG_WARN_SEC))


def t_lag_warn_on_status():
    """积压锁死要**顶在状态行最前面**（比负载告警还优先），恢复后自己消失。

    为什么优先级要压倒负载告警：负载告警是"为什么变慢"的解释，而积压锁死是一个
    **已经发生的故障状态**（延迟几秒、自己回不来），不处置会一直是坏的。
    """
    app, lp, p, made, orig = _panel()
    try:
        base = {"size": (1366, 768), "recv_fps": 101.0, "proc_fps": 40.0,
                "dropped": 3000, "show_fps": 30.0, "boxes": 3, "probe_on": True,
                "delay_ms": 3100.0}
        p._on_stats(dict(base, lag_warn="延迟积压锁死：3100 ms 已持续 12 秒",
                         lag_detail="读线程 101.0 fps ｜ 主回路 40.0 fps\n怎么办："
                                    "1) 停止再开始预览 2) A 机降到 60fps 档"))
        txt = p.lbl_stats.text()
        check("积压锁死" in txt, "状态行没显示积压告警：%r" % txt)
        check(txt.lstrip().startswith("【延迟积压锁死"),
              "积压告警没被顶到最前面：%r" % txt[:60])
        tip = p.lbl_stats.toolTip()
        check("读线程 101.0 fps" in tip and "停止再开始预览" in tip,
              "明细（含怎么办）没进 tooltip：%r" % tip)

        # 与负载告警同时出现时：积压在前，负载在后，两条都在
        p._on_stats(dict(base, lag_warn="延迟积压锁死：3100 ms 已持续 12 秒",
                         load_warn="负载告警：并行子进程 11 个(11.8GB)"))
        txt2 = p.lbl_stats.text()
        check(txt2.lstrip().startswith("【延迟积压锁死") and "负载告警" in txt2,
              "两条告警没有按优先级并存：%r" % txt2[:120])

        # 恢复（延迟回落）→ 告警自己消失
        p._on_stats(dict(base, delay_ms=85.0))
        check("积压锁死" not in p.lbl_stats.text(),
              "延迟恢复后状态行还挂着告警：%r" % p.lbl_stats.text())
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

        _feed(p, frame)
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

        # ★ 「浓淡」那一格**真的作用在像素上**（用户 2026-09-26 要确认"拖动条是否生效"）。
        #   上面那条只证明了 alpha=1 会画；这里钉住**0 和中间值**：
        #   · 0（滑块最左端 = 看不见）⇒ 那块必须还是原本的红底，不能还透出绿 ✗；
        #   · 50 ⇒ 红绿混出来 —— 证明是**按比例混**，不是"要么全画要么不画"（那样两种
        #     情形都能骗过只测 0/1 的用例 ✗）。
        p.set_minimap_overlay(green, None, rect, 0.0)
        c0 = p.view.pixmap().toImage().pixelColor(cx, cy)
        check(c0.red() > 150 and c0.green() < 80,
              "浓淡 0（最左端 = 看不见）却还画着：rgb(%d, %d, %d)"
              % (c0.red(), c0.green(), c0.blue()))
        p.set_minimap_overlay(green, None, rect, 0.5)
        c5 = p.view.pixmap().toImage().pixelColor(cx, cy)
        check(60 < c5.green() < 200 and c5.red() > 60,
              "浓淡 50%% 没混色（该是半绿半红底）：rgb(%d, %d, %d)"
              % (c5.red(), c5.green(), c5.blue()))
        p.set_minimap_overlay(green, None, rect, 1.0)   # 还原：下面几条按 alpha=1 写的

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


def t_vision_box_updates_every_frame():
    """视野框（那四条虚线）必须**每帧现算** —— 用户 2026-09-27 报"更新速度太慢"。

    根因：它原来是 `if time.perf_counter() - _vision_last[0] >= 3.0:` **每 3 秒**才算一次 ✗
    —— 角色/镜头一直在动，而框 3 秒才动一下 ⇒ 看上去就是"卡住的虚线"。
    `_vision_box_for` 只是几次整数加减（微秒级，比画那四条虚线还便宜）⇒ 没有理由节流 ✓。

    行为要真起线程 + 模型才看得到 ⇒ 这条钉**源码**（照 `t_raw_frame_before_draw` 的做法）：
      ① 那个 3 秒节流的变量（`_vision_last`）必须已经不在；
      ② 每帧那次调用必须在、而且**紧挨着它的上一行不许是时间比较**（再套回去就红 ✗）。
    """
    src = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    # ⚠ 只在**代码行**里找（注释里就写着"原来是每 3 秒"这句说明 —— 拿整份源码找会把
    #   说明文字本身判红 ✗，这个形状踩过好几次了）
    code = [l for l in src.splitlines() if not l.strip().startswith("#")]
    joined = "\n".join(code)
    check("_vision_last" not in joined,
          "还留着 `_vision_last`（= 那个「每 3 秒算一次」的节流器）⇒ 视野框又会卡 ✗")
    i = next((n for n, l in enumerate(code)
              if "_vision_box[0] = _vision_box_for(" in l), None)
    check(i is not None, "找不到每帧计算视野框那一句（`_vision_box[0] = _vision_box_for(...)`）")
    prev = code[i - 1].strip() if i else ""
    check(not prev.startswith("if time.perf_counter()"),
          "视野框又被套进「每 N 秒算一次」里了（上一行：%r）⇒ 虚线会卡住 ✗" % prev)
    check("def _vision_box_for" in src,
          "`_vision_box_for` 没了（它是纯整数运算，不该被别的东西换掉 ✗）")


def t_det_vision_only_crop():
    """⭐⭐⭐ 「**只检测视野框**」（用户 2026-10-15 ✓ 原话："我们是否可以只让YOLO检测视野框
    剪裁的画面？这样能省不少性能"✓）。

    这条钉的是一件**哑巴亏**：剪裁之后模型给回来的框是"**剪裁块自己的坐标系**"（原点在块左上角）
    ✗ —— 偏移忘了加回去，界面上"看着能跑"、但**所有框整体偏一块** ✗（画框 / 追踪 / 掉落拾取
    全错，而且**不报任何错** ✓ 最难查的就是这一种）。

    钉五件：
      ① **关**（默认）⇒ 喂**原图** + 零偏移（**老行为一字不变** ✓）；
      ② **开** ⇒ 喂的是剪裁块（形状 = 框的高 × 宽 ✓ 且是**视图、没白拷内存** ✓）
         + 偏移 = 框左上角 `(left, top)` ✓；
      ③ **退化框**（宽或高 ≤ 0）⇒ **退回原图** ✓ —— `vision_*` 是用户可填的，
         `left == right` 真会出现 ✓；空数组喂进 ultralytics 会崩在很深的地方、
         报错还看不出跟这个设置有关 ✗；
      ④ 偏移**加回坐标** ⇒ 与"全图检测"的坐标一致 ✓；
      ⑤ **源码**：偏移只在"刚从 GPU 拿回来"那**一处**加 ✓、且主回路那次 `predict` 喂的
         确实是剪裁块 ✓（开关白勾 / 加两处都会红 ✓）。
    """
    from gui import live_thread as lt

    vis = np.zeros((300, 400, 3), np.uint8)
    vis[100:140, 120:200, 0] = 200          # 在视野框里点一块 ⇒ 好认出"喂的是哪一块" ✓

    # ① 关（默认）⇒ 原图 + 零偏移
    _src, _off = lt._det_crop_for(vis, (100, 50, 350, 250), False)
    check(_src is vis and _off == (0, 0),
          "关着的时候必须喂**原图**、偏移 (0,0)（老行为一字不变 ✗）：%r" % (_off,))

    # ② 开 ⇒ 剪裁块（形状 / 内容 / 是视图）+ 偏移
    _src, _off = lt._det_crop_for(vis, (100, 50, 350, 250), True)
    check(_src.shape[:2] == (200, 250),
          "剪出来的不是「框的高 × 宽」= (200, 250)：%r" % (_src.shape,))
    check(_off == (100, 50), "偏移不等于框左上角 (100, 50)：%r" % (_off,))
    check(bool(np.all(_src[50:90, 20:100, 0] == 200)),
          "剪出来的**不是框里那一块**（内容对不上 ⇒ 剪错位置了 ✗）")
    check(not _src.flags.owndata,
          "剪裁块**拷了内存**（`vis[top:bottom, left:right]` 该是**视图** —— "
          "每帧白拷 ~1MB 没必要 ✗）")

    # ③ 退化框 / 没框 ⇒ 退回原图（**不许**把空图喂给模型 ✗）
    for _bad in ((100, 50, 100, 250), (100, 50, 350, 50), (100, 50, 90, 40)):
        _s2, _o2 = lt._det_crop_for(vis, _bad, True)
        check(_s2 is vis and _o2 == (0, 0),
              "退化框 %r 没退回原图（把空图喂进模型会崩在很深的地方 ✗）：%r"
              % (_bad, _o2))
    _s3, _o3 = lt._det_crop_for(vis, None, True)
    check(_s3 is vis and _o3 == (0, 0), "框是 None 时没退回原图 ✗")

    # ④ 偏移加回坐标 ⇒ 与「全图检测」等价（下游拿到的必须是**全画面坐标** ✓）
    _xyxy = np.array([[10.0, 10.0, 50.0, 40.0]])          # 模型在**剪裁块**里给的框
    _back = lt._det_off_xyxy(_xyxy, _off)
    check(_back[0].tolist() == [110.0, 60.0, 150.0, 90.0],
          "偏移加回之后不是全画面坐标（画框 / 追踪 / 掉落拾取会整体偏一块 ✗）：%r"
          % (_back[0].tolist(),))
    check(lt._det_off_xyxy(_xyxy, (0, 0)) is _xyxy,
          "零偏移时不该动那个数组（没剪裁那条路每帧都白算一遍 ✗）")

    # ⑤ 源码：加偏移**只有一处**、主回路喂的**是剪裁块**
    # ⚠ 2026-10-20：这一段（含 `_det_crop_for` / `_det_off_xyxy`）已搬进
    #   `runtime/frame_infer.py`（阶段 1 第二刀 ✓）⇒ 源码钉要读**那个文件** ✓
    #   （`lt._det_crop_for` 仍然取得到 ✓ 那边是 re-export ✓）
    src = (ROOT / "runtime" / "frame_infer.py").read_text(encoding="utf-8")
    code = [l for l in src.splitlines() if not l.strip().startswith("#")]
    check(sum(1 for l in code if "xyxy = _det_off_xyxy(xyxy, _det_off)" in l) == 1,
          "「把偏移加回 `xyxy`」那一句不在了、或不止一处 ⇒ 剪裁开着时框会整体位移 ✗")
    check(sum(1 for l in code if "res = model.predict(_det_src," in l) == 1,
          "主回路那一次推理喂的**不是**剪裁块（`_det_src`）⇒ 开关白勾 ✗")
    check("def _det_crop_for" in src, "`_det_crop_for` 没了（剪裁那一步没了 ✗）")
    check("frame_infer.step(" in (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8"),
          "`_run` 里没调 `frame_infer.step(`（推理段搬完没接上 ⇒ 一帧都不推理 ✗）")


def t_vis_color_alpha_and_toggles():
    """「外观 → 辅助线与标记」那一组的新行为（用户 2026-09-27 一次提的三条）。

    ① 颜色**支持透明度**（弹窗里有拖动条 ⇒ 存的是 `#AARRGGBB`）⇒ 校验/解析两条路都得认，
       否则"配好的透明度下次读回来就没了"（会被当非法值静默退回默认色 ✗）；
    ② 那一组里**每项前面有开关**（`*_on`）⇒ 关掉 = 不画这一项，颜色留着 ✓；
    ③ 两项"攻击距离线颜色"换成**三个框**（攻击范围框 / 攻击盲区框 / 跳跃攻击范围框）✓，
       名字末尾的「颜色」两字去掉 ✓。

    钉五件（都在会真出错的地方）：
      · `_valid_color` 收 7 位与 9 位、拒明显坏值；
      · `hex_to_bgra` 的**通道顺序**是 (b, g, r, a) —— 写反了整幅画面的颜色都会变 ✗；
        `hex_to_bgr` 必须与它一致（老调用方一堆 ✓）；
      · `load_vis` 的开关键缺省/写坏都算 **True**（= 画 = 老行为 ✓），
        `jump_attack_color` 也存在（占位项也要有 ✓）；
      · `_blit_alpha`：不透明**原样直画**、半透明按 alpha 混、混不到框外 ✓；
      · 设置界面源码：老名字一个都不许留、六项各带自己的开关 ✓。
    """
    import cv2

    from gui import live_thread as lt
    from gui import theme

    # ① 颜色合法性：7 位（老配置）与 9 位（带透明度）都收
    check(theme._valid_color("#f9ab00") and theme._valid_color("#80f9ab00"),
          "颜色校验收不下 `#AARRGGBB`（配好的透明度下次读回来会丢 ✗）")
    check(not theme._valid_color("#f9ab0") and not theme._valid_color("f9ab00"),
          "颜色校验把明显的坏值放进来了")

    # ② 通道顺序（BGR + alpha）
    check(theme.hex_to_bgra("#f9ab00") == (0, 171, 249, 255),
          "`hex_to_bgra` 不是 (b, g, r, a)：%r" % (theme.hex_to_bgra("#f9ab00"),))
    check(theme.hex_to_bgra("#8000abf9") == (249, 171, 0, 128),
          "带 alpha 的解析不对（该 (249, 171, 0, 128)）：%r"
          % (theme.hex_to_bgra("#8000abf9"),))
    check(theme.hex_to_bgr("#8000abf9") == (249, 171, 0),
          "`hex_to_bgr` 与 `hex_to_bgra` 不一致（老调用方会拿到错的通道 ✗）")

    # ③ 开关默认「画」+ 占位项的颜色键在
    vis = theme.load_vis()
    for k in ("lock_on", "attack_on", "min_attack_on", "jump_attack_on",
              "chase_jump_on", "vision_on", "timer_on"):
        check(vis.get(k) is True,
              "「%s」默认该是 True（= 画，老配置里没有这些键 ✓）：%r" % (k, vis.get(k)))
    check("jump_attack_color" in vis,
          "「跳跃攻击范围框」的颜色键没进配置（占位项也要能配颜色 ✓）")
    check("chase_jump_color" in vis,
          "「追击起跳框」的颜色键没进配置（它现在是个框，颜色不能再写死在 live_thread ✗）")

    # ④ `_blit_alpha`：不透明直画 / 半透明混色 / 不越界
    img = np.zeros((20, 20, 3), np.uint8)
    lt._blit_alpha(img, (0, 0, 255, 255),
                   lambda t, c: cv2.rectangle(t, (0, 0), (9, 9), c, -1))
    check(tuple(int(v) for v in img[0, 0]) == (0, 0, 255),
          "不透明时该原样画上去（零开销那条路）：%r" % (tuple(int(v) for v in img[0, 0]),))
    img2 = np.zeros((20, 20, 3), np.uint8)
    lt._blit_alpha(img2, (0, 0, 255, 128),
                   lambda t, c: cv2.rectangle(t, (0, 0), (9, 9), c, -1))
    _v = int(img2[0, 0, 2])
    check(110 <= _v <= 145, "半透明混色不对（该 ≈128）：%r" % _v)
    check(tuple(int(x) for x in img2[0, 15]) == (0, 0, 0),
          "混色的结果溢到图形外面了 ✗")

    # ⑤ 设置界面源码（控件那层在 settings_dialog 的 `_page_appearance` 里，建整个对话框太重）
    src = (ROOT / "gui" / "settings_dialog.py").read_text(encoding="utf-8")
    for bad in ('"锁定框颜色"', '"最大攻击距离线颜色"', '"最小攻击距离线颜色"',
                '"视野线颜色"', '"定时任务颜色"'):
        check(bad not in src,
              "「辅助线与标记」里还留着老名字 %s（用户要求去掉「颜色」两字 ✗）" % bad)
    for name, key in (("锁定框", "lock_on"), ("攻击范围框", "attack_on"),
                      ("攻击盲区框", "min_attack_on"),
                      ("跳跃攻击范围框", "jump_attack_on"),
                      ("追击起跳框", "chase_jump_on"),
                      ("视野线", "vision_on"), ("定时任务", "timer_on")):
        check('"%s"' % name in src and 'on_key="%s"' % key in src,
              "「%s」这一项没有做出来 / 没带自己的开关 %s ✗" % (name, key))
    # ⑥ 叠图的**摆法**（源码约定：那些框在 live_thread 的收流循环里画，行为要真起线程 +
    #    YOLO 才测得到 ⇒ 这里钉源码 ✓）：
    #    用户 2026-09-27 的图上三个框是**并排、首尾相接**的
    #    （`玩家 │ 攻击盲区框 │ 攻击范围框 │ 追击起跳框`）✓
    lsrc = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    check("from_d=min_ad" in lsrc,
          "「攻击范围框」没带 `from_d=min_ad` ⇒ 会从角色中心一路铺过去，和盲区框重叠 ✗"
          "（图上两者是并排的）")
    check("from_d=max_ad + jlo" in lsrc,
          "「追击起跳框」没按区间画（该是 [最大 + min, 最大 + max] ✓）")
    # ⭐⭐ 2026-10-12：绿框的**上边**要抬到「跳跃攻击的高度」（用户原话："追击起跳框的判定距离
    #   需要**向上延伸到跳跃攻击的高度**"✓）—— 上边**只有一处**算：`agent.chase_jump_ceiling`
    #   （判据 `agent._in_chase_jump_box` 用的**同一条上边** ✓）⇒ live_thread 里**不许**自己
    #   加一个高度 / 自己抄一份 ✗（各算一套 ⇒ 画的和判的迟早对不上 ✓）。
    check("chase_jump_ceiling" in lsrc,
          "「追击起跳框」没跟上边那条口径（`agent.chase_jump_ceiling` ✗）—— 画出来的绿框会停在"
          "攻击框上边，而判据（`_in_chase_jump_box`）已经抬到跳跃高度 ⇒ **画的和判的对不上** ✓")
    check("_in_chase_jump_range" not in lsrc,
          "live_thread 里还留着老那句「它的判定仍是水平距离（`_in_chase_jump_range`）」✗ ——"
          "2026-10-12 起判定已经是 `_in_chase_jump_box`（含上边那条高度 ✓）⇒ 那句话现在是错的 ✓")
    check("_chase_jump_color" in lsrc and "_on_chase" in lsrc,
          "「追击起跳框」没用设置里的颜色/开关 ⇒ 又写死回 live_thread 了 ✗")
    check("(0, 200, 0)" not in lsrc,
          "live_thread 里还有写死的那个绿（追击起跳的颜色该只在 `gui/theme.py` 一处 ✗）")
    # ⭐⭐ 「**跳跃攻击范围框**」**2026-10-12 起真的会画了**（用户那天定了：攻击范围框上边
    #   再往上「跳跃高度」那一条**白框** ✓ 见 `decision/agent.py::jump_attack_box_poly` ✓）
    #   ⇒ 老钉子（"逻辑还没做 ⇒ **不该画**"✗）**作废** ✓，改成钉新的两件：
    #     ① 颜色 / 显示开关走设置（不再写死在 live_thread ✓）；
    #     ② 形状与"成不成立"**只有一处**（`jump_attack_box_poly` / `jump_attack_active` ✓）
    #        ⇒ live_thread 里**不许**自己算高度 / 四条边 ✗（各算一套 ⇒ 框和判据迟早对不上 ✓）。
    check("_jump_attack_color" in lsrc and "_on_jump_attack" in lsrc
          and "jump_attack_color" in lsrc,
          "「跳跃攻击范围框」没用设置里的颜色/开关 ⇒ 又写死回 live_thread 了 ✗"
          "（用户 2026-10-12 ✓ 见 `gui/theme.py` 的 `jump_attack_color` / `jump_attack_on` ✓）")
    check("jump_attack_box_poly" in lsrc and "jump_attack_active" in lsrc,
          "「跳跃攻击范围框」没走 `agent.jump_attack_box_poly` / `jump_attack_active` ✗ ——"
          "形状与「成不成立」必须一处算 ✓（自己算一遍 ⇒ 画出来的框和判据迟早对不上 ✓）")

    check("QColorDialog.ShowAlphaChannel" in src
          and "QColorDialog.DontUseNativeDialog" in src,
          "颜色弹窗没有开透明度拖动条（`ShowAlphaChannel` 缺了就没法调透明 ✗；"
          "`DontUseNativeDialog` 缺了在 Windows 上会走**系统**弹窗、同样没有 alpha 条 ✗）")
    check("HexArgb" in src,
          "选完颜色没按 `#AARRGGBB` 存 ⇒ 透明度当场丢掉 ✗")


def t_current_frame_is_raw():
    """`current_frame()` 必须是**原生帧**：显示帧（画着检测框/视野虚线）不许进那条路。

    用户 2026-09-27 现场报的 bug：**框选小地图时，视野那条灰色虚线也被框进去** ——
    虚线/检测框都是收流线程在帧上**原地**画的，而框选拿的原来是同一份 ✗。修法是把两路
    分开（`raw_frame_ready` / `frame_ready`）。这条钉三件：

      ① 行为：先喂原生、再喂显示 ⇒ `current_frame()` 是**原生**那份（不是显示那份 ✗）；
      ② 行为：面板**不可见**时原生帧也照更新（不然切回来一眼就框到旧画面 ✗）；
      ③ 源码：`live_panel.py` 里给 `_last_bgr` 赋值的地方**只有 `_on_raw_frame` 一处**
         （谁在 `_on_frame` 里再写一次，等于把 bug 写回来 ✗）。
    """
    app, lp, p, made, orig = _panel()
    try:
        raw = _frame(31)
        disp = _frame(97)          # 显示帧故意用能区分的像素值
        p._on_raw_frame(raw)
        p._on_frame(disp)
        cur = p.current_frame()
        check(cur is raw,
              "current_frame() 不是原生帧 —— 框选会框到画上去的框线/虚线（用户现场那个 bug）")
        check(int(cur[0, 0, 0]) == 31,
              "显示帧把原生帧盖掉了（拿到 %d，31 才是原生）" % int(cur[0, 0, 0]))

        p.hide()
        app.processEvents()
        raw2 = _frame(32)
        p._on_frame(_frame(98))    # 显示帧：不可见 ⇒ 连画都不画
        p._on_raw_frame(raw2)
        check(p.current_frame() is raw2,
              "不可见时原生帧没更新 —— 切回来看一眼再框选，框到的是旧画面 ✗")

        # ③ 源码：`_last_bgr` 的赋值只许在 `_on_raw_frame` 里
        src = (ROOT / "gui" / "live_panel.py").read_text(encoding="utf-8")
        body_raw = src.split("def _on_raw_frame", 1)[1].split("def _on_frame", 1)[0]
        body_disp = src.split("def _on_frame", 1)[1].split("def _draw_pending", 1)[0]
        check("self._last_bgr = " in body_raw,
              "`_on_raw_frame` 没存下原生帧（current_frame() 就没来源了）")
        # ⚠ 只找**赋值**（`self._last_bgr = `）：说明文字里就写着"别在这里写
        #   `_last_bgr = img`"，拿裸名字找会把**解释**一起判红（这个形状踩过好几次 ✗）
        check("self._last_bgr = " not in body_disp,
              "`_on_frame`（显示帧）里又在给 `_last_bgr` 赋值 —— 那就是把「框选框到视野"
              "虚线」那个 bug 写回来了 ✗")
        n = src.count("self._last_bgr = img")
        check(n == 1,
              "给 `_last_bgr` 写 img 的地方有 %d 处（只许 `_on_raw_frame` 那一处）" % n)
    finally:
        lp._bgr_to_pixmap = orig
        p.close()


def t_raw_frame_before_draw():
    """收流线程：**先留原生帧、再就地画** —— 且只有"真要画"时才拷（收画面阶段零开销）。

    这条是源码约定（行为在 `selftest_live_panel` 里验不了：要真起线程 + YOLO）：
      ① `raw = vis.copy()` 必须在**第一个** `cv2.*(vis` 之前（画过就回不去了 ✗）；
      ② 所有往 `vis` 上的绘制都在 `if self._infer.is_set():` 那一块里
         （不然"推理关着就不拷"这个前提不成立 ⇒ 原生帧会被污染 ✗）；
      ③ `raw_frame_ready.emit(raw)` 必须**在显示限流之外**（`if should_show:` 那层
         之外）—— 显示可以限流，取帧不许 ✗；
      ④ `frame_ready.emit(vis)` 仍在（显示那份不许断）。
    """
    src = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    # ⚠ 2026-10-20：**这一段已经搬进 `runtime/frame_pic.py`**（阶段 1「编排收口」第二刀 ✓）
    #   ⇒ 判据得**按文件**看，别再把两个文件拼起来比下标 ✗
    #   （拆块时最常踩的：只读 `live_thread.py` ⇒ 假红 ⇒ 有人顺手把判据删了 ✗✓）。
    pic = (ROOT / "runtime" / "frame_pic.py").read_text(encoding="utf-8")

    # ① 先拷再画（画是**原地**改数组 ⇒ 拷晚了原始像素就回不来了）
    LINE_COPY = "ctx.raw = ctx.vis.copy() if th._infer.is_set() else ctx.vis"
    check(LINE_COPY in pic,
          "`runtime/frame_pic.py` 里没按原来那一行留原生帧（框选会拿到画过的帧 ✗）")
    # ⚠ 判"有没有画"要**按绘制调用**看，不能查 `"cv2." in src` ✗ ——
    #   那段的说明里本来就要写 `cv2` 这两个字（注释/文档串会把判据骗红 ✓ 拆块老坑）。
    _DRAW_CALLS = ("cv2.rectangle(", "cv2.line(", "cv2.putText(", "cv2.circle(",
                   "cv2.arrowedLine(", "_draw_dashed_line(")
    _draw_lines = [ln.strip() for ln in pic.splitlines()
                   if ln.strip() and not ln.strip().startswith("#")
                   and any(k in ln for k in _DRAW_CALLS)]
    check(not _draw_lines,
          "`runtime/frame_pic.py` 里出现了绘制（那一段必须只读 `vis`、一个字都不画 ✗）：%r"
          % (_draw_lines[:2],))
    i_copy = src.find("frame_pic.step(self, ctx)")      # 接线点（不是拷贝点 ✓）
    check(i_copy > 0,
          "`_run` 里没调 `frame_pic.step(self, ctx)`（那段搬完没接上 ⇒ 原生帧/面板全没了 ✗）")
    draws = [src.find(kw) for kw in ("cv2.rectangle(vis,", "cv2.line(vis,",
                                     "cv2.putText(vis,", "cv2.circle(vis,",
                                     "cv2.arrowedLine(vis,", "_draw_dashed_line(vis,")]
    draws = [j for j in draws if j >= 0]
    check(draws, "一个往 `vis` 上的绘制都没找到（源码约定该改写了？）")
    # ⭐ 2026-10-05：画框代码搬进了 `_draw_show_frame`（用户要的 #1 ✓）⇒ 位置比较要
    #   **相对那个方法**来比 ✓（原来拿全文下标比 ✗ —— 方法定义在文件更前面 ⇒ 必然误报 ✗）。
    #   原意一字不变：**先拷原生帧、再画** ✓；而且是"循环里只调一次、画的全在方法里" ✓。
    i_def = src.find("def _draw_show_frame(")
    i_call = src.find("self._draw_show_frame(")
    check(i_def > 0 and i_call > i_def, "画框方法没抽出来 / 循环里没调它 ✗")
    check(i_copy < i_call,
          "循环里是先调画框、后拷原生帧（那画过的像素就回不去了 ✗）")
    in_def = [d for d in draws if d > i_def]
    check(len(in_def) == len(draws),
          "循环里还留着往 `vis` 上直接画（该全在画框方法里 ✗）")

    # ② 拷贝条件必须**就写在那一行上**：它假设"所有绘制都在下面那道推理门里"
    check(LINE_COPY in pic,
          "原生帧的拷贝条件不是那一行（`%s`）—— 换了条件就等于假设「别处也会画」✗"
          % LINE_COPY)
    i_gate = src.find("if should_show and draw and self._infer.is_set():")
    check(i_gate > i_def, "没找到推理那道门（画框方法里那句 `if ... self._infer.is_set():`）")
    check(all(i_gate < d for d in in_def),
          "有往 `vis` 上的绘制跑到推理门**外面**去了（原生帧会被污染 ✗）")
    check(src.count("self.raw_frame_ready.emit(raw)") == 1,
          "`raw_frame_ready.emit(raw)` 不止一处（取帧那条链要唯一 ✓）")
    check("self.frame_ready.emit(vis)" in src, "显示那份 `frame_ready.emit(vis)` 不见了 ✗")

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

    from core import probe_codec as pc
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

    from core import probe_codec as pc
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

        # ②b **对时偏置要摆出来**（用户 2026-09-26 要求）：延迟里**整段加着**它，
        #     它旧了/测偏了，延迟数就整体高或低那么多 —— 实测那次是 1255 ms 显示值
        #     对 1144.5 ms 偏置（链路其实只有百毫秒级 ✗），光看"锁死 1.25 秒"根本
        #     看不出来 ⇒ 摆出来才能一眼对上 ✓。
        p._on_stats(dict(base, delay_ms=1255.0, probe_mono=True, probe_jumpy=0,
                         probe_samples=20, clock_offset_ms=1144.5))
        txt2b = p.lbl_stats.text()
        check("1255" in txt2b and "1144" in txt2b,
              "延迟旁边没把对时偏置摆出来（这次就是 1255 ≈ 1144 才看出来的）：%r" % txt2b)
        tip2b = p.lbl_stats.toolTip()
        check("对时偏置" in tip2b and "clock_sync" in tip2b,
              "对时偏置的 tooltip 没说清怎么验证/重对：%r" % tip2b[:90])
        # 老调用方（没给这个键）不许崩、也不许瞎编一个偏置写上去
        p._on_stats(dict(base, delay_ms=142.0, probe_mono=True, probe_jumpy=0,
                         probe_samples=20))
        check("对时偏置" not in p.lbl_stats.text(),
              "没给偏置却把它写进状态行了：%r" % p.lbl_stats.text())

        # ③ 框选保存前那道闸：拿实时流跑几帧验单调性（画对了才放行）
        geo_ok = {"x": 20.0, "y": 20.0, "cell": 16.0, "gap": 2.0}
        geo_drift = dict(geo_ok, gap=1.5)          # 采样式偏 0.5px/块（实测那类偏差）
        bits = 40
        # ⚠⚠ **把"现在几点"钉死**（2026-10-13 ✓ 查这条用例为什么**偶尔红**）——
        #   ① 这里用 `pc.now_ms()` 合成码带 ✓，而判据里那道"解出的时刻要接近此刻"
        #      （`ts_plausible` ✓ 它内部**直接读真时钟** ✓）也在跑 ✓；
        #   ② 「偏 0.5px/块」（gap 2.0 → 1.5 ✓ 40 块累计偏 20px ≈ **一个方块节距** ✓）
        #      解出来**恰好是真值的 1/2**（读串了一格 ✓ —— 实测 2026-10-13：真值 1221635
        #      ⇒ 解出 610817 ✓），而步长**照旧单调**（16~17ms ✓）⇒ 单调性那道闸
        #      **抓不到它** ✗（"大跳"一次都没有 ✓ 也一次没"倒退" ✓）；
        #   ③ ⇒ 唯一能抓它的是那道 ±10 分钟窗口 ✗ ⇒ **能不能抓住全看当时几点** ✗：
        #      「真值 / 2」落在真时刻 ±10 分钟内 ⇔ 本地时间在 **00:20 之前 / 23:40 之后**
        #      ⇒ 那段时间跑就**误放行** ⇒ 用例红 ✓（2026-10-13 00:2x 就是这么红的 ✓；
        #      同一台机器 2026-10-12 23:xx 跑是绿的 ✓）。
        #   ⇒ 钉一个固定时刻（**正午** ✓）：「真值 / 2」离它 6 小时 ⇒ 谁也蒙不过 ✓
        #     这样一来，这条用例只验"判据对不对"，不再跟墙上的钟绑定 ✓。
        _real_now_ms = pc.now_ms
        pc.now_ms = lambda: 43_200_000            # 12:00:00.000（当天偏移毫秒 ✓）
        try:
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
            pc.now_ms = _real_now_ms
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


def t_limit_reason_rules():
    """「这一段卡在谁身上」的判据：输入受限 / 本机受限 / 说不清（纯函数，2026-09-26）。

    **为什么要它**：状态行上「输入 fps」和「处理 fps」是不丢帧时**必然相等**的两个数，
    于是"处理速度掉到 20"看着像 B 机算不动 —— 实测最慢那段其实是输入只有 34fps
    （`gap_ms` 中位 29.6ms，而本机一拍才 24ms，30fps 都还有余量）。
    判据必须拿**输入间隔**和**本机耗时**比，而不是那两个必然相等的 fps。
    """
    from gui.live_thread import limit_reason

    # ① 输入只有 34fps、本机一拍 13ms ⇒ 上游给的少（实测慢段的形状）
    check(limit_reason(34.0, 34.0, 13.0, 29.4) == "input",
          "输入受限没认出来：%r" % limit_reason(34.0, 34.0, 13.0, 29.4))
    # ② 收到 60 却只处理 40（丢帧在涨）⇒ 本机跟不上
    check(limit_reason(60.0, 40.0, 20.0, 16.7) == "self",
          "本机跟不上没认出来：%r" % limit_reason(60.0, 40.0, 20.0, 16.7))
    # ③ 两个都慢（本机 24ms、输入间隔 29.6ms）⇒ **说不清就别指方向**
    check(limit_reason(34.0, 34.0, 24.0, 29.6) == "",
          "两个都慢时乱指方向（会把人引去查错机器）：%r"
          % limit_reason(34.0, 34.0, 24.0, 29.6))
    # ④ 都不慢 / 还没数 ⇒ 不贴标签
    check(limit_reason(60.0, 60.0, 13.0, 16.7) == "", "都够快时贴了标签")
    check(limit_reason(0, 0, 0, 0) == "", "还没数就贴标签")


def t_src_lag_degrades_display():
    """⭐⭐ 「持续跟不上源 ⇒ 自动把显示刷新降档」（用户 2026-10-05 ✓ 原话："经常报负载警告和
    延迟挤压锁死，这时候画面就像被慢放了，操作也慢很多。如何能在这种情况下保持流畅性？"）。

    **为什么要它**（`perf.log` 实测 1724 个 30s 窗口）：e2e 中位 >500ms 的"锁死"窗口占 **13%**，
    形状是 `recv 53 fps / proc 26 fps`、`avail_gb 3.8`（健康段 `58 / 44`、`10.3`）
    ⇒ **本机跟不上输入 ⇒ 上游（解码队列 / 内核缓冲）只增不减 ⇒ 延迟锁死** ✓。
    仓库里"报"已经有了（`lag_watchdog` + `limit_reason` + 一段"怎么办"）✓，但**没有任何自动
    动作** ✗ —— 这里补的就是那一步，而且只做最保守的一件：**压显示刷新**（少画 ⇒ 少抢 CPU ⇒
    读线程与主回路反而更快 ✓）。钉三件：
      ① 判据（纯函数）：持续 10 秒才算 / 9 秒不算 / 恢复要连续 5 秒 / **忽真忽假不闪** ✓；
      ② 接线（源码钉）：判据**复用** `limit_reason`（不另抄阈值 ✗）、档位算出来真的写进
         `_show_iv`、限流那句用**实际生效**的间隔、换档各留一条 `perf.note` ✓；
      ③ 面板（真喂一份 stats）：短句写「已自动降预览」、明细进 tooltip、恢复后**自己消失** ✓。
    """
    from gui.live_thread import (PREVIEW_FPS_MIN, PREVIEW_HOLD_S, PREVIEW_TIER_MID,
                                 SRCLAG_CLEAR_SEC, SRCLAG_SEC, preview_fps_target,
                                 src_lag_signal, src_lag_watchdog)

    # ①′ ⭐⭐ **"跟不上源"不能只看速率**（2026-10-10 现场 ✓）：主回路**故意丢帧**（只留最新 ✓）
    #    ⇒ `proc < recv` 是**常态**⇒ 只看它会在健康段一直判落后 ⇒ 预览被**永久**压在最低档
    #    ⇒ 用户喊"卡得很"✗（实测那时端到端只有 **131 ms** ✓ 而 `src_lag=on` ✗）。
    check(src_lag_signal(True, 131.0) is False,
          "延迟才 131ms（健康 ✓）却还判「跟不上源」⇒ 预览被永久压到最低档 ✗")
    check(src_lag_signal(True, 900.0) is True, "延迟锁死时没判落后 ✗")
    check(src_lag_signal(False, 900.0) is False, "速率够快时只凭延迟判落后（会误降档 ✗）")
    check(src_lag_signal(True, None) is True, "探针没开时没退回「只看速率」✗")

    # ① 纯函数
    st = [None, False, None]
    check(src_lag_watchdog(True, 0.0, st) == (False, False), "第一拍就判落后了 ✗")
    check(src_lag_watchdog(True, SRCLAG_SEC - 0.1, st) == (False, False),
          "还没够 %.0f 秒就判落后了 ✗" % SRCLAG_SEC)
    check(src_lag_watchdog(True, SRCLAG_SEC, st) == (True, True), "够秒数了没判定 ✗")
    check(src_lag_watchdog(True, SRCLAG_SEC + 5, st) == (True, False),
          "判定之后又报一次（该一直挂着、只报一次 ✓）✗")
    # 忽真忽假（健康段也会抖）⇒ 计时器清掉、不闪
    st2 = [None, False, None]
    src_lag_watchdog(True, 0.0, st2)
    src_lag_watchdog(False, 5.0, st2)
    check(src_lag_watchdog(True, 9.0, st2) == (False, False),
          "忽真忽假还把计时接着算（会一闪一闪地降档 ✗）")
    check(src_lag_watchdog(True, 19.1, st2) == (True, True), "重新计满没判定 ✗")
    # 恢复要连续 CLEAR 秒
    check(src_lag_watchdog(False, 20.0, st2) == (True, False), "刚不成立就撤销了 ✗")
    check(src_lag_watchdog(True, 21.0, st2)[0] is True, "抖一下就撤销了 ✗")
    check(src_lag_watchdog(False, 22.0, st2) == (True, False), "恢复计时该重来 ✗")
    check(src_lag_watchdog(False, 22.0 + SRCLAG_CLEAR_SEC, st2) == (False, False),
          "连续 %.0f 秒不成立还没撤销 ✗" % SRCLAG_CLEAR_SEC)
    check(src_lag_watchdog(False, 0.0, [None, False, None]) == (False, False),
          "一次都没判过就报落后（凭空降档 ✗）")

    # ①b **预览刷新的动态档位**（用户 2026-10-05 ✓ "够用的时候希望能流畅点"）：三档 + 迟滞
    pv = [None, 0.0]
    check(preview_fps_target(50.0, 60.0, "", False, 0.0, pv) == 50.0,
          "够用（延迟 60ms、无负载告警）时没用满用户配的上限（不够流畅 ✗）")
    check(preview_fps_target(50.0, 300.0, "", False, 0.1, pv) == 25.0,
          "有点吃紧（延迟 300ms）没降到半速：%r" % pv)
    # ⭐⭐ 用户 2026-10-10 口径修正（原话："**健康 肯定 用满**"✓）：**探针开着**时**以延迟为准** ✗ ——
    #   延迟 60ms（健康）而"别人开着别的程序"（负载告警）⇒ **不许**因此砍一半 ✓
    #   （实景：延迟 131ms 却把预览压到 12fps ⇒ 人喊"卡得很"✗）。
    pv_up = [None, 0.0]
    check(preview_fps_target(50.0, 60.0, "负载告警：内存 2.2/31.8GB", False, 0.0, pv_up) == 50.0,
          "延迟健康（60ms）却因为「别的程序占着机器」就砍一半 ✗（用户：「健康 肯定 用满」✓）")
    # 探针**没开**（delay=None）⇒ 没有延迟可比 ⇒ 退回"负载告警 ⇒ 半速"（老口径 ✓ 不然就瞎了 ✓）
    pv_lw = [None, 0.0]
    check(preview_fps_target(50.0, None, "负载告警：内存 2.2/31.8GB", False, 0.0, pv_lw) == 25.0,
          "探针没开又无延迟可比时，负载告警没兜住（老口径丢了 ✗）")
    # 降档**立刻**生效（升档才要等）
    check(preview_fps_target(50.0, 900.0, "", True, 0.3, pv) == PREVIEW_FPS_MIN,
          "锁死/跟不上源时没降到最低档：%r" % pv)
    # 升档要稳定 PREVIEW_HOLD_S 秒（防抖：档位一跳一跳比不降更难受 ✗）
    check(preview_fps_target(50.0, 60.0, "", False, 0.3 + PREVIEW_HOLD_S - 0.1, pv)
          == PREVIEW_FPS_MIN, "刚够用一瞬就升回去了（会一闪一闪 ✗）：%r" % pv)
    check(preview_fps_target(50.0, 60.0, "", False, 0.3 + PREVIEW_HOLD_S, pv) == 50.0,
          "稳定够用了却不升回去（那就永远慢着了 ✗）：%r" % pv)
    # 中间档 = show_fps 的这个比例（用户改上限 ⇒ 跟着变 ✓）
    pv2 = [None, 0.0]
    check(abs(preview_fps_target(30.0, 300.0, "", False, 0.0, pv2)
              - 30.0 * PREVIEW_TIER_MID) < 1e-6,
          "中间档没跟着用户配的上限走 ✗")
    # 探针没开（delay 为 None）⇒ 靠"负载告警 / 看门狗"判，不猜延迟
    pv3 = [None, 0.0]
    check(preview_fps_target(50.0, None, "", False, 0.0, pv3) == 50.0,
          "探针没开时凭空空降（够用就该流畅 ✓）")

    # ② 源码钉
    src = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    check("src_lag_watchdog(" in src, "看门狗没接线 ✗")
    check('limit_reason(_recv_w, _proc_w, _infer_med, _gap_med) == "self"' in src,
          "降档判据没复用 `limit_reason`（另抄一套阈值必然漂 ✗）")
    check("preview_fps_target(show_fps, _med_delay, self._load[0]," in src,
          "预览档位没按负载接线（动态刷新是空的 ✗）")
    check("_show_iv[0] = 1.0 / max(1.0, _pv_fps)" in src,
          "算了档位却没真把显示间隔压下去（降档是空的 ✗）")
    check("now - t_last_show >= _show_iv[0]" in src,
          "限流那句没用**实际生效**的间隔（压了也不生效 ✗）")
    check('perf.note("src_lag"' in src, "进出降档没留痕 ✗")
    check('perf.note("preview_fps"' in src, "换档没留痕 ✗")
    check('"src_lag": bool(_lagging)' in src, "状态里没带 src_lag ✗")
    check('"preview_fps": round(float(_pv_fps), 1)' in src,
          "状态里没带 preview_fps（面板没法说清降到多少 ✗）")

    # ③ 面板：短句 + tooltip + 恢复自清
    app, lp, p, made, orig = _panel()
    try:
        base = {"size": (1366, 768), "recv_fps": 53.1, "proc_fps": 26.3,
                "dropped": 2, "show_fps": 15.0, "boxes": 3, "probe_on": False,
                "infer_ms": 22.7, "limit": "self", "gap_med": 18.0}
        p._on_stats(dict(base, src_lag=True, preview_fps=12.0, preview_capped=True,
                         src_lag_detail="主回路持续跟不上输入（处理 26.3 fps vs "
                                        "输入 53.1 fps）⇒ 预览刷新已自动降到 12 fps"))
        txt = p.lbl_stats.text()
        check("预览已自动降到 12 fps" in txt,
              "降档了状态行没说清降到了多少（人只会以为帧率掉了 / 程序坏了 ✗）：%r" % txt)
        check("预览刷新已自动降到 12 fps" in p.lbl_stats.toolTip(),
              "降档明细没进 tooltip：%r" % p.lbl_stats.toolTip())
        p._on_stats(dict(base, src_lag=False, preview_fps=50.0, preview_capped=False))
        check("预览已自动降到" not in p.lbl_stats.text(),
              "够用（没被降档）了状态行还挂着「已自动降到」✗：%r" % p.lbl_stats.text())
    finally:
        lp._bgr_to_pixmap = orig
        p.close()


def t_pixmap_handles_padded_frame():
    """显示链路：**带行 padding 的帧**也要画对，且不再多拷一次（2026-09-26 性能）。

    两个坑一起钉：
      · padding：`to_ndarray(bgr24)` 的帧 stride 比 3*w 大，按 3*w 读会整幅错位；
      · 那次多余的 `qimg.copy()`：1920×1080 实测 9.43 → 7.02 ms/帧，而它跑在
        **GUI 主线程**上 —— 去掉它等于把 CPU 还给推理。
        （去掉 `ascontiguousarray` 反而更慢：14.77 ms/帧，跨步 tobytes 走通用路径。）
    """
    from PyQt5.QtWidgets import QApplication
    from gui.live_panel import _bgr_to_pixmap

    app = QApplication.instance() or QApplication([])      # noqa: F841
    pad = np.zeros((8, 34, 3), np.uint8)                   # 真画面只有前 32 列
    pad[:, :32] = (10, 20, 30)                             # BGR
    pix = _bgr_to_pixmap(pad[:, :32])
    check((pix.width(), pix.height()) == (32, 8),
          "带 padding 的帧画出来尺寸不对：%dx%d" % (pix.width(), pix.height()))
    c = pix.toImage().pixelColor(4, 4)
    check((c.red(), c.green(), c.blue()) == (30, 20, 10),
          "像素串了（BGR 30/20/10 该读成 RGB 30/20/10）：%r"
          % ((c.red(), c.green(), c.blue()),))
    # 源码约定：不许再出现那次「白拷一整幅」的调用。
    # ⚠ 判据写成**完整调用**（不是 `qimg.copy()`）：`_bgr_to_pixmap` 的 docstring 里
    # 正解释着"为什么不要它"，宽判据会把注释也算成违规（写这条时就是这么红的）。
    src = (ROOT / "gui" / "live_panel.py").read_text(encoding="utf-8")
    check("QPixmap.fromImage(qimg.copy())" not in src,
          "`_bgr_to_pixmap` 里又出现了 `QPixmap.fromImage(qimg.copy())`"
          "（白拷一整幅，1080p 约 2 ms/帧，还全在 GUI 主线程上）")


def t_mob_box_labels():
    """怪框上那行「地点」怎么画（用户 2026-09-27 两条要求 ✓）：

      ① **「查询到的怪框」的地点 ⇒ 写在框下方靠左**（原话："查询到的怪框地点标在框下方靠左显示"）✓；
      ② **锁定框不再写地点**（原话："**以前的锁定框表地点就不要了**" ✗）。

    做法：
      · ① 那行文字来自 `queried_mob_boxes()`（`mob_sets_of` 那个**唯一漏斗**记的账 ✓）——
        绘制层**只读** ✓；位置 = x 贴**框左边**、y 在**框下沿 + 14**，贴画面下沿放不下才翻上去 ✓；
      · ①' ⭐⭐ **框跟着「这一帧的检出框」走**（用户 2026-10-02 ✓ 原话："能不能让他只是作为
        『怪物框的标记』，跟着怪物的检出框走"）：账里**只存怪号**（不存框 ✓）⇒ 这里按怪号在
        `_mob_by_id`（= 这一帧的 `ws.mobs` ✓ 和绿框/锁定框同一份）里现查；**查不到就跳过**
        （画旧框 = 用户报的"红框在原地残留，可读性极差"✗）。
      · ② 锁定框那块**不许**再出现 `current_target_sets(...)` 的**调用** ✓（那份缓存本身也
        随之删掉了 ✗ —— "死数据不留"）。

    ⚠ 为什么钉这么细：绘制那一支要是自己调 `mob_sets_of` / `foothold_below`
      ⇒ **每帧扫几百条 foothold** ✗（纪律原文：只对当前那一只调用、每拍最多一次 ✓）。
      所以两处都按**源码**钉（这一段没有可驱动的最小夹具 ✓）。
    """
    src = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")

    # ① 查过的怪框：**框下方靠左**
    # ⚠ 定位锚用**绘制那处的入口**（`queried_mob_draw_list(ws.mobs)` ✓）—— **别**再用
    #   `queried_mob_boxes()`：它现在第一个出现的地方是 `queried_mob_draw_list` 的 docstring
    #   （账→框的换算那一段 ✓）⇒ 拿它当锚会切到**别的方法**里去 ⇒ 这几条断言全跑偏 ✗。
    i = src.index("queried_mob_draw_list(ws.mobs)")
    # ⚠ 窗口**别卡太死**（2026-09-28 踩过 ✗）：原来 1800，而那段里后来陆续加了注释
    #   （"判不出只写失败"那段 ✓）⇒ 把 `_qy2 + 14` **挤出窗口** ⇒ 用例**假红** ✗。
    #   这条钉的是"那几样在不在"，不是"隔多远" ⇒ 给宽一点 ✓。
    seg = src[i: i + 3000]
    check("_qlbl" in seg and "putText" in seg, "「查过的怪框」那行地点没了 ✗")
    check("_qy2 + 14" in seg,
          "那行没画在**框下方**（用户 2026-09-27 要求「框下方靠左」✗）")
    check("putText(vis,_qlbl,(_qx1" in seg.replace(" ", ""),
          "那行没贴**框左边**（「靠左」✗）")
    check("_qy1 - 6" in seg or "_qy1" in seg,
          "贴画面下沿时的兜底（翻回框上方）没了 ✗（那样字会掉出画面外）")
    # ⚠ 只看**真调用**（带括号 ✓）：注释里本来就会提到这两个名字，别把注释算成调用 ✗
    check("mob_sets_of(" not in seg and "foothold_below(" not in seg,
          "绘制那一支里自己重算了集合（每帧扫 foothold ✗ 纪律不允许）")

    # ①' ⭐⭐ **框是"这只怪身上的标记"、跟着这一帧的检出框走**（用户 2026-10-02 ✓ 原话：
    #     "能不能让他只是作为『怪物框的标记』，跟着怪物的检出框走"）
    #     算法在 `queried_mob_draw_list()` 里（**纯函数** ⇒ `selftest_minimap.
    #     t_queried_mob_boxes_follow_detection` 真跑它 ✓）；绘制这段**只许照它给的坐标画** ✓。
    check("queried_mob_draw_list(" in seg,
          "绘制那段没走「按怪号在**这一帧的检出框**里现查」那个 helper（用账里的旧框 ⇒ 怪一走开"
          "红框就留在原地 ✗ 用户 2026-10-02 报的那件事）")
    check("_qbox" not in seg and "queried_mob_boxes()" not in seg,
          "绘制那段又直接从账里取框算了（绕过 helper ⇒ 「原地残留」会回来 ✗）：账里已经没有框了 ✗")

    # ② 锁定框**不再写地点**（钉**真调用** ✓：先把 `#` 注释行丢掉再找 ——
    #    那一段的注释里本来就会提到 `agent.current_target_sets()`（"以前在这里读过 ✗" ✓），
    #    不丢注释就会被自己那句话绊倒 ✗）
    j = src.index("锁定目标怪")
    lock_code = "\n".join(ln for ln in src[j: j + 1500].splitlines()
                          if not ln.lstrip().startswith("#"))
    check("current_target_sets(" not in lock_code,
          "锁定框那里还在调 `current_target_sets()` 写地点 —— 用户 2026-09-27 明确不要了 ✗")


def t_auto_precheck():
    """⭐⭐ **开自动前的体检**（用户 2026-09-29 ✓ 原话："开启自动前，检查一下条件吧，
    然后出弹窗提示"）。

    背景（这次踩的坑）：`森林迷宫III` 的小地图标定**只做了一半**
    （`datasets/map/105040303.mapcalib.json` 里只有 `world_offset`/`alpha`/`mode`，
    **缺 `scale`/`offset`** = 没有「面板 → 底图」换算 ✗）⇒ 玩家世界坐标恒为 `None`
    ⇒ 决策层判"没有玩家位置"（`decision/agent.py:4789`）⇒ 3 分钟后**自己把自动停了**
    （`player_lost_stop`）⇒ 表现成"开自动却只站着不打"✗ —— 查了半天 ✓。
    ⇒ 这几件事**开之前就查得出来** ⇒ 所以做在 `_toggle_auto` 里 ✓。

    钉四件：
      ① 没选地图 ⇒ 只报「没选地图」并**立刻返回**（不再往下判 ✓）；
      ② 地图不存在 ⇒ 报「没标定」+「没地形」，**两条都要带"→ 去哪修"** ✓；
      ③ ⭐ **源码级**：`_toggle_auto` 里真的在"开"之前调了体检、且不通过会**把按钮拨回去** ✓
         —— ⚠ 这条最要紧：**光有体检函数、没接上就等于没有** ✗；
      ④ ⛔⭐ **「禁用杀怪寻路」开着 ⇒ 一条都不报**（用户 2026-10-04 ✓），且 `_precheck_auto`
         **真的把那个开关传进 `need_mmap`** ✓（光有形参不传 = 没接上 ✗）。
    """
    from gui.player_panel import PlayerPanel

    _none = PlayerPanel._precheck_problems("")
    check(len(_none) == 1 and "没选地图" in _none[0],
          "没选地图时该只报「没选地图」并立刻返回 ✗：%r" % (_none,))

    # 一个**根本不存在**的地图 id ⇒ 标定 + 地形都该报，且都要写清去哪修 ✓
    _bad = PlayerPanel._precheck_problems("999999999")
    check(len(_bad) == 2,
          "不存在的地图该报「没标定」+「没地形」两条 ✗：%r" % (_bad,))
    check(any("标定" in x for x in _bad) and any("地形" in x for x in _bad),
          "少报了标定 / 地形其中一项 ✗：%r" % (_bad,))
    check(all("→" in x for x in _bad),
          "报问题却没写「→ 去哪修」⇒ 用户只知道不行、不知道怎么办 ✗：%r" % (_bad,))

    # ⚠ **不写"某个具体 map_id 必须通过"** —— 那是**本机数据**（换台机器 / 换项目就红 ✗）。
    #   改成"对任何输入都不许崩、结论必须是「非空字符串列表」" ✓
    #   （体检自己炸掉会把"开自动"也一起带崩 ✗，那比不体检更糟）。
    for _mid in ("", "999999999", "105040303", None, 12345):
        _r = PlayerPanel._precheck_problems(_mid)
        check(isinstance(_r, list) and all(isinstance(x, str) and x for x in _r),
              "体检对 %r 的结论不是「非空字符串列表」⇒ 会崩或弹出空窗 ✗：%r" % (_mid, _r))

    # ⛔⭐ 「**禁用杀怪寻路**」开着 ⇒ **一条都不报**（用户 2026-10-04 ✓ 原话："在设置里禁用
    #   杀怪寻路时，让开启自动也能正常运转"）：这一页查的三件（地图 id / 标定几何 / 地形图）
    #   全都只服务"世界坐标那一套"，而那个模式**根本不读世界坐标**（锁定/追击/走位全用画面
    #   坐标 ✓，见 `decision/agent.py::_player_located` ✓）⇒ 它们不再是"开了自动会不动 /
    #   自己停掉"的原因 ⇒ 再报就是**假警报**：弹窗默认按钮是「否」⇒ 点下去**自动压根没开
    #   起来** ⇒ 现场看着就是"这个开关一开就不干活" ✗。
    for _mid in ("", "999999999", "105040303", None):
        check(PlayerPanel._precheck_problems(_mid, need_mmap=False) == [],
              "「禁用杀怪寻路」开着还报体检问题（%r）⇒ 弹窗默认「否」会把自动挡在门外 ✗"
              % (_mid,))

    _src = (Path(__file__).resolve().parent.parent
            / "gui" / "player_panel.py").read_text(encoding="utf-8")
    check("if on and not self._precheck_auto():" in _src,
          "`_precheck_auto` 没接进 `_toggle_auto` ⇒ 体检等于没有（用户点了开、条件不齐、"
          "照样开 ⇒ 过几分钟自己停）✗")
    check("def _precheck_problems(" in _src and "@staticmethod" in _src,
          "判据没抽成 `@staticmethod` ⇒ 没法像这条用例这样单测 ✗")
    check("QMessageBox.warning(" in _src and "开自动前的检查没通过" in _src,
          "没弹窗 / 弹窗标题丢了 ⇒ 用户不知道发生了什么 ✗")
    # ⚠ 光有 `need_mmap` 形参、`_precheck_auto` 不传 ⇒ 等于没接上（那条开关还是会被弹窗挡 ✗）
    check("need_mmap=not bool(getattr(settings, \"disable_chase_pathfinding\", False))" in _src,
          "`_precheck_auto` 没把「禁用杀怪寻路」传进 `need_mmap` ⇒ 那个开关开着照样弹"
          "「条件没凑齐」（默认「否」⇒ 自动开不起来）✗")


def t_reconnect_resume_gate():
    """⭐⭐⭐ **断线重连回到游戏 ⇒ 体检通过才恢复自动**（用户 2026-10-07 ✓ 三选一里选了「体检」）。

    背景：`reconnect.py::_finish` 原来直接 `settings.enabled = True` ✗ ⇒ **绕过**了
    「开自动前体检」（2026-09-29 ✓ `_precheck_problems`）⇒ 标定只做了一半 / 没地形图时，
    重连回来照样把自动打开 ⇒ "开了却只站着不打、过几分钟自己停" ✗（正是那天的病根 ✓）。

    钉六件：
      ① **纯函数** `_resume_auto_verdict`：有问题 ⇒ `(False, 原因非空)`；没问题 ⇒ `(True, "")`；
      ② **行为级**（替身面板 ✓ 不吃 Qt）：旗子 + 体检不过 ⇒ `enabled` 仍是 False ✓ +
         状态栏写明「没恢复自动」✓ + **旗子被清**（不然每 500ms 重来一遍 ✗）+ 日志 ok=False ✓；
      ③ **行为级**：旗子 + 体检通过 ⇒ `enabled = True` ✓ + 界面跟着刷 ✓ + 日志 ok=True ✓；
      ④ **行为级**：**人已经自己开着** ⇒ 一条都不动（保持 True ✓、**不白跑体检** ✓、不刷界面 ✓）；
      ⑤ **源码钉**：`_poll_auto_state` 里**真的调了**那个消费点 ✓（光有函数没接上 = 没有 ✗
         —— 本仓库栽过好几次）+ 前台（弹窗）与后台（重连）走**同一份** `_auto_precheck_problems` ✓；
      ⑥ **源码钉**：`reconnect.py` 里**不许**再有 `enabled = True` ✗（那句就是绕过体检的元凶 ✓）
         + 置旗子那句在 ✓。
    """
    import types
    from gui.player_panel import PlayerPanel
    from decision import agent as _ag
    from core import behavior as _bh

    _s = _ag.settings
    _saved = (_s.enabled, getattr(_s, "reconnect_resume_pending", None),
              _s.reconnect_note)
    evs = []
    _orig_ev = _bh.event
    _bh.event = lambda name, **kw: evs.append((name, kw))
    try:
        # ---- ① 纯函数 ----
        ok, why = PlayerPanel._resume_auto_verdict([])
        check(ok is True and why == "", "① 没问题该放行（回 `(True, '')` ✗）：%r" % ((ok, why),))
        ok, why = PlayerPanel._resume_auto_verdict(["· **没标定**\n  → 去标定"])
        check(ok is False and "没标定" in why,
              "① 有问题该拦住并把原因带出来（弹窗那条用的是同一份判据 ✓）：%r" % ((ok, why),))

        # ---- ②③④ 行为级：替身面板（那个方法只认 self._auto_precheck_problems /
        #      self._refresh_auto_ui 两个口子 ✓ ⇒ 不用真建 Qt 控件 ✓）----
        def _panel(probs):
            calls = []
            return types.SimpleNamespace(
                _auto_precheck_problems=lambda: list(probs),
                # 判据用**真**那个（纯函数 ✓ 不碰 Qt ✓）⇒ 顺带把"原因怎么拼"也测了 ✓
                _resume_auto_verdict=PlayerPanel._resume_auto_verdict,
                _refresh_auto_ui=lambda: calls.append(1)), calls

        # ② 体检不过 ⇒ 不恢复
        _s.enabled = False
        _s.reconnect_resume_pending = True
        _s.reconnect_note = ""
        p, calls = _panel(["· **地图「X」还没标定**\n  → 去「路线识别」页标定"])
        evs.clear()
        PlayerPanel._poll_reconnect_resume(p)
        check(_s.enabled is False,
              "② 体检不过却把自动开了（用户要的是「不过就不恢复」✗）")
        check(_s.reconnect_resume_pending is False,
              "② 旗子没清 ⇒ 每 500ms 重来一遍（体检白跑 + 状态栏一直刷 ✗）")
        check("没恢复自动" in _s.reconnect_note and "标定" in _s.reconnect_note,
              "② 状态栏没写清「为什么不恢复」⇒ 人只知道自动没开、不知道去修什么 ✗：%r"
              % (_s.reconnect_note,))
        check([e for e in evs if e[0] == "reconnect_resume" and e[1].get("ok") is False],
              "② 没落日志 ⇒ 复盘时看不到「重连回来了但没恢复自动」✗：%r" % (evs,))
        check(not calls, "② 没恢复却把界面刷了一遍 ✗")

        # ③ 体检通过 ⇒ 恢复
        _s.enabled = False
        _s.reconnect_resume_pending = True
        _s.reconnect_note = ""
        p, calls = _panel([])
        evs.clear()
        PlayerPanel._poll_reconnect_resume(p)
        check(_s.enabled is True, "③ 体检通过该恢复自动（置旗子的意义就在这儿 ✗）")
        check(calls, "③ 恢复了却没刷界面（开关按钮要点亮 ✓）")
        check([e for e in evs if e[0] == "reconnect_resume" and e[1].get("ok") is True],
              "③ 恢复成功没留痕 ✗：%r" % (evs,))

        # ④ 人自己已经开着 ⇒ 不动他、也不白跑体检
        _s.enabled = True
        _s.reconnect_resume_pending = True
        _s.reconnect_note = "（人自己开的）"
        checked = []
        p, calls = _panel([])
        p._auto_precheck_problems = lambda: (checked.append(1), ["· 有问题"])[1]
        evs.clear()
        PlayerPanel._poll_reconnect_resume(p)
        check(_s.enabled is True and not checked and not calls and not evs,
              "④ 人已经自己开着时动了手（体检不过会**又给他关掉** ✗ 那是越权）："
              "enabled=%r 体检跑了=%r" % (_s.enabled, bool(checked)))
        check(_s.reconnect_resume_pending is False, "④ 旗子该照样清掉（别留着 ✗）")

        # ---- ⑤⑥ 源码钉 ----
        _root = Path(__file__).resolve().parent.parent
        _src = (_root / "gui" / "player_panel.py").read_text(encoding="utf-8")
        check("self._poll_reconnect_resume()" in _src,
              "消费点没接进 `_poll_auto_state`（500ms 那趟车 ✓）⇒ 光有函数没接上 = 没有 ✗")
        check(_src.count("self._auto_precheck_problems()") == 2,
              "前台（弹窗）与后台（重连恢复）**没有共用同一份体检** ✗：%d 处"
              % _src.count("self._auto_precheck_problems()"))
        _rc = (_root / "decision" / "reconnect.py").read_text(encoding="utf-8")
        check("enabled = True" not in _rc,
              "`reconnect.py` 还在自己开自动 ✗ —— 那就是**绕过体检**的那一句"
              "（2026-10-07 用户选了「体检」✓）")
        check("reconnect_resume_pending = True" in _rc,
              "`reconnect.py` 没置那个请求旗子 ⇒ 回到游戏再也不会恢复自动 ✗")
    finally:
        _bh.event = _orig_ev
        _s.enabled, _s.reconnect_resume_pending, _s.reconnect_note = _saved


def t_capture_est_frames():
    """⭐⭐ 采集卡片「**选中文件后显示预估多少帧**」（用户 2026-09-29 ✓ 原话："采集、选中
    文件后，能否在下面显示预估多少帧？"）。

    钉四件：
      ① `path` 控件要支持 **`on_pick` 回调**（`gui/steps/base.py`，**选完文件**才触发 ✓
         —— 取消选择不许把上一次的预估刷掉 ✗）；
      ② ⭐ **预估公式必须和真正抽帧的那份一模一样**（`tools/extract_frames` 的
         `总帧数 // stride + 1` ✓）—— 两处不一致就会出现"预估 100、实际 97"，
         人的第一反应是"是不是抽漏了"✗；
      ③ **拿不到总帧数时要兜底**：MKV 录屏（OBS 之类）`stream.frames` 和 `stream.duration`
        常常**都是 0** ✗ ⇒ 靠 `container.duration ÷ av.time_base × fps` 兜 ✓
         （实测 `plain02.mkv`：3579 vs 真值 3580 ✓）；
      ④ 估算要有**说明**（"是估的" / "被最多张数封顶"）⇒ 别让人当成精确值 ✗。

    ⚠ 用**源码级**钉（真拿视频文件来测 ⇒ 换台机器 / 没录屏就红 ✗）；行为侧已实测
    （mp4 8422 帧、mkv 3579 帧、改 stride/limit 会重算、坏路径不崩、耗时 66~73ms ✓）。
    """
    _root = Path(__file__).resolve().parent.parent
    _b = (_root / "gui" / "steps" / "base.py").read_text(encoding="utf-8")
    _c = (_root / "gui" / "steps" / "cards.py").read_text(encoding="utf-8")
    check('on_pick=kw.get("on_pick")' in _b and "if callable(on_pick):" in _b,
          "`path` 控件没有 `on_pick` 钩子 ⇒ 选完文件不会刷新预估 ✗")
    check("total // stride + 1" in _c,
          "采集预估的公式和 `tools/extract_frames` 那份不一致（那边是 "
          "`total_frames // stride + 1`）⇒ 「预估/实际」会对不上 ✗")
    check("_cd / float(av.time_base)" in _c,
          "没做**容器级时长**的兜底 ⇒ MKV 录屏（`frames`/`duration` 都是 0）会永远显示"
          "「这个文件没报总帧数」✗")
    check('已被「最多张数」封顶' in _c and "按容器时长×帧率估的" in _c,
          "「是估的」/「被封顶」没标出来 ⇒ 用户会把估算当精确值 ✗")
    check("for _k in (\"stride\", \"limit\")" in _c,
          "`stride`/`limit` 改了没重算 ⇒ 标签会停在旧数字上 ✗")


def t_key_caps_overlay():
    """左下「按键帽」：按住的**半透明绿填充**、没按的极淡灰；文案随键位映射自适应。

    2026-09-30 用户要求 ✓（"以'半透明背景色填充'的形式在画面左下蓝框位置显示
    Agent 当前正在按住的键，从左到右分别是：4个方向键、跳、输出"）。
    """
    from gui.live_thread import (_KEYCAP_SLOTS, draw_key_caps,
                                 keycap_positions)

    class _KS:
        def __init__(self, keys):
            self._k = set(keys)

        def pressed(self):
            return set(self._k)

    class _S:
        keymap = {"left": "left", "up": "up", "down": "down", "right": "right",
                  "jump": "alt", "attack": "ctrl"}

    class _A:
        settings = _S()
        keys = _KS({"left", "alt"})           # ← 按住 + 跳(Alt) 按住

    vis = np.zeros((240, 320, 3), np.uint8)
    draw_key_caps(vis, _A())                  # 不炸 ✓
    # 布局用**实现同款算式**（`keycap_positions` ✓ 别各算各的 ✗ —— 倒 T 布局 ✓）
    pos, side = keycap_positions(240, 320)
    for slot in _KEYCAP_SLOTS:
        px, py = pos[slot]
        # 采样**帽内左上角**：居中那行是文字笔画（LINE_AA 的灰/绿都占 ✗ 采样会被骗）
        b, g, r = vis[py + 4, px + 4]
        on = slot in ("left", "jump")
        if on:
            check(g > b and g > r and g > 60,
                  "按住的键帽（%s）该是半透明绿填充 ✗：BGR=(%d,%d,%d)"
                  % (slot, b, g, r))
        else:
            check(abs(g - b) < 30 and abs(g - r) < 30 and g < 40,
                  "没按的键帽（%s）该是极淡灰、不该带绿色 ✗：BGR=(%d,%d,%d)"
                  % (slot, b, g, r))
    # 替身/缺失 ⇒ 静默跳过（不炸、画面不动 ✓）
    vis2 = np.zeros((240, 320, 3), np.uint8)
    draw_key_caps(vis2, object())
    check(not vis2.any(), "拿不到 settings/keys 时该什么都不画 ✗")
    class _S2:
        keymap = {"left": "a", "up": "w", "down": "s", "right": "d",
                  "jump": "space", "attack": "j"}
    class _A2:
        settings = _S2()
        keys = _KS(set())
    vis3 = np.zeros((240, 320, 3), np.uint8)
    draw_key_caps(vis3, _A2())                # 改键位 ⇒ 不炸（文案自适应走 ASCII/原名 ✓）


def t_lie_recording():
    """⭐⭐ **测谎现场录屏**（用户 2026-10-02 ✓ 原话："实时触发测谎时录屏，最后确认弹窗关闭后
    结束录屏（录制带所有我们后期线条的）好保留现场" ✓）。

    口径（逐条钉）：
      · 起 = 状态进 `lie_*`、止 = 状态离开 `lie_*`（`lie_success` 关掉 / 回战斗 ✓）；
      · 录**原生帧**（不含战斗框线 ✓）⇒ **绝不能污染 `raw`** ✗（它还要走 `raw_frame_ready`
        的测量链路 ✓ 见 `current_frame()` 那段口径 ✓）；
      · 叠加 = **时间戳 + 帧号 + 屏幕状态**（用户选 ✓）；落盘到 `record_dir()` ✓、文件名
        `lie_<时间戳>.mp4` ✓。
    ⚠ 这里直接调 `LiveThread` 的那几个方法（`object.__new__` 绕过 `__init__` ⇒ 不起线程、
      不要流 ✓），并把 `record_dir` 指到**临时目录**（绝不碰用户数据 ✓）。
    """
    import tempfile
    import time as _time

    from core import config as cfg
    from gui.live_thread import LiveThread

    # ⚠ `LiveThread` 是 PyQt 的 `QThread` 子类 ⇒ 只能用 `LiveThread.__new__`（`object.__new__`
    #   会抛 "is not safe" ✗）；绕过 `__init__` ⇒ 不起线程、不要流、不建 UI ✓。
    t = LiveThread.__new__(LiveThread)
    t._screen_state = "combat"
    t._lie_rec = None
    t._lie_rec_pending = 0.0
    t._lie_rec_path = ""
    t._lie_rec_n = 0
    t._lie_rec_t0 = 0.0
    t._lie_rec_err = ""

    check(t._lie_rec_pending == 0.0, "初始状态不该在录")
    t._lie_rec_stop(why="空停")               # 没在录时停 ⇒ 必须无害 ✓
    check(t._lie_rec is None and t._lie_rec_pending == 0.0, "空停把状态搞脏了")

    # ⚠⚠ 用例**不许依赖用户配置**（2026-10-09 ✓ 逮到过一次 ✗）：起录前读的是 `_rec_allowed`
    #   （它看 `config/live.yaml` 的 `rec_lie` / `rec_disc` ✓）—— 而用户完全可能把
    #   「保留测谎录屏」关掉（他现在的配置就是 `rec_lie: false` ✓）⇒ 这条用例会红在
    #   "触发（进 lie_*）没起录"上，**看着像功能坏了** ✗，其实是被自己的配置挡住 ✓。
    #   ⇒ 按本文件已有的手法**注入配置**（下面那一段也这么干 ✓），测完立刻还原 ✓。
    #   ⚠ 别去罩 `_rec_allowed` 本身 ✗：这条用例后半段**就是要测它**（开关关掉要如实回 False ✓）。
    _old_live0 = cfg.load_live
    cfg.load_live = lambda: {"rec_lie": True, "rec_disc": True}
    try:
        t._lie_rec_start()
        check(t._lie_rec_pending > 0.0, "触发（进 lie_*）没起录")
        t._lie_rec_stop(why="单元")
        check(t._lie_rec_pending == 0.0 and t._lie_rec is None, "停录没清状态")
    finally:
        cfg.load_live = _old_live0

    _d = tempfile.TemporaryDirectory()
    _old = cfg.record_dir
    cfg.record_dir = lambda: Path(_d.name)
    try:
        # ⚠ 这一段（起录 + 写两帧）**同样要注入开关** ✗（同上面那一段的口径 ✓）：不注入的话，
        #   用户把「保留测谎录屏」关着（他现在的 `live.yaml` 就是 `rec_lie: false` ✓）⇒
        #   `_lie_rec_start()` 直接返回 False ⇒ `_lie_rec_write` 什么都不做 ⇒ 红在
        #   "没按帧写进容器（实际写 0 帧）"上 ✗ —— **看着像录制坏了**，其实是被自己的配置挡住 ✓。
        _old_live1 = cfg.load_live
        cfg.load_live = lambda: {"rec_lie": True, "rec_disc": True}
        try:
            t._lie_rec_start()
            _raw = _frame(7)
            t._lie_rec_write(_raw, _time.monotonic())
            t._lie_rec_write(_raw, _time.monotonic())
        finally:
            cfg.load_live = _old_live1
        check(t._lie_rec is not None and t._lie_rec_n == 2,
              "没按帧写进容器（实际写 %d 帧）" % t._lie_rec_n)
        check(int(_raw[0, 0, 0]) == 7,
              "**污染了原生帧**（raw 左上像素被改：%d ≠ 7）—— 测量链路还指望它干净 ✗"
              % int(_raw[0, 0, 0]))
        _p = Path(t._lie_rec_path)
        t._lie_rec_stop(why="单元")
        check(_p.exists() and _p.stat().st_size > 0,
              "停录后文件不存在或为空：%s" % _p)
        check(_p.name.startswith("lie_") and _p.suffix == ".mp4",
              "文件名不符合约定（要 lie_<时间戳>.mp4）：%s" % _p.name)

        # ⭐⭐ **两类现场 + 设置页那两个开关**（用户 2026-10-02 ✓ 原话："在设置（数据工作台
        #   上面的按钮弹窗）→ 保护与恢复页签 最顶部加开关『保留测谎录屏』『保留断线录屏』" ✓）：
        #   · 开关落 `config/live.yaml`（`rec_lie` / `rec_disc` ✓ 默认开 ✓）；
        #   · **关掉 ⇒ 该类不起录**（一个字节都不写 ✓）；开机时 `_rec_kind` 记种类（文件名前缀 ✓）。
        # ⚠⚠ **这一段必须注入配置再断言**（`load_live = {}` ⇒ 走默认 True ✓）：
        #   原来直接读**用户那份** `config/live.yaml` ✗ ⇒ 用户只要在设置里关掉
        #   「保留断线录屏」（他 2026-10-04 就关了 —— `rec_disc: false` ✓），
        #   ①「两个开关默认都开着」②「断线类起录 / 落盘」两处都会红 ✗
        #   —— 而且**看起来像代码坏了**（同 `selftest_machineload.t_degrades_without_psutil`
        #   那次教训：**别拿本机环境当输入** ✓）。
        t._rec_kind = ""
        _old_live = cfg.load_live
        try:
            cfg.load_live = lambda: {}              # 注入空配置 ⇒ 默认全开 ✓
            check(t._rec_allowed("lie") is True and t._rec_allowed("disc") is True,
                  "两个录屏开关**默认都开着**（`config/live.yaml` 没写这两个键 ⇒ 默认 True ✓ "
                  "宁可不小心留下现场，也别静默丢掉 ✓）")
            cfg.load_live = lambda: {"rec_lie": False, "rec_disc": False}
            check(t._rec_allowed("lie") is False and t._rec_allowed("disc") is False,
                  "开关关掉 ⇒ `_rec_allowed` 如实回 False ✓")
            check(t._lie_rec_start("lie") is False and t._lie_rec_pending == 0.0,
                  "**关掉的类别不起录**（返回 False、`pending` 不动 ✓ ⇒ 一个字节都不写 ✓）")
            cfg.load_live = lambda: {}              # 回到"默认开"再做断线那一段 ✓
            _rd = Path(_d.name)
            _before = set(_rd.glob("*.mp4"))
            t._lie_rec_start("disc")
            check(t._rec_kind == "disc" and t._lie_rec_pending > 0.0,
                  "断线类起录 ⇒ `_rec_kind='disc'`（文件名会带 `disc_` 前缀 ✓）")
            _raw2 = _frame(9)
            t._lie_rec_write(_raw2, _time.monotonic())
            t._lie_rec_stop(why="回到游戏")
            _new = sorted(p.name for p in (set(_rd.glob("*.mp4")) - _before))
            check(len(_new) == 1 and _new[0].startswith("disc_"),
                  "断线那段落盘的文件名要用 `disc_` 前缀（实际 %s ✓）" % (_new,))
        finally:
            cfg.load_live = _old_live
        check(t._rec_kind == "",
              "停录后 `_rec_kind` 必须清空 ✓（否则下一拍 `combat` 会拿它误触发一次停录 ✗）")
    finally:
        cfg.record_dir = _old
        _d.cleanup()

    # ⭐⭐ **设置页那两个开关**（用户 2026-10-02 ✓ 原话："在设置（数据工作台上面的按钮弹窗）→
    #   保护与恢复页签 **最顶部**加开关『保留测谎录屏』『保留断线录屏』" ✓）—— 源码级钉住
    #   三件事（`SettingsDialog` 太重，不适合在这套自检里真建一个来测 ✓）：
    #     · 两个 `QCheckBox` 存在 ✓；· 点确定后落 `config/live.yaml` 的 `rec_lie` / `rec_disc`
    #     （= 刚才那些 `_rec_allowed` 读的同一对键 ✓ 两边不许分家 ✗）；· 位置在**保护与恢复
    #     页签的最顶部**（该页签第一个 `_head`「朝向无变化」之前 ✓）。
    import pathlib as _p10

    _sd = (_p10.Path(__file__).resolve().parent.parent / "gui"
           / "settings_dialog.py").read_text(encoding="utf-8")
    check('self.ck_rec_lie = QCheckBox("保留测谎录屏")' in _sd
          and 'self.ck_rec_disc = QCheckBox("保留断线录屏")' in _sd,
          "设置页有「保留测谎录屏 / 保留断线录屏」两个开关 ✓")
    check("update_live(rec_lie=rl)" in _sd and "update_live(rec_disc=rd)" in _sd,
          "两个开关点确定后落 `config/live.yaml`（`rec_lie` / `rec_disc` ✓ —— 与实时线程 "
          "`_rec_allowed` 读的必须是同一对键 ✓ 不许两边各写一套 ✗）")
    _ip = _sd.index("def _page_protect")
    _ij = _sd.index("def _page_judge")
    _ic = _sd.index("ck_rec_lie")
    check(_ip < _ic < _ij and _ic < _sd.index("朝向无变化停止自动"),
          "两个开关落在**保护与恢复**页签内、且在**最顶部**（该页签第一个控件之前 ✓ "
          "用户点名的位置 ✓）")


def t_engine_imgsz_alignment():
    """⭐⭐ TensorRT 引擎下「imgsz 随便填」不许把实时开不起来（用户 2026-10-03 ✓ 现场：
    「现在开始实时，如果imgsz不是640会报错，能不能让其兼容可以随便填写？」）。

    本机实测（`寺院通道2/models/detect_v5.engine` ✓）：
      · **`.engine`**：填 960 / 512 直接
        `AssertionError: input size … not equal to max model size (1, 3, 640, 640)`
        ⇒ 从 `_run` 抛出去被 `failed` 收掉 ⇒ **实时根本起不来** ✗（用户踩的就是它 ✓）；
      · **`.pt`**：640 / 960 / 512 / 641 **都能跑** ✓（641 还被 ultralytics 对齐到 672 ✓）
        ⇒ 那条路**一个字都不许改** ✗。
    引擎的输入尺寸**导出时焊死** ⇒ 想让 960 "真生效"是不可能的 ✗；能做的只有
    **按引擎尺寸跑 + 如实说清** ✓（假装 960 生效 ⇒ 画面尺度与框尺度悄悄不一致 ✗）。

    钉五件：
      ① 从报错里抠得出尺寸（`(1, 3, 640, 640)` ⇒ 640 ✓）；
      ② **认不出来返回 None**（无关报错 / 非方形 profile ✓）⇒ 调用方 `raise` 照旧报上去 ✗
         （**不许假装兼容** ✓ —— 那会把别的问题吞掉 ✓）；
      ③ 预演**只对 `.engine`** 做（`.pt` 那条路不许被碰 ✗）；
      ④ 预演 + 对齐必须在**主循环那次 `predict` 之前**（否则报错发生在循环里 ⇒ 线程被收掉 ✓）；
      ⑤ 失败上报要**把栈打全**（不然只有一句「类型: 消息」，定不了案 ✓）。
    """
    from gui.live_thread import LiveThread

    _err = AssertionError("input size torch.Size([1, 3, 960, 960]) not equal to "
                          "max model size (1, 3, 640, 640)")
    check(LiveThread._engine_imgsz_from_error(_err) == 640,
          "抠不出引擎尺寸（用户报的就是这条 ✗）：%r"
          % (LiveThread._engine_imgsz_from_error(_err),))
    check(LiveThread._engine_imgsz_from_error(RuntimeError("别的问题")) is None,
          "无关报错也该返回 None（否则会把别的问题当尺寸问题吃掉 ✗）")
    check(LiveThread._engine_imgsz_from_error(
        AssertionError("input size torch.Size([1, 3, 960, 640]) not equal to "
                       "max model size (1, 3, 960, 640)")) is None,
          "非方形 profile 不许瞎猜 ✗")

    src = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    check("_engine_imgsz_from_error(_e)" in src,
          "找不到「引擎预演 + 抠尺寸」那段（被删了？）⇒ 实时又要起不来 ✗")
    # ⚠ **要看"这段被 `.engine` 守卫着"** ✗ —— 光断言 `endswith(".engine")` 在**文件里**
    #   出现抓不住"守卫被删"：**别处**（FP16 那段 ✓）本来就有同一个写法 ⇒ 反向验证实测
    #   会假绿 ✓；⇒ 改成"取预演那段**前面**的窗口，里面必须有那个条件" ✓。
    _win = src[max(0, src.index("_probe_img") - 500):src.index("_probe_img")]
    check('endswith(".engine")' in _win,
          "预演没被 `endswith(\".engine\")` 守卫住 ⇒ `.pt` 那条路也白多跑一次推理 ✗")
    # ⚠ **锚点 2026-10-15 改过**（原来钉的是字面串 `res = model.predict(vis` ✗）：
    #   「只检测视野框」那次改动把主循环喂的对象从 `vis` 换成了 `_det_src`
    #   （可能被剪裁 ✓ 见 `t_det_vision_only_crop` ✓）⇒ 老锚点当场 `substring not found` ✗。
    #   ⚠ 这条钉要的**是"预演/对齐发生在主循环那次推理之前"**（不是"喂的一定是 `vis`"✓）⇒
    #   锚点改成 `res = model.predict(` —— 它在这个文件里**仍然唯一** ✓（预演那次是
    #   `model.predict(_probe_img, …`、不赋给 `res` ✓）：
    # ⚠ 2026-10-20：**主循环那次推理搬进 `runtime/frame_infer.py`**（阶段 1 第二刀 ✓）
    #   ⇒ "对齐在前"不再是**同文件里的先后**，而是**结构保证**：对齐在模型加载块里、
    #   加载只在**进循环之前**跑一次 ✓ ⇒ 钉的改成"两半都还在 + 主循环只有一个口" ✓
    _inf_src0 = (ROOT / "runtime" / "frame_infer.py").read_text(encoding="utf-8")
    check("_engine_imgsz_from_error(_e)" in src
          and _inf_src0.count("res = model.predict(") == 1,
          "引擎尺寸对齐那段没了 / 主循环那次推理不止一处 ✗ ⇒ 报错照样把线程收掉 ✓")
    check("if _eng is None:" in src
          and "raise" in src.split("if _eng is None:")[1][:220],
          "认不出尺寸时没 `raise` ⇒ 别的问题会被悄悄吞掉 ✗")
    check("traceback.print_exc()" in src,
          "失败上报没打完整栈 ⇒ 下次还是只有一句「类型: 消息」，定不了案 ✗")


def t_mmap_fail_reason_is_logged():
    """⭐ 小地图定位**失败的原因**要进日志，不许只在界面上说（用户 2026-10-03 ✓ 现场：
    "寺院通道2 现在开启自动怎么没用了"）。

    现场是这么查出来的：`perf.log` 里 `mmap_miss` **恒为 1** ✓（定位一直失败 ✓），
    可"**为什么**失败"当时只在界面状态行 ✗ ⇒ 看日志只能猜 ✓；而 agent 正是靠它判
    「未定位玩家」⇒ 回 idle ⇒ 超时还**自己把自动关掉** ⇒ 现象就是"开启自动没用" ✓。

    钉两件（源码约定 ✓ —— 行为级要靠真定位器，成本不值 ✓）：
      ① `mmap_miss` 那条计数**旁边**就有 `perf.note("mmap_note", …)`（失败原因进段头 ✓）；
      ② 只在**原因变化**时才写（`_mmap_note_last` 去重 ✗ 否则每拍一行把段头刷爆 ✓）。
    """
    src = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    _i = src.index('perf.count("mmap_ok" if loc.get("ok") else "mmap_miss")')
    # ⚠ 这个窗口是"从计数那句往下这一整块"的**启发式** ⇒ 块里加注释/加分支时要跟着放宽 ✓
    #   （2026-10-04 加了"恢复写 ok"那条 + 它的长注释 ⇒ 900 装不下 `perf.note("mmap_note"`
    #    了 ⇒ 本用例当场红 ✓ 正好说明窗口别卡太紧 ✓）。
    # ⚠ 2026-10-06：`mmap_panel` 那条**只读诊断**（把喂给定位器的那块面板存一张下来 ✓）
    #   插在这一段里 ⇒ 2400 又装不下 `perf.note("mmap_note"` 了 ✗ ⇒ 照本用例那句
    #   "块里加注释/加分支时要跟着放宽 ✓" 放宽到 4000 ✓（判据本身一个字没动 ✓）。
    _seg = src[_i:_i + 6000]        # 实测相隔 4313 字（2026-10-06 ✓ 见上面那段 ✓）
    check('perf.note("mmap_note"' in _seg,
          "定位失败的原因没进日志 ⇒ 下次还只能看界面那行、日志里只有一个计数 ✗")
    check("_mmap_note_last" in _seg,
          "失败原因没去重 ⇒ 段头会被同一句刷爆 ✗")

    # ⭐⭐ **"现在是哪张图"这条线要真接上**（用户 2026-10-03 ✓ 现场："A 显示 当前地图还没有"
    #   + "没有收到小地图流"）—— 根因是 `MiniMapClient.switch_map` **写好了却一直没人调** ✗
    #   ⇒ A 永不收到 `MAP` ⇒ A 那行「还没有」✓、且 A **没区域就不抓屏** ✗ ⇒ B 收 0 帧 ✓。
    #   ⚠ 所以这里必须钉**调用点**（`self._push_map_to_a()` ✓），不是"函数存在" ✗
    #     —— 上一轮的盲区就是这个形状（断言"槽存在"抓不住 connect 被删 ✓）。
    check("self._push_map_to_a()" in src,
          "`_push_map_to_a` 没被调用 ⇒ A 还是收不到 `MAP` ⇒ 那行永远「还没有」+ 不推帧 ✗")
    check("def _push_map_to_a(self):" in src and "switch_map(" in src,
          "没走已有那条实现（`MiniMapClient.switch_map` ✓ 不许另写一套 ✗）")
    _h = src[src.index("def _push_map_to_a(self):"):]
    check('self._mmap_src == "live"' in _h[:1200],
          "没排除「来源＝从实时画面」⇒ 那条路不经过 A 的推流，白发一条控制连接 ✗")
    # ⚠⚠ 这两条**必须断言那一行本身**，不能只看名字 ✗（反向验证实测踩了两个盲区：
    #    ① 断言 `"_mmap_map_sent" in src` —— 去掉去重那句后**赋值那句**还在 ⇒ 假绿 ✓；
    #    ② 断言 `'perf.note("mmap_map"'` —— 我自己的**注释里**也写了这串 ⇒ 假绿 ✓✓）。
    # ⚠⚠ 判重必须**只认成功**（`_mmap_map_ok` ✓）—— 用户 2026-10-03 ✓ 现场铁证：
    #   原来记在**尝试之前** ✗ ⇒ 23:12 那次失败后**再也不发** ✓ ⇒ 用户把 A 换成新代码
    #   并重启后 B 一个字没发 ⇒ A 那行永远「还没有」+ 一直"等 MAP"✓。
    check('if mid == getattr(self, "_mmap_map_ok", None):' in src,
          "判重还是「发过了就算」✗ ⇒ 失败不会重试、A 修好了也不会自愈 ✓")
    check("self._mmap_map_ok = mid" in src and "_mmap_map_try" in src,
          "失败没有重试（或没有节流）⇒ 要么永不自愈、要么每拍开一条连接 ✗")
    check('_perf.note("mmap_map", ("ok " if ok else "失败 ")' in src,
          "A 的回答没进日志 ⇒ 失败时人不知道「A 还缺什么」✗")
    # ③ 决策给出的理由（`agent.tick` 的 `reason`）也必须进日志 —— 这次"开启自动没用"
    #    最直接的答案就是它那句「未定位玩家」✓（以前谁都不记 ✗）。
    check('_perf2.note("agent_why"' in src or 'note("agent_why"' in src,
          "决策理由没进日志 ⇒ 只能看到 st=idle / move=0，看不到「为什么」✗")
    check("_agent_why_last" in src,
          "决策理由没去重 ⇒ 每拍一样的那句会把段头刷爆 ✗")


def t_mmap_note_says_ok_on_recovery():
    """⭐⭐ `mmap_note` **恢复之后必须说一句 `ok`**（2026-10-04 ✓ 现场我差点看错 ✗）。

    背景：`perf.note` 的值**整个会话都不清**（`core/perf.py` 的 `_notes` 只在 `configure()`
    清一次 ✓）⇒ 失败那句话写一次，就会**一直挂在后面每一段的段头** ✗ —— 我看 01:5x 的
    log 时正是把它当成"现在还在失败"，而**同一段**里 `mmap_ok` 明明在涨 ✓
    （那种"注记和计数互相打脸"的日志，下次谁看都会踩 ✗）。
    ⇒ 现在"上一次是失败"这一刻**写一次 `ok`** ✓ ⇒ 注记的含义变成"**最后一次变化是什么**" ✓。

    钉四件（**真跑 `_locate_latest`** ✓ —— `LiveThread.__new__` 绕过 `__init__`，不起线程、
    不要流 ✓ 同 `t_lie_recording` 的写法 ✓；`_locate_mmap` 换成替身、`core.perf` 三个口
    记下来 ✓）：
      ① 失败 ⇒ 注记 = 原因（截断到 70 字 ✓）；
      ② **同一句再来** ⇒ **不写**（去重：段头不许被刷爆 ✗）；
      ③ 恢复 ⇒ 注记 = `ok`（**这一条就是本次要修的**）；
      ④ 一直 ok ⇒ **不重复写**（不刷屏 ✓）；之后**同一句失败再来** ⇒ 还要写 ✓（去重标记
         在恢复时清掉了 ✓，别把"恢复过"记成"这句说过了" ✗）。
    """
    from unittest.mock import patch

    from core import perf as _perf
    from gui.live_thread import LiveThread

    t = LiveThread.__new__(LiveThread)          # 绕过 __init__ ⇒ 不起线程/不要流 ✓
    notes = []

    def _fail(note="认不出玩家标记（黄族/color层：像点的块 0 个）"):
        return {"ok": False, "note": note}

    _ok = {"ok": True, "note": ""}
    _cur = {"loc": _fail()}
    t._locate_mmap = lambda _panel: _cur["loc"]           # 定位结果由用例摆 ✓

    with patch.object(_perf, "note", lambda k, v: notes.append((k, v))), \
            patch.object(_perf, "count", lambda *a, **k: None), \
            patch.object(_perf, "ms", lambda *a, **k: None):
        # ① 失败 ⇒ 原因进注记
        t._locate_latest()
        check(notes == [("mmap_note", "认不出玩家标记（黄族/color层：像点的块 0 个）")],
              "定位失败的原因没进注记：%r" % (notes,))

        # ② 同一句再来 ⇒ 不写（去重）
        t._locate_latest()
        check(len(notes) == 1, "同一句失败原因写了两遍（段头会被刷爆 ✗）：%r" % (notes,))

        # ③ 恢复 ⇒ **必须说一句 `ok`**
        _cur["loc"] = _ok
        t._locate_latest()
        check(notes[-1] == ("mmap_note", "ok"),
              "定位恢复了，注记却还停在失败那句上 ⇒ 看日志的人会以为**现在还在失败** ✗："
              "%r" % (notes,))

        # ④ 一直 ok ⇒ 不重复写；同一句失败再来 ⇒ 照旧要写 ✓
        t._locate_latest()
        check(len(notes) == 2, "一直 ok 却每拍写一次 ⇒ 段头刷屏 ✗：%r" % (notes,))
        _cur["loc"] = _fail()
        t._locate_latest()
        check(notes[-1][0] == "mmap_note" and notes[-1][1].startswith("认不出玩家标记"),
              "恢复之后**同一句失败**没再写（去重标记没在恢复时清掉 ✗）：%r" % (notes,))


def t_live_weights_selectable():
    """⭐⭐⭐ 「开始推理」下面那个**权重能自己选**（用户 2026-10-04 ✓ 原话："开始推理按钮下面
    的权重为什么不能自己选（就像在 训练模型页签→训练卡片→基础权重 一样）"）。

    原来那格是**只读** `QLineEdit`（`ed_weights` ✓）：内容只能由 `bind(project)` 自动挑
    （`.engine` 优先 ✓）⇒「拿别的项目训好的模型先试试」「手头这份实验权重跑一下」
    「刚导出的新引擎想立刻切过去」统统做不到 ✗（只能去动目录、或等 mtime 自己挑 ✓）。

    钉七件（②⑦ 是**老行为没变**的对照 ✓）：
      ① 控件是**能选的下拉**（不再是只读输入框 ✗），且没有 `ed_weights` 残留 ✗；
      ② 没选过 ⇒ 停在「（自动…）」且**真的等于老口径**（`.engine` 最新 > `.pt` 最新 >
         `runs/**/best.pt` ✓，全按 mtime ✓）；
      ③ 候选里**本项目的 `.engine` 排在 `.pt` 前面**（几个尺寸的引擎共存，都得列到 ✓）；
      ④ 候选里有**别的项目训好的**（同一份 `tools.yolo_augment.list_weights()` ✓ ⇒
         与训练卡片「基础权重」同口径，不另写扫描 ✗）；
      ⑤ 选了 ⇒ **按项目记住**（`project.yaml` 的 `live.weights` ✓，**真落盘** ✓），
         `bind` 回来还选中它 ✓；
      ⑥ 存的那份**被删了** ⇒ 候选里有兜底项、并标明「文件不在了」✗ 静默吞掉不许 ✓；
      ⑦ 选「（自动…）」⇒ 存储清掉、回到自动挑 ✓。
    """
    import os as _os
    import shutil
    import tempfile

    from PyQt5.QtWidgets import QApplication, QComboBox

    import gui.live_panel as lp
    from gui.project import Project

    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")

    tmp = Path(tempfile.mkdtemp(prefix="live_w_"))
    try:
        proj = Project.create(tmp, name="用例项目", map_id="101030404")
        # 造三档权重，**mtime 拉开**（自动挑是按 mtime，同秒建的文件会分不出先后 ✓）
        t0 = 1_700_000_000.0
        files = {}
        for i, rel in enumerate(("models/old.pt", "models/eng_640.engine",
                                 "models/eng_960.engine",
                                 "runs/detect_v1/weights/best.pt")):
            f = proj.root / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("x", encoding="utf-8")
            _os.utime(str(f), (t0 + i * 10, t0 + i * 10))
            files[rel] = f
        newest_engine = files["models/eng_960.engine"]
        other_pt = files["models/old.pt"]

        pnl = lp.LivePanel()
        try:
            # ① 是"能选的下拉"
            check(isinstance(pnl.cmb_weights, QComboBox),
                  "「权重」不是下拉控件（用户要的就是能自己选 ✗）：%r"
                  % type(getattr(pnl, "cmb_weights", None)).__name__)
            check(not hasattr(pnl, "ed_weights"),
                  "还留着那个**只读**输入框 `ed_weights` ⇒ 两个控件并存必然对不上 ✗")

            # ② 没选过 ⇒ 自动；口径 = .engine 最新
            pnl.bind(proj)
            check(pnl.cmb_weights.currentData() == lp.LivePanel.WEIGHTS_AUTO,
                  "没选过时该停在「（自动…）」上：%r" % (pnl.cmb_weights.currentText(),))
            check(pnl.cur_weights() == str(newest_engine),
                  "「自动」没用老口径（.engine 优先、按 mtime 最新 ✓）：%r"
                  % (pnl.cur_weights(),))

            # ③ engine 排在 .pt 前面
            check(pnl.cmb_weights.findData(str(newest_engine)) >= 0
                  and pnl.cmb_weights.findData(str(other_pt)) >= 0,
                  "本项目自己的权重没进候选")
            check(pnl.cmb_weights.findData(str(newest_engine))
                  < pnl.cmb_weights.findData(str(other_pt)),
                  "候选顺序不对：引擎该排在 .pt 前面（引擎才是实时首选 ✓）")

            # ④ 与训练卡片同口径：别的项目训好的也列出来
            from tools.yolo_augment import list_weights
            others = [q for _l, q, _p in list_weights()][:3]
            miss = [q for q in others if pnl.cmb_weights.findData(q) < 0]
            check(not miss,
                  "别的项目训好的权重没进候选（换图先拿旧模型试是主要用法之一 ✗）：%s"
                  % (miss[:2],))

            # ⑤ 选了 ⇒ 按项目记住（真落盘）
            i = pnl.cmb_weights.findData(str(other_pt))
            pnl.cmb_weights.setCurrentIndex(i)
            pnl._on_weight_pick()
            check(pnl.cur_weights() == str(other_pt),
                  "选完却没用它：%r" % (pnl.cur_weights(),))
            saved = Project.open(proj.root).sec("live").get("weights")
            check(str(saved) == str(other_pt),
                  "选择没**落盘**到项目的 `live.weights`（下次打开就丢了 ✗）：%r" % (saved,))
            pnl.bind(proj)
            check(pnl.cmb_weights.currentData() == str(other_pt),
                  "重新绑定项目后没选回存的那份（界面显示 A、实际用 B ✗）：%r"
                  % (pnl.cmb_weights.currentText(),))

            # ⑥ 存的那份被删了 ⇒ 兜底项 + 标明"文件不在了"（不许静默 ✗）
            other_pt.unlink()
            pnl.bind(proj)
            j = pnl.cmb_weights.findData(str(other_pt))
            check(j >= 0, "存的权重被删后**候选里没有它** ⇒ 界面会显示成别的 ✗")
            check("文件不在了" in pnl.cmb_weights.itemText(j),
                  "兜底项没标明文件不在了（人会以为还在用它 ✗）：%r"
                  % (pnl.cmb_weights.itemText(j),))
            check(pnl.cur_weights() == str(other_pt),
                  "界面显示 = 实际用的（这条没坏就行）：%r" % (pnl.cur_weights(),))

            # ⑦ 选回「自动」⇒ 存储清掉、回到自动挑
            pnl.cmb_weights.setCurrentIndex(
                pnl.cmb_weights.findData(lp.LivePanel.WEIGHTS_AUTO))
            pnl._on_weight_pick()
            check(Project.open(proj.root).sec("live").get("weights") in ("", None),
                  "选「自动」没把项目里手选的那份清掉 ⇒ 永远回不到自动 ✗")
            check(pnl.cur_weights() == str(newest_engine),
                  "切回自动后没按老口径挑：%r" % (pnl.cur_weights(),))
        finally:
            pnl.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # 源码级：**唯一出口**（`start()` 与导出引擎都只许读 `cur_weights()` ✓ ——
    # 两处各读一个控件 / 各写一遍自动挑的逻辑，迟早"界面显示 A、实际用 B" ✗）
    src = (ROOT / "gui" / "live_panel.py").read_text(encoding="utf-8")
    check("ed_weights" not in src, "`live_panel.py` 里还有 `ed_weights` 残留 ✗")
    check(src.count("self.cur_weights()") == 2,
          "读权重的地方不是 2 处（`start()` + 导出引擎 ✓）：%d"
          % src.count("self.cur_weights()"))
    check('p.sec("live")["weights"]' in src,
          "手选结果没写进项目的 `live.weights`（那就等于没记住 ✗）")


def t_live_params_follow_project():
    """⭐⭐⭐ 「实时」页那一排参数**跟着项目走**（用户 2026-10-09 ✓ 原话："把主视区所有的配置
    根据项目存一下 —— 现在本地窗口来源、框选区域、怪 conf 等等参数每次都要重新填"）。

    钉五件：
      ① 改一格 ⇒ **真落盘**进项目（`project.yaml` 的 `live` 段 ✓）；**同时也写一份全局
         `config/live.yaml`** ✓（那是"新项目的播种值"、也是命令行工具读的那份 ✓）；
      ② **来源 ＋ 框选区域**也在清单里 ✓（用户点名的就是这两项 ✓）；
      ③ ⭐ **回填是只读动作** ✗：`bind()` 全程不许写盘（程序性 `setValue/setChecked` 不许
         伪装成"人改了参数"✗ —— 同 `route_panel._apply_route_cfg` 那条教训 ✓）；
      ④ 项目里**没存过的键** ⇒ 保持现在的值（= 全局那份 ✓ 老项目行为一字不变 ✓）；
      ⑤ 存档口**只有一处**（`_save_project_params` ✓）：控件 / 来源 / 框选 / `start()` 全汇到它 ✓。
    """
    import shutil
    import tempfile
    import unittest.mock as mock

    from PyQt5.QtWidgets import QApplication

    import gui.live_panel as lp
    from gui.project import Project

    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")

    tmp = Path(tempfile.mkdtemp(prefix="live_p_"))
    try:
        pa = Project.create(tmp / "a", name="项目A", map_id="107000100")
        pb = Project.create(tmp / "b", name="项目B", map_id="105090600")

        # 全局那份**换成假的**：真 `config/live.yaml` 一个字节都不许动 ✗（自检纪律 ✓）
        written = []
        with mock.patch.object(lp, "update_live",
                               lambda **kw: written.append(dict(kw))):
            pnl = lp.LivePanel()
            try:
                # ① 改一格 ⇒ 按项目落盘 ＋ 同步写全局
                pnl.bind(pa)
                n0 = len(written)
                pnl.sp_conf_mob.setValue(0.55)      # 真信号 ⇒ 走 `_save_live_params` ✓
                v = Project.open(pa.root).sec("live")
                check(abs(float(v.get("conf_mob") or 0) - 0.55) < 1e-6,
                      "改了「怪conf」却**没按项目存**（下次打开又得重填 ✗）：%r"
                      % (v.get("conf_mob"),))
                check(len(written) > n0
                      and abs(float(written[-1].get("conf_mob") or 0) - 0.55) < 1e-6,
                      "没同步写全局 live.yaml（命令行工具读的是它 ✗）：%r"
                      % (written[-1:] or None,))

                # ② 来源 ＋ 框选区域也进项目
                pnl.cmb_source.setCurrentIndex(pnl.cmb_source.findData("window"))
                pnl._rect = [10, 20, 300, 400]
                pnl._save_project_params()
                v = Project.open(pa.root).sec("live")
                check(str(v.get("capture_source")) == "window",
                      "「本地窗口来源」没按项目存：%r" % (v.get("capture_source"),))
                check([int(x) for x in (v.get("rect") or [])] == [10, 20, 300, 400],
                      "「框选区域」没按项目存（这是最烦的重填项 ✗）：%r" % (v.get("rect"),))

                # ②b 「抓屏方式」（`wgc` / `bitblt` ✓ 用户 2026-10-09 ✓ 那圈"莫名其妙的描边"
                #     就是它：Win11 会给被 WGC 采集的窗口画系统边框 ✗）也要按项目存 ✓
                pnl.cmb_wcap.setCurrentIndex(pnl.cmb_wcap.findData("bitblt"))
                check("window_capture" in lp.PROJ_LIVE_KEYS,
                      "`window_capture` 不在「按项目存」的清单里（那就不会回填 ✗）")
                v = Project.open(pa.root).sec("live")
                check(str(v.get("window_capture")) == "bitblt",
                      "「抓屏方式」没按项目存：%r" % (v.get("window_capture"),))

                # ③ 回填是只读：切到"啥都没存过"的项目 B ⇒ 一个字节都不许写 ✗
                n1 = len(written)
                pnl.bind(pb)
                check(len(written) == n1,
                      "`bind()` 回填时写盘了（程序性 setValue 被当成「人改了参数」✗）：%r"
                      % (written[n1:],))
                # ⚠ 项目**建出来时**就带一条 `live.weights: ''`（`Project.create` 的默认段 ✓）
                #   ⇒ 判据是"**没多出别的键**"（回填不许把这一排参数写进去 ✗）。
                check(set(Project.open(pb.root).sec("live") or {}) <= {"weights"},
                      "回填把项目 B 的 live 段写脏了（多出：%r）"
                      % (sorted(set(Project.open(pb.root).sec("live") or {}) - {"weights"}),))

                # ④ 项目里没存过的键 ⇒ 保持现在的值（老项目一字不变 ✓）
                check(abs(pnl.sp_conf_mob.value() - 0.55) < 1e-6,
                      "项目 B 没存过 conf_mob，控件却被改掉（老项目会被改坏 ✗）：%r"
                      % (pnl.sp_conf_mob.value(),))

                # ⑤ 换回项目 A ⇒ 回填成 A 那一份（含来源/框选区域）
                pnl.bind(pa)
                check(abs(pnl.sp_conf_mob.value() - 0.55) < 1e-6,
                      "换回项目 A 没回填它那份 conf：%r" % (pnl.sp_conf_mob.value(),))
                check(pnl.cmb_source.currentData() == "window",
                      "换回项目 A 没回填来源：%r" % (pnl.cmb_source.currentData(),))
                check(list(pnl._rect or []) == [10, 20, 300, 400],
                      "换回项目 A 没回填框选区域：%r" % (pnl._rect,))
                check(pnl.cmb_wcap.currentData() == "bitblt",
                      "换回项目 A 没回填抓屏方式：%r" % (pnl.cmb_wcap.currentData(),))
            finally:
                pnl.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ⑤ 源码级：这一排参数的存档口**只有一处**（`_save_project_params` ✓）——
    #   ⚠ 别去数"整个文件里 `update_live(` 出现了几次" ✗：`live.yaml` 里还有别的语义
    #   （「自动测谎」那一组参数也写它 ✓ 而且那组**故意不进项目** ✓ 见 `PROJ_LIVE_KEYS` 上面那段
    #   说明里"那几个键不许按项目存"的理由 ✓）⇒ 数总数会把别人的正常写口当成漂 ✓
    #   （2026-10-09 踩过：那条 pin 报"2 处"红 ✓）。
    src = (ROOT / "gui" / "live_panel.py").read_text(encoding="utf-8")
    check(src.count("self._save_project_params()") >= 3,
          "存档口没汇到一处：`self._save_project_params()` 出现 %d 次"
          % src.count("self._save_project_params()"))
    check('p.sec("live").update(' in src,
          "没把这一排参数写进项目的 `live` 段（那就等于没记住 ✗）")
    # ⭐ 「抓屏方式」必须真的**传到抓帧那一路**（用户 2026-10-09 ✓ 界面上选了 BitBlt，
    #   抓帧却还是 WGC ⇒ 那圈系统描边照样在 ✗ —— 这种"选了没用"正是最该钉的 ✓）
    _lt = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    check("use_wgc=use_wgc" in _lt and "window_capture" in _lt,
          "本地窗口那条路没把「抓屏方式」传下去（界面上选了 BitBlt 也不会有用 ✗）")
    # ⚠⚠ **不许把"抓屏方式"改写成"只抓源窗口"** ✗（2026-10-09 ✓ 我改过一版、被用户否掉 ✓）：
    #   用户原话："**我需要的是盖住的窗口和游戏都显示**，因为**登陆界面是外接的窗口**，
    #   你不显示断线重连会有问题" ✓ —— 也就是：这条路的帧必须**如实反映屏幕** ✓
    #   （BitBlt = 屏幕合成 ✓），否则**外接登录窗不进画面** ⇒ 断线重连 / 测谎判不出界面 ✗。
    check("hwnd=win_hwnd" not in _lt and "is_occluded" not in _lt,
          "又把「抓屏方式」改成「只抓认死的那个窗口」了 ✗ —— 那样盖在游戏上的外接登录窗"
          "就不进画面 ⇒ 断线重连 / 测谎会判不出界面 ✗（用户 2026-10-09 明确否掉过 ✓）")
    # 同一件事的下游：`wincap` / `wgc_capture` 也不许再长回"认死 hwnd"那套 ✗
    _wc = (ROOT / "core" / "wincap.py").read_text(encoding="utf-8")
    _wc_body = _wc.split("def grab_rect(rect, use_wgc=True)")[-1][:400]
    check("hwnd" not in _wc_body,
          "`wincap.grab_rect` 又长出了 hwnd 参数 ✗（那条路已被用户否掉 ✓ 见它 docstring ✓）")
    _wg = (ROOT / "core" / "wgc_capture.py").read_text(encoding="utf-8")
    check("盖在它上面的别的窗口不会进画面" in _wg,
          "`wgc_capture` 没把「**单窗口采集** ⇒ 盖在它上面的窗口不进画面」写清楚 ✗ "
          "（那正是「选错抓屏方式 ⇒ 断线重连判不出界面」的根源 ✓）")


def t_project_save_keeps_deletions():
    """⭐⭐⭐ **删掉的键必须真的落盘**（用户 2026-10-17 ✓ 原话：
    「**重新开gui后，之前删掉的自定义按键还会出现**」✗）。

    ⛔ 病根（`gui/project.py` `_merge_disk` ✓）：多实例落盘走"**只盖本窗口改过的那几处**"
      （`_leaf_diff` ✓）—— 可它**只认改值 / 加键、不认删键** ✗（docstring 里还写着
      "实际也没人这么用" ✗）⇒ 被删的键**不在 `mine` 里** ⇒ 写回去的是**磁盘那份**
      （还带着它 ✓）⇒ **删除永远落不了盘** ✓ ⇒ 重启（从文件读回）又出现 ✓ = 现场 ✓。
      ⚠ 打穿那个假设的正是「自定义按键」面板（`_remove_custom_key` 干的就是 `pop` ✓）。
    判据（三条 ✓）：
      ① 删一个子键 ⇒ `save()` 之后**文件里就没有它** ✓（重启不会复活 ✓）；
      ② **没动过的兄弟键还在** ✓（不许多抹 ✗）；
      ③ **另一个窗口刚写进来的改动要留住** ✓（多实例那条规矩不许回归 ✗）。
    """
    import tempfile
    from pathlib import Path

    from gui.project import Project

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "proj"
        p = Project.create(root, name="t", map_id="1")
        p.set("decision", {"custom_keys": {"甲": "f1", "乙": "f2"},
                           "其它": {"留着": 1}})
        p.save()
        # ① 删掉「乙」（= 面板的 `pop` ✓）
        p.set("decision", {"custom_keys": {"甲": "f1"}, "其它": {"留着": 1}})
        p.save()
        back = Project.open(root).get("decision") or {}
        check("乙" not in (back.get("custom_keys") or {}),
              "删掉的键**又回来了** ✗（这就是「重新开 gui 后还出现」✓）：%r"
              % (back.get("custom_keys"),))
        check((back.get("custom_keys") or {}).get("甲") == "f1"
              and (back.get("其它") or {}).get("留着") == 1,
              "没动过的键被弄丢了 ✗：%r" % (back,))
        # ③ 多实例：另一个窗口新增的键必须留住
        p2 = Project.open(root)
        _d2 = dict(p2.get("decision") or {})
        _d2["custom_keys"] = {"甲": "f1", "丙": "f3"}       # 别的窗口加了一个 ✓
        p2.set("decision", _d2)
        p2.save()
        p.set("decision", {"custom_keys": {"甲": "f1"}, "其它": {"留着": 1}})
        p.save()
        _ck = ((Project.open(root).get("decision") or {}).get("custom_keys") or {})
        check("乙" not in _ck, "第二次保存又把删掉的键带回来了 ✗：%r" % (_ck,))
        check(_ck.get("丙") == "f3",
              "另一个窗口新增的键被我盖掉了 ✗（多实例那条规矩不许回归 ✓）：%r" % (_ck,))


def t_missing_weights_msg():
    """⭐⭐ 权重路径**失效**时要说的那句话（用户 2026-10-04 ✓ 现场："重新导出960的，但是推理报错"
    —— 界面上只有一句 `FileNotFoundError: '…\\detect_v5.engine' does not exist` ✗）。

    为什么需要它（而不是让原始报错当唯一线索）：路径是**点「开始」那一刻**定下、随 `params`
    带进线程的 ✓，而**重新导出引擎会把旧的 `<权重名>.engine` 改名**成
    `<权重名>_<尺寸>.engine`（`tools/export_engine.export_for` ✓ 几个尺寸才能共存 ✓）
    ⇒ 中间任何一个动作（导出 / 改名 / 删归档）都能让它当场失效 ✗ —— 那时只甩一行路径，
    人不知道该干什么 ✓。

    钉四件：
      ① 文件真在 ⇒ **不吭声**（`None` ✓，别给正常路径平白加一行红字 ✗）；
      ② 文件不在（**带目录**的路径）⇒ 有话说、含「重选一份」（怎么办 ✓）、含路径（哪份 ✓）；
      ③ ⚠ **「重选一份」必须排在路径前面**：界面那行是 `lbl_stats`（**单行标签**，长了会被截 ✓）
         ⇒ 该被截掉的是路径，不是"怎么办" ✗；
      ④ 裸名字（`yolo26n.pt` 这种**官方权重名**）⇒ **不吭声** ✓
         （ultralytics 会自己联网下载 ✓，拿 `is_file()` 拦它是误伤 ✗）。
    """
    import shutil
    import tempfile

    from gui.live_thread import missing_weights_msg

    d = Path(tempfile.mkdtemp(prefix="w_gone_"))
    try:
        ok = d / "detect_v5.pt"
        ok.write_text("x", encoding="utf-8")
        check(missing_weights_msg(str(ok)) is None,
              "文件明明在还报「不在了」⇒ 正常路径平白多一行红字 / 弹窗 ✗：%r"
              % (missing_weights_msg(str(ok)),))
        gone = d / "detect_v5.engine"
        m = missing_weights_msg(str(gone))
        check(m and "重选一份" in m,
              "文件不在了却没告诉人「怎么办」（用户就是被这一行卡住的 ✗）：%r" % (m,))
        check(str(gone) in m, "没带上到底是哪份权重：%r" % (m,))
        check(m.index("重选一份") < m.index(str(gone)),
              "「怎么办」排在路径**后面** ⇒ 界面那行单行标签一截就把「怎么办」截没了 ✗：%r"
              % (m,))
        for bare in ("yolo26n.pt", "yolo26n.engine", ""):
            check(missing_weights_msg(bare) is None,
                  "裸名字 %r 被当成「文件不在」了 —— 官方权重名 ultralytics 会自己联网下载，"
                  "拦下来是误伤 ✗" % (bare,))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t_export_done_takes_over():
    """⭐⭐⭐ 导出成功后**当场接管新引擎**（用户 2026-10-04 ✓ 原话："重新导出960的，但是推理报错"）。

    现象链路（三处都在代码里 ✓）：
      ① `tools/export_engine.export_for` 把 ultralytics 写出的 `<权重名>.engine` **改名**成
         `<权重名>_<尺寸>.engine` ⇒ **旧名字当场没了** ✗；
      ② 权重路径是**点「开始」那一刻**定下、带进线程的 ✓ ⇒ 已经在跑的实时还捏着旧路径 ✗；
      ③ 首次加载（点「开始推理」）⇒ `FileNotFoundError: … does not exist` ✗。

    钉五件：
      ① 「自动」时导出完 ⇒ 下拉**自动项指向刚导出的那份**（界面不许还是旧的 ✗）、
         `cur_weights()` 也换成它 ✓（重新点「开始」即用它 ✓）；被改名走的那份**不在候选里** ✓；
      ② 手选的那份**正是被改名的那个** ⇒ 自动跟到新引擎**并落盘** ✓
         （否则那份选择会永远指向一个不存在的文件 ✗）；
      ③ 手选的是**别的、还在**的引擎 ⇒ **不许抢**（只在"被改名的正是它"时才跟 ✓）；
      ④ 状态行文案要写着「**重新点「开始」**」（含糊的"下次启动生效"正是这次踩空的来源 ✗）。
    """
    import os as _os
    import shutil
    import tempfile

    from PyQt5.QtWidgets import QApplication, QMessageBox

    import gui.live_panel as lp
    from gui.project import Project

    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")

    tmp = Path(tempfile.mkdtemp(prefix="exp_take_"))
    _info = QMessageBox.information
    QMessageBox.information = staticmethod(lambda *a, **k: None)   # 离屏：弹窗会挂住 ✗
    try:
        proj = Project.create(tmp, name="用例项目", map_id="105090600")
        t0 = 1_700_000_000.0

        def _mk(rel, off):
            f = proj.root / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("x", encoding="utf-8")
            _os.utime(str(f), (t0 + off, t0 + off))
            return f

        _mk("models/detect_v5.pt", 0)
        old = _mk("models/detect_v5.engine", 10)        # 当前"在用"的那份
        other = str(_mk("models/detect_v4_640.engine", 5))   # 另一份**还在**的引擎

        pnl = lp.LivePanel()
        try:
            pnl.bind(proj)
            check(pnl.cur_weights() == str(old),
                  "前置不对：自动该挑到最新那份引擎 ✓：%r" % (pnl.cur_weights(),))

            # ① 模拟 `export_for` 那步"改名"（`os.replace` ✓ 同那条实现）
            new = proj.root / "models" / "detect_v5_960.engine"
            _os.replace(str(old), str(new))
            pnl._on_export_done(str(new))
            check(pnl.cur_weights() == str(new),
                  "导出完「自动」没指向刚导出的那份 ⇒ 点「开始」还是用旧路径 ⇒ 再报一次同样的错 ✗"
                  "：%r" % (pnl.cur_weights(),))
            check(pnl.cmb_weights.findData(str(old)) < 0,
                  "被改名走的那份还留在候选里 ⇒ 列表没跟着重扫 ✗")
            check(new.name in pnl.cmb_weights.itemText(0),
                  "下拉里「自动」那项显示的还是旧名字（界面与实际不一致 ✗）：%r"
                  % (pnl.cmb_weights.itemText(0),))
            check("重新点" in pnl.lbl_stats.text(),
                  "状态行没说要「重新点开始」（含糊的「下次启动生效」正是踩空的来源 ✗）：%r"
                  % (pnl.lbl_stats.text(),))

            # ② 手选的那份**正是被改名的那个** ⇒ 跟到新引擎 + 落盘
            #   ⚠ 必须写**面板手上那个对象**（另开 `Project.open` 写盘 ⇒ 面板内存里那份
            #     还是空的 ⇒ 它会走"自动"，用例就成了假红/假绿 ✗）
            proj.sec("live")["weights"] = str(old)
            proj.save()
            pnl.bind(proj)
            check(pnl.cur_weights() == str(old),
                  "前置不对：存着的手选该被选中 ✓：%r" % (pnl.cur_weights(),))
            pnl._on_export_done(str(new))
            check(Project.open(proj.root).sec("live").get("weights") == str(new),
                  "手选被改名后没跟到新引擎 ⇒ 那份选择会**永远指向不存在的文件** ✗")
            check(pnl.cmb_weights.currentData() == str(new),
                  "手选跟过去了、下拉却没选中它（显示与实际不一致 ✗）：%r"
                  % (pnl.cmb_weights.currentData(),))

            # ③ 手选的是**别的、还在**的引擎 ⇒ 不许抢
            pnl.cmb_weights.setCurrentIndex(pnl.cmb_weights.findData(other))
            pnl._on_weight_pick()
            pnl._on_export_done(str(new))
            check(pnl.cur_weights() == other,
                  "把用户手选的**另一份**引擎抢走了（只有「被改名的正是它」时才该跟 ✗）：%r"
                  % (pnl.cur_weights(),))
        finally:
            pnl.close()
    finally:
        QMessageBox.information = _info
        shutil.rmtree(tmp, ignore_errors=True)


def _probe_half_wiring_in_child():
    """在**子进程**里真调一遍"接线"（**先 torch 后 Qt** ⇒ 本套件那个进程起不来 torch ✗）。

    **为什么要这么绕**：本套件是 **Qt 进程**，而 Windows 上"**先 Qt 后 torch**"会让
    `torch\\lib\\c10.dll` 初始化直接炸（`OSError [WinError 1114]` ✗ —— 仓库纪律，
    见 `tools/lie_dets_worker.py` 顶部）。子进程里**先 import ultralytics** 就没这个问题 ✓
    ⇒ "logger/handler 上真装上了 filter"能**真验一遍** ✓，而不是只留在源码钉上 ✗
    （源码钉只能证明"写了"，证明不了"接上了" ✓）。

    返回 True = 真调过 ✓；False = 子进程也起不来 torch（那就退回源码钉 ✓ **如实说** ✓）。
    """
    import re
    import shutil
    import subprocess
    import sys
    import tempfile

    body = (
        "import logging, sys\n"
        "sys.stdout.reconfigure(encoding='utf-8')\n"
        "from ultralytics.utils import LOGGER, deprecation_warn\n"   # ⚠ 先 torch ✓
        "sys.path.insert(0, %r)\n"
        "from gui.live_thread import mute_half_deprecation\n"
        "got = []\n"
        "class Cap(logging.Handler):\n"
        "    def emit(self, r): got.append(r.getMessage())\n"
        "cap = Cap(); LOGGER.addHandler(cap)\n"
        "mute_half_deprecation()\n"
        "deprecation_warn('half', 'quantize')\n"
        "a = list(got); got.clear()\n"
        "deprecation_warn('end2end', 'nms')\n"
        "print('HALF=%%d OTHER=%%d' %% (len(a), len(got)))\n"
    ) % str(ROOT)
    tmp = Path(tempfile.mkdtemp(prefix="halfw_"))
    p = tmp / "probe.py"
    p.write_text(body, encoding="utf-8")
    try:
        env = dict(os.environ)
        env.pop("QT_QPA_PLATFORM", None)          # 子进程不碰 Qt ✓（先 torch 才有可能 ✓）
        r = subprocess.run([sys.executable, "-X", "utf8", str(p)], cwd=str(ROOT),
                           capture_output=True, text=True, timeout=180, env=env)
        out = (r.stdout or "") + (r.stderr or "")
        m = re.search(r"HALF=(\d+) OTHER=(\d+)", out)
        if not m:
            print("      （子进程接线检查没结果：%s）"
                  % (" / ".join(out.strip().splitlines()[-2:])[:160],))
            return False
        check(int(m.group(1)) == 0,
              "接线之后那句告警还在往外打（`mute_half_deprecation` 没真接上 ✗）：%s 条"
              % m.group(1))
        check(int(m.group(2)) >= 1, "接线把 ultralytics **别的**告警一起吞了 ✗")
        return True
    except Exception as e:                        # noqa: BLE001
        print("      （子进程接线检查跳过：%s：%s）" % (type(e).__name__, str(e)[:80]))
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_half_warn_not_spamming():
    """⭐⭐ 「每帧一条 ultralytics 弃用告警」必须静音（用户 2026-10-04 ✓ 原话："指令堆积了，看看log"）。

    **现场证据**：`stdout.log` **406 MB / 27 分钟**，尾部采样 89888 行里 **99.9% 是同一句**
    `WARNING 'half' is deprecated … Use 'quantize' instead.` ⇒ ≈ **每秒 60 条 = 每帧一条** ✗
    （我们每帧都 `model.predict(…, half=_half)` ✓，而 ultralytics 8.4 把 `half` 改名
    `quantize` 并在参数校验里 `deprecation_warn("half", "quantize")` ✓）。

    钉四件：
      ① 静音后**那条**记录收不到（用 `deprecation_warn("half", "quantize")` **真调一次** ✓）；
      ② **别的** ultralytics 告警照旧收得到（`deprecation_warn("end2end", "nms")` ⇒ 收得到 ✓
         —— 一锅端会把排查时的眼睛一起瞎掉 ✗）；
      ③ **幂等**（再调一次不报错、那条仍然静音 ✓）；
      ④ 源码钉：两处 `predict` 都改用 `**_half_kw`、且 **`.engine` 那条路把它清空** ✓
         （只静音不接上 / 引擎照样传 half ⇒ 告警照旧 ✗）+ 加载模型时**真的调了**
         `mute_half_deprecation()` ✓（写好了没人调 = 没修 ✗）。
    """
    import logging

    from gui.live_thread import _DropHalfDeprecation, mute_half_deprecation

    def _dep_msg(arg, new=None):
        """照 `ultralytics.utils.deprecation_warn` 的**原文**拼（一字不差才不会假绿 ✓）。"""
        m = "'%s' is deprecated and will be removed in the future." % arg
        return m + ((" Use '%s' instead." % new) if new else "")

    got = []

    class _Cap(logging.Handler):
        def emit(self, record):
            got.append(record.getMessage())

    # ① 判据本身（**纯逻辑** ✓ 不依赖 ultralytics 能不能 import —— 本套件是 **Qt 进程**，
    #    "先 Qt 后 torch" 会让 `torch\lib\c10.dll` 初始化直接炸 `OSError WinError 1114` ✗
    #    见 `tools/lie_dets_worker.py` 顶部那段说明 ✓）
    lg = logging.getLogger("用例-half-过滤器")
    lg.propagate = False                  # 别串到 root（那会把 pytest/控制台搅进来 ✓）
    cap = _Cap()
    lg.addHandler(cap)
    lg.addFilter(_DropHalfDeprecation())
    try:
        lg.warning(_dep_msg("half", "quantize"))
        check(not got,
              "那句「每帧一条」的告警没被过滤掉（stdout.log 就是被它写成 400MB 的 ✗）：%r"
              % (got,))
        # ② 别的告警不许被吞（吞了就等于把 ultralytics 的话全静音 ✗）
        lg.warning(_dep_msg("end2end", "nms"))
        check(len(got) == 1 and "'end2end' is deprecated" in got[0],
              "把 ultralytics **别的**告警一起吞了（排查时的眼睛没了 ✗）：%r" % (got,))
    finally:
        lg.removeHandler(cap)
        lg.filters = []

    # ③ 接线（**能 import ultralytics 才真调** ✓；Qt 进程里那一步会 WinError 1114 ✗）
    _wired = False
    try:
        from ultralytics.utils import LOGGER as _ul, deprecation_warn
        _ul.addHandler(cap)
        try:
            check(mute_half_deprecation() is None, "静音函数不该抛异常 ✓")
            got.clear()
            deprecation_warn("half", "quantize")
            check(not got,
                  "接线之后那句告警还在往外打（`-_half_kw` 或 `mute…` 没接上 ✗）：%r" % (got,))
            deprecation_warn("end2end", "nms")
            check(any("'end2end'" in m for m in got),
                  "接线把 ultralytics 别的告警一起吞了 ✗：%r" % (got,))
            check(mute_half_deprecation() is None, "第二次调用（幂等）不该抛异常 ✗")
            got.clear()
            deprecation_warn("half", "quantize")
            check(not got, "第二次调用后不静音了（重复装 filter 出问题 ✗）：%r" % (got,))
            _wired = True
        finally:
            _ul.removeHandler(cap)
    except Exception as e:                                # noqa: BLE001
        print("      （本进程 import ultralytics 失败：%s: %s ⇒ 判据本身已验 ✓，"
              "接线改由子进程验 ✓）" % (type(e).__name__, str(e)[:60]))
    if not _wired:
        _wired = _probe_half_wiring_in_child()            # ⚠ 别只留源码钉 ✓

    print("      （接线检查：%s）" % ("已真调 ultralytics ✓" if _wired
                                 else "跳过 —— 本进程 import 不了 torch ✓（Qt 进程那条纪律 ✓）"))

    # ④ 源码钉（静态那两处 + 接上没接上）
    # ⚠ 2026-10-20：**主循环那次推理搬进 `runtime/frame_infer.py`** ⇒ 「两处 predict 都带
    #   `**_half_kw`」要**跨两个文件**数 ✓（预演那处仍在 `live_thread` ✓）
    src = ((ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
           + (ROOT / "runtime" / "frame_infer.py").read_text(encoding="utf-8"))
    check(src.count("**_half_kw") == 2,
          "两处 `predict` 不是都走 `**_half_kw`（漏一处 ⇒ 那条路每帧照旧打一条 ✗）：%d"
          % src.count("**_half_kw"))
    check("_half_kw = {}" in src,
          "`.engine` 那条路没把 half 清掉（FP16 已固化在引擎里，传了只触发告警 ✗）")
    calls = src.count("mute_half_deprecation()") - src.count("def mute_half_deprecation()")
    check(calls >= 1,
          "写好了 `mute_half_deprecation` 却**没人调** ⇒ 等于没修 ✗")


def t_engine_export_one_click():
    """⭐⭐ 「一键导出该尺寸引擎」（用户 2026-10-03 ✓ 原话："想真用别的尺寸：重新导出那个尺寸
    的引擎 —— 就不能填写后自动导出吗？不合适的话弹提示指引然后搞个按钮一键导出也行"）。

    钉五件：
      ① **从当前权重推出该导哪份 `.pt`**：`.engine` ⇒ 同目录的 `.pt` ✓ ——
         先认"**同名**"✓；再认"**去掉结尾尺寸**"✓（`m1_960.engine` ⇒ `m1.pt` ✓，
         ⚠⚠ 用户 2026-10-05 撞的那个框 ✓ 见下面那条 ✓）；
         都没有 ⇒ 返回 `None` + 指路（**不许**去 `runs/**/best.pt` 瞎翻 ✗ ——
         那会导出一份"名字对、内容错"的引擎 ✓）；
      ② **输出名带尺寸**（`<stem>_<imgsz>.engine` ✓）⇒ 几个尺寸共存、互不覆盖 ✓；
         而实时挑权重是**按 mtime 取最新的 `.engine`** ✓ ⇒ 刚导出的下次启动自动生效 ✓；
      ③ **后台线程**（`EngineExportThread` 是 `QThread` ✓）—— 导出几分钟，放主线程 = 界面假死 ✗；
      ④ **复用已有那条实现**（`tools/export_engine.export_for` ✓ 不许另写 ✗ —— 它还处理了
         "权重路径含中文"那个坑 ✓ 而本项目项目名全中文 ✓）；
      ⑤ 面板有按钮，且**成功 / 失败都要有反馈** ✗（不许点了没动静 ✓）。
    """
    import tempfile
    from pathlib import Path as _P

    from PyQt5.QtCore import QThread
    from PyQt5.QtWidgets import QApplication
    from tools.export_engine import pt_for
    import gui.live_panel as lp

    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")

    d = _P(tempfile.mkdtemp(prefix="eng_export_"))
    (d / "m1.pt").write_text("x", encoding="utf-8")
    (d / "m1.engine").write_text("x", encoding="utf-8")
    got, why = pt_for(str(d / "m1.engine"))
    check(got is not None and got.name == "m1.pt",
          "从 .engine 没推出同目录同名的 .pt（这是「一键导出」的前置 ✗）：%r / %r"
          % (got, why))
    got2, why2 = pt_for(str(d / "没有这个.engine"))
    check(got2 is None and why2,
          "没有同名 .pt 时该**返回 None + 指路**（不猜 ✗）：%r" % (got2,))
    # ⭐⭐⭐⭐⭐ **带尺寸的引擎名**（`<pt名>_<尺寸>.engine` ✓ = 本模块自己导出的默认名 ✓）——
    #   ⚠⚠ 用户 2026-10-05 撞的那个框：`detect_v5_960.engine` ✓ 旁边明明有 `detect_v5.pt` ✓
    #     却在找 `detect_v5_960.pt` ✗ ⇒ 弹「找不到……」✗（**导出器自己从不生成那个名字** ✗✗）。
    (d / "m1_960.engine").write_text("x", encoding="utf-8")
    got4, why4 = pt_for(str(d / "m1_960.engine"))
    check(got4 is not None and got4.name == "m1.pt",
          "带尺寸的引擎（`m1_960.engine` ✓、只会旁边那份 `m1.pt` ✓）没推出 `m1.pt` ✗：%r / %r —— "
          "⚠ 导出时**故意**带尺寸改名 ✓ 而查找只认「同名」✗ ⇒ 用带尺寸引擎时**必然弹框** ✓"
          % (got4, why4))
    got5, why5 = pt_for(str(d / "没有这个_960.engine"))
    check(got5 is None and why5,
          "带尺寸且**确实没有**来源 .pt 时，照样**返回 None + 指路**（不猜 ✗）：%r" % (got5,))
    got3, _ = pt_for(str(d / "m1.pt"))
    check(got3 is not None and got3.name == "m1.pt", "给 .pt 时该原样用它：%r" % (got3,))

    src = (ROOT / "tools" / "export_engine.py").read_text(encoding="utf-8")
    check('"%s_%d.engine"' in src,
          "输出的引擎名没带尺寸 ⇒ 换个尺寸导出会把原来那个覆盖掉 ✗（640/960 没法共存 ✓）")
    check("os.replace(" in src, "改名那步没做 ⇒ 还是覆盖同名引擎 ✗")

    check(issubclass(lp.EngineExportThread, QThread),
          "导出没放在 QThread 里 ⇒ 几分钟的重编译会把界面冻住 ✗")
    _w = (ROOT / "gui" / "live_panel.py").read_text(encoding="utf-8")
    check("from tools.export_engine import export_for" in _w,
          "没复用已有的导出实现（另有实现 = 迟早两套不一致 ✗）")
    # ⚠ 必须断言**接上了**，不是"函数定义了" ✗ —— 反向验证实测：只看函数名在文件里出现，
    #   把 `connect(...)` 那一行删掉**照样绿** ✓（定义还在 ✓）⇒ 得直接看 connect ✓。
    for _name, _sig in (("_on_export_done", "_t.done.connect(self._on_export_done)"),
                        ("_on_export_failed", "_t.failed.connect(self._on_export_failed)"),
                        ("_on_export_finished",
                         "_t.finished.connect(self._on_export_finished)")):
        check(_sig in _w,
              "`%s` 没接到导出线程上 ⇒ 那一步**没有反馈** ✗（点了没动静 ✓）" % _name)
    check("btn_export_engine" in _w and "imgsz_mismatch" in _w,
          "「导出该尺寸引擎」按钮 / 「imgsz 不生效」提示没有接上 ✗")

    # ⑥⑦⑧ ⭐⭐ **导出要有日志和进度条**（用户 2026-10-04 ✓ 原话："新的imgsz导出时，
    #   数据工作台没有日志和进度条"）—— 那几分钟里原来**一个字都看不到** ✗。
    from unittest.mock import patch as _patch

    import tools.export_engine as _ee

    # ⑥ **行为级**：导出过程的打印真的被 tee 进 `line`（不只看"信号存在"✗）
    _lines, _done_sig, _fail_sig = [], [], []
    _th = lp.EngineExportThread(str(d / "m1.pt"), 960)
    _th.line.connect(lambda s: _lines.append(str(s)))
    _th.done.connect(lambda p: _done_sig.append(p))
    _th.failed.connect(lambda m: _fail_sig.append(m))

    def _fake_export(weights, imgsz, **_kw):
        print("ultralytics 假装在导（imgsz=%d）" % imgsz)      # 导出过程会 print ✓
        return str(d / ("m1_%d.engine" % imgsz))

    with _patch.object(_ee, "export_for", _fake_export):
        _th.run()                                  # 同步跑一遍（不起线程 ✓）
    check(_done_sig and str(_done_sig[-1]).endswith("m1_960.engine"),
          "导出的成功信号不对（失败信号：%r）：%r" % (_fail_sig, _done_sig))
    check(any("ultralytics 假装在导" in s for s in _lines),
          "导出过程打印的东西**没进 `line`** ⇒ 界面上还是没日志 ✗：%r" % (_lines,))
    check(any("开始导出" in s for s in _lines) and any("✓ 导出完成" in s for s in _lines),
          "开头/结尾那两句没进日志（人看不出这几分钟在干什么 ✗）：%r" % (_lines,))

    # ⑦ 进度窗：**忙式**进度条（拿不到真百分比 ⇒ 不编 ✗）+ 只读日志框 + 结束收起来
    from gui import theme as _theme
    with _patch.object(_theme, "save_window", lambda *a, **k: None), \
            _patch.object(_theme, "load_window", lambda *a, **k: None):
        _dlg = lp.EngineExportDialog(960)
        try:
            check(_dlg.bar.minimum() == 0 and _dlg.bar.maximum() == 0,
                  "进度条不是**忙式**（TensorRT 不给真百分比 ⇒ 编一个比不给更坏 ✗）："
                  "(%d, %d)" % (_dlg.bar.minimum(), _dlg.bar.maximum()))
            check(_dlg.log.isReadOnly(), "日志框不是只读的 ✗")
            check(_dlg.log.objectName() == "Log",
                  "日志框没走全局日志样式（`objectName=Log` ✓ 同标注/训练页 ✓）")
            _dlg.append("hello 导出日志")
            check("hello 导出日志" in _dlg.log.toPlainText(),
                  "`append` 没把日志写进框里 ✗")
            _dlg.mark_finished(ok=True)
            check(not _dlg._timer.isActive(), "导出结束后计时器还在跑 ✗")
            check("完成" in _dlg.lbl_time.text(),
                  "结束后没标「完成」：%r" % _dlg.lbl_time.text())
            check(_dlg.btn_hide.text() == "关闭", "结束后按钮该变成「关闭」✓")
        finally:
            _dlg.close()

    # ⑧ 源码级：`line` 真的接到进度窗上（⚠ 断言 `connect` 那一行 ✓ —— 只看有 append 方法
    #   抓不住"没接上"✗，这种盲区这个仓库踩过好几次 ✓）
    _w2 = (ROOT / "gui" / "live_panel.py").read_text(encoding="utf-8")
    check("_t.line.connect(_dlg.append)" in _w2,
          "导出线程的 `line` 没接到进度窗上 ⇒ 接了也没有日志 ✗")


def t_mmap_map_id_follows_project():
    """⭐⭐ 送 A 机的地图 id **必须跟着项目走**，且状态**常显**（用户 2026-10-04 ✓ 原话：
    "开启自动怎么没反应了" ⇒ 查下来：项目里写着 `105040306` ✓，而 B **一直**给 A 推
    旧图 `105040303` ✗（`perf.log` 里每 5 秒一条 `no-such-map` ✓）⇒ A 收不到 → 定位不了 →
    自动不动作 ✓。用户追问原话："这是我手动更换的地图，A 机理应也收到这个 id，但实际没有
    任何变化" —— 一针见血 ✓）。

    两个毛病（都在这一条上翻出来 ✓）：
      ① ⛔ **面板缓存不跟着项目走** ✗：`_mmap_mid` 只在 `bind(project)` 和"路线页手动推"时
         更新 ⇒ 项目里改了它不知道 ✓ ⇒ 实时线程拿旧图去推 ✓
         ⇒ 修：`_sync_mmap_map_id()`（**权威源只有 `project.map_id`** ✓）+ 在 `start()` /
         `start_infer()` 里各叫一次 ✓。
      ② ⛔ **被拒那句只进 stdout** ✗（重定向后还会被**缓冲**住 ⇒ 人根本看不见 ✓）
         ⇒ 修：线程把推图现状塞进 stats（`mmap_push_state()` ✓），面板 `_on_stats` 里
         **常显一行**（`lbl_mmap` ✓）。

    钉四件：① 面板缓存落后于项目 ⇒ sync 能纠回来（幂等 ✓）；
    ② `start()` / `start_infer()` **真的会调它**（源码钉 ✓ 行为级要起真线程，不值 ✗）；
    ③ 那一行三种状态说得对（被拒 / 成功 / 来源=画面 ✓）；
    ④ 状态数据的**唯一出处**是线程的 `mmap_push_state` ✓（面板不许自己拼 ✗）。
    """
    import shutil
    import tempfile

    from gui import live_panel as _lpm
    from gui.project import Project

    # ⚠ 用**真 Project**（假对象要补的键太多 ✗：`bind` 会读 `train` 段、`root` 等 ✓）
    tmp = tempfile.mkdtemp(prefix="mmap_id_")
    app, _lp, panel, _made, _orig = _panel()
    try:
        proj = Project.create(Path(tmp) / "proj", name="用例", map_id="105040306")
        panel.bind(proj)                       # ← bind 会把项目里的 map_id 带过来 ✓
        check(panel._mmap_mid == "105040306",
              "bind 之后面板拿的不是项目里的图：%r" % (panel._mmap_mid,))

        panel._mmap_mid = "105040303"           # 模拟现场：缓存还停在旧图 ✓
        check(panel._sync_mmap_map_id() is True and panel._mmap_mid == "105040306",
              "缓存落后于项目时没纠回来（现场就是这样一直推旧图 ✗）：%r" % (panel._mmap_mid,))
        check(panel._sync_mmap_map_id() is False,
              "sync 不幂等（每次都当'变了'⇒ 白重推 ✓）")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    check(panel._mmap_mid == "105040306",
          "bind 之后面板拿的不是项目里的图：%r" % (panel._mmap_mid,))

    panel._mmap_mid = "105040303"                 # 模拟现场：缓存还停在旧图 ✓
    check(panel._sync_mmap_map_id() is True and panel._mmap_mid == "105040306",
          "缓存落后于项目时没纠回来（现场就是这样一直推旧图 ✗）：%r" % (panel._mmap_mid,))
    check(panel._sync_mmap_map_id() is False,
          "sync 不幂等（每次都当'变了'⇒ 白重推 ✓）")

    # ③ 那一行三种状态
    panel._on_stats({"mmap_push": {"want": "105040306", "ok": "",
                                   "reply": "A 机拒绝了：no-such-map 这张图还没框过（105040306）",
                                   "src": "stream"}})
    _t1 = panel.lbl_mmap.text()
    check("105040306" in _t1 and "no-such-map" in _t1,
          "被拒时那一行没把「推的是哪张图 + A 的原话」说全 ✗：%r" % (_t1,))
    panel._on_stats({"mmap_push": {"want": "105040306", "ok": "105040306",
                                   "reply": "ok", "src": "stream"}})
    check("收下" in panel.lbl_mmap.text(),
          "成功时那一行没说清「A 收下了」✗：%r" % (panel.lbl_mmap.text(),))
    panel._on_stats({"mmap_push": {"want": "", "ok": "", "reply": "", "src": "live"}})
    check("不走" in panel.lbl_mmap.text(),
          "来源＝从画面时那一行没说「不走 A 的推流」✗：%r" % (panel.lbl_mmap.text(),))

    src = (ROOT / "gui" / "live_panel.py").read_text(encoding="utf-8")
    # ⚠ 这个文件里**有两个** `def start(self):` ✗（还有一个是引擎导出窗的 ✓）⇒ 必须取**最后一个**
    #   （= LivePanel 那个 ✓）—— 第一版用 `index` 取到前面的 ⇒ 假红 ✓（本轮踩过 ✓）
    check("self._sync_mmap_map_id()" in src[src.rindex("def start(self):"):][:400],
          "`start()` 没重读项目里的地图 id ⇒ 换了图还是推旧的 ✗（本轮现场就是这么来的 ✓）")
    _si = src[src.rindex("def start_infer(self):"):][:600]
    check("self._sync_mmap_map_id()" in _si,
          "`start_infer()` 没重读（「开始之后才改项目地图」那种情况救不回来 ✗）")
    check("def _sync_mmap_map_id" in src and 'p.get("map_id")' in
          src[src.index("def _sync_mmap_map_id"):][:1800],
          "重读的**权威源**不是 `project.map_id` ✗（不许另立一份口径 ✓）")
    check("self._show_mmap_push(s.get(\"mmap_push\"))" in src,
          "`_on_stats` 没刷那一行 ⇒ 常显是假的 ✗")
    _lt = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    check("\"mmap_push\": self.mmap_push_state()," in _lt and "def mmap_push_state" in _lt,
          "线程没把推图现状交出来（面板就只能自己猜 ✗）")


def t_pot_zero_means_dead():
    """⭐⭐ **血条读出 0 要采信（连续几拍）—— 这就是"HP 不变 0%"的修复点**
    （用户 2026-10-05 ✓ 原话："角色死了之后画面中心会出现这个弹窗，并且 HP 是 0% 为什么自动
    喝药的 Hp 当前显示不是 0%？"）。

    病根：原来 `if ratio > 0.0: _pot_vals[0] = ratio`（**没有 else** ✗）—— 本意是
    "0% = 读丢/抖动 ⇒ 保持上次值" ✓，可**真的 0% 也是 0%** ✗ ⇒ 人死了读数反被丢掉
    ⇒ 面板一直显示**死前那个值** ✓（实测序列 0.80→0.78→0→0→0：旧规则停在 **0.78** ✗、
    新规则连到门限拍认 **0** ✓）。钉五件：
      ① 单次 0 **不算数**（抗抖动的老本意不能丢 ✗）；② 连到 `POT_ZERO_N` 拍 ⇒ 认 0 ✓；
      ③ 之后有 >0 ⇒ 立刻恢复（连零计数归零 ✓）；
      ④ **HP/MP 都走这一个出口** ✓（源码钉 —— 别再各写一份 `if ratio > 0` ✗）；
      ⑤ ⭐ **门限要够长，顶得住"血条受击闪烁"**（用户 2026-10-05 第二轮 ✓ 原话："血条闪烁是有
         动画的，这段时间 Hp 识别不到，会频繁的让 gui 多一行少一行"）：3 拍（0.3s）顶不住 ✗
         ⇒ 判据在"空 / 不空"之间翻 ⇒ 面板红字一亮一灭 ✓（= 用户看到的"多一行少一行" ✓）
         ⇒ 提到 **5 拍（0.5s）** ✓；⚠ 它同时就是**死亡检测延迟**（`agent._death_beat` ✓）
         ⇒ 只许"略微增加"、不许当旋钮拧 ✗。
    """
    from gui import live_thread as lt

    n = int(lt.POT_ZERO_N)
    check(n >= 5,
          "连零门限太小（血条**受击会闪** ⇒ 那几拍采样出来就是空 ⇒ 门限太短会被闪满 ⇒ "
          "判据翻来翻去 ⇒ 面板红字一亮一灭 / 版面多一行少一行 ✗）：%r" % n)

    v, z = lt._pot_accept(0.0, 0, 0.8, zero_n=n)
    check(v == 0.8 and z == 1, "单次读到 0 就改了值（抗抖动丢了 ✗）：%r/%r" % (v, z))
    v, z = lt._pot_accept(0.0, n - 1, 0.8, zero_n=n)
    check(v == 0.0, "连续 %d 拍读到 0 还不认（人死了却不变 0 ✗）：%r" % (n, v))
    v, z = lt._pot_accept(0.55, z, 0.0, zero_n=n)
    check(v == 0.55 and z == 0, "血量回来之后没恢复（连零计数也没归零 ✗）：%r/%r" % (v, z))

    src = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    check(src.count("_pot_accept(") >= 3,
          "HP/MP 没都走 `_pot_accept`（唯一采信出口 ✗）：%d 处" % src.count("_pot_accept("))
    check("if ratio > 0.0:   # 0% = 丢失/抖动，保持上次值" not in src,
          "旧的 `if ratio > 0.0:` 分支还在（真的 0% 仍会被丢 ✗）")
    # ⚠ 2026-10-13 阶段 1 编排收口：这段从 `_run` 主循环抽进了 `_read_hpmp()` 方法，
    #   参数名 `_pot_vals` → `pot_vals`（方法内简洁名）⇒ 锚点跟着变 ✓。语义不变：仍是
    #   "血条空了写进 `ws.player.dead`（死亡检测的数据源）"。
    check("ws.player.dead = bool(pot_vals[0] <= 0.0)" in src,
          "没把「血条空了」写进 `ws.player.dead`（死亡检测就没数据源 ✗）")


def t_imgsz_mismatch_auto_exports():
    """⭐⭐ 「imgsz 不生效」那个弹窗：**点确定就把导出跑起来**（用户 2026-10-05 ✓ 原话："现在调成
    640imgsz 会弹窗，期望这个弹窗不管写什么，点确定就自动后台完成导出步骤"）。

    **老口径**（2026-10-03）只"指到按钮、由人再点一次" ✗ ⇒ 现在**一次点击** ✓
    （"要不要花几分钟重编译"就在这个弹窗里问过了 ⇒ **不再叠一层确认** ✗）。
    钉三件：
      ① 点**确定** ⇒ 真起后台导出（线程 + 进度窗都在 ✓，尺寸 = 你填的那个 ✓）；
      ② 点**取消** ⇒ **什么都不起** ✓（"不管写什么都导"是不行的 ✗ —— 那会偷偷烧几分钟 GPU ✓）；
      ③ 源码钉：弹窗那条走 `_start_engine_export` ✓，而按钮 `_on_export_engine` **仍自带确认** ✓
         （两条路各问一次、**不叠** ✓）。
    """
    import shutil
    import tempfile

    from PyQt5.QtWidgets import QMessageBox

    from gui import live_panel as _lp

    app, lp, p, made, orig = _panel()
    _dir = Path(tempfile.mkdtemp(prefix="enginex_"))
    _pt = _dir / "fake.pt"
    _pt.write_bytes(b"x")
    _started, _dlgs = [], []

    class _Sig:                                  # 假信号：只需要能 `.connect` ✓
        def connect(self, *_a, **_k):
            pass

    class _FakeThread:                           # ⚠ 别真跑 TensorRT（几分钟 + 吃 GPU）✗
        def __init__(self, pt, imgsz, device="0", parent=None):
            _started.append((str(pt), int(imgsz)))
            self.done = self.failed = self.finished = self.line = _Sig()

        def isRunning(self):
            return False

        def start(self):
            pass

    class _FakeDlg:
        def __init__(self, imgsz, parent=None):
            _dlgs.append(int(imgsz))

        def append(self, *_a):
            pass

        def show(self):
            pass

        def close(self):
            pass

    _old_t, _old_d = _lp.EngineExportThread, _lp.EngineExportDialog
    _old_q = QMessageBox.question
    try:
        p.cur_weights = lambda: str(_pt)         # 唯一出口：指到一份存在的 .pt ✓
        _lp.EngineExportThread, _lp.EngineExportDialog = _FakeThread, _FakeDlg

        # ① 点「确定」⇒ 真起导出（尺寸 = 填的那个 ✓）
        QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Ok)
        p._on_imgsz_mismatch(640, 960)
        check(_started == [(str(_pt), 640)],
              "点确定没把后台导出跑起来（或尺寸不对 ✗）：%r" % (_started,))
        check(640 in _dlgs, "导出起了却没有进度窗（人看不到日志/进度 ✗）：%r" % (_dlgs,))

        # ② 点「取消」⇒ 什么都不起（不许偷偷烧 GPU ✗）
        _started.clear()
        _dlgs.clear()
        QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Cancel)
        p._on_imgsz_mismatch(640, 960)
        check(not _started, "点了取消还是把导出跑起来了（偷偷烧几分钟 GPU ✗）：%r"
              % (_started,))

        # ③ 源码钉
        _src = (ROOT / "gui" / "live_panel.py").read_text(encoding="utf-8")
        _mm = _src[_src.index("def _on_imgsz_mismatch"):]
        _mm = _mm[:_mm.index("\n    def ", 10)]
        check("_start_engine_export(want)" in _mm,
              "那个弹窗点确定后没有真的开导（还只指路 ✗）")
        _btn = _src[_src.index("def _on_export_engine"):]
        _btn = _btn[:_btn.index("\n    def ", 10)]
        check("QMessageBox.question(" in _btn and "_start_engine_export(imgsz)" in _btn,
              "按钮那条路的确认被弄丢了（会变成「点了就直接导」✗）")
    finally:
        QMessageBox.question = _old_q
        _lp.EngineExportThread, _lp.EngineExportDialog = _old_t, _old_d
        lp._bgr_to_pixmap = orig
        p.close()
        shutil.rmtree(_dir, ignore_errors=True)


def t_draw_args_defined_when_stream_only():
    """⭐⭐⭐ **只收流（没开推理）时也不许崩**（用户 2026-10-05 ✓ 现场：以别的权重点「开始收流」⇒
    `失败: UnboundLocalError: local variable '_ploc_fo' referenced before assignment`）。

    病根：从决策环里"单独拎出来"的**画框段**（`_draw_show_frame` ✓）是在
    `if self._infer.is_set():` **块外**被调用的 ✓（收流阶段本来就会走到 ✓），而它用到的几个名字
    **只在那个块里赋值** ✗ ⇒ 传参那一步就炸 ✓。
    ⚠ 函数体里那句 `if … and self._infer.is_set():` **拦不住** ✗：参数是**调用之前**求值的 ✓。
    ⚠ **只补一个不管用**：用 AST 把调用点的 31 个名字对了一遍 ⇒ 只在块内赋值的**是三个**：
      `_ploc_fo` / `action` / `player_box` ✓（只收流时本来没有推理结果 ⇒ `{}` / `None` 就是正确答案 ✓）。

    这条用例把规矩**钉在源码上**：调用点用到的每个名字，要么在 `if self._infer.is_set():` **之前**
    就有赋值 ✓，要么……没有"要么" ✗ —— 以后往画框段里加代码、新引用了块内变量（那段自己的注释
    也警告过 ✓）⇒ 这里**当场红** ✓，而不是等用户点「开始收流」才炸 ✓。
    """
    import ast

    src = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    calls = [c for c in ast.walk(tree) if isinstance(c, ast.Call)
             and getattr(c.func, "attr", None) == "_draw_show_frame"]
    check(calls, "找不到 `_draw_show_frame` 的调用点（改名/挪走了 ⇒ 这条用例要跟着改 ✓）")
    names = {a.id for c in calls for a in c.args if isinstance(a, ast.Name)}
    check(len(names) > 20, "调用点参数表看着不对（才 %d 个名字 ✗）" % len(names))

    guard = next((n for n in ast.walk(tree)
                  if isinstance(n, ast.If) and isinstance(n.test, ast.Call)
                  and getattr(n.test.func, "attr", None) == "is_set"
                  and getattr(getattr(n.test.func, "value", None), "attr", None) == "_infer"),
                 None)
    check(guard is not None, "找不到 `if self._infer.is_set():`（结构变了 ⇒ 用例要跟着改 ✓）")

    inside = {t.id for t in ast.walk(guard)
              if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)}
    before = {t.id for t in ast.walk(tree)
              if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)
              and t.lineno < guard.lineno}
    risky = sorted(n for n in names if n in inside and n not in before)
    check(not risky,
          "这几个名字只在「开推理」块里赋值，却被**块外**的画框调用用到 ⇒ "
          "只收流时会 `UnboundLocalError`（用户 2026-10-05 报的就是它 ✓）：%r" % (risky,))
    for n in ("_ploc_fo", "action", "player_box"):
        check(n in before,
              "`%s` 没在「开推理」块**之前**给默认值 ✗（只收流时要用它 ✓）" % n)


def t_mob_cd_gray_box():
    """⭐⭐ 「目标被攻击CD」的**灰框 + 右下角剩余 CD**（用户 2026-10-06 ✓ 原话："处于目标被
    攻击CD的框，变成灰色，并且右下角显示一位小数剩余CD"）。

    钉四件（判据抽成**模块级纯函数** `mob_cd_style` ✓ ⇒ 能直接调 ✓；画框那个三十几个参数的
    大函数驱动不起来 ✗ 同 `queried_mob_draw_list` 的手法 ✓）：
      ① 不在冷却（空表 / 表里没它 / 剩余 ≤ 0 / 坏值 / 键不是 str ✓）⇒ `(None, "")`
         ⇒ 调用方照**原来的绿框**画 = **这功能没开时一字不差** ✓；
      ② 在冷却 ⇒ **灰**（`MOB_CD_GRAY` ✓）+ **一位小数** ✓；
      ③ 键按 `str(怪号)` 查（agent 那边的账也是 `str` ✓ —— 两处口径必须一致 ✗
         拿 int 去查一个 str 键的表，会**永远查不到**、灰框一次都不出现 ✓）；
      ④ **源码钉**：绘制段真的用它 ✓、框色走它给的返回值 ✓、那行字画在**右下角** ✓、
         快照是**决策线程**上取好塞进 `action["mob_cd"]` 的（画框可能跑在 `_DrawWorker`
         那条线程上 ✗ ⇒ 绝不许直接摸 agent 的账 ✓）。
    """
    ROOT = Path(__file__).resolve().parent.parent
    from gui import live_thread as lt

    # ① 不在冷却 ⇒ 老行为（调用方照绿框画 ✓）
    _off = (("空表", {}), ("表里没这只", {"1": 2.0}), ("剩余 0", {"1": 0.0}),
            ("剩余负", {"1": -0.5}), ("坏值", {"1": "坏值"}), ("None", {"1": None}))
    for _name, _cds in _off:
        if _name == "表里没这只":
            check(lt.mob_cd_style(_cds, 9) == (None, ""),
                  "表里没有这只怪却给了颜色（那就不是它进冷却 ✗）：%r" % (_name,))
            continue
        check(lt.mob_cd_style(_cds, 1) == (None, ""),
              "%s ⇒ 该当「没在冷却」（否则那框会莫名变灰 ✗）：%r"
              % (_name, lt.mob_cd_style(_cds, 1)))
    check(lt.mob_cd_style(None, 1) == (None, ""), "快照是 None 时该当「没有」✗")

    # ② 在冷却 ⇒ 灰 + 一位小数
    check(lt.mob_cd_style({"1": 1.234}, 1) == (lt.MOB_CD_GRAY, "1.2"),
          "在冷却该给「灰框 + 一位小数」：%r（用户原话是「变成灰色 + 右下角一位小数」✗）"
          % (lt.mob_cd_style({"1": 1.234}, 1),))
    check(lt.mob_cd_style({"1": 2.0}, 1) == (lt.MOB_CD_GRAY, "2.0"),
          "整数秒也该写成 `2.0`（一位小数 ✓）：%r" % (lt.mob_cd_style({"1": 2.0}, 1),))

    # ③ 键口径：`str(怪号)`（两处必须一致 ✗）
    check(lt.mob_cd_style({"1": 2.0}, "1")[0] == lt.MOB_CD_GRAY,
          "拿 `str(怪号)` 查不到（agent 的账就是按 str 记的 ✗）")
    check(lt.mob_cd_style({1: 2.0}, 1)[0] is None,
          "键是 int 的表被查到了 —— 那说明查表没按 `str(怪号)`（一处口径 ✗）")

    # ④ 源码钉（绘制段 + 决策线程取快照）
    src = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    check('mob_cd_style(_mob_cds, getattr(m, "id", None))' in src,
          "怪框那段没走 `mob_cd_style`（那灰框判据就是第二份实现 ✗）")
    check("_m_color if _cd_col is None else _cd_col" in src,
          "框色没走它给的返回值（灰不灰就白算了 ✗）")
    check("gx2 - _tw - 2" in src and "gy2 - 3" in src,
          "那行剩余 CD 没画在**右下角**（用户点名的位置 ✗）")
    check('action["mob_cd"] = agent.mob_cd_snapshot()' in src,
          "快照没在**决策线程**上取好塞进 `action`（画框在 `_DrawWorker` 里跑 ⇒ "
          "直接摸 agent 的账 = 跨线程读 ✗）")
    check("mob_cd_snapshot" in src,
          "`mob_cd_snapshot` 在实时回路里一个字都没出现（那灰框永远画不出来 ✗）")


def t_drop_touch():
    """⭐⭐ 「自动拾取」的判据：**玩家框和掉落物框相交**（用户 2026-10-06 ✓ 原话："当玩家与
    掉落物接触时连续点按拾取"—— "接触"那两个字就落在这一条上 ✓）。

    为什么单拎一条：那个大函数（收流回路）驱动不起来 ✗ ⇒ 判据抽成模块级纯函数 `drop_touch` ✓
    （同 `mob_cd_style` / `out_cd_dash` 的手法 ✓）—— 决策那边**只读**它算出来的布尔 ✓。

    钉四件：
      ① **相交 ⇒ True**（含"贴边"也算接触 ✓）；
      ② **分开 ⇒ False**（两个轴都要分开才算没碰上 —— 只在一个轴上重叠不算 ✗）；
      ③ 缺料 / 坏值（`player_box=None` / 空表 / 坏条目）⇒ **False 且不抛** ✓
         （20ms 回路里抛出去会带崩 ✓）；
      ④ **源码钉**：判据在**感知层**算（`drop_touch(player_box or …, _drop_boxes)` ✓ 每帧都写 ✓），
         **决策层只读 `player.touching_drop`** ✓（⛔ 别让 agent 自己摸检测框 —— 那是跨层 ✗）；
         掉落框**只从掉落类**（`CLASS_DROP`）来 ✓（宠物/NPC/其他玩家不参与拾取 ✓）。
    """
    ROOT = Path(__file__).resolve().parent.parent
    from gui import live_thread as lt

    _P = (500.0, 500.0, 530.0, 0.9, 40.0, 60.0)   # (cx, cy, bottom, conf, w, h)

    # ① 相交 / 贴边 ⇒ True
    check(lt.drop_touch(_P, [(490.0, 500.0, 510.0, 520.0)]) is True,
          "玩家框和掉落框**明明交着**却判没接触（那永远捡不到 ✗）")
    check(lt.drop_touch(_P, [(480.0, 470.0, 520.0, 530.0)]) is True,
          "掉落框把玩家框**整个包住**了还判没接触 ✗")
    check(lt.drop_touch(_P, [(520.0, 500.0, 540.0, 520.0)]) is True,
          "**贴着边**不算接触（「接触」就是碰着 ✓ 差一像素就不算太严了 ✗）")

    # ② 分开 ⇒ False（两个轴都得分得开）
    check(lt.drop_touch(_P, [(600.0, 500.0, 620.0, 520.0)]) is False,
          "离着老远也判「接触」（那会一路狂点拾取键 ✗）")
    check(lt.drop_touch(_P, [(490.0, 600.0, 510.0, 620.0)]) is False,
          "**只在水平方向**重叠（竖直差得远）也判接触（掉落物在楼下 ⇒ 不该捡 ✗）")
    check(lt.drop_touch(_P, [(600.0, 600.0, 620.0, 620.0)]) is False,
          "两个方向都不沾边还判接触 ✗")

    # ③ 缺料 / 坏值 ⇒ False 且不抛
    check(lt.drop_touch(None, [(490.0, 500.0, 510.0, 520.0)]) is False,
          "这一帧**没定位到玩家**（player_box=None）却判接触 ✗")
    check(lt.drop_touch(_P, []) is False and lt.drop_touch(_P, None) is False,
          "没有掉落框却判接触 ✗")
    check(lt.drop_touch((1, 2), [(490.0, 500.0, 510.0, 520.0)]) is False,
          "`player_box` 元组长度不对时该当「没接触」（别抛 ✗）")
    check(lt.drop_touch(_P, [("坏",), None, (490.0, 500.0, 510.0, 520.0)]) is True,
          "坏条目该**跳过**、后面的好框照旧算（一条坏数据不该把整帧废掉 ✗）")

    # ④ 源码钉
    # ⚠ 2026-10-20：掉落框那几行搬进 `runtime/frame_infer.py` ⇒ 两文件一起看 ✓
    src = ((ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
           + (ROOT / "runtime" / "frame_infer.py").read_text(encoding="utf-8"))
    # ⚠ 2026-10-13 阶段 1 编排收口：这段从 `_run` 主循环抽进了 `_decide_frame_on_infer()`，
    #   参数名 `_last_player` → `last_player` ⇒ 锚点跟着变 ✓。语义不变：仍在感知层算「接没接触」。
    check("drop_touch(player_box or last_player[0]," in src,
          "感知层没把「接没接触」算出来 / 没按玩家框那张算（决策那边就没得读 ✗）")
    check(src.count("ws.player.touching_drop =") == 1,
          "`touching_drop` 该**只有一处**写口（每帧覆盖写 ✓ 别留上一帧的 True ✗）：%d 处"
          % src.count("ws.player.touching_drop ="))
    check("_extra_map.get(int(_cls)) == CLASS_DROP" in src,
          "掉落框不是**只从掉落类**挑的（宠物/NPC 混进来 ⇒ 站在宠物旁边也在点拾取 ✗）")
    _agt = (ROOT / "decision" / "agent.py").read_text(encoding="utf-8")
    check('getattr(getattr(ws, "player", None), "touching_drop", False)' in _agt,
          "决策那边没读 `touching_drop`（那这个开关点不动任何键 ✗）")
    check("CLASS_DROP" not in _agt,
          "决策层直接引用了掉落类 ⇒ 它开始自己摸检测框了（跨层 ✗ 该只读 WorldState ✓）")


def t_out_cd_dashed_attack_box():
    """⭐⭐ 「**输出行为CD 没走完 ⇒ 攻击范围框画虚线**」（用户 2026-10-06 ✓ 原话："每次执行输出
    行为后，输出行为CD未结束时，将攻击范围框变成虚线"）。

    钉四件（画框那个三十几个参数的大函数驱动不起来 ✗ ⇒ 判据抽成模块级纯函数 `out_cd_dash` ✓、
    虚线本身也抽成 `_draw_dashed_segment` / `_draw_poly_dashed` ✓ ⇒ 都能直接调 ✓）：

      ① `_draw_dashed_segment`：**真的是虚线**（有实段、也有空隙 ✓ 按像素验 ✓ —— 只看"画了
         像素"的话，画成实线也能蒙过去 ✗）；
      ② `_draw_poly_dashed`：四条边都画到、**框内一个像素不动**（别顺手涂个实心框 ✗）；
      ③ `out_cd_dash(action)` 的口径：`> 0` ⇒ True；`≤ 0` / 缺键 / 坏值 / `None`（只收流 ✓）
         / `{}` ⇒ False（= **照实线** = 老观感一字不变 ✓）；
      ④ **源码钉**：快照在**决策线程**上取好（`action["out_cd"] = agent.out_cd_left_s()` ✓
         **只此一处** ✓ 同 `mob_cd` 那条的理由 ✓）、绘制段用它给的答案 ✓、
         `dashed=` **只喂给攻击范围框**那一个（盲区框 / 追击起跳框 / 视野线一动不动 ✓）。
    """
    ROOT = Path(__file__).resolve().parent.parent
    from gui import live_thread as lt

    from numpy import zeros as _zeros

    # ---- ① 虚线：有实段、也有空隙 ----
    img = _zeros((40, 60, 3), "uint8")
    lt._draw_dashed_segment(img, (0, 20), (59, 20), (255, 0, 0),
                            dash_len=10, gap=6, thickness=1)
    check(int(img[20, 5].max()) > 0,
          "虚线的**实段**一个像素都没画（那就是压根没画 ✗）")
    check(int(img[20, 12].max()) == 0 and int(img[20, 14].max()) == 0,
          "整条线被涂满了（那是**实线**，不是虚线 ✗）：dash=10 gap=6 ⇒ x∈[10,16) 该是空的")
    check(int(img[19].max()) == 0 and int(img[21].max()) == 0,
          "线宽 1 却画到了别的行（虚线画歪了 ✗）")
    _img3 = _zeros((40, 40, 3), "uint8")
    lt._draw_dashed_segment(_img3, (10, 10), (10, 35), (0, 0, 255),
                            dash_len=8, gap=4, thickness=2)
    check(int(_img3[12, 10].max()) > 0 and int(_img3[12, 11].max()) > 0,
          "竖着的虚线没按线宽画（thickness=2 该有两列 ✗）")

    # ---- ② 多边形（攻击范围框就可能是梯形 ✓）：四条边都虚、里面不动 ----
    img2 = _zeros((60, 80, 3), "uint8")
    lt._draw_poly_dashed(img2, [(10, 10), (70, 10), (70, 50), (10, 50)],
                         (0, 255, 0), thickness=1)
    check(int(img2[30, 10].max()) > 0 and int(img2[50, 65].max()) > 0
          and int(img2[30, 70].max()) > 0 and int(img2[10, 13].max()) > 0,
          "四条边没都画到（漏边 ⇒ 那个框缺一块 ✗）")
    check(int(img2[30, 40].max()) == 0,
          "框**里面**被涂了（攻击范围框该只有四条虚线边 ✗）")
    check(int(img2[10, 22].max()) == 0,
          "上边是**连续涂满**的（不是虚线 ✗）：dash=10 gap=6 ⇒ x∈[20,26) 该是空的")
    check(lt._draw_poly_dashed(_zeros((10, 10, 3), "uint8"), None, (0, 0, 0)) is None,
          "`pts` 是 None 时该安安静静返回（20ms 回路里抛出去会带崩 ✓）")

    # ---- ③ 判据口径 ----
    check(lt.out_cd_dash({"out_cd": 0.2}) is True,
          "CD 里（还剩 0.2s）该画虚线 ✗")
    for _name, _act in (("出了 CD", {"out_cd": 0.0}), ("负值", {"out_cd": -3.0}),
                        ("缺键", {}), ("坏值", {"out_cd": "坏"}), ("None 值", {"out_cd": None}),
                        ("只收流", None), ("空 action", {})):
        check(lt.out_cd_dash(_act) is False,
              "%s ⇒ 该当「照实线画」（老观感一字不变 ✗）：%r"
              % (_name, lt.out_cd_dash(_act)))

    # ---- ④ 源码钉 ----
    src = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    check(src.count('action["out_cd"] = agent.out_cd_left_s()') == 1,
          "那份「输出行为CD」快照没在**决策线程**上取好 / 取了两处（画框在 `_DrawWorker` "
          "那条线程上 ⇒ 直接摸 agent 的账 = 跨线程读 ✗）：%d 处"
          % src.count('action["out_cd"] = agent.out_cd_left_s()'))
    check("_out_cd_dash = out_cd_dash(action)" in src,
          "绘制段没走 `out_cd_dash`（那「什么时候虚线」就是第二份口径 ✗）")
    check(src.count("dashed=_") == 1 and "dashed=_out_cd_dash" in src,
          "`dashed=` 不止喂给「攻击范围框」一处（用户点名的是**它** ✗）：%d 处"
          % src.count("dashed=_"))
    # ⚠ 2026-10-12：签名多了 `up` / `down`（「追击起跳框」要用它把上边抬到跳跃攻击的高度 ✓
    #   见 `t_chase_jump_box_height` / `agent.chase_jump_ceiling` ✓）⇒ 这条钉改成"**有**
    #   `dashed` 这个口子"（**不钉整串签名** ✗ —— 那样每加一个口径就得改一次钉子 ✓）。
    check("def _draw_box(ad, color, on, from_d=0.0, width=2, dashed=False," in src
          and "up=None, down=None):" in src,
          "`_draw_box` 没有 `dashed` / `up` 这些口子（那虚线、「抬上边」就得各写一条画框的"
          "路 ⇒ 形状迟早分叉 ✗）")
    check("cv2.rectangle(t, p0, p1, c, width)" in src and "_draw_poly_dashed(" in src,
          "实线那条路被弄丢了 / 虚线没接上（两者必须都留着 ✓）")


def t_class_box_switches():
    """⭐⭐⭐ **「检测框颜色」每一类前面的勾选框**（用户 2026-10-06 ✓ 两条原话：先问
    "**为何没看到掉落物、宠物的检出框？**"，再定："在设置→界面页签→检测框颜色 **每一项
    前面加勾选框，默认全部勾选，勾选后实时显示**"）。

    为什么要这么一圈：那几类（掉落 / NPC / 其他玩家 / 宠物）**本来是画不出来的** ✗ ——
    它们在 `split_dets`（"谁进决策"的唯一出口 ✓）就被丢了，而 `WorldState` 里只有
    `mobs` / `player` 两个字段 ⇒ 绘制层**无米下锅** ✓。这一轮补的是**显示那一路**：
    同一次推理的**同一批框**分两条走 —— 决策那条照旧只认 mob/player ✓，画框这条按类别画 ✓。

    钉六件：
      ① **theme**：六个类别的开关键**从类别表生成**（`CLASS_ON_KEYS` ✓ 加类别自动多一条 ✓）、
         默认**全 True**（= 画 ✓ ⇒ 老配置 / 老观感一字不变 ✓）、**存得进读得回**（存 False
         能读回 False ✓）、**坏值当没存过**（⇒ True ✓ 同 `load_vis` 其它开关的口径 ✓）；
      ② **纯函数** `class_box_draw_list`：勾着 ⇒ 画（颜色 + `"掉落 0.88"` 这种文字 ✓）；
         取消勾 ⇒ **一条都不画** ✓；开关**缺键 / 坏值 ⇒ 照画** ✓；**没有颜色的类别 ⇒ 不画**
         （宁可看不见，也不糊一个白框让人以为认出来了 ✗）；
      ③ **分流** `extra_class_dets`：只挑**其余类别**（怪 / 玩家一个都不进 ✓ 它们各有自己的
         绘制路径 ✓）；`{模型类号: 我们的类号}` 方向不能反 ✓；坏值 / 未知类不炸 ✓；
      ④ **决策不受影响**（源码钉）：`ws.mobs` 仍旧只由 `mob_dets` 喂 ✓ —— 画框那份
         （`_cls_dets`）只许进 `_draw_show_frame` ✓；
      ⑤ **怪 / 玩家各自也吃自己那个勾**（源码钉：`mob_on` / `player_on` ✓ 用户要的是
         "每一项"✓ 不是只给新的那几类 ✓）；
      ⑥ **设置弹窗真建一遍**：六个勾**都在、默认全勾**；取消一个 ⇒ **当场写盘**（不用按
         确定 ✓ "勾选后实时显示"✓），而且**只写那一个键**（不许顺手把别的冲掉 ✗）。
    """
    import shutil
    import tempfile
    import unittest.mock as mock

    from gui import theme
    from perception.detect import classes as pc

    tmp = Path(tempfile.mkdtemp(prefix="clsvis_"))
    try:
        # ⚠ 全程把 `config/ui.yaml` **指到临时文件**（§11：用例绝不许碰真配置 ✓）
        with mock.patch.object(theme, "CFG", tmp / "ui.yaml"):
            # ---- ① theme：生成 / 默认全勾 / 存得进读得回 / 坏值 ----
            check(len(theme.CLASS_ON_KEYS) == len(pc.CLASSES),
                  "开关键不是**按类别表**生成的（加类别会漏一条 ✗）：%r"
                  % (theme.CLASS_ON_KEYS,))
            check(all(("%s_on" % en) in theme.CLASS_ON_KEYS
                      for _c, en, _z, _b in pc.CLASSES),
                  "类别表里有类别没有自己的开关键（那一行就没有勾选框 ✗）：%r"
                  % (theme.CLASS_ON_KEYS,))
            vis = theme.load_vis()
            for k in theme.CLASS_ON_KEYS:
                check(vis.get(k) is True,
                      "`%s` 默认不是 True（用户点名「默认全部勾选」✗；而且老配置没有这些键 ⇒ "
                      "默认必须是「画」才对 ✓）：%r" % (k, vis.get(k)))
            theme.save_vis({"pet_on": False})
            check(theme.load_vis()["pet_on"] is False,
                  "取消勾**存了读不回来**（`load_vis` 漏了这批开关 ⇒ 勾了没用 ✗）")
            theme.save_vis({"npc_on": "坏值"})
            check(theme.load_vis()["npc_on"] is True,
                  "坏值该当「没存过」（= 画 ✓ 同其它开关的口径 ✗）：%r"
                  % (theme.load_vis()["npc_on"],))
            theme.save_vis({"pet_on": True, "npc_on": True})     # 还原成默认那份 ✓

            from gui import live_thread as lt

            # ---- ② 纯函数：勾着画 / 取消勾不画 / 坏值照画 / 没颜色不画 ----
            _colors = {pc.CLASS_DROP: (0, 255, 255), pc.CLASS_PET: (180, 180, 180)}
            _dets = [(pc.CLASS_DROP, 10.4, 20.6, 30.2, 40.9, 0.876),
                     (pc.CLASS_PET, 50.0, 60.0, 70.0, 80.0, 0.5)]
            _on = {"drop_on": True, "pet_on": True}
            got = lt.class_box_draw_list(_dets, _on, _colors)
            check(len(got) == 2, "勾着的两类都该画出来（现在画了 %d 个 ✗）：%r"
                  % (len(got), got))
            check(got[0] == (10, 21, 30, 41, (0, 255, 255), "掉落 0.88"),
                  "掉落那个框该是「取整坐标 + 那一类的颜色 + 类别名 两位小数」（同怪框款式 ✓）：%r"
                  % (got[0],))
            check(got[1][5].startswith("宠物"),
                  "文字该用**中文类别名**（别写错到别的类 ✗）：%r" % (got[1][5],))
            _off = lt.class_box_draw_list(_dets, {"drop_on": False, "pet_on": True},
                                          _colors)
            check([g[5] for g in _off] == ["宠物 0.50"],
                  "取消「掉落」那个勾之后它还画着（用户要的是**勾选后**才显示 ✗）：%r"
                  % ([g[5] for g in _off],))
            check(len(lt.class_box_draw_list(_dets, {}, _colors)) == 2,
                  "缺键（老配置 / 还没存过）该当「画」（同 `load_vis` 口径 ✗）")
            check(len(lt.class_box_draw_list(_dets, {"drop_on": "坏值", "pet_on": None},
                                             _colors)) == 2,
                  "开关是坏值时该当「画」（别把坏值读成「关」⇒ 一上手画面上一个框都没有 ✗）")
            check(lt.class_box_draw_list(_dets, _on, {}) == [],
                  "查不到颜色的类别也画了 —— 那只能糊个白框，会让人误以为认出来了 ✗")
            check(lt.class_box_draw_list(None, _on, _colors) == []
                  and lt.class_box_draw_list([("坏",)], _on, _colors) == [],
                  "`dets` 是 None / 坏条目时该稳稳当当跳过（20ms 回路里抛出去会带崩 ✓）")

            # ---- ③ 分流：只挑"其余类别"，方向不能反 ----
            _xyxy = [(1, 1, 2, 2), (3, 3, 4, 4), (5, 5, 6, 6), (7, 7, 8, 8)]
            _cfs = [0.9, 0.8, 0.7, 0.6]
            #                                 模型类号：0=玩家 7=宠物 5=怪 99=不认识
            _clss = [pc.CLASS_PLAYER, 7, pc.CLASS_MOB, 99]
            _map = {7: pc.CLASS_PET}
            got3 = lt.extra_class_dets(_xyxy, _cfs, _clss, _map)
            check(got3 == [(pc.CLASS_PET, 3.0, 3.0, 4.0, 4.0, 0.8)],
                  "「其余类别」的分流不对（该只挑宠物那一条、且**用我们的类号** ✓）：%r"
                  % (got3,))
            check(lt.extra_class_dets(_xyxy, _cfs, _clss, {}) == [],
                  "映射空表还硬塞（那是「按类别表 id 硬认」⇒ 张冠李戴 ✗）：%r"
                  % (lt.extra_class_dets(_xyxy, _cfs, _clss, {}),))
            check(lt.extra_class_dets([(1, 2, 3, 4)], ["坏", ], ["坏", ], {1: 2}) == [],
                  "坏值该跳过（不许抛 ✓）")

            # ---- ④⑤ 源码钉：决策那条链不动、每一类各吃自己的勾 ----
            src = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
            # ⚠ 2026-10-11 起多一个**低分档** `mob_low`（只给怪续命 ✓ 见 `split_dets` ✓）——
            #   钉的写法跟着改 ✓ **但这条的要害没变**：怪那条链**只吃 `split_dets` 那一路** ✓
            #   画框那份（`_cls_dets` ✓ 只给显示 ✓）**一个都不许进来** ✗。
            check("ws.mobs = tuple(mob_tracker.update(mob_dets, mob_low" in src,
                  "怪那条链不吃 `mob_dets`（+ 低分档 `mob_low`）了 —— 画框那份"
                  "（只给显示 ✓）可能混进了决策 ✗")
            # ⭐⭐ 2026-10-13：语义没变，但调用**多了 `player_world=`**（用户的「角色相对系」✓
            #   见 `MobTracker._shift_anchor` ✓）—— 钉的还是"**只吃 `mob_dets`/`mob_low`**"✓。
            check("player_world=_pw" in src,
                  "没把**角色的世界坐标**递给 tracker ✗（用户 2026-10-13："
                  "「根据角色的世界坐标去算相对位移、相对速度」✓ 少了它 ⇒ 相机一动同一只怪换 id ✗）")
            check("mob_tracker.update(_cls_dets)" not in src
                  and "PlayerTracker(_cls_dets" not in src,
                  "**画框那份**被喂给 tracker 了（= 掉落/宠物进决策 ✗ 用户 2026-10-04 明确否掉）")
            check("extra_class_dets(xyxy, cfs, clss, _extra_map)"
                  in (ROOT / "runtime" / "frame_infer.py").read_text(encoding="utf-8"),
                  "没有那条「只给画框」的分流（那几类还是整批丢掉 ⇒ 画面上永远看不到 ✗）")
            check('_mobs_iter = ws.mobs if _vis_cfg.get("mob_on", True) else ()' in src,
                  "怪物框没吃它自己那个勾（用户要的是**每一项**都有勾 ✗）")
            check('and _vis_cfg.get("player_on", True)' in src,
                  "玩家框没吃它自己那个勾 ✗")
            check("class_box_draw_list(" in src and "_cls_dets" in src,
                  "画框段没走那个纯函数（判据成了第二份 ✗）")
            check("class_box_draw_list(\n" in src
                  and "_cls_dets, _vis_cfg, _cls_colors)" in src,
                  "画框段没把**这一帧的框 + 开关配置 + 类别色**一起交给那个纯函数"
                  "（少给一个 ⇒ 那些框还是画不出来 / 或自己去算一套 ✗）")
            # ⚠ 数**三处**：函数签名一处 + **两个调用点**各一处（同步那条 + 开"画异步"
            #   那条 ✓）—— 只改一处 = 换个开关那几类框又没了（本仓库栽过好几次的地方 ✓）。
            check(src.count("_cls_colors, _cls_dets, _last_player") == 3,
                  "画框段那两个调用点没都带上那份框（只改一处 = 开「画异步」时又不画了 ✗）：%d 处"
                  % src.count("_cls_colors, _cls_dets, _last_player"))

            # ---- ⑥ 真建设置弹窗：六个勾都在、默认全勾、取消一个当场写盘 ----
            from PyQt5.QtWidgets import QApplication
            from gui.settings_dialog import SettingsDialog

            app = QApplication.instance() or QApplication([])      # noqa: F841
            dlg = SettingsDialog()
            try:
                for cid, en, zh, _bgr in pc.CLASSES:
                    k = "%s_on" % en
                    check(k in dlg._vis_on,
                          "「检测框颜色」里「%s框颜色」前面**没有勾选框**（%s ✗ 用户点名"
                          "「每一项前面加勾选框」）" % (zh, k))
                    check(dlg._vis_on[k].isChecked(),
                          "「%s框颜色」那个勾默认没勾上（用户：**默认全部勾选** ✗）" % zh)
                dlg._vis_on["drop_on"].setChecked(False)
                check(theme.load_vis()["drop_on"] is False,
                      "取消勾**没有当场写盘** ⇒ 实时预览那 1 秒重读拿到的还是旧值，"
                      "画面上看不出变化（用户要的是「勾选后实时显示」✗）")
                check(theme.load_vis()["pet_on"] is True,
                      "写盘时顺手把别的键也改了（该只写这一个 ✓）：%r"
                      % (theme.load_vis()["pet_on"],))
                dlg._vis_on["drop_on"].setChecked(True)
                check(theme.load_vis()["drop_on"] is True,
                      "再勾回来没生效（勾选框与配置已经不同步了 ✗）")
            finally:
                dlg.close()
                dlg.deleteLater()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_vt_lie_sdk():
    """⭐⭐ 「自动测谎（visual_tracking SDK）」的接线（2026-10-09 接入 ✓）。

    钉八件（全是"接错了就不工作 / 危及画面"的点 ✓）：
      ① 喂的是 **`raw`（画框之前的原生帧）** —— SDK 明说"输入图不能包含人为绘制的检测框"
         （`INTEGRATION.md:28` ✓）；喂 `vis` 就是把蓝框/绿框当目标 ✗；
      ② 投递口**只留最新一帧** + 推理在**独立线程**（`_vt_loop`）—— 探针实测单帧中位 ~40ms、
         **最慢 2232ms** ✗ ⇒ 放主回路就是"画面卡住两秒"；
      ③ 成功后点「确定」**走现成口径** `execution.mouse_aim.click_ratio`（换算只有一处 ✓），
         且**只对本机窗口**来源发（收流是远端坐标，点了没用 ✗）；
      ④ 点完要**重开一轮**（`_vt_cmd = "reset"`）—— 不重开整个会话就停在 ENDED ✗；
      ⑤ 界面要有开关 + 状态行（`vt_status` 接上了才看得见"它在干什么"✓），
         开关要随「开始」进线程（否则重开预览后状态对不上 ✗）。

    ⚠ **行为层**（真喂帧、真匹配、真成功）在 `tools/selftest_vt_sdk.py` 里量 ✓ —— 那边要
      GPU + 素材，跑一次 ~30s ⇒ 这两条分工：这里钉"接线"，那边钉"数" ✓。
    """
    lt = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    lp = (ROOT / "gui" / "live_panel.py").read_text(encoding="utf-8")
    # ⚠ 钉**第一参数必须是 `raw`** ✓，别钉后面那两个 ✗：原点现在会**跟着窗口**走
    #   （2026-10-10 ✓ 窗口挪了也要把画面坐标折回屏幕 ⇒ 传的是 `_ox/_oy` ✓
    #    见 `gui/live_thread` 里那句"原点跟着窗口"的说明 ✓）—— 钉死 `origin_x, origin_y`
    #   会变成**假红** ✗（真正要钉的是"喂原帧"这一条 ✓）。
    # ⚠ 2026-10-20：投帧那一小段已搬 `runtime/frame_vt.py`（阶段 1 第二刀 ✓）⇒ 两边都要认 ✓
    _vt_src = (ROOT / "runtime" / "frame_vt.py").read_text(encoding="utf-8")
    check("th._vt_beat(raw," in _vt_src or "self._vt_beat(raw," in lt,
          "喂的不是画框前的原生帧 `raw` ✗（SDK 要求输入不能带人为绘制的框）")
    check("frame_vt.step(self, raw" in lt,
          "`_run` 里没调 `frame_vt.step(...)`（那段搬完没接上 ⇒ 测谎一帧都不喂 ✗）")
    check("self._vt_frame = (frame," in lt,
          "没有「只留最新一帧」的投递口 ⇒ 会排队积压 ✗")
    check("threading.Thread(target=self._vt_loop" in lt,
          "SDK 没跑在独立线程里 ✗（实测单帧最慢 2232ms ⇒ 会卡住实时画面）")
    check("from perception.vt_sdk import VtLieSdk" in lt and "mouse_aim.click_ratio" in lt,
          "接入 SDK 或「点确定」没走现成口径 ✗（换算必须只有一处）")
    check('!= "window"' not in lt and "vt_confirm_click" in lt,
          "又按「来源」拦鼠标了 ✗ —— 鼠标是**经 relay 打到游戏机**的 ✓ 不该看 `source` ✓"
          "（2026-10-09 现场：收流那套被这条拦成「一帧都没发」✗）")
    check('self._vt_cmd = "reset"' in lt,
          "点完不重开一轮 ⇒ 下一局不会动 ✗")
    check("vt_status.connect(self._on_vt_status)" in lp,
          "实时线程的状态没接上 ⇒ 人看不到它在干什么 ✗")
    check("vt_status = pyqtSignal(str)" in lp and "self.vt_status.emit" in lp,
          "「实时」页没把状态**转发**出去 ⇒ 搬去挂机保护页的那行永远是一句死的 ✗")
    pp_src = (ROOT / "gui" / "player_panel.py").read_text(encoding="utf-8")
    check("set_vt_status" in pp_src and "vt_status.connect(self.set_vt_status)" in pp_src,
          "挂机保护页那行**没接**转发信号 ⇒ 状态停在那儿不动 ✗")
    check("lbl_vt" not in lp and "lbl_vt" in pp_src,
          "状态行没搬干净（实时页还留着一份 / 挂机保护页没有）✗"
          "（用户 2026-10-09：这个状态肯定要一起迁移过去 ✓）")
    pp = pp_src                       # 同一份源码（读两次没必要 ✓ 一处口径 ✓）
    check("_build_vt_group" in pp and "lay.addWidget(self._build_vt_group())" in pp,
          "「自动测谎（视觉追踪）」组没建 / 没挂进「挂机保护」页 ✗（用户 2026-10-09 点名的位置）")
    check("update_live(vt_on=" in pp and "vt_cool_s=" in pp,
          "组里的参数没写回 `live.yaml` ⇒ 实时线程读不到、改完不生效 ✗")
    check('self._vt_want = bool(_live.get("vt_on"' in lt and "self._vt_cfg[" in lt,
          "实时线程没**热读**那几格 ⇒ 「挂机保护」页改了不生效（要等重开 ✗）")
    check('"show_fps", "vt_on")' not in lp
          and '"vt_on": bool(getattr(self, "ck_vt"' not in lp,
          "`live_panel` 里还留着 `vt_on` 的**按项目**存取 ⇒ 控件已经搬走，它会把这格写成 "
          "False ⇒ **每存一次参数就把开关自己关掉** ✗（2026-10-09 的连带坑 ✓）")
    # ⭐ 2026-10-09 现场补的两条真事（用户原话：没看到鼠标指着真目标 / 没看到描边和框 ✓）
    ma_src = ((ROOT / "execution" / "mouse_aim.py").read_text(encoding="utf-8")
                  + (ROOT / "execution" / "aim_convert.py").read_text(encoding="utf-8"))
    check("def _vt_act" in lt and "aim_to(" in lt and "def _vt_aim_loop" in lt,
          "「鼠标指着目标」没接全：要有 `aim_to`（设目标 ✓）+ `_vt_aim_loop`（逐拍插值 ✓）")
    check("def aim_to" in ma_src and "def aim_tick" in ma_src
          and "def aim_ratio" not in ma_src,
          "「平滑瞄准」那套不对：该是 `aim_to` + `aim_tick`（旧的 `aim_ratio` 已废 ✗）")
    check("0.0 <= age <= 0.2" in lt,
          "「什么时候该动鼠标」的新鲜度闸丢了（该照抄 SDK 的 `MouseOutput`：0.2 秒 ✓）")
    check(') not in ("LOCKED", "COAST")' in lt,
          "只有 `LOCKED`/`COAST` 才许动鼠标 —— 丢了就会在成功界面上乱指 ✗")
    check("aim_tick(" in lt and 'get("aim_ms"' in lt,
          "逐拍插值的走位线程没了 ⇒ 又变回「一顿一顿跳」✗（用户 2026-10-09 报的 ✓）")
    check("ck_vt_aim_ms" not in lt and "sp_vt_aim_ms" in pp,
          "「鼠标跟随间隔(ms)」那格没接上 / 没落盘 ✗")
    check("def _vt_overlay" in lt and "frame_rect_of" in lt,
          "没把 ROI 描边 / 目标框画到显示帧上 ✗（用户原话：没看到剪裁窗口描边、没看到框 ✓）")
    check("_vt_overlay(vis, self._vt_last_obs)" in lt,
          "画框那一段没调 `_vt_overlay`（画了也没人画 ✗）")
    check("ck_vt_aim" in pp and "ck_vt_draw" in pp and "vt_aim=" in pp and "vt_draw=" in pp,
          "两个新开关没进「挂机保护」那一组 / 没落 `live.yaml` ✗")
    # ⭐ 2026-10-09 第三条（用户原话：测谎触发时"玩家、怪物检测还在跑"✗）
    # ⚠ 2026-10-20：`res = None`（测谎中不推理）搬进 `runtime/frame_infer.py` ⇒ 两文件一起看 ✓
    _inf_txt = (ROOT / "runtime" / "frame_infer.py").read_text(encoding="utf-8")
    check("_lie_mode" in lt and 'startswith("lie")' in lt and "res = None" in (lt + _inf_txt),
          "测谎期间没停掉「玩家/怪物」那条链（还在推理/还在画框 ✗）")
    check("_vt_aim_logged" in lt and "vt_aim_ok" in lt,
          "「鼠标发成功」没留痕 ⇒ 下次还是只能靠猜（用户现场问过「到底发了没」✓）")
    check("_vt_reset_recent" in lt and 'phase == "ENDED"' in lt,
          "`ENDED`（120 秒上限 / 出错）没自动重开一轮 ⇒ 跑两分钟就哑了 ✗")
    check("aim_available()" in lt and "_ok, _why = mouse_aim_c.aim_available()" in lt,
          "`aim_available()` 又当布尔用了 ✗（它返回 `(ok, why)` 元组 ⇒ 那道闸恒不拦 ✓）"
          "  ⚠ 2026-10-17 起换算那半搬到 `execution/aim_convert.py` ⇒ 名字前缀是 `mouse_aim_c` ✓")


def t_vt_group_writes_live():
    """⭐⭐ 「自动测谎」那组控件**真的会写配置**（2026-10-09 ✓ 用户要求：挂机保护页单独一组）。

    为什么还要一条行为层（`t_vt_lie_sdk` 只钉"那几个字在不在" ✗）：这一组的通路是
    `控件改动 → _on_vt_changed → update_live(...)`，任何一环接错（信号没连 / 键名写错 /
    "构造期开闸" 那一步漏了）现象都是「**改了没反应**」✗ —— 那正是这次要避免的。

    ⚠ 自检**绝不写用户配置**（本仓库硬规矩 ✓）：把 `player_panel.update_live` 换成**记账器** ✓
      （回调调的是那个模块名字空间里的 `update_live` ✓ ⇒ 换掉它即可 ✓）。
    ⚠ 也不建整个 `PlayerPanel`（那要项目 / 地形 / 一堆面板 ✗）：直接用**裸对象**接住那个方法 ✓
      —— 它只往 `self.xxx` 上挂控件 ✓（见 `_build_vt_group` ✓），不需要别的成员 ✓。
    """
    from PyQt5.QtWidgets import QApplication

    import gui.player_panel as pp

    app = QApplication.instance() or QApplication([])
    rec = []
    orig = pp.update_live
    pp.update_live = lambda **kw: rec.append(kw)          # 记账，不落盘 ✓
    try:
        holder = type("_VTHolder", (), {})()
        # ⚠ 回调必须**绑到这个裸对象**上（组里连的是 `self._on_vt_changed` ✓）—— 不绑就是
        #   `AttributeError: '_VTHolder' object has no attribute '_on_vt_changed'` ✗
        #   （这条用例第一次写出来就是这么红的 ✓）。
        holder._on_vt_changed = pp.PlayerPanel._on_vt_changed.__get__(holder)
        # ⚠ **必须留住返回值**（`grp`）：Qt 的父对象回收靠的是 Python 引用 —— 组一丢引用，
        #   它和它里面的复选框会**当场被删** ⇒ 后面 `setChecked` 就是
        #   `RuntimeError: wrapped C/C++ object … has been deleted` ✗（第一次写这条用例就踩了 ✓）。
        holder._vt_grp = pp.PlayerPanel._build_vt_group(holder)
        for n in ("ck_vt", "ck_vt_click", "sp_vt_conf", "cmb_vt_region", "sp_vt_cool"):
            check(getattr(holder, n, None) is not None, "「自动测谎」组少了控件 %s ✗" % n)
        check(rec == [], "**构造期**就写盘了（`_vt_built` 闸没生效 ⇒ 会拿半套默认值覆盖 ✓）：%r"
              % (rec,))
        # ① 拨一下开关 ⇒ 必须写一条 `vt_on=<新值>`，而且**五个键一起落** ✓
        #   ⚠ **取反**、不要写死 `setChecked(True)` ✗：初始值是从**用户的 live.yaml** 读的
        #     （他已经把它设成 True 了 ✓）⇒ 再 `setChecked(True)` 是**空操作**、不发信号 ✗
        #     ⇒ 用例会红在"没写 vt_on"上，看着像接线断了，其实是"这一下等于没动" ✓
        #     （2026-10-09 第一次写就是这么红的 ✓）。取反 ⇒ 与配置无关 ✓。
        want = not bool(holder.ck_vt.isChecked())
        holder.ck_vt.setChecked(want)
        check(rec and rec[-1].get("vt_on") is want,
              "拨了「启用」却没写 `vt_on`（信号没连 / 键名写错）：%r" % (rec,))
        for k in ("vt_click", "vt_aim", "vt_draw", "vt_conf", "vt_region", "vt_cool_s",
                  "vt_aim_ms"):
            check(k in rec[-1], "写回时漏了 %s（八个键要一起落 ✓）" % k)
        # ② 改冷却 ⇒ 值要真的进去
        rec.clear()
        holder.sp_vt_cool.setValue(7.5)
        check(rec and abs(float(rec[-1]["vt_cool_s"]) - 7.5) < 1e-6,
              "改「成功冷却」没写对：%r" % (rec,))
        # ③ 地区下拉的**值**必须是 SDK 认的那两个（`region='CN'/'TW'` ✓）
        check(holder.cmb_vt_region.itemData(0) == "CN"
              and holder.cmb_vt_region.itemData(1) == "TW",
              "「成功模板地区」下拉值不是 CN/TW ⇒ SDK 建会话时会 `ValueError` ✗")
        check(app is not None, "")
    finally:
        pp.update_live = orig


def t_vt_aim_is_smooth():
    """⭐ 「瞄准」必须**只发增量**（用户 2026-10-09 ✓ 原话："鼠标不是平滑的，而是在左上角和
    目标点之间高频来回跳动" ✗）。

    根因：瞄准备是**每 100~200ms 发一次**的连续动作，而绝对值定位每发都要先撞角归零 ⇒
    光标被甩回左上角再走向目标 ⇒ 看着就是来回抽 ✗（`click_ratio` 那种低频点击看不出 ✓）。

    用**替身 dinput**（不碰真硬件 ✓）钉三件：
      ① 第一发 ⇒ 先撞角归零建原点，再走一个**绝对增量**；
      ② 紧接第二发 ⇒ **不再撞角**，只发差值 ✓（就是这条治"来回抽"）；
      ③ 停顿超过 `AIM_FRESH_S`（人可能碰过鼠标 / 换了一轮）⇒ **重新归零一次** ✓。
    """
    import execution.mouse_aim as ma
       # 换算那半（2026-10-17 拆出）

    # ⚠ 2026-10-17：`aim_to` 现在会**用真光标同步基准**（本地 ✓ 见 `_sync_aim_cur` ✓）——
    #   而这一条钉的是**兜底那条**（读不到光标 / 网络 relay ✓）⇒ 在这里把它钉成
    #   「读不到」✓，强制走「第一发撞角建原点」✓ 原断言一个字不用改 ✓；
    #   「读得到那只手」由 `selftest_decision.t_aim_cur_synced_to_real_cursor` 专门钉 ✓。
    _old_own = ma._own_cursor
    ma._own_cursor = lambda: None

    moves = []

    class _Fake:
        @staticmethod
        def mouse_available():
            return True

        @staticmethod
        def mouse_move(dx, dy):
            moves.append((dx, dy))

    old_d, old_gain, old_ttl = ma.dinput, ma_c.load_gain, ma.AIM_TGT_TTL_S
    # ⚠ 本条只验"小步插值" ✓，不验标定口径 ✓ ⇒ 把"这个尺寸量过没有"那条守卫挡住 ✓
    #   （2026-10-10 ✓ 改成按**帧尺寸**找标定之后 ✓，200 宽的小画布在本机那份真标定里
    #    当然找不到 ✗ ⇒ 会判"没量过 ⇒ 不点" ⇒ **假红** ✗ —— 守卫自己由
    #    `selftest_decision.t_mouse_gain_frame_mismatch` 专门验 ✓）。
    _old_chk = ma_c.gain_mismatch_block
    ma_c.gain_mismatch_block = lambda _fs: (False, "")
    # ⚠ 把虚拟屏钉成**单显示器**（`0,0` ✓）：多屏那件事由
    #   `selftest_decision.t_corner_zero_multimonitor` 专门钉 ✓；这一条只验"小步插值" ✓
    #   ⇒ 钉成单显示器，`moves[1:]` 才全是插值小步 ✓。
    #   （⚠ 2026-10-10 记一笔：`corner_zero()` 现在**永远只发那一下** ✓ —— 之前有一版会
    #    "再补一发"把基准挪到主屏 ✗，那会让**副屏够不着** ✗，已经改掉了 ✓ 见它说明 ✓。）
    import core.wincap as _wc
    _old_vs = _wc.virtual_screen
    _wc.virtual_screen = lambda: (0, 0, 1920, 1080)
    ma.dinput = ma_c.dinput = _Fake
    ma_c.load_gain = lambda *a, **k: (1.0, 1.0)   # gain=1 ⇒ 画面像素 == 指令单位（好对数 ✓）
    try:
        ma.aim_reset()
        ok, why = ma.aim_to((100, 200, 3), 0.25, 0.50)   # frame_shape=(h,w,c) ⇒ 目标 (50, 50)
        check(ok, "设目标就失败了：%s" % why)
        check(moves == [], "`aim_to` 该**只设目标**、不动鼠标 ✗：%r" % (moves,))
        n = 0
        for _ in range(40):
            if ma.aim_tick(tick_ms=16.0, settle_ms=120.0):
                n += 1
        check(n >= 3, "只走了 %d 拍 ⇒ 又成「跳」了（要逐拍小步 ✓）" % n)
        check(moves[0] == (-4000, -4000), "第一拍没先建原点：%r" % (moves[:2],))
        _steps = moves[1:]
        check(max(max(abs(a), abs(b)) for a, b in _steps) <= 26,
              "单步太大（不是小步插值 ⇒ 光标会跳）：%r" % (_steps[:6],))
        _sum = (sum(a for a, _b in _steps), sum(b for _a, b in _steps))
        check(_sum == (50, 50), "多拍加起来没走到目标（插值走歪了）：%r" % (_sum,))
        moves.clear()
        ma.AIM_TGT_TTL_S = -1.0                # 假装"目标过期"（不动真时钟 ✓）
        check(ma.aim_tick(tick_ms=16.0, settle_ms=120.0) is False and moves == [],
              "目标过期了还在发指令（会对着过期点晃 ✗）：%r" % (moves,))
    finally:
        ma.dinput, ma_c.load_gain, ma.AIM_TGT_TTL_S = old_d, old_gain, old_ttl
        ma_c.gain_mismatch_block = _old_chk
        ma._own_cursor = _old_own
        _wc.virtual_screen = _old_vs
        ma.aim_reset()


def t_run_shared_names_defined_before_split():
    """⭐⭐⭐ `_run` 里**两个来源共用的那段**，只许读「按 source 分岔之前就赋过值」的名字。

    病（用户 2026-10-10 现场：点「开始推理」⇒
    `失败: UnboundLocalError: cannot access local variable 'cap_ref'`）：新加的口袋
    `cap_ref` 只在 `if source == "window":` 里 `= [""]` ✗ —— 而它被**两个来源共用**的
    `stats_ready.emit`（`cap_note` ✓）读 ⇒ 走**收流**时必炸 ✓。

    ⚠ **这是第三次同一类**（`interval` ✗ / 画框那几个名字 ✗ / 这次 `cap_ref` ✗ ——
    见 `budget = 0.0` 上面那段说明与 `t_draw_args_defined_when_stream_only` ✓）
    ⇒ 这次**按结构钉**：AST 找出「按 `source` 分岔的那个 if」✓，把它**内部才赋值**的名字
    列出来 ✓，再看**它之后**（= 共用尾段 ✓）有没有读 ✗ —— 有就是那颗雷 ✓。
    ⚠ 判据刻意做**窄**（只报"分岔里赋值 + 之后被读"✗）：宁可漏掉别的形态，也**不许假红** ✗
      （假红会让人开始无视这条 ❗）。
    """
    import ast
    import builtins

    def _stores(nodes):
        """这些语句里**被赋值**的名字（含 for 目标 / with as / except as / import ✓）。

        ⚠⚠ **不许进嵌套函数 / 类体** ✗（那是它们**自己的作用域** ✓）——
          第一版图省事用了 `ast.walk` ✗ ⇒ 把函数体里的名字也算成"已定义" ⇒ **连真雷都被
          当成安全的** ⇒ 这条守卫**空转** ✗（2026-10-10 自查时当场发现 ✗ 所以下面还有
          一段**自检**：拿一个合成的坏样例确认它**真的会报**✓）。
        """
        out = set()

        def rec(n):
            for x in ast.iter_child_nodes(n):
                if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    out.add(x.name)          # 只收"这个名字被定义了" ✓ 里面的局部名不算 ✓
                    continue
                if isinstance(x, ast.Lambda):
                    continue
                if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store):
                    out.add(x.id)
                elif isinstance(x, ast.ExceptHandler):
                    if x.name:
                        out.add(x.name)
                elif isinstance(x, ast.alias):
                    out.add((x.asname or x.name).split(".")[0])
                rec(x)

        for n in nodes:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.add(n.name)
                continue
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                out.add(n.id)
            elif isinstance(n, ast.alias):
                out.add((n.asname or n.name).split(".")[0])
            rec(n)
        return out

    def _loads(nodes):
        """这些语句里**被读**的名字（同样不进嵌套函数 ✓ 宁窄勿假红 ✓）。"""
        out = set()

        def rec(n):
            for x in ast.iter_child_nodes(n):
                if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load):
                    out.add(x.id)
                rec(x)

        for n in nodes:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                out.add(n.id)
            rec(n)
        return out

    def _scan(src):
        """⇒ **只在某一支里赋值、却在之后（共用尾段）被读**的名字 ✓。

        ⚠ 判据要**窄**（下面三条都是实测踩出来的假红 ✗ 2026-10-10 ✓）：
          · 两个分支**都**赋了值 ⇒ 安全 ✓（`_reader` ✓ 窗口与收流各定义一份 ✓）；
          · 尾段自己**也**赋了值 ⇒ 按"安全"处理 ✓（`_win` / `v` ✓ 那是尾段自己的局部量 ✓
            —— ⚠ 这属**放宽**：真雷 `cap_ref` 尾段从来没有赋值 ✓ 所以照样抓得住 ✓）；
          · 分岔之前 / 模块级 / 内置 ⇒ 安全 ✓。
        """
        tree = ast.parse(src)
        fn = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_run":
                fn = node
        if fn is None:
            return None
        split = None
        for i, st in enumerate(fn.body):
            if isinstance(st, ast.If) and "source" in ast.dump(st.test):
                split = i
                break
        if split is None:
            return None
        # 各支的语句（含 elif 链 ✓）
        bodies = [fn.body[split].body]
        el = fn.body[split].orelse
        while el:
            if len(el) == 1 and isinstance(el[0], ast.If):
                bodies.append(el[0].body)
                el = el[0].orelse
            else:
                bodies.append(el)
                el = []
        _any = set()
        for b in bodies:
            _any |= _stores(b)
        _every = set(_stores(bodies[0]))
        for b in bodies[1:]:
            _every &= _stores(b)
        safe = (_stores(fn.body[:split]) | _stores(tree.body) | set(dir(builtins))
                | _every | _stores(fn.body[split + 1:]))
        safe.add("self")
        return sorted((_any - safe) & _loads(fn.body[split + 1:]))

    # ---- ⭐ 先**自检守卫本身**（防它空转 / 防它假红 ✗ —— 两个都真踩过 ✓）----
    _demo = ("def _run(self):\n"
             "    source = 1\n"
             "    if source == 'window':\n"
             "        b = ['']          # 只在窗口那支赋值、尾段又要读 ⇒ 真雷 ✓\n"
             "        both = 1\n"
             "        later = 1\n"
             "    else:\n"
             "        both = 2          # 两支都赋 ⇒ 安全 ✓\n"
             "    later = 3             # 尾段自己也赋 ⇒ 安全 ✓\n"
             "    print(b, both, later)\n")
    check(_scan(_demo) == ["b"],
          "这条守卫自己不对 ✗ —— 期望**只报 b**（真雷 ✓），实际 %r；"
          "（「两支都赋」与「尾段自己赋」都该放行 ✓，唯独一支赋的必须报 ✓）"
          % (_scan(_demo),))

    _src = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    _bad = _scan(_src)
    check(_bad is not None,
          "`gui/live_thread.py` 里找不到 `_run` 或「按 source 分岔」那个 if（结构变了 ⇒ "
          "这条要跟着改 ✓）")
    check(not _bad,
          "这些名字**只在「按 source 分岔」里面**赋值，却被**两个来源共用**的代码读 ✗ ⇒ "
          "另一个来源一跑就 `UnboundLocalError`（用户现场：点「开始推理」当场失败 ✓）：%r"
          % (_bad,))


def t_pick_rect_auto():
    """⭐⭐⭐ 「对准窗口」自动框选（用户 2026-10-10 ✓ 原话："**框选窗口能做个自适应吗？就是我
    鼠标指着的时候它自动匹配窗口画面的尺寸 —— 不然我每次自己框都会有偏差**"✓）。

    钉五件：
      ① 界面有那个按钮 ✓ 且跟着"本地窗口"来源显隐 ✓（收流时它没意义 ✓）；
      ② **取的是客户区**（`wincap.client_rect` ✓ = 游戏真正绘制的画面 ✓ 不含标题栏/边框 ✓）
         —— ⚠ 与 `window_geom`（DWM 外框 ✓ 给 `rect_rel` 当基准 ✓）**不是一回事** ✗；
      ③ **连续两拍同一个窗口才认账** ✓（鼠标扫过去那一瞬不算 ✓）；
      ④ 认账后：`_rect` = 客户区 ✓、`_rect_win` = 那个窗口 ✓、`_rect_rel` 基准对得上 ✓、
         **按项目存了**（`_save_project_params` ✓）；
      ⑤ 超时 ⇒ **什么都不改** ✓ + 说清怎么办 ✓（不许静默 ✓ 也不许乱框 ✗）。
    """
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import QLabel, QPushButton

    import core.wincap as wc
    from gui.live_panel import LivePanel

    fake_client = (300, 200, 1600, 900)
    fake_frame = (292, 168, 1616, 942)          # DWM 外框（比客户区大一圈 ✓）
    win = {"hwnd": 424242, "title": "冒险岛怀旧服", "rect": fake_client, "pid": 999}

    def _panel():
        # ⚠⚠ **用普通替身对象**，再调**真实方法**（`LivePanel._pick_rect_auto(p)` ✓）——
        #   别用 `LivePanel.__new__(LivePanel)` 造"半成品 Qt 对象" ✗：那种壳子没有跑过
        #   `__init__`，销毁时踩 sip 的雷 ⇒ **静默原生崩**（2026-10-10 ✓ 本用例第一版就是这么
        #   把整个套件带崩的 ✗ 20 行就没了 ✓）。
        class _Stub:
            pass

        p = _Stub()
        # ⚠ 替身没有类的常量 ⇒ 手工挂上（`_pick_rect_auto` 要读它们 ✓）
        p.AUTO_PICK_SEC = LivePanel.AUTO_PICK_SEC
        p.AUTO_PICK_MS = LivePanel.AUTO_PICK_MS
        p._rect = None
        p._rect_rel = None
        p._rect_win = None
        p._saved = 0
        p.lbl_rect = QLabel()
        p.lbl_stats = QLabel()
        p.btn_pick_win = QPushButton()
        p._auto_timer = QTimer()
        p._dead = lambda: False
        p._refresh_rect_label = lambda: None            # 不依赖界面回填 ✓（那件事别处钉 ✓）
        p._save_project_params = lambda: setattr(p, "_saved", p._saved + 1)
        # ⚠ 把**真实方法**绑到这个替身上（替身只是"没有 Qt 的壳"✓ 逻辑一行都不改 ✓）
        p._pick_rect_auto = lambda: LivePanel._pick_rect_auto(p)
        p._auto_rect_tick = lambda: LivePanel._auto_rect_tick(p)
        p._bind_rect_window = lambda *a, **k: LivePanel._bind_rect_window(p, *a, **k)
        return p

    src = (ROOT / "gui" / "live_panel.py").read_text(encoding="utf-8")
    check("btn_pick_win = QPushButton" in src and
          "self.btn_pick_win.setVisible(is_win)" in src,
          "「对准窗口」按钮没建 / 没跟着「本地窗口」来源显隐 ✗")
    check("skip_pid=os.getpid()" in src,
          "自动框选没**排掉工作台自己** ✗ ⇒ 指着工作台就会把它框进去（比手框还糟 ✗）")

    _o_under = wc.window_under_cursor
    _o_geom = wc.window_geom
    _o_event = None
    try:
        import core.behavior as _bh
        _o_event = _bh.event
        _bh.event = lambda *a, **k: None                # 打点别写进真日志 ✓
    except Exception:                                   # noqa: BLE001
        pass
    try:
        wc.window_geom = lambda h: (fake_frame if int(h) == win["hwnd"] else None)
        wc.window_under_cursor = lambda **k: dict(win)

        # ---- ③ 第一拍：只记下、**不认账** ✓ ----
        p = _panel()
        p._pick_rect_auto()
        check(p._rect is None, "第一拍就认账了 ✗（鼠标扫过去那一瞬会被抓成区域 ✓）：%r" % (p._rect,))
        p._auto_rect_tick()
        check(p._rect is None, "第二拍（同一窗口 ✓）之前就写进去了 ✗")
        p._auto_rect_tick()                             # 第二次看到同一个 ⇒ 认账 ✓
        check(tuple(p._rect) == tuple(fake_client),
              "自动框选取到的不是**客户区** ✗（客户区 = 游戏画面本身 ✓ 手框差的就是这一圈 ✓）：%r"
              % (p._rect,))
        check(p._rect_win and p._rect_win[0] == win["hwnd"] and "冒险岛" in p._rect_win[1],
              "没记住「相对哪个窗口」✗（那样窗口挪了又要重框 ✓）：%r" % (p._rect_win,))
        check(p._rect_rel == wc.rel_from_rect(fake_client, fake_frame),
              "`rect_rel` 的基准不对 ✗ —— 它必须是**相对 `window_geom`（DWM 外框 ✓）**\n"
              "（用客户区当基准 ⇒ 运行期 `binding_rect` 折算出来的区域会偏 ✓）：%r"
              % (p._rect_rel,))
        check(p._saved == 1, "自动框完没**按项目存** ✗（重启/换项目就得再框一遍 ✓）：%d" % p._saved)

        # ---- ⑤ 超时：什么都不改 + 说清 ✓ ----
        p2 = _panel()
        wc.window_under_cursor = lambda **k: None
        p2._pick_rect_auto()
        for _ in range(LivePanel.AUTO_PICK_SEC * 1000 // LivePanel.AUTO_PICK_MS + 3):
            p2._auto_rect_tick()
        check(p2._rect is None and p2._saved == 0,
              "超时了还在动区域 ✗（宁可不动，也别框错 ✓）：%r" % (p2._rect,))
        check("超时" in p2.lbl_stats.text(),
              "超时了**不说一声** ✗（人只会以为点了没用 ✓）：%r" % (p2.lbl_stats.text(),))
    finally:
        wc.window_under_cursor = _o_under
        wc.window_geom = _o_geom
        if _o_event is not None:
            try:
                import core.behavior as _bh2
                _bh2.event = _o_event
            except Exception:                           # noqa: BLE001
                pass


def t_covered_note_wired():
    """⭐⭐ 「本地窗口来源**被别的窗口盖住** ⇒ 改抓整屏」必须**接上**（用户 2026-10-10 ✓）。

    病（用户原话："把本地实时窗口来源被遮挡的问题修掉" ✓）：`window_capture: wgc` 是
    **单窗口采集** ✗ ⇒ 盖在源窗口上的外接登录窗 / 断线提示框**不进画面** ✗（断线重连、
    测谎判不出界面 ✓）；WGC 还按**框中心那个点**重挑窗口 ⇒ 中心被占就抓成遮挡物 ✗。

    口径（用户选的 ✓）：**盖住的东西也要显示** ⇒ 被盖住时这一拍改走整屏合成 ✓，
    没被盖照旧 WGC ✓。判据与"换抓法"在 `core/wincap.py`（**行为级**用例见
    `selftest_winperf.t_covered_switches_to_screen` ✓）—— 这里只钉**接线**：
      ① 实时线程要**读** `wincap.last_note()` ✓（不读 ⇒ 日志与状态行永远是空的 ✓）；
      ② 只**在变化时**记一条 `capture_covered` ✓（每拍都记 = 拿日志刷屏 ✗）；
      ③ `cap_note` 要**带进 stats** ✓、面板要把它**缀进状态行** ✓（否则人只会看到
         "帧里突然多了别的窗口"，不知道是换了抓法 ✓）；
      ④ ⚠ **不许认死窗口** ✗：`live_thread` 里不许出现 `hwnd=` / `is_occluded` ✓
         （那是用户 2026-10-09 明确否掉过的方向 ✓ 见上面那条反向钉 ✓）。
    """
    lt = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    check("wincap.last_note()" in lt,
          "实时线程没读 `wincap.last_note()` ⇒ 「被谁盖住」这件事一个字都不会进日志/状态行 ✗")
    check("capture_covered" in lt,
          "没把「换抓法」记进 `behavior.log`（`capture_covered` ✓）⇒ 事后没法复盘 ✗")
    # ⭐⭐ 框选区域那条（用户 2026-10-10 "把两台显示器的全部内容展示了" ✓）：
    #   判定必须走 `wincap.binding_rect`（一处实现 ✓ 行为级用例在 winperf 里 ✓），
    #   而且"退回你框的那块"这件事要**记得进日志/状态行** ✗（不许静默 ✓）。
    check("wincap.binding_rect(" in lt,
          "实时线程没走 `wincap.binding_rect` ⇒ 「窗口问不到」时会去按虚拟屏幕折算 ⇒ "
          "抓成整块桌面（两台显示器全屏 ✗ 用户报过 ✓）")
    check("capture_rect" in lt,
          "「框选区域退回固定区域」没进 `behavior.log`（`capture_rect` ✓）⇒ 事后没法复盘 ✗")
    check('"cap_note": "；".join(' in lt or 'rect_note_ref[0]' in lt,
          "框选区域那句提示没带进 stats ⇒ 状态行不会说\"它按哪块屏在抓\" ✗")
    # ⭐⭐ **警报专用那一格要用"留住接管"版**（用户 2026-10-10 ✓ "本地实时断线后没有重连警报" ✓）：
    #   面板报警音是"`combat → 非 combat` 的**沿**" ✓ 读的是 stats 里 `alarm` 那一格 ✓ ——
    #   给它**原生** `_screen_state` 的话，测谎小游戏那段匹配不上 ⇒ 回落 combat ⇒ 沿被反复
    #   触发 ✗（响个不停 ✓）；断线那种"外接窗时有时无"的抖动也会把沿打散 ✗（该响没响 ✓）。
    #   ⚠ `screen` 那一格**保持原生** ✗（录屏/样本/状态行读它 ✓ `selftest_screen_state` 里
    #     有一条**故意**钉着它 ✓ 两件事不许合并 ✗）。
    check('"alarm": (getattr(self, "_takeover_st", None)' in lt,
          "stats 里没有**警报专用**的 `alarm` 格（或它退回原生值 ✗）⇒ 报警沿会在测谎小游戏"
          "那段反复触发、断线那一下反而漏响 ✗")
    check('"screen": self._screen_state' in lt,
          "`screen` 那格不是**原生识别结果**了 ✗ —— 录屏/样本/状态行（还有 "
          "`selftest_screen_state` 里那条钉子）都指望它是原生值 ✓ 别合并 ✗")
    lp2 = (ROOT / "gui" / "live_panel.py").read_text(encoding="utf-8")
    check('s.get("alarm")' in lp2,
          "面板的报警音没读 `alarm` 那一格（读了就拿不到\"留住接管\"的口径 ✗）")
    check("hwnd=" not in lt and "is_occluded" not in lt,
          "又改回\"只抓认死的那个窗口\"了 ✗ —— 那样盖住游戏的外接登录窗不进画面 ⇒ "
          "断线重连 / 测谎判不出界面 ✗（用户 2026-10-09 否掉过 ✓）")
    lp = (ROOT / "gui" / "live_panel.py").read_text(encoding="utf-8")
    check('s.get("cap_note")' in lp,
          "面板没把 `cap_note` 缀进状态行 ⇒ 人看不出\"它换了抓法\"（只会觉得画面变怪 ✗）")


def t_live_map_marker_is_authoritative():
    """⭐⭐⭐⭐⭐ **地形图上的玩家圈只能用「唯一那一份」世界坐标**（用户 2026-10-10 ✓ 原话：
    "**地形图都锁到红点上去了，报的坐标也与信息栏不一致，这个不应该自己走独立逻辑，
    它就应该根据最终信息栏读到的玩家世界坐标来绘制**"✓）。

    **病根**：`gui/route_panel._refresh_live_map` 原来把"**差分自己挑的那块连通域**"
    （`_diff_dot_center` ✓ "差得够亮的那一块"）当成玩家圈上去 ✗ ⇒ 底图/地图上任何
    **与面板不同源**的东西（**墙的红点**、地形线、集合色块 ✓）差得够亮就会被当成玩家
    ⇒ **锁到红点上** ✗，而且它报的世界坐标与信息栏那行**必然是两个数** ✗
    （两条链 —— 正是 2026-10-06 拆过一次的老毛病 ✓ 见 `decision/agent.LAST_POS` ✓）。

    口径（本次）：圈**只**来自 `LAST_POS` ✓（实时回路是唯一写者 ✓）、换算只走**地形**
    （`terrain.world_to_canvas` ✓ 与标定/显示区无关 ⇒ 不会因为 `view` 跟丢而漂 ✓）、
    **换图 / 没在跑 / 还没定上位 ⇒ 不画** ✓（绝不自己再算一份 ✗）；
    差分那层**照旧画**（它是给人看"差在哪"的 ✓）但**不再是玩家** ✗。

    钉四件：① 全文件里**只有一处**把 `_live_mark` 设成"带点的列表"；
    ② 那句在 `LAST_POS` 之后、且带 `map_id` 比对；③ 换算走 `t.world_to_canvas`；
    ④ **"差分找到点"这种当玩家的措辞不许再出现** ✗（回潮就红 ✓）。
    """
    src = (ROOT / "gui" / "route_panel.py").read_text(encoding="utf-8")
    check(src.count("self._live_mark = [_mark]") == 1,
          "① 设玩家圈的写法不止一处（又长出第二条链 ✗）：%d 处"
          % src.count("self._live_mark = [_mark]"))
    _i_lp = src.find('LAST_POS')
    _i_mk = src.find("self._live_mark = [_mark]")
    check(_i_lp >= 0 and _i_mk > _i_lp,
          "② 玩家圈不是从 `LAST_POS` 读出来的（先读坐标再画 ✓ 顺序反了 ✗）")
    check(src[_i_lp:_i_mk].count("str(mid)") >= 1,
          "② 没比 `map_id` ⇒ 换图之后会拿别的图的坐标往这张图上画 ✗")
    # ⚠ 换算那两行在**赋值之前**（先算 `_cx/_cy` 再拿去拼标记 ✓）⇒ 窗口要往前也取一段 ✓
    check("t.world_to_canvas(float(_wx), float(_wy))" in src[max(0, _i_mk - 1400):_i_mk + 200],
          "③ 换算没走地形（`terrain.world_to_canvas` ✓）⇒ 会随标定/显示区漂 ✗")
    check("差分找到点" not in src,
          "④ 又把「差分找到点」当成玩家位置了 ✗（用户 2026-10-10 明确否掉：它锁到墙的红点上 ✓）")
    # ⑤ 黄点像素那条优先（用户 2026-10-20 ✓ "它就应该标在实时小地图你认为黄点所在的位置上"）
    _seg = src[max(0, _i_mk - 2200):_i_mk + 300]
    check('_lp.get("dot_px")' in _seg,
          "⑤ 没读 `LAST_POS.dot_px` ⇒ 圈还是按世界坐标画，**压不到贴图上的黄点** ✗"
          "（用户 2026-10-20：它就应该标在实时小地图你认为黄点所在的位置上 ✓）")
    check("ox + _px * kx" in _seg and "oy + _py * ky" in _seg,
          "⑤ 黄点像素没走**贴图那一套映射**（`ox + _px * kx` ✓ 见 `_live_map_rect`）"
          "⇒ 圈和贴图各算各的、看着还是对不上 ✗")


def t_lie_rec_stops_after_success():
    """⭐⭐⭐⭐⭐ **测谎成功后必须收工**（用户 2026-10-10 ✓ 原话："**测谎了录屏录了很久，
    没有在成功后切断**"✗）。

    **现场**（用户给的录像 `lie_20261010_025017.mp4` ✓ 逐帧扫出来 ✓）：`lie_warn`(t≈0s, 0.856 ✓)
    → 小游戏（匹不上 ✗）→ **`lie_success`(t≈18s, **0.879** ✓ 认出来了 ✓)** → `combat`
    ⇒ **此后再也不变** ✗；而那两段"按时间到点就该做"的判据（"待停到点了吗" ✓ /"太久没再见
    测谎了吗"✓）原来待在 `if st == self._screen_state: return` 的**后面** ✗ ⇒
    **只有"状态又变一次"才会被评估** ✗✗ ⇒ 待停时刻永远没人检查 ⇒ 录屏一路录到
    `LIE_REC_MAX_S`=300 秒的保险才切 ✗（实测那段录像 **6819 帧 / 02:50:17→02:55:17 = 整 300 秒** ✓）。

    钉三件：
      ① **状态不再变**（同一拍 `st == _screen_state`）时，只要"待停到点" ⇒ **必须停** ✓；
      ② 同理"太久没再见到 `lie_*`"（GAP 兜底 ✓）也要在**不变**的那一拍生效 ✓；
      ③ 源码级：这两条必须在 `if st == self._screen_state` **之前** ✗（挪回去就红 ✓）。
    """
    import time as _time

    from gui.live_thread import LiveThread
    from perception.screen import screen_state as ss
    from perception.screen import ui_state as us

    t = LiveThread.__new__(LiveThread)
    t._screen_state = "combat"
    t._takeover_st = "combat"
    t._lie_seen_game = False
    t._lie_warn_at = 0.0
    t._lie_sess_t0 = 0.0
    t._lie_rec_top_game = 0.0
    t._lie_rec = object()                  # 假装"正在录"（只要非 None ✓）
    t._lie_rec_pending = 0.0
    t._lie_rec_saw_success = False
    t._lie_rec_last_lie = 0.0
    t._lie_rec_stop_at = 0.0
    t._rec_kind = "lie"
    stopped = []
    t._lie_rec_stop = lambda why="": stopped.append(str(why))
    t._lie_rec_start = lambda *a, **k: None
    t._lie_rec_write = lambda *a, **k: None

    cur = {"st": "combat"}
    _o_cf, _o_ui, _o_cap = ss.check_frame, us.detect, ss.capture_frame
    ss.check_frame = lambda raw: (cur["st"], {"score": 0.9})
    us.detect = lambda raw: None
    ss.capture_frame = lambda *a, **k: None
    try:
        # ⚠ **必须先 `lie_warn` 开一场** ✗ —— 不然 `lie_success` 会被 A 规则故意判成 `combat`
        #   （用户 2026-10-08 定的："必须先见过 `lie_warn` 才算测谎"✓），这条用例第一版就是这么红的 ✓
        cur["st"] = "lie_warn"
        t._screen_beat(None)
        check(bool(t._lie_warn_at), "warn 没开成一场（那后面认什么都白搭 ✗）")

        # 走到"见到成功弹窗 ⇒ 离开 lie_* ⇒ 记下待停时刻"
        cur["st"] = "lie_success"
        t._screen_beat(None)
        check(t._lie_rec_saw_success, "成功弹窗没被记下来（那后面就永远不收了 ✗）")
        cur["st"] = "combat"
        t._screen_beat(None)
        check(bool(t._lie_rec_stop_at), "离开测谎没记下「待停时刻」✗")

        # ① 让待停时刻**过期**，然后在"状态**不变**"的一拍上再 beat 一次 ⇒ 必须收工 ✓
        t._lie_rec_stop_at = _time.monotonic() - 1.0
        stopped[:] = []
        t._screen_beat(None)                    # st 仍是 combat == _screen_state ⇒ 早退那条路
        check(stopped, "⭐⭐ 状态不变的那一拍**没有收工** ⇒ 录屏会一路录到 300 秒保险 ✗✗"
                       "（用户 2026-10-10 报的就是这个 ✓）")

        # ② GAP 兜底（没见过成功弹窗、太久没再见 lie_*）也要在"不变"的一拍生效 ✓
        t._lie_rec_saw_success = False
        t._lie_rec_stop_at = 0.0
        t._lie_rec_last_lie = _time.monotonic() - (float(LiveThread.LIE_REC_GAP_S) + 5.0)
        stopped[:] = []
        t._screen_beat(None)
        check(stopped, "⭐ 「太久没再见到测谎 ⇒ 收工」也没在状态不变的那一拍生效 ✗")

        # ③ 源码级：两条必须在「状态没变就短路」之前，而且必须在**同一个口（每拍口）里**
        src = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
        # ⚠ 2026-10-15（阶段 1）：这一拍拆成了两个口 ⇒ 锚点跟着改 ✓ 口径一字不变 ✓：
        #   · 短路从 `if st == self._screen_state: return` 变成 `return st != self._screen_state` ✓
        #     （它现在待在**每拍口** `_screen_every_tick` 里 ✓）；
        #   · 于是"在短路之前"这句话必须连**同一方法**一起判 ✗ —— 只判"文件里的先后"
        #     是不够的（很可能一条在每拍口、另一条挪进了过渡口 ✗ 那正是踩过三次的坑 ✓）。
        _i_beat = src.find("def _screen_every_tick(self, raw, st, det):")
        _i_next = src.find("\n    def _screen_on_transition(self, raw, st, det):", _i_beat)
        _i_ret = src.find("return st != self._screen_state", _i_beat)
        _i_stop = src.find('self._lie_rec_stop("离开测谎（已稳定', _i_beat)
        _i_gap = src.find("长时间没再见到测谎", _i_beat)
        check(_i_beat > 0 and _i_next > _i_beat,
              "`_screen_every_tick` / `_screen_on_transition` 这两个口找不到了 ✗"
              "（结构变了 ⇒ 这条钉子要跟着改 ✓ 但**别把两半粘回一个函数** ✗）")
        check(_i_stop > 0 and _i_ret > 0 and _i_stop < _i_ret < _i_next,
              "「待停到点就收工」那条**不在每拍口里、或排在短路之后** ✗"
              "（那就只有「状态又变一次」才被评估 ✓ 用户 2026-10-10 报的正是这个 ✗）")
        check(_i_gap > 0 and _i_gap < _i_ret,
              "GAP 兜底（太久没再见到测谎 ⇒ 收工）也必须在短路之前、且在每拍口里 ✗")
    finally:
        ss.check_frame, us.detect, ss.capture_frame = _o_cf, _o_ui, _o_cap


def t_locate_skipped_while_taken_over():
    """⭐⭐⭐⭐ **界面接管期间这一拍不定位**（用户 2026-10-10 ✓ 原话："实时在跑 单这一拍还没定上位？
    我们定位有这么慢吗？在黑色背景里找个大黄点很难吗？"✓）。

    **实测**（`perf.log` 2026-10-10 01:55~01:57 ✓ 挨着的四段）：正常战斗 `locate_ms`
    **0.39 / 0.44 ms** ✓ vs `agent_why=界面接管（lie_warn）` 那两段 **73.04 / 42.97 ms** ✗
    —— 慢两个数量级 ✓。
    **为什么**：接管时小地图面板被弹窗盖住 ⇒ 认不出 ⇒ `PlayerLocator._search` 的兜底
    "小块里不像玩家点 ⇒ 退回**全画面重找**"（那条注释实测 62 ms ✗）⇒ 每拍都走重路 ✓
    （离线复盘同期真画面报的是「颜色层不可用（被淹）」✓ 对得上 ✓）。
    ⇒ 这段时间位置没人要（角色被锁操作 + Agent 早已停手 ✓）⇒ **直接不定位** ✓。

    钉三件：
      ① 接管态（`_takeover_st != combat`）⇒ **一次都不调定位器** ✓ 且返回 `None` ✓；
      ② 回到 `combat` ⇒ **照旧定位** ✓（正常路径一个字不许被碰 ✗）；
      ③ 源码级：这道闸必须在 `perf.ms("locate_ms", …)` **之前** ✗ —— 否则跳过的拍会被记成
         "0 ms 的定位"，`locate_ms` 这把尺子从此不可信 ✓（本次定案正是靠它 ✓）；且判据必须是
         **接管版** `_takeover_st`（用原生 `_screen_state` ⇒ 小游戏那段一回落就又去烧 60 ms ✗）。
    """
    from unittest import mock

    from core import perf as _perf
    from gui.live_thread import LiveThread

    src = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    _i_gate = src.find('if str(self.__dict__.get("_takeover_st") or "combat") != "combat":')
    # ⚠ 找**代码形**（带 `t0` 那个实参 ✓）：注释里也出现过 `perf.ms("locate_ms", …)` ✗
    #   ⇒ 光找函数名会命中注释、把这条断言变成"恒真/恒假"（本轮就这么翻过一次车 ✓）
    _i_ms = src.find('perf.ms("locate_ms", t0)')
    check(_i_gate >= 0 and _i_ms > _i_gate,
          "① 闸没装在 `perf.ms(\"locate_ms\")` 之前 ✗（跳过的拍会被记成 0 ms ⇒ 那把尺子废了 ✓）")
    check('perf.count("locate_skip")' in src[_i_gate:_i_gate + 400],
          "① 没记 `locate_skip`（事后看不出这段省了多少 ✗）")

    t = LiveThread.__new__(LiveThread)          # ⚠ QThread 只能 __new__（见本文件另一条 ✓）
    calls = []
    t._locate_mmap = lambda panel: (calls.append(panel) or {"ok": True})
    with mock.patch.object(_perf, "ms", lambda *a, **k: None), \
         mock.patch.object(_perf, "count", lambda *a, **k: None):
        for st in ("lie_warn", "lie_game", "login_err"):
            t._takeover_st = st
            calls[:] = []
            check(t._locate_latest() is None and calls == [],
                  "① 接管态（%s）还在每拍定位 ⇒ 白烧 40~70ms（面板被弹窗盖住，**认不出来** ✗）"
                  % st)
        t._takeover_st = "combat"
        calls[:] = []
        check(t._locate_latest() is not None and len(calls) == 1,
              "② 正常战斗时必须照旧定位（这条优化不许碰正常路径 ✗）：calls=%d" % len(calls))


def t_lie_takeover_holds_through_minigame():
    """⭐⭐⭐⭐⭐ **测谎开始后 Agent 不许再操作角色**（用户 2026-10-10 ✓ 原话："**为什么测谎开始后
    Agent 还在操作角色按方向键？**"✓）。

    **病根**：Agent 的停手判据是 `ws.screen != "combat"`（`decision/agent.py` 的「界面状态接管」✓），
    而**测谎小游戏那段画面本来就匹配不上任何模板** ✗（`gui/live_thread.py` 那段实测自己写着
    "中间那段'画面不匹配任何模板'是**常态**"✓）⇒ 状态回落 `combat` ⇒
    **Agent 当场把方向盘拿回去、又开始按方向键** ✗✗。
    ⚠ 录屏那侧早就用"宽限 ＋ 没见过成功弹窗一律不停"绕过了同一个坑 ✓（`_lie_rec_stop_at` /
      `_lie_rec_saw_success` ✓）—— **Agent 这侧漏了** ✗（本轮补 ✓：`_takeover_st` ✓）。

    钉七件（真跑 `_screen_beat` ✓ 三级判据全打桩 ⇒ 不起线程、不要流 ✓）：
      ① 平时（`combat`）⇒ 照样报 `combat` ✓（正常打怪**一字不变** ✓）；
      ② `lie_warn` ⇒ 立刻接管 ✓；
      ③ **warn 弹窗关掉、还没进小游戏**那几秒 ⇒ **也要一直接管** ✓（宽限 `LIE_REC_HOLD_S` ✓）；
      ④ `lie_game` ⇒ 接管 ✓（且记住"这场进过小游戏" ✓）；
      ⑤ ⭐⭐ **小游戏里画面匹配不上（回落 `combat`）30 秒** ⇒ **必须仍然接管** ✗✗
         （以前 1 拍回落 ⇒ Agent 就动了 ✓ 这条就是用户报的那个病 ✓）；
      ⑥ `lie_success` ⇒ 仍接管 ✓；回到 `combat` ⇒ **还权** ✓（测谎完了该恢复打怪 ✓）；
      ⑦ **没 `lie_warn` 打头的 `lie_game`**（误判路线 ✓）⇒ **不算数** ✓ 照旧报 `combat` ✓；
      ⑧ ⭐⭐ **`lie_warn` 响过 ⇒ 这一场（60s 内）一直留接管** ✗ —— ⚠ 这条 **2026-10-10 第二轮
         改了口径**（原来是"只见过 warn、没进小游戏 ⇒ 宽限一到就还权"✗）：`lie_game` 模板裁坏了
         （真小游戏 0.788~0.794 **低于门限 0.80** ⇒ 那段**根本认不出来**✓ 仓库里早记着 ✓）
         ⇒ 等"进过小游戏"= **永不接管** ✗（现场：`lie_rec` 录满 300 秒、而 `agent_why=-` ✓
         ⇒ Agent 全程在按移动/跳跃 ✓ 正是用户第二轮报的 ✓）；场次超时 ⇒ **必须还权** ✓。
    """
    import time as _time

    from gui import live_thread as LT
    from gui.live_thread import LiveThread
    from perception.screen import screen_state as ss
    from perception.screen import ui_state as us

    # ⚠ `LiveThread` 是 `QThread` ⇒ 只能 `__new__`（`object.__new__` 会抛 ✗ 见本文件另一条 ✓）
    t = LiveThread.__new__(LiveThread)
    t._screen_state = "combat"
    t._takeover_st = "combat"
    t._lie_seen_game = False
    t._lie_warn_at = 0.0
    t._lie_sess_t0 = 0.0
    t._lie_rec_top_game = 0.0
    t._lie_rec = None
    t._lie_rec_pending = 0.0
    t._lie_rec_path = ""
    t._lie_rec_n = 0
    t._lie_rec_t0 = 0.0
    t._lie_rec_err = ""
    t._lie_rec_saw_success = False
    t._lie_rec_last_lie = 0.0
    t._lie_rec_stop_at = 0.0
    t._rec_kind = ""
    t._lie_rec_start = lambda *a, **k: None      # 录屏不测（测的是"报给 Agent 的状态"✓）
    t._lie_rec_stop = lambda *a, **k: None

    cur = {"st": "combat"}
    _o_cf, _o_ui, _o_cap = ss.check_frame, us.detect, ss.capture_frame
    ss.check_frame = lambda raw: (cur["st"], {"score": 0.9})
    us.detect = lambda raw: None
    ss.capture_frame = lambda *a, **k: None
    try:
        def _beat():
            t._screen_beat(None)

        _beat()
        check(t._takeover_st == "combat", "① 平时不该接管：%r" % (t._takeover_st,))

        cur["st"] = "lie_warn"
        _beat()
        check(t._takeover_st == "lie_warn",
              "② `lie_warn` 出现后必须立刻接管（Agent 该停手）：%r" % (t._takeover_st,))

        cur["st"] = "combat"
        _beat()
        check(t._takeover_st != "combat",
              "③ warn 弹窗关掉、还没进小游戏那几秒就被还权了 ⇒ Agent 会立刻动 ✗：%r"
              % (t._takeover_st,))

        cur["st"] = "lie_game"
        _beat()
        check(t._takeover_st == "lie_game" and t._lie_seen_game,
              "④ 进小游戏后必须接管且记住这场进过小游戏：%r / seen=%r"
              % (t._takeover_st, t._lie_seen_game))

        # ⑤ 小游戏里画面匹配不上（回落 combat）—— 已过 30 秒（远小于一场的上限 60 秒 ✓）
        t._lie_warn_at = _time.monotonic() - 30.0
        cur["st"] = "combat"
        _beat()
        check(t._takeover_st != "combat",
              "⑤⭐⭐ **小游戏里画面匹配不上就回落 `combat`** ⇒ Agent 当场拿回方向盘、"
              "又开始按方向键 ✗✗（用户 2026-10-10 报的就是这个）：%r" % (t._takeover_st,))

        cur["st"] = "lie_success"
        _beat()
        check(t._takeover_st == "lie_success",
              "⑥ 成功弹窗期间仍该接管：%r" % (t._takeover_st,))

        cur["st"] = "combat"
        _beat()
        check(t._takeover_st == "combat",
              "⑥ 测谎完了该**还权**（不然永不恢复打怪 ✗）：%r" % (t._takeover_st,))

        # ⑦-1 没有 warn 打头的 lie_game（误判路线）⇒ 不算数
        cur["st"] = "lie_game"
        _beat()
        check(t._takeover_st == "combat",
              "⑦ **没见过 `lie_warn` 的 `lie_game` 不算测谎**（误判不许冻住 Agent ✗）：%r"
              % (t._takeover_st,))

        # ⑦-2 ⭐⭐ **`lie_warn` 响过 ⇒ 这一场之内一直留接管**（用户 2026-10-10 第二轮 ✓
        #    原话："**测谎的时候 Agent 还在操作移动、跳跃**"✗）—— ⚠ 这条**替掉了**上一版
        #    "只见过 warn、没进小游戏 ⇒ 宽限一过就还权"✗✗：因为 `lie_game` 那块模板**裁坏了**
        #    （仓库里早记着：真小游戏 0.788~0.794 **低于门限 0.80** ⇒ 小游戏那段**根本认不出来** ✗）
        #    ⇒ 等"进过小游戏"= **永不接管** ✗（现场就是 `lie_rec` 录满 300 秒、而 `agent_why=-` ✓）。
        #    真测谎 warn 之后几秒进小游戏、全程 ≈25 秒 ⇒ 60 秒窗口完全盖得住 ✓。
        cur["st"] = "lie_warn"
        _beat()
        check(t._takeover_st == "lie_warn", "⑦ 重启一场的 warn 该接管 ✓")
        t._lie_warn_at = _time.monotonic() - 30.0        # 小游戏进行到 30 秒（仍在这一场里 ✓）
        cur["st"] = "combat"
        _beat()
        check(t._takeover_st != "combat",
              "⑦⭐ warn 响过、小游戏**认不出来**（画面回落 `combat`）时把操作权还回去了 ⇒ "
              "Agent 又开始按移动/跳跃 ✗✗（用户 2026-10-10 第二轮报的就是这个）：%r"
              % (t._takeover_st,))
        # 这一场超时（> LIE_SESSION_MAX_S）⇒ 必须还权（不许把 Agent 永久冻住 ✗）
        t._lie_warn_at = _time.monotonic() - (float(LT.LIE_SESSION_MAX_S) + 1.0)
        cur["st"] = "combat"
        _beat()
        check(t._takeover_st == "combat",
              "⑧ 这一场都超时 %g 秒了还留着接管（把 Agent 永久冻住 ✗）：%r"
              % (float(LT.LIE_SESSION_MAX_S), t._takeover_st))
    finally:
        ss.check_frame, us.detect, ss.capture_frame = _o_cf, _o_ui, _o_cap


def t_convert_only_when_wanted():
    """⭐⭐ **解出来的帧"没人要"就不做颜色转换**（用户 2026-10-10 ✓ 原话："**A 推流 60fps 不影响吧，
    我们的重点是提升 B 机的效率，A 就算推 240fps 我们也需要能应对**"✓）。

    本机实测（走产品这条路 ✓ 见 `tools/perf_audit` 以外的这次直接基准 ✓）：
      · **纯解码 0.71 ms/帧（1399 帧/秒）** ✓；
      · **解码 + `bgr24` 转换 5.93 ms/帧（169 帧/秒）** ✗ —— **贵 8 倍** ✓。
    ⇒ 瓶颈**不在解码**，在"AV帧 → numpy BGR"那一步 ✓；而它原来是**每帧都做** ✗
      ⇒ 输入 60fps 就烧掉约三分之一核、**240fps 要一个半核** ✗✗（"推 240 扛不住"的真因 ✓）。

    做法：`PyAVSource(keep_check=…)` —— **解码照旧喂满**（UDP 下跳包会破坏参考帧 ✗ 这是两回事 ✓），
    只有 `keep_check()` 为真才做那步昂贵转换 ✓ ⇒ 转换成本被**消费速度**封顶 ✓
    （判据 `_LatestSlot.is_idle` ✓ = "槽里没有待取走的帧，这帧真有人要" ✓）。

    钉四件（前两条源码级、后两条真跑）：
      ① 跳过必须发生在 `to_ndarray` **之前** ✗（放后面 = 白省 ✓）；
      ② 主回路必须传判据 + 记 `convert_skip`（"省了多少活"的证据 ✓）；
      ③ **真跑**：拒绝前 5 帧 ⇒ 恰好 5 帧被丢、第 6 帧正常返回 ✓（误差 0 ✓）；
      ④ `keep_check=None` ⇒ **老行为一字不变** ✓（不传就不省 ✓）。
      ⚠ ⑤ 没有样本视频 ⇒ 只跑源码那两条（本仓库有跳过依赖的先例 ✓ 别让用例硬失败 ✗）。
    """
    src = (ROOT / "link" / "pyav_source.py").read_text(encoding="utf-8")
    _i_chk = src.find("if self.keep_check is not None:")
    _i_nd = src.find("av_frame.to_ndarray(format=self.decode_format)")
    check(_i_chk >= 0 and _i_nd > _i_chk,
          "`keep_check` 的跳过没放在 `to_ndarray` **之前** ⇒ 那一步照样白做 ✗")
    ltsrc = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    check("keep_check=_want_frame" in ltsrc and 'perf.count("convert_skip")' in ltsrc
          and "def is_idle(self):" in ltsrc,
          "主回路没把「有人要吗」的判据传下去（或没记 `convert_skip`）⇒ 这条优化不会生效 ✗")

    files = sorted((ROOT / "data" / "recordings").glob("*.mp4"))
    if not files:
        return          # ⑤ 没样本 ⇒ 只跑源码那两条 ✓
    sys.path.insert(0, str(ROOT))
    from link import PyAVSource
    n_seen = [0]

    def _kc():
        n_seen[0] += 1
        return n_seen[0] > 5

    s = PyAVSource(str(files[0]), decode_format="bgr24", keep_check=_kc)
    try:
        s.open()
        f = s.read()
    finally:
        s.close()
    check(f is not None and s.skipped_convert == 5,
          "「没人要就不转换」不精确：期望丢 5 帧、返回第 6 帧，实际丢 %d 帧、返回 %s ✗"
          % (s.skipped_convert, "有帧" if f is not None else "None"))

    s2 = PyAVSource(str(files[0]), decode_format="bgr24", keep_check=None)
    try:
        s2.open()
        f2 = s2.read()
    finally:
        s2.close()
    check(f2 is not None and s2.skipped_convert == 0,
          "不传判据时行为被改了（老路应当**每帧都转换** ✓）：skipped=%d ✗" % s2.skipped_convert)


def t_frame_ctx_discipline():
    """⭐⭐⭐ **一帧的上下文（`runtime.pipeline.FrameCtx`）三条纪律**（用户 2026-10-15 选的口径 B ✓）。

    `_run` 1651 行的根子是「一帧的东西散在循环体里」✗ —— 盒子是来收它的 ✓。但盒子自己
    有两三种**典型坏法**，全是「能跑、不报错、事后极难查」✗ ⇒ 先把纪律钉死，再往里搬：

      ① **字段写死**（`__slots__` ✓）：写一个没声明的字段必须**当场报错** ✗ —— 不然它退化成
         「谁都能塞的袋子」= 换了个名字的全局变量 ✓（那比现在更难查 ✓）；
      ② **一帧一个**（**不复用** ✓）：两个盒子互不干扰 ✓（复用省的那点分配，跟「上一帧的怪框
         漏进这一帧」这种静默 bug 比不值 ✓）；
      ③ **跨帧的东西一律不进** ✗：帧计数 / 上一帧时刻 / 采样窗口 / 配置 / `*_ref` 那种回填
         小列表 ⇒ 留在线程上 ✓（判据一句话：**「下一帧还会用吗？」** ✓）。
    """
    from runtime.pipeline import FrameCtx

    _fields = ("f", "vis", "raw", "t_take", "t_pipe", "ws")

    # ① 声明的字段：默认 None ✓（不是空串 / 空列表那种**会骗人**的默认值 ✗）、能读能写 ✓
    c = FrameCtx()
    for _k in _fields:
        check(getattr(c, _k) is None,
              "`FrameCtx.%s` 的默认值不是 None ⇒ 阶段分不清「还没填」和「填了个空」✗：%r"
              % (_k, getattr(c, _k)))
    c.t_take, c.ws = 12.5, object()
    check(c.t_take == 12.5 and c.ws is not None, "填进去的值读不出来 ✗")

    # ① 高墙：没声明的字段 ⇒ 当场炸（**这条是重点** ✓ 它挡的正是"退化成袋子" ✓）
    try:
        c.not_declared = 1
        check(False,
              "给 `FrameCtx` 写一个**没声明的字段**居然成功了 ✗ ⇒ 它退化成「谁都能往里塞」"
              "的袋子（= 换名字的全局变量）⇒ 静默、事后极难查 ✓")
    except AttributeError:
        pass
    check(not hasattr(c, "__dict__"),
          "`FrameCtx` 实例居然有 `__dict__`（`__slots__` 失效）⇒ 又能塞没声明的字段 ✗")

    # ② 一帧一个：互不干扰
    _a, _b = FrameCtx(), FrameCtx()
    _a.vis = "A"
    check(_b.vis is None, "两个盒子串了（写 A 影响到了 B）⇒ 复用了同一个盒子 ✗：%r" % (_b.vis,))

    # ③ 结构钉子：字段清单本身就是"什么属于这一帧"的口径 ✓
    check(FrameCtx.__slots__ == _fields,
          "`FrameCtx` 的字段清单变了（2026-10-15 起 = %r）—— 加字段是**正常**的 ✓ 但要盯两件："
          "① 加进来的必须**只属于这一帧** ✗（帧计数 / 上一帧时刻 / 采样窗口 / 配置 / `*_ref`"
          "那类回填小列表 ⇒ 留在线程上 ✓ 判据：「下一帧还会用吗？」✓）；② 顺手把这条钉子的"
          "清单和 `FrameCtx` 的文档一起改 ✓" % (FrameCtx.__slots__,))

    # ④ ⚠⚠ 「**抽出来的阶段方法要真能跑**」—— 这条是拿一次**现场事故**换来的（2026-10-15 ✓
    #   用户当场截图：「失败：NameError: name 'perf' is not defined」✗）。
    #   根因：`perf`（还有 `behavior` / `mmap` / `WorldState` …）在这个文件里**一直是局部导入** ✗
    #   ⇒ 把它写进**新方法**时，那个方法**必须自己再 import 一次** ✓
    #   （漏了 ⇒ 一开预览当场炸 ✗ 而**自检全绿** ✗ —— 没人真调过那个新方法 ✓）。
    #   ⚠ 这条**刻意不靠 pyflakes** ✗：本机 `python -m pyflakes` 直接报 `No module named
    #     pyflakes` ⇒ 那条静态检查钉一直在**静默跳过** ✓（今天这个错它本来能抓 ✓）
    #     ⇒ 这里改成**真调一次**：只要方法里少任何一句它自己的 import / 少任何一句口径，
    #     这一句就会当场抛 `NameError` 或断言失败 ✓。
    import types as _types

    from core import perf as _cperf
    from runtime.pipeline import HeavyGate as _HG

    from gui import live_thread as _lt

    #   ⚠ 替身还得摆 `_infer`（`_heavy_gate` 要问它"现在跑不跑重活" ✓ —— 第一版漏了 ⇒
    #     报的是 `'SimpleNamespace' object has no attribute '_infer'` ✗ 而不是我们要验的那个错 ✗
    #     所以这条钉**自己也要反向验证**才敢信 ✓）
    _th = _types.SimpleNamespace(_gate=_HG(),
                                 _infer=_types.SimpleNamespace(is_set=lambda: True))
    _th._gate.fps_limit = 0.0                     # 0 = 不限制 ⇒ 这一拍必放行 ✓
    # ⚠ 替身绕过 `__init__` ⇒ `_stage_frame_ready` 里 `self._heavy_gate(...)` 得先绑上 ✓
    #   （同 `selftest_screen_state` 里那几个替身的手法 ✓）
    _th._heavy_gate = _lt.LiveThread._heavy_gate.__get__(_th)
    _hits, _old_fa = [], _cperf.frame_arrived
    _cperf.frame_arrived = lambda t: _hits.append(t)
    try:
        _c2 = _types.SimpleNamespace(f=_types.SimpleNamespace(image="VIS"), t_take=1.0,
                                     t_pipe=None, vis=None, ws=None, raw=None)
        _ok2 = _lt.LiveThread._stage_frame_ready(_th, _c2, None, None)
    finally:
        _cperf.frame_arrived = _old_fa
    check(_ok2 is True and _c2.vis == "VIS" and len(_hits) == 1,
          "`_stage_frame_ready` **真调一遍**没过（0=不限制 ⇒ 该放行 ✓、画面该入盒 ✓、"
          "`perf.frame_arrived` 该被调一次 ✓）—— 实得 ok=%r vis=%r hits=%d ✗"
          % (_ok2, _c2.vis, len(_hits)))


def t_heavy_gate_unit():
    """⭐⭐ 重活闸的**本体**当纯逻辑跑（`runtime.pipeline.HeavyGate` ✓ 2026-10-15 阶段 1 ✓）。

    上面那条钉的是"**接线还在不在**"（源码锚点 ✓），这条钉的是"**本体算得对不对**" ✓
    —— 搬家最怕的就是"搬的过程中顺手改了一点东西" ✗，那种改动**开游戏也很难看出来** ✗
    （限速错了只表现为"节奏怪怪的"✓）。四条：

      ① `0` = **不限制** ⇒ 永远放行、且**一次都不喂**时间拍（老行为一字不变 ✓ 默认必须是 0 ✓）；
      ② 到点前**跳过**、到点后**放行**、放行后时刻前移（第二拍又开始跳 ✓ 节奏对 ✓）；
      ③ ⚠⚠ 跳过的拍**必须喂时间拍**（`tick_timing` ✓）—— 漏了就是"限了速 ⇒ 攻击节奏漂移"✗；
         **只在跳过时**喂 ✓（放行那一拍多喂一次也错 ✗ 会双倍喂 ⇒ 节奏偏快 ✗）；
      ④ 不在推理（`active=False`）⇒ **不拦**、也**不喂**（那条路另有主人 ✓）；
      ⑤ 时间拍抛异常**不许炸回路**（`try/except` 兜住 ⇒ 仍返回"跳过"✓）；
      ⑥ 打点齐全（`timing_ms` / `timing_calls` / `skip_heavy` ✓ ⇒ 事后看得出闸在干活 ✓）。

    ⚠ 全程**不碰真 `perf`**：把它两个口换成记录器 ✓（不然会写进现场 `perf.log` ✗）。
    """
    from runtime import pipeline as _pl
    from runtime.pipeline import HeavyGate

    class _A:
        def __init__(self, boom=False):
            self.calls = 0
            self.boom = boom

        def tick_timing(self, ws):
            self.calls += 1
            if self.boom:
                raise RuntimeError("时间拍炸了")

    _seen = []
    _old_ms, _old_cnt = _pl.perf.ms, _pl.perf.count
    _pl.perf.ms = lambda name, *a: _seen.append(("ms", name))
    _pl.perf.count = lambda name, *a: _seen.append(("cnt", name))
    try:
        g, a = HeavyGate(), _A()
        # ① 0 = 不限制
        check(g.fps_limit == 0.0, "默认档位不是 0（不限制）✗：%r" % (g.fps_limit,))
        check(g.skip_now(1000.0, True, a, None) is False
              and g.skip_now(1000.001, True, a, None) is False,
              "`0` = 不限制时**不该跳过**任何一拍 ✗（老行为一字不变 ✓）")
        check(a.calls == 0 and _seen == [],
              "不限制那一路**喂了时间拍 / 打了点** ✗（它该像闸不存在一样 ✓）：%r" % (_seen,))

        # ②③ 到点前跳、到点后放行；跳过的那拍喂时间拍、放行的不喂
        g.fps_limit = 30.0                      # 每 1/30 s 放行一拍
        _t = 5000.0
        check(g.skip_now(_t, True, a, None) is False,
              "第一拍就该放行（从没跑过 ⇒ 必放行 ✗）—— 否则开局要先白等一个周期 ✓")
        check(g.skip_now(_t + 0.005, True, a, None) is True, "没到点却没跳过 ✗")
        check(g.skip_now(_t + 1.0 / 30.0, True, a, None) is False, "到点了却不放行 ✗")
        check(a.calls == 1,
              "⚠⚠ 跳过的那一拍**没喂时间拍**（或多喂了）⇒ 输出CD/delay/跳间隔会漂 ✗："
              "tick_timing 调了 %d 次（该 1 次：只有跳过那拍喂 ✓）" % a.calls)
        check([n for k, n in _seen if n == "skip_heavy"] == ["skip_heavy"],
              "跳过的那一拍没计 `skip_heavy`（事后看不出闸在干活 ✗）：%r" % (_seen,))
        check([n for k, n in _seen if k == "ms"] == ["timing_ms"],
              "喂时间拍没打 `timing_ms`（口径变了 ✗）：%r" % (_seen,))

        # ④ 不在推理 ⇒ 不拦也不喂
        _n = a.calls
        check(g.skip_now(_t + 1.0 / 30.0 + 0.001, False, a, None) is False,
              "没在推理时闸还拦（那时候没有重活可跳 ✗）")
        check(a.calls == _n, "没在推理时还喂了时间拍（那条路另有主人 ✗）")

        # ⑤ 时间拍炸了 ⇒ 不许炸回路（仍按"跳过"处理 ✓）
        b = _A(boom=True)
        check(g.skip_now(_t + 1.0 / 30.0 + 0.002, True, b, None) is True,
              "时间拍抛异常时没兜住 ⇒ 整个实时回路会被一个打点炸掉 ✗")
    finally:
        _pl.perf.ms, _pl.perf.count = _old_ms, _old_cnt


def t_heavy_gate_when_machine_busy():
    """⭐⭐ **重载保命闸**：每秒最多跑多少拍「推理＋追踪＋决策＋画框」（用户 2026-10-10 ✓
    原话："**我就是想在挂游戏+浏览器+音乐+steam+玩大型游戏的时候，实时还能流畅跑**"✓）。

    为什么要这道闸（实测 ✓）：`pipe_ms` 里**推理占 66~70%**（`infer_ms` 13ms / `pipe_ms` 20ms ✓），
    而输入推流是 **60fps** ⇒ 天棚只比输入高一点点 ⇒ 机器上再有人抢 GPU/CPU ⇒ **必然积压** ✗
    （2026-10-10 现场：e2e 3.7~4.7 秒、`src_lag=on` ✓，而人**就是**要同时开游戏/浏览器/音乐 ✓）。
    ⇒ 唯一的出路是**给每秒的活量封顶** ✓（不是"等机器空出来"✗）。

    做法：到点才跑那条重链（`_infer` 分支 ✓），不到点的帧**直接丢**（同 `_LatestSlot` ✓）
    —— 我们永远只处理**最新**帧 ⇒ **延迟不升** ✓（反而因为收流侧更松而降 ✓）。

    钉四件（源码级 ✓ 这几条只在真跑起来时才看得出 ✗）：
      ① `pipe_fps` 必须来自 **live 配置热读** ✓（能现场调、1 秒生效 ✓）；
      ② 闸必须在**重活之前**、且**跳过的拍 `continue`** ✓（不是"跑完再丢"✗）；
      ③ ⚠⚠ **跳过的拍必须照喂时间拍**（`agent.tick_timing` ✓）—— 输出CD / delay / 跳间隔
         那条序列走**本机绝对时钟** ✓ ⇒ 跳拍**不影响精确度** ✓；漏了这条就是"限了速 ⇒
         攻击节奏变慢/延迟漂移"✗（很难查 ✓）；
      ④ `0` = **不限制** ⇒ 这一整段等于不存在 ✓（**老行为一字不变** ✓ 默认值必须是 0 ✓）。
    """
    # ⭐ 2026-10-15 阶段 1「编排收口」：闸的**实现**搬进了 `runtime/pipeline.py::HeavyGate` ✓
    #   ⇒ 这条改成**两个文件各查各的那一半** ✓（判据口径一字不改 ✓）：
    #     · `gui/live_thread.py` 该留：配置热读 / 初值 / 档位委托口 / 主循环那一下 `continue` ✓；
    #     · `runtime/pipeline.py` 该留：判据本体（到点没有 + 跳过必喂时间拍 + 计数 + 回信号 ✓）。
    #   ⚠ 关键的一条是"**本体不许留副本**" ✗：判据在两边各写一份 ⇒ 迟早分叉 ⇒
    #     到时候"到底限没限速"说不清 ✓（那正是排查性能时最怕的 ✓）。
    src = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    pipe = (ROOT / "runtime" / "pipeline.py").read_text(encoding="utf-8")
    # ⚠ 热读那一行 2026-10-17 改过形状（测谎让路 ⇒ 先算 `_base`、再 `vt_yield_pipe_fps` ✓）——
    #   锚点跟着改成"**真的热读了 + 真的设到闸上**"，别钉死某一行写法 ✗
    check('_live.get("pipe_fps"' in src
          and ("self._pipe_fps = vt_yield_pipe_fps(_base" in src
               or "self._pipe_fps = max(0.0" in src),
          "重活闸没从 live 配置热读 `pipe_fps`（现场调不了 ✗）")
    check("if self._pipe_fps > 0.0 and self._infer.is_set():" not in src,
          "闸的**本体**又出现在 `gui/live_thread.py` 里了（该只在 `runtime/pipeline.py` ✓）"
          "—— 两边各一份判据 = 迟早分叉 ✗")
    check("if self.fps_limit <= 0.0 or not active:" in pipe,
          "闸没装在**重活之前**（跑完再丢 = 白跑 ✗）：`runtime/pipeline.py` 里找不到那道判据 ✓")
    _i_gate = pipe.find("if self.fps_limit <= 0.0 or not active:")
    _i_cnt = pipe.find('perf.count("skip_heavy")')
    check(_i_gate >= 0 and _i_cnt > _i_gate,
          "跳过的那一帧没计数（`skip_heavy` ✓ ⇒ 事后看不出闸有没有在干活 ✗）")
    _mid = pipe[_i_gate:_i_cnt]
    check("agent.tick_timing(ws)" in _mid,
          "⚠⚠ 跳过的帧**没喂时间拍**（`agent.tick_timing`）⇒ 输出CD/delay/跳间隔会跟着变慢 ✗"
          "（那条序列走本机绝对时钟 ✓ 跳拍本可以完全不影响它 ✓）")
    # ⚠ 2026-10-13 阶段 1：闸抽成 `_heavy_gate()` 方法后，跳过的语义 = 方法 `return True`
    #   + 主循环 `continue` ⇒ 锚点跟着变 ✓（原"`skip_heavy` 后 260 字符内找 `continue`"作废 ✗）。
    check("return True" in pipe[_i_cnt:_i_cnt + 200],
          "跳过的帧没**返回跳过信号**（该 `return True` 让主循环去取下一帧 ✗）—— 那就成了「跑完再丢」白跑 ✗")
    check("if not self._stage_frame_ready(ctx, agent, ws):\n                    continue" in src,
          "主循环没按**阶段**的返回值 `continue` ✗（2026-10-15 阶段 1：闸搬进了 "
          "`_stage_frame_ready` ✓ 主循环只认它的返回值 ✓）")
    check("if self._heavy_gate(agent, ws):\n            return False" in src,
          "阶段里没按闸的返回值 `return False` ✗（口径：闸说跳过 ⇒ 这一拍就到这儿 ✓）")
    # ⭐ 2026-10-15 阶段 1：「这一帧到手之后」那一段搬进了 `_stage_frame_ready` ✓ ⇒ 顺手钉住
    #   三个**时间戳/盒子**（⚠ 这三个是"搬家时最容易用错、而且**开游戏也看不出来**"的那一类 ✓
    #   —— 我把 `t_pipe` 换成 `t_take` 试过一次，**61/61 全绿** ✗ 实测抓到的缺口 ✓）：
    check("ctx.t_pipe = time.perf_counter()" in src,
          "「收到这一帧 → 决策完」的起点**不是现在** ✗（拿 `t_take` 顶替 ⇒ 槽等待被算进 "
          "`pipe_ms` / `out_key_ms` ⇒ 全线偏大，还看不出来 ✓）")
    check("ctx = FrameCtx(f=f, t_take=_t_take)" in src,
          "主循环没给这一帧建盒子 ✗（口径 B：阶段之间只认 `FrameCtx` ✓ 一帧一个 ✓）")
    check("_t_take = time.perf_counter()" in src,
          "取帧那一刻 `_t_take` 没了 ✗（`slot_wait_ms` 全靠它算 ✓ 见那段注释 ✓）")
    check("self._pipe_fps = 0.0\n" in src,
          "默认不是「不限制」（0）⇒ 老行为被改了 ✗")


def t_precheck_source_per_map():
    """⭐⭐⭐ 「开自动前的检查」必须与**路线识别页**用**同一处来源口径**（用户 2026-10-10 ✓
    原话：**"为什么我标定过了还显示这个"** ✓）。

    现场（一条命令量的 ✓）：
      · `datasets/map/107000100.route.json` 里 `mmap_src: "live"`（=「从实时画面」✓
        标定就在这条下 ✓ 有 `mode`/`scale`/`offset` ✓）；
      · 全局 `config/live.yaml` 里是 `stream`（=「独立推流」✗ 那一格**只有 `alpha`** ✗）；
      · 来源 **2026-10-03 起按地图 id 存** ✓，而体检当时读的是**全局**那份 ✗
        ⇒ 去问 `stream` ⇒ 报"还没标定" ✗ —— **人明明刚标过** ✓（标定按来源分开存 ✓
        见 `core/mapdata.calib_path` ✓）。

    钉四件：
      ① `mm.source_for_map(map_id)`：**按图优先** ✓、本图没灌过才回退全局 ✓、非法值回退 ✓；
      ② 体检**真的**用按图那份（喂"本图 = live 且有几何" ⇒ **不许报问题** ✓）；
      ③ 当前来源没量、**另一条来源量过** ⇒ 提示里要**点名那条来源** ✓
         （别让人对着"没标定"发呆 ✓）；
      ④ **源码级**：体检里**不许**再出现裸 `mm.live_src()` ✗ —— 那正是口径分叉的入口 ✓。
    """
    import core.mapdata as md
    import perception.minimap as mm
    from core import route_cfg
    from perception import map_const
    from gui.player_panel import PlayerPanel

    _geo = {"mode": "fit", "scale": 1.0, "offset": [0.0, 0.0]}
    _saved = (md.load_calib, md.load_calibs, md.load, mm.source_for_map, route_cfg.load)
    try:
        # ---- ① `source_for_map`：按图优先 / 回退全局 / 非法值回退 ✓ ----
        route_cfg.load = lambda mid: {"mmap_src": "live"}
        check(mm.source_for_map("107000100") == map_const.SRC_LIVE,
              "本图灌过来源 ⇒ 该用它（体检与路线识别页必须同一条）✗")
        route_cfg.load = lambda mid: {"mmap_src": "不合法"}
        check(mm.source_for_map("107000100", {"mmap_src": "stream"}) == map_const.SRC_STREAM,
              "本图那份不合法 ⇒ 该回退全局 `live.yaml` ✗")
        route_cfg.load = lambda mid: {}
        check(mm.source_for_map("107000100", {"mmap_src": "live"}) == map_const.SRC_LIVE,
              "本图没灌过 ⇒ 该用全局那份 ✗")

        # ---- ② 体检真的用按图那份（现场就是这首 ✓）----
        route_cfg.load = lambda mid: {"mmap_src": "live"}     # 本图 = live（有几何 ✓）
        md.load_calib = lambda mid, src, zoom=None: (_geo if src == "live" else {"alpha": 77})
        md.load_calibs = lambda mid: {"live": _geo, "stream": {"alpha": 77}}

        class _T:
            canvas = object()

        md.load = lambda mid, with_canvas=False: _T()
        _p = PlayerPanel._precheck_problems("107000100", need_mmap=True)
        check(not _p,
              "本图那条来源（live ✓）**量过了**却仍报问题 ✗ —— 体检读错来源了 ✓"
              "（= 用户现场\"我标定过了还显示这个\"✓）：%r" % (_p,))

        # ---- ③ 当前来源没量、另一条量过 ⇒ 点名那条 ✓ ----
        route_cfg.load = lambda mid: {"mmap_src": "stream"}
        _p2 = PlayerPanel._precheck_problems("107000100", need_mmap=True)
        _t2 = "\n".join(_p2)
        check("还没标定" in _t2, "当前来源确实没量 ⇒ 该报\"还没标定\"：%r" % (_t2,))
        check("从实时画面" in _t2,
              "另一条来源（live =「从实时画面」✓）量过了却**没点名** ⇒ 人只能对着"
              "\"没标定\"发呆 ✗（用户 2026-10-10 ✓）：%r" % (_t2,))
    finally:
        (md.load_calib, md.load_calibs, md.load,
         mm.source_for_map, route_cfg.load) = _saved

    # ---- ④ 源码级：体检里不许再用裸 `mm.live_src()` ✗ ----
    _txt = (ROOT / "gui" / "player_panel.py").read_text(encoding="utf-8")
    _body = _txt.split("def _precheck_problems", 1)[-1].split("def _auto_precheck_problems",
                                                             1)[0]
    check("mm.source_for_map(" in _body,
          "体检没走 `mm.source_for_map` ⇒ 又和路线识别页分叉了 ✗（按图存的那份会被忽略 ✓）")
    check("mm.live_src()" not in _body,
          "体检又出现裸 `mm.live_src()` 了 ✗ —— 那是**全局**那份，正是"
          "\"我标定过了还显示没标定\"的病根 ✓（用户 2026-10-10 ✓）")

    # ---- ⑤ 全仓库不许再出现「拿**全局**来源去读本图标定」✗ ----
    #      （那是同一族病：本图那份会被忽略 ✓ 现场 `stream` 桶里只剩 `alpha:77` ✓
    #        就是 `settings_dialog` 存浓淡时用全局来源写出来的 ✓）
    for _f in ("gui/player_panel.py", "gui/settings_dialog.py", "gui/route_panel.py"):
        _bad = [ln.strip() for ln in (ROOT / _f).read_text(encoding="utf-8").splitlines()
                if "load_calib(" in ln and "live_src(" in ln]
        check(not _bad,
              "%s 里还拿**全局** `live_src()` 去读标定 ✗ ⇒ 本图那份被忽略（同一族病 ✓）：%r"
              % (_f, _bad))


def t_mob_id_label_optional():
    """⭐⭐ **怪框角标 `#id conf/peak`**（用户 2026-10-11 ✓ 原话："**同个怪物的检出框疯狂
    闪烁**"✓）。

    为什么要有它：两种"闪"的成因**完全不同** ✗ —— ① 同一个 id 一直在（置信度忽高忽低 ✓）
    = **检测不稳** ✓；② **每帧 id 都换** ✓ = 认领 / 新建那一步有问题 ✓
    ⇒ 不把 id 打出来**只能靠猜** ✓（沿用"坏了要看得见"那条 ✔ 见 SKILL 315 ✓）。

    钉五件：
      ① 默认**关** ✓（老观感一字不变 ✓）；
      ② ⚠⚠ **不在 `_VIS_ON_KEYS`** ✗（那批的语义是"缺省 / 写坏 ⇒ **True（画）**"✓
         ⇒ 塞进去会让**所有人**的框上都多一串数字 ✗）；
      ③ **存得进读得回**（坏值回 `False` ✓ 用完无论如何还原 ✓）；
      ④ 画的时候真读了这个键 ✓、格式里真有 **id / 两个置信度**（`%.2f/%.2f` ✓）；
      ⑤ 关着时**一字不变**（还是老的 `"%s %.2f" % (_m_label, m.conf)` ✓）。
    """
    from gui import theme

    # ⚠ 2026-10-11 起是**两个键**（怪 `mob_show_id` / 玩家 `player_show_id` ✓
    #   用户："怪物框上面…" 之后接着问"玩家的检出框是否用了类似算法"⇒ 给玩家也补上 ✓）
    for _k in ("mob_show_id", "player_show_id"):
        check(theme.VIS_DEFAULTS.get(_k) is False,
              "`%s` 默认不是 False（= 偷改观感 ✗）：%r"
              % (_k, theme.VIS_DEFAULTS.get(_k)))
        check(_k not in theme._VIS_ON_KEYS,
              "`%s` 被塞进 `_VIS_ON_KEYS` ✗（那批是「缺省 ⇒ True」⇒ 所有人的框上都多字 ✗）"
              % _k)
        _bak = theme.load_vis().get(_k)
        try:
            theme.save_vis({_k: True})
            _got = theme.load_vis().get(_k)
            theme.save_vis({_k: "yes"})                 # 坏值
            _bad = theme.load_vis().get(_k)
        finally:
            theme.save_vis({_k: bool(_bak)})
        check(_got is True, "`%s` 勾上存了却读不回来（改了没用 ✗）：%r" % (_k, _got))
        check(_bad is False, "`%s` 坏值该回 `False`（老观感 ✓）：%r" % (_k, _bad))

    _lt = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    for _k in ("mob_show_id", "player_show_id"):
        check('_vis_cfg.get("%s", False)' % _k in _lt,
              "画角标那段没读 `%s`（勾了不生效 ✗）" % _k)
    check("%.2f/%.2f" in _lt and "peak_conf" in _lt,
          "怪角标里没有「这一拍 / 峰值」两个置信度（那就分不清是哪种闪 ✗）")
    check('"%s %.2f" % (_m_label, m.conf)' in _lt,
          "关着时的老文案被改掉了（老观感该一字不变 ✗）")
    # ⚠⚠ **幽灵标记必须在最前面** ✗（2026-10-11 用户当场问："为什么…后面一个带*"✗
    #   —— 那时它贴在第二个置信度后面、看着像在修饰它 ✓）⇒ 钉住这个排版约定 ✓
    check('_mtxt = "%s#%s %s %.2f/%.2f"' in _lt,
          "怪角标的幽灵标记没放在**最前面**（又会被读成「第二个数带星号」✗）")
    check("last_missed" in _lt and "last_low" in _lt,
          "玩家角标没读「外推 / 低分续命」那两个只读镜像（看不出来这拍可不可信 ✗）")


def t_track_events_wired():
    """⭐⭐ **"闪 / 抢框"的现场要真写进 `behavior.log`**（用户 2026-10-11 ✓）。

    用户原话："**同个怪物的检出框疯狂闪烁**"✓ / "**角色使用传送点后被其他玩家抢检出框**"✓
    —— 这两件事以前**只能靠肉眼看画面** ✗（判不出是"同一个 id 在闪"还是"每帧换 id" ✓，
    也看不出"到底是谁抢走了框"✓）⇒ 现在两条链各留一份现场 ✓：
      · `MobTracker.drain_flick()` → `mob_flick`（id / 存活拍数 / peak / 漏检 ✓）；
      · `PlayerTracker.drain_events()` → `player_<what>`（旧框→新框 / 竞争者 / 残差 / ΔW ✓）。

    钉四件（**两半**：真能取到 ✓ ＋ 真接线了 ✓ —— 只钉一半都会"永远什么都不写"✗）：
      ① 两个 `drain_*` 在真对象上存在且**取走即清空** ✓；
      ② 实时回路真调了它们 ✓（源码钉）；
      ③ 真往 `behavior.log` 写了（源码钉：事件名 `mob_flick` / `player_` 前缀 ✓）；
      ④ 那条打点**不许**把回路带崩（整个块包在 `try/except` 里 ✓ 源码钉）。
    """
    import perception.track.tracker as trk

    t = trk.MobTracker()
    check(hasattr(t, "drain_flick") and t.drain_flick() == [],
          "`MobTracker` 没有 `drain_flick()`（那现场就永远写不出去 ✗）")
    p = trk.PlayerTracker()
    check(hasattr(p, "drain_events") and p.drain_events() == [],
          "`PlayerTracker` 没有 `drain_events()`（抢框现场写不出去 ✗）")

    src = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    for key in ("mob_tracker.drain_flick()", "player_tracker.drain_events()"):
        check(key in src, "实时回路没调 `%s`（有现场也没人取 ⇒ 等于没做 ✗）" % key)
    check('"mob_flick"' in src, "没往 behavior.log 写 `mob_flick`（那这条记事本就没内容 ✗）")
    check('"player_" + str(' in src, "没写 `player_*` 事件（换锁/隔离/传送都没留痕 ✗）")
    # ⭐ 「认过的怪刚丢」也要留现场（用户 2026-10-17 ✓「离得近时开始频繁丢失」✗）——
    #   两半都钉：取得到 ✓ ＋ 真接线写进 behavior.log ✓（只钉一半 = 永远什么都不写 ✗）
    check(hasattr(t, "drain_lost") and t.drain_lost() == [],
          "`MobTracker` 没有 `drain_lost()`（丢框现场写不出去 ⇒ 分不清是不是被挡住 ✗）")
    check("mob_tracker.drain_lost()" in src and '"mob_lost"' in src,
          "实时回路没把「刚丢的确认轨迹」写进 `behavior.log`（`mob_lost` ✗）")
    check("ov=_ov" in src and "dist=_dist" in src,
          "`mob_lost` 没带「与玩家框的重叠 / 距离」（那就分不出是**遮挡**还是**模型漏检** ✗）")
    # ④ 打点块必须被 try/except 包住（`event()` 出事不许把实时回路带崩 ✗）
    #   ⚠ 窗口别抠太紧 ✗ —— 这段还在长（2026-10-17 加了 `mob_lost` 那一截 ✓
    #     900 字符就不够了 ⇒ 假红 ✓）。
    _i = src.find("mob_tracker.drain_flick()")
    _blk = src[max(0, _i - 600):_i + 2600]
    check("except Exception" in _blk, "打点那块没包 `except`（读日志的代码不许拖垮回路 ✗）")


def t_lie_rec_codec_h264():
    """⭐⭐⭐ **测谎录屏必须优先写 H.264**（用户 2026-10-17 ✓ 原话：
    「**最近一次测谎录屏全是花屏**」✗）。

    ⚠ 先说明"花屏"是哪来的（**不许再归错因** ✗）：
      · 老写法固定 `mp4v`（MPEG-4 Part 2 ✓ 文件里四字码 `FMP4`/`mp4v`）；
      · **文件本身是好的** —— 实测把 `lie_20261017_174814.mp4` 全部 718 帧解出来，
        帧 0 / 400 都是清晰的游戏画面 ✓；
      · 但 **Windows 自带播放器大多不带 MPEG-4 Part 2 解码器** ⇒ 一放就是花屏 ✓；
      · ⇒ 换成系统 **Media Foundation 的 H.264**（`CAP_MSMF` + `H264` ⇒ 容器里是 `avc1` ✓），
        所有播放器都认 ✓。
    钉三件：
      ① 源码里**先试 MSMF/H264**（顺序钉死：先 h264 再 mp4v ✓ 反了就白改 ✗）；
      ② 真写一个小文件、真回读 ⇒ 帧数对得上（不是"开得起来就算" ✗）；
      ③ 名字里必须有 `mp4v` 这条**退回链**（老机器没有 MF 编码器时不能整段录不了 ✗）。
    """
    from gui.live_thread import LiveThread

    check(hasattr(LiveThread, "_rec_writer"),
          "没有 `_rec_writer`（录屏编码还写死在 `mp4v` ⇒ Windows 自带播放器花屏 ✗）")
    src = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    _i_h = src.find("cv2.CAP_MSMF")
    _i_m = src.find('"mp4v"', _i_h + 1)
    check(_i_h > 0 and _i_m > _i_h,
          "必须**先试** `CAP_MSMF`+`H264`、再退回 `mp4v`（顺序反了等于没改 ✗）")

    # ⭐⭐ **录像文件名必须独占**（用户 2026-10-17 ✓ 现场 `lie_20261017_162813.mp4`：
    #   前 2 秒远程画面、后面上半截是另一路 ✓ 解码器报 `ac-tex damaged` ✓）
    #   —— 老名字只到**秒** ⇒ 同一秒开两段录制 ⇒ **两个 writer 写一个文件** ⇒ 画面交织 ✓。
    #   判据：连着占两次名 ⇒ **必须不同**，且两个文件都真被建出来了 ✓（"xb" 独占 ✓）。
    _rp_dir = ROOT / "data" / "debug"
    _rp_dir.mkdir(parents=True, exist_ok=True)
    _p1 = LiveThread._rec_path(_rp_dir, "_selftest_recname")
    _p2 = LiveThread._rec_path(_rp_dir, "_selftest_recname")
    try:
        check(_p1 != _p2,
              "同一秒连开两段录制会**撞同一个文件名** ⇒ 两路写一个文件 ⇒ 画面交织/花屏 ✗：%r"
              % (_p1,))
        check(Path(_p1).exists() and Path(_p2).exists(),
              "占名后文件没真建出来（那就没占住 ⇒ 还是会撞 ✗）")
        # ⭐⭐ **名字要带来源标签**（用户 2026-10-17 ✓「**没有看到A机的录屏**」——
        #     两份其实都在 ✓ 但名字一样、还都在同一个目录 ⇒ 分不清哪份是 A 机的 ✗）
        _p3 = LiveThread._rec_path(_rp_dir, "_selftest_recname", "remote_stream")
        try:
            check("remote_stream" in Path(_p3).name,
                  "录像名里没有来源标签（`remote_stream`/`local_window`）⇒ 两份录像分不清 ✗：%r"
                  % (Path(_p3).name,))
            check("_selftest_recname" in Path(_p3).name,
                  "加了标签却把种类名（`lie_`/`disc_`）冲掉了 ✗：%r" % (Path(_p3).name,))
        finally:
            try:
                Path(_p3).unlink()
            except Exception:                # noqa: BLE001
                pass
    finally:
        for _p in (_p1, _p2):
            try:
                Path(_p).unlink()
            except Exception:                    # noqa: BLE001
                pass

    import numpy as np
    import cv2
    _d = ROOT / "data" / "debug"
    _d.mkdir(parents=True, exist_ok=True)
    _p = str(_d / "_selftest_rec_codec.mp4")
    try:
        _w, _codec = LiveThread._rec_writer(_p, 30.0, (320, 240))
        check(_w.isOpened(), "录像容器打不开（%s）" % (_codec,))
        for _i in range(12):
            _w.write(np.full((240, 320, 3), 20 + _i * 10, np.uint8))
        _w.release()
        _cap = cv2.VideoCapture(_p)
        _n = 0
        while True:
            _ok, _f = _cap.read()
            if not _ok:
                break
            _n += 1
        _cap.release()
        check(_n == 12, "写进去 12 帧、回读只有 %d 帧（容器/编码不对 ✗）" % (_n,))
        check(_codec in ("h264", "mp4v"), "编码名不认识：%r" % (_codec,))
    finally:
        try:
            import os
            os.remove(_p)
        except Exception:                        # noqa: BLE001
            pass


def t_lie_rec_src_black_logged():
    """⭐⭐ **录制期间\"源画面全黑\"必须留一句日志**（用户 2026-10-17 ✓ 现场：
    「刚才AB机同时测谎，没有看到A机的录屏」✗ —— 其实**两份都录了** ✓，
    只是每份**结尾都有一段全黑** ✓，而日志里**一个字都没有** ✗ ⇒ 只能靠猜 ✓）。

    ⚠ 要区分的两种事故 ✗（别再归错因）：
      · **录制器停了** ⇒ 文件里没有帧（`lie_rec done frames=` 会明显偏少 ✓）；
      · **源断了**（本次现场 ✓）⇒ 帧照写、叠加层照画、**画面全黑** ✓
        = 抓屏/推流断了（窗口最小化 / 切桌面 / 游戏关了 ✓）。

    钉四条：
      ① 连续全黑够 `LIE_REC_BLACK_S` 秒 ⇒ **报一次** `report` ✓；
      ② 同一段里**只报一次**（不然 30fps 会把日志刷爆 ✗）；
      ③ 画面恢复亮 ⇒ 状态复位 ⇒ **下一段还能再报** ✓（一次录制断三次就该有三条 ✓）；
      ④ **暗场景不许误报**：均值 30/60（地牢、夜晚）必须一声不响 ✓。
    """
    from gui.live_thread import LiveThread as L

    check(hasattr(L, "_src_black_step"),
          "没有 `_src_black_step` ⇒ 源画面全黑时**没有任何记录**（用户只能猜 ✗）")
    check(float(L.LIE_REC_BLACK_MEAN) <= 20.0,
          "纯黑阈值定得太高（%r）⇒ 会把地牢/夜晚这种暗场景误报成断流 ✗"
          % (L.LIE_REC_BLACK_MEAN,))

    st = {"t0": 0.0, "logged": False}
    res = []
    t = 100.0
    for _ in range(90):                       # 3 秒纯黑（30fps）
        t += 1.0 / 30.0
        res.append(L._src_black_step(st, t, 2.0))
    check(res.count("report") == 1,
          "3 秒纯黑里报了 %d 次（该**只报一次** ✓ 每帧一条会刷爆日志 ✗）" % res.count("report"))

    check(L._src_black_step(st, t + 0.1, 120.0) is None, "亮帧不该有事件")
    res2 = []
    for _ in range(90):
        t += 1.0 / 30.0
        res2.append(L._src_black_step(st, t, 1.5))
    check(res2.count("report") == 1,
          "恢复之后**第二段**全黑没再报（一段录制里断两次就该有两条 ✓）：%r"
          % (res2.count("report"),))

    st3 = {"t0": 0.0, "logged": False}
    dark = [L._src_black_step(st3, 10.0 + i / 30.0, 30.0) for i in range(90)]
    dark += [L._src_black_step(st3, 20.0 + i / 30.0, 60.0) for i in range(60)]
    check(not any(dark),
          "**暗场景（均值 30/60）被误报成断流** ✗（那用户会以为推流老掉 ✓）")

    # 源码钉：这条日志必须真从写入路径发出去（只写个函数没人调 = 白写 ✗）
    src = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    _i = src.find("def _lie_rec_write")
    _seg = src[_i:_i + 4000]
    check("_src_black_step" in _seg and "lie_rec_src_black" in _seg,
          "`_lie_rec_write` 里没接\"源画面全黑\"的记账（写了函数却没调用 ✗）")


def t_vt_fps_limit():
    """⭐⭐⭐ 「帧率限制(fps)」+ **-1 = 不限制**（用户 2026-10-18 ✓ 原话：

      「把 `vt_max_fps` 做成配置"帧率限制"加在**自动测谎组**里；**配 -1 则不限制**」）。

    为什么要有这格：帧率上限原来是**只有 `live.yaml` 里一行**的隐形参数 ✓ ——
      人在界面上看不到、也没法按机器调 ✗（弱机想降到 10、强机想拉到 60 / 不限 ✓）。

    钉六条（缺一条就是"配了不生效"或"把老功能带走" ✗）：
      ① 控件在「自动测谎（视觉追踪）」组里，范围含 **-1**；
      ② `-1` 显示成「不限制」（不用让人记"这个 -1 是什么意思"✓）；
      ③ 写回 `live.yaml` 的键是 `vt_max_fps`，且**不许把 -1 折成默认值**（折了 = 配了没生效 ✗）；
      ④ 实时线程热读时负数**原样保留**；0 / 0.3 这种笔误值抬到 1fps（别让人看着像卡死 ✗）；
      ⑤ 喂帧那一拍：`-1` ⇒ **既不按间隔丢帧、也不做积压降速**（人说不限制就别偷偷降 ✓）；
      ⑥ 有限制时那两道老闸（上限 + `VT_FPS_UNDER_LAG` 让路）**照旧在** ✓（别为了 -1 删掉 ✓）。
    """
    lt = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    pp = (ROOT / "gui" / "player_panel.py").read_text(encoding="utf-8")

    check("self.sp_vt_fps = NoWheelSpinBox()" in pp and "setRange(-1, 120)" in pp,
          "「自动测谎」组里没有「帧率限制」这一格（范围要含 **-1** ✗）")
    check('setSpecialValueText("不限制")' in pp,
          "`-1` 没显示成「不限制」⇒ 人得自己去记 -1 是什么意思 ✗")
    check('f.addRow("帧率限制(fps)"' in pp,
          "这一格没挂进表单 ⇒ 界面上看不见 ✗")
    check("vt_max_fps=int(self.sp_vt_fps.value())" in pp,
          "控件没写回 `vt_max_fps` ⇒ 改完不生效 ✗")
    check("-1.0 if _fps < 0 else max(1.0, _fps)" in lt,
          "热读把 `-1` 折掉了（该原样存负值 ⇒ 才叫「不限制」✗）")
    check("_unlimited = _want < 0" in lt
          and "(not _unlimited) and (_nowm - self._vt_fed_at)" in lt,
          "喂帧那一拍没认 `-1` ⇒ 界面上写了「不限制」其实还在限速 ✗")
    check("if not _unlimited:" in lt and "VT_FPS_UNDER_LAG" in lt,
          "有限制时那两道老闸（上限 + 积压让路）被删了 ✗（别为了 -1 把老功能带走）")


def t_canvas_marker_visible():
    """⭐⭐⭐ 地形图上的「准心」**必须真的画出来**（用户 2026-10-19："路线识别页签→地形图里看不到准心了"✗）。

    病根（2026-10-18 我加的）：把标记组标成 `ItemIgnoresTransformations`（想让尺寸不随缩放变小 ✓），
    实测**整组不渲染** —— 无头渲染同一位置**黄色像素 0** 个 ✗（组的 `sceneBoundingRect` 还落到
    (-24,-28) 这种错地方 ✓）。⇒ 改回"**画在场景坐标里、按视图缩放折算大小**"（`r / s`）+
    缩放/尺寸变化时重画（`_marks_follow_zoom` ✓）。

    钉三条：① 圈/十字**真的有像素画出来**（按颜色数）② 缩放一倍后**屏幕尺寸不变**
    （这就是"看不清"要的那件事）③ 收起来（空列表）时**真的清干净**（不留残影 ✓）。
    """
    from PyQt5.QtGui import QImage, QPixmap
    from gui.canvas import ImageCanvas

    cv = ImageCanvas()
    cv.resize(400, 300)
    img = QImage(300, 400, QImage.Format_RGB32)
    img.fill(0x202020)
    cv.load(QPixmap.fromImage(img), editable=False, fit=True)
    # ⚠ **必须 show 再画**：没显示的控件还没走布局/变换 ⇒ `transform()` 是单位阵
    #   ⇒ 按 `r / s` 折算出来的半径是错的（实测：39x25 而不是 50x50 ✗ —— 那是量具的错，不是绘制的错）
    cv.show()
    _QApp.processEvents()                 # ⚠ 离屏环境下"显示/布局"是异步的：
    cv.fit()                              #   不跑一圈事件循环，`transform()` 还是旧的 ✗
    _QApp.processEvents()

    def yellow():
        cv.viewport().repaint()
        pim = cv.grab().toImage()
        xs, ys, n = [], [], 0
        for yy in range(pim.height()):
            for xx in range(pim.width()):
                c = pim.pixelColor(xx, yy)
                if c.red() > 180 and c.green() > 150 and c.blue() < 120:
                    n += 1
                    xs.append(xx)
                    ys.append(yy)
        return (n, (max(xs) - min(xs) + 1, max(ys) - min(ys) + 1)) if n else (0, (0, 0))

    mx, my = 150.0, 200.0
    cv.set_markers([(mx, my, "玩家 世界(0, 0)", "#ffd400", 14.0)])
    check(bool(cv._mark_items), "`set_markers` 一个图元都没建 ✗")
    for g in cv._mark_items:
        # ⚠ **不许再用 `ItemIgnoresTransformations`**（2026-10-19 现场 ✓）：它对
        #   **组**不成立 —— 无头渲染同一位置**一个像素都不画** ✗（这就是"看不到准心"）
        check(not (g.flags() & g.GraphicsItemFlag.ItemIgnoresTransformations),
              "标记组又标成了 `ItemIgnoresTransformations` ⇒ 整组不渲染、准心看不见 ✗")
    n1, (w1, h1) = yellow()
    check(n1 > 50,
          "地形图上一个准心像素都没画出来（%d 个）⇒ 用户看到的就是「看不到准心」✗" % n1)

    # ⭐⭐⭐ **位置也要对**（2026-10-19 用户贴图：准心画到了**画面左上角**✗ ——
    #   根因是 `grp.setPos()` 写在 `addToGroup()` **之前**：`addToGroup` 会把子件挪到
    #   "场景位置保持不变"的地方 ⇒ 等于反向平移，整组落回场景原点附近 ✓）
    from PyQt5.QtCore import QPointF
    want = cv.mapFromScene(QPointF(mx, my))
    _im = cv.grab().toImage()
    xs, ys = [], []
    for yy in range(_im.height()):
        for xx in range(_im.width()):
            c = _im.pixelColor(xx, yy)
            if c.red() > 180 and c.green() > 150 and c.blue() < 120:
                xs.append(xx)
                ys.append(yy)
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    check(abs(cx - want.x()) <= 6 and abs(cy - want.y()) <= 6,
          "准心画的位置不对：该在屏幕 (%d,%d)，实际中心 (%d,%d) "
          "⇒ 用户看到的就是「准心跑到画面左上角」✗"
          % (want.x(), want.y(), cx, cy))

    cv.scale(0.5, 0.5)
    cv._marks_follow_zoom()
    n2, (w2, h2) = yellow()
    check(n2 > 50, "缩放之后准心消失了（%d 个像素）✗" % n2)
    # ⚠ 只比**宽**：高度容易被视口边界切（那是量具的事，不是绘制的错 ✗）
    check(abs(w2 - w1) <= 4,
          "准心的**屏幕宽度**跟着缩放变了（%d → %d）⇒ 缩小时就又看不见了 ✗" % (w1, w2))

    cv.set_markers([])
    n3, _ = yellow()
    check(n3 == 0, "收起来之后还留着 %d 个黄色像素 ⇒ 残影 ✗" % n3)


TESTS = (
    ("⭐⭐⭐ 地形图上的准心**真的画出来**、且缩放后屏幕尺寸不变（用户 2026-10-19：看不到准心）",
     t_canvas_marker_visible),
    ("⭐⭐ 录制期间源画面全黑必须留日志（用户 2026-10-17「没看到A机的录屏」其实是结尾黑了一段）："
     "每段只报一次 · 恢复后能再报 · 暗场景不许误报", t_lie_rec_src_black_logged),
    ("⭐⭐⭐ 测谎录屏优先写 H.264（用户 2026-10-17「录屏全是花屏」）：先 MSMF/H264 再退回 mp4v · "
     "真写进去真回读 · 退回链不许删", t_lie_rec_codec_h264),
    ("⭐⭐ 怪框角标 `#id conf/peak`（用户 2026-10-11）：默认关 · 不混进「缺省⇒True」名单 · "
     "存得进读得回 · 关着时老文案一字不变（用来分清「同一个 id 在闪」还是「每帧换 id」）",
     t_mob_id_label_optional),
    ("⭐⭐ 「闪 / 抢框」的现场真写进 behavior.log（两半都钉：取得到 + 真接线 + 包了 except）",
     t_track_events_wired),
    ("⭐⭐ 怪 / 玩家框角标（`#12 …` / `#我 …`，外推 `*`、低分 `~` **都在最前面**）："
     "默认关 · 不混进「缺省⇒True」名单 · 存得进读得回",
     t_mob_id_label_optional),
    ("⭐⭐⭐⭐⭐ 「地形图的玩家圈只用唯一那一份世界坐标」（用户 2026-10-10）：不许再用「差分自己"
     "挑的点」当玩家（会锁到墙的红点上 · 与信息栏两个数）· 换算只走地形 · 换图/没跑/没定上位不画",
     t_live_map_marker_is_authoritative),
    ("⭐⭐⭐⭐⭐ 「测谎成功后必须收工」（用户 2026-10-10「录了很久没在成功后切断」）：按时间的"
     "收工判据必须在早退**之前**——状态不再变的那一拍也要到点收 · GAP 兜底同理 · 源码级钉住顺序",
     t_lie_rec_stops_after_success),
    ("⭐⭐⭐⭐ 「界面接管期间这一拍不定位」（用户 2026-10-10「定位有这么慢吗」）：实测 0.4ms vs"
     "接管时 43~73ms（面板被弹窗盖住 ⇒ 每拍退回全画面重找）⇒ 接管时直接不定位",
     t_locate_skipped_while_taken_over),
    ("⭐⭐⭐⭐⭐ 「测谎开始后 Agent 不许再按方向键」（用户 2026-10-10）：小游戏那段画面匹配不上"
     "也不许把操作权还回去（留住接管）· 全程接管到成功弹窗 · 测谎完自动还权 · "
     "没 warn 的 lie_game / warn 误判都不许冻住 Agent",
     t_lie_takeover_holds_through_minigame),
    ("⭐⭐⭐ 「一帧的上下文」`FrameCtx` 三条纪律（口径 B 立盒子 ✓ 用户 2026-10-15）：字段写死"
     "（没声明的字段当场报错 ✗ 挡「退化成谁都能塞的袋子」）· 一帧一个（不复用 ⇒ 上一帧的值漏不进来）"
     "· 跨帧的东西一律不进（帧计数/采样窗口/配置留下来 ✓ 判据「下一帧还会用吗？」）",
     t_frame_ctx_discipline),
    ("⭐⭐ 「重载保命闸」的**本体**（`runtime.pipeline.HeavyGate`）：0=不限制 · 到点前跳/到点后放 · "
     "跳过必喂时间拍且只喂那一次 · 没推理不拦不喂 · 时间拍炸了不炸回路 · 打点齐全",
     t_heavy_gate_unit),
    ("⭐⭐ 「重载保命闸」：每秒最多跑 N 拍重活（用户 2026-10-10「挂游戏+浏览器+音乐+steam+大型游戏"
     "也要流畅」）—— 到点才跑 · 跳过的帧只喂时间拍（CD/delay 精度不受影响）· 默认 0 = 不限制",
     t_heavy_gate_when_machine_busy),
    ("⭐⭐ 「解出来没人要 ⇒ 不做颜色转换」（用户 2026-10-10「重点是提升 B 机的效率，A 就算推"
     "240fps 也要能应对」）：实测解码 0.71ms vs 转换 5.93ms（贵 8 倍）⇒ 转换成本按消费速度封顶",
     t_convert_only_when_wanted),
    ("⭐⭐⭐ 只收流（没开推理）时画框段不许崩：调用点用到的名字都要在块前有值（用户 2026-10-05）",
     t_draw_args_defined_when_stream_only),
    ("⭐⭐ 「目标被攻击CD」的灰框 + 右下角剩余 CD（用户 2026-10-06）", t_mob_cd_gray_box),
    ("⭐⭐⭐ 「检测框颜色」每一类前面的勾选框（默认全勾 + 勾选后实时显示，用户 2026-10-06）"
     "—— 掉落/宠物/NPC/其他玩家终于画得出来了", t_class_box_switches),
    ("⭐⭐ 「输出行为CD」没走完 ⇒ 攻击范围框画**虚线**（用户 2026-10-06）", t_out_cd_dashed_attack_box),
    ("⭐⭐ 「自动拾取」的判据：玩家框与掉落物框**相交**（用户 2026-10-06）", t_drop_touch),
    ("⭐ 血条读出 0 要采信（连几拍才算「人死了」）——「HP 不变 0%」的修复",
     t_pot_zero_means_dead),
    ("⭐⭐ 测谎现场录屏：进 lie_* 起录 / 离开即停、录原生帧副本（不污染 raw）",
     t_lie_recording),
    ("⭐⭐ 自动测谎（visual_tracking）：喂画框前的 raw / 独立线程 / 成功点「确定」走 "
     "mouse_aim 且只在本机窗口 / 点完重开一轮 / 开关与状态行接通（行为见 selftest_vt_sdk）",
     t_vt_lie_sdk),
    ("⭐⭐⭐ 「帧率限制(fps)」+ -1 = 不限制（组里有这格 · 热读不折 · 不偷偷降速 ✓）",
     t_vt_fps_limit),
    ("⭐⭐ 自动测谎那组控件**真的会写配置**：构造期不写 · 勾开关五键一起落 · 冷却值进得去 · "
     "地区下拉只能是 CN/TW（挂机保护页 ✓）", t_vt_group_writes_live),
    ("⭐⭐ 自动测谎「瞄准」是**增量**的：第一发建原点 · 第二发只发差值（治「左上角⇄目标来回抽」）· "
     "久停之后肯重建原点", t_vt_aim_is_smooth),
    ("⭐⭐ 采集：选中文件后显示预估张数（公式与 extract_frames 一份 + MKV 容器时长兜底）",
     t_capture_est_frames),
    ("⭐⭐ 本地窗口被别的窗口盖住 ⇒ 改抓整屏的那条**接线**（用户 2026-10-10）",
     t_covered_note_wired),
    ("⭐⭐⭐ `_run` 共用尾段只许读「分岔之前就赋过值」的名字（治 UnboundLocalError）",
     t_run_shared_names_defined_before_split),
    ("⭐⭐⭐ 「对准窗口」自动框选：取客户区 · 两拍确认 · 记住那个窗口 · 超时不乱框（用户 2026-10-10）",
     t_pick_rect_auto),
    ("⭐⭐ 开自动前体检：标定/地形没凑齐要先弹窗说清（用户 2026-09-29；⚠ 光有函数没接上=没有）",
     t_auto_precheck),
    ("⭐⭐⭐ 体检的来源要与路线识别页同一处（按地图 id ✓）：本图量过就不许报、"
     "另一条量过要点名（用户 2026-10-10「我标定过了还显示这个」）",
     t_precheck_source_per_map),
    ("⭐⭐⭐ 断线重连回到游戏 ⇒ 体检通过才恢复自动（用户 2026-10-07 选「体检」；"
     "不过就不恢复 + 写清原因）", t_reconnect_resume_gate),
    ("连推 10 帧只画最新那帧（合并，不排队）", t_coalesce),
    ("不可见时一帧都不画，但帧仍是最新的", t_hidden_skips_draw),
    ("视野框（虚线）：每帧现算，不许再套「每 N 秒算一次」的节流", 
     t_vision_box_updates_every_frame),
    ("只检测视野框：关=原图 / 开=剪裁块+偏移 / 退化框退回原图 / 偏移只加一处（用户 2026-10-15）",
     t_det_vision_only_crop),
    ("辅助线与标记：颜色支持透明度（#AARRGGBB）+ 每项一个开关 + 四个「框」项"
     "（含追击起跳框）+ 三个框并排的画法", t_vis_color_alpha_and_toggles),
    ("current_frame() 必须是原生帧：显示帧（框线/视野虚线）不许进那条路",
     t_current_frame_is_raw),
    ("收流线程：先留原生帧再就地画，且只有真要画时才拷（收画面阶段零开销）",
     t_raw_frame_before_draw),
    ("状态行露出「绘制 ms / 合并丢弃」（源码约定）", t_stats_show_draw),
    ("负载告警顶在状态行最前面，恢复后自己消失", t_load_warn_on_status),
    ("积压锁死看门狗：持续高才报、回落自己消失、迟滞不闪", t_lag_watchdog_rules),
    ("积压锁死顶在状态行最前面（压过负载告警）", t_lag_warn_on_status),
    ("小地图叠图：只画显示层、帧数据不许改", t_minimap_overlay_is_display_only),
    ("小地图叠图：源超出叠加图时画出来的范围要对", t_overlay_source_out_of_image),
    ("探针单调性闸：几何可疑不报延迟、保存前拦错几何", t_probe_mono_gate),
    ("实时画面上的采样框：颜色跟判据、只画在副本上", t_probe_box_overlay),
    ("「卡在谁身上」：输入受限 / 本机受限 / 说不清（纯函数）", t_limit_reason_rules),
    ("⭐⭐ 持续跟不上源 ⇒ 自动降显示刷新（判据/接线/面板三件；用户 2026-10-05）",
     t_src_lag_degrades_display),
    ("⭐⭐ 「imgsz 不生效」弹窗：点确定 ⇒ 直接后台开导；点取消 ⇒ 不起（用户 2026-10-05）",
     t_imgsz_mismatch_auto_exports),
    ("显示链路：带 padding 的帧画得对，且不再白拷一整幅", t_pixmap_handles_padded_frame),
    ("⭐⭐ 引擎 imgsz：随便填不再把实时开不起来（按引擎尺寸跑 + 如实说清）",
     t_engine_imgsz_alignment),
    ("⭐ 小地图定位失败的原因要进日志（只在界面说 ⇒ 事后查不出来）",
     t_mmap_fail_reason_is_logged),
    ("⭐⭐ 推图 id 跟着项目走 + 状态常显（用户 2026-10-04：改了图 A 却收到旧的）",
     t_mmap_map_id_follows_project),
    ("⭐ `mmap_note` 恢复时要写一句 ok（注记整个会话不清 ⇒ 不然一直挂着失败那句骗人）",
     t_mmap_note_says_ok_on_recovery),
    ("⭐⭐⭐ 实时页「权重」能自己选（候选同训练卡片口径 + 按项目记住 + 「自动」可退回）",
     t_live_weights_selectable),
    ("⭐⭐⭐ 实时页那一排参数按项目存（来源/框选区域/conf… ＋ 回填只读 ＋ 老项目不变）",
     t_live_params_follow_project),
    ("⭐⭐⭐ 导出成功后**当场接管新引擎**（旧 `<名字>.engine` 会被改名走 ⇒ 不许再捏着它 ✗）",
     t_export_done_takes_over),
    ("⭐⭐ 权重路径失效时要说清「怎么办」（别只甩一行 does not exist）",
     t_missing_weights_msg),
    ("⭐⭐⭐ 删掉的键要真的落盘（「重开 gui 后删掉的自定义按键又出现」的真根）",
     t_project_save_keeps_deletions),
    ("⭐⭐ 每帧一条 ultralytics 弃用告警要静音（stdout.log 被写成 406MB ✗，但别的告警不许吞）",
     t_half_warn_not_spamming),
    ("⭐⭐ 一键导出该尺寸引擎（从 .engine 推同名 .pt / 名字带尺寸 / 后台线程 / 有反馈）",
     t_engine_export_one_click),
    ("静态检查：会当场炸的名字错误（pyflakes）", t_static_check_no_crash_classes),
    ("左下按键帽：按住=半透明绿填充、文案随键位映射自适应（2026-09-30 用户要求）",
     t_key_caps_overlay),
    ("怪框那行地点：「查过的怪框」写在框下方靠左；锁定框不再写地点（绘制层不许自己扫 foothold）",
     t_mob_box_labels),
)




def t_infer_conf_feeds_low_band():
    """⭐⭐⭐⭐ **推理门槛必须低到「低分档」那一层**（用户 2026-10-13 ✓ 原话："**检出框疯狂闪烁，
    防抖时间不生效**"✓）。

    病根（一处对不上的口径 ✓ 害了整整两轮排查 ✗）：`LiveThread._run` 里推理用的门槛原来是
      `min(conf_mob, conf_player)`（= 0.25 ✓）✗，而 `split_dets` 的**低分档**门槛是
      `conf_mob × MOB_LOW_CONF_RATIO`（0.25 × 0.5 = **0.125** ✓）⇒ **模型 0.25 以下直接丢掉** ✗
      ⇒ 低分档**永远收不到框** ✗ ⇒ BYTE 那条「低分只续命」的链**从来没生效过** ✓
      ⇒ 框一掉到阈值以下就只能靠**防抖幽灵**撑着（`debounce_ms` ✓）⇒ 撑完就死、
      再出现就是新 id ✓ = 用户看到的"疯狂闪烁 ＋ 防抖时间不生效" ✓✓。
    ⚠ 上面那两行注释本来就写着"保证两类的**低分框**都进结果" ✗ —— 实现漏了那个比例 ✓。

    钉两件（源码级 ✓ 真跑一次推理不现实 ✗）：
      ① 门槛取**两档里更低的那个**（`* MOB_LOW_CONF_RATIO` / `* PLAYER_LOW_CONF_RATIO` ✓）；
      ② 分档那份口径（`split_dets` ✓）**照旧**：它只负责分档 ✓ 判据不写第二遍 ✓。
    """
    _p = Path(__file__).resolve().parents[1] / "gui" / "live_thread.py"
    src = _p.read_text(encoding="utf-8")
    check("conf = min(conf_mob * MOB_LOW_CONF_RATIO" in src,
          "推理门槛又退回 `min(conf_mob, conf_player)` 了 ✗ —— 那会让**低分档永远空** ⇒ "
          "「防抖时间」对「擦着阈值」的怪彻底不生效 ✓（用户 2026-10-13 的现场 ✓）")
    check("PLAYER_LOW_CONF_RATIO)" in src or "PLAYER_LOW_CONF_RATIO)" in src.replace(" ", ""),
          "玩家那一档的门槛没跟着降 ✗ —— 玩家低分框同样喂不进 tracker ✓")
    # ⚠⚠⚠ **两处都要**（2026-10-13 实测踩到 ✓ 这一条比上面那条还值钱 ✗）：
    #   `_run` 开头那处只算"建会话时"的；主循环里还有一格**热重载**
    #   （读 `config/live.yaml` 重算 `conf` ✓）⇒ 只改一处的话，**1 秒内就被盖回去** ✗
    #   （现场表现 = 改完看不出任何变化 ✓ 与"防抖时间不生效"同源 ✓）
    #   ⇒ 判据 = 两处都带比例 + **旧的 `min(conf_mob, conf_player)` 一处都不许剩** ✓。
    check(src.count("min(conf_mob * MOB_LOW_CONF_RATIO") == 2,
          "带「低分档比例」的推理门槛该有**两处**（建会话 + 热重载 ✓），实得 %d 处 ✗ —— "
          "少一处 ⇒ 那一处会把口径盖回去 ✓（用户 2026-10-13 现场 ✓）"
          % (src.count("min(conf_mob * MOB_LOW_CONF_RATIO"),))
    check("conf = min(conf_mob, conf_player)" not in src,
          "还留着旧的 `conf = min(conf_mob, conf_player)` ✗ —— 它会让低分档永远空 ✓")


def t_max_det_minus_one_is_unlimited():
    """⭐⭐ **怪框上限 `-1` = 不限**（用户 2026-10-17 ✓ 原话：「**怪框上限可以做成默认 -1
    无限吗？**」✓）。

    钉四件（缺一件这功能就不成立 ✓）：
      ① 界面那格的**下限必须是 -1** ✗ —— 不然用户**填不进 -1**（会被夹回 10 ⇒ 填了没效果 ✓）；
      ② 引擎里 **两处读数都要把 -1 翻译成大上限** ✗ —— 只翻译一处 ⇒ 另一处（热重载 ✓）
         会把**负数原样**塞给 Ultralytics ⇒ **当场报错** ✓；
      ③ **缺键的兜底是 -1** ✓（这就是用户要的「默认不限」✓）；
      ④ 那格的 **tooltip 必须写明「-1 = 不限」** ✓（不然没人知道能填负数 ✓）。
    """
    import io
    import os

    # ⚠ `_panel()` 返回的是**元组** ✓ `(app, lp, p, made, orig)`（实测上报
    #   `'tuple' object has no attribute ...' ✓ 就是这个坑 ✓）⇒ 解包取那个面板 `p` ✓。
    _app, _lp, p, _made, _orig = _panel()
    _lp._bgr_to_pixmap = _orig       # 本用例用不到那支探针 ⇒ 立刻还原 ✓（别漏给后面的用例 ✓）
    check(p.sp_max_det.minimum() == -1,
          "怪框上限的下限不是 -1 ✗ ⇒ 用户**填不进 -1**（界面上会被夹回 10 ⇒ 填了没效果 ✓）")
    check("-1" in p.sp_max_det.toolTip() and "不限" in p.sp_max_det.toolTip(),
          "怪框上限的 tooltip 没写明「-1 = 不限」✗ ⇒ 没人知道能填负数 ✓")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lt = io.open(os.path.join(root, "gui", "live_thread.py"), encoding="utf-8").read()
    check(lt.count("if max_det < 0") == 2,
          "`-1` 的翻译不是**两处** ✗（实测 %d 处）⇒ 热重载那处会把负数塞给模型"
          " ⇒ 当场报错 ✓" % lt.count("if max_det < 0"))
    check(lt.count('max_det", -1') == 1,
          "建会话那处的兜底不是 -1 ✗ ⇒ 「默认不限」没生效 ✓"
          "（⚠ 老项目文件里写了数的照旧生效 ✓ 这条只管缺键 ✓）")
    # ⚠ 2026-10-20：那次推理搬进 `runtime/frame_infer.py` ⇒ 跨两文件数**仍然只许一处** ✓
    check((lt + io.open(os.path.join(root, "runtime", "frame_infer.py"),
                        encoding="utf-8").read()).count("res = model.predict") == 1,
          "推理口变成不止一处了 ✗（这条链上只许有一个 —— 见那边的源码钉 ✓）")


def t_vt_yield_pipe_fps():
    """⭐⭐⭐⭐ 测谎在跑 ⇒ **主回路主动让路**（`runtime.pipeline.vt_yield_pipe_fps` ✓）。

    用户 2026-10-17 ✓ 原话：「**同一窗口实例的测谎和怪物识别是不能同时进行的**」✗ ——
    两套 YOLO 抢同一块 GPU，老做法是"先全速跑、判到积压再把测谎降到 2fps"（**事后刹车** ✗），
    中间那一段两边都慢。这条换成**先分配**：测谎真在喂帧 ⇒ 主回路先降到 `vt_yield_pipe_fps`。

    钉五条：
      ① `yield_fps = 0`（默认）⇒ **原样返回基线** ✓（老行为一字不变 ✓ 默认必须免打扰）；
      ② 测谎没在跑（`vt_on=False`）⇒ 基线 ✓（**勾选着但没喂帧不算** ✗）；
      ③ 基线 30 + 让路 8 ⇒ 8 ✓；
      ④ ⚠ 基线 `0`（不限）时**不能** `min(0, 8)` 成 0（那等于还是不限 ✗）⇒ 必须返回 8 ✓；
      ⑤ 让路档位比基线还大（30/60）⇒ 不许把基线**抬高** ✗（让路只能降速 ✓ 取 min）；
    另外钉**接线**：`live_thread` 里真的用了这个函数，而且把配置键写进 hot-read（不是只写了个函数 ✗）。
    """
    from runtime.pipeline import vt_yield_pipe_fps

    check(vt_yield_pipe_fps(30, True, 0) == 30.0,
          "`yield_fps=0`（默认）时改了基线 ⇒ 老行为被破坏 ✗")
    check(vt_yield_pipe_fps(30, False, 8) == 30.0,
          "测谎没在喂帧也让路了 ⇒ 白降主检测帧率 ✗（判据要用 `_vt_on` ✗ 不是勾选框）")
    check(vt_yield_pipe_fps(30, True, 8) == 8.0, "让路没生效（30/8 ⇒ 应为 8）✗")
    check(vt_yield_pipe_fps(0, True, 8) == 8.0,
          "基线=不限（0）时让路算成了 0 ⇒ 等于没让 ✗（`min(0, 8)` 这个坑 ✓）")
    check(vt_yield_pipe_fps(30, True, 60) == 30.0,
          "让路档位比基线还大时把基线**抬高**了 ✗（让路只能降速 ✓）")

    lt = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    check("vt_yield_pipe_fps(_base, self._vt_on" in lt,
          "`live_thread` 没真调 `vt_yield_pipe_fps` ⇒ 函数写了没用（主回路照旧全速 ✗）")
    check('"vt_yield_pipe_fps"' in lt,
          "hot-read 里没有 `vt_yield_pipe_fps` 这个键 ⇒ 改了配置也不生效 ✗")


def t_vt_gate_and_pause():
    """⭐⭐⭐⭐ 测谎**先嗅探、触发才跑** + 触发期间**暂停全部战斗检测**（用户 2026-10-17 ✓）。

    原话：「**按 B 实现**」+「**触发时暂停所有的战斗检测与逻辑，直到测谎窗口消失
    「有真目标、假目标的那个窗口」**」✓。

    钉六条：
      ① `vt_should_feed("screen", False)` ⇒ **不喂**（没小游戏时第二套 YOLO 空转 ✗ 省 GPU ✓）；
      ② `("screen", True)` ⇒ 喂 ✓；
      ③ `("off", False)` ⇒ 喂 ✓（一键退回老语义 A ✓ 别把退路焊死 ✗）；
      ④ `lie_hold_until`：窗口在场 ⇒ 截止时间**往后推** ✓；
      ⑤ 窗口消失 ⇒ **不再推**（宽限走完就恢复 ✓ 不是"一旦测谎过就永久停"✗）；
      ⑥ 接线：`_vt_loop` 真按 `_vt_should_feed()` 跳过喂帧 ✓、
         `_screen_beat` 里**每拍**都推有效期（且在 `_screen_every_tick` 之前 ✓）、
         `_lie_mode` 真的 `or self._vt_lie_active()` ✓、热读里真有 `vt_gate` ✓。
    """
    from runtime.pipeline import lie_hold_until, vt_should_feed

    check(vt_should_feed("screen", False) is False,
          "没在测谎时还在喂帧 ⇒ B 语义没生效（第二套 YOLO 白占算力 ✗）")
    check(vt_should_feed("screen", True) is True, "测谎窗口出现了却不喂帧 ✗")
    check(vt_should_feed("off", False) is True,
          "`vt_gate: off` 退不回老语义 A ⇒ 出问题没法一键回退 ✗")

    check(lie_hold_until(0.0, True, 100.0, 1.5) == 101.5,
          "窗口在场时没把有效期往后推 ✗（测谎期间必须一直算在场 ✓）")
    check(lie_hold_until(101.5, False, 105.0, 1.5) == 101.5,
          "窗口消失后还在推有效期 ✗（那会变成「测过一次就永久停战斗」✗）")

    src = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    check("if not self._vt_should_feed():" in src,
          "`_vt_loop` 没按 `_vt_should_feed()` 跳过喂帧 ⇒ 勾上就一直跑（还是 A 语义 ✗）")
    _i = src.find("def _screen_beat(")
    _j = src.find("def _screen_every_tick(")
    _seg = src[_i:_j]
    check("lie_hold_until(" in _seg,
          "`_screen_beat` 里没推有效期 ⇒ `_vt_lie_active()` 永远是 False（B 语义整条是死的 ✗）")
    check(_seg.find("lie_hold_until(") < _seg.find("self._screen_every_tick("),
          "有效期推得太晚（在 `_screen_every_tick` 之后）⇒ 状态不变的那几拍算不到 ✗")
    check("or self._vt_lie_active()" in src,
          "`_lie_mode` 没有并入窗口有效期 ⇒ 窗口还在时战斗检测就恢复了 ✗")
    check('"vt_gate"' in src, "热读里没有 `vt_gate` ⇒ 改了配置也不生效 ✗")


def t_vt_arm_window():
    """⭐⭐⭐⭐⭐ 「喂帧预算」—— 界面状态认不出小游戏窗口时，靠它兜住整场（用户 2026-10-17 现场 ✓）。

    现场证据（`behavior.log` 2026-10-17）：
        21:29:48.877  测谎窗口出现 ⇒ 开始喂帧
        21:29:49.207  phase=LOCATING
        21:29:52.394  **测谎窗口消失** ⇒ 停喂      ← 只喂了 **3.5 秒**
        21:29:52.544  locate dump: **reason=MATCHED, title=0.944** ✓ 定位其实是好的
    再拿那一帧的画面复算：`lie_game` 0.653（门槛 0.75）、`lie_success` 0.745（门槛 0.80）
    ⇒ `perception/screen_state` **认不出小游戏窗口** ⇒ 回落 `combat` ⇒ 门一关，
    真正要跟的那一段**一帧都没喂到** ✗（用户看到的就是"没有框、没有 ROI、没有操控"✓）。

    钉六条：① 两个时间窗取**大**；② 都过期才算不在场；
            ③ 预算只在小游戏期间成立（`SUCCESS/ENDED` 立刻清零）；
            ④ 只在**测谎功能开着**时押预算（没开不许白停战斗 ✗）；
            ⑤ 热读里有 `vt_arm_s`；⑥ 配置里真有这个键。
    """
    from runtime.pipeline import vt_lie_active

    check(vt_lie_active(0.0, 0.0, 100.0) is False, "两个窗都过期了还算在场 ✗")
    check(vt_lie_active(101.0, 0.0, 100.0) is True, "短窗（界面状态）没生效 ✗")
    check(vt_lie_active(0.0, 125.0, 100.0) is True,
          "长预算没生效 ⇒ 界面状态一掉线就停喂（现场 3.5 秒就断）✗")
    check(vt_lie_active(90.0, 80.0, 100.0) is False, "都过期了还说在场（战斗检测会一直停 ✗）")
    check(vt_lie_active(90.0, 125.0, 100.0) is True, "两窗该取**大**（125 > 90）✗")

    src = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    check('"_vt_arm_until"' in src and "_vt_arm_s" in src,
          "没有长预算字段 ⇒ 这条防线不存在 ✗")
    check('_d.get("_vt_on")' in src, "押预算时没判 `vt_on` ⇒ 测谎没开也会白停战斗 ✗")
    check('if phase in ("SUCCESS", "SUCCESS_PENDING", "ENDED")' in src,
          "`SUCCESS/ENDED` 时没收预算 ⇒ 一场结束后还要白停 25 秒 ✗")
    check('"vt_arm_s"' in src, "热读里没有 `vt_arm_s` ⇒ 现场调不了 ✗")

    cfg = (ROOT / "config" / "live.yaml").read_text(encoding="utf-8")
    check("vt_arm_s:" in cfg, "`config/live.yaml` 里没有 `vt_arm_s` ✗")


def t_hid_session_guard():
    """⭐⭐⭐⭐ **远程桌面会话里，本机 ProMicro（硬件键盘）打不进本会话**（用户 2026-10-19 ✓
    "C机远程控制B机环境下 B机的本地ProMicro操作失效了"✓）。

    现场（B 机 `LOCKERSHOME`，C 正用**远程桌面**控着它）：
    ```text
    qwinsta →
      >rdp-tcp#0  Locker  1  Active   ← C 的远程桌面会话（工作台 + Maplestory_Classic.exe 都在这）
       console             2  Conn     ← 物理控制台：硬件 HID 只会进这里
    本进程 session = 1
    COM8 Arduino Leonardo          ← 串口好好的（不是占用、不是驱动问题）
    ```
    ⇒ 链路上查一辈子也查不出来 ✗：ProMicro 是真 USB 键盘，Windows 只把物理 HID 喂给
    **控制台会话** ✓。判据与实测都写在 `core/session.py` ✓，界面照实说 ✓。

    钉四条：
      ① 同会话 ⇒ 不提示（正常机器绝不打扰 ✓）；
      ② 不同会话（远程桌面）⇒ 提示里**带两个会话号**，且说清"硬件键盘"这件事 ✓；
      ③ 判不了（拿不到会话号）⇒ **不拦不吓人** ✓（没证据不判死，同全项目的纪律 ✓）；
      ④ **接线**：`player_panel._on_device_connected` 的"本地 ProMicro 连上"那支
        真的调了 `session.warning_text()` ✓（只写模块不接 = 白写 ✗）。
    """
    from core import session as sess

    orig = (sess._proc_sid, sess._console_sid)
    try:
        # ① 同会话（正常单机）
        sess._proc_sid = lambda: 1
        sess._console_sid = lambda: 1
        sess._INFO.clear()
        check(sess.info()["hid_ok"] is True and sess.warning_text() == "",
              "同一个会话也在报警（正常单机会被无端打扰 ✗）")

        # ② 远程桌面：本进程在 1、控制台在 2
        sess._console_sid = lambda: 2
        sess._INFO.clear()
        d = sess.info()
        check(d["hid_ok"] is False,
              "进程会话 != 控制台会话，却没判出「硬件键盘打不进来」✗")
        w = sess.warning_text()
        check("远程桌面" in w and "1" in w and "2" in w,
              "提示里没把两个会话号说清（用户看不懂该改什么 ✗）：%r" % w)
        check("硬件键盘" in w or "ProMicro" in w,
              "提示没说清「ProMicro 是硬件键盘」这个关键点 ✗：%r" % w)

        # ③ 判不了 ⇒ 不拦
        sess._proc_sid = lambda: None
        sess._INFO.clear()
        check(sess.info()["hid_ok"] is True and sess.warning_text() == "",
              "拿不到会话号还报警（没证据就判死 ✗）")
    finally:
        sess._proc_sid, sess._console_sid = orig
        sess._INFO.clear()

    # ④ 接线
    src = (ROOT / "gui" / "player_panel.py").read_text(encoding="utf-8")
    i = src.find("def _on_device_connected")
    check(i > 0, "找不到 `_on_device_connected`（接线钉要看它 ✗）")
    seg = src[i:i + 2600]
    check("session" in seg and "warning_text(" in seg,
          "`_on_device_connected` 里没调 `session.warning_text()` ⇒ 连接成功也不提示 ✗")


def t_big_mob_ui_sync():
    """⭐⭐⭐ **「大怪优先」打开项目时三个控件必须一起回填**（用户 2026-10-19 ✓
    原话："大怪优先配置项失效了"）。

    现场形状（B 机 13:10，项目 `森林迷宫_珠缨` / map 105040303）：
    ```text
    projects/森林迷宫_珠缨/project.yaml →  big_mob_enabled: false
    界面上「大怪优先」却显示**勾着**（用户以为开着的 ✓）
    behavior.log 里 `big_lock_mark v=0.0` 共 137 拍、一次 `v=1.0` 都没有
    ```
    病根：`_sync_from_settings()`（打开/切换项目后的回填）里**只回填了两个数字**
    （`big_mob_ratio` / `big_mob_range_px`）✗ —— **勾选框没回填**、也没重跑
    `_apply_big_mob_enabled()` ⇒ **看到的不是用的** ✗（界面上是上一个项目的状态，
    逻辑吃的是这个项目的 `settings.big_mob_enabled`）。

    钉三条：
      ① `big_mob_ui_state()` 口径对（关着就是 False、数字原样）✓；
      ② 老 settings 对象没有这几个字段 ⇒ 默认 True / 1.5 / 600（老行为不变 ✓）；
      ③ **接线**：`_sync_from_settings` 里必须 `ck_big_mob.setChecked(...)` ＋
         `_apply_big_mob_enabled()`（只回填数字 = 白搭 ✗）。
    """
    import types

    from gui.player_panel import big_mob_ui_state

    st = types.SimpleNamespace(big_mob_enabled=False, big_mob_ratio=2.0,
                               big_mob_range_px=800.0)
    d = big_mob_ui_state(st)
    check(d["enabled"] is False, "关着的项目算出来还是开（界面就会勾着 ✗）：%r" % (d,))
    check(abs(d["ratio"] - 2.0) < 1e-9 and abs(d["range_px"] - 800.0) < 1e-9,
          "倍数 / 半径没跟着项目走 ✗：%r" % (d,))

    d2 = big_mob_ui_state(types.SimpleNamespace())
    check(d2["enabled"] is True and abs(d2["ratio"] - 1.5) < 1e-9
          and abs(d2["range_px"] - 600.0) < 1e-9,
          "老 settings 没这几个字段时兜底变了（老行为要保持 True/1.5/600 ✓）：%r" % (d2,))

    src = (ROOT / "gui" / "player_panel.py").read_text(encoding="utf-8")
    i = src.find("def _sync_from_settings")
    check(i > 0, "找不到 `_sync_from_settings`（回填就在这里 ✓）")
    seg = src[i:i + 12000]
    check("ck_big_mob.setChecked" in seg,
          "③ 回填里没给「大怪优先」勾选框摆值 ⇒ 打开项目后界面和逻辑对不上 ✗")
    check("_apply_big_mob_enabled()" in seg,
          "③ 回填后没刷「两个参数可不可配」⇒ 勾着却还是灰的 ✗")


def t_panel_mismatch_says_why():
    """⭐⭐⭐ **标定与面板尺寸不符时，界面必须说"去重标定"，不许说成"黄点还没锁住"**
    （用户 2026-10-20 ✓ 原话："天空项目 **永远**认不出黄点"）。

    现场（map 200010302 / 天空，`data/dot_probe/app_panel.png` 110×123）：
    ```text
    黄点每一拍都认出来了：px=71.6 py=84.7 ✓（扇形/菱形标记完好）
    但 PlayerLocator.update → ok=False
      note: 这份标定对不上现在这块面板 ⇒ 已拒绝使用
            （标定那块是 97×122，现在推来的是 110×123，差 1.13 倍）
    ```
    ⇒ 用户在「路线识别」只看到"这一拍还没定上位（刚开 / 黄点还没锁住）"✗
    （`route_panel` 把它盖掉了 ✓ 见日志 506）⇒ 只能得出"永远认不出黄点"这个结论 ✓。

    钉四条：
      ① 尺寸不符 ⇒ `panel_mismatch` **拒**，且 why 里**两个尺寸都要出现**（人才知道错在哪 ✓）；
      ② 尺寸一致 ⇒ 放行 ✓；
      ③ 差 1px（110×122 vs 110×123）⇒ **放行** ✓（推流边界会让高度抖 1px，别为此反复拒 ✗）；
      ④ **接线**：`route_panel` 的"没坐标"那支必须**先认 `panel_mismatch`**（否则又被盖成黄点没锁住 ✗）。
    """
    import numpy as np

    from perception import map_calib as map_calib
    from perception import map_const as map_const

    cal = {"mode": map_const.MODE_FIT, "scale": 1.0, "offset": [0.0, 0.0],
           "view": [0.0, 0.0], "panel": [97, 122]}
    big = np.zeros((123, 110, 3), np.uint8)

    ok, why = map_calib.panel_mismatch(cal, big, (93, 122))
    check(ok is False, "标定 97×122、推来 110×123 却放行了（会整倍数错 ✗）")
    check("97" in why and "110" in why and "重新标定" in why,
          "拒用理由没说清两个尺寸 / 没让人去重新标定 ✗：%r" % (why[:120],))

    ok2, _ = map_calib.panel_mismatch(dict(cal, panel=[110, 123]), big, (93, 122))
    check(ok2 is True, "尺寸一致也拒了（正常机器上会无缘无故不出位置 ✗）")

    ok3, why3 = map_calib.panel_mismatch(dict(cal, panel=[110, 122]), big, (93, 122))
    check(ok3 is True, "高度差 1px 就拒（推流边界本来就会抖 ⇒ 会反复拒 ✗）：%r" % (why3,))

    # ④ 接线：`route_panel` 里"没坐标"那支必须先认 `panel_mismatch`
    src = (ROOT / "gui" / "route_panel.py").read_text(encoding="utf-8")
    i_pm = src.find('_pm = bool(r.get("panel_mismatch"))')
    check(i_pm > 0, "`route_panel` 没认 `panel_mismatch` ⇒ 标定尺寸不符会被说成黄点问题 ✗")
    i_gen = src.find("这一拍还没定上位", i_pm)
    check(i_gen > i_pm,
          "`panel_mismatch` 那支排在「还没定上位」之后 ⇒ 真原因还是会被盖掉 ✗")


def t_panel_adapt_by_ratio():
    """⭐⭐⭐⭐ **面板像素变了、但框选区域没变 ⇒ 标定按比例换算继续用，不用重标**
    （用户 2026-10-20 ✓ 原话："为什么经常出现这个问题" ⇒ 选方案 A ✓）。

    病根：`mmap_crop` 有**比例副本**（帧尺寸变了自动跟着缩 ✓），而标定钉的是**像素**尺寸 ✗
    ⇒ 改推流分辨率 / 挪窗口之后，**所有图**的标定一起作废，症状却是"认不出黄点"
    （2026-10-20 天空 ✓ 日志 506）。方案 A：标定里再记一份 `panel_rel`（面板 / 帧 ✓），
    运行时拿 `frame_wh` 反推"区域没变的话面板该多大" ⇒ 对得上就**等比缩放几何** ✓。

    钉五条：
      ① **等比缩放对**：面板 200×120 → 300×180（1.5 倍）⇒ `scale × 1.5`、`offset × 1.5`、
         `panel` 改成新尺寸 ✓（再用 `panel_mismatch` 复验：必须已经能过 ✓）；
      ② **真重框 ⇒ 不救**：比例对不上（宽高比变了）⇒ `None` + 说"框选区域变了" ✓；
      ③ **老标定（没 `panel_rel`）⇒ 不救**（照旧拒用，老行为一字不变 ✓）；
      ④ **没给帧尺寸 ⇒ 不救**（拿不出证据就别猜 ✗）；
      ⑤ **接线**：`PlayerLocator.update` 里真调了 `adapt_panel`、`live_thread` /
         `route_panel` 真把 `frame_wh` 传下去 ✓（只写函数不接 = 白写）。
    """
    import numpy as np

    from perception import map_calib as map_calib
    from perception import map_const as map_const

    cal = {"mode": map_const.MODE_FIT, "scale": 2.0, "offset": [10.0, 4.0],
           "view": [0.0, 0.0], "panel": [200, 120],
           # 标定时那一帧是 1000×600 ⇒ 面板占 0.2 / 0.2 ✓
           "panel_rel": [0.2, 0.2], "frame_wh": [1000, 600]}
    panel_ok = np.zeros((180, 300, 3), np.uint8)      # 帧放大 1.5 倍后的面板 ✓

    out, info = map_calib.adapt_panel(cal, panel_ok, frame_wh=(1500, 900))
    check(out is not None, "区域没变却救不回来（那就还得每张图重标 ✗）：%r" % (info,))
    if out:
        check(abs(float(out["scale"]) - 3.0) < 1e-6,
              "scale 没跟着放大（该 2.0×1.5=3.0 ✓）：%r" % (out.get("scale"),))
        check(abs(out["offset"][0] - 15.0) < 1e-6 and abs(out["offset"][1] - 6.0) < 1e-6,
              "offset 没跟着放大（该 [15, 6] ✓）：%r" % (out.get("offset"),))
        check(list(out.get("panel") or []) == [300, 180],
              "换算后的 `panel` 该改成**现在的**尺寸（否则后面那道闸还会拒 ✗）：%r"
              % (out.get("panel"),))
        _ok, _why = map_calib.panel_mismatch(out, panel_ok, (500, 300))
        check(_ok is True, "换算完还是被判「对不上」（等于没救 ✗）：%r" % (_why[:80],))

    # ② 真重框：宽高比变了（300×100）⇒ 不许救
    out2, info2 = map_calib.adapt_panel(cal, np.zeros((100, 300, 3), np.uint8),
                                        frame_wh=(1500, 900))
    check(out2 is None, "宽高比都变了还算「区域没变」⇒ 会算出错的世界坐标 ✗")
    check("框选区域变了" in str(info2.get("why") or ""),
          "拒救的理由没说清是「框选区域变了」（人不知道去重框 ✗）：%r" % (info2,))

    # ③ 老标定没有 panel_rel
    _old = {k: v for k, v in cal.items() if k not in ("panel_rel", "frame_wh")}
    out3, info3 = map_calib.adapt_panel(_old, panel_ok, frame_wh=(1500, 900))
    check(out3 is None, "老标定（没记比例）也敢按比例套 ⇒ 拿错几何算世界坐标 ✗")

    # ④ 没给帧尺寸
    out4, info4 = map_calib.adapt_panel(cal, panel_ok, frame_wh=None)
    check(out4 is None, "没有帧尺寸还硬救（没证据就猜 ✗）")

    # ⑤ 接线
    loc_src = (ROOT / "perception" / "map_locator.py").read_text(encoding="utf-8")
    check("adapt_panel(" in loc_src and "frame_wh" in loc_src,
          "`PlayerLocator.update` 里没接 `adapt_panel` / `frame_wh` ⇒ 白写 ✗")
    lt = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    check("frame_wh=(self._mmap_frame_wh if src == \"live\" else None)" in lt,
          "实时回路没把帧尺寸传给定位器 ⇒ 现场永远不会触发救回 ✗")
    rp = (ROOT / "gui" / "route_panel.py").read_text(encoding="utf-8")
    check("frame_wh=_frame_wh" in rp,
          "路线识别那一路没把帧尺寸传给定位器（两条链口径要一致 ✓）✗")
    mc = (ROOT / "gui" / "minimap_calib.py").read_text(encoding="utf-8")
    check("panel_rel" in mc and "src_wh" in mc,
          "标定保存时没记 `panel_rel`（那运行时就永远救不了 ✗）")


def t_aim_health_and_link_gate():
    """⭐⭐⭐⭐ **远程这套里"鼠标没跟上准心"要能查出来**
    （用户 2026-10-20 ✓ 原话："刚才测谎鼠标没有跟上准心的位置" ＋ "**是远程**"）。

    远程的含义（这套部署的硬事实 ✓）：画面**收流**、键鼠走 **relay** ✓ ⇒
      · 看不见 A 机光标 ⇒ 瞄准起算点只能靠 relay 答 `CURSOR?` 后台采样（`_cur_loop` ✓）；
      · 采不到 ⇒ 每拍"位置不可信"就 `corner_zero()` ⇒ 光标**先飞左上角再走回来** ✗
        （用户 2026-10-17 就问过"为什么先到左上角"✓）⇒ 现场看着就是"乱跑、不跟准心" ✓；
      · relay 断了（14:09:47 `link-dead`）⇒ 走位线程还在每 16ms 发**永远到不了**的指令 ✗。

    钉四条：
      ① `aim_health()` 把"撞角次数 / 为什么 / 光标采样多旧"报出来 ✓（现场不用再猜 ✗）；
      ② 没有光标采样 ⇒ 撞角，且原因写**人话**（"relay 的 CURSOR? 没回"✓）；
      ③ 有**新鲜**采样 ⇒ **不许撞角**（从当前位置走 ✓ 用户 2026-10-17 的要求 ✓）；
      ④ 接线：走位线程按 `link_health()` 暂停/恢复，且把撞角记成 `vt_aim_home` ✓。
    """
    import time

    from execution import aim_convert as ma_c
    from execution import input as dinput
    from execution import mouse_aim as ma

    # ---- ④ 源码钉（纯文本检查，最省 ✓）----
    _src = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    _seg = _src[_src.index("def _vt_aim_loop"):]
    _seg = _seg[:_seg.index("\n    def ", 10)]
    check("link_health()" in _seg and "vt_aim_hold" in _seg,
          "走位线程没看链路健康状况 ⇒ relay 断着还在空发指令（用户看到的\"没跟上\"✗）")
    check("vt_aim_resume" in _seg, "链路恢复不说一声 ⇒ 也不知道什么时候自己好了 ✗")
    check("aim_health()" in _seg and "vt_aim_home" in _seg,
          "撞角没记账 ⇒ 现场分不清是 relay 不给采样还是我们自己算错 ✗")

    _old = (ma._AIM_CUR, ma._AIM_AT, ma._AIM_TGT, ma._AIM_TGT_AT,
            ma._CUR_HINT, ma._CUR_HINT_AT, ma._CUR_SCREEN, ma._CUR_SCREEN_AT,
            ma._HOME_N, ma._HOME_WHY, ma._HOME_AT)
    _old_mv, _old_av, _old_cz = dinput.mouse_move, dinput.mouse_available, ma_c.corner_zero
    _sent = []
    try:
        dinput.mouse_available = lambda: True
        dinput.mouse_move = lambda dx, dy: _sent.append((dx, dy))
        _cz = []
        ma_c.corner_zero = lambda *a, **k: (_cz.append(1), True)[1]

        # ---- ① 结构 ----
        h = ma.aim_health()
        check(set(h) >= {"homes", "home_why", "cur_age", "hint_age"},
              "`aim_health()` 该报「撞角次数 / 原因 / 采样多旧」（现场就靠它 ✗）：%r" % (h,))

        # ---- ② 没有采样 ⇒ 撞角 + 人话原因 ----
        ma._AIM_CUR, ma._AIM_AT = None, 0.0
        ma._CUR_HINT, ma._CUR_HINT_AT = None, 0.0
        ma._CUR_SCREEN, ma._CUR_SCREEN_AT = None, 0.0
        ma._HOME_N, ma._HOME_WHY = 0, ""
        ma._AIM_TGT, ma._AIM_TGT_AT = (100, 200), time.monotonic()
        _n0 = int(ma.aim_health()["homes"])
        ma.aim_tick(tick_ms=16.0)
        check(int(ma.aim_health()["homes"]) == _n0 + 1,
              "没有光标采样时该撞一次角（否则连原点都没有 ✗）")
        check("CURSOR" in str(ma.aim_health()["home_why"]) or "光标" in str(ma.aim_health()["home_why"]),
              "撞角原因没写人话（现场只能看见光标乱飞，看不出为什么 ✗）：%r"
              % (ma.aim_health()["home_why"],))

        # ---- ③ 有新鲜采样 ⇒ 不许撞角 ----
        _t = time.monotonic()
        ma._AIM_CUR, ma._AIM_AT = None, 0.0
        ma._CUR_HINT, ma._CUR_HINT_AT = (10, 20), _t
        ma._CUR_SCREEN, ma._CUR_SCREEN_AT = (15, 64, 0, 0), _t
        ma._AIM_TGT, ma._AIM_TGT_AT = (300, 400), _t
        _cz2, _n1 = len(_cz), int(ma.aim_health()["homes"])
        ma.aim_tick(tick_ms=16.0)
        check(len(_cz) == _cz2 and int(ma.aim_health()["homes"]) == _n1,
              "有**新鲜**光标采样还撞角 ⇒ 光标先飞左上角（用户 2026-10-17 就否掉了 ✗）")
        check(bool(_sent), "有新鲜采样却一步都没走（瞄准彻底不动了 ✗）")
    finally:
        (ma._AIM_CUR, ma._AIM_AT, ma._AIM_TGT, ma._AIM_TGT_AT,
         ma._CUR_HINT, ma._CUR_HINT_AT, ma._CUR_SCREEN, ma._CUR_SCREEN_AT,
         ma._HOME_N, ma._HOME_WHY, ma._HOME_AT) = _old
        dinput.mouse_move, dinput.mouse_available, ma_c.corner_zero = _old_mv, _old_av, _old_cz


def t_frame_reconnect_stage():
    """⭐⭐⭐⭐ 第 5 段（断线重连）的五个出口各发**正确的**输入、且每条都回填
    （`runtime/frame_reconnect.py` ✓ 2026-10-20 搬出）。

    这段是**唯一会替人按键/点鼠标**的地方 ⇒ 搬错一下就是"点到别的频道 / 进错服务器"✓
    （用户 2026-10-17：「**地铁二号线 无法选中频道**」✗ 那条就是撞角与走位同一拍发出去）。

    钉六条（每条都对应一次现场事故 ✓）：
      ① 状态机没动作 ⇒ **什么都不发** ✓；
      ② `tap` ⇒ 只按键，不动鼠标 ✓；
      ③ `home` ⇒ **只撞角、这一拍不点** ✗（同一拍发两条 ⇒ 起点不是角 ⇒ 点歪 ✓ 2026-10-17）；
      ④ `click` ⇒ 走 `click_ratio`，且原点/ROI 来自 `_aim_geo`（折回整帧 ✓ 2026-10-17）＋回填 ✓；
      ⑤ `click2` ⇒ 只补左键（**不重新走位** ✗ 光标还在原地 ✓）；
      ⑥ `scroll` ⇒ 带 up/down **和坐标**（不带坐标 = 在画面上方滚 ⇒ 列表一格没滚 ✓ 2026-10-09）。
    """
    from types import SimpleNamespace

    from runtime import frame_reconnect

    class _RC:
        def __init__(self, act):
            self.act = act
            self.fb = []

        def update(self, vis, found, now, screen):
            self.found = found
            return self.act

        def feedback(self, ok, why):
            self.fb.append((ok, why))

    class _Mouse:
        def __init__(self):
            self.calls = []

        def corner_only(self, shape):
            self.calls.append(("corner_only",))
            return (True, "ok")

        def click_ratio(self, *a, **kw):
            self.calls.append(("click_ratio", a, kw))
            return (True, "ok")

        def repeat_click(self):
            self.calls.append(("repeat_click",))
            return (True, "ok")

        def scroll(self, *a, **kw):
            self.calls.append(("scroll", a, kw))
            return (True, "ok")

    _old = frame_reconnect.mouse_aim
    m = _Mouse()
    frame_reconnect.mouse_aim = m
    taps = []

    def _tap(key):
        taps.append(key)

    def _mk(act):
        rc = _RC(act)
        th = SimpleNamespace(_reconnector=rc, _screen_state="combat")
        th._aim_geo = lambda vis, org: ((9, 9), ("roi",), ("full",), 2.5)
        return th, rc

    vis = np.zeros((100, 120, 3), np.uint8)
    try:
        # ① 没动作 ⇒ 什么都不发
        th, rc = _mk(None)
        out = frame_reconnect.step(th, vis, None, None, 3, 4, _tap)
        check(out == "" and not m.calls and not taps and not rc.fb,
              "① 状态机没动作却发了东西（会乱按键 ✗）：%r %r" % (m.calls, taps))

        # ② tap ⇒ 只按键
        th, rc = _mk({"act": "tap", "key": "enter"})
        out = frame_reconnect.step(th, vis, None, None, 3, 4, _tap)
        check(out == "tap" and taps == ["enter"] and not m.calls,
              "② `tap` 没按键 / 还动了鼠标 ✗：%r %r" % (taps, m.calls))
        check(rc.found is False, "② 把「没找到玩家」传成了 True（状态机会以为人还在 ✗）")

        # ③ home ⇒ 只撞角，**这一拍不点**
        m.calls.clear()
        th, rc = _mk({"act": "home"})
        frame_reconnect.step(th, vis, None, None, 3, 4, _tap)
        check([c[0] for c in m.calls] == ["corner_only"],
              "③ `home` 这一拍既该撞角、又**不该点**（同一拍发两条 ⇒ 起点不是角 ✗）：%r"
              % ([c[0] for c in m.calls],))
        check(rc.fb and rc.fb[0][0] is True, "③ 撞角结果没回填（界面只剩「准备点…」✗）")

        # ④ click ⇒ click_ratio + 折回整帧 + 回填
        m.calls.clear()
        th, rc = _mk({"act": "click", "x": 0.5, "y": 0.5, "homed": True})
        frame_reconnect.step(th, vis, None, None, 3, 4, _tap)
        _c = [c for c in m.calls if c[0] == "click_ratio"]
        check(len(_c) == 1, "④ `click` 没走 `click_ratio`（换算必须只有一处 ✗）：%r" % (m.calls,))
        check(_c[0][2].get("origin") == (9, 9) and _c[0][2].get("roi") == ("roi",)
              and _c[0][2].get("gain") == 2.5,
              "④ 没按 `_aim_geo` 折回整帧（窗口挪过就会点歪 ✗）：%r" % (_c[0][2],))
        check(_c[0][2].get("homed") is True,
              "④ 丢了 `homed` ⇒ 会再撞一次角（起点又不是角了 ✗）")
        check(bool(rc.fb), "④ 点击结果没回填 ✗")

        # ⑤ click2 ⇒ 只补左键
        m.calls.clear()
        th, rc = _mk({"act": "click2"})
        frame_reconnect.step(th, vis, None, None, 3, 4, _tap)
        check([c[0] for c in m.calls] == ["repeat_click"],
              "⑤ 双击第二下该**只补左键**、不重新走位 ✗：%r" % ([c[0] for c in m.calls],))

        # ⑥ scroll ⇒ 带 up/down 和坐标
        m.calls.clear()
        th, rc = _mk({"act": "scroll", "up": 3, "down": 0, "x": 0.2, "y": 0.8})
        frame_reconnect.step(th, vis, None, None, 3, 4, _tap)
        _s = [c for c in m.calls if c[0] == "scroll"]
        check(len(_s) == 1, "⑥ 滚动没走 `mouse_aim.scroll` ✗：%r" % (m.calls,))
        check(_s[0][1][0] == 3 and _s[0][1][1] == 0,
              "⑥ 滚动的 up/down 传错了（滚错方向 ⇒ 找不到频道 ✗）：%r" % (_s[0][1][:2],))
        check(_s[0][2].get("origin") == (3, 4) and _s[0][1][3] == 0.2 and _s[0][1][4] == 0.8,
              "⑥ 滚动没带坐标 / 原点（滚轮只作用在光标底下 ⇒ 一格没滚 ✗）：%r" % (_s[0],))
    finally:
        frame_reconnect.mouse_aim = _old


def t_frame_locate_stage():
    """⭐⭐⭐⭐ 第 7 段（玩家定位）**真的跑一遍**（`runtime/frame_locate.py` ✓ 2026-10-20 搬出）。

    为什么要"真跑"：这一段的**前置顺序**和**两条独立的路**都只在运行时才露出来
    （第 6 段刚教过一次：源码钉一条不红、第一帧就炸 ✗ 见日志 525 ✓）。

    钉六条（每条都有现场账）：
      ① **脚底偏移**灌进 `world_state.FOOT_OFFSET_PX`（画「玩家位置」箭头靠它 ✓ 用户 2026-09-28）；
      ② **框面积闸**：上一拍框比"过去 N 秒基线"小太多 ⇒ 这一拍**不取**世界坐标
         （`loc=None` ⇒ 沿上一拍 = "不定位"，而不是"定到错的地方"✗）；
      ③ 两个"关"的口子：占比 0 / 容差 0 都关着 ⇒ **照常查询**（老行为一字不变 ✓）；
      ④ **两条取数路**：来源=live ⇒ `_locate_latest()`；来源=收流 + 回路在跑 ⇒ `_loc_slot_take()`；
      ⑤ 世界坐标**喂给 tracker**（`world=(x, y)`）、解不出就给 `None`（不猜 ✓）；
      ⑥ 相机：`camera()` 透传 · `cam_rejected` 打点 · 第一次拿到时记 `first_cam_s` ✓。
    """
    import time
    from types import SimpleNamespace

    from runtime import frame_locate

    class _Perf:
        def __init__(self):
            self.counts = []
            self.samples = []
            self.notes = []
            self.ms_calls = []

        def count(self, k):
            self.counts.append(k)

        def ms(self, k, t0):
            self.ms_calls.append(k)

        def sample(self, k, v, **kw):
            self.samples.append((k, v))

        def note(self, k, v):
            self.notes.append((k, v))

    class _Trk:
        def __init__(self):
            self.max_jump = 0.0
            self.got = []
            self.cam = None
            self.cam_rejected = False
            self.cam_resid = None
            self.inp = None

        def set_input(self, dx, dy):
            self.inp = (dx, dy)

        def update(self, cands, world=None, low=None):
            self.got.append((list(cands), world, list(low or [])))
            return (100.0, 200.0, 140.0, 320.0, 40.0, 120.0)

        def camera(self):
            return self.cam

    _old_perf, _old_beh = frame_locate.perf, frame_locate.behavior
    perf_fake, beh_fake = _Perf(), _Perf()
    frame_locate.perf, frame_locate.behavior = perf_fake, beh_fake
    try:
        from perception.world import world_state as ws_mod

        settings = SimpleNamespace(player_foot_offset_px=7.0,
                                   player_box_min_area_pct=0.0,
                                   player_box_area_tol_pct=0.0,
                                   player_box_area_base_s=3.0,
                                   player_track_jump=0.0)
        agent = SimpleNamespace(keys=SimpleNamespace(pressed=lambda: {"right", "up"}))
        trk = _Trk()

        def _th(src="live", mid="105040303", thread=None):
            return SimpleNamespace(_mmap_src=src, _mmap_mid=mid, _loc_thread=thread,
                                   _locate_latest=lambda: {"world_x": 1.5, "world_y": 2.5},
                                   _loc_slot_take=lambda: {"world_x": 9.0, "world_y": 8.0})

        # ① 脚底偏移
        out = frame_locate.step(_th(), trk, [(1, 2, 3, 4, 0.9)], [], settings, agent,
                                [None], [], [None], 0.0)
        check(ws_mod.FOOT_OFFSET_PX == 7.0 and out.foot_off == 7.0,
              "① 脚底偏移没灌进 `world_state.FOOT_OFFSET_PX`（箭头会画偏 ✗）：%r"
              % (ws_mod.FOOT_OFFSET_PX,))

        # ④ live ⇒ `_locate_latest`；世界坐标喂给 tracker
        check(out.world_now == (1.5, 2.5) and trk.got[-1][1] == (1.5, 2.5),
              "④ 来源=live 时没现算 / 世界坐标没喂给 tracker（黄点那条传感器白搭 ✗）：%r"
              % (trk.got[-1],))
        # ④' 来源=收流 + 回路在跑 ⇒ 取"最新一份"
        th2 = _th(src="stream", thread=object())
        frame_locate.step(th2, trk, [], [], settings, agent, [None], [], [None], 0.0)
        check(trk.got[-1][1] == (9.0, 8.0),
              "④ 来源=收流时没取回路那份最新结论（会退回每拍现算 ⇒ 白丢帧 ✗）：%r"
              % (trk.got[-1],))
        # ⑤ 解不出世界坐标 ⇒ None（不许猜 ✓）
        th3 = _th()
        th3._locate_latest = lambda: {"world_x": None, "world_y": None}
        frame_locate.step(th3, trk, [], [], settings, agent, [None], [], [None], 0.0)
        check(trk.got[-1][1] is None,
              "⑤ 解不出世界坐标却给了个值/给了 0（tracker 会被带偏 ✗）：%r" % (trk.got[-1],))

        # ② 面积闸：上一拍框比基线小太多 ⇒ 这一拍不取
        now = time.monotonic()
        hist = [(now - 0.1 * i, 40000.0) for i in range(10)]      # 基线 40000
        settings2 = SimpleNamespace(**{**vars(settings), "player_box_min_area_pct": 50.0})
        perf_fake.counts.clear()
        out = frame_locate.step(_th(), trk, [], [],
                                settings2, agent,
                                [(0.0, 0.0, 100.0, 100.0, 10.0, 10.0)],   # 面积 100 ⇒ 远小于 50%
                                hist, [None], 0.0)
        check(out.loc is None and "player_box_too_small" in perf_fake.counts,
              "② 框明显变小了还在取世界坐标（框选错 ⇒ 世界坐标跟着错 ✗）：%r"
              % (out.loc,))

        # ③ 两个口子都关着 ⇒ 照常查询
        perf_fake.counts.clear()
        out = frame_locate.step(_th(), trk, [], [], settings, agent,
                                [(0.0, 0.0, 100.0, 100.0, 10.0, 10.0)],
                                hist, [None], 0.0)
        check(out.loc is not None and not perf_fake.counts,
              "③ 两个闸都关着却仍旧拦了（老行为被改了 ✗）：%r %r"
              % (out.loc, perf_fake.counts))

        # ⑥ 相机三件
        trk.cam = (11.0, 22.0)
        trk.cam_resid = 3.0
        _first = [None]
        out = frame_locate.step(_th(), trk, [], [], settings, agent,
                                [None], [], _first, 0.0)
        check(out.cam_now == (11.0, 22.0) and _first[0] is not None,
              "⑥ 可信相机没透传 / 第一次拿到时没记时刻（`first_cam_s` 打不出来 ✗）")
        check(("cam_x", 11.0) in beh_fake.samples and ("cam_resid", 3.0) in beh_fake.samples,
              "⑥ 相机与残差没进行为日志（\"世界坐标整体偏\"只能靠它排查 ✗）")
        check(("cam_valid", 1.0) in perf_fake.samples,
              "⑥ `cam_valid` 没打（有效比例那个指标就没了 ✗）")
        trk.cam_rejected = True
        perf_fake.counts.clear()
        frame_locate.step(_th(), trk, [], [], settings, agent, [None], [], [None], 0.0)
        check("cam_reject" in perf_fake.counts,
              "⑥ 这一拍\"框与黄点对不上\"没打点（一直涨说明锁错了 ✗）")

        # 按键意图要喂进去（输入感知的运动预测 ✓ 用户 2026-10-01）
        check(trk.inp == (1, 1),
              "按键意图没喂给 tracker（漏检那几帧会朝错的方向外推 ✗）：%r" % (trk.inp,))
    finally:
        frame_locate.perf, frame_locate.behavior = _old_perf, _old_beh


def t_frame_infer_stage():
    """⭐⭐⭐⭐ 第 6 段（一次推理 + 谁进决策）**真的跑一遍**
    （`runtime/frame_infer.py` ✓ 2026-10-20 搬出）。

    为什么这条必须**真跑**（不能只做源码钉）：搬这段时最容易出的错是**作用域**
    —— 实测我自己就漏了：`getattr(self, "_vision_player_miss")` 里的裸 `self`
    和被改名的 `settings` 都没跟着改 ⇒ **实时线程第一帧推理就 NameError** ✗，
    而源码钉一条都不会红 ✓（它们只看文本在不在）。

    钉六条：
      ① 测谎中（`_lie_mode`）⇒ **不推理**、四路产物全空（框自然不画 ✓ 但显示照旧 ✓）；
      ② 关着「只检测视野框」⇒ 喂**整帧**、偏移 `(0,0)`、打点 `det_full` ✓；
      ③ 开着 + 上一拍认出了玩家 ⇒ 喂**剪裁块**，且**偏移加回坐标**（框回到全画面 ✓）；
      ④ 上一拍**没**认出玩家（`_vision_player_miss`）⇒ 这一拍退回**整帧** ✓
         （不许开第二个推理口 ✗ 见那个文件开头）；
      ⑤ 四路产物（怪 / 低分怪 / 玩家 / 低分玩家）分别喂 `split` 的结果 ✓ 掉落只挑掉落类 ✓；
      ⑥ 推理耗时进 `infer_ms` 窗口（封顶 60 ✓）。
    """
    from types import SimpleNamespace

    from runtime import frame_infer

    class _Arr:
        def __init__(self, a):
            self.a = np.asarray(a, dtype=float)

        def cpu(self):
            return self

        def numpy(self):
            return self.a

    class _Boxes:
        def __init__(self, xyxy, conf, cls):
            self.xyxy = _Arr(xyxy)
            self.conf = _Arr(np.asarray(conf, float).reshape(-1))
            self.cls = _Arr(np.asarray(cls, float).reshape(-1))

        def __len__(self):                   # ⚠ 真 ultralytics 的 Boxes 支持 len() ✓
            return len(self.conf.a)

    class _Res:
        def __init__(self, boxes):
            self.boxes = boxes

    class _Model:
        def __init__(self, boxes):
            self.boxes = boxes
            self.shapes = []

        def predict(self, src, **kw):
            self.shapes.append(tuple(src.shape[:2]))
            return [_Res(self.boxes)]

    class _Perf:
        def __init__(self):
            self.counts = []

        def count(self, k):
            self.counts.append(k)

        def ms(self, k, t0):
            pass

        def sample(self, *a, **k):
            pass

    _old_perf = frame_infer.perf
    fake_perf = _Perf()
    frame_infer.perf = fake_perf
    try:
        vis = np.zeros((300, 400, 3), np.uint8)
        det = {"conf": 0.25, "imgsz": 640, "device": "cpu", "iou": 0.85,
               "max_det": 300, "agnostic_nms": False, "half_kw": {}}
        settings_off = SimpleNamespace(vision_detect_only=False)
        settings_on = SimpleNamespace(vision_detect_only=True)

        def _mk(boxes):
            return _Model(_Boxes(boxes[0], boxes[1], boxes[2]))

        # ① 测谎中 ⇒ 不推理
        m = _mk(([[10, 10, 50, 60]], [0.9], [1]))
        th = SimpleNamespace(_vision_player_miss=False)
        out = frame_infer.step(th, vis, m, det, 1, 0, {}, 0.25, 0.5,
                               True, None, settings_off, [], None, None)
        check(not m.shapes and out.mob == [] and out.player == [] and out.cls_dets == []
              and out.drop == [] and out.k == 0,
              "① 测谎中还在推理 / 产物没清空（战斗框会照旧画 ✗）：%r" % (out,))

        # ② 整帧那条路
        _calls = {}

        def _split(xyxy, cfs, clss, mc, pc, cm, cp):
            _calls["split"] = (len(xyxy), mc, pc, cm, cp)
            return ([(1,)], [(2,)], [(3,)], [(4,)])

        def _extra(xyxy, cfs, clss, cls_map):
            _calls["extra"] = len(xyxy)
            return [("cls",)]

        _x = [[100 + i, 0, 160 + i, 60] for i in range(2)]
        m = _mk((_x, [0.9, 0.9], [1, 1]))
        box_ms = []
        out = frame_infer.step(th, vis, m, det, 1, 0, {},
                               0.25, 0.5, False, None, settings_off, box_ms,
                               _split, _extra)
        check(m.shapes and m.shapes[-1] == (300, 400),
              "② 没喂整帧（关着「只检测视野框」时该喂原图 ✗）：%r" % (m.shapes,))
        check(_calls.get("split", (0,))[0] == 2 and _calls["split"][1:3] == (1, 0),
              "② `split` 没拿到「全部框 + 类号」(mob=1, player=0)：%r" % (_calls.get("split"),))
        check(out.mob == [(1,)] and out.mob_low == [(3,)] and out.player == [(2,)]
              and out.player_low == [(4,)],
              "⑤ 四路产物没各归各位（怪/低分怪/玩家/低分玩家 ✗）：%r" % (out,))
        check("det_full" in fake_perf.counts, "② 整帧那条路没打 `det_full` 点 ✗")
        check(len(box_ms) == 1 and box_ms[0] >= 0.0, "⑥ 推理耗时没进窗口 ✗")

        # ③ 剪裁 + 偏移加回
        _calls.clear()
        m = _mk(([[10, 10, 50, 60]], [0.9], [1]))
        th = SimpleNamespace(_vision_player_miss=False)
        out = frame_infer.step(th, vis, m, det, 1, 0, {},
                               0.25, 0.5, False, (100, 50, 350, 250), settings_on,
                               [], _split, _extra)
        check(m.shapes and m.shapes[-1] == (200, 250),
              "③ 开着「只检测视野框」却没喂剪裁块（白勾 ✗）：%r" % (m.shapes,))
        check(out.crop_off == (100, 50), "③ 没把剪裁偏移记下来 ✗：%r" % (out.crop_off,))
        check("det_crop" in fake_perf.counts, "③ 剪裁那条路没打 `det_crop` 点 ✗")

        # ④ 上一拍没认出玩家 ⇒ 退回整帧
        m = _mk(([[10, 10, 50, 60]], [0.9], [1]))
        th = SimpleNamespace(_vision_player_miss=True)
        out = frame_infer.step(th, vis, m, det, 1, 0, {},
                               0.25, 0.5, False, (100, 50, 350, 250), settings_on,
                               [], _split, _extra)
        check(m.shapes and m.shapes[-1] == (300, 400),
              "④ 上一拍没认出玩家、这一拍还在剪裁（玩家会**永久定住** ✗ 用户 2026-10-16）：%r"
              % (m.shapes,))

        # ⑤ 掉落只挑掉落类（宠物 / NPC 混进来会让人站在宠物旁边也在拾取 ✗）
        from perception.detect.classes import CLASS_DROP, CLASS_MOB
        m = _mk(([[0, 0, 5, 5], [0, 0, 6, 6]], [0.9, 0.9], [CLASS_MOB, CLASS_DROP]))
        th = SimpleNamespace(_vision_player_miss=False)
        out = frame_infer.step(th, vis, m, det, 1, 0, {CLASS_DROP: CLASS_DROP,
                                                      CLASS_MOB: CLASS_MOB},
                               0.25, 0.5, False, None, settings_off, [],
                               _split, _extra)
        check(len(out.drop) == 1 and out.drop[0][2] == 6.0,
              "⑤ 掉落框没只挑掉落类（宠物/NPC 混进来 ⇒ 站在宠物旁边也在点拾取 ✗）：%r"
              % (out.drop,))

        # ⑥ 耗时窗口封顶
        m = _mk(([[0, 0, 5, 5]], [0.9], [1]))
        _ms = [0.0] * 60
        for _i in range(5):
            frame_infer.step(th, vis, m, det, 1, 0, {}, 0.25, 0.5, False, None,
                             settings_off, _ms, _split, _extra)
        check(len(_ms) == 60, "⑥ 推理耗时窗口没封顶（会一直涨 ✗）：%d" % len(_ms))
    finally:
        frame_infer.perf = _old_perf


def t_frame_probe_stage():
    """⭐⭐⭐⭐ 第 4 段（探针解码）的四类账要**分得开**（`runtime/frame_probe.py` ✓ 2026-10-20 搬出）。

    为什么这条值一条用例（现场账，`_run` 里那段老注释记着 ✓）：
      · 解码失败（`miss`）会让延迟样本**偏小** ⇒ 看着像"延迟变好了"（实测 1433→161 个样本、
        中位 35ms→12ms ✓）⇒ 必须能与"成功"分开数 ✓；
      · 值 ≥ 一天（`invalid`）= **几何错了**（采样点没落在方块上）✗ 不是时钟问题 ——
        混进"去对时"会把人引错方向（实测解出 85 小时 ✓）；
      · 值合法但算不出延迟（`reject`）= 双机时钟没对齐 ⇒ 和"没解出"是两件事 ✓。

    钉七条：① 关着不跑 ② miss ③ invalid ④ ok + 延迟进窗口 ⑤ reject 不进窗口
    ⑥ 几何一变重置判据（换标定那一刻不该被算成"乱跳"）⑦ 窗口上限 120 ✓
    """
    from types import SimpleNamespace

    from runtime import frame_probe

    class _Verdict:
        def __init__(self):
            self.mono = True
            self.value_ok = True
            self.step_med = 0.0
            self.ts_hist = []
            self.value_off_ms = None
            self.resets = 0
            self.added = []

        def reset(self):
            self.resets += 1

        def add(self, ts, step, delay_raw_ms=None):
            self.added.append((ts, delay_raw_ms))

    class _Codec:
        DAY_MS = 24 * 60 * 60 * 1000

        def __init__(self):
            self.delay = 42.0

        def day_start_ms(self):
            return 0.0

        def resolve_delay_ms(self, t_recv_ms, ts_a):
            return self.delay

    class _Perf:
        def __init__(self):
            self.counts = []
            self.samples = []

        def count(self, k):
            self.counts.append(k)

        def sample(self, k, v):
            self.samples.append((k, v))

    _old_codec, _old_perf = frame_probe.probe_codec, frame_probe.perf
    codec, perf_fake = _Codec(), _Perf()
    frame_probe.probe_codec, frame_probe.perf = codec, perf_fake
    try:
        f = SimpleNamespace(t_recv_wall=1000.0)
        vis = np.full((40, 60, 3), 128, np.uint8)
        st = frame_probe.ProbeState(vd=_Verdict())
        hits = []

        def _geom(shape):
            return (1.0, 2.0, 3.0, 4.0)

        def _decode(gray, px, py, cell, gap, bits):
            hits.append(1)
            return _decode.next_ts

        _decode.next_ts = None

        # ① 关着不跑（连灰度都不转 ✓）
        check(frame_probe.step(st, False, vis, f, _geom, _decode, lambda: 0.0, 40) is None
              and not hits and st.miss == 0,
              "① 探针关着还在解（每帧白转一次灰度 ✗）")

        # ② 解不出来 ⇒ miss（不是 ok）
        _decode.next_ts = None
        frame_probe.step(st, True, vis, f, _geom, _decode, lambda: 0.0, 40)
        check(st.miss == 1 and st.ok == 0 and "probe_miss" in perf_fake.counts,
              "② 解码失败没记成 miss（延迟样本会偏小、看着像变好 ✗）：%r" % (st,))

        # ③ 值不可能（≥ 一天）⇒ invalid（**不是**去对时 ✓）
        _decode.next_ts = _Codec.DAY_MS + 1
        frame_probe.step(st, True, vis, f, _geom, _decode, lambda: 0.0, 40)
        check(st.invalid == 1 and "probe_invalid" in perf_fake.counts,
              "③ 不可能的值没记成 invalid（会被误导去对时 ✗）：%r" % (st,))

        # ④ 解出合法值 ⇒ ok + 延迟进窗口 + e2e 更新
        _decode.next_ts = 5_000
        d = frame_probe.step(st, True, vis, f, _geom, _decode, lambda: 0.0, 40)
        check(d == 42.0 and st.ok == 1 and st.delays == [42.0] and st.e2e_ms == 42.0,
              "④ 解出合法值却没进延迟窗口 / 没更新 e2e：d=%r %r" % (d, st))
        check(("e2e_probe_ms", 42.0) in perf_fake.samples,
              "④ 端到端延迟没进 perf（那是控制质量的真指标 ✗）")

        # ⑤ 算不出延迟 ⇒ reject，且**不进窗口**
        codec.delay = None
        _decode.next_ts = 6_000
        check(frame_probe.step(st, True, vis, f, _geom, _decode, lambda: 0.0, 40) is None
              and st.reject == 1 and len(st.delays) == 1,
              "⑤ 算不出延迟却写进了窗口（会报一个假延迟 ✗）：%r" % (st,))
        codec.delay = 42.0

        # ⑥ 几何一变 ⇒ 重置判据（换标定那一刻不该被算成"乱跳" ✓）
        _r0 = st.vd.resets
        frame_probe.step(st, True, vis, f, _geom, _decode, lambda: 0.0, 40)
        check(st.vd.resets == _r0,
              "⑥ 几何没变也重置判据（等于永远学不到「这份几何可不可信」✗）")
        frame_probe.step(st, True, vis, f, lambda shape: (9.0, 2.0, 3.0, 4.0),
                         _decode, lambda: 0.0, 40)
        check(st.vd.resets == _r0 + 1,
              "⑥ 几何变了没重置判据（刚存完标定就报「几何可疑」，人会被自己吓到 ✗）")

        # ⑦ 窗口上限
        for _i in range(frame_probe.DELAY_WINDOW + 20):
            frame_probe.step(st, True, vis, f, _geom, _decode, lambda: 0.0, 40)
        check(len(st.delays) == frame_probe.DELAY_WINDOW,
              "⑦ 延迟窗口没封顶（涨到 %d 个 ⇒ 中位数会被很久以前的样本拖着 ✗）"
              % len(st.delays))
        # ⑧ `core.probe_codec` **不可用时要降级**（复核补的 ✓ 2026-10-20）：
        #    老代码在 `live_thread` 里是 try-import + 置 None ⇒ 预览照开、探针关掉 ✓。
        #    本模块若裸 import，一旦那边坏了就是**整个实时线程起不来** ✗（比探针不工作严重得多）。
        frame_probe.probe_codec = None
        st2 = frame_probe.ProbeState()               # 不许抛（判据给替身 ✓）
        check(frame_probe.step(st2, True, vis, f, _geom, _decode, lambda: 0.0, 40) is None,
              "⑧ codec 不可用时没降级（会把整个实时线程拖死 ✗）")
        check(bool(st2.vd.mono) and st2.vd.ts_hist == [],
              "⑧ 降级用的替身判据不该被当成「真的没问题」写进历史 ✗：%r" % (st2.vd,))
    finally:
        frame_probe.probe_codec, frame_probe.perf = _old_codec, _old_perf


def t_frame_vis_stage():
    """⭐⭐⭐⭐ 第 3 段（可视化配置热读）的口径与节流（`runtime/frame_vis.py` ✓ 2026-10-20 搬出）。

    **为什么这些值挂在 `th._vis_ui` 而不是 `FrameCtx`**（用户选的 B ✓）：
      它们**跨帧活着**（1 秒重读一次、中间每拍用上一份 ✓）—— 按 `FrameCtx` 自己的判据
      "下一帧还会用吗？"⇒ 会 ⇒ 不进盒子 ✗（进了就变成"跨帧状态藏在仅本帧的名字里"✓）。

    钉五条（都是搬动时最容易走样的）：
      ① **第一拍必算**（`_vis_refresh_last` 初值 0 ⇒ 老代码就是"首拍必算" ✓）；
      ② **1 秒内不重算**，而且返回的是**同一份对象** ✓（别每拍新建 ⇒ 后面的别名会拿到新对象、
         和"上一拍那份"不再同一 ✓）；
      ③ **到点重算**并换新对象 ✓；
      ④ **类号变了，下一次重算就跟着变**（模型加载后会按名字重对齐 ✓ 见 `_run` ✓）；
      ⑤ **缺省口径不许变**：`lock_width` 缺省 3、其余线宽缺省 2、`on_*` 缺省 True（观感一字不变 ✓）、
         类号在 `class_colors` 里查不到 ⇒ 回退到 `CLASS_PLAYER` / `CLASS_MOB` ✓。
    """
    from types import SimpleNamespace

    from perception.detect.classes import CLASS_MOB, CLASS_PLAYER, ZH_NAMES
    from runtime import frame_vis

    class _Theme:
        """替身：只提供本段要用的四个口（真身是 `gui/theme.py` ✓ 由**调用方**传 ✓）。"""

        def __init__(self):
            self.loads = 0

        def load_vis(self):
            self.loads += 1
            return {"lock_color": "#101112", "attack_color": "#20222344",
                    "min_attack_color": "#30323455", "chase_jump_color": "#40424466",
                    "jump_attack_color": "#50525477", "vision_color": "#60626488",
                    "vision_width": 2}                      # 其余键故意不给 ⇒ 走缺省 ✓

        def class_colors(self):
            return {CLASS_PLAYER: (11, 22, 33), CLASS_MOB: (44, 55, 66)}

        def hex_to_bgr(self, h):
            return ("bgr", h)

        def hex_to_bgra(self, h):
            return ("bgra", h)

    th = SimpleNamespace()
    theme = _Theme()

    # ① 第一拍必算
    ui1 = frame_vis.step(th, CLASS_PLAYER, CLASS_MOB, theme, now=100.0)
    check(ui1 is not None and theme.loads == 1,
          "① 第一拍没算（`_vis_refresh_last` 初值 0 就该算 ✓ 老行为 ✓）：loads=%d" % theme.loads)
    # ② 1 秒内不重算 ⇒ 同一份对象
    ui2 = frame_vis.step(th, CLASS_PLAYER, CLASS_MOB, theme, now=100.5)
    check(ui2 is ui1 and theme.loads == 1,
          "② 1 秒内又重读了 / 换了一份新对象（后面每拍拿到的就不是同一份 ✓）：loads=%d" % theme.loads)
    # ③ 到点重算 ⇒ 新对象
    ui3 = frame_vis.step(th, CLASS_PLAYER, CLASS_MOB, theme, now=101.2)
    check(ui3 is not ui1 and theme.loads == 2,
          "③ 过了一秒还没重读（改颜色不会生效 ✗）：loads=%d" % theme.loads)
    # ④ 类号查不到 ⇒ 回退到类别表里的默认项（颜色 / 名字都要 ✓）
    ui4 = frame_vis.step(th, 7, CLASS_MOB, theme, now=102.5)
    check(ui4.p_color == (11, 22, 33) and ui4.m_color == (44, 55, 66),
          "④ 类号查不到时没回退到 `CLASS_PLAYER` / `CLASS_MOB` 那一项"
          "（换过权重的那份模型会画错颜色 ✗）：%r" % ((ui4.p_color, ui4.m_color),))
    check(ui4.p_label == ZH_NAMES.get(CLASS_PLAYER),
          "④ 标签没跟着回退（会打出原始类号 / KeyError ✗）：%r" % (ui4.p_label,))
    ui5 = frame_vis.step(th, CLASS_MOB, CLASS_PLAYER, theme, now=104.0)
    check(ui5.p_color == (44, 55, 66) and ui5.m_color == (11, 22, 33),
          "④ 类号变了、颜色没跟着变（换了权重之后画框颜色会串 ✗）：%r" % ((ui5.p_color, ui5.m_color),))
    # ⑤ 缺省口径
    check(ui5.lock_width == 3 and ui5.attack_width == 2 and ui5.min_attack_width == 2
          and ui5.jump_attack_width == 2 and ui5.chase_jump_width == 2,
          "⑤ 线宽缺省值变了（原来是 锁定 3 / 其余 2 ✓ 观感会变 ✗）：%r"
          % ((ui5.lock_width, ui5.attack_width, ui5.min_attack_width),))
    check(ui5.on_lock and ui5.on_attack and ui5.on_vision,
          "⑤ 显示开关缺省值变了（原来缺键 = **画** ✓ 现在变成不画 ✗）")


def t_frame_vt_stage():
    """⭐⭐⭐⭐ 第 2 段（投帧给测谎）的**进出条件**（`runtime/frame_vt.py` ✓ 2026-10-20 搬出）。

    这段只有三行，但每条都有现场账 —— 搬的时候最容易"顺手简化"掉其中一条 ✗：
      ① **两个开关都要能进**（`_vt_want` 或 `_vt_on`）：只看 `_vt_want` ⇒ 关开关那一下
         走不到 `_vt_beat` ⇒ `_vt_ensure(False)` 不执行 ⇒ **线程收不掉**（下一局还在喂旧帧 ✗）；
      ② **原点优先"跟着窗口走"**（`origin_ref`）：窗口一挪还用常量原点 ⇒ 喂进去的画面坐标
         全偏 ⇒ 瞄准跟着歪（用户 2026-10-10 ✓）；
      ③ **原样投原生帧**（同一个对象 ✓）：换成 `vis` / 拷一份都会破坏"SDK 输入不带人画框"
         或者白花一次 memcpy ✓。

    钉五条：两个开关各一次 · 全关不投 · `origin_ref` 优先 · 兜底原点 · 帧对象 identity ✓。
    """
    from types import SimpleNamespace

    from runtime import frame_vt

    class _Th(SimpleNamespace):
        def __init__(self):
            super().__init__()
            self.got = []

        def _vt_beat(self, frame, ox, oy):
            self.got.append((id(frame), ox, oy))

    raw = object()                       # 拿 identity 判"是不是原样投进去的" ✓
    th = _Th()
    th._vt_want, th._vt_on = True, False
    check(frame_vt.step(th, raw, None, (1, 2)) is True and th.got[-1][1:] == (1, 2),
          "`_vt_want` 开着却没用兜底原点投帧 ✓：%r" % (th.got[-1:],))
    th._vt_want, th._vt_on = False, True
    check(frame_vt.step(th, raw, (11, 22), (1, 2)) is True and th.got[-1][1:] == (11, 22),
          "① 只剩 `_vt_on` 时不上报 ⇒ 关开关那一下收不掉线程 ✗：%r" % (th.got[-1:],))
    check(th.got[-1][0] == id(raw),
          "③ 投进去的不是**原来那份原生帧**（拷过 / 换成 vis 都会毁 SDK 输入 ✗）")
    th._vt_want, th._vt_on = False, False
    _n = len(th.got)
    check(frame_vt.step(th, raw, (11, 22), (1, 2)) is False and len(th.got) == _n,
          "两个开关都关着还在投帧（白喂 + 线程收不掉 ✗）")


def t_frame_stage_order():
    """⭐⭐⭐⭐ **一帧的阶段顺序**（R2「一帧一个顺序」✓ 阶段 1「编排收口」的判据）。

    为什么值得单独一条（`docs/架构-AI三层.md` R2 ✓）：这几段之间**没有类型能保证顺序** ——
    全靠人记住"先留像素、再判界面、再画、再决策" ✓。顺序错了**不会报错**，只会静默地用坏数据：
      · 先画后拷 ⇒ 原生帧带着我们的框/线（框选小地图会把虚线框进去 ✓ 2026-09-27 现场）；
      · 先画后判界面 ⇒ 整帧模板匹配被自己的绘制污染（漏报/误报都来 ✓）；
      · 先决策后留像素 ⇒ 决策读到的是**画过**的帧（黄点/血条识别全歪 ✓ 同一类坑 ✓）。

    钉的是**接线顺序**（源码位置 ✓ 行为级验不了：要真起线程 + YOLO ✓）：
      ① `frame_pic.step(self, ctx)` ⇒ ② `raw = ctx.raw`（原生帧从**盒子**拿，不许再自己拷 ✗）
      ⇒ ③ `frame_ui.step(self, raw)` ⇒ ④ `frame_vt.step(self, raw` ⇒ ⑤ `self._decide_frame_on_infer(`
      ⇒ ⑥ `self._draw_show_frame(` ⇒ ⑦ `self.raw_frame_ready.emit(raw)`。

    ⚠ ④ 那个"投帧"已经搬进 `runtime/frame_vt.py`（2026-10-20 ✓）⇒ 这里钉的是**接线点** ✓
      （`_run` 里调它）；"喂的必须是 `raw`"那条在 `frame_vt.py` 里、由 `t_vt_*` 那条用例管 ✓。

    ⚠ **画框在决策后面**（2026-10-20 实测顺序 ✓ 我第一版判据把它写在决策前面、当场红 ✗）：
      画的是"这一拍决策完了"的画面（框 + 攻击线 + 视野虚线），决策本身读的是**没画过**的帧 ✓
      —— 别看着"画框"就以为它在前面 ✓。
    """
    src = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    steps = [
        ("① 留原生帧 + 裁面板", "frame_pic.step(self, ctx)"),
        ("② 取回原生帧", "raw = ctx.raw"),
        ("③ 判界面状态", "frame_ui.step(self, raw)"),
        ("④ 喂测谎（原生帧）", "frame_vt.step(self, raw"),
        ("⑤ 决策", "self._decide_frame_on_infer("),
        ("⑥ 画框", "self._draw_show_frame("),
        ("⑦ 发原生帧（取帧测量的功能读它）", "self.raw_frame_ready.emit(raw)"),
    ]
    pos = []
    for name, key in steps:
        i = src.find(key)
        check(i > 0, "%s：`_run` 里找不到 `%s`（阶段搬完没接上 / 顺序判据该改了 ✗）"
                     % (name, key))
        pos.append((name, i))
    for (n1, i1), (n2, i2) in zip(pos, pos[1:]):
        check(i1 < i2, "阶段顺序错了：**%s** 排到了 **%s** 后面（静默用坏数据 ✗ 见本文说明）"
                       % (n1, n2))
    # 原生帧只能来自盒子：搬完之后**不许**再有一份自己的拷贝 ✗（两份迟早不同步 ✓）
    check("raw = vis.copy()" not in src,
          "`_run` 里又出现了 `raw = vis.copy()`（该由 `frame_pic.step` 负责 ✓ 两份拷贝迟早不一致 ✗）")


def t_pic_dead_watchdog():
    """⭐⭐⭐⭐ 画面断了（抓屏/推流全黑）⇒ **明确报 + 停录 + 判定不冤枉**
    （用户 2026-10-20 ✓ 原话："刚才测谎**鼠标没有跟上准心的位置，也没有录屏**"）。

    现场（`behavior.log` 14:09，四个症状一条根）：
      ```text
      14:09:31  测谎起、录像开始（lie_rec ok path=…140931_116.mp4）
      14:09:44  抓屏**全黑** ⇒ cam_* 数值 8 秒不变、mmap_pick 断流
      14:09:47  link_auto_reset … why=link-dead（输入链路也断）
      14:09:51  player_dead（盲了 7 秒）
      14:09:52  lie_rec_src_black frames=463（**只有这一句**，还埋在日志里 ✗）
      ```
      录像后 8 秒纯黑（ROI 均值 0.19）⇒ 用户以为"没录屏"✗；
      画面没了 ⇒ 准心跟踪全停 ⇒ 鼠标"没跟上准心"✗；
      30 秒窗口照旧判"失败"⇒ 冤枉（其实什么都没看见 ✗）。

    钉五条：
      ① 黑够 `PIC_DEAD_S` ⇒ 记**一条** `pic_dead` + **停录**（不写黑片子 ✓）；
      ② 一直在黑 ⇒ **不许刷第二条** `pic_dead`（每段一条 ✓）；
      ③ 亮回来 ⇒ `pic_alive`，且断的时长累加到 `_lie_pic_dead_s` ✓；
      ④ **正常的暗场景（均值 40）不许误报** ✓（地牢/夜晚都在 30~60 ✓）；
      ⑤ 窗口到点若"断过 ≥ `LIE_PIC_DEAD_JUDGE_S`" ⇒ 判 **unknown**（`ok=None`），
         没断过 ⇒ 照旧 `fail` ✓（老行为不变 ✗）。
    """
    import numpy as np

    from gui import live_thread as lt

    _bev = []
    _old_ev = lt.behavior.event
    lt.behavior.event = lambda name, **kw: _bev.append((name, kw))
    try:
        t = lt.LiveThread.__new__(lt.LiveThread)
        t._screen_state = "lie_game"
        t._lie_rec = object()                    # 假装在录 ⇒ `_pic_watch` 该停它 ✓
        t._lie_rec_pending = 1.0
        # ⚠ QThread 子类用 `__new__` 绕过 `__init__` 时，**未设置的属性**用 `getattr` 取
        #   会抛 `RuntimeError`（PyQt 的 `__getattr__` ✓ 不是 AttributeError ✗）⇒ 该有的先摆上 ✓
        t._pic_black_t0 = -1.0
        t._pic_dead = False
        t._lie_pic_dead_s = 0.0
        _stopped = []
        t._lie_rec_stop = lambda why="": _stopped.append(str(why))

        _black = np.zeros((96, 96, 3), np.uint8)
        t._pic_watch(_black, 0.0)
        t._pic_watch(_black, 0.4)
        check(not any(n == "pic_dead" for n, _ in _bev),
              "才黑 0.4 秒就报「断了」（一次眨眼就报警 ✗）：%r" % (_bev,))
        t._pic_watch(_black, 1.0)
        check(sum(1 for n, _ in _bev if n == "pic_dead") == 1,
              "黑够 %.1f 秒还没报 `pic_dead`（用户只会看到「没录屏」✗）" % lt.LiveThread.PIC_DEAD_S)
        check(bool(_stopped) and "画面断了" in _stopped[0],
              "画面断了却**继续录**（写出 30 秒黑片子 ⇒ 用户以为没录屏 ✗）：%r" % (_stopped,))
        check(bool(getattr(t, "_pic_dead", False)), "断了没记状态 ⇒ 判定那边用不上 ✗")

        t._pic_watch(_black, 1.6)                # 一直在黑
        t._pic_watch(_black, 2.2)
        check(sum(1 for n, _ in _bev if n == "pic_dead") == 1,
              "同一段黑里刷了多条 `pic_dead`（一秒一条把日志刷满 ✗）")

        _bright = np.full((96, 96, 3), 40, np.uint8)
        t._pic_watch(_bright, 2.6)
        check(any(n == "pic_alive" for n, _ in _bev), "亮回来了没说一声 ✗")
        check(t._pic_dead is False, "亮回来了还挂着「断了」⇒ 后面全被拦 ✗")
        check(1.0 <= float(t._lie_pic_dead_s) <= 3.5,
              "断的时长没累计（判定要用它减用时 ✓）：%r" % (t._lie_pic_dead_s,))

        # ④ 暗场景不误报
        _bev2 = _bev[:]
        t2 = lt.LiveThread.__new__(lt.LiveThread)
        t2._lie_rec = None
        t2._lie_rec_pending = 0.0
        t2._pic_black_t0 = -1.0
        t2._pic_dead = False
        t2._lie_pic_dead_s = 0.0
        for i in range(10):
            t2._pic_watch(np.full((96, 96, 3), 40, np.uint8), float(i))
        check(not any(n == "pic_dead" for n, _ in _bev[len(_bev2):]),
              "正常暗场景（均值 40）被当成断流了（地牢/夜晚就是这亮度 ✗）")

        # ⑤ 判定：断过 ⇒ unknown；没断 ⇒ 照旧 fail
        def _mk(dead_s):
            o = lt.LiveThread.__new__(lt.LiveThread)
            o._screen_state = "lie_game"
            o._lie_rec = None
            o._lie_rec_pending = 0.0
            o._lie_rec_saw_success = False
            o._lie_sess_t0 = 100.0
            o._lie_warn_at = 100.0
            o._lie_seen_game = True
            o._lie_rec_top_game = 0.9
            o._lie_rec_stop_at = 0.0
            o._lie_rec_last_lie = 0.0
            o._lie_pic_dead_s = float(dead_s)
            o._pic_dead = False
            o._pic_black_t0 = -1.0
            o._lie_last_verdict = ""
            o._lie_last_verdict_at = 0.0
            o._lie_last_verdict_s = 0.0
            o._lie_rec_stop = lambda why="": None
            o.vt_status = type("S", (), {"emit": lambda self, *_a: None})()
            return o

        _bev.clear()
        _mk(5.0)._lie_window_over(30.0)
        _v = [kw for n, kw in _bev if n == "lie_verdict"]
        check(bool(_v) and _v[0].get("ok") is None and _v[0].get("verdict") == "unknown",
              "断过 5 秒还判「失败」⇒ 冤枉人（那是没看见，不是没操作 ✗）：%r" % (_v,))
        _bev.clear()
        _mk(0.0)._lie_window_over(30.0)
        _v2 = [kw for n, kw in _bev if n == "lie_verdict"]
        check(bool(_v2) and _v2[0].get("ok") is False,
              "没断过却不判「失败」了（老行为被改坏 ✗）：%r" % (_v2,))

        # 接线：看门狗必须真的接在帧循环里、且排在录制前面
        _src = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
        _i_w = _src.find("self._pic_watch(raw, now)")
        _i_r = _src.find("self._lie_rec_write(raw, now)")
        check(_i_w > 0 and 0 < _i_w < _i_r,
              "看门狗没接进帧循环 / 排在录制后面（断了这一拍还会往里写 ✗）")
    finally:
        lt.behavior.event = _old_ev


TESTS = TESTS + (
    ("⭐⭐⭐⭐ 第 7 段（玩家定位）**真跑**：脚底偏移 · 面积闸 + 两个关的口子 · "
     "live/收流两条取数路 · 世界坐标喂 tracker（解不出给 None）· 相机三件", t_frame_locate_stage),
    ("⭐⭐⭐⭐ 第 6 段（一次推理 + 谁进决策）**真跑**：测谎中不推理 · 整帧/剪裁/退回整帧 · "
     "偏移加回 · 四路产物各归各位 · 掉落只挑掉落类 · 耗时封顶", t_frame_infer_stage),
    ("⭐⭐⭐⭐ 第 5 段（断线重连）五个出口：没动作不发 · tap 只按键 · home 只撞角不点 · "
     "click 走 _aim_geo 折回整帧 + homed · click2 只补左键 · scroll 带坐标", t_frame_reconnect_stage),
    ("⭐⭐⭐⭐ 第 4 段（探针解码）四类账分得开：关着不跑 · miss · invalid（几何错不是对时）· "
     "ok+进窗口 · reject 不进窗口 · 几何变才重置判据 · 窗口封顶 120", t_frame_probe_stage),
    ("⭐⭐⭐⭐ 第 3 段（可视化配置热读）的口径：首拍必算 · 1 秒内同一份对象 · 到点换新 · "
     "类号变了跟着变 + 查不到回退 · 线宽/开关缺省（锁定 3 / 其余 2 / 缺键=画）",
     t_frame_vis_stage),
    ("⭐⭐⭐⭐ 第 2 段（投帧给测谎）的进出条件：两个开关各一次 · 全关不投 · "
     "`origin_ref` 优先 · 兜底原点 · 帧对象 identity", t_frame_vt_stage),
    ("⭐⭐⭐⭐ 一帧的阶段顺序（R2）：留像素 → 取回 → 判界面 → 喂测谎 → 画框 → 决策，"
     "且原生帧只许来自盒子（`ctx.raw`）", t_frame_stage_order),
    ("⭐⭐⭐⭐ 画面断了（全黑）⇒ 记账 + 停录 + 判定不冤枉（用户 2026-10-20"
     "「鼠标没跟上准心、也没有录屏」）：一条 pic_dead · 不刷屏 · 亮回来报 alive · "
     "暗场景不误报 · 断过判 unknown", t_pic_dead_watchdog),
    ("⭐⭐⭐⭐ 远程这套「鼠标没跟上准心」要查得出来（用户 2026-10-20「是远程」）："
     "aim_health 报撞角次数/原因/采样新旧 · 没采样才撞角且说人话 · 有新鲜采样不许撞角 · "
     "走位线程按 link_health 暂停/恢复并记 vt_aim_home", t_aim_health_and_link_gate),
    ("⭐⭐⭐⭐ 面板像素变了但框选区域没变 ⇒ 标定按比例换算、不用重标（用户 2026-10-20 方案 A）："
     "等比缩放对 · 真重框不救 · 老标定不救 · 没帧尺寸不救 · 两条链都真的传 frame_wh",
     t_panel_adapt_by_ratio),
    ("⭐⭐⭐ 标定与面板尺寸不符要说「去重标定」（天空「永远认不出黄点」）："
     "两个尺寸都要说 · 差 1px 放行 · 界面必须先认 panel_mismatch",
     t_panel_mismatch_says_why),
    ("⭐⭐⭐ 「大怪优先」打开项目要三个控件一起回填（勾选框漏了 = 看得到用不上）："
     "口径对 · 老配置兜底不变 · 回填必须 setChecked + 刷可配状态",
     t_big_mob_ui_sync),
    ("⭐⭐⭐⭐ 远程桌面会话里本机 ProMicro 打不进游戏（HID 只进控制台会话）："
     "同会话不提示 · 跨会话说清两个会话号 · 判不了不拦 · 界面必须真的接上",
     t_hid_session_guard),
    ("⭐⭐ 怪框上限 `-1` = 不限（界面下限 / 两处翻译 / 缺键兜底 / tooltip 都要对）",
     t_max_det_minus_one_is_unlimited),
    ("⭐⭐⭐⭐⭐ 测谎「喂帧预算」（界面认不出小游戏窗口时兜住整场；SUCCESS/ENDED 立刻收）",
     t_vt_arm_window),
    ("⭐⭐⭐⭐ 测谎先嗅探（B 语义）· 触发即暂停战斗检测直到窗口消失（可 `vt_gate: off` 退回）",
     t_vt_gate_and_pause),
    ("⭐⭐⭐⭐ 测谎在跑 ⇒ 主回路主动让路（`vt_yield_pipe_fps`：默认免打扰 · 不限档不踩 min(0) 坑）",
     t_vt_yield_pipe_fps),
    ("⭐⭐⭐⭐ 推理门槛必须低到「低分档」那一层（否则防抖 / 续命整条链是死的）",
     t_infer_conf_feeds_low_band),
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
