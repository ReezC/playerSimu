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
    check("_chase_jump_color" in lsrc and "_on_chase" in lsrc,
          "「追击起跳框」没用设置里的颜色/开关 ⇒ 又写死回 live_thread 了 ✗")
    check("(0, 200, 0)" not in lsrc,
          "live_thread 里还有写死的那个绿（追击起跳的颜色该只在 `gui/theme.py` 一处 ✗）")
    check("jump_attack_color" not in lsrc,
          "「跳跃攻击范围框」被画出来了 —— 它的逻辑还没做、尺寸都没定义 ⇒ 不该画 ✗")

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

    # ① 先拷再画（画是**原地**改数组 ⇒ 拷晚了原始像素就回不来了）
    LINE_COPY = "raw = vis.copy() if self._infer.is_set() else vis"
    i_copy = src.find("raw = vis.copy()")
    check(i_copy > 0, "没找到 `raw = vis.copy()` —— 原生帧没留（框选会拿到画过的帧）")
    draws = [src.find(kw) for kw in ("cv2.rectangle(vis,", "cv2.line(vis,",
                                     "cv2.putText(vis,", "cv2.circle(vis,",
                                     "cv2.arrowedLine(vis,", "_draw_dashed_line(vis,")]
    draws = [j for j in draws if j >= 0]
    check(draws, "一个往 `vis` 上的绘制都没找到（源码约定该改写了？）")
    check(i_copy < min(draws),
          "原生帧是在画过之后才拷的（`vis` 是原地改的，那时原始像素已经没了 ✗）")

    # ② 拷贝条件必须**就写在那一行上**：它假设"所有绘制都在下面那道推理门里"
    check(LINE_COPY in src,
          "原生帧的拷贝条件不是那一行（`%s`）—— 换了条件就等于假设「别处也会画」✗"
          % LINE_COPY)
    i_infer = src.find("if self._infer.is_set():")
    check(i_infer > 0, "没找到推理那道门（`if self._infer.is_set():`）")
    for kw in ("cv2.rectangle(vis,", "cv2.line(vis,", "cv2.putText(vis,",
               "cv2.circle(vis,", "cv2.arrowedLine(vis,", "_draw_dashed_line(vis"):
        j = src.find(kw)
        check(j == -1 or j > i_infer,
              "%s 跑到 `if self._infer.is_set():` 外面了 —— 推理关着时原生帧会被画脏 ✗"
              % kw)

    # ③ 原生帧的 emit 不受显示限流：它必须**先发**，而且缩进比显示那句**浅**
    #    （显示那句在 `if should_show:` 里面 ⇒ 缩进更深 ✓；比它深就是被限流包住了 ✗）
    i_emit_raw = src.find("self.raw_frame_ready.emit(raw)")
    i_emit_disp = src.find("self.frame_ready.emit(vis)")
    check(i_emit_raw > 0, "没发原生帧信号（`raw_frame_ready`）—— 框选还是拿到显示帧")
    check(0 < i_emit_raw < i_emit_disp,
          "原生帧的 emit 不在显示帧之前（面板要先拿到原生帧 ✓）")

    def _indent(text, pos):
        b = text.rfind("\n", 0, pos) + 1
        line = text[b: text.find("\n", pos)]
        return len(line) - len(line.lstrip())

    check(_indent(src, i_emit_raw) < _indent(src, i_emit_disp),
          "原生帧的 emit 缩进比显示那句还深 ⇒ 被 `if should_show:` 包住了："
          "取帧会跟着显示限流一起变旧 ✗")
    check("self.frame_ready.emit(vis)" in src,
          "显示帧的 emit 没了（画面就黑了）")


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
      · ② 锁定框那块**不许**再出现 `current_target_sets(...)` 的**调用** ✓（那份缓存本身也
        随之删掉了 ✗ —— "死数据不留"）。

    ⚠ 为什么钉这么细：绘制那一支要是自己调 `mob_sets_of` / `foothold_below`
      ⇒ **每帧扫几百条 foothold** ✗（纪律原文：只对当前那一只调用、每拍最多一次 ✓）。
      所以两处都按**源码**钉（这一段没有可驱动的最小夹具 ✓）。
    """
    src = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")

    # ① 查过的怪框：**框下方靠左**
    i = src.index("queried_mob_boxes()")
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

    钉三件：
      ① 没选地图 ⇒ 只报「没选地图」并**立刻返回**（不再往下判 ✓）；
      ② 地图不存在 ⇒ 报「没标定」+「没地形」，**两条都要带"→ 去哪修"** ✓；
      ③ ⭐ **源码级**：`_toggle_auto` 里真的在"开"之前调了体检、且不通过会**把按钮拨回去** ✓
         —— ⚠ 这条最要紧：**光有体检函数、没接上就等于没有** ✗。
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

    _src = (Path(__file__).resolve().parent.parent
            / "gui" / "player_panel.py").read_text(encoding="utf-8")
    check("if on and not self._precheck_auto():" in _src,
          "`_precheck_auto` 没接进 `_toggle_auto` ⇒ 体检等于没有（用户点了开、条件不齐、"
          "照样开 ⇒ 过几分钟自己停）✗")
    check("def _precheck_problems(" in _src and "@staticmethod" in _src,
          "判据没抽成 `@staticmethod` ⇒ 没法像这条用例这样单测 ✗")
    check("QMessageBox.warning(" in _src and "开自动前的检查没通过" in _src,
          "没弹窗 / 弹窗标题丢了 ⇒ 用户不知道发生了什么 ✗")


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


TESTS = (
    ("⭐⭐ 采集：选中文件后显示预估张数（公式与 extract_frames 一份 + MKV 容器时长兜底）",
     t_capture_est_frames),
    ("⭐⭐ 开自动前体检：标定/地形没凑齐要先弹窗说清（用户 2026-09-29；⚠ 光有函数没接上=没有）",
     t_auto_precheck),
    ("连推 10 帧只画最新那帧（合并，不排队）", t_coalesce),
    ("不可见时一帧都不画，但帧仍是最新的", t_hidden_skips_draw),
    ("视野框（虚线）：每帧现算，不许再套「每 N 秒算一次」的节流", 
     t_vision_box_updates_every_frame),
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
    ("显示链路：带 padding 的帧画得对，且不再白拷一整幅", t_pixmap_handles_padded_frame),
    ("静态检查：会当场炸的名字错误（pyflakes）", t_static_check_no_crash_classes),
    ("左下按键帽：按住=半透明绿填充、文案随键位映射自适应（2026-09-30 用户要求）",
     t_key_caps_overlay),
    ("怪框那行地点：「查过的怪框」写在框下方靠左；锁定框不再写地点（绘制层不许自己扫 foothold）",
     t_mob_box_labels),
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
