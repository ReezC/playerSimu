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
    check(LINE_COPY in src,
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
                                 src_lag_watchdog)

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
    check(preview_fps_target(50.0, 60.0, "负载告警：内存 2.2/31.8GB", False, 0.2, pv) == 25.0,
          "有负载告警也没降档（那正是「为什么变慢」的解释 ✗）")
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

    t._lie_rec_start()
    check(t._lie_rec_pending > 0.0, "触发（进 lie_*）没起录")
    t._lie_rec_stop(why="单元")
    check(t._lie_rec_pending == 0.0 and t._lie_rec is None, "停录没清状态")

    _d = tempfile.TemporaryDirectory()
    _old = cfg.record_dir
    cfg.record_dir = lambda: Path(_d.name)
    try:
        t._lie_rec_start()
        _raw = _frame(7)
        t._lie_rec_write(_raw, _time.monotonic())
        t._lie_rec_write(_raw, _time.monotonic())
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
    check(src.index("_engine_imgsz_from_error(_e)") < src.index("res = model.predict(vis"),
          "对齐写在**主循环那次推理之后** ✗ ⇒ 报错照样把线程收掉 ✓")
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
    src = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
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
    check("ws.player.dead = bool(_pot_vals[0] <= 0.0)" in src,
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
    src = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    check("drop_touch(player_box or _last_player[0]," in src,
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
    check("def _draw_box(ad, color, on, from_d=0.0, width=2, dashed=False):" in src,
          "`_draw_box` 没有 `dashed` 这个口子（那虚线就得另写一条画框的路 ⇒ 形状会分叉 ✗）")
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
    from perception import classes as pc

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
            check("ws.mobs = mob_tracker.update(mob_dets)" in src,
                  "怪那条链不吃 `mob_dets` 了 —— 画框那份（只给显示 ✓）可能混进了决策 ✗")
            check("mob_tracker.update(_cls_dets)" not in src
                  and "PlayerTracker(_cls_dets" not in src,
                  "**画框那份**被喂给 tracker 了（= 掉落/宠物进决策 ✗ 用户 2026-10-04 明确否掉）")
            check("extra_class_dets(xyxy, cfs, clss, _extra_map)" in src,
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


TESTS = (
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
    ("⭐⭐ 采集：选中文件后显示预估张数（公式与 extract_frames 一份 + MKV 容器时长兜底）",
     t_capture_est_frames),
    ("⭐⭐ 开自动前体检：标定/地形没凑齐要先弹窗说清（用户 2026-09-29；⚠ 光有函数没接上=没有）",
     t_auto_precheck),
    ("⭐⭐⭐ 断线重连回到游戏 ⇒ 体检通过才恢复自动（用户 2026-10-07 选「体检」；"
     "不过就不恢复 + 写清原因）", t_reconnect_resume_gate),
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
    ("⭐⭐⭐ 导出成功后**当场接管新引擎**（旧 `<名字>.engine` 会被改名走 ⇒ 不许再捏着它 ✗）",
     t_export_done_takes_over),
    ("⭐⭐ 权重路径失效时要说清「怎么办」（别只甩一行 does not exist）",
     t_missing_weights_msg),
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
