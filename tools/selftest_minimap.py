"""小地图标定自检：定位往返 + 标定弹窗（离屏 + 假帧，不连 A 机、不碰真实标定）。

**为什么要它**
    标定量出来的几个数（scale / offset / view）直接决定「我在哪块平台上」——
    偏一点就是「人在这块平台，程序以为在另一块」。而且这套东西在实验室里
    没有 A 机的流也能验：拿 `datasets/map/<id>.png`（底图）**合成**一张
    「面板画面」，再用 locate 反推，看能不能还原出当初合成时用的几何。

    弹窗那部分同样能脱机验：塞一个**假收帧器**（stub）进去，不碰网络；
    保存标定也走不到 —— 只验它算出来的标定字典，不写 datasets/ 里的真文件。

跑法：
    python -m tools.selftest_minimap      # 全过返回 0，有失败返回 1
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2                                                  # noqa: E402
import numpy as np                                          # noqa: E402

from core import mapdata                                    # noqa: E402
from perception import minimap as mm                        # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

#: 优先用这张（寻路正在做的图）；它不在就随便找一张有底图的
PREFER = "100040102"


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def pick_map():
    """找一张**有底图**的地图 → (map_id, 底图 numpy)。找不到返回 (None, None)。"""
    names = [PREFER] + [p.stem for p in sorted(mapdata.map_dir().glob("*.png"))]
    for mid in names:
        t = mapdata.load(mid, with_canvas=True)
        if t is not None and t.canvas is not None:
            return t.id, t.canvas
    return None, None


def synth_fit_panel(canvas, scale, x, y, pad=40, bg=(28, 28, 28)):
    """合成一张「全局小地图」面板：整张底图放大 scale 倍，摆在 (x, y)。"""
    h = int(round(canvas.shape[0] * scale))
    w = int(round(canvas.shape[1] * scale))
    big = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_NEAREST)
    panel = np.full((h + 2 * pad, w + 2 * pad, 3), 0, np.uint8)
    panel[:] = bg
    panel[pad:pad + h, pad:pad + w] = big
    return np.ascontiguousarray(panel)


def synth_crop_panel(canvas, x, y, w, h):
    """合成一张「局部小地图」面板：底图上 (x, y) 起、面板那么大的一块。"""
    return np.ascontiguousarray(canvas[y:y + h, x:x + w])


# ---------------------------------------------------------------- 定位往返

def t_fit_roundtrip():
    """fit：合成 s=2.0/@(pad,pad) 的面板 → locate_fit 应当还原出这个 s 与偏移。"""
    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return
    pad = 40
    panel = synth_fit_panel(canvas, 2.0, pad, pad)
    loc = mm.locate_fit(panel, canvas)
    check(loc is not None, "fit 定位没对上（合成帧应当必中）")
    check(abs(loc["scale"] - 2.0) < 0.26,
          "拟合出的缩放 %.3f 离 2.0 太远" % loc["scale"])
    check(abs(loc["offset"][0] - pad) <= 2 and abs(loc["offset"][1] - pad) <= 2,
          "拟合出的偏移 %s 离 (%d, %d) 太远" % (loc["offset"], pad, pad))
    check(loc["score"] > 0.9, "匹配分只有 %.3f，合成帧应当接近 1" % loc["score"])


def t_crop_roundtrip():
    """crop：合成 (x, y) 起的一块 → locate_crop 的 view 与 panel_to_canvas 要对上。

    ⚠ inset 的约定：模板是从面板里**裁掉 inset 再匹配**的，所以 locate_crop 返回的
    view 是「底图坐标 + inset」；换算时 (panel - inset) + view 正好抵消 —— 这里
    连带把这条约定一起钉住（以前这里多减一次 inset，整体偏 4 像素 = 64 世界像素）。
    """
    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return
    ch, cw = canvas.shape[:2]
    w, h = min(60, cw - 10), min(100, ch - 10)
    if w < 20 or h < 20:
        print("      （底图太小，跳过）")
        return
    x, y = 9, 40
    panel = synth_crop_panel(canvas, x, y, w, h)
    loc = mm.locate_crop(panel, canvas)
    check(loc is not None, "crop 定位没对上（合成帧应当必中）")
    inset = int(loc["offset"][0])
    check(loc["view"] == [x + inset, y + inset],
          "view %s 应该是 [%d, %d]" % (loc["view"], x + inset, y + inset))
    # 真正要紧的是这一条：面板左上角换回底图坐标 = 当初裁剪的原点
    cx, cy = mm.panel_to_canvas(0, 0, loc)
    check((cx, cy) == (x, y),
          "panel_to_canvas 换回 (%s, %s)，应该是 (%d, %d)" % (cx, cy, x, y))


def t_crop_scale_not_faked():
    """1:1 裁块量出来的 **scale 必须是 1.0**（曾经会报一个假的 0.54）。

    **为什么单列**：`_crop_scales` 会补一档「面板刚好装下底图」的比例；当它落在
    [0.5, 1) 时（例：底图 78×203、面板 70×110 → 0.542），老代码 `step = max(1,
    round(0.542)) = 1` → **模板根本没缩**，分数照样 1.0，却把 scale 记成 0.542 ——
    view 是对的、scale 是错的，于是「面板左上 → 世界」整体偏 1/s 倍。这种"分数满分
    却错一倍"的东西最难发现，所以用几种面板尺寸和位置把它钉住。
    """
    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return
    ch, cw = canvas.shape[:2]
    for (w, h, x, y) in ((70, 110, 7, 11), (60, 100, 9, 40), (70, 110, 0, 0)):
        if x + w > cw or y + h > ch:
            continue
        panel = synth_crop_panel(canvas, x, y, w, h)
        loc = mm.locate_crop(panel, canvas)
        check(loc is not None, "(%dx%d)@(%d,%d) 没对上" % (w, h, x, y))
        check(abs(loc["scale"] - 1.0) < 1e-6,
              "(%dx%d)@(%d,%d) 是 1:1 裁块，scale 应当是 1.0，实际 %.4f"
              % (w, h, x, y, loc["scale"]))
        got = mm.panel_to_canvas(0, 0, loc)
        check(got == (x, y),
              "(%dx%d)@(%d,%d) 换算回 (%s, %s)，应该是 (%d, %d)"
              % (w, h, x, y, got[0], got[1], x, y))


def t_magnified_crop_roundtrip():
    """crop 的「放大后取块」：面板 = 底图放大 s 倍后裁的一块 → 要量回 (s, view)。

    **为什么必须测**：实测猴林迷宫I 的底图只有 78×203，而面板 753×612 —— 光宽度
    就差 9.7 倍。客户端把小地图放大后再切块是常态，只按 1:1 找**永远量不出来**
    （人和程序都失败，表现是「匹配分 0.7 上下、怎么调都不对」）。
    """
    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return
    ch, cw = canvas.shape[:2]
    s_true = 9
    # 底图放大 9 倍（最近邻 = 客户端像素画放大的样子），再裁一块当面板
    big = cv2.resize(canvas, (cw * s_true, ch * s_true),
                     interpolation=cv2.INTER_NEAREST)
    ox, oy = 5 * s_true, 12 * s_true            # 裁的位置（底图坐标 × s）
    pw, ph = min(700, big.shape[1] - ox), min(600, big.shape[0] - oy)
    if pw < 40 or ph < 40:
        print("      （底图太小，跳过）")
        return
    panel = np.ascontiguousarray(big[oy:oy + ph, ox:ox + pw])

    loc = mm.locate_crop(panel, canvas)
    check(loc is not None, "放大取块的面板应当能定位（返回 None）")
    check(abs(loc["scale"] - s_true) <= 1.0,
          "量出的放大倍数 %.2f 与真实的 %d 差太多" % (loc["scale"], s_true))
    inset = int(loc["offset"][0])
    cx, cy = mm.panel_to_canvas(0, 0, loc)
    check(abs(cx - ox / s_true) <= 1.5 and abs(cy - oy / s_true) <= 1.5,
          "面板左上角换回底图坐标 (%s, %s)，应该是 (%d, %d)"
          % (cx, cy, ox // s_true, oy // s_true))
    check(loc["score"] > 0.9, "合成帧的匹配分只有 %.3f" % loc["score"])
    # 面板覆盖的底图范围：应当是「裁的那一块」那么大
    r_canvas, _r_over = mm.view_rects(loc, (pw, ph), (cw, ch))
    check(abs(r_canvas[2] - r_canvas[0] - pw / loc["scale"]) <= 2,
          "view_rects 算的底图覆盖宽度不对：%s" % (r_canvas,))


def t_wrong_mode_fails():
    """面板是 crop 那种一块、却按 fit 去量 → 应当**量不出来**（不是给个错值）。"""
    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return
    ch, cw = canvas.shape[:2]
    w, h = min(60, cw - 10), min(100, ch - 10)
    if w < 20 or h < 20:
        print("      （底图太小，跳过）")
        return
    panel = synth_crop_panel(canvas, 9, 40, w, h)
    loc = mm.locate(panel, canvas, mm.MODE_FIT)
    check(loc is None, "拿一块局部当整图量，居然量出来了：%s" % (loc,))


# ---------------------------------------------------------------- 弹窗（离屏）

class StubClient:
    """假收帧器：把合成帧当流喂给弹窗，不碰网络。"""

    def __init__(self, frame=None):
        self.frame = frame
        self.n_recv = 1
        self.connected = True
        self.err = ""
        self.fps = 10.0
        self.stopped = False
        self.started = False

    def start(self):
        self.started = True
        return self

    def stop(self):
        self.stopped = True

    def latest(self, clear=False):
        return self.frame, 1.0


def t_dialog_with_stub():
    """弹窗：塞假帧 → 自动定位 → 标定字典正确 → 拖动能反算 → 关窗处置收流。"""
    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return

    from PyQt5.QtWidgets import QApplication
    from gui.minimap_calib import MinimapCalibDialog
    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")

    ch, cw = canvas.shape[:2]
    w, h = min(60, cw - 10), min(100, ch - 10)
    x, y = 9, 40
    panel = synth_crop_panel(canvas, x, y, w, h)

    # 1) 用 crop（局部小地图）的方式量
    stub = StubClient(panel)
    dlg = MinimapCalibDialog(mid, mode=mm.MODE_CROP, client=stub)
    try:
        check(stub.started is False, "外部传进来的收帧器不该被弹窗 start")
        dlg.ck_live.setChecked(True)
        dlg._on_tick()                    # 假流 → 弹窗拿到帧
        check(dlg._frame is not None, "弹窗没取到帧")
        dlg._locate_now()
        check(dlg.score > 0.9, "弹窗自动定位的匹配分只有 %.3f" % dlg.score)
        cal = dlg.calib()
        check(cal["mode"] == mm.MODE_CROP, "标定字典里的方式不对：%s" % cal["mode"])
        check(cal["scale"] == 1.0, "crop 的缩放应当固定 1.0，实际 %s" % cal["scale"])
        cx, cy = mm.panel_to_canvas(0, 0, cal)
        check((cx, cy) == (x, y),
              "弹窗量出来的换算偏了：面板左上 → 底图 (%s, %s)，应该是 (%d, %d)"
              % (cx, cy, x, y))
        # 判据那行要显示世界坐标与世界范围（人靠它判断对不对）
        check("世界范围" in dlg.lbl_judge.text(),
              "判据行没显示世界范围：%r" % dlg.lbl_judge.text())

        # 2) crop 下缩放控件**必须可用**：小底图会被客户端放大后取块（实测到
        #    9~10 倍），自动定位失败时就靠它手动对齐 —— 以前这里禁用了，
        #    等于把唯一的手动出路堵死（缩放上限也只有 4 倍，够不着）。
        check(dlg.sld.isEnabled() and dlg.sp_scale.isEnabled(),
              "crop 下缩放控件必须是可用的（放大后取块的图靠它手动对）")
        check(dlg.sp_scale.maximum() >= 12.0,
              "缩放上限应当 ≥12 倍，实际 %.1f" % dlg.sp_scale.maximum())

        # 3) 手动拖动叠加层 = 改 view（这就是「手动目测」那条路）。
        #    拖动量按 scale 折算：挪「1 个底图像素」= 挪 scale 个面板像素。
        before = list(dlg.block)
        s = float(dlg.scale)
        dlg._ov_item.setPos(dlg._ov_item.pos().x() - 7 * s,
                            dlg._ov_item.pos().y() + 3 * s)
        check(dlg.block == [before[0] + 7, before[1] - 3],
              "拖动没反算回 view：%s（应当 %s）"
              % (dlg.block, [before[0] + 7, before[1] - 3]))

        # 4) 方向键微调：一格 = 1 个底图像素。方向语义 = **挪叠加层**
        #    （和拖动一致）：按左 → 叠加层左移 → 看到的是底图更靠右的一块。
        from PyQt5.QtCore import QEvent, Qt
        from PyQt5.QtGui import QKeyEvent
        v0 = list(dlg.block)
        dlg.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Left, Qt.NoModifier))
        check(dlg.block == [v0[0] + 1, v0[1]],
              "方向键没把叠加层挪 1 个底图像素：view=%s（应当 %s）"
              % (dlg.block, [v0[0] + 1, v0[1]]))
    finally:
        dlg._shutdown()
    check(stub.stopped is False, "借来的收帧器不该被弹窗 stop（只有自己连的才关）")

    # 5) 自己连的那条路：关窗要停掉，别留后台线程和占着的端口
    import unittest.mock as mock
    made = StubClient(panel)

    def _fake_client(host, port=5003, timeout=5.0):
        made.host, made.port = host, port
        return made

    with mock.patch.object(mm, "MiniMapClient", _fake_client):
        dlg2 = MinimapCalibDialog(mid, mode=mm.MODE_CROP)
        try:
            check(made.started is True, "自己连那条路应当 start 收帧器")
        finally:
            dlg2._shutdown()
    check(made.stopped is True, "自己连的收帧器在关窗时没被 stop")

    # 6) 没帧时的说法（A 机没推流时人看到的就是这句）
    empty = StubClient(None)
    dlg3 = MinimapCalibDialog(mid, mode=mm.MODE_CROP, client=empty)
    try:
        dlg3._on_tick()
        check("等帧" in dlg3.lbl_conn.text(),
              "没帧时的连接状态不明确：%r" % dlg3.lbl_conn.text())
        dlg3._locate_now()
        check("小地图推流" in dlg3.lbl_say.text(),
              "没帧时点定位没告诉人该去启动 A 机的推流：%r" % dlg3.lbl_say.text())
    finally:
        dlg3._shutdown()


def t_dialog_auto_fit():
    """全局小地图 + 「持续自动定位」：邻近尺度跟踪能对上，跟丢了要能整体重搜一次。

    这条走的是**另一条代码路径**（locate_fit 带 scales=）：搜索范围只有当前尺度
    ±2%，代价从"二十来次匹配"降到一次 —— 但跟丢时必须退回去整体搜，否则会
    一直卡在一个错的尺度上。
    """
    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return

    from PyQt5.QtWidgets import QApplication
    from gui.minimap_calib import MinimapCalibDialog
    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")

    # ⚠ pad 必须是 0：fit 的候选尺度是「面板刚好装下底图」那个比例附近的 ±10%
    # （点一次到位，不做全范围搜索）。留了边距几何比例就偏了，近邻搜不到真值 ——
    # 这是「自动定位只微调、不重搜」的设计代价，测试得按这个语义造数据。
    # 紧贴的整图面板（pad=0；注意第 3/4 个参数是**摆放位置** x,y，pad 是第 5 个）
    pad = 0
    panel = synth_fit_panel(canvas, 2.0, 0, 0, pad=pad)
    stub = StubClient(panel)
    dlg = MinimapCalibDialog(mid, mode=mm.MODE_FIT, client=stub)
    try:
        dlg.ck_auto.setChecked(True)
        # 第一次：当前 scale 还是个瞎猜的值 → 邻近尺度搜不到 → 应当退回整体搜索
        dlg._on_tick()
        check(abs(dlg.scale - 2.0) < 0.26,
              "自动定位（首次，需整体重搜）量出的缩放 %.3f 不对" % dlg.scale)
        check(abs(dlg.offset[0] - pad) <= 2 and abs(dlg.offset[1] - pad) <= 2,
              "自动定位量出的偏移 %s 不对" % (dlg.offset,))
        check("ms" in dlg.lbl_say.text(),
              "定位结果里没报耗时：%r" % dlg.lbl_say.text())

        # 第二次：换一帧（新对象），尺度已经对了 → 走邻近尺度那条路
        stub.frame = panel.copy()
        dlg._last_locate = 0.0          # 假装过了节流窗口
        dlg._on_tick()
        check(abs(dlg.scale - 2.0) < 0.26,
              "自动跟踪（邻近尺度）把尺度带偏了：%.3f" % dlg.scale)
        check(abs(dlg.offset[0] - pad) <= 2,
              "自动跟踪把偏移带偏了：%s" % (dlg.offset,))
    finally:
        dlg._shutdown()


def t_dialog_ui_feedback():
    """弹窗的"点了有没有反馈"三件事（都被人抱怨过）。

    1. 「抓一帧」按钮删掉 —— 它和"取消勾选实时画面"重复，没人知道它干啥；
    2. 状态行**不许**报累计帧数（只会一直变大的数字，看它没意义）；
    3. 「打开叠加图」必须**弹出来**（以前只把框写进文件就完了，人点下去
       看不见任何东西 —— 反馈只有状态行一行小字，很容易以为"没反应"）。
    """
    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return

    from PyQt5.QtWidgets import QApplication
    from gui.minimap_calib import MinimapCalibDialog
    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")

    ch, cw = canvas.shape[:2]
    w, h = min(60, cw - 10), min(100, ch - 10)
    if w < 20 or h < 20:
        print("      （底图太小，跳过）")
        return
    panel = synth_crop_panel(canvas, 9, 40, w, h)
    dlg = MinimapCalibDialog(mid, mode=mm.MODE_CROP, client=StubClient(panel))
    try:
        check(not hasattr(dlg, "btn_grab"), "「抓一帧」按钮应当删掉（与暂停重复）")

        # 叠加参照可以切到「地形叠加图」：几何字段一像素都不许动，只是换图层。
        # （客户端的小地图是按 foothold 现画时，人工对齐只能靠这张）
        geo0 = (dlg.scale, list(dlg.offset), list(dlg.block))
        j = dlg.cmb_ref.findData("terrain")
        check(j >= 0, "参照下拉里没有「地形叠加图」")
        dlg.cmb_ref.setCurrentIndex(j)
        dlg._on_ref()
        check((dlg.scale, list(dlg.offset), list(dlg.block)) == geo0,
              "换参照图不该改几何（scale/offset/view 要原样）")
        if dlg.ref == "terrain":
            check(dlg._ref_zoom > 1,
                  "地形叠加图比底图大，ref_zoom 应当是它的放大倍数")
            dlg._sync_overlay()
            check(abs(dlg._ov_item.scale() - dlg.scale / dlg._ref_zoom) < 1e-6,
                  "叠加层缩放要按 ref_zoom 折算：实际 %.4f"
                  % dlg._ov_item.scale())
            dlg._on_overlay_moved()          # 反算要能回到同一套几何
            check(abs(dlg.scale - geo0[0]) < 1e-6,
                  "换参照后反算的 scale 变了：%s → %s" % (geo0[0], dlg.scale))

        dlg._on_tick()
        check("已收" not in dlg.lbl_conn.text(),
              "状态行还在报累计帧数：%r" % dlg.lbl_conn.text())
        check("拍/秒" in dlg.lbl_conn.text() or "停住" in dlg.lbl_conn.text(),
              "状态行既没说速率也没说停住：%r" % dlg.lbl_conn.text())

        # 「自动定位」按钮：**同步一发即中**，和持续模式同一条计算 ——
        # 老实现跑"完整搜索"（几十秒），人看到的就是"点了没反应"。
        check(not hasattr(dlg, "btn_overlay"),
              "「打开叠加图」按钮应当去掉（看那张图走「叠加参照」下拉）")
        t0 = time.perf_counter()
        dlg._on_locate_clicked()
        cost = time.perf_counter() - t0
        check(cost < 2.0,
              "点一次「自动定位」用了 %.2f 秒 —— 又退回完整搜索了" % cost)
        check("正在算" in dlg.lbl_say.text() or "定位" in dlg.lbl_say.text()
              or "没对上" in dlg.lbl_say.text(),
              "点完没给任何结果/说明：%r" % dlg.lbl_say.text())
        # 手动与持续必须是同一条计算：候选倍数由同一个函数给（不各自搜一套）
        ns = dlg._near_scales()
        check(ns is None or len(ns) == 5,
              "近邻候选应当要么 None（crop 自己搜倍数）要么 5 个：%s" % (ns,))
        check(dlg._near_scales() == ns, "候选倍数两次调用不一致")

        dlg._locate_now()

        # 透明度拖动条：只改叠加层浓淡，几何一点不动，而且要存进标定
        geo1 = (dlg.scale, list(dlg.offset), list(dlg.block))
        dlg.sld_alpha.setValue(20)
        dlg._on_alpha(20)
        check(abs(dlg._ov_item.opacity() - 0.20) < 1e-6,
              "透明度没作用到叠加层：%.2f" % dlg._ov_item.opacity())
        check((dlg.scale, list(dlg.offset), list(dlg.block)) == geo1,
              "调透明度不该改几何")
        check(dlg.calib().get("alpha") == 20,
              "透明度没存进标定：%s" % dlg.calib().get("alpha"))

        # 那张整图（地形画在 WZ 素材上）现在从「叠加参照」下拉看，不再有按钮
        check(not hasattr(dlg, "btn_overlay"), "「打开叠加图」按钮应当已经去掉")
    finally:
        dlg._shutdown()


def t_save_then_reopen():
    """保存 → 重开：**手工对齐**的几何必须原样回来，且存下去的换算和对齐一致。

    **为什么单列一条**（现场现象：手动标完点「保存标定」→ 关窗 → 重开就"被还原"）：
      1. `_geom_ready` 原先看的是 `score` —— 手工对齐没有匹配分（score=0），
         重开时弹窗以为「从没标过」，把**已存好的几何覆盖**成粗略猜测；
      2. crop 模式下 `calib()` 存的是 `self.offset`（从文件读来的，可能是 [0,0]），
         而人工对齐用的是 `inset`(=4) —— 存下去的文件比实际对齐差 4 个底图像素，
         在 px_per_world≈16 的图上正好是 **63 世界单位**（用户报的就是这个差）。
      3. **弹窗从不说明这份几何是哪来的**：于是"还原成功"和"程序又猜了一个"在
         屏幕上一模一样，「我标过没有」没法自证。现场报的就是这个 —— 寺院通道2
         压根没有 `.mapcalib.json`，人却以为标定存过了。现在顶部单独一行写明，
         并且关窗时对「没保存的改动」问一句（`reject`），写文件失败也要报出来。

    全程用**临时标定文件**（patch `mapdata.calib_path`），不碰 datasets/ 里的真文件。
    """
    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return

    import shutil
    import tempfile
    import unittest.mock as mock

    from PyQt5.QtWidgets import QApplication, QMessageBox
    from gui.minimap_calib import MinimapCalibDialog

    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")

    ch, cw = canvas.shape[:2]
    w, h = min(80, cw - 16), min(120, ch - 16)
    if w < 40 or h < 40:
        print("      （底图太小，跳过）")
        return
    x, y = 7, 11
    panel = synth_crop_panel(canvas, x, y, w, h)

    tmpdir = Path(tempfile.mkdtemp(prefix="mmcalib_"))
    tmp = tmpdir / ("%s.mapcalib.json" % mid)
    try:
        with mock.patch.object(mapdata, "calib_path", lambda _mid: tmp):
            # ① 用户那条路：手动拖到正确位置（不做自动定位 → score 保持 0）
            stub = StubClient(panel)
            dlg = MinimapCalibDialog(mid, mode=mm.MODE_CROP, client=stub)
            try:
                check("还没标定过" in dlg.lbl_loaded.text(),
                      "第一次打开要写明「这张图还没标定过」（不然人分不清现在摆的"
                      "是上次存的还是程序猜的），实际：%r" % dlg.lbl_loaded.text())
                dlg._on_tick()
                check(dlg._frame is not None, "弹窗没取到帧")
                # crop 的放置约定：pos = inset - view * scale
                dlg._ov_item.setPos(dlg.inset - (x + dlg.inset),
                                    dlg.inset - (y + dlg.inset))
                check(list(dlg.block) == [x + dlg.inset, y + dlg.inset],
                      "拖出来的 view 不对：%s" % (dlg.block,))
                with mock.patch.object(QMessageBox, "question",
                                       return_value=QMessageBox.Yes):
                    dlg._on_save()
                check(dlg.saved, "点了「保存标定」却没标记 saved")
                check("已载入上次标定" in dlg.lbl_loaded.text(),
                      "存完之后顶部那行要立刻变成「已载入上次标定」：%r"
                      % dlg.lbl_loaded.text())
                check(not dlg._unsaved(), "刚保存完不该还报「有没保存的改动」")
            finally:
                dlg._shutdown()
            check(tmp.exists(), "保存后标定文件没生成：%s" % tmp)

            saved = mapdata.load_calib(mid) or {}
            check(saved.get("mode") == mm.MODE_CROP,
                  "存下来的方式不对：%s" % saved.get("mode"))
            check(saved.get("offset") == [dlg.inset, dlg.inset],
                  "crop 的 offset 必须写成 inset（模板内缩量），实际 %s —— 写成别的"
                  "值会让程序算出来的世界坐标和人工对齐差 inset 个底图像素"
                  % (saved.get("offset"),))
            got = mm.panel_to_canvas(0, 0, saved)
            check(got == (x, y),
                  "存下去的换算和人工对齐不一致：面板左上 → 底图 (%s, %s)，"
                  "应该是 (%d, %d)" % (got[0], got[1], x, y))

            # ② 重开：几何要原样回来（不能被「粗略猜测」覆盖）
            stub2 = StubClient(panel)
            dlg2 = MinimapCalibDialog(mid, mode=mm.MODE_CROP, client=stub2)
            try:
                dlg2._on_tick()
                check(list(dlg2.block) == list(saved["view"]),
                      "重开后 view 被改了：%s，存的 %s —— 手工对齐的结果必须保留"
                      % (dlg2.block, saved["view"]))
                check(mm.panel_to_canvas(0, 0, dlg2.calib()) == (x, y),
                      "重开后的换算又漂了：%s" % (mm.panel_to_canvas(0, 0,
                                                                dlg2.calib()),))
                # 顶部那行必须**说清几何是从文件读回来的** —— 不然"还原成功"和
                # "程序又猜了一个"在屏幕上一模一样，人只能猜（现场就是这么卡住的）
                check("已载入上次标定" in dlg2.lbl_loaded.text()
                      and "手工对齐" in dlg2.lbl_loaded.text(),
                      "重开时没说明「几何是从文件读回来的 / 手工对齐」：%r"
                      % dlg2.lbl_loaded.text())
                check(not dlg2._unsaved(),
                      "刚打开、什么都没动，不该说「有没保存的改动」")

                # ③ 在提示里选「否」时必须说出来（不能说"点了没反应"）
                with mock.patch.object(QMessageBox, "question",
                                       return_value=QMessageBox.No):
                    dlg2._on_save()
                check("没有保存" in dlg2.lbl_say.text(),
                      "拒绝保存后状态行要写明「没有保存」，实际：%r"
                      % dlg2.lbl_say.text())
            finally:
                dlg2._shutdown()

            # ④ 关窗守卫：动过又没存 → 必须问一句；选「否」= 放弃并真的关掉
            dlg3 = MinimapCalibDialog(mid, mode=mm.MODE_CROP,
                                      client=StubClient(panel))
            try:
                dlg3._on_tick()
                check(not dlg3._unsaved(), "刚打开就报「有没保存的改动」")
                pos = dlg3._ov_item.pos()
                dlg3._ov_item.setPos(pos.x() + 5, pos.y())
                check(dlg3._unsaved(), "拖过之后应当认出「有没保存的改动」")

                before = mapdata.load_calib(mid)
                asked = [0]

                def _no(*_a, **_kw):
                    asked[0] += 1
                    return QMessageBox.No

                with mock.patch.object(QMessageBox, "question", _no):
                    dlg3.reject()
                check(asked[0] == 1, "关窗时没问「还有没保存的改动」")
                check(not dlg3._timer.isActive(),
                      "选了「否」之后应当真的关掉（定时器还在跑 = 没关成）")
                check(mapdata.load_calib(mid) == before,
                      "选了「否」不该把放弃的那份改动写进文件")
            finally:
                dlg3._shutdown()

            # ⑤ 写文件失败必须**说出来**：这个槽外面包着 safe_slot，而它只是
            #    `traceback.print_exc()`；正常启动走 pythonw（没有控制台）——
            #    写失败会一点痕迹都没有：人点了保存、界面什么都没说，以为存上了。
            dlg4 = MinimapCalibDialog(mid, mode=mm.MODE_CROP,
                                      client=StubClient(panel))
            try:
                dlg4._on_tick()
                pos = dlg4._ov_item.pos()
                dlg4._ov_item.setPos(pos.x() + 3, pos.y())
                before = mapdata.load_calib(mid)
                with mock.patch.object(mapdata, "save_calib",
                                       side_effect=OSError("磁盘满")):
                    with mock.patch.object(QMessageBox, "question",
                                           return_value=QMessageBox.Yes):
                        dlg4._on_save()
                check("保存失败" in dlg4.lbl_say.text(),
                      "写文件失败必须写在状态行上（不然人以为存上了）：%r"
                      % dlg4.lbl_say.text())
                check(not dlg4.saved, "写失败了却把 saved 置成 True")
                check(dlg4._unsaved(), "写失败后应当仍然算「有没保存的改动」")
                check(mapdata.load_calib(mid) == before, "写失败却把文件改了")
            finally:
                dlg4._shutdown()
    finally:
        shutil.rmtree(str(tmpdir), ignore_errors=True)


def t_live_source_region():
    """来源「从实时画面框选」：裁那一块 / 报错怎么说 / 标定弹窗直接能吃它。

    **为什么单列一条**：这条来源不连 A 机的小地图推流，而是从**实时预览那一帧**
    上裁一块喂给标定弹窗（工作台「小地图来源」，实验功能）。它换的是"喂帧的人"，
    所以最容易出的错不是算法而是**接线**：裁错位置、把同一帧当成新帧反复裁、
    画面尺寸变了还照裁（给一块错位的图却没人发现）、弹窗里那句"等 A 机推流"
    说错地方。全都能离屏验：合成一张"实时画面"，把面板贴在某处，框那块区域。
    """
    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return

    from PyQt5.QtWidgets import QApplication
    from gui.live_panel import LiveFrameRegionClient
    from gui.minimap_calib import MinimapCalibDialog

    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")

    ch, cw = canvas.shape[:2]
    w, h = min(70, cw - 16), min(110, ch - 16)
    if w < 40 or h < 40:
        print("      （底图太小，跳过）")
        return
    x, y = 7, 11
    panel = synth_crop_panel(canvas, x, y, w, h)

    # 合成一张"实时画面"：面板贴在 (rx, ry)，四周是别的内容（背景不能和面板像）
    rx, ry = 40, 25
    live = np.full((ry + h + 30, rx + w + 30, 3), 40, np.uint8)
    live[ry:ry + h, rx:rx + w] = panel

    class _FakeLive:
        """实时预览面板的最小替身：只提供 current_frame()。"""

        def __init__(self, frame):
            self.frame = frame

        def current_frame(self):
            return self.frame

    lp = _FakeLive(live)
    cli = LiveFrameRegionClient(lp, [rx, ry, w, h])
    check(bool(cli.label) and bool(cli.wait_hint) and bool(cli.hint),
          "适配器没给弹窗提供来源说明文字（弹窗会说错话）")

    # 1) 裁出来的就是面板那一块（逐像素相等）
    crop, _t = cli.latest()
    check(crop is not None and crop.shape[:2] == (h, w),
          "裁出来的形状不对：%s，应该是 %s" % (None if crop is None
                                             else crop.shape, (h, w)))
    check(np.array_equal(crop, panel), "裁出来的不是框选那一块（位置或大小错了）")

    # 2) 同一帧不能反复裁（弹窗靠 `is not` 判断"来了新帧"，重复裁会让它以为帧在刷）
    again, _ = cli.latest()
    check(again is crop, "同一帧应当返回同一个对象（不然弹窗以为帧一直在更新）")
    check(cli.n_recv == 1, "同一帧被记成了 %d 帧" % cli.n_recv)
    n_before = cli.n_recv

    # 3) 换一帧 → 记一帧新的
    lp.frame = live.copy()
    crop2, _ = cli.latest()
    check(crop2 is not crop and cli.n_recv == n_before + 1,
          "换了帧却没记上新帧（n_recv=%d）" % cli.n_recv)
    check(np.array_equal(crop2, panel), "新帧裁出来的内容不对")

    # 4) 没画面 / 区域超出画面：都要说清楚，绝不静默给一块错的
    lp.frame = None
    got, _ = cli.latest()
    check(got is None and "预览" in cli.err,
          "没有实时画面时的说明不对：%r" % cli.err)
    lp.frame = live
    cli.region = [rx, ry, w + 5000, h]
    got, _ = cli.latest()
    check(got is None and "超出" in cli.err,
          "区域超出画面时的说明不对：%r" % cli.err)
    cli.region = [rx, ry, w, h]

    # 5) 标定弹窗直接吃这个来源（一行都不用改）：量出来的换算要和框选的位置一致
    stub = LiveFrameRegionClient(lp, [rx, ry, w, h])
    dlg = MinimapCalibDialog(mid, mode=mm.MODE_CROP, client=stub)
    try:
        dlg._on_tick()
        check(dlg._frame is not None, "弹窗没从实时画面拿到那块图")
        check("实时画面" in dlg.lbl_where.text(),
              "弹窗标题行没写出来源：%r" % dlg.lbl_where.text())
        dlg._locate_now()
        check(dlg.score > 0.9,
              "从实时画面来的那块定位失败（匹配分 %.3f）" % dlg.score)
        got = mm.panel_to_canvas(0, 0, dlg.calib())
        check(got == (x, y),
              "弹窗量出来的换算偏了：面板左上 → 底图 %s，应该是 (%d, %d)"
              % (got, x, y))
    finally:
        dlg._shutdown()


def t_live_source_wiring():
    """来源这条路的**接线**（源码约定）：下拉、按钮显隐、框选走规范那一份。"""
    rp = (ROOT / "gui" / "route_panel.py").read_text(encoding="utf-8")
    for kw in ("cmb_mmap_src", "btn_mmap_region", "_pick_mmap_region",
               "select_region_on_image", "LiveFrameRegionClient",
               "update_live(mmap_src="):
        check(kw in rp, "路线识别面板里缺 %s（来源这条路没接上）" % kw)
    check("btn_mmap_region.setVisible" not in rp,
          "「框选小地图」又跟着来源显隐了 —— 它是**必做的一步**：叠图要知道往"
          "画面的哪儿画，收流来源同样需要（主画面里也有小地图，只是被压过）")
    # 框选一律走 region_picker（放大镜/像素网格/Esc/<4px 当误点 —— UI规范 §8）：
    # 这里只许**调用**它，不许在面板里再写一份（当年就是这么长出第二份框选的）
    body = rp.split("def _pick_mmap_region", 1)[-1].split(
        "def _verify_mmap_region", 1)[0]
    check("select_region_on_image(" in body,
          "框选没走 gui/region_selector（它套着带放大镜的 region_picker）")
    check("mousePressEvent" not in body and "QDialog" not in body,
          "框选在面板里被自己实现了一遍 —— 规范要求只有 region_picker 一份")

    mw = (ROOT / "gui" / "main_window.py").read_text(encoding="utf-8")
    check("self.route_panel.live_panel = self.live_panel" in mw,
          "主窗口没把实时页给路线识别面板 —— 「从实时画面框选」拿不到画面")

    live = (ROOT / "config" / "live.yaml").read_text(encoding="utf-8")
    check("mmap_src" in live and "mmap_crop" in live,
          "config/live.yaml 里没有 mmap_src / mmap_crop（来源与框选区域存哪）")


def t_mmap_rows_split():
    """「小地图定位」要分两行：来源及其右边的东西另起一行（UI规范 §4）。

    **为什么量坐标，而不是匹配源码**：源码里 `row.addWidget` 写在哪儿，和它到底
    落在哪一行是两回事 —— 一个带 stretch 的 HBox 里，往右加多少次都还是同一行。
    只有把面板真建出来量 y，才说明视觉结果，也才拦得住「顺手往右续一个控件」。

    窄窗口那一半同样要量：一路往右堆的真实后果不是"太长"，而是**窗口一窄标签
    先被压没**（只剩几个看不出是什么的下拉框），所以换行必须在窄窗口下也成立。
    """
    from PyQt5.QtCore import QPoint
    from PyQt5.QtWidgets import QApplication

    from gui.route_panel import RoutePanel

    app = QApplication.instance() or QApplication([])
    p = RoutePanel()
    p.resize(900, 700)
    p.show()
    app.processEvents()

    def y(w):
        return w.mapTo(p, QPoint(0, 0)).y()

    try:
        y_up, y_dn = y(p.cmb_mmap_mode), y(p.cmb_mmap_src)
        check(y_dn > y_up,
              "「小地图来源」还跟「显示方式」挤在同一行（y %d vs %d）—— "
              "UI规范 §4：一行只放一个参数组，放不下就另起一行" % (y_dn, y_up))
        check(y(p.btn_mmap_calib) == y_dn,
              "「标定…」应当跟着「小地图来源」在第二行（y %d），实际 y %d"
              % (y_dn, y(p.btn_mmap_calib)))
        check(p.cmb_mmap_src.x() < p.cmb_mmap_mode.x(),
              "第二行没从左边起（来源 x=%d 在显示方式 x=%d 的右边）—— "
              "那只是把一个更长的行折了个位置，没解决问题"
              % (p.cmb_mmap_src.x(), p.cmb_mmap_mode.x()))

        # 窄窗口下两行不许合并（这才是「无脑往右加」真正的后果）
        p.resize(520, 700)
        app.processEvents()
        check(y(p.cmb_mmap_src) > y(p.cmb_mmap_mode),
              "窗口变窄之后两行又挤回一行了")
    finally:
        p.close()


def t_overlay_on_live():
    """把标定好的地形叠图画到实时画面上：几何 + 接线 + 「不碰帧」的约定。

    **为什么要单列一条**：这个功能横跨三处（几何在 perception/minimap、
    画在 gui/live_panel、算与接在 gui/route_panel），每一处单独看都没问题，
    合起来错的方式却很隐蔽：叠图偏一截（两边各写了一份公式）、或者叠图被
    烘进了帧数据（那时标定弹窗会**拿叠图和它自己匹配**，匹配分虚高）。
    """
    # ① 几何：和 panel_to_canvas **交叉验证**（不是把公式抄一遍）
    canvas_wh = (100, 50)
    z = mm.overlay_zoom(canvas_wh[0])
    crop = {"mode": mm.MODE_CROP, "scale": 2.0, "offset": [4, 4],
            "view": [10, 20]}
    src, dst = mm.overlay_draw_rects(crop, (60, 40), canvas_wh)
    check(dst == (0, 0, 60, 40),
          "crop 的目标矩形应当是**整个面板**（画面就是底图的一块）：%s" % (dst,))
    cx, cy = mm.panel_to_canvas(0, 0, crop)          # 面板左上 ↔ 底图哪儿
    check((src[0] / float(z), src[1] / float(z)) == (int(cx), int(cy)),
          "crop 的源矩形左上角（%s → 底图 %.2f, %.2f）和 panel_to_canvas 说的"
          "(%d, %d) 对不上 —— 两处公式漂了" % (src, src[0] / float(z),
                                            src[1] / float(z), int(cx), int(cy)))
    check(src[2] - src[0] == int(60 / 2.0 * z),
          "crop 的源宽度不对：%s（面板 60px ÷ scale 2.0 × overlay_zoom %d）"
          % (src[2] - src[0], z))

    fit = {"mode": mm.MODE_FIT, "scale": 0.5, "offset": [7, 9]}
    src2, dst2 = mm.overlay_draw_rects(fit, (60, 40), canvas_wh)
    check(src2 is None, "fit 取的是**整张**叠加图（源矩形该是 None）：%s" % (src2,))
    check(dst2 == (7, 9, 50, 25),
          "fit 的目标矩形应当是 offset 起、底图×scale 大：%s" % (dst2,))

    # ② 画面坐标：面板在画面 (6,72) 处，目标矩形整体平移过去
    _s3, d3 = mm.frame_overlay_rects(fit, (6, 72, 60, 40), canvas_wh)
    check(d3 == (13, 81, 50, 25), "面板在画面里的偏移没加上：%s" % (d3,))

    # ③ 接线：开关、状态行、框选不再跟来源显隐、也不再顺手改来源
    rp = (ROOT / "gui" / "route_panel.py").read_text(encoding="utf-8")
    for kw in ("ck_mmap_draw", "_refresh_overlay", "set_minimap_overlay(",
               "frame_overlay_rects("):
        check(kw in rp, "路线识别面板里缺 %s（叠图这条没接上）" % kw)
    check("_overlay_blocker" in rp,
          "叠图没画出来时不说明原因 —— 勾了没反应，人只能猜")
    check("update_live(mmap_src=mm.SRC_LIVE, mmap_crop=" not in rp,
          "「框选小地图」又把来源改掉了 —— 它只该存位置，不该顺手改另一个设置")

    lpn = (ROOT / "gui" / "live_panel.py").read_text(encoding="utf-8")
    check("def set_minimap_overlay" in lpn and "_ov_minimap" in lpn,
          "实时面板没有叠图那一层")
    body = lpn.split("def _paint_minimap_overlay", 1)[-1].split("\n    def ", 1)[0]
    check("_last_bgr" not in body,
          "叠加里动了帧数据 —— 会污染 current_frame()：探针/血条框选，以及"
          "小地图「从实时画面框选」的标定弹窗都会拿到画了叠图的帧")

    mw = (ROOT / "gui" / "main_window.py").read_text(encoding="utf-8")
    check("_refresh_overlay()" in mw,
          "主窗口没在接上实时面板之后补一次叠图 —— 开关本来就开着的用户要先去"
          "「路线识别」页点一下才看得到，表现就是「勾了没反应」")


def t_map_image_size_shown():
    """「地形图」那行要报出图片的**实际**尺寸（几×几），不能是推算出来的数。

    **为什么单列一条**：这个数最容易顺手写成"底图 × overlay_zoom"—— 而推算一旦
    和实际文件不一致（换过图、改过生成参数、手改过），报出来的就是**假数，还看着
    像对的**。所以这里拿 QPixmap 真读一遍文件来对。
    """
    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return

    from PyQt5.QtGui import QPixmap
    from PyQt5.QtWidgets import QApplication

    from gui.route_panel import RoutePanel

    app = QApplication.instance() or QApplication([])

    class _Proj:
        def get(self, k, d=None):
            return {"map_id": mid}.get(k, d)

    p = RoutePanel()
    try:
        p.project = _Proj()
        p._refresh_map_image()
        path, _title, _extra = p._map_image_path(mid)
        check(path is not None, "这张图没有可显示的图（底图和叠加图都没有？）")
        pm = QPixmap(str(path))
        txt = p.lbl_map_img.text()
        check("%d×%d 像素" % (pm.width(), pm.height()) in txt,
              "「地形图」信息里没报图片实际尺寸（%d×%d）：%r"
              % (pm.width(), pm.height(), txt))
        # 位置也要对：掉到附注（可能要跟一串命令行）后面就等于没报
        check(txt.index("%d×%d" % (pm.width(), pm.height()))
              < txt.index("（"),
              "尺寸排到了附注后面（要和命令行提示挤在一起）：%r" % txt)
    finally:
        p.close()


def _dot_panel(w=134, h=109, bg=(70, 55, 45)):
    """合成一块小地图面板（暗底），尺寸取实测的实时流框选大小 134×109。"""
    p = np.zeros((h, w, 3), np.uint8)
    p[:] = bg
    return p


class _Proj:
    """假项目：`RoutePanel._map_id()` 只读 `project.get("map_id")`。"""

    def __init__(self, mid):
        self._mid = mid

    def get(self, key, default=None):
        return self._mid if key == "map_id" else default


def t_dot_yellow_found():
    """亮黄点（实测玩家标记色）要能找出来，且位置准。

    颜色用**实测中心色 BGR(65,243,245)**，不是纯黄 (0,255,255) —— 压缩/抗锯齿
    会把纯黄渗成这个样子，按纯色卡阈值就找不到（这是实测踩到的）。
    """
    p = _dot_panel()
    p[50:56, 40:46] = (65, 243, 245)
    r = mm.find_player_dot(p)
    check(r["ok"], "6×6 的亮黄点没找到：%s" % r["reason"])
    check(abs(r["x"] - 42.5) <= 1.5 and abs(r["y"] - 52.5) <= 1.5,
          "点找出来了但位置偏了：(%.1f, %.1f)，应约 (42.5, 52.5)"
          % (r["x"], r["y"]))
    check(r["area"] >= 20, "点的像素数不对：%s" % r["area"])
    check(r["layer"] == "color" and not r["flood"],
          "干净面板不该走底图层/报 flood：layer=%s flood=%s"
          % (r["layer"], r["flood"]))


def t_dot_cyan_fallback():
    """无素材时客户端画的是 5×5 **青色**方块 —— 那也是玩家，同样要认。"""
    p = _dot_panel()
    p[30:35, 20:25] = (255, 255, 0)          # BGR：青
    r = mm.find_player_dot(p)
    check(r["ok"], "青色兜底点没找到：%s" % r["reason"])
    check(abs(r["x"] - 22.0) <= 1.5 and abs(r["y"] - 32.0) <= 1.5,
          "青色点位置偏了：(%.1f, %.1f)" % (r["x"], r["y"]))


def t_dot_reject_npc():
    """NPC / portal 的蓝点（132,216,243）**不是玩家** —— 不许认成玩家。

    认错的后果不是"少一帧"，而是"以为自己在别处"，寻路会朝错的方向走。
    """
    p = _dot_panel()
    p[30:36, 20:26] = (243, 216, 132)        # BGR：npc 蓝
    r = mm.find_player_dot(p)
    check(not r["ok"], "把 NPC/portal 的蓝点认成玩家了：(%.1f, %.1f)"
          % (r["x"], r["y"]))


def t_dot_flood_says_unusable():
    """图里亮黄块一大堆时，颜色层要**当场说不可用**，而不是随便指一个。

    实测背景：东部岩山V 那张图的底图本身就到处黄褐色，同一套颜色阈值命中
    2000~7000 像素 / 87~110 块。这时候给一个"最像的块"，下游会当成"我在哪块
    平台上" —— 比认不出糟得多。
    """
    p = _dot_panel()
    rng = np.random.default_rng(3)
    for _ in range(60):                      # 撒一把 4×4 亮黄碎块
        x = int(rng.integers(5, 125))
        y = int(rng.integers(5, 100))
        p[y:y + 4, x:x + 4] = (60, 240, 245)
    r = mm.find_player_dot(p)
    check(r["flood"], "命中这么多亮黄块，却没报「颜色层不可用」")
    check(not r["ok"], "颜色层不可用还给了一个位置（%.1f, %.1f）—— 下游会当真"
          % (r["x"], r["y"]))
    check("不可用" in r["reason"], "理由没说清是颜色层不可用：%r" % r["reason"])


def t_dot_basemap_layer():
    """底图相减层：底图自带一堆黄块时，只有「多出来的那个」才是玩家。

    **为什么必须有这一层**：同一张图里，颜色层会把底图的黄块一起收进来，而
    排序是"越大越像"，底图那些（放大后更大的）块会排在玩家点前面 —— 于是位置
    是错的。底图相减把"底图上本来就黄的地方"减掉，只剩下玩家标记。
    """
    class _T:                                # 只用到 .canvas，做个替身就够
        canvas = None

    canvas = np.zeros((44, 60, 3), np.uint8)
    canvas[:] = (70, 55, 45)
    for i in range(40):                      # 左半张底图撒黄块（都在 x<25）
        cx, cy = 3 + (i % 8) * 2, 3 + (i // 8) * 4
        canvas[cy:cy + 3, cx:cx + 3] = (60, 240, 245)

    scale = 3
    panel = cv2.resize(canvas, (60 * scale, 44 * scale),
                       interpolation=cv2.INTER_NEAREST)
    panel[100:106, 140:146] = (65, 243, 245)     # 玩家点（底图里没有这块）
    calib = {"mode": mm.MODE_FIT, "scale": float(scale), "offset": [0, 0],
             "score": 1.0}
    t = _T()
    t.canvas = canvas

    r0 = mm.find_player_dot(panel)                # 不给底图 → 颜色层被淹
    check(r0["flood"], "底图那堆黄块没把颜色层淹掉？mask=%s" % r0["mask_px"])
    check(not (r0["ok"] and abs(r0["x"] - 143) <= 2 and abs(r0["y"] - 103) <= 2),
          "颜色层直接给出了玩家点？那这一层就没必要了（%s）" % r0["reason"])

    r = mm.find_player_dot(panel, calib=calib, terrain=t)
    check(r["ok"], "有底图相减还找不到玩家点：%s" % r["reason"])
    check(r["layer"] == "basemap",
          "没走底图相减层（layer=%s）—— 那就还是被底图黄块带偏了" % r["layer"])
    check(abs(r["x"] - 143) <= 2 and abs(r["y"] - 103) <= 2,
          "底图相减后位置还是不对：(%.1f, %.1f)，应约 (143, 103)"
          % (r["x"], r["y"]))


def t_dot_hold_last_on_miss():
    """漏检时**沿用上一帧位置**（同怪物防抖的幽灵框），超过窗口才老实认不出。

    用户要的就是这个：认不出黄点就取上一帧 —— 否则读数一秒里闪好几次"认不出"，
    而且寻路白丢一小段惯性。口径照 `perception/tracker.py` 的 `debounce_ms`。
    """
    tr = mm.PlayerDotTracker(hold_ms=400)
    p1 = _dot_panel()
    p1[50:56, 40:46] = (65, 243, 245)
    r1 = tr.update(p1, now=100.0)
    check(r1["ok"] and not r1["held"], "第一拍应当是新找到的：%s" % r1["reason"])
    p2 = _dot_panel()
    p2[50:56, 41:47] = (65, 243, 245)
    r2 = tr.update(p2, now=100.04)
    check(r2["confirmed"] and not r2["held"], "第二拍应当确认、且不是沿用")

    blank = _dot_panel()                       # 这一拍认不出黄点
    r3 = tr.update(blank, now=100.08)
    check(r3["ok"] and r3["held"],
          "漏检一拍照样该给上一帧位置：%s" % r3["reason"])
    check(abs(r3["x"] - r2["x"]) <= 4 and abs(r3["y"] - r2["y"]) <= 4,
          "沿用的位置离上一帧太远：(%.1f, %.1f) vs (%.1f, %.1f)"
          % (r3["x"], r3["y"], r2["x"], r2["y"]))
    check(r3["missed"] == 1, "漏检计数不对：%s" % r3["missed"])
    check("上一帧" in r3["reason"], "没说清这是上一帧的位置：%r" % r3["reason"])

    r4 = tr.update(blank, now=100.40)          # 离上次成功 0.36s < 400ms
    check(r4["ok"] and r4["held"] and r4["missed"] == 2,
          "窗口内应当继续沿用、并累计漏检：%s" % r4)
    r5 = tr.update(blank, now=100.60)          # 0.56s > 400ms → 过期
    check(not r5["ok"], "超过防抖窗口还报上一帧位置 —— 那位置早过期了")
    check(not r5["held"], "过期了还标 held")


def t_dot_roi_search_local():
    """有上一帧位置时**只在它周围一小块里搜**（又快又稳）。

    稳：整块面板撒满黄碎块时，全画面搜会判「被淹」而拒绝；但只要有一个上一帧
    位置，就照样跟得住 —— 因为那一小块里没那么多杂色。
    快：按**搜索区尺寸**量（收流那条来源的面板 753×612，全图掩码+连通域每拍
    要几毫秒，缩到 37×37 基本免费）。
    """
    tr = mm.PlayerDotTracker()
    p1 = _dot_panel()
    p1[50:56, 40:46] = (65, 243, 245)
    r1 = tr.update(p1, now=1.0)
    check(r1["ok"], "第一拍都没找到：%s" % r1["reason"])

    dot = (47.0, 53.0)                          # 这一拍玩家点挪了一点
    noisy = _dot_panel()
    rng = np.random.default_rng(11)
    for _ in range(80):                         # 撒满黄碎块（躲开玩家点附近）
        x = int(rng.integers(5, 125))
        y = int(rng.integers(5, 100))
        if abs(x + 2 - dot[0]) <= 20 and abs(y + 2 - dot[1]) <= 20:
            continue
        noisy[y:y + 4, x:x + 4] = (60, 240, 245)
    noisy[50:56, 44:50] = (65, 243, 245)
    check(not mm.find_player_dot(noisy)["ok"],
          "没有先验时该拒绝（满屏碎块 = 颜色层不可用），这条前提不对")

    r2 = tr.update(noisy, now=1.04)
    check(r2["ok"], "有上一帧位置时应当照样跟住：%s" % r2["reason"])
    check(abs(r2["x"] - dot[0]) <= 2.5 and abs(r2["y"] - dot[1]) <= 2.5,
          "跟到的位置不对：(%.1f, %.1f)，期望约 %s" % (r2["x"], r2["y"], dot))
    roi = tr.last_roi
    check(roi is not None, "没走局部搜索这条便宜路")
    check(roi[2] - roi[0] <= 2 * mm.DOT_ROI_PAD + 2
          and roi[3] - roi[1] <= 2 * mm.DOT_ROI_PAD + 2,
          "搜索区太大（%s）—— 那就没省下什么" % (roi,))


def t_dot_flood_scale_invariant():
    """「多大算被淹」必须跟着**面板大小**走：同样的比例要给同样的结论。

    实测背景：两条来源的面板差 5.6 倍（独立推流 753×612、从实时画面 134×109），
    而玩家标记在两种面板里都只占 **0.09%**；反过来，"被淹"的实测样本都在 5% 以上。
    所以这条线只能按比例给 —— 绝对值只兜住"面板特别小"。

    踩过的坑（这条用例就是钉它）：绝对值原来写 400，于是**同一个比例**给出两种
    结论 —— 小面板上 400 像素 = 2.7%（放行），大面板上 400 像素 = 0.09%（等于
    没管）。表现是"换一条来源，同一张图的判据就变了"。
    """
    # ① 干净：一个小面板 13 像素的点、一个大面板 411 像素的点，都不该算被淹
    small = _dot_panel()
    small[50:56, 40:46] = (65, 243, 245)
    r = mm.find_player_dot(small)
    check(r["ok"] and not r["flood"], "小面板上正常的点被判被淹：%s" % r["reason"])

    big = _dot_panel(w=753, h=612)
    big[380:404, 230:258] = (99, 255, 255)       # 28×24 = 411 像素（实测值）
    r = mm.find_player_dot(big)
    check(r["ok"] and not r["flood"],
          "大面板上 0.09%% 的点被判被淹（成片 %s）：%s" % (r["dense_px"], r["reason"]))
    check(r["candidates"] == 1, "大面板候选数不对：%d" % r["candidates"])

    # ② 都被淹：两块面板各撒 **2%** 的成片标记色（小 300px / 大 9216px）
    small2 = _dot_panel()
    for x, y in ((20, 20), (40, 20), (20, 50)):
        small2[y:y + 10, x:x + 10] = (60, 240, 245)
    big2 = _dot_panel(w=753, h=612)
    for i in range(9):
        x, y = 40 + (i % 3) * 60, 40 + (i // 3) * 60
        big2[y:y + 32, x:x + 32] = (60, 240, 245)
    rs, rb = mm.find_player_dot(small2), mm.find_player_dot(big2)
    check(rs["flood"], "小面板 2%% 的成片量没判被淹（成片 %s / 阈值应约 %d）"
          % (rs["dense_px"], int(0.01 * 134 * 109)))
    check(rb["flood"], "大面板 2%% 的成片量没判被淹（成片 %s）" % rb["dense_px"])
    check(not rs["ok"] and not rb["ok"],
          "判了被淹却还给了位置 —— 那等于没拦")


def t_dot_tracker_confirm_and_jump():
    """跨帧：连续两拍才算确认；跳变太大的候选当噪声，且下一拍要能重捕。

    单帧亮斑屏幕上到处都是，只有"连续两拍都在附近"才当玩家 —— 这是把误检
    挡在定位外面的最后一道。
    """
    tr = mm.PlayerDotTracker()
    p1 = _dot_panel()
    p1[50:56, 40:46] = (65, 243, 245)
    r1 = tr.update(p1)
    check(r1["ok"], "第一拍就找不到点：%s" % r1["reason"])
    check(not r1["confirmed"], "第一拍就确认了 —— 单帧噪声会直接进定位")

    p2 = _dot_panel()
    p2[50:56, 43:49] = (65, 243, 245)          # 往右挪 3px
    r2 = tr.update(p2)
    check(r2["confirmed"], "连着两拍都在附近，却没确认")

    p3 = _dot_panel()
    p3[8:14, 8:14] = (65, 243, 245)            # 跳到 40+ px 外
    r3 = tr.update(p3)
    check(not r3["ok"], "位置跳变 %s px 也认了 —— 那是把噪声当玩家" % "大")

    r4 = tr.update(p3)                          # 下一拍：应当重新捕获
    check(r4["ok"], "跳变丢弃后没有重捕（人真传送过去就永远追不上了）：%s"
          % r4["reason"])


def t_segment_of_basics():
    """`segment_of` / `find_below`：拿**真实地形**验「点 → 脚下平台 → 哪条段」。

    这两条以前**一个用例都没有**，而"我在哪块平台上"全靠它们 —— 后面寻路的
    每条判定都建在它上面。这里钉住三件：出生点能落到某条段上、段号可用、
    离地图很远的点在"图外"要老实返回 None（而不是硬给一条最近的段）。
    """
    mid, canvas = pick_map()
    t = mapdata.load(mid, with_canvas=True) if mid else None
    if t is None or not t.segments:
        print("      （没有带段的地形数据，跳过）")
        return

    # 拿一根**非墙**的 foothold 的中点当"站在地上"的点（墙不是平台，见下）
    hit = None
    for s in t.segments:
        for f in s.footholds:
            if not f.is_wall:
                hit = (s, f)
                break
        if hit:
            break
    check(hit is not None, "地形里一根非墙 foothold 都没有（数据不完整）")
    seg, f0 = hit
    x = (f0.left + f0.right) / 2.0
    y = f0.y_at(x)

    below = t.find_below(x, y)
    check(below is not None,
          "站在 foothold (%.0f, %.0f) 上，脚下却找不到平台 —— find_below 坏了，"
          "寻路第一个判据就废了" % (x, y))
    got = t.segment_of(x, y)
    check(got is not None, "站在平台上却判不出在哪条段（segment_of 坏了）")
    check(got.index == seg.index,
          "判到了别的段：站在第 %d 段上，返回第 %s 段" % (seg.index, got.index))
    check(got.left - 1 <= x <= got.right + 1,
          "判出来的段不含这个 x：段 x[%d..%d]，点 x=%.0f"
          % (got.left, got.right, x))

    # 墙（x1 == x2）**不是平台**：不能把一堵墙当成"脚下的地面"
    wall = next((f for f in t.footholds if f.is_wall), None)
    if wall is not None:
        wy = (wall.y1 + wall.y2) / 2.0
        got = t.find_below(wall.x1, wy)
        check(got is None or abs(got[1] - wall.y_at(wall.x1)) > 0.51,
              "把一堵墙当成了脚下的平台（is_wall 那条判据没生效）：%s" % (got,))

    b = t.bounds
    far = (b[0] - 100000, b[1] - 100000)
    check(t.segment_of(*far) is None,
          "地图外十万像素的点也判出了一条段 —— 那是硬给，不是判据")
    check(t.find_below(*far) is None, "地图外的点也找到了脚下的平台")


def _panel_with_dot_at(canvas, cx, cy, scale=1, dot=6):
    """合成一块「fit 方式」的面板：底图放大 scale 倍，在 (cx,cy) 上画个黄点。

    ⚠ 这里刻意用 `scale=1`：**面板像素 = 底图像素**，黄点放哪就是哪，
    算出来的世界坐标可以直接和 `canvas_to_world` 的期望值比。
    """
    h = max(1, int(canvas.shape[0] * scale))
    w = max(1, int(canvas.shape[1] * scale))
    big = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_NEAREST)
    x = int(round(cx * scale - dot / 2.0))
    y = int(round(cy * scale - dot / 2.0))
    x = max(0, min(w - dot, x))
    y = max(0, min(h - dot, y))
    big[y:y + dot, x:x + dot] = (65, 243, 245)      # 实测的玩家点颜色
    return big, x + dot // 2, y + dot // 2          # (面板, 点的实际中心)


def t_world_pos_chain():
    """整条链：面板黄点 → 世界坐标 → 哪条段（S3 的成品）。

    判据不是"跑通了"，而是**数值对得上**：把黄点画在某个已知世界点对应的
    底图像素上，算出来的世界坐标必须回到那个点（容差 = 1 个底图像素对应的
    世界距离 —— 黄点本身有好几个像素，取整必然差一点）。
    """
    mid, canvas = pick_map()
    t = mapdata.load(mid, with_canvas=True) if mid else None
    if t is None or not t.segments:
        print("      （没有带段的地形数据，跳过）")
        return

    # 目标世界点：一根**非墙** foothold 的中点（一定落在那条段上）。
    # ⚠ 不能取 footholds[0]：段的第一根常常是**墙**（x1==x2），而墙不是平台
    # （find_below 会跳过它）—— 拿它当目标点会变成"点悬在墙中间"，判不出段。
    seg, f0 = next((s, f) for s in t.segments for f in s.footholds
                   if not f.is_wall)
    wx = (f0.left + f0.right) / 2.0
    wy = f0.y_at(wx)
    ccx, ccy = t.world_to_canvas(wx, wy)
    panel, px, py = _panel_with_dot_at(canvas, ccx, ccy)

    calib = {"mode": mm.MODE_FIT, "scale": 1.0, "offset": [0, 0],
             "view": [0, 0], "score": 1.0}
    loc = mm.PlayerLocator(mid)
    r = loc.update(panel, src=mm.SRC_LIVE, calib=calib, terrain=t)
    check(r["ok"], "整条链没跑通：%s（黄点层：%s）" % (r["note"], r["dot"]))
    tol = t.px_per_world + 2          # 一个底图像素 + 取整余量
    check(abs(r["world_x"] - wx) <= tol and abs(r["world_y"] - wy) <= tol,
          "世界坐标不对：(%.1f, %.1f)，期望 (%.1f, %.1f)（容差 %.1f）"
          % (r["world_x"], r["world_y"], wx, wy, tol))
    check(r["segment_id"] == seg.index,
          "落在第 %s 段上，却报成第 %s 段" % (seg.index, r["segment_id"]))

    # 认不出黄点（面板上什么都不画）→ **坐标必须是 None**，不是 0
    blank = panel.copy()
    blank[py - 4:py + 4, px - 4:px + 4] = (70, 55, 45)
    loc2 = mm.PlayerLocator(mid)
    r2 = loc2.update(blank, src=mm.SRC_LIVE, calib=calib, terrain=t)
    check(not r2["ok"], "面板上没黄点却说定位成功")
    check(r2["world_x"] is None and r2["world_y"] is None,
          "定位失败时坐标应当是 None —— 0 是合法的世界坐标（地图西北角），"
          "把「没定位」当成「我在那儿」会让寻路朝地图角落走")
    check(r2["note"], "定位失败没写原因 —— 界面上就只剩一个空读数")

    # 没标定 → 认出黄点也换不出坐标，但要说清是「没标定」
    loc3 = mm.PlayerLocator(mid)
    r3 = loc3.update(panel, src=mm.SRC_LIVE, calib={}, terrain=t)
    check(not r3["ok"] and "标定" in r3["note"],
          "没标定时该说「还没量过换算」：%s" % r3["note"])


def t_apply_to_player():
    """定位结论写进 `WorldState.Player`：算不出来写 None，且**别串到视觉平台编号**。

    `Player.current_platform_id`（视觉平台）和 `segment_id`（地形段）都是整数、
    都叫"平台"，串了会得出"看着对其实错"的结论 —— 这条把它钉死。
    """
    from perception.world_state import Player
    p = Player()
    p.current_platform_id = 7
    mm.apply_to_player(p, {"world_x": 123.5, "world_y": -45.0, "segment_id": 3,
                           "note": "", "held": True})
    check(p.world_x == 123.5 and p.world_y == -45.0, "世界坐标没写进去")
    check(p.segment_id == 3, "段号没写进去")
    check(p.world_held is True,
          "「位置是沿用上一帧的」没写进 Player —— 决策层分不出新鲜观测和补位")
    check(p.current_platform_id == 7,
          "把地形段号写进了 current_platform_id（那是**视觉平台**编号）")

    mm.apply_to_player(p, {"world_x": None, "world_y": None, "segment_id": None,
                           "note": "没认出黄点"})
    check(p.world_x is None and p.segment_id is None,
          "定位失败时应当写 None（不是 0）")
    check(p.world_note == "没认出黄点", "失败原因没带到 Player 上")
    check(p.current_platform_id == 7, "失败那次把视觉平台编号也抹了")


def t_world_label_text():
    """「世界坐标」那行：勾上叠图才显示，字数/颜色跟着结论走。

    按**行为**验（真建面板、真喂一帧）：`_tick_world` 里全是判据分支，写错一个
    分支的表现是"读数一直空着"或"显示了一个假坐标"，光看代码看不出来。
    """
    mid, canvas = pick_map()
    t = mapdata.load(mid, with_canvas=True) if mid else None
    if t is None or not t.segments:
        print("      （没有带段的地形数据，跳过）")
        return

    import re
    import unittest.mock as mock

    from PyQt5.QtWidgets import QApplication
    import gui.route_panel as rp
    from gui.route_panel import RoutePanel

    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")

    # 同 t_world_pos_chain：取非墙 foothold 的中点当目标（墙不是平台）
    seg, f0 = next((s, f) for s in t.segments for f in s.footholds
                   if not f.is_wall)
    wx = (f0.left + f0.right) / 2.0
    wy = f0.y_at(wx)
    ccx, ccy = t.world_to_canvas(wx, wy)
    panel, _px, _py = _panel_with_dot_at(canvas, ccx, ccy)

    class _FakeLive:
        """假实时面板：给一帧画面，并记下叠图/读数被贴了什么。"""
        def __init__(self, frame):
            self._f = frame
            self.ov = None
            self.note_only = 0      # "只换那行字"这条便宜路走了几次

        def current_frame(self):
            return self._f

        def set_minimap_overlay(self, pix, src=None, frame_rect=None,
                                alpha=0.55, note=""):
            self.ov = (pix, src, frame_rect, alpha, note)

        def set_overlay_note(self, note):
            """真面板在挂着叠图时会走这条（不重读地形/底图）。这里如实模拟。"""
            if self.ov is None:
                return False
            self.ov = tuple(self.ov[:4]) + (str(note or ""),)
            self.note_only += 1
            return True

    live = _FakeLive(panel)
    p = RoutePanel()
    p.live_panel = live
    p.project = _Proj(mid)
    p.ck_mmap_draw.blockSignals(True)
    p.ck_mmap_draw.setChecked(True)
    p.ck_mmap_draw.blockSignals(False)
    try:
        # 配置用**假的那一份**：`load_live` 读它、`update_live` 写它 ——
        # 真 live.yaml 一个字节都不碰（自检不许改用户配置），而开关来回拨的状态
        # 又是真的（不然"关掉后这行该藏起来"根本测不到）。
        cfg = {"mmap_src": mm.SRC_LIVE,
               "mmap_crop": [0, 0, panel.shape[1], panel.shape[0]],
               "mmap_draw": True}
        with mock.patch.object(rp, "load_live", lambda: dict(cfg)), \
             mock.patch.object(rp, "update_live",
                               lambda **kw: cfg.update(kw)), \
             mock.patch.object(mapdata, "load_calib",
                               lambda _mid, src=None: dict(
                                   mode=mm.MODE_FIT, scale=1.0,
                                   offset=[0, 0], view=[0, 0], score=1.0,
                                   alpha=55)):
            # ⚠ 必须先 show：`isVisible()` 在父控件没显示时**一律是 False**，
            # 不 show 就会把"没显示"误报成"这行没露出来"。
            p.show()
            app.processEvents()
            p._refresh_world()              # 勾上开关后的正常入口：显隐 + 立刻出数
            txt = p.lbl_mmap_world.text()
            check(p.lbl_mmap_world.isVisible(), "勾上叠图了，这行还没显示")
            check("玩家世界坐标" in txt, "这行没写世界坐标：%r" % txt)
            check("第 %d 段" % seg.index in txt,
                  "没报出是第几段（期望第 %d 段）：%r" % (seg.index, txt))
            # 数值要对得上（容差 = 一个底图像素：黄点本身有 6 个像素、取整也差一点）
            m = re.search(r"\((-?\d+), (-?\d+)\)", txt)
            check(m is not None, "这行没写出坐标数值：%r" % txt)
            gx, gy = int(m.group(1)), int(m.group(2))
            tol = t.px_per_world + 2
            check(abs(gx - wx) <= tol and abs(gy - wy) <= tol,
                  "读数里的坐标不对：(%d, %d)，期望约 (%.0f, %.0f)"
                  % (gx, gy, wx, wy))
            check(live.ov is not None and ("第 %d 段" % seg.index) in live.ov[4],
                  "画面里那块框下面没贴读数（段号）：%r"
                  % (live.ov[4] if live.ov else None))
            check(len(live.ov[4]) <= 30,
                  "贴到画面上的那行太长（%d 字）：%r"
                  % (len(live.ov[4]), live.ov[4]))
            check(live.note_only >= 1,
                  "读数没走「只换那行字」那条便宜路 —— 那会每 250ms 把地形 JSON "
                  "和底图 PNG 重读一遍，全在 GUI 主线程上")

            # 没画面 → 说清要先去开始预览，而不是留个空读数
            live._f = None
            p._tick_world()
            check("预览" in p.lbl_mmap_world.text(),
                  "没画面时该提示去「实时」页开预览：%r" % p.lbl_mmap_world.text())

            # 认不出黄点 → 防抖窗口内**沿用上一帧位置**，读数要标出来
            live._f = np.zeros_like(panel)          # 整块全黑：这一拍一定认不出
            p._tick_world()
            check("上一帧" in p.lbl_mmap_world.text(),
                  "漏检时该沿用上一帧位置并标出来：%r" % p.lbl_mmap_world.text())

            # 窗口关掉 + 还是认不出 → 这才走"认不出"那条（**画面上那行必须短**：
            # 完整诊断一百多字，贴上去就是一条横穿半屏的黑带），详情进 tooltip
            p._locator.tracker.hold_ms = 0.0
            p._tick_world()
            txt = p.lbl_mmap_world.text()
            check("认不出" in txt, "认不出黄点却没在读数里说出来：%r" % txt)
            check(len(txt) <= 40,
                  "面板这行太长了（%d 字）—— 人看的就是这一行：%r" % (len(txt), txt))
            osd = (live.ov[4] if live.ov else "")
            check(osd and len(osd) <= 20,
                  "贴到画面上的那行太长（%d 字）：%r" % (len(osd), osd))
            check(len(p.lbl_mmap_world.toolTip()) > 40,
                  "详情没进 tooltip —— 那长诊断就白算了：%r"
                  % p.lbl_mmap_world.toolTip())

            # 关掉开关 → 这行藏起来
            p.ck_mmap_draw.setChecked(False)
            p._refresh_world()
            check(not p.lbl_mmap_world.isVisible(), "关掉叠图了这行还显示着")
    finally:
        p._world_timer.stop()
        p.close()


def t_live_thread_mmap_panel_copy():
    """实时回路裁小地图面板**必须复制、且在画框之前**（这是条实打实的坑）。

    为什么按行为验：那一帧是主回路的**就地画布**，后面会往上面画玩家蓝框、
    平台线、攻击线。切个视图（不复制）的话，画上去的饱和色会串进黄点识别里
    —— 而黄点正是按饱和色找的。这条测试的做法：裁完之后改原帧，面板不该跟着变。
    """
    mid, canvas = pick_map()
    t = mapdata.load(mid, with_canvas=True) if mid else None
    if t is None or not t.segments:
        print("      （没有带段的地形数据，跳过）")
        return

    import unittest.mock as mock

    from PyQt5.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from gui.live_thread import LiveThread

    seg, f0 = next((s, f) for s in t.segments for f in s.footholds
                   if not f.is_wall)
    wx = (f0.left + f0.right) / 2.0
    ccx, ccy = t.world_to_canvas(wx, f0.y_at(wx))
    panel, _px, _py = _panel_with_dot_at(canvas, ccx, ccy)

    # 一整帧：左上角放"小地图面板"，其余填成底色
    frame = np.full((panel.shape[0] + 40, panel.shape[1] + 40, 3), 30, np.uint8)
    frame[:panel.shape[0], :panel.shape[1]] = panel

    th = LiveThread({"mmap_map_id": mid, "mmap_src": mm.SRC_LIVE,
                     "mmap_crop": [0, 0, panel.shape[1], panel.shape[0]]})
    try:
        got = th._mmap_panel_from_frame(frame)
        check(got is not None, "从实时画面裁不出面板（框选/来源没接上）")
        check(got.shape == panel.shape, "裁出来的尺寸不对：%s" % (got.shape,))
        before = got.copy()
        frame[10:16, 10:16] = (0, 255, 0)      # 模拟主回路画上去的框线
        check(np.array_equal(got, before),
              "裁下来的是原帧的**视图**不是副本 —— 主回路画上去的框线会串进"
              "黄点识别（黄点就是按饱和色找的）")

        with mock.patch.object(mapdata, "load_calib",
                               lambda _m, src=None: dict(
                                   mode=mm.MODE_FIT, scale=1.0, offset=[0, 0],
                                   view=[0, 0], score=1.0)):
            r = th._locate_mmap(got)
        check(r is not None and r.get("ok"),
              "实时回路这条定位没跑通：%s" % ((r or {}).get("note"),))
        check(r["segment_id"] == seg.index,
              "段号不对：报 %s，期望 %d" % (r["segment_id"], seg.index))

        # 界面上改黄点容差 → 实时线程这一份也要跟着改（**在拍与拍之间对齐**，
        # 不在 set_mmap 里直接动另一条线程正在用的 locator）
        th.set_mmap(track={"mmap_hold_ms": 0, "mmap_roi_pad": 9})
        th._locate_mmap(got)
        check(th._locator.tracker.hold_ms == 0
              and th._locator.tracker.roi_pad == 9,
              "实时线程那份 locator 没跟上界面参数：%s / %s"
              % (th._locator.tracker.hold_ms, th._locator.tracker.roi_pad))

        # 没地图 → 整件事不干（route_enabled 关掉时也一样，别白算）
        th.set_mmap(map_id="")
        check(th._locate_mmap(got) is None, "没有地图却在定位")
    finally:
        th.stop()


def t_track_rows_layout():
    """「标记跟踪」那四个容差要**排成两行**，不许挤一行（docs/UI规范.md §4）。

    为什么量坐标而不是看源码：一个带 stretch 的 HBox 里 `addWidget` 写多少次都还是
    同一行（同 `t_mmap_rows_split` 的理由）。挤一行的真实后果是实测出来的：面板最小
    宽度从 521 涨到 **714**，窗口一窄就是标签先被压没。
    """
    import unittest.mock as mock

    from PyQt5.QtCore import QPoint
    from PyQt5.QtWidgets import QApplication, QLabel
    import gui.route_panel as rp
    from gui.route_panel import RoutePanel

    app = QApplication.instance() or QApplication([])
    p = RoutePanel()
    try:
        with mock.patch.object(rp, "load_live",
                               lambda: {"mmap_draw": False,
                                        "mmap_src": mm.SRC_LIVE,
                                        "mmap_crop": [0, 0, 40, 40],
                                        "mmap_hold_ms": 500, "mmap_roi_pad": 18,
                                        "mmap_max_jump": 40,
                                        "mmap_ghost_shift": 8}):
            p.resize(560, 800)
            p.show()
            app.processEvents()
            ys = {k: sp.mapTo(p, QPoint(0, 0)).y()
                  for k, sp in p._sp_track.items()}
            check(len(set(ys.values())) == 2,
                  "四个容差没排成两行（y=%s）—— 挤一行会把面板撑到 700+" % ys)
            check(ys["mmap_hold_ms"] == ys["mmap_ghost_shift"]
                  and ys["mmap_roi_pad"] == ys["mmap_max_jump"]
                  and ys["mmap_hold_ms"] != ys["mmap_roi_pad"],
                  "两行的分组不对（应当 沿用+外推 / 搜索半径+跳变）：%s" % ys)
            w = p.minimumSizeHint().width()
            check(w <= 580, "面板最小宽度被撑到 %d —— 一行里塞太多了" % w)
            for lb in p.findChildren(QLabel):
                if lb.text() in ("沿用", "搜索半径", "跳变上限", "外推上限"):
                    check(lb.width() > 0,
                          "标签「%s」被压没了（宽度 0）" % lb.text())
    finally:
        p._world_timer.stop()
        p.close()


def t_dot_track_params_are_config():
    """黄点容差是**界面参数**（config/live.yaml）：回填、改一下就生效、坏值退回默认。

    为什么这组值得有用例：它们决定"要不要信这一帧的黄点"，而自检里最容易出的错是
    "界面拨了没生效"或"回填时把配置又写了一遍"（UI规范 §4/§5）。所以：按**行为**验
    —— 真建面板、真拨一下输入框、看配置和 locator 有没有跟着动。
    """
    import unittest.mock as mock

    from PyQt5.QtWidgets import QApplication
    import gui.route_panel as rp
    from gui.route_panel import RoutePanel

    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")

    # 配置坏一个值（字符串）→ 其余照旧，坏的退回默认（不能炸）
    trk = mm.track_params({"mmap_hold_ms": "写坏了", "mmap_roi_pad": 33})
    check(trk["mmap_hold_ms"] == mm.DOT_HOLD_MS,
          "配置写坏了没退回默认：%s" % trk["mmap_hold_ms"])
    check(trk["mmap_roi_pad"] == 33, "正常的值没读进来：%s" % trk["mmap_roi_pad"])

    cfg = {"mmap_src": mm.SRC_LIVE, "mmap_crop": [0, 0, 40, 40],
           "mmap_draw": False, "mmap_hold_ms": 700, "mmap_roi_pad": 25}
    pushed = []

    class _LP:
        """假实时面板：只要 `set_mmap`（面板刷新时还会调叠图那一层，一并给上）。"""

        def set_mmap(self, **kw):
            pushed.append(kw)

        def set_minimap_overlay(self, *a, **kw):
            pass

        def current_frame(self):
            return None

    p = RoutePanel()
    p.live_panel = _LP()
    p.project = _Proj("")          # 没地图也算：这组参数与地图无关
    p._locator = mm.PlayerLocator()
    try:
        with mock.patch.object(rp, "load_live", lambda: dict(cfg)), \
             mock.patch.object(rp, "update_live",
                               lambda **kw: cfg.update(kw)):
            p._refresh_mmap()
            check(p._sp_track["mmap_hold_ms"].value() == 700,
                  "输入框没回填配置里的值：%s" % p._sp_track["mmap_hold_ms"].value())
            check(p._sp_track["mmap_roi_pad"].value() == 25,
                  "搜索半径没回填：%s" % p._sp_track["mmap_roi_pad"].value())
            check(p._locator.tracker.hold_ms == 700,
                  "回填之后 locator 没跟着用上：%s" % p._locator.tracker.hold_ms)

            cfg["mmap_hold_ms"] = 700          # 回填不许把配置写一遍
            before = dict(cfg)
            p._refresh_mmap()
            check(cfg == before, "回填时把配置又写了一遍：%s" % cfg)

            pushed.clear()
            p._sp_track["mmap_hold_ms"].setValue(0)   # 拨一下：关掉沿用
            check(cfg["mmap_hold_ms"] == 0,
                  "改了输入框没存进配置：%s" % cfg["mmap_hold_ms"])
            check(p._locator.tracker.hold_ms == 0,
                  "改了输入框没在面板这份 locator 上生效")
            check(pushed and pushed[-1].get("track", {}).get("mmap_hold_ms") == 0,
                  "没把新参数推给实时线程：%s" % (pushed[-1:] or None))
    finally:
        p._world_timer.stop()
        p.close()


def t_dot_track_params_survive_map_change():
    """跟踪参数要**跟着 locator 走**，换图重建 tracker 时不能被丢掉。

    这是最容易漏的一处：换图会重建 tracker（位置不连续，别跨图跟踪），而参数就
    挂在 tracker 上 —— 不特意带上，"界面里调过一次、一换图又回默认"。
    """
    mid, _canvas = pick_map()
    loc = mm.PlayerLocator()
    loc.set_track(hold_ms=123, roi_pad=7, max_jump=9, ghost_max_shift=1.5)
    check(loc.tracker.hold_ms == 123 and loc.tracker.roi_pad == 7,
          "set_track 没生效")
    loc.load(mid or "不存在的图")          # 换图 → 重建 tracker
    check(loc.tracker.hold_ms == 123 and loc.tracker.roi_pad == 7
          and loc.tracker.max_jump == 9 and loc.tracker.ghost_max_shift == 1.5,
          "换图之后跟踪参数回默认了：hold=%s roi=%s jump=%s shift=%s"
          % (loc.tracker.hold_ms, loc.tracker.roi_pad, loc.tracker.max_jump,
             loc.tracker.ghost_max_shift))
    # 配置形状（mmap_*）也能直接喂
    loc.use_track_config({"mmap_hold_ms": 250, "mmap_max_jump": 0})
    check(loc.tracker.hold_ms == 250 and loc.tracker.max_jump == 0,
          "use_track_config 没生效：%s / %s"
          % (loc.tracker.hold_ms, loc.tracker.max_jump))


def t_safe_slot_drops_extra_signal_args():
    """`safe_slot` 必须**按槽的签名**裁掉多余的信号参数。

    Qt 的 `clicked` 带一个 bool、`currentIndexChanged` 带一个 int，而
    `safe_slot` 返回的是 `wrapper(*args)` —— PyQt 看它"什么都能接"就照发，
    零参的槽于是 TypeError，**又被这层自己吞掉**：按钮点了没反应。
    实测：标定弹窗的「保存标定」「自动定位」两个按钮都死在这上面
    （stderr.log 里躺着 `_on_save() takes 1 positional argument but 2 were given`）。
    """
    from gui.worker import safe_slot

    seen = []
    zero = safe_slot(lambda: seen.append("zero"))
    one = safe_slot(lambda v: seen.append(("one", v)))
    many = safe_slot(lambda *a: seen.append(("many", a)))

    zero(False)                    # 模拟 clicked(False)
    zero()
    one(False)
    many(1, 2, 3)
    check(seen == ["zero", "zero", ("one", False), ("many", (1, 2, 3))],
          "参数没按签名裁：%s" % (seen,))

    # 吞异常这条**不能丢**：它挡的是 PyQt 未捕获异常 → qFatal → 无征兆闪退
    boom = safe_slot(lambda: 1 / 0)
    boom()                          # 不该抛出去
    check(True, "（未抛异常）")


def t_calib_buttons_are_alive():
    """标定弹窗的按钮要**点得动**（走真实信号 —— 这正是老用例漏掉的那条路）。

    为什么单列：`dlg._on_save()` 直接调是通的，`dlg.btn_save.click()` 却是死的
    （见 t_safe_slot_drops_extra_signal_args）。用户看到的现象就是
    「我点了保存，结果没保存」—— 而且屏幕上和存过了一模一样。
    """
    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return

    import shutil
    import tempfile
    import unittest.mock as mock

    from PyQt5.QtWidgets import QApplication, QMessageBox
    from gui.minimap_calib import MinimapCalibDialog

    app = QApplication.instance() or QApplication([])
    ch, cw = canvas.shape[:2]
    w, h = min(80, cw - 16), min(120, ch - 16)
    if w < 40 or h < 40:
        print("      （底图太小，跳过）")
        return
    x, y = 5, 9
    panel = synth_crop_panel(canvas, x, y, w, h)

    tmpdir = Path(tempfile.mkdtemp(prefix="mmbtn_"))
    tmp = tmpdir / ("%s.mapcalib.json" % mid)
    try:
        with mock.patch.object(mapdata, "calib_path", lambda _mid: tmp):
            dlg = MinimapCalibDialog(mid, mode=mm.MODE_CROP,
                                     client=StubClient(panel), src=mm.SRC_LIVE)
            try:
                dlg._on_tick()
                dlg._ov_item.setPos(dlg.inset - (x + dlg.inset),
                                    dlg.inset - (y + dlg.inset))
                with mock.patch.object(QMessageBox, "question",
                                       return_value=QMessageBox.Yes):
                    dlg.btn_save.click()          # ← 用户点的是这里，不是 _on_save
                check(dlg.saved, "「保存标定」按钮点了没反应（走真实信号就废了）")
                saved = mapdata.load_calib(mid, mm.SRC_LIVE) or {}
                check(mm.has_geometry(saved),
                      "按钮点了、也置了 saved，但文件里没有几何：%s" % saved)
            finally:
                dlg._shutdown()
    finally:
        shutil.rmtree(str(tmpdir), ignore_errors=True)


def t_calib_per_source():
    """标定**按来源分开存**：两条来源各留各的几何，互不覆盖。

    为什么非这样不可（实测）：同一张图，独立推流那条面板 753×612、从实时画面那条
    134×109 —— 差 5.6 倍。一份几何只对一条来源成立；共用的话算出来的世界坐标整体
    错，而错的位置会被下游当成「我在哪块平台上」。
    """
    mid, _canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return

    import shutil
    import tempfile
    import unittest.mock as mock

    tmpdir = Path(tempfile.mkdtemp(prefix="mmcs_"))
    tmp = tmpdir / ("%s.mapcalib.json" % mid)
    try:
        with mock.patch.object(mapdata, "calib_path", lambda _mid: tmp):
            live_c = {"mode": mm.MODE_CROP, "scale": 4.0, "offset": [4, 4],
                      "view": [11, 22], "score": 0.9}
            st_c = {"mode": mm.MODE_FIT, "scale": 1.0, "offset": [0, 0],
                    "view": [0, 0], "score": 0.8}
            mapdata.save_calib(mid, live_c, mm.SRC_LIVE)
            mapdata.save_calib(mid, st_c, mm.SRC_STREAM)

            check(mapdata.load_calib(mid, mm.SRC_LIVE)["scale"] == 4.0,
                  "live 那份被覆盖了：%s" % mapdata.load_calib(mid, mm.SRC_LIVE))
            check(mapdata.load_calib(mid, mm.SRC_STREAM)["scale"] == 1.0,
                  "stream 那份被覆盖了：%s"
                  % mapdata.load_calib(mid, mm.SRC_STREAM))
            check(set(mapdata.load_calibs(mid)) == {mm.SRC_LIVE, mm.SRC_STREAM},
                  "两份没并存：%s" % sorted(mapdata.load_calibs(mid)))
            check(mapdata.load_calib(mid, "cyber") is None,
                  "没标过的来源应当给 None（不能拿别的来源顶上）")
            check(mapdata.load_calib(mid) is None,
                  "两份都在时不该猜一份给调用方 —— 猜错就是错的几何")

            # 再存一次 live：不能碰到 stream
            mapdata.save_calib(mid, dict(live_c, scale=5.0), mm.SRC_LIVE)
            check(mapdata.load_calib(mid, mm.SRC_STREAM)["scale"] == 1.0,
                  "存 live 把 stream 那份抹了：%s"
                  % mapdata.load_calib(mid, mm.SRC_STREAM))
    finally:
        shutil.rmtree(str(tmpdir), ignore_errors=True)


def t_calib_legacy_readable():
    """老格式（整份平铺、没记来源）仍读得出来并被标上 legacy；保存后归到该来源。

    这条保证**升级不把已有标定作废**：老文件是在「还没有按来源分开存」的时候量的，
    直接当成「没标过」会逼用户白量一次（而现场正是那种"我明明标过"的场景）。
    """
    mid, _canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return

    import json
    import shutil
    import tempfile
    import unittest.mock as mock

    tmpdir = Path(tempfile.mkdtemp(prefix="mmlegacy_"))
    tmp = tmpdir / ("%s.mapcalib.json" % mid)
    try:
        with mock.patch.object(mapdata, "calib_path", lambda _mid: tmp):
            old = {"mode": mm.MODE_FIT, "scale": 2.0, "offset": [3, 4],
                   "view": [0, 0], "score": 0.7, "src": "auto"}
            tmp.write_text(json.dumps(old), encoding="utf-8")

            got = mapdata.load_calib(mid, mm.SRC_LIVE)
            check(mm.has_geometry(got or {}),
                  "老格式的几何读不出来了 —— 升级会把已有标定作废")
            check((got or {}).get("legacy"),
                  "老格式没被标出来，界面就没法提醒「换过来源请重量」")
            check((mapdata.load_calib(mid) or {}).get("scale") == 2.0,
                  "不带来源时读不出老格式")

            # 保存一次 → 变成 v2、归到这条来源；另一条来源仍然算「没标过」
            mapdata.save_calib(mid, dict(got, scale=2.5), mm.SRC_LIVE)
            again = mapdata.load_calib(mid, mm.SRC_LIVE) or {}
            check(again.get("scale") == 2.5, "保存后没更新：%s" % again)
            check(not again.get("legacy"), "legacy 只是读的时候的标记，不该写进文件")
            check(mapdata.load_calib(mid, mm.SRC_STREAM) is None,
                  "老格式那份不该被另一条来源继承（几何只对一条成立）")
            check(json.loads(tmp.read_text(encoding="utf-8")).get("v") == 2,
                  "保存后文件没升级成 v2")
    finally:
        shutil.rmtree(str(tmpdir), ignore_errors=True)


def t_calib_legacy_is_visible():
    """老格式（没记来源）必须在**界面上说出来**，而不是默默当当前来源用。

    为什么这条值得单列：老格式那份几何只对**当时那条来源**成立，而现在默认
    「哪条来源都能读到它」。不吭声的话，人在另一条来源下会拿到一份错的换算——
    算出来的位置整体错，还会被当成「我在哪块平台上」。

    ⚠ 必须塞**假收帧器**（同其它弹窗用例）：不传 client 时弹窗会自己连 A 机那口
    推流，起来一条后台线程 —— 实测这么写在退出时直接把进程搞崩（0xC0000409）。
    """
    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return

    import json
    import shutil
    import tempfile
    import unittest.mock as mock

    from PyQt5.QtWidgets import QApplication
    from gui.minimap_calib import MinimapCalibDialog

    app = QApplication.instance() or QApplication([])
    ch, cw = canvas.shape[:2]
    w, h = min(80, cw - 16), min(120, ch - 16)
    if w < 40 or h < 40:
        print("      （底图太小，跳过）")
        return
    panel = synth_crop_panel(canvas, 5, 9, w, h)

    tmpdir = Path(tempfile.mkdtemp(prefix="mmlegacyui_"))
    tmp = tmpdir / ("%s.mapcalib.json" % mid)
    try:
        with mock.patch.object(mapdata, "calib_path", lambda _mid: tmp):
            tmp.write_text(json.dumps(
                {"mode": mm.MODE_FIT, "scale": 1.0, "offset": [0, 0],
                 "view": [0, 0], "score": 0.7}), encoding="utf-8")
            dlg = MinimapCalibDialog(mid, mode=mm.MODE_FIT,
                                     client=StubClient(panel), src=mm.SRC_LIVE)
            try:
                txt = dlg.lbl_loaded.text()
                check("老格式" in txt,
                      "老格式的标定被当成了「来源：从实时画面」：%r" % txt)
                check("老格式" in dlg.lbl_loaded.toolTip(),
                      "tooltip 里没提醒「换过来源请重量」：%r"
                      % dlg.lbl_loaded.toolTip()[:80])
            finally:
                dlg._shutdown()
    finally:
        shutil.rmtree(str(tmpdir), ignore_errors=True)


def t_calib_dialog_saves_its_source():
    """标定弹窗把几何存到**自己那条来源**下；再标另一条来源不覆盖它。

    弹窗是唯一写标定的入口，所以这条要按**行为**验（不是看源码调没调对参数）。
    """
    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return

    import json
    import shutil
    import tempfile
    import unittest.mock as mock

    from PyQt5.QtWidgets import QApplication, QMessageBox
    from gui.minimap_calib import MinimapCalibDialog

    app = QApplication.instance() or QApplication([])
    ch, cw = canvas.shape[:2]
    w, h = min(80, cw - 16), min(120, ch - 16)
    if w < 40 or h < 40:
        print("      （底图太小，跳过）")
        return
    x, y = 5, 9
    panel = synth_crop_panel(canvas, x, y, w, h)

    tmpdir = Path(tempfile.mkdtemp(prefix="mmdlgsrc_"))
    tmp = tmpdir / ("%s.mapcalib.json" % mid)
    try:
        with mock.patch.object(mapdata, "calib_path", lambda _mid: tmp):
            first = True
            prev = {}
            for src in (mm.SRC_LIVE, mm.SRC_STREAM):
                stub = StubClient(panel)
                dlg = MinimapCalibDialog(mid, mode=mm.MODE_CROP, client=stub,
                                         src=src)
                try:
                    check(dlg.src == src, "弹窗没记住自己是哪条来源：%s" % dlg.src)
                    dlg._on_tick()
                    dlg._ov_item.setPos(dlg.inset - (x + dlg.inset),
                                        dlg.inset - (y + dlg.inset))
                    with mock.patch.object(QMessageBox, "question",
                                           return_value=QMessageBox.Yes):
                        dlg._on_save()
                    check(dlg.saved, "（%s）保存没成功" % src)
                finally:
                    dlg._shutdown()
                saved = mapdata.load_calib(mid, src) or {}
                check(mm.has_geometry(saved),
                      "（%s）存下来的几何读不出来：%s" % (src, saved))
                # 另一条来源：第一轮应当**根本不存在**；第二轮应当**原样还在**
                #（新一轮标定只该碰自己那条）
                other = mm.SRC_STREAM if src == mm.SRC_LIVE else mm.SRC_LIVE
                og = mapdata.load_calib(mid, other)
                if first:
                    check(og is None,
                          "标 %s 的时候把 %s 那份也写出来了（整份覆盖）" % (src, other))
                else:
                    check(og == prev[other],
                          "标 %s 把 %s 那份改掉了：%s → %s"
                          % (src, other, prev[other], og))
                prev[src] = saved
                first = False
            body = json.loads(tmp.read_text(encoding="utf-8"))
            check(set(body.get("sources") or {}) == {mm.SRC_LIVE, mm.SRC_STREAM},
                  "两条来源没并存：%s" % sorted((body.get("sources") or {}).keys()))
    finally:
        shutil.rmtree(str(tmpdir), ignore_errors=True)


def t_route_panel_button():
    """「路线识别」面板上要有「标定…」按钮，且已经接到弹窗（源码级检查）。"""
    src = (ROOT / "gui" / "route_panel.py").read_text(encoding="utf-8")
    check("btn_mmap_calib" in src, "面板上没有标定按钮")
    check("MinimapCalibDialog(" in src, "标定按钮没接到弹窗")
    check("python -m perception.minimap --map" in src,
          "命令行那条等价做法应当留在 tooltip 里（排查用）")


TESTS = (
    ("fit 往返：合成整图 → 量回缩放/偏移", t_fit_roundtrip),
    ("crop 往返：合成一块 → 量回 view（含 inset 约定）", t_crop_roundtrip),
    ("1:1 裁块的 scale 不许被顶替成假值", t_crop_scale_not_faked),
    ("crop 放大取块：底图放大 9 倍后裁一块 → 量回 (scale, view)", t_magnified_crop_roundtrip),
    ("方式选错时量不出来（不是给个错值）", t_wrong_mode_fails),
    ("标定弹窗（假流，不碰网络/真实标定）", t_dialog_with_stub),
    ("保存→重开：手工对齐的几何原样回来", t_save_then_reopen),
    ("来源「从实时画面框选」：裁块 / 报错 / 弹窗直接能用", t_live_source_region),
    ("来源这条路的接线（下拉/按钮显隐/框选走规范）", t_live_source_wiring),
    ("小地图定位的参数分两行（来源及其右边另起一行）", t_mmap_rows_split),
    ("把标定好的叠图画到实时画面上（几何/接线/不碰帧）", t_overlay_on_live),
    ("「地形图」那行报出图片实际尺寸（几×几）", t_map_image_size_shown),
    ("弹窗的持续自动定位（邻近尺度 + 跟丢回退）", t_dialog_auto_fit),
    ("路线识别面板上的入口", t_route_panel_button),
    ("黄点：亮黄点找得出且位置准（实测色，不是纯黄）", t_dot_yellow_found),
    ("黄点：青色兜底方块也要认", t_dot_cyan_fallback),
    ("黄点：NPC/portal 的蓝点不许认成玩家", t_dot_reject_npc),
    ("黄点：图里亮黄块成堆时判「颜色层不可用」", t_dot_flood_says_unusable),
    ("黄点：底图相减层只留下「多出来的」那个", t_dot_basemap_layer),
    ("黄点：「被淹」的判据跟着面板大小走（同比例同结论）",
     t_dot_flood_scale_invariant),
    ("黄点：跨帧连续两拍才确认，跳变丢弃并重捕", t_dot_tracker_confirm_and_jump),
    ("黄点：漏检沿用上一帧位置（超窗口才认不出）", t_dot_hold_last_on_miss),
    ("标记跟踪那四个容差排成两行（别挤一行撑宽面板）", t_track_rows_layout),
    ("黄点容差是界面参数（回填/生效/坏值退默认）", t_dot_track_params_are_config),
    ("黄点容差换图不丢（重建 tracker 要带上）",
     t_dot_track_params_survive_map_change),
    ("黄点：有先验时只在周围一小块搜（又快又稳）", t_dot_roi_search_local),
    ("标定按来源分开存：两条来源各留各的几何", t_calib_per_source),
    ("标定：老格式仍读得出（升级不作废）+ 保存后归到该来源",
     t_calib_legacy_readable),
    ("标定弹窗只写自己那条来源（不整份覆盖）", t_calib_dialog_saves_its_source),
    ("safe_slot 按签名裁掉多余的信号参数", t_safe_slot_drops_extra_signal_args),
    ("标定弹窗的按钮点得动（走真实信号那条路）", t_calib_buttons_are_alive),
    ("标定：老格式要在界面上说出来（不冒充当前来源）", t_calib_legacy_is_visible),
    ("段查询：出生点落得到段上、图外的点老实返回 None", t_segment_of_basics),
    ("世界坐标：黄点 → 世界坐标 → 哪条段（数值对得上）", t_world_pos_chain),
    ("世界坐标写进 Player：失败写 None，不串视觉平台编号", t_apply_to_player),
    ("实时回路裁面板要复制（别把画上去的框线串进识别）",
     t_live_thread_mmap_panel_copy),
    ("「世界坐标」那行：勾上才显示、数值与读数同步贴到画面", t_world_label_text),
    ("弹窗反馈：抓一帧去掉 / 不报累计帧数 / 叠加图要弹窗", t_dialog_ui_feedback),
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
