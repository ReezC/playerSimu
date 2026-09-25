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

                # ③ 在提示里选「否」时必须说出来（不能说"点了没反应"）
                with mock.patch.object(QMessageBox, "question",
                                       return_value=QMessageBox.No):
                    dlg2._on_save()
                check("没有保存" in dlg2.lbl_say.text(),
                      "拒绝保存后状态行要写明「没有保存」，实际：%r"
                      % dlg2.lbl_say.text())
            finally:
                dlg2._shutdown()
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
    check("btn_mmap_region.setVisible(src_kind == mm.SRC_LIVE)" in rp,
          "「框选…」没有跟着来源显隐（要求：选「从实时画面框选」才出现）")
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
    ("弹窗的持续自动定位（邻近尺度 + 跟丢回退）", t_dialog_auto_fit),
    ("路线识别面板上的入口", t_route_panel_button),
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
