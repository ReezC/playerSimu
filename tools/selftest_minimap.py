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

        # 2) crop 下缩放控件应当被禁掉（1:1 显示，调缩放没意义）
        check(not dlg.sld.isEnabled() and not dlg.sp_scale.isEnabled(),
              "crop 下缩放控件应当是禁用的")

        # 3) 手动拖动叠加层 = 改 view（这就是「手动目测」那条路）
        before = list(dlg.block)
        dlg._ov_item.setPos(-(before[0] + 7), -(before[1] - 3))
        check(dlg.block == [before[0] + 7, before[1] - 3],
              "拖动没反算回 view：%s（应当 %s）"
              % (dlg.block, [before[0] + 7, before[1] - 3]))

        # 4) 方向键微调 1 像素。方向语义 = **挪叠加层**（和拖动一致）：
        #    按左 → 叠加层左移 → 看到的是底图更靠右的一块 → view 的 x 变大。
        from PyQt5.QtCore import QEvent, Qt
        from PyQt5.QtGui import QKeyEvent
        v0 = list(dlg.block)
        dlg.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Left, Qt.NoModifier))
        check(dlg.block == [v0[0] + 1, v0[1]],
              "方向键没把叠加层往左挪 1 像素：view=%s（应当 %s）"
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

    pad = 40
    panel = synth_fit_panel(canvas, 2.0, pad, pad)
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
    ("方式选错时量不出来（不是给个错值）", t_wrong_mode_fails),
    ("标定弹窗（假流，不碰网络/真实标定）", t_dialog_with_stub),
    ("弹窗的持续自动定位（邻近尺度 + 跟丢回退）", t_dialog_auto_fit),
    ("路线识别面板上的入口", t_route_panel_button),
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
