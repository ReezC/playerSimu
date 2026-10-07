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

# --------------------------------------------------------------- 自检不许改用户文件
# ⛔ `config/ui.yaml` 是**用户的**文件（窗口几何 / 字号 / 颜色）⇒ 自检**绝不许**写它 ✗
#    （全仓库规矩 ✓，见 `selftest_main_window._fake_store` 的说明）。
#    本套件会建/关标定弹窗（它们接了 `theme.bind_window_state` ⇒ 一 show/hide 就把几何写
#    进用户的文件 ✗ —— 2026-09-29 逐个套件量出来本套件写 `windows.minimap_calib` /
#    `windows.two_point_calib`）⇒ 把 theme 的**落点**指到临时文件，一个字节都不碰用户的 ✓。
import tempfile as _tempfile                                # noqa: E402

from gui import theme as _theme                             # noqa: E402

_theme.CFG = Path(_tempfile.mkdtemp(prefix="psimu_ui_")) / "ui.yaml"

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

def t_locate_fit_refined():
    """`locate_fit` 的**细化**（2026-09-27）：非整数倍 scale + 亚像素偏移都要量得出来。

    **为什么**（用户 2026-09-27："先把全局小地图弄准"）：粗搜的尺度是**离散**的
    （0.25 一档）、峰值是**整数**像素 ⇒ 真值落在两档中间时每一档都差 6% 以上 ——
    文档里那个「真值 1.874、粗搜给 1.933 ⇒ 最远角差 11 个面板像素 ≈ **93 世界像素**」
    就是这么来的 ✗。这种尺子量出来的"标定偏了多少"没有意义 ⇒ **先把尺子弄准**，
    才谈得上把全局小地图弄准 ✓。

    钉五件：
      ① 非整数倍 scale（1.874）⇒ 细化后的相对误差 < 0.5%；
      ② 偏移**是亚像素**：面板是平移 33.40 个像素做的 ⇒ 量出来必须接近 33.4（不是 33）；
      ③ 细化**不离开窗口**（±`span0/2` = 粗搜格子的一半）：分数面平时不许漂到别的档
         （实测踩过：从 1.25 漂到 0.84，两套几何最远角差 6600 世界像素，全是噪声 ✗）；
      ④ 分数只许**变高**（细化只用更高的分 ✓）；粗搜量不出来时，细化也要能救回来 ✓。
    """
    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return
    true = round(canvas.shape[1] * 1.874) / float(canvas.shape[1])
    p = synth_fit_panel(canvas, 1.874, 40, 40, pad=40)
    # ② 亚像素：再整幅平移 0.4 个像素（双线性）⇒ 真值偏移 = 40.4
    p = cv2.warpAffine(p, np.float32([[1, 0, 0.4], [0, 1, 0.4]]),
                       (p.shape[1], p.shape[0]), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    coarse = mm.locate_fit(p, canvas, refine=False)
    ref = mm.locate_fit(p, canvas)
    check(ref is not None,
          "细化之后反而「量不出来」了 —— 门槛必须在**细化之后**判"
          "（真值落在两档中间时粗搜每档都差 6% ✗）")
    err = abs(ref["scale"] - true) / true
    check(err < 0.005,
          "细化后的 scale 相对误差 %.2f%% > 0.5%%（真值 %.4f、量出 %.4f）"
          % (err * 100.0, true, ref["scale"]))
    check(ref.get("refined") is True, "细化结果没标 `refined`（下游没法知道它是粗是细）")
    check(float(ref["score"]) >= 0.8,
          "合成面板（整张底图压进去）分数不该低：%.3f" % float(ref["score"]))
    ox, oy = (float(v) for v in ref["offset"])
    check(abs(ox - 40.4) <= 0.8 and abs(oy - 40.4) <= 0.8,
          "亚像素偏移没量对（真值 40.40，量出 %.2f / %.2f）—— 是不是又退回整数了？"
          % (ox, oy))
    if coarse is not None:
        check(abs(ref["scale"] - coarse["scale"]) <= 0.125 + 1e-9,
              "细化漂出窗口了（粗 %.4f → 细 %.4f；只许 ±0.125 = 粗搜格子的一半）"
              % (coarse["scale"], ref["scale"]))
        check(float(ref["score"]) >= float(coarse["score"]) - 1e-9,
              "细化之后分数变低了（%.3f → %.3f）—— 越弄越差 ✗"
              % (float(coarse["score"]), float(ref["score"])))


def t_calib_check_and_reverse():
    """`check_calib`（量"这份标定差多少"）+ `world_to_panel`（反向换算，一处实现）。

    用户 2026-09-27：容差按像素定值，但**表述一律世界坐标** ⇒ 核对也必须报**世界像素**
    （现状：`score` 无量纲、`resid_px` 只在两点法里且是面板像素、取证器量的是识别率
    ⇒ "差几个世界像素"**到处问不出来** ✗）。

    钉四件：
      ① 拿"尺子自己"当标定 ⇒ 偏差 ≈ 0（不报假误差 ✓），且 `verdict` 里两种口径都写出来；
      ② 故意把偏移推 5 个面板像素 ⇒ 报出来的**世界**偏差 ≈ 5 ×（世界像素/面板像素），
         而且**主要差在 x**（方向不许反 ✗）；
      ③ 匹配分低于 `TRUST_SCORE` 时**如实说"只能当参考"**（不许给个像模像样的数 ✗）；
         连量都量不出来时要说清为什么 ✓；
      ④ `world_to_panel` 与 `panel_to_world` **互逆**（fit / crop 两种方式都试 ✓）。
    """
    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return
    t = mapdata.load(mid, with_canvas=True)
    p = synth_fit_panel(canvas, 1.5, 40, 40, pad=40)
    loc = mm.locate_fit(p, canvas)
    check(loc is not None, "合成面板定不出来，后面没法验")
    good = {k: loc[k] for k in ("mode", "scale", "offset", "view")}

    # ① 尺子自己 ⇒ 偏差 ~0
    r0 = mm.check_calib(p, t, good)
    check(r0["err_world"] is not None and r0["err_world"] < 1.0,
          "拿尺子自己当标定却报出 %.1f 世界像素的偏差（量错地方了 ✗）" % (r0["err_world"] or -1))
    check("世界像素" in r0["verdict"] and "实时像素" in r0["verdict"],
          "结论文里没把两种口径都写出来（决策④要世界坐标表述、现场看实时像素）：%r"
          % r0["verdict"])

    # ② 推 5 个面板像素 ⇒ 世界偏差 = 5 × (世界像素/面板像素)，主要差在 x
    bad = dict(good, offset=[good["offset"][0] + 5.0, good["offset"][1]])
    r1 = mm.check_calib(p, t, bad)
    want = 5.0 * float(t.px_per_world) / float(good["scale"])
    check(abs(r1["err_world"] - want) / want < 0.05,
          "推 5 个面板像素，报出来的世界偏差该 ≈ %.0f，实际 %.0f"
          % (want, r1["err_world"]))
    check(abs(r1["err_world_x"]) > abs(r1["err_world_y"]),
          "只推了 x，却报「主要差在 y」（方向反了 ✗）：dx=%.1f dy=%.1f"
          % (r1["err_world_x"], r1["err_world_y"]))
    check(float(r1["err_px"]) > 0 and float(r1["err_px"]) < 12,
          "实时像素口径没换算对：%.2f 像素" % float(r1["err_px"]))

    # ③ 匹配分低 ⇒ 必须说"只能当参考"（噪声图：尺子量不出来 ⇒ 说清原因）
    noise = (np.random.RandomState(7).rand(160, 160, 3) * 255).astype(np.uint8)
    r2 = mm.check_calib(noise, t, good)
    check(r2["ok"] is False and r2["why"],
          "噪声上量不出东西却没给理由：%r / %r" % (r2["ok"], r2["why"]))
    check(r2["verdict"] == "" or "参考" in r2["verdict"] or "没量出" in r2["why"],
          "量不出来时不该给个像模像样的结论：%r" % r2["verdict"])

    # ④ 正反换算互逆（fit / crop）
    for cal in (good, {"mode": mm.MODE_CROP, "scale": 1.5, "scale_y": 1.5,
                       "offset": [4, 4], "view": [20, 30]}):
        for (px, py) in ((0.0, 0.0), (12.5, 33.25)):
            wx, wy = mm.panel_to_world(px, py, cal, t)
            bx, by = mm.world_to_panel(wx, wy, cal, t)
            check(abs(bx - px) < 1e-6 and abs(by - py) < 1e-6,
                  "正反换算不互逆（%s）：(%.2f, %.2f) → (%.4f, %.4f)"
                  % (cal["mode"], px, py, bx, by))


def t_route_panel_mmap_check():
    """「实测精度」按钮（2026-09-27 用户要求：把核对做成**一次点击**）。

    它回答这一页最要紧的问题：**"我这份标定差多少世界像素？"** —— 在这之前全仓库都
    问不出来（匹配分是无量纲的"像不像"、两点法的残差是**面板像素**、取证工具量的是
    黄点识别率 ✗）⇒ 只能靠"寻路看起来对不对"猜。

    钉六件：
      ① 归组 + 位置：按钮在「寻路配置」组里，且在「框选小地图」**下面**（量坐标 ✓）；
      ② **只读**：整条链路不许写标定 / 配置 / 项目（写入口全 mock，跑完断言没被调 ✓）；
      ③ 面板取法：来源=从实时画面 ⇒ 按**本项目**框选区域从 `current_frame()` 上裁
         （裁出来的尺寸和像素要对 ✓ —— 裁错地方等于核对了个寂寞 ✗）；
      ④ 读数：结论行写"世界像素"、明细里两种口径都有（决策④ ✓），对得上时报绿 ✓；
      ⑤ 缺东西时说**人话**：没画面 / 没标定 / 没选地图各有各的话（不许静默 ✗）；
      ⑥ 算法只有一处：面板侧只**调用** `mm.check_calib`，自己不许再写一套匹配 ✗。
    """
    import unittest.mock as mock

    from PyQt5.QtCore import QPoint
    from PyQt5.QtWidgets import QApplication, QGroupBox

    import gui.route_panel as rpmod
    from gui.route_panel import RoutePanel

    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return
    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")

    panel = synth_fit_panel(canvas, 1.5, 40, 40, pad=40)      # 整张底图铺进"面板"
    ph, pw = panel.shape[:2]
    ox, oy = 30, 20
    frame = np.zeros((ph + 2 * oy, pw + 2 * ox, 3), np.uint8)  # "实时画面"
    frame[oy:oy + ph, ox:ox + pw] = panel
    crop = [ox, oy, pw, ph]                                   # 本项目框的就是它
    cal = mm.locate_fit(panel, canvas)                        # 这份标定"就是尺子"
    check(cal is not None, "合成面板定不出来，后面没法验")
    good = {k: cal[k] for k in ("mode", "scale", "offset", "view")}
    cfg = {"mmap_src": mm.SRC_LIVE, "mmap_draw": True, "mmap_crop": None}

    class _Live:
        """假实时面板：核对要 `current_frame()`；`bind()` 里那条叠图刷新也要接得住。"""

        def current_frame(self):
            return frame

        def set_minimap_overlay(self, pix, src=None, frame_rect=None,
                                alpha=0.55, note=""):
            pass

        def set_overlay_note(self, note):
            pass

    proj = _Proj(mid, mmap_crop=crop)
    tips = []
    p = RoutePanel()
    p.resize(900, 700)
    try:
        with mock.patch.object(rpmod, "load_live", lambda: dict(cfg)), \
             mock.patch.object(mapdata, "load_calib", lambda _m, src=None: dict(good)), \
             mock.patch.object(mapdata, "save_calib",
                               side_effect=AssertionError("实测精度不该写标定")) \
                as m_save, \
             mock.patch.object(rpmod, "update_live",
                               side_effect=AssertionError("实测精度不该写配置")), \
             mock.patch.object(rpmod.QMessageBox, "information",
                               lambda *a, **k: tips.append(a[2] if len(a) > 2 else "")):
            p.bind(proj)
            # ① 归组 + 位置（真控件树 + 量坐标，不靠代码顺序 ✓）
            grp, q = None, p.btn_mmap_check.parent()
            while q is not None and grp is None:
                grp = q if isinstance(q, QGroupBox) else None
                q = q.parent()
            check(grp is not None and "寻路配置" in grp.title(),
                  "「实测精度」没归在「寻路配置」那一组里：%r"
                  % (grp.title() if grp is not None else None,))
            p.show()
            app.processEvents()
            check(p.btn_mmap_check.mapTo(p, QPoint(0, 0)).y()
                  > p.btn_mmap_crop.mapTo(p, QPoint(0, 0)).y(),
                  "「实测精度」没在「框选小地图」**下面**（用户要求挨着它）")
            p.hide()

            # ③ 面板取法：裁的就是框选那一块（尺寸 + 像素都要对）
            seen = {}
            real_check = mm.check_calib

            def spy(pan, terr, c, **kw):
                seen["shape"] = pan.shape
                seen["same"] = bool(np.array_equal(pan, panel))
                return real_check(pan, terr, c, **kw)

            p.live_panel = _Live()
            with mock.patch.object(mm, "check_calib", spy):
                p._check_mmap_calib()
            check(seen.get("shape") == (ph, pw, 3),
                  "核对拿到的面板尺寸不对（该是框选那一块 %s，实际 %s）"
                  % ((ph, pw, 3), seen.get("shape")))
            check(seen.get("same"),
                  "核对裁错了地方（拿到的不是框选区域那块像素）")

            # ④ 真跑一遍（不 mock）：结论行 + 明细 + 颜色
            p._check_mmap_calib()
            note = p.lbl_check_note.text()
            check("世界像素" in note and "实时像素" in note,
                  "结论行没把两种口径都写出来（决策④）：%r" % note)
            check("1 实时像素" in p.lbl_check_detail.text(),
                  "明细里没写「1 实时像素 = 多少世界像素」：%r" % p.lbl_check_detail.text())
            check("#188038" in p.lbl_check_note.styleSheet(),
                  "拿尺子当标定（偏差≈0）却没报绿：%r（%s）"
                  % (note, p.lbl_check_note.styleSheet()))

            # ⑤ 缺东西说人话（不是静默）
            tips.clear()
            p.bind(None)
            p._check_mmap_calib()
            check(tips and any("选地图" in t for t in tips),
                  "没选地图时没指路：%s" % tips)
            p.bind(proj)
            tips.clear()
            with mock.patch.object(mapdata, "load_calib", lambda _m, src=None: {}):
                p._check_mmap_calib()
            check(tips and any("标定" in t for t in tips),
                  "这条来源还没标定时没指路：%s" % tips)
            p.live_panel = None
            p._check_mmap_calib()
            check("实时" in p.lbl_check_note.text() or any("实时" in t for t in tips),
                  "没画面时没说清去哪开预览：%r / %s" % (p.lbl_check_note.text(), tips))

            # ② 只读：标定/配置/项目一个字都没写
            check(m_save.call_count == 0, "实测精度写了标定文件 ✗")
            check(not proj.saved, "实测精度写了 project.yaml ✗")

        # ⑥ 算法只有一处：面板侧只**调用** `mm.check_calib`，不许自己再写一套匹配 ✗
        src = (ROOT / "gui" / "route_panel.py").read_text(encoding="utf-8")
        body = src.split("def _check_mmap_calib", 1)[1].split("\n    def ", 1)[0]
        check("mm.check_calib(" in body,
              "「实测精度」没走 `perception.minimap.check_calib`（那是对核算法唯一的一份 ✗）")
        check("matchTemplate" not in body and "locate_fit" not in body,
              "「实测精度」里自己又写了一套匹配 —— 规范要求只有检查函数那一份 ✗")
    finally:
        p.close()
        p.deleteLater()


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
    # ⚠ 2026-09-29 起 `view` 是**亚像素**的（`_peak_subpix` 抛物线拟合 + 按贴合分抠到
    #   0.25 底图像素 ✓）⇒ 不能再用"整像素相等"钉它（那会把新精度钉掉 ✗）。
    #   要钉的是**这件事**：差得不超过半个底图像素 ✓
    check(loc["view"] == [] or (abs(loc["view"][0] - (x + inset)) <= 0.5
                                and abs(loc["view"][1] - (y + inset)) <= 0.5),
          "view %s 应该在 [%d, %d] 的半个像素之内" % (loc["view"], x + inset, y + inset))
    # 真正要紧的是这一条：面板左上角换回底图坐标 = 当初裁剪的原点（±0.5 底图像素 ✓）
    cx, cy = mm.panel_to_canvas(0, 0, loc)
    check(abs(cx - x) <= 0.5 and abs(cy - y) <= 0.5,
          "panel_to_canvas 换回 (%s, %s)，应该是 (%d, %d)±0.5" % (cx, cy, x, y))


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
        check(abs(loc["scale"] - 1.0) < 0.01,
              "(%dx%d)@(%d,%d) 是 1:1 裁块，scale 应当是 1.0，实际 %.4f"
              % (w, h, x, y, loc["scale"]))
        got = mm.panel_to_canvas(0, 0, loc)
        # ⚠ 亚像素（见 `t_crop_roundtrip` 那条说明）⇒ 容差半个底图像素 ✓
        check(abs(got[0] - x) <= 0.5 and abs(got[1] - y) <= 0.5,
              "(%dx%d)@(%d,%d) 换算回 (%s, %s)，应该是 (%d, %d)±0.5"
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
    """弹窗：塞假帧 → 自动定位 → 标定字典正确 → 偏移条/方向键能改几何 → 关窗处置收流。

    ⚠ 2026-09-29 起**图上不能拖**了（用户第 ① 条）：改位置只有「x/y 轴偏移」两条拖动条
    与方向键（都是可复现的值 ✓）；「锁定 xy 缩放」与「不改缩放」两个勾也在别的用例里 ✓。
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
        # ⚠ 亚像素（`_peak_subpix` + 按贴合分抠到 0.25 底图像素 ✓）⇒ 容差半个像素 ✓
        #   （放大 5 倍的面板上，1 个底图像素 = 5~10 个面板像素 ⇒ 这半像素是有意义的 ✓）
        check(abs(cx - x) <= 0.5 and abs(cy - y) <= 0.5,
              "弹窗量出来的换算偏了：面板左上 → 底图 (%s, %s)，应该是 (%d, %d)±0.5"
              % (cx, cy, x, y))
        # 判据那行要显示世界坐标与世界范围（人靠它判断对不对）
        # ⚠ 2026-09-29 行文收短了（"世界"只说一次）：断言跟着改成两个**关键信息**都在 ✓
        check("左上" in dlg.lbl_judge.text() and "范围" in dlg.lbl_judge.text(),
              "判据行没显示世界坐标与世界范围：%r" % dlg.lbl_judge.text())

        # 2) crop 下缩放控件**必须可用**：小底图会被客户端放大后取块（实测到
        #    9~10 倍），自动定位失败时就靠它手动对齐 —— 以前这里禁用了，
        #    等于把唯一的手动出路堵死（缩放上限也只有 4 倍，够不着）。
        check(dlg.sld.isEnabled() and dlg.sp_scale.isEnabled(),
              "crop 下缩放控件必须是可用的（放大后取块的图靠它手动对）")
        check(dlg.sp_scale.maximum() >= 12.0,
              "缩放上限应当 ≥12 倍，实际 %.1f" % dlg.sp_scale.maximum())

        # 3) ⭐ **图上不再能拖**（用户 2026-09-29 第 ① 条："移除手动在图示区拖动修改的
        #    功能"✓）—— 改位置只剩「x/y 轴偏移」两条拖动条 + 方向键（都是可复现的值 ✓）。
        from PyQt5.QtWidgets import QGraphicsPixmapItem
        check(not (dlg._ov_item.flags() & QGraphicsPixmapItem.ItemIsMovable),
              "叠加层还是「可拖」的（用户要求去掉图上拖动 ✗）")
        check(not hasattr(dlg, "_on_overlay_moved"),
              "还留着「拖动反算」那条路（`_on_overlay_moved`）—— 现在只有 "
              "`_pull_from_overlay`（方向键用 ✓）")
        before = list(dlg.block)
        s = float(dlg.scale)
        # ④ 两条偏移拖动条：crop 下单位是**底图像素**（fit 才是面板像素）
        dlg.sp_ox.setValue(before[0] + 1.0)
        dlg.sp_oy.setValue(before[1] - 2.0)
        # ⚠ 拖动条**只有 0.1 的精度**（`off_store` ✓），而 `view` 现在是亚像素的
        #   （如 43.998）⇒ 比的是"拖动条把值搬过去了没有"，容差就是它的一格 0.1 ✓
        check(all(abs(dlg.block[i] - (before[i] + (1 if i == 0 else -2))) <= 0.1
                  for i in (0, 1)),
              "「x/y 轴偏移」没写进 view：%s（应当 %s ±0.1）"
              % (dlg.block, [before[0] + 1, before[1] - 2]))
        check(dlg._off_unit() == "底图像素",
              "crop 下偏移单位该写「底图像素」（写错人就会按面板像素去调 ✗）：%s"
              % dlg._off_unit())

        # 4) 方向键微调：一格 = **1 个面板像素**（用户第 ③ 条："不能是底图像素，要尽量小"
        #    ✓）⇒ view 那一步只动 1/scale 个底图像素 ✓。方向语义 = **挪叠加层**：
        #    按左 → 叠加层左移 → 看到的是底图更靠右的一块。
        from PyQt5.QtCore import QEvent, Qt
        from PyQt5.QtGui import QKeyEvent
        v0 = list(dlg.block)
        dlg.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Left, Qt.NoModifier))
        exp = v0[0] + 1.0 / s
        check(abs(dlg.block[0] - exp) < 1e-3 and dlg.block[1] == v0[1],
              "方向键没按「1 个面板像素」挪：view=%s（应当 %.4f）" % (dlg.block, exp))
        # 一步 = **1 个面板像素** ⇒ 底图像素上的步长 = 1/scale。
        #   ⚠ 这张合成面板是 1:1 的（scale=1 ⇒ 1 面板像素 = 1 底图像素，没法更小 ✓）；
        #   真机上 crop 的 scale 是 5~10 ⇒ 一步只有 0.1~0.2 个底图像素 ✓（用户第 ③ 条 ✓）。
        check(abs((dlg.block[0] - v0[0]) * s - 1.0) < 1e-3,
              "方向键一步不是「1 个面板像素」：底图上挪了 %.4f，scale=%.4f"
              % (dlg.block[0] - v0[0], s))
        check((dlg.block[0] - v0[0]) <= 1.0 + 1e-6,
              "方向键一步超过 1 个底图像素（用户要求「要尽量小」✗）：%s → %s"
              % (v0, dlg.block))
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


def t_dot_feet_subpixel():
    """`dot_feet` **不许取整** —— 用户 2026-09-27 报的"同一个位置读数差 9"就是它。

    这张图上 **1 个面板像素 = 8.55 世界单位**（`px_per_world / scale` = 16.082 / 1.880076）
    ⇒ 把亚像素重心（`cv2` 质心）取整 = 把读数量化成 8.55 一格一格跳：**形状半个像素的
    变化**（抗锯齿、压缩噪声、人在绳上/地上时那个点的画法略有差别 —— 眼睛看着"没动"）
    就能让读数跳 **9** ✗✗（现场：L1 上 x=56，顺绳下来 x=47，正差一格）。
    锚点（下沿 = 重心 + (h-1)/2）本身不变，只是**保留小数** ✓。
    """
    x, y = mm.dot_feet({"x": 61.4, "y": 91.0, "h": 6})
    check(abs(x - 61.4) < 1e-9, "x 被取整了（一格 = 8.55 世界单位）：%r" % (x,))
    check(abs(y - 93.5) < 1e-9,
          "脚底锚点被取整了（h=6 的点占 88..93 行 ⇒ 91 + 2.5 = 93.5）：%r" % (y,))
    # 亚像素位移必须**如实体现**（差 0.4 面板像素 ≈ 3.4 世界单位，不许被量化成 8.55）
    x2, _ = mm.dot_feet({"x": 61.8, "y": 91.0, "h": 6})
    check(abs((x2 - x) - 0.4) < 1e-9,
          "亚像素位移被量化掉了：%r → %r（该差 0.4）" % (x, x2))
    # 采样那一侧（界面是整数框）由调用方取整 —— 口径仍是同一个函数 ✓
    check(abs(mm.dot_feet({"x": 61.0, "y": 91.0, "h": 6})[1] - 93.5) < 1e-9,
          "整数输入时脚底锚点算错了：%s" % (mm.dot_feet({"x": 61.0, "y": 91.0, "h": 6}),))
    # 锚点是"下**边缘**"（不是"最后一行"）：h=6、重心 91 ⇒ 92.5..93.5 那一格的下边 = 93.5
    # —— 差值只有半个面板像素（≈4 世界单位），但它决定 y 读数是否还会跳一格 ✓
    check(abs((mm.dot_feet({"x": 61.0, "y": 91.0, "h": 6})[1]
               - mm.dot_feet({"x": 61.0, "y": 91.0, "h": 5})[1]) - 0.5) < 1e-9,
          "点高变化没算对（h 每多 1，下沿多半个像素）：%.3f / %.3f"
          % (mm.dot_feet({"x": 61.0, "y": 91.0, "h": 6})[1],
             mm.dot_feet({"x": 61.0, "y": 91.0, "h": 5})[1]))


def t_calib_dialog_two_axis():
    """「标定…」窗必须认**两轴**（2026-09-27 用户报："双点标定完点开标定窗发现不太对"）。

    事实对比：双点标定写的是 `scale` + `scale_y`（`src=two_point`、**没有** `score`）；
    而这扇窗原来只认**一个**缩放 —— 读 x 轴、叠加层按**等比**画、`calib()` 连 `scale_y`
    都不写 ⇒（a）叠图纵向看着差一点，（b）**"打开看一眼、点一下保存"就把 y 轴抹成 x** ✗，
    还把 `src` 降级成 `manual`、给没有匹配分的几何安上 `score: 0` ✗。

    这条钉六件（每件对应上面一个具体症状）：
      ① 两轴都读进来（`scale` / `scale_y`）；
      ② 叠加层按**两轴各自的缩放**画（不是等比）；
      ③ 没动过 ⇒ 保存后 `scale_y` **还在**、`src` 还是 `two_point`、不凭空多一个 `score`，
         而且量出来的数**一位都不许被舍**（别把 1.880076 改成 1.8801）；
      ④ 人改过缩放/位置之后才降级成 `manual`；
      ⑤ 改缩放时两轴**按载入的比例**一起变（不把"两轴不同"顺手抹平）；
      ⑥ 顶部那行认出"双点标定（两轴 …）"（不然显示成"自动匹配 0.00" ✗）。
    """
    import unittest.mock as mock

    from PyQt5.QtWidgets import QApplication

    from gui.minimap_calib import MinimapCalibDialog

    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return
    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")

    # 用户现场那份（105090600 / stream，双点标定刚存下的样子，原样照抄）
    s_x, s_y = 1.880076070446757, 1.8900600092321893
    cal = {"mode": mm.MODE_FIT, "scale": s_x, "scale_y": s_y,
           "offset": [-0.452, -2.602], "view": [0, 0], "src": "two_point",
           "ref": "canvas", "alpha": 35}
    panel = synth_fit_panel(canvas, 1.88, -1, -3, pad=0)

    with mock.patch.object(mapdata, "load_calib", lambda _m, src=None: dict(cal)):
        dlg = MinimapCalibDialog(mid, mode=mm.MODE_FIT, client=StubClient(panel),
                                 src=mm.SRC_STREAM)
        try:
            # ① 两轴都读进来
            check(abs(dlg.scale - s_x) < 1e-9 and abs(dlg.scale_y - s_y) < 1e-9,
                  "两轴没都读进来：scale=%r scale_y=%r" % (dlg.scale, dlg.scale_y))
            # ② 叠加层逐轴缩放（等比的话 y 轴画成 1.880076 —— 和换算用的数不一致 ✗）
            t = dlg._ov_item.transform()
            check(abs(t.m11() - s_x) < 1e-6 and abs(t.m22() - s_y) < 1e-6,
                  "叠加层没按两轴各自的缩放画：m11=%r m22=%r" % (t.m11(), t.m22()))
            check(abs(t.m11() - t.m22()) > 1e-6,
                  "两轴不同的标定画出来却是等比的（m11 == m22）")
            # ⑥ 顶部那行要认出"双点标定·两轴"
            check("双点" in dlg.lbl_loaded.text(),
                  "顶部没认出这份几何是双点标定的（会显示成「自动匹配 0.00」）：%r"
                  % dlg.lbl_loaded.text())
            # ③ 没动过 ⇒ 原样：两轴都在、来源不降级、不多 `score`、数不舍
            c = dlg.calib()
            check(abs(mm.scales_of(c)[1] - s_y) < 1e-9,
                  "「打开看一眼」就会把 y 轴抹成 x（`scale_y` 没了）：%s" % c)
            check(c.get("src") == "two_point",
                  "没动过却把来源降级了：%s" % c.get("src"))
            check("score" not in c,
                  "没有匹配分的几何被安了一个 score：%s" % c.get("score"))
            check(abs(float(c["scale"]) - s_x) < 1e-9,
                  "没动过却把量出来的缩放舍了：%r（应 %r）" % (c["scale"], s_x))
            check(c.get("offset") == cal["offset"],
                  "没动过却把偏移改了：%s" % c.get("offset"))
            # ⑤ **两条**缩放拖动条（x / y 各一条）—— 用户 2026-09-27 要求：
            #    「旧的标定弹窗应该有 2 个拖动条吧？x 和 y」。（另一条老滑条是透明度，
            #    它不改几何；这里钉的是**缩放**那两条。）
            check(hasattr(dlg, "sld") and hasattr(dlg, "sld_y"),
                  "这扇窗没有两条**缩放**拖动条（x / y）")
            check(hasattr(dlg, "sp_scale_y"),
                  "没有 y 轴那个数字框（拖动条要配一个准数框）")
            # ⭐ **「锁定 xy 缩放」**（用户 2026-09-29 第 ④ 条）：默认勾上 ⇒ 改一条两轴一起变 ✓；
            #   去掉勾才各调各的（双点标定量出两轴不同的场合 ✓）。两种情况都要钉住。
            check(dlg.ck_lock_xy.isChecked(),
                  "「锁定 xy 缩放」默认没勾上（游戏是等比缩放，默认就该锁 ✓）")
            dlg.sp_scale.setValue(2.0)                  # 锁着 ⇒ 改 x，y 跟着
            check(abs(dlg.scale - 2.0) < 1e-6 and abs(dlg.scale_y - 2.0) < 1e-6,
                  "锁着却只改了一轴：x=%.6f y=%.6f" % (dlg.scale, dlg.scale_y))
            check(abs(dlg.sp_scale_y.value() - 2.0) < 1e-6,
                  "锁着时另一轴的数字框没跟着变（界面与几何不一致）：%s"
                  % dlg.sp_scale_y.value())
            dlg.ck_lock_xy.setChecked(False)            # 解锁 ⇒ 各管自己
            dlg.sp_scale.setValue(2.5)                  # 只改 x（数字框那条路；换个值才发信号 ✓）
            check(abs(dlg.scale - 2.5) < 1e-6 and abs(dlg.scale_y - 2.0) < 1e-6,
                  "解锁后改 x 却把 y 也带着动了（两条该各管自己）：x=%.6f y=%.6f"
                  % (dlg.scale, dlg.scale_y))
            dlg.sld_y.setValue(int(round(3.0 * 1000)))  # 只改 y（拖动条那条路）
            check(abs(dlg.scale_y - 3.0) < 1e-6 and abs(dlg.scale - 2.5) < 1e-6,
                  "y 轴那条拖动条没生效 / 把 x 也改了：x=%.6f y=%.6f"
                  % (dlg.scale, dlg.scale_y))
            check(abs(dlg.sp_scale_y.value() - 3.0) < 1e-6,
                  "y 轴数字框没跟着拖动条走（界面和几何不一致）：%s"
                  % dlg.sp_scale_y.value())
            # ④ 人动过之后：来源才降级成 manual、`score` 如实写出来；两轴照实写回
            c2 = dlg.calib()
            check(c2.get("src") == "manual",
                  "人改过缩放却没记成「人眼对齐」：%s" % c2.get("src"))
            check("score" in c2, "人改过之后该如实写上匹配分：%s" % c2)
            # 存起来的是 6 位小数（`calib()` 里 round）⇒ 比**两轴各自的值**
            _s2x, _s2y = mm.scales_of(c2)
            check(abs(_s2x - dlg.scale) < 1e-5 and abs(_s2y - dlg.scale_y) < 1e-5,
                  "人动过之后两轴没照实写回：%s（界面是 %.6f / %.6f）"
                  % (c2, dlg.scale, dlg.scale_y))
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
            # ⚠ 从 2026-09-27 起叠加层走 `setTransform`（两轴各自缩，`setScale` 只有一个
            #   数 ⇒ 双点标定的 y 轴差别画不出来 ✗）⇒ 查缩放要读**变换矩阵**，不能读
            #   `item.scale()`（那个属性在 setTransform 之后恒为 1）
            _tt = dlg._ov_item.transform()
            check(abs(_tt.m11() - dlg.scale / dlg._ref_zoom) < 1e-6
                  and abs(_tt.m22() - dlg.scale_y / dlg._ref_zoom) < 1e-6,
                  "叠加层缩放要按 ref_zoom 折算（两轴各自）：实际 %.4f / %.4f"
                  % (_tt.m11(), _tt.m22()))
            dlg._pull_from_overlay()         # 反算要能回到同一套几何（方向键走的就是这条 ✓）
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


def t_calib_open_says_why_blank():
    """一打开"看着不对劲"的两种情况必须**说出来**（否则会被当成"没保存"）：

      ① 文件里是 `alpha: 0` ⇒ 叠图**完全透明**、一片空白（几何其实存着）；
      ② 文件里的几何是**另一种显示方式**量的（fit ↔ crop 的 `offset`/`view` 含义不同）
         ⇒ 位置整个不对，看着也像"没保存"。

    用户 2026-09-27 报"打开普通标定、点保存，再打开发现没保存"，而他文件里正是 `alpha: 0`。
    """
    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return

    import tempfile
    import unittest.mock as mock

    from PyQt5.QtWidgets import QApplication

    from gui.minimap_calib import MinimapCalibDialog

    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")
    panel = synth_fit_panel(canvas, 1.88, -1, -2, pad=0)
    tmpdir = Path(tempfile.mkdtemp(prefix="mmblank_"))
    tmp = tmpdir / ("%s.mapcalib.json" % mid)
    try:
        with mock.patch.object(mapdata, "calib_path", lambda _mid: tmp):
            mapdata.save_calib(mid, {"mode": mm.MODE_FIT, "scale": 1.88,
                                     "scale_y": 1.89, "offset": [-0.45, -1.6],
                                     "view": [0, 0], "src": "two_point",
                                     "ref": "canvas", "alpha": 0}, mm.SRC_STREAM)
            # ① 透明度 0 ⇒ 状态行说"几何是存着的"、那行读数写"看不见"
            dlg = MinimapCalibDialog(mid, mode=mm.MODE_FIT,
                                     client=StubClient(panel), src=mm.SRC_STREAM)
            try:
                check("0%" in dlg.lbl_say.text() and "存着" in dlg.lbl_say.text(),
                      "透明度 0%% 时没说清「几何是存着的、只是看不见」：%r"
                      % dlg.lbl_say.text())
                check("看不见" in dlg.lbl_alpha.text(),
                      "透明度那行没写「叠图看不见」：%r" % dlg.lbl_alpha.text())
                check(dlg.calib().get("alpha") == 0,
                      "0%% 被当默认值改掉了（0 是合法存档值）：%s"
                      % dlg.calib().get("alpha"))
            finally:
                dlg._shutdown()
            # ② fit 的几何 + crop 方式打开 ⇒ 要说清"这是另一种方式量的"
            dlg2 = MinimapCalibDialog(mid, mode=mm.MODE_CROP,
                                      client=StubClient(panel), src=mm.SRC_STREAM)
            try:
                check("方式" in dlg2.lbl_say.text() and "fit" in dlg2.lbl_say.text(),
                      "显示方式与文件不一致时没说：%r" % dlg2.lbl_say.text())
            finally:
                dlg2._shutdown()
    finally:
        import shutil
        shutil.rmtree(str(tmpdir), ignore_errors=True)


def t_calib_windows_share_store():
    """「标定…」和「双点标定」是**同一份数据**（用户 2026-09-27 报"打开普通标定、点保存，
    再打开发现没保存"）。

    两扇窗写/读的都是 `datasets/map/<id>.mapcalib.json` 的 `sources.<来源>`
    （来源由调用方给：`route_panel._mmap_src()` —— 两处传的都是它 ✓）。这条用例用
    **真文件**（临时路径，不碰 `datasets/`）把这条回路走一遍，钉四件：
      ① 双点标定写进去的几何，「标定…」一打开就认（两轴、`src=two_point`、没有 `score`）；
      ② 在里面**没动过**就保存 ⇒ 那条来源的每个字段**原样**（不该降级成 manual / 抹平两轴 /
         给没有匹配分的几何安 `score` / 舍掉量出来的精度）；
      ③ 在里面**动过**再保存 ⇒ 重开一扇新的也看得到，且记为 `manual`；
      ④ 存**另一条来源**不动本条（`save_calib` 的合并）。
    ⚠ 为什么单列一条：老用例都把 `save_calib` 打桩了 ⇒ 写盘与"按来源合并"那一路根本测不到，
    而"存了却读不见"恰恰只会出错在那里。
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
    check(app is not None, "建不起 QApplication")

    panel = synth_fit_panel(canvas, 1.88, -1, -2, pad=0)
    tmpdir = Path(tempfile.mkdtemp(prefix="mmshare_"))
    tmp = tmpdir / ("%s.mapcalib.json" % mid)
    try:
        with mock.patch.object(mapdata, "calib_path", lambda _mid: tmp):
            # ① 双点标定那条路：解几何 → 存（真 `calib_from_two_point` + 真 `save_calib`）
            ch, cw = canvas.shape[:2]
            cA, cB = (5.0, 4.0), (float(cw - 6), float(ch - 5))
            s_x, s_y, o_x, o_y = 1.88, 1.892, -0.452, -2.602
            r = mm.solve_two_point(cA, (cA[0] * s_x + o_x, cA[1] * s_y + o_y),
                                   cB, (cB[0] * s_x + o_x, cB[1] * s_y + o_y))
            check(r["ok"], "两点解没成：%s" % (r.get("why"),))
            mapdata.save_calib(mid, mm.calib_from_two_point(
                {"mode": mm.MODE_FIT, "alpha": 35, "ref": "canvas",
                 "world_offset": [0, 0]}, r), mm.SRC_STREAM)
            check(tmp.exists(), "双点标定那条路没写文件：%s" % tmp)
            before = json.loads(tmp.read_text(encoding="utf-8"))
            check(mm.SRC_STREAM in before.get("sources", {}),
                  "写进去的不是 sources.<来源>：%s" % before)

            # 「标定…」看得见**同一份**（这就是用户要的"一套数据"）
            dlg = MinimapCalibDialog(mid, mode=mm.MODE_FIT,
                                     client=StubClient(panel), src=mm.SRC_STREAM)
            try:
                check(abs(dlg.scale - r["scale"]) < 1e-9
                      and abs(dlg.scale_y - r["scale_y"]) < 1e-9,
                      "「标定…」没读到双点标定那份几何：%.6f / %.6f"
                      % (dlg.scale, dlg.scale_y))
                check("双点" in dlg.lbl_loaded.text(),
                      "顶部没认出这是双点标定那份：%r" % dlg.lbl_loaded.text())
                # ② 没动过就保存 ⇒ 那条来源的每个字段原样（含不该多出来的 score）
                with mock.patch.object(QMessageBox, "question",
                                       return_value=QMessageBox.Yes):
                    dlg._on_save()
                check(dlg.saved, "「标定…」没保存成功：%r" % dlg.lbl_say.text())
                g0 = before["sources"][mm.SRC_STREAM]
                g1 = json.loads(tmp.read_text(encoding="utf-8"))["sources"][mm.SRC_STREAM]
                for _k in ("mode", "scale", "scale_y", "offset", "view", "src",
                           "alpha", "world_offset", "ref"):
                    check(g0.get(_k) == g1.get(_k),
                          "「没动过就保存」把 %s 改了：%r → %r"
                          % (_k, g0.get(_k), g1.get(_k)))
                check("score" not in g1,
                      "没动过却给双点几何安了 score：%s" % g1.get("score"))
                # ③ 动过再保存 ⇒ 重开一扇新的看得到，且记 manual
                dlg.sp_scale.setValue(1.5)
                with mock.patch.object(QMessageBox, "question",
                                       return_value=QMessageBox.Yes):
                    dlg._on_save()
                check(dlg.saved, "改完没保存成功：%r" % dlg.lbl_say.text())
            finally:
                dlg._shutdown()
            dlg2 = MinimapCalibDialog(mid, mode=mm.MODE_FIT,
                                      client=StubClient(panel), src=mm.SRC_STREAM)
            try:
                check(abs(dlg2.scale - 1.5) < 1e-6,
                      "重开那扇没看到刚保存的缩放：%.4f" % dlg2.scale)
                check(dlg2.calib().get("src") == "manual",
                      "人改过却还记成 %r" % dlg2.calib().get("src"))
            finally:
                dlg2._shutdown()

            # ④ 存另一条来源**不动**本条
            mapdata.save_calib(mid, {"mode": mm.MODE_FIT, "scale": 2.0,
                                     "offset": [1, 1], "view": [0, 0]},
                               mm.SRC_LIVE)
            d = json.loads(tmp.read_text(encoding="utf-8"))
            check(mm.SRC_STREAM in d["sources"] and mm.SRC_LIVE in d["sources"],
                  "存另一条来源把原来的抹掉了：%s" % list(d["sources"]))
            check(abs(d["sources"][mm.SRC_STREAM]["scale"] - 1.5) < 1e-6,
                  "存另一条来源动了本条：%s" % d["sources"][mm.SRC_STREAM])
    finally:
        shutil.rmtree(str(tmpdir), ignore_errors=True)


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
            # ① 用户那条路：用「x/y 轴偏移」两条拖动条把位置调到正确的地方
            #    （不做自动定位 → 没有匹配分，测的正是"手工那一路"✓）。
            #    ⚠ 以前这里用 `_ov_item.setPos(...)` 假装"拖"—— 2026-09-29 起图上不能拖
            #    了（用户第 ① 条 ✓），必须走拖动条/数字框这条路 ✓。
            stub = StubClient(panel)
            dlg = MinimapCalibDialog(mid, mode=mm.MODE_CROP, client=stub)
            try:
                check("还没标定过" in dlg.lbl_loaded.text(),
                      "第一次打开要写明「这张图还没标定过」（不然人分不清现在摆的"
                      "是上次存的还是程序猜的），实际：%r" % dlg.lbl_loaded.text())
                dlg._on_tick()
                check(dlg._frame is not None, "弹窗没取到帧")
                # crop 下「轴偏移」就是 `view`（底图像素 ✓）：调到 (x + inset, y + inset)
                # 就等于把叠加层摆到"面板 (inset, inset) 对底图 (x, y)" ✓
                dlg.sp_ox.setValue(float(x + dlg.inset))
                dlg.sp_oy.setValue(float(y + dlg.inset))
                check(list(dlg.block) == [x + dlg.inset, y + dlg.inset],
                      "「轴偏移」没写进 view：%s" % (dlg.block,))
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
                #    ⚠ 2026-09-29 起提醒的判据是**按重叠区算的贴合分**（用户第 ⑤ 条 ✓）：
                #    刚对齐好的几何分数很高 ⇒ **不再弹窗**（这正是那条要求的目的 ✓）⇒
                #    想测"拒绝保存"得先把几何挪歪（挪 30 个底图像素，贴合分必掉 ✓）。
                dlg2.sp_ox.setValue(float(saved["view"][0]) + 30.0)
                with mock.patch.object(QMessageBox, "question",
                                       return_value=QMessageBox.No):
                    dlg2._on_save()
                check("没有保存" in dlg2.lbl_say.text(),
                      "拒绝保存后状态行要写明「没有保存」，实际：%r"
                      % dlg2.lbl_say.text())
                dlg2.sp_ox.setValue(float(saved["view"][0]))   # 挪回来（下面还有检查 ✓）
            finally:
                dlg2._shutdown()

            # ④ 关窗守卫：动过又没存 → 必须问一句；选「否」= 放弃并真的关掉
            dlg3 = MinimapCalibDialog(mid, mode=mm.MODE_CROP,
                                      client=StubClient(panel))
            try:
                dlg3._on_tick()
                check(not dlg3._unsaved(), "刚打开就报「有没保存的改动」")
                # 动一下（**图上不能拖**了 ⇒ 走「轴偏移」那条 ✓）：
                dlg3.sp_ox.setValue(float(dlg3._off_get(0)) + 5.0)
                check(dlg3._unsaved(), "改过偏移之后应当认出「有没保存的改动」")

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
                dlg4.sp_ox.setValue(float(dlg4._off_get(0)) + 3.0)   # 动一下（图上不能拖 ✓）
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


def t_calib_save_drops_stale_scale_y():
    """「标定」保存把几何改成**等比**后，旧的双点标定 `scale_y` 不许残留（用户 2026-09-28 报）。

    现场："用小地图定位组的「标定」修改标定数据后，玩家的世界坐标没有变化，重开弹窗发现
    上次修改没被应用（它和双点标定应该共用一套数据，两种标定法出的数据不一样就是 bug）"。

    根因：`_on_save` 用 `cal.update(self.calib())` 合并新几何，而 `self.calib()` 在两轴相同时
    **不写 `scale_y`**（`with_scales` 抹掉 ✓）—— 但 `dict.update` **不删旧键** ⇒ 旧的双点
    标定 `scale_y`（比如 1.94）残留下来，和新的等比 `scale`（比如 1.874）混在一起 ⇒
    世界坐标 y 轴按错的缩放算（实测 106010105 正是这个形状：scale=1.874 / scale_y=1.940…）。
    """
    import json
    import shutil
    import tempfile
    import unittest.mock as mock

    from PyQt5.QtWidgets import QApplication, QMessageBox

    from gui.minimap_calib import MinimapCalibDialog

    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return
    app = QApplication.instance() or QApplication([])

    tmpdir = Path(tempfile.mkdtemp(prefix="mmcalib_scale_y_"))
    tmp = tmpdir / ("%s.mapcalib.json" % mid)
    # 双点标定的两轴结果（像 105090600）
    cal0 = {"mode": "fit", "scale": 1.880076, "scale_y": 1.9402855390486573,
            "offset": [-0.45, -1.6], "view": [0, 0], "src": "two_point",
            "ref": "canvas", "alpha": 50}
    tmp.write_text(json.dumps({"v": 2, "sources": {"stream": cal0}},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        with mock.patch.object(mapdata, "calib_path", lambda _m: tmp):
            dlg = MinimapCalibDialog(mid, mode=mm.MODE_FIT, src="stream",
                                     client=StubClient(None))
            try:
                check(dlg._src_loaded == "two_point", "前提不成立：读入的该是双点标定")
                # 模拟「自动定位」把几何改成等比：scale_y 跟 scale 走（_locate_now 就这么干的）
                dlg.scale = 1.874
                dlg.scale_y = 1.874
                dlg.manual = False
                with mock.patch.object(QMessageBox, "question",
                                       return_value=QMessageBox.Yes):
                    dlg._on_save()
                saved = mapdata.load_calib(mid, "stream") or {}
                check("scale_y" not in saved,
                      "几何改成等比后，旧的双点标定 scale_y 还残留在文件里（%r）—— 世界坐标"
                      " y 轴会按错的缩放算，看着像\"改了没生效 / 两种标定法数据不一样\" ✗"
                      % (saved.get("scale_y"),))
                check(float(saved.get("scale") or 0) == 1.874,
                      "等比保存的 scale 不对：%r" % saved.get("scale"))
            finally:
                dlg._shutdown()
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
        # ⚠ 亚像素（见 `t_crop_roundtrip` 那条说明）⇒ 容差半个底图像素 ✓
        check(abs(got[0] - x) <= 0.5 and abs(got[1] - y) <= 0.5,
              "弹窗量出来的换算偏了：面板左上 → 底图 %s，应该是 (%d, %d)±0.5"
              % (got, x, y))
    finally:
        dlg._shutdown()


def t_live_source_wiring():
    """来源这条路 + 「框选小地图」的**接线**（源码约定）。

    2026-09-27 用户要求：框选小地图**搬回「路线识别 → 寻路配置」**（就在「寻路编辑器」
    正下方）—— 它必须**按项目（地图）**各存一份：不同地图的小地图面板尺寸/位置完全
    不同，放"设置"（全局一份）里换个项目就是错的 ✗。⇒ 这条用例钉四件事：

      ① 面板里来源那条路没坏（下拉 / 处理 / 裁帧适配器 / 只存来源）；
      ② 框选**在面板里**（按钮 + 两个处理器），**设置窗里一点痕迹都不许有**（只能一处 ✗）；
      ③ 落点是**项目**（`set("mmap_crop", …, save=True)`），谁都不许再往
         `config/live.yaml` 写它 ✗（那正是"换项目就错"的来源）；
      ④ 主窗口把实时页给**面板**（框选要拿当前帧），并且**不再**给设置窗塞（那儿已经没了）。
    """
    rp = (ROOT / "gui" / "route_panel.py").read_text(encoding="utf-8")
    for kw in ("cmb_mmap_src", "_on_mmap_src", "LiveFrameRegionClient",
               "update_live(mmap_src="):
        check(kw in rp, "路线识别面板里缺 %s（来源这条路没接上）" % kw)
    for kw in ("btn_mmap_crop", "_pick_mmap_crop", "_verify_mmap_crop",
               "lbl_mmap_crop", "select_region_on_image", "region_match_score",
               'set("mmap_crop"'):
        check(kw in rp, "路线识别面板里缺 %s（「框选小地图」没接回这一页）" % kw)

    sd = (ROOT / "gui" / "settings_dialog.py").read_text(encoding="utf-8")
    for kw in ("btn_mmap_crop", "_pick_mmap_crop", "_verify_mmap_crop",
               "lbl_mmap_crop", "mmap_crop", "update_live(mmap_crop="):
        check(kw not in sd,
              "设置窗里还有框选小地图的东西（%s）—— 它按项目存，只能长在路线识别页 ✓"
              % kw)

    # 框选一律走 region_picker（放大镜/像素网格/Esc/<4px 当误点 —— UI规范 §8）：
    # 只许**调用**它，不许在面板里再写一份（当年就是这么长出第二份框选的）
    body = rp.split("def _pick_mmap_crop", 1)[-1].split(
        "def _verify_mmap_crop", 1)[0]
    check("select_region_on_image(" in body,
          "框选没走 gui/region_selector（它套着带放大镜的 region_picker）")
    check("mousePressEvent" not in body and "QDialog" not in body,
          "框选在面板里被自己实现了一遍 —— 规范要求只有 region_picker 一份")
    # 「框完当场验证」只有一份（`perception.minimap.region_match_score`）；
    # 「按项目取框选区域」也只有一处（`perception.minimap.crop_of`）
    mmtxt = (ROOT / "perception" / "minimap.py").read_text(encoding="utf-8")
    check("region_match_score" in mmtxt,
          "当场验证没抽到 perception.minimap（两处各算一遍 = 迟早分叉）")
    check("def crop_of(" in mmtxt and 'get("mmap_crop")' in mmtxt,
          "「按项目取框选区域」的口径不在 perception.minimap.crop_of 里（一处实现 ✗）")
    check('"mmap_crop"' in (ROOT / "gui" / "project.py").read_text(encoding="utf-8"),
          "project.yaml 的默认结构里没有 mmap_crop 这一格（按项目存往哪儿写）")
    check("update_live(mmap_crop=" not in rp,
          "还有人在往 config/live.yaml 写 mmap_crop —— 那正是「换项目就错」的来源 ✗")

    mw = (ROOT / "gui" / "main_window.py").read_text(encoding="utf-8")
    check("self.route_panel.live_panel = self.live_panel" in mw,
          "主窗口没把实时页给路线识别面板 —— 「框选小地图」拿不到当前帧")
    check("dlg.live_panel" not in mw,
          "主窗口还在给设置窗塞实时页 —— 框选已经不在那儿了（留着只会让人以为还在 ✗）")
    check("dlg.show()" in mw and "_settings_dlg" in mw,
          "设置窗没按**非模态**开（`show()` + 单实例 `_settings_dlg`）—— "
          "改完浓淡要对着实时画面看效果，锁住整个工作台就只能关窗重开 ✗")
    # ⚠ 只找**调用**（`dlg.exec_()`）—— 方法说明里就写着"为什么不再用 exec_()"，
    #   拿 `exec_()` 全文找会把**解释**一起判红（这个形状踩过好几次 ✗）
    _body = mw.split("def _on_settings", 1)[-1].split("\n    def ", 1)[0]
    check("SettingsDialog(" in _body and "dlg.exec_()" not in _body,
          "设置窗又用回了 exec_()（模态 ⇒ 开着它没法去看实时画面）")

    live = (ROOT / "config" / "live.yaml").read_text(encoding="utf-8")
    check("mmap_src" in live, "config/live.yaml 里没有 mmap_src（来源存哪）")
    # 老的那份全局 `mmap_crop` **允许留着**：它是升级前唯一的一份，现在只当**只读兜底**
    # （本项目还没框过时先用着它 ✓，见 perception.minimap.crop_of 的说明）。


def t_mmap_crop_is_per_project():
    """框选区域**按项目取**（`perception.minimap.crop_of`）：项目优先、没框过回退老的那份。

    用户 2026-09-27 现场：不同地图的小地图面板**尺寸/位置完全不同** —— 全局一份的话
    换个项目还是上一张图的框 ⇒ 叠图/定位全错 ✗。所以三种情况都要钉住：
      · 本项目框过     ⇒ 用**本项目**的（哪怕老的那份全局值还在 ✓）；
      · 本项目没框过   ⇒ 用老的那份（升级后不用重框就能接着用 ✓）；
      · 都没有 / 没项目 ⇒ None（界面就说"还没框"，不许编一个数 ✗）；
      · 坏值（不是 4 个数）⇒ 当"没填"，别拿半截数去裁画面 ✗。
    """
    a = _Proj("111111111", mmap_crop=[1, 2, 3, 4])
    b = _Proj("222222222")                          # 本项目还没框过
    cfg = {"mmap_crop": [9, 9, 9, 9]}               # 老的那份（全局）
    check(mm.crop_of(a, cfg) == [1, 2, 3, 4],
          "本项目框过了，却不是用本项目的：%s" % (mm.crop_of(a, cfg),))
    check(mm.crop_of(b, cfg) == [9, 9, 9, 9],
          "本项目没框过时没用老的那份兜底（升级后要重框才行 ✗）：%s"
          % (mm.crop_of(b, cfg),))
    check(mm.crop_of(b) is None and mm.crop_of(None, {}) is None,
          "两边都没有却说「有一个框」（界面就会说谎 ✗）")
    check(mm.crop_of(_Proj("333", mmap_crop=[1, 2, 3]), cfg) == [9, 9, 9, 9],
          "长度不是 4 的坏值该当「没填」（别拿半截数去裁画面 ✗）：%s"
          % (mm.crop_of(_Proj("333", mmap_crop=[1, 2, 3]), cfg),))


def t_mmap_rows_split():
    """「小地图来源」**搬进「寻路配置」卡**（用户 2026-10-03 ✓ 原话："「小地图定位」的来源、
    黄点跟踪参数挪到寻路配置最上方"）+ 原来那条"两行不许挤一行"的口径照旧管着。

    **为什么量坐标，而不是匹配源码**：源码里 `row.addWidget` 写在哪儿，和它到底
    落在哪一行是两回事 —— 一个带 stretch 的 HBox 里，往右加多少次都还是同一行。
    只有把面板真建出来量 y，才说明视觉结果。

    ⚠ **为什么这条钉子的口径变了**（2026-10-03 ✓）：它原来钉的是"来源在「小地图定位」
      卡的第二行、和「标定…」同行" ✗ —— 用户要求把来源挪去「寻路配置」⇒ 那条前提**不存在了**
      （照旧写会假红 ✗）。改成钉"**搬到位**"：来源的父组 = 「寻路配置」、整卡在「小地图定位」
      之上；「标定…」**留在**「小地图定位」（它读的就是那张卡里的几何 ✓）且不再与来源同行 ✓；
      窄窗口下来源那行的标签仍不许被压没 ✓。
    """
    from PyQt5.QtCore import QPoint
    from PyQt5.QtWidgets import QApplication, QGroupBox, QLabel

    from gui.route_panel import RoutePanel

    app = QApplication.instance() or QApplication([])
    p = RoutePanel()
    p.resize(900, 700)
    p.show()
    app.processEvents()

    def y(w):
        return w.mapTo(p, QPoint(0, 0)).y()

    def group_of(w):
        q = w.parent()
        while q is not None and not isinstance(q, QGroupBox):
            q = q.parent()
        return q.title() if q is not None else None

    try:
        check(group_of(p.cmb_mmap_src) == "寻路配置",
              "「小地图来源」没搬进「寻路配置」卡（用户 2026-10-03 要求）：%r"
              % (group_of(p.cmb_mmap_src),))
        check(y(p.cmb_mmap_src) < y(p.cmb_mmap_mode),
              "「来源」还在「小地图定位」卡那把（该在它**上面**那张卡里）：y %d vs %d"
              % (y(p.cmb_mmap_src), y(p.cmb_mmap_mode)))
        check(group_of(p.btn_mmap_calib) == "小地图定位",
              "「标定…」被连坐搬走了 —— 它该留在读那张几何的卡里：%r"
              % (group_of(p.btn_mmap_calib),))
        check(y(p.btn_mmap_calib) != y(p.cmb_mmap_src),
              "「标定…」和「来源」挤成同一行了（两张卡的东西混一行 ✗）：%d"
              % (y(p.btn_mmap_calib),))
        # 跟踪那四个容差也一起搬了（用户同一句话里要求的 ✓）；两行分组由
        # `t_track_rows_layout` 量着 ✓ 这里只钉"在不在「寻路配置」里" ✓
        check(all(group_of(sp) == "寻路配置" for sp in p._sp_track.values()),
              "「黄点跟踪参数」没跟着搬进「寻路配置」：%s"
              % ({k: group_of(sp) for k, sp in p._sp_track.items()},))

        # 窄窗口下「来源」那行仍在（标签不许被压没）
        p.resize(520, 700)
        app.processEvents()
        _lbs = [lb for lb in p.findChildren(QLabel) if lb.text() == "小地图来源"]
        check(_lbs and _lbs[0].width() > 0,
              "窗口变窄之后「小地图来源」标签被压没了（§4 说的就是这个后果）")
    finally:
        p.close()


def t_route_panel_current_map_row():
    """⭐⭐ 路线识别 →「寻路配置」**最顶部**那行「当前地图」+「手动更换」。
    （用户 2026-10-02 ✓ 原话："在路线识别→寻路配置最顶部 增加只读参数『当前地图』，
    默认为 模型训练 页签→识别目标选项里的地图参数；增加按钮『手动更换』，用以修改这个 id"）

    钉七件：
      ① **真的在最顶部**：它比同组第一个控件（「寻路编辑器」）还靠上 —— 量**坐标**，
         不看源码顺序（同 `t_mmap_rows_split` 的理由：源码顺序 ≠ 视觉位置 ✓）；
      ② **只读**：显示是 `QLabel`（没有输入框），且**两处文本一字不差**
         （本组顶部 +「小地图定位」卡里那个「当前地图」⇒ 一件事一个口径 ✓）；
      ③ 默认跟着项目的**地图参数**走（`project.map_id` ✓ 与「模型训练 → 识别目标」同一份），
         格式 = `名字_id`（用户给的样式 ✓ 例 `森林迷宫III_105040303`）；
      ④ 「没打开项目」和「项目里还没选地图」**两种说法**（不许都写"没打开项目" ✗）；
      ⑤ 「手动更换」走**同一个写口** `wzexport.apply_map_choice`（源码级钉 route_panel 真调它；
         功能级钉那个写口本身：地图 + 怪列表 + `mobs_cleared` 四件一起写 ✓、清单里没有就
         **一个字节都不写** ✗）；
      ⑥ 换图要**通知出去**（`map_changed` 信号 —— 玩家的「定点休息」集合下拉按地图 id 存 ✓）；
      ⑦ 弹窗：**没选中不许确定**、默认（没输关键词时）只列当前那张、筛选走 `list_maps`
         （清单几百张图，全列出来没法看 ✓ 也**别自己写一套匹配** ✗）。

    ⚠ 清单**喂一份假的**（`wzexport.list_maps` 打桩 ✓）：这条钉子不该依赖本机 `datasets`
      里恰好有什么图（那种用例换台机器就红 ✗）。
    """
    import types

    from PyQt5.QtCore import QPoint, Qt
    from PyQt5.QtWidgets import QApplication, QLabel

    from core import wzexport
    from gui import map_picker as mp_mod
    from gui.route_panel import RoutePanel

    POOL = [{"id": "105040303", "name": "森林迷宫III", "region": "", "mobs": ["1", "2"],
             "mob_names": ["蜗牛", "蓝蜗牛"], "has_mob": True, "score": 1,
             "label": "105040303   森林迷宫III   · 蜗牛/蓝蜗牛"}]
    real_lm = wzexport.list_maps

    def _fake_lm(only_with_mob=True, keyword=""):
        """假的清单：**照真实现的语义**按关键词过滤（真实现是打分的，这里子串就够 ✓）——
        打桩**不能把筛选也顺手关掉** ✗，否则"筛不到要说清"那条永远测不出来（第一版就踩了 ✓）。
        """
        kw = (keyword or "").strip().lower()
        return [dict(x) for x in POOL
                if not kw or kw in str(x["label"]).lower()]

    wzexport.list_maps = _fake_lm
    app = QApplication.instance() or QApplication([])

    class _Proj:
        """只有 RoutePanel 会用到的那几件（`get` / `set` / `save` ✓）。"""

        def __init__(self):
            self.data = {}
            self.saved = 0

        def get(self, k, d=None):
            return self.data.get(k, d)

        def set(self, k, v, save=False):
            self.data[k] = v

        def save(self):
            self.saved += 1

    p = RoutePanel()
    try:
        p.resize(900, 700)
        p.show()
        app.processEvents()

        # ④ 没打开项目
        check("没打开项目" in p.lbl_cur_map.text(),
              "没有项目时「当前地图」没说清：%r" % p.lbl_cur_map.text())
        # ④' 有项目、但没选地图 ⇒ **另一种**说法
        proj = _Proj()
        p.project = proj
        p._refresh_current_map()
        check("还没选地图" in p.lbl_cur_map.text(),
              "项目开着却没选地图时，该说「还没选地图」（不是「没打开项目」✗）：%r"
              % p.lbl_cur_map.text())

        # ③ 默认跟着项目的地图参数走 + ② 两处一致
        proj.data["map_id"] = "105040303"
        p._refresh_mmap()                        # 真走那条刷新链（bind / showEvent 走的也是它 ✓）
        app.processEvents()
        want = "森林迷宫III_105040303"
        check(p.lbl_cur_map.text() == want,
              "「当前地图」没按「名字_id」显示（用户给的样式 ✓）：%r" % p.lbl_cur_map.text())
        check(p.lbl_mmap.text() == want,
              "同一件事两处显示不一样（本组顶部 %r vs 小地图定位卡 %r）—— 会让以为它是两个东西 ✗"
              % (p.lbl_cur_map.text(), p.lbl_mmap.text()))
        check(isinstance(p.lbl_cur_map, QLabel),
              "「当前地图」该是**只读**的 QLabel（用户要的是只读参数 ✓）")
        check(hasattr(p, "btn_map_change") and p.btn_map_change.text() == "手动更换",
              "这一行没有「手动更换」按钮（用户点名要的 ✓）")

        # ① 真在最顶部（比同组第一个控件还靠上）
        def y(w):
            return w.mapTo(p, QPoint(0, 0)).y()

        check(y(p.lbl_cur_map) < y(p.btn_zones),
              "「当前地图」没在「寻路配置」最顶部（y %d vs 寻路编辑器 %d）—— "
              "下面每一件都以「现在是哪张图」为前提，得先说清 ✓"
              % (y(p.lbl_cur_map), y(p.btn_zones)))
        check(y(p.lbl_cur_map) < y(p.btn_mmap_crop),
              "「当前地图」跑到「框选小地图」下面去了（y %d vs %d）"
              % (y(p.lbl_cur_map), y(p.btn_mmap_crop)))

        # ⑤ 写口只有一个：route_panel / 卡片 都调 `apply_map_choice`
        _rp = (ROOT / "gui" / "route_panel.py").read_text(encoding="utf-8")
        _cd = (ROOT / "gui" / "steps" / "cards.py").read_text(encoding="utf-8")
        check("wzexport.apply_map_choice(" in _rp and "MapPickDialog" in _rp,
              "「手动更换」没走共用的选图弹窗 + 共用写口（自己 set map_id 会漏写怪列表 ✗）")
        check("wzexport.apply_map_choice(" in _cd,
              "「模型训练 → 识别目标」那个下拉没走共用写口 —— 两处各写一遍迟早分叉 ✗")
        # ⑤' 功能级：那个写口自己对不对
        proj.data.clear()
        proj.saved = 0
        m = wzexport.apply_map_choice(proj, "105040303")
        check(m is not None and proj.data.get("map_id") == "105040303",
              "写口没把地图写进项目：%r" % (proj.data,))
        check(proj.data.get("mobs") == ["1", "2"]
              and proj.data.get("mob_names") == ["蜗牛", "蓝蜗牛"],
              "写口没把**怪列表**一起写（「确认要识别怪物」会空着 ✗ 实测踩过）：%r" % (proj.data,))
        check(proj.data.get("mobs_cleared") is False and proj.saved == 1,
              "写口没把「上一张的清空记录」作废 / 没落盘：%r saved=%d"
              % (proj.data, proj.saved))
        before = dict(proj.data)
        check(wzexport.apply_map_choice(proj, "999999999") is None
              and proj.data == before,
              "清单里没有的图居然写进去了（半提交 = 地图换了、怪列表还是上一张的 ✗）：%r"
              % (proj.data,))

        # ⑥ 换图要通知外面（玩家的集合下拉按地图 id 存 ✓）。源码级 + 信号存在性
        got = []
        p.map_changed.connect(lambda mid: got.append(mid))
        p.map_changed.emit("105040303")
        check(got == ["105040303"], "`map_changed` 信号没接上/没收发：%r" % (got,))
        check("self.map_changed.emit(" in _rp,
              "改完地图没发 `map_changed` ⇒ 别的面板（玩家那两个集合下拉）还用着上一张图的集合 ✗")
        _mw = (ROOT / "gui" / "main_window.py").read_text(encoding="utf-8")
        check("route_panel.map_changed.connect(" in _mw
              and "set_zone_sets(self.route_panel.zone_sets())" in _mw,
              "主窗口没接 `map_changed` 重新推集合下拉（`_bind_cards` 里那一步的同一件事 ✓）")

        # ⑦ 弹窗：没选中不许确定 / 默认只列当前那张 / 筛选走 list_maps
        dlg = mp_mod.MapPickDialog("105040303")
        try:
            check(dlg.btn_ok.isEnabled() is False,
                  "还没选中就允许确定（点了会「什么都没换」✗）：%r" % dlg.chosen)
            check(dlg.lst.count() == 1,
                  "没输关键词时该只列**当前那张**（几百张图全列出来没法看 ✗）：%d"
                  % dlg.lst.count())
            it = dlg.lst.item(0)
            check(it is not None and it.data(Qt.UserRole) == "105040303"
                  and "当前" in it.text(),
                  "默认那张没标出「当前」/ 数据不是地图 id：%r" % (it.text() if it else None,))
            dlg.lst.setCurrentItem(it)
            check(dlg.btn_ok.isEnabled() is True, "选中之后「换成这张」还是灰的 ✗")
            dlg._on_ok()
            check(dlg.chosen == "105040303" and dlg.result() == dlg.Accepted,
                  "点确定没把选中的 id 交出来：%r" % (dlg.chosen,))
            dlg.ed_filter.setText("森林")
            check(dlg.lst.count() == 1, "按地图名筛选没命中：%d" % dlg.lst.count())
            dlg.ed_filter.setText("zzz不存在的图")
            check(dlg.lst.count() == 0 and "没有匹配" in dlg.lbl_count.text(),
                  "筛不到东西时没说清（人会以为界面卡了 ✗）：%r" % dlg.lbl_count.text())
        finally:
            dlg.close()
    finally:
        p.close()
        wzexport.list_maps = real_lm


def t_two_point_solve():
    """两点标定：两对「底图像素 ↔ 面板像素」→ 两轴缩放 + 偏移 + **自查**。

    用户 2026-09-27 定的三件：① 支持 x/y 各自一个缩放（`scale_y`）；
    ② 两轴不一致要看得出来；③ 「坐标系偏移」要顺手核对（只给建议、不写配置）。

    ⚠ 这条钉住的第一个坑：**两轴各自解是"恰好定解"**（每轴 2 个方程 2 个未知量）
      ⇒ 残差恒为 0，**不能**拿它当自查 ✗。真正的自查是"两轴一致性"（游戏等比缩放，
      `sx` 与 `sy` 本该几乎相等），残差只在**按等比再拟合一次**时才有意义。
    """
    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return
    t = mapdata.load(mid, with_canvas=True)
    cw, ch = canvas.shape[1], canvas.shape[0]

    # ① 老文件兼容：没有 `scale_y` ⇒ 两轴相同（这是新老两个方向都不炸的关键）
    check(mm.scales_of({"scale": 1.874}) == (1.874, 1.874),
          "老标定（只有 scale）读出来不是两轴相同：%s" % (mm.scales_of({"scale": 1.874}),))
    check(mm.scales_of({"scale": 1.874, "scale_y": 1.9}) == (1.874, 1.9),
          "两轴不同的标定读错了：%s" % (mm.scales_of({"scale": 1.874, "scale_y": 1.9}),))
    check(mm.scales_of({"scale": "写坏了", "scale_y": "也坏了"}) == (1.0, 1.0),
          "坏值没兜住：%s" % (mm.scales_of({"scale": "写坏了", "scale_y": "也坏了"}),))
    check("scale_y" not in mm.with_scales({"scale": 2.0, "scale_y": 2.5}, 2.0, 2.0),
          "两轴相同时不该留 `scale_y`（老文件里就没这个键）")
    check(mm.with_scales({}, 2.0, 2.5).get("scale_y") == 2.5,
          "两轴不同时没把 y 轴存进去：%s" % (mm.with_scales({}, 2.0, 2.5),))

    # ② 两轴各自的换算：造一块**非等比**面板，来回一趟必须对得上
    sx, sy, ox, oy = 1.874, 1.9, 6.0, 72.0
    cal_xy = {"mode": mm.MODE_FIT, "scale": sx, "scale_y": sy, "offset": [ox, oy]}
    for c in ((0.0, 0.0), (10.0, 20.0), (cw - 1.0, ch - 1.0)):
        p = (c[0] * sx + ox, c[1] * sy + oy)
        got = mm.panel_to_canvas(p[0], p[1], cal_xy)
        check(abs(got[0] - c[0]) < 1e-6 and abs(got[1] - c[1]) < 1e-6,
              "两轴各自的换算来回不一致：底图 %s → 面板 %s → 底图 %s" % (c, p, got))
    cal_1 = {"mode": mm.MODE_FIT, "scale": sx, "offset": [ox, oy]}      # 老文件（等比）
    one = mm.panel_to_canvas(c[0] * sx + ox, c[1] * sx + oy, cal_1)
    check(abs(one[1] - c[1]) < 1e-6,
          "老文件（只有 scale）的 y 轴没走同一个值：%s" % (one,))

    # ③ 两点解：造一组**已知**的两轴几何 → 必须精确复原
    cA, cB = (5.0, 4.0), (float(cw - 6), float(ch - 5))
    pA = (cA[0] * sx + ox, cA[1] * sy + oy)
    pB = (cB[0] * sx + ox, cB[1] * sy + oy)
    r = mm.solve_two_point(cA, pA, cB, pB)
    check(r["ok"], "两点解没成：%s" % (r.get("why"),))
    check(abs(r["scale"] - sx) < 1e-9 and abs(r["scale_y"] - sy) < 1e-9,
          "两轴缩放没复原：(%.4f, %.4f)，该 (%.4f, %.4f)"
          % (r["scale"], r["scale_y"], sx, sy))
    check(abs(r["offset"][0] - ox) < 1e-6 and abs(r["offset"][1] - oy) < 1e-6,
          "偏移没复原：%s，该 (%s, %s)" % (r["offset"], ox, oy))
    check(r["axis_gap_pct"] > 1.0,
          "两轴差 1.39%% 却没报出来（这就是用户要的那个提示）：%s" % r["axis_gap_pct"])

    # ④ 等比时：两轴一致、等比拟合残差≈0（反过来证明"残差"这条路没坏）
    r2 = mm.solve_two_point(cA, (cA[0] * 2.0 + 3, cA[1] * 2.0 + 4),
                            cB, (cB[0] * 2.0 + 3, cB[1] * 2.0 + 4))
    check(r2["axis_gap_pct"] < 1e-9 and r2["resid_px"] < 1e-6,
          "等比的两点却报了不一致/残差：gap=%.6f%% resid=%.6f px"
          % (r2["axis_gap_pct"], r2["resid_px"]))

    # ⑤ 退化必须**拦住**（给个假数比报错坏得多：人会拿着错的几何去寻路）
    degen = (
        (cA, pA, cA, pA, "同一个点"),
        (cA, pA, (cA[0], cA[1] + 30.0), (pA[0], pA[1] + 40.0), "同一列"),
        (cA, pA, (cA[0] + 30.0, cA[1]), (pA[0] + 40.0, pA[1]), "同一行"),
        (cA, pA, cB, (pA[0], pA[1]), "负"),          # 面板两点对调 ⇒ 缩放成负的
    )
    for d1, d2, d3, d4, tag in degen:
        rd = mm.solve_two_point(d1, d2, d3, d4)
        check(not rd["ok"] and rd["why"],
              "退化输入（%s）没被拦住：%s" % (tag, rd))
        check(tag in rd["why"] or "负" in rd["why"],
              "退化输入（%s）的理由没说清：%r" % (tag, rd.get("why")))

    # ⑥ 读点抖 ±1 像素：**两点取得越远越准**（这条钉住"为什么让你取最远的两点"）
    def _err(pt_c, pt_p):
        rr = mm.solve_two_point((10.0, 10.0), (10.0 * sx + ox, 10.0 * sy + oy),
                                pt_c, pt_p)
        return abs(rr["scale"] - sx)
    near_c = (30.0, 25.0)
    near = _err(near_c, (near_c[0] * sx + ox + 1.0, near_c[1] * sy + oy))
    far_c = (float(cw - 5), float(ch - 4))
    far = _err(far_c, (far_c[0] * sx + ox + 1.0, far_c[1] * sy + oy))
    check(far < near,
          "读点抖 1px 时，两点取远反而更不准（近 %.5f / 远 %.5f）—— 和「越远越准」相反 ✗"
          % (near, far))

    # ⑦ 折算（叠图画到实时画面）两轴都要折：y 轴不同 ⇒ 高度跟着变、宽度不动
    crop = (6, 72, cw, ch + 8)
    panel_wh = (int(round(cw * sx)), int(round(ch * sy)))
    _s1, d_1 = mm.frame_overlay_rects(cal_1, crop, (cw, ch), calib_panel=panel_wh)
    _s2, d_2 = mm.frame_overlay_rects(cal_xy, crop, (cw, ch), calib_panel=panel_wh)
    check(d_2[2] == d_1[2] and d_2[3] > d_1[3],
          "折算没按两轴走：等比 %s / 非等比 %s（x 该一样、y 该更大）" % (d_1, d_2))

    # ⑧ 写回标定：两种显示方式都要能"解出来 → 存进去 → 换算回来"对得上
    #    （crop 的 `offset` 是那套约定里的 inset，`view` 才装截距 —— 写错就是整体平移
    #     几十世界像素，而看着"差不多对" ✗）
    base_fit = {"mode": mm.MODE_FIT, "alpha": 35, "ref": "canvas", "score": 0.73,
                "world_offset": [7, 28]}
    for mode in (mm.MODE_FIT, mm.MODE_CROP):
        base = dict(base_fit, mode=mode)
        saved = mm.calib_from_two_point(base, r, inset=4)
        check(saved["src"] == "two_point" and "score" not in saved,
              "写回的标定没标成两点来的/还留着旧的匹配分：%s" % saved)
        check(abs(saved["scale"] - sx) < 1e-9
              and abs(mm.scales_of(saved)[1] - sy) < 1e-9,
              "两轴缩放没写进去：%s" % saved)
        check(saved.get("alpha") == 35 and saved.get("ref") == "canvas",
              "写回时把别的东西弄丢了：%s" % saved)
        # ⚠ **遗留的「坐标系偏移」必须被清掉**（2026-09-27 修）：锚点 2026-09-27 起是
        #   黄点**下沿**，这一项的正常值就是 (0, 0)；老的那份（重心口径，(7,28) 这种）
        #   被原样带过去 = 读数凭空偏 28 个世界像素（≈3 个面板像素，看着像标定没量准 ✗）。
        check("world_offset" not in saved,
              "重新量几何时把老的「坐标系偏移」带过去了（凭空偏几十世界像素 ✗）：%s" % saved)
        # 解出来的几何 → 存进这套写法 → 用 panel_to_canvas 换算回来，必须还是那 4 个点
        for c, p in ((cA, pA), (cB, pB)):
            got = mm.panel_to_canvas(p[0], p[1], saved)
            # crop 下 `view` 取整了 ⇒ 容差 1 底图像素（fit 下应当精确）
            tol = 1e-6 if mode == mm.MODE_FIT else 1.0
            check(abs(got[0] - c[0]) <= tol and abs(got[1] - c[1]) <= tol,
                  "%s 写回后换算对不上：底图 %s → 面板 %s → 底图 %s（容差 %s）"
                  % (mode, c, p, got, tol))

    # ⑨ 「坐标系偏移」核对：**黄点采样**（面板坐标 + 那块像素站着的世界坐标）→ 建议值
    #
    # ⚠ 这里必须分清两类采样（写第一版时混过，实算才发现，坑记在这儿）：
    #   · **黄点采样**：面板坐标是**黄点**（玩家标记）的位置。它是"玩家原点 + 偏移"画出来
    #     的 ⇒ `偏移 = 世界坐标 - panel_to_world(黄点)` ✓ 这才是世界偏移该问的问题。
    #   · **地标采样**：平台角那种。它的底图坐标和世界坐标都由地图数据定死 ⇒ 拿它算偏移
    #     **恒等于 0**，是个死结论 ✗（只能查"你点歪了没"，那是另一件事）。
    w = (650.0, 283.0)                       # 用户现场报过的那个点
    off_true = (7.0, 28.0)                   # 现存标定里的 world_offset
    cc_dot = t.world_to_canvas(w[0] - off_true[0], w[1] - off_true[1])   # 黄点画在哪
    pp_dot = (cc_dot[0] * sx + ox, cc_dot[1] * sy + oy)
    adv = mm.world_offset_advice([(pp_dot, w)], cal_xy, off_true, t)
    check(adv["ok"], "偏移核对没成：%s" % (adv.get("why"),))
    check(adv["suggest"] == (7, 28),
          "按黄点采样反推的偏移不对：%s，该 (7, 28)" % (adv["suggest"],))
    check(adv["delta"] == (0, 0) and adv["spread"] < 1e-6,
          "和现存值一致时却报了差额/不一致：delta=%s spread=%.6f"
          % (adv["delta"], adv["spread"]))
    adv2 = mm.world_offset_advice([(pp_dot, w)], cal_xy, (17, 5), t)
    check(adv2["delta"] == (-10, 23),
          "和现存值不一致时没算出该改多少：delta=%s，该 (-10, 23)" % (adv2["delta"],))
    # 两次采样互相矛盾 ⇒ spread 要摆出来（**不设**阈值，数字说话）
    adv3 = mm.world_offset_advice([(pp_dot, w), (pp_dot, (w[0] + 80, w[1]))],
                                  cal_xy, off_true, t)
    check(adv3["ok"] and abs(adv3["spread"] - 80.0) < 1e-6,
          "两次采样差 80 世界像素却没报 spread：%s" % (adv3.get("spread"),))
    # 「地标算不出偏移」这件事本身也钉一条：拿平台角（底图坐标由地图数据定死）
    # 去算，结果必然 ≈ 0 —— 以后谁想"顺手用平台角量偏移"会在这儿看到为什么不行 ✓
    cc_lm = t.world_to_canvas(w[0], w[1])                # 地标在底图上的真位置
    pp_lm = (cc_lm[0] * sx + ox, cc_lm[1] * sy + oy)
    adv4 = mm.world_offset_advice([(pp_lm, w)], cal_xy, off_true, t)
    check(adv4["ok"] and adv4["suggest"] == (0, 0) and adv4["delta"] == (-7, -28),
          "用平台角算偏移本该恒得 (0,0)（它不含偏移信息）：%s" % (adv4["suggest"],))
    # 什么都没填 ⇒ 老实说"没可核对的"，不许编
    adv5 = mm.world_offset_advice([(pp_dot, None)], cal_xy, off_true, t)
    check(not adv5["ok"] and adv5["suggest"] is None,
          "没给世界坐标却编了个建议值：%s" % (adv5,))


def t_two_point_dialog():
    """两点标定弹窗：两次采样 → 算一算 → 保存（真控件、真信号；配置/标定全 mock）。

    **布局与交互是用户 2026-09-27 定的**（他画了草图）：一个下拉（离线点坐标类型）+
    A/B 两行四格 + 每行右边一个「实时填入当前黄点脚底」。草图把口径也定死了：
      · 「离线」= 那个位置**在地图上的**坐标（默认世界坐标 —— 寻路编辑器悬停那行就读得到）；
      · 「实时」= 黄点的**下沿（脚底）**，按钮自动认，不用手描。
    这条钉的就是这些（核心算法本身由 `t_two_point_solve` 钉）：
      ① 界面对不对（两行四格 + 两个按钮 + 下拉默认世界坐标）；
      ② **脚底口径**（`mm.dot_feet` = 外接矩形下沿，不是重心）—— 口径漂了整份标定会抬高
         半个点，叠图和读数一起偏，最难查；且**采样与读数必须是同一个口径**（见
         `t_two_point_readout_roundtrip`）；
      ③ 认黄点填进**那一行**；认不出 ⇒ **不瞎填** + 说清原因；
      ④ 填一组已知几何 ⇒ 两轴缩放/偏移报出来、「保存标定」写进当前来源且不丢字段；
      ⑤ 退化输入（两个点同一列）⇒ 当场说清、**不写文件**（写下去就是一份假几何 ✗）；
      ⑥ 离线按「世界坐标」填 ⇒ 换算后解出来的几何与按像素一致，且结果里写明换算过。
    ⚠ 实时画面用假 client（不连网络）；底图/地形用真文件 ✓；配置与标定全 mock ✓。
    """
    import unittest.mock as mock

    from PyQt5.QtCore import QPoint
    from PyQt5.QtWidgets import QApplication, QLabel

    from gui.two_point_calib import TwoPointCalibDialog

    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return
    cw, ch = canvas.shape[1], canvas.shape[0]
    t = mapdata.load(mid, with_canvas=True)     # 「世界 ↔ 底图」换算要用它

    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")

    class _Cli:
        """假实时小地图：一帧固定画面。"""

        def __init__(self, frame):
            self.frame = frame
            self.hint = ""

        def latest(self):
            return self.frame, 0.0

        def stop(self):
            pass

    box = {}
    cal_in = {"mode": mm.MODE_FIT, "scale": 1.0, "offset": [0, 0], "view": [0, 0],
              "alpha": 22, "score": 0.4, "ref": "canvas", "world_offset": [7, 28]}

    def _save(_m, c, src=None):
        box["cal"] = dict(c)
        box["src"] = src

    # 认黄点那一拍给的面板：暗底 + 一个实测色的黄点（和 `t_dot_yellow_found` 同一套色）
    dot_panel = _dot_panel(cw, ch + 8)
    dot_panel[88:94, 58:64] = (65, 243, 245)        # 6×6 ⇒ 重心约 (61, 91)、下沿 93

    with mock.patch.object(mapdata, "load_calib", lambda _m, src=None: dict(cal_in)), \
         mock.patch.object(mapdata, "save_calib", _save):
        cli = _Cli(np.zeros((ch + 8, cw, 3), np.uint8))
        dlg = TwoPointCalibDialog(mid, src=mm.SRC_STREAM, client=cli,
                                  mode=mm.MODE_FIT)
        try:
            # ① 界面对了：两行四格 + 两行各一个按钮 + 下拉默认世界坐标
            check(all(k in dlg._sp for k in (("A", "off"), ("A", "live"),
                                            ("B", "off"), ("B", "live"))),
                  "坐标格子没摆全：%s" % (sorted(dlg._sp),))
            check(len(dlg._btn_dot) == 2 and dlg._btn_dot[0].text().startswith("实时填入"),
                  "每行右边那个「实时填入当前黄点脚底」没摆上：%s"
                  % [b.text() for b in dlg._btn_dot])
            check(dlg._off_is_world() and dlg.cmb_unit.count() == 2,
                  "离线点坐标类型默认该是「世界坐标」（编辑器悬停读得到的那套）")
            check(dlg._frame is not None and "实时画面" in dlg.lbl_live.text(),
                  "实时画面那行没报状态：%r" % dlg.lbl_live.text())

            # ② 脚底口径：外接矩形**下沿**，不是重心；且**口径只有一处**
            #    （`perception.minimap.dot_feet`）—— 采样与读数各写一份迟早分叉，
            #    2026-09-27 用户报的"填脚底、读重心"就是这么来的 ✗
            #    ⚠ 下沿是**亚像素**的"下边缘"（h=6 的点占 88..93 行 ⇒ 重心 91 + 2.5 =
            #      93.5）—— 取整那条口径 2026-09-27 已删（它让读数一格一格跳 8.55 世界
            #      单位，用户报"同一个位置差 9"就是这个）。采样框是整数，调用方取整 ✓。
            check(mm.dot_feet({"x": 61.0, "y": 91.0, "h": 6}) == (61.0, 93.5),
                  "黄点脚底算错了：%s（该 (61.0, 93.5)）"
                  % (mm.dot_feet({"x": 61.0, "y": 91.0, "h": 6}),))
            _src2p = (ROOT / "gui" / "two_point_calib.py").read_text(encoding="utf-8")
            check("def dot_feet" not in _src2p and "mm.dot_feet" in _src2p,
                  "弹窗自己又写了一份 dot_feet —— 口径分叉（采样和读数各算一套）")

            # ③ 认黄点：填进**这一行**的实时两格（此时没"算一算"⇒ 用文件里那份标定）
            cli.frame = dot_panel
            dlg._tick()
            dlg._on_pick_dot(0)
            got = [b.value() for b in dlg._samp_row(0)]
            check(abs(got[0] - 61) <= 2 and abs(got[1] - 93) <= 2,
                  "认黄点没把脚底填进那一行：%s（该约 (61, 93)）" % (got,))
            check(dlg._dot is not None and "下沿" in dlg.lbl_say.text(),
                  "认到了却没把「下沿（脚底）」报出来：%r" % dlg.lbl_say.text())
            check([b.value() for b in dlg._samp_row(1)] == [0, 0],
                  "认黄点填到另一行去了（两行各认一次才对）：%s"
                  % ([b.value() for b in dlg._samp_row(1)],))

            # ④ 认不出 ⇒ 不瞎填 + 说清原因
            dlg._samp_row(0)[0].setValue(0)
            dlg._samp_row(0)[1].setValue(0)
            cli.frame = _dot_panel(cw, ch + 8)      # 一块没有黄点的面板
            dlg._tick()
            dlg._on_pick_dot(0)
            check([b.value() for b in dlg._samp_row(0)] == [0, 0],
                  "黄点认不出来却把格子填了（编了个位置）：%s"
                  % ([b.value() for b in dlg._samp_row(0)],))
            check("没认出来" in dlg.lbl_say.text(),
                  "认不出时没说原因：%r" % dlg.lbl_say.text())

            # ⑤ 填一组**已知**几何（按像素）→ 算 → 存
            #   顺带钉住：非模态（界面上一模一样，但模态会锁住工作台）+
            #   保存后 `saved_now` 要叫一声（非模态没有 exec_() 的返回值可看）
            check(not dlg.isModal(), "双点标定窗是模态的 —— 开着它就没法操作工作台 ✗")
            fired = []
            dlg.saved_now.connect(lambda: fired.append(1))
            dlg.cmb_unit.setCurrentIndex(1)         # 1 = 像素
            check(not dlg._off_is_world(), "口径下拉没切到像素")
            sx, sy, ox, oy = 1.874, 1.9, 6.0, 72.0
            for tag, c in (("A", (5.0, 4.0)), ("B", (cw - 6.0, ch - 5.0))):
                dlg._sp[(tag, "off")][0].setValue(int(c[0]))
                dlg._sp[(tag, "off")][1].setValue(int(c[1]))
                dlg._sp[(tag, "live")][0].setValue(int(round(c[0] * sx + ox)))
                dlg._sp[(tag, "live")][1].setValue(int(round(c[1] * sy + oy)))
            dlg._on_calc()
            txt = dlg.lbl_say.text()
            check("缩放 x" in txt and "偏移" in txt and "两轴差" in txt,
                  "结果行没把两轴缩放/偏移报出来：%r" % txt)
            # 采样几何也要摆出来：只有"两轴差"这个百分比的话，分不清"两点没对准"
            # 还是"两点太近"（用户 2026-09-27 那份 y 方向只隔 27 个实时像素 ⇒
            # 1 像素读数差 = 3.7% 的 sy 误差 —— 那 8.43% 基本是噪声）
            check("两点跨度" in txt and "每错 1 像素" in txt and "残差" in txt,
                  "结果行没把「跨度 / 每像素灵敏度 / 等比残差」摆出来：%r" % txt)
            check(dlg._result and abs(dlg._result["scale_y"] - sx) < 0.05,
                  "解出来的 y 轴缩放偏太多：%s" % (dlg._result or {}).get("scale_y"))
            dlg._on_save()
            check(dlg.saved and box.get("src") == mm.SRC_STREAM,
                  "没写进当前来源：%s" % (box.get("src"),))
            check(fired == [1],
                  "保存后没发 `saved_now` —— 非模态下外面就不知道「存过了」，"
                  "状态行与叠图不会跟着刷新：%s" % (fired,))
            cal = box.get("cal") or {}
            check(abs(cal.get("scale", 0) - sx) < 0.05
                  and abs(cal.get("scale_y", 0) - sy) < 0.05,
                  "两轴缩放没写进标定：%s" % cal)
            check(cal.get("alpha") == 22 and "world_offset" not in cal
                  and cal.get("src") == "two_point" and "score" not in cal,
                  "保存时把别的字段弄丢 / 留着旧的匹配分 / 没清掉老的「坐标系偏移」：%s" % cal)
            # ⚠ 解出来的 offset **不许取整**（2026-09-27 改）：这张图上 1 个面板像素
            #   = 7.9~8.6 世界像素 ⇒ 取整就是系统性偏最多半个像素 = 4 个世界像素，
            #   **看着像点没点准**（用户那个 (48,-222) 里就有这 4 个像素 ✗）
            check(all(abs(float(cal["offset"][i]) - float(dlg._result["offset"][i])) < 1e-3
                      for i in (0, 1)),
                  "存下来的 offset 又被取整了：存的 %s、解出来的是 %s"
                  % (cal.get("offset"), dlg._result["offset"]))

            # ⑤b 改一格 ⇒ 上次的结果**作废**（这扇窗关掉不销毁 ⇒ 结果也会跟着留 ✗，
            #    不设这道闸，"改了数忘了重算、直接点保存"就会把上一版几何写进标定）
            dlg._sp[("A", "off")][0].setValue(dlg._sp[("A", "off")][0].value() + 7)
            check(dlg._result is None,
                  "改了坐标却还留着上次的解 ⇒ 会存下旧几何 ✗")
            box.clear()
            dlg._on_save()
            check(not box, "结果作废之后点保存还是写文件了：%s" % box)

            # ⑥ 退化：两个点同一列 ⇒ 当场说清，而且**不许**写文件
            box.clear()
            dlg._sp[("B", "off")][0].setValue(dlg._sp[("A", "off")][0].value())
            dlg._on_calc()
            check("同一列" in dlg.lbl_say.text(),
                  "两个点同一列时没说清原因：%r" % dlg.lbl_say.text())
            check(dlg._result is None, "退化输入还留着一份结果，后面会存下去 ✗")
            dlg._on_save()
            check(not box, "退化输入也把标定写下去了（写的是假几何）：%s" % box)

            # ⑦ 离线那侧按「世界坐标」填：换算后解出来的几何必须和按像素那次一致
            dlg.cmb_unit.setCurrentIndex(0)         # 0 = 世界坐标
            for tag, c in (("A", (5.0, 4.0)), ("B", (cw - 6.0, ch - 5.0))):
                wx, wy = t.canvas_to_world(c[0], c[1])   # ⚠ 底图→世界；别写成反过来那步
                dlg._sp[(tag, "off")][0].setValue(int(round(wx)))
                dlg._sp[(tag, "off")][1].setValue(int(round(wy)))
                dlg._sp[(tag, "live")][0].setValue(int(round(c[0] * sx + ox)))
                dlg._sp[(tag, "live")][1].setValue(int(round(c[1] * sy + oy)))
            dlg._on_calc()
            r2 = dlg._result or {}
            check(r2.get("ok") and abs(r2.get("scale", 0) - sx) < 0.05,
                  "按世界坐标填出来的几何和按像素填的不一样：%s" % (r2,))
            check("按世界坐标换算" in dlg.lbl_say.text(),
                  "按世界坐标填时没说换算过（对不上时就没法查）：%r" % dlg.lbl_say.text())

            # ⑧ 版式：每个标签都得有宽度、且两行按钮就在各自那一行上
            dlg.show()
            app.processEvents()
            for lb in dlg.findChildren(QLabel):
                check(lb.width() > 0 or not lb.text(),
                      "标签「%s」被压成 0 宽（一行里塞太多了）" % lb.text()[:20])
            for i in (0, 1):
                dy = (dlg._btn_dot[i].mapTo(dlg, QPoint(0, 0)).y()
                      - dlg._sp[("AB"[i], "live")][0].mapTo(dlg, QPoint(0, 0)).y())
                check(abs(dy) <= 6,
                      "第 %d 行的按钮没和那一行对齐（差 %d px）" % (i + 1, dy))
        finally:
            dlg._shutdown()
            dlg.close()


def t_two_point_readout_roundtrip():
    """**双点采样 → 读数**必须闭环：站回采样过的那个点，读数就是那个世界坐标。

    用户 2026-09-27 报（现场真数据：105090600、A=L3 上端 (1245,8)、B=L1 上端 (56,-205)，
    标定 `scale=1.880076 / scale_y=2.038575 / offset=[0,-11]`）：
        站在 L1 上端 ⇒ 该显示 (56,-205)，实际显示 (48,-222)。
    根因：**采样与读数取的不是黄点的同一个位置** —— 弹窗「实时填入当前黄点脚底」填的是
    黄点**下沿**（`dot_feet` = 重心 + (h-1)/2），而 `PlayerLocator.update` 拿**重心**换算。
    差半个点高；这张图 1 个面板像素 = 7.9 世界像素 ⇒ 6 像素高的点差 12 世界像素
    （探针实测：h=5~6 报 (52,-217)、h=7~8 报 (52,-225)，用户看到的 (48,-222) 正落在其中）。

    钉三件：
      ① 走弹窗那条真路：离线填**世界坐标**、实时按按钮认黄点脚底 → 「算一算」→「保存」；
      ② 站回 B（同一帧、同一个点）跑 `PlayerLocator.update` ⇒ 世界坐标必须 ≈ B 的世界坐标；
      ③ **反证**：故意按重心换算（`r["px"], r["py"]`）必须差出 ≥ 1 个底图像素 ——
         这条用例得能抓住"口径又漂回重心"，不然它只是把代码复述一遍。
    ⚠ 实时画面用假 client（不连网络）；配置/标定全 mock；底图/地形用真文件 ✓。
    """
    import unittest.mock as mock

    from PyQt5.QtWidgets import QApplication

    from gui.two_point_calib import TwoPointCalibDialog

    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return
    t = mapdata.load(mid, with_canvas=True)
    if t is None or not t.segments:
        print("      （没有带段的地形数据，跳过）")
        return
    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")

    # 两个**离得远**的采样点（非墙 foothold 的中点：一定落在平台上）。
    # 口径上"对角最稳"（跨度越小越不准，见弹窗结果行那几句）⇒ 挑底图上最远的、且
    # 两轴都拉开的一对；退化的（同行/同列）会被 `solve_two_point` 拒，所以先筛掉。
    pts = []
    for s in t.segments:
        for f in s.footholds:
            if f.is_wall:
                continue
            x = (f.left + f.right) / 2.0
            pts.append(t.world_to_canvas(x, f.y_at(x)))
            break
    pairs = [(a, b) for i, a in enumerate(pts) for b in pts[i + 1:]
             if abs(a[0] - b[0]) > 2 and abs(a[1] - b[1]) > 2]
    if not pairs:
        print("      （这张图上挑不出两轴都拉开的两个平台点，跳过）")
        return
    cA, cB = max(pairs, key=lambda ab: (ab[0][0] - ab[1][0]) ** 2
                 + (ab[0][1] - ab[1][1]) ** 2)
    wA = t.canvas_to_world(*cA)
    wB = t.canvas_to_world(*cB)

    class _Cli:
        def __init__(self):
            self.frame = None
            self.hint = ""

        def latest(self):
            return self.frame, 0.0

        def stop(self):
            pass

    box = {}
    cal_in = {"mode": mm.MODE_FIT, "scale": 1.0, "offset": [0, 0], "view": [0, 0],
              "score": 1.0, "alpha": 55, "world_offset": [0, 0]}

    def _save(_m, c, src=None):
        box["cal"] = dict(c)

    cli = _Cli()
    panels = {}
    for tag, c in (("A", cA), ("B", cB)):
        # ⚠ `_panel_with_dot_at` 把黄点的**下沿**摆在这个底图像素上（和口径一致）
        panels[tag] = _panel_with_dot_at(canvas, c[0], c[1])[0]

    with mock.patch.object(mapdata, "load_calib", lambda _m, src=None: dict(cal_in)), \
         mock.patch.object(mapdata, "save_calib", _save):
        dlg = TwoPointCalibDialog(mid, src=mm.SRC_STREAM, client=cli,
                                  mode=mm.MODE_FIT)
        try:
            check(dlg._off_is_world(),
                  "离线口径默认该是世界坐标（编辑器悬停读得到的那套）")
            for i, (tag, w) in enumerate((("A", wA), ("B", wB))):
                sp = dlg._sp[(tag, "off")]
                sp[0].setValue(int(round(w[0])))
                sp[1].setValue(int(round(w[1])))
                cli.frame = panels[tag]
                dlg._tick()
                dlg._on_pick_dot(i)              # ← 认黄点：填的是**下沿（脚底）**
                check([b.value() for b in dlg._samp_row(i)] != [0, 0],
                      "第 %s 行没认到黄点脚底：%s" % (tag, dlg.lbl_say.text()))
            dlg._on_calc()
            check(dlg._result and dlg._result["ok"],
                  "两点解不出来：%s" % (dlg._result or {}).get("why"))
            dlg._on_save()
            saved = box.get("cal") or {}
            check(bool(saved), "「保存标定」没写文件")
        finally:
            dlg._shutdown()
            dlg.close()

    # ② 站回 B：同一帧、同一个点 ⇒ 读数必须回到 B 的世界坐标
    loc = mm.PlayerLocator(mid)
    r = loc.update(panels["B"], src=mm.SRC_STREAM, calib=saved, terrain=t)
    check(r["ok"], "站回 B 却定位失败：%s（黄点层：%s）" % (r["note"], r["dot"]))
    tol = t.px_per_world + 3          # 一个底图像素（黄点本身有 ±1 像素）+ 余量
    check(abs(r["world_x"] - wB[0]) <= tol and abs(r["world_y"] - wB[1]) <= tol,
          "采样过的点读数却对不上：(%.1f, %.1f)，期望 (%.1f, %.1f)（容差 %.1f）"
          % (r["world_x"], r["world_y"], wB[0], wB[1], tol))

    # ③ 反证：按**重心**换算就得差出 ≥ 1 个底图像素（谁把口径改回重心，这条当场红）
    ox2, oy2 = mm.offset_of(saved)
    cx2, cy2 = mm.panel_to_world(r["px"], r["py"], saved, t)
    d_canvas = ((cx2 + ox2 - r["world_x"]) ** 2
                + (cy2 + oy2 - r["world_y"]) ** 2) ** 0.5
    check(d_canvas >= 1.0,
          "重心与下沿只差 %.2f 底图像素 —— 这条用例抓不住口径漂移"
          "（它那步「反证」就失效了）" % d_canvas)


def t_world_reads_new_calib():
    """保存新标定后，那行世界读数必须**立刻**跟着变（用户 2026-09-27 报的现象）。

    现场原话："点保存标定后，实时画面小地图下面显示的世界坐标应该有变化，但是现在就算
    乱填也没变化"。

    链上唯一能拖住它的是 `PlayerLocator.calib_for` 的 **1 秒缓存**（那是给 30 拍/秒的
    实时回路省读盘用的）；而 `load()` 只在**换图**时清缓存、同图刷新直接 return ⇒
    保存后最多 1 秒才生效，看着就像"没生效/没变化"。所以 `_refresh_mmap` 里显式
    `forget_calib()`，保存完那条路（`saved_now` → `_refresh_mmap`）立刻生效 ✓。

    这条走**保存完的那条真路**：`_refresh_world()` 出一版读数 → 改标定（乱填）→
    `_refresh_mmap()` + `_tick_world()` → 读数必须换。**把 `forget_calib()` 那行删掉，
    它当场红**（1 秒缓存还没过期）。
    """
    import re
    import unittest.mock as mock

    from PyQt5.QtWidgets import QApplication

    import gui.route_panel as rp
    from gui.route_panel import RoutePanel

    mid, canvas = pick_map()
    t = mapdata.load(mid, with_canvas=True) if mid else None
    if t is None or not t.segments:
        print("      （没有带段的地形数据，跳过）")
        return
    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")

    seg, f0 = next((s, f) for s in t.segments for f in s.footholds if not f.is_wall)
    wx = (f0.left + f0.right) / 2.0
    wy = f0.y_at(wx)
    ccx, ccy = t.world_to_canvas(wx, wy)
    panel, _px, _py = _panel_with_dot_at(canvas, ccx, ccy)

    class _FakeLive:
        def __init__(self, frame):
            self._f = frame
            self.ov = None

        def current_frame(self):
            return self._f

        def set_minimap_overlay(self, pix, src=None, frame_rect=None,
                                alpha=0.55, note=""):
            self.ov = None if pix is None else (pix, src, frame_rect, alpha, note)

        def set_overlay_note(self, note):
            if self.ov is None:
                return False
            self.ov = tuple(self.ov[:4]) + (note or "",)
            return True

    live = _FakeLive(panel)
    p = RoutePanel()
    p.live_panel = live
    p.project = _Proj(mid)
    # 假的"标定文件"：`load_calib` 每次给它一个副本 ⇒ 改它就是"标定变了"
    cal = {"mode": mm.MODE_FIT, "scale": 1.0, "offset": [0, 0], "view": [0, 0],
           "score": 1.0, "alpha": 55}
    cfg = {"mmap_src": mm.SRC_LIVE,
           "mmap_crop": [0, 0, panel.shape[1], panel.shape[0]],
           "mmap_draw": True}
    try:
        with mock.patch.object(rp, "load_live", lambda: dict(cfg)), \
             mock.patch.object(rp, "update_live", lambda **kw: cfg.update(kw)), \
             mock.patch.object(mapdata, "load_calib", lambda _m, src=None: dict(cal)):
            p.show()
            app.processEvents()
            p._refresh_world()
            t1 = p.lbl_mmap_world.text()
            check("玩家世界坐标" in t1 or "世界坐标" in t1, "第一版读数就没出来：%r" % t1)

            # 乱填一份新标定（就是用户说的"乱填"）
            cal["scale"] = 3.0
            cal["offset"] = [50, 20]
            # 保存后那条真路：信号 → `_refresh_mmap`（顺手忘掉标定缓存）→ 下一拍重算
            p._refresh_mmap()
            p._tick_world()
            t2 = p.lbl_mmap_world.text()
            check(t2 != t1,
                  "换了标定，世界读数一字没变 —— 十有八九是 `calib_for` 那 1 秒缓存"
                  "没被忘掉（`_refresh_mmap` 里少了 `forget_calib()`）：%r" % t2)
            # 数值也要真跟着换（不只是文案从"有数"变成"算不到"）
            n1 = re.search(r"\((-?\d+), (-?\d+)\)", t1)
            n2 = re.search(r"\((-?\d+), (-?\d+)\)", t2)
            check(n1 is None or n2 is None or n1.groups() != n2.groups(),
                  "读数文案变了、坐标却还是同一组：%r → %r" % (t1, t2))
    finally:
        p._world_timer.stop()
        p.close()


def t_two_point_keeps_state():
    """双点标定窗关掉再打开：**填过的数还在**，收帧那一路自己接回来。

    用户 2026-09-27 报："关掉双点标定弹窗后，上次的数据就丢了"。
    根因是我上一轮那句"不可见就重建"（为了换图/换来源）—— 它把"关过窗"也当成要重建 ✗。

    钉三件：
      ① 关窗（`close()`）后实例与格子里的数字都还在；
      ② 再 `show()` 时收帧那一路是活的（关窗时被 `_shutdown` 停过；不接回来
         「认黄点」会一直说"还没收到画面"，而界面上没有任何迹象 ✗）；
      ③ 入口那段**不许**再出现"看不见就重建"（`not dlg.isVisible()`）—— 那正是丢数据的原因。
    """
    import unittest.mock as mock

    from PyQt5.QtWidgets import QApplication

    import perception.minimap as mmmod
    from gui.two_point_calib import TwoPointCalibDialog

    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return
    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")

    class _Cli:
        """假 client：数它被调了几次 `latest`（用来判断"接回来了没有"）。"""

        def __init__(self):
            self.hint = ""
            self.calls = 0

        def latest(self):
            self.calls += 1
            return None, 0.0

        def stop(self):
            pass

    with mock.patch.object(mapdata, "load_calib", lambda _m, src=None: {}), \
         mock.patch.object(mapdata, "save_calib", lambda *a, **kw: None):
        cli = _Cli()
        dlg = TwoPointCalibDialog(mid, src=mm.SRC_STREAM, client=cli,
                                  mode=mm.MODE_FIT)
        try:
            dlg._sp[("A", "off")][0].setValue(123)
            dlg._sp[("A", "off")][1].setValue(-45)
            dlg._sp[("A", "live")][0].setValue(7)
            dlg.show()
            app.processEvents()
            dlg.close()                      # 关窗 = 隐藏（不销毁）
            app.processEvents()
            check(dlg._sp[("A", "off")][0].value() == 123
                  and dlg._sp[("A", "off")][1].value() == -45
                  and dlg._sp[("A", "live")][0].value() == 7,
                  "关掉窗之后填过的数丢了：%s"
                  % [[b.value() for b in dlg._sp[k]] for k in
                     (("A", "off"), ("A", "live"))])
            check(not dlg._timer.isActive(), "关窗没收掉收帧节拍（白占资源）")
            n0 = cli.calls
            dlg.show()                       # 再打开
            app.processEvents()
            check(dlg._timer.isActive(), "再打开没收帧节拍了（认黄点会一直说没画面）")
            check(cli.calls > n0, "再打开没去取帧（收帧那一路没接回来）")
            # 自己连的那条流：关窗停过 ⇒ 再打开要重建（用假类替换掉真客户端）
            calls = []

            class _FakeClient:
                def __init__(self, *a, **kw):
                    calls.append(1)

                def start(self):
                    calls.append("start")
                    return self

                def latest(self):
                    return None, 0.0

                def stop(self):
                    pass

            dlg._own_client = True
            dlg._client_off = True
            with mock.patch.object(mmmod, "MiniMapClient", _FakeClient):
                dlg.close()
                dlg.show()
                app.processEvents()
            check(calls == [1, "start"] and not dlg._client_off,
                  "自己那路收流没在重新打开时接回来：%s" % (calls,))
        finally:
            dlg._shutdown()
            dlg.close()

    # ③ 入口那段不许再"看不见就重建"
    src = (ROOT / "gui" / "route_panel.py").read_text(encoding="utf-8")
    i0 = src.find("def _open_two_point_calib")
    i1 = src.find("\n    def ", i0 + 10)
    body = src[i0:i1 if i1 > 0 else len(src)]
    check("not dlg.isVisible()" not in body,
          "入口又写回了「看不见就重建」—— 那会让关窗后的数据全丢（用户报过的）✗")


def t_two_point_dialog_nonmodal():
    """双点标定窗必须**非模态**（用户 2026-09-27 要求：开着它还能操作工作台主窗口）。

    为什么单列一条：模态与非模态**界面长得一模一样**，但模态会把整个工作台锁住 ——
    量这个的过程本来就要一边操作工作台（到「实时」页开预览、切「路线识别」看世界读数、
    对着叠图核），锁住就只能反复关窗开窗 ✗。所以三处都要钉：
      ① 窗口自己 `isModal()` 为假；
      ② 入口那一段是 `show()` 而**不是** `exec_()`（源码里就一行，改回去谁也不会注意）；
      ③ 非模态之后**必须**有的那根回传线：`saved_now` 信号 + 单实例挂主窗口
         （否则"存了但状态行/叠图没刷"、"开两个窗改同一份标定"）。
    """
    from PyQt5.QtWidgets import QApplication

    from gui.two_point_calib import TwoPointCalibDialog

    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")

    class _Cli:
        """假 client：不连网络（这条只问"模态不模态"）。"""

        def __init__(self):
            self.hint = ""

        def latest(self):
            return None, 0.0

        def stop(self):
            pass

    mid, _canvas = pick_map()
    if mid is not None:
        dlg = TwoPointCalibDialog(mid, src=mm.SRC_STREAM, client=_Cli(),
                                  mode=mm.MODE_FIT)
        try:
            check(not dlg.isModal(),
                  "双点标定窗是模态的 —— 开着它就没法操作工作台 ✗")
        finally:
            dlg._shutdown()
            dlg.close()

    src = (ROOT / "gui" / "route_panel.py").read_text(encoding="utf-8")
    i0 = src.find("def _open_two_point_calib")
    check(i0 > 0, "路线面板里找不到「双点标定」的入口")
    i1 = src.find("\n    def ", i0 + 10)
    body = src[i0:i1 if i1 > 0 else len(src)]
    # ⚠ 只找**调用**（`dlg.show()` / `dlg.exec_()`），别拿 `exec_()` 全文找：
    #   函数说明里就写着"非模态就没有 `exec_()` 的返回值了"——那是**解释这个坑**，
    #   把它一起判红就成了"解释也算错"（`t_overlay_on_live` 里踩过同一个形状 ✗）。
    check("dlg.show()" in body and "dlg.exec_()" not in body,
          "「双点标定」入口还用着 dlg.exec_() —— 模态会锁住工作台 ✗")
    check("saved_now" in body and "_two_point_dlg" in body,
          "非模态之后缺了回传线（saved_now 信号）或单实例挂载（_two_point_dlg）")


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

    # ③ 接线：叠图那一层在路线面板；**开关与浓淡在设置窗口**（2026-09-26 搬走 ✓）
    rp = (ROOT / "gui" / "route_panel.py").read_text(encoding="utf-8")
    for kw in ("_refresh_overlay", "set_minimap_overlay(", "frame_overlay_rects("):
        check(kw in rp, "路线识别面板里缺 %s（叠图这条没接上）" % kw)
    check("_overlay_blocker" in rp,
          "叠图没画出来时不说明原因 —— 勾了没反应，人只能猜")
    # ④ **浓淡 0 就是 0**（2026-09-26 用户报："拖动条在最左端时透明度是 55% 而不是 0%"）：
    #    根因是 `cal.get("alpha") or 55` —— **0 是 falsy** ⇒ 被默认值吃掉 ✗。
    #    所以取值只走 `mm.overlay_alpha_pct()`，默认值也只有那一处 ✓。
    check(mm.overlay_alpha_pct({"alpha": 0}) == 0,
          "存着 0（完全看不见）却读成了 %r —— `or 55` 那个坑又回来了 ✗"
          % mm.overlay_alpha_pct({"alpha": 0}))
    check(mm.overlay_alpha({"alpha": 0}) == 0.0,
          "0%% 换成 Qt 透明度时不是 0.0：%r" % mm.overlay_alpha({"alpha": 0}))
    check(mm.overlay_alpha_pct({}) == mm.DEFAULT_ALPHA_PCT,
          "没写 alpha 时的默认浓淡不对：%r" % mm.overlay_alpha_pct({}))
    check(mm.overlay_alpha_pct({"alpha": 120}) == 100
          and mm.overlay_alpha_pct({"alpha": "写坏了"}) == mm.DEFAULT_ALPHA_PCT,
          "越界/坏值没被兜住：%r" % mm.overlay_alpha_pct({"alpha": 120}))
    for f in (ROOT / "gui" / "route_panel.py", ROOT / "gui" / "minimap_calib.py",
              ROOT / "gui" / "settings_dialog.py"):
        # ⚠ 只在**代码行**里找：注释里写「原来写的是 `... or 55`」是在解释这个坑 ✓，
        #   把它一起判红就成了"解释也算错"（写这条时就这么红过 ✗）。
        _bad = [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines()
                if "or 55" in ln and not ln.strip().startswith("#")]
        check(not _bad,
              "%s 里又出现了 `or 55` —— 0 会被它换成 55%% ✗：%r" % (f.name, _bad))
    check("apply_overlay_settings" in rp,
          "路线识别面板里没有 apply_overlay_settings —— 设置里改了开关没人重画 ✗")
    sd = (ROOT / "gui" / "settings_dialog.py").read_text(encoding="utf-8")
    for kw in ("ck_mmap_draw", "sld_alpha", "overlay_changed"):
        check(kw in sd, "设置弹窗里缺 %s（叠图开关/浓淡没搬过去）" % kw)
    check("overlay_changed" in (ROOT / "gui" / "main_window.py")
          .read_text(encoding="utf-8"),
          "主窗口没接 overlay_changed —— 弹窗里改了叠图，实时画面不会重画 ✗")
    # 「脚下是哪条 foothold」的 **x 容差**（用户 2026-09-26：(650,283) 站在平台边外 7px
    # 却报"脚下没有平台"）：容差从设置里来，而且**三处必须同口径** —— 少一处就会出现
    # "路线面板说在这块平台、起点判定说不知道"这种最难查的分歧 ✗。
    for f, label in ((ROOT / "gui" / "live_thread.py", "实时线程"),
                     (ROOT / "gui" / "route_panel.py", "路线面板"),
                     (ROOT / "perception" / "minimap.py", "定位器")):
        check("fh_xtol" in f.read_text(encoding="utf-8"),
              "%s 里没接「脚下 foothold 的 x 容差」（fh_xtol）" % label)
    for f, label in ((ROOT / "gui" / "live_thread.py", "实时线程"),
                     (ROOT / "gui" / "route_panel.py", "路线面板")):
        check("align_tol_px" in f.read_text(encoding="utf-8"),
              "%s 没把设置里的「坐标对齐误差范围」当 x 容差传下去" % label)
    # ⭐ `xtol` 的**用法**（2026-09-28 改过口径，别走回去 ✗）：用户报"(594,162) 应该属于
    #   「小平台」、却被解析成了「底层」" —— 病根正是老口径"**只在 dx=0 一条都找不到时**
    #   才放宽"✗：人站在「小平台」右边界**外 3px**、而 23px 之下的「底层」x 范围罩住了他
    #   ⇒ 老口径**判得出**（判错 ✗）⇒ 放宽那一遍**永远不跑** ✗。
    #   新口径 = **两遍都比、只有放宽后更近时才改判**（严格 `<`）✓。
    #   ⚠ 这里只钉**写法**（防"无条件放宽"✗）；行为由
    #     `t_foothold_below_prefers_nearer_within_xtol` 用真图钉 ✓。
    _md = (ROOT / "core" / "mapdata.py").read_text(encoding="utf-8")
    check("xtol" in _md and "d2 < bd" in _md,
          "foothold_below 的 `xtol` 不是「两遍都比、更近才改判」的写法"
          "（无条件放宽会误认隔壁平台 ✗；只在找不到时放宽又会漏掉「站边外 3px」那种真现场 ✗）")
    # 「框选小地图」只该存**位置**，不该顺手改另一个设置（来源）——
    # 2026-09-27 框选搬回路线面板了（按项目存 ✓）⇒ 两个文件都钉：
    # 谁把那句"顺手改来源"写回来都算 ✗
    for _f, _label in ((ROOT / "gui" / "route_panel.py", "路线面板"),
                       (ROOT / "gui" / "settings_dialog.py", "设置窗")):
        check("update_live(mmap_src=mm.SRC_LIVE, mmap_crop=" not in
              _f.read_text(encoding="utf-8"),
              "%s 里「框选小地图」又把来源改掉了 —— 它只该存位置" % _label)

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


def t_terrain_image_shows_zones():
    """「路线识别」的地形图 = **地形编辑器的结果**（颜色=集合、名字标在平台上）。

    对应 2026-09-26 的要求 2。钉两件事：
      ① `render(zones=…)` 画出来的图里真的出现了**集合色**（不是"每段一色"那版）；
      ② 面板挑图时优先挑 `<id>_zones.png`，没有才退回叠加图 ——
         不然"改了显示"只改了一半（编辑器结果画出来了，面板却还在看旧的）。
    """
    import tempfile
    import unittest.mock as mock

    mid, _canvas = pick_map()
    t = mapdata.load(mid, with_canvas=True) if mid else None
    if t is None or not t.segments:
        print("      （没有带段的地形数据，跳过）")
        return

    # ⚠ 本文件的约定：**要建控件的用例都得先保证 QApplication 存在** ——
    # 没有它就直接 QWidget 会被 Qt 判为致命错误、进程 abort（不是 Python 异常，
    # 所以看不到堆栈，只看到一个退出码）。
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")

    from core import zones as zones_mod
    from tools.map_terrain_view import hex_bgr, render

    z = zones_mod.Zones(mid)
    f0 = next(f for f in t.footholds if not f.is_wall)
    z.add_set("甲平台", [str(f0.fid)])
    col = hex_bgr(z.sets["甲平台"]["color"])

    tmp = Path(tempfile.mkdtemp(prefix="zrender_"))
    try:
        out = tmp / "z.png"
        render(t, out, zones=z)
        img = cv2.imread(str(out))
        check(img is not None, "集合图没画出来")
        # ① 集合色必须真的落在图上（那版"每段一色"用的是固定调色板，不会有这个色）
        want = np.array(col, dtype=np.int16)
        hit = int((np.abs(img.astype(np.int16) - want).sum(axis=2) <= 24).sum())
        check(hit >= 100,
              "图里找不到集合色 %s（命中 %d 像素）—— 画的还是段色那版？"
              % (col, hit))

        # ② 面板优先 `<id>_zones.png`，没有才退回 `<id>_overlay.png`
        import gui.route_panel as rp
        d = tmp / "map"
        d.mkdir()
        (d / ("%s_zones.png" % mid)).write_bytes(out.read_bytes())
        (d / ("%s_overlay.png" % mid)).write_bytes(out.read_bytes())
        p = rp.RoutePanel()
        p._map_id = lambda: mid
        try:
            with mock.patch.object(mapdata, "map_dir", lambda: d):
                path, title, _extra = p._map_image_path(mid)
            check(path is not None and path.name.endswith("_zones.png"),
                  "面板没有优先挑集合图：%s" % path)
            check("集合图" in title, "标题没写清是集合图：%r" % title)
            # 集合图不在（老项目没重画）→ 退回叠加图，并把"怎么拿到集合版"写在附注里
            (d / ("%s_zones.png" % mid)).unlink()
            with mock.patch.object(mapdata, "map_dir", lambda: d):
                path2, _t2, extra2 = p._map_image_path(mid)
            check(path2 is not None and path2.name.endswith("_overlay.png"),
                  "没有集合图时没退回叠加图：%s" % path2)
            check("生成地形图" in extra2,
                  "没告诉人怎么画出集合版：%r" % extra2)
        finally:
            p.close()
    finally:
        import shutil
        shutil.rmtree(str(tmp), ignore_errors=True)


def t_goto_hands_over_whole_path():
    """「命令前往」交出去的是**整条路径**（2026-09-26），碰到没做的执行器**整条拒发**。

    为什么要单开一条：原来那条真实数据用例挑到的图**没有爬边** ⇒ 它的"下半段"一直
    在跳过（"这张图没有带绳号的「爬」边，跳过下半段"）⇒ 多步下发等于**没人测** ✗。
    这里用**合成 zones**（临时目录里自己写边）造一条两段路径：
      甲 →（走）乙 →（走）丙
    ⇒ 断言两段各造一个任务、**一次**交给 `start_route`、顺序对；
    再造一条含「跳」的 ⇒ 断言**整条不下发**、且提示里点出是第几步。
    """
    import tempfile
    import time as _time
    import unittest.mock as mock

    from PyQt5.QtWidgets import QApplication
    from core import zones as zones_mod
    from decision import agent as agent_mod
    from decision import route as route_mod
    from decision.agent import settings as dsettings
    from gui.route_panel import RoutePanel

    app = QApplication.instance() or QApplication([])      # noqa: F841
    mid, _canvas = pick_map()
    t = mapdata.load(mid, with_canvas=True) if mid else None
    if t is None or not t.footholds:
        print("      （没有地形数据，跳过）")
        return
    walk = [f for f in t.footholds if not f.is_wall]
    if len(walk) < 3:
        print("      （非墙 foothold 少于 3 条，跳过）")
        return
    tmp = Path(tempfile.mkdtemp(prefix="gotopath_"))
    try:
        z = zones_mod.Zones(mid)
        for name, f in (("甲平台", walk[0]), ("乙平台", walk[1]),
                        ("丙平台", walk[2])):
            z.add_set(name, [str(f.fid)])
        z.add_edge("甲平台", "乙平台", "walk", why="用例")
        z.add_edge("乙平台", "丙平台", "walk", why="用例")
        # ⚠ **别在这份文件里再加"甲→丙 跳"**：`find_path` 会挑最短的那条（1 段 < 2 段）
        # ⇒ ① 就撞在跳上了（我第一版就是这么红的 ✗）。跳那条留给 ② 单独写一份文件。
        z.save(tmp / ("%s.zones.json" % mid))

        got = []

        class _Fake:
            def start_route(self, jobs, why=""):
                got.append((list(jobs), why))
                return True

            def start_climb(self, job):
                got.append(([job], ""))
                return True

        p = RoutePanel()
        p._map_id = lambda: mid
        try:
            with mock.patch.object(zones_mod, "ZONES_DIR", tmp), \
                 mock.patch.object(dsettings, "save", lambda *a, **k: None), \
                 mock.patch.object(agent_mod, "CURRENT", _Fake()):
                # ① 两段都做得了 ⇒ **一次**交出两个任务，顺序 = 路径顺序
                p._refresh_goto()
                p.cmb_goto.setCurrentIndex(p.cmb_goto.findData("丙平台"))
                p._fh_seen = (str(walk[0].fid), _time.monotonic())
                p._on_goto()
                txt = p.lbl_goto.text()
                check("已命令" in txt and "整条 2 步" in txt,
                      "没把整条路径交出去 / 没说清几步：%r" % txt)
                check(len(got) == 1 and len(got[0][0]) == 2,
                      "不是「一次交出整条」（该只调一次 start_route）：%r" % (got,))
                jobs = got[0][0]
                check(all(isinstance(j, route_mod.WalkJob) for j in jobs),
                      "造出来的不是走任务：%r" % (jobs,))
                check(jobs[0].dst_set == "乙平台" and jobs[1].dst_set == "丙平台",
                      "两段顺序不对：%s / %s" % (jobs[0].dst_set, jobs[1].dst_set))

                # ② **算不出路**（目标集合是孤岛：一条边都没有）⇒ 不下发，并说清边界。
                #    ⚠ 2026-09-27：这条原来用「传送门」冒充"还没做执行器的通行方式"——
                #    传送门 / 待确认已经从 `zones.EDGE_KINDS` **移除**了 ✓（四种通行方式
                #    **都有执行器**）⇒ "拒发"现在只剩这一种情形：**根本算不出路** ✓。
                got.clear()
                z3 = zones_mod.Zones(mid)
                for name, f in (("甲平台", walk[0]), ("丙平台", walk[2])):
                    z3.add_set(name, [str(f.fid)])
                z3.save(tmp / ("%s.zones.json" % mid))      # 故意**不加任何边**
                p._refresh_goto()
                p.cmb_goto.setCurrentIndex(p.cmb_goto.findData("丙平台"))
                p._fh_seen = (str(walk[0].fid), _time.monotonic())
                p._on_goto()
                txt2 = p.lbl_goto.text()
                check(not got, "算不出路却还是下发了命令：%r" % (got,))
                check("走不到" in txt2,
                      "没路时没说清边界（从起点能到哪些 / 缺哪条边）：%r" % txt2)

                # ③ 「跳」**会**下发（2026-09-27 初版执行器 ✓）：造出来的必须是 `JumpJob`
                got.clear()
                z4 = zones_mod.Zones(mid)
                for name, f in (("甲平台", walk[0]), ("丙平台", walk[2])):
                    z4.add_set(name, [str(f.fid)])
                z4.add_edge("甲平台", "丙平台", "jump", why="用例：跳")
                z4.save(tmp / ("%s.zones.json" % mid))
                p._refresh_goto()
                p.cmb_goto.setCurrentIndex(p.cmb_goto.findData("丙平台"))
                p._fh_seen = (str(walk[0].fid), _time.monotonic())
                p._on_goto()
                check(got and len(got[0][0]) == 1
                      and isinstance(got[0][0][0], route_mod.JumpJob),
                      "「跳」边没被下发成 JumpJob（执行器 2026-09-27 已接）：%r" % (got,))
                check("已命令" in p.lbl_goto.text(),
                      "「跳」边下发了命令却没说出来：%r" % p.lbl_goto.text())
            p.close()
        finally:
            pass
    finally:
        import shutil
        shutil.rmtree(str(tmp), ignore_errors=True)


def t_goto_commands_climb():
    """「命令前往」要**真的下发命令**（2026-09-27 起走 / 爬 / 下跳 / **跳** 四类都下发；
    只剩「传送门」如实说"还没做"）。

    2026-09-26 用户报的：点「命令前往」角色一动不动 —— 因为那时它**只算路、没接线**
    （执行器造好了，但全仓库没有一处调 `start_climb` ✗）。这条钉三件事：
      · 第一步是「爬」⇒ 从那条边造出 ClimbJob 并挂到**当前 agent**（`start_climb`）；
      · 第一步是别的通行方式 ⇒ 明说"这种的执行器还没做"（别让人以为点了没反应）；
      · 实时没在跑（没有当前 agent）⇒ 明说"先去「实时」页开始"，**不许**静悄悄什么都不做。
    """
    import tempfile
    import time as _time
    import unittest.mock as mock

    from PyQt5.QtWidgets import QApplication
    from core import zones as zones_mod
    from decision import agent as agent_mod
    from decision.agent import settings as dsettings
    from gui.route_panel import RoutePanel

    app = QApplication.instance() or QApplication([])
    mid, _canvas = pick_map()
    t = mapdata.load(mid, with_canvas=True) if mid else None
    if t is None or not t.footholds:
        print("      （没有地形数据，跳过）")
        return
    walk = [f for f in t.footholds if not f.is_wall]
    if len(walk) < 2:
        print("      （foothold 太少，跳过）")
        return

    tmp = Path(tempfile.mkdtemp(prefix="gotocmd_"))
    p = RoutePanel()
    p._map_id = lambda: mid
    try:
        # ---- ① 第一步是「走」⇒ 只算路，并**明说**这种执行器还没做 ----
        z = zones_mod.Zones(mid)
        z.add_set("甲平台", [str(walk[0].fid)])
        z.add_set("乙平台", [str(walk[1].fid)])
        z.add_edge("甲平台", "乙平台", "walk", why="用例")
        z.save(tmp / ("%s.zones.json" % mid))
        with mock.patch.object(zones_mod, "ZONES_DIR", tmp), \
             mock.patch.object(dsettings, "save", lambda *a, **k: None), \
             mock.patch.object(agent_mod, "CURRENT", None):
            p._refresh_goto()
            p.cmb_goto.setCurrentIndex(p.cmb_goto.findData("乙平台"))
            p._fh_seen = (str(walk[0].fid), _time.monotonic())
            p._on_goto()
            txt = p.lbl_goto.text()
            check("能走到" in txt, "该算得出路：%r" % txt)
            # 走（walk）的执行器 2026-09-26 做了 ⇒ 这一步**不再**是"这种执行器还没做"
            #（那句现在只该留给「跳」「传送门」）。这里没有实时在跑（CURRENT=None）
            # ⇒ 应该如实说"先去「实时」页开始"。
            check("执行器还没做" not in txt,
                  "「走」的执行器已经做了，却还在说没做：%r" % txt)
            check("实时" in txt and "只算给你看" in txt,
                  "没有实时在跑时没说清：%r" % txt)

        # ---- ② 真实数据里那条「爬」边：命令真的下发到当前 agent ----
        try:
            zr = zones_mod.load(mid)
        except Exception as e:                      # noqa: BLE001
            print("      （读不到真实集合文件，跳过下半段：%s）" % e)
            return
        cl = next((e for e in zr.edges
                   if e.get("kind") == "climb" and e.get("ladder")), None)
        if cl is None:
            print("      （这张图没有带绳号的「爬」边，跳过下半段）")
            return
        fids = (zr.sets.get(cl["from"]) or {}).get("footholds") or []
        if not fids:
            print("      （那条爬边的起点集合里没有 foothold，跳过下半段）")
            return
        got = []

        class _Fake:
            def start_climb(self, job):
                got.append(job)
                return job

        with mock.patch.object(dsettings, "save", lambda *a, **k: None), \
             mock.patch.object(agent_mod, "CURRENT", _Fake()):
            p._refresh_goto()
            j = p.cmb_goto.findData(cl["to"])
            check(j >= 0, "预览下拉里没有那条爬边的终点：%s" % cl["to"])
            p.cmb_goto.setCurrentIndex(j)
            p._fh_seen = (str(fids[0]), _time.monotonic())
            p._on_goto()
            txt = p.lbl_goto.text()
            check(got, "命令没下发（爬边第一步也该交给执行器）：%r" % txt)
            check("已命令" in txt, "下发了命令却没说出来：%r" % txt)
            job = got[0]
            check(job.dst_set == cl["to"] and job.ladder_id == cl.get("ladder"),
                  "下发的任务不对：dst=%s 绳=%s（该是 %s / %s）"
                  % (job.dst_set, job.ladder_id, cl["to"], cl.get("ladder")))
            lids = zones_mod.ladder_ids(t)
            lx = next(x.x for x in t.ladders if lids.get(id(x)) == cl.get("ladder"))
            # 任务的 x 就是**绳的地形 x**（地形数据本来就在世界坐标系里 ⇒ 直接比 ✓）。
            # ⚠ 不给它加「坐标系偏移」：那个偏移修的是"黄点重心 ↔ 玩家原点"，属于**玩家读数
            # 那一侧**（2026-09-26 用户澄清 —— 我在这里改错过一次）。
            check(abs(job.x - lx) < 1e-6, "任务的 x 不是那根绳的 x：%s vs %s"
                  % (job.x, lx))
        # ---- ③ 说了"爬"，但实时没在跑 ⇒ 命令发不出去，得说出来 ----
        with mock.patch.object(agent_mod, "CURRENT", None):
            p._on_goto()
            check("实时" in p.lbl_goto.text(),
                  "没有当前 agent 时该说清「先去实时页开始」：%r" % p.lbl_goto.text())
        p.close()
    finally:
        import shutil
        shutil.rmtree(str(tmp), ignore_errors=True)


def t_osd_task_and_timers():
    """画面那几行：**当前任务**（战斗 / 前往：集合）+ **计时任务**逐行列出（2026-09-26 要求 1、2）。

    钉四件事（错了都会让人看错）：
      · 没有寻路任务时「当前任务」必须显示**战斗**（用户指定的默认口径）；
      · 有任务时显示「前往：{集合名}」—— 名字走 agent 的**公开口径**
        （`current_goto_set()`），不是去摸私有的 `_climb`；
      · 计时任务**每项一行**、**休息排最前**，然后才是自定义定时行为；
      · 这些行**不铺底色**、字色取「设置 → 定时任务颜色」（用户明确要求）。
    外加要求 3：地形图那行「命令前往」右边有个「结束当前寻路」。
    """
    import time as _time
    import unittest.mock as mock

    from PyQt5.QtWidgets import QApplication
    from decision import agent as agent_mod
    from decision.agent import settings as ds
    from gui import theme
    from gui.route_panel import RoutePanel

    app = QApplication.instance() or QApplication([])
    p = RoutePanel()
    keys = ("rest_state", "rest_until_monotonic", "rest_pending",
            "next_afk_monotonic", "custom_timers", "custom_timer_next",
            "resetall_interval")
    saved = {k: getattr(ds, k) for k in keys}
    try:
        ds.rest_state = ""
        ds.rest_pending = False
        ds.next_afk_monotonic = 0.0
        ds.custom_timers = []
        ds.custom_timer_next = {}
        ds.resetall_interval = 0
        lines = p._osd_lines("世界 (1, 2)")
        check(lines[0] == "世界 (1, 2)",
              "第一行不是世界坐标（老行为被改了）：%r" % (lines[0],))
        check(any(not isinstance(l, str) and "当前任务" in l[0] and "战斗" in l[0]
                  for l in lines),
              "没有寻路任务时「当前任务」该显示「战斗」：%r" % (lines,))

        # ⭐ **决策状态行**（2026-09-29 用户要求：从画面左上角**搬进信息栏**、放世界坐标
        #   **下面**、与世界坐标**统一格式** ✓）：是**普通 str**（= 黑底白字、基础字号，
        #   与世界坐标同一份样式 ✓）；拿不到状态（_Fake 没有 state）⇒ 不画 ✓。
        class _FakeS:
            state = "attack"

        with mock.patch.object(agent_mod, "CURRENT", _FakeS()):
            d_lines = p._osd_lines("世界 (1, 2)")
        d_texts = [l if isinstance(l, str) else l[0] for l in d_lines]
        check(d_texts[0] == "世界 (1, 2)" and d_texts[1] == "决策：attack",
              "决策行该紧跟世界坐标、且同款式（普通 str = 黑底白字）✗：%r"
              % (d_texts[:3],))

        # ⭐ 当前区域配了 idle 回归 foothold ⇒「决策：idle → foothold#41」
        #   （2026-09-29 用户要求 ✓；目标 id = agent._idle_walk_beat 每拍写的 ✓）
        class _FakeI:
            state = "idle"
            _idle_fid = "41"

        with mock.patch.object(agent_mod, "CURRENT", _FakeI()):
            i_lines = p._osd_lines("世界 (1, 2)")
        i_texts = [l if isinstance(l, str) else l[0] for l in i_lines]
        check(i_texts[1] == "决策：idle → foothold#41",
              "idle 回归目标没显示（该是「决策：idle → foothold#41」✗）：%r"
              % (i_texts[:3],))
        class _FakeNoSt:                  # 没有 state 属性 = 拿不到状态 ✓（_Fake 在后面才定义 ✗）
            pass

        with mock.patch.object(agent_mod, "CURRENT", _FakeNoSt()):
            nd_texts = [l if isinstance(l, str) else l[0]
                        for l in p._osd_lines("世界 (1, 2)")]
        check(all(not t.startswith("决策：") for t in nd_texts),
              "拿不到状态却画了「决策：?」废行 ✗：%r" % (nd_texts[:3],))

        class _Fake:
            def current_goto_set(self):
                return "左上平台"

        with mock.patch.object(agent_mod, "CURRENT", _Fake()):
            ls = p._osd_lines("x")
            check(any(not isinstance(l, str) and "前往：左上平台" in l[0] for l in ls),
                  "有寻路任务时没写「前往：{集合名}」：%r" % (ls,))

        # 计时任务排布
        ds.rest_state = "afk_rest"
        ds.rest_until_monotonic = _time.monotonic() + 125
        ds.custom_timers = [{"name": "喂宠", "interval": [5, 10]},
                            {"name": "喊话", "interval": [3, 4],
                             "paused": True, "paused_left": 42.0}]
        ds.custom_timer_next = {"喂宠": _time.monotonic() + 66}
        ds.resetall_interval = 60
        rows = [l for l in p._osd_lines("x") if not isinstance(l, str)]
        # ⚠ 2026-09-28 起行元组可以是 **4 项**：`(文本, 颜色, 底色, 字号pt)` ✓（用户要求
        #   "当前任务以及下一行缩进的说明字体稍微大点" ✓ 见 `live_panel._draw_note`）
        #   ⇒ 只要求 **3 或 4 项**（老的 3 项仍然合法 ✓ 向后兼容 ✓）。
        check(all(len(r) in (3, 4) for r in rows),
              "任务行不是 (文本, 颜色, 底色[, 字号])：%r" % (rows,))
        # ⭐ **「当前任务」那行必须标大字号**（用户 2026-09-28 要求 ✓）：非 0 ⇒ `_draw_note`
        #   用大字 ✓。它的**缩进说明行**（`　` 开头）同样标了 —— 只是那句话要等有寻路任务时
        #   才出现（`_note` 非空才 append ✓），所以这里只保证"出现的都带字号" ✓。
        check(bool(rows) and len(rows[0]) == 4 and int(rows[0][3]) != 0,
              "「当前任务」那行没标大字号（用户 2026-09-28 点名要更大 ✗）：%r"
              % (rows[0] if rows else None,))
        _ind = [r for r in rows if str(r[0]).startswith("\u3000")]
        check(all(len(r) == 4 and int(r[3]) != 0 for r in _ind),
              "缩进的说明行（全角空格开头）没跟着「当前任务」一起变大（用户要求 ✗）：%r"
              % (_ind,))
        check(all(r[2] is False for r in rows),
              "任务行铺了底色（用户要求：这些文本不要背景色）：%r" % (rows,))
        col = theme.load_vis()["timer_color"]
        check(all(r[1] == col for r in rows),
              "任务行没用设置里的「定时任务颜色」（%s）：%r" % (col, rows))
        # 「定时任务」那一项的**显示开关**（用户 2026-09-27："辅助线与标记组里每项参数前
        # 加开关"）：关掉 ⇒ **这几行一个字都不画** ✓。
        # ⚠ 第一行（世界坐标等**读数**）不归这个开关管 —— 它照旧要在 ✓。
        _vis_off = dict(theme.load_vis())
        _vis_off["timer_on"] = False
        with mock.patch.object(theme, "load_vis", lambda: _vis_off):
            off_lines = p._osd_lines("世界 (1, 2)")
        check(off_lines == ["世界 (1, 2)"],
              "「定时任务」开关关掉了却还在画那几行：%r" % (off_lines,))

        # 「**任务队列**」：排在「当前任务」下方、**一行一个、按顺序**（用户 2026-09-27 要求 ✓）
        class _FakeQ(_Fake):
            def goto_queue(self):
                return ["左上平台", "乙平台"]

        with mock.patch.object(agent_mod, "CURRENT", _FakeQ()):
            q_lines = p._osd_lines("世界 (1, 2)")
        q_texts = [l if isinstance(l, str) else l[0] for l in q_lines]
        check(q_texts[0] == "世界 (1, 2)" and "当前任务" in q_texts[1],
              "队列行的位置不对（该在世界坐标 / 当前任务**之后**）：%r" % (q_texts[:3],))
        check(q_texts[2] == "队列 1：前往 左上平台" and q_texts[3] == "队列 2：前往 乙平台",
              "队列没有一行一个、按顺序显示：%r" % (q_texts,))
        check(all(not isinstance(l, str) and l[2] is False for l in q_lines[2:4]),
              "队列行不该铺底色（和任务那几行一致 ✓）：%r" % (q_lines[2:4],))
        # 没排队时**不多这几行**（画面干净 ✓）
        with mock.patch.object(agent_mod, "CURRENT", _Fake()):
            _noq = p._osd_lines("世界 (1, 2)")
        check(not any("队列" in (l if isinstance(l, str) else l[0]) for l in _noq),
              "没排队时也画了「队列」行：%r"
              % ([l if isinstance(l, str) else l[0] for l in _noq],))
        # ⭐⭐ **「最大战斗时长」倒计时**（用户 2026-09-29 ✓ 原话："将 foothold 集合的
        #   最大战斗时间倒计时显示在小地图下面的信息栏（当前任务及说明下面，自定义定时
        #   行为上面）"）—— 位置、内容、以及"没在计时就一个字都不画"三件 ✓。
        class _FakeF(_Fake):
            def fight_remain(self):
                return ("上层平台", 754.0, 1200.0)

        class _FakeN(_Fake):
            def fight_remain(self):
                return None                 # 没在计时（不在区域 / 那一项不限 ✓）

        # 先把 settings 上那份**镜像**写成另一个集合：显示**必须不理它** ✓
        ds.fight_zone_name, ds.fight_elapsed_s, ds.fight_cap_s = "别的集合", 0.0, 9999.0
        with mock.patch.object(agent_mod, "CURRENT", _FakeF()):
            f_lines = p._osd_lines("世界 (1, 2)")
        f_texts = [l if isinstance(l, str) else l[0] for l in f_lines]
        _fi = next((i for i, t in enumerate(f_texts) if t.startswith("战斗时长")), None)
        check(_fi is not None, "没显示「最大战斗时长」倒计时：%r" % (f_texts,))
        check(sum(1 for t in f_texts if t.startswith("战斗时长")) == 1,
              "「战斗时长」那行画了不止一次（两处各 append 一次 ✗）：%r" % (f_texts,))
        check("上层平台" in f_texts[_fi] and "12:34" in f_texts[_fi]
              and "20:00" in f_texts[_fi],
              "倒计时那句不对（该是：剩余 MM:SS / MM:SS + 集合名）：%r" % f_texts[_fi])
        check(_fi >= 1 and f_texts[_fi - 1].startswith("当前任务"),
              "倒计时没落在「当前任务」那行**下面**（用户点名的位置 ✗）：%r"
              % (f_texts[: _fi + 2],))
        _ri2 = next((i for i, t in enumerate(f_texts) if t.startswith("休息")), None)
        check(_ri2 is not None and _fi < _ri2,
              "倒计时没排在「休息 / 定时行为」那几行**上面**（用户点名的位置 ✗）：%r"
              % (f_texts,))
        # ⚠ **口径一处**：数值只从 `agent.fight_remain()` 来 ✓（上面那份环境里
        #   `ds.fight_zone_name` 写着"别的集合" ⇒ 一旦去读镜像，这行就会换成那个名字 ✗）
        check("别的集合" not in f_texts[_fi],
              "倒计时读了 settings 上那份镜像（两处各算一套 ✗）：%r" % f_texts[_fi])
        # 不在计时（`fight_remain()` 给 None）⇒ **一个字都不画**（绝不写 0:00 / 未排期 ✗）
        for _ag_obj, _tag in ((_FakeN(), "fight_remain()=None"),
                              (_Fake(), "agent 里压根没这个接口")):
            with mock.patch.object(agent_mod, "CURRENT", _ag_obj):
                _n_texts = [l if isinstance(l, str) else l[0] for l in p._osd_lines("x")]
            check(not any("战斗时长" in t for t in _n_texts),
                  "没在计时（%s）却画了「战斗时长」那行（会让人以为马上要换地方 ✗）：%r"
                  % (_tag, _n_texts))
            check(not any("0:00" in t for t in _n_texts),
                  "没在计时（%s）却写了个 0:00 ✗：%r" % (_tag, _n_texts))

        names = [r[0] for r in rows]
        rest_i = next((i for i, n in enumerate(names) if n.startswith("休息")), None)
        tim_i = next((i for i, n in enumerate(names) if n.startswith("定时行为")), None)
        check(rest_i == 1, "休息没紧跟「当前任务」（休息最优先）：%r" % (names,))
        check(tim_i is not None and tim_i > rest_i,
              "自定义定时行为没排在休息后面：%r" % (names,))
        check(len(names) == len(set(names)), "有重复行（每项一行）：%r" % (names,))
        check(any(n.startswith("休息") and ("剩余 2:0" in n) for n in names),
              "休息没写出剩余时间：%r" % (names,))
        check(any(n.startswith("定时行为") and "剩余 1:0" in n for n in names),
              "自定义定时行为没写出剩余时间：%r" % (names,))
        check(not any("喊话" in n for n in names),
              "**暂停的定时行为不该显示**（2026-09-26 用户要求）：%r" % (names,))
        # ⚠ 「定点休息」也要显示（2026-09-26 用户报：手动进入定点休息后，界面写着
        #   「未休息」✗）—— 这里钉**实时页**那行；玩家面板那张卡片与"每个状态都得有
        #   说法"由 `selftest_decision.t_rest_state_text_covers_all_states` 钉。
        ds.rest_state = "afk_spot_rest"
        ds.rest_until_monotonic = _time.monotonic() + 61
        names2 = [r[0] for r in p._osd_lines("x") if not isinstance(r, str)]
        check(any(n.startswith("休息") and "定点休息中" in n for n in names2),
              "定点休息没出现在「计时任务」里：%r" % (names2,))
        check(any(n.startswith("休息") and "剩余 1:0" in n for n in names2),
              "定点休息没写出剩余时间：%r" % (names2,))
        ds.rest_state = "afk_spot_walk"          # 去休息点的路上（还没开始计时）
        ds.rest_until_monotonic = 0.0
        names3 = [r[0] for r in p._osd_lines("x") if not isinstance(r, str)]
        check(any(n.startswith("休息") and "前往休息点" in n for n in names3),
              "去休息点那一段没显示：%r" % (names3,))

        # ③ 「当前任务」在休息时必须写**「休息」**（用户 2026-09-26 要求）——
        #    「定点休息」会挂着一条"走过去"的任务，那行若还写「前往：X」，会让人以为
        #    在执行「命令前往」（目的地确实是休息点 ✓，但主人是休息机器 ✓，目的地
        #    由上面那行「前往休息点…「X」」说清 ✓）。
        ds.rest_state = "afk_spot_rest"
        ds.rest_until_monotonic = _time.monotonic() + 61
        lines_r = [r[0] for r in p._osd_lines("x") if not isinstance(r, str)]
        check(any(n.startswith("当前任务") and "休息" in n for n in lines_r),
              "休息时「当前任务」没写「休息」：%r" % (lines_r,))
        check(not any(n.startswith("当前任务") and "战斗" in n for n in lines_r),
              "休息时「当前任务」还写着「战斗」：%r" % (lines_r,))
        # ⚠ 「自动喂宠」那一行**已移除**（用户 2026-09-26 去掉整个功能）—— 这里原本断言
        #    它是计时任务之一；现在"喂宠"只会作为**用户自己配的定时行为**出现（上面那条
        #    `custom_timers` 里的「喂宠」就是那样用的 ✓，走的是同一条循环 ✓）。
        # 定时清键要**倒计时**，不是"每 N s"（2026-09-26 用户要求）
        check(any(n.startswith("定时清键") for n in names),
              "定时清键是计时任务，该列出来：%r" % (names,))
        check(not any("每 " in n for n in names),
              "定时清键还写着「每 N s」—— 要倒计时：%r" % (names,))
        check(any(n.startswith("定时清键") and "未排期" in n for n in names),
              "实时没在跑时「定时清键」该写「未排期」（别显示 0:00 骗人）：%r" % (names,))

        class _Fake3:
            def current_goto_set(self):
                return ""

            def resetall_left(self):
                return 25.0

        with mock.patch.object(agent_mod, "CURRENT", _Fake3()):
            names2 = [r[0] for r in p._osd_lines("x") if not isinstance(r, str)]
            check(any(n.startswith("定时清键") and "剩余 0:2" in n for n in names2),
                  "定时清键没写出倒计时：%r" % (names2,))

        # 要求 3：「命令前往」右边有「结束当前寻路」
        check(hasattr(p, "btn_stop_goto") and "结束" in p.btn_stop_goto.text(),
              "「命令前往」右边没有「结束当前寻路」按钮")

        # 要求 4（2026-09-26）：「选择平台 / 命令前往 / 结束当前寻路」整块**显示在
        #   决策参数页的操控区**，但**控件与逻辑仍归路线识别面板** —— 这样「预览」
        #   才画得到它自己那张地形图，两边也不会各造一套（必然分叉 ✗）。
        from gui.player_panel import PlayerPanel
        check(hasattr(p, "goto_box"), "路线识别面板里没有 `goto_box`（搬控件要用它）")
        check(hasattr(PlayerPanel, "mount_goto"),
              "玩家面板里没有 `mount_goto`（没法把那一块搬进操控区）")
        pp = PlayerPanel()
        try:
            check(pp._goto_mounted is None, "还没搬就以为搬过了")
            pp.mount_goto(p.goto_box)
            check(pp._goto_mounted is p.goto_box
                  and p.goto_box.parent() is pp.goto_holder,
                  "「前往平台」没被放进操控区的占位框里")
            check(p.cmb_goto.parent() is p.goto_box,
                  "搬迁把控件从原来的容器里摘出去了（逻辑还指着它们 ✗）")
            pp.mount_goto(p.goto_box)                 # 幂等：再搬一次不许插两份
            # ⚠ 不能用 `_goto_lay.count() == 1` 判了：「命令」组里现在还有**手动休息那一行**
            #   （用户 2026-09-26 从「防掉线」组搬来 ✓）⇒ 要按"这块控件出现了几次"判 ✓
            _hits = [i for i in range(pp._goto_lay.count())
                     if pp._goto_lay.itemAt(i).widget() is p.goto_box]
            check(len(_hits) == 1, "重复 mount 插了两份（%d 份）" % len(_hits))
            check(pp._goto_lay.indexOf(p.goto_box) == 0,
                  "「前往平台」没排在「命令」组最上面（下面那行是手动休息 ✓）：index=%d"
                  % pp._goto_lay.indexOf(p.goto_box))
            # 三件手动休息控件要**搬进「命令」组**（用户 2026-09-26；原来在「防掉线」✗）
            from PyQt5.QtWidgets import QGroupBox as _GB
            check(pp.goto_holder.title() == "命令",
                  "「前往平台」那块没改名叫「命令」：%r" % pp.goto_holder.title())
            for _w, _name in ((pp.btn_start_rest, "手动进入休息"),
                              (pp.btn_end_rest, "手动结束休息"),
                              (pp.lbl_rest, "休息状态")):
                _par = _w.parentWidget()
                while _par is not None and not isinstance(_par, _GB):
                    _par = _par.parentWidget()
                check(_par is pp.goto_holder,
                      "%s 不在「命令」组里（现在挂在 %s）"
                      % (_name, _par.title() if _par is not None else "？"))
        finally:
            # ⚠ 先**卸下来**再销毁面板：`mount_goto` 把 `goto_box` 的父级换成了 `pp`，
            #   直接销毁 `pp` 会连它一起析构 ✗ ⇒ `RoutePanel` 手里那份就成了悬空引用 ⇒
            #   use-after-free（这条用例就是这么崩的 ✗）。真程序里两个面板一直活着，
            #   不会走到这一步；用例要收工就得自己收拾干净 ✓。
            p.goto_box.setParent(None)
            # 立刻销毁（`deleteLater` 在自检里**没有事件循环** ⇒ 等于没销毁 ✗，
            # 对象活到进程退出、和 GC 抢析构 ⇒ 访问违例）——写在用它的地方，
            # 免得两个自检脚本互相 import（它们都是能单独跑的脚本 ✓）。
            import gc
            from PyQt5 import sip
            pp.close()
            pp.setParent(None)
            sip.delete(pp)
            del pp
            gc.collect()
        called = []

        class _Fake2:
            def current_goto_set(self):
                return "甲平台"

            # ⚠ 只实现 `stop_route`（= **整条路线**）：2026-09-26 起「结束当前寻路」走的是它
            #   （只停当前一步会把后面几步留成"死账"，见 docs/寻路设计.md 的「限制战斗区域」）。
            #   **故意不实现 `stop_climb`** ⇒ 哪天按钮改回"只停一步"，这里当场 AttributeError ✓
            #   （这次改动就是被它抓到的：'_Fake2' object has no attribute 'stop_route'）。
            def stop_route(self, why=""):
                called.append(why)
                return True

        with mock.patch.object(agent_mod, "CURRENT", _Fake2()):
            p._on_stop_goto()
        check(called, "点了「结束当前寻路」却没撤整条路线（stop_route 没被调）")
        with mock.patch.object(agent_mod, "CURRENT", None):
            p._on_stop_goto()
            check("实时" in p.lbl_goto.text(),
                  "没有实时在跑时该说清（不是静悄悄什么都不做）：%r"
                  % p.lbl_goto.text())
        p.close()
    finally:
        for k, v in saved.items():
            setattr(ds, k, v)


def t_goto_picker_and_preview():
    """「选择平台」下拉 = 注册过的集合；选中要框在图上；「命令前往」要算得出路。

    对应 2026-09-26 的要求（给路线测试用的那组控件）。三点各自会让用例红：
      · 下拉不跟着集合文件走 → 新圈的集合要重启才选得到；
      · 预览坐标自己算一份 → 框画在别处，看着像"集合圈错了"（这种最难查）；
      · 定位不到/读数过期时还照常算路 → 拿一个假起点报"能走到"，会让人照它去测。
    """
    import tempfile
    import unittest.mock as mock

    from PyQt5.QtWidgets import QApplication
    from core import zones as zones_mod
    from decision.agent import settings as dsettings
    from gui.route_panel import RoutePanel

    app = QApplication.instance() or QApplication([])
    mid, _canvas = pick_map()
    t = mapdata.load(mid, with_canvas=True) if mid else None
    if t is None or not t.segments:
        print("      （没有带段的地形数据，跳过）")
        return
    walk = [f for f in t.footholds if not f.is_wall]
    if len(walk) < 2:
        print("      （foothold 太少，跳过）")
        return

    tmp = Path(tempfile.mkdtemp(prefix="goto_"))
    zp = tmp / ("%s.zones.json" % mid)
    z = zones_mod.Zones(mid)
    z.add_set("甲平台", [str(walk[0].fid)])
    z.add_set("乙平台", [str(walk[1].fid)])
    z.save(zp)
    p = RoutePanel()
    p._map_id = lambda: mid
    # 「路线识别」这一组**只剩「寻路编辑器」**（2026-09-26 用户定：其余没意义）。
    # 那个「启用路线识别」门控的是**感知**（平台识别/落点预测/小地图定位）—— 关掉时
    # 「命令前往」连"你在哪个平台"都答不出来，是"点一下没反应"的典型来源，所以
    # 连开关带它门控的那几处一起撤了（见 gui/live_thread.py 与 route_panel.__init__）。
    check(not hasattr(p, "ck_enabled"), "「启用路线识别」开关又回来了")
    check(not hasattr(p, "lbl_hint"), "那行状态提示又回来了")
    # ⚠ 文案 2026-09-26 用户改为「寻路编辑器」（原「编辑集合…」）—— 用例跟着改；
    #   这里钉的是"这个**入口**还在"（这一组唯一要留的），不是文案本身，
    #   所以用 `btn_zones` 这个**属性名**判在不在，文案只做一次 `in` 兜底。
    check(hasattr(p, "btn_zones") and "寻路编辑器" in p.btn_zones.text(),
          "「寻路编辑器」按钮被误删 / 改名了（那是这一组唯一要留的）：%r"
          % (p.btn_zones.text() if hasattr(p, "btn_zones") else None,))
    check(not hasattr(dsettings, "route_enabled"),
          "DecisionSettings 里又出现了 route_enabled —— 感知那道闸已经撤了")
    try:
        with mock.patch.object(zones_mod, "ZONES_DIR", tmp), \
             mock.patch.object(dsettings, "save", lambda *a, **k: None):
            p._refresh_goto()
            items = [p.cmb_goto.itemData(i) for i in range(p.cmb_goto.count())]
            check(items and items[0] == "", "第一项不是空项：%s" % items)
            check("甲平台" in items and "乙平台" in items,
                  "下拉没按注册的集合填：%s" % items)
            check(p.btn_goto.isEnabled(), "有集合时「命令前往」应当可用")

            # 「上绳梯失败后延迟激活时间」（2026-09-26 用户要求 1）：这一页要有它、
            # 显示当前值、改了要写回配置（它是决策参数，跟着项目存）
            check(hasattr(p, "sp_retry"),
                  "「路线识别」页里没有「上绳梯失败后延迟激活时间」")
            check(abs(float(p.sp_retry.value())
                      - float(dsettings.climb_retry_delay_s)) < 1e-6,
                  "控件没显示当前值：%s vs %s"
                  % (p.sp_retry.value(), dsettings.climb_retry_delay_s))
            was_rd = dsettings.climb_retry_delay_s
            p.sp_retry.setValue(2.5)
            check(abs(float(dsettings.climb_retry_delay_s) - 2.5) < 1e-6,
                  "改了延迟没写回配置：%r" % dsettings.climb_retry_delay_s)
            # 换项目（bind）要重读当前项目的值，而且**不许**触发写回
            dsettings.climb_retry_delay_s = 3.0
            p.bind(None)
            check(abs(float(p.sp_retry.value()) - 3.0) < 1e-6,
                  "bind 没重读延迟（切项目会显示上一个项目的值）：%s"
                  % p.sp_retry.value())
            check(abs(float(dsettings.climb_retry_delay_s) - 3.0) < 1e-6,
                  "bind 里的 setValue 又写回配置了（会拿旧项目的值覆盖新项目）：%r"
                  % dsettings.climb_retry_delay_s)
            dsettings.climb_retry_delay_s = was_rd

            # 选中 → 存进配置 + 图上框出来（坐标必须和画图那套换算一致）
            p.cmb_goto.setCurrentIndex(p.cmb_goto.findData("乙平台"))
            p._on_goto_pick()
            check(dsettings.route_goto_set == "乙平台",
                  "选择没存进配置：%r" % dsettings.route_goto_set)
            boxes = p._preview_boxes(mid)
            check(len(boxes) == 1, "预览没给出框：%s" % boxes)
            # 平台是一根横线：包围盒可能是 0 高，框得**补到看得见**（不然等于没框）
            check(boxes[0][3] >= 8 and boxes[0][4] >= 8,
                  "预览框太小了（%.1f×%.1f）—— 平台那种扁的会变成一条发丝"
                  % (boxes[0][3], boxes[0][4]))
            from tools.map_terrain_view import image_xy
            sp = zones_mod.set_span(t, z.sets["乙平台"]["footholds"])
            ex0, ey0 = image_xy(t, sp[0], sp[2])
            ex1, ey1 = image_xy(t, sp[1], sp[3])
            # 比**中心**：扁平台的框会被补到最少 8px（中心不变），比左上角会差那 4px
            cx = boxes[0][1] + boxes[0][3] / 2.0
            cy = boxes[0][2] + boxes[0][4] / 2.0
            check(abs(cx - (ex0 + ex1) / 2.0) < 1.5
                  and abs(cy - (ey0 + ey1) / 2.0) < 1.5,
                  "预览框没框在平台上（中心对不上）：(%.1f, %.1f) vs (%.1f, %.1f)"
                  % (cx, cy, (ex0 + ex1) / 2.0, (ey0 + ey1) / 2.0))

            # 定位不到玩家 / 读数过期 ⇒ 明说，**不许**拿假起点算路
            p._fh_seen = ("", 0.0)
            p._on_goto()
            check("还不知道你在哪个平台" in p.lbl_goto.text(),
                  "定位不到玩家时没说清：%r" % p.lbl_goto.text())
            p._fh_seen = (str(walk[0].fid), 0.0)
            p._on_goto()
            check("还不知道你在哪个平台" in p.lbl_goto.text(),
                  "用过期的读数当起点了：%r" % p.lbl_goto.text())

            # 有新鲜位置、但两个集合之间**没有边** ⇒ 走不到，并说出边界
            p._fh_seen = (str(walk[0].fid), time.monotonic())
            p._on_goto()
            check("走不到" in p.lbl_goto.text(),
                  "没有边却说能走到：%r" % p.lbl_goto.text())
            check("甲平台" in p.lbl_goto.text(),
                  "走不到时没说出边界（从哪能到哪）：%r" % p.lbl_goto.text())

            # 补上边 ⇒ 能走到，并把逐步路线写出来
            z.add_edge("甲平台", "乙平台", "walk")
            z.save(zp)
            p._on_goto()
            check("能走到" in p.lbl_goto.text()
                  and "甲平台 → 乙平台" in p.lbl_goto.text(),
                  "加了边还说走不到：%r" % p.lbl_goto.text())
    finally:
        p.close()
        import shutil
        shutil.rmtree(str(tmp), ignore_errors=True)


def t_settings_overlay_takes_effect():
    """「设置 → 界面」里那两项**真的生效**（2026-09-26 用户要求确认）。

    链路要一次全走通，缺一环都表现为"改了没反应"✗：
        弹窗勾选/拖条 → `_accept` 写配置（live.yaml 的 `mmap_draw` / 标定里的 `alpha`）
        → `overlay_changed` 信号 → `route_panel.apply_overlay_settings`
        → `set_minimap_overlay(..., alpha)` 推到实时画面那一层。
    ⚠ 这里**不手工调** `apply_overlay_settings` —— 就是为了钉住"信号真的发了"
      （少接一根线，样子就是"设置里改了、画面不动"✗，光看代码看不出来）。
    ⚠ 还有 0：拖动条拉到最左端必须**真的是 0%**（不是被默认 55% 吃掉 ✗）。
    """
    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return

    import unittest.mock as mock

    from PyQt5.QtWidgets import QApplication
    import gui.route_panel as rp
    import gui.settings_dialog as sdmod
    from gui.route_panel import RoutePanel
    from gui.settings_dialog import SettingsDialog

    app = QApplication.instance() or QApplication([])

    class _FakeLive:
        """假实时面板：只记「那一层被挂了什么」。`pix=None` = 卸掉（同真面板 ✓）。"""

        def __init__(self):
            self.ov = None

        def current_frame(self):
            return None

        def set_minimap_overlay(self, pix, src=None, frame_rect=None,
                                alpha=0.55, note=""):
            self.ov = None if pix is None else (pix, src, frame_rect, alpha, note)

        def set_overlay_note(self, note):
            return False

    live = _FakeLive()
    p = RoutePanel()
    p.live_panel = live
    p.project = _Proj(mid)
    # 假配置 + 假标定：真 live.yaml / 真标定文件**一个字节都不碰** ✓（自检不许改用户配置）
    cfg = {"mmap_src": mm.SRC_LIVE, "mmap_draw": False,
           "mmap_crop": [0, 0, int(canvas.shape[1]), int(canvas.shape[0])]}
    cal = {"mode": mm.MODE_FIT, "scale": 1.0, "offset": [0, 0], "view": [0, 0],
           "score": 1.0, "alpha": 55}
    with mock.patch.object(rp, "load_live", lambda: dict(cfg)), \
         mock.patch.object(rp, "update_live", lambda **kw: cfg.update(kw)), \
         mock.patch.object(sdmod, "load_live", lambda: dict(cfg)), \
         mock.patch.object(sdmod, "update_live", lambda **kw: cfg.update(kw)), \
         mock.patch.object(sdmod, "last_opened", lambda: _Proj(mid)), \
         mock.patch.object(mapdata, "load_calib", lambda _m, src=None: dict(cal)), \
         mock.patch.object(mapdata, "save_calib",
                           lambda _m, c, src=None: cal.update(c)):
        try:
            p._refresh_overlay()
            check(live.ov is None, "开关关着就把叠图挂上去了：%s" % (live.ov,))

            sd = SettingsDialog()
            check(sd.ck_mmap_draw.isChecked() is False,
                  "弹窗没按配置回填（开关该是关的）")
            check(sd.sld_alpha.value() == 55,
                  "弹窗没按标定回填浓淡：%s" % sd.sld_alpha.value())
            # 和主窗口同一条接线（少这一根，改了设置画面不动 ✗）
            sd.overlay_changed.connect(p.apply_overlay_settings)

            sd.ck_mmap_draw.setChecked(True)
            sd.sld_alpha.setValue(20)
            sd._accept()
            check(cfg.get("mmap_draw") is True, "勾了开关没写进配置：%s" % cfg)
            check(int(cal.get("alpha")) == 20, "拖了浓淡没写进标定：%s" % cal)
            check(live.ov is not None, "勾了开关、画面那层却没挂上（信号没接上？）")
            check(abs(float(live.ov[3]) - 0.20) < 1e-9,
                  "浓淡没传到画面那层：alpha=%r（该 0.20）" % (live.ov[3],))
            check("已画在画面" in p.lbl_mmap_draw.text(),
                  "状态行没说画上了：%r" % p.lbl_mmap_draw.text())

            # 重开设置：要显示**刚存下**的值（回填不对 = 下次按错的值覆盖 ✗）
            sd2 = SettingsDialog()
            check(sd2.ck_mmap_draw.isChecked() is True and sd2.sld_alpha.value() == 20,
                  "重开设置没回填：开关=%s 浓淡=%s"
                  % (sd2.ck_mmap_draw.isChecked(), sd2.sld_alpha.value()))
            sd2.overlay_changed.connect(p.apply_overlay_settings)

            # ★ 拖到**最左端**：存档要 0、画面那层要真变成 0.0（不是被默认 55% 吃掉 ✗）
            sd2.sld_alpha.setValue(0)
            sd2._accept()
            check(int(cal.get("alpha")) == 0, "拉到最左端没存成 0：%s" % cal)
            check(live.ov is not None and float(live.ov[3]) == 0.0,
                  "拉到最左端画面那层不是 0.0（0 被默认值吃掉了）：%r"
                  % (live.ov[3] if live.ov else None,))

            # 关掉开关 ⇒ 那一层要卸掉、状态行要说明
            sd3 = SettingsDialog()
            sd3.overlay_changed.connect(p.apply_overlay_settings)
            sd3.ck_mmap_draw.setChecked(False)
            sd3._accept()
            check(cfg.get("mmap_draw") is False, "关掉开关没写进配置：%s" % cfg)
            check(live.ov is None, "关掉开关了那一层还挂着：%s" % (live.ov,))
            check("（没开）" in p.lbl_mmap_draw.text(),
                  "状态行没跟着说「没开」：%r" % p.lbl_mmap_draw.text())
        finally:
            p.close()


def t_route_panel_mmap_crop():
    """「框选小地图」在**路线识别页**、而且**按项目存**（2026-09-27 用户要求搬回）。

    用户 2026-09-27：① 它**不该在设置里** —— 不同项目的小地图尺寸完全不同，必须
    **按项目**各存一份；现在长在「路线识别 → 寻路配置」，紧贴「寻路编辑器」下面 ✓。
    ② 框完那次的匹配分就在那一页当场报 ✓。

    钉五件：
      ① 真控件树：按钮归在「寻路配置」组里，而且**在「寻路编辑器」下方**（量坐标 ✓）；
      ② 没打开项目 / 没选地图 ⇒ **一个字都不写** + 说清"先选地图"（按项目存，没处存 ✓）；
      ③ 没画面 ⇒ **不写** + 指路（去「实时」页点开始）；
      ④ 有画面 ⇒ 写进**项目**（`save=True` ⇒ 真落盘）、**不动 `mmap_src`**、
         **也不碰 config/live.yaml**（那正是"换项目就错"的来源 ✗），并当场报分（报绿）；
      ⑤ 状态行三种情况都说实话：本项目框过 / 本项目没框过（暂用老的）/ 两边都没有。
    ⚠ 项目对象、配置、标定全 mock；`select_region_on_image` 也 mock（不然会弹真框选窗）；
      真 `project.yaml` / `live.yaml` / 标定文件一个字节都不碰 ✓。
    """
    import unittest.mock as mock

    from PyQt5.QtCore import QPoint
    from PyQt5.QtWidgets import QApplication, QGroupBox

    import gui.route_panel as rpmod
    from gui.route_panel import RoutePanel

    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return
    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")

    cw, ch = canvas.shape[1], canvas.shape[0]
    # 「实时画面」就放一张**和底图一样大**的图（scale=1）⇒ 拿整张当框选区时，
    # 裁出来那块应当和底图对得上（匹配分 ≈ 1.0）✓
    panel = synth_fit_panel(canvas, 1.0, 0, 0, pad=0)
    crop = [0, 0, cw, ch]
    cfg = {"mmap_src": mm.SRC_STREAM, "mmap_draw": True, "mmap_crop": None}
    cal = {"mode": mm.MODE_FIT, "scale": 1.0, "offset": [0, 0], "view": [0, 0],
           "score": 1.0, "alpha": 55}
    proj = _Proj(mid)                      # 本项目**还没框过**

    class _Live:
        """假实时面板：框选要 `current_frame()`，框完 `_refresh_overlay()` 要能接住叠图。"""

        def __init__(self):
            self.ov = None

        def current_frame(self):
            return panel

        def set_minimap_overlay(self, pix, src=None, frame_rect=None,
                                alpha=0.55, note=""):
            self.ov = None if pix is None else (pix, src, frame_rect, alpha, note)

        def set_overlay_note(self, note):
            pass

    tips = []
    p = RoutePanel()
    p.resize(900, 700)
    try:
        with mock.patch.object(rpmod, "load_live", lambda: dict(cfg)), \
             mock.patch.object(mapdata, "load_calib", lambda _m, src=None: dict(cal)), \
             mock.patch("core.route_cfg.save", lambda *a, **k: None), \
             mock.patch.object(rpmod.QMessageBox, "information",
                               # information(parent, title, text) ⇒ 正文在 a[2]
                               lambda *a, **k: tips.append(a[2] if len(a) > 2 else "")):
            # ① 归组 + **位置**（真控件树量坐标：不是靠代码顺序 ✓）
            grp, q = None, p.btn_mmap_crop.parent()
            while q is not None and grp is None:
                grp = q if isinstance(q, QGroupBox) else None
                q = q.parent()
            check(grp is not None and "寻路配置" in grp.title(),
                  "「框选小地图」没归在「寻路配置」那一组里：%r"
                  % (grp.title() if grp is not None else None,))
            p.show()
            app.processEvents()
            y_crop = p.btn_mmap_crop.mapTo(p, QPoint(0, 0)).y()
            y_zones = p.btn_zones.mapTo(p, QPoint(0, 0)).y()
            check(y_crop > y_zones,
                  "「框选小地图」没在「寻路编辑器」**下方**（y %d vs %d）—— "
                  "用户要求就放在那个按钮下面" % (y_crop, y_zones))
            p.hide()

            # ② 没打开项目 / 没选地图 ⇒ 一个字都不写 + 说清"先选地图"
            p.bind(None)
            p._pick_mmap_crop()
            check(cfg.get("mmap_crop") is None,
                  "没打开项目却把框选区写进了 config/live.yaml：%s" % cfg.get("mmap_crop"))
            check(tips and any("选地图" in t for t in tips),
                  "没项目时没指路（该说「先选地图」）：%s" % tips)

            # ③ 有项目、还没画面 ⇒ 不写 + 指路
            p.bind(proj)
            tips.clear()
            p.live_panel = None
            p._pick_mmap_crop()
            check(proj.get("mmap_crop") is None,
                  "没有画面却把框选区写进了项目：%s" % proj.get("mmap_crop"))
            check(tips and any("实时" in t for t in tips),
                  "没画面时没指路（该说「去「实时」页点开始」）：%s" % tips)

            # ④ 有画面：框一次（框选窗 mock 掉）⇒ 写进**项目**、不动来源、不碰 live.yaml
            p.live_panel = _Live()
            with mock.patch("gui.region_selector.select_region_on_image",
                            lambda *a, **k: list(crop)):
                p._pick_mmap_crop()
            check([int(v) for v in (proj.get("mmap_crop") or [])]
                  == [int(v) for v in crop],
                  "框完没写进**项目**（框选必须按项目存）：%s" % proj.get("mmap_crop"))
            check(proj.saved, "框完没落盘（project.yaml 里还是空的）")
            check(cfg.get("mmap_crop") is None,
                  "框选又去写 config/live.yaml 了（那正是「换项目就错」的来源 ✗）：%s"
                  % cfg.get("mmap_crop"))
            check(cfg.get("mmap_src") == mm.SRC_STREAM,
                  "框选顺手把**来源**改掉了 —— 它只该存位置：%s" % cfg.get("mmap_src"))
            note = p.lbl_crop_note.text()
            check("已框选" in note and "匹配分" in note, "没当场报匹配分：%r" % note)
            check("#188038" in p.lbl_crop_note.styleSheet(),
                  "和底图对上了却没报绿：%r（%s）" % (note, p.lbl_crop_note.styleSheet()))

            # ⑤ 状态行三种情况都要说实话
            p._refresh_crop_status()
            check("本项目：" in p.lbl_mmap_crop.text() and "×" in p.lbl_mmap_crop.text(),
                  "状态行没跟着换成刚框的那一块：%r" % p.lbl_mmap_crop.text())
            cfg["mmap_crop"] = [5, 6, 7, 8]              # 老的那份（全局）还在
            p.bind(_Proj(mid))                           # 换个"本项目没框过"的项目
            check("暂用老的那份" in p.lbl_mmap_crop.text(),
                  "本项目没框过、正在用老的那份值，界面却不说明：%r"
                  % p.lbl_mmap_crop.text())
            cfg["mmap_crop"] = None                      # 两边都没有
            p.bind(_Proj(mid))
            check("还没框" in p.lbl_mmap_crop.text(),
                  "哪里都没有框选区、界面却说有：%r" % p.lbl_mmap_crop.text())
    finally:
        p.close()
        p.deleteLater()


def t_overlay_waits_first_frame():
    """叠图**不许在"折算比例还不知道"时先画一版错的**（2026-09-26 用户报）。

    现象（用户原话）：「小地图上显示的叠加图…的尺寸缩放在刚打开实时的时候是不正确的，
    要点一下路线识别页签才正常」。已复现（现场数字：底图 134×101、标定
    `sources.stream{scale:1.874, offset:[0,0]}`、`mmap_crop=[6,72,134,109]`）：

        ① 刚打开实时（收流还没来帧）→ 叠图矩形 (6, 72, 251, 189)  ← 宽 1.873 倍，糊到框外
        ② 收流来帧之后、什么都不点   → 还是 (6, 72, 251, 189)     ← 冻住了，不会自己变对
        ③ 点一下「路线识别」页签      → (6, 72, 134, 101) ✓

    根因两条：
      · "标定当时那块面板多大"（`_calib_panel_wh`）来源=收流时只能问**当前收流帧**，
        而收流客户端是**懒建**的 ⇒ 第一次刷新必然拿不到帧 ⇒ 给 None ⇒
        `frame_overlay_rects(..., calib_panel=None)` **不折算** ⇒ 直接拿标定的 scale 画；
      · 画上之后 **没人再重算**：250ms 那个节拍（`_tick_world`）只在"叠图还没挂上"时
        才整算一次 ⇒ 尺寸冻结在第一次 ✗（点页签走的是 `showEvent → _refresh_mmap
        → _refresh_overlay`，那时早就有帧了 ⇒ 所以你看到"点一下才对"）。

    钉五件事：
      ① 帧没来 ⇒ **不画**，而且状态行说的是"在等收流第一帧"，**不许**冒出那句
         "标定和这块框对不上"（那时候标定是好的，说它不对是冤枉 ✗）；
      ② 帧来了 ⇒ 250ms 那条重试路会把它画上，且尺寸**折算到面板框那么宽**（≈ 底图宽，
         而不是 底图×1.874）；
      ③ 来源=「从实时画面」**不许**被这条拦住（那块面板就是框出来的这一块，比例本来就是 1）；
      ④ 标定**真**和框对不上时（`scale=5.63` 那种），状态行要把**两个尺寸**都摆出来；
      ⑤ 全程只读假配置 / 假标定 + 真底图文件 ⇒ 一个字节都不碰用户配置。
    """
    import unittest.mock as mock

    from PyQt5.QtWidgets import QApplication

    import gui.route_panel as rp
    from gui.route_panel import RoutePanel

    # 得挑一张**有叠加图**的图（`_overlay_blocker` 要求 `<id>_overlay.png` 在）——
    # 没有它，叠图那条路根本走不到折算这一步，用例就测空。
    mid = canvas = None
    for q in sorted(mapdata.map_dir().glob("*_overlay.png")):
        m = q.name[: -len("_overlay.png")]
        tt = mapdata.load(m, with_canvas=True)
        if tt is not None and tt.canvas is not None:
            mid, canvas = m, tt.canvas
            break
    if mid is None:
        print("      （没有带叠加图的地图，跳过）")
        return
    cw, ch = canvas.shape[1], canvas.shape[0]
    SU = 1.874                                   # 现场标定里的 scale
    panel_cal = (int(round(cw * SU)), int(round(ch * SU)))   # 标定当时那块收流面板
    crop = (6, 72, cw, ch + 8)                   # 现场 mmap_crop 的形状（折算只看宽）
    cal = {"mode": mm.MODE_FIT, "scale": SU, "offset": [0, 0], "view": [0, 0],
           "score": 0.7272, "alpha": 35}         # 现场那份（**没有 `panel` 字段**）

    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")

    class _FakeLive:
        """假实时面板：如实模拟真面板那两条（挂/卸 + 只换那行字）。"""

        def __init__(self):
            self.ov = None

        def current_frame(self):
            return None

        def set_minimap_overlay(self, pix, src=None, frame_rect=None,
                                alpha=0.55, note=""):
            if pix is None or pix.isNull() or frame_rect is None:
                self.ov = None            # 真面板：传 None / 空图 / 没矩形 ⇒ 卸掉
                return
            self.ov = (pix, src, frame_rect, alpha, note)

        def set_overlay_note(self, note):
            if self.ov is None:
                return False              # ← 这一句就是"没挂上 ⇒ 调用方整算一次"的开关
            self.ov = tuple(self.ov[:4]) + (note or "",)
            return True

    class _NoFrameCli:
        """假收流客户端：`latest()` 一开始必然给 None（真客户端是懒建的）。"""

        def __init__(self):
            self.frame = None
            self.err = ""

        def latest(self):
            return self.frame, 0.0

    live = _FakeLive()
    cli = _NoFrameCli()
    p = RoutePanel()
    p.live_panel = live
    p.project = _Proj(mid)
    cfg = {"mmap_src": mm.SRC_STREAM, "mmap_draw": True, "mmap_crop": list(crop)}
    try:
        with mock.patch.object(rp, "load_live", lambda: dict(cfg)), \
             mock.patch.object(mapdata, "load_calib", lambda _m, src=None: dict(cal)):
            p._stream_client = lambda: cli
            # ① 收流第一帧还没来
            p._refresh_overlay()
            check(live.ov is None,
                  "折算比例还不知道就把叠图画上去了（尺寸必错）：%r"
                  % (live.ov[2] if live.ov else None,))
            txt = p.lbl_mmap_draw.text()
            check("第一帧" in txt, "没告诉人在等收流第一帧：%r" % txt)
            check("对不上" not in txt,
                  "把自己没画的原因说成「标定和框对不上」—— 那时标定是好的 ✗：%r" % txt)
            # ② 帧来了 → 250ms 那条重试路（`_tick_world` 的 `_say` 里就是这两句）
            cli.frame = np.zeros((panel_cal[1], panel_cal[0], 3), np.uint8)
            if not live.set_overlay_note("（读数那行换一次字）"):
                p._refresh_overlay()
            r = live.ov[2] if live.ov else None
            check(r is not None,
                  "收流来帧了也没自己补画上（250ms 那条重试路断了）")
            check(abs(r[2] - cw) <= 1 and abs(r[3] - ch) <= 1,
                  "画上了但没折算到面板框：%s（该约 %dx%d）" % (r, cw, ch))
            check(r[2] < int(crop[2]) * 1.2, "叠图比面板框还大 —— 折算没生效：%s" % (r,))
            # ③ 来源=从实时画面：那块面板就是框出来的这一块 ⇒ 不许被这条拦住
            cfg["mmap_src"] = mm.SRC_LIVE
            cal["scale"] = 1.0
            live.ov = None
            p._refresh_overlay()
            check(live.ov is not None,
                  "来源=「从实时画面」时被「等收流第一帧」拦住了（它压根不用收流）")
            # ④ 标定**真**不对（现场踩过：A 机 zoom=3 的帧上拟合出 scale=5.63）⇒ 要说明白
            cfg["mmap_src"] = mm.SRC_STREAM
            cfg["mmap_crop"] = [6, 72, 60, 40]
            cal["scale"] = 5.63
            live.ov = None
            p._refresh_overlay()
            t4 = p.lbl_mmap_draw.text()
            check("标定和这块框对不上" in t4,
                  "标定真和框对不上时没说清：%r" % t4)
            check("60×40" in t4,
                  "没说清面板框多大（人看不出差一截还是差几倍）：%r" % t4)
            check(cfg.get("mmap_draw") is True and cfg["mmap_crop"] == [6, 72, 60, 40]
                  and abs(cal["scale"] - 5.63) < 1e-9,
                  "这条用例把假配置/假标定改了：%s %s" % (cfg, cal))
    finally:
        p._world_timer.stop()
        p.close()


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
    """假项目：`RoutePanel._map_id()` 只读 `project.get("map_id")`。

    2026-09-27 扩了一格 `mmap_crop`（+ `set` / `saved`）：框选小地图**按项目存**
    （用户要求），用例得能验证"写进的是**项目**、不是 config/live.yaml" ✓ ——
    不扩的话就只能去碰真 `project.yaml` ✗（自检不许改用户文件）。
    `set()` 的签名照抄 `gui.project.Project.set(key, val, save=False)` ✓。
    """

    def __init__(self, mid, mmap_crop=None):
        self._d = {"map_id": mid, "mmap_crop": mmap_crop}
        self.saved = False

    def get(self, key, default=None):
        return self._d.get(key, default)

    def set(self, key, val, save=False):
        self._d[key] = val
        if save:
            self.saved = True


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

    ⚠ **黄点的「下沿（脚底）」对着 (cx,cy)**，不是重心（2026-09-27 改，见
    `perception.minimap.dot_feet`）：游戏把小地图上的玩家标记画成一个小黄点、下沿
    对着脚下，而读数与双点标定采样取的都是下沿 ⇒ 这里照这个口径摆，读数才能落回 (cx,cy)。
    按"重心对着"摆的话，读出来会凭空高半个点（这张图上 ≈ 2.5 个底图像素 = 40 世界像素 ✗）。
    """
    h = max(1, int(canvas.shape[0] * scale))
    w = max(1, int(canvas.shape[1] * scale))
    big = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_NEAREST)
    x = int(round(cx * scale - dot / 2.0))
    y = int(round(cy * scale)) - (dot - 1)          # 下沿（最后一行）落在 cy 上
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

    # **在不在绳梯上**（2026-09-26 要求：小地图那行要能报「绳梯：L2」）：
    # 把黄点挪到某根绳的中段 ⇒ ladder_id 必须是那根绳的编号（和编辑器画在绳上的、
    # 以及「爬」那条边里写的**同一个函数**算出来的编号）。
    from core import zones
    lids = zones.ladder_ids(t)
    L = t.ladders[0]
    lx, ly = t.world_to_canvas(L.x, (min(L.y1, L.y2) + max(L.y1, L.y2)) / 2.0)
    panel_l, _, _ = _panel_with_dot_at(canvas, lx, ly)
    rl = mm.PlayerLocator(mid).update(panel_l, src=mm.SRC_LIVE, calib=calib,
                                      terrain=t)
    check(rl["ok"], "站在绳上却定位失败：%s（黄点层：%s）" % (rl["note"], rl["dot"]))
    check(rl["ladder_id"] == lids[id(L)],
          "站在 %s 上却报成 %r" % (lids[id(L)], rl["ladder_id"]))
    check("ladder_id" in r, "定位结果里没有 ladder_id（那行就没法报绳梯）")
    check(r["ladder_id"] is None,
          "站在平台上却报「在绳上」：%r" % r["ladder_id"])

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


def t_minimap_client_timeout_says_where():
    """小地图收流超时要说清**卡在哪一步**（2026-09-26 排查用）。

    ⚠ 原来 `_recv_exact` 把异常全吞了（`except Exception: return None`）⇒ 超时被报成
    「ConnectionError: 对端关闭」—— A 机明明活着（只是没在推 / 卡住了），人却去查网络和
    进程。三种情况必须分得开：
      · 连上一帧都没来 → 「一帧都没来」+ 该查什么（推流在不在跑 / 端口 / 防火墙）；
      · 收过帧之后没动静 → 「推到一半停了」（那是 A 机侧卡住 / 被别的窗口盖住）；
      · 对端**真的**关了 → 才说「对端关闭」。
    """
    import socket as _socket
    import struct as _struct
    import time as _time
    import unittest.mock as mock

    import cv2
    import numpy as np

    from perception.minimap import MiniMapClient

    jpg = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))[1].tobytes()

    class _FakeSock:
        """连得上；先按脚本给字节，给完就开始超时（模拟 A 机不推 / 卡住）。"""

        def __init__(self, payload=b""):
            self.buf = payload
            self.pos = 0

        def settimeout(self, _t):
            pass

        def recv(self, n):
            if self.pos < len(self.buf):
                out = self.buf[self.pos:self.pos + n]
                self.pos += len(out)
                return out
            raise _socket.timeout("timed out")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def run_once(payload):
        cli = MiniMapClient("1.2.3.4", 5003, timeout=0.01)
        with mock.patch.object(_socket, "create_connection",
                               lambda *a, **k: _FakeSock(payload)):
            cli.start()
            for _ in range(300):
                if cli.err:
                    break
                _time.sleep(0.01)
            cli.stop()
        return cli

    cli = run_once(b"")
    check("一帧都没来" in cli.err,
          "首帧超时没说清是「一帧都没来」：%r" % cli.err)
    check("5003" in cli.err and "防火墙" in cli.err,
          "首帧超时没给出该查什么（端口 / 防火墙）：%r" % cli.err)

    cli = run_once(_struct.pack(">I", len(jpg)) + jpg)
    check(cli.n_recv >= 1, "假流里那一帧没被收下：n_recv=%d" % cli.n_recv)
    check("推到一半" in cli.err,
          "收过帧之后断了却说错（该说「推到一半停了」）：%r" % cli.err)


def t_apply_to_player():
    """定位结论写进 `WorldState.Player`：算不出来写 None（**不是 0**）。

    ⚠ 这条原来还钉着「别把地形段号写进**视觉平台编号**（`current_platform_id`）」——
    平台识别那套 2026-09-26 已整块移除（那个字段也跟着没了），所以只剩下面两组断言。
    """
    from perception.world_state import Player
    p = Player()
    mm.apply_to_player(p, {"world_x": 123.5, "world_y": -45.0, "segment_id": 3,
                           "note": "", "held": True})
    check(p.world_x == 123.5 and p.world_y == -45.0, "世界坐标没写进去")
    check(p.segment_id == 3, "段号没写进去")
    check(p.world_held is True,
          "「位置是沿用上一帧的」没写进 Player —— 决策层分不出新鲜观测和补位")

    mm.apply_to_player(p, {"world_x": None, "world_y": None, "segment_id": None,
                           "note": "没认出黄点"})
    check(p.world_x is None and p.segment_id is None,
          "定位失败时应当写 None（不是 0，0 是地图西北角这个合法坐标）")
    check(p.world_note == "没认出黄点", "失败原因没带到 Player 上")


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
            """真面板在挂着叠图时会走这条（不重读地形/底图）。这里如实模拟。

            ⚠ **原样存**，别 `str()`：真面板存的是可能是**多行列表**（世界坐标 +
            当前任务 + 计时任务，见 live_panel._draw_note）—— 套了 str 就会把整个
            列表画/测成一行字面量（这里踩过：假面板自己 stringify，用例跟着一起错）。
            """
            if self.ov is None:
                return False
            self.ov = tuple(self.ov[:4]) + (note or "",)
            self.note_only += 1
            return True

    live = _FakeLive(panel)
    p = RoutePanel()
    p.live_panel = live
    p.project = _Proj(mid)
    # ⚠ 叠图开关 2026-09-26 搬去**设置窗口**了 ⇒ 这里不再拨控件，而是直接改配置
    #   （配置下面的 `cfg` 里已经是 `mmap_draw: True` ✓）
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
            # 这行显示的**不再是「第 N 段」**（段号是自动串出来的、语义不可靠），
            # 而是"脚下的 foothold 属于哪个人工圈的集合"——那才是寻路的判据
            # （见 docs/寻路设计.md §12）。这里只钉**格式**，不钉具体集合名：
            # 集合是人在编辑器里圈的，自检不该依赖某个人当下的分组。
            check("位于fh：" in txt,
                  "没报出「位于fh：集合」：%r" % txt)
            check("第 %d 段" % seg.index not in txt,
                  "还在显示段号（语义不可靠，已换成 foothold 集合）：%r" % txt)
            # 数值要对得上（容差 = 一个底图像素：黄点本身有 6 个像素、取整也差一点）
            m = re.search(r"\((-?\d+), (-?\d+)\)", txt)
            check(m is not None, "这行没写出坐标数值：%r" % txt)
            gx, gy = int(m.group(1)), int(m.group(2))
            tol = t.px_per_world + 2
            check(abs(gx - wx) <= tol and abs(gy - wy) <= tol,
                  "读数里的坐标不对：(%d, %d)，期望约 (%.0f, %.0f)"
                  % (gx, gy, wx, wy))
            # ⚠ 贴到画面那一格现在是**多行**（世界坐标 + 当前任务 + 计时任务，
            # 2026-09-26 要求 1、2）：老形态是单个字符串。这里统一成"行文本列表"
            # 再断言 —— 要害没变：**每一行都得是短句**（写长了就是横穿半屏的黑带）。
            def osd_texts():
                n = (live.ov[4] if live.ov else "") or ""
                rows = list(n) if isinstance(n, (list, tuple)) else [n]
                return [r if isinstance(r, str) else r[0] for r in rows]

            check(live.ov is not None and any("fh：" in s for s in osd_texts()),
                  "画面里那块框下面没贴读数（fh 集合）：%r"
                  % (live.ov[4] if live.ov else None))
            check(all(len(s) <= 30 for s in osd_texts()),
                  "贴到画面上的行太长（每行都该是短句）：%r" % (osd_texts(),))
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
            osd = osd_texts()[0] if osd_texts() else ""
            check(osd and len(osd) <= 20,
                  "贴到画面上的那行太长（%d 字）：%r" % (len(osd), osd))
            check(len(p.lbl_mmap_world.toolTip()) > 40,
                  "详情没进 tooltip —— 那长诊断就白算了：%r"
                  % p.lbl_mmap_world.toolTip())

            # 关掉开关（= 配置关掉 ✓）→ 这行藏起来
            cfg["mmap_draw"] = False
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

        # 没地图 → 整件事不干（别白算）
        th.set_mmap(map_id="")
        check(th._locate_mmap(got) is None, "没有地图却在定位")
    finally:
        th.stop()


def t_route_panel_scrolls():
    """**长面板必须能滚**（用户 2026-09-27 报："路线识别页签不支持滚动？现在攀爬参数组的行和
    行都重叠了"）。

    这两个症状是**同一个病**：`RoutePanel` 的页面布局原来直接挂在 `QVBoxLayout(self)` 上
    —— **没有 `QScrollArea`** ✗ ⇒ 内容比页签高时 Qt 只能**硬挤** ⇒ 卡片里相邻的行被压到
    **互相重叠**（"不能滚"和"行重叠"一起来 ✓）。规范：docs/UI规范.md「长面板放进 QScrollArea」；
    实现只有一处 `gui.widgets.scroll_page` ✓。

    钉五件（都在**故意压到很矮**的窗口下量 —— 长面板的极限情况）：
      ① 面板里有一个 `QScrollArea`、`widgetResizable()` 为真 ✓；
      ② 压到 420px 高时**出竖直滚动条**（内容保持原高度 ⇒ 真能滚，不是把内容挤扁 ✓）；
      ③ `攀爬参数` 组里**没有两行是重叠的**（按几何算 —— 看源码看不出这件事 ✓）；
      ④ 组里的标签**没有被压成 0 高** ✓；
      ⑤ 从参数控件往上找，能找到那个滚动区（`NoWheel*` 的滚轮转发要认它 ✓ UI规范 §5）。
    """
    from PyQt5.QtCore import QPoint
    from PyQt5.QtWidgets import (QApplication, QGroupBox, QLabel, QScrollArea)

    from gui.route_panel import RoutePanel

    app = QApplication.instance() or QApplication([])
    p = RoutePanel()
    try:
        p.resize(760, 420)          # 故意很矮：长面板装不下时的样子
        p.show()
        app.processEvents()

        sc = p.findChild(QScrollArea)
        check(sc is not None and sc.widgetResizable(),
              "路线识别页里没有 QScrollArea（或者没 setWidgetResizable(True)）"
              "—— 用户 2026-09-27 报的「不支持滚动 / 攀爬参数行重叠」就是这个 ✗")
        check(sc.verticalScrollBar().maximum() > 0,
              "窗口压到 420px 高还是没出竖直滚动条（内容被硬挤进去了 ✗）")

        grp = next((g for g in p.findChildren(QGroupBox)
                    if g.title() == "攀爬参数"), None)
        check(grp is not None, "找不到「攀爬参数」组")
        rows = [lb for lb in grp.findChildren(QLabel) if lb.text().strip()]
        check(len(rows) >= 3,
              "「攀爬参数」组里的标签只有 %d 个 —— 这条量不出重叠" % len(rows))

        def box(w):
            tl = w.mapTo(p, QPoint(0, 0))
            return (tl.x(), tl.y(), tl.x() + w.width(), tl.y() + w.height())

        bad = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = box(rows[i]), box(rows[j])
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    bad.append((rows[i].text(), rows[j].text()))
        check(not bad,
              "「攀爬参数」组里有**互相重叠的行**（用户 2026-09-27 报的现象 ✗）：%s" % bad)
        for lb in rows:
            check(lb.height() > 0, "标签「%s」被压成 0 高" % lb.text())

        up, found = p.sp_align_gap.parentWidget(), False
        while up is not None:
            if isinstance(up, QScrollArea):
                found = True
                break
            up = up.parentWidget()
        check(found,
              "参数控件往上找不到滚动区 —— `NoWheel*` 的滚轮转发认不到它"
              "（指针压在参数框上就滚不动页面 ✗ UI规范 §5）")

        # ⑥ ⚠ **画布不许进滚动区**（2026-09-27 实测踩到的坑，别再"顺手"把它塞回去 ✗）：
        #    `ImageCanvas` 是 `QGraphicsView`（**它自己就是个滚动视图**）—— 套进外层
        #    `QScrollArea` 之后，工作台**退出时会偶发 0xC0000005**（access violation；
        #    实测 8/8 崩 ✗；把画布换成普通控件立刻 8/8 好 ✓）。
        #    ⇒ 版式定成「参数区滚动、地形图常驻」（`card(..., into=self.layout())` ✓）。
        chain, up = [], p.canvas.parentWidget()
        while up is not None and up is not p:
            chain.append(type(up).__name__)
            up = up.parentWidget()
        check(not any(n.endswith("ScrollArea") for n in chain),
              "「地形图」的画布被放进了滚动区（退出会偶发崩 ✗）：%s" % chain)

        # ⑦ ⭐ 用户 2026-09-27："寻路配置的地形图……**太占位置了**" ⇒ 版式必须是
        #   **可拖动的两栏**：上栏 = 参数区（滚动 ✓）、下栏 = 地形图（常驻 ✓）——
        #   拖分隔条就能决定它占多高（项目口径 `QSplitter` + `setChildrenCollapsible(False)` ✓）。
        #   ⚠ 顺带钉住那个**踩过的坑**：`card(..., into=...)` 里**不许**写 `(into or page)`
        #     —— PyQt 的**空布局是"假"的**（`QLayout` 有 `__len__`）⇒ 下栏那个空布局会被
        #     当成 falsy、卡片又落回参数页（画布就进了滚动区 ✗，正是上面 ⑥ 那条抓出来的 ✓）。
        from PyQt5.QtWidgets import QSplitter

        sp = p.findChild(QSplitter)
        check(sp is not None and sp.count() == 2,
              "「地形图」那一栏没做成可拖动的分割（用户报的「太占位置」✗）：%r" % (sp,))
        check(not sp.childrenCollapsible(),
              "分割条被允许拖成 0（UI规范 §4 要求 `setChildrenCollapsible(False)` ✗）")
        check(sp.widget(0) is sc and sp.widget(1) is p.maplayout.parentWidget(),
              "两栏内容不对（上 = 参数区滚动区 / 下 = 地形图 ✓）：%r"
              % ([type(sp.widget(i)).__name__ for i in range(sp.count())],))
    finally:
        p.close()


def t_screen_to_world():
    """`screen_to_world`：**游戏画面坐标 → 世界坐标**（用玩家做锚 ✓）。

    用户 2026-09-27 要"chase 判怪位于的 foothold 集合"⇒ 而**怪只有画面坐标** ✗
    ⇒ 需要这一步（全仓唯一一处 ✓，`perception/minimap.py`）。

    口径（`docs/寻路设计.md`：`屏幕 = 世界 − Camera`，**没有比例尺** ⇒ 前提是画面与世界
    1:1 ✓）：`Camera = 玩家世界坐标 − 玩家画面坐标`；x 用**框中心**、y 用**框底**
    （`world_y` 是黄点脚底 ✓、`player.bottom` 也是框底 ✓ —— 两边口径必须一致 ✓）。

    钉三件：① 玩家自己那一点要原样换回来；② 有偏移时按「世界 = 画面 + 相机」换算；
    ③ 拿不到玩家世界坐标 ⇒ 返回 `None`（**不猜** ✓，不许拿 0 当坐标 ✗）。
    """
    from perception.minimap import screen_to_world
    from perception.world_state import Player

    p = Player(x=400.0, y=300.0, bottom=340.0, w=40.0, h=60.0,
               world_x=1000.0, world_y=-200.0)
    # 相机 = (1000-400, -200-340) = (600, -540)
    check(screen_to_world(p, 400.0, 340.0) == (1000.0, -200.0),
          "玩家自己那一点该原样换回来：%r" % (screen_to_world(p, 400.0, 340.0),))
    check(screen_to_world(p, 500.0, 300.0) == (1100.0, -240.0),
          "没按「世界 = 画面 + 相机」换算：%r" % (screen_to_world(p, 500.0, 300.0),))
    check(screen_to_world(Player(x=400.0, y=300.0, bottom=340.0),
                          500.0, 340.0) is None,
          "拿不到玩家世界坐标时该返回 None（拿 0 当坐标会把人引到地图角落 ✗）")


def t_mob_layer_direction_check():
    """**方向自洽校验**：怪在画面上比玩家高 ⇒ 它那一层必须在玩家层**上方** ✓
    （用户 2026-09-28 思路里**真正有效**的那半："**至少顶层判定成底层这种离谱的向量都反了的
    应该能及时发现**"✓；另一半"y 投影 = 高度差"是**恒等式**、校不出"框底 ≠ 脚底"✗）。

    为什么要有它：`_resolve` 的"挑最近面"**根本不管方向** ✗ ⇒ 玩家在**一楼**、怪在**顶层**
    （画面上明明高一大截）时，只要"更下面那层离框底更近"就会判成**底层** ✗ ——
    **符号都反了**。⇒ 挑完（含各级兜底）后校验一次：方向反 ⇒ **丢** ⇒ `sets` 空 ⇒
    **不判**（照旧追 ✓ 宁缺勿错 ✓）。

    钉四件（`player` 摆成 `bottom=200 / world_y=0` ⇒ `cam_y = -200` ⇒ 怪框底世界 y = `mob.y+30-200`）：
      ① 怪**更高**（框底 −70）、挑中的面也**更高**（−300）⇒ 照常给集合 ✓（不许误伤 ✗）；
      ② 怪**更高**、挑中的面却**更低**（+400，离谱）⇒ **拒判**（`sets` 空 ✓）；
      ③ 怪**更低**（框底 +80）、面也**更低**（+400）⇒ 照常 ✓；
      ④ **同高**（框底 0、面 0）⇒ 不算"反" ⇒ 照常 ✓（`0 * x = 0` ✓）。
    """
    import types

    from gui import live_thread as lt

    def _probe(fy, mob_y, mob_h=60.0):
        class _T:
            footholds = ()

            def foothold_below(self, x, y, above_tol=None, band=None):
                return types.SimpleNamespace(fid="9", y_at=lambda _x: fy)

        class _Z:
            def set_of(self, fid):
                return ["某层"]

        th = types.SimpleNamespace(
            _mmap_mid="假图", _mob_queries={},
            agent=types.SimpleNamespace(_zone_cd_s=lambda zone=None: 3.0))
        th._route_ctx = lambda m: (_T(), _Z())
        th._mob_query_ttl = lambda: 3.0
        th._mark_mob_query = lambda mob, names, why="": None
        res = lt.LiveThread._make_mob_sets_resolver(th)
        # ⚠ 假 player **必须带 `y`**（框中心 ✓）—— 校验现在用"**画面上的框中心**"判上下
        #   （2026-09-28 修 ✗：原来用"框底"，符号会乱 ⇒ 误杀 ⇒ 现场"全都没找到" ✗）。
        #   少了它 `getattr(...,"y",0.0)=0` ⇒ 会被读成"怪在画面下方" ⇒ ① 直接红 ✓（踩过 ✓）。
        player = types.SimpleNamespace(x=100.0, y=300.0, bottom=400.0,
                                       world_x=0.0, world_y=0.0)
        mob = types.SimpleNamespace(id=1, x=100.0, y=mob_y, w=40.0, h=mob_h)
        return res(player, mob)

    for tag, fy, mob_y, want in (
            ("① 怪更高·面也更高", -300.0, 100.0, True),
            ("② 怪更高·面却更低（离谱）", 400.0, 100.0, False),
            ("③ 怪更低·面也更低", 400.0, 450.0, True),
            ("④ 同高", 0.0, 170.0, True),
            # ⑤ ⭐ **中心太近就不校验**（`MOB_DIR_MIN_PX` ✓ 2026-09-28 加）：
            #    同一层的怪中心本来只差几像素，硬判符号只会**误杀** ✓。
            #    ⚠ 构造要**真的会被误杀**：中心差 **+3**（>0 = 画面下方）而世界那侧在**上方**
            #      （`fy = -300` ⇒ 乘积 `< 0` ✓）⇒ 有护栏 ⇒ **跳过校验、照常给** ✓；
            #      没护栏 ⇒ 被拒 ⇒ 这条当场红 ✓（第一版没这么摆，反向验证时**没红** ✓ 踩过 ✗）。
            ("⑤ 中心只差 3px ⇒ 不校验", -300.0, 303.0, True)):
        info = _probe(fy, mob_y)
        got = bool(info.get("sets"))
        check(got == want,
              "%s ⇒ %s（该 %s）—— 方向反了就该**拒判**、别乱下前往 ✗：%r"
              % (tag, "给了集合" if got else "拒判",
                 "给集合" if want else "拒判", info))

    # ⑥ ⭐ **阈值由框自己定**（`max(MOB_DIR_MIN_PX, 框高×0.5)` ✓ —— 用户 2026-09-28 质疑
    #    「**怪框的 4 条边长都会有误差，凭什么认为框中心「不受框底偏差影响」？**」✓ 之后改的 ✓）。
    #    ⚠ 他的质疑是对的：`框中心 = (框顶+框底)/2` ⇒ 上下沿有误差中心就有误差 ✗（"不受影响"
    #      那个说法**不准确** ✓）。真正的差别是**敏感度**：旧方案拿"**换算出的世界 y**"当参照
    #      （和被校验的量**同一套坐标** ⇒ 误差直接进符号 ⇒ 超过**半层**就翻 ✗）；这条只回答
    #      "**谁在上面**"，跨层的画面高度差是**几百像素** ⇒ 上下沿各差几十像素不改结论 ✓。
    #    ⇒ 阈值取"**框高的一半**"= 这个框自身的固有不确定度 ✓，只有**明显跨层**才校验 ✓。
    #    这一件：**大框** `h=400`（阈值 ⇒ 200）时中心差 150 **还不算明显** ⇒ **不校验** ⇒ 照常给 ✓；
    #    ⚠ 同一份数据若按小框 `h=60`（阈值 30）⇒ 150 就该校验 ⇒ 会被拒 ✗（阈值必须随框变 ✓）。
    info6 = _probe(-300.0, 450.0, mob_h=400.0)
    check(bool(info6.get("sets")),
          "大框的阈值没跟着框高放大（中心差 150 在 h=400 时不该算「明显跨层」✗ —— "
          "那是**误杀**，正是「全都没找到」那类 bug ✓）：%r" % (info6,))


def t_player_tracker_world_anchor():
    """用小地图的**玩家世界坐标**辅助玩家框防抖（用户 2026-09-27 提的）。

    黄点是**独立于 YOLO 的第二个传感器** ⇒ 挑"哪个框是我"时，除了**像素连续性**（离
    "上一帧位置 + 速度外推"最近），还能问"**它现在到底在哪**"：候选按**上一帧**的相机
    换算成世界坐标，离**本帧黄点**最近的那个才是「我」✓（外推被误导时能把选择拉回来 ✓）。

    ⚠ 必须"**本帧黄点 + 上一帧的框**"：同一帧的 world 与 box 相减会退化成"离上一帧框
      多远"（= 老判据），等于没加 ✗ ⇒ `gui/live_thread.py` 里黄点定位被提到**挑框之前** ✓。

    钉四件：① 第一拍没有上一帧相机 ⇒ 老口径（取置信度最高）且照常锁上 ✓；
    ② 外推被误导那一拍：老口径选"更贴外推位置"的 ✗、世界口径选"黄点对得上"的 ✓；
    ③ 不给 `world` ⇒ 完全老行为 ✓；④ 中途不给 ⇒ 退回老判据、**锁不丢** ✓。
    """
    from perception.tracker import PlayerTracker

    def _two():
        """两套一样的追踪器：一帧静止 → 一帧 +40（让"速度外推"开始起作用 ✓）。"""
        a, b = PlayerTracker(), PlayerTracker()
        for t in (a, b):
            t.update([(380.0, 260.0, 420.0, 340.0, 0.90)], world=(1000.0, -200.0))
            t.update([(420.0, 260.0, 460.0, 340.0, 0.90)], world=(1040.0, -200.0))
        return a, b

    # ① 第一拍 ⇒ 老口径（取置信度最高 ⇒ cx=320），照常锁上
    b0 = PlayerTracker().update([(300.0, 260.0, 340.0, 340.0, 0.95),
                                 (380.0, 260.0, 420.0, 340.0, 0.80)],
                                world=(1000.0, -200.0))
    check(b0 is not None and abs(b0[0] - 320.0) < 1e-6,
          "第一拍该走老口径（取置信度最高 ⇒ cx=320）：%r" % (b0,))

    # ② 外推被误导：更贴外推位置的是误检，真正是我的是黄点对得上的那个
    cands = [(480.0, 260.0, 520.0, 340.0, 0.95),      # cx=500：贴外推位置（老口径会选 ✗）
             (380.0, 260.0, 420.0, 340.0, 0.85)]      # cx=400：黄点说我又回到世界 1000 ✓
    t_old, t_new = _two()
    b_old = t_old.update(cands)
    check(abs(b_old[0] - 500.0) < 1e-6,
          "不给 world 时该走老口径（离外推位置最近 = 500）：%r" % (b_old,))
    b_new = t_new.update(cands, world=(1000.0, -200.0))
    check(abs(b_new[0] - 400.0) < 1e-6,
          "给了世界坐标时该选**对得上黄点**的那个（400），而不是外推最近的那个（500）：%r"
          % (b_new,))

    # ③ 不给 world ⇒ 老行为一字不差
    t3, _ = _two()
    b3 = t3.update(cands)
    check(abs(b3[0] - 500.0) < 1e-6, "不给 world 时行为该一字不差：%r" % (b3,))

    # ④ 中途不给 ⇒ 退回老判据、**锁不丢**
    t4, _ = _two()
    t4.update(cands, world=(1000.0, -200.0))
    check(t4.update([(380.0, 260.0, 420.0, 340.0, 0.9)]) is not None,
          "中途不给 world 那一拍把锁丢了 ✗")


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


def t_route_resolver_start_from_player_sets():
    """给 agent 的**路径解析器**：起点取"我现在站哪个集合"（`_player_here`）。

    2026-09-26 实锤的 bug：`_fill_route_ctx` **没有**把 `here_sets` 留一份给解析器
    （`self._player_here` 全文件只有"读"、没有"写" ✗）⇒ 解析器只好退回
    `settings.route_goto_set`（那是**路线面板在跟踪玩家时**才写的东西 ✗）⇒
    「定点休息」经常报"你现在站的这块没圈进任何集合" ⇒ **角色根本不过去** ✗
    （用户在右下点手动休息、原地不动，就是这么来的）。
    """
    import types

    from gui import live_thread as lt

    # 用**假地形 + 假集合**：这条钉的是**接线**（`_fill_route_ctx` 要把 here_sets 留给
    # 解析器），喂一对最小的替身就够 —— 真地图 / 真集合那条路由 `t_goto_*` 那几条覆盖 ✓。
    class _Z:
        sets = {"甲平台": {"footholds": ["1"]}, "乙平台": {"footholds": ["2"]}}
        edges = []

        def set_of(self, fid):
            return ["甲平台"] if str(fid) == "1" else []

    class _T:
        def ladder_at(self, x, y):
            return None

    # 借 LiveThread 的真方法跑（不建线程、不碰设备）：只喂它要的那几个属性
    # ⚠ 这里**不**要写 `ROUTE_CTX_TTL_S`：那是 `LiveThread.__init__` 里的**实例属性**，
    #   不是模块常量（写成 `lt.ROUTE_CTX_TTL_S` ⇒ AttributeError ✗）；而且下面的
    #   `_route_ctx` 是替身，压根不读它 ✓。
    th = types.SimpleNamespace(_route_cache={}, _mmap_mid="假图", _player_here=[],
                               _pos_state=lt.PositionStateMachine())
    th._route_ctx = lambda m: (_T(), _Z())
    th._fill_route_ctx = lambda pl, loc: lt.LiveThread._fill_route_ctx(th, pl, loc)
    stg = types.SimpleNamespace(align_tol_px=6, align_hold_ms=250, route_goto_set="")

    class _P:
        world_x = 100.0
        world_y = -200.0
        here_sets = []
        ladder_id = None

    pl = _P()
    th._fill_route_ctx(pl, {"foothold_id": "1"})
    check(th._player_here == ["甲平台"],
          "`_fill_route_ctx` 没给解析器留一份「我现在站哪个集合」：%r"
          % (th._player_here,))

    # 行为判据：解析器**不许**再说"起点不知道"（那正是用户碰上的那条 ✗）
    res = lt.LiveThread._make_route_resolver(th, stg)("乙平台")
    check("没圈进任何集合" not in (res.get("why") or ""),
          "解析器还说起点不知道（没吃到 `_player_here`）：%s" % (res,))

    # ⭐⭐ 定位没有输出时：**沿用上一次**（用户 2026-09-28 **翻转**了这条口径 ✗✗ —— 原话：
    #   "位置状态给容错：**不许存在没站在平台上这种空类**，如果找不到就**按上一个位置状态**"✓）。
    #   ⚠ 为什么必须翻（现场事故 ✓）：清空 ⇒ `here_sets` 空 ⇒ 「战斗区域」判成"不在能打区"
    #     ⇒ 每拍下「回能打区」⇒ 而那条路又造不出来（`task_plan_fail`）⇒ **每 3 秒重试、卡死** ✗✗。
    #   ⚠ 换图仍要清（见 `PositionStateMachine.reset()` + `live_thread` 里那段 ✓）—— 那条没翻 ✗。
    th._fill_route_ctx(_P(), None)
    check(th._player_here == ["甲平台"],
          "定位没输出时**没沿用**上一次的 `here_sets`（用户 2026-09-28：「如果找不到就按上一个"
          "位置状态」✗ —— 清空会让「回能打区」陷进死循环 ✓）：%r" % (th._player_here,))

    # ⭐⭐ **但"沿用"只能在【同一张图内】**（2026-09-28 **现场事故**修 ✗✗ —— 用户报
    #   "**右下到左上的寻路报错了**"✓）：
    #   现场链路（`behavior.log` + 项目配置全对上 ✓）：用户**中途切过图**
    #   （`task_begin dst=一楼/二楼/三楼/底层` = **106010105** 的集合名 ✓，而 `mob_fh` 一直报
    #     `mid=105090600` ✓）⇒ 上一版把"换图检测"写在**"定位没输出"那条早退之后** ✗
    #     ⇒ **那条路根本走不到它** ⇒ `_player_here` 里留着**上一张图的集合名「小平台」** ✗✗
    #   ⇒ 择路拿它当**起点** ⇒ `core/zones.py` 报 `集合不存在：小平台`
    #     （那句的判据是 `src if src not in zones.sets else dst` ✓ ⇒ 说明报的是 **src = 起点** ✓）
    #   ⇒「右下 → 左上」**永远造不出路线** ⇒ `zone_goto_fail` 每 3 秒重试 ⇒ **卡死** ✗✗✗。
    #   ⇒ 钉两件：① **换图必须清**（机器那份 + `_player_here` + `_player_at`，**三样一起** ✗）；
    #            ② **检测必须在函数最前**（在"定位没输出"那条早退**之前** ✓ —— 顺序就是本 bug 的根 ✗）。
    _src = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    _i_mid = _src.index("if mid != getattr(self, \"_pos_state_mid\", None):")
    _i_early = _src.index("if x is None or y is None or not mid:")
    check(_i_mid < _i_early,
          "「换图检测」被放在**「定位没输出」早退之后**了 ✗✗ —— 那条路走不到它 ⇒ 换图后"
          "起点集合会跨图残留 ⇒ 报「集合不存在：<上一张图的集合名>」⇒ 寻路永远失败（现场事故 ✓）")
    _seg = _src[_i_mid:_i_mid + 400]
    check("self._pos_state.reset()" in _seg and "self._player_here = []" in _seg
          and "self._player_at = None" in _seg,
          "换图时**没把三样一起清**（机器那份 / `_player_here` / `_player_at` ✗ —— "
          "少清 `_player_here` 就是现场那个 bug ✓）：%r" % (_seg[:200],))


def t_fill_route_ctx_hold_vert_wired():
    """`ladder_id` 的许可（`hold_vert`）**真的接到了执行器** —— 2026-09-28 修的接线 bug。

    现场（用户 2026-09-28）："**我感觉不是 LADDER_PAD 的问题，因为之前没有限制绳梯判定
    入口的时候是很准确及时的**" ⇒ 一查：`_fill_route_ctx` 原来写成
    `getattr(self, "agent", None)`，而 `self.agent` **从来没人赋过值** ✗ ⇒ `_hold_vert`
    恒 `False` ⇒ `pos_state.ladder_id` **恒 `None`** ⇒ 执行器永远判不出"在绳上"
    （爬绳全废、"位置状态判定失误" ✗ —— 日志里就是"对齐好了 ⇒ 2s 没回到绳上 ⇒ 掉下来了"✗）；
    顺带 `_mob_query_ttl` 也恒走 except ⇒ 设置里「前往重下间隔(s)」实际不生效 ✗。

    钉四件：
      ① 真图真绳：`hold_vert=True` ⇒ `ladder_id` **出得来**（许可通了 ✓）；
      ② 同一位置 `hold_vert=False` ⇒ `ladder_id=None`（"走路碰到绳子不被粘住"那条防线
         **还在** ✓ —— 用户 2026-09-27 要的 ✓），但 `on_rope_pos` 照给（**只看位置**那份
         不受许可影响 ✓）；
      ③ **源码级**：`run()` 里必须把 agent 挂到 `self.agent`（`_fill_route_ctx` 与
         `_mob_query_ttl` 两处都读它）；
      ④ **源码级**：调用点必须把 `agent.climbing_vertical()` 传进去；`_fill_route_ctx`
         **不许**再自己 `getattr(self, "agent", None)` 找 ✗（就是它把许可掐成恒 False 的）。
    """
    import inspect
    import types

    from core import mapdata
    from core import zones as zones_mod
    from gui import live_thread as lt

    # ---- ①② 行为：真图 + 真绳（找第一张带绳、且编得出号的图 ✓）----
    pick = None
    for q in sorted(mapdata.map_dir().glob("*.png")):
        if q.stem.endswith(("_overlay", "_zones")):
            continue
        try:
            _t = mapdata.load(q.stem, with_canvas=False)
        except Exception:                       # noqa: BLE001 —— 读不动的图跳过 ✓
            continue
        _ids = zones_mod.ladder_ids(_t) or {}
        if getattr(_t, "ladders", None) and _ids:
            pick = (q.stem, _t, _ids)
            break
    if pick is None:
        print("      （没有带绳的图，跳过行为那两条）")
    else:
        _mid, _t, _ids = pick
        _lid = sorted(_ids.values())[0]
        _L = [x for x in _t.ladders if _ids.get(id(x)) == _lid][0]

        class _Z:
            def set_of(self, fid):
                return []

        th = types.SimpleNamespace(_route_cache={}, _mmap_mid=_mid, _player_here=[],
                                   _pos_state=lt.PositionStateMachine())
        th._route_ctx = lambda m: (_t, _Z())
        th._fill_route_ctx = lambda pl, loc, hold_vert=False: \
            lt.LiveThread._fill_route_ctx(th, pl, loc, hold_vert)

        class _P:
            here_sets = []
            here_span = None
            world_x = 0.0
            world_y = 0.0
            ladder_id = None
            on_rope_pos = None
            at_ladder_top = None
            at_ladder_bottom = None
            ground_y = None

        _ymid = (min(_L.y1, _L.y2) + max(_L.y1, _L.y2)) / 2.0

        p_on = _P()
        p_on.world_x, p_on.world_y = float(_L.x), _ymid
        th._fill_route_ctx(p_on, {"foothold_id": ""}, True)
        check(str(p_on.ladder_id) == _lid,
              "按着 ↑/↓（许可开）站在绳上，`ladder_id` 还是判不出来（%r，该 %r）—— "
              "位置状态对绳的判定等于全废 ✗" % (p_on.ladder_id, _lid))

        p_off = _P()
        p_off.world_x, p_off.world_y = float(_L.x), _ymid
        th._fill_route_ctx(p_off, {"foothold_id": ""}, False)
        check(p_off.ladder_id is None,
              "没按 ↑/↓ 却还是判成在绳上（走路路过绳口会被粘住 ✗ —— 用户 2026-09-27 的"
              "那条防线不许破）：%r" % (p_off.ladder_id,))
        check(str(p_off.on_rope_pos) == _lid,
              "`on_rope_pos`（**只看位置**那份）该与许可无关、照给：%r" % (p_off.on_rope_pos,))

    # ---- ③④ 源码级：接线别哪天又断 ----
    src = (ROOT / "gui" / "live_thread.py").read_text(encoding="utf-8")
    check("self.agent = agent" in src,
          "`run()` 里没把 agent 挂到 `self.agent` ⇒ `_fill_route_ctx` 拿不到执行器通知、"
          "`ladder_id` 恒 None ✗（2026-09-28 报的那条）")
    check("agent.climbing_vertical()" in src,
          "调用点没把执行器通知（`agent.climbing_vertical()`）传给 `_fill_route_ctx` ✗")
    _fn = inspect.getsource(lt.LiveThread._fill_route_ctx)
    # ⚠ 只查**代码行**：注释里会提到那个写法（那正是"踩过的坑"的记录 ✓，别误伤）
    _code = "\n".join(_ln for _ln in _fn.splitlines()
                      if not _ln.lstrip().startswith("#"))
    check('getattr(self, "agent", None)' not in _code,
          "`_fill_route_ctx` 又在**自己**找 `self.agent` ✗ —— 那个属性一旦没人赋，"
          "许可就恒 False、`ladder_id` 恒 None（改成由调用方传 `hold_vert` ✓）")


def t_climb_failed_needs_stuck():
    """「攀爬失败」（`climb_failed`）**必须还要"僵着不动"** —— 不然**斜跳必被杀**（用户 2026-09-28）。

    现场（`behavior.log` 里 `route_diag` + `climb_*` 探针那条真数据）：

    ```
    01:49:02.571  斜跳上绳：离绳 -71 px（区间 20~80…）⇒ 按住 ← + 按跳   what=起跳
    01:49:02.861  climb_y v=-293.3   climb_dx v=21.0   climb_on_ladder v=1   ← 已在绳上、y 在升
    01:49:02.862  攀爬失败：x 偏出「坐标对齐误差范围」（位置状态广播）  → failed
    ```

    ⇒ 斜跳起跳后 **0.29s** 就被判死 ⇒ 用户报"**斜向跳上绳全部失败**" ✗（原地跳全好 ✓）。

    为什么：`climb_failed` 原来**只看两条** —— 「贴着绳（`LADDER_DX`=24 那把**宽**尺 ✓）」+
    「x 偏出「坐标对齐误差范围」（`tol`=10）」；而**斜跳飞过去时必然穿过 `10 < |dx| ≤ 24`
    这一段**（x 从 70 收到 0 ✓），那时人**已经吸上绳、y 正在上升** ⇒ 只看 x 就是"每跳必杀" ✗
    —— 与"算不算在绳上"那把**宽**尺口径打架（用户 2026-09-27 踩过的同款坑的翻版 ✓）。
    ⚠ 原地跳不中招：它跳之前已经对齐到 `|dx| ≤ tol` ✓。

    ⇒ 加"**僵着**"（y 在「移动操作尝试间隔」内一次都没动 ✓）：飞过 ⇒ 不算 ✓；贴歪磨绳 ⇒ 才算 ✓
      —— 用户 2026-09-28 要的"抓空 ⇒ 失败重来"一个字没丢 ✓。

    钉四件（真图真绳 + 真 `PositionStateMachine`）：
      ① dx=21（`tol` 10 < 21 ≤ `LADDER_DX` 24）+ y 在上升 ⇒ `climb_failed=False`（**用户那个现场** ✓）；
      ② 停在那个高度、还没到「移动操作尝试间隔」⇒ 仍 False ✓；
      ③ 僵着超过那个窗口 ⇒ 变 True ✓（贴歪磨绳才算抓空 ✓）；
      ④ 对照：**x 对齐**（dx=3）僵着 ⇒ `climb_failed=False`、只有 `climb_stalled=True` ✓（那支
         交给"补按 ↑"，不是失败 ✓）。
    """
    from core import mapdata
    from core import zones as zones_mod
    from perception import pos_state

    pick = None
    for q in sorted(mapdata.map_dir().glob("*.png")):
        if q.stem.endswith(("_overlay", "_zones")):
            continue
        try:
            _t = mapdata.load(q.stem, with_canvas=False)
        except Exception:                       # noqa: BLE001 —— 读不动的图跳过 ✓
            continue
        _ids = zones_mod.ladder_ids(_t) or {}
        if not (getattr(_t, "ladders", None) and _ids):
            continue
        _L = _t.ladders[0]
        _lid = _ids.get(id(_L))
        if _lid:                                # ⚠ 绳号编不出来 ⇒ 走"不在绳上"那支、测不到 ✓
            pick = (_t, _L, _lid)
            break
    if pick is None:
        print("      （没有带绳的图，跳过）")
        return
    _t, _L, _lid = pick
    _ymid = (min(_L.y1, _L.y2) + max(_L.y1, _L.y2)) / 2.0

    class _Z:
        def set_of(self, fid):
            return []

    # 「坐标对齐误差范围」= 10 / 「移动操作尝试间隔」= 3000ms（都是默认口径 ✓）
    _kw = dict(terrain=_t, zones=_Z(), hold_vert=True,
               align_tol_px=10, move_retry_ms=3000)

    def _loc(dx, dy=0.0):
        return {"world_x": float(_L.x) + dx, "world_y": _ymid + dy, "foothold_id": ""}

    # ① **斜跳飞过**：dx=21（10 < 21 ≤ 24）+ y 在上升 ⇒ 不许判抓空
    m = pos_state.PositionStateMachine()
    o = m.update(100.0, loc=_loc(21.0), **_kw)
    check(str(o.ladder_id) == _lid,
          "用例前提：dx=21 该算「在绳上」（`LADDER_DX`=24 那把宽尺 ✓）：%r" % (o.ladder_id,))
    o = m.update(100.1, loc=_loc(21.0, -30.0), **_kw)
    check(not o.climb_failed,
          "**斜跳飞过**（dx=21、y 正在上升）被判成「攀爬失败」⇒ 每一跳都被杀 ✗（用户 2026-09-28 "
          "报的「斜向跳上绳全部失败」）：%s" % (o.fmt(),))
    # ② 停在那个高度、还没到「移动操作尝试间隔」⇒ 仍不算
    o = m.update(101.0, loc=_loc(21.0, -30.0), **_kw)
    check(not o.climb_failed,
          "刚停下、还没到「移动操作尝试间隔」就判抓空（太快 ✗）：%s" % (o.fmt(),))
    # ③ 僵着超过窗口 ⇒ 这才是"贴着绳磨" ⇒ 算抓空 ✓（诉求保留）
    o = m.update(104.0, loc=_loc(21.0, -30.0), **_kw)
    check(o.climb_failed,
          "贴着绳 + x 偏 + **僵着超过「移动操作尝试间隔」**，却没判抓空（用户 2026-09-28 要的"
          "那条丢了 ✗）：%s" % (o.fmt(),))
    # ④ 对照：x 对齐（dx=3 ≤ tol）僵着 ⇒ 只"补按 ↑"、不算抓空 ✓
    m2 = pos_state.PositionStateMachine()
    m2.update(100.0, loc=_loc(3.0), **_kw)
    o2 = m2.update(104.0, loc=_loc(3.0), **_kw)
    check(not o2.climb_failed and o2.climb_stalled,
          "x 对齐了却判成抓空（那支该只「补按 ↑」）：failed=%s stalled=%s"
          % (o2.climb_failed, o2.climb_stalled))


def t_foothold_below_prefers_nearer_within_xtol():
    """「脚下是哪条 foothold」的 `xtol` 口径（用户 2026-09-28 报的**真现场**）。

    用户原话："位置解析出问题了，**(594,162) 坐标应该属于小平台**，位置状态机解析成了**底层**"。

    真数据（图 106010105）：
      · `fh=13`「小平台」面 y=155、x 范围 579~591 ⇒ 人的 x=594 在它右边界**外 3px**
        （定位抖动就这个量级 ⇒ 这正是 `xtol` 存在的理由 ✓）；
      · `fh=46`「底层」  面 y=185、x 范围 585~675 ⇒ **把 x=594 罩住了**（所以老口径判得出来 ✗）。
    老口径是"**只在 `dx=0` 一条都找不到时**才放宽"✗ ⇒ **「小平台」压根没参与比较** ⇒ 判给了
    23px 之下的「底层」✗ ⇒ `foothold_id` 错 ⇒ `here_sets` 错 ⇒ 归属集合 / 限制战斗区域 /
    起点判定**全跟着错** ✗。

    新口径：**两遍都比（dx=0 / dx=xtol），只有放宽后"更近"（严格 `<`）时才改判** ✓。钉三件：
      ① `(594,162)`：`xtol=0` ⇒ 仍是 46「底层」（**老结果一字不变** ✓）、`xtol=10` ⇒ 13「小平台」 ✓；
      ② **不许误伤**：人真站在「底层」上（600,185）⇒ `xtol=10` 也**还是 46** ✓；
      ③ 三个消费方**同一口径**：`segment_of` / `find_below` 都转调 `foothold_below` ⇒ 同一坐标
        给出同一答案 ✓（少一处就会出现"路线面板说在这块平台、起点判定说不知道"✗）。
    """
    from core import mapdata, zones as zones_mod

    mid = "106010105"
    if not (mapdata.map_dir() / ("%s.png" % mid)).exists():
        print("      （没有 %s 这张图，跳过）" % mid)
        return
    t = mapdata.load(mid, with_canvas=False)
    z = zones_mod.load(mid)

    def _sets(f):
        fid = getattr(f, "fid", None)
        return z.set_of(fid) if fid else []

    # ① 用户现场：老口径判「底层」，放宽后必须判「小平台」
    f0 = t.foothold_below(594.0, 162.0, xtol=0)
    check("小平台" not in _sets(f0),
          "用例前提变了：`xtol=0` 时 (594,162) 本来就判「小平台」了（这条现场不再成立）：%s"
          % (_sets(f0),))
    f1 = t.foothold_below(594.0, 162.0, xtol=10)
    check(str(getattr(f1, "fid", "")) == "13" and _sets(f1) == ["小平台"],
          "「小平台右边界外 3px」还是判不出来（放宽后的候选没参与比较 ✗）—— 现场 (594,162) "
          "会被判成 23px 之下的「底层」✗：%s %s" % (getattr(f1, "fid", None), _sets(f1)))
    # ② 不许误伤：人真站在「底层」上 ⇒ 放宽也不许改判
    f2 = t.foothold_below(600.0, 185.0, xtol=10)
    check(_sets(f2) == ["底层"],
          "人明明站在「底层」上，放宽 x 之后被抢到别的平台了（放宽该「只在更近时」才改判 ✗）："
          "%s %s" % (getattr(f2, "fid", None), _sets(f2)))
    # ③ 三个消费方同一口径（段 / 落点都转调 `foothold_below` ✓）
    seg = t.segment_of(594.0, 162.0, xtol=10)
    fb = t.find_below(594.0, 162.0, xtol=10)
    check(seg is not None and fb is not None,
          "`segment_of` / `find_below` 与 `foothold_below` 口径不一致"
          "（一个说在平台上、一个说没有 ⇒ 最难看的分歧 ✗）")
    check(str(getattr(f1, "fid", "")) in [str(getattr(x, "fid", "")) for x in seg.footholds],
          "`segment_of` 给的段里没有 `foothold_below` 判出的那条（两处口径打架 ✗）")

    # ④ ⭐ **`above_tol`**（2026-09-28 加，专给**怪**用 ✓）：当天现场"**查#171 一个二楼的怪
    #    显示在底层，现在已经卡住了**"✗。给怪算的是"**怪框底**换算的世界 y"（`live_thread` ✓），
    #    而**大怪的检测框并不包住脚** —— 实测那只二楼石人，框底比它真正站的二楼面**高 84px** ✗
    #    ⇒ 老口径"面最多比 y 高 40"**把二楼那条面直接跳过** ⇒ 只剩"下方最近"的**一楼** ⇒
    #    判在下一层 ✗（`mob_goto_ok 已出发：一楼 → 底层` / `底层 → 一楼` 来回卡死 ⟲）。
    #    真数据（图 106010105，x=896 附近）：三楼 -178 / **二楼 -84** / 一楼 95 / 底层 185。
    a0 = t.foothold_below(896.0, 0.0, xtol=10)
    check(_sets(a0) == ["一楼"],
          "用例前提变了：老口径下 (896,0) 本来就判「二楼」了（这条现场不再成立）：%s"
          % (_sets(a0),))
    a1 = t.foothold_below(896.0, 0.0, xtol=10, above_tol=200.0)
    check(_sets(a1) == ["二楼"],
          "**二楼的怪还是被判到下一层**（面在框底上方 84px 就被跳过 ✗）—— 现场 #171 "
          "会来回下「前往一楼 / 底层」✗：%s %s"
          % (getattr(a1, "fid", None), _sets(a1)))
    # ⑤ 放宽**不许误伤**：怪真站在一楼上（框底=脚 ⇒ y≈95）⇒ 照旧「一楼」✓
    a2 = t.foothold_below(850.0, 95.0, xtol=10, above_tol=200.0)
    check(_sets(a2) == ["一楼"],
          "怪真站在「一楼」上却被抢到别的层了（放宽该只在「更近」时生效 ✗）：%s %s"
          % (getattr(a2, "fid", None), _sets(a2)))

    # ⑥ ⭐⭐ **`band`（给怪用的定稿口径）**：用户 2026-09-28 又报"**又把顶层怪判成底层怪了**"✗。
    #    要害是**层距**：这张图 105040303 的层间隔 **540**（顶层 375 / 三楼 915 / 二楼 1455 /
    #    底层 1995 ✓）⇒ 上面那条固定容差（`above_tol=200`）**连半层都盖不住** ✗；
    #    而且只要"框底算出来的 y"偏过半层，"**取最近**"就**必然**挑到隔壁那层 ✗
    #    （实测：框底偏差 300 ⇒ 固定容差给「三楼」✗）。
    #    ⇒ 口径换成"**候选面 = 怪框覆盖的那一段 y**（框顶 ~ 框底 ✓）"：怪**一定站在穿过
    #      它身体的那条面**上 ✓ —— 框底到底偏了多少**不用猜** ✓。
    if (mapdata.map_dir() / "105040303.png").exists():
        t2 = mapdata.load("105040303", with_canvas=False)
        z2 = zones_mod.load("105040303")
        X2 = 135.0

        def _sets2(f):
            return z2.set_of(str(f.fid)) if f is not None else None

        # 真值锚：x=135 这条竖线上是 顶层 375 / 三楼 915 / 二楼 1455 / 底层 1995 ✓
        check(_sets2(t2.foothold_below(X2, 375.0)) == ["顶层"],
              "用例前提变了：x=135 上 y=375 不是「顶层」")
        # ① **框覆盖段**（框高 300、框底算到 675 ⇒ 区间 (375, 675)）⇒ 顶层 ✓
        b1 = t2.foothold_below(X2, 675.0, band=(375.0, 675.0))
        check(_sets2(b1) == ["顶层"],
              "「框覆盖段」里没挑到顶层（这就是现场「顶层怪被判成底层」那条 ✗）：%s"
              % (_sets2(b1),))
        # ② 对照：同一个框底、改用**老的「框底 ± 框高」**⇒ 会挑到**更近的三楼** ✗
        #    （证明"固定容差 + 取最近"这条路必然崩 ✓ 别改回去 ✗）
        b2 = t2.foothold_below(X2, 675.0, band=(675.0 - 300.0, 675.0 + 300.0))
        check(_sets2(b2) == ["三楼"],
              "对照组前提变了（老口径本该挑到更近的「三楼」）：%s" % (_sets2(b2),))
        # ③ 兜底那一级（**只往上**放半框高）⇒ 仍然顶层 ✓（⛔ 往**下**放宽会挑错 ✗，
        #    见 `live_thread` 里那两行注释 ✓）
        b3 = t2.foothold_below(X2, 675.0, band=(375.0 - 150.0, 675.0))
        check(_sets2(b3) == ["顶层"], "往上放半框高那级没挑到顶层：%s" % (_sets2(b3),))
        # ④ 区间里一条面都没有 ⇒ 回 `None`（**宁可没找到、不要判错** ✓ —— 没找到 =
        #    不判 = 照旧追；判错 = 乱下前往、来回跑 ✗）
        b4 = t2.foothold_below(X2, 2000.0, band=(1000.0, 1200.0))
        check(b4 is None, "区间里没有面却挑了一条出来（该回 None ✓）：%r" % (b4,))

        # ⑦ ⭐⭐ **用户的「向下垂线」方案**（2026-09-28 用户原话："用 <玩家脚底到怪框**中心**的
        #    向量>在世界坐标对应到**怪框中心**的世界坐标后，**向下做垂线**，接触到的第一个
        #    foothold 集合就是该怪的"✓）—— 它是 `band` 找不到面时的**兜底** ✓（`live_thread`
        #    里那一级 ② 就是它 ✓）。
        #    真数据（同一张图、同一条 x）：框**整个在面上方**（框顶 100 / 框底 300）
        #    ⇒「顶层」375 **不在** `band=(100,300)` 里 ⇒ `band` 给 `None` ✗（"全都没找到"的一种 ✓）；
        #    而**从框中心（200）往下第一条**正好是「顶层」✓。
        t3 = mapdata.load("105040303", with_canvas=False)
        z3 = zones_mod.load("105040303")

        def _sets3(f):
            return z3.set_of(str(f.fid)) if f is not None else None

        X3 = 135.0
        check(t3.foothold_below(X3, 200.0, band=(100.0, 300.0)) is None,
              "用例前提变了：框整个在面上方时 `band` 本该找不到")
        d1 = t3.foothold_below(X3, 200.0, above_tol=0.0)     # ← 向下垂线（往下第一条 ✓）
        check(_sets3(d1) == ["顶层"],
              "「向下垂线」没挑到顶层（用户 2026-09-28 那条方案 ✗）：%s" % (_sets3(d1),))
        # ⑧ 它的**上界**判据（`live_thread` 里 `y_at ≤ hi + h/2` ✓）：框底 300 + 半框 100 = 400
        d2 = t3.foothold_below(X3, 1200.0, above_tol=0.0)    # 框中心 1200 ⇒ 第一条是 1455「二楼」
        check(d2 is not None and float(d2.y_at(X3)) > 1300.0 + 50.0,
              "用例前提变了：这一段该属于「明显跑远」那一档（框底 1300 + 半框 50 = 1350 ✗）")


def t_mob_fh_observability():
    """「怪判不出集合」时**必须一眼看出卡在哪**（用户 2026-09-28："**又开始全程找不到怪的
    foothold 集合了，这不应该**"✗）。

    那轮排查**全靠猜**：`behavior.log` 里只有一句"怪底下没找到 foothold" ✗ ——
    **用的哪张图没有**、**玩家世界 x 没有**、**四级挑面里卡在哪一级也没有** ✗
    （最后是把能想到的图全试了一遍，才排掉"图不对"这一档 ✓）。

    钉四件：
      ① ⛔ **相机不许再做平滑**（`_cam_smooth` 必须**恒等于** `world_y − bottom` ✓）——
         撤掉 EMA 的理由与实测见 `_make_mob_sets_resolver` 里那段（实测偏 **47~362 像素**、
         挑面命中率 **38% → 38%**、一点没提高 ⇒ 纯有害 ✓）。这一件**故意先塞一个残留的
         平滑值** ⇒ 它必须被覆盖成原始相机 ✓；
      ② `why` **分档**：挑面各级都空时，要说清**卡到第几级** ✓（这里两级全空 ⇒ 该报 ②；
         ⚠ **原来那级「再往上放半个框高」已删** —— 用户 2026-09-28："这一步没有必要，去掉"✓）；
      ③ `mob_fh` 打点带上 `mid`（哪张图 ✓）/ `ply_wx` / `ply_cx` / `ply_bot` ✓
         （**源码级**钉 ✗：这个文件里没有 `Harness`，跑出来的 `behavior` 事件不好断言 ✓）；
      ④ **真数据**跑一次 ⇒ 相机与 `world_y − bottom` **逐位一致** ✓（防"平滑"换皮回来 ✗）。
    """
    import types

    from core import mapdata, zones
    from gui import live_thread as lt

    class _Z:
        def set_of(self, fid):
            return ["甲平台"]

    class _TNone:
        """四级都挑不到面。⚠ 签名要跟上真类（`above_tol` / `band` / `xtol` ✓ 少一个就 TypeError ✗）。"""

        def foothold_below(self, x, y, above_tol=None, band=None, xtol=0):
            return None

    ply = types.SimpleNamespace(x=100.0, y=200.0, bottom=260.0,
                               world_x=10.0, world_y=140.0)
    mob = types.SimpleNamespace(id=7, x=100.0, y=180.0, w=40.0, h=60.0)

    # ①④ 真数据：相机 = `world_y − bottom`（**不是**平滑值 ✓）
    t, z = mapdata.load("105090600"), zones.load("105090600")
    check(t is not None and z is not None, "那张真地图读不出来（前提不成立）")
    th = types.SimpleNamespace(_mmap_mid="105090600", _cam_smooth=999.0)   # 塞个残留值 ✓
    th._route_ctx = lambda m: (t, z)
    th._mark_mob_query = lambda mob, names, why="": None
    lt.LiveThread._make_mob_sets_resolver(th)(ply, mob)
    check(abs(float(th._cam_smooth) - (140.0 - 260.0)) < 1e-6,
          "相机不是 `world_y − bottom`（又变成**平滑值**了 ✗ —— 实测它偏 47~362 像素、"
          "挑面命中率一点没提高 ⇒ 纯有害 ✓）：%r" % (th._cam_smooth,))

    # ② `why` 分档：两级全空 ⇒ 该说清"卡到 ②"（⚠ 原 ③「再往上放半框高」已删 ✓）
    th2 = types.SimpleNamespace(_mmap_mid="假图", _cam_smooth=None)
    th2._route_ctx = lambda m: (_TNone(), _Z())
    th2._mark_mob_query = lambda mob, names, why="": None
    info2 = lt.LiveThread._make_mob_sets_resolver(th2)(ply, mob)
    check(info2.get("sets") == [], "假地形没空掉（前提不成立）：%r" % (info2,))
    check("②" in str(info2.get("why") or ""),
          "挑面各级都空时 `why` 没说清**卡到第几级**（下次还得靠猜 ✗）：%r"
          % (info2.get("why"),))

    # ③ `mob_fh` 打点必须带上"哪张图 / 玩家世界 x / 画面 x / 框底"
    from pathlib import Path

    _src = (Path(__file__).resolve().parents[1] / "gui" / "live_thread.py"
            ).read_text(encoding="utf-8")
    check('"mob_fh"' in _src, "`mob_fh` 打点不见了（排查的主要依据 ✗）")
    for _k in ("mid=", "ply_wx=", "ply_cx=", "ply_bot="):
        check(_k in _src,
              "`mob_fh` 打点里没有 `%s`（那轮就缺它 ⇒ 排查全靠猜 ✗）" % _k)


def t_queried_mob_boxes_marked():
    """「**查过的怪框**」要标出来、**缓存失效再移除**（用户 2026-09-27："只要是**查询到的
    怪框地点**，就标出来，**缓存失效再移除**"）。

    钉五件：
      ① **查过**才记（入口是 `mob_sets_of` 那个**唯一漏斗** ✓ —— agent 那边一处都不用改 ✓）；
      ② 时效 = **缓存那一把尺**（`agent._goto_retry_s()` = 「前往重下间隔(s)」✓，它本来就是
        区域筛缓存 `_zone_cache` 的时效 ✓）—— 到点**自己消失**（读口顺手剪 ✓）；
      ③ 再查一次 ⇒ 时效**续上**（不是"查过一次就永远是旧时刻" ✗）；
      ④ ⭐ **"有 → 失败"时沿用上次有效集合 + 续期**（用户 2026-09-28："不要空"✓）；
        但**第一次就失败**（没上次可沿用）⇒ 仍照记成"空 + why"✓（那一档不能丢 ✗）；
        `why` 一律照记（给 `mob_fh` 那条 log 看 ✓ 画面与 log 两边都不丢 ✓）；
      ⑤ ⭐⭐ **账里只存"怪号 + 集合名 + 时刻"，不存框**（用户 2026-10-02 ✓ 原话：
        "能不能让他只是作为『怪物框的标记』，跟着怪物的检出框走"）—— 框由**绘制那一刻**
        按怪号去**这一帧的检出框**（`ws.mobs`）现查 ✓。原来存的是"**查询那一刻**的框"✗
        ⇒ 怪走开之后红框**留在原地**（用户现场："红框在原地残留，可读性极差"✗）。
        ⚠ 所以这条断言从"记的是画面框"改成"**账里没有框**"（画框那一侧按**源码**钉，见
        `selftest_live_panel.t_mob_query_label_placement` ✓ —— 那一段没有可驱动的最小夹具 ✓）。
    """
    import types
    from unittest import mock

    from gui import live_thread as lt

    th = types.SimpleNamespace(
        agent=types.SimpleNamespace(_zone_cd_s=lambda zone=None: 2.0), _mob_queries={})
    th._mob_query_ttl = lambda: lt.LiveThread._mob_query_ttl(th)
    # ② 时效取的是"缓存那把尺"（`agent._goto_retry_s()` ✓），不是写死的常量 ✓
    check(abs(th._mob_query_ttl() - 2.0) < 1e-9,
          "「查过的怪框」的时效没取缓存那把尺（该取 `agent._goto_retry_s()` ✓）：%r"
          % (th._mob_query_ttl(),))

    mob = types.SimpleNamespace(id=7, x=100.0, y=200.0, w=40.0, h=60.0)

    # ①' ⭐ **接线**：真跑一次那个解析器（`mob_sets_of` ✓ = 唯一漏斗）⇒ 必须**自动**记一笔 ✓
    #    （只测 `_mark_mob_query` 是不够的 —— 那样把"接线断了"漏掉 ✗，反向验证就是这么发现的 ✓）
    class _Z2:
        def set_of(self, fid):
            return ["甲平台"]

    class _T2:
        # ⚠ 签名要跟上真类（2026-09-28：真 `foothold_below` 多了 `above_tol` —— 给**怪**
        #   放宽"面可以比框底高多少"那个口子 ✓，见 `t_foothold_below_prefers_nearer_within_xtol`
        #   的 ④ ✓）：替身少一个参数 ⇒ 解析器一调就 TypeError ⇒ 这条用例红 ✗（踩过 ✓）。
        def foothold_below(self, x, y, above_tol=None, band=None):
            return types.SimpleNamespace(fid="1")

    th2 = types.SimpleNamespace(
        _mmap_mid="假图", _mob_queries={},
        agent=types.SimpleNamespace(_zone_cd_s=lambda zone=None: 2.0))
    th2._route_ctx = lambda m: (_T2(), _Z2())
    th2._mob_query_ttl = lambda: lt.LiveThread._mob_query_ttl(th2)
    # 替身 self 也得有那个记账入口（解析器内部是 `self._mark_mob_query(...)` ✓）
    th2._mark_mob_query = lambda mob, names, why="": lt.LiveThread._mark_mob_query(
        th2, mob, names, why)
    # ⚠ 2026-09-28：解析器改成**自己按（平滑后的）相机**算世界坐标（⛔ 不再走
    #   `screen_to_world` —— 那里用的是**未平滑**的 cam，会和怪那份打架 ✓）⇒
    #   假 player **必须带** `x / bottom / world_x / world_y`（少一个就早退 ⇒ `sets` 空
    #   ⇒ 这条红 ✓ **踩过** ✓；原来这里是 `mock.patch.object(mm, "screen_to_world", …)` ✗
    #   现在不起作用了 ✓）。
    _ply = types.SimpleNamespace(x=100.0, y=200.0, bottom=260.0,
                                 world_x=10.0, world_y=20.0)
    info = lt.LiveThread._make_mob_sets_resolver(th2)(_ply, mob)
    check(info.get("sets") == ["甲平台"], "假地形没给出集合（前提不成立）：%r" % (info,))
    got2 = lt.LiveThread.queried_mob_boxes(th2, now=time.monotonic() + 0.1)
    check(len(got2) == 1 and got2[0][0] == "7",
          "调了 `mob_sets_of`（唯一漏斗）却没自动记账（画面上就不会标 ✗）：%r" % (got2,))

    lt.LiveThread._mark_mob_query(th, mob, ["甲平台"], "", now=100.0)
    got = lt.LiveThread.queried_mob_boxes(th, now=100.5)
    check(len(got) == 1 and got[0][0] == "7",
          "查过的怪没被记下来（画面上就不会标 ✗）：%r" % (got,))
    # ⑤ ⭐⭐ 账里**不许再有框**（用户 2026-10-02 ✓）：有框 = 画的时候就会用旧位置 ✗
    _rec = got[0][1]
    check(len(_rec) == 4 and _rec[0] == ["甲平台"],
          "账的形状不是 `(集合名, why, 首查时刻, 失效时刻)`（多存了框 ⇒ 红框会留在原地 ✗）：%r"
          % (_rec,))
    check(not any(isinstance(_v, tuple) for _v in _rec),
          "账里还是塞了「查询那一刻的框」（标记就跟不住怪了 ✗）：%r" % (_rec,))

    # ② 到点就移除（"缓存失效再移除" ✓）；③ 再查一次 ⇒ 续上
    check(len(lt.LiveThread.queried_mob_boxes(th, now=101.9)) == 1,
          "还没到点就被移除（标记一闪就没 ✗）")
    check(len(lt.LiveThread.queried_mob_boxes(th, now=102.1)) == 0,
          "过了时效还留着（用户要求「缓存失效再移除」✗）")
    lt.LiveThread._mark_mob_query(th, mob, ["甲平台"], "", now=102.0)
    check(len(lt.LiveThread.queried_mob_boxes(th, now=103.9)) == 1,
          "再查一次没把时效续上（会提前消失 ✗）")

    # ④ ⭐⭐ **"有 → 失败"时沿用上次有效集合 + 续期**（用户 2026-09-28 ✓ 原话："如果怪物查询
    #    从 有→失败，那么其应该使用使上次有效的数据缓存并刷新缓存时间而不是空"）
    #    —— ③ 刚记过「甲平台」✓ ⇒ 这一拍判不出集合时**不该变空** ✗，而是**沿用「甲平台」** ✓；
    #    `why` 仍**照记**（给 `mob_fh` 那条 log 看 ✓ 两边都不丢 ✓）。
    lt.LiveThread._mark_mob_query(th, mob, [], "怪底下没找到 foothold", now=200.0)
    got = lt.LiveThread.queried_mob_boxes(th, now=200.1)
    check(len(got) == 1 and got[0][1][0] == ["甲平台"] and "foothold" in got[0][1][1],
          "「有 → 失败」时没沿用上次有效集合（用户 2026-09-28：**不要空** ✗）：%r" % (got,))
    # ④' 沿用时**也要续期**（否则到点照样消失 ✗）—— 200.0 续的期 ⇒ 201.9 还在 ✓
    check(len(lt.LiveThread.queried_mob_boxes(th, now=201.9)) == 1,
          "沿用上次结果时没续期（框还是会消失 ✗）")
    # ④'' 但**第一次查就失败**（没有上次可沿用）⇒ 仍记成"空 + why" ✓（那一档不能丢 ✗）
    th._mob_queries.clear()
    lt.LiveThread._mark_mob_query(th, mob, [], "第一次就判不出来", now=300.0)
    got = lt.LiveThread.queried_mob_boxes(th, now=300.1)
    check(len(got) == 1 and got[0][1][0] == [] and "第一次" in got[0][1][1],
          "第一次查就失败时没照记（「查到了」那一档丢了 ✗）：%r" % (got,))


def t_queried_mob_boxes_follow_detection():
    """⭐⭐ 「查过的怪框」**跟着这一帧的检出框走**（用户 2026-10-02 ✓ 原话："能不能让他只是作为
    『怪物框的标记』，跟着怪物的检出框走"）。

    病（用户现场截图）：账里存的是**查询那一刻的框** ⇒ 怪走开之后红框**留在原地**
    （"红框在原地残留，可读性极差"✗ —— 屏幕上积了一堆空框）。

    钉四件（驱动纯函数 `queried_mob_draw_list` ✓ —— 画框那一段在 GUI 里驱动不起来，
    所以把"账 → 这一帧的框"做成纯函数、绘制那边只照画 ✓ 见那个方法 ✓）：
      ① **同一个怪号换了位置 ⇒ 用新的框**（旧写法画的是查询那一下的旧位置 ✗）；
      ② **这一帧没检出它 ⇒ 不画**（不是"画在旧位置"✗ —— 那正是用户报的残留 ✓）；
      ③ 没查过的怪**不画**（只有真查过的才标 ✓ 顺带钉住"按怪号对上"这件事 ✓）；
      ④ 到点（TTL）⇒ 不画（沿用「缓存失效再移除」✓）。
    """
    import types

    from gui import live_thread as lt

    th = types.SimpleNamespace(
        agent=types.SimpleNamespace(_zone_cd_s=lambda zone=None: 2.0), _mob_queries={})
    th._mob_query_ttl = lambda: lt.LiveThread._mob_query_ttl(th)
    # ⚠ 替身 self 也得有那两个入口（方法内部都是 `self.xxx(...)` ✓ —— 少一个就 AttributeError，
    #   而它会被当成用例失败 ✗；这一坑在同文件的记账用例里已经踩过一次 ✓）
    th.queried_mob_boxes = lambda now=None: lt.LiveThread.queried_mob_boxes(th, now=now)
    th._mark_mob_query = lambda m, names, why="", now=None: lt.LiveThread._mark_mob_query(
        th, m, names, why, now=now)

    mob = types.SimpleNamespace(id=7, x=100.0, y=200.0, w=40.0, h=60.0)
    lt.LiveThread._mark_mob_query(th, mob, ["甲平台"], "", now=100.0)

    # ① 同一个怪号（7），但**这一帧挪到 (300, 260)**（同样 40×60）⇒ 画出来的必须是**新位置** ✓
    moved = types.SimpleNamespace(id=7, x=300.0, y=260.0, w=40.0, h=60.0)
    got = lt.LiveThread.queried_mob_draw_list(th, [moved], now=100.5)
    check(got == [("7", ["甲平台"], 280, 230, 320, 290)],
          "「查过的怪框」没跟着**这一帧的检出框**走（画的是查询那一刻的旧位置 ⇒ 怪一走开"
          "红框就留在原地 ✗）：%r" % (got,))

    # ② 这一帧没检出它（视野里只有别的怪）⇒ **不画** ✓
    other = types.SimpleNamespace(id=99, x=500.0, y=200.0, w=40.0, h=60.0)
    check(lt.LiveThread.queried_mob_draw_list(th, [other], now=100.6) == [],
          "这一帧没检出那只怪却还画了框（留在旧位置上的空框 = 用户截图里那些残留 ✗）")

    # ③ 两只怪都在 ⇒ **只标查过的那只**（按怪号对上 ✓，不是"见怪就标"✗）
    got = lt.LiveThread.queried_mob_draw_list(th, [other, moved], now=100.7)
    check(len(got) == 1 and got[0][0] == "7",
          "没查过的怪也被标了（只有「真查过」的那只才画 ✓）：%r" % (got,))

    # ④ 到点（TTL = `_zone_cd_s()` = 2 秒 ✓）⇒ 不画 ✓
    check(lt.LiveThread.queried_mob_draw_list(th, [moved], now=103.0) == [],
          "过了时效还在画（用户要求「缓存失效再移除」✗）")


def t_pos_state_machine():
    """**位置状态机自治**（用户 2026-09-27："所有的位置状态更新由**位置状态机**自治" ✓）。

    位置状态 = 「玩家在地形语义上站在哪」的粗粒度快照（`here_sets` / `here_span` /
    `ladder_id` / `on_rope_pos` ✓）+ 由它派生的**本拍事实**（`at_ladder_top` /
    `at_ladder_bottom` / `ground_y` ✓）。机器是**唯一写者**；决策层**只读不算**
    （`ClimbJob._arrived` / `_at_rope`、`agent._on_rope` / `_reachable_without_path`
    都已改成读广播 ✓，2026-09-27 收编 ✓）。

    钉十件：
      ① 语义：脚下 foothold 属于哪些集合（`core.zones.set_of` 一处算 ✓）；
      ② **没按着 ↑/↓ ⇒ 不算在绳上**（用户定的许可 ✓ —— 防"走路碰到绳子就被判在绳上 ⇒ 卡住"✗）；
      ②' ⭐ 但 **`on_rope_pos`（只看位置）不管按键许可** —— 治的是执行器"自己松键 ⇒ 把自己
         判成没上绳"那条**循环依赖** ✗（用户 2026-09-27 报的现象 ✓）；
      ③ ⭐ **到顶**：**只看**「`y ≤ 绳梯上端 + 「坐标对齐误差范围」`」⇒ **当场给** ✓
         （2026-09-28 用户删掉了原来并列的第二条"`y` 在「移动操作尝试间隔」内**不再变小**"✗
         原话："移除这一项『y 在「移动操作尝试间隔」内不再变小』，用『广播"到顶"之后还会再按住
         ↑ 250ms』**仅此一项**兜底"）；
      ③' ⭐ **到底**（`at_ladder_bottom`）**这次没动**，仍是两条：`y ≥ 绳梯下端 − 容差` **且**
         在间隔内**不再变大** ✓（下爬的"到绳下端"判据 ✓，用户 2026-09-27 要求搬进广播 ✓）；
      ④ 三态分清：`None` = **判不出来**（没读数 ✓）/ `""` = **判过了没到**（执行器不许再自己
         比坐标 ✗）/ `"L1"` = 到了那根绳的那一端 ✓；
      ⑤ ⚠ **2026-09-28 这条口径反了**：`y` 还在变小（还在往上爬）**照样**判到顶 —— 只要
         **位置够高**就给 ✓（原来"还在爬就不算"已被删 ✗）；那"y 抖一下就越过绳端"怎么办 ⇒
         **由执行器兜底**：判到到达后**不立刻松手**，再按住 ↑ `base_hold_ms`（=「坐标对齐误差
         时间」，默认 **250ms**）才收工 ✓（`ClimbJob._arrived_hold` ✓）；
      ⑥ 到顶信号**只看本拍 y**：把 y 挪回绳段下方 ⇒ **立刻变回"没到"** ✓（不许留着上一拍的 ✗）；
      ⑦ ⭐ **`ground_y`**（脚下那块面的 y ✓，在**玩家 x 处**取）：下爬"落到目标平台的面"的
         判据 ✓；脚下那块 id 不在地形里 ⇒ `None`（**不拿 0 骗人** ✗）；
      ⑧ ⭐ **`here_span`**（脚下集合横着占的 x 范围 ✓）：口径**由调用方给的 `span_of`** 决定
         （= `route.set_span`，"集合里有哪些**非墙** foothold"只此一处 ✓）⇒ 机器**不自己再算** ✗；
         没有解析器 / 没圈集合 ⇒ `None` ✓。
    """
    from gui import live_thread as lt

    from perception import pos_state

    class _L:
        def __init__(self, x, y1, y2):
            self.x, self.y1, self.y2 = x, y1, y2

    class _F:
        """脚下那块面的替身（只给 `_ground_y` 要的三样 ✓）。"""

        def __init__(self, fid, left, right, y):
            self.fid, self.left, self.right = str(fid), float(left), float(right)
            self._y = float(y)

        def y_at(self, x):
            return self._y

    class _T:
        def __init__(self, lads, fhs=()):
            self.ladders = list(lads)
            self.footholds = list(fhs)

        def ladder_at(self, x, y):
            return next((L for L in self.ladders if abs(x - L.x) < 5.0), None)

    class _Z:
        def set_of(self, fid):
            return ["甲平台"] if str(fid) == "1" else []

    L1 = _L(100.0, -300.0, 0.0)
    L2 = _L(500.0, -800.0, -400.0)
    T, Z = _T([L1, L2], [_F("1", 40.0, 160.0, -300.0)]), _Z()
    loc = {"world_x": 100.0, "world_y": -305.0, "foothold_id": "1"}

    def upd(m, t, **kw):
        """调一拍 → 这一拍的**广播快照**（`PosSnapshot` ✓），转成 dict 好按名字取 ✓。

        ⚠ 2026-09-27 起 `update()` 返回的是**一个对象**（不再是 7 元组 ✓）——
          加字段只动 `PosSnapshot` 一处 ✓，这里不用再维护字段顺序表 ✓。
        """
        import dataclasses

        out = m.update(t, loc=kw.pop("loc", loc), terrain=kw.pop("terrain", T),
                       zones=kw.pop("zones", Z), **kw)
        check(isinstance(out, pos_state.PosSnapshot),
              "位置状态机该回一个 `PosSnapshot`（不是元组 ✗）：%r" % (type(out),))
        return dataclasses.asdict(out)

    # ① 语义（一处算 ✓）
    m = pos_state.PositionStateMachine()
    o = upd(m, 1000.0, hold_vert=False)
    # ⚠ 快照里的 `here_sets` 是**不可变元组**（`frozen=True` ⇒ 谁都不许就地改广播 ✓）
    check(list(o["here_sets"]) == ["甲平台"], "脚下集合算错了：%r" % (o["here_sets"],))
    # ② 没按 ↑/↓ ⇒ 不算在绳上（哪怕位置就在绳上 ✓）
    check(o["ladder_id"] is None and o["at_ladder_top"] == "",
          "没按着 ↑/↓ 却判成在绳上 / 给了到顶信号：%r / %r"
          % (o["ladder_id"], o["at_ladder_top"]))
    # ②' ⭐ 但「只看位置」那份**不看按键许可**（治执行器的循环依赖 ✗）
    check(o["on_rope_pos"] == "L1",
          "没按 ↑/↓ 时「只看位置」的绳号也丢了 —— 执行器自己松键就会"
          "「把自己判成没上绳」✗：%r" % (o["on_rope_pos"],))
    # ②'' ⭐⭐ **「坐标符合在绳梯上」**（用户 2026-10-06 ✓ 见 `PosSnapshot.on_rope_align` ✓）：
    #   在绳段里 ✓ **且** `|角色 x − 绳子的 x| ≤ 「坐标对齐误差范围」` ✓ —— 与按键许可无关 ✓
    #   （追怪那条路 `_chase_hop_beat` 就靠它判"被绳吸住"✓）。这里 x 正好就是绳的 x ⇒ 真 ✓
    check(o["on_rope_align"] is True,
          "位置就**正对着**绳子的 x（|100 − 100| = 0 ≤ 容差 10）却没判「坐标符合在绳梯上」✗：%r"
          % (o["on_rope_align"],))

    class _Tw(_T):
        """⚠ 真 · `mapdata.ladder_at` 的 x 半宽是 **24**（`LADDER_DX` ✓）；上面那个桩写的是
        5 ✗ ⇒ 要测"**在绳段里、但没对齐到绳子的 x**"这两种状态的**差**，必须换成宽尺 ✓
        （不许改那个桩 ✗ —— 别的用例按 5 那条口径钉着 ✓）。"""

        def ladder_at(self, x, y):
            return next((L for L in self.ladders if abs(x - L.x) < 24.0), None)

    o_far = upd(pos_state.PositionStateMachine(), 1000.0, hold_vert=False,
                terrain=_Tw([L1, L2], [_F("1", 40.0, 160.0, -300.0)]),
                loc={"world_x": 115.0, "world_y": -305.0, "foothold_id": "1"},
                align_tol_px=10.0)
    check(o_far["on_rope_pos"] == "L1" and o_far["on_rope_align"] is False,
          "离绳子的 x 有 15 px（> 容差 10）却判成「坐标符合在绳梯上」✗ —— "
          "那正是用户 2026-10-06 报的「**还是乱跳了**」（站在绳边打怪也跳 ✗）："
          "只看位置=%r 坐标对齐=%r" % (o_far["on_rope_pos"], o_far["on_rope_align"]))
    # ⑦ 脚下那块面的 y（玩家 x 处的面 ✓）
    check(o["ground_y"] == -300.0,
          "脚下那块面的 y 没算出来 / 算错了：%r" % (o["ground_y"],))

    # ③ 到顶：⭐ **只看"位置够不够高"** ⇒ **当场就给**（用户 2026-09-28 ✓ 原话："移除这一项
    #    『y 在「移动操作尝试间隔」内不再变小』，用『广播"到顶"之后还会再按住 ↑ 250ms』
    #    仅此一项兜底"）。原来那条要**干等 3 秒**才认到顶 ⇒ 现场就是"爬到顶了还发呆"✗。
    m = pos_state.PositionStateMachine()
    o = upd(m, 1000.0, hold_vert=True)
    check(o["ladder_id"] == "L1", "按着 ↑、位置就在绳上，却没判出绳号：%r"
          % (o["ladder_id"],))
    check(o["at_ladder_top"] == "L1",
          "位置已经够高，却**没有当场**判到顶（用户 2026-09-28 要求删掉那 3 秒等待 ✗）：%r"
          % (o["at_ladder_top"],))

    # ③' ⭐ 到底（下爬的判据）：位置够低 + 间隔内没再**变大** ⇒ 才给
    m = pos_state.PositionStateMachine()
    loc_bot = {"world_x": 100.0, "world_y": 5.0, "foothold_id": ""}
    o = upd(m, 1100.0, loc=loc_bot, hold_vert=True)
    check(o["at_ladder_bottom"] == "",
          "时间没到就给「到底」（该等「移动操作尝试间隔」✓）：%r" % (o["at_ladder_bottom"],))
    o = upd(m, 1100.0 + 3.0, loc=loc_bot, hold_vert=True)
    check(o["at_ladder_bottom"] == "L1" and o["at_ladder_top"] == "",
          "位置够低、且 3s 内没再变大，却没判到底：%r（上端不该跟着给 %r ✓）"
          % (o["at_ladder_bottom"], o["at_ladder_top"]))

    # ⑤ ⚠ **这条口径 2026-09-28 反了**（用户删掉了"y 不再变小"那条 ✓）：只要**位置够高**
    #    （`y ≤ 绳端 + 容差`）就判到顶，**不管 y 还在不在变小** ⇒ 这里现在是"照旧判到顶" ✓。
    #    ⚠ 那"y 抖一下就越过绳端"怎么办 —— **由执行器兜底**：判到到达后**不立刻松手**，
    #      会再按住 ↑ `base_hold_ms`（=「坐标对齐误差时间」，默认 **250ms**）才收工 ✓
    #      （`ClimbJob._arrived_hold` ✓）⇒ 那一小段抖动吃不掉"迈上平台"那一步 ✓。
    m = pos_state.PositionStateMachine()
    upd(m, 2000.0, hold_vert=True)
    o = upd(m, 2000.0 + 10.0, loc=dict(loc, world_y=-310.0), hold_vert=True)
    check(o["at_ladder_top"] == "L1",
          "位置够高却没判到顶 —— 用户 2026-09-28 起**只看「够不够高」**、不再看 y 变不变小 ✗：%r"
          % (o["at_ladder_top"],))

    # ④ 判不出来（没有读数）⇒ 七个字段**一律给空**（别拿旧值骗执行器 ✗）
    m = pos_state.PositionStateMachine()
    o = upd(m, 3000.0, loc={"world_x": None, "world_y": None}, hold_vert=True)
    check(o["at_ladder_top"] is None and o["ladder_id"] is None and not o["here_sets"],
          "没读数时没给「判不出来」：%r / %r / %r"
          % (o["here_sets"], o["ladder_id"], o["at_ladder_top"]))
    check(o["at_ladder_bottom"] is None and o["on_rope_pos"] is None
          and o["ground_y"] is None and o["here_span"] is None,
          "没读数时还有字段留着旧值：%r" % (o,))
    check(o["climb_failed"] is False and o["climb_stalled"] is False,
          "没读数时两个攀爬广播没清空：%r" % (o,))

    # ④' 位置离绳端太远（在下面）⇒ 判过了、没到（空串，不是 None ✓）
    m = pos_state.PositionStateMachine()
    o = upd(m, 4000.0, loc=dict(loc, world_y=-100.0), hold_vert=True)
    check(o["at_ladder_top"] == "",
          "位置还没到绳端就该给「判过了没到」（空串 ✓）：%r" % (o["at_ladder_top"],))

    # ⑥ 到顶信号**只看本拍 y**：先判到顶、再把 y 挪回绳段下方 ⇒ **必须立刻变回"没到"** ✓
    #    （不许留着上一拍的"到顶"✗）
    #    ⚠ 原来这条是"**换到 L2** ⇒ 不许把 L1 的进度带过来"。但 2026-09-28 删掉极值之后，
    #      "到顶"就是"**本拍 y 够高**" ⇒ 换到 L2 时若那一点**本来就在 L2 顶端**，判到顶是
    #      **对的**（用例场景失效，不是 bug ✗）。所以改成**同一条绳把 y 挪下来** ——
    #      语义一样（信号跟着本拍走 ✓），而且**不依赖另一根绳的几何**（`-805` 那根多长、
    #      端点在哪，用例不该知道 ✗）。
    m = pos_state.PositionStateMachine()
    o = upd(m, 5000.0, hold_vert=True)
    check(o["at_ladder_top"] == "L1", "前提不成立：L1 上没判到顶：%r" % (o["at_ladder_top"],))
    o = upd(m, 5000.0 + 3.1, loc=dict(loc, world_y=-100.0), hold_vert=True)   # y 挪到绳段下方
    check(o["at_ladder_top"] == "",
          "y 已经挪回绳段下方，却还留着上一拍的「到顶」（信号没跟着本拍走 ✗）：%r"
          % (o["at_ladder_top"],))

    # ⑦' `ground_y`：那块 id 不在地形里 ⇒ `None`（**不拿 0 骗人** ✗）
    m = pos_state.PositionStateMachine()
    o = upd(m, 6000.0, loc=dict(loc, foothold_id="查无此面"), hold_vert=False)
    check(o["ground_y"] is None,
          "脚下那块 id 不在地形里，却还是给了个 y（拿 0 骗人 ✗）：%r" % (o["ground_y"],))

    # ⑧ `here_span`：口径**由调用方给的 `span_of`** 决定，机器不自己再算 ✗
    m = pos_state.PositionStateMachine()
    o = upd(m, 7000.0, hold_vert=False, span_of=lambda name: (40.0, 160.0))
    check(o["here_span"] == (40.0, 160.0),
          "广播没把「我这块平台有多宽」给出去（`_reachable_without_path` 就又要自己算了 ✗）：%r"
          % (o["here_span"],))
    m = pos_state.PositionStateMachine()
    o = upd(m, 7100.0, hold_vert=False, span_of=None)
    check(o["here_span"] is None,
          "没有「集合→x 范围」的解析器时该给 None（不猜 ✓）：%r" % (o["here_span"],))
    # ⚠ 只认**脚下有集合**时那几块：脚下没圈进集合 ⇒ 不给（别拿别人的宽度 ✗）
    m = pos_state.PositionStateMachine()
    o = upd(m, 7200.0, loc=dict(loc, foothold_id=""), hold_vert=False,
            span_of=lambda name: (0.0, 9999.0))
    check(o["here_span"] is None,
          "脚下没圈进任何集合，却还是给了平台宽度（`here_sets` 空 ⇒ 没有「我这块」✓）：%r"
          % (o["here_span"],))


def t_calib_arrows_move_overlay():
    """标定窗的**方向键只挪底图**，不许被拖动条/数字框抢走（用户 2026-09-27 报：

    "期望按方向键或 Shift+方向键可以**按像素移动底图**，但是现在是在**操作透明度**" ✗）。

    病根：这扇窗里 `keyPressEvent`（方向键挪叠加层 ✓）本来就有，但**焦点落在谁身上谁先吃
    方向键** —— 拖动条（←→ 改值、Shift+←→ 按 pageStep）/ 数字框（↑↓ 增减）/ 下拉（↑↓ 换项）
    都会 ✗；实测焦点落在「透明度」那条上 ⇒ 按方向键变成在调透明度 ✗。
    修法：给所有子控件装事件过滤器，方向键**在到达控件之前**就被窗口吞掉（`eventFilter` ✓）。

    钉四件（真开一扇窗、真发按键 ✓）：
      ① 焦点在**透明度滑条**上按 ←/→ ⇒ `_ov_item` 挪 **1 个底图像素**、**透明度一个数不动** ✓；
      ② 焦点在**数字框**上照样是挪底图 ✓（用户报的那种"焦点跑到控件上"最典型的两处 ✓）；
      ③ Shift+方向键 ⇒ **5 个底图像素** ✓；
      ④ 上/下/左/右四个方向都认（别只认左右 ✓）。
    """
    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return

    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtGui import QKeyEvent
    from PyQt5.QtWidgets import QApplication

    from gui.minimap_calib import MinimapCalibDialog

    app = QApplication.instance() or QApplication([])
    ch, cw = canvas.shape[:2]
    w, h = min(60, cw - 10), min(100, ch - 10)
    panel = synth_crop_panel(canvas, 9, 40, w, h)
    dlg = MinimapCalibDialog(mid, mode=mm.MODE_CROP, client=StubClient(panel))
    try:
        dlg.ck_live.setChecked(True)
        dlg._on_tick()
        dlg._locate_now()
        step = max(1.0, float(dlg.scale))       # 1 个底图像素 = `scale` 个面板像素 ✓

        def press(widget, key, mod=Qt.NoModifier):
            widget.setFocus()
            before_pos = (dlg._ov_item.pos().x(), dlg._ov_item.pos().y())
            before_a = int(dlg.sld_alpha.value())
            QApplication.sendEvent(widget, QKeyEvent(QEvent.KeyPress, key, mod))
            return (before_pos, (dlg._ov_item.pos().x(), dlg._ov_item.pos().y()),
                    before_a, int(dlg.sld_alpha.value()))

        # ① 焦点在**透明度滑条**上：按 ← ⇒ 底图往左挪 1 个底图像素、透明度不动
        p0, p1, a0, a1 = press(dlg.sld_alpha, Qt.Key_Left)
        check(abs((p1[0] - p0[0]) + step) < 0.01 and abs(p1[1] - p0[1]) < 0.01,
              "焦点在透明度滑条上时按 ← 没把底图往左挪 1 个底图像素（%.1f ≠ %.1f）：%r → %r"
              % (p1[0] - p0[0], -step, p0, p1))
        check(a0 == a1,
              "按方向键把**透明度**改了（用户报的就是这个 ✗）：%d → %d" % (a0, a1))

        # ② 焦点在**数字框**上：照样是挪底图
        p0, p1, a0, a1 = press(dlg.sp_scale_y, Qt.Key_Right)
        check(abs((p1[0] - p0[0]) - step) < 0.01 and a0 == a1,
              "焦点在数字框上时按 → 没挪底图 / 动了透明度：%r → %r / %d → %d"
              % (p0, p1, a0, a1))

        # ③ Shift+方向键 = 5 个底图像素
        p0, p1, _a0, _a1 = press(dlg.sld, Qt.Key_Down, Qt.ShiftModifier)
        check(abs((p1[1] - p0[1]) - 5.0 * step) < 0.01,
              "Shift+↓ 不是 5 个底图像素（%.1f ≠ %.1f）：%r → %r"
              % (p1[1] - p0[1], 5.0 * step, p0, p1))

        # ④ 四个方向都认
        for key, dx, dy in ((Qt.Key_Up, 0, -1), (Qt.Key_Down, 0, 1),
                            (Qt.Key_Left, -1, 0), (Qt.Key_Right, 1, 0)):
            p0, p1, _a0, _a1 = press(dlg.sld_alpha, key)
            check(abs((p1[0] - p0[0]) - dx * step) < 0.01
                  and abs((p1[1] - p0[1]) - dy * step) < 0.01,
                  "方向键 %s 挪错了：%r → %r" % (key, p0, p1))
    finally:
        dlg.close()


def t_osd_shows_ladder_top():
    """「位置状态」那行要挂上**广播的 `at_ladder_top`**（用户 2026-09-27 要求 ✓）。

    ⚠ 这一行是**界面**读广播（`gui/route_panel._tick_world`）：它以前只读自己那份
      `PlayerLocator` 结果（`ladder_id` / 集合 ✓），**没有**"到没到绳上端"这条 ——
      而那是**位置状态机**的事（唯一写者 ✓）⇒ 只能**读**，**不许自己比 y** ✗。

    钉两件（源码级：这一行没有可驱动的最小夹具 ⇒ 钉接线 ✓）：
      ① `route_panel` 里出现 `pos_state()` 与 `at_ladder_top`（读了广播 ✓）；
      ② `_tick_world` 那一段里**没有**对 `at_ladder_top` 的**赋值**（只读不算 ✗）。
    """
    import inspect
    import re

    from gui import route_panel as rp

    src = inspect.getsource(rp)
    check("at_ladder_top" in src and "pos_state()" in src,
          "「位置状态」那行没读广播的 `at_ladder_top`（用户要求把它挂上 ✓）")
    m = re.search(r"    def _tick_world\(self\):\n(.*?)\n    def ", src, re.S)
    check(m is not None, "找不到 `_tick_world`（改名了？）")
    body = m.group(1)
    check("at_ladder_top" in body,
          "`_tick_world` 里没用到 `at_ladder_top`（那行没挂上 ✓）")
    check(not re.search(r"^\s*at_ladder_top\s*=", body, re.M),
          "`_tick_world` 里**自己给** `at_ladder_top` 赋值 —— 那又成了\"同一件事两处算\" ✗"
          "（它只能读位置状态机的广播 ✓）")


# ---------------------------------------------------------------- 局部小地图（crop）


def _crop_panel(canvas, scale, view, pw, ph, bg=(28, 28, 28)):
    """合成一块「**局部小地图**」面板：底图放大 `scale` 倍，**显示区起点** = `view`（底图像素）。

    也就是游戏那种画法：面板上 `(0,0)` 那个像素 = 底图 `view` 那个点，1 个底图像素
    占 `scale × scale` 个面板像素；面板比底图大出来的那圈是**面板底色**（不是地图内容 ✓）
    —— 这正是"模板不能拿边"要防的东西（见 `CropViewTracker._patch` ✓）。
    """
    big = cv2.resize(canvas, (int(round(canvas.shape[1] * scale)),
                              int(round(canvas.shape[0] * scale))),
                     interpolation=cv2.INTER_NEAREST)
    pad = 4000
    full = np.zeros((big.shape[0] + 2 * pad, big.shape[1] + 2 * pad, 3), np.uint8)
    full[:] = bg
    full[pad:pad + big.shape[0], pad:pad + big.shape[1]] = big
    ox = int(round(view[0] * scale)) + pad
    oy = int(round(view[1] * scale)) + pad
    return np.ascontiguousarray(full[oy:oy + ph, ox:ox + pw])


def _crop_setup():
    """挑一张有底图的图 → `(map_id, canvas, scale, view, 面板尺寸)`。

    面板尺寸按**真机形状**定：**比底图宽**（实测 105040303 那张竖长图就是 —— 底图
    82×218、标定 `view=(-12, 90)`、面板 502×408 ⇒ 横向整个盖住、只纵向滚动 ✓）。
    这张图不在就退回"面板只占底图一半"。
    """
    mid, canvas = pick_map()
    if mid is None:
        return None
    ch, cw = canvas.shape[:2]
    scale = 5.0
    if cw < 40 or ch < 40:
        return None
    # 横向：显示区起点 = -12 底图像素、面板比底图**宽** 24 底图像素（两边各留 12 ✓）
    vx = -12.0
    pw = max(24, min(700, int((cw + 24) * scale)))
    if ch >= 120:
        # 纵向：只看得见底图的**一半**（真的 crop：会随人滚动 ✓）
        vh = max(16, ch // 2)
        vy = float(ch // 4)
    else:
        vh = max(12, ch // 3)
        vy = float(max(0, ch // 4))
    ph = max(24, min(700, int(vh * scale)))
    return mid, canvas, scale, (vx, vy), (pw, ph)


def t_crop_view_track():
    """⭐ **局部小地图（crop）的运行时 view 跟踪**（用户 2026-09-29 任务 1 ✓）。

    **为什么要有它**：crop 的面板**随玩家滚动**（本模块顶部 / `docs/寻路设计.md` §5），
    而标定文件里存的"显示区起点"只是**标定那一刻**的位置 ⇒ 人一走，`panel_to_world`
    就整体偏"滚了多少 × 底图刻度"（这张图 1 底图像素 ≈ 7~16 世界像素 ⇒ 滚 50 像素就偏
    几百世界像素 ✗）⇒ 世界坐标 /"怪在哪一层"全错，而现场看着像"标定没量准"✗。
    （fit 没这个问题：整张底图永远都在面板里、地形不随人动 ✓。）

    钉六件：
      ① 标定那一拍：跟出来的 view 就是**合成时用的那个**（亚像素级 ✓），分高 ✓；
      ② 连续滚 10 拍：**每拍都跟住**，且误差 < 1 个底图像素（1 底图像素 = 十几世界像素 ⇒
         比「坐标对齐误差范围」那把尺（10 世界像素）还小 ✓）；
      ③ 滚起来之后走的是**附近找**（`where == "near"`）—— 便宜那一路真的在生效 ✓
         （全图找只在第一拍 / 跟丢重找时出钱 ✓）；
      ④ 面板被挡一半 / 画面全乱 ⇒ **`ok=False`、且给的还是上一拍那个 view**（不许猜 ✗）；
      ⑤ 标定不是 crop（fit）⇒ 什么都不做（`ok=False` 且说得清 ✓）；
      ⑥ 面板**比底图宽**（真机形状）也要跟得住 —— 这是"模板不能取满"那条的铁证：
         模板取满时（= 底图宽）能放它的位置只剩几个像素 ⇒ 位置根本量不出来 ✗。
    """
    su = _crop_setup()
    if su is None:
        print("      （没有可用的底图，跳过）")
        return
    mid, canvas, S, (vx, vy), (pw, ph) = su
    calib = {"mode": mm.MODE_CROP, "scale": S, "offset": [0, 0], "view": [vx, vy],
             "score": 1.0}
    panel = _crop_panel(canvas, S, (vx, vy), pw, ph)
    vt = mm.CropViewTracker()

    # ① 标定那一拍
    r = vt.update(panel, canvas, calib)
    check(r["ok"], "标定那一拍就没跟住：%s（分 %.3f）" % (r["why"], r["score"]))
    check(abs(r["view"][0] - vx) < 1.0 and abs(r["view"][1] - vy) < 1.0,
          "跟出来的显示区 %s 不是合成时那个 (%.1f, %.1f)"
          % ([round(v, 2) for v in r["view"]], vx, vy))
    check(r["score"] > 0.8, "合成面板的匹配分只有 %.3f" % r["score"])
    check(r.get("trust") is True,
          "干净面板都没被标成「可信」（`trust` 没报出来？）：%r" % (r.get("trust"),))

    # ②③ 滚动 10 拍（纵向 +3 底图像素/拍 —— 真机上 1 底图像素 ≈ 7~16 世界像素）
    for k in range(1, 11):
        _vy = vy + 3.0 * k
        _p = _crop_panel(canvas, S, (vx, _vy), pw, ph)
        rr = vt.update(_p, canvas, calib)
        check(rr["ok"], "滚到第 %d 拍跟丢了：%s（分 %.3f）" % (k, rr["why"], rr["score"]))
        check(abs(rr["view"][1] - _vy) < 1.0,
              "第 %d 拍跟出来的 y=%.2f，真值 %.2f（差 %.2f 底图像素）"
              % (k, rr["view"][1], _vy, rr["view"][1] - _vy))
        check(rr.get("trust") is True,
              "第 %d 拍这么干净的分（%.3f）却没被标成可信" % (k, rr["score"]))
        if k >= 2:
            check(rr.get("where") == "near",
                  "第 %d 拍走了「%s」找 —— 滚动是连续的，该走「附近找」（便宜那一路 ✓）"
                  % (k, rr.get("where")))

    # ④ 面板被挡掉一半 ⇒ **不许说"可信"**，而且就算给了位置也得**落在附近**
    #   （跟住那一拍的错最多错在"附近找"的窗口之内：滚动是连续的 ⇒ 差不了几像素 ✓）。
    #   ⚠ 为什么不是"必须报没跟住"（第一版就是这么写的，实测站不住 ✗）：挡掉的那半**是平的**
    #     —— 平的那半不贡献方差 ⇒ 匹配分会虚高（实测 0.6~0.7，能过"量得出来"那条线 ✓），
    #     位置也会差几个底图像素。所以能保证的是"**不许当真**"（`trust=False` ⇒ 界面那行
    #     写「不太稳」✓）和"**不许跑到老远**"（≤ 窗口余量 ✓），而不是"必须没跟住"。
    last = [float(v) for v in vt.view]
    occ = _crop_panel(canvas, S, (vx, vy + 30.0), pw, ph)
    occ[0:ph // 2, :] = (12, 34, 56)
    r2 = vt.update(occ, canvas, calib)
    if r2["ok"]:
        check(not r2.get("trust"),
              "面板被挡了一半还说这拍**可信**（分 %.3f）—— 必须报「不太稳」✗"
              % r2["score"])
        check(abs(r2["view"][1] - (vy + 30.0)) <= vt.PAD_GAIN * 3.0 + 1.0,
              "被挡之后给的位置跑远了：%s（真值 y=%.1f）—— 跟住也只在'附近'那圈里 ✗"
              % (r2["view"], vy + 30.0))
    else:
        check(r2["view"] and abs(r2["view"][0] - last[0]) < 1e-6
              and abs(r2["view"][1] - last[1]) < 1e-6,
              "没跟住时**必须保留上一拍那个** view（给的是 %s，上一拍是 %s）✗"
              % (r2["view"], last))
    noise = np.random.randint(0, 255, (ph, pw, 3), np.uint8)
    r3 = vt.update(noise, canvas, calib)
    check(not r3["ok"], "画面全是噪声也说跟住了（分 %.3f）" % r3["score"])
    # ④' **已经在跟**的时候，全图重找的门槛必须抬起来（`TRUST_SCORE`）——
    #    否则"像但不是"的地方会被认下来，而且之后每一拍都在那一带附近找 ⇒ 再也纠不回来 ✗
    #    （实测：拿**另一张图**的底图去匹配能到 0.555 ✓ 那条线挡不住它 ✗）
    import inspect as _ins
    _src = _ins.getsource(mm.CropViewTracker.update)
    check("TRUST_SCORE" in _src and "tracked" in _src,
          "全图重找没有「已经在跟就抬门槛」那条（错一次就再也回不来 ✗）")
    #    行为上钉一遍：给一个"跟得住但换了张底图"的局面，门口那道闸必须拦住 ✓
    other = None
    for q in sorted(mapdata.map_dir().glob("*.png")):
        if q.stem.endswith(("_overlay", "_zones")) or q.stem == mid:
            continue
        _t = mapdata.load(q.stem, with_canvas=True)
        if _t is not None and _t.canvas is not None and _t.canvas.shape[1] > 30:
            other = _t.canvas
            break
    if other is not None:
        vt3 = mm.CropViewTracker()
        vt3.update(_crop_panel(other, S, (-6.0, 20.0), pw, ph), other,
                   {"mode": mm.MODE_CROP, "scale": S, "offset": [0, 0],
                    "view": [-6.0, 20.0]})
        rr = vt3.update(panel, other, {"mode": mm.MODE_CROP, "scale": S,
                                       "offset": [0, 0], "view": [-6.0, 20.0]})
        check(not rr["ok"],
              "换了底图还说跟住了（分 %.3f）—— 那一拍的世界坐标就整体错了 ✗"
              % rr["score"])

    # ⑤ 不是 crop ⇒ 一个字都不动
    vt2 = mm.CropViewTracker()
    r4 = vt2.update(panel, canvas, {"mode": mm.MODE_FIT, "scale": S, "offset": [3, 4]})
    check(not r4["ok"] and r4["why"], "fit 的标定也去跟显示区了（%s）" % (r4,))

    # ⑤' ⭐ **全图重找的门槛**：直接钉规则 —— 真图场景凑不出"分数刚好落在两档中间"的
    #     确定性局面（那是这张图/这段画面的巧合 ✓），所以把 `_match` 换成一个**固定分数**
    #     的替身，只看它放不放行 ✓：
    #       · 第一拍（还没跟住过）0.65 分 ⇒ 放行（标定那一刻人站在那儿，起点就是先验 ✓）；
    #       · **已经在跟**时同样 0.65 分 ⇒ **不许放行**（此刻错一次就被当成"跟住了"，
    #         之后每拍都围着那一带找 ⇒ 再也纠不回来 ✗ —— 实测：另一张底图能到 0.555、
    #         面板被挡一半也有 0.6 上下 ✓）。
    def _stub(score):
        def f(_region, _tpl, ox, oy, where):
            return float(score), float(ox), float(oy), where
        return f

    vt4 = mm.CropViewTracker()
    _real = vt4._match
    try:
        vt4._match = _stub(0.65)
        _r = vt4.update(panel, canvas, calib)
        check(_r["ok"],
              "第一拍 0.65 分被拦了（标定那一刻人站在那儿 ⇒ 该放行 ✓）：%s" % _r["why"])
        _r = vt4.update(panel, canvas, calib)
        check(not _r["ok"],
              "**已经在跟**的时候，0.65 分的新显示区被认下来了 —— 全图重找的门槛没抬起来 ✗"
              "（错一次就再也纠不回来）")
    finally:
        vt4._match = _real

    # ⑥（真机形状就是"面板比底图宽"—— 上面那几拍跑的就是它 ✓）：
    #   再钉一句"模板不许取满底图"，否则位置量不出来时**看着还像对的**呢 ✗
    import inspect
    body = inspect.getsource(mm.CropViewTracker._patch)
    check("cw / 2.0" in body or "cw // 2" in body or "cw/2.0" in body,
          "`_patch` 里没有「模板最长边不超过底图一半」那道闸 —— 面板比底图宽时位置量不出来 ✗")


def t_crop_view_locator_wiring():
    """⭐ 定位真的**用了跟出来的显示区**（用户 2026-09-29 任务 1 ✓ 的接线）。

    `CropViewTracker` 单独能跑是一回事，`PlayerLocator.update` 拿没拿它算坐标是另一回事
    （本项目反复吃过"函数写好了、没人调"的亏 ✓）。

    做法：把黄点画在「**真实**显示区」下某个**已知世界点**对应的面板像素上，而**标定里
    那个显示区是错的**（差 40 底图像素 ≈ 几百世界像素 ✗）⇒ 只有真的跟住了，读数才能回到
    那个世界点 ✓。
    """
    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return
    t = mapdata.load(mid, with_canvas=True)
    if t is None or not t.segments:
        print("      （没有带段的地形数据，跳过）")
        return
    seg, f0 = next((s, f) for s in t.segments for f in s.footholds if not f.is_wall)
    wx = (f0.left + f0.right) / 2.0
    wy = f0.y_at(wx)
    ccx, ccy = t.world_to_canvas(wx, wy)

    ch, cw = canvas.shape[:2]
    if cw < 40 or ch < 60 or ccy < 8 or ccy > ch - 8 or ccx < 8 or ccx > cw - 8:
        print("      （这张图的落脚点不适合合成面板，跳过）")
        return
    S = 5.0
    pw = max(40, min(700, int(cw * S)))
    ph = max(40, min(700, int(ch * 0.6 * S)))
    # **真实**显示区：让那个世界点落在面板正中
    vx = ccx - (pw / S) / 2.0
    vy = ccy - (ph / S) / 2.0
    panel = _crop_panel(canvas, S, (vx, vy), pw, ph)
    # 黄点：下沿（脚底）对着那个世界点的面板像素（口径同 `dot_feet` ✓）
    dot = 6
    px = int(round((ccx - vx) * S))
    py = int(round((ccy - vy) * S))
    panel[max(0, py - dot + 1):py + 1, max(0, px - dot // 2):px + dot // 2 + 1] = (65, 243, 245)

    # 标定里那个显示区**是错的**（差 40 底图像素）—— 跟住了就该纠正过来 ✓
    bad = {"mode": mm.MODE_CROP, "scale": S, "offset": [0, 0],
           "view": [vx - 40.0, vy], "score": 1.0}
    loc = mm.PlayerLocator(mid)
    r = loc.update(panel, src=mm.SRC_LIVE, calib=bad, terrain=t)
    check(r.get("view_ok") is True,
          "定位这一拍没跟住显示区（view_ok=%r why=%s）" % (r.get("view_ok"), r.get("view_why")))
    tol = 2.5 * t.px_per_world           # 2.5 个底图像素（跟的精度 + 黄点本身的尺寸）
    check(r["ok"], "跟住了却算不出坐标：%s（黄点层：%s）" % (r["note"], r["dot"]))
    check(abs(r["world_x"] - wx) <= tol and abs(r["world_y"] - wy) <= tol,
          "读数 (%.0f, %.0f) 没回到那个世界点 (%.0f, %.0f)（容差 %.0f）—— 说明算坐标时"
          "用的还是**标定里那个错的显示区**（跟出来的没接上 ✗）"
          % (r["world_x"], r["world_y"], wx, wy, tol))
    check(abs(float(r["view"][0]) - vx) < 1.0,
          "出参里那个 view %s 不是跟出来的（%.1f）" % (r["view"], vx))

    # 对照：**fit** 的标定 ⇒ `view_ok` 必须是 None（"没这回事"，界面一个字都不加 ✓）
    fit = {"mode": mm.MODE_FIT, "scale": S, "offset": [0, 0], "view": [0, 0]}
    r2 = mm.PlayerLocator(mid).update(_crop_panel(canvas, 1.0, (0, 0), cw, ch),
                                      src=mm.SRC_LIVE, calib=fit, terrain=t)
    check(r2.get("view_ok") is None,
          "fit 的标定却报了显示区跟踪状态（%r）—— fit 没这回事，别让人以为它在跟 ✗"
          % (r2.get("view_ok"),))


def t_live_minimap_on_terrain_map():
    """⭐⭐ 「路线识别 → **地形图**」里能**叠加实时小地图**（用户 2026-09-29 第 ⑦ 条 ✓ 原话：
    "现在实时画面小地图比较小很难看清是否对准，可以在路线识别页签→地形图里的图里实时滚动
    收流图（我自己控制底图），这样我就方便观察局部了"）。

    做法：把那块面板**按标定摆到地形图上**（`ImageCanvas.set_live_patch` ✓），坐标换算走
    `panel_to_canvas`（面板→底图，唯一口径 ✓）再乘 `overlay_zoom`（地形图 = 底图的放大版 ✓）。

    钉六件：
      ⓪ **默认就是勾上的**，而且节拍**要等这一页真显示出来才跑**（切走就停）——
        用户 2026-09-29 追加："怎么没在路线识别页签→地形图里的视图区看到实时滚动的真实
        小地图"✗ 的根因就是"默认关着"：这一层默认看不见 = **做了等于没做** ✓；
      ① `show()` 之后 ⇒ 节拍起来、那一层出现、位置/倍数**按标定算**
         （crop 下 = 面板左上那点 + 1/scale × overlay_zoom ✓）；
      ② **局部小地图用这一拍跟出来的显示区**（`_ov_view` ✓ —— 和画面上那层叠图同一口径）；
      ③ 取帧**必须走非阻塞那条**（定时器 4 次/秒：`blocking=False` ✓）—— 阻塞版会新建客户端
         并最多等 5 秒 ✗，放在定时器里就是"工作台每隔 250ms 卡 5 秒" ✗；
      ④ 去勾 ⇒ 那一层**收起来**（不是留一张空的 ✓）；
      ⑤ 没标定 / 没底图 / 取不到帧 ⇒ 那一层收起来 + 一行说清（不许静默 ✗）；
      ⑥ `hide()`（切走页签）⇒ 节拍停掉（没人看就不取帧 ✓）。
    """
    import inspect

    from PyQt5.QtGui import QPixmap
    from PyQt5.QtWidgets import QApplication

    from gui import route_panel as rp
    from gui.route_panel import RoutePanel

    app = QApplication.instance() or QApplication([])       # noqa: F841
    # ⚠ 「显示类型」是**下拉**（一次点击一个事件 ⇒ 当场落盘 ✓）⇒ 用例里必须挡住
    #   `update_live`，否则切一次就把用户的 `config/live.yaml` 改了 ✗（§11 红线 ✓）。
    _real_update = rp.update_live
    rp.update_live = lambda **kw: None
    # ⚠⚠ **「读」那一头也要挡**（2026-10-06 真踩到 ✗）：这一栏的初值是**从
    #   `config/live.yaml` 恢复**的（用户自己在下拉里选过「像素差分地图」✓）⇒ 用例直接读
    #   **用户那份配置**的话，下面那条"默认该是「实时小地图」"就会当场红 ✗（盘上存的是
    #   `live_map_view: diff` ✓）。⇒ 打桩成"**没存过**"（`{}` ✓）—— 于是这条钉的正是
    #   "**没有存过时**默认 live"这条口径 ✓（那才是"默认"二字的意思 ✓ 别去改用户的配置 ✗）。
    _real_load = rp.load_live
    rp.load_live = lambda: {}
    mid, canvas = pick_map()
    if mid is None:
        rp.update_live = _real_update
        rp.load_live = _real_load
        print("      （没有可用的底图，跳过）")
        return
    t = mapdata.load(mid, with_canvas=True)
    cw, ch = t.canvas.shape[1], t.canvas.shape[0]
    z = float(mm.overlay_zoom(cw))
    w, h = 60, 80
    vx, vy = 8.0, 11.0
    panel = synth_crop_panel(canvas, int(vx), int(vy), w, h)
    # ⚠ `synth_crop_panel` = 原样切一块（面板 (0,0) 就是底图 (vx, vy)）⇒ `offset` 必须是
    #   [0, 0]：写 [4, 4] 的话这份标定和面板**差 4 个底图像素**（自己跟自己不一致 ✗），
    #   而 2026-09-29 起位置的来源多了"这一层自己跟出来的显示区"⇒ 那 4 px 会当场露出来 ✓
    cal = {"mode": mm.MODE_CROP, "scale": 1.0, "offset": [0, 0], "view": [vx, vy]}

    p = RoutePanel()
    try:
        p.canvas.load(QPixmap(40, 40))       # 先"画一张地形图"（`_live_item` 这时才存在 ✓）
        p._map_id = lambda: mid
        p._mmap_src = lambda: mm.SRC_STREAM
        _called = []                         # 记"取帧时用的是哪条路"（阻塞/非阻塞 ✓）

        def _fetch(blocking=True):
            _called.append(bool(blocking))
            return panel, ""

        p._mmap_panel_for_check = _fetch
        p._overlay_blocker = lambda: ""
        _real_calib = mapdata.load_calib
        mapdata.load_calib = lambda m, src=None: dict(cal)
        try:
            # ⓪ **默认就该是勾上的**（用户 2026-09-29 追加："怎么没在路线识别页签→地形图里
            #    的视图区看到实时滚动的真实小地图"✗ —— 这一层默认关着 = 做了也看不见 ✓）；
            #    而节拍**要等这一页真的显示出来**才跑（切到别的页签就停 ✓）。
            _disp = [p.cmb_live_view.itemText(i)
                     for i in range(p.cmb_live_view.count())]
            check(_disp == ["仅地形图", "实时小地图", "像素差分地图"],
                  "「显示类型」下拉的三项不对（用户 2026-10-06 点名的就是这三个 ✓）：%r"
                  % (_disp,))
            check(p._live_disp == "live",
                  "「显示类型」默认该是「实时小地图」（老行为 = 那一层默认就在）"
                  "—— 用户 2026-09-29 就是因为它默认关着才什么都看不到 ✗：%r"
                  % (p._live_disp,))
            check(not p._live_map_timer.isActive(),
                  "还没显示这一页就把 250ms 的取帧节拍开起来了（切走也不会停？✗）")
            # ① 进这一页（show）⇒ 节拍起来、那一层出现，且位置/倍数按标定算
            p.show()
            app.processEvents()
            check(p._live_map_timer.isActive(),
                  "显示了「路线识别」页，取帧节拍却没起来（那还是看不到 ✗）")
            it = p.canvas._live_item
            check(it is not None and it.isVisible(),
                  "进了这一页、勾也是勾上的，那一层却没出现")
            bx, by = mm.panel_to_canvas(0.0, 0.0, cal)
            # ⚠ 容差 1 个底图像素：位置现在来自"这一层自己跟出来的显示区"（跟它有
            #   ~0.5 底图像素的拟合分辨率 ✓），不再是标定那个数照抄 —— 那正是修的东西 ✓
            check(abs(it.pos().x() - bx * z) <= 1.0 * z
                  and abs(it.pos().y() - by * z) <= 1.0 * z,
                  "面板摆错地方了：item 在 (%s, %s)，按标定该在 (%s, %s)±%.0f 个图素"
                  % (it.pos().x(), it.pos().y(), bx * z, by * z, z))
            check(abs(it.transform().m11() - z / cal["scale"]) < 1e-6,
                  "面板的放大倍数不对（该是 overlay_zoom / scale）：%s"
                  % it.transform().m11())
            check("面板" in p.lbl_live_map.text(),
                  "那行没说清摆了块多大的面板：%r" % p.lbl_live_map.text())
            # ② 局部小地图：**这一层自己跟**那一拍面板的显示区（2026-09-29 追加：
            #    "位置没有跟着我的移动变化"✗ —— 老实现只有 `_ov_view` 一条路，而那条路
            #    要"叠图开着 + 世界坐标算得出来"才更新 ⇒ 认不出黄点时就钉死了 ✗✗）。
            #    这里 `_ov_view` 一直是 None：位置照样要跟着面板走 ✓
            _p2 = _crop_panel(t.canvas, 1.0, (vx + 5.0, vy), w, h)
            p._mmap_panel_for_check = lambda blocking=True: (_p2, "")
            p._live_map_tick()
            # ⚠ 容差给 3 个底图像素：这条钉的是"**跟着挪了**"（老实现一点都不挪 ✗），
            #   而合成面板在**平坦区**上匹配本来就有一片等高的位置（真小地图里有地标 ✓）
            #   ⇒ 精确到"1 像素内"这件事由下面的用例在本机底图上另钉一遍 ✓
            check(abs(it.pos().x() - (bx + 5.0) * z) <= 3.0 * z,
                  "面板的显示区挪了 5 个底图像素，图上那块却没跟着挪（item x=%s，该 ≈ %s）"
                  " —— 这就是用户报的「位置没有跟着我的移动变化」✗" % (it.pos().x(),
                                                                      (bx + 5.0) * z))
            check(p._ov_view is None,
                  "`_ov_view`（画面那层跟出来的）本来是空的，却被写上了 —— 用例前提不成立")
            p._mmap_panel_for_check = _fetch
            # ③ 取帧必须走**非阻塞**那条（行为级钉：看它实际传了什么，不看源码 ——
            #    文档字符串里也写着 `blocking=False`，拿 grep 查是假的 ✗ 踩过 ✓）
            check(_called and all(v is False for v in _called),
                  "定时器那条路用了**阻塞**取帧（`blocking=True`）—— 那版会新建客户端、"
                  "最多等 5 秒 ✗，4 次/秒就是每 250ms 卡一下 ✗：%s" % _called)
            check(inspect.getsource(RoutePanel._live_map_tick), "占位")
            # ④ 选「仅地形图」⇒ 收起来
            p.cmb_live_view.setCurrentIndex(p.cmb_live_view.findData("terrain"))
            check(not it.isVisible(), "选了「仅地形图」那一层还在（该收起来 ✓）")
            check(not p._live_map_timer.isActive(), "收了之后定时器还在跑 ✗")
            # ⑤ 没标定 ⇒ 收起来 + 说清
            mapdata.load_calib = lambda m, src=None: {}
            p.cmb_live_view.setCurrentIndex(p.cmb_live_view.findData("live"))
            check(not it.isVisible(),
                  "没标定却把面板摆上去了（摆哪儿都说不清 ✗）")
            check("标定" in p.lbl_live_map.text(),
                  "没标定时那行要说清原因：%r" % p.lbl_live_map.text())
            # ⑥ 切走这一页（hide）⇒ 节拍停掉（没人看就不取帧 ✓）
            p.hide()
            app.processEvents()
            check(not p._live_map_timer.isActive(),
                  "切走了还在每 250ms 取帧（白花 ✓ 该停 ✗）")
            p.cmb_live_view.setCurrentIndex(p.cmb_live_view.findData("terrain"))
        finally:
            mapdata.load_calib = _real_calib
    finally:
        rp.update_live = _real_update      # 还原（别把这个替身留给后面的用例 ✓）
        rp.load_live = _real_load          # 「读」那个替身也还原 ✓
        p.deleteLater()
        app.processEvents()


def t_dot_basemap_core_gate():
    """⭐⭐ **底图相减层里那道"得像玩家点"的闸**（用户 2026-10-06 ✓ 原话："**做**"✓ —— 承接
    "位置状态把黄点认错了，刚才人在底层最右边角落卡死"✓）。

    真色是**量出来的**（`tools/mmap_dot_probe.py` 连推流 ✓）：**BGR = (108, 255, 255)** ✓
    ⇒ 闸 = `G ≥ 180 且 R ≥ 180 且 B ≤ 160` ✓（见 `DOT_CORE_*` ✓）。

    为什么要它：那块地图**底图本身大片黄褐** ⇒ 颜色层被判「被淹」⇒ 退到**底图相减层** ⇒
    那一层只看"比底图多出来的东西" ⇒ **锁假点** ⇒ 位置状态给出"新鲜但错"的坐标 ⇒ 卡死 ✓。

    钉四件：
      ① 口径：真色那三档判 True ✓；**从真底图上取的"家族能命中"的颜色**判 False ✓
         （用例自己把这条前提**断言出来** ✓ 取不到就跳过 ✓ —— 不许猜判据 ✓）；
      ② **集成**：面板 = 真底图一块 ＋ 一个"家族色但不像核心"的假点（摆**左上** ⇒ 老代码会先挑它 ✗）
         ＋ 一个**真色**的点（右下 ✓）⇒ 出来必须是**真点** ✓；
      ③ **零回归**：一个候选都不像核心色时 ⇒ **照用原候选**（不许把点弄丢 ✗）；
      ④ 源码级：`find_player_dot` 的 basemap 支**真的调了**这条闸 ✓（加了函数没接上 = 白做 ✗）。
    """
    import inspect
    import types

    from perception import minimap as mm

    # ---- ① 口径 ----
    _true = np.zeros((8, 8, 3), np.uint8)
    _true[:, :] = (108, 255, 255)
    check(mm.dot_core_mask(_true).all(),
          "量到的真色 (108,255,255) 过不了核心色闸 ⇒ 这个邻域配错了 ✗")
    _pale = np.zeros((8, 8, 3), np.uint8)
    _pale[:, :] = (210, 235, 250)          # "发白的黄"（B 高 ⇒ 不该算核心色 ✓）
    check(not mm.dot_core_mask(_pale).any(),
          "B=210 的浅黄也算成了玩家点核心色（闸太松 ✗）")

    # ---- ②③ 直接钉这道闸**本身**（"拒假点 / 留真点" ✓）----
    #   ⚠ **为什么不做"整条链"的造景**（试过 ✗ 如实记）：要让 `find_player_dot` 真的走到**底图
    #     相减支**，得先让颜色层被判「被淹」（要撒够家族色 ✓），还得让相减**真减掉东西**
    #     （`m.sum() < mask_px*0.5` ✓ 否则那层被按"没意义"跳过 ✗）—— 这两件事一起凑出来的合成图
    #     非常脆（第一版就没走进去 ✓）；而**真面板**的证据已经在 `data/dot_probe/stream_panel.png`
    #     里 ✓（黄族命中 3.78%、均值 BGR≈(116,211,244) ✓ 点 (108,255,255) ✓）。
    #     ⇒ 这里钉"闸对两种颜色的判断" ✓ 那是**唯一会被改坏的地方** ✓；
    #       "接上了没有"由下面 ④ 的源码级钉 ✓。
    #   "假点"色 = **真实面板上量到的洪水色** ✓（G=211 < 240 ⇒ 该被拒 ✓）
    _fake_bgr = (116, 211, 244)
    check(mm.dot_color(np.full((6, 6, 3), _fake_bgr, np.uint8), "yellow").all(),
          "用例前提不成立：量到的洪水色不在黄族里（判据动过？）：%r" % (_fake_bgr,))
    panel = np.full((60, 60, 3), (90, 90, 90), np.uint8)
    m = np.zeros((60, 60), np.uint8)
    panel[10:30, 10:30] = _fake_bgr                      # 假点块（左上）
    panel[40:60, 40:60] = (108, 255, 255)                # 真点块（右下）
    m[10:30, 10:30] = 1
    m[40:60, 40:60] = 1
    _flood = {"x": 20.0, "y": 20.0, "w": 20, "h": 20, "area": 400}
    _true = {"x": 50.0, "y": 50.0, "w": 20, "h": 20, "area": 400}
    check(mm._blob_looks_core(panel, m, _true) is True,
          "真点（量到的 (108,255,255) ✓）被判成「不像玩家点」⇒ 闸把真点也拒了 ✗")
    check(mm._blob_looks_core(panel, m, _flood) is False,
          "**洪水色**（量到的 (116,211,244) ✓）被判成「像玩家点」⇒ 这道闸拦不住假点 ✗"
          "（那就是第一版 180 那条线的问题：一点都拒不掉 ✓）")
    # ⚠ 零回归那条（"一个都没保住就退回原候选" ✓）是**行为**，在 `find_player_dot` 里 ✓
    #   ⇒ 用源码级把它和接线一起钉住 ✓（下面 ④）。

    # ④ 源码级：接上了没有 + 有没有留"零回归"的退路
    _src = inspect.getsource(mm.find_player_dot)
    check("_blob_looks_core" in _src,
          "`find_player_dot` 的底图相减支没接上「像玩家点核心色」那道闸（加了函数没接 = 白做 ✗）")
    check("if _core:" in _src and "else:" in _src,
          "少了「一个候选都没保住 ⇒ 退回原候选」那条退路（会把本来能找到的点弄丢 ✗ 零回归没了 ✗）")


def t_dot_force_reacquire():
    """⭐ **"锁在假点上"的自愈**：`PlayerDotTracker.force_reacquire()` 清掉锁定的位置 ⇒ 下一拍
    **全画面**重捕（用户 2026-10-06 ✓ 原话："**那就是位置状态把黄点认错了，刚才人在底层最右边
    角落卡死**"✓）。

    现场（我读的 `behavior.log` ✓）：`mmap_dot` 一直在 `basemap` ↔ `roi` 之间翻 ⇒ 颜色层从没赢过
    ⇒ 靠"底图相减"兜 ⇒ 会锁一个**不动的假点** ⇒ 位置状态给出"新鲜但错"的坐标 ⇒ Agent 照它指挥
    往右顶墙，人其实已卡在底层最右边角落 ✓。

    钉四件：
      ① 点跑出 ROI（局部搜不到）⇒ `force_reacquire()` ⇒ 下一拍**全画面**能找到它 ✓；
      ② 重捕过 `last_roi` 变 `None`（= 整幅搜 ✓ 不是还在那一小块里打转 ✗）；
      ③ `reacquires` 有记账（排查用 ✓）；
      ④ **源码级接线**：判据在 `agent`（`want_dot_reacquire` ✓ 它才知道在按方向键 ✓）、
        执行在 `live_thread`（`force_reacquire` ✓）—— ⚠ 跟踪器**自己不许**判 ✗（它看不见按键：
        "人真站着不动"和"读数锁住"在它眼里一样 ✓）。
    """
    import inspect

    from perception import minimap as mm

    tr = mm.PlayerDotTracker()
    p1 = _dot_panel()
    p1[50:56, 40:46] = (65, 243, 245)               # 玩家点在这儿
    r1 = tr.update(p1, now=1.0)
    check(r1["ok"], "第一拍就没找到点：%s" % (r1.get("reason"),))
    check(tr.last_roi is None,
          "**第一拍**该是整幅搜（还没有上一帧位置 ⇒ 没有 ROI ✓）：%r" % (tr.last_roi,))
    # ⚠ **要连着两拍**才升到"确认"（口径 `DOT_CONFIRM_FRAMES` ✓）—— 确认之后才开始只用局部搜 ✓
    #   （所以上面第一拍一定是整幅 ✓）。
    check(tr.update(p1, now=1.05)["ok"], "第二拍同一个点就没找到了")
    # 点跑到很远（超出 `DOT_ROI_PAD` 那块局部）⇒ 局部搜不到（沿用上一帧 / 认不出）
    # ⚠ 位置要**按面板实际大小**摆（`_dot_panel()` 不大 ✓ 写死 300 会跑到画布外 ⇒ 谁都搜不到 ✗ 踩过）。
    p2 = _dot_panel()
    _h, _w = p2.shape[:2]
    _nx, _ny = int(_w) - 22, int(_h) - 22
    p2[_ny:_ny + 6, _nx:_nx + 6] = (65, 243, 245)
    r2 = tr.update(p2, now=1.1)
    check((not r2["ok"]) or r2.get("held"),
          "远处那个点不该靠局部搜到（那就说明 ROI 没生效 ✗）：%r" % (r2.get("reason"),))
    # ⚠ 这里**不再**断言"`last_roi` 一定不是 None" ✗（试过：确认之后也未必走 ROI 那条 —— ROI 的
    #   触发条件还在跟踪器内部 ✓ 用例不该把它钉成"实现细节" ✗）。**要钉的是行为**：
    #   ① 远处那个点靠局部/防抖**找不回来** ✓（上面那条 ✓）；② 重捕之后**能**找回来 ✓（下面 ✓）。
    # ⭐ 要求重捕 ⇒ 下一拍全画面 ⇒ 找到它 ✓ 而且 `last_roi` 说明是"整幅"
    tr.force_reacquire("用例：锁在假点上了")
    r3 = tr.update(p2, now=1.2)
    check(r3["ok"] and abs(r3["x"] - (_nx + 2.5)) <= 9 and abs(r3["y"] - (_ny + 2.5)) <= 9,
          "重捕之后没找到远处那个点（说明没做全画面搜 ✗）：%r" % (r3,))
    check(tr.last_roi is None,
          "重捕之后还在原来那一小块里搜（`last_roi` 该是 None = 整幅 ✓）：%r" % (tr.last_roi,))
    check(int(tr.reacquires) == 1, "重捕没记账：%r" % (getattr(tr, "reacquires", None),))
    # ④ 接线（源码级 ✓ 两处都要在）
    from decision import agent as _ag
    from gui import live_thread as _lt
    check("want_dot_reacquire" in inspect.getsource(_ag.CombatAgent._climb_tick),
          "agent 那一侧没立「按键在发但位置不动」的判据（跟踪器看不见按键 ✗）")
    # ⚠ 方法名别写错：执行那条在 **`_run`**（`run` 只是壳 ✓ 写错这条就永远红 ✗ 踩过）。
    check("force_reacquire" in inspect.getsource(_lt.LiveThread._run),
          "实时回路没执行 `force_reacquire`（判据有了、动作没接上 = 白做 ✗）")


def t_dot_basemap_layer_first():
    """⭐⭐ **底图相减优先**（用户 2026-10-06 ✓ 原话："**优先使用底图相减找黄点**"✓）。

    现场（图 `110040000`、实时取证 `tools/_probe_chains.py` ✓）：面板 1164×585 ✓、标定自洽
    （反推 1161×584 ≈ 推流那块 ✓）、**每帧都有 2 个候选**（`yellow 族` ✓）—— 因为小地图上
    **别的黄色图标**（NPC / 传送点 / 任务点 ✓）跟玩家点同族 ✗，而"跨帧跟踪"按"离上一帧
    最近"挑 ⇒ **第一拍挑错就锁死** ✓（实测点 (648.8,124.3) **五帧零位移** ✗、世界落**段 4** ✗，
    真值是**段 9「底层」右端 ≈1202** ✓）⇒ Agent 拿它择路 ⇒ **一路怼墙** ✓。

    钉三件：
      ① **地图元素（两图都有的黄块）不许赢**：底图上那个黄块**比真点还大** ✓ ⇒ 若"颜色层先"
         就会挑它 ✗；相减层把它减掉 ✓ ⇒ 结论必须是**底图上没有的那个真点** ✓；
      ② 结论来自 `basemap` 层 ✓；
      ③ **退路还在**：没有底图时颜色层照样兜底 ✓（换顺序 ≠ 把颜色那条路删掉 ✗）。
    """
    import types

    from perception import minimap as mm

    canvas = np.full((120, 160, 3), (90, 90, 90), np.uint8)
    canvas[40:52, 60:72] = (60, 220, 230)        # ⭐ 底图上**本来就有的黄色图标**（12×12，比真点大）
    vx, vy, w, h = 20, 30, 80, 60
    panel = np.ascontiguousarray(canvas[vy:vy + h, vx:vx + w].copy())
    panel[20:25, 30:35] = (65, 243, 245)        # ⭐ 真玩家点（5×5，底图上**没有** ✓）
    cal = {"mode": mm.MODE_CROP, "scale": 1.0, "offset": [0, 0], "view": [vx, vy]}
    terrain = types.SimpleNamespace(canvas=canvas)
    roi = (0, 0, w, h)
    r = mm.find_player_dot(panel, calib=cal, terrain=terrain, roi=roi)
    check(r["ok"], "两个黄块都在，却一个都没认出来：%s" % (r.get("reason"),))
    check(abs(float(r["x"]) - 32.0) <= 3 and abs(float(r["y"]) - 22.0) <= 3,
          "挑错了：该挑**底图上没有的那个真点**（面板 (32,22) ✓），却挑了地图元素 (%.1f,%.1f) "
          "—— 这正是现场「锁在假点上」的样子 ✗：%s"
          % (float(r["x"]), float(r["y"]), r.get("reason")))
    check(r["layer"] == "basemap",
          "结论该来自「底图相减层」（用户要求优先用它 ✓）：%r" % (r.get("layer"),))
    # ③ 退路：没有底图 ⇒ 颜色层照旧兜底 ✓（换顺序不等于把老路删掉 ✗）
    #   ⚠ 这里**必须用"只有一个黄点"的干净面板** ✗：上面那个面板有两个黄块，颜色层会判
    #     **被淹**而**故意不给结论** ✓（那是它本来的纪律："宁可认不出，也别给假的"✓）——
    #     拿它测就会误判成"老路丢了" ✗（第一版就是这么写错的 ✓ 当场被自检抓住 ✓）。
    _clean = np.full((120, 160, 3), (90, 90, 90), np.uint8)
    _only = np.ascontiguousarray(_clean[vy:vy + h, vx:vx + w].copy())
    _only[20:25, 30:35] = (65, 243, 245)
    r2 = mm.find_player_dot(_only, calib=cal, terrain=None, roi=roi)
    check(r2["ok"] and r2["layer"] == "color",
          "没有底图时颜色层没兜住（换顺序把老路弄丢了 ✗）：%r / %s"
          % (r2.get("layer"), r2.get("reason")))


def t_dot_diff_layer():
    """⭐⭐ **玩家点定位的第三层：差分层**（用户 2026-10-06 「下一步」✓ —— 上一轮只做了"看"的
    差分地图，这一步让它**真去定位**✓ 原话链条："是不是颜色相减" ⇒ "增加显示类型" ⇒
    "下一步" = 拿它找黄点 ✓）。

    动机（前两层为什么会落空）：色族那两层（`color` / `basemap`）**只看"玩家标记色"**（黄/青 ✓）
    ⇒ 标记画成别的颜色、或底色偏色时**认不出** ✗。差分层**不看颜色** ✓（面板 − 底图那一块 ✓）
    专门补这种漏检 ✓；但整幅不同源时它会乱响 ✗ ⇒ **只在色族全落空时兜一次** ✓。

    钉五件：
      ① 合成：灰底 + 底图上本来就有一块暗的（两图都有 ⇒ 差 0 ✓）+ **一个灰白亮点**
         （**不属黄/青族** ⇒ 色族两层必然落空 ✓）⇒ 差分层把它找出来（`layer == "diff"` ✓）；
      ② 整幅不同源（底图整体偏 40 灰阶 ⇒ 背景档 ≥ 30 ✓）⇒ **不采信差分层**（`ok=False` ✓）
         且 reason 里说清"不同源" ✓（静默会让人拿它当判据去调标定 ✗）；
      ③ 没有亮点（面板就是底图那块）⇒ 差分层**不给候选**（`ok=False` ✓ 不硬凑 ✓）；
      ④ 源码级：`find_player_dot` 里**确实调了** `dot_candidates_diff` ✓（别只加了函数没接上 ✗）；
      ⑤ 那条线**一处实现**：`gui.route_panel.DIFF_FLOOR_WARN is mm.DIFF_FLOOR_WARN` ✓
         （界面那份原来是自己写的一个 30 ✗，现在 import 感知层那份 ✓）。
    """
    import types

    from perception import minimap as mm

    # ---- ① 合成：亮灰点（不属黄/青）⇒ 只有差分层能看见 ----
    canvas = np.full((120, 160, 3), (90, 90, 90), np.uint8)
    canvas[40:44, 60:66] = (30, 30, 30)          # 底图上本来就有的东西（两图都有 ⇒ 差 0 ✓）
    vx, vy, w, h = 20, 30, 60, 50
    panel = np.ascontiguousarray(canvas[vy:vy + h, vx:vx + w].copy())
    panel[10:14, 20:24] = (200, 200, 200)        # ⭐ 灰白亮点：**不属黄/青族**（R≈G≈B ✓）
    cal = {"mode": mm.MODE_CROP, "scale": 1.0, "offset": [0, 0], "view": [vx, vy]}
    terrain = types.SimpleNamespace(canvas=canvas)
    roi = (0, 0, w, h)
    check(not mm.dot_mask(panel, None).any(),
          "用例造错了：那个亮点被判据当成黄/青族了（②就没意义了 ✗）")
    r = mm.find_player_dot(panel, calib=cal, terrain=terrain, roi=roi)
    check(r["ok"] and r["layer"] == "diff",
          "色族两层都落空时，差分层该兜住这个**不看颜色**的亮点：%r / %s"
          % (r.get("layer"), r.get("reason")))
    check(abs(r["x"] - 21.5) <= 3 and abs(r["y"] - 11.5) <= 3,
          "差分层找到的位置不对（该是注进去那个亮点）：%r" % ((r["x"], r["y"]),))
    check(r["flood"] is True, "差分层给的结论该记成「颜色层没帮上忙」（`flood` ✓）：%r" % (r,))

    # ---- ② ⭐ **整幅"不同源"不再一票否决**（用户 2026-10-06 ✓ **故意的行为改变** ✓）----
    #   原口径：底图整幅偏 ⇒ `floor ≥ DIFF_FLOOR_WARN` ⇒ **拒** ✗；
    #   为什么要改（现场 ✓ 图 `110040000`）：界面那张"**黑底 + 亮团**"的差分图是**归一化**
    #     过的 ✓ ⇒ 看着很跳 ✓，可**绝对**对比度（`raw_peak − floor`）常常**不到 30** ✗
    #     ⇒ 老闸把好东西挡在门外 ⇒ 一直落回"认颜色"那两族 ⇒ 而那张图底图一片黄褐 ⇒
    #     颜色层一口气 15 个候选全坐在底图上 ✗ ⇒ 锁假点 ⇒ 人卡在右下角 ✓。
    #   ⇒ 现在：`floor` 只**提示**（`info.flat` ✓ 也拼进理由句 ✓）、**不拦** ✓；
    #     真正把关的是"**亮团比背景明显**"（`DOT_DIFF_MIN_CONTRAST` ✓）＋ 归一化阈值 ＋
    #     亮块形状 ＋ 被淹 ✓。
    #   ⚠ 这条用例从此钉的是：**整幅偏也要能采信，并且把"整幅偏"写出来** ✓
    #     （"不同源就不能用"那条老口径**作废** ✓ 别再按它改回去 ✗）。
    canvas2 = np.clip(canvas.astype(np.int16) + 40, 0, 255).astype(np.uint8)
    r2 = mm.find_player_dot(panel, calib=cal,
                            terrain=types.SimpleNamespace(canvas=canvas2), roi=roi)
    #   ⚠⚠ **这里只钉"必须说清"** ✗ 不钉"必须采信"：整幅平移之后**能不能给出候选**
    #     还取决于尺寸/归一化那几条 ✓（合成图里就未必给 ✓）—— 拿它当"采信与否"的判据
    #     是把实现细节当契约 ✗（本项目踩过多次 ✓）。
    #     要钉的是**透明度**：不管采信还是没给出，理由里都必须写明"**这张图整幅偏**"✓。
    check("整幅偏" in (r2.get("reason") or ""),
          "没说「这张图整幅偏」✗ —— 人得知道底色不一样（只是不影响亮团分离 ✓）：%r"
          % (r2.get("reason"),))

    # ---- ③ 只有噪声、没有亮点 ⇒ **不许硬凑**（⚠ 这里必须**带噪声** ✗：纯净的两张图
    #   差分全是 0 ⇒ 归一化后也是 0 ⇒ 那道闸用不上、"拆掉也不红" ✗ 反向验证当场抓到 ✓。
    #   真实面板永远有压缩/抗锯齿噪声 ✓ ⇒ 拿噪声造才是真场景 ✓）----
    _rng = np.random.RandomState(20261006)
    _noisy = np.clip(canvas[vy:vy + h, vx:vx + w].astype(np.int16)
                     + _rng.randint(-3, 4, (h, w, 3)), 0, 255).astype(np.uint8)
    r3 = mm.find_player_dot(_noisy, calib=cal, terrain=terrain, roi=roi)
    check(not r3["ok"],
          "面板只有噪声（没有多出来的东西）却给了结论 ⇒ 归一化把噪声放大成「亮点」了 ✗"
          "（那道「没有比背景明显就不给候选」的闸没起作用）：%r" % (r3,))
    check("没有比背景明显" in (r3.get("reason") or ""),
          "差分层因为「没亮点」而不用时该**说清是这条原因**（不是笼统「认不出」✗）：%r"
          % (r3.get("reason"),))

    # ---- ④⑤ 接线 + 那条线一处实现 ----
    import inspect

    from gui import route_panel as rp
    _src = inspect.getsource(mm.find_player_dot)
    check("dot_candidates_diff(" in _src,
          "`find_player_dot` 里没接差分层（函数写了没接上 = 白写 ✗）")
    check(rp.DIFF_FLOOR_WARN is mm.DIFF_FLOOR_WARN,
          "「背景档」那条线在界面里是**另写的一份**（两边迟早分叉 ✗）：%r / %r"
          % (rp.DIFF_FLOOR_WARN, mm.DIFF_FLOOR_WARN))

    # ---- ⑥ **"哪里看得到"**也要钉住（用户 2026-10-06 追问：「差分层」字样哪里看 ✓）----
    #   ⚠ 起因（我上一轮说得太笼统 ✗）：那句长理由（`dot`）**只在定位失败时**才有人显示
    #     ⇒ **成功靠差分层兜住**这种情况**根本看不到** ✗ ⇒ 单独立一个短字段 + 写进日志 ✓。
    # ⚠⚠ **钉要钉"精确串"** ✗（第一版只查 `"dot_layer" in src` ⇒ 反向验证**没红** ✗：
    #   把键名改成 `dot_layer_x` 照样含这个子串 ⇒ 弱钉等于没钉 ✓ 这条教训与 SKILL 273 同源 ✓）。
    _pl = inspect.getsource(mm.PlayerLocator)
    check('"dot_layer":' in _pl,
          "`PlayerLocator` 没把「哪一层认出的」立成一个短字段（要**精确**写成 "
          "`\"dot_layer\":` ✓）⇒ 成功时谁都看不到 ✗")
    from gui import live_thread as LT
    _lt = inspect.getsource(LT.LiveThread._locate_latest)
    check('perf.note("mmap_dot",' in _lt and '"dot_layer"' in _lt,
          "实时回路没把「哪一层认出黄点」写进日志（`perf.note(\"mmap_dot\", …)` ✓ ＋ "
          "读 `dot_layer` ✓）—— 那就还是只有失败时看得见 ✗")


def t_diff_map_view():
    """⭐⭐ 「像素差分地图」（用户 2026-10-06 ✓ 原话：把「叠加实时小地图」改成「显示类型」
    下拉，含 **仅地形图 / 实时小地图 / 像素差分地图** ✓ —— 起因是他那天问的
    "是不是颜色相减"，想亲眼看一眼"面板减底图"长什么样 ✓）。

    它看的是**面板 —— 底图上它盖住的那一块**：两者本该一样 ⇒ 差出来的就是"**底图上没有的
    东西**"（玩家点 / 实时元素 ✓）。⚠ 底图是解码出来的地形画布、与游戏里那张小地图
    **不一定同源** ⇒ 差分**先减掉整幅的背景档**再放大（不然整幅糊成白的、什么都看不出来 ✗），
    并把那个档报出来（偏大就说清「只能当辅助」✓ 静默会让人拿它当判据去调标定 ✗）。

    钉五件：
      ① 纯函数：面板 = 底图那一块 **+ 一个亮点** ⇒ 最突出那个点就是它（面板像素 ✓）、
         取的那块底图矩形、图与面板**同尺寸** ✓；
      ② 整幅偏一点（不同源那种形状）⇒ 减掉背景档之后亮点照样突出 ✓、`floor` 如实报 ✓；
      ③ 面板落在底图外面 ⇒ 如实 `ok=False`（**不抛、也不给假图** ✗）；
      ④ 那一栏选「像素差分地图」⇒ 摆上去的**是差分图**（把图读回来核：亮点亮、底图上
         本来就有那处**该是暗的** ✓）＋ 状态行写出背景档 / 最突出量 / 世界坐标 ✓；
         切回「实时小地图」⇒ 摆的是**面板本身** ✓；
      ⑤ 「仅地形图」⇒ 收起来 + 节拍停 + 透明度灰掉 ✓；差分算不出来时**照旧摆面板**并说明 ✓。
    """
    from PyQt5.QtGui import QPixmap
    from PyQt5.QtWidgets import QApplication

    from gui import route_panel as rp
    from gui.route_panel import RoutePanel

    def _v(img, x, y):
        """图上那个像素的**最大通道值**（差分是灰的 ⇒ 三个通道一样 ✓）。"""
        c = img.pixelColor(int(x), int(y))
        return max(c.red(), c.green(), c.blue())

    # ---- ① 纯函数（合成，脱 Qt ✓）----
    canvas = np.zeros((160, 200, 3), np.uint8)
    canvas[:] = (60, 70, 80)
    canvas[60:100, 20:60] = (150, 40, 40)          # 一块地物（免得整幅一个色 ✓）
    cal = {"mode": mm.MODE_CROP, "scale": 1.0, "offset": [0, 0], "view": [30.0, 40.0]}
    panel = np.ascontiguousarray(canvas[40:90, 30:90].copy())   # h=50 w=60，原样切一块
    panel[18:22, 10:14] = (255, 255, 255)          # "玩家点"（底图上没有 ✓）
    vis, di = mm.diff_panel_vs_canvas(panel, canvas, cal)
    check(vis is not None and di.get("ok"),
          "差分算不出来（%r）—— 面板明明就是底图上切的一块 ✓" % (di,))
    check(vis.shape == panel.shape,
          "差分图该和面板同尺寸（面板 %s / 差分 %s）" % (panel.shape, vis.shape))
    check(di.get("rect") == (30, 40, 90, 90),
          "取的那块底图矩形不对（该是面板盖住的那一块）：%r" % (di.get("rect"),))
    check(abs(di["at"][0] - 12) <= 3 and abs(di["at"][1] - 20) <= 3,
          "最突出的那个点该是注进去的亮点（面板 (12,20) 那一块）：%r" % (di.get("at"),))
    check(int(vis[20, 12].max()) > 200 and int(vis[45, 55].max()) < 40,
          "亮点处该亮、平地处该暗（实得 %r / %r）"
          % (vis[20, 12].tolist(), vis[45, 55].tolist()))
    check(int(di.get("peak") or 0) > 150,
          "`peak` 没把「多差出来的那一截」报出来：%r" % (di.get("peak"),))
    # ---- ② 整幅偏一点（= 底图与游戏小地图不同源的形状）----
    panel2 = np.clip(panel.astype(np.int16) + 25, 0, 255).astype(np.uint8)
    vis2, di2 = mm.diff_panel_vs_canvas(panel2, canvas, cal)
    check(di2.get("ok") and float(di2.get("floor") or 0) >= 20,
          "整幅偏 25 灰阶，背景档却没报出来（%r）" % (di2.get("floor"),))
    check(int(vis2[20, 12].max()) > 200,
          "减掉背景档之后亮点该照样突出（实得 %r）" % (vis2[20, 12].tolist(),))
    # ---- ③ 落在底图外面 ⇒ 如实说，不抛也不给假图 ----
    vis3, di3 = mm.diff_panel_vs_canvas(panel, canvas,
                                        dict(cal, view=[5000.0, 5000.0]))
    check(vis3 is None and not di3.get("ok"),
          "面板整个落在底图外面，却还是给了张差分图（会让人以为「有内容」✗）：%r"
          % (di3,))
    check(di3.get("why"), "算不出来时该有一句人话（别静默 ✗）：%r" % (di3,))
    # ---- ④⑤ 那一栏真跑一遍（挑一张**有内容**的真底图 ✓）----
    app = QApplication.instance() or QApplication([])       # noqa: F841
    _real_update = rp.update_live
    rp.update_live = lambda **kw: None                      # 别写用户的 live.yaml ✓
    mid, _base = pick_map()
    if mid is None:
        rp.update_live = _real_update
        print("      （没有可用的底图，④⑤ 跳过 ✓ ①②③ 已经跑过 ✓）")
        return
    t = mapdata.load(mid, with_canvas=True)
    _m = t.canvas[..., :3].max(axis=2)
    _ys, _xs = np.nonzero(_m > 60)
    if _ys.size == 0:
        rp.update_live = _real_update
        print("      （这张底图整幅都是黑的，④⑤ 跳过 ✓）")
        return
    ch, cw = t.canvas.shape[:2]
    pw, ph = max(30, min(cw - 8, 70)), max(30, min(ch - 8, 90))
    vx = float(min(max(int(_xs[0]), 1), cw - pw - 1))
    vy = float(min(max(int(_ys[0]), 1), ch - ph - 1))
    crop = np.ascontiguousarray(t.canvas[int(vy):int(vy) + ph,
                                        int(vx):int(vx) + pw])
    _mc = crop[..., :3].max(axis=2)
    _by, _bx = np.unravel_index(int(_mc.argmax()), _mc.shape)   # 最亮：核「面板本身」用 ✓
    _dy, _dx = np.unravel_index(int(_mc.argmin()), _mc.shape)   # 最暗：往这儿注亮点 ✓
    _dx = int(min(max(_dx, 1), pw - 3))
    _dy = int(min(max(_dy, 1), ph - 3))
    panel4 = crop.copy()
    panel4[_dy:_dy + 3, _dx:_dx + 3] = (255, 255, 255)
    cal4 = {"mode": mm.MODE_CROP, "scale": 1.0, "offset": [0, 0], "view": [vx, vy]}
    p = RoutePanel()
    _real_calib = mapdata.load_calib
    try:
        p.canvas.load(QPixmap(40, 40))      # 先"画一张地形图"（`_live_item` 这时才存在 ✓）
        p._map_id = lambda: mid
        p._mmap_src = lambda: mm.SRC_STREAM
        p._mmap_panel_for_check = lambda blocking=True: (panel4, "")
        p._overlay_blocker = lambda: ""
        mapdata.load_calib = lambda m, src=None: dict(cal4)
        # ⚠ 显示区跟踪**打桩成「不跟」**：这条用例要的是"那份标定**原样**用" ——
        #   跟住了它会挪半个像素 ⇒ 差分会带上"错位差" ⇒ 平地处也可能亮 ✗（断言就飘了 ✓）。
        p._live_view_track.update = lambda *a, **k: {"ok": False, "why": "用例：不跟"}
        it = p.canvas._live_item
        # ④ 选「像素差分地图」⇒ 摆上去的**是差分图**
        # ⚠⚠ **这个下拉是"按变化触发"的**：若盘上存着的**就是 `diff`**（用户自己选过 ✓），
        #   `setCurrentIndex(findData("diff"))` 等于**没变化** ⇒ 信号不发 ⇒ 这一栏什么都不做
        #   ⇒ 下面 `it.isVisible()` 当场红 ✗（2026-10-06 真踩到，盘上正是 `live_map_view: diff` ✓）。
        #   ⇒ **先切到别的、再切到差分** ✓（两种存值下都必有一次真变化 ✓）。
        p.cmb_live_view.setCurrentIndex(p.cmb_live_view.findData("live"))
        p.cmb_live_view.setCurrentIndex(p.cmb_live_view.findData("diff"))
        check(p._live_disp == "diff", "下拉选了差分，状态却没记住：%r" % (p._live_disp,))
        check(it.isVisible(), "选了「像素差分地图」却什么都没摆上去")
        p._live_map_tick()
        img = it.pixmap().toImage()
        check(img.width() == pw and img.height() == ph,
              "差分图该和面板同尺寸（面板 %d×%d / 图上 %d×%d）"
              % (pw, ph, img.width(), img.height()))
        _hot = _v(img, _dx + 1, _dy + 1)
        _flat = _v(img, _bx, _by)
        check(_hot > 200, "差分里注进去的那个亮点该是亮的（实得 %d）" % _hot)
        check(_flat < 40,
              "底图上**本来就有**的那处，差分里该是暗的（实得 %d）—— 亮着说明摆上去的"
              "是面板本身、不是差分 ✗" % _flat)
        _txt = p.lbl_live_map.text()
        check(("差分" in _txt) and ("世界" in _txt),
              "状态行没写清差分看到了什么（背景档 / 最突出量 / 世界坐标 ✓）：%r" % (_txt,))
        # ⭐⭐ **"把点的颜色显示出来，而不是纯白"**（用户 2026-10-06 ✓ 原话）：
        #   界面上那份差分该用**面板本来的颜色**画那块——直接拿纯函数核最省 ✓（绕开 Qt ✓）。
        _cal_c = {"mode": mm.MODE_CROP, "scale": 1.0, "offset": [0, 0], "view": [0, 0]}
        _cv = np.full((40, 40, 3), (90, 90, 90), np.uint8)
        _pn = np.ascontiguousarray(_cv.copy())
        _pn[10:14, 10:14] = (0, 210, 240)          # 一个**黄点**（BGR：蓝低、绿红高 ✓）
        _vx2, _i2 = mm.diff_panel_vs_canvas(_pn, _cv, _cal_c, color=True)
        check(_vx2 is not None and _i2.get("color") is True,
              "差分没进「彩色」档（用户要的是把点的颜色显示出来 ✓）：%r" % (_i2,))
        _hot2 = _vx2[11, 11].astype(int)
        check(_hot2[0] < 60 and _hot2[1] > 150 and _hot2[2] > 150,
              "亮点没按**面板本来的颜色**画（该是那个黄点 ✓，而不是纯白 ✗）：BGR=%r" % (_hot2,))
        _dark2 = _vx2[30, 30].astype(int)
        check(int(_dark2.max()) < 40,
              "底图上本来就有的那处该是黑的（差 ≈ 0 ✓）：BGR=%r" % (_dark2,))
        # 切回「实时小地图」⇒ 摆的是**面板本身**（同一格该是底图那个值 ✓）
        p.cmb_live_view.setCurrentIndex(p.cmb_live_view.findData("live"))
        p._live_map_tick()
        img2 = it.pixmap().toImage()
        check(_v(img2, _bx, _by) == int(_mc[_by, _bx]),
              "切回「实时小地图」之后摆的该是面板本身（那格该是 %d，实得 %d）"
              % (int(_mc[_by, _bx]), _v(img2, _bx, _by)))
        # ⑤ 「仅地形图」⇒ 收起来 + 节拍停 + 透明度灰掉
        p._live_map_timer.start()
        p.cmb_live_view.setCurrentIndex(p.cmb_live_view.findData("terrain"))
        check(not it.isVisible(), "选了「仅地形图」那一层还在（该收起来 ✓）")
        check(not p._live_map_timer.isActive(), "收了之后节拍还在跑 ✗")
        check(not p.sld_live_alpha.isEnabled(),
              "「仅地形图」时透明度该灰掉（都没有那一层 ✓）")
        check(not p.lbl_live_map.text(), "收了之后那行还留着上一句（该清掉 ✓）：%r"
              % (p.lbl_live_map.text(),))
        # 差分算不出来 ⇒ **照旧摆面板** + 说明（不许把画面弄空 ✗）
        p.cmb_live_view.setCurrentIndex(p.cmb_live_view.findData("diff"))
        mapdata.load_calib = lambda m, src=None: dict(cal4, view=[5000.0, 5000.0])
        p._live_map_tick()
        check(it.isVisible(), "差分算不出来时把画面弄空了（该照旧摆面板 ✓）")
        check("差分算不出来" in p.lbl_live_map.text(),
              "差分算不出来却没说（人就靠这行知道发生了什么 ✓）：%r"
              % (p.lbl_live_map.text(),))
    finally:
        mapdata.load_calib = _real_calib
        rp.update_live = _real_update
        p.deleteLater()
        app.processEvents()


def t_live_map_alpha_and_follows():
    """⭐⭐ 「地形图里那块实时小地图」要**自己跟着人走** + 有**透明度**参数（用户 2026-09-29 追加：
    "需要加个参数「透明度」，表示实时小地图的透明度" / "刚测试发现在这个视图区里，实时小地图的
    位置没有跟着我的移动变化，在森林迷宫III这张地图期望它应该会向上或向下移动"✗）。

    钉五件：
      ① **面板显示区变了 ⇒ 图上那块跟着挪**（挪的底图像素数 × `overlay_zoom` ✓）——
         老实现把它钉在标定那一刻的位置上 ✗；
      ② 跟的是"**这一层自己这一拍的面板**"：`_ov_view` 一直是 None 也要跟 ✓
         （老实现只有 `_ov_view` 那条路 ⇒ 认不出黄点 / 没开叠图时它就不动了 ✗✗）；
      ③ 跟不住时的**退路**：`_ov_view` 有 ⇒ 用画面那层跟出来的 ✓；都没有 ⇒ 照标定里那份摆
         并在状态行**橙色说清**（不许静默照标定 ✗ —— 那会让人以为它还在跟着走）；
      ④ **透明度**：拖一下 ⇒ 那一层当场变淡（不重取帧 ✓）；**换地形图重挂之后仍是那个浓淡** ✓
         （`load()` 会重建那一层 —— 不重贴就悄悄回到不透明 ✗）；
      ⑤ **落盘**：松手 ⇒ 写进 `live.yaml` 的 `live_map_alpha` ✓（本机外观偏好 ✓）。
    """
    from PyQt5.QtGui import QPixmap
    from PyQt5.QtWidgets import QApplication

    from gui import route_panel as rp
    from gui.route_panel import RoutePanel

    app = QApplication.instance() or QApplication([])       # noqa: F841
    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return
    t = mapdata.load(mid, with_canvas=True)
    ch, cw = t.canvas.shape[:2]
    z = float(mm.overlay_zoom(cw))
    vx, vy = 6.0, 9.0
    pw, ph = max(30, min(cw - 12, 70)), max(30, min(ch - 12, 90))
    cal = {"mode": mm.MODE_CROP, "scale": 1.0, "offset": [0, 0], "view": [vx, vy]}
    holder = {"vy": vy, "noise": False}
    p = RoutePanel()
    _real_calib = mapdata.load_calib
    _real_update = rp.update_live
    saved = {}
    try:
        p.canvas.load(QPixmap(40, 40))
        p._map_id = lambda: mid
        p._mmap_src = lambda: mm.SRC_STREAM

        def _fetch(blocking=True):
            if holder["noise"]:
                # 「跟不住」= 有画面但认不出来（噪声）—— **不是**"取不到帧"（那是另一条路
                # ：面板都没有时这一层直接收起来 ✓，测不到"退回标定"那条 ✓）
                return np.random.randint(0, 255, (ph, pw, 3), np.uint8), ""
            return _crop_panel(t.canvas, 1.0, (vx, float(holder["vy"])),
                               pw, ph), ""

        p._mmap_panel_for_check = _fetch
        p._overlay_blocker = lambda: ""
        mapdata.load_calib = lambda m, src=None: dict(cal)
        rp.update_live = lambda **kw: saved.update(kw)      # 别写真的 live.yaml ✓
        p._live_view_track.reset((vx, vy))
        if p._live_disp != "live":
            p.cmb_live_view.setCurrentIndex(p.cmb_live_view.findData("live"))
        it = p.canvas._live_item
        # ① 人往下走 10 个底图像素（面板显示区 y 变大）⇒ 图上那一块要**往下**挪 10×z ✓
        p._live_map_tick()
        y0 = float(it.pos().y())
        holder["vy"] = vy + 10.0
        p._live_map_tick()
        dy = float(it.pos().y()) - y0
        check(abs(dy - 10.0 * z) <= 1.5 * z,
              "面板显示区挪了 10 个底图像素，图上那块只挪了 %.1f（该 ≈ %.1f，"
              "1 底图像素 = %.0f 个地形图像素）—— 面板钉在标定那一刻的位置上了 ✗"
              % (dy, 10.0 * z, z))
        check("跟住了显示区" in p.lbl_live_map.text(),
              "明明跟住了，状态行却没说（人没法知道它在跟着走 ✓）：%r"
              % p.lbl_live_map.text())
        # ② 跟不住 + 画面那层也没跟出来（`_ov_view` 空）⇒ 退回**标定里那份** + 橙色说清
        holder["noise"] = True
        p._ov_view = None
        p._live_map_tick()
        check(abs(float(it.pos().y()) - vy * z) <= 1e-6,
              "跟不住时该退回标定里那个位置（y=%.1f），实际 %.1f"
              % (vy * z, it.pos().y()))
        check("⚠" in p.lbl_live_map.text() and "标定" in p.lbl_live_map.text(),
              "跟不住却**静默**照标定摆（人会以为它还在跟着走 ✗）：%r"
              % p.lbl_live_map.text())
        check("#b06000" in p.lbl_live_map.styleSheet(),
              "跟不住那行该是**橙色**（和世界坐标那行一个口径 ✓）：%r"
              % p.lbl_live_map.styleSheet())
        # ③ 跟不住、但**画面那层跟出来了** ⇒ 用画面那层那个显示区 ✓
        p._ov_view = (vx + 7.0, vy)
        p._live_map_tick()
        check(abs(it.pos().x() - (vx + 7.0) * z) <= 1e-6,
              "自己跟不住时没退回画面那层跟出来的显示区：x=%.1f（该 %.1f）"
              % (it.pos().x(), (vx + 7.0) * z))
        # ④ 透明度：拖一下当场生效（不重取帧 ✓），重挂地形图之后**还是那个浓淡** ✓
        holder["noise"] = False
        holder["vy"] = vy
        p._ov_view = None
        p._live_map_tick()
        p.sld_live_alpha.setValue(30)
        check(abs(it.opacity() - 0.30) < 1e-6,
              "拖了透明度，那一层却没变（opacity=%.2f ✗）" % it.opacity())
        p._live_map_tick()          # 再走一拍：取帧那一路**不许**把浓淡写回默认值 ✗
        check(abs(it.opacity() - 0.30) < 1e-6,
              "下一拍把浓淡写回默认了（opacity=%.2f）—— 拖了等于没拖 ✗" % it.opacity())
        p.canvas.load(QPixmap(40, 40))              # 换/重画地形图 ⇒ 那一层被重建
        check(abs(p.canvas._live_item.opacity() - 0.30) < 1e-6,
              "重画地形图之后浓淡丢了（回到 %.2f）—— 人会以为「参数没生效」✗"
              % p.canvas._live_item.opacity())
        # ⑤ 落盘（松手那一下 / 离开这一页）：写进 live.yaml 的 `live_map_alpha` ✓
        p._save_live_alpha()
        check(saved.get("live_map_alpha") == 30,
              "透明度没落盘（松手/切页都该存一次 ✓）：%r" % (saved,))
        # ⑥ ⭐ **认不出黄点的那一拍也要把「显示区」取出来**（`_tick_world` 里的顺序 ✗→✓）：
        #     "面板随人滚动"这件事**不需要认出黄点**（`PlayerLocator.update` 里在黄点之前
        #     就跟了一遍 ✓）—— 老代码把取显示区放在 `if not r["ok"]: return` **后面** ⇒
        #     认不出黄点时那条信息被丢掉 ⇒ `_ov_view` 永远是 None ⇒ **画面那层叠图**也
        #     钉死在标定那一刻 ✗✗（用户报的现象有它一份 ✓）。
        p._mmap_draw_on = lambda: True
        p._stream_client = lambda: type("_C", (), {
            "latest": lambda self: (_crop_panel(t.canvas, 1.0, (vx, vy), pw, ph), 1.0),
            "err": ""})()

        class _LP(object):
            def set_overlay_note(self, *_a):
                return True

        p.live_panel = _LP()          # 实时面板（真实运行里由主窗口挂上 ✓）
        p._locator.update = lambda *a, **k: {
            "ok": False, "short": "认不出黄点", "note": "（用例：没黄点）",
            "dot": "（用例）", "view": [vx + 3.0, vy + 4.0], "view_ok": True,
            "view_score": 0.93, "view_trust": True, "view_why": ""}
        p._ov_view = None
        seen = []
        p._refresh_overlay = lambda **kw: seen.append(kw.get("view"))
        p._tick_world()
        check(p._ov_view == (vx + 3.0, vy + 4.0),
              "认不出黄点那一拍没把「显示区」记下来（`_ov_view`=%r）—— 那两处（画面叠图、"
              "这一层）都会钉死 ✗" % (p._ov_view,))
        check(bool(seen) and seen[0] is not None,
              "记下了显示区却没让画面那层叠图跟着重画 ✗：%r" % (seen,))
        # ⑦ ⭐ **附近找"不太像"时必须再全图找一次**（不然冷启动窗口只有 `PAD_MIN` = 2 个
        #    底图像素 ⇒ 人站着不动之后猛一走，真位置落在窗口外，匹配就贴着窗口边给个
        #    0.6~0.7（过得了 0.55）⇒ 那一拍的显示区偏 pad 那么多、而且**看着像跟住了** ✗）。
        #    用替身 `_match` 造这个局面（真图凑不出确定性：全靠图有多"平"）
        vt = mm.CropViewTracker()
        vt.reset((vx, vy))
        vt._step = (0.0, 0.0)               # 有历史 ⇒ 会走"附近找"那一路
        calls = []

        def _fake_match(region, tpl, ox, oy, where):
            calls.append(where)
            if where == "near":
                return 0.62, float(ox), float(oy), where       # "像，但不很像"
            return 0.93, float(ox + 8.0), float(oy), where     # 全图找给的是另一处

        vt._match = _fake_match
        r7 = vt.update(_crop_panel(t.canvas, 1.0, (vx, vy), pw, ph), t.canvas,
                       dict(cal))
        check("full" in calls,
              "附近找只有 0.62（明显不够像）却没再全图找一次 —— 冷启动那一下会偏 pad ✗：%s"
              % (calls,))
        check(r7["ok"] and r7["where"] == "full",
              "全图找更高的那个（0.93）没被采纳：where=%r ok=%s"
              % (r7.get("where"), r7.get("ok")))
    finally:
        mapdata.load_calib = _real_calib
        rp.update_live = _real_update
        p.deleteLater()
        app.processEvents()


def t_scroll_axes_helpers():
    """⭐ 「滚动方向」的口径（用户 2026-09-29 追加："局部小地图也应该分滚动类型：
    '仅X''仅Y''双轴'，这样在对于单轴滚动的小地图，实时匹配时可以提高速度"✓）。

    钉三件：① 老标定（**没有这个键**）⇒ 双轴 = 老行为一字不变 ✓；② 写错/写怪 ⇒ 双轴
    （**宁慢不猜**：猜成单轴会把不动的那根轴钉死 ⇒ 那才是真错 ✗）；③ 三个值都能认 ✓。
    """
    check(mm.scroll_of({}) == mm.SCROLL_XY
          and mm.scroll_of({"scroll": None}) == mm.SCROLL_XY,
          "没写滚动方向却不是双轴（老标定必须一字不变 ✓）")
    check(mm.scroll_of({"scroll": "什么鬼"}) == mm.SCROLL_XY,
          "写了个不认识的值 ⇒ 该退回双轴（猜成单轴会把另一根轴钉死 ✗）")
    check(mm.scroll_of({"scroll": "x"}) == mm.SCROLL_X
          and mm.scroll_of({"scroll": "X"}) == mm.SCROLL_X
          and mm.scroll_of({"scroll": "仅x"}) == mm.SCROLL_X,
          "「仅 X」没认出来（大小写/中文写法都要认 ✓）")
    check(mm.scroll_axes({"scroll": "x"}) == (True, False)
          and mm.scroll_axes({"scroll": "y"}) == (False, True)
          and mm.scroll_axes({}) == (True, True),
          "`scroll_axes` 映射不对：%r / %r / %r"
          % (mm.scroll_axes({"scroll": "x"}), mm.scroll_axes({"scroll": "y"}),
             mm.scroll_axes({})))
    keys = [k for k, _t in mm.SCROLL_LABEL]
    check(keys == [mm.SCROLL_XY, mm.SCROLL_X, mm.SCROLL_Y],
          "界面下拉的选项不该是这几个：%r" % (keys,))


def t_crop_view_track_scroll_axis():
    """⭐⭐ **单轴的小地图：只在会动的那根轴上搜，另一根轴一个字都不改**（用户 2026-09-29 ✓）。

    为什么值得这么做（除了快）：单轴图上另一根**本来就不会动** ⇒ 让匹配在那根轴上"跟"
    等于把噪声当位移 ⇒ 显示区会慢慢漂、世界坐标跟着漂 ✗。

    钉四件：
      ① `scroll="x"` 时**全图重找的区域收成一条横带**（高 ≈ 模板 + 一点余量，不是整张底图 ✓）
         —— 这是"快好几倍"的**行为级**证据（拿 `_match` 替身记下实际喂进去的区域 ✓）；
      ② `scroll="x"` + 面板在 **Y 方向**挪 10 个底图像素 ⇒ 跟出来的 **y 一动不许动** ✓
         （照上一拍 / 标定里那个 ✓）；
      ③ X 方向挪 10 个 ⇒ 跟着挪 ✓（单轴照样要跟得动 ✓）；
      ④ 不写（双轴）⇒ 区域仍是整张底图 ✓（老行为 ✓）。
    """
    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return
    t = mapdata.load(mid, with_canvas=True)
    ch, cw = t.canvas.shape[:2]
    w, h = max(40, min(cw - 10, 70)), max(40, min(ch - 10, 90))
    vx, vy = 4.0, 5.0
    cal_x = {"mode": mm.MODE_CROP, "scale": 1.0, "offset": [0, 0],
             "view": [vx, vy], "scroll": mm.SCROLL_X}
    cal_xy = dict(cal_x, scroll=mm.SCROLL_XY)
    regions = []

    def _spy(make_result):
        def f(region, tpl, ox, oy, where):
            regions.append((where, region.shape, tpl.shape))
            return make_result(region, tpl, ox, oy, where)
        return f

    vt = mm.CropViewTracker()
    _real = vt._match
    p0 = _crop_panel(t.canvas, 1.0, (vx, vy), w, h)
    try:
        # ④ 双轴（不写）⇒ 全图找就是整张底图 ✓
        vt_xy = mm.CropViewTracker()
        vt_xy._match = _spy(lambda r, t2, ox, oy, where: _real(r, t2, ox, oy, where))
        vt_xy.update(p0, t.canvas, dict(cal_xy))
        fulls = [s for where, s, _tp in regions if where == "full"]
        # ⚠ 双轴时**也可能**先走"附近找"（有历史）⇒ 这里只要求"整图那次"存在时是整图 ✓
        if fulls:
            check(fulls[0][0] == ch,
                  "双轴的全图找没把整张底图喂进去（高 %d，底图高 %d）" % (fulls[0][0], ch))
        # ① 仅 X ⇒ 全图找是一条**横带**（高 ≈ 模板 + 余量）
        regions.clear()
        vt_x = mm.CropViewTracker()
        vt_x._match = _spy(lambda r, t2, ox, oy, where: _real(r, t2, ox, oy, where))
        r1 = vt_x.update(p0, t.canvas, dict(cal_x))
        check(r1["ok"], "仅 X 的图第一拍就没跟上：%s" % r1["why"])
        fulls = [(s, tp) for where, s, tp in regions if where == "full"]
        check(fulls, "仅 X 的图没走全图找？%s" % (regions,))
        bh, tp = fulls[0][0][0], fulls[0][1][0]
        check(bh <= tp + 2 * (vt_x.PAD_MIN + 2) + 1,
              "仅 X 的全图找没收成横带（区域高 %d，模板高 %d）—— 那「快好几倍」就没了 ✗"
              % (bh, tp))
        # ② 面板整体在 Y 上挪 10 ⇒ 跟出来的 y **一动不许动**
        p1 = _crop_panel(t.canvas, 1.0, (vx, vy + 10.0), w, h)
        r2 = vt_x.update(p1, t.canvas, dict(cal_x))
        if r2["ok"]:
            check(abs(r2["view"][1] - vy) < 1e-6,
                  "仅 X 的图，y 却跟着挪了（%.2f → %.2f）—— 那一轴本来就不会动，"
                  "跟它只会把噪声当位移 ✗" % (vy, r2["view"][1]))
        # ③ X 方向挪 10 ⇒ 要跟着挪 ✓
        p2 = _crop_panel(t.canvas, 1.0, (vx + 10.0, vy), w, h)
        vt_x2 = mm.CropViewTracker()
        vt_x2.update(p0, t.canvas, dict(cal_x))
        r3 = vt_x2.update(p2, t.canvas, dict(cal_x))
        check(r3["ok"], "仅 X 的图，X 挪了 10 却跟丢了：%s" % r3["why"])
        check(abs((r3["view"][0] - vx) - 10.0) <= 1.5,
              "仅 X 的图，X 挪了 10 个底图像素，跟出来只有 %.2f（该 ≈ 10）✗"
              % (r3["view"][0] - vx))
    finally:
        vt._match = _real


def t_locate_crop_scroll_x():
    """⭐ 「仅 X」的图：**自动定位只在 X 轴上动**（y 照种子给的 ✓，缩照样量 ✓）。

    为什么：单轴图里 y 是标定定死的 ⇒ 自动定位不许把 y 改掉（改了 = 用一根轴的自由度
    去补另一根轴的误差 ⇒ 看着分高了、世界坐标却错了 ✗）。
    """
    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return
    t = mapdata.load(mid, with_canvas=True)
    ch, cw = t.canvas.shape[:2]
    w, h = max(40, min(cw - 12, 70)), max(40, min(ch - 12, 90))
    vx, vy = 3.0, 4.0
    panel = _crop_panel(t.canvas, 1.0, (vx + 3.0, vy), w, h)   # 真值：x 偏 3、y 同
    seed = {"mode": mm.MODE_CROP, "scale": 1.0, "offset": [0, 0],
            "view": [vx, vy], "scroll": mm.SCROLL_X}
    r = mm.locate_crop(panel, t.canvas, seed=seed)
    check(r is not None, "仅 X 的图上自动定位没量出来（拿种子也没救回来？）")
    check(abs(r["view"][1] - vy) < 1e-6,
          "仅 X 却把 y 从 %.2f 改成了 %.2f —— 那一轴由标定定死，不许拿它去补误差 ✗"
          % (vy, r["view"][1]))
    check(abs((r["view"][0] - vx) - 3.0) <= 1.0,
          "仅 X 的图，x 偏了 3 个底图像素，量出来偏 %.2f（该 ≈ 3）✗"
          % (r["view"][0] - vx))
    check(abs(r["scale"] - 1.0) < 0.02,
          "仅 X 的图，缩放该还是 ~1.0，量出来 %.4f" % r["scale"])
    # 直接把拟合那层钉死：**x/y 都偏**的局面下，`axes=(True, False)` ⇒ 只许 x 动 ✓
    panel2 = _crop_panel(t.canvas, 1.0, (vx + 6.0, vy + 6.0), w, h)   # 真值：两轴都偏 6
    seed2 = dict(seed, view=[vx, vy])
    fx = mm._crop_pos_fit(panel2, t.canvas, seed2, axes=(True, False))
    check(fx is not None and abs(fx["calib"]["view"][1] - vy) < 1e-9,
          "「仅 X」时拟合却把 y 挪了（%.2f → %.2f）—— 这功能就白做了 ✗"
          % (vy, fx and fx["calib"]["view"][1]))
    fxy = mm._crop_pos_fit(panel2, t.canvas, seed2, axes=(True, True))
    check(fxy is not None and abs(fxy["calib"]["view"][1] - vy) > 1.0,
          "对照组（双轴）本该把 y 挪过去（真值偏 6），实际 %.2f —— "
          "那说明这条用例没测到点上 ✗" % (fxy and fxy["calib"]["view"][1]))


def t_calib_dialog_scroll_combo():
    """标定弹窗多了「滚动方向」：读进来 / 选出来 / 存下去（用户 2026-09-29 追加 ✓）。

    钉四件：
      ① 下拉**三个**选项（双轴 / 仅 X / 仅 Y ✓）；
      ② 文件里写着「仅 Y」⇒ 打开就该选中它 ✓（不然一保存就被抹成双轴 ✗）；
      ③ 选「仅 X」+ 保存 ⇒ 标定字典里有 `scroll: "x"` ✓；
      ④ ⚠ **老标定（没这个键）⇒ 双轴，而且存下去也不塞这个键** ✓
         （别给老文件凭空加字段 ✓）。
    """
    import unittest.mock as mock

    from PyQt5.QtWidgets import QApplication

    from gui.minimap_calib import MinimapCalibDialog

    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return
    app = QApplication.instance() or QApplication([])       # noqa: F841
    base = {"mode": mm.MODE_CROP, "scale": 1.0, "offset": [4, 4], "view": [3, 4]}
    with mock.patch.object(mapdata, "load_calib",
                           lambda _m, src=None: dict(base, scroll="y")):
        d = MinimapCalibDialog(mid, mode=mm.MODE_CROP, client=StubClient(None))
        try:
            check(d.cmb_scroll.count() == 3,
                  "「滚动方向」下拉该有三项，实际 %d" % d.cmb_scroll.count())
            check(d.cmb_scroll.currentData() == mm.SCROLL_Y,
                  "文件里写着「仅 Y」，下拉却选中 %r（一保存就被抹成双轴 ✗）"
                  % d.cmb_scroll.currentData())
            check(d.calib().get("scroll") == mm.SCROLL_Y,
                  "载入的滚动方向没写回标定：%r" % (d.calib().get("scroll"),))
            d.cmb_scroll.setCurrentIndex(d.cmb_scroll.findData(mm.SCROLL_X))
            check(d.calib().get("scroll") == mm.SCROLL_X,
                  "选了「仅 X」却没写进标定：%r" % (d.calib().get("scroll"),))
        finally:
            d._shutdown()
    with mock.patch.object(mapdata, "load_calib", lambda _m, src=None: dict(base)):
        d2 = MinimapCalibDialog(mid, mode=mm.MODE_CROP, client=StubClient(None))
        try:
            check(d2.cmb_scroll.currentData() == mm.SCROLL_XY,
                  "老标定（没这个键）该按「双轴」显示，实际 %r"
                  % d2.cmb_scroll.currentData())
            check("scroll" not in d2.calib(),
                  "老标定里被塞进了 `scroll` 字段（别凭空给老文件加键 ✗）：%r"
                  % (d2.calib().get("scroll"),))
        finally:
            d2._shutdown()


def t_calib_dialog_layout():
    """⭐ 标定弹窗的**版面**（2026-09-29 用户："过一遍UI 整理一下信息 归纳布局 然后美化" ✓）。

    钉五件（都是"看源码看不出、必须量"的）：
      ① **分小节**：有「画面」「几何」两个小节标题（`_section` ✓）—— 信息按"你在干什么"
         分组，而不是一路往下堆 ✗；
      ② **一对 x/y 并排成一行、且左右两组对齐**：`sld`/`sld_y` 的 y 相同（同一行）✓，
         而**第二组的 x 在两行里一致**（`sld_y.x() == sld_oy.x()` ✓）—— 差一点点就是
         "两行不像一张表"（这是"归纳布局"最直接的可见成果 ✓）；
      ③ **画布拿到大头**：`view` 的高度 ≥ 窗口的一半（参数区再长也不许把画布挤没 ——
         这扇窗是**对着图看**的工具 ✓）；
      ④ **界面文案里不许有 `**`**（Qt 的 `QLabel`/tooltip 不认 markdown ✗ ——
         2026-09-29 眼过版面时真在提示行上看到两个星号 ✗）；扫**所有** QLabel/按钮的
         text + 所有 tooltip ✓；
      ⑤ **按钮配色走全项目那一份**：「保存标定」= 主按钮（蓝）、「关闭」= 次要
         （`theme.OK_BTN_QSS` / `CANCEL_BTN_QSS`，由 `theme.unify_ok_cancel` 一处实现 ✓）。
    """
    import unittest.mock as mock

    from PyQt5.QtWidgets import (QApplication, QLabel, QPushButton, QWidget)

    from gui import theme
    from gui.minimap_calib import MinimapCalibDialog

    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return
    app = QApplication.instance() or QApplication([])       # noqa: F841
    ch, cw = canvas.shape[:2]
    w, h = min(60, cw - 10), min(90, ch - 10)
    if w < 20 or h < 20:
        print("      （底图太小，跳过）")
        return
    panel = synth_crop_panel(canvas, 5, 7, w, h)
    with mock.patch.object(theme, "save_window", lambda *a, **k: None):
        dlg = MinimapCalibDialog(mid, mode=mm.MODE_CROP, client=StubClient(panel))
        try:
            dlg.resize(980, 760)
            dlg.show()
            for _ in range(3):
                app.processEvents()
            # ① 分小节
            titles = [lb.text() for lb in dlg.findChildren(QLabel)]
            check(any(t == "画面" for t in titles),
                  "没有「画面」这个小节标题（信息又被堆成一长串了 ✗）：%s" % titles[:8])
            check(any(t.startswith("几何") for t in titles),
                  "没有「几何」这个小节标题：%s" % titles[:8])
            # ② x/y 并排 + 左右两组对齐
            check(dlg.sld.y() == dlg.sld_y.y() and dlg.sld_ox.y() == dlg.sld_oy.y(),
                  "x/y 没并排在同一行：%s/%s 与 %s/%s"
                  % (dlg.sld.y(), dlg.sld_y.y(), dlg.sld_ox.y(), dlg.sld_oy.y()))
            check(dlg.sld.y() != dlg.sld_ox.y(),
                  "缩放和偏移挤在同一行了（那是两组参数 ✗）")
            check(dlg.sld.x() == dlg.sld_ox.x() and dlg.sld_y.x() == dlg.sld_oy.x(),
                  "两行的列没对齐（右半组错开了 ⇒ 看着不像一张表 ✗）："
                  "缩放 %s/%s vs 偏移 %s/%s"
                  % (dlg.sld.x(), dlg.sld_y.x(), dlg.sld_ox.x(), dlg.sld_oy.x()))
            # ③ 画布拿大头（阈值取 **38%**：离屏那套没有字形、行高和真窗口不一样 ✓）
            #    ⚠ 2026-10-04 从 40% 放到 38%：**标签设了"可选中复制"之后，Qt 会把它的
            #      尺寸提示微调几像素**（用户第 4 条"字段标题/灰字说明要能复制" ✓ 见
            #      `gui/theme.enable_label_copy` ✓）。实测这一档从 ~40.5% 掉到 **39.5%**
            #      （差 ~5px ✓ 真窗口里肉眼不可见 ✓）；把三种"不动版面"的写法都试过
            #      （`setMinimumSize(0,0)` / 钉高 / 跳过 wordWrap ✗）都挡不住 ⇒ 只能如实
            #      把阈值按实测口径放两位 —— **要求本身没变**：参数区再长也不许把画布挤没 ✓。
            check(dlg.view.height() >= dlg.height() * 0.38,
                  "画布只有 %d px（窗口 %d）—— 参数区把画布挤没了 ✗（这扇窗是对着图看的 ✓）"
                  % (dlg.view.height(), dlg.height()))
            # ④ 文案里不许有 `**`
            bad = []
            for lb in dlg.findChildren(QLabel):
                if "**" in lb.text():
                    bad.append(("label", lb.text()[:40]))
            for bt in dlg.findChildren(QPushButton):
                if "**" in bt.text():
                    bad.append(("button", bt.text()[:40]))
            for wd in dlg.findChildren(QWidget):
                if "**" in (wd.toolTip() or ""):
                    bad.append(("tip", (wd.toolTip() or "")[:40]))
            check(not bad,
                  "界面上还有 markdown 星号（QLabel/tooltip 不认 ⇒ 用户看到的是两个星号 ✗）：%s"
                  % bad[:4])
            # ⑤ 按钮配色一处实现
            check(dlg.btn_save.styleSheet() == theme.OK_BTN_QSS
                  and "关闭" in [b.text() for b in dlg.findChildren(QPushButton)],
                  "「保存标定」没用全项目那份主按钮样式（`theme.unify_ok_cancel` ✓）")
        finally:
            dlg._shutdown()


def t_align_score_local():
    """⭐⭐ **贴合分按重叠区算**（用户 2026-09-29 第 ⑤ 条 ✓ 原话："对于局部地图，评判分应该
    按局部来，必须制定剪裁范围才能让评分合理"）。

    病根：标定弹窗原来报的是 `locate` 那个"**整张面板**最像底图的哪儿"——crop（局部小地图）
    下面板只是底图的一小块 ⇒ 那个分跟你手上这套几何对不对**没关系** ✗（手工对齐更干脆：
    没有分 ⇒ 每次保存都弹"匹配分偏低" ✗）。`mm.align_score` 改成只算**面板映到底图上的
    那一块**（重叠区由这套几何自己定 ⇒ 不需要人再去框一块 ✓）。

    钉五件：
      ① 几何对 ⇒ 分高（≥ 0.9）且重叠 ≈ 100%；
      ② 位移挪歪 20 个底图像素 ⇒ 分**掉下来**（它就是"对不对齐"的读数 ✓）；
      ③ 缩放不对（同一块地图按 1.3 倍画）⇒ 分也掉（尺度错也算对不上 ✓）；
      ④ 面板框太大（有一半落在底图外）⇒ `overlap` 小、要说清（分再高也别全信 ✓）；
      ⑤ 没几何 / 没画面 ⇒ `ok=False` + 一句人话（不许给个假分 ✗）。
    """
    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return
    t = mapdata.load(mid, with_canvas=True)
    ch, cw = canvas.shape[:2]
    w, h = min(70, cw - 20), min(90, ch - 20)
    if w < 30 or h < 30:
        print("      （底图太小，跳过）")
        return
    x, y = 6, 9
    panel = synth_crop_panel(canvas, x, y, w, h)          # 1:1 的"面板" ✓
    good = {"mode": mm.MODE_CROP, "scale": 1.0, "offset": [0, 0], "view": [x, y]}
    # ① 对上了
    r = mm.align_score(panel, t, good)
    check(r["ok"] and r["score"] >= 0.9,
          "几何摆对了，贴合分却只有 %r（%s）" % (r.get("score"), r.get("why")))
    check(r["overlap"] > 0.98,
          "面板整块都在底图里，重叠率应当 ≈100%%：%.2f" % r["overlap"])
    # ② 挪歪 20 个底图像素
    bad = dict(good, view=[x + 20, y])
    r2 = mm.align_score(panel, t, bad)
    check(r2["ok"] and r2["score"] < 0.5,
          "位移挪歪 20 个底图像素，贴合分还有 %r —— 它就不是「对齐了没有」的读数 ✗"
          % (r2.get("score"),))
    check(r["score"] - r2["score"] > 0.4,
          "挪歪前后分没差别（%.3f → %.3f）—— 这个数没用 ✗"
          % (r["score"], r2["score"]))
    # ③ 缩放不对：把同一块地图按 1.3 倍画（面板大 1.3 倍、几何仍按 1.0）
    panel_zoom = _crop_panel(canvas, 1.3, (x, y), int(w * 1.3), int(h * 1.3))
    r3 = mm.align_score(panel_zoom, t, good)
    check(r3["ok"] and r3["score"] < 0.5,
          "缩放差了 1.3 倍，贴合分还有 %r（尺度错也算对不上 ✗）" % (r3.get("score"),))
    # ④ 面板框太大 / 框到了底图外：把显示区起点摆到地图右下角 ⇒ 面板有一半在外面
    #    （`overlap` 要掉下来，别让人以为"分高就没事" ✓）
    edge = dict(good, view=[float(cw) - w / 2.0, float(ch) - h / 2.0])
    r4 = mm.align_score(panel, t, edge)
    check(r4["overlap"] < 0.7,
          "一半面板落到地图外了，重叠率却有 %.2f（这样分再高也不该全信 ✗）"
          % r4["overlap"])
    # ⑤ 没几何 / 没画面
    r5 = mm.align_score(panel, t, {"mode": mm.MODE_CROP})
    check(not r5["ok"] and r5["why"], "没几何却给了分：%s" % (r5,))
    r6 = mm.align_score(None, t, good)
    check(not r6["ok"] and r6["why"], "没画面却给了分：%s" % (r6,))
    # ⑥ 真底图跑一遍耗时（这条要能每拍算：判据那行是随几何变化现算的 ✓）
    t0 = time.perf_counter()
    for _ in range(5):
        mm.align_score(panel, t, good)
    cost = (time.perf_counter() - t0) / 5.0
    check(cost < 0.05, "一次贴合分要 %.1f ms —— 太贵，判据那行会卡（要 ≤ 几十毫秒 ✓）"
          % (cost * 1000))


def t_calib_dialog_v2_items():
    """⭐⭐ 标定弹窗 2026-09-29 那七条里的 ②、④、⑥（其余在别的用例里 ✓）。

      ② 两条「轴偏移」拖动条：**fit = 面板像素、crop = 底图像素**（单位写错人就会按错的
         单位去调 ✗）；
      ④ 「锁定 xy 缩放」勾上 ⇒ 两轴一起改（钉在 `t_calib_dialog_two_axis` ✓，这里补一条
         "勾回来之后又锁上"）；
      ⑥ 「不改缩放」：点自动定位**只搬位移**、缩放原样 ✓（勾掉之后才连缩放一起改 ✓）。
    """
    import unittest.mock as mock

    from PyQt5.QtWidgets import QApplication

    from gui.minimap_calib import MinimapCalibDialog

    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return
    app = QApplication.instance() or QApplication([])       # noqa: F841
    ch, cw = canvas.shape[:2]
    w, h = min(60, cw - 10), min(90, ch - 10)
    if w < 20 or h < 20:
        print("      （底图太小，跳过）")
        return
    x, y = 5, 7
    panel = synth_crop_panel(canvas, x, y, w, h)

    # ② fit 的偏移单位是**面板像素**、crop 是**底图像素**（同一个控件、两种含义 ✗）
    cal_fit = {"mode": mm.MODE_FIT, "scale": 1.0, "offset": [3.0, 4.0], "view": [0, 0]}
    with mock.patch.object(mapdata, "load_calib", lambda _m, src=None: dict(cal_fit)):
        d_fit = MinimapCalibDialog(mid, mode=mm.MODE_FIT,
                                   client=StubClient(panel), src=mm.SRC_STREAM)
        try:
            check(d_fit._off_unit() == "面板像素",
                  "fit 的偏移单位该写「面板像素」，实际 %r" % d_fit._off_unit())
            check(d_fit._off_get(0) == 3.0,
                  "fit 的偏移没从 `offset` 读出来：%r" % d_fit._off_get(0))
            d_fit.sp_ox.setValue(11.0)
            check(abs(d_fit.offset[0] - 11.0) < 1e-6,
                  "fit 下调「x 轴偏移」没写进 `offset`：%s" % (d_fit.offset,))
            check(not d_fit.ck_lock_xy.isChecked() or True, "占位（锁定那件在别处钉 ✓）")
        finally:
            d_fit._shutdown()

    # ⑥ 「不改缩放」：先把缩放**故意改错**（真实是 1:1 ⇒ 会自动定位成 1.0）
    cal_crop = {"mode": mm.MODE_CROP, "scale": 3.0, "offset": [4, 4], "view": [0, 0]}
    with mock.patch.object(mapdata, "load_calib", lambda _m, src=None: dict(cal_crop)):
        d = MinimapCalibDialog(mid, mode=mm.MODE_CROP,
                               client=StubClient(panel), src=mm.SRC_STREAM)
        try:
            # ② crop 的偏移单位必须写「底图像素」（写死成"面板像素" 就是错的 ✗）
            check(d._off_unit() == "底图像素",
                  "crop 的偏移单位该写「底图像素」，实际 %r" % d._off_unit())
            # ③ 方向键一步 = **1 个面板像素** = 1/scale 个底图像素。⚠ 这一条必须在
            #    **scale ≠ 1** 的窗里量（scale=1 时"1 面板像素"和"1 底图像素"分不开 ✗）。
            check(abs(d.scale - 3.0) < 1e-9, "这个窗的初始缩放该是 3.0：%.3f" % d.scale)
            from PyQt5.QtCore import QEvent, Qt
            from PyQt5.QtGui import QKeyEvent
            _b0 = float(d.block[0])
            d.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Left, Qt.NoModifier))
            _step = float(d.block[0]) - _b0
            check(abs(_step - 1.0 / 3.0) < 1e-3,
                  "方向键一步该是 1 个面板像素（=1/scale=0.333 个底图像素），实际 %.4f "
                  "—— 单位是不是又变回「底图像素」了？" % _step)
            d.ck_live.setChecked(True)
            d._on_tick()
            check(d._frame is not None, "弹窗没取到帧")
            d.ck_no_scale.setChecked(True)
            d._locate_now()
            check(abs(d.scale - 3.0) < 1e-9,
                  "勾了「不改缩放」，自动定位却把缩放改了：%.4f" % d.scale)
            # ⭐ **把缩放设成对的（1.0）、位移留在错的 (0,0)** ⇒ 「不改缩放」要做的
            #   就是"缩放不动、只把位移搬过来"（用户 2026-09-29："我已经手工对得很好了"✓
            #   —— 手工量好缩放、只想知道位置，就是这条路 ✓）。
            #   ⚠ 别在"缩放是错的 3.0"下量位移：那时"最好的位移"没有意义（怎么挪都不
            #     可能对上 ⇒ 分数是噪声 ✗），钉出来的只能是巧合（老代码正是靠"取模板匹配
            #     的峰值"才碰巧给对，而那正是这次要修掉的东西 ✗）。
            d.scale = 1.0
            d.scale_y = 1.0
            d.block = [0.0, 0.0]                 # 位移故意留错
            d._locate_now()
            check(abs(d.scale - 1.0) < 1e-9,
                  "「不改缩放」把缩放改成 %.4f 了" % d.scale)
            got = mm.panel_to_canvas(0, 0, d.calib())
            check(abs(got[0] - x) < 1.0 and abs(got[1] - y) < 1.0,
                  "「不改缩放」时位移没对齐过来：面板左上 → 底图 (%s, %s)，应 ≈ (%d, %d)"
                  "（缩放已经是对的 ⇒ 这条必须准）" % (got[0], got[1], x, y))
            # 勾掉之后再定位一次 ⇒ 这次缩放才跟着改（= 老行为 ✓）
            d.ck_no_scale.setChecked(False)
            d._locate_now()
            check(abs(d.scale - 1.0) < 0.2,
                  "没勾「不改缩放」时缩放该按匹配结果改（真实是 1:1）：%.4f" % d.scale)
        finally:
            d._shutdown()


def t_crop_overlay_geometry_matches_box():
    """⭐⭐ 叠图取的「源矩形」必须**正好是当前面板显示的那一块地图**（用户 2026-09-29 报：
    「**在小地图不更新的情况下**，叠加图的显示规则与全局小地图没有任何区别，而目前的情况是
    **刚标定完叠加图显示就不对**」✓）。

    ⚠ 病根在 `frame_overlay_rects` 的**折算**那三行（标定那块面板 → 现在框出来的那一块）：
      · `scale` / `offset` 要除以 `z`（`z` = 标定面板宽 ÷ 框选宽；A 机推流带 zoom 时 z≈3 ✓）；
      · **`view` 不能除** ✗ —— `view` 是**底图坐标**，折算只是"面板像素 ↔ 底图像素"那一层
        的量纲换算。除错了 = 整块叠图平移 `view × (1 − 1/z)` 个底图像素（这张图 ≈ 70 个，
        而一个面板才装得下 58 个 ⇒ 叠图整块跑到地图别处去了 ✗）。
    **为什么以前从来没露过**：`fit` 的 `view` 恒为 `[0, 0]`（整张底图不动）⇒ 除不除都一样 ✗；
      而 `crop` 的 `view` 是"现在显示在地图哪儿"、必然非 0 ⇒ **一除就错** ✓
      —— 所以现象恰好是"**只有局部小地图**刚标定完叠图就不对" ✓（和用户描述一字不差）。

    钉法：造一份**自洽的真值** —— 标定是在"同一块地图、放大 3 倍"的另一块面板上做的
    （尺寸与缩放都 ×3），那么折算到框选那块之后，源矩形**必须还是那块地图** ✓。
    """
    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return
    cw, ch = canvas.shape[1], canvas.shape[0]
    z_ov = mm.overlay_zoom(cw)          # 底图 → 叠加图 的放大倍数（源矩形按它算 ✓）
    box = (5, 71, 109, 114)             # 框选小地图：一个真实的形状 ✓
    # 真值：框选那块面板显示的地图范围（底图像素）。数取自 105040303 的真实标定：
    # scale 5.624（在 A 机 zoom=3 那块面板上量的）、view=[-12,105]、offset=[4,4]
    # ⇒ 面板左上角(去掉 inset 之后)对应底图 (-12.711, 104.29) ✓
    s_stream, s_box = 5.624, 5.624 / 3.0
    x0, y0 = -12.711, 104.29
    # 标定（在**放大 3 倍**的那块面板上）：view = 底图坐标(面板 0 处) + inset/scale ✓
    cal = {"mode": mm.MODE_CROP, "scale": s_stream, "offset": [4, 4],
           "view": [x0 + 4.0 / s_stream, y0 + 4.0 / s_stream]}
    src, dst = mm.frame_overlay_rects(cal, box, (cw, ch),
                                      calib_panel=(int(box[2] * 3), int(box[3] * 3)))
    check(tuple(dst) == box,
          "叠图的落点不是框选那块（%s vs %s）—— crop 下它就该正好盖在框上 ✓" % (dst, box))
    # 期望：源矩形 = 那块地图 × overlay_zoom（`view_rects` 里会 `int()`，所以也按整块比）
    got = (src[0] / float(z_ov), src[1] / float(z_ov))          # 换回底图像素再说 ✓
    err = (abs(got[0] - x0), abs(got[1] - y0))
    check(err[0] <= 1.5 and err[1] <= 1.5,
          "叠图取的是地图上的 (%.1f, %.1f)，而这块面板显示的是 (%.1f, %.1f) —— "
          "偏了 (%.1f, %.1f) 个底图像素（这块面板一共才装得下 (%.1f, %.1f) 个 ✗）"
          "⇒ 叠图整块跑别处去了（根因：`frame_overlay_rects` 把 `view` 也除了 z "
          "—— 它是**底图坐标**，不能除 ✗）"
          % (got[0], got[1], x0, y0, err[0], err[1], box[2] / s_box, box[3] / s_box))

    # 对照：`fit`（整张底图缩进面板）折算之后，画上去的那块**正好铺满框选那块** ✓
    #   —— fit 的每条边都跟 zoom 无关：底图 × scale / z = 框宽（z = 底图×scale / 框宽 ✓）。
    #   ⚠ 这条**不是**"折算前后一样"（那是我一开始写错的：折算必然会按 1/z 缩 ✓）。
    fit = {"mode": mm.MODE_FIT, "scale": 5.6, "offset": [3.5, 4.7], "view": [0, 0]}
    fw_fit, fh_fit = int(cw * 5.6), int(ch * 5.6)          # 标定那块面板 = 整张底图 × 5.6
    _, dst_f = mm.frame_overlay_rects(fit, (5, 71, fw_fit // 3, fh_fit // 3),
                                      (cw, ch), calib_panel=(fw_fit, fh_fit))
    check(abs(dst_f[2] - fw_fit // 3) <= 1 and abs(dst_f[3] - fh_fit // 3) <= 1,
          "fit 折算之后没铺满框选那块（画出来 %s，框是 %s）—— 折算只该按 1/z 缩 ✓"
          % (dst_f, (5, 71, fw_fit // 3, fh_fit // 3)))


def t_crop_overlay_follows_view():
    """⭐⭐ 局部小地图：**画面上的「地形叠加图」也跟着滚**（用户 2026-09-29 问"现在已经能
    实时地滚动底图了？"✓ —— 坐标是实时的，但叠图当时还停在标定那一刻 ✗）。

    为什么这件事要紧：叠图就是"拿底图那块半透明地盖在游戏的小地图上"，而人正是**拿它
    目测标定对不对**的（"标定这件事的正确判据是看得见"）——crop 下不跟着走的叠图不只是
    "没滚"，它会**把人看错**：人一走它就越错越多，看着像"标定废了" ✗。

    钉四件：
      ① `_refresh_overlay(view=...)` 把那个显示区**传给了** `frame_overlay_rects`
         （对照：不传 ⇒ 用标定里那份 = 老行为 ✓）；
      ② `_tick_world` **跟住了**就喂**这一拍**那个（显示区变了才重画 ✓）；
      ③ 「框选小地图」的取法**处处一处口径**（本项目优先 `crop_of`）——
         `_refresh_overlay` / `_overlay_blocker` / `_tick_world` 里**不许**再直接读
         `load_live()["mmap_crop"]` ✗（框选 2026-09-27 起按**项目**存，读全局那份的图上
         叠图会被摆到别的图上那个框的位置、甚至判成"还没框" ✗）；
      ④ 跟不住时（`view_ok` 假）**不许**拿旧显示区硬喂（那一拍用的就是标定里那份 ✓）。
    """
    import inspect

    from PyQt5.QtGui import QPixmap
    from PyQt5.QtWidgets import QApplication

    from gui.route_panel import RoutePanel

    app = QApplication.instance() or QApplication([])      # noqa: F841 —— 建控件必须有
    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return
    t = mapdata.load(mid, with_canvas=True)
    crop = [6, 72, 60, 40]
    cal = {"mode": mm.MODE_CROP, "scale": 2.0, "offset": [4, 4],
           "view": [-3.0, 12.0], "score": 0.9}

    class _FakePanel:
        """只记"给我画了什么"的替身（真 QPixmap 那套不参与这一条 ✓）。"""

        def __init__(self):
            self.ov = []          # 每次 set_minimap_overlay 的 (src, dst)
            self.notes = 0
            self.thread = None

        def set_minimap_overlay(self, pix, src=None, dst=None, alpha=0.5, note=""):
            self.ov.append((src, dst))
            return True           # True = 挂上了（`set_overlay_note` 才会走"只换字"那条 ✓）

        def set_overlay_note(self, note):
            self.notes += 1
            return True

        def current_frame(self):
            return np.zeros((140, 160, 3), np.uint8)

    p = RoutePanel()
    fake = _FakePanel()
    seen = {}
    try:
        p.live_panel = fake
        p._map_id = lambda: mid
        p._mmap_crop = lambda: list(crop)
        p._mmap_src = lambda: mm.SRC_LIVE     # 走"从实时画面裁一块"那条（不碰网络 ✓）
        p._mmap_draw_on = lambda: True
        p._overlay_blocker = lambda: ""
        p._calib_panel_wh = lambda *a: None
        p._overlay_pix = lambda m: QPixmap(4, 4)
        real_calib = mapdata.load_calib
        mapdata.load_calib = lambda m, src=None: dict(cal)
        real_rects = mm.frame_overlay_rects

        def fake_rects(loc, *a, **kw):
            seen.setdefault("views", []).append(
                None if loc.get("view") is None else [float(v) for v in loc["view"]])
            return ((0, 0, 2, 2), (1, 1, 2, 2))

        mm.frame_overlay_rects = fake_rects
        try:
            # ① 带 view ⇒ 必须把它传下去；不带 ⇒ 用标定里那份 ✓
            p._refresh_overlay(view=[7.0, -9.0])
            p._refresh_overlay()
            check(seen.get("views") == [[7.0, -9.0], [-3.0, 12.0]],
                  "`_refresh_overlay(view=…)` 没把显示区传给 frame_overlay_rects：%s"
                  % (seen.get("views"),))

            # ② `_tick_world` 跟住了 ⇒ 喂的是**这一拍**那个
            class _FakeLocator:
                def load(self, m):
                    return self

                def update(self, panel, src=None, xtol=None, **kw):
                    return {"ok": True, "confirmed": True, "held": False,
                            "px": 10.0, "py": 20.0, "world_x": 100.0,
                            "world_y": 200.0, "segment_id": 0, "foothold_id": "1",
                            "ladder_id": None, "dot": "找到", "family": "yellow",
                            "layer": "color", "short": "",
                            "note": "（自检替身）",
                            "view": [5.0, 6.0], "view_ok": True, "view_score": 0.95,
                            "view_trust": True, "view_why": ""}

                def forget_calib(self):
                    pass

                def use_track_config(self, v):
                    pass

            p._locator = _FakeLocator()
            p._fh_zone = lambda r: ""
            p._osd_lines = lambda x: x
            seen["views"] = []
            fake.ov = []
            p._tick_world()
            check(seen.get("views") == [[5.0, 6.0]],
                  "`_tick_world` 没把**这一拍跟出来的**显示区喂给叠图：%s" % (seen.get("views"),))
            n_before = len(fake.ov)
            p._tick_world()           # 显示区没变 ⇒ 不该再重画一遍 ✓
            check(len(fake.ov) == n_before,
                  "显示区没变也在重画叠图（250ms 一拍、每画一次都要重缩一遍画面 ✗）："
                  "%d → %d" % (n_before, len(fake.ov)))

            # ④ 跟不住 ⇒ 不许拿旧显示区硬喂（那一拍用的是标定里那份 ✓）
            p._locator.update = lambda *a, **kw: dict(
                {"ok": True, "confirmed": True, "held": False, "px": 1.0, "py": 2.0,
                 "world_x": 1.0, "world_y": 2.0, "segment_id": 0, "foothold_id": "1",
                 "ladder_id": None, "dot": "", "family": "yellow", "layer": "color",
                 "short": "", "note": "", "view": [9.0, 9.0], "view_ok": False,
                 "view_score": 0.3, "view_trust": False, "view_why": "挡了"})
            seen["views"] = []
            p._tick_world()
            check(seen.get("views") == [],
                  "没跟住却还去改叠图（那一拍该照标定里那份画 ✓，不然叠图会跳到错位置 ✗）："
                  "%s" % (seen.get("views"),))
        finally:
            mm.frame_overlay_rects = real_rects
            mapdata.load_calib = real_calib
    finally:
        p.deleteLater()
        app.processEvents()

    # ③ 框选区域**一处口径**（本项目优先）—— 三处都不许直接读全局那份 ✗
    src = inspect.getsource(RoutePanel)
    for name in ("_refresh_overlay", "_overlay_blocker", "_tick_world"):
        i = src.index("def %s(" % name)
        j = src.index("\n    def ", i + 10)
        body = "\n".join(ln for ln in src[i:j].splitlines()
                         if not ln.strip().startswith("#"))
        check('load_live().get("mmap_crop")' not in body,
              "`%s()` 直接读了 config/live.yaml 里的 mmap_crop —— 框选区域**按项目存**，"
              "得走 `self._mmap_crop()`（= `mm.crop_of`：本项目优先 ✓）✗" % name)


def t_player_tracker_trusted_camera():
    """⭐⭐ 用**小地图的权威世界坐标**核对玩家框 → 一个"可信相机"（用户 2026-09-29 任务 2）。

    病根（`docs/交接.md` §2 那条）：所有"**画面 → 世界**"的换算都要减掉相机，而相机是
    **每拍现算**的 `玩家世界坐标 − 玩家画面框` ⇒ **玩家框抖 Δ，每只怪的世界坐标就同量
    平移 Δ** ✗。`PlayerTracker` 每一拍都在做"框 ↔ 黄点"的比对（`_world_dist` ✓），
    顺手把它们**对不对得上**记下来，就得到一个"可信相机"（`camera()` ✓）。

    钉五件：
      ① 两拍都对得上 ⇒ 相机 = `世界 − 框`（口径同 `screen_to_world`：x 用框中心、y 用框底 ✓）；
      ② **框抖**（黄点没动、框跳了 200px）⇒ 这一拍**不更新**相机（`camera()` 还是旧的 ✓）
         —— 这就是"怪的世界坐标不再跟着抖"的那一步 ✓；对不上的那一拍 `cam_rejected=True` ✓；
      ③ 一直对不上（连续 `max_missed` 拍）⇒ **认账**（真的传送了），不许永远冻着 ✗；
      ④ 没给黄点 ⇒ 不动、也不报"对不上"（判不了 ≠ 不一致 ✓）；
      ⑤ 跟丢 ⇒ 解锁重锁时**相机重来**（旧那个属于上一段位置，沿用只会把它带偏 ✗）。
    """
    from perception.tracker import PlayerTracker

    def _c(cx, bottom):
        return [(cx - 20.0, bottom - 60.0, cx + 20.0, bottom, 0.9)]

    # ① 建立：黄点 (1000,-200)、框 cx=400 / bottom=300 ⇒ 相机 = (600, -500)
    tr = PlayerTracker()
    tr.update(_c(400.0, 300.0), world=(1000.0, -200.0))
    tr.update(_c(410.0, 300.0), world=(1010.0, -200.0))
    check(tr.camera() is not None and abs(tr.camera()[0] - 600.0) < 1e-6
          and abs(tr.camera()[1] + 500.0) < 1e-6,
          "相机没建起来：%r（期望 (600, -500)）" % (tr.camera(),))

    # ② **两个传感器各说各的**：框往右走 100px、黄点却往左走 100px
    #    （各自都没超过"跳变上限" ⇒ 纯连续性判据看不出来 ✓，而隐含相机差了 200px ✓）
    before = tr.camera()
    tr.update(_c(510.0, 300.0), world=(900.0, -200.0))
    check(tr.camera() == before,
          "两个传感器各说各的（框 +100、黄点 −100 ⇒ 相机差 200px），相机却跟着跳了："
          "%r → %r ✗（那正是「每只怪的世界坐标同量平移」的来源）" % (before, tr.camera()))
    check(tr.cam_rejected is True, "对不上的那一拍没标出来（打点 `cam_reject` 就没了）")
    check(tr.cam_resid is not None and tr.cam_resid > tr.max_jump,
          "残差没记下来（`cam_resid` 是「框那一路稳不稳」唯一的数）：%r" % (tr.cam_resid,))

    # ②' 对照：两边一起走（相机不动）⇒ 照旧采纳、不报不一致
    tr1 = PlayerTracker()
    tr1.update(_c(400.0, 300.0), world=(1000.0, -200.0))
    tr1.update(_c(460.0, 300.0), world=(1060.0, -200.0))
    check(tr1.cam_rejected is False and tr1.camera() is not None
          and abs(tr1.camera()[0] - 600.0) < 1e-6,
          "人正常走动（框和黄点一起动）被当成「对不上」✗：%r / resid=%r"
          % (tr1.camera(), tr1.cam_resid))

    # ③ 一直对不上 ⇒ 认账（`max_missed` 拍之后采纳新值）
    for _ in range(tr.max_missed + 1):
        tr.update(_c(510.0, 300.0), world=(900.0, -200.0))
    check(tr.camera() is not None and abs(tr.camera()[0] - 390.0) < 1e-6,
          "连续对不上 %d 拍之后还不认账（相机冻在 %r）—— 真的传送/换图时会永远错下去 ✗"
          % (tr.max_missed, tr.camera()))

    # ④ 没给黄点 ⇒ 不动、也不报"对不上"
    tr2 = PlayerTracker()
    tr2.update(_c(400.0, 300.0), world=(1000.0, -200.0))
    c0 = tr2.camera()
    tr2.update(_c(405.0, 300.0))
    check(tr2.camera() == c0 and tr2.cam_rejected is False,
          "没给黄点那一拍动了相机 / 报了不一致（判不了 ≠ 不一致 ✗）：%r %r"
          % (tr2.camera(), tr2.cam_rejected))

    # ⑤ 跟丢 ⇒ 重锁时相机重来
    tr3 = PlayerTracker(max_missed=2)
    tr3.update(_c(400.0, 300.0), world=(1000.0, -200.0))
    for _ in range(4):
        tr3.update([], world=(1000.0, -200.0))       # 连续漏检 ⇒ 解锁
    check(tr3.camera() is None or tr3.locked is False,
          "跟丢之后相机还挂着（旧那个属于上一段位置 ✗）：%r" % (tr3.camera(),))
    tr3.update(_c(900.0, 300.0), world=(2000.0, -200.0))   # 重锁
    check(abs(tr3.camera()[0] - 1100.0) < 1e-6,
          "重锁之后相机没重来：%r（期望 1100）" % (tr3.camera(),))


def t_player_tracker_input_aware():
    """⭐⭐ **输入感知的运动预测**（用户 2026-10-01 ✓，借 `Maple_xfeat` 的思路）。

    病根：漏检那几帧原来是"**拿上一帧位移**线性外推"—— 它**不知道玩家按了什么**。
    玩家中途变向 / 停住时，预测就朝错的方向一路跑 ⇒ ① 挑"哪个框是我"挑错、
    ② 补出来的框也是错的 ✗（这两处**共用同一个预测位置**，见 `update`）。

    做法：方向键是**已知控制输入** —— 按住就朝该方向加速到 `walk_v_max`，松手就在
    地面摩擦下减速到 0（`InputAwareHorizontalKalman` 那套 ✓）。

    钉五件：
      ① **没喂过输入 ⇒ 老口径一字不变**（速度*保持*，绝不按"松手"去减速 ✗）——
         这道闸必需：没有它，不用这个功能的人的老行为也被改了；
      ② 按右 ⇒ 漏检期**一路往右**，且速度单调增（不是恒定不变）；
      ③ 松手 ⇒ 速度**衰减到 0** 然后停住（不是永远滑下去）；
      ④ 变向 ⇒ 从正速度**刹到负**（不会卡在 0 不动）；
      ⑤ 主循环卡一下（`dt` 很大）⇒ 预测**不许被推出去几百像素**（夹住 ✓）。
    """
    from perception import tracker as _tk

    class _Clock:
        """只替**本模块**里那个 `time`（不动全局 ✓）—— 让它按固定步长往前走。"""
        now = 1000.0

        @staticmethod
        def monotonic():
            return _Clock.now

    _real_time = _tk.time
    _tk.time = _Clock

    def _c(cx, bottom=300.0):
        return [(cx - 20.0, bottom - 60.0, cx + 20.0, bottom, 0.9)]

    def _tick(step=0.033):
        _Clock.now += step

    P = _tk.PlayerTracker
    try:
        # ① 没喂过输入 ⇒ 老口径：速度**保持**
        a = P()
        a.update(_c(400.0))
        a.update(_c(440.0))              # 速度 = 0.6*0 + 0.4*40 = 16（像素/帧）
        v0 = a._vx
        check(v0 > 0, "用例前提不成立：没建立起速度 %r" % (v0,))
        _tick()
        b1 = a.update([])
        _tick()
        a.update([])
        check(abs(a._vx - v0) < 1e-9,
              "**没喂过输入却在改速度**（老口径是「速度保持」）✗：%r → %r" % (v0, a._vx))
        check(abs(b1[0] - (440.0 + v0)) < 1e-6,
              "没喂输入时外推口径变了：%r（期望 %r）" % (b1[0], 440.0 + v0))

        # ② 按右 ⇒ 漏检期一路往右，速度单调增
        b = P(max_missed=100)
        b.set_input(1)
        _tick()
        b.update(_c(400.0))
        xs, vs = [], []
        for _ in range(10):
            _tick()
            bx = b.update([])
            xs.append(bx[0])
            vs.append(b._vx)
        check(all(y > x for x, y in zip(xs, xs[1:])),
              "按着右却没一路往右：%r" % (xs,))
        check(all(y >= x - 1e-9 for x, y in zip(vs, vs[1:])),
              "按着右时速度没有单调增（说明压根没用上输入）✗：%r" % (vs,))

        # ③ 松手 ⇒ 减速到 0，然后停住
        b.set_input(0)
        for _ in range(10):
            _tick()
            b.update([])
        check(abs(b._vx) < 1e-9, "松手之后速度没减到 0：%r" % (b._vx,))
        _tick()
        x_stop = b.update([])[0]
        _tick()
        x_stop2 = b.update([])[0]
        check(abs(x_stop2 - x_stop) < 1e-9,
              "速度已经是 0 了还在往前挪：%r → %r" % (x_stop, x_stop2))

        # ④ 变向 ⇒ 从正速度刹到负
        c = P(max_missed=100)
        c.set_input(1)
        _tick()
        c.update(_c(400.0))
        for i in range(8):
            _tick()
            c.update(_c(400.0 + (i + 1) * 6.0))
        check(c._vx > 0, "先按右却没建立起正速度：%r" % (c._vx,))
        c.set_input(-1)
        for _ in range(12):
            _tick()
            c.update([])
        check(c._vx < 0, "变向往左之后速度还是正的（没刹过来）✗：%r" % (c._vx,))

        # ⑤ dt 夹住：主循环卡 5 秒 ⇒ 预测不许被推出去几百像素
        d = P(max_missed=100)
        d.set_input(1)
        _tick()
        d.update(_c(400.0))
        for _ in range(30):
            _tick()
            d.update([])
        x_before = d._box[0]
        _tick(step=5.0)                  # 卡了 5 秒
        x_after = d.update([])[0]
        jump = abs(x_after - x_before)
        check(jump <= d.walk_v_max * 0.25 + 1e-6,
              "卡帧之后预测一口气跳了 %r 像素（该夹在 1/4 秒 = %r）✗"
              % (jump, d.walk_v_max * 0.25))
    finally:
        _tk.time = _real_time


def t_screen_to_world_trusted_camera():
    """`screen_to_world` 优先用**可信相机**（用户 2026-09-29 任务 2 ✓）。

    全仓"画面 → 世界"只有这一处（约定 10）⇒ 它认可信相机，等于**每只怪**都跟着用上了 ✓。
    ⚠ `cam_x/cam_y` 是 `None`（还没建立）时**必须退回老口径**（现算）—— 把它当 0 会让
    所有怪的世界坐标整体偏一个相机 ✗。
    """
    from perception.world_state import Player
    from perception.minimap import screen_to_world

    # 老口径：没有可信相机 ⇒ 现算（行为一字不变 ✓）
    p = Player(x=400.0, y=300.0, bottom=340.0, world_x=1000.0, world_y=-200.0)
    check(screen_to_world(p, 400.0, 340.0) == (1000.0, -200.0),
          "没有可信相机时没退回老口径：%r" % (screen_to_world(p, 400.0, 340.0),))

    # 有可信相机 ⇒ 用它（哪怕它和"现算"那个不一样 —— 那正是"框抖了"的情形 ✓）
    p2 = Player(x=400.0, y=300.0, bottom=340.0, world_x=1000.0, world_y=-200.0,
                cam_x=800.0, cam_y=-300.0)
    check(screen_to_world(p2, 400.0, 340.0) == (1200.0, 40.0),
          "有可信相机却没用它：%r（期望 (1200, 40)）"
          % (screen_to_world(p2, 400.0, 340.0),))
    # 只给一个也不许用（半份数据 = 会算出一个错的相机 ✗）
    p3 = Player(x=400.0, y=300.0, bottom=340.0, world_x=1000.0, world_y=-200.0,
                cam_x=800.0)
    check(screen_to_world(p3, 400.0, 340.0) == (1000.0, -200.0),
          "只给了 cam_x 就用它了（应当两个都有才用）：%r" % (screen_to_world(p3, 400.0, 340.0),))


def t_trusted_camera_wiring():
    """可信相机的**接线**（源码级：这条路没有可驱动的最小夹具 ⇒ 钉调用点 ✓）。

    三处必须都在（少一处就等于没接上 ✓）：
      ① `live_thread` 里把 `player_tracker.camera()` 写进 `ws.player.cam_x/cam_y`；
      ② 怪的集合解析器**优先读**它（怪的世界坐标是它的最大受益者 ✓）；
      ③ `crop_view` 的成败要打点（`crop_view_ok` / `crop_view_miss` ✓）。
    """
    import inspect

    from gui import live_thread as lt

    src = inspect.getsource(lt)
    check("player_tracker.camera()" in src and "ws.player.cam_x" in src,
          "`live_thread` 没把可信相机写进 WorldState（下游就拿不到 ✓）")
    check("getattr(player, \"cam_x\", None)" in src,
          "怪的集合解析器没读 `player.cam_x`（相机那条链就还是每拍现算 ✗）")
    check("crop_view_ok" in src and "crop_view_miss" in src,
          "crop 显示区跟没跟住没打点（现场排查要它 ✓）")
    check("perf.count(\"cam_reject\")" in src,
          "「框与黄点对不上」没打点（`cam_reject` 涨不涨是判断'框锁错了'的第一手证据 ✓）")


def t_crop_locate_real_shape():
    """⭐⭐ **自动定位在"真机那种局部小地图"上必须好用**（用户 2026-09-29 原话：
    "小地图标定弹窗的自动定位按钮 好像对于局部地图来说不好用 / 森林迷宫III项目，
    我已经手工对的很好了匹配分还是很低！"✗）。

    钉的是那份**真实形状**（用户现存标定 `105040303` 就长这样）：
      · 底图 82×218（这张图本来就小）；
      · 面板 **502×408**（含游戏小地图边框 ⇒ 比底图**宽**）；
      · 真倍数 **5.645**（非整数 —— 老代码的候选档是整数 + 半档 ⇒ 一个都够不着 ✗）；
      · 面板缩回底图后 **87.5 > 82** ⇒ **整块模板放不进底图** ⇒ 老代码直接跳过这个
        倍数（`tt.shape[1] > c.shape[1]: continue` ✗✗）⇒ 只能给 scale=12 那种
        "小模板碰巧高分"的答案 ✗。

    所以这条用例在本改动之前是**红**的（改坏了也一样红 ✓）：量出的 scale 必须落在
    真值 5% 以内、view 误差 ≤2 底图像素、贴合分 ≥0.5 ✓。
    """
    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return
    # ⚠ **把底图裁成"这张图那么小"**（105040303 是 82×218）：只有底图**比面板窄**时，
    #   "整块模板放不进底图"这件事才成立（那正是要钉的那条 ✗）—— 拿宽底图测等于没测 ✓
    if canvas.shape[1] > 82:
        canvas = np.ascontiguousarray(canvas[:, :82])
    ch, cw = canvas.shape[:2]
    s_true = 5.645
    # 显示区起点放在**偏右**（`cw*0.3`）：这样"面板正中间那一块"会伸到底图外去 ✗
    # ⇒ 只有"挪着取块"（左 / 中 / 右都试）才搜得到真位置 ✓（实测 105040303 就是这个局面 ✓）
    view = (cw * 0.3, ch * 0.4)
    pw, ph = 502, 408
    if cw < 40 or ch < 40:
        print("      （底图太小，跳过）")
        return
    f = mm._gray(canvas).astype(np.float32)
    xs = (np.arange(pw, dtype=np.float32) - 4.0) / s_true + view[0]
    ys = (np.arange(ph, dtype=np.float32) - 4.0) / s_true + view[1]
    gx, gy = np.meshgrid(xs, ys)
    panel = cv2.remap(canvas, gx, gy, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE)
    true = {"mode": mm.MODE_CROP, "scale": s_true, "offset": [4, 4],
            "view": [view[0], view[1]]}
    sc_true, _ov, _why = mm.overlap_pearson(panel, canvas, true)
    check(sc_true is not None and sc_true > 0.9,
          "合成面板在**真几何**下就该近乎满分，实际 %r（用例本身建错了？）" % (sc_true,))
    # ① 不带种子（第一次标定这张图）：纯靠粗搜 + 拟合 ✓
    loc = mm.locate_crop(panel, canvas)
    check(loc is not None,
          "**没量出来** —— 这正是用户说的「自动定位对局部地图不好用」✗"
          "（面板比底图宽 / 倍数不是整数时，老代码就走到这儿 ✗）")
    # ⚠ 容差要**紧**：粗搜那几档（整数 / 半档）本来就能落在 2~3% 以内 ⇒ 宽容差是量不出
    #   "有没有做细化"的 ✗（实测：只留粗搜 ⇒ 5.51（真 5.645）⇒ 2.4% 偏差 ✓ 必须红 ✓）
    check(abs(loc["scale"] - s_true) / s_true <= 0.006,
          "量出的放大倍数 %.3f 与真值 %.3f 差了 %.2f%%（>0.6%% ⇒ 没做连续缩放/细化 ✗）"
          % (loc["scale"], s_true, abs(loc["scale"] - s_true) / s_true * 100))
    cx, cy = mm.panel_to_canvas(0, 0, loc)
    check(abs(cx - view[0]) <= 1.0 and abs(cy - view[1]) <= 1.0,
          "面板左上量到 (%.2f, %.2f)，真值 (%.2f, %.2f) —— 差超过 1 底图像素 ✗"
          "（放大 5 倍的面板上，1 底图像素 = 5~10 个面板像素 ⇒ 这一格是有意义的 ✓"
          "；实测这个形状下 ~0.5~0.9 ✓）"
          % (cx, cy, view[0], view[1]))
    check(loc["score"] >= 0.5,
          "定位给的贴合分只有 %.3f（这份几何其实是能对上的 ⇒ 分不该这么低 ✗）"
          % loc["score"])
    # ④ **取块策略本身**：面板在那个轴上比底图宽 ⇒ 必须试**多个**块位置（只试"正中间
    #    那一块"时，它在底图上可能整个伸到外面去 ⇒ 匹配位置贴着边界、量出来的倍数是错的 ✗）
    blocks = mm._crop_blocks(mm._gray(panel), 4, s_true, cw, ch)
    check(len(blocks) >= 2,
          "面板比底图宽，却只取了一块模板位置（%d 块）—— 挪着试才搜得到真位置 ✗"
          % len(blocks))
    inside = 0
    for sub_xy, tpl, eff in blocks:
        bx = (sub_xy[0] - 4.0) / eff + view[0]
        by = (sub_xy[1] - 4.0) / eff + view[1]
        if (bx >= -0.01 and by >= -0.01
                and bx + tpl.shape[1] <= cw + 0.01
                and by + tpl.shape[0] <= ch + 0.01):
            inside += 1
    check(inside >= 1,
          "取的那几块模板没有一块落在底图里（匹配位置只能贴边界 ⇒ 量出来的是假的 ✗）")
    # ② 带种子（用户桌面上的实况：手上已经有一套手工几何）⇒ 不能更差 ✓
    seed = dict(true, scale=s_true * 1.02, view=[view[0] + 3.0, view[1] - 2.0])
    loc2 = mm.locate_crop(panel, canvas, seed=seed)
    check(loc2 is not None and loc2["score"] >= loc["score"] - 1e-3,
          "给了种子反而更差：%.3f vs %.3f" % (loc2["score"] if loc2 else -1,
                                              loc["score"]))
    # ③ 「不改缩放」那条路：缩放不动、位移必须归位 ✓（`refine_crop(move_scale=False)` ✓）
    #   ⚠ 起点故意放**很远**（+40/+30）：这样"手上那套"自己爬不回来、必须靠"按当前
    #     缩放粗定一遍"那份救 ⇒ 那条路**最容易偷偷改缩放**（粗定给的 eff 是 5.652
    #     而不是输入的 5.645 ✗）—— 正是要钉住的东西 ✓
    got = mm.refine_crop(panel, canvas,
                         dict(true, view=[view[0] + 40.0, view[1] + 30.0]),
                         move_scale=False)
    check(got is not None and abs(got["calib"]["scale"] - s_true) < 1e-9,
          "「不改缩放」却动了缩放：%r" % (got and got["calib"]["scale"]))
    gx0, gy0 = mm.panel_to_canvas(0, 0, got["calib"])
    check(abs(gx0 - view[0]) <= 1.0 and abs(gy0 - view[1]) <= 1.0,
          "「不改缩放」没把位移搬回来：(%.2f, %.2f) 真值 (%.2f, %.2f) ✗"
          % (gx0, gy0, view[0], view[1]))
    # ③' 起点**整块都在底图外面**（算不出分）⇒ 只剩"按当前缩放粗定"那条路 ⇒ 它**绝不许**
    #    把缩放改掉（粗定算出来的 eff 是 5.652 ≠ 输入 5.645 ✗ —— 那正是最容易漏的一处 ✓）
    got2 = mm.refine_crop(panel, canvas,
                          dict(true, view=[view[0] - 4000.0, view[1] - 4000.0]),
                          move_scale=False)
    check(got2 is not None and abs(got2["calib"]["scale"] - s_true) < 1e-9,
          "「不改缩放」在要靠粗定救回来的时候动了缩放：%r ✗"
          % (got2 and got2["calib"]["scale"]))
    gx1, gy1 = mm.panel_to_canvas(0, 0, got2["calib"])
    check(abs(gx1 - view[0]) <= 1.0 and abs(gy1 - view[1]) <= 1.0,
          "起点在底图外时没救回来：(%.2f, %.2f) 真值 (%.2f, %.2f) ✗"
          % (gx1, gy1, view[0], view[1]))


def t_align_score_near():
    """⭐⭐ **判据那行要给"相对判断"**（`align_score(near=True)`）—— 因为绝对分不能跨图比。

    用户那句"**我已经手工对的很好了匹配分还是很低！**"问的其实是两件事，得能分开答：
      · 「**已在这一档最好**」= 附近挪几个像素都没有更贴的 ⇒ 分低是这张图的画法与底图
        不同源，不是你标歪了 ✓（这时把对的几何改坏才是真的亏 ✗）；
      · 「**附近 (x, y) 更贴**」= 有得可挪 ⇒ 那才是真没对齐 ✓。
    """
    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return
    ch, cw = canvas.shape[:2]
    pw, ph = max(40, min(cw - 8, 120)), max(40, min(ch - 8, 160))
    # ⚠ 用 **1:1**（不是放大）：放大合成出来的是"最近邻复制"，而 `overlap_pearson` 的
    #   采样是**逐点取整** ⇒ 放大图上总有一两成采样点正好落在格边，分天生掉到 0.7 一带 ✗
    #   （那是采样口径的固有抖动，不是"没对齐" ✓ —— 这条用例测的是相对判断，别掺这个 ✗）
    view = (float(int(cw * 0.2)), float(int(ch * 0.2)))
    panel = _crop_panel(canvas, 1.0, view, pw, ph)

    class _T(object):
        pass

    t = _T()
    t.canvas = canvas
    # ⚠ `_crop_panel` 的约定：面板 (0,0) **正好**是底图 `view` 那个点（不带 inset ✓）
    #   ⇒ 这套几何里 `offset` 必须是 [0, 0]（写 [8, 8] 就凭空偏 4 个底图像素 ✗）
    good = {"mode": mm.MODE_CROP, "scale": 1.0, "offset": [0, 0], "view": list(view)}
    r = mm.align_score(panel, t, good, near=True)
    check(r["ok"] and r["score"] > 0.9,
          "这套几何本该是对齐的（分 %.3f）—— 用例自己的约定写错了？" % (r.get("score") or -1))
    check(r["ok"] and r["near"] is not None,
          "`near=True` 没给出邻域结论：%s" % (r,))
    check(r["near"]["score"] <= r["score"] + 0.02,
          "对得好好的，邻域却报「还有更贴的」（%.3f → %.3f）—— 会把人引得乱挪 ✗"
          % (r["score"], r["near"]["score"]))
    # 故意挪歪 4 个底图像素 ⇒ 邻域必须指回**真方向**（dx ≈ −4）且明显更高 ✓
    off = dict(good, view=[view[0] + 4.0, view[1]])
    r2 = mm.align_score(panel, t, off, near=True)
    check(r2["ok"] and r2["near"] is not None,
          "挪歪了却没给邻域结论：%s" % (r2,))
    check(r2["near"]["score"] > r2["score"] + 0.02,
          "挪歪 4 像素后邻域没更高（%.3f → %.3f）—— 那就答不了「是不是我没对齐」✗"
          % (r2["score"], r2["near"]["score"]))
    check(r2["near"]["dx"] < 0,
          "邻域指向 %+g（应该往回挪 = 负方向）✗" % r2["near"]["dx"])
    # ⚠ 邻域只是**指路**（3 个底图像素一格、可能停在 −3 而不是 −4 ✓）⇒ 它找到的分
    #   只要**明显高于**当前、且**不会高过真对齐那一档**就对了 ✓
    check(r2["near"]["score"] <= r["score"] + 0.02,
          "邻域报的分（%.3f）比**真对齐**那档（%.3f）还高 —— 那这个分就没意义了 ✗"
          % (r2["near"]["score"], r["score"]))
    # 关掉时一分钱不多花（老调用方不受影响 ✓）
    check(mm.align_score(panel, t, good).get("near") is None,
          "`near=False`（默认）却把邻域也算了一遍（白花 ✗）")


def t_calib_judge_relative():
    """判据那行**文字**：贴好了写「已在这一档最好」、挪歪了写「附近 (x, y) 更贴」✓
    —— 这正是用户要的那个区分（分低到底是"这张图天生如此"还是"我没对齐"）。
    """
    from PyQt5.QtWidgets import QApplication

    from gui.minimap_calib import MinimapCalibDialog

    mid, canvas = pick_map()
    if mid is None:
        print("      （没有可用的底图，跳过）")
        return
    app = QApplication.instance() or QApplication([])       # noqa: F841
    ch, cw = canvas.shape[:2]
    pw, ph = max(40, min(cw - 8, 120)), max(40, min(ch - 8, 160))
    view = (float(int(cw * 0.2)), float(int(ch * 0.2)))
    panel = _crop_panel(canvas, 1.0, view, pw, ph)
    dlg = MinimapCalibDialog(mid, mode=mm.MODE_CROP, client=StubClient(panel))
    try:
        dlg.ck_live.setChecked(True)
        dlg._on_tick()
        check(dlg._frame is not None, "弹窗没取到帧")
        # 走**弹窗自己那条路**摆好几何（`_locate_now` ✓）—— 手写几何容易跟弹窗的
        # `offset`(inset) 约定错开几个像素，那测的就不是"相对判断"了 ✗
        dlg._locate_now()
        check(dlg.score > 0.5, "弹窗自己都没摆对（分 %.3f）" % dlg.score)
        dlg._refresh_judge()
        check("已在这一档最好" in dlg.lbl_judge.text(),
              "对齐好了的时候该说「已在这一档最好」，实际：%r" % dlg.lbl_judge.text())
        # 相对它刚摆好的位置挪歪 4 个底图像素（比写死 view 稳 ✓）
        dlg.block = [float(dlg.block[0]) + 4.0, float(dlg.block[1])]
        dlg._refresh_judge()
        txt = dlg.lbl_judge.text()
        check("更贴" in txt and "附近" in txt,
              "挪歪了 4 个底图像素，判据行却没指路：%r" % txt)
        check("−" in txt or "-" in txt,
              "指路那句要带符号（往哪个方向）：%r" % txt)
        check("已在这一档最好" not in txt,
              "挪歪了还说「已在这一档最好」✗：%r" % txt)
    finally:
        dlg._shutdown()


TESTS = (
    ("定位细化：非整数倍 scale + 亚像素偏移也量得出",
     t_locate_fit_refined),
    ("标定核对：差多少 + 反向换算（一处实现）",
     t_calib_check_and_reverse),
    ("「实测精度」按钮：一次点击做完核对",
     t_route_panel_mmap_check),
    ("fit 往返：合成整图 → 量回缩放/偏移",
     t_fit_roundtrip),
    ("crop 往返：合成一块 → 量回 view（含 inset）",
     t_crop_roundtrip),
    ("1:1 裁块的 scale 不许被顶替成假值",
     t_crop_scale_not_faked),
    ("crop 放大后取块：要量回 (s, view)",
     t_magnified_crop_roundtrip),
    ("面板是 crop 那块、却按 fit 量 ⇒ 量不出来（不给错值）",
     t_wrong_mode_fails),
    ("局部小地图：运行时跟踪显示区（跟不住不许猜）",
     t_crop_view_track),
    ("局部小地图：定位真的用了跟出来的显示区",
     t_crop_view_locator_wiring),
    ("局部小地图：叠图取的源矩形 = 当前面板显示的那块地图（折算不许动 view）",
     t_crop_overlay_geometry_matches_box),
    ("局部小地图：画面上的叠图也跟着滚（+ 框选区一处口径）",
     t_crop_overlay_follows_view),
    ("可信相机：用黄点的权威世界坐标核对玩家框",
     t_player_tracker_trusted_camera),
    ("⭐ 输入感知的运动预测：按着方向键就照它推演、松手就减速、卡帧要夹住；"
     "**没喂过输入 ⇒ 老口径一字不变**（用户 2026-10-01）",
     t_player_tracker_input_aware),
    ("画面→世界：优先用可信相机（没有就退回老口径）",
     t_screen_to_world_trusted_camera),
    ("可信相机 / 显示区跟踪的接线（源码级）",
     t_trusted_camera_wiring),
    ("地形图里叠实时小地图（跟着显示区滚，自己控制底图）",
     t_live_minimap_on_terrain_map),
    ("地形图里那块实时小地图：自己跟着走 + 透明度参数",
     t_live_map_alpha_and_follows),
    ("「显示类型」下拉 + 像素差分地图（面板减底图那一块，看底图上没有的东西）",
     t_diff_map_view),
    ("⭐ 黄点层优先：**底图相减优先**（用户 2026-10-06「优先使用底图相减找黄点」"
     "⇒ 地图上的假黄色图标不许赢过真点）",
     t_dot_basemap_layer_first),
    ("玩家点定位的第三层：**差分层**（不看颜色 ⇒ 补色族漏检；不同源 / 没亮点时不采信）",
     t_dot_diff_layer),
    ("玩家点「核心色」闸：底图相减层里拒掉洪水色（量测 (108,255,255) vs (116,211,244)）"
     "—— 用户 2026-10-06：位置状态把黄点认错 ⇒ 人在底层最右角落卡死", t_dot_basemap_core_gate),
    ("滚动方向：「双轴/仅X/仅Y」的口径（老标定一字不变）",
     t_scroll_axes_helpers),
    ("滚动方向：单轴只搜那根轴（横带）+ 另一根轴不许跟歪",
     t_crop_view_track_scroll_axis),
    ("滚动方向：「仅 X」的图自动定位只在 X 上动",
     t_locate_crop_scroll_x),
    ("标定弹窗「滚动方向」：读进来 / 选出来 / 存下去",
     t_calib_dialog_scroll_combo),
    ("贴合分：按面板与底图的重叠区算（局部小地图才有意义）",
     t_align_score_local),
    ("贴合分：邻域相对判断（分低是「天生」还是「没对齐」）",
     t_align_score_near),
    ("局部小地图：自动定位要量得出真机那种形状（面板比底图宽 + 非整数倍）",
     t_crop_locate_real_shape),
    ("标定弹窗：判据行说清「已在这一档最好」还是「附近更贴」",
     t_calib_judge_relative),
    ("标定弹窗 v2：偏移条单位 / 「不改缩放」",
     t_calib_dialog_v2_items),
    ("标定弹窗版面：分小节 + x/y 对齐 + 画布占大头 + 没有 markdown 星号",
     t_calib_dialog_layout),
    ("弹窗全链：假帧 → 定位 → 标定字典 → 偏移条/方向键改几何 → 收流",
     t_dialog_with_stub),
    ("持续自动定位：邻近尺度跟踪 + 跟丢重搜",
     t_dialog_auto_fit),
    ("dot_feet 不许取整（读数差 9 的元凶）",
     t_dot_feet_subpixel),
    ("「标定…」窗必须认两轴",
     t_calib_dialog_two_axis),
    ("弹窗「点了有没有反馈」三件事",
     t_dialog_ui_feedback),
    ("一打开看着空白/不对：必须说清为什么",
     t_calib_open_says_why_blank),
    ("「标定…」与「双点标定」共用一份数据",
     t_calib_windows_share_store),
    ("保存 → 重开：手工对齐的几何原样回来",
     t_save_then_reopen),
    ("「标定」把几何改成等比后，旧的双点标定 scale_y 不许残留（用户 2026-09-28 报）",
     t_calib_save_drops_stale_scale_y),
    ("来源「从实时画面框选」：裁块/报错/弹窗能用",
     t_live_source_region),
    ("来源这条路的接线（源码约定）",
     t_live_source_wiring),
    ("框选区域按项目取：本项目优先、回退老那份",
     t_mmap_crop_is_per_project),
    ("「小地图定位」分两行（来源另起一行）",
     t_mmap_rows_split),
    ("⭐ 寻路配置**最顶部**「当前地图」+「手动更换」（只读 / 一处分口径 / 同一个写口）",
     t_route_panel_current_map_row),
    ("双点标定：两对点 → 两轴缩放 + 偏移 + 自查",
     t_two_point_solve),
    ("双点标定弹窗：两次采样 → 算 → 存",
     t_two_point_dialog),
    ("双点采样 → 读数闭环（回到那点就是那个坐标）",
     t_two_point_readout_roundtrip),
    ("换标定后世界读数立刻跟着变（不卡缓存）",
     t_world_reads_new_calib),
    ("双点标定窗关掉再开：填过的数还在",
     t_two_point_keeps_state),
    ("双点标定窗必须非模态",
     t_two_point_dialog_nonmodal),
    ("叠图画到实时画面：几何/接线/不碰帧",
     t_overlay_on_live),
    ("地形图 = 地形编辑器的结果（集合色、面板优先）",
     t_terrain_image_shows_zones),
    ("命令前往交整条路径（含跳 ⇒ 整条拒发并说清）",
     t_goto_hands_over_whole_path),
    ("命令前往真下发（爬 = 造任务挂给当前 agent）",
     t_goto_commands_climb),
    ("画面那几行：当前任务 + 计时任务逐行",
     t_osd_task_and_timers),
    ("预览平台下拉 + 命令前往：填集合/框在图上/算路",
     t_goto_picker_and_preview),
    ("「设置 → 界面」那两项真的生效",
     t_settings_overlay_takes_effect),
    ("「框选小地图」在路线识别页、按项目存",
     t_route_panel_mmap_crop),
    ("叠图不许在折算比例未知时先画一版错的",
     t_overlay_waits_first_frame),
    ("「地形图」那行报图片实际尺寸",
     t_map_image_size_shown),
    ("亮黄点：找得出、位置准",
     t_dot_yellow_found),
    ("无素材时的 5×5 青色方块也算玩家",
     t_dot_cyan_fallback),
    ("NPC/portal 的蓝点不是玩家",
     t_dot_reject_npc),
    ("亮黄块一大堆：颜色层当场说不可用",
     t_dot_flood_says_unusable),
    ("底图相减层：只有多出来的那块才是玩家",
     t_dot_basemap_layer),
    ("漏检沿用上一帧位置（超窗口才认不出）",
     t_dot_hold_last_on_miss),
    ("有上一帧位置时就近搜",
     t_dot_roi_search_local),
    ("「被淹」判据跟着面板大小走",
     t_dot_flood_scale_invariant),
    ("跨帧：连续两拍才算确认；跳变当噪声",
     t_dot_tracker_confirm_and_jump),
    ("segment_of / find_below：真实地形上验一遍",
     t_segment_of_basics),
    ("面板黄点 → 世界坐标 → 哪条段（整链）",
     t_world_pos_chain),
    ("小地图收流超时要分清卡在哪一步",
     t_minimap_client_timeout_says_where),
    ("定位结论写进 Player：算不出写 None（不是 0）",
     t_apply_to_player),
    ("「世界坐标」那行：勾上叠图才显示",
     t_world_label_text),
    ("实时回路裁面板要复制（别串进框线）",
     t_live_thread_mmap_panel_copy),
    ("长面板必须能滚",
     t_route_panel_scrolls),
    ("screen_to_world：画面坐标 → 世界坐标",
     t_screen_to_world),
    ("用世界坐标辅助玩家框防抖",
     t_player_tracker_world_anchor),
    ("怪所在层的**方向自洽校验**：画面上更高 ⇒ 层必须在玩家上方，反了拒不判"
     "（用户 2026-09-28：「顶层判定成底层这种向量都反了的」）",
     t_mob_layer_direction_check),
    ("「标记跟踪」四个容差排两行",
     t_track_rows_layout),
    ("黄点容差是界面参数：回填/生效/坏值退回",
     t_dot_track_params_are_config),
    ("黄点容差换图不丢",
     t_dot_track_params_survive_map_change),
    ("safe_slot 按槽签名裁掉多余参数",
     t_safe_slot_drops_extra_signal_args),
    ("标定弹窗的按钮点得动（走真实信号）",
     t_calib_buttons_are_alive),
    ("标定按来源分开存",
     t_calib_per_source),
    ("标定老格式仍读得出（升级不作废）",
     t_calib_legacy_readable),
    ("老格式必须在界面上说出来",
     t_calib_legacy_is_visible),
    ("标定弹窗存到自己那条来源下",
     t_calib_dialog_saves_its_source),
    ("路线识别页上要有「标定…」按钮（源码级）",
     t_route_panel_button),
    ("路径解析器的起点取「我现在站哪个集合」",
     t_route_resolver_start_from_player_sets),
    ("`ladder_id` 的许可（hold_vert）真的接到了执行器（用户 2026-09-28 报「不是 LADDER_PAD」）",
     t_fill_route_ctx_hold_vert_wired),
    ("「攀爬失败」还要「僵着不动」才算 —— 不然斜跳必被杀（用户 2026-09-28 报「斜向跳全失败」）",
     t_climb_failed_needs_stuck),
    ("脚下 foothold 的 `xtol`：两遍都比、更近才改判（用户 2026-09-28 报 (594,162) 被解析成"
     "「底层」，该是「小平台」）",
     t_foothold_below_prefers_nearer_within_xtol),
    ("查过的怪框标出来、缓存失效再移除",
     t_queried_mob_boxes_marked),
    ("⭐ 「查过的怪框」跟着**这一帧的检出框**走（用户 2026-10-02：红框原地残留 ⇒ 现在按怪号"
     "现查、查不到就不画）",
     t_queried_mob_boxes_follow_detection),
    ("「怪判不出集合」时的可观测性：相机不许再平滑、why 分档说清卡在哪级、"
     "mob_fh 打点带上图名与玩家 x/框底（用户 2026-09-28：又开始全程找不到，排查全靠猜）",
     t_mob_fh_observability),
    ("位置状态机自治：七个广播字段一处算",
     t_pos_state_machine),
    ("「位置状态」那行挂上广播的 at_ladder_top",
     t_osd_shows_ladder_top),
    ("标定窗方向键只挪底图（滑条/数字框不许抢）",
     t_calib_arrows_move_overlay),
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
