"""路线识别面板：平台识别 / 跳跃落点预测 的总开关 + 小地图定位 + 地形图。

「总开关」这几项每帧做图像处理（平台顶边检测、玩家运动追踪），比较吃 CPU，
默认关闭。**关掉 = 这块一点活都不干**（包括地形关系识别，见 live_thread），
所以它是一条真的性能开关，不是只影响画面。

**小地图定位**：寻路要用小地图上的黄点算出世界坐标，而「小地图面板怎么显示
底图」有两种方式，**不同地图可能不一样**（所以界面上的名字按"看见的是什么"来）：

    全局小地图（fit）   整张底图缩放到面板里     → 地形图不随人移动，只有黄点在动
    局部小地图（crop）  面板 1:1 显示底图的一块  → 地形图跟着人滑动（黄点基本在中间）

这里**让你选**（每张图存一份，见 datasets/map/<id>.mapcalib.json），
程序不替你猜。判断方法就写在面板上：进游戏左右走两步看一眼即可。

**地形图**：把 `datasets/map/<id>_overlay.png`（tools/map_terrain_view.py 生成：
每条平台一色 + 传送点/绳梯/刷怪点）显示出来。小地图定位对不对、寻路要往哪走，
看着这张图才有概念 —— 只有一串数字的话没法判断。
"""

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QApplication, QCheckBox, QHBoxLayout, QLabel,
                             QMessageBox, QPushButton, QVBoxLayout, QWidget)

from core import mapdata
from decision.agent import settings
from gui.canvas import ImageCanvas
from gui.minimap_calib import MinimapCalibDialog   # 量「面板 ↔ 底图」的弹窗
from gui.widgets import NoWheelComboBox      # 滚轮不许改参数（docs/UI规范.md）
from perception import minimap as mm
from tools.config import load_live, update_live    # 来源/框选区域存 config/live.yaml


def _generate_task(params, ctx):
    """后台：缺地形数据就先从 WZ 导，再画叠加图。**不碰任何 Qt 控件。**

    （跑在 gui.worker.TaskThread 里，见那里的铁律：工作线程只 emit 信号。）
    """
    mid = str(params["map_id"])
    out = mapdata.map_dir()
    json_path = out / ("%s.json" % mid)

    if json_path.exists() and not params.get("force_export"):
        # 地形数据已经有了：只重画叠加图（毫秒级）—— 大部分时候用户只是想看
        # 最新的那张，没必要每次去解析一遍 Map.wz（那是几十秒）。
        ctx.log("已有地形数据 %s，只重画叠加图" % json_path.name)
    else:
        from core import wzexport
        wzexport.run_export_terrain({"map_id": mid, "out": str(out)}, ctx)

    if ctx.canceled():
        return {"summary": "已取消"}

    from tools.map_terrain_view import render
    t = mapdata.load(mid, with_canvas=True)
    if t is None:
        raise RuntimeError("读不到地形 JSON：%s" % json_path)

    target = out / ("%s_overlay.png" % mid)
    # render 返回的是 foothold 条数（其中墙多少），不是"串成的段数" —— 别写错标签
    w, h, n_fh, n_wall = render(t, target)
    ctx.log("叠加图 %s  %d×%d（foothold %d / 其中墙 %d，串成 %d 段）"
            % (target.name, w, h, n_fh, n_wall, len(t.segments)), "ok")
    return {"summary": "已生成 %s" % target.name, "path": str(target)}


class RoutePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        title = QLabel("路线识别")
        title.setStyleSheet("font-weight: 600;")
        root.addWidget(title)

        note = QLabel("平台识别、跳跃落点预测、扫平台等寻路功能的总开关。\n"
                      "这些功能每帧做图像处理，比较吃 CPU，默认关闭。")
        note.setWordWrap(True)
        note.setStyleSheet("color: #5f6368;")
        root.addWidget(note)

        self.ck_enabled = QCheckBox("启用路线识别")
        self.ck_enabled.setChecked(bool(settings.route_enabled))
        self.ck_enabled.toggled.connect(self._on_toggle)
        root.addWidget(self.ck_enabled)

        self.lbl_hint = QLabel()
        self.lbl_hint.setStyleSheet("color: #80868b;")
        self.lbl_hint.setWordWrap(True)
        root.addWidget(self.lbl_hint)
        self._refresh_hint()

        # ---- 小地图定位（寻路用；方式由你选，程序不猜）----
        root.addSpacing(12)
        lbl2 = QLabel("小地图定位")
        lbl2.setStyleSheet("font-weight: 600;")
        root.addWidget(lbl2)

        # 面板上只留「参数 + 状态」；怎么判断选哪种、标定怎么跑，都放 tooltip
        # （docs/UI规范.md：正文别塞说明，细节交给 tooltip）
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(QLabel("当前地图"))
        self.lbl_mmap = QLabel("—")
        row.addWidget(self.lbl_mmap)
        row.addSpacing(12)
        row.addWidget(QLabel("显示方式"))
        self.cmb_mmap_mode = NoWheelComboBox()
        # 名字按「你在小地图上看到的是什么」取：整张都在 = 全局；
        # 只看到一块、跟着人滑动 = 局部。（fit/crop 是实现里的叫法，别摆到界面上）
        self.cmb_mmap_mode.addItem("全局小地图", mm.MODE_FIT)
        self.cmb_mmap_mode.addItem("局部小地图", mm.MODE_CROP)
        self.cmb_mmap_mode.setToolTip(
            "小地图面板怎么显示底图，两种方式每张图可能不一样，所以由你选"
            "（每张图存一份）。\n\n"
            "怎么判断：进游戏左右走两步，盯着小地图看 ——\n"
            "  · 地形图**不动**，只有黄点在移动 → 全局小地图\n"
            "  · 地形图**跟着你滑动**（黄点差不多在中间） → 局部小地图")
        self.cmb_mmap_mode.currentIndexChanged.connect(self._on_mmap_mode)
        row.addWidget(self.cmb_mmap_mode)

        # ---- 面板画面从哪来（**实验**开关；存 config/live.yaml，和地图无关）----
        #   收流        —— A 机那条独立推流：原始像素，黄点最清楚（默认）
        #   从实时画面  —— 不另推一路，直接在实时预览那一帧上裁一块
        #                  （画面是 H.264 压过的；面板↔底图匹配没问题，黄点待实测）
        row.addSpacing(12)
        row.addWidget(QLabel("小地图来源"))
        self.cmb_mmap_src = NoWheelComboBox()
        self.cmb_mmap_src.addItem("收流（A 机小地图推流）", mm.SRC_STREAM)
        self.cmb_mmap_src.addItem("从实时画面框选（实验）", mm.SRC_LIVE)
        self.cmb_mmap_src.setToolTip(
            "小地图面板的画面从哪来。\n\n"
            "收流（默认）：A 机「被控机部署台 → 小地图推流」单独推一路原始像素 ——\n"
            "  黄点（玩家点）只有几个像素，这一路是唯一能保证它不糊的画质。\n\n"
            "从实时画面框选（实验）：不另推一路，直接在「实时」页那一帧上裁一块。\n"
            "  省掉 A 机一次截屏 + 一路 TCP，但画面是压过的 —— 面板和底图还能对上，\n"
            "  黄点识别能不能稳还没实测。选了它右边会出现「框选…」。\n\n"
            "两者只在「画面从哪来」这一步不同：标定、换算、世界坐标都一样。")
        self.cmb_mmap_src.currentIndexChanged.connect(self._on_mmap_src)
        row.addWidget(self.cmb_mmap_src)

        # 「框选…」只在「从实时画面框选」时出现（别的来源没有要框的东西）。
        # 走 gui/region_selector（放大镜 + 像素网格 + Esc + <4px 当误点，
        # 见 docs/UI规范.md §8：框选一律走那一份，不许各写一份）。
        self.btn_mmap_region = QPushButton("框选…")
        self.btn_mmap_region.setToolTip(
            "在**实时画面**上把小地图面板框出来（框完立刻和底图核对一次，\n"
            "把匹配分显示在下面）。\n\n"
            "先到「实时」页点开始预览，看到画面里的游戏小地图再回来框。\n"
            "只框**面板本身**：多框进来的血条/聊天会一起算进去，匹配分会掉下来。\n\n"
            "画面尺寸变了（换分辨率 / 改推流参数）要重框一次。")
        self.btn_mmap_region.clicked.connect(self._pick_mmap_region)
        self.btn_mmap_region.setVisible(False)
        row.addWidget(self.btn_mmap_region)

        # 标定做成弹窗（照「手动目测标定尺度」那套交互）：量出来的是
        # 「面板 ↔ 底图」的缩放/偏移，而判据是**看得见**的重合程度 ——
        # 摆在这块面板里放不下，也不该让人去命令行跑一遍。
        self.btn_mmap_calib = QPushButton("标定…")
        self.btn_mmap_calib.setToolTip(
            "量这张图的「面板 ↔ 底图」换算（存 datasets/map/<id>.mapcalib.json）。\n\n"
            "弹窗里：A 机推来的实时面板 + 半透明底图叠在一起 ——\n"
            "「自动定位」用模板匹配量（准）；匹配不上时可以手动拖动对齐，\n"
            "目测重合后「保存标定」。\n\n"
            "前置：A 机的「被控机部署台 → 小地图推流」已经在跑。")
        self.btn_mmap_calib.clicked.connect(self._open_mmap_calib)
        row.addWidget(self.btn_mmap_calib)
        row.addStretch(1)
        root.addLayout(row)

        self.lbl_mmap_hint = QLabel()
        self.lbl_mmap_hint.setStyleSheet("color: #80868b;")
        self.lbl_mmap_hint.setWordWrap(True)
        root.addWidget(self.lbl_mmap_hint)

        # ---- 地形图（看得见才好判断小地图定位对不对）----
        root.addSpacing(12)
        lbl3 = QLabel("地形图")
        lbl3.setStyleSheet("font-weight: 600;")
        root.addWidget(lbl3)

        self.lbl_map_img = QLabel()
        self.lbl_map_img.setStyleSheet("color: #80868b;")
        self.lbl_map_img.setWordWrap(True)
        root.addWidget(self.lbl_map_img)

        # 只读看图画布：和质检台同一套交互（滚轮缩放 / 中键平移 / 双击适应）
        self.canvas = ImageCanvas()
        self.canvas.setMinimumHeight(320)
        root.addWidget(self.canvas, 1)

        # ---- 生成：一条命令干完「导出 WZ 地形 → 画叠加图」----
        # 数据缺就导（几十秒，后台跑），已有就只重画（毫秒级）。
        self.task = None
        gen = QHBoxLayout()
        gen.setSpacing(6)
        self.btn_gen = QPushButton("生成地形图")
        self.btn_gen.setToolTip(
            "自动跑完导出地形这件事：\n"
            "  1. datasets/map/<地图id>.json 不存在 → 起 WzProbe.exe dump-terrain\n"
            "     只导这一张（解析 Map.wz 要几十秒，期间界面可以继续用）；\n"
            "  2. 已经有地形数据 → 跳过第 1 步，只重画叠加图（毫秒级）；\n"
            "  3. 画完自动刷新上面的图，不用重启工作台。\n\n"
            "资源/客户端换过之后要**重新导出**地形数据的话，先删掉\n"
            "datasets/map/<地图id>.json 再点这个按钮。")
        self.btn_gen.clicked.connect(self._generate)
        gen.addWidget(self.btn_gen)

        self.lbl_gen = QLabel("")
        self.lbl_gen.setStyleSheet("color: #80868b;")
        self.lbl_gen.setWordWrap(True)
        gen.addWidget(self.lbl_gen, 1)
        root.addLayout(gen)

        self._refresh_mmap()
        self._refresh_map_image()

    # ---------------- 小地图定位 ----------------

    def _map_id(self):
        p = getattr(self, "project", None)
        return (p.get("map_id") or "").strip() if p is not None else ""

    @staticmethod
    def _mode_label(mode):
        """内部值 → 界面上的名字（fit = 全局 / crop = 局部）。"""
        if mode == mm.MODE_FIT:
            return "全局小地图"
        if mode == mm.MODE_CROP:
            return "局部小地图"
        return str(mode)

    # ---------------- 小地图来源（面板画面从哪来）----------------

    def _mmap_src(self):
        """当前来源（config/live.yaml 的 `mmap_src`，默认收流）。

        存 B 机本地配置、而不是每张图的标定里：它取决于本机的推流/画面几何，
        和"这是哪张地图"无关（切项目不该跟着变）。
        """
        v = load_live().get("mmap_src")
        return v if v in (mm.SRC_STREAM, mm.SRC_LIVE) else mm.SRC_STREAM

    def _on_mmap_src(self, _i):
        """换来源：立刻存下来 + 让「框选…」按它显隐。"""
        src = self.cmb_mmap_src.currentData()
        update_live(mmap_src=src)
        self.btn_mmap_region.setVisible(src == mm.SRC_LIVE)
        self._refresh_mmap()

    def _pick_mmap_region(self):
        """在**实时画面**上框出小地图面板（实验来源用）。

        规范（docs/UI规范.md §8）：框选一律走 `gui/region_picker`（放大镜、
        `+/-` 调倍率、Esc 取消、**<4px 当误点**）；**框完当场验证** —— 小地图
        这条的验证就是"这块画面和底图能对上多少分"，所以框完立刻算一次分。
        """
        lp = getattr(self, "live_panel", None)
        frame = None if lp is None else lp.current_frame()
        if frame is None:
            QMessageBox.information(
                self, "先开始预览",
                "「从实时画面框选」要在实时画面上框：\n"
                "先到「实时」页点「开始」，看到画面（含游戏的小地图面板）再回来框。")
            return
        from gui.region_selector import select_region_on_image

        rect = select_region_on_image(frame, self)
        if rect is None:
            return                  # 取消（<4px 的框在框选里就按误点丢掉了）
        update_live(mmap_src=mm.SRC_LIVE, mmap_crop=[int(v) for v in rect])
        self._refresh_mmap()        # 先把来源/区域刷进状态行
        self._verify_mmap_region(frame, rect)

    def _verify_mmap_region(self, frame, rect):
        """框完当场验证：裁出来那块 ↔ 底图 对一次分，写在状态行上。

        **为什么必须当场验**：框大了（把血条/聊天栏框进去）、面板被游戏 UI
        挡住、「显示方式」选错 —— 这三种的现象都是"寻路看起来坏了"，
        而在这个入口上（匹配分）一眼就能看出来。
        """
        x, y, w, h = (int(v) for v in rect)
        head = "已框选 (x=%d, y=%d, %d×%d)" % (x, y, w, h)
        mid = self._map_id()
        t = mapdata.load(mid, with_canvas=True) if mid else None
        if t is None or t.canvas is None:
            self._set_hint(head + "　（这张图还没有底图 —— 先在下面点「生成地形图」）",
                           "#b06000")
            return
        self._set_hint(head + "　正在和底图核对…")
        QApplication.processEvents()    # 先把上面那行画出来（核对要几十毫秒）
        mode = self.cmb_mmap_mode.currentData() or mm.MODE_FIT
        loc = mm.locate(frame[y:y + h, x:x + w], t.canvas, mode)
        if loc is None:
            self._set_hint(head + "　匹配不上 —— 可能框大了（含血条/聊天栏）、"
                                  "面板被游戏 UI 挡住，或「显示方式」选错了",
                           "#c5221f")
            return
        ok = loc["score"] >= 0.8
        self._set_hint(head + "　匹配分 %.2f%s"
                       % (loc["score"], "" if ok else "（偏低：0.8 以上才算对上）"),
                       "#188038" if ok else "#b06000")

    def _set_hint(self, text, color="#80868b"):
        """状态行即时反馈（带颜色：中性灰 / 提示橙 / 对不上红 / 对上了绿）。"""
        self.lbl_mmap_hint.setText(text)
        self.lbl_mmap_hint.setStyleSheet("color: %s;" % color)

    def _open_mmap_calib(self):
        """打开标定弹窗（量「面板 ↔ 底图」的换算）。

        显示方式**按上面选的那个**带进去：方式要进游戏走两步才知道该用哪种，
        弹窗里不替你改（和命令行工具的态度一致，见 perception/minimap.py）。
        """
        mid = self._map_id()
        if not mid:
            QMessageBox.information(
                self, "先选地图",
                "先在①「识别目标选项」里选地图 —— 标定是按每张图各存一份的。")
            return
        t = mapdata.load(mid, with_canvas=True)
        if t is None or t.canvas is None:
            QMessageBox.information(
                self, "这张图没有小地图底图",
                "标定要有底图（datasets/map/%s.png）才能对齐。\n\n"
                "先在下面点「生成地形图」把地形和底图导出来。" % mid)
            return
        # 画面来源按上面选的那个：
        #   收流       → client=None，弹窗自己连 A 机那一口（原路不变）
        #   从实时画面 → 塞一个"裁实时帧"的适配器进去（弹窗那边一行都不用改）
        client = None
        if self._mmap_src() == mm.SRC_LIVE:
            region = load_live().get("mmap_crop")
            lp = getattr(self, "live_panel", None)
            if not region or len(region) != 4:
                QMessageBox.information(
                    self, "先框选区域",
                    "来源是「从实时画面框选」—— 先在实时画面上把小地图面板框出来：\n"
                    "点上面的「框选…」。")
                return
            if lp is None or lp.current_frame() is None:
                QMessageBox.information(
                    self, "先开始预览",
                    "标定要拿实时画面里那一块 —— 先到「实时」页点「开始」，\n"
                    "看到画面再回来。")
                return
            from gui.live_panel import LiveFrameRegionClient
            client = LiveFrameRegionClient(lp, region)
        dlg = MinimapCalibDialog(mid, mode=self.cmb_mmap_mode.currentData(),
                                 client=client, parent=self)
        dlg.exec_()
        if dlg.saved:
            self._refresh_mmap()      # 状态行立刻变成「已标定 … 匹配分 …」

    def _refresh_mmap(self):
        mid = self._map_id()
        p = getattr(self, "project", None)
        cal = mapdata.load_calib(mid) if mid else None

        # 状态行颜色归位：框选验证时会染成绿/橙/红，任何一次刷新都该回到中性灰
        self.lbl_mmap_hint.setStyleSheet("color: #80868b;")
        # 来源跟着配置走（blockSignals：不然 setCurrentIndex 会反过来写回去）
        src_kind = self._mmap_src()
        self.cmb_mmap_src.blockSignals(True)
        j = self.cmb_mmap_src.findData(src_kind)
        self.cmb_mmap_src.setCurrentIndex(j if j >= 0 else 0)
        self.cmb_mmap_src.blockSignals(False)
        self.btn_mmap_region.setVisible(src_kind == mm.SRC_LIVE)

        # 「没打开项目」和「项目里还没选地图」是两回事 ——
        # 以前一律写「（没打开项目）」，项目明明开着却看到这句，像项目丢了。
        # （地图是在①里选的；选完这里靠 showEvent 重读，见下面。）
        if p is None:
            self.lbl_mmap.setText("（没打开项目）")
        elif not mid:
            self.lbl_mmap.setText("（这个项目还没选地图）")
        else:
            self.lbl_mmap.setText(mid)

        self.cmb_mmap_mode.setEnabled(bool(mid))
        self.btn_mmap_calib.setEnabled(bool(mid))
        self.cmb_mmap_mode.blockSignals(True)      # 不挡信号会把刚读的值又写回去
        i = self.cmb_mmap_mode.findData((cal or {}).get("mode"))
        self.cmb_mmap_mode.setCurrentIndex(i if i >= 0 else 0)
        self.cmb_mmap_mode.blockSignals(False)

        if not mid:
            self.lbl_mmap_hint.setText("先在①选地图" if p is not None
                                       else "打开项目后再设")
            self.lbl_mmap_hint.setToolTip("")
        elif not mm.has_geometry(cal):
            # **判的是「有没有几何」，不是「score 有没有」**：手工对齐没有
            # 匹配分（score=0），但它照样是一份能用的标定 —— 拿 score 判会让人
            # 看到「未标定」，以为自己白标了（见 perception/minimap.has_geometry）。
            self.lbl_mmap_hint.setText("未标定")
            if src_kind == mm.SRC_LIVE:
                # 来源是"从实时画面框选"时，前置条件完全不同 —— 别让人去 A 机
                # 找那口推流（他可能压根没在用）。
                tip = ("这张图还没量过「面板像素 → 底图像素」的换算。\n"
                       "来源是「从实时画面框选」：先在实时画面上把小地图面板框出来\n"
                       "（右边的「框选…」），再点「标定…」量一次（弹窗里看得见重合）。")
            else:
                tip = ("这张图还没量过「面板像素 → 底图像素」的换算。\n"
                       "在 A 机「被控机部署台 → 小地图推流」启动之后，\n"
                       "点上面的「标定…」量一次（弹窗里看得见重合不重合）。\n\n"
                       "等价的命令行做法（排查用）：\n"
                       "    python -m perception.minimap --map %s" % mid)
            self.lbl_mmap_hint.setToolTip(tip)
        else:
            geo = ("缩放 %.2f　偏移 %s" % (cal["scale"], cal.get("offset"))
                   if cal.get("mode") == mm.MODE_FIT
                   else "显示区起点 %s" % (cal.get("view"),))
            # 手工对齐的没有匹配分：写「手工对齐」，别摆一个 0.00 让人以为坏了
            how = ("手工对齐" if (cal.get("src") == "manual"
                                  or not cal.get("score"))
                   else "匹配分 %.2f" % cal["score"])
            self.lbl_mmap_hint.setText(
                "已标定　%s　%s　%s" % (self._mode_label(cal["mode"]), geo, how))
            crop = load_live().get("mmap_crop")
            self.lbl_mmap_hint.setToolTip(
                "量出来的几何参数：缩放 / 偏移 / 显示区起点。\n"
                "改「显示方式」只换方式，这些数不会丢。\n\n"
                "画面来源：%s%s"
                % ("从实时画面框选" if src_kind == mm.SRC_LIVE
                   else "A 机小地图推流",
                   ("，区域 %s（在「实时」页的画面上框的）" % (crop,)
                    if src_kind == mm.SRC_LIVE and crop else "")))

    # ---------------- 地形图 ----------------

    def _map_image_path(self, mid):
        """这张图该显示哪张图 → (路径或 None, 说明文字)。

        优先 `<id>_overlay.png`：那是 tools/map_terrain_view.py 画出来的"看得懂的
        地形图"（每条平台一色、黄=传送点、青=绳梯、品红=刷怪点）。没有就退到
        小地图底图 `<id>.png`；都没有说明地形还没导出，把该跑的命令写出来。
        """
        d = mapdata.map_dir()
        over = d / ("%s_overlay.png" % mid)
        if over.exists():
            return over, ("地形叠加图 %s　（每条平台一色 · 黄=传送点 · "
                          "青=绳梯 · 品红=刷怪点）" % over.name)
        base = d / ("%s.png" % mid)
        if base.exists():
            return base, ("小地图底图 %s　（还没有叠加图 —— 点下面的「生成地形图」"
                          "画一张看得更清楚的；也可以手跑：\n"
                          "    python -m tools.map_terrain_view %s）"
                          % (base.name, mid))
        return None, (
            "这张图还没有地形数据（datasets/map/%s.json 不存在）。\n"
            "点下面的「生成地形图」会自动导出并画出来（也可以手跑：\n"
            "    WzProbe.exe dump-terrain <WZ目录> datasets/map --only %s）"
            % (mid, mid))

    def _refresh_map_image(self):
        mid = self._map_id()
        if not mid:
            self.lbl_map_img.setText(
                "先在①选地图。" if getattr(self, "project", None) is not None
                else "打开一个项目后再看。")
            self.canvas.load(QPixmap(), [], editable=False, fit=True)
        else:
            path, note = self._map_image_path(mid)
            self.lbl_map_img.setText(note)
            pm = QPixmap(str(path)) if path is not None else QPixmap()
            self.canvas.load(pm, [], editable=False, fit=True)

        # 没地图就没得生成；正在跑的时候也不让重复点
        self.btn_gen.setEnabled(bool(mid) and self.task is None)

    # ---------------- 生成地形图 ----------------

    def _generate(self):
        """点「生成地形图」：缺 WZ 数据就先导出，然后画叠加图。

        整件事丢到后台线程（见 gui/worker.py 的铁律：工作线程只 emit 信号，
        绝不碰控件）—— 导出要解析 Map.wz，几十秒，卡住界面就没法用了。
        """
        mid = self._map_id()
        if not mid:
            QMessageBox.information(
                self, "先选地图",
                "先在①「识别目标选项」里选地图 —— 才知道要导哪一张。")
            return
        if self.task is not None:
            return

        self.btn_gen.setEnabled(False)
        self.lbl_gen.setStyleSheet("color: #5f6368;")
        self.lbl_gen.setText("开始…")

        from gui.worker import TaskThread, safe_slot
        self.task = TaskThread(_generate_task, {"map_id": mid}, self)
        # 槽函数一律过 safe_slot：槽里抛异常会让 PyQt5 直接 abort() 整个程序
        self.task.sig_log.connect(safe_slot(self._on_gen_log))
        self.task.sig_done.connect(safe_slot(self._on_gen_done))
        self.task.finished.connect(safe_slot(self._on_gen_finished))
        self.task.start()

    def _on_gen_log(self, msg, _level="info"):
        """把子进程最后一行输出显示在按钮旁边（长路径截断）。"""
        line = " ".join(str(msg).split())
        self.lbl_gen.setText(line[:120] + ("…" if len(line) > 120 else ""))

    def _on_gen_done(self, ok, summary, _result=None):
        self._refresh_map_image()          # 画好了立刻换图
        self.lbl_gen.setStyleSheet("color: %s;" % ("#137333" if ok else "#c5221f"))
        self.lbl_gen.setText(("✓ " if ok else "✗ ") + (summary or ""))

    def _on_gen_finished(self):
        # 线程真正结束才释放引用（和 export_dialog / main_window 同一套写法）
        t = self.task
        self.task = None
        if t is not None:
            t.deleteLater()
        self.btn_gen.setEnabled(bool(self._map_id()))

    def _on_mmap_mode(self, _i):
        mid = self._map_id()
        if not mid:
            return
        cal = mapdata.load_calib(mid) or {}
        # **只改方式**，量出来的几何参数（scale/offset/view）保留
        cal["mode"] = self.cmb_mmap_mode.currentData()
        cal["picked_by"] = "gui"
        mapdata.save_calib(mid, cal)
        self._refresh_mmap()

    def bind(self, project):
        """切项目：把开关刷成当前项目的值。

        `route_enabled` 也是一个决策参数，**按项目存**（见
        MainWindow._bind_decision_params），所以切项目必须重新读一遍，
        否则显示的是上一个项目的值。

        这里必须 blockSignals：不挡的话 setChecked 会触发 _on_toggle，
        把刚读出来的值又 save 回去 —— 而且 save 的是新项目，
        等于拿旧项目的开关覆盖新项目。
        """
        self.ck_enabled.blockSignals(True)
        self.ck_enabled.setChecked(bool(settings.route_enabled))
        self.ck_enabled.blockSignals(False)
        self._refresh_hint()
        self.project = project
        self._refresh_mmap()
        self._refresh_map_image()

    def showEvent(self, e):
        """切到「路线识别」页签时重读一次。

        为什么需要：地图是在①「识别目标选项」里选的，而这个页签**不会**跟着刷新 ——
        选完地图切过来还是旧值（甚至停在「（这个项目还没选地图）」），
        看着像①的选择没生效。按页签刷新的时机来重读，代价只是一次文件读取。
        """
        super().showEvent(e)
        if getattr(self, "project", None) is not None:
            self._refresh_mmap()
            self._refresh_map_image()

    def _refresh_hint(self):
        self.lbl_hint.setText(
            "已启用：实时预览会每帧检测平台顶边、预测跳跃落点（更耗 CPU）。"
            if self.ck_enabled.isChecked() else
            "已关闭：跳过平台识别和跳跃预测，实时预览更流畅。")

    def _on_toggle(self, on):
        settings.route_enabled = bool(on)
        settings.save()
        self._refresh_hint()
