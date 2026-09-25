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

from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QApplication, QCheckBox, QHBoxLayout, QLabel,
                             QMessageBox, QPushButton, QVBoxLayout, QWidget)

from core import mapdata
from decision.agent import settings
from gui.canvas import ImageCanvas
from gui.minimap_calib import MinimapCalibDialog   # 量「面板 ↔ 底图」的弹窗
from gui.widgets import NoWheelComboBox, NoWheelSpinBox   # 滚轮不许改参数（UI规范 §5）
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
        #
        # **分两行**（docs/UI规范.md §4：一行只放一个参数组，放不下就另起一行，
        # 不许在行尾继续 addWidget 堆）：
        #   第一行「这是哪张图 · 怎么显示」
        #   第二行「画面从哪来 · 框选/标定动作」
        # 要加新参数时先想它属于哪一组，再决定放哪一行 —— 一路往右加的话，
        # 窗口一窄就是所有标签一起被压没，而且再也说不清哪几个是一组。
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
        # 第一行到这儿结束。**别在这行后面继续加** —— 加上去的东西会一路往右堆，
        # 窗口一窄先被压没的就是标签（§4 就是这么被违反的）。要加先归组。
        row.addStretch(1)
        root.addLayout(row)

        # ---- 第二行：画面从哪来 + 框选/标定动作 ----
        row2 = QHBoxLayout()
        row2.setSpacing(6)

        # ---- 面板画面从哪来（**实验**开关；存 config/live.yaml，和地图无关）----
        #   收流        —— A 机那条独立推流：原始像素，黄点最清楚（默认）
        #   从实时画面  —— 不另推一路，直接在实时预览那一帧上裁一块
        #                  （画面是 H.264 压过的；面板↔底图匹配没问题，黄点待实测）
        row2.addWidget(QLabel("小地图来源"))
        self.cmb_mmap_src = NoWheelComboBox()
        self.cmb_mmap_src.addItem("收流（A 机小地图推流）", mm.SRC_STREAM)
        self.cmb_mmap_src.addItem("从实时画面框选（实验）", mm.SRC_LIVE)
        self.cmb_mmap_src.setToolTip(
            "小地图面板的画面从哪来。\n\n"
            "收流（默认）：A 机「被控机部署台 → 小地图推流」单独推一路原始像素 ——\n"
            "  黄点（玩家点）只有几个像素，这一路是唯一能保证它不糊的画质。\n\n"
            "从实时画面框选（实验）：不另推一路，直接在「实时」页那一帧上裁一块。\n"
            "  省掉 A 机一次截屏 + 一路 TCP，但画面是压过的 —— 面板和底图还能对上，\n"
            "  黄点识别能不能稳还没实测。\n\n"
            "（「框选小地图」和来源**无关**，是必做的一步：两种来源都要知道\n"
            "  小地图面板在实时画面的哪儿 —— 收流时主画面里也有小地图，只是压过。）\n\n"
            "两者只在「画面从哪来」这一步不同：标定、换算、世界坐标都一样。")
        self.cmb_mmap_src.currentIndexChanged.connect(self._on_mmap_src)
        row2.addWidget(self.cmb_mmap_src)

        # 「框选小地图」是**必做的一步**，不再跟着来源显隐：它记的是
        # 「小地图面板在实时画面里的哪个位置」（config/live.yaml 的 mmap_crop），
        # 两种来源都要用它 ——
        #   收流      → 主画面里**也有**小地图（只是被 H.264 压过）：叠图往哪画靠它；
        #   从实时画面 → 额外还要靠它把面板裁出来喂标定弹窗（LiveFrameRegionClient）。
        # 走 gui/region_selector（放大镜 + 像素网格 + Esc + <4px 当误点，
        # 见 docs/UI规范.md §8：框选一律走那一份，不许各写一份）。
        self.btn_mmap_region = QPushButton("框选小地图")
        self.btn_mmap_region.setToolTip(
            "在**实时画面**上把游戏的小地图面板框出来（存 config/live.yaml）。\n\n"
            "这是必做的一步，两个用途：\n"
            "  · 「在实时画面上叠地形图」得知道往画面的哪儿画；\n"
            "  · 来源选「从实时画面框选」时，还要靠它把面板裁出来喂给标定弹窗。\n\n"
            "先到「实时」页点开始预览，看到画面里的游戏小地图再回来框。\n"
            "只框**面板本身**：多框进来的血条/聊天栏会一起算进去，匹配分会掉下来。\n"
            "框完当场和底图核对一次，匹配分显示在下面。\n\n"
            "画面尺寸变了（换分辨率 / 改推流参数）要重框一次。")
        self.btn_mmap_region.clicked.connect(self._pick_mmap_region)
        row2.addWidget(self.btn_mmap_region)

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
        row2.addWidget(self.btn_mmap_calib)
        row2.addStretch(1)
        root.addLayout(row2)

        # ---- 行3/4：标记跟踪（黄点的四个容差）----
        # 这一组**分两行**（同「小地图定位」那一组，docs/UI规范.md §4：一行放不下
        # 就另起一行，别顺着行尾续 —— 四个框挤一行时面板最小宽度会从 521 涨到 714，
        # 窗口一窄标签就先被压没）：
        #   行3「漏检怎么办」：沿用窗口 / 外推上限
        #   行4「怎么搜、怎么判噪声」：搜索半径 / 跳变上限
        # 四个都是"这一帧的黄点要不要信"：小 → 反应快、抗噪差；大 → 稳、但会跟丢。
        # 存 config/live.yaml（和「来源」「框选」一样：取决于本机画面与帧率，与地图无关）。
        self._sp_track = {}          # 配置键名 → 输入框（键名以 mm.TRACK_KEYS 为准）

        def _track_spin(key, name, lo, hi, suffix, tip):
            """造一个容差输入框：滚轮不改值（UI规范 §5）、带单位、说明进 tooltip。"""
            sp = NoWheelSpinBox()
            sp.setRange(lo, hi)
            sp.setSuffix(suffix)
            sp.setFixedWidth(84)
            sp.setToolTip("%s：\n%s" % (name, tip))
            sp.valueChanged.connect(self._on_mmap_track)
            self._sp_track[key] = sp
            return sp

        # 行3：漏检怎么办（沿用 + 外推）
        row4 = QHBoxLayout()
        row4.setSpacing(6)
        row4.addWidget(QLabel("标记跟踪"))
        row4.addWidget(QLabel("沿用"))
        row4.addWidget(_track_spin(
            "mmap_hold_ms", "沿用", 0, 5000, " ms",
            "黄点这一拍认不出时，**沿用上一帧位置**多久（毫秒）。\n"
            "窗口内位置会按速度外推一点点；窗口过了才当真跟丢。\n"
            "0 = 不用这条（认不出就直接说认不出）。"))
        row4.addWidget(QLabel("外推上限"))
        row4.addWidget(_track_spin(
            "mmap_ghost_shift", "外推上限", 0, 100, " px",
            "沿用期间位置最多往外推这么多像素（面板像素，1 px ≈ 16 世界像素）。"))
        row4.addStretch(1)
        root.addLayout(row4)

        # 行4：怎么搜、怎么判噪声（搜索半径 + 跳变上限）。**别往上面那行续加**。
        row5 = QHBoxLayout()
        row5.setSpacing(6)
        row5.addWidget(QLabel(""))
        row5.addWidget(QLabel("搜索半径"))
        row5.addWidget(_track_spin(
            "mmap_roi_pad", "搜索半径", 0, 200, " px",
            "有上一帧位置时，只在它周围这么大一块里找（像素）。\n"
            "实际半径还会自动放大到能盖住黄点本身（面板大、点也大）。\n"
            "越小越快；人跑得快时找不到会自动退回全画面重找。"))
        row5.addWidget(QLabel("跳变上限"))
        row5.addWidget(_track_spin(
            "mmap_max_jump", "跳变上限", 0, 500, " px",
            "两拍之间位置跳超过这么多像素就当噪声：丢掉位置、下一拍重捕。\n"
            "调小 = 更不信突变（适合跟丢少、噪声多的画面）；0 = 不判跳变。"))
        row5.addStretch(1)
        root.addLayout(row5)

        # ---- 世界坐标那行 ----
        # 摆在**「框选小地图」这一行的下面**：它读的就是画面里框出来的那块面板，
        # 挨着放才看得出"这行数是从那块画面算出来的"。
        # 只在勾上「在实时画面上叠地形图」时才显示（要读数就得先看那块面板对不对）。
        self.lbl_mmap_world = QLabel()
        self.lbl_mmap_world.setStyleSheet("color: #80868b;")
        self.lbl_mmap_world.setWordWrap(True)
        self.lbl_mmap_world.setVisible(False)
        root.addWidget(self.lbl_mmap_world)
        #: 玩家定位器：面板画面 → 世界坐标 → 在哪条段（perception.minimap）。
        #: **和实时线程各持一份**：那一份写进 WorldState（决策用），这一份只做读数；
        #: 共用一份会让两边的跨帧跟踪互相打乱（同一套跳变判据被两边各推一次）。
        self._locator = mm.PlayerLocator()
        self._world_note = ""           # 这行字（也贴到画面里那块框下面）
        self._ov_pix = None             # 叠图源图缓存（见 _overlay_pix）
        self._world_timer = QTimer(self)
        self._world_timer.setInterval(250)      # 4 次/秒：够看，又不占主线程
        self._world_timer.timeout.connect(self._tick_world)

        # ---- 第三行：把叠图画到实时画面上 ----
        # **纯显示层**：只在 Qt 那一层往缩小后的画面上补一层，numpy 帧一个字节
        # 都不改 —— 实时画面那帧还要喂探针/血条/小地图标定弹窗，烘进去会让
        # 标定弹窗拿叠图和它自己匹配（匹配分虚高）。成本实测多 0.06 ms/帧。
        row3 = QHBoxLayout()
        row3.setSpacing(6)
        self.ck_mmap_draw = QCheckBox("在实时画面上叠地形图")
        self.ck_mmap_draw.setToolTip(
            "把这张图的 `<id>_overlay.png`（按 foothold 画的地形图，平台形状）\n"
            "按**你标定出来的换算**半透明地画到实时画面的小地图面板上。\n\n"
            "看什么：叠上去和游戏小地图里的地形**重合不重合** —— 不重合就说明\n"
            "标定不对（或者小地图面板的位置框错了）。第一版是**静止**的：\n"
            "几何用标定时那一份，不会跟着人物走（跟着走要每帧重新定位，见寻路设计）。\n\n"
            "为什么只叠地形图：平台形状和 foothold 是同一套坐标，能直接看出\n"
            "「程序以为你在哪块平台上」；小地图底图是 WZ 的素材画，没这个信息。\n\n"
            "浓淡沿用「标定…」弹窗里那个透明度滑块（一起存在标定文件里）。\n"
            "没画出来时，右边会写明卡在哪一步。")
        self.ck_mmap_draw.toggled.connect(self._on_mmap_draw)
        row3.addWidget(self.ck_mmap_draw)
        self.lbl_mmap_draw = QLabel()
        self.lbl_mmap_draw.setStyleSheet("color: #80868b;")
        self.lbl_mmap_draw.setWordWrap(True)
        row3.addWidget(self.lbl_mmap_draw, 1)
        root.addLayout(row3)

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
        """换来源：立刻存下来，并刷一遍状态。

        「框选小地图」**不跟着来源显隐**了 —— 它记的是面板在画面里的位置，
        两种来源都要用（收流时主画面里也有小地图，只是被压过）。
        """
        src = self.cmb_mmap_src.currentData()
        update_live(mmap_src=src)
        self._refresh_mmap()

    def _pick_mmap_region(self):
        """在**实时画面**上框出小地图面板在哪儿（**必做的一步**，与来源无关）。

        两个用途：「在实时画面上叠地形图」要知道往哪画；来源选「从实时画面框选」
        时还要靠它把面板裁出来喂标定弹窗。收流来源同样需要 —— 主画面里也有
        小地图面板，只是被 H.264 压过。

        规范（docs/UI规范.md §8）：框选一律走 `gui/region_picker`（放大镜、
        `+/-` 调倍率、Esc 取消、**<4px 当误点**）；**框完当场验证** —— 小地图
        这条的验证就是"这块画面和底图能对上多少分"，所以框完立刻算一次分。
        """
        lp = getattr(self, "live_panel", None)
        frame = None if lp is None else lp.current_frame()
        if frame is None:
            QMessageBox.information(
                self, "先开始预览",
                "框选要在实时画面上做：先到「实时」页点「开始」，\n"
                "看到画面（含游戏的小地图面板）再回来框。")
            return
        from gui.region_selector import select_region_on_image

        rect = select_region_on_image(frame, self)
        if rect is None:
            return                  # 取消（<4px 的框在框选里就按误点丢掉了）
        # **只存位置，不动来源**：框面板的位置这件事和"画面从哪来"是两回事，
        # 以前这里顺手把来源改成「从实时画面框选」—— 那等于替用户改了另一个设置。
        update_live(mmap_crop=[int(v) for v in rect])
        # _refresh_mmap 里会把叠图按新位置重算（于是框完就能看到它画上去），
        # 所以这里只调它一个 —— 别再多调一次 _refresh_overlay（会白重算一遍）。
        self._refresh_mmap()
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
                                 client=client, parent=self,
                                 src=self._mmap_src())
        dlg.exec_()
        if dlg.saved:
            self._refresh_mmap()      # 状态行立刻变成「已标定 … 匹配分 …」

    def _refresh_mmap(self):
        mid = self._map_id()
        p = getattr(self, "project", None)
        # 来源要先读：**标定是按来源分开存的**，读错那一份就是错的几何
        # （两条来源的面板尺寸差 5.6 倍，见 core/mapdata.calib_path）。
        src_kind = self._mmap_src()
        cal = mapdata.load_calib(mid, src_kind) if mid else None

        # 状态行颜色归位：框选验证时会染成绿/橙/红，任何一次刷新都该回到中性灰
        self.lbl_mmap_hint.setStyleSheet("color: #80868b;")
        # 来源跟着配置走（blockSignals：不然 setCurrentIndex 会反过来写回去）
        self.cmb_mmap_src.blockSignals(True)
        j = self.cmb_mmap_src.findData(src_kind)
        self.cmb_mmap_src.setCurrentIndex(j if j >= 0 else 0)
        self.cmb_mmap_src.blockSignals(False)
        # 「框选小地图」**不跟着来源显隐**：它是必做的一步（叠图要知道往哪画；
        # 「从实时画面框选」那条还要靠它裁面板），见 _pick_mmap_region。

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
        elif not load_live().get("mmap_crop"):
            # 框选是**必做的一步**（叠图得知道往哪画），而且流程上先框位置、再量
            # 换算 —— 所以它在「未标定」前面报，免得人来回跑两趟。
            self.lbl_mmap_hint.setText("还没框选小地图在画面里的位置")
            self.lbl_mmap_hint.setToolTip(
                "「在实时画面上叠地形图」要知道小地图面板在画面的哪儿；\n"
                "来源选「从实时画面框选」时，还要靠它把面板裁出来喂给标定弹窗。\n\n"
                "点右边的「框选小地图」，在实时画面上框出游戏的小地图面板。")
        elif not mm.has_geometry(cal):
            # **判的是「有没有几何」，不是「score 有没有」**：手工对齐没有
            # 匹配分（score=0），但它照样是一份能用的标定 —— 拿 score 判会让人
            # 看到「未标定」，以为自己白标了（见 perception/minimap.has_geometry）。
            #
            # 标定按来源分开存：**另一条来源标过，不代表这条标过** —— 那样算出来的
            # 世界坐标整体错。所以这里要把「另一条标过」直接说出来，人才知道
            # 不是自己白标了、而是切了来源。
            others = [mm.SRC_LABEL.get(s, s) for s, c in
                      mapdata.load_calibs(mid).items()
                      if s and s != src_kind and mm.has_geometry(c)]
            self.lbl_mmap_hint.setText(
                "未标定（当前来源：%s%s）"
                % (mm.SRC_LABEL.get(src_kind, src_kind),
                   "；%s 已标过" % "、".join(others) if others else ""))
            if src_kind == mm.SRC_LIVE:
                # 来源是"从实时画面框选"时，前置条件完全不同 —— 别让人去 A 机
                # 找那口推流（他可能压根没在用）。
                tip = ("这张图还没量过「面板像素 → 底图像素」的换算。\n"
                       "来源是「从实时画面框选」：小地图那块已经从实时画面上框出来了\n"
                       "（「框选小地图」），点「标定…」量一次就能对上（看得见重合）。")
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
            # 老格式（没记来源）**不能**写成「来源：从实时画面」—— 那是猜的。
            # 这份几何只对当时那条来源成立，说成当前来源就是让人放心用错的数。
            label = ("来源未记·老格式" if cal.get("legacy")
                     else mm.SRC_LABEL.get(src_kind, src_kind))
            self.lbl_mmap_hint.setText(
                "已标定（%s）　%s　%s　%s"
                % (label, self._mode_label(cal["mode"]), geo, how))
            crop = load_live().get("mmap_crop")
            legacy_note = (
                "⚠ 这份标定是**老格式**（没有记来源）—— 它只对当时那条来源成立。\n"
                "如果你换过小地图来源，请重量一次并保存（保存后就会归到当前来源）。\n\n"
                if cal.get("legacy") else "")
            self.lbl_mmap_hint.setToolTip(
                legacy_note +
                "量出来的几何参数：缩放 / 偏移 / 显示区起点。\n"
                "改「显示方式」只换方式，这些数不会丢。\n\n"
                "画面来源：%s（**标定按来源分开存**，换来源要重量一次 —— "
                "两条来源的面板尺寸不一样，同一份几何在另一条上算出来的位置是错的）\n"
                "小地图在画面里的位置（config/live.yaml 的 mmap_crop）：%s"
                % (mm.SRC_LABEL.get(src_kind, src_kind),
                   crop or "（还没框）"))

        # 叠图开关：读配置回填（blockSignals —— 回填别把配置又写一遍），
        # 再照当前状态重算一次「画什么、往哪画、没画的话卡在哪一步」。
        self.ck_mmap_draw.blockSignals(True)
        self.ck_mmap_draw.setChecked(self._mmap_draw_on())
        self.ck_mmap_draw.blockSignals(False)
        self.ck_mmap_draw.setEnabled(bool(mid))
        self._refresh_overlay()
        # 跟踪参数回填（blockSignals —— 回填别把配置又写一遍）
        trk = mm.track_params(load_live())
        for k, sp in self._sp_track.items():
            sp.blockSignals(True)
            sp.setValue(int(round(trk[k])))
            sp.blockSignals(False)
        # 世界坐标那行（勾上叠图才显示）：跟着地图一起刷新 —— 换了图，地形和
        # 标定都换了，读数必须重算，否则显示的是上一张图的段号。
        self._locator.load(mid or None)
        self._locator.use_track_config(trk)   # load() 会重建 tracker，参数要再喂一次
        self._refresh_world()
        self._push_mmap()

    # ---------------- 把叠图画到实时画面上（显示层）----------------

    def _mmap_draw_on(self):
        """要不要把叠图画到实时画面上（config/live.yaml 的 `mmap_draw`，默认关）。"""
        return bool(load_live().get("mmap_draw", False))

    def _on_mmap_draw(self, state):
        update_live(mmap_draw=bool(state))
        self._refresh_overlay()
        self._refresh_world()       # 世界坐标那行跟着这个开关显隐

    # ---------------- 玩家世界坐标（读数）----------------

    def _mmap_track_values(self):
        """那排输入框现在的值 → {配置键名: 数值}。"""
        return {k: sp.value() for k, sp in self._sp_track.items()}

    def _on_mmap_track(self, _v=None):
        """改黄点容差：**立刻存配置 + 立刻生效**（不用重开预览）。

        面板自己那份 locator 当场改；实时线程那份靠 `_push_mmap` 推过去 ——
        两边各持一份（见 _tick_world 的说明），参数必须一起同步。
        """
        vals = self._mmap_track_values()
        update_live(**vals)
        self._locator.use_track_config(vals)
        self._push_mmap()

    def _push_mmap(self):
        """把「小地图定位」要的三样推给实时线程（切项目 / 换来源 / 重框都要叫）。

        实时线程那边靠它才知道：用哪张图的地形与标定、面板从哪来、裁哪一块。
        没跑就只存在实时面板上，等下次 start() 由参数带进去。
        """
        lp = getattr(self, "live_panel", None)
        if lp is None or not hasattr(lp, "set_mmap"):
            return
        cfg = load_live()
        crop = cfg.get("mmap_crop") or []
        lp.set_mmap(map_id=self._map_id() or "",
                    src=self._mmap_src(),
                    crop=[int(v) for v in crop] if len(crop) == 4 else None,
                    track=mm.track_params(cfg))

    def _refresh_world(self):
        """世界坐标那行的显隐与节拍：只在勾上「叠地形图」时才跑。"""
        on = self._mmap_draw_on()
        self.lbl_mmap_world.setVisible(on)
        if on:
            if not self._world_timer.isActive():
                self._world_timer.start()
            self._tick_world()              # 立刻出一次数，别让人等 250ms
        else:
            if self._world_timer.isActive():
                self._world_timer.stop()
            self._world_note = ""

    def _tick_world(self):
        """算一次玩家世界坐标并显示（只在这行可见时跑）。

        **面板取的是「实时画面里框出来的那一块」**（`mmap_crop`），所以标定也取
        「从实时画面」那一份 —— 和叠图看的是同一块像素。读数与叠图因此永远对得上；
        换成别条来源的面板尺寸都不一样（实测差 5.6 倍），标定会错位。
        """
        def _say(text, color="#80868b", tip="", osd=None):
            """`text` 给面板那一行，`osd` 给**贴在画面上**的那行（默认同 text）。

            **两行分开**：画面那行是一行不换行的字，长一点就横穿整个画面（实测
            "认不出黄点"那条诊断有 150+ 字，贴上去就是一条黑带盖住半屏、还看不清）。
            所以面板上可以写全，画面上只用一句话版本，详情进 tooltip。
            """
            self._world_note = osd if osd is not None else text
            self.lbl_mmap_world.setText(text)
            self.lbl_mmap_world.setStyleSheet("color: %s;" % color)
            self.lbl_mmap_world.setToolTip(tip)
            # 同一句话贴在画面里那块框的下面（见 live_panel._draw_note）。
            # **只换那行字**：这条路每 250ms 走一次，整算叠图会连地形 JSON 和
            # 底图 PNG 一起重读（几十毫秒 × 4 次/秒，全在 GUI 主线程上）。
            lp2 = getattr(self, "live_panel", None)
            if lp2 is None or not lp2.set_overlay_note(self._world_note):
                self._refresh_overlay()

        if not self._mmap_draw_on():
            return
        lp = getattr(self, "live_panel", None)
        mid = self._map_id()
        if lp is None or not mid:
            return _say("玩家世界坐标：（没打开项目 / 没选地图）")
        frame = lp.current_frame()
        crop = load_live().get("mmap_crop") or []
        if frame is None:
            return _say("玩家世界坐标：先到「实时」页点「开始」预览")
        if len(crop) != 4:
            return _say("玩家世界坐标：先「框选小地图」（读数要从那块画面算）")
        x, y, w, h = [int(v) for v in crop]
        H, W = frame.shape[:2]
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > W or y + h > H:
            return _say("玩家世界坐标：框选的区域超出当前画面 %d×%d —— 重框一次"
                        % (W, H))

        # 复制一份再用：那一帧是实时线程的就地画布（上面有玩家框/平台线），而且
        # 随时会被下一帧覆盖 —— 直接切视图会读到写了一半的像素。
        panel = frame[y:y + h, x:x + w].copy()
        loc = self._locator.load(mid)
        r = loc.update(panel, src=mm.SRC_LIVE)
        if not r["ok"]:
            # 面板上写一句话，**详情进 tooltip**；画面上那行更短（见 _say 的说明）——
            # 完整诊断有一百多字，贴到画面上就是一条横穿半屏的黑带。
            return _say("玩家世界坐标：%s（鼠标放上去看详情）" % r["short"],
                        "#b06000",
                        "%s\n\n黄点那一层：%s" % (r["note"], r["dot"]),
                        osd="世界坐标：%s" % r["short"])
        seg = r["segment_id"]
        head = "玩家世界坐标 (%d, %d)" % (round(r["world_x"]), round(r["world_y"]))
        if seg is None:
            return _say("%s　脚下没有平台" % head, "#b06000", r["note"],
                        osd="世界 (%d, %d)　没站在平台上"
                            % (round(r["world_x"]), round(r["world_y"])))
        color = "#188038" if r["confirmed"] else "#b06000"
        # 「上一帧」= 这一拍没认出黄点、位置沿用上一帧（防抖窗口内）—— 标出来，
        # 别让人以为这是这一拍量到的
        held = "（上一帧）" if r.get("held") else ""
        tail = "" if r["confirmed"] else "（还没连续确认）"
        return _say("%s　第 %d 段%s%s" % (head, seg, held, tail),
                    color,
                    "来源：从实时画面　标定：%s%s"
                    % (mm.SRC_LABEL[mm.SRC_LIVE],
                       ("\n\n" + r["note"]) if r.get("held") else ""),
                    osd="世界 (%d, %d)　第 %d 段%s"
                        % (round(r["world_x"]), round(r["world_y"]), seg, held))

    def _overlay_blocker(self):
        """现在画不了的话卡在哪一步（空串 = 能画）。

        四种卡法都要能说出来 —— 这一块最容易变成"勾了没反应"，而人对着一个
        没反应的勾选框只能猜。
        """
        mid = self._map_id()
        if not mid:
            return "还没选地图"
        if not mm.has_geometry(mapdata.load_calib(mid, self._mmap_src()) or {}):
            return ("这张图在当前小地图来源下还没标定（点右边「标定…」量一次并保存）")
        if not load_live().get("mmap_crop"):
            return "还没框选小地图在画面里的位置（点「框选小地图」）"
        if not (mapdata.map_dir() / ("%s_overlay.png" % mid)).exists():
            return "还没有 %s_overlay.png（先点下面的「生成地形图」）" % mid
        t = mapdata.load(mid, with_canvas=True)
        if t is None or t.canvas is None:
            return "这张图还没有小地图底图（先点「生成地形图」）"
        return ""

    def _overlay_pix(self, mid):
        """叠图源图（`<id>_overlay.png`）的 QPixmap，**带缓存**。

        为什么要缓存：世界坐标那行每 250ms 重贴一次，而这条路会重画叠图 ——
        每次都从磁盘读一张 200KB 级的 PNG（4 次/秒）纯属浪费，而且卡主线程。
        """
        p = str(mapdata.map_dir() / ("%s_overlay.png" % mid))
        if self._ov_pix is None or self._ov_pix[0] != p:
            self._ov_pix = (p, QPixmap(p))
        return self._ov_pix[1]

    def _refresh_overlay(self):
        """照标定算出「叠图的哪一块画到画面的哪里」，交给实时面板去画。

        算出来的矩形是**画面坐标**：面板在画面里的位置来自 `mmap_crop`，
        面板里怎么摆来自标定（`perception/minimap.frame_overlay_rects` 一份公式，
        和标定弹窗的叠加层同源）。

        画/没画、画的哪儿、没画卡在哪 —— 都写在勾选框旁边那行上。
        """
        lp = getattr(self, "live_panel", None)
        on = self._mmap_draw_on()
        why = self._overlay_blocker()
        if on and not why and lp is not None:
            mid = self._map_id()
            t = mapdata.load(mid, with_canvas=True)
            crop = load_live().get("mmap_crop") or []
            if len(crop) == 4 and t is not None and t.canvas is not None:
                cal = mapdata.load_calib(mid, self._mmap_src()) or {}
                src, dst = mm.frame_overlay_rects(
                    cal, [int(v) for v in crop],
                    (t.canvas.shape[1], t.canvas.shape[0]))
                pix = self._overlay_pix(mid)
                alpha = float(cal.get("alpha") or 55) / 100.0
                # `note` = 玩家世界坐标那行：贴在画面里那块框的下面（见
                # gui/live_panel._draw_note）。每次重画都要重贴 —— 帧是新的。
                lp.set_minimap_overlay(pix, src, dst, alpha,
                                       note=self._world_note)
                self.lbl_mmap_draw.setText(
                    "已画在画面 (%d, %d) %d×%d　浓淡 %.0f%%（在「标定…」里调）"
                    % (dst[0], dst[1], dst[2], dst[3], alpha * 100))
                self.lbl_mmap_draw.setStyleSheet("color:#188038;")
                return
            why = "算不出往画面的哪儿画（框选/标定/底图不齐全）"
        if lp is not None:
            lp.set_minimap_overlay(None)
        if not on:
            self.lbl_mmap_draw.setText("（没开）")
            self.lbl_mmap_draw.setStyleSheet("color:#80868b;")
        else:
            self.lbl_mmap_draw.setText("没画：%s" % why)
            self.lbl_mmap_draw.setStyleSheet("color:#b06000;")

    # ---------------- 地形图 ----------------

    def _map_image_path(self, mid):
        """这张图该显示哪张图 → (路径或 None, 标题, 附注)。

        优先 `<id>_overlay.png`：那是 tools/map_terrain_view.py 画出来的"看得懂的
        地形图"（每条平台一色、黄=传送点、青=绳梯、品红=刷怪点）。没有就退到
        小地图底图 `<id>.png`；都没有说明地形还没导出，把该跑的命令写出来。

        **分三段返回**（不是拼成一整句）：中间那段「几×几 像素」得**量了文件**才知道，
        而那只 QPixmap 在调用方（它本来就要为显示加载一次，不多花一次解码）。
        拼成一整句的话，"几×几"要么掉到命令行提示后面，要么得在这儿再解码一遍。
        """
        d = mapdata.map_dir()
        over = d / ("%s_overlay.png" % mid)
        if over.exists():
            return over, ("地形叠加图 %s" % over.name), (
                "（每条平台一色 · 黄=传送点 · 青=绳梯 · 品红=刷怪点）")
        base = d / ("%s.png" % mid)
        if base.exists():
            return base, ("小地图底图 %s" % base.name), (
                "（还没有叠加图 —— 点下面的「生成地形图」画一张看得更清楚的；"
                "也可以手跑：\n    python -m tools.map_terrain_view %s）" % mid)
        return None, "这张图还没有地形数据", (
            "（datasets/map/%s.json 不存在。）\n"
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
            path, title, extra = self._map_image_path(mid)
            pm = QPixmap(str(path)) if path is not None else QPixmap()
            # 「几×几」放在**标题后面、附注前面**：它是这张图的第一眼信息，
            # 掉到那串命令行提示后面就没人看得见了。数用**真文件**量出来的
            # （`pm` 本来就要为显示加载，不多花一次解码）—— **不是推算**：
            # 推算一旦和实际文件不一致（换过图、改过生成参数），报出来的就是
            # 假数，而"看着像对的假数"最难发现。
            size = "" if pm.isNull() else "%d×%d 像素" % (pm.width(), pm.height())
            self.lbl_map_img.setText("　".join(
                x for x in (title, size, extra) if x))
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
        self._ov_pix = None                # 叠加图重画了 → 丢掉缓存的旧图
        self._refresh_map_image()          # 画好了立刻换图
        self._refresh_overlay()            # 叠到实时画面那一层也要换成新的
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
        src_kind = self._mmap_src()
        cal = mapdata.load_calib(mid, src_kind) or {}
        # **只改方式**，量出来的几何参数（scale/offset/view）保留
        cal["mode"] = self.cmb_mmap_mode.currentData()
        cal["picked_by"] = "gui"
        cal.pop("legacy", None)
        # 带来源：不带就会整份覆盖，把另一条来源的标定抹掉（人看不出来）
        mapdata.save_calib(mid, cal, src_kind)
        self._refresh_mmap()        # 里面会顺带按新方式重算叠图（fit/crop 画法不同）

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
