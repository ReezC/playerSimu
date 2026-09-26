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

**地形图**：显示 `datasets/map/<id>_zones.png` —— **地形编辑器的结果**（圈进集合的
平台按集合颜色画 + 名字标在平台上方，没圈的画暗灰）。"程序认得的平台"就是它，
所以寻路要往哪走、小地图定位对不对，看着这张图才有概念。

同目录还留一张 `<id>_overlay.png`（每段一色 + 段号，`tools/map_terrain_view.py`
生成）：**叠到实时画面上用的仍是它** —— 那里要看的是"所有几何位置对不对"，
不是"我圈了哪几块"。两张都由「生成地形图」一起画。
"""

import time

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QApplication, QCheckBox, QGroupBox, QHBoxLayout,
                             QLabel, QMessageBox, QPushButton, QVBoxLayout,
                             QWidget)

from core import mapdata
from decision.agent import settings
from gui.canvas import ImageCanvas
from gui.minimap_calib import MinimapCalibDialog   # 量「面板 ↔ 底图」的弹窗
from gui.widgets import (NoWheelComboBox, NoWheelDoubleSpinBox,
                         NoWheelSlider, NoWheelSpinBox)   # 滚轮不许改参数（UI规范 §5）
from perception import minimap as mm
from tools.config import load_live, update_live    # 来源/框选区域存 config/live.yaml


def _mmss(sec):
    """秒 → `M:SS`（倒计时统一这个写法，和 player_panel 那边一致）。"""
    sec = max(0, int(sec))
    return "%d:%02d" % (sec // 60, sec % 60)


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

    from core import zones as zones_mod
    from tools.map_terrain_view import render
    t = mapdata.load(mid, with_canvas=True)
    if t is None:
        raise RuntimeError("读不到地形 JSON：%s" % json_path)

    target = out / ("%s_overlay.png" % mid)
    # render 返回的是 foothold 条数（其中墙多少），不是"串成的段数" —— 别写错标签
    w, h, n_fh, n_wall = render(t, target)
    ctx.log("叠加图 %s  %d×%d（foothold %d / 其中墙 %d，串成 %d 段）"
            % (target.name, w, h, n_fh, n_wall, len(t.segments)), "ok")

    # 第二张：**地形编辑器的结果**（颜色=集合、名字标在平台上）——「路线识别」的
    # 「地形图」显示的是它。两张都画：叠图那版仍要留着（叠到实时画面上时，要看的是
    # "所有几何位置对不对"，不是"我圈了哪几块"）。都只有毫秒级。
    z = zones_mod.load(mid)
    ztarget = out / ("%s_zones.png" % mid)
    render(t, ztarget, zones=z)
    ctx.log("集合图 %s（%d 个集合%s）"
            % (ztarget.name, len(z.sets),
               "；还没圈集合，点「编辑集合…」" if not z.sets else ""), "ok")
    return {"summary": "已生成 %s" % ztarget.name, "path": str(ztarget)}


class RoutePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        page = QVBoxLayout(self)
        page.setContentsMargins(12, 12, 12, 12)
        page.setSpacing(8)

        def card(title, tip="", stretch=0):
            """按 UI 规范把**一组**控件装进一张卡片 → 返回卡片内的布局。

            为什么是 `QGroupBox` 而不是工作流那套 `StepCard`：那个自带状态灯和运行
            按钮，是给"一步一步的任务"用的；这里是**参数页** —— 项目里参数分组一律
            写 `QGroupBox("标题")`（`gui/player_panel.py` 里 12 处都是），白底圆角 +
            灰色小标题的样式由 `main_window` 的全局 QSS 给（`QGroupBox` /
            `QGroupBox::title`）。**别在这儿自己写样式**：写了就跟着主题走不动了。

            ⚠ 顺序仍然是**代码顺序 = 视觉顺序**（docs/UI规范.md §4）：卡片按下面
            出现的先后从上到下排，要挪位置就整块搬（见下面三次 `root = card(...)`）。
            """
            box = QGroupBox(title)
            if tip:
                box.setToolTip(tip)
            lay = QVBoxLayout(box)
            lay.setContentsMargins(10, 6, 10, 8)
            lay.setSpacing(6)
            page.addWidget(box, stretch)
            return lay

        # ---- ① 路线识别：集合 + 执行器参数 ----
        # 下面这段所有 `root.add*` 都落进这张卡片（`root` 依次指向各卡片的内布局）。
        root = card("路线识别")

        # 这一组**只剩「编辑集合」**（2026-09-26 用户定：其余都没意义）。
        # 原来上面还有一段说明 + `启用路线识别` 开关 + 一行状态提示 —— 那个开关门控的是
        # **感知**（平台识别 / 落点预测 / 小地图定位），关掉时「命令前往」连"你在哪个
        # 平台"都答不出来，正是"点一下没反应"的典型来源。现在这些感知**一直跑**
        # （见 gui/live_thread.py），页面上只留真正要人动手的那一件事。
        # foothold 集合编辑器：寻路模块的第一块（「我在不在 A 平台」靠它）
        self.btn_zones = QPushButton("编辑集合…")
        self.btn_zones.setToolTip(
            "打开 foothold 集合编辑器：把这张图的地形画出来，点选/框选 foothold\n"
            "注册成命名集合（存 datasets/map/<id>.zones.json，**按地图 id 一份**）。\n\n"
            "「我在不在 A 平台」这条判据就靠它：自动串段会把「要跳/攀才能互通」的\n"
            "并成同一段（实测 105090600 的第 0 段把一面 388 像素高的悬崖当成了平台\n"
            "边缘），所以分组只能由人圈。详见 docs/寻路设计.md §12。")
        self.btn_zones.clicked.connect(self._on_edit_zones)
        root.addWidget(self.btn_zones)

        # ---- 上绳梯失败后的**延迟激活**（2026-09-26 用户要求 1）----
        # 以前失败了是**立即**重新激活 —— 失败那一下人往往还在原地、朝向也没变，
        # 立刻重来容易在同一处再歪一次。这个参数只管"等多久再重来"，不碰感知
        # （感知没有开关，见上面那段说明）。
        row_retry = QHBoxLayout()
        row_retry.setSpacing(6)
        row_retry.addWidget(QLabel("上绳梯失败后延迟激活时间"))
        self.sp_retry = NoWheelDoubleSpinBox()
        self.sp_retry.setRange(0.0, 30.0)
        self.sp_retry.setDecimals(1)
        self.sp_retry.setSingleStep(0.5)
        self.sp_retry.setSuffix(" s")
        self.sp_retry.setMinimumWidth(90)
        self.sp_retry.setValue(float(getattr(settings, "climb_retry_delay_s", 1.0)))
        self.sp_retry.setToolTip(
            "上绳梯 / 下跳**失败后等多久**再重新激活（就是原来的\"重新对齐再来一次\"）。\n\n"
            "0 = 立即重来（老行为）。\n"
            "等一会儿的好处：失败那一下人往往还在原地、朝向也没变，立刻重来容易在同一处\n"
            "再歪一次；先站稳一小会儿再重来，成功率更高。\n\n"
            "⚠ 等待期间**不按键**（角色站着不动）；如果那时已经站在目标集合里，\n"
            "任务会**直接收工**，不再重来。")
        self.sp_retry.valueChanged.connect(self._on_retry_delay)
        row_retry.addWidget(self.sp_retry)
        row_retry.addStretch(1)
        root.addLayout(row_retry)

        # ---- ② 小地图定位（寻路用；方式由你选，程序不猜）----
        root = card("小地图定位")

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

        # ---- 「坐标系偏移」(x, y)：算出来的世界坐标**加上**它（2026-09-26 用户要求）----
        # 单独一行（docs/UI规范.md §4：一行只放一个参数组）；存在 config/live.yaml 的
        # `mmap_world_offset`，和 `mmap_src` 同一份（B 机本地、与地图无关）。
        row_off = QHBoxLayout()
        row_off.setSpacing(6)
        row_off.addWidget(QLabel("坐标系偏移"))
        self._sp_off = []
        for _lbl in ("x", "y"):
            row_off.addWidget(QLabel(_lbl))
            sp = NoWheelSpinBox()
            sp.setRange(-5000, 5000)
            sp.setMinimumWidth(84)
            sp.setToolTip(
                "算出来的世界坐标会**加上**这个向量 —— 用来把"
                "「黄点重心」和你要的玩家原点（脚下/身体中心）对齐。\n\n"
                "怎么量：走到一个你知道确切世界坐标的点，看那行读数差多少，"
                "把差值填进来（读数是 算出来的 + 这里）。\n"
                "它只影响读数与寻路判断，**不碰标定**（面板↔底图那套照旧）。")
            sp.valueChanged.connect(self._on_world_offset)
            row_off.addWidget(sp)
            self._sp_off.append(sp)
        row_off.addStretch(1)
        root.addLayout(row_off)

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
        # 浓淡（透明度）：**勾上叠图才出现**（2026-09-26 用户要求）—— 以前只能进
        # 「标定…」弹窗里调，看一眼调一下要来回开窗。写回的是**同一处**（标定文件的
        # `alpha`，按来源分开存），所以两个入口永远一致。
        self.lbl_alpha = QLabel("浓淡")
        self.lbl_alpha.setVisible(False)
        row3.addWidget(self.lbl_alpha)
        self.sld_alpha = NoWheelSlider(Qt.Horizontal)
        self.sld_alpha.setRange(0, 100)
        self.sld_alpha.setMinimumWidth(120)
        self.sld_alpha.setToolTip(
            "叠图的透明度（0 = 看不见，100 = 完全实）。\n\n"
            "存进这张图的**标定文件**（按来源分开），和「标定…」弹窗里那个滑块是"
            "同一处 —— 在哪儿调都一样。\n\n"
            "看不清楚就调低一点：小地图面板本来就小，太实会盖住底图上的细节。")
        self.sld_alpha.valueChanged.connect(self._on_overlay_alpha)
        self.sld_alpha.setVisible(False)
        row3.addWidget(self.sld_alpha)
        self.lbl_mmap_draw = QLabel()
        self.lbl_mmap_draw.setStyleSheet("color: #80868b;")
        self.lbl_mmap_draw.setWordWrap(True)
        row3.addWidget(self.lbl_mmap_draw, 1)
        root.addLayout(row3)

        self.lbl_mmap_hint = QLabel()
        self.lbl_mmap_hint.setStyleSheet("color: #80868b;")
        self.lbl_mmap_hint.setWordWrap(True)
        root.addWidget(self.lbl_mmap_hint)

        # ---- ③ 地形图（看得见才好判断小地图定位对不对）----
        # 这张卡片吃掉剩余高度：里面的画布是 `addWidget(canvas, 1)`（见下）
        root = card("地形图", stretch=1)

        self.lbl_map_img = QLabel()
        self.lbl_map_img.setStyleSheet("color: #80868b;")
        self.lbl_map_img.setWordWrap(True)
        root.addWidget(self.lbl_map_img)

        # ---- 预览平台 + 命令前往（路线测试用）----
        # 下拉里是**地形编辑器里注册过的集合**（跟着集合文件走，编辑器一保存就刷新）。
        # 「预览」= 在下面那张图上把这个平台的包围盒框出来 —— 先确认"我要去的那个平台
        # 到底是哪块"，再去测路；「命令前往」现在先算一遍**这条路通不通**（走边图），
        # 等 P4 的执行器做出来，同一个按钮就真的让它走。
        row_goto = QHBoxLayout()
        row_goto.setSpacing(6)
        row_goto.addWidget(QLabel("预览平台"))
        self.cmb_goto = NoWheelComboBox()
        self.cmb_goto.setMinimumWidth(150)
        self.cmb_goto.setToolTip(
            "要去的平台（下拉里是**地形编辑器注册过的集合**）。\n\n"
            "选中就把它在下面的地形图上框出来 —— 先看清是哪块，再谈路怎么走。\n"
            "空项 = 不预览（图上不叠框）。\n\n"
            "⚠ 还没圈集合时这里是空的：先到「编辑集合…」里圈一个。")
        self.cmb_goto.currentIndexChanged.connect(self._on_goto_pick)
        row_goto.addWidget(self.cmb_goto)
        self.btn_goto = QPushButton("命令前往")
        self.btn_goto.setToolTip(
            "**现在**：按边图算一遍「从现在所在平台 → 预览的平台」通不通，\n"
            "把路线（哪一步是走/爬/传送门）写在下面那行里。\n"
            "走不到时会说出**边界**（从起点能到哪些集合）—— 那就是缺边的位置。\n\n"
            "**将来**（P4 的执行器）：同一个按钮会真的命令角色走过去，\n"
            "途中遇到怪先打（攻击优先仲裁），超时报警。")
        self.btn_goto.clicked.connect(self._on_goto)
        row_goto.addWidget(self.btn_goto)
        # 「结束当前寻路」（2026-09-26 用户要求 3）：撤掉挂着的上绳/下跳任务 ——
        # 也就是让画面那行「当前任务」回到"战斗"。挨着「命令前往」放：一开一关一对。
        self.btn_stop_goto = QPushButton("结束当前寻路")
        self.btn_stop_goto.setToolTip(
            "撤掉**当前挂着的寻路任务**（上绳 / 下跳），角色立刻回到全权战斗。\n\n"
            "挂任务的是「命令前往」；这个按钮就是它的对手 —— 命令走错了、或者\n"
            "你不想让它爬了，点这里（不用去关自动）。\n\n"
            "没有任务在跑时会明说，不会静悄悄什么都不做。")
        self.btn_stop_goto.clicked.connect(self._on_stop_goto)
        row_goto.addWidget(self.btn_stop_goto)
        row_goto.addStretch(1)
        root.addLayout(row_goto)

        self.lbl_goto = QLabel()
        self.lbl_goto.setStyleSheet("color: #80868b;")
        self.lbl_goto.setWordWrap(True)
        root.addWidget(self.lbl_goto)

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
        self._refresh_goto()

    # ---------------- 预览平台 / 命令前往（路线测试）----------------

    # ---------------- 预览平台 / 命令前往（路线测试）----------------

    def _refresh_goto(self):
        """「预览平台」下拉：按**地形编辑器注册过的集合**填（空项永远在最前）。

        也跟着集合文件走 —— 编辑器保存后（`_on_zones_saved`）与重开面板时都重读，
        所以新圈的集合不用重启就能选。
        """
        mid = self._map_id()
        cur = self._goto_name() or str(getattr(settings, "route_goto_set", "") or "")
        names = []
        if mid:
            try:
                from core import zones as zones_mod
                names = list(zones_mod.load(mid).sets)
            except Exception:                       # noqa: BLE001
                names = []      # 文件坏了不该让面板起不来（下面那句提示会说明）
        self.cmb_goto.blockSignals(True)
        self.cmb_goto.clear()
        self.cmb_goto.addItem("（不预览）", "")
        for n in names:
            self.cmb_goto.addItem(n, n)
        j = self.cmb_goto.findData(cur)
        self.cmb_goto.setCurrentIndex(j if j >= 0 else 0)
        self.cmb_goto.blockSignals(False)
        self.cmb_goto.setEnabled(bool(names))
        self.btn_goto.setEnabled(bool(names))
        if not names:
            self._say_goto("还没有平台集合 —— 点上面的「编辑集合…」圈一个（路线按集合走）。",
                           "#b06000")

    def _goto_name(self):
        v = self.cmb_goto.currentData()
        return str(v or "")

    def _preview_boxes(self, mid):
        """预览平台在图上那个框 → [(cls, x, y, w, h, manual)]，空 = 没选/算不出来。

        坐标走 `tools.map_terrain_view.image_xy` —— 和**画那张图**用的是同一套换算。
        自己再算一遍迟早会漂（漂了就是"框画在别处"，看着像集合圈错了，最难查）。
        """
        name = self._goto_name()
        if not (name and mid):
            return []
        try:
            from core import zones as zones_mod
            from tools.map_terrain_view import image_xy
            t = mapdata.load(mid, with_canvas=True)
            z = zones_mod.load(mid)
            s = z.sets.get(name)
            if t is None or s is None:
                return []
            sp = zones_mod.set_span(t, s.get("footholds") or [])
            if sp is None:
                return []
            x0, y0 = image_xy(t, sp[0], sp[2])
            x1, y1 = image_xy(t, sp[1], sp[3])
            # **最少 8 像素**：平台是一根横线（包围盒高度可能是 0），按原样框出来就是
            # 一条 1px 的发丝 —— 等于没框（预览的作用就是"看得见是哪块"）。不够就在
            # 中心两侧补到 8px。
            if x1 - x0 < 8:
                cx = (x0 + x1) / 2.0
                x0, x1 = cx - 4, cx + 4
            if y1 - y0 < 8:
                cy = (y0 + y1) / 2.0
                y0, y1 = cy - 4, cy + 4
            return [(1, x0, y0, x1 - x0, y1 - y0, False)]
        except Exception:                           # noqa: BLE001
            return []

    def _on_goto_pick(self):
        """选了平台 → 存进配置（跟着项目走）+ 在图上把它框出来。"""
        name = self._goto_name()
        if str(getattr(settings, "route_goto_set", "") or "") != name:
            settings.route_goto_set = name
            settings.save()     # 没打开项目时不落盘（见 decision/agent 的 set_save_hook）
        self._say_goto("" if name else "（不预览：图上看全集）")
        self._refresh_map_image()

    def _here_set(self, z, fresh_s=3.0):
        """现在所在的集合名（拿不到给空串）。

        `fresh_s`：定位读数的新鲜度上限。**过期的读数不能当起点** —— 人会走，
        拿几分钟前的位置算出来的路是错的，而且看着像"边图坏了"。
        """
        fid, ts = getattr(self, "_fh_seen", ("", 0.0))
        if not fid or time.monotonic() - ts > fresh_s:
            return ""
        names = z.set_of(str(fid))
        return names[0] if names else ""

    def _on_goto(self):
        """算一遍「现在所在平台 → 预览平台」，**并把第一步交给执行器**。

        两件事一起做：
          · 算路：通不通 + 沿途每一步靠什么过去（走/爬绳/跳/传送门）+ 走不到时的**边界**
            （从起点能到哪些集合）—— 那正是路线测试要的信息（缺哪条边）；
          · 下命令：第一步是「爬」或「下跳」就真的下发（2026-09-26 用户定，见
            `_command_first_step`）。以前这里**只算不走**，点了按钮角色一动不动。
        """
        mid = self._map_id()
        dst = self._goto_name()
        if not mid or not dst:
            self._say_goto("先在左边选一个平台。", "#b06000")
            return
        try:
            from core import zones as zones_mod
            z = zones_mod.load(mid)
        except Exception as e:                      # noqa: BLE001
            self._say_goto("集合文件读不出来：%s" % e, "#c5221f")
            return
        src = self._here_set(z)
        if not src:
            self._say_goto("还不知道你在哪个平台：先勾「在实时画面上叠地形图」、"
                           "到「实时」页跑一小会儿（下面那行要显示得出「位于fh：…」）。",
                           "#b06000")
            return
        path, why = zones_mod.find_path(z, src, dst)
        if path is None:
            self._say_goto("%s" % why, "#c5221f")
            return
        cmd, sent = self._command_first_step(z, mid, path)
        steps = " → ".join(path)
        detail = "；".join("%s --%s-->"
                           % (a, zones_mod.edge_text(z, a, b))
                           for a, b in zip(path, path[1:]))
        here = z.set_of(str(getattr(self, "_fh_seen", ("", 0.0))[0]))
        note = ("（你现在同时属于 %d 个集合：%s；按「%s」当起点）"
                % (len(here), "、".join(here), src) if len(here) > 1 else "")
        # 命令没发出去 ⇒ 用**橙色**：那是在说"这一步现在做不到"，不是路线本身不对。
        self._say_goto("能走到（%s）：%s%s%s%s"
                       % (why, steps, ("　｜　" + detail) if detail else "",
                          note, cmd),
                       "#0b8043" if sent else "#b06000")

    def _on_stop_goto(self):
        """「结束当前寻路」：撤掉挂着的任务，让「当前任务」回到"战斗"。

        判定用 `current_goto_set()`（agent 的公开口径），**不去摸 `_climb`** ——
        那是私有的运行时状态。
        """
        ag = self._live_agent()
        if ag is None:
            self._say_goto("现在没有在跑的实时推理（先去「实时」页开始）—— "
                           "没有寻路任务可结束。", "#b06000")
            return
        if not self._current_goto_set():
            self._say_goto("现在没有寻路任务（当前任务：战斗）。", "#b06000")
            return
        ag.stop_climb("手动结束寻路")
        self._say_goto("已结束当前寻路任务（当前任务：战斗）。", "#0b8043")

    def _current_goto_set(self):
        """当前寻路任务的目标集合名（没有任务 / 实时没在跑时给空串）。"""
        ag = self._live_agent()
        if ag is None:
            return ""
        try:
            return str(ag.current_goto_set() or "")
        except Exception:                           # noqa: BLE001
            return ""

    def _timer_lines(self):
        """所有**计时任务**的名称 + 计时信息（一行一个）：休息最优先，其次自定义定时行为。

        数据全是 `decision.agent.settings` 上的**运行时状态**（agent 写、界面只读），
        口径和「决策参数」页那张休息卡片一致（见 `gui/player_panel._rest_text`、
        `_tick_feed_cd`）。⚠ 时间是 `time.monotonic()` 秒 —— 两侧同进程同一时钟源，
        所以这里直接减（别换成 wall clock，那会被系统对时带偏）。
        """
        out = []

        def left(until):
            if not until or until <= 0:
                return ""
            return "　剩余 %s" % _mmss(until - time.monotonic())

        st = getattr(settings, "rest_state", "")
        if st == "afk_enter":
            out.append("休息　进入隐身…%s" % left(settings.rest_until_monotonic))
        elif st == "afk_rest":
            out.append("休息　休息中%s" % left(settings.rest_until_monotonic))
        elif st == "afk_exit":
            out.append("休息　退出隐身…")
        elif getattr(settings, "rest_pending", False):
            # 到点了但攻击范围内还有怪：卡在这步最容易被当成"坏了"，写出来
            out.append("休息　待休息（等清空攻击范围内的怪）")
        elif getattr(settings, "next_afk_monotonic", 0.0) > 0:
            out.append("休息　下次%s" % left(settings.next_afk_monotonic))
        else:
            out.append("休息　未排期（防掉线关着）")
        for t in (getattr(settings, "custom_timers", None) or []):
            if t.get("paused"):
                # **暂停的不列**（2026-09-26 用户要求）：它现在不会触发，列出来只会
                # 让人以为"还有一项在跑"。暂停状态在「决策参数」页的列表里看得到。
                continue
            name = str(t.get("name") or "?")
            nx = float((getattr(settings, "custom_timer_next", None) or {})
                       .get(name, 0.0))
            # 还没排期（刚加/刚编辑过）⇒ 按区间下限占位，别显示 0:00
            tail = (left(nx) if nx > 0 else
                    "　剩余 %s" % _mmss((t.get("interval") or [5, 10])[0] * 60.0))
            out.append("定时行为「%s」%s" % (name, tail))
        if getattr(settings, "auto_feed_pet", False):
            # agent 还没排期（刚打开开关）⇒ 明写"未排期"，不要一行光秃秃的"喂宠"
            out.append("喂宠%s" % (left(getattr(settings, "feed_next_monotonic", 0.0))
                                  or "　未排期"))
        if int(getattr(settings, "resetall_interval", 0) or 0) > 0:
            # **要倒计时**，不写"每 N s"（2026-09-26 用户要求）：排期在 agent 的 tick
            # 里，所以它没在跑（或刚开自动还没排到）时只有"未排期"——
            # 公开口径见 `CombatAgent.resetall_left`，别在这儿摸 `_next_resetall`。
            ag = self._live_agent()
            try:
                sec = ag.resetall_left() if ag is not None else None
            except Exception:                       # noqa: BLE001
                sec = None
            out.append("定时清键%s" % ("　剩余 %s" % _mmss(sec) if sec is not None
                                     else "　未排期"))
        return out

    def _osd_lines(self, first):
        """画面那几行（小地图框下方，从上到下）：世界坐标 → **当前任务** → 定时任务。

        2026-09-26 用户要求 1、2：
          · 「当前任务」默认「战斗」（全权战斗 Agent），有寻路任务时「前往：{集合名}」；
          · 再往下**换行**罗列所有计时任务（休息最优先，其次自定义定时行为）；
          · 这几行**不要背景色**，字色取设置里的「定时任务颜色」。
        **合成一个列表**交给 live_panel（而不是分几次推）：它们要连成一片贴在同一处，
        分开推会出现"上一行的位置被下一行占掉"。
        """
        from gui import theme
        col = theme.load_vis().get("timer_color") or "#ffeb3b"
        dst = self._current_goto_set()
        lines = [first,                       # str ⇒ 保持黑底白字（读数是排查用的）
                 ("当前任务　%s" % ("前往：%s" % dst if dst else "战斗"), col, False)]
        lines += [(t, col, False) for t in self._timer_lines()]
        return lines

    def _live_agent(self):
        """当前在跑的 `CombatAgent`（没在跑给 None）——「命令前往」用它下发任务。

        它是 `decision/agent.py` 的模块级 `CURRENT`（「实时」线程启动时登记、退出时注销）：
        以前 agent 只是实时线程里的局部变量 ⇒ 面板够不着，只能"算路给你看"。
        """
        from decision import agent as agent_mod
        return getattr(agent_mod, "CURRENT", None)

    def _command_first_step(self, z, mid, path):
        """把路线的**第一步**交给执行器 ⇒ (一句给人看的话, 是否真的下发了命令)。

        2026-09-26 用户定：**只接「爬」/「下跳」**。这两种是"贴着绳 / 贴着边缘按上去"，
        对起跳时机不敏感（不必先做跳跃标定）；「走」「跳」「传送门」的执行器还没写 ——
        那时候**如实说**"这一步还没做"，而不是静悄悄什么都不发生（用户就是这么撞上的：
        点了「命令前往」角色一动不动，因为那时它只算路、根本没接线）。
        """
        from core import zones as zones_mod
        e = zones_mod.edge_between(z, path[0], path[1])
        if e is None:
            return "　（这两块之间找不到那条可达 —— 回「编辑集合…」看看）", False
        kind = e.get("kind") or ""
        if kind not in ("climb", "drop"):
            return ("　（第一步靠「%s」过去 —— 这种的执行器还没做，"
                    "现在只做到「爬」和「下跳」）"
                    % zones_mod.kind_label(kind)), False
        ag = self._live_agent()
        if ag is None:
            return ("　（这条命令要发给**实时**里跑着的角色 —— 先去「实时」页开始；"
                    "现在只算给你看）"), False
        try:
            from core import mapdata
            from decision import route as route_mod
            t = mapdata.load(mid, with_canvas=True)
            if t is None:
                return "　（读不到地形数据，造不出上绳任务）", False
            job = route_mod.job_for_edge(t, z, e,
                                         tol_px=int(settings.align_tol_px),
                                         hold_ms=int(settings.align_hold_ms))
        except ValueError as ex:            # 绳找不到 / 说不清上下 ⇒ 如实说，不猜
            return "　（没法下这条命令：%s）" % ex, False
        except Exception as ex:             # noqa: BLE001
            return "　（下命令时出错：%s）" % ex, False
        ag.start_climb(job)
        tail = ("" if len(path) <= 2 else
                "　（后面 %d 步不会自动接着走 —— 执行器现在只做爬/下跳）"
                % (len(path) - 2))
        return ("　｜　**已命令**：%s%s"
                % (zones_mod.edge_text(z, path[0], path[1]), tail)), True

    def _say_goto(self, text, color="#80868b"):
        self.lbl_goto.setText(text)
        self.lbl_goto.setStyleSheet("color: %s;" % color)

    def _map_id(self):
        p = getattr(self, "project", None)
        return (p.get("map_id") or "").strip() if p is not None else ""

    def _on_edit_zones(self):
        """打开/聚焦 foothold 集合编辑器（按当前项目的地图）。

        **非模态 + 单实例**（2026-09-26 改）：
          · 非模态：圈集合时要照着「实时」页的画面判断哪块是 A 平台、还要反复跑一下
            看世界坐标对不对 —— 模态对话框会把整个工作台锁住，只能不停地关窗开窗；
          · 单实例：编辑器挂在**主窗口**上（`win._zone_editor`），再点一次就把它提到
            前面来，不新开第二个窗口（两个窗口改同一份文件，谁覆盖谁说不清）。
        换地图时会关掉旧窗口重建 —— 否则那扇窗画的还是上一张图，人会以为"集合丢了"。

        **槽里抛异常 = 整个工作台 abort**（见 gui/worker.py 的说明），所以自己兜住：
        地形没导、文件坏了、地图 id 不对，都变成一句能看懂的话，而不是进程消失。
        """
        mid = self._map_id()
        if not mid:
            QMessageBox.information(self, "先选地图",
                                    "先在①「识别目标选项」里选地图 —— 集合是按地图存的。")
            return
        win = self.window()
        try:
            from gui.zone_editor import ZoneEditorDialog
            dlg = getattr(win, "_zone_editor", None)
            if dlg is not None and getattr(dlg, "map_id", "") != mid:
                dlg.close()
                dlg.deleteLater()
                dlg = None
            if dlg is None:
                dlg = ZoneEditorDialog(mid, parent=win)
                setattr(win, "_zone_editor", dlg)
            # 面板可能被重建过（换项目）⇒ 提示要接到**当前**这块面板上。
            # 记一下接到谁，避免同一个面板重复连接、旧的连接自然失效。
            if getattr(dlg, "_hint_to", None) is not self:
                dlg.saved_now.connect(self._on_zones_saved)
                dlg._hint_to = self
            # 窗口已经开着时（`show()` 对可见窗口是空操作、`raise_()` 也不触发
            # showEvent）要手动重读一次设置 —— 否则「设置 → 线条宽度」改完再点这个
            # 按钮，还是按旧的宽度画，人会以为设置没生效。
            if hasattr(dlg, "reload_line_width"):
                dlg.reload_line_width()
            dlg.show()                     # show() 而不是 exec_() —— 非模态
            dlg.raise_()                   # 已经在开着就提到前面来（切换焦点）
            dlg.activateWindow()
        except Exception as e:                      # noqa: BLE001
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "编辑器打不开", "%s: %s" % (type(e).__name__, e))

    def _on_zones_saved(self, map_id, n_sets):
        """编辑器保存后更新面板提示 + **立刻重画集合图**。

        为什么在这里重画：编辑器保存后，下面那张「地形图」显示的就是旧结果了
        （少一个集合、颜色也变了），而它正是用来看"圈得对不对"的 ——
        所以保存即刷新，不用人再去点一次「生成地形图」。
        """
        self._set_hint("集合已保存（%s：%d 个集合）" % (map_id, n_sets), "#0b8043")
        self._render_zones_png(map_id)
        self._refresh_map_image()
        self._refresh_goto()        # 新圈/改名的集合要能立刻在下拉里选到

    def _render_zones_png(self, mid):
        """把「地形编辑器的结果」画成 `<id>_zones.png`（同步跑，上百毫秒）。

        为什么不丢后台线程：这是一张 1280 宽的小图（实测上百毫秒），而它要**立刻**
        反映到上面的图里；后台线程是留给"导出 WZ"那种几十秒的活的。
        出错只打印 —— 画不出图不该把人挡在"保存成功"之外。
        """
        try:
            from core import zones as zones_mod
            from tools.map_terrain_view import render
            t = mapdata.load(mid, with_canvas=True)
            if t is None:
                return
            render(t, mapdata.map_dir() / ("%s_zones.png" % mid),
                   zones=zones_mod.load(mid))
        except Exception:                   # noqa: BLE001
            import traceback
            traceback.print_exc()

    def _load_zones(self, mid):
        """读这张图的 foothold 集合（**按 mtime 缓存**）→ `core.zones.Zones`。

        为什么看 mtime：编辑器和这块面板是**两个窗口**，那边一保存，这边 4 次/秒的
        读数要立刻跟着变 —— 但也没必要每次重读文件。
        """
        from core import zones
        p = zones.zones_path(mid)
        try:
            mt = p.stat().st_mtime if p.exists() else 0.0
        except OSError:
            mt = 0.0
        c = getattr(self, "_zones_cache", None)
        if c is not None and c[0] == mid and c[1] == mt:
            return c[2]
        z = zones.load(mid)
        self._zones_cache = (mid, mt, z)
        return z

    def _fh_zone(self, loc):
        """脚下那条 foothold 属于哪些集合 → 名字（空串 = 这条判据用不上）。

        **判据是「脚下的 foothold id ∈ 集合」，不是段号**（见 docs/寻路设计.md §12）：
        自动串段会把"要跳/攀才能互通"的并成同一段（实测 105090600 的第 0 段把一面
        388 像素高的悬崖当成了平台边缘），所以"我在不在 A 平台"只能靠人工圈的集合。
        没圈过集合时明说「未分组」，**不显示成空白** —— 空白会被当成程序坏了。
        """
        fid = loc.get("foothold_id")
        if not fid:
            return ""
        try:
            names = self._load_zones(self._map_id()).set_of(fid)
        except Exception:                   # noqa: BLE001
            return ""
        return "、".join(names) if names else "未分组"

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

    def _on_overlay_alpha(self, v):
        """改了叠图浓淡 ⇒ 写回**标定文件**（和「标定…」弹窗同一处）并立刻重画。

        为什么存标定而不是 live.yaml：那儿本来就是它的家（弹窗的滑块写的就是它，
        按来源分开存）—— 放两处迟早对不上，而"看着没变"是最难查的一类。
        """
        mid = self._map_id()
        if not mid:
            return
        try:
            src = self._mmap_src()
            cal = mapdata.load_calib(mid, src) or {}
            cal["alpha"] = int(v)
            mapdata.save_calib(mid, cal, src)
        except Exception:                           # noqa: BLE001
            pass
        self._refresh_overlay()

    def _on_world_offset(self, _v=None):
        """改了「坐标系偏移」⇒ 写进**这张图的标定文件**（按地图 id + 来源，2026-09-26 用户定）。

        和 `scale/offset/view/alpha` 同一份：面板与实时线程都从标定里读（`PlayerLocator`
        那边已经按来源取好了）⇒ 天然一致，不会再出现"两边各存一份、看着没变"。
        """
        mid = self._map_id()
        if not mid:
            return
        try:
            src = self._mmap_src()
            cal = mapdata.load_calib(mid, src) or {}
            cal["world_offset"] = [int(sp.value()) for sp in self._sp_off]
            mapdata.save_calib(mid, cal, src)
        except Exception:                           # noqa: BLE001
            pass

    def _stream_client(self):
        """收流那条客户端（懒建、复用）—— 来源=收流时定位就用它。

        为什么必须跟来源走：两条来源的**面板尺寸差好几倍**（A 机推流还带 zoom），
        标定也是**按来源分开存**的（`sources.stream` / `sources.live`）——
        拿 A 那份几何去量 B 那块画面，结果就是"标定过了却没坐标"（2026-09-26 踩过）。
        """
        if getattr(self, "_mmap_cli", None) is None:
            try:
                from tools.config import get
                self._mmap_cli = mm.MiniMapClient(
                    get("a_host"), port=get("minimap", "port", 5003)).start()
            except Exception:                       # noqa: BLE001
                self._mmap_cli = None
        return self._mmap_cli

    def _on_mmap_src(self, _i):
        """换来源：立刻存下来，并刷一遍状态。

        「框选小地图」**不跟着来源显隐**了 —— 它记的是面板在画面里的位置，
        两种来源都要用（收流时主画面里也有小地图，只是被压过）。

        换来源要把收流那条客户端**收掉**：留着它白占 A 机一路连接（现在是广播、
        不会再饿死别人，但没必要），下次切回来按时会重连。
        """
        src = self.cmb_mmap_src.currentData()
        update_live(mmap_src=src)
        cli = getattr(self, "_mmap_cli", None)
        if cli is not None:
            try:
                cli.stop()
            except Exception:                       # noqa: BLE001
                pass
            self._mmap_cli = None
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
        # 「坐标系偏移」从**这张图的标定**里回填（按地图 id + 来源，别把 setValue 当用户改动）
        off = (cal or {}).get("world_offset") or []
        if not off:
            # 一次性搬运：早先临时存在 live.yaml 里的那份 —— 标定里还没有就搬过来
            #（搬过来之后就以标定为准；live.yaml 那个键不再读，留着不碍事）
            off = load_live().get("mmap_world_offset") or []
            if len(off) == 2 and mid:
                try:
                    cc = dict(cal or {})
                    cc["world_offset"] = [int(off[0]), int(off[1])]
                    mapdata.save_calib(mid, cc, src_kind)
                except Exception:                   # noqa: BLE001
                    pass
        else:
            off = [off[0], off[1]]
        if len(off) == 2:
            for sp, v in zip(self._sp_off, off):
                sp.blockSignals(True)
                sp.setValue(int(v))
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
            self._world_note = self._osd_lines(osd if osd is not None else text)
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
        # **画面按「小地图来源」走**（2026-09-26 踩过）：这里原来写死"从主画面裁一块 +
        # `src=SRC_LIVE`" ⇒ 来源选**收流**时读的是**另一份标定**（`sources.live`，
        # 多半压根没存过）⇒ 表现就是"标定完了却没坐标"；而且喂的是 H.264 压过的画面，
        # 黄点更糊 —— 跟"选收流"的初衷正好相反。实时线程那份（`live_thread`）一直是对的，
        # 这里照它分一次流。
        src = self._mmap_src()
        panel = None
        if src == mm.SRC_STREAM:
            cli = self._stream_client()
            if cli is None:
                return _say("玩家世界坐标：小地图推流连不上（link.yaml 里读到 a_host 了吗）",
                            "#b06000")
            panel, _t = cli.latest()
            if panel is None:
                return _say("玩家世界坐标：还没收到小地图推流（%s）"
                            % (cli.err or "A 机那一路起了吗？"), "#b06000")
        else:
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
        r = loc.update(panel, src=src)
        if not r["ok"]:
            # 面板上写一句话，**详情进 tooltip**；画面上那行更短（见 _say 的说明）——
            # 完整诊断有一百多字，贴到画面上就是一条横穿半屏的黑带。
            return _say("玩家世界坐标：%s（鼠标放上去看详情）" % r["short"],
                        "#b06000",
                        "%s\n\n黄点那一层：%s" % (r["note"], r["dot"]),
                        osd="世界坐标：%s" % r["short"])
        # 记下"现在在哪条 foothold + 什么时候读到的"：命令前往要用它当起点，
        # 而**过期的读数不能当起点**（人会走）—— 见 _here_set。
        self._fh_seen = (r.get("foothold_id") or "", time.monotonic())
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
        # 显示的**不再是「第 N 段」**（段号是自动串出来的、语义上不可靠，见 _fh_zone），
        # 而是"脚下的 foothold 属于哪个人工圈的集合" —— 那才是寻路要用的判据。
        zone = self._fh_zone(r)
        where = "　位于fh：%s" % zone if zone else ""
        # 在绳梯上要说出来（2026-09-26 要求）：爬绳时**通常不站在任何 foothold 上**
        # （人在绳上），那时"fh：未分组"会让人以为定位坏了 —— 报「绳梯：L2」才对得上
        # 编辑器画在绳上的编号，也才对得上「爬」那条边里写的绳号。
        lad = r.get("ladder_id")
        lad_s = "　绳梯：%s" % lad if lad else ""
        # 贴到画面上的那行**必须短**（它不换行，长了横穿半屏）—— 集合名可以很长
        # （"右下休息平台"就是 6 个字），所以那里截断，面板那一行保留全名
        zone_s = zone if len(zone) <= 8 else zone[:7] + "…"
        return _say("%s%s%s%s%s" % (head, where, lad_s, held, tail),
                    color,
                    "来源：从实时画面　标定：%s%s"
                    % (mm.SRC_LABEL[mm.SRC_LIVE],
                       ("\n\n" + r["note"]) if r.get("held") else ""),
                    osd="世界 (%d, %d)%s%s%s"
                        % (round(r["world_x"]), round(r["world_y"]),
                           ("　fh：%s" % zone_s) if zone_s else "", lad_s, held))

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

    def _calib_panel_wh(self, cal=None):
        """**标定当时那块面板的尺寸** `(w, h)` —— 画叠图时要把标定几何折算到当前画面。

        为什么需要（2026-09-26 用户报「叠图被放大」）：标定记的是"那块面板像素 ↔ 底图
        像素"的换算，而叠图要画在"**现在**框出来的那一块"上 —— 两者尺寸可能差好几倍
        （典型：A 机推流带 zoom）⇒ 不折算就把整块叠图放大。用户的原话也是这个意思：
        「那个倍率只是算坐标用的」。

        取法（按可靠程度）：
          1. 标定文件里的 `panel`（新存的有；老标定没有）；
          2. **当前收流帧**的尺寸 —— 来源=收流时它正是标定那块面板（zoom=1 时 =
             真实面板大小）；
          3. 都拿不到 ⇒ None（退回老行为，不折算；状态行会提示"比面板框还大"）。
        """
        if cal is None:
            try:
                cal = mapdata.load_calib(self._map_id(), self._mmap_src()) or {}
            except Exception:                       # noqa: BLE001
                cal = {}
        p = cal.get("panel")
        if p and len(p) == 2:
            try:
                if int(p[0]) > 0 and int(p[1]) > 0:
                    return (int(p[0]), int(p[1]))
            except (TypeError, ValueError):
                pass
        if self._mmap_src() == mm.SRC_STREAM:
            cli = self._stream_client()
            if cli is not None:
                f, _t = cli.latest()
                if f is not None:
                    return (int(f.shape[1]), int(f.shape[0]))
        return None

    def _refresh_overlay(self):
        """照标定算出「叠图的哪一块画到画面的哪里」，交给实时面板去画。

        算出来的矩形是**画面坐标**：面板在画面里的位置来自 `mmap_crop`，
        面板里怎么摆来自标定（`perception/minimap.frame_overlay_rects` 一份公式，
        和标定弹窗的叠加层同源）。

        画/没画、画的哪儿、没画卡在哪 —— 都写在勾选框旁边那行上。
        """
        lp = getattr(self, "live_panel", None)
        on = self._mmap_draw_on()
        # 浓淡滑块跟着开关显隐（勾上才有意义；关着时它只会让人以为能调）
        self.lbl_alpha.setVisible(on)
        self.sld_alpha.setVisible(on)
        why = self._overlay_blocker()
        if on and not why and lp is not None:
            mid = self._map_id()
            t = mapdata.load(mid, with_canvas=True)
            crop = load_live().get("mmap_crop") or []
            if len(crop) == 4 and t is not None and t.canvas is not None:
                cal = mapdata.load_calib(mid, self._mmap_src()) or {}
                src, dst = mm.frame_overlay_rects(
                    cal, [int(v) for v in crop],
                    (t.canvas.shape[1], t.canvas.shape[0]),
                    calib_panel=self._calib_panel_wh(cal))
                pix = self._overlay_pix(mid)
                alpha = float(cal.get("alpha") or 55) / 100.0
                # `note` = 玩家世界坐标那行：贴在画面里那块框的下面（见
                # gui/live_panel._draw_note）。每次重画都要重贴 —— 帧是新的。
                lp.set_minimap_overlay(pix, src, dst, alpha,
                                       note=self._world_note)
                # ⚠ 叠图比**面板框**还大 ⇒ 标定八成不对（2026-09-26 实测踩过：
                # 标定文件里 `scale=5.63`（在 A 机 zoom=3 的收流帧上按"整图 fit"
                # 拟合出来的、匹配分才 0.70）⇒ 底下算出来 134×101×5.63 = 754×569，
                # **整屏都是它**。说在明面上，别让人对着画面猜「是不是程序画错了」。
                warn = ""
                if dst[2] > int(crop[2]) * 1.2 or dst[3] > int(crop[3]) * 1.2:
                    warn = ("　⚠ 比面板框 %dx%d 还大 —— 标定八成不对：确认 A 机推流的 "
                            "zoom 是 1、「显示方式」全局/局部选对了，再标一次"
                            % (int(crop[2]), int(crop[3])))
                self.lbl_mmap_draw.setText(
                    "已画在画面 (%d, %d) %d×%d　浓淡 %.0f%%（在「标定…」里调）%s"
                    % (dst[0], dst[1], dst[2], dst[3], alpha * 100, warn))
                self.lbl_mmap_draw.setStyleSheet(
                    "color:#b06000;" if warn else "color:#188038;")
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

        优先 `<id>_zones.png`（**地形编辑器的结果**：颜色 = 集合、名字标在平台上方、
        灰 = 还没圈进任何集合）—— 那才是"程序认得的平台"，寻路要照它走。
        没有就退到 `<id>_overlay.png`（每段一色 + 段号；**叠到实时画面上用的仍是它**），
        再退到小地图底图 `<id>.png`；都没有说明地形还没导出，把该跑的命令写出来。

        **分三段返回**（不是拼成一整句）：中间那段「几×几 像素」得**量了文件**才知道，
        而那只 QPixmap 在调用方（它本来就要为显示加载一次，不多花一次解码）。
        拼成一整句的话，"几×几"要么掉到命令行提示后面，要么得在这儿再解码一遍。
        """
        d = mapdata.map_dir()
        zi = d / ("%s_zones.png" % mid)
        if zi.exists():
            from core import zones as zones_mod
            try:
                n = len(zones_mod.load(mid).sets)
            except Exception:               # noqa: BLE001
                n = -1
            tip = ("（地形编辑器的结果：颜色 = 集合，名字标在平台上方）"
                   if n != 0 else
                   "（还没有集合 —— 点「编辑集合…」圈一个，颜色和名字才出得来）")
            return zi, ("集合图 %s" % zi.name), tip
        over = d / ("%s_overlay.png" % mid)
        if over.exists():
            return over, ("地形叠加图 %s" % over.name), (
                "（每段一色 + 段号。想要**集合版**就点下面的「生成地形图」重画一张）")
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
            # 集合图是**派生数据**（实测 38ms），缺了就顺手补上 —— 否则打开面板看到的
            # 还是旧那版叠加图，"地形图显示编辑器结果"这件事就只改了一半，
            # 还得人自己去点一次「生成地形图」。写不出来就退回叠加图（下面那条路）。
            if not (mapdata.map_dir() / ("%s_zones.png" % mid)).exists():
                self._render_zones_png(mid)
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
            # 选了「预览平台」就把它框出来（只读覆盖层，图本身不变）
            self.canvas.load(pm, self._preview_boxes(mid), editable=False, fit=True)

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

    def _on_retry_delay(self, v):
        """改了"失败后延迟激活时间" ⇒ 写进配置（它是决策参数，跟着项目存）。"""
        settings.climb_retry_delay_s = float(v)
        settings.save()

    def bind(self, project):
        """切项目：本页要跟着换的是**图**和**集合下拉**（都按地图 id 存）。

        以前这里还要回填一个"启用路线识别"的开关（还专门处理过 blockSignals 的坑）——
        那个开关和它门控的感知一起撤掉了（见 __init__ 的说明），所以只剩这几件。
        """
        self.project = project
        # 延迟激活时间也是决策参数（按项目存）⇒ 换项目重读一遍；blockSignals 别把
        # 刚读出来的值又写回去（和以前那个开关踩过的坑是同一个）。
        self.sp_retry.blockSignals(True)
        self.sp_retry.setValue(float(getattr(settings, "climb_retry_delay_s", 1.0)))
        self.sp_retry.blockSignals(False)
        self._refresh_mmap()
        self._refresh_map_image()
        self._refresh_goto()        # 换图/换项目：下拉要跟着换成这张图的集合

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


