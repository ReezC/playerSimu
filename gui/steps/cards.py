"""八个步骤卡片的具体实现。

业务函数都延迟导入（放在 make_task 里），避免拖慢启动。

实现状态：
    ② 采集    已可用（调用 tools/extract_frames）
    其余      骨架就位，待 P2/P4 接入实际逻辑
"""

import json
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QCheckBox, QDialog, QDoubleSpinBox, QHBoxLayout,
                             QLabel, QMessageBox, QPushButton, QVBoxLayout)

from core import wincap, wzexport
from gui.widgets import NoWheelComboBox

from .base import StepCard


def _latest_mtime(paths):
    """一组路径（文件或目录）的最新修改时间（秒），无则 0。

    用于「上游变更 → 下游产物过期」判断：人工修正标注后，
    数据集 / 训练 / 验证 / 迭代这些下游产物就该退回未处理状态。
    """
    t = 0.0
    for p in paths:
        p = Path(p)
        if p.is_file():
            t = max(t, p.stat().st_mtime)
        elif p.is_dir():
            for f in p.rglob("*"):
                if f.is_file():
                    t = max(t, f.stat().st_mtime)
    return t


# ══════════════════════════════════════════════════════════════
# ① 地图 / 怪种
# ══════════════════════════════════════════════════════════════
class MapCard(StepCard):
    """地图只能从下拉列表选，列表来自「WZ 导出」生成的地图清单。

    不让手输 ID：ID 一旦拼错，后面的采集、标注、训练会一路错下去而且很难发现。
    下拉列表保证选到的 ID 和怪种一定对得上。
    选完即生效（自动写回项目），所以这张卡片没有「运行」按钮。
    """

    def __init__(self):
        super().__init__(1, "map", "识别目标选项",
                         hint="选地图确定要识别的怪，选角色确定要识别的玩家。选完自动生效")
        self._pool = []       # 全量地图缓存，不受筛选影响
        self._shown = []      # 当前筛选后展示的
        self._loading = False

        self.btn_run.setVisible(False)      # 没有可运行的任务

    # ---------------- 界面 ----------------

    def build_params(self, form):
        self.field(form, "map_filter", "筛选", "str", "")
        self.field(form, "only_mob", "只看有怪", "bool", True)

        self.cmb = NoWheelComboBox()
        self.cmb.setMaxVisibleItems(20)
        self.cmb.setMinimumWidth(240)
        self.cmb.currentIndexChanged.connect(self._on_picked)
        self.widgets["map_id"] = (self.cmb, "combo")
        form.addRow("地图", self.cmb)

        row = QHBoxLayout()
        row.setSpacing(4)
        self.lbl_count = QLabel("—")
        self.lbl_count.setStyleSheet("color:#80868b;")
        row.addWidget(self.lbl_count, 1)

        btn = QPushButton("刷新列表")
        btn.setFixedWidth(76)
        btn.clicked.connect(self.reload_maps)
        row.addWidget(btn)
        form.addRow("", row)

        self.widgets["map_filter"][0].textChanged.connect(self._refill)
        self.widgets["only_mob"][0].stateChanged.connect(self._refill)

        # ---- 确认要识别的怪物 ----
        self.btn_pick_mobs = QPushButton("确认要识别怪物")
        self.btn_pick_mobs.setToolTip("打开弹窗，手动增删这张图要识别的怪物")
        self.btn_pick_mobs.clicked.connect(self._pick_mobs)
        form.addRow("怪物", self.btn_pick_mobs)

        # ---- 角色（玩家模板）选择 ----
        self.cmb_player = NoWheelComboBox()
        self.cmb_player.setMinimumWidth(240)
        self.cmb_player.currentIndexChanged.connect(self._on_player_picked)
        self.widgets["player_id"] = (self.cmb_player, "combo")
        form.addRow("角色", self.cmb_player)

        prow = QHBoxLayout()
        prow.setSpacing(4)
        self.lbl_player = QLabel("—")
        self.lbl_player.setStyleSheet("color:#80868b;")
        prow.addWidget(self.lbl_player, 1)
        btn_refresh_player = QPushButton("刷新列表")
        btn_refresh_player.setFixedWidth(76)
        btn_refresh_player.setToolTip("重新扫描角色模板目录（导出新角色后点这里）")
        btn_refresh_player.clicked.connect(self._refresh_players)
        prow.addWidget(btn_refresh_player)
        form.addRow("", prow)

    # ---------------- 列表 ----------------

    def reload_maps(self):
        """重读地图清单（工具栏导出完成后要刷一次）。"""
        self._pool = wzexport.list_maps(only_with_mob=False, keyword="")
        self._refill()

    PLAYER_ROOT = Path("datasets/sprites/player")

    def reload_players(self):
        """读角色模板目录，填角色下拉。"""
        self.cmb_player.blockSignals(True)
        self.cmb_player.clear()
        if self.PLAYER_ROOT.is_dir():
            for p in sorted(self.PLAYER_ROOT.iterdir()):
                if p.is_dir() and list(p.glob("*.png")):
                    # 必须传 userData：combo 的 value() 读 currentData，
                    # 不传的话选什么都取到 None
                    self.cmb_player.addItem(p.name, p.name)
        self.cmb_player.blockSignals(False)

    def _refresh_players(self):
        """刷新角色下拉，并尽量恢复当前选中项。"""
        cur = self.value("player_id")
        self.reload_players()
        if cur:
            idx = self.cmb_player.findData(cur)
            if idx >= 0:
                self.cmb_player.setCurrentIndex(idx)

    def _on_player_picked(self, _idx=None):
        if self._loading or self.project is None:
            return
        pid = self.value("player_id")
        if not pid:
            return
        self.project.set("player_id", pid)
        self.project.save()
        n = len(list((self.PLAYER_ROOT / pid).glob("*.png")))
        self.lbl_player.setText("模板 %d 帧" % n)

    def _refill(self):
        self._shown = wzexport.list_maps(only_with_mob=self.value("only_mob"),
                                        keyword=self.value("map_filter"))
        self._fill_combo()

    def _fill_combo(self):
        keep = self.project.get("map_id") if self.project else None

        self.cmb.blockSignals(True)
        self.cmb.clear()
        for m in self._shown:
            self.cmb.addItem(m["label"], m["id"])

        # 项目里选的地图若被筛选条件挡掉了，补一条占位让它仍能显示。
        # 否则下拉会回落到第一项，下次运行 map_id 就被悄悄改掉了。
        if keep and self.cmb.findData(keep) < 0:
            self.cmb.insertItem(0, "%s   （不在当前筛选范围内）" % keep, keep)
            self.cmb.setCurrentIndex(0)

        self.cmb.blockSignals(False)

        if not self._pool:
            self.lbl_count.setText("清单为空，请先点工具栏「WZ 导出」")
        elif not self._shown:
            # 筛不出东西不一定是「没有」——很可能是「只看有怪」把它挡住了。
            # 说清楚，否则用户会以为这张地图不存在。
            hint = "0 条"
            if self.value("only_mob") and self.value("map_filter").strip():
                alt = wzexport.list_maps(only_with_mob=False,
                                         keyword=self.value("map_filter"))
                if alt:
                    hint = ("0 条 —— 但有 %d 张匹配的地图没有怪，"
                            "取消「只看有怪」可见" % len(alt))
            self.lbl_count.setText(hint)
        else:
            self.lbl_count.setText("显示 %d / 共 %d 张"
                                   % (len(self._shown), len(self._pool)))

    def _on_picked(self, _idx=None):
        if self._loading or self.project is None:
            return

        mid = self.value("map_id")
        if not mid:
            return

        m = next((x for x in self._pool if x["id"] == mid), None)
        if m is None:
            return

        self.project.set("map_id", mid)
        self.project.set("mobs", m["mobs"])
        self.project.set("mob_names", m["mob_names"])
        self.project.save()
        self.set_result(self._describe(mid, m["mobs"], m["mob_names"]))

    def _pick_mobs(self):
        """打开「确认要识别怪物」弹窗，手动增删这张图要处理的怪。"""
        if self.project is None:
            return
        from gui.mob_picker import MobPickDialog

        dlg = MobPickDialog(self.project.get("mobs") or [], self)
        if dlg.exec_():
            new = dlg.result_mobs()
            name_map = wzexport.build_mob_name_map()
            self.project.set("mobs", new)
            self.project.set("mob_names", [name_map.get(m, "") for m in new])
            self.project.save()
            self.refresh()

    # ---------------- 项目交互 ----------------

    def load_from_project(self, p):
        self._loading = True
        try:
            if not self._pool:
                self._pool = wzexport.list_maps(only_with_mob=False, keyword="")
            self._shown = wzexport.list_maps(only_with_mob=self.value("only_mob"))
            self._fill_combo()
            self.set_value("map_id", p.get("map_id"))
            self.reload_players()
            pid = p.get("player_id")
            if not pid and self.cmb_player.count() > 0:
                # 项目里没存过玩家：默认选第一个（下拉框本已默认选中第一项，
                # 但 blockSignals 挡住了信号，没写回 project）
                pid = self.cmb_player.itemData(0)
                p.set("player_id", pid)
                p.save()
            self.set_value("player_id", pid)
        finally:
            self._loading = False

    def sync(self, p):
        mid = self.value("map_id")
        if not mid:
            return
        p.set("map_id", mid)
        m = next((x for x in self._pool if x["id"] == mid), None)
        if m is not None:
            p.set("mobs", m["mobs"])
            p.set("mob_names", m["mob_names"])

    def summarize(self, p):
        parts = []
        mid = p.get("map_id") or ""
        parts.append(self._describe(mid, p.get("mobs") or [], p.get("mob_names") or [])
                     if mid else "未选地图")
        pid = p.get("player_id") or ""
        parts.append("角色: %s" % pid if pid else "未选角色")
        return "   ·   ".join(parts)

    def detect_state(self, p):
        if not p.get("map_id"):
            return ("idle", "未选")
        if not (p.get("mobs") or []):
            return ("warn", "该图无怪")
        if not p.get("player_id"):
            return ("warn", "未选角色")
        return ("done", "已选")

    @staticmethod
    def _describe(mid, mobs, mob_names=None):
        """优先用怪物名 —— 有 680 张地图没有名字，只能靠怪种辨认。"""
        if not mobs:
            return "%s   ·   该地图无怪物记录" % mid

        names = [x for x in (mob_names or []) if x]
        if names:
            head = "/".join(names[:6])
            if len(names) > 6:
                head += " …"
        else:
            head = ", ".join(map(str, mobs[:6]))
            if len(mobs) > 6:
                head += " …"

        return "%s   ·   %d 种怪: %s" % (mid, len(mobs), head)

    def check_deps(self, p):
        if not self.value("map_id"):
            return False, "请先从下拉列表选择一张地图"
        if not (p.get("mobs") or []):
            return False, "该地图没有怪物记录，请换一张有怪的地图"
        if not self.value("player_id"):
            return False, "请选择要识别的角色（玩家模板）"
        return True, ""

    def make_task(self, p):
        return None     # 选完就已写回项目，无需任务


# ══════════════════════════════════════════════════════════════
# ② 采集
# ══════════════════════════════════════════════════════════════
class CaptureCard(StepCard):
    """采集。两种方式，差别很大：

    窗口捕获 —— 直接抓游戏窗口的客户区。画面干净，桌面/任务栏进不来，
                也不用事后设 region 排除。**推荐**。
    文件抽帧 —— 从已有的录屏抽帧。录屏往往把整个桌面录进去了，
                标注时要额外设 region，否则模型会学到任务栏和桌面图标。
    """

    def __init__(self):
        super().__init__(2, "capture", "采集",
                         hint="窗口捕获最干净；文件抽帧需要事后裁掉桌面区域")

    def build_params(self, form):
        self.field(form, "source", "来源", "choice", "window",
                   choices=["window", "file", "stream"])

        self.field(form, "file", "文件", "path", "",
                   filter="视频文件 (*.mkv *.mp4 *.avi *.mov);;所有文件 (*)")

        self.cmb_win = NoWheelComboBox()
        self.cmb_win.setMinimumWidth(230)
        self.widgets["win_rect"] = (self.cmb_win, "combo")
        form.addRow("窗口", self.cmb_win)

        row = QHBoxLayout()
        row.setSpacing(4)
        self.lbl_win = QLabel("—")
        self.lbl_win.setStyleSheet("color:#80868b;")
        row.addWidget(self.lbl_win, 1)

        self.btn_win_refresh = QPushButton("刷新")
        self.btn_win_refresh.setFixedWidth(56)
        self.btn_win_refresh.clicked.connect(self.reload_windows)
        row.addWidget(self.btn_win_refresh)

        self.btn_win_preview = QPushButton("预览")
        self.btn_win_preview.setFixedWidth(56)
        self.btn_win_preview.setToolTip("抓一帧看看选中的窗口对不对")
        self.btn_win_preview.clicked.connect(self.preview_window)
        row.addWidget(self.btn_win_preview)
        form.addRow("", row)

        # 刷新/预览行在切换来源时要整体隐藏，单独存一份引用
        self._win_extras = [self.lbl_win, self.btn_win_refresh, self.btn_win_preview]

        self.field(form, "url", "流地址", "str", "udp://0.0.0.0:5000")
        self.field(form, "fps", "抓帧频率", "float", 5.0,
                   minimum=0.5, maximum=60.0, decimals=1, step=0.5)
        self.field(form, "stride", "抽帧间隔", "int", 6, minimum=1, maximum=1000)
        self.field(form, "seconds", "时长(秒)", "float", 120.0,
                   minimum=0, maximum=36000, decimals=0, step=30)
        w = self.field(form, "dedup", "去重阈值", "float", 0.01,
                       minimum=0.0, maximum=1.0, decimals=3, step=0.005)
        w.setToolTip("相似帧去重：缩略图里亮度差超过 8 的像素占比（0~1）小于它就跳过。\n"
                     "0 = 不去重；值越大去重越激进。\n"
                     "完全静止≈0；待机动画/角色移动会明显高于它，用 0.01 能保留这些帧。")
        self.field(form, "limit", "最多张数", "int", 0, minimum=0, maximum=1000000)

        w = self.field(form, "clean", "重抽前清空", "bool", True)
        w.setToolTip("勾上：先删掉输出目录里旧的 frame_*.png 再抽。\n"
                     "不勾：只覆盖同编号的文件，新视频较短时会残留旧帧。")

        self.widgets["source"][0].currentTextChanged.connect(self._on_source)
        self._on_source("window")     # 初始状态（此时还没绑定项目）

    def _on_source(self, txt):
        """按来源切换参数：不相关的整行藏掉，界面只留当前来源要填的。"""
        is_file = (txt == "file")
        is_win = (txt == "window")
        is_stream = (txt == "stream")

        for k, on in (("file", is_file),
                      ("stride", is_file),
                      ("clean", is_file),
                      ("win_rect", is_win),
                      ("fps", is_win),
                      ("seconds", is_win or is_stream),
                      ("url", is_stream),
                      ("dedup", True),
                      ("limit", True)):
            self.set_row_visible(k, on)

        # 刷新/预览行没注册 key，单独控制
        for w in self._win_extras:
            w.setVisible(is_win)

        if is_win and self.cmb_win.count() == 0:
            self.reload_windows()

    # ---------------- 窗口列表 ----------------

    def reload_windows(self):
        ws = wincap.list_windows() if wincap.available() else []

        self.cmb_win.blockSignals(True)
        self.cmb_win.clear()
        for w in ws:
            self.cmb_win.addItem("%s   %d×%d"
                                 % (w["title"][:38], w["size"][0], w["size"][1]),
                                 w["rect"])
        self.cmb_win.blockSignals(False)

        if not wincap.available():
            self.lbl_win.setText("窗口捕获不可用（需要 pywin32）")
        elif not ws:
            self.lbl_win.setText("没有可用窗口")
        else:
            self.lbl_win.setText("共 %d 个窗口" % len(ws))

    def preview_window(self):
        rect = self.value("win_rect")
        if not rect:
            QMessageBox.warning(self, "提示", "请先选择一个窗口")
            return

        try:
            from core.wincap import grab_rect
            img = grab_rect(rect)
        except Exception as e:
            QMessageBox.critical(self, "抓取失败", "%s: %s" % (type(e).__name__, e))
            return

        h, w = img.shape[:2]
        qimg = QImage(img.data, w, h, img.strides[0],
                      QImage.Format_RGB888).rgbSwapped()
        pix = QPixmap.fromImage(qimg)

        dlg = QDialog(self)
        dlg.setWindowTitle("窗口预览  %d×%d" % (w, h))
        lay = QVBoxLayout(dlg)
        lbl = QLabel()
        lbl.setPixmap(pix.scaled(min(w, 1280), min(h, 820),
                                 Qt.KeepAspectRatio, Qt.SmoothTransformation))
        lay.addWidget(lbl)
        dlg.exec_()

    # ---------------- 项目交互 ----------------

    _KEYS = ("source", "file", "url", "seconds", "stride", "dedup", "limit",
             "fps", "clean")

    def load_from_project(self, p):
        sec = p.sec("capture")
        for k in self._KEYS:
            self.set_value(k, sec.get(k))
        self._on_source(self.value("source"))

    def sync(self, p):
        p.sec("capture").update(self.values(self._KEYS))

    def refresh(self, force=False):
        super().refresh(force)      # force 要透传，否则切项目时状态不重算
        if self.value("source") == "window" and self.cmb_win.count() == 0:
            self.reload_windows()

    def summarize(self, p):
        n = p.snapshot()["frames"]
        return "已有 %d 张画面" % n if n else "—"

    def detect_state(self, p):
        n = p.snapshot()["frames"]
        return ("done", "%d 帧" % n) if n else ("idle", "")

    def check_deps(self, p):
        src = self.value("source")

        if src == "window":
            if not wincap.available():
                return False, "窗口捕获需要 pywin32：pip install pywin32"
            if not self.value("win_rect"):
                return False, "请先选择一个窗口（点「刷新」列出当前窗口）"
            return True, ""

        if src == "file":
            if not self.value("file"):
                return False, "尚未选择录制文件"
            return True, ""

        return False, "实时流采集待实现，请用「窗口」或「文件」"

    def make_task(self, p):
        src = self.value("source")

        if src == "window":
            from core.wincap import run_capture
            return run_capture, {
                "rect": self.value("win_rect"),
                "out": str(p.frames),
                "fps": self.value("fps"),
                "seconds": self.value("seconds"),
                "dedup": self.value("dedup"),
                "limit": self.value("limit"),
            }

        if src == "file":
            from tools.extract_frames import run_extract
            return run_extract, {
                "file": self.value("file"),
                "out": str(p.frames),
                "stride": self.value("stride"),
                "dedup": self.value("dedup"),
                "limit": self.value("limit"),
                "clean": self.value("clean"),
            }

        return None


# ══════════════════════════════════════════════════════════════
# ③ 标定尺度
# ══════════════════════════════════════════════════════════════
class CalibCard(StepCard):
    """标定画面里精灵的实际大小。

    界面上**不出现「缩放比例」** —— 那是系统内部的中间量，
    取决于游戏分辨率、窗口大小、推流缩放，用户既算不出来也用不上。

    用户只需要知道三件事：
      · 当前画面多大
      · 标定过没有
      · 什么时候该重标（画面尺寸变了）
    """

    def __init__(self):
        super().__init__(3, "calib", "标定尺度",
                         hint="量出画面里精灵的实际大小。改过分辨率或窗口大小后要重标")
        self.btn_run.setVisible(False)   # 无自动标定任务，纯手动

    def build_params(self, form):
        self.lbl_res = QLabel("—")
        self.lbl_res.setStyleSheet("")
        form.addRow("画面", self.lbl_res)

        self.lbl_mob_state = QLabel("—")
        self.lbl_mob_state.setWordWrap(True)
        form.addRow("怪物状态", self.lbl_mob_state)

        self.lbl_player_state = QLabel("—")
        self.lbl_player_state.setWordWrap(True)
        form.addRow("玩家状态", self.lbl_player_state)

        # ---- 一键设置尺度：所有精灵统一基准 ----
        note = QLabel("给所有怪物/玩家统一设一个尺度基准（之后可在手动标定里逐怪微调）。")
        note.setStyleSheet("color: #80868b;")
        note.setWordWrap(True)
        form.addRow(note)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.sp_apply_all = QDoubleSpinBox()
        self.sp_apply_all.setRange(0.05, 8.0)
        self.sp_apply_all.setDecimals(3)
        self.sp_apply_all.setSingleStep(0.05)
        self.sp_apply_all.setValue(1.0)
        self.sp_apply_all.setSuffix("×")
        row.addWidget(self.sp_apply_all, 1)
        btn_apply = QPushButton("一键设置尺度")
        btn_apply.setToolTip("把所有怪物/玩家的尺度基准统一设为这个值，并清空逐怪覆盖")
        btn_apply.clicked.connect(self._apply_all_scale)
        row.addWidget(btn_apply)
        form.addRow("尺度基准", row)

        # ---- 手动目测标定 ----
        btn_manual = QPushButton("手动目测标定")
        btn_manual.setToolTip("在画面上叠加模板，手动拖动 + 调缩放对齐，目测确定尺度")
        btn_manual.clicked.connect(self._manual_calib)
        form.addRow("手动标定", btn_manual)

    def _manual_calib(self):
        """打开手动目测标定弹窗，把目测的缩放写回项目。"""
        if self.project is None:
            return
        p = self.project
        from gui.calib_manual import CalibManualDialog

        dlg = CalibManualDialog(p.get("mobs") or [], p.get("player_id") or "",
                                p.frames, p.get("mob_scales") or {},
                                p.get("scale") or 1.41,
                                p.get("player_scale") or 1.41, self)
        if dlg.exec_():
            scale = round(dlg.scale_value(), 3)
            target = dlg.result_target()
            mid = dlg.result_mob()       # 怪 id 或玩家 id
            frame = dlg.result_frame()   # 帧 stem
            w, h = self._frame_size()
            # 帧级尺度：每个帧独立，写 mob_scales["id:帧stem"]
            mob_scales = dict(p.get("mob_scales") or {})
            if mid and frame:
                mob_scales["%s:%s" % (mid, frame)] = scale
            p.set("mob_scales", mob_scales)
            if target == "player":
                p.set("player_scale_at", {"width": w, "height": h,
                                          "confidence": "manual"})
            else:
                p.set("scale_at", {"width": w, "height": h,
                                   "mob": mid, "confidence": "manual"})
            p.save()
            self.refresh()

    def _apply_all_scale(self):
        """一键设置尺度：所有怪物/玩家统一用这个尺度基准，并清空逐怪覆盖。"""
        if self.project is None:
            return
        p = self.project
        v = round(self.sp_apply_all.value(), 3)
        w, h = self._frame_size()
        p.set("scale", v)
        p.set("player_scale", v)
        p.set("mob_scales", {})
        p.set("scale_at", {"width": w, "height": h, "confidence": "manual"})
        p.set("player_scale_at", {"width": w, "height": h, "confidence": "manual"})
        p.save()
        self.refresh()

    # ---------------- 画面信息 ----------------

    def _frame_size(self):
        if self.project is None:
            return 0, 0

        f = next(iter(sorted(self.project.frames.glob("*.png"))), None)
        if f is None:
            return 0, 0

        try:
            from core.imgio import imread
            im = imread(f)
            return (im.shape[1], im.shape[0]) if im is not None else (0, 0)
        except Exception:
            return 0, 0

    def _one_state(self, scale_key, at_key):
        """单个目标（怪或人）的标定状态，返回 (文字, 颜色, 说明)。"""
        p = self.project
        if p is None:
            return "—", "#5f6368", ""

        at = p.get(at_key) or {}

        if not p.get(scale_key):
            return "未标定", "#b06000", "点「手动目测标定」实测一次"

        if not at:
            return "已标定", "#137333", ""

        aw, ah = int(at.get("width") or 0), int(at.get("height") or 0)
        cw, ch = self._frame_size()

        if cw and aw and (aw, ah) != (cw, ch):
            return "需要重标", "#b06000", \
                "上次标定时画面 %d×%d，现在是 %d×%d" % (aw, ah, cw, ch)

        conf = at.get("confidence")
        if conf == "fail":
            return "标定不可信", "#c5221f", "画面里目标太少，建议换素材重标"
        if conf == "low":
            return "存疑", "#b06000", "判别度偏低，建议增加目标数量后重标"

        return "已标定", "#137333", ""

    def _mob_state(self):
        return self._one_state("scale", "scale_at")

    def _player_state(self):
        return self._one_state("player_scale", "player_scale_at")

    def refresh(self, force=False):
        super().refresh(force)

        w, h = self._frame_size()
        self.lbl_res.setText("%d × %d" % (w, h) if w else "—（还没有画面）")

        text, color, note = self._mob_state()
        self.lbl_mob_state.setText(text + ("　" + note if note else ""))
        self.lbl_mob_state.setStyleSheet("color: %s;" % color)

        text, color, note = self._player_state()
        self.lbl_player_state.setText(text + ("　" + note if note else ""))
        self.lbl_player_state.setStyleSheet("color: %s;" % color)

        # 尺度基准输入框回填为当前全局怪物尺度 —— 否则重启后显示默认值，
        # 用户会误以为之前设的基准丢了（其实 scale 已落盘，只是输入框没同步）
        if self.project is not None:
            base = self.project.get("scale")
            self.sp_apply_all.blockSignals(True)
            self.sp_apply_all.setValue(float(base) if base else 1.41)
            self.sp_apply_all.blockSignals(False)

    # ---------------- 项目交互 ----------------

    def summarize(self, p):
        mt, _, mn = self._mob_state()
        pt, _, pn = self._player_state()
        parts = ["怪:" + mt + ("　" + mn if mn else ""),
                 "人:" + pt + ("　" + pn if pn else "")]
        return "  ·  ".join(parts)

    def detect_state(self, p):
        # 怪、人两个状态分别算，再合并成卡片总状态：
        # 任一 fail → fail；任一 warn（未标定/存疑/需重标）→ warn；都 done → done。
        mt, mc, _mn = self._mob_state()
        pt, pc, _pn = self._player_state()
        cs = {mc, pc}
        if "#c5221f" in cs:
            st = "fail"
        elif "#b06000" in cs:
            st = "warn"
        else:
            st = "done"
        return (st, "怪:%s 人:%s" % (mt, pt))


# ══════════════════════════════════════════════════════════════
# ④ 自动标注
# ══════════════════════════════════════════════════════════════
class LabelCard(StepCard):
    """用 WZ 精灵做模板匹配，自动生成 YOLO 标注。

    准优先于全：阈值故意设得高，宁可漏标也不标错 ——
    标错会污染训练集（YOLO 会忠实学会错误的框），
    漏标只是少几个样本，靠训练后的泛化能力补回来。
    """

    _BTN = ("QPushButton { background:#1a73e8; color:#ffffff; border:none;"
            " border-radius:5px; font-weight:600; padding:6px 14px; }"
            "QPushButton:hover { background:#4285f4; }"
            "QPushButton:disabled { background:#dadce0; color:#ffffff; }")

    def __init__(self):
        super().__init__(4, "label", "自动标注",
                         hint="用 WZ 精灵做模板匹配。阈值越高越准，召回越低")
        self._label_target = "mob"   # mob / player
        self._yolo_player = {}       # 权重路径 -> 所属项目玩家角色

        # 通用「运行」按钮换成两个主标注按钮。
        self.btn_run.hide()
        foot = self.layout().itemAt(self.layout().count() - 1).layout()
        self.btn_label_mob = QPushButton("标注怪物")
        self.btn_label_mob.setStyleSheet(self._BTN)
        self.btn_label_mob.clicked.connect(lambda: self._emit_run("mob"))
        self.btn_label_player = QPushButton("标注玩家")
        self.btn_label_player.setStyleSheet(self._BTN)
        self.btn_label_player.clicked.connect(lambda: self._emit_run("player"))
        foot.addWidget(self.btn_label_mob)
        foot.addWidget(self.btn_label_player)

        # ---- YOLO 辅助开关 + 权重（在标注按钮下方）----
        aux = QHBoxLayout()
        aux.setSpacing(6)
        self.ck_yolo_mob = QCheckBox("YOLO 辅助怪物")
        self.ck_yolo_mob.setToolTip("勾选后，「标注怪物」会额外用其它项目模型补怪物框")
        self.ck_yolo_mob.toggled.connect(self._on_yolo_toggle)
        aux.addWidget(self.ck_yolo_mob)
        self.ck_yolo_player = QCheckBox("YOLO 辅助玩家")
        self.ck_yolo_player.setToolTip("勾选后，「标注玩家」会额外用角色一致的模型补玩家框")
        self.ck_yolo_player.toggled.connect(self._on_yolo_toggle)
        aux.addWidget(self.ck_yolo_player)
        self.lbl_yolo_weight = QLabel("辅助权重")
        aux.addWidget(self.lbl_yolo_weight)
        self.cmb_yolo = NoWheelComboBox()
        self.cmb_yolo.setMaxVisibleItems(12)
        aux.addWidget(self.cmb_yolo, 1)
        self.layout().addLayout(aux)
        self.lbl_yolo_weight.setVisible(False)
        self.cmb_yolo.setVisible(False)

    def _on_yolo_toggle(self):
        show = self.ck_yolo_mob.isChecked() or self.ck_yolo_player.isChecked()
        self.lbl_yolo_weight.setVisible(show)
        self.cmb_yolo.setVisible(show)

    def _emit_run(self, target):
        self._label_target = target
        if not self._confirm_relabel(target):
            return
        self.run_clicked.emit(self)

    def _confirm_relabel(self, target):
        """标过的情况下弹二次确认。重标只覆盖自动框，人工修正会保留。"""
        p = self.project
        if p is None:
            return True

        from gui import labelio
        prog = labelio.label_progress(p)
        key = "mob" if target == "mob" else "player"
        if prog.get(key, 0) == 0:
            return True   # 这类还没标过，直接跑

        name = "怪物" if target == "mob" else "玩家"
        r = QMessageBox.question(
            self, "确认重新标注",
            "已标过 %s（%d 帧）。\n\n"
            "重新标注会重新生成自动标注框，\n"
            "人工修正的框会保留（不会被覆盖）。\n\n"
            "确定要重标吗？" % (name, prog[key]),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return r == QMessageBox.Yes

    def set_busy(self, busy):
        super().set_busy(busy)
        self.btn_label_mob.setEnabled(not busy)
        self.btn_label_player.setEnabled(not busy)

    def build_params(self, form):
        self.field(form, "thresh", "匹配阈值", "float", 0.90,
                   minimum=0.0, maximum=1.0, decimals=2, step=0.01)
        self.field(form, "min_distinct", "区分度", "float", 0.06,
                   minimum=0.0, maximum=1.0, decimals=3, step=0.01)
        self.field(form, "per_mob", "最大模板帧", "int", 20, minimum=2, maximum=60)
        self.widgets["per_mob"][0].setToolTip(
            "每只怪的模板帧**上限**（非死亡动画的帧都会用上）。\n"
            "  · 非死亡帧数 ≤ 这个数 → 全用，不截断（绝大多数怪都属此类）\n"
            "  · 超过时才按动作轮询分配，保证每个动作都有代表帧\n\n"
            "死亡动画（die）始终排除 —— 怪临死时的样子和活着时差异太大，\n"
            "拿它当模板会把「正在消失的怪」也框出来。\n\n"
            "注意别调太小：设成 6 时曾把绿水灵 stand 的第三个相位挤掉，\n"
            "结果那个相位的怪从未被标注，训练集里一个样本都没有。")
        self.field(form, "max_peaks", "每模板峰数", "int", 4, minimum=1, maximum=20)
        self.field(form, "downscale", "降采样", "int", 1, minimum=1, maximum=4)

        # ---- 玩家标注参数（模板匹配，class 0）----
        # 玩家 scale 由 ③ 标定尺度自动测，这里只读展示，不让手填
        self.lbl_player_scale = QLabel("—（先跑 ③ 标定）")
        self.lbl_player_scale.setStyleSheet("color:#5f6368;")
        form.addRow("玩家 scale", self.lbl_player_scale)

        self.field(form, "player_thresh", "玩家阈值", "float", 0.78,
                   minimum=0.0, maximum=1.0, decimals=2, step=0.01)

    def load_from_project(self, p):
        sec = p.sec("label")
        for k in ("thresh", "min_distinct", "per_mob", "max_peaks", "downscale",
                  "player_thresh"):
            self.set_value(k, sec.get(k))

    def sync(self, p):
        p.sec("label").update(
            self.values(["thresh", "min_distinct", "per_mob", "max_peaks", "downscale",
                         "player_thresh"]))

    def refresh(self, force=False):
        super().refresh(force)
        if self.project is not None:
            ps = self.project.get("player_scale")
            self.lbl_player_scale.setText(
                ("%.3f" % ps) if ps else "—（先跑 ③ 标定）")
        self._fill_yolo_weights()

    def _fill_yolo_weights(self):
        """填充 YOLO 辅助的权重下拉（列出所有项目训练好的模型）。"""
        from tools.yolo_augment import list_weights
        cur = self.cmb_yolo.currentData()
        self._yolo_player = {}
        self.cmb_yolo.blockSignals(True)
        self.cmb_yolo.clear()
        wlist = list_weights()
        if not wlist:
            self.cmb_yolo.addItem("（还没有训练好的模型）", None)
        else:
            for label, path, pid in wlist:
                self._yolo_player[path] = pid
                tail = (" · 玩家:%s" % pid) if pid else ""
                self.cmb_yolo.addItem(label + tail, path)
        if cur is not None:
            i = self.cmb_yolo.findData(cur)
            if i >= 0:
                self.cmb_yolo.setCurrentIndex(i)
        self.cmb_yolo.blockSignals(False)

    # ---------------- 摘要 ----------------

    @staticmethod
    def _read_stats(p):
        f = p.dir_of("labels_auto") / "stats.json"
        if not f.exists():
            return None
        try:
            with open(f, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None

    def summarize(self, p):
        from gui import labelio
        prog = labelio.label_progress(p)
        total = prog["total"]
        if not total:
            return "尚未标注"
        return ("已标怪物 %d 帧（%.0f%%）· 已标角色 %d 帧（%.0f%%）"
                % (prog["mob"], 100.0 * prog["mob"] / total,
                   prog["player"], 100.0 * prog["player"] / total))

    def detect_state(self, p):
        from gui import labelio
        prog = labelio.label_progress(p)
        total = prog["total"]
        if not total:
            return ("idle", "")

        mob_pct = 100.0 * prog["mob"] / total
        player_pct = 100.0 * prog["player"] / total

        if prog["mob"] == 0 and prog["player"] == 0:
            return ("idle", "")

        # 任何一类没标满 100% → 都不是完成态，标黄并给出各自进度
        if prog["mob"] < total or prog["player"] < total:
            return ("warn", "未标满：怪物 %.0f%% · 角色 %.0f%%"
                    % (mob_pct, player_pct))

        return ("done", "怪物 100% · 角色 100%")

    def check_deps(self, p):
        if p.snapshot()["frames"] == 0:
            return False, "还没有画面，请先完成 ② 采集"
        if self._label_target == "player":
            if not (p.get("player_id") or "").strip():
                return False, "还没选择角色，请先完成 ① 地图"
            if not p.get("player_scale"):
                return False, "玩家还没标定尺度，请先完成 ③ 标定尺度"
            if self.ck_yolo_player.isChecked():
                w = self.cmb_yolo.currentData()
                if not w:
                    return False, "还没有训练好的模型 —— 先在其它项目完成 ⑦ 训练"
                wpid = self._yolo_player.get(w, "")
                pid = (p.get("player_id") or "").strip()
                if wpid != pid:
                    return False, ("模型角色「%s」≠ 当前角色「%s」，辅助玩家会误标，"
                                   "请选一个角色一致的模型"
                                   % (wpid or "（无）", pid or "（无）"))
            return True, ""
        if not (p.get("mobs") or []):
            return False, "还没确定怪种，请先完成 ① 地图"
        if not wzexport.sprite_dir_path().is_dir():
            return False, ("精灵库不存在：\n%s\n\n"
                           "需要用 WzProbe dump-mob 导出，"
                           "或修改 config/wz.yaml 的 sprite_dir"
                           % wzexport.sprite_dir_path())
        if self.ck_yolo_mob.isChecked() and not self.cmb_yolo.currentData():
            return False, "还没有训练好的模型 —— 先在其它项目完成 ⑦ 训练"
        return True, ""

    def make_task(self, p):
        sec = p.sec("label")

        if self._label_target == "player":
            from tools.yolo_augment import run_detect_player_augmented
            params = {
                "detect": {
                    "frames": str(p.frames),
                    "out": str(p.dir_of("labels_auto")),
                    "player_id": p.get("player_id") or "",
                    "scale": p.get("player_scale", 1.0),
                    "frame_scales": p.get("mob_scales") or {},
                    "thresh": sec.get("player_thresh", 0.78),
                    "vis": True,
                    "vis_dir": str(p.vis) + "_player",
                },
            }
            if self.ck_yolo_player.isChecked():
                w = self.cmb_yolo.currentData()
                if not w:
                    raise ValueError("还没有训练好的模型，先在其它项目完成 ⑦ 训练")
                params["augment"] = {
                    "weights": w,
                    "frames": str(p.frames),
                    "out": str(p.dir_of("labels_auto")),
                    "mode": "player",
                    "player_id": p.get("player_id") or "",
                    "weights_player_id": self._yolo_player.get(w, ""),
                    "conf": 0.40,
                    "imgsz": p.sec("train").get("imgsz", 960),
                    "device": p.sec("train").get("device", "0"),
                }
            return run_detect_player_augmented, params

        from tools.yolo_augment import run_detect_mob_augmented
        params = {
            "detect": {
                "mobs": p.get("mobs") or [],
                "sprites": str(wzexport.sprite_dir_path()),
                "frames": str(p.frames),
                "out": str(p.dir_of("labels_auto")),
                "vis_dir": str(p.vis),
                "scale": p.get("scale") or 1.12,
                "mob_scales": p.get("mob_scales") or {},
                "downscale": sec.get("downscale", 1),
                "thresh": sec.get("thresh", 0.90),
                "min_distinct": sec.get("min_distinct", 0.06),
                "per_mob": sec.get("per_mob", 20),
                "max_peaks": sec.get("max_peaks", 4),
                "vis": True,
            },
        }
        if self.ck_yolo_mob.isChecked():
            w = self.cmb_yolo.currentData()
            if not w:
                raise ValueError("还没有训练好的模型，先在其它项目完成 ⑦ 训练")
            params["augment"] = {
                "weights": w,
                "frames": str(p.frames),
                "out": str(p.dir_of("labels_auto")),
                "mode": "mob",
                "conf": 0.40,
                "imgsz": p.sec("train").get("imgsz", 960),
                "device": p.sec("train").get("device", "0"),
            }
        return run_detect_mob_augmented, params


# ══════════════════════════════════════════════════════════════
# ⑤ 标注校验（编辑器 / 质检台 / 金标准）
# ══════════════════════════════════════════════════════════════
class EditorCard(StepCard):
    """⑤ 标注校验 —— 质检台的主入口。

    这一步**不放「运行」按钮**：校验是人工动作，没有可后台跑的任务。
    挂个点了没反应的按钮，只会让人以为程序坏了。
    """

    def __init__(self):
        super().__init__(5, "editor", "标注校验",
                         hint="翻看框准不准、修正错框、剔除坏帧 —— 这一步决定模型的上限")

        self.btn_run.setVisible(False)          # 人工环节，无可运行任务
        self.btn_view.setText("打开质检台")      # 「查看」在这里的实际含义

    def build_params(self, form):
        note = QLabel(
            "质检台能做：\n"
            "  · 翻看带框原图（滚轮缩放、中键平移）\n"
            "  · 筛异常帧：空帧 / 多框 / 新框出现 / 旧框消失\n"
            "  · 空白处拖拽建框、Del 删框、剔除整帧\n"
            "  · 修正写在 labels/，自动标注的 labels_auto/ 永不改动")
        note.setStyleSheet("color: #80868b;")
        note.setWordWrap(True)
        form.addRow(note)

    def summarize(self, p):
        snap = p.snapshot()
        auto, manual = snap["labels_auto"], snap["labels"]
        if not auto and not manual:
            return "还没有标注"
        if manual:
            return ("自动 %d 帧 · 人工修正 %d 帧（%.0f%%）"
                    % (auto, manual, 100.0 * manual / max(1, auto)))
        return "自动 %d 帧 · 尚未人工修正" % auto

    def detect_state(self, p):
        snap = p.snapshot()
        auto, manual = snap["labels_auto"], snap["labels"]
        if not auto and not manual:
            return ("idle", "")
        if not manual:
            # 有标注但一帧没看过：不算完成 —— 这一步的价值就在于"有人看过"
            return ("warn", "未校验")
        return ("done", "已校 %d 帧" % manual)

    def refresh(self, force=False):
        # 基类刷新时会收起 btn_view，所以这里按「有没有标注」再决定露不露出来。
        # 没标注时点开只有空面板，不如不显示。
        super().refresh(force)
        snap = self.project.snapshot() if self.project else {}
        self.btn_view.setVisible(bool(snap.get("labels_auto") or snap.get("labels")))

    def check_deps(self, p):
        if p.snapshot()["labels_auto"] == 0 and p.snapshot()["labels"] == 0:
            return False, "还没有标注，请先完成 ④ 自动标注"
        return True, ""

    def make_task(self, p):
        return None     # 人工环节，没有后台任务


# ══════════════════════════════════════════════════════════════
# ⑥ 数据集整理
# ══════════════════════════════════════════════════════════════
class DatasetCard(StepCard):
    def __init__(self):
        super().__init__(6, "dataset", "数据集",
                         hint="把帧和标注整理成 YOLO 目录结构并划分 train/val")

    def build_params(self, form):
        self.field(form, "val_ratio", "验证集比例", "float", 0.2,
                   minimum=0.05, maximum=0.5, decimals=2, step=0.05)
        self.field(form, "min_boxes", "最少框数", "int", 1, minimum=0, maximum=100)
        self.widgets["min_boxes"][0].setToolTip(
            "少于这个框数的帧直接丢弃。\n"
            "默认 1：空帧不算负样本 —— 因为空帧很可能是漏检，\n"
            "留着等于教模型「这里没有怪」。\n"
            "改成 0 则保留空帧当负样本。")

    def load_from_project(self, p):
        sec = p.sec("dataset")
        for k in ("val_ratio", "min_boxes"):
            self.set_value(k, sec.get(k))

    def sync(self, p):
        p.sec("dataset").update(self.values(["val_ratio", "min_boxes"]))

    def summarize(self, p):
        y = p.dataset / "data.yaml"
        if not y.exists():
            return "未生成"
        tr = len(list((p.dataset / "images" / "train").glob("*.jpg")))
        va = len(list((p.dataset / "images" / "val").glob("*.jpg")))
        return "train %d / val %d" % (tr, va)

    def detect_state(self, p):
        y = p.dataset / "data.yaml"
        if not y.exists():
            return ("idle", "")
        # 人工修正标注后（labels 比数据集新），数据集过期，退回未处理
        up = [p.dir_of("labels"), p.dir_of("labels_iter"), p.dir_of("labels_auto")]
        if _latest_mtime(up) > y.stat().st_mtime:
            return ("warn", "标注已更新")
        tr = sum(1 for _ in (p.dataset / "images" / "train").glob("*.jpg"))
        if not tr:
            return ("fail", "训练集为空")
        return ("done", "train %d" % tr)

    def check_deps(self, p):
        snap = p.snapshot()
        if snap["labels_auto"] == 0 and snap["labels"] == 0:
            return False, "还没有标注，请先完成 ④ 自动标注"
        return True, ""

    def make_task(self, p):
        from perception.prepare_dataset import run_dataset

        sec = p.sec("dataset")
        return run_dataset, {
            "frames": str(p.frames),
            # 顺序即优先级，从可信到不可信：
            #   1. labels/      人工修正过的，最可信
            #   2. labels_iter/ ⑨ 迭代产物（跑过才有，目录可能是空的）
            #   3. labels_auto/ 原始自动标注，兜底
            # 顺序不能乱 —— 人工的结果被自动结果盖掉，等于白改。
            "labels": [str(p.dir_of("labels")),
                       str(p.dir_of("labels_iter")),
                       str(p.dir_of("labels_auto"))],
            "out": str(p.dataset),
            "val_ratio": sec.get("val_ratio", 0.2),
            "min_boxes": sec.get("min_boxes", 1),
            "quality": sec.get("quality", 92),
        }


# ══════════════════════════════════════════════════════════════
# ⑦ 训练
# ══════════════════════════════════════════════════════════════
class TrainCard(StepCard):
    def __init__(self):
        super().__init__(7, "train", "训练", hint="训练 YOLO 检测器，每轮指标实时刷新")

    def build_params(self, form):
        self.field(form, "model", "基础权重", "str", "yolo26n.pt")
        self.widgets["model"][0].setToolTip(
            "预训练模型，决定速度和精度。\n"
            "n 最小最快（数据少/显存小选它），s/m/l/x 越来越准也越来越慢。\n"
            "本地没有会自动联网下载。")

        self.field(form, "epochs", "轮数", "int", 120, minimum=1, maximum=5000)
        self.widgets["epochs"][0].setToolTip(
            "训练多少轮。数据几百张时 100~200 轮通常够。\n"
            "设大了没关系 —— 连续 40 轮不涨会自动早停。")

        self.field(form, "imgsz", "输入尺寸", "int", 960, minimum=320, maximum=2048)
        self.widgets["imgsz"][0].setToolTip(
            "训练时把画面缩放到多大再喂网络。\n"
            "目标越小、画面越精细，尺寸要越大（960 适合 100px 级目标）。\n"
            "越大越吃显存。")

        self.field(form, "batch", "批大小", "int", 8, minimum=1, maximum=128)
        self.widgets["batch"][0].setToolTip(
            "每步喂几张图。显存越大能开越大，训练越稳。\n"
            "报显存不足（OOM）就调小，比如 4 或 2。")

        self.field(form, "device", "设备", "str", "0")
        self.widgets["device"][0].setToolTip(
            "0 = 第一块 GPU，cpu = 用 CPU（会很慢）。\n"
            "多卡时写 0,1 或 0,1,2。")

    def load_from_project(self, p):
        sec = p.sec("train")
        for k in ("model", "epochs", "imgsz", "batch", "device"):
            self.set_value(k, sec.get(k))

    def sync(self, p):
        p.sec("train").update(
            self.values(["model", "epochs", "imgsz", "batch", "device"]))

    def summarize(self, p):
        best = list(p.dir_of("runs").glob("**/weights/best.pt"))
        if not best:
            return "未训练"
        model = list(p.dir_of("models").glob("*.pt"))
        extra = "   ·   已归档 %s" % model[0].name if model else ""
        return "best.pt %s%s" % (best[0].parent.parent.name, extra)

    def detect_state(self, p):
        best = sorted(p.dir_of("runs").glob("**/weights/best.pt"))
        if not best:
            return ("idle", "")
        # 数据集比模型新 → 模型过期，退回未处理
        y = p.dataset / "data.yaml"
        if y.exists() and y.stat().st_mtime > best[-1].stat().st_mtime:
            return ("warn", "数据集已更新")
        return ("done", "已训练")

    def check_deps(self, p):
        if not (p.dataset / "data.yaml").exists():
            return False, "缺少数据集，请先完成 ⑥ 数据集"
        return True, ""

    def make_task(self, p):
        from perception.train import run_train

        sec = p.sec("train")
        return run_train, {
            "data": str(p.dataset / "data.yaml"),
            "model": sec.get("model", "yolo26n.pt"),
            "epochs": sec.get("epochs", 120),
            "imgsz": sec.get("imgsz", 960),
            "batch": sec.get("batch", 8),
            "device": sec.get("device", "0"),
            "patience": sec.get("patience", 40),
            # 输出落在项目里，不是全局 runs/ —— 项目要能整个拷走
            "project_dir": str(p.dir_of("runs")),
            "name": "detect_v1",
            "copy_to": str(p.dir_of("models")),
        }


# ══════════════════════════════════════════════════════════════
# ⑧ 验证
# ══════════════════════════════════════════════════════════════
class VerifyCard(StepCard):
    """⑧ 验证 —— 拿训练好的模型跑一批画面，肉眼确认漏检和误检。

    **和 ⑦ 的区别（这是这张卡片存在的理由）**
        ⑦ 报的 mAP 是在 val 集上算的，而那批图和训练图同源：
        同一张地图、同一批怪、同一个视角、同样的光照。
        这种"同分布"下的高分天然偏乐观。

        所以这里允许把画面目录指到**任意位置** —— 包括新录的视频。
        只有跳出训练分布，才看得出模型是真会认怪，
        还是只记住了这个场景的背景纹理。
    """

    def __init__(self):
        super().__init__(8, "verify", "验证",
                         hint="用训练好的模型跑画面，看漏检和误检")
        self.btn_view.setText("查看结果")

    def build_params(self, form):
        self.field(form, "source", "画面目录", "path", "", mode="dir")
        self.widgets["source"][0].setToolTip(
            "验证要指向**训练分布之外**的画面（新录的视频、MapleNecrocer 截图等）。\n"
            "所以这里**不设默认值** —— 必须手动选一个目录，\n"
            "避免误跑成训练帧（那样等于把 val 指标再算一遍，没意义）。")

        self.field(form, "conf", "置信度阈值", "float", 0.30,
                   minimum=0.0, maximum=1.0, decimals=2, step=0.05)
        self.field(form, "imgsz", "推理尺寸", "int", 960,
                   minimum=320, maximum=2048)
        self.field(form, "limit", "限制张数", "int", 0, minimum=0, maximum=100000)
        self.widgets["limit"][0].setToolTip("只跑前 N 张，0 = 全部。先跑几十张看效果更快。")

    # ---------------- 项目交互 ----------------

    _KEYS = ("source", "conf", "imgsz", "limit")

    def load_from_project(self, p):
        sec = p.sec("verify")
        for k in ("conf", "imgsz", "limit"):
            self.set_value(k, sec.get(k))
        # 不默认 frames：验证的意义就是"跳出训练分布"，
        # 默认指向训练帧等于鼓励用户白验一次。
        self.set_value("source", sec.get("source") or "")

    def sync(self, p):
        p.sec("verify").update(self.values(self._KEYS))

    def refresh(self, force=False):
        super().refresh(force)
        # 有结果图才显示「查看结果」，否则点开是空面板
        has = bool(self.project) and bool(list(self.project.dir_of("verify").glob("*.jpg")))
        self.btn_view.setVisible(has)

    @staticmethod
    def _weights(p):
        """优先用归档到 models/ 的，没有再回退 runs/ 里的 best.pt。"""
        got = sorted(p.dir_of("models").glob("*.pt"))
        if got:
            return str(got[-1])
        best = sorted(p.dir_of("runs").glob("**/weights/best.pt"))
        return str(best[-1]) if best else ""

    @staticmethod
    def _read_stats(p):
        f = p.dir_of("verify") / "stats.json"
        if not f.exists():
            return None
        try:
            with open(f, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None

    def summarize(self, p):
        st = self._read_stats(p)
        if not st:
            return "未验证"
        n = st.get("frames", 0) or 1
        return ("%d 帧 / %d 框 · 有检出 %.0f%% · 空帧 %d"
                % (st.get("frames", 0), st.get("boxes", 0),
                   100.0 * st.get("frames_with", 0) / n,
                   st.get("zero_frames", 0)))

    def detect_state(self, p):
        n = sum(1 for _ in p.dir_of("verify").glob("*.jpg"))
        if not n:
            return ("idle", "")
        # 模型比验证结果新 → 验证过期
        w = self._weights(p)
        if w and Path(w).stat().st_mtime > _latest_mtime([p.dir_of("verify")]):
            return ("warn", "模型已更新")
        return ("done", "%d 张" % n)

    def check_deps(self, p):
        if not self._weights(p):
            return False, "还没有模型，请先完成 ⑦ 训练"

        src = self.value("source")
        if not src:
            return False, "请先选择要验证的画面目录（建议指向新录的画面或 MapleNecrocer 截图）"
        src = Path(src)
        if not src.is_dir():
            return False, "画面目录不存在：\n%s" % src
        if not (list(src.glob("*.png")) or list(src.glob("*.jpg"))):
            return False, "画面目录里没有 png/jpg：\n%s" % src
        return True, ""

    def make_task(self, p):
        from perception.predict import run_predict

        w = self._weights(p)
        if not w:
            raise ValueError("找不到权重，请先完成 ⑦ 训练")

        return run_predict, {
            "weights": w,
            "source": self.value("source"),
            "out": str(p.dir_of("verify")),
            "conf": self.value("conf"),
            "imgsz": self.value("imgsz"),
            "device": p.sec("train").get("device", "0"),
            "limit": self.value("limit"),
        }


# ══════════════════════════════════════════════════════════════
# ⑨ 迭代自训练
# ══════════════════════════════════════════════════════════════
class IterateCard(StepCard):
    """用当前模型的预测重写标注，再训一轮。

    **为什么值得多做一轮**
        模板匹配的召回被"姿态必须和模板一致"卡住，本项目实测 4.07 框/帧；
        训出来的 YOLO 学的是外观特征，同一批画面能给 6 框/帧 —— 多出来的
        正是模板匹配漏掉的目标。把它们写回标注，下一轮模型就会更强。

    **为什么必须保护 labels/**
        伪标注的错误会自我强化：模型漏掉的目标在新标注里同样消失，
        下一轮漏得更多。人工修正过的帧是唯一确定的真值 —— 整帧跳过，
        一个字节都不动。

    **为什么输出另存 labels_iter/**
        不覆盖 labels_auto/，才能随时回退，也才能把两套标注各训一个模型
        做对比。否则"迭代到底有没有用"就只能靠感觉。
    """

    def __init__(self):
        super().__init__(9, "iterate", "迭代自训练",
                         hint="用模型预测重写标注，再训一轮。可选，但通常能再涨一截")

    def build_params(self, form):
        self.field(form, "conf", "置信度阈值", "float", 0.60,
                   minimum=0.0, maximum=1.0, decimals=2, step=0.05)
        self.widgets["conf"][0].setToolTip(
            "只有模型置信度高于这个值的框才会被写进新标注。\n"
            "调低 → 标注更全，但模型的误检也会一起写进去；\n"
            "调高 → 更保守，改动更小。")

        self.field(form, "keep_empty", "无检出时保留原标注", "bool", True)
        self.widgets["keep_empty"][0].setToolTip(
            "模型在某帧什么都没检出时，是否沿用原标注。\n"
            "建议勾选：「模型没检出」更可能是模型的问题，"
            "不是画面里真没怪。")

        self.field(form, "limit", "限制张数", "int", 0, minimum=0, maximum=100000)

    _KEYS = ("conf", "keep_empty", "limit")

    def load_from_project(self, p):
        sec = p.sec("iterate")
        for k in self._KEYS:
            self.set_value(k, sec.get(k))

    def sync(self, p):
        p.sec("iterate").update(self.values(self._KEYS))

    @staticmethod
    def _weights(p):
        got = sorted(p.dir_of("models").glob("*.pt"))
        if got:
            return str(got[-1])
        best = sorted(p.dir_of("runs").glob("**/weights/best.pt"))
        return str(best[-1]) if best else ""

    @staticmethod
    def _read_stats(p):
        f = p.dir_of("labels_iter") / "stats.json"
        if not f.exists():
            return None
        try:
            with open(f, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None

    def summarize(self, p):
        st = self._read_stats(p)
        if not st:
            n = sum(1 for _ in p.dir_of("labels_iter").glob("*.txt"))
            return "尚未迭代" if not n else "%d 帧（无统计）" % n
        return ("重写 %d 帧 / %d 框 · 保护 %d 帧"
                % (st.get("from_model", 0), st.get("boxes", 0),
                   st.get("protected", 0)))

    def detect_state(self, p):
        n = sum(1 for _ in p.dir_of("labels_iter").glob("*.txt"))
        if not n:
            return ("idle", "")
        # 模型比迭代产物新 → 迭代过期
        w = self._weights(p)
        if w and Path(w).stat().st_mtime > _latest_mtime([p.dir_of("labels_iter")]):
            return ("warn", "模型已更新")
        return ("done", "%d 帧" % n)

    def check_deps(self, p):
        if not self._weights(p):
            return False, "还没有模型，请先完成 ⑦ 训练"
        if p.snapshot()["frames"] == 0:
            return False, "还没有画面，请先完成 ② 采集"
        return True, ""

    def make_task(self, p):
        from perception.relabel import run_relabel

        w = self._weights(p)
        if not w:
            raise ValueError("找不到权重，请先完成 ⑦ 训练")

        sec = p.sec("iterate")
        return run_relabel, {
            "weights": w,
            "frames": str(p.frames),
            "out": str(p.dir_of("labels_iter")),
            "src_labels": str(p.dir_of("labels_auto")),
            # 人工修正过的帧整帧跳过 —— 这是唯一确定的真值，不能被模型盖掉
            "protect": [str(p.dir_of("labels"))],
            "conf": sec.get("conf", 0.60),
            "imgsz": p.sec("train").get("imgsz", 960),
            "device": p.sec("train").get("device", "0"),
            "limit": sec.get("limit", 0),
            "keep_empty": sec.get("keep_empty", True),
        }


ALL_CARDS = [MapCard, CaptureCard, CalibCard, LabelCard,
             EditorCard, DatasetCard, TrainCard, VerifyCard,
             IterateCard]
