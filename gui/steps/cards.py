"""八个步骤卡片的具体实现。

业务函数都延迟导入（放在 make_task 里），避免拖慢启动。

实现状态：
    ② 采集    已可用（调用 tools/extract_frames）
    其余      骨架就位，待 P2/P4 接入实际逻辑
"""

import json
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QCheckBox, QDialog, QDoubleSpinBox, QFileDialog,
                             QFormLayout, QHBoxLayout, QLabel, QMessageBox,
                             QPushButton, QVBoxLayout, QWidget)

from core import petlib, wincap, wzexport
from gui import theme
from gui.icon_grid import IconGrid      # ⭐ 背包窗格 + 悬停信息窗（掉落物那块用 ✓）
from gui.widgets import (NoWheelComboBox, NoWheelDoubleSpinBox,
                         field_tip, title_label)   # 提示挂标题 / 标题可复制 ✓

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

    #: 这张卡把「项目的地图 id」改掉了（用户 2026-10-04 ✓）—— 名字与
    #: `gui/route_panel.py::map_changed` **故意一致** ✓（主窗口那边用 `hasattr(card, "map_changed")`
    #: 统一接 ✓ 两个写口一条口径 ✓）。
    map_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__(1, "map", "识别目标选项",
                         hint="选一张要识别的地图。选完自动生效 —— "
                              "要识别哪些怪、用哪个角色，都在卡片 4「自动标注」里调")
        self._pool = []       # 全量地图缓存，不受筛选影响
        self._shown = []      # 当前筛选后展示的
        self._loading = False

        self.btn_run.setVisible(False)      # 没有可运行的任务
        # ⭐⭐ **卡片最下面那行灰字整个移除**（用户 2026-10-05 ✓ 原话："卡片1最下面的灰字移除"，
        #   并明确指的是**选中地图后显示的那行** ——
        #   「{地图id}·{n}种怪：[{怪名}] · 角色：{角色名}」✓）
        #   ⇒ 那行就是 `summarize` → `set_result`（`base.py:338`）✓；怪种与角色都已经搬去
        #     卡片 4 了 ✓ ⇒ 这行再显示就是**两处口径** ✗。
        #   ⚠ 这里**连标签一起藏掉** ✗ —— 只把文案写成空的话，基类会显示 "—"
        #     （`base.set_result` 的 `text or "—"` ✓）那不算"移除" ✗。
        self.result.setVisible(False)

    # ---------------- 界面 ----------------

    def build_params(self, form):
        self.field(form, "map_filter", "筛选", "str", "",
                   tip="在地图下拉里筛选要显示哪张地图。\n  · 匹配范围含 ID / 地图名 / 地区（记不全时逐字兜底）；\n  · 留空 = 不过滤，列出全部；\n  · 只影响下拉里显示哪些图，不会替你选中某一张。"
        )
        self.field(form, "only_mob", "只看有怪", "bool", True,
                   tip="勾上：地图下拉里只列「有怪物记录」的地图（默认勾选）。\n  · 取消：连没有怪的地图也列出来（这类图选进去没怪可识别）；\n  · 筛空了下方会提示「取消『只看有怪』可见」；\n  · 只影响下拉范围，不改动已经选中的地图。"
        )

        self.cmb = NoWheelComboBox()
        self.cmb.setMaxVisibleItems(20)
        self.cmb.setMinimumWidth(240)
        self.cmb.currentIndexChanged.connect(self._on_picked)
        self.widgets["map_id"] = (self.cmb, "combo")
        form.addRow("地图", self.cmb)
        field_tip(form.labelForField(self.cmb),
            "要识别的地图，只能从下拉里选（不让手输 ID，免得拼错一路错下去）。\n  · 选中即生效：会同时把「这张图 + 它的怪列表」写进项目，这张卡片没有「运行」按钮；\n  · 显示「（未选地图）」= 项目里还没选，此时打开怪物弹窗一个候选都没有；\n  · 上面的「筛选」只影响列表显示，不会替你选中某张图。")

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

        # ⭐⭐ 「确认要识别怪物」那个按钮 + 「角色」那个下拉**搬到卡片 4「自动标注」了**
        #   （用户 2026-10-05 ✓ 原话："现在可以把 模型训练的 卡片1里的内容归并整理到卡片4自动标注了 /
        #    现在卡片1只留：筛选、只看有怪、地图（以及下面的灰字、刷新列表按钮）"）——
        #   卡片 1 从此只管「选哪张图」✓：
        #     · 怪种 = 卡片 4「要标注的怪物」那块的背包窗格 + 添加…/移出选中 ✓
        #       （`LabelCard._mob_add` / `_mob_del` ✓）；
        #     · 角色 = 卡片 4「要标注的玩家」那块 ✓（`LabelCard._player_add` ✓ 走
        #       `gui/player_picker.PlayerPickDialog` ✓）。
        #   ⚠ 两边写的是**同一份账** ✓：`mobs` / `mob_names` / `mobs_cleared` 与 `player_id` ✓
        #     —— 卡片 4 那边走 `_write_mobs` / `_player_add` ✓ 一处口径 ✓。

    # ---------------- 列表 ----------------

    def reload_maps(self):
        """重读地图清单（工具栏导出完成后要刷一次）。"""
        self._pool = wzexport.list_maps(only_with_mob=False, keyword="")
        self._refill()

    # ⚠ 2026-10-05：原 `PLAYER_ROOT = Path("datasets/sprites/player")` 一并删掉 ✓ ——
    #   它只服务于上面那三个已删的方法 ✓；角色模板目录那条口径现在只在
    #   `gui/player_picker.py::PLAYER_ROOT` / `list_players()` 一处 ✓（两份必然漂 ✗）。

    #: 地图下拉里「还没选地图」的占位项。项目没选地图时**显示的就是它** ——
    #: 不能空着让下拉自动落到第一张图上：那样用户以为已经选中了，其实项目里是空的
    #: （"显示 ≠ 已提交"），接着打开怪物弹窗一个怪都没有（实测踩过）。
    NO_MAP = "（未选地图）"

    # ⚠ 2026-10-05：`reload_players` / `_refresh_players` / `_on_player_picked` **整组删掉** ✓
    #   —— 角色选择搬到卡片 4「标注玩家」段了 ✓（`LabelCard._player_add` ✓ 走
    #   `gui/player_picker.PlayerPickDialog` ✓ 那边有一份 `list_players()` ✓ 口径只此一处 ✓）。
    #   ⛔ 留在这儿是最糟的组合：它们引用的 `self.cmb_player` / `self.lbl_player` 已经不存在 ✗
    #   ⇒ 死代码 + 谁一调就 `AttributeError` ✗。

    def _refill(self):
        self._shown = wzexport.list_maps(only_with_mob=self.value("only_mob"),
                                        keyword=self.value("map_filter"))
        self._fill_combo()

    def _fill_combo(self):
        keep = self.project.get("map_id") if self.project else None

        self.cmb.blockSignals(True)
        self.cmb.clear()
        if not keep:
            # 项目还没选地图 → 显示占位项。**必须显示出来**：以前空着就自动显示
            # 列表的第一张图，用户以为"已经选中这张了"，其实项目里还是空的 ——
            # 接着点「确认要识别怪物」，弹窗一个怪都没有（实测踩过）。
            self.cmb.addItem(self.NO_MAP, "")
        for m in self._shown:
            self.cmb.addItem(m["label"], m["id"])

        idx = self.cmb.findData(keep) if keep else 0
        if keep and idx < 0:
            # 项目选的地图若被筛选条件挡掉了，补一条占位让它仍能显示。
            # 否则下拉会回落到第一项，下次运行 map_id 就被悄悄改掉了。
            self.cmb.insertItem(0, "%s   （不在当前筛选范围内）" % keep, keep)
            idx = 0
        if idx >= 0:
            # 列表重建后 currentIndex 会掉到第一项 → 显式恢复到项目里那张图。
            # 保证**下拉显示的就是项目里的**：改一下筛选框就会"显示 A、存的 B"
            # （这条也是实测踩出来的，和上面那条是同一类坑）。
            self.cmb.setCurrentIndex(idx)

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
            return                      # 选到了「（未选地图）」占位项：项目里没有可写的

        self._commit_pick(mid)

    def _commit_pick(self, mid):
        """选地图 = 把**这张图**和**它的怪列表**一起写进项目。返回是否写成。

        怪列表必须跟着地图一起写：项目只改了 map_id、`mobs` 还是空的话，
        「确认要识别怪物」弹窗就会一个怪都没有（实测踩过）。
        ⚠ 那四件**统一由 `wzexport.apply_map_choice` 写**（2026-10-02 收编 ✓）：
          路线识别页的「手动更换」走的是**同一个写口** —— 各写一份迟早一边漏一件 ✗。
        """
        m = self._manifest_entry(mid)
        if m is None:
            # 别静默失败 —— 之前这里 `return`，界面看着像"选上了"，其实什么都没存
            self.set_result("⚠ 地图清单里找不到 %s —— 点「刷新列表」再选一次" % mid)
            return False

        wzexport.apply_map_choice(self.project, mid, pool=self._pool)
        self.set_result(self._describe(mid, m["mobs"], m["mob_names"]))
        # ⭐⭐ **通知"这张图变了"**（用户 2026-10-04 ✓ 现场：换图后 A 机「当前地图」不变、
        #    小地图框选还按上一张图取 ⇒ `mmap_note` 报「框选区超出当前画面」✗）。
        #    ⚠⚠ 原来这里**一个信号都不发** ✗ —— 只有**路线识别页**的「手动更换」才会推
        #    （`route_panel._push_mmap` ✓）⇒ 从**这张卡的下拉**换图时，实时面板/线程里那份
        #    「现在是哪张图」还是旧的 ✗ ⇒ ① 小地图那块按旧图取区域 ✗ ② 也不会再给 A 发
        #    `MAP`（除非实时本来就是收流模式 ✓）✓。落点见 `MainWindow._on_map_changed` ✓。
        self.map_changed.emit(mid)
        return True

    def _manifest_entry(self, mid):
        """按 id 查地图清单条目（缓存为空时先补读一次）。查不到返回 None。"""
        if not mid:
            return None
        if not self._pool:
            self._pool = wzexport.list_maps(only_with_mob=False, keyword="")
        return next((x for x in self._pool if x["id"] == mid), None)

    def _fill_mobs_from_manifest(self, p, mid, override=False):
        """项目里有地图、却没有怪 → 按清单把「该识别的怪」补上。返回是否补了。

        **为什么需要**：`mobs` 只在「选地图」时写入，而"新建项目时手填地图 ID"
        或"下拉显示 ≠ 已提交"这两种情况都会留下「有地图、怪列表空」的项目 →
        弹窗空着，看着像 bug（实测踩过）。这种状态在界面上本来就是 warn
        （「该图无怪」），补上是安全的。

        override=False 时只在**空**的前提下补，不覆盖用户手动增删的结果；
        用户主动清空过（`mobs_cleared`）也不补 —— 那是他的决定，不该每次打开
        弹窗都给他加回来。
        """
        mid = (mid or "").strip()
        if not p or not mid:
            return False
        if (p.get("mobs") or []) and not override:
            return False
        if p.get("mobs_cleared") and not override:
            return False
        m = self._manifest_entry(mid)
        if m is None or not m["mobs"]:
            return False                # 清单里这张图确实没有怪：不编造
        p.set("mobs", list(m["mobs"]))
        p.set("mob_names", list(m["mob_names"]))
        p.set("mobs_cleared", False)
        p.save()
        return True

    def _pick_mobs(self):
        """打开「确认要识别怪物」弹窗，手动增删这张图要处理的怪。"""
        if self.project is None:
            return

        # 用**当前显示的那张图**（若还没提交，就在这一刻提交）。
        # 以前直接读 project.get("mobs")：下拉显示 A、项目里是空的/是 B 的时候，
        # 弹窗要么空、要么列出另一张图的怪（实测踩过）。
        mid = ((self.value("map_id") or "").strip()
               or (self.project.get("map_id") or "").strip())
        if not mid:
            QMessageBox.information(
                self, "先选地图",
                "还没选地图 —— 先在「地图」下拉里选一张（列表太长就用上面的「筛选」）。\n"
                "「要识别的怪物」是跟着地图走的，所以得先有地图。")
            return
        if (self.project.get("map_id") or "") != mid:
            if not self._commit_pick(mid):
                return
        else:
            self._fill_mobs_from_manifest(self.project, mid)

        cur = self.project.get("mobs") or []
        if not cur:
            # 空列表要给出**原因**（图本身没怪 / 被清空过 / 清单里找不到），
            # 否则又是一句"没有怪物"让人猜。
            if self.project.get("mobs_cleared"):
                why = "这张图是你上次手动清空的（可以在这里重新添加）。"
            elif self._manifest_entry(mid) is None:
                why = "地图清单里没有这张图 —— 先在工具栏「WZ 导出」重导地图清单。"
            else:
                why = ("地图清单里 %s 没有登记任何怪。\n"
                       "如果确定这张图有怪，先在工具栏「WZ 导出」重导地图清单。" % mid)
            QMessageBox.information(self, "这张图没有怪", why)

        from gui.mob_picker import MobPickDialog

        dlg = MobPickDialog(cur, self)
        if dlg.exec_():
            new = dlg.result_mobs()
            name_map = wzexport.build_mob_name_map()
            self.project.set("mobs", new)
            self.project.set("mob_names", [name_map.get(m, "") for m in new])
            # 记一笔「用户主动清空」：否则下次打开弹窗会被 _fill_mobs_from_manifest
            # 按清单加回来，等于悄悄撤销他的决定
            self.project.set("mobs_cleared", not new)
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
            # 「有地图、没怪」的项目（新建时手填了地图 ID 的那种）在这里补一次：
            # 不补的话卡片 4 那个「添加…」弹窗里没有预选（实测踩过）✓
            self._fill_mobs_from_manifest(p, p.get("map_id"))
            # ⚠ **角色的回填不在这儿了** ✗（用户 2026-10-05 ✓ 搬到卡片 4「标注玩家」段了 ✓）——
            #   那张卡自己从项目读 `player_id` 并画格子 ✓（`LabelCard._refresh_player_list` ✓），
            #   没选过就是 0 个格子 ✓（不替人默认挑一个 ✗ 那会悄悄改项目 ✓）。
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
        """卡片最下面那行灰字 —— **2026-10-05 起整行不要了**。

        用户原话："「卡片1最下面的灰字移除」指 选中地图后显示的
        「{地图id}·{n}种怪：[{怪名}] · 角色：{角色名}」" ✓
        ⇒ 那行就是这里产出的（`base.refresh` → `set_result` ✓）。

        ⚠ 三件事说清（别以后有人"顺手补回来" ✗）：
          · 怪种与角色**都搬到卡片 4 了** ✓ ⇒ 这里再写就是**两处口径** ✗；
          · `__init__` 里已经把 `self.result` **整个藏了** ✓ —— 返回 "" 只会看到 "—"
            （`base.set_result` 的 `text or "—"` ✓）那不算"移除" ✗；
          · 留这个空实现是为了**别回落到基类那句 "—"** ✓（`base.summarize` 返回 "—" ✓）。
        """
        return ""

    def detect_state(self, p):
        # ⚠ 2026-10-05：「未选角色」那一条**去掉了** ✗ —— 角色现在在卡片 4「标注玩家」选 ✓
        #   卡片 1 的状态只说清"这张图选定没有、它有没有怪" ✓（那才是它现在管的事 ✓）。
        if not p.get("map_id"):
            return ("idle", "未选")
        if not (p.get("mobs") or []):
            return ("warn", "该图无怪")
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
                   choices=["window", "file", "stream"],
                   tip="采集方式，三选一；切换只显隐对应参数行，不会清掉已填的值。\n  · 窗口 = 直接抓游戏窗口客户区（画面最干净，推荐）；\n  · 文件 = 从已有录屏抽帧（录屏常把整个桌面录进去，标注时还得另设区域）；\n  · 流 = 实时流采集，目前还没实现，选了跑不了。"
        )

        self.field(form, "file", "文件", "path", "",
                   filter="视频文件 (*.mkv *.mp4 *.avi *.mov);;所有文件 (*)",
                   on_pick=self._on_file_picked,
                   tip="要抽帧的录制文件（.mkv/.mp4/.avi/.mov）。\n  · 点「浏览」选一个，选完下面会按容器元数据预估大约能抽出多少张；\n  · 留空 = 还没选，运行时报「尚未选择录制文件」；\n  · 文件不存在同样会报错，换电脑后要重选路径。"
        )

        self.cmb_win = NoWheelComboBox()
        self.cmb_win.setMinimumWidth(230)
        self.widgets["win_rect"] = (self.cmb_win, "combo")
        form.addRow("窗口", self.cmb_win)
        field_tip(form.labelForField(self.cmb_win),
            "要抓取的游戏窗口。\n  · 列表是当前可见窗口，点「刷新」重新枚举；\n  · 抓的是窗口客户区（桌面 / 任务栏进不来，自动排除边框）；\n  · 没选（空）= 运行时报「请先选择一个窗口」；窗口捕获还需要 pywin32。")

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

        self.field(form, "url", "流地址", "str", "udp://0.0.0.0:5000",
                   tip="实时流的接收地址（来源 = 流 时用）。\n  · 形如 udp://0.0.0.0:5000，表示在本机 5000 端口收 UDP 流；\n  · 目前「流」这个来源还没实现，填了也用不上，请先用「窗口」或「文件」。"
        )
        self.field(form, "fps", "抓帧频率", "float", 5.0,
                   minimum=0.5, maximum=60.0, decimals=1, step=0.5,
                   tip="窗口捕获时每秒抓几张，单位 帧/秒。\n  · 范围 0.5~60（控件填不进 0 或负值）；\n  · 调大 → 覆盖更全，但文件涨得快、后面标注更费时；\n  · 只对「窗口」来源生效 —— 文件抽帧用的是「抽帧间隔」。"
        )
        self.field(form, "stride", "抽帧间隔", "int", 6, minimum=1, maximum=1000,
                   tip="文件抽帧时每隔多少帧保留 1 张，单位 帧。\n  · 1 = 每帧都抽（最多，几秒视频就能堆出上百张）；\n  · 调大 → 抽得更稀、张数更少、省标注时间，但可能漏掉短动作；\n  · 改它下面会实时刷新「预估张数」（与真正抽帧同一套公式）。"
        )

        # ⭐⭐ **预估多少张**（用户 2026-09-29 ✓ 原话："采集、选中文件后，能否在下面显示
        #   预估多少帧？" ⇒ 又要求"信息应该写在「抽帧间隔」参数下面"✓）—— 所以这行就摆在
        #   **「抽帧间隔」的正下方** ✓（它本来就是"÷ 这个间隔"的结果，摆这儿最顺 ✓）。
        #   ⚠ **公式必须和 `tools/extract_frames` 一模一样**（`总帧数 // stride + 1` ✓）：
        #     真正抽多少张是那边决定的 ⇒ 两处口径不一致会出现"预估 100、实际 97"，
        #     人的第一反应是"是不是抽漏了" ✗。
        self.lbl_est = QLabel("—")
        self.lbl_est.setStyleSheet("color:#80868b;")
        self.lbl_est.setWordWrap(True)
        form.addRow("", self.lbl_est)
        self.field(form, "seconds", "时长(秒)", "float", 120.0,
                   minimum=0, maximum=36000, decimals=0, step=30,
                   tip="窗口捕获持续多少秒。\n  · 0 = 一直抓到手动停（默认 120 = 抓满 2 分钟自动停）；\n  · 调大 → 录更久，注意磁盘占用（张数还受「抓帧频率」「最多张数」限制）；\n  · 只对「窗口」来源生效。"
        )
        w = self.field(form, "dedup", "去重阈值", "float", 0.01,
                       minimum=0.0, maximum=1.0, decimals=3, step=0.005)
        w.setToolTip("相似帧去重：缩略图里亮度差超过 8 的像素占比（0~1）小于它就跳过。\n"
                     "0 = 不去重；值越大去重越激进。\n"
                     "完全静止≈0；待机动画/角色移动会明显高于它，用 0.01 能保留这些帧。")
        self.field(form, "limit", "最多张数", "int", 0, minimum=0, maximum=1000000,
                   tip="本次最多保存多少张。\n  · 0 = 不限（抓到时长到 / 文件抽完为止）；窗口与文件两种来源都生效；\n  · 预估张数超过它时会被封顶（下面会提示「已被『最多张数』封顶」）；\n  · 调小可先快速采一批，避免一次生成上千张难标注。"
        )

        w = self.field(form, "clean", "重抽前清空", "bool", True)
        w.setToolTip("勾上：先删掉输出目录里旧的 frame_*.png 再抽。\n"
                     "不勾：只覆盖同编号的文件，新视频较短时会残留旧帧。")

        # ⚠⚠ **信号连接必须放在这里（`build_params` 末尾）** ——
        #   不能在加标签那一刻就连：`stride` / `limit` 两个控件在**下面**才被 `field()`
        #   创建 ⇒ 那时 `self.widgets` 里还没有它们 ⇒ `widgets.get(...)` 拿到 `None`
        #   ⇒ **连接静默失效** ✗（表现就是用户报的"我改了抽帧间隔，这个数字不变"✓
        #     2026-09-29 踩过 ✓）。放在末尾 ⇒ 两个控件都已经在 `widgets` 里 ✓。
        for _k in ("stride", "limit"):
            _w = self.widgets.get(_k, (None, None))[0]
            if _w is not None:
                _w.valueChanged.connect(lambda *_a: self._refresh_est())

        self.widgets["source"][0].currentTextChanged.connect(self._on_source)
        self._on_source("window")     # 初始状态（此时还没绑定项目）

    def _on_file_picked(self, path):
        """选完文件 ⇒ 刷新「预估多少张」✓（`field(..., "path", on_pick=...)` 的回调 ✓）。"""
        self._refresh_est(path)

    def _refresh_est(self, path=None):
        """算「大概能抽出多少张」写进 `self.lbl_est`（文件行下面那行灰字 ✓）。

        ⭐ 用户 2026-09-29 ✓："采集、选中文件后，能否在下面显示预估多少帧？"

        ⚠ **公式照抄 `tools/extract_frames`**（`总帧数 // stride + 1` ✓ 见那边 `:127-130`）：
        真正抽多少张由那边决定 ⇒ 两处口径一份，否则"预估/实际"对不上，人会以为漏帧 ✗。
        ⚠ 用 **PyAV**（`av`，和 `extract_frames` **同一个库** ✓ 别为这事再引入 cv2 那套 ✗）
        读**容器元数据** —— 只读头部、**不解码** ⇒ 本地文件**毫秒级** ✓ 放 UI 线程可以 ✓。
        ⚠ 网络盘 / 超大文件可能慢一点 ⇒ 整体包 `try`，失败就直说"量不出来" ✓
        **绝不抛异常、绝不卡住卡片** ✗（估算而已，量不出来不影响抽帧 ✓）。
        """
        lbl = getattr(self, "lbl_est", None)
        if lbl is None:
            return
        if path is None:
            path = self.value("file")
        if not path:
            lbl.setText("—")
            return
        total = 0
        note = ""
        try:
            import av
            c = av.open(str(path), mode="r")
            try:
                vs = list(getattr(c, "streams", None).video) if c.streams else []
                st = vs[0] if vs else None
                if st is not None:
                    fps = float(getattr(st, "average_rate", 0) or 0)
                    _tb = float(getattr(st, "time_base", 0) or 0)
                    # ① 容器直接报的总帧数（最准 ✓ 实测 mp4 一般都有）
                    total = int(getattr(st, "frames", 0) or 0)
                    # ② 流自己的时长 × 帧率（有的容器 `frames` 是 0 但 `duration` 有值 ✓）
                    if not total and fps and _tb:
                        _d = float(getattr(st, "duration", 0) or 0)
                        if _d > 0:
                            total = int(_d * _tb * fps)
                            note = "（按时长×帧率估的）"
                    # ③ ⭐ **容器级时长**兜底 —— **MKV 录屏（OBS 之类）常常 ①② 都拿不到** ✗，
                    #    但 `container.duration / av.time_base` 有值 ✓。
                    #    实测 `data/recordings/plain02.mkv`：99.4 秒 × 36 fps ≈ **3579** 帧，
                    #    真值（数包）**3580** ⇒ 只差 1 ✓ 够用了 ✓
                    #    （⚠ 别改成"数包"：实测 3580 包要 0.49s、14471 包要 1.22s ⇒ 卡 UI ✗）
                    if not total and fps:
                        _cd = float(getattr(c, "duration", 0) or 0)
                        if _cd > 0:
                            total = int(_cd / float(av.time_base) * fps)
                            note = "（按容器时长×帧率估的）"
                else:
                    note = "（文件里没有视频轨）"
            finally:
                c.close()
        except Exception as e:                  # noqa: BLE001 —— 估算失败不该影响任何事 ✓
            lbl.setText("预估：读不出这个文件的信息（%s）" % e)
            return
        try:
            stride = max(1, int(self.value("stride") or 1))
            limit = int(self.value("limit") or 0)
        except (TypeError, ValueError):
            stride, limit = 1, 0
        if not total:
            lbl.setText("预估：这个文件没报总帧数（抽的时候才知道%s）" % note)
            return
        est = total // stride + 1
        capped = bool(limit > 0 and est > limit)
        if capped:
            est = limit
        lbl.setText("预估约 **%d** 张  ←  总 %d 帧 ÷ 每 %d 帧抽 1 张%s%s"
                    % (est, total, stride, note,
                       "，已被「最多张数」封顶" if capped else ""))
        lbl.setToolTip(
            "这只是**按容器元数据**算的上限估计（和真正抽帧用的是同一个公式 ✓）。\n"
            "实际张数还会少一些，因为「去重阈值」会把相似帧跳掉。")

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
        # ⭐ 预估那行也要跟来源走（它不是 `field` ⇒ 不会被 `set_row_visible` 管到 ✗）：
        #   窗口 / 流模式下没有"既定帧数"可估 ⇒ 必须藏掉，否则会留上一次的残留数字 ✗
        lbl = getattr(self, "lbl_est", None)
        if lbl is not None:
            lbl.setVisible(is_file)

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
        # ⭐ 回填完**顺手刷一次预估**（用户 2026-09-29 的连带项 ✓）。
        #   ⚠ **为什么必须补这一下**：`on_pick` 只在**用户点「浏览」选文件**时触发 ⇒
        #     从项目文件**回填**上次选的那个视频时**不会**触发 ⇒ 标签一直停在 `—` 上 ✗
        #     （明明有文件却不显示，看着像功能没做 ✓ 2026-09-29 实测到的 ✓）。
        #   ⚠ 必须加在**这个**方法里 —— 卡片里本来就有 `load_from_project`，
        #     另起一个同名方法会把它**整个顶掉**（回填就废了 ✗ 2026-09-29 踩过 ✓）。
        try:
            self._refresh_est()
        except Exception:                   # noqa: BLE001 —— 估算不许把回填搞崩 ✓
            pass

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

        # ⭐⭐ **掉落物 / 宠物也能单独标定**（用户 2026-10-05 ✓ 原话："卡片3标定尺度 也要支持
        #   标定掉落物、宠物类"）—— 两类各存一个尺度（`drop_scale` / `pet_scale` ✓ 与玩家的
        #   `player_scale` 同款 ✓）：它们的模板来自**另外两套图库**（掉落物图标 / 宠物组合外观 ✓），
        #   跟怪、人不是同一批精灵 ⇒ 该能量就单独量 ✓（④ 里那两条 `make_task` 会优先用它 ✓
        #   见 `LabelCard._sprite_scale(p, "drop"/"pet")` ✓；**没标过就回落到总尺度** ✓ 老项目照旧 ✓）。
        #   ⚠ 状态**只在 ④ 里勾了那一类**时才计入卡片总状态 ✓（见 `_class_state` ✓）——
        #     没打算标掉落物的人，不该被一个"未标定"把卡片卡成黄灯 ✗。
        self.lbl_drop_state = QLabel("—")
        self.lbl_drop_state.setWordWrap(True)
        form.addRow("掉落物状态", self.lbl_drop_state)

        self.lbl_pet_state = QLabel("—")
        self.lbl_pet_state.setWordWrap(True)
        form.addRow("宠物状态", self.lbl_pet_state)

        # ---- 一键设置尺度：所有精灵统一基准 ----
        note = QLabel("给怪物 / 玩家 / 掉落物 / 宠物统一设一个尺度基准"
                      "（之后可在手动标定里分开微调）。")
        note.setStyleSheet("color: #80868b;")
        note.setWordWrap(True)
        form.addRow(note)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.sp_apply_all = NoWheelDoubleSpinBox()
        self.sp_apply_all.setRange(0.05, 8.0)
        self.sp_apply_all.setDecimals(3)
        self.sp_apply_all.setSingleStep(0.05)
        self.sp_apply_all.setValue(1.0)
        row.addWidget(self.sp_apply_all, 1)
        # 单位写在框**外面**（UI 规范 §9：不写进编辑框）
        row.addWidget(QLabel("×"))
        btn_apply = QPushButton("一键设置尺度")
        btn_apply.setToolTip("把怪物 / 玩家 / 掉落物 / 宠物的尺度基准都设为这个值，"
                             "并清空逐怪覆盖（之后可在「手动标定」里分开微调）")
        btn_apply.clicked.connect(self._apply_all_scale)
        row.addWidget(btn_apply)
        form.addRow("尺度基准", row)

        # ---- 手动目测标定 ----
        btn_manual = QPushButton("手动目测标定")
        btn_manual.setToolTip("在画面上叠加模板，手动拖动 + 调缩放对齐，目测确定尺度。\n"
                              "目标里可以选：怪物 / 玩家 / 掉落物 / 宠物（后两类各记一个自己的尺度）")
        btn_manual.clicked.connect(self._manual_calib)
        form.addRow("手动标定", btn_manual)

    def _manual_calib(self):
        """打开手动目测标定弹窗，把目测的缩放写回项目。

        ⭐ 目标多两类（用户 2026-10-05 ✓ "也要支持标定掉落物、宠物类"）——
        两类各自的**模板来源**由弹窗自己解析 ✓（掉落物图标库 / 宠物组合外观 ✓ 见
        `CalibManualDialog._on_target` ✓），这里只把**清单**递过去 ✓。
        """
        if self.project is None:
            return
        p = self.project
        from gui.calib_manual import CalibManualDialog

        sec = p.sec("label")
        dlg = CalibManualDialog(p.get("mobs") or [], p.get("player_id") or "",
                                p.frames, p.get("mob_scales") or {},
                                p.get("scale") or 1.41,
                                p.get("player_scale") or 1.41, self,
                                # ⚠ 这两样走**关键字**传 ✓：老调用方是位置参数（上面那 7 个 ✓），
                                #   加在末尾当 kw 就不会动到任何一处老写法 ✓。
                                drops=list(sec.get("drops") or []),
                                pets=list(sec.get("pets") or []),
                                drop_default=p.get("drop_scale") or p.get("scale") or 1.41,
                                pet_default=p.get("pet_scale") or p.get("scale") or 1.41)
        if dlg.exec_():
            self._save_manual_scale(dlg.result_target(), dlg.result_mob(),
                                    dlg.result_frame(), round(dlg.scale_value(), 3))

    def _save_manual_scale(self, target, mid, frame, scale):
        """把「手动目测标定」的结果写回项目 ⇒ True = 真写了 ✓（False = 这一下没东西可写 ✓）。

        ⚠ 抽出来是为了**能测** ✓：弹窗本身是模态的（`exec_()` ✓）跑不了 ✓，
        而"结果往哪存"才是会出错的地方 ✓（本轮新增的两类就错在这里的风险最大 ✓）。

        口径（用户 2026-10-05 ✓ 加了掉落物 / 宠物两类）：
          · `mob` / `player` ⇒ 照旧写**帧级**覆盖 `mob_scales["id:帧"]` ✓
            （那条链 `detect_mobs` / `detect_player` 本来就吃它 ✓ `frame_scales` ✓）；
          · `drop` / `pet` ⇒ 写**这一类自己的尺度** `drop_scale` / `pet_scale`
            （+ `*_at` 记下当时的画面尺寸 ✓ ⇒ 换分辨率后状态会变「需要重标」✓）；
            ⛔ **不写** `mob_scales` ✗ —— 掉落物 / 宠物**没有逐帧标定那一套** ✓
            （键是「怪 id:帧」✓ 见 `detect_mobs` ✓；硬塞进去只会把尺度带歪 ✓
             同 `LabelCard._sprite_scale` 里那段说明 ✓）。
        """
        if self.project is None or not scale:
            return False
        p = self.project
        w, h = self._frame_size()
        if target in ("drop", "pet"):
            key = "drop_scale" if target == "drop" else "pet_scale"
            p.set(key, float(scale))
            p.set(key + "_at", {"width": w, "height": h, "confidence": "manual"})
            p.save()
            self.refresh()
            return True
        # 怪 / 玩家：帧级覆盖（老口径 ✓ 一字不变 ✓）
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
        return True

    def _apply_all_scale(self):
        """一键设置尺度：怪物 / 玩家 / 掉落物 / 宠物统一用它，并清空逐怪覆盖。"""
        if self.project is None:
            return
        p = self.project
        v = round(self.sp_apply_all.value(), 3)
        w, h = self._frame_size()
        at = {"width": w, "height": h, "confidence": "manual"}
        p.set("scale", v)
        p.set("player_scale", v)
        # ⭐ 掉落物 / 宠物也一样设（用户 2026-10-05 ✓）—— 它们与"总尺度"同源，
        #   一键设完全一致 ✓；要分开就在「手动标定」里各调各的 ✓（那时起才真正分叉 ✓）。
        p.set("drop_scale", v)
        p.set("pet_scale", v)
        p.set("mob_scales", {})
        p.set("scale_at", dict(at))
        p.set("player_scale_at", dict(at))
        p.set("drop_scale_at", dict(at))
        p.set("pet_scale_at", dict(at))
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

    #: 掉落物 / 宠物：尺度键 ↔ ④ 里那个"这一类要不要标"的开关（`label` 段 ✓）
    _CLASS_CALIB = {"drop": ("drop_scale", "drop_scale_at", "label_drops_on"),
                    "pet": ("pet_scale", "pet_scale_at", "label_pets_on")}

    def _class_state(self, kind):
        """掉落物 / 宠物的标定状态 ⇒ (文字, 颜色, 说明) ✓（用户 2026-10-05 ✓）。

        ⭐⭐ **只有 ④ 里勾了那一类才算数**：没打算标掉落物的人不该被一个"未标定"把
        整张卡片卡成黄灯 ✗（`detect_state` 靠**颜色**判档 ✓ 这里的"未用"用灰色 ⇒
        天然不计入 fail/warn ✓ 见 `detect_state` ✓）。
        """
        key, at_key, sw_key = self._CLASS_CALIB[kind]
        p = self.project
        if p is None:
            return "—", "#5f6368", ""
        if not (p.sec("label") or {}).get(sw_key):
            return "未用", "#5f6368", "④ 里勾上「标注%s」后在这里标定" % (
                "掉落物" if kind == "drop" else "宠物")
        state = self._one_state(key, at_key)
        if state[0] == "未标定":
            # ⚠ 允许不标 ✓（不标 ⇒ ④ 那条回落**总尺度** ✓ 见 `LabelCard._sprite_scale` ✓）——
            #   所以这里说清"可以标，也可以不标"，别写成"必须先标" ✗。
            return state[0], state[1], "点「手动目测标定」实测一次（不标则沿用总尺度）"
        return state

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

        for kind, lbl in (("drop", self.lbl_drop_state), ("pet", self.lbl_pet_state)):
            text, color, note = self._class_state(kind)
            lbl.setText(text + ("　" + note if note else ""))
            lbl.setStyleSheet("color: %s;" % color)

        # 尺度基准输入框回填为当前全局怪物尺度 —— 否则重启后显示默认值，
        # 用户会误以为之前设的基准丢了（其实 scale 已落盘，只是输入框没同步）
        if self.project is not None:
            base = self.project.get("scale")
            self.sp_apply_all.blockSignals(True)
            self.sp_apply_all.setValue(float(base) if base else 1.41)
            self.sp_apply_all.blockSignals(False)

    # ---------------- 项目交互 ----------------

    def summarize(self, p):
        """卡片最下面那行灰字：**报"现在拿哪个数在跑"** —— `{类名}:{尺寸基准}` ✓
        （用户 2026-10-05 ✓ 原话："卡片3里最下面的灰字改成显示「{类名}:{尺寸基准}」而不是
        「{类名}：已标定」"）。

        ⚠ 它与上面那几行**状态**分工不同（别合并 ✗）：
          · 上面几行（怪物状态 / 玩家状态 / 掉落物状态 / 宠物状态）= **状态词** + 该怎么办 ✓
            —— 回答"要不要动手" ✓；
          · 这一行 = **读数**（`怪:1.406　·　人:1.406 …` ✓）—— 回答"现在用的是哪个数" ✓。
        ⚠ 两条例外（都没有数可报 ✗）：
          · 没标过（值缺失）⇒ 还是写状态词（`怪:未标定` ✓）；
          · 有数但状态不是"已标定" ⇒ 数后面挂个**短状态**（`1.406（需要重标）` ✓）——
            不然"换分辨率要重标"这条就只藏在上面那行了 ✗（本轮特意留着它 ✓；
            长说明仍只在上面那行 ✓ 免得这行被撑爆 ✗）。
        """
        def _cell(label, key, state):
            """一格：有数 ⇒ `类名:尺度`（+ 非"已标定"时挂短状态 ✓）；没数 ⇒ `类名:状态词` ✓。"""
            try:
                _raw = p.get(key) if p is not None else None
                val = float(_raw) if _raw else None
            except (TypeError, ValueError):        # 老配置里存了脏值 ⇒ 当没标过 ✓
                val = None
            if val is None:
                return "%s:%s" % (label, state[0])
            if state[0] == "已标定":
                return "%s:%.3f" % (label, val)
            return "%s:%.3f（%s）" % (label, val, state[0])

        parts = [_cell("怪", "scale", self._mob_state()),
                 _cell("人", "player_scale", self._player_state())]
        # ⭐ 掉落物 / 宠物：**用了才写进摘要** ✓（没勾就别把那一行撑长 ✗）
        for kind, label in (("drop", "掉落"), ("pet", "宠物")):
            st = self._class_state(kind)
            if st[0] != "未用":
                parts.append(_cell(label, self._CLASS_CALIB[kind][0], st))
        return "  ·  ".join(parts)

    def detect_state(self, p):
        # 怪、人、掉落物、宠物四个状态分别算，再合并成卡片总状态：
        # 任一 fail → fail；任一 warn（未标定/存疑/需重标）→ warn；都 done → done。
        # ⚠ 掉落物 / 宠物**没被 ④ 勾上**时是灰色的「未用」✓ ⇒ 天然不计入 fail/warn ✓
        #   （不用在这里写 if ✓ 一处口径在 `_class_state` ✓）。
        mt, mc, _mn = self._mob_state()
        pt, pc, _pn = self._player_state()
        dt, dc, _dn = self._class_state("drop")
        et, ec, _en = self._class_state("pet")
        cs = {mc, pc, dc, ec}
        if "#c5221f" in cs:
            st = "fail"
        elif "#b06000" in cs:
            st = "warn"
        else:
            st = "done"
        msg = "怪:%s 人:%s" % (mt, pt)
        for t, kind in ((dt, "drop"), (et, "pet")):
            if t != "未用":
                msg += " %s:%s" % ("掉落" if kind == "drop" else "宠物", t)
        return (st, msg)


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
        # 选帧弹窗的结果：{"mob": [帧名…] / None, "player": …}，None = 不筛（全选）
        self._only = {}

        # 通用「运行」按钮隐藏 —— 活交给**四段各自那两个按钮**（`模板匹配标注{类}` /
        # `YOLO标注{类}` ✓ 用户 2026-10-05 ✓ 原话："模板匹配标注{类名}→按钮、YOLO标注{类名}→按钮"）。
        # ⚠ 2026-10-05 之前是页脚两个「标注怪物 / 标注玩家」+ 一行「YOLO 标注（怪物/玩家）」✗
        #   ⇒ 四段统一之后**都搬进各自的段**了 ✓（每段一对 ✓ 作用对象一眼看得见 ✓）。
        self.btn_run.hide()
        self._yolo_only = False          # 本次运行是不是「只跑 YOLO 标注」

        # ---- YOLO 权重（**四段共用这一行** ✓ 只有它没法塞进某一段）----
        aux = QHBoxLayout()
        aux.setSpacing(6)
        self.lbl_yolo_weight = QLabel("YOLO 权重")
        aux.addWidget(self.lbl_yolo_weight)
        self.cmb_yolo = NoWheelComboBox()
        self.cmb_yolo.setMaxVisibleItems(12)
        field_tip(self.lbl_yolo_weight, 
            "给「YOLO 标注」用的模型。\n\n"
            "顺带说明：跑「标注怪物 / 标注玩家」时，如果这里选了模型，\n"
            "模板匹配会先用它**粗定位**（只在模型圈出的区域里找）—— 实测快几十倍，\n"
            "还能避开画面里长得像目标的固定物件（例如地图上的绳子）。\n"
            "模型没圈到的帧仍然全图匹配，所以不会因此漏标。")
        aux.addWidget(self.cmb_yolo, 1)
        self.layout().addLayout(aux)

        # ⭐⭐ **怪物段 / 玩家段**（用户 2026-10-05 ✓ 原话："标注{类名}…要标注的{类名}…背包窗格
        #   （玩家、怪物也用同样的背包窗格的形式添加，怪物用stand或fly当icon，玩家用stand当icon）…
        #   背包窗格的添加、移出选中按钮"）—— 与掉落物 / 宠物那两块同构 ✓。
        #   ⚠ 这两段的 5 个参数**暂时还在上面那份表单里**（纯搬移，下一步做 ✓）；
        #     背包窗格双击选模板帧也还没接这两块（各一行 ✓ 管线与存储就绪 ✓）。
        self._build_player_section()
        self._build_mob_section()

        # ---- ⭐⭐ 掉落物那一块（**卡片最下面** ✓ 用户 2026-10-04 ✓）----
        #   勾选「标注掉落物」才出现（**整块 `setVisible`** ✓ 照 `player_panel._refresh_spot_loop_ui`
        #   的做法 —— 藏起来才是"出现" ✓ 灰着仍是"一直在那儿" ✗）。
        #   列表 = 要标注的掉落物（增删走 `gui/drop_picker.py`：**按 id / 名称搜索** ✓）；
        #   两个按钮 = 两条标注路径（模板匹配 / YOLO 补框 ✓ 与上面"怪物"那两个同一对算法 ✓）。
        self._drops = []
        self._pets = []            # 宠物那一块（方案 C ✓ 见 `load_from_project` ✓）                 # [{"id", "name"}] ✓
        self._drop_mode = "template"     # template / yolo（本次要跑哪条 ✓）
        #: ⭐⭐ 「这条目用哪些**模板帧**」（任务 3 ✓ 用户 2026-10-05 ✓）——
        #:   键 `<target>:<模板 id>` ✓（掉落物 = 掉落 id ✓；宠物 = 组合外观目录名 ✓）；
        #:   **缺键 / 空 = 走老逻辑**（`_default_frames` ✓）⇒ 老项目零影响 ✓。
        self._frames_sel = {}
        self.drop_box = QWidget()
        dv = QVBoxLayout(self.drop_box)
        dv.setContentsMargins(0, 2, 0, 0)
        dv.setSpacing(4)

        # ⭐⭐ **提示挂在标题上**（用户 2026-10-04 ✓ 第 3 条：改成"指着字段标题时弹出。
        #   例如「要标注的掉落物」"）—— 原来这段说明挂在**窗格控件自己**身上 ✗ ⇒ 鼠标指到
        #   格子上既弹信息窗、又弹它 ⇒ 两个框叠着 ✓。`title_label` 顺手让标题**可复制** ✓
        #   （第 4 条 ✓）。规矩见 `docs/UI规范.md` §6 ✓。
        lbl = title_label(
            "要标注的掉落物",
            tip="**这个字段是干嘛的**：下面格子里列出「这次要标的掉落物」"
                "（按 id / 名称 / 描述从**道具图库**里搜出来 ✓）。\n\n"
                "· **鼠标指到格子上** ⇒ 弹出信息窗：名称 / id / 类别 / 几帧 / 描述 ✓；\n"
                "· 单击选中 ⇒ 点右边「移出选中」就能拿掉这一项 ✓；\n"
                "· 至少要有一项，下面两个按钮才会亮 ✓。")
        lbl.setStyleSheet("color:#5f6368; font-weight:600;")
        self.lbl_drop_title = lbl        # 存一份（用例要验"提示在标题上" ✓）
        dv.addWidget(lbl)

        # ⭐⭐ **背包窗格**（用户 2026-10-04 ✓ 原话："自动标注卡片上，要标注的掉落物能不能以
        #   背包窗格的形式展示项目，鼠标指着的时候显示信息弹窗（MapleNecrocer 应该有相关实现）"）。
        #   原来是 92px 高的一行行列表 ✗ ⇒ 现在换成**格子**（`gui/icon_grid.IconGrid` ✓，
        #   尺寸/版式照 MapleNecrocer 的 `AfrmItem.cs`（33×33 格 ✓）与 `ItemTooltipRender.cs`
        #   （290 宽信息窗、68 图标 ✓）—— **鼠标指到格子上就弹出名称 / id / 类别 / 几帧** ✓
        #   （见 `icon_grid.tip_lines` ✓ 口径只此一处 ✓）。
        self.grid_drop = IconGrid()
        #   三行格子（37×3 + 边框）—— 再多就滚动 ✓ 别把卡片撑太长 ✓
        # ⭐ **高度按内容自适应**（用户 2026-10-04 ✓ 原话："背包窗格需要自适应有几行（至少一行）"✓）
        #   —— 原来写死三行 ✗：只加 1~2 项时留一大片空白 ✓；`IconGrid.fit_rows` 会在
        #   `set_entries` / 宽度变化时自己重算 ✓，这里只是给个**至少一行**的起点 ✓。
        # ⭐⭐ `hard=True` = **上限也钉住**（用户 2026-10-04 ✓ 第 2 条："模型训练里的掉落物、
        #   宠物自动标注类背包窗格没有按选中的项目数量限定行数（背包窗格显示区的高度）"）——
        #   只给 `min` 不够 ✗：离屏量过，`minimumHeight` 明明是 41（1 行 ✓）而 `height()`
        #   **恒为 480** ✗（布局把多余空间全塞给它）⇒ 显示区一直那么高 ✓ 见 `IconGrid.fit_rows` ✓。
        self.grid_drop.fit_rows(min_rows=1, hard=True)
        # ⭐⭐ **双击格子 ⇒ 选模板帧**（任务 3 ✓ 用户 2026-10-05 ✓ 原话："在背包窗格中双击选中的
        #   怪物、玩家、掉落物、宠物，显示弹窗展示所有的帧，用户可以自由决定选中哪些帧用于匹配"）。
        #   ⚠ `IconGrid` 只有 `activated_id`（双击 / 回车都会发 ✓）⇒ 没另造双击信号 ✓。
        self.grid_drop.activated_id.connect(lambda _i: self._pick_entry_frames("drop"))
        # ⚠ **背包窗格自己不挂 tip**（用户 2026-10-04 ✓ 第 2 条："这个背包窗格就不需要我们数据
        #   集工作台的 tips 了"）—— 每格**鼠标一指就弹信息窗**（名称/id/类别/几帧/描述 ✓ 见
        #   `gui/icon_grid.IconTip` ✓），再挂一层 Qt 提示只会**两个框叠在一起** ✗。
        #   字段怎么用 ⇒ 挂在**标题**「要标注的掉落物」上 ✓（第 3 条的新规矩 ✓）。默认 tips 关。
        dv.addWidget(self.grid_drop)

        # ⭐⭐ **这一块自己的「匹配阈值」**（用户 2026-10-04 ✓ 原话："勾选标注掉落物、标注宠物后，
        #   需要有参数「匹配阈值」"）。
        #   为什么必须**分开**：原来掉落物直接吃上面那个总的「匹配阈值」✗（`make_task` 里
        #   `sec.get("thresh")` ✓），可**掉落物图标又小又统一**（金币才十几像素 ✗）⇒ 和怪物
        #   常常不该是同一个数 ✓（太小/太像 ⇒ 要么假匹配一堆、要么一个都匹配不上 ✓）。
        #   ⚠ 用 `field()` 而不是手搓控件：它顺手登记进 `self.widgets` ✓ `set_value` / `values`
        #     / 提示挂标题 / UI 规范那几条**全都自动吃到** ✓（手搓就得一样样补 ✗）。
        #   ⚠ 放在 `drop_box` **里面** ⇒ 没勾「标注掉落物」时它跟着一起藏 ✓（用户说的正是
        #     "勾选…后需要有这个参数" ✓）。
        _tform = QFormLayout()
        _tform.setContentsMargins(0, 0, 0, 0)
        _tform.setSpacing(4)
        self.field(_tform, "drop_thresh", "匹配阈值", "float", 0.90,
                   minimum=0.0, maximum=1.0, step=0.01, decimals=2,
                   tip="**这一块自己的匹配阈值**（掉落物模板匹配用 ✓）。\n\n"
                       "· 默认 = 上面那个总的「匹配阈值」的值 ✓（老项目一字不变 ✓）；\n"
                       "· 掉落物图标**又小又统一**（金币才十几像素）⇒ 和怪物常常不该一个数：\n"
                       "    太小/太像 ⇒ 要么**假匹配一堆**、要么**一个都匹配不上** ✓；\n"
                       "· 调它只影响**掉落物这一块** ✓，怪物的那个不受影响 ✓。")
        # ⭐⭐ **这一块自己的另外三个参数**（用户 2026-10-05 ✓ 原话："4类按照统一的格式整理…
        #   匹配阈值→两位小数 / 区分度→三位小数 / 最大模板帧→整数 / 每帧最多几个框→整数 /
        #   降采样→整数"）—— 原来它们跟怪物**共用**一份 ✗ ⇒ 这里给掉落物各存一份 ✓
        #   （⚠ 老项目没这些键 ⇒ **回落总那个** ✓ 见 `make_task` ✓ 行为一字不变 ✓）。
        self.field(_tform, "drop_min_distinct", "区分度", "float", 0.06,
                   minimum=0.0, maximum=1.0, decimals=3, step=0.01,
                   tip="同一帧里，同一个模板找第 2、3 个峰时的「去重」闸：\n"
                       "新峰的分数必须比已经收下的那个再高出这么多才算另一处。\n"
                       "  · 调小（→0）⇒ 稍亮一点就算另一只 ⇒ 同一只怪被反复框上（重叠框 ✗）\n"
                       "  · 调大 ⇒ 只有明显更亮的第二处才算另一只 ⇒ 两只挤在一起时容易只框到一只\n\n"
                       "它是相对分数（不是像素）。0 = 完全不设这道闸。\n"
                       "⚠ 老项目没存过这一项 ⇒ 用上面那个总的「区分度」✓（行为不变 ✓）。")
        self.field(_tform, "drop_per_mob", "最大模板帧", "int", 20,
                   minimum=2, maximum=60,
                   tip="这一块每条掉落物最多用几帧当模板（掉落物图标通常就几帧 ✓ 一般用不上）。\n"
                       "  · 帧数 ≤ 它 ⇒ 全用（绝大多数情况 ✓）；超过才按动作轮询取样；\n"
                       "  · 调小会漏掉「转到侧面」之类的相位 ⇒ 那种角度的图标就匹配不上 ✗\n\n"
                       "⚠ 老项目没存过 ⇒ 用上面那个总的「最大模板帧」✓。")
        self.field(_tform, "drop_max_peaks", "每帧最多几个框", "int", 4,
                   minimum=-1, maximum=999,
                   tip="同一个模板在一帧里最多写几个框（够了就停）。\n"
                       "  · 一屏同种掉落物（例：一地金币）超过这个数 ⇒ 多出来的不标 ✗\n"
                       "  · 调成 1 ⇒ 一屏只标一个（那是漏标 ✗）\n"
                       "⭐ **负数（-1）= 无限**：一屏有多少标多少 ✓（⚠ 一地金币常常几十个 ✓）；\n"
                       "⭐ **0 = 一个都不标** ⇒ 这一段那两个按钮会**灰着**（点了也没意义 ✓）；\n"
                       "⭐ 最大 **999** ✓。\n\n"
                       "⚠ 老项目没存过 ⇒ 用上面那个总的「每帧最多几个框」✓。")
        self.widgets["drop_max_peaks"][0].valueChanged.connect(
            lambda _v: self._refresh_drop_buttons())
        # ⭐⭐ **这一块自己的降采样**（用户 2026-10-04 ✓ 实测逼出来的 ✓）：见 `make_task` 里
        #   那段说明 —— 怪物那块存的 `downscale`（本项目 = 2 ✓）会把金币这种十几像素的图标
        #   **压糊** ⇒ 匹配全灭 ✓。默认 **1**（**不**继承总降采样 ✗：那正是坑 ✓）。
        self.field(_tform, "drop_downscale", "降采样", "int", 1, minimum=1, maximum=4,
                   tip="**这一块自己的降采样**（把画面缩小几倍再匹配 ✓ 数字越大越快、越糊）。\n\n"
                       "· 默认 **1**（= 原尺寸匹配 ✓）—— ⚠ **不会**跟着上面那个总的「降采样」\n"
                       "  走：那个常按**怪物**（约 100 像素）设的，金币图标才十几像素，\n"
                       "  缩一半就糊成一团 ⇒ **一个都匹配不上** ✓（用户实测：同 8 帧，\n"
                       "  `1` 能检出、`2` 是 0 框 ✗）；\n"
                       "· 真的嫌慢再往上调 ✓，但要盯着检出率 ✓；\n"
                       "· 调它只影响**掉落物这一块** ✓。")
        dv.addLayout(_tform)

        drow = QHBoxLayout()
        drow.setSpacing(6)
        self.btn_drop_add = QPushButton("添加…")
        self.btn_drop_add.setToolTip("按 **id 或名称** 搜索 WZ 里的掉落物，选中后加进列表 ✓")
        self.btn_drop_add.clicked.connect(self._drop_add)
        drow.addWidget(self.btn_drop_add)
        self.btn_drop_del = QPushButton("移出选中")
        self.btn_drop_del.clicked.connect(self._drop_del)
        drow.addWidget(self.btn_drop_del)
        self.lbl_drop_state = QLabel("—")
        self.lbl_drop_state.setStyleSheet("color:#80868b;")
        self.lbl_drop_state.setWordWrap(True)
        drow.addWidget(self.lbl_drop_state, 1)
        dv.addLayout(drow)

        brow = QHBoxLayout()
        brow.setSpacing(6)
        self.btn_drop_tpl = QPushButton("模板匹配标注掉落物")
        self.btn_drop_tpl.setStyleSheet(self._BTN)
        self.btn_drop_tpl.setToolTip(
            "用**掉落物图标**当模板做匹配（与「标注怪物」**同一条算法** ✓），\n"
            "写出去的类别号是 **class 2（drop）** ✓。")
        self.btn_drop_tpl.clicked.connect(lambda: self._emit_drop("template"))
        brow.addWidget(self.btn_drop_tpl)
        self.btn_drop_yolo = QPushButton("YOLO标注掉落物")
        self.btn_drop_yolo.setStyleSheet(self._BTN)
        self.btn_drop_yolo.setToolTip(
            "用上面选的「YOLO 权重」补一轮框（与「YOLO 标注（怪物）」**同一条逻辑** ✓），\n"
            "只合并 **class 2（drop）** 的预测 ✓。")
        self.btn_drop_yolo.clicked.connect(lambda: self._emit_drop("yolo"))
        brow.addWidget(self.btn_drop_yolo)
        brow.addStretch(1)
        dv.addLayout(brow)

        self.layout().addWidget(self.drop_box)
        self._refresh_drop_ui()

        # ---- ⭐⭐ 「标注宠物」（用户 2026-10-04 ✓ 方案 C：标的是**组合外观** ✓）----
        #   与掉落物那块**同一套做法**（勾选显隐 / 背包窗格 / 两个按钮 / 自己的匹配阈值 ✓）——
        #   ⚠ 唯一的区别在**模板**：这里不是"按 id 查库"✗ 而是**现算的组合外观** ✓
        #   （宠物 + 它戴的装备按 origin 叠起来 ✓ 见 `core/petlib.compose_pet` ✓）。

        self.pet_box = QWidget()
        pv = QVBoxLayout(self.pet_box)
        pv.setContentsMargins(0, 2, 0, 0)
        pv.setSpacing(4)

        plbl = title_label(
            "要标注的宠物",
            tip="**这个字段是干嘛的**：下面格子里列出「这次要标的宠物」——\n"
                "每格 = **一只宠物 + 它戴的装备**（组合外观 ✓）。\n\n"
                "· **鼠标指到格子上** ⇒ 弹出信息窗：名称 / 宠物 id / 装备 ids ✓；\n"
                "· **添加…** ⇒ 打开「选择要标注的宠物」：先挑宠物、再挑它戴的装备、\n"
                "  看着组合预览确认 ✓；\n"
                "· 至少要有一项，下面两个按钮才会亮 ✓。")
        plbl.setStyleSheet("color:#5f6368; font-weight:600;")
        self.lbl_pet_title = plbl
        pv.addWidget(plbl)

        self.grid_pet = IconGrid()
        # ⭐ 同掉落物那块：**按内容自适应**（用户 2026-10-04 ✓ 见 `IconGrid.fit_rows` ✓）
        self.grid_pet.fit_rows(min_rows=1, hard=True)
        # ⭐⭐ 双击格子 ⇒ 选模板帧（同掉落物那条 ✓ 任务 3 ✓）
        self.grid_pet.activated_id.connect(lambda _i: self._pick_entry_frames("pet"))      # 同掉落物那块（见上面那段说明 ✓）
        pv.addWidget(self.grid_pet)

        _pform = QFormLayout()
        _pform.setContentsMargins(0, 0, 0, 0)
        _pform.setSpacing(4)
        self.field(_pform, "pet_thresh", "匹配阈值", "float", 0.90,
                   minimum=0.0, maximum=1.0, step=0.01, decimals=2,
                   tip="**这一块自己的匹配阈值**（宠物组合外观做模板匹配时用 ✓）。\n\n"
                       "· 默认 = 上面那个总的「匹配阈值」的值 ✓（老项目一字不变 ✓）；\n"
                       "· 宠物的模板比金币大、但比怪小 ⇒ 和怪物常常不该一个数 ✓；\n"
                       "· 调它只影响**宠物这一块** ✓。")
        # ⭐⭐ 这一块自己的另外三个参数（用户 2026-10-05 ✓ 同掉落物那块 ✓
        #   ⚠ 老项目没这些键 ⇒ **回落总那个** ✓ 见 `make_task` ✓）
        self.field(_pform, "pet_min_distinct", "区分度", "float", 0.06,
                   minimum=0.0, maximum=1.0, decimals=3, step=0.01,
                   tip="同一帧里，同一个模板找第 2、3 个峰时的「去重」闸：\n"
                       "新峰的分数必须比已经收下的那个再高出这么多才算另一处。\n"
                       "  · 调小（→0）⇒ 稍亮一点就算另一只 ⇒ 同一只被反复框上（重叠框 ✗）\n"
                       "  · 调大 ⇒ 两只挤在一起时容易只框到一只\n\n"
                       "0 = 完全不设这道闸。⚠ 老项目 ⇒ 用上面那个总的「区分度」✓。")
        self.field(_pform, "pet_per_mob", "最大模板帧", "int", 20,
                   minimum=2, maximum=60,
                   tip="这一块每套组合外观最多用几帧当模板。\n"
                       "  · 帧数 ≤ 它 ⇒ 全用（组合外观默认做了 stand0/stand1/move/jump/hungry ✓）；\n"
                       "  · 调小会漏掉某些姿态 ⇒ 那些姿态的宠物就匹配不上 ✗\n\n"
                       "⚠ 老项目没存过 ⇒ 用上面那个总的「最大模板帧」✓。")
        self.field(_pform, "pet_max_peaks", "每帧最多几个框", "int", 4,
                   minimum=-1, maximum=999,
                   tip="同一个模板在一帧里最多写几个框（够了就停）。\n"
                       "  · 一屏同种宠物超过这个数 ⇒ 多出来的不标 ✗；调成 1 ⇒ 只标一个 ✗\n"
                       "⭐ **负数（-1）= 无限**：一屏有多少标多少 ✓；\n"
                       "⭐ **0 = 一个都不标** ⇒ 这一段那两个按钮会**灰着**（点了也没意义 ✓）；\n"
                       "⭐ 最大 **999** ✓。\n\n"
                       "⚠ 老项目没存过 ⇒ 用上面那个总的「每帧最多几个框」✓。")
        self.widgets["pet_max_peaks"][0].valueChanged.connect(
            lambda _v: self._refresh_pet_buttons())
        # ⭐⭐ **这一块自己的降采样**（用户 2026-10-04 ✓ 实测逼出来的 ✓）：宠物在画面上约
        #   70 像素（49 像素的图 × 标定 1.406 ✓）⇒ 怪物那块存的 `downscale=2` 会把它压糊
        #   ⇒ 0 检出 ✓（实测同 8 帧：`1` 能检出、`2` 是 0 框 ✗）。默认 **1** ✓
        #   （**不**继承总降采样 ✗ —— 那正是坑 ✓）。
        self.field(_pform, "pet_downscale", "降采样", "int", 1, minimum=1, maximum=4,
                   tip="**这一块自己的降采样**（把画面缩小几倍再匹配 ✓ 数字越大越快、越糊）。\n\n"
                       "· 默认 **1**（= 原尺寸匹配 ✓）—— ⚠ **不会**跟着上面那个总的「降采样」\n"
                       "  走：那个常按**怪物**（约 100 像素）设的，宠物/掉落物在画面上只有\n"
                       "  几十像素，缩一半就糊 ⇒ **一个都匹配不上** ✓（用户实测：同 8 帧，\n"
                       "  `1` 能检出、`2` 是 0 框 ✗）；\n"
                       "· 调它只影响**宠物这一块** ✓。")
        pv.addLayout(_pform)

        prow = QHBoxLayout()
        prow.setSpacing(6)
        self.btn_pet_add = QPushButton("添加…")
        self.btn_pet_add.setToolTip("打开「选择要标注的宠物」：**先挑宠物、再挑它戴的装备**，"
                                    "看着组合预览确认 ✓")
        self.btn_pet_add.clicked.connect(self._pet_add)
        prow.addWidget(self.btn_pet_add)
        self.btn_pet_del = QPushButton("移出选中")
        self.btn_pet_del.clicked.connect(self._pet_del)
        # ⭐⭐ **选中变化必须重算这个按钮**（用户 2026-10-04 ✓ 原话："现在选中已添加的宠物，
        #   移出选中按钮不能交互" ✗）。病根：`_refresh_pet_buttons` 里那条判据
        #   「**当前有选中项**才亮」✓ 只在「添加…/移出选中/忙闲」时才跑 ✗ ⇒ 你在格子里
        #   点中一只那一刻**没人重算** ⇒ 按钮一直灰着 ✓ 点了毫无反应 ✓（离屏复现：
        #   `current_id` 已经是对的 ✓ 而 `isEnabled()` 还是 False ✗）。
        #   ⚠ 接在这一句之后（按钮已经建好了 ✓）：`_refresh_pet_buttons` 会 setToolTip/setEnabled
        #     那几个控件 ⇒ 建完再连，避免半构建状态下被信号打到 ✗。
        self.grid_pet.itemSelectionChanged.connect(self._refresh_pet_buttons)
        prow.addWidget(self.btn_pet_del)
        self.lbl_pet_state = QLabel("—")
        self.lbl_pet_state.setStyleSheet("color:#80868b;")
        self.lbl_pet_state.setWordWrap(True)
        prow.addWidget(self.lbl_pet_state, 1)
        pv.addLayout(prow)

        pbrow = QHBoxLayout()
        pbrow.setSpacing(6)
        self.btn_pet_tpl = QPushButton("模板匹配标注宠物")
        self.btn_pet_tpl.setStyleSheet(self._BTN)
        self.btn_pet_tpl.setToolTip(
            "用**组合外观**当模板做匹配（与「标注怪物」**同一条算法** ✓），\n"
            "写出去的类别号是 **class 5（宠物）** ✓。")
        self.btn_pet_tpl.clicked.connect(lambda: self._emit_pet("template"))
        pbrow.addWidget(self.btn_pet_tpl)
        self.btn_pet_yolo = QPushButton("YOLO标注宠物")
        self.btn_pet_yolo.setStyleSheet(self._BTN)
        self.btn_pet_yolo.setToolTip(
            "用上面选的「YOLO 权重」补一轮框（同一条逻辑 ✓），只合并 **class 5（宠物）** 的预测 ✓。")
        self.btn_pet_yolo.clicked.connect(lambda: self._emit_pet("yolo"))
        pbrow.addWidget(self.btn_pet_yolo)
        pbrow.addStretch(1)
        pv.addLayout(pbrow)

        self.layout().addWidget(self.pet_box)
        self._refresh_pet_ui()

    def _emit_run(self, target):
        self._label_target = target
        if not self._confirm_relabel(target):
            return
        # 选帧：默认全选（重标场景就是全部重跑一遍）。只补没跑过的，用弹窗里的
        # 「只选未处理」—— 配合抽帧「取消勾选重抽前清空 = 追加素材」，新加的帧
        # 正好都在「未处理」里。
        ok, picked = self._pick_frames(target)
        if not ok:
            return                      # 用户取消
        self._only[target] = picked     # None = 全选（不筛）
        self.run_clicked.emit(self)

    def _pick_frames(self, target):
        """弹选帧窗。返回 (是否继续, 帧名列表 或 None=全选)。"""
        p = self.project
        if p is None:
            return True, None           # 没项目就不筛，交给后面的依赖检查去报错
        from gui.frame_picker import FramePickDialog
        dlg = FramePickDialog(self.window(), p, target)
        if dlg.exec_() != QDialog.Accepted:
            return False, None
        stems = dlg.selected_stems()
        if not stems:
            QMessageBox.information(self, "一帧都没选", "至少要选一帧，否则没什么可标注的。")
            return False, None
        if len(stems) == len(dlg.stems):
            return True, None           # 全选 → 不筛，和以前的行为完全一致
        return True, stems

    def _emit_yolo(self, target):
        """只跑模型这一遍：补框，**不重跑模板匹配**。

        目标由按下的那个按钮**显式给定**（不做"跟着上次点的主按钮变"——
        那种设计用户看不见作用对象）。
        补框是追加操作（只加"和已有框不重叠"的），不会覆盖模板匹配的框，
        所以不弹「重新标注」确认 —— 只让你挑一次帧（和主按钮同一套选帧弹窗）。
        """
        self._label_target = target
        self._yolo_only = True
        ok, picked = self._pick_frames(target)
        if not ok:
            self._yolo_only = False
            return
        self._only[target] = picked
        self.run_clicked.emit(self)

    # ---------------- 掉落物（class 2，用户 2026-10-04 ✓）----------------

    def _emit_drop(self, mode):
        """掉落物两个按钮的入口（`template` / `yolo` ✓）。

        ⭐⭐ **先弹选帧窗**（用户 2026-10-04 ✓ 原话："标注掉落物、宠物的提交按钮需要先开选帧
        弹窗！（参考标注怪物、标注玩家、YOLO标注怪物、YOLO标注玩家）"）—— 与那四个按钮
        **完全同一套**（见 `_emit_run` / `_emit_yolo` ✓）：
          · `template`（模板匹配标注）= 这一类会**重写**那些帧上的自动框 ⇒ 先 `_confirm_relabel`
            （标过就问一句 ✓）再选帧 ✓；
          · `yolo`（YOLO标注）= **只补框**（不覆盖已有的 ✓ 追加性质）⇒ 只选帧、不确认 ✓
            —— 与「YOLO标注怪物 / YOLO标注玩家」严格一致 ✓。
        ⚠ 别再"直接整份画面跑一遍"✗ —— 那等于把"只补没跑过的"这一整套（弹窗里的
          `只选未处理` ✓ 配合台账 `labelio.processed_set(dir, "drop")` ✓）全跳过了 ✓。
        """
        self._label_target = "drop"
        self._drop_mode = str(mode or "template").strip().lower()
        self._yolo_only = False          # ⚠ 清掉上一次「只跑 YOLO」的残留（那是 mob/player 的 ✓）
        if self._drop_mode != "yolo" and not self._confirm_relabel("drop"):
            return                       # 用户选了「否」
        ok, picked = self._pick_frames("drop")
        if not ok:
            return                       # 用户取消选帧 ⇒ 什么都不跑 ✓
        self._only["drop"] = picked      # None = 全选（不筛）✓；否则只跑勾中的那些帧 ✓
        self.run_clicked.emit(self)

    def _refresh_drop_ui(self):
        """勾选框 → 显示/隐藏那一整块（**藏起来**才是"出现" ✓ 照 `player_panel` 那个做法 ✓）。"""
        if not getattr(self, "drop_box", None):
            return
        self.drop_box.setVisible(bool(self.ck_drop.isChecked()))
        self._refresh_drop_buttons()

    def _refresh_drop_list(self):
        """重填背包窗格（名字/类别/几帧由 `IconGrid` 自己去查清单 ✓ 见 `normalize_entry` ✓）。"""
        self.grid_drop.set_entries(self._drops)
        self._refresh_drop_state()
        self._refresh_drop_buttons()

    def _refresh_drop_state(self):
        """列表旁边那句**现状**：图库在不在、缺哪一步（用户 2026-10-04 ✓ 不假报空 ✗）。"""
        from core import wzexport
        ok, why = wzexport.drop_library_state()
        n = len(self._drops)
        head = ("已选 %d 项" % n) if n else "还没有选掉落物"
        self.lbl_drop_state.setText(head if ok else (head + " · " + why.splitlines()[0]))
        self.lbl_drop_state.setToolTip(why)

    def _peaks_of(self, key):
        """那一段的「每帧最多几个框」当前值 ✓（**`0` ⇒ 这一段一个框都不写** ⇒ 那对按钮要灰 ✓）。

        ⚠ 控件还没建好 / 取不到 ⇒ 回 `None`（**不拦** ✓ —— 那时候按钮也还没建 ✓）。
        ⚠ 四条链各一份键名：`max_peaks`（怪物）/`player_max_peaks`/`drop_max_peaks`/`pet_max_peaks` ✓。
        """
        w = (getattr(self, "widgets", None) or {}).get(key)
        try:
            return int(w[0].value())
        except Exception:                       # noqa: BLE001 —— 没建 / 类型不对 ⇒ 不拦 ✓
            return None

    def _refresh_drop_buttons(self):
        """⭐ 用户 2026-10-04 §2：**列表里有项目**时两个按钮才可交互；灰着要说清为什么 ✓。

        （"1个以上"按"**至少 1 项**"实现 —— 只标一种掉落物是常见用法 ✓；
          若你要的是"必须 ≥2 项"，说一声改一行即可 ✓。）
        ⭐ 2026-10-05 追加一条同款的灰：**「每帧最多几个框」= 0 ⇒ 这一趟一个框都不写** ✓
          （用户原话："填0时启动标注的按钮都置灰不可交互" ✓）—— 那种情况点了也是白跑一趟 ✓。
        """
        busy = bool(getattr(self, "_busy", False))
        n = len(self._drops)
        peaks = self._peaks_of("drop_max_peaks")
        on = n > 0 and not busy and peaks != 0
        tips = {
            "template": ("用**掉落物图标**当模板做匹配（与「标注怪物」同一条算法 ✓），"
                         "写出去的类别号是 **class 2（drop）** ✓。"),
            "yolo": ("用上面选的「YOLO 权重」补一轮框（与「YOLO 标注（怪物）」同一条逻辑 ✓），"
                     "只合并 **class 2（drop）** 的预测 ✓。"),
        }
        if busy:
            why = "\n\n（现在灰着：正在跑别的任务 —— 等它跑完 ✓）"
        elif peaks == 0:
            why = ("\n\n（现在灰着：「每帧最多几个框」= 0 ⇒ 这一趟一个框都不写 ✓"
                   " 想标就填回 ≥1，或填 -1 = 无限 ✓）")
        elif n == 0:
            why = "\n\n（现在灰着：列表里还没有掉落物 —— 先点「添加…」至少加一项 ✓）"
        else:
            why = "\n\n已选 %d 项 ✓" % n
        self.btn_drop_tpl.setEnabled(on)
        self.btn_drop_yolo.setEnabled(on)
        self.btn_drop_tpl.setToolTip(tips["template"] + why)
        self.btn_drop_yolo.setToolTip(tips["yolo"] + why)

    # ---------------- 怪物段 / 玩家段（用户 2026-10-05）----------------

    def _mob_rows(self):
        """怪物那块的格子条目 ⇒ icon = 该怪的 **stand / fly 第一帧**（用户点名 ✓）。

        ⚠ `gui/icon_grid.normalize_entry` **本来就吃调用方给的 `icon` 路径** ✓（原样带着 ✓）
          ⇒ 不用改它 ✓；`has_img` / `frames_disk` 也一并给 ✓（信息窗与行数才准 ✓）。
        """
        if self.project is None:
            return []
        names = wzexport.build_mob_name_map()
        out = []
        for mid in (self.project.get("mobs") or []):
            mid = str(mid)
            d = wzexport.sprite_dir_path() / mid
            pngs = ([p for p in sorted(d.glob("*.png")) if p.stem != "meta"]
                    if d.is_dir() else [])
            # ⭐⭐ 按**帧号**取第一帧（用户 2026-10-05 ✓ 报"选择标注怪物的图标错了"）——
            #   原来按**文件名**排序取第一个 ✗ ⇒ `stand_10` 排在 `stand_2` 前面 ✗
            #   ⇒ 拿到的是动画中段那一张（姿势很怪 ✓ 这就是"图标错了"）。
            #   `wzexport.mob_action_frames` 内部用 `_frame_no` 排序 ✓ 而且 stand→fly
            #   有优先级 ✓、都不匹配时回落到"全部 png（排除 meta）"✓ —— 正是这里要的口径 ✓。
            _act, _fs = wzexport.mob_action_frames(d, ("stand", "fly"))
            icon = str(_fs[0]) if _fs else None
            e = {"id": mid, "name": names.get(mid) or mid,
                 "frames": len(pngs), "frames_disk": len(pngs), "has_img": bool(icon)}
            if icon:
                e["icon"] = icon
            out.append(e)
        return out

    def _player_rows(self):
        """玩家那块的格子条目（**单选** ⇒ 0 或 1 个 ✓）⇒ icon = 该角色的 **stand 第一帧** ✓。"""
        if self.project is None:
            return []
        pid = (self.project.get("player_id") or "").strip()
        if not pid:
            return []
        d = Path("datasets/sprites/player") / pid
        pngs = sorted(d.glob("*.png")) if d.is_dir() else []
        # ⭐ 帧名形如「0 10-43-39-…」⇒ 按**开头的帧号**排 ✓（字符串排会把 10 排到 2 前面 ✗）
        # ⚠⚠ 这里**不许用 `re`** ✗ —— `cards.py` 顶层没有 `import re`（2026-10-05 就栽在这：
        #   玩家段直接 `NameError` ⇒ 「添加…」怎么点都加不上 ✓ 用户报的正是这个 ✓）。
        #   纯字符串解析，零依赖 ✓。
        def _fno(q):
            head = q.stem.split(" ", 1)[0].strip()
            try:
                return (0, int(head), q.stem)        # 数字开头 ⇒ 按帧号 ✓
            except ValueError:
                return (1, 0, q.stem)                # 不是数字 ⇒ 排后面（组内按名字 ✓）

        _pngs = sorted(pngs, key=_fno)
        icon = next((p for p in _pngs if "stand" in p.stem), None) or (_pngs[0] if _pngs else None)
        e = {"id": pid, "name": pid, "frames": len(pngs), "frames_disk": len(pngs),
             "has_img": icon is not None}
        if icon is not None:
            e["icon"] = str(icon)
        return [e]

    def _build_player_section(self):
        """「标注玩家」那一段 ✓（用户 2026-10-05 ✓）：背包窗格 + 添加 / 移出选中 + 两个按钮 ✓。"""
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 6, 0, 0)
        v.setSpacing(4)
        v.addWidget(title_label(
            "标注玩家",
            tip="「标注玩家」这一段：选一个角色，用他那套外观模板在画面里找自己（class 0）✓。\n"
                "· 角色就是 ① 里那个 `player_id` ✓ —— 这里换的就是它 ✓；\n"
                "· 模板来自 `datasets/sprites/player/<角色>/`（多个动作 × 朝向 ✓）；\n"
                "· 两个按钮：模板匹配标注玩家 / YOLO标注玩家 ✓。"))
        v.addWidget(title_label(
            "要标注的玩家",
            tip="格子里是**要标的那个角色**（单选 ✓）。\n"
                "· 「添加…」从角色模板库（`datasets/sprites/player/`）里挑一个 ✓\n"
                "  —— 导出新角色后，列表要重开工作台或重扫目录才会出现 ✓；\n"
                "· 「移出选中」= 清掉角色 ⇒ 标注玩家那两项会提示「没有指定玩家」✓。"))
        self.grid_player = IconGrid()
        self.grid_player.fit_rows(min_rows=1, hard=True)     # 一行的格子（用户点名 ✓）
        v.addWidget(self.grid_player)
        # ⭐⭐ **这一段自己的 5 个匹配参数**（用户 2026-10-05 ✓ 原话："4类按照统一的格式整理"）。
        #   原来它们摆在卡片**顶部**那份参数表单里 ✗（掉落物 / 宠物那两段**早就**在各自段里了 ✓
        #   用的是 `__init__` 里 `_tform` / `_pform` 那个写法 ✓）⇒ 这轮照同款搬进本段 ✓
        #   （`QFormLayout` 直接挂进段的竖排布局 ✓）。
        #   ⚠⚠ **键一个字都没改** ✗ —— `load_from_project` / `sync` / `make_task` / 四套用例
        #   全按 key 认 ✓；这一趟是**纯版式** ✓ 值、默认值、tip 全部原样 ✓。
        _plform = QFormLayout()
        _plform.setSpacing(4)
        _plform.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        _plform.addRow("", self._section("玩家", "玩家（class 0）单独一套阈值与尺度 ✓"))
        # 玩家 scale 由 ③ 标定尺度自动测，这里只读展示，不让手填
        self.lbl_player_scale = QLabel("—（先跑 ③ 标定）")
        self.lbl_player_scale.setStyleSheet("color:#5f6368;")
        _plform.addRow("玩家 scale", self.lbl_player_scale)

        self.field(_plform, "player_thresh", "玩家阈值", "float", 0.78,
                   minimum=0.0, maximum=1.0, decimals=2, step=0.01,
                   tip="只有「标注玩家」（class 0）这一条走它 —— 怪物 / 掉落物 / 宠物\n"
                       "各用上面各自那一节里的阈值。\n"
                       "  · 调低 ⇒ 画面里长得像玩家的固定东西也会被当成玩家（实测：地图上垂下来的\n"
                       "    一根绳子能到 0.851，比真玩家还高 ⇒ 阈值压到它之上才拦得住）\n"
                       "  · 调高 ⇒ 玩家走路 / 施法时被特效或怪挡住的那些帧会漏\n\n"
                       "0.78 比怪物那个 0.90 松（玩家朝向 / 动作帧多，太严会漏）。\n"
                       "⚠ 上面「玩家 scale」是 ③ 标定自动测出来的，别手填（这一行只读）。")
        # ⭐⭐ **玩家这一段自己的另外四个参数**（用户 2026-10-05 ✓ 原话："4类按照统一的格式整理…
        #   匹配阈值 / 区分度 / 最大模板帧 / 每帧最多几个框 / 降采样" + "缺功能的就补功能"）——
        #   它们的**能力**是本轮先在 `perception/player_locator.py` 里补出来的 ✓
        #   （原来玩家这条链只有 阈值 / 尺度 ✗ 见 `tools/detect_player.py` ✓）。
        #   ⚠⚠ 这几个的默认值**故意与怪物不同** ✗：
        #     · 「区分度」对玩家是**可信度闸**（最佳 vs 次佳分差 ✓）而不是去重闸 ✓
        #       —— 玩家是唯一目标，而模板是一堆动作帧（彼此极像 ✓）⇒ 给 0.06 会把绝大多数帧
        #       判成"存疑" ✗ ⇒ 默认 **0**（不看这项 ✓）；调大 = 更严 ✓。
        #     · 「最大模板帧」0 = **不截断**（玩家模板本来就要多动作 ✓ 别默认砍到 20 ✗）；
        #     · 「每帧最多几个框」1 = 只取最好那个（单一目标 ✓）。
        #   ⚠ 立项目没有这些键 ⇒ 走上面这些默认 ✓（老项目一字不变 ✓）。
        self.field(_plform, "player_min_distinct", "区分度", "float", 0.0,
                   minimum=0.0, maximum=1.0, decimals=3, step=0.01,
                   tip="玩家这条链的**可信度闸**（不是去重闸 ✗）：最佳位置与次佳位置的\n"
                       "分数差小于它 ⇒ 这次定位**判存疑**（当没定位到 ✓）。\n"
                       "  · 0（默认）= 不看这项 ✓ —— 玩家模板是一堆动作帧、彼此极像，\n"
                       "    给一个大于 0 的值会把**绝大多数帧**判成存疑 ✗；\n"
                       "  · 调大 = 更严 ✓（画面里有长得像玩家的固定东西时才值得开 ✓）。")
        self.field(_plform, "player_per_mob", "最大模板帧", "int", 0,
                   minimum=0, maximum=200,
                   tip="玩家模板最多用几帧（**0 = 不截断**，默认 ✓）。\n"
                       "  · 玩家模板来自 `datasets/sprites/player/<角色>/`（多个动作 × 朝向 ✓）；\n"
                       "  · 截断时是**在排序好的名单上等距抽** ✓ ⇒ 每个动作都能留到代表帧 ✓；\n"
                       "  · 调小 ⇒ 更快，但某些动作 / 朝向可能匹配不上 ✗。")
        self.field(_plform, "player_max_peaks", "每帧最多几个框", "int", 1,
                   minimum=-1, maximum=999,
                   tip="每个模板在一帧里最多取几个峰（**1 = 只取最好那个**，默认 ✓ 单一目标够用）。\n"
                       "  · 调大 ⇒ 同一个模板的第二好位置也会进候选 ✓（想同时盯多个位置时才用）；\n"
                       "  · ⚠ 它会影响「区分度」那道闸看得见什么 ✓（峰多了才有「次佳」可比 ✓）。\n"
                       "⭐ **负数（-1）= 无限**：有几个峰取几个 ✓（同一位置会被抹掉、不会重复取 ✓）；\n"
                       "⭐ **0 = 一个都不标** ⇒ 这一段那两个按钮会**灰着**（点了也没意义 ✓）；\n"
                       "⭐ 最大 **999** ✓。")
        self.widgets["player_max_peaks"][0].valueChanged.connect(
            lambda _v: self._refresh_player_buttons())
        self.field(_plform, "player_downscale", "降采样", "int", 1,
                   minimum=1, maximum=4,
                   tip="匹配之前把画面缩到 1/N（**1 = 不缩**，默认 ✓）。\n"
                       "  · 玩家模板通常几十个 ⇒ 这是实时帧率的主要瓶颈，n 能快约 N² 倍 ✓\n"
                       "  · 代价：小目标 / 细节被压糊 ⇒ 定位变飘 ✓ 别为了快开到 3~4 ✗\n\n"
                       "⚠ 与「标注玩家」那一趟用的是**同一个**数 ✓（`detect_player` ✓）。")

        # ---- ⭐⭐ 「标注掉落物」（用户 2026-10-04 ✓ 原话："自动标注卡片最下面增加勾选参数
        #      「标注掉落物」，勾选后下面增加列表区，可以增删项目（根据 id、名称搜索 wz 中的图），
        #      再下面就是按钮「模板匹配标注」、「YOLO标注」"）----
        #   勾选 ⇒ 卡片**最下面**出现那一块（列表 + 两个按钮 ✓ 见 `__init__` 里 `drop_box` ✓）。
        #   ⚠ 类别号用类别表里的 **class 2 = drop**（`perception/classes.py` ✓ 早就在表里 ✓）。
        v.addLayout(_plform)
        # ⭐⭐ **双击某个角色 ⇒ 选模板帧**（用户 2026-10-05 ✓「四块窗格各接一行双击选模板帧」）——
        #   与掉落物 / 宠物那两块**同一行** ✓（`IconGrid.activated_id` = 双击或回车都会发 ✓）。
        #   存进 `label.frames_sel` 的键是 `player:<角色id>` ✓ —— 与管线那把钥匙**完全一致** ✓
        #   （`detect_player.py:164` 就是摘 `player:<pid>` ✓）。
        self.grid_player.activated_id.connect(
            lambda _i: self._pick_entry_frames("player"))
        row = QHBoxLayout()
        row.setSpacing(6)
        self.btn_player_add = QPushButton("添加…")
        self.btn_player_add.setToolTip("从角色模板库里挑一个角色 ✓（等价于 ① 里那个「角色」下拉 ✓）")
        self.btn_player_add.clicked.connect(self._player_add)
        row.addWidget(self.btn_player_add)
        self.btn_player_del = QPushButton("移出选中")
        self.btn_player_del.setToolTip("清掉角色（清掉之后「标注玩家」会提示「没有指定玩家」✓）")
        self.btn_player_del.clicked.connect(self._player_del)
        row.addWidget(self.btn_player_del)
        self.lbl_player_state = QLabel("—")
        self.lbl_player_state.setStyleSheet("color:#80868b;")
        self.lbl_player_state.setWordWrap(True)
        row.addWidget(self.lbl_player_state, 1)
        v.addLayout(row)
        brow = QHBoxLayout()
        brow.setSpacing(6)
        self.btn_player_tpl = QPushButton("模板匹配标注玩家")
        self.btn_player_tpl.setStyleSheet(self._BTN)
        self.btn_player_tpl.setToolTip(
            "用**角色外观模板**做匹配（玩家不走训练 ✓）写出去的类别号是 class 0（玩家）✓。\n"
            "点了会先问「这一趟跑哪些画面帧」✓。")
        self.btn_player_tpl.clicked.connect(lambda: self._emit_run("player"))
        brow.addWidget(self.btn_player_tpl)
        self.btn_player_yolo = QPushButton("YOLO标注玩家")
        self.btn_player_yolo.setStyleSheet(self._BTN)
        self.btn_player_yolo.setToolTip(
            "用上面选的「YOLO 权重」补一轮框（不重跑模板匹配 ✓），只补 class 0 ✓。\n"
            "⚠ 模型所属角色必须与当前角色一致（不同角色会被误标成当前角色 ✗）。")
        self.btn_player_yolo.clicked.connect(lambda: self._emit_yolo("player"))
        brow.addWidget(self.btn_player_yolo)
        brow.addStretch(1)
        v.addLayout(brow)
        self.player_box = box
        self.layout().addWidget(box)
        self._refresh_player_list()

    def _build_mob_section(self):
        """「标注怪物」那一段 ✓（同玩家段 ✓）：背包窗格 + 添加 / 移出选中 + 两个按钮 ✓。"""
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 6, 0, 0)
        v.setSpacing(4)
        v.addWidget(title_label(
            "标注怪物",
            tip="「标注怪物」这一段：挑这张图要识别的怪，用它们的精灵做模板匹配（class 1）✓。\n"
                "· 怪列表**跟着地图走**（① 选完图就有一份默认的 ✓ 这里可以增删 ✓）；\n"
                "· 两个按钮：模板匹配标注怪物 / YOLO标注怪物 ✓。"))
        v.addWidget(title_label(
            "要标注的怪物",
            tip="格子里是**这张图要识别的怪**（icon 用它的 stand / fly 第一帧 ✓）。\n"
                "· 「添加…」按 id / 名称从精灵库搜（`WzProbe dump-mob` 导出的那份 ✓）；\n"
                "· 「移出选中」把选中的怪拿掉 ✓（拿空了 ⇒ 上面两个按钮会灰着并说清原因 ✓）。"))
        self.grid_mob = IconGrid()
        self.grid_mob.fit_rows(min_rows=1, hard=True)        # 一行的格子
        v.addWidget(self.grid_mob)
        # ⭐⭐ **这一段自己的 5 个匹配参数**（用户 2026-10-05 ✓ 原话："4类按照统一的格式整理"）。
        #   原来它们摆在卡片**顶部**那份参数表单里 ✗（掉落物 / 宠物那两段**早就**在各自段里了 ✓
        #   用的是 `__init__` 里 `_tform` / `_pform` 那个写法 ✓）⇒ 这轮照同款搬进本段 ✓
        #   （`QFormLayout` 直接挂进段的竖排布局 ✓）。
        #   ⚠⚠ **键一个字都没改** ✗ —— `load_from_project` / `sync` / `make_task` / 四套用例
        #   全按 key 认 ✓；这一趟是**纯版式** ✓ 值、默认值、tip 全部原样 ✓。
        _mform = QFormLayout()
        _mform.setSpacing(4)
        _mform.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        _mform.addRow("", self._section(
            "模板匹配参数", "「标注怪物 / 标注玩家 / 掉落物-模板匹配」用的匹配参数 ✓"))
        self.field(_mform, "thresh", "匹配阈值", "float", 0.90,
                   minimum=0.0, maximum=1.0, decimals=2, step=0.01,
                   tip="模板匹配的分数线（0~1，越大越挑剔）：画面某处与模板的相似度 ≥ 它才写成框。\n"
                       "  · 0 = 全收（画面每处都框上，等于没标）；1 = 只收几乎一模一样的\n"
                       "  · 调低 ⇒ 检出更多，假框也更多（背景纹理、别的怪长得像都会被框上）\n"
                       "  · 调高 ⇒ 干净，但略微变形 / 被技能特效盖住的怪会漏\n\n"
                       "0.90 是绝大多数怪能用的值。掉落物、宠物那两块各有自己的阈值（见下面两节），\n"
                       "「标注玩家」用的是单独的「玩家阈值」。")
        self.field(_mform, "min_distinct", "区分度", "float", 0.06,
                   minimum=0.0, maximum=1.0, decimals=3, step=0.01,
                   tip="同一帧里，同一个模板找第 2、3 个峰时的「去重」闸：\n"
                       "新峰的分数必须比已经收下的那个再高出这么多才算另一处。\n"
                       "  · 调小（→0）⇒ 稍亮一点就算另一只 ⇒ 同一只怪被反复框上（重叠框 ✗）\n"
                       "  · 调大 ⇒ 只有明显更亮的第二处才算另一只 ⇒ 两只怪挤在一起时容易只框到一只\n\n"
                       "它是相对分数（不是像素）。0 = 完全不设这道闸（重叠框会变多）。")
        self.field(_mform, "per_mob", "最大模板帧", "int", 20, minimum=2, maximum=60)
        self.tip("per_mob", 
            "每只怪的模板帧**上限**（非死亡动画的帧都会用上）。\n"
            "  · 非死亡帧数 ≤ 这个数 → 全用，不截断（绝大多数怪都属此类）\n"
            "  · 超过时才按动作轮询分配，保证每个动作都有代表帧\n\n"
            "死亡动画（die）始终排除 —— 怪临死时的样子和活着时差异太大，\n"
            "拿它当模板会把「正在消失的怪」也框出来。\n\n"
            "注意别调太小：设成 6 时曾把绿水灵 stand 的第三个相位挤掉，\n"
            "结果那个相位的怪从未被标注，训练集里一个样本都没有。")
        self.field(_mform, "max_peaks", "每帧最多几个框", "int", 4, minimum=-1, maximum=999,
                   tip="同一个模板在一帧里最多写几个框（够了就停）。\n"
                       "  · 一屏同种怪超过这个数 ⇒ 多出来的不标（例：一地金币、一群绿水灵 ✗）\n"
                       "  · 调成 1 ⇒ 一屏只标一只（那是漏标 ✗）；调大基本没代价（只多算几处）\n"
                       "⭐ **负数（-1）= 无限**：一屏有多少就标多少 ✓（后台另有防跑飞的安全上限 ✓）；\n"
                       "⭐ **0 = 一个都不标** ⇒ 这一段那两个标注按钮会**灰着**（点了也没意义 ✓）；\n"
                       "⭐ 最大 **999** ✓（以前封顶 20 ✗ —— 一屏几十只怪时那明显不够 ✓）。\n\n"
                       "⚠ 每收下一个峰，都会把它周围抹掉（防同一只怪被反复计入）⇒ 这一项只管\n"
                       "「不同位置」的那几只，重叠的怪本来就不会重复计数。")
        # ⭐ 值一变就重算按钮（用户 2026-10-05 ✓：填 0 ⇒ 那两个按钮当场灰 ✓ 见 `_refresh_mob_buttons`）
        self.widgets["max_peaks"][0].valueChanged.connect(
            lambda _v: self._refresh_mob_buttons())
        self.field(_mform, "downscale", "降采样", "int", 1, minimum=1, maximum=4,
                   tip="匹配之前，画面和模板都先缩到 1/N 再做（1 = 不缩，默认）。\n"
                       "  · 越大越快（约 N² 倍）——大画面下嫌慢时才动它\n"
                       "  · 代价：小目标被压糊 ⇒ 直接标不出来。实测：怪物这块设 2 还能用，\n"
                       "    十几像素的金币图标设 2 就是 0 个框（所以掉落物/宠物各有独立的降采样，默认 1）\n\n"
                       "⚠ 模板最短边被压到 8 像素以下时，那个模板会被整帧跳过（匹配里的下限）。")
        # ⭐⭐ **框到可见部分**（用户 2026-10-05 ✓ 原话："现在模板匹配的自动标注框不管实际漏出目标的
        #   部分有多少，都用完整贴图大小的框，这个能自适应识别出的框大小吗？" ✓ 实现见
        #   `tools/detect_mobs._visible_box` ✓）—— **默认关** ✓（这是改**标注语义** ✗：
        #   框住整只（含被挡部分）是 COCO/YOLO 的通行约定 ✓ 老行为不算错 ✓；开了之后
        #   **同一帧的框会变小** ⇒ 要重跑这一趟标注 ✓）。
        self.field(_mform, "visible", "框到可见部分", "bool", False,
                   tip="**只框看得见的那部分**（被别的怪/地形/UI 挡掉的像素不进框 ✓）。\n"
                       "  · ⭐ 关（默认）：框 = **完整贴图尺寸** —— 这是 COCO/YOLO 的通行约定 ✓\n"
                       "    （人手工标注也这么标 ✓），老行为一字不变 ✓；\n"
                       "  · 开：按峰值那处的**像素证据**收框（灰度 + 边缘都得对得上 ✓）⇒\n"
                       "    适度遮挡（被挡掉一两成）时框会明显更贴合 ✓。\n\n"
                       "⚠ 三件要知道的：\n"
                       "  ① 这是改**标注语义** ⇒ 同一帧的框会变小 ⇒ 已有数据集与指标都会变 ✓\n"
                       "     （开了就**重跑这一趟标注** ✓）；\n"
                       "  ② **被挡掉太多（≥半边）本来就检不到** —— 带掩码的分数会掉到阈值下 ✓\n"
                       "     这功能救的是「分够、但框包满」那种适度遮挡 ✓ 两件事别混 ✗；\n"
                       "  ③ 判据靠纹理/边缘 ⇒ **大片纯色的精灵判不了**（那里一律回退完整尺寸 ✓\n"
                       "     故意保守 ✓）。\n\n"
                       "⚠ 玩家那一段走的是另一套定位（`perception/player_locator.py`）——\n"
                       "   ⭐ **2026-10-06 起它也吃这一格了** ✓（判据是**同一份**实现：\n"
                       "   `perception/visible_box.py` ✓ ⇒ 四段现在**一致** ✓）。\n"
                       "   ⚠ 它只在**标注**那条链生效（实时定位那条不传 ✓ 看的是「人在哪」、\n"
                       "   不是「露出来多少」✓）。")
        self.field(_mform, "visible_keep", "最小保留比例", "float", 0.2,
                   minimum=0.05, maximum=0.9, step=0.05, decimals=2,
                   tip="收框的**下限**：相符的证据不足「模板边缘像素 × 它」时 ⇒ **不给收** ✓\n"
                       "（宁可留个完整框 ✗ 也不许吐出一个荒谬的小框 ✓）。\n"
                       "  · 调大（→0.5）⇒ 更保守、更少收；调小（→0.1）⇒ 更愿意收（也更可能收歪）。\n\n"
                       "⚠ 还有两道保守护栏（不在这里调 ✓）：缩得不足 15% 就当没缩（防抖动 ✓）、\n"
                       "框任一边小于 6 像素就当没缩 ✓ —— 都是为了「别把没被挡的框啃小」✗。")

        # ---- 玩家标注参数（模板匹配，class 0）----
        v.addLayout(_mform)
        # ⭐⭐ 同上（怪物那块）：双击 ⇒ 选模板帧 ✓ 键 = `mob:<怪id>` ✓
        #   （`detect_mobs.py:655` 那句 `split(":", 1)[1]` **就是**剥这个前缀 ✓）。
        self.grid_mob.activated_id.connect(
            lambda _i: self._pick_entry_frames("mob"))
        row = QHBoxLayout()
        row.setSpacing(6)
        self.btn_mob_add = QPushButton("添加…")
        self.btn_mob_add.setToolTip("按 id / 名称从精灵库里挑要识别的怪（可多选 ✓）")
        self.btn_mob_add.clicked.connect(self._mob_add)
        row.addWidget(self.btn_mob_add)
        self.btn_mob_del = QPushButton("移出选中")
        self.btn_mob_del.setToolTip("把格子里选中的那只怪移出列表 ✓")
        self.btn_mob_del.clicked.connect(self._mob_del)
        row.addWidget(self.btn_mob_del)
        self.lbl_mob_state = QLabel("—")
        self.lbl_mob_state.setStyleSheet("color:#80868b;")
        self.lbl_mob_state.setWordWrap(True)
        row.addWidget(self.lbl_mob_state, 1)
        v.addLayout(row)
        brow = QHBoxLayout()
        brow.setSpacing(6)
        self.btn_mob_tpl = QPushButton("模板匹配标注怪物")
        self.btn_mob_tpl.setStyleSheet(self._BTN)
        self.btn_mob_tpl.setToolTip(
            "用**怪物精灵**当模板做匹配 ✓ 写出去的类别号是 class 1（怪物）✓。\n"
            "点了会先问「这一趟跑哪些画面帧」✓（可只跑没标过的 ✓）。")
        self.btn_mob_tpl.clicked.connect(lambda: self._emit_run("mob"))
        brow.addWidget(self.btn_mob_tpl)
        self.btn_mob_yolo = QPushButton("YOLO标注怪物")
        self.btn_mob_yolo.setStyleSheet(self._BTN)
        self.btn_mob_yolo.setToolTip("用上面选的「YOLO 权重」补一轮框（不重跑模板匹配 ✓），"
                                     "只补 class 1（怪物）✓。")
        self.btn_mob_yolo.clicked.connect(lambda: self._emit_yolo("mob"))
        brow.addWidget(self.btn_mob_yolo)
        brow.addStretch(1)
        v.addLayout(brow)
        self.mob_box = box
        self.layout().addWidget(box)
        self._refresh_mob_list()

    def _mob_add(self):
        """「添加…」⇒ 精灵库选怪 ⇒ 写回 `mobs` / `mob_names` / `mobs_cleared` ✓。

        ⚠ 与卡片 1（以及 `wzexport.apply_map_choice`）**同一份账** ✓ —— 那三个键是"这张图要
          识别哪些怪"的唯一存处 ✓（`mobs_cleared` 记"用户清空过"，换图时由写口作废 ✓）。
        """
        if self.project is None:
            return
        from gui.mob_picker import MobPickDialog
        dlg = MobPickDialog(self.project.get("mobs") or [], self)
        if not dlg.exec_():
            return
        self._write_mobs([str(m) for m in (dlg.result_mobs() or [])])

    def _mob_del(self):
        """「移出选中」⇒ 把那**一只**怪从列表里拿掉 ✓（其余照旧 ✓）。"""
        if self.project is None:
            return
        mid = str(self.grid_mob.current_id() or "")
        if not mid:
            return
        self._write_mobs([str(m) for m in (self.project.get("mobs") or [])
                          if str(m) != mid])

    def _write_mobs(self, mobs):
        """把怪列表写回项目（**唯一口径** ✓ `mobs` / `mob_names` / `mobs_cleared` 三件一起 ✓）。"""
        names = wzexport.build_mob_name_map()
        self.project.set("mobs", list(mobs))
        self.project.set("mob_names", [names.get(m) or m for m in mobs])
        self.project.set("mobs_cleared", not mobs)
        self.project.save()
        self._refresh_mob_list()

    def _player_add(self):
        """「添加…」⇒ 从角色模板库挑一个 ⇒ 写回 `player_id` ✓。"""
        if self.project is None:
            return
        from gui.player_picker import PlayerPickDialog
        dlg = PlayerPickDialog(self.project.get("player_id") or "", self)
        if not dlg.exec_():
            return
        pid = dlg.result_id()
        if not pid:
            return
        self.project.set("player_id", pid)
        self.project.save()
        self._refresh_player_list()

    def _player_del(self):
        """「移出选中」⇒ 清掉角色 ✓（之后「标注玩家」会提示"没有指定玩家" ✓ 不静默 ✗）。"""
        if self.project is None:
            return
        self.project.set("player_id", "")
        self.project.save()
        self._refresh_player_list()

    def _refresh_mob_list(self):
        if getattr(self, "grid_mob", None) is None:
            return
        self.grid_mob.set_entries(self._mob_rows())
        self._refresh_mob_buttons()

    def _refresh_mob_buttons(self):
        """按钮的亮/灰 + 那行"现状"（照掉落物那块的做法 ✓ 灰着要说清为什么 ✓）。"""
        if getattr(self, "grid_mob", None) is None:
            return
        busy = bool(getattr(self, "_busy", False))
        n = len(self.grid_mob.entries())
        # ⭐ 「每帧最多几个框」= 0 ⇒ 一个框都不写 ⇒ 这一对按钮灰着（用户 2026-10-05 ✓）
        peaks = self._peaks_of("max_peaks")
        on = n > 0 and not busy and peaks != 0
        why = ("\n\n（现在灰着：正在跑别的任务 ✓）" if busy else
               ("\n\n（现在灰着：「每帧最多几个框」= 0 ⇒ 这一趟一个框都不写 ✓"
                " 想标就填回 ≥1，或填 -1 = 无限 ✓）" if peaks == 0 else
                ("\n\n（现在灰着：还没有选怪 —— 先点「添加…」✓）" if not n
                 else "\n\n已选 %d 种 ✓" % n)))
        for btn in (getattr(self, "btn_mob_tpl", None), getattr(self, "btn_mob_yolo", None)):
            if btn is not None:
                btn.setEnabled(on)
                btn.setToolTip(btn.toolTip().split("\n\n（现在灰着")[0].split("\n\n已选")[0] + why)
        self.btn_mob_add.setEnabled(not busy)
        self.btn_mob_del.setEnabled(bool(self.grid_mob.current_id()) and not busy)
        self.lbl_mob_state.setText(("已选 %d 种怪" % n) if n else "还没有选怪")

    def _refresh_player_list(self):
        if getattr(self, "grid_player", None) is None:
            return
        self.grid_player.set_entries(self._player_rows())
        self._refresh_player_buttons()

    def _refresh_player_buttons(self):
        if getattr(self, "grid_player", None) is None:
            return
        busy = bool(getattr(self, "_busy", False))
        # ⚠⚠ **"有没有角色"只认项目（`_player_rows()` ✓），不许拿格子数当数** ✗ ——
        #   格子是"显示"、真值在 `project.player_id` ✓，两者**会背离**：切 / 关项目那一刻
        #   格子还留着上一条、而 `_player_rows()` 已经是 `[]` ✗ ⇒ 原来那句
        #   `self._player_rows()[0]["id"]` 直接 `IndexError` ✗（用户 2026-10-05 的本条用例
        #   一跑就撞出来 ✓）。⚠ 而它是在**槽里**被调的（`set_busy` 从 `_on_done` 过来 ✓）⇒
        #   槽里抛异常 = PyQt `qFatal` = 整个程序无征兆闪退 ✓（见 `gui/worker.py::safe_slot` ✓）。
        #   ⇒ 只认 `rows` ✓（它本身就是格子那一行的来源 ✓ 见 `_refresh_player_list` ✓）。
        rows = self._player_rows()
        n = len(rows)
        # ⭐ 「每帧最多几个框」= 0 ⇒ 一个框都不写 ⇒ 这一对按钮灰着（用户 2026-10-05 ✓）
        peaks = self._peaks_of("player_max_peaks")
        on = n > 0 and not busy and peaks != 0
        if busy:
            _why = "（现在灰着：正在跑别的任务 ✓）"
        elif peaks == 0:
            _why = "（现在灰着：「每帧最多几个框」= 0 ⇒ 一个都不标 ✓ 想标就填回 ≥1，或 -1 = 无限 ✓）"
        elif not n:
            _why = "（现在灰着：还没有选角色 —— 先点「添加…」✓）"
        else:
            _why = ""
        for btn in (getattr(self, "btn_player_tpl", None),
                    getattr(self, "btn_player_yolo", None)):
            if btn is None:
                continue
            btn.setEnabled(on)
            # ⚠ 这两个按钮的提示是**建的时候**写的 ⇒ 这里**第一次**先把原文存下来、之后只追加
            #   那段"为什么灰" ✗（直接 setToolTip 会把原文冲掉 ✓ 那是丢信息 ✓）。
            _base = btn.property("_tip_base")
            if _base is None:
                _base = btn.toolTip() or ""
                btn.setProperty("_tip_base", _base)
            btn.setToolTip(_base + _why)
        self.btn_player_add.setEnabled(not busy)
        self.btn_player_del.setEnabled(bool(self.grid_player.current_id()) and not busy)
        self.lbl_player_state.setText(("已选角色 %s" % rows[0]["id"]) if rows
                                      else "还没有选角色")

    # ---------------- 「这条目用哪些帧当模板」（任务 3 ✓ 用户 2026-10-05）----------------

    def _entry_key(self, target, entry):
        """条目在 `label.frames_sel` 里的键 ⇒ `<target>:<模板 id>` ✓。

        ⚠ 键里的 id 必须是**管线看到的那个模板 id** ✗（别用显示名 ✓）：掉落物 = 掉落 id ✓；
          宠物 = **组合外观的目录名**（`petlib.combo_key` ✓）—— `load_templates` 就是按这个
          目录名去找模板的 ✓（见 `tools/detect_mobs.py` ✓）。
        """
        if target == "drop":
            return "drop:%s" % (entry.get("id") or "")
        if target == "pet":
            return "pet:%s" % petlib.combo_key(entry.get("pet") or "",
                                               entry.get("equips") or ())
        return "%s:%s" % (target, entry.get("id") or entry.get("pet") or "")

    def _entry_frames(self, target, entry):
        """这条目**全部**能当模板的帧名（顺序照库里那个 ✓）—— 给"算默认勾选"用 ✓。

        ⭐ 2026-10-05：真正的**来源**搬进 `_entry_frame_files` 了 ✓（那里连**图片路径**
        一起给出 ✓ 供"选模板帧"弹窗看图 ✓ 用户原话："预览模板帧需要能看到图，
        光看名字判断不出来"）。这里只取键 ⇒ **顺序一字不变** ✓（dict 保序 ✓）。
        """
        return list(self._entry_frame_files(target, entry))

    def _entry_frame_files(self, target, entry):
        """这条目**全部**能当模板的帧（stem ✓ 顺序照库里那个 ✓）。

        ⚠ 与管线**同源** ✓：掉落物 = `wzexport.drop_frame_files` ✓；宠物 = 组合外观目录里的
          `*.png`（`petlib.compose_pet` **现算** ✓ 必须与 ④ 标定时用的同一份 ✓
          不然量的就是别的图 ✗）。
        """
        if target == "drop":
            return {f.stem: f for f in wzexport.drop_frame_files(entry.get("id") or "")}
        if target == "pet":
            try:
                root = petlib.compose_pet(entry.get("pet") or "",
                                          entry.get("equips") or ())
            except Exception:                      # noqa: BLE001 —— 做不出来 ⇒ 如实空 ✓
                root = None
            if root is None:
                return {}
            return {p.stem: p for p in sorted(Path(root).glob("*.png"), key=lambda q: q.stem)
                if p.stem != "icon"}
        # ⭐⭐ **怪物**（用户 2026-10-05 ✓ 四段统一）：模板就是精灵库里那个怪目录下的
        #   `*.png` ✓（`meta` 是元数据不是帧 ⇒ 排除 ✓ 与 `_mob_rows` 的 icon 口径一致 ✓）。
        if target == "mob":
            d = wzexport.sprite_dir_path() / str(entry.get("id") or "")
            if not d.is_dir():
                return {}
            return {p.stem: p for p in sorted(d.glob("*.png"), key=lambda q: q.stem)
                    if p.stem != "meta"}
        # ⭐⭐ **玩家**（同上）：模板在 `datasets/sprites/player/<角色id>/` ✓
        #   （`icon` 是格子上的缩略图、不是模板帧 ⇒ 排除 ✓ 与宠物那块同款 ✓）。
        if target == "player":
            d = Path("datasets/sprites/player") / str(entry.get("id") or "")
            if not d.is_dir():
                return {}
            return {p.stem: p for p in sorted(d.glob("*.png"), key=lambda q: q.stem)
                    if p.stem != "icon"}
        return {}

    def _default_frames(self, target, stems):
        """**旧逻辑**会用的那些帧 ⇒ 弹窗的默认勾选 ✓（用户说的"默认是旧逻辑中的那些"✓）。

        ⚠ 直接借管线那把尺子（`tools.detect_mobs._pick_frames` ✓：排除死亡动画 ✓
          不超过「最大模板帧」就全用 ✓ 超了按动作轮询 ✓）—— **别在界面层另写一份** ✗
          （两处必然漂 ✓）。
        ⭐ 2026-10-05 修正：上限原来四段都取**总的** `per_mob` ✗ —— 可管线那边各段用的是
        **各自**那个数（掉落 `drop_per_mob` ✓ 宠物 `pet_per_mob` ✓ 玩家 `player_per_mob` ✓
        见 `detect_player.py:158` ✓）⇒ 弹窗的"默认勾选"必须跟管线**同源** ✓
        不然"默认"与"真跑"就是两份名单 ✗（而"默认 = 旧逻辑"正是用户要的 ✓）。
        """
        from tools.detect_mobs import _pick_frames
        cap_key = {"mob": "per_mob", "drop": "drop_per_mob",
                   "pet": "pet_per_mob", "player": "player_per_mob"}.get(target, "per_mob")
        cap = int(self.value(cap_key) or 0)
        picked = _pick_frames([Path("%s.png" % s) for s in stems], cap)
        return [p.stem for p in picked]

    def _pick_entry_frames(self, target):
        """背包窗格**双击**某个条目 ⇒ 弹"选模板帧"⇒ 确定了就落盘 ✓。"""
        if self.project is None:
            return
        # ⭐⭐ **四段都接上**（用户 2026-10-05 ✓：「四块窗格各接一行双击选模板帧」）——
        #   管线那边**早就认这份名单**了 ✓，之前只是"卡片没把它递下去" ✗（界面能选、管线收不到 ✗）：
        #     · 怪物：`detect_mobs.py:655` 那句 `str(k).split(":", 1)[1]` **就是**剥 `mob:` 前缀 ✓；
        #     · 玩家：`detect_player.py:164` 按 `player:<角色id>` 摘 ✓
        #       （键格式与 `_entry_key` 的兜底 `"<target>:<id>"` ✓ 和 `_player_rows` 的 `id` ✓ 正好对上 ✓）。
        grid = {"drop": self.grid_drop, "pet": self.grid_pet,
                "mob": getattr(self, "grid_mob", None),
                "player": getattr(self, "grid_player", None)}.get(target)
        if grid is None:
            return
        e = grid.current_entry() or {}
        key = self._entry_key(target, e)
        # ⭐⭐ 顺手把**每帧的图片路径**也拿到 ✓ —— 弹窗要**看图**（用户 2026-10-05 ✓ 原话：
        #   "预览模板帧需要能看到图，光看名字判断不出来"）⇒ `files={stem: 路径}` ✓
        #   ⚠ 四段的图片位置各不相同 ⇒ 由卡片给 ✓ 界面层不自己猜 ✗（`_entry_frame_files` ✓）。
        _files = self._entry_frame_files(target, e)
        allst = list(_files)                                # 与路径同序 ✓
        if not allst:
            return                      # 这一条在图库里一帧都没有 ⇒ 弹窗也没东西可勾 ✓
        _set = set(allst)
        cur = [s for s in (self._frames_sel.get(key) or []) if s in _set]
        if not cur:
            cur = self._default_frames(target, allst)      # 没存过 ⇒ 默认 = 老逻辑 ✓
        from gui.frame_sel_dialog import FrameSelDialog
        dlg = FrameSelDialog(self.window(),
                             str(e.get("name") or e.get("id") or e.get("pet") or key),
                             allst, cur, files=_files)
        if dlg.exec_():
            self._save_frames_sel(key, dlg.selected_stems())

    def _save_frames_sel(self, key, stems):
        """把"这条目用哪些帧"落盘 ⇒ ⚠ **空列表也照样存** ✓（= 这一条这一趟不用模板 ✓
        别静默回落到"全用" ✗）—— 与 `label.drops` / `label.pets` 一样**当场存** ✓。"""
        self._frames_sel[key] = sorted(str(s) for s in (stems or []))
        try:
            self.sync(self.project)
            self.project.save()
        except Exception as e:                     # noqa: BLE001
            print("[label] 存模板帧选择失败:", e)

    def _drop_add(self):
        from gui.drop_picker import DropPickDialog
        dlg = DropPickDialog(self._drops, self.window())
        if dlg.exec_() != QDialog.Accepted:
            return
        self._drops = dlg.result_drops()
        self._refresh_drop_list()
        self._save_drop_choice()

    def _drop_del(self):
        did = self.grid_drop.current_id()        # ⚠ 唯一出口（别直接读行号 ✗ 格子版式换过）
        if not did:
            return
        self._drops = [d for d in self._drops if d["id"] != did]
        self._refresh_drop_list()
        self._save_drop_choice()

    # ---------------- 宠物（class 5，用户 2026-10-04 ✓ 方案 C）----------------

    def _emit_pet(self, mode):
        """宠物两个按钮的入口（`template` / `yolo` ✓）。⚠ 与掉落物那条**同一个套路** ✓。

        ⭐ **先弹选帧窗**（用户 2026-10-04 ✓ 同掉落物那条 ✓）：`template` ⇒ 确认重标 + 选帧；
        `yolo` ⇒ 只选帧（追加性质 ✓）。台账按 target = `"pet"` 分 ✓（见 `tools/label_pets.py` ✓）。
        """
        self._label_target = "pet"
        self._pet_mode = str(mode or "template").strip().lower()
        self._yolo_only = False          # 清掉上一次「只跑 YOLO」的残留（那是 mob/player 的 ✓）
        if self._pet_mode != "yolo" and not self._confirm_relabel("pet"):
            return
        ok, picked = self._pick_frames("pet")
        if not ok:
            return
        self._only["pet"] = picked       # None = 全选（不筛）✓
        self.run_clicked.emit(self)

    def _refresh_pet_ui(self):
        """勾选框 → 显示/隐藏那一整块（**藏起来**才是"出现" ✓ 同掉落物那块 ✓）。"""
        if not getattr(self, "pet_box", None):
            return
        self.pet_box.setVisible(bool(self.ck_pet.isChecked()))
        self._refresh_pet_buttons()

    def _pet_rows(self):
        """列表要显示的条目 —— ⚠ **口径只此一处** ✓（`gui.pet_picker.combo_entry` ✓，
        组合图现算 ✓ 有缓存 ✓）。"""
        from gui.pet_picker import combo_entry
        return [combo_entry({"id": d["pet"], "name": d.get("name")}, d.get("equips"))
                for d in self._pets]

    def _refresh_pet_list(self):
        self.grid_pet.set_entries(self._pet_rows())
        self._refresh_pet_state()
        self._refresh_pet_buttons()

    def _refresh_pet_state(self):
        """列表旁边那句现状：图库在不在、缺哪一步（用户 2026-10-04 ✓ 不假报空 ✗）。"""
        ok, why = petlib.pet_library_state()
        n = len(self._pets)
        head = ("已选 %d 套" % n) if n else "还没有选宠物"
        self.lbl_pet_state.setText(head if ok else (head + " · " + why.splitlines()[0]))
        self.lbl_pet_state.setToolTip(why)

    def _refresh_pet_buttons(self):
        """列表里有项目、且图库可用时两个按钮才亮；灰着要说清为什么 ✓（同掉落物那块 ✓）。"""
        busy = bool(getattr(self, "_busy", False))
        # ⭐ 「每帧最多几个框」= 0 ⇒ 一个框都不写 ⇒ 这一对按钮灰着（用户 2026-10-05 ✓）
        peaks = self._peaks_of("pet_max_peaks")
        on = len(self._pets) > 0 and not busy and peaks != 0
        tips = {
            "template": ("用**组合外观**（宠物 + 它戴的装备 ✓）当模板做匹配，"
                         "写出去的类别号是 **class 5（宠物）** ✓。"),
            "yolo": ("用上面选的「YOLO 权重」补一轮框，只合并 **class 5（宠物）** 的预测 ✓。"),
        }
        if busy:
            why = "\n\n（现在灰着：正在跑别的任务 —— 等它跑完 ✓）"
        elif peaks == 0:
            why = ("\n\n（现在灰着：「每帧最多几个框」= 0 ⇒ 这一趟一个框都不写 ✓"
                   " 想标就填回 ≥1，或填 -1 = 无限 ✓）")
        elif not len(self._pets):
            why = "\n\n（现在灰着：先在「要标注的宠物」里加至少一套 ✓）"
        else:
            why = ""
        for btn, mode in ((self.btn_pet_tpl, "template"), (self.btn_pet_yolo, "yolo")):
            btn.setEnabled(on)
            btn.setToolTip(tips[mode] + why)
        self.btn_pet_add.setEnabled(not busy)
        # ⭐ 「移出选中」= 只有格子里**选中了一套**才能点 ✓（选中变化时会被重算 ✓
        #   见上面那处 `itemSelectionChanged` 的接线 ✓）；灰着把原因写进 tooltip ✓
        #   （不然又是一个"点了没反应"的迷案 ✗ —— 用户这次报的就是它 ✓）。
        sel = str(self.grid_pet.current_id() or "")
        self.btn_pet_del.setEnabled(bool(sel) and not busy)
        self.btn_pet_del.setToolTip(
            "把格子里**选中的那一套**（宠物 + 它戴的装备）移出列表 ✓"
            + ("\n\n（现在灰着：先在左边格子里**点中一套** ✓）" if not sel else
               "\n\n当前选中：%s ✓" % sel)
            + ("\n\n（正在跑别的任务 —— 等它跑完 ✓）" if busy else ""))

    def _pet_add(self):
        from gui.pet_picker import PetPickDialog
        dlg = PetPickDialog(self._pets, self.window())
        if dlg.exec_() != QDialog.Accepted:
            return
        self._pets = petlib.norm_pets(dlg.picked_pets())
        self._refresh_pet_list()
        self._save_pet_choice()
        self._export_missing_pet_frames()    # ⭐ 关窗之后：这套组合缺图 ⇒ 当场读条补导 ✓

    # ⭐⭐ 加完宠物、**弹窗关掉之后**：缺的图当场补导（用户 2026-10-05 ✓）
    def _export_missing_pet_frames(self):
        """这套「宠物 + 装备」要用的图**还没导出** ⇒ 在**工作台**上显式读条补导 ✓。

        用户 2026-10-05 ✓ 原话："如果没有导出，在添加的带装备的宠物、关闭弹窗后，在数据集
        工作台显式读条导出" ✓（起因 = 这一轮报的现场：加了带装备的宠物，双击格子一看
        **除了 stand0 全都没有装备** ✗ —— 真因是装备图库只导了 `stand0` ✓ 见
        `core/petlib.py` 那段"组合要用的图导没导"✓）。

        判据与命令都**不在这一层**：缺什么 = `petlib.combo_export_plan` ✓、
        怎么跑 = `wzexport.pet_export_task` ✓（界面层不自己拼命令 ✗ 一处口径 ✓）。

        ⚠ 三条纪律：
          · **显式**：跑在工作台那条读条 + 日志 + 取消上（`MainWindow.run_export` ✓）——
            补数据也是"有处理过程的行为"✓ 别偷偷跑 ✗；
          · **不静默**：这里没有工作台（用例、别处嵌的卡片 ✓）、或工作台正忙（起不来 ✓）
            ⇒ 把"缺什么 + 怎么补"写进这一块的状态行 ✓（见 `_pet_note` ✓）；
          · **导完要刷**：成功/失败都把宠物列表重新铺一遍 ✓ —— 成功时组合模板会**自动重做**
            ✓（源目录指纹变了 ✓ `petlib._stamp` ✓），图标与弹窗预览跟着变对 ✓。
        """
        try:
            plan = petlib.combo_export_plan(self._pets)
        except Exception as e:                  # noqa: BLE001 —— 判断坏了别把"加宠物"弄挂 ✗
            self._pet_note("⚠ 检查图库时出错：%s: %s" % (type(e).__name__, e))
            return
        if not plan:
            return                              # 图都齐了 ⇒ 什么都不做（也不打扰人 ✓）
        want = sum(len(s.get("ids") or []) for s in plan)
        fn, params = wzexport.pet_export_task(plan)
        run = getattr(self.window(), "run_export", None)
        if run is None:                         # 没有工作台 ⇒ 如实说，别静默 ✗
            self._pet_note("⚠ 这套组合还缺 %d 项图（装备只导了 stand0 那种 ✓），"
                           "但这里没有工作台可以跑导出" % want)
            return
        if not run(fn, params, "补导宠物图库（%d 项）" % want,
                   on_done=self._on_pet_export_done):
            self._pet_note("⚠ 有别的任务在跑 ⇒ 这套组合还缺 %d 项图；"
                           "等它跑完后**再点一次「添加…」**就会自动补导 ✓" % want)

    def _pet_note(self, text):
        """宠物那块状态行上写一句实话（**别静默** ✗）—— 下次刷新会被 `_refresh_pet_state` 覆盖 ✓。"""
        try:
            self.lbl_pet_state.setText(str(text))
            self.lbl_pet_state.setToolTip(str(text))
        except Exception:                       # noqa: BLE001 —— 界面不在（用例 ✓）就别崩 ✗
            pass

    def _on_pet_export_done(self, ok, summary):
        """补导结束（`MainWindow.run_export` 的收尾回调 ✓ 在 GUI 线程里 ✓）。

        ⚠ 顺序要紧：**先重铺列表**（组合模板这一趟会按新图重算 ✓ 图标/预览跟着对 ✓），
          再写那句状态 —— 反了的话 `_refresh_pet_state` 会把这句话冲掉 ✓。
        """
        self._refresh_pet_list()
        if ok:
            self._pet_note("✔ 已补导：%s ⇒ 这套组合现在带装备了 ✓"
                           "（双击格子看一眼那些帧 ✓）" % (summary or ""))
        else:
            self._pet_note("⚠ 补导失败：%s（工作台日志里有完整输出 ✓）" % (summary or ""))

    def _pet_del(self):
        e = self.grid_pet.current_entry() or {}
        pid = str(e.get("pet") or self.grid_pet.current_id() or "")
        if not pid:
            return
        want = sorted(str(x) for x in (e.get("equips") or []))
        for i, d in enumerate(self._pets):        # 按"那一套"删 ✓（同一只可能配了多套 ✓）
            if d["pet"] == pid and sorted(d.get("equips") or []) == want:
                self._pets.pop(i)
                break
        self._refresh_pet_list()
        self._save_pet_choice()

    def _save_pet_choice(self):
        """增删完**当场存进项目**（`label.pets` ✓）—— 别等"跑一次"才落盘 ✗（同掉落物 ✓）。"""
        p = self.project
        if p is None:
            return
        try:
            self.sync(p)
            p.save()
        except Exception as e:                            # noqa: BLE001
            print("⚠ 宠物选择没存进项目（%s: %s）" % (type(e).__name__, e))

    def _save_drop_choice(self):
        """增删完**当场存进项目**（`label.drops` ✓）—— 别等"跑一次"才落盘 ✗。

        ⚠ 存不上**不许拦人**（只读目录 / yaml 坏了 ✓）：本次选择照旧生效（界面就是真相 ✓），
          只印一句 ✓。
        """
        p = self.project
        if p is None:
            return
        try:
            self.sync(p)
            p.save()
        except Exception as e:                            # noqa: BLE001
            print("⚠ 掉落物选择没存进项目（%s: %s）" % (type(e).__name__, e))

    def _confirm_relabel(self, target):
        """标过的情况下弹二次确认。重标只覆盖自动框，人工修正会保留。"""
        p = self.project
        if p is None:
            return True

        from gui import labelio
        prog = labelio.label_progress(p)
        # ⚠⚠ **key 就是 target 本身**（"mob"/"player"/"drop"/"pet" ✓）—— 原来写死
        #   `"mob" if target == "mob" else "player"` ✗ ⇒ 掉落物/宠物一律被当成 player 查
        #   （`label_progress` 里补上 drop/pet 之后才对得上 ✓ 见 `gui/labelio` ✓）。
        key = str(target)
        if prog.get(key, 0) == 0:
            return True   # 这类还没标过，直接跑

        # ⚠ 文案里的说法走唯一出口 `labelio.target_zh`（类别表兜底 + 少数例外 ✓）—— 同上：
        #   写死三元式会让掉落物/宠物在弹窗里被叫成「玩家」✗（用户 2026-10-04 ✓ 这条引起的）。
        name = labelio.target_zh(target, target)
        # 这里只负责「提醒你这一类标过了」，**不是**问「要不要全部重标」——
        # 范围在下一步的选帧窗里定（默认全选）。所以文案必须点出下一步，
        # 否则点「否」的人以为没别的选择了，其实他要的是「只补几帧」。
        r = QMessageBox.question(
            self, "确认重新标注",
            "已标过 %s（%d 帧）。\n\n"
            "重新标注会重新生成自动标注框，\n"
            "人工修正的框会保留（不会被覆盖）。\n\n"
            "点「是」之后还会让你选要处理哪些帧：\n"
            "默认全选（= 全部重标），只想补几帧就在那一步挑。\n\n"
            "继续吗？" % (name, prog[key]),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return r == QMessageBox.Yes

    def set_busy(self, busy):
        """忙 / 闲：四段那几对按钮 + 添加/删除 + 那几行"现状" —— **全都从这一个口收** ✓。

        ⛔⛔ **`self._busy` 必须先赋值、再 refresh**（用户 2026-10-05 ✓ 现场原话："为什么 YOLO
        自动标注、手动取消正在运行的模板匹配后，玩家、怪物的标注按钮就灰了再也不能点了？"）：
        四个 `_refresh_*` 的启用判据**都是** `n > 0 and not self._busy` ✓ ⇒ 原来 `_busy` 写在了
        `_refresh_mob_buttons()` / `_refresh_player_buttons()` **之后** ✗ ⇒ 那两次读到的是
        **上一次**的忙闲 ✗：
          · `set_busy(False)`（跑完、**取消也走它** ✓ —— 取消时 `sig_done(False, …)` ⇒ `_on_done` ✓）
            反而把按钮**关灰** ✗ ⇒ 每收一次工就关一次 ⇒ 用户看到的"灰了再也不能点" ✓
            （只有再去动一下怪物/角色列表才会重新点亮 ✗ —— 那正是"有时又好像能点"的来源 ✓）；
          · 反过来 `set_busy(True)` 会让它们**亮着** ✗（跑着还能点 ⇒ 两个任务能叠着跑 ✓ 另一个方向错）。
        ⇒ 一条规矩：**先 `_busy`，后 refresh** ✓（四个 refresh 一个都不许排到它前面 ✗）。
        ⚠ 也**不再在这里** `setEnabled` 一遍 ✗：启用条件（列表空不空 + 忙不忙）**只在四个 refresh
          里各写一处** ✓ —— 两份口径必然漂 ✓（原来那句 `for _btn … setEnabled(not busy)` 正是
          漂的那一份：先点亮、又被 refresh 按旧值关灰 ✓）。
        """
        super().set_busy(busy)
        self._busy = bool(busy)          # ⚠ 必须在下面四个 refresh **之前** ✓
        self._refresh_mob_buttons()      # 亮/灰 + 「已选 N 种」+ 添加/删除 ✓
        self._refresh_player_buttons()
        self._refresh_drop_buttons()     # 掉落物：列表空 ⇒ 仍然灰（用户 2026-10-04 §2 ✓）
        self._refresh_pet_buttons()      # 宠物同理（漏了它 ⇒ 跑任务时不会灰 ⇒ 能重复点 ✗）

    @staticmethod
    def _section(text, tip=""):
        """分区小标题（用户 2026-10-04 ✓ 原话："整理…参数、按钮分类布局，要求功能模块划分合理"）。

        ⚠ 为什么用**小标题行**而不是 QGroupBox：这张卡片是"一列参数 + 一排按钮"的窄卡片，
          再套一层带边框的组会把行高撑两倍、可读性反而更差 ✗（同款做法见
          `gui/player_panel` 里那些"缩进块"的注释 ✓）。
        """
        w = QLabel(text)
        w.setStyleSheet("color:#5f6368; font-weight:600; padding-top:4px;")
        if tip:
            w.setToolTip(tip)
        return w

    def build_params(self, form):
        form.addRow("", self._section("标注对象 · 掉落物",
                                      "掉落物（class 2）走跟怪物**同一条**模板匹配/补框算法，"
                                      "只是类别号和模板来源（道具图库）不同 ✓"))
        self.ck_drop = self.field(form, "label_drops_on", "标注掉落物", "bool", False)
        field_tip(form.labelForField(self.ck_drop), 
            "勾上之后，卡片最下面会出现**掉落物列表**与两个按钮（模板匹配标注掉落物 / YOLO标注掉落物）。\n\n"
            "它标的是**掉落物**（类别表里的 class 2 = drop）—— 和「标注怪物」同一套算法，\n"
            "只是模板换成**道具图标**、写出去的类别号是 2 ✓。\n\n"
            "⚠ 模板/图标来自**掉落物图库**（`datasets/sprites/drop/<id>/*.png` +\n"
            "   清单 `datasets/drops.json`）—— 这一套要等 WzProbe 有道具导出通道才做得出 ✓。")
        self.ck_drop.stateChanged.connect(lambda _s: self._refresh_drop_ui())

        form.addRow("", self._section(
            "标注对象 · 宠物",
            "宠物（class 5）走跟怪物**同一条**模板匹配/补框算法；模板 = **宠物 + 它戴的装备**"
            "（组合外观 ✓）—— 戴了帽子/丝带之后样子就变了，模板得按「戴着的样子」做 ✓"))
        self.ck_pet = self.field(form, "label_pets_on", "标注宠物", "bool", False)
        field_tip(form.labelForField(self.ck_pet),
            "勾上之后，卡片最下面会出现**宠物列表**与两个按钮（模板匹配标注宠物 / YOLO标注宠物）。\n\n"
            "它标的是**宠物**（类别表里的 class 5 ✓）—— 那是**干扰类**：标它只为让模型\n"
            "学会区分「宠物 vs 怪」（被宠物挡住的怪也能检测、宠物不被误检为怪 ✓）。\n\n"
            "⚠ 模板来自**组合外观**（`datasets/sprites/pet_combo/<宠物>__<装备>/` ✓，由宠物图库\n"
            "   现算出来 ✓）—— 宠物图库要先用 `WzProbe dump-pets / dump-petequips` 导出 ✓。")
        self.ck_pet.stateChanged.connect(lambda _s: self._refresh_pet_ui())

    def load_from_project(self, p):
        sec = p.sec("label")
        for k in ("thresh", "min_distinct", "per_mob", "max_peaks", "downscale",
                  "player_thresh", "label_drops_on", "label_pets_on"):
            self.set_value(k, sec.get(k))
        # ⭐ 2026-10-05：玩家这一段另外 4 个（用户："4类按照统一的格式整理" + "缺功能的就补功能" ✓）——
        #   ⚠ 默认值**故意与怪物不同**（见 `build_params` 里那段说明 ✓）：
        #   区分度 0（不看 ✓）｜最大模板帧 0（不截 ✓）｜每帧最多几个框 1｜降采样 1 ✓。
        self.set_value("player_min_distinct", sec.get("player_min_distinct", 0.0))
        self.set_value("player_per_mob", sec.get("player_per_mob", 0))
        self.set_value("player_max_peaks", sec.get("player_max_peaks", 1))
        self.set_value("player_downscale", sec.get("player_downscale", 1))
        # ⭐⭐ **框到可见部分**（用户 2026-10-05 ✓）：四段**共用一份** ✓（默认关 ✓ 老项目一字不变 ✓）
        self.set_value("visible", sec.get("visible", False))
        self.set_value("visible_keep", sec.get("visible_keep", 0.2))
        # ⭐ 掉落物列表（用户 2026-10-04 ✓）：开关走 `set_value`（自动 ✓），列表手工灌 ✓
        #   ⚠ 归一化用 `wzexport.norm_drops`（老配置可能只存了 id ✓ 一处口径 ✓）
        # ⭐⭐ **默认 = 那 4 种金币**（用户 2026-10-04 ✓ 原话："将4种金币默认添加进要标注的
        #   掉落物"）—— ⚠ **只在"这一项从来没存过"时给**（`raw is None` ✓）：
        #   存过就照他存的来 ✓（哪怕被清成 `[]` ✓）—— 否则"把金币删掉"会被当成"没配过"、
        #   下次打开又自己长回来 ✗（这种"删不掉的东西"最招人烦 ✓）。
        _raw = sec.get("drops")
        self._drops = wzexport.norm_drops(_raw)
        if _raw is None:
            self._drops = wzexport.norm_drops(wzexport.default_drop_ids())
        # ⭐ 「匹配阈值（掉落物）」（用户 2026-10-04 ✓）：**老项目没有这个键** ⇒ 显示
        #   "现在生效的那个数"（= 上面那个总阈值 ✓）—— ⚠ 别写成 0 ✗（那等于一个都匹配不上 ✓，
        #   见 `make_task` 里那处兜底 ✓）。
        self.set_value("drop_thresh", sec.get("drop_thresh", sec.get("thresh", 0.90)))
        # ⭐ 「降采样（掉落物）」（用户 2026-10-04 ✓）：**不**跟着总的那个走 ✓ ——
        #   老项目没有这个键 ⇒ 显示 **1**（这一块的默认 ✓ 见 `make_task` 里那段说明 ✓）。
        self.set_value("drop_downscale", sec.get("drop_downscale", 1))
        # ⭐ 2026-10-05：这一块另外三个参数（⚠ 没存过 ⇒ **显示"现在生效的那个数"** ✓
        #   别显示常量 ✗ —— 那会让人以为改过 ✓）
        self.set_value("drop_min_distinct", sec.get("drop_min_distinct",
                                                    sec.get("min_distinct", 0.06)))
        self.set_value("drop_per_mob", sec.get("drop_per_mob", sec.get("per_mob", 20)))
        self.set_value("drop_max_peaks", sec.get("drop_max_peaks", sec.get("max_peaks", 4)))
        # ⭐ 宠物那一块（用户 2026-10-04 ✓ 方案 C）：列表同样手工灌 ✓（`petlib.norm_pets` 一处口径 ✓）
        # ⭐⭐ 「这条目用哪些模板帧」（任务 3 ✓）：整份字典读进来 ✓，**缺键 = 老逻辑** ✓
        self._frames_sel = {str(k): [str(x) for x in (v or [])]
                            for k, v in (sec.get("frames_sel") or {}).items()}
        self._pets = petlib.norm_pets(sec.get("pets"))
        self._refresh_pet_list()
        self._refresh_pet_ui()
        self.set_value("pet_thresh", sec.get("pet_thresh", sec.get("thresh", 0.90)))
        # ⭐ 「降采样（宠物）」：同掉落物那条（**不**继承 ✓ 默认 1 ✓）
        self.set_value("pet_downscale", sec.get("pet_downscale", 1))
        # ⭐ 2026-10-05：这一块另外三个参数（同掉落物 ✓ 没存过就显示"现在生效的那个数" ✓）
        self.set_value("pet_min_distinct", sec.get("pet_min_distinct",
                                                   sec.get("min_distinct", 0.06)))
        self.set_value("pet_per_mob", sec.get("pet_per_mob", sec.get("per_mob", 20)))
        self.set_value("pet_max_peaks", sec.get("pet_max_peaks", sec.get("max_peaks", 4)))
        # ⭐ 新建的两段（怪物 / 玩家 ✓ 用户 2026-10-05）跟着项目重建格子 ✓
        #   （`bind` → `load_from_project` 就是这条路 ✓ 换项目/开项目都会走到 ✓）
        self._refresh_mob_list()
        self._refresh_player_list()
        self._refresh_drop_list()
        self._refresh_drop_ui()

    def sync(self, p):
        sec = p.sec("label")
        sec.update(
            self.values(["thresh", "min_distinct", "per_mob", "max_peaks", "downscale",
                         # ⭐ 2026-10-05：四类**各自**那 5 个参数（用户："4类按照统一的格式整理"✓）
                         "player_thresh", "player_min_distinct", "player_per_mob",
                         "player_max_peaks", "player_downscale",
                         "drop_min_distinct", "drop_per_mob", "drop_max_peaks",
                         "pet_min_distinct", "pet_per_mob", "pet_max_peaks",
                         "label_drops_on",
                         # ⭐ 掉落物那块自己的匹配阈值（用户 2026-10-04 ✓ 见 `make_task` ✓）
                         "drop_thresh",
                         # ⭐⭐ 两块**各自**的降采样（用户 2026-10-04 ✓ 实测：跟着怪物那个 2
                         #   会把小目标压糊 ⇒ 0 框 ✗ 见 `make_task` ✓）
                         "drop_downscale",
                         # ⭐ 宠物那一块（方案 C ✓）：勾选 / 自己的匹配阈值 —— 列表单独存 ✓
                         "label_pets_on", "pet_thresh", "pet_downscale"]))
        sec["pets"] = [dict(d) for d in self._pets]       # 列表单独存（不当参数行 ✓）
        sec["drops"] = [dict(d) for d in self._drops]       # 列表单独存（不当参数行 ✓）
        # ⭐⭐ 模板帧选择（任务 3 ✓）：{`<target>:<id>`: [帧 stem …]} ✓ —— 与列表一样**不是参数行** ✓
        sec["frames_sel"] = {str(k): list(v) for k, v in (self._frames_sel or {}).items()}

    def refresh(self, force=False):
        super().refresh(force)
        if self.project is not None:
            ps = self.project.get("player_scale")
            self.lbl_player_scale.setText(
                ("%.3f" % ps) if ps else "—（先跑 ③ 标定）")
        self._fill_yolo_weights()
        self._refresh_drop_ui()          # 勾选状态 → 那一块显隐（切项目也要跟上 ✓）

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

        # ⭐⭐ 掉落物（class 2；用户 2026-10-04 ✓）—— 走**自己**那套前提，与 mob/player 无关：
        #   要列表非空 ✓、图库能用 ✓、选中的至少有一个有图 ✓、YOLO 那条还额外要权重 ✓。
        #   ⚠ 必须放在 `_yolo_only` **之前**：那个标志是 mob/player 那两个 YOLO 按钮的 ✓
        #     （掉落物这条不用它 ✓ 混在一起会走进怪物的依赖检查 ✗）。
        if self._label_target == "drop":
            if not self._drops:
                return False, ("还没有选掉落物 —— 先勾上「标注掉落物」、"
                               "再点列表下面的「添加…」至少加一项 ✓")
            ok, why = wzexport.drop_library_state()
            if not ok:
                return False, why
            if not any(wzexport.drop_icon_path(d["id"]) for d in self._drops):
                return False, ("选中的掉落物在图库里**一张图都没有** ⇒ 没有模板可用。\n\n%s"
                               % why)
            if self._drop_mode == "yolo" and not self.cmb_yolo.currentData():
                return False, "还没选 YOLO 权重 —— 先在「YOLO 权重」里选一个模型 ✓"
            return True, ""

        # 「YOLO 标注」独立跑：只要求权重（玩家模式还要求角色一致），
        # 不要求模板匹配那些前提 —— 它是补框，和模板匹不匹配没关系。
        if self._yolo_only:
            w = self.cmb_yolo.currentData()
            if not w:
                return False, "还没有训练好的模型 —— 先在其它项目完成 ⑦ 训练"
            if self._label_target == "player":
                pid = (p.get("player_id") or "").strip()
                if not pid:
                    return False, "还没选择角色，请先完成 ① 地图"
                wpid = self._yolo_player.get(w, "")
                if wpid != pid:
                    return False, ("模型角色「%s」≠ 当前角色「%s」，用它标玩家会误标，"
                                   "请选一个角色一致的模型"
                                   % (wpid or "（无）", pid or "（无）"))
            return True, ""

        if self._label_target == "player":
            if not (p.get("player_id") or "").strip():
                return False, "还没选择角色，请先完成 ① 地图"
            if not p.get("player_scale"):
                return False, "玩家还没标定尺度，请先完成 ③ 标定尺度"
            return True, ""
        if not (p.get("mobs") or []):
            return False, "还没确定怪种，请先完成 ① 地图"
        if not wzexport.sprite_dir_path().is_dir():
            return False, ("精灵库不存在：\n%s\n\n"
                           "需要用 WzProbe dump-mob 导出，"
                           "或修改 config/wz.yaml 的 sprite_dir"
                           % wzexport.sprite_dir_path())
        return True, ""

    def _sprite_scale(self, p, kind="mob"):
        """这一份项目里「游戏把精灵画成多大」= ③ 标定出来的尺度 ✓（**唯一出口** ✓）。

        ⭐⭐ 掉落物 / 宠物这两条**必须传它**（用户 2026-10-04 ✓ 现场原话：小白雪人 5000020
        "模板匹配标注0检出"；同日实测：本项目标定 **1.406** ✓）：
          · 怪物那条一直是 `p.get("scale") or 1.12` ✓、玩家那条是 `player_scale` ✓；
          · 而**掉落物 / 宠物**这两条原来**一个尺度都没传** ✗ ⇒ `label_drops` / `label_pets`
            各自落到默认 **1.0** ✓ ⇒ 模板比画面上的目标小 ~40% ✗ ⇒ 模板匹配**一个都找不到** ✗
            ——实测（同 8 帧、同模板）：`1.0` 下**任何阈值**都 0 框 ✓✗；`1.406` 下能检出 ✓。
            ⇒ 这跟"阈值调低"完全无关 ✗（调低也没用 ✓），别再往那个方向找 ✓。

        ⭐ **`kind` 三档**（用户 2026-10-05 ✓ 原话："卡片3标定尺度 也要支持标定掉落物、宠物类"）：
          · `"mob"`（默认 ✓ 老调用一字不变 ✓）⇒ 总尺度 `scale`，没标过兜底 **1.12** ✓ 别改 ✗；
          · `"drop"` ⇒ **这一类自己的** `drop_scale`；没标过 ⇒ **回落到总尺度** ✓
            （= 老行为一字不变 ✓ 老项目照样能跑 ✓）；
          · `"pet"` ⇒ `pet_scale`，同上 ✓。
        ⚠ 只用**尺度** ✓，别顺手传逐帧那份 `mob_scales` ✗：它的键是「怪 id:帧」（见
          `calib_manual` / `detect_mobs` ✓），掉落物 / 宠物**没有那套标定** ⇒ 硬套会把尺度带歪 ✓。
        """
        if kind in ("drop", "pet"):
            v = p.get("drop_scale" if kind == "drop" else "pet_scale")
            if v:
                return float(v)              # ③ 为这一类**专门**标定的那个 ✓
        return float(p.get("scale") or 1.12)     # 1.12 = 怪物那条一直用的兜底 ✓ 别改 ✗

    def _visible_params(self, sec):
        """⭐ **框到可见部分**那两个参数（用户 2026-10-05 ✓ 实现见 `perception/visible_box.py` ✓）。

        ⚠ **一处口径** ✓：怪物 / 掉落物 / 宠物 **/** 玩家**四段都从这儿取 ✓ —— 免得哪个 cfg
          漏了它、界面上勾了却没生效 ✗（"界面能选、管线收不到"这种已经栽过一次 ✓）。
          ⭐ 玩家那段是 **2026-10-06 补上的**（用户原话："**所有的匹配都需要「框到可见部分」
          参数**"✓）：它走的是另一套定位（`perception/player_locator` ✓），当时那三段
          （卡片 / `run_detect_player` / `PlayerLocator`）**一起补** ✓ —— 判据是**同一份**实现 ✓。
        ⚠ **缺键 ⇒ 关** ✓（老项目一字不变 ✓ 这是改**标注语义**的功能 ✗）。
        ⚠ `visible_tol` **不在界面里**（内核支持、界面没开这一格 ✗ 用户没要 ⇒ 别自作主张加 ✗）
          ⇒ 它恒走实现那边那个常量 ✓。
        """
        return {"visible": bool(sec.get("visible", False)),
                "visible_keep": float(sec.get("visible_keep", 0.2))}

    def make_task(self, p):
        sec = p.sec("label")

        # ⭐⭐ 掉落物（class 2；用户 2026-10-04 ✓）—— 与 mob/player 分派**并列**的一条：
        #   两条路径都不在这里另写算法，而是复用 `tools/label_drops.run_label_drops`
        #   （它内部再转发给 `detect_mobs.run_detect(cls=2)` / `yolo_augment(mode="drop")` ✓）。
        #   ⚠ 放在最前面：`_yolo_only` 是 mob/player 的开关，别让它把掉落物这条带偏 ✗。
        if self._label_target == "drop":
            from tools.label_drops import run_label_drops
            w = self.cmb_yolo.currentData() or ""
            return run_label_drops, {
                "mode": self._drop_mode,             # "template" / "yolo"
                "drops": [dict(d) for d in self._drops],
                "frames": str(p.frames),
                "out": str(p.dir_of("labels_auto")),
                "vis_dir": str(p.vis) + "_drop",     # 可视化单独一个目录（别和怪物的混 ✓）
                # ⭐⭐ **尺度**（用户 2026-10-04 ✓ 见 `_sprite_scale` 那段：不传 ⇒ 默认 1.0 ⇒
                #   模板小 40% ⇒ 一个都找不到 ✗）；⭐ 2026-10-05：③ 里**专门为掉落物**标过
                #   `drop_scale` 就用它 ✓，没标过回落总尺度 ✓（老项目一字不变 ✓）。
                "scale": self._sprite_scale(p, "drop"),
                # ⭐⭐ 这条目**手工挑的模板帧**（任务 3 ✓ 缺键 = 老逻辑 ✓ 见 `_default_frames` ✓）
                # ⭐ 2026-10-05：**只递这一段自己的键** ✗ —— 原来是整份字典原样递下去 ✓，
                #   而怪物链（`detect_mobs.py:655`）是**对每个键都 `split(":", 1)[1]` 剥前缀**的 ✗
                #   ⇒ 掉落的键会被剥成"掉落 id"当怪 id 用 ✗（实测：四段的键混在一起递 ✓）。
                #   现在只留 `<这一段>:` 开头的 ✓（键格式见 `_entry_key` ✓）。
                "frames_sel": {str(k): list(v)
                               for k, v in (self._frames_sel or {}).items()
                               if str(k).startswith("drop:")},
                # ⭐⭐ **匹配阈值：这一块自己的优先**（用户 2026-10-04 ✓ 原话："勾选标注掉落物、
                #   标注宠物后，需要有参数「匹配阈值」"）。
                #   ⚠ 兜底链是 `drop_thresh` → 总的 `thresh` → 0.90：**老项目**（还没存过
                #     `drop_thresh` 的）取到的就是**原来那个数** ✓ 行为一字不变 ✓；
                #     其余几个参数（min_distinct / per_mob / max_peaks / downscale）仍与怪物
                #     同一份 ✓ —— 这次只把**阈值**拆出来（用户点名要的就是它 ✓）。
                "thresh": sec.get("drop_thresh", sec.get("thresh", 0.90)),
                # ⭐⭐ 框到可见部分（用户 2026-10-05 ✓ 一处口径见 `_visible_params` ✓）
                **self._visible_params(sec),
                # ⭐ 2026-10-05：这三项也**各存一份**了（用户："4类按照统一的格式整理"✓）——
                #   ⚠ 兜底链 `drop_*` → 总的那个 → 常量 ✓ ⇒ **老项目一字不变** ✓。
                "min_distinct": sec.get("drop_min_distinct", sec.get("min_distinct", 0.06)),
                "per_mob": sec.get("drop_per_mob", sec.get("per_mob", 20)),
                "max_peaks": sec.get("drop_max_peaks", sec.get("max_peaks", 4)),
                # ⭐⭐ **掉落物自己的降采样**（用户 2026-10-04 ✓ 实测逼出来的 ✓ 见上面字段的说明）：
                #   ⛔ **不许**回落到总的 `downscale` ✗ —— 那个常按怪物（~100px）设成 2 ✓，
                #   金币这种十几像素的图标缩一半就糊 ⇒ **0 框** ✓（实测同 8 帧：`1` 检出、
                #   `2` 是 0 框 ✗）。默认 **1** ✓。
                "downscale": int(sec.get("drop_downscale", 1) or 1),
                "vis": True,
                # YOLO 那条（mode=yolo 时才用 ✓）
                "weights": w,
                "player_id": p.get("player_id") or "",
                "weights_player_id": self._yolo_player.get(w, ""),
                "conf": 0.40,
                "imgsz": p.sec("train").get("imgsz", 960),
                "device": p.sec("train").get("device", "0"),
                "only": self._only.get("drop") or [],
            }

        # ⭐⭐ 宠物（class 5；用户 2026-10-04 ✓ 方案 C）—— 与掉落物那条**并列**：
        #   模板要**现算组合外观** ⇒ 交给 `tools/label_pets.run_label_pets` ✓（它内部再转发给
        #   `detect_mobs.run_detect(cls=5)` / `yolo_augment(mode="pet")` ✓）。
        if self._label_target == "pet":
            from tools.label_pets import run_label_pets
            w = self.cmb_yolo.currentData() or ""
            return run_label_pets, {
                "mode": self._pet_mode,              # "template" / "yolo"
                "pets": [dict(d) for d in self._pets],
                "frames": str(p.frames),
                "out": str(p.dir_of("labels_auto")),
                "vis_dir": str(p.vis) + "_pet",      # 可视化单独一个目录（别和怪物/掉落的混 ✓）
                # ⭐⭐ **尺度**（用户 2026-10-04 ✓ 就是这条引起的"模板匹配标注 0 检出"✗：
                #   不传 ⇒ `label_pets` 默认 1.0 ✓ 而本项目标定 1.406 ⇒ 模板小 40% ✗ ⇒ 0 框 ✓）；
                #   ⭐ 2026-10-05：③ 里**专门为宠物**标过 `pet_scale` 就用它 ✓，没标回落总尺度 ✓
                #   （老项目一字不变 ✓）。
                "scale": self._sprite_scale(p, "pet"),
                # ⭐⭐ 这条目**手工挑的模板帧**（任务 3 ✓ 缺键 = 老逻辑 ✓）
                # ⭐ 2026-10-05：**只递这一段自己的键** ✗ —— 原来是整份字典原样递下去 ✓，
                #   而怪物链（`detect_mobs.py:655`）是**对每个键都 `split(":", 1)[1]` 剥前缀**的 ✗
                #   ⇒ 宠物/掉落的键会被剥成"目录名 / 掉落 id"当怪 id 用 ✗
                #   （实测：四段的键混在一起递 ✓）。现在只留 `pet:` 开头的 ✓（见 `_entry_key` ✓）。
                "frames_sel": {str(k): list(v)
                               for k, v in (self._frames_sel or {}).items()
                               if str(k).startswith("pet:")},
                # ⭐ 匹配阈值：这一块自己的优先 ✓（兜底链 `pet_thresh` → 总的 `thresh` → 0.90 ✓
                #   老项目取到的就是原来那个数 ✓ 行为一字不变 ✓）
                "thresh": sec.get("pet_thresh", sec.get("thresh", 0.90)),
                # ⭐⭐ 框到可见部分（用户 2026-10-05 ✓ 一处口径见 `_visible_params` ✓）
                **self._visible_params(sec),
                # ⭐ 2026-10-05：这三项各存一份 ✓（兜底链 `pet_*` → 总的 → 常量 ✓ 老项目不变 ✓）
                "min_distinct": sec.get("pet_min_distinct", sec.get("min_distinct", 0.06)),
                "per_mob": sec.get("pet_per_mob", sec.get("per_mob", 20)),
                "max_peaks": sec.get("pet_max_peaks", sec.get("max_peaks", 4)),
                # ⭐⭐ **宠物自己的降采样**（用户 2026-10-04 ✓ 实测逼出来的 ✓ 见上面字段的说明）：
                #   ⛔ 同上，**不许**回落到总的 `downscale` ✗（本项目存的是 2 ✓ ⇒ 宠物在画面上
                #   才 ~70 像素 ⇒ 缩一半就糊 ⇒ **0 框** ✓；实测 `1` 检出、`2` 是 0 框 ✗）。
                "downscale": int(sec.get("pet_downscale", 1) or 1),
                "vis": True,
                "weights": w,
                "player_id": p.get("player_id") or "",
                "weights_player_id": self._yolo_player.get(w, ""),
                "conf": 0.40,
                "imgsz": p.sec("train").get("imgsz", 960),
                "device": p.sec("train").get("device", "0"),
                "only": self._only.get("pet") or [],
            }

        # 「YOLO 标注」单独跑：只做模型这一遍（往 labels_auto 追加框），
        # **不重跑模板匹配** —— 流程由用户自己安排。
        if self._yolo_only:
            self._yolo_only = False
            w = self.cmb_yolo.currentData()
            if not w:
                raise ValueError("还没有训练好的模型，先在其它项目完成 ⑦ 训练")
            from tools.yolo_augment import run_yolo_augment
            target = self._label_target
            return run_yolo_augment, {
                "weights": w,
                "frames": str(p.frames),
                "out": str(p.dir_of("labels_auto")),
                "mode": target,                  # "mob" / "player"，与按钮文字一致
                "player_id": p.get("player_id") or "",
                "weights_player_id": self._yolo_player.get(w, ""),
                "conf": 0.40,
                "imgsz": p.sec("train").get("imgsz", 960),
                "device": p.sec("train").get("device", "0"),
                "only": self._only.get(target) or [],
            }

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
                    # ⭐⭐ 框到可见部分（用户 2026-10-06 ✓ "**所有的匹配都需要**这个参数"）——
                    #   ⚠ 玩家这条是**最后一段补上的** ✗：原来它走 `player_locator`（另一套定位 ✓）
                    #   ⇒ 这一格在它上面**一直没接**（卡片没传 → `run_detect_player` 没转发 →
                    #   `PlayerLocator` 连能力都没有 ✓ 2026-10-06 三段一起补 ✓）。
                    #   ⚠ **与怪物/宠物/掉落共用同一份实现**（`perception/visible_box.py` ✓）
                    #   和**同一个设置值**（`_visible_params` 一处口径 ✓）。
                    **self._visible_params(sec),
                    # ⭐ 2026-10-05：玩家这一段自己的另外 4 个（用户："缺功能的就补功能" ✓）——
                    #   能力先在 `perception/player_locator.py` 里补出来的 ✓
                    #   ⚠ 默认值**故意与怪物不同**（区分度 0 ✓ 最大模板帧 0 = 不截 ✓
                    #     每帧最多几个框 1 ✓ 降采样 1 ✓ 见 `build_params` 里那段说明 ✓）。
                    "min_distinct": float(sec.get("player_min_distinct", 0.0) or 0.0),
                    "per_mob": int(sec.get("player_per_mob", 0) or 0),
                    "max_peaks": int(sec.get("player_max_peaks", 1) or 1),
                    "downscale": float(sec.get("player_downscale", 1) or 1),
                    # ⭐⭐ **这条目手工挑的模板帧**（用户 2026-10-05 ✓ 四段统一）——
                    #   ⚠⚠ **只递 `player:` 那几把钥匙** ✗：键是 `<target>:<id>` ✓，
                    #   往下递时**只能带这一段自己的** ✓（怪物链那边是**每个键都剥前缀**的 ✗
                    #   见 `detect_mobs.py:655` ✓ ⇒ 把掉落/宠物的键混进去会被当成怪 id ✗）。
                    "frames_sel": {k: list(v) for k, v in (self._frames_sel or {}).items()
                                   if str(k).startswith("player:")},
                    "vis": True,
                    "vis_dir": str(p.vis) + "_player",
                },
            }
            # 粗定位（**自动**，不是开关）：选了权重就先用它圈出玩家区域，
            # 模板匹配只在区域里找 —— 实测 11 秒/帧 → 零点几秒，还能避开画面里
            # 长得像玩家、位置又固定的东西（实测：地图上垂下来的一根绳子分数
            # 0.851，比真玩家还高）。没圈到的帧仍走全图，漏检不会被漏掉；
            # 模型角色与当前角色不一致时 player_rois 会自己放弃并打日志。
            w = self.cmb_yolo.currentData()
            if w:
                params["detect"]["yolo"] = {
                    "weights": w,
                    "player_id": p.get("player_id") or "",
                    "weights_player_id": self._yolo_player.get(w, ""),
                    "conf": 0.25,       # 粗定位宁可多圈一点（漏检有全图兜底）
                    "imgsz": p.sec("train").get("imgsz", 960),
                    "device": p.sec("train").get("device", "0"),
                }
            # 选帧：只处理勾中的帧（[] = 全选不筛）
            params["detect"]["only"] = self._only.get("player") or []
            return run_detect_player_augmented, params

        from tools.yolo_augment import run_detect_mob_augmented
        params = {
            "detect": {
                "mobs": p.get("mobs") or [],
                "sprites": str(wzexport.sprite_dir_path()),
                "frames": str(p.frames),
                "out": str(p.dir_of("labels_auto")),
                "vis_dir": str(p.vis),
                # ⭐⭐ 框到可见部分（用户 2026-10-05 ✓ 一处口径见 `_visible_params` ✓）
                **self._visible_params(sec),
                "scale": self._sprite_scale(p),      # = `p.get("scale") or 1.12`（一处口径 ✓）
                "mob_scales": p.get("mob_scales") or {},
                "downscale": sec.get("downscale", 1),
                "thresh": sec.get("thresh", 0.90),
                "min_distinct": sec.get("min_distinct", 0.06),
                "per_mob": sec.get("per_mob", 20),
                "max_peaks": sec.get("max_peaks", 4),
                # ⭐⭐ 怪物这块手工挑的**模板帧**（用户 2026-10-05 ✓）—— 键 `mob:<怪id>` ✓
                #   由 `detect_mobs.py:655` 剥前缀之后用 ✓。
                #   ⚠⚠ **只递 `mob:` 的键** ✗（那条链对**每个**键都剥前缀 ✗ 混进别的键就串味 ✗）。
                "frames_sel": {k: list(v) for k, v in (self._frames_sel or {}).items()
                               if str(k).startswith("mob:")},
                "vis": True,
            },
        }
        # 选帧：只处理勾中的帧（[] = 全选不筛）
        params["detect"]["only"] = self._only.get("mob") or []
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
        # ---- **换个颜色把它拎出来**（用户 2026-09-27 要求）----
        # 别的卡片的「查看」是"看结果"，这一个不一样：它是**进入人工环节的入口**——
        # 点下去是**去干活**（翻帧、修框、剔除坏帧），而这一步"决定模型的上限"（见 hint ✓）。
        # 用 `theme.ENTRY_BTN_QSS` = 和「寻路编辑器」**同一个角色**（色值只写一处 ✓）：
        # 那套配色的角色划分是 蓝=运行/主按钮、靛蓝=**打开一个干活的地方**、绿=通过、
        # 橙=提醒、红=错误 —— 用主按钮蓝会和「运行」混在一起 ✗（那张卡片上恰好没有
        # 「运行」，蓝色会变成"这里是主操作"的错觉：它的主操作其实是**人去看**✓）。
        self.btn_view.setStyleSheet(theme.ENTRY_BTN_QSS)

    def build_params(self, form):
        note = QLabel(
            "质检台能做：\n"
            "  · 翻看带框原图（滚轮缩放、中键平移）\n"
            # ⚠ 这里的清单**必须与 `gui/review.py::ReviewPanel.FILTERS` 对齐** ✗
            #   （2026-10-09 改 ✓）：原来还写着「新框出现 / 旧框消失」—— 那两条筛选**早已移除**
            #   （靠相邻帧 IoU 配对推断误检/漏检，目标一直在动 ⇒ 配出来的"新增/消失"多是伪影 ✓
            #   见 `review.py` 那段说明 ✓）⇒ 文案不改就是**误导**：人会去下拉里找不存在的项 ✗。
            "  · 筛异常帧：空帧 / 多框 / 玩家框丢失或重复 / 有掉落物 / 有宠物\n"
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
                   minimum=0.05, maximum=0.5, decimals=2, step=0.05,
                   tip="验证集（val）占全部帧的比例 —— 是比例不是张数（0.2 = 20%）。\n  · 范围 0.05~0.5（填不进 0 或负值）；\n  · 老帧沿用上次划分，只有新帧按它补进 val（是目标比例，不是硬性）；\n  · 调大 → val 更可信，但训练样本变少；调小则反之。"
        )
        self.field(form, "min_boxes", "最少框数", "int", 1, minimum=0, maximum=100)
        self.tip("min_boxes", 
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
# 模型版本（⑦ 训练 / ⑧ 验证 / ⑨ 迭代 共用）
# ══════════════════════════════════════════════════════════════
#
# ⑦ 给每次训练起名 detect_vN（见 TrainCard._next_run_name），还往运行目录里写
# 一份 run.json（见 perception/train.py）。所以「第几版、跑出多少 mAP」是可以
# 从产物里客观读出来的 —— 下面这几个函数专门读它：
#     · ⑦ 显示历次版本的指标，并和上一版比一下；
#     · ⑧ 让你直接挑某一版去验证，换版本对比效果；
#     · ⑨ 用最新那版做迭代。
#
# **不要按路径字符串排序取最后一个**：detect_v10.pt < detect_v9.pt，版本一多
# 就会静默挑错模型（v10 出来了还在用 v9）。一律按版本号排。


def _ver_of(name):
    """`detect_v3` → 3；不是这个格式返回 None。"""
    tail = str(name).rsplit("v", 1)[-1]
    return int(tail) if tail.isdigit() else None


def _read_json(path):
    """读一份 json；不在、坏了、不是预期结构都返回 None（调用方自己回退）。"""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _frames_newer_than_dataset(project, data_yaml):
    """`frames/` 里有几张**比数据集新**的帧（= 还没进数据集的）→ N（数不出来就 0 ✓）。

    **为什么用 mtime 而不是比张数**：⑥ 每次都会重写 `data.yaml` ✓，所以"比它新"= "上次准备
    数据集之后又补了帧" ✓；而比张数天然对不上 —— ⑥ 会按 `min_boxes` 过滤掉一部分帧 ✗。
    ⚠ 数不出来（目录不在 / 权限问题）**一律返回 0**：这是"挡人"的前置检查，
      宁可漏挡也不能因为数不出来就把人锁在门外 ✗。
    """
    try:
        t = data_yaml.stat().st_mtime
        fr = project.dir_of("frames")
        if not fr.is_dir():
            return 0
        return sum(1 for f in fr.glob("*") if f.is_file() and f.stat().st_mtime > t)
    except Exception:                       # noqa: BLE001
        return 0


def _run_dirs(project):
    """项目里训过的版本 [(版本号, 名字, run 目录, run.json 内容)]，新的在前。"""
    got = []
    for d in project.dir_of("runs").glob("detect_v*"):
        if not (d / "weights" / "best.pt").exists():
            continue
        v = _ver_of(d.name)
        got.append((v if v is not None else -1, d.name, d,
                    _read_json(d / "run.json")))
    got.sort(key=lambda it: (-it[0], it[1]))
    return got


def _model_files(project):
    """可以拿去推理的权重 [(名字, 路径, 版本号, run.json 内容)]，新的在前。

    两个来源：归档到 models/ 的（正式产物，实时层用的就是它）+ runs/ 里还没
    归档的 best.pt。同名以 models/ 里的为准（那份才是归档产物）。
    名字解析不出版本号的（手工塞进来的权重）排在后面，但仍然可选。
    """
    got = {}
    runs = project.dir_of("runs")
    for d in runs.glob("detect_v*"):
        b = d / "weights" / "best.pt"
        if b.exists():
            got[d.name] = (b, _read_json(d / "run.json"))
    for f in project.dir_of("models").glob("*.pt"):
        got[f.stem] = (f, _read_json(runs / f.stem / "run.json"))

    def key(it):
        try:
            mt = it[1].stat().st_mtime
        except OSError:
            mt = 0.0
        return (1 if it[2] is not None else 0, it[2] or 0, mt)

    items = [(name, path, _ver_of(name), info)
             for name, (path, info) in got.items()]
    items.sort(key=key, reverse=True)
    return items


def _newest_weights(project):
    """最新那版的权重路径（一版都没有就返回空串）。"""
    items = _model_files(project)
    return str(items[0][1]) if items else ""


def _fmt3(v):
    """指标格式化：取不到就「—」，别在卡片上显示 None。"""
    try:
        return "%.3f" % float(v)
    except (TypeError, ValueError):
        return "—"


def _ver_label(path):
    """权重路径 → 版本名：`models/detect_v3.pt` 和
    `runs/detect_v3/weights/best.pt` 都得到 `detect_v3`。

    （production 侧同规则的 `_wlabel` 在 perception/predict.py —— 那边写历史、
    这边读产物，各自留一份免得让界面去 import 推理模块。）
    """
    p = Path(str(path or ""))
    if p.stem == "best" and p.parent.name == "weights":
        return p.parent.parent.name
    return p.stem or "?"


# ══════════════════════════════════════════════════════════════
# ⑦ 训练
# ══════════════════════════════════════════════════════════════
class TrainCard(StepCard):
    def __init__(self):
        super().__init__(7, "train", "训练", hint="训练 YOLO 检测器，每轮指标实时刷新")
        # 「查看报告」（2026-09-26 用户要求）：把**本项目所有权重**和每版的评估 + 建议
        # 放进一个弹窗（原来那段建议直接贴在结果区，占地方、也只能看最新一版）。
        # 卡片只负责开窗 —— 权重枚举（`_model_files`）与建议文字（`perception.metrics`）
        # 都在外面，各自能脱离 Qt 自检。
        foot = self.layout().itemAt(self.layout().count() - 1).layout()
        self.btn_report = QPushButton("查看报告")
        self.btn_report.setToolTip(
            "看本项目训练过的每一个权重：点左边一版，右边说它这一版读到什么、"
            "下一步做什么。")
        self.btn_report.clicked.connect(self._open_report)
        foot.addWidget(self.btn_report)
        #: 「基础权重」下拉里**上一次的合法取值**：选了「（自定义 / 浏览…）」之后又取消时，
        #: 靠它把选择还原（别停在那一项上 —— 它不是个真的权重 ✗）。
        self._model_prev = None

    #: 「基础权重」下拉里"自定义"那一项的 userData（选中它会弹文件选择器 —— 不是权重名 ✓）
    MODEL_CUSTOM = "\x00custom"

    def build_params(self, form):
        # ---- 基础权重：**下拉**（2026-09-27 用户要求"整理一下"）----
        # 以前是**文本框**：人根本不知道有哪些能选、该填什么 ✗。候选三组（见 `_fill_model_options`）：
        #   官方预训练 / **各项目已经练好的权重** / 自定义路径。
        # 为什么值得单独做：新开的项目和旧项目**背景、怪都很像**时，直接挑**旧项目那一版**
        # 当起点微调，比从 `yolo26n.pt` 从零学省得多 ✓（模型学的是外观特征，跨图能用 ——
        # ④ 的「YOLO 权重」下拉当初就是为这件事做的 ✓）。
        self.field(form, "model", "基础权重", "combo")
        cmb = self.widgets["model"][0]
        cmb.setMaxVisibleItems(20)
        cmb.setToolTip(
            "拿哪个权重当**起点**训练（决定速度和精度，也决定起步有多好）。\n\n"
            "· **官方预训练**（yolo26n.pt / yolov8n.pt…）：从零学，最慢、也最「干净」；\n"
            "· **各项目练好的**（项目名 / 权重名）：新图和那张图**背景、怪很像**时选它 ——\n"
            "  拿旧模型微调，比从零学快得多也更准 ✓（本项目上一版也在这一组里）；\n"
            "· **（自定义 / 浏览…）**：挑任意 .pt（例如 runs 里某一版 best.pt）。\n\n"
            "n 最小最快（数据少/显存小选它），s/m/l/x 越来越准也越来越慢。\n"
            "本地没有的官方权重会自动联网下载。\n"
            "⚠ 换了基础权重，这一版的 mAP 就不能和上一版直接比了（起点不同 —— "
            "「查看报告」里会写出每版用的基础权重 ✓）。")
        cmb.activated[int].connect(self._on_model_pick)

        self.field(form, "epochs", "轮数", "int", 120, minimum=1, maximum=5000)
        self.tip("epochs", 
            "训练多少轮。数据几百张时 100~200 轮通常够。\n"
            "设大了没关系 —— 连续 40 轮不涨会自动早停。")

        self.field(form, "imgsz", "输入尺寸", "int", 960, minimum=320, maximum=2048)
        self.tip("imgsz", 
            "训练时把画面缩放到多大再喂网络。\n"
            "目标越小、画面越精细，尺寸要越大（960 适合 100px 级目标）。\n"
            "越大越吃显存。")

        self.field(form, "batch", "批大小", "int", 8, minimum=1, maximum=128)
        self.tip("batch", 
            "每步喂几张图。显存越大能开越大，训练越稳。\n"
            "报显存不足（OOM）就调小，比如 4 或 2。")

        self.field(form, "device", "设备", "str", "0")
        self.tip("device", 
            "0 = 第一块 GPU，cpu = 用 CPU（会很慢）。\n"
            "多卡时写 0,1 或 0,1,2。")

        # ⭐⭐ **接着上次跑**（用户 2026-09-29 ✓ 原话："自动找最近一个 run 的 last.pt、崩了
        #    能一键续、并且在续之前先报一句当前显存够不够"）。
        #  · 勾上 ⇒ 自动找本项目 `runs/` 里**最近**一个带 `last.pt` 的 run（按文件 mtime，
        #    不是目录名 —— `detect_v10` 会排到 `detect_v9` 前面 ✗）⇒ 从那个 epoch 接着训 ✓；
        #  · ⚠ **上面那些参数全部不生效**（轮数 / 尺寸 / 批 / 基础权重）—— ultralytics 续训时
        #    只认检查点里那套 ✓。**尤其是"想续训顺便把批改小"是做不到的** ✗：显存不够就得
        #    不勾它、换个批大小从头跑 ✓（上面那句提醒就是为这件事）。
        #  · 训练崩了（比如 cuDNN / OOM）进度**不会丢** —— `last.pt` 每个 epoch 都写 ✓，
        #    回来勾上它再跑一次就行 ✓。
        self.field(form, "resume", "接着上次跑", "bool", False)
        self.tip("resume", 
            "从本项目**最近一次**训练断掉的地方接着训（自动找最后一个 last.pt）。\n\n"
            "· 训练中途崩了（显存不足 / cuDNN 报错）⇒ 勾上它再跑一次，进度不丢 ✓；\n"
            "· ⚠ 勾上之后**上面那些参数都不生效**（轮数 / 尺寸 / 批 / 基础权重都沿用\n"
            "  上次那套）—— 尤其「想续训顺便把批改小」是做不到的，显存不够就得\n"
            "  不勾它、换个批大小从头跑；\n"
            "· 续训前日志里会先报一句**当前显存够不够**，不够会提醒你先关掉\n"
            "  Edge / QQ / 微信 / Steam 这些占显存的程序；\n"
            "· 本项目还没跑过训练、或 runs 里那个 run 被删了 ⇒ 会直接报错告诉你。")

    def _fill_model_options(self, p=None, cur=None):
        """填「基础权重」下拉：官方预训练 + **所有项目**已训好的权重 + 自定义。

        候选来自两处**现成**设施（不新写一份扫描 ✗）：
          · 仓库根目录的 `*.pt` —— 官方预训练（本机实测有 `yolo26n.pt` / `yolov8n.pt` ✓）；
          · `tools.yolo_augment.list_weights()` —— 每个项目 `models/*.pt`（④ 那张卡的
            「YOLO 权重」下拉就是它，**同一处口径** ✓；顺带带出该项目的角色 id ✓）。

        ⚠ **当前值一定要在列表里**（连"老项目里手打过的怪路径"也算一项 ✓）：`set_value`
          是按 userData `findData` 匹配的，不在列表里就**静默不选中** ⇒ 界面显示的和你
          实际拿去训练的会不一致 ✗（这类不一致最难查）。
        """
        w = self.widgets.get("model", (None, None))[0]
        if w is None:
            return
        if cur is None:
            cur = w.currentData()
        root = Path(__file__).resolve().parents[2]      # gui/steps/cards.py → 仓库根
        official = sorted(q.name for q in root.glob("*.pt"))
        if "yolo26n.pt" not in official:
            official.insert(0, "yolo26n.pt")            # 默认值永远要有（没有会联网下载 ✓）
        try:
            from tools.yolo_augment import list_weights
            trained = list_weights()
        except Exception:                               # noqa: BLE001
            trained = []

        w.blockSignals(True)
        w.clear()
        for nm in official:
            w.addItem("%s%s" % (nm, "（默认）" if nm == "yolo26n.pt" else ""), nm)
        if trained:
            w.insertSeparator(w.count())
        # 当前项目的名字：**目录名优先**（`list_weights` 的标签前缀就是目录名 ✓；
        # `Project.name` 读的是 project.yaml 里那格 `name`，没填过就是空的 ⇒ 只认它会漏标 ✗）
        mine = str(getattr(getattr(p, "root", None), "name", "")
                   or getattr(p, "name", "") or "")
        for label, path, pid in trained:
            tail = ("　· 玩家:%s" % pid) if pid else ""
            if mine and label.startswith(mine + " / "):
                tail += "　（本项目）"
            w.addItem(label + tail, path)
        w.insertSeparator(w.count())
        w.addItem("（自定义 / 浏览…）", self.MODEL_CUSTOM)
        if cur and w.findData(cur) < 0:
            # 存着的值不在候选里（手打过的路径 / 已删掉的旧版本…）⇒ 补一项，**别吞掉** ✗
            w.insertItem(0, "（当前填写）%s" % cur, cur)
        i = w.findData(cur) if cur else -1
        if i >= 0:
            w.setCurrentIndex(i)
        w.blockSignals(False)
        self._model_prev = cur

    def _on_model_pick(self, idx):
        """下拉里选了「（自定义 / 浏览…）」⇒ 弹文件选择；选别的只记住它（供取消时还原 ✓）。

        为什么留这个逃生口：下拉**只能选候选**，而"别的机器上训的 / 手头的实验权重"也该
        填得进去 —— 不给口子就等于把原来"随便填"的能力弄丢了 ✗（用户明确要的是"整理"，
        不是"限制"）。
        """
        w = self.widgets.get("model", (None, None))[0]
        if w is None:
            return
        data = w.itemData(idx)
        if data != self.MODEL_CUSTOM:
            self._model_prev = data
            return
        start = str(self._model_prev or "")
        path, _f = QFileDialog.getOpenFileName(
            self, "选一个 .pt 当基础权重",
            (str(Path(start).parent) if start else ""),
            "PyTorch 权重 (*.pt);;所有文件 (*)")
        if not path:
            i = w.findData(self._model_prev)            # 取消 ⇒ 还原（别停在"自定义"上 ✗）
            w.setCurrentIndex(i if i >= 0 else 0)
            return
        if w.findData(path) < 0:
            w.insertItem(w.count() - 1, "（自定义）%s" % path, path)
        w.setCurrentIndex(w.findData(path))
        self._model_prev = path

    def load_from_project(self, p):
        sec = p.sec("train")
        # ⚠ 顺序：**先填候选、再回填取值** —— `set_value("model", …)` 是按 userData 找项，
        # 列表里没有那一项就静默不选中（界面上显示的是别的权重，训练却用存的那个 ✗）。
        self._fill_model_options(p, cur=sec.get("model") or "yolo26n.pt")
        for k in ("model", "epochs", "imgsz", "batch", "device", "resume"):
            self.set_value(k, sec.get(k))

    def sync(self, p):
        p.sec("train").update(
            self.values(["model", "epochs", "imgsz", "batch", "device", "resume"]))

    def _open_report(self):
        """开「训练报告」弹窗（懒导入：弹窗不该在卡片模块里被 import 进来）。"""
        if self.project is None:
            self.set_result("先选一个项目")
            return
        from gui.train_report import TrainReportDialog
        TrainReportDialog(self.project, parent=self).exec_()

    def summarize(self, p):
        """最新一版的指标 + 历次版本 mAP50（全部从产物读，不靠记忆）。"""
        vs = _run_dirs(p)
        if not vs:
            return "未训练"
        _v, name, _d, info = vs[0]
        m = (info or {}).get("metrics") or {}
        line1 = ("%s · mAP50 %s · mAP50-95 %s · %s 轮"
                 % (name, _fmt3(m.get("map50")), _fmt3(m.get("map")),
                    (info or {}).get("epochs", "?")))
        # 加数据/调参重训，最想知道的就是「比上一版有没有变好」
        if len(vs) > 1:
            pm = (vs[1][3] or {}).get("metrics") or {}
            if m.get("map50") is not None and pm.get("map50") is not None:
                line1 += "（比 %s %+.3f）" % (vs[1][1], m["map50"] - pm["map50"])
        # ⚠ **通俗评估 + 建议已经移到「查看报告」弹窗里**（2026-09-26 用户要求"移位"）：
        # 它要按权重**逐个**翻看，贴在结果区既占地方、又只能看到最新那一版。
        # 数据仍是同一份（`runs/detect_vN/run.json`）⇒ 卡片摘要和弹窗不会两处对不上。
        return line1 + "\n历次 mAP50：%s" % " · ".join(self._hist_bits(vs))

    @staticmethod
    def _hist_bits(vs, keep=4):
        """「v3 0.871 · v2 0.842 …」——版本多了只列最近几版，免得卡片撑爆。"""
        bits = []
        for _v, name, _d, info in vs[:keep]:
            m50 = ((info or {}).get("metrics") or {}).get("map50")
            bits.append("%s %s" % (name, _fmt3(m50)))
        if len(vs) > keep:
            bits.append("…（共 %d 版）" % len(vs))
        return bits

    def _history_text(self, p):
        """鼠标停在这张卡片的结果上时显示完整对照表。"""
        vs = _run_dirs(p)
        if not vs:
            return ""
        rows = ["历次训练（新 → 旧）：", ""]
        for _v, name, d, info in vs:
            m = (info or {}).get("metrics") or {}
            if m.get("map50") is None:
                rows.append("%-11s 没有 run.json，指标看 %s/results.csv"
                            % (name, d.as_posix()))
                continue
            rows.append("%-11s mAP50 %s   mAP50-95 %s   P %s   R %s   %s 轮   %s"
                        % (name, _fmt3(m.get("map50")), _fmt3(m.get("map")),
                           _fmt3(m.get("precision")), _fmt3(m.get("recall")),
                           (info or {}).get("epochs", "?"),
                           (info or {}).get("finished_at", "")))
            if (info or {}).get("base"):
                rows.append("%-11s   基础权重 %s" % ("", info["base"]))
        rows.append("")
        rows.append("val 集是固定的（dataset/split.json），所以两版 mAP 可比；"
                    "但画面同源，别当真实效果看。")
        return "\n".join(rows)

    def refresh(self, force=False):
        super().refresh(force)
        self.result.setToolTip(
            self._history_text(self.project) if self.project else "")

    def detect_state(self, p):
        vs = _run_dirs(p)
        if not vs:
            return ("idle", "")
        # 数据集比模型新 → 模型过期，退回未处理（比的是**最新那版**）
        y = p.dataset / "data.yaml"
        best = vs[0][2] / "weights" / "best.pt"
        if y.exists() and y.stat().st_mtime > best.stat().st_mtime:
            return ("warn", "数据集已更新")
        return ("done", "已训练")

    def check_deps(self, p):
        y = p.dataset / "data.yaml"
        if not y.exists():
            return False, "缺少数据集，请先完成 ⑥ 数据集"
        # ⭐⭐ **补了帧但没重跑 ⑥** 要**当场挡住**（用户 2026-09-29 ✓ 原话："我后补的帧数据
        #   训练，它如果早退了那是不是白补了？"）：训练只吃 `dataset/images/{train,val}` 这份
        #   **物理拷贝**（`perception/train.py` 全文不扫 `frames/`、不读 `split.json` ✗）⇒
        #   忘了跑 ⑥ 时那批帧**一张都用不上** ✗✗。原来只有一条"数据集已更新"的弱提示
        #   （纯 mtime 比较，而且**训练完之后**才显示 ✗）⇒ 现在开跑前就挡 ✓。
        #   ⚠ 判据是"**比 data.yaml 新**"（⑥ 每次都会重写它 ✓），不是比张数 —— ⑥ 会按
        #     `min_boxes` 过滤掉一部分帧，张数天然对不上 ✗。
        n = _frames_newer_than_dataset(p, y)
        if n:
            return False, (
                "有 %d 张新帧还没进数据集（frames/ 里有比 data.yaml 更新的帧）——\n"
                "训练只读数据集里那份拷贝，**不会**自动带上它们 ⇒ 直接训练等于这批帧白补。\n\n"
                "先跑一次 ⑥ 数据集（没标注/框太少的会被自动跳过 ✓），然后再训练 ✓。"
                % n)
        return True, ""

    def _next_run_name(self, p):
        """下一个不撞车的运行名：detect_v1 / detect_v2 / …

        **为什么要版本化**：原来固定写死 detect_v1 且 `exist_ok=True` —— 重训一次，
        上一版的 `runs/detect_v1/` 和 `models/detect_v1.pt` 就被**静默覆盖**，
        等于没有历史。而「加了数据到底有没有变好」全靠新旧两版能摆在一起比。
        （实时层是按 mtime 取最新权重，所以新名字会自动被用上，别处不用改。）
        """
        used = set()
        for d in p.dir_of("runs").glob("detect_v*"):
            v = _ver_of(d.name)
            if v is not None:
                used.add(v)
        for f in p.dir_of("models").glob("detect_v*.pt"):
            v = _ver_of(f.stem)
            if v is not None:
                used.add(v)
        n = 1
        while n in used:
            n += 1
        return "detect_v%d" % n

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
            # ⭐ 「接着上次跑」（用户 2026-09-29 ✓）—— `run_train` 见到它 ⇒ 忽略上面那些参数、
            #   改去续训最近一个 run 的 `last.pt` ✓（一并报显存够不够 ✓）
            "resume": bool(sec.get("resume")),
            # 输出落在项目里，不是全局 runs/ —— 项目要能整个拷走
            "project_dir": str(p.dir_of("runs")),
            "name": self._next_run_name(p),     # 不覆盖上一版，留下可对比的历史
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
        # 模型版本放第一位：这张卡片的一半用途就是**换版本对比效果**。
        # 候选项运行时才填（见 _fill_models），因为每训一次就多一版。
        self.field(form, "model", "模型版本", "combo")
        self.tip("model", 
            "用哪一版模型跑这次验证。默认「最新」= 版本号最大的那版（也就是实时层\n"
            "在用的那版）。\n"
            "想比较版本效果：同一批画面分别选 detect_v2 / detect_v3 各跑一次，\n"
            "卡片摘要会把两次的「有检出」并排列出来。\n"
            "选项里附的 mAP50 是那一版在 val 上的成绩（来自训练时写的 run.json）。")

        self.field(form, "source", "画面目录", "path", "", mode="dir",
                   tip="要验证的画面目录，尽量指向训练分布之外的画面（新录的视频等）。\n  · 别指向训练帧 —— 那等于把 val 指标再算一遍，没意义；\n  · 目录里要有 png/jpg，否则报「画面目录里没有 png/jpg」；\n  · 换了目录，摘要里的历次对比不参与比较（只在同一目录内比）。"
        )
        self.tip("source", 
            "验证要指向**训练分布之外**的画面（新录的视频、MapleNecrocer 截图等）。\n"
            "所以这里**不设默认值** —— 必须手动选一个目录，\n"
            "避免误跑成训练帧（那样等于把 val 指标再算一遍，没意义）。")

        self.field(form, "conf", "置信度阈值", "float", 0.30,
                   minimum=0.0, maximum=1.0, decimals=2, step=0.05,
                   tip="只显示置信度高于这个值的框，单位是 0~1 的概率。\n  · 范围 0.0~1.0；填 0 = 不过滤（候选几乎全画出来，误检很多）；\n  · 调低 → 召回高、漏检少，但误检多；调高 → 干净但可能漏掉模糊目标；\n  · 默认 0.30 偏宽松，适合先看模型到底能认出什么。"
        )
        self.field(form, "imgsz", "推理尺寸", "int", 960,
                   minimum=320, maximum=2048,
                   tip="验证推理时把画面缩放到多大的正方形，单位 像素（边长）。\n  · 范围 320~2048，填不进 0；应与训练时的「输入尺寸」大致一致；\n  · 调大 → 小目标更容易检出，但更慢更吃显存；\n  · 调小 → 更快，但太小的目标会被缩没。"
        )
        self.field(form, "limit", "限制张数", "int", 0, minimum=0, maximum=100000)
        self.tip("limit", "只跑前 N 张，0 = 全部。先跑几十张看效果更快。")

    # ---------------- 项目交互 ----------------

    _KEYS = ("model", "source", "conf", "imgsz", "limit")

    def load_from_project(self, p):
        sec = p.sec("verify")
        for k in ("conf", "imgsz", "limit"):
            self.set_value(k, sec.get(k))
        # 不默认 frames：验证的意义就是"跳出训练分布"，
        # 默认指向训练帧等于鼓励用户白验一次。
        self.set_value("source", sec.get("source") or "")
        # 版本下拉必须**先填候选项**再 set_value —— combo 是按 userData 匹配的，
        # 列表还空着的话，存着的版本名会找不到，静默退回第一项。
        self._fill_models(p)
        self.set_value("model", sec.get("model") or "")

    def sync(self, p):
        p.sec("verify").update(self.values(self._KEYS))

    def refresh(self, force=False):
        super().refresh(force)
        # 有结果图才显示「查看结果」，否则点开是空面板
        has = bool(self.project) and bool(list(self.project.dir_of("verify").glob("*.jpg")))
        self.btn_view.setVisible(has)
        # 刚训完新版本时下拉要立刻能选（主窗口跑完任务会统一 refresh 所有卡片）
        self._fill_models(self.project)
        self.result.setToolTip(self._history_text(self.project))

    def _fill_models(self, p):
        """重填版本下拉：「最新」+ 每一版（带上它在 val 上的 mAP50，供比较）。

        显示文字给人看，真正取值走 userData = 版本名。
        当前选中的那版要保住 —— refresh 会被频繁调用，不能一刷新就跳回最新。
        """
        if p is None or "model" not in self.widgets:
            return
        cmb = self.widgets["model"][0]
        cur = cmb.currentData()
        items = _model_files(p)
        cmb.blockSignals(True)
        cmb.clear()
        if not items:
            cmb.addItem("（还没有训练好的模型）", "")
        else:
            for i, (name, _path, ver, info) in enumerate(items):
                m50 = ((info or {}).get("metrics") or {}).get("map50")
                tag = "最新" if i == 0 else (
                    "v%d" % ver if ver is not None else "外部权重")
                if m50 is None:
                    cmb.addItem("%s   （无 mAP 记录 · %s）" % (name, tag), name)
                else:
                    cmb.addItem("%s   mAP50 %.3f   （%s）" % (name, m50, tag), name)
            cmb.insertItem(0, "最新（自动 → %s）" % items[0][0], "")
        i = cmb.findData(cur)
        cmb.setCurrentIndex(i if i >= 0 else 0)
        cmb.blockSignals(False)

    def _weights(self, p):
        """本次要用的权重：下拉选中的那一版；「最新」= 版本号最大的那版。

        不按路径字符串取最后一个：detect_v10.pt < detect_v9.pt，版本一多就会
        静默挑错模型。选中的那版没了（换项目 / 被删）也退回最新，别让运行直接失败。
        """
        items = _model_files(p)
        if not items:
            return ""
        want = self.value("model") if "model" in self.widgets else ""
        for name, path, _ver, _info in items:
            if name == want:
                return str(path)
        return str(items[0][1])

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

    # ---------------- 历次验证：换版本对比 ----------------

    @staticmethod
    def _history(p):
        """历次验证记录（perception/predict.py 每跑一次追加一条）。"""
        if p is None:
            return []
        got = _read_json(p.dir_of("verify") / "history.json")
        return got if isinstance(got, list) else []

    def _last_label(self, p, st):
        """这次结果是哪一版跑的：以 stats.json 里记的权重为准（它就是本次产物）。

        不能直接拿历史最后一条 —— 老产物没写历史，历史又可能被手改过；
        stats.json 是这次跑出来顺手记的，最可靠。
        """
        lbl = _ver_label(st.get("weights")) if st.get("weights") else ""
        if lbl and lbl != "?":
            return lbl
        h = self._history(p)
        return (h[-1].get("weights") or "") if h else ""

    def _compare_line(self, p, st):
        """同一批画面上各版本「有检出」的对比（新 → 旧，最多 4 次）。

        **只在同一个画面目录里比**：换了目录，帧数、怪物密度都不一样，摆在一起
        看反而误导，所以按 source 过滤。也**不加涨跌箭头** —— 有检出变高不等于
        变好（误检也会让它涨），得连着图看。
        """
        h = [r for r in self._history(p) if r.get("source") == st.get("source")]
        if len(h) < 2:
            return ""
        bits = []
        for r in h[-4:][::-1]:
            n = r.get("frames", 0) or 1
            bits.append("%s %.0f%%" % (r.get("weights") or "?",
                                       100.0 * (r.get("frames_with", 0) or 0) / n))
        return "同批画面：%s（新 → 旧）" % " · ".join(bits)

    def _history_text(self, p):
        """鼠标停在这张卡片的结果上时显示完整对照表。"""
        h = self._history(p)
        if not h:
            return ""
        rows = ["历次验证（新 → 旧）。「有检出」= 至少检出一个框的帧占比。", ""]
        for r in h[-10:][::-1]:
            n = r.get("frames", 0) or 1
            rows.append("%-11s %4d 帧  %5d 框   有检出 %3.0f%%   空帧 %3d   %s"
                        % (r.get("weights") or "?", r.get("frames", 0),
                           r.get("boxes", 0),
                           100.0 * (r.get("frames_with", 0) or 0) / n,
                           r.get("zero_frames", 0), r.get("at", "")))
            rows.append("%-11s   画面目录 %s（置信度 %.2f）"
                        % ("", r.get("source") or "?", r.get("conf_thr") or 0))
        rows.append("")
        rows.append("摘要行里的对比只统计同一个画面目录的几次；换了目录不参与比较。")
        return "\n".join(rows)

    def summarize(self, p):
        st = self._read_stats(p)
        if not st:
            return "未验证"
        n = st.get("frames", 0) or 1
        lbl = self._last_label(p, st)
        head = ("%s%d 帧 / %d 框 · 有检出 %.0f%% · 空帧 %d"
                % (lbl + " · " if lbl else "", st.get("frames", 0),
                   st.get("boxes", 0),
                   100.0 * st.get("frames_with", 0) / n, st.get("zero_frames", 0)))
        cmp_line = self._compare_line(p, st)
        return head + ("\n" + cmp_line if cmp_line else "")

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
        self.tip("conf", 
            "只有模型置信度高于这个值的框才会被写进新标注。\n"
            "调低 → 标注更全，但模型的误检也会一起写进去；\n"
            "调高 → 更保守，改动更小。")

        self.field(form, "keep_empty", "无检出时保留原标注", "bool", True)
        self.tip("keep_empty", 
            "模型在某帧什么都没检出时，是否沿用原标注。\n"
            "建议勾选：「模型没检出」更可能是模型的问题，"
            "不是画面里真没怪。")

        self.field(form, "limit", "限制张数", "int", 0, minimum=0, maximum=100000,
                   tip="只迭代前 N 张（按文件名排序）。\n  · 0 = 全部（先跑几十张试效果更快，满意了再放开成 0）；\n  · 值大于总张数等于全跑，不会报错；\n  · 只影响本次写进 labels_iter/ 的帧，不动 labels/ 里人工修正过的标注。"
        )

    _KEYS = ("conf", "keep_empty", "limit")

    def load_from_project(self, p):
        sec = p.sec("iterate")
        for k in self._KEYS:
            self.set_value(k, sec.get(k))

    def sync(self, p):
        p.sec("iterate").update(self.values(self._KEYS))

    @staticmethod
    def _weights(p):
        """用最新的那版权重。按版本号取，不按路径字符串 —— `detect_v10.pt` <
        `detect_v9.pt`，字符串排序会让 v10 出来之后还在用 v9（详见模块顶部）。"""
        return _newest_weights(p)

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
