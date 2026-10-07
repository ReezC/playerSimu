"""**背包窗格**（玛普尔风格的道具格子）+ **悬停信息弹窗**（用户 2026-10-04 ✓ 原话：\"自动标注
卡片上，要标注的掉落物能不能以背包窗格的形式展示项目，鼠标指着的时候显示信息弹窗
（MapleNecrocer 应该有相关实现）\"）。

参考实现就在本机：`E:\\MyPrograms\\MapleRogue\\MapleNecrocer\\MapleNecrocer\\CharaSimControl\\`
（用户指的路 ✓ 两处规格照抄，别自己发明 ✗）：

  · **格子** —— `AfrmItem.cs:624`
      `if (new Rectangle(getItemIconOrigin(idx), new Size(33, 33)).Contains(point))`
    ⇒ **一格 33×33**（图标画在格子里、格子有 1px 边框，悬停/选中各有高亮 ✓）。
  · **信息弹窗** —— `ItemTooltipRender.cs`
      · 宽 **290**（`new Bitmap(290, picHeight)` ✓）；
      · 背景 `GearBackBrush` 铺 `(2, 2, 286, H-4)` ⇒ **深色框 + 2px 边**（`GetBorderPath` ✓）；
      · 图标格 `FillRectangle(GearIconBackBrush, 14, iconY, 68, 68)` ⇒ **68×68** ✓，
        图标 **×2 放大**（`GearGraphics.EnlargeBitmap` ✓）居中画进去；
      · 右侧 **x=145** 起依次写：**名称**（`ItemNameFont` 白 ✓）→ 属性（橙）→ 等级要求
        → **描述**（`sr.Desc + sr.AutoDesc` ✓ 自动换行到 272 宽 ✓）。

**为什么做成**一个控件（`IconGrid`）**而不是各处各画一套**：卡片（"要标注的掉落物" ✓）与
选物弹窗（已选 / 候选 ✓）要的是同一件事 —— 格子、高亮、悬停弹窗、顺序 ✓；各写一套必然
"有一处忘了标名字/id" ✗（这类不一致正是本仓库反复吃亏的地方 ✓）。

⚠ **我们与 MapleNecrocer 差在哪**（如实说 ✓ 别假装一样 ✗）：
  · 它的弹窗背景/边框是 WZ 里那张 tooltip 贴图 ✓，我们用近似的纯色渐变 ✓；
  · 它右侧有"属性 / 等级要求"（装备才有 ✓），掉落物这条**只有 名称 / id / 类别 / 几帧** ✓；
  · 它还有**描述**（`String.wz` 的 `desc` ✓），我们的 `drops.json` **没有这一项** ✗
    ⇒ 有 `desc` 字段就显示、没有就不显示（别编 ✗）。
"""

from PyQt5.QtCore import QPoint, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QPainter, QPixmap
from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QLabel, QListWidget,
                             QListWidgetItem, QVBoxLayout, QWidget)

from core import wzexport

#: 一格里的图标边长（MapleNecrocer `AfrmItem.cs:624` 的 33×33 ✓）
SLOT_ICON = 33
#: 一格占的横竖总尺寸（33 图标 + 边框/间隙 ✓）
SLOT_GRID = 37
#: 信息弹窗的宽（`ItemTooltipRender.cs` 的 290 ✓）
TIP_WIDTH = 290
#: 弹窗里那块图标格的边长（`FillRectangle(…, 68, 68)` ✓）
TIP_ICON = 68

#: 格子外观：深色底 + 1px 边框；悬停提亮；选中描黄边（同 Maple 的"指到就发光 / 选中带框" ✓）
_GRID_QSS = """
QListWidget#IconGrid {
    background: #2b2b3d;
    border: 1px solid #4a4a63;
    border-radius: 4px;
}
QListWidget#IconGrid::item {
    border: 1px solid #5b5b7d;
    background: #3d3d55;
    border-radius: 2px;
}
QListWidget#IconGrid::item:hover  { border: 1px solid #9a9ac8; background: #4d4d6d; }
QListWidget#IconGrid::item:selected { border: 2px solid #ffd24a; background: #565680; }
"""

#: 信息弹窗外观：深色圆角 + 2px 边（照 Maple 的 `GearBackBrush` + `GetBorderPath` ✓）
_TIP_QSS = """
QWidget#IconTip {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #4a4a6e, stop:1 #2a2a42);
    border: 2px solid #8a8ab8;
    border-radius: 6px;
}
QLabel#IconTipIcon {
    background: #23233a;
    border: 1px solid #7a7aa8;
    border-radius: 3px;
}
QLabel#IconTipTitle { color: #ffffff; font-weight: 700; font-size: 13px; }
QLabel#IconTipBody  { color: #d7d7ea; font-size: 12px; }
"""


def normalize_entry(d, names=None, meta=None):
    """把一处（`{id}` / `{id, name}` / 来自 `list_sprite_drops()` 的完整一份 ✓）统一成
    `{id, name, category, frames, has_img}` ✓（缺的能查就查 ✓，查不到就留空 —— **不许编** ✗）。

    ⚠ `names` / `meta` **由调用方一次性取好传进来**（`drop_name_map()` / `load_drop_meta()`
      每次都要读一遍 JSON ✗ —— 一个格子读一次，335 格就是 335 次 ✗）。
    """
    d = d if isinstance(d, dict) else {"id": str(d or "")}
    did = str(d.get("id") or "").strip()
    out = {"id": did, "name": str(d.get("name") or "").strip(),
           "category": str(d.get("category") or "").strip(),
           "frames": int(d.get("frames") or 1), "desc": str(d.get("desc") or "").strip()}
    # ⭐⭐ **自带图标路径**（用户 2026-10-04 ✓ 宠物那块要的）：宠物/组合外观的"图标"是**算出来的**
    #   （`petlib.compose_pet` 的 `stand0_0.png` ✓）而不是"按 id 查库" ✗ ⇒ 有它就原样带着 ✓
    #   （`entry_icon` 会优先用它 ✓）；掉落物那种不带 ⇒ 老路一字不变 ✓。
    if d.get("icon"):
        out["icon"] = str(d["icon"])
    if did and not out["name"] and names is not None:
        out["name"] = str(names.get(did) or "").strip()
    if did and meta is not None:
        m = meta.get(did) or {}
        out["category"] = out["category"] or str(m.get("category") or "").strip()
        if int(d.get("frames") or 0) <= 0:
            out["frames"] = int(m.get("frames") or 1)
        out["desc"] = out["desc"] or str(m.get("desc") or "").strip()
    if "has_img" in d:
        out["has_img"] = bool(d["has_img"])
    else:
        out["has_img"] = bool(did) and (wzexport.drop_icon_path(did) is not None)
    out["frames"] = max(1, int(out["frames"] or 1))
    # ⭐ **磁盘上实际几帧**（用户 2026-10-04 ✓ 手工删过多余的帧 ⇒ 清单那份会过期 ✗）：
    #   带 `frames_disk` 就用 ✓（`list_sprite_drops` 会给 ✓），没带就现查一次 ✓
    #   （⚠ 只在**没有**这个键时才查：一个格子查一次目录在 4549 项上是可感的 ✓ 见
    #   `drop_frame_counts` ✓；替身/老数据没有那个键 ⇒ 退回现查 ✓ 语义不变 ✓）。
    if "frames_disk" in d:
        out["frames_disk"] = max(0, int(d.get("frames_disk") or 0))
    elif did:
        out["frames_disk"] = max(0, int(wzexport.drop_frame_counts().get(did, 0)))
    else:
        out["frames_disk"] = 0
    # ⭐⭐ **原始条目里多出来的字段，一并带着**（用户 2026-10-04 ✓ 现场原话："现在选中已添加的
    #   宠物，移出选中按钮不能交互"✓ —— 挖下去是**两个**毛病，这是第二个 ✗）：
    #   本函数是**重拼**一个新 dict ✓ ⇒ `combo_entry`（宠物那套 ✓）带的 `pet` / `equips`
    #   **被丢掉了** ✗ ⇒ 卡片从格子里拿回选中项后**认不出"是哪一套"** ✓
    #   （同一只宠物可以配好几套装备 ✓ `cards._pet_del` 正是靠这两个字段配对删的 ✗）
    #   ⇒ 结果是"**戴了装备的那套永远删不掉**"✗（没装备那套恰好因为 `want=[]` 对得上才删得掉 ✓
    #   —— 这也是为什么这个 bug 藏了这么久 ✓）。
    #   ⚠ 只补**本函数没算出来**的键 ✗（上面那些是权威值：`id`/`name`/`icon`/`frames`… ✓
    #     不许被原值覆盖 ✓）。
    for k, v in d.items():
        if k not in out:
            out[k] = v
    return out


def tip_lines(entry):
    """信息弹窗该写的那些行（**纯函数 ⇒ 能单测** ✓）：`[标题, ...明细]` ✓。

    第一行是**标题**（名称 / 或如实说没有名字 ✓），其余是明细（id / 类别 / 几帧 / 描述 /
    警告 ✓）—— 界面按这个顺序画 ✓，用例按这个顺序断言 ✓（只此一处口径 ✓）。
    """
    e = normalize_entry(entry)
    nm = e["name"]
    lines = [nm if nm else "（WZ 里没有名字）"]
    lines.append("id：%s" % (e["id"] or "?"))
    if e["category"]:
        lines.append("类别：%s" % e["category"])
    fr = int(e["frames"] or 1)
    _fd = int(e.get("frames_disk") or 0)
    if _fd and _fd != fr:
        # ⭐⭐ **以磁盘为准，并如实说清差在哪**（用户 2026-10-04 ✓ 现场："我移除了部分的
        #   icon_3.png 以及 meta 数据（因为这个图太小了容易误标）"✓）：清单里的 `frames` 是
        #   **导出那一刻**的数字 ✗ —— 手工删掉多余帧之后它还写着 4 ✓，照着念就是假话 ✗。
        #   ⚠ 标注只用**磁盘上现有的**这几张（`drop_frame_files` ✓）⇒ 这里就该说磁盘那份 ✓。
        # ⚠ 这里是**纯文本**（`IconTip` 的 QLabel 不认 markdown ✗）⇒ 不许写 `**加粗**` ✗
        #   （写了就原样显示成星号 ✓ 本轮差点就这么发出去 ✓）
        lines.append("图标：磁盘上 %d 张（清单写的是 %d 帧 —— 删过帧之后清单会过期，"
                     "标注只用磁盘上这几张 ✓）" % (_fd, fr))
    else:
        lines.append("图标：%s" % ("%d 帧（会动 —— 标注时这几帧都会当模板 ✓）" % fr
                                 if fr > 1 else "1 张静态图"))
    # ⭐ 宠物那条：直说**模板覆盖了哪几个姿态**（用户 2026-10-04 ✓ 要求"站立/走路/跳/饿都要是
    #   匹配对象"✓ —— 信息窗里能自己看到 ✓ 别让人猜 ✗）。
    #   ⚠ 掉落物那种没有 `states` 键 ⇒ 一个字都不多写 ✓（老行为不变 ✓）。
    if str((entry or {}).get("states") or "").strip():
        lines.append("模板姿态：%s（这几套都会当模板 ✓）"
                     % str(entry.get("states")).strip())
    if e["desc"]:
        lines.append(e["desc"])
    if not nm:
        # ⚠ 必须说清"为什么没名字 + 怎么办"✗ 只显一串 id 会让人以为读坏了
        #   （金币整类在 String.wz 里都没有 ✓ 见 SKILL 约定 190 ✓）
        lines.append("（WZ 里就没有名字 —— 可以手工往 datasets/drops.json 里补 ✓）")
    if not e["has_img"]:
        lines.append("⚠ 图库里没有它的图标 ⇒ 先导出图标才标得了 ✓")
    return lines


def entry_pixmap(entry):
    """这一份的图标**原图**（`QPixmap`；没有 ⇒ **空** ✓ 不回退成别的图 ✗）。

    ⭐⭐ **这是"图标从哪来"的唯一出口**（2026-10-04 收 ✓ 用户原话："指着宠物的时候信息弹窗
      没有 icon" ✗）：`entry["icon"]`（= 一个 png 路径 ✓ 宠物/组合外观是**算出来的**
      —— `petlib.compose_pet` ✓ 见 `normalize_entry` ✓）优先 ✓；没有（掉落物那种 ✓）
      才按 id 查掉落物图库 ✓。
    ⛔⛔ **别在别处自己拼路径** ✗：`IconTip.set_entry` 原来就是这么干的 ✗（只走
      `wzexport.drop_icon_path(e["id"])` ✓）⇒ **宠物的 id 在掉落物图库里当然查不到**
      ⇒ 弹窗里那个 68×68 的格子一直空着 ✓（"指着宠物没有 icon"就是这么来的 ✓）。
    """
    p = str((entry or {}).get("icon") or "").strip() or None
    did = str((entry or {}).get("id") or "")
    if p is None:
        p = wzexport.drop_icon_path(did) if did else None
    if p is None:
        return QPixmap()
    return QPixmap(str(p))                       # 坏路径 ⇒ `isNull()` 为真 ⇒ 调用方照样知道 ✓


def entry_icon(entry, px):
    """这一份的图标（没有图 ⇒ **空 QIcon** ✓ 不回退成别的图 ✗）。

    ⭐⭐ 用户 2026-10-05 ✓ 原话：**「每一个项目 grid 的长宽应该是固定的，中间的贴图做
      缩放+居中」** —— 所以这里不是"把原图缩一下就交差"✗，而是：
        ① 先按长边贴合 `px`（`KeepAspectRatio` ✓ Smooth ✓，小图会被放大到贴边 ✓）；
        ② 再画在一张 **`px×px` 的透明画布**上**居中** ✓。
    为什么非要②✗：只 `scaled` 的话，格子里图标的占用尺寸**随精灵长宽比飘** ✗（宽精灵缩成
      一条、矮精灵顶不满 ✓），而 QListWidget 的行高/文本位置是按 `iconSize` 算的
      ⇒ 看上去就是"有的格子鼓、有的格子空、贴图不在正中间" ✗（用户截图就是这症状 ✓）。
      固定画布之后：**每格永久一样大、图永远居中** ✓。
    """
    pm = entry_pixmap(entry)
    if pm.isNull():
        return QIcon()
    px = max(1, int(px))
    scaled = pm.scaled(px, px, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    canvas = QPixmap(px, px)
    canvas.fill(Qt.transparent)                       # 透明底 ⇒ 选中高亮透出来 ✓
    painter = QPainter(canvas)
    painter.drawPixmap((px - scaled.width()) // 2,
                       (px - scaled.height()) // 2, scaled)
    painter.end()
    return QIcon(canvas)


class IconTip(QWidget):
    """**悬停信息弹窗** —— 无边框小窗（照 `ItemTooltipRender.cs` 的 290 宽 / 68 图标 ✓）。

    ⚠ 用 `Qt.ToolTip` 当窗口类型：**不抢焦点、不占任务栏、总在最前** ✓（这正是游戏里那个
      信息窗的行为 ✓，也是我们不用 `QToolTip` 的原因 —— 那个只能显示一行富文本 ✗，
      画不出"图标格 + 名称 + 多行明细" ✓）。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("IconTip")
        self.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_StyledBackground, True)   # 让上面的样式表真画出来 ✓
        self.setFixedWidth(TIP_WIDTH)
        self.setStyleSheet(_TIP_QSS)

        row = QHBoxLayout(self)
        row.setContentsMargins(14, 12, 14, 12)            # 同 Maple 的 14 起始 ✓
        row.setSpacing(12)
        self.lbl_icon = QLabel()
        self.lbl_icon.setObjectName("IconTipIcon")
        self.lbl_icon.setFixedSize(TIP_ICON, TIP_ICON)
        self.lbl_icon.setAlignment(Qt.AlignCenter)
        row.addWidget(self.lbl_icon, 0, Qt.AlignTop)
        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)
        self.lbl_title = QLabel()
        self.lbl_title.setObjectName("IconTipTitle")
        self.lbl_title.setWordWrap(True)
        col.addWidget(self.lbl_title)
        self.lbl_body = QLabel()
        self.lbl_body.setObjectName("IconTipBody")
        self.lbl_body.setWordWrap(True)
        col.addWidget(self.lbl_body)
        col.addStretch(1)
        row.addLayout(col, 1)

    def set_entry(self, entry):
        """填内容（图标 ×2 放大进 68×68 格 ✓、标题 + 明细 ✓）。"""
        e = normalize_entry(entry)
        lines = tip_lines(e)
        self.lbl_title.setText(lines[0])
        self.lbl_body.setText("\n".join(lines[1:]))
        # ⭐⭐ 走**唯一出口** `entry_pixmap`（宠物那种"算出来的图标"也认 ✓）——
        #   原来这里自己拼 `drop_icon_path(e["id"])` ✗ ⇒ 宠物 id 在掉落物图库里查不到
        #   ⇒ 格子一直空着 ✓（用户 2026-10-04：\"指着宠物的时候信息弹窗没有 icon\"✗）。
        pm = entry_pixmap(e)
        if not pm.isNull():
            # 格子里按**放大**画（同 Maple 的 `EnlargeBitmap` ✓），留 6px 边距别贴边 ✓
            self.lbl_icon.setPixmap(pm.scaled(max(1, TIP_ICON - 6), max(1, TIP_ICON - 6),
                                              Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.lbl_icon.setPixmap(QPixmap())            # 空着（明细里会写明没有图 ✓）
        self.adjustSize()
        return self


class IconGrid(QListWidget):
    """**背包窗格**：`IconMode` 网格，一格一个图标（照 `AfrmItem.cs` 的 33×33 ✓）。

    ⚠ **不是列表**（`ListMode` ✗）—— 用户 2026-10-04 ✓ 要的就是"背包那种格子" ✓。
    ⚠ 每格**不写文字**（格子里放不下中文 ✗，Maple 也不写 ✓）：名称 / id / 类别 / 几帧
      全在**悬停信息弹窗**里 ✓（`tip_lines` ✓ 那份口径）。
    """

    #: 悬停到某一格（传出它的 id；离开时传 `""` ✓）—— 调用方想额外显示啥就接这个 ✓
    hovered = pyqtSignal(str)
    #: 双击某一格（传出 id ✓）—— 弹窗里"双击加进来"用得上 ✓
    activated_id = pyqtSignal(str)

    def __init__(self, parent=None, slot_icon=SLOT_ICON):
        super().__init__(parent)
        self.setObjectName("IconGrid")
        self._slot_icon = int(slot_icon)
        self._tips_on = True
        self._tip = IconTip(self)

        self.setViewMode(QListWidget.IconMode)          # ⭐ 背包格子 ✓
        self.setFlow(QListWidget.LeftToRight)
        self.setWrapping(True)
        self.setResizeMode(QListWidget.Adjust)
        self.setMovement(QListWidget.Static)
        self.setUniformItemSizes(True)
        self.setIconSize(QSize(self._slot_icon, self._slot_icon))
        self.setGridSize(QSize(SLOT_GRID, SLOT_GRID))
        self.setSelectionMode(QListWidget.SingleSelection)
        self.setEditTriggers(QListWidget.NoEditTriggers)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setStyleSheet(_GRID_QSS)

        self.itemEntered.connect(self._on_item_entered)
        self.itemActivated.connect(self._on_item_activated)
        # ⚠ `itemEntered` **只在"进到某一格"时**发 ✓（移到格子之间的空白**不发** ✗）⇒ 光靠它
        #   会出现"鼠标已经不在任何格子上、信息窗还挂着" ✗（离屏实测过 ✓）。所以再装一道
        #   viewport 过滤：移到空白 ⇒ 收起来 ✓（同游戏里"手指离开格子就消失" ✓）。
        self.viewport().installEventFilter(self)

    def eventFilter(self, obj, ev):
        """viewport 上的鼠标移动：**不在任何格子上 ⇒ 收起信息窗** ✓（见上面那段说明 ✓）。"""
        if obj is self.viewport() and ev.type() == ev.MouseMove and self._tips_on:
            if self.itemAt(ev.pos()) is None:
                self.hide_tip()
        return super().eventFilter(obj, ev)

    # ---------------- 尺寸自适应 ----------------

    def fit_rows(self, min_rows=1, hard=None):
        """把高度调成**内容需要几行就是几行**（至少 `min_rows` 行）⇒ 返回行数 ✓。

        ⭐ 用户 2026-10-04 ✓ 原话："要标注的掉落物、宠物的背包窗格需要自适应有几行
          （至少一行）" ✓。
        ⚠ 为什么原来不行：卡片上两块都写死 `37*3+12`（**固定三行** ✗，见 `cards.py`）⇒
          只加 1~2 项时留一大片空白 ✗、加多了又冒滚动条 ✓。
        ⚠ 列数按**当前宽度**算（`viewport().width() // gridSize().width()` ✓）—— 与
          `IconMode + Wrapping + ResizeMode(Adjust)` 是同一套口径 ✓；宽度还没布局出来
          （0）⇒ 当 1 列 ✓（别除零 ✗）。
        ⚠ **只在算出来的值变了**时才 `setMinimumHeight` ✓：不然每次 resize 都改高度 ⇒
          抖动、甚至和布局互相触发 ✓（`_fit_h` 就是干这个的 ✓）。

        ⭐⭐ `hard`（用户 2026-10-04 ✓ 第 2 条原话："模型训练里的掉落物、宠物自动标注类背包窗格
          没有按选中的项目数量**限定**行数（背包窗格显示区的高度）"）：
          · `None`（默认 ✓）= **保持上次设的那个**（`set_entries` 里调 `fit_rows()` 时不会
            把卡片要的"硬限定"冲掉 ✗）；
          · `True` = **同时钉上限**（`setMaximumHeight(h)` ✓）—— 只给 `min` 是不够的 ✗：
            实测（离屏量 ✓）卡片里的窗格 `minimumHeight` 明明是 41（= 1 行 ✓），可
            **`height()` 恒为 480** ✗ —— 布局把多余空间全塞给了它 ⇒ 显示区一直那么高 ✓
            （用户看到的就是这个 ✓）。钉了上限 ⇒ 内容几行就几行 ✓。
          · `False` = 只钉下限（老行为 ✓ 选择弹窗那种"给块地方随便长"的用它 ✓）。
        """
        gw = int(self.gridSize().width() or (self._slot_icon + 4))
        gh = int(self.gridSize().height() or (self._slot_icon + 4))
        w = int(self.viewport().width() or self.width() or 0)
        cols = max(1, w // max(1, gw))
        rows = max(int(min_rows), -(-int(self.count()) // cols))     # 向上取整 ✓
        # ⚠ 纯算术可能把列数**估多**（⇒ 行数估少 ⇒ 明明放不下却只给那么高 ⇒ 白冒滚动条 ✗）
        #   ⇒ 再按**实际布局**数一遍（`visualItemRect` 的 top 有几个不同值就是几行 ✓），
        #     取两者**较大**的 ✓。取不到（还没布局 / 空）⇒ 就用算术那个 ✓。
        try:
            tops = {self.visualItemRect(self.item(i)).top() for i in range(self.count())}
            rows = max(rows, len(tops))
        except Exception:                        # noqa: BLE001 —— 数不出来就算了 ✓
            pass
        h = rows * gh + 2 * int(self.frameWidth() or 0) + 2
        if hard is not None:
            self._fit_hard = bool(hard)              # ⚠ 记下来：`set_entries`/resize 都照旧 ✓
        _hard = bool(getattr(self, "_fit_hard", False))
        if h != int(getattr(self, "_fit_h", -1)) or _hard != bool(getattr(self, "_fit_max", False)):
            self._fit_h = h
            self.setMinimumHeight(h)
            # ⭐ `hard` ⇒ **上限也钉住**（不然布局会把多余高度全塞进来 ⇒ 显示区恒高 ✗ 见 docstring）
            self.setMaximumHeight(h if _hard else 16777215)
            self._fit_max = _hard
        return rows

    def resizeEvent(self, ev):
        """宽度一变 ⇒ 每行能放几个也变 ⇒ 行数跟着重算 ✓（见 `fit_rows` ✓）。"""
        super().resizeEvent(ev)
        try:
            self.fit_rows()
        except Exception:                        # noqa: BLE001 —— 只是个尺寸调整，别崩 ✗
            pass

    # ---------------- 内容 ----------------

    def set_entries(self, entries):
        """重填格子（`entries` 可以是 `{id}` / `{id, name}` / `list_sprite_drops()` 的完整份 ✓）。

        ⚠ 名字表与清单**只读一次** ✓（335 格逐个读 JSON 会明显卡 ✗ 见 `normalize_entry`）。
        """
        self.hide_tip()
        names = meta = None
        if entries:
            try:
                names, meta = wzexport.drop_name_map(), wzexport.load_drop_meta()
            except Exception:                            # noqa: BLE001
                names, meta = {}, {}
        self.clear()
        for d in (entries or []):
            e = normalize_entry(d, names, meta)
            it = QListWidgetItem()
            it.setData(Qt.UserRole, e["id"])
            it.setData(Qt.UserRole + 1, e)
            it.setIcon(entry_icon(e, self._slot_icon))
            # 纯 Qt tooltip 不用（会和我们的信息弹窗叠成两层 ✗）；"没有图"这类话在弹窗里说 ✓
            self.addItem(it)
        # ⭐ 填完就**按内容自适应高度**（用户 2026-10-04 ✓ 见 `fit_rows` ✓）—— 卡片上
        #   掉落物/宠物那两块靠它；空表 ⇒ 也是 1 行高 ✓（不会塌成 0 ✓）。
        try:
            self.fit_rows()
        except Exception:                            # noqa: BLE001 —— 尺寸而已，别崩 ✗
            pass
        return self

    def entries(self):
        return [self.item(i).data(Qt.UserRole + 1) for i in range(self.count())]

    def ids(self):
        return [self.item(i).data(Qt.UserRole) for i in range(self.count())]

    def entry_at(self, row):
        it = self.item(int(row))
        return it.data(Qt.UserRole + 1) if it is not None else None

    def current_entry(self):
        return self.entry_at(self.currentRow())

    def current_id(self):
        it = self.currentItem()
        return str(it.data(Qt.UserRole) or "") if it is not None else ""

    def set_current_id(self, drop_id):
        """按 id 选中（找不到 ⇒ 不选 ✓ 返回 False ✓ —— 别静默选错一个 ✗）。"""
        for i in range(self.count()):
            if str(self.item(i).data(Qt.UserRole) or "") == str(drop_id or ""):
                self.setCurrentRow(i)
                return True
        return False

    def set_tips_enabled(self, on):
        """关掉悬停弹窗（自检/离屏时省事 ✓；关掉同时把已显示的收起来 ✓）。"""
        self._tips_on = bool(on)
        if not self._tips_on:
            self.hide_tip()

    # ---------------- 悬停信息窗 ----------------

    def _on_item_entered(self, item):
        if not self._tips_on or item is None:
            return
        self.show_tip(item)

    def _on_item_activated(self, item):
        if item is not None:
            self.activated_id.emit(str(item.data(Qt.UserRole) or ""))

    def show_tip(self, item=None):
        """把信息弹窗贴在**这一格右边**显示（照游戏里"鼠标指着就弹在格子旁" ✓）。"""
        item = self.currentItem() if item is None else item
        if item is None:
            return None
        self._tip.set_entry(item.data(Qt.UserRole + 1))
        r = self.visualItemRect(item)
        pos = self.viewport().mapToGlobal(QPoint(r.right() + 6, r.top() - 4))
        scr = QApplication.desktop().availableGeometry(self)
        pos.setX(min(pos.x(), scr.right() - self._tip.width() - 4))
        pos.setY(min(max(pos.y(), scr.top() + 4), scr.bottom() - self._tip.height() - 4))
        self._tip.move(pos)
        self._tip.show()
        self.hovered.emit(str(item.data(Qt.UserRole) or ""))
        return self._tip

    def hide_tip(self):
        try:
            self._tip.hide()
        except Exception:                                # noqa: BLE001
            pass
        self.hovered.emit("")

    def tip_visible(self):
        return bool(self._tip.isVisible())

    def tip_text(self):
        """当前弹窗里的全部文字（用例用 ✓ —— 弹窗**看得见的内容**就是这些 ✓）。"""
        return "\n".join([self._tip.lbl_title.text(), self._tip.lbl_body.text()]).strip()

    # ---------------- 事件（滚/点/离开都要收起来 ✓ 同游戏 ✓）----------------

    def leaveEvent(self, ev):
        self.hide_tip()
        super().leaveEvent(ev)

    def hideEvent(self, ev):
        self.hide_tip()
        super().hideEvent(ev)

    def mousePressEvent(self, ev):
        self.hide_tip()
        super().mousePressEvent(ev)

    def wheelEvent(self, ev):
        self.hide_tip()
        super().wheelEvent(ev)
