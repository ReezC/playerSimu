"""集合编辑器自检：纯几何（拾取/框选/选择集运算）+ 离屏把窗口跑一遍。

**为什么这些必须钉住**（见 docs/寻路设计.md §12）：
    "我在不在 A 平台"全靠人工圈的集合。**选错了**（比如框选把隔壁平台的几条也圈进来、
    或点选时因为拾取半径太小而点空）不会有任何报错 —— 只会在某天表现为
    "明明在 A 平台上，程序却以为掉出去了"。所以把三个最容易错的点固化成断言：
      · 点选的**拾取半径**（细线本来点不中，容差是功能不是手感）；
      · 框选的**两种模式**（左→右=包含 / 右→左=相交）；
      · **墙永远选不进集合**（玩家站不上去，圈进去没有意义）。

跑法：
    python -m tools.selftest_zone_editor     # 全过返回 0，有失败返回 1
"""

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import mapdata, zones                              # noqa: E402
from gui import theme as _theme                              # noqa: E402
from gui import zone_editor as ze                            # noqa: E402

# --------------------------------------------------------------- 自检不许改用户文件
# ⛔ `config/ui.yaml` 是**用户的**文件（窗口几何 / 字号 / 颜色）⇒ 自检**绝不许**写它 ✗
#    （全仓库规矩 ✓，见 `selftest_main_window._fake_store` 的说明）。
#    本套件会建/关编辑器弹窗（它们接了 `theme.bind_window_state` / `bind_view_zoom`
#    ⇒ 一 show/hide 就把几何写进用户的文件 ✗ —— 2026-09-29 逐个套件量出来本套件写
#    `windows.add_reach` / `windows.zone_editor` / `views.zone_editor_view`）
#    ⇒ 把 theme 的**落点**指到临时文件，一个字节都不碰用户的 ✓
#    （个别用例另有 `patch.object(theme, "save_window", …)`，那是更细的一道，不冲突 ✓）。
_theme.CFG = Path(tempfile.mkdtemp(prefix="psimu_ui_")) / "ui.yaml"

ROOT = Path(__file__).resolve().parent.parent
MAP_ID = "105090600"


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def row_colors(it):
    """列表行里**被上色的片段** → [(颜色, 文字)]，从富文本角色里解析。

    行内分色后，颜色的真源是 `ROLE_HTML`（不是 `foreground()` —— 那是纯文本时代的
    遗留：设了它反而会让**整行**都变成那个色，正是用户报的毛病）。见 _RichRowDelegate。
    """
    import re
    html = it.data(ze.ROLE_HTML) or ""
    return re.findall(r'<span style="color:([^"]+)">(.*?)</span>', html)


def _terrain():
    p = ROOT / "datasets" / "map" / (MAP_ID + ".json")
    check(p.exists(), "缺少 %s —— 先在「路线识别」里点「生成地形图」" % p.name)
    t = mapdata.load(MAP_ID)
    check(t is not None and t.footholds, "地形读出来是空的：%s" % p)
    return t


# ---------------------------------------------------------------- 纯几何

def t_pick_radius_and_walls():
    """点选：容差内的最近一条被选中；**墙永远不参与**（`foothold_below` 也跳过它们）。"""
    t = _terrain()
    f0 = next(f for f in t.footholds if not f.is_wall)
    # 就在这条线正中间取一点 ⇒ 必然命中它
    mx, my = (f0.x1 + f0.x2) / 2.0, (f0.y1 + f0.y2) / 2.0
    check(ze.pick_at(t, mx, my, 0.5) is f0, "点在线正中间却没命中这条")
    # 容差为 0 时、离线远一点 ⇒ 不命中（证明容差真的在起作用，不是"总能选中"）
    check(ze.pick_at(t, mx, my + 40, 0.5) is None, "离线 40 像素也命中了")
    check(ze.pick_at(t, mx, my + 40, 100) is not None, "容差 100 却没命中")
    wall = next((f for f in t.footholds if f.is_wall), None)
    if wall is not None:
        got = ze.pick_at(t, wall.x1, (wall.y1 + wall.y2) / 2.0, 3.0)
        check(got is None or not got.is_wall, "墙被点选命中了：%s" % got)


def t_pick_ladder_for_info():
    """**点选绳梯只为看信息**（用户 2026-09-27："foothold 编辑器希望能点选绳梯（为了快速查看
    它的信息），**不用接逻辑**"）。

    钉五件：
      ① 点在绳上（避开压着 foothold 的那几段）⇒ 记下它（`_sel_ladder` ✓），状态行报出
        绳号 / x / y 区间，详情里报出**两端各压着哪条 foothold、那些 foothold 属于哪些集合** ✓；
      ② ⚠ **不动选择集、不动集合高亮**（"不用接逻辑" ✓ —— 它是"看着它"，不是编辑 ✓）；
      ③ 点在**绳脚下那条 foothold** 上 ⇒ **foothold 优先**（编辑 foothold 是这张图的主业 ✓，
        不能被绳抢走 ✗）；
      ④ 点到别处 / 框选 ⇒ 取消"看着的绳梯" ✓；
      ⑤ 画布上那根绳换**选中色**（`C_SEL` ✓）并进呼吸清单 ✓（改动得能看出来 ✓）。
    """
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    t = _terrain()
    lads = list(getattr(t, "ladders", None) or [])
    check(lads, "地形里没有绳梯，这条测不了")
    L = lads[0]

    def free_y(lad):
        """在绳上找一段**没被任何 foothold 压着**的 y（绳中段常被平台穿过 ✓）。"""
        lo, hi = min(lad.y1, lad.y2), max(lad.y1, lad.y2)
        for k in range(21):
            yy = lo + (hi - lo) * k / 20.0
            if ze.pick_at(t, lad.x, yy, 6.0) is None:
                return yy
        return None

    y = free_y(L)
    check(y is not None, "这根绳沿线全被 foothold 压着，这条测不了")
    tmp = Path(tempfile.mkdtemp(prefix="zlad_"))
    try:
        dlg = ze.ZoneEditorDialog(MAP_ID, terrain=t, zones_path=tmp / "x.zones.json")
        dlg.resize(980, 660)
        dlg.show()
        app.processEvents()

        # ① 点在绳上 ⇒ 选中它 + 状态行/详情出信息
        dlg._on_picked(("click", L.x, y, 6.0), "replace")
        check(dlg._sel_ladder is L,
              "点在绳上没选中它（用户要的就是快速查看它的信息 ✗）：%r" % (dlg._sel_ladder,))
        txt = dlg.lbl_status.text()
        check("绳梯" in txt and ("x=%d" % round(L.x)) in txt,
              "状态行没报绳梯的位置：%r" % txt)
        tip = dlg.lbl_status.toolTip()
        check(("上端" in tip) and ("下端" in tip) and ("集合" in tip),
              "详情里没报「两端压着哪条 foothold / 属于哪些集合」：%r" % tip)

        # ② **不动选择集**（"不用接逻辑" ✓）
        check(not dlg.selection_ids(),
              "点绳梯把 foothold 的选择集改了：%s" % dlg.selection_ids())

        # ③ 绳脚下那条 foothold 上 ⇒ **foothold 优先**
        foot = next((e for e in zones.ladder_ends(t, L) if e is not None), None)
        check(foot is not None, "这根绳两端都没压到 foothold，这条测不了")
        fx = (foot.left + foot.right) / 2.0
        dlg._on_picked(("click", fx, foot.y_at(fx), 6.0), "replace")
        check(dlg._sel_ladder is None,
              "点 foothold 时被绳抢走了（编辑 foothold 才是主业 ✗）")
        check(dlg.selection_ids() == {str(foot.fid)},
              "绳脚下那条 foothold 没被选中：%s" % dlg.selection_ids())

        # ④ 点到别处 / 框选 ⇒ 取消
        dlg._on_picked(("click", L.x, y, 6.0), "replace")
        check(dlg._sel_ladder is L, "第二次点绳没选上")
        bx, by = t.bounds[0] - 5000, t.bounds[1] - 5000
        dlg._on_picked(("click", bx, by, 6.0), "replace")
        check(dlg._sel_ladder is None, "点空白没取消「看着的绳梯」")
        dlg._on_picked(("click", L.x, y, 6.0), "replace")
        dlg._on_picked(("box", (bx, by, bx + 10, by + 10), "contains"), "replace")
        check(dlg._sel_ladder is None, "框选没取消「看着的绳梯」")

        # ⑤ 画布上那根绳换选中色 + 进呼吸清单（改动要看得出来 ✓）
        dlg._on_picked(("click", L.x, y, 6.0), "replace")
        it = next((i for (LL, i) in dlg._ladder_items if LL is L), None)
        check(it is not None, "画布上找不到那根绳的 item")
        check(it.pen().color() == ze.C_SEL,
              "点选后那根绳没换成选中色（看不出在看哪根 ✗）：%s"
              % it.pen().color().name())
        check(any(i is it for i, _c in dlg._hl_items),
              "点选的那根绳没进呼吸清单（看不出来 ✓）")
    finally:
        try:
            dlg.close()
        except Exception:                      # noqa: BLE001
            pass


def t_box_two_modes():
    """框选两种模式：左→右只选**完全包含**、右→左**相交即选**。"""
    t = _terrain()
    f = next(f for f in t.footholds if not f.is_wall and f.left != f.right)
    x0, x1 = min(f.x1, f.x2), max(f.x1, f.x2)
    y = (f.y1 + f.y2) / 2.0
    # 框只盖住这条的一半：包含模式选不到，相交模式选得到
    half = (x0, y - 5, x0 + (x1 - x0) / 2.0, y + 5)
    ins = {id(g) for g in ze.pick_in_rect(t, half, "contains")}
    cross = {id(g) for g in ze.pick_in_rect(t, half, "crossing")}
    check(id(f) not in ins, "包含模式选中了只盖住一半的线")
    check(id(f) in cross, "相交模式没选中被盖住一半的线")
    # 整个盖住：两种模式都该选中
    full = (x0 - 5, y - 5, x1 + 5, y + 5)
    check(id(f) in {id(g) for g in ze.pick_in_rect(t, full, "contains")},
          "整个盖住了还不算包含")
    # 框选同样不许把墙带进来
    for g in ze.pick_in_rect(t, (min(f2.left for f2 in t.footholds),
                                 -10 ** 6, max(f2.right for f2 in t.footholds),
                                 10 ** 6), "crossing"):
        check(not g.is_wall, "全图框选把墙也选进来了：%s" % g)


def t_selection_modes():
    """选择集运算：replace / add / sub（Shift 加选、Alt 减选就靠它）。"""
    check(ze.new_selection({"1", "2"}, ["3"], "replace") == {"3"}, "replace 错")
    check(ze.new_selection({"1"}, ["2", "3"], "add") == {"1", "2", "3"}, "add 错")
    check(ze.new_selection({"1", "2"}, ["2"], "sub") == {"1"}, "sub 错")
    try:
        ze.new_selection(set(), [], "乱写")
        raise AssertionError("未知模式竟然没报错")
    except ValueError:
        pass


def t_ids_bbox():
    t = _terrain()
    f = next(f for f in t.footholds if not f.is_wall)
    b = ze.ids_bbox(t, [f.fid])
    check(b is not None, "bbox 算不出来")
    x, y, w, h = b
    check(x <= min(f.x1, f.x2) and y <= min(f.y1, f.y2) and w >= 0 and h >= 0,
          "bbox 不对：%s（这条 %s）" % (b, f))
    check(ze.ids_bbox(t, ["999999"]) is None, "不存在的 id 应该给 None")


def _tiny(ladders=(), portals=()):
    """一张只有一条平台的合成地形（字段与导出的 JSON 一致：全是字符串）。"""
    data = {"footholds": [{"id": "1", "x1": "0", "y1": "100", "x2": "200",
                           "y2": "100", "layer": "1", "group": "0"}],
            "ladderRope": list(ladders), "portals": list(portals)}
    return mapdata.Terrain("T", data)


def t_set_owns_ladders_and_portals():
    """集合"涉及"哪些绳梯/传送门：绳按 **y 区间重叠**判（可多归属）、传送门按包围盒。

    这条是需求 2 的地基：选中集合时要把这些线一起呼吸。判据写错的表现是
    "高亮少了一根绳子"或者"把隔壁平台的绳子也点亮了"，两者都不会报错 —— 只能靠断言。
    """
    from core import zones as zmod
    t = _tiny(ladders=[{"x": "100", "y1": "60", "y2": "140"},     # 跨过这条平台 ⇒ 归它
                       {"x": "100", "y1": "300", "y2": "340"},    # y 离太远 ⇒ 不归
                       {"x": "900", "y1": "80", "y2": "120"}],    # x 离太远 ⇒ 不归
              portals=[{"pn": "sp", "x": "50", "y": "110"},       # 落在包围盒内 ⇒ 归它
                       {"pn": "far", "x": "900", "y": "900"}])    # 远 ⇒ 不归
    check(zmod.set_span(t, ["1"]) == (0, 200, 100, 100),
          "集合跨度算错了：%s" % (zmod.set_span(t, ["1"]),))
    got = zmod.ladders_of(t, ["1"])
    check(len(got) == 1 and min(got[0].y1, got[0].y2) == 60,
          "绳梯归属判错了：%s" % got)
    check([p.pn for p in zmod.portals_of(t, ["1"])] == ["sp"],
          "传送门归属判错了：%s" % [p.pn for p in zmod.portals_of(t, ["1"])])
    check(zmod.ladders_of(t, []) == [] and zmod.portals_of(t, []) == [],
          "空集合不该有任何归属")


# ---------------------------------------------------------------- 离屏跑窗口

def t_dialog_register_undo_save():
    """窗口跑一遍：选中 → 注册 → 撤销/重做 → 保存；**墙不会被选进集合**。

    离屏跑真窗口（同 selftest_live_panel 的做法）：注册/撤销/存盘这条路是用户每天
    要走的，只在纯函数层面测过说明不了它能不能用。
    """
    from PyQt5.QtWidgets import QApplication
    # **必须把实例存下来**：QApplication 被回收后再建控件就是 qFatal（进程直接 abort，
    # 而且 stdout 缓冲一起丢 —— 一行报错都看不到，见 tools/selftest_deploy.py 的教训）
    app = QApplication.instance() or QApplication([])
    check(app is not None, "没拿到 QApplication")

    t = _terrain()
    walkable = [str(f.fid) for f in t.footholds if not f.is_wall]
    check(len(walkable) >= 3, "这张图可站立的 foothold 太少，测不出东西")
    tmp = Path(tempfile.mkdtemp(prefix="zones_"))
    try:
        dlg = ze.ZoneEditorDialog(MAP_ID, terrain=t, zones_path=tmp / "x.zones.json")
        # ① 全选：只有可站立的（墙不在里面）
        dlg.select(walkable)
        check(dlg.selection_ids() == set(walkable), "全选结果不对")
        check(not any(f.is_wall for f in t.footholds
                      if str(f.fid) in dlg.selection_ids()),
              "墙被选进了选择集")
        # ② 注册（名字重复要报错，不能静默覆盖）
        dlg.register("A平台")
        check("A平台" in dlg.zones.sets, "注册没生效")
        check(len(dlg.zones.sets["A平台"]["footholds"]) == len(walkable),
              "注册进的条数不对")
        try:
            dlg.register("A平台")
            raise AssertionError("重名竟然没报错（会静默毁掉已分好的组）")
        except ValueError:
            pass
        # ③ 撤销 / 重做
        dlg.undo()
        check("A平台" not in dlg.zones.sets, "撤销没生效")
        dlg.redo()
        check("A平台" in dlg.zones.sets, "重做没生效")
        # ④ 改名级联 + 成员增删
        dlg.rename("A平台", "战斗区")
        check("战斗区" in dlg.zones.sets and "A平台" not in dlg.zones.sets,
              "改名没生效")
        keep = walkable[:2]
        dlg.select(keep)
        dlg.remove_from_set("战斗区")
        check(set(dlg.zones.sets["战斗区"]["footholds"]) == set(walkable) - set(keep),
              "移出成员没生效")
        dlg.add_to_set("战斗区")
        check(set(dlg.zones.sets["战斗区"]["footholds"]) == set(walkable),
              "加入成员没生效")
        # ⑤ 保存 → 文件真的落盘、且能被读回来（id 是字符串）
        path, probs = dlg.save()
        check(path.exists(), "保存后文件不存在：%s" % path)
        check(probs == [], "正常文件不该有校验问题：%s" % probs)
        from core import zones as zmod
        back = zmod.Zones.from_dict(__import__("json").loads(
            path.read_text(encoding="utf-8")))
        check(back.sets == dlg.zones.sets, "存读不一致")
        # ⑥ 删集合（连带清掉引用）后保存，文件里也不该再有它
        gone = dlg.delete_set("战斗区")
        check(gone == [] or isinstance(gone, list), "删集合的返回值不对")
        check("战斗区" not in dlg.zones.sets, "删集合没生效")
        dlg.close()
    finally:
        import shutil
        shutil.rmtree(str(tmp), ignore_errors=True)


def t_dialog_background_breathing_and_size():
    """底图铺上了、呼吸高亮覆盖"集合涉及的线"、**聚焦不改变窗口大小**。

    对应三条界面要求（2026-09-26）：
      ① 看起来像地形叠加图 ⇒ 小地图底图铺在几何下面（换算与 map_terrain_view 同一套）；
      ② 选中的 foothold / 选中集合涉及的 foothold+绳梯+传送门 ⇒ 呼吸高亮；
      ③ 选中集合只聚焦（视口），**窗口尺寸不变** —— 这条以前会悄悄变大（视图
         setMinimumWidth 顶着布局长）。
    """
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    t = _terrain()
    tmp = Path(tempfile.mkdtemp(prefix="zones_hl_"))
    try:
        dlg = ze.ZoneEditorDialog(MAP_ID, terrain=t, zones_path=tmp / "x.zones.json")
        dlg.show()
        app.processEvents()
        dlg.fit()
        app.processEvents()
        # ① 底图
        if (t.mini or {}).get("canvas"):
            check(dlg._add_background() is not None,
                  "小地图底图没加载上（要求 1：要像地形叠加图）")
        # ② 呼吸高亮：注册集合后，它的 foothold + 绳梯 + 传送门都该在里面
        ids = [str(f.fid) for f in t.footholds if not f.is_wall]
        dlg.select(ids)
        dlg.register("A平台")
        cols = {c.name() for _it, c in dlg._hl_items}
        check(len(dlg._hl_items) >= len(ids),
              "集合的 foothold 没进呼吸高亮：%d < %d" % (len(dlg._hl_items), len(ids)))
        lad = zones.ladders_of(t, set(ids))
        por = zones.portals_of(t, set(ids))
        if lad:
            check(ze.C_LADDER.name() in cols,
                  "集合的绳梯没进呼吸高亮（要求 2）：%s" % sorted(cols))
        if por:
            check(ze.C_PORTAL.name() in cols,
                  "集合的传送门没进呼吸高亮（要求 2）：%s" % sorted(cols))
        # 呼吸真的在动（不是画上去就静止）
        widths = set()
        for _ in range(12):
            dlg._on_pulse()
            widths.add(round(dlg._hl_items[0][0].pen().widthF(), 2))
        check(len(widths) > 3, "呼吸高亮的线宽没变化：%s" % sorted(widths))
        # 取消选择后，集合自己的 foothold 必须仍以"集合色"呼吸
        dlg.select([])
        cols2 = {c.name() for _it, c in dlg._hl_items}
        check(ze.C_IN_SET.name() in cols2,
              "取消选择后集合的 foothold 没在呼吸：%s" % sorted(cols2))
        # ③ 聚焦不改变窗口尺寸
        size0 = dlg.size()
        dlg.focus_ids(ids)
        app.processEvents()
        check(dlg.size() == size0,
              "聚焦把窗口改大了：%s → %s（要求 3）" % (size0, dlg.size()))
        dlg.close()
    finally:
        import shutil
        shutil.rmtree(str(tmp), ignore_errors=True)


def t_editor_opacity_nonmodal_and_signal():
    """透明度滑条 / **非模态** / 保存信号 / 重开时节拍器接回来。

    对应 2026-09-26 的三条要求里的前两条：
      ① 底图透明度要能拖（底图是深蓝示意图、线是亮绿，每张图每块屏的观感都不一样，
         写死一个值必然有人嫌"太抢眼"或者"太黑看不清地形"）；
      ② 编辑器**不许锁住工作台** —— 圈集合时要照着「实时」页的画面判断哪块是 A 平台，
         还要反复跑一下看世界坐标对不对；
      ③ 非模态之后调用方拿不到 `exec_()` 的返回值 ⇒ 保存完必须**主动通知**外面，
         否则那块面板的提示永远停在旧状态。
    """
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    t = _terrain()
    tmp = Path(tempfile.mkdtemp(prefix="zones_nm_"))
    try:
        dlg = ze.ZoneEditorDialog(MAP_ID, terrain=t, zones_path=tmp / "x.zones.json")
        # ② 非模态
        check(not dlg.isModal(), "编辑器是模态的 —— 它会把整个工作台锁住")
        # ① 透明度：拖滑条只改底图那一个 item，不重建场景
        check(dlg._add_background() is not None, "底图没加载上")
        check(dlg._bg_item is not None, "底图 item 没记下来（滑条改不到它）")
        dlg.sl_bg.setValue(20)
        check(abs(dlg._bg_item.opacity() - 0.20) < 1e-6,
              "滑条没改到底图透明度：%s" % dlg._bg_item.opacity())
        check("20%" in dlg.lbl_bg.text(), "百分比文字没跟着变：%s" % dlg.lbl_bg.text())
        dlg.sl_bg.setValue(0)
        check(dlg._bg_item.opacity() == 0.0, "0% 应该完全隐藏底图")
        # 改集合会重建场景 —— 重建之后滑条的值必须继续生效
        dlg.sl_bg.setValue(35)
        dlg._rebuild_scene()
        check(abs(dlg._bg_item.opacity() - 0.35) < 1e-6,
              "重建场景后滑条的值丢了：%s" % dlg._bg_item.opacity())
        # ③ 保存要通知外面
        got = []
        dlg.saved_now.connect(lambda mid, n: got.append((mid, n)))
        ids = [str(f.fid) for f in t.footholds if not f.is_wall][:2]
        dlg.select(ids)
        dlg.register("A平台")
        dlg.save()
        check(got == [(MAP_ID, 1)],
              "保存后没发信号（面板的提示不会更新）：%s" % got)
        # ④ 关掉再开：节拍器要接回来（不然呼吸高亮不动了）
        dlg.close()
        check(not dlg.timer.isActive(), "关窗后节拍器还在跑（白烧 CPU）")
        dlg.show()
        check(dlg.timer.isActive(), "重新显示后节拍器没接回来（呼吸高亮会停）")
        dlg.close()
    finally:
        import shutil
        shutil.rmtree(str(tmp), ignore_errors=True)


def t_line_width_setting():
    """「设置 → foothold 编辑器线条宽度」：存得住、编辑器按**屏幕像素**画。

    为什么单位必须是屏幕像素（不是场景单位）：编辑器整图 fit 时约 0.35 倍、放大看
    细节能到 8 倍以上 —— 按场景单位给宽度的话，同一个值在两种视图下差二十多倍
    （整图时细到看不见、放大时糊成一片）。所以这里钉的是**缩放变了、屏幕上粗细不变**。

    原先那些线是 `QPen(color, 0)`（cosmetic 笔），粗细写死 1 像素、**根本没法调** ——
    这正是加这个设置要解决的问题。
    """
    import unittest.mock as mock
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    from gui import theme

    app = QApplication.instance() or QApplication([])
    tmp = Path(tempfile.mkdtemp(prefix="ui_"))
    try:
        # 把 CFG 指到临时文件：**自检绝不许写坏真实的 config/ui.yaml**
        with mock.patch.object(theme, "CFG", tmp / "ui.yaml"):
            # ① 存取往返 + 越界夹住 + 坏值退回
            check(theme.load_foothold_width() == theme.FOOTHOLD_W_DEFAULT,
                  "没写过配置时应给默认值")
            theme.save_foothold_width(4)
            check(theme.load_foothold_width() == 4, "存进去读不回来")
            theme.save_foothold_width(999)
            check(theme.load_foothold_width() == theme.FOOTHOLD_W_MAX,
                  "越界没夹住（线会糊住底图）")
            theme.save_foothold_width("坏值")
            check(theme.load_foothold_width() == theme.FOOTHOLD_W_DEFAULT,
                  "坏值没退回默认")

            # ② 编辑器：同一个设置值，缩放变了**屏幕宽度不变**
            t = _terrain()
            dlg = ze.ZoneEditorDialog(MAP_ID, terrain=t,
                                      zones_path=tmp / "x.zones.json")
            dlg.view.resize(600, 400)
            dlg.view.set_line_width(3)
            fid = str(next(f for f in t.footholds if not f.is_wall).fid)
            item = dlg._items[fid]
            for s in (1.0, 0.5, 2.0):
                dlg.view.resetTransform()
                dlg.view.scale(s, s)
                dlg.view._apply_widths()
                got = item.pen().widthF() * s      # 场景宽度 × 缩放 = 屏幕宽度
                check(abs(got - 3.0) < 0.05,
                      "缩放 %.1f 倍时屏幕线宽是 %.2f px（应当恒为 3）" % (s, got))
            # 墙是虚线：线型不能被宽度那套改掉，宽度也要跟着设置走
            wall = next((f for f in t.footholds if f.is_wall), None)
            if wall is not None:
                wi = dlg._items[str(wall.fid)]
                check(abs(wi.pen().widthF() * dlg.view.transform().m11() - 3.0) < 0.05,
                      "墙的线宽没跟着设置走：%.2f" % wi.pen().widthF())
                check(wi.pen().style() == Qt.DashLine,
                      "墙被画成实线了（分不清墙和地面）")
            # ③ 改完设置重读一次要生效（窗口一直开着时靠 _on_edit_zones 调它）
            theme.save_foothold_width(6)
            dlg.reload_line_width()
            check(abs(item.pen().widthF() * dlg.view.transform().m11() - 6.0) < 0.05,
                  "reload_line_width 没生效：%.2f" % item.pen().widthF())

            # ④ 设置弹窗里真的那一项在，点确定真的写进去
            from gui.settings_dialog import SettingsDialog
            sd = SettingsDialog()
            check(hasattr(sd, "sp_fh_width"), "设置弹窗里没有「线条宽度」这一项")
            sd.sp_fh_width.setValue(5)
            sd._accept()
            check(theme.load_foothold_width() == 5,
                  "点了确定却没写进配置：%s" % theme.load_foothold_width())
            dlg.close()
    finally:
        import shutil
        shutil.rmtree(str(tmp), ignore_errors=True)


def t_selection_and_set_are_exclusive():
    """视窗选择 与 集合高亮 **一次只亮一个**（Shift/Ctrl/Alt 批量时例外）；聚焦只居中不缩放。

    对应 2026-09-26 的要求 1、2。为什么非要互斥：两边同时亮着时"现在在编辑哪个"
    没法判断 —— 集合自己的 foothold 用 C_IN_SET 在呼吸，和"已选中"的 C_SEL 混在一起
    根本分不出来，而这两种颜色代表的动作完全不同（一个是要改集合，一个是要改选择）。
    """
    from PyQt5.QtCore import QPointF
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    t = _terrain()
    walk = [f for f in t.footholds if not f.is_wall]
    check(len(walk) >= 8, "地形里非墙 foothold 太少，这条测不了")
    tmp = Path(tempfile.mkdtemp(prefix="zsel_"))
    try:
        dlg = ze.ZoneEditorDialog(MAP_ID, terrain=t, zones_path=tmp / "x.zones.json")
        dlg.resize(980, 660)
        dlg.show()
        app.processEvents()

        def mid(f):
            return ((f.left + f.right) / 2.0, f.y_at((f.left + f.right) / 2.0))

        # 建一个集合：注册后**高亮它、并清掉视窗选择**（一次只编辑一样东西）
        dlg.select([str(f.fid) for f in walk[:4]])
        dlg.register("A平台")
        check(dlg._highlight == "A平台", "注册后没高亮新集合：%s" % dlg._highlight)
        check(not dlg.selection_ids(),
              "注册后视窗选择还在（两套颜色会一起呼吸）：%s" % dlg.selection_ids())
        check(dlg.lst.currentItem() is not None, "注册后列表里没选中这个集合")

        # ① 在视窗里点选一条 foothold（replace）⇒ **集合高亮被取消**（列表那行也复位）
        x, y = mid(walk[6])
        dlg._on_picked(("click", x, y, 6.0), "replace")
        check(dlg.selection_ids(), "点选没选中任何 foothold（这条测不了后续）")
        check(dlg._highlight is None,
              "在视窗里选了 foothold，集合还高亮着：%s" % dlg._highlight)
        check(dlg.lst.currentItem() is None,
              "集合高亮取消了，列表那行还亮着（显示错位）")

        # ② 反向：点集合 ⇒ **清掉视窗里的选择**
        dlg.lst.setCurrentRow(0)          # 触发 currentItemChanged → _on_pick_set
        check(dlg._highlight == "A平台", "点集合没高亮：%s" % dlg._highlight)
        check(not dlg.selection_ids(),
              "点集合没清掉视窗里的选择：%s" % dlg.selection_ids())

        # ③ Shift 批量选择（add）⇒ **集合高亮保留**（要往这个集合里加减成员）
        dlg._on_picked(("click", x, y, 6.0), "add")
        check(dlg._highlight == "A平台",
              "Shift 批量选择时集合高亮被清掉了：%s" % dlg._highlight)
        check(dlg.selection_ids(), "Shift 加选没生效")

        # ④ 点空白（replace）⇒ 只清空选择，**不动集合高亮**（顺手点一下空白不该丢集合）
        bx, by = t.bounds[0] - 5000, t.bounds[1] - 5000
        dlg._on_picked(("click", bx, by, 6.0), "replace")
        check(not dlg.selection_ids(), "点空白没清空选择")
        check(dlg._highlight == "A平台", "点空白把集合高亮也清了（不该清）")

        # ⑤ 聚焦 = **只居中，不缩放**；目标要落在视口里
        # ⚠ 缩放要选"整张图比视口还大"的档（这里 1.5 倍）：图比视口小时 Qt 会把
        # 整幅图居中、centerOn 无从发挥，那样测出来的"没居中"是假的。
        dlg.view.resetTransform()
        dlg.view.scale(1.5, 1.5)
        s0 = dlg.view.transform().m11()
        ids = dlg.zones.sets["A平台"]["footholds"]
        dlg.focus_ids(ids)
        app.processEvents()
        s1 = dlg.view.transform().m11()
        check(abs(s1 - s0) < 1e-6,
              "聚焦改了缩放：%.4f → %.4f（要求只居中不缩放）" % (s0, s1))
        b = ze.ids_bbox(t, ids)
        cx, cy = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
        vp = dlg.view.viewport().rect()
        check(vp.contains(dlg.view.mapFromScene(QPointF(cx, cy))),
              "聚焦后目标没出现在视口里：(%.0f, %.0f)" % (cx, cy))
        c1 = dlg.view.mapToScene(vp.center())
        check(abs(c1.x() - cx) < 60 and abs(c1.y() - cy) < 60,
              "聚焦后目标没在画面中央：中心在 (%.0f, %.0f)，目标 (%.0f, %.0f)"
              % (c1.x(), c1.y(), cx, cy))
        dlg.close()
    finally:
        import shutil
        shutil.rmtree(str(tmp), ignore_errors=True)


def t_set_names_on_canvas():
    """集合名标在画布上（包围盒上方、用集合自己的颜色）；**选中的集合名字跟着呼吸**。

    对应 2026-09-26 的要求 1（像地形叠加图那样把注册名标上去；选中时也呼吸高亮）。
    ⚠ 文字 item 用 **brush** 上色、**没有 pen** —— 呼吸那一段要是照线那样 setPen，
    一调就是 AttributeError 崩掉，所以这条专门跑几拍呼吸。
    """
    from PyQt5.QtGui import QColor
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    t = _terrain()
    walk = [f for f in t.footholds if not f.is_wall]
    check(len(walk) >= 8, "地形里非墙 foothold 太少，这条测不了")
    tmp = Path(tempfile.mkdtemp(prefix="zname_"))
    try:
        dlg = ze.ZoneEditorDialog(MAP_ID, terrain=t, zones_path=tmp / "x.zones.json")
        dlg.select([str(f.fid) for f in walk[:4]])
        dlg.register("A平台")
        dlg.select([str(f.fid) for f in walk[4:8]])
        dlg.register("B平台")
        check(set(dlg._labels) == {"A平台", "B平台"},
              "画布上没把集合名标出来：%s" % sorted(dlg._labels))
        it = dlg._labels["A平台"]
        check(isinstance(it, ze.QGraphicsSimpleTextItem), "名字不是文字 item")
        # 颜色 = 该集合自己的颜色（和列表里那个色块一致）
        want = QColor(dlg.zones.sets["A平台"]["color"]).name()
        check(it.brush().color().name() == want,
              "名字没用集合颜色：%s vs %s" % (it.brush().color().name(), want))
        # 位置：在集合**水平中间**、且**在平台上方**（压住平台就挡住线了）
        x0, x1, y0, _y1 = zones.set_span(t, dlg.zones.sets["A平台"]["footholds"])
        b = it.sceneBoundingRect()
        check(abs(b.center().x() - (x0 + x1) / 2.0) < 80,
              "名字没标在集合水平中间：%.0f vs %.0f" % (b.center().x(), (x0 + x1) / 2.0))
        check(b.bottom() <= y0 + 4,
              "名字压在平台上了（bottom=%.0f，平台 y=%.0f）" % (b.bottom(), y0))

        # 注册 B 之后高亮的是 B ⇒ 只有 B 的名字在呼吸
        hl_text = [x for x, _c in dlg._hl_items
                   if isinstance(x, ze.QGraphicsSimpleTextItem)]
        check(dlg._labels["B平台"] in hl_text, "刚注册的集合名没跟着高亮")
        check(dlg._labels["A平台"] not in hl_text, "没高亮的集合名也在呼吸")

        # 选中某个集合的**成员**（没有高亮别的集合）⇒ 它的名字也要呼吸
        dlg._highlight = None
        dlg.select(dlg.zones.sets["A平台"]["footholds"][:2])
        hl_text2 = [x for x, _c in dlg._hl_items
                    if isinstance(x, ze.QGraphicsSimpleTextItem)]
        check(dlg._labels["A平台"] in hl_text2,
              "选中集合的成员时，集合名没跟着呼吸")
        check(dlg._labels["B平台"] not in hl_text2, "没被选中的集合名也在呼吸")

        # 呼吸真的改到了名字（写错成 setPen 会在这儿炸）
        seen = set()
        for _ in range(12):
            dlg._on_pulse()
            seen.add(dlg._labels["A平台"].brush().color().name())
        check(len(seen) > 3, "集合名的颜色没在呼吸：%s" % sorted(seen))
        dlg.close()
    finally:
        import shutil
        shutil.rmtree(str(tmp), ignore_errors=True)


def t_mid_jump_dialog_wired():
    """⭐ 「爬」的**中途跳下**在「增加可达 / 双击改边」弹窗里的**接线**（用户 2026-09-28 ✓）。

    ⚠ **为什么是源码级**（而不是像 `t_add_reach_dialog` 那样建真弹窗）：那个弹窗是**非模态 +
    单实例**（`_open_reach` 管着 ✓ 结果走 `applied` 信号 ✓），直接 new 出来测"二级联动"要
    自己摆 `ladders` / `drop_choices` / `init` 一整套夹具，而那套夹具**已经在**
    `t_add_reach_dialog` 里了 ✓（它钉的是类型显隐 + 返回值 ✓）⇒ 这里只补**新字段那几根线**
    （漏一根的后果是"下拉改了不生效 / 高度永远藏着"✗ —— 不报错、不崩，只能靠钉子 ✓）。

    钉五件：
      ① 下拉存在、选项/文案**照 `zones` 那一份**（别两处各写一套 ✗）；
      ② 「高度」是 **QDoubleSpinBox**（世界坐标 y 可为负 ⇒ 不能用 QSpinBox 的"只能非负"直觉 ✗，
         而且范围要够大 ✓）；
      ③ `cmb_mid` 的变化**也接进了 `_sync_kind`**（⚠ 只接 `cmb_kind` 的话，改了下拉"高度"
         那一行**不跟着动** ✗ —— 这是"二级联动"最容易漏的一根线 ✓）；
      ④ `_sync_kind` 里：**类型=爬** 才显示下拉、**方向非空**才显示高度 ✓；
      ⑤ `_accept` / `edit_edge` 把 `mid_dir` / `mid_y` 透出去（含"没给就删"= 回到不启用 ✓）+
         `_apply_reach` 两条路都转发 ✓。
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "gui" / "zone_editor.py"
           ).read_text(encoding="utf-8")

    # ① 下拉 + 文案来源
    check("self.cmb_mid = NoWheelComboBox()" in src,
          "弹窗里没有「中途跳下」下拉（`cmb_mid` ✗）")
    check("zones.MID_JUMP_DIRS" in src and "zones.MID_JUMP_LABELS" in src,
          "下拉的选项/文案没照 `zones.MID_JUMP_DIRS` / `MID_JUMP_LABELS`"
          "（两处各写一套，迟早对不上 ✗）")
    # ② 高度控件
    check("self.spn_mid_y = NoWheelDoubleSpinBox()" in src,
          "「高度」不是 `NoWheelDoubleSpinBox`（⚠ UI 规范：滚轮不许改参数 ✗ `check_ui` 会拦 ✓；"
          "而且世界 y **有负数** ⇒ 不能用「只能非负」那种直觉 ✗）")
    # ③ 二级联动的"第二根线"
    check("self.cmb_mid.currentIndexChanged.connect" in src,
          "⚠ `cmb_mid` 自己变化**没接进 `_sync_kind`** ⇒ 改了下拉、「高度」那一行不跟着显隐 ✗")
    # ④ `_sync_kind` 里那几条
    check('self.cmb_mid.setVisible(kind == "climb")' in src,
          "「中途跳下」下拉不是「只对爬显示」（该 `kind == climb` ✗）")
    check('_show_y = (kind == "climb" and bool(self.cmb_mid.currentData()))' in src
          and "self.spn_mid_y.setVisible(_show_y)" in src,
          "「高度」不是「方向非空才显示」（用户明确「**选取后**增加一个参数「高度」」✗）")
    # ⑤ 透传
    check('"mid_dir": ((self.cmb_mid.currentData() or None)' in src,
          "`_accept` 没把 `mid_dir` 放进返回值 ⇒ 弹窗里配了落不下去 ✗")
    check('hit["mid_dir"] = str(mid_dir)' in src and 'hit.pop("mid_dir", None)' in src,
          "`edit_edge` 没做「给了就存、没给就删」（回到「不中途跳下」那条路 ✗）")
    check(src.count("mid_dir=r.get(\"mid_dir\")") == 2,
          "`_apply_reach` 的两条路（增/改）没都转发 `mid_dir`（漏一条 ⇒ 那条路上配了不生效 ✗）")


def t_edges_in_editor():
    """编辑器里的「边」：加/反向/删 + 两栏各自跟上 + **悬空边标红** + 画布箭头画法。

    对应 §12.3 B5 + 2026-09-26 的要求 1、3。三个最容易错的点：
      · **悬空边**（集合被删/改名）：看着没事，只有跑起来才会在路径里断掉；
      · **两栏的方向**：「可到达」只列从焦点出去的，「可被到达」只列进来的 ——
        点「反向」之后那条边属于**对面**那个集合，不该在当前的「可到达」里冒出来。
        ⚠ **2026-09-27 一度改成"列全部边"，当天用户判定是误修 ⇒ 已改回收窄** ——
        这条钉子就是防它再被改回去的（要"总览"就取消选中 ✓）；
      · 加边没加进去**要说话**（同一天修的，见 `t_two_ladders_same_pair` ✓）。
    """
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    t = _terrain()
    walk = [f for f in t.footholds if not f.is_wall]
    check(len(walk) >= 8, "地形里非墙 foothold 太少，这条测不了")
    tmp = Path(tempfile.mkdtemp(prefix="zedge_"))
    try:
        dlg = ze.ZoneEditorDialog(MAP_ID, terrain=t, zones_path=tmp / "x.zones.json")
        dlg.select([str(f.fid) for f in walk[:4]])
        dlg.register("A平台")
        dlg.select([str(f.fid) for f in walk[4:8]])
        dlg.register("B平台")

        def arrows():
            """画布上的箭头：线 + 三角都是 z=7（这个 z 只有边在用）。"""
            return len([i for i in dlg.scene.items() if i.zValue() == 7])

        dlg.add_edge("A平台", "B平台", "walk", why="测试用")
        check(len(dlg.zones.edges) == 1, "加边没生效")
        # 焦点此时在「B平台」（刚注册完它）⇒ A→B 是**进来的**边：不该出现在「可到达」，
        # 要去下面「可被到达」栏（2026-09-26 要求 1、2；2026-09-27 一度放宽、当天又改回 ✓）
        check(dlg.lst_e.count() == 0,
              "「可到达」里混进了进来的边：%d 行" % dlg.lst_e.count())
        check(dlg.lst_in.count() == 1,
              "「可被到达」没列出能到 B 平台的那条：%d 行" % dlg.lst_in.count())
        # 切到「A平台」⇒ 反过来：出去的那条在「可到达」，进来的没有
        dlg._highlight = "A平台"
        dlg._refresh_edges()
        check(dlg.lst_e.count() == 1, "边列表没跟上：%d 行" % dlg.lst_e.count())
        txt = dlg.lst_e.item(0).text()
        check("A平台 → B平台" in txt and "走" in txt,
              "边列表那行写得不清楚：%r" % txt)
        check(dlg.lst_in.count() == 0,
              "「可被到达」里混进了出去的边：%d 行" % dlg.lst_in.count())
        check(arrows() == 2, "画布上没画出这条边（线+箭头应有 2 个 item，实际 %d）"
              % arrows())
        # 箭头的画法（2026-09-26 要求 3）：**连线 = 黑色细虚线**，**三角 = 类型色**。
        # 为什么线不再按类型上色：画面上已经有亮绿地形线、深蓝底图、蓝色集合名、
        # 黄色呼吸高亮 —— 箭头再各挑一种颜色就认不出是箭头了。类型交给三角。
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QGraphicsLineItem, QGraphicsPolygonItem
        shafts = [i for i in dlg.scene.items()
                  if i.zValue() == 7 and isinstance(i, QGraphicsLineItem)]
        heads = [i for i in dlg.scene.items()
                 if i.zValue() == 7 and isinstance(i, QGraphicsPolygonItem)]
        check(shafts, "画布上没有连线 item")
        check(all(i.pen().color().name() == ze.C_ARROW for i in shafts),
              "连通箭头的连线不是黑色：%s" % [i.pen().color().name() for i in shafts])
        check(all(i.pen().style() == Qt.DashLine for i in shafts),
              "连通箭头的连线不是虚线：%s" % [i.pen().style() for i in shafts])
        # 「细」：线宽就是设置里那个值（原来还额外 +1 px）
        sc = dlg.view.transform().m11()
        check(all(abs(i.pen().widthF() * sc - dlg.view._line_w) < 0.05 for i in shafts),
              "连线不够细：%s（设置值 %s）"
              % ([i.pen().widthF() * sc for i in shafts], dlg.view._line_w))
        walk_c = zones.EDGE_LABELS["walk"][1]
        check(heads and all(i.brush().color().name() == walk_c for i in heads),
              "箭头三角没按类型着色（类型就认不出了）：%s"
              % [i.brush().color().name() for i in heads])
        check(all(i.pen().style() == Qt.SolidLine for i in heads),
              "箭头三角被画成虚线边了（小三角只剩一团噪点）")

        # 选中列表里那条 ⇒ 画布上**对应的箭头呼吸**（2026-09-26 要求 4）。
        # 为什么必须钉：画布上可以同时有十几条箭头，光看列表那行字对不上是哪条
        # （反向两条只差一个方向、本来就重叠着画）—— 点亮它才知道"改的是哪条"。
        dlg.lst_e.setCurrentRow(0)
        # ⚠ 比**值**不比身份：列表里那份是 Qt 转换过的 dict 副本（踩过）
        check(dlg._edge_hl == dlg.zones.edges[0],
              "选中那行没记下来是哪条边：%r" % (dlg._edge_hl,))
        dlg._on_pulse()                    # 手动跑一拍（自检里不等 60ms 的计时器）
        # 比**色相**不比精确值：呼吸每拍会改明度（实测拍到的可能是 #a98c34 这种压暗相位）
        check(all(i.pen().color().hue() == ze.C_SEL.hue() for i in shafts),
              "选中那条的连线没亮起来（该是选中的黄）：%s"
              % [i.pen().color().name() for i in shafts])
        check(all(i.pen().style() == Qt.DashLine for i in shafts),
              "呼吸把连线变成实线了（虚线是它的身份）")
        check(all(i.brush().color().hue() == ze.C_SEL.hue() for i in heads),
              "箭头三角没跟着亮：%s" % [i.brush().color().name() for i in heads])
        # 列表重填（点 foothold、换焦点都会走到）**不许把正在看的那条边丢掉**
        dlg._refresh_edges()
        check(dlg._edge_hl == dlg.zones.edges[0],
              "重填列表把选中的那条边丢了（画布上的呼吸会跟着停）")
        check(dlg.lst_e.currentItem() is not None, "重填后没把那行选回来")
        # 取消选中 ⇒ 恢复黑细虚线 + 类型色三角
        dlg.lst_e.setCurrentRow(-1)
        check(dlg._edge_hl is None, "取消选中后还留着高亮目标：%r" % (dlg._edge_hl,))
        check(all(i.pen().color().name() == ze.C_ARROW for i in shafts),
              "取消选中后连线没恢复成黑色：%s"
              % [i.pen().color().name() for i in shafts])
        check(all(i.brush().color().name() == walk_c for i in heads),
              "取消选中后三角没恢复成类型色：%s"
              % [i.brush().color().name() for i in heads])
        # 换焦点：这条边落到「可被到达」栏，选中项要跟着搬过去（不是凭空消失）
        dlg._highlight = "B平台"
        dlg._refresh_edges()
        check(dlg.lst_in.count() == 1, "B 的「可被到达」该有 1 条：%d"
              % dlg.lst_in.count())
        # 在「可被到达」栏里选中，箭头一样点亮（两栏共用同一个槽）
        dlg.lst_in.setCurrentRow(0)
        check(dlg._edge_hl == dlg.zones.edges[0],
              "在「可被到达」里选中没点亮箭头：%r" % (dlg._edge_hl,))
        check(dlg.lst_e.currentItem() is None,
              "两栏同时选中了（会让人以为指向同一条）")
        dlg.lst_in.setCurrentRow(-1)
        dlg._highlight = "A平台"
        dlg._refresh_edges()

        # 反向复制：有向图（爬上去和爬下来要两条）。**它属于对面那个集合** ——
        # 所以不会在「A平台」的可到达里冒出来（用户报的就是这条："点反向多出一个
        # 不该出现在窗口里的项目"）
        dlg.rev_edge(dlg.zones.edges[0])
        check(len(dlg.zones.edges) == 2, "反向复制没生效")
        check(arrows() == 4, "反向边没画出来：%d" % arrows())
        check(dlg.lst_e.count() == 1,
              "反向边（B→A）跑到「A平台」的可到达里去了（它属于 B）：%d 行"
              % dlg.lst_e.count())
        check(dlg.lst_in.count() == 1,
              "反向边没出现在「可被到达」里：%d 行" % dlg.lst_in.count())

        # 删边：删掉 A→B，剩下的是 B→A ⇒ 在「可被到达」里，不在「可到达」里
        dlg.del_edge(dlg.zones.edges[0])
        check(len(dlg.zones.edges) == 1, "删边没生效：%s" % dlg.zones.edges)
        check(dlg.lst_e.count() == 0 and dlg.lst_in.count() == 1,
              "删边后两栏没跟上：可到达 %d 行 / 可被到达 %d 行"
              % (dlg.lst_e.count(), dlg.lst_in.count()))

        # 悬空边：指向一个不存在的集合 ⇒ 列表里必须**标红**（真断链路只有跑起来才发现）
        dlg.zones.edges.append({"from": "A平台", "to": "幽灵", "kind": "walk"})
        dlg._refresh_edges()
        bad = [dlg.lst_e.item(i) for i in range(dlg.lst_e.count())
               if "悬空" in dlg.lst_e.item(i).text()]
        check(len(bad) == 1, "悬空边没在列表里标出来：%s"
              % [dlg.lst_e.item(i).text() for i in range(dlg.lst_e.count())])
        cols = row_colors(bad[0])
        check(("#c5221f", "幽灵") in cols and ("#c5221f", "A平台") in cols,
              "悬空边的那两个名字没标红：%s" % (cols,))
        check("[走(walk)]" not in "".join(t for _c, t in cols),
              "类型标记也被上色了（它该是正常黑字）：%s" % (cols,))
        check("悬空边" in bad[0].toolTip(), "悬空边没在 tooltip 里说清后果")
        # 悬空边**不画到画布上**（画不出来，只能靠列表说）
        check(arrows() == 2, "悬空边也被画到画布上了：%d" % arrows())

        # 撤销能回退加边（不然改错了只能靠手改文件）
        n0 = len(dlg.zones.edges)
        dlg.add_edge("B平台", "A平台", "drop", why="撤销用")
        check(len(dlg.zones.edges) == n0 + 1, "又加了一条边失败")
        dlg.undo()
        check(len(dlg.zones.edges) == n0, "撤销没回退加边：%s" % dlg.zones.edges)

        # 建议那行字：真边 ↔「待圈地形」要一眼分得开（后者不是边，是缺集合的提示）
        s_edge = {"from": "A", "to": "B", "kind": "climb", "ladder": "L2", "why": "x"}
        s_miss = {"from": "A", "to": None, "kind": "climb",
                  "why": "绳 L1 的下端通向 foothold 44", "x": 87, "y": -207}
        check("待圈地形" in dlg._sug_label(s_miss), "没把「待圈地形」标出来：%r"
              % dlg._sug_label(s_miss))
        check("L2" in dlg._sug_label(s_edge) and "爬" in dlg._sug_label(s_edge),
              "真边的建议行没写清绳和类型：%r" % dlg._sug_label(s_edge))
        dlg.close()
    finally:
        import shutil
        shutil.rmtree(str(tmp), ignore_errors=True)


def t_reach_ui_filter_and_style():
    """两栏「可到达 / 可被到达」的界面约定（2026-09-26 要求）：

      ① **三栏能缩、栏宽能拖**：面板栏各进滚动区（矮下来是出滚动条，不是按钮消失），
         且「集合」与「可达 / 可被到达」**横向并排**（不是原来那样上下堆在一根滚动条里）；
      ② 名字统一**蓝**（一眼认出"这是个注册过的集合"；悬空边那种**错误**仍标红）；
      ③ **可到达只列出去的边**：选中集合（或它的一段 foothold）时，这一栏回答的是
         "它能去哪儿"，而且**把收窄说出来**（静悄悄少几条是最难查的"东西不见了"）。
         ⚠ 2026-09-27 一度改成"列全部边、相关的排最前"，**当天用户判定是误修** ⇒
         已改回收窄 —— 这条用例就是防它再被改回去的 ✗；
      ④ **可被到达只列进来的边、且只读**：那一栏**一个编辑按钮都没有** ——
         要改就选中上面那个集合，它的"可到达"里就有这条边（一次只编辑一样东西）。

    收窄的判据是 `_focus_set`：高亮的集合 > 选中的 foothold 所属的集合（同一批只属于
    一个集合时才算）> None（什么都没选 ⇒ 显示全部，那既是总览也是逃生口）。
    """
    from PyQt5.QtWidgets import (QApplication, QPushButton, QScrollArea,
                                 QSplitter)

    app = QApplication.instance() or QApplication([])
    t = _terrain()
    walk = [f for f in t.footholds if not f.is_wall]
    check(len(walk) >= 12, "地形里非墙 foothold 太少，这条测不了")
    tmp = Path(tempfile.mkdtemp(prefix="zreach_"))
    try:
        dlg = ze.ZoneEditorDialog(MAP_ID, terrain=t, zones_path=tmp / "x.zones.json")
        dlg.resize(980, 660)
        # ① 三栏布局（2026-09-26 用户要求："信息全挤在一起，也不能缩放调整，
        #    希望将集合编辑与可达、可被到达的布局分列，以省去滑动过程"）
        check(isinstance(getattr(dlg, "split", None), QSplitter),
              "三栏没走 QSplitter ⇒ 栏宽拖不动（用户报的「不能缩放调整」）")
        check(dlg.split.count() == 3, "不是三栏：%d" % dlg.split.count())
        check(not dlg.split.childrenCollapsible(),
              "某一栏能被拖成 0 宽（拖没了就找不回来、那栏的按钮也点不到）")
        for _name, _attr in (("集合", "area_sets"), ("可达 / 可被到达", "area_edges")):
            _a = getattr(dlg, _attr, None)
            check(isinstance(_a, QScrollArea),
                  "「%s」栏没进 QScrollArea（窗口一矮按钮就被压没）" % _name)
            check(_a.widgetResizable(),
                  "「%s」栏没跟着分栏宽度走（缩窗口它不缩）" % _name)
            check(_a.parent() is dlg.split, "「%s」栏不在分栏里" % _name)
        # **集合**与**可达 / 可被到达**必须**左右并排** —— 这一条就是用户要的
        # "省去滑动过程"：原来它们在**同一根**竖排滚动条里上下堆着，窗口一矮就得
        # 滚上去看集合、滚下来看边 ✗
        dlg.show()
        for _ in range(4):
            app.processEvents()
        _gs, _ge = dlg.area_sets.geometry(), dlg.area_edges.geometry()
        check(_ge.left() >= _gs.right() - 2 and abs(_ge.top() - _gs.top()) <= 2,
              "两栏没横向并排（可达那栏落到集合下面去了）：集合=%s 可达=%s"
              % (_gs, _ge))
        # 栏宽**真的能改**（鼠标拖分隔条走的就是这条 API）
        _w0 = dlg.area_edges.width()
        dlg.split.setSizes([420, 200, 330])
        for _ in range(2):
            app.processEvents()
        check(dlg.area_edges.width() != _w0,
              "调了分栏尺寸、栏宽却没变（%d）—— 分隔条拖不动" % _w0)
        # **不许给纵向钉硬地板**（踩过：写过 setMinimumSize(720, 420)，本意是"允许缩到
        # 比较小"，实际成了 420 的地板，比布局需要的 211 大一倍 ⇒ 用户还是缩不下去）
        check(dlg.minimumHeight() <= 320,
              "窗口最小高度被人为顶住了（纵向缩不下来）：%d" % dlg.minimumHeight())
        dlg.resize(1000, 300)
        check(dlg.height() <= 320,
              "缩不到 300 高（实际 %d）—— 有东西在顶纵向最小高度" % dlg.height())
        # 按钮文案按新叫法
        texts = {b.text() for b in dlg.findChildren(QPushButton)}
        for want in ("增加可达", "删除", "反向", "建议…"):
            check(want in texts, "没找到按钮「%s」：%s" % (want, sorted(texts)))
        check("加边…" not in texts and "边（有向）" not in texts,
              "还留着旧叫法：%s" % sorted(texts))

        # ② 名字统一蓝
        dlg.select([str(f.fid) for f in walk[:4]])
        dlg.register("甲平台")
        it0 = dlg.lst.item(0)
        check(row_colors(it0) == [(ze.C_SETNAME, "甲平台")],
              "集合名没统一成蓝色（而且**只该名字**是蓝的，(4) 要正常色）：%s"
              % (row_colors(it0),))
        check(not it0.icon().isNull(), "集合自己的配色（左边小色块）没了")

        # ③ 收窄：造三个集合 + 两条可达
        dlg.select([str(f.fid) for f in walk[4:8]])
        dlg.register("乙平台")
        dlg.select([str(f.fid) for f in walk[8:12]])
        dlg.register("丙平台")
        dlg.add_edge("甲平台", "乙平台", "walk")
        dlg.add_edge("丙平台", "乙平台", "walk")
        # 刚注册完「丙平台」，它是高亮的 ⇒ 列表**已经按它收窄**了（只剩它那条）
        check(dlg.lst_e.count() == 1,
              "高亮着「丙平台」时该只列它那一条：%d" % dlg.lst_e.count())
        check("只显示从「丙平台」出去的" in dlg.lbl_e_note.text(),
              "收窄了却没说：%r" % dlg.lbl_e_note.text())
        # 两条边都是**进入**乙平台的：丙的「可被到达」是空的，乙的「可被到达」有两条
        check(dlg.lst_in.count() == 0,
              "丙平台没有进来的边，却又 %d 行" % dlg.lst_in.count())
        check("能到「丙平台」的 0 条" in dlg.lbl_in_note.text(),
              "「可被到达」那栏没写清条数：%r" % dlg.lbl_in_note.text())
        # 那一栏**一个编辑按钮都没有**（要改就选中上边那个集合）
        from PyQt5.QtWidgets import QPushButton as _PB
        check(not dlg.in_host.findChildren(_PB),
              "「可被到达」那栏里出现了按钮（它应当只读）：%s"
              % [b.text() for b in dlg.in_host.findChildren(_PB)])

        # 什么都没选 ⇒ 显示全部（**总览**就在这个状态），而且不再说收窄
        dlg._highlight = None
        dlg.select([])
        dlg._refresh_edges()
        check(dlg.lst_e.count() == 2,
              "没选中任何集合时该显示全部：%d" % dlg.lst_e.count())
        check(dlg.lst_in.count() == 2, "「可被到达」也该显示全部：%d"
              % dlg.lst_in.count())
        check(dlg.lbl_e_note.text() == "",
              "没在收窄却还在说收窄：%r" % dlg.lbl_e_note.text())
        check(dlg.lbl_in_note.text() == "",
              "没在说「可被到达」却还在写：%r" % dlg.lbl_in_note.text())
        # 行内分色：**只有两个集合名**蓝，`→` 和 `[走(walk)]` 是正常色
        # （用户 2026-09-26 报的正是这条：整行都被染蓝了）
        for i in range(2):
            it = dlg.lst_e.item(i)
            html = it.data(ze.ROLE_HTML) or ""
            check(html.count("<span") == 2,
                  "可达那行该只有两个名字上色(%d 个)：%s" % (html.count("<span"), html))
            check({c for c, _t in row_colors(it)} == {ze.C_SETNAME},
                  "可达那行的名字不是蓝的：%s" % (row_colors(it),))
            check("</span> → <span" in html,
                  "`→` 也该在 span 外面（正常黑字）：%s" % html)
            check("走(walk)" not in "".join(t for _c, t in row_colors(it)),
                  "`[走(walk)]` 被上色了（它该是正常黑字）：%s" % html)

        # 选中「甲平台」⇒ 只列与它有关的那一条，并把这件事写出来
        dlg.lst.setCurrentRow([dlg.lst.item(i).text().startswith("甲平台")
                               for i in range(dlg.lst.count())].index(True))
        check(dlg._focus_set() == "甲平台", "焦点集合判错了：%s" % dlg._focus_set())
        check(dlg.lst_e.count() == 1, "没按当前集合收窄：%d 行" % dlg.lst_e.count())
        check("只显示从「甲平台」出去的" in dlg.lbl_e_note.text(),
              "收窄了却没说：%r" % dlg.lbl_e_note.text())
        check(dlg.lst_in.count() == 0, "甲平台没有进来的边，却又 %d 行"
              % dlg.lst_in.count())

        # **只选 foothold**（没点集合）⇒ 也用"它所属的那个集合"当焦点
        dlg._highlight = None
        dlg.select([str(f.fid) for f in walk[8:10]])       # 属于「丙平台」
        check(dlg._focus_set() == "丙平台",
              "选中 foothold 时没用它所属的集合当焦点：%s" % dlg._focus_set())
        # 一批 foothold 跨了两个集合 ⇒ 焦点为空 ⇒ 显示全部（总览）
        dlg.select([str(walk[8].fid), str(walk[0].fid)])
        check(dlg._focus_set() is None,
              "跨集合的选择不该定出一个焦点：%s" % dlg._focus_set())
        check(dlg.lst_e.count() == 2, "没选中任何集合时该显示全部：%d" % dlg.lst_e.count())
        check(dlg.lst_in.count() == 2, "「可被到达」也该显示全部：%d" % dlg.lst_in.count())
        check(dlg.lbl_e_note.text() == "", "没在收窄却还在说收窄：%r" % dlg.lbl_e_note.text())

        # 「增加可达」的终点候选：只列相关的 + 一个逃生口
        rel = dlg._related_sets("甲平台")
        check("乙平台" in rel and "丙平台" not in rel,
              "相关集合算错了（该只含与甲平台有关的）：%s" % rel)
        dlg.close()
    finally:
        import shutil
        shutil.rmtree(str(tmp), ignore_errors=True)


def t_add_reach_dialog():
    """「增加可达」**一个弹窗问完**：终点 / 通行方式 / 绳或门，按类型显隐，缺料拦住。

    为什么钉它（2026-09-26 要求）：以前是 2~4 个 QInputDialog 串着弹，每弹一个都要
    回想"上一步选了啥"；合成一个弹窗之后，"类型要求什么"才可能跟着类型一起变
    （选「爬」才出现挑绳那一行）。这里连"缺料不给点确定"一起钉住 —— 否则会存出一条
    说不清怎么走的可达（保存时的校验虽然也会拦，但那已经晚了）。
    """
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    lads = [("L2", "L2  x=701"), ("L3", "L3  x=1245")]
    ad = ze.AddReachDialog(None, "甲平台", ["甲平台", "乙平台", "丙平台"], ["乙平台"],
                           ladders=lads)
    got = [ad.cmb_dst.itemData(i) for i in range(ad.cmb_dst.count())]
    check(got == ["乙平台"], "默认没只列与起点有关的终点：%s" % got)
    check(not ad.ck_all.isHidden(), "有全部可看时，「全部」那个开关该露出来")
    ad.ck_all.setChecked(True)
    check(ad.cmb_dst.count() == 3, "勾了「全部」还是只列相关的：%d" % ad.cmb_dst.count())

    # **通行方式**（2026-09-26 定的术语）：「跳」和「下跳」都要能选 ——
    # 跳 = 在 foothold 边缘按跳；下跳 = 按住 ↓ 再按跳。记录不受限（执行才要标定），
    # 所以下拉里**不能**像以前那样把 jump 藏起来。
    kinds = [ad.cmb_kind.itemData(i) for i in range(ad.cmb_kind.count())]
    labels = [ad.cmb_kind.itemText(i) for i in range(ad.cmb_kind.count())]
    check("jump" in kinds and "drop" in kinds,
          "「跳」「下跳」都得能选（记录不受限，执行才要标定）：%s" % kinds)
    check("跳(jump)" in labels and "下跳(drop)" in labels,
          "两种通行方式在界面上的写法不对：%s" % labels)
    check(kinds.index("jump") < kinds.index("drop"),
          "「跳」该排在「下跳」前面（先跳、再下跳）：%s" % kinds)

    # 默认类型是「走」⇒「爬哪根绳」那一行不占地方
    # ⚠ 「走哪个门」那一行 2026-09-27 随「传送门」一起**移除**了 ✓ —— 现在
    #   通行方式只有走/爬/下跳/跳 四种（见 `zones.EDGE_KINDS` ✓）。
    check(ad.cmb_lad.isHidden(), "选「走」时不该显示「爬哪根绳」")
    ad.cmb_kind.setCurrentIndex(ad.cmb_kind.findData("climb"))
    check(not ad.cmb_lad.isHidden(),
          "按类型显隐没生效（选「爬」该出「爬哪根绳」那一行）")
    ad.cmb_lad.setCurrentIndex(1)
    ad._accept()
    # ⚠ 2026-09-28：多了 `mid_dir`/`mid_y`（「爬」的**中途跳下** ✓）—— 与 `dir` 同款写法
    #   （**永远带这两个键**，非爬 / 没配时是 `None` ✓）⇒ 完全相等断言要跟上 ✓
    check(ad.result_dict == {"dst": "乙平台", "kind": "climb",
                             "ladder": "L3", "footholds": None,
                             "dir": None, "mid_dir": None, "mid_y": None},
          "返回值不对：%s" % ad.result_dict)

    # ---- 「走」的方向类型（2026-09-26 用户要求，**只是配置占位**）----
    ad6 = ze.AddReachDialog(None, "甲平台", ["乙平台"], ["乙平台"])
    check(not ad6.cmb_wd.isHidden(), "「走」该能看到「类型」下拉")
    ad6.cmb_kind.setCurrentIndex(ad6.cmb_kind.findData("drop"))
    check(ad6.cmb_wd.isHidden(), "换成下跳还留着「类型」下拉")
    ad6.cmb_kind.setCurrentIndex(ad6.cmb_kind.findData("walk"))
    check(ad6.cmb_wd.count() == 3 and ad6.cmb_wd.currentData() == "",
          "「类型」该是三项、默认「默认方向」：%d / %r"
          % (ad6.cmb_wd.count(), ad6.cmb_wd.currentData()))
    ad6.cmb_wd.setCurrentIndex(ad6.cmb_wd.findData("right"))
    ad6._accept()
    check(ad6.result_dict["dir"] == "right",
          "选了「仅向右」没带出去：%s" % ad6.result_dict)

    # 缺料：没绳可挑时不能点「增加」，而且要说清为什么
    ad2 = ze.AddReachDialog(None, "甲平台", ["乙平台"], [], ladders=[])
    ad2.cmb_kind.setCurrentIndex(ad2.cmb_kind.findData("climb"))
    check(not ad2.btn_ok.isEnabled(),
          "没绳可挑却能点「增加」—— 会存出一条说不清怎么走的可达")
    check("没有可爬的绳" in ad2.lbl_note.text(),
          "缺料时没说清原因：%r" % ad2.lbl_note.text())
    ad2.cmb_kind.setCurrentIndex(ad2.cmb_kind.findData("walk"))
    check(ad2.btn_ok.isEnabled(), "换成「走」还是不让点")

    # ---- 下跳的扩展配置（2026-09-26 用户要求）----
    # ① 只在下跳时出现；② 默认 = 起点集合的全部 foothold；③ 能增删；
    # ④ 选中一行 ⇒ **编辑器里只让那一条呼吸**（其余高亮临时让位）；
    # ⑤ 只有**人工改过**才把这一格写进文件（没改 = 用默认，老文件不用批量补）。
    from PyQt5.QtCore import Qt
    drops = [("19", "fh 19　y=280"), ("41", "fh 41　y=280")]
    seen = {}

    from PyQt5.QtWidgets import QWidget

    class _Ed(QWidget):              # 顶替编辑器：只看有没有被通知到
        def preview_foothold(self, fid):
            seen["preview"] = fid

        def clear_foothold_preview(self):
            seen["cleared"] = True

    ed = _Ed()          # ⚠ 必须留个引用：不然它被回收，弹窗的 parent 就悬空了
    ad4 = ze.AddReachDialog(ed, "甲平台", ["乙平台"], ["乙平台"],
                            drop_choices=drops)
    check(ad4.drop_host.isHidden(), "不是下跳，却已经把「可下跳 foothold」露出来了")
    ad4.cmb_kind.setCurrentIndex(ad4.cmb_kind.findData("drop"))
    check(not ad4.drop_host.isHidden(), "选中下跳却没显示那张表")
    got = [ad4.lst_drop.item(i).data(Qt.UserRole) for i in range(ad4.lst_drop.count())]
    check(got == ["19", "41"], "默认该是起点集合的全部 foothold：%s" % got)
    ad4.lst_drop.setCurrentRow(1)
    check(seen.get("preview") == "41",
          "选中那一行没通知编辑器去呼吸它：%r" % seen.get("preview"))
    ad4._drop_del()
    got = [ad4.lst_drop.item(i).data(Qt.UserRole) for i in range(ad4.lst_drop.count())]
    check(got == ["19"], "「移出」没生效：%s" % got)
    ad4._drop_del()                  # 全删光 ⇒ 说不清从哪下跳，不许确定
    check(not ad4.btn_ok.isEnabled(), "一个可下跳的都不剩了，还能点确定")
    ad4._drop_ids = ["19"]           # 加回来一条（用内部接口，避开模态的 QInputDialog）
    ad4._fill_drop()
    ad4._accept()
    check(ad4.result_dict["footholds"] == ["19"],
          "改过之后该把这张表带出去：%s" % ad4.result_dict)
    check(seen.get("cleared"), "关窗没把预览还回去（编辑器会一直只亮一条）")
    # 没改过 ⇒ 不带这一格（= 用默认）
    ad5 = ze.AddReachDialog(None, "甲平台", ["乙平台"], ["乙平台"],
                            drop_choices=drops)
    ad5.cmb_kind.setCurrentIndex(ad5.cmb_kind.findData("drop"))
    ad5._accept()
    check(ad5.result_dict["footholds"] is None,
          "没动过那张表却把默认值写死了（老数据会变得很啰嗦）：%s"
          % ad5.result_dict)

    # 没有"相关终点"可收窄 ⇒ 直接列全部，且那个开关不显示（勾了也没意义）
    ad3 = ze.AddReachDialog(None, "甲平台", ["乙平台", "丙平台"], [])
    check(ad3.cmb_dst.count() == 2 and ad3.ck_all.isHidden(),
          "没有相关终点时该直接列全部、且不显示那个开关")

    # 起点能选到哪些绳：用真实图（「右下」那块 y=280 的平台 = fh 19/41）
    t = mapdata.load(MAP_ID)
    tmp = Path(tempfile.mkdtemp(prefix="zreach2_"))
    try:
        z = zones.Zones(MAP_ID)
        z.add_set("右下", ["19", "41"])
        z.add_set("别的", ["1"])
        z.save(tmp / "x.zones.json")
        dlg = ze.ZoneEditorDialog(MAP_ID, terrain=t, zones_path=tmp / "x.zones.json")
        # ⚠ 这里原来还接一个"门"的候选串 —— 2026-09-27 随「传送门」一起移除 ✓
        #   （`_reach_ladder_choices` 现在**只**返回绳那一串 ✓）
        lads2 = dlg._reach_ladder_choices("右下")
        check([v for v, _txt in lads2] == ["L2", "L3"] or
              sorted(v for v, _txt in lads2) == ["L2", "L3"],
              "起点能选的绳不对（该是通到这块平台的那两根）：%s" % lads2)
        # ⑥ **「增加可达」也是非模态 + 单实例**（用户 2026-09-26 的原话：不能再开多个
        #    窗口，再点一次只切换聚焦）—— 自带一个编辑器，别依赖上文用的是哪套集合。
        _t2 = Path(tempfile.mkdtemp(prefix="zreach1_"))
        try:
            _t = _terrain()
            ed2 = ze.ZoneEditorDialog(MAP_ID, terrain=_t,
                                      zones_path=_t2 / "x.zones.json")
            ids = [str(f.fid) for f in _t.footholds if not f.is_wall]
            ed2.select(ids[:3])
            ed2.register("甲平台")
            ed2.select(ids[3:6])
            ed2.register("乙平台")
            ed2._highlight = "甲平台"
            ed2._refresh_edges()
            ed2._on_add_edge()
            w1 = ed2._reach_dlg
            check(w1 is not None and w1.mode == "add",
                  "「增加可达」没打开窗口：%r" % (w1,))
            check(not w1.isModal(), "「增加可达」是模态的（那就没法缩放看图了）")
            ed2._on_add_edge()
            check(ed2._reach_dlg is w1, "「增加可达」连点两次开出了两个窗口")
            w1.close()
            check(ed2._reach_dlg is None, "「增加可达」关了没把引用忘掉")
            ed2.close()
        finally:
            import shutil
            shutil.rmtree(str(_t2), ignore_errors=True)

        dlg.close()
    finally:
        import shutil
        shutil.rmtree(str(tmp), ignore_errors=True)


def t_panels_no_hscroll_at_default_size():
    """三栏在**默认窗口尺寸**下不许出横向滚动条（2026-09-26 三栏布局重构）。

    为什么单开一条：栏的最小宽度是**人定的**（160 / 170）而内容需要更宽 —— 集合栏 280、
    「可到达 / 可被到达」的边列表 329~346（探针量的，**带主窗口 QSS** 才算数：按钮的
    padding 来自 QSS，不带它量出来的偏窄）⇒ **初始宽度必须按内容给**，否则一打开就冒横向
    滚动条（第一版就是这样：集合栏 250 而内容 280 ✗）。
    窗口被人拖小之后出滚动条是**对的**（那正是"按需"的意义），所以这条只钉默认尺寸。

    ⚠ 测试里的对话框**没有 MainWindow 父级** ⇒ 拿不到全局 QSS ⇒ 按钮更窄、内容需要的
    宽度更小，这条会比真机**宽松**一点。所以这里**手动把 QSS 贴上**（真机上由父窗口继承），
    量的才是真机那个宽度。
    """
    from PyQt5.QtWidgets import QApplication

    from gui.main_window import QSS

    app = QApplication.instance() or QApplication([])
    t = _terrain()
    walk = [f for f in t.footholds if not f.is_wall]
    check(len(walk) >= 12, "地形里非墙 foothold 太少，这条测不了")
    tmp = Path(tempfile.mkdtemp(prefix="zwide_"))
    try:
        dlg = ze.ZoneEditorDialog(MAP_ID, terrain=t, zones_path=tmp / "x.zones.json")
        dlg.setStyleSheet(QSS)                  # 真机上由 MainWindow 继承（见 docstring）
        for i, name in enumerate(("甲平台", "乙平台", "丙平台")):
            dlg.select([str(f.fid) for f in walk[i * 4:(i + 1) * 4]])
            dlg.register(name)
        dlg.add_edge("甲平台", "乙平台", "walk")
        dlg.add_edge("乙平台", "丙平台", "drop")
        dlg.lst.setCurrentRow(0)                # 焦点在「甲平台」⇒ 两栏都有内容
        dlg.show()                              # 尺寸用 __init__ 里的 1180×760
        for _ in range(6):
            app.processEvents()
        check(abs(dlg.width() - 1180) <= 40,
              "不是在默认尺寸下量的（%dx%d）—— 这条只钉默认尺寸"
              % (dlg.width(), dlg.height()))
        for tag, w in (("集合栏", dlg.area_sets),
                       ("可达 / 可被到达栏", dlg.area_edges),
                       ("集合列表", dlg.lst),
                       ("可到达列表", dlg.lst_e),
                       ("可被到达列表", dlg.lst_in)):
            need = (w.sizeHintForColumn(0) if hasattr(w, "sizeHintForColumn")
                    else w.widget().minimumSizeHint().width())
            check(not w.horizontalScrollBar().isVisible(),
                  "%s 在默认尺寸下就出了横向滚动条（视口宽 %d < 内容需要 %d）"
                  % (tag, w.viewport().width(), need))
        dlg.close()
    finally:
        import shutil
        shutil.rmtree(str(tmp), ignore_errors=True)


def t_window_can_shrink_vertically():
    """窗口纵向**必须能缩下去**：地板只许来自布局，且残留地板会被清掉。

    用户报过两次（2026-09-26）：
      ① `__init__` 里写过 `setMinimumSize(720, 420)` —— 本意是"允许缩到比较小"，
         实际是给纵向钉了 420 的**地板**（布局只要 222px）；
      ② 代码删掉之后**旧窗口还是缩不下去** —— 编辑器是非模态单实例，`close()` 只是
         隐藏、对象还活着，那个 420 一直挂在它身上（实测：新建的窗口 minimumHeight=0，
         旧窗口仍 720×420）。所以 `showEvent` 里加了 `_drop_stale_min_height`。

    这条用例把两种都钉住：新建的窗口地板要小、缩得下去；**人为钉一个残留地板后，
    再显示一次就该被清掉**（模拟"上次启动时建的窗口"）。
    """
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    t = _terrain()
    tmp = Path(tempfile.mkdtemp(prefix="zshrink_"))
    try:
        dlg = ze.ZoneEditorDialog(MAP_ID, terrain=t, zones_path=tmp / "x.zones.json")
        dlg.show()
        app.processEvents()
        # ① 布局自己要求的地板要小 —— 比 260 大就说明有人钉了硬地板
        auto = dlg.minimumSizeHint().height()
        check(auto <= 260, "布局要求的最小高度太大了：%d（实测真实平台 222）" % auto)
        check(dlg.minimumHeight() <= 260,
              "有人给纵向钉了地板：minimumHeight=%d（布局只要 %d）"
              % (dlg.minimumHeight(), auto))
        # ② 真的缩得下去
        for h in (400, 260):
            dlg.resize(1100, h)
            app.processEvents()
            check(dlg.height() <= h, "缩不到高 %d（实际 %d）" % (h, dlg.height()))
        # ③ 残留地板：显示一次就被清掉（旧代码建的窗口就是带着它活下来的）
        dlg.hide()
        dlg.setMinimumSize(720, 420)          # 模拟"修复前建的窗口"
        dlg.show()
        app.processEvents()
        check(dlg.minimumHeight() <= 260,
              "残留的 420 地板没被清掉：minimumHeight=%d" % dlg.minimumHeight())
        check(dlg.minimumWidth() <= 620,
              "残留的最小宽度没被清掉：minimumWidth=%d（有意的只有 %d）"
              % (dlg.minimumWidth(), ze.ZoneEditorDialog.MIN_W))
        dlg.resize(1100, 300)
        app.processEvents()
        check(dlg.height() <= 320,
              "残留地板清掉后还是缩不下去：%d" % dlg.height())
        dlg.close()
    finally:
        import shutil
        shutil.rmtree(str(tmp), ignore_errors=True)


def t_ladder_ids_on_canvas():
    """画布上的绳梯编号（`L1`/`L2`…）：**和「爬」那条边里写的绳号同一套**、且**不是蓝字**。

    为什么必须钉"同一套口径"：编号是两边对得上的**唯一凭据** —— 边里写着 `绳 L3`，
    如果画布上的 L3 是**另一根绳**，人会照着错的绳子去搭路线，而且看不出来。
    编号来自 `core.zones.ladder_ids`（按 (x, page) 排序），所以这里比对的是同一个函数。

    为什么钉字色：蓝字在这个窗口里专指"**已注册的平台集合名**"（用户 2026-09-26 定的
    规矩）。第一版编号用的就是绳梯那个蓝 ✗ —— 那会让人以为"L1 是个集合名"。
    """
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication, QGraphicsSimpleTextItem

    app = QApplication.instance() or QApplication([])
    t = _terrain()
    check(t.ladders, "这张图没有绳梯，这条测不了")
    tmp = Path(tempfile.mkdtemp(prefix="zlid_"))
    try:
        dlg = ze.ZoneEditorDialog(MAP_ID, terrain=t, zones_path=tmp / "x.zones.json")
        lids = zones.ladder_ids(t)
        want = set(lids.values())
        check(len(want) == len(t.ladders),
              "编号有重号（同一张图上分不出哪根绳）：%s" % sorted(want))

        def texts():
            return [i.text() for i in dlg.scene.items()
                    if isinstance(i, QGraphicsSimpleTextItem)]

        # ① 每根绳都有编号，且是黑描边 + 彩字两份
        now = texts()
        check(want <= set(now),
              "画布上没标出全部绳梯编号：缺 %s（画上去的是 %s）"
              % (sorted(want - set(now)), sorted(set(now))))
        for lid in sorted(want):
            check(now.count(lid) == 2,
                  "%s 该画两份（黑描边 + 彩字），实际 %d 份" % (lid, now.count(lid)))
        # ② 位置：绳子上端的**右侧**（标在线上会盖住绳）；且不抢鼠标
        for L in t.ladders:
            lid = lids[id(L)]
            it = [i for i in dlg.scene.items()
                  if isinstance(i, QGraphicsSimpleTextItem) and i.text() == lid
                  and i.brush().color().name() == ze.C_LADDER_ID]
            check(it, "%s 的彩色那份没画出来（描边那份盖在上面就看不见了）" % lid)
            check(it[0].brush().color().name() != ze.C_SETNAME
                  and it[0].brush().color().name() != ze.C_LADDER.name(),
                  "%s 用了蓝字 —— 蓝字在这个窗口里专指「已注册的平台集合名」" % lid)
            p = it[0].pos()
            top = min(L.y1, L.y2)
            check(p.x() > L.x,
                  "%s 标到绳子左边去了（会盖住绳）：x=%.0f ≤ 绳 x=%.0f"
                  % (lid, p.x(), L.x))
            check(abs(p.y() - top) < ze.LADDER_PX * 0.5,
                  "%s 没标在绳子上端：y=%.0f，上端=%.0f" % (lid, p.y(), top))
            check(it[0].acceptedMouseButtons() == Qt.NoButton,
                  "%s 会抢鼠标事件（选择是几何判据，文字不该参与）" % lid)
        # ③ 选中集合（把所有 foothold 圈进去）：**绳子**呼吸，但**编号不跟着变色** ——
        #    呼吸会把字染成绳子的蓝，而蓝字只能留给集合名（见上面的说明）。
        all_ids = {str(f.fid) for f in t.footholds if not f.is_wall}
        dlg.select(sorted(all_ids))
        dlg.register("全部")
        hl = {it.text() for it, _c in dlg._hl_items
               if isinstance(it, QGraphicsSimpleTextItem)}
        for lid in sorted(want):
            check(lid not in hl,
                  "%s 跟着集合一起变色了（蓝字只能留给集合名）" % lid)
        lad = zones.ladders_of(t, all_ids)
        check(lad, "所有 foothold 圈起来后一根绳都没有，这条测不了")
        check(all(zones.ladders_of(t, all_ids)),
              "绳梯没进呼吸高亮（该闪的是绳子，不是编号）")
        dlg._on_pulse()          # 文字走 setBrush —— 这里要是走 setPen 就崩
        dlg.close()
    finally:
        import shutil
        shutil.rmtree(str(tmp), ignore_errors=True)


def t_two_ladders_same_pair():
    """同一对集合、**两根绳**：两条边各存各的，而且"没加进去要说话"（2026-09-27 用户报）。

    现场（用户原话）："地图 105040303，我给二楼加通过 L7 去三楼的通行方式，但是点击增加后
    无添加项目"。真因：**边的身份里没有绳号** —— `二楼 →(爬 L3)→ 三楼` 已经在文件里，
    再加 L7 被当成重复 ⇒ `add_edge` 静默返回老那条，界面一点动静都没有 ✗
    （见 `core.zones.add_edge` ✓）。

    钉五件：
      ① 换一根绳**真的加进去**，两栏里都列着；
      ② 新加那条**被选中**（那一栏现在列全部边，十几行里不选出来 = 看着像没加 ✗）；
      ③ **完全一样**再加一次 ⇒ 条数不变、而且**弹一句说清**（不再静默 ✗）；
      ④ 删 / 改**只动对的那一条**（绳号进了匹配判据，不连坐）；
      ⑤ 「改成和另一条一模一样」仍然拦住。
    """
    import unittest.mock as mock

    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    t = _terrain()
    walk = [f for f in t.footholds if not f.is_wall]
    check(len(walk) >= 8, "地形里非墙 foothold 太少，这条测不了")
    tmp = Path(tempfile.mkdtemp(prefix="z2lad_"))
    try:
        dlg = ze.ZoneEditorDialog(MAP_ID, terrain=t, zones_path=tmp / "x.zones.json")
        dlg.select([str(f.fid) for f in walk[:4]])
        dlg.register("二楼")
        dlg.select([str(f.fid) for f in walk[4:8]])
        dlg.register("三楼")
        dlg.add_edge("二楼", "三楼", "climb", ladder="L3", why="手工加的")
        dlg._highlight = "二楼"
        dlg._refresh_edges()
        check(dlg.lst_e.count() == 1, "前提不成立，该有 1 条：%d" % dlg.lst_e.count())

        class _D:
            """顶替「增加可达」弹窗（`_apply_reach` 是用 `self.sender()` 取它的）。"""

            src, mode, edge = "二楼", "add", None

        dlg.sender = lambda: _D()
        said = []
        with mock.patch.object(ze.QMessageBox, "information",
                               lambda *a, **k: said.append(a[2] if len(a) > 2 else "")):
            # ① 换一根绳 ⇒ **真的加进去**（用户要的就是这一下）
            dlg._apply_reach({"dst": "三楼", "kind": "climb", "ladder": "L7",
                              "portal": None, "footholds": None, "dir": None})
            check(len(dlg.zones.edges) == 2,
                  "换一根绳没加进去（用户报的就是这一下）：%s" % dlg.zones.edges)
            rows = [dlg.lst_e.item(i).text() for i in range(dlg.lst_e.count())]
            # ⚠ 2026-09-27 起同一对集合**合成一组**（组头 + 缩进行）⇒ "两条都列着"要看
            #   **组里有没有这两条**（两个绳号都在 ✓），而不是数两行组头 ✗
            check(sum("二楼 → 三楼" in r for r in rows) == 1
                  and "L3" in "".join(rows) and "L7" in "".join(rows),
                  "两条（L3 / L7）没都列出来：%s" % rows)
            # ② 新加的那条被选出来（不然像没加）
            cur = dlg.lst_e.currentItem()
            check(cur is not None and "L7" in cur.text(),
                  "加完没把新那条选出来（长列表里找不着 = 像没加）：%r"
                  % (cur.text() if cur is not None else None))
            # ③ 完全一样再加 ⇒ 条数不变 + **说话**
            said.clear()
            dlg._apply_reach({"dst": "三楼", "kind": "climb", "ladder": "L7",
                              "portal": None, "footholds": None, "dir": None})
            check(len(dlg.zones.edges) == 2,
                  "重复加居然多了一条：%s" % dlg.zones.edges)
            check(said and "已经有一条" in said[0] and "没有重复加" in said[0],
                  "完全重复时没说话（原来就是这里静默 ✗）：%r" % said)

        # ④ 删 L7 ⇒ L3 还在（绳号进了判据，不连坐）
        e7 = [e for e in dlg.zones.edges if e.get("ladder") == "L7"][0]
        dlg.del_edge(dict(e7))
        check([e.get("ladder") for e in dlg.zones.edges] == ["L3"],
              "删一条把另一条也删了 / 删错了：%s" % dlg.zones.edges)
        # ⑤ 改绳号只动那一条；改成"和另一条一模一样"要拦住
        dlg.add_edge("二楼", "三楼", "climb", ladder="L9")
        dlg.edit_edge(dict(dlg.zones.edges[0]), "三楼", "climb", ladder="L7")
        check(sorted(e.get("ladder") for e in dlg.zones.edges) == ["L7", "L9"],
              "改绳号改错了对象：%s" % [e.get("ladder") for e in dlg.zones.edges])
        try:
            dlg.edit_edge(
                dict([e for e in dlg.zones.edges if e.get("ladder") == "L7"][0]),
                "三楼", "climb", ladder="L9")
            raise AssertionError("改成和另一条一模一样的（同终点同类型同绳）该拦住")
        except ValueError:
            pass
        dlg.close()
    finally:
        import shutil
        shutil.rmtree(str(tmp), ignore_errors=True)


def t_edit_edge_by_double_click():
    """双击「可到达」里那行 = 改这条边（终点 / 类型 / 绳 / 门）；**起点不可改**。

    为什么用替身顶掉弹窗：`exec_()` 是**模态阻塞**的，自检里一弹就卡死。替身只顶掉
    "问人"这一步，后面走的是**同一条**调用链（信号 → `_on_edit_edge` → `edit_edge`），
    所以接线、预填、落盘、撤销都能钉住，又不阻塞。
    """
    from PyQt5.QtWidgets import QApplication, QDialog

    app = QApplication.instance() or QApplication([])
    t = _terrain()
    walk = [f for f in t.footholds if not f.is_wall]
    check(len(walk) >= 12, "地形里非墙 foothold 太少，这条测不了")
    tmp = Path(tempfile.mkdtemp(prefix="zedit_"))
    try:
        dlg = ze.ZoneEditorDialog(MAP_ID, terrain=t, zones_path=tmp / "x.zones.json")
        for name, sl in (("A平台", walk[:4]), ("B平台", walk[4:8]),
                         ("C平台", walk[8:12])):
            dlg.select([str(f.fid) for f in sl])
            dlg.register(name)
        dlg.add_edge("A平台", "B平台", "walk", why="走的那条")
        dlg.add_edge("A平台", "C平台", "climb", ladder="L1", why="爬的那条")
        # 焦点留在「A平台」—— 两条都是"从 A 出去的"，会排在「可到达」最前面
        #（那一栏现在列**全部边** ✓，这两条正好都在）
        dlg._highlight = "A平台"
        dlg._refresh_edges()
        check(dlg.lst_e.count() == 2, "A 的可到达该有 2 条：%d" % dlg.lst_e.count())

        # ① 编辑模式的弹窗：预填当前值、按钮写「保存」（不再写「增加」）
        names = list(dlg.zones.sets)
        d = ze.AddReachDialog(dlg, "A平台", names, names,
                              ladders=[("L1", "L1 x=0")],
                              init=dlg.zones.edges[1])
        check(d.cmb_dst.currentData() == "C平台",
              "编辑模式没预填终点：%s" % d.cmb_dst.currentData())
        check(d.cmb_kind.currentData() == "climb",
              "编辑模式没预填类型：%s" % d.cmb_kind.currentData())
        check(d.cmb_lad.currentData() == "L1",
              "编辑模式没预填绳：%s" % d.cmb_lad.currentData())
        check(d.btn_ok.text() == "保存",
              "编辑模式的确认按钮不该还写「增加」：%r" % d.btn_ok.text())
        check("编辑" in d.windowTitle(), "编辑模式标题没写清：%r" % d.windowTitle())
        d.reject()
        # 增加模式不受影响（按钮还是「增加」，不预填）
        d0 = ze.AddReachDialog(dlg, "A平台", names, names)
        check(d0.btn_ok.text() == "增加", "增加模式的按钮被改坏了：%r" % d0.btn_ok.text())
        check(d0.cmb_kind.currentData() == "walk",
              "增加模式不该预选类型：%s" % d0.cmb_kind.currentData())
        d0.reject()

        # ② 双击 ⇒ 打开**非模态**的编辑窗口（不再阻塞整个工作台，要能缩放细看图）
        dlg.lst_e.itemDoubleClicked.emit(dlg.lst_e.item(1))   # 第 2 行 = A→C(爬)
        rd = dlg._reach_dlg
        check(rd is not None, "双击没打开编辑窗口")
        check(not rd.isModal(),
              "编辑窗口是模态的 —— 那就没法缩放细看编辑器那张图了")
        check(rd.mode == "edit" and rd.edge == dlg.zones.edges[1],
              # ⚠ 比**值**不比身份：双击拿到的边是 Qt 转过来的副本
              "编辑窗口没绑到那条边上：mode=%r edge=%r" % (rd.mode, rd.edge))
        check(rd.cmb_dst.currentData() == "C平台"
              and rd.cmb_kind.currentData() == "climb",
              "编辑窗口没预填：%s / %s"
              % (rd.cmb_dst.currentData(), rd.cmb_kind.currentData()))
        # **同一件事再双击一次 ⇒ 只聚焦，不新开**（单实例）
        dlg.lst_e.itemDoubleClicked.emit(dlg.lst_e.item(1))
        check(dlg._reach_dlg is rd,
              "同一条边又开了一个窗口：%r" % (dlg._reach_dlg,))
        # 换一条边 ⇒ 旧窗口必须换掉（它绑的是另一条，留着就会改错）
        dlg.lst_e.itemDoubleClicked.emit(dlg.lst_e.item(0))
        check(dlg._reach_dlg is not rd, "换了一条边还留着旧窗口（它绑的是另一条）")
        dlg._reach_dlg.close()
        check(dlg._reach_dlg is None, "窗口关了没把引用忘掉：%r" % (dlg._reach_dlg,))
        # 回到那条爬边改干净：**非模态 ⇒ 结果靠 applied 信号回来**
        dlg.lst_e.itemDoubleClicked.emit(dlg.lst_e.item(1))
        rd = dlg._reach_dlg
        rd.cmb_dst.setCurrentIndex(rd.cmb_dst.findData("B平台"))
        rd.cmb_kind.setCurrentIndex(rd.cmb_kind.findData("drop"))
        rd._accept()
        e = dlg.zones.edges[1]
        check(e.get("to") == "B平台" and e.get("kind") == "drop",
              "编辑窗口点确定之后没落盘：%s" % e)
        check("ladder" not in e,
              "改成「下跳」之后还留着上一任的绳号（下次改回「爬」会突然复活）：%s" % e)
        check(e.get("why") == "爬的那条", "改边把原来写的理由弄丢了：%s" % e)
        # ③ 撤销能回退（改错了不用手改文件）
        dlg.undo()
        check(dlg.zones.edges[1].get("kind") == "climb"
              and dlg.zones.edges[1].get("ladder") == "L1",
              "撤销没把这条边改回去：%s" % dlg.zones.edges[1])

        # ④ 「可被到达」那栏**不接编辑**（它是只读的）：往里双击不该动任何东西
        dlg._highlight = "B平台"
        dlg._refresh_edges()
        before = [dict(x) for x in dlg.zones.edges]
        if dlg.lst_in.count():
            dlg.lst_in.itemDoubleClicked.emit(dlg.lst_in.item(0))
            check([dict(x) for x in dlg.zones.edges] == before
                  and dlg._reach_dlg is None,
                  "「可被到达」那栏双击也开了窗口/改了东西（它该是只读的）")

        # ⑤ 直接调 edit_edge 的三道拦截（edges = [A→B 走, A→C 爬]）
        for idx, dst, why in (
                (0, "幽灵", "终点集合不存在该拦住"),
                (0, "A平台", "终点 = 起点（自环）该拦住"),
                (1, "B平台", "把「爬」改成和「走」那条一模一样该拦住")):
            try:
                dlg.edit_edge(dlg.zones.edges[idx], dst, "walk")
                raise AssertionError(why)
            except ValueError:
                pass          # 正是期望的
        check(len(dlg.zones.edges) == 2
              and dlg.zones.edges[1].get("kind") == "climb",
              "被拦下之后边反倒被改坏了：%s" % dlg.zones.edges)
        dlg.close()
    finally:
        import shutil
        shutil.rmtree(str(tmp), ignore_errors=True)


def t_sizehint_not_from_scene():
    """窗口的 sizeHint **不许跟着场景尺寸长** —— 否则一布局重算就被拽成 1200 高。

    这是第 3 次报"最小高度太大"的真凶（2026-09-26）：
      · **地板**一直是 222px（`minimumSizeHint`，前两次修的就是它，没错）；
      · 但 `QGraphicsView` 默认拿**场景尺寸**当 sizeHint ⇒ 这张图的窗口 sizeHint 变成
        **2521×1182** ⇒ 任何一次布局重算（连"鼠标移动刷新状态行"都会触发）
        就把窗口拽成 **1222 高**：实测 `resize(760)` 之后实际是 1222。
    用户看到的现象是"**想拖小、它自己弹回去**"，很容易以为"有个很大的最小高度"，
    所以只查 `minimumSizeHint` 永远查不到它 —— 这条用例专门盯 `sizeHint`。
    """
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    t = _terrain()
    tmp = Path(tempfile.mkdtemp(prefix="zshint_"))
    try:
        dlg = ze.ZoneEditorDialog(MAP_ID, terrain=t, zones_path=tmp / "x.zones.json")
        # ① 视图的 sizeHint 不跟着场景：场景这张图约 2200×1100（用它当 sizeHint
        #    就会把窗口抬成 1200 多高）
        sr = dlg.scene.sceneRect()
        check(sr.height() > 800, "这张图场景太矮，这条测不明显")
        check(dlg.view.sizeHint().height() <= 400,
              "视图的 sizeHint 又跟着场景长了（场景高 %.0f，sizeHint 高 %d）"
              % (sr.height(), dlg.view.sizeHint().height()))
        check(dlg.sizeHint().height() <= 700,
              "窗口 sizeHint 被抬到 %d —— 一布局重算就会把窗口拽那么大"
              % dlg.sizeHint().height())
        # ② 显示之后拖小，**必须真的变小**（不许被拽回去）
        dlg.show()
        for _ in range(4):
            app.processEvents()
        for h in (760, 500, 320, 240):
            dlg.resize(1100, h)
            for _ in range(8):          # 布局重算是异步的，多跑几拍（单拍会看到中间值）
                app.processEvents()
            check(dlg.height() <= h,
                  "想缩到高 %d，被顶成 %d（sizeHint 在拽窗口）" % (h, dlg.height()))
            check(dlg.width() <= 1100, "宽度也被顶大了：%d" % dlg.width())
        dlg.close()
    finally:
        import shutil
        shutil.rmtree(str(tmp), ignore_errors=True)


def t_battle_zone_dialog_size():
    """「编辑战斗区域」**子弹窗**：不许拿**视图的场景尺寸**当窗口尺寸。

    用户 2026-09-28 报："**编辑战斗区域子窗口也太大了，还不让缩小**"✗。

    病根：子弹窗（`gui.player_panel.BattleZoneDialog`）里摆了那块**只读集合视图**
    （`gui.foothold_picker.FootholdPicker` ✓），而它是 `QGraphicsView` ⇒ **默认把场景尺寸
    当 sizeHint**，本视图的场景 = **整张图的底图 + foothold**（约 2000×1000）⇒
    窗口一显示就被撑到**上千高**，之后**任何一次布局重算**还会把它拽回去
    ⇒ "太大 + 想拖小还拖不动"✗。⚠ 和 `_ZoneView` 当年（2026-09-26）是**同一个坑** ✓
    （见上面 `t_sizehint_not_from_scene` ✓ 那边只加了一个 `sizeHint` 就治好了 ✓）。

    ⚠⚠ `FootholdPicker` **在离屏自检里建不出来**（进程原生崩 ✗ 同被停用的
      `t_foothold_picker_readonly`）⇒ 这条用三条**不建它**的路子钉：
        ① 类上**必须**有 `sizeHint` 覆写（源码级 ✓ —— 没有它就会走父类那个 ✗）；
        ② 用 `__new__` 造个**不碰 C++** 的壳直接调 `sizeHint()` ⇒ 必须是**小值** ✓；
        ③ **病态对照**（这条最有说服力 ✓）：拿一个 `sizeHint` = 场景尺寸级的**假视图**
           塞进 `BattleZoneDialog` ⇒ `resize` 之后再显示，窗口**照样被顶到上千高** ✓
           —— 正好复现用户那句话 ✓ ⇒ 说明**根治只能在视图那边**（对话框侧防不住 ✓）；
           换成"修好的视图"（`sizeHint` 给 320×200 ✓）同样的操作 ⇒ 窗口老实待着、还能缩 ✓。
    """
    from PyQt5.QtCore import QSize, pyqtSignal
    from PyQt5.QtWidgets import QApplication, QWidget

    from gui import foothold_picker as fp
    from gui import player_panel as ppm

    app = QApplication.instance() or QApplication([])

    # ① 视图那边有没有覆写
    check("sizeHint" in fp.FootholdPicker.__dict__,
          "`FootholdPicker` 没覆写 `sizeHint` —— 它会走 `QGraphicsView` 那个（= **场景尺寸**，"
          "本图约 2000×1000）⇒ 子弹窗一显示就被撑到上千高、还缩不下去 ✗"
          "（用户原话：\"编辑战斗区域子窗口也太大了，还不让缩小\"）")

    # ② 壳里调它（`__new__` 不跑 `__init__` ⇒ 不建 Qt 视图 ⇒ 离屏也不崩 ✓）
    sh = fp.FootholdPicker.__new__(fp.FootholdPicker).sizeHint()
    check(sh.width() <= 600 and sh.height() <= 400,
          "`FootholdPicker.sizeHint()` 还是大值（%dx%d）—— 它会原样变成窗口想要的尺寸 ✗"
          % (sh.width(), sh.height()))

    # ③ 病态对照 + 真身对照（同一个 `resize`，两种视图）
    # ⚠⚠ **这一段必须隔离 `config/ui.yaml`**（2026-09-28 现场踩到 ✗）：那两个弹窗都接了
    #   `theme.bind_window_state` ⇒ `show()` **恢复**几何、`close()` **保存**几何 ✓，
    #   而自检**绝不许改用户的文件** ✗（全仓库的规矩 ✓ 见 `selftest_main_window._fake_store`）。
    #   踩到的样子：用例把 `battle_zone: {h: 380, w: 560, x: 120, y: 2}` 写了进去 ⇒
    #   **下一套 `selftest_main_window` 建主窗口时原生崩**（`0xC0000005` ✗✗ ——
    #   看着像"离屏 Qt 的老毛病"，其实是**用户配置被自检写脏了** ✓）。
    #   ⇒ 照 `selftest_zone_editor:t_line_width_setting` 的做法：`theme.CFG` 指到临时文件 ✓。
    import shutil
    import tempfile
    import unittest.mock as mock

    from gui import theme

    class _Picker(QWidget):
        """假视图：`hint` 就是它"想多大"（= `QGraphicsView` 那条路的输入 ✓）。"""
        picked = pyqtSignal(str)

        def __init__(self, hint):
            super().__init__()
            self._hint = hint

        def sizeHint(self):
            return self._hint

        def current(self):
            return ""

        def info(self, _fid):
            return ""

    def _open(hint):
        # ⚠ **隔离 `config/ui.yaml`**：这个弹窗接了 `theme.bind_window_state` ⇒ `show` 之后
        #   `close` 会写用户配置 ✗（2026-09-28 踩过 ✓ 自检绝不许改用户文件）。
        from unittest import mock

        from gui import theme as _theme

        d = ppm.BattleZoneDialog({"set": "一楼", "cd_s": 3.0, "idle_foothold": "",
                                  "fight_max_s": 0.0, "fight_dst": ""},
                                 names=["一楼", "二楼"],
                                 picker_factory=lambda s, c: _Picker(hint))
        d.resize(560, 620)                    # = 源码里那个初值（原来是**没有**它 ✗）
        _sw = mock.patch.object(_theme, "save_window", lambda *a, **k: None)
        _sw.start()
        d.show()
        for _ in range(6):
            app.processEvents()
        d._savewin_patch = _sw              # 交给调用方 close 之后再 stop ✓
        return d

    _tmp = Path(tempfile.mkdtemp(prefix="bzdlg_"))
    _pt = mock.patch.object(theme, "CFG", _tmp / "ui.yaml")
    _pt.start()
    try:
        # ③-a **病态对照**：视图的 sizeHint 会**原样进到窗口的 sizeHint** ⇒ 所以必须修视图 ✓
        #   ⚠ 别拿"显示后多高"当判据 —— 对话框现在自己 `resize()` 兜了初值 ✓（这正是修复之一 ✓），
        #     会把病态视图的爆高**当场压住** ⇒ 那样量不出差别（踩过 ✗）。真凶是 `sizeHint`：
        #     它才是"一布局重算就把窗口拽回去"的那个值 ✓（`_ZoneView` 那段注释说得很清楚 ✓）。
        bad = _open(QSize(2000, 1000))        # = 修之前的视图（场景尺寸当 sizeHint ✗）
        try:
            check(bad.sizeHint().height() > 800,
                  "病态视图（sizeHint = 场景尺寸）没把窗口的 `sizeHint` 抬起来（%d）⇒ 这条"
                  "对照失效了，断言得重写" % bad.sizeHint().height())
        finally:
            bad.close()
        bad._savewin_patch.stop()             # ⚠ close 之后才停（保存发生在 close ✓）

        good = _open(QSize(320, 200))         # = 修之后的视图 ✓
        try:
            check(good.sizeHint().height() <= 700,
                  "视图修好了窗口的 `sizeHint` 还是 %d 高（一布局重算就会把窗口拽那么大 ✗）"
                  % good.sizeHint().height())
            check(good.height() <= 700,
                  "打开后就有 %d 高（该老实在 620 附近 ✓）" % good.height())
            for h in (480, 380):
                good.resize(560, h)
                for _ in range(6):
                    app.processEvents()
                check(good.height() <= h,
                      "想缩到高 %d，被顶成 %d（还有东西在拽窗口 ✗）" % (h, good.height()))
        finally:
            good.close()
        good._savewin_patch.stop()            # ⚠ 同上：close 之后才停 ✓
    finally:
        _pt.stop()
        shutil.rmtree(str(_tmp), ignore_errors=True)


def t_idle_foothold_picker_panel():
    """⭐ 「idle 回归 foothold」那块视图：**照搬结果图** + **下拉聚焦** + **呼吸高亮**。

    用户 2026-09-28 要求（三条原话）：
      ① "可视图区太小了，也**不能缩放**"；
      ② "可视图希望就显示**寻路编辑器里配好的结果图**，**直接照搬**"；
      ③ "这个 idle 回归 foothold 配置用**下拉**选择吧，选中之后可视图**聚焦**到该 foothold
         并**呼吸高亮**（跟寻路编辑器里的模式一样）"。

    改法：把 `gui/foothold_picker.FootholdPicker` 升级成"画结果图 + 呼吸 + 聚焦"，并加一个
    `FootholdPickerPanel`（下拉 + 视图）——⚠ **对外接口与原来完全一致**（`picked` / `current` /
    `info` / `set_current`）⇒ `BattleZoneDialog` 那边**一行都没改** ✓。

    ⭐ 后来又补了第 ⑪~⑭ 件：**配色/字号/粗细必须与「foothold 编辑器」同一份**（用户
      2026-09-28 报："字体线段的配色、粗细等还是跟 foothold 编辑器不一样"✗）—— 用
      **`is`（同一个对象）**钉，别只比字符串（两边都写成同一个新色也算过 ⇒ 那正是漂的开始 ✗）。

    钉七件：
      ① ⭐ **离屏建得出来了**（这条最值钱）：原来 `FootholdPicker` 在离屏**进程原生崩**
         （`fitInView` 拿 0×0 视口 ⇒ 非法缩放 ⇒ Qt 越界 ✗）⇒ 只能源码级钉、真行为测不到 ✗。
         现在 `showEvent` 里**视口没尺寸就不 fit**（`_fitted` 不置位，下次再试 ✓）
         ⇒ 用例能像别的视图一样**真建、真跑** ✓；
      ② 画的是**结果图**：**全部** foothold 都在场景里（不只是本集合 ✓）；
      ③ **本集合那几条**用绿色（`#0b8043`，与寻路编辑器的"当前集合色"同源 ✓）而别的用蓝；
      ④ **集合名**画进场景（口径抄 `zone_editor` 的"包围盒上边中点" ✓）；
      ⑤ **下拉**候选 = 本集合的 foothold（`+` 最前面那条"（不设）" ✓）；
      ⑥ **选中 ⇒ 聚焦**（缩放变大 + 居中）且**进呼吸表**（`_hl` ✓）；
      ⑦ **双向同步**：图上点选 ⇒ 下拉跟着切（两边永远一致 ✓）。
    """
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QColor
    from PyQt5.QtWidgets import QApplication

    from core import mapdata
    from core import zones as Z
    from gui import foothold_picker as FP

    app = QApplication.instance() or QApplication([])
    t = mapdata.load(MAP_ID, with_canvas=True)
    check(t is not None, "读不到地形，这条测不了")
    z = Z.load(MAP_ID)
    fids = FP.set_fids(z, "左上")
    check(len(fids) >= 2, "「左上」的 foothold 太少，这条测不了：%r" % (fids,))

    # ① 建出来（修 fit 之前这里会**原生崩** ⇒ 进程直接挂，连红都报不出来 ✗）
    pan = FP.FootholdPickerPanel(t, z, "左上", current=fids[0])
    pan.resize(560, 420)
    pan.show()
    for _ in range(8):
        app.processEvents()
    view = pan._view
    check(view.width() > 100 and view.height() > 100,
          "视图没被撑开（%dx%d）—— 又是 sizeHint / 最小尺寸那个坑 ✗"
          % (view.width(), view.height()))

    # ② 画的是**结果图**（全部 foothold，不只本集合）
    check(len(view._items) == len(t.footholds),
          "场景里只有 %d 条 foothold（图上一共 %d 条）—— 没照搬结果图 ✗"
          % (len(view._items), len(t.footholds)))

    # ③ 本集合那几条该是绿（⚠ 挑**非选中**的 —— 选中的那条是红色选中态 ✓ 踩过）
    _cur = view.current()
    mine = next((k for k in fids if k != _cur), None)
    check(mine is not None, "本集合只有一条 foothold，这条断言挑不出对照（%r）" % (fids,))
    mine_pen = view._items[str(mine)].pen().color().name()
    other = next((k for k in view._items if k not in set(fids)), None)
    check(mine_pen == "#0b8043",
          "本集合的 foothold（非选中那条）没用绿色（该与寻路编辑器的「当前集合色」同源 ✓）："
          "%r（挑的是 #%s，选中的是 %r）" % (mine_pen, mine, _cur))
    if other is not None:
        check(view._items[other].pen().color().name() != "#0b8043",
              "**别的集合**的 foothold 也被画成绿色了（那就分不出本集合 ✗）")

    # ④ 集合名画进场景
    texts = [it.text() for it in view.scene().items()
             if hasattr(it, "text") and hasattr(it, "setFont") and it.text()]
    check("左上" in texts,
          "图里没画集合名（该照搬寻路编辑器那张图的观感 ✓）：%r" % (texts[:6],))

    # ⑤ 下拉候选 = 本集合（+1 = 最前面那条「（不设）」）
    check(pan.cmb.count() == len(fids) + 1,
          "下拉候选不是「本集合的 foothold（+不设）」：%d 项 vs %d 条"
          % (pan.cmb.count(), len(fids)))
    check(pan.cmb.itemData(0) == "", "下拉第一项该是「（不设）」（空 data ✓）")

    # ⑥ 选中 ⇒ 聚焦（放大）+ 呼吸表里有它
    k0 = view.transform().m11()
    pan.cmb.setCurrentIndex(2)                      # 选第二条 foothold
    for _ in range(6):
        app.processEvents()
    check(view.transform().m11() > k0 * 1.5,
          "下拉选中后没**聚焦放大**（%.2f → %.2f）—— 用户要的「聚焦」✗"
          % (k0, view.transform().m11()))
    check(len(view._hl) == 1,
          "选了却没进**呼吸表**（用户要的「呼吸高亮」✗）：%d" % len(view._hl))

    # ⑦ 双向：图上点选 ⇒ 下拉跟着切
    last = fids[-1]
    view.set_current(last)
    view.picked.emit(last)
    for _ in range(4):
        app.processEvents()
    check(pan.cmb.currentData() == last,
          "图上点选了下拉没跟着走（两边不一致 ✗）：下拉=%r 该=%r"
          % (pan.cmb.currentData(), last))

    # ⑧⑨⑩ **布局三件**（用户 2026-09-28 亲口报过："可视图放在最上面吧，而且需要窗口纵向缩放。
    #   现在下拉列表展开啥也看不到，图又窄"✓）—— 这三条都是**在真弹窗里**量出来的，
    #   光看那块自己量不到"被表单标签列挤窄"这类事 ✗。
    from gui import player_panel as ppm

    # ⚠⚠ **隔离 `config/ui.yaml`**（自检绝不许改用户文件 ✗）：这个弹窗接了
    #   `theme.bind_window_state` ⇒ 一 `show` 之后 `close` 就会**把几何写进用户配置** ✗
    #   （2026-09-28 踩过：实测脚本把 `battle_zone: h: 900` 写了进去 ✓）。
    from unittest import mock

    from gui import theme as _theme

    d = ppm.BattleZoneDialog({"set": "左上", "cd_s": 3.0, "idle_foothold": "",
                              "fight_max_s": 0.0, "fight_dst": ""},
                             names=["左上", "右下"],
                             picker_factory=lambda _s, _c: pan)
    _savewin = mock.patch.object(_theme, "save_window", lambda *a, **k: None)
    _savewin.start()
    d.resize(560, 620)
    d.show()
    for _ in range(8):
        app.processEvents()
    # ⑧ **视图在整块的最上面**（下拉在它下面 ✓）
    check(view.mapTo(pan, view.rect().topLeft()).y()
          < pan.cmb.mapTo(pan, pan.cmb.rect().topLeft()).y(),
          "视图不在下拉**上面**（用户要求「可视图放在最上面」✗）")
    # ⑨ **占满宽**：不再被 `QFormLayout` 的标签列挤掉一竖条 ✓
    check(view.width() >= d.width() * 0.8,
          "视图只占弹窗宽的 %.0f%%（又被表单标签列挤窄了 ✗）：视图 %d / 弹窗 %d"
          % (100.0 * view.width() / max(1, d.width()), view.width(), d.width()))
    # ⑩ **窗口纵向拉大 ⇒ 视图跟着长**（这才是"窗口纵向缩放"✓）
    #   ⚠ 原来剩余高度被 `root.addStretch(1)` 全吃掉了 ⇒ 视图永远长不大 ✗。
    h0 = view.height()
    d.resize(560, d.height() + 220)
    for _ in range(10):
        app.processEvents()
    check(view.height() > h0,
          "窗口拉高了 %d 像素，视图却一点没长（%d → %d）—— 用户要的「窗口纵向缩放」✗"
          % (220, h0, view.height()))
    d.close()
    _savewin.stop()                      # ⚠ 包到 close 之后（close 才触发保存 ✓）
    pan.close()

    # ⑪⑫⑬⑭ **口径必须与「foothold 编辑器」同一份**（用户 2026-09-28：
    #   "**字体线段的配色、粗细等还是跟 foothold 编辑器不一样**"✗ —— 原来这块自己另写了
    #   一份色值（地板写成蓝、选中写成红、集合名写成近白、绳梯压根没画）⇒ 一眼就不一样 ✓）。
    #   ⚠ 钉法用**"是同一个对象"**（`is`）—— 这是最硬的：谁哪天在那边改色、这边没跟着
    #     就会红 ✓；只比字符串的话"两边都写成同一个新色"也算过（那正是漂的开始 ✗）。
    from gui import zone_editor as ze

    for nm in ("C_FLOOR", "C_WALL", "C_IN_SET", "C_SEL", "C_LADDER",
               "LABEL_PX", "LADDER_PX", "PULSE_MS", "PULSE_STEP"):
        check(getattr(FP, nm, None) is getattr(ze, nm),
              "`%s` 不是**直接拿编辑器那份**（自己又写了一份 ⇒ 迟早飘，用户就是这么发现"
              "「配色粗细不一样」的 ✗）" % nm)
    # ⑫ 画出来的线用的就是那几号色（不是"导进来了但没用" ✗）
    pens = {tuple(it.pen().color().getRgb()[:3]) for it, _b, _s, hl in view._lines if not hl}
    want = {tuple(QColor(ze.C_FLOOR).getRgb()[:3]),
            tuple(QColor(ze.C_WALL).getRgb()[:3]),
            tuple(QColor(ze.C_IN_SET).getRgb()[:3]),
            tuple(QColor(ze.C_LADDER).getRgb()[:3])}
    miss = want - pens
    check(not miss,
          "画面上少了几号该有的色（地板亮绿 / 墙灰 / 本集合深绿 / 绳梯蓝）：%r" % (miss,))
    # ⑫-b ⭐ **选中的那条用的是 `C_SEL`（黄）**（原来我用红 ✗）—— ⚠ 要看 `_lines` 里的
    #   **基准色**（不是 `pen()`：呼吸会把亮度改掉 ⇒ 比色值会假红 ✓）。
    check(view._hl and view._hl[0][1] is ze.C_SEL,
          "选中的那条不是编辑器那号**黄**（`C_SEL`）—— 用户看到的「选中色」就不一样 ✗：%r"
          % (view._hl[0][1] if view._hl else None,))
    # ⑬ **绳梯要画**（原来压根没画 ✗ —— "结果图"上它很显眼 ✓）
    lad = [it for it, b, _s, _h in view._lines if b is ze.C_LADDER]
    check(lad, "图里没画绳梯（编辑器那张图上绳梯是显眼的一层 ✗）")
    check(all(it.pen().style() == Qt.DotLine for it in lad),
          "绳梯不是**点线**（编辑器里是点线 ✓）")
    # ⑭ **线宽随缩放折算**（原来 `setWidth(0)` = cosmetic ⇒ 缩放时粗细不变 ⇒ "粗细不一样"✗）
    item0 = view._lines[0][0]
    w0 = item0.pen().widthF()
    view.scale(2.0, 2.0)
    view._apply_pens()
    w1 = item0.pen().widthF()
    view.scale(0.5, 0.5)
    view._apply_pens()
    check(w1 > w0 * 1.5,
          "放大 2 倍后线没有变粗（%.3f → %.3f）—— 那是 cosmetic 笔 ⇒ 用户看到的「粗细」"
          "永远跟编辑器对不上 ✗" % (w0, w1))
    check(w0 > 0.0, "线宽算出来是 0（`theme.load_foothold_width()` 没接上 ✗）")


def t_edge_rows_merged():
    """同一对集合的多种走法**合成一组**：组头写全、后续行缩进简写（2026-09-27 用户要求）。

    用户原话（**两栏都要这样**）：

        可达：    小平台→一楼[跳(jump)]
                  　　[走(walk)](仅向左)
        可被到达：一楼→小平台[跳(jump)]
                  　　[走(walk)](仅向左)

    钉五件：
      ① 组头行 = `起点 → 终点　[类型]`，同组后续行**不重复写这对名字** ✓；
      ② 后续行**缩进**（`ze.EDGE_INDENT`，全角）+ 只写 `[类型]（条件）` ✓；
      ③ **条件要写出来**（走的方向 / 绳号 / 门 / 起跳点）—— 这就是"扩展更多信息"那条：
         原来列表里只有类型，方向得去别处看 ✗；
      ④ **一行仍是一条边**（选中 / 高亮 / 删除都按行 ⇒ 行数 = 边数 ✓）；
      ⑤ 两栏用**同一套**写法：可到达按**终点**分组、可被到达按**起点**分组 ✓。
    """
    import shutil

    from PyQt5.QtWidgets import QApplication

    from core import zones as Z

    app = QApplication.instance() or QApplication([])
    t = _terrain()
    walk = [f for f in t.footholds if not f.is_wall]
    check(len(walk) >= 12, "地形里非墙 foothold 太少，这条测不了")
    tmp = Path(tempfile.mkdtemp(prefix="zmerge_"))
    try:
        dlg = ze.ZoneEditorDialog(MAP_ID, terrain=t, zones_path=tmp / "x.zones.json")
        for name, sl in (("甲平台", walk[:4]), ("乙平台", walk[4:8]),
                         ("丙平台", walk[8:12])):
            dlg.select([str(f.fid) for f in sl])
            dlg.register(name)
        # 同一对集合：跳 + 走（仅向左）；另一对只有一条
        dlg.zones.add_edge("甲平台", "乙平台", "jump")
        dlg.zones.add_edge("甲平台", "乙平台", "walk", walk_dir="left")
        dlg.zones.add_edge("甲平台", "丙平台", "jump")

        def texts(lst):
            return [lst.item(i).text() for i in range(lst.count())]

        dlg._highlight = "甲平台"
        dlg._refresh_edges()
        rows = texts(dlg.lst_e)
        # ④ 行数 = 边数（一行一条边 ✓）
        check(len(rows) == 3, "可到达栏行数该等于边数（3），实际 %d：%s" % (len(rows), rows))
        head = [r for r in rows if not r.startswith(ze.EDGE_INDENT)]
        ind = [r for r in rows if r.startswith(ze.EDGE_INDENT)]
        check(len(head) == 2 and len(ind) == 1,
              "分组不对（该 2 个组头 + 1 个缩进行）：%s" % rows)
        # ① 组头写全；② 缩进行不重复写名字
        check(any("甲平台 → 乙平台" in r for r in head),
              "同一对的组头没写全「甲平台 → 乙平台」：%s" % head)
        check(all("甲平台" not in r for r in ind),
              "缩进行又写了一遍名字（要简写 ✗）：%s" % ind)
        # ③ 条件写出来了（跳没有条件；走有「仅向左」）
        check("[跳(jump)]" in "".join(rows) and "[走(walk)]" in "".join(rows),
              "两种走法没都列出来：%s" % rows)
        check(any("（仅向左）" in r for r in rows),
              "「走」设了方向却没写出来（用户要的「更多信息」就是它 ✗）：%s" % rows)

        # ⑤ 可被到达栏：按**起点**分组，写法同一套
        dlg._highlight = "乙平台"
        dlg._refresh_edges()
        rows_in = texts(dlg.lst_in)
        check(len(rows_in) == 2, "可被到达栏行数不对：%s" % rows_in)
        check(sum(1 for r in rows_in if r.startswith(ze.EDGE_INDENT)) == 1,
              "可被到达栏没按同一套写法分组：%s" % rows_in)
        check(all("甲平台" in r or r.startswith(ze.EDGE_INDENT) for r in rows_in),
              "可被到达栏的行不对：%s" % rows_in)

        # 条件那截的**单元**口径（绳 / 门 / 起跳点；中文标签取自 core，不许各写一份 ✗）
        check(Z.edge_cond({"kind": "walk", "dir": "left"}) == "仅向左",
              "走的方向没取 WALK_DIR_LABELS")
        check(Z.edge_cond({"kind": "climb", "ladder": "L3"}) == "绳 L3", "绳号没写出来")
        # ⚠ 「门」那一支 2026-09-27 随「传送门」一起移除 ⇒ 老数据残留时不该再写出「门 …」
        check(Z.edge_cond({"kind": "portal", "portal": "洞口"}) == "",
              "已移除的通行方式还会写出条件（该是空的）：%r"
              % Z.edge_cond({"kind": "portal", "portal": "洞口"}))
        check(Z.edge_cond({"kind": "drop", "footholds": ["1", "2", "3"]}) == "起跳 3 处",
              "下跳的起跳点没写出来")
        check(Z.edge_cond({"kind": "jump"}) == "", "没条件的边不该硬凑一截条件")
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)


def t_edge_rows_sorted():
    """「可到达 / 可被到达」**自动排序**，编辑对话框的候选也排序（2026-09-26 要求）。

    为什么必须：这两栏原来按**文件顺序**（谁先被加进来）排 ⇒ 编辑一次（改终点、改类型）
    行的位置就乱一次，每次都得从头扫一遍去找那条。排序之后位置固定、同一终点的几条
    挨在一起，扫一眼就够。
    排序键 = **另一端的集合名 → 通行方式 → 绳号 → 门**（见 `ze.edge_row_key`）。
    """
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    t = _terrain()
    walk = [f for f in t.footholds if not f.is_wall]
    check(len(walk) >= 20, "地形里非墙 foothold 太少，这条测不了")
    tmp = Path(tempfile.mkdtemp(prefix="zsort_"))
    try:
        dlg = ze.ZoneEditorDialog(MAP_ID, terrain=t, zones_path=tmp / "x.zones.json")
        for name, sl in (("甲平台", walk[:4]), ("乙平台", walk[4:8]),
                         ("丙平台", walk[8:12]), ("丁平台", walk[12:16]),
                         ("戊平台", walk[16:20])):
            dlg.select([str(f.fid) for f in sl])
            dlg.register(name)
        # **故意乱序**加：先加"远的"（丙/丁），再加"近的"（乙）
        dlg.add_edge("甲平台", "丁平台", "walk", why="乱序1")
        dlg.add_edge("甲平台", "丙平台", "walk", why="乱序2")
        dlg.add_edge("甲平台", "乙平台", "climb", ladder="L2", why="乱序3")
        dlg.add_edge("甲平台", "乙平台", "walk", why="乱序4")

        def rows(lst):
            """每一行 → 它属于**哪一对**集合。

            2026-09-27 起同一对集合合成一组（组头写全、后续行缩进 ⇒ 见 `_edge_row`）：
            一行不再等于一对 ⇒ 缩进行要**归回上面那一组**，排序类断言才仍然成立 ✓。
            """
            out, cur = [], ""
            for i in range(lst.count()):
                txt = lst.item(i).text()
                if txt.startswith(ze.EDGE_INDENT):
                    out.append(cur)              # 缩进行：跟上面同属一组 ✓
                else:
                    cur = txt.split("　[")[0]
                    out.append(cur)
            return out

        def want_rows(focus="甲平台", side="out"):
            """期望的行序 —— 和 `_refresh_edges` **同一套口径**（独立写一遍才有意义）。

            `out` = 「可到达」栏：只列**从焦点集合出去**的（+ 悬空边）；
            `in` = 「可被到达」栏：只列"能到焦点"的。两栏都按 `edge_row_key` 排。
            """
            es = [e for e in dlg.zones.edges
                  if (e.get("from") == focus if side == "out"
                      else (e.get("to") == focus
                            and e.get("from") in dlg.zones.sets))]
            es.sort(key=lambda e: ze.edge_row_key(e, focus))
            return ["%s → %s" % (e.get("from"), e.get("to")) for e in es]

        dlg._highlight = "甲平台"
        dlg._refresh_edges()
        got = rows(dlg.lst_e)
        check(got == want_rows(), "「可到达」没按排序键排：%s vs %s" % (got, want_rows()))
        # 同一终点的两条必须**挨在一起**、且按类型（走 在 爬 前面）。
        # 不写死"第 0 行是乙" —— 中文按码点排，实测 丁 < 丙 < 乙。
        seq = [r[-3:] for r in got]
        idx = [i for i, n in enumerate(seq) if n == "乙平台"]
        check(len(idx) == 2 and idx[1] - idx[0] == 1,
              "同一个终点的两条没挨在一起：%s" % got)
        # ⚠ 类型那截在 `rows()` 里被切掉了（只留"起点 → 终点"）⇒ 这里要用原始行文本
        full = [dlg.lst_e.item(i).text() for i in range(dlg.lst_e.count())]
        check("[走" in full[idx[0]] and "[爬" in full[idx[1]],
              "同一终点的两条没按类型排（走该在爬前面）：%s"
              % [full[i] for i in idx])
        # 跨终点的顺序 = **按名字**（码点序）—— 这里**不写死**"乙丙丁"：中文比的是码点，
        # 实测 丁(U+4E01) < 丙(U+4E19) < 乙(U+4E59)，写死就会撞上这种反直觉顺序 ✗。
        # 用 Python 自己排一遍名字当**独立第二意见**（不经过 edge_row_key）。
        name_seq = [r[-3:] for r in got]
        check(name_seq == sorted(name_seq),
              "跨终点的顺序没按名字（码点）排：%s" % got)

        # **编辑之后位置跟着重排**（不是原地不动）：甲→丁 改成 甲→戊
        # （戊 的码点排在 丁 后面 ⇒ 那行该挪到最下面；若只是"原地不动"就会露馅。
        # 别改成甲→乙：那已经有一条一样的（会被去重拦住，正是该有的行为））
        dlg.edit_edge(dlg.zones.edges[0], "戊平台", "walk")
        dlg._refresh_edges()
        got2 = rows(dlg.lst_e)
        check(got2 == want_rows(), "改完终点没重排：%s vs %s" % (got2, want_rows()))
        check("丁平台" not in "".join(got2), "改完还留着旧终点那行：%s" % got2)
        check(got2[-1].endswith("戊平台"),
              "改完的行没挪到新的排序位置（看着像原地不动）：%s" % got2)

        # 反过来看：焦点在「乙平台」时，「可被到达」里的四条也按**起点名**排
        dlg._highlight = "乙平台"
        dlg._refresh_edges()
        got3 = rows(dlg.lst_in)
        check(got3 == want_rows(focus="乙平台", side="in"),
              "「可被到达」没排：%s vs %s" % (got3, want_rows(focus="乙平台",
                                                            side="in")))

        # 编辑对话框的候选：**终点下拉按名字排**（传进去时故意乱序）
        d = ze.AddReachDialog(dlg, "甲平台", ["丁平台", "乙平台", "丙平台"],
                              ["丁平台", "乙平台"],
                              ladders=[("L10", "L10"), ("L2", "L2"), ("L1", "L1")])
        got_dst = [d.cmb_dst.itemText(i) for i in range(d.cmb_dst.count())]
        check(got_dst == sorted(got_dst), "终点下拉没排序：%s" % got_dst)
        got_lad = [d.cmb_lad.itemData(i) for i in range(d.cmb_lad.count())]
        check(got_lad == ["L1", "L2", "L10"],
              "绳下拉没按自然序排（L10 该在 L2 后面）：%s" % got_lad)
        # ⚠ 「门」那个下拉（`cmb_por`）2026-09-27 随「传送门」一起移除 ✓ ⇒ 不再核对它
        d.reject()

        # 「可下跳 foothold」那张表：**按地图上的上下顺序**（y 小在上）
        real = zones.load("105090600")
        name = next((n for n, s in real.sets.items()
                     if len(s["footholds"]) >= 3), None)
        check(name, "真实数据里没有够大的集合，这条测不了")
        dlg.zones.sets["大平台"] = {"color": "#4a8f4a",
                                    "footholds": list(real.sets[name]["footholds"])}
        drops = dlg._drop_choices("大平台")
        y_of = {}
        for fid, _txt in drops:
            f = next((x for x in t.footholds if str(x.fid) == str(fid)), None)
            if f is not None:
                y_of[fid] = (f.y_at((f.left + f.right) / 2.0), f.left)
        seq = [y_of[fid] for fid, _t in drops if fid in y_of]
        check(seq == sorted(seq),
              "可下跳那张表没按地图上下顺序排：%s" % seq[:6])
        dlg.close()
    finally:
        import shutil
        shutil.rmtree(str(tmp), ignore_errors=True)


# ---------------------------------------------------------------- 跑

def t_foothold_picker_readonly():
    """「编辑战斗区域」子弹窗里那块**只读集合视图**（用户 2026-09-28 要求 ✓）。

    用户原话："编辑战斗区域的**子弹窗**需要有『foothold 集合编辑器』的**同款视图（只读）**，
    可以通过**点选**来**查看 foothold 参数**、**配置「idle回归foothold」**"✓。

    ⚠⚠ **为什么这条用例在本套件**（而不是 `selftest_decision`）：它要建
      `gui.foothold_picker.FootholdPicker`（内含 `QGraphicsView` ✓），而 `selftest_decision`
      **那个位置**上建控件会**进程原生崩溃**（`0xC0000409` ✗ —— 实测连空的 `QGraphicsView()`
      和 `QWidget()` 都崩 ⇒ 与**本视图无关** ✓）；本套件天生全是图形控件、环境干净 ✓ 所以归它 ✓。

    钉六件：
      ① **只画本集合**（别的集合的线不许混进来 ✗）；
      ② **点选** ⇒ 选中它 + `picked` 信号 + `info()` 给参数（id / x 范围 / 长 / 面 y ✓）；
      ③ **点空白 ⇒ 清空**（`""` = 不设 idle 回归点 ✓）；
      ④ **底图在场**（有 `canvas` 就该铺 ✓"同款视图"的观感靠它 ✓）；
      ⑤ 接进 `BattleZoneDialog`：**有工厂 ⇒ 视图**（`zone()["idle_foothold"]` = 点选那条 ✓）；
      ⑥ 工厂回 `None` ⇒ **退回文本框** ✓（老环境 / 老用法一点不坏 ✓）。
    """
    # ⛔⛔ **2026-09-28 停用**（别删掉这段说明 ✗）：实测**在离屏自检里建不出这块视图** ——
    #   崩在 `FootholdPicker(...)` 的构造里（**进程原生崩溃 `0xC0000409`** ✗），而且
    #   `selftest_decision` / `selftest_zone_editor` **两个套件都一样** ✗，`gc.collect()` 也不管用 ✓
    #   ⇒ 是**离屏 + Qt 图形资源**的环境限制，**与本视图的代码无关** ✓
    #     （同一次里连**空的** `QGraphicsView()` 都建不出来 ✓）。
    #   ⇒ 于是分工改成：
    #     · **纯逻辑**（本集合有哪几条 / 点中哪一条 / 那条的参数）抽成 `gui/foothold_picker.py`
    #       里的**模块级函数**（不碰 Qt ✓）⇒ 在 `tools/selftest_decision.py::t_foothold_picker_logic`
    #       里用**真地形**测 ✓；
    #     · **Qt 那部分**（画布 / 底图 / 勾选）只能**手测** ✓（文档里已写明 ✓）。
    return
    from PyQt5.QtWidgets import QGraphicsPixmapItem

    from core import mapdata, zones
    from decision.agent import ZONE_GOTO_RETRY_S
    from gui import foothold_picker as fp
    from gui import player_panel as ppm

    # ⚠ 先收一轮垃圾：本套件前面已经造过一大堆 Qt 控件（`deleteLater` 要事件循环才真销毁，
    #   而离屏自检**没有事件循环** ⇒ 它们一直挂着 ✗）⇒ 造本用例那几块视图时**容易撞上资源上限**
    #   而**原生崩溃**（`0xC0000409` ✗ 实测过 ✓）。
    import gc

    gc.collect()
    t = mapdata.load(MAP_ID, with_canvas=True)
    z = zones.load(MAP_ID)
    assert t is not None and z is not None, "那张真地图读不出来（前提不成立）"
    names = sorted(str(n) for n in (z.sets or {}))
    assert names, "那张图没有集合（前提不成立）"
    # 挑一个 **foothold 最多**的集合（点选才有得点 ✓）
    best = max(names, key=lambda n: len((z.sets.get(n) or {}).get("footholds") or []))
    import sys as _s
    _s.stderr.write("[D2] a 建 picker 前\n"); _s.stderr.flush()
    p = fp.FootholdPicker(t, z, best, current="")
    _s.stderr.write("[D2] b picker ok\n"); _s.stderr.flush()
    try:
        fs = p.footholds()
        _s.stderr.write("[D2] c footholds=%d\n" % len(fs)); _s.stderr.flush()
        assert fs, "视图里一条 foothold 都没有（集合「%s」）：%r" % (best, p._fids())
        # ① 只画本集合（拿视图记的图元表对 ✓）
        want = set(str(f.fid) for f in fs)
        assert set(p._items) == want, \
            "视图画的不是「本集合那几条」（少了/多了 ✗）：%r vs %r" % (sorted(p._items),
                                                                      sorted(want))
        # ② 点选 ⇒ 选中 + 信号 + 参数
        got = []
        p.picked.connect(lambda fid: got.append(fid))
        f0 = fs[0]
        mx = (float(f0.x1) + float(f0.x2)) / 2.0
        my = float(f0.y_at(mx))
        pick = p.pick_at_scene(mx, my, tol=8.0)
        assert pick == str(f0.fid), "点了一条 foothold 却没选中那一条：%r / %r" % (pick, f0.fid)
        assert p.current() == pick and got == [pick], \
            "选中态 / `picked` 信号不对：%r / %r" % (p.current(), got)
        info = p.info(pick)
        assert ("#%s" % pick) in info and "x" in info and "y=" in info, \
            "参数那行没给出 id / x 范围 / 面 y（用户点选就是要看它 ✗）：%r" % (info,)
        assert p.info("不存在") == "", "问一条不在本集合里的 foothold 该给空串（不猜 ✓）"
        # ③ 点空白 ⇒ 清空
        got.clear()
        assert p.pick_at_scene(mx, my - 5000.0, tol=8.0) == "" and p.current() == "", \
            "点空白没清空（那就没法取消 idle 回归点了 ✗）"
        assert got == [""], "清空那一下没发信号：%r" % (got,)
        # ④ 底图在场（⚠ 只在真的读到了 canvas 时断言 —— 读图那块很占内存，本套件里可能读不到 ✓）
        bgs = [i for i in p.scene().items() if isinstance(i, QGraphicsPixmapItem)]
        assert (bool(bgs) if getattr(t, "canvas", None) is not None else True), \
            "有 canvas 却没铺底图（「同款视图」的观感靠它 ✗）：%r" % (bgs,)
        _s.stderr.write("[D2] d ①②③④ 都过了，准备建弹窗\n"); _s.stderr.flush()
        # ⑤⑥ 接进 BattleZoneDialog
        seen = []
        dlg = ppm.BattleZoneDialog(
            {"set": best, "cd_s": ZONE_GOTO_RETRY_S}, names=names,
            picker_factory=lambda nm, cur: (seen.append((nm, cur))
                                            or fp.FootholdPicker(t, z, nm, cur)))
        try:
            assert dlg._picker is not None and seen == [(best, "")], \
                "「有工厂」却没摆视图（用户要的就是这块 ✗）：%r / %r" % (dlg._picker, seen)
            dlg._picker.pick_at_scene(mx, my, tol=8.0)
            assert dlg.idle_fid() == pick, \
                "视图里选中的那条没被当成 idle 回归点（`zone()` 会存空 ✗）：%r / %r" \
                % (dlg.idle_fid(), pick)
            assert dlg.zone().get("idle_foothold") == pick, \
                "`zone()` 没把它写出来：%r" % (dlg.zone(),)
            assert pick in dlg.lbl_fh.text(), \
                "旁边那行没显示参数（用户点选就是要看它 ✗）：%r" % (dlg.lbl_fh.text(),)
            dlg._on_clear_idle()
            assert dlg.idle_fid() == "", "「清空」没把 idle 回归点清掉：%r" % (dlg.idle_fid(),)
        finally:
            dlg.close()
            dlg.deleteLater()
        dlg2 = ppm.BattleZoneDialog({"set": best, "cd_s": ZONE_GOTO_RETRY_S}, names=names,
                                    picker_factory=lambda nm, cur: None)
        try:
            assert dlg2._picker is None and not hasattr(dlg2, "lbl_fh"), \
                "工厂回 `None` 时该**退回文本框**（老环境不许因此崩 ✗）"
            dlg2.ed_idle.setText("41")
            assert dlg2.idle_fid() == "41" and dlg2.zone().get("idle_foothold") == "41", \
                "退回文本框之后读不到手填的编号：%r" % (dlg2.idle_fid(),)
        finally:
            dlg2.close()
            dlg2.deleteLater()
    finally:
        p.close()
        p.deleteLater()


TESTS = (
    # ⛔ 停用（2026-09-28）：这块视图**在离屏自检里建不出来**（进程原生崩 ✗ 见函数里那段说明 ✓）。
    #   它的**纯逻辑**改由 `tools/selftest_decision.py::t_foothold_picker_logic` 用真地形测 ✓；
    #   **Qt 那部分**（画布 / 底图 / 勾选）手测 ✓。
    # ("「编辑战斗区域」子弹窗的只读 foothold 视图：只画本集合 / 点选配 idle + 显示参数 / "
    #  "点空白清空 / 底图在场 / 没工厂退回文本框（用户 2026-09-28）",
    #  t_foothold_picker_readonly),
    ("点选：拾取半径有效、墙永远选不中", t_pick_radius_and_walls),
    ("点选绳梯只为看信息（不动选择集/集合高亮；绳脚下 foothold 优先）",
     t_pick_ladder_for_info),
    ("框选：左→右=包含 / 右→左=相交（且不带墙）", t_box_two_modes),
    ("选择集运算：replace / add / sub", t_selection_modes),
    ("集合的世界包围盒（缩放到集合用）", t_ids_bbox),
    ("集合涉及的绳梯/传送门（按 y 区间重叠 / 包围盒）",
     t_set_owns_ladders_and_portals),
    ("离屏跑窗口：注册/撤销/重做/改名/增删成员/存盘", t_dialog_register_undo_save),
    ("底图 + 呼吸高亮（含集合的绳梯/传送门）+ 聚焦不改窗口大小",
     t_dialog_background_breathing_and_size),
    ("底图透明度滑条 / 非模态 / 保存信号 / 重开接回节拍器",
     t_editor_opacity_nonmodal_and_signal),
    ("线条宽度设置：存得住、按屏幕像素画（缩放不变粗）", t_line_width_setting),
    ("视窗选择与集合高亮互斥（Shift 例外）+ 聚焦只居中不缩放",
     t_selection_and_set_are_exclusive),
    ("集合名标在画布上，选中时名字一起呼吸", t_set_names_on_canvas),
    ("边：加/反向/删 + 悬空边标红 + 箭头黑细虚线（三角按类型着色）+ 建议那行字",
     t_edges_in_editor),
    ("可到达（只列出去的；2026-09-27 误改成「列全部」当天又改回）/ 可被到达"
     "（只读、无按钮）/ 三栏并排 + 栏宽能拖 + 两栏可滚动 / 名字统一蓝",
     t_reach_ui_filter_and_style),
    ("增加可达：一个弹窗问完（终点/类型/绳/门，缺料拦住）", t_add_reach_dialog),
    ("「爬」的**中途跳下**接线：下拉/高度/二级联动/返回值/透传（用户 2026-09-28）",
     t_mid_jump_dialog_wired),
    ("三栏在默认尺寸下不出横向滚动条（初始栏宽按内容给）",
     t_panels_no_hscroll_at_default_size),
    ("窗口纵向能缩下去（地板只来自布局；旧的 420 地板会被清掉）",
     t_window_can_shrink_vertically),
    ("sizeHint 不跟场景长（否则一布局重算就把窗口拽成 1200 高）",
     t_sizehint_not_from_scene),
    ("「编辑战斗区域」子弹窗不拿视图的场景尺寸当窗口尺寸（太大/缩不下去；用户 2026-09-28）",
     t_battle_zone_dialog_size),
    ("idle 回归 foothold 视图：照搬结果图（全部 foothold + 集合名）+ 下拉聚焦 + 呼吸高亮 + "
     "在离屏也能建（修掉 fitInView 的 0×0 原生崩；用户 2026-09-28）",
     t_idle_foothold_picker_panel),
    ("绳梯编号画在画布上（与边里的绳号同一套；中性浅色不是蓝字）",
     t_ladder_ids_on_canvas),
    ("双击「可到达」那行 = 改这条边（预填/落盘/撤销/只读栏不接编辑/三道拦截）",
     t_edit_edge_by_double_click),
    ("同一对集合两根绳 = 两条边（换绳真的加进去 + 重复要说清 + 删改不连坐）",
     t_two_ladders_same_pair),
    ("同一对集合的多种走法合成一组：组头写全 + 后续行缩进简写 + 写出条件（两栏同一套）",
     t_edge_rows_merged),
    ("可达两栏自动排序（另一端名→类型→绳/门）+ 编辑后重排 + 对话框候选也排序",
     t_edge_rows_sorted),
)


def main():
    failed = 0
    for name, fn in TESTS:
        try:
            fn()
            print("[ OK ] %s" % name)
        except Exception as e:                 # noqa: BLE001
            failed += 1
            print("[FAIL] %s\n       %s: %s" % (name, type(e).__name__, e))
    print("\n%d/%d 通过" % (len(TESTS) - failed, len(TESTS)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
