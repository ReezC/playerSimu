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
from gui import zone_editor as ze                            # noqa: E402

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


def t_edges_in_editor():
    """编辑器里的「边」：加/反向/删 + 两栏各自跟上 + **悬空边标红** + 画布箭头画法。

    对应 §12.3 B5 + 2026-09-26 的要求 1、3。两个最容易错的点：
      · **悬空边**（集合被删/改名）：看着没事，只有跑起来才会在路径里断掉；
      · **两栏的方向**：「可到达」只列从焦点出去的，「可被到达」只列进来的 ——
        点「反向」之后那条边属于**对面**那个集合，不该在当前的「可到达」里冒出来。
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
        # 要去下面「可被到达」栏（2026-09-26 要求 1、2）
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

      ① **窗口能纵向缩**：右栏那一串控件进滚动区（矮下来是出滚动条，不是按钮消失）；
      ② 名字统一**蓝**（一眼认出"这是个注册过的集合"；悬空边那种**错误**仍标红）；
      ③ **可到达只列出去的边**：选中集合（或它的一段 foothold）时，这一栏回答的是
         "它能去哪儿"，并且**把收窄说出来**（静悄悄少几条是最难查的"东西不见了"）；
      ④ **可被到达只列进来的边、且只读**：那一栏**一个编辑按钮都没有** ——
         要改就选中上面那个集合，它的"可到达"里就有这条边（一次只编辑一样东西）。

    收窄的判据是 `_focus_set`：高亮的集合 > 选中的 foothold 所属的集合（同一批只属于
    一个集合时才算）> None（什么都没选 ⇒ 显示全部，那既是总览也是逃生口）。
    """
    from PyQt5.QtWidgets import QApplication, QPushButton, QScrollArea

    app = QApplication.instance() or QApplication([])
    t = _terrain()
    walk = [f for f in t.footholds if not f.is_wall]
    check(len(walk) >= 12, "地形里非墙 foothold 太少，这条测不了")
    tmp = Path(tempfile.mkdtemp(prefix="zreach_"))
    try:
        dlg = ze.ZoneEditorDialog(MAP_ID, terrain=t, zones_path=tmp / "x.zones.json")
        dlg.resize(980, 660)
        # ① 右栏在滚动区里 + 窗口最小高度压得下来
        check(isinstance(getattr(dlg, "right_area", None), QScrollArea),
              "右栏没进 QScrollArea（窗口一矮按钮就被压没）")
        check(dlg.right_area.widgetResizable(), "滚动区里的右栏没跟着窗口宽度走")
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

        # 什么都没选 ⇒ 显示全部，而且不再说收窄
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

    # 默认类型是「走」⇒ 绳/门那两行不占地方
    check(ad.cmb_lad.isHidden() and ad.cmb_por.isHidden(),
          "选「走」时不该显示绳/门那两行")
    ad.cmb_kind.setCurrentIndex(ad.cmb_kind.findData("climb"))
    check(not ad.cmb_lad.isHidden() and ad.cmb_por.isHidden(),
          "按类型显隐没生效（选「爬」该只出绳那一行）")
    ad.cmb_lad.setCurrentIndex(1)
    ad._accept()
    check(ad.result_dict == {"dst": "乙平台", "kind": "climb",
                             "ladder": "L3", "portal": None, "footholds": None,
                             "dir": None},
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
        lads2, pors2 = dlg._reach_ladder_choices("右下")
        check([v for v, _txt in lads2] == ["L2", "L3"] or
              sorted(v for v, _txt in lads2) == ["L2", "L3"],
              "起点能选的绳不对（该是通到这块平台的那两根）：%s" % lads2)
        check(pors2 == [], "这块平台上没有可用的门，却列出了：%s" % pors2)
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
        # 焦点留在「A平台」——「可到达」才列得全（它只列**出去**的边）
        dlg._highlight = "A平台"
        dlg._refresh_edges()
        check(dlg.lst_e.count() == 2, "A 的可到达该有 2 条：%d" % dlg.lst_e.count())

        # ① 编辑模式的弹窗：预填当前值、按钮写「保存」（不再写「增加」）
        names = list(dlg.zones.sets)
        d = ze.AddReachDialog(dlg, "A平台", names, names,
                              ladders=[("L1", "L1 x=0")], portals=[],
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
            return [lst.item(i).text().split("　[")[0] for i in range(lst.count())]

        def want_rows(focus="甲平台", side="out"):
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
                              ladders=[("L10", "L10"), ("L2", "L2"), ("L1", "L1")],
                              portals=[("west02", "west02"), ("west01", "west01")])
        got_dst = [d.cmb_dst.itemText(i) for i in range(d.cmb_dst.count())]
        check(got_dst == sorted(got_dst), "终点下拉没排序：%s" % got_dst)
        got_lad = [d.cmb_lad.itemData(i) for i in range(d.cmb_lad.count())]
        check(got_lad == ["L1", "L2", "L10"],
              "绳下拉没按自然序排（L10 该在 L2 后面）：%s" % got_lad)
        got_por = [d.cmb_por.itemData(i) for i in range(d.cmb_por.count())]
        check(got_por == sorted(got_por), "门下拉没排序：%s" % got_por)
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

TESTS = (
    ("点选：拾取半径有效、墙永远选不中", t_pick_radius_and_walls),
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
    ("可到达（只列出去的）/ 可被到达（只读、无按钮）/ 右栏可滚动 / 名字统一蓝",
     t_reach_ui_filter_and_style),
    ("增加可达：一个弹窗问完（终点/类型/绳/门，缺料拦住）", t_add_reach_dialog),
    ("窗口纵向能缩下去（地板只来自布局；旧的 420 地板会被清掉）",
     t_window_can_shrink_vertically),
    ("sizeHint 不跟场景长（否则一布局重算就把窗口拽成 1200 高）",
     t_sizehint_not_from_scene),
    ("绳梯编号画在画布上（与边里的绳号同一套；中性浅色不是蓝字）",
     t_ladder_ids_on_canvas),
    ("双击「可到达」那行 = 改这条边（预填/落盘/撤销/只读栏不接编辑/三道拦截）",
     t_edit_edge_by_double_click),
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
