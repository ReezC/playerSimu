"""「掉落物（道具）自动标注」自检（用户 2026-10-04 ✓）。

需求原文（三句）：
  1. 模型训练 → 自动标注卡片**最下面**加勾选参数「标注掉落物」；勾上后下面出现**列表区**
     （增删项目：**按 id / 名称搜索 WZ 里的图**），再下面是按钮「模板匹配标注」「YOLO标注」；
  2. 列表里有项目时，那两个按钮才可交互；
  3. 整理自动标注卡片的参数/按钮分类布局。

本轮是**界面先行**（用户明说"先做界面功能" ✓），所以这里钉的是：
  · 界面四件事都对（勾选 → 整块显隐 / 列表增删 / 按钮**启用条件** / 存进项目 ✓）；
  · 按钮**不是假的**：点下去真的走到 `label_drops.run_label_drops` ✓，它再转发给
    **既有的**两个算法（`detect_mobs.run_detect(cls=2)` / `yolo_augment(mode="drop")` ✓）；
  · 「图库还不存在」这件事**必须如实说**（缺什么、要等 WzProbe 出 `dump-item` ✓）——
    ⛔ 最坏的做法是空跑当成功 ✗。

跑法（离屏）：
    python -m tools.selftest_label_drops
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _tmpdir():
    import tempfile
    return Path(tempfile.mkdtemp(prefix="drops_"))


class _Lib:
    """造一个**假的掉落物图库**（清单 + 图标目录 ✓）—— 用完把全局那几个函数还原 ✓。"""

    def __init__(self, root, drops=(("04030001", "金币"), ("04030002", "药水")),
                 with_icons=True, with_index=True, meta=None):
        import json
        from core import wzexport
        self.wz = wzexport
        self.root = Path(root)
        self.spr = self.root / "drop"
        self.idx = self.root / "drops.json"
        if with_index:
            rows = []
            for i, n in drops:
                row = {"id": i, "name": n}
                row.update((meta or {}).get(i) or {})   # 类别 / 帧数这些 ✓（弹窗要用 ✓）
                rows.append(row)
            self.idx.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        if with_icons:
            for i, _n in drops:
                d = self.spr / i
                d.mkdir(parents=True, exist_ok=True)
                # 一张 4×4 的空 PNG 就够（判据只看"有没有那张图" ✓ 不真读像素 ✓）
                (d / "icon_0.png").write_bytes(bytes.fromhex(
                    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
                    "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"))
        self._old = (wzexport.drop_sprite_dir, wzexport.drop_index_path)
        wzexport.drop_sprite_dir = lambda cfg=None: self.spr
        wzexport.drop_index_path = lambda cfg=None: self.idx

    def restore(self):
        self.wz.drop_sprite_dir, self.wz.drop_index_path = self._old


def t_drop_library():
    """① 掉落物图库那套取数函数（`core/wzexport.py`）—— 纯逻辑，**一处口径** ✓。

    钉五件：
      · `norm_drops` 容三种历史写法（只存 id / dict / `[id, name]` ✓ 少一种就会悄悄丢项 ✗）；
      · 清单三种结构都能读（`[{id,name}]` / `{"drops":[…]}` / `[[id,name]]` ✓）；
      · `list_sprite_drops` 按 **id 或名称** 筛 ✓（两种都要能搜到 —— 用户要的就是"按 id、名称"✓）；
      · **清单 ∪ 图标目录** 是并集 ✓（有名字没图 / 有图没名字都不许丢 ✗）；
      · `drop_library_state()` **如实**区分"没有"和"空" ✓（现在必然是没有 ⇒ 说明里要点出
        WzProbe 缺 `dump-item` ✓，不然人只会以为名字打错了 ✗）。
    """
    import shutil

    from core import wzexport

    # ① norm_drops：三种写法
    got = wzexport.norm_drops(["04030001", {"id": "04030002", "name": "药水"},
                               ["04030003", "炸弹"]])
    check([d["id"] for d in got] == ["04030001", "04030002", "04030003"],
          "`norm_drops` 丢了项（老配置只存 id 的那批会消失 ✗）：%r" % (got,))
    check(got[2]["name"] == "炸弹", "`[id, name]` 这种写法没读出名字：%r" % (got[2],))

    tmp = _tmpdir()
    try:
        lib = _Lib(tmp, drops=(("04030001", "金币"), ("04030002", "恢复药水")))
        try:
            # ② 清单三种结构
            import json
            p = tmp / "b.json"
            p.write_text(json.dumps({"drops": [["04030009", "戒指"]]},
                                    ensure_ascii=False), encoding="utf-8")
            got2 = wzexport.load_drop_index(p)
            check(got2 == [{"id": "04030009", "name": "戒指"}],
                  "`{\"drops\": [[id, name]]}` 这种结构和 `[id, name]` 没读对：%r" % (got2,))
            check(wzexport.load_drop_index(tmp / "没有这个.json") == [],
                  "清单不存在该返回空表（不抛 ✗）")

            # ③ 按 id / 按名称都能搜
            check([d["id"] for d in wzexport.list_sprite_drops("04030001")] == ["04030001"],
                  "按 id 搜不到 ✗")
            byname = wzexport.list_sprite_drops("药水")
            check([d["id"] for d in byname] == ["04030002"],
                  "按**名称**搜不到（用户要的就是「按 id、名称」两种 ✓）：%r" % (byname,))

            # ④ 并集：再塞一个"只有图、没有名字"的目录
            (tmp / "drop" / "04030077").mkdir(parents=True)
            (tmp / "drop" / "04030077" / "icon_0.png").write_bytes(b"x")
            ids = [d["id"] for d in wzexport.list_sprite_drops("04030077")]
            check(ids == ["04030077"],
                  "只有图标、清单里没有的那一项被漏了（并集 ✗）：%r" % (ids,))
            check(wzexport.drop_icon_path("04030077") is not None, "`drop_icon_path` 没找到图 ✗")

            # ⑤ 现状说明
            ok, why = wzexport.drop_library_state()
            check(ok and "清单" in why, "图库齐了却判不可用：%r" % (why,))
        finally:
            lib.restore()

        # 空图库 ⇒ 必须点出"缺哪一步"（现在现场就是这个状态 ✓）
        lib2 = _Lib(_tmpdir(), with_icons=False, with_index=False)
        try:
            ok2, why2 = wzexport.drop_library_state()
            check(not ok2, "图库什么都没有却判可用 ✗")
            check("dump-item" in why2 or "没" in why2,
                  "空图库的说明没点出缺什么（人只会以为名字打错了 ✗）：%r" % (why2,))
        finally:
            lib2.restore()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_card_ui():
    """② 卡片那四件事：勾选 → 整块显隐 / 列表 / **按钮启用条件** / 存进项目。

    钉六件（②③ 是用户 §2 那条 ✓）：
      ① 默认**不显示**掉落物那一块（没勾 ⇒ 界面上不该多出来 ✓）——用的是
         `setVisible`（藏起来才是"出现" ✓ 灰着仍是"一直在那儿" ✗）；
      ② 列表**空** ⇒ 两个按钮**灰**，且 tooltip 必须说清**为什么**（不然"点不动"和"坏了"分不清 ✗）；
      ③ 列表**有 1 项** ⇒ 两个按钮**亮**（"1 个以上"按"至少 1 项"实现 ✓ 见下面注释）；
      ④ 增删项能落盘（`label.drops` ✓）且**当场存**（不等跑一次 ✗）；
      ⑤ 勾选状态也按项目存（`label_drops_on` ✓）；
      ⑥ `make_task` 分派到 `tools.label_drops.run_label_drops` ✓ 且带全参数（不看这条
         ⇒ 按钮就是假的 ✗）。
    """
    import shutil

    from PyQt5.QtWidgets import QApplication
    from gui.project import Project
    from gui.steps.cards import LabelCard

    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")

    c = LabelCard()
    tmp = _tmpdir()
    try:
        # ① 默认整块藏着
        check(not c.drop_box.isVisibleTo(c),
              "没勾「标注掉落物」就把那一块显示出来了（卡片最下面会平白多一大块 ✗）")

        # ② 空列表 ⇒ 灰（并说清为什么）
        check(not c.btn_drop_tpl.isEnabled() and not c.btn_drop_yolo.isEnabled(),
              "列表还是空的，两个按钮就亮了 ⇒ 点下去只会报错 ✗")
        check("没有掉落物" in c.btn_drop_tpl.toolTip(),
              "灰着却没说为什么（「点不动」和「坏了」分不清 ✗）：%r" % (c.btn_drop_tpl.toolTip(),))

        # 勾上 ⇒ 出现
        c.ck_drop.setChecked(True)
        check(c.drop_box.isVisibleTo(c), "勾上「标注掉落物」那一块却没出现 ✗")

        # ③ 加一项 ⇒ 两个按钮都亮（⚠ "1 个以上" = 至少 1 项 ✓）
        c._drops = [{"id": "04030001", "name": "金币"}]
        c._refresh_drop_list()
        check(c.btn_drop_tpl.isEnabled() and c.btn_drop_yolo.isEnabled(),
              "列表里已经有 1 项，两个按钮还是灰的（用户 §2 ✗）")
        # ⚠ 换控件后看**格子里的条目**（背包窗格不写行文本 ✗ 名称/id 在悬停信息窗里 ✓）
        check(c.grid_drop.count() == 1,
              "背包窗格没显示那一项：%d" % c.grid_drop.count())
        check(c.grid_drop.entry_at(0).get("name") == "金币",
              "格子里没带上名称（悬停信息窗就没名字可显示 ✗）：%r" % (c.grid_drop.entry_at(0),))

        # ④⑤ 存读
        p = Project.create(tmp / "proj", name="用例", map_id="105090600")
        c.project = p
        c.sync(p)
        p.save()
        sec = Project.open(tmp / "proj").sec("label")
        check(sec.get("label_drops_on") is True,
              "勾选状态没按项目存（切项目就丢 ✗）：%r" % (sec.get("label_drops_on"),))
        check(sec.get("drops") == [{"id": "04030001", "name": "金币"}],
              "掉落物列表没存住：%r" % (sec.get("drops"),))
        c._drops = []
        c.ck_drop.setChecked(False)
        c.load_from_project(Project.open(tmp / "proj"))
        check(c._drops == [{"id": "04030001", "name": "金币"}] and c.ck_drop.isChecked(),
              "回填没把列表/勾选恢复：%r / %r" % (c._drops, c.ck_drop.isChecked()))

        # ⑥ 分派
        c._label_target = "drop"
        c._drop_mode = "template"
        fn, params = c.make_task(p)
        check(getattr(fn, "__name__", "") == "run_label_drops",
              "掉落物按钮没走到 `tools/label_drops.run_label_drops`（那按钮就是假的 ✗）：%r" % (fn,))
        check(params.get("mode") == "template" and params.get("drops"),
              "分派参数不对（mode / drops）：%r" % ({k: params[k] for k in ("mode", "drops")},))
        check(str(params.get("out", "")).endswith("labels_auto"),
              "标注输出目录不对（该写 labels_auto ✓）：%r" % (params.get("out"),))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_drop_buttons_open_frame_picker():
    """②c ⭐⭐ 掉落物那两个按钮**先开选帧弹窗**（用户 2026-10-04 ✓ 原话："标注掉落物、宠物的
    提交按钮需要先开选帧弹窗！（参考标注怪物、标注玩家、YOLO标注怪物、YOLO标注玩家）"）。

    与那四个按钮**逐条对齐**（见 `_emit_run` / `_emit_yolo` ✓）：
      ① **模板匹配标注** ⇒ 先确认重标（标过才问 ✓）**再**选帧；取消选帧 ⇒ **什么都不跑** ✓
         （老实现直接 `run_clicked` ⇒ 取消都没地方取消 ✗）；
      ② **YOLO标注** ⇒ 只选帧、**不确认**（追加性质 ✓ 同 YOLO 怪物/玩家 ✓）；
      ③ 选帧结果**透传**：`_only["drop"]` = 勾中的那些帧 ✓（None/空 = 全选 ✓）；
      ④ ⭐ **台账按 `"drop"` 分** —— 这轮顺带抓到一个冒名 bug：`detect_mobs` 里
         `mark_processed(out, "mob", …)` 是**写死**的 ✗ ⇒ 掉落物跑完会记进「怪物」那本台账
         ⇒ 选帧弹窗的「只选未处理」把掉落物跑过的帧当成**怪物标过了** ✗（现在跟着 `target` ✓）。
      ⑤ 选帧弹窗**标题**要叫「标注掉落物」（类别表里 `drop` 的短名是「掉落」✓ 那是给框/图例
         用的 ✓ ⇒ 文案例外收在 `labelio.TARGET_ZH` ✓ 一处口径 ✓）。
    """
    import shutil

    from PyQt5.QtWidgets import QApplication
    from gui import labelio
    from gui.frame_picker import FramePickDialog
    from gui.project import Project
    from gui.steps.cards import LabelCard

    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")
    c = LabelCard()
    c._drops = [{"id": "09000000", "name": "金币"}]
    c._refresh_drop_buttons()
    check(c.btn_drop_tpl.isEnabled(), "列表里有东西按钮却是灰的 ⇒ 下面几步就测不了 ✗")

    calls, emitted = [], []
    c.run_clicked.connect(lambda _w: emitted.append(c._label_target))
    c._confirm_relabel = lambda target: True

    # ① 模板匹配：先选帧；取消 ⇒ 一个信号都不发
    c._pick_frames = lambda target: (calls.append(target), (False, None))[1]
    c.btn_drop_tpl.click()
    check(calls == ["drop"] and not emitted,
          "模板匹配没先开选帧 / 取消之后照样跑了（老行为 ✗）：%r / %r" % (calls, emitted))

    # ② YOLO：弹选帧 + 不确认
    calls.clear()
    c._confirm_relabel = lambda target: (_ for _ in ()).throw(
        AssertionError("YOLO 是追加性质 ⇒ 不该问「确认重新标注」（同 YOLO 怪物/玩家 ✗）"))
    c.btn_drop_yolo.click()
    check(calls == ["drop"], "YOLO 那按钮没弹选帧：%r" % (calls,))

    # ③ 选帧结果透传
    calls.clear()
    c._confirm_relabel = lambda target: True
    c._pick_frames = lambda target: (calls.append(target), (True, ["frame_00007"]))[1]
    emitted.clear()
    c.btn_drop_yolo.click()
    check(c._only.get("drop") == ["frame_00007"] and emitted == ["drop"],
          "选帧结果没透传给执行端（那勾了等于没勾 ✗）：%r / %r"
          % (c._only.get("drop"), emitted))

    # ④⑤ 台账按 target 分 + 弹窗标题
    tmp = _tmpdir()
    try:
        p = Project.create(Path(tmp) / "proj", name="用例", map_id="105090600")
        p.frames.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            (p.frames / ("frame_%05d.png" % i)).write_bytes(b"x")
        labels = p.dir_of("labels_auto")
        check(labelio.processed_set(labels, "drop") == set(),
              "还没跑过就说跑过了：%r" % (labelio.processed_set(labels, "drop"),))
        labelio.mark_processed(labels, "drop", ["frame_00001"])
        check(labelio.processed_set(labels, "drop") == {"frame_00001"}
              and labelio.processed_set(labels, "mob") == set(),
              "台账没按 target 分（掉落物记进怪物那本 ✗）：%r / %r"
              % (labelio.processed_set(labels, "drop"), labelio.processed_set(labels, "mob")))
        # 执行端真会把 target 传下去（源码钉 ✓ —— 这条是"冒名"那个 bug 的护栏 ✓）
        #   ⚠ 只看**非注释行** ✓ 否则注释里引用的那句老代码会把断言带红 ✗（本轮就踩了一次 ✓）
        _dm = (ROOT / "tools" / "detect_mobs.py").read_text(encoding="utf-8")
        check("labelio.mark_processed(out, target, " in _dm,
              "`detect_mobs` 没把 target 传给台账（掉落物/宠物跑完会冒名记进怪物台账 ✗）")
        _bad = [ln for ln in _dm.splitlines()
                if "mark_processed(" in ln and '"mob"' in ln
                and not ln.strip().startswith("#")]
        check(not _bad,
              "`detect_mobs` 又把台账写死成 \"mob\" 了 ⇒ 掉落物/宠物跑完会冒名记进怪物台账 ✗：%r"
              % (_bad[:2],))
        #   ⚠ 钉**具体值**（只查 `"target":` 存在的话，传成 `"mob"` 也照样过 ✗ 反向验证抓到过 ✓）
        for f, want in (("tools/label_drops.py", '"target": "drop"'),
                        ("tools/label_pets.py", '"target": "pet"')):
            _f = (ROOT / f).read_text(encoding="utf-8")
            check(want in _f, "%s 没把 target=%r 传给执行端（台账会冒名记进 mob ✗）"
                              % (f, want.split(": ")[1]))
        # ⭐ 进度要**真的数**这两类（`_confirm_relabel` 按 target 查它 ✓）——
        #   少了 drop/pet ⇒ 那两类**第二次点**时一律被当成"还没标过"⇒ 不再问"已标过，继续吗"✗
        #   ⚠⚠ 必须查**数值**！只查"键在不在"是不够的 ✗：反向验证实测——删掉计数那几行，
        #     字典里照样有 `drop: 0`（return 里写死的键 ✓）⇒ 弱断言照样过 ✗（本轮踩过 ✓）。
        check(set(labelio.label_progress(p)) >= {"mob", "player", "drop", "pet", "total"},
              "`label_progress` 缺 drop/pet 这个键 ⇒ 重标时不会提醒 ✗：%r"
              % (sorted(labelio.label_progress(p)),))
        #   ⇒ 造一帧「只有掉落物框」的标注，看它数不数得出来 ✓
        (p.dir_of("labels_auto") / "frame_00000.txt").write_text(
            "2 0.500000 0.500000 0.100000 0.100000\n", encoding="utf-8")
        _prog = labelio.label_progress(p)
        check(_prog["drop"] == 1 and _prog["mob"] == 0 and _prog["pet"] == 0,
              "`label_progress` 没真数掉落物/宠物（只给个 0 的键 ⇒ 重标不提醒 ✗）：%r" % (_prog,))
        # 弹窗标题
        d = FramePickDialog(None, p, "drop")
        check("掉落物" in d.windowTitle(),
              "选帧弹窗对掉落物的标题不对（该是「标注掉落物」✓）：%r" % (d.windowTitle(),))
        check(d.processed == {"frame_00001"},
              "弹窗没按 target 读台账（「只选未处理」就废了 ✗）：%r" % (d.processed,))
        d2 = FramePickDialog(None, p, "pet")
        check("宠物" in d2.windowTitle() and d2.processed == set(),
              "弹窗对宠物的标题/台账不对：%r / %r" % (d2.windowTitle(), d2.processed))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_drop_task_carries_sprite_scale():
    """②d ⭐⭐ **掉落物任务也必须带项目的标定尺度**（与宠物那条同一个病 ✓，用户 2026-10-04 ✓）。

    掉落物和宠物一样是**游戏里画的精灵** ✓ ⇒ 尺度必须跟着项目标定走 ✓（怪那条一直是
    `p.get("scale") or 1.12` ✓）。⚠ 本条与宠物那条**一起**改的：两条原来都一个尺度都没传 ✗
    ⇒ 落到各自默认 1.0 ✓ ⇒ 同一批帧上什么都找不到 ✓（实测：1.0 下任何阈值 0 框 ✓、
    1.406 下能检出 ✓）—— 这也是本项目 `class2` 一直是 0 框的原因之一 ✓。
    """
    import shutil
    import tempfile
    from pathlib import Path

    from PyQt5.QtWidgets import QApplication
    from gui.project import Project
    from gui.steps.cards import LabelCard

    # ⚠⚠ **没它建 QWidget 会原生崩**（`0xC0000005`、且一句 Python 异常都没有 ✗ 本轮踩过 ✓）
    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")

    tmp = Path(tempfile.mkdtemp(prefix="drop_scale_"))
    try:
        p = Project.create(tmp / "proj", name="用例", map_id="105090600")
        p.frames.mkdir(parents=True, exist_ok=True)
        (p.frames / "frame_00000.png").write_bytes(b"x")
        c = LabelCard()
        c._drops = [{"id": "09000000", "name": "金币"}]
        c._label_target = "drop"
        c._drop_mode = "template"

        p.set("scale", 1.406)
        _fn, pr = c.make_task(p)
        check(pr.get("scale") == 1.406,
              "掉落物任务没带项目标定尺度（模板会小 40%% ⇒ 0 检出 ✗）：%r" % (pr.get("scale"),))

        # 怪那条走的是**同一个** helper ✓（口径一处 ✓）
        c._label_target = "mob"
        _fn, pr = c.make_task(p)
        check(pr["detect"].get("scale") == 1.406,
              "怪物那条的尺度被改坏了（本该还是项目标定值 ✗）：%r" % (pr["detect"].get("scale"),))

        # ⭐⭐ **③ 给掉落物单独标过 ⇒ 就用那一个**（用户 2026-10-05 ✓ 原话：
        #   "卡片3标定尺度 也要支持标定掉落物、宠物类"）—— 上面那条钉的是**没标时的
        #   兜底**（回落总尺度 ✓ 老项目一字不变 ✓），这条钉**标了之后它优先生效** ✓。
        c._label_target = "drop"
        p.set("drop_scale", 1.9)
        _fn, pr = c.make_task(p)
        check(pr.get("scale") == 1.9,
              "③ 给掉落物单独标定的尺度没被用上（还在拿总尺度跑 ✗）：%r"
              % (pr.get("scale"),))
        p.set("drop_scale", 0)          # 0 = 没标过（回落总尺度 ✓）
        _fn, pr = c.make_task(p)
        check(pr.get("scale") == 1.406,
              "把 drop_scale 清掉后没回落总尺度（老项目会崩 ✗）：%r" % (pr.get("scale"),))

        # ⭐⭐ **降采样也得是自己的**（用户 2026-10-04 ✓ 实测：本项目 `downscale=2` ⇒
        #   十几像素的金币图标被压糊 ⇒ **0 框** ✗；`1` ⇒ 检出 ✓）
        c._label_target = "drop"
        p.sec("label")["downscale"] = 2            # ⚠ 降采样在 **label 段**里（不是项目顶层 ✗）
        _fn, pr = c.make_task(p)
        check(pr.get("downscale") == 1,
              "掉落物任务的降采样跟着怪物那个走了（金币会被压糊 ⇒ 0 框 ✗）：%r"
              % (pr.get("downscale"),))
        c._label_target = "mob"
        _fn, pr = c.make_task(p)
        check(pr["detect"].get("downscale") == 2,
              "怪物那条的降采样被改坏了（它本来就该跟总的走 ✓）：%r"
              % (pr["detect"].get("downscale"),))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_coarse_gate():
    """②e ⭐ **粗筛**（用户 2026-10-04 ✓ 原话："匹配时，外观不符合的就不要运算了" ✓）。

    它只负责「**完全不像就别进最贵的那一步**」✓ —— 实测（同一批帧）：
      · 宠物那条：开/关 ⇒ 检出**完全一致**（28 框 = 28 框 ✓）、时间没降 ✗；
      · 掉落物那条：一样（17 = 17 ✓）；
      ⇒ 所以**默认关** ✗（见 `detect_mobs.COARSE_MIN_SCORE` 与 `coarse_gate` 的说明 ✓），
        开关与放行线都留着 ✓（模板彼此**不像**的场合才值得试 ✓）。
    这里的单元钉三件：① 明显不像 ⇒ 拦 ✓；② 明显像 ⇒ 放行 ✓（**绝不许多拦** ✗）；
    ③ 太小 / 没有底图 / 尺寸怪 ⇒ **一律放行** ✓（粗筛只许省时间，绝不许造成漏检 ✓）。
    """
    import cv2
    import numpy as np

    from tools.detect_mobs import COARSE_FACTOR, COARSE_MIN_SCORE, coarse_gate

    # ⭐ 两个底图（都要**粗结构** ✓ 细节缩到 1/8 会被平均掉 ✗ 就测不到东西了 ✓）：
    #   · 图案底：左上有一块 40×40 的暗块 ✓；· 纯色底：什么都没有 ✓。
    tpl_block = np.full((80, 80), 200.0, np.float32)
    tpl_block[0:40, 0:40] = 40.0                      # 与"图案底"里那处**逐像素一致** ✓
    frame_hit = np.full((240, 320), 200.0, np.float32)
    frame_hit[60:100, 40:80] = 40.0
    frame_flat = np.full((240, 320), 128.0, np.float32)   # 纯色画面：没有任何东西可匹配 ✓
    hit_small = cv2.resize(frame_hit, None, fx=1.0 / COARSE_FACTOR, fy=1.0 / COARSE_FACTOR,
                           interpolation=cv2.INTER_AREA)
    flat_small = cv2.resize(frame_flat, None, fx=1.0 / COARSE_FACTOR, fy=1.0 / COARSE_FACTOR,
                            interpolation=cv2.INTER_AREA)
    check(coarse_gate(hit_small, tpl_block, 80, 80) is True,
          "画面里明明有**一模一样**的图案 ⇒ 该放行，被拦了 ✗（那就是**漏检** ✓）")
    check(coarse_gate(flat_small, tpl_block, 80, 80) is False,
          "纯色画面上还想找图案 ⇒ 该拦（这就是**省时间**的场合 ✓）：%r"
          % (coarse_gate(flat_small, tpl_block, 80, 80),))
    check(coarse_gate(hit_small, np.full((80, 80), 200.0, np.float32), 80, 80) is True,
          "**平坦模板**（纯色 ✓ CCOEFF 下恒为 0 ✗）⇒ 该放行，被拦了 ✗")
    check(coarse_gate(flat_small, tpl_block, 80, 80) is False,
          "棋盘格落在纯色底上 ⇒ 该拦（这就是省时间的场合 ✓）：%r"
          % (coarse_gate(flat_small, tpl_block, 80, 80),))
    check(coarse_gate(None, tpl_block, 80, 80) is True, "没有底图 ⇒ 该放行 ✗")
    check(coarse_gate(hit_small, tpl_block, 8, 8) is True,
          "模板太小（缩完没形状 ✓）⇒ 该放行 ✗")
    check(coarse_gate(hit_small, tpl_block, 99999, 99999) is True, "尺寸怪 ⇒ 该放行 ✗")
    # ⚠ **平坦模板那条兜底用源码钉**（行为断言盖不住它 ✗：纯色模板在 CCOEFF 下取到什么值
    #   取决于 OpenCV 的退化处理 ✓ —— 反向验证实测"删掉它断言照样过" ✓ ⇒ 只能钉住"它在不在" ✓）
    _dm = (ROOT / "tools" / "detect_mobs.py").read_text(encoding="utf-8")
    check("float(ts.std()) < 1.0" in _dm,
          "粗筛里那条「**平坦模板一律放行**」的兜底没了 ✗（纯色模板会被误拦 ✓ 那是漏检 ✓）")
    check(0.0 < COARSE_MIN_SCORE < 0.6, "放行线跑到离谱区间：%r" % (COARSE_MIN_SCORE,))
    # ④ 默认开关 = **关**（实测没省时间 ✗ 要开得显式 ✓ 别偷偷开 ✗）
    src = (ROOT / "tools" / "detect_mobs.py").read_text(encoding="utf-8")
    bad = [ln for ln in src.splitlines()
           if 'params.get("coarse"' in ln and "True" in ln and not ln.strip().startswith("#")]
    check(not bad, "粗筛被默认打开了（实测没省时间 ✗ 要开得显式 ✓）：%r" % (bad[:2],))


def t_map_label_is_name_and_id():
    """②f ⭐ **地图一律显示成 `{地图名}_{id}`**（用户 2026-10-04 ✓ 原话："A 机部署台的当前地图
    显示格式应该是 `{地图名}_{id}`" ✓）。

    为什么值得单独一条：光给一串 `105040306` 人认不出是哪张图 ✓ —— 而且现场那次
    "**项目叫森林迷宫III、地图却是巨人之林**"就是这么暴露出来的 ✓（`{名}_{id}` 一眼可见 ✓）。
    钉四件：① 查得到名 ⇒ `名_id`（真实清单 ✓）；② 查不到 ⇒ **只给 id**（**不编** ✗）；
    ③ 空 id ⇒ 空串 ✓；④ 两次读走缓存但结果一致 ✓；⑤ 部署台那行**真的用它** ✓（源码钉 ✓
    —— 部署台在 A 机上，跑不到它的界面 ✓）。
    """
    from core import wzexport

    m = wzexport.map_label("105040303")
    check(m.endswith("_105040303") and len(m) > len("_105040303"),
          "真实地图 id 没拼成「地图名_id」✗：%r（清单在 datasets/maps.json ✓）" % (m,))
    check(wzexport.map_name_of("105040303"),
          "地图名查不出来（`map_name_of` ✗）：%r" % (wzexport.map_name_of("105040303"),))
    check(wzexport.map_label("999999999") == "999999999",
          "查不到名字时该**只给 id**（不许编名字 ✗）：%r" % (wzexport.map_label("999999999"),))
    check(wzexport.map_label("") == "",
          "空 id 该给空串 ✓：%r" % (wzexport.map_label(""),))
    check(wzexport.map_label("105040303") == m,
          "第二次读（走缓存）结果变了 ✗ ⇒ 缓存键写错了 ✓")

    # ⚠ 这条钉**那一行本身** ✗（第一版只查 `"wzexport.map_label" in 文件` ⇒ 反向验证实测：
    #   "当前地图"那行改回裸 id、可 `_label()` 这个助手还在文件里 ⇒ 断言照样过 ⇒ **假绿** ✓）
    _dp = (ROOT / "deploy" / "app.py").read_text(encoding="utf-8")
    check('return "%s（%s%s）" % (_label(mid)' in _dp,
          "部署台「当前地图」那行没用统一口径（该走 `wzexport.map_label` ✓ 不然迟早两处不一致 ✗）")


def t_icon_grid_fit_rows():
    """②g ⭐ **背包窗格按内容自适应行数**（用户 2026-10-04 ✓ 原话："要标注的掉落物、宠物的
    背包窗格需要自适应有几行（至少一行）" ✓）。

    ⚠ 原来卡片上两块都写死 `37*3+12`（固定三行 ✗）：只加 1~2 项时留一大片空白 ✓。
    钉三件：① 0 项 / 1 项 ⇒ **1 行**（"至少一行" ✓ 不许塌成 0 ✓）；② 项多了 ⇒ 行数**跟着涨** ✓；
    ③ 宽度变大（每行放得更多）⇒ 行数**不增** ✓；④ 源码钉：卡片里不许再写死三行 ✗。
    """
    from PyQt5.QtWidgets import QApplication

    from gui.icon_grid import IconGrid

    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")
    g = IconGrid()
    g.resize(400, 40)
    app.processEvents()

    g.set_entries([])
    app.processEvents()
    check(g.fit_rows() == 1 and g.minimumHeight() > 0,
          "空表该是**一行高**（塌成 0 就看不见了 ✗）：行 %r 高 %r" % (g.fit_rows(), g.minimumHeight()))
    h1 = g.minimumHeight()

    g.set_entries([{"id": "09000000"}])
    app.processEvents()
    check(g.fit_rows() == 1 and g.minimumHeight() == h1,
          "1 项时和空表该一样高（都是一行 ✓）：%r / %r" % (g.fit_rows(), g.minimumHeight()))

    g.set_entries([{"id": "090000%02d" % i} for i in range(60)])
    app.processEvents()
    r_many = g.fit_rows()
    check(r_many > 1 and g.minimumHeight() > h1,
          "项多了行数/高度没跟上（还写死一行？✗）：行 %r 高 %r" % (r_many, g.minimumHeight()))

    g.resize(1200, g.minimumHeight())
    app.processEvents()
    check(g.fit_rows() <= r_many,
          "宽度变大后每行能放更多 ⇒ 行数不该变多 ✗：%r ⇒ %r" % (r_many, g.fit_rows()))

    _cards = (ROOT / "gui" / "steps" / "cards.py").read_text(encoding="utf-8")
    check("37 * 3" not in _cards,
          "卡片里又把背包窗格的高度**写死**了 ✗（要 `fit_rows` 自适应 ✓）")


def t_drop_thresh_param():
    """②b ⭐ 「掉落物那块**自己**的匹配阈值」（用户 2026-10-04 ✓ 原话："勾选标注掉落物、标注宠物后，
    需要有参数「匹配阈值」"）。

    为什么必须拆出来：原来掉落物**直接吃总的那个阈值** ✗（`make_task` 里 `sec.get("thresh")` ✓
    注释还写着"与标注怪物同一份"✓），而掉落物图标**又小又统一**（金币才十几像素 ✓）⇒ 和怪物
    常常不该是一个数 ✓（太小/太像 ⇒ 要么假匹配一堆、要么一个都匹配不上 ✓）。

    钉五件：
      ① 卡片上**有**这个字段，而且它**在掉落物那一块里**（没勾 ⇒ 跟着一起藏 ✓ 正是用户说的
         "勾选…后需要有这个参数" ✓）；
      ② **老项目**（`label` 段只有 `thresh`、没有 `drop_thresh`）⇒ 字段显示**那个数** ✓，
         喂给 worker 的也是那个数 ✓（**老行为一字不变** ✗ 别把已经配好的项目改坏）；
      ③ 只改这一项 ⇒ worker 拿**新的** ✓，而总的 `thresh` **不动** ✓（只影响掉落物这一块 ✓）；
      ④ 这一项**能存进项目**（切项目不丢 ✓）；
      ⑤ 源码钉兜底链：`drop_thresh` → `thresh` → 0.90 ✓（老项目就靠它 ✓）。
    """
    import shutil

    from PyQt5.QtWidgets import QApplication
    from gui.project import Project
    from gui.steps.cards import LabelCard

    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")
    tmp = _tmpdir()
    c = LabelCard()
    try:
        # ① 有这个字段 + 它在掉落物那一块里
        check("drop_thresh" in getattr(c, "widgets", {}),
              "掉落物那块没有「匹配阈值」这个参数（用户 2026-10-04 点名要 ✗）")
        w = c.widgets["drop_thresh"][0]
        check(c.drop_box.isAncestorOf(w),
              "这个参数没放在「标注掉落物」那一块里 ⇒ 没勾选时它还杵在卡片上 ✗")
        check(not c.drop_box.isVisibleTo(c),
              "没勾「标注掉落物」时，那一块（连同这个参数）就该藏着 ✗")

        # ② 老项目：只有 thresh ⇒ 显示/使用都该是它
        p = Project.create(tmp / "proj", name="用例", map_id="105090600")
        c.project = p
        p.sec("label")["thresh"] = 0.86
        c.load_from_project(p)
        check(abs(float(w.value()) - 0.86) < 1e-9,
              "老项目（没有 drop_thresh）时该显示「现在生效的那个数」（= 总阈值 ✓）：%r"
              % (w.value(),))
        c._label_target = "drop"
        c._drop_mode = "template"
        c._drops = [{"id": "04030001", "name": "金币"}]
        _fn, params = c.make_task(p)
        check(abs(float(params.get("thresh")) - 0.86) < 1e-9,
              "老项目喂给 worker 的阈值变了（老行为该一字不变 ✗）：%r" % (params.get("thresh"),))

        # ③④ 只改这一项 ⇒ worker 拿新的、总的不动、而且存得住
        w.setValue(0.72)
        c.sync(p)
        p.save()
        _fn2, params2 = c.make_task(p)
        check(abs(float(params2.get("thresh")) - 0.72) < 1e-9,
              "改了掉落物自己的阈值却没生效（那这个参数就是假的 ✗）：%r"
              % (params2.get("thresh"),))
        sec = Project.open(tmp / "proj").sec("label")
        check(abs(float(sec.get("drop_thresh", -1)) - 0.72) < 1e-9,
              "这一项没存进项目（切项目就丢 ✗）：%r" % (sec.get("drop_thresh"),))
        check(abs(float(sec.get("thresh", -1)) - 0.86) < 1e-9,
              "改掉落物阈值把**总阈值**也写坏了（不该 ✗）：%r" % (sec.get("thresh"),))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ⑤ 源码钉兜底链（老项目就靠它 ✓）
    _csrc = (ROOT / "gui" / "steps" / "cards.py").read_text(encoding="utf-8")
    check('sec.get("drop_thresh", sec.get("thresh", 0.90))' in _csrc,
          "`make_task` 里没有「drop_thresh → thresh → 0.90」这条兜底 ⇒ 老项目会被卡死或改坏 ✗")


def t_worker_dispatch():
    """③ 按钮的执行端：**两条都复用既有算法**，且类别号是 **class 2（drop）** ✓。

    这是整件事"真不真"的分界（⛔ 空跑当成功是最坏的做法 ✗）⇒ 用**假算法**把参数截下来钉：
      ① 前置没图库 ⇒ `RuntimeError` 且说明里点出缺什么 ✓（不静默成功 ✗）；
      ② 列表空 ⇒ `ValueError` ✓；
      ③ 模板那条 ⇒ 转发 `detect_mobs.run_detect`，且 `cls == CLASS_DROP`、
         `sprites` 指向**掉落物图库**、`mobs` 就是选中的 id ✓；
      ④ YOLO 那条 ⇒ 转发 `yolo_augment.run_yolo_augment`，`mode == "drop"` ✓；
      ⑤ `yolo_augment.MODE_CLASS` 是**一处口径**（加一类只改这里 ✓）且 drop ⇒ 2 ✓；
      ⑥ ⭐ **掉落物链自己那两处收紧真的传下去了**（用户 2026-10-06 ✓）：
         `min_energy_ratio == DROP_MIN_ENERGY_RATIO`（0.55 > 怪物那个 0.35 ✓）、
         `mirror is False`（道具图标没有左右朝向 ✓）。
    """
    import shutil

    from core.context import TaskContext
    from core import wzexport
    from perception.classes import CLASS_DROP
    from tools import label_drops as LD

    ctx = TaskContext()
    # ① 没图库 ⇒ 说清缺什么
    lib0 = _Lib(_tmpdir(), with_icons=False, with_index=False)
    try:
        try:
            LD.run_label_drops({"mode": "template", "drops": [{"id": "1"}],
                                "frames": "x", "out": "y"}, ctx)
            raise AssertionError("图库什么都没有却当成成功（⛔ 空跑当成功 ✗）")
        except RuntimeError as e:
            check("图库" in str(e), "报错没说清是图库的问题：%r" % (str(e)[:80],))
    finally:
        lib0.restore()

    tmp = _tmpdir()
    lib = _Lib(tmp, drops=(("04030001", "金币"),))
    try:
        # ② 列表空
        try:
            LD.run_label_drops({"mode": "template", "drops": [], "frames": "x",
                                "out": "y"}, ctx)
            raise AssertionError("没选掉落物却跑起来了 ✗")
        except ValueError:
            pass

        # ③ 模板那条：截参数（把 run_detect 换成假的 ✓ 它是在函数里 import 的 ⇒ 打模块属性 ✓）
        from tools import detect_mobs as DM
        got = {}
        _real = DM.run_detect

        def _fake_detect(params, ctx=None):
            got.update(params)
            return {"ok": True}

        DM.run_detect = _fake_detect
        try:
            LD.run_label_drops({"mode": "template", "drops": [{"id": "04030001"}],
                                "frames": str(tmp), "out": str(tmp / "out"),
                                "thresh": 0.9}, ctx)
        finally:
            DM.run_detect = _real
        check(got.get("cls") == CLASS_DROP,
              "模板那条没写 **class 2（drop）**（写错类 = 标到怪物头上 ✗）：%r" % (got.get("cls"),))
        check(str(got.get("sprites")) == str(lib.spr),
              "模板目录没指向掉落物图库：%r" % (got.get("sprites"),))
        check(got.get("mobs") == ["04030001"],
              "传给模板匹配的 id 不对：%r" % (got.get("mobs"),))
        # ⭐⭐ **掉落物链自己的两处收紧**（用户 2026-10-06 ✓ 都在 `tools/label_drops.py` ✓）：
        #   ① 窗口能量比（小图标会在"纯色 / 黑边"上留高分假峰 ⇒ 用能量闸砍 ✓）；
        #   ③ 不镜像（道具图标没有左右朝向 ✓）。
        #   ⚠ 两条都是"**不传就等于没生效**"的形态 ✗（`run_detect` 有自己的怪物默认值 ✓）
        #     ⇒ 必须在这儿钉"真的传下去了" ✓。
        check(float(got.get("min_energy_ratio", 0)) == float(LD.DROP_MIN_ENERGY_RATIO),
              "「窗口能量比」没传给模板匹配（那它就用怪物那个 0.35 ⇒ 小图假峰照旧 ✗）：%r"
              % (got.get("min_energy_ratio"),))
        check(float(LD.DROP_MIN_ENERGY_RATIO) > 0.35,
              "掉落物的「窗口能量比」没有比怪物那个（0.35）更严（那就等于没调 ✓）：%r"
              % (LD.DROP_MIN_ENERGY_RATIO,))
        check(got.get("mirror") is False,
              "掉落物还在**镜像**模板（道具图标没有朝向 ⇒ 白花一倍时间 + 多一批假框 ✗）：%r"
              % (got.get("mirror"),))

        # ④ YOLO 那条
        from tools import yolo_augment as YA
        got2 = {}
        _real2 = YA.run_yolo_augment

        def _fake_yolo(params, ctx=None):
            got2.update(params)
            return {"ok": True}

        YA.run_yolo_augment = _fake_yolo
        try:
            LD.run_label_drops({"mode": "yolo", "drops": [{"id": "04030001"}],
                                "frames": str(tmp), "out": str(tmp / "out"),
                                "weights": str(tmp / "w.pt")}, ctx)
        finally:
            YA.run_yolo_augment = _real2
        check(got2.get("mode") == "drop",
              "YOLO 那条没传 mode=drop（会去合并 class 1 怪物框 ✗）：%r" % (got2.get("mode"),))

        # ⑤ 类别映射一处口径
        check(YA.MODE_CLASS.get("drop") == CLASS_DROP and
              YA.MODE_CLASS.get("mob") == 1 and YA.MODE_CLASS.get("player") == 0,
              "`MODE_CLASS` 不对（加一类该只改这张表 ✓）：%r" % (YA.MODE_CLASS,))

        # ⑥ 源码钉：`detect_mobs` 的类别号是**参数化**的（默认仍是 1 = 老行为 ✓）
        dm_src = (ROOT / "tools" / "detect_mobs.py").read_text(encoding="utf-8")
        check('params.get("cls", CLASS_MOB)' in dm_src,
              "`run_detect` 没有 `cls` 参数（掉落物就没法写 class 2 ✗）")
        check('cfg.get("cls", CLASS_MOB)' in dm_src and '"cls": cls,' in dm_src,
              "`cls` 没顺着 `cfg` 进到**子进程**里（写出去的还是 class 1 ✗）")
        check('if ln.split() and ln.split()[0] != str(_cls)' in dm_src,
              "覆盖旧框时没按 `_cls` 过滤 ⇒ 标掉落物会把**怪物框**一起清掉 ✗")
    finally:
        lib.restore()
        shutil.rmtree(tmp, ignore_errors=True)


def t_drop_picker_rows():
    """④ 掉落物弹窗：**两区都是背包窗格**，信息在**悬停信息窗**里（用户 2026-10-04 ✓ 两条原话
    —— 先"希望能像「确认要识别的怪物」那样看到 icon、名称、id" ✓，后"以背包窗格的形式展示
    项目，鼠标指着的时候显示信息弹窗（MapleNecrocer 应该有相关实现）" ✓）。

    改造前实测：图标是好的 ✓、但文本只有 `09000000`（金币整类在 `String.wz` 里**没有名字** ✗
    而界面不说为什么 ✗），而且**不打字就 0 个候选** ✗（照抄 `mob_picker` 那条"输入才显示"的
    规矩 ✓ —— 怪有名字、掉落物没有 ⇒ 那条规矩在这里等于**逼人去猜 8 位 id** ✗）。

    钉六件：
      ① 已选区是格子：**有图标**、格子里存着 id ✓；
      ② 悬停信息里有 **id**，没名字的如实写「（WZ 里没有名字）」+ 怎么补 ✓
         —— 只显一串数字会让人以为读坏了 ✗；
      ③ ⭐ **不打字也能浏览**（>0 格 ✓、且 ≤ 上限 ✓、提示语带「图库共」✓）；
      ④ 打了关键词 ⇒ 只剩匹配的 ✓（按 **id** 与按 **名字** 两条都验 ✓），
         且**已选的那项不再出现**在候选里 ✓；
      ⑤ 悬停信息里带**类别 / 几帧**（"金币会动、有 4 帧"要一眼看得到 ✓）；
      ⑥ 源码钉：两区**共用** `IconGrid` ✓、弹窗里**不再自己拼行文本**（`_drop_row_text`
         那套已删 ✓ 口径只此一处 ✓）、空关键词真的会去列 ✓。
    """
    import shutil

    from PyQt5.QtWidgets import QApplication

    from gui.drop_picker import DropPickDialog
    from gui.icon_grid import tip_lines

    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")

    tmp = _tmpdir()
    lib = _Lib(tmp, drops=(("04030001", "金币"), ("09000000", ""),
                           ("09000001", "")),
               meta={"09000000": {"category": "Special", "frames": 4},
                     "09000001": {"category": "Special", "frames": 4}})
    try:
        dlg = DropPickDialog([{"id": "09000000", "name": ""}])
        # ③ 不打字就有候选
        check(dlg.cand.count() > 0,
              "不打字一个候选都不列 ⇒ 掉落物**没有名字**时等于逼人去猜 id ✗")
        check(dlg.cand.count() <= 200, "空关键词没受上限约束（图库大时会卡 ✗）")
        check("图库共" in dlg.lbl_match.text(),
              "空关键词那句提示没说清「图库里一共多少」：%r" % (dlg.lbl_match.text(),))

        # ① 已选区：格子 + 图标 + id
        check(dlg.lst.count() == 1 and not dlg.lst.item(0).icon().isNull(),
              "已选区那一格没有图标（用户要的就是看 icon ✗）")
        check(dlg.lst.entry_at(0)["id"] == "09000000",
              "已选区那一格没存住 id：%r" % (dlg.lst.entry_at(0),))
        # ② 悬停信息：id 要在、没名字要写明、要说怎么补
        t1 = "\n".join(tip_lines(dlg.lst.entry_at(0)))
        check("09000000" in t1, "悬停信息里没写出 id：%r" % (t1,))
        check("WZ 里没有名字" in t1,
              "没名字的项只显示一串数字（人会以为读坏了 ✗）：%r" % (t1,))
        check("drops.json" in t1, "没说「没名字怎么办」✗：%r" % (t1,))

        # ④ 关键词筛选 + 已选不重复（按 id ✓ 按名字 ✓ 两条都验）
        dlg.ed_filter.setText("0900")
        ids = dlg.cand.ids()
        check(ids and all(i.startswith("0900") for i in ids),
              "按 id 筛之后混进了别的项：%r" % (ids[:4],))
        check("09000000" not in ids,
              "已经选过的那项还在候选里（会重复添加 ✗）：%r" % (ids[:4],))
        dlg.ed_filter.setText("金币")
        check(dlg.cand.ids() == ["04030001"],
              "按**名字**搜不到（名字表没用上 ✗）：%r" % (dlg.cand.ids(),))

        # ⑤ 悬停信息带类别/几帧
        #   ⚠ 筛 `09000001`（不能用 `09000000` —— 那是**已选**的，候选里被排掉了 ✗
        #     第一版就是这么写出假红的 ✓）
        dlg.ed_filter.setText("09000001")
        check(dlg.cand.count() >= 1,
              "筛 `09000001` 一个候选都没有（它没被选过，该出现 ✓）：%d" % dlg.cand.count())
        t2 = "\n".join(tip_lines(dlg.cand.entry_at(0)))
        check("Special" in t2 and "4 帧" in t2,
              "悬停信息没带类别/几帧（「金币会动」这件事看不到 ✗）：%r" % (t2,))
    finally:
        lib.restore()
        shutil.rmtree(tmp, ignore_errors=True)

    # ⑥ 源码钉
    _psrc = (ROOT / "gui" / "drop_picker.py").read_text(encoding="utf-8")
    check(_psrc.count("IconGrid()") >= 2,
          "已选区 / 候选区没用同一个背包窗格控件（各画一套 ⇒ 迟早一边缺东西 ✗）")
    check("def _drop_row_text" not in _psrc,
          "弹窗里又自己拼行文本了（名称/id 的口径要**只此一处**：`icon_grid.tip_lines` ✓）")
    check("pool = self._all if not kw else" in _psrc,
          "空关键词又变成「不列候选」了（掉落物没名字 ⇒ 那就是逼人猜 id ✗）")


def t_drop_desc_shows():
    """⭐⭐ 掉落物**描述（desc）**要能导出并显示（用户 2026-10-04 ✓ 第 1 条："现在可以全量导出。
    记得用 `dump-items` 把 desc 一并导出**并做进显示功能**"）。

    钉五件：
      ① `clean_desc`（纯函数 ✓）：字面 `\\r\\n` / `\\n` ⇒ **真换行** ✓、真制表符 ⇒ 空格 ✓、
         `#c…#`（Maple 颜色码）⇒ 去掉 ✓、空 ⇒ 空 ✓、**正文里的 `%` 不许动** ✗；
      ② 清单 meta 带 desc ✓ 且**出口就洗过** ✓（消费方不必各 replace 一遍 ✓）；
      ③ 候选 `list_sprite_drops()` 也带 desc ✓；
      ④ `tip_lines()` 里**显示**它 ✓（信息窗里那行 ✓ = "做进显示功能" ✓）；
      ⑤ ⭐ **真数据完整性**：`datasets/drops.json` 在 ⇒ 必须能解析**且非空** ✓ ——
         本轮它真坏过一次（导出器 `J()` 没转控制字符 ⇒ 16 行带真制表符 ⇒ 整份不是合法 JSON ✗，
         而 `load_drop_index` **静默吞异常 ⇒ 返回空表** ✗ ⇒ 界面看着只是"图库没了" ✓ 最难查 ✓）。
    """
    import json
    import shutil

    from core import wzexport
    from gui.icon_grid import tip_lines

    # ① 清洗
    check(wzexport.clean_desc("红色药水.\\n恢复HP约50.") == "红色药水.\n恢复HP约50.",
          "字面 `\\n` 没换成真换行：%r" % (wzexport.clean_desc("红色药水.\\n恢复HP约50."),))
    check(wzexport.clean_desc("a\\r\\nb") == "a\nb", "字面 `\\r\\n` 没换成真换行 ✗")
    check(wzexport.clean_desc("a\tb") == "a b", "真制表符没换空格 ✗")
    check(wzexport.clean_desc("#c红色#的药水") == "红色的药水", "`#c…#` 颜色码没去掉 ✗")
    check(wzexport.clean_desc("") == "" and wzexport.clean_desc(None) == "",
          "空描述该原样给空 ✗")
    check("%d%%" in wzexport.clean_desc("伤害 %d%%"),
          # ⚠ 这个字符串里**不能再出现裸的 `%`** ✗ —— 它后面跟着 `% r` 这套格式化参数，
          #   写「`%`」会被当成格式符 ⇒ `ValueError: unsupported format character` ✓（本轮踩过 ✓）
          "把正文里的百分号也替换了（那是真文字 ✗）：%r"
          % (wzexport.clean_desc("伤害 %d%%"),))

    tmp = _tmpdir()
    lib = _Lib(tmp, drops=(("02000000", "红色药水"),),
               meta={"02000000": {"category": "Consume", "frames": 1,
                                  "desc": "红色药草研磨作成的药水.\\n恢复HP约50. #c红色#"}})
    try:
        # ②③④ 一条链：清单 → meta → 候选 → 信息窗
        m = wzexport.load_drop_meta().get("02000000") or {}
        check(m.get("desc") == "红色药草研磨作成的药水.\n恢复HP约50. 红色",
              "清单里的 desc 没洗好 / 没读出来：%r" % (m.get("desc"),))
        c = wzexport.list_sprite_drops("02000000")
        check(c and c[0].get("desc") == m["desc"],
              "候选里没带 desc（信息窗就没得显示 ✗）：%r" % (c[0] if c else None,))
        lines = tip_lines(c[0])
        check(any("红色药草研磨作成的药水." in l for l in lines),
              "信息窗里没显示描述（第 1 条要「做进显示功能」✗）：%r" % (lines,))
    finally:
        lib.restore()
        shutil.rmtree(tmp, ignore_errors=True)

    # ⑤ 真数据完整性（本机有就查 ✓ 没有就跳过 ✓ —— 别拿"环境没有"当失败 ✗）
    real = ROOT / "datasets" / "drops.json"
    if real.exists() and real.stat().st_size > 2:
        try:
            rows = json.loads(real.read_text(encoding="utf-8"))
        except Exception as e:                        # noqa: BLE001
            raise AssertionError(
                "datasets/drops.json **不是合法 JSON**（%s: %s）—— 导出器的字符串转义漏了"
                "控制字符就会这样（真发生过 ✗），而 Python 侧**静默吞成空表** ⇒ 界面只是"
                "看着「图库没了」✗ 最难查" % (type(e).__name__, str(e)[:80]))
        check(rows, "datasets/drops.json 解析出来是空的 ✗")
        check(len(wzexport.load_drop_index()) == len(rows),
              "`load_drop_index()` 读到的条数与文件对不上（%d vs %d）⇒ 读的那条路有坑 ✗"
              % (len(wzexport.load_drop_index()), len(rows)))


def t_field_tips_and_copyable():
    """⭐⭐ 提示挂**字段标题** + 标题/说明**可复制**（用户 2026-10-04 ✓ 第 3、4 条）。

    原文："重构所有的 tip 弹出逻辑。现在是在哪里编辑（输入框、按钮、触控板等）就在哪里弹出，
    改成**指着字段标题时弹出**。例如「要标注的掉落物」。然后修改 UI 规范" /
    "所有字段标题、灰字说明文本希望能够**复制文本**"。

    钉七件：
      ① 卡片里「要标注的掉落物」**标题**上有提示 ✓，**背包窗格自己没提示** ✓
         （第 2 条：窗格每格已有信息窗 ⇒ 再挂一层会两个框叠着 ✗）；
      ② 字段提示确实搬到了**标题**上、**控件上没有** ✓（抽 `per_mob` 一个 ✓）；
      ③ 那个标题**能选中复制** ✓；
      ④ `title_label(...)` / `field_tip(...)` 两个 helper 都让标签可复制 ✓；
      ⑤ `theme.enable_label_copy(窗口)` 把整窗 QLabel 设成可选中 ✓，且**只加鼠标那个标志** ✓
         （加了 `TextSelectableByKeyboard` 会让每个标签变成可聚焦 ⇒ Tab 顺序里多一堆停靠点 ✗）；
      ⑥ 源码钉：`bind_window_state` 里**真的调了** `enable_label_copy` ✓（那是"一处保证" ✓
         20 个窗口/面板的收尾都走它 ✓ 只写函数不接上 = 没修 ✗）；
      ⑦ 规范同步：`docs/UI规范.md` §6 里这两条都写着 ✓（规则改了文档不改 ⇔ 下一个人照旧 ✗）。
    """
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication, QLabel, QWidget

    from gui import theme
    from gui.steps.cards import LabelCard
    from gui.widgets import field_tip, title_label

    app = QApplication.instance() or QApplication([])

    # ① 卡片：掉落物标题 + 窗格
    c = LabelCard()
    try:
        check("这次要标的掉落物" in c.lbl_drop_title.toolTip(),
              "「要标注的掉落物」**标题**上没有提示（第 3 条要的就是指标题弹 ✓）：%r"
              % (c.lbl_drop_title.toolTip(),))
        check(c.grid_drop.toolTip() == "",
              "背包窗格自己还挂着 tip（用户第 2 条：它不需要 ✓ 而且会和信息窗叠成两层 ✗）：%r"
              % (c.grid_drop.toolTip(),))
        check(bool(c.lbl_drop_title.textInteractionFlags() & Qt.TextSelectableByMouse),
              "字段标题选不中、复制不了（第 4 条 ✗）")

        # ② 搬过来的字段提示：在标题上、控件上没有
        w = c.widgets["per_mob"][0]
        # ⚠⚠ 2026-10-05：`per_mob` 已经**搬进「要标注的怪物」那一段**了 ✗ ⇒ 它的标题在
        #   **那一段自己的表单**里 ⇒ `c.form.labelForField(w)` 返回 None ✗（本条就是这么红的 ✓）。
        #   卡片把「每个 key 建在哪个表单里」记在 `_field_forms` 上 ✓（`base.tip()` 也靠它 ✓
        #   一处口径 ✓）⇒ 用例跟着它找 ✓ 别自己写死主表单 ✗。
        _form = (getattr(c, "_field_forms", None) or {}).get("per_mob") or c.form
        lab = _form.labelForField(w)
        check(lab is not None and "模板帧" in lab.toolTip(),
              "字段提示没挂到标题上（第 3 条 ✗）：%r" % (lab.toolTip() if lab else None,))
        check(w.toolTip() == "",
              "提示**两处都留**了 ⇒ 鼠标一指会弹两个框叠在一起（比不搬还糟 ✗）：%r" % (w.toolTip(),))
        check(bool(lab.textInteractionFlags() & Qt.TextSelectableByMouse),
              "字段标题选不中（第 4 条 ✗）")
    finally:
        c.close()

    # ④ 两个 helper
    l1 = title_label("标题", tip="说明")
    check(bool(l1.textInteractionFlags() & Qt.TextSelectableByMouse) and l1.toolTip() == "说明",
          "`title_label` 没做到「可复制 + 提示在标题上」✗")
    l2 = QLabel("标题2")
    field_tip(l2, "说明2")
    check(bool(l2.textInteractionFlags() & Qt.TextSelectableByMouse) and l2.toolTip() == "说明2",
          "`field_tip` 没做到「可复制 + 提示在标题上」✗")

    # ⑤ 整窗扫一遍
    box = QWidget()
    for txt in ("a", "b", "c"):
        QLabel(txt, box)
    n = theme.enable_label_copy(box)
    check(n == 3, "整窗可复制没扫到全部标签：%d（该 3 ✓）" % n)
    check(all(bool(x.textInteractionFlags() & Qt.TextSelectableByMouse)
              for x in box.findChildren(QLabel)),
          "有标签没被设成可选中 ✗")
    check(not any(bool(x.textInteractionFlags() & Qt.TextSelectableByKeyboard)
                  for x in box.findChildren(QLabel)),
          "顺手加了键盘选中 ⇒ 每个标签都可聚焦、Tab 顺序里多一堆停靠点 ✗")
    box.deleteLater()

    # ⑧ ⭐⭐ **用户实测的那一条**（2026-10-04 ✓ 原话："实测没有实现提示挂字段标题 例如
    #   决策参数页签→ 输出行为CD 输入框，指着还是有提示弹出。标题「输出行为CD(ms)」指着也没有
    #   标题弹出，并且不能复制 我说的是 gui 所有的配置项"）。
    #   根因两条：⑴ 那 62 处提示**本来就还在控件上** ✗（上一版只搬了卡片 ✓ 说好下一批 ✓）；
    #   ⑵ ⚠⚠ **页签面板根本不走收尾钩子** ✗ —— `bind_window_state` 只有**弹窗**在调 ✗
    #   ⇒ `enable_label_copy` / 搬家都没跑过 ✓（"实测没有实现"就是这么来的 ✓）。
    #   现在收尾是 `theme.finish_window` ✓，**弹窗 / 主窗口 / 页签工厂**三处都调 ✓。
    #   ⚠ 这一段用**等价最小复刻**（QFormLayout + 数字框 + 提示 ✓）而不是去建真的
    #   `PlayerPanel` —— 真面板在离屏套件里会**卡死**（本轮实测：整套 400s 超时 ✗）⇒
    #   真面板那条路由下面几条**源码钉**兜住（弹窗 / 主窗口 / 页签工厂三处都调 ✓）。
    from PyQt5.QtWidgets import (QCheckBox, QFormLayout, QHBoxLayout,
                                 QPushButton, QSpinBox)

    holder = QWidget()
    form = QFormLayout(holder)
    sp = QSpinBox()
    sp.setToolTip("输出行为之后锁定多久（毫秒）")
    form.addRow("输出行为CD(ms)", sp)
    try:
        lab = theme._row_title_of(sp)
        check(lab is not None and lab.text().strip() == "输出行为CD(ms)",
              "认不出「输出行为CD(ms)」那一行的标题（收尾得先认得它才能搬 ✓）：%r"
              % (lab.text() if lab is not None else None,))
        _n_tips, _n_lbl = theme.finish_window(holder)
        check(_n_tips == 1 and _n_lbl >= 1,
              "收尾没搬这一处 / 没设可复制：%r" % ((_n_tips, _n_lbl),))
        check(sp.toolTip() == "",
              "收尾之后**控件上还留着提示** ⇒ 指着输入框照样弹 ✗（用户实测那条 ✗）：%r"
              % (sp.toolTip(),))
        check(lab.toolTip() == "输出行为之后锁定多久（毫秒）",
              "提示没搬到标题上 ⇒ 指着「输出行为CD(ms)」还是没反应 ✗（用户实测那条 ✗）")
        check(bool(lab.textInteractionFlags() & Qt.TextSelectableByMouse),
              "标题还是选不中、复制不了 ✗（用户实测那条 ✗）")
        # ⚠ 找不到标题的（工具栏按钮那种）**原样留着** ✓ —— 宁可留在控件上，别把提示弄丢 ✗
        btn = QPushButton("复制")
        btn.setToolTip("复制选中的框（Ctrl+C）")
        # ⚠ 更要紧的是这种：**它本身是配置控件，但那一行没有标题**（孤立勾选框 ✓）——
        #   收尾**必须原样留着**提示 ✗（清了就是"提示直接弄丢" ✓ 比搬不动更糟 ✓）。
        #   只拿按钮举例会**假绿**（按钮本来就不在搬运范围内 ✓ 本轮反向验证就是这么假绿过 ✓）。
        ck = QCheckBox("白天不挂机")
        ck.setToolTip("勾上后只在白天的时段执行")
        holder2 = QWidget()
        _h = QHBoxLayout(holder2)
        _h.addWidget(btn)
        _h.addWidget(ck)
        _t2, _l2 = theme.finish_window(holder2)
        check(btn.toolTip() != "" and ck.toolTip() != "" and _t2 == 0,
              "没有字段标题的控件被清空了提示 ⇒ 提示直接弄丢 ✗（该原样留着 ✓）：%r / %r"
              % (btn.toolTip(), ck.toolTip()))
        holder2.deleteLater()
    finally:
        holder.deleteLater()

    # ⑨ ⚠⚠ **"普通 HBox 行"和"嵌套布局当字段"这两条路必须也走得通** ——
    #   2026-10-04 现场：整个工作台**卡死在启动里** ✗（窗口永远不出来、进程活着、没有异常 ✓
    #   用户报的就是"现在数据工作台打不开了"✓）。根因在 `theme._row_title_of` 的
    #   `while`：控件在**普通 HBox 行**里时 `node.parentWidget().layout()` 返回的还是
    #   **同一个布局** ⇒ 原地打转、UI 线程死锁 ✗。
    #   ⚠ 当时之所以没抓着，是因为本用例只造了 **QFormLayout 行** ✗（那条路在循环里就
    #     `return` 了 ✓）⇒ ① 这次把三种行都钉上；② "防环"没法用断言直接测（卡死不是异常 ✗）
    #     ⇒ 用**源码钉**兜住（见下面 ⑨b ✓）。
    from PyQt5.QtWidgets import QComboBox, QLabel

    # ⑨a-1 **普通 HBox 行**（没有 QFormLayout ✓）—— 原来就是这里死循环 ✗
    hbox_w = QWidget()
    hbox = QHBoxLayout(hbox_w)
    hbox_lab = QLabel("筛选")
    hbox.addWidget(hbox_lab)
    hbox_cb = QComboBox()
    hbox_cb.setToolTip("输入 id / 名称缩小范围")
    hbox.addWidget(hbox_cb)
    try:
        check(theme._row_title_of(hbox_cb) is hbox_lab,
              "HBox 行认不出标题（该认「筛选」✓）")
        check(theme.move_tips_to_titles(hbox_w) == 1 and hbox_cb.toolTip() == "",
              "HBox 行没把提示搬到标题上 ✗")
        check(hbox_lab.toolTip() == "输入 id / 名称缩小范围",
              "HBox 行标题上没拿到提示 ✗：%r" % (hbox_lab.toolTip(),))
    finally:
        hbox_w.deleteLater()

    # ⑨a-2 **嵌套布局当字段**（`addRow("标题", 行布局)` ✓ 路线识别页那种）
    nest_w = QWidget()
    nform = QFormLayout(nest_w)
    nrow = QHBoxLayout()
    ncb = QComboBox()
    ncb.setToolTip("嵌套行里的控件")
    nrow.addWidget(ncb)
    nrow.addWidget(QLabel("按钮"))
    nform.addRow("嵌套标题", nrow)
    try:
        _lab = theme._row_title_of(ncb)
        check(_lab is not None and _lab.text() == "嵌套标题",
              "嵌套布局当字段时认不出标题（提示会留在控件上 ✗）：%r"
              % (_lab.text() if _lab is not None else None,))
        check(theme.move_tips_to_titles(nest_w) == 1 and ncb.toolTip() == "",
              "嵌套行没把提示搬到标题上 ✗")
    finally:
        nest_w.deleteLater()

    # ⑨b ⚠ 防环**必须**在（卡死不是异常 ⇒ 只能钉源码 ✓）：见过的不再走 + 层数上限 ✓
    _t = (ROOT / "gui" / "theme.py").read_text(encoding="utf-8")
    _fn = _t[_t.index("def _row_title_of("):]
    _fn = _fn[:_fn.index("\ndef ") if "\ndef " in _fn else len(_fn)]
    check("id(node) in seen" in _fn and "for _ in range(24)" in _fn,
          "`_row_title_of` 的防环保险没了 ⇒ 普通 HBox 行会把界面**卡死** ✗"
          "（2026-10-04 就是这么打不开工作台的 ✓）")

    # ⑥⑦ 源码钉 + 规范同步
    _src = (ROOT / "gui" / "theme.py").read_text(encoding="utf-8")
    check("def enable_label_copy(" in _src, "theme 里没有 `enable_label_copy` ✗")
    _bw = _src[_src.index("def bind_window_state(dlg, key):"):]
    _bw = _bw[:_bw.index("\ndef ") if "\ndef " in _bw else len(_bw)]
    check("finish_window(" in _bw,
          "`bind_window_state`（弹窗收尾）里没调 `finish_window` ⇒ 弹窗的标签选不中 ✗")
    _fw = _src[_src.index("def finish_window("):]
    _fw = _fw[:_fw.index("\ndef ") if "\ndef " in _fw else len(_fw)]
    check("enable_label_copy(" in _fw and "move_tips_to_titles(" in _fw,
          "`finish_window` 没把两件事都做全（搬运 + 可复制 ✓ 光写函数不接上 = 没修 ✗）")
    _spec = (ROOT / "docs" / "UI规范.md").read_text(encoding="utf-8")
    check("提示挂在「字段标题」上" in _spec, "UI规范 §6 没写「提示挂字段标题」那条 ✗")
    check("必须能选中复制" in _spec, "UI规范 §6 没写「标题/说明要能复制」那条 ✗")
    # ⚠ 收尾必须**三条路都有**（少一条就有一类界面没效果 ✓ 用户实测正是漏了页签那条 ✓）
    _mw = (ROOT / "gui" / "main_window.py").read_text(encoding="utf-8")
    check("theme.finish_window(self)" in _mw,
          "主窗口建完没调 `finish_window` ⇒ **页签面板**的提示永远搬不动、标题永远复制不了 ✗")
    check("theme.finish_window(widget)" in _mw,
          "新开的页签（懒建那些）没调 `finish_window` ✗")
    _tools = (ROOT / "tools" / "check_ui.py").read_text(encoding="utf-8")
    # ⚠ 2026-10-04 改口径：**"按文件盘点提示"那条已撤** ✗ —— 提示现在由收尾**自动搬** ✓
    #   再按文件报处数就永远是错的（它扫的是源码 ✗ 而搬运发生在运行时 ✓）；
    #   检查改成钉"收尾机制 + 三条路都在" ✓（见 `check_window_labels_copyable` 里那几条 ✓）。
    check("check_window_labels_copyable" in _tools,
          "check_ui 里没有收尾检查（规则不进检查 ⇒ 下一轮又漂回去 ✗）")


def t_icon_grid_backpack():
    """⭐⭐⭐ 掉落物要用**背包窗格**展示、鼠标指上去**弹信息窗**（用户 2026-10-04 ✓ 原话：
    "自动标注卡片上，要标注的掉落物能不能以背包窗格的形式展示项目，鼠标指着的时候显示信息
    弹窗（MapleNecrocer 应该有相关实现）"）。

    参考实现规格（本机 MapleNecrocer 的 `CharaSimControl\` ✓ 用户指的路 ✓ 照抄别发明 ✗）：
      · 格子 `AfrmItem.cs:624` ⇒ 一格 **33×33** 命中区；
      · 信息窗 `ItemTooltipRender.cs` ⇒ 宽 **290**、图标格 **68×68**（图标放大画进去 ✓）。

    钉六件：
      ① 是 **IconMode 网格**（不是一行行列表 ✗）、格子 37 / 图标 33（照那两处规格 ✓）；
      ② 每格**有图标**、且格子里**存着 id** ✓；
      ③ ⭐ **真鼠标移上去**（`QTest.mouseMove` ✓）⇒ 信息窗**真弹出来**、里面有 **名称 + id** ✓；
         移到**空白处** ⇒ **收起来** ✓（⚠ 只挂 `itemEntered` 不够：移到格子之间的空白它**不发**
         消息 ✗ 信息窗会一直挂着 ✗ —— 这条正是离屏实测抓出来的 ✓）；
      ④ 没名字的项 ⇒ 弹窗写明「（WZ 里没有名字）」+ 怎么补 ✓；
      ⑤ 图库里没图的项 ⇒ 弹窗明说"没有图标"（不许静默 ✗）；
      ⑥ 卡片那块用的**就是它**（源码钉 ✓ 且**没有** `lst_drop` 残留 ✗ —— 换控件时留一个旧列表
         就是"两个真相" ✓）。
    """
    import shutil

    from PyQt5.QtCore import QPoint, Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication, QListWidget

    from core import wzexport
    from gui.icon_grid import IconGrid, tip_lines

    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")

    tmp = _tmpdir()
    lib = _Lib(tmp, drops=(("04030001", "金币"), ("09000000", "")),
               meta={"04030001": {"category": "Consume", "frames": 1},
                     "09000000": {"category": "Special", "frames": 4}})
    try:
        g = IconGrid()
        g.resize(230, 170)
        g.set_entries(wzexport.list_sprite_drops())
        g.show()
        app.processEvents()

        # ①② 网格 + 图标 + id
        check(g.viewMode() == QListWidget.IconMode,
              "掉落物不是**背包格子**（还在一行行列表里 ✗）：%r" % (g.viewMode(),))
        check(g.gridSize().width() == 37 and g.iconSize().width() == 33,
              "格子尺寸不是照 MapleNecrocer 那份（图标 33 / 格子 37 ✓）：%dx%d / %dx%d"
              % (g.gridSize().width(), g.gridSize().height(),
                 g.iconSize().width(), g.iconSize().height()))
        check(g.count() == 2, "格子数不对：%d" % g.count())
        check(g.entry_at(0)["id"] == "04030001",
              "格子里存的条目不对：%r" % (g.entry_at(0),))
        check(str(g.item(0).data(Qt.UserRole)) == "04030001",
              "格子里没把 id 存进 `UserRole`（「移出选中」就没法按 id 找 ✗）")
        check(not g.item(0).icon().isNull(), "格子里没有图标（背包窗格要的就是图标 ✗）")

        # ③ 真悬停：指到格子 ⇒ 弹；移到空白 ⇒ 收
        r = g.visualItemRect(g.item(0))
        QTest.mouseMove(g.viewport(), r.center())
        app.processEvents()
        check(g.tip_visible(), "鼠标指到格子上没弹信息窗（用户要的就是这个 ✗）")
        txt = g.tip_text()
        check("金币" in txt and "04030001" in txt,
              "信息窗里没有「名称 + id」：%r" % (txt,))
        QTest.mouseMove(g.viewport(),
                        g.viewport().rect().bottomRight() - QPoint(6, 6))
        app.processEvents()
        check(not g.tip_visible(),
              "鼠标移开格子（落到空白）信息窗还挂着 ⇒ 实际用起来会「甩不掉」✗")

        # ④ 没名字那种：如实写 + 怎么补
        #   ⚠ 盯的是**标题行**（`tip_lines(...)[0]` ✓）—— 只 `in` 整段会被"怎么补"那句里的
        #     「WZ 里就没有名字」蒙对 ✗（本轮反向验证里差点假绿 ✓）
        t2_lines = tip_lines({"id": "09000000"})
        t2 = "\n".join(t2_lines)
        check("没有名字" in t2_lines[0] and "drops.json" in t2,
              "没名字的项没如实写清（人会以为读坏了 ✗）：%r" % (t2,))
        # ⑤ 没图那种：明说
        t3 = "\n".join(tip_lines({"id": "04030001", "name": "金币", "has_img": False}))
        check("图库里没有" in t3,
              "图库里没有图标却什么都不说（静默 ✗）：%r" % (t3,))
    finally:
        lib.restore()
        shutil.rmtree(tmp, ignore_errors=True)

    # ⑥ 卡片那块用的就是它
    _csrc = (ROOT / "gui" / "steps" / "cards.py").read_text(encoding="utf-8")
    check("self.grid_drop = IconGrid()" in _csrc,
          "卡片「要标注的掉落物」那块没用背包窗格 ✗")
    check("lst_drop" not in _csrc,
          "卡片里还留着旧列表 `lst_drop`（两个控件并存就是「两个真相」✗）")


def t_drop_cache_and_invalidation():
    """⭐⭐⭐ **掉落物那几层缓存：要快，但不许"过期叫不动"**（用户 2026-10-04 ✓ 原话："现在点
    「添加」打开掉落物选择弹窗会非常卡，能解决一下吗？"）。

    量出来的病根（离屏实测 ✓）：老实现
      · `drop_icon_path(id)` = **每个 id 一次 `glob("*.png")` + sorted** ✗ ⇒ **1.3 ms/个** ✓
        （弹窗填 200 格 = 0.28 s ✗；全量 4549 个理论上 6 s ✗）；
      · `list_sprite_drops()` **每次调用都从头建 4549 行** ✗（**0.29 s/次** ✓，弹窗一开调两趟 ✗）；
      · `load_cfg()` 被每个 `*_path()` 调一次、每次都**重开 YAML** ✗（0.5 ms/次 ✓）。
    改后实测：弹窗 **1168 ms ⇒ 173 ms** ✓、打字筛 **262 ms ⇒ 5.6 ms** ✓ —— 口径见 `wzexport`
    里 `drop_icon_index` / `drop_rows` / `load_cfg` 三处的说明 ✓。

    钉六件（**结构 + 行为** ✓ 不用计时阈值 ✗ —— 那玩意在负载高的机器上会假红 ✓）：
      ① `drop_icon_index()` / `drop_rows()` **重复调用返回同一个对象**（= 真缓存 ✓）；
      ② ⭐ **目录变了缓存必须立刻失效**（新建一个掉落物目录 ⇒ 马上认出来 ✓，
         漏了这条 = 新导出的图标要重启才看得见 ✗）；
      ③ ⭐ **`load_cfg` 跟着文件走**：改了 `config/wz.yaml` ⇒ 下一次就读到新值 ✓
         （"改完要重启"是最坏的那种"优化" ✗）；
      ④ 返回的是**拷贝**（界面随手改一下不许把缓存弄脏 ✗）；
      ⑤ **语义没变**：`drop_icon_path` 仍给**排序后第一张** ✓、`meta` 不算 ✓；
      ⑥ `default_drop_ids()` = 图库里 `09` 开头那些金币 ✓（用户第 2 条 ✓）。
    """
    import json
    import shutil

    from core import wzexport

    tmp = _tmpdir()
    spr = Path(tmp) / "drop"
    idx = Path(tmp) / "drops.json"
    old_dir, old_idx = wzexport.drop_sprite_dir, wzexport.drop_index_path
    try:
        (spr / "04000001").mkdir(parents=True)
        (spr / "04000001" / "b.png").write_bytes(b"x")
        (spr / "04000001" / "a.png").write_bytes(b"x")       # 排序后第一张该是 a.png ✓
        (spr / "04000001" / "meta.tsv").write_text("n", encoding="utf-8")
        (spr / "09000000").mkdir()
        (spr / "09000000" / "0.png").write_bytes(b"x")
        idx.write_text(json.dumps([{"id": "04000001", "name": "蝴蝶结"},
                                   {"id": "09000000", "name": ""}],
                                  ensure_ascii=False), encoding="utf-8")
        wzexport.drop_sprite_dir = lambda cfg=None: spr
        wzexport.drop_index_path = lambda cfg=None: idx
        wzexport.clear_drop_caches()

        # ⑤ 语义：排序后的第一张、meta 不算
        _ic = wzexport.drop_icon_path("04000001")
        check(_ic is not None and _ic.name == "a.png",
              "`drop_icon_path` 语义变了（该给**排序后第一张**、`meta` 不算 ✗）：%r" % (_ic,))
        # ① 真缓存
        check(wzexport.drop_icon_index() is wzexport.drop_icon_index(),
              "`drop_icon_index` 每次都在重建（= 没缓存 ⇒ 还是每个 id 一次目录扫描 ✗）")
        check(wzexport.drop_rows() is wzexport.drop_rows(),
              "`drop_rows` 每次都在重建全量行（打字时每敲一下都要 0.29 s ✗）")
        _rows = wzexport.list_sprite_drops()
        check([d["id"] for d in _rows] == ["04000001", "09000000"],
              "候选行不对：%r" % ([d["id"] for d in _rows],))
        check(_rows[0]["has_img"] is True,
              "`has_img` 判错了（该按图标索引 ✓）：%r" % (_rows[0],))
        # ④ 返回拷贝
        _rows[0]["name"] = "改坏了"
        check(wzexport.list_sprite_drops()[0]["name"] == "蝴蝶结",
              "返回的不是拷贝 ⇒ 界面随手改一下就把缓存弄脏了 ✗")

        # ② 目录变了 ⇒ 立刻跟上
        (spr / "09000001").mkdir()
        (spr / "09000001" / "0.png").write_bytes(b"x")
        ids = [d["id"] for d in wzexport.list_sprite_drops()]
        check("09000001" in ids and "09000001" in wzexport.drop_icon_index(),
              "新建的掉落物目录**没被认出来**（缓存过期叫不动 ⇒ 新导出的图标要重启才可见 ✗）："
              "%r" % (ids,))

        # ③ 改配置 ⇒ 下一次 load_cfg 就读到
        #   ⚠⚠⚠ **只能写"临时 CONFIG_PATH"，绝不许写真实的 `config/wz.yaml`** ✗✗✗ ——
        #   第一版这里就是 `wzexport.CONFIG_PATH.write_text(...)` ⇒ **把用户的真实配置整个
        #   覆盖成了那个临时目录**（用户 2026-10-04 现场："drop 标注的背包窗格这些项目就报错了"
        #   ✗ —— 弹窗里那句"图标目录 …\AppData\Local\Temp\drops_xxxx\drop"就是它 ✓）。
        #   那份文件**没进 git** ✗，只好从历史提交（`d0d9922^`）里捞回来 ✓ ——
        #   "自检绝不写用户的配置"这条纪律（`selftest_decision` 里对 `DecisionSettings.save`
        #   也做了同样的事 ✓）这里被我违反了一次 ✗，所以下面**再加一道当场校验** ✓。
        _real_cfg = wzexport.CONFIG_PATH
        _real_bytes = _real_cfg.read_bytes() if _real_cfg.exists() else None
        _cfg_tmp = Path(tmp) / "wz.yaml"
        wzexport.CONFIG_PATH = _cfg_tmp
        try:
            _before = dict(wzexport.load_cfg())
            _cfg_tmp.write_text("drop_sprite_dir: %s\n" % spr.as_posix(), encoding="utf-8")
            _after = wzexport.load_cfg()
            check(_after != _before and "drop_sprite_dir" in _after,
                  "改完配置文件后 `load_cfg()` 还是老值（那就变成「改完要重启」✗）")
        finally:
            wzexport.CONFIG_PATH = _real_cfg
            wzexport._CFG_CACHE.clear()
        # ⭐⭐ **当场钉住：真实配置一个字节都不许动** ✓（这条就是上面那次事故的护栏 ✓）
        check((_real_cfg.read_bytes() if _real_cfg.exists() else None) == _real_bytes,
              "自检动了**真实**的 `config/wz.yaml` ✗（2026-10-04 就是它把用户配置覆盖掉的 ✓）")

        # ⑦ 磁盘**实际帧数**（用户 2026-10-04 ✓ 手工删过 `icon_3.png` ⇒ 清单那份会过期 ✗）
        check(wzexport.drop_frame_counts().get("04000001") == 2,
              "`drop_frame_counts` 没数对（那个目录有 2 张非 meta 的 png ✓ `meta.tsv` 不算 ✓）：%r"
              % (wzexport.drop_frame_counts(),))
        _r0 = [d for d in wzexport.list_sprite_drops() if d["id"] == "04000001"][0]
        check(_r0.get("frames_disk") == 2,
              "候选行里没带上「磁盘上实际几帧」（人核对「我删干净没有」就看不到 ✗）：%r" % (_r0,))
        from gui.icon_grid import tip_lines as _tip

        _t = "\\n".join(_tip({"id": "04000001", "name": "蝴蝶结", "frames": 4,
                              "frames_disk": 1, "has_img": True}))
        check("磁盘上" in _t and "4 帧" in _t,
              "手工删过帧之后信息窗还照着清单念（那是假话 ✗）：%r" % (_t,))
        _t2 = "\\n".join(_tip({"id": "04000001", "name": "蝴蝶结", "frames": 4,
                               "frames_disk": 4, "has_img": True}))
        check("磁盘上" not in _t2 and "4 帧" in _t2,
              "清单与磁盘**一致**时该照老样子写（别多嘴 ✗）：%r" % (_t2,))

        # ⑥ 默认掉落物 = 图库里 09 开头的那些（这里造了 2 个 ⇒ 就认这 2 个 ✓ 口径一致 ✓）
        check(wzexport.default_drop_ids() == ["09000000", "09000001"],
              "默认掉落物没按「图库里 09 开头的」取：%r" % (wzexport.default_drop_ids(),))
    finally:
        wzexport.drop_sprite_dir, wzexport.drop_index_path = old_dir, old_idx
        wzexport.clear_drop_caches()
        shutil.rmtree(tmp, ignore_errors=True)


def t_drop_e2e():
    """⑤ **端到端真跑一遍**（用户 2026-10-04 ✓ 原话："先导出所有的金币 把链路跑通"）。

    链路：`WzProbe dump-items` 导出道具图标+清单 ⇒ `datasets/sprites/drop/<id>/*.png` +
    `datasets/drops.json` ⇒ `label_drops.run_label_drops(mode="template")`
    ⇒ `detect_mobs.run_detect(cls=2)` ⇒ `labels_auto/<帧>.txt` 里写出 **class 2** 的行 ✓。

    做法：把 **图标贴在灰底上造一帧**（位置随便选、带 alpha 混合 ✓），跑真匹配，钉三件：
      ① 能搜到那个 id（清单位 + 图标位都对 ✓）；
      ② 真有检出（把它贴上去都检不到 ⇒ 说明模板/闸/尺度哪一环断了 ✗）；
      ③ 写出来的**类别号是 2**（drop ✓ 不是 1 怪物 ✗ —— 这条错了训练集就全歪 ✓）。
    ⚠ 本机没导过图库（`datasets/sprites/drop` 空）⇒ **明确跳过**，不假装通过 ✗。
    ⚠ 用 `workers=1`：这一步是**验证**不是生产，别为它开一池子进程（也省得慢 ✓）。
    """
    import shutil
    import tempfile

    import numpy as np

    from core.context import TaskContext
    from core.imgio import imread, imwrite
    from core import wzexport
    from gui.project import Project
    from tools import label_drops as LD

    ok, why = wzexport.drop_library_state()
    if not ok:
        print("      （本机还没有道具图库 —— 跳过端到端：%s ✓）"
              % why.splitlines()[0])
        return
    cands = wzexport.list_sprite_drops("0900")
    if not cands:
        print("      （图库里没有 0900* 的金币 —— 跳过端到端 ✓）")
        return
    drop = cands[0]
    icon = imread(wzexport.drop_icon_path(drop["id"]), -1)
    check(icon is not None and icon.ndim == 3 and icon.shape[2] == 4,
          "取出来的图标不是带 alpha 的 4 通道图（模板要它做 mask ✗）：%r" % (icon,))

    ih, iw = icon.shape[:2]
    frame = np.full((480, 640, 3), 40, np.uint8)
    y0, x0 = 300, 300
    al = icon[:, :, 3:4].astype(np.float32) / 255.0
    roi = frame[y0:y0 + ih, x0:x0 + iw].astype(np.float32)
    frame[y0:y0 + ih, x0:x0 + iw] = (roi * (1 - al)
                                     + icon[:, :, :3].astype(np.float32) * al).astype(np.uint8)

    tmp = _tmpdir()
    try:
        proj = Project.create(tmp / "proj", name="掉落物用例", map_id="105090600")
        imwrite(proj.frames / "frame_00001.png", frame, quality=95)
        res = LD.run_label_drops({
            "mode": "template",
            "drops": [{"id": drop["id"], "name": drop.get("name") or ""}],
            "frames": str(proj.frames),
            "out": str(proj.dir_of("labels_auto")),
            "vis_dir": str(proj.vis) + "_drop",
            "thresh": 0.90,
            "vis": True,
            "workers": 1,          # 验证用：单进程 ✓
        }, TaskContext())
        check(int(res.get("detections", 0)) >= 1,
              "把图标原样贴上去都检不到 ⇒ 链路上有一环断了（模板/小图闸/尺度 ✓）：%r" % (res,))
        lab = proj.dir_of("labels_auto") / "frame_00001.txt"
        check(lab.exists(), "没写出标注文件：%s" % lab)
        cls = [ln.split()[0] for ln in lab.read_text(encoding="utf-8").splitlines()
               if ln.strip()]
        check(cls and all(c == "2" for c in cls),
              "写出来的类别号不是 **class 2（drop）**（写成 1 就是标到怪物头上 ✗）：%r" % (cls,))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_visible_box_adapts():
    """⭐⭐⭐ 模板匹配的框要能**自适应到可见部分**（用户 2026-10-05 ✓ 原话："现在模板匹配的自动标注框
    不管实际漏出目标的部分有多少，都用完整贴图大小的框，这个能自适应识别出的框大小吗？"）。

    做法（合成 + 真跑管线 ✓ 确定可复现 ✓）：造一个**有边缘可用**的精灵（固定种子的随机纹理 ✓
    周期纹理不行 ✗ —— 会匹配到一堆等价位置 ✓），贴到灰底上 ⇒ 跑真的 `run_detect`（`workers=1` ✓），
    **逐帧比框**（YOLO txt 里就是归一化宽高 ✓）：

      ⛔① **默认关** ⇒ 三帧的框都 = **完整贴图尺寸**（= 老行为**一字不变** ✓ 这条最关键 ✓）；
      ② 开 `visible=True` 且**完整贴上** ⇒ 框**尺寸一模一样**（没被挡就不许收 ✗ ——
         抗锯齿啃掉一两像素就收，等于把"没遮挡"也改小了 ✓）；
      ③ 开 `visible=True` 且**盖掉右半**（盖上另一种纹理 ⇒ 边缘方向/强度都对不上 ✓）⇒
         框**收到左半** ✓（宽明显变小 ✓、**左边界不动** ✓）；
      ④ 盖到**只剩一小条** ⇒ 触发下限 ⇒ **回退完整尺寸** ✓（宁可不收 ✗ 别吐荒谬小框 ✓）。

    另钉两件"故意保守"的天花板：**纯色精灵**（没有边缘可用 ✓）一律回退完整 ✓；
    判据必须用**最大连通块** ✗（用"全部相符像素"求包围盒会被零散几点撑回全尺寸 ✓）。
    """
    import pathlib
    import shutil
    import tempfile

    import cv2
    import numpy as np

    from core.context import TaskContext
    from core.imgio import imread, imwrite
    from tools import detect_mobs as DM

    MID = "9000000"
    N = 24
    rng = np.random.RandomState(7)
    tex = rng.randint(40, 215, size=(N, N)).astype(np.uint8)          # 精灵：唯一纹理 ✓
    occl = np.random.RandomState(8).randint(40, 215, size=(N, N)).astype(np.uint8)  # 挡它的东西 ✓
    flat = np.full((N, N), 128, np.uint8)                            # 纯色精灵（天花板 ✓）

    def _bgra(g):
        bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        return np.dstack([bgr, np.full((N, N), 255, np.uint8)])

    X0, Y0 = 20, 20
    base = np.full((80, 80, 3), 30, np.uint8)
    f_ok = base.copy()
    f_ok[Y0:Y0 + N, X0:X0 + N] = cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)
    # ⚠ 只盖 **25%**（6/24 ✓）：带掩码的归一化分数是**按掩码**算的 ✓ ⇒ 被盖掉一半时分数会掉到
    #   阈值下 ⇒ **整个检不到** ✓（那是"漏检"，不是"框偏大" ✓ 两件事别混 ✗）——可见框能救的是
    #   **适度遮挡**这种"分够、但框照旧包满"的情形 ✓ 正是用户说的那个 ✓。
    COVER = 6
    f_cov = f_ok.copy()
    f_cov[Y0:Y0 + N, X0 + N - COVER:X0 + N] = \
        cv2.cvtColor(occl, cv2.COLOR_GRAY2BGR)[:, :COVER]

    def _run(tag, visible, sprite, frames):
        spr = tmp / ("spr_" + tag)
        (spr / MID).mkdir(parents=True, exist_ok=True)
        imwrite(spr / MID / "stand_0.png", sprite)
        fr = tmp / ("fr_" + tag)
        fr.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            imwrite(fr / ("frame_0000%d.png" % (i + 1)), f)
        out = tmp / ("out_" + tag)
        p = {"mobs": [MID], "sprites": str(spr), "frames": str(fr), "out": str(out),
             "vis": False, "workers": 1, "scale": 1.0, "thresh": 0.60, "max_peaks": 2,
             # ⚠ 区分度放 0：合成帧是**整幅灰底** ⇒ 响应图特别"平" ✓ ⇒ 「峰值要显著高于背景」那道闸
             #   会把真峰否掉 ✓（实测：基线 0 检出、放开它 3 检出 ✓）—— 用例要验的是**框的形状** ✓
             #   不是那道闸 ✓（真实帧有纹理，不这样 ✓）。
             "min_energy_ratio": 0.2, "min_distinct": 0.0, "coarse": False}
        if visible:
            p["visible"] = True
        res = DM.run_detect(p, TaskContext())
        boxes = {}
        for t in sorted(out.glob("*.txt")):
            boxes[t.stem] = [ln.split() for ln in t.read_text(encoding="utf-8").splitlines()
                             if ln.strip()]
        return res, boxes

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="visbox_"))
    try:
        want_w = N / 80.0

        # ⛔① 默认关 ⇒ 两帧都写**完整贴图尺寸**（老行为一字不变 ✓）
        res, off = _run("off", False, _bgra(tex), (f_ok, f_cov))
        check(int(res.get("detections", 0)) >= 2,
              "合成帧都检不到 ⇒ 这一条等于没测（模板/闸/尺度哪一环断了 ✗）：%r" % (res,))
        for stem, lines in off.items():
            for ln in lines:
                check(abs(float(ln[3]) - want_w) < 0.005 and abs(float(ln[4]) - want_w) < 0.005,
                      "**默认关**时框就该是完整贴图尺寸（老行为一字不变 ✓）：%s w=%s h=%s"
                      % (stem, ln[3], ln[4]))

        # ② 开了之后：没被挡的帧**一字不变**；③ 盖掉 25% 的帧要**收小**且左边界不动 ✓
        _res, on = _run("on", True, _bgra(tex), (f_ok, f_cov))
        l_ok = on.get("frame_00001") or []
        l_cov = on.get("frame_00002") or []
        check(bool(l_ok) and bool(l_cov), "开了之后有帧一个框都没有 ✗：%r" % (on,))

        check(abs(float(l_ok[0][3]) - want_w) < 0.02 and abs(float(l_ok[0][4]) - want_w) < 0.02,
              "**没被挡**的帧框也被收了 ✗（那就不该开这个功能）：w=%s h=%s"
              % (l_ok[0][3], l_ok[0][4]))

        check(float(l_cov[0][3]) < want_w * 0.85,
              "**盖掉 25%%** 之后框没收小（宽还是 %s ≈ 完整 %s ✗）⇒ 可见框没生效"
              % (l_cov[0][3], want_w))
        check(float(l_cov[0][4]) > want_w * 0.9,
              "只盖了右边 ⇒ **高度不该收**（%s ✗）：见「最大连通块」那条 ✓" % l_cov[0][4])
        check(float(l_cov[0][1]) < (X0 + N / 2.0) / 80.0,
              "收了之后**中心该往左移**（可见部分在左 ✓）：cx=%s" % l_cov[0][1])

        # ---- 下限 / 天花板 / 出界：直接调 `visible_box`（不用等检出 ✓ 那几种本来也检不到 ✓）----
        # ⚠ 实现 2026-10-06 搬去 `perception/visible_box.py` 了（玩家那条内核也要用 ✓
        #   而 `perception` 不能反向 import `tools` ✗）⇒ **这里跟着搬** ✓（判据一字不改 ✓）。
        from perception.visible_box import visible_box as _visible_box

        tpl_bgr = cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)
        tg = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        tgx = cv2.Sobel(tg, cv2.CV_32F, 1, 0, ksize=3)
        tgy = cv2.Sobel(tg, cv2.CV_32F, 0, 1, ksize=3)
        mask = np.ones((N, N), bool)
        imgf = np.full((80, 80), 30, np.float32)
        imgf[Y0:Y0 + N, X0:X0 + N] = tex.astype(np.float32)

        check(_visible_box(imgf, tg, tgx, tgy, mask, X0, Y0) == (X0, Y0, N, N),
              "完整可见 ⇒ 一个字都不许动（这条不对，用户会在没被挡的帧上看到框变小 ✗）")

        img2 = imgf.copy()                      # 只剩 4px 露着 ⇒ 下限生效 ⇒ 回退完整 ✓
        img2[Y0:Y0 + N, X0 + 4:X0 + N] = occl[:, :N - 4].astype(np.float32)
        check(_visible_box(img2, tg, tgx, tgy, mask, X0, Y0) == (X0, Y0, N, N),
              "只剩一小条时该**回退完整尺寸**（宁可不收 ✓ 下限没生效 ⇒ 会吐出荒谬小框 ✗）")

        fg = np.full((N, N), 128, np.float32)   # 纯色精灵：没有边缘可用 ⇒ 判不了 ⇒ 完整 ✓
        check(_visible_box(img2, fg, np.zeros((N, N), np.float32),
                           np.zeros((N, N), np.float32), mask, X0, Y0) == (X0, Y0, N, N),
              "纯色精灵判不了 ⇒ 该回退完整尺寸 ✓（这是**故意**保守 ✓）")

        check(_visible_box(imgf, tg, tgx, tgy, mask, 999, 999) == (999, 999, N, N),
              "峰值出界时该原样回全尺寸、且**不抛** ✓（这只是把框收小一点 ✓ 坏了就别收 ✓）")

        # 判据必须是**最大连通块**（见 `visible_box` ✓）——拿"全部相符像素"求包围盒会被零散几点撑满 ✗
        # ⚠ 查的是**实现所在的文件**（`perception/visible_box.py` ✓ 2026-10-06 搬过去的 ✓）
        import ast as _ast
        _tree = _ast.parse((ROOT / "perception" / "visible_box.py").read_text(encoding="utf-8"))
        _fv = next(n for n in _ast.walk(_tree)
                   if isinstance(n, _ast.FunctionDef) and n.name == "visible_box")
        _names = {getattr(getattr(c, "func", None), "attr", None) for c in _ast.walk(_fv)
                  if isinstance(c, _ast.Call)}
        check("connectedComponentsWithStats" in _names,
              "`_visible_box` 没用连通块（拿全部相符像素求包围盒 ⇒ 零散几点就把框撑回全尺寸 ✗）")
        check(imread(str(tmp / "fr_off" / "frame_00001.png")) is not None,
              "合成帧没落盘（夹具自己坏了 ✗）")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_visible_box_player_segment():
    """⭐⭐⭐ **玩家那一段也吃「框到可见部分」**（用户 2026-10-06 ✓ 原话："**所有的匹配都需要
    「框到可见部分」参数**"）。

    为什么要单拎一条：怪物 / 宠物 / 掉落三条走的是 `tools/detect_mobs.work` ✓，而**玩家**走的是
    **另一套内核**（`perception/player_locator.PlayerLocator` ✓ 它自己 `cv2.matchTemplate` ✓）
    ⇒ 这条链上**三段**原本**全断** ✗（卡片没传 → `run_detect_player` 没转发 →
    `PlayerLocator` 连这个能力都没有 ✓ 2026-10-06 三段一起补 ✓）⇒ 三段各钉一件：

      ① **真行为**：合成帧（精灵 + 盖掉右边 25% ✓）⇒ `visible=False` 给**完整尺寸** ✓；
         `visible=True` 给**收窄**的框且**中心左移** ✓；**没被盖**的帧开了也**一字不变** ✓
         —— 判据是**同一份实现**（`perception/visible_box.py` ✓ 怪物那条靠
         `t_visible_box_adapts` 钉 ✓ 两条走同一份 ✓）；
      ② **中间层真的转发**（`run_detect_player` 的 cfg ⇒ 构造 `PlayerLocator` 时那三个形参 ✓）：
         ⚠ 这条链上**两处**构造 locator（子进程 `init_worker` + 主进程那个预检 ✓）
         **都得给** ✗ —— 只改一处 = 没接上（本仓库的老毛病 ✓ 实测踩过 ✓）；
      ③ **卡片真的传下去**（`LabelCard.make_task` 出来的 `detect` 里带 `visible` ✓）——
         "界面能选、管线收不到"已经栽过一次 ✓。

    反向验证：把卡片那句 `**self._visible_params(sec)` 拆掉 / 把 locator 里那段收框拆掉 ⇒
    各自当场红 ✓（见用例末尾注释里的两条命令 ✓）。
    """
    import shutil
    import tempfile

    import cv2
    import numpy as np

    from core.context import TaskContext
    from core.imgio import imwrite
    from perception.player_locator import PlayerLocator
    from tools import detect_player as DP
    from perception import player_locator as PL

    MID = "珠缨"
    N = 24
    tex = np.random.RandomState(11).randint(40, 215, size=(N, N)).astype(np.uint8)
    occl = np.random.RandomState(12).randint(40, 215, size=(N, N)).astype(np.uint8)

    def _bgra(g):
        bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        return np.dstack([bgr, np.full((N, N), 255, np.uint8)])

    X0, Y0, W, H = 20, 20, 80, 80
    base = np.full((H, W, 3), 30, np.uint8)
    f_ok = base.copy()
    f_ok[Y0:Y0 + N, X0:X0 + N] = cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)
    # ⚠ 只盖 **25%**：带掩码的归一化分数是**按掩码**算的 ⇒ 盖掉一半时分数会掉到阈值下
    #   ⇒ **整个定位不到** ✗（那是"漏检"、不是"框偏大" ✓ 两件事别混 ✗ —— 与
    #   `t_visible_box_adapts` 里那条说明同一个口径 ✓）。
    COVER = 6
    f_cov = f_ok.copy()
    f_cov[Y0:Y0 + N, X0 + N - COVER:X0 + N] = \
        cv2.cvtColor(occl, cv2.COLOR_GRAY2BGR)[:, :COVER]

    tmp = Path(tempfile.mkdtemp(prefix="visplayer_"))
    try:
        # ---- ① 真行为：同一个内核，开关一下框会不会收 ----
        root = tmp / "sprites"
        (root / MID).mkdir(parents=True, exist_ok=True)
        imwrite(root / MID / ("%s-stand-0.png" % MID), _bgra(tex))

        def _loc(visible):
            return PlayerLocator(MID, root=str(root), scale=1.0, threshold=0.60,
                                 max_peaks=1, visible=visible)

        off, on = _loc(False), _loc(True)
        r_off_ok, r_off_cov = off.locate(f_ok), off.locate(f_cov)
        r_on_ok, r_on_cov = on.locate(f_ok), on.locate(f_cov)
        for _nm, _r in (("关着/没盖", r_off_ok), ("关着/盖了", r_off_cov),
                        ("开了/没盖", r_on_ok), ("开了/盖了", r_on_cov)):
            check(_r is not None, "%s：压根没定位到（夹具坏了 / 阈值太高 ✗）" % _nm)

        check(abs(r_off_ok[2] - N) < 0.6 and abs(r_off_cov[2] - N) < 0.6,
              "**默认关**时玩家框就该是完整贴图尺寸（老行为一字不变 ✗）：%s / %s"
              % (r_off_ok[2], r_off_cov[2]))
        check(abs(r_on_ok[2] - N) < 1.0 and abs(r_on_ok[3] - N) < 1.0,
              "**没被盖**的帧开了也把框收了（那就不该开这个功能 ✗）：%s x %s"
              % (r_on_ok[2], r_on_ok[3]))
        check(r_on_cov[2] < N * 0.85,
              "**盖掉 25%%** 之后玩家框没收小（宽还是 %s ≈ 完整 %s ✗）⇒ 玩家这段的可见框没生效"
              % (r_on_cov[2], N))
        check(r_on_cov[3] > N * 0.9,
              "只盖了右边 ⇒ **高度不该收**（%s ✗）：见「最大连通块」那条 ✓" % (r_on_cov[3],))
        check(r_on_cov[0] < r_off_cov[0] - 1.0,
              "收了之后**中心该往左移**（可见部分在左 ✓）：%s vs %s"
              % (r_on_cov[0], r_off_cov[0]))

        # ---- ② 中间层：`run_detect_player` 的两个 locator 构造点都得拿到 ----
        calls = []

        class _FakeLoc:
            def __init__(self, pid, **kw):
                calls.append(dict(kw, player_id=pid))
                self.template_count = 3
                self.stopped = False

            def locate(self, bgr, search_rect=None, downscale=None):
                return None

        fr = tmp / "fr"
        fr.mkdir(parents=True, exist_ok=True)
        imwrite(fr / "frame_00001.png", f_ok)
        _real = PL.PlayerLocator
        PL.PlayerLocator = _FakeLoc
        try:
            DP.run_detect_player({"frames": str(fr), "out": str(tmp / "out"),
                                  "player_id": MID, "workers": 1, "vis": False,
                                  "visible": True, "visible_tol": 30.0,
                                  "visible_keep": 0.3}, TaskContext())
        finally:
            PL.PlayerLocator = _real
        check(bool(calls), "`run_detect_player` 一个 locator 都没建（夹具坏了 ✗）")
        check(all(c.get("visible") is True for c in calls),
              "只有一处 locator 拿到「框到可见部分」（另一处漏了 ⇒ `workers<=1` / 子进程那条"
              "等于没接 ✗）：%r" % ([c.get("visible") for c in calls],))
        check(float(calls[-1].get("visible_tol")) == 30.0
              and float(calls[-1].get("visible_keep")) == 0.3,
              "两个阈值没原样传下去（改了也不生效 ✗）：%r" % (calls[-1],))

        # ---- ③ 卡片：`make_task` 出来的玩家任务要带上它 ----
        from PyQt5.QtWidgets import QApplication
        from gui.project import Project
        from gui.steps.cards import LabelCard

        _app = QApplication.instance() or QApplication([])
        check(_app is not None, "建不起 QApplication")
        proj = Project.create(tmp / "proj", name="用例", map_id="105090600")
        (proj.frames).mkdir(parents=True, exist_ok=True)
        c = LabelCard()
        c.bind(proj)
        c._label_target = "player"
        proj.sec("label")["visible"] = True        # `sec()` 给的是引用 ✓ `make_task` 读的就是它 ✓
        _fn, pr = c.make_task(proj)
        check(pr.get("detect", {}).get("visible") is True,
              "③ 卡片没把「框到可见部分」传进玩家任务（界面上勾了、管线收不到 ✗）：%r"
              % (pr.get("detect", {}).get("visible"),))
        proj.sec("label")["visible"] = False
        _fn, pr = c.make_task(proj)
        check(pr.get("detect", {}).get("visible") is False,
              "③ 没勾的时候不该是 True（默认关是老行为 ✓）：%r"
              % (pr.get("detect", {}).get("visible"),))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_second_run_protects_and_reports():
    """⭐⭐ 跑第二趟**不许把上一波的框悄悄清空** + 报告**按类别分开念**（用户 2026-10-05 ✓）。

    起因（用户现场 ✓ 原话："模板匹配怪物一波之后往后每次提示：检出 0 框 … 每帧框数: 中位 4"✗）：
    两件事**都对**，只是口径不同 ✓ ——
      · `检出 N 框` = **本次这一类**新收下的框 ✓；
      · `每帧框数 / 空帧 / 框尺寸` 扫的是**整个目录**（含别的类、含上一波 ✓ `collect_stats`
        不分类 ✓）—— 实测那个项目 793 框 = 玩家 133 + 怪 288 + 宠物 372 ✓。
    ⚠ 底下还藏着一个**真坑**：写盘时"只保留别的类、本类旧框**无条件覆盖**"✗ ⇒ 本次 0 检出
      ⇒ 那一波标出来的框**被悄悄清空** ✓（实测那个项目 59 帧只剩玩家/宠物框 ✓）。

    钉五件：
      ① 目录里有**别的类**的框时，报告能**分类别**念出来（`stats["per_class"]` ✓）；
      ② 第二趟**本类 0 检出** ⇒ 原来本类的框**原样保留** ✓，`protected` 计数 = 1 ✓；
      ③ 日志里**明说**"保留了旧框"（"没检出" ≠ "被清空" ✓ 这两件事必须能分开说 ✓）；
      ④ **检出非空时照旧覆盖** ✓（那才是"重标一遍"的正常语义 ⇒ 旧框必须消失 ✓ 不然新旧混一起 ✓）；
      ⑤ 别的类**一根都不许动** ✓（标怪时不许碰玩家/宠物框 ✓）。
    """
    import pathlib
    import shutil
    import tempfile

    import cv2
    import numpy as np

    from core.imgio import imwrite
    from tools import detect_mobs as DM

    MID = "9000001"
    N = 24
    tex = np.random.RandomState(21).randint(40, 215, size=(N, N)).astype(np.uint8)
    bgra = np.dstack([cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR), np.full((N, N), 255, np.uint8)])
    X0 = Y0 = 20
    with_icon = np.full((80, 80, 3), 30, np.uint8)
    with_icon[Y0:Y0 + N, X0:X0 + N] = cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)
    bg_only = np.full((80, 80, 3), 30, np.uint8)

    class _Log:
        """收日志的小替身（`run_detect` 只用到 log / progress / canceled ✓）。"""
        def __init__(self):
            self.lines = []

        def log(self, msg, level="info"):
            self.lines.append(str(msg))

        def progress(self, *a, **k):
            pass

        def canceled(self):
            return False

        def text(self):
            return "\n".join(self.lines)

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="proto_"))
    try:
        spr = tmp / "spr"
        (spr / MID).mkdir(parents=True)
        imwrite(spr / MID / "stand_0.png", bgra)
        fr = tmp / "fr"
        fr.mkdir()
        imwrite(fr / "frame_00001.png", with_icon)     # 有目标 ⇒ 第一趟能检出 ✓
        imwrite(fr / "frame_00002.png", bg_only)       # 没目标 ✓
        out = tmp / "out"

        def _run(**kw):
            p = {"mobs": [MID], "sprites": str(spr), "frames": str(fr), "out": str(out),
                 "vis": False, "workers": 1, "scale": 1.0, "thresh": 0.60, "max_peaks": 2,
                 "min_energy_ratio": 0.2, "min_distinct": 0.0, "coarse": False}
            p.update(kw)
            ctx = _Log()
            return DM.run_detect(p, ctx), ctx

        # 先手动放一个**别的类**的框（用户/宠物框那种 ✓）—— 全程都不许被动 ✓
        out.mkdir(parents=True, exist_ok=True)
        (out / "frame_00002.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

        # ---- 第一趟：有目标 ⇒ 真有检出 ----
        res1, _c1 = _run()
        check(int(res1.get("detections", 0)) >= 1,
              "第一趟就该检出（合成帧里有目标 ✓）：%r" % (res1,))
        f1 = (out / "frame_00001.txt").read_text(encoding="utf-8")
        check({ln.split()[0] for ln in f1.splitlines() if ln.strip()} == {"1"},
              "第一趟没写出 class 1 的框：%r" % (f1,))
        check(int(res1["stats"]["per_class"].get("0", 0)) == 1,
              "① 报告没有按类别算（别的类的框也要数进来 ✓）：%r"
              % (res1["stats"].get("per_class"),))

        # ---- 第二趟：把**搜索区域**挪到没目标的那一角 ⇒ 本类 0 检出；旧框必须**原样保留** ----
        #   ⚠ 别用"阈值提到 0.99"来造 0 检出 ✗：合成帧是整幅纯色 ⇒ 会出现**高分假匹配** ✓
        #     （实测 0.99 仍检到 1 个 ✓）⇒ 那样测的就不是"0 检出"了 ✓。
        res2, ctx2 = _run(region="0,0,20,20")
        check(int(res2.get("detections", 0)) == 0, "第二趟本该 0 检出：%r" % (res2,))
        f1b = (out / "frame_00001.txt").read_text(encoding="utf-8")
        check(f1b == f1,
              "② 本类 0 检出时把上一波的框改了/清空了（**这是本轮要修的那个坑** ✗）："
              "%r -> %r" % (f1, f1b))
        check(int(res2.get("protected", 0)) >= 1,
              "② 没把「保住了几帧旧框」报出来（`protected` ✓）：%r" % (res2.get("protected"),))
        check("原样保留" in ctx2.text(),
              "③ 日志没说清「旧框被保留了」⇒ 人只会看到「检出 0 框」、以为被清空 ✗：\n%s"
              % ctx2.text()[-300:])

        # ---- 第三趟：正常阈值 ⇒ 检出非空 ⇒ **照旧覆盖**（旧框要没 ✓ 新旧不许混）----
        (out / "frame_00001.txt").write_text(
            "1 0.111111 0.111111 0.111111 0.111111\n1 0.222222 0.222222 0.111111 0.111111\n",
            encoding="utf-8")            # 两条"旧的假框" ✓
        res3, _c3 = _run()
        f1c = (out / "frame_00001.txt").read_text(encoding="utf-8")
        check(int(res3.get("detections", 0)) >= 1, "第三趟本该有检出：%r" % (res3,))
        check("0.111111" not in f1c and "0.222222" not in f1c,
              "④ 检出非空时**旧框必须被覆盖**（不然新旧混在一起更坏 ✗）：%r" % (f1c,))
        check({ln.split()[0] for ln in f1c.splitlines() if ln.strip()} == {"1"},
              "④ 覆盖之后这一帧该只剩 class 1：%r" % (f1c,))

        # ---- ⑤ 别的类全程一根都没动 ✓（本类怎么写都不许碰它 ✓）----
        f2 = (out / "frame_00002.txt").read_text(encoding="utf-8")
        check("0 0.5 0.5 0.1 0.1" in [ln for ln in f2.splitlines() if ln.strip()],
              "⑤ 别的类（class 0 玩家/宠物框那种）被动过了 ✗：%r" % (f2,))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_card_layout():
    """④ 布局整理（用户 §3："参数、按钮分类布局，要求功能模块划分合理"）—— 源码级钉。

    · 三个分区小标题在（模板匹配参数 / 玩家 / 标注对象 · 掉落物 ✓）；
    · 掉落物那块挂在**卡片根布局最下面**（`self.layout().addWidget(self.drop_box)` ✓
      —— 用户原话就是"卡片最下面" ✓）；
    · 两个按钮的文案就是「模板匹配标注」「YOLO标注」（用户点名的字 ✓）。
    """
    src = (ROOT / "gui" / "steps" / "cards.py").read_text(encoding="utf-8")
    for cap in ("模板匹配参数", "玩家", "标注对象 · 掉落物"):
        check(('_section("%s"' % cap) in src or ('"%s"' % cap) in src,
              "少了分区小标题「%s」（用户 §3 要「功能模块划分合理」✗）" % cap)
    check("self.layout().addWidget(self.drop_box)" in src,
          "掉落物那一块没加在卡片**最下面**（用户 §1 ✗）")
    check('QPushButton("模板匹配标注掉落物")' in src
          and 'QPushButton("YOLO标注掉落物")' in src
          and 'QPushButton("模板匹配标注宠物")' in src
          and 'QPushButton("YOLO标注宠物")' in src,
          "两个按钮的文案不是「模板匹配标注{类}」/「YOLO标注{类}」✗")


def t_icon_grid_fit_rows_hard():
    """②h ⭐⭐ **窗格高度要「钉死」，不只是给下限**（用户 2026-10-04 ✓ 原话："模型训练里的掉落物、
    宠物自动标注类背包窗格没有按选中的项目数量限定行数（背包窗格显示区的高度）"）。

    为什么 `min` 不够（**离屏量过** ✓）：卡片里窗格的 `minimumHeight` 明明是 41（= 一行 ✓），
    可 `height()` **恒为 480** ✗ —— 布局把多余空间全塞给它 ⇒ 显示区一直那么高 ✓。
    ⇒ `fit_rows(hard=True)` **上下限一起钉** ✓；⚠ 关键是 `set_entries` 里那次
    `fit_rows()`（不带 `hard` ✓）**不许把它冲掉** ✗（冲掉就等于没修 ✓）。
    """
    from PyQt5.QtWidgets import QApplication

    from gui.icon_grid import IconGrid

    app = QApplication.instance() or QApplication([])
    g = IconGrid()
    g.resize(400, 40)
    app.processEvents()
    g.set_entries([{"id": "09000000"}])
    app.processEvents()
    g.fit_rows(min_rows=1, hard=True)
    app.processEvents()
    check(g.maximumHeight() == g.minimumHeight() > 0,
          "`hard=True` 没把上限也钉住（布局会继续把显示区撑高 ✗）：min=%r max=%r"
          % (g.minimumHeight(), g.maximumHeight()))
    h1 = g.minimumHeight()

    g.set_entries([{"id": "09000000"}])
    app.processEvents()
    check(g.maximumHeight() == g.minimumHeight() == h1,
          "`set_entries` 把硬限定冲掉了（= 没修 ✗）：min=%r max=%r"
          % (g.minimumHeight(), g.maximumHeight()))

    g.set_entries([{"id": "090%05d" % i} for i in range(40)])
    app.processEvents()
    check(g.minimumHeight() > h1 and g.maximumHeight() == g.minimumHeight(),
          "项多了高度没跟着涨 / 上限又脱开：min=%r max=%r"
          % (g.minimumHeight(), g.maximumHeight()))

    import tempfile
    from pathlib import Path as _P

    from gui.project import Project
    from gui.steps.cards import LabelCard
    tmp = _P(tempfile.mkdtemp(prefix="fitrows_"))
    proj = Project.create(tmp / "p", name="窗格", map_id="105090600")
    c = LabelCard()
    c.bind(proj)
    c.resize(900, 900)
    c.show()
    app.processEvents()
    drops = [{"id": "0%07d" % i, "name": "物%d" % i, "desc": "", "has_img": False}
             for i in range(1, 41)]
    c.grid_drop.set_entries(drops[:1])
    app.processEvents()
    h_1 = c.grid_drop.height()
    c.grid_drop.set_entries(drops)
    app.processEvents()
    h_n = c.grid_drop.height()
    check(c.grid_drop.maximumHeight() == c.grid_drop.minimumHeight() and h_n > h_1,
          "卡片上的**掉落物**窗格没按项数限定高度：1 项 %r → 40 项 %r（max/min=%r/%r）"
          % (h_1, h_n, c.grid_drop.maximumHeight(), c.grid_drop.minimumHeight()))
    c.grid_pet.set_entries([{"id": "5000020", "name": "宠", "desc": "", "frames": 3}])
    app.processEvents()
    check(c.grid_pet.maximumHeight() == c.grid_pet.minimumHeight(),
          "卡片上的**宠物**窗格没钉上限（用户点名的两块之一 ✗）：max=%r min=%r"
          % (c.grid_pet.maximumHeight(), c.grid_pet.minimumHeight()))

    _cards = (ROOT / "gui" / "steps" / "cards.py").read_text(encoding="utf-8")
    # ⭐ 2026-10-05：四段各有自己的背包窗格（掉落物 / 宠物 / **怪物 / 玩家** ✓）⇒ 4 处 ✓
    check(_cards.count("fit_rows(min_rows=1, hard=True)") == 4,
          "卡片里那四个窗格没都写 `hard=True`（少一个就等于没修 ✗）：%d 处"
          % _cards.count("fit_rows(min_rows=1, hard=True)"))
    # ⭐ 四段的窗格都要有（用户 2026-10-05："玩家、怪物也用同样的背包窗格的形式添加" ✓）
    for _attr in ("grid_drop", "grid_pet", "grid_mob", "grid_player"):
        check("self.%s = IconGrid()" % _attr in _cards,
              "卡片里没有 `%s` 那块背包窗格（四段各一块 ✗）" % _attr)


def t_label_card_fields_all_have_tips():
    """⭐⭐ **「模型训练」卡片 4（自动标注）的每个参数都要有 tip**（用户 2026-10-05 ✓ 原话：
    "模型训练页签卡片4里的有些参数没有tips，补一下"）。

    口径 = **用户在界面上看到的那个**：鼠标停在**字段标题**上要弹提示 ✓（UI规范 §6：提示挂
    标题 ✓ 不挂输入框 / 按钮 ✓）⇒ 这里查的就是"这一行的**标题**有没有 tooltip" ✓ ——
    ⚠ 只查控件本身会**漏掉"tip 在标题上"的那些** ✗（老写法就是这样 ✓ 别拿它当判据 ✗）。
    钉三件：
      ① **字段就是 12 个** ✓（删字段不会让这条静默通过 ✓）；
      ② 12 个**一个都不能缺 tip** ✓ —— 本轮补齐的 5 个是 匹配阈值 / 区分度 / 每模板峰数 /
         降采样 / 玩家阈值 ✓；
      ③ 那 5 条的 tip 是**真说明**（规范 §6 要求说清：是什么+单位 · 0 什么含义 ·
         调大调小的代价 ✓）⇒ 长度兜一道（短于 40 字基本就是一句废话 ✗）。
    ⚠ 只管**卡片 4**（用户点名的那张 ✓）；别的卡片还有存量缺口 ✓ —— 要补照这个用例的形状来 ✓
      （`LabelCard()` → 遍历 `card.widgets` → 用 `QFormLayout.labelForField(w)` 找标题 ✓）。
    """
    from PyQt5.QtWidgets import QApplication, QFormLayout

    from gui.steps.cards import LabelCard

    _app = QApplication.instance() or QApplication([])
    check(_app is not None, "建不起 QApplication")

    # 卡片的标题提示有两种挂法都要认：`field(tip=…)` / `self.tip(key, …)`（都落到标题上 ✓）
    _FILLED = ("thresh", "min_distinct", "max_peaks", "downscale", "player_thresh")
    c = LabelCard()
    try:
        _keys = list((c.widgets or {}).keys())
        check(len(_keys) == 24,
              "卡片 4 的字段数变了（用例里写死 24 = 四类各 5 个参数 + 2 个勾选框 + "
              "「框到可见部分」那两个（四段共用 ✓ 用户 2026-10-05 ✓）"
              "⇒ 加/删字段时这里要跟着改 ✓）：%d 个 %r" % (len(_keys), _keys))
        _forms = c.findChildren(QFormLayout)
        miss, short = [], []
        for key in _FILLED:
            check(key in (c.widgets or {}),
                  "卡片 4 里没有字段 %r（改了名字就要改这个用例 ✓）" % key)
        for key, tup in (c.widgets or {}).items():
            w = tup[0]
            lab = None
            for form in _forms:
                if form.labelForField(w) is not None:
                    lab = form.labelForField(w)
                    break
            tip = ((lab.toolTip() if lab is not None else "") or "").strip() \
                or (w.toolTip() or "").strip()
            if not tip:
                miss.append(key)
            elif key in _FILLED and len(tip) < 40:
                short.append((key, tip))
        check(not miss,
              "卡片 4 还有参数没 tip（鼠标停在字段标题上什么都不弹 ✗）：%r" % (miss,))
        check(not short,
              "本轮补的那 5 条 tip 太短（规范 §6：要说清是什么+单位 / 0 什么含义 / "
              "调大调小的代价 ✗）：%r" % (short,))
    finally:
        c.deleteLater()


def t_calib_card_class_scales():
    """⭐⭐ **卡片 3「标定尺度」也支持标定掉落物 / 宠物**（用户 2026-10-05 ✓ 原话：
    "卡片3标定尺度 也要支持标定掉落物、宠物类"）。

    为什么值得单独给它们一个尺度 ✓：那两类的模板来自**另外两套图库**（掉落物图标 /
    宠物组合外观 ✓ 见 ④ 的 `label_drops` / `label_pets` ✓），不是怪、人那批精灵 ✓。
    钉七件：
      ① 卡片 3 多两行状态（掉落物状态 / 宠物状态 ✓）；
      ② ④ 里**没勾**那一类 ⇒ 状态「未用」且**不计入**卡片总状态 ✓（没打算标掉落物的人
         不该被一个"未标定"把卡片卡成黄灯 ✗）；
      ③ 勾了但没标 ⇒ 「未标定」+ 卡片 warn ✓（提示里写明"不标则沿用总尺度" ✓ 不是硬要求 ✓）；
      ④ 标了 ⇒ 「已标定」+ 卡片回绿 ✓；
      ⑤ 「一键设置尺度」把**四类**都写上（`scale` / `player_scale` / `drop_scale` /
         `pet_scale` + 各自的 `*_at` ✓）；
      ⑥ `_save_manual_scale`：`drop` / `pet` ⇒ 写**类尺度** + `*_at`，且**不碰** `mob_scales` ✓
         （掉落物 / 宠物没有逐帧标定那一套 ✗ 硬塞会把尺度带歪 ✓）；`mob` ⇒ 照旧写帧级
         `mob_scales["id:帧"]` ✓（老口径一字不变 ✓）；
      ⑦ 换分辨率 ⇒ 「需要重标」✓（`*_at` 里记了当时的画面尺寸 ✓）。
    """
    import shutil
    import tempfile
    from pathlib import Path

    import numpy as np
    from PyQt5.QtWidgets import QApplication

    from core.imgio import imwrite
    from gui.project import Project
    from gui.steps.cards import CalibCard

    # ⚠⚠ 没它建 QWidget 会原生崩（`0xC0000005`，一句 Python 异常都没有 ✗ 本轮踩过 ✓）
    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")

    tmp = Path(tempfile.mkdtemp(prefix="calib_class_"))
    try:
        p = Project.create(tmp / "proj", name="用例", map_id="105090600")
        p.frames.mkdir(parents=True, exist_ok=True)
        # ⚠ 要**真能读**的 png —— `_frame_size()` 走 `imread` ✓（写字节串那种读不出来 ⇒ 尺寸 0 ✓
        #   那样"需要重标"这条就永远测不到 ✗）
        imwrite(p.frames / "frame_00000.png",
                np.zeros((48, 64, 3), dtype=np.uint8))
        # 怪 / 人先标好（这样卡片状态就只由"掉落物/宠物"这一条决定 ✓）
        p.set("scale", 1.4)
        p.set("player_scale", 1.4)

        c = CalibCard()
        c.bind(p)
        check(hasattr(c, "lbl_drop_state") and hasattr(c, "lbl_pet_state"),
              "卡片 3 没有掉落物 / 宠物那两行状态 ✗")

        # ② ④ 里没勾 ⇒ 「未用」，且卡片状态**不受影响** ✓
        st, msg = c.detect_state(p)
        check(c._class_state("drop")[0] == "未用" and c._class_state("pet")[0] == "未用",
              "没勾的那一类没显示「未用」✗：%r / %r"
              % (c._class_state("drop"), c._class_state("pet")))
        check(st == "done" and "掉落" not in msg,
              "没勾那一类却把卡片状态带成 %r（人没打算标它 ✗ 说明：%r）" % (st, msg))

        # ③ 勾了但没标 ⇒ 未标定 + warn（提示里要写明"不标则沿用总尺度"✓）
        p.sec("label")["label_drops_on"] = True
        t3, col3, note3 = c._class_state("drop")
        check(t3 == "未标定" and col3 == "#b06000" and "总尺度" in note3,
              "勾了没标掉落物时的状态不对（该是黄色「未标定」+ 说明可不标 ✓）：%r"
              % ((t3, col3, note3),))
        check(c.detect_state(p)[0] == "warn",
              "勾了却没标掉落物 ⇒ 卡片该是黄灯（提醒该去标 ✓）：%r" % (c.detect_state(p)[0],))

        # ④ 标了 ⇒ 已标定 + 回绿
        check(c._save_manual_scale("drop", "09000000", "0", 1.9) is True,
              "`_save_manual_scale('drop')` 没写回 ✗")
        check(float(p.get("drop_scale") or 0) == 1.9,
              "掉落物的尺度没落盘 ✗：%r" % (p.get("drop_scale"),))
        check(p.get("drop_scale_at"),
              "没记 `drop_scale_at`（换分辨率时就判不出「需要重标」✗）")
        check(not (p.get("mob_scales") or {}),
              "掉落物不该写进 `mob_scales`（那是「怪 id:帧」的账 ✗ 会把尺度带歪 ✓）")
        check(c._class_state("drop")[0] == "已标定",
              "标了之后状态还是 %r ✗" % (c._class_state("drop")[0],))
        check(c.detect_state(p)[0] == "done",
              "标完掉落物卡片没回绿 ✗：%r" % (c.detect_state(p)[0],))
        check("掉落:已标定" in c.detect_state(p)[1],
              "卡片状态文字里没写掉落物那一项 ✗：%r" % (c.detect_state(p)[1],))

        # ⑥ 老口径不变：`mob` 还是写**帧级** `mob_scales["id:帧"]` ✓
        check(c._save_manual_scale("mob", "1234567", "stand0_0", 1.55) is True,
              "`_save_manual_scale('mob')` 没写回 ✗")
        check((p.get("mob_scales") or {}).get("1234567:stand0_0") == 1.55,
              "怪那条的帧级覆盖没落盘（老口径被改坏了 ✗）：%r" % (p.get("mob_scales"),))
        check(p.get("scale_at"),
              "怪那条没写 `scale_at` ✗")

        # ⑦ 换分辨率 ⇒ 需要重标（`*_at` 记的是**当时**的画面尺寸 ✓）
        check(c._class_state("drop")[0] == "已标定",
              "刚标完就显示要重标了（`*_at` 的尺寸写错了 ✗）：%r"
              % (c._class_state("drop"),))
        p.set("drop_scale_at", {"width": 999, "height": 999, "confidence": "manual"})
        check(c._class_state("drop")[0] == "需要重标",
              "画面尺寸变了却没提示「需要重标」✗：%r" % (c._class_state("drop"),))
        p.set("drop_scale_at", {"width": 64, "height": 48, "confidence": "manual"})

        # ⑤ 一键设置尺度：四类都要写上（+ 各自的 `*_at`）
        c.sp_apply_all.setValue(1.25)
        c._apply_all_scale()
        for k in ("scale", "player_scale", "drop_scale", "pet_scale"):
            check(abs(float(p.get(k) or 0) - 1.25) < 1e-6,
                  "「一键设置尺度」没设到 %s（只设了一部分 ✗）：%r" % (k, p.get(k)))
        for k in ("scale_at", "player_scale_at", "drop_scale_at", "pet_scale_at"):
            check(p.get(k), "「一键设置尺度」没写 %s（换分辨率判不出来 ✗）" % k)

        # ③′ 宠物那条同样走一遍（勾上 ⇒ 未标定 ⇒ 标 ⇒ 绿）
        p.sec("label")["label_pets_on"] = True
        c.refresh()
        check(c._class_state("pet")[0] == "已标定",
              "一键设完之后宠物该是「已标定」（刚被设过 ✓）：%r" % (c._class_state("pet"),))

        # ⑧ ⭐⭐ **最下面那行灰字报的是「尺寸基准」**（用户 2026-10-05 ✓ 原话："卡片3里最下面的
        #   灰字改成显示「{类名}:{尺寸基准}」而不是「{类名}：已标定」"）。
        #   钉四件：
        #     ① 有数 ⇒ 写数（`怪:1.250` ✓）而不是状态词 ✓；
        #     ② 四类都在（怪 / 人 / 掉落 / 宠物 ✓ 后两类只在 ④ 勾了时出现 ✓ 沿用老口径 ✓）；
        #     ③ 没数（没标过）⇒ 还是写状态词 ✓（没有数可报 ✓）；
        #     ④ 有数但状态不是"已标定" ⇒ 数后面挂**短状态** ✓（`1.900（需要重标）` ✓
        #        —— 换了分辨率却不提示，人会拿着一个错的数跑 ✗）。
        #   ⚠ 与上面那几行**状态**分工不同 ✓（那几行 = 状态词 + 该怎么办 ✓）：
        #     所以顺手钉一下"上面那行还是状态词"（别被一起改掉 ✗）。
        _sum = c.summarize(p)
        check("怪:1.250" in _sum and "人:1.250" in _sum,
              "最下面那行没报尺寸基准（还在写「怪:已标定」✗）：%r" % (_sum,))
        check("掉落:1.250" in _sum and "宠物:1.250" in _sum,
              "掉落物 / 宠物那两格没进摘要（勾了就该有 ✓）：%r" % (_sum,))
        check("已标定" not in _sum,
              "最下面那行还在写状态词「已标定」（用户要求改的就是它 ✗）：%r" % (_sum,))
        check(c.detect_state(p)[1].startswith("怪:已标定"),
              "上面那行（状态灯旁）被一起改成数了 ✗ —— 分工是：那行说状态、最下面那行报数 ✓：%r"
              % (c.detect_state(p)[1],))
        p.set("player_scale", 0)                   # 没标过 ⇒ 没有数可报 ✓
        check("人:未标定" in c.summarize(p),
              "没标过的那一类没退回状态词（报了个空数 ✗）：%r" % (c.summarize(p),))
        p.set("player_scale", 1.25)
        p.set("drop_scale_at", {"width": 999, "height": 999, "confidence": "manual"})
        check("掉落:1.250（需要重标）" in c.summarize(p),
              "有数但状态不对时没挂短状态（换分辨率不提示 ⇒ 拿着错的数跑 ✗）：%r"
              % (c.summarize(p),))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_calib_dialog_class_targets():
    """⭐ **手动标定弹窗里有 掉落物 / 宠物 两个目标；图库里没有那些 id 时说得清**（用户 2026-10-05 ✓）。

    弹窗是**模态**的（`exec_()` ✓）跑不了整条交互 ⇒ 这里只驱动它的**填充**那一步 ✓
    —— 那正是本轮新写的代码 ✓（模板从哪儿来 ✓ 空的时候说什么 ✓）。
    钉三件：
      ① 目标下拉是 **4 项**（怪 / 人 / 掉落物 / 宠物 ✓）；
      ② 清单里那些 id **在图库里不存在** ⇒ 模板下拉给一条**说清原因**的占位 ✓
         （不静默留空下拉 ✗ —— "选不到东西"和"弹窗坏了"得分得清 ✓），
         且它 **data 为空** ✓ ⇒ 画布上不会叠一张错的图 ✓；
      ③ 两类都给"该先干什么"（导出图库 / 勾上那一类 ✓）✓。
    """
    import shutil
    import tempfile
    from pathlib import Path

    from PyQt5.QtWidgets import QApplication

    from gui.calib_manual import CalibManualDialog

    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")

    tmp = Path(tempfile.mkdtemp(prefix="calibdlg_"))
    try:
        d = CalibManualDialog(
            [], "", tmp, {}, 1.41, 1.41, None,
            drops=[{"id": "00000000", "name": "不存在的道具"}],
            pets=[{"pet": "0000000", "name": "不存在的宠物", "equips": []}])
        try:
            tags = [d.cmb_target.itemData(i) for i in range(d.cmb_target.count())]
            check(tags == ["mob", "player", "drop", "pet"],
                  "标定目标不是 4 项（掉落物 / 宠物没进去 ✗）：%r" % (tags,))
            for tag in ("drop", "pet"):
                d.cmb_target.setCurrentIndex(tags.index(tag))
                n = d.cmb_tpl.count()
                txt = d.cmb_tpl.itemText(0) if n else ""
                check(n == 1 and not str(d.cmb_tpl.itemData(0) or "").strip(),
                      "%s 那条在图库里没有时没给「说清原因」的占位"
                      "（或占位带了路径 ⇒ 会叠一张错图 ✗）：%d 项 %r" % (tag, n, txt))
                check(("图库" in txt) or ("导出" in txt),
                      "%s 的占位没说清该先干什么（导出图库 / 勾上那一类 ✓）：%r" % (tag, txt))
        finally:
            d.deleteLater()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_all_cards_fields_have_tips():
    """⭐⭐ **「模型训练」全卡片的字段都要有 tip**（用户 2026-10-05 ✓ 原话："把所有卡片的tips补齐"）。

    口径 = **用户在界面上看到的那个**：鼠标停在**字段标题**上要弹提示 ✓（UI规范 §6 ✓
    提示挂标题 ✓ 不挂输入框 / 按钮 ✓）。
    ⚠⚠ **path 类字段的"字段"是那一行的行布局** ✗（`gui/steps/base.py` 的 `kind == "path"`
      分支把**行布局**交给 `addRow` ✓）⇒ 只拿控件去 `labelForField` 会把它们**误判成"没 tip"** ✗
      （本轮就踩了：报 `CaptureCard.file` / `VerifyCard.source` 没 tip，其实**有** ✓）。
    ⚠ 卡片 1 有两行、卡片 2 有一行是**手搓**的（`map_id` / `player_id` / `win_rect` ✓
      不是 `field()` 建的 ✓）⇒ 判据得覆盖"控件"和"它所在的行布局"两条路 ✓。
    钉两件：
      ① 全卡片**一个字段都不缺 tip** ✓ —— 本轮补了 **17** 个
         （MapCard 4 / CaptureCard 8 / DatasetCard 1 / VerifyCard 3 / IterateCard 1 ✓）；
      ② 那 17 条是**真说明**（≥ 40 字 ✓ 规范 §6：说清 是什么+单位 / 0 什么含义 /
         调大调小的代价 ✓）。
    """
    from PyQt5.QtWidgets import QApplication, QFormLayout

    from gui.steps import cards as C

    _app = QApplication.instance() or QApplication([])
    check(_app is not None, "建不起 QApplication")

    #: 本轮补齐的字段（卡片名, key）⇒ 要额外钉"不是一句废话" ✓
    _NEW = {("MapCard", "map_filter"), ("MapCard", "only_mob"),
            ("MapCard", "map_id"), ("MapCard", "player_id"),
            ("CaptureCard", "source"), ("CaptureCard", "file"),
            ("CaptureCard", "win_rect"), ("CaptureCard", "url"),
            ("CaptureCard", "fps"), ("CaptureCard", "stride"),
            ("CaptureCard", "seconds"), ("CaptureCard", "limit"),
            ("DatasetCard", "val_ratio"),
            ("VerifyCard", "source"), ("VerifyCard", "conf"), ("VerifyCard", "imgsz"),
            ("IterateCard", "limit")}

    def _title_tip(card, w):
        """这一行的标题上的 tip（控件本身那条也算 ✓ 两种挂法都认 ✓）。"""
        for form in card.findChildren(QFormLayout):
            lab = form.labelForField(w)
            if lab is not None and (lab.toolTip() or "").strip():
                return lab.toolTip().strip()
            # ⚠ path 类：真正交给 `addRow` 的是**行布局** ✗（不是那个 QLineEdit ✓）
            for i in range(form.rowCount()):
                it = form.itemAt(i, QFormLayout.FieldRole)
                lay = it.layout() if it is not None else None
                if lay is not None and lay.indexOf(w) >= 0:
                    lab = form.labelForField(lay)
                    if lab is not None and (lab.toolTip() or "").strip():
                        return lab.toolTip().strip()
        return (w.toolTip() or "").strip()

    miss, short = [], []
    for cls in C.ALL_CARDS:
        card = cls()
        try:
            for key, tup in (card.widgets or {}).items():
                tip = _title_tip(card, tup[0])
                if not tip:
                    miss.append("%s.%s" % (cls.__name__, key))
                elif (cls.__name__, key) in _NEW and len(tip) < 40:
                    short.append(("%s.%s" % (cls.__name__, key), tip))
        finally:
            card.deleteLater()
    check(not miss,
          "这些字段鼠标停在标题上什么都不弹（用户点名的就是这件 ✗）：%r" % (miss,))
    check(not short,
          "本轮补的 tip 太短（规范 §6：要说清是什么+单位 / 0 什么含义 / 调大调小的代价 ✗）：%r"
          % (short,))


def t_player_locator_params():
    """⭐⭐ **玩家定位那条链现在也有「区分度 / 最大模板帧 / 每模板峰数 / 降采样」**
    （用户 2026-10-05 ✓ 原话："参数都放，缺功能的就补功能" —— 之前玩家只有 阈值 / 尺度 ✗，
    ④ 里那 5 个参数摆上去就是**摆设** ✗ 规范 §7 不许 ✓）。

    全部用**合成图**（不碰真实素材 ✓也不碰配置 ✓）。钉四件：
      ① **默认值 = 老行为**：不截帧（`per_mob=0` ✓）、单峰（`max_peaks=1` ✓）、
         不设区分度闸（`min_distinct=0` ✓）⇒ 库里老调用方一字不变 ✓；
      ② `per_mob=N` ⇒ 模板被截到 N 帧（×2 镜像 ✓）且**等距抽**（每个动作都留到代表帧 ✓）；
      ③ `max_peaks=2` ⇒ 每模板能收**第二个峰**（画面里贴两份同样的模板 ⇒ 两处都成候选 ✓）；
      ④ `min_distinct` ⇒ **可信度闸**（对玩家不是去重闸 ✗ 见实现里那段说明 ✓）：
         两份几乎同分 ⇒ 分差 < 门限 ⇒ 判**存疑**（返回 None ✓）；
         `=0`（默认 ✓）⇒ 不看这项 ⇒ 照样返回 ✓。
    """
    import shutil
    import tempfile
    from pathlib import Path

    import numpy as np

    from core.imgio import imwrite
    from perception.player_locator import PlayerLocator

    tmp = Path(tempfile.mkdtemp(prefix="ploc_"))
    try:
        pid = "用例角色"
        d = tmp / pid
        d.mkdir(parents=True, exist_ok=True)
        # 有辨识度的模板（带 alpha ✓）—— ⚠⚠ **必须左右不对称** ✗：对称模板的"镜像"分一样高，
        #   那就会顶替"同一个模板的第二个峰"⇒ ③ 里想单独钉 `max_peaks` 就钉不出来 ✓
        #   （本轮先用了对称的，栽在最后那条断言上 ✓）。做法：红色**从左到右递减** ✓
        tpl = np.zeros((24, 16, 4), dtype=np.uint8)
        tpl[4:20, 3:13, 2] = np.linspace(230, 40, 10, dtype=np.uint8)   # 左亮右暗（BGR 红 ✓）
        tpl[:, :, 3] = 255                # 全不透明（`_load_tpl` 会裁到包围盒 ✓）

        def _paste(canvas, cx, cy):
            h, w = tpl.shape[:2]
            x0, y0 = int(cx - w / 2), int(cy - h / 2)
            reg = canvas[y0:y0 + h, x0:x0 + w]
            a = (tpl[:, :, 3:4].astype(np.float32) / 255.0)
            reg[:] = (tpl[:, :, :3].astype(np.float32) * a
                      + reg.astype(np.float32) * (1 - a)).astype(np.uint8)

        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        _paste(frame, 200, 150)
        frame_p = tmp / "frame.png"
        imwrite(frame_p, frame)

        # ① 默认：找得到（老行为 ✓）
        imwrite(d / ("%s-stand1-0.png" % pid), tpl)
        loc = PlayerLocator(pid, root=tmp, scale=1.0, threshold=0.5)
        r = loc.locate(frame)
        check(r is not None, "默认参数下**没定位到**（老行为被改坏了 ✗）：%r" % (r,))
        check(abs(r[0] - 200) < 8 and abs(r[1] - 150) < 8,
              "定位到的位置不对（该 ≈ (200,150) ✗）：%r" % (r[:2],))
        check(loc.template_count == 2, "一张模板该出 2 个（含镜像 ✓）：%d"
              % (loc.template_count,))

        # ② `per_mob=N` 截帧 + 等距抽（6 帧 ⇒ 留 2 帧 ⇒ 模板 4 个 ✓）
        for i in range(1, 6):
            imwrite(d / ("%s-stand1-%d.png" % (pid, i)), tpl)
        loc_full = PlayerLocator(pid, root=tmp, scale=1.0, threshold=0.5)
        check(loc_full.template_count == 12, "没截帧时该是 6 帧 ×2 = 12 个模板 ✗：%d"
              % (loc_full.template_count,))
        loc_cap = PlayerLocator(pid, root=tmp, scale=1.0, threshold=0.5, per_mob=2)
        check(loc_cap.template_count == 4,
              "「最大模板帧」没生效（该截到 2 帧 ⇒ 4 个模板 ✓）：%d" % (loc_cap.template_count,))

        # ③④ 两个同分位置：`max_peaks` 决定看不看得见"次佳" ✓，
        #     `min_distinct` 决定"看见之后认不认"  ✓
        frame2 = np.zeros((360, 640, 3), dtype=np.uint8)
        _paste(frame2, 150, 120)
        _paste(frame2, 460, 260)          # 同样一块 ⇒ 两个峰几乎同分 ✓
        imwrite(tmp / "frame2.png", frame2)
        d2 = tmp / pid
        for f in d2.glob("*stand1-*.png"):          # 只留一张模板，别让 6 帧干扰 ✓
            if f.stem != "%s-stand1-0" % pid:
                f.unlink()
        _ok = PlayerLocator(pid, root=tmp, scale=1.0, threshold=0.5, max_peaks=2,
                            min_distinct=0.0).locate(frame2)
        check(_ok is not None, "`min_distinct=0`（不看这项 ✓）却判成没找到 ✗：%r" % (_ok,))
        _bad = PlayerLocator(pid, root=tmp, scale=1.0, threshold=0.5, max_peaks=2,
                             min_distinct=0.05).locate(frame2)
        check(_bad is None,
              "两份几乎同分的位置 + 区分度 0.05 ⇒ 该判**存疑**（返回 None ✓），却给了：%r"
              % (_bad,))
        # ⚠ 这条是**专门钉 `max_peaks`** 的（模板已经改成左右不对称 ✓ ⇒ 镜像那条分很低 ✗）：
        #   `max_peaks=1` ⇒ 第二个峰根本没进候选 ⇒ "次佳"是那条低分镜像 ⇒ 分差很大 ⇒ **返回** ✓
        _one = PlayerLocator(pid, root=tmp, scale=1.0, threshold=0.5, max_peaks=1,
                             min_distinct=0.05).locate(frame2)
        check(_one is not None,
              "`max_peaks=1`（第二个峰没进候选 ⇒ 与次佳的分差很大 ✓）时不该判存疑 ✗：%r" % (_one,))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_frames_sel_restricts_templates():
    """⭐⭐ **用户手工挑的模板帧**（任务 3 ✓ 用户 2026-10-05 ✓ 原话："在背包窗格中双击选中的
    怪物、玩家、掉落物、宠物，显示弹窗展示所有的帧，用户可以自由决定选中哪些帧用于匹配
    （默认是旧逻辑中的那些）"）。

    两根轴别混 ✗：**画面帧**（`_only` ✓ 这一趟处理哪些素材）vs **模板帧**（`label.frames_sel` ✓
    这个条目拿哪几张图去匹配 ✓）—— 本用例只钉后者 ✓。
    钉四件（都不依赖真实图库 ✓ 全用临时目录 ✓）：
      ① `load_templates(frames_sel=…)` ⇒ 名单**真的**把模板压下来 ✓，
         且名单是**候选集**（后面那套"上限/轮询"照旧在它之上跑 ✓ 不是替换 ✓）；
      ② ⚠ 名单里一个都没命中 ⇒ 这一条**没有模板** ✓（**不许**静默回落成全用 ✗ ——
         用户全取消勾 = "这条别用" ✓）；
      ③ 卡片这条链：`make_task` 把 `frames_sel` **原样透传** ✓ + `label.frames_sel` 能存能读 ✓；
      ④ 玩家那条链也认这份名单（`PlayerLocator(frames_sel=…)` ✓ —— 它是**另一套**加载器 ✗
         不接的话"双击选玩家模板帧"就是假的 ✓）。
    """
    import shutil
    import tempfile
    from pathlib import Path

    import numpy as np
    from PyQt5.QtWidgets import QApplication

    from core.imgio import imwrite
    from gui.project import Project
    from gui.steps.cards import LabelCard
    from perception.player_locator import PlayerLocator
    from tools.detect_mobs import load_templates

    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")

    tmp = Path(tempfile.mkdtemp(prefix="fsel_"))
    try:
        # 一张"够格"的模板（≥ min_side 20px ✓ ≥ min_alpha 200 不透明像素 ✓）
        tpl = np.zeros((24, 24, 4), dtype=np.uint8)
        tpl[:, :, 2] = 180
        tpl[:, :, 3] = 255
        root = tmp / "sprites"
        d = root / "怪物甲"
        d.mkdir(parents=True, exist_ok=True)
        for st in ("stand_0", "stand_1", "move_0"):
            imwrite(d / ("%s.png" % st), tpl)

        # ① 名单 = 候选集（只留 stand_0 ⇒ 1 张 ×2 镜像 = 2 个模板 ✓）
        n_all = len(load_templates(root, ["怪物甲"]))
        n_sel = len(load_templates(root, ["怪物甲"], frames_sel={"怪物甲": ["stand_0"]}))
        check(n_all == 6, "没名单时该是 3 张 ×2 镜像 = 6 个模板 ✗：%d" % n_all)
        check(n_sel == 2, "名单没把模板压下来（该剩 stand_0 一张 ⇒ 2 个 ✓）：%d" % n_sel)
        # ⭐⭐ **`mirror=False` ⇒ 模板数减半**（用户 2026-10-06 ✓ 掉落物那条用它 ✓）：
        #   道具图标没有左右朝向 ⇒ 镜像那份只是多花一倍时间 + 多一批"只有镜像才像"的假框 ✗。
        #   ⚠ 默认**必须仍是 True** ✗（怪物 / 宠物有朝向 ✓ 老行为一字不变 ✓）—— 两条一起钉 ✓。
        n_nomirror = len(load_templates(root, ["怪物甲"], mirror=False))
        check(n_nomirror == 3,
              "`mirror=False` 没把镜像那份去掉（该剩 3 张原图 ✓）：%d" % n_nomirror)
        check(len(load_templates(root, ["怪物甲"], mirror=True)) == n_all,
              "`mirror=True` 跟默认不一致（默认就该是 True = 老行为 ✗）")
        # ⚠ 名单是候选集：名单里给两张、上限给 1 ⇒ 仍会被"上限"再削一道 ✓
        n_both = len(load_templates(root, ["怪物甲"], max_per_mob=1,
                                    frames_sel={"怪物甲": ["stand_0", "move_0"]}))
        check(n_both == 2, "名单与「最大模板帧」该是**两根正交的轴**（2 张候选 + 上限 1 ⇒ 1 张 ⇒ 2 个 ✓）：%d"
              % n_both)
        # ② 一个都没命中 ⇒ 这一条没有模板（不许回落 ✓）
        n_none = len(load_templates(root, ["怪物甲"], frames_sel={"怪物甲": ["不存在的帧"]}))
        check(n_none == 0,
              "名单全不命中却回落成全用了（那是把用户的「别用」当耳旁风 ✗）：%d" % n_none)

        # ④ 玩家那条链也认这份名单（另一套加载器 ✓）
        pd = root / "角色甲"
        pd.mkdir(parents=True, exist_ok=True)
        for st in ("角色甲-stand1-0", "角色甲-stand1-1", "角色甲-move-0"):
            imwrite(pd / ("%s.png" % st), tpl)
        p_all = PlayerLocator("角色甲", root=root, scale=1.0)
        p_sel = PlayerLocator("角色甲", root=root, scale=1.0,
                              frames_sel=["角色甲-stand1-0"])
        check(p_all.template_count == 6, "玩家默认该是 3 张 ×2 = 6 个模板 ✗：%d"
              % p_all.template_count)
        check(p_sel.template_count == 2,
              "玩家那条没认这份名单（那「双击选玩家模板帧」就是假的 ✗）：%d"
              % p_sel.template_count)

        # ③ 卡片这条链：透传 + 存取
        p = Project.create(tmp / "proj", name="用例", map_id="105090600")
        p.frames.mkdir(parents=True, exist_ok=True)
        c = LabelCard()
        c.bind(p)
        c._frames_sel = {"drop:09000000": ["icon_0", "icon_1"]}
        c.sync(p)
        p.save()
        c2 = LabelCard()
        c2.bind(p)
        check(c2._frames_sel.get("drop:09000000") == ["icon_0", "icon_1"],
              "`label.frames_sel` 没存住 / 没读回来 ✗：%r" % (c2._frames_sel,))
        c2._drops = [{"id": "09000000", "name": "金币"}]
        c2._label_target = "drop"
        c2._drop_mode = "template"
        _fn, pr = c2.make_task(p)
        check(pr.get("frames_sel", {}).get("drop:09000000") == ["icon_0", "icon_1"],
              "任务里没带 `frames_sel`（挑了也白挑 ✗）：%r" % (pr.get("frames_sel"),))
        # ⚠ 没存过 ⇒ 老项目照旧（空字典 ✓ 不是报错 ✓）
        p2 = Project.create(tmp / "proj2", name="老项目", map_id="105090600")
        c3 = LabelCard()
        c3.bind(p2)
        check(c3._frames_sel == {}, "老项目里 `frames_sel` 该是空 ✓：%r" % (c3._frames_sel,))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_class_params_five_each():
    """⭐⭐ **四类各有自己的 5 个参数**（用户 2026-10-05 ✓ 原话："4类按照统一的格式整理…" +
    "参数都放，缺功能的就补功能"）。

    那 5 项 = 匹配阈值 / 区分度 / 最大模板帧 / 每模板峰数 / 降采样 ✓（顺序也是用户点的 ✓）。
    钉五件：
      ① 四类 × 5 项**都在卡片上**（键名齐 ✓）；
      ② ⚠ **精度照用户点名**：匹配阈值 **2 位** / 区分度 **3 位** / 后三个是**整数**框 ✓；
      ③ **透传**：四类各设一组**互不相同**的值 ⇒ 各自的任务里读到的就是自己那组 ✓
         （掉落入/宠物/玩家的都进 pipeline ✓ 见 `make_task` 与 `detect_player` ✓）；
      ④ ⚠⚠ **老项目兜底**：掉落物 / 宠物那三项**没存过** ⇒ 用**总那个**（不是常量 ✗ 一字不变 ✓）；
      ⑤ ⚠⚠ **玩家那四项的默认故意与怪物不同**（区分度 **0** ✓ 最大模板帧 **0**=不截 ✓
         每模板峰数 **1** ✓ 降采样 **1** ✓）—— 因为对玩家，区分度是"可信度闸"，
         给 0.06 会把绝大多数帧判存疑 ✗（见 `perception/player_locator.py` 里那段说明 ✓）。
    """
    import shutil
    import tempfile
    from pathlib import Path

    from PyQt5.QtWidgets import QApplication, QDoubleSpinBox, QSpinBox

    from gui.project import Project
    from gui.steps.cards import LabelCard

    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")

    _FIVE = {"mob": ("thresh", "min_distinct", "per_mob", "max_peaks", "downscale"),
             "player": ("player_thresh", "player_min_distinct", "player_per_mob",
                        "player_max_peaks", "player_downscale"),
             "drop": ("drop_thresh", "drop_min_distinct", "drop_per_mob",
                      "drop_max_peaks", "drop_downscale"),
             "pet": ("pet_thresh", "pet_min_distinct", "pet_per_mob",
                     "pet_max_peaks", "pet_downscale")}
    tmp = Path(tempfile.mkdtemp(prefix="params5_"))
    try:
        p = Project.create(tmp / "proj", name="用例", map_id="105090600")
        p.frames.mkdir(parents=True, exist_ok=True)
        (p.frames / "frame_00000.png").write_bytes(b"x")
        c = LabelCard()
        c.bind(p)

        # ① 四类 × 5 项都在
        for kind, keys in _FIVE.items():
            miss = [k for k in keys if k not in c.widgets]
            check(not miss, "%s 那一段缺参数（该有 5 个 ✓）：%r" % (kind, miss))

        # ② 精度照用户点名（阈值 2 位 / 区分度 3 位 / 后三个整数框）
        for kind, keys in _FIVE.items():
            for i, want in enumerate((2, 3, None, None, None)):
                w = c.widgets[keys[i]][0]
                if want is None:
                    check(isinstance(w, QSpinBox) and not isinstance(w, QDoubleSpinBox),
                          "%s 的「%s」该是整数框 ✗：%r" % (kind, keys[i], type(w).__name__))
                else:
                    check(isinstance(w, QDoubleSpinBox) and w.decimals() == want,
                          "%s 的「%s」该 %d 位小数 ✗：%r"
                          % (kind, keys[i], want, getattr(w, "decimals", lambda: "?")()))

        # ③ 四类各设一组互不相同的值 ⇒ 各自任务读到自己那组
        sec = p.sec("label")
        sec.update({"thresh": 0.91, "min_distinct": 0.011, "per_mob": 21, "max_peaks": 2,
                    "downscale": 1,
                    "player_thresh": 0.72, "player_min_distinct": 0.022,
                    "player_per_mob": 22, "player_max_peaks": 3, "player_downscale": 2,
                    "drop_thresh": 0.73, "drop_min_distinct": 0.033, "drop_per_mob": 23,
                    "drop_max_peaks": 4, "drop_downscale": 3,
                    "pet_thresh": 0.74, "pet_min_distinct": 0.044, "pet_per_mob": 24,
                    "pet_max_peaks": 5, "pet_downscale": 4})
        c._drops = [{"id": "09000000", "name": "金币"}]
        c._pets = [{"pet": "5000020", "name": "小白雪人", "equips": []}]
        p.set("scale", 1.4)
        p.set("player_scale", 1.4)
        p.set("player_id", "角色甲")
        c._drop_mode = c._pet_mode = "template"

        c._label_target = "drop"
        _fn, pr = c.make_task(p)
        check((pr["thresh"], pr["min_distinct"], pr["per_mob"], pr["max_peaks"],
               pr["downscale"]) == (0.73, 0.033, 23, 4, 3),
              "掉落物那 5 个没各自生效（还在吃共用的 ✗）：%r" % (
                  (pr["thresh"], pr["min_distinct"], pr["per_mob"], pr["max_peaks"],
                   pr["downscale"]),))
        c._label_target = "pet"
        _fn, pr = c.make_task(p)
        check((pr["thresh"], pr["min_distinct"], pr["per_mob"], pr["max_peaks"],
               pr["downscale"]) == (0.74, 0.044, 24, 5, 4),
              "宠物那 5 个没各自生效 ✗：%r" % (
                  (pr["thresh"], pr["min_distinct"], pr["per_mob"], pr["max_peaks"],
                   pr["downscale"]),))
        c._label_target = "player"
        _fn, pr = c.make_task(p)
        d = pr["detect"]
        check((d["thresh"], d["min_distinct"], d["per_mob"], d["max_peaks"],
               d["downscale"]) == (0.72, 0.022, 22, 3, 2),
              "玩家那 5 个没各自生效（上一轮补的能力没接上 ✗）：%r" % (
                  (d["thresh"], d["min_distinct"], d["per_mob"], d["max_peaks"],
                   d["downscale"]),))
        c._label_target = "mob"
        _fn, pr = c.make_task(p)
        check(pr["detect"].get("thresh") == 0.91,
              "怪物那条被带歪了（它本来就该吃总那份 ✓）：%r" % (pr["detect"].get("thresh"),))

        # ④ 老项目兜底：掉落物 / 宠物那三项没存过 ⇒ 用**总那个**（不是常量 ✗）
        for k in ("drop_min_distinct", "drop_per_mob", "drop_max_peaks",
                  "pet_min_distinct", "pet_per_mob", "pet_max_peaks"):
            sec.pop(k, None)
        c._label_target = "drop"
        _fn, pr = c.make_task(p)
        check((pr["min_distinct"], pr["per_mob"], pr["max_peaks"]) == (0.011, 21, 2),
              "老项目（没存过那三项）没回落到**总那个**（回落到常量就改了老行为 ✗）：%r"
              % ((pr["min_distinct"], pr["per_mob"], pr["max_peaks"]),))

        # ⑤ 玩家那四项的默认（故意与怪物不同 ✓）
        for k in ("player_min_distinct", "player_per_mob", "player_max_peaks",
                  "player_downscale"):
            sec.pop(k, None)
        c._label_target = "player"
        _fn, pr = c.make_task(p)
        d = pr["detect"]
        check((d["min_distinct"], d["per_mob"], d["max_peaks"], d["downscale"])
              == (0.0, 0, 1, 1.0),
              "玩家那四项的默认不对（区分度该 0、最大模板帧该 0=不截 ✗）：%r"
              % ((d["min_distinct"], d["per_mob"], d["max_peaks"], d["downscale"]),))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_mob_player_sections():
    """⭐⭐ **卡片 4 里新加的「标注玩家」「标注怪物」两段**（用户 2026-10-05 ✓ 原话："标注{类名}…
    要标注的{类名}…背包窗格（玩家、怪物也用同样的背包窗格的形式添加，怪物用stand或fly当icon，
    玩家用stand当icon）…背包窗格的添加、移出选中按钮"）。

    钉五件（**不碰真图库** ✓ 精灵目录用临时目录 + `mock` 顶掉查表 ✓）：
      ① 两段各有一块自己的**背包窗格**（`grid_mob` / `grid_player` ✓）且都钉了行数上限 ✓；
      ② 「要标注的怪物」条目 = 项目 `mobs` ✓ 且 **icon 取 stand → fly 的第一帧** ✓（用户点名 ✓）；
      ③ 「要标注的玩家」是**单选** ⇒ `player_id` 空 = 0 个 ✓、有值 = 1 个 ✓（不假报多选 ✓）；
      ④ 移出选中 / 清角色都**写回项目**（`mobs`/`mob_names`/`mobs_cleared` / `player_id` ✓
         —— 与卡片 1 那份账**同一处** ✓）；
      ⑤ 两个挑选弹窗都建得起来（`MobPickDialog` / `PlayerPickDialog` ✓ 后者走统一几何 ✓）。
    """
    import shutil
    import tempfile
    from pathlib import Path
    from unittest import mock

    import numpy as np
    from PyQt5.QtWidgets import QApplication

    from core.imgio import imwrite
    from gui.project import Project
    from gui.steps.cards import LabelCard

    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")

    tmp = Path(tempfile.mkdtemp(prefix="sect_"))
    try:
        # 造一只假的"怪"：stand_0 / fly_0 各一张（⚠ 要先有 **stand** ⇒ 它该被选成 icon ✓）
        md = tmp / "mob" / "1234567"
        md.mkdir(parents=True, exist_ok=True)
        tpl = np.zeros((24, 24, 4), dtype=np.uint8)
        tpl[:, :, 2] = 200
        tpl[:, :, 3] = 255
        for st in ("stand_0", "fly_0"):
            imwrite(md / ("%s.png" % st), tpl)

        p = Project.create(tmp / "proj", name="用例", map_id="105090600")
        p.frames.mkdir(parents=True, exist_ok=True)
        p.set("mobs", ["1234567"])
        c = LabelCard()
        # ⚠ 精灵目录由 `core.wzexport.sprite_dir_path()` 从 config 读 ✗ ⇒ 临时目录得**顶掉**它 ✓
        #   （`cards.py` 里是 `from core import wzexport` ⇒ 顶它**模块上的属性**就够 ✓）
        from gui.steps import cards as _cm
        with mock.patch.object(_cm.wzexport, "sprite_dir_path", lambda: tmp / "mob"), \
                mock.patch.object(_cm.wzexport, "build_mob_name_map",
                                  lambda: {"1234567": "绿水灵"}):
            c.bind(p)
            # ① 两块窗格都在、都钉了上限
            check(getattr(c, "grid_mob", None) is not None
                  and getattr(c, "grid_player", None) is not None,
                  "怪物 / 玩家那两块背包窗格没建出来 ✗")
            for g, tag in ((c.grid_mob, "怪物"), (c.grid_player, "玩家")):
                check(g.maximumHeight() == g.minimumHeight() > 0,
                      "%s窗格没钉行数上限（显示区会被撑高 ✗）：max=%r min=%r"
                      % (tag, g.maximumHeight(), g.minimumHeight()))
            # ② 怪物条目 = 项目 mobs，icon 取 stand 第一帧
            _rows = c._mob_rows()
            check(len(_rows) == 1 and _rows[0]["id"] == "1234567"
                  and _rows[0]["name"] == "绿水灵",
                  "「要标注的怪物」条目不对（名字/ id 应从项目与名字表来 ✓）：%r" % (_rows,))
            check(str(_rows[0].get("icon") or "").endswith("stand_0.png"),
                  "怪物 icon 没用 **stand** 第一帧（用户点名：stand 或 fly ✓）：%r"
                  % (_rows[0].get("icon"),))
            # ③ 玩家是单选
            check(c._player_rows() == [], "没选角色时该是 **0 个**格子（单选 ✓）：%r"
                  % (c._player_rows(),))
            c.project.set("player_id", "不存在的角色")
            check(len(c._player_rows()) == 1, "选了角色之后该只有 1 个格子 ✗：%r"
                  % (c._player_rows(),))
        # ④ 写回（与卡片 1 同一份账）
        c._write_mobs([])
        check(c.project.get("mobs") == [] and c.project.get("mob_names") == []
              and c.project.get("mobs_cleared") is True,
              "「移出选中」没按卡片 1 那份账写回（`mobs`/`mob_names`/`mobs_cleared` ✓）：%r/%r/%r"
              % (c.project.get("mobs"), c.project.get("mob_names"),
                 c.project.get("mobs_cleared")))
        c._player_del()
        check((c.project.get("player_id") or "") == "",
              "「移出选中」没把角色清掉 ✗：%r" % (c.project.get("player_id"),))
        # ⑤ 两个弹窗建得起来
        from gui.mob_picker import MobPickDialog
        from gui.player_picker import PlayerPickDialog, list_players
        d1 = MobPickDialog([], None)
        d2 = PlayerPickDialog("", None)
        check(d1 is not None and d2 is not None, "两个挑选弹窗建不起来 ✗")
        check(isinstance(list_players(), list), "`list_players()` 该返回清单（没有也给空表 ✓）")
        d1.deleteLater()
        d2.deleteLater()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_rows_build_with_real_project():
    """⭐⭐ **两段格子条目真能建起来**（用户 2026-10-05 ✓ 原话："鳄鱼潭1 无法添加珠缨为要标注
    的玩家了"）。

    这条专治那类**源码看着对、一调就炸**的错 ✗：本轮真凶就是 `_player_rows()` 里用了没导入的
    `re` ✗ ⇒ 玩家段一刷新 `NameError` ⇒ 「添加…」怎么点都加不上 ✓（弹窗与写盘其实都是好的 ✓
    —— 光看代码看不出来 ✓）。
    ⚠ 既有用例都用替身项目、**没真调这两个函数** ⇒ 漏了 ✗
    ⇒ 改 UI 侧代码，必须有一条"真调一次"的用例 ✓（这就是那条 ✓）。
    """
    import tempfile
    from pathlib import Path

    from PyQt5.QtWidgets import QApplication

    from gui.project import Project
    from gui.steps.cards import LabelCard

    _app = QApplication.instance() or QApplication([])         # noqa: F841
    tmp = tempfile.mkdtemp(prefix="rowsreal_")
    p = Project.create(Path(tmp) / "p", name="两段条目", map_id="107000100")
    if not (p.get("mobs") or []):
        p.set("mobs", ["0100100"])
    p.set("player_id", "数值策划")            # 有玩家才会走「按帧号排」那段 ✓
    c = LabelCard()
    c.bind(p)
    if not c._mob_rows():
        raise AssertionError("怪物段条目建不出来 ✗")
    rows_p = c._player_rows()
    if not rows_p or rows_p[0].get("id") != "数值策划":
        raise AssertionError("玩家段条目建不出来 ✗：%r" % (rows_p,))
    c._refresh_mob_list()                     # 这两步原来会 NameError ⇒ 现在必须过 ✓
    c._refresh_player_list()
    if c.grid_player.entries()[0].get("id") != "数值策划":
        raise AssertionError("玩家段刷新后格子里不是那个角色 ✗")


def t_max_peaks_range_and_zero_gates_buttons():
    """⭐⭐ 「每帧最多几个框」：**上限 999 / 负数 = 无限 / 0 ⇒ 那一段按钮灰着**（用户 2026-10-05 ✓
    原话："这个怎么最多是10？明显不合适，改为<0代表无限，填0时启动标注的按钮都置灰不可交互，
    最大999 记得补在tips里"）。

    ⚠ **后端早就已经是这口径** ✓ 别重复改 ✗：`detect_mobs` 的取峰循环（`-1` ⇒ 安全上限 4096 ✓、
    `0` ⇒ 这个模板直接跳过 ✓）与 `PlayerLocator`（`-1` ⇒ 64 兜底 ✓、`0` ⇒ 一轮都不跑 ✓）都实现了 ✓
    ⇒ 本轮**只动界面**：范围 + 灰按钮 + tips ✓。
    钉四件：
      ① 四个（怪物 / 玩家 / 掉落物 / 宠物）范围都是 **-1 .. 999** ✓（以前封顶 20 ✗ —— 一屏几十只
         怪时明显不够 ✓ 用户点名的就是这事 ✓）；
      ② 填 **0** ⇒ 那一段的两个标注按钮**灰** ✓ 且提示里**说清原因** ✓（否则点了也是白跑一趟 ✓）；
      ③ 填 **-1（无限）/ 填回 ≥1** ⇒ **又亮** ✓（值一变当场重算 ✓ 不用等别的动作 ✓）；
      ④ 源码钉：四条 tip 里都写着「负数（-1）= 无限」「0 = 一个都不标」「999」✓（用户要求补 tips ✓）。
    """
    from PyQt5.QtWidgets import QApplication

    from gui.steps.cards import LabelCard

    _app = QApplication.instance() or QApplication([])
    c = LabelCard()
    _pairs = (("max_peaks", c.btn_mob_tpl), ("player_max_peaks", c.btn_player_tpl),
              ("drop_max_peaks", c.btn_drop_tpl), ("pet_max_peaks", c.btn_pet_tpl))
    try:
        # ① 范围
        for _key, _btn in _pairs:
            _w = c.widgets[_key][0]
            check(_w.minimum() == -1 and _w.maximum() == 999,
                  "「%s」的范围还是 %d..%d（用户点名要 -1..999 ✓，以前封顶 20 ✗）"
                  % (_key, _w.minimum(), _w.maximum()))
        # 夹具：四段各摆一项（否则按钮本来就因为"列表空"而灰 ⇒ 等于没测 ✓）
        c.project = {"player_id": "角色"}
        c.grid_mob.set_entries([{"id": "甲"}])
        c.grid_player.set_entries(c._player_rows())
        c._drops = [{"id": "1", "name": "金币"}]
        c._pets = [{"pet": "1", "name": "宠", "equips": []}]
        c.set_busy(False)
        check(all(b.isEnabled() for _k, b in _pairs),
              "夹具没摆好（四段都该亮着才有意义 ✓）：%r"
              % [(k, b.isEnabled()) for k, b in _pairs])
        # ②③ 0 ⇒ 灰；-1 / ≥1 ⇒ 又亮（改值当场生效 ✓）
        for _key, _btn in _pairs:
            _w = c.widgets[_key][0]
            _w.setValue(0)
            check(not _btn.isEnabled(),
                  "「%s」= 0 却没把按钮灰掉（那一趟一个框都不写 ⇒ 点了白跑 ✓）" % _key)
            check("每帧最多几个框" in _btn.toolTip(),
                  "「%s」= 0 时没说清为什么灰（本仓库的规矩是灰着要说清 ✓）：%r"
                  % (_key, _btn.toolTip()[:60]))
            _w.setValue(-1)
            check(_btn.isEnabled(), "「%s」填 -1（无限）却没亮回来 ✗" % _key)
            _w.setValue(2)
            check(_btn.isEnabled(), "「%s」填回 ≥1 却没亮回来 ✗" % _key)
    finally:
        c.deleteLater()

    # ④ 四条 tip 都补上了这两条口径（用户："记得补在tips里" ✓）
    _src = (Path(__file__).resolve().parents[1] / "gui" / "steps" / "cards.py"
            ).read_text(encoding="utf-8")
    for _key in ("max_peaks", "player_max_peaks", "drop_max_peaks", "pet_max_peaks"):
        _seg = _src.split('"%s", "每帧最多几个框"' % _key, 1)[1].split("self.widgets[", 1)[0]
        check("负数（-1）= 无限" in _seg, "「%s」的 tip 没写「负数 = 无限」✗" % _key)
        check("0 = 一个都不标" in _seg, "「%s」的 tip 没写「0 = 一个都不标」✗" % _key)
        check("999" in _seg, "「%s」的 tip 没写上限 999 ✗" % _key)


def t_template_load_is_cancellable():
    """⭐⭐ **模板加载要"边读边看取消"**（用户 2026-10-05 ✓ 原话："能不等他返回吗？"）。

    病：`run_detect` / `run_detect_player` 开头都是**整块**加载模板 ✗（逐张 `imread` 几百张精灵 ✓）
    ⇒ 点了「取消」得等它**整块读完**才轮到下一个 `ctx.canceled()` ✓ ⇒ 界面上就是
    "已请求取消：正在收尾…"卡着不动 ✓（用户截图那条 ✓）。
    钉三件：
      ① `detect_mobs.load_templates(..., should_stop=…)`：说停就**当场**停（不是读完了才看 ✗），
         返回的是**已经读到的**那部分 ✓；
      ② 玩家那条是**另一套**加载器 ✓（`perception.player_locator.PlayerLocator` ✓）也同口径：
         半路叫停 ⇒ `stopped=True` ✓ 且**不抛异常** ✗（用户点的是"取消"，不是"这里坏了" ✓）；
      ③ 源码钉：两个入口都接上（`should_stop=ctx.canceled` ✓）**并且紧接着**再查一次
         `ctx.canceled()` 才往下走 ✓（半份模板绝不许当完整的用 ✗）。
    """
    import shutil
    import tempfile
    from pathlib import Path as _P

    import numpy as np

    from core.imgio import imwrite
    from perception.player_locator import PlayerLocator
    from tools.detect_mobs import load_templates

    tmp = _P(tempfile.mkdtemp(prefix="tpl_cancel_"))
    try:
        tpl = np.zeros((30, 30, 4), np.uint8)
        tpl[:, :, 3] = 255
        d = tmp / "怪物甲"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(6):
            imwrite(d / ("stand_%d.png" % i), tpl)

        n_all = len(load_templates(tmp, ["怪物甲"]))
        check(n_all == 12, "夹具不对（6 张 ×2 镜像 = 12 ✓）：%d" % n_all)
        _c = {"n": 0}

        def _stop():
            _c["n"] += 1
            return _c["n"] > 2

        n_cut = len(load_templates(tmp, ["怪物甲"], should_stop=_stop))
        check(0 < n_cut < n_all,
              "`should_stop` 没让加载半路收工（实得 %d / 全量 %d）—— 它要是在**末尾**才看，"
              "点了取消照样得等它读完 ✗" % (n_cut, n_all))
        check(_c["n"] <= 4,
              "叫停之后还在继续读（问了 %d 次 ⇒ 没在循环里查 ✗）" % _c["n"])

        # ② 玩家那条（另一套加载器 ✓）：同口径 + 不抛异常
        pd = tmp / "角色甲"
        pd.mkdir(parents=True, exist_ok=True)
        for i in range(6):
            imwrite(pd / ("角色甲-stand-%d.png" % i), tpl)
        _c2 = {"n": 0}

        def _stop2():
            _c2["n"] += 1
            return _c2["n"] > 2

        loc = PlayerLocator("角色甲", root=tmp, should_stop=_stop2)
        check(loc.stopped, "玩家模板加载被打断却没留痕（`stopped` ✓）")
        check(loc.template_count > 0, "半路收工时前面读到的那几张该留着 ✓")
        check(_c2["n"] <= 4, "叫停之后还在读玩家模板（问了 %d 次 ✗）" % _c2["n"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ③ 源码钉：两个入口都接上，且**紧接着**再查一次（半份模板不许往下走 ✗）
    _root = _P(__file__).resolve().parents[1]
    for _rel in ("tools/detect_mobs.py", "tools/detect_player.py"):
        _src = (_root / _rel).read_text(encoding="utf-8")
        check("should_stop=ctx.canceled" in _src,
              "%s 没把「取消」接进模板加载（点了取消还得等它读完 ✗）" % _rel)
        # ⚠ 要查"**紧跟其后**"那一次 ✗（`index(..., 0)` 会拿到函数最开头那个检查点 ✓ 那是另一回事）
        _i = _src.index("should_stop=ctx.canceled")
        check(_src.index("if ctx.canceled():", _i) > _i,
              "%s 接上了却没紧跟一次 `ctx.canceled()` —— 半份模板会被当完整的用 ✗" % _rel)
        check(_src.count("if ctx.canceled():") >= 2,
              "%s 只剩一个取消检查点（加载之后那次被删了 ✗）" % _rel)


def t_frames_sel_four_targets():
    """⭐⭐ **四段都能"双击选模板帧"**（用户 2026-10-05 ✓ 原话："在背包窗格中双击选中的怪物、
    玩家、掉落物、宠物，显示弹窗展示所有的帧…" + "四块窗格各接一行双击选模板帧"）。

    这条只钉**界面把名单交到管线手里**这一步（管线那一半在上面那条用例里钉过 ✓）：
      ① 四块窗格都接了 `activated_id`（双击 / 回车 ✓）—— 少一块就是"双击没反应" ✗；
      ② 键的格式：`mob:<怪id>` / `player:<角色id>` ✓（⛔ 必须与管线**摘/剥**的那把钥匙一字不差 ✗
         —— 怪物链 `detect_mobs.py:655` 剥 `mob:` 前缀 ✓、玩家链 `detect_player.py:164`
         摘 `player:<pid>` ✓）；
      ③ `_default_frames` 的上限**各取各段自己的** ✓（原来四段都取总的 `per_mob` ✗
         ⇒ 弹窗的"默认勾选"和真跑用的不是一份名单 ✗，而"默认 = 旧逻辑"正是用户要的 ✓）；
      ④ `make_task` **只递这一段自己的键** ✓ —— ⛔ 别整份字典下去 ✗：怪物链对**每个**键都
         剥前缀 ✗（实测过：掉落/宠物的键会被剥成"掉落 id / 目录名"当怪 id 用 ✗）。
    """
    import tempfile
    from pathlib import Path

    from PyQt5.QtWidgets import QApplication

    from gui.project import Project
    from gui.steps.cards import LabelCard

    _app = QApplication.instance() or QApplication([])          # noqa: F841
    bad = []

    def ck(name, good):
        if not good:
            bad.append(name)

    c = LabelCard()
    # ① 四块窗格都接上了（`receivers` = 挂在这个信号上的槽数 ✓）
    for nm in ("grid_drop", "grid_pet", "grid_mob", "grid_player"):
        g = getattr(c, nm, None)
        n = g.receivers(g.activated_id) if g is not None else 0
        ck("%s 没接双击（双击没反应 ✗）" % nm, bool(n and n > 0))

    # ② 键的格式（与管线那把钥匙一字不差 ✓）
    ck("怪物键不是 mob:<怪id>", c._entry_key("mob", {"id": "1234"}) == "mob:1234")
    ck("玩家键不是 player:<角色>", c._entry_key("player", {"id": "甲"}) == "player:甲")

    # ③ 各段取各自的上限
    seen = []
    _real = c.value
    c.value = lambda k, *a, **kw: (seen.append(k), _real(k, *a, **kw))[1]
    WANT = {"mob": "per_mob", "drop": "drop_per_mob",
            "pet": "pet_per_mob", "player": "player_per_mob"}
    try:
        for tgt, key in WANT.items():
            seen.clear()
            try:
                c._default_frames(tgt, [])
                got = seen[0] if seen else ""
            except Exception:                                   # noqa: BLE001
                got = "（抛了）"
            ck("%s 取的上限是 %s（该是 %s ✗）" % (tgt, got, key), got == key)
    finally:
        c.value = _real

    # ④ 只递这一段自己的键（临时项目里四段各摆一条 ✓）
    tmp = tempfile.mkdtemp(prefix="fselfour_")
    p = Project.create(Path(tmp) / "p", name="四段帧选择", map_id="105090600")
    c.bind(p)
    c._frames_sel = {"mob:111": ["stand_0"], "drop:222": ["a"],
                     "pet:333": ["b"], "player:甲": ["stand_1"]}
    for tgt in ("mob", "drop", "player"):
        c._label_target = tgt
        try:
            _fn, params = c.make_task(p)
            got = (params.get("detect") or params).get("frames_sel") or {}
        except Exception as ex:                                 # noqa: BLE001
            got = {"（抛了）": str(ex)[:40]}
        want = set(k for k in c._frames_sel if k.startswith(tgt + ":"))
        ck("make_task 给 %s 递的不是自己那段：%r（该 %r ✗）" % (tgt, got, want),
           set(got) == want)

    if bad:
        print("  [NG] " + "；".join(bad))
        raise AssertionError("；".join(bad))


def t_frame_sel_dialog_shows_images():
    """⭐⭐ **选模板帧的弹窗要能看图**（用户 2026-10-05 ✓ 原话："预览模板帧需要能看到图，
    光看名字判断不出来"）。

    钉四件（全用**临时图** ✓ 不依赖真图库 ✓）：
      ① 传了 `files` ⇒ **每格都有图** ✓（名字之外还能看图 ✓ 用户要的就是这件 ✓）；
      ② 右侧**预览**：一进来就有图 ✓、点另一格换成那一格 ✓（还报原图尺寸 ✓）；
      ③ 读不出的图**照常列出来**（只看名字 ✓ ⛔ 别把这一帧藏了 ✗ —— 藏了会被当成
         "库里没有这一帧" ✗）且**不许**把上一帧的图留在预览里 ✗（会看错 ✓）；
      ④ 老调用方式（**不传 `files`** ✓）一字不改照样能用 ✓、`selected_stems()` 语义不变 ✓。
    """
    import tempfile
    from pathlib import Path

    from PyQt5.QtGui import QImage
    from PyQt5.QtWidgets import QApplication

    from gui.frame_sel_dialog import FrameSelDialog

    _app = QApplication.instance() or QApplication([])          # noqa: F841
    tmp = Path(tempfile.mkdtemp(prefix="framessel_"))
    files = {}
    for i, (w, h) in enumerate(((30, 20), (12, 40), (25, 25))):
        im = QImage(w, h, QImage.Format_ARGB32)
        im.fill(0xFF3366CC)
        p = tmp / ("f%d.png" % i)
        if not im.save(str(p)):
            raise AssertionError("临时图没写出来 ✗")
        files["f%d" % i] = str(p)
    stems = list(files)

    d = FrameSelDialog(None, "测试", stems, ["f1"], files=files)
    try:
        if d.lst.count() != 3:
            raise AssertionError("格子数不对 ✗：%d" % d.lst.count())
        n_icon = sum(1 for i in range(d.lst.count()) if not d.lst.item(i).icon().isNull())
        if n_icon != 3:
            raise AssertionError("有格子没图（用户要的就是看图 ✗）：%d/3" % n_icon)
        if d.selected_stems() != ["f1"]:
            raise AssertionError("默认勾选不对 ✗：%r" % (d.selected_stems(),))
        # ② 预览跟着当前格走
        d.lst.setCurrentRow(2)
        if d.preview.pixmap() is None or d.preview.pixmap().isNull():
            raise AssertionError("点了格子右边还是空的（看不到图 ✗）")
        if "f2" not in d.lbl_name.text() or "25×25" not in d.lbl_name.text():
            raise AssertionError("右边没写清是哪一帧 / 原图多大 ✗：%r" % d.lbl_name.text())
    finally:
        d.deleteLater()

    # ③ 缺图那格：照常列出、且预览**不许**留上一帧的图
    d2 = FrameSelDialog(None, "缺图", ["ok", "gone"],
                        files={"ok": files["f0"], "gone": str(tmp / "nope.png")})
    try:
        if d2.lst.count() != 2:
            raise AssertionError("缺图的帧被藏了 ✗：%d 格" % d2.lst.count())
        if not d2.lst.item(1).icon().isNull():
            raise AssertionError("不存在的图却给它安了图标 ✗")
        d2.lst.setCurrentRow(1)
        pm = d2.preview.pixmap()
        if pm is not None and not pm.isNull():
            raise AssertionError("缺图那格右边还留着上一帧的图（会看错 ✗）")
    finally:
        d2.deleteLater()

    # ④ 老调用方式
    d3 = FrameSelDialog(None, "老调用", ["a", "b"], ["a"])
    try:
        if d3.lst.count() != 2 or d3.selected_stems() != ["a"]:
            raise AssertionError("不传 files 的老调用方式坏了 ✗")
    finally:
        d3.deleteLater()


def main():
    # ⚠⚠ **整套自检一律不碰用户真实的 `config/wz.yaml`**（2026-10-04 血的教训 ✗）：有一个用例
    #   直接拿 `wzexport.CONFIG_PATH` 当草稿纸写 ⇒ 把用户配置覆盖成了临时目录 ✓，现场就是
    #   "背包窗格这些项目报错 + 图标目录指向 Temp"✗，而那份文件**没进 git** ✗、只能从历史提交
    #   里捞回来 ✓。这里把它指到临时文件 ⇒ **谁手滑都写不到用户那份** ✓（同
    #   `selftest_decision` 里把 `DecisionSettings.save` 换成空操作那条纪律 ✓）。
    import tempfile as _tf

    from core import wzexport as _wz

    _real_cfg = _wz.CONFIG_PATH
    _wz.CONFIG_PATH = Path(_tf.mkdtemp()) / "wz.yaml"
    _wz._CFG_CACHE.clear()
    tests = [
    ("⭐⭐⭐ 第二趟 0 检出**不许清空上一波** + 报告**按类别**念（用户 2026-10-05）",
     t_second_run_protects_and_reports),
    ("⭐⭐⭐ 框能自适应到**可见部分**（用户 2026-10-05）：默认关 ⇒ 完整尺寸一字不变；"
     "没被挡不许收；盖掉右半要收到左半且左边界不动；只剩一小条回退完整；纯色精灵保守回完整",
     t_visible_box_adapts),
    ("⭐⭐⭐ 玩家那一段也吃「框到可见部分」（用户 2026-10-06：所有的匹配都要它）——"
     "内核真收框 + 中间层两处构造都拿到 + 卡片真传下去", t_visible_box_player_segment),
    ("⭐⭐ 卡片 4 的「标注玩家 / 标注怪物」两段（背包窗格 / 单选 / 写回；用户 2026-10-05）",
     t_mob_player_sections),
    ("⭐⭐ 四类各 5 个参数（精度 / 透传 / 老项目兜底 / 玩家默认；用户 2026-10-05）",
     t_class_params_five_each),
    ("⭐⭐ 手工挑的模板帧（frames_sel）：压模板 / 不透传就是不生效（用户 2026-10-05）",
     t_frames_sel_restricts_templates),
    ("⭐⭐ 四段都能双击选模板帧（四块窗格接线 / 键与管线一致 / 各段取自己的上限；用户 2026-10-05）",
     t_frames_sel_four_targets),
    ("⭐⭐ 两段格子条目真能建起来（真调一次 _mob_rows/_player_rows；用户 2026-10-05）",
     t_rows_build_with_real_project),
    ("⭐⭐ 选模板帧的弹窗要能看图（缩略图 / 预览跟着走 / 缺图照列；用户 2026-10-05）",
     t_frame_sel_dialog_shows_images),
    ("⭐⭐ 玩家定位也有 区分度 / 最大模板帧 / 每模板峰数（缺功能的补上；用户 2026-10-05）",
     t_player_locator_params),
    ("⭐⭐ 全卡片的字段都要有 tip（含 path 类行布局 / 手搓行；用户 2026-10-05）",
     t_all_cards_fields_have_tips),
    ("⭐⭐ 卡片 3 也能标定掉落物 / 宠物（状态 / 一键 / 写回 / 重标；用户 2026-10-05）",
     t_calib_card_class_scales),
    ("⭐ 手动标定弹窗：掉落物 / 宠物两个目标 + 图库空时说得清（用户 2026-10-05）",
     t_calib_dialog_class_targets),
    ("⭐ 卡片 4 的每个参数都要有 tip（标题上；用户 2026-10-05）",
     t_label_card_fields_all_have_tips),
    ("②h ⭐⭐ 窗格高度要「钉死」不只是给下限（`hard=True`；用户报的显示区恒高）",
     t_icon_grid_fit_rows_hard),
        ("掉落物图库取数（norm_drops / 清单 / 按 id 名称搜 / 并集 / 现状说明）", t_drop_library),
        ("卡片：勾选显隐 + 列表 + 按钮启用条件（§2）+ 存进项目", t_card_ui),
        ("⭐⭐ 掉落物按钮**先开选帧弹窗**（同怪物的那四个）：取消⇒不跑、YOLO 不问确认、"
         "结果透传；台账按 `drop` 分、弹窗标题叫「标注掉落物」",
         t_drop_buttons_open_frame_picker),
        ("⭐⭐ 掉落物任务**也要带项目标定尺度**（与宠物同一个病：不传 ⇒ 1.0 ⇒ "
         "模板小 40% ⇒ 0 检出）",
         t_drop_task_carries_sprite_scale),
        ("⭐ **地图显示 = `{地图名}_{id}`**（查不到就只给 id，不编）；部署台那行走同一口径",
         t_map_label_is_name_and_id),
        ("⭐ **背包窗格按内容自适应行数**（至少一行；项多会长；变宽不会更多行）",
         t_icon_grid_fit_rows),
        ("⭐ **粗筛**（「外观不符合就别运算」）：像的放行、完全不像才拦、太小/无底图一律"
         "放行；⚠ 默认**关**（实测检出一致但没省时间）",
         t_coarse_gate),
        ("⭐ 掉落物那块**自己**的「匹配阈值」：在块里跟着显隐；老项目用它现在的数；"
         "只改这一项不影响总阈值，且存得住",
         t_drop_thresh_param),
        ("执行端：复用既有算法 + 类别号 class 2（不许空跑当成功）", t_worker_dispatch),
        ("弹窗两区都是背包窗格（悬停看名称/id/类别/几帧；没名字要写明；不打字也能浏览）",
         t_drop_picker_rows),
        ("⭐⭐ 掉落物 desc：清洗 → 清单 → 候选 → 信息窗；外加真清单必须能解析且非空",
         t_drop_desc_shows),
        ("⭐⭐ 提示挂字段标题 + 标题/说明可复制（老写法挂控件 / 整窗扫一遍保证可复制）",
         t_field_tips_and_copyable),
        ("⭐ 掉落物背包窗格：格子里存 id、悬停弹信息窗、移开要收起（照 MapleNecrocer 规格）",
         t_icon_grid_backpack),
        ("⭐ 掉落物缓存与失效：索引/全量行只建一次、目录一变立刻跟上、`load_cfg` 跟着文件走；"
         "默认掉落物 = 图库里 09 开头那些金币（用户 2026-10-04）",
         t_drop_cache_and_invalidation),
        ("端到端：贴一张金币图标 ⇒ 真检出 ⇒ 写出 class 2（图库不在则明确跳过）", t_drop_e2e),
        ("布局整理（§3）：三个分区小标题 + 掉落物块在卡片最下面", t_card_layout),
        ("⭐⭐ 模板加载**边读边看取消**（怪物 / 玩家两条加载器同一口径；用户 2026-10-05）",
         t_template_load_is_cancellable),
        ("⭐⭐ 「每帧最多几个框」= -1..999（负数无限 / 0 ⇒ 按钮灰 / tips 补上；用户 2026-10-05）",
         t_max_peaks_range_and_zero_gates_buttons),
    ]
    ok = 0
    try:
        for name, fn in tests:
            try:
                fn()
            except Exception as e:                        # noqa: BLE001
                print("[FAIL] %s\n       %s: %s" % (name, type(e).__name__, e))
            else:
                print("[ OK ] %s" % name)
                ok += 1
    finally:
        # 收尾：把 `CONFIG_PATH` 还回去（同开头那段说明 ✓ —— 自检不许留改过的东西 ✗）
        _wz.CONFIG_PATH = _real_cfg
        _wz._CFG_CACHE.clear()
    print("\n%d/%d 通过" % (ok, len(tests)))
    return 0 if ok == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
