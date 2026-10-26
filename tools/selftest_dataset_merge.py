"""自检：多项目素材合并（外来项目丢掉「玩家」类）—— `perception/merge_projects.py`。

钉的是这几件（每条都对应一个**会静默出错**的坑 ✓）：

  ① **同名不串台**：两个项目都有 `frame_00000.png` ⇒ 暂存里必须是**两张**，
     外来那张叫 `p01__frame_00000.png`，而且**标注跟着自己的图走** ✓
     （扁平合并的话，B 的框会画在 A 的画面上，训练照跑、不报错 ✗）；
  ② **只丢指定的类**：外来项目的 `0`（玩家）删掉，`1`（怪物）/`4`（其他玩家）保留 ✓；
  ③ **主项目的玩家标注不动**：同一个角色当然还要认 ✓；
  ④ **丢完变空的帧跳过**：整帧只有玩家框 ⇒ 不出图（配空白标注就是教模型
     "这里什么都没有" ✗，而那帧本来有玩家，只是这个类不要 ✓）；
  ⑤ **图是真图**（硬链接/复制都不改像素 ✓）；
  ⑥ **端到端**：`--build` 出来的数据集里，外来帧的标注行里**一个 0 类都没有** ✓；
  ⑦ **界面那两格**（⑥ 数据集卡的「并入其他项目 / 外来项目丢弃类」）：分隔符与去重、
    写错的类 id 要**报错而不是静默放过** ✗，留空时**还是老那条路**（`run_dataset`）✓。

跑：`python -m tools.selftest_dataset_merge`
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from core.context import ConsoleContext
from core.imgio import imread, imwrite
from perception import merge_projects as M


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _mk(root, name, frames):
    """造一个假项目：frames / labels / project.yaml。frames = {帧号: ([框行], 颜色)}"""
    p = Path(root) / name
    (p / "frames").mkdir(parents=True)
    (p / "labels").mkdir(parents=True)
    (p / "project.yaml").write_text("name: %s\nmap_id: '1'\n" % name,
                                    encoding="utf-8")
    for idx, (lines, color) in frames.items():
        img = np.zeros((24, 32, 3), np.uint8)
        img[:, :] = color
        imwrite(p / "frames" / ("frame_%05d.png" % idx), img, quality=95)
        (p / "labels" / ("frame_%05d.txt" % idx)).write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
    return p


def t_merge_projects():
    root = Path(tempfile.mkdtemp(prefix="merge_test_"))
    try:
        # 主项目：玩家 + 怪物（玩家**要留着** ✓）
        main = _mk(root, "主项目", {0: (["0 0.50 0.50 0.20 0.20",
                                        "1 0.40 0.40 0.10 0.10"], (10, 20, 30))})
        # 外来项目：同名帧 frame_00000，但画面不一样 ⇒ 必须分开存 ✓
        other = _mk(root, "外来项目", {0: (["0 0.50 0.50 0.30 0.30",
                                           "1 0.20 0.20 0.10 0.10",
                                           "4 0.70 0.70 0.10 0.10"], (200, 100, 0))})
        # 只有玩家框的项目 ⇒ 丢完为空 ⇒ 整帧不进数据集 ✓
        only_player = _mk(root, "只有玩家", {3: (["0 0.50 0.50 0.30 0.30"],
                                                (0, 0, 200))})

        stage = root / "stage"
        st = M.stage({"into": str(main), "from": [str(other), str(only_player)],
                      "stage": str(stage)}, ConsoleContext())

        frames = sorted(p.name for p in (stage / "frames").glob("*.png"))
        # ① 同名不串台：主项目原名 + 外来 p01__ 前缀；只有玩家的那帧没有 ✓
        check(frames == ["frame_00000.png", "p01__frame_00000.png"],
              "暂存目录的帧不对（该是主项目原名 + `p01__` 前缀，且没有"
              "「只有玩家」那帧）：%r" % (frames,))
        check(st["sources"]["p02"]["frames"] == 0,
              "「只有玩家」那个项目丢完该是 0 帧，实得 %r"
              % (st["sources"]["p02"]["frames"],))

        # ② 外来的 0 类被删、1/4 保留
        _cls = sorted({ln.split()[0] for ln in
                       (stage / "labels" / "p01__frame_00000.txt")
                       .read_text(encoding="utf-8").splitlines() if ln.strip()})
        check(_cls == ["1", "4"],
              "外来项目的标注该只剩怪物(1)/其他玩家(4)，实得 %r ⇒ **玩家类没被剔除** ✗"
              % (_cls,))
        # ③ 主项目的 0 类原样留着
        _cls2 = sorted({ln.split()[0] for ln in
                        (stage / "labels" / "frame_00000.txt")
                        .read_text(encoding="utf-8").splitlines() if ln.strip()})
        check(_cls2 == ["0", "1"],
              "主项目的玩家标注被删了（实得 %r）⇒ 自己的角色也要认 ✗" % (_cls2,))

        # ⑤ 图是真图（外来那张是硬链接/复制过来的原像素）
        _a = imread(main / "frames" / "frame_00000.png")
        _b = imread(stage / "frames" / "frame_00000.png")
        _c = imread(stage / "frames" / "p01__frame_00000.png")
        check(_a is not None and _b is not None and np.array_equal(_a, _b),
              "主项目那张图在暂存里像素变了（该原样 ✓）")
        check(_c is not None and int(_c[0, 0, 0]) == 200,
              "外来项目那张图在暂存里**不是原图**（BGR 首像素该是 200，实得 %r）✗"
              % (None if _c is None else int(_c[0, 0, 0]),))

        # ⑥ 端到端：--build 出来的数据集里，外来帧一个 0 类行都没有
        got2 = M.run_merge({"into": str(main), "from": [str(other)],
                            "stage": str(root / "stage2"), "build": True,
                            "min_boxes": 1, "val_ratio": 0.5},
                           ConsoleContext())
        check(int(got2.get("train", 0)) + int(got2.get("val", 0)) == 2,
              "构建出来的数据集该有 2 张（主 1 + 外来 1），实得 train=%r val=%r"
              % (got2.get("train"), got2.get("val")))
        ds = main / "dataset"
        _lbl = list((ds / "labels").glob("*/*.txt"))
        check(len(_lbl) == 2, "数据集里标注文件该有 2 个，实得 %d" % len(_lbl))
        for f in _lbl:
            if not f.name.startswith("p01__"):
                continue
            _c3 = sorted({ln.split()[0] for ln in
                          f.read_text(encoding="utf-8").splitlines() if ln.strip()})
            check("0" not in _c3,
                  "数据集里外来帧 %s 还带着玩家框（%r）⇒ 剔类没生效 ✗"
                  % (f.name, _c3))
        # ⑦ 丢弃的类可以换：drop=[0,4] ⇒ 4 也丢
        st3 = M.stage({"into": str(main), "from": [str(other)],
                       "stage": str(root / "stage3"), "drop": [0, 4]},
                      ConsoleContext())
        _cls4 = sorted({ln.split()[0] for ln in
                        (root / "stage3" / "labels" / "p01__frame_00000.txt")
                        .read_text(encoding="utf-8").splitlines() if ln.strip()})
        check(_cls4 == ["1"],
              "`--drop 0 --drop 4` 之后该只剩怪物(1)，实得 %r ⇒ 多选的类没生效 ✗"
              % (_cls4,))
        check(st3["sources"]["p01"]["drop"] == [0, 4],
              "来源记账里没写清丢了哪些类 ⇒ 以后没法追溯 ✗：%r"
              % (st3["sources"]["p01"]["drop"],))

        # ⑧ **白名单**：别的项目只标了怪物 ⇒ `keep=[1]` 只取怪物（用户 2026-10-24：
        #   "其他项目是没有标注宠物、掉落物的，我只想取他们的怪物标注"）
        root4 = root / "stage4"
        st4 = M.stage({"into": str(main), "from": [str(other)],
                       "stage": str(root4), "keep": [1]}, ConsoleContext())
        _cls5 = sorted({ln.split()[0] for ln in
                        (root4 / "labels" / "p01__frame_00000.txt")
                        .read_text(encoding="utf-8").splitlines() if ln.strip()})
        check(_cls5 == ["1"],
              "`keep=[1]` 之后外来项目该只剩怪物(1)，实得 %r ⇒ 白名单没生效 ✗"
              % (_cls5,))
        _cls6 = sorted({ln.split()[0] for ln in
                        (root4 / "labels" / "frame_00000.txt")
                        .read_text(encoding="utf-8").splitlines() if ln.strip()})
        check(_cls6 == ["0", "1"],
              "白名单把**主项目**的标注也过滤了（实得 %r）⇒ 它只该作用在外来项目 ✗"
              % (_cls6,))
        # ⑨ ⚠ `drop=[]`（界面上"丢弃方式 + 一个都不勾"）必须**真的一个都不丢** ✗
        #    —— `stage()` 里只有"**没传** drop"才默认丢玩家 ✓；拿 `or [0]` 兜底的话，
        #      "勾空了"会**静默**继续丢玩家 ✓（看不出来，训练集里玩家就没了 ✓）。
        root5 = root / "stage5"
        M.stage({"into": str(main), "from": [str(other)], "drop": [],
                 "stage": str(root5)}, ConsoleContext())
        _cls7 = sorted({ln.split()[0] for ln in
                        (root5 / "labels" / "p01__frame_00000.txt")
                        .read_text(encoding="utf-8").splitlines() if ln.strip()})
        check(_cls7 == ["0", "1", "4"],
              "传了 `drop=[]`（一个都不丢）却还是把玩家丢了（实得 %r）⇒ "
              "「丢弃方式下勾空」会偷偷继续丢 ✗" % (_cls7,))

        check(st4["sources"]["p01"]["keep"] == [1],
              "来源记账里没写白名单 ⇒ 以后没法追溯 ✗：%r"
              % (st4["sources"]["p01"]["keep"],))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def t_card_fields():
    """⑥ 数据集卡那两格：解析、报错、留空走老路（用户 2026-10-24："**能否做成 gui 里的功能？**"）。

    钉四件：
      ① 分隔符随便混（逗号/顿号/分号/换行/空白）都能拆开，**重复的只收一次** ✓
        （同一个项目收两遍 = 同一张图进两次训练 ✗）；
      ② 「丢弃类」写错（`o` / `0,x`）⇒ **报错**，不许静默变成"一个都不丢" ✗
        （那等于带着别人的玩家框去训练，还看不出哪里错 ✓）；
      ③ 留空 ⇒ **还是老那条 `run_dataset`**（老项目一个字都不变 ✓）；
      ④ 填了项目 ⇒ 走 `run_merge`，参数对得上（into=本项目 / from / drop / build=True ✓）。
    """
    # ① 解析
    check(M.parse_projects("森林迷宫_珠缨, 鳄鱼潭1；天空\n黄金海滩　鳄鱼潭1")
          == ["森林迷宫_珠缨", "鳄鱼潭1", "天空", "黄金海滩"],
          "「并入其他项目」的分隔符/去重不对：%r"
          % (M.parse_projects("森林迷宫_珠缨, 鳄鱼潭1；天空\n黄金海滩　鳄鱼潭1"),))
    check(M.parse_projects("") == [] and M.parse_projects("  ,  ") == [],
          "留空该解析成空列表（= 只用本项目 ✓）")

    # ② 丢弃类：正常解析 + 写错要报错
    check(M.parse_drop("0,4") == [0, 4] and M.parse_drop("0，0,4") == [0, 4],
          "「丢弃类」解析不对：%r / %r" % (M.parse_drop("0,4"), M.parse_drop("0，0,4")))
    check(M.parse_drop("") == [],
          "「丢弃类」留空该是「一个都不丢」（[]），实得 %r" % (M.parse_drop(""),))
    for bad in ("o", "0,x", "玩家"):
        try:
            M.parse_drop(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "「丢弃类」写错（%r）居然没报错 ⇒ 会静默变成\"一个都不丢\"，"
                "训练集里还留着别人的玩家框 ✗" % bad)

    # ③④ 卡片分派（要 Qt；起不来就**说出来**再跳过，不静默变绿 ✓）
    try:
        from PyQt5.QtCore import Qt
        from PyQt5.QtTest import QTest
        from PyQt5.QtWidgets import QApplication
    except Exception as e:                 # noqa: BLE001
        print("   ⚠ 跳过卡片分派那两条：本机起不来 Qt（%s: %s）" % (type(e).__name__, e))
        return
    import gui.steps.cards as cards
    from gui.project import Project

    app = QApplication.instance() or QApplication([])      # noqa: F841
    root = Path(tempfile.mkdtemp(prefix="merge_card_"))
    try:
        proj_root = root / "主项目"
        proj_root.mkdir(parents=True)
        (proj_root / "project.yaml").write_text("name: 主项目\nmap_id: '1'\n",
                                                encoding="utf-8")
        # ⭐ 兄弟项目（候选下拉的选项就从这儿来 ✓ 2026-10-24：不许手打项目名 ✗）
        for name in ("外来项目", "另一个"):
            d = root / name
            d.mkdir()
            (d / "project.yaml").write_text("name: %s\nmap_id: '1'\n" % name,
                                            encoding="utf-8")
        p = Project.open(proj_root)
        c = cards.DatasetCard()
        c.bind(p)
        check(c.make_task(p)[0].__name__ == "run_dataset",
              "「并入其他项目」一个都不勾时该走老的 `run_dataset`，实得 %r"
              % (c.make_task(p)[0].__name__,))

        # ① 候选**从盘上来的**、且**不含自己**（并自己 = 同一张图进两次训练 ✗）
        #   ⚠ 第 0 行是**摘要行**，选项从第 1 行起 ✗（别把摘要当候选 ✓）
        _w = c.widgets["from_projects"][0]
        _items = [_w._model.item(i).text() for i in range(1, _w._model.rowCount())]
        check(set(_items) == {"外来项目", "另一个"},
              "候选项目下拉该是「兄弟项目」（不含自己）—— 实得 %r ⇒ 还是得手打 ✗"
              % (_items,))

        c.set_value("from_projects", ["外来项目", "另一个"])
        # ⭐ 类别那两件事现在只有**一对**：方式（丢 / 只取）+ 勾选哪些类
        #   （用户 2026-10-24："**为什么需要同时配置「外来项目只取这些类」「外来项目丢弃类」**"
        #    ⇒ 答：不需要，已合并 ✓）
        c.set_value("class_mode", cards.CLASS_MODE_KEEP)
        c.set_value("class_ids", [1])          # 白名单：只取怪物
        # ①' 框里得有**人看得懂**的摘要（用户要的就是"别让我记参数" ✓）
        #   ⚠ 摘要走的是**第 0 行"摘要行"** ⇒ 读 `currentText()`（不是 `lineEdit()` ✗
        #     非可编辑下拉根本没有 lineEdit —— 那也是它**不会**再被 Qt 清空的原因 ✓）
        _txt = _w.currentText()
        check(("已选 2" in _txt) and ("外来项目" in _txt),
              "多选下拉里没显示\"已选 2 项：…\"（实得 %r）⇒ 勾完看不出选了什么 ✗" % (_txt,))
        # ①'' 重填候选（换项目时会重填 ✓）**不能把已勾的清掉** ✗
        _w.set_items([("外来项目", "外来项目"), ("另一个", "另一个")])
        check(set(_w.checked()) == {"外来项目", "另一个"},
              "重填候选之后勾选被清掉了（实得 %r）⇒ 换项目就白点 ✗" % (_w.checked(),))

        # ①''' ⭐ **点框里任何地方都要能展开**（用户 2026-10-24 现场："**点选没反应**" ✓）：
        #   老实现（可编辑下拉）的文字区是个 `QLineEdit` 子控件，它把点击吃掉了 ✗
        #   ⇒ 只有右边小箭头能展开 ✓。现在**不用可编辑模式** ⇒ 框里根本没有子控件 ✓
        #   （⚠ 这条也是"原生崩溃"那件事的护栏：`setEditable(True)` 不许回来 ✗）
        check(_w.lineEdit() is None,
              "下拉框又变成可编辑模式了（`lineEdit()` 存在）⇒ 点击会被吃掉，"
              "而且主窗口反复开关会原生崩溃 ✗")
        check(_w.childAt(_w.rect().center()) is None,
              "下拉框中间那块还有子控件在吃点击（实得 %r）⇒ 用户点文字区没反应 ✗"
              % (_w.childAt(_w.rect().center()),))
        _w._popup_shown = False
        QTest.mousePress(_w, Qt.LeftButton)
        check(bool(_w._popup_shown), "点一下没展开列表 ⇒ 还是「点选没反应」 ✗")
        # ①'''' 点列表里的一项 ⇒ 勾上，**而且列表不收起**（多选要连着点 ✓）
        #   ⚠ 第 0 行是**摘要行**（藏着、不可勾）⇒ 必须点第 1 行 ✗
        _w.set_checked([])
        _w._on_pressed(_w._model.index(1, 0))
        check(bool(_w.checked()), "点了列表项却没勾上 ✗")
        check(bool(_w._popup_shown),
              "点一项就把列表收起来了 ⇒ 选 3 项要点开 3 次 ✗")

        # ①''''' ⭐⭐⭐ **摘要永远跟勾选一致**（用户 2026-10-24 现场："**我勾了还是显示
        #   （未选）**" ✓）：老实现是"我们写字、Qt 按 `currentIndex` 擦掉" ✗；
        #   现在摘要就是**当前项**（第 0 行）⇒ Qt 只可能把它**摆回**这一行 ✓。
        _w.set_checked(["外来项目"])
        check(_w.currentText().startswith("已选 1"),
              "勾一项后框里没显示「已选 1 项」（实得 %r）⇒ 勾了看不出来 ✗"
              % (_w.currentText(),))
        # 模拟 Qt 把当前项挪到某个选项上（它真的会这么干：点列表项时 ✓）
        _w.setCurrentIndex(1)
        check(_w.currentText().startswith("已选 1"),
              "当前项被挪到选项上之后框里显示成了选项文字（实得 %r）⇒ "
              "和摘要对不上，正是现场那个 bug ✗" % (_w.currentText(),))
        _w.showPopup()
        _w.hidePopup()
        app.processEvents()
        check(_w.currentText().startswith("已选 1"),
              "开关一次弹窗之后摘要就丢了（实得 %r）⇒ 现场那个 bug ✗"
              % (_w.currentText(),))

        # ①'''''' 就算勾选是**外部**改的（Qt delegate 点勾选框 / 键盘空格），摘要也得跟着变 ✓
        #   —— 现场那个"对不上"的根子就是"只有我们自己的代码才刷摘要" ✗
        _w.set_checked(["外来项目", "另一个"])
        _w._model.item(1).setCheckState(Qt.Unchecked)
        check(_w.currentText().startswith("已选 1"),
              "勾选是外部改的（直接改模型）时摘要没跟着变（实得 %r）⇒ 框里和列表对不上 ✗"
              % (_w.currentText(),))
        _w._model.item(1).setCheckState(Qt.Checked)      # 还原成 2 项（后面断言按它走）

        # 复原成"另一个"已勾的状态（后面断言按它走）
        _w.set_checked(["外来项目", "另一个"])
        check(c.value("class_ids") == [1],
              "多选下拉取回来的值不对（实得 %r）⇒ 勾了等于没勾 ✗"
              % (c.value("class_ids"),))
        fn, pr = c.make_task(p)
        check(fn.__name__ == "run_merge",
              "勾了「并入其他项目」却还走 `%s` ⇒ 界面上的功能没接上 ✗" % fn.__name__)
        check(pr.get("keep") == [1],
              "「只取勾选的类」没传下去（实得 %r）⇒ 界面上勾了没用 ✗"
              % (pr.get("keep"),))
        check(pr.get("drop") == [],
              "白名单方式下不该再塞一份 drop（实得 %r）⇒ 两套规则同时生效 ✗"
              % (pr.get("drop"),))
        check(set(pr["from"]) == {"外来项目", "另一个"},
              "传下去的项目列表不对：%r" % (pr.get("from"),))
        check(Path(pr["into"]).resolve() == proj_root.resolve(),
              "`into` 该是**本项目**（数据集落在本项目下 ✓），实得 %r" % (pr.get("into"),))
        check(pr.get("build") is True,
              "卡片这条路必须 `build=True`（否则只暂存、不产数据集 ⇒ 界面上看不到结果 ✗）")

        # ② ⭐ 换成"丢弃"方式 ⇒ 同一份勾选换个读法（这次要冒号后面的 drop ✓）
        c.set_value("class_mode", cards.CLASS_MODE_DROP)
        c.set_value("class_ids", [0, 4])
        _fn, pr2 = c.make_task(p)
        check(pr2.get("keep") == [] and pr2.get("drop") == [0, 4],
              "「丢弃勾选的类」该传 drop=[0,4]、keep=[]，实得 keep=%r drop=%r ✗"
              % (pr2.get("keep"), pr2.get("drop")))

        # ②' ⚠ **丢弃方式下一个都不勾** ⇒ 必须真的一个都不丢（`drop=[]` 而不是"没传"✓
        #     —— 这里曾经会静默回落到"默认丢玩家" ✗，用户勾空了却还在丢，看不出来 ✓）
        c.set_value("class_ids", [])
        _fn, pr3 = c.make_task(p)
        check(pr3.get("drop") == [],
              "一个都不勾，却还传了 drop=%r ⇒ 会「偷偷」继续丢玩家 ✗"
              % (pr3.get("drop"),))

        # ②'' 白名单方式下一个都不勾 ⇒ 外来项目**一类都不要**（全靠"全丢"表达 ✓）
        c.set_value("class_mode", cards.CLASS_MODE_KEEP)
        _fn, pr4 = c.make_task(p)
        _all = [i for i, _t in cards._class_choices()]
        check(sorted(pr4.get("drop") or []) == sorted(_all),
              "白名单方式下一个都不勾，该把外来项目的类**全丢**（让它们一帧都不进）—— "
              "实得 drop=%r ✗" % (pr4.get("drop"),))
        c.set_value("class_mode", cards.CLASS_MODE_KEEP)
        c.set_value("class_ids", [1])

        # ② 本项目还没标注、但勾了别的项目 ⇒ **不许拦**（那几份标注也是素材 ✓）
        ok, why = c.check_deps(p)
        check(bool(ok), "勾了「并入其他项目」还被 `check_deps` 拦：%s ⇒ 界面上跑不起来 ✗" % why)

        # ③ 参数随项目存（用户 2026-10-23 那条规矩 ✓）：存的是**列表**、读回来还能勾上
        c.sync(p)
        _saved = p.sec("dataset").get("from_projects")
        check(set(_saved) == {"外来项目", "另一个"},
              "「并入其他项目」没按列表存进 project.yaml 的 dataset 段（实得 %r）⇒ 换项目/重启就丢 ✗"
              % (_saved,))
        c2 = cards.DatasetCard()
        c2.bind(p)
        check(set(c2.value("from_projects")) == {"外来项目", "另一个"},
              "重新打开项目后勾选没回填（实得 %r）⇒ 每次都要重勾 ✗"
              % (c2.value("from_projects"),))
        # ③' 存的是**新那对**（方式 + 勾选），且**老的两格被删掉**（留着会让人以为还在用 ✗）
        _sec = p.sec("dataset")
        check(_sec.get("class_mode") == cards.CLASS_MODE_KEEP
              and _sec.get("class_ids") == [1],
              "「方式 + 勾选」没存进 project.yaml（实得 mode=%r ids=%r）⇒ 重启就丢 ✗"
              % (_sec.get("class_mode"), _sec.get("class_ids")))
        check("keep_classes" not in _sec and "drop_classes" not in _sec,
              "老的两格还留在 project.yaml 里（%r）⇒ 界上一堆没用的键，以后准有人照着改 ✗"
              % ([k for k in ("keep_classes", "drop_classes") if k in _sec],))

        # ④ **老项目迁移**：盘上只有老键（`drop_classes` 是字符串 "0,4"）⇒ 打开要变成
        #    "丢弃 + [0,4]"（不能炸、不能空、更不能把设置丢掉 ✗）
        p.sec("dataset").pop("class_mode", None)
        p.sec("dataset").pop("class_ids", None)
        p.sec("dataset")["drop_classes"] = "0,4"
        c3 = cards.DatasetCard()
        c3.bind(p)
        check(c3.value("class_mode") == cards.CLASS_MODE_DROP
              and c3.value("class_ids") == [0, 4],
              "老项目（只有 drop_classes=\"0,4\"）没迁成「丢弃 + [0,4]」—— "
              "实得 mode=%r ids=%r ⇒ 打开就丢设置 ✗"
              % (c3.value("class_mode"), c3.value("class_ids")))
        # ④' 老键是**白名单**（keep_classes）⇒ 迁成「只取」方式（白名单更窄，优先 ✓）
        p.sec("dataset").pop("drop_classes", None)
        p.sec("dataset")["keep_classes"] = [1]
        c4 = cards.DatasetCard()
        c4.bind(p)
        check(c4.value("class_mode") == cards.CLASS_MODE_KEEP
              and c4.value("class_ids") == [1],
              "老项目（只有 keep_classes=[1]）没迁成「只取 + [1]」—— 实得 mode=%r ids=%r ✗"
              % (c4.value("class_mode"), c4.value("class_ids")))
    finally:
        shutil.rmtree(root, ignore_errors=True)


TESTS = [
    ("⭐⭐⭐⭐ 多项目合并：外来项目剔掉「玩家」类、同名帧不串台、空帧跳过、端到端可训",
     t_merge_projects),
    ("⭐⭐⭐ ⑥ 数据集卡那几格：**多选下拉**（候选来自磁盘、不丢已勾的、有摘要）、"
     "解析不猜、留空仍是老路、随项目存", t_card_fields),
]


def main() -> int:
    failed = 0
    for name, fn in TESTS:
        try:
            fn()
        except Exception as e:                 # noqa: BLE001 —— 自检要把错因打全 ✓
            failed += 1
            print("[FAIL] %s\n       %s: %s" % (name, type(e).__name__, e))
        else:
            print("[ OK ] %s" % name)
    print("\n%d/%d 通过" % (len(TESTS) - failed, len(TESTS)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
