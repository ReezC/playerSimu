"""行为编辑器「保存 / 加载预设」自检（纯逻辑 + 离屏把窗口跑一遍）。

**为什么这些必须钉住**（用户 2026-09-26 定的 T1，见 docs/开发计划.md）：
预设是"换号换项目不用重敲"的救命功能，可它**坏起来是无声的**：
  · 加载坏文件时静默清空 ⇒ 用户以为加载成功了，其实序列没了 ⇒ **按键行为直接变了** ✗；
  · 保存漏了 `then` 嵌套 / `prob` ⇒ 存下来的和眼睛看到的不一样 ✗（回身输出的子序列
    丢了，角色就只按一下、不输出 ✗）；
  · 元素里写错一个字段名（`prob` 写成 `probs`）⇒ 那条动作**静默失效** ✗。
所以：存→读→一致、坏文件必报错**且不动当前内容**、未知字段必报错 —— 三条全钉死。

跑法：
    python -m tools.selftest_seq_editor        # 全过返回 0，有失败返回 1
"""

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import seq_presets as sp                            # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


#: 把三种元素 + `prob` + `then` 嵌套 + 中文自定义键名都用上（就是"回身输出"那种形状）
NESTED = [
    {"type": "down", "key": "back", "prob": 80, "then": [
        {"type": "down", "key": "attack"},
        {"type": "delay", "ms": 150},
        {"type": "up", "key": "attack"}]},
    {"type": "up", "key": "back"},
    {"type": "delay", "ms": 200},
    {"type": "down", "key": "自定义喊话键"},
]


def t_roundtrip_nested():
    """① 存 → 读 → **一致**（含 `then` 嵌套、`prob`、中文自定义键名）。"""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "回身输出.json"
        sp.save_preset(p, NESTED, name="回身输出", now="2026-09-26 15:04:22")
        info = sp.load_preset(p)
        check(info["name"] == "回身输出", "名字没存住：%r" % info["name"])
        check(info["saved_at"] == "2026-09-26 15:04:22", "存的时间没读回来")
        check(info["seq"] == NESTED,
              "存读不一致（嵌套 / prob / 中文键名）：\n%r\n%r" % (info["seq"], NESTED))
        # 文件得是**人能读**的：缩进、中文不转义、带说明
        txt = p.read_text(encoding="utf-8")
        check("回身输出" in txt and '"seq"' in txt, "文件不是人能读的样子：\n%s" % txt[:200])
        check("\\u" not in txt, "中文被转义成 \\uXXXX 了（那就没法直接手改）")
        check(txt.count("\n") > 5 and '  "seq"' in txt, "不是缩进好的 JSON")
        check(not list(Path(td).glob("*.tmp")), "留下了 .tmp 临时文件（原子写没做干净）")


def t_empty_and_normalize():
    """② 空序列 + 两处归一化（`prob`=100 / 空 `then` 不写出来）+ **不动入参**。"""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "empty.json"
        sp.save_preset(p, [])
        check(sp.load_preset(p)["seq"] == [], "空序列没能原样存读")
    check(sp.normalize([{"type": "down", "key": "a", "prob": 100}])
          == [{"type": "down", "key": "a"}], "prob=100 没归一化掉（会越存越啰嗦）")
    check(sp.normalize([{"type": "down", "key": "a", "then": []}])
          == [{"type": "down", "key": "a"}], "空 then 没归一化掉")
    check(sp.normalize([{"type": "delay", "ms": 0}]) == [{"type": "delay", "ms": 0}],
          "0 毫秒的延迟被吃掉了")
    check(sp.count(NESTED) == 7, "元素总数（含嵌套）算错：%s" % sp.count(NESTED))
    src = [{"type": "down", "key": "a", "prob": 100}]
    sp.normalize(src)
    check(src[0].get("prob") == 100, "normalize 把调用方的对象改了（该是深拷贝）")
    check(sp.is_valid(NESTED) and not sp.is_valid([{"type": "down"}]), "is_valid 判错")


def t_bad_files_all_raise():
    """③ 坏文件**必须报错、且说清哪一段坏**；坏序列**根本不许落盘**。"""
    cases = [
        ("不是 JSON", "{ this is not json", "JSON"),
        ("顶层不是对象", "[1, 2]", "对象"),
        ("没有 seq", '{"format": "playerSimu.sequence"}', "seq"),
        ("格式不对", '{"format": "别的东西", "seq": []}', "预设"),
        ("seq 不是列表", '{"seq": {"type": "down", "key": "a"}}', "列表"),
        ("元素不是对象", '{"seq": ["按一下"]}', "对象"),
        ("type 不认识", '{"seq": [{"type": "press", "key": "a"}]}', "type"),
        ("down 没有 key", '{"seq": [{"type": "down"}]}', "key"),
        ("key 是空的", '{"seq": [{"type": "down", "key": "   "}]}', "key"),
        ("delay 没有 ms", '{"seq": [{"type": "delay"}]}', "ms"),
        ("ms 是小数", '{"seq": [{"type": "delay", "ms": 1.5}]}', "ms"),
        ("ms 为负", '{"seq": [{"type": "delay", "ms": -1}]}', "ms"),
        ("ms 超上限", '{"seq": [{"type": "delay", "ms": %d}]}'
         % (sp.MAX_DELAY_MS + 1), "ms"),
        ("prob 超范围", '{"seq": [{"type": "down", "key": "a", "prob": 101}]}', "prob"),
        ("then 不是列表", '{"seq": [{"type": "down", "key": "a", "then": {}}]}', "then"),
        ("不认识的字段", '{"seq": [{"type": "down", "key": "a", "probs": 50}]}', "probs"),
        ("版本比程序新", '{"version": 99, "seq": []}', "版本"),
    ]
    with tempfile.TemporaryDirectory() as td:
        for name, body, frag in cases:
            p = Path(td) / "bad.json"
            p.write_text(body, encoding="utf-8")
            try:
                sp.load_preset(p)
            except ValueError as ex:
                check(frag in str(ex),
                      "「%s」的报错没说到点子上（该提到 %r）：%s" % (name, frag, ex))
            except Exception as ex:                       # noqa: BLE001
                raise AssertionError("「%s」抛的不是 ValueError（界面只接它）：%r"
                                     % (name, ex))
            else:
                raise AssertionError("「%s」居然读成功了 —— 坏文件必须报错 ✗" % name)
        # 嵌套里的坏元素要指到**具体路径**
        p = Path(td) / "deep.json"
        p.write_text('{"seq": [{"type": "down", "key": "a", "then": '
                     '[{"type": "delay", "ms": -5}]}]}', encoding="utf-8")
        try:
            sp.load_preset(p)
            raise AssertionError("嵌套里的坏元素没报错")
        except ValueError as ex:
            check("seq[0].then[0].ms" in str(ex),
                  "报错没指到哪一段坏（该含 seq[0].then[0].ms）：%s" % ex)
    # 文件不存在 ⇒ OSError（调用方照样只报错、不动内容）
    try:
        sp.load_preset(Path(tempfile.gettempdir()) / "这个预设不存在.json")
        raise AssertionError("文件不存在却没报错")
    except OSError:
        pass
    # 坏序列**根本不落盘**：不许把已经存好的预设毁掉
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "keep.json"
        sp.save_preset(p, [{"type": "down", "key": "a"}])
        before = p.read_text(encoding="utf-8")
        try:
            sp.save_preset(p, [{"type": "down"}])
            raise AssertionError("坏序列居然存下去了")
        except ValueError:
            pass
        check(p.read_text(encoding="utf-8") == before,
              "存坏序列把原来的预设改动了")


def t_safe_name_and_list():
    """④ 文件名安全化 + 列目录（读不了的**要列出来**，不许静默消失）。"""
    check(sp.safe_name("回身/输出:1") == "回身_输出_1",
          "文件名没安全化：%r" % sp.safe_name("回身/输出:1"))
    check(sp.safe_name("   ") == "行为预设", "空名字没兜底")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        sp.save_preset(d / "b.json", [], name="乙")
        sp.save_preset(d / "a.json", [], name="甲")
        (d / "broken.json").write_text("{ 坏", encoding="utf-8")
        rows = sp.list_presets(d)
        # 排序按**名字**；中文按码点（乙 U+4E59 < 甲 U+7532），读不了的也列出来
        check([n for n, _p in rows] == ["broken（读不了）", "乙", "甲"],
              "目录列表不对：%s" % [n for n, _p in rows])
        check(sp.list_presets(d / "（不存在的目录）") == [], "目录不存在时该给空列表")


def t_dialog_has_buttons():
    """⑤ 编辑器**上方**要有「保存」「加载」两个按钮（用户 2026-09-26 要求）。"""
    from PyQt5.QtWidgets import QApplication

    from gui.seq_editor import SeqEditorDialog

    app = QApplication.instance() or QApplication([])            # noqa: F841
    dlg = SeqEditorDialog(NESTED, title="回身输出")
    try:
        check(dlg.editor.btn_save.text() == "保存"
              and dlg.editor.btn_load.text() == "加载",
              "编辑器上没有「保存 / 加载」按钮")
        # 位置：**最上面**那一排就是它们
        first = dlg.editor.layout().itemAt(0)
        check(first is not None and first.layout() is not None
              and first.layout().itemAt(0).widget() is dlg.editor.btn_save,
              "「保存 / 加载」不在编辑器最上面")
        # 默认文件名跟着标题走（省得每存一次重新打一遍）
        check(dlg.editor.preset_name == "回身输出",
              "默认预设名没跟着弹窗标题：%r" % dlg.editor.preset_name)
        check(dlg.seq() == NESTED, "弹窗里的序列和传进来的不一致（round-trip 断了）")
    finally:
        dlg.reject()


def t_load_replaces_and_bad_keeps():
    """⑥ 加载：好文件**整体替换**；坏文件**只报错、当前内容一个字不动**。

    ⚠ 这是本功能的**保命**那一条：静默清空 = 用户以为加载成功、其实序列没了
    ⇒ 按键行为直接变了 ✗（T1 判据里写死的那句）。
    """
    from PyQt5.QtWidgets import QApplication, QMessageBox

    from gui import seq_editor as se

    app = QApplication.instance() or QApplication([])             # noqa: F841
    warned, infos, asked = [], [], []
    old = (QMessageBox.warning, QMessageBox.information, QMessageBox.question)
    # 离屏跑：任何**模态**框都会把测试挂住 ⇒ 三个都换成记录器
    # （⚠ 别漏 `question`：加载时会先问"要不要整体替换"，不接住就是无限等待 ✗）
    QMessageBox.warning = staticmethod(
        lambda *a, **k: warned.append(a[2] if len(a) > 2 else ""))
    QMessageBox.information = staticmethod(
        lambda *a, **k: infos.append(a[2] if len(a) > 2 else ""))
    QMessageBox.question = staticmethod(
        lambda *a, **k: (asked.append(a[2] if len(a) > 2 else ""), QMessageBox.Yes)[1])
    try:
        with tempfile.TemporaryDirectory() as td:
            good = Path(td) / "好.json"
            sp.save_preset(good, [{"type": "down", "key": "shop"},
                                  {"type": "up", "key": "shop"}], name="商城")
            bad = Path(td) / "坏.json"
            bad.write_text('{"seq": [{"type": "delay", "ms": -3}]}', encoding="utf-8")

            w = se.SeqEditorWidget([{"type": "down", "key": "attack"}])
            try:
                # 坏文件：只报错，**不许动**当前序列
                check(w._apply_preset_file(str(bad)) is False, "坏文件居然报了成功")
                check(warned, "坏文件没弹警告（静默失败 = 最坏的那种）")
                check(w.seq() == [{"type": "down", "key": "attack"}],
                      "坏文件把当前序列**改动/清空了** ✗：%r" % w.seq())
                # 好文件：整体替换
                check(w._apply_preset_file(str(good)) is True, "好文件没加载成功")
                check(w.seq() == [{"type": "down", "key": "shop"},
                                  {"type": "up", "key": "shop"}],
                      "加载后内容不对：%r" % w.seq())
                check(w.preset_name == "商城",
                      "加载后默认预设名没跟上：%r" % w.preset_name)
                check(asked, "整体替换前没问一声（当前序列非空，该先确认）")
                # 是**替换**不是叠加：原来那条 attack 不许留着
                check(all(e.get("key") != "attack" for e in w.seq()),
                      "加载成了「叠加」而不是「替换」✗：%r" % w.seq())
                # 存出去的能读回来（走的是同一套代码）
                out = Path(td) / "再存一份.json"
                sp.save_preset(out, w.seq(), name="商城")
                check(sp.load_preset(out)["seq"] == w.seq(), "存了又读，内容变了")
            finally:
                w.deleteLater()
    finally:
        (QMessageBox.warning, QMessageBox.information,
         QMessageBox.question) = old


# ⚠ 这里原来是 `TESTS = (...)` 的**表头**，已经挪到文件末尾（`main()` 之前）——
#    因为 `TESTS` 的每一项在**模块导入时**就要求值 ⇒ 它引用的用例函数必须**先定义** ✓。
#    （我把它放到函数前面过一次 ⇒ 当场 `undefined name` ✗。）


def t_seq_editor_add_down_pairs_up():
    """行为编辑器：加「键按下」要**自动配一条同键的「键松开」**（用户 2026-09-26 要求）。

    为什么钉：只按不松是最常见的坑（键卡住、后面动作按不出来 ✗），而这条是"加了就该生效"
    的手感 —— 顺序反了 / 配错键 / 接错位置，光看界面都不容易立刻发现 ✓。
    """
    from PyQt5.QtWidgets import QApplication

    from gui.seq_editor import SeqEditorWidget

    app = QApplication.instance() or QApplication([])            # noqa: F841
    w = SeqEditorWidget([])
    try:
        w._pick_key = lambda: "shop"          # 不弹选键框
        w._add_down()
        check(w.seq() == [{"type": "down", "key": "shop"},
                          {"type": "up", "key": "shop"}],
              "加「键按下」没自动配同键的「键松开」（或者顺序反了）：%r" % (w.seq(),))
        # 再来一次 ⇒ 接在**这一对之后**（顺序天然是 down/up、down/up ✓）
        w._add_down()
        check([e["type"] for e in w.seq()] == ["down", "up", "down", "up"],
              "第二对没接在后面 / 顺序乱了：%r" % (w.seq(),))
        # 「＋键松开」按钮本身还在（手动只加半对仍然可以 ✓）
        w._add_up()
        check(w.seq()[-1]["type"] == "up", "「＋键松开」按钮被弄坏了：%r" % (w.seq(),))
    finally:
        w.deleteLater()


def t_seq_editor_double_click_edits_by_type():
    """**双击元素 ⇒ 按类型弹编辑窗**（用户 2026-09-26 要求）：`delay` ⇒ 改毫秒、键 ⇒ 改键。

    为什么钉：这两个"改"以前**只能删了重加** ✗（右键菜单里只有几率 / 触发后执行 ✓）；
    而且改完要**三处一起更新**（存的字典、行文本、配色）—— 漏掉写回就是"看着改了、其实
    没改"（`item.data()` 返回的是**副本** ✓）。
    """
    from PyQt5.QtWidgets import QApplication, QInputDialog

    from gui import seq_editor as se

    app = QApplication.instance() or QApplication([])            # noqa: F841
    w = se.SeqEditorWidget([{"type": "down", "key": "shop"},
                            {"type": "delay", "ms": 200}])
    try:
        # ① delay ⇒ 弹的是"毫秒"窗；改完字典 + 行文本都要变
        old = QInputDialog.getInt
        QInputDialog.getInt = staticmethod(lambda *a, **k: (1500, True))
        try:
            w._on_double_click(w.tree.topLevelItem(1))
        finally:
            QInputDialog.getInt = old
        check(w.seq()[1] == {"type": "delay", "ms": 1500},
              "双击 delay 没改成新的毫秒数：%r" % (w.seq(),))
        check("1500ms" in w.tree.topLevelItem(1).text(0),
              "行文本没跟着改（看着还是旧值 ✗）：%r" % w.tree.topLevelItem(1).text(0))
        # ② 键元素 ⇒ 弹的是选键框；改完字典 + 行文本都要变
        w._pick_key = lambda: "esc"
        w._on_double_click(w.tree.topLevelItem(0))
        check(w.seq()[0] == {"type": "down", "key": "esc"},
              "双击键元素没改成新键：%r" % (w.seq(),))
        check("Esc" in w.tree.topLevelItem(0).text(0),
              "键改了但行文本没变：%r" % w.tree.topLevelItem(0).text(0))
        # ③ 取消 ⇒ 一个字都不许动
        old2 = QInputDialog.getInt
        QInputDialog.getInt = staticmethod(lambda *a, **k: (0, False))
        try:
            w._on_double_click(w.tree.topLevelItem(1))
        finally:
            QInputDialog.getInt = old2
        check(w.seq()[1]["ms"] == 1500, "取消编辑却把值改了：%r" % (w.seq(),))
    finally:
        w.deleteLater()


TESTS = (
    ("编辑器：加「键按下」自动配同键「键松开」（顺序 / 位置 / 半对仍可手动加）",
     t_seq_editor_add_down_pairs_up),
    ("编辑器：双击元素按类型进编辑（delay⇒毫秒、键⇒改键；取消不动）",
     t_seq_editor_double_click_edits_by_type),
    ("预设：存 → 读 → 一致（含 then 嵌套 / prob / 中文键名，文件人能读）",
     t_roundtrip_nested),
    ("预设：空序列 + 归一化（prob=100 / 空 then 不写出来）+ 不动入参",
     t_empty_and_normalize),
    ("预设：坏文件必报错并指到哪一段（17 种）+ 坏序列不落盘", t_bad_files_all_raise),
    ("预设：文件名安全化 + 列目录（读不了的也要列出来）", t_safe_name_and_list),
    ("编辑器：最上面有「保存 / 加载」，默认名跟着标题", t_dialog_has_buttons),
    ("加载：好文件整体替换；坏文件只报错、当前内容一个字不动",
     t_load_replaces_and_bad_keeps),
)


def main():
    n_ok = 0
    for name, fn in TESTS:
        try:
            fn()
        except Exception as ex:                                  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print("[NG] %s\n     %s" % (name, ex))
        else:
            n_ok += 1
            print("[OK] %s" % name)
    print("行为编辑器预设自检：%d/%d 通过" % (n_ok, len(TESTS)))
    return 0 if n_ok == len(TESTS) else 1


if __name__ == "__main__":
    sys.exit(main())
