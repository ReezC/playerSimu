"""被控机部署台自检：参数表 / 命令行 / 配置 / 自检项 / 卡片界面。

**为什么要有它**
    「A 机不敲命令，所有操作走部署台」之后，A 机上的东西**全都通过这张参数表**
    落到命令行。改这里最容易出的三类错都不会报错、只会在现场表现为怪现象：

      · 参数表加了键、忘记加进 `config.DEFAULTS` → 界面能填，但一保存就被
        `_deep_merge` 丢掉，下次打开又变回默认（「我明明框了区域」）；
      · 界面能改、`build_cmd` 忘了带 → 卡片上显示得清清楚楚，命令里没有；
      · 框选区域解析不出来时**猜**一个值 → 推一片无关画面出去，B 机「底图对不上」，
        看着像寻路坏了。

    这些都在这里固化成断言。跑法：

        python -m tools.selftest_deploy        # 全过返回 0，有失败返回 1

**不碰真实配置**：只读 `config/deploy.json`（不存在就不读），
所有构造都用 `config.DEFAULTS` 的副本，不写任何文件。
"""

import copy
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # 界面部分不开真窗口
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deploy import config as dcfg                        # noqa: E402
from deploy import selfcheck, services                   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MMAP = {"bind": "0.0.0.0", "port": 5003, "zoom": 3, "fps": 10, "quality": 100,
        "x": 100, "y": 40, "w": 200, "h": 150}
MMAP_NO_REGION = dict(MMAP, x=None, y=None, w=None, h=None)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


# ---------------------------------------------------------------- 参数表

def t_service_table():
    """ORDER 里的每个服务都得有标题/副标题/参数表，且命令拼得出来。"""
    for key in services.ORDER:
        for table, name in ((services.TITLE, "TITLE"), (services.SUB, "SUB"),
                            (services.PARAMS, "PARAMS")):
            check(key in table, "%s 里缺 %s" % (name, key))
        check(services.PARAMS[key], "%s 的参数表是空的" % key)
        for spec in services.PARAMS[key]:
            for field in ("key", "keys", "label", "kind", "tip"):
                check(field in spec, "%s 的参数 %r 缺字段 %s"
                      % (key, spec.get("key"), field))
        cmd = services.build_cmd(key, copy.deepcopy(dcfg.DEFAULTS[key]))
        check(cmd and all(isinstance(a, str) for a in cmd),
              "%s 的 build_cmd 没拼出命令行" % key)
        check(services.format_cmd(cmd).count('"') % 2 == 0,
              "%s 的命令行引号没配对" % key)
    check("mmap" in services.ORDER, "ORDER 里没有小地图推流（mmap）")


def t_mmap_cmd():
    """小地图推流的命令：参数要真的带上，区域没框就别瞎传。"""
    cmd = services.build_cmd("mmap", MMAP)
    txt = " ".join(cmd)
    check("tools.minimap_push" in txt, "没拼上 tools.minimap_push")
    for want in ("--port 5003", "--zoom 3", "--fps 10", "--quality 100",
                 "--x 100", "--y 40", "--w 200", "--h 150"):
        check(want in txt, "命令行里缺 %r：%s" % (want, txt))
    check("--bind 0.0.0.0" in txt, "命令行里没带监听地址")

    # 没框区域：命令里就不该有 --x（宁可启动被拦住，也不传个猜的区域）
    txt2 = " ".join(services.build_cmd("mmap", MMAP_NO_REGION))
    check("--x" not in txt2, "没框区域却传了 --x：%s" % txt2)
    check("--port 5003" in txt2, "没框区域时其它参数也该照传")

    # 端口是字符串化过的数字（subprocess 只吃 str）
    check(all(isinstance(a, str) for a in cmd), "命令行里有非字符串参数")


def t_region_set():
    """region_set 的边界：缺一个、空串、非数字都不算填好。"""
    check(services.region_set(MMAP), "填全了的区域被判成没填")
    for bad in (dict(MMAP, x=None), dict(MMAP, h=""), dict(MMAP, w="abc"),
                dict(MMAP, y="1.5"), {}):
        check(not services.region_set(bad), "这些该判成没填好：%r" % (bad,))


def t_missing_hint():
    """启动前的拦截：没框区域要说「框选」，框小了/负数也要说清楚。"""
    hint = services.missing_hint("mmap", MMAP_NO_REGION)
    check(hint and "框选" in hint, "没框区域时没提示去框选：%r" % hint)
    check(services.missing_hint("mmap", MMAP) == "", "正常的区域不该被拦")
    check("太小" in services.missing_hint("mmap", dict(MMAP, w=5, h=5)),
          "区域太小没被拦")
    check("负" in services.missing_hint("mmap", dict(MMAP, x=-3)),
          "负数坐标没被拦")


# ---------------------------------------------------------------- 配置

def t_config_defaults():
    """DEFAULTS 必须含 mmap 的每个键 —— 否则界面填了也会被 _deep_merge 丢掉。"""
    # 「区域」这一个参数摊平成四个键（x/y/w/h），其余参数一对一
    one_to_one, expanded = set(), set()
    for spec in services.PARAMS["mmap"]:
        (expanded if spec["kind"] == "region" else one_to_one).update(spec["keys"])
    check(one_to_one | expanded == set(dcfg.DEFAULTS["mmap"]),
          "参数表和 DEFAULTS 的键对不上：参数表 %s vs DEFAULTS %s"
          % (sorted(one_to_one | expanded), sorted(dcfg.DEFAULTS["mmap"])))
    check(expanded, "小地图区域参数没摊平成坐标键")


def t_config_merge():
    """_deep_merge 只认 DEFAULTS 里的键：新加的键必须在，垃圾键必须被丢。"""
    merged = dcfg._deep_merge(copy.deepcopy(dcfg.DEFAULTS),
                              {"mmap": {"w": 321, "垃圾键": 1, "port": 5003}})
    check(merged["mmap"]["w"] == 321, "配置里的 w 没被读进来")
    check("垃圾键" not in merged["mmap"], "多余的键没被丢掉")
    check(merged["mmap"]["x"] is None, "缺的键没落回默认值")


def t_link_expect():
    """link.yaml 里的小地图端口要进一致性自检，不一致要报出来。"""
    expect = dcfg.link_expect()
    check("小地图端口" in expect, "link_expect 里没有小地图端口：%s" % list(expect))
    want, where = expect["小地图端口"]
    check(where == "mmap.port", "自检里写的路径不对：%s" % where)

    cfg = copy.deepcopy(dcfg.DEFAULTS)
    cfg["mmap"]["port"] = int(want) if want is not None else 5003
    rows = selfcheck.check_link(cfg, expect)
    # 只关心小地图这一项：别的项（推流帧率/串口）本来就可能和 link.yaml 不一致，
    # 那是现场参数，不是这次要验的东西。
    mine = [r for r in rows if "小地图端口" in r["title"]]
    check(not mine, "端口一致时不该报小地图端口：%s" % mine)

    cfg["mmap"]["port"] = int(cfg["mmap"]["port"]) + 1
    rows = selfcheck.check_link(cfg, expect)
    mine = [r for r in rows if "小地图端口" in r["title"]]
    check(mine and mine[0]["level"] == "warn",
          "端口不一致时没报出来：%s" % rows)


def t_check_region():
    """自检里的小地图区域项：没框=warn（不是 bad），填错=bad，框好=ok。"""
    lv = lambda c: selfcheck.check_region(c)[0]["level"]        # noqa: E731
    check(lv({}) == "warn", "没框区域应该是 warn（不用寻路的人不该看到红）")
    check(lv(dict(MMAP)) == "ok", "框好的区域应该是 ok")
    check(lv(dict(MMAP, w="abc")) == "bad", "填错应该是 bad")
    check(lv(dict(MMAP, w=5, h=5)) == "warn", "太小应该是 warn")


def t_read_real_config():
    """真实配置文件（存在的话）能读出来，且带上了 mmap 段。"""
    if not dcfg.PATH.exists():
        print("      （config/deploy.json 不存在，跳过）")
        return
    cfg = dcfg.read()
    check("mmap" in cfg, "读出来的配置里没有 mmap 段")
    for k in ("port", "zoom", "fps", "quality", "x", "y", "w", "h"):
        check(k in cfg["mmap"], "mmap 段里缺 %s" % k)


# ---------------------------------------------------------------- 界面（离屏）

def t_app_source():
    """部署台源码层面的两条约束（比建窗口更早暴露问题）。"""
    src = (ROOT / "tools" / "minimap_push.py").read_text(encoding="utf-8")
    check("def ask_region(" in src, "minimap_push 里没有 ask_region")
    # 只查**代码行**（顶格缩进的语句）：说明文字里会提到 app.exec_()，那是解释
    check("\n    app.exec_()" not in src,
          "框选又自己跑 app.exec_() 了 —— 在部署台里会卡住（事件循环已在跑）")
    check("hidden.hide()" in src and "hidden.show()" in src,
          "框选没有「先藏起调用方窗口再抓屏」")

    app_src = (ROOT / "deploy" / "app.py").read_text(encoding="utf-8")
    check('"mmap"' in app_src.split("SERVICE_COLOR")[1].split("}")[0],
          "SERVICE_COLOR 里没有 mmap")


def t_cards():
    """五张卡片都能建起来；小地图卡片有「框选…」且框完能落到配置里。"""
    from PyQt5.QtWidgets import QApplication, QPushButton
    from deploy.app import ServiceCard, _parse_region, _region_text

    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")

    for key in services.ORDER:
        card = ServiceCard(key, copy.deepcopy(dcfg.DEFAULTS[key]))
        check(card.cmd_text().strip(), "%s 卡片没算出「将执行」命令行" % key)

    card = ServiceCard("mmap", copy.deepcopy(dcfg.DEFAULTS))
    btns = [b.text() for b in card.findChildren(QPushButton)]
    check("框选…" in btns, "小地图卡片上没有「框选…」按钮：%s" % btns)

    # 空配置 → 输入框空、values() 是 None（会被 missing_hint 拦住）
    role, line, _getter = card._getters["region"]
    check(line.text() == "", "没框过区域时输入框应该是空的")
    check([card.values()[k] for k in role["keys"]] == [None] * 4,
          "没框过区域时不该给出坐标：%s" % card.values())

    # 框完（＝按钮写回文本）→ 四个键都落进配置，命令行也跟着变
    line.setText("100,40,200,150")
    got = card.values()
    check([got[k] for k in role["keys"]] == [100, 40, 200, 150],
          "框选结果没变成 x/y/w/h：%s" % got)
    card.refresh_cmd()
    check("--w 200" in card.lbl_cmd.text(),
          "框完之后「将执行」里没有区域：%s" % card.lbl_cmd.text())

    # 手输的容错：中文逗号/空格/小数都能认，垃圾则一律 None
    for text, want in (("100,40,200,150", (100, 40, 200, 150)),
                       ("100，40，200，150", (100, 40, 200, 150)),
                       ("100, 40, 200, 150", (100, 40, 200, 150)),
                       ("100,40,200", (None, None, None, None)),
                       ("", (None, None, None, None)),
                       ("a,b,c,d", (None, None, None, None))):
        check(_parse_region(text) == want,
              "%r 解析成 %s，应该是 %s" % (text, _parse_region(text), want))

    check(_region_text({"x": 1, "y": 2, "w": 3, "h": 4}) == "1,2,3,4",
          "_region_text 拼得不对")
    check(_region_text({"x": None, "y": 2, "w": 3, "h": 4}) == "",
          "缺键时 _region_text 该给空串")


def t_ask_region_modal():
    """框选能在一个**已有事件循环**的程序里跑完（部署台就是这种）。

    **为什么必须测**：老实现结尾是 `app.exec_()`，从部署台按钮里调会
    「event loop is already running」并卡住 —— 界面上按一下没反应，
    而命令行与部署台共用同一个函数，必须两边都能跑。

    做法是自动替人拉框：QTimer 里找出那个模态 QDialog，直接给它发鼠标事件。
    不用 `activeModalWidget()`（离屏平台下窗口拿不到激活，它可能是 None）。
    万一没关掉，watchdog 会兜住 —— 自检**绝不能挂住**。

    ⚠ 尺寸是 `QRect` 的**含两端**约定：从 (100,40) 拖到 (300,190) 覆盖的是
    201x151 个像素（x=100..300 全含），正好等于抓屏时 grabWindow 抓到的范围。
    """
    from PyQt5.QtCore import QEvent, QPointF, Qt, QTimer
    from PyQt5.QtGui import QMouseEvent
    from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow
    from tools.minimap_push import ask_region

    app = QApplication.instance() or QApplication([])
    win = QMainWindow()
    win.resize(400, 300)
    win.show()

    def find_dialog():
        for w in QApplication.topLevelWidgets():
            if isinstance(w, QDialog) and w.isVisible():
                return w
        return None

    seen = {}

    def drag(x1, y1, x2, y2):
        def go():
            dlg = find_dialog()
            if dlg is None:
                return                       # watchdog 会收场，断言随后报错
            seen["owner_hidden"] = not win.isVisible()
            for ev in (QMouseEvent(QEvent.MouseButtonPress, QPointF(x1, y1),
                                   Qt.LeftButton, Qt.LeftButton, Qt.NoModifier),
                       QMouseEvent(QEvent.MouseMove, QPointF(x2, y2),
                                   Qt.NoButton, Qt.LeftButton, Qt.NoModifier),
                       QMouseEvent(QEvent.MouseButtonRelease, QPointF(x2, y2),
                                   Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)):
                QApplication.sendEvent(dlg, ev)
        return go

    def watchdog():
        dlg = find_dialog()
        if dlg is not None:
            dlg.close()

    def run_drag(x1, y1, x2, y2, owner):
        QTimer.singleShot(150, drag(x1, y1, x2, y2))
        QTimer.singleShot(2500, watchdog)
        return ask_region(owner=owner)

    try:
        rect = run_drag(100, 40, 300, 190, win)
        check(rect == (100, 40, 201, 151),
              "框选返回 %s，应该是 (100,40,201,151)" % (rect,))
        check(seen.get("owner_hidden") is True,
              "框选时没把调用方窗口藏起来（会框到部署台自己）")
        check(win.isVisible(), "框选结束后没把调用方窗口放回来")

        # 太小的框 = 误点，当取消（宁可重框，也别推一个 3 像素的区域出去）
        check(run_drag(10, 10, 13, 13, win) is None, "太小的框没当成取消")
        check(win.isVisible(), "取消之后窗口没被放回来")

        # 命令行那条路：没有 owner 也要能框
        check(run_drag(50, 60, 150, 160, None) == (50, 60, 101, 101),
              "没有 owner 时框选结果不对")
    finally:
        win.close()


# ---------------------------------------------------------------- 跑

TESTS = (
    ("服务表（标题/参数/命令行）完整", t_service_table),
    ("小地图推流的命令行", t_mmap_cmd),
    ("区域填好没有的判据", t_region_set),
    ("启动前的拦截提示", t_missing_hint),
    ("DEFAULTS 与参数表同步", t_config_defaults),
    ("_deep_merge 只认已声明的键", t_config_merge),
    ("小地图端口进一致性自检", t_link_expect),
    ("自检里的小地图区域项", t_check_region),
    ("读真实 config/deploy.json", t_read_real_config),
    ("源码约束（框选不再自己跑事件循环）", t_app_source),
    ("卡片与框选控件（离屏）", t_cards),
    ("框选能在已有事件循环里跑（部署台那条路）", t_ask_region_modal),
)


def main() -> int:
    failed = 0
    for name, fn in TESTS:
        try:
            fn()
        except Exception as e:
            failed += 1
            print("[FAIL] %s\n       %s: %s" % (name, type(e).__name__, e))
        else:
            print("[ OK ] %s" % name)
    print("\n%d/%d 通过" % (len(TESTS) - failed, len(TESTS)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
