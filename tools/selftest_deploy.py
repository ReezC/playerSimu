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
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # 界面部分不开真窗口
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deploy import config as dcfg                        # noqa: E402
from deploy import selfcheck, services                   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
#: mmap 那张卡片的参数 fixture —— ⚠ **里面没有 x/y/w/h**（2026-10-02 起区域不再存
#: deploy.json ✗ 改按地图 id 存 `config/minimap_regions/<id>.json` ✓ 见 `_with_mmap_regions` ✓）。
# ⚠ 2026-10-03：fps/quality 默认值改成 **60 / 80**（用户批的 ✓ 小地图要尽量高帧率更新 ✓）
MMAP = {"bind": "0.0.0.0", "port": 5003, "zoom": 3, "fps": 60, "quality": 80}


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _with_mmap_regions(fn):
    """把 `tools.mmap_regions` 的**三个文件位置**指到临时目录里跑 `fn(mr)`。

    **绝不碰真实配置**（同本文件开头那条：所有构造都用副本、不写文件 ✓）：
    区域库（`DIR`）、老单值兜底、以及"当前图"那份记忆（`CURRENT_FILE`）全是模块级
    路径常量 ⇒ 换掉它们就够了 ✓（`selftest_mmap_regions` 也是这么干的 ✓）。
    """
    from tools import mmap_regions as mr
    tmp = Path(tempfile.mkdtemp(prefix="selftest-deploy-mmap-"))
    old = (mr.DIR, mr.LEGACY_REGION_FILE, mr.CURRENT_FILE)
    mr.DIR = tmp / "regions"
    mr.LEGACY_REGION_FILE = tmp / "legacy.json"
    mr.CURRENT_FILE = tmp / "current.json"
    try:
        return fn(mr)
    finally:
        mr.DIR, mr.LEGACY_REGION_FILE, mr.CURRENT_FILE = old
        shutil.rmtree(tmp, ignore_errors=True)


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
    """小地图推流的命令：**区域按"当前是哪张图"传**（用户 2026-10-02 ✓）。

    钉四件：
      ① 有当前图、且**那张图框过** ⇒ 带 `--map-id <id>`（区域与 zoom 都由
         `config/minimap_regions/<id>.json` 说了算 ⇒ 与界面显示的那份是同一份 ✓）；
      ② 那张图**还没框过** ⇒ 一个区域参数都不带（`--map-id` 也不带：带了 A 机会直接
         报错退出 ✗，而我们要的是"先起来等 B 机"✓）；
      ③ **不许**再有 `--x/--y/--w/--h`（老口径 ✗）—— 那会推一块"这张图无关"的画面 ✗；
      ④ 有 `--map-id` 时**不传 `--zoom`**（那张图那份里存着它自己的 zoom ✓）。
    """
    def run(mr):
        cmd0 = services.build_cmd("mmap", copy.deepcopy(MMAP))
        txt0 = " ".join(cmd0)
        check("tools.minimap_push" in txt0, "没拼上 tools.minimap_push")
        for want in ("--port 5003", "--fps 60", "--quality 80", "--bind 0.0.0.0"):
            check(want in txt0, "命令行里缺 %r：%s" % (want, txt0))
        # 还没有"当前图" ⇒ 不带任何区域参数（服务照起、等 B 机 MAP ✓）
        check("--map-id" not in txt0, "还不知道是哪张图就传了 --map-id：%s" % txt0)

        # ② 知道图了，但那图还没框过 ⇒ 照样不带区域参数（别让 A 机一起就退 ✗）
        mr.set_current("105040303", by="B机")
        txt1 = " ".join(services.build_cmd("mmap", copy.deepcopy(MMAP)))
        check("--map-id" not in txt1,
              "这张图还没框过却传了 --map-id —— A 机会直接报错退出，"
              "而我们要的是「先起来等 B 机」：%s" % txt1)

        # ①③④ 框过了 ⇒ 只带 --map-id（区域/zoom 都在那份文件里 ✓）
        mr.save("105040303", 100, 40, 200, 150, zoom=4)
        cmd = services.build_cmd("mmap", copy.deepcopy(MMAP))
        txt = " ".join(cmd)
        check("--map-id 105040303" in txt, "框过了却没按地图 id 传区域：%s" % txt)
        for bad in ("--x", "--y", "--w", "--h", "--zoom"):
            check(bad not in txt,
                  "区域/zoom 又跑到命令行里了（%s）—— 该由那张图那份文件说了算 ✗：%s"
                  % (bad, txt))

        # 端口是字符串化过的数字（subprocess 只吃 str）
        check(all(isinstance(a, str) for a in cmd), "命令行里有非字符串参数")

    _with_mmap_regions(run)


def t_missing_hint():
    """启动前的拦截：**mmap 不再拦"还没框过"**（用户 2026-10-02 ✓），别的照旧。"""
    check(services.missing_hint("mmap", copy.deepcopy(MMAP)) == "",
          "小地图还没框过就被拦住了 —— 但用户定的流程是「先起来等 B 机告诉是哪张图」"
          "（B 机那句 MAP 必须走这条服务的连接 ⇒ 拦了它就没法闭环 ✗）")
    # 别的服务照旧拦（别把这条豁免顺手带过去 ✗）
    check(services.missing_hint("kbd", {}) != "", "键盘缺串口号该被拦住")
    check(services.missing_hint("push", {}) != "", "推流缺 ffmpeg 该被拦住")


# ---------------------------------------------------------------- 配置

def t_config_defaults():
    """DEFAULTS 必须含 mmap 的每个键 —— 否则界面填了也会被 _deep_merge 丢掉。

    ⭐ 2026-10-02：**区域那几个键不在 DEFAULTS 里了**（用户定"按地图 id 各存一份、
    且只存 A 机本地"✓）⇒ 这里反过来钉"它**不许**回来"✗（回来了就说明又有人把
    屏幕坐标塞进 deploy.json，那正是"换图还推上一张图"的老坑 ✓）。
    """
    keys = set()
    for spec in services.PARAMS["mmap"]:
        keys |= set(spec["keys"])
    check(keys == set(dcfg.DEFAULTS["mmap"]),
          "参数表和 DEFAULTS 的键对不上：参数表 %s vs DEFAULTS %s"
          % (sorted(keys), sorted(dcfg.DEFAULTS["mmap"])))
    for k in ("x", "y", "w", "h"):
        check(k not in dcfg.DEFAULTS["mmap"],
              "「%s」又回到 deploy.json 的 mmap 段里了 ✗ —— 区域现在按地图 id 存在 "
              "config/minimap_regions/<id>.json（A 机本地 ✓ 见 tools/mmap_regions.py）" % k)


def t_config_merge():
    """_deep_merge 只认 DEFAULTS 里的键：新加的键必须在，垃圾键必须被丢。"""
    merged = dcfg._deep_merge(copy.deepcopy(dcfg.DEFAULTS),
                              {"mmap": {"port": 5099, "垃圾键": 1, "x": 5, "w": 321}})
    check(merged["mmap"]["port"] == 5099, "配置里的 port 没被读进来")
    check("垃圾键" not in merged["mmap"], "多余的键没被丢掉")
    check("x" not in merged["mmap"] and "w" not in merged["mmap"],
          "老 deploy.json 里的 x/y/w/h 又被读进来了 ✗（区域已改按地图 id 存 ✓）")
    check(merged["mmap"]["zoom"] == dcfg.DEFAULTS["mmap"]["zoom"],
          "没填的键没落回默认值")


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
    """自检里的小地图区域项：**按"当前是哪张图"判**（用户 2026-10-02 ✓）三档都只 warn ✓。

    ① 还不知道是哪张图（B 机还没说）⇒ warn，且文案要说"去 B 机开始实时"；
    ② 知道图、但**这张图还没框过** ⇒ warn，文案要带上已配过哪些图（免得框错地方 ✗）；
    ③ 框过 ⇒ ok，带 id / 屏幕坐标 / 该图的 zoom ✓。
    ⚠ 都用临时目录（`_with_mmap_regions` ✓ 不碰真实配置 ✓）。
    """
    def run(mr):
        r = selfcheck.check_region({})[0]
        check(r["level"] == "warn" and "哪张图" in r["title"] and "B 机" in r["detail"],
              "还没有当前图时该 warn 并说清「去 B 机开始实时」：%r" % (r,))

        mr.set_current("105040303", by="B机")
        r = selfcheck.check_region({})[0]
        check(r["level"] == "warn" and "还没框过" in r["title"],
              "知道图但没框过该 warn：%r" % (r,))
        check("105040303" in r["detail"], "警告里没写是哪张图：%r" % (r,))

        mr.save("105040303", 10, 20, 200, 150, zoom=3)
        r = selfcheck.check_region({})[0]
        check(r["level"] == "ok" and "105040303" in r["title"],
              "框过了该给 ok（并且带上图 id）：%r" % (r,))
        check("zoom=3" in r["detail"], "ok 那行没带上该图的 zoom：%r" % (r,))

    _with_mmap_regions(run)


def t_read_real_config():
    """真实配置文件（存在的话）能读出来，且带上了 mmap 段。"""
    if not dcfg.PATH.exists():
        print("      （config/deploy.json 不存在，跳过）")
        return
    cfg = dcfg.read()
    check("mmap" in cfg, "读出来的配置里没有 mmap 段")
    for k in ("port", "zoom", "fps", "quality", "bind"):
        check(k in cfg["mmap"], "mmap 段里缺 %s" % k)
    # ⚠ 老 deploy.json 里可能还留着 x/y/w/h ⇒ **读进来必须没有它们**（区域改按地图 id 存 ✓）
    for k in ("x", "y", "w", "h"):
        check(k not in cfg["mmap"], "mmap 段里又有 %s 了（区域不该存 deploy.json ✗）" % k)


# ---------------------------------------------------------------- 界面（离屏）

def t_app_source():
    """部署台源码层面的两条约束（比建窗口更早暴露问题）。"""
    src = (ROOT / "tools" / "minimap_push.py").read_text(encoding="utf-8")
    check("def ask_region(" in src, "minimap_push 里没有 ask_region")
    # 只查**代码行**（顶格缩进的语句）：说明文字里会提到 app.exec_()，那是解释
    check("\n    app.exec_()" not in src,
          "框选又自己跑 app.exec_() 了 —— 在部署台里会卡住（事件循环已在跑）")
    # 「藏起调用方窗口再抓屏」现在归共用实现（gui/region_picker.py）管，
    # 这里只验**它确实走的是共用实现**（细节由 tools/selftest_region.py 钉）
    check("select_screen_region" in src,
          "部署台的框选没走全仓库唯一那份实现（gui.region_picker）")
    picker = (ROOT / "gui" / "region_picker.py").read_text(encoding="utf-8")
    check("hidden.hide()" in picker and "hidden.show()" in picker,
          "共用框选没有「先藏起调用方窗口再抓屏」")

    app_src = (ROOT / "deploy" / "app.py").read_text(encoding="utf-8")
    check('"mmap"' in app_src.split("SERVICE_COLOR")[1].split("}")[0],
          "SERVICE_COLOR 里没有 mmap")

    # 工具栏里的「设置…」：用户找字号只有这一条路。放源码层验（不用把整个部署台
    # 界面拉起来 —— 那个窗口一构造就会建卡片、起定时器、跑一次环境自检）。
    check('QAction("设置…"' in app_src, "工具栏里没有「设置…」按钮")
    check("from deploy.settings_dialog import DeploySettingsDialog" in app_src,
          "没导入设置弹窗")
    check("DeploySettingsDialog(self.cfg" in app_src,
          "「设置…」没把活的配置交给弹窗（弹窗收 cfg + on_saved，见其模块说明）")
    check("act_set.triggered.connect(self._on_settings)" in app_src,
          "「设置…」没接到 _on_settings")


def t_cards():
    """五张卡片都能建起来；小地图卡片有「框选…」+ 两行只读（当前地图 / 小地图区域）。"""
    from PyQt5.QtWidgets import QApplication, QPushButton
    from deploy.app import ServiceCard

    app = QApplication.instance() or QApplication([])
    check(app is not None, "建不起 QApplication")

    for key in services.ORDER:
        card = ServiceCard(key, copy.deepcopy(dcfg.DEFAULTS[key]))
        check(card.cmd_text().strip(), "%s 卡片没算出「将执行」命令行" % key)

    card = ServiceCard("mmap", copy.deepcopy(dcfg.DEFAULTS["mmap"]))
    btns = [b.text() for b in card.findChildren(QPushButton)]
    check("框选…" in btns, "小地图卡片上没有「框选…」按钮：%s" % btns)
    check("重读" in btns, "小地图卡片上没有「重读」（B 机说了新图之后立刻重读用 ✓）：%s" % btns)

    # 键盘卡片的「生成证书…」：A 机的原则是**不敲命令**（deploy/app.py 开头那条），
    # 而生成证书有个只做一半就很坑的分支（私钥留 A、cert.pem 必须拷到 B）——
    # 所以它得是个按钮，而且点了之后要把"拷哪一份"说清楚。
    card_kbd = ServiceCard("kbd", copy.deepcopy(dcfg.DEFAULTS["kbd"]))
    btns_kbd = [b.text() for b in card_kbd.findChildren(QPushButton)]
    check("生成证书…" in btns_kbd,
          "键盘卡片上没有「生成证书…」按钮：%s" % btns_kbd)

    # ⭐ 那两行**只读**（用户 2026-10-02）：还没有当前图时文案要说清"去 B 机"✓，
    #   而且**一个配置键都不产出**（区域不进 deploy.json ✓）
    def run(mr):
        card.refresh_displays()
        cur_w = card._getters["current"][1]
        reg_w = card._getters["region"][1]
        check("B 机" in cur_w.text(), "没有当前图时该说「先去 B 机」：%r" % cur_w.text())
        check("还没" in reg_w.text() and "图" in reg_w.text(),
              "还不知道哪张图时，区域那行该说清楚：%r" % reg_w.text())
        check(not (set(card.values()) & {"x", "y", "w", "h"}),
              "只读两行居然产出了区域键（区域不该进 deploy.json ✗）：%s" % card.values())
        check("--x" not in card.cmd_text() and "--map-id" not in card.cmd_text(),
              "还没有当前图时命令里就有区域了：%s" % card.cmd_text())

        # B 机告诉本机是哪张图了 ⇒ 两行当场变（refresh_displays 是"现读" ✓）
        mr.set_current("105040303", by="B机")
        card.refresh_displays()
        check("105040303" in cur_w.text() and "B机" in cur_w.text(),
              "「当前地图」没跟着当前图变：%r" % cur_w.text())
        check("还没框过" in reg_w.text(), "知道图但没框过时说错了：%r" % reg_w.text())

        # 框过了 ⇒ 区域那行给屏幕坐标 + zoom，而且命令行变成 --map-id ✓
        mr.save("105040303", 100, 40, 200, 150, zoom=4)
        card.refresh_displays()
        card.refresh_cmd()
        check("200x150" in reg_w.text() and "zoom=4" in reg_w.text(),
              "框过之后那行没给出坐标/zoom：%r" % reg_w.text())
        check("--map-id 105040303" in card.lbl_cmd.text(),
              "框过之后「将执行」里没按地图 id 传区域：%s" % card.lbl_cmd.text())

    _with_mmap_regions(run)


def t_ask_region_modal():
    """框选能在一个**已有事件循环**的程序里跑完（部署台就是这种）。

    **为什么必须测**：老实现结尾是 `app.exec_()`，从部署台按钮里调会
    「event loop is already running」并卡住 —— 界面上按一下没反应，
    而命令行与部署台共用同一个函数，必须两边都能跑。

    做法是自动替人拉框：**轮询等**那个模态 QDialog 出现，再给它发鼠标事件。
    （不能固定延时：抓屏那条路会先"藏起调用方窗口 + 等 250ms"，窗口出现得比
    固定延时要晚 —— 抢跑的话事件发给空气，看着就像"框选没返回"。）
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
        go = drag(x1, y1, x2, y2)
        tick = QTimer()
        tick.setInterval(30)
        waited = [0]

        def poll():
            if find_dialog() is None and waited[0] < 4000:
                waited[0] += 30          # 窗口还没出来（藏窗口 + 250ms），接着等
                return
            tick.stop()
            go()

        tick.timeout.connect(poll)
        tick.start()
        wd = QTimer()
        wd.setSingleShot(True)
        wd.setInterval(5000)
        wd.timeout.connect(watchdog)
        wd.start()
        try:
            return ask_region(owner=owner)
        finally:
            tick.stop()
            wd.stop()

    try:
        rect = run_drag(100, 40, 300, 190, win)
        check(rect == (100, 40, 201, 151),
              "框选返回 %s，应该是 (100,40,201,151)" % (rect,))
        check(seen.get("owner_hidden") is True,
              "框选时没把调用方窗口藏起来（会框到部署台自己）")
        check(win.isVisible(), "框选结束后没把调用方窗口放回来")

        # 太小的框 = 误点，当取消（宁可重框，也别推一个 2 像素的区域出去）。
        # 阈值是框选规范里的 4px，且尺寸按 QRect 含两端算：拖 (10,10)→(13,13)
        # 是 4×4（= 刚好合格），要更小才是误点，所以这里拖 (10,10)→(11,11)。
        check(run_drag(10, 10, 11, 11, win) is None, "太小的框没当成取消")
        check(win.isVisible(), "取消之后窗口没被放回来")

        # 命令行那条路：没有 owner 也要能框
        check(run_drag(50, 60, 150, 160, None) == (50, 60, 101, 101),
              "没有 owner 时框选结果不对")
    finally:
        win.close()


# ---------------------------------------------------------------- 设置弹窗

def t_settings_dialog_font():
    """设置弹窗的字号：**点确定才写配置**，拖动过程中只更新预览。

    为什么单列一条：字号是**两个界面共用一份**（config/ui.yaml）。一旦在拖动时就
    写盘，"点开看一眼又取消"也会把工作台那边的字号一起改掉 —— 这种"我没改啊"
    最难查。另外钉住"点确定后立即生效"：那是这个功能存在的全部意义（不用重启）。
    """
    import shutil
    import tempfile
    import unittest.mock as mock

    from PyQt5.QtWidgets import QApplication
    from deploy.settings_dialog import LOG_DEFAULT, DeploySettingsDialog
    from gui import theme

    app = QApplication.instance() or QApplication([])
    tmpdir = Path(tempfile.mkdtemp(prefix="depui_"))
    cfg = copy.deepcopy(dcfg.DEFAULTS)
    cfg["log"]["max_lines"] = LOG_DEFAULT
    saved = []
    try:
        with mock.patch.object(theme, "CFG", tmpdir / "ui.yaml"):
            theme.save_size(11)
            dlg = DeploySettingsDialog(cfg, on_saved=lambda: saved.append(1))
            # ① 拖动：只改预览，配置一个字节都不许动
            dlg.slider.setValue(16)
            check(theme.load_size() == 11,
                  "还没点确定就把字号写进配置了（取消也撤不回来）")
            check("16" in dlg.lbl_val.text(),
                  "数值标签没跟着滑块走：%r" % dlg.lbl_val.text())
            # ② 取消：什么都不该落
            dlg.reject()
            check(not saved, "取消也通知了落盘")
            check(theme.load_size() == 11, "取消之后字号变了")
            # ③ 确定：写配置 + 立即应用 + 通知窗口
            dlg2 = DeploySettingsDialog(cfg, on_saved=lambda: saved.append(1))
            dlg2.slider.setValue(16)
            dlg2.sp_lines.setValue(1200)
            dlg2._accept()
            check(theme.load_size() == 16, "点确定后字号没落盘")
            check(app.font().pointSize() == 16,
                  "点确定后没立即生效，字体还是 %d px" % app.font().pointSize())
            check(cfg["log"]["max_lines"] == 1200,
                  "日志行数没写进 cfg：%s" % (cfg["log"],))
            check(saved == [1], "确定后没通知窗口落盘：%s" % (saved,))

            # ★ 真正要钉住的那条：**界面上已经存在的控件**要跟着变。
            # 只断言 app.font() 是不够的 —— 实测过：打过样式表的控件在 setFont 之后
            # 一动不动（父窗口一份 QSS、里面一个字号都没写，子控件 11 → 11）。
            # "设置里改了、界面上没变"就是这么来的，所以这里用**真控件**量一次。
            from PyQt5.QtWidgets import QLabel, QVBoxLayout
            from PyQt5.QtWidgets import QWidget as _Widget
            holder = _Widget()
            holder.setStyleSheet("QLabel { color: #202124; }")
            hl = QVBoxLayout(holder)
            lab = QLabel("测试")
            hl.addWidget(lab)
            holder.show()
            theme.apply(app, 11)
            was = lab.font().pointSize()
            dlg3 = DeploySettingsDialog(cfg, on_saved=None)
            dlg3.slider.setValue(18)
            dlg3._accept()
            check(lab.font().pointSize() == 18,
                  "点确定后**已有控件**的字号没跟着变：%d → %d"
                  "（只有 app.font() 变了，界面看着一模一样）"
                  % (was, lab.font().pointSize()))
            holder.close()
    finally:
        theme.apply(app, theme.DEFAULT_SIZE)     # 别把字号留给后面的用例
        shutil.rmtree(str(tmpdir), ignore_errors=True)


def t_log_history_applies():
    """日志保留行数：**两处一起改**（历史缓冲 + 控件上限），旧行真的被丢掉。

    只改一处就会出现"切一下筛选能看见几百行前的日志、控件里却没有"（或反过来），
    这种"同一个面板两套内容"最难查。顺带钉住 DEFAULTS 里有这个键 —— 没声明的话
    会被 read() 的 _deep_merge 静默丢掉（见本文件开头的三类错）。
    """
    from PyQt5.QtWidgets import QApplication
    from deploy.app import LogPane

    # **必须把实例存下来**：QApplication 一旦被 Python 回收，后面建控件就是 qFatal
    # （"Must construct a QApplication before a QWidget"）—— 表现为进程**直接 abort**
    # （退出码 0xC0000409），而且 stdout 缓冲区一起丢掉，连一行报错都看不到。
    # 实测踩过：写成 `QApplication.instance() or QApplication([])`（不赋值）就是这样，
    # 查了半天才发现"崩在测试里"和"崩在退出时"长得一模一样。
    app = QApplication.instance() or QApplication([])
    check(app is not None, "没拿到 QApplication")
    check("max_lines" in dcfg.DEFAULTS["log"],
          "DEFAULTS 的 log 段没有 max_lines —— 存了也会被 _deep_merge 丢掉")

    pane = LogPane()
    try:
        pane.set_history(600)
        check(pane.txt.maximumBlockCount() == 600,
              "控件上限没跟着改：%d" % pane.txt.maximumBlockCount())
        for i in range(800):
            pane.append("ui", "line %d" % i)
        check(len(pane._buf) == 600, "历史缓冲没按新上限截断：%d" % len(pane._buf))
        text = pane.txt.toPlainText()
        check("line 799" in text, "最近的行反而被截掉了")
        check("line 200" in text, "边界那条丢了：应当正好保留最近 600 行")
        check("line 199" not in text, "旧行没被丢掉（上限没生效）")
    finally:
        pane.close()


def t_cards_show_cmd_at_startup():
    """**刚打开部署台时，每张卡的「将执行」就必须是满的。**

    实测（用户报的）：五张卡的「将执行」全空 ✗，随手碰一下任意参数才填上。
    根因是 `_rebuild_cards()` 重建卡片后没人调 `refresh_cmd()` —— 构造期间
    `changed` 信号还没连上（连的是重建之后的对象），所以也不会顺带触发；
    旁边那个 `_refresh_cmd_all()` 当时是**死代码**。

    为什么值得用真窗口测：这个标签是卡片存在的意义（"界面里能改的东西和命令里
    能传的东西是同一份定义"，见部署台 docstring）。空着不但没用，还误导 ——
    看着像"这个服务没配好"。命令行本身是好的（`cmd_text()` 现算），所以
    "能启动、能复制"都掩盖不了它。
    """
    from PyQt5.QtWidgets import QApplication
    from deploy.app import DeployWindow

    app = QApplication.instance() or QApplication([])
    win = DeployWindow()
    try:
        app.processEvents()
        empty = [k for k, c in win.cards.items() if not c.lbl_cmd.text().strip()]
        check(not empty, "启动后这些卡片的「将执行」是空的：%s" % (empty,))
        for k, c in win.cards.items():
            check(c.lbl_cmd.text() == c.cmd_text(),
                  "%s 卡片显示的「将执行」和现算的不一致（是占位串？）" % k)
    finally:
        win.close()


def t_sweep_guard_asks_for_probe_and_clock():
    """启动推流自检前**必须先查探针/时钟两张卡**，没起就问一句、并帮着起。

    实测（2026-09-25）：这两张卡没起，A 机照样跑满 18 段，B 机预检把整轮拒掉 ——
    A 机侧那几个数是有用的，但那一轮**延迟一个也没量到**，8 分钟白跑。
    """
    import unittest.mock as mock

    from PyQt5.QtWidgets import QApplication, QMessageBox
    from deploy.app import DeployWindow

    app = QApplication.instance() or QApplication([])

    # ① 点「否」：什么都不该发生（不起服务、不开跑）
    win = DeployWindow()
    try:
        with mock.patch.object(QMessageBox, "question") as q:
            q.return_value = QMessageBox.No
            win._start_push_sweep()
            # 参数是 (parent, 标题, 正文, ...) —— 要看**正文**里提没提那两张卡
            asked = str(q.call_args[0][2]) if q.call_args else ""
        check("探针" in asked, "问的时候没提到探针卡：%r" % asked)
        check("时钟" in asked, "问的时候没提到时钟卡：%r" % asked)
        check(not win.procs, "点了「否」却还是把服务起起来了")
        check(win._sweep is None, "点了「否」却开始跑了")
    finally:
        win.close()

    # ② 点「是」：起这两张卡 → **等就绪** → 才继续到确认框
    win2 = DeployWindow()
    try:
        started, ran = [], []
        # singleShot 要**立刻执行**回调：不然它把回调吞了，"等就绪"这条永远走不到后面
        # （第一版就是这么写的，自己绊自己一跳）。
        with mock.patch.object(QMessageBox, "question",
                               return_value=QMessageBox.Yes), \
                mock.patch.object(win2, "start_service",
                                  side_effect=lambda k: started.append(k)), \
                mock.patch("PyQt5.QtCore.QTimer.singleShot",
                           side_effect=lambda _ms, fn: fn()) as ss, \
                mock.patch.object(win2, "_confirm_and_run_sweep",
                                  side_effect=lambda: ran.append(1)):
            win2._start_push_sweep()
        check(sorted(started) == ["clock", "probe"],
              "点「是」时该起 clock+probe，实际起了 %s" % (started,))
        check(ss.called, "起完服务没等就绪就往下走（探针还没画上码带）")
        check(ran == [1], "起完服务后没走到确认框")
    finally:
        win2.close()


def t_bat_files_are_ascii():
    """仓库根的 .bat **必须是纯 ASCII**：cmd.exe 按控制台代码页读批处理，
    非 ASCII 的注释会被**当成命令执行**（现场就是一堆 mojibake 报错）。

    实测（2026-09-25）：`一键测延迟.bat` 里混了中文和 `─` 框线字符，双击后
    控制台把 `rem` 后面的片段当命令跑，报了三条 "'xx' is not recognized"，
    然后**照常往下跑**了 —— 所以这事不致命但很吓人，直接钉住。
    """
    for p in sorted(ROOT.glob("*.bat")):
        data = p.read_bytes()
        bad = [i for i, c in enumerate(data) if c >= 128]
        check(not bad, "%s 里有非 ASCII 字节（位置 %s）—— cmd 会把它当命令执行"
                       % (p.name, bad[:5]))

    # 部署台启动器里必须挂着"配置漂移预检"：GUI 用 pythonw 起、**没有控制台**，
    # 漂移只能靠弹窗说话 —— 不挂的话"跑的是旧文件"就一直隐形
    # （2026-09-25 实测：A 机跑旧版 sweep_link.py，握手静默退回老流程）。
    dep = (ROOT / "被控机部署台.bat").read_text(encoding="ascii")
    check("config_sync --preflight" in dep,
          "部署台启动器里没挂配置漂移预检（至少留个带日期的记录）")
    # **故意不弹窗**：同一个结论在部署台右上角的环境自检里就有一行，而修它的入口
    # 也在那儿（比如键盘卡片的「生成证书…」）—— 启动时弹一个只会打断人。
    check("--msgbox" not in dep,
          "启动器又在弹配置漂移窗了 —— 那类提示归部署台里的自检项")
    # 依赖预检要走**同一份清单**（deploy/deps.py）：写死在 .bat 里的三个名字
    # 曾经漏掉 cryptography/psutil，现场才炸 —— "装完了"必须等于"真能用"。
    check("deploy.deps" in dep,
          "启动器的依赖预检没走 deploy/deps.py（清单会各写一份，迟早又漏）")


def t_push_scale_mode_builds_both_chains():
    """「缩放位置」两种模式要拼出**该有的**过滤器串，而且默认那档一个参数都不变。

    为什么要它：2026-09-25 实测 A 机 18 条候选 speed 全贴 1.00、实际 fps 只有 ~73 ——
    瓶颈在"整屏 2560×1440 拷回内存 + CPU 缩放"这条链。GPU 那条（hwmap→scale_cuda）
    是解法候选，但它依赖 A 机 ffmpeg 支持 scale_cuda，所以：
      · 默认（cpu）必须与历史完全一致（否则历史表对不上、也没法回退）；
      · cuda 那条要能在命令行里看出 `scale_cuda` 与 cuda 设备（起不来时好定位）。
    """
    cfg_cpu = copy.deepcopy(dcfg.DEFAULTS["push"])
    cfg_gpu = dict(cfg_cpu, scale_mode="cuda")
    cpu = " ".join(services.build_cmd("push", cfg_cpu))
    gpu = " ".join(services.build_cmd("push", cfg_gpu))
    check("hwdownload,format=bgra,scale=1366:768,format=nv12" in cpu,
          "默认那档的采集链变了（历史表会对不上）：%s" % cpu[:160])
    check("scale_cuda" not in cpu, "默认那档混进了 GPU 缩放")
    check("scale_cuda=1366:768" in gpu and "hwmap=derive_device=cuda" in gpu,
          "GPU 缩放那档没拼出该有的链：%s" % gpu[:200])
    check("cuda=cu" in gpu, "GPU 缩放那档没给 cuda 设备（scale_cuda 会起不来）")
    check("hwdownload,format=bgra" not in gpu,
          "GPU 那档还在先把整屏拷回内存（那就白改了）")


def t_sweep_row_says_why():
    """推流自检的 A 侧：**这一段为什么没成**必须写进 why，表里要看得见。

    实测 2026-09-26：GPU 缩放那两档 ffmpeg 立刻退出（`out_fps=0 / frames=0`），
    可报告里记着 `ok: True` —— 因为 `_run_once` 只认"进程起没起来"，ffmpeg 那句报错
    被丢掉了。于是表上只剩一句"没量到延迟"，原因只能靠猜（我就猜了一轮）。
    现在 ffmpeg 输出的尾巴一起留下、写进 why。
    """
    from deploy.push_sweep import annotate_a_row

    row = {"name": "x", "speed": 0.99, "ok": True, "why": ""}
    extra = {"ffmpeg_tail": [
        "[cuda @ 000001] Cannot load cu",
        "[Parsed_ddagrab_0 @ 000002] Setting 'framerate' to value '144'",
        "Impossible to convert between the formats supported by the filter "
        "'graph 0 input from stream 0:0' and the filter 'auto_scaler_0'"]}
    annotate_a_row(row, [], extra, True)       # 起来了、但一句 stats 都没有
    check(row["ok"] is False and "立刻退出" in row["why"],
          "ffmpeg 立刻退出没被判成失败：%s" % row)
    check("Impossible to convert" in row["why"],
          "why 里没有 ffmpeg 那句报错（那才是原因）：%s" % row["why"])
    check(row["speed"] is None, "失败的那段还留着 speed，会被当成有效数据")

    # 正常的一段**不许**被动（既有报告、既有表都靠它）
    good = {"name": "y", "speed": 0.998, "ok": True, "why": ""}
    annotate_a_row(good, [{"speed": 0.998}], {"ffmpeg_tail": []}, True)
    check(good["ok"] is True and good["why"] == "" and good["speed"] == 0.998,
          "正常的一段被改了：%s" % good)


def t_cert_entry_is_findable():
    """证书入口要**显眼**：工具栏「证书…」必须在，且点开能说清现状。

    实测（2026-09-25）：证书一开始只挂在键盘卡片第三行那个小按钮上，现场两次都
    找不到 —— 人在找"证书相关的入口"，而不会去翻某张卡片的第三行。所以工具栏也要有，
    内容与卡片上那个共用同一份实现（`_cert_summary` / `_gen_cert`）。
    """
    import unittest.mock as mock

    from PyQt5.QtWidgets import QApplication, QMessageBox
    from deploy.app import DeployWindow

    app = QApplication.instance() or QApplication([])
    win = DeployWindow()
    try:
        check(hasattr(win, "act_cert") and win.act_cert.text() == "证书…",
              "工具栏上没有「证书…」入口")
        text = win._cert_summary()
        for want in ("cert.pem", "key.pem", "配对", "与另一台"):
            check(want in text, "证书现状里少了 %r：\n%s" % (want, text))
        # 点开它：只是个对话框，不该抛，也不该顺手把证书换掉
        with mock.patch.object(QMessageBox, "exec_", return_value=0):
            win._on_cert()
    finally:
        win.close()


def t_deps_all_importable():
    """A 机运行所需的模块，在**跑这个用例的机器上**必须都能 import。

    为什么单列：`python -m remote_kbd.gen_cert` 缺 cryptography 那次，根因是
    "入口只 import 了标准库 + 一个没列进依赖清单的包" —— 这类缺口现场才暴露，
    而且现象跟缺的东西八竿子打不着（生成证书失败 / 键盘连不上 / 探针窗口起不来）。
    清单只有一处：deploy/deps.py（装环境脚本、启动器、环境自检都照它）。
    """
    from deploy import deps

    check(deps.MODULES, "清单是空的 —— 那这个检查就是摆设")
    for name, why in deps.MODULES:
        check(len(why or "") > 4, "%s 没写「为什么需要它」" % name)
    miss = deps.check()
    check(not miss, "缺这些模块：%s" % (miss,))


def t_gen_cert_pair_loads():
    """「生成证书」生成出来的那一对，要能被 TLS **真的加载**（relay 就是这么用的）。

    只验"文件存在"不够：写坏了、私钥和证书不配对，都算"生成成功"，
    而现场的现象只是"键盘连不上"。
    """
    import shutil
    import ssl
    import tempfile

    from remote_kbd import gen_cert

    tmp = Path(tempfile.mkdtemp(prefix="cert_"))
    try:
        cert_p, key_p = gen_cert.generate(tmp)
        check(cert_p.exists() and key_p.exists(),
              "证书/私钥没生成：%s %s" % (cert_p, key_p))
        check(gen_cert.out_dir().name == "certs",
              "证书默认目录变了？out_dir=%s" % gen_cert.out_dir())
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(cert_p), str(key_p))     # 与 remote_kbd/relay.py 同一用法
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)


# ---------------------------------------------------------------- 跑

TESTS = (
    ("服务表（标题/参数/命令行）完整", t_service_table),
    ("小地图推流的命令行（按「当前是哪张图」传 --map-id）", t_mmap_cmd),
    ("启动前的拦截提示（mmap 不再拦「还没框过」）", t_missing_hint),
    ("DEFAULTS 与参数表同步（区域键不许回 deploy.json）", t_config_defaults),
    ("_deep_merge 只认已声明的键", t_config_merge),
    ("小地图端口进一致性自检", t_link_expect),
    ("自检里的小地图区域项", t_check_region),
    ("读真实 config/deploy.json", t_read_real_config),
    ("源码约束（框选不再自己跑事件循环）", t_app_source),
    ("卡片与框选控件（离屏）", t_cards),
    ("框选能在已有事件循环里跑（部署台那条路）", t_ask_region_modal),
    ("设置弹窗：字号点确定才落盘", t_settings_dialog_font),
    ("日志保留行数两处一起改", t_log_history_applies),
    ("启动后卡片的「将执行」不为空", t_cards_show_cmd_at_startup),
    ("自检前先查探针/时钟两张卡", t_sweep_guard_asks_for_probe_and_clock),
    ("批处理文件必须是纯 ASCII", t_bat_files_are_ascii),
    ("A 机运行依赖齐全（清单只有一处）", t_deps_all_importable),
    ("生成的证书能被 TLS 加载", t_gen_cert_pair_loads),
    ("证书入口显眼（工具栏「证书…」）", t_cert_entry_is_findable),
    ("采集链两种模式都拼得对（默认档不变）", t_push_scale_mode_builds_both_chains),
    ("自检里那一段没成时，why 要带上 ffmpeg 的报错", t_sweep_row_says_why),
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
