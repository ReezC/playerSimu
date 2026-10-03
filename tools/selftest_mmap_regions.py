"""自检：**小地图推流区域「按地图 id 存放 + 运行中换图」**（2026-10-01 新增）。

为什么要有它 —— 这三件事**错了都不报错**，只在很后面才表现为"寻路坏了"：

  · **按 id 取**：用哪一块现在是三条口径（本图有 → 没有就用老的那份兜底 → 都没有
    就是没有），和 B 机那边 `mmap_crop`（`perception.minimap.crop_of`）是**同一套
    规矩** ⇒ 这里的断言刻意和 `tools/selftest_minimap.py` 的
    `t_mmap_crop_is_per_project` 对齐 ✓；拿错图 ⇔ 推的不是当前这块 ⇔ B 机
    "底图对不上"，看着像定位坏了 ✗；
  · **换图那一瞬**：`MAP` 命令到底把区域换掉了没有、换失败时会不会留下**半截状态**
    （那是比没切成更难查的东西 ✗）；
  · **zoom × calib**：标定是**对着某个 zoom 量出来的**，zoom 变了照套旧标定 ⇒ 坐标
    整倍数错。所以这里钉死"**zoom 不符 ⇒ 不给标定**"，而不是"先凑合发出去"。

协议（握手 / 命令 / 回执）也一样：**A 机和 B 机必须讲同一句**，这里用源码级检查
拦住"两边各抄一份魔数"那种分叉。

跑法：
    python -m tools.selftest_mmap_regions
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Windows 控制台默认是 **GBK**：用例名里带 ⇒ / ✔ 这类字符时，`print` 会直接抛
# `UnicodeEncodeError` —— 那和被测的东西一点关系都没有，纯粹是终端的编码问题。
# ⇒ 这里把 stdout 切成 UTF-8（Python 3.7+ 才有 `reconfigure` ⇒ 老环境忽略即可 ✓）。
try:
    sys.stdout.reconfigure(encoding="utf-8")            # type: ignore[attr-defined]
except Exception:                                       # noqa: BLE001
    pass

from tools import mmap_regions                          # noqa: E402
from tools import minimap_push                          # noqa: E402
from core import mapdata                                # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


_TMP = None


def _fresh():
    """每个用例开头调一次：把库换到一个**干净的空目录**（不许碰真实数据 ✓）。

    ⚠ **三份都要换**（2026-10-02 补 `CURRENT_FILE` ✗）：区域的每图一份、老单值兜底、
      以及「现在该推哪张图」那份记忆。少换一份的后果很具体：`_answer_ctl` 处理 `MAP`
      时会**写进真实的那份**（`config/minimap_current.json` ✗）⇒ 自检把人本机的
      状态改了，还可能让别的用例看机器状态吃饭（假绿/假红都来过 ✓）。
    """
    global _TMP
    if _TMP is None:
        _TMP = Path(tempfile.mkdtemp(prefix="selftest-mmap-regions-"))
    _dir = _TMP / "regions"
    if _dir.exists():
        shutil.rmtree(_dir)
    _dir.mkdir(parents=True, exist_ok=True)
    mmap_regions.DIR = _dir
    mmap_regions.LEGACY_REGION_FILE = _TMP / "minimap_region.json"
    if mmap_regions.LEGACY_REGION_FILE.exists():
        mmap_regions.LEGACY_REGION_FILE.unlink()
    mmap_regions.CURRENT_FILE = _TMP / "minimap_current.json"
    if mmap_regions.CURRENT_FILE.exists():
        mmap_regions.CURRENT_FILE.unlink()
    return _dir


def _q(*_a, **_k):
    """吞掉推流那边的日志（它默认 `print`，自检里不需要它说话）。"""


class _Sock:
    """假的控制连接：只记下 A 机回出去的每一行。"""

    def __init__(self):
        self.sent = []

    def sendall(self, raw):
        self.sent.append(raw)
        return len(raw)


# ══════════════════════ 用例 ══════════════════════

def t_load_and_legacy():
    """`load`（每图一份）与 `legacy`（老单值兜底）各自的边界。

    ⭐ 2026-10-02：原来这里测的是 `resolve()`（"本图没有就用老那份"）—— 那个口径**搬到**
    `startup_region()` 里了（用户新口径：**点名的图没框过就别拿别的图顶上** ✗），
    `resolve` 随之删掉（死代码不留 ✓）。这里只钉这两个基础读取：
      ① `load` 给的是**本图**那份（哪怕老的那份还在 ✓ 不串台 ✓）；
      ② `legacy()` 读得回老那份（升级后不用重框 ✓ 口径同 `perception.minimap.crop_of`）；
      ③ **`load` 不兜底**：本图没框过就是 `None`（兜底只发生在 `startup_region` 那一处 ✓）；
    第 4 件照旧：**坏值当"没填"** —— 拿半截数去抓屏，推的是一片无关画面。
    """
    _fresh()
    mmap_regions.save("111111111", 1, 2, 200, 150, zoom=3)

    # ① 本图框过 ⇒ 用本图的（哪怕老的那份还在 ✓）
    mmap_regions.LEGACY_REGION_FILE.parent.mkdir(parents=True, exist_ok=True)
    mmap_regions.LEGACY_REGION_FILE.write_text(
        '{"region": [7, 7, 70, 70], "zoom": 2}', encoding="utf-8")
    got = mmap_regions.load("111111111")
    check(got is not None and got["x"] == 1 and got["zoom"] == 3,
          "本图框过了却没用本图的：%s" % (got,))

    # ② 老单值读得回来（它现在是**只读兜底**：只有"这台机还没有任何 id"时才用 ✓）
    old = mmap_regions.legacy()
    check(old is not None and old["x"] == 7 and old["zoom"] == 2,
          "老单值那份读不出来了（升级后要人把每张图重框一遍 ✗）：%s" % (old,))

    # ③ `load` **不兜底**（兜底只在 `startup_region` 那一处 ✓）
    check(mmap_regions.load("222222222") is None, "本图没框过，load 却拿出了别的框 ✗")
    check(mmap_regions.load("") is None, "空 id 居然配出了一个框 ✗")
    check(mmap_regions.load(None) is None, "None id 居然配出了一个框 ✗")

    # ④ 坏值 ⇒ 当"没填"（不是拿去抓屏 ✗）
    _p = mmap_regions.path_of("444444444")
    _p.write_text('{"x": 5, "y": 5, "w": 3, "h": 3}', encoding="utf-8")   # 太小
    check(mmap_regions.load("444444444") is None, "太小的区域不该算配过 ✗")
    _p.write_text('{"x": -1, "y": 5, "w": 300, "h": 300}', encoding="utf-8")  # 负坐标
    check(mmap_regions.load("444444444") is None, "负坐标的坏值不该算配过 ✗")
    _p.write_text('{"x": "abc", "y": 5, "w": 300, "h": 300}', encoding="utf-8")  # 不是数
    check(mmap_regions.load("444444444") is None, "非数字的坏值不该算配过 ✗")
    _p.write_text('{"x": 5, "y": 5, "w": 300, "h": 300}', encoding="utf-8")      # 好的
    check(mmap_regions.load("444444444") is not None, "合格的一份被判成坏值了 ✗")

    # ⑤ 文件坏了 = 没配过（宁可让人重框，别把它当真值 ✗）
    _p.write_text("{不是 json", encoding="utf-8")
    check(mmap_regions.load("444444444") is None, "文件坏了却被当成有效配置 ✗")


def t_list_and_remove():
    """库里有哪些图（`LIST` 命令给 B 机的"还能选哪些"）—— 按 id 排好序。"""
    _fresh()
    check(mmap_regions.list_ids() == [], "空库不该列出东西")
    for mid in ("300", "100", "200"):
        mmap_regions.save(mid, 0, 0, 200, 150)
    check(mmap_regions.list_ids() == ["100", "200", "300"],
          "列出来的图没排好序：%s" % (mmap_regions.list_ids(),))
    check(mmap_regions.remove("200") is True, "删一张已配的图失败")
    check(mmap_regions.remove("200") is False, "删一张已经没有的居然成功了")
    check(mmap_regions.list_ids() == ["100", "300"], "删完列表不对")


def t_save_rejects_empty_id():
    """**不知道是哪张图就不许存** —— 存进去的那份将来会推到错的图上 ✗。"""
    _fresh()
    try:
        mmap_regions.save("", 1, 2, 200, 150)
    except ValueError:
        return
    raise AssertionError("没给地图 id 也存进去了 —— 那份将来会推到错的图上 ✗")


def t_protocol_roundtrip():
    """回执格式：**拼得出来就解得回去**（每一份 *_reply 都对上一个 parse_reply）。

    ⚠ `STATE` 那一行在"还没指定图"时写的是 `-`，解回来必须是 `""`
      （被当成一张真的图 id，后果是拿不对的区域去匹配 ✗）。
    """
    rep = mmap_regions.parse_reply(mmap_regions.ok_reply("105090600", (1, 2, 200, 150), 3))
    check(rep is not None and rep["ok"] is True, "OK 回执解不开 ✗")
    check(rep["kind"] == "MAP" and rep["map_id"] == "105090600", "id 解错了：%s" % (rep,))
    check(rep["box"] == [1, 2, 200, 150], "区域解错了：%s" % (rep,))
    check(rep["zoom"] == 3, "zoom 没搭回来（上层要靠它挑标定）：%s" % (rep,))

    rep = mmap_regions.parse_reply(mmap_regions.list_reply(["100", "200"]))
    check(rep is not None and rep["ids"] == ["100", "200"], "LIST 解不开：%s" % (rep,))
    check(mmap_regions.parse_reply(mmap_regions.list_reply([]))["ids"] == [],
          "空 LIST 该解成空列表")

    rep = mmap_regions.parse_reply(mmap_regions.state_reply("105090600", (1, 2, 3, 4), 1))
    check(rep is not None and rep["kind"] == "STATE" and rep["map_id"] == "105090600",
          "STATE 解不开：%s" % (rep,))

    rep = mmap_regions.parse_reply(mmap_regions.state_reply("-", (0, 0, 0, 0), 1))
    check(rep is not None and rep["map_id"] == "",
          "「还没指定图」的 `-` 没解成空 id（会被当成一张真的图 ✗）：%s" % (rep,))

    rep = mmap_regions.parse_reply(mmap_regions.err_reply("no-such-map", "这张图还没框过"))
    check(rep is not None and rep["ok"] is False and rep["code"] == "no-such-map",
          "ERR 解不开：%s" % (rep,))
    check("还没框过" in (rep.get("why") or ""), "ERR 里那句人话丢了：%s" % (rep,))

    # 看不懂的行 ⇒ None（**不知道就是不知道**，别猜 ✓）
    for bad in ("", "   ", "hello", "ZZZZ whatever"):
        check(mmap_regions.parse_reply(bad) is None, "一行垃圾不该解出东西：%r" % bad)


def t_map_command_switches_region():
    """**推流开着不停 ⇒ 内容自己变更**：一条 `MAP` 就把 `state.box` 换了 ✓。

    这是整套东西的落点：A 机那一拍读的还是旧区域、下一拍就是新区域（见
    `tools/minimap_push.py` 帧循环那句「每拍重新读 state」）。
    """
    _fresh()
    mmap_regions.save("111", 10, 20, 200, 150, zoom=3)
    mmap_regions.save("222", 30, 40, 260, 190, zoom=1)
    state = {"map_id": "", "box": [0, 0, 0, 0], "zoom": 1}
    s = _Sock()

    minimap_push._answer_ctl(s, b"MAP 222", state, log=_q)
    check(state["map_id"] == "222", "换了图但没记下是哪张：%s" % (state,))
    check(state["box"] == [30, 40, 260, 190], "区域没换成新图的：%s" % (state,))
    check(state["zoom"] == 1, "zoom 没跟着图一起换（标定会对不上 ✗）：%s" % (state,))
    check(mmap_regions.parse_reply(s.sent[-1].decode("utf-8"))["ok"] is True,
          "换图成功却没回 OK：%r" % (s.sent[-1],))

    # 再换一张 ⇒ **区域跟着变**（这才叫"内容自己变更"）
    minimap_push._answer_ctl(s, b"MAP 111", state, log=_q)
    check(state["box"] == [10, 20, 200, 150] and state["zoom"] == 3,
          "第二次换图没生效：%s" % (state,))

    # ⭐ **"当前是哪张图"顺手记在本机**（用户 2026-10-02 ✓："只靠 B 机推"）——
    #   部署台的「当前地图」读的就是它、框选也存到这张图名下 ✓ ⇒ 不记的话人根本
    #   不知道该往哪个 id 下框 ✗。
    cur = mmap_regions.current()
    check(cur and cur["map_id"] == "111",
          "换图成功却没把「当前是哪张图」记在本机（部署台/框选就没法对上 ✗）：%r" % (cur,))
    check(cur.get("by") == "B机", "记下来的来源不对（该写清是 B 机说的 ✓）：%r" % (cur,))

    # 没配过的图 ⇒ **一个字节都不动**（半切换的状态比没切成还难查 ✗）
    before = list(state["box"])
    minimap_push._answer_ctl(s, "MAP 不在库里的图".encode("utf-8"), state, log=_q)
    check(state["box"] == before, "换图失败却把当前区域改掉了 ✗：%s" % (state,))
    rep = mmap_regions.parse_reply(s.sent[-1].decode("utf-8"))
    check(rep is not None and rep["ok"] is False, "失败该回 ERR：%r" % (s.sent[-1],))
    check((rep.get("why") or ""), "失败原因一句话都不说，人只能靠猜 ✗")

    # ⭐⭐ **未命中也要把 id 记下来**（用户 2026-10-03 ✓ 现场："改成收流了 A 那行还是
    #   「还没有」"✗）—— 本模块开头那段设计**本来就是这个意思** ✓（"没有 ⇒ 回 ERR，
    #   人再去部署台卡片里框一次（**那时 id 已经知道了** ✓）"），可代码只写了回 ERR ✗
    #   ⇒ 部署台那行永远读不到 id ⇒ 人**没法**按这张图的名字去框（框选是存到"当前这张图"
    #   名下的 ✓）⇒ **死结** ✓。
    #   ⚠ 记的只是**"当前地图"这个标签** ✓，抓取区域（`state`）**一个字节不动** ✗
    #     —— 上面那条"半切换"的纪律照样守着 ✓（唯一变化：人现在知道该往哪张图名下框 ✓）。
    cur2 = mmap_regions.current()
    check(cur2 and cur2["map_id"] == "不在库里的图",
          "未命中时没把 id 记下来 ⇒ 部署台永远「还没有」、人也框不到这张图名下 ✗：%r"
          % (cur2,))
    check(cur2.get("by") == "B机(未命中)",
          "记下来时没标注「未命中」（人会以为区域已经就位 ✗）：%r" % (cur2,))
    check(state["box"] == before,
          "未命中时不该动抓取区域（还是上面那条纪律 ✗）：%s" % (state,))

    # PING / LIST / 未知命令
    minimap_push._answer_ctl(s, b"PING", state, log=_q)
    check(s.sent[-1].strip() == b"PONG", "PING 不回 PONG：%r" % (s.sent[-1],))
    minimap_push._answer_ctl(s, b"LIST", state, log=_q)
    check(mmap_regions.parse_reply(s.sent[-1].decode("utf-8"))["ids"] == ["111", "222"],
          "LIST 没列出库里的图：%r" % (s.sent[-1],))
    minimap_push._answer_ctl(s, "乱写 一通".encode("utf-8"), state, log=_q)
    check(mmap_regions.parse_reply(s.sent[-1].decode("utf-8"))["code"] == "bad-command",
          "未知命令没回 bad-command ✗")


def t_current_map_memory():
    """⭐ 「现在该推哪张图」那份记忆（用户 2026-10-02 ✓ 原话："只靠 B 机推（A 机不能
    自己选）" + "地图与框选数据另存一份文件，仅 A 机本地储存"）。

    钉五件：
      ① 没记过 ⇒ `current()` 是 `None`（**不是**编一个空 id ✗）；
      ② 记了 ⇒ 读回来带着 `map_id` / `by`（谁说的）/ `updated` ✓；
      ③ **空 id ⇒ 删掉那份记忆**（回到"还不知道是哪张图"✓ 别留半截 ✗）；
      ④ 文件坏了 / 不是 dict / 没有 map_id ⇒ 一律 `None`（**不猜** ✓）；
      ⑤ `clear_current()` 能忘掉它（没记过返回 False ✓）。
    """
    _fresh()
    check(mmap_regions.current() is None, "从没记过却说有当前图 ✗")

    mmap_regions.set_current("105040303", by="B机")
    got = mmap_regions.current()
    check(got and got["map_id"] == "105040303" and got["by"] == "B机" and got["updated"],
          "记下来的当前图不完整：%r" % (got,))
    check(mmap_regions.CURRENT_FILE.exists(),
          "「当前图」没落到本机文件里（重启一次就丢 ⇒ 部署台又说「不知道哪张图」✗）")

    # ③ 空 id ⇒ 删掉记忆（不是"记一个空 id"）
    mmap_regions.set_current("", by="手动")
    check(mmap_regions.current() is None, "空 id 没把记忆清掉 ✗")
    check(not mmap_regions.CURRENT_FILE.exists(), "空 id 之后文件还在 ✗")

    # ④ 坏文件一律当"没记过"
    for bad in ("不是 json", "[]", '{"by": "B机"}', '{"map_id": "  "}'):
        mmap_regions.CURRENT_FILE.parent.mkdir(parents=True, exist_ok=True)
        mmap_regions.CURRENT_FILE.write_text(bad, encoding="utf-8")
        check(mmap_regions.current() is None,
              "坏掉的「当前图」文件被当成有效配置了（%r ⇒ 会去推一张不存在的图 ✗）"
              % bad)

    # ⑤ 忘掉它
    mmap_regions.set_current("222", by="B机")
    check(mmap_regions.clear_current() is True, "清当前图失败")
    check(mmap_regions.current() is None and mmap_regions.clear_current() is False,
          "清完之后状态不对 ✗")


def t_startup_region_priority():
    """⭐ 推流启动时"到底推哪一块"的**唯一一处**口径（用户 2026-10-02 ✓）。

    优先级：`--map-id` → **本机记着的当前图** → 老单值兜底 → **一块都没有**（⇒ 照旧
    起监听、等 B 机 `MAP` ✓）。两条"别顶替"的规矩是这套东西的重点：
      · **点名了却没有** ⇒ `None`（拿别的图顶上 = 推错一块画面 ✗ B 机那边像寻路坏了 ✗）；
      · **当前图明确、可它还没框过** ⇒ `None`（**不许**退到老单值 —— 那是**另一张图**的
        屏幕坐标 ✗ 同一条理由 ✓）。
    返回里的 `src` 要说清这块框是从哪来的（`map-id` / `当前图` / `老单值` ✓ 排查用 ✓）。
    """
    _fresh()
    # 老单值放一份（升级前的那份）
    mmap_regions.LEGACY_REGION_FILE.parent.mkdir(parents=True, exist_ok=True)
    mmap_regions.LEGACY_REGION_FILE.write_text(
        json.dumps({"region": [1, 2, 300, 200], "zoom": 2}), encoding="utf-8")

    # ④ 一个 id 都没有 ⇒ **老单值兜底**（升级后不用重框 ✓）
    got = mmap_regions.startup_region("")
    check(got is not None and got[3] == "老单值" and got[1] == [1, 2, 300, 200],
          "还没有任何 id 时该回退老单值（升级后不用重框 ✓）：%r" % (got,))

    mmap_regions.save("111", 10, 20, 200, 150, zoom=3)
    mmap_regions.set_current("111", by="B机")
    got = mmap_regions.startup_region("")
    check(got is not None and got[0] == "111" and got[3] == "当前图"
          and got[1] == [10, 20, 200, 150] and got[2] == 3,
          "有当前图时该用**当前图**那份（含它自己的 zoom ✓）：%r" % (got,))

    # ① 命令行点名 ⇒ 用点名那张（哪怕当前图是另一张 ✓）
    mmap_regions.save("222", 30, 40, 260, 190, zoom=1)
    got = mmap_regions.startup_region("222")
    check(got is not None and got[0] == "222" and got[3] == "map-id",
          "点名了哪张图就该推哪张：%r" % (got,))

    # ② 点名了却没有 ⇒ None（**不拿别的图顶上** ✗）
    check(mmap_regions.startup_region("999") is None,
          "点名了一张没框过的图，却拿了别的图的框顶上（推错画面 ✗）")

    # ③ 当前图明确、可它还没框过 ⇒ None（**不退老单值** ✗）
    mmap_regions.set_current("888", by="B机")
    check(mmap_regions.startup_region("") is None,
          "当前图还没框过时退回了老单值 —— 那是**另一张图**的屏幕坐标（换图就全错 ✗）"
          "B 机那边会表现为「底图对不上」，看着像寻路坏了 ✗")

    # ⑤ 当前图的记忆是坏的 ⇒ 当"没有 id" ⇒ 老单值兜底（别把坏记忆当结论 ✗）
    mmap_regions.CURRENT_FILE.write_text("坏", encoding="utf-8")
    got = mmap_regions.startup_region("")
    check(got is not None and got[3] == "老单值",
          "「当前图」文件坏了就该当成「没记过」⇒ 走老单值兜底：%r" % (got,))

    # ⑥ 连老单值都没有 ⇒ None（调用方照旧起监听等 B 机 ✓）
    mmap_regions.LEGACY_REGION_FILE.unlink()
    check(mmap_regions.startup_region("") is None,
          "一块都没有时该给 None（让服务「先起来等 B 机」✓）")


def t_push_waits_without_region():
    """⭐ **"还没框过也先起来等 B 机"**（用户 2026-10-02 ✓ 的必然要求）——**源码级**钉。

    为什么只能源码级：这一段要真起 Qt 抓屏 + 一个 TCP 服务（自检里建不起来 ✗）。
    但这条**必须钉**，因为它是整条链的闭环处：B 机那句 `MAP <id>` 走的就是这条服务的
    连接 ⇒ 服务起不来 ⇒ A 机**永远**知道不了是哪张图（框选都无从谈起 ✗）。
    钉三件：
      ① 启动解析走**共用那一处** `startup_region`（别在推流里再写一套优先级 ✗）；
      ② 没有区域时**不再直接退出**、而是照常绑定端口进主循环 ✓；
      ③ 帧循环在没有区域时**一帧都不抓**（抓了也不知道抓哪儿 ✗）且会提醒一句 ✓。
    """
    src = (Path(__file__).resolve().parent / "minimap_push.py").read_text(encoding="utf-8")
    check("mmap_regions.startup_region(" in src,
          "推流启动没走 `mmap_regions.startup_region`（优先级口径该只有一处 ✗）")
    check('"box": (list(box0) if box0 else None)' in src,
          "state 没允许「还没有区域」（box=None）⇒ 起不来就等不到 B 机的 MAP ✗")
    body = src.split("controls = _poll_controls(controls, state)", 1)
    check(len(body) == 2, "帧循环里找不到处理控制命令那段（改动太大，脚本要跟上 ✗）")
    check('if not state["box"]:' in body[1][:400],
          "帧循环没有「还没有区域就跳过抓屏」这一档 ⇒ 会拿一个空/旧区域乱抓 ✗")
    check("还没收到图" in body[1][:900],
          "没有区域时不提醒一句 ⇒ B 机那边「连上却一帧不来」，日志里也看不出为什么 ✗")


def t_calib_zoom():
    """zoom × calib：**同一个 zoom 才肯发货**，变了 ⇒ None（不静默套错几何 ✓）。

    同时保住一条：**不传 zoom 的老调用方行为完全不变**（那边一行都不要改 ✓）。
    """
    _fresh()
    old = mapdata.calib_path
    _d = Path(tempfile.mkdtemp(prefix="selftest-calib-zoom-"))
    mapdata.calib_path = lambda mid: _d / ("%s.mapcalib.json" % mid)
    try:
        mid = "999888777"
        mapdata.save_calib(mid, {"mode": "fit", "scale": 2.0, "offset": [1, 2]},
                           "stream", zoom=3)

        # ① 不传 zoom ⇒ 老行为：照样拿得到
        got = mapdata.load_calib(mid, "stream")
        check(got is not None and float(got["scale"]) == 2.0,
              "不传 zoom 时拿不到了 —— 那是破坏了老调用方 ✗")
        check(mapdata.zoom_of(got) == 3, "zoom 没写进标定文件：%s" % (got,))

        # ② zoom 相符 ⇒ 给
        check(mapdata.load_calib(mid, "stream", zoom=3) is not None,
              "同一个 zoom 却不给标定 ✗")
        # ③ zoom 不符 ⇒ **None** —— 这次要修的坑就在这一步
        check(mapdata.load_calib(mid, "stream", zoom=1) is None,
              "zoom 变了还把标定发出去 —— 坐标会整倍数错且不报错 ✗")
        check(mapdata.load_calib(mid, "stream", zoom=5) is None,
              "zoom=5 却套用了 zoom=3 标出来的那份 ✗")

        # ④ 老文件（**没有** zoom 字段）⇒ 一律当 zoom=1 ⇒ 不许猜
        mapdata.save_calib(mid, {"mode": "fit", "scale": 2.0}, "live")
        check(mapdata.zoom_of(mapdata.load_calib(mid, "live")) == 1,
              "没记 zoom 的老标定，zoom_of 该是 1（不许替它猜当时的 zoom）")
        check(mapdata.load_calib(mid, "live", zoom=1) is not None,
              "老格式在 zoom=1 下该可用")
        check(mapdata.load_calib(mid, "live", zoom=3) is None,
              "老格式（不知道 zoom）在 zoom=3 下不该被套用 ✗")

        # ⑤ `save_calib` 不带 zoom ⇒ 不写这个键（别给老标定凭空编一个 zoom ✗）
        mapdata.save_calib(mid, {"mode": "fit", "scale": 3.0}, "extra")
        check("zoom" not in (mapdata.load_calib(mid, "extra") or {}),
              "没传 zoom 却在文件里写了个 zoom ✗")
    finally:
        mapdata.calib_path = old
        shutil.rmtree(_d, ignore_errors=True)


def t_blackout_detection():
    """「进传送门 → 黑屏」那一瞬的判据。

    ⚠ **取不到帧 ⇒ False** 这条最要紧：把"还没收到帧"（断流的症状 ✗）判成"正在
    换图"，正好把人往错的方向带。
    """
    import numpy as np

    import perception.minimap as mm

    black = np.zeros((60, 80, 3), dtype=np.uint8)
    check(mm.is_blackout(black) is True, "全黑的一帧没判成黑屏 ✗")

    bright = np.full((60, 80, 3), 200, dtype=np.uint8)
    check(mm.is_blackout(bright) is False, "有内容的一帧不该是黑屏 ✗")

    # 还剩一点点内容（比如面板边框）⇒ 仍是黑屏（"还剩 1% 有内容"不算已经亮起 ✗）
    nearly = np.zeros((60, 80, 3), dtype=np.uint8)
    nearly[0, :20] = 255
    check(mm.is_blackout(nearly) is True, "只剩一小条亮边就被判成「已经亮起」了 ✗")

    # 反过来：有一半内容 ⇒ 不是黑屏
    half = np.zeros((60, 80, 3), dtype=np.uint8)
    half[:30, :] = 255
    check(mm.is_blackout(half) is False, "半张图有内容却判成黑屏 ✗")

    # 取不到帧 ⇒ False（不许猜 ✓）
    check(mm.is_blackout(None) is False, "空帧不该判成黑屏（那是断流，不是换图）✗")
    check(mm.is_blackout(np.zeros((0, 0, 3), dtype=np.uint8)) is False,
          "0 像素的帧不该判成黑屏 ✗")


def t_protocol_is_single_sourced():
    """协议**只有一份**：A、B 两边都得引用 `mmap_regions` 的常量，不许各写一份魔数。

    为什么连这个也要钉：握手那串字节只要一边改了、另一边没跟着改，现象就是
    "连得上、叫不应" —— 而且两边各自的日志看着都很正常 ✗（见 `MiniMapClient.switch_map`
    失败时那句"它可能还没支持 MAP 命令"，那就是照着这种难查的样子写的）。
    """
    push = (ROOT / "tools" / "minimap_push.py").read_text(encoding="utf-8")
    mm = (ROOT / "perception" / "minimap.py").read_text(encoding="utf-8")

    check("mmap_regions.CTL_HELLO" in push, "A 机侧的握手魔数没走共用那一份 ✗")
    check("mmap_regions.CTL_HELLO" in mm, "B 机侧的握手魔数没走共用那一份 ✗")
    check("mmap_regions.CMD_MAP" in push, "A 机侧自己拼了命令名 ✗")
    check("mmap_regions.CMD_MAP" in mm, "B 机侧自己拼了命令名 ✗")

    # 回执格式不许两边各拼一份（`OK %s …` 那种字符串）
    for who, src in (("tools/minimap_push.py", push), ("perception/minimap.py", mm)):
        check('"OK %s' not in src and "'OK %s" not in src,
              "%s 自己拼了 OK 回执格式（该调 mmap_regions.ok_reply）✗" % who)
        check('"ERR %s' not in src,
              "%s 自己拼了 ERR 回执格式（该调 mmap_regions.err_reply）✗" % who)

    # 握手等待必须放在同一种地 —— 超时时间写在 mm.py 库里，两边共用
    check("CTL_HANDSHAKE_TIMEOUT" in push,
          "A 机等握手的时间没用共用常量（两边不一致就会出现「连得上、叫不应」）✗")


def t_import_without_numpy():
    """模拟 A 机（部署台只装 PyQt5）：挡住 numpy / cv2 / core 也要能 import ✓。

    这条比"源码里不许有 import numpy"更硬：谁往里加一个 numpy 依赖，A 机会在启动
    部署台时直接报 `No module named 'numpy'` —— 而那边根本装不了（A 机不需要也
    不该装这套）。开子进程跑，免得污染本进程。
    """
    code = (
        "import sys\n"
        "for name in ('numpy', 'cv2', 'core', 'perception'):\n"
        "    sys.modules[name] = None\n"          # None ⇒ 再 import 直接 ImportError
        "import tools.mmap_regions as r\n"
        "assert r.MIN_WH >= 1\n"
        "assert callable(r.startup_region) and callable(r.set_current)\n"
        "assert callable(r.parse_reply)\n"
        "assert callable(r.ok_reply) and callable(r.err_reply)\n"
        "print('ok')\n"
    )
    p = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                       capture_output=True, text=True, timeout=120)
    check(p.returncode == 0 and "ok" in (p.stdout or ""),
          "mmap_regions 在纯标准库环境下 import 失败（A 机没有 numpy/cv2/core）：%s"
          % ((p.stderr or "").strip()[-300:]))


TESTS = (
    ("load 与老单值兜底：本图不串台 / legacy 读得回 / load 自己不兜底 / 坏值当没填",
     t_load_and_legacy),
    ("库列表与删除（LIST 命令要用）", t_list_and_remove),
    ("没给地图 id 不许存（会推到错的图上）", t_save_rejects_empty_id),
    ("协议回执：拼得出来就解得回去（含 `-` 那种空 id）", t_protocol_roundtrip),
    ("MAP 命令：推流不停 ⇒ 内容自己变更（含换失败了不许动原区域）",
     t_map_command_switches_region),
    ("zoom × calib：同一个 zoom 才发货，变了给 None（老调用方不受影响）",
     t_calib_zoom),
    ("黑屏判据（含「取不到帧不许当黑屏」）", t_blackout_detection),
    ("协议只有一份：A/B 共用常量，不许各写一份魔数", t_protocol_is_single_sourced),
    ("A 机环境：没有 numpy/cv2 也要能 import", t_import_without_numpy),
    ("⭐ 「现在该推哪张图」那份记忆（B 机推来 / 本机本地 / 坏了当没记过）",
     t_current_map_memory),
    ("⭐ 启动推哪一块：点名 → 当前图 → 老单值兜底 → 一块都没有等 B 机"
     "（两条「不许顶替」的规矩）", t_startup_region_priority),
    ("⭐ 「还没框过也先起来等 B 机」（源码级：共用解析 + 跳过抓屏 + 提醒一句）",
     t_push_waits_without_region),
)


def main() -> int:
    failed = 0
    for name, fn in TESTS:
        try:
            fn()
        except Exception as e:                             # noqa: BLE001
            failed += 1
            print("[FAIL] %s\n       %s: %s" % (name, type(e).__name__, e))
        else:
            print("[ OK ] %s" % name)
    print("\n%d/%d 通过" % (len(TESTS) - failed, len(TESTS)))
    if _TMP is not None:                       # 临时目录只在这个进程里用过
        shutil.rmtree(_TMP, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
