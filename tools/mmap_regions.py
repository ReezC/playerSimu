"""【A 机】小地图**推流区域**的**按地图 id 存放** —— 一张图一份文件。

同时也放了 **A、B 两台机共用的控制协议**（魔数 / 命令 / 回执格式，见下面第二节），
因为协议两边都要讲同一句话：**各写一份迟早分叉** ✗（这条规矩在 `gui/region_picker`
那份「全仓唯一实现」上已经付过学费了 ✓）。

为什么要按地图 id 分 —— 见 `perception/minimap.py` 的 `crop_of`：它管的「B 机画面里
那个框」早就是**按项目（＝图）存**的，理由是用户 2026-09-27 现场踩的：**不同图的小
地图面板尺寸/位置完全不同**，全局一份 ⇒ 换个图还是上一张图的框 ⇒ 叠图/定位全错。
这里（A 机的屏幕坐标）是**同一件事的同一个理由**：
  · A 机的矩形是**屏幕坐标**（取决于 A 机分辨率/UI 缩放/窗口位置）；
  · B 机的 `mmap_crop` 是**流画面坐标**（取决于推流分辨率）；
  ⚠ 所以这两份**不能合并**，它们只是**按同一个 id 对齐** ✓。

**这个文件只依赖标准库**：A 机部署台那边只有 PyQt5（没装 numpy/cv2），所以这里
不许 import 它们 —— `tools/selftest_region.py` 的 `t_import_without_numpy` 那套
检查就是守这条线的 ✓。
"""

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: 每张图一份：`config/minimap_regions/<map_id>.json`
DIR = ROOT / "config" / "minimap_regions"

#: 老的那份**单值**区域（升级前唯一的一份，见 `tools/minimap_push.py` 的
#: `REGION_FILE`）。它现在是**只读兜底**：某张图还没框过时先用着它 ⇒ 已经在跑的
#: 机器升级后不用把所有图重框一遍 ✓（口径同 `perception.minimap.crop_of` 的兜底 ✓）。
LEGACY_REGION_FILE = ROOT / "config" / "minimap_region.json"

#: 最小区域（像素）—— 和 `deploy/services.py` 的 `missing_hint` 那条判据同一口径 ✓：
#: 「几个像素的区域推出去，B 机认不出是哪个地图」。
MIN_WH = 20

#: ⭐ **"现在该推哪张图"**（用户 2026-10-02 ✓）—— **A 机本地**的一份小状态：
#: `config/minimap_current.json` = `{"map_id": ..., "by": "B机"/"手动", "updated": ...}`。
#:
#: 为什么要有它：用户 2026-10-02 定「**当前是哪张图**由 B 机推（A 机界面不许自己选）」，
#: 而 A 机的框选（部署台「框选…」）**必须知道这次框的是哪张图** ⇒ 得有个地方**记着
#: 最近一次 B 机告知的 id**：
#:   · B 机发 `MAP <id>` ⇒ `tools/minimap_push` 处理成功时写它（`set_current` ✓）；
#:   · 部署台卡片读它 ⇒ 显示「当前地图」、框选就存到这张图名下 ✓；
#:   · 重启推流（没带 `--map-id` 时）也读它 ⇒ 起来就还是上一张图 ✓。
#:
#: ⚠ **A 机本地、绝不同步**（用户 2026-10-02 原话："地图与框选数据另存一份文件，仅 A 机本地
#:   储存"✓）：它说的是"**这台机器的屏幕**"（分辨率/窗口位置各机不同）⇒ 进了 `LOCAL`
#:   清单（`tools/config_sync.py`）并被 `.gitignore` 挡住 ✓。
#: ⚠ 它**不是**区域库：区域还是每图一份（`DIR/<id>.json` ✓）；这份只回答"现在用哪张"。
#: ⚠ 放在 `DIR` **之外**是故意的：`list_ids()` 是 `DIR.glob("*.json")` ⇒ 塞进那个目录会
#:   被当成一张叫 `_current` 的图 ✗。
CURRENT_FILE = ROOT / "config" / "minimap_current.json"


# ══════════════════════════════════════════════════════════════
# 【A/B 共用】控制协议
#
# 走**已经存在的那个端口**（= 小地图推流的 TCP 口，`config/link.yaml` 的
# minimap.port）：那条连接 TCP 本来就是全双工的，多连一条专门说话就行 ⇒
# **不用新端口、不用新的防火墙规则、不用多起一个常驻服务** ✓。
#
# 为什么**单独一条连接**而不是往帧流里塞命令：B 机收帧时把「长度 ≤ 0」当协议
# 错误（`perception/minimap.py` 读帧头那段），把控制字节混进帧里既会让旧版本
# 客户端解错，也要动已经稳定的帧格式。连上先发 `CTL_HELLO` 报个到 ⇒ A 机按
# 控制连接对待（不往它推帧），对老客户端零影响 ✓。
# ══════════════════════════════════════════════════════════════

#: 连上后**第一件事**发这个 ⇒ 这是一条控制连接（不是要收帧的推流连接）
CTL_HELLO = b"HELLO mmap-ctl\n"

#: 握手的等待上限（秒）：accept 之后只肯等这么久，超时就按推流连接处理。
#: 短一点是为了不拖慢帧循环 —— 正常连上来会**立刻**发握手。
CTL_HANDSHAKE_TIMEOUT = 0.25

CMD_MAP = "MAP"          # MAP <map_id>  → 切到那张图的区域（下一帧生效）
CMD_STATE = "STATE"      # STATE         → 现在在推哪张图 / 哪个框
CMD_LIST = "LIST"        # LIST          → 库里已经配了哪些图
CMD_PING = "PING"        # PING          → PONG（探活）


# ---------------- 回执（两边只用这两套收发，别各写一份） ----------------

def ok_reply(mid, box, zoom):
    """成功：**切过之后**回报实际在用的那一套（区域与 zoom 都要带回去 ✓）。

    `box = (x, y, w, h)`。B 机拿这个 zoom 去挑/校对该图的标定（见
    `core.mapdata.load_calib` 的 zoom 参数）—— **标定是按某个 zoom 标出来的**，
    zoom 变了还套旧标定，坐标会整倍数错且不报错（就是这次要修的那个坑）。
    """
    x, y, w, h = (int(v) for v in box)
    return "OK %s %d %d %d %d %d" % (str(mid or ""), x, y, w, h, int(zoom or 1))


def err_reply(code, why=""):
    """失败。`code` 用短横线词（`no-such-map` / `bad-command` …），`why` 是人话。"""
    return "ERR %s %s" % (str(code or "err"), str(why or "").replace("\n", " "))


def list_reply(ids):
    """回了哪些图。空的时候就是 `LIST `（后面没东西）。"""
    return "LIST %s" % ",".join(str(i) for i in (ids or []))


def state_reply(mid, box, zoom):
    """回当前状态；**还没指定图**时 `-`。"""
    return "STATE %s" % ok_reply(mid, box, zoom).split(" ", 1)[1]


def parse_reply(line):
    """一行回执 → dict；两边各自的"还算人话"的判断都在这里。

    → `{"ok": True, "kind": "MAP"|"STATE"|"LIST"|"PING", "map_id": str,
         "box": [x,y,w,h], "zoom": int, "ids": [...]}`
      或 `{"ok": False, "code": str, "why": str}`
      或 `None`（空/看不懂 ⇒ **当作不知道**，别猜 ✓）

    ⚠ `-`（STATE 表示"没指定图"）会翻成 `map_id == ""`，不要把它当成一个真的 id。
    """
    s = str(line or "").strip()
    if not s:
        return None
    parts = s.split(None, 1)
    head, rest = parts[0].upper(), (parts[1] if len(parts) > 1 else "")
    if head == "ERR":
        rp = rest.split(None, 1)
        return {"ok": False, "code": (rp[0] if rp else "err"),
                "why": (rp[1] if len(rp) > 1 else "")}
    if head == "PONG":
        return {"ok": True, "kind": "PING"}
    if head == "LIST":
        return {"ok": True, "kind": "LIST",
                "ids": [v for v in rest.replace(" ", "").split(",") if v]}
    if head not in ("OK", "STATE"):
        return None
    f = rest.split()
    if head == "STATE":
        # STATE [-] x y w h zoom    （没指定图时 map_id 那一格是 `-`）
        if len(f) == 5:                        # STATE x y w h zoom（无 id）
            mid, nums = "", f
        elif len(f) == 6:
            mid, nums = f[0], f[1:]
        else:
            return None
        if mid == "-":
            mid = ""
    else:
        if len(f) != 6:
            return None
        mid, nums = f[0], f[1:]
    try:
        box = [int(v) for v in nums[:4]]
        zoom = int(nums[4]) if len(nums) > 4 else 1
    except (TypeError, ValueError):
        return None
    return {"ok": True, "kind": ("STATE" if head == "STATE" else "MAP"),
            "map_id": mid, "box": box, "zoom": max(1, zoom)}


# ══════════════════════════════════════════════════════════════
# 【A 机】每图区域的存 / 取
# ══════════════════════════════════════════════════════════════

def path_of(map_id):
    """这张图的区域文件：`config/minimap_regions/<map_id>.json`。"""
    mid = str(map_id or "").strip()
    return DIR / ("%s.json" % mid)


def _clean(d, map_id):
    """原始 dict → **洗干净的**区域 dict；不合格返回 None（**不猜** ✓）。

    坏值一律当"没填" —— 理由同 `crop_of`：拿半截数去抓屏，推的是一片无关画面，
    B 机那边表现为「底图对不上」，看着完全像寻路坏了 ✗。
    """
    if not isinstance(d, dict):
        return None
    try:
        x, y = int(d.get("x")), int(d.get("y"))
        w, h = int(d.get("w")), int(d.get("h"))
    except (TypeError, ValueError):
        return None
    if x < 0 or y < 0 or w < MIN_WH or h < MIN_WH:
        return None
    try:
        zoom = max(1, int(d.get("zoom") or 1))
    except (TypeError, ValueError):
        zoom = 1
    return {"map_id": str(d.get("map_id") or map_id),
            "x": x, "y": y, "w": w, "h": h, "zoom": zoom,
            "note": str(d.get("note") or ""),
            "updated": str(d.get("updated") or "")}


def load(map_id):
    """读这张图的区域 → dict | None（**None = 这张图还没框过**，不是"一块都没配"）。

    ⚠ **这里不做任何兜底**：想知道"现在到底推哪一块"，用 `startup_region()` ——
    兜底规则（老单值、以及"当前图没框过就别顶替"）**只有那一处** ✓
    （口径同 B 机侧 `perception.minimap.crop_of` 的老兜底 ✓）。
    """
    mid = str(map_id or "").strip()
    if not mid:
        return None
    p = path_of(mid)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                        # noqa: BLE001
        return None                       # 文件坏了 = 没配过（让人重框，别把它当真值）
    return _clean(raw, mid)


def legacy():
    """老的那份**单值**区域（升级前的 `config/minimap_region.json`）→ dict | None。

    它的 `zoom` 取老配置里的（缺省 1）—— 那份配置本来就带着 zoom，别在这猜 ✓。
    """
    if not LEGACY_REGION_FILE.exists():
        return None
    try:
        raw = json.loads(LEGACY_REGION_FILE.read_text(encoding="utf-8"))
    except Exception:                                        # noqa: BLE001
        return None
    box = None
    if isinstance(raw, dict):
        box = raw.get("region")
    if isinstance(raw, list):
        box = raw
    if not box or len(box) != 4:
        return None
    d = dict(zip(("x", "y", "w", "h"), box))
    if isinstance(raw, dict):
        d["zoom"] = (raw.get("zoom") or 1)
    return _clean(d, "")


def save(map_id, x, y, w, h, zoom=1, note=""):
    """写一张图的区域。**已经洗过的值**才准往里传（调用方负责 `str→int`）。"""
    mid = str(map_id or "").strip()
    if not mid:
        raise ValueError("没有指定地图 id —— 不知道这次框的是哪张图就没法存")
    body = {"map_id": mid, "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "zoom": max(1, int(zoom or 1)), "note": str(note or ""),
            "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    p = path_of(mid)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def remove(map_id):
    """删掉一张图配的那份（**只是忘了的配置**，不是禁用）。没删过返回 False。"""
    p = path_of(map_id)
    if not p.exists():
        return False
    p.unlink()
    return True


def list_ids():
    """库里已经配了哪些图（排好序的 id 列表）。没有就是 `[]`。"""
    if not DIR.exists():
        return []
    out = [p.stem for p in DIR.glob("*.json") if p.is_file()]
    # `.json` 结尾但和 `*.json` 匹配不到 ⇒ stem 上是最后一个点的处理：`Path.stem`
    # 对 "105090600.json" 给出 "105090600" ✓；带多个点的 id 名按文件名前那段取。
    return sorted(out)


def ids_including_legacy():
    """`list_ids()` + （老的那份存在时）一个 `""` —— 给 CLI/界面"还能选哪些"用。"""
    out = list_ids()
    if legacy() is not None:
        out = out + [""]
    return out


# ══════════════════════════════════════════════════════════════
# 【A 机】"现在该推哪张图"（B 机推来的那个 id，记在本机）
# ══════════════════════════════════════════════════════════════

def current():
    """本机记着的"当前图" → `{"map_id", "by", "updated"}` | `None`（还没人告诉过）。

    ⚠ **只是一份记忆**，不是区域：真正的框在 `path_of(map_id)` 那份里 ✓
      （这里回答的是"框选该存给谁 / 启动该推哪张"✓）。
    """
    if not CURRENT_FILE.exists():
        return None
    try:
        raw = json.loads(CURRENT_FILE.read_text(encoding="utf-8"))
    except Exception:                                        # noqa: BLE001
        return None                       # 文件坏了 = 没记过（**不猜** ✓）
    if not isinstance(raw, dict):
        return None
    mid = str(raw.get("map_id") or "").strip()
    if not mid:
        return None
    return {"map_id": mid,
            "by": str(raw.get("by") or ""),
            "updated": str(raw.get("updated") or "")}


def set_current(map_id, by=""):
    """记下"现在该推这张图"（`by` = 谁说的：`"B机"` / `"手动"` ✓）。

    `map_id` 空 ⇒ **删掉这份记忆**（= 回到"还不知道是哪张图"✓，别留一个半截 id ✗）。
    """
    mid = str(map_id or "").strip()
    if not mid:
        return clear_current()
    body = {"map_id": mid, "by": str(by or ""),
            "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    CURRENT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CURRENT_FILE.write_text(json.dumps(body, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    return CURRENT_FILE


def clear_current():
    """忘掉"当前图"。没记过返回 False。"""
    if not CURRENT_FILE.exists():
        return False
    CURRENT_FILE.unlink()
    return True


def startup_region(map_id=None):
    """**推流启动时到底推哪一块** → `(mid, box, zoom, src)` | `None`（一块都定不下来）。

    优先级**只有这一处**（`tools/minimap_push` 与部署台自检**共用** ✓ 别各写一份 ✗）：
      ① 命令行给了 `--map-id` ⇒ 用**那张图**的框（`load` ✓ 查不到 ⇒ 整条作废 ✗ 别退到别人的框 ✗）；
      ② 没给 ⇒ 用**本机记着的当前图**（`current()` ✓ 那是 B 机最近一次推来的 ✓）；
         ⚠ **当前图明确、可它还没框过 ⇒ 也不退老单值**（`None` ✓）：老那份是**另一张图**的
           屏幕坐标 ⇒ 顶上就是"推错一块画面"✗（B 机那边看着像寻路坏了 ✓ 最难查的一类）。
      ③ 前两条都没有（**本机还没任何 id 记忆**）⇒ 用**老的单值兜底**（`legacy()` ✓ 升级后
         不用重框一遍 ✓ —— 口径同 B 机侧 `perception.minimap.crop_of` 的老兜底 ✓）；
      ④ 都没有 ⇒ `None` ⇒ 调用方**照旧起监听**、等 B 机 `MAP` 给图（用户 2026-10-02 的
         "只靠 B 机推"就是这么闭环的 ✓ —— 没框过也能先把服务开起来 ✓）。

    `src` 说明这块框是从哪来的（打进日志/界面用 ✓，排查时一眼看得出"它为什么推这块"）：
    `"map-id"` / `"当前图"` / `"老单值"`。
    """
    mid = str(map_id or "").strip()
    if mid:
        got = load(mid)
        if got:
            return (got["map_id"], [got["x"], got["y"], got["w"], got["h"]],
                    int(got["zoom"]), "map-id")
        return None                       # 点名了却没有 ⇒ 别拿别的图顶上 ✗
    cur = current()
    if cur:
        got = load(cur["map_id"])
        if got:
            return (got["map_id"], [got["x"], got["y"], got["w"], got["h"]],
                    int(got["zoom"]), "当前图")
        return None                       # 当前图明确了却没框过 ⇒ 不回退老单值（见上面 ② ✓）
    old = legacy()
    if old:
        return ("", [old["x"], old["y"], old["w"], old["h"]],
                int(old["zoom"]), "老单值")
    return None
