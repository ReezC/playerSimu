"""部署台设置：读写 config/deploy.json（缺省值从 config/link.yaml 推）。

**为什么另存一份，而不是直接改 config/link.yaml**
    link.yaml 是**双机共用**的链路配置：B 机读它收流、探针坐标也在里面。
    A 机的部署台要是直接改它，很容易连带把 B 机依赖的字段一起动到；而且两台
    机器各有一份文件，改了一边另一边不会自动同步 —— 「看起来同步了、其实没有」
    是最坑的。所以这里只存 A 机这一侧的开关与参数，
    「和 link.yaml 对不对得上」交给界面上的自检项**明说**，不做隐式写入。

**每次改动立刻落盘**：需求就是「关闭后下次打开能复现」，所以不留内存态。
写入是几十字节的小 json，改一次存一次不会成为负担。
"""

import copy
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "config" / "deploy.json"
LINK = ROOT / "config" / "link.yaml"

# 默认值 = 「配置里允许有哪些字段」的唯一定义。read() 会按它补齐缺的键，
# 所以以后加了新参数，老的 deploy.json 也能直接打开（新字段取默认值）。
DEFAULTS = {
    "clock": {
        "host": "0.0.0.0",
        "port": 5001,
    },
    "probe": {
        # ⭐ **探针放在哪（流画面坐标 ✓ 2026-10-17 用户要求 ✓）** —— 以前只有 out_scale，
        #   位置只能手改 `link.yaml` 的 `probe.x/y` ✗。现在这几个键在这里存（= 部署台界面）
        #   ＋ 保存时**写回 link.yaml**（A、B 共用一份 ⇒ B 才知道该在哪采样 ✓）。
        #   ⚠ 默认值与 `link.yaml` 的回落值一致（见 `probe_gen` 的 `get("probe", ...)` ✓）。
        "x": 100,
        "y": 8,
        "cell": 16.0,
        "gap": 2.0,
        "bits": 40,
        # ⚠ `out_scale` **已移除**（用户 2026-10-17 ✓「把探针输出缩放参数移除」✓）：
        #   它现在是**派生值**（`screen_out_scale(cfg)` ✓ 由 `[screen]` + 推流宽算 ✓），
        #   `apply_derived()` 临时塞进内存给命令行用 ✓ **不落盘** ✗
        #   （落盘就会变成第二份真相 ⇒ 改分辨率忘改它 ⇒ 探针画错位置 ✓ 老口径的病根 ✓）。
    },
    # ⭐⭐⭐ **A 机屏幕 / DPI**（用户 2026-10-17 ✓ 部署台最上面那张「分辨率设置」卡片 ✓）
    #   这三项是**唯一来源** ⇒ 所有派生值（逻辑分辨率 / 输出缩放）一律现算 ✓
    #   （`core/screen.py` ✓ A、B 共用同一个算法 ✓）。
    "screen": {
        "width": 2560,       # A 机屏幕**物理**分辨率
        "height": 1440,
        "dpi_scale": 1.5,    # Windows 显示缩放：100%⇒1.0 / 125%⇒1.25 / 150%⇒1.5
    },
    "kbd": {
        "serial": "COM3",
        "port": 9000,
        "cert": "remote_kbd/certs/cert.pem",
        "key": "remote_kbd/certs/key.pem",
    },
    "push": {
        "ffmpeg": "ffmpeg",
        "capture": "ddagrab",
        "encoder": "h264_nvenc",
        "host": "192.168.1.2",
        "port": 5000,
        "width": 1366,
        "height": 768,
        "fps": 50,
        "bitrate": "6M",
        "gop": 60,
        "low_latency": True,
        "passthrough": False,
        "pkt_size": 1316,
        # ⭐ **推流区域**（用户 2026-10-17 ✓「没有在部署台看到选择推流区域的按钮」✗）：
        #   四个 0 = 整屏（老行为 ✓ 不抓区域 ✗）；由卡片上的「框选…」写入 ✓
        #   （A 机屏幕**逻辑坐标** ✓ = `ask_region` 抓屏那套坐标系 ✓）。
        "region_x": 0,
        "region_y": 0,
        "region_w": 0,
        "region_h": 0,
        # ⭐ **跟随窗口**（用户 2026-10-17 ✓「就不能像本地一样自动适应窗口尺寸吗？」✓）：
        #   `region_win = [hwnd, 标题]` + `region_rel = [nx, ny, nw, nh]`（**相对窗口外框** ✓
        #   与本地 `rect_rel`/`rect_win` 同一套 ✓ 见 `core/wincap`）。
        #   `None` = 不跟随（用上面那四个绝对坐标 ✓ 老行为 ✓）。
        "region_win": None,
        "region_rel": None,
    },
    "mmap": {            # 小地图推流（寻路定位用；不用寻路就别起这张卡片）
        "bind": "0.0.0.0",
        "port": 5003,    # 与 link.yaml 的 minimap.port 一致
        # ⭐ 「放大倍数」的默认值 —— 点「框选…」时写进**那张图**那份里
        #   （`config/minimap_regions/<id>.json` 的 zoom ✓ 两者必须同一份：标定是按
        #   某个 zoom 标的，见 core.mapdata.load_calib）。改它可以在当前图的框上
        #   直接生效（部署台会把 zoom 同步进那张图的文件 ✓）。
        "zoom": 3,
        # ⭐ 2026-10-03：**帧率 30 → 60、质量 100 → 80**（用户批的 ✓ 原话："A机 fps 已经
        #   60 了，quality 允许降；游戏（小地图）应该有 200+ [刷新率]"）——
        #   小地图是权威世界坐标 ⇒ 帧率越高越新 ✓；B 机侧已经有"按推流帧率取用"的高频
        #   定位回路（定位 ~0.5ms ✓）⇒ 30 那次提不上来的理由（B 机每帧都要解）已经不成立 ✓。
        #   quality 80：面板就一两百像素，80 与 100 肉眼无差，而 JPEG 体积约减半 ⇒
        #   省下的正是 A 机编码时间与带宽（60fps 的开销来源 ✓）。
        "fps": 60,
        "quality": 80,
        # ⛔ **这里曾经有 x/y/w/h**（小地图面板在 A 机屏幕上的矩形）—— 2026-10-02 删掉 ✗：
        #   用户定「当前是哪张图**只由 B 机推**」，区域随之改成**按地图 id 各存一份**
        #   （`config/minimap_regions/<id>.json`，A 机本地 ✓ 见 `tools/mmap_regions.py`）。
        #   留着一份"全局的"正是用户 2026-09-27 在 B 机侧踩过的那个坑：**换图还推上一张
        #   图的框** ✗（A 机这边同样是"屏幕坐标"，换图/换布局就全错 ✓）。
        #   ⚠ 老 deploy.json 里可能还留着这四个键 ⇒ `read()` 只按 DEFAULTS 补键、
        #     不认识的键**直接丢掉** ✓（不读、也不再写回 ✓）。
    },
    "sweep": {           # 推流自检的握手通道（与 link.yaml 的 sweep.port 一致）
        # A 机往 B 机的这个 UDP 口发"第几段开始"，B 机的回执也回到这个口上
        # （见 tools/sweep_link.py）。两边都绑这个号，所以 A 机自己也占着它。
        "port": 5002,
    },
    "window": {          # 上次的窗口大小，下次开原样
        "w": 1280,
        "h": 860,
    },
    "log": {             # 上次选的日志筛选
        "filter": "all",
        # 右下角「运行日志」保留多少行历史（设置弹窗里改）。
        # **必须在 DEFAULTS 里声明**：read() 的 _deep_merge 只认这里已有的键，
        # 没声明的键会被静默丢掉 —— 表现为「设置里改了、下次打开又变回默认」。
        "max_lines": 4000,
    },
}


def link_cfg():
    """读 config/link.yaml；不在或坏了返回 {}。"""
    try:
        with open(LINK, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _deep_merge(base, data):
    """把 data 合进 base：**只认 base 里已有的键**，多出来的垃圾直接忽略。

    这样配置文件被手工改坏（多了一堆没用的键、类型写错）也不会带进界面。
    """
    for k, v in (data or {}).items():
        if k not in base:
            continue
        if isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def seed_from_link():
    """首次运行的默认值：能从 link.yaml 推的都推过来，省得手输一遍。"""
    cfg = copy.deepcopy(DEFAULTS)
    lk = link_cfg()
    stream = lk.get("stream") or {}
    clock = lk.get("clock_sync") or {}
    probe = lk.get("probe") or {}
    kbd = lk.get("kbd") or {}

    def put(section, key, val):
        if val not in (None, ""):
            cfg[section][key] = val

    put("clock", "port", clock.get("server_port"))
    # ⭐ **屏幕 / DPI**（2026-10-17 ✓）：首次运行按 `link.yaml [screen]` 填好 ✓
    #   （`out_scale` 不再从 link.yaml 推 ✗ —— 它现在是派生值 ✓ 见 `apply_derived` ✓）
    _sc = lk.get("screen") or {}
    for _k in ("width", "height", "dpi_scale"):
        put("screen", _k, _sc.get(_k))
    # ⭐ 探针**位置/尺寸**也一并推过来（2026-10-17 ✓）：老部署台没有这几项 ⇒ 首次读完
    #   就按 link.yaml 现值填好 ✓（省得人照着 yaml 手抄一遍 ✗）。
    for _k in ("x", "y", "cell", "gap", "bits"):
        put("probe", _k, probe.get(_k))
    put("kbd", "port", kbd.get("port"))
    put("kbd", "serial", kbd.get("serial"))
    put("kbd", "cert", kbd.get("cert"))
    # 私钥没写在 link.yaml 里（agent 不用），按证书同目录推
    cert = str(cfg["kbd"]["cert"] or "")
    if cert.endswith("cert.pem"):
        put("kbd", "key", cert[:-len("cert.pem")] + "key.pem")

    put("push", "host", lk.get("b_host"))
    put("push", "port", stream.get("port"))
    put("push", "width", stream.get("width"))
    put("push", "height", stream.get("height"))
    put("push", "fps", stream.get("fps"))
    # 小地图推流端口（B 机 perception/minimap.py 读同一个键）。
    # **区域不在这里**：它依 A 机屏幕布局，只能框出来，不能从配置推。
    put("mmap", "port", (lk.get("minimap") or {}).get("port"))
    # 自检握手端口（B 机 tools/stream_sweep.py 读同一个键）
    put("sweep", "port", (lk.get("sweep") or {}).get("port"))
    return cfg


def read():
    """读配置：没有文件 → 用 link.yaml 推的默认值并**落盘一份**（方便直接看/改）。"""
    if not PATH.exists():
        cfg = seed_from_link()
        save(cfg)
        return cfg
    try:
        data = json.loads(PATH.read_text(encoding="utf-8"))
    except Exception:
        # 坏了不要崩：退回默认值，界面照常能开
        return copy.deepcopy(DEFAULTS)
    return _deep_merge(copy.deepcopy(DEFAULTS), data)


def save(cfg):
    """写配置。失败只返回 False，由调用方决定要不要提示（不抛异常）。

    ⚠ **派生值不落盘** ✗（用户 2026-10-17 ✓「一律走取这个配置的自计算」✓）：
      `probe.out_scale` 是算出来的（`apply_derived` 只在内存里塞 ✓）——
      写进 `deploy.json` 就会变成**第二份真相** ✓（改分辨率忘改它 ⇒ 探针画错位置 ✓
      那正是老口径的病根 ✓）。
    """
    try:
        PATH.parent.mkdir(parents=True, exist_ok=True)
        out = dict(cfg or {})
        if "probe" in out:
            _p = dict(out.get("probe") or {})
            _p.pop("out_scale", None)          # 派生值 ⇒ 不进文件 ✓
            out["probe"] = _p
        PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        return True
    except Exception:
        return False


def screen_out_scale(cfg):
    """**输出缩放** = 推流宽 ÷ A 机逻辑分辨率宽 ✓（用户 2026-10-17 ✓）。

    算法**只有一份**（`core/screen.out_scale` ✓）—— 这里只是把它接到
    `cfg["screen"]` / `cfg["push"]["width"]` 上 ✓；B 机那份读 `link.yaml` 同一个模块 ✓。
    """
    from core import screen as _screen

    sc = (cfg or {}).get("screen") or {}
    push = (cfg or {}).get("push") or {}
    try:
        w = int(float(push.get("width") or 0))
        # ⚠ 一律走 `core/screen.out_scale` ✗ **别在这儿自己除** ——
        #   那会变成第二份算法 ✓（用户要的是"一处自计算"✓ 以后改口径只改那儿 ✓）。
        return float(_screen.out_scale(w, sc.get("width"), sc.get("height"),
                                       sc.get("dpi_scale")))
    except (TypeError, ValueError):
        return 1.0


def apply_derived(cfg):
    """把**派生值**刷进内存配置 ✓（现在只有 `probe.out_scale`）。

    ⚠ **不落盘** ✗：`deploy.json` 里只存真相（`screen.*` / `push.width` ✓）——
      派生值一落盘就成第二份真相 ✓（改分辨率忘改它 ⇒ 探针画错 ✓ 老口径的病根 ✓）。
    """
    if not cfg:
        return cfg
    cfg.setdefault("probe", {})["out_scale"] = screen_out_scale(cfg)
    return cfg


def sync_screen_to_link(cfg):
    """把屏幕三项写回 `config/link.yaml` 的 `[screen]`（**A、B 共用** ✓）。

    用户 2026-10-17 ✓：「需要用到的地方（包括 **B 机**如果需要这些数据的话）
    一律走取这个配置的自计算」✓ ⇒ A 机在部署台填一次，写回共用配置，
    B 机（探针采样 / 换算 / 设置界面显示）读同一份 + 用同一个算法 ✓。
    ⚠ 派生值（`out_scale`）**不写回** ✗（谁要谁算 ✓ 一份真相 ✓）。
    """
    from core.config import update_link

    sc = (cfg or {}).get("screen") or {}
    kw = {}
    for _k in ("width", "height", "dpi_scale"):
        _v = sc.get(_k)
        if _v in (None, ""):
            continue
        try:
            kw["screen." + _k] = float(_v)
        except (TypeError, ValueError):
            return False, "「%s」不是数字：%r" % (_k, _v)
    if not kw:
        return False, "没有可写的屏幕参数"
    try:
        update_link(**kw)
    except Exception as e:                # noqa: BLE001
        return False, "%s: %s" % (type(e).__name__, e)
    return True, ("屏幕参数已写回 config/link.yaml（%s）⇒ 输出缩放自动 = %.4f"
                  % (" ".join("%s=%g" % (k.split(".", 1)[1], v) for k, v in kw.items()),
                     screen_out_scale(cfg)))


def sync_probe_to_link(probe):
    """探针几何写回 `config/link.yaml` 的 `[probe]`（**A、B 共用一份** ✓）。

    ⭐⭐⭐ 为什么必须写回（用户 2026-10-17 ✓ 原话：「**延迟探针可以让 A 机自己在部署台
      配置放在哪吗？**」✓）：
      · A 机（部署台）决定\"画在哪\" ✓；B 机是按 `link.yaml` 里**同一组数字采样** ✓
        ⇒ 只改 `deploy.json`（A 机自己用）的话，B 机还在**老位置**采 ✗
        ⇒ 整条码解不出来 ✓（表现为\"探针明明移动了、延迟却一直 100% 失败\"✓）。
      · 走 `core.config.update_link` ✓ —— 它**逐行改值**，注释与排版一个字节都不动 ✓
        （`link.yaml` 是注释密集的说明书 ✓ 整文件 dump 会把注释全冲掉 ✗ 见它自己的说明 ✓）。
    返回 `(ok, msg)`；`ok=False` 时 `msg` 是原因（调用方打在运行日志里 ✓ 不许静默 ✗）。
    """
    from core.config import update_link

    pr = probe or {}
    kw = {}
    for _k, _cast in (("x", int), ("y", int), ("cell", float), ("gap", float),
                      ("bits", int)):
        _v = pr.get(_k)
        if _v in (None, ""):
            continue                      # 没填 ⇒ 这一项不动（别把已有值抹掉 ✗）
        try:
            kw["probe." + _k] = _cast(_v)
        except (TypeError, ValueError):
            return False, "「%s」不是合法数字：%r" % (_k, _v)
    if not kw:
        return False, "没有可写的探针几何（x/y/cell/gap/bits 都是空的）"
    try:
        update_link(**kw)
    except Exception as e:                # noqa: BLE001 —— 文件只读 / 格式坏了 ✓ 说清楚
        return False, "%s: %s" % (type(e).__name__, e)
    return True, ("探针几何已写回 config/link.yaml："
                  + " ".join("%s=%s" % (k.split(".", 1)[1], v) for k, v in kw.items()))


def sync_push_size_to_link(w, h):
    """推流分辨率写回 `config/link.yaml` 的 `stream.width/height`（**B 机按它准备** ✓）。

    ⭐ 为什么要写（用户 2026-10-17 ✓ 加了「框选推流区域」之后这件事变重要了 ✓）：
      · 框了区域 ⇒ 输出尺寸**跟着框变** ✓；`link.yaml` 里那份要是还写着 1366×768，
        B 机（和探针 re-solve）就按**老尺寸**算 ⇒ 画面比例 / 探针坐标全错 ✗；
      · 走 `core.config.update_link` ✓（逐行改值，注释不动 ✓ 同 `sync_probe_to_link` ✓）。
    返回 `(ok, msg)`；`ok=False` 时 `msg` 是原因（调用方打在运行日志里 ✓）。
    """
    from core.config import update_link

    try:
        _w, _h = int(w), int(h)
    except (TypeError, ValueError):
        return False, "尺寸不是数字：%r×%r" % (w, h)
    if _w <= 0 or _h <= 0:
        return False, "尺寸无效：%d×%d" % (_w, _h)
    try:
        update_link(**{"stream.width": _w, "stream.height": _h})
    except Exception as e:                # noqa: BLE001
        return False, "%s: %s" % (type(e).__name__, e)
    return True, "已写回 config/link.yaml（stream %d×%d）" % (_w, _h)


def link_expect():
    """link.yaml 里与 A 机有关的期望值，给「一致性」自检用。

    返回 {字段说明: (link.yaml 里的值, 读到的路径描述)}，缺的项不出现在结果里。
    """
    lk = link_cfg()
    stream = lk.get("stream") or {}
    clock = lk.get("clock_sync") or {}
    kbd = lk.get("kbd") or {}
    out = {}
    pairs = (
        ("B 机 IP", lk.get("b_host"), "push.host"),
        ("推流端口", stream.get("port"), "push.port"),
        ("流分辨率", (stream.get("width"), stream.get("height")), "push.size"),
        ("推流帧率", stream.get("fps"), "push.fps"),
        ("对时端口", clock.get("server_port"), "clock.port"),
        ("键盘端口", kbd.get("port"), "kbd.port"),
        ("串口号", kbd.get("serial"), "kbd.serial"),
        ("小地图端口", (lk.get("minimap") or {}).get("port"), "mmap.port"),
        ("自检握手端口", (lk.get("sweep") or {}).get("port"), "sweep.port"),
    )
    for label, val, where in pairs:
        if val not in (None, "", (None, None)):
            out[label] = (val, where)
    return out
