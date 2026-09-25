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
        "out_scale": 0.8004,
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
    },
    "mmap": {            # 小地图推流（寻路定位用；不用寻路就别起这张卡片）
        "bind": "0.0.0.0",
        "port": 5003,    # 与 link.yaml 的 minimap.port 一致
        "zoom": 3,
        "fps": 10,
        "quality": 100,
        # 小地图面板在 A 机屏幕上的矩形（屏幕坐标）。None = 还没框选，
        # 界面上的「框选…」按钮就是往这四个键里写。
        "x": None,
        "y": None,
        "w": None,
        "h": None,
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
    put("probe", "out_scale", probe.get("out_scale"))
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
    """写配置。失败只返回 False，由调用方决定要不要提示（不抛异常）。"""
    try:
        PATH.parent.mkdir(parents=True, exist_ok=True)
        PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        return True
    except Exception:
        return False


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
