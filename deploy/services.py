"""被控机（A 机）服务的参数表 + 命令行构造。

**参数表驱动界面**（deploy/app.py 照着 PARAMS 生成控件）：这样「界面能改什么」
和「命令里能传什么」天然一致 —— 不会出现界面上摆了开关、命令却不认，或者反过来
改了参数但命令没带上。

**A 机上要跑的事**（手册见 docs/A_SETUP.md）：
    clock  对时服务。B 机的延迟统计依赖它，**必须先在跑**。
    probe  屏幕时间码探针。B 机靠画面里这条码测端到端延迟。
    kbd    键盘中继。收 B 机 agent 的按键指令 → 转发给 Pro Micro 硬件键盘。
    push   屏幕推流。桌面 → H.264 → UDP/mpegts 发给 B 机。
    mmap   小地图推流（**只在用寻路定位时才要**）。小地图面板单独截屏 →
           JPEG/TCP 给 B 机；主画面那一路压过一遍，黄点只有几个像素就糊了。

**顺序也是部署顺序**（ORDER）：对时 → 探针 → 键盘 → 推流 → 小地图。
小地图排在最后：前面的没起来时，它推出去也没人收。
"""

import shutil
import sys
import time
from pathlib import Path

# 部署顺序 = 界面上的排列顺序 = 「全部启动」的顺序
ORDER = ("clock", "probe", "kbd", "push", "mmap")

TITLE = {
    "clock": "时钟对时服务",
    "probe": "屏幕时间码探针",
    "kbd": "键盘中继（Pro Micro）",
    "push": "屏幕推流（ffmpeg）",
    "mmap": "小地图推流",
}

SUB = {
    "clock": "B 机的延迟统计依赖它，先起这个",
    "probe": "画面顶部那条黑白码，B 机靠它量端到端延迟",
    "kbd": "把 B 机发来的按键转发给 Pro Micro 硬件键盘",
    "push": "桌面 → H.264 → UDP，发给 B 机收流",
    "mmap": "小地图面板单独截屏发给 B 机（寻路定位用，不用寻路就别起）",
}

# 每个参数一项：key / 标签 / 控件类型 / 提示。
#   kind: int | float | text | check | choice | combo_edit | size | serial | path
#         | region（「框选…」按钮，值写进 keys 里的四个坐标键）
#   keys: 真正写进配置的键（只有「分辨率」和「小地图区域」是多键）
PARAMS = {
    "clock": [
        dict(key="port", keys=("port",), label="UDP 端口", kind="int",
             minimum=1, maximum=65535, width=110,
             tip="A 机监听这个端口做对时。\n"
                 "**必须与 config/link.yaml 的 clock_sync.server_port 一致** ——\n"
                 "B 机按那份配置来问时间，不一致就永远对不上。\n\n"
                 "改了端口记得在（管理员）PowerShell 放行入站 UDP：\n"
                 'netsh advfirewall firewall add rule name="playerSimu clock 端口" '
                 "dir=in action=allow protocol=UDP localport=端口"),
        dict(key="host", keys=("host",), label="监听地址", kind="text", width=140,
             tip="0.0.0.0 = 所有网卡（推荐，换网络不用改）"),
    ],
    "probe": [
        dict(key="out_scale", keys=("out_scale",), label="输出缩放", kind="float",
             minimum=0.05, maximum=4.0, decimals=4, step=0.01, width=110,
             tip="A 机屏幕 → 流画面的缩放比 = 流分辨率 ÷ A 机屏幕**逻辑**分辨率。\n"
                 "probe_gen 用它把 link.yaml 里的「流画面坐标」换算成 A 机屏幕\n"
                 "坐标去画，所以 A、B 共用同一份 link.yaml，A 机不用手工换算。\n\n"
                 "例：A 机 2560x1440、Windows 缩放 150%（逻辑 1706x960），\n"
                 "推流 1366x768 → 1366 ÷ 1706.67 ≈ 0.8004。\n"
                 "改了分辨率或缩放比例就要重算这个值（改完 B 机 re-solve 一次）。"),
    ],
    "kbd": [
        dict(key="serial", keys=("serial",), label="串口号", kind="serial", width=150,
             tip="Pro Micro 的串口（列表 = 当前插着的设备，会自动刷新）。\n"
                 "填错的表现是启动后日志里「串口打不开 / 无响应」。\n"
                 "必须与 config/link.yaml 的 kbd.serial（游戏机那侧）一致。"),
        dict(key="port", keys=("port",), label="TCP 端口", kind="int",
             minimum=1, maximum=65535, width=110,
             tip="等 B 机 agent 连上来的 TLS 端口，与 link.yaml 的 kbd.port 一致。"),
        dict(key="cert", keys=("cert",), label="证书", kind="cert", mode="file",
             filt="证书 (*.pem);;所有文件 (*)",
             tip="TLS 证书。**没有就点右边「生成证书…」** —— 一次生成一对，"
                 "并告诉你该把哪一份拷到控制机（B）。\n"
                 "注意：cert.pem 要与 B 机那份**一致**（B 拿它校验这台机器的证书），"
                 "私钥 key.pem 只留在这台机器。"),
        dict(key="key", keys=("key",), label="私钥", kind="path", mode="file",
             filt="私钥 (*.pem);;所有文件 (*)",
             tip="TLS 私钥（证书旁边那个 key.pem）。"),
    ],
    "push": [
        dict(key="host", keys=("host",), label="B 机 IP", kind="text", width=140,
             tip="推流发给谁。就是 B 机的 IPv4（在 B 机 ipconfig 看），\n"
                 "与 link.yaml 的 b_host 一致。"),
        dict(key="port", keys=("port",), label="推流端口", kind="int",
             minimum=1, maximum=65535, width=110,
             tip="B 机在收的 UDP 端口，与 link.yaml 的 stream.port 一致\n"
                 "（B 机自己监听 0.0.0.0 的这个端口）。"),
        dict(key="size", keys=("width", "height"), label="分辨率", kind="size",
             choices=("1366x768", "1280x720", "1600x900", "1920x1080"), width=130,
             tip="推流分辨率（= B 机看到的画面尺寸）。\n"
                 "与 link.yaml 的 stream.width/height 一致；改完 A 机的\n"
                 "「输出缩放」要跟着重算，B 机的探针坐标要 re-solve。\n"
                 "越小延迟越低 —— 131 ms 那个成绩就是 1920x1080@144 测的。"),
        dict(key="fps", keys=("fps",), label="帧率", kind="combo_edit", cast=int,
             choices=("50", "60", "144"), width=110,
             tip="采集帧率。实测抓得越快端到端延迟越低：\n"
                 "60fps → 316 ms，144fps → 131 ms（同一台机器）。\n"
                 "要和仿真环境本身的帧率对齐，不然会重复帧/丢帧。"),
        dict(key="bitrate", keys=("bitrate",), label="码率", kind="combo_edit",
             choices=("6M", "12M", "20M"), width=110,
             tip="目标码率。1366x768@50 用 6M 够；分辨率/帧率上去要跟着加。\n"
                 "局域网里给足带宽比省流量重要。"),
        dict(key="gop", keys=("gop",), label="GOP", kind="int",
             minimum=1, maximum=600, width=110,
             tip="关键帧间隔。约等于帧率的 1 秒量（60fps → 60）。\n"
                 "调小：丢包后恢复快、码率略涨；调大：省码率、但丢一个包要等更久。"),
        dict(key="capture", keys=("capture",), label="采集方式", kind="choice",
             choices=(("ddagrab（DXGI 硬采，推荐）", "ddagrab"),
                      ("gdigrab（GDI 软采，兜底）", "gdigrab")), width=230,
             tip="ddagrab：显卡直接抓桌面，延迟低、几乎不吃 CPU，需要 D3D11 + N 卡。\n"
                 "gdigrab：传统 GDI 抓屏，兼容性最好，但延迟高、CPU 占用高。\n"
                 "独占全屏的游戏用 ddagrab 才抓得到；无边框窗口两者都行。"),
        dict(key="scale_mode", keys=("scale_mode",), label="缩放位置", kind="choice",
             choices=(("CPU 缩放（默认，最稳）", "cpu"),
                      ("GPU 缩放（cuda，省内存带宽）", "cuda")), width=230,
             tip="整屏（A 机 2560×1440）缩到输出尺寸这一步**在哪做**，差别很大：\n"
                 "  CPU：抓完 hwdownload（整屏 BGRA ≈ 15MB/帧 拷回内存）+ CPU scale；\n"
                 "       实测这条链就是 A 机的天花板（speed 一直贴 1.00、实际 fps 只有 ~73）。\n"
                 "  GPU：hwmap 到 cuda 后 scale_cuda，只在显存里缩放，再把小图拷回来。\n\n"
                 "GPU 那条要 A 机 ffmpeg 支持 scale_cuda（Gyan 版有）。\n"
                 "**起不来的话「推流自检」会报 ffmpeg 没起来 —— 改回 CPU 即可**。\n"
                 "哪种更快，用「推流自检」量一轮看 speed 与实际 fps 就知道。"),
        dict(key="encoder", keys=("encoder",), label="编码器", kind="choice",
             choices=(("h264_nvenc（N 卡硬件编码）", "h264_nvenc"),
                      ("libx264（CPU 软编码）", "libx264")), width=230,
             tip="有 N 卡就用 h264_nvenc（本机自检会告诉你可不可用）。\n"
                 "没有就选 libx264 —— 720p 以内还能跑，再高 CPU 会跟不上。"),
        dict(key="low_latency", keys=("low_latency",), label="低延迟档", kind="check",
             tip="勾上 = 关掉编码器与 mpegts 的缓冲：\n"
                 "  -tune ull -zerolatency 1 -rc cbr\n"
                 "  -flush_packets 1 -muxdelay 0 -muxpreload 0\n"
                 "实测这些缓冲值上秒钟的量级，局域网里没有不勾的理由。"),
        dict(key="passthrough", keys=("passthrough",), label="不转帧率", kind="check",
             tip="-fps_mode passthrough：不做帧率转换。\n"
                 "不勾的话 ffmpeg 会默认按输入帧率重采样，实测多出约 225 ms 的\n"
                 "隐形缓冲（a_push.ps1 里的实测结论）。"),
        dict(key="ffmpeg", keys=("ffmpeg",), label="ffmpeg", kind="path", mode="file",
             filt="可执行文件 (*.exe);;所有文件 (*)",
             tip="ffmpeg.exe 的位置。装了但没进 PATH 时（本机就是这样）在这里指过去。\n"
                 "安装：winget install --id Gyan.FFmpeg -e"),
    ],
    "mmap": [
        # ⭐⭐ **「当前地图」+「小地图区域」= 只读两行**（用户 2026-10-02 定 ✓ 原话："只靠
        #   B 机推（A 机不能自己选）" + "地图与框选数据另存一份文件，仅 A 机本地储存"）：
        #   · 「当前地图」= 本机记着的那个 id（B 机 `MAP <地图id>` 顺手写下的 ✓ 见
        #     `tools/mmap_regions.set_current`）+ 右边一个「重读」；
        #   · 「小地图区域」= **当前这张图**那份框（`config/minimap_regions/<id>.json` ✓
        #     A 机本地）+ 右边**「框选…」**。
        #   ⚠ **deploy.json 里不再存 x/y/w/h**（已从 DEFAULTS 删掉 ✗）：区域跟着**地图 id**
        #     走，一台机上好几张图各一份 —— 存成一份"全局的"正是"换图还推上一张图的框"✗
        #     （用户 2026-09-27 在 B 机侧踩过同一个坑 ✓ 见 `perception.minimap.crop_of`）。
        #   ⚠ **没框过也能「启动」**：服务起来先监听、等 B 机 `MAP <id>`（见 `_mmap_cmd` ✓）——
        #     这就是"只靠 B 机推"能闭环的原因（id 是 B 机给的，框是拿到 id 之后才框的 ✓）。
        dict(key="current", keys=(), label="当前地图", kind="mmap_current", width=250,
             tip="**B 机现在跑的是哪张图** —— 由 B 机告诉本机（A 机不能自己选）。\n"
                 "B 机开始实时（以后还有换图）时会发一句 `MAP <地图id>`，这里跟着变；\n"
                 "右下角「框选…」存的也是这张图名下。\n\n"
                 "还空着 = 还没有机器告诉过本机 —— 先去 B 机开始实时。\n"
                 "点右边的「重读」可以立刻再查一次（A 机自己不会变，只能等 B 机说）。"),
        dict(key="region", keys=(), label="小地图区域", kind="mmap_region", width=250,
             tip="**当前这张图**的小地图面板在 A 机屏幕上的矩形（屏幕坐标）。\n"
                 "点右边的「框选…」，把**小地图面板本身**框出来。\n"
                 "框选时跟着光标的放大镜是 8×（按 +/- 调，4~16 倍）—— 面板边框\n"
                 "差一两个像素，在放大镜里数像素格对齐。\n\n"
                 "框的时候只框面板：多框进来的血条/聊天/其它 UI 会一起推给 B 机，"
                 "既占带宽，又会让 B 机「底图对不上」——那种现象看着像寻路坏了，"
                 "其实是区域框大了。\n\n"
                 "部署台会在框选前把自己藏起来再抓屏，所以框到的是游戏画面；\n"
                 "但**运行期间**没有这种保护 —— 推的是屏幕这一块，谁压在上面就推谁，\n"
                 "所以启动后别让别的窗口盖在小地图上。\n"
                 "游戏窗口移动过、换过分辨率/显示缩放、改过 UI 布局 → 要重新框一次。\n\n"
                 "⚠ 区域**按地图 id 分别存**，而且只存在这台机器上：换图/换项目不用重框，"
                 "各图各一份（另一台 A 机有自己的屏幕布局，不会互相覆盖 ✓）。\n"
                 "框完如果推流正在跑，它会自动重启一次（下一帧就用新区域）。"),
        dict(key="port", keys=("port",), label="TCP 端口", kind="int",
             minimum=1, maximum=65535, width=110,
             tip="A 机监听这个端口，B 机连进来收小地图帧。\n"
                 "**与 config/link.yaml 的 minimap.port 一致**（B 机的 "
                 "perception/minimap.py 读那份）。\n\n"
                 "改了端口要在（管理员）PowerShell 放行入站 TCP：\n"
                 'netsh advfirewall firewall add rule name="playerSimu 小地图端口" '
                 "dir=in action=allow protocol=TCP localport=端口"),
        dict(key="bind", keys=("bind",), label="监听地址", kind="text", width=140,
             tip="0.0.0.0 = 所有网卡（推荐，换网络不用改）。"),
        dict(key="zoom", keys=("zoom",), label="放大倍数", kind="int",
             minimum=1, maximum=8, width=110,
             tip="抓到的这块区域先放大几倍再发。\n"
                 "小地图面板本身像素很少，黄点只有 2~5 像素：放大只是让它在\n"
                 "JPEG 编码里少掉点细节，**不增加信息量**，B 机会按比例缩回去。\n"
                 "面板本身够大（≥150px）就不用放大。\n\n"
                 "⚠ **这个值跟着图存**（点「框选…」时写进那张图那份里 ✓）：标定是按\n"
                 "某个 zoom 标出来的，换图时得用它自己那个 zoom（见 core.mapdata.load_calib）。"
                 "所以它不写进 deploy.json 的区域里，改了它要重新框（或换图）才生效。"),
        # ⭐⭐ 2026-10-03（用户 ✓ 原话："A机 fps 已经 60 了，quality 允许降；游戏（小地图）
        #   应该有 200+ [刷新率]"）：小地图是**权威世界坐标** ⇒ 帧率越高坐标越新 ✓；
        #   而"B 机每帧都要解一次、匹配一次"这条**已经不是瓶颈**了（定位 ~0.5ms ✓，
        #   而且 B 机现在有**高频定位回路**按推流帧率取用 ✓ 见 `gui/live_thread.py`）⇒
        #   默认从 30 提到 **60** ✓；`quality` 100 → **80**（面板本就一两百像素 ✓ 80 肉眼
        #   看不出差别，JPEG 体积约减半 ⇒ A 机编码（它自述"编码是大头"✓）与带宽都省 ✓）。
        dict(key="fps", keys=("fps",), label="帧率", kind="combo_edit", cast=int,
             choices=("30", "45", "60", "90"), width=110,
             tip="用户 2026-09-27 定：**实时小地图位置是权威**，帧率越高位置状态越新 ✓。\n"
                 "默认 60（2026-10-03 从 30 提上来 —— B 机侧已经改成按推流帧率取用 ✓，\n"
                 "定位本身只要 ~0.5ms ⇒ 不再是瓶颈 ✓）。\n"
                 "⚠ 这一路的开销在 **A 机**（抓屏 + JPEG 编码，它自述「编码是大头」✓）和\n"
                 "带宽上 ⇒ 真吃紧再降回 30；**别低于游戏小地图自己的刷新率**（那会让坐标\n"
                 "变旧 ✗，用户的游戏是 200+ ✓ 所以越低损失越明显）。"),
        dict(key="quality", keys=("quality",), label="JPEG 质量", kind="int",
             minimum=30, maximum=100, width=110,
             tip="默认 80（2026-10-03 从 100 降下来 ✓ 用户批的：「quality 允许降」）。\n"
                 "这一路的**唯一价值**就是像素清晰（主画面那路压过一遍，黄点早糊了），\n"
                 "但面板本身才一两百像素 ⇒ 80 与 100 的差别肉眼看不出来，而 JPEG 体积约\n"
                 "减半 ⇒ **A 机编码时间和带宽都省一半** ✓（这正好是 60fps 的开销来源 ✓）。\n"
                 "黄点认不出来（`mmap_miss` 一直涨）才需要往上调。"),
    ],
}


def python_exe():
    """跑 python 服务的解释器 —— 用启动本界面的那个（venv 里也一样对）。"""
    return sys.executable or "python"


def build_cmd(key, cfg):
    """按配置拼出要执行的命令行（list，直接交给 subprocess）。

    **全部走绝对路径**：`--cert remote_kbd/certs/cert.pem` 这种相对路径只在
    仓库根下才成立，而工作目录一旦不是仓库根就会报「找不到证书」，
    那种错看起来像是证书坏了，很费时间。证书/私钥这里都会转成绝对路径。
    """
    p = cfg or {}
    root = Path(__file__).resolve().parent.parent

    def num(v, default, cast=float):
        try:
            return cast(v)
        except (TypeError, ValueError):
            return default

    def abs_path(v):
        """相对路径按仓库根解析成绝对路径（已经是绝对的就不动）。"""
        s = str(v or "").strip()
        if not s:
            return s
        path = Path(s)
        return str(path if path.is_absolute() else (root / path))

    if key == "clock":
        return [python_exe(), "-m", "tools.clock_server",
                "--host", str(p.get("host") or "0.0.0.0"),
                "--port", str(int(num(p.get("port"), 5001, int)))]

    if key == "probe":
        return [python_exe(), "-m", "tools.probe_gen",
                "--out-scale", "%g" % num(p.get("out_scale"), 0.8004)]

    if key == "kbd":
        return [python_exe(), "-m", "remote_kbd.relay",
                "--serial", str(p.get("serial") or "COM3"),
                "--port", str(int(num(p.get("port"), 9000, int))),
                "--cert", abs_path(p.get("cert")),
                "--key", abs_path(p.get("key"))]

    if key == "push":
        return _push_cmd(p, num)

    if key == "mmap":
        return _mmap_cmd(p, num)

    raise KeyError("未知服务 %r" % (key,))


def _mmap_cmd(p, num):
    """小地图推流的命令行。

    ⭐ **区域按"当前是哪张图"传**（用户 2026-10-02 ✓）：本机记着当前图、而且那张图
    **框过** ⇒ 传 `--map-id <id>` —— 区域与 zoom 都由 `config/minimap_regions/<id>.json`
    说了算，而**那正是界面上「小地图区域」显示的那一份** ✓ ⇒ 「界面显示 A、命令跑的
    却是 B」这条规矩照样成立 ✓（见那个文件顶部说明）。
    没有当前图（或那张图还没框过）⇒ **一个区域参数都不带**：服务照起、先监听，等 B 机
    那句 `MAP <id>`（用户 2026-10-02 的流程 ✓ 见 `tools/minimap_push` 顶部说明）——
    ⚠ 别拿老的单值兜底硬凑一个区域 ✗：那会推一块**这张图无关**的画面出去，
      B 机那边看着就像寻路坏了（区域是**屏幕坐标**，换图/换布局就全错 ✓）。
    ⚠ 有 `--map-id` 时**不传 `--zoom`**：那张图那份里存着它自己的 zoom
      （标定是按某个 zoom 标的 ⇒ 两者必须是同一份 ✓ 见 `core.mapdata.load_calib`）。
    """
    cmd = [python_exe(), "-m", "tools.minimap_push",
           "--bind", str(p.get("bind") or "0.0.0.0"),
           "--port", str(int(num(p.get("port"), 5003, int))),
           # ⭐ 默认与卡片/DEFAULTS 一致（2026-10-03：30→60、100→80 ✓ 见上面那个 spec ✓）
           "--fps", str(int(num(p.get("fps"), 60, int))),
           "--quality", str(int(num(p.get("quality"), 80, int)))]
    from tools import mmap_regions                 # 懒导入（同 app.py 的框选那份 ✓）
    cur = mmap_regions.current() or {}
    mid = str(cur.get("map_id") or "")
    if mid and mmap_regions.load(mid) is not None:
        cmd += ["--map-id", mid]
    return cmd


def mmap_region_now():
    """**当前这张图**的推流区域 → `(map_id, box, zoom)` | `(map_id|"", None, None)`。

    部署台那两行显示（「当前地图」/「小地图区域」）、自检、框选落盘**共用这一处**
    （别各自去读文件 ✗ 约定：一条口径一处实现 ✓）。`box` 是 `(x, y, w, h)`。
    """
    from tools import mmap_regions
    cur = mmap_regions.current() or {}
    mid = str(cur.get("map_id") or "")
    if not mid:
        return "", None, None
    got = mmap_regions.load(mid)
    if not got:
        return mid, None, None
    return mid, (got["x"], got["y"], got["w"], got["h"]), int(got["zoom"])


def _push_cmd(p, num):
    """推流命令行。默认对齐 tools/a_push.ps1 实测的那套（端到端 ~131 ms）。"""
    ff = str(p.get("ffmpeg") or "ffmpeg").strip() or "ffmpeg"
    w = int(num(p.get("width"), 1366, int))
    h = int(num(p.get("height"), 768, int))
    fps = int(num(p.get("fps"), 50, int))
    gop = int(num(p.get("gop"), 60, int))
    br = str(p.get("bitrate") or "6M")
    host = str(p.get("host") or "192.168.1.2").strip()
    port = int(num(p.get("port"), 5000, int))
    pkt = int(num(p.get("pkt_size"), 1316, int))
    enc = str(p.get("encoder") or "h264_nvenc")
    low = bool(p.get("low_latency"))

    cmd = [ff, "-hide_banner", "-loglevel", "error", "-nostats"]

    if str(p.get("capture")) == "gdigrab":
        # GDI 抓屏：分辨率/帧率直接写在这，后面不用再 scale
        cmd += ["-f", "gdigrab", "-framerate", str(fps),
                "-video_size", "%dx%d" % (w, h), "-i", "desktop"]
    elif str(p.get("scale_mode") or "cpu") == "cuda":
        # DXGI 硬采 + **GPU 缩放**：整屏那一帧留在显存里（hwmap 到 cuda），
        # 只在显存里缩到输出尺寸，再把小图拷回内存。
        # 省掉的是"整屏 BGRA ≈15MB/帧 拷回内存" —— 实测 CPU 那条链的天花板就是它
        # （2026-09-25：18 条候选 speed 全贴 1.00、实际 fps 只有 ~73）。
        cmd += ["-init_hw_device", "d3d11va=dx", "-init_hw_device", "cuda=cu",
                "-filter_hw_device", "dx", "-filter_complex",
                "ddagrab=framerate=%d,hwmap=derive_device=cuda,"
                "scale_cuda=%d:%d,hwdownload,format=nv12" % (fps, w, h)]
    else:
        # DXGI 硬采（默认）：抓完 hwdownload 回内存，再 scale + 转像素格式
        cmd += ["-init_hw_device", "d3d11va=dx", "-filter_hw_device", "dx",
                "-filter_complex",
                "ddagrab=framerate=%d,hwdownload,format=bgra,scale=%d:%d,format=nv12"
                % (fps, w, h)]

    if p.get("passthrough"):
        cmd += ["-fps_mode", "passthrough"]

    if enc == "libx264":
        cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency"]
    else:
        cmd += ["-c:v", "h264_nvenc", "-preset", "p1"]
        if low:
            cmd += ["-tune", "ull", "-zerolatency", "1", "-rc", "cbr"]

    cmd += ["-b:v", br, "-bufsize", "1M", "-g", str(gop), "-bf", "0"]
    if enc != "h264_nvenc":
        cmd += ["-pix_fmt", "yuv420p"]

    if low:
        cmd += ["-flush_packets", "1", "-muxdelay", "0", "-muxpreload", "0"]

    cmd += ["-f", "mpegts", "udp://%s:%d?pkt_size=%d" % (host, port, pkt)]
    return cmd


def format_cmd(cmd):
    """命令行 → 一行可读文本（带引号，方便照抄到 PowerShell 里手动跑）。"""
    out = []
    for a in cmd:
        a = str(a)
        out.append('"%s"' % a if (" " in a or "," in a) else a)
    return " ".join(out)


def release_all(serial_port):
    """停掉键盘中继之后，自己给 Pro Micro 补一条 RELEASEALL。返回 (ok, 说明)。

    **为什么必须补这一下**：中继是被硬停的（Windows 上没法优雅通知它，见
    deploy/runner.py 的说明），它来不及松开按住的键 —— 那一刻正好有键按着的话，
    Pro Micro 会一直按着，游戏里就是「角色自己走 / 一直攻击」，只能重启中继
    （relay 启动时会自己发一条 RELEASEALL）才恢复。

    打开串口会复位 32U4（DTR 拉低），复位本身就会清掉按键状态 ——
    所以这条路无论如何都能把键松开。而 relay 每次启动本来也要经历同样的复位，
    不是这里新增的副作用。
    """
    port = str(serial_port or "").strip()
    if not port:
        return False, "没填串口号，跳过松键"
    try:
        import serial
    except ImportError:
        return False, "pyserial 没装，跳过松键"
    try:
        ser = serial.Serial(port, 115200, timeout=0.3, write_timeout=1.0)
    except Exception as e:
        return False, "串口 %s 打不开（%s）—— 如果键真的卡住了，重启中继也能松开" % (
            port, e)
    try:
        # 和中继同样的处理：先禁 DTR/RTS，别再多折腾一次板子
        try:
            ser.dtr = False
            ser.rts = False
        except Exception:
            pass
        ser.reset_input_buffer()
        ser.write(b"RELEASEALL\n")
        time.sleep(0.15)          # 给固件一点时间处理完
        return True, "已给 Pro Micro 补发 RELEASEALL（松开卡住的键）"
    except Exception as e:
        return False, "写 RELEASEALL 失败：%s" % e
    finally:
        try:
            ser.close()
        except Exception:
            pass


#: 已知错误特征 → 一句人话。特征按**小写**子串匹配（一条命中就够）。
#:
#: 为什么要在我们这边翻：ffmpeg 只会给「Error number -10051」这种，要去查
#: Windows 错误码表才知道那是 WSAENETUNREACH（网络不可达）—— 而这类错最长见的
#: 原因就那几条（网段不对 / 没拿到 IP / B 机改了 IP），直接写出来省一轮排查。
_ERROR_HINTS = (
    (("-10051", "network is unreachable", "no route to host", "-10065"),
     "本机到 B 机没有路由（-10051 = Windows 网络不可达）—— 这是本机的路由错误，"
     "不是 B 机拒绝你。看它是「一启动就报」还是「挂了一阵才报」：\n"
     "  · 一启动就报 = 网段/路由没配通：① A 机自己的 IP 是否和 B 机同网段"
     "（ipconfig /all；169.254.x.x = 没拿到 IP）；② ping B 机；③ B 机现在的 IP "
     "还是配置里那个吗；④ 不同网段就改 config/link.yaml 的 b_host，再点「按 link.yaml 填」。\n"
     "  · 挂了一阵才报 = 配置本来是对的、运行中途路由丢了：最常见是系统睡眠/"
     "现代待机、网卡省电、USB 或 Wi-Fi 网卡掉线（挂机时没人碰键鼠最容易触发）。"
     "查 powercfg /a，按 docs/A_SETUP.md 末尾那节把睡眠/硬盘/显示器/网卡省电全关掉。"),
    (("-10013", "permission denied"),
     "被拒绝（-10013）：多半是本机防火墙 / 安全软件拦了出站 UDP。"),
    (("cannot load nvcuda", "no capable devices",
      "unknown encoder 'h264_nvenc'", "error while opening encoder"),
     "显卡编码起不来：把推流的「编码器」改成 libx264（720p 以内够用），"
     "或确认 N 卡驱动 / 是不是被别的进程占满了编码器会话。"),
    (("only one usage of each socket address", "address already in use", "10048"),
     "端口被占用（10048 = WSAEADDRINUSE）：A 机上已经有一个同类服务在跑，"
     "或者别的程序占着这个端口。看卡片状态和任务管理器，别重复启动。"),
)


def explain(text):
    """把已知的错误特征翻成一句人话；没有对应特征返回空串。

    纯函数、不读配置 —— 便于直接拿来测（见 tools/ 里的自检思路）。
    """
    low = str(text or "").lower()
    for marks, hint in _ERROR_HINTS:
        if any(m in low for m in marks):
            return hint
    return ""


def missing_hint(key, cfg):
    """启动前能一眼看出来的问题，返回提示文字；没问题返回空串。

    只查「不用等日志就能判定」的：文件在不在、参数空不空。
    **不查端口占用** —— 那要等子进程报错，硬查反而会误伤（我们自己的服务
    正在跑时当然占用着）。
    """
    p = cfg or {}
    root = Path(__file__).resolve().parent.parent

    if key == "kbd":
        if not str(p.get("serial") or "").strip():
            return "还没填 Pro Micro 的串口号。"
        for lbl, rel in (("证书", p.get("cert")), ("私钥", p.get("key"))):
            if not str(rel or "").strip():
                return "还没指定 TLS %s。" % lbl
            path = Path(str(rel))
            path = path if path.is_absolute() else (root / path)
            if not path.exists():
                return ("TLS %s 不存在：\n%s\n\n"
                        "用 python -m remote_kbd.gen_cert 生成一份，"
                        "或在卡片里指到已有的 .pem。" % (lbl, path))

    if key == "push":
        ff = str(p.get("ffmpeg") or "").strip()
        if not ff:
            return "还没指定 ffmpeg.exe 的位置。"
        if not (Path(ff).is_file() if ("\\" in ff or "/" in ff) else shutil.which(ff)):
            return ("找不到 ffmpeg：%s\n\n"
                    "装一个（winget install --id Gyan.FFmpeg -e），\n"
                    "或者把「ffmpeg」一栏指到 ffmpeg.exe 的实际路径。" % ff)
        if not str(p.get("host") or "").strip():
            return "还没填 B 机 IP。"
        try:
            w, h = int(p.get("width")), int(p.get("height"))
        except (TypeError, ValueError):
            return "分辨率填得不对，应该是「宽x高」，例如 1366x768。"
        if w < 160 or h < 120:
            return "分辨率太小了（%dx%d）。" % (w, h)

    if key == "mmap":
        # ⭐ **不再拦"还没框过"的启动**（用户 2026-10-02 ✓）：现在"当前是哪张图"只由 B 机推，
        #   而 B 机那句 `MAP <id>` 必须走这条服务的连接 ⇒ **必须先让它起来**（先监听、
        #   再被告诉是哪张图，拿到 id 之后才谈得上框 ✓ 见 `tools/minimap_push` 顶部说明）。
        #   ⇒ 这一项现在交给**环境自检**提示（warn：还没图/这张图还没框过 ✓ 不红 ✗）。
        #   ⚠ 区域本身的合法性（太小 / 负数 / 非数字）由 `mmap_regions._clean` 那一处判 ✓
        #     （那份配置进不了库 ⇒ `load` 给 None ⇒ 与"还没框过"同一档 ✓ 一处判据 ✓）。
        return ""

    if key == "probe":
        try:
            s = float(p.get("out_scale"))
        except (TypeError, ValueError):
            return "输出缩放填得不对，应该是个小数，例如 0.8004。"
        if not (0.05 <= s <= 4.0):
            return "输出缩放 %.3f 超出范围（0.05 ~ 4.0）。" % s

    return ""
