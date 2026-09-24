"""环境自检：启动之前最想知道的那几件事。

**为什么值得单独做一块**
    A 机上「按了没反应」的原因，九成是这五件事之一：ffmpeg 没装 / 没有 NVENC /
    串口填错 / B 机不可达 / 参数和 link.yaml 不一致。这些都能在**启动之前**查出来，
    比等日志里一串英文报错再回头猜省事得多。

**在哪跑**：界面里会另起一个线程调 run() —— ping 和 ffmpeg 都要几百毫秒到几秒，
放界面线程里会卡住窗口。返回值是纯数据（list of dict），与 Qt 无关。
"""

import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PING_RE = re.compile(r"[=<]\s*(\d+)\s*ms", re.I)


def ok(title, detail=""):
    return {"level": "ok", "title": title, "detail": detail}


def warn(title, detail=""):
    return {"level": "warn", "title": title, "detail": detail}


def bad(title, detail=""):
    return {"level": "bad", "title": title, "detail": detail}


# ---------------------------------------------------------------- 单项检查

def _run(cmd, timeout=8.0):
    """跑一条命令拿输出。失败返回 (None, 错误文字)。"""
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, encoding="utf-8", errors="replace",
                           timeout=timeout,
                           creationflags=(0x08000000 if sys.platform == "win32" else 0))
    except FileNotFoundError:
        return None, "找不到可执行文件：%s" % cmd[0]
    except subprocess.TimeoutExpired:
        return None, "超时（%.0fs）" % timeout
    except Exception as e:
        return None, "%s: %s" % (type(e).__name__, e)
    return r.stdout or "", ""


def resolve_exe(name):
    """配置里可能是绝对路径，也可能只是个命令名 —— 两种都认。"""
    s = str(name or "").strip()
    if not s:
        return ""
    if Path(s).is_file():
        return s
    return shutil.which(s) or ""


def check_ffmpeg(cfg):
    """ffmpeg 在不在 + 有没有 NVENC 编码器。"""
    out = []
    exe = resolve_exe(cfg.get("ffmpeg"))
    if not exe:
        out.append(bad("ffmpeg 未找到",
                       "装一个：winget install --id Gyan.FFmpeg -e\n"
                       "已经装了就把「ffmpeg」一栏指到 ffmpeg.exe"))
        return out

    ver, err = _run([exe, "-hide_banner", "-version"], timeout=8)
    if ver is None:
        out.append(bad("ffmpeg 跑不起来", err))
        return out
    first = (ver.splitlines() or [""])[0].strip()
    out.append(ok("ffmpeg 可用", "%s\n%s" % (first, exe)))

    enc, err = _run([exe, "-hide_banner", "-encoders"], timeout=10)
    if enc is None:
        out.append(warn("查不到编码器列表", err))
    elif "h264_nvenc" in enc:
        out.append(ok("h264_nvenc 可用", "有 N 卡硬件编码，延迟和 CPU 占用都最优"))
    else:
        out.append(warn("没有 h264_nvenc",
                        "这台机器的 ffmpeg 编译时没带 NVENC，或没有 N 卡。\n"
                        "推流的「编码器」请改成 libx264（720p 以内还能跑）"))
    return out


def check_serial(cfg):
    """串口列表 + 配置里那个在不在。"""
    try:
        from serial.tools import list_ports
    except ImportError:
        return [warn("pyserial 没装", "pip install pyserial（键盘中继要用）")]

    ports = [p.device for p in list_ports.comports()]
    want = str(cfg.get("serial") or "").strip()
    if not ports:
        return [bad("没找到任何串口", "Pro Micro 插好了吗？装了驱动吗？")]
    if not want:
        return [warn("串口列表 %s" % ", ".join(ports), "还没选串口号")]
    if want in ports:
        return [ok("串口 %s 在线" % want, "当前设备：%s" % ", ".join(ports))]
    return [warn("串口 %s 不在列表里" % want,
                 "当前设备：%s\n插拔后要重新自检；串口号会变（COM4 → COM7 很常见）"
                 % ", ".join(ports))]


def check_host(cfg, timeout_ms=800):
    """B 机通不通（ping 一次）。"""
    host = str(cfg.get("host") or "").strip()
    if not host:
        return [warn("还没填 B 机 IP")]
    if sys.platform != "win32":
        return [warn("跳过 ping 检查（只在 Windows 上查）")]

    out, err = _run(["ping", "-n", "1", "-w", str(timeout_ms), host],
                    timeout=timeout_ms / 1000.0 + 3)
    if out is None:
        return [warn("ping 失败：%s" % err)]
    if "TTL=" in out.upper():
        m = PING_RE.search(out)
        ms = ("%s ms" % m.group(1)) if m else "延迟未知"
        return [ok("B 机 %s 可达" % host, "RTT %s" % ms)]
    return [warn("B 机 %s 不通" % host,
                 "推流会发出去但 B 机收不到。检查：B 机开机了？同网段？\n"
                 "A 机自己的 IP 和它同网段吗（ipconfig /all；"
                 "169.254.x.x = 没拿到 IP）？\n"
                 "B 机防火墙放行了 UDP %s？" % cfg.get("port"))]


def check_certs(cfg):
    """TLS 证书/私钥在不在（键盘中继要用）。"""
    out = []
    for label, key in (("证书", "cert"), ("私钥", "key")):
        rel = str(cfg.get(key) or "").strip()
        if not rel:
            out.append(warn("没指定 TLS %s" % label))
            continue
        path = Path(rel)
        path = path if path.is_absolute() else (ROOT / path)
        if path.exists():
            out.append(ok("TLS %s 在" % label, str(path)))
        else:
            out.append(bad("TLS %s 不存在" % label,
                           "%s\n用 python -m remote_kbd.gen_cert 生成一份" % path))
    return out


def check_link(cfg, expect):
    """部署台的参数和 config/link.yaml 对不对得上。

    **这是最容易白折腾半天的一项**：B 机是照 link.yaml 收流/对时的，
    两边不一致时现象是「流发出去了一切正常，B 机就是没画面」。
    """
    if not expect:
        return [warn("读不到 config/link.yaml", "跳过一致性检查")]

    pairs = {
        "B 机 IP": cfg.get("push", {}).get("host"),
        "推流端口": cfg.get("push", {}).get("port"),
        "流分辨率": (cfg.get("push", {}).get("width"), cfg.get("push", {}).get("height")),
        "推流帧率": cfg.get("push", {}).get("fps"),
        "对时端口": cfg.get("clock", {}).get("port"),
        "键盘端口": cfg.get("kbd", {}).get("port"),
        "串口号": cfg.get("kbd", {}).get("serial"),
    }
    out = []
    for label, (want, where) in expect.items():
        got = pairs.get(label)
        if got is None:
            continue
        same = (str(got) == str(want)) if not isinstance(want, tuple) else (
            tuple(str(x) for x in got) == tuple(str(x) for x in want))
        if same:
            continue
        out.append(warn("和 link.yaml 不一致：%s" % label,
                        "部署台 %s = %s\nlink.yaml = %s\n"
                        "B 机按 link.yaml 来，两边不一致会「发了但收不到」。"
                        "点下面的「按 link.yaml 填」即可对齐。"
                        % (where, got, want)))
    if not out:
        out.append(ok("与 link.yaml 一致", "推流目标、端口、分辨率、串口都对得上"))
    return out


# ---------------------------------------------------------------- 汇总

def run(cfg, expect=None):
    """跑全部检查，返回 [{level, title, detail}]。

    任何单项炸了都不能让整块自检消失 —— 每一项都单独 try，出错就变成一条 warn。
    """
    cfg = cfg or {}
    items = []

    def guard(fn, *a):
        try:
            items.extend(fn(*a) or [])
        except Exception as e:
            items.append(warn("%s 检查出错" % getattr(fn, "__name__", "?"),
                              "%s: %s" % (type(e).__name__, e)))

    guard(check_ffmpeg, cfg.get("push", {}))
    guard(check_serial, cfg.get("kbd", {}))
    guard(check_host, cfg.get("push", {}))
    guard(check_certs, cfg.get("kbd", {}))
    guard(check_link, cfg, expect)
    return items


def local_ips():
    """本机可能被 B 机看到的 IPv4（给界面提示用，取不到就空）。"""
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))       # 不发包，只为拿到出口网卡地址
        ips.append(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    return ips
