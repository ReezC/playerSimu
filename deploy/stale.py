"""启动前清掉「残留推流进程」——同一个 UDP 端口上已经有一个 ffmpeg 在推。

**为什么要有它**（2026-10-17 实测，用户现场 ✓）：

    A 机上同时有两条流推到 B 机 5000：
        192.168.1.9:55947   8739 包  18.1 Mbps   ← 旧进程（孤儿）
        192.168.1.9:51902   5541 包  11.4 Mbps   ← 新进程（12M ✓ 刚启动的）
    两条 TS 交错在**同一个 PID** 上 ⇒ 解码器一会儿拿 A 的帧、一会儿拿 B 的帧
    ⇒ 画面花屏 / 抖动 / 扭曲，而"停止"只 `kill_tree` **自己记着的那个 PID**
    （`deploy/runner.py:122`），孤儿进程杀不到 ⇒ 再点「启动」就变两条。

**边界**：只认「ffmpeg + `mpegts` + 同一个 `udp://` 端口」这一种 ✓；
         录制、转码、别的端口的推流一律不碰 ✗（宁可漏杀，不许误杀 ✓）。

**为什么两个函数分开**：`is_push_to` 是**纯函数**（只吃命令行字符串），
   自检里直接喂样本就能验；`sweep` 才碰 psutil / 真的杀进程 ✓。
"""

import os

#: 只处理这些进程名（大小写无关；Windows 上是 `ffmpeg.exe`）
FFMPEG_NAMES = ("ffmpeg", "ffmpeg.exe")


def udp_target(cmd):
    """从命令行（list 或 str）里取 `udp://host:port?...` 的端口 ⇒ `int` 或 `None`。"""
    if isinstance(cmd, (list, tuple)):
        parts = [str(x) for x in cmd]
    else:
        parts = str(cmd).split()
    for tok in parts:
        low = tok.lower()
        i = low.find("udp://")
        if i < 0:
            continue
        rest = tok[i + len("udp://"):]
        # 去掉 query 和路径，只看 host:port
        rest = rest.split("?")[0].split("/")[0]
        if ":" not in rest:
            continue
        try:
            return int(rest.rsplit(":", 1)[1])
        except (TypeError, ValueError):
            continue
    return None


def is_push_to(cmdline, port):
    """这条命令行是不是「用 mpegts 往 `udp://…:port` 推流」的 ffmpeg。

    :param cmdline: 一条完整命令行（`str` ✓ psutil 给的 `cmdline` 列表也行 ✓）
    :param port: 端口（`None`/`0` ⇒ 恒 `False` ✓ 拿不到端口就不动手 ✗）
    """
    if not cmdline or not port:
        return False
    text = " ".join(cmdline) if isinstance(cmdline, (list, tuple)) else str(cmdline)
    if "mpegts" not in text.lower():
        return False
    return udp_target(text) == int(port)


def stale_pushes(port, skip_pids=()):
    """找出现在**还活着**的、往这个端口推的老进程 ⇒ `[(pid, cmdline_str), …]`。

    永远不抛：拿不到权限 / psutil 没装 ⇒ 返回空表（启动流程不该被它拦住 ✓）。
    """
    if not port:
        return []
    skip = set(int(x) for x in skip_pids if x) | {os.getpid()}
    out = []
    try:
        import psutil
    except Exception:                     # noqa: BLE001
        return out
    try:
        it = psutil.process_iter(["pid", "name", "cmdline"])
    except Exception:                     # noqa: BLE001
        return out
    for pr in it:
        try:
            info = pr.info
            name = str(info.get("name") or "").lower()
            if name not in FFMPEG_NAMES and not name.startswith("ffmpeg"):
                continue
            pid = int(info.get("pid") or 0)
            if not pid or pid in skip:
                continue
            cl = info.get("cmdline") or []
            if not is_push_to(cl, port):
                continue
            out.append((pid, " ".join(cl)))
        except Exception:                 # noqa: BLE001 —— AccessDenied / 进程已退出 ✓
            continue
    return out


def sweep(port, skip_pids=(), log=None):
    """把残留推流进程**杀掉** ⇒ 返回 `[(pid, cmdline), …]`（杀掉的那些）。

    :param log: `log("…")` 可选回调（界面把这句话直接显示出来 ✓ 别静默 ✓）
    """
    killed = []
    for pid, cmd in stale_pushes(port, skip_pids=skip_pids):
        ok = False
        try:
            from core import procguard
            ok = bool(procguard.kill_tree(pid))
        except Exception:                 # noqa: BLE001
            ok = False
        if not ok:
            try:
                import psutil
                psutil.Process(pid).kill()
                ok = True
            except Exception:             # noqa: BLE001
                ok = False
        if ok:
            killed.append((pid, cmd))
            if log:
                log("清掉残留推流进程 PID=%d（端口 %d 上还有一条老流，"
                    "两条交错会花屏 ✓）" % (pid, int(port)))
    return killed
