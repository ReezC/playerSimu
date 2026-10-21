"""**实例身份** —— 同一台机器上开多个工作台（用户 2026-10-17 ✓ 原话：
「**我们一定要能支持多实例工作**」）。

怎么用（三选一，优先级从高到低）：

```bat
python -m gui.app --instance b        :: 命令行
set PSIMU_INSTANCE=b  && python -m gui.app   :: 环境变量
python -m gui.app                     :: 都不给 ⇒ **空名字 = 老行为** ✓
```

⚠ 空名字时**所有路径与今天一字不差** ✓（`config/live.yaml`、`behavior.log`、
`data/recordings/`、`config/session.json`）—— 只开一个工作台的人**感觉不到这次改动** ✓。

每个名字各自一份（互相不覆盖 ✗）：

| 用途 | 默认实例（空名字） | 具名实例 `<n>` |
|---|---|---|
| 实时预览参数 | `config/live.yaml` | `config/live_<n>.yaml` |
| 行为日志 | `behavior.log` | `behavior_<n>.log` |
| 崩溃日志 | `crash.log` | `crash_<n>.log` |
| 录像 | `data/recordings/` | `data/recordings/<n>/` |
| "上次打开的项目" | `config/session.json` | `config/session_<n>.json` |

⚠⚠ **`config/link.yaml` 故意不分实例** ✗ —— 里面的 A/B 机 IP、A 机屏幕分辨率/DPI
是**机器事实** ✓，两个实例共用同一份才对 ✓（一个实例改了 IP 另一个也该跟着变 ✓）。

⚠⚠ 名字**只允许** `[A-Za-z0-9_-]`（会出现在文件名里 ✓ 别让中文/空格/斜杠进来 ✗）。
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"

#: 环境变量名（比命令行短 ✓ 适合做 `.bat` 快捷方式 ✓）
ENV = "PSIMU_INSTANCE"

_resolved: list = []          # 只解析一次 ⇒ [(名字, slug)]


def _parse_argv(argv):
    """从命令行里挑 `--instance X` / `--instance=X`（挑不出来 ⇒ `None` ✓）。"""
    for i, a in enumerate(list(argv or [])):
        if a == "--instance" and i + 1 < len(argv):
            return argv[i + 1]
        if isinstance(a, str) and a.startswith("--instance="):
            return a.split("=", 1)[1]
    return None


def name():
    """本进程的实例名（**可能为空** ✓ 空 = 老行为）。"""
    if not _resolved:
        raw = _parse_argv(sys.argv)
        if not raw:
            raw = os.environ.get(ENV) or ""
        _resolved.append((str(raw or "").strip(), _slug(raw)))
    return _resolved[0][0]


def slug():
    """实例名 → 能安全用在文件名里的那一段（空名字 ⇒ `""` ✓）。"""
    name()
    return _resolved[0][1]


def _slug(raw):
    s = re.sub(r"[^0-9A-Za-z_-]+", "-", str(raw or "").strip()).strip("-")
    return s[:32]


def suffix(sep="_"):
    """拼在文件名后面的那一段（空实例 ⇒ `""` ✓）。"""
    s = slug()
    return ("%s%s" % (sep, s)) if s else ""


def label():
    """界面标题/日志里用的短标签（空实例 ⇒ `"默认"`）。"""
    return slug() or "默认"


# ---------------- 每实例的路径 ----------------

def live_path():
    """实时预览参数那份 yaml（具名实例**第一次用时从默认那份复制** ✓）。

    ⚠ 为什么复制而不是"从空开始"（✗）：用户切到具名实例时，期望是"今天能用的设置
      照样能用"✓；从空开始的话所有开关都回默认值 ⇒ 一上来就像坏了 ✓。
    """
    s = slug()
    if not s:
        return CONFIG_DIR / "live.yaml"
    p = CONFIG_DIR / ("live_%s.yaml" % s)
    if not p.exists():
        try:
            d = CONFIG_DIR / "live.yaml"
            if d.exists():
                shutil.copyfile(d, p)
        except Exception:                    # noqa: BLE001 —— 复制不了就让上层按"没有"处理 ✓
            pass
    return p


def behavior_path():
    s = slug()
    return ROOT / ("behavior_%s.log" % s) if s else (ROOT / "behavior.log")


def crash_path():
    s = slug()
    return ROOT / ("crash_%s.log" % s) if s else (ROOT / "crash.log")


def session_path():
    s = slug()
    return CONFIG_DIR / ("session_%s.json" % s) if s else (CONFIG_DIR / "session.json")


# ---------------- 端口占用：同机多实例最硬的那条约束 ----------------
#
# ⚠⚠ 为什么需要它（用户 2026-10-17 ✓「多实例」）：UDP 推流是**单播到一个端口** ✓，
#   本机**只有一个 socket 能收到**它 ✗ —— 第二个实例 bind 同一端口时：
#     · 有的平台直接失败 ✓；
#     · 但 ffmpeg/PyAV 常带 `SO_REUSEADDR` ⇒ **两个都能 bind 成功** ✗
#       ⇒ 包只落到其中一个 ⇒ 另一个"能打开、但帧率莫名其妙一半/全是花屏"✓
#       —— 这比直接失败更难查 ✗（现场"第二个工作台收不到帧/抖"就是这么来的 ✓）。
#   ⇒ 不靠 bind 探测 ✗，改成**文件锁 + PID**：谁能创建 `data/locks/udp_<端口>.lock`
#     谁收这一路 ✓；占着的人还活着就如实报「被谁占了」✓ 死了就自动接管 ✓。

LOCK_DIR = DATA_DIR / "locks"


def _pid_alive(pid):
    """`pid` 这个进程还活着吗（Windows 用 OpenProcess/WaitForSingleObject ✓）。"""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            k = ctypes.windll.kernel32
            h = k.OpenProcess(0x00100000, False, pid)      # SYNCHRONIZE
            if not h:
                # ⚠ 打不开**不等于**不存在 ✗：`ERROR_ACCESS_DENIED(5)` = 进程在、只是没权限
                #   （系统进程 / 别的用户跑的 ✓）⇒ 必须当"活着" ✓
                #   只有 `ERROR_INVALID_PARAMETER(87)` 才是"真没有这个 PID" ✓
                return k.GetLastError() == 5
            try:
                return k.WaitForSingleObject(h, 0) == 0x102  # WAIT_TIMEOUT ⇒ 还在跑
            finally:
                k.CloseHandle(h)
        except Exception:                  # noqa: BLE001 —— 查不出来就当"活着"（宁可不接管 ✓）
            return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def udp_port_of(url):
    """从 `udp://0.0.0.0:5000?...` 里取端口（不是 udp / 解析不出来 ⇒ `0`）。"""
    s = str(url or "")
    if not s.lower().startswith("udp://"):
        return 0
    s = s[6:]
    s = s.split("?")[0].rstrip("/")
    if ":" not in s:
        return 0
    try:
        return int(s.rsplit(":", 1)[1])
    except ValueError:
        return 0


def claim_port(port, host="0.0.0.0"):
    """占住「本机这个 UDP 端口归我收」⇒ `(True, "")` / `(False, "谁占着")`。

    · **同一个进程**重复占（预览重启 / `LiveThread` 重建）⇒ 放行 ✓（不能把自己顶掉 ✗）；
    · 占着的人**已经死了** ⇒ 自动接管 ✓（不然崩过一次就永远说"被占"✗）。
    """
    if not port:
        return True, ""
    try:
        LOCK_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:                      # noqa: BLE001
        return True, ""                    # 建不出锁目录 ⇒ 不拦（别把功能弄死 ✓）
    p = LOCK_DIR / ("udp_%d.lock" % int(port))
    for _ in range(2):
        try:
            with open(p, "x", encoding="utf-8") as f:
                f.write("pid=%d\ninst=%s\nport=%d\nhost=%s\n"
                        % (os.getpid(), label(), int(port), host))
            return True, ""
        except FileExistsError:
            info = {}
            try:
                for ln in p.read_text(encoding="utf-8").splitlines():
                    if "=" in ln:
                        k, v = ln.split("=", 1)
                        info[k.strip()] = v.strip()
            except Exception:              # noqa: BLE001 —— 读不出来当"没有主人"✓
                pass
            pid = info.get("pid", "")
            if str(pid) == str(os.getpid()):
                return True, ""            # 自己占的 ⇒ 放行 ✓
            if _pid_alive(pid):
                return False, "实例「%s」PID %s" % (info.get("inst") or "?", pid)
            try:
                p.unlink()                 # 主人已经没了 ⇒ 清掉重占 ✓
            except Exception:              # noqa: BLE001
                return True, ""
        except OSError:
            return True, ""                # 权限之类 ⇒ 不拦 ✓（让真正 bind 的人去报 ✓）
    return True, ""


def _read_lock(path):
    """读一份占位文件 → `{}`（读不出来当"没有主人" ✓ 同 `claim_port` 的口径）。"""
    info = {}
    try:
        for ln in Path(path).read_text(encoding="utf-8").splitlines():
            if "=" in ln:
                k, v = ln.split("=", 1)
                info[k.strip()] = v.strip()
    except Exception:                      # noqa: BLE001
        pass
    return info


def release_port(port):
    """放掉「这个 UDP 端口归我收」—— **只删自己那份** ✓（`True` = 真删掉了）。

    ⚠ 为什么要显式放（2026-10-21 用户：「**窗口关掉但是实例清不掉**」✓）：
      以前**没有任何释放路径** ⇒ 全指望"下次启动时发现主人 PID 已经死了 ⇒ 接管" ✓。
      可进程要是**没死成**（当时的现场：窗口关了、`pythonw` 还挂着 ✅ 见
      `gui/app.py::_hard_exit` 里的栈），锁就一直被一个**活着**的 PID 占着
      ⇒ 下次启动只会说"UDP 端口已被另一个工作台实例占用" ✗ —— 也就是用户说的"实例清不掉" ✓。
    ⇒ 退出前由 `gui/app.py` 调 `release_my_ports()` 统一放 ✓（`os._exit` 不走 atexit ✗）。
    """
    try:
        port = int(port or 0)
    except (TypeError, ValueError):
        return False
    if port <= 0:
        return False
    p = LOCK_DIR / ("udp_%d.lock" % port)
    try:
        if not p.exists():
            return False
        if str(_read_lock(p).get("pid", "")) != str(os.getpid()):
            return False                   # 别人的 ⇒ 一个字都不许动 ✗
        p.unlink()
        return True
    except Exception:                      # noqa: BLE001
        return False


def release_my_ports():
    """把**本进程**占着的 UDP 端口全放掉 ⇒ 返回放掉的端口列表 ✓（`os._exit` 之前调 ✓）。

    扫整份锁目录、只删 `pid` 是自己的那些 ✓ —— 别的实例那份**绝不动** ✗（多实例 ✓）。
    """
    out = []
    try:
        if not LOCK_DIR.exists():
            return out
        for p in list(LOCK_DIR.glob("udp_*.lock")):
            if str(_read_lock(p).get("pid", "")) != str(os.getpid()):
                continue
            try:
                port = int(p.stem.split("_", 1)[1])
            except (IndexError, ValueError):
                port = 0
            try:
                p.unlink()
                if port:
                    out.append(port)
            except Exception:              # noqa: BLE001
                pass
    except Exception:                      # noqa: BLE001
        pass
    return out


def record_dir(base=None):
    """录像目录（具名实例落在 `data/recordings/<n>/` ✓ 两个实例的录像不混 ✓）。"""
    d = Path(base) if base else (DATA_DIR / "recordings")
    s = slug()
    if s:
        d = d / s
    d.mkdir(parents=True, exist_ok=True)
    return d
