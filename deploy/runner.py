"""子进程托管：启动 / 停止 / 日志回传。

**为什么是硬停止（terminate → kill），而不是先发 CTRL_BREAK 请它自己退**
    试过这条路，**在 Windows 上不成立**：要发 CTRL_BREAK_EVENT，目标必须和调用方
    共用一个控制台，而我们的子进程是用 CREATE_NO_WINDOW 起的（为了不弹黑窗），
    实测信号根本送不到 —— 白等 grace 秒才落到 terminate，反而更慢。
    可选方案只有两个，都不划算：
        · 不加 CREATE_NO_WINDOW → 每启动一个服务都弹一个黑窗，界面白做了；
        · 改子进程（tools/、remote_kbd/）加「停」协议 → 改动扩散到三个工具。

    所以这里就硬停，但把**硬停会造成的那点后果**在别处补上：
    键盘中继被硬停时来不及给 Pro Micro 发 RELEASEALL（游戏里按住的键会留在
    按住状态），所以停完中继之后，界面会用 deploy.services.release_all()
    自己把串口打开写一条 RELEASEALL（详见那个函数的说明）。

**stdout/stderr 合并成一股**：日志面板里按时间顺序看最省事（ffmpeg 的日志
本来就在 stderr 上，分开两股反而要来回对时间）。
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 不给子进程弹控制台窗口（其它平台恒为 0，代码不用分支）
CREATE_NO_WINDOW = 0x08000000


class Proc:
    """一个被托管的子进程。

    on_line(text)  每收一行调一次（**在读取线程里**调用 —— 界面侧只管发信号，
                   跨线程发 Qt 信号是安全的；别在回调里碰控件）
    on_exit(code)  进程结束时调一次（同样在读取线程里）
    """

    def __init__(self, name, cmd, cwd=None, on_line=None, on_exit=None,
                 hide_window=True, env_extra=None):
        self.name = name
        self.cmd = [str(c) for c in cmd]
        self.cwd = str(cwd or ROOT)
        self.on_line = on_line or (lambda _t: None)
        self.on_exit = on_exit or (lambda _c: None)
        self.flags = (CREATE_NO_WINDOW if
                      (hide_window and sys.platform == "win32") else 0)
        self.env_extra = env_extra or {}

        self.proc = None
        self.started_at = 0.0
        self._lock = threading.Lock()

    # ---------------- 状态 ----------------

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def pid(self):
        return self.proc.pid if self.proc is not None else 0

    def uptime(self):
        """已运行秒数；没在跑返回 0。"""
        return (time.time() - self.started_at) if self.running() else 0.0

    # ---------------- 启停 ----------------

    def start(self):
        """启动。返回 (ok, 错误文字)。**不抛异常** —— 界面只要一句话提示。"""
        with self._lock:
            if self.running():
                return False, "已经在运行了"

            env = dict(os.environ)
            # Python 服务不缓冲输出：否则日志要等缓冲区满才刷出来，
            # 看起来就像「启动了但什么都没发生」。
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            env.update(self.env_extra)

            try:
                self.proc = subprocess.Popen(
                    self.cmd, cwd=self.cwd, env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    text=True, encoding="utf-8", errors="replace", bufsize=1,
                    creationflags=self.flags)
            except FileNotFoundError:
                self.proc = None
                return False, "找不到可执行文件：%s" % self.cmd[0]
            except Exception as e:
                self.proc = None
                return False, "%s: %s" % (type(e).__name__, e)

            self.started_at = time.time()
            threading.Thread(target=self._pump, args=(self.proc,),
                             daemon=True).start()
            return True, ""

    def _pump(self, proc):
        """把子进程输出逐行喂给回调，结束后报一次退出码。"""
        try:
            for line in proc.stdout:
                self.on_line(line.rstrip())
        except Exception:
            pass
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass
        # 管道读到 EOF 时进程**通常**已经退出，但不保证 —— 先等一下再取码，
        # 否则会报出「已退出（code None）」，看不出到底是正常退还是崩了。
        try:
            proc.wait(timeout=3.0)
        except Exception:
            pass
        self.on_exit(proc.poll())

    def stop(self):
        """停止。返回 (ok, 说明文字)。硬停 —— 理由见模块文档。"""
        with self._lock:
            if not self.running():
                return False, "没在运行"

            for how in ("terminate", "kill"):
                try:
                    getattr(self.proc, how)()
                except Exception:
                    pass
                try:
                    self.proc.wait(timeout=1.5)
                    return True, "已停止"
                except subprocess.TimeoutExpired:
                    continue
            return False, "停不下来，进程可能已经僵死（PID %d）" % self.pid()
