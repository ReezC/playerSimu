"""数据集工作台 —— 入口。

    python -m gui.app

也可以带项目目录直接打开：
    python -m gui.app projects/101040001_野猪领土
"""

import faulthandler
import os
import sys
import threading
import time
import traceback
from pathlib import Path

# ⭐ **输出（含被重定向的日志）统一按 utf-8**（2026-10-03 ✓ 现场排查吃过这个亏）：
#   Windows 下把 stdout 重定向到文件时，默认用**本地编码（GBK）** ⇒ 路径里的**中文会被吃掉**
#   （实测：`projects\石人寺院III\runs\...` 在日志里变成 `projects\III\runs\...` ✗）⇒
#   事后排查很容易**认错项目** ✗（"那条警告说的到底是哪个项目？"）。
#   `reconfigure` 只改编码、不动缓冲（Python 3.7+ ✓）。
#   ⚠ 失败就算了（例如 stdout 被替换成不支持 `reconfigure` 的对象 ✓）—— 绝不能因为
#     日志编码把程序起不来 ✗。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ⚠ 这段必须在导入 PyQt5 **之前**执行，位置不能挪。
#
# PyQt5/Qt5/bin/ 里自带一整套 MSVC 运行时（msvcp140.dll、vcruntime140.dll 等）。
# 它一旦先加载，这套旧版本就被钉在进程里；之后 torch 的 c10.dll 要链接
# 系统里的新版运行时，就会报 WinError 1114（DLL 初始化例程失败）。
# 表现极具迷惑性：**同一个训练脚本，命令行能跑，在 GUI 里必失败**。
#
# 这里先把系统目录那份拉起来（ctypes.CDLL 不带路径 → 走系统搜索顺序），
# PyQt5 之后就会复用它，不再加载自己那份。
#
# 为什么不直接 import torch：那样每次启动都要多花 3~5 秒导入 torch，
# 哪怕这次根本不训练。预加载只要约 1 毫秒。
import ctypes as _ctypes

for _dll in ("msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
             "vcruntime140.dll", "vcruntime140_1.dll"):
    try:
        _ctypes.CDLL(_dll)
    except OSError:
        pass

from PyQt5.QtCore import Qt                      # noqa: E402
from PyQt5.QtWidgets import QApplication          # noqa: E402

from gui.widgets import install_wheel_guard       # noqa: E402

# 项目根加入 sys.path，这样从任意工作目录启动都能 import tools / link / core
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CRASH_LOG = ROOT / "crash.log"


def _ensure_std_streams():
    """给 sys.stdout / stderr 兜底。

    **为什么必须做**
        用 pythonw.exe 启动（双击 bat 就是这样）时**没有控制台**，
        sys.stdout 和 sys.stderr 直接是 None。而不少第三方库会无条件
        往它们写东西 —— 比如 ultralytics 的 tqdm 在收尾时执行
        self.file.write("\\n")，拿到 None 就抛
        AttributeError: 'NoneType' object has no attribute 'write'。

        报错信息指向 tqdm，完全看不出真正原因是"没有 stdout"，
        在本项目里表现为「训练刚起步就失败，命令行却跑得好好的」。

    指向日志文件而不是 devnull：既不崩，又能事后翻原始输出排查。
    """
    import os

    for name, path in (("stdout", ROOT / "stdout.log"),
                       ("stderr", ROOT / "stderr.log")):
        if getattr(sys, name, None) is not None:
            continue
        try:
            f = open(path, "a", encoding="utf-8", buffering=1)
        except Exception:
            try:
                f = open(os.devnull, "w", encoding="utf-8")
            except Exception:
                continue
        setattr(sys, name, f)


def _install_crash_handlers():
    """让崩溃留下证据。

    **原生级崩溃**（段错误、Qt 的 0xc0000409 之类）不走 Python 的异常机制 ——
    没有这层，表现就是"程序突然没了"，而 Windows 事件日志只会告诉你
    「出错模块 Qt5Core.dll」，查不出是哪一行。

    faulthandler 会把崩溃瞬间的 Python 堆栈写进 crash.log。
    """
    try:
        f = open(CRASH_LOG, "a", buffering=1, encoding="utf-8")
        f.write("\n=== 启动 %s ===\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
        faulthandler.enable(f)
    except Exception:
        pass

    def hook(exc_type, exc, tb):
        try:
            with open(CRASH_LOG, "a", encoding="utf-8") as fh:
                fh.write("\n=== 未捕获异常 %s ===\n"
                         % time.strftime("%Y-%m-%d %H:%M:%S"))
                traceback.print_exception(exc_type, exc, tb, file=fh)
        except Exception:
            pass
        traceback.print_exception(exc_type, exc, tb)

    sys.excepthook = hook


#: ⭐⭐ **「关了必须走 / 卡住不许烧核」的两道看门狗**（用户 2026-10-05 ✓ 原话："保证以后
#:   不要关了该占进程就行"）。为什么必须做成代码里的硬保证、而不是"下次注意"：
#:   实测有 **4 个工作台进程卡在启动里烧满一个核、烧了 25.7 小时** ✗（`py-spy` 抓栈：
#:   `theme._row_title_of` 那个已经修掉的死循环 ✓ —— 当时窗口**从来没出来**、进程活着、
#:   没有任何异常 ⇒ 用户看到的就是"打不开 + 机器变卡"✓）。这类 bug 写下去能跑、不报错、
#:   只是卡 ✗ ⇒ **只能靠一个到点就动手的东西兜底** ✓。
#: 退出码分开（看日志一眼知道卡在哪一段 ✓）：**3 = 启动卡住** / **4 = 收尾卡住** ✓。
WATCHDOG_STARTUP_S = 90.0
WATCHDOG_EXIT_S = 20.0


def _arm_watchdog(seconds, tag, done, code):
    """起一条守护线程：`seconds` 秒后若 `done` 还没置上 ⇒ **留栈 + 强制退出** ✓。

    · 留栈：`faulthandler.dump_traceback()` 写进 `crash.log`（**哪个线程卡在哪一行** ✓
      —— 这正是这次定位 4 个僵尸进程用的那份证据 ✓）；
    · 强制退出：`os._exit(code)` ✓ —— **不走解释器收尾**（卡住的收尾本来就是元凶 ✗）。
    ⚠ 正常路径**绝不会**被误杀：`done` 一置上它就安静退出 ✓（用例钉着这条 ✓）。
    ⚠ 只是"守护线程 + 等事件"，不 join、不占 CPU ✓。
    """
    def _fire():
        if done.wait(timeout=max(0.1, float(seconds))):
            return
        try:
            with open(CRASH_LOG, "a", encoding="utf-8") as fh:
                fh.write("\n=== %s看门狗触发（%s：%.0f 秒没走完）===\n"
                         % (tag, time.strftime("%Y-%m-%d %H:%M:%S"), seconds))
                faulthandler.dump_traceback(fh)
                fh.write("=== 已强制退出（exit=%d）—— 上面那份栈就是卡住的地方 ===\n"
                         % int(code))
        except Exception:                            # noqa: BLE001 —— 留不下证据也要退 ✗
            pass
        os._exit(int(code))

    th = threading.Thread(target=_fire, name="watchdog-%s" % tag, daemon=True)
    th.start()
    return th


def main():
    # 必须在最前面：后面所有库（包括崩溃处理）都可能碰 stdout
    _ensure_std_streams()
    _install_crash_handlers()
    # ⭐ **启动看门狗**：从这一刻起到"窗口建好"为止（历史上这一段死过 ✗ 见上面那段）
    _boot_done = threading.Event()
    _arm_watchdog(WATCHDOG_STARTUP_S, "启动", _boot_done, 3)

    # 性能保活：向系统声明「别把我当后台程序降级」——关电源节流（EcoQoS 降频）、
    # 进程优先级→高于正常、定时器精度→1 ms、防挂起（见 core/winperf.py）。
    #
    # **为什么非做不可**：关键回路是 10 ms 级的时序拍（decision/agent.py 的
    # TIMING_TICK），而 Windows 默认定时器粒度约 15.6 ms —— 想睡 10 ms 实际睡
    # 15.6 ms；再叠加系统对**非前台进程**的降频与定时器合并，症状就是
    # 「窗口一失焦就卡」。原先全项目没有任何一处向系统声明过这件事
    #（timeBeginPeriod 只在 wgc_capture 被 import 时调过，而那条路只有
    # 「来源 = 本地窗口」抓屏才走 → **收流模式全程停在 15.6 ms**）。
    #
    # 放在这里（而不是某个回路启动时）：进程级设置越早声明越好，且必须赶在
    # 任何线程/定时器起来之前。开关在「设置 → 性能保活」。
    from core import winperf
    keepalive = None
    try:
        if winperf.enabled_by_config():
            # probe=5：顺带实测 5 次「睡 10 ms 到底睡了多久」（约 50 ms 启动开销）。
            # 这是唯一能自己证明"真的生效"的数字，值回这点时间。
            keepalive = winperf.apply(probe=5)
    except Exception as e:                 # 保活失败绝不能影响启动
        keepalive = None
        print("性能保活未生效: %s: %s" % (type(e).__name__, e))

    # 高分屏：必须在 QApplication 创建之前设置
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # Windows 任务栏图标：显式绑定 AppUserModelID，否则 pythonw 启动时
    # 任务栏会显示 python 的图标而不是我们设置的图标。
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "playerSimu.workbench")
        except Exception:
            pass

    # ⭐⭐ **让主线程（Qt 事件循环）抢得到 GIL**（用户 2026-10-06 ✓ 现场原话：
    #   "**视频预览只有 10~13fps，整个窗口都发木**"✓）。为什么是这一项：
    #     · `draw_ms` 只有 **0.01 ms** ✓ ⇒ **不是画得慢，是轮不到它** ✗；
    #     · 后台那条流水线每拍 ~24 ms ✓，里头全是 Python 级的 numpy / 格式化 / str ✓
    #       ⇒ CPython 默认**每 5 ms 才切一次 GIL** ✓ ⇒ 主线程排队等 ✗ ⇒ 预览十几帧、
    #       菜单/悬停全木 ✓（两个现象同一个原因 ✓）。
    #   ⇒ `setswitchinterval(1 ms)`：**切得勤一点** ✓ —— 对吞吐几乎无损（切片开销可以忽略 ✓），
    #     换来主线程"随叫随到" ✓。⚠ 必须在**起线程之前**设 ✓（后面那些线程都会继承它 ✓）。
    try:
        sys.setswitchinterval(0.001)
    except Exception:                       # noqa: BLE001 —— 设不动就算了，别影响启动 ✓
        pass

    app = QApplication(sys.argv)
    # 和窗口标题用同一个名字（定义在 gui/main_window.py，那边要拼上项目名）
    from gui.main_window import APP_NAME
    app.setApplicationName(APP_NAME)

    # 滚轮不许改参数（数字框/下拉框/滑块/标签栏）。装在应用级是故意的：
    # 靠「每个控件记得用 NoWheel* 子类」已经漏了十几处，必须有个兜底。
    # 详见 docs/UI规范.md。
    install_wheel_guard(app)

    # 应用图标：MapleNecrocer 图标去掉红色通道（见 gui/icon.ico）
    from PyQt5.QtGui import QIcon
    _icon = ROOT / "gui" / "icon.ico"
    if _icon.exists():
        app.setWindowIcon(QIcon(str(_icon)))

    # 应用保存的界面字号（在窗口创建之前，这样一开就是设置好的大小）
    from gui import theme
    theme.apply(app)

    from gui.main_window import MainWindow
    win = MainWindow()
    if _icon.exists():
        win.setWindowIcon(QIcon(str(_icon)))

    # 保活结果写进运行日志：**看得见**才不会变成"玄学调优" ——
    # 失焦变卡这类问题，人会下意识怀疑"是不是设置问题"，日志里有这行就能直接排除。
    from core import winperf as _wp
    if keepalive is None:
        win.log("性能保活：未启用（设置 → 性能保活 里可以打开）", "warn")
    else:
        msg = _wp.describe()
        if keepalive.get("probe_ms") is not None:
            # **只报"现在是多少"**，不写"未提升精度时约 15.6 ms" —— 实测过：
            # 有的机器（含本机的 B 机）默认就已经是 1 ms 级（别的程序早把系统
            # 定时器分辨率提上去了）。写上那句，会让"这一项到底是不是我的瓶颈"
            # 变成一句空话。想知道改前/改后，跑 tools.selftest_winperf（它实测两次）。
            msg += "　|　实测量到 sleep(10ms) = %.1f ms" % keepalive["probe_ms"]
        win.log(msg, "info")

    if len(sys.argv) > 1:
        try:
            win.open_project(Path(sys.argv[1]))
        except Exception as e:
            win.log("打开项目失败: %s" % e, "error")

    win.show()
    # ⭐ 窗口出来了 ⇒ 启动看门狗收工（后面若还卡，就不是"打不开"那一段了 ✓）
    _boot_done.set()

    # ⭐⭐ **GC 调优**（用户 2026-10-06 ✓ 同一次现场：`pipe_ms` **最大 829 ms** ✗、
    #   整个窗口发木 ✓；而**内存是稳定的** ✓ —— 2.8 GB 平台期，**不是泄漏** ✗）：
    #     · `gc.freeze()`：把**此刻已经活着**的常驻对象（torch / onnxruntime / Qt / 本体 ✓）
    #       挪进"永久代"⇒ 之后的 gen2 全扫**不再扫它们** ✓（这么大一堆每次全扫都是几十毫秒 ✓
    #       ⇒ 正好对上那种"跑几秒突然一顿"✗）；
    #     · `set_threshold` 放宽：少触发几轮 gen0/gen1 ✓（垃圾仍然会被回收 ✓ —— 只是
    #       "什么时候扫"变稀 ✓）。
    #   ⚠ 放在 `exec_()` **之前**：那时 torch 等大件都已经 import 完 ✓（模型本体是后面
    #     懒加载的 ✓ ⇒ 它不在冻结集里 ✓ —— 这是明知的小折扣：宁可少冻一部分，
    #     也不想把 freeze 塞进实时线程那条路上 ✓）。
    try:
        import gc as _gc
        _gc.collect()
        _gc.freeze()
        _gc.set_threshold(50000, 30, 30)
    except Exception:                       # noqa: BLE001 —— 调不动就算了 ✓
        pass

    rc = app.exec_()

    # ⭐⭐ **收尾看门狗**（"关了必须走" ✓）：关窗之后的清理——停实时/绘制线程、存界面状态、
    #    释放保活——**任何一步卡住都可能是那个"关了还占进程"✗** ⇒ 到点留栈 + `os._exit(4)` ✓。
    _exit_done = threading.Event()
    _arm_watchdog(WATCHDOG_EXIT_S, "收尾", _exit_done, 4)

    # **退出顺序要定死**：先关窗、让 Qt 把销毁事件处理完，最后才轮到 QApplication。
    # 反过来的话（QApplication 先被回收）解释器退出时会去碰已经失效的 Qt 内部，
    # 表现成关掉界面后弹一个"程序已停止工作"，退出码 0xC0000005。
    # 界面里的 QGraphicsView 越多越容易撞上（实测加一个就够）。
    win.close()
    del win
    app.processEvents()
    # 把定时器精度还回去（timeEndPeriod）：不还的话，本进程占着 1 ms 精度的
    # 名额不放 —— 在高精度定时器是稀缺资源的系统上，别的程序会吃不到。
    _wp.release()
    _exit_done.set()                 # ⭐ 收尾看门狗收工（正常路径绝不被误杀 ✓）
    return rc


if __name__ == "__main__":
    sys.exit(main())
