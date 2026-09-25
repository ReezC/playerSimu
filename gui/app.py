"""数据集工作台 —— 入口。

    python -m gui.app

也可以带项目目录直接打开：
    python -m gui.app projects/101040001_野猪领土
"""

import faulthandler
import sys
import time
import traceback
from pathlib import Path

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


def main():
    # 必须在最前面：后面所有库（包括崩溃处理）都可能碰 stdout
    _ensure_std_streams()
    _install_crash_handlers()

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

    if len(sys.argv) > 1:
        try:
            win.open_project(Path(sys.argv[1]))
        except Exception as e:
            win.log("打开项目失败: %s" % e, "error")

    win.show()
    rc = app.exec_()

    # **退出顺序要定死**：先关窗、让 Qt 把销毁事件处理完，最后才轮到 QApplication。
    # 反过来的话（QApplication 先被回收）解释器退出时会去碰已经失效的 Qt 内部，
    # 表现成关掉界面后弹一个"程序已停止工作"，退出码 0xC0000005。
    # 界面里的 QGraphicsView 越多越容易撞上（实测加一个就够）。
    win.close()
    del win
    app.processEvents()
    return rc


if __name__ == "__main__":
    sys.exit(main())
