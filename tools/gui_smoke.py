"""界面冒烟测试：把每张卡片的 check_deps / make_task 都过一遍。

**为什么需要它**：这些方法只在点击「运行」时才执行，平时不跑。
一个拼写错误（比如把 wzexport 写成 wzxexport）就能让某张卡片
静默失效到用户点下去那一刻。

跑法：
    python -m tools.gui_smoke            # 用第一个项目
    python -m tools.gui_smoke <项目目录>
"""

import os
import pathlib
import sys
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # 无窗口也能跑

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt5.QtWidgets import QApplication  # noqa: E402


def main():
    app = QApplication.instance() or QApplication(sys.argv)

    from gui.main_window import MainWindow

    win = MainWindow()

    if len(sys.argv) > 1:
        target = pathlib.Path(sys.argv[1])
    else:
        found = sorted((ROOT / "projects").glob("*/project.yaml"))
        if not found:
            print("没有可用的项目 —— 先在工作台里建一个")
            return 1
        target = found[0].parent

    win.open_project(target)
    print("项目:", target.name)
    print()

    bad = 0

    for card in win.cards:
        print("── %s ──" % card.title)

        try:
            deps = card.check_deps(win.project)
            print("   check_deps :", deps)
        except Exception as e:
            bad += 1
            print("   check_deps 失败: %s: %s" % (type(e).__name__, e))
            traceback.print_exc()

        try:
            made = card.make_task(win.project)
            if made:
                print("   make_task  : ok  ->  %s" % made[0].__name__)
            else:
                print("   make_task  : None（该步骤无任务，正常）")
        except Exception as e:
            bad += 1
            print("   make_task 失败: %s: %s" % (type(e).__name__, e))
            traceback.print_exc()

        try:
            card.summarize(win.project)
        except Exception as e:
            bad += 1
            print("   summarize 失败: %s: %s" % (type(e).__name__, e))
            traceback.print_exc()

        print()

    # 顺带验证几个通用工具函数，它们也只在运行时才被调用
    try:
        from gui.worker import safe_slot
        safe_slot(lambda: 1 / 0)()
        print("safe_slot   : ok（异常被吞掉）")
    except Exception as e:
        bad += 1
        print("safe_slot   失败: %s" % e)

    print()

    # **退出顺序要定死**：先关窗、让 Qt 处理完销毁，最后才轮到 QApplication。
    # 不这么做的话，解释器退出时 Python 和 Qt 的析构顺序是不定的 —— QApplication
    # 先没、图形视图后没，就会访问已经失效的 Qt 内部，表现成退出码 0xC0000005
    # （Windows 上还能弹一个"程序已停止工作"）。实测：界面里多一个 QGraphicsView
    # （「路线识别」里的地形图）就会把命中率推到 8 次里 1~3 次。
    win.close()
    del win
    app.processEvents()

    if bad:
        print("发现 %d 处问题" % bad)
        return 1

    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
