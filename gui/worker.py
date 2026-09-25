"""后台任务线程。

铁律：QThread.run() 里只能 emit 信号，绝不能碰任何 Qt 控件。
控件只能在主线程操作，跨线程访问会随机崩溃（而且很难复现）。

Qt 的信号是线程安全的：从工作线程 emit，会自动排队到接收者所在线程
（这里是主线程）执行，所以界面更新是安全的。
"""

import inspect
import traceback

from PyQt5.QtCore import QThread, pyqtSignal

from core.context import TaskContext


def safe_slot(fn):
    """把槽函数包一层异常保护。

    **这条规则必须记住**：PyQt5 里槽函数抛出未捕获异常时，会调用 Qt 的
    qFatal()，也就是直接 abort() —— 整个程序无征兆闪退，
    Windows 事件日志里只留下「出错模块 Qt5Core.dll，异常代码 0xc0000409」。

    从工作线程连过来的槽（日志 / 进度 / 完成）必须过这一层，
    否则任何一处写错都能让程序直接消失。

    **另外：按槽自己的签名裁掉多余的信号参数。**
        Qt 发信号带的参数常常比槽需要的多（`QPushButton.clicked` 就带一个
        bool，`currentIndexChanged` 带一个 int）。而本函数返回的是
        `wrapper(*args, **kwargs)` —— PyQt 看到"什么参数都能接"，就把那个 bool
        原样传进来，于是**零参的槽抛 TypeError，又被这层自己吞掉**：
        按钮点了没有任何反应，只在 stderr.log 里留一行 traceback。
        实测踩过：标定弹窗的「保存标定」「自动定位」两个按钮都是死的，而
        自检没抓到 —— 那些用例直接调 `_on_save()`，绕过了信号。
        所以这里按形参个数裁一刀，多的丢掉（`*args` 的槽照旧原样转发）。
    """
    try:
        _params = list(inspect.signature(fn).parameters.values())
        _n_pos = len([p for p in _params
                      if p.kind in (p.POSITIONAL_ONLY,
                                    p.POSITIONAL_OR_KEYWORD)])
        _vargs = any(p.kind == p.VAR_POSITIONAL for p in _params)
    except (TypeError, ValueError):     # 内置/partial 之类拿不到签名 → 老行为
        _n_pos, _vargs = 0, True

    def wrapper(*args, **kwargs):
        try:
            return fn(*(args if _vargs else args[:_n_pos]), **kwargs)
        except Exception:
            traceback.print_exc()
    return wrapper


class QtContext(TaskContext):
    """把 ctx 调用转成 Qt 信号。信号跨线程自动排队，主线程安全接收。"""

    def __init__(self, thread):
        self._t = thread

    def log(self, msg, level="info"):
        self._t.sig_log.emit(str(msg), str(level))

    def progress(self, cur, total=0, text=""):
        self._t.sig_progress.emit(int(cur), int(total), str(text))

    def canceled(self):
        return bool(self._t.is_canceled())


class TaskThread(QThread):
    """跑一个业务函数。fn(params, ctx) -> dict | str | None

    返回 dict 时取其中的 "summary" 作为完成摘要。
    """

    sig_log = pyqtSignal(str, str)            # message, level
    sig_progress = pyqtSignal(int, int, str)  # current, total, text
    sig_done = pyqtSignal(bool, str, object)  # success, summary, result

    def __init__(self, fn, params, parent=None):
        super().__init__(parent)
        self._fn = fn
        self._params = params
        self._cancel = False

    def is_canceled(self):
        return self._cancel

    def cancel(self):
        """请求取消。业务函数需要自己在循环里检查 ctx.canceled()。"""
        self._cancel = True

    def run(self):
        ctx = QtContext(self)
        try:
            result = self._fn(self._params, ctx)
        except Exception as e:
            # 完整堆栈进日志，一句话摘要给状态灯 —— 两者都要，不然没法定位
            self.sig_log.emit(traceback.format_exc(), "error")
            self.sig_done.emit(False, "%s: %s" % (type(e).__name__, e), None)
            return

        if isinstance(result, dict):
            summary = result.get("summary", "") or ""
        elif isinstance(result, str):
            summary = result
        else:
            summary = ""

        # result 一并回传 —— 有些步骤（如标定）需要把结果写回项目配置
        self.sig_done.emit(not self._cancel, summary, result)
