"""任务上下文：把「业务逻辑」和「界面」解耦。

业务函数只认三个方法：

    ctx.log("正在抽帧", "info")
    ctx.progress(3, 10, "frame_00003")
    if ctx.canceled(): return          # 长循环里定期检查，支持中途取消

这样同一份业务代码，CLI 和 GUI 都能跑：
    CLI  -> ConsoleContext（打到 stdout）
    GUI  -> gui.worker.QtContext（转成 Qt 信号）

放在 core/ 而不是 gui/ 下，是为了不让 tools/ 反向依赖界面代码。
"""


class TaskContext:
    """默认实现什么都不做 —— 直接调用业务函数时用它，安静跑完。"""

    def log(self, msg, level="info"):
        """level: info | ok | warn | error"""

    def progress(self, cur, total=0, text=""):
        """total=0 表示总量未知（界面会显示不确定态进度条）。"""

    def canceled(self):
        return False


class ConsoleContext(TaskContext):
    """给 CLI / 调试用：直接打印。"""

    PREFIX = {"info": "", "ok": "[ok] ", "warn": "[warn] ", "error": "[err] "}

    def __init__(self, verbose=True):
        self.verbose = verbose

    def log(self, msg, level="info"):
        if not self.verbose:
            return
        text = self.PREFIX.get(level, "") + str(msg)
        try:
            print(text, flush=True)
        except UnicodeEncodeError:
            # Windows 控制台默认 GBK，编不了的字符会让 print 抛异常。
            # 一行日志不值得把整个任务搞崩，退化成 ascii 安全输出。
            print(text.encode("ascii", "replace").decode("ascii"), flush=True)

    def progress(self, cur, total=0, text=""):
        if not self.verbose or not total:
            return
        step = max(1, total // 10)
        if cur == total or cur % step == 0:      # 每 10% 打一次，别刷屏
            print("  [%d/%d] %s" % (cur, total, text), flush=True)


class CollectContext(TaskContext):
    """把日志收进列表，供测试断言。"""

    def __init__(self):
        self.logs = []
        self.progress_last = None

    def log(self, msg, level="info"):
        self.logs.append((level, str(msg)))

    def progress(self, cur, total=0, text=""):
        self.progress_last = (cur, total, text)

    def text(self):
        return "\n".join(m for _, m in self.logs)
