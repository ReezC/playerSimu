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

import queue
import threading


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


def cancelable(iterator, ctx, timeout=0.2):
    """把 iterator 包成「一取消就停」的迭代器（配合 ctx.canceled()）。

    **为什么不能直接用 for**：一个结果都还没出来时，`for` 会一直阻塞在 next()
    上 —— 那期间 ctx.canceled() 根本读不到。多进程跑自动标注时这段特别长：
    第一个结果要等 worker 预热（Windows 上每个子进程都得重新 import 一遍
    cv2/torch 再加载模板，199 帧那次实测等了约 12 秒），于是表现成「点了取消
    没反应」。（顺带：Python 3.13 起 Pool.imap 返回的是普通生成器，早年的
    IMapIterator 连同它的 next(timeout) 一起没了，不能靠它做超时轮询。）

    做法：另起一个线程专门去阻塞地收，主线程带超时从队列里取，每 timeout 秒
    回头看一眼取消；取消就立刻结束迭代 —— 调用方该 terminate 池子就 terminate。

    **注意**：被取消掐断时迭代是「正常结束」的，调用方要自己再问一次
    `ctx.canceled()` 才知道是收完了还是被掐了（见 tools/detect_mobs.py 的 consume）。
    """
    q = queue.Queue()
    done = object()

    def pump():
        try:
            for item in iterator:
                q.put(item)
        except BaseException:
            # 池子被 terminate / 迭代器自己报错：都当收尾处理。异常若从线程里
            # 冒出去，只会在 stderr 上留一段没人接的 traceback。
            pass
        finally:
            q.put(done)

    threading.Thread(target=pump, daemon=True).start()

    while True:
        if ctx.canceled():
            return
        try:
            item = q.get(timeout=timeout)
        except queue.Empty:
            continue
        if item is done:
            return
        yield item


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
