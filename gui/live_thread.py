"""实时收流 + 推理的后台线程。

**为什么必须独立线程**
    收流（等帧）和推理（等 GPU）都是阻塞操作。放主线程里窗口会立刻失去响应。

**为什么显示要限流，而且统计要拆成三个数**
    这是一条流水线，三个环节速度不同：

        收流 60fps  →  推理 8ms/帧（125fps 上限）  →  界面显示 30fps 就够看

    如果每帧都推给界面、还要等它画完，显示就成了瓶颈 ——
    推理被迫降到 30fps，测出来的帧率是假的，等于把最该看的数字掩盖了。

    所以：**推理照常跑，显示按自己的节奏抽帧**。界面跟不上就少显示几帧，
    但绝不能让界面拖慢推理。

    同时把三个速度分开上报（收流 fps / 推理 ms / 显示 fps）——
    只给一个 "fps" 的话，一出问题根本分不清是网络掉帧、模型太慢，
    还是界面拖累。
"""

import threading
import time

from PyQt5.QtCore import QThread, pyqtSignal

# 类别 → 框颜色（BGR，和 data.yaml 的 class 对齐）
BOX_COLORS = {
    0: (255, 128, 0),    # player 蓝
    1: (60, 220, 60),    # mob 绿
    2: (0, 255, 255),    # drop 黄
    3: (0, 0, 255),      # npc 红
}


class _LatestSlot:
    """只保留最新一帧的槽位。

    **为什么需要它（这是实时链路里最要命的一环）**
        PyAVSource.read() 是阻塞的，它会按顺序把积压的包一帧帧交给你。
        可 B 机的处理速度（解码+推理）通常追不上 60fps 的推流速度，
        于是每秒欠下的帧全部堆在内核 UDP 缓冲里（默认几 MB）——

            A 机推 60fps  →  B 机只能处理 30fps  →  内核里越堆越多
                                                 ↓
                            画面一直在播「几秒前」的内容，像慢动作

        而且积压不会自己消失：处理多慢，延迟就一直累积下去。

        实时系统的铁律是**宁可丢帧、绝不排队**。所以让一个独立线程拼命读、
        只往这里放最新一帧，消费端取到的永远是最新画面，中间的直接丢。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._evt = threading.Event()
        self.dropped = 0
        self.put_count = 0

    def put(self, frame):
        """放一帧。已有未取走的帧就被覆盖 —— 那正是要丢掉的旧画面。"""
        with self._lock:
            self.put_count += 1
            if self._frame is not None:
                self.dropped += 1
            self._frame = frame
        self._evt.set()

    def take(self, timeout=0.5):
        """取最新一帧；超时返回 None（用于让调用方有机会检查停止标志）。"""
        if not self._evt.wait(timeout):
            return None
        with self._lock:
            f = self._frame
            self._frame = None
            self._evt.clear()
            return f

    def clear(self):
        with self._lock:
            self._frame = None
        self._evt.clear()


class LiveThread(QThread):
    frame_ready = pyqtSignal(object)     # numpy BGR 图（已画好框）
    stats_ready = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self._p = dict(params)
        self._stop = threading.Event()

    # ---------------- 控制 ----------------

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.is_set()

    # ---------------- 主循环 ----------------

    def run(self):
        try:
            self._run()
        except Exception as e:
            if not self._stop.is_set():
                self.failed.emit("%s: %s" % (type(e).__name__, e))

    def _run(self):
        import cv2

        from link import PyAVSource

        url = self._p["url"]
        conf = float(self._p.get("conf", 0.30))
        imgsz = int(self._p.get("imgsz", 960))
        device = str(self._p.get("device", "0"))
        weights = self._p.get("weights") or ""
        draw = bool(self._p.get("draw", True))
        show_fps = max(1.0, float(self._p.get("show_fps", 30.0)))

        from ultralytics import YOLO
        model = YOLO(weights)

        # BGR 直出，省一次颜色转换
        src = PyAVSource(url, decode_format="bgr24",
                         container_format=self._p.get("format"))
        src.open()

        # 读线程独立于推理：它拼命读，积压的旧帧在槽位里被直接覆盖丢掉。
        # 这样无论推理多慢，看到的都是最新画面，延迟不会累积。
        slot = _LatestSlot()
        reader_done = threading.Event()

        def _reader():
            try:
                while not self._stop.is_set() and not reader_done.is_set():
                    f = src.read()
                    if f is None:
                        break
                    slot.put(f)
            except Exception:
                pass
            finally:
                reader_done.set()

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()

        show_interval = 1.0 / show_fps
        t_start = time.perf_counter()
        t_last_show = 0.0
        t_last_stat = t_start

        n = 0
        n_show = 0
        n_boxes = 0
        infer_ms = []
        gaps = []
        last_mono = None

        # 端到端延迟：解码 A 机屏幕上的时间码探针。
        #
        # **为什么必须用探针，而不是拿 pts 推算**
        #   pts 是 A 机编码器给的时间戳。拿它和"第一帧到达时刻"对齐，
        #   测出来的其实只是网络抖动 —— 编码器前面攒了多久，基线对齐时
        #   被整个抵消掉了。表现就是监控显示"延迟几毫秒"，体感却滞后几百毫秒。
        #
        #   探针是唯一可靠的基准：它是**画在画面里的绝对时刻**，
        #   跟着画面一起被采集编码传输，B 机解出来和本地时钟一比就是真实延迟。
        #   前提是 B 机做过时钟同步（tools.clock_sync --host <A机IP> --save）。
        probe_on = bool(self._p.get("probe", False))
        px = int(self._p.get("probe_x", 100))
        py = int(self._p.get("probe_y", 8))
        cell = int(self._p.get("probe_cell", 16))
        gap = float(self._p.get("probe_gap", 2))
        bits = int(self._p.get("probe_bits", 40))
        offset_ms = float(self._p.get("clock_offset_ms", 0.0) or 0.0)

        delays = []
        probe_miss = 0
        if probe_on:
            # 自动对时：offset 会随 Windows NTP 漂移，后台线程里重新对一次，
            # 失败就沿用 params 传进来的旧值。
            try:
                from tools.clock_sync import sync_offset
                off = sync_offset(save=True, quiet=True)
                if off is not None:
                    offset_ms = off
            except Exception:
                pass
            try:
                from tools.probe_codec import decode_ms, resolve_delay_ms
            except Exception as e:
                self.failed.emit("探针解码模块不可用: %s" % e)
                probe_on = False
                decode_ms = resolve_delay_ms = None

        try:
            while not self._stop.is_set():
                f = slot.take(0.5)
                if f is None:
                    if reader_done.is_set():
                        break          # 流结束了
                    continue           # 只是暂时没新帧，继续等
                n += 1

                if last_mono is not None:
                    gaps.append((f.t_recv_mono - last_mono) * 1000.0)
                last_mono = f.t_recv_mono
                vis = f.image          # BGR（decode_format="bgr24" 直出）

                if probe_on:
                    # f.image 是 BGR，转灰度解码
                    gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
                    ts_a = decode_ms(gray, px, py, cell, gap, bits)
                    if ts_a is None:
                        probe_miss += 1
                    else:
                        d = resolve_delay_ms(
                            (f.t_recv_wall + offset_ms / 1000.0) * 1000.0, ts_a)
                        # 超过 5 秒视为解码错误（而不是真的有 5 秒延迟）
                        if d is not None and d < 5000:
                            delays.append(d)
                            if len(delays) > 120:
                                del delays[0]

                t0 = time.perf_counter()
                res = model.predict(vis, conf=conf, imgsz=imgsz,
                                    device=device, verbose=False)[0]
                dt_ms = (time.perf_counter() - t0) * 1000.0
                infer_ms.append(dt_ms)
                if len(infer_ms) > 60:
                    del infer_ms[0]

                k = 0
                boxes = getattr(res, "boxes", None)
                if boxes is not None and len(boxes):
                    xyxy = boxes.xyxy.cpu().numpy()
                    cfs = boxes.conf.cpu().numpy()
                    try:
                        clss = boxes.cls.cpu().numpy().astype(int)
                    except Exception:
                        clss = [1] * len(cfs)
                    for (x1, y1, x2, y2), c, cls in zip(xyxy, cfs, clss):
                        k += 1
                        if not draw:
                            continue
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        color = BOX_COLORS.get(int(cls), BOX_COLORS[1])
                        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                        ty = y1 - 5 if y1 > 14 else y1 + 16
                        cv2.putText(vis, "%.2f" % c, (x1 + 2, ty),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    color, 1, cv2.LINE_AA)
                n_boxes = k

                now = time.perf_counter()

                # 显示限流：到点了才推一帧。emit 是队列信号，不阻塞推理，
                # 所以推理始终按自己的速度跑。
                if now - t_last_show >= show_interval:
                    t_last_show = now
                    n_show += 1
                    self.frame_ready.emit(vis)

                if now - t_last_stat >= 0.5:
                    el = now - t_start
                    gaps_recent = gaps[-60:]
                    self.stats_ready.emit({
                        # 读线程真正收到的帧率 —— 这是链路能给的输入速度
                        "recv_fps": slot.put_count / el if el > 0 else 0.0,
                        # 我们实际处理了多少帧 —— 两个数差得越多，说明丢帧越多
                        "proc_fps": n / el if el > 0 else 0.0,
                        "dropped": slot.dropped,
                        "infer_ms": sorted(infer_ms)[len(infer_ms) // 2] if infer_ms else 0.0,
                        "delay_ms": (sorted(delays)[len(delays) // 2]
                                     if delays else None),
                        "probe_on": probe_on,
                        "probe_miss": probe_miss,
                        "show_fps": n_show / el if el > 0 else 0.0,
                        "boxes": n_boxes,
                        "frames": n,
                        "gap_p95": (sorted(gaps_recent)[int(len(gaps_recent) * 0.95)]
                                    if gaps_recent else 0.0),
                        "bad": getattr(src, "bad_packets", 0),
                        "size": src.size,
                        "fps_src": src.fps,
                    })
                    t_last_stat = now

        finally:
            reader_done.set()
            try:
                src.close()
            except Exception:
                pass
            reader.join(timeout=1.0)
