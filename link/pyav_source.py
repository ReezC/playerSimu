import time

import av
import numpy as np

from .base import FrameSource
from .frame import Frame

# 低延迟参数（保守版）
# 注意：不要设 analyzeduration=0 或过小的 probesize，否则 mpegts 探测不到
# PAT/PMT，container.streams.video 会为空并报 IndexError。
LOW_LATENCY_OPTIONS = {
    # 不缓冲输入，解码器也不做额外延迟
    "fflags": "nobuffer",
    "flags": "low_delay",
    # 关键：否则 ffmpeg 会为了"平滑"而额外攒一批帧再吐出来。
    # 这两个参数没设的话，光靠上层丢帧是治不了延迟的 ——
    # 帧还卡在 ffmpeg 内部的队列里，上层根本读不到它。
    "max_delay": "0",
    "reorder_queue_size": "0",
    # 探测流格式用，只在开流时一次性生效，不影响持续延迟。
    # 不要设成 0 或过小：mpegts 的 PAT/PMT 探测不到会导致找不到视频轨。
    "analyzeduration": "1000000",
    "probesize": "1000000",
    # UDP 读缓冲。小一点让积压更快暴露、更早丢弃；太大反而成了隐藏的延迟源。
    #
    # 131072 曾经是这里的问题：实测端到端延迟稳定在 665ms 且不随时间增长，
    # 说明不是"越积越多"，而是一条固定深度的缓冲管道。按常见码率折算，
    # 131KB 相当于 2Mbps 下 524ms / 4Mbps 下 262ms 的排队量 —— 量级对得上。
    #
    # 调小到 32768（2Mbps 下约 130ms）。代价是突发丢包时更容易溢出丢弃，
    # 表现为画面花屏或 miss 上升；如果出现，就回退到 65536 折中。
    "fifo_size": "32768",
    # socket 层接收缓冲（SO_RCVBUF）。默认 64KB 甚至更大，是 kernel 层
    # 的隐藏积压点 —— 数据到得比应用读得快时，先堆在这里，再进 fifo。
    # 和 fifo 一样设小，让积压更快暴露、更早丢弃。
    "buffer_size": "32768",
    "rtbufsize": "262144",
    "overrun_nonfatal": "1",
}


class PyAVSource(FrameSource):
    """基于 PyAV 的实时流接收（udp / srt / rtsp）。

    url 形如：
      udp : "udp://0.0.0.0:5000"   <- 必须写 0.0.0.0；PyAV 不认 udp://@:5000
      srt : "srt://0.0.0.0:9000?mode=listener"
      rtsp: "rtsp://192.168.1.8:8554/live"

    实现要点：用 demux + 逐包 decode，而不是 container.decode()。
    因为 UDP 丢包会产生坏包，container.decode() 一旦抛异常整个生成器就废了；
    逐包解码可以只跳过坏包，并统计丢包数。
    """

    def __init__(self, url, options=None, decode_format="rgb24",
                 decode_all=False, skip_nonref=False, container_format=None):
        """
        skip_nonref: 只解参考帧（I/P），跳过 B 帧等非参考帧。
            能显著减轻解码负担，但帧率会掉到 GOP 内的参考帧数量
            （推流端 -g 60 时大约还剩 30fps）。延迟敏感、又跑不动解码时值得开。

        container_format: 强制指定输入格式，对应 av.open(format=...)。
            **裸流必须指定** —— 例如 ffmpeg 用 `-f mjpeg` 推出来的流
            没有容器头（既不是 mpegts 也不是 mp4），PyAV 的自动探测会失败：
                InvalidDataError: Invalid data found when processing input
            这时传 container_format="mjpeg" 就能正常打开。
        """
        self.url = url
        self.options = {**LOW_LATENCY_OPTIONS, **(options or {})}
        self.decode_format = decode_format
        self.decode_all = decode_all
        self.skip_nonref = skip_nonref
        self.container_format = container_format

        self._container = None
        self._stream = None
        self._gen = None
        self._frame_id = 0
        self._bad_packets = 0
        self._size = None
        self._fps = None
        # 每帧带上 time_base，调用方才能把 pts 换算成秒。
        # 端到端延迟的估算全靠它：不知道 pts 的单位就没法算出
        # "这帧本该什么时候到"。
        self._meta = {}

    def open(self):
        # format=None 时 PyAV 走自动探测；裸流（如 -f mjpeg 推出来的）必须显式给
        url = self.url
        # UDP 加超时：否则 A 机停止推流后 recv 会无限阻塞，stop() 时 close 和
        # read 抢同一个 ffmpeg 上下文直接死锁。timeout 单位是微秒（5 秒）。
        # 不能太短：探针可能已抢先收掉流开头的 PAT/PMT，open 要等下一个关键包，
        # 太短会在探测成功前就超时失败。
        if url.startswith("udp") and "timeout=" not in url:
            sep = "&" if "?" in url else "?"
            url = url + sep + "timeout=5000000"
        self._container = av.open(url, mode="r", options=self.options,
                                  format=self.container_format)

        streams = self._container.streams.video
        if not streams:
            raise RuntimeError(
                "未找到视频轨: " + str(self.url) + "\n"
                "常见原因: probesize/analyzeduration 过小导致 mpegts 探测失败;"
                " 或发送端未推流; 或 URL 主机名写法不对(应为 0.0.0.0)。"
            )

        self._stream = streams[0]

        # 解码线程：FRAME 并行，但**把预取量钉死在 2 帧**。
        #
        # 三种取值实测对比（1080p60、16 核）：
        #
        #   "AUTO"  帧级并行 + 按核数预取（16 帧 ≈ 267ms 固定缓冲）。
        #           症状：延迟恒定在 655ms，不随积压变化 —— 正是固定
        #           深度缓冲的特征。注意这一层**不受 fflags/max_delay/
        #           reorder_queue_size 影响**，那些管的是容器层，
        #           管不到编解码器内部的预取队列。
        #
        #   "NONE"  单线程，预取缓冲消失（延迟从恒定值变成一路上降），
        #           但 1080p60 每帧只有 16.7ms 预算，单核解不过来：
        #           实测 fps 只有 7，积压把延迟冲到 2 秒再慢慢追。
        #
        #   "FRAME" + thread_count=2  折中：保留并行解码能力，
        #           预取仅 2 帧（约 33ms）。这是低延迟直播的常用配法。
        self._stream.thread_type = "FRAME"
        try:
            self._stream.codec_context.thread_count = 2
        except Exception:
            # 老版本 PyAV 可能没有这个属性，退回 AUTO 也不致命
            self._stream.thread_type = "AUTO"

        if self.skip_nonref:
            self._stream.codec_context.skip_frame = "NONREF"

        self._size = (self._stream.codec_context.width, self._stream.codec_context.height)
        try:
            self._fps = float(self._stream.average_rate)
        except Exception:
            self._fps = None

        self._meta = {}
        try:
            if self._stream.time_base:
                self._meta["time_base"] = float(self._stream.time_base)
        except Exception:
            pass

        self._gen = self._container.demux(self._stream)

    def read(self):
        if self._gen is None:
            raise RuntimeError("PyAVSource 未 open()")

        while True:
            try:
                packet = next(self._gen)
            except StopIteration:
                return None
            except Exception:
                # next 拿不到包（udp 超时 / 读取错误）= 暂时没帧，抛给调用方，
                # 让它有机会检查停止标志。坏包是拿到 packet 后 decode 失败，走下面分支。
                raise TimeoutError()

            if packet.size == 0:
                continue

            try:
                frames = packet.decode()
            except Exception:
                # UDP 丢包导致的坏包：跳过，不要中断整个流
                self._bad_packets += 1
                continue

            for av_frame in frames:
                t_mono = time.perf_counter()
                t_wall = time.time()
                try:
                    img = av_frame.to_ndarray(format=self.decode_format)
                except Exception:
                    continue

                if (not self._size or not self._size[0]) and av_frame.width:
                    self._size = (av_frame.width, av_frame.height)

                self._frame_id += 1
                return Frame(
                    image=img,
                    frame_id=self._frame_id,
                    t_recv_wall=t_wall,
                    t_recv_mono=t_mono,
                    pts=av_frame.pts,
                    meta=self._meta,
                )

    def close(self):
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
        self._container = None
        self._stream = None
        self._gen = None

    @property
    def size(self):
        return self._size

    @property
    def fps(self):
        return self._fps

    @property
    def bad_packets(self):
        return self._bad_packets

    def to_bgr(self, img):
        return img[..., ::-1]
