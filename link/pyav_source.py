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
    "fifo_size": "131072",
    "rtbufsize": "524288",
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
                 decode_all=False, skip_nonref=False):
        """
        skip_nonref: 只解参考帧（I/P），跳过 B 帧等非参考帧。
            能显著减轻解码负担，但帧率会掉到 GOP 内的参考帧数量
            （推流端 -g 60 时大约还剩 30fps）。延迟敏感、又跑不动解码时值得开。
        """
        self.url = url
        self.options = {**LOW_LATENCY_OPTIONS, **(options or {})}
        self.decode_format = decode_format
        self.decode_all = decode_all
        self.skip_nonref = skip_nonref

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
        self._container = av.open(self.url, mode="r", options=self.options)

        streams = self._container.streams.video
        if not streams:
            raise RuntimeError(
                "未找到视频轨: " + str(self.url) + "\n"
                "常见原因: probesize/analyzeduration 过小导致 mpegts 探测失败;"
                " 或发送端未推流; 或 URL 主机名写法不对(应为 0.0.0.0)。"
            )

        self._stream = streams[0]
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
                self._bad_packets += 1
                continue

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
