import time

import av
import numpy as np

from .base import FrameSource
from .frame import Frame

# 低延迟关键参数：缺任一都会引入数十到数百毫秒缓冲
LOW_LATENCY_OPTIONS = {
    "fflags": "nobuffer",
    "flags": "low_delay",
    "probesize": "32768",
    "analyzeduration": "0",
    "max_delay": "0",
    "reorder_queue_size": "0",
}


class PyAVSource(FrameSource):
    """基于 PyAV 的实时流接收（udp / srt / rtsp）。

    url 形如：
      udp : "udp://@:5000?fifo_size=1000000&overrun_nonfatal=1"
      srt : "srt://0.0.0.0:9000?mode=listener&latency=20000"
      rtsp: "rtsp://192.168.1.8:8554/live"  (需 rtsp_transport=tcp/udp)
    """

    def __init__(
        self,
        url: str,
        options: dict | None = None,
        decode_format: str = "rgb24",
        decode_all: bool = False,
    ) -> None:
        self.url = url
        self.options = {**LOW_LATENCY_OPTIONS, **(options or {})}
        self.decode_format = decode_format
        # True  = 解所有帧（含 B 帧，延迟高）
        # False = 丢非参考帧，优先低延迟
        self.decode_all = decode_all

        self._container = None
        self._stream = None
        self._gen = None
        self._frame_id = 0
        self._size: tuple[int, int] | None = None
        self._fps: float | None = None

    def open(self) -> None:
        self._container = av.open(self.url, mode="r", options=self.options)
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"
        if not self.decode_all:
            # 跳过非参考帧，显著降低解码延迟
            self._stream.codec_context.skip_frame = "NONREF"
        self._size = (self._stream.codec_context.width, self._stream.codec_context.height)
        try:
            self._fps = float(self._stream.average_rate)
        except Exception:
            self._fps = None
        self._gen = self._container.decode(self._stream)

    def read(self) -> Frame | None:
        if self._gen is None:
            raise RuntimeError("PyAVSource 未 open()")

        for av_frame in self._gen:
            t_mono = time.perf_counter()
            t_wall = time.time()
            try:
                img = av_frame.to_ndarray(format=self.decode_format)
            except Exception:
                continue

            self._frame_id += 1
            return Frame(
                image=img,
                frame_id=self._frame_id,
                t_recv_wall=t_wall,
                t_recv_mono=t_mono,
                pts=av_frame.pts,
            )
        return None

    def close(self) -> None:
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
        self._container = None
        self._stream = None
        self._gen = None

    @property
    def size(self) -> tuple[int, int] | None:
        return self._size

    @property
    def fps(self) -> float | None:
        return self._fps

    def to_bgr(self, img: np.ndarray) -> np.ndarray:
        """rgb24 -> bgr24（OpenCV 生态需要）。"""
        return img[..., ::-1]
