import time

import av
import numpy as np

from .base import FrameSource
from .frame import Frame


class FileSource(FrameSource):
    """离线回放录制文件。

    用途：调感知/决策时不必每次都开 A 机，迭代速度翻倍。
    realtime=True 时按原始帧率节流，模拟实时流到达节奏。
    """

    def __init__(
        self,
        path: str,
        realtime: bool = False,
        speed: float = 1.0,
        loop: bool = False,
        decode_format: str = "rgb24",
    ) -> None:
        self.path = path
        self.realtime = realtime
        self.speed = speed
        self.loop = loop
        self.decode_format = decode_format

        self._container = None
        self._stream = None
        self._frame_id = 0
        self._t0 = 0.0
        self._size = None
        self._fps = None

    def open(self) -> None:
        self._container = av.open(self.path, mode="r")
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"
        self._size = (self._stream.codec_context.width, self._stream.codec_context.height)
        try:
            self._fps = float(self._stream.average_rate)
        except Exception:
            self._fps = None
        self._reset()

    def _reset(self) -> None:
        self._container.seek(0, stream=self._stream)
        self._gen = self._container.decode(self._stream)
        self._t0 = time.perf_counter()
        self._base_wall = time.time()

    def read(self) -> Frame | None:
        if self._gen is None:
            raise RuntimeError("FileSource 未 open()")

        for av_frame in self._gen:
            if self.realtime and self._fps:
                target = self._frame_id / (self._fps * self.speed)
                while (time.perf_counter() - self._t0) < target:
                    time.sleep(0.0005)

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

        if self.loop:
            self._reset()
            return self.read()
        return None

    def close(self) -> None:
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
        self._container = None

    @property
    def size(self):
        return self._size

    @property
    def fps(self):
        return self._fps
