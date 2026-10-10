"""Timestamped H.264 MP4 previews, lossless MKV, and legacy playback."""
from __future__ import annotations

from fractions import Fraction
import math
from pathlib import Path

import cv2
import numpy as np


VIDEO_TIME_BASE = Fraction(1, 1_000_000)


class LosslessTimestampVideoWriter:
    """One CPU encoder thread with source PTS; optional lossy MP4 preview."""

    def __init__(self, path: str | Path, size: tuple[int, int], *, lossy: bool = False) -> None:
        import av

        self._av = av
        self._size = size
        self._first_timestamp: float | None = None
        self._last_timestamp: float | None = None
        self._container = av.open(str(path), "w", format="mp4" if lossy else "matroska")
        try:
            self._container.metadata["mscly_recording"] = "tracking-preview-v1" if lossy else "tracking-input-v1"
            self._container.metadata["mscly_timing"] = "source-timestamps"
            self._stream = self._container.add_stream("libx264" if lossy else "libx264rgb")
            self._stream.width, self._stream.height = size
            self._stream.pix_fmt = "yuv420p" if lossy else "bgr24"
            self._stream.time_base = VIDEO_TIME_BASE
            codec = self._stream.codec_context
            codec.time_base = VIDEO_TIME_BASE
            # A nominal FPS would incorrectly describe a variable-rate source.
            codec.framerate = Fraction(0, 1)
            codec.thread_count = 1
            self._stream.options = {
                "crf": "23" if lossy else "0", "preset": "ultrafast", "tune": "zerolatency",
            }
        except Exception:
            self._container.close()
            raise

    def write(self, image: np.ndarray, timestamp_s: float) -> None:
        stamp = float(timestamp_s)
        if not math.isfinite(stamp):
            raise ValueError("录像源时间必须为有限数值")
        if self._last_timestamp is not None and stamp <= self._last_timestamp:
            raise ValueError("录像源时间必须严格递增")
        if image.shape != (self._size[1], self._size[0], 3):
            raise ValueError("录像图像尺寸或通道发生变化")
        if self._first_timestamp is None:
            self._first_timestamp = stamp
        frame = self._av.VideoFrame.from_ndarray(image, format="bgr24")
        frame.time_base = VIDEO_TIME_BASE
        frame.pts = round((stamp - self._first_timestamp) / VIDEO_TIME_BASE)
        # The next PTS determines playback timing. For the final frame only,
        # retain the last observed interval as its display-duration estimate.
        interval = (stamp - self._last_timestamp
                    if self._last_timestamp is not None else 1 / 60)
        frame.duration = max(1, round(interval / VIDEO_TIME_BASE))
        for packet in self._stream.encode(frame):
            # libx264 can omit packet duration. MP4 otherwise ends at the
            # final PTS and some decoders drop that final zero-duration frame.
            if not packet.duration:
                packet.duration = max(1, round(interval / (packet.time_base or VIDEO_TIME_BASE)))
            self._container.mux(packet)
        self._last_timestamp = stamp

    def release(self) -> None:
        try:
            for packet in self._stream.encode():
                self._container.mux(packet)
        finally:
            self._container.close()


class TimestampVideoCapture:
    """OpenCV-shaped reader that also exposes the current and next PTS."""

    has_embedded_timestamps = True

    def __init__(self, path: str | Path) -> None:
        self._container = None
        self._pending = None
        self.timestamp_s: float | None = None
        self.next_timestamp_s: float | None = None
        self.error = ""
        try:
            import av

            self._container = av.open(str(path))
            self._stream = self._container.streams.video[0]
            self._frames = iter(self._container.decode(self._stream))
        except Exception as exc:
            self.error = str(exc)
            self.release()

    def isOpened(self) -> bool:
        return self._container is not None

    def get(self, prop: int) -> float:
        if not self.isOpened():
            return 0.0
        if prop == cv2.CAP_PROP_FPS:
            return float(self._stream.average_rate or 0.0)
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self._stream.frames or 0)
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._stream.width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._stream.height)
        return 0.0

    @staticmethod
    def _timestamp(frame) -> float:
        if frame.pts is None or frame.time_base is None:
            raise ValueError("录像缺少帧时间戳")
        stamp = float(frame.pts * frame.time_base)
        if not math.isfinite(stamp):
            raise ValueError("录像帧时间戳无效")
        return stamp

    def read(self):
        if not self.isOpened():
            return False, None
        current = self._pending
        if current is None:
            current = next(self._frames, None)
        if current is None:
            return False, None
        self.timestamp_s = self._timestamp(current)
        self._pending = next(self._frames, None)
        self.next_timestamp_s = (None if self._pending is None
                                 else self._timestamp(self._pending))
        return True, current.to_ndarray(format="bgr24")

    def release(self) -> None:
        if self._container is not None:
            self._container.close()
            self._container = None
        self._pending = None


def open_video_capture(path: str | Path):
    if Path(path).suffix.lower() == ".mkv" or (Path(path).suffix.lower() == ".mp4" and Path(path).is_file()):
        return TimestampVideoCapture(path)
    return cv2.VideoCapture(str(path))
