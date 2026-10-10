"""YOLO rectangle detection for the independent LiesCD experiment."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from .inference_profile import configure_geometry_environment, geometry_predict_options

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PACKAGE_ROOT / "models" / "geometry.pt"
DEFAULT_IMAGE_SIZE = 640
COMMON_INFERENCE_ASPECT_RATIOS = (1.0, 5 / 4, 4 / 3, 3 / 2, 16 / 9)

# The opening target is bright and almost unsaturated.  Only the inner portion
# of a YOLO rectangle is sampled so the moving background has less influence.
WHITE_S_MAX = 55
WHITE_V_MIN = 225
INNER_FRACTION = 0.75


def warmup_frame_shapes(image_size: int) -> tuple[tuple[int, int], ...]:
    """Return representative ROI shapes for CUDA kernel warmup.

    Ultralytics pads each input to its model stride.  These ratios cover the
    square, 5:4, 4:3, 3:2 and 16:9 buckets used by recorded and live LiesCD
    regions, preventing the first real frame from paying a new-shape startup
    cost on the production geometry model.
    """

    size = max(32, int(image_size))
    shapes: list[tuple[int, int]] = []
    for aspect_ratio in COMMON_INFERENCE_ASPECT_RATIOS:
        shape = (max(32, round(size / aspect_ratio)), size)
        if shape not in shapes:
            shapes.append(shape)
    return tuple(shapes)


def full_warmup_frame_shapes(image_size: int, stride: int) -> tuple[tuple[int, int], ...]:
    """Cover every rectangular letterbox tensor bucket at batch one."""
    stride = max(1, int(stride))
    size = max(stride, ((int(image_size) + stride - 1) // stride) * stride)
    return tuple(dict.fromkeys(
        shape for edge in range(stride, size + 1, stride)
        for shape in ((edge, size), (size, edge))
    ))


@dataclass(frozen=True, slots=True)
class Detection:
    """One geometry rectangle in original-frame pixel coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float
    highlight: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) * 0.5, (self.y0 + self.y1) * 0.5)

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def diagonal(self) -> float:
        return float(np.hypot(self.width, self.height))


def highlight_score(
    frame_bgr: np.ndarray,
    box: tuple[float, float, float, float],
) -> float:
    """Return the white-pixel ratio inside the central part of ``box``.

    The recorded green cursor is deliberately left untouched.  Its saturated
    pixels do not satisfy the white predicate and therefore do not help the
    opening lock.
    """

    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame must be a BGR image")
    height, width = frame_bgr.shape[:2]
    x0, y0, x1, y1 = (float(value) for value in box)
    if x1 <= x0 or y1 <= y0:
        return 0.0

    inset_x = (x1 - x0) * (1.0 - INNER_FRACTION) * 0.5
    inset_y = (y1 - y0) * (1.0 - INNER_FRACTION) * 0.5
    left = max(0, min(width, int(np.floor(x0 + inset_x))))
    top = max(0, min(height, int(np.floor(y0 + inset_y))))
    right = max(0, min(width, int(np.ceil(x1 - inset_x))))
    bottom = max(0, min(height, int(np.ceil(y1 - inset_y))))
    if right <= left or bottom <= top:
        return 0.0

    hsv = cv2.cvtColor(frame_bgr[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] <= WHITE_S_MAX) & (hsv[:, :, 2] >= WHITE_V_MIN)
    return float(np.count_nonzero(white)) / float(white.size)


class GeometryDetector:
    """A persistent single-thread YOLO backend, shared across local sessions."""

    def __init__(
        self,
        weights: str | Path = DEFAULT_WEIGHTS,
        *,
        image_size: int = DEFAULT_IMAGE_SIZE,
    ) -> None:
        self.weights = Path(weights).resolve()
        self.image_size = int(image_size)
        self._model: Any | None = None
        self._device: int | str = "cpu"
        self._error = ""
        self._lock = threading.RLock()
        self._state = "COLD"
        self._closed = False
        self._worker_thread: threading.Thread | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="liescd-local-inference",
            initializer=self._record_worker,
        )
        self._prepare_future: Future[bool] | None = None
        self._frame_future: Future[list[Detection]] | None = None
        self._prepare_ms: float | None = None
        self._warmed_shapes = 0
        self._last_timing: dict[str, Any] = {}

    def _record_worker(self) -> None:
        self._worker_thread = threading.current_thread()

    @property
    def error(self) -> str:
        with self._lock:
            return self._error

    @property
    def device_name(self) -> str:
        return "GPU" if self._device == 0 else "CPU"

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._state == "READY" and not self._closed

    @property
    def available(self) -> bool:
        """Blocking preparation for standalone callers; GUI uses prepare_async."""
        return self.prewarm()

    def prepare_async(self, *, retry: bool = False) -> Future[bool]:
        """Share one preparation; repeated sessions never enqueue another warmup."""
        with self._lock:
            if self._closed:
                raise RuntimeError("本机视觉模型已关闭")
            if self._prepare_future is not None:
                if not (retry and self._state == "FAILED"):
                    return self._prepare_future
            self._state = "WARMING"
            self._error = ""
            self._prepare_future = self._executor.submit(self._prepare)
            return self._prepare_future

    def prewarm(self) -> bool:
        return self.prepare_async().result()

    def _prepare(self) -> bool:
        configure_geometry_environment()
        started = time.perf_counter()
        try:
            with self._lock:
                if self._closed:
                    return False
            if self._model is None:
                if not self.weights.is_file():
                    raise FileNotFoundError(f"模型不存在: {self.weights}")
                import torch
                from ultralytics import YOLO

                self._device = 0 if torch.cuda.is_available() else "cpu"
                self._model = YOLO(str(self.weights))
            stride = max(1, int(self._model.model.stride.max().item()))
            shapes = (
                full_warmup_frame_shapes(self.image_size, stride)
                if self._device == 0
                else ((self.image_size, self.image_size),)
            )
            with self._lock:
                self._warmed_shapes = 0
            for height, width in shapes:
                with self._lock:
                    if self._closed:
                        return False
                sample = np.zeros((height, width, 3), dtype=np.uint8)
                # Keep local precision and rect preprocessing unchanged.
                self._model.predict(
                    sample, **geometry_predict_options(
                        confidence=0.101, image_size=self.image_size, device=self._device),
                )
                with self._lock:
                    self._warmed_shapes += 1
            # Complete device work and highlight initialization on this thread
            # before publishing READY.
            highlight_score(np.zeros((32, 32, 3), np.uint8), (0, 0, 32, 32))
            if self._device == 0:
                import torch
                torch.cuda.synchronize(self._device)
            with self._lock:
                if self._closed:
                    return False
                self._state = "READY"
            return True
        except Exception as exc:
            with self._lock:
                self._error = f"模型准备失败: {type(exc).__name__}: {exc}"
                if not self._closed:
                    self._state = "FAILED"
            return False
        finally:
            with self._lock:
                self._prepare_ms = (time.perf_counter() - started) * 1000.0
                print(f"[liescd] 本机模型 {self._state} ({self.device_name}) "
                      f"shapes={self._warmed_shapes} prepare_ms={self._prepare_ms:.1f}"
                      + (f" error={self._error}" if self._error else ""))

    def detect(self, frame_bgr: np.ndarray, confidence: float) -> list[Detection]:
        """Process one real frame on the prepared thread, with no hidden warmup."""
        submitted = time.perf_counter()
        with self._lock:
            if self._closed or self._state != "READY":
                raise RuntimeError(self._error or f"本机视觉模型尚未就绪: {self._state}")
            if self._frame_future is not None and not self._frame_future.done():
                raise RuntimeError("本机视觉模型已有处理中的帧")
            future = self._executor.submit(self._detect, frame_bgr, confidence, submitted)
            self._frame_future = future
        return future.result()

    def _detect(
        self, frame_bgr: np.ndarray, confidence: float, submitted: float,
    ) -> list[Detection]:
        started = time.perf_counter()
        conf = min(0.99, max(0.01, float(confidence)))
        try:
            results = self._model.predict(
                frame_bgr, **geometry_predict_options(
                    confidence=conf, image_size=self.image_size, device=self._device),
            )
            predicted = time.perf_counter()
            detections: list[Detection] = []
            for result in results:
                boxes = getattr(result, "boxes", None)
                if boxes is None or boxes.xyxy is None:
                    continue
                coordinates = boxes.xyxy.detach().cpu().numpy()
                confidences = boxes.conf.detach().cpu().numpy()
                for raw_box, raw_conf in zip(coordinates, confidences, strict=False):
                    x0, y0, x1, y1 = (float(value) for value in raw_box)
                    if x1 <= x0 or y1 <= y0:
                        continue
                    detections.append(Detection(
                        x0=x0, y0=y0, x1=x1, y1=y1,
                        confidence=float(raw_conf),
                        highlight=highlight_score(frame_bgr, (x0, y0, x1, y1)),
                    ))
            finished = time.perf_counter()
            with self._lock:
                self._error = ""
                self._last_timing = {
                    "queue_ms": (started - submitted) * 1000.0,
                    "predict_ms": (predicted - started) * 1000.0,
                    "convert_ms": (finished - predicted) * 1000.0,
                    "total_ms": (finished - submitted) * 1000.0,
                    "shape": list(frame_bgr.shape[:2]),
                }
            return detections
        except Exception as exc:
            with self._lock:
                self._error = f"模型推理失败: {type(exc).__name__}: {exc}"
            raise

    def preparation_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "device": self.device_name,
                "prepare_ms": self._prepare_ms,
                "warmed_shapes": self._warmed_shapes,
                "periodic_keepalive": False,
                "thread_alive": (
                    self._worker_thread is not None and self._worker_thread.is_alive()
                ),
                "error": self._error,
                "last_timing": dict(self._last_timing),
            }

    def close(self, timeout_s: float = 1.0) -> bool:
        """Stop accepting work; finish in-flight inference without unloading it."""
        with self._lock:
            self._closed = True
            self._state = "CLOSED"
        self._executor.shutdown(wait=False, cancel_futures=True)
        thread = self._worker_thread
        if thread is None:
            # A just-submitted executor may not have entered its initializer.
            return self._prepare_future is None
        if thread is not threading.current_thread():
            thread.join(max(0.0, float(timeout_s)))
        return not thread.is_alive()


__all__ = [
    "DEFAULT_WEIGHTS", "Detection", "GeometryDetector", "highlight_score",
    "warmup_frame_shapes", "full_warmup_frame_shapes",
]
