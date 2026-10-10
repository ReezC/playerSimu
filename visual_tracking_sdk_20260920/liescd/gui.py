"""Minimal drag-and-drop video GUI for the independent LiesCD tracker."""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from liescd.detector import Detection, GeometryDetector
from liescd.experimental import (
    MotionCompass,
    MotionTemplate,
)
from liescd.ocsort_tracker import (
    DEFAULT_MOTION_TOLERANCE,
    OCSORT_LOW_CONFIDENCE,
    create_liescd_ocsort_backend,
)
from liescd.realtime_stream import prepare_tracking_frame
from liescd.runtime_tracker import LIESCD_TRACK_BUFFER, LiescdTargetTracker
from liescd.recorded_input import RecordedInput
from liescd.tracker import TrackResult, TrackState
from liescd.trajectory import AimTrajectorySmoother, DEFAULT_SMOOTHING_STRENGTH
from liescd.video_io import open_video_capture

# Keep the historical module-level name as the GUI's injection seam.  All
# runtime entry points now resolve to the same validated tracker selection.
OCSortTargetTracker = LiescdTargetTracker

VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".wmv"}
NORMAL_BOX_COLOR = (0, 220, 255)
TARGET_BOX_COLOR = (0, 0, 255)
TRAIL_COLOR = (255, 255, 0)
AIM_PROTECTION_RADIUS = 11
AIM_RING_RADIUS = 15
COMPASS_COLOR = (255, 120, 0)
COMPASS_RING_RADIUS = 20
COMPASS_ARROW_LENGTH = 34
TEMPLATE_COLOR = (220, 0, 220)
TEMPLATE_ARROW_LENGTH = 72


def _load_recorded_timestamps(video_path: str | Path) -> dict[int, float]:
    path = Path(video_path).with_suffix(".frames.jsonl")
    if not path.is_file():
        return {}
    timestamps: dict[int, float] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if 'frame' not in item:
                continue
            frame_index = int(item["frame"])
            timestamp_s = float(item["timestamp_s"])
            if frame_index >= 0 and timestamp_s >= 0.0:
                timestamps[frame_index] = timestamp_s
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return {}
    return timestamps


def _recorded_playback_interval(
    frame_index: int,
    timestamp_s: float,
    recorded_timestamps: dict[int, float],
    fps: float,
) -> float:
    """Return the real source-frame delay, falling back to the AVI rate."""
    fallback = 1.0 / max(1.0, float(fps))
    next_timestamp = recorded_timestamps.get(int(frame_index) + 1)
    if next_timestamp is None:
        return fallback
    interval = float(next_timestamp) - float(timestamp_s)
    return interval if interval > 0.0 else fallback


class _TargetTracker(Protocol):
    @property
    def state(self) -> TrackState: ...

    def update(
        self,
        timestamp: float,
        detections: Sequence[Detection] | None,
    ) -> TrackResult: ...


def render_frame(
    frame_bgr: np.ndarray,
    detections: Sequence[Detection],
    result: TrackResult,
    trail: Sequence[tuple[float, float] | None],
    show_boxes: bool,
    motion_compasses: Sequence[MotionCompass] = (),
    motion_template: MotionTemplate | None = None,
    show_ids: bool = False,
    id_labels: Sequence[tuple[int, Detection, bool]] = (),
    show_directions: bool = False,
) -> np.ndarray:
    """Render boxes optionally; the aim trail is always rendered."""

    output = frame_bgr.copy()
    if show_boxes:
        for item in detections:
            cv2.rectangle(
                output,
                (int(round(item.x0)), int(round(item.y0))),
                (int(round(item.x1)), int(round(item.y1))),
                NORMAL_BOX_COLOR,
                2,
            )
        target = result.target_bbox
        if target is not None:
            cv2.rectangle(
                output,
                (int(round(target.x0)), int(round(target.y0))),
                (int(round(target.x1)), int(round(target.y1))),
                TARGET_BOX_COLOR,
                4,
            )

    for compass in (motion_compasses if show_directions else ()):
        center = tuple(int(round(value)) for value in compass.center)
        cv2.circle(
            output,
            center,
            COMPASS_RING_RADIUS,
            COMPASS_COLOR,
            1,
            cv2.LINE_AA,
        )
        if compass.vector is None:
            continue
        vector = np.asarray(compass.vector, dtype=np.float64)
        magnitude = float(np.linalg.norm(vector))
        if magnitude <= 0.0 or not np.isfinite(magnitude):
            continue
        endpoint = tuple(
            int(
                round(
                    center[index]
                    + vector[index] / magnitude * COMPASS_ARROW_LENGTH
                )
            )
            for index in range(2)
        )
        cv2.arrowedLine(
            output,
            center,
            endpoint,
            COMPASS_COLOR,
            2,
            cv2.LINE_AA,
            tipLength=0.28,
        )

    if (
        show_directions
        and motion_template is not None
        and motion_template.ready
        and motion_template.vector is not None
    ):
        template_vector = np.asarray(motion_template.vector, dtype=np.float64)
        template_magnitude = float(np.linalg.norm(template_vector))
        if template_magnitude > 0.0 and np.isfinite(template_magnitude):
            height, width = output.shape[:2]
            center = (width // 2, height // 2)
            endpoint = tuple(
                int(
                    round(
                        center[index]
                        + template_vector[index]
                        / template_magnitude
                        * TEMPLATE_ARROW_LENGTH
                    )
                )
                for index in range(2)
            )
            cv2.arrowedLine(
                output,
                center,
                endpoint,
                TEMPLATE_COLOR,
                4,
                cv2.LINE_AA,
                tipLength=0.25,
            )

    segment: list[tuple[int, int]] = []
    for point in (*trail, None):
        if point is not None:
            segment.append(tuple(int(round(value)) for value in point))
            continue
        if len(segment) >= 2:
            points = np.asarray(segment, dtype=np.int32)
            cv2.polylines(output, [points], False, TRAIL_COLOR, 2, cv2.LINE_AA)
        segment = []
    if result.aim_point is not None:
        x, y = (int(round(value)) for value in result.aim_point)
        height, width = output.shape[:2]
        left = max(0, x - AIM_PROTECTION_RADIUS)
        top = max(0, y - AIM_PROTECTION_RADIUS)
        right = min(width, x + AIM_PROTECTION_RADIUS + 1)
        bottom = min(height, y + AIM_PROTECTION_RADIUS + 1)
        output[top:bottom, left:right] = frame_bgr[top:bottom, left:right]

        cv2.circle(output, (x, y), AIM_RING_RADIUS, TRAIL_COLOR, 2, cv2.LINE_AA)
        tick_inner = AIM_RING_RADIUS + 2
        tick_outer = AIM_RING_RADIUS + 7
        cv2.line(output, (x - tick_outer, y), (x - tick_inner, y), TRAIL_COLOR, 2)
        cv2.line(output, (x + tick_inner, y), (x + tick_outer, y), TRAIL_COLOR, 2)
        cv2.line(output, (x, y - tick_outer), (x, y - tick_inner), TRAIL_COLOR, 2)
        cv2.line(output, (x, y + tick_inner), (x, y + tick_outer), TRAIL_COLOR, 2)
    if show_ids:
        for ident, box, observed in id_labels:
            target = ident == result.target_id
            text = f'{"TARGET " if target else ""}ID:{ident}{"" if observed else " (P)"}'
            color = TARGET_BOX_COLOR if target else NORMAL_BOX_COLOR
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .5, 1)
            h, w = output.shape[:2]
            x = max(0, min(int(round(box.x0)), max(0, w - tw - 3)))
            y = max(th + 3, min(int(round(box.y0)) - 5, h - baseline - 2))
            cv2.rectangle(output, (x, y-th-2), (x+tw+2, y+baseline+1), (0, 0, 0), -1)
            cv2.putText(output, text, (x+1, y), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
    return output


class VideoWorker(QThread):
    frame_ready = Signal(object, object)
    done = Signal(object)

    def __init__(
        self,
        video_path: str,
        *,
        confidence: float,
        stride: int,
        show_boxes: bool = True,
        show_ids: bool = False,
        show_directions: bool = False,
        motion_tolerance: float = DEFAULT_MOTION_TOLERANCE,
        smoothing_strength: int = DEFAULT_SMOOTHING_STRENGTH,
        detector: GeometryDetector | None = None,
        tracker_factory: Callable[[], _TargetTracker] | None = None,
        recorded_settings: bool = True,
    ) -> None:
        super().__init__()
        self.video_path = str(video_path)
        self.recorded_settings = bool(recorded_settings)
        self.stride = max(1, int(stride))
        self.detector = detector
        self._tracker_factory = tracker_factory
        self._confidence = float(confidence)
        self._show_boxes = bool(show_boxes)
        self._show_ids = bool(show_ids)
        self._show_directions = bool(show_directions)
        self._motion_tolerance = float(motion_tolerance)
        self._smoothing_strength = int(smoothing_strength)
        self._settings_lock = threading.Lock()
        self._resume_event = threading.Event()
        self._resume_event.set()
        self._stop_event = threading.Event()

    @property
    def is_paused(self) -> bool:
        return not self._resume_event.is_set()

    def set_confidence(self, value: float) -> None:
        with self._settings_lock:
            self._confidence = float(value)

    def set_show_boxes(self, value: bool) -> None:
        with self._settings_lock:
            self._show_boxes = bool(value)

    def set_motion_tolerance(self, value: float) -> None:
        with self._settings_lock:
            self._motion_tolerance = float(value)

    def set_show_ids(self, value: bool) -> None:
        with self._settings_lock:
            self._show_ids = bool(value)

    def set_smoothing_strength(self, value: int) -> None:
        with self._settings_lock:
            self._smoothing_strength = int(value)

    def set_show_directions(self, value: bool) -> None:
        with self._settings_lock:
            self._show_directions = bool(value)

    def pause(self) -> None:
        self._resume_event.clear()

    def resume(self) -> None:
        self._resume_event.set()

    def stop(self) -> None:
        self._stop_event.set()
        self._resume_event.set()

    def _settings(self) -> tuple[float, bool, float, int]:
        with self._settings_lock:
            return (
                self._confidence,
                self._show_boxes,
                self._motion_tolerance,
                self._smoothing_strength,
            )

    def run(self) -> None:
        capture = open_video_capture(self.video_path)
        if not capture.isOpened():
            self.done.emit({"error": f"视频打开失败: {self.video_path}"})
            return

        detector = self.detector or GeometryDetector()
        self.detector = detector
        if not detector.available:
            capture.release()
            self.done.emit({"error": detector.error or "YOLO模型不可用"})
            return

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        fps = fps if fps > 0.0 else 30.0
        recorded_timestamps = _load_recorded_timestamps(self.video_path)
        try:
            recorded_input = RecordedInput(self.video_path)
            if self.recorded_settings:
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                # Matroska usually has no frame count in its header. Validate
                # metadata now and decoded count again at the end of the run.
                if frame_count <= 0 and getattr(capture, "has_embedded_timestamps", False):
                    frame_count = len(recorded_input.frames)
                recorded_input.validate_complete(frame_count)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            capture.release()
            self.done.emit({"error": str(exc)})
            return
        try:
            if self._tracker_factory is None:
                tracker = OCSortTargetTracker(
                    backend=create_liescd_ocsort_backend(
                        track_buffer=LIESCD_TRACK_BUFFER,
                        detection_threshold=self._confidence,
                    ),
                    motion_tolerance=self._motion_tolerance,
                )
            else:
                tracker = self._tracker_factory()
        except Exception as exc:
            capture.release()
            self.done.emit({"error": f"OC-SORT初始化失败: {exc}"})
            return
        frame_index = 0
        rendered = 0
        trail: list[tuple[float, float] | None] = []
        started = time.perf_counter()
        next_frame_at = started
        applied_motion_tolerance: float | None = None
        applied_confidence: float | None = None
        trajectory = AimTrajectorySmoother(self._smoothing_strength)
        applied_smoothing_strength: int | None = None
        parity_frames = parity_mismatches = 0
        parity_max_distance = 0.0

        try:
            while not self._stop_event.is_set():
                if not self._resume_event.wait(0.05):
                    continue
                if self._stop_event.is_set():
                    break
                ok, frame = capture.read()
                if not ok:
                    break
                frame = recorded_input.image(frame_index, frame)
                frame, _tracking_dims = prepare_tracking_frame(frame)

                embedded_timestamp = getattr(capture, "timestamp_s", None)
                timestamp = recorded_timestamps.get(
                    frame_index,
                    embedded_timestamp if embedded_timestamp is not None else frame_index / fps,
                )
                (
                    confidence,
                    show_boxes,
                    motion_tolerance,
                    smoothing_strength,
                ) = self._settings()
                recorded_row = recorded_input.frames.get(frame_index, {})
                saved = recorded_row.get('settings') if self.recorded_settings else None
                if saved:
                    confidence = float(saved['confidence'])
                    motion_tolerance = float(saved['motion_tolerance'])
                    smoothing_strength = int(saved['smoothing_strength'])
                if motion_tolerance != applied_motion_tolerance:
                    set_tolerance = getattr(tracker, "set_motion_tolerance", None)
                    if set_tolerance is not None:
                        set_tolerance(motion_tolerance)
                    applied_motion_tolerance = motion_tolerance
                if confidence != applied_confidence:
                    set_threshold = getattr(
                        tracker,
                        "set_detection_threshold",
                        None,
                    )
                    if set_threshold is not None:
                        set_threshold(confidence)
                    applied_confidence = confidence
                if smoothing_strength != applied_smoothing_strength:
                    trajectory.set_strength(smoothing_strength)
                    applied_smoothing_strength = smoothing_strength
                set_frame_size = getattr(tracker, "set_frame_size", None)
                if set_frame_size is not None:
                    set_frame_size(frame.shape[1], frame.shape[0])
                run_yolo = (bool(recorded_row['run_yolo']) if saved and 'run_yolo' in recorded_row
                            else frame_index % self.stride == 0)
                if run_yolo:
                    tracked_detections = detector.detect(
                        frame,
                        min(confidence, OCSORT_LOW_CONFIDENCE),
                    )
                    detections = [
                        item
                        for item in tracked_detections
                        if item.confidence >= confidence
                    ]
                    update_detections: Sequence[Detection] | None = (
                        tracked_detections
                    )
                else:
                    detections = []
                    update_detections = None
                result = tracker.update(timestamp, update_detections)
                motion_compasses = (
                    tuple(getattr(tracker, "motion_compasses", ()))
                    if update_detections is not None
                    else ()
                )
                motion_template = (
                    getattr(tracker, "motion_template", None)
                    if update_detections is not None
                    else None
                )
                smoothed = trajectory.update(timestamp, result.aim_point, result.target_bbox)
                comparison = recorded_input.compare(frame_index, result, smoothed) if saved else None
                if comparison is not None:
                    different, distance = comparison
                    parity_frames += 1
                    parity_mismatches += int(different)
                    parity_max_distance = max(parity_max_distance, distance)
                result = replace(result, aim_point=smoothed)
                if result.aim_point is not None:
                    if result.trail_break and trail and trail[-1] is not None:
                        trail.append(None)
                    if not trail or result.aim_point != trail[-1]:
                        trail.append(result.aim_point)

                with self._settings_lock:
                    show_ids = self._show_ids
                    show_directions = self._show_directions
                labels = {}
                if show_ids:
                    for item in getattr(tracker, '_current_active_tracks', ()):
                        labels[item.track_id] = (item.track_id, item.bbox, item.observed)
                    if result.target_id is not None and result.target_bbox is not None:
                        labels[result.target_id] = (result.target_id, result.target_bbox, result.observed)
                visual = render_frame(
                    frame,
                    detections,
                    result,
                    trail,
                    show_boxes,
                    motion_compasses,
                    motion_template,
                    show_ids=show_ids,
                    show_directions=show_directions,
                    id_labels=tuple(labels.values()),
                )
                elapsed = max(1e-6, time.perf_counter() - started)
                self.frame_ready.emit(
                    visual,
                    {
                        "frame": frame_index,
                        "state": result.state.value,
                        "detections": len(detections),
                        "fps": (rendered + 1) / elapsed,
                        "device": detector.device_name,
                        "target_id": result.target_id,
                        "parity_mismatches": parity_mismatches,
                        "motion_compasses": motion_compasses,
                        "motion_template": motion_template,
                    },
                )
                rendered += 1
                playback_interval = _recorded_playback_interval(
                    frame_index,
                    timestamp,
                    recorded_timestamps,
                    fps,
                )
                next_timestamp = getattr(capture, "next_timestamp_s", None)
                if (frame_index + 1 not in recorded_timestamps
                        and embedded_timestamp is not None and next_timestamp is not None):
                    playback_interval = max(0.0, next_timestamp - embedded_timestamp)
                frame_index += 1

                next_frame_at += playback_interval
                wait_seconds = next_frame_at - time.perf_counter()
                if wait_seconds > 0.0:
                    time.sleep(wait_seconds)
                else:
                    next_frame_at = time.perf_counter()
            if (self.recorded_settings and recorded_input.exact and not self._stop_event.is_set()
                    and rendered != len(recorded_input.frames)):
                raise ValueError("录像解码未读完全部追踪输入，无法验证原始路径")
        except Exception as exc:
            self.done.emit({"error": f"运行失败: {exc}", "frames": rendered})
            return
        finally:
            capture.release()

        self.done.emit(
            {
                "parity_frames": parity_frames,
                "parity_mismatches": parity_mismatches,
                "parity_max_distance": parity_max_distance,
                "frames": rendered,
                "stopped": self._stop_event.is_set(),
                "state": tracker.state.value,
            }
        )


class VideoPane(QLabel):
    video_dropped = Signal(str)

    def __init__(self) -> None:
        super().__init__("把测试视频拖到这里")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(720, 480)
        self.setAcceptDrops(True)
        self.setStyleSheet("background:#17191d;color:#aab0b8;border:1px solid #3a3f46;")
        self._frame: np.ndarray | None = None
        self.setObjectName("videoPane")
        self.setAccessibleName("videoPane")

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt API
        urls = event.mimeData().urls()
        if any(
            Path(url.toLocalFile()).suffix.lower() in VIDEO_SUFFIXES for url in urls
        ):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt API
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                self.video_dropped.emit(str(path))
                event.acceptProposedAction()
                return

    def show_frame(self, frame_bgr: np.ndarray) -> None:
        self._frame = frame_bgr.copy()
        self._refresh()

    def _refresh(self) -> None:
        if self._frame is None:
            return
        rgb = cv2.cvtColor(self._frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        image = QImage(
            rgb.data, width, height, width * 3, QImage.Format.Format_RGB888
        ).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pixmap)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._refresh()


def _identify(widget: QWidget, name: str) -> None:
    widget.setObjectName(name)
    widget.setAccessibleName(name)


class MainWindow(QWidget):
    """Drag a video in, then test the new single-target tracker."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("视觉追踪测试台（V2.5.1 / V4模型）")
        self.resize(1120, 780)
        self.video_path: str | None = None
        self.first_frame: np.ndarray | None = None
        self.worker: VideoWorker | None = None
        self._detector: GeometryDetector | None = None
        self._gui_paused = False
        self._last_rendered_frame = -1
        self._resume_after_frame: int | None = None

        self.video_pane = VideoPane()
        self.video_pane.video_dropped.connect(self.load_video)

        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(5, 95)
        self.conf_slider.setValue(15)
        self.conf_slider.setFixedWidth(240)
        _identify(self.conf_slider, "confSlider")
        self.conf_value = QLabel("0.15")
        self.conf_slider.valueChanged.connect(self._confidence_changed)

        self.stride_spin = QSpinBox()
        self.stride_spin.setRange(1, 10)
        self.stride_spin.setValue(1)
        _identify(self.stride_spin, "strideSpin")

        from liescd.widgets import MotionToleranceSpinBox
        self.motion_tolerance_spin = MotionToleranceSpinBox()
        self.motion_tolerance_spin.setRange(0.5, 20.0)
        self.motion_tolerance_spin.setSingleStep(0.5)
        self.motion_tolerance_spin.setDecimals(15)
        self.motion_tolerance_spin.setSuffix(" px")
        self.motion_tolerance_spin.setValue(DEFAULT_MOTION_TOLERANCE)
        self.motion_tolerance_spin.setToolTip(
            "干扰目标单次位移相对多数运动允许的YOLO抖动误差"
        )
        _identify(self.motion_tolerance_spin, "motionToleranceSpin")
        self.motion_tolerance_spin.valueChanged.connect(
            self._motion_tolerance_changed
        )

        self.smoothing_spin = QSpinBox()
        self.smoothing_spin.setRange(0, 100)
        self.smoothing_spin.setValue(DEFAULT_SMOOTHING_STRENGTH)
        self.smoothing_spin.setSuffix(" %")
        self.smoothing_spin.setToolTip("0为原始轨迹；数值越高越平滑")
        _identify(self.smoothing_spin, "smoothingSpin")
        self.smoothing_spin.valueChanged.connect(self._smoothing_changed)

        self.show_boxes_check = QCheckBox("显示矩形框")
        self.show_boxes_check.setChecked(True)
        _identify(self.show_boxes_check, "showBoxesCheck")
        self.show_boxes_check.toggled.connect(self._show_boxes_changed)
        self.show_ids_check = QCheckBox("显示ID")
        _identify(self.show_ids_check, "showIdsCheck")
        self.show_ids_check.setToolTip("显示实际跟踪ID；TARGET为主目标，(P)为预测框。播放时可切换")
        self.show_ids_check.toggled.connect(self._show_ids_changed)
        self.show_directions_check = QCheckBox("显示方向箭头")
        _identify(self.show_directions_check, "showDirectionsCheck")
        self.show_directions_check.setToolTip("显示各图形及中央整体运动方向；仅影响绘图")
        self.show_directions_check.toggled.connect(self._show_directions_changed)
        self.recorded_settings_check = QCheckBox("按录像参数回放")
        self._has_recorded_input = False
        self.recorded_settings_check.setToolTip("新录像逐帧校验原始像素、时间、参数及接收端结果；取消后可手动调整参数")
        self.recorded_settings_check.setEnabled(False)
        self.recorded_settings_check.toggled.connect(self._recorded_settings_changed)

        self.pause_button = QPushButton("暂停")
        self.pause_button.setEnabled(False)
        _identify(self.pause_button, "pauseButton")
        self.pause_button.clicked.connect(self._toggle_pause)

        self.start_button = QPushButton("开始")
        _identify(self.start_button, "startButton")
        self.start_button.clicked.connect(self._toggle_start)

        self.status_label = QLabel("把测试视频拖入窗口")
        _identify(self.status_label, "statusLabel")

        controls = QHBoxLayout()
        controls.addWidget(QLabel("匹配度"))
        controls.addWidget(self.conf_slider)
        controls.addWidget(self.conf_value)
        controls.addSpacing(16)
        controls.addWidget(QLabel("步长"))
        controls.addWidget(self.stride_spin)
        controls.addSpacing(16)
        controls.addWidget(QLabel("运动容差"))
        controls.addWidget(self.motion_tolerance_spin)
        controls.addSpacing(16)
        controls.addWidget(QLabel("轨迹平滑"))
        controls.addWidget(self.smoothing_spin)
        controls.addSpacing(16)
        controls.addWidget(self.show_boxes_check)
        controls.addWidget(self.show_ids_check)
        controls.addWidget(self.show_directions_check)
        controls.addWidget(self.recorded_settings_check)
        controls.addStretch(1)
        controls.addWidget(self.pause_button)
        controls.addWidget(self.start_button)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.video_pane, 1)
        layout.addWidget(self.status_label)

    def load_video(self, path: str) -> None:
        if self.worker is not None:
            self.status_label.setText("正在运行，停止后才能更换视频")
            return
        capture = open_video_capture(path)
        if not capture.isOpened():
            self.status_label.setText(f"视频打开失败: {path}")
            return
        ok, frame = capture.read()
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        variable_rate = getattr(capture, "has_embedded_timestamps", False)
        capture.release()
        if not ok:
            self.status_label.setText(f"视频首帧读取失败: {path}")
            return
        self.video_path = str(Path(path).resolve())
        try:
            recorded = RecordedInput(path)
            if count <= 0 and recorded.exact:
                count = len(recorded.frames)
            self._has_recorded_input = recorded.exact
            self.recorded_settings_check.setEnabled(recorded.exact)
            self.recorded_settings_check.setChecked(False)
            self._recorded_first_settings = recorded.frames.get(0, {}).get('settings', {})
            frame = recorded.image(0, frame)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.status_label.setText(f"读取录像参数失败: {exc}")
            return
        self.first_frame = frame
        self.video_pane.show_frame(frame)
        height, width = frame.shape[:2]
        rate_label = "可变帧率（真实时间）" if variable_rate else f"{fps:.1f}fps"
        count_label = f"{count}帧" if count > 0 else "帧数待解码"
        self.status_label.setText(
            f"{Path(path).name}  {width}×{height}  {rate_label}  {count_label}"
        )

    def _toggle_start(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.start_button.setEnabled(False)
            self.start_button.setText("停止中…")
            self.pause_button.setEnabled(False)
            return
        if not self.video_path:
            self.status_label.setText("请先把测试视频拖入窗口")
            return
        if self.first_frame is not None:
            self.video_pane.show_frame(self.first_frame)

        self.worker = VideoWorker(
            self.video_path,
            confidence=self.conf_slider.value() / 100.0,
            stride=self.stride_spin.value(),
            show_boxes=self.show_boxes_check.isChecked(),
            show_ids=self.show_ids_check.isChecked(),
            show_directions=self.show_directions_check.isChecked(),
            motion_tolerance=self.motion_tolerance_spin.value(),
            smoothing_strength=self.smoothing_spin.value(),
            detector=self._detector,
            recorded_settings=self.recorded_settings_check.isChecked(),
        )
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.done.connect(self._on_done)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()
        self._gui_paused = False
        self._last_rendered_frame = -1
        self._resume_after_frame = None
        self.start_button.setText("停止")
        self.pause_button.setText("暂停")
        self.pause_button.setEnabled(True)
        self.stride_spin.setEnabled(False)
        self.recorded_settings_check.setEnabled(False)
        self.status_label.setText("正在加载模型并开始检测…")

    def _toggle_pause(self) -> None:
        if self.worker is None:
            return
        if self._gui_paused:
            self.worker.resume()
            self._gui_paused = False
            self._resume_after_frame = self._last_rendered_frame
            self.pause_button.setText("暂停")
            self.status_label.setText("继续播放…")
        else:
            self.worker.pause()
            self._gui_paused = True
            self.pause_button.setText("继续")
            self.status_label.setText("已暂停，当前框和轨迹保持不变")

    def _confidence_changed(self, value: int) -> None:
        self.conf_value.setText(f"{value / 100:.2f}")
        if self.worker is not None:
            self.worker.set_confidence(value / 100.0)

    def _recorded_settings_changed(self, checked: bool) -> None:
        widgets = (self.conf_slider, self.stride_spin, self.motion_tolerance_spin, self.smoothing_spin)
        if checked:
            if getattr(self, '_manual_settings', None) is None:
                self._manual_settings = tuple(widget.value() for widget in widgets)
            first = getattr(self, '_recorded_first_settings', {})
            if first:
                for widget, value in zip(widgets, (round(float(first['confidence']) * 100),
                        int(first['stride']), float(first['motion_tolerance']), int(first['smoothing_strength']))):
                    widget.setValue(value)
        elif getattr(self, '_manual_settings', None) is not None:
            for widget, value in zip(widgets, self._manual_settings):
                widget.setValue(value)
            self._manual_settings = None
        for widget in (self.conf_slider, self.motion_tolerance_spin, self.smoothing_spin):
            widget.setEnabled(not checked)
        self.stride_spin.setEnabled(not checked and self.worker is None)

    def _show_boxes_changed(self, value: bool) -> None:
        if self.worker is not None:
            self.worker.set_show_boxes(value)

    def _show_directions_changed(self, value: bool) -> None:
        if self.worker is not None:
            self.worker.set_show_directions(value)

    def _show_ids_changed(self, value: bool) -> None:
        if self.worker is not None:
            self.worker.set_show_ids(value)

    def _motion_tolerance_changed(self, value: float) -> None:
        if self.worker is not None:
            self.worker.set_motion_tolerance(value)

    def _smoothing_changed(self, value: int) -> None:
        if self.worker is not None:
            self.worker.set_smoothing_strength(value)

    def _on_frame(self, frame: np.ndarray, status: dict) -> None:
        frame_index = int(status["frame"])
        if self._gui_paused:
            return
        if (
            self._resume_after_frame is not None
            and frame_index <= self._resume_after_frame
        ):
            return
        self._resume_after_frame = None
        self._last_rendered_frame = frame_index
        self.video_pane.show_frame(frame)
        self.status_label.setText(
            f"帧 {frame_index}  图形 {status['detections']}  "
            f"{status['state']}  目标ID {status.get('target_id') or '-'}  "
            f"{status['device']}  处理 {status['fps']:.1f}fps"
        )

    def _on_done(self, summary: dict) -> None:
        if error := summary.get("error"):
            self.status_label.setText(str(error))
            return
        suffix = "（手动停止）" if summary.get("stopped") else ""
        if summary.get('parity_frames'):
            suffix += (f"；原始结果校验 {summary['parity_frames']}帧，"
                       f"差异 {summary['parity_mismatches']}帧，"
                       f"最大坐标差 {summary['parity_max_distance']:.6f}px")
        self.status_label.setText(
            f"播放结束{suffix}：{summary.get('frames', 0)}帧，"
            f"最终状态 {summary.get('state', 'WAITING')}"
        )

    def _on_finished(self) -> None:
        worker = self.worker
        if worker is None:
            return
        worker.wait()
        self._detector = worker.detector
        self.worker = None
        self._gui_paused = False
        self._resume_after_frame = None
        self.pause_button.setText("暂停")
        self.pause_button.setEnabled(False)
        self.start_button.setText("开始")
        self.start_button.setEnabled(True)
        self.recorded_settings_check.setEnabled(self._has_recorded_input)
        self._recorded_settings_changed(self.recorded_settings_check.isChecked())

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait()
        detector = self.worker.detector if self.worker is not None else self._detector
        if detector is not None:
            detector.close()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    if len(sys.argv) > 1 and Path(sys.argv[1]).is_file():
        window.load_video(sys.argv[1])
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
