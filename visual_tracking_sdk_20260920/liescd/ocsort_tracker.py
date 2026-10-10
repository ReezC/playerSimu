"""Original GitHub OC-SORT adapter and motion-consensus identity binding.

OC-SORT supplies short-lived identities, predictions and observed tracklets.
LiesCD binds the opening white target to one of those identities.  If an
overlap later causes an ID switch, a small semantic layer compares each local
candidate's motion with the robust majority motion of all visible geometries:
the majority-following branch is background/fake and the single motion outlier
is rebound as the real target.

The implementation is vendored from ``noahcao/OC_SORT`` at commit
``8462e7e729a93ccd3bd995c0a79a890336cb3a0b``.  This file only converts the
project's ``Detection`` objects to the original ``N x 5`` API and exposes
predicted tracks to the existing LiesCD state machine.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from liescd._vendor.ocsort.ocsort import OCSort
from liescd.detector import Detection
from liescd.tracker import TrackResult, TrackState

OCSORT_SOURCE_COMMIT = "8462e7e729a93ccd3bd995c0a79a890336cb3a0b"
DEFAULT_MOTION_TOLERANCE = 2.034259190993516
# The original OC-SORT Byte pass treats detections in (0.1, det_thresh) as
# recovery-only observations: they may update an existing track but cannot
# create a new identity.  Keep these runtime defaults in one module so the
# GUI, local controller and remote host cannot silently diverge.
OCSORT_LOW_CONFIDENCE = 0.101
OCSORT_MATCH_THRESHOLD = 0.20
MOTION_HISTORY_SIZE = 5
MOTION_CONFIRMATIONS = 3
RECOVERY_EVIDENCE_FRAMES = 15


@dataclass(frozen=True, slots=True)
class TrackedDetection:
    """One OC-SORT identity snapshot in original-frame coordinates."""

    track_id: int
    bbox: Detection
    detection_index: int | None
    observed: bool


class OCSortBackend(Protocol):
    """Small seam shared by the original adapter and test doubles."""

    def update(self, detections: Sequence[Detection]) -> list[TrackedDetection]: ...

    def predict_only(self) -> None: ...

    def get_track(self, track_id: int) -> TrackedDetection | None: ...

    def protect_track(self, track_id: int) -> None: ...

    def set_detection_threshold(self, value: float) -> None: ...

    def reset(self) -> None: ...


class GitHubOCSortBackend:
    """Adapter around the original ``noahcao/OC_SORT`` implementation.

    The original implementation expects already-scaled detections in an
    ``N x 5`` NumPy array and an image-size pair.  Our detector already returns
    coordinates in the video frame, so both image-size pairs are ``(1, 1)`` to
    make the source code's scale factor exactly one.  Association remains in
    the vendored implementation: its optional Byte pass may use weak YOLO
    detections to maintain an existing identity, but cannot create a new one.
    """

    def __init__(
        self,
        *,
        track_buffer: int = 90,
        match_threshold: float = 0.3,
        inertia: float = 0.2,
        det_threshold: float = 0.05,
        min_hits: int = 1,
        delta_t: int = 3,
        use_byte: bool = False,
    ) -> None:
        self.track_buffer = max(1, int(track_buffer))
        self.match_threshold = min(1.0, max(0.0, float(match_threshold)))
        self.inertia = min(1.0, max(0.0, float(inertia)))
        self.det_threshold = float(det_threshold)
        if not np.isfinite(self.det_threshold) or not 0.0 <= self.det_threshold <= 1.0:
            raise ValueError("det_threshold must be a finite value in [0, 1]")
        if use_byte and self.det_threshold <= 0.1:
            raise ValueError(
                "use_byte requires det_threshold > 0.1 in the original OC-SORT API"
            )
        self.min_hits = max(1, int(min_hits))
        self.delta_t = max(1, int(delta_t))
        self.use_byte = bool(use_byte)
        self._tracker = self._new_tracker()
        self._active: dict[int, TrackedDetection] = {}
        self._protected_track_id: int | None = None
        self._time_step = 1.0

    def set_time_step(self, value: float) -> None:
        self._time_step = float(value)

    def _new_tracker(self) -> OCSort:
        return OCSort(
            det_thresh=self.det_threshold,
            max_age=self.track_buffer,
            min_hits=self.min_hits,
            iou_threshold=self.match_threshold,
            delta_t=self.delta_t,
            asso_func="iou",
            inertia=self.inertia,
            use_byte=self.use_byte,
        )

    @property
    def frame_id(self) -> int:
        return int(self._tracker.frame_count)

    def set_detection_threshold(self, value: float) -> None:
        """Update the high-confidence gate without resetting track state."""

        parsed = float(value)
        if not np.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
            raise ValueError("det_threshold must be a finite value in [0, 1]")
        self.det_threshold = parsed
        self.use_byte = parsed > 0.1
        self._tracker.det_thresh = parsed
        self._tracker.use_byte = self.use_byte

    def update(self, detections: Sequence[Detection]) -> list[TrackedDetection]:
        batch = self._to_array(detections)
        rows = self._tracker.update(batch, (1, 1), (1, 1), dt=self._time_step)
        self._active = {}
        if np.asarray(rows).size == 0:
            return []

        rows_array = np.asarray(rows, dtype=np.float64).reshape(-1, 5)
        output: list[TrackedDetection] = []
        used_detection_indexes: set[int] = set()
        for row in rows_array:
            track_id = int(round(float(row[4])))
            raw_track = self._find_raw_track(track_id)
            bbox = self._raw_bbox(raw_track, row[:4])
            detection_index = self._match_detection_index(
                bbox, detections, used_detection_indexes
            )
            source = detections[detection_index] if detection_index is not None else None
            confidence = 0.0
            if raw_track is not None:
                observation = np.asarray(raw_track.last_observation).reshape(-1)
                if observation.size >= 5 and float(observation[4]) >= 0.0:
                    confidence = float(observation[4])
            item = TrackedDetection(
                track_id=track_id,
                bbox=Detection(
                    x0=float(bbox[0]),
                    y0=float(bbox[1]),
                    x1=float(bbox[2]),
                    y1=float(bbox[3]),
                    confidence=confidence,
                    highlight=source.highlight if source is not None else 0.0,
                ),
                detection_index=detection_index,
                observed=True,
            )
            output.append(item)
            self._active[item.track_id] = item
        return output

    def predict_only(self) -> None:
        """Advance motion for an intentionally skipped YOLO frame.

        The original public API has only ``update``.  This is the same
        prediction/update-none portion used by its own ``update`` method.
        Internally OC-SORT still advances ``time_since_update`` and its
        Kalman/ORU state; only the LiesCD outer state machine distinguishes
        this deliberate stride skip from a detector-confirmed empty frame.
        """

        self._tracker.frame_count += 1
        to_delete: list[int] = []
        for index, track in enumerate(self._tracker.trackers):
            position = track.predict(dt=self._time_step)[0]
            if np.any(np.isnan(position)):
                to_delete.append(index)
        for track in self._tracker.trackers:
            track.update(None)
        for index in reversed(to_delete):
            self._tracker.trackers.pop(index)
        self._prune_expired()
        self._active = {}

    def get_track(self, track_id: int) -> TrackedDetection | None:
        active = self._active.get(int(track_id))
        if active is not None:
            return active

        raw_track = self._find_raw_track(track_id)
        if raw_track is None:
            return None
        if int(raw_track.time_since_update) < 1:
            bbox = self._raw_bbox(raw_track)
        else:
            # The official tracker has already advanced its Kalman state in
            # predict()/update(None).  Returning last_observation here would
            # freeze COAST and stride-skipped output at the previous point.
            bbox = np.asarray(raw_track.get_state(), dtype=np.float64).reshape(-1)[:4]
        observation = np.asarray(raw_track.last_observation).reshape(-1)
        confidence = (
            float(observation[4])
            if observation.size >= 5 and float(observation[4]) >= 0.0
            else 0.0
        )
        return TrackedDetection(
            track_id=int(track_id),
            bbox=Detection(
                x0=float(bbox[0]),
                y0=float(bbox[1]),
                x1=float(bbox[2]),
                y1=float(bbox[3]),
                confidence=confidence,
                highlight=0.0,
            ),
            detection_index=None,
            observed=int(raw_track.time_since_update) < 1,
        )

    def protect_track(self, track_id: int) -> None:
        """Remember the selected ID for diagnostics.

        The original OC-SORT has no Ultralytics-style duplicate-pool cleanup,
        so there is no extra restoration hook to run here. Identity protection
        is enforced by ``OCSortTargetTracker``.  Its semantic target binding
        may later move to another OC-SORT ID when motion consensus proves that
        an overlap caused an ID switch.
        """

        self._protected_track_id = int(track_id)

    def reset(self) -> None:
        self._tracker = self._new_tracker()
        self._active.clear()
        self._protected_track_id = None
        self._time_step = 1.0

    @staticmethod
    def _to_array(detections: Sequence[Detection]) -> np.ndarray:
        if not detections:
            return np.empty((0, 5), dtype=np.float64)
        return np.asarray(
            [
                [item.x0, item.y0, item.x1, item.y1, item.confidence]
                for item in detections
            ],
            dtype=np.float64,
        )

    def _find_raw_track(self, track_id: int | None) -> Any | None:
        if track_id is None:
            return None
        for track in self._tracker.trackers:
            if int(track.id) + 1 == int(track_id):
                return track
        return None

    @staticmethod
    def _raw_bbox(raw_track: Any | None, fallback: Any | None = None) -> np.ndarray:
        if raw_track is not None:
            observation = np.asarray(raw_track.last_observation, dtype=np.float64)
            if observation.size >= 4 and float(observation[0]) >= 0.0:
                return observation[:4].copy()
            state = np.asarray(raw_track.get_state(), dtype=np.float64).reshape(-1)
            if state.size >= 4:
                return state[:4].copy()
        if fallback is None:
            return np.zeros(4, dtype=np.float64)
        return np.asarray(fallback, dtype=np.float64).reshape(-1)[:4].copy()

    @staticmethod
    def _match_detection_index(
        bbox: np.ndarray,
        detections: Sequence[Detection],
        used: set[int],
    ) -> int | None:
        candidates = [
            (index, item)
            for index, item in enumerate(detections)
            if index not in used
        ]
        if not candidates:
            return None
        distances = [
            float(
                np.max(
                    np.abs(
                        np.asarray(
                            [item.x0, item.y0, item.x1, item.y1], dtype=np.float64
                        )
                        - bbox
                    )
                )
            )
            for _, item in candidates
        ]
        position = int(np.argmin(distances))
        index = candidates[position][0]
        used.add(index)
        return index

    def _prune_expired(self) -> None:
        self._tracker.trackers = [
            track
            for track in self._tracker.trackers
            if track.elapsed_since_update <= self.track_buffer
        ]


def create_liescd_ocsort_backend(
    *,
    track_buffer: int,
    detection_threshold: float,
) -> GitHubOCSortBackend:
    """Build the shared production OC-SORT configuration for every runtime."""

    return GitHubOCSortBackend(
        track_buffer=track_buffer,
        match_threshold=OCSORT_MATCH_THRESHOLD,
        det_threshold=detection_threshold,
        use_byte=detection_threshold > 0.1,
    )


class OCSortTargetTracker:
    """Bind the opening target and repair overlap-induced OC-SORT ID switches."""

    def __init__(
        self,
        *,
        backend: OCSortBackend | None = None,
        lock_confirmations: int = 2,
        highlight_minimum: float = 0.20,
        highlight_margin: float = 0.08,
        coast_seconds: float = 0.25,
        motion_tolerance: float = DEFAULT_MOTION_TOLERANCE,
    ) -> None:
        self._backend = backend or GitHubOCSortBackend()
        self._detection_threshold = float(
            getattr(self._backend, "det_threshold", 0.0)
        )
        self.lock_confirmations = max(2, int(lock_confirmations))
        self.highlight_minimum = float(highlight_minimum)
        self.highlight_margin = float(highlight_margin)
        self.coast_seconds = max(0.01, float(coast_seconds))
        self.set_motion_tolerance(motion_tolerance)
        self._clear_local_state()

    def _clear_local_state(self) -> None:
        self._state = TrackState.WAITING
        self._pending_track_id: int | None = None
        self._pending_count = 0
        self._target_track_id: int | None = None
        self._last_observed_time: float | None = None
        self._last_bbox: Detection | None = None
        self._last_aim: tuple[float, float] | None = None
        self._detection_epoch = 0
        self._previous_centers: dict[int, tuple[int, tuple[float, float]]] = {}
        self._motion_residuals: dict[int, deque[float]] = {}
        self._previous_active_boxes: dict[int, Detection] = {}
        self._recovery_armed_until = -1
        self._dim_target_observations = 0
        self._identity_recovery_enabled = False
        self._previous_timestamp: float | None = None
        self._reference_interval: float | None = None
        self._motion_time = 1.0
        self._last_motion_time: float | None = None
        self._motion_step_scale = 1.0

    @property
    def state(self) -> TrackState:
        return self._state

    @property
    def target_track_id(self) -> int | None:
        return self._target_track_id

    def set_motion_tolerance(self, value: float) -> None:
        """Change the per-observation displacement tolerance in pixels."""

        parsed = float(value)
        if not np.isfinite(parsed):
            raise ValueError("motion_tolerance must be finite")
        self.motion_tolerance = min(20.0, max(0.5, parsed))

    def set_detection_threshold(self, value: float) -> None:
        """Keep OC-SORT's high/weak association split in sync with the UI."""

        setter = getattr(self._backend, "set_detection_threshold", None)
        if setter is not None:
            setter(value)
        self._detection_threshold = float(value)

    def reset(self) -> None:
        self._backend.reset()
        self._clear_local_state()

    def update(
        self,
        timestamp: float,
        detections: Sequence[Detection] | None,
    ) -> TrackResult:
        now = float(timestamp)
        if not np.isfinite(now):
            raise ValueError("source timestamp must be finite")
        step = 1.0
        if self._previous_timestamp is not None:
            elapsed = now - self._previous_timestamp
            if elapsed <= 0:
                raise ValueError("source timestamps must increase")
            if self._reference_interval is None:
                self._reference_interval = min(0.05, max(1 / 120, elapsed))
            step = elapsed / self._reference_interval
            # Suppress floating-point noise for unchanged fixed-FPS replays.
            if abs(step - 1.0) < 1e-6:
                step = 1.0
            self._motion_time += step
        self._previous_timestamp = now
        setter = getattr(self._backend, "set_time_step", None)
        if setter is not None:
            setter(step)
        if detections is None:
            self._backend.predict_only()
            return self._update_bound_target(now, (), skipped=True)

        active_tracks = self._backend.update(detections)
        self._record_motion(active_tracks)
        if self._target_track_id is None:
            return self._update_opening(now, detections, active_tracks)
        return self._update_bound_target(now, active_tracks, skipped=False)

    def _update_opening(
        self,
        now: float,
        detections: Sequence[Detection],
        active_tracks: Sequence[TrackedDetection],
    ) -> TrackResult:
        winner_index = self._unique_highlight_index(detections)
        if winner_index is None:
            self._clear_pending()
            return self._result(observed=False)

        winner_track = next(
            (
                item
                for item in active_tracks
                if item.detection_index == winner_index and item.observed
            ),
            None,
        )
        if winner_track is None:
            self._clear_pending()
            return self._result(observed=False)

        if winner_track.track_id == self._pending_track_id:
            self._pending_count += 1
        else:
            self._pending_track_id = winner_track.track_id
            self._pending_count = 1

        if self._pending_count < self.lock_confirmations:
            return self._result(observed=False)

        self._target_track_id = winner_track.track_id
        self._backend.protect_track(winner_track.track_id)
        self._state = TrackState.LOCKED
        self._last_observed_time = now
        self._last_bbox = winner_track.bbox
        self._last_aim = winner_track.bbox.center
        self._previous_active_boxes = {
            item.track_id: item.bbox for item in active_tracks if item.observed
        }
        self._dim_target_observations = 0
        self._identity_recovery_enabled = False
        return self._result(observed=True)

    def _update_bound_target(
        self,
        now: float,
        active_tracks: Sequence[TrackedDetection],
        *,
        skipped: bool,
    ) -> TrackResult:
        if self._target_track_id is None:
            return self._result(observed=False)

        target = self._backend.get_track(self._target_track_id)
        if skipped and self._state is TrackState.LOCKED:
            if target is not None:
                self._last_bbox = target.bbox
                self._last_aim = target.bbox.center
            return self._result(observed=False)

        self._update_recovery_arming(active_tracks, target)
        replacement = self._motion_rebind_candidate(active_tracks, target)
        if replacement is not None and replacement.track_id != self._target_track_id:
            resumed_after_loss = self._state is TrackState.LOST
            self._target_track_id = replacement.track_id
            self._backend.protect_track(replacement.track_id)
            self._state = TrackState.LOCKED
            self._last_observed_time = now
            self._last_bbox = replacement.bbox
            self._last_aim = replacement.bbox.center
            self._recovery_armed_until = -1
            return self._result(observed=True, trail_break=resumed_after_loss)

        if target is not None and target.observed:
            resumed_after_loss = self._state is TrackState.LOST
            self._state = TrackState.LOCKED
            self._last_observed_time = now
            self._last_bbox = target.bbox
            self._last_aim = target.bbox.center
            return self._result(
                observed=True,
                trail_break=resumed_after_loss,
            )

        missing_age = (
            float("inf")
            if self._last_observed_time is None
            else max(0.0, now - self._last_observed_time)
        )
        if target is not None and missing_age <= self.coast_seconds:
            self._state = TrackState.COAST
            self._last_bbox = target.bbox
            self._last_aim = target.bbox.center
        elif self._state is not TrackState.LOST:
            self._state = TrackState.LOST
        return self._result(observed=False)

    def _record_motion(
        self,
        active_tracks: Sequence[TrackedDetection],
    ) -> None:
        """Record motion residuals against the visible strict majority.

        Absolute coordinates are intentionally discarded.  Each vector is one
        observed center displacement since the previous YOLO observation.  A
        component-wise median is robust to the one real target being the sole
        outlier.  The user-facing tolerance absorbs YOLO rectangle jitter.
        """

        self._detection_epoch += 1
        if self._last_motion_time is not None:
            self._motion_step_scale = 1.0 / max(1e-9, self._motion_time - self._last_motion_time)
        self._last_motion_time = self._motion_time
        epoch = self._detection_epoch
        current = {
            item.track_id: item.bbox.center for item in active_tracks if item.observed
        }
        steps: dict[int, np.ndarray] = {}
        for track_id, center in current.items():
            previous = self._previous_centers.get(track_id)
            if previous is not None and previous[0] == epoch - 1:
                steps[track_id] = (np.asarray(center, dtype=np.float64) - np.asarray(
                    previous[1], dtype=np.float64
                )) * self._motion_step_scale

        consensus = self._majority_motion(steps)
        if consensus is not None:
            for track_id, step in steps.items():
                residual = float(np.linalg.norm(step - consensus))
                history = self._motion_residuals.setdefault(
                    track_id,
                    deque(maxlen=MOTION_HISTORY_SIZE),
                )
                history.append(residual)

        self._previous_centers.update(
            {track_id: (epoch, center) for track_id, center in current.items()}
        )
        stale_before = epoch - MOTION_HISTORY_SIZE * 3
        self._previous_centers = {
            track_id: sample
            for track_id, sample in self._previous_centers.items()
            if sample[0] >= stale_before
        }
        active_or_recent = set(self._previous_centers)
        self._motion_residuals = {
            track_id: history
            for track_id, history in self._motion_residuals.items()
            if track_id in active_or_recent
        }

    def _majority_motion(
        self,
        steps: dict[int, np.ndarray],
    ) -> np.ndarray | None:
        if len(steps) < 3:
            return None
        vectors = np.stack(list(steps.values()))
        center = np.median(vectors, axis=0)
        distances = np.linalg.norm(vectors - center, axis=1)
        inliers = vectors[distances <= self.motion_tolerance]
        if len(inliers) < 2 or len(inliers) <= len(vectors) / 2:
            return None
        return np.median(inliers, axis=0)

    def _motion_score(self, track_id: int) -> float | None:
        history = self._motion_residuals.get(track_id)
        if history is None or len(history) < MOTION_CONFIRMATIONS:
            return None
        recent = list(history)[-MOTION_CONFIRMATIONS:]
        return float(np.median(np.asarray(recent, dtype=np.float64)))

    def _motion_rebind_candidate(
        self,
        active_tracks: Sequence[TrackedDetection],
        target: TrackedDetection | None,
    ) -> TrackedDetection | None:
        """Return the sole locally visible motion outlier, if fully proven."""

        if (
            self._detection_epoch > self._recovery_armed_until
            or not active_tracks
            or self._last_aim is None
            or self._last_bbox is None
        ):
            return None
        anchor = target.bbox.center if target is not None else self._last_aim
        gate_radius = max(
            36.0,
            min(180.0, max(self._last_bbox.width, self._last_bbox.height) * 0.85),
        )
        local = [
            item
            for item in active_tracks
            if item.observed
            and float(
                np.hypot(
                    item.bbox.center[0] - anchor[0],
                    item.bbox.center[1] - anchor[1],
                )
            )
            <= gate_radius
        ]
        target_is_observed = target is not None and target.observed
        if not local or (target_is_observed and len(local) < 2):
            return None

        scored = [
            (item, score)
            for item in local
            if (score := self._motion_score(item.track_id)) is not None
        ]
        if not scored or (target_is_observed and len(scored) < 2):
            return None

        fake_like = [
            item for item, score in scored if score <= self.motion_tolerance
        ]
        true_threshold = self.motion_tolerance * 1.5
        outliers = [item for item, score in scored if score >= true_threshold]
        if len(outliers) != 1:
            return None

        winner = outliers[0]
        if target_is_observed:
            if not fake_like:
                return None
            target_score = self._motion_score(target.track_id)
            if target_score is None or target_score > self.motion_tolerance:
                return None
        return winner

    def _update_recovery_arming(
        self,
        active_tracks: Sequence[TrackedDetection],
        target: TrackedDetection | None,
    ) -> None:
        """Arm semantic rebinding only around a real merge/loss event.

        Motion outliers exist during normal play too, especially while the
        opening target is stationary.  They must never be allowed to trigger a
        rebind by themselves.  A short recovery window is opened only when the
        bound target is missed, its box changes like a merged box, two current
        boxes overlap, or a previously nearby box disappears into it.
        """

        current = {
            item.track_id: item.bbox for item in active_tracks if item.observed
        }
        if not self._identity_recovery_enabled:
            if (
                target is not None
                and target.observed
                and target.bbox.highlight < self.highlight_minimum
            ):
                self._dim_target_observations += 1
            else:
                self._dim_target_observations = 0
            if self._dim_target_observations >= MOTION_CONFIRMATIONS:
                self._identity_recovery_enabled = True
            self._previous_active_boxes = current
            if not self._identity_recovery_enabled:
                return

        should_arm = target is None or not target.observed
        if target is not None and target.observed:
            if self._last_bbox is not None:
                old_area = max(1.0, self._last_bbox.area)
                area_ratio = target.bbox.area / old_area
                should_arm = should_arm or area_ratio > 1.35 or area_ratio < 0.70

            for item in active_tracks:
                if item.track_id == target.track_id or not item.observed:
                    continue
                if self._bbox_iou(target.bbox, item.bbox) >= 0.03:
                    should_arm = True
                    break

            if self._last_bbox is not None:
                vanished = set(self._previous_active_boxes) - set(current)
                nearby_radius = max(
                    30.0,
                    min(
                        160.0,
                        max(self._last_bbox.width, self._last_bbox.height) * 0.70,
                    ),
                )
                for track_id in vanished:
                    if track_id == self._target_track_id:
                        continue
                    old = self._previous_active_boxes[track_id]
                    if float(
                        np.hypot(
                            old.center[0] - self._last_bbox.center[0],
                            old.center[1] - self._last_bbox.center[1],
                        )
                    ) <= nearby_radius:
                        should_arm = True
                        break

        if should_arm:
            self._recovery_armed_until = max(
                self._recovery_armed_until,
                self._detection_epoch + RECOVERY_EVIDENCE_FRAMES,
            )
        self._previous_active_boxes = current

    @staticmethod
    def _bbox_iou(first: Detection, second: Detection) -> float:
        width = max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0))
        height = max(0.0, min(first.y1, second.y1) - max(first.y0, second.y0))
        intersection = width * height
        union = first.area + second.area - intersection
        return intersection / union if union > 1e-6 else 0.0

    def _unique_highlight_index(
        self,
        detections: Sequence[Detection],
    ) -> int | None:
        if not detections:
            return None
        ordered = sorted(
            enumerate(detections),
            key=lambda pair: pair[1].highlight,
            reverse=True,
        )
        winner_index, winner = ordered[0]
        runner_up = ordered[1][1].highlight if len(ordered) > 1 else 0.0
        if winner.highlight < self.highlight_minimum:
            return None
        if winner.highlight - runner_up < self.highlight_margin:
            return None
        return winner_index

    def _clear_pending(self) -> None:
        self._pending_track_id = None
        self._pending_count = 0

    def _result(self, *, observed: bool, trail_break: bool = False) -> TrackResult:
        return TrackResult(
            state=self._state,
            aim_point=self._last_aim,
            target_bbox=self._last_bbox,
            observed=observed,
            target_id=self._target_track_id,
            trail_break=trail_break,
        )


__all__ = [
    "DEFAULT_MOTION_TOLERANCE",
    "GitHubOCSortBackend",
    "OCSORT_LOW_CONFIDENCE",
    "OCSORT_MATCH_THRESHOLD",
    "OCSortTargetTracker",
    "OCSORT_SOURCE_COMMIT",
    "OCSortBackend",
    "TrackedDetection",
    "create_liescd_ocsort_backend",
]
