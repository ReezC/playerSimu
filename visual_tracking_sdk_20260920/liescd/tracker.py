"""Minimal opening-highlight lock and single-target motion filter."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from liescd.detector import Detection


class TrackState(str, Enum):
    WAITING = "WAITING"
    LOCKED = "LOCKED"
    COAST = "COAST"
    LOST = "LOST"


@dataclass(frozen=True, slots=True)
class TrackResult:
    state: TrackState
    aim_point: tuple[float, float] | None
    target_bbox: Detection | None
    observed: bool
    target_id: int | None = None
    trail_break: bool = False


class LockedTargetTracker:
    """Track one opening-highlighted rectangle and never choose a new identity."""

    def __init__(
        self,
        *,
        lock_confirmations: int = 2,
        highlight_minimum: float = 0.20,
        highlight_margin: float = 0.08,
        coast_seconds: float = 0.25,
        alpha: float = 0.72,
        beta: float = 0.18,
    ) -> None:
        self.lock_confirmations = max(2, int(lock_confirmations))
        self.highlight_minimum = float(highlight_minimum)
        self.highlight_margin = float(highlight_margin)
        self.coast_seconds = max(0.01, float(coast_seconds))
        self.alpha = min(1.0, max(0.0, float(alpha)))
        self.beta = min(1.0, max(0.0, float(beta)))
        self.reset()

    def reset(self) -> None:
        self._state = TrackState.WAITING
        self._pending: Detection | None = None
        self._pending_time: float | None = None
        self._pending_count = 0
        self._position: tuple[float, float] | None = None
        self._velocity = (0.0, 0.0)
        self._width = 0.0
        self._height = 0.0
        self._last_observed_width = 0.0
        self._last_observed_height = 0.0
        self._awaiting_split = False
        self._last_time: float | None = None
        self._last_observed_time: float | None = None

    @property
    def state(self) -> TrackState:
        return self._state

    def update(
        self,
        timestamp: float,
        detections: Sequence[Detection] | None,
    ) -> TrackResult:
        now = float(timestamp)
        if self._state is TrackState.WAITING:
            return self._update_opening(now, detections)
        assert self._position is not None
        assert self._last_time is not None

        dt = max(1e-4, now - self._last_time)
        prediction = (
            self._position[0] + self._velocity[0] * dt,
            self._position[1] + self._velocity[1] * dt,
        )

        # ``None`` means inference was intentionally skipped.  It must not be
        # start a miss by itself.  Once a real empty inference has put the
        # tracker in COAST, however, wall-clock time must keep aging the miss.
        if detections is None:
            if self._state is not TrackState.LOCKED:
                return self._missing(now, dt)
            self._position = prediction
            self._last_time = now
            return self._result(observed=False)

        candidate = self._select_candidate(prediction, detections, dt)
        if candidate is None:
            return self._missing(now, dt)

        residual_x = candidate.center[0] - prediction[0]
        residual_y = candidate.center[1] - prediction[1]
        self._position = (
            prediction[0] + self.alpha * residual_x,
            prediction[1] + self.alpha * residual_y,
        )
        self._velocity = (
            self._velocity[0] + self.beta * residual_x / dt,
            self._velocity[1] + self.beta * residual_y / dt,
        )
        self._width = 0.75 * self._width + 0.25 * candidate.width
        self._height = 0.75 * self._height + 0.25 * candidate.height
        self._last_observed_width = candidate.width
        self._last_observed_height = candidate.height
        self._awaiting_split = False
        self._last_time = now
        self._last_observed_time = now
        self._state = TrackState.LOCKED
        return TrackResult(
            state=self._state,
            aim_point=self._position,
            target_bbox=candidate,
            observed=True,
        )

    def _update_opening(
        self,
        now: float,
        detections: Sequence[Detection] | None,
    ) -> TrackResult:
        if detections is None:
            return self._result(observed=False)
        winner = self._unique_highlight(detections)
        if winner is None:
            self._pending = None
            self._pending_time = None
            self._pending_count = 0
            return self._result(observed=False)

        if self._pending is not None and self._same_opening_shape(
            self._pending, winner
        ):
            self._pending_count += 1
        else:
            self._pending = winner
            self._pending_time = now
            self._pending_count = 1

        previous = self._pending
        previous_time = self._pending_time
        self._pending = winner
        if self._pending_count < self.lock_confirmations:
            return self._result(observed=False)

        velocity = (0.0, 0.0)
        if previous is not None and previous_time is not None and now > previous_time:
            elapsed = now - previous_time
            velocity = (
                (winner.center[0] - previous.center[0]) / elapsed,
                (winner.center[1] - previous.center[1]) / elapsed,
            )
        self._state = TrackState.LOCKED
        self._position = winner.center
        self._velocity = velocity
        self._width = winner.width
        self._height = winner.height
        self._last_observed_width = winner.width
        self._last_observed_height = winner.height
        self._last_time = now
        self._last_observed_time = now
        return TrackResult(
            state=self._state,
            aim_point=self._position,
            target_bbox=winner,
            observed=True,
        )

    def _unique_highlight(self, detections: Sequence[Detection]) -> Detection | None:
        if not detections:
            return None
        ordered = sorted(detections, key=lambda item: item.highlight, reverse=True)
        winner = ordered[0]
        runner_up = ordered[1].highlight if len(ordered) > 1 else 0.0
        if winner.highlight < self.highlight_minimum:
            return None
        if winner.highlight - runner_up < self.highlight_margin:
            return None
        return winner

    @staticmethod
    def _same_opening_shape(first: Detection, second: Detection) -> bool:
        distance = math.dist(first.center, second.center)
        gate = max(20.0, first.diagonal)
        if distance > gate:
            return False
        if first.area <= 1.0 or second.area <= 1.0:
            return False
        area_ratio = second.area / first.area
        return 0.5 <= area_ratio <= 2.0

    def _select_candidate(
        self,
        prediction: tuple[float, float],
        detections: Sequence[Detection],
        dt: float,
    ) -> Detection | None:
        if not detections:
            return None
        diagonal = max(1.0, math.hypot(self._width, self._height))
        speed = math.hypot(*self._velocity)
        if self._state is TrackState.LOST:
            gate = max(12.0, diagonal * 0.16)
        else:
            # A same-sized false shape one box-width away is not a safe
            # continuation.  Allow normal velocity and reversals, but keep
            # the spatial gate tied mostly to observed motion.  The small
            # scale term accommodates resolution changes without letting a
            # large rectangle create an enormous association radius.
            gate = max(12.0, diagonal * 0.16, speed * dt * 2.0 + 6.0)

        scored: list[tuple[float, Detection]] = []
        nearby_count = 0
        previous_area = max(1.0, self._width * self._height)
        previous_aspect = max(1e-3, self._width / max(1e-3, self._height))
        for candidate in detections:
            distance = math.dist(prediction, candidate.center)
            if candidate.area <= 1.0:
                continue
            area_ratio = candidate.area / previous_area
            if not 0.45 <= area_ratio <= 2.20:
                continue
            width_ratio = candidate.width / max(1e-3, self._last_observed_width)
            height_ratio = candidate.height / max(1e-3, self._last_observed_height)
            size_change = max(
                abs(math.log(max(1e-3, width_ratio))),
                abs(math.log(max(1e-3, height_ratio))),
            )
            if distance <= gate * 1.35:
                nearby_count += 1
            if distance > gate:
                continue
            if distance > gate * 0.65 and size_change > math.log(1.15):
                self._awaiting_split = True
                continue
            aspect = candidate.width / max(1e-3, candidate.height)
            cost = (
                distance / gate
                + 0.25 * abs(math.log(area_ratio))
                + 0.10 * abs(math.log(max(1e-3, aspect / previous_aspect)))
            )
            scored.append((cost, candidate))
        if self._awaiting_split:
            if nearby_count < 2:
                return None
            self._awaiting_split = False
        if not scored:
            return None
        scored.sort(key=lambda item: item[0])
        if len(scored) > 1 and scored[1][0] - scored[0][0] < 0.16:
            return None
        return scored[0][1]

    def _missing(self, now: float, dt: float) -> TrackResult:
        assert self._position is not None
        assert self._last_time is not None
        assert self._last_observed_time is not None
        previous_age = max(0.0, self._last_time - self._last_observed_time)
        remaining = max(0.0, self.coast_seconds - previous_age)
        motion_dt = min(dt, remaining)
        if self._state is not TrackState.LOST and motion_dt > 0.0:
            self._position = (
                self._position[0] + self._velocity[0] * motion_dt,
                self._position[1] + self._velocity[1] * motion_dt,
            )
        self._last_time = now
        missing_age = now - self._last_observed_time
        if missing_age <= self.coast_seconds and self._state is not TrackState.LOST:
            self._state = TrackState.COAST
        else:
            self._state = TrackState.LOST
            self._velocity = (0.0, 0.0)
            self._awaiting_split = False
        return self._result(observed=False)

    def _result(self, *, observed: bool) -> TrackResult:
        bbox = self._predicted_bbox() if self._position is not None else None
        return TrackResult(
            state=self._state,
            aim_point=self._position,
            target_bbox=bbox,
            observed=observed,
        )

    def _predicted_bbox(self) -> Detection | None:
        if self._position is None or self._width <= 0.0 or self._height <= 0.0:
            return None
        cx, cy = self._position
        return Detection(
            x0=cx - self._width * 0.5,
            y0=cy - self._height * 0.5,
            x1=cx + self._width * 0.5,
            y1=cy + self._height * 0.5,
            confidence=0.0,
            highlight=0.0,
        )


__all__ = ["LockedTargetTracker", "TrackResult", "TrackState"]
