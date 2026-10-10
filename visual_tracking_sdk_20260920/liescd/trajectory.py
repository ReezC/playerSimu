"""Low-cost presentation-layer smoothing for the LiesCD aim point."""

from __future__ import annotations

import math
from collections import deque
from statistics import median

from liescd.detector import Detection

DEFAULT_SMOOTHING_STRENGTH = 5
VELOCITY_HISTORY_SIZE = 7

Point = tuple[float, float]


class AimTrajectorySmoother:
    """Turn noisy target centers into an acceleration-limited aim trajectory.

    This class never participates in detection, OC-SORT association or target
    identity decisions.  Strength zero is an exact bypass for A/B comparison.
    """

    def __init__(self, strength: int = DEFAULT_SMOOTHING_STRENGTH) -> None:
        self._strength = 0
        self.set_strength(strength)
        self.reset()

    @property
    def strength(self) -> int:
        return self._strength

    def set_strength(self, value: int | float) -> None:
        self._strength = min(100, max(0, int(round(float(value)))))

    def reset(self) -> None:
        self._position: Point | None = None
        self._velocity: Point = (0.0, 0.0)
        self._filtered_measurement: Point | None = None
        self._last_measurement_time: float | None = None
        self._last_time: float | None = None
        self._velocity_history: deque[Point] = deque(
            maxlen=VELOCITY_HISTORY_SIZE
        )
        self._pending_measurement: Point | None = None
        self._pending_count = 0

    def update(
        self,
        timestamp: float,
        measurement: Point | None,
        target_bbox: Detection | None = None,
    ) -> Point | None:
        if measurement is None:
            return None

        now = float(timestamp)
        raw = (float(measurement[0]), float(measurement[1]))
        if self._strength == 0:
            self._synchronize(now, raw)
            return raw

        if self._position is None or self._last_time is None:
            self._synchronize(now, raw)
            return raw

        elapsed = now - self._last_time
        if not math.isfinite(elapsed) or elapsed <= 0.0 or elapsed > 0.25:
            self._synchronize(now, raw)
            return raw
        dt = min(0.10, max(1.0 / 240.0, elapsed))

        filtered = self._accept_measurement(now, raw, target_bbox)
        target_velocity = self._robust_target_velocity()
        strength = self._strength / 100.0
        response_seconds = 0.035 + 0.185 * strength
        lookahead_seconds = 0.010 + 0.035 * strength
        desired_position = (
            filtered[0] + target_velocity[0] * lookahead_seconds,
            filtered[1] + target_velocity[1] * lookahead_seconds,
        )
        error = (
            desired_position[0] - self._position[0],
            desired_position[1] - self._position[1],
        )
        desired_velocity = (
            target_velocity[0] + error[0] / response_seconds,
            target_velocity[1] + error[1] / response_seconds,
        )

        box_scale = self._box_scale(target_bbox)
        target_speed = math.hypot(*target_velocity)
        catch_up_ratio = 2.0 - 0.45 * strength
        max_speed = max(40.0, box_scale * 1.5, target_speed * catch_up_ratio)
        desired_velocity = self._limit_magnitude(desired_velocity, max_speed)

        max_acceleration = max(600.0, max_speed / response_seconds)
        velocity_delta = (
            desired_velocity[0] - self._velocity[0],
            desired_velocity[1] - self._velocity[1],
        )
        velocity_delta = self._limit_magnitude(
            velocity_delta,
            max_acceleration * dt,
        )
        next_velocity = (
            self._velocity[0] + velocity_delta[0],
            self._velocity[1] + velocity_delta[1],
        )
        next_position = (
            self._position[0] + (self._velocity[0] + next_velocity[0]) * 0.5 * dt,
            self._position[1] + (self._velocity[1] + next_velocity[1]) * 0.5 * dt,
        )
        next_position = self._keep_inside_box(next_position, target_bbox)

        self._position = next_position
        self._velocity = next_velocity
        self._last_time = now
        return next_position

    def _synchronize(self, timestamp: float, point: Point) -> None:
        self._position = point
        self._velocity = (0.0, 0.0)
        self._filtered_measurement = point
        self._last_measurement_time = timestamp
        self._last_time = timestamp
        self._velocity_history.clear()
        self._pending_measurement = None
        self._pending_count = 0

    def _accept_measurement(
        self,
        timestamp: float,
        raw: Point,
        target_bbox: Detection | None,
    ) -> Point:
        assert self._filtered_measurement is not None
        assert self._last_measurement_time is not None

        measurement_dt = max(1.0 / 240.0, timestamp - self._last_measurement_time)
        target_velocity = self._robust_target_velocity()
        predicted = (
            self._filtered_measurement[0] + target_velocity[0] * measurement_dt,
            self._filtered_measurement[1] + target_velocity[1] * measurement_dt,
        )
        residual = math.hypot(raw[0] - predicted[0], raw[1] - predicted[1])
        gate = max(
            4.0,
            self._box_scale(target_bbox) * 0.12,
            math.hypot(*target_velocity) * measurement_dt * 1.8,
        )

        accepted = residual <= gate
        if not accepted:
            pending_distance = (
                float("inf")
                if self._pending_measurement is None
                else math.hypot(
                    raw[0] - self._pending_measurement[0],
                    raw[1] - self._pending_measurement[1],
                )
            )
            if pending_distance <= gate:
                self._pending_count += 1
            else:
                self._pending_measurement = raw
                self._pending_count = 1
            accepted = self._pending_count >= 2

        if not accepted:
            self._filtered_measurement = predicted
            self._last_measurement_time = timestamp
            return predicted

        previous = self._filtered_measurement
        sample_velocity = (
            (raw[0] - previous[0]) / measurement_dt,
            (raw[1] - previous[1]) / measurement_dt,
        )
        self._velocity_history.append(sample_velocity)
        self._filtered_measurement = raw
        self._last_measurement_time = timestamp
        self._pending_measurement = None
        self._pending_count = 0
        return raw

    def _robust_target_velocity(self) -> Point:
        if not self._velocity_history:
            return (0.0, 0.0)
        return (
            float(median(value[0] for value in self._velocity_history)),
            float(median(value[1] for value in self._velocity_history)),
        )

    @staticmethod
    def _box_scale(target_bbox: Detection | None) -> float:
        if target_bbox is None:
            return 40.0
        return max(10.0, min(target_bbox.width, target_bbox.height))

    @staticmethod
    def _limit_magnitude(vector: Point, maximum: float) -> Point:
        magnitude = math.hypot(*vector)
        if magnitude <= maximum or magnitude <= 1e-9:
            return vector
        scale = maximum / magnitude
        return vector[0] * scale, vector[1] * scale

    @staticmethod
    def _keep_inside_box(
        point: Point,
        target_bbox: Detection | None,
    ) -> Point:
        if target_bbox is None:
            return point
        return (
            min(target_bbox.x1, max(target_bbox.x0, point[0])),
            min(target_bbox.y1, max(target_bbox.y0, point[1])),
        )


__all__ = [
    "AimTrajectorySmoother",
    "DEFAULT_SMOOTHING_STRENGTH",
]
