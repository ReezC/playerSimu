"""Deterministic, bounded-latency frame selection for live LiesCD tracking."""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


LIESCD_CADENCE_FPS = (20.0, 15.0, 10.0)
# One in-flight frame plus one queued frame = a two-frame end-to-end window.
LIESCD_PENDING_LIMIT = 1
LIESCD_OVERFLOW_LIMIT = 3
LIESCD_TRACKING_IMAGE_SIZE = 640


def prepare_tracking_frame(
    frame: np.ndarray,
    max_dimension: int = LIESCD_TRACKING_IMAGE_SIZE,
) -> tuple[np.ndarray, dict[str, int]]:
    """Return the canonical content-only tracking image and its dimensions."""
    source_h, source_w = frame.shape[:2]
    limit = max(32, int(max_dimension))
    scale = min(1.0, limit / max(source_w, source_h))
    sent_w = max(1, int(round(source_w * scale)))
    sent_h = max(1, int(round(source_h * scale)))
    if (sent_w, sent_h) == (source_w, source_h):
        sent = frame
    else:
        sent = cv2.resize(
            frame,
            (sent_w, sent_h),
            interpolation=cv2.INTER_AREA,
        )
    return np.ascontiguousarray(sent, dtype=np.uint8), {
        "source_width": int(source_w),
        "source_height": int(source_h),
        "sent_width": int(sent_w),
        "sent_height": int(sent_h),
    }


@dataclass(frozen=True)
class CadenceSlot:
    source_timestamp_s: float
    tracking_timestamp_s: float
    slot_index: int
    target_fps: float
    missed_slots: int = 0


class FixedCadenceGate:
    """Select source frames on a time grid and downgrade only under congestion.

    The gate never manufactures or repeats an old frame.  When a source gap
    crosses several grid positions, the next real frame is assigned to the
    latest crossed slot so tracker time still reflects the missing interval.
    """

    def __init__(
        self,
        fps_tiers: tuple[float, ...] = LIESCD_CADENCE_FPS,
        *,
        overflow_limit: int = LIESCD_OVERFLOW_LIMIT,
    ):
        tiers = tuple(float(value) for value in fps_tiers if float(value) > 0.0)
        if not tiers:
            raise ValueError("fps_tiers must contain a positive value")
        self._tiers = tiers
        self._overflow_limit = max(1, int(overflow_limit))
        self.reset()

    def reset(self) -> None:
        self._tier_index = 0
        self._next_due_s: float | None = None
        self._last_source_s: float | None = None
        self._slot_index = -1
        self._overflow_streak = 0
        self._healthy_accepts = 0
        self._downgrades = 0

    @property
    def target_fps(self) -> float:
        return self._tiers[self._tier_index]

    @property
    def downgrade_count(self) -> int:
        return self._downgrades

    def select(self, source_timestamp_s: float) -> CadenceSlot | None:
        source_timestamp_s = float(source_timestamp_s)
        if not math.isfinite(source_timestamp_s):
            return None
        if (
            self._last_source_s is not None
            and source_timestamp_s <= self._last_source_s
        ):
            return None
        self._last_source_s = source_timestamp_s
        interval_s = 1.0 / self.target_fps
        if self._next_due_s is None:
            tracking_timestamp_s = source_timestamp_s
            missed_slots = 0
            self._next_due_s = tracking_timestamp_s + interval_s
        elif source_timestamp_s + 1e-9 < self._next_due_s:
            return None
        else:
            crossed = max(
                1,
                int((source_timestamp_s - self._next_due_s) // interval_s) + 1,
            )
            tracking_timestamp_s = (
                self._next_due_s + (crossed - 1) * interval_s
            )
            missed_slots = crossed - 1
            self._next_due_s += crossed * interval_s
        self._slot_index += missed_slots + 1
        return CadenceSlot(
            source_timestamp_s=source_timestamp_s,
            tracking_timestamp_s=tracking_timestamp_s,
            slot_index=self._slot_index,
            target_fps=self.target_fps,
            missed_slots=missed_slots,
        )

    def accepted(self, *, congested: bool = False) -> None:
        if congested:
            self._healthy_accepts = 0
            return
        if self._overflow_streak <= 0:
            return
        self._healthy_accepts += 1
        # A single transient full queue must not poison the whole session, but
        # intermittent accepts during sustained overload must not erase the
        # pressure either. Six clean cadence accepts are roughly 300-600 ms.
        if self._healthy_accepts >= self._overflow_limit * 2:
            self._overflow_streak = 0
            self._healthy_accepts = 0

    def overflowed(self, source_timestamp_s: float) -> bool:
        """Record one full-queue drop; return True when cadence downgraded."""
        self._healthy_accepts = 0
        self._overflow_streak += 1
        if (
            self._overflow_streak < self._overflow_limit
            or self._tier_index >= len(self._tiers) - 1
        ):
            return False
        self._tier_index += 1
        self._downgrades += 1
        self._overflow_streak = 0
        self._healthy_accepts = 0
        self._next_due_s = float(source_timestamp_s) + 1.0 / self.target_fps
        return True


__all__ = [
    "CadenceSlot",
    "FixedCadenceGate",
    "LIESCD_CADENCE_FPS",
    "LIESCD_OVERFLOW_LIMIT",
    "LIESCD_PENDING_LIMIT",
    "LIESCD_TRACKING_IMAGE_SIZE",
    "prepare_tracking_frame",
]
