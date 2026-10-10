"""Local identity recovery when an ROI boundary truncates a collision branch.

An OC-SORT ID may survive an exchange of physical objects.  This auditor uses
short observed trajectories in the moving coordinate system of the majority,
not the ID's continued existence, as evidence for a local correction.  Bounds
are reconstructed only for motion measurement; output always uses a real box.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
import math
from statistics import median

import numpy as np

from liescd.detector import Detection
from liescd.ocsort_tracker import TrackedDetection


HISTORY_SECONDS = .30
HISTORY_GAP_SECONDS = .20
MIN_HISTORY_SECONDS = .12
EDGE_EPISODE_SECONDS = 2.
EDGE_MARGIN = 2.
MIN_CONFIDENCE = .20
MIN_VISIBLE_FRACTION = .40
MAX_EXTENT_RATIO = 1.60
GAP_JUMP_SIZE_RATIO = .20
GAP_JUMP_MINIMUM_PX = 6.
GAP_JUMP_SUPPORT_STEPS = 3


@dataclass(frozen=True, slots=True)
class EdgeIdentityDecision:
    candidate: TrackedDetection | None = None
    pending: bool = False


class EdgeIdentityAudit:
    """Bounded, per-session motion evidence; never receives pixels or truth."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Discard all motion and authorization evidence for a new geometry."""
        self._frame_size: tuple[int, int] | None = None
        self._previous: dict[int, Detection] = {}
        self._histories: dict[int, deque[tuple[float, np.ndarray]]] = {}
        self._gap_jumps: dict[int, float] = {}
        self._shift = np.zeros(2, dtype=np.float64)
        self._size: np.ndarray | None = None
        self._scores: dict[int, float] = {}
        self._previous_time: float | None = None
        self._observation_epoch = -1
        self._edge_until = float('-inf')
        self._candidate_id: int | None = None
        self._candidate_count = 0
        self._candidate_since = 0.
        self._candidate_epoch = -1
        self._decision_epoch = -1
        self._last_decision = EdgeIdentityDecision()

    def _full(self, box: Detection) -> bool:
        if self._frame_size is None:
            return False
        width, height = self._frame_size
        return (box.x0 > EDGE_MARGIN and box.y0 > EDGE_MARGIN
                and box.x1 < width - EDGE_MARGIN and box.y1 < height - EDGE_MARGIN)

    def _clear_confirmation(self) -> None:
        self._candidate_id = None
        self._candidate_count = 0
        self._candidate_epoch = -1

    def _motion_center(self, box: Detection) -> np.ndarray:
        center = np.asarray(box.center, dtype=np.float64)
        if self._size is None or self._frame_size is None:
            return center
        width, height = self._frame_size
        # The visible, opposite edge is measured.  Assuming the common object
        # size avoids interpreting the changing cropped extent as half-speed
        # motion.  This inferred center is never used as the public aim point.
        if box.x0 <= EDGE_MARGIN and box.x1 < width - EDGE_MARGIN:
            center[0] = box.x1 - self._size[0] * .5
        elif box.x1 >= width - EDGE_MARGIN and box.x0 > EDGE_MARGIN:
            center[0] = box.x0 + self._size[0] * .5
        if box.y0 <= EDGE_MARGIN and box.y1 < height - EDGE_MARGIN:
            center[1] = box.y1 - self._size[1] * .5
        elif box.y1 >= height - EDGE_MARGIN and box.y0 > EDGE_MARGIN:
            center[1] = box.y0 + self._size[1] * .5
        return center

    def observe(
        self, now: float, epoch: int, tracks: Sequence[TrackedDetection], *,
        frame_size: tuple[int, int] | None, target_id: int | None,
        motion_tolerance: float,
    ) -> None:
        """Consume an independent detector observation, never a poll/prediction."""
        if frame_size != self._frame_size:
            self.reset()
            self._frame_size = frame_size
        if frame_size is None or epoch <= self._observation_epoch:
            return
        self._observation_epoch = epoch
        elapsed = 0. if self._previous_time is None else now - self._previous_time
        self._previous_time = now
        if elapsed < 0 or elapsed > HISTORY_SECONDS:
            self._previous.clear()
            self._histories.clear()
            self._shift[:] = 0.
            self._clear_confirmation()
        observed = [item for item in tracks if item.observed]
        full = [item.bbox for item in observed
                if item.bbox.confidence >= .35 and self._full(item.bbox)]
        if len(full) >= 3:
            measured = np.array((median(b.width for b in full),
                                 median(b.height for b in full)))
            if self._size is None:
                self._size = measured
            else:
                weight = 1. - math.exp(-max(0., elapsed) * 3.160815469734789)
                self._size = self._size * (1. - weight) + measured * weight

        current = {item.track_id: item.bbox for item in observed}
        steps = [np.asarray(item.bbox.center) - np.asarray(previous.center)
                 for item in observed
                 if item.track_id != target_id
                 and (previous := self._previous.get(item.track_id)) is not None
                 and self._full(item.bbox) and self._full(previous)]
        consensus = None
        if len(steps) >= 3:
            vectors = np.asarray(steps)
            center = np.median(vectors, axis=0)
            deviations = np.linalg.norm(vectors - center, axis=1)
            good = vectors[deviations <= max(4., motion_tolerance)]
            if len(good) > len(vectors) / 2:
                consensus = np.median(good, axis=0)
        self._scores.clear()
        if consensus is None:
            # An unknown interval cannot be integrated as zero group motion.
            # Restart evidence instead of preserving a stale relative path.
            self._histories.clear()
            self._shift[:] = 0.
            self._clear_confirmation()
        else:
            self._shift += consensus
        for item in observed:
            history = self._histories.setdefault(item.track_id, deque(maxlen=64))
            if history and now - history[-1][0] > HISTORY_GAP_SECONDS:
                history.clear()
            history.append((now, self._motion_center(item.bbox) - self._shift))
            while history and now - history[0][0] > HISTORY_SECONDS:
                history.popleft()
            if (item.track_id not in self._previous and len(history) >= 2
                    and self._size is not None):
                jump = float(np.linalg.norm(history[-1][1] - history[-2][1]))
                if jump > max(GAP_JUMP_MINIMUM_PX, float(min(self._size)) * GAP_JUMP_SIZE_RATIO):
                    self._gap_jumps[item.track_id] = now
            if len(history) >= 5 and now - history[0][0] >= MIN_HISTORY_SECONDS:
                points = np.asarray([point for _, point in history])
                delta = np.median(points[-3:], axis=0) - np.median(points[:3], axis=0)
                self._scores[item.track_id] = float(np.linalg.norm(delta))
        self._histories = {key: history for key, history in self._histories.items()
                           if history and now - history[-1][0] <= HISTORY_GAP_SECONDS}
        self._gap_jumps = {
            key: stamp for key, stamp in self._gap_jumps.items()
            if key in self._histories and self._histories[key][0][0] < stamp <= now
        }
        self._previous = current

    def _corner_jump_is_unconfirmed(self, item: TrackedDetection) -> bool:
        """Reject a single reacquisition jump as continuing corner motion.

        Both coordinates of a corner-clipped box require extent compensation.
        An ID association change across a missing observation can therefore
        create a large residual without physical motion. Re-reading that one
        displacement on later frames is not independent movement evidence.
        Continuous tracks, single-edge boxes and small changes retain the
        existing policy. No pixels or reference positions enter this check.
        """
        if (item.track_id not in self._gap_jumps or self._frame_size is None
                or self._size is None):
            return False
        width, height = self._frame_size
        box = item.bbox
        if not ((box.x0 <= EDGE_MARGIN or box.x1 >= width - EDGE_MARGIN)
                and (box.y0 <= EDGE_MARGIN or box.y1 >= height - EDGE_MARGIN)):
            return False
        history = self._histories.get(item.track_id)
        if history is None or len(history) < 5:
            return False
        points = np.asarray([point for _, point in history])
        steps = np.diff(points, axis=0)
        jump_limit = max(GAP_JUMP_MINIMUM_PX, float(min(self._size)) * GAP_JUMP_SIZE_RATIO)
        if float(np.max(np.linalg.norm(steps, axis=1))) <= jump_limit:
            return False
        delta = np.median(points[-3:], axis=0) - np.median(points[:3], axis=0)
        distance = float(np.linalg.norm(delta))
        if distance < 1e-6:
            return True
        projections = steps @ (delta / distance)
        significant = max(1., distance * .1)
        return int(np.count_nonzero(projections >= significant)) < GAP_JUMP_SUPPORT_STEPS

    def decide(
        self, now: float, epoch: int, tracks: Sequence[TrackedDetection], *,
        target: TrackedDetection | None, target_id: int | None,
        last_box: Detection | None, eligible_ids: set[int], enabled: bool,
    ) -> EdgeIdentityDecision:
        """Authorize only a unique, repeatedly observed local collision branch."""
        if epoch != self._observation_epoch or now != self._previous_time:
            return EdgeIdentityDecision()
        if not enabled:
            self._clear_confirmation()
            self._decision_epoch = epoch
            self._last_decision = EdgeIdentityDecision()
            return self._last_decision
        if epoch == self._decision_epoch:
            return self._last_decision
        self._decision_epoch = epoch
        decision = self._decide(now, epoch, tracks, target=target,
                                target_id=target_id, last_box=last_box,
                                eligible_ids=eligible_ids, enabled=enabled)
        self._last_decision = decision
        return decision

    def _decide(
        self, now: float, epoch: int, tracks: Sequence[TrackedDetection], *,
        target: TrackedDetection | None, target_id: int | None,
        last_box: Detection | None, eligible_ids: set[int], enabled: bool,
    ) -> EdgeIdentityDecision:
        if not enabled or target_id is None or last_box is None or self._size is None:
            self._clear_confirmation()
            return EdgeIdentityDecision()
        if target is not None and target.observed and not self._full(target.bbox):
            self._edge_until = now + EDGE_EPISODE_SECONDS
        background = [score for key, score in self._scores.items() if key not in eligible_ids]
        if len(background) < 3:
            self._clear_confirmation()
            return EdgeIdentityDecision()
        noise = median(background)
        scatter = median(abs(value - noise) for value in background)
        limit = max(6., float(min(self._size)) * .05, noise + 3. * scatter)
        missing = target is None or not target.observed
        target_score = self._scores.get(target_id)
        if not missing and (target_score is None or target_score >= limit * .8):
            self._clear_confirmation()
            return EdgeIdentityDecision()

        candidates: list[tuple[float, TrackedDetection]] = []
        for item in tracks:
            box = item.bbox
            if (not item.observed or item.track_id == target_id
                    or item.track_id not in eligible_ids or box.confidence < MIN_CONFIDENCE):
                continue
            if (box.width < self._size[0] * MIN_VISIBLE_FRACTION
                    or box.height < self._size[1] * MIN_VISIBLE_FRACTION
                    or box.width > self._size[0] * MAX_EXTENT_RATIO
                    or box.height > self._size[1] * MAX_EXTENT_RATIO):
                continue
            if math.dist(box.center, last_box.center) > max(self._size) * 1.1:
                continue
            if missing and self._corner_jump_is_unconfirmed(item):
                continue
            score = self._scores.get(item.track_id, 0.)
            if score >= limit:
                candidates.append((score, item))
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        if not candidates:
            self._clear_confirmation()
            return EdgeIdentityDecision()
        score, winner = candidates[0]
        # Full boxes remain competing evidence even when this boundary-only
        # recovery path cannot authorize them. Excluding them before ranking
        # makes a weaker clipped branch falsely appear to be the unique mover.
        if now > self._edge_until and self._full(winner.bbox):
            self._clear_confirmation()
            return EdgeIdentityDecision()
        runner = candidates[1][0] if len(candidates) > 1 else 0.
        if score - runner < limit * .5:
            self._clear_confirmation()
            return EdgeIdentityDecision()

        if self._candidate_id == winner.track_id and self._candidate_epoch == epoch - 1:
            self._candidate_count += 1
        else:
            self._candidate_id = winner.track_id
            self._candidate_count = 1
            self._candidate_since = now
        self._candidate_epoch = epoch
        if self._candidate_count < 2 or now - self._candidate_since < .02:
            return EdgeIdentityDecision(pending=True)
        return EdgeIdentityDecision(candidate=winner)
