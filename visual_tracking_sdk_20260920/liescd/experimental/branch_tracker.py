"""Branch-aware identity supervision for OC-SORT association recovery.

The implementation keeps the proven opening-lock and OC-SORT behaviour,
remembers plausible merge/split branches, and continuously audits every visible
track against the fake-motion majority.  Global evidence may repair an ID swap
only for a track with recent collision provenance; unrelated outliers never
receive takeover authority.  It was validated in the standalone test bench and
is promoted to production through :mod:`liescd.runtime_tracker`; the historical
module and class names are retained to keep the validation trail reviewable.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from liescd.detector import Detection
from liescd.edge_identity import EdgeIdentityAudit
from liescd.ocsort_tracker import (
    DEFAULT_MOTION_TOLERANCE,
    OCSortBackend,
    OCSortTargetTracker,
    TrackedDetection,
)
from liescd.tracker import TrackResult, TrackState


BRANCH_CAPTURE_FRAMES = 12
BRANCH_RETENTION_FRAMES = 120
GLOBAL_SUPERVISOR_CONFIRMATIONS = 3
GLOBAL_SUPERVISOR_CONFIRMATION_S = 0.08
GLOBAL_SUPERVISOR_MIN_BELIEF = 0.6298260137416737
GLOBAL_SUPERVISOR_EVIDENCE_WEIGHT = 0.39805252395074353
OPENING_IDENTITY_PROTECTION_S = 6.81722669232694
EDGE_TAKEOVER_MARGIN_PX = 2.0
COLLISION_COHORT_RETENTION_S = 2.0
COLLISION_NEAR_MARGIN_SCALE = 0.20
COLLISION_COMPASS_CONTACT_MAX_EPOCHS = 45
COLLISION_COMPASS_MIN_SCORE = 0.45
COLLISION_COMPASS_TARGET_MAX_SCORE = 0.40
COLLISION_COMPASS_MIN_MARGIN = 0.10004642158694585
COLLISION_COMPASS_CONFIRMATIONS = 2
COLLISION_COMPASS_CONFIRMATION_S = 0.04
MISSING_COMPASS_MIN_SCORE = 0.39
MISSING_COMPASS_MIN_MARGIN = 0.21605278925691423
MISSING_COMPASS_CONFIRMATIONS = 2
MISSING_COMPASS_CONFIRMATION_S = 0.03
MISSING_LINEAGE_MAX_GAP_EPOCHS = 8
MISSING_LINEAGE_NEAR_MARGIN_SCALE = 0.30
MISSING_LINEAGE_CENTER_SCALE = 1.25
MISSING_SPATIAL_CONFIRMATIONS = 2
MISSING_SPATIAL_COMMIT_S = 0.45
MISSING_SPATIAL_GATE_MIN_PX = 24.0
MISSING_SPATIAL_GATE_MAX_PX = 100.0
MISSING_SPATIAL_ADVANTAGE_MIN_PX = 10.0
MOTION_COMPASS_HISTORY_SIZE = 9
MOTION_COMPASS_MIN_SAMPLES = 4
MOTION_COMPASS_MIN_SPEED = 0.65
MOTION_TEMPLATE_HISTORY_SIZE = 48
MOTION_TEMPLATE_MIN_SAMPLES = 12
MOTION_TEMPLATE_HOLD_EPOCHS = 3
MOTION_TEMPLATE_TARGET_MAX_SCORE = 0.35
MOTION_TEMPLATE_OUTLIER_SCORE = 0.55
MOTION_TEMPLATE_MIN_MARGIN = 0.20
MOTION_TEMPLATE_CONFIRMATIONS = 3
EDGE_TEMPLATE_OUTLIER_SCORE = 0.40
EDGE_TEMPLATE_MIN_MARGIN = 0.18
EDGE_TEMPLATE_CONFIRMATIONS = 2
RECOVERY_TAKEOVER_MIN_CONFIDENCE = 0.20

# These values belong to the branch-aware tracker and deliberately do not alter
# the rollback-capable base OC-SORT adapter.  The opening highlight can
# repair an association exchange while it is still strong; after a short
# fade it is closed permanently.  Recovery vectors use a shorter window than
# the display compass so a branch can be judged soon after a split.
OPENING_HIGHLIGHT_REBIND_MINIMUM = 0.34
OPENING_HIGHLIGHT_REBIND_MARGIN = 0.12
OPENING_HIGHLIGHT_REBIND_CONFIRMATIONS = 2
OPENING_HIGHLIGHT_FADE_FRAMES = 3
RECOVERY_VECTOR_HISTORY_SIZE = 5
# A split candidate may be visible for only a couple of detector frames before
# it is occluded again.  Two samples provide a provisional vector; the
# independent two-frame takeover confirmation and target-outlier guard below
# keep that provisional estimate from acting on a single jittery box.
RECOVERY_VECTOR_MIN_SAMPLES = 2
RECOVERY_VECTOR_MIN_SPEED = 0.65
RECOVERY_VECTOR_OUTLIER_SCORE = 0.45
RECOVERY_VECTOR_TARGET_MAX_SCORE = 0.36
RECOVERY_VECTOR_MIN_MARGIN = 0.12
RECOVERY_VECTOR_CONFIRMATIONS = 2
# Two consecutive detector observations are the primary confirmation.  Keep
# the wall-clock floor below one additional 29--30 FPS frame so a confirmed
# split is not held back for three frames by a second independent timer.
RECOVERY_VECTOR_CONFIRMATION_S = 0.04

# OC-SORT may keep the same numerical ID while briefly associating it with the
# other rectangle in a crossing.  Identity-only supervision cannot see that
# exchange.  The experiment therefore quarantines a same-ID observation when
# a collision-proven alternative is substantially more continuous with the
# last accepted target rectangle.  Three detector frames are short enough to
# coast through the exchange without allowing a stale position to persist.
ASSOCIATION_QUARANTINE_FRAMES = 3
ASSOCIATION_JUMP_MIN_PX = 24.0
ASSOCIATION_JUMP_MAX_PX = 96.0
ASSOCIATION_JUMP_DIAGONAL_SCALE = 0.22
ASSOCIATION_COMPETITOR_DISTANCE_MARGIN_PX = 16.0
ASSOCIATION_COMPETITOR_IOU_MARGIN = 0.10
CONTINUITY_MAX_EPOCHS = 16
CONTINUITY_CONFIRMATIONS = 2
CONTINUITY_ADVANTAGE_MIN_PX = 12.0
CONTINUITY_GATE_MIN_PX = 18.0
CONTINUITY_GATE_MAX_PX = 80.0


@dataclass(frozen=True, slots=True)
class MotionCompass:
    """Read-only robust motion estimate for one currently observed track."""

    track_id: int
    center: tuple[float, float]
    vector: tuple[float, float] | None
    sample_count: int
    deviation_score: float | None = None
    rank: int | None = None


@dataclass(frozen=True, slots=True)
class MotionTemplate:
    """The current majority motion used as the fake-motion benchmark."""

    vector: tuple[float, float] | None
    sample_count: int
    ready: bool
    inlier_count: int
    track_count: int


@dataclass(frozen=True, slots=True)
class IdentityTuning:
    """Instance-local research settings; defaults preserve production behavior."""

    collision_margin: float = COLLISION_COMPASS_MIN_MARGIN
    missing_margin: float = MISSING_COMPASS_MIN_MARGIN
    supervisor_belief: float = GLOBAL_SUPERVISOR_MIN_BELIEF
    evidence_weight: float = GLOBAL_SUPERVISOR_EVIDENCE_WEIGHT

    def __post_init__(self) -> None:
        for name in ("collision_margin", "missing_margin", "supervisor_belief", "evidence_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be finite and in (0, 1)")


class ExperimentalBranchTargetTracker(OCSortTargetTracker):
    """OC-SORT tracker with branch retention and global identity supervision.

    Candidate membership is intentionally strict: an ID may join only near the
    merge anchor or after physically meeting the bound target.  After joining
    it remains eligible for this recovery episode regardless of distance.  The
    always-on supervisor still scores every visible ID, but a transfer requires
    collision provenance, repeated outlier evidence, and a fake-like current
    target.
    """

    def __init__(
        self,
        *,
        backend: OCSortBackend | None = None,
        lock_confirmations: int = 2,
        highlight_minimum: float = 0.20,
        highlight_margin: float = 0.08,
        coast_seconds: float = 0.25,
        motion_tolerance: float = DEFAULT_MOTION_TOLERANCE,
        opening_protection_seconds: float = OPENING_IDENTITY_PROTECTION_S,
        identity_tuning: IdentityTuning | None = None,
    ) -> None:
        self.identity_tuning = identity_tuning or IdentityTuning()
        self.opening_protection_seconds = max(
            0.0,
            float(opening_protection_seconds),
        )
        super().__init__(
            backend=backend,
            lock_confirmations=lock_confirmations,
            highlight_minimum=highlight_minimum,
            highlight_margin=highlight_margin,
            coast_seconds=coast_seconds,
            motion_tolerance=motion_tolerance,
        )

    def _clear_local_state(self) -> None:
        super()._clear_local_state()
        self._edge_identity_audit = EdgeIdentityAudit()
        self._previous_public_result: TrackResult | None = None
        self._branch_candidate_ids: set[int] = set()
        self._branch_anchor: tuple[float, float] | None = None
        self._branch_capture_until = -1
        self._branch_expires_at = -1
        self._branch_lineage_boxes: dict[int, tuple[int, Detection]] = {}
        self._supervisor_beliefs: dict[int, float] = {}
        self._supervisor_candidate_id: int | None = None
        self._supervisor_candidate_count = 0
        self._supervisor_candidate_since_s: float | None = None
        self._supervisor_candidate_last_epoch = -1
        self._supervisor_last_epoch = -1
        self._current_timestamp_s = 0.0
        self._opening_protected_until_s = float("-inf")
        self._frame_size: tuple[int, int] | None = None
        self._collision_candidate_ids: set[int] = set()
        self._collision_contact_epochs: dict[int, int] = {}
        self._collision_expires_at_s = float("-inf")
        self._collision_compass_candidate_id: int | None = None
        self._collision_compass_candidate_count = 0
        self._collision_compass_candidate_since_s: float | None = None
        self._collision_compass_candidate_last_epoch = -1
        self._missing_compass_candidate_id: int | None = None
        self._missing_compass_candidate_count = 0
        self._missing_compass_candidate_since_s: float | None = None
        self._missing_compass_candidate_last_epoch = -1
        self._compass_histories: dict[
            int,
            deque[tuple[float, tuple[float, float]]],
        ] = {}
        self._compass_previous_centers: dict[
            int,
            tuple[int, tuple[float, float]],
        ] = {}
        self._motion_compasses: tuple[MotionCompass, ...] = ()
        self._motion_template = MotionTemplate(None, 0, False, 0, 0)
        self._template_vector: np.ndarray | None = None
        self._template_valid_epochs: deque[int] = deque(
            maxlen=MOTION_TEMPLATE_HISTORY_SIZE
        )
        self._template_last_valid_epoch = -1
        self._template_candidate_id: int | None = None
        self._template_candidate_count = 0
        self._template_candidate_last_epoch = -1
        self._opening_highlight_active = False
        self._opening_highlight_misses = 0
        self._opening_highlight_candidate_id: int | None = None
        self._opening_highlight_candidate_count = 0
        self._opening_highlight_candidate_result: TrackedDetection | None = None
        self._recovery_vector_previous: dict[
            int,
            tuple[int, tuple[float, float]],
        ] = {}
        self._recovery_vector_histories: dict[
            int,
            deque[tuple[float, tuple[float, float]]],
        ] = {}
        self._recovery_vectors: dict[int, np.ndarray] = {}
        self._recovery_reference_vector: np.ndarray | None = None
        self._recovery_vector_candidate_id: int | None = None
        self._recovery_vector_candidate_count = 0
        self._recovery_vector_candidate_since_s: float | None = None
        self._recovery_event_active = False
        self._recovery_event_started_epoch = -1
        self._recovery_target_was_observed: bool | None = None
        self._current_active_tracks: tuple[TrackedDetection, ...] = ()
        self._association_quarantine_count = 0
        self._association_quarantined = False
        self._continuity_anchor_epoch = -1
        self._continuity_anchor: tuple[float, float] | None = None
        self._continuity_vector: tuple[float, float] | None = None
        self._continuity_candidate_id: int | None = None
        self._continuity_candidate_count = 0
        self._continuity_candidate_last_epoch = -1
        self._missing_spatial_candidate_id: int | None = None
        self._missing_spatial_candidate_count = 0
        self._missing_spatial_candidate_last_epoch = -1
        self._missing_spatial_candidate_since_s: float | None = None
        self._provisional_output: TrackedDetection | None = None

    @property
    def branch_candidate_ids(self) -> frozenset[int]:
        """Expose the current episode membership for diagnostics and tests."""

        return frozenset(self._branch_candidate_ids)

    def _update_bound_target(
        self,
        now: float,
        active_tracks: Sequence[TrackedDetection],
        *,
        skipped: bool,
    ) -> TrackResult:
        """Expose a local split hypothesis without replacing OC-SORT identity.

        The original bound ID remains authoritative while it is temporarily
        missing.  A collision-proven local branch may supply the current aim
        point, but it is committed as the new identity only after independent
        motion evidence confirms it.  If the original OC-SORT ID resumes, the
        hypothesis disappears automatically instead of causing an ID switch.
        """

        self._provisional_output = None
        result = super()._update_bound_target(
            now,
            active_tracks,
            skipped=skipped,
        )
        provisional = self._provisional_output
        if provisional is None or result.observed:
            return result
        return TrackResult(
            # A collision-proven local branch is safe enough to drive the
            # output layer, but not yet strong enough to replace the
            # authoritative OC-SORT identity.  COAST already carries exactly
            # that contract through both live controllers: observed=False,
            # mouse output allowed, and no identity commit.  If the local
            # branch disappears, the next update falls back to LOST and the
            # mouse freezes normally.
            state=TrackState.COAST,
            aim_point=provisional.bbox.center,
            target_bbox=provisional.bbox,
            observed=False,
            target_id=result.target_id,
            trail_break=result.trail_break,
        )

    @property
    def supervisor_candidate(self) -> tuple[int | None, int]:
        """Return the current global candidate and consecutive confirmation."""

        return self._supervisor_candidate_id, self._supervisor_candidate_count

    @property
    def supervisor_beliefs(self) -> dict[int, float]:
        """Return a diagnostic snapshot of the current true-target beliefs."""

        return dict(self._supervisor_beliefs)

    @property
    def motion_compasses(self) -> tuple[MotionCompass, ...]:
        """Return compass estimates without granting them decision authority."""

        return self._motion_compasses

    @property
    def motion_template(self) -> MotionTemplate:
        """Return the current majority-vector benchmark for diagnostics."""

        return self._motion_template

    @property
    def opening_identity_protected(self) -> bool:
        """Whether the opening highlight lock still has exclusive authority."""

        return self._current_timestamp_s < self._opening_protected_until_s

    @property
    def ambiguity_candidate_ids(self) -> frozenset[int]:
        """Expose recent collision provenance for diagnostics and tests."""

        if self._current_timestamp_s > self._collision_expires_at_s:
            return frozenset()
        return frozenset(self._collision_candidate_ids)

    @property
    def association_quarantined(self) -> bool:
        """Whether this frame rejected a same-ID association discontinuity."""

        return self._association_quarantined

    def set_frame_size(self, width: int, height: int) -> None:
        """Provide the current frame boundary for takeover eligibility checks."""

        parsed = int(width), int(height)
        frame_size = parsed if min(parsed) > 0 else None
        if frame_size != self._frame_size:
            self._previous_public_result = None
        self._frame_size = frame_size

    def update(
        self,
        timestamp: float,
        detections: Sequence[Detection] | None,
    ) -> TrackResult:
        result = self._update_recovery_frame(timestamp, detections)
        self._previous_public_result = result
        return result

    def _update_recovery_frame(
        self,
        timestamp: float,
        detections: Sequence[Detection] | None,
    ) -> TrackResult:
        """Remember wall-clock time and protect a newly highlighted identity."""

        self._current_timestamp_s = float(timestamp)
        had_target = self._target_track_id is not None
        previous_target_id = self._target_track_id
        previous_state = self._state
        previous_observed_time = self._last_observed_time
        previous_bbox = self._last_bbox
        previous_aim = self._last_aim
        self._association_quarantined = False
        result = super().update(timestamp, detections)
        if not had_target and self._target_track_id is not None:
            self._opening_protected_until_s = (
                self._current_timestamp_s + self.opening_protection_seconds
            )
            self._opening_highlight_active = True
            self._opening_highlight_misses = 0
            self._opening_highlight_candidate_id = None
            self._opening_highlight_candidate_count = 0
            self._opening_highlight_candidate_result = None
            self._clear_branch_episode()
            self._clear_supervisor_decision(clear_beliefs=True)
            self._association_quarantine_count = 0
            return result

        if (
            detections is not None
            and previous_target_id is not None
            and result.target_id == previous_target_id
            and result.observed
            and previous_bbox is not None
            and previous_aim is not None
            and self._association_quarantine_count
            < ASSOCIATION_QUARANTINE_FRAMES
            and self._same_id_association_is_discontinuous(
                previous_target_id,
                previous_bbox,
                self._current_active_tracks,
            )
        ):
            # The backend and motion histories still consume this frame, but
            # the public target observation does not.  Restoring the last
            # accepted rectangle keeps the mouse inside the overlap while the
            # two associations settle, without switching to an unproven ID.
            self._association_quarantine_count += 1
            self._association_quarantined = True
            self._target_track_id = previous_target_id
            self._last_observed_time = previous_observed_time
            self._last_bbox = previous_bbox
            self._last_aim = previous_aim
            missing_age = (
                float("inf")
                if previous_observed_time is None
                else max(0.0, self._current_timestamp_s - previous_observed_time)
            )
            self._state = (
                previous_state
                if previous_state is TrackState.LOST
                else (
                    TrackState.COAST
                    if missing_age <= self.coast_seconds
                    else TrackState.LOST
                )
            )
            return self._result(observed=False)

        self._association_quarantine_count = 0
        return result

    def _record_motion(
        self,
        active_tracks: Sequence[TrackedDetection],
    ) -> None:
        """Record production motion evidence, then build display-only vectors."""

        self._current_active_tracks = tuple(active_tracks)
        super()._record_motion(active_tracks)
        epoch = self._detection_epoch
        observed = [item for item in active_tracks if item.observed]

        for item in observed:
            history = self._compass_histories.setdefault(
                item.track_id,
                deque(maxlen=MOTION_COMPASS_HISTORY_SIZE),
            )
            history.append((self._motion_time, item.bbox.center))

        current_centers = {
            item.track_id: item.bbox.center for item in observed
        }
        steps: dict[int, np.ndarray] = {}
        for track_id, center in current_centers.items():
            previous = self._compass_previous_centers.get(track_id)
            if previous is not None and previous[0] == epoch - 1:
                steps[track_id] = (np.asarray(center, dtype=np.float64) - np.asarray(
                    previous[1], dtype=np.float64
                )) * self._motion_step_scale
        self._update_motion_template(steps, observed)
        self._compass_previous_centers = {
            track_id: (epoch, center)
            for track_id, center in current_centers.items()
        }

        # Keep the diagnostic window short: an ID that disappears and later
        # reappears should not drag a long, possibly swapped history behind it.
        stale_before = self._motion_time - MOTION_COMPASS_HISTORY_SIZE
        self._compass_histories = {
            track_id: history
            for track_id, history in self._compass_histories.items()
            if history and history[-1][0] >= stale_before
        }

        compasses: list[MotionCompass] = []
        for item in observed:
            history = self._compass_histories[item.track_id]
            compasses.append(
                MotionCompass(
                    track_id=item.track_id,
                    center=item.bbox.center,
                    vector=self._robust_compass_vector(history),
                    sample_count=len(history),
                )
            )
        self._motion_compasses = self._rank_motion_compasses(compasses)
        self._record_recovery_vectors(observed)
        self._edge_identity_audit.observe(
            self._current_timestamp_s, self._detection_epoch, active_tracks,
            frame_size=self._frame_size, target_id=self._target_track_id,
            motion_tolerance=self.motion_tolerance,
        )

    def _same_id_association_is_discontinuous(
        self,
        target_id: int,
        previous_bbox: Detection,
        active_tracks: Sequence[TrackedDetection],
    ) -> bool:
        """Detect a temporary rectangle exchange hidden behind a stable ID.

        A large target displacement alone is not enough: a second visible ID
        from the current collision/branch episode must match the previous
        rectangle materially better in both position and overlap.  The method
        abstains when no such alternative exists, so legitimate fast movement
        remains under OC-SORT's control.
        """

        target = next(
            (
                item
                for item in active_tracks
                if item.observed and item.track_id == target_id
            ),
            None,
        )
        if target is None:
            return False

        previous_center = previous_bbox.center
        target_distance = float(
            np.hypot(
                target.bbox.center[0] - previous_center[0],
                target.bbox.center[1] - previous_center[1],
            )
        )
        jump_limit = max(
            ASSOCIATION_JUMP_MIN_PX,
            min(
                ASSOCIATION_JUMP_MAX_PX,
                previous_bbox.diagonal * ASSOCIATION_JUMP_DIAGONAL_SCALE,
            ),
        )
        if target_distance <= jump_limit:
            return False

        eligible_ids = self._eligible_global_ids()
        if target_id not in eligible_ids:
            return False

        target_iou = self._bbox_iou(previous_bbox, target.bbox)
        alternatives: list[tuple[float, float]] = []
        for item in active_tracks:
            if (
                not item.observed
                or item.track_id == target_id
                or item.track_id not in eligible_ids
                or not self._candidate_can_take_over(item)
            ):
                continue
            distance = float(
                np.hypot(
                    item.bbox.center[0] - previous_center[0],
                    item.bbox.center[1] - previous_center[1],
                )
            )
            overlap = self._bbox_iou(previous_bbox, item.bbox)
            alternatives.append((distance, overlap))

        if not alternatives:
            return False
        best_distance, best_iou = min(
            alternatives,
            key=lambda entry: (entry[0], -entry[1]),
        )
        return (
            best_distance <= jump_limit
            and target_distance - best_distance
            >= ASSOCIATION_COMPETITOR_DISTANCE_MARGIN_PX
            and best_iou - target_iou >= ASSOCIATION_COMPETITOR_IOU_MARGIN
        )

    def _record_recovery_vectors(
        self,
        observed: Sequence[TrackedDetection],
    ) -> None:
        """Maintain short robust vectors for collision-branch candidates.

        The production tracker stores residuals only after a strict recovery
        trigger.  This experiment also keeps a bounded center history for all
        observed OC-SORT IDs.  A vector is a median pairwise slope, so one
        jittery rectangle does not dominate the estimate.  It is diagnostic
        state until a branch episode has explicitly captured the ID.
        """

        epoch = self._detection_epoch
        current = {item.track_id: item.bbox.center for item in observed}
        for track_id, center in current.items():
            history = self._recovery_vector_histories.setdefault(
                track_id,
                deque(maxlen=RECOVERY_VECTOR_HISTORY_SIZE),
            )
            previous = self._recovery_vector_previous.get(track_id)
            if previous is None or previous[0] != epoch - 1:
                history.clear()
            history.append((self._motion_time, center))

        self._recovery_vector_previous = {
            track_id: (epoch, center)
            for track_id, center in current.items()
        }
        stale_before = self._motion_time - RECOVERY_VECTOR_HISTORY_SIZE * 2
        self._recovery_vector_histories = {
            track_id: history
            for track_id, history in self._recovery_vector_histories.items()
            if history and history[-1][0] >= stale_before
        }
        self._recovery_vectors = {}
        for track_id, history in self._recovery_vector_histories.items():
            vector = self._robust_recovery_vector(history)
            if vector is not None:
                self._recovery_vectors[track_id] = vector

        target_id = self._target_track_id
        visible_ids = set(current)
        pool = [
            vector
            for track_id, vector in self._recovery_vectors.items()
            if track_id in visible_ids and track_id != target_id
        ]
        if len(pool) < 3:
            pool = list(self._recovery_vectors.values())
        if len(pool) < 3:
            self._recovery_reference_vector = None
            return

        vectors = np.stack(pool)
        center = np.median(vectors, axis=0)
        distances = np.linalg.norm(vectors - center, axis=1)
        inlier_limit = max(2.0, self.motion_tolerance * 1.5)
        inliers = vectors[distances <= inlier_limit]
        if len(inliers) < 2 or len(inliers) <= len(vectors) / 2:
            self._recovery_reference_vector = None
            return
        reference = np.median(inliers, axis=0)
        speed = float(np.linalg.norm(reference))
        self._recovery_reference_vector = (
            reference.astype(np.float64, copy=True)
            if np.isfinite(speed) and speed >= RECOVERY_VECTOR_MIN_SPEED
            else None
        )

    @staticmethod
    def _robust_recovery_vector(
        history: Sequence[tuple[float, tuple[float, float]]],
    ) -> np.ndarray | None:
        if len(history) < RECOVERY_VECTOR_MIN_SAMPLES:
            return None
        samples = list(history)
        slopes: list[tuple[float, float]] = []
        for start_index, (start_epoch, start_center) in enumerate(samples[:-1]):
            for end_epoch, end_center in samples[start_index + 1 :]:
                elapsed = end_epoch - start_epoch
                if elapsed < 1:
                    continue
                slopes.append(
                    (
                        (end_center[0] - start_center[0]) / elapsed,
                        (end_center[1] - start_center[1]) / elapsed,
                    )
                )
        if not slopes:
            return None
        vector = np.median(np.asarray(slopes, dtype=np.float64), axis=0)
        speed = float(np.linalg.norm(vector))
        if not np.isfinite(speed) or speed < RECOVERY_VECTOR_MIN_SPEED:
            return None
        return vector

    @staticmethod
    def _recovery_vector_score(
        vector: np.ndarray | None,
        reference: np.ndarray | None,
    ) -> float | None:
        if vector is None or reference is None:
            return None
        speed = float(np.linalg.norm(vector))
        reference_speed = float(np.linalg.norm(reference))
        if (
            not np.isfinite(speed)
            or not np.isfinite(reference_speed)
            or speed < RECOVERY_VECTOR_MIN_SPEED
            or reference_speed < RECOVERY_VECTOR_MIN_SPEED
        ):
            return None
        cosine = float(np.dot(vector, reference) / (speed * reference_speed))
        angle_score = math.acos(max(-1.0, min(1.0, cosine))) / math.pi
        speed_score = min(
            1.0,
            abs(speed - reference_speed) / max(0.5, reference_speed),
        )
        return 0.70 * angle_score + 0.30 * speed_score

    def _update_motion_template(
        self,
        steps: dict[int, np.ndarray],
        observed: Sequence[TrackedDetection],
    ) -> None:
        """Update a majority-motion template without trusting any one ID.

        The opening target is excluded after it has been locked.  Before that
        point there is no identity to exclude, so no template is accumulated.
        A single frame is never enough to make the template ready.
        """

        if self._target_track_id is None:
            self._motion_template = MotionTemplate(None, 0, False, 0, 0)
            self._template_vector = None
            self._template_valid_epochs.clear()
            self._template_last_valid_epoch = -1
            return

        template_steps = dict(steps)
        template_steps.pop(self._target_track_id, None)
        if len(template_steps) < 3:
            # If excluding a currently bound ID leaves too few tracks, use the
            # robust all-track majority; one real target is still an outlier.
            template_steps = steps
        consensus = self._majority_motion(template_steps)
        if consensus is None:
            if (
                self._template_last_valid_epoch < 0
                or self._detection_epoch - self._template_last_valid_epoch
                > MOTION_TEMPLATE_HOLD_EPOCHS
            ):
                self._template_vector = None
            self._motion_template = MotionTemplate(
                self._vector_tuple(self._template_vector),
                len(self._template_valid_epochs),
                self._template_vector is not None
                and len(self._template_valid_epochs) >= MOTION_TEMPLATE_MIN_SAMPLES,
                0,
                len(template_steps),
            )
            return

        if self._template_vector is None:
            self._template_vector = consensus.astype(np.float64, copy=True)
        else:
            self._template_vector = (
                0.65 * self._template_vector + 0.35 * consensus
            )
        self._template_valid_epochs.append(self._detection_epoch)
        self._template_last_valid_epoch = self._detection_epoch
        self._motion_template = MotionTemplate(
            self._vector_tuple(self._template_vector),
            len(self._template_valid_epochs),
            len(self._template_valid_epochs) >= MOTION_TEMPLATE_MIN_SAMPLES,
            len(template_steps),
            len(observed),
        )

    def _rank_motion_compasses(
        self,
        compasses: Sequence[MotionCompass],
    ) -> tuple[MotionCompass, ...]:
        template_vector = self._motion_template.vector
        if not self._motion_template.ready or template_vector is None:
            return tuple(compasses)

        scored: list[tuple[MotionCompass, float]] = []
        for compass in compasses:
            score = self._template_deviation(compass.vector, template_vector)
            if score is not None:
                scored.append((compass, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        ranks = {item.track_id: index + 1 for index, (item, _) in enumerate(scored)}
        values = {item.track_id: score for item, score in scored}
        return tuple(
            MotionCompass(
                track_id=item.track_id,
                center=item.center,
                vector=item.vector,
                sample_count=item.sample_count,
                deviation_score=values.get(item.track_id),
                rank=ranks.get(item.track_id),
            )
            for item in compasses
        )

    @staticmethod
    def _vector_tuple(vector: np.ndarray | None) -> tuple[float, float] | None:
        if (
            vector is None
            or vector.shape != (2,)
            or not np.all(np.isfinite(vector))
        ):
            return None
        return float(vector[0]), float(vector[1])

    @staticmethod
    def _template_deviation(
        vector: tuple[float, float] | None,
        template: tuple[float, float],
    ) -> float | None:
        if vector is None:
            return None
        current = np.asarray(vector, dtype=np.float64)
        reference = np.asarray(template, dtype=np.float64)
        current_speed = float(np.linalg.norm(current))
        reference_speed = float(np.linalg.norm(reference))
        if (
            not np.isfinite(current_speed)
            or not np.isfinite(reference_speed)
            or current_speed <= 1e-6
            or reference_speed <= 1e-6
        ):
            return None
        cosine = float(
            np.dot(current, reference) / (current_speed * reference_speed)
        )
        angle_score = math.acos(max(-1.0, min(1.0, cosine))) / math.pi
        speed_score = min(
            1.0,
            abs(current_speed - reference_speed) / max(0.5, reference_speed),
        )
        return 0.70 * angle_score + 0.30 * speed_score

    @staticmethod
    def _robust_compass_vector(
        history: Sequence[tuple[float, tuple[float, float]]],
    ) -> tuple[float, float] | None:
        """Estimate direction with a Theil-Sen-style median slope.

        Pairwise samples must be at least two detection epochs apart so one
        frame of YOLO box jitter cannot dominate the displayed direction.
        """

        if len(history) < MOTION_COMPASS_MIN_SAMPLES:
            return None

        samples = list(history)
        slopes: list[tuple[float, float]] = []
        for start_index, (start_epoch, start_center) in enumerate(samples[:-1]):
            for end_epoch, end_center in samples[start_index + 1 :]:
                elapsed = end_epoch - start_epoch
                if elapsed < 2:
                    continue
                slopes.append(
                    (
                        (end_center[0] - start_center[0]) / elapsed,
                        (end_center[1] - start_center[1]) / elapsed,
                    )
                )
        if not slopes:
            return None

        vector = np.median(np.asarray(slopes, dtype=np.float64), axis=0)
        speed = float(np.linalg.norm(vector))
        if not np.isfinite(speed) or speed < MOTION_COMPASS_MIN_SPEED:
            return None
        return float(vector[0]), float(vector[1])

    def _clear_branch_episode(self) -> None:
        self._branch_candidate_ids.clear()
        self._branch_lineage_boxes.clear()
        self._branch_anchor = None
        self._branch_capture_until = -1
        self._branch_expires_at = -1
        self._clear_recovery_vector_decision()
        self._clear_continuity_episode()

    def _clear_continuity_decision(self) -> None:
        self._continuity_candidate_id = None
        self._continuity_candidate_count = 0
        self._continuity_candidate_last_epoch = -1

    def _clear_continuity_episode(self) -> None:
        self._continuity_anchor_epoch = -1
        self._continuity_anchor = None
        self._continuity_vector = None
        self._clear_continuity_decision()

    def _clear_missing_spatial_decision(self) -> None:
        self._missing_spatial_candidate_id = None
        self._missing_spatial_candidate_count = 0
        self._missing_spatial_candidate_last_epoch = -1
        self._missing_spatial_candidate_since_s = None

    def _missing_spatial_rebind_candidate(
        self,
        active_tracks: Sequence[TrackedDetection],
        target: TrackedDetection | None,
    ) -> TrackedDetection | None:
        candidate = self._select_missing_spatial_candidate(active_tracks, target)
        previous = self._previous_public_result
        if (candidate is None or previous is None or previous.aim_point is None
                or previous.target_bbox is None):
            return candidate
        box = candidate.bbox
        x, y = previous.aim_point
        gate = max(24.0, min(60.0, min(previous.target_bbox.width,
                                      previous.target_bbox.height) * 0.45))
        # Validate the winner after ranking all observations. Filtering the
        # candidate pool first could erase a runner-up and manufacture a
        # spurious unique winner. A full box containing the last public aim
        # may legitimately restore a clipped box despite its shifted center.
        if (math.dist(box.center, previous.aim_point) > gate
                and not (box.x0 <= x <= box.x1 and box.y0 <= y <= box.y1)):
            self._clear_missing_spatial_decision()
            return None
        return candidate

    def _select_missing_spatial_candidate(
        self,
        active_tracks: Sequence[TrackedDetection],
        target: TrackedDetection | None,
    ) -> TrackedDetection | None:
        """Continue one unambiguous local OC-SORT branch after target loss.

        This is a provisional OC-SORT-first handoff, not a global motion
        guess.  The replacement must already belong to the collision cohort,
        remain close to the missing target's Kalman prediction, preserve box
        area, stay fully inside the ROI, and win twice consecutively.  The
        normal motion supervisor remains able to correct the provisional ID
        if later evidence proves that it followed the fake majority.
        """

        if (
            not self._identity_recovery_enabled
            or self._last_bbox is None
            or self._target_track_id is None
        ):
            self._clear_missing_spatial_decision()
            return None

        # Once one local branch has survived the two-frame confirmation, keep
        # following that OC-SORT identity while it remains a valid full box.
        # Re-ranking every missing frame by distance to the stale target
        # prediction lets a nearby fake branch steal the provisional output
        # even though OC-SORT is still maintaining the original split branch.
        if (
            self._missing_spatial_candidate_id is not None
            and self._missing_spatial_candidate_count
            >= MISSING_SPATIAL_CONFIRMATIONS
            and self._missing_spatial_candidate_last_epoch
            in (self._detection_epoch - 1, self._detection_epoch)
        ):
            retained = next(
                (
                    item
                    for item in active_tracks
                    if item.observed
                    and item.track_id == self._missing_spatial_candidate_id
                    and self._candidate_can_take_over(item)
                    and 0.35
                    <= item.bbox.area / max(1.0, self._last_bbox.area)
                    <= 2.20
                ),
                None,
            )
            if retained is not None:
                self._missing_spatial_candidate_last_epoch = self._detection_epoch
                return retained

        anchor = (
            target.bbox.center
            if target is not None
            else self._last_bbox.center
        )
        eligible = self._eligible_global_ids()
        candidates: list[tuple[float, TrackedDetection]] = []
        for item in active_tracks:
            if (
                not item.observed
                or item.track_id == self._target_track_id
                or item.track_id not in eligible
                or not self._candidate_can_take_over(item)
            ):
                continue
            area_ratio = item.bbox.area / max(1.0, self._last_bbox.area)
            if not 0.45 <= area_ratio <= 2.20:
                continue
            distance = float(np.hypot(
                item.bbox.center[0] - anchor[0],
                item.bbox.center[1] - anchor[1],
            ))
            candidates.append((distance, item))
        if not candidates:
            self._clear_missing_spatial_decision()
            return None

        candidates.sort(key=lambda value: value[0])
        winner_distance, winner = candidates[0]
        box_scale = max(
            self._last_bbox.width,
            self._last_bbox.height,
            winner.bbox.width,
            winner.bbox.height,
        )
        gate = max(
            MISSING_SPATIAL_GATE_MIN_PX,
            min(MISSING_SPATIAL_GATE_MAX_PX, box_scale * 0.85),
        )
        advantage = max(
            MISSING_SPATIAL_ADVANTAGE_MIN_PX,
            min(30.0, box_scale * 0.15),
        )
        if winner_distance > gate or (
            len(candidates) > 1
            and candidates[1][0] - winner_distance < advantage
        ):
            self._clear_missing_spatial_decision()
            return None

        consecutive = (
            self._missing_spatial_candidate_id == winner.track_id
            and self._missing_spatial_candidate_last_epoch
            == self._detection_epoch - 1
        )
        if consecutive:
            self._missing_spatial_candidate_count += 1
        else:
            self._missing_spatial_candidate_id = winner.track_id
            self._missing_spatial_candidate_count = 1
            self._missing_spatial_candidate_since_s = self._current_timestamp_s
        self._missing_spatial_candidate_last_epoch = self._detection_epoch
        if self._missing_spatial_candidate_count < MISSING_SPATIAL_CONFIRMATIONS:
            return None
        return winner

    def _start_continuity_episode(self) -> None:
        """Freeze the target trajectory immediately before a collision."""

        self._clear_continuity_episode()
        if self._target_track_id is None or self._last_bbox is None:
            return
        history = self._compass_histories.get(self._target_track_id)
        if history is None:
            return
        prior = [sample for sample in history if sample[0] < self._motion_time]
        vector = self._robust_compass_vector(prior)
        if vector is None:
            return
        self._continuity_anchor_epoch = prior[-1][0]
        self._continuity_anchor = self._last_bbox.center
        self._continuity_vector = vector

    def _continuity_rebind_candidate(
        self,
        active_tracks: Sequence[TrackedDetection],
        target: TrackedDetection | None,
    ) -> TrackedDetection | None:
        """Prefer the collision branch continuous with the pre-contact path.

        This evidence is independent of the numerical OC-SORT identity.  It is
        deliberately short-lived and may choose only an already collision-
        proven branch, so a normal distant outlier cannot gain authority.
        """

        if (
            target is None
            or not target.observed
            or self._continuity_anchor is None
            or self._continuity_vector is None
            or self._continuity_anchor_epoch < 0
        ):
            self._clear_continuity_decision()
            return None
        elapsed = self._motion_time - self._continuity_anchor_epoch
        if elapsed <= 0 or elapsed > CONTINUITY_MAX_EPOCHS:
            self._clear_continuity_episode()
            return None

        eligible = self._eligible_global_ids()
        candidates = [
            item
            for item in active_tracks
            if item.observed
            and item.track_id != target.track_id
            and item.track_id in eligible
            and self._candidate_can_take_over(item)
        ]
        if not candidates:
            self._clear_continuity_decision()
            return None

        target_motion_score = self._recovery_vector_score(
            self._recovery_vectors.get(target.track_id),
            self._recovery_reference_vector,
        )
        if (
            target_motion_score is None
            or target_motion_score > RECOVERY_VECTOR_TARGET_MAX_SCORE
        ):
            self._clear_continuity_decision()
            return None

        predicted = (
            self._continuity_anchor[0] + self._continuity_vector[0] * elapsed,
            self._continuity_anchor[1] + self._continuity_vector[1] * elapsed,
        )

        def distance(item: TrackedDetection) -> float:
            return float(
                np.hypot(
                    item.bbox.center[0] - predicted[0],
                    item.bbox.center[1] - predicted[1],
                )
            )

        winner = min(candidates, key=distance)
        winner_distance = distance(winner)
        target_distance = distance(target)
        box_scale = max(
            target.bbox.width,
            target.bbox.height,
            winner.bbox.width,
            winner.bbox.height,
        )
        gate = max(
            CONTINUITY_GATE_MIN_PX,
            min(CONTINUITY_GATE_MAX_PX, box_scale * 0.65),
        )
        advantage = max(
            CONTINUITY_ADVANTAGE_MIN_PX,
            min(36.0, box_scale * 0.15),
        )
        if winner_distance > gate or target_distance - winner_distance < advantage:
            self._clear_continuity_decision()
            return None

        consecutive = (
            self._continuity_candidate_id == winner.track_id
            and self._continuity_candidate_last_epoch == self._detection_epoch - 1
        )
        if consecutive:
            self._continuity_candidate_count += 1
        else:
            self._continuity_candidate_id = winner.track_id
            self._continuity_candidate_count = 1
        self._continuity_candidate_last_epoch = self._detection_epoch
        if self._continuity_candidate_count < CONTINUITY_CONFIRMATIONS:
            return None
        return winner

    def _clear_recovery_vector_decision(self) -> None:
        """Forget a vector takeover candidate after any invalid observation."""

        self._recovery_vector_candidate_id = None
        self._recovery_vector_candidate_count = 0
        self._recovery_vector_candidate_since_s = None

    def _refresh_opening_highlight(
        self,
        active_tracks: Sequence[TrackedDetection],
    ) -> None:
        """Update the one-shot opening-highlight authority.

        This is intentionally separate from the base opening lock.  The base
        class uses the highlight to create the first identity; this second
        pass only has authority during that same opening window and can repair
        an OC-SORT label exchange. Competition alone is not evidence that the
        bound target has faded: keep the pass while that reliable observation
        is still bright, but never rebind on ambiguous evidence. Three misses
        without that anchor permanently close it, as does the original deadline.
        """

        self._opening_highlight_candidate_result = None
        if not self._opening_highlight_active:
            return
        if (
            self._opening_protected_until_s != float("-inf")
            and self._current_timestamp_s >= self._opening_protected_until_s
        ):
            self._opening_highlight_active = False
            self._opening_highlight_candidate_id = None
            self._opening_highlight_candidate_count = 0
            return

        observed = [item for item in active_tracks if item.observed]
        ordered = sorted(
            observed,
            key=lambda item: item.bbox.highlight,
            reverse=True,
        )
        winner = ordered[0] if ordered else None
        runner_up = ordered[1].bbox.highlight if len(ordered) > 1 else 0.0
        reliable = (
            winner is not None
            and winner.bbox.highlight >= OPENING_HIGHLIGHT_REBIND_MINIMUM
            and winner.bbox.highlight - runner_up
            >= OPENING_HIGHLIGHT_REBIND_MARGIN
        )
        if not reliable:
            bound_still_highlighted = any(
                item.track_id == self._target_track_id
                and item.bbox.highlight >= OPENING_HIGHLIGHT_REBIND_MINIMUM
                and self._candidate_can_take_over(item)
                for item in observed
            )
            self._opening_highlight_misses = (
                0 if bound_still_highlighted else self._opening_highlight_misses + 1
            )
            self._opening_highlight_candidate_id = None
            self._opening_highlight_candidate_count = 0
            if self._opening_highlight_misses >= OPENING_HIGHLIGHT_FADE_FRAMES:
                self._opening_highlight_active = False
            return

        assert winner is not None
        self._opening_highlight_misses = 0
        if winner.track_id == self._target_track_id:
            self._opening_highlight_candidate_id = None
            self._opening_highlight_candidate_count = 0
            return
        if winner.track_id == self._opening_highlight_candidate_id:
            self._opening_highlight_candidate_count += 1
        else:
            self._opening_highlight_candidate_id = winner.track_id
            self._opening_highlight_candidate_count = 1
        if (
            self._opening_highlight_candidate_count
            >= OPENING_HIGHLIGHT_REBIND_CONFIRMATIONS
            and self._candidate_can_take_over(winner)
        ):
            self._opening_highlight_candidate_result = winner

    def _clear_supervisor_decision(self, *, clear_beliefs: bool = False) -> None:
        self._supervisor_candidate_id = None
        self._supervisor_candidate_count = 0
        self._supervisor_candidate_since_s = None
        self._supervisor_candidate_last_epoch = -1
        if clear_beliefs:
            self._supervisor_beliefs.clear()

    def _candidate_can_take_over(self, candidate: TrackedDetection) -> bool:
        """Require recovery confidence and a full ROI box for target takeover."""

        return (
            candidate.bbox.confidence >= self._takeover_confidence_threshold()
            and self._bbox_is_full(candidate)
        )

    def _takeover_confidence_threshold(self) -> float:
        """Separate association-only weak boxes from recovery candidates."""

        return RECOVERY_TAKEOVER_MIN_CONFIDENCE

    def _bbox_is_full(self, candidate: TrackedDetection) -> bool:
        """Return whether a detection is not clipped by the ROI boundary."""

        if self._frame_size is None:
            return True
        width, height = self._frame_size
        margin = EDGE_TAKEOVER_MARGIN_PX
        bbox = candidate.bbox
        return not (
            bbox.x0 <= margin
            or bbox.y0 <= margin
            or bbox.x1 >= width - margin
            or bbox.y1 >= height - margin
        )

    @staticmethod
    def _box_gap(first: Detection, second: Detection) -> float:
        gap_x = max(0.0, first.x0 - second.x1, second.x0 - first.x1)
        gap_y = max(0.0, first.y0 - second.y1, second.y0 - first.y1)
        return float(np.hypot(gap_x, gap_y))

    def _clear_collision_cohort(self) -> None:
        self._collision_candidate_ids.clear()
        self._collision_contact_epochs.clear()
        self._collision_expires_at_s = float("-inf")
        self._clear_collision_compass_decision()

    def _clear_collision_compass_decision(self) -> None:
        self._collision_compass_candidate_id = None
        self._collision_compass_candidate_count = 0
        self._collision_compass_candidate_since_s = None
        self._collision_compass_candidate_last_epoch = -1

    def _clear_missing_compass_decision(self) -> None:
        self._missing_compass_candidate_id = None
        self._missing_compass_candidate_count = 0
        self._missing_compass_candidate_since_s = None
        self._missing_compass_candidate_last_epoch = -1

    def _update_collision_cohort(
        self,
        active_tracks: Sequence[TrackedDetection],
        target: TrackedDetection | None,
    ) -> None:
        """Remember only IDs that physically met the bound target recently."""

        if self._current_timestamp_s > self._collision_expires_at_s:
            self._clear_collision_cohort()
        if target is None or not target.observed:
            return

        related: set[int] = set()
        for item in active_tracks:
            if not item.observed or item.track_id == target.track_id:
                continue
            size = max(
                target.bbox.width,
                target.bbox.height,
                item.bbox.width,
                item.bbox.height,
            )
            near_margin = max(
                4.0,
                min(24.0, size * COLLISION_NEAR_MARGIN_SCALE),
            )
            if self._box_gap(target.bbox, item.bbox) <= near_margin:
                related.add(item.track_id)

        if related:
            self._collision_candidate_ids.add(target.track_id)
            self._collision_candidate_ids.update(related)
            self._collision_contact_epochs[target.track_id] = self._detection_epoch
            for track_id in related:
                self._collision_contact_epochs[track_id] = self._detection_epoch
            self._collision_expires_at_s = max(
                self._collision_expires_at_s,
                self._current_timestamp_s + COLLISION_COHORT_RETENTION_S,
            )

    def _eligible_global_ids(self) -> set[int]:
        eligible = set(self._branch_candidate_ids)
        if self._current_timestamp_s <= self._collision_expires_at_s:
            eligible.update(self._collision_candidate_ids)
        return eligible

    def _branch_gate_radius(self) -> float:
        # Promoted ai_00002: allow wider merge/split recovery branches.
        if self._last_bbox is None:
            return 44.0
        return max(
            44.0,
            min(220.0, max(self._last_bbox.width, self._last_bbox.height) * 0.98),
        )

    def _capture_branch_candidates(
        self,
        active_tracks: Sequence[TrackedDetection],
        target: TrackedDetection | None,
    ) -> None:
        if self._detection_epoch > self._branch_expires_at:
            self._clear_branch_episode()
            return

        if self._target_track_id is not None:
            self._branch_candidate_ids.add(self._target_track_id)

        if self._detection_epoch > self._branch_capture_until:
            return

        # While the merge still looks like one branch, follow its latest
        # position so a later split is captured where it actually happens.
        if len(self._branch_candidate_ids) <= 1 and target is not None and target.observed:
            self._branch_anchor = target.bbox.center
        if self._branch_anchor is None:
            return

        radius = self._branch_gate_radius()
        for item in active_tracks:
            if not item.observed:
                continue
            distance = float(
                np.hypot(
                    item.bbox.center[0] - self._branch_anchor[0],
                    item.bbox.center[1] - self._branch_anchor[1],
                )
            )
            if distance <= radius:
                self._branch_candidate_ids.add(item.track_id)

    def _propagate_missing_branch_lineage(
        self,
        active_tracks: Sequence[TrackedDetection],
        target: TrackedDetection | None,
    ) -> None:
        """Carry branch authority across a spatially continuous OC-SORT ID.

        The transfer is open only while the protected target is explicitly
        missing and only from a branch already captured by the local
        merge/loss event.  This repairs ID fragmentation without granting a
        distant global outlier takeover rights during normal tracking.
        """

        if self._detection_epoch > self._branch_expires_at:
            self._branch_lineage_boxes.clear()
            return

        observed = [item for item in active_tracks if item.observed]
        stale_before = self._detection_epoch - MISSING_LINEAGE_MAX_GAP_EPOCHS
        previous = {
            track_id: sample
            for track_id, sample in self._branch_lineage_boxes.items()
            if sample[0] >= stale_before
        }

        # Freeze the predecessor set before considering newcomers.  This
        # prevents several adjacent boxes in one frame from forming a chain.
        refreshed = dict(previous)
        for item in observed:
            if item.track_id in self._branch_candidate_ids:
                refreshed[item.track_id] = (self._detection_epoch, item.bbox)

        target_observed = target is not None and target.observed
        if target_observed or not self._identity_recovery_enabled or not previous:
            self._branch_lineage_boxes = refreshed
            return

        inherited: list[TrackedDetection] = []
        for item in observed:
            if item.track_id in self._branch_candidate_ids:
                continue
            for _, predecessor in previous.values():
                size = max(
                    predecessor.width,
                    predecessor.height,
                    item.bbox.width,
                    item.bbox.height,
                )
                near_margin = max(
                    4.0,
                    min(30.0, size * MISSING_LINEAGE_NEAR_MARGIN_SCALE),
                )
                center_limit = max(
                    18.0,
                    min(72.0, size * MISSING_LINEAGE_CENTER_SCALE),
                )
                center_distance = float(
                    np.hypot(
                        item.bbox.center[0] - predecessor.center[0],
                        item.bbox.center[1] - predecessor.center[1],
                    )
                )
                area_ratio = item.bbox.area / max(1.0, predecessor.area)
                if (
                    0.40 <= area_ratio <= 2.50
                    and center_distance <= center_limit
                    and (
                        self._bbox_iou(predecessor, item.bbox) >= 0.03
                        or self._box_gap(predecessor, item.bbox) <= near_margin
                    )
                ):
                    inherited.append(item)
                    break

        for item in inherited:
            self._branch_candidate_ids.add(item.track_id)
            refreshed[item.track_id] = (self._detection_epoch, item.bbox)
        self._branch_lineage_boxes = refreshed

    def _update_recovery_arming(
        self,
        active_tracks: Sequence[TrackedDetection],
        target: TrackedDetection | None,
    ) -> None:
        previous_armed_until = self._recovery_armed_until
        previous_boxes = self._previous_active_boxes
        self._refresh_opening_highlight(active_tracks)
        super()._update_recovery_arming(active_tracks, target)

        # The opening highlight is the only trustworthy identity evidence.
        # Build motion history during the countdown, but never let a dimmed
        # highlight or a passing fake take over the target.  We still capture
        # a nearby branch while the highlight is fading; only the decision
        # method below is gated during the opening authority window.

        self._update_collision_cohort(active_tracks, target)

        event = self._recovery_event_detected(
            active_tracks, target, previous_boxes=previous_boxes,
        )
        target_observed = target is not None and target.observed
        target_loss_edge = (
            not target_observed
            and self._recovery_target_was_observed is not False
        )
        event_edge = (event and not self._recovery_event_active) or target_loss_edge
        if event and not self._recovery_event_active:
            self._recovery_event_started_epoch = self._detection_epoch
        elif not event:
            self._recovery_event_started_epoch = -1
        self._recovery_event_active = event
        if event_edge:
            self._clear_branch_episode()
            self._start_continuity_episode()
            # A later collision/loss starts a new local capture around the
            # current position.  Keeping the first episode's anchor would
            # make a valid branch at the other side of the ROI invisible.
            self._branch_anchor = (
                target.bbox.center
                if target is not None and target.observed
                else self._last_aim
            )
            self._branch_capture_until = self._detection_epoch + BRANCH_CAPTURE_FRAMES
            self._branch_expires_at = self._detection_epoch + BRANCH_RETENTION_FRAMES

        self._recovery_target_was_observed = target_observed

        if (
            self._branch_expires_at >= 0
            and self._detection_epoch > self._branch_expires_at
        ):
            self._clear_branch_episode()

        newly_armed = self._recovery_armed_until > previous_armed_until
        episode_active = self._detection_epoch <= self._branch_expires_at
        if newly_armed and not episode_active:
            anchor = (
                target.bbox.center
                if target is not None and target.observed
                else self._last_aim
            )
            self._branch_anchor = anchor
            self._branch_candidate_ids = (
                {self._target_track_id}
                if self._target_track_id is not None
                else set()
            )
            self._branch_capture_until = (
                self._detection_epoch + BRANCH_CAPTURE_FRAMES
            )
            self._branch_expires_at = (
                self._detection_epoch + BRANCH_RETENTION_FRAMES
            )
        if self._detection_epoch <= self._branch_expires_at:
            self._capture_branch_candidates(active_tracks, target)
            self._propagate_missing_branch_lineage(active_tracks, target)

    def _recovery_event_detected(
        self,
        active_tracks: Sequence[TrackedDetection],
        target: TrackedDetection | None,
        *,
        previous_boxes: dict[int, Detection] | None = None,
    ) -> bool:
        """Detect a possible exchange without requiring the base dim lock."""

        if target is None or not target.observed:
            return (
                self._target_track_id is not None
                and self._recovery_target_was_observed is not False
            )
        if self._last_bbox is not None:
            old_area = max(1.0, self._last_bbox.area)
            area_ratio = target.bbox.area / old_area
            if area_ratio > 1.35 or area_ratio < 0.70:
                return True
        for item in active_tracks:
            if item.track_id == target.track_id or not item.observed:
                continue
            if self._bbox_iou(target.bbox, item.bbox) >= 0.03:
                return True
            size = max(
                target.bbox.width,
                target.bbox.height,
                item.bbox.width,
                item.bbox.height,
            )
            if self._box_gap(target.bbox, item.bbox) <= max(
                4.0,
                min(24.0, size * COLLISION_NEAR_MARGIN_SCALE),
            ):
                return True
        current_ids = {item.track_id for item in active_tracks if item.observed}
        nearby_radius = max(
            30.0,
            min(
                160.0,
                max(target.bbox.width, target.bbox.height) * 0.70,
            ),
        )
        if previous_boxes is None:
            previous_boxes = self._previous_active_boxes
        for track_id in set(previous_boxes) - current_ids:
            if track_id == self._target_track_id:
                continue
            old = previous_boxes[track_id]
            if (
                float(np.hypot(
                    old.center[0] - target.bbox.center[0],
                    old.center[1] - target.bbox.center[1],
                ))
                <= nearby_radius
            ):
                return True
        return False

    def _motion_rebind_candidate(
        self,
        active_tracks: Sequence[TrackedDetection],
        target: TrackedDetection | None,
    ) -> TrackedDetection | None:
        opening = self._opening_highlight_candidate_result
        if opening is not None:
            self._clear_branch_episode()
            self._clear_collision_cohort()
            self._clear_supervisor_decision(clear_beliefs=True)
            self._recovery_event_active = False
            return opening
        edge_decision = self._edge_identity_audit.decide(
            self._current_timestamp_s, self._detection_epoch, active_tracks,
            target=target, target_id=self._target_track_id,
            last_box=self._last_bbox, eligible_ids=self._eligible_global_ids(),
            enabled=self._identity_recovery_enabled and not self._opening_highlight_active,
        )
        if edge_decision.candidate is not None:
            # This narrow boundary exception has its own confidence, visible
            # extent, local provenance, majority and temporal evidence gates.
            # Keep the generic full-box takeover policy unchanged elsewhere.
            self._clear_branch_episode()
            self._clear_collision_cohort()
            self._clear_supervisor_decision(clear_beliefs=True)
            self._recovery_event_active = False
            return edge_decision.candidate
        if edge_decision.pending:
            return None
        if target is None or not target.observed:
            self._clear_recovery_vector_decision()
            self._template_candidate_id = None
            self._template_candidate_count = 0
            self._clear_supervisor_decision()
            provisional = self._missing_spatial_rebind_candidate(
                active_tracks,
                target,
            )
            candidate = self._missing_target_compass_rebind_candidate(
                active_tracks
            )
            candidate_since = self._missing_spatial_candidate_since_s
            if (
                candidate is None
                and provisional is not None
                and candidate_since is not None
                and self._current_timestamp_s - candidate_since
                >= MISSING_SPATIAL_COMMIT_S
            ):
                candidate = provisional
            if candidate is not None:
                self._clear_branch_episode()
                self._clear_collision_cohort()
                self._clear_supervisor_decision(clear_beliefs=True)
                self._recovery_event_active = False
            elif provisional is not None:
                self._provisional_output = provisional
            return candidate
        self._clear_missing_spatial_decision()
        self._clear_missing_compass_decision()
        # Keep the opening lock authoritative while the one-shot highlighter is
        # still active.  Once the highlight has faded, the branch/motion
        # arbiters may use the collected branch evidence to repair an exchange.
        # This prevents an early OC-SORT label wobble without freezing a stale
        # identity for the entire five-second opening window.
        if self.opening_identity_protected and self._opening_highlight_active:
            self._template_candidate_id = None
            self._template_candidate_count = 0
            self._clear_recovery_vector_decision()
            self._clear_supervisor_decision()
            return None
        candidate = self._continuity_rebind_candidate(active_tracks, target)
        template_checked = False
        # A boundary-clipped current box is already semantically disqualified
        # from being the real target.  Let the mature collective-motion
        # template audit that case before the generic recovery-vector guard;
        # otherwise a noisy short vector can suppress the stronger evidence
        # until the clipped fake has travelled far outside the true target.
        if (
            candidate is None
            and target is not None
            and target.observed
            and not self._bbox_is_full(target)
        ):
            template_checked = True
            candidate = self._template_rebind_candidate(active_tracks, target)
        # A visible target whose short motion is already a strong outlier is
        # the evidence we are trying to preserve.  Do not let one of the
        # older, less selective fallback arbiters replace it merely because a
        # different branch also looks unusual.  This is especially important
        # when a box briefly changes size while the true target is still
        # present: the vector layer rejects a takeover, so the lower layers
        # must not re-introduce an unconditional long-distance jump.
        if candidate is None and target is not None and target.observed:
            target_vector_score = self._recovery_vector_score(
                self._recovery_vectors.get(target.track_id),
                self._recovery_reference_vector,
            )
            if (
                target_vector_score is not None
                and target_vector_score > RECOVERY_VECTOR_TARGET_MAX_SCORE
            ):
                self._template_candidate_id = None
                self._template_candidate_count = 0
                self._clear_recovery_vector_decision()
                self._clear_supervisor_decision()
                return None
        if candidate is None:
            candidate = self._recovery_vector_rebind_candidate(active_tracks, target)
        if candidate is None and not self.opening_identity_protected:
            candidate = self._collision_compass_rebind_candidate(
                active_tracks,
                target,
            )
        # During the opening protection window, only the opening highlighter
        # and this experiment's vector evidence may repair identity.  The
        # legacy template/global/distant layers are deliberately held back so
        # they cannot turn an early OC-SORT wobble into a long-distance jump.
        if (candidate is None and not self.opening_identity_protected
                and not template_checked):
            candidate = self._template_rebind_candidate(active_tracks, target)
        # Once the one-shot opening highlight has genuinely faded, the strict
        # global auditor may repair a collision-proven ID exchange even while
        # the conservative five-second opening timer is still running.  The
        # early return above keeps the highlight authoritative while it is
        # visible, and the auditor still requires a fake majority, a unique
        # outlier, recent contact provenance, and repeated evidence.  Keep the
        # broader template/distant/base fallbacks behind the full timer.
        if candidate is None:
            candidate = self._global_supervisor_rebind_candidate(
                active_tracks,
                target,
            )
        if candidate is None and not self.opening_identity_protected:
            candidate = self._distant_branch_rebind_candidate(active_tracks, target)
        if candidate is None and not self.opening_identity_protected:
            candidate = super()._motion_rebind_candidate(active_tracks, target)
        if candidate is not None and not self._candidate_can_take_over(candidate):
            candidate = None
        if candidate is not None and candidate.track_id != self._target_track_id:
            self._clear_branch_episode()
            self._clear_collision_cohort()
            self._clear_supervisor_decision(clear_beliefs=True)
            # A successful repair closes the triggering episode.  The next
            # collision or loss must be allowed to create a fresh edge; if the
            # old event flag stayed set, a later loss would never start a new
            # branch and the stale IDs could be retained indefinitely.
            self._recovery_event_active = False
        return candidate

    def _missing_target_compass_rebind_candidate(
        self,
        active_tracks: Sequence[TrackedDetection],
    ) -> TrackedDetection | None:
        """Abstain during a merge unless one replacement is strongly proven.

        A predicted OC-SORT identity can disappear for a handful of frames
        while the same physical target is still inside a merged rectangle.  A
        merely visible branch is therefore not sufficient evidence to switch.
        Recovery while the bound ID is unobserved requires a recent collision
        candidate to be the sole strong compass outlier for both multiple
        frames and a wall-clock interval.  Otherwise the tracker keeps coasting
        and lets the original ID resume when the split settles.
        """

        if not self._identity_recovery_enabled:
            self._clear_missing_compass_decision()
            return None
        missing_age = (
            float("inf")
            if self._last_observed_time is None
            else max(0.0, self._current_timestamp_s - self._last_observed_time)
        )
        if missing_age <= self.coast_seconds:
            self._clear_missing_compass_decision()
            return None

        observed = {
            item.track_id: item for item in active_tracks if item.observed
        }
        scored = {
            item.track_id: item.deviation_score
            for item in self._motion_compasses
            if item.deviation_score is not None and item.track_id in observed
        }
        if len(scored) < 3:
            self._clear_missing_compass_decision()
            return None
        ordered = sorted(scored.items(), key=lambda item: item[1], reverse=True)
        winner_id, winner_score = ordered[0]
        runner_score = ordered[1][1]
        if (
            winner_score < MISSING_COMPASS_MIN_SCORE
            or winner_score - runner_score < self.identity_tuning.missing_margin
            or winner_id not in self._eligible_global_ids()
        ):
            self._clear_missing_compass_decision()
            return None

        fake_like = sum(
            score <= COLLISION_COMPASS_TARGET_MAX_SCORE
            for score in scored.values()
        )
        if fake_like < 2 or fake_like <= len(scored) / 2:
            self._clear_missing_compass_decision()
            return None

        winner = observed[winner_id]
        if not self._candidate_can_take_over(winner):
            self._clear_missing_compass_decision()
            return None
        consecutive = (
            self._missing_compass_candidate_id == winner_id
            and self._missing_compass_candidate_last_epoch
            == self._detection_epoch - 1
        )
        if consecutive:
            self._missing_compass_candidate_count += 1
        else:
            self._missing_compass_candidate_id = winner_id
            self._missing_compass_candidate_count = 1
            self._missing_compass_candidate_since_s = self._current_timestamp_s
        self._missing_compass_candidate_last_epoch = self._detection_epoch
        confirmation_age = (
            0.0
            if self._missing_compass_candidate_since_s is None
            else max(
                0.0,
                self._current_timestamp_s - self._missing_compass_candidate_since_s,
            )
        )
        if (
            self._missing_compass_candidate_count
            < MISSING_COMPASS_CONFIRMATIONS
            or confirmation_age < MISSING_COMPASS_CONFIRMATION_S
        ):
            return None
        return winner

    def _collision_compass_rebind_candidate(
        self,
        active_tracks: Sequence[TrackedDetection],
        target: TrackedDetection | None,
    ) -> TrackedDetection | None:
        """Repair a late split using only recent collision-proven identities.

        A long-lived overlap can keep the recovery event continuously active,
        so the short branch-capture window may expire before the meaningful
        split happens.  This audit does not reopen global takeover authority:
        it considers only IDs that physically met the target recently, then
        requires one unique compass outlier while the currently bound ID moves
        with the fake majority.  Two time-separated confirmations prevent a
        single jittery rectangle from changing identity.
        """

        if (
            target is None
            or not target.observed
            or not self._identity_recovery_enabled
            or not self._motion_template.ready
            or not self._recovery_event_active
            or self._recovery_event_started_epoch < 0
            or self._detection_epoch - self._recovery_event_started_epoch
            <= BRANCH_CAPTURE_FRAMES
            or len(self._branch_candidate_ids) >= 2
        ):
            self._clear_collision_compass_decision()
            return None

        observed = {
            item.track_id: item for item in active_tracks if item.observed
        }
        scored = {
            item.track_id: item.deviation_score
            for item in self._motion_compasses
            if item.deviation_score is not None and item.track_id in observed
        }
        if len(scored) < 3:
            self._clear_collision_compass_decision()
            return None

        target_score = scored.get(target.track_id)
        if (
            target_score is None
            or target_score > COLLISION_COMPASS_TARGET_MAX_SCORE
        ):
            self._clear_collision_compass_decision()
            return None

        ordered = sorted(scored.items(), key=lambda item: item[1], reverse=True)
        winner_id, winner_score = ordered[0]
        runner_score = ordered[1][1]
        if winner_id == target.track_id:
            self._clear_collision_compass_decision()
            return None
        if (
            winner_score < COLLISION_COMPASS_MIN_SCORE
            or winner_score - runner_score < self.identity_tuning.collision_margin
        ):
            self._clear_collision_compass_decision()
            return None

        fake_like = sum(
            score <= COLLISION_COMPASS_TARGET_MAX_SCORE
            for score in scored.values()
        )
        if fake_like < 2 or fake_like <= len(scored) / 2:
            self._clear_collision_compass_decision()
            return None

        contact_epoch = self._collision_contact_epochs.get(winner_id)
        if (
            contact_epoch is None
            or self._detection_epoch - contact_epoch
            > COLLISION_COMPASS_CONTACT_MAX_EPOCHS
        ):
            self._clear_collision_compass_decision()
            return None
        winner = observed[winner_id]
        if not self._candidate_can_take_over(winner):
            self._clear_collision_compass_decision()
            return None

        consecutive = (
            self._collision_compass_candidate_id == winner_id
            and self._collision_compass_candidate_last_epoch
            == self._detection_epoch - 1
        )
        if consecutive:
            self._collision_compass_candidate_count += 1
        else:
            self._collision_compass_candidate_id = winner_id
            self._collision_compass_candidate_count = 1
            self._collision_compass_candidate_since_s = self._current_timestamp_s
        self._collision_compass_candidate_last_epoch = self._detection_epoch

        confirmation_age = (
            0.0
            if self._collision_compass_candidate_since_s is None
            else max(
                0.0,
                self._current_timestamp_s
                - self._collision_compass_candidate_since_s,
            )
        )
        if (
            self._collision_compass_candidate_count
            < COLLISION_COMPASS_CONFIRMATIONS
            or confirmation_age < COLLISION_COMPASS_CONFIRMATION_S
        ):
            return None
        return winner

    def _recovery_vector_rebind_candidate(
        self,
        active_tracks: Sequence[TrackedDetection],
        target: TrackedDetection | None,
    ) -> TrackedDetection | None:
        """Use short majority-vector evidence to recover a captured branch.

        OC-SORT IDs are only labels here.  A candidate can be considered from
        any distance after the local event captured its ID, but it still needs
        a unique, repeated deviation from the majority vector.  This keeps a
        far-away unrelated box from receiving takeover authority.
        """

        if (
            self._detection_epoch > self._branch_expires_at
            or len(self._branch_candidate_ids) < 2
            or self._recovery_reference_vector is None
        ):
            self._clear_recovery_vector_decision()
            return None

        scored: list[tuple[float, TrackedDetection]] = []
        for item in active_tracks:
            if (
                not item.observed
                or item.track_id not in self._branch_candidate_ids
                or not self._candidate_can_take_over(item)
            ):
                continue
            score = self._recovery_vector_score(
                self._recovery_vectors.get(item.track_id),
                self._recovery_reference_vector,
            )
            if score is not None:
                scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        if not scored:
            self._clear_recovery_vector_decision()
            return None

        winner_score, winner = scored[0]
        runner_score = scored[1][0] if len(scored) > 1 else 0.0
        target_score = (
            self._recovery_vector_score(
                self._recovery_vectors.get(target.track_id),
                self._recovery_reference_vector,
            )
            if target is not None and target.observed
            else None
        )
        if winner.track_id == self._target_track_id:
            self._clear_recovery_vector_decision()
            return None
        if winner_score < RECOVERY_VECTOR_OUTLIER_SCORE:
            self._clear_recovery_vector_decision()
            return None
        if len(scored) > 1 and winner_score - runner_score < RECOVERY_VECTOR_MIN_MARGIN:
            self._clear_recovery_vector_decision()
            return None
        if target is not None and target.observed:
            if target_score is None or target_score > RECOVERY_VECTOR_TARGET_MAX_SCORE:
                self._clear_recovery_vector_decision()
                return None

        if winner.track_id == self._recovery_vector_candidate_id:
            self._recovery_vector_candidate_count += 1
        else:
            self._recovery_vector_candidate_id = winner.track_id
            self._recovery_vector_candidate_count = 1
            self._recovery_vector_candidate_since_s = self._current_timestamp_s
        elapsed = (
            0.0
            if self._recovery_vector_candidate_since_s is None
            else max(
                0.0,
                self._current_timestamp_s - self._recovery_vector_candidate_since_s,
            )
        )
        if (
            self._recovery_vector_candidate_count < RECOVERY_VECTOR_CONFIRMATIONS
            or elapsed < RECOVERY_VECTOR_CONFIRMATION_S
        ):
            return None
        return winner

    def _template_rebind_candidate(
        self,
        active_tracks: Sequence[TrackedDetection],
        target: TrackedDetection | None,
    ) -> TrackedDetection | None:
        """Use the matured collective template to audit a known branch.

        Every visible track is ranked for diagnostics, but switching is limited
        to the branch cohort already established by the existing recovery
        arming logic.  This permits a distant correction after a real crossing
        without promoting an unrelated far-away outlier to the target.
        """

        template = self._motion_template
        target_is_edge_clipped = (
            target is not None
            and target.observed
            and not self._bbox_is_full(target)
        )
        candidate_ids = (
            self._eligible_global_ids()
            if target_is_edge_clipped
            else self._branch_candidate_ids
        )
        if (
            not template.ready
            or target is None
            or not target.observed
            or len(candidate_ids) < 2
        ):
            self._template_candidate_id = None
            self._template_candidate_count = 0
            return None

        by_id = {
            item.track_id: item
            for item in self._motion_compasses
            if item.deviation_score is not None
            and item.track_id in candidate_ids
        }
        target_compass = by_id.get(target.track_id)
        if target_compass is None or target_compass.deviation_score is None:
            self._template_candidate_id = None
            self._template_candidate_count = 0
            return None

        ranked = sorted(
            (
                compass
                for compass in by_id.values()
                if compass.track_id != target.track_id
                and compass.deviation_score is not None
            ),
            key=lambda compass: float(compass.deviation_score),
            reverse=True,
        )
        if not ranked:
            self._template_candidate_id = None
            self._template_candidate_count = 0
            return None

        winner = ranked[0]
        runner_score = (
            float(ranked[1].deviation_score)
            if len(ranked) > 1 and ranked[1].deviation_score is not None
            else 0.0
        )
        target_score = float(target_compass.deviation_score)
        winner_score = float(winner.deviation_score)
        outlier_score = (
            EDGE_TEMPLATE_OUTLIER_SCORE
            if target_is_edge_clipped
            else MOTION_TEMPLATE_OUTLIER_SCORE
        )
        minimum_margin = (
            EDGE_TEMPLATE_MIN_MARGIN
            if target_is_edge_clipped
            else MOTION_TEMPLATE_MIN_MARGIN
        )
        confirmations = (
            EDGE_TEMPLATE_CONFIRMATIONS
            if target_is_edge_clipped
            else MOTION_TEMPLATE_CONFIRMATIONS
        )
        if (
            target_score > MOTION_TEMPLATE_TARGET_MAX_SCORE
            or winner_score < outlier_score
            or winner_score - max(target_score, runner_score)
            < minimum_margin
        ):
            self._template_candidate_id = None
            self._template_candidate_count = 0
            return None

        if (self._template_candidate_id == winner.track_id
                and self._template_candidate_last_epoch
                in (self._detection_epoch - 1, self._detection_epoch)):
            if self._template_candidate_last_epoch != self._detection_epoch:
                self._template_candidate_count += 1
        else:
            self._template_candidate_id = winner.track_id
            self._template_candidate_count = 1
        self._template_candidate_last_epoch = self._detection_epoch
        if self._template_candidate_count < confirmations:
            return None
        return next(
            (
                item
                for item in active_tracks
                if item.observed and item.track_id == winner.track_id
            ),
            None,
        )

    def _global_supervisor_rebind_candidate(
        self,
        active_tracks: Sequence[TrackedDetection],
        target: TrackedDetection | None,
    ) -> TrackedDetection | None:
        """Audit every identity but authorize only collision-proven candidates.

        OC-SORT IDs are treated as temporary labels.  The supervisor waits for
        a strict fake-motion majority, one unique outlier, a fake-like current
        target, recent collision provenance, and repeated agreement before it
        is allowed to override the bound ID.
        """

        if self._supervisor_last_epoch == self._detection_epoch:
            return None
        self._supervisor_last_epoch = self._detection_epoch

        observed = [item for item in active_tracks if item.observed]
        active_ids = {item.track_id for item in observed}
        for track_id in tuple(self._supervisor_beliefs):
            if track_id not in active_ids:
                faded = self._supervisor_beliefs[track_id] * 0.5
                if faded < 0.05:
                    del self._supervisor_beliefs[track_id]
                else:
                    self._supervisor_beliefs[track_id] = faded

        if not self._identity_recovery_enabled:
            self._clear_supervisor_decision()
            return None

        scored = [
            (item, score)
            for item in observed
            if (score := self._motion_score(item.track_id)) is not None
        ]
        if len(scored) < 3:
            self._clear_supervisor_decision()
            return None

        tolerance = self.motion_tolerance
        outlier_threshold = tolerance * 1.5
        fake_like = [item for item, score in scored if score <= tolerance]
        outliers = [item for item, score in scored if score >= outlier_threshold]

        for item, score in scored:
            evidence = min(
                1.0,
                max(0.0, (score - tolerance) / max(0.5, tolerance * 0.5)),
            )
            previous = self._supervisor_beliefs.get(item.track_id, 0.0)
            weight = self.identity_tuning.evidence_weight
            self._supervisor_beliefs[item.track_id] = (
                previous * (1.0 - weight) + evidence * weight
            )

        strict_fake_majority = (
            len(fake_like) >= 2 and len(fake_like) > len(scored) / 2
        )
        if not strict_fake_majority or len(outliers) != 1:
            self._clear_supervisor_decision()
            return None

        winner = outliers[0]
        if winner.track_id == self._target_track_id:
            self._clear_supervisor_decision()
            return None

        # Global motion remains useful as evidence, but it may only transfer
        # identity to an ID captured by a real merge/overlap recovery episode.
        # An unrelated outlier elsewhere in the frame has no takeover rights.
        if (
            winner.track_id not in self._eligible_global_ids()
        ):
            self._clear_supervisor_decision()
            return None

        # The global layer is an auditor of a visible but wrongly associated
        # target.  A missing target is not proof that the current identity is
        # fake; leave that case to the stricter merge/loss recovery paths.
        if target is None or not target.observed:
            self._clear_supervisor_decision()
            return None
        target_score = self._motion_score(target.track_id)
        if target_score is None or target_score > tolerance:
            self._clear_supervisor_decision()
            return None

        consecutive = (
            self._supervisor_candidate_id == winner.track_id
            and self._supervisor_candidate_last_epoch == self._detection_epoch - 1
        )
        if consecutive:
            self._supervisor_candidate_count += 1
        else:
            self._supervisor_candidate_id = winner.track_id
            self._supervisor_candidate_count = 1
            self._supervisor_candidate_since_s = self._current_timestamp_s
        self._supervisor_candidate_last_epoch = self._detection_epoch

        belief = self._supervisor_beliefs.get(winner.track_id, 0.0)
        confirmation_age = (
            0.0
            if self._supervisor_candidate_since_s is None
            else max(
                0.0,
                self._current_timestamp_s - self._supervisor_candidate_since_s,
            )
        )
        if (
            self._supervisor_candidate_count
            < GLOBAL_SUPERVISOR_CONFIRMATIONS
            or confirmation_age < GLOBAL_SUPERVISOR_CONFIRMATION_S
            or belief < self.identity_tuning.supervisor_belief
        ):
            return None
        return winner

    def _distant_branch_rebind_candidate(
        self,
        active_tracks: Sequence[TrackedDetection],
        target: TrackedDetection | None,
    ) -> TrackedDetection | None:
        """Return a proven branch candidate without applying a distance gate."""

        if (
            self._detection_epoch > self._branch_expires_at
            or len(self._branch_candidate_ids) < 2
        ):
            return None

        retained = [
            item
            for item in active_tracks
            if item.observed and item.track_id in self._branch_candidate_ids
        ]
        target_is_observed = target is not None and target.observed
        if not retained or (target_is_observed and len(retained) < 2):
            return None

        scored = [
            (item, score)
            for item in retained
            if (score := self._motion_score(item.track_id)) is not None
        ]
        if not scored or (target_is_observed and len(scored) < 2):
            return None

        true_threshold = self.motion_tolerance * 1.5
        outliers = [item for item, score in scored if score >= true_threshold]
        if len(outliers) != 1:
            return None

        winner = outliers[0]
        if winner.track_id == self._target_track_id:
            return None
        if target_is_observed:
            target_score = self._motion_score(target.track_id)
            if target_score is None or target_score > self.motion_tolerance:
                return None
        return winner


__all__ = [
    "ASSOCIATION_COMPETITOR_DISTANCE_MARGIN_PX",
    "ASSOCIATION_COMPETITOR_IOU_MARGIN",
    "ASSOCIATION_JUMP_DIAGONAL_SCALE",
    "ASSOCIATION_JUMP_MAX_PX",
    "ASSOCIATION_JUMP_MIN_PX",
    "ASSOCIATION_QUARANTINE_FRAMES",
    "BRANCH_CAPTURE_FRAMES",
    "BRANCH_RETENTION_FRAMES",
    "COLLISION_COHORT_RETENTION_S",
    "COLLISION_NEAR_MARGIN_SCALE",
    "EDGE_TAKEOVER_MARGIN_PX",
    "ExperimentalBranchTargetTracker",
    "GLOBAL_SUPERVISOR_CONFIRMATION_S",
    "GLOBAL_SUPERVISOR_CONFIRMATIONS",
    "GLOBAL_SUPERVISOR_MIN_BELIEF",
    "OPENING_IDENTITY_PROTECTION_S",
    "OPENING_HIGHLIGHT_FADE_FRAMES",
    "OPENING_HIGHLIGHT_REBIND_CONFIRMATIONS",
    "OPENING_HIGHLIGHT_REBIND_MARGIN",
    "OPENING_HIGHLIGHT_REBIND_MINIMUM",
    "RECOVERY_VECTOR_CONFIRMATION_S",
    "RECOVERY_VECTOR_CONFIRMATIONS",
    "RECOVERY_VECTOR_HISTORY_SIZE",
    "RECOVERY_VECTOR_MIN_MARGIN",
    "RECOVERY_VECTOR_MIN_SAMPLES",
    "RECOVERY_VECTOR_MIN_SPEED",
    "RECOVERY_VECTOR_OUTLIER_SCORE",
    "RECOVERY_VECTOR_TARGET_MAX_SCORE",
]
