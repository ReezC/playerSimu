"""Stable runtime tracker selection shared by every LiesCD entry point.

The original :class:`~liescd.ocsort_tracker.OCSortTargetTracker` remains the
rollback-capable OC-SORT base.  Runtime callers import ``LiescdTargetTracker``
from this module so the standalone GUI, local controller, and physical GPU
host cannot silently drift onto different identity-binding algorithms.
"""

from liescd.experimental.branch_tracker import ExperimentalBranchTargetTracker


# User-visible algorithm generation.  UI surfaces import this value from the
# same module that selects the actual runtime class, so the label cannot drift
# from the implementation without changing this selection point.
LIESCD_TRACKER_VERSION = "V2.5.1"

# Shared OC-SORT age budget; never derive it from a video's container FPS.
LIESCD_TRACK_BUFFER = 60

# Promote the validated branch/motion-consensus tracker without copying its
# implementation.  Keeping this stable semantic name also lets the old base
# remain available for a narrow rollback if live evidence requires one.
LiescdTargetTracker = ExperimentalBranchTargetTracker


__all__ = ["LIESCD_TRACKER_VERSION", "LIESCD_TRACK_BUFFER", "LiescdTargetTracker"]
