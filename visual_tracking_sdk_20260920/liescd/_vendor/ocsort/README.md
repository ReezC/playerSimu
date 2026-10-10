# Vendored OC-SORT

These three Python files are copied from the author-maintained repository:

- Source: https://github.com/noahcao/OC_SORT
- Pinned commit: `8462e7e729a93ccd3bd995c0a79a890336cb3a0b`
- License: see [`LICENSE`](LICENSE) (MIT)

The pinned source is the baseline, with these local runtime extensions:

- Each OCSort instance owns its ID counter. Creating/resetting another session
  cannot reset IDs still being allocated inside an active session.
- Optional `dt` advances the Kalman transition and process uncertainty using
  elapsed source time in session reference-step units. Track expiry uses the
  same elapsed units, while observation counts still count actual observations.
- Observation-centric recovery retains each transition and interpolates its
  virtual bridge using elapsed time, including unequal gaps. Virtual bridge
  observations remain internal to the filter and do not confirm target identity.

Default `dt=1` preserves the fixed-step path. `liescd/ocsort_tracker.py` supplies
the session clock; these extensions are not claimed to be upstream features.

`kalmanfilter.py` is the upstream FilterPy-derived implementation and retains
its own attribution/license notice in the source; the runtime dependency is
declared directly as `filterpy` in the project metadata.
