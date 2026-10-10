"""Shared numerical contract for GUI, local and remote geometry inference."""
from __future__ import annotations

import os


GEOMETRY_INFERENCE_PROFILE = "geometry-fp32-rect640-batch1-v1"


def configure_geometry_environment() -> None:
    # Must run before the process performs its first convolution. Explicit
    # diagnostic environment overrides remain visible in the runtime profile.
    os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "0")


def geometry_predict_options(*, confidence: float, image_size: int = 640,
                             device=0) -> dict:
    return dict(conf=confidence, imgsz=image_size, device=device, quantize=32,
                classes=[0], verbose=False, rect=True)


def geometry_runtime_profile() -> dict:
    return {"id": GEOMETRY_INFERENCE_PROFILE, "precision": "fp32", "batch": 1,
            "cudnn_v8_disabled": os.environ.get("TORCH_CUDNN_V8_API_DISABLED", "0")}


def geometry_profile_compatible(profile) -> bool:
    """Reject old hosts and diagnostic backends that diverge from the GUI."""
    expected = {"id": GEOMETRY_INFERENCE_PROFILE, "precision": "fp32", "batch": 1,
                "cudnn_v8_disabled": "0"}
    return isinstance(profile, dict) and all(profile.get(k) == v for k, v in expected.items())
