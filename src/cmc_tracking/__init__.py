"""OC-SORT multi-object tracking with optional camera motion compensation."""

from .cmc import CameraMotion, CameraMotionConfig, estimate_camera_motion
from .ocsort import OCSort, OCSortConfig

__all__ = [
    "CameraMotion",
    "CameraMotionConfig",
    "OCSort",
    "OCSortConfig",
    "estimate_camera_motion",
]

__version__ = "0.1.0"
