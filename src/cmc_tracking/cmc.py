"""Estimate camera motion between frames from background points.

Camera motion compensation (CMC) is also known as global motion compensation (GMC). This
follows the common recipe used by trackers such as BoT-SORT: track corner points with sparse
optical flow, then fit a 2D map with RANSAC.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraMotionConfig:
    max_corners: int = 1000
    quality_level: float = 0.01
    min_distance: int = 8
    max_forward_backward_error: float = 1.0
    ransac_threshold: float = 3.0
    min_inliers: int = 30
    min_inlier_ratio: float = 0.5
    max_side: int = 960


@dataclass(frozen=True)
class CameraMotion:
    """affine maps previous-frame coordinates to current-frame coordinates.

    source and target are the tracked points in full-resolution pixels, and inliers marks
    the points that agree with the fit. They are for inspection and plotting.
    """

    affine: np.ndarray
    n_tracked: int
    n_inliers: int
    reliable: bool
    source: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    target: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    inliers: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))


IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _background_mask(shape: tuple[int, int], boxes: np.ndarray | None, scale: float) -> np.ndarray:
    mask = np.full(shape, 255, dtype=np.uint8)
    if boxes is not None:
        for x1, y1, x2, y2 in np.asarray(boxes).reshape(-1, 4) * scale:
            mask[max(int(y1), 0) : max(int(y2), 0), max(int(x1), 0) : max(int(x2), 0)] = 0
    return mask


def estimate_camera_motion(
    previous: np.ndarray,
    current: np.ndarray,
    foreground_boxes: np.ndarray | None = None,
    config: CameraMotionConfig | None = None,
) -> CameraMotion:
    """Estimate the map between two grayscale frames, ignoring points inside foreground_boxes.

    The result is the identity map with reliable=False when too few points agree, because a
    bad correction moves every track and does more harm than no correction.
    """
    # Limitations:
    # - The map is a similarity (shift, rotation, zoom), so it cannot model parallax.
    # - It assumes most tracked points lie on the static background. A large moving object
    #   that is not in foreground_boxes can win the fit.
    # - Thresholds are defaults, not tuned per dataset.
    config = config or CameraMotionConfig()
    scale = min(1.0, config.max_side / max(previous.shape))
    if scale < 1.0:
        size = (round(previous.shape[1] * scale), round(previous.shape[0] * scale))
        previous = cv2.resize(previous, size, interpolation=cv2.INTER_AREA)
        current = cv2.resize(current, size, interpolation=cv2.INTER_AREA)

    corners = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=config.max_corners,
        qualityLevel=config.quality_level,
        minDistance=config.min_distance,
        mask=_background_mask(previous.shape, foreground_boxes, scale),
    )
    if corners is None:
        return CameraMotion(IDENTITY.copy(), 0, 0, False)

    forward, ok_forward, _ = cv2.calcOpticalFlowPyrLK(previous, current, corners, None)
    backward, ok_backward, _ = cv2.calcOpticalFlowPyrLK(current, previous, forward, None)
    error = np.linalg.norm((corners - backward).reshape(-1, 2), axis=1)
    keep = ok_forward.ravel().astype(bool) & ok_backward.ravel().astype(bool)
    keep &= error < config.max_forward_backward_error
    source, target = corners.reshape(-1, 2)[keep], forward.reshape(-1, 2)[keep]
    if len(source) < config.min_inliers:
        return CameraMotion(IDENTITY.copy(), int(len(source)), 0, False)

    affine, inliers = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=config.ransac_threshold
    )
    n_inliers = 0 if inliers is None else int(inliers.sum())
    points = {
        "source": source / scale,
        "target": target / scale,
        "inliers": np.zeros(len(source), dtype=bool) if inliers is None else inliers.ravel() > 0,
    }
    reliable = (
        affine is not None
        and n_inliers >= config.min_inliers
        and n_inliers / len(source) >= config.min_inlier_ratio
    )
    if not reliable:
        return CameraMotion(IDENTITY.copy(), int(len(source)), n_inliers, False, **points)

    # Convert the translation back to full-resolution pixels.
    affine = affine.astype(np.float64)
    affine[:, 2] /= scale
    return CameraMotion(affine, int(len(source)), n_inliers, True, **points)
