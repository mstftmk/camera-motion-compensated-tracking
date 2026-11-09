import cv2
import numpy as np
import pytest
from conftest import warp

from cmc_tracking import CameraMotionConfig, estimate_camera_motion


def affine_from(dx=0.0, dy=0.0, angle_deg=0.0, scale=1.0, center=(480.0, 360.0)):
    angle = np.deg2rad(angle_deg)
    linear = scale * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    offset = np.array(center) - linear @ np.array(center) + np.array([dx, dy])
    return np.column_stack([linear, offset])


def mean_point_error(estimated, true, shape=(720, 960)):
    ys, xs = np.mgrid[100 : shape[0] - 100 : 40, 100 : shape[1] - 100 : 40]
    points = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    a = points @ estimated[:, :2].T + estimated[:, 2]
    b = points @ true[:, :2].T + true[:, 2]
    return float(np.linalg.norm(a - b, axis=1).mean())


@pytest.mark.parametrize(
    "true",
    [
        affine_from(dx=12.5, dy=-7.25),
        affine_from(dx=2.0, angle_deg=1.5),
        affine_from(scale=1.03),
        affine_from(dx=-5.0, dy=3.0, angle_deg=-1.0, scale=0.98),
    ],
    ids=["translation", "rotation", "zoom", "combined"],
)
def test_recovers_known_camera_motion_within_half_a_pixel(background, true):
    motion = estimate_camera_motion(background, warp(background, true))
    assert motion.reliable
    assert mean_point_error(motion.affine, true) < 0.5


def test_no_motion_gives_the_identity(background):
    motion = estimate_camera_motion(background, background)
    assert motion.reliable
    np.testing.assert_allclose(motion.affine, [[1, 0, 0], [0, 1, 0]], atol=1e-3)


def test_downscaled_estimation_returns_full_resolution_translation(background):
    true = affine_from(dx=12.5, dy=-7.25)
    motion = estimate_camera_motion(
        background, warp(background, true), config=CameraMotionConfig(max_side=480)
    )
    assert motion.reliable
    assert mean_point_error(motion.affine, true) < 1.0


def test_moving_objects_do_not_pull_the_estimate_when_masked(background):
    # A large textured object covers half the frame and moves against the camera.
    true = affine_from(dx=6.0)
    previous, current = background.copy(), warp(background, true)
    patch = np.flipud(background)[100:600, 100:500]
    previous[100:600, 100:500] = patch
    current[100:600, 130:530] = patch
    boxes = np.array([[90, 90, 540, 610]], dtype=float)
    motion = estimate_camera_motion(previous, current, foreground_boxes=boxes)
    assert motion.reliable
    assert mean_point_error(motion.affine, true) < 0.5


def test_flat_image_is_unreliable_and_returns_the_identity():
    flat = np.full((240, 320), 128, dtype=np.uint8)
    motion = estimate_camera_motion(flat, flat)
    assert not motion.reliable
    np.testing.assert_array_equal(motion.affine, [[1, 0, 0], [0, 1, 0]])


def test_unrelated_frames_are_unreliable(background):
    other = np.random.default_rng(5).integers(0, 255, background.shape, dtype=np.uint8)
    motion = estimate_camera_motion(background, other)
    assert not motion.reliable
    np.testing.assert_array_equal(motion.affine, [[1, 0, 0], [0, 1, 0]])


def test_forward_backward_check_is_applied(background):
    motion = estimate_camera_motion(
        background,
        warp(background, affine_from(dx=3.0)),
        config=CameraMotionConfig(max_forward_backward_error=0.0),
    )
    assert not motion.reliable
    assert motion.n_tracked == 0


def test_low_inlier_ratio_is_unreliable_and_returns_the_identity(background):
    motion = estimate_camera_motion(
        background,
        warp(background, affine_from(dx=3.0)),
        config=CameraMotionConfig(min_inlier_ratio=1.1),
    )
    assert not motion.reliable
    assert motion.n_inliers > 0
    np.testing.assert_array_equal(motion.affine, [[1, 0, 0], [0, 1, 0]])


def test_mask_stops_a_dominant_textured_object_from_winning():
    # Nearly all strong corners sit on the object, so only the mask reveals the camera motion.
    rng = np.random.default_rng(1)
    faint = rng.random((720, 960)).astype(np.float32) * 255
    faint = cv2.GaussianBlur(faint, (0, 0), 4.0)
    faint = np.clip(120 + (faint - faint.mean()) * 4.0, 0, 255).astype(np.uint8)
    true = affine_from(dx=8.0, dy=-3.0)
    previous, current = faint.copy(), warp(faint, true)
    sharp = rng.integers(0, 255, (560, 760), dtype=np.uint8)
    sharp = cv2.GaussianBlur(sharp, (0, 0), 1.5)
    sharp = np.clip((sharp.astype(np.float32) - 127) * 6 + 127, 0, 255).astype(np.uint8)
    previous[80:640, 100:860] = sharp
    current[60:620, 130:890] = sharp
    boxes = np.array([[90, 50, 900, 650]], dtype=float)
    masked = estimate_camera_motion(previous, current, foreground_boxes=boxes)
    unmasked = estimate_camera_motion(previous, current)
    assert masked.reliable
    assert abs(masked.affine[0, 2] - 8.0) < 0.5
    assert abs(masked.affine[1, 2] + 3.0) < 0.5
    # Without the mask the object's points dominate, so the true motion is not recovered.
    assert abs(unmasked.affine[0, 2] - 8.0) > 1.5


def test_tracked_points_are_returned_in_full_resolution(background):
    true = affine_from(dx=12.0, dy=-6.0)
    motion = estimate_camera_motion(
        background, warp(background, true), config=CameraMotionConfig(max_side=480)
    )
    assert len(motion.source) == len(motion.target) == len(motion.inliers) == motion.n_tracked
    assert motion.inliers.sum() == motion.n_inliers
    assert motion.source[:, 0].max() > 480
    moved = motion.target[motion.inliers] - motion.source[motion.inliers]
    assert np.median(moved, axis=0) == pytest.approx([12.0, -6.0], abs=0.5)
