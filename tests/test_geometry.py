import numpy as np
import pytest

from cmc_tracking.geometry import box_to_state, iou_matrix, state_to_box, warp_boxes


def test_iou_known_values():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30], [0, 0, 5, 5]])
    np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 50 / 150, 0.0, 25 / 100]])


def test_iou_handles_empty_inputs():
    assert iou_matrix(np.empty((0, 4)), np.zeros((3, 4))).shape == (0, 3)
    assert iou_matrix(np.zeros((2, 4)), np.empty((0, 4))).shape == (2, 0)


def test_box_state_round_trip():
    box = np.array([10.0, 20.0, 40.0, 100.0])
    np.testing.assert_allclose(box_to_state(box), [25, 60, 2400, 30 / 80])
    np.testing.assert_allclose(state_to_box(box_to_state(box)), box)


def test_warp_translation_moves_boxes():
    affine = np.array([[1.0, 0, 5], [0, 1.0, -3]])
    moved = warp_boxes(np.array([[0.0, 0, 10, 20]]), affine)
    np.testing.assert_allclose(moved, [[5, -3, 15, 17]])


def test_warp_scale_scales_size_and_position():
    affine = np.array([[2.0, 0, 0], [0, 2.0, 0]])
    moved = warp_boxes(np.array([[10.0, 10, 20, 30]]), affine)
    np.testing.assert_allclose(moved, [[20, 20, 40, 60]])


def test_warp_rotation_moves_center_and_keeps_size():
    affine = np.array([[0.0, -1.0, 0], [1.0, 0.0, 0]])
    moved = warp_boxes(np.array([[10.0, 0, 20, 4]]), affine)[0]
    assert (moved[:2] + moved[2:]) / 2 == pytest.approx([-2.0, 15.0])
    assert (moved[2:] - moved[:2]) == pytest.approx([10.0, 4.0])
