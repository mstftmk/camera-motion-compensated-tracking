import numpy as np
import pytest

from cmc_tracking.association import assign, direction_cost, match_with_direction


def test_direction_cost_is_highest_for_the_same_direction():
    previous = np.array([[0.0, 0, 10, 10]])
    velocity = np.array([[1.0, 0.0]])
    detections = np.array(
        [[10.0, 0, 20, 10], [-10.0, 0, 0, 10], [0.0, 10, 10, 20]]
    )
    cost = direction_cost(velocity, previous, detections)
    assert cost[:, 0] == pytest.approx([0.5, -0.5, 0.0], abs=1e-3)


def test_track_without_velocity_gets_no_direction_term():
    cost = direction_cost(np.zeros((1, 2)), np.array([[0.0, 0, 10, 10]]), np.array([[10.0, 0, 20, 10]]))
    assert cost[0, 0] == pytest.approx(0.0)


def test_assign_rejects_pairs_below_the_iou_threshold():
    iou = np.array([[0.9, 0.0], [0.0, 0.2]])
    matches, free_rows, free_cols = assign(iou, iou, 0.3)
    assert matches == [(0, 0)]
    assert free_rows == [1]
    assert free_cols == [1]


def test_assign_finds_the_best_overall_pairing():
    iou = np.array([[0.8, 0.7], [0.7, 0.1]])
    matches, _, _ = assign(iou, iou, 0.3)
    assert sorted(matches) == [(0, 1), (1, 0)]


def test_assign_handles_empty_inputs():
    empty = np.empty((0, 2))
    assert assign(empty, empty, 0.3) == ([], [], [0, 1])


def test_direction_decides_between_two_equal_overlaps():
    predicted = np.array([[50.0, 0, 70, 40]])
    detections = np.array([[58.0, 0, 78, 40], [42.0, 0, 62, 40]])
    previous = np.array([[40.0, 0, 60, 40]])
    moving_right = np.array([[1.0, 0.0]])
    matches, _, _ = match_with_direction(
        detections, np.array([0.9, 0.9]), predicted, moving_right, previous, 0.3, 0.2
    )
    assert matches == [(0, 0)]
    moving_left = np.array([[-1.0, 0.0]])
    previous_left = np.array([[60.0, 0, 80, 40]])
    matches, _, _ = match_with_direction(
        detections, np.array([0.9, 0.9]), predicted, moving_left, previous_left, 0.3, 0.2
    )
    assert matches == [(1, 0)]


def test_direction_cannot_override_an_unambiguous_match():
    # One candidate only, moving against the track's direction: it must still match.
    predicted = np.array([[50.0, 0, 70, 40]])
    detections = np.array([[46.0, 0, 66, 40]])
    previous = np.array([[40.0, 0, 60, 40]])
    matches, _, _ = match_with_direction(
        detections, np.array([0.9]), predicted, np.array([[1.0, 0.0]]), previous, 0.3, 50.0
    )
    assert matches == [(0, 0)]
