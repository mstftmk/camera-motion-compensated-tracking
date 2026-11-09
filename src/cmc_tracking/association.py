"""Match detections to tracks with IoU and, as in OC-SORT, direction consistency."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import box_centers, iou_matrix


def _unit_directions(from_boxes: np.ndarray, to_boxes: np.ndarray) -> np.ndarray:
    """Unit vectors from each box in from_boxes (tracks) to each box in to_boxes (detections)."""
    delta = box_centers(to_boxes)[None, :, :] - box_centers(from_boxes)[:, None, :]
    return delta / (np.linalg.norm(delta, axis=2, keepdims=True) + 1e-6)


def direction_cost(
    velocities: np.ndarray,
    previous_boxes: np.ndarray,
    detections: np.ndarray,
) -> np.ndarray:
    """Reward matches that continue a track's direction: from +0.5 (same) to -0.5 (opposite).

    Shape is (detections, tracks). A track with no velocity yet has a zero vector and gets 0.
    """
    directions = _unit_directions(previous_boxes, detections)
    cosine = np.clip(np.sum(directions * velocities[:, None, :], axis=2), -1.0, 1.0)
    score = (np.pi / 2 - np.arccos(cosine)) / np.pi
    return score.T


def assign(
    scores: np.ndarray, iou: np.ndarray, iou_threshold: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Solve a maximum-score assignment and keep pairs that overlap enough."""
    n_rows, n_cols = iou.shape
    matches: list[tuple[int, int]] = []
    if n_rows and n_cols:
        for row, col in zip(*linear_sum_assignment(-scores)):
            if iou[row, col] >= iou_threshold:
                matches.append((int(row), int(col)))
    matched_rows = {row for row, _ in matches}
    matched_cols = {col for _, col in matches}
    return (
        matches,
        [row for row in range(n_rows) if row not in matched_rows],
        [col for col in range(n_cols) if col not in matched_cols],
    )


def match_with_direction(
    detections: np.ndarray,
    det_scores: np.ndarray,
    predictions: np.ndarray,
    velocities: np.ndarray,
    previous_boxes: np.ndarray,
    iou_threshold: float,
    inertia: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """First association stage: predicted-box IoU plus the direction term."""
    iou = iou_matrix(detections, predictions)
    if iou.size == 0:
        return assign(iou, iou, iou_threshold)
    # When every detection and track has at most one candidate, there is nothing to decide,
    # and the direction term must not be able to change the outcome.
    candidates = iou > iou_threshold
    if candidates.sum(axis=1).max() <= 1 and candidates.sum(axis=0).max() <= 1:
        return assign(candidates.astype(float), iou, iou_threshold)
    direction = direction_cost(velocities, previous_boxes, detections)
    return assign(iou + inertia * direction * det_scores[:, None], iou, iou_threshold)
