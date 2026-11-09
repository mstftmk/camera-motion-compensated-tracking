"""Box geometry. Boxes are [x1, y1, x2, y2] arrays."""

from __future__ import annotations

import numpy as np


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Intersection over union of every box in boxes_a with every box in boxes_b."""
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    left = np.maximum(a[:, None, 0], b[None, :, 0])
    top = np.maximum(a[:, None, 1], b[None, :, 1])
    right = np.minimum(a[:, None, 2], b[None, :, 2])
    bottom = np.minimum(a[:, None, 3], b[None, :, 3])
    intersection = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)


def box_centers(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64)
    return np.stack([(boxes[..., 0] + boxes[..., 2]) / 2, (boxes[..., 1] + boxes[..., 3]) / 2], -1)


def box_to_state(box: np.ndarray) -> np.ndarray:
    """Convert a box to [center x, center y, area, aspect ratio]."""
    width, height = box[2] - box[0], box[3] - box[1]
    return np.array([box[0] + width / 2, box[1] + height / 2, width * height, width / height])


def state_to_box(state: np.ndarray) -> np.ndarray:
    """Convert [center x, center y, area, aspect ratio, ...] to a box."""
    area = max(float(state[2]), 1e-9)
    ratio = max(float(state[3]), 1e-9)
    width = np.sqrt(area * ratio)
    height = area / width
    return np.array(
        [state[0] - width / 2, state[1] - height / 2, state[0] + width / 2, state[1] + height / 2]
    )


def warp_boxes(boxes: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Move boxes with a 2 x 3 affine map, keeping them axis-aligned around the moved center.

    Width and height scale with the map's overall scale. Rotation of the box itself is
    ignored, because tracked boxes stay axis-aligned.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    centers = box_centers(boxes)
    moved = centers @ affine[:, :2].T + affine[:, 2]
    scale = np.sqrt(abs(np.linalg.det(affine[:, :2])))
    half = (boxes[:, 2:] - boxes[:, :2]) * scale / 2
    return np.concatenate([moved - half, moved + half], axis=1)
