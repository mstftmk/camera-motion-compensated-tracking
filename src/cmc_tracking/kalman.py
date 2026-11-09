"""Constant-velocity Kalman filter over a box, with OC-SORT's re-update after a gap."""

from __future__ import annotations

import numpy as np

from .geometry import box_to_state, state_to_box

_F = np.eye(7)
_F[0, 4] = _F[1, 5] = _F[2, 6] = 1.0
_H = np.eye(4, 7)


class BoxKalman:
    """State is [x, y, area, ratio, vx, vy, varea]. Noise settings follow SORT."""

    def __init__(self, box: np.ndarray) -> None:
        self.x = np.zeros(7)
        self.x[:4] = box_to_state(box)
        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 1000.0
        self.Q = np.eye(7)
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0
        self._observed = True
        self._saved: tuple[np.ndarray, np.ndarray] | None = None
        self._last_box = np.asarray(box, dtype=np.float64).copy()
        self._missed = 0

    def predict(self) -> None:
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0.0
        self.x = _F @ self.x
        self.P = _F @ self.P @ _F.T + self.Q

    def mark_missed(self) -> None:
        """Record a frame without an observation, saving the state at the first miss."""
        if self._observed:
            self._saved = (self.x.copy(), self.P.copy())
            self._observed = False
            self._missed = 0
        self._missed += 1

    def update(self, box: np.ndarray) -> None:
        """Correct with an observation, first replaying a virtual path across any gap."""
        box = np.asarray(box, dtype=np.float64)
        if not self._observed:
            self._replay_gap(box)
        self._correct(box_to_state(box))
        self._observed = True
        self._last_box = box.copy()

    def _replay_gap(self, box: np.ndarray) -> None:
        # Observation-centric re-update: errors grow without observations, so roll back
        # to the last good state and walk a straight path to the new observation.
        self.x, self.P = self._saved
        start, end = box_to_state(self._last_box), box_to_state(box)
        start_wh = np.array([np.sqrt(start[2] * start[3]), np.sqrt(start[2] / start[3])])
        end_wh = np.array([np.sqrt(end[2] * end[3]), np.sqrt(end[2] / end[3])])
        steps = self._missed + 1
        for i in range(1, steps + 1):
            fraction = i / steps
            xy = start[:2] + fraction * (end[:2] - start[:2])
            wh = start_wh + fraction * (end_wh - start_wh)
            self._correct(np.array([xy[0], xy[1], wh[0] * wh[1], wh[0] / wh[1]]))
            if i < steps:
                self.predict()
        self._saved = None

    def _correct(self, z: np.ndarray) -> None:
        residual = z - _H @ self.x
        covariance = _H @ self.P @ _H.T + self.R
        gain = self.P @ _H.T @ np.linalg.inv(covariance)
        self.x = self.x + gain @ residual
        identity = np.eye(7)
        self.P = (identity - gain @ _H) @ self.P

    def box(self) -> np.ndarray:
        return state_to_box(self.x)

    def warp(self, affine: np.ndarray) -> None:
        """Move the filter into the current frame's coordinates after the camera moved."""
        self.x, self.P = _warp_state(self.x, self.P, affine)
        if self._saved is not None:
            self._saved = _warp_state(*self._saved, affine)
        linear, offset = affine[:, :2], affine[:, 2]
        scale = np.sqrt(abs(np.linalg.det(linear)))
        center = (self._last_box[:2] + self._last_box[2:]) / 2
        half = (self._last_box[2:] - self._last_box[:2]) * scale / 2
        moved = linear @ center + offset
        self._last_box = np.concatenate([moved - half, moved + half])


def _warp_state(x: np.ndarray, P: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    linear, offset = affine[:, :2], affine[:, 2]
    area_scale = abs(np.linalg.det(linear))
    transform = np.eye(7)
    transform[:2, :2] = linear
    transform[4:6, 4:6] = linear
    transform[2, 2] = transform[6, 6] = area_scale
    new_x = transform @ x
    new_x[:2] += offset
    return new_x, transform @ P @ transform.T
