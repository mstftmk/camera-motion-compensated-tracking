"""OC-SORT (Cao et al., CVPR 2023), written from the paper, with optional camera motion input."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .association import assign, match_with_direction
from .geometry import iou_matrix, warp_boxes
from .kalman import BoxKalman


@dataclass(frozen=True)
class OCSortConfig:
    """Defaults follow the OC-SORT paper's MOT17 settings, except det_threshold."""

    det_threshold: float = 0.5
    iou_threshold: float = 0.3
    max_age: int = 30
    min_hits: int = 3
    delta_t: int = 3
    inertia: float = 0.2


class _Track:
    def __init__(self, box: np.ndarray, track_id: int) -> None:
        self.id = track_id
        self.kalman = BoxKalman(box)
        self.age = 0
        self.time_since_update = 0
        self.hit_streak = 0
        self.last_observation = np.asarray(box, dtype=np.float64).copy()
        self.observations: dict[int, np.ndarray] = {0: self.last_observation.copy()}
        self.velocity = np.zeros(2)

    def predict(self) -> None:
        self.kalman.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

    def previous_observation(self, delta_t: int) -> np.ndarray:
        """The observation delta_t frames ago, or the closest newer one that exists."""
        for gap in range(delta_t, 0, -1):
            if self.age - gap in self.observations:
                return self.observations[self.age - gap]
        return self.last_observation

    def update(self, box: np.ndarray, delta_t: int) -> None:
        center = (box[:2] + box[2:]) / 2
        previous = self.previous_observation(delta_t)
        delta = center - (previous[:2] + previous[2:]) / 2
        self.velocity = delta / (np.linalg.norm(delta) + 1e-6)
        self.kalman.update(box)
        self.last_observation = np.asarray(box, dtype=np.float64).copy()
        self.observations[self.age] = self.last_observation.copy()
        # Only the last delta_t observations are ever read, so older ones can go.
        self.observations = {
            age: obs for age, obs in self.observations.items() if self.age - age <= delta_t
        }
        self.time_since_update = 0
        self.hit_streak += 1

    def warp(self, affine: np.ndarray) -> None:
        self.kalman.warp(affine)
        self.last_observation = warp_boxes(self.last_observation, affine)[0]
        self.observations = {
            age: warp_boxes(box, affine)[0] for age, box in self.observations.items()
        }
        self.velocity = _unit(affine[:, :2] @ self.velocity)


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / (np.linalg.norm(vector) + 1e-6)


class OCSort:
    """Feed one frame of detections at a time; pass camera motion to compensate for it."""

    def __init__(self, config: OCSortConfig | None = None) -> None:
        self.config = config or OCSortConfig()
        self.tracks: list[_Track] = []
        self._next_id = 1
        self.frame = 0

    def update(self, detections: np.ndarray, affine: np.ndarray | None = None) -> np.ndarray:
        """Return rows of [x1, y1, x2, y2, track id] for confirmed tracks in this frame.

        detections has rows of [x1, y1, x2, y2, score]. affine is an optional 2 x 3 map from
        the previous frame's coordinates to this frame's.
        """
        # Limitations:
        # - Matching uses only motion and overlap, with no appearance model, so people who
        #   cross behind each other can swap IDs.
        # - Small input changes can alter one match and every later one, so per-sequence
        #   scores carry noise. See scripts/run_noise_control.py.
        config = self.config
        self.frame += 1
        detections = np.asarray(detections, dtype=np.float64).reshape(-1, 5)
        detections = detections[detections[:, 4] > config.det_threshold]
        boxes, scores = detections[:, :4], detections[:, 4]

        for track in self.tracks:
            track.predict()
            if affine is not None:
                track.warp(affine)
        self.tracks = [t for t in self.tracks if np.isfinite(t.kalman.x).all()]

        predictions = np.array([t.kalman.box() for t in self.tracks]).reshape(-1, 4)
        velocities = np.array([t.velocity for t in self.tracks]).reshape(-1, 2)
        previous = np.array([t.previous_observation(config.delta_t) for t in self.tracks])
        previous = previous.reshape(-1, 4)
        matches, free_dets, free_tracks = match_with_direction(
            boxes, scores, predictions, velocities, previous, config.iou_threshold, config.inertia
        )
        for det, trk in matches:
            self.tracks[trk].update(boxes[det], config.delta_t)

        # Observation-centric recovery: retry leftovers against each track's last real box.
        if free_dets and free_tracks:
            last = np.array([self.tracks[t].last_observation for t in free_tracks])
            iou = iou_matrix(boxes[free_dets], last)
            rematch, left_dets, left_tracks = assign(iou, iou, config.iou_threshold)
            for det, trk in rematch:
                self.tracks[free_tracks[trk]].update(boxes[free_dets[det]], config.delta_t)
            free_dets = [free_dets[i] for i in left_dets]
            free_tracks = [free_tracks[i] for i in left_tracks]

        for trk in free_tracks:
            self.tracks[trk].kalman.mark_missed()
        for det in free_dets:
            self.tracks.append(_Track(boxes[det], self._next_id))
            self._next_id += 1

        output = []
        for track in self.tracks:
            confirmed = track.hit_streak >= config.min_hits or self.frame <= config.min_hits
            if track.time_since_update < 1 and confirmed:
                output.append([*track.last_observation, track.id])
        self.tracks = [t for t in self.tracks if t.time_since_update <= config.max_age]
        return np.array(output, dtype=np.float64).reshape(-1, 5)
