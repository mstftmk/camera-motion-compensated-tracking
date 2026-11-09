import numpy as np

from cmc_tracking import OCSort, OCSortConfig


def detection(x, y=100.0, w=30.0, h=80.0, score=0.9):
    return [x, y, x + w, y + h, score]


def run(frames, affines=None, config=None):
    tracker = OCSort(config)
    outputs = []
    for i, frame in enumerate(frames):
        affine = None if affines is None else affines[i]
        outputs.append(tracker.update(np.array(frame).reshape(-1, 5), affine))
    return outputs


def ids(outputs):
    return {int(row[4]) for frame in outputs for row in frame}


def test_one_object_keeps_one_id():
    outputs = run([[detection(10 + 3 * t)] for t in range(40)])
    assert ids(outputs) == {1}
    assert all(len(frame) == 1 for frame in outputs)


def test_output_boxes_are_the_detections():
    outputs = run([[detection(10 + 3 * t)] for t in range(10)])
    np.testing.assert_allclose(outputs[-1][0, :4], detection(37)[:4])


def test_low_score_detections_are_ignored():
    outputs = run([[detection(10, score=0.2)] for _ in range(10)])
    assert ids(outputs) == set()


def test_new_track_needs_min_hits_after_the_first_frames():
    tracker = OCSort(OCSortConfig(min_hits=3))
    for _ in range(5):
        tracker.update(np.empty((0, 5)))
    shown = [len(tracker.update(np.array([detection(10)]))) for _ in range(4)]
    assert shown == [0, 0, 0, 1]


def test_short_gap_keeps_the_id():
    frames = [[detection(10 + 3 * t)] if not 15 <= t < 22 else [] for t in range(40)]
    assert ids(run(frames)) == {1}


def test_gap_longer_than_max_age_starts_a_new_id():
    config = OCSortConfig(max_age=5)
    frames = [[detection(10 + 3 * t)] if not 15 <= t < 30 else [] for t in range(50)]
    assert ids(run(frames, config=config)) == {1, 2}


def test_last_observation_recovers_a_stopped_object():
    # The object stops during the gap, so the prediction runs ahead of it.
    # The second association stage against its last real box brings it back.
    frames = [[detection(10 + 8 * t)] for t in range(15)]
    frames += [[] for _ in range(4)]
    frames += [[detection(10 + 8 * 14)] for _ in range(10)]
    assert ids(run(frames)) == {1}


def test_two_objects_keep_separate_ids():
    frames = [[detection(10 + 2 * t), detection(400 - 2 * t, y=300)] for t in range(30)]
    outputs = run(frames)
    assert ids(outputs) == {1, 2}
    assert all(len(frame) == 2 for frame in outputs)


def test_crossing_objects_keep_their_ids_using_direction():
    frames = [
        [detection(100 + 6 * t, y=100), detection(300 - 6 * t, y=102)] for t in range(35)
    ]
    outputs = run(frames)
    first, last = outputs[3], outputs[-1]
    start = {int(row[4]): row[0] for row in first}
    end = {int(row[4]): row[0] for row in last}
    assert ids(outputs) == {1, 2}
    for track_id in start:
        assert (end[track_id] > start[track_id]) == (start[track_id] < 200)


def camera_shift_frames(n=40, shift=45.0):
    """A still object seen by a camera that pans: it jumps shift pixels per frame in the image."""
    return [[detection(1000 - shift * t)] for t in range(n)]


def test_fast_camera_pan_breaks_the_id_without_compensation():
    assert len(ids(run(camera_shift_frames()))) > 1


def test_fast_camera_pan_keeps_the_id_with_compensation():
    affine = np.array([[1.0, 0, -45.0], [0, 1.0, 0]])
    outputs = run(camera_shift_frames(), affines=[affine] * 40)
    assert ids(outputs) == {1}


def test_identity_motion_changes_nothing():
    frames = [[detection(10 + 3 * t), detection(500 - t, y=300)] for t in range(30)]
    plain = run(frames)
    identity = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    compensated = run(frames, affines=[identity] * 30)
    for a, b in zip(plain, compensated):
        np.testing.assert_allclose(a, b)


def test_compensation_also_moves_tracks_during_a_gap():
    affine = np.array([[1.0, 0, -45.0], [0, 1.0, 0]])
    frames = camera_shift_frames(30)
    for t in range(10, 16):
        frames[t] = []
    outputs = run(frames, affines=[affine] * 30)
    assert ids(outputs) == {1}


def test_warp_moves_the_track_history_with_the_camera():
    tracker = OCSort()
    for t in range(6):
        tracker.update(np.array([detection(100 + 4 * t)]))
    track = tracker.tracks[0]
    before = track.last_observation.copy()
    old = track.observations[track.age - 1].copy()
    velocity = track.velocity.copy()
    track.warp(np.array([[1.0, 0, 30.0], [0, 1.0, -5.0]]))
    np.testing.assert_allclose(track.last_observation - before, [30, -5, 30, -5])
    np.testing.assert_allclose(track.observations[track.age - 1] - old, [30, -5, 30, -5])
    np.testing.assert_allclose(track.velocity, velocity, atol=1e-4)
