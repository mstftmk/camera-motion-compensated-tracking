import numpy as np
import pytest

from cmc_tracking.kalman import BoxKalman


def box_at(x, y=50.0, w=20.0, h=60.0):
    return np.array([x, y, x + w, y + h])


def run_constant_motion(steps=40, speed=3.0):
    kalman = BoxKalman(box_at(0.0))
    for t in range(1, steps):
        kalman.predict()
        kalman.update(box_at(speed * t))
    return kalman


def test_velocity_converges_to_true_speed():
    assert run_constant_motion().x[4] == pytest.approx(3.0, abs=0.01)


def test_predict_moves_by_velocity():
    kalman = run_constant_motion()
    before = kalman.x[0]
    kalman.predict()
    assert kalman.x[0] - before == pytest.approx(3.0, abs=0.01)


def test_area_never_goes_negative():
    kalman = BoxKalman(box_at(0.0))
    kalman.x[6] = -1e6
    kalman.predict()
    assert kalman.x[2] > 0


def test_replay_gives_a_better_velocity_after_a_gap_than_a_plain_update():
    # The object turns during the gap, so the frozen velocity is wrong afterwards.
    replayed = run_constant_motion()
    plain = run_constant_motion()
    position = 3.0 * 39
    for t in range(1, 11):
        for kalman in (replayed, plain):
            kalman.predict()
        replayed.mark_missed()
    position_after = position - 2.0 * 11
    for kalman in (replayed, plain):
        kalman.predict()
    replayed.update(box_at(position_after))
    plain._correct(np.array([position_after + 10.0, 80.0, 1200.0, 1 / 3]))
    assert abs(replayed.x[4] - (-2.0)) < abs(plain.x[4] - (-2.0))


def test_warp_translation_shifts_position_and_keeps_velocity():
    kalman = run_constant_motion()
    x, y, vx = kalman.x[0], kalman.x[1], kalman.x[4]
    kalman.warp(np.array([[1.0, 0, 10.0], [0, 1.0, -4.0]]))
    assert kalman.x[0] == pytest.approx(x + 10.0)
    assert kalman.x[1] == pytest.approx(y - 4.0)
    assert kalman.x[4] == pytest.approx(vx)


def test_warp_scale_updates_area_velocity_and_uncertainty():
    kalman = run_constant_motion()
    area, vx, p_xx = kalman.x[2], kalman.x[4], kalman.P[0, 0]
    kalman.warp(np.array([[2.0, 0, 0], [0, 2.0, 0]]))
    assert kalman.x[2] == pytest.approx(4 * area)
    assert kalman.x[4] == pytest.approx(2 * vx)
    assert kalman.P[0, 0] == pytest.approx(4 * p_xx)


def test_warp_also_moves_the_saved_state_used_for_replay():
    kalman = run_constant_motion()
    kalman.predict()
    kalman.mark_missed()
    saved_x = kalman._saved[0][0]
    kalman.warp(np.array([[1.0, 0, 7.0], [0, 1.0, 0]]))
    assert kalman._saved[0][0] == pytest.approx(saved_x + 7.0)
    np.testing.assert_allclose(kalman._last_box[[0, 2]], box_at(3.0 * 39 + 7.0)[[0, 2]])
