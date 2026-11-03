"""Chosen arms (Drillis & Contini, Atom 2.29 m) and the punch sequence."""
import numpy as np

from robot_foundations import moves as M
from robot_foundations.bodies import ATOM, CHARLIE, K


def test_charlie_proportions():
    # Winter fig. 4.1: upper arm 0.186 H, forearm 0.146 H, hand 0.108 H
    assert np.isclose(CHARLIE.L1, 0.186 * 1.80)
    assert np.isclose(CHARLIE.L2, (0.146 + 0.054) * 1.80)


def test_atom_is_charlie_scaled():
    assert np.isclose(K, 2.29 / 1.80)
    assert np.allclose(np.array(ATOM) / np.array(CHARLIE), K)


def test_min_jerk_ends():
    s = np.array([0.0, 1.0])
    assert np.allclose(M.min_jerk(s), [0, 1])
    h = 1e-6
    for e in (0.0, 1.0):  # zero speed and acceleration at both ends
        d1 = (M.min_jerk(e + h) - M.min_jerk(e - h)) / (2 * h)
        assert abs(d1) < 1e-5
    assert np.isclose(M.min_jerk(0.5), 0.5)


def test_sequence_hits_poses_and_loops():
    for t0, t1, name in M.key_times():
        th = M.angles_at(t1 - 1e-12)[0]
        assert np.allclose(np.degrees(th), M.POSES_DEG[name],
                           atol=1e-6)
    a = M.angles_at(0.0)[0]
    b = M.angles_at(M.LOOP_S - 1e-9)[0]
    assert np.allclose(a, b)
    assert np.isclose(M.LOOP_S, 3.0)
    land = M.landing_times()
    assert np.isclose(land["jab"], 0.50)
    assert np.isclose(land["hook"], 1.80)


def test_sequence_is_smooth():
    t = np.arange(0, M.LOOP_S, 1e-4)
    th = M.angles_at(t)
    step = np.abs(np.diff(th, axis=0)).max()
    assert step < np.radians(0.2)   # no jumps between samples


def test_straight_jab_path():
    t, xy = M.straight_jab(0.6, 0.884, 5.0, 1e-4, hold=0.01)
    assert np.allclose(xy[:, 1], 0)
    v = np.diff(xy[:, 0]) / 1e-4
    moving = v > 0
    assert np.allclose(v[moving][:-1], 5.0)
    assert np.isclose(xy[-1, 0], 0.884)
    assert np.isclose(t[-1], (0.884 - 0.6) / 5.0 + 0.01, atol=2e-4)
