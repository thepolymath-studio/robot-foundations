"""Planar 2-link arm: FK, IK, workspace, Jacobian (research S3-S8)."""
import numpy as np
import pytest
from scipy.optimize import least_squares

from robot_foundations import arm as A
from robot_foundations.bodies import ATOM, CHARLIE
from robot_foundations.dh import chain, planar_rows

L1, L2 = ATOM
RNG = np.random.default_rng(2011)


def test_fk_special_poses():
    # straight along x, folded back, elbow at a right angle
    assert np.allclose(A.fist_position(0, 0, L1, L2), (L1 + L2, 0))
    assert np.allclose(A.fist_position(0, np.pi, L1, L2),
                       (L1 - L2, 0), atol=1e-15)
    assert np.allclose(A.fist_position(0, np.pi / 2, L1, L2),
                       (L1, L2))
    assert np.allclose(A.fist_position(np.pi / 2, 0, L1, L2),
                       (0, L1 + L2), atol=1e-15)


def test_fk_law_of_cosines():
    t = RNG.uniform(-np.pi, np.pi, (1000, 2))
    x, y = A.fist_position(t[:, 0], t[:, 1], L1, L2)
    r2 = L1**2 + L2**2 + 2 * L1 * L2 * np.cos(t[:, 1])
    assert np.allclose(x**2 + y**2, r2, rtol=0, atol=1e-14)


def test_fk_equals_dh_chain():
    # the 2-link formula is the DH product T = T1 T2 (S1, S2)
    for t1, t2 in RNG.uniform(-np.pi, np.pi, (200, 2)):
        T = chain(planar_rows(t1, t2, L1, L2))
        assert np.allclose(T[:2, 3], A.fist_position(t1, t2, L1, L2),
                           atol=1e-14)
        c, s = np.cos(t1 + t2), np.sin(t1 + t2)
        assert np.allclose(T[:3, :3],
                           [[c, -s, 0], [s, c, 0], [0, 0, 1]])


@pytest.mark.parametrize("elbow", [+1, -1])
def test_ik_round_trip(elbow):
    r_in, r_out = A.reach_ring(L1, L2)
    r = RNG.uniform(r_in + 1e-6, r_out - 1e-6, 500)
    phi = RNG.uniform(-np.pi, np.pi, 500)
    for x, y in zip(r * np.cos(phi), r * np.sin(phi)):
        t1, t2 = A.solve_ik(x, y, L1, L2, elbow)
        assert np.sign(t2) == elbow
        assert np.allclose(A.fist_position(t1, t2, L1, L2), (x, y),
                           atol=1e-12)


def test_ik_two_answers_are_mirror_images():
    # the two elbows are reflections across the shoulder-fist line
    x, y = 0.6, 0.2
    u = np.array([x, y]) / np.hypot(x, y)
    ep = np.array(A.elbow_position(A.solve_ik(x, y, L1, L2, +1)[0], L1))
    em = np.array(A.elbow_position(A.solve_ik(x, y, L1, L2, -1)[0], L1))
    mirror = 2 * (ep @ u) * u - ep
    assert np.allclose(mirror, em, atol=1e-12)
    assert np.linalg.norm(ep - em) > 0.1


def test_ik_matches_numerical_solver():
    # library check: scipy least squares finds the same poses
    x, y = 0.55, -0.3
    for elbow in (+1, -1):
        t = np.array(A.solve_ik(x, y, L1, L2, elbow))
        f = lambda q: np.subtract(A.fist_position(*q, L1, L2), (x, y))
        sol = least_squares(f, t + 0.2, xtol=1e-15, ftol=1e-15)
        assert np.allclose(sol.x, t, atol=1e-7)


def test_workspace_counts():
    r_in, r_out = A.reach_ring(L1, L2)
    assert A.n_solutions(0.5, 0.1, L1, L2) == 2
    assert A.n_solutions(r_out, 0, L1, L2) == 1        # straight
    assert A.n_solutions(0, r_in, L1, L2) == 1         # folded
    assert A.n_solutions(r_out + 1e-6, 0, L1, L2) == 0  # too far
    assert A.n_solutions(r_in / 2, 0, L1, L2) == 0     # in the hole
    assert A.solve_ik(1.0, 0.0, L1, L2) is None
    assert A.solve_ik(r_in / 2, 0, L1, L2) is None
    # on the outer edge both branches collapse onto t2 = 0
    assert A.solve_ik(r_out, 0, L1, L2) == (0.0, 0.0)


def test_jacobian_finite_differences():
    h = 1e-6
    for t1, t2 in RNG.uniform(-np.pi, np.pi, (50, 2)):
        J = A.jacobian(t1, t2, L1, L2)
        for j, d in enumerate(np.eye(2) * h):
            p = np.array(A.fist_position(t1 + d[0], t2 + d[1], L1, L2))
            m = np.array(A.fist_position(t1 - d[0], t2 - d[1], L1, L2))
            assert np.allclose(J[:, j], (p - m) / (2 * h), atol=1e-9)


def test_det_jacobian():
    for t1, t2 in RNG.uniform(-np.pi, np.pi, (50, 2)):
        J = A.jacobian(t1, t2, L1, L2)
        assert np.isclose(np.linalg.det(J),
                          A.det_jacobian(t2, L1, L2), atol=1e-14)
    # singular straight and folded, largest at a right angle
    assert A.det_jacobian(0.0, L1, L2) == 0.0
    assert abs(A.det_jacobian(np.pi, L1, L2)) < 1e-15
    assert np.isclose(A.det_jacobian(np.pi / 2, L1, L2), L1 * L2)
    J0 = A.jacobian(0.3, 0.0, L1, L2)
    assert np.linalg.matrix_rank(J0) == 1


def test_straight_arm_cannot_move_along_itself():
    # at t2 = 0 every joint motion moves the fist sideways only
    t1 = 0.4
    J = A.jacobian(t1, 0.0, L1, L2)
    along = np.array([np.cos(t1), np.sin(t1)])
    assert np.allclose(along @ J, 0, atol=1e-15)


def test_ik_path_raises_out_of_reach():
    with pytest.raises(ValueError):
        A.ik_path([(0.3, 0.1), (2.0, 0.0)], *CHARLIE)
