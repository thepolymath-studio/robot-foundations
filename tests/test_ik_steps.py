"""Step-by-step IK near the stretched-arm singularity (research S6-S8)."""
import numpy as np
import pytest

from robot_foundations import arm as A
from robot_foundations import shadow as S
from robot_foundations.bodies import ATOM
from robot_foundations.ik_steps import (dls_step, needed_joint_speeds,
                                        plain_step, track)

L1, L2 = ATOM
RNG = np.random.default_rng(7)


def test_dls_tends_to_plain_inverse():
    J = A.jacobian(0.3, 1.2, L1, L2)
    dx = np.array([0.01, -0.02])
    assert np.allclose(dls_step(J, dx, 1e-9), plain_step(J, dx))
    assert np.allclose(J @ plain_step(J, dx), dx)


def test_dls_matches_svd_formula():
    # dtheta = sum_i sigma_i / (sigma_i^2 + lam^2) (u_i . dx) v_i
    lam = 0.02
    for t1, t2 in RNG.uniform(-np.pi, np.pi, (30, 2)):
        J = A.jacobian(t1, t2, L1, L2)
        dx = RNG.normal(size=2)
        U, s, Vt = np.linalg.svd(J)
        want = Vt.T @ (s / (s**2 + lam**2) * (U.T @ dx))
        assert np.allclose(dls_step(J, dx, lam), want, atol=1e-10)


def test_dls_is_bounded_at_the_singularity():
    lam = 0.02
    J = A.jacobian(0.0, 0.0, L1, L2)             # straight arm
    dx = np.array([1.0, 0.0])                    # along the arm
    with pytest.raises(np.linalg.LinAlgError):
        plain_step(J, dx)
    assert np.allclose(dls_step(J, dx, lam), 0)  # can't move that way
    for t2 in np.radians([5, 1, 0.1, 0.01]):
        J = A.jacobian(0.0, t2, L1, L2)
        assert (np.linalg.norm(dls_step(J, dx, lam))
                <= 1 / (2 * lam) + 1e-9)


def test_needed_elbow_speed_closed_form():
    # radial fist speed v: r' = -L1 L2 sin(t2) t2' / r
    v = 5.0
    gaps = np.geomspace(1e-4, 0.2, 30)
    w, th = S.needed_speeds_vs_gap(ATOM, gaps, v)
    r = L1 + L2 - gaps
    want = -r * v / (L1 * L2 * np.sin(th[:, 1]))
    assert np.allclose(w[:, 1], want, rtol=1e-9)
    # and together they move the fist straight out, not sideways
    for q, wi in zip(th, w):
        assert np.allclose(A.jacobian(*q, L1, L2) @ wi, [v, 0])


def test_needed_speed_grows_as_one_over_root_gap():
    w, _ = S.needed_speeds_vs_gap(ATOM, [1e-3, 1e-4, 1e-5], 5.0)
    ratio = w[1:, 1] / w[:-1, 1]
    assert np.allclose(ratio, np.sqrt(10), rtol=2e-3)


def test_plain_track_follows_a_safe_path():
    # far from the singularity both rules follow a circle exactly
    dt, gain = 1e-3, 50.0
    phi = np.linspace(0, np.pi / 2, 400)
    path = 0.6 * np.column_stack([np.cos(phi), np.sin(phi)])
    th0 = A.solve_ik(*path[0], L1, L2, +1)
    for step in (plain_step, lambda J, d: dls_step(J, d, 1e-4)):
        th = track(path, th0, ATOM, step, dt, gain)
        p = S.fist_path(th, ATOM)
        assert np.abs(p - path).max() < 1e-4


def test_plain_inverse_has_no_ceiling_dls_does():
    # faster ticks: the plain inverse asks for ever more (~1/sqrt(dt));
    # damped least squares converges to one bounded answer
    peak = {}
    for dt in (1e-3, 1e-4, 1e-5):
        r = S.stretched_jab(ATOM, 0.6, 5.0, dt, 0.02, 50.0)
        sp = {k: np.abs(np.diff(r[k][:, 1])).max() / dt
              for k in ("plain", "dls")}
        peak[dt] = sp
    assert peak[1e-4]["plain"] > 2.5 * peak[1e-3]["plain"]
    assert peak[1e-5]["plain"] > 2.5 * peak[1e-4]["plain"]
    assert abs(peak[1e-5]["dls"] / peak[1e-4]["dls"] - 1) < 0.1
    assert peak[1e-5]["dls"] < peak[1e-5]["plain"] / 5


def test_dls_miss_small_and_shrinks_with_lambda():
    misses = []
    for lam in (0.05, 0.02, 0.01):
        r = S.stretched_jab(ATOM, 0.6, 5.0, 1e-4, lam, 50.0, 0.1)
        m = np.hypot(*(S.fist_path(r["dls"], ATOM) - r["path"]).T)
        misses.append(m.max())
        assert m[-1] < m.max()            # creeps closer when held
    assert misses[0] > misses[1] > misses[2]
    assert misses[1] < 0.005              # under 5 mm at lam = 2 cm


def test_needed_joint_speeds_move_the_fist():
    th = np.array([[0.2, 1.0], [-0.5, 2.0]])
    v = np.array([1.0, -2.0])
    w = needed_joint_speeds(th, v, ATOM)
    for q, wi in zip(th, w):
        assert np.allclose(A.jacobian(*q, L1, L2) @ wi, v)


def test_stretched_jab_deterministic():
    a = S.stretched_jab(ATOM, 0.6, 5.0, 1e-4, 0.02, 50.0)
    b = S.stretched_jab(ATOM, 0.6, 5.0, 1e-4, 0.02, 50.0)
    assert np.array_equal(a["dls"], b["dls"])
    assert np.array_equal(a["plain"], b["plain"])


def test_example_numbers_sim():
    import runpy
    from pathlib import Path
    from robot_foundations.results import check, clean
    ex = Path(__file__).parents[1] / "examples"
    mod = runpy.run_path(str(ex / "2026-11-wk1_shadow_sim.py"))
    res, _ = mod["results"]()
    assert check(clean(res), mod["EXPECTED"]) == []
