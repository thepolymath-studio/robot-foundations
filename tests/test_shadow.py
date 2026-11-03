"""Shadow mode: copy the angles vs copy the fist (research S15, S16)."""
import numpy as np
import pytest

from robot_foundations import moves as M
from robot_foundations import shadow as S
from robot_foundations.bodies import ATOM, CHARLIE, K, Arm

T = np.arange(0.0, M.LOOP_S, 1e-3)
TH = M.angles_at(T)


def test_copy_angles_scales_fist_by_k():
    # S15: same angles on k-times-longer links -> p' = k p
    pc = S.fist_path(TH, CHARLIE)
    pa = S.fist_path(S.copy_angles(TH, ATOM), ATOM)
    assert np.allclose(pa, K * pc, atol=1e-14)
    miss = S.miss(TH, TH, CHARLIE, ATOM)
    assert np.allclose(miss, (K - 1) * np.hypot(*pc.T), atol=1e-14)


def test_unequal_proportions_change_the_shape():
    # S15, second half: a longer forearm only is not a zoom
    long_fore = Arm(CHARLIE.L1, K * CHARLIE.L2)
    pl, pc = S.fist_path(TH, long_fore), S.fist_path(TH, CHARLIE)
    ratio = np.hypot(*pl.T) / np.hypot(*pc.T)
    assert ratio.max() - ratio.min() > 0.02
    turn = np.arctan2(pl[:, 1], pl[:, 0]) - np.arctan2(pc[:, 1], pc[:, 0])
    assert np.degrees(np.abs(turn)).max() > 5    # points elsewhere


def test_copy_fist_lands_on_charlies_fist():
    f = S.copy_fist(TH, CHARLIE, ATOM)
    assert S.miss(f, TH, CHARLIE, ATOM).max() < 1e-12
    assert np.all(np.sign(f[:, 1]) == np.sign(TH[:, 1]))
    # Atom needs a more bent elbow to reach the nearer point
    assert np.all(np.abs(f[:, 1]) >= np.abs(TH[:, 1]) - 1e-12)


def test_copy_fist_flags_unreachable():
    big = Arm(0.1, 0.1)      # an arm too short for Charlie's jab
    with pytest.raises(ValueError):
        S.copy_fist(TH, CHARLIE, big)


def test_example_numbers_concept():
    import runpy
    from pathlib import Path
    ex = Path(__file__).parents[1] / "examples"
    mod = runpy.run_path(str(ex / "2026-11-wk1_shadow_mode.py"))
    from robot_foundations.results import check, clean
    res = clean(mod["results"]())
    assert check(res, mod["EXPECTED"]) == []
    assert res == clean(mod["results"]())      # determinism


def test_min_elbow_bend_copying_the_fist():
    # closed form at Charlie's full reach r = L1 + L2 (Charlie's):
    # cos t2 = (r^2 - L1'^2 - L2'^2) / (2 L1' L2') on Atom's arm
    from robot_foundations import arm as A
    r = sum(CHARLIE)
    c2 = (r**2 - ATOM.L1**2 - ATOM.L2**2) / (2 * ATOM.L1 * ATOM.L2)
    want = np.degrees(np.arccos(c2))
    assert np.isclose(want, 76.4, atol=0.05)
    # the bend only grows as the point comes closer to the shoulder
    rs = np.linspace(A.reach_ring(*ATOM)[0] + 1e-9, r, 2000)
    th = A.ik_path(np.column_stack([rs, 0 * rs]), *ATOM)
    assert np.all(np.diff(np.abs(th[:, 1])) < 0)
    assert np.isclose(np.degrees(abs(th[-1, 1])), want)
    f = S.copy_fist(TH, CHARLIE, ATOM)
    assert np.degrees(np.abs(f[:, 1]).min()) >= want
