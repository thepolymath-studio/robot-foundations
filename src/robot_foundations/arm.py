"""Planar 2-link arm: one shoulder, one elbow, one flat plane.

Angles (radians): t1 is the shoulder angle, measured from the
x axis; t2 is the elbow angle, measured from the upper arm
(t2 = 0 is a straight arm). L1 is the upper arm and L2 the
forearm to the middle of the fist, both in metres. The
shoulder sits at the origin.

Sources: Lynch & Park, Modern Robotics (2017), ch. 4-6 (FK,
velocity kinematics, IK); research S3-S8.
"""
from __future__ import annotations

import numpy as np


def fist_position(t1, t2, L1, L2):
    """Joint angles in, fist position out."""
    x = L1 * np.cos(t1) + L2 * np.cos(t1 + t2)
    y = L1 * np.sin(t1) + L2 * np.sin(t1 + t2)
    return x, y


def elbow_position(t1, L1):
    """Where the elbow is (the end of the upper arm)."""
    return L1 * np.cos(t1), L1 * np.sin(t1)


def reach_ring(L1, L2):
    """Inner and outer radius of the reachable ring."""
    return abs(L1 - L2), L1 + L2


def solve_ik(x, y, L1, L2, elbow=+1):
    """Fist position in, joint angles out.

    Two arms can put the fist on (x, y): the elbow
    bent one way (elbow=+1) or the other (-1).
    Returns None when the point is out of reach.
    """
    c2 = (x*x + y*y - L1*L1 - L2*L2) / (2*L1*L2)
    if abs(c2) > 1 + 1e-12:
        return None              # out of reach
    t2 = elbow * np.arccos(min(1.0, max(-1.0, c2)))
    k1 = L1 + L2 * np.cos(t2)
    k2 = L2 * np.sin(t2)
    t1 = np.arctan2(y, x) - np.arctan2(k2, k1)
    return t1, t2


def n_solutions(x, y, L1, L2, tol=1e-12):
    """How many arm poses reach (x, y): 2, 1 or 0."""
    r = np.hypot(x, y)
    r_in, r_out = reach_ring(L1, L2)
    if r < r_in - tol or r > r_out + tol:
        return 0
    edge = min(abs(r - r_in), abs(r - r_out))
    return 1 if edge <= tol else 2


def jacobian(t1, t2, L1, L2):
    """J: how the fist moves per unit turn of each joint.

    Column 1 is d(x, y)/d t1, column 2 is d(x, y)/d t2,
    in metres per radian.
    """
    s1, c1 = np.sin(t1), np.cos(t1)
    s12, c12 = np.sin(t1 + t2), np.cos(t1 + t2)
    return np.array([[-L1 * s1 - L2 * s12, -L2 * s12],
                     [L1 * c1 + L2 * c12, L2 * c12]])


def det_jacobian(t2, L1, L2):
    """det J = L1 L2 sin t2: zero when straight or folded."""
    return L1 * L2 * np.sin(t2)


def ik_path(xy, L1, L2, elbow=+1):
    """solve_ik along a path of fist points, shape (N, 2).

    Raises ValueError if any point is out of reach.
    """
    out = np.empty_like(np.asarray(xy, float))
    for i, (x, y) in enumerate(xy):
        sol = solve_ik(x, y, L1, L2, elbow)
        if sol is None:
            raise ValueError(f"point {i}: out of reach")
        out[i] = sol
    return out
