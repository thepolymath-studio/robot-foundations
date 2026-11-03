"""Shadow mode: Atom copies Charlie, two ways.

Both stand at one shared shoulder (the origin), so the
target for Atom's fist is simply where Charlie's fist is.
    copy the angles   Atom takes Charlie's t1, t2
    copy the fist     Atom solves IK for Charlie's fist point
Step 2 (the stretched jab) lives in stretched_jab().
"""
from __future__ import annotations

import numpy as np

from .arm import fist_position, ik_path, solve_ik
from .ik_steps import dls_step, needed_joint_speeds, plain_step
from .ik_steps import track
from .moves import straight_jab


def fist_path(theta, arm):
    """Fist positions (N, 2) for joint angles (N, 2)."""
    x, y = fist_position(theta[:, 0], theta[:, 1], *arm)
    return np.column_stack([x, y])


def copy_angles(theta, atom):
    """Atom takes Charlie's joint angles as they are."""
    return np.array(theta, float)


def copy_fist(theta, charlie, atom):
    """Atom solves IK to put its fist on Charlie's.

    Keeps the elbow bent the same way as Charlie's.
    """
    target = fist_path(theta, charlie)
    elbow = np.where(theta[:, 1] >= 0, 1, -1)
    out = np.empty_like(target)
    for i, (x, y) in enumerate(target):
        sol = solve_ik(x, y, *atom, elbow[i])
        if sol is None:
            raise ValueError(f"point {i}: out of reach")
        out[i] = sol
    return out


def miss(theta_atom, theta, charlie, atom):
    """Distance (m) from Atom's fist to Charlie's fist."""
    d = fist_path(theta_atom, atom) - fist_path(theta, charlie)
    return np.hypot(d[:, 0], d[:, 1])


def stretched_jab(atom, r0, speed, dt, lam, gain,
                  hold=0.0):
    """Atom's fist straight out to full stretch.

    Returns a dict: t, target path, joint angles under
    the plain inverse and under damped least squares
    (lam, m); gain (1/s) is the error feedback.
    """
    R = atom.L1 + atom.L2
    t, path = straight_jab(r0, R, speed, dt, hold)
    th0 = solve_ik(path[0, 0], path[0, 1], *atom, +1)
    plain = track(path, th0, atom, plain_step, dt, gain)
    dls = track(path, th0, atom,
                lambda J, dx: dls_step(J, dx, lam), dt, gain)
    return {"t": t, "path": path, "R": R, "plain": plain,
            "dls": dls, "theta0": np.array(th0)}


def needed_speeds_vs_gap(atom, gaps, speed):
    """Elbow and shoulder speeds (rad/s) that the exact
    J^-1 asks for, with the fist moving straight out at
    `speed`, `gaps` (m) short of full stretch."""
    R = atom.L1 + atom.L2
    xy = np.column_stack([R - np.asarray(gaps),
                          np.zeros(len(gaps))])
    th = ik_path(xy, *atom, +1)
    return needed_joint_speeds(th, np.array([speed, 0.0]),
                               atom), th
