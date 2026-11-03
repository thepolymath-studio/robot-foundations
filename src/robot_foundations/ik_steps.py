"""Step-by-step IK: nudge the joints so the fist follows a path.

Each tick: dx = the path's next move plus a share of the
fist's current error (resolved-rate control with feedback);
dtheta from one of two rules, then add it.
    plain inverse        dtheta = J^-1 dx
    damped least squares dtheta = J^T (J J^T + lam^2 I)^-1 dx
The plain inverse is exact but asks for huge turns when
det J -> 0 (arm straight or folded). Damped least squares
(Wampler 1986; Nakamura & Hanafusa 1986) gives up a little
accuracy there to keep the turns bounded; lam is in metres.
Research S6-S8; Lynch & Park ch. 6.
"""
from __future__ import annotations

import numpy as np

from .arm import fist_position, jacobian


def plain_step(J, dx):
    """dtheta = J^-1 dx (exact; blows up near det J = 0)."""
    return np.linalg.solve(J, dx)


def dls_step(J, dx, lam):
    """Damped least squares: a bounded J^-1."""
    A = J @ J.T + lam**2 * np.eye(2)
    return J.T @ np.linalg.solve(A, dx)


def track(path, theta0, arm, step, dt, gain):
    """Follow a fist path (N, 2) sampled every dt (s).

    Each tick asks for the path's next move plus
    gain * dt (gain in 1/s) of the fist's current
    error. step(J, dx) -> dtheta. Returns angles (N, 2).
    """
    th = np.array(theta0, float)
    out = [th.copy()]
    for prev, p in zip(path[:-1], path[1:]):
        J = jacobian(*th, *arm)
        err = prev - np.array(fist_position(*th, *arm))
        th = th + step(J, p - prev + gain * dt * err)
        out.append(th.copy())
    return np.array(out)


def needed_joint_speeds(theta, v, arm):
    """Joint speeds (rad/s) that move the fist at velocity
    v (m/s) exactly: J^-1 v, per pose in theta (N, 2)."""
    return np.array([np.linalg.solve(jacobian(*th, *arm), v)
                     for th in theta])
