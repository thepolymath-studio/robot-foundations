"""Charlie's jab and hook as joint angles over time.

Chosen poses (degrees, shoulder t1 / elbow t2) in the plane
of bodies.py (seen from above, x towards the opponent):
    guard      -40 / 120   fist in front of the chin
    jab         10 /  10   lead arm almost straight
    hook_load  -80 /  90   elbow out to the side
    hook        -20 /  90   fist swept across, elbow at 90
Between poses the joints move along a minimum-jerk curve
(Flash & Hogan 1985), the smooth shape of a human reach.
The tempo is chosen for display (a 3 s loop, a 0.2 s jab):
slower than a real punch. Starts and ends in guard, so it
loops.
"""
from __future__ import annotations

import numpy as np

POSES_DEG = {"guard": (-40.0, 120.0), "jab": (10.0, 10.0),
             "hook_load": (-80.0, 90.0), "hook": (-20.0, 90.0)}

# (pose reached at the end, duration s); same pose = hold
SEQUENCE = [("guard", 0.30), ("jab", 0.20), ("jab", 0.05),
            ("guard", 0.30), ("guard", 0.40),
            ("hook_load", 0.30), ("hook", 0.25),
            ("hook", 0.05), ("guard", 0.40), ("guard", 0.75)]
LOOP_S = sum(d for _, d in SEQUENCE)


def min_jerk(s):
    """Smooth 0 -> 1 with zero speed and acceleration
    at both ends (Flash & Hogan 1985)."""
    s = np.clip(s, 0.0, 1.0)
    return s**3 * (10 - 15 * s + 6 * s * s)


def key_times():
    """Start time of each segment and its end pose."""
    t, out = 0.0, []
    for name, dur in SEQUENCE:
        out.append((t, t + dur, name))
        t += dur
    return out


def angles_at(t):
    """Charlie's joint angles (rad) at time t (s), shape
    (len(t), 2); t wraps around the loop."""
    t = np.mod(np.atleast_1d(np.asarray(t, float)), LOOP_S)
    th = np.empty((t.size, 2))
    prev = np.radians(POSES_DEG[SEQUENCE[-1][0]])
    for t0, t1, name in key_times():
        nxt = np.radians(POSES_DEG[name])
        m = (t >= t0) & (t < t1)
        s = min_jerk((t[m] - t0) / (t1 - t0))[:, None]
        th[m] = prev + s * (nxt - prev)
        prev = nxt
    return th


def landing_times():
    """When the jab and the hook land (end of the swing)."""
    out = {}
    for t0, t1, name in key_times():
        if name in ("jab", "hook") and name not in out:
            out[name] = t1
    return out


def straight_jab(r0, r1, speed, dt, hold=0.0):
    """Fist path straight out along +x at constant speed.

    From r0 to r1 (m) at `speed` (m/s), sampled every dt
    (s), then held at r1 for `hold` s. Returns t, xy.
    """
    n = int(round((r1 - r0) / (speed * dt)))
    t = np.arange(n + 1 + int(round(hold / dt))) * dt
    x = np.minimum(r0 + speed * t, r1)
    return t, np.column_stack([x, np.zeros_like(x)])
