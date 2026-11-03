"""Denavit-Hartenberg chain: T = T1(t1) T2(t2) ... Tn(tn).

Each joint is four numbers (theta, d, a, alpha), classic
convention (Denavit & Hartenberg 1955; Craig ch. 3 uses the
modified one). A planar 2-link arm is the rows
(t1, 0, L1, 0) and (t2, 0, L2, 0); its T[:2, 3] is
arm.fist_position. Used as the general form the 2-link
formula is a special case of (research S1, S2).
"""
from __future__ import annotations

import numpy as np


def dh_transform(theta, d, a, alpha):
    """One joint's 4x4 transform (classic DH)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st * ca, st * sa, a * ct],
                     [st, ct * ca, -ct * sa, a * st],
                     [0.0, sa, ca, d],
                     [0.0, 0.0, 0.0, 1.0]])


def chain(rows):
    """Multiply the joints' transforms: T = T1 T2 ... Tn."""
    T = np.eye(4)
    for theta, d, a, alpha in rows:
        T = T @ dh_transform(theta, d, a, alpha)
    return T


def planar_rows(t1, t2, L1, L2):
    """DH rows of the planar 2-link arm."""
    return [(t1, 0.0, L1, 0.0), (t2, 0.0, L2, 0.0)]
