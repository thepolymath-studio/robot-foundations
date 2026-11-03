"""Charlie's and Atom's arms (chosen, labelled; not film data).

Charlie: a man H = 1.80 m tall (a chosen round height, not
Hugh Jackman's), with the Drillis & Contini segment lengths
as fractions of height (Winter, Biomechanics and Motor
Control of Human Movement, fig. 4.1):
    upper arm 0.186 H, forearm 0.146 H, hand 0.108 H.
The forearm link runs from the elbow to the middle of the
closed fist: forearm + half the hand = 0.200 H.

Atom: 2.29 m tall (7 ft 6 in, Legacy Effects via Core77;
research F11). Assumption: Atom is Charlie scaled up
uniformly, k = 2.29 / 1.80, every link k times longer.
The plane: seen from above at shoulder height; x points at
the opponent, y across the body (towards the chin).
"""
from __future__ import annotations

from typing import NamedTuple

H_CHARLIE = 1.80          # m, chosen
H_ATOM = 2.29             # m, research F11
UPPER_ARM = 0.186         # x H (Drillis & Contini)
FOREARM = 0.146           # x H
HAND = 0.108              # x H
K = H_ATOM / H_CHARLIE    # Atom / Charlie


class Arm(NamedTuple):
    L1: float             # shoulder to elbow, m
    L2: float             # elbow to middle of fist, m


def human_arm(height):
    """A 2-link arm with human proportions."""
    return Arm(UPPER_ARM * height,
               (FOREARM + HAND / 2) * height)


def scaled(arm, k):
    """The same arm with every link k times longer."""
    return Arm(k * arm.L1, k * arm.L2)


CHARLIE = human_arm(H_CHARLIE)
ATOM = scaled(CHARLIE, K)
