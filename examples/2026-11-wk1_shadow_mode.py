"""2026-11 wk1 concept post (Real Steel): copy the angles, and
the fist lands somewhere else.

One system: Atom's arm as a planar 2-link arm (shoulder +
elbow, seen from above at shoulder height), and Charlie's as
the same arm, smaller. Charlie: a 1.80 m man with Drillis &
Contini proportions; Atom: 2.29 m, the same arm scaled by
k = 2.29 / 1.80 (chosen, labelled; robot_foundations.bodies).

1. Charlie's jab and hook as joint angles; Atom copies the
   angles. Where does each fist land (shared shoulder)?
2. The reverse question for Atom's arm (IK): the two
   answers for the point Charlie's jab lands on, a point
   out of reach, and det J -> 0 as the arm straightens.
3. Slide-1 loop data: both arms swing the same angles.

Writes outputs/2026-11-wk1/: shadow_loop.npz + shadow_loop.json
(sidecar) and results.json.

    python examples/2026-11-wk1_shadow_mode.py [--check]

--check asserts the numbers in posts/2026-11-wk1-real-steel/
code-results.md (exit 1 on a mismatch). Under a second.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from robot_foundations import arm as A
from robot_foundations import moves as M
from robot_foundations import shadow as S
from robot_foundations.bodies import ATOM, CHARLIE, H_CHARLIE
from robot_foundations.bodies import HAND, K
from robot_foundations.results import check, clean
from robot_foundations.results import write_arrays

OUT = (Path(__file__).resolve().parents[1] / "outputs"
       / "2026-11-wk1")
FPS = 60
DT = 1e-3                        # s, for the punch numbers
UNREACHABLE = (1.00, 0.00)       # m, 1 m straight ahead
IN_HOLE = (0.01, 0.00)           # m, 1 cm from the shoulder
DET_ANGLES = (90, 45, 10, 1, 0)  # elbow angle, deg

# Values quoted in code-results.md, checked by --check: (value, tol)
EXPECTED = {
    "arms.k": (1.272, 0.0005),
    "arms.charlie_L1_cm": (33.5, 0.05),
    "arms.charlie_L2_cm": (36.0, 0.05),
    "arms.charlie_reach_cm": (69.5, 0.05),
    "arms.atom_L1_cm": (42.6, 0.05),
    "arms.atom_L2_cm": (45.8, 0.05),
    "arms.atom_reach_cm": (88.4, 0.05),
    "arms.atom_hole_cm": (3.2, 0.05),
    "arms.charlie_hand_cm": (19.4, 0.05),
    "arms.reach_gain_cm": (18.9, 0.05),
    "punch.jab.charlie_dist_cm": (69.2, 0.05),
    "punch.jab.atom_dist_cm": (88.1, 0.05),
    "punch.jab.miss_cm": (18.8, 0.05),
    "punch.hook.charlie_dist_cm": (49.2, 0.05),
    "punch.hook.atom_dist_cm": (62.5, 0.05),
    "punch.hook.miss_cm": (13.4, 0.05),
    "punch.guard.miss_cm": (9.5, 0.05),
    "punch.ratio_min": (1.272, 0.0005),
    "punch.ratio_max": (1.272, 0.0005),
    "punch.miss_max_cm": (18.8, 0.05),
    "ik.target_cm.0": (66.8, 0.05),
    "ik.target_cm.1": (18.1, 0.05),
    "ik.n_solutions": (2, None),
    "ik.plus.shoulder_deg": (-25.0, 0.05),
    "ik.plus.elbow_deg": (77.0, 0.05),
    "ik.minus.shoulder_deg": (55.3, 0.05),
    "ik.minus.elbow_deg": (-77.0, 0.05),
    "ik.fk_error_max_m": (0.0, 1e-12),
    "ik.unreachable.n_solutions": (0, None),
    "ik.unreachable.beyond_cm": (11.6, 0.05),
    "ik.in_hole.n_solutions": (0, None),
    "jacobian.det_pct.90": (100.0, 0.05),
    "jacobian.det_pct.45": (70.7, 0.05),
    "jacobian.det_pct.10": (17.4, 0.05),
    "jacobian.det_pct.1": (1.7, 0.05),
    "jacobian.det_pct.0": (0.0, 1e-9),
}


def cm(x):
    return 100 * float(x)


def arm_numbers():
    ci, co = A.reach_ring(*CHARLIE)
    ai, ao = A.reach_ring(*ATOM)
    return {"k": K, "charlie_L1_cm": cm(CHARLIE.L1),
            "charlie_L2_cm": cm(CHARLIE.L2),
            "charlie_reach_cm": cm(co), "charlie_hole_cm": cm(ci),
            "atom_L1_cm": cm(ATOM.L1), "atom_L2_cm": cm(ATOM.L2),
            "atom_reach_cm": cm(ao), "atom_hole_cm": cm(ai),
            "charlie_hand_cm": cm(HAND * H_CHARLIE),
            "reach_gain_cm": cm(ao - co)}


def at_time(t, theta_fn=M.angles_at):
    """Both fists for Charlie's angles at time t."""
    th = theta_fn(t)
    pc = S.fist_path(th, CHARLIE)[0]
    pa = S.fist_path(th, ATOM)[0]
    return {"t_s": t, "shoulder_deg": np.degrees(th[0, 0]),
            "elbow_deg": np.degrees(th[0, 1]),
            "charlie_fist_cm": 100 * pc, "atom_fist_cm": 100 * pa,
            "charlie_dist_cm": cm(np.hypot(*pc)),
            "atom_dist_cm": cm(np.hypot(*pa)),
            "miss_cm": cm(np.hypot(*(pa - pc)))}


def punch_numbers():
    """Copy the angles: where the fists land (step 1)."""
    land = M.landing_times()
    out = {n: at_time(land[n] - 1e-9) for n in land}
    out["guard"] = at_time(0.0)
    t = np.arange(0.0, M.LOOP_S, DT)
    th = M.angles_at(t)
    pc, pa = S.fist_path(th, CHARLIE), S.fist_path(th, ATOM)
    ratio = np.hypot(*pa.T) / np.hypot(*pc.T)
    miss = S.miss(th, th, CHARLIE, ATOM)
    out.update(ratio_min=ratio.min(), ratio_max=ratio.max(),
               miss_max_cm=cm(miss.max()),
               miss_min_cm=cm(miss.min()))
    return out


def ik_numbers():
    """Atom's arm: two answers, none, and det J."""
    th_jab = M.angles_at(M.landing_times()["jab"] - 1e-9)
    tx, ty = S.fist_path(th_jab, CHARLIE)[0]
    res = {"target_cm": [cm(tx), cm(ty)],
           "n_solutions": A.n_solutions(tx, ty, *ATOM)}
    errs = []
    for name, e in (("plus", +1), ("minus", -1)):
        t1, t2 = A.solve_ik(tx, ty, *ATOM, e)
        fx, fy = A.fist_position(t1, t2, *ATOM)
        errs.append(np.hypot(fx - tx, fy - ty))
        res[name] = {"shoulder_deg": np.degrees(t1),
                     "elbow_deg": np.degrees(t2),
                     "elbow_cm": [cm(v) for v in
                                  A.elbow_position(t1, ATOM.L1)]}
    res["fk_error_max_m"] = max(errs)
    ux, uy = UNREACHABLE
    res["unreachable"] = {
        "target_cm": [cm(ux), cm(uy)],
        "n_solutions": A.n_solutions(ux, uy, *ATOM),
        "solve_ik": A.solve_ik(ux, uy, *ATOM),
        "beyond_cm": cm(np.hypot(ux, uy) - sum(ATOM))}
    hx, hy = IN_HOLE
    res["in_hole"] = {"target_cm": [cm(hx), cm(hy)],
                      "n_solutions": A.n_solutions(hx, hy, *ATOM)}
    return res


def jacobian_numbers():
    big = A.det_jacobian(np.pi / 2, *ATOM)
    pct = {str(a): 100 * A.det_jacobian(np.radians(a), *ATOM)
           / big for a in DET_ANGLES}
    return {"det_max_m2": big, "det_pct": pct}


def results():
    return {"arms": arm_numbers(), "punch": punch_numbers(),
            "ik": ik_numbers(), "jacobian": jacobian_numbers()}


def loop_arrays(res):
    t = np.arange(int(round(M.LOOP_S * FPS))) / FPS
    th = M.angles_at(t)
    ce = np.column_stack(A.elbow_position(th[:, 0], CHARLIE.L1))
    ae = np.column_stack(A.elbow_position(th[:, 0], ATOM.L1))
    ik = res["ik"]
    t2 = np.linspace(-180, 180, 721)
    return {
        "t_s": t,
        "theta_deg": np.degrees(th),
        "charlie_elbow_m": ce,
        "charlie_fist_m": S.fist_path(th, CHARLIE),
        "atom_elbow_m": ae,
        "atom_fist_m": S.fist_path(th, ATOM),
        "miss_m": S.miss(th, th, CHARLIE, ATOM),
        "shoulder_m": np.zeros(2),
        "charlie_ring_m": np.array(A.reach_ring(*CHARLIE)),
        "atom_ring_m": np.array(A.reach_ring(*ATOM)),
        "ik_target_m": np.array(ik["target_cm"]) / 100,
        "ik_elbow_plus_m": np.array(ik["plus"]["elbow_cm"]) / 100,
        "ik_elbow_minus_m":
            np.array(ik["minus"]["elbow_cm"]) / 100,
        "unreachable_m": np.array(UNREACHABLE),
        "det_curve_elbow_deg": t2,
        "det_curve_pct": 100 * np.sin(np.radians(t2)),
    }


DESC = {
    "t_s": ("s", f"loop clock, {FPS} fps, 0 .. 3 s (frame 0 "
            "follows the last frame: guard to guard)"),
    "theta_deg": ("deg", "Charlie's joint angles [t, (shoulder,"
                  " elbow)]; Atom copies them exactly"),
    "charlie_elbow_m": ("m", "Charlie's elbow [t, xy]"),
    "charlie_fist_m": ("m", "Charlie's fist [t, xy]"),
    "atom_elbow_m": ("m", "Atom's elbow, same angles [t, xy]"),
    "atom_fist_m": ("m", "Atom's fist, same angles [t, xy] "
                    "= 1.272 x charlie_fist_m"),
    "miss_m": ("m", "distance Atom's fist - Charlie's fist [t]"),
    "shoulder_m": ("m", "the shared shoulder (origin)"),
    "charlie_ring_m": ("m", "reachable ring: inner, outer "
                       "radius"),
    "atom_ring_m": ("m", "Atom's reachable ring: inner, outer"),
    "ik_target_m": ("m", "IK figure: the point Charlie's jab "
                    "lands on"),
    "ik_elbow_plus_m": ("m", "Atom's elbow, answer 1 (bent "
                        "like Charlie's); fist on ik_target_m"),
    "ik_elbow_minus_m": ("m", "Atom's elbow, answer 2 (bent the"
                         " other way)"),
    "unreachable_m": ("m", "a point out of Atom's reach (1 m "
                      "straight ahead)"),
    "det_curve_elbow_deg": ("deg", "elbow angle for the det J "
                            "curve; 0 = straight arm"),
    "det_curve_pct": ("%", "det J as % of its largest value "
                      "(= sin of the elbow angle)"),
}


def meta(res):
    return {
        "frame": "seen from above at shoulder height; origin = "
                 "the shared shoulder; x towards the opponent, "
                 "y across the body (towards the chin)",
        "units": "m, s, deg",
        "time_base": "t_s = 0 is the start of the guard pose; "
                     "jab lands at 0.50 s, hook at 1.80 s; loops"
                     " at 3.0 s",
        "model": "planar 2-link arm (shoulder + elbow); joint "
                 "angles on minimum-jerk curves",
        "assumptions": "Charlie 1.80 m with Drillis & Contini "
                       "proportions (chosen); Atom = Charlie x "
                       f"{K:.3f} (2.29 m, research F11); tempo "
                       "chosen for display, slower than a real "
                       "punch",
        "loop_hint": "draw Charlie's arm thin/grey and Atom's "
                     "yellow from one shoulder; the fists trace "
                     "the same shape, Atom's 1.27x bigger; "
                     f"miss is {res['punch']['jab']['miss_cm']:.1f}"
                     " cm when the jab lands",
    }


def report(res):
    a, p, ik = res["arms"], res["punch"], res["ik"]
    print(f"  k = {a['k']:.3f}; reach Charlie "
          f"{a['charlie_reach_cm']:.1f} cm, Atom "
          f"{a['atom_reach_cm']:.1f} cm")
    for n in ("guard", "jab", "hook"):
        print(f"  {n:5s}: Charlie {p[n]['charlie_dist_cm']:.1f} "
              f"cm, Atom {p[n]['atom_dist_cm']:.1f} cm, miss "
              f"{p[n]['miss_cm']:.1f} cm")
    for n in ("plus", "minus"):
        print(f"  IK {n}: shoulder {ik[n]['shoulder_deg']:.1f}, "
              f"elbow {ik[n]['elbow_deg']:.1f} deg")
    print("  det J %:", {k: round(v, 1) for k, v in
                         res["jacobian"]["det_pct"].items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = clean(results())
    write_arrays(OUT, "shadow_loop", loop_arrays(res), DESC,
                 meta(res))
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    report(res)
    print(f"wrote {OUT}")
    if args.check:
        bad = check(res, EXPECTED)
        for b in bad:
            print("FAIL", b)
        print("CHECK FAILED" if bad else "CHECK OK")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
