"""2026-11 wk1 Sim post (Real Steel): shadow mode in code.

The concept post's system: Charlie's arm (1.80 m man, Drillis
& Contini proportions) and Atom's, the same arm 1.272x longer
(2.29 m). One shared shoulder, seen from above.

Step 1. Copy the angles or copy the fist? Over Charlie's
   jab-and-hook, Atom (a) takes Charlie's joint angles, or
   (b) solves IK for the point Charlie's fist is on. The
   miss distance between the fists, all the way through.
Step 2. The stretched jab. Atom's fist goes straight out at
   5 m/s (chosen) to full stretch. Joint speeds asked for by
   the plain inverse J^-1 grow without limit; damped least
   squares stays bounded at a small, measured miss.

Writes outputs/2026-11-wk1-sim/: shadow_sim.npz +
shadow_sim.json (sidecar) and results.json.

    python examples/2026-11-wk1_shadow_sim.py [--check]

--check asserts the numbers in posts/2026-11-wk1-real-steel-sim/
code-results.md (exit 1 on a mismatch). A few seconds.
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
from robot_foundations.bodies import ATOM, CHARLIE
from robot_foundations.results import check, clean
from robot_foundations.results import write_arrays

OUT = (Path(__file__).resolve().parents[1] / "outputs"
       / "2026-11-wk1-sim")
FPS = 60
DT1 = 1e-3                    # s, step 1 sampling
R0 = 0.60                     # m, stretched jab starts here
SPEED = 5.0                   # m/s, chosen fist speed
LAM = 0.02                    # m, damping
GAIN = 50.0                   # 1/s, error feedback
DT = 1e-5                     # s, controller tick (converged)
TICKS = {"1ms": 1e-3, "100us": 1e-4, "10us": 1e-5}
LAMS = {"10mm": 0.01, "20mm": 0.02, "50mm": 0.05,
        "100mm": 0.10}
HOLD = 0.10                   # s held at full stretch
GAPS = {"10cm": 0.10, "1cm": 0.01, "1mm": 1e-3,
        "0p1mm": 1e-4}            # m short of full stretch
STEP2_FRAME = 50              # keep every 50th tick (0.5 ms)

# Values quoted in code-results.md, checked by --check: (value, tol)
EXPECTED = {
    "step1.angles.miss_max_cm": (18.8, 0.05),
    "step1.angles.miss_min_cm": (9.5, 0.05),
    "step1.angles.miss_jab_cm": (18.8, 0.05),
    "step1.angles.miss_hook_cm": (13.4, 0.05),
    "step1.fist.miss_max_m": (0.0, 1e-12),
    "step1.fist.jab_elbow_deg": (77.0, 0.05),
    "step1.angles.jab_elbow_deg": (10.0, 0.05),
    "step1.fist.jab_reach_pct": (78.3, 0.05),
    "step1.fist.min_bend_any_reach_deg": (76.4, 0.05),
    "step1.fist.min_bend_at_cm": (69.5, 0.05),
    "step1.fist.min_bend_loop_deg": (77.0, 0.05),
    "step2.punch_s": (0.0568, 0.00005),
    "step2.needed_elbow_deg_s.10cm": (1404, 1),
    "step2.needed_elbow_deg_s.1cm": (4324, 1),
    "step2.needed_elbow_deg_s.1mm": (13640, 1),
    "step2.needed_elbow_deg_s.0p1mm": (43122, 1),
    "step2.plain_peak_elbow_deg_s.1ms": (4314, 1),
    "step2.plain_peak_elbow_deg_s.100us": (12480, 1),
    "step2.plain_peak_elbow_deg_s.10us": (33958, 1),
    "step2.dls_peak_elbow_deg_s.1ms": (3623, 1),
    "step2.dls_peak_elbow_deg_s.100us": (5351, 1),
    "step2.dls_peak_elbow_deg_s.10us": (5686, 1),
    "step2.dls.miss_arrive_mm": (3.4, 0.05),
    "step2.dls.miss_max_mm": (3.4, 0.05),
    "step2.dls.miss_end_mm": (0.37, 0.005),
    "step2.dls.elbow_end_deg": (3.3, 0.05),
    "step2.needed_elbow_rpm.10cm": (234, 1),
    "step2.needed_elbow_rpm.1cm": (721, 1),
    "step2.needed_elbow_rpm.1mm": (2273, 1),
    "step2.needed_elbow_rpm.0p1mm": (7187, 1),
    "step2.dls.peak_elbow_rpm": (948, 1),
    "step2.lam_sweep.10mm.peak_elbow_deg_s": (10033, 1),
    "step2.lam_sweep.20mm.peak_elbow_deg_s": (5686, 1),
    "step2.lam_sweep.50mm.peak_elbow_deg_s": (2754, 1),
    "step2.lam_sweep.100mm.peak_elbow_deg_s": (1674, 1),
    "step2.lam_sweep.10mm.miss_arrive_mm": (1.2, 0.05),
    "step2.lam_sweep.20mm.miss_arrive_mm": (3.4, 0.05),
    "step2.lam_sweep.50mm.miss_arrive_mm": (12.8, 0.05),
    "step2.lam_sweep.100mm.miss_arrive_mm": (31.9, 0.05),
}


def min_bend_any_reach():
    """Smallest elbow bend (deg) Atom needs to copy the
    fist anywhere Charlie can reach: at Charlie's full
    reach (the bend only grows as the point comes in)."""
    r = np.linspace(*A.reach_ring(*CHARLIE), 20001)
    r = r[r >= A.reach_ring(*ATOM)[0]]
    th = A.ik_path(np.column_stack([r, 0 * r]), *ATOM)
    i = np.argmin(np.abs(th[:, 1]))
    return np.degrees(abs(th[i, 1])), 100 * r[i]


def step1():
    """Copy the angles vs copy the fist, at DT1."""
    t = np.arange(0.0, M.LOOP_S, DT1)
    th = M.angles_at(t)
    a = S.copy_angles(th, ATOM)
    f = S.copy_fist(th, CHARLIE, ATOM)
    ma = S.miss(a, th, CHARLIE, ATOM)
    mf = S.miss(f, th, CHARLIE, ATOM)
    land = M.landing_times()
    ij = int(round(land["jab"] / DT1)) - 1
    ih = int(round(land["hook"] / DT1)) - 1
    pj = S.fist_path(f[ij:ij + 1], ATOM)[0]
    bend, at_cm = min_bend_any_reach()
    return {
        "angles": {"miss_max_cm": 100 * ma.max(),
                   "miss_min_cm": 100 * ma.min(),
                   "miss_jab_cm": 100 * ma[ij],
                   "miss_hook_cm": 100 * ma[ih],
                   "jab_elbow_deg": np.degrees(a[ij, 1])},
        "fist": {"miss_max_m": mf.max(),
                 "min_bend_any_reach_deg": bend,
                 "min_bend_at_cm": at_cm,
                 "min_bend_loop_deg":
                     np.degrees(np.abs(f[:, 1]).min()),
                 "jab_elbow_deg": np.degrees(f[ij, 1]),
                 "jab_reach_pct": 100 * np.hypot(*pj)
                 / sum(ATOM)}}


def speeds(theta, dt):
    """Joint speeds (deg/s) between ticks, (N-1, 2)."""
    return np.degrees(np.diff(theta, axis=0)) / dt


def fist_miss(run, key):
    d = S.fist_path(run[key], ATOM) - run["path"]
    return np.hypot(d[:, 0], d[:, 1])


def peak_elbow(run, key, dt):
    return np.abs(speeds(run[key], dt)[:, 1]).max()


def step2():
    """The stretched jab: plain inverse vs DLS."""
    w, _ = S.needed_speeds_vs_gap(
        ATOM, np.array(list(GAPS.values())), SPEED)
    out = {"punch_s": (sum(ATOM) - R0) / SPEED,
           "needed_elbow_deg_s": {
               g: abs(np.degrees(v[1]))
               for g, v in zip(GAPS, w)},
           "plain_peak_elbow_deg_s": {},
           "dls_peak_elbow_deg_s": {}, "lam_sweep": {}}
    for name, dt in TICKS.items():
        r = S.stretched_jab(ATOM, R0, SPEED, dt, LAM, GAIN)
        out["plain_peak_elbow_deg_s"][name] = \
            peak_elbow(r, "plain", dt)
        out["dls_peak_elbow_deg_s"][name] = \
            peak_elbow(r, "dls", dt)
    for name, lam in LAMS.items():
        r = S.stretched_jab(ATOM, R0, SPEED, DT, lam, GAIN, HOLD)
        out["lam_sweep"][name] = summary(r, "dls")
    run = S.stretched_jab(ATOM, R0, SPEED, DT, LAM, GAIN, HOLD)
    out["dls"] = summary(run, "dls")
    out["plain"] = summary(run, "plain")
    out["needed_elbow_rpm"] = {      # turns per minute
        g: v / 6 for g, v in out["needed_elbow_deg_s"].items()}
    out["dls"]["peak_elbow_rpm"] = \
        out["dls"]["peak_elbow_deg_s"] / 6
    return out, run


def summary(run, key):
    m = fist_miss(run, key)
    n = int(round((run["R"] - R0) / (SPEED * DT)))
    return {"peak_elbow_deg_s": peak_elbow(run, key, DT),
            "miss_arrive_mm": 1e3 * m[n],
            "miss_max_mm": 1e3 * m.max(),
            "miss_end_mm": 1e3 * m[-1],
            "elbow_end_deg": np.degrees(run[key][-1, 1])}


def results():
    s2, run = step2()
    return {"step1": step1(), "step2": s2}, run


def step1_arrays():
    t = np.arange(int(round(M.LOOP_S * FPS))) / FPS
    th = M.angles_at(t)
    a = S.copy_angles(th, ATOM)
    f = S.copy_fist(th, CHARLIE, ATOM)
    return {
        "s1_t_s": t,
        "s1_charlie_theta_deg": np.degrees(th),
        "s1_charlie_fist_m": S.fist_path(th, CHARLIE),
        "s1_angles_theta_deg": np.degrees(a),
        "s1_angles_fist_m": S.fist_path(a, ATOM),
        "s1_angles_miss_m": S.miss(a, th, CHARLIE, ATOM),
        "s1_fist_theta_deg": np.degrees(f),
        "s1_fist_fist_m": S.fist_path(f, ATOM),
        "s1_fist_miss_m": S.miss(f, th, CHARLIE, ATOM),
    }


def step2_arrays(run):
    k = slice(0, None, STEP2_FRAME)
    gaps = np.geomspace(1e-4, sum(ATOM) - R0, 200)
    w, _ = S.needed_speeds_vs_gap(ATOM, gaps, SPEED)
    out = {"s2_t_s": run["t"][k],
           "s2_target_m": run["path"][k]}
    for key in ("plain", "dls"):
        sp = np.vstack([speeds(run[key], DT), [0.0, 0.0]])
        out[f"s2_{key}_theta_deg"] = np.degrees(run[key][k])
        out[f"s2_{key}_fist_m"] = S.fist_path(run[key][k], ATOM)
        out[f"s2_{key}_miss_m"] = fist_miss(run, key)[k]
        out[f"s2_{key}_speed_deg_s"] = sp[k]
    out["s2_gap_m"] = gaps
    out["s2_needed_elbow_deg_s"] = np.abs(np.degrees(w[:, 1]))
    out["s2_needed_shoulder_deg_s"] = np.abs(np.degrees(w[:, 0]))
    return out


DESC = {
    "s1_t_s": ("s", f"step 1 clock, {FPS} fps, 0 .. 3 s, loops"
               " (the concept post's jab-and-hook)"),
    "s1_charlie_theta_deg": ("deg", "Charlie's (shoulder, elbow)"
                             " [t, 2]"),
    "s1_charlie_fist_m": ("m", "Charlie's fist = Atom's target "
                          "[t, xy]"),
    "s1_angles_theta_deg": ("deg", "Atom copying the angles "
                            "(= Charlie's) [t, 2]"),
    "s1_angles_fist_m": ("m", "Atom's fist, copying the angles "
                         "[t, xy]"),
    "s1_angles_miss_m": ("m", "Atom's fist to Charlie's, copying"
                         " the angles [t]"),
    "s1_fist_theta_deg": ("deg", "Atom copying the fist (IK, "
                          "elbow bent like Charlie's) [t, 2]"),
    "s1_fist_fist_m": ("m", "Atom's fist, copying the fist "
                       "[t, xy] (on Charlie's)"),
    "s1_fist_miss_m": ("m", "copying the fist: miss (~1e-16, "
                       "rounding) [t]"),
    "s2_t_s": ("s", "step 2 clock, every 0.5 ms; 0 = fist at "
               "60 cm; full stretch at 0.0568 s; held to "
               "0.1568 s"),
    "s2_target_m": ("m", "where Atom's fist should be [t, xy] "
                    "(straight out along +x at 5 m/s)"),
    "s2_plain_theta_deg": ("deg", "plain inverse: (shoulder, "
                           "elbow) [t, 2]"),
    "s2_plain_fist_m": ("m", "plain inverse: fist [t, xy]"),
    "s2_plain_miss_m": ("m", "plain inverse: fist to target [t]"),
    "s2_plain_speed_deg_s": ("deg/s", "plain inverse: joint "
                             "speeds [t, (shoulder, elbow)], "
                             "per 10 us tick"),
    "s2_dls_theta_deg": ("deg", "damped least squares: "
                         "(shoulder, elbow) [t, 2]"),
    "s2_dls_fist_m": ("m", "damped least squares: fist [t, xy]"),
    "s2_dls_miss_m": ("m", "damped least squares: fist to target"
                      " [t]"),
    "s2_dls_speed_deg_s": ("deg/s", "damped least squares: joint"
                           " speeds [t, (shoulder, elbow)]"),
    "s2_gap_m": ("m", "distance short of full stretch, 0.1 mm "
                 ".. 28.4 cm (log-spaced) [g]"),
    "s2_needed_elbow_deg_s": ("deg/s", "elbow speed the exact "
                              "J^-1 asks for at that gap [g]; "
                              "no limit as gap -> 0"),
    "s2_needed_shoulder_deg_s": ("deg/s", "same, shoulder [g]"),
}


def meta(res):
    d = res["step2"]["dls"]
    return {
        "frame": "seen from above at shoulder height; origin = "
                 "the shared shoulder; x towards the opponent, "
                 "y across the body",
        "units": "m, s, deg, deg/s",
        "time_base": "s1_*: the concept loop's clock (jab lands"
                     " 0.50 s, hook 1.80 s); s2_*: t = 0 when "
                     "the stretched jab starts",
        "model": "planar 2-link arm; step 1 closed-form IK; "
                 "step 2 resolved-rate control, dtheta = "
                 "step(J, next move + 50/s x dt x error), "
                 "plain J^-1 or damped least squares "
                 f"(lambda {LAM} m)",
        "assumptions": "Charlie 1.80 m, Atom = Charlie x 1.272 "
                       "(2.29 m); fist speed 5 m/s chosen; "
                       "joint motors have no speed limit",
        "step1_hint": "two Atoms from one shoulder over "
                      "Charlie's ghost: copying the angles "
                      "overshoots (miss 9.5 .. 18.8 cm), copying"
                      " the fist lands on it",
        "step2_hint": "plot s2_needed_elbow_deg_s vs s2_gap_m "
                      "(log-log: a straight line, slope -1/2); "
                      "DLS peaks at "
                      f"{d['peak_elbow_deg_s']:.0f} deg/s and ends"
                      f" {d['miss_arrive_mm']:.1f} mm short",
    }


def report(res):
    s1, s2 = res["step1"], res["step2"]
    print(f"  step 1: copy angles miss "
          f"{s1['angles']['miss_min_cm']:.1f}.."
          f"{s1['angles']['miss_max_cm']:.1f} cm; copy fist "
          f"{s1['fist']['miss_max_m']:.1e} m")
    print("  step 2 needed elbow deg/s by gap cm:",
          {k: round(v) for k, v in
           s2["needed_elbow_deg_s"].items()})
    print("  plain peak by tick:", {k: round(v) for k, v in
                                    s2["plain_peak_elbow_deg_s"]
                                    .items()})
    print("  dls peak by tick:", {k: round(v) for k, v in
                                  s2["dls_peak_elbow_deg_s"]
                                  .items()})
    print("  dls:", {k: round(v, 3) for k, v in
                     s2["dls"].items()})
    for lam, v in s2["lam_sweep"].items():
        print(f"  lam {lam}: peak {v['peak_elbow_deg_s']:.0f}"
              f" deg/s, miss {v['miss_arrive_mm']:.2f} mm")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res, run = results()
    res = clean(res)
    arrays = step1_arrays() | step2_arrays(run)
    write_arrays(OUT, "shadow_sim", arrays, DESC, meta(res))
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
