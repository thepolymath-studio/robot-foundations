# robot-foundations

The tested robotics behind [@thepolymath.studio](https://instagram.com/thepolymath.studio)'s
robotics posts. Every number a post quotes is produced by a script in `examples/` and checked by
`tests/`.

```bash
pip install -e ".[test]"
pytest
```

The models live in `src/robot_foundations/`. Numerics come from numpy and scipy. The kinematics
are written out by hand, because they are what the posts explain, and the tests check them against
the Denavit–Hartenberg product, finite differences and scipy's least-squares solver.

| module | what |
|---|---|
| `arm.py` | planar 2-link arm: forward kinematics, closed-form IK (both elbow branches), the reachable ring, the Jacobian and det J = L1 L2 sin θ2 |
| `dh.py` | the general chain T = T1(θ1) T2(θ2) … Tn(θn) in Denavit–Hartenberg form; the planar arm as two DH rows |
| `bodies.py` | Charlie's and Atom's arms (chosen and labelled, see below) |
| `moves.py` | Charlie's jab and hook as joint angles on minimum-jerk curves; a straight-out jab path |
| `ik_steps.py` | step-by-step IK: the plain inverse J⁻¹ and damped least squares, resolved-rate tracking |
| `shadow.py` | shadow mode: copy the angles vs copy the fist; the stretched jab |
| `results.py` | JSON-safe results, the `--check` comparison, npz + JSON sidecar writer |

## 2026-11 wk1: Real Steel, copy the angles and the fist lands somewhere else

In the film's final fight, Atom's voice control is damaged and Charlie shadow-boxes outside the
ring while Atom copies him. The film never says *how* Atom copies. The simplest reading is joint
by joint: Atom takes Charlie's angles. Forward kinematics says where that puts Atom's fist.

One system, both posts: an arm as a planar 2-link arm (one shoulder, one elbow, a flat plane seen
from above at shoulder height; x points at the opponent). A real arm has more joints.

- **Charlie** is a man 1.80 m tall (a chosen round height). Segment lengths follow Drillis &
  Contini as fractions of height (Winter, *Biomechanics and Motor Control of Human Movement*,
  fig. 4.1): upper arm 0.186 H = 33.5 cm, forearm to the middle of the fist (forearm + half the
  hand) 0.200 H = 36.0 cm. Reach 69.5 cm.
- **Atom** is 2.29 m tall (7 ft 6 in, Legacy Effects via Core77). We assume he is Charlie
  scaled up evenly, k = 2.29 / 1.80 = 1.272: upper arm 42.6 cm, forearm 45.8 cm, reach 88.4 cm.
  The film gives no arm lengths.
- **The punches** are chosen poses (guard, jab, hook) joined by minimum-jerk curves (Flash &
  Hogan 1985), in a 3 s loop. The tempo is for display and is slower than a real punch.

```bash
python examples/2026-11-wk1_shadow_mode.py --check    # < 1 s
```

Both arms share one shoulder, so Atom's target is simply Charlie's fist. Copying the angles puts
Atom's fist exactly 1.272 times as far out, in the same direction: it misses by 9.5 cm in guard,
18.8 cm when the jab lands and 13.4 cm when the hook lands. The reverse question (IK) for the
point the jab lands on has two answers for Atom's arm (elbow 77.0° one way or the other); a point
1 m ahead is 11.6 cm beyond reach and has none; det J falls to 17.4 % of its largest value at a
10° elbow, 1.7 % at 1° and 0 when the arm is straight.

This writes `outputs/2026-11-wk1/`: `shadow_loop.npz` + the `shadow_loop.json` sidecar (the
Slide-1 loop at 60 fps: both arms on the same angles from one shoulder, the reach rings, the two IK
elbows, the det J curve) and `results.json`.

## 2026-11 wk1 Sim post: shadow mode in code

The same arms. **Step 1:** copy the angles or copy the fist? Over the whole jab-and-hook, Atom
either takes Charlie's angles or solves IK (closed form, elbow bent the same way as Charlie's) for
the point Charlie's fist is on. Copying the angles misses by 9.5 to 18.8 cm all the way through;
copying the fist misses by 2.5e-16 m (rounding). To land the jab on Charlie's point, Atom bends
his elbow 77.0° instead of Charlie's 10°, at 78.3 % of his reach.

**Step 2:** the stretched jab. Atom's fist goes straight out at 5 m/s (chosen) from 60 cm to full
stretch (88.4 cm, 0.057 s). Exact joint speeds J⁻¹v for that fist motion grow as one over the root
of the gap: 1,404°/s at 10 cm short, 4,324°/s at 1 cm, 13,640°/s at 1 mm, 43,122°/s at 0.1 mm,
and no finite answer at full stretch (det J = 0). A controller that steps with the plain inverse
asks for more the faster it ticks (peak 4,314 / 12,480 / 33,958°/s at 1 ms / 0.1 ms / 0.01 ms
ticks). Damped least squares (λ = 2 cm, error feedback 50/s) converges to a peak of 5,686°/s
and arrives 3.4 mm short of the target, creeping to 0.37 mm after a 0.1 s hold. More damping means
gentler joints and a bigger miss (λ = 1 / 2 / 5 / 10 cm: 10,033 / 5,686 / 2,754 / 1,674°/s and
1.2 / 3.4 / 12.8 / 31.9 mm).

```bash
python examples/2026-11-wk1_shadow_sim.py --check     # ~2 s
```

This writes `outputs/2026-11-wk1-sim/`: `shadow_sim.npz` + the `shadow_sim.json` sidecar (step 1
on the loop clock, both copies; step 2 every 0.5 ms with angles, fists, misses and joint speeds for
both rules, plus the needed-speed curve against the gap) and `results.json`.

Idealisations: planar, two joints, no masses or motor limits (joint speeds are what the
kinematics *asks for*, not what a motor could give), no delay between Charlie and Atom.

Tests (`tests/test_arm.py`, `test_bodies_moves.py`, `test_shadow.py`, `test_ik_steps.py`):

- FK special poses (straight, folded, right angle) and the law of cosines on 1,000 random poses;
  FK equals the DH product T1 T2, rotation part included;
- IK round trips to 1e-12 on both branches; the two elbows are mirror images across the
  shoulder–fist line; scipy's least squares finds the same poses; 2 / 1 / 0 answers inside, on
  the edge and outside the ring (and in the hole);
- the Jacobian equals central finite differences; det J equals L1 L2 sin θ2, is 0 straight and
  folded, and a straight arm cannot move its fist along itself;
- copying the angles gives p′ = k p exactly; unequal proportions (a longer forearm only) change
  the shape, not just the size; copying the fist lands on it to 1e-12 and flags points out of reach;
- damped least squares equals its SVD form Σ σ/(σ² + λ²), tends to J⁻¹ as λ → 0 and is bounded by
  1/(2λ) at the singularity, where the plain inverse fails;
- the needed elbow speed equals the closed form r v / (L1 L2 sin θ2) and grows by √10 for every
  10× closer to full stretch; the plain inverse's peak grows with the tick rate while damped least
  squares converges; its miss shrinks with λ;
- every number the posts quote (both scripts' `EXPECTED`), and determinism.
