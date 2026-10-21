# Week 2: click-to-reach inverse kinematics

Forward kinematics asks where the hand goes for given joint angles. Inverse
kinematics (IK) asks which joint angles put the hand at a target position.
This lab solves position IK for the existing planar two-link arm. Both joints
can turn freely; there are no collision or joint-limit constraints.

## Try it

Run `.venv/bin/python math/arm_viewer.py` from this checkout, or from the original
repo run `.venv/bin/python .worktrees/week2-planar-arm/math/arm_viewer.py` while
PR #6 is being reviewed. Open http://127.0.0.1:8765 and choose **Hand target · IK**.

1. Click or drag in Plan view, or enter x and y in metres. The crosshair marks
   the target; Python computes the solutions and verifies them through FK.
2. Choose **Example** with the default 1 m / 0.75 m links. The target is
   (1,0.75) m. Elbow-down is (q1,q2) = (0°,90°); elbow-up is approximately
   (73.74°,−90°). Switch branches: the hand position stays fixed, while its
   heading changes. The dashed arm and hollow configuration-space point show
   the alternate solution. Toggle Show alternate solution to hide the dashed arm.
3. Choose **Origin**. It lies in the 0.25 m central hole and is unreachable.
   The arm keeps its last configuration. **Beyond reach** tests the outer bound.
4. Enter (1.75,0) m with the default links. The arm is fully extended: the two
   branches merge into one configuration. At the inner boundary, (0.25,0) m,
   they also merge, this time fully folded.
5. Set both links to 1 m and choose Origin. There are infinitely many shoulder
   angles with the elbow folded at 180°. The lab shows a representative using
   the previous shoulder angle and marks the solution family along the periodic
   edges of the configuration-space chart.
6. Switch back to **Joint angles · FK** to play prescribed joint sweeps or edit
   angles. Clicking the configuration-space map also returns to FK mode.

IK selects a configuration directly. Switching branches does not calculate a
safe path between them. This is geometry, not torque control or a dynamics
simulation. The 3D schematic remains available to inspect each solution, but
target picking happens in Plan view. Keyboard users can enter target coordinates
and select branch buttons.

## Derivation

Let r² = x² + y². The law of cosines gives:

```text
c2 = cos(q2) = (r² − L1² − L2²) / (2 L1 L2)
s2 = ±sqrt(1 − c2²)
q2 = atan2(s2, c2)
q1 = atan2(y, x) − atan2(L2 s2, L1 + L2 c2)
```

The two signs give two configurations for an ordinary interior target. In this
lab **elbow-up** means that the elbow lies to the left of the directed line from
base to target (q2 < 0); **elbow-down** means the right side (q2 > 0). These names
do not refer to screen height when the target or camera rotates.

Position is reachable precisely when:

```text
|L1 − L2| ≤ r ≤ L1 + L2
```

An out-of-reach target returns no solutions, rather than clamping it to a
different target. A small tolerance covers floating-point roundoff at a boundary.
At either nonzero boundary there is one configuration modulo full turns.
Equal-length links at the origin are a separate case: q2 = π and q1 is arbitrary.
Two joints specify x and y; arbitrary hand heading is not an additional
independent constraint we can satisfy.

The implementation uses scaled lengths and a factorized triangle-area formula
for sin(q2), retaining precision near the reach limits and for near-origin
targets. Every returned joint pair is passed through the existing FK code;
**FK position error** is the distance between the computed hand and the requested
target. Displayed angles are rounded, while calculations retain solver precision.

## Code and verification

- `math/planar_arm.py`: `PlanarArm.inverse_kinematics`, result dataclasses and
  `ik_payload`; the payload includes each verified FK pose.
- `math/arm_viewer.py`: `/api/ik`, with finite-input and geometry validation.
- `math/arm_viewer.js`: FK/IK modes, target picking, request cancellation,
  branch selection and reach feedback. `arm_scene.js` draws both configurations.
- `tests/test_inverse_kinematics.py`: known branches, fixed-seed FK→IK round
  trips, both link orderings, reach boundaries, equal-link family, near-origin
  precision, invalid inputs and independent MuJoCo site checks.
- `tests/test_ik_api.py`: reachable, unreachable, singular and invalid HTTP requests.
- `tests/js/test_arm_motion.mjs`: plan-coordinate picking, branch selection and
  unreachable-result handling, alongside existing motion tests.

On October 21, 2026 the full Python suite passed **110 tests** and the JavaScript
suite passed **8 tests**. Browser checks covered branch switching with an unchanged
hand position, central-hole feedback and the equal-link folded family.

The next artifact is now implemented: [the Jacobian velocity lab](jacobian.md)
relates joint speeds to hand velocity and visualizes local motion directions.
