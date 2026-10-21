# Week 2: Jacobian and hand velocity

FK maps angles to hand position. Position IK reverses that question. The
Jacobian connects **joint speeds** to the hand's **instantaneous velocity**:

```text
[vx]   [J11 J12] [qdot1]
[vy] = [J21 J22] [qdot2]
```

This lab uses the world xy frame, joint speeds in radians/second and hand
velocity in metres/second. It is a 2×2 position Jacobian, not a 6D spatial twist.

## Open and explore

Launch the existing arm viewer:

```bash
.venv/bin/python math/arm_viewer.py
```

During review, launch from the original repository with:

```bash
.venv/bin/python .worktrees/week2-planar-arm/math/arm_viewer.py
```

Open http://127.0.0.1:8765. Follow **Velocity & singularities** below the arm
scene, or scroll to **Jacobian · from joint speed to hand velocity**.

- Select **Bent** and **Shoulder only**. The blue arrow is the hand velocity
  caused by qdot1 = 1 rad/s. **Elbow only** reveals the second column of J in
  orange. The green arrow is their vector sum for the selected speeds.
- Select **Oppose joints**. This sets qdot = (1,−1), so the velocity is the first
  column minus the second. Opposing rates do not generally cancel hand motion.
- Select **Zero speeds**. Hand velocity becomes zero, while the ellipse stays:
  it describes possible velocities at this configuration, independent of the
  currently selected rates.
- Choose **Near straight**, then **Straight**. At 5°, the ellipse is thin but
  still has area. At 0°, it collapses to a line. **Folded** is also singular.
- Change the shoulder angle. The directions rotate in world coordinates; the
  singular values and condition number stay unchanged for a fixed elbow angle.
- Switch to Hand target · IK. Selecting another target or elbow branch updates
  J for that actual configuration. Changing link lengths updates J as well.

The rates are a preview at the current pose: they do not integrate or drive
the arm. Changing a preview speed pauses prescribed playback so the selected
configuration stays fixed. During playback, the preview recalculates at each
pose; it is not a measurement of the player's joint speeds. The arm arrows use
a 0.20 s drawing scale: 1 m/s is drawn as 0.20 m, without predicting a trajectory.
The separate velocity-space chart uses m/s directly and maintains its scale
while the speed sliders change at a given pose.

## Differentiate the existing FK equations

Starting from:

```text
x = L1 cos(q1) + L2 cos(q1 + q2)
y = L1 sin(q1) + L2 sin(q1 + q2)
```

Take partial derivatives, with angles in radians:

```text
J = [ −L1 sin(q1) − L2 sin(q1+q2)    −L2 sin(q1+q2) ]
    [  L1 cos(q1) + L2 cos(q1+q2)     L2 cos(q1+q2) ]
```

Each column is the hand velocity caused by a unit speed at one joint while
the other joint stays fixed. The shoulder column is tangent to a circle about
the base. The elbow column is tangent to a circle about the elbow.

At q = (0,0), with the default links:

```text
J = [ 0      0    ]
    [ 1.75   0.75 ]
```

Both columns are vertical. No joint-speed combination can produce instantaneous
horizontal hand velocity. For example, qdot = (0.75,−1.75) gives zero hand
velocity even though both joints move. Over a finite interval, J changes and
the hand can move; zero instantaneous velocity does not mean permanent rest.

For small dt, `p(q + qdot*dt) ≈ p(q) + J(q)*qdot*dt`. The approximation error
shrinks quadratically with dt. This is why the velocity arrow is not a path.

## Singularity, ellipse and numerical readouts

The determinant simplifies to `L1 L2 sin(q2)`. It vanishes at q2 = 0 or π:
fully extended or folded. J has rank 1 there, so the available instantaneous
hand velocities occupy a line rather than the whole plane. This holds for
both equal and unequal links. Equal links folded at the base have a zero
shoulder column, but elbow speed still moves the hand.

The manipulability ellipse is the image of the **unit Euclidean joint-rate
circle**: `v = J qdot` with `||qdot|| = 1 rad/s`. Its interior corresponds to
combined joint-speed norm at most 1. It is not the image of the independent
slider bounds (a square), and it does not encode torque or hardware limits.
A selected rate pair with norm greater than 1 can legitimately lie outside it.

The SVD `J = U diag(sigma1,sigma2) Vᵀ` gives:

- Columns of U: principal hand-velocity directions.
- sigma1 and sigma2: ellipse semiaxes for a unit joint-rate norm.
- Condition number sigma1/sigma2: relative stretching; infinite at rank loss.
- `|det(J)| = sigma1*sigma2`: manipulability; ellipse area is π times this value.

The display calls condition > 30 “near singular” as a teaching threshold.
There is no universal boundary at 30. Numerical rank uses
`2 * machine_epsilon * sigma1`, allowing for roundoff such as sin(π).
An infinite condition number is represented by JSON null and displayed as ∞.
No matrix inversion or velocity controller is implemented in this step.

## Implementation and verification

`PlanarArm.jacobian` and `differential_kinematics` in `math/planar_arm.py`
compute J and its SVD in Python. Every FK and IK pose payload includes these
results. Browser helpers multiply that J by the selected rates and sample its
unit-circle image; they do not implement another derivative formula.

`tests/test_jacobian.py` checks the derivative using central differences over
four link geometries, compares it with MuJoCo's independent `mj_jacSite`, checks
the determinant and SVD, distinguishes near-singular from singular, and checks
quadratic convergence of a short-step prediction. HTTP tests check that a
singular payload remains valid JSON. JavaScript tests cover vector addition,
zero rates, unit-circle mapping and collapse to a line.

On October 21, 2026 the full suite passed **132 Python tests** and **11 JavaScript
tests**. Browser checks confirmed a vertical collapsed ellipse at q = (0,0),
v = (0,1.05) m/s for qdot = (0.6,0), and a finite condition number of about
55.35 at q = (0°,5°).

## References and next step

[Modern Robotics §5.3: Singularities](https://modernrobotics.northwestern.edu/nu-gm-book-resource/5-3-singularities/)
explains lost velocity directions when Jacobian rank drops.
[§5.4: Manipulability](https://modernrobotics.northwestern.edu/nu-gm-book-resource/5-4-manipulability/)
describes mapping a joint-rate sphere to an end-effector velocity ellipsoid.
The planar ellipse here is the two-dimensional version of that construction.

The next experiment is implemented in [closed-loop target tracking](resolved_rate_control.md):
use damped least squares to turn a hand-position error into joint rates.
