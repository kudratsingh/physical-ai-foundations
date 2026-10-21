# Week 2: closed-loop hand-position control

The Jacobian lab previewed `v = J qdot`. This experiment closes the loop:
measure the hand position, request a velocity toward a target, solve for joint
rates, integrate them, then measure again. The hand now moves gradually rather
than jumping to an analytic IK solution.

## Try it

Launch `.venv/bin/python math/arm_viewer.py` from the feature checkout. From the
original repository during PR review, use:

```bash
.venv/bin/python .worktrees/week2-planar-arm/math/arm_viewer.py
```

Open http://127.0.0.1:8765 and choose **Track target**. Enter x/y or click a
location in Plan view. Start the loop with **Run controller**, pause it with the
same button, use **One step** for one 0.04 s update, or **Reset start** to restore
the starting joint configuration. Target editing pauses the loop; press Run
again to follow the new target.

Scroll to **Move the hand · closed-loop velocity control** for experiments,
settings, an error chart and velocity readouts:

1. **Reach target** sets a bent start and an interior target. Press Run and
   watch the hand trail and distance-to-target curve. It stops within 0.01 m.
2. **Near singular** starts at q = (0°,5°). Compare Damped least squares with
   Pseudoinverse. Use Reset start between runs so the comparison uses the same
   initial configuration, target and gain. Raw peak rate shows solver demand
   before limiting; actual joint rates always obey the same speed limit.
3. Increase damping. The solver suppresses rates along weak directions more
   strongly, trading velocity accuracy for smaller rates. Position feedback
   keeps correcting, but excessive damping can make convergence slow.
4. **Singular trap** starts with a straight arm and an inward target on the
   x-axis. The target is geometrically reachable, but the local Jacobian only
   produces y velocity. The lab reports Stalled instead of perturbing a joint
   or injecting an analytic IK solution. Choose a bent starting pose to escape.
5. Choose Beyond reach or Origin with unequal links. These targets are rejected
   and the arm holds its current configuration. Reachability does not guarantee
   that a local controller can reach a target from every initial condition.

Experiments scale targets to the selected link lengths. Track-target inputs
accept coordinates between −5 and 5 m, covering the viewer's full workspace.
FK and analytic IK remain separate modes; changing modes pauses control.

## The controller

For position error `e = target − FK(q)`:

```text
v_requested = k e                 # cap norm at 0.8 m/s
qdot = Jᵀ (J Jᵀ + λ² I)⁻¹ v_requested
q_next = q + 0.04 qdot            # Euler integration, radians
```

The implementation uses SVD rather than explicitly computing an inverse:

```text
J = U diag(σ) Vᵀ
qdot = V diag(σ / (σ² + λ²)) Uᵀ v_requested
```

This minimizes `||J qdot − v_requested||² + λ² ||qdot||²` before speed limiting.
The damping parameter is not the pendulum PD controller's Kd: it regularizes
an inverse-kinematics calculation rather than applying a braking torque.

The comparison pseudoinverse uses `1/σ` for numerically nonzero singular values
and zero for a missing direction. Near a singularity, a small σ can produce
large raw joint rates. DLS suppresses this amplification, but cannot create an
instantaneously missing direction.

If any requested joint rate exceeds 1.5 rad/s, both rates are scaled by the
same factor so the largest magnitude becomes 1.5. This preserves their ratio.
The displayed achieved velocity is `J qdot` after this limit. Red arrows show
requested velocity; green arrows show achieved velocity at the **displayed**
pose. The velocity preview is recomputed after each step, so it aligns with
the current arm and Jacobian. It is not a finite-difference measurement of the
last step's displacement. The raw peak-rate readout exposes solver demand
before the limiter; rate limits still apply in both methods.

## State, timing and stopping

`math/resolved_rate.py` contains `controller_command` and `control_payload`.
The localhost `/api/control` endpoint is stateless: it receives the last joint
configuration, target, geometry and settings, and returns the next FK pose plus
feedback. A zero-duration request updates the preview without moving.

Every animated step advances 0.04 simulated seconds. Network and rendering can
make playback slower than real time; the error chart uses simulated time.
Requests are sequential. Pausing, changing settings/target/geometry, leaving the
tab or switching modes cancels pending updates so stale responses cannot move
the arm afterward. Changing a target or settings starts a new error trace at
the current position; Reset start restores the session's initial joint angles.

The loop stops on:

- **Reached:** error ≤ 0.01 m; joint rates become zero.
- **Unreachable:** target outside the position-workspace annulus; no motion.
- **Stalled:** nonzero position error but achieved speed below 1e−7 m/s.
- **Time limit:** 15 simulated seconds without reaching; residual error remains.

The stall threshold can also detect a slow local equilibrium. It is not proof
that a target is globally impossible. Neither fixed damping nor a local
pseudoinverse guarantees convergence from every start. This implementation
assumes unrestricted joints and ignores collisions.

This is **kinematic velocity control**. It does not model inertia, gravity,
contact or motor torque. The [MuJoCo physics experiment](arm_physics.md) now connects these geometric
targets to joint PD and torque-limited motors.

## Verification

Tests check convergence for both solver methods, other link geometries, bounded
rates, the regularized least-squares equation, damping near singularity, an
exact radial trap, rejection of unreachable goals, zero-duration preview,
consistent velocities at the returned pose, and invalid API inputs. The full
Python suite passed **163 tests** and the JavaScript helpers passed **11 tests**
on October 21, 2026. Browser checks confirmed the Reach target preset stopped
at 0.0095 m error after 2.44 simulated seconds; Reset restored (30°,60°), One
step advanced 0.04 s, and pausing prevented a pending update from moving the arm.
The singular-trap preset displayed a nonzero requested inward velocity with
zero achievable velocity and no motion.

## References

[Samuel R. Buss, Introduction to Inverse Kinematics with Jacobian Transpose,
Pseudoinverse and Damped Least Squares Methods](https://people.csail.mit.edu/jbarry/spring2011PR2/readings/jactrans.pdf)
derives the pseudoinverse and regularized least-squares methods with SVD.
[Modern Robotics, Chapter 6](https://modernrobotics.northwestern.edu/chapters/chapter6/)
introduces iterative inverse kinematics using local Jacobian updates.
