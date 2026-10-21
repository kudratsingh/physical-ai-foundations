# Week 2: two-link MuJoCo torque control

The browser's **MuJoCo physics** mode closes the loop through motor torques:

```text
hand target → measured position error → Jacobian/DLS joint rates
→ integrated joint references → joint PD → limited motor torques
→ MuJoCo dynamics → measured qpos/qvel → next observation
```

The earlier Track target mode directly integrates joint rates. Physics mode
instead sets `data.ctrl` and calls `mj_step`; actual angles and speeds are
MuJoCo state. Inertia, gravity, joint damping and motor limits affect tracking.

## Run it

From this feature checkout, run `.venv/bin/python math/arm_viewer.py`. While
PR #6 is being reviewed, the equivalent command from the original repo is:

```bash
.venv/bin/python .worktrees/week2-planar-arm/math/arm_viewer.py
```

Open http://127.0.0.1:8765, choose **MuJoCo physics**, choose a target in Plan
view or enter x/y, and press **Run controller**. **One step** advances 0.04 s
(20 physics steps); **Reset start** restores the initial pose at rest.

Physics continues while holding a target. Pause or the 15 s experiment limit
stops simulated time. Target and settings edits pause the run while preserving
joint position and velocity. Reset, a mode change or link-geometry change starts
a fresh dynamics state; resizing links is a new experiment, not physical motion.
A cancelled browser request has no hidden server-side state to advance.

## Experiments

1. With default settings, choose **Reach target** and Run. The measured hand
   approaches the crosshair. The dashed arm is the integrated joint reference;
   small differences between it and the solid measured arm produce PD torque.
   Inspect position error, joint tracking errors and the torque traces.
2. Once Holding appears, click **Kick elbow · +2 rad/s**. This explicitly adds
   2 rad/s to the elbow velocity once. It is a velocity perturbation, not a
   specified force or torque impulse. The error rises and PD brakes the arm;
   the controller should regain the target. The green chart marker locates
   the kick. Strong damping can remove most of the extra speed within one
   displayed frame, while the error/torque traces retain its effect.
3. Reduce **Motor torque limit** to 1 N·m, Reset and Run with gravity enabled.
   Requested torque exceeds the limit, applied torque clips, and the arm cannot
   hold the same target. Red plot markers include saturation in any 2 ms
   substep, even when the last sample of that frame is not saturated.
4. Turn gravity off and repeat. Less holding torque is needed. Restore gravity
   and toggle bias compensation off to compare the joint-reference lag needed
   to generate gravity-supporting PD torque. The outer loop can still correct
   hand position; a persistent joint-reference error is not the same as a
   persistent hand-target error.
5. Change Kp/Kd. Higher Kp increases position correction; Kd tracks reference
   velocity and brakes velocity error. Very low damping can oscillate. Large
   gains do not create extra actuator authority when torque is saturated.

## Model and controller

`mujoco/two_link_control.xml` is separate from the Week-2 FK reference model.
It has two z-axis hinges, two direct torque motors (`gear=1`), link masses
1 kg and 0.75 kg, capsule geometry with automatically calculated inertia,
joint damping 0.1, and no collision contacts. Changing lengths recalculates
inertia while keeping those masses fixed.

Gravity is deliberately `(0,−9.81,0)` in the arm's xy plane, so it loads the
z-axis hinges. The original FK reference model used gravity along −z, which
would not load an ideal arm restricted to the xy plane. The browser's 3D
schematic still displays the same xy coordinates; its visual raised joints
are schematic decoration, not additional physical degrees of freedom.

The outer loop runs every 0.02 s. It reads measured joint angles, uses the
existing damped-least-squares controller to obtain reference joint rates,
and integrates these into joint references. Desired rate magnitude remains
limited to 1.5 rad/s per joint. Actual speed is not clamped and may exceed
that bound after a perturbation.

The joint controller runs every 0.002 s:

```text
tau_requested = Kp*(q_ref − q) + Kd*(qdot_ref − qvel) + optional qfrc_bias
```

Default Kp is 80 N·m/rad and Kd is 12 N·m·s/rad, for both joints. Bias
compensation uses MuJoCo's `qfrc_bias`: gravity plus Coriolis/centrifugal effects,
not joint damping. This is a model-based feedforward term alongside PD.
MuJoCo clamps the motor controls to the selected ±torque limit, initially
20 N·m. The dashboard reads the applied generalized motor torque from
`qfrc_actuator`; with these direct independent motors it equals joint torque.

Joint references are kept within 0.35 rad of measured angles to prevent their
integrator from winding up indefinitely during saturation. This reference
leash is displayed separately from torque saturation. It limits stored
position error; it does not guarantee reaching a target.

Holding requires position error ≤0.01 m and joint-speed norm <0.03 rad/s for
0.3 s. Physics then continues so disturbances can be observed. A local
Jacobian stall does not stop physics: inertia and gravity still act. Unreachable
targets are rejected; the animation pauses. Singularities, inadequate torque,
poor gains or the 15 s experiment limit can leave residual error.

## Measurements and state

`math/arm_dynamics.py` compiles a bounded cache of read-only models by link
lengths, gravity and torque limit. Each request gets its own `MjData`.
`POST /api/physics` carries unwrapped qpos/qvel, integrated joint references,
time and holding duration. Paused previews use dt=0 and preserve measured
state. Display angles are wrapped; control state remains continuous.

The error chart uses simulated experiment time. Each browser frame runs 20
physics steps; playback can be slower than real time if networking/rendering
is slow. Torque/error lines sample at frame boundaries; saturation flags and
peak-applied torque cover every physics substep in that frame. The live torque
snapshot and velocity arrows are evaluated at the returned measured state.

## Verification

The tests exercise actual MuJoCo stepping and motor clamping. They cover the
model's masses/axes/timestep, finite-inertia reference lag, reaching and holding,
kick displacement and recovery, insufficient-torque failure, the reference
leash, gravity compensation, adjustable geometry, state isolation, zero-time
previews, independent site/force comparisons and bad API requests.

On October 21, 2026 the full Python suite passed **199 tests** and the browser
helper suite passed **11 tests**. The default headless experiment held within
0.01 m; a +2 rad/s elbow perturbation recovered; a 1 N·m limit produced visible
tracking failure while all applied torques remained inside the bound.

## References

[MuJoCo computation](https://mujoco.readthedocs.io/en/stable/computation/index.html)
describes the dynamics and bias-force terms.
[MuJoCo actuator XML](https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-general)
documents motor control limits and transmissions.

Browser validation confirmed a 0.0093 m final error with a shoulder holding
torque near 11.4 N·m. The elbow kick recovered to a 0.0057 m error. Torque and
joint-error plots retained the kick marker and transient before returning to
steady values.

With the browser torque limit reduced to 1 N·m, the same experiment ended with
2.2842 m hand error. A snapshot requested (32.506,−55.842) N·m while MuJoCo
applied (1,−1) N·m; the saturation and reference-leash indicators were active.
