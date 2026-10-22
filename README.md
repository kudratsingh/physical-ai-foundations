# physical-ai-foundations

[![tests](https://github.com/kudratsingh/physical-ai-foundations/actions/workflows/tests.yml/badge.svg)](https://github.com/kudratsingh/physical-ai-foundations/actions/workflows/tests.yml)

Stage 0 of my Physical AI roadmap: build the physical-world mental model that
robot learning, VLAs and hardware all assume. Week 1 covers configuration
space, coordinate frames and SE(3), a closed-loop MuJoCo simulation, and ROS 2
basics. Everything here is meant to be small, reproducible and explainable
without notes.

## Week 2 in progress

The first artifact is a two-link planar arm: tested forward and inverse kinematics and an
interactive browser lab. Orbit its 3D schematic or use Plan view, play joint
sweeps, choose configurations on the joint-angle map, and compare the hand trail
with its position history. Adjustable link lengths, frame/angle overlays and
the live transform chain connect the geometry to the Week-1 concepts.

```bash
.venv/bin/python math/arm_viewer.py       # opens a browser; Ctrl+C stops the server
.venv/bin/python math/planar_arm.py       # worked 30-degree / 60-degree example
```

See [the arm walkthrough](notes/planar_arm.md) for the derivation, presets and
validation. Choose **Hand target · IK** to click a target, compare both elbow
solutions and explore reach limits. See [the IK walkthrough](notes/inverse_kinematics.md).
The **Jacobian velocity lab** maps joint speeds to hand velocity, draws the
manipulability ellipse, and shows rank loss at straight/folded configurations.
See [the Jacobian walkthrough](notes/jacobian.md). **Track target** closes the
loop with a damped-least-squares velocity controller, error chart and singularity
experiments. See [the control walkthrough](notes/resolved_rate_control.md).
**MuJoCo physics** extends this loop through joint PD and torque-limited motors,
with gravity, kick recovery and torque/tracking plots. See [the physics walkthrough](notes/arm_physics.md).
Use the Mac venv for this viewer.

The next experiment connects real ROS processes to this arm. The **ROS arm systems lab**
shows measured motion, target commands, latency, stale-command rejection and isolated
bag replay at [localhost:8766](http://localhost:8766). Start it with Docker using
[the ROS arm walkthrough](ros2_ws/ARM_README.md).

## Week 3: first robot-learning experiment

[ManiSkill PushCube + PPO](learning/maniskill/README.md) uses an existing
manipulation benchmark, a separate Python 3.11 environment, and CPU physics.
It records training configuration, validation curves, fixed-seed evaluation
against random/zero-action baselines, native simulator videos and failure cases.
See the walkthrough for the exact commands and the first measured results.

## Week 1 tracks

| day | track | artifact | status |
|---|---|---|---|
| Mon | configuration space | `notes/configuration_space.md` | done |
| Tue–Wed | frames, SO(3), SE(3) | `math/transforms.py`, `tests/test_transforms.py`, `notes/frames_se3.md` | done |
| Thu | first MuJoCo physics loop | `mujoco/one_joint.xml`, `mujoco/simulate.py`, [`mujoco/README.md`](mujoco/README.md) | done |
| Fri | PD controller | `mujoco/pd_control.py`, [`mujoco/PD_NOTES.md`](mujoco/PD_NOTES.md) + plot | done |
| Sat | ROS 2 graph | [`ros2_ws/README.md`](ros2_ws/README.md), `ros2_ws/src/stage0_basics/` | built and verified; hands-on review next |
| Sun | system map + oral gate | `architecture/week1_system_map.md` | map written; oral gate pending |

## Setup (MuJoCo track, macOS or Linux)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python mujoco/simulate.py              # 3 logged rollouts
.venv/bin/python mujoco/simulate.py --render     # screenshot, GIF, plot
.venv/bin/python mujoco/pd_control.py           # PD gain comparison + plot
.venv/bin/mjpython mujoco/simulate.py --viewer   # interactive (macOS needs mjpython)
.venv/bin/python tests/test_transforms.py        # transforms tests, no pytest needed
```

ROS 2 runs in Ubuntu 24.04 (Jazzy) through Docker Desktop on the Mac, separately
from this venv. See [the ROS workspace](ros2_ws/README.md) for setup, a verified
publisher/subscriber + service + rosbag exercise, and a hands-on walkthrough.

## Tests

```bash
.venv/bin/pip install -r requirements-dev.txt    # pytest
.venv/bin/python -m pytest                       # headless math, MuJoCo and viewer tests
node --test tests/js/*.mjs                       # motion helpers (Node 22+)
.venv/bin/python -m pytest -m "not slow"         # skip anything marked slow
```

What `tests/` covers (no GUI or OpenGL; CSV logs go to a temp dir):

- `test_transforms.py` - rotations and SE(3) transforms: orthonormality,
  inverses, composition, the worked world/base/camera example.
- `test_planar_arm.py` - 2R forward kinematics, workspace bounds, transform chain,
  and independent agreement with the MuJoCo two-link model.
- `test_arm_dynamics.py` - real MuJoCo stepping, holding and kick recovery,
  applied torque limits, gravity compensation, geometry and state isolation.
- `test_resolved_rate.py` - closed-loop convergence, damping vs pseudoinverse,
  rate limits, singular stalls, unreachable targets and preview consistency.
- `test_jacobian.py` - analytic derivatives vs finite differences and MuJoCo,
  determinant, singular values, rank loss and local velocity prediction.
- `test_inverse_kinematics.py` and `test_ik_api.py` - both elbow solutions, reach
  boundaries, singular families, FK round trips, MuJoCo comparisons and IK HTTP validation.
- `test_arm_viewer.py` - the browser viewer's localhost HTTP interface and
  adjustable geometry and input validation; no graphical browser is needed in CI.
- `tests/js/test_arm_motion.mjs` - prescribed motion boundaries, fixed-joint
  sweeps, angle wrapping, configuration-space trace seams, target picking and IK branch selection.
- `test_model.py` - `one_joint.xml`: 1 DoF hinge, 2 ms timestep, motor with
  gear 1 and ctrlrange ±50 N·m, joint damping 0.05, gravity on.
- `test_simulate.py` - `reset()` initial state, a free swing stays bounded
  and loses energy, rest at the bottom stays at rest, the three rollouts end
  at different angles.
- `test_pd_control.py` - tuned gains (Kp=500, Kd=40) settle within 5% before
  0.5 s and recover from the kick within 0.5 s; Kd=0 overshoots >30%; low Kp
  sags >0.2 rad; applied torque is clamped to ±50 N·m.

GitHub Actions (`.github/workflows/tests.yml`) runs the suite plus headless
smoke runs of `simulate.py` and `pd_control.py` on Python 3.12 for every push
and pull request to `main`.

The separate `ROS 2` workflow builds the Ubuntu/Jazzy container and runs the ROS
package tests, including real node communication, service calls, and rosbag
recording/replay. Run those tests inside the container as described in
[`ros2_ws/README.md`](ros2_ws/README.md).

## Week-1 gate

Move to Week 2 only when, without notes, I can:

- compose and invert transforms and explain a frame change
- trace qpos/qvel -> controller -> ctrl -> physics, and explain good vs bad PD gains
- inspect a ROS 2 graph and explain topic vs service vs action
- point to where perception, policy, controller, ROS and physics each live

## Not this week

No RL, Isaac Lab, VLAs, MoveIt or hardware.
