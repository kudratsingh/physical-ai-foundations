# physical-ai-foundations

[![tests](https://github.com/kudratsingh/physical-ai-foundations/actions/workflows/tests.yml/badge.svg)](https://github.com/kudratsingh/physical-ai-foundations/actions/workflows/tests.yml)

Stage 0 of my Physical AI roadmap: build the physical-world mental model that
robot learning, VLAs and hardware all assume. Week 1 covers configuration
space, coordinate frames and SE(3), a closed-loop MuJoCo simulation, and ROS 2
basics. Everything here is meant to be small, reproducible and explainable
without notes.

## Week 1 tracks

| day | track | artifact | status |
|---|---|---|---|
| Mon | configuration space | `notes/configuration_space.md` | done |
| Tue–Wed | frames, SO(3), SE(3) | `math/transforms.py`, `tests/test_transforms.py`, `notes/frames_se3.md` | done |
| Thu | first MuJoCo physics loop | `mujoco/one_joint.xml`, `mujoco/simulate.py`, [`mujoco/README.md`](mujoco/README.md) | done |
| Fri | PD controller | `mujoco/pd_control.py`, [`mujoco/PD_NOTES.md`](mujoco/PD_NOTES.md) + plot | done |
| Sat | ROS 2 graph | `ros2_ws/src/stage0_basics/` | todo |
| Sun | system map + oral gate | `architecture/week1_system_map.md` | done |

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

ROS 2 work runs in Ubuntu 24.04 (Jazzy), not in this venv.

## Tests

```bash
.venv/bin/pip install -r requirements-dev.txt    # pytest
.venv/bin/python -m pytest                       # whole suite, well under a second
.venv/bin/python -m pytest -m "not slow"         # skip anything marked slow
```

What `tests/` covers (no rendering, no viewer; CSV logs go to a temp dir):

- `test_transforms.py` - rotations and SE(3) transforms: orthonormality,
  inverses, composition, the worked world/base/camera example.
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

## Week-1 gate

Move to Week 2 only when, without notes, I can:

- compose and invert transforms and explain a frame change
- trace qpos/qvel -> controller -> ctrl -> physics, and explain good vs bad PD gains
- inspect a ROS 2 graph and explain topic vs service vs action
- point to where perception, policy, controller, ROS and physics each live

## Not this week

No RL, Isaac Lab, VLAs, MoveIt or hardware.
