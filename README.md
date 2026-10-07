# physical-ai-foundations

Stage 0 of my Physical AI roadmap: build the physical-world mental model that
robot learning, VLAs and hardware all assume. Week 1 covers configuration
space, coordinate frames and SE(3), a closed-loop MuJoCo simulation, and ROS 2
basics. Everything here is meant to be small, reproducible and explainable
without notes.

## Week 1 tracks

| day | track | artifact | status |
|---|---|---|---|
| Mon | configuration space | `notes/configuration_space.md` | todo |
| Tue–Wed | frames, SO(3), SE(3) | `math/transforms.py`, `math/test_transforms.py`, `notes/frames_se3.md` | todo |
| Thu | first MuJoCo physics loop | `mujoco/one_joint.xml`, `mujoco/simulate.py`, [`mujoco/README.md`](mujoco/README.md) | done |
| Fri | PD controller | `mujoco/pd_control.py` + plot | todo |
| Sat | ROS 2 graph | `ros2_ws/src/stage0_basics/` | todo |
| Sun | system map + oral gate | `architecture/week1_system_map.md` | todo |

## Setup (MuJoCo track, macOS or Linux)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python mujoco/simulate.py              # 3 logged rollouts
.venv/bin/python mujoco/simulate.py --render     # screenshot, GIF, plot
.venv/bin/mjpython mujoco/simulate.py --viewer   # interactive (macOS needs mjpython)
.venv/bin/python math/test_transforms.py         # once transforms.py is implemented
```

ROS 2 work runs in Ubuntu 24.04 (Jazzy), not in this venv.

## Week-1 gate

Move to Week 2 only when, without notes, I can:

- compose and invert transforms and explain a frame change
- trace qpos/qvel -> controller -> ctrl -> physics, and explain good vs bad PD gains
- inspect a ROS 2 graph and explain topic vs service vs action
- point to where perception, policy, controller, ROS and physics each live

## Not this week

No RL, Isaac Lab, VLAs, MoveIt or hardware.
