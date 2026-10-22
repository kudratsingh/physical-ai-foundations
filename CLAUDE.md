# physical-ai-foundations

Stage 0, Weeks 1–2 of a physical-AI learning roadmap. Small, heavily commented,
reproducible artifacts; the owner must be able to explain every file.

## Layout
- `notes/` — written-in-own-words theory notes (C-space, frames, SE(3))
- `math/` — NumPy rotation/transform functions, two-link FK and browser explorer
- `tests/` — pytest suite (transforms, MJCF model, simulate.py, pd_control.py); CI in `.github/workflows/tests.yml`
- `mujoco/` — one-joint MJCF model, `simulate.py` (Thu), `pd_control.py` (Fri), generated `logs/` and `media/`
- `ros2_ws/` — ROS 2 Jazzy packages, Ubuntu 24.04 via Docker Desktop; see README and ARM_README
  for the separate Week-1 exercise and Week-2 MuJoCo/ROS arm systems lab.
- `architecture/` — Week-1 system map and review

## Commands
```bash
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python mujoco/simulate.py            # headless, prints + saves logs
.venv/bin/python mujoco/simulate.py --render   # saves mujoco/media/*
.venv/bin/mjpython mujoco/simulate.py --viewer # interactive, macOS needs mjpython
.venv/bin/python tests/test_transforms.py
.venv/bin/python -m pytest                     # full suite (fast); add -m slow / -m 'not slow' to filter
.venv/bin/python math/arm_viewer.py             # Week 2 browser visualization; Ctrl+C stops it
```

## Conventions
- Use `.venv/bin/python` for Mac/MuJoCo work. ROS uses Ubuntu's `python3` inside
  the Jazzy container after sourcing ROS and the colcon workspace; never install
  ROS into the Mac venv.
- Use feature branches and PRs, worktrees when parallel changes need isolation,
  and appropriate tests. Implement requested work directly; explain it for review.
- Don't run `--viewer` in automated sessions (needs a display). Use `--duration N` if you must.
- Regenerate `mujoco/media/` and `mujoco/logs/` via the script, never by hand.
- No secrets. `.venv/` and CSV logs are git-ignored.
