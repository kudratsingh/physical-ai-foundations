# physical-ai-foundations

Stage 0, Week 1 of a physical-AI learning roadmap. Small, heavily commented,
reproducible artifacts; the owner must be able to explain every file.

## Layout
- `notes/` — written-in-own-words theory notes (C-space, frames, SE(3))
- `math/` — NumPy rotation/transform functions + tests (no library wrappers)
- `mujoco/` — one-joint MJCF model, `simulate.py` (Thu), `pd_control.py` (Fri), generated `logs/` and `media/`
- `ros2_ws/` — ROS 2 Jazzy package, Ubuntu only
- `architecture/` — Week-1 system map and review

## Commands
```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python mujoco/simulate.py            # headless, prints + saves logs
.venv/bin/python mujoco/simulate.py --render   # saves mujoco/media/*
.venv/bin/mjpython mujoco/simulate.py --viewer # interactive, macOS needs mjpython
.venv/bin/python math/test_transforms.py
```

## Conventions
- Always `.venv/bin/python`, never system Python.
- This is a learning repo: don't implement the TODO stubs for the owner; scaffold, explain, review.
- Don't run `--viewer` in automated sessions (needs a display). Use `--duration N` if you must.
- Regenerate `mujoco/media/` and `mujoco/logs/` via the script, never by hand.
- No secrets. `.venv/` and CSV logs are git-ignored.
