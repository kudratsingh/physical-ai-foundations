# mujoco-simulation

Learning project: a one-joint MuJoCo pendulum, stepped from Python, with logged joint state.

## Layout
- `mujoco/one_joint.xml` — minimal MJCF model (one hinge joint, one motor actuator)
- `simulate.py` — loads the model, steps it, logs time/qpos/qvel/ctrl, runs three rollouts
- `logs/` — CSV rollouts (generated)
- `media/` — screenshot, GIF, plots for the README (generated with `--render`)
- `README.md` — setup, what I learned, experiment results

## Commands
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python simulate.py            # headless run, prints + saves logs
.venv/bin/python simulate.py --render   # saves media/*.png and media/*.gif
.venv/bin/mjpython simulate.py --viewer # interactive viewer (macOS needs mjpython)
```

## Conventions
- Always use `.venv/bin/python`; never the system Python.
- Keep the model and script small and heavily commented; this is a teaching repo.
- Don't run `--viewer` in automated sessions; it needs a display.
- Regenerate `media/` and `logs/` via the script rather than editing them by hand.
- No secrets or credentials in this repo. `.venv/` is git-ignored.
