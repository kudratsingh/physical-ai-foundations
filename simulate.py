#!/usr/bin/env python3
"""simulate.py - step a one-joint MuJoCo model and compare three rollouts.

What this script shows
----------------------
1. Load an MJCF model (``mujoco/one_joint.xml``) into an ``MjModel`` (the
   static description: geometry, masses, timestep, actuators) and create an
   ``MjData`` (the changing state: time, qpos, qvel, ctrl, ...).
2. Advance physics with ``mujoco.mj_step`` for ~2 s, logging time, qpos,
   qvel and ctrl each step, printing a short table and saving a CSV to
   ``logs/rollout_<name>.csv``.
3. Run three rollouts and compare them:
     a) baseline       : qpos0 = 0.5 rad, ctrl = 0
     b) init_1.5rad    : qpos0 = 1.5 rad, ctrl = 0   (different initial condition)
     c) const_torque   : qpos0 = 0.5 rad, ctrl = 4.0 (same IC, constant motor torque)
   and confirm that the final qpos differs between them.

Usage
-----
    .venv/bin/python simulate.py              # headless: 3 rollouts + CSVs + plot
    .venv/bin/python simulate.py --render     # also save media/one_joint.png + .gif
    .venv/bin/mjpython simulate.py --viewer   # interactive viewer (macOS needs mjpython)

On macOS the passive viewer must be started with ``mjpython`` (shipped with the
mujoco pip package, at ``.venv/bin/mjpython``); plain ``python`` will raise an
error. On Linux/Windows plain ``python`` works.

The GIF is written with Pillow (installed as a matplotlib dependency).
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "mujoco" / "one_joint.xml"
LOG_DIR = ROOT / "logs"
MEDIA_DIR = ROOT / "media"

SIM_SECONDS = 2.0


@dataclass
class Rollout:
    """Configuration for one rollout plus (after running) its logged data."""

    name: str
    qpos0: float          # initial joint angle [rad]
    ctrl: float           # constant motor command applied every step
    log: np.ndarray | None = None  # columns: time, qpos, qvel, ctrl


ROLLOUTS = [
    Rollout("baseline", qpos0=0.5, ctrl=0.0),
    Rollout("init_1.5rad", qpos0=1.5, ctrl=0.0),
    Rollout("const_torque", qpos0=0.5, ctrl=4.0),
]


def load_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load the MJCF file and allocate a matching MjData.

    MjModel is read-only "what the robot is" (incl. ``model.opt.timestep``,
    the duration of one mj_step). MjData holds "what the robot is doing now".
    """
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    return model, data


def reset(model: mujoco.MjModel, data: mujoco.MjData, qpos0: float, ctrl: float) -> None:
    """Reset to the model defaults, then set the initial condition.

    mj_resetData zeroes time, velocities, controls, etc. so rollouts don't
    leak state into each other.
    """
    mujoco.mj_resetData(model, data)
    # qpos = generalized positions. For a single hinge it is just the angle
    # in radians (0 = hanging straight down, positive = rotation about +y).
    data.qpos[0] = qpos0
    # qvel = generalized velocities; for the hinge, angular velocity [rad/s].
    data.qvel[0] = 0.0
    # ctrl = actuator inputs. Our <motor> applies torque = gear * ctrl to the
    # hinge (gear=1, clamped to ctrlrange [-5, 5]).
    data.ctrl[0] = ctrl
    # mj_forward computes derived quantities (positions of bodies, etc.)
    # without advancing time - handy before rendering the very first frame.
    mujoco.mj_forward(model, data)


def run_rollout(model: mujoco.MjModel, data: mujoco.MjData, ro: Rollout,
                print_every: int = 100, verbose: bool = True) -> np.ndarray:
    """Simulate one rollout for SIM_SECONDS, log every step, save a CSV."""
    reset(model, data, ro.qpos0, ro.ctrl)
    # Number of steps = duration / timestep (timestep comes from <option>).
    n_steps = int(round(SIM_SECONDS / model.opt.timestep))
    log = np.zeros((n_steps + 1, 4))
    log[0] = (data.time, data.qpos[0], data.qvel[0], data.ctrl[0])

    if verbose:
        print(f"\n--- rollout '{ro.name}': qpos0={ro.qpos0:+.2f} rad, ctrl={ro.ctrl:+.2f} "
              f"({n_steps} steps of dt={model.opt.timestep}s) ---")
        print(f"{'step':>6} {'time[s]':>8} {'qpos[rad]':>10} {'qvel[rad/s]':>12} {'ctrl':>6}")
        print(f"{0:>6} {log[0,0]:>8.3f} {log[0,1]:>10.4f} {log[0,2]:>12.4f} {log[0,3]:>6.2f}")

    for i in range(1, n_steps + 1):
        # Set the control BEFORE stepping: mj_step reads data.ctrl, computes
        # actuator torque, gravity, damping, then integrates qpos/qvel forward
        # by one timestep and increments data.time.
        data.ctrl[0] = ro.ctrl
        mujoco.mj_step(model, data)
        log[i] = (data.time, data.qpos[0], data.qvel[0], data.ctrl[0])
        if verbose and i % print_every == 0:
            print(f"{i:>6} {log[i,0]:>8.3f} {log[i,1]:>10.4f} {log[i,2]:>12.4f} {log[i,3]:>6.2f}")

    LOG_DIR.mkdir(exist_ok=True)
    csv_path = LOG_DIR / f"rollout_{ro.name}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "qpos", "qvel", "ctrl"])
        w.writerows(log.tolist())
    if verbose:
        print(f"saved {csv_path.relative_to(ROOT)}")
    ro.log = log
    return log


def summarize(rollouts: list[Rollout]) -> None:
    """Print a side-by-side summary and check that the final qpos differ."""
    print("\n=== Summary (final state after {:.1f} s) ===".format(SIM_SECONDS))
    print(f"{'rollout':<14} {'qpos0':>7} {'ctrl':>6} {'final qpos':>11} {'final qvel':>11} "
          f"{'max|qpos|':>10} {'mean qpos':>10}")
    for ro in rollouts:
        t, q, v, _ = ro.log[-1]
        print(f"{ro.name:<14} {ro.qpos0:>7.2f} {ro.ctrl:>6.2f} {q:>11.4f} {v:>11.4f} "
              f"{np.abs(ro.log[:,1]).max():>10.4f} {ro.log[:,1].mean():>10.4f}")

    # Side-by-side qpos at a few time points so the divergence is visible.
    print("\nqpos side by side:")
    print(f"{'time[s]':>8} " + " ".join(f"{ro.name:>14}" for ro in rollouts))
    n = len(rollouts[0].log)
    for idx in np.linspace(0, n - 1, 9).astype(int):
        print(f"{rollouts[0].log[idx,0]:>8.2f} "
              + " ".join(f"{ro.log[idx,1]:>14.4f}" for ro in rollouts))

    base = rollouts[0]
    print()
    for ro in rollouts[1:]:
        diff = abs(ro.log[-1, 1] - base.log[-1, 1])
        max_diff = np.abs(ro.log[:, 1] - base.log[:, 1]).max()
        assert diff > 1e-3, f"rollout '{ro.name}' did not differ from baseline!"
        print(f"OK: rollout '{ro.name}' differs from baseline: |final qpos diff| = {diff:.4f} rad "
              f"(max diff over trajectory = {max_diff:.4f} rad)")


def plot(rollouts: list[Rollout]) -> None:
    """Save qpos and qvel vs time for all rollouts to media/qpos_qvel.png."""
    import matplotlib
    matplotlib.use("Agg")  # headless backend: write a file, no window
    import matplotlib.pyplot as plt

    MEDIA_DIR.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for ro in rollouts:
        label = f"{ro.name} (q0={ro.qpos0}, ctrl={ro.ctrl})"
        ax1.plot(ro.log[:, 0], ro.log[:, 1], label=label)
        ax2.plot(ro.log[:, 0], ro.log[:, 2], label=label)
    ax1.set_ylabel("qpos [rad]")
    ax2.set_ylabel("qvel [rad/s]")
    ax2.set_xlabel("time [s]")
    ax1.set_title("One-joint pendulum: three rollouts")
    ax1.legend(loc="lower right", fontsize=8)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = MEDIA_DIR / "qpos_qvel.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out.relative_to(ROOT)}")


def make_camera() -> mujoco.MjvCamera:
    """A free camera looking at the pendulum from the side (along -y)."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0.0, 0.0, 1.0)
    cam.distance = 3.2
    cam.azimuth = 90.0     # side view: the x-z swing plane faces the camera
    cam.elevation = -10.0
    return cam


def render(model: mujoco.MjModel, data: mujoco.MjData, fps: int = 30) -> None:
    """Offscreen render: a PNG mid-swing and a ~2 s GIF of the baseline rollout."""
    from PIL import Image

    MEDIA_DIR.mkdir(exist_ok=True)
    cam = make_camera()
    width, height = 640, 480
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        # Use a bigger swing for the video so it's visually interesting.
        reset(model, data, qpos0=1.2, ctrl=0.0)
        n_steps = int(round(SIM_SECONDS / model.opt.timestep))
        # Render one frame every (1/fps) seconds of sim time.
        steps_per_frame = max(1, int(round(1.0 / (fps * model.opt.timestep))))
        frames = []
        for i in range(n_steps + 1):
            if i % steps_per_frame == 0:
                renderer.update_scene(data, camera=cam)
                frames.append(renderer.render().copy())
            data.ctrl[0] = 0.0
            mujoco.mj_step(model, data)

        # Mid-swing frame: ~0.25 s in, the link is visibly tilted and moving
        # (at ~0.45 s it hangs straight down and hides in front of the post).
        frame_dt = steps_per_frame * model.opt.timestep
        mid = Image.fromarray(frames[min(len(frames) - 1, int(round(0.25 / frame_dt)))])
        png = MEDIA_DIR / "one_joint.png"
        mid.save(png)
        print(f"saved {png.relative_to(ROOT)}")

        gif = MEDIA_DIR / "one_joint.gif"
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(gif, save_all=True, append_images=imgs[1:],
                     duration=int(round(1000 * frame_dt)), loop=0)
        print(f"saved {gif.relative_to(ROOT)} ({len(frames)} frames @ {fps} fps)")
    finally:
        renderer.close()


def run_viewer(model: mujoco.MjModel, data: mujoco.MjData, duration: float = 30.0) -> None:
    """Interactive passive viewer running in (approximately) real time.

    Applies a small sinusoidal torque so you can see the actuator at work.
    On macOS this requires launching with ``mjpython``.
    """
    import mujoco.viewer

    reset(model, data, qpos0=0.5, ctrl=0.0)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < duration:
            step_start = time.time()
            # Small control input: 1 N*m sinusoidal torque at 0.5 Hz.
            data.ctrl[0] = 1.0 * np.sin(2 * np.pi * 0.5 * data.time)
            mujoco.mj_step(model, data)
            viewer.sync()
            # Sleep so one timestep of sim time ~= one timestep of wall time.
            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--viewer", action="store_true",
                        help="launch the interactive viewer (macOS: use mjpython)")
    parser.add_argument("--render", action="store_true",
                        help="save media/one_joint.png and media/one_joint.gif offscreen")
    parser.add_argument("--print-every", type=int, default=100,
                        help="print a table row every N steps (default 100 = 0.2 s)")
    args = parser.parse_args()

    model, data = load_model()
    print(f"loaded {MODEL_PATH.relative_to(ROOT)}: nq={model.nq} nv={model.nv} "
          f"nu={model.nu} timestep={model.opt.timestep}s")

    for ro in ROLLOUTS:
        run_rollout(model, data, ro, print_every=args.print_every)
    summarize(ROLLOUTS)
    plot(ROLLOUTS)

    if args.render:
        try:
            render(model, data)
        except Exception as e:  # e.g. no OpenGL context available
            print(f"[render] failed: {e!r}\n"
                  "  try: MUJOCO_GL=egl or MUJOCO_GL=osmesa (Linux), default cgl on macOS")

    if args.viewer:
        try:
            run_viewer(model, data)
        except Exception as e:
            print(f"[viewer] could not launch: {e!r}\n"
                  "  on macOS run: .venv/bin/mjpython simulate.py --viewer")


if __name__ == "__main__":
    main()
