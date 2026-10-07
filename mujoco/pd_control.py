#!/usr/bin/env python3
"""pd_control.py - Week 1 (Friday): close the loop on one_joint.xml with a PD controller.

The idea in one picture
-----------------------
Thursday's simulate.py was OPEN LOOP: we picked a torque in advance and
never looked at where the pendulum actually was. Today the controller
reads the state every step and reacts to it - a FEEDBACK loop:

      observe            decide                     act              observe ...
    q, qvel  ──►  error = q_target - q        ──►  data.ctrl[0] = u ──► mj_step ──►
    (sensors)     u = Kp*error - Kd*qvel            (motor torque)      (physics)

  * Kp (proportional gain, N*m/rad): "spring". The further from the target,
    the harder we push back toward it.
  * Kd (derivative gain, N*m*s/rad): "shock absorber". The faster we move,
    the harder we brake. We use -Kd*qvel (the derivative of the error when
    the target is constant), which avoids a huge kick when the target jumps.

What this script does
---------------------
1. Steps the target from 0 rad (hanging straight down) to q_target = 1.0 rad.
2. Runs several gain settings (see GAINS below) for ~3 s each:
     open_loop   : no feedback, just a constant torque equal to the gravity
                   torque at the target ("I computed the right answer once").
     underdamped : big Kp, no Kd -> fast but rings like a bell.
     low_kp      : small Kp -> gentle, but gravity drags it short of the target.
     tuned       : same big Kp as underdamped, plus enough Kd -> fast, no ringing.
3. At t = 1.5 s every run gets the same "kick": the joint velocity jumps by
   +3 rad/s (like someone bumping the link). Feedback runs pull it back,
   open loop has no way of noticing. Turn it off with --no-disturbance.
4. Logs time, q_target, qpos, qvel, ctrl, applied torque to logs/pd_<name>.csv,
   prints a metrics table and saves ONE plot: media/pd_control.png.

Usage (from the repo root)
--------------------------
    .venv/bin/python mujoco/pd_control.py                     # rollouts + table + plot
    .venv/bin/python mujoco/pd_control.py --no-disturbance    # clean step responses
    .venv/bin/python mujoco/pd_control.py --target 0.5 --duration 4
    .venv/bin/mjpython mujoco/pd_control.py --viewer          # watch the tuned PD live

On macOS the interactive viewer needs ``mjpython`` (see mujoco/README.md).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mujoco

# Reuse Thursday's helpers: load_model() builds MjModel + MjData from
# one_joint.xml, reset() puts the state back to a chosen start angle.
# Adding this folder to sys.path lets the script run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulate import load_model, reset  # noqa: E402

ROOT = Path(__file__).resolve().parent          # the mujoco/ folder
LOG_DIR = ROOT / "logs"
MEDIA_DIR = ROOT / "media"

DEFAULT_TARGET = 1.0       # q_target [rad] (about 57 degrees from hanging down)
DEFAULT_DURATION = 3.0     # simulated seconds per run
DIST_TIME = 1.5            # when the disturbance kick happens [s]
DIST_DQ = 3.0              # size of the kick: qvel += DIST_DQ [rad/s]
SS_TIME = 1.4              # where we read "steady-state" error (just before the kick)


@dataclass
class Gains:
    """One controller setting to try.

    open_loop=True means: ignore Kp/Kd and the measurements entirely and just
    apply a constant torque (the gravity torque at the target).
    """

    name: str
    Kp: float                # proportional gain [N*m/rad]
    Kd: float                # derivative gain   [N*m*s/rad]
    open_loop: bool = False
    note: str = ""

    def label(self) -> str:
        if self.open_loop:
            return f"{self.name} (const torque, no feedback)"
        return f"{self.name} (Kp={self.Kp:g}, Kd={self.Kd:g})"


# The gain settings to compare. Edit this list to experiment.
#
# Rough physics to pick numbers (from the model: I ~ 1.60 kg*m^2 about the
# pivot, gravity torque m*g*l*sin(q) with m*g*l ~ 20.9 N*m):
#   * Near the target the loop behaves like a mass-spring-damper:
#       I*q'' + Kd*q' + (Kp + gravity stiffness)*q = ...
#     so Kp sets how stiff/fast it is and Kd sets how damped it is.
#   * Plain PD has no gravity model. At rest it needs Kp*error = gravity
#     torque, so a residual error of about 20.9*sin(q)/Kp remains. That is why
#     a small Kp "sags" below the target and why tuned uses a large Kp.
#   * The motor saturates at +-50 N*m (ctrlrange in one_joint.xml), so with
#     Kp=500 the first ~0.2 s is "full torque" and the gains only matter
#     once the error gets small.
GAINS = [
    Gains("open_loop", Kp=0.0, Kd=0.0, open_loop=True,
          note="constant gravity torque at the target, never looks at q"),
    Gains("underdamped", Kp=500.0, Kd=0.0,
          note="POOR on purpose: stiff spring, no brake -> rings"),
    Gains("low_kp", Kp=40.0, Kd=10.0,
          note="POOR on purpose: soft spring -> gravity sag, never reaches target"),
    Gains("tuned", Kp=500.0, Kd=40.0,
          note="the fix: keep Kp=500, add Kd=40 -> fast and no overshoot"),
]
TUNED = next(g for g in GAINS if g.name == "tuned")


def gravity_torque(model: mujoco.MjModel, data: mujoco.MjData, q: float) -> float:
    """Torque the motor must supply to HOLD the link still at angle q.

    mj_forward with qvel = 0 fills data.qfrc_bias with the gravity (and
    Coriolis, zero here) generalized force; holding still needs exactly that
    much torque. This is what the open-loop run applies.
    """
    reset(model, data, qpos0=q, ctrl=0.0)
    return float(data.qfrc_bias[0])


def pd_rollout(model: mujoco.MjModel, data: mujoco.MjData, g: Gains, q_target: float,
               duration: float, disturbance: bool, tau_ff: float) -> np.ndarray:
    """Simulate one gain setting. Returns a log with columns
    time, q_target, qpos, qvel, ctrl, tau (tau = torque actually applied after clamping).
    """
    # Start hanging straight down, at rest, with zero command.
    reset(model, data, qpos0=0.0, ctrl=0.0)
    dt = model.opt.timestep
    n_steps = int(round(duration / dt))
    dist_step = int(round(DIST_TIME / dt))        # step index of the kick
    log = np.zeros((n_steps + 1, 6))
    log[0] = (data.time, q_target, data.qpos[0], data.qvel[0], data.ctrl[0], 0.0)

    for i in range(1, n_steps + 1):
        # Disturbance: an instantaneous bump to the joint velocity, the same
        # in every run. The controller is NOT told about it; it can only see
        # its effect through q and qvel on the next step.
        if disturbance and i == dist_step:
            data.qvel[0] += DIST_DQ

        # ---------------- the feedback loop ----------------
        # 1) OBSERVE: read the current state from the simulator.
        #    (On a real robot these would come from an encoder.)
        q = data.qpos[0]
        qvel = data.qvel[0]
        Kp, Kd = g.Kp, g.Kd

        # 2) DECIDE: compute the motor command from the error.
        if g.open_loop:
            # Open loop: a fixed torque chosen in advance. q is never used.
            u = tau_ff
        else:
            error = q_target - q            # how far are we from where we want to be?
            u = Kp*error - Kd*qvel          # spring toward target, brake on velocity

        # 3) ACT: send the command to the motor. MuJoCo clamps it to
        #    ctrlrange (+-50 N*m), like a real motor's torque limit.
        data.ctrl[0] = u

        # 4) Physics advances one timestep (gravity + motor + joint damping)...
        mujoco.mj_step(model, data)
        # ...and on the next iteration we OBSERVE the new q, qvel: the loop closes.
        # ---------------------------------------------------

        # actuator_force = torque the motor really applied (after clamping).
        log[i] = (data.time, q_target, data.qpos[0], data.qvel[0],
                  data.ctrl[0], data.actuator_force[0])

    LOG_DIR.mkdir(exist_ok=True)
    with (LOG_DIR / f"pd_{g.name}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "q_target", "qpos", "qvel", "ctrl", "applied_torque"])
        w.writerows(log.tolist())
    return log


def metrics(log: np.ndarray, q_target: float, disturbance: bool) -> dict:
    """Step-response numbers, all computed on the window BEFORE the kick so the
    disturbance doesn't pollute them; recovery is measured after the kick."""
    t, q, ctrl, tau = log[:, 0], log[:, 2], log[:, 4], log[:, 5]
    err = q_target - q
    tol10, tol5 = 0.10 * abs(q_target), 0.05 * abs(q_target)
    t_end_step = DIST_TIME if disturbance else t[-1] + 1e-9
    pre = t < t_end_step

    # Rise time: first time we get within 10% of the target.
    within = np.where(pre & (np.abs(err) <= tol10))[0]
    rise = t[within[0]] if len(within) else None

    # Overshoot: how far past the target we went, as % of the step size.
    over = max(0.0, (q[pre].max() - q_target) / q_target * 100.0) if q_target > 0 else \
        max(0.0, (q_target - q[pre].min()) / abs(q_target) * 100.0)

    # Settling time: the LAST time |error| was > 5% of the target. If the
    # error is still > 5% at the end of the window, it never settled.
    outside = np.where(pre & (np.abs(err) > tol5))[0]
    if len(outside) == 0:
        settle = 0.0
    elif t[outside[-1]] >= t[pre][-1] - 1e-9:
        settle = None
    else:
        settle = t[outside[-1]]

    # Steady-state error, read just before the kick.
    i_ss = int(np.argmin(np.abs(t - SS_TIME)))
    ss_err = err[i_ss]

    # Disturbance response. We compare against where the link was just
    # before the kick (not the target), so a run that already sat short of
    # the target (low_kp) is judged on "did it come back to where it was?".
    #   kick_dev = biggest deviation the kick caused [rad]
    #   recover  = time after the kick until it stays within 5% of the
    #              target size of its pre-kick angle again
    recover = kick_dev = None
    if disturbance:
        i_kick = int(np.argmin(np.abs(t - DIST_TIME))) - 1     # last logged step before the kick
        dev = np.abs(q - q[i_kick])
        post = t >= DIST_TIME
        kick_dev = dev[post].max()
        out_post = np.where(post & (dev > tol5))[0]
        if len(out_post) == 0:
            recover = 0.0
        elif t[out_post[-1]] < t[-1] - 1e-9:
            recover = t[out_post[-1]] - DIST_TIME

    return dict(rise=rise, overshoot=over, settle=settle, ss_err=ss_err, recover=recover,
                kick_dev=kick_dev, max_ctrl=np.abs(ctrl).max(), max_tau=np.abs(tau).max())


def print_table(results: list[tuple[Gains, dict]], disturbance: bool) -> None:
    fmt = lambda x, p=3: "never" if x is None else f"{x:.{p}f}"   # noqa: E731
    print("\n=== PD metrics (step 0 -> target; rise/overshoot/settle measured before the kick) ===")
    hdr = (f"{'setting':<12} {'Kp':>6} {'Kd':>5} {'rise[s]':>8} {'overshoot[%]':>13} "
           f"{'settle[s]':>10} {'ss_err@1.4s[rad]':>17} {'kick_dev[rad]':>14} {'recover[s]':>11} "
           f"{'max|ctrl| cmd':>14} {'max|torque| applied':>20}")
    print(hdr)
    print("-" * len(hdr))
    for g, m in results:
        kp = "-" if g.open_loop else f"{g.Kp:g}"
        kd = "-" if g.open_loop else f"{g.Kd:g}"
        rec = fmt(m["recover"]) if disturbance else "n/a"
        dev = f"{m['kick_dev']:.3f}" if disturbance else "n/a"
        print(f"{g.name:<12} {kp:>6} {kd:>5} {fmt(m['rise']):>8} {m['overshoot']:>13.1f} "
              f"{fmt(m['settle']):>10} {m['ss_err']:>+17.4f} {dev:>14} {rec:>11} "
              f"{m['max_ctrl']:>14.1f} {m['max_tau']:>20.1f}")
    print("rise   = first time within 10% of target;  settle = last time |error| > 5% of target")
    print("kick_dev = largest |q - q_before_kick| after the kick;  recover = time after the kick until")
    print("           q stays within 5% of target size of its pre-kick angle;  'never' = not within the run")
    print("ctrl is what the controller asked for; applied torque is after the +-50 N*m motor limit")


def plot(runs: list[tuple[Gains, np.ndarray]], q_target: float, disturbance: bool,
         lim: float) -> Path:
    """One figure: angle vs target (top) and applied motor torque (bottom)."""
    import matplotlib
    matplotlib.use("Agg")          # headless: write a PNG, no window
    import matplotlib.pyplot as plt

    colors = {"open_loop": "#7f7f7f", "underdamped": "#d62728",
              "low_kp": "#ff7f0e", "tuned": "#1f77b4"}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 7.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    for g, log in runs:
        c = colors.get(g.name)
        lw = 2.4 if g.name == "tuned" else 1.6
        ls = "--" if g.open_loop else "-"
        ax1.plot(log[:, 0], log[:, 2], color=c, lw=lw, ls=ls, label=g.label())
        ax2.plot(log[:, 0], log[:, 5], color=c, lw=lw, ls=ls, label=g.label())

    ax1.axhline(q_target, color="black", lw=1.2, ls=":", label=f"target q = {q_target:g} rad")
    ax1.axhspan(q_target * 0.95, q_target * 1.05, color="green", alpha=0.08,
                label="±5% settling band")
    ax2.axhline(lim, color="black", lw=0.8, ls=":")
    ax2.axhline(-lim, color="black", lw=0.8, ls=":")
    ax2.set_ylim(-1.25 * lim, 1.25 * lim)
    ax2.text(0.0, lim * 1.05, f"motor limit ±{lim:g} N·m", fontsize=9, va="bottom")
    for ax in (ax1, ax2):
        if disturbance:
            ax.axvline(DIST_TIME, color="purple", lw=1.2, ls="--")
        ax.grid(alpha=0.3)
    if disturbance:
        ax1.text(DIST_TIME + 0.03, 2.15 * q_target,
                 f"kick: qvel += {DIST_DQ:g} rad/s", color="purple", fontsize=9, va="top")

    # Fix the angle axis around the interesting region. The open-loop run
    # swings over the top and keeps spinning (q grows by many rad), which
    # would squash every other curve flat; we clip it and say so on the plot.
    ax1.set_ylim(-0.2, 2.2 * q_target)
    for g, log in runs:
        if g.open_loop and log[:, 2].max() > 2.2 * q_target:
            i_out = int(np.argmax(log[:, 2] > 2.0 * q_target))   # where it exits the chart
            ax1.annotate(f"open loop leaves the chart:\nswings over the top,\n"
                         f"q = {log[-1, 2]:.1f} rad at t = {log[-1, 0]:.1f} s",
                         xy=(log[i_out, 0], log[i_out, 2]),
                         xytext=(log[i_out, 0] + 0.12, 1.75 * q_target),
                         fontsize=9, color="#555555",
                         arrowprops=dict(arrowstyle="->", color="#555555"))
            break
    ax1.set_ylabel("joint angle q [rad]")
    ax1.set_title("One-joint pendulum: PD feedback vs open loop (step 0 → "
                  f"{q_target:g} rad{', kick at t=%.1f s' % DIST_TIME if disturbance else ''})")
    # Legend outside the axes so it never hides a curve.
    ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, frameon=False)
    ax2.set_ylabel("applied torque [N·m]")
    ax2.set_xlabel("time [s]")
    fig.tight_layout()
    MEDIA_DIR.mkdir(exist_ok=True)
    out = MEDIA_DIR / "pd_control.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def run_viewer(model: mujoco.MjModel, data: mujoco.MjData, q_target: float,
               duration: float | None) -> None:
    """Watch the TUNED PD controller live, in (roughly) real time.

    Keys (viewer window focused):
      Space      pause / resume
      D          disturbance: kick the link (qvel += 3 rad/s) and watch PD pull it back
      Backspace  reset to hanging down (0 rad) -> watch the step response again
    You can also double-click the ball and Ctrl + right-drag to push it by hand.
    On macOS this must be launched with ``mjpython``.
    """
    import mujoco.viewer

    state = {"paused": False, "kick": False}

    def on_key(keycode: int) -> None:
        # Runs on the viewer's thread; only set flags / reset here.
        if keycode == 32:                        # Space
            state["paused"] = not state["paused"]
            print(f"[viewer] {'PAUSED' if state['paused'] else 'running'}", flush=True)
        elif keycode in (ord("D"), ord("d")):    # D
            state["kick"] = True
        elif keycode == 259:                     # Backspace
            reset(model, data, qpos0=0.0, ctrl=0.0)
            print("[viewer] reset to q=0 (hanging down)", flush=True)

    Kp, Kd = TUNED.Kp, TUNED.Kd
    reset(model, data, qpos0=0.0, ctrl=0.0)
    print(f"\n[viewer] tuned PD: Kp={Kp:g}, Kd={Kd:g}, target={q_target:g} rad\n"
          "  Space pause/resume | D kick the link | Backspace reset | close window to quit\n"
          "  double-click the ball, then Ctrl + right-drag to push it yourself\n")
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        start = time.time()
        next_print = 0.0
        while viewer.is_running() and (duration is None or time.time() - start < duration):
            step_start = time.time()
            if not state["paused"]:
                if state["kick"]:
                    data.qvel[0] += DIST_DQ
                    state["kick"] = False
                    print(f"[viewer] kick! qvel += {DIST_DQ:g} rad/s", flush=True)
                # observe -> decide -> act -> step (same law as the rollouts)
                q, qvel = data.qpos[0], data.qvel[0]
                error = q_target - q
                u = Kp*error - Kd*qvel
                data.ctrl[0] = u
                mujoco.mj_step(model, data)
                if data.time < next_print - 0.5:   # time jumped back after a reset
                    next_print = 0.0
                if data.time >= next_print:
                    print(f"  t={data.time:6.2f}s  q={data.qpos[0]:+7.3f}  error={error:+7.3f}  "
                          f"torque={data.actuator_force[0]:+6.1f} N*m", flush=True)
                    next_print = data.time + 0.5
            viewer.sync()
            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)
    print("[viewer] window closed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--duration", type=float, default=None,
                        help=f"simulated seconds per run (default {DEFAULT_DURATION}); "
                             "with --viewer: auto-close after this many wall seconds")
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET,
                        help=f"target angle q_target in rad (default {DEFAULT_TARGET})")
    parser.add_argument("--no-disturbance", action="store_true",
                        help=f"skip the qvel += {DIST_DQ} rad/s kick at t={DIST_TIME} s")
    parser.add_argument("--viewer", action="store_true",
                        help="watch the tuned PD live (macOS: use mjpython); skips the rollouts")
    args = parser.parse_args()

    model, data = load_model()
    lim = float(model.actuator_ctrlrange[0, 1])

    if args.viewer:
        try:
            run_viewer(model, data, args.target, args.duration)
        except Exception as e:
            print(f"[viewer] could not launch: {e!r}\n"
                  "  on macOS run: .venv/bin/mjpython mujoco/pd_control.py --viewer")
        return

    duration = args.duration if args.duration is not None else DEFAULT_DURATION
    disturbance = not args.no_disturbance and duration > DIST_TIME
    tau_ff = gravity_torque(model, data, args.target)
    print(f"loaded one_joint.xml: timestep={model.opt.timestep}s, motor limit ±{lim:g} N*m")
    print(f"step 0 -> {args.target:g} rad over {duration:g} s; disturbance "
          f"{'qvel += %g rad/s at t=%g s' % (DIST_DQ, DIST_TIME) if disturbance else 'off'}")
    print(f"gravity torque at target = {tau_ff:.2f} N*m (used by open_loop)\n")

    runs, results = [], []
    for g in GAINS:
        log = pd_rollout(model, data, g, args.target, duration, disturbance, tau_ff)
        runs.append((g, log))
        results.append((g, metrics(log, args.target, disturbance)))
        print(f"ran {g.label():<36} -> logs/pd_{g.name}.csv   [{g.note}]")

    print_table(results, disturbance)
    out = plot(runs, args.target, disturbance, lim)
    print(f"\nsaved {out.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
