"""Resolved-rate targets -> joint PD -> MuJoCo motor torques -> measured state.

Models are cached read-only. Every request builds a separate MjData from an
explicit state, making paused/cancelled browser requests safe to discard.
"""
from functools import lru_cache
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from planar_arm import PlanarArm, pose_payload
from resolved_rate import controller_command

MODEL_PATH = Path(__file__).resolve().parents[1] / "mujoco/two_link_control.xml"
TIMESTEP = 0.002


@lru_cache(maxsize=16)
def dynamics_model(l1=1.0, l2=0.75, torque_limit=20.0, gravity=True):
    """Cached model; callers must not mutate it. Masses stay 1 and 0.75 kg."""
    root = ET.parse(MODEL_PATH).getroot()
    root.find(".//geom[@name='link1']").set("fromto", f"0 0 0 {l1} 0 0")
    root.find(".//geom[@name='link2']").set("fromto", f"0 0 0 {l2} 0 0")
    root.find(".//body[@name='elbow']").set("pos", f"{l1} 0 0")
    root.find(".//site[@name='hand']").set("pos", f"{l2} 0 0")
    root.find("option").set("gravity", "0 -9.81 0" if gravity else "0 0 0")
    for motor in root.findall("actuator/motor"):
        motor.set("ctrlrange", f"{-torque_limit} {torque_limit}")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def _pair(state, name, default):
    value = np.asarray(state.get(name, default), dtype=float)
    if value.shape != (2,) or not np.isfinite(value).all() or np.max(np.abs(value)) > 10000:
        raise ValueError(f"{name} must be a finite pair with magnitude <= 10000")
    return value.copy()


def physics_payload(state, target, *, l1=1.0, l2=0.75, kp=80.0, kd=12.0,
                    torque_limit=20.0, gravity=True, compensate=True, dt=0.04,
                    kick=0.0, gain=2.0, damping=0.12, method="dls"):
    """Advance 0 or 20 physics steps; preview torques at the returned pose.

    qpos/qvel and joint_target are radians/rad/s, without display wrapping.
    The outer loop updates at 50 Hz; joint PD runs at 500 Hz. A 0.35-rad
    reference leash prevents integrator windup when torque is insufficient.
    """
    limits = {"l1": (l1, .25, 1.5), "l2": (l2, .25, 1.5),
              "kp": (kp, 1, 300), "kd": (kd, 0, 40),
              "torque_limit": (torque_limit, 1, 30), "kick": (kick, -3, 3),
              "gain": (gain, .1, 5), "damping": (damping, .01, 1)}
    for name, (value, lo, hi) in limits.items():
        if not isinstance(value, (float, int)) or not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f"{name} must be between {lo} and {hi}")
    if type(gravity) is not bool or type(compensate) is not bool:
        raise ValueError("gravity and compensate must be booleans")
    if dt not in (0, 0.04):
        raise ValueError("Physics dt must be 0 or 0.04 seconds")
    if not isinstance(state, dict):
        raise ValueError("state must be an object")
    target = np.asarray(target, dtype=float)
    if target.shape != (2,) or not np.isfinite(target).all() or np.max(np.abs(target)) > 5:
        raise ValueError("target must be a finite xy pair within +/-5 metres")
    q = _pair(state, "qpos", np.radians([30, 60]))
    qvel = _pair(state, "qvel", [0, 0])
    if np.max(np.abs(qvel)) > 100:
        raise ValueError("Initial joint speed must be within +/-100 rad/s")
    reference = _pair(state, "joint_target", q)
    time = state.get("time", 0.0)
    hold_time = state.get("hold_time", 0.0)
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 3600 for v in (time, hold_time)):
        raise ValueError("time and hold_time must be finite seconds in [0,3600]")
    model = dynamics_model(l1, l2, torque_limit, gravity)
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:], data.time = q, qvel, time
    if kick:
        hold_time = 0.0
    data.qvel[1] += kick  # explicit velocity perturbation, not a torque impulse
    mujoco.mj_forward(model, data)
    arm = PlanarArm(l1, l2)
    settings = dict(gain=gain, damping=damping, method=method)
    command = controller_command(arm, data.qpos, target, **settings)
    initial_error = command["error_norm"]
    rates = np.array(command["joint_rates"])
    peak_applied = np.zeros(2)
    peak_requested = np.zeros(2)
    saturated, leash_active = False, False

    def torque():
        return kp*(reference-data.qpos) + kd*(rates-data.qvel) + (data.qfrc_bias if compensate else 0)

    for i in range(round(dt/TIMESTEP)):
        if i % 10 == 0:
            command = controller_command(arm, data.qpos, target, **settings)
            rates = np.array(command["joint_rates"])
        reference += TIMESTEP*rates
        limited = data.qpos + np.clip(reference-data.qpos, -.35, .35)
        leash_active |= bool(np.any(np.abs(limited-reference) > 1e-12))
        reference[:] = limited
        requested = torque()
        data.ctrl[:] = requested  # MuJoCo clamps motor ctrl internally
        mujoco.mj_step(model, data)
        peak_requested = np.maximum(peak_requested, np.abs(requested))
        peak_applied = np.maximum(peak_applied, np.abs(data.qfrc_actuator))
        saturated |= bool(np.any(np.abs(requested) > torque_limit))
        mujoco.mj_forward(model, data)
        error = np.linalg.norm(target-arm.joint_positions(*data.qpos)[-1])
        settled = error <= .01 and np.linalg.norm(data.qvel) < .03
        hold_time = hold_time+TIMESTEP if settled else 0.0
    if not np.isfinite([data.qpos,data.qvel,reference]).all() or not math.isclose(data.time,time+dt,abs_tol=1e-9):
        raise ValueError("Physics integration became unstable; reset or reduce gains")
    command = controller_command(arm, data.qpos, target, **settings)
    rates = np.array(command["joint_rates"])
    requested = torque()
    data.ctrl[:] = requested
    mujoco.mj_forward(model, data)
    actual_velocity = arm.jacobian(*data.qpos) @ data.qvel
    command["reference_joint_rates"] = command["joint_rates"]
    command["joint_rates"] = data.qvel.tolist()
    command["achieved_velocity"] = actual_velocity.tolist()
    command["velocity_residual"] = float(np.linalg.norm(np.array(command["requested_velocity"])-actual_velocity))
    if command["status"] == "reached":
        command["status"] = "holding" if hold_time >= .3 else "settling"
    elif command["status"] == "stalled":
        command["status"] = "local_stall"  # PD and gravity still evolve the physical state
    canonical = (data.qpos+math.pi) % (2*math.pi)-math.pi
    reference_canonical = (reference+math.pi) % (2*math.pi)-math.pi
    return {
        "pose": pose_payload(*np.degrees(canonical), l1, l2),
        "reference_pose": pose_payload(*np.degrees(reference_canonical), l1, l2),
        "state": {"qpos": data.qpos.tolist(), "qvel": data.qvel.tolist(),
                  "joint_target": reference.tolist(), "time": float(data.time), "hold_time": hold_time},
        "command": command, "error_before": initial_error, "dt": dt,
        "dynamics": {"requested_torque": requested.tolist(), "applied_torque": data.qfrc_actuator.tolist(),
                     "joint_error": (reference-data.qpos).tolist(), "torque_limit": torque_limit,
                     "peak_applied": peak_applied.tolist(), "peak_requested": peak_requested.tolist(),
                     "saturated": saturated or bool(np.any(np.abs(requested)>torque_limit)),
                     "reference_limited": leash_active, "gravity": gravity, "compensate": compensate,
                     "kick": kick, "hand_speed": float(np.linalg.norm(actual_velocity))},
    }
