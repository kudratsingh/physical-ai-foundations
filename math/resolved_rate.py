"""Position feedback using the planar arm's differential kinematics.

This is kinematic velocity control: integrate commanded joint rates. There
are no masses, torques, contacts or PD dynamics in this experiment.
"""
import math

import numpy as np

from planar_arm import PlanarArm, pose_payload

POSITION_TOLERANCE = 0.01  # metres


def controller_command(arm, q, target, *, gain=2.0, damping=0.12,
                       method="dls", max_hand_speed=0.8, max_joint_speed=1.5):
    q, target = np.asarray(q, dtype=float), np.asarray(target, dtype=float)
    if q.shape != (2,) or target.shape != (2,) or not np.isfinite([q, target]).all():
        raise ValueError("Joint angles and target must be finite pairs")
    if method not in ("dls", "pinv"):
        raise ValueError("Method must be dls or pinv")
    for name, value in (("gain", gain), ("damping", damping),
                        ("max_hand_speed", max_hand_speed), ("max_joint_speed", max_joint_speed)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    position = arm.joint_positions(*q)[-1]
    error = target - position
    error_norm = float(np.linalg.norm(error))
    inner, outer = arm.workspace_radii
    radius = math.hypot(*target)
    reachable = inner - 32*math.ulp(outer) <= radius <= outer + 32*math.ulp(outer)
    requested = gain * error
    hand_limited = bool(np.linalg.norm(requested) > max_hand_speed)
    if hand_limited:
        requested *= max_hand_speed / np.linalg.norm(requested)
    J = arm.jacobian(*q)
    U, sigma, Vt = np.linalg.svd(J)
    if method == "dls":
        factors = sigma / (sigma**2 + damping**2)
    else:
        factors = np.zeros_like(sigma)
        np.divide(1, sigma, out=factors, where=sigma > 2*np.finfo(float).eps*sigma[0])
    raw_rates = Vt.T @ (factors * (U.T @ requested))
    peak = float(np.max(np.abs(raw_rates)))
    scale = min(1.0, max_joint_speed / peak) if peak else 1.0
    rates = raw_rates * scale  # scale both together, preserving direction
    status = "tracking"
    if not reachable:
        status = "unreachable"
        rates[:] = 0
    elif error_norm <= POSITION_TOLERANCE:
        status = "reached"
        rates[:] = 0
        requested[:] = 0
        raw_rates[:] = 0
        scale = 1.0
        hand_limited = False
    achieved = J @ rates
    if status == "tracking" and np.linalg.norm(achieved) < 1e-7:
        status = "stalled"
        rates[:] = 0
        achieved[:] = 0
    return {
        "target": target.tolist(), "error": error.tolist(), "error_norm": error_norm,
        "requested_velocity": requested.tolist(), "achieved_velocity": achieved.tolist(),
        "raw_joint_rates": raw_rates.tolist(), "joint_rates": rates.tolist(),
        "velocity_residual": float(np.linalg.norm(requested-achieved)),
        "hand_limited": hand_limited, "rate_limited": scale < 1 and status == "tracking",
        "reachable": bool(reachable), "status": status,
        "method": method, "damping": damping, "tolerance": POSITION_TOLERANCE,
    }


def control_payload(q1_deg, q2_deg, x, y, l1=1.0, l2=0.75, *, dt=0.04, **settings):
    """One bounded Euler step plus a preview evaluated at the RETURNED pose.

    dt=0 refreshes feedback without moving. The server is stateless; the client
    sends its last configuration each step. No analytic IK branch is injected.
    """
    if not math.isfinite(dt) or not 0 <= dt <= 0.05:
        raise ValueError("dt must be between 0 and 0.05 seconds")
    arm = PlanarArm(l1, l2)
    q = np.radians([q1_deg, q2_deg])
    before = controller_command(arm, q, [x, y], **settings)
    q_next = q + dt*np.array(before["joint_rates"])
    q_next = (q_next + math.pi) % (2*math.pi) - math.pi
    after = controller_command(arm, q_next, [x, y], **settings)
    return {
        "pose": pose_payload(*np.degrees(q_next), l1, l2),
        "command": after, "applied_joint_rates": before["joint_rates"],
        "error_before": before["error_norm"], "dt": dt,
    }
