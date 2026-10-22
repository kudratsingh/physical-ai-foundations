"""Synthetic joint motion for the middleware exercise; no physics engine."""
import math


def joint_state(elapsed: float) -> tuple[float, float]:
    """Return q [rad] and dq/dt [rad/s]: amplitude 0.5 rad, frequency 0.5 Hz."""
    return 0.5 * math.sin(math.pi * elapsed), 0.5 * math.pi * math.cos(math.pi * elapsed)
