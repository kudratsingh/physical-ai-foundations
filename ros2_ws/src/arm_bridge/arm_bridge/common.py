"""Shared units, validation and diagnostics; no simulator/controller coupling."""

import math
import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get("FOUNDATIONS_ROOT", Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(ROOT / "math"))

PERIOD = 0.01  # 100 Hz ROS exchange; 5 x 2 ms MuJoCo steps per observation
WATCHDOG = 0.15  # seconds, based on command generation time


def ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def finite_pair(values, limit=10000):
    try:
        return len(values) == 2 and all(
            math.isfinite(v) and abs(v) <= limit for v in values
        )
    except (TypeError, ValueError):
        return False


def diagnostic(node, publisher, name, values, level=0):
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

    msg = DiagnosticArray()
    msg.header.stamp = node.get_clock().now().to_msg()
    item = DiagnosticStatus(
        name=name,
        hardware_id="mujoco_arm",
        level=bytes([level]),
        message="ok" if level == 0 else "check telemetry",
    )
    item.values = [KeyValue(key=k, value=str(v)) for k, v in values.items()]
    msg.status = [item]
    publisher.publish(msg)
