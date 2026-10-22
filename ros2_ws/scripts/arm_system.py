#!/usr/bin/env python3
"""Launch three independent ROS processes; stop only children owned by this run."""

import os
import signal
import subprocess
import sys
import time

children = []
stop = False


def requested_stop(*args):
    global stop
    stop = True


signal.signal(signal.SIGTERM, requested_stop)
signal.signal(signal.SIGINT, requested_stop)
try:
    for module in ("controller", "simulator", "dashboard"):
        children.append(
            subprocess.Popen(
                [sys.executable, "-m", f"arm_bridge.{module}"], start_new_session=True
            )
        )
    while not stop:
        for child in children:
            if child.poll() is not None:
                raise RuntimeError(
                    f"Arm process {child.pid} exited with {child.returncode}"
                )
        time.sleep(0.2)
finally:
    for child in children:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)
    for child in children:
        try:
            child.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
