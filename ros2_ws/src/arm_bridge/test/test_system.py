"""Real separate-process ROS/DDS, services, latency injection and bag replay."""

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest

ROOT = Path(os.environ["FOUNDATIONS_ROOT"])
URL = "http://127.0.0.1:8766"


def get(source="live"):
    with urlopen(URL + "/api/ros/status?source=" + source, timeout=3) as response:
        return json.load(response)


def post(action, body=None):
    request = Request(
        URL + "/api/ros/" + action,
        data=json.dumps(body or {}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=8) as response:
        return json.load(response)


def wait_for(predicate, timeout=12, source="live"):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            last = get(source)
            if predicate(last):
                return last
        except (URLError, ConnectionError):
            pass
        time.sleep(0.05)
    pytest.fail(f"Timed out; last observation: {last}")


def distance(s):
    goal = [float(s["controller"]["target_x"]), float(s["controller"]["target_y"])]
    return math.hypot(*(a - b for a, b in zip(s["pose"]["positions"][-1], goal)))


@pytest.fixture(scope="module", autouse=True)
def system(tmp_path_factory):
    log = (tmp_path_factory.mktemp("arm") / "system.log").open("w+")
    p = subprocess.Popen(
        [sys.executable, str(ROOT / "ros2_ws/scripts/arm_system.py")],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        wait_for(lambda s: s.get("state") and len(s["nodes"]) >= 3, timeout=20)
        yield p
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait(timeout=3)
        log.seek(0)
        print(log.read())
        log.close()


def test_independent_nodes_and_real_convergence(system):
    s = wait_for(lambda s: s.get("pose") and s["controller"] and distance(s) < 0.02)
    assert {"arm_simulator", "arm_controller", "arm_dashboard"} <= set(s["nodes"])
    assert s["state_rate_hz"] > 50
    assert 0 <= s["loop_latency_ms"] < 100
    # One process per node; no in-process callback shortcut.
    process_list = subprocess.check_output(["ps", "-eo", "args"], text=True)
    for module in ["simulator", "controller", "dashboard"]:
        assert f"-m arm_bridge.{module}" in process_list
    assert max(abs(v) for v in s["state"]["effort"]) <= 20.00001


def test_topic_target_service_reset_pause_and_kick():
    assert post("target", {"target": [1.1, 0.8]})["success"]
    wait_for(lambda s: float(s["controller"].get("target_x", 0)) == 1.1)
    wait_for(lambda s: distance(s) < 0.02)
    assert post("pause", {"paused": True})["success"]
    before = wait_for(lambda s: s["state"]["paused"])
    assert post("reset")["success"]
    reset = wait_for(lambda s: s["state"]["epoch"] > before["state"]["epoch"])
    assert reset["state"]["sim_time"] == 0
    assert reset["state"]["velocity"] == [0, 0]
    assert post("kick")["success"]
    kick = wait_for(lambda s: s["state"]["velocity"][1] > 1.9)
    assert kick["state"]["sim_time"] == 0  # paused physics preserves the perturbation
    assert post("pause", {"paused": False})["success"]
    wait_for(
        lambda s: distance(s) < 0.02 and math.hypot(*s["state"]["velocity"]) < 0.05
    )


def test_delay_is_measured_and_stale_commands_are_rejected():
    assert post("delay", {"delay_ms": 50})["success"]
    delayed = wait_for(
        lambda s: (
            s["command_transport_ms"] is not None and s["command_transport_ms"] >= 45
        )
    )
    assert delayed["loop_latency_ms"] >= 45
    assert post("delay", {"delay_ms": 200})["success"]
    old = int(delayed["simulator"]["stale_dropped"])
    stale = wait_for(
        lambda s: (
            s["simulator"].get("watchdog") == "True"
            and int(s["simulator"]["stale_dropped"]) > old
        )
    )
    assert stale["state"]["effort"] == [0, 0]
    post("delay", {"delay_ms": 0})
    post("reset")
    wait_for(lambda s: s["simulator"].get("watchdog") == "False" and distance(s) < 0.02)


def test_invalid_target_and_delay_do_not_replace_valid_commands():
    old = get()["controller"].get("target_x")
    for action, body in [
        ("target", {"target": [4, 0]}),
        ("target", {"target": ["bad", 0]}),
        ("target", {"target": None}),
        ("delay", {"delay_ms": 301}),
    ]:
        with pytest.raises(HTTPError) as error:
            post(action, body)
        assert error.value.code == 409
    assert get()["controller"]["target_x"] == old


def test_recorded_bag_replays_unchanged_on_isolated_topics():
    import rclpy
    from rclpy.node import Node
    from arm_interfaces.msg import ArmState
    from rclpy.serialization import deserialize_message, serialize_message
    import rosbag2_py

    assert post("record")["success"]
    bag = ROOT / get()["bags"]["bag"]
    log = bag.parent / "record.log"
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        if log.exists() and "All requested topics are subscribed" in log.read_text():
            break
        time.sleep(0.05)
    else:
        pytest.fail("Recorder did not discover all requested topics")
    time.sleep(2.5)
    post("stop_record")
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    originals = []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == "/arm/state":
            originals.append(serialize_message(deserialize_message(data, ArmState)))
    assert len(originals) > 100
    rclpy.init()
    probe = Node("arm_replay_probe")
    received = []
    probe.create_subscription(
        ArmState,
        "/replay/arm/state",
        lambda m: received.append(serialize_message(m)),
        2000,
    )
    try:
        before = get()["state"]["sequence"]
        assert post("replay")["success"]
        deadline = time.monotonic() + 12
        while len(received) < len(originals) and time.monotonic() < deadline:
            rclpy.spin_once(probe, timeout_sec=0.1)
        assert received == originals
        print(
            f"Verified all {len(originals)} replayed state messages unchanged (canonical CDR)."
        )
        replay = wait_for(lambda s: s.get("state") is not None, source="replay")
        assert replay["source"] == "replay"
        assert (
            get()["state"]["sequence"] > before
        )  # live simulation kept running independently
        # Bag player has no publisher on the live torque input.
        publishers = probe.get_publishers_info_by_topic("/arm/torque_command")
        assert len(publishers) == 1 and publishers[0].node_name == "arm_controller"
    finally:
        post("stop_replay")
        probe.destroy_node()
        rclpy.try_shutdown()
