"""Integration checks run inside Jazzy, using real nodes and DDS transport."""
from pathlib import Path
import subprocess
import sys

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from stage0_basics.joint_publisher import JointPublisher
from stage0_basics.joint_subscriber import JointSubscriber


def test_live_state_and_reset_service():
    rclpy.init()
    publisher = JointPublisher()
    subscriber = JointSubscriber()
    client_node = Node("reset_test_client")
    executor = SingleThreadedExecutor()
    for node in (publisher, subscriber, client_node):
        executor.add_node(node)
    try:
        import time
        deadline = time.monotonic() + 10
        while subscriber.received_count < 3 and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
        assert subscriber.received_count >= 3
        msg = subscriber.last_state
        assert msg.name == ["hinge"]
        assert len(msg.position) == len(msg.velocity) == 1
        assert abs(msg.position[0]) <= 0.5
        assert msg.header.stamp.sec > 0
        client = client_node.create_client(Trigger, "reset_joint")
        assert client.wait_for_service(timeout_sec=5)
        old_origin = publisher.origin
        future = client.call_async(Trigger.Request())
        executor.spin_until_future_complete(future, timeout_sec=5)
        assert future.done() and future.result().success
        assert publisher.origin > old_origin
        assert (publisher.get_clock().now() - publisher.origin).nanoseconds < 1e9

        # A bad message must not crash the callback or overwrite valid state.
        valid_state = subscriber.last_state
        malformed = JointState(name=["hinge"])
        subscriber.on_state(malformed)
        assert subscriber.last_state is valid_state
    finally:
        executor.shutdown()
        for node in (publisher, subscriber, client_node):
            node.destroy_node()
        rclpy.try_shutdown()


def test_real_cli_record_and_replay(tmp_path):
    script = Path(__file__).resolve().parents[3] / "scripts" / "run_demo.py"
    result = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(tmp_path / "evidence")],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=100)
    assert result.returncode == 0, result.stdout
    assert "recorded messages unchanged" in result.stdout


def test_reset_client_fails_when_server_is_absent():
    result = subprocess.run(["ros2", "run", "stage0_basics", "reset_client"],
                            text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=15)
    assert result.returncode != 0
    assert "unavailable" in result.stdout
