#!/usr/bin/env python3
"""Exercise real ROS processes, save a bag, and verify every replayed field.

Run with Ubuntu's ROS-enabled Python after sourcing install/setup.bash.
All child processes are owned by this script and cleaned up on failure.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import os
import signal
import subprocess
import time

import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
import rosbag2_py
from sensor_msgs.msg import JointState


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output_dir or Path("evidence") / f"ros2_demo_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    output = output.resolve()
    children, handles, transcript = [], [], []

    def report(text):
        print(text, flush=True)
        transcript.append(text)

    def start(label, command):
        handle = (output / f"{label}.log").open("w")
        handles.append(handle)
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        children.append(process)
        return process

    def stop(process):
        if process.poll() is None:
            # Signal ros2 run and its child node, not just the CLI wrapper.
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)

    def command(*argv):
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=20)
        report("$ " + " ".join(argv) + "\n" + result.stdout.strip())
        result.check_returncode()
        return result.stdout

    rclpy.init()
    probe = Node("exercise_probe")
    received = []
    def message_key(msg):
        # Compare every message field, independent of CDR alignment padding.
        return (msg.header.stamp.sec, msg.header.stamp.nanosec, msg.header.frame_id,
                tuple(msg.name), tuple(msg.position), tuple(msg.velocity), tuple(msg.effort))

    def subscribe():
        return probe.create_subscription(
            JointState, "joint_states", lambda msg: received.append(message_key(msg)), 100)

    subscription = subscribe()

    def wait_until(predicate, description, timeout=10):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Timed out waiting for {description}; inspect {output}")
            rclpy.spin_once(probe, timeout_sec=0.05)

    try:
        # First prove the standard ROS installation works before our own nodes.
        talker = start("talker", ["ros2", "run", "demo_nodes_cpp", "talker"])
        listener = start("listener", ["ros2", "run", "demo_nodes_cpp", "listener"])
        wait_until(lambda: "I heard" in (output / "listener.log").read_text(),
                   "standard talker/listener exchange")
        stop(talker)
        stop(listener)
        report("PASS: standard talker/listener exchanged messages")

        publisher = start("publisher", ["ros2", "run", "stage0_basics", "joint_publisher"])
        subscriber = start("subscriber", ["ros2", "run", "stage0_basics", "joint_subscriber"])
        wait_until(lambda: len(received) >= 5 and
                   "sample=" in (output / "subscriber.log").read_text(),
                   "our publisher/subscriber exchange")
        report("PASS: own publisher and subscriber exchanged JointState messages")

        # Save actual graph inspection output as evidence, not expected output.
        command("ros2", "node", "list")
        command("ros2", "node", "info", "/joint_publisher")
        command("ros2", "topic", "list", "-t")
        command("ros2", "topic", "info", "/joint_states", "--verbose")
        command("ros2", "topic", "echo", "/joint_states", "--once")
        command("ros2", "service", "list", "-t")
        command("ros2", "service", "type", "/reset_joint")
        result = command("ros2", "run", "stage0_basics", "reset_client")
        if "Joint motion restarted" not in result:
            raise RuntimeError("Reset client did not report success")
        result = command("ros2", "service", "call", "/reset_joint", "std_srvs/srv/Trigger", "{}")
        if "success=True" not in result:
            raise RuntimeError("Reset service did not report success")
        report("PASS: Python client and CLI both called the reset service")

        bag = output / "joint_states_bag"
        recorder = start("recorder", ["ros2", "bag", "record", "--storage", "sqlite3",
                                       "-o", str(bag), "--topics", "/joint_states"])
        wait_until(lambda: probe.count_subscribers("/joint_states") >= 3,
                   "bag recorder subscription")
        count_before = len(received)
        wait_until(lambda: len(received) >= count_before + 25, "25 recorded samples")
        stop(recorder)
        command("ros2", "bag", "info", str(bag))

        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="sqlite3"),
                    rosbag2_py.ConverterOptions("", ""))
        recorded = []
        while reader.has_next():
            topic, data, _ = reader.read_next()
            if topic == "/joint_states":
                recorded.append(message_key(deserialize_message(data, JointState)))
        del reader
        if len(recorded) < 20:
            raise RuntimeError(f"Bag has too few messages: {len(recorded)}")
        report(f"PASS: recorded {len(recorded)} joint-state messages")

        stop(publisher)
        stop(subscriber)
        wait_until(lambda: probe.count_publishers("/joint_states") == 0,
                   "live publisher to disappear")
        # Drop unread DDS samples from the live run before observing playback.
        probe.destroy_subscription(subscription)
        subscription = subscribe()
        replay_subscriber = start("replay_subscriber", [
            "ros2", "run", "stage0_basics", "joint_subscriber"])
        wait_until(lambda: probe.count_subscribers("/joint_states") >= 2,
                   "replay subscriber discovery")
        received.clear()
        # Start paused so DDS discovery completes before the first bag sample.
        player = start("player", ["ros2", "bag", "play", str(bag), "--start-paused",
                                   "--disable-keyboard-controls", "--wait-for-all-acked", "2000"])
        wait_until(lambda: probe.count_publishers("/joint_states") == 1,
                   "bag player publisher discovery")
        command("ros2", "service", "call", "/rosbag2_player/resume",
                "rosbag2_interfaces/srv/Resume", "{}")
        wait_until(lambda: player.poll() is not None, "bag playback completion", timeout=20)
        if player.returncode != 0:
            raise RuntimeError("Bag playback failed")
        report(f"Playback finished: probe received {len(received)} of {len(recorded)} messages")
        wait_until(lambda: len(received) >= len(recorded), "all replayed messages")
        if Counter(received) != Counter(recorded):
            raise RuntimeError(f"Replay differs from recording: received {len(received)}, "
                               f"expected {len(recorded)} messages")
        wait_until(lambda: "sample=" in (output / "replay_subscriber.log").read_text(),
                   "own subscriber to display replay")
        stop(replay_subscriber)
        report(f"PASS: replay delivered all {len(recorded)} recorded messages unchanged, "
               "with the live publisher stopped")
        report(f"Evidence and bag: {output}")
    except Exception as error:
        report(f"FAIL: {error}")
        raise
    finally:
        for child in reversed(children):
            stop(child)
        for handle in handles:
            handle.close()
        probe.destroy_subscription(subscription)
        probe.destroy_node()
        rclpy.try_shutdown()
        (output / "transcript.txt").write_text("\n\n".join(transcript) + "\n")


if __name__ == "__main__":
    main()
