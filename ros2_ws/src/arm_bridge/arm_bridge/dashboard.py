"""ROS observation/command bridge plus a localhost-published browser dashboard.

HTTP threads enqueue ROS operations; the ROS executor handles publications and
service futures. The dashboard never calls the simulator's implementation.
"""

from collections import deque
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
import json
import math
import os
import queue
import signal
import subprocess
import threading
import time
from urllib.parse import urlsplit, parse_qs

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.srv import SetParameters
from arm_interfaces.msg import ArmState, ArmCommand
from geometry_msgs.msg import PointStamped
from diagnostic_msgs.msg import DiagnosticArray
from std_srvs.srv import Trigger, SetBool
from .common import ROOT, ns, finite_pair
from planar_arm import PlanarArm, pose_payload
from arm_viewer import ArmHandler

TOPICS = [
    "/arm/state",
    "/arm/joint_states",
    "/arm/torque_command",
    "/arm/target",
    "/arm/simulator_status",
    "/arm/controller_status",
]


class BagManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.recorder = None
        self.player = None
        self.last_bag = None
        self.log = None

    def stop_process(self, p):
        if p is not None and p.poll() is None:
            os.killpg(p.pid, signal.SIGINT)
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGTERM)
                p.wait(timeout=3)

    def action(self, action):
        with self.lock:
            if action == "record":
                if self.recorder and self.recorder.poll() is None:
                    return {"success": False, "message": "Already recording"}
                run = (
                    ROOT
                    / "ros2_ws/evidence"
                    / ("arm_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
                )
                run.mkdir(parents=True)
                self.last_bag = run / "bag"
                log = (run / "record.log").open("w")
                self.recorder = subprocess.Popen(
                    [
                        "ros2",
                        "bag",
                        "record",
                        "--storage",
                        "sqlite3",
                        "-o",
                        str(self.last_bag),
                        *TOPICS,
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                log.close()
                return {
                    "success": True,
                    "message": "Recording live state, commands, targets and diagnostics",
                }
            if action == "stop_record":
                self.stop_process(self.recorder)
                return {
                    "success": True,
                    "message": "Recording stopped; metadata finalized",
                }
            if action == "replay":
                if self.recorder and self.recorder.poll() is None:
                    return {"success": False, "message": "Stop recording before replay"}
                if not self.last_bag or not (self.last_bag / "metadata.yaml").exists():
                    return {"success": False, "message": "Record and stop a bag first"}
                self.stop_process(self.player)
                log = (self.last_bag.parent / "play.log").open("w")
                self.player = subprocess.Popen(
                    [
                        "ros2",
                        "bag",
                        "play",
                        str(self.last_bag),
                        "--delay",
                        "1",
                        "--remap",
                        *[f"{t}:=/replay{t}" for t in TOPICS],
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                log.close()
                return {
                    "success": True,
                    "message": "Replay publishes only /replay/arm/*; live control is isolated",
                }
            if action == "stop_replay":
                self.stop_process(self.player)
                return {"success": True, "message": "Replay stopped"}
            raise ValueError("Unknown bag action")

    def status(self):
        with self.lock:
            return dict(
                recording=bool(self.recorder and self.recorder.poll() is None),
                replaying=bool(self.player and self.player.poll() is None),
                bag=str(self.last_bag.relative_to(ROOT)) if self.last_bag else None,
            )

    def close(self):
        with self.lock:
            self.stop_process(self.recorder)
            self.stop_process(self.player)


class ArmDashboard(Node):
    def __init__(self):
        super().__init__("arm_dashboard")
        self.lock = threading.Lock()
        self.jobs = queue.Queue()
        self.cache = {
            source: dict(
                state=None,
                command=None,
                state_received=0,
                command_received=0,
                arrivals=deque(maxlen=200),
                simulator={},
                controller={},
                simulator_received=0,
                controller_received=0,
            )
            for source in ("live", "replay")
        }
        self.nodes = []
        self.bags = BagManager()
        self.target_publisher = self.create_publisher(PointStamped, "/arm/target", 1)
        self.ros_clients = {
            "reset": self.create_client(Trigger, "/arm/reset"),
            "kick": self.create_client(Trigger, "/arm/kick"),
            "pause": self.create_client(SetBool, "/arm/pause"),
            "delay": self.create_client(
                SetParameters, "/arm_controller/set_parameters"
            ),
        }
        for source, prefix in (("live", ""), ("replay", "/replay")):
            self.create_subscription(
                ArmState,
                prefix + "/arm/state",
                lambda msg, s=source: self.on_state(s, msg),
                1,
            )
            self.create_subscription(
                ArmCommand,
                prefix + "/arm/torque_command",
                lambda msg, s=source: self.on_command(s, msg),
                1,
            )
            for kind in ("simulator", "controller"):
                self.create_subscription(
                    DiagnosticArray,
                    prefix + f"/arm/{kind}_status",
                    lambda msg, s=source, k=kind: self.on_diagnostic(s, k, msg),
                    1,
                )
        self.create_timer(0.01, self.dispatch)
        self.create_timer(1, self.update_nodes)

    def update_nodes(self):
        with self.lock:
            self.nodes = sorted(set(self.get_node_names()))

    def on_state(self, source, msg):
        if not finite_pair(msg.position) or not finite_pair(msg.velocity, 100):
            return
        now = time.monotonic()
        with self.lock:
            c = self.cache[source]
            if (
                source == "live"
                and c["state"]
                and (msg.epoch, msg.sequence) <= (c["state"].epoch, c["state"].sequence)
            ):
                return
            c["state"], c["state_received"] = msg, now
            c["arrivals"].append(now)

    def on_command(self, source, msg):
        with self.lock:
            c = self.cache[source]
            c["command"], c["command_received"] = msg, time.monotonic()
            c["command_transport_ms"] = (
                self.get_clock().now().nanoseconds - ns(msg.header.stamp)
            ) / 1e6
            c["loop_latency_ms"] = (
                self.get_clock().now().nanoseconds - ns(msg.observation_stamp)
            ) / 1e6

    def on_diagnostic(self, source, kind, msg):
        values = {v.key: v.value for item in msg.status for v in item.values}
        with self.lock:
            self.cache[source][kind] = values
            self.cache[source][kind + "_received"] = time.monotonic()

    def operate(self, action, body):
        event = threading.Event()
        result = {}
        self.jobs.put((action, body, event, result))
        if not event.wait(3):
            event.set()
            return {
                "success": False,
                "message": "ROS operation timed out; check current state before retrying",
            }
        return result

    def dispatch(self):
        for _ in range(8):
            try:
                action, body, event, result = self.jobs.get_nowait()
            except queue.Empty:
                return
            if event.is_set():
                continue
            if action == "target":
                target = body.get("target", [])
                if (
                    not finite_pair(target, 5)
                    or not PlanarArm().inverse_kinematics(*target).reachable
                ):
                    result.update(
                        success=False,
                        message="Target must be inside the arm's reachable annulus",
                    )
                else:
                    msg = PointStamped()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.header.frame_id = "world"
                    msg.point.x, msg.point.y = float(target[0]), float(target[1])
                    self.target_publisher.publish(msg)
                    result.update(
                        success=True,
                        message="Target published; controller diagnostics confirm receipt",
                    )
                event.set()
                continue
            client = self.ros_clients.get(action)
            if client is None or not client.service_is_ready():
                result.update(success=False, message="ROS service unavailable")
                event.set()
                continue
            if action == "pause":
                request = SetBool.Request(data=bool(body.get("paused", True)))
            elif action == "delay":
                value = body.get("delay_ms", 0)
                if (
                    not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not 0 <= value <= 300
                ):
                    result.update(success=False, message="Delay must be 0 to 300 ms")
                    event.set()
                    continue
                request = SetParameters.Request(
                    parameters=[
                        Parameter(
                            "command_delay_ms", Parameter.Type.DOUBLE, float(value)
                        ).to_parameter_msg()
                    ]
                )
            else:
                request = Trigger.Request()
            future = client.call_async(request)

            def complete(f, event=event, result=result, action=action):
                try:
                    response = f.result()
                    if action == "delay":
                        result.update(
                            success=all(r.successful for r in response.results),
                            message="Command delay parameter updated",
                        )
                    else:
                        result.update(
                            success=response.success, message=response.message
                        )
                except Exception as error:
                    result.update(success=False, message=str(error))
                event.set()

            future.add_done_callback(complete)

    def snapshot(self, source):
        now = time.monotonic()
        wall = self.get_clock().now().nanoseconds
        with self.lock:
            c = self.cache[source]
            state, cmd = c["state"], c["command"]
            arrivals = [t for t in c["arrivals"] if now - t < 2]
            result = dict(
                source=source,
                nodes=list(self.nodes),
                simulator=dict(c["simulator"]),
                controller=dict(c["controller"]),
                state_receive_age_ms=(now - c["state_received"]) * 1000
                if state
                else None,
                controller_receive_age_ms=(now - c["controller_received"]) * 1000
                if c["controller_received"]
                else None,
                state_rate_hz=(len(arrivals) - 1) / (arrivals[-1] - arrivals[0])
                if len(arrivals) > 1 and arrivals[-1] > arrivals[0]
                else 0,
                command_age_ms=(wall - ns(cmd.header.stamp)) / 1e6 if cmd else None,
                command_transport_ms=c.get("command_transport_ms"),
                loop_latency_ms=c.get("loop_latency_ms"),
            )
            if state:
                q = [(q + math.pi) % (2 * math.pi) - math.pi for q in state.position]
                result.update(
                    pose=pose_payload(*[math.degrees(v) for v in q]),
                    state=dict(
                        epoch=state.epoch,
                        sequence=state.sequence,
                        sim_time=state.sim_time,
                        position=list(state.position),
                        velocity=list(state.velocity),
                        effort=list(state.effort),
                        paused=state.paused,
                    ),
                    state_age_ms=(wall - ns(state.header.stamp)) / 1e6,
                )
            if cmd and state and cmd.epoch == state.epoch:
                q = [(q + math.pi) % (2 * math.pi) - math.pi for q in cmd.position]
                result.update(
                    reference_pose=pose_payload(*[math.degrees(v) for v in q]),
                    command=dict(
                        epoch=cmd.epoch,
                        observation_sequence=cmd.observation_sequence,
                        torque=list(cmd.torque),
                        joint_target=list(cmd.position),
                        velocity=list(cmd.velocity),
                    ),
                )
        if source == "replay":
            for field, key in (
                ("command_transport_ms", "transport_ms"),
                ("loop_latency_ms", "loop_latency_ms"),
            ):
                value = result["simulator"].get(key)
                try:
                    result[field] = float(value)
                except (TypeError, ValueError):
                    result[field] = None
        result["bags"] = self.bags.status()
        return result


class RosHandler(ArmHandler):
    bridge = None

    def do_GET(self):
        url = urlsplit(self.path)
        if url.path in ("/", "/ros"):
            self.send_content(
                200,
                (ROOT / "ros2_ws/web/arm_ros.html").read_bytes(),
                "text/html; charset=utf-8",
            )
        elif url.path == "/arm_ros.js":
            self.send_content(
                200, (ROOT / "ros2_ws/web/arm_ros.js").read_bytes(), "text/javascript"
            )
        elif url.path == "/api/ros/status":
            source = parse_qs(url.query).get("source", ["live"])[0]
            if source not in ("live", "replay"):
                self.send_content(400, b"Invalid source", "text/plain")
                return
            self.send_content(
                200,
                json.dumps(self.bridge.snapshot(source), allow_nan=False).encode(),
                "application/json",
            )
        elif url.path == "/lab":
            self.send_content(
                200,
                (ROOT / "math/arm_viewer.html").read_bytes(),
                "text/html; charset=utf-8",
            )
        else:
            super().do_GET()

    def do_POST(self):
        path = urlsplit(self.path).path
        if not path.startswith("/api/ros/"):
            super().do_POST()
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 4096:
                raise ValueError("Invalid request size")
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError("Request must be an object")
            action = path.rsplit("/", 1)[-1]
            if action in ("record", "stop_record", "replay", "stop_replay"):
                result = self.bridge.bags.action(action)
                if action == "replay" and result["success"]:
                    with self.bridge.lock:
                        c = self.bridge.cache["replay"]
                        c.update(
                            state=None,
                            command=None,
                            state_received=0,
                            command_received=0,
                            simulator={},
                            controller={},
                            simulator_received=0,
                            controller_received=0,
                            command_transport_ms=None,
                            loop_latency_ms=None,
                        )
                        c["arrivals"].clear()
            elif action in ("target", "reset", "kick", "pause", "delay"):
                result = self.bridge.operate(action, body)
            else:
                raise ValueError("Unknown ROS operation")
            self.send_content(
                200 if result["success"] else 409,
                json.dumps(result).encode(),
                "application/json",
            )
        except (ValueError, TypeError, OverflowError) as error:
            self.send_content(
                400,
                json.dumps({"success": False, "message": str(error)}).encode(),
                "application/json",
            )


def main():
    rclpy.init()
    node = ArmDashboard()
    RosHandler.bridge = node
    server = ThreadingHTTPServer(("0.0.0.0", 8766), RosHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print("ROS arm dashboard: http://localhost:8766", flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        node.bags.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
