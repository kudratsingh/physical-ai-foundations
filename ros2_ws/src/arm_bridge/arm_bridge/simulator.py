"""Physics process: only subscribed torque commands drive MuJoCo motors."""

import time
import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
from arm_interfaces.msg import ArmState, ArmCommand
from sensor_msgs.msg import JointState
from diagnostic_msgs.msg import DiagnosticArray
from std_srvs.srv import Trigger, SetBool
from .common import PERIOD, WATCHDOG, ns, finite_pair, diagnostic
from arm_dynamics import dynamics_model


class ArmSimulator(Node):
    def __init__(self):
        super().__init__("arm_simulator")
        self.model = dynamics_model()
        self.data = mujoco.MjData(self.model)
        self.epoch = 0
        self.sequence = 0
        self.command = None
        self.command_received = 0.0
        self.last_transport_ms = None
        self.last_loop_latency_ms = None
        self.stale_dropped = 0
        self.invalid_dropped = 0
        self.last_sequence = -1
        self.paused = False
        self.watchdog = True
        self.publisher = self.create_publisher(ArmState, "/arm/state", 1)
        self.joints = self.create_publisher(JointState, "/arm/joint_states", 1)
        self.status = self.create_publisher(DiagnosticArray, "/arm/simulator_status", 1)
        self.create_subscription(ArmCommand, "/arm/torque_command", self.on_command, 1)
        self.create_service(Trigger, "/arm/reset", self.on_reset)
        self.create_service(Trigger, "/arm/kick", self.on_kick)
        self.create_service(SetBool, "/arm/pause", self.on_pause)
        self.reset()
        self.create_timer(PERIOD, self.step)
        self.create_timer(0.1, self.report)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = np.radians([30, 60])
        mujoco.mj_forward(self.model, self.data)
        self.epoch += 1
        self.sequence = 0
        self.command = None
        self.last_sequence = -1
        self.watchdog = True

    def on_command(self, msg):
        now_ns = self.get_clock().now().nanoseconds
        age = (now_ns - ns(msg.header.stamp)) / 1e9
        if (
            not finite_pair(msg.torque, 1000)
            or not finite_pair(msg.position)
            or not finite_pair(msg.velocity, 2)
        ):
            self.invalid_dropped += 1
            return
        observation_age = (now_ns - ns(msg.observation_stamp)) / 1e9
        self.last_transport_ms = age * 1000
        self.last_loop_latency_ms = observation_age * 1000
        if (
            msg.epoch != self.epoch
            or not 0 <= age <= WATCHDOG
            or not 0 <= observation_age <= WATCHDOG
            or msg.observation_sequence <= self.last_sequence
        ):
            self.stale_dropped += 1
            return
        self.command = msg
        self.command_received = time.monotonic()
        self.last_sequence = msg.observation_sequence

    def step(self):
        now_ns = self.get_clock().now().nanoseconds
        age = (
            (now_ns - ns(self.command.header.stamp)) / 1e9
            if self.command
            else float("inf")
        )
        self.watchdog = (
            not self.command
            or age > WATCHDOG
            or time.monotonic() - self.command_received > WATCHDOG
        )
        self.data.ctrl[:] = [0, 0] if self.watchdog else self.command.torque
        if not self.paused:
            for _ in range(5):
                mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        msg = ArmState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.epoch, self.sequence = self.epoch, self.sequence + 1
        msg.sequence = self.sequence
        msg.sim_time = float(self.data.time)
        msg.position = self.data.qpos.tolist()
        msg.velocity = self.data.qvel.tolist()
        msg.effort = self.data.qfrc_actuator.tolist()
        msg.paused = self.paused
        self.publisher.publish(msg)
        joints = JointState(
            header=msg.header,
            name=["shoulder", "elbow"],
            position=msg.position,
            velocity=msg.velocity,
            effort=msg.effort,
        )
        self.joints.publish(joints)

    def report(self):
        diagnostic(
            self,
            self.status,
            "simulator",
            dict(
                epoch=self.epoch,
                sequence=self.sequence,
                watchdog=self.watchdog,
                stale_dropped=self.stale_dropped,
                invalid_dropped=self.invalid_dropped,
                paused=self.paused,
                sim_time=float(self.data.time),
                transport_ms=self.last_transport_ms,
                loop_latency_ms=self.last_loop_latency_ms,
            ),
            level=1 if self.watchdog else 0,
        )

    def on_reset(self, request, response):
        self.reset()
        response.success, response.message = (
            True,
            f"Reset to (30,60) degrees at rest; epoch {self.epoch}.",
        )
        return response

    def on_kick(self, request, response):
        self.data.qvel[1] += 2
        mujoco.mj_forward(self.model, self.data)
        response.success, response.message = True, "Added +2 rad/s to elbow velocity."
        return response

    def on_pause(self, request, response):
        self.paused = request.data
        response.success, response.message = (
            True,
            "Physics paused." if self.paused else "Physics resumed.",
        )
        return response


def main():
    rclpy.init()
    node = ArmSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
