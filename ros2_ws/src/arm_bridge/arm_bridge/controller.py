"""Independent ROS feedback controller. It never calls mj_step or edits physics state."""

from collections import deque
import time
import numpy as np
import mujoco
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from arm_interfaces.msg import ArmState, ArmCommand
from geometry_msgs.msg import PointStamped
from diagnostic_msgs.msg import DiagnosticArray
from .common import finite_pair, diagnostic
from planar_arm import PlanarArm
from arm_dynamics import dynamics_model
from resolved_rate import controller_command


class ArmController(Node):
    def __init__(self):
        super().__init__("arm_controller")
        self.declare_parameter("command_delay_ms", 0.0)
        self.add_on_set_parameters_callback(self.parameters_changed)
        self.arm = PlanarArm()
        self.model = dynamics_model()
        self.data = mujoco.MjData(
            self.model
        )  # model-based bias estimate, no physics integration
        self.target = np.array([12 / 13, 5 / 13])
        self.reference = None
        self.epoch = None
        self.last_sequence = -1
        self.last_sim_time = 0.0
        self.last_outer = -1.0
        self.rates = np.zeros(2)
        self.queue = deque(maxlen=64)
        self.last_command = None
        self.last_state_received = 0.0
        self.invalid = 0
        self.publisher = self.create_publisher(ArmCommand, "/arm/torque_command", 1)
        self.status = self.create_publisher(
            DiagnosticArray, "/arm/controller_status", 1
        )
        self.create_subscription(ArmState, "/arm/state", self.on_state, 1)
        self.create_subscription(PointStamped, "/arm/target", self.on_target, 1)
        self.create_timer(0.002, self.flush)
        self.create_timer(0.1, self.report)

    def parameters_changed(self, parameters):
        for p in parameters:
            if p.name == "command_delay_ms" and (
                type(p.value) not in (int, float) or not 0 <= p.value <= 300
            ):
                return SetParametersResult(
                    successful=False, reason="delay must be 0 to 300 ms"
                )
        self.queue.clear()  # changes do not release commands from the old delay experiment
        return SetParametersResult(successful=True)

    def on_target(self, msg):
        goal = [msg.point.x, msg.point.y]
        if (
            not finite_pair(goal, 5)
            or msg.point.z != 0
            or not self.arm.inverse_kinematics(*goal).reachable
        ):
            self.invalid += 1
            return
        self.target = np.array(goal)

    def on_state(self, msg):
        if not finite_pair(msg.position) or not finite_pair(msg.velocity, 100):
            self.invalid += 1
            return
        if self.epoch is not None and msg.epoch < self.epoch:
            return
        if msg.epoch != self.epoch:
            self.epoch = msg.epoch
            self.reference = np.array(msg.position)
            self.last_sequence = -1
            self.last_sim_time = msg.sim_time
            self.last_outer = -1
            self.queue.clear()
        if msg.sequence <= self.last_sequence:
            return
        self.last_sequence = msg.sequence
        self.last_state_received = time.monotonic()
        dt = max(0.0, min(0.03, msg.sim_time - self.last_sim_time))
        self.last_sim_time = msg.sim_time
        q, qvel = np.array(msg.position), np.array(msg.velocity)
        if msg.sim_time - self.last_outer >= 0.019 or self.last_outer < 0:
            c = controller_command(self.arm, q, self.target)
            self.rates = np.array(c["joint_rates"])
            self.last_outer = msg.sim_time
        self.reference += dt * self.rates
        self.reference = q + np.clip(self.reference - q, -0.35, 0.35)
        self.data.qpos[:], self.data.qvel[:] = q, qvel
        mujoco.mj_forward(self.model, self.data)
        torque = (
            80 * (self.reference - q) + 12 * (self.rates - qvel) + self.data.qfrc_bias
        )
        output = ArmCommand()
        output.header.stamp = self.get_clock().now().to_msg()
        output.header.frame_id = "world"
        output.observation_stamp = msg.header.stamp
        output.epoch, output.observation_sequence = msg.epoch, msg.sequence
        output.position = self.reference.tolist()
        output.velocity = self.rates.tolist()
        output.torque = torque.tolist()
        self.last_command = output
        delay = float(self.get_parameter("command_delay_ms").value) / 1000
        if delay == 0:
            self.publisher.publish(output)
        else:
            self.queue.append((time.monotonic() + delay, output))

    def flush(self):
        now = time.monotonic()
        if now - self.last_state_received > 0.15:
            self.queue.clear()  # old state cannot produce commands indefinitely
            return
        while self.queue and self.queue[0][0] <= now:
            self.publisher.publish(self.queue.popleft()[1])

    def report(self):
        diagnostic(
            self,
            self.status,
            "controller",
            dict(
                target_x=self.target[0],
                target_y=self.target[1],
                delay_ms=self.get_parameter("command_delay_ms").value,
                epoch=self.epoch,
                invalid=self.invalid,
                state_age_ms=(time.monotonic() - self.last_state_received) * 1000
                if self.last_state_received
                else -1,
            ),
        )


def main():
    rclpy.init()
    node = ArmController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
