"""Publish a synthetic hinge state at 10 Hz and serve a phase-reset request."""
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from stage0_basics.motion import joint_state


class JointPublisher(Node):
    def __init__(self):
        super().__init__("joint_publisher")
        # Reliable, volatile, keep-last-10 QoS: both ends use the same profile.
        self.publisher = self.create_publisher(JointState, "joint_states", 10)
        self.origin = self.get_clock().now()
        self.service = self.create_service(Trigger, "reset_joint", self.reset_joint)
        self.timer = self.create_timer(0.1, self.publish_state)

    def publish_state(self):
        now = self.get_clock().now()
        q, qvel = joint_state((now - self.origin).nanoseconds / 1e9)
        msg = JointState()
        msg.header.stamp = now.to_msg()
        msg.name = ["hinge"]
        msg.position = [q]
        msg.velocity = [qvel]
        # Effort is unknown: this exercise generates state, not motor torque.
        self.publisher.publish(msg)

    def reset_joint(self, request, response):
        self.origin = self.get_clock().now()
        response.success = True
        response.message = "Joint motion restarted at q=0 rad."
        self.get_logger().info(response.message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = JointPublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
