"""Display the same angle and velocity fields used in the MuJoCo exercises."""
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointSubscriber(Node):
    def __init__(self):
        super().__init__("joint_subscriber")
        self.received_count = 0
        self.last_state = None
        self.subscription = self.create_subscription(
            JointState, "joint_states", self.on_state, 10)

    def on_state(self, msg):
        if "hinge" not in msg.name:
            self.get_logger().warning("Ignoring state without hinge joint")
            return
        i = msg.name.index("hinge")
        if len(msg.position) <= i or len(msg.velocity) <= i:
            self.get_logger().warning("Ignoring state with missing position/velocity")
            return
        self.received_count += 1
        self.last_state = msg
        self.get_logger().info(
            f"sample={self.received_count} q={msg.position[i]:+.3f} rad "
            f"qvel={msg.velocity[i]:+.3f} rad/s")


def main(args=None):
    rclpy.init(args=args)
    node = JointSubscriber()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
