"""Call the reset service once, with bounded waits and meaningful exit codes."""
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


def main(args=None):
    rclpy.init(args=args)
    node = Node("reset_client")
    client = node.create_client(Trigger, "reset_joint")
    try:
        if not client.wait_for_service(timeout_sec=5.0):
            node.get_logger().error("reset_joint unavailable; start joint_publisher first")
            return 1
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
        if not future.done():
            node.get_logger().error("reset_joint request timed out")
            return 1
        response = future.result()
        node.get_logger().info(response.message)
        return 0 if response.success else 1
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
