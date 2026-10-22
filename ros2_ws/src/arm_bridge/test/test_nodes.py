"""Node protocol tests in Jazzy, before starting the integration processes."""

import math
import rclpy
from arm_interfaces.msg import ArmCommand, ArmState
from geometry_msgs.msg import PointStamped
from rclpy.parameter import Parameter
from arm_bridge.simulator import ArmSimulator
from arm_bridge.controller import ArmController


def test_simulator_rejects_old_epochs_stale_and_malformed_commands():
    rclpy.init()
    node = ArmSimulator()
    try:
        cmd = ArmCommand()
        cmd.header.stamp = node.get_clock().now().to_msg()
        cmd.observation_stamp = cmd.header.stamp
        cmd.epoch = node.epoch
        cmd.observation_sequence = 1
        cmd.torque = [1.0, 2.0]
        cmd.position = [0.0, 0.0]
        cmd.velocity = [0.0, 0.0]
        node.on_command(cmd)
        assert node.command is cmd
        node.reset()
        node.on_command(cmd)
        assert node.command is None and node.stale_dropped == 1
        cmd.epoch = node.epoch
        cmd.header.stamp.sec -= 1
        node.on_command(cmd)
        assert node.stale_dropped == 2
        cmd.header.stamp = node.get_clock().now().to_msg()
        cmd.observation_stamp.sec -= 1
        node.on_command(cmd)
        assert node.stale_dropped == 3
        cmd.torque = [math.nan, 0.0]
        node.on_command(cmd)
        assert node.invalid_dropped == 1
        node.step()
        assert node.watchdog and list(node.data.ctrl) == [0, 0]
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def test_controller_rejects_unreachable_targets_and_invalid_delay():
    rclpy.init()
    node = ArmController()
    try:
        goal = node.target.copy()
        msg = PointStamped()
        msg.point.x = 4.0
        msg.point.y = 0.0
        node.on_target(msg)
        assert list(node.target) == list(goal)
        result = node.set_parameters([Parameter("command_delay_ms", value=-1.0)])
        assert not result[0].successful
        state = ArmState()
        state.header.stamp = node.get_clock().now().to_msg()
        state.epoch = 2
        state.sequence = 1
        state.position = [0.5, 1.0]
        state.velocity = [0.0, 0.0]
        node.on_state(state)
        assert (
            node.last_command.epoch == 2 and node.last_command.observation_sequence == 1
        )
        assert node.last_command.observation_stamp == state.header.stamp
        assert max(abs(v) for v in node.last_command.velocity) <= 1.5
        # Reset clears delayed work and anchors the reference in the new epoch.
        node.queue.append((999999, node.last_command))
        state.epoch = 3
        state.sequence = 1
        node.on_state(state)
        assert not node.queue and node.last_command.epoch == 3
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
