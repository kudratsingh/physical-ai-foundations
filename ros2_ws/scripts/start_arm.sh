#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
cd /workspaces/foundations/ros2_ws
python3 /usr/bin/colcon build --symlink-install --packages-select arm_interfaces arm_bridge --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
exec python3 scripts/arm_system.py
