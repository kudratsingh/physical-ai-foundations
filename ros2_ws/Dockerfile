FROM ros:jazzy-ros-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-colcon-common-extensions python3-pytest \
    ros-jazzy-sensor-msgs ros-jazzy-std-srvs ros-jazzy-rosbag2 \
    ros-jazzy-demo-nodes-cpp \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces/ros2_ws
CMD ["sleep", "infinity"]
