# ROS 2 validation

Verified on October 20, 2026 using Docker Desktop on an Apple Silicon Mac,
with Ubuntu 24.04 aarch64, ROS 2 Jazzy and Python 3.12 inside the container.

## Commands and results

From the repository root:

```bash
docker compose -f ros2_ws/compose.yaml up -d --build
docker compose -f ros2_ws/compose.yaml exec -T ros2 bash -c '
  source /opt/ros/jazzy/setup.bash &&
  colcon build --symlink-install --packages-select stage0_basics &&
  source install/setup.bash &&
  python3 -m pytest -v src/stage0_basics/test
'
```

Build: one package succeeded. Tests: **8 passed**. The tests exercised real
DDS communication and real `ros2` CLI subprocesses, including:

- Standard talker/listener exchange.
- Our publisher/subscriber exchanging timestamped hinge positions and velocities.
- Our Python reset client and `ros2 service call` receiving successful responses.
- Node, topic, service and QoS inspection.
- Recording a nonempty SQLite bag and inspecting its metadata.
- Stopping the live publisher, waiting for playback discovery, resuming the
  paused bag player through its service, and checking every recorded message
  field arrives unchanged during replay.
- Our subscriber displaying replayed messages.
- Malformed joint-state handling and bounded client failure without a server.

The existing Mac suite was also run with `.venv/bin/python -m pytest`:
**31 passed**.

## Saved evidence

`python3 scripts/run_demo.py` inside the sourced Ubuntu shell saves a new
`evidence/ros2_demo_<UTC timestamp>/` directory containing the actual graph
transcript, node logs, recorder/player logs and `joint_states_bag/` with its
SQLite data and metadata. These generated artifacts persist on the Mac through
the workspace mount and are ignored by Git.

## Scope of this result

This verifies the Week-1 ROS communication artifacts. The publisher generates
synthetic joint motion; there is no MuJoCo bridge or action server. Actions are
covered conceptually in the walkthrough. Hands-on inspection and answering
the oral questions without notes are still the owner's competency checks.
