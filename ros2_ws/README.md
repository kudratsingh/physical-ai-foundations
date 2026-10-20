# Week 1 ROS 2 exercise

Run ROS 2 Jazzy in Ubuntu 24.04 using Docker Desktop on the Mac. This workspace
uses Ubuntu's Python inside the container; the root `.venv` is for MuJoCo.

The exercise publishes a synthetic hinge angle and angular velocity, displays
them in another node, resets the motion with a service, and records/replays the
topic. There is no MuJoCo connection yet.

## Start and build

From the repository root, with Docker Desktop running:

```bash
docker compose -f ros2_ws/compose.yaml up -d --build
docker compose -f ros2_ws/compose.yaml exec ros2 bash
```

Inside that Ubuntu shell:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select stage0_basics
source install/setup.bash
```

The workspace is mounted from the Mac, so source changes and generated bags
remain available locally. Every new container shell needs both setup scripts
sourced. All exercise nodes run in this one container, in ROS domain 42, with
localhost discovery. Docker host networking is not required for this setup.

## Run the full verified exercise

Inside the sourced Ubuntu shell:

```bash
python3 scripts/run_demo.py
```

It runs standard talker/listener nodes, runs our nodes, prints graph inspection
commands and their actual output, calls the reset service through our Python
client and the ROS CLI, records a SQLite bag, stops the live publisher, and
replays the bag. It checks that every recorded joint-state message arrives
unchanged during replay. Leave manual exercise nodes stopped while running it.

Each run saves its bag, node logs and `transcript.txt` under
`ros2_ws/evidence/ros2_demo_<timestamp>/` on the Mac. Generated evidence and
build outputs are ignored by Git.

## Explore and test

See [the package walkthrough](src/stage0_basics/README.md) for the graph,
individual commands, and what you should observe at each step.

Run the ROS tests inside the sourced Ubuntu shell:

```bash
python3 -m pytest -v src/stage0_basics/test
```

The suite checks motion units and phase, real publication/subscription and
service callbacks, malformed-message handling, client failure without a
server, and the full CLI recording/replay exercise. The GitHub ROS workflow
builds the same container and runs the same suite on Ubuntu. The original
MuJoCo/transform tests continue to run separately.

See [validation results](VALIDATION.md) for the environment and verified scope.

Stop the container from the repository root when finished:

```bash
docker compose -f ros2_ws/compose.yaml down
```

For native Ubuntu 24.04, install Jazzy plus the apt dependencies listed in
`Dockerfile`, then follow the same build, exercise and test commands.
