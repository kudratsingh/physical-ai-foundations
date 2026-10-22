# Week 3: ManiSkill PushCube and PPO

The roadmap's Week-3 instruction is to run an Isaac Lab or ManiSkill RL
quickstart, train/evaluate one policy, and document observations/actions/reward.
This experiment uses **ManiSkill's existing PushCube-v1 task**, unchanged, with
its official CPU Gymnasium wrapper and Stable-Baselines3 PPO. It is a Mac CPU
adaptation of the learning workflow, not a reproduction of the upstream CUDA
PPO baseline or its published performance. ROS stays outside training.

## Setup

Use the Week-3 feature checkout. Until merged, from the original repo:

```bash
cd .worktrees/week3-maniskill
```

This checkout has its own Python 3.11 environment; the root MuJoCo/ROS dependency
environments remain separate.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r learning/maniskill/requirements-macos.lock.txt
# Apple Silicon/macOS 14+: local MoltenVK driver, no system install or shell edits.
.venv/bin/python -m learning.maniskill.install_macos_runtime
.venv/bin/python -m learning.maniskill.experiment smoke
```

The Mac nightly currently forces a render-system initialization even with
`render_backend="none"`. The runtime helper downloads the official Homebrew
MoltenVK 1.4.2 Apple Silicon Sonoma bottle, checks its published SHA-256, and
extracts it under ignored `learning/.runtime/`. The experiment automatically
selects that local driver. SAPIEN supplies its own Vulkan loader. A configured
Vulkan SDK can instead be selected via `VK_ICD_FILENAMES`. See the upstream
[Mac instructions](https://github.com/mani-skill/ManiSkill/blob/main/docs/source/user_guide/getting_started/macos_install.md)
and [Homebrew package](https://formulae.brew.sh/formula/molten-vk).

For **Linux x86-64 CPU**, use Python 3.11, install `libgl1 libegl1 libvulkan1 mesa-vulkan-drivers`,
install CPU PyTorch from its official wheel index, then install `requirements.in`.
Select the Lavapipe ICD in `VK_ICD_FILENAMES` on machines without a GPU.
SAPIEN still needs a render-capable device to create scene materials, even for
state-only learning; the experiment selects its CPU-compatible render backend.
The full Mac lock captures the exact observed environment; Linux resolution is
separate. CI tests that Linux CPU path. CUDA-scale training is a later setup,
not something this Mac run silently enables.

## Task contract

| Choice | Meaning |
|---|---|
| Robot/task | Franka Panda pushes a cube into a goal region on a table |
| Observation | 35 float32 privileged state values: robot qpos/qvel, tool pose, goal position and cube pose |
| Action | 8 normalized values in [-1,1]: 7 joint position increments (scaled to ±0.1 rad) plus gripper position command |
| Low-level control | ManiSkill's built-in joint PD; actions are neither torques nor Cartesian deltas |
| Reward | Unmodified upstream `normalized_dense`: reaching behind the cube, gated placement/height shaping, success reward; dense reward divided by 4 |
| Success | Cube xy is within 0.1 m of goal xy, and cube z is below half-size + 0.005 m |
| Horizon | 50 control steps; no early success termination during evaluation |
| Rates | Measured: 100 Hz physics, 20 Hz control; five physics steps per action |
| Reset | Upstream random cube position and robot joint noise; scene reconfigures each reset |

State observations include ground truth object pose. This is a **state-based**
learning baseline, not a vision policy. Native rendered video is for inspection
and never becomes additional policy input. The experiment uses `CPUGymWrapper`
to remove ManiSkill's one-environment tensor batch dimension.

The wrapper's optional per-step `info['episode']` metrics are disabled: they
conflict with SB3 Monitor's terminal-only `r/l/t` format. Evaluation independently
accumulates both success definitions over every full episode, following the
[upstream evaluation protocol](https://github.com/mani-skill/ManiSkill/blob/main/docs/source/user_guide/reinforcement_learning/setup.md).

## Train and evaluate

```bash
.venv/bin/python -m learning.maniskill.experiment train \
  --timesteps 262144 --num-envs 4 --episodes 20 \
  --output learning/runs/pushcube_ppo_seed42
```

Training uses CPU PyTorch, four independent CPU physics processes (spawn), seed
42, 256 rollout steps per worker, batches of 256, eight update epochs, learning
rate 3e-4, gamma .99, GAE lambda .95, clip .2, and separate 256×256 actor/critic
networks. This is a small introductory budget, not the upstream multi-million
transition GPU recipe. Worker PyTorch threads are limited to one.

Every 16,384 transitions saves `latest.zip` and evaluates validation seeds
1000–1004. The final checkpoint is selected by the fixed training budget, not by
test-set success. Final tests use seeds 10000–10019 for PPO, random, and zero
actions. PPO evaluation is deterministic. Random actions have a separate seeded
RNG. Zero actions mean zero joint increments plus the neutral gripper command;
they do not mean zero applied torque.

Outputs under ignored `learning/runs/<run>/`:

- `config.json`, `training.json`: settings, versions, source hash and actual budget/time.
- `tensorboard/`, `validation.json`: optimization logs and validation learning curve.
- `latest.zip`, `final.zip`: local policy checkpoints.
- `test_ppo.json`, `test_random.json`, `test_zero.json`: per-seed outcomes and measured tool/cube/goal/joint trajectories.
- `comparison.json`, `results.png`: aggregate metrics and plots.

A run directory must be new; existing evidence is never overwritten by training.

```bash
# Native simulator video for the same first held-out seed, including failures.
.venv/bin/python -m learning.maniskill.experiment evaluate \
  --checkpoint learning/runs/pushcube_ppo_seed42/final.zip \
  --episodes 1 --seed-start 10000 --video \
  --output learning/runs/pushcube_ppo_seed42/video_ppo.json

# TensorBoard (optional)
.venv/bin/tensorboard --logdir learning/runs/pushcube_ppo_seed42/tensorboard
```

## What the result establishes

See [the first-run report](RESULTS.md). Report success at **any** step and at the
**last** step separately, returns across the same seeds, and a Wilson interval
for success. One trained model and 20 episodes are an introductory result;
the interval captures episode sampling uncertainty, not training-seed variance.
An increased shaped return alone does not prove the cube-pushing task is solved.

The next learning experiment should use the evidence to choose a longer budget
or a controlled change, then compare on a fresh untouched test set. Week 4 adds
imitation learning on the same task. The Stage-1 success gate remains separate
from completing this Week-3 training/evaluation artifact.

## Tests and references

```bash
.venv/bin/python -m pytest tests/learning learning/maniskill/test
```

Pure metric tests run in the foundations suite. Real ManiSkill tests check full
horizon/terminal semantics, SB3 Monitor interoperability, finite state, seeded
repeatability and trace recording. CI additionally trains a tiny policy, saves
it, evaluates all three baselines and generates the report plot. That CI run is
a pipeline check, not a policy-quality benchmark.

- [Official ManiSkill quickstart](https://github.com/mani-skill/ManiSkill/blob/main/docs/source/user_guide/getting_started/quickstart.md)
- [Upstream PushCube task/reward](https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/envs/tasks/tabletop/push_cube.py)
- [Upstream PPO baseline](https://github.com/mani-skill/ManiSkill/tree/main/examples/baselines/ppo)
- [SB3 PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

## PickCube failure investigation

The [placement and BC-initialized PPO investigation](BC_PPO_README.md) adds
observation-only stop diagnostics, matched actor initialization, isolated
evaluation RNG, stochastic training metrics and paired native videos. It reports
the early selected improvement separately from fixed-budget regression.
