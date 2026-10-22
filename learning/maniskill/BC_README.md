# Week 4: behavior cloning on the same PushCube task

This follows the guide's imitation-learning step: train a policy from
observed state/action pairs, compare it with the RL baseline, publish an
experiment report, and select the next manipulation benchmark.

The source is **official ManiSkill RL-generated demonstrations**, not our
Week-3 PPO policy. The archive already contains `pd_joint_delta_pos` actions,
so both learned approaches use the same 35-dimensional state, eight joint/gripper
commands, 50-step horizon, CPU physics, reward and success definition.

## Prepare demonstrations

Use this feature checkout and the Week-3 Python 3.11 dependencies. Until merged:

```bash
cd .worktrees/week4-behavior-cloning
```

This local worktree reuses the existing Week-3 `.venv` and rendering runtime.
For a fresh checkout, follow [the Week-3 setup](README.md) to create them.

```bash
.venv/bin/python -m learning.maniskill.prepare_bc --count 512
```

The helper downloads only the pinned 24.6 MB official PushCube archive, verifies
its SHA-256, preserves raw data, and invokes the upstream replay tool on the
first 512 joint-delta trajectories. It restores each original initial state,
then **simulates the recorded actions in CPU physics** to generate observations.
It does not teleport through the entire trajectory or use the downloaded expert
checkpoint. Only successful CPU replays are retained. A working metadata copy
selects the CPU-compatible renderer; the original metadata remains intact.

The archive SHA-256 is
`1ce24408bf93658501faaa0ef164a0cdc29539cc412612a413fac746697eeab2`.
Replay provenance captures raw/replayed hashes and retained/attempted counts.
CPU/GPU differences can change replay success; retention is part of the result,
not an assumption that every downloaded episode stays successful.

The official batched collection uses a common reset seed for many distinct
trajectories. Split by trajectory ID and verify distinct initial observations,
rather than treating that shared seed as an independent episode identifier.

## Train the two BC baselines and compare

```bash
.venv/bin/python -m learning.maniskill.behavior_cloning run \
  --dataset learning/.assets/bc_replay/trajectory.state.pd_joint_delta_pos.physx_cpu.h5 \
  --ppo ../week3-maniskill/learning/runs/pushcube_ppo_seed42/final.zip \
  --output learning/runs/pushcube_bc_comparison \
  --updates 5000 --episodes 50
```

`--ppo` points to the **unchanged Week-3 checkpoint**. If using a fresh clone,
train that checkpoint with the [Week-3 command](README.md) and supply its path.
Model checkpoints and full demonstration data are ignored by Git.

The measured local run used the same upstream replay command directly on the
raw archive's RL trajectory file. Its actual input hashes and all split IDs are
recorded in `protocol.json`; the preparation helper provides a portable command
for subsequent runs without depending on that original raw-data directory.

## What BC learns

Each pair is `(observation[t], action[t])`. A trajectory contains T+1 observations
and T actions, so the terminal observation is excluded. Shifting labels by one
timestep would teach a different, incorrect mapping.

A 35→256→256→8 network with ReLU hidden layers and Tanh output minimizes action
MSE with Adam, learning rate 3e-4 and batches of 256. PyTorch runs on CPU with one
thread. Observation means and per-feature standard deviations come **only from
training episodes**, with a 1e-3 standard-deviation floor, and are saved in the
checkpoint. The output is bounded to the normalized controller action space.

Some stored expert commands exceed [-1,1]. ManiSkill clips normalized commands
before scaling them to physical targets; labels are clipped to the same effective
commands. This preprocessing choice is recorded in the protocol. No reward
optimization or critic is used during BC training; rewards are measured only
when evaluating rollouts.

The **32 versus 256 demonstration** experiment keeps architecture, optimizer,
training seed 42 and 5,000 gradient updates fixed. The smaller set is nested in
the larger set. It is update-budget matched, not epoch matched. More data does
not necessarily produce a higher score in one small experiment.

## Separation and model selection

- Dataset split RNG: 2026. Hold out 64 whole demonstration episodes for action MSE.
  Reserve 256 other episodes for training, with the first 32 forming the small set.
- Rollout validation: reset seeds **30000–30009**, full 50-step episodes.
- Checkpoint selection: validation success at any step, then final success, then
  lower demonstration-validation MSE. Selection never uses final test outcomes.
- Fresh final tests: seeds **20000–20049**, identically reset for both BC policies,
  fixed PPO, random actions and zero actions. The old Week-3 test set is not reused.

The selected BC checkpoint is reloaded from disk before final testing. Both
success at any step and success at the last step are reported. The unchanged
PPO is rerun on these fresh seeds; its Week-3 20-episode result is not treated as
an interchangeable score on the new distribution sample.

This is a performance comparison under the stated training conditions, not a
compute-matched comparison of algorithms. BC receives an external expert's data;
the cost of producing that expert is not included in BC optimizer time. Success-
only replay retention also selects a subset of expert starting conditions.

## Inspect results and videos

See [the Week-4 experiment report](BC_RESULTS.md). The committed evidence includes
per-episode results, loss/rollout curves, split/provenance metadata and native
videos from the first test seeds. Full traces and checkpoints remain in ignored
`learning/runs/pushcube_bc_comparison/`.

```bash
# First three fresh test seeds, with the selected BC policy.
.venv/bin/python -m learning.maniskill.behavior_cloning video \
  --checkpoint learning/runs/pushcube_bc_comparison/bc_256/best.pt \
  --seed-start 20000 --episodes 3 \
  --output learning/runs/pushcube_bc_comparison/video_bc.json
```

Low action MSE on held-out expert states does not guarantee good rollouts. The
student's errors change the states it later visits; these states can fall outside
the demonstration distribution. This is covariate shift and compounding error.
Inspect complete trajectories and success, rather than selecting a model from
supervised loss alone. The first-run report distinguishes measured evidence from
possible explanations of individual failures.

## Tests and references

```bash
.venv/bin/python -m pytest learning/maniskill/test tests/learning
```

Tests cover observation/action alignment, action clipping, whole-episode splits,
train-only normalization, bounded inference, exact checkpoint reload and a trained
BC policy executing a complete real simulator episode. The existing ManiSkill CI
runs these alongside the PPO checkpoint pipeline. No expert dataset download is
needed for CI's synthetic data-contract checks.

- [Official imitation-learning setup](https://github.com/mani-skill/ManiSkill/blob/main/docs/source/user_guide/learning_from_demos/setup.md)
- [Official BC baseline](https://github.com/mani-skill/ManiSkill/tree/main/examples/baselines/bc)
- [Official demonstrations](https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations)
- [Upstream trajectory replay](https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/trajectory/replay_trajectory.py)
