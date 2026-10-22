# Stage 1: Panda PickCube state baselines

The roadmap's Stage-1 project is one manipulation benchmark with BC, PPO,
fixed-seed evaluation, videos, learning curves and an ablation, followed by
vision or a harder distribution shift. PickCube-v1 is that benchmark here.
PushCube remains the fast reference. This is the first state-baseline experiment;
it does not complete Stage 1's vision, meaningful-RL-performance or deliberate
break-and-recover gates.

Read [the measured results](PICKCUBE_RESULTS.md) and open
[the video report](evidence/stage1_pickcube/index.html). Until merged, use the
`feat/stage1-pickcube` checkout at `.worktrees/stage1-pickcube` from the original
repository. It is stacked on `feat/week4-behavior-cloning`.

## What the policy observes and controls

| Setting | Contract |
|---|---|
| Task/robot | Unmodified ManiSkill PickCube-v1, Franka Panda |
| Observation | 42 privileged float32 state values: qpos 9, qvel 9, grasp flag 1, tool pose 7, goal xyz 3, cube pose 7, tool-to-cube displacement 3, cube-to-goal displacement 3 |
| Action | 8 normalized commands: seven joint increments scaled to ±0.1 rad, plus gripper position command |
| Low-level controller | Built-in joint PD; policy actions are not motor torques |
| Reward | Upstream normalized dense reward: reaching, grasping, placement and stillness, divided by 5; success gives 1 per step |
| Success | 3D cube-center/goal distance ≤0.025 m **and** maximum absolute velocity across the seven arm joints ≤0.2 rad/s |
| Horizon/rates | Full 50 control steps at 20 Hz, five 100 Hz physics steps per action; no early success termination |
| Physics | PhysX CPU, separate scene per reset, upstream random cube position/yaw, goal position/height and joint noise |

The position tolerance is three-dimensional. Applying PushCube's xy-only test
would miscount lifting failures as successes. The static condition excludes the
two finger joints, matching the installed Panda implementation. Success does
not separately require the grasp flag; it uses the official predicate above.
Grasp and lift diagnostics do not change the reward or success condition.
State observations include simulator object pose; videos do not make this a
vision policy.

## Reproduce

Use the separate Python 3.11 learning environment and local Vulkan runtime from
[the learning setup](README.md#setup). Existing local checkouts share the
Week-3 environment/runtime via symlinks; a fresh clone can install them there.
No changes to the root MuJoCo venv or ROS container are needed.

```bash
.venv/bin/python -m learning.maniskill.experiment smoke --env-id PickCube-v1

# Download the pinned official archive; replay actual actions in CPU physics.
.venv/bin/python -m learning.maniskill.prepare_bc \
  --env-id PickCube-v1 --count 997 --output learning/.assets/pickcube_cpu_full

# Two clones, each 10,000 gradient updates. Whole-episode split and train-only stats.
.venv/bin/python -m learning.maniskill.pickcube fit \
  --dataset learning/.assets/pickcube_cpu_full/trajectory.state.pd_joint_delta_pos.physx_cpu.h5

# PPO from scratch, one fixed 1,048,576-transition budget, four CPU workers.
.venv/bin/python -m learning.maniskill.experiment train --env-id PickCube-v1 \
  --timesteps 1048576 --validation-interval 131072 --validation-episodes 10 \
  --validation-seed-start 40000 --episodes 50 --seed-start 50000 \
  --output learning/runs/pickcube_ppo_seed42

# After both fits finish: test clones, import the matching PPO/baseline tests,
# render the first three predeclared test seeds and verify rendered results match.
.venv/bin/python -m learning.maniskill.pickcube evaluate
.venv/bin/python -m learning.maniskill.pickcube publish
cp learning/.assets/pickcube_cpu_full/provenance.json \
  learning/maniskill/evidence/stage1_pickcube/data_provenance.json

# Serve the committed static report and native videos.
.venv/bin/python -m http.server 8769 --bind 127.0.0.1 \
  --directory learning/maniskill/evidence/stage1_pickcube
```

Fit and replay output directories must be new. Checkpoints/datasets are ignored
under `learning/runs/` and `learning/.assets/`; configs, hashes, losses,
per-episode outcomes, plots and videos are committed. Existing local checkpoints
are `learning/runs/pickcube_bc/bc_{32,256}/best.pt` and
`learning/runs/pickcube_ppo_seed42/best.zip` in this checkout.

## Protocol and selection

Training seed is 42, demonstration permutation seed 2026. The first 64 randomly
permuted successful CPU replay episodes are validation; the next 256 are
training. BC-32 uses the first 32 of that same training subset. Terminal
observations have no action labels and are dropped. Expert action labels are
clipped to [-1,1], matching the normalized controller limits. Each clone uses
its own training-only mean/std (floor 1e-3), a 42–256–256–8 ReLU/Tanh MLP, Adam
3e-4 and batches of 256. Source trajectory IDs and initial-observation hashes
are recorded so repeated source seed values cannot disguise duplicate episodes.

Validation rollouts use seeds 40000–40009, every 1,000 BC updates (also update
1) and every 131,072 PPO transitions. Checkpoints maximize validation success
at any step, then final success, then negative held-out action MSE for BC or
mean return for PPO. This selection rule intentionally exposes temporary versus
sustained success. Final testing uses only the selected checkpoints on seeds
50000–50049, with deterministic actions and full horizons. All policies see the
same test resets. Random controls have a separate seeded RNG. Zero commands
hold joint targets; they do not imply zero torque.

PPO uses the existing CPU adaptation: separate 256×256 actor/value networks,
learning rate 3e-4, gamma .99, GAE .95, clip .2, eight epochs, 256 steps per
worker, batch 256, no entropy bonus. The upstream CUDA recipe uses a much larger
budget and parallelism; this run is not a reproduction of that result. BC has
external expert knowledge; this comparison matches task/protocol, not acquisition
cost. Confidence intervals describe episode sampling, not training-seed variance.

## Diagnostics and verification

Each episode records any/final success, return, first-success step, any grasp,
any placement, maximum cube height, 5 cm lift relative to reset, terminal cube
and goal positions, and seven arm velocities. Mutually exclusive outcome
categories are final success, reached then lost, placed without ever satisfying
static success, lifted without success, grasped without a 5 cm lift, and never
grasped. These describe measured milestones, not inferred intentions.

Run optional learning tests in the learning venv:

```bash
.venv/bin/python -m pytest -q learning/maniskill/test tests/learning
```

Tests distinguish 3D position from xy placement, check the static condition,
run full real-simulator episodes, reject mismatched demonstration tasks, and
round-trip both new 42-input and legacy 35-input clone checkpoints. Root CI
independently recomputes reported metrics and terminal success from saved
positions/velocities. Linux ManiSkill CI trains/reloads tiny PPO checkpoints
for both tasks; it does not download demonstrations or retrain full baselines.

Primary references: [PickCube implementation](https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/envs/tasks/tabletop/pick_cube.py),
[Panda controller/static predicate](https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/agents/robots/panda/panda.py),
[official demonstrations](https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations),
[upstream PPO examples](https://github.com/mani-skill/ManiSkill/blob/main/examples/baselines/ppo/examples.sh).
Exact installed package versions and archive SHA-256 are in the evidence.
