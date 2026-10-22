# PickCube PPO: online observation normalization

This controlled Stage-1 assignment asks whether scaling state observations improves
PickCube placement and holding. It retains the upstream task, reward, 50-step
horizon, eight-dimensional joint-delta action space and existing PPO architecture.
The only training treatment is Stable-Baselines3 `VecNormalize` with `norm_obs=True`,
`norm_reward=False` and `clip_obs=10`. There are no warmup samples or demonstrations.

The primary comparison uses two **fixed-budget final checkpoints**, each trained
from scratch for 1,048,576 transitions with seed 42 and four CPU workers. The raw
checkpoint is the original PickCube run's `final.zip`, rather than its selected
`best.zip`. Both use the same learning rate, batch size, rollout length, optimizer
epoch count and actor/critic networks. Fifty fresh paired test seeds, 90000–90049,
are reserved before training. Native videos show the first three seeds.

[Visual report](evidence/pickcube_normalization/index.html) ·
[Results and interpretation](NORMALIZATION_RESULTS.md)

## Run in the isolated learning environment

Use the learning worktree's Python 3.11 `.venv`, with the pinned ManiSkill
requirements and, on macOS, the existing MoltenVK runtime. Do not install these
packages into the root MuJoCo environment.

```bash
.venv/bin/python -m pytest -q learning/maniskill/test/test_normalized_ppo.py
.venv/bin/python -m learning.maniskill.normalized_ppo train \
  --output learning/runs/pickcube_normalization_seed42
.venv/bin/python -m learning.maniskill.normalization_comparison \
  --output learning/runs/pickcube_normalization_seed42 \
  --raw-checkpoint ../stage1-pickcube/learning/runs/pickcube_ppo_seed42/final.zip
.venv/bin/python -m learning.maniskill.action_saturation \
  --raw-checkpoint ../stage1-pickcube/learning/runs/pickcube_ppo_seed42/final.zip \
  --normalized-snapshot learning/runs/pickcube_normalization_seed42/final \
  --output learning/maniskill/evidence/pickcube_normalization/action_saturation.json
.venv/bin/python -m learning.maniskill.normalization_report \
  --run learning/runs/pickcube_normalization_seed42 \
  --output learning/maniskill/evidence/pickcube_normalization
.venv/bin/python -m pytest -q tests/learning/test_normalization_evidence.py
.venv/bin/python -m http.server 8771 \
  --directory learning/maniskill/evidence/pickcube_normalization
```

Each training output directory must be new. Checkpoints and their pickled local
statistics stay in ignored `learning/runs/`; the report exports JSON statistics,
checksums, configuration, validation curves, episode geometry, plots and MP4s.
The raw checkpoint path above is relative to this isolated feature worktree.
The comparison verifies the actual loaded policies each contain 1,048,576
transitions and expose 42 observation and eight action components; filenames and
sidecar training metadata alone are insufficient.

## Evaluation and synchronization

Every snapshot contains a model, its contemporaneous normalization statistics,
a JSON export and SHA-256 checksums. `FrozenPolicy.load` verifies the pair, creates
an independent normalizer and disables updates. The shared evaluator supplies raw
observations; the adapter normalizes them with the frozen training statistics
before policy prediction. Reward normalization is disabled throughout, so returns
retain original task units. SB3 itself normalizes terminal observations correctly.

Validation seeds 80000–80009 are separate from training and final tests. Validation
loads a separate snapshot and cannot change training statistics. SB3 `PPO.load`
reconstructs a seeded policy and resets global Python, NumPy and Torch generators;
a scoped RNG-preservation context covers loading, evaluation and cleanup to avoid
altering the learner's future sampling. The real-simulator regression tests check
these next random draws exactly. An initial pilot stopped after discovering this
confound is excluded from the definitive results and retained only in ignored runs.

The normalized validation-selected checkpoint is reported as **secondary**. It is
chosen lexicographically by any success, final success and mean return. The raw
run originally used different validation seeds, so comparing selected policies
would not isolate normalization as cleanly as the primary final-versus-final test.

Official success requires 3D cube-goal distance at most 0.025 m **and** all seven
arm-joint speeds at most 0.2 rad/s. Episodes run all 50 steps even after first
success. A 5 cm lift is only a diagnostic. Video reruns must reproduce every saved
non-trajectory test field, rather than merely matching returns. Independent tests
recompute final success from saved coordinates and velocities.

## What this assignment can establish

This is one trained seed per treatment and 50 paired resets, not a multi-seed
algorithm ranking. It can expose a practical observation-scaling issue for these
two concrete policies. Training time is descriptive: the existing raw baseline
was trained earlier on the same Mac, and the treatment includes its own validation
overhead. No test outcome is used to select a training budget or checkpoint.

The optional post-training action audit uses only validation seeds 80000–80009.
It saves all 500 pre-clip deterministic Gaussian means per policy and checks that
clipping them exactly reproduces the policy's commands. Arm-joint command
saturation is separate from PPO's probability-ratio clipping and from gripper
closure. It describes behavior; it does not establish a cause or test a fix.
