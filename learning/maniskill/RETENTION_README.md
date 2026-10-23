# Seeded actor retention after BC initialization

[Predeclared protocol](RETENTION_PROTOCOL.md) · [Measured results](RETENTION_RESULTS.md) ·
[Native visual dashboard](evidence/actor_retention/index.html)

This compares two bounded-critic PPO conditions initialized from the same BC
teacher. One uses the usual PPO objective; the other adds a frozen-teacher
mean-action loss on PPO training states. Teacher inference supplies supervision,
so this is teacher-regularized reinforcement learning, not PPO from scratch.
The environment reward and original task-success benchmark stay unchanged.

The policy mean, rather than a clipped stochastic command or network pre-tanh
output, is regularized. The loss sums squared differences over all eight actions
then averages over states. It gives no direct gradient to the separate critic
or exploration log-standard-deviation. Shared clipping, Adam and changed state
visitation can still change their updates indirectly. The local PPO target-KL
guard remains active in both arms; it is not a fixed-teacher KL penalty.

Use the separate ManiSkill Python 3.11 environment from [setup](README.md).
Full checkpoints and the original teacher remain ignored local artifacts.
Use a new run directory; coefficient and launch sources must stay frozen after
calibration begins. The six performance conditions start fresh from the teacher.

```bash
.venv/bin/python -m learning.maniskill.retention_experiment declare \
  --output learning/runs/retention_fresh \
  --teacher learning/runs/pickcube_bc/bc_256/best.pt
.venv/bin/python -m learning.maniskill.first_rollout \
  --output learning/runs/retention_fresh/first_rollout --workers 4 \
  --teacher learning/runs/pickcube_bc/bc_256/best.pt
.venv/bin/python -m learning.maniskill.retention_experiment calibrate \
  --output learning/runs/retention_fresh \
  --teacher learning/runs/pickcube_bc/bc_256/best.pt

# Repeat both baseline and anchored conditions for seeds 42,1042,2042.
.venv/bin/python -m learning.maniskill.retention_experiment train \
  --output learning/runs/retention_fresh --seed 42 --arm baseline \
  --teacher learning/runs/pickcube_bc/bc_256/best.pt
.venv/bin/python -m learning.maniskill.retention_experiment train \
  --output learning/runs/retention_fresh --seed 42 --arm anchored \
  --teacher learning/runs/pickcube_bc/bc_256/best.pt

# Refuses held-out evaluation until all six actual final budgets pass.
.venv/bin/python -m learning.maniskill.retention_compare \
  --run learning/runs/retention_fresh \
  --teacher learning/runs/pickcube_bc/bc_256/best.pt
.venv/bin/python -m learning.maniskill.retention_report \
  --run learning/runs/retention_fresh --output learning/runs/retention_fresh/report

# The committed dashboard needs no simulator dependencies.
python3 -m http.server 8778 --bind 127.0.0.1 \
  --directory learning/maniskill/evidence/actor_retention
```

The dashboard separates fixed-budget final and validation-selected performance,
shows three training-seed replicates and passive holding diagnostics, and lets
you choose a training seed for paired native videos. Conditioned subsets have
explicit denominators. Immediate-success states differ from all later states
after the first success; neither is an outcome-selected success-only benchmark.
Gripper command changes, actual finger motion and grasp-loss flags are separate.
Their associations do not establish contact/control causality.

The extended optimizer update derives from pinned SB3 2.9.0; its upstream source
hash is checked before construction, and [SB3's MIT license](SB3_LICENSE.txt) is
included. A zero-coefficient regression requires bitwise identical PPO weights.
The frozen teacher is stored with checkpoints and excluded from policy Adam.
All gradients are observed read-only; the decomposition probe runs outside the
actual gradient hooks and preserves the minibatch-shuffle RNG.

A preflight compares every captured field in a full first stochastic rollout
across four separately seeded workers, including automatic resets. Value outputs
and timeout-adjusted buffer rewards are excluded because critic parameterization
can change them. Actual first performance-rollout summaries and initial tensor,
RNG, teacher and optimizer manifests are checked again within all three pairs.

Validation uses 40 fixed resets, and final evaluation uses 50 fresh paired resets
shared across training seeds. Report training-seed results and dispersion; do
not pool the repeated reset list into 150 independent trials. CI runs contracts,
short simulator tests and saved-evidence audits; it does not reproduce the full
six-condition learning experiment.
