# PickCube: stopping and bounds-aware action experiment

The [privileged stopping diagnostic](STOPPING_RESULTS.md) tests a narrow causal
question after natural placement; its successes are not autonomous PPO. The
separate bounded-action experiment trains a new policy from scratch with only
the action distribution changed. Read the [predeclared protocol](BOUNDED_ACTION_PROTOCOL.md)
before interpreting its outcomes. The completed [results](BOUNDED_ACTION_RESULTS.md)
and [visual report](evidence/pickcube_bounded/index.html) retain the negative outcome.

The tanh-Gaussian policy uses SB3's squashed distribution and its log-probability
Jacobian. Actor and critic initial weights, initial latent log standard deviations,
optimizer, task, reward, controller, horizon, observation and transition budget
remain unchanged. All eight actions pass through tanh. The latent Gaussian mean
and standard deviation are not the mean and standard deviation of executed
commands. Tanh can still saturate near the bounds; removing an external clip is
not itself evidence of fine control or task success.

Use the isolated Python 3.11 learning `.venv` and the local simulator runtime:

```bash
.venv/bin/python -m pytest -q learning/maniskill/test/test_bounded_ppo.py
.venv/bin/python -m learning.maniskill.bounded_ppo train \
  --output learning/runs/pickcube_bounded_seed42
.venv/bin/python -m learning.maniskill.bounded_comparison \
  --run learning/runs/pickcube_bounded_seed42 \
  --raw-checkpoint ../stage1-pickcube/learning/runs/pickcube_ppo_seed42/final.zip
.venv/bin/python -m learning.maniskill.bounded_trace \
  --run learning/runs/pickcube_bounded_seed42 \
  --raw-checkpoint ../stage1-pickcube/learning/runs/pickcube_ppo_seed42/final.zip
.venv/bin/python -m learning.maniskill.bounded_report \
  --run learning/runs/pickcube_bounded_seed42 \
  --output learning/maniskill/evidence/pickcube_bounded
.venv/bin/python -m http.server 8774 \
  --directory learning/maniskill/evidence/pickcube_bounded
```

Training output directories must be new. Checkpoints remain in ignored runs;
configuration, hashes, outcome geometry, native videos and synchronized validation
traces are exported as evidence. The relative baseline path assumes this feature
worktree. Both loaded final checkpoints must actually contain1,048,576 transitions
and have 42 observations/eight actions. Native videos reproduce every saved
non-trajectory outcome field on the first two test seeds.

Future base and normalized PPO trainers rank validation checkpoints by **final
success, any-step success, then return**. The continuation trainer already used
that order. This correction does not rewrite the historical any-first selection
in earlier saved evidence. Primary comparisons use fixed-budget final weights;
validation-selected policies remain secondary. Validation runs in callbacks before
the corresponding optimizer update; final weights are saved after the last update.

Validation uses 100000–100009, fresh paired final tests 110000–110049. The command
audit groups near-goal actions by **pre-action** distance≤5cm and measures whether
any executed arm command has magnitude≥0.99. These are diagnostic thresholds,
not altered task success thresholds. Plots pair pre-action commands with resulting
post-step geometry and velocity. No stopping override is used in this experiment.

The stopping report additionally includes selected activation-case videos. They
illustrate the first naturally activated seed per policy, chosen after inspecting
activation timing; they are labeled separately from the first-two-seed videos and
do not change the aggregate diagnostic outcomes.
