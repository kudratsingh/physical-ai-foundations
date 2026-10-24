# How much teacher anchoring preserves precision manipulation?

[Protocol](STRENGTH_PROTOCOL.md) · [Results](STRENGTH_RESULTS.md) ·
[Native visual report](evidence/retention_strength/index.html)

PR #19 compares the **same frozen-teacher action objective** at coefficients 10
and 25. The original PickCube reward, controller, 50-step horizon, bounded critic,
optimizer, training budget and validation selection remain unchanged. All six
models start fresh from the original BC-256 teacher; PR #18's selected models are
preserved separately and supply neither initialization nor additional targets.
Evaluation remains fully autonomous, with no teacher override.

Use the separate Python 3.11 ManiSkill environment documented in [setup](README.md).
Do not install these dependencies into the root Python environment. Full models,
including historical selected-model copies, remain ignored local artifacts.

```bash
.venv/bin/python -m learning.maniskill.strength_experiment declare \
  --output learning/runs/strength_fresh \
  --teacher learning/runs/pickcube_bc/bc_256/best.pt \
  --reference learning/runs/actor_retention_v2

# Repeat both lambda10 and lambda25 for seeds 42,1042,2042.
.venv/bin/python -m learning.maniskill.strength_experiment train \
  --output learning/runs/strength_fresh --seed 42 --arm lambda10 \
  --teacher learning/runs/pickcube_bc/bc_256/best.pt
.venv/bin/python -m learning.maniskill.strength_experiment train \
  --output learning/runs/strength_fresh --seed 42 --arm lambda25 \
  --teacher learning/runs/pickcube_bc/bc_256/best.pt

# Checks every declared budget and checkpoint before creating a test environment.
.venv/bin/python -m learning.maniskill.strength_compare \
  --run learning/runs/strength_fresh \
  --teacher learning/runs/pickcube_bc/bc_256/best.pt
.venv/bin/python -m learning.maniskill.strength_report \
  --run learning/runs/strength_fresh --output learning/runs/strength_fresh/report

python3 -m http.server 8779 --bind 127.0.0.1 \
  --directory learning/maniskill/evidence/retention_strength
```

The machine protocol and launch sources freeze before training. A fixed value of
25 is motivated by PR #18's uncapped calibration, not by a new coefficient search.
Forty existing validation resets select checkpoints; 50 new paired test resets
250000–250049 supply the primary fixed-budget results and secondary selected-model
results. Sharing test resets across three training seeds does not create 150
independent reset trials. Neither the test set nor historical selected models
supply teacher supervision.

The report includes every successful-then-static-displaced episode, including
unfavorable cases, alongside complete compressed passive traces. Distinguish
commands from measured motion, task-static from zero velocity, and observed grasp
flags from contact causality. Videos use the first two declared test resets,
regardless of outcomes; their complete records must match full evaluation.

The pinned SB3 update and [license](SB3_LICENSE.txt) remain unchanged from PR #18.
Contract tests check frozen coefficients, teacher state, checkpoint restoration,
reset partitions and guards. Thin evidence audits independently reconstruct the
physical outcomes and event conditioning. CI smoke runs do not independently
repeat the full six-model training comparison.
