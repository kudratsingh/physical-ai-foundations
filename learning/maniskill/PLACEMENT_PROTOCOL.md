# P0: observation-only stopping and placement diagnostics

This additive assignment performs no training, checkpoint selection or final test
evaluation. It investigates two questions using existing fixed checkpoints.

1. Does an observation-only hand-coded stop gate exactly reproduce the previous
   privileged diagnostic on validation seeds 80000–80009?
2. How do ordinary BC, raw PPO and normalized PPO approach and occupy the goal
   region on fresh shared validation seeds 130000–130009?

The installed flattened state observation is verified numerically against the
simulator: `qpos[0:9]`, `qvel[9:18]`, grasp flag at 18, TCP pose at 19:26, goal position at 26:29,
cube pose at 29:36, TCP-to-cube vector at 36:39, cube-to-goal vector at 39:42. The pure gate
reads only the raw 42-dimensional observation and tests the last vector's float32 norm against
float32(0.025). It has no environment, info dictionary or actor input.

For the hybrid, an observed placed state at time t latches zero seven-arm
increments from the next action onward, preserving each step's original policy
gripper command. Privileged placement is audit-only and is never an input to the gate
or command decision. Every reset and post-step observation must match
the privileged audit. Entire episode records, actions and outcomes must exactly
match the saved privileged override reference. This remains a hand-coded hybrid,
not PPO alone; zero delta retargets current measured joints rather than fixing
one activation pose. Normalized-policy statistics remain frozen.

The placement comparison uses the existing 256-demo BC best checkpoint, raw PPO
final checkpoint and normalized PPO final snapshot. There is no override. All
three get identical resets, raw 42 observations, joint-delta actions and the
unchanged 50-step horizon. Each post-action record includes grasping, geometry,
speeds, command magnitudes, distance changes and remaining steps. Occupancy at
10, 5 and 2.5 cm is pooled over grasped steps and reported with its denominator;
those correlated steps are not independent trials. All ten episodes per policy
are summarized first. The BC final-success subset is a separately labeled,
post-hoc conditional inspection, not a performance comparison against PPO.
External expert data and PPO training compute were not matched.

A protocol JSON with timestamp, source/gate/checkpoint/installed-schema hashes,
versions, seed lists and measurement definitions is saved before any diagnostic
run. The entire load/run is wrapped in CPU Python/NumPy/Torch RNG preservation.
Source and checkpoint hashes are checked again at run start. No parameters,
normalization statistics or data splits are fitted here.

Videos use the first two placement seeds for each policy, regardless of outcome,
and must reproduce full episode records exactly. Hybrid videos reuse the native
privileged reference only after complete observation-only trajectory equality;
manifest hashes document that reuse. Compact JSONL stores one episode per line.
Video manifests contain trajectory digests instead of duplicated full traces.

```bash
.venv/bin/python -m pytest -q learning/maniskill/test/test_placement_diagnostic.py
.venv/bin/python -m learning.maniskill.placement_diagnostic declare \
  --output learning/runs/pickcube_placement_p0_v2 \
  --raw-checkpoint ../stage1-pickcube/learning/runs/pickcube_ppo_seed42/final.zip \
  --normalized-snapshot ../pickcube-normalization/learning/runs/pickcube_normalization_seed42/final \
  --teacher ../stage1-pickcube/learning/runs/pickcube_bc/bc_256/best.pt
.venv/bin/python -m learning.maniskill.placement_diagnostic run \
  --output learning/runs/pickcube_placement_p0_v2 \
  --raw-checkpoint ../stage1-pickcube/learning/runs/pickcube_ppo_seed42/final.zip \
  --normalized-snapshot ../pickcube-normalization/learning/runs/pickcube_normalization_seed42/final \
  --teacher ../stage1-pickcube/learning/runs/pickcube_bc/bc_256/best.pt
```


Export the completed visualization and audit its compact evidence:

```bash
.venv/bin/python -m learning.maniskill.placement_report \
  --run learning/runs/pickcube_placement_p0_v2 \
  --output learning/maniskill/evidence/pickcube_placement
.venv/bin/python -m pytest -q tests/learning/test_placement_evidence.py
```
