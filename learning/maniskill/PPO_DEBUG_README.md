# PickCube: controlled PPO continuation and an observation fault

The first PickCube experiment exposed a grasp/lift policy that did not complete
the 3D placement-and-stillness test. This experiment tests one optimization
hypothesis, then performs the roadmap's deliberate break-and-recover exercise.
See [the measured report](PPO_DEBUG_RESULTS.md) and
[the plots/native videos](evidence/ppo_debug/index.html).

## Controlled learning-rate comparison

Both forks restore the **same final-budget PPO checkpoint**, its actor, critic,
log standard deviation and populated Adam optimizer. They each collect exactly
524,288 new transitions, bringing their total to 1,572,864. The treatment is the
constant learning rate: 3e-4 versus 1e-4. Reward, 42-state observations,
8-dimensional joint-delta actions, PD controller, physical dynamics, seed 42,
network architecture, optimizer settings and 50-step task horizon stay fixed.
The source is `final.zip`, after the last optimizer update, rather than the
previous report's callback-selected `best.zip`. The starting model is therefore
re-evaluated here instead of reusing an earlier success count.

These are model/optimizer continuations in new, identically seeded environments.
SB3 checkpoints do not preserve the original simulator states or every random
state; this is not a bit-exact resume of the original process. The rollout
buffer is newly collected. Each fork has a separate log directory. Canonical
fingerprints of every tensor and optimizer field verify identical initial
state; a real training/reload test checks that Adam survives and the new LR
schedule actually updates its parameter groups.

Validation uses new seeds **60000–60009**, initially and every 131,072 additional
transitions. Selection prioritizes **final success**, then any-step success,
then mean return. It includes the unchanged initial model at step zero. Step
callbacks evaluate after environment transitions, before the next PPO update;
`final.zip` includes the last update. Both fixed-budget final models and
validation-selected models are tested and reported, so preserving a good
initial checkpoint cannot be presented as an RL improvement.

The reserved final test set is **70000–70049** for every condition. It is
disjoint from validation, earlier experiments' test sets and the BC reference's
source demonstration seeds. Deterministic actions and full 50-step episodes
follow the previous task contract: 3D distance ≤0.025 m and all seven arm speeds
≤0.2 rad/s simultaneously. Native video uses the first two test seeds,
predeclared in code, for both final-budget PPO models and clean/faulty BC.
The clone is a task reference with external expert knowledge, not part of the
causal learning-rate comparison. One common trained checkpoint and one
continuation seed do not establish a general ranking across training seeds.

## Deliberate observation fault and recovery

A PickCube state observation contains the goal twice: its xyz position and the
cube-to-goal displacement. Corrupting just one representation would leave the
other as an information leak. The fault wrapper replaces goal xyz by the current
cube xyz and zeroes the relative displacement. The false observations therefore
consistently say “the cube is already at the goal.” Robot state, tool/cube poses,
actual simulator goal, physics and success scoring stay unchanged.

The clean clone, faulty clone and restored clone use the same checkpoint. After
testing the fault, disabling it on the same wrapper restores the original
observations without retraining. All 50 restored episode records must exactly
match the clean reference. This diagnoses an input-contract bug; it does not
claim the learned policy is robust to unknown sensor faults or has learned to
repair them itself. Tests verify the pinned observation indices against real
simulator poses and check that neither input arrays nor simulator goals are
mutated.

## Reproduce

Use the learning Python 3.11 environment and local runtime described in
[the PickCube walkthrough](PICKCUBE_README.md). Current local work is isolated in
`.worktrees/pickcube-ppo-debug`, branch `feat/pickcube-ppo-debug`, stacked on
`feat/stage1-pickcube`. The default evaluation paths point to the existing
sibling PickCube checkpoint. In a fresh checkout, first produce those checkpoints
with the original walkthrough and pass their paths explicitly.

```bash
# Existing local checkpoint:
PPO_SOURCE=../stage1-pickcube/learning/runs/pickcube_ppo_seed42/final.zip
BC_SOURCE=../stage1-pickcube/learning/runs/pickcube_bc/bc_256/best.pt

.venv/bin/python -m learning.maniskill.ppo_continuation \
  --checkpoint "$PPO_SOURCE" --output learning/runs/continuation_control \
  --learning-rate 0.0003
.venv/bin/python -m learning.maniskill.ppo_continuation \
  --checkpoint "$PPO_SOURCE" --output learning/runs/continuation_slow \
  --learning-rate 0.0001

.venv/bin/python -m learning.maniskill.ppo_debug evaluate \
  --checkpoint "$PPO_SOURCE" --bc "$BC_SOURCE"
.venv/bin/python -m learning.maniskill.ppo_debug publish
.venv/bin/python -m http.server 8770 --bind 127.0.0.1 \
  --directory learning/maniskill/evidence/ppo_debug
```

Continuation and evaluation output directories must be new. Large checkpoints,
initial-parameter snapshots and raw traces stay under ignored `learning/runs/`.
Committed evidence includes hashes, protocol, model selection, optimizer curves,
per-episode terminal geometry/velocities, plots and videos. Short logging probes
were stopped before the definitive paired runs; their budgets are not included
in the completed-fork comparison.

```bash
.venv/bin/python -m pytest -q learning/maniskill/test
# Root MuJoCo venv: full foundations tests, including independent evidence audits.
/path/to/root/.venv/bin/python -m pytest -q
```

The Linux ManiSkill CI job executes the real optimizer-resume and fault-contract
tests. Core CI recomputes metrics and success from terminal geometry/velocity,
audits controlled settings and seed partitions, checks selection against its
validation history, and verifies native videos against the declared test prefix.
It also checks exact recovery after restoring the goal observations.

Reference: [Stable-Baselines3 PPO API](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html).
Installed package versions, actual schedules and runtime behavior are recorded
and tested rather than inferred from the latest documentation alone.
