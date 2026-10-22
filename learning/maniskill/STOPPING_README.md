# Placement-triggered stopping diagnostic

This validation-only diagnostic asks whether a privileged intervention after first
placement changes the observed stopping failures. It does **not** repair PPO or
establish an autonomous manipulation result. It neither trains nor changes task
rewards, physics, policy weights, normalization statistics or final test evidence.

The predeclared conditions are raw-final PPO and normalized-final PPO, each run
normally and with a latched override on validation seeds 80000–80009. After the
first placed state at control step t, actions from t+1 through the end contain
zero for the seven arm increments. The gripper keeps the policy's original
command recomputed from that condition's current observation each step. All four
conditions run all 50 steps. A placement at reset step0 would activate action1.
Videos use the first two validation seeds regardless of success or activation.

**Zero delta is not a fixed hold.** The installed Panda `pd_joint_delta_pos`
controller has `use_delta=True` and `use_target=False`. It maps normalized arm
increments ±1 to ±0.1 rad, then sets the target to the **current measured** joint
position plus the increment each control step. Zero therefore recomputes the
current measured pose each step; it does not retain the pose at activation and
does not guarantee a stationary arm or a stationary cube. Real-controller tests
and every active diagnostic step check this target behavior.

Each step logs all eight pre-clip Gaussian means and standard deviations, the
original clipped deterministic command, the executed command, the arm target and
pre-step measured positions, cube-goal distance, seven-arm speeds and official
placement/static/success flags. Activation and post-activation outcomes are
reported separately. Normal rollouts must reproduce the 500 action means from
the earlier validation-only saturation audit exactly. Every native video rerun
must reproduce its complete diagnostic trajectory exactly.

The protocol, timestamp, installed controller-source hashes, code hash and model/
statistics hashes are written **before** any diagnostic rollout. No forced
placement or goal-local reset is part of this diagnostic. If natural placement
never occurs, zero activations are the result.

```bash
.venv/bin/python -m pytest -q learning/maniskill/test/test_stopping_diagnostic.py
.venv/bin/python -m learning.maniskill.stopping_diagnostic declare \
  --output learning/runs/pickcube_stopping_validation \
  --raw-checkpoint ../stage1-pickcube/learning/runs/pickcube_ppo_seed42/final.zip \
  --normalized-snapshot ../pickcube-normalization/learning/runs/pickcube_normalization_seed42/final
.venv/bin/python -m learning.maniskill.stopping_diagnostic run \
  --output learning/runs/pickcube_stopping_validation \
  --raw-checkpoint ../stage1-pickcube/learning/runs/pickcube_ppo_seed42/final.zip \
  --normalized-snapshot ../pickcube-normalization/learning/runs/pickcube_normalization_seed42/final
.venv/bin/python -m learning.maniskill.stopping_report \
  --run learning/runs/pickcube_stopping_validation \
  --output learning/maniskill/evidence/pickcube_stopping
```

Use the isolated Python3.11 learning environment and existing local Vulkan
runtime. The output directory must be new for declaration. The run checks the
predeclared hashes and refuses to overwrite condition outcomes. No final test
seeds are used. [Visual report](evidence/pickcube_stopping/index.html).

For later bounded-action diagnostics, `action_trace.evaluate_trace(config, policy,
seeds, output=None, video=False)` provides synchronized per-step records without
any stopping override. It supports raw and frozen normalized policies, and uses
the distribution's deterministic action method. `latent_gaussian_mean` and
`latent_gaussian_std` are before a possible tanh transform;
`deterministic_distribution_action` is after that transform; `executed_action`
is checked against `policy.predict`. This keeps tanh actions distinct from
clipped Gaussian means. The historical stopping experiment source remains
unchanged and intentionally checks its original ordinary-Gaussian contract.
