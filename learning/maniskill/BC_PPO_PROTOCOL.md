# BC-initialized PPO: predeclared paired experiment

The treatment copies the existing PickCube BC-256 actor into a compatible PPO
policy. Fixed BC observation normalization, two 256-wide ReLU layers and the
final linear layer are transferred explicitly. The Gaussian mean is the tanh of
the actor output; exploration adds ordinary Gaussian noise around that mean.
PPO scores/stores the raw sample, and the environment command is clipped to its
bounds. There is no inverse tanh or transformed-sample Jacobian in this recipe. This is
initialization, with no corrective controller, stopping postprocessor, continued
BC loss or privileged policy input during training/evaluation.

A genuinely fresh paired control has random actor weights. Both arms share the
BC-derived normalization, ReLU architecture, Gaussian with bounded mean, initial
log standard deviation −3 (std ≈ 0.0498), learning rate 0.00003, fresh identical critic,
fresh Adam optimizer, environment seeds and RNG isolation. The treatment is the
actor tensors. The control is **not** the archived original raw-Gaussian recipe;
both arms have demonstration-derived preprocessing. These recipe choices are
fixed before training, not chosen from final-test performance.

Task: original PickCube-v1, raw 42-component observations, original normalized-dense
reward, pd_joint_delta_pos, original 50-step horizon and success thresholds.
Both arms train exactly 1,048,576 transitions with seed 42/four CPU workers,
rollout 256, batch 256, up to eight epochs, gamma 0.99, lambda 0.95, PPO ratio clip 0.2,
zero entropy coefficient and target-KL 0.01 early-update guard. Only critic
normalized inputs are clipped to ±10; actor normalization stays exactly BC.
The initial critic tensors, preprocessing and optimizer settings match between arms. No critic warmup, curriculum, extra samples, task
resets near the goal, altered damping/reward or online normalization.

The preflight gate requires single-state action error ≤1e-6 on 500 teacher-rollout
observations and matching success/outcomes across ten validation episodes to
within 1e-4 numerical tolerance. The observed errors were exactly zero. Initial
non-actor tensors and empty optimizer state must match. Save each actual initial
model, actor/non-actor fingerprints and pre-learning Python/NumPy/Torch RNG digest;
compare these independent run manifests before interpreting final results.

Validate at step 0 and every 65,536 transitions on 140000–140009. Select by final
success, then any-step success, then return. The fixed-budget final checkpoint is
primary; selected warm checkpoints are secondary and may be step 0. Report that
explicitly if fine-tuning regresses. Validation runs before the corresponding
optimizer update; final models are saved after the last update.

Only after both runs complete, evaluate teacher, initialized PPO, warm final,
matched control final and selected warm policy on 50 fresh paired reset seeds
150000–150049. Native videos use the first two seeds for teacher/warm/control.
Compare initial teacher and initialized policy again on these final resets.
Report final and any-step success, return, physical goal error and joint speed.
Higher return with worse final success is regression, not progress. A promising
improvement requires at least five additional final successes over BOTH teacher
and matched control; it would need independent training-seed replication before
an algorithm-level claim. No test outcome selects the budget or checkpoint.

Training telemetry counts every stochastic on-policy transition separately from
validation: grasp, placement, official success, static arm, reward, goal error,
joint speed, near-boundary executed commands and out-of-bounds sampled commands,
plus grasp-conditioned occupancy within 10 cm/5 cm/2.5 cm and per-episode any/final
success. A deterministic evaluation zero is not evidence of no training successes.

Historical raw/bounded experiments retain their original evidence. The archived
raw run lacked explicit evaluation RNG isolation and a saved initial fingerprint.
Matching new constructors does not prove its historical initialization. The new
paired runs address these reproducibility limitations; they do not retroactively
make the archived comparison perfectly controlled.

An earlier implementation pilot failed numerically before completing training.
It used a squashed Gaussian and an unclipped shared critic feature stream. It is
preserved separately and excluded from performance results. The revised recipe
above was redeclared before a fresh restart, after exact teacher-transfer and
finite stochastic-update checks. No failed-pilot weights or optimizer state are reused.
