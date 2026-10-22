# Tanh-Gaussian PPO did not fix placement or stopping

The predeclared final-checkpoint comparison produced **0/50 any-step and final
successes for both policies** on fresh paired test seeds 110000–110049. Changing
the action distribution alone did not solve the upstream PickCube task at the
fixed 1,048,576-transition budget. The result did not meet the predeclared
replication trigger, so additional training seeds were not launched.

| Final checkpoint | Any / final successes | Return: mean ± sample SD | Median goal error | Median maximum arm speed | Static at end |
|---|---:|---:|---:|---:|---:|
| Original raw Gaussian | 0 / 0 of 50 | 19.722 ± 4.030 | 9.59 cm | 0.621 rad/s | 0/50 |
| Tanh-Gaussian treatment | 0 / 0 of 50 | 18.574 ± 3.024 | 19.42 cm | 0.616 rad/s | 0/50 |
| Selected bounded, secondary | 0 / 0 of 50 | 18.599 ± 2.783 | 39.28 cm | 0.625 rad/s | 0/50 |

Complete success requires goal distance ≤2.5 cm and maximum seven-arm-joint speed
≤0.2 rad/s. Final outcomes were independently recomputed from recorded geometry
and velocities. A 0/50 success rate has a 95% Wilson interval of 0–7.14% for these
fixed policies under sampled resets; it does not measure training-seed variation.
The treatment had two placement-without-success episodes, 44 lift-only and four
grasp-only. The raw control had four placement-without-success, 44 lift-only and
two never-grasped episodes. Neither final policy ended any episode with a static arm.

## What changed, and what did not

The treatment used SB3's tanh-transformed Gaussian during training and evaluation,
including the change-of-variables log probability. Both constructors had exactly
matching initial actor/critic tensors and latent log standard deviations. The
actual loaded checkpoints were checked for equal budget, spaces and PPO settings:
network sizes, optimizer/epsilon, learning rate, rollout/batch/epoch settings,
gamma, GAE lambda, ratio clip, entropy/value coefficients and gradient clipping.
There was no observation normalization, controller/reward change, extra warmup,
demonstration data, stopping override or horizon extension. The treatment completed
in 809.114 seconds. Runtime is descriptive because the original control was trained
earlier on the same Mac.

All eight validation checkpoints also had zero successes. Validation seeds
100000–100009 were disjoint from the final tests. Validation preserved training
RNG streams. Fixed-budget final weights are primary; the selected bounded policy
is secondary because the original raw run used different validation seeds.
Callbacks validate before the corresponding optimizer update; final weights are
saved after the last update. The launch source is unchanged and its hash is saved.

## What the synchronized commands show

Across 500 validation steps, an executed arm command had magnitude ≥0.99 on 94.0%
of raw-policy steps and 54.4% of bounded-policy steps. Median maximum absolute arm
command was 1.000 versus 0.992. These are still large joint increments; a bounded
output is not automatically a fine-control output. The bounded distribution does
not need an external hard clip, but tanh itself can approach its boundary.

There were 32 raw-policy actions with pre-action goal distance ≤5 cm, all containing
a near-boundary arm command. The bounded policy had **no** such actions on these
validation seeds, so no near-goal bounded conditional rate is reported. The command
audit logs each latent Gaussian mean/std, transformed distribution output, executed
command, resulting distance and arm speed. Latent exploration std is not the
standard deviation of executed actions after tanh.

This weakens the idea that replacing hard clipping alone will solve the failure.
It does not show that clipping was harmless or establish the population-level root
cause. The [separate stopping diagnostic](STOPPING_RESULTS.md) gives a narrower,
positive result: changing post-placement commands recovered all three naturally
placed validation cases, under a privileged trigger. Those intervention successes
remain distinct from the zero autonomous successes reported here. The Stage-1 PPO
success gate remains open; training budget and learning the placement/holding
behavior remain unresolved.

[Visual comparison](evidence/pickcube_bounded/index.html) ·
[Run instructions](BOUNDED_ACTION_README.md) ·
[Predeclared protocol](BOUNDED_ACTION_PROTOCOL.md)

## Later reproducibility review

The archived raw trainer did not explicitly preserve global RNG state during
validation, and its initial policy fingerprint was not saved. Matching newly
constructed initial policies is not independent proof of the archived initial
state. The negative task outcomes remain valid, but a perfectly controlled
single-change causal interpretation is not warranted. Future evaluation now
uses a common RNG-isolation guard; a real-simulator regression shows identical
learner weights with and without intervening evaluation. The new paired
[BC-initialized experiment](BC_PPO_PROTOCOL.md) saves both actual initial models
and RNG fingerprints.
