# BC-initialized PPO: early improvement, then loss of holding behavior

The validation-selected checkpoint achieved 31/50 final successes on fresh test
resets, compared with 23/50 for its BC teacher. Continuing training to the declared
1,048,576-transition budget regressed to 0/50. A matched random-actor control also
ended at 0/50. This is evidence of a useful early checkpoint and unstable
fine-tuning, not a reliable fixed-budget PPO improvement.

| Policy on seeds 150000–150049 | Final success | Any-step success | Mean return ± SD | Median final goal error | Median maximum arm speed |
|---|---:|---:|---:|---:|---:|
| BC-256 teacher | 23/50 | 42/50 | 31.37 ± 8.30 | 2.58 cm | 0.053 rad/s |
| Saved PPO initialization | 23/50 | 42/50 | 31.37 ± 8.30 | 2.58 cm | 0.053 rad/s |
| BC-initialized PPO, fixed-budget final | 0/50 | 12/50 | 21.78 ± 6.84 | 8.13 cm | 0.284 rad/s |
| Matched random actor, fixed-budget final | 0/50 | 0/50 | 2.27 ± 1.23 | 19.97 cm | 0.425 rad/s |
| BC-initialized PPO, selected checkpoint (secondary) | 31/50 | 40/50 | 30.88 ± 13.03 | 1.98 cm | 0.065 rad/s |

The table uses the original PickCube reward, 50-step horizon, joint-delta
controller and success definition: goal distance ≤2.5 cm and maximum arm speed
≤0.2 rad/s. No stopping postprocessor, privileged trigger or corrective controller
is used. These fresh seeds differ from earlier baseline reports; the teacher's
23/50 here is a new paired evaluation, not a revision of earlier 27/50 evidence.

## Transfer and experimental control

The actor transplant reproduces BC exactly on 500 teacher validation observations
and ten complete validation episodes. The actual saved PPO initialization also
matches every teacher episode field on all 50 fresh test resets, including return,
geometry, velocity and success. Actual initial checkpoint hashes, actor tensors,
non-actor tensors, empty optimizers and saved pre-learning RNG digests were checked
against independently saved manifests. Actual loaded final checkpoints each have
exactly 1,048,576 transitions. Source hashes and optional launch snapshots match.

Both arms use BC-derived fixed observation normalization, two 256-wide ReLU
layers, an ordinary Gaussian around a tanh-bounded mean, initial std exp(−3),
learning rate 0.00003 and a target-KL 0.01 guard. Only actor weights are the paired
treatment. The random actor shares demonstration-derived preprocessing and is
**not** the archived original PPO recipe. The actor keeps unmodified BC features;
only the separate critic inputs are clipped to ±10. Tests inspect the actual
critic MLP input during rollout, likelihood evaluation and value/bootstrap calls.

## Training and selection

Validation final success started at 8/10, rose to 10/10 at 65,536 transitions,
fell to 7/10 at 131,072 and 4/10 at 196,608, then remained 0/10 from 262,144
onward. The predeclared final-success-first selection retained step 65,536.
Its held-out 31/50 is secondary; it does not replace the negative primary result
at the fixed budget. The selected checkpoint has eight more final successes than
the teacher on these resets, but one training seed and ten validation seeds do
not establish a robust improvement or justify choosing a new budget from tests.

Every stochastic training transition was instrumented independently of validation:

| Training totals | BC-initialized | Matched random |
|---|---:|---:|
| Transitions | 1,048,576 | 1,048,576 |
| Grasped steps | 816,448 | 1 |
| Placed steps | 190,128 | 2,600 |
| Official successful steps | 168,357 | 1,573 |
| Episodes with any success | 11,770/20,968 | 52/20,968 |
| Episodes ending successfully | 2,578/20,968 | 11/20,968 |

Training steps are correlated; these are exposure/occupancy counts, not independent
performance trials. The task's official success flag does not require a recorded
grasp, so rare random-control successful states do not establish learned picking.
Episode counts exclude four unfinished episodes at the rollout boundary.
Deterministic validation zeros do not mean stochastic training never succeeded.

Critic loss grew sharply in both arms while reported policy KL approached zero.
The last console values were approximately 2.6e10 and 9.75e10, respectively.
These metrics describe the preceding optimizer update, not a separately measured
post-final loss. All saved weights and metrics remain finite. Clipped critic
features bound inputs, not learned values or bootstrapped targets; no bypassed
clipping path was found. The logs establish instability but do not isolate its
cause. The warm final arm was static in 17/50 episodes, yet none ended placed;
its placement and holding behavior both regressed.

An earlier implementation pilot using a squashed Gaussian with an unclipped
shared critic stream encountered NaN distribution parameters. It was stopped,
archived and excluded before final testing. The revised recipe was redeclared
before fresh runs, after exact-transfer and finite stochastic-update checks.
Neither failed-pilot weights nor optimizer state were reused. Multiple numerical
changes were made, so the pilot alone does not identify the cause of its NaNs.

## Evidence and next experiment

The [visual report](evidence/bc_ppo_paired/index.html) includes training/validation
curves, optimizer traces, all five test outcomes and six native simulator videos.
Videos use the first two predeclared test seeds without outcome selection; all
non-trajectory fields match their test episodes. Compact JSONL outcomes support
independent geometric success reconstruction, training count checks and video
hash verification. Full checkpoint/log files remain local ignored artifacts.
CI tests the contracts and saved evidence, not an independent full training run.

Before another significant run, diagnose value-target and critic-gradient growth
and test an explicitly separate stabilization condition. Instrument value/return
scales and actor/critic gradient norms, then consider critic warmup or separate
critic optimization while keeping the original task evaluation unchanged.
Preserve and compare the useful selected policy. Replicate any promising
validation-selected procedure across training seeds before claiming a learning
improvement. Reward/curriculum changes remain later, separate conditions.

[Protocol](BC_PPO_PROTOCOL.md) · [Commands](BC_PPO_README.md) ·
[Observation-only stopping and placement findings](PLACEMENT_RESULTS.md)
