# PickCube: demonstrations help; lifting alone does not pass

October 22, 2026. First Stage-1 state-baseline experiment on the Apple M3 Pro.
The task, normalized dense reward, joint-delta control and 50-step horizon are
unchanged. All models use seed 42 for training. Checkpoints were selected on
10 validation seeds before final tests on 50 reserved seeds, 50000–50049.

## Measured comparison

| Policy | Any-step success | Final success | Final 95% Wilson interval | Return mean ± sample SD |
|---|---:|---:|---:|---:|
| BC, 32 demos | 24/50 · 48% | 10/50 · 20% | 11.2%–33.0% | 25.130 ± 10.828 |
| BC, 256 demos | 44/50 · 88% | 27/50 · 54% | 40.4%–67.0% | 33.166 ± 8.640 |
| PPO, 1,048,576 transitions | 0/50 · 0% | 0/50 · 0% | 0%–7.1% | 19.540 ± 3.761 |
| Random | 0/50 · 0% | 0/50 · 0% | 0%–7.1% | 2.689 ± 0.998 |
| Zero commands | 0/50 · 0% | 0/50 · 0% | 0%–7.1% | 3.158 ± 0.489 |

Success requires both 3D cube–goal distance ≤2.5 cm and all seven arm joint
speeds ≤0.2 rad/s. These intervals reflect episode sampling for one trained
model per condition, not variability across training seeds. BC receives external
expert data; the comparison is not compute- or expert-acquisition-cost matched.
The 32/256 comparison changes the amount of data while keeping the validation
set, training budget, architecture and evaluation protocol fixed.

## What failed

| Exclusive outcome | BC-32 | BC-256 | PPO |
|---|---:|---:|---:|
| Final success | 10 | 27 | 0 |
| Succeeded earlier, then lost it | 14 | 17 | 0 |
| Entered goal region, never satisfied stillness | 1 | 0 | 3 |
| Lifted ≥5 cm without task success | 18 | 5 | 46 |
| Grasped without a 5 cm lift | 2 | 1 | 1 |
| Never grasped | 5 | 0 | 0 |

PPO grasped in **50/50** episodes and lifted ≥5 cm in **49/50**. Its improved
validation return therefore measures partial progress, not completion. Three
episodes entered the 3D goal region but never satisfied the simultaneous static
condition. The remaining lift failures did not reach the complete placement
criterion. Those observations locate the bottleneck after grasping; they do not
by themselves prove whether exploration, optimization, action representation or
the training budget caused it.

The larger clone's 44 any-step successes became only 27 final successes. Its
17 lost successes show that reaching a valid pose briefly is easier than
sustaining it through the rest of the episode. The full-horizon protocol exposes
that failure instead of stopping the video and counting a permanent win at the
first successful frame.

## Why offline error is insufficient

BC-32's training action MSE fell from 0.000585 at update 1,000 to 0.000016 at
10,000, while held-out demonstration MSE rose from 0.008205 to 0.013535.
Validation final success peaked at 20% and ended at 10%; the selected checkpoint
was update 2,000, using the declared success/MSE tie-breaking rule. This is
consistent with overfitting the small demonstration subset.

BC-256 was selected at update 7,000: validation success was 100% at any step and
80% at the end. At update 9,000, held-out action MSE was slightly lower
(0.001829 versus 0.001835), but validation final success was only 40%. Small
action-prediction improvements did not translate to stable closed-loop behavior.
That relationship supports rollout-based selection; ten validation episodes
also leave substantial selection uncertainty. No test seeds were used to choose
an update or change the training recipe.

## Data and compute

The pinned official RL archive contained 997 joint-delta episodes. Only **593
of 997 (59.5%)** replayed successfully in CPU physics. The initial 512-episode
probe retained 301, insufficient for the planned 256-training/64-validation
split; the full archive was then replayed before either fit. Failed replays were
excluded, so the training data are a subset that transfers successfully from the
source GPU environment. This filter may change the demonstration distribution;
final evaluation still samples the unchanged task's fresh resets.

Whole-episode permutation seed 2026 selected 64 validation episodes and 256
training episodes; BC-32 uses the first 32 of those 256. Each trajectory has 50
aligned state/action pairs: 1,600 versus 12,800 training pairs, with 3,200
validation pairs. All retained initial-state hashes are distinct. Recorded
expert labels are clipped to the controller's normalized bounds. See
[the protocol](evidence/stage1_pickcube/protocol.json) for the measured clipping
fraction, IDs and source seeds, and
[provenance](evidence/stage1_pickcube/data_provenance.json) for archive/replay
hashes. Only trajectory data were used; external expert checkpoints were not
loaded.

Each clone trained for 10,000 Adam updates. Observed fit/validation time was
24.3 s for BC-32 and 22.0 s for BC-256, excluding demonstration acquisition,
replay and final tests. PPO used four CPU physics workers and 830.4 s for
1,048,576 environment transitions, including periodic validation. Its selected
checkpoint was the final budget point, whose validation return was 20.555 with
zero successes. These wall times are observations on this machine, not
hardware-independent efficiency comparisons. The training config records the
source-file hash at launch; later formatting, CLI validation and PushCube
compatibility edits did not change this PickCube run's settings or dynamics.

## Visualization and next experiment

[Open the report](evidence/stage1_pickcube/index.html) for six learning/evaluation
plots and six native simulator videos. The videos pair BC-256 with PPO on the
first three predeclared test seeds, including failures. Rendering reproduces
every recorded test field exactly. All 250 test episodes have committed terminal
geometry and velocity evidence; the independent tests reconstruct task success
without calling the simulator.

The roadmap's state-BC, fixed-seed evaluation and data-amount-ablation artifacts
now exist for this flagship. **The meaningful-RL-performance gate is still
open.** Next, isolate the PPO placement/stillness bottleneck with one controlled
change and fresh test seeds, then repeat across training seeds. Preserve the
current reward, success definition and horizon as the reference. Vision or a
harder distribution shift follows a stronger state-based baseline; it should
not conceal the present control failure. The deliberate break-and-recover
debugging gate also remains to be demonstrated.
