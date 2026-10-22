# PickCube: observation normalization did not fix placement and holding

At an equal fixed budget of 1,048,576 transitions, neither the original raw-state
PPO checkpoint nor the observation-normalized treatment succeeded on any of the
50 fresh paired test episodes. The small increase in return does not establish
an improvement in task completion.

| Checkpoint | Any official success | Final official success | Raw return, mean ± sample SD |
|---|---:|---:|---:|
| Raw PPO final | 0/50 | 0/50 | 19.732 ± 4.001 |
| Observation-normalized PPO final | 0/50 | 0/50 | 20.431 ± 2.514 |
| Selected normalized, secondary | 0/50 | 0/50 | 20.464 ± 2.469 |

The primary comparison is final versus final: both models were trained from
scratch with seed 42, four CPU workers, the same PPO settings, and the unchanged
50-step PickCube task. The only treatment was online observation normalization
with clipping at ±10. Rewards remained in original task units, and there were no
extra warmup transitions. Fresh test seeds were 90000–90049. Each policy's
50-episode any-success Wilson interval is 0–7.14%; this describes these fixed
policies under sampled resets, not variation over training seeds.

The treatment did learn to approach and grasp. Its mutually exclusive failure
counts were five episodes that reached placement without ever satisfying full
success, 35 that lifted at least 5 cm without reaching placement/full success,
and ten that grasped without a 5 cm lift. No episode was in the never-grasped
category. Raw PPO had seven placement-without-success, 41 lift-only, one
grasp-only and one never-grasped episode. Placement alone is insufficient:
official success also requires a static robot. These coarse categories cannot
identify the cause of failure by themselves.

A separate post-training audit on validation seeds 80000–80009 found at least one
arm-joint Gaussian mean outside the normalized action bounds on **94.8%** of raw
policy steps and **100%** of normalized policy steps. Across all seven arm
coordinates, the rates were 45.49% and 62.17%. Clipping those means exactly
reproduced every deterministic command. Gripper closure is excluded from these
arm rates. These are command-bound clips, not PPO probability-ratio clips.
[The saved means and checkpoint hashes](evidence/pickcube_normalization/action_saturation.json)
make this diagnostic auditable. Frequent saturation motivates an action-distribution
ablation; it does not by itself prove why either policy failed.

All eight normalized validation checkpoints also had zero success on seeds
80000–80009. Mean validation return rose from 7.095 at 131,072 transitions to
21.011 at 917,504. That 917,504 checkpoint was selected by the predeclared
validation rule and reported only as secondary, because the original raw run
used different validation seeds. Selection never used the final tests.

The definitive treatment completed exactly 1,048,576 transitions in 826.065 s;
the existing raw run took 830.439 s. These runs happened at different times on
the same Mac, so elapsed time is descriptive. Normalization statistics counted
1,048,580.0001 observations: one reset per worker, each training transition,
and the numerical initialization count. Validation and tests contributed none.

The actual loaded final models each report 1,048,576 transitions, 42 observation
components and eight actions. Their model/statistics pairs have saved checksums.
Evaluation used frozen copies of training statistics. A regression also verifies
that loading and evaluating a validation snapshot leaves Python, NumPy and Torch
training RNG streams unchanged. The interrupted pilot that exposed the RNG-loading
issue is excluded from these results.

[The visual report](evidence/pickcube_normalization/index.html) contains native
videos from the first three predeclared test seeds, learning curves, paired goal
distances and complete provenance. All six video rollouts reproduced every saved
non-trajectory test field exactly. Independent tests recomputed all 150 final
success outcomes from 3D positions and seven arm-joint velocities.

This result supports keeping observation normalization as an evaluated option,
not treating it as a demonstrated fix. The next useful work is a targeted
placement/holding diagnostic and a predeclared controller/action-interface
ablation, before adding vision or claiming robust manipulation. Multiple training
seeds will be needed to estimate how stable any eventual improvement is.
