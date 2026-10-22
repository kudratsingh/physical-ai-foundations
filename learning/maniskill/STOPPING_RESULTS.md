# Privileged stopping diagnostic: three naturally placed episodes recovered

Neither ordinary PPO checkpoint succeeded on validation seeds 80000–80009.
A privileged override activated in two raw-policy episodes and one normalized-
policy episode after natural placement. All three satisfied the unchanged
physical task success condition on the next control step and at episode end.
These are intervention outcomes, not repaired autonomous PPO results.

| Condition | Ever placed | Any official success | Final official success | Activated episodes |
|---|---:|---:|---:|---:|
| Raw final, ordinary | 2/10 | 0/10 | 0/10 | 0 |
| Raw final, privileged zero-arm override | 2/10 | 2/10 | 2/10 | 2 |
| Normalized final, ordinary | 1/10 | 0/10 | 0/10 | 0 |
| Normalized final, privileged zero-arm override | 1/10 | 1/10 | 1/10 | 1 |

Raw seed 80002 first placed at step 28; the override began at action 29 and first
success occurred at step 29. Raw seed 80009 placed at 23, override/success at 24.
Normalized seed 80008 placed at 16, override/success at 17. Their maximum arm speeds
at placement were 0.659, 0.697 and 0.652 rad/s, above the 0.2 rad/s static threshold.
Final speeds under intervention were 0.062, 0.043 and 0.044 rad/s, with final cube-goal
distances 0.0194, 0.0236 and 0.0147 m. All 50 steps were retained in every episode.

The intervention zeroed only the seven arm increments. It preserved the policy's
current gripper command each step, used no forced placement and changed no
physics, damping or reward. Zero delta re-sets the joint target to each step's
current measured position; it does not preserve one fixed activation pose.
The actual controller targets matched measured pre-step positions on all 83 active
steps across the three episodes. The ordinary policy had continued supplying
its own nonzero arm increments. This provides direct evidence that changing
post-placement commands can recover these specific cases. It does not establish
that a bounded distribution, a learned stopping rule or generalization will work.

The protocol was timestamped and hashed before all diagnostic rollouts. Frozen
normalization statistics remained unchanged. Both ordinary conditions reproduced
all 500 pre-clip Gaussian means from the earlier 80000 validation audit exactly.
The full trajectories before intervention were identical between paired normal
and override conditions. All eight native video reruns reproduced every logged
record exactly. The predeclared first two video seeds never reached placement
for either policy, so their overrides never activated and paired videos are
identical. The report's trace examples show the first naturally activated seed
per policy, selected by activation order rather than success.

No training, checkpoint selection or final test evaluation was performed. The
sample contains only three naturally placed episodes and relies on privileged
task state for triggering. These findings support a targeted stopping-command
hypothesis; they do not replace the separate bounds-aware PPO experiment.

[Visual report](evidence/pickcube_stopping/index.html) ·
[Protocol and implementation](STOPPING_README.md)
