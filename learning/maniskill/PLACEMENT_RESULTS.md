# P0 findings: observations contain placement; BC reaches and stays near it

The observation-only stop gate exactly reproduced the previous privileged
intervention on all 20 raw/normalized PPO validation episodes. On a separate set
of ten shared validation resets, ordinary BC reached official success in every
episode and ended successfully in eight; neither existing PPO policy succeeded.
This assignment performed no training, checkpoint selection or final tests.

| Ordinary policy on seeds 130000–130009 | Any official success | Final official success | Grasped steps | Within 2.5 cm while grasped |
|---|---:|---:|---:|---:|
| BC, 256 demonstrations | 10/10 | 8/10 | 437/500 | 299/437 |
| Raw PPO final | 0/10 | 0/10 | 356/500 | 4/356 |
| Normalized PPO final | 0/10 | 0/10 | 416/500 | 19/416 |

All policies used identical reset seeds and the unchanged 50-step task. Official
success requires a cube-goal distance of at most 0.025 m and all seven arm-joint
speeds at most 0.2 rad/s. Pooled occupancy uses post-action grasped states as its
denominator; correlated steps are not independent trials. Existing external
expert demonstrations and PPO training compute were not matched, so these ten
validation episodes are a diagnostic comparison rather than an algorithm ranking.

Within 2.5 cm while grasped, mean executed arm-command L2 was 0.096 for BC,
1.989 for raw PPO and 2.290 for normalized PPO. Mean maximum arm speeds in those
same states were 0.050, 0.620 and 0.648 rad/s. BC first entered the grasped
2.5 cm region with 26–43 steps remaining across all ten episodes. Raw PPO entered
in only two episodes, with 9 and 34 steps remaining; normalized PPO entered in
four, with 5, 5, 12 and 9 steps remaining. The traces show differences in approach,
near-goal occupancy and continued commands; they do not isolate a causal fix.

The eight BC episodes ending successfully are also shown as an explicitly
post-hoc conditional subset. They contributed 262/347 grasped states within
2.5 cm, with mean arm-command L2 0.093 and mean maximum arm speed 0.046 rad/s.
That outcome-selected subset is not compared as performance against unfiltered
PPO. All ten episodes per policy remain the primary table above.

The pure stop gate has no environment or privileged-state input. It reads the
ordinary raw observation's cube-to-goal vector at indices 39:42 and compares its
float32 norm with float32(0.025). Each reset and post-step state was logged and
checked numerically against the installed schema and privileged placement flag:
1,020 states in total. Every complete hybrid episode dictionary matched the
prior privileged override exactly, including all commands, geometry and outcomes.
The hand-coded hybrids retained the earlier 2/10 raw-policy and 1/10 normalized-
policy final successes on seeds 80000–80009. They remain hybrids, not PPO alone.

Six native placement videos use the first two predeclared shared seeds. Full
video rerun trajectories matched the compact JSONL traces. Four hybrid videos
were reused only after complete trajectory equality with their prior privileged
native references. The first two hybrid seeds did not activate; the reuse is
explicit in the visualization and manifest. Frozen normalization statistics
remained unchanged, and CPU RNG states were preserved around model loading and
all evaluation. No final test seeds were used.

The final instrumentation run added ordinary-observation vectors for independent
auditing. Its protocol and source hashes were saved before evaluation. Its
summaries exactly matched the earlier instrumentation pass, which is retained
only in ignored runs and is not additional performance evidence.

[Native visualization](evidence/pickcube_placement/index.html) ·
[Protocol and reproduction](PLACEMENT_PROTOCOL.md)
