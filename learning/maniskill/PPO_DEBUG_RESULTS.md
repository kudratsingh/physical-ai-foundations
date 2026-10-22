# Smaller PPO updates did not solve placement; a goal fault was recoverable

October 22, 2026. This controlled continuation starts both PPO forks from the
same final-budget 1,048,576-transition checkpoint, including populated Adam
state. Each receives 524,288 additional transitions. The only treatment is
learning rate, 3e-4 versus 1e-4. Both use unchanged task dynamics, reward,
42-state observations, joint-delta control and full 50-step evaluation.

## Fresh held-out comparison

All rows use seeds 70000–70049. “Selected” uses only validation seeds
60000–60009, prioritizing final success, then any-step success, then return.
The fixed-budget final model includes the last optimizer update; step-callback
validation candidates are saved before the subsequent update.

| Policy | Any-step success | Final success | Return mean ± sample SD |
|---|---:|---:|---:|
| Starting PPO, 1,048,576 transitions | 0/50 | 0/50 | 20.551 ± 2.697 |
| Continued PPO, 3e-4, final budget | 0/50 | 0/50 | 21.625 ± 1.461 |
| Continued PPO, 1e-4, final budget | 0/50 | 0/50 | 21.696 ± 2.388 |
| Continued PPO, 3e-4, validation-selected | 0/50 | 0/50 | 21.688 ± 1.420 |
| Continued PPO, 1e-4, validation-selected | 0/50 | 0/50 | 21.880 ± 1.405 |
| BC-256 reference, correct goal | 47/50 · 94% | 34/50 · 68% | 34.471 ± 6.927 |
| Same clone, deliberately incorrect goal | 1/50 · 2% | 0/50 | 20.353 ± 2.126 |
| Same clone, correct goal restored | 47/50 · 94% | 34/50 · 68% | 34.471 ± 6.927 |

The final-success 95% Wilson interval is 0%–7.1% for each zero-success model,
and 54.2%–79.2% for the clean/restored clone. These are episode-sampling
intervals for one model per condition, not training-seed uncertainty. The clone
is an external-expert-data reference, not part of the causal learning-rate
comparison. Its 68% here and the previous 54% used different test seeds; that
change is not a training improvement.

## The optimization intervention worked, but task success did not improve

Across 64 recorded optimization points per fork, mean approximate policy KL
fell from **0.05655 to 0.02042** when learning rate was reduced. Mean clipping
fraction fell from **0.3805 to 0.2051**. Both models therefore made observably
different update sizes. This verifies the intervention rather than assuming a
loaded checkpoint respected the CLI setting; tests also verify that the restored
Adam parameter groups use the new schedule during training.

| Fixed-budget model | Median final 3D goal error | Median maximum arm-joint speed |
|---|---:|---:|
| Starting PPO | 0.0979 m | 0.6131 rad/s |
| Continued PPO, 3e-4 | 0.0876 m | 0.6118 rad/s |
| Continued PPO, 1e-4 | 0.0662 m | 0.6354 rad/s |
| Clean clone | 0.0197 m | 0.0445 rad/s |

The lower-rate model was closer by this median measure, but remained outside
the 0.025 m tolerance and far above the 0.2 rad/s static threshold. A smaller
learning rate is a smaller parameter update, not a lower physical joint-speed
command. Six control episodes and four lower-rate episodes entered the goal
region at some point without satisfying simultaneous stillness. Every PPO
episode grasped and lifted; progress in dense reward did not become complete
behavior.

The control's selected validation candidate came at 524,288 added transitions;
the lower-rate candidate came at 393,216. All validation success rates were zero,
so selection used return as the tie-breaker. Reporting both selected and final
models avoids hiding a failed fixed-budget model behind selection.

This result provides no evidence that reducing learning rate alone solves this
checkpoint's failure within the tested budget. It does not prove learning rate
can never matter, that the task is unsolvable by PPO, or that one fork generally
outperforms the other across seeds. Both finished in about 563.5 seconds while
running concurrently; wall time is machine- and scheduling-dependent. Exact
source/checkpoint hashes, identical initial tensor/Adam fingerprints, settings,
validation history and optimizer traces are committed in the evidence.

## Deliberately break and recover an observation contract

The clone sees goal xyz and a redundant cube-to-goal displacement. The fault
replaces both consistently: goal xyz becomes the current cube xyz, and relative
displacement becomes zero. The actual simulator goal and success scoring stay
unchanged. Changing only one goal encoding would leave another source of goal
information and weaken this test.

With this false goal, final success fell from 34/50 to 0/50, even though every
episode still grasped. The fault model's median final speed was only 0.0642
rad/s, but median goal error was 0.2079 m. It often became still at the wrong
place. This contrasts with PPO's persistent motion and demonstrates why position
and velocity must be inspected separately.

Disabling the fault on the same policy wrapper, without retraining, recovered
all 50 clean episode records exactly: same return, milestones, terminal geometry,
velocities and success outcomes. That is a concrete observation-bug recovery.
It does not mean the robot detected or autonomously repaired a sensor failure;
the experimenter restored the input contract. Thirteen clean/restored episodes
still lost an earlier success, so the clone's holding limitations remain visible.

## Artifacts and next test

[The report](evidence/ppo_debug/index.html) contains six diagnostic plots and eight
native simulator videos. Videos show the first two predeclared test seeds for
both final PPO models, clean BC and the goal-fault clone. Every rendered outcome
matches its corresponding test episode. Core tests independently reconstruct
success from terminal positions and all seven arm velocities, check seed
partitions and controlled settings, and verify exact recovery. Optional learning
tests exercise real PPO optimizer continuation, reload and actual observation
indices against simulator poses.

The roadmap's deliberate break-and-recover exercise is now demonstrated for an
observation-contract fault. **The meaningful-PPO-success gate remains open.**
The next controlled test is observation normalization, with fixed-budget raw and
normalized PPO compared on another fresh seed partition. The current clone
normalizes inputs while PPO uses raw heterogeneous state scales; this difference
motivates a test but does not establish causation. SB3 recommends input
normalization for PPO in its [official practical guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html).
Keep rewards, action semantics, success thresholds and horizon fixed, freeze
normalization statistics during evaluation, and save them alongside each model.
