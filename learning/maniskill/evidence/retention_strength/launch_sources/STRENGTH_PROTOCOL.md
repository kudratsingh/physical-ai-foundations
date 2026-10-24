# Fixed regularization-strength ablation, declared before training

PR #18 established that frozen-teacher supervision reduces regression under its
recipe. Its coefficient hit the calibration ceiling: the uncapped suggestion was
22.73, while the chosen value was 10. This study compares **lambda 10 versus 25**.
The latter is a fixed hypothesis motivated by that calibration, not an estimated
optimum. There is no new calibration, coefficient search, reward change,
controller change, architecture change, curriculum or evaluation override.

Both conditions start fresh from the same original BC-256 teacher. Preserve the
three PR #18 validation-selected models separately with their selection metadata
and checksums; never initialize from them or use them for auxiliary supervision.
The existing PPO update, teacher mean-action loss over all eight actions, fixed
observation normalization, bounded critic, optimizer and passive instrumentation
are imported unchanged. The teacher stays frozen outside policy Adam.

Use paired training seeds **42, 1042, 2042**, with four corrected workers seeded
base plus rank. Each receives exactly **262,144 transitions**, with rollout 256,
batch 256, up to eight PPO epochs, learning rate 0.00003, gamma .99, GAE .95,
ratio clip .2, entropy coefficient zero, initial Gaussian standard deviation
exp(-3), and local target KL .01. Critic input clips at ±10 and its output is
100 tanh(raw/100). The coefficient is the sole treatment difference. Run all six
conditions regardless of outcomes; no adaptive stopping or budget extensions.

Validation uses the same **40 resets 230000–230039** as PR #18, at zero and every
32,768 transitions. Rank by final success, then any-step success, then return;
zero is eligible. Selection occurs before that boundary's optimizer update,
whereas fixed-budget finals include the final update. Primary outcome is
fixed-budget final success; validation-selected performance is secondary.
Keeping the training and validation seeds isolates coefficient strength on the
known failure. It does not establish performance on new training seeds.

Reserve **50 fresh test resets 250000–250049**, disjoint from PR #18's 240000–240049.
Do not use them in training, validation, teacher supervision, coefficient choice
or stopping decisions. Before creating any test environment, load and verify all
18 actual initial/final/selected checkpoints, final budgets, coefficients, teacher
tensors, source and protocol hashes, paired initial tensors/RNG, and matching
first pre-update rollout summaries. Report the BC teacher once on these new
resets and all six final and six selected policies. The same test list is reused
across training seeds: these are three seed-level comparisons, not 150 independent
reset trials. Preserve episode-level paired gains/losses.

The **predeclared descriptive promising criterion** requires all of:

- Lambda 25 gains at least five final successes out of 50 over lambda 10 in
  training seed 2042, the explicitly targeted PR #18 failure.
- Lambda 25 loses at most two final successes in **each** of seeds 42 and 1042.
- Mean signed selected-minus-final collapse is no larger with lambda 25.

This rule is a practical hypothesis check, not a statistical significance claim.
Report every seed, failures, selection timing and comparisons with the fresh-test
teacher regardless of whether it passes. No outcome is required to match the
historical percentages, and this budget does not test million-transition stability.

Use unchanged passive holding traces: post-first-success and immediately-successful
pre-state actions, arm speed, teacher action drift, gripper commands and measured
finger changes, grasp flags, occupancy and success-loss events. Inspect **every**
ever-successful episode ending static but displaced, using complete episode
trajectories and declared denominators. These observations cannot establish
contact or controller causality. Native videos use the first two test resets
250000 and 250001 for the teacher and every final model, regardless of outcome.
Require every video evaluation field and passive record to match full evaluation.

Freeze launch sources and machine protocol before training. Keep PR #18 evidence
unchanged. Archive complete compact outcomes, compressed full traces, training
telemetry, native videos and provenance separately from implementation commits.
