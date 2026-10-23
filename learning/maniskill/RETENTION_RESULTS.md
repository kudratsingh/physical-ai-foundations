# Frozen-teacher loss substantially improves retention, with one unstable seed

The teacher-anchored condition beats its matched PPO baseline at the fixed budget
in all three training seeds and beats the BC teacher in two. It meets the
predeclared promising criterion. It does **not** solve retention consistently:
seed 2042 finishes below the teacher and loses substantial selected-checkpoint
performance. All results below use the same 50 untouched held-out resets.

| Training seed | Baseline selected | Baseline final | Anchored selected | Anchored final |
|---|---:|---:|---:|---:|
| 42 | 42/50 | 10/50 | 38/50 | 40/50 |
| 1042 | 33/50 | 4/50 | 41/50 | 37/50 |
| 2042 | 36/50 | 5/50 | 40/50 | 25/50 |

BC teacher: **26/50**. Fixed-budget final performance is primary; validation-
selected performance is secondary. Selection is based on 40 separate validation
resets and never on these test outcomes. The selected seed 42 anchored policy is
at 262,144 collected transitions before that boundary's final optimizer update;
the fixed final checkpoint includes that update. Its test result improves from
38 to 40, illustrating why signed selected-minus-final differences can be negative.

Mean final success across three training seeds rises from 6.33/50 to 34/50. Mean
selected-minus-final loss falls from 30.67 to 5.67 successes. Anchored seeds 42 and
1042 exceed both teacher and paired final baseline by at least five successes;
2042 does not. The same 50 reset seeds are reused across training seeds, so these
are three seed-level results, not 150 independent reset trials or a significance
claim. The experiment uses a 262,144-transition budget, not a million-step test.

## Two distinct outcomes

Validation-only selection produces better-than-teacher policies in every seed
for both methods. This is repeated evidence of useful autonomous improvement
following BC initialization. Continued unregularized PPO then loses most of that
performance. The anchored objective retains far more, while still regressing in
one seed. Neither selection nor mean performance should conceal that exception.

Against the teacher on paired reset cases, anchored seed 42 gains 18 teacher-failed
cases while losing 4 teacher successes; seed 1042 gains 14/loses 3; seed 2042 gains 6/
loses 7. Against its own selected checkpoint, final anchored seed 42 loses 1 case
but gains 3; seed 1042 loses 4/gains 0; seed 2042 loses 16/gains 1. Aggregate differences
therefore do not imply that exactly the same episodes remained successful.

## What changes after success

The following means condition on each policy's own states strictly after its
first post-step success, including later states where success has already been
lost. The command that first reaches success is excluded. These subsets differ
between policies; the dashboard and JSON expose their denominators.

| Seed | Later command steps, baseline→anchored | Mean arm-command L2 | Mean post-state max arm speed | Success occupancy | Success→failure events |
|---|---:|---:|---:|---:|---:|
| 42 | 1259→1342 | .364→.130 | .153→.065 rad/s | 59.2→92.5% | 31→5 |
| 1042 | 555→1273 | .402→.145 | .180→.072 rad/s | 62.0→90.7% | 14→6 |
| 2042 | 991→1116 | .399→.102 | .152→.051 rad/s | 47.0→82.7% | 26→13 |

Post-success mean arm-action deviation from BC drops from .395/.456/.395 to
.040/.042/.043. These are normalized action-space distances on the student's
current states, not physical position errors or matched-state causal effects.
The independently exported immediate-pre-success subset answers a different
question: what commands were issued while success was true just before acting.

No lost grasp flag coincides with any of the 95 success-to-failure transitions
across the six final learners. Post-success gripper command magnitudes are near
one in every condition. It would be wrong to attribute improvement to a steadier
gripper: some anchored conditions have larger command changes and measured
finger motion in the broader after-first-success subset. Commands, actual
finger-position deltas and grasp flags are separate observations.

The remaining seed 2042 failure is specific: it reaches success in 38/50 episodes,
ends successful in 25, and all 13 lost-success episodes end task-static but outside
the placement tolerance. Those losses retain the grasp flag. Lower commands and
speed alone do not guarantee holding the cube within 2.5 cm. Task-static means max
seven-arm joint speed ≤.2 rad/s, not zero motion. No contact-force or counterfactual
intervention was performed, so same-step associations do not prove causality.

## Objective, calibration and controls

Both arms use the same bounded critic, original task reward/controller/horizon,
BC actor initialization, fixed normalization, corrected worker streams and PPO
settings. Treatment adds a frozen original-BC mean-action L2 loss on current
PPO training minibatches only. There is no demonstration replay, test-state
supervision, privileged trigger or corrective controller. This is an explicitly
changed training objective; the original task evaluation is unchanged.

The coefficient was selected once before performance training. A separate 8,192-
transition calibration pilot measured PPO and teacher actor-gradient norms on
four on-policy minibatches. The declared median rule suggested 22.7303, then the
predeclared cap fixed lambda 10. The realized median scaled-teacher/PPO ratio is
about .224, below the reference .5; a ceiling hit does not satisfy that target.
Pilot weights/optimizer were discarded; every replicate starts fresh. The
teacher remains frozen outside policy Adam and survives checkpoint reload.

A full pre-update check covers 256 stochastic steps × 4 workers per arm, including
20 timeouts. Initial/post observations, Gaussian samples, executed commands, raw
rewards and pre-reset terminal observations match exactly. Values and timeout-
corrected buffer rewards may differ by design. Actual initial tensors/RNG and
first performance-rollout summaries match again within each of the three pairs.
This repaired experiment is distinct from the historical PR17 reset caveat.

The PPO update derives from pinned SB3 2.9.0 with an upstream source-hash guard
and its MIT license included. Coefficient zero is tested to reproduce standard
PPO weights bitwise. Direct auxiliary gradients touch neither critic nor
exploration log-std; shared clipping/Adam and changed state visitation can still
change their updates indirectly. Gradient decomposition is logged separately
from actual first-step Adam deltas. The independent audit reconstructs combined
actor gradient norms from PPO/teacher norms and cosine, for every rollout.

All six actual final budgets and 18 initial/final/selected checkpoints were
verified before any held-out environment was created. Independent evidence
audits reconstruct all 650 episode outcomes across primary and secondary evaluations and 32,500 state/action records, event
conditioning, calibration arithmetic, training counts and hashes. Fourteen
native files use the first two fixed test seeds for teacher and all final models;
every episode field and passive holding record matches the full test evaluation.
CI checks contracts and saved evidence, not a full repeat of six training runs.

The result supports teacher anchoring as a promising retention method under this
recipe. The unresolved seed 2042 placement drift and the cap-limited coefficient
remain concrete limits. Further strength/reference-policy changes require a new
predeclared experiment and fresh tests; this run does not establish an optimal
coefficient, longer-budget stability, or reliable performance across all seeds.

[Interactive native report](evidence/actor_retention/index.html) ·
[Protocol](RETENTION_PROTOCOL.md) · [Reproduction](RETENTION_README.md)
