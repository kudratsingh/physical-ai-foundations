# Stronger global anchoring helps the targeted seed but sacrifices other gains

The fixed lambda-25 treatment fails the predeclared promising criterion. It gains
five final successes in the explicitly targeted training seed 2042, but loses
four and three successes in the other seeds. The rule allowed losses of at most
two in each of those seeds. Mean signed selection-to-final collapse improves,
while mean fixed-budget success falls slightly. These are mixed results, not a
reason to declare lambda 25 the better general recipe.

All models receive 262,144 transitions. The following results use the same **new
50 held-out resets 250000–250049** within each pair. The BC teacher scores **31/50**
on this list. PR #18 used a different test list; its percentages should not be
compared directly with these percentages as though the reset cases were identical.

| Training seed | Lambda 10 final | Lambda 25 final | Final change | Lambda 10 selected | Lambda 25 selected |
|---|---:|---:|---:|---:|---:|
| 42 | 40/50 | 36/50 | −4 | 40/50 | 41/50 |
| 1042 | 40/50 | 37/50 | −3 | 45/50 | 35/50 |
| 2042 (declared target) | 32/50 | 37/50 | +5 | 43/50 | 37/50 |

The primary mean across training seeds is **74.7% versus 73.3%**, with seed-level
ranges 64–80% versus 72–74%. The same reset list is shared across seeds; this is not
150 independent reset trials, a significance test, or a general reliability claim.
All six final policies exceed the teacher on this particular list, by differing
margins. The three training seeds, teacher and short budget limit generalization.

Mean signed selected-minus-final collapse falls from **5.33 to 1.00 successes**.
This does not imply uniform retention: lambda 25 seed 42 loses five selected
successes, while seed 1042 finishes two successes above its selected checkpoint.
The lambda-25 selected models are from 229,376, 32,768 and 32,768 transitions;
the lambda-10 selections are from 262,144, 196,608 and 196,608. Validation selection
precedes that boundary's optimizer update; final models include the last update.
Selection uses final success, then any-step success, then return on 40 validation
resets, not held-out results. A validation winner need not beat the final model on
test resets. Both primary finals and secondary selected outcomes are preserved.

## Paired episode outcomes

| Training seed | New successes with lambda 25 | Lambda-10 successes lost | Retained successes |
|---|---:|---:|---:|
| 42 | 2 | 6 | 34 |
| 1042 | 1 | 4 | 36 |
| 2042 | 6 | 1 | 31 |

Thus the targeted gain is not six entirely additional successes with none lost.
The report exports every gained/lost reset ID, each strength's comparison with
the fresh-test teacher, and selected-to-final gained/lost identities separately.

## Precision and holding remain distinct from smaller teacher drift

| Seed | After-first-success steps, 10→25 | Mean arm command L2 | Mean post-state max arm speed, rad/s | Teacher arm deviation L2 | Success occupancy | Success-loss transitions |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 1449→1367 | .1375→.1375 | .0653→.0646 | .0416→.0266 | 91.9→86.9% | 9→11 |
| 1042 | 1416→1344 | .1359→.1073 | .0693→.0562 | .0401→.0257 | 91.0→90.8% | 9→10 |
| 2042 | 1381→1303 | .1052→.1398 | .0551→.0649 | .0403→.0217 | 81.8→89.3% | 16→10 |

Each subset uses its own policy's states strictly after first post-step success,
including later failed states. The command first reaching success is excluded.
The policies visit different states, so these are conditional associations rather
than matched-state causal comparisons. Immediately-successful pre-state summaries
and their counts are exported separately. Teacher deviation is normalized action
space distance, not a physical position error.

Higher strength reduces mean post-success teacher deviation in all three seeds,
and also reduces the final rollout's scored-minibatch teacher loss in every seed.
That does not guarantee better precision placement. Target seed 2042 improves
holding even though its mean post-success command magnitude and arm speed rise.
A universal account based only on reducing motion is not supported by these data.

Seed 2042 reaches success in 47/50 episodes at lambda 10 and 46/50 at lambda 25,
then finishes successful in 32 and 37. Successful-then-displaced final failures
fall from 15 to 9; 15 and 8 respectively finish task-static. Across all six final
learners there are **59** successful-then-displaced final episodes, **55** static.
All 59 complete cube-to-goal distance trajectories appear in the report, including
moving final cases. None of **65** success-loss transitions coincides with a lost
grasp flag. That does not rule out earlier gripper changes, contact forces or
residual motion. No contact-force analysis or corrective intervention was run.
Task-static means maximum arm joint speed at most .2 rad/s, not zero movement.

## Controls and reproducibility

The coefficient is the only intended training treatment. Both use the same PR #18
PPO implementation, frozen BC teacher, normalization, bounded critic, original
reward/controller/50-step horizon, optimizer settings, corrected reset streams
and validation procedure. No new pilot, coefficient search, replay supervision,
reward shaping, reference controller or curriculum was introduced. Lambda 25 was
fixed before training because PR #18's uncapped calibration suggested 22.73;
this study does not establish an optimal coefficient.

Each new lambda-10 final policy reproduces its PR #18 reference **bit for bit in
all policy tensors**, including the critic and exploration parameters. All three
validation histories and selected-checkpoint metadata also reproduce exactly.
The ZIP container hashes differ because metadata differs; canonical tensor hashes
match. This check uses no new evaluation. PR #18's three selected checkpoints and
metadata remain intact and are also copied locally with checksums; they are never
used for initialization, auxiliary targets or evaluation in this comparison.

Before creating a held-out environment, the guard loaded all 18 initial, final
and selected models, verified all six actual budgets, coefficients and recipes,
checked frozen sources/protocol and teacher tensors, and matched initial policy
and RNG fingerprints plus first pre-update rollout summaries within every pair.
The original full four-worker stochastic-rollout repair test remains covered by
PR #18's unchanged suite. No historical evidence is relabeled.

The committed report contains all **650 episode outcomes**, **32,500 passive
state/action records**, training telemetry, source snapshots and hashes, paired
reset IDs, all selected displacement trajectories and **14 native videos**. Videos
use the first two declared test resets regardless of outcome; every episode field
and passive record matches full evaluation. Full checkpoints remain local ignored
artifacts. CI checks contracts, short simulator runs and independent evidence
reconstruction; it does not repeat the full six-model training study.

## What this changes

Keep PR #18 and lambda 10 as the reference. Global lambda 25 provides evidence of
a retention-versus-improvement tradeoff under this recipe, not uniform recovery.
Another increase in the global penalty is less compelling after this result.
The next independent protocol should test a placement-aware objective or a
reference constraint that protects successful manipulation states, with fresh
tests and a clear account of which states supply supervision. Do not retrospectively
choose a coefficient or checkpoint using this test set. Long-horizon stability,
different BC teachers and new training seeds remain untested.

[Native report](evidence/retention_strength/index.html) ·
[Protocol](STRENGTH_PROTOCOL.md) · [Reproduction](STRENGTH_README.md)
