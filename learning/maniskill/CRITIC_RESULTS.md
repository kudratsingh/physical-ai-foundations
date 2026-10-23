# Critic bounds control numerical growth but do not preserve the learned skill

At the fixed 262,144-transition budget, bounded-critic PPO achieved 10/50 final
successes versus 7/50 for the raw critic and 28/50 for the BC teacher. The bound
controlled critic/target growth and preserved larger measured actor updates,
but both final policies regressed below their initial teacher. Three extra
successes on one training seed do not establish a performance improvement.

| Policy on fresh seeds 190000–190049 | Final success | Any-step success | Mean return ± SD |
|---|---:|---:|---:|
| BC teacher | 28/50 | 43/50 | 32.14 ± 8.11 |
| Raw critic, final | 7/50 | 33/50 | 28.14 ± 8.20 |
| Bounded critic, final | 10/50 | 41/50 | 28.47 ± 10.46 |
| Raw critic, selected at 98,304 (secondary) | 41/50 | 47/50 | 35.12 ± 8.08 |
| Bounded critic, selected at 65,536 (secondary) | 37/50 | 46/50 | 34.71 ± 8.88 |

All policies use the original reward, joint-delta controller, 50-step horizon,
observation schema and success rule. Selection uses only ten fixed validation
resets, ranked final success, then any-step success, then return. The 41/50 and
37/50 selected results cannot replace the negative fixed-budget retention result.
Different seeds and budget make these distinct from PR16's million-step results.

## Confirmed mechanism and limits

An independent offline audit first evaluated earlier saved critics on 408 fixed
states from the existing BC-validation demonstration split. The old warm-final
critic's median prediction was 104,178 and maximum 589,283; 92.2% exceeded 100.
These are off-policy calibration probes, not recovered historical training data.
The installed normalized reward lies in [0,1]; with gamma 0.99 a true continuing
discounted value cannot exceed 100. Input feature clipping alone did not constrain
learned value predictions. Checkpoints exclude rollout buffers, so historical
on-policy gradient norms cannot be reconstructed from saved Adam moments.

The new runs directly measured on-policy values, returns, timeout corrections and
actual first optimizer steps. The final collected rollout showed:

| Measurement | Raw critic | Bounded critic |
|---|---:|---:|
| Median current value | 358.81 | 73.01 |
| Median GAE lambda-return target | 486.55 | 79.71 |
| Median pre-reset timeout value | 997.39 | 88.70 |
| Critic gradient L2, before clipping | 46,830.19 | 211.01 |
| Actor gradient L2, after clipping | 0.000321 | 0.093139 |
| Actual first Adam actor parameter delta L2 | 0.0000326 | 0.001160 |

The gradient and parameter-delta rows describe only the **first executed
optimizer step of the final rollout**, not all minibatches or an entire update.
Both networks share one global gradient clip and one Adam optimizer. The raw
critic dominated the shared norm while its actual actor update shrank sharply.
Adam's adaptation means gradient rescaling alone is not enough evidence; the
actual parameter deltas were measured separately. All 256 rollout records per
arm are retained. The bounded critic/GAE targets stayed within ±100, while the
raw timeout estimates and targets exceeded the physically valid value range.

No double timeout bootstrap or cross-episode GAE error was found. SB3 adds
`gamma * V(pre-reset terminal observation)` once and masks across the reset.
The new audit independently reconstructs that correction from raw versus buffer
reward sums for every rollout. Changing to zero-terminal targets would change
the objective; it is not included as a bug fix. The bound clips neither raw
rewards nor return targets, and it does not alter the task's success definition.

This narrows the diagnosis: runaway values and shared clipping are an observable
numerical problem, and the bound suppresses that pattern. Keeping predictions
inside the valid range does not establish an accurate critic. Preventing the
pattern was insufficient to retain precise placement/holding. The result does not prove
critic instability was the sole or original cause of behavioral regression.

## Reproducibility and checks

Both fresh arms copied exactly the same teacher actor and canonical underlying
critic tensors, preprocessing and initial exploration. Actual initial checkpoint
hashes, canonical parameter fingerprints, empty optimizer states and saved
pre-learning RNG digests match. All ten initial teacher/raw/bounded validation
episode dictionaries are identical. A separate 4,096-transition stochastic smoke
was finite; those weights and optimizer were discarded. The initial dry-preflight
directory likewise contains implementation checks rather than training results.

The reward/timeout contract is unchanged, and the treatment is only
`V = 100*tanh(raw_V/100)`. The soft transform has slope 1 near zero and permits
negative predictions while containing the theoretical nonnegative value range.
Actual loaded models and recipes pass independent source, initialization,
selection and budget guards. Frozen launch source snapshots are committed.

A regression test requires bitwise identical training weights with and without
both gradient hooks and timeout probes on a synthetic environment. A source
audit found a specific reset-seeding gap: SB3 queues seeds for the
initial resets, then auto-resets workers without seed arguments. ManiSkill's
default non-enhanced-determinism path initializes new cube/goal states from
worker Torch RNG that this factory never explicitly seeded. The parent RNG
manifest excludes those worker states. Each first 256-step rollout includes five
such resets per worker, explaining possible differences before optimization.
First-rollout reward summaries differ. These runs therefore match recipe,
initial tensors and parent RNG, not fully controlled training reset streams.
This limits causal arm comparisons; the small final-success difference is not
treated as a fix. A subsequent commit repairs worker seeding for future runs;
the archived evidence is not retroactively changed. The repaired factory also
uses ManiSkill's `enhanced_determinism=True`: global seeding alone repeated the
first scene on the first automatic reset. The integration regression checks
reproducible complete trajectories and distinct scenes across resets/ranks.

Compact outcomes independently reconstruct 3D goal distance, seven-arm speeds,
full-horizon success, count budgets, clipping arithmetic and timeout corrections.
Six native videos use the first two predeclared test seeds regardless of outcome;
all non-trajectory episode fields match the test evidence and video hashes match.
CI checks the implementation and committed evidence rather than retraining both
262,144-transition policies. One training seed and a short budget cannot establish
million-transition stability or training-seed robustness.

Next, test an explicit actor-retention condition (such as an auxiliary BC or
reference-policy KL term), keeping the original evaluation task and reporting
any new training objective separately. Preserve the useful selected policies
and replicate a promising procedure across training seeds. Do not spend another
long budget assuming the critic bound alone solves retention.

[Visual report](evidence/critic_bound/index.html) · [Protocol](CRITIC_PROTOCOL.md) ·
[Commands](CRITIC_README.md)
