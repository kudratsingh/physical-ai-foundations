# Seeded frozen-teacher actor-retention ablation

Two conditions share a BC-initialized actor, bounded critic, corrected worker
reset streams and all PPO hyperparameters. Baseline uses lambda0. Treatment adds
lambda * mean(sum((Gaussian_mean - frozen_BC_action)**2, action_dimensions))
to the usual PPO policy/value/entropy objective. All eight action dimensions,
including the gripper, participate. Supervision uses current PPO training
minibatches only; no demonstration/test/validation states enter the loss.
The BC teacher is frozen, outside the policy optimizer, and checkpointed with
its normalization. Neither critic nor exploration log-std receives auxiliary
gradients. The original reward and evaluation objective remain unchanged.

Before training, compare the complete first stochastic256-step rollout in two
fresh four-worker populations with raw and bounded critics: initial/post-step
observations, raw and executed actions, raw environment rewards, reset masks
and terminal observations must match. Critic values and timeout-corrected buffer
rewards may differ and are excluded. No optimizer updates occur in this check.
Also require initial mean actions to reproduce the teacher, an empty optimizer,
actual matching initial tensors/RNG within each treatment pair, finite stochastic
updates, and bitwise lambda-zero equivalence with the pinned SB3 PPO update.

Calibration is declared before running it: seed91042, four corrected workers,
8,192 baseline transitions, no validation or test evaluation. At the final pilot
weights, use the four256-state minibatches from its last on-policy rollout.
Measure PPO actor gradient norm and unweighted teacher-loss actor gradient norm.
Choose lambda = clip(median(0.5 * PPO_norm / teacher_norm), 0.01,10), using only
finite positive ratios with both gradient norms finite and above1e-12. At least three valid ratios are required.
The pilot weights/optimizer are discarded. Freeze the exact resulting coefficient
and calibration digest before performance training; reuse it for every replicate.
There is no coefficient search using validation or final tests.

Train fresh paired conditions for seeds42,1042,2042, with distinct worker seeds
base_seed+rank. Each arm receives exactly262,144 transitions with four workers,
rollout256, batch256, up to8 epochs, gamma0.99, lambda_GAE0.95, ratio clip0.2,
learning rate0.00003, initial Gaussian std exp(-3), entropy coefficient0 and local
PPO target-KL0.01. Critic features clip at±10 and its output is100*tanh(raw/100)
in both arms; actor features retain fixed BC normalization. Both arms use the
same read-only telemetry. KL early stopping can change optimizer-step counts;
transition budgets match. All six conditions run, irrespective of pilot success.

Validate at step0 and every32,768 transitions on40 fixed reset seeds230000–230039.
Rank checkpoints by final success, then any-step success, then return; selection
is secondary and may select step0. Fixed-budget final success is primary.
Validation runs before that boundary's optimizer update; the final model is
saved after the last update. No early stopping or changing the declared budget.

Only after all six runs finish, evaluate teacher once and each arm's final and
selected policies on the same50 fresh paired resets240000–240049. The reset list
is shared across training seeds; these are not150 independent reset samples.
Report each training seed separately and the mean/dispersion of three seed-level
results, not a falsely pooled binomial interval. Compare final versus selected
collapse, teacher gains/losses, and retained successful behavior. Claim promising
retention only if final anchored success exceeds both teacher and matched final
baseline by at least five episodes in at least two of three seeds and the mean
selected-to-final collapse is lower. Otherwise retain the negative result.

Record post-first-success commands, arm speed, teacher action drift, gripper
changes, grasp state, success occupancy and success-to-failure transitions. Trace
commands act before the recorded post-state; diagnostic teacher inference never
changes executed commands. Keep final-static-but-displaced losses separate from
moving failures. These are observations, not causal contact interventions.
Videos use the first two test seeds for the teacher and both final policies of
every training seed, without selecting outcomes. Auxiliary-loss, gradient
contribution, exploration variance and deterministic-mean drift logs are separate
from optimizer updates and policy quality. Preserve raw Gaussian likelihoods.

Source snapshots and hashes are frozen before the pilot/training. No edits to
launch sources while runs are active. Historical PR17 results retain their reset
caveat; the corrected experiment is a new comparison, not a relabeling.
