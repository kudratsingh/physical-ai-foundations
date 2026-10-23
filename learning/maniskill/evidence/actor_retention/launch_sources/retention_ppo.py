"""Teacher-anchored PPO objective, derived from pinned SB3 2.9.0 (MIT).

The base update below retains SB3's algorithm/order, adding only the frozen
teacher action loss and its metrics. Lambda-zero equivalence is tested bitwise.
DiagnosticPPO wraps this update through multiple inheritance, without modifying
historical source files or intercepting/replacing gradients.
"""

import copy
import hashlib
import inspect

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance

from .behavior_cloning import BehaviorPolicy
from .critic_diagnostics import DiagnosticPPO, parameter_group
from .experiment import save_json
from .rng import preserve_rng

SB3_TRAIN_SHA256 = "7c8d09940123242f0ac401c5ec2cc858b8bf3339f571c222382f048ffd8306ae"


def teacher_loss(policy, teacher, observations):
    mean = policy.get_distribution(observations).distribution.mean
    with th.no_grad():
        target = teacher(observations)
    return (mean - target).square().sum(dim=-1).mean()


def gradient_norm(gradients):
    return (
        sum(
            float(g.detach().double().square().sum())
            for g in gradients
            if g is not None
        )
        ** 0.5
    )


def retention_probe(model, batch):
    """First minibatch gradient decomposition, outside actual-backward hooks."""
    _values, logp, _entropy = model.policy.evaluate_actions(
        batch.observations, batch.actions
    )
    advantages = batch.advantages
    if model.normalize_advantage and len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    ratio = th.exp(logp - batch.old_log_prob)
    clip = model.clip_range(model._current_progress_remaining)
    pg = -th.min(
        advantages * ratio, advantages * th.clamp(ratio, 1 - clip, 1 + clip)
    ).mean()
    aux = teacher_loss(model.policy, model.teacher, batch.observations)
    actor = [
        p for n, p in model.policy.named_parameters() if parameter_group(n) == "actor"
    ]
    gp = th.autograd.grad(pg, actor, retain_graph=True, allow_unused=True)
    ga = th.autograd.grad(aux, actor, allow_unused=True)
    pn, an = gradient_norm(gp), gradient_norm(ga)
    dot = sum(
        float((p.detach().double() * a.detach().double()).sum())
        for p, a in zip(gp, ga, strict=True)
        if p is not None and a is not None
    )
    with th.no_grad():
        mean = model.policy.get_distribution(batch.observations).distribution.mean
        target = model.teacher(batch.observations)
        difference = (mean - target).square()
        return {
            "ppo_actor_gradient_norm": pn,
            "teacher_actor_gradient_norm": an,
            "scaled_teacher_gradient_norm": model.retention_coefficient * an,
            "gradient_cosine": dot / (pn * an) if pn and an else None,
            "teacher_action_loss": float(difference.sum(-1).mean()),
            "arm_teacher_drift_mse_sum": float(difference[:, :7].sum(-1).mean()),
            "gripper_teacher_drift_mse": float(difference[:, 7].mean()),
            "exploration_variance": th.exp(2 * model.policy.log_std).cpu().tolist(),
            "deterministic_mean_action_variance": mean.var(dim=0, unbiased=False)
            .cpu()
            .tolist(),
        }


class TeacherObjectivePPO(PPO):
    def __init__(self, *args, teacher=None, retention_coefficient=0.0, **kwargs):
        if (
            hashlib.sha256(inspect.getsource(PPO.train).encode()).hexdigest()
            != SB3_TRAIN_SHA256
        ):
            raise RuntimeError(
                "SB3 PPO.train changed; review the derived retention update"
            )
        if not np.isfinite(retention_coefficient) or retention_coefficient < 0:
            raise ValueError("Retention coefficient must be finite and nonnegative")
        self.teacher = (
            copy.deepcopy(teacher)
            if teacher is not None
            else BehaviorPolicy(obs_dim=42)
        )
        self.teacher.eval().requires_grad_(False)
        self.retention_coefficient = float(retention_coefficient)
        super().__init__(*args, **kwargs)

    def _get_torch_save_params(self):
        states, variables = super()._get_torch_save_params()
        return [*states, "teacher"], variables

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        auxiliary_losses = []
        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                # Frozen BC supervision on this on-policy training minibatch only.
                # Lambda-zero leaves the original PPO objective/graph unchanged.
                if self.retention_coefficient:
                    auxiliary = teacher_loss(
                        self.policy, self.teacher, rollout_data.observations
                    )
                else:
                    with th.no_grad():
                        auxiliary = teacher_loss(
                            self.policy, self.teacher, rollout_data.observations
                        )
                auxiliary_losses.append(float(auxiliary.detach()))
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )
                if self.retention_coefficient:
                    loss = loss + self.retention_coefficient * auxiliary

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # Logs
        self.logger.record("train/auxiliary_loss_mean", np.mean(auxiliary_losses))
        self.logger.record(
            "train/scaled_auxiliary_loss_mean",
            self.retention_coefficient * np.mean(auxiliary_losses),
        )
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


class RetentionPPO(DiagnosticPPO, TeacherObjectivePPO):
    """Read-only diagnostics wrap the explicitly extended PPO objective."""

    def train(self):
        # get() reshapes the buffer once; preserve its shuffle RNG for the real update.
        with preserve_rng():
            self.policy.set_training_mode(True)
            batch = next(self.rollout_buffer.get(self.batch_size))
            probe = retention_probe(self, batch)
        super().train()
        if not hasattr(self, "retention_history"):
            self.retention_history = []
        point = {
            "timesteps": self.num_timesteps,
            "coefficient": self.retention_coefficient,
            "scope": "First training minibatch before update, same shuffled batch as first optimizer attempt. Teacher/PPO actor contributions measured before combined global clipping.",
            "probe": probe,
            "optimizer_steps": self.diagnostics[-1]["optimizer_steps"],
            "auxiliary_loss_mean_scored_minibatches": self.logger.name_to_value[
                "train/auxiliary_loss_mean"
            ],
            "scaled_auxiliary_loss_mean_scored_minibatches": self.logger.name_to_value[
                "train/scaled_auxiliary_loss_mean"
            ],
        }
        self.retention_history.append(point)
        if getattr(self, "retention_path", None):
            save_json(self.retention_path, {"updates": self.retention_history})
