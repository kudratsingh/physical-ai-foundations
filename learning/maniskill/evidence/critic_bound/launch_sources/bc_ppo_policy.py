"""Exact BC actor transplant with fixed BC normalization and a PPO critic."""

import numpy as np
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from .behavior_cloning import BehaviorPolicy


class FixedNormalization(BaseFeaturesExtractor):
    def __init__(self, observation_space, mean, std, clip=None):
        super().__init__(observation_space, features_dim=42)
        self.clip = clip
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        if (
            self.mean.shape != (42,)
            or self.std.shape != (42,)
            or (self.std < 1e-3).any()
        ):
            raise ValueError("Expected the fixed 42-component BC normalization")

    def forward(self, observation):
        normalized = (observation - self.mean) / self.std
        return (
            normalized.clamp(-self.clip, self.clip)
            if self.clip is not None
            else normalized
        )


class BCMeanGaussianPolicy(ActorCriticPolicy):
    """Gaussian exploration around BC's bounded deterministic output.

    PPO stores/log-scores the raw Gaussian sample; only the environment command
    is clipped. No inverse tanh of float32-saturated BC actions is required.
    The critic has its own clipped features; the actor remains exactly BC.
    """

    def __init__(self, *args, **kwargs):
        kwargs["share_features_extractor"] = False
        super().__init__(*args, **kwargs)
        self.vf_features_extractor.clip = 10.0

    def _get_action_dist_from_latent(self, latent_pi):
        return self.action_dist.proba_distribution(
            self.action_net(latent_pi).tanh(), self.log_std
        )


def build(
    env, teacher, seed=42, initialize=True, rollout_steps=256, batch_size=256, epochs=8
):
    """Both arms share BC-derived statistics; actor weights are the treatment."""
    model = PPO(
        BCMeanGaussianPolicy,
        env,
        seed=seed,
        device="cpu",
        verbose=1,
        n_steps=rollout_steps,
        batch_size=batch_size,
        n_epochs=epochs,
        learning_rate=3e-5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0,
        target_kl=0.01,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=nn.ReLU,
            log_std_init=-3,
            features_extractor_class=FixedNormalization,
            features_extractor_kwargs=dict(
                mean=teacher.mean.tolist(), std=teacher.std.tolist()
            ),
        ),
    )
    if initialize:
        with torch.no_grad():
            for source, dest in [
                (teacher.net[0], model.policy.mlp_extractor.policy_net[0]),
                (teacher.net[2], model.policy.mlp_extractor.policy_net[2]),
                (teacher.net[4], model.policy.action_net),
            ]:
                dest.weight.copy_(source.weight)
                dest.bias.copy_(source.bias)
    return model


def action_agreement(teacher, model, observations):
    """Check ordinary single-observation inference, not just batched tensors."""
    errors = []
    for obs in np.asarray(observations):
        expected, _ = teacher.predict(obs, deterministic=True)
        actual, _ = model.predict(obs, deterministic=True)
        errors.append(float(np.max(np.abs(expected - actual))))
    return dict(observations=len(errors), max_absolute_action_error=max(errors))


def load_teacher(path):
    teacher = BehaviorPolicy.load(path)
    if len(teacher.mean) != 42:
        raise ValueError("Need the PickCube BC-256 teacher")
    return teacher
