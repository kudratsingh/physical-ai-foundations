"""Inspect latent Gaussian parameters and executed deterministic distribution actions.

Unlike the original Gaussian-only saturation audit, this helper also handles a
squashed/tanh Gaussian: deterministic output comes from get_actions(), not from
assuming that clip(latent_mean) is the policy command.
"""

import numpy as np
import torch


def inspect_action_distribution(policy, observation):
    """Return latent mean/std, distribution output and bounded policy command."""
    model = getattr(policy, "model", policy)
    normalized = (
        policy.normalizer.normalize_obs(observation)
        if hasattr(policy, "normalizer")
        else observation
    )
    with torch.no_grad():
        tensor, _ = model.policy.obs_to_tensor(normalized)
        distribution = model.policy.get_distribution(tensor)
        latent_mean = distribution.distribution.mean.cpu().numpy()[0]
        latent_std = distribution.distribution.stddev.cpu().numpy()[0]
        distribution_action = (
            distribution.get_actions(deterministic=True).cpu().numpy()[0]
        )
    action, _ = policy.predict(observation, deterministic=True)
    low, high = model.action_space.low, model.action_space.high
    expected = np.clip(distribution_action, low, high)
    if not np.array_equal(action, expected):
        raise ValueError(
            "Deterministic distribution output does not reproduce policy command"
        )
    if not (
        np.isfinite(latent_mean).all()
        and np.isfinite(latent_std).all()
        and (latent_std > 0).all()
        and np.isfinite(action).all()
    ):
        raise ValueError("Need finite Gaussian parameters and action")
    return dict(
        distribution_class=type(distribution).__name__,
        latent_gaussian_mean=latent_mean.tolist(),
        latent_gaussian_std=latent_std.tolist(),
        deterministic_distribution_action=distribution_action.tolist(),
        executed_action=action.tolist(),
        note="Latent Gaussian mean is before any distribution transform; deterministic_distribution_action applies that transform before environment-bound clipping",
    )
