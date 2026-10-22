"""BC/PPO critic clipping and raw Gaussian likelihoods across SB3 call paths."""

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from learning.maniskill.bc_ppo_policy import build
from learning.maniskill.behavior_cloning import BehaviorPolicy
from learning.maniskill.rng import preserve_rng


class ObservationActionContract(gym.Env):
    """Only shapes are needed; this regression never executes an environment."""

    observation_space = spaces.Box(-np.inf, np.inf, (42,), dtype=np.float32)
    action_space = spaces.Box(-1, 1, (8,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(42, dtype=np.float32), {}

    def step(self, action):
        raise AssertionError("Critic call-path regression must not run simulation")


def test_critic_clips_all_value_paths_actor_stays_exact_and_raw_logprobs_match():
    torch.set_num_threads(1)
    teacher = BehaviorPolicy(mean=np.zeros(42), std=np.full(42, 0.25))
    with torch.no_grad():
        teacher.net[4].weight.zero_()
        teacher.net[4].bias.fill_(20.0)  # Float32 tanh is exactly saturated at +1.
    env = ObservationActionContract()
    model = build(env, teacher)
    policy = model.policy
    assert not policy.share_features_extractor
    assert policy.pi_features_extractor is not policy.vf_features_extractor
    assert policy.pi_features_extractor.clip is None
    assert policy.vf_features_extractor.clip == 10.0
    observation = (
        policy.pi_features_extractor.mean + policy.pi_features_extractor.std * 100.0
    )[None, :].repeat(2, 1)
    extractor_outputs, critic_inputs = [], []
    extractor_hook = policy.vf_features_extractor.register_forward_hook(
        lambda module, args, output: extractor_outputs.append(output.detach().clone())
    )
    input_hook = policy.mlp_extractor.value_net.register_forward_pre_hook(
        lambda module, args: critic_inputs.append(args[0].detach().clone())
    )
    try:
        with preserve_rng(), torch.no_grad():
            torch.manual_seed(123)
            action, value, old_logprob = policy(observation, deterministic=False)
            direct_value = policy.predict_values(observation)
            evaluated_value, replay_logprob, _ = policy.evaluate_actions(
                observation, action
            )
            assert len(extractor_outputs) == len(critic_inputs) == 3
            for features, received in zip(
                extractor_outputs, critic_inputs, strict=True
            ):
                torch.testing.assert_close(
                    features, torch.full_like(features, 10.0), rtol=0, atol=0
                )
                torch.testing.assert_close(features, received, rtol=0, atol=0)
            actor_features = policy.pi_features_extractor(observation)
            torch.testing.assert_close(
                actor_features, torch.full_like(actor_features, 100.0), rtol=0, atol=0
            )
            distribution = policy.get_distribution(observation).distribution
            torch.testing.assert_close(
                distribution.mean, torch.ones_like(distribution.mean), rtol=0, atol=0
            )
            torch.testing.assert_close(value, direct_value, rtol=0, atol=0)
            torch.testing.assert_close(value, evaluated_value, rtol=0, atol=0)
            torch.testing.assert_close(old_logprob, replay_logprob, rtol=0, atol=0)
            assert torch.isfinite(old_logprob).all() and torch.isfinite(value).all()
            assert (action > 1).any()  # Raw Gaussian samples can exceed command bounds.
            _, incorrectly_clipped_logprob, _ = policy.evaluate_actions(
                observation, action.clamp(-1, 1)
            )
            assert not torch.allclose(old_logprob, incorrectly_clipped_logprob)
    finally:
        extractor_hook.remove()
        input_hook.remove()
        model.get_env().close()
