"""Real controller, policy adapter and frozen-statistics contracts."""

import numpy as np
import torch
from stable_baselines3 import PPO

from learning.maniskill.experiment import Config, make_env
from learning.maniskill.normalized_ppo import (
    FrozenPolicy,
    make_normalized_env,
    stats_json,
)
from learning.maniskill.stopping_diagnostic import (
    controller_contract,
    policy_action,
    rollout,
)


def test_zero_delta_retargets_measured_qpos_each_time_in_real_controller():
    env = make_env(Config(env_id="PickCube-v1"))
    try:
        env.reset(seed=81000)
        arm = controller_contract(env)
        first = arm.qpos.clone()
        arm.set_action(torch.zeros((1, 7)))
        torch.testing.assert_close(arm._target_qpos, first, rtol=0, atol=0)
        # Change a real joint state without any goal-local reset or diagnostic run.
        qpos = env.unwrapped.agent.robot.get_qpos().clone()
        qpos[:, 0] += 0.03
        env.unwrapped.agent.robot.set_qpos(qpos)
        second = arm.qpos.clone()
        assert not torch.equal(first, second)
        arm.set_action(torch.zeros((1, 7)))
        torch.testing.assert_close(arm._target_qpos, second, rtol=0, atol=0)
        assert not torch.equal(first, arm._target_qpos)
    finally:
        env.close()


def test_gaussian_adapter_matches_clipped_prediction_and_freezes_stats():
    config = Config(env_id="PickCube-v1", num_envs=1)
    normalized = make_normalized_env(config)
    model = PPO(
        "MlpPolicy", normalized, n_steps=16, batch_size=16, seed=172, device="cpu"
    )
    normalized.reset()
    policy = FrozenPolicy(model, normalized)
    env = make_env(config)
    try:
        observation, _ = env.reset(seed=81001)
        before = stats_json(normalized)
        original_observation = observation.copy()
        mean, std, action = policy_action(policy, observation)
        np.testing.assert_array_equal(action, np.clip(mean, -1, 1))
        np.testing.assert_array_equal(observation, original_observation)
        assert std.shape == (8,) and np.all(std > 0)
        results = rollout(policy, [81001], override=True)
        assert results["episodes"][0]["steps"] == 50
        for record in results["episodes"][0]["records"]:
            assert record["executed_action"][7] == record["original_action"][7]
        assert stats_json(normalized) == before
        assert results["scope"].startswith("Privileged validation diagnostic")
    finally:
        env.close()
        policy.close()


def test_generic_distribution_reader_distinguishes_tanh_output_from_latent_mean():
    from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
    from learning.maniskill.action_distribution import inspect_action_distribution

    env = make_env(Config(env_id="PickCube-v1"))
    model = PPO("MlpPolicy", env, n_steps=16, batch_size=16, seed=173, device="cpu")
    try:
        # Exercise the actual SB3 distribution path with a nontrivial latent mean.
        model.policy.action_dist = SquashedDiagGaussianDistribution(8)
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.fill_(3)
        obs, _ = env.reset(seed=81002)
        result = inspect_action_distribution(model, obs)
        np.testing.assert_allclose(result["latent_gaussian_mean"], [3] * 8)
        np.testing.assert_allclose(
            result["executed_action"],
            np.tanh(np.full(8, 3, dtype=np.float32)),
            rtol=0,
            atol=1e-7,
        )
        assert all(value < 1 for value in result["executed_action"])
        from learning.maniskill.action_trace import evaluate_trace
        from learning.maniskill.experiment import evaluate

        task = Config(env_id="PickCube-v1")
        traced = evaluate_trace(task, model, [81002])
        reference = evaluate(task, model, [81002])
        assert traced["summary"] == reference["summary"]
        assert len(traced["episodes"][0]["records"]) == 50
        assert all(
            r["distribution_class"] == "SquashedDiagGaussianDistribution"
            for r in traced["episodes"][0]["records"]
        )
    finally:
        env.close()
