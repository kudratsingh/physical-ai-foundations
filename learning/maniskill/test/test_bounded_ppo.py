"""Bounds-aware policy probability and real PPO checkpoint contracts."""

import json
import numpy as np
import torch
from stable_baselines3 import PPO
from learning.maniskill.bounded_ppo import build_model, policy_fingerprint, train
from learning.maniskill.experiment import Config, env_thunk
from stable_baselines3.common.vec_env import DummyVecEnv


def test_same_initial_parameters_bounded_samples_and_log_prob_identity():
    torch.set_num_threads(1)
    config = Config(env_id="PickCube-v1", num_envs=1)
    env = DummyVecEnv([env_thunk(config)])
    try:
        raw = build_model(env, config, bounded=False)
        bounded = build_model(env, config)
        assert policy_fingerprint(raw) == policy_fingerprint(bounded)
        obs = torch.as_tensor(np.repeat(env.reset(), 128, axis=0))
        with torch.no_grad():
            actions, _, logp = bounded.policy(obs)
            _, recomputed, entropy = bounded.policy.evaluate_actions(obs, actions)
            torch.testing.assert_close(logp, recomputed, rtol=0, atol=0)
            assert entropy is None
            assert torch.isfinite(logp).all() and (actions.abs() <= 1).all()
            dist = bounded.policy.get_distribution(obs)
            latent_mean = dist.distribution.mean
            deterministic = bounded.policy._predict(obs, deterministic=True)
            torch.testing.assert_close(deterministic, latent_mean.tanh())
            probe = torch.full_like(actions, 0.25)
            gaussian = dist.distribution.log_prob(torch.atanh(probe)).sum(1)
            jacobian = torch.log(1 - probe.square() + dist.epsilon).sum(1)
            torch.testing.assert_close(dist.log_prob(probe), gaussian - jacobian)
    finally:
        env.close()


def test_real_bounded_ppo_trains_and_reloads(tmp_path):
    config = Config(
        env_id="PickCube-v1",
        total_timesteps=128,
        num_envs=1,
        rollout_steps=64,
        batch_size=64,
        epochs=1,
        validation_interval=64,
        validation_seed_start=100000,
        validation_episodes=1,
    )
    output = tmp_path / "bounded"
    train(config, output)
    assert json.loads((output / "training.json").read_text())["actual_timesteps"] == 128
    model = PPO.load(output / "final.zip", device="cpu")
    assert all(torch.isfinite(v).all() for v in model.policy.state_dict().values())
    env = env_thunk(config)()
    try:
        obs, _ = env.reset(seed=110000)
        for _ in range(50):
            action, _ = model.predict(obs, deterministic=True)
            assert (
                action.shape == (8,)
                and np.isfinite(action).all()
                and np.max(abs(action)) <= 1
            )
            obs, _, _, _, _ = env.step(action)
    finally:
        env.close()
