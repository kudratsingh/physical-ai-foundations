"""Normalization integration contracts, using the actual PickCube simulator."""

import json

import numpy as np
import pytest

from learning.maniskill.experiment import Config, env_thunk
from learning.maniskill.normalized_ppo import (
    FrozenPolicy,
    make_normalized_env,
    preserve_rng,
    stats_json,
    train,
)


def test_rewards_terminal_observation_and_frozen_statistics():
    task = Config(env_id="PickCube-v1", num_envs=1, seed=42)
    normalized = make_normalized_env(task)
    raw = env_thunk(task)()
    try:
        obs = normalized.reset()
        raw.reset(seed=42)
        for step in range(50):
            action = np.zeros((1, 8), dtype=np.float32)
            obs, reward, done, info = normalized.step(action)
            raw_obs, raw_reward, _, raw_done, raw_info = raw.step(action[0])
            assert reward[0] == pytest.approx(raw_reward, abs=1e-6)
            assert normalized.get_original_reward()[0] == pytest.approx(
                raw_reward, abs=1e-6
            )
            if step == 49:
                assert done[0] and raw_done
                assert info[0]["episode"]["r"] == pytest.approx(
                    raw_info["episode"]["r"]
                )
                np.testing.assert_allclose(
                    info[0]["terminal_observation"], normalized.normalize_obs(raw_obs)
                )
        normalized.training = False
        before = stats_json(normalized)
        normalized.reset()
        normalized.step(np.zeros((1, 8), dtype=np.float32))
        assert stats_json(normalized) == before
    finally:
        normalized.close()
        raw.close()


def test_tiny_real_training_snapshot_roundtrip_and_frozen_prediction(tmp_path):
    task = Config(
        env_id="PickCube-v1",
        total_timesteps=128,
        num_envs=1,
        rollout_steps=64,
        batch_size=64,
        epochs=1,
        validation_interval=64,
        validation_seed_start=80000,
        validation_episodes=1,
    )
    run = tmp_path / "tiny"
    train(task, run)
    assert json.loads((run / "training.json").read_text())["actual_timesteps"] == 128
    # The real checkpoint constructor and simulator rollout may use all global
    # RNGs. Validation must leave the learner's next random samples unchanged.
    import random
    import torch
    from learning.maniskill.experiment import evaluate

    random.seed(1847)
    np.random.seed(1847)
    torch.manual_seed(1847)
    with preserve_rng():
        expected = (random.random(), np.random.random(), torch.rand(4))
    with preserve_rng():
        loaded = FrozenPolicy.load(run / "final", task)
        try:
            evaluate(task, loaded, [80000])
        finally:
            loaded.close()
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    torch.testing.assert_close(torch.rand(4), expected[2], rtol=0, atol=0)
    first = FrozenPolicy.load(run / "final", task)
    second = FrozenPolicy.load(run / "final", task)
    env = env_thunk(task)()
    try:
        obs, _ = env.reset(seed=90000)
        before = stats_json(first.normalizer)
        a, _ = first.predict(obs)
        b, _ = second.predict(obs)
        np.testing.assert_array_equal(a, b)
        from learning.maniskill.normalization_comparison import policy_contract

        assert policy_contract(first.model, expected_timesteps=128)[
            "observation_shape"
        ] == [42]
        with pytest.raises(ValueError, match="transition budget"):
            policy_contract(first.model)
        assert np.all(np.isfinite(a)) and np.max(np.abs(a)) <= 1
        assert stats_json(first.normalizer) == before
        assert json.loads((run / "final/snapshot.json").read_text())["timesteps"] == 128
        # Loading the wrong statistics file cannot silently corrupt an evaluation.
        path = run / "final/vecnormalize.pkl"
        path.write_bytes((run / "step_64/vecnormalize.pkl").read_bytes())
        with pytest.raises(ValueError, match="checksum mismatch"):
            FrozenPolicy.load(run / "final", task)
    finally:
        first.close()
        second.close()
        env.close()
