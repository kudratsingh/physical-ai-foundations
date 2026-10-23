"""Worker-global streams and real spawned PickCube auto-reset reproducibility."""

import random

import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv

from learning.maniskill import seeded_env
from learning.maniskill.experiment import Config
from learning.maniskill.rng import preserve_rng


def draws(count=4):
    return random.random(), np.random.random(count), torch.rand(count)


def assert_draws_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    torch.testing.assert_close(left[2], right[2], rtol=0, atol=0)


def test_seeds_all_worker_streams_before_and_after_constructor(monkeypatch):
    config = Config(seed=42)
    observed = []
    constructor_draws = 7

    def fake_training_env(task, *, enhanced_determinism=False):
        assert task is config
        assert enhanced_determinism is True

        def construct():
            observed.append(draws(constructor_draws))
            return object()

        return construct

    monkeypatch.setattr(seeded_env, "training_env", fake_training_env)
    with preserve_rng():
        factory = seeded_env.seeded_training_env(config, rank=3)
        factory()
        after_first = draws()
        factory()
        after_second = draws()
        assert_draws_equal(observed[0], observed[1])
        assert_draws_equal(after_first, after_second)
        random.seed(45)
        np.random.seed(45)
        torch.manual_seed(45)
        assert_draws_equal(after_first, draws())
        constructor_draws = 53
        factory()
        assert_draws_equal(after_first, draws())
        seeded_env.seeded_training_env(config, rank=4)()
        assert draws()[0] != after_first[0]


@pytest.mark.parametrize(
    "rank, error",
    [(-1, ValueError), (1.5, TypeError), (True, TypeError), ("1", TypeError)],
)
def test_invalid_rank_is_rejected_before_constructing(rank, error):
    with pytest.raises(error, match="nonnegative integer"):
        seeded_env.seeded_training_env(Config(), rank)


def test_seed_overflow_is_rejected():
    with pytest.raises(ValueError, match="32-bit"):
        seeded_env.seeded_training_env(Config(seed=2**32 - 1), rank=1)


def collect_spawned_population(config):
    env = SubprocVecEnv(
        [seeded_env.seeded_training_env(config, rank) for rank in range(2)],
        start_method="spawn",
    )
    try:
        env.seed(config.seed)
        observations = [env.reset().copy()]
        terminals, done_steps = [], []
        for step in range(1, 111):
            obs, _, dones, infos = env.step(np.zeros((2, 8), dtype=np.float32))
            observations.append(obs.copy())
            if dones.any():
                assert dones.all()
                assert all(info["TimeLimit.truncated"] for info in infos)
                done_steps.append(step)
                terminals.append(
                    np.stack([info["terminal_observation"] for info in infos])
                )
        return np.stack(observations), np.stack(terminals), done_steps
    finally:
        env.close()


@pytest.mark.slow
def test_separate_spawned_pickcube_populations_reproduce_two_auto_reset_horizons():
    """Real CPU physics, no policy or training; reset states include cube+goal."""
    config = Config(env_id="PickCube-v1", num_envs=2, seed=42)
    first, terminal_first, done_first = collect_spawned_population(config)
    second, terminal_second, done_second = collect_spawned_population(config)
    assert first.shape == second.shape == (111, 2, 42)
    assert done_first == done_second == [50, 100]
    # Compare the complete ordinary state and terminal observations, not only
    # returns or reset summaries; this also checks intermediate native physics.
    np.testing.assert_allclose(first, second, rtol=0, atol=1e-7)
    np.testing.assert_allclose(terminal_first, terminal_second, rtol=0, atol=1e-7)
    # Installed PickCube state schema: goal26:29, cube29:32. Compare reset poses
    # across ranks and across episodes to rule out a repeated single scene.
    reset_positions = first[[0, 50, 100], :, 26:32]
    assert all(
        not np.allclose(row[0], row[1], rtol=0, atol=1e-7) for row in reset_positions
    )
    for rank in range(2):
        assert not np.allclose(
            reset_positions[0, rank], reset_positions[1, rank], rtol=0, atol=1e-7
        )
        assert not np.allclose(
            reset_positions[1, rank], reset_positions[2, rank], rtol=0, atol=1e-7
        )
