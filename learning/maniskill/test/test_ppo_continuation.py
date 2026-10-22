"""A real saved Adam optimizer must survive continuation and use the new LR."""

from dataclasses import asdict
import json
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from learning.maniskill.experiment import Config, make_env, save_json
from learning.maniskill.ppo_continuation import continue_training, validation_score


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


def test_real_continuation_preserves_initial_weights_adam_and_uses_override(tmp_path):
    torch.set_num_threads(1)
    task = Config(
        env_id="PickCube-v1",
        num_envs=1,
        rollout_steps=32,
        batch_size=16,
        epochs=1,
        total_timesteps=64,
    )
    env = make_env(task)
    try:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=32,
            batch_size=16,
            n_epochs=1,
            learning_rate=3e-4,
            seed=42,
            device="cpu",
        )
        model.learn(64)
        checkpoint = tmp_path / "source.zip"
        model.save(checkpoint)
        save_json(tmp_path / "config.json", asdict(task))
        original = model.get_parameters()
        assert original["policy.optimizer"]["state"]  # populated Adam moments
    finally:
        env.close()
    for name, rate in [("control", 3e-4), ("slow", 1e-4)]:
        out = tmp_path / name
        continue_training(
            checkpoint,
            out,
            rate,
            steps=64,
            num_envs=1,
            interval=32,
            validation_seeds=[60000],
        )
        initial = torch.load(out / "initial_parameters.pt", weights_only=True)
        assert_nested_equal(initial, original)
        training = json.loads((out / "training.json").read_text())
        assert training["actual_additional_timesteps"] == 64
        assert training["total_timesteps"] == 128
        assert training["final_learning_rate"] == rate
        history = json.loads((out / "validation.json").read_text())
        assert [p["additional_timesteps"] for p in history] == [0, 32, 64]
        restored = PPO.load(out / "final.zip", device="cpu")
        assert np.isfinite(restored.predict(np.zeros(42, dtype=np.float32))[0]).all()


def test_seed_overlap_and_invalid_budget_rejected_before_work(tmp_path):
    with pytest.raises(ValueError, match="disjoint"):
        continue_training(
            tmp_path / "missing.zip", tmp_path / "bad", 1e-4, validation_seeds=[70000]
        )
    with pytest.raises(ValueError, match="Positive"):
        continue_training(tmp_path / "missing.zip", tmp_path / "bad", 0)


def test_selection_prioritizes_final_success():
    assert validation_score(
        dict(success_at_end_rate=0.5, success_once_rate=0.5, return_mean=20)
    ) > validation_score(
        dict(success_at_end_rate=0.4, success_once_rate=1, return_mean=40)
    )


def test_goal_fault_removes_redundant_goal_and_restores_real_policy_input():
    from learning.maniskill.observation_fault import GoalObservationFault, corrupt_goal
    from learning.maniskill.behavior_cloning import BehaviorPolicy

    env = make_env(Config(env_id="PickCube-v1"))
    try:
        observation, _ = env.reset(seed=60000)
        goal = env.unwrapped.goal_site.pose.p[0].numpy().copy()
        cube = env.unwrapped.cube.pose.p[0].numpy().copy()
        assert np.array_equal(observation[26:29], goal)
        assert np.array_equal(observation[29:32], cube)
        assert np.allclose(observation[39:42], goal - cube)
        original = observation.copy()
        broken = corrupt_goal(observation)
        assert np.array_equal(observation, original)
        assert np.array_equal(broken[26:29], cube)
        assert np.array_equal(broken[39:42], np.zeros(3))
        assert np.array_equal(broken[:26], original[:26])
        assert np.array_equal(broken[29:39], original[29:39])
        assert np.array_equal(env.unwrapped.goal_site.pose.p[0].numpy(), goal)
        policy = BehaviorPolicy(obs_dim=42)
        faulty = GoalObservationFault(policy)
        faulty.predict(observation)
        faulty.enabled = False
        assert np.array_equal(
            faulty.predict(observation)[0], policy.predict(observation)[0]
        )
    finally:
        env.close()
