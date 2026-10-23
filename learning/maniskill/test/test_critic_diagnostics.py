"""Telemetry must preserve the learner, and value bounds cover every path."""

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from learning.maniskill.critic_diagnostics import DiagnosticPPO, RolloutDiagnostics
from learning.maniskill.critic_policy import BoundedCriticPolicy


class FakeTask(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (42,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, (8,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(42, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        return (
            np.full(42, self.steps * 0.01, dtype=np.float32),
            float(np.mean(action)),
            False,
            self.steps >= 10,
            {},
        )


def test_actual_gradient_hooks_do_not_change_training_weights():
    torch.set_num_threads(1)
    outcomes = []
    for cls in [PPO, DiagnosticPPO]:
        m = cls(
            "MlpPolicy",
            FakeTask(),
            seed=123,
            n_steps=32,
            batch_size=16,
            n_epochs=2,
            device="cpu",
        )
        m.learn(128, callback=RolloutDiagnostics() if cls is DiagnosticPPO else None)
        outcomes.append(m.policy.state_dict())
        if cls is DiagnosticPPO:
            assert len(m.diagnostics) == 4
            for p in m.diagnostics:
                assert p["optimizer_steps"] == 4
                assert p["global_norm_before_clip"] > 0
                assert (
                    sum(v * v for v in p["gradient_norm_after_clip"].values()) ** 0.5
                    <= 0.500001
                )
                assert p["first_step_parameter_delta_l2"]["actor"] > 0
                assert p["first_step_parameter_delta_l2"]["critic"] > 0
    for key in outcomes[0]:
        torch.testing.assert_close(outcomes[0][key], outcomes[1][key], rtol=0, atol=0)


def test_value_bound_applies_in_every_path_and_reload(tmp_path):
    kwargs = {
        "features_extractor_kwargs": {"mean": [0.0] * 42, "std": [1.0] * 42},
        "net_arch": {"pi": [16], "vf": [16]},
    }
    # BCMeanGaussianPolicy uses fixed normalization from its explicit extractor.
    from learning.maniskill.bc_ppo_policy import FixedNormalization

    kwargs["features_extractor_class"] = FixedNormalization
    m = DiagnosticPPO(
        BoundedCriticPolicy,
        FakeTask(),
        seed=8,
        n_steps=32,
        batch_size=16,
        policy_kwargs=kwargs,
    )
    with torch.no_grad():
        m.policy.value_net[0].bias.fill_(1e8)
    obs = torch.full((4, 42), 100.0)
    action, values, _logp = m.policy(obs)
    _, _, _ = m.policy.evaluate_actions(obs, action)
    for value in [
        values,
        m.policy.predict_values(obs),
        m.policy.evaluate_actions(obs, action)[0],
    ]:
        assert torch.isfinite(value).all() and value.max() <= 100
    ids = {id(p) for g in m.policy.optimizer.param_groups for p in g["params"]}
    assert all(id(p) in ids for p in m.policy.value_net.parameters())
    m.save(tmp_path / "bounded")
    loaded = DiagnosticPPO.load(tmp_path / "bounded")
    torch.testing.assert_close(
        values, loaded.policy.predict_values(obs), rtol=0, atol=0
    )


def test_paired_initial_tensors_and_rng_differ_only_by_value_function():
    from learning.maniskill.bc_initialized_ppo import rng_fingerprint
    from learning.maniskill.behavior_cloning import BehaviorPolicy
    from learning.maniskill.critic_experiment import build, canonical_fingerprint

    teacher = BehaviorPolicy(obs_dim=42)
    raw = build(FakeTask(), teacher, "raw")
    raw_rng = rng_fingerprint()
    bounded = build(FakeTask(), teacher, "bounded")
    assert rng_fingerprint() == raw_rng
    assert canonical_fingerprint(raw) == canonical_fingerprint(bounded)
    assert not raw.policy.optimizer.state and not bounded.policy.optimizer.state
    obs = np.random.default_rng(4).normal(size=(64, 42)).astype(np.float32)
    expected, _ = teacher.predict(obs)
    for model in [raw, bounded]:
        actual, _ = model.predict(obs, deterministic=True)
        np.testing.assert_array_equal(actual, expected)
