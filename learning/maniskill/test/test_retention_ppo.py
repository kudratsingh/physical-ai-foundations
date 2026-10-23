"""Objective equivalence, frozen teacher isolation, gradients and checkpoint reload."""

import copy

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from learning.maniskill.bc_ppo_policy import FixedNormalization
from learning.maniskill.behavior_cloning import BehaviorPolicy
from learning.maniskill.critic_diagnostics import DiagnosticPPO
from learning.maniskill.critic_policy import BoundedCriticPolicy
from learning.maniskill.retention_ppo import RetentionPPO, teacher_loss


class FakeTask(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (42,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, (8,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.zeros(42, dtype=np.float32), {}

    def step(self, a):
        self.t += 1
        return (
            np.full(42, 0.01 * self.t, dtype=np.float32),
            float(a.mean()),
            False,
            self.t == 10,
            {},
        )


def build(cls, teacher=None, coefficient=0.0):
    kwargs = {
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
        "activation_fn": torch.nn.ReLU,
        "log_std_init": -3,
        "features_extractor_class": FixedNormalization,
        "features_extractor_kwargs": {"mean": [0.0] * 42, "std": [1.0] * 42},
    }
    extra = (
        {"teacher": teacher, "retention_coefficient": coefficient}
        if cls is RetentionPPO
        else {}
    )
    return cls(
        BoundedCriticPolicy,
        DummyVecEnv([FakeTask]),
        seed=7,
        n_steps=32,
        batch_size=16,
        n_epochs=2,
        device="cpu",
        policy_kwargs=kwargs,
        **extra,
    )


def test_zero_coefficient_is_bitwise_standard_ppo_and_teacher_is_not_trained(tmp_path):
    torch.set_num_threads(1)
    teacher = BehaviorPolicy(obs_dim=42)
    original = copy.deepcopy(teacher.state_dict())
    base = build(DiagnosticPPO)
    initial = copy.deepcopy(base.policy.state_dict())
    base.learn(128)
    anchored = build(RetentionPPO, teacher, 0.0)
    for k, v in initial.items():
        torch.testing.assert_close(v, anchored.policy.state_dict()[k], rtol=0, atol=0)
    anchored.learn(128)
    for k, v in base.policy.state_dict().items():
        torch.testing.assert_close(v, anchored.policy.state_dict()[k], rtol=0, atol=0)
    for k, v in original.items():
        torch.testing.assert_close(v, anchored.teacher.state_dict()[k], rtol=0, atol=0)
    assert not any(
        p.requires_grad or p.grad is not None for p in anchored.teacher.parameters()
    )
    optimizer = {
        id(p)
        for group in anchored.policy.optimizer.param_groups
        for p in group["params"]
    }
    assert not optimizer & {id(p) for p in anchored.teacher.parameters()}
    anchored.save(tmp_path / "model")
    loaded = RetentionPPO.load(tmp_path / "model")
    for k, v in original.items():
        torch.testing.assert_close(v, loaded.teacher.state_dict()[k], rtol=0, atol=0)
    assert not any(p.requires_grad for p in loaded.teacher.parameters())
    obs = np.random.default_rng(7).normal(size=(20, 42)).astype(np.float32)
    np.testing.assert_array_equal(
        loaded.predict(obs, deterministic=True)[0],
        anchored.predict(obs, deterministic=True)[0],
    )
    assert len(anchored.retention_history) == 4


def test_teacher_loss_is_l2_sum_over_actions_and_cannot_update_critic(tmp_path):
    teacher = BehaviorPolicy(obs_dim=42)
    model = build(RetentionPPO, teacher, 1.0)
    obs = torch.randn(12, 42)
    mean = model.policy.get_distribution(obs).distribution.mean
    with torch.no_grad():
        target = model.teacher(obs)
    actual = teacher_loss(model.policy, model.teacher, obs)
    torch.testing.assert_close(
        actual, ((mean - target) ** 2).sum(-1).mean(), rtol=0, atol=0
    )
    model.policy.optimizer.zero_grad()
    actual.backward()
    assert model.policy.action_net.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.policy.value_net.parameters())
    assert all(
        p.grad is None for p in model.policy.mlp_extractor.value_net.parameters()
    )
    assert model.policy.log_std.grad is None
    assert all(p.grad is None for p in model.teacher.parameters())
    model.learn(64)
    assert all(torch.isfinite(v).all() for v in model.policy.state_dict().values())
    assert model.retention_history[-1]["coefficient"] == 1.0

    model.save(tmp_path / "positive")
    loaded = RetentionPPO.load(tmp_path / "positive")
    assert loaded.retention_coefficient == 1.0
    assert not any(p.requires_grad for p in loaded.teacher.parameters())
    for k, v in model.teacher.state_dict().items():
        torch.testing.assert_close(v, loaded.teacher.state_dict()[k], rtol=0, atol=0)
