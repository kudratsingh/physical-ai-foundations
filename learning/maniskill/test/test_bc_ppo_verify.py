"""Checkpoint guard regression; constructors/inference only, no simulation/train."""

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from learning.maniskill.bc_initialized_ppo import SOURCES, fingerprint, rng_fingerprint
from learning.maniskill.bc_ppo_policy import build
from learning.maniskill.bc_ppo_verify import assert_episode_identity, sha256, verify
from learning.maniskill.behavior_cloning import BehaviorPolicy
from learning.maniskill.experiment import Config, save_json


class ShapesOnly(gym.Env):
    observation_space = spaces.Box(-np.inf, np.inf, (42,), dtype=np.float32)
    action_space = spaces.Box(-1, 1, (8,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        raise AssertionError("Guard regression must not run an environment")

    def step(self, action):
        raise AssertionError("Guard regression must not train or step an environment")


@pytest.fixture
def constructed_run(tmp_path):
    teacher = BehaviorPolicy(obs_dim=42)
    teacher_path = tmp_path / "teacher.pt"
    teacher.save(teacher_path)
    run = tmp_path / "paired"
    run.mkdir()
    config = Config(
        env_id="PickCube-v1",
        num_envs=1,
        total_timesteps=128,
        rollout_steps=16,
        batch_size=16,
        epochs=1,
    )
    metadata = {}
    for arm in ["control", "warm"]:
        model = build(
            ShapesOnly(),
            teacher,
            initialize=arm == "warm",
            rollout_steps=16,
            batch_size=16,
            epochs=1,
        )
        directory = run / arm
        directory.mkdir()
        model.save(directory / "initial")
        metadata[arm] = {
            "actor_fingerprint": fingerprint(model, True),
            "non_actor_fingerprint": fingerprint(model, False),
            "rng_fingerprint": rng_fingerprint(),
            "optimizer_empty": True,
            "checkpoint_sha256": sha256(directory / "initial.zip"),
        }
        save_json(directory / "initialization.json", metadata[arm])
        model.get_env().close()
    sources = Path(__file__).resolve().parents[1]
    save_json(
        run / "protocol.json",
        {
            "config": asdict(config),
            "teacher_sha256": sha256(teacher_path),
            "sources": {name: sha256(sources / name) for name in SOURCES},
            "learning_rate": 3e-5,
            "target_kl": 0.01,
            "critic_feature_clip": 10.0,
            "initial_latent_log_std": -3,
            "test_seeds": [150000, 150001],
        },
    )
    save_json(
        run / "preflight.json",
        {
            "warm_actor_fingerprint": metadata["warm"]["actor_fingerprint"],
            "random_actor_fingerprint": metadata["control"]["actor_fingerprint"],
            "non_actor_fingerprint": metadata["warm"]["non_actor_fingerprint"],
        },
    )
    return run, teacher_path, sources


def test_guard_checks_actual_initials_and_rejects_hash_or_rng_tampering(
    constructed_run,
):
    run, teacher, sources = constructed_run
    result = verify(run, teacher)
    assert result["actual_initial_actor_equals_teacher"]
    assert result["actual_initial_optimizers_empty"]
    assert not result["launch_archive_present"]
    assert result["finals_present"] == {"warm": False, "control": False}
    with pytest.raises(ValueError, match="Missing completed"):
        verify(run, teacher, complete=True)
    archive = run / "launch_sources"
    archive.mkdir()
    for name in SOURCES:
        shutil.copy2(sources / name, archive / name)
    assert verify(run, teacher)["launch_archive_present"]
    path = archive / SOURCES[0]
    original = path.read_bytes()
    path.write_bytes(original + b"\n# modified\n")
    with pytest.raises(ValueError, match="Archived launch source mismatch"):
        verify(run, teacher)
    path.write_bytes(original)
    original = teacher.read_bytes()
    teacher.write_bytes(original + b"changed")
    with pytest.raises(ValueError, match="teacher SHA"):
        verify(run, teacher)
    teacher.write_bytes(original)
    path = run / "warm/initial.zip"
    original = path.read_bytes()
    path.write_bytes(original + b"changed")
    with pytest.raises(ValueError, match="initial checkpoint hash"):
        verify(run, teacher)
    path.write_bytes(original)
    path = run / "control/initialization.json"
    metadata = json.loads(path.read_text())
    metadata["rng_fingerprint"] = "0" * 64
    save_json(path, metadata)
    with pytest.raises(ValueError, match="RNG fingerprints differ"):
        verify(run, teacher)


def test_full_episode_identity_rejects_same_summary_different_physics():
    teacher = [
        {
            "seed": 150000,
            "steps": 50,
            "success_once": True,
            "success_at_end": True,
            "return_": 20,
            "final_arm_qvel": [0.0] * 7,
        }
    ]
    initialized = json.loads(json.dumps(teacher))
    assert assert_episode_identity(teacher, initialized)["full_episode_identity"]
    initialized[0]["final_arm_qvel"][0] = 0.01
    with pytest.raises(ValueError, match="final_arm_qvel"):
        assert_episode_identity(teacher, initialized)


def test_final_budget_uses_actual_loaded_counter_not_sidecar(constructed_run):
    from stable_baselines3 import PPO

    run, teacher, _ = constructed_run
    # Synthetic serialization fixture, not a trained model or performance run.
    model = PPO.load(run / "warm/initial.zip", device="cpu")
    model.num_timesteps = 127
    model.save(run / "warm/final")
    save_json(run / "warm/training.json", {"actual_timesteps": 128, "complete": True})
    with pytest.raises(ValueError, match="final budget mismatch"):
        verify(run, teacher)
    model.num_timesteps = 128
    model.save(run / "warm/final")
    result = verify(run, teacher)
    assert result["finals_present"] == {"warm": True, "control": False}
    assert result["actual_checkpoints"]["warm"]["final"]["recipe"]["timesteps"] == 128
    assert not result["all_final_budgets_verified"]
