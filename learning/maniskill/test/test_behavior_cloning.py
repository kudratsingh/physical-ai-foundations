"""Data separation, action semantics, and a real BC checkpoint round trip."""

import json
import h5py
import numpy as np
import pytest
import torch
from learning.maniskill.bc_data import load_episodes, split_episodes, arrays
from learning.maniskill.behavior_cloning import BehaviorPolicy, fit


def dataset(tmp_path, malformed=False):
    p = tmp_path / "demos.h5"
    metadata = dict(
        env_info=dict(
            env_id="PushCube-v1",
            env_kwargs=dict(
                obs_mode="state",
                control_mode="pd_joint_delta_pos",
                sim_backend="physx_cpu",
            ),
        ),
        episodes=[],
    )
    with h5py.File(p, "w") as f:
        for index in range(6):
            g = f.create_group(f"traj_{index}")
            observation = np.zeros((5, 35), dtype=np.float32)
            observation[:, 0] = index
            observation[:, 1] = np.arange(5)
            g["obs"] = observation[:-1] if malformed else observation
            g["actions"] = np.full((4, 8), 2 if index == 0 else 0.2, dtype=np.float32)
            metadata["episodes"].append(
                dict(episode_id=index, episode_seed=40000 + index, success=True)
            )
    p.with_suffix(".json").write_text(json.dumps(metadata))
    return p


def test_alignment_clipping_and_episode_disjointness(tmp_path):
    episodes = load_episodes(dataset(tmp_path))
    assert episodes[0]["obs"].shape == (4, 35)
    assert episodes[0]["obs"][-1, 1] == 3  # terminal observation 4 has no label
    assert np.max(episodes[0]["actions"]) == 1
    assert episodes[0]["raw_clipped_scalars"] == 32
    train, validation = split_episodes(episodes, 4, 2)
    assert {e["id"] for e in train}.isdisjoint({e["id"] for e in validation})
    assert [e["id"] for e in split_episodes(episodes, 4, 2)[0]] == [
        e["id"] for e in train
    ]
    assert arrays(train)[0].shape == (16, 35)


def test_bad_alignment_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="T\\+1"):
        load_episodes(dataset(tmp_path, malformed=True))


def test_normalization_excludes_validation_and_checkpoint_matches(tmp_path):
    torch.set_num_threads(1)
    episodes = load_episodes(dataset(tmp_path))
    train, validation = episodes[:4], episodes[4:]
    policy = fit(
        train,
        validation,
        tmp_path / "fit",
        updates=20,
        interval=10,
        rollout_validation=False,
    )
    assert policy.mean[0].item() == pytest.approx(1.5)
    assert policy.mean[0].item() != pytest.approx(arrays(episodes)[0][:, 0].mean())
    obs = train[0]["obs"][0]
    restored = BehaviorPolicy.load(tmp_path / "fit/best.pt")
    assert np.array_equal(policy.predict(obs)[0], restored.predict(obs)[0])
    assert np.all(np.abs(restored.predict(np.ones(35, dtype=np.float32) * 1e6)[0]) <= 1)


def test_trained_bc_checkpoint_runs_a_complete_physical_episode(tmp_path):
    from learning.maniskill.experiment import Config, evaluate

    episodes = load_episodes(dataset(tmp_path))
    policy = fit(
        episodes[:4],
        episodes[4:],
        tmp_path / "policy",
        updates=2,
        interval=2,
        rollout_validation=False,
    )
    result = evaluate(Config(), policy, [20000])
    assert result["policy"] == "Behavior cloning deterministic"
    assert result["episodes"][0]["steps"] == 50
    assert np.isfinite(result["episodes"][0]["return"])


def test_test_seeds_cannot_overlap_rollout_validation(tmp_path):
    from learning.maniskill.behavior_cloning import run

    checkpoint = tmp_path / "checkpoint.zip"
    checkpoint.write_bytes(b"not loaded before split checks")
    with pytest.raises(ValueError, match="disjoint"):
        run(
            dataset(tmp_path),
            checkpoint,
            tmp_path / "run",
            test_episodes=10001,
            large_count=4,
            small_count=1,
            validation_count=2,
        )
