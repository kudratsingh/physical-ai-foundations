"""Fixed probe provenance and fail-closed installed reward inspection; no physics."""

import importlib.metadata
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from learning.maniskill.critic_audit import (
    load_probes,
    reward_bound_contract,
    sha256,
)


@pytest.fixture
def saved_probes(tmp_path):
    dataset = tmp_path / "demos.h5"
    with h5py.File(dataset, "w") as file:
        for identifier in [1, 2]:
            group = file.create_group(f"traj_{identifier}")
            group["obs"] = np.full((3, 42), identifier, dtype=np.float32)
            group["actions"] = np.zeros((2, 8), dtype=np.float32)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "dataset_sha256": sha256(dataset),
                "validation_ids": [2, 1],
                "train_ids": [100],
            }
        )
    )
    return dataset, protocol


def test_probes_follow_saved_split_order_and_include_endpoint(saved_probes):
    dataset, protocol = saved_probes
    observations, provenance = load_probes(dataset, protocol, episodes=1)
    np.testing.assert_array_equal(observations, np.full((3, 42), 2, np.float32))
    assert provenance["trajectory_ids"] == [2]
    assert provenance["observation_count"] == 3
    assert provenance["dataset_sha256"] == sha256(dataset)
    assert provenance["split_protocol_sha256"] == sha256(protocol)
    assert [row["observation_index"] for row in provenance["indices"]] == [0, 1, 2]
    assert [row["initial"] for row in provenance["indices"]] == [True, False, False]
    assert [row["endpoint"] for row in provenance["indices"]] == [False, False, True]


def test_probes_reject_data_mutation_and_training_overlap(saved_probes):
    dataset, protocol = saved_probes
    saved = json.loads(protocol.read_text())
    saved["train_ids"] = [2]
    protocol.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="overlap"):
        load_probes(dataset, protocol, episodes=1)
    saved["train_ids"] = [100]
    protocol.write_text(json.dumps(saved))
    with h5py.File(dataset, "a") as file:
        file["traj_2/obs"][0, 0] = 100
    with pytest.raises(ValueError, match="differs"):
        load_probes(dataset, protocol, episodes=1)


def test_probes_reject_misaligned_demo_even_with_updated_dataset_hash(saved_probes):
    dataset, protocol = saved_probes
    with h5py.File(dataset, "a") as file:
        del file["traj_2/actions"]
        file["traj_2/actions"] = np.zeros((1, 8), dtype=np.float32)
    saved = json.loads(protocol.read_text())
    saved["dataset_sha256"] = sha256(dataset)
    protocol.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="alignment"):
        load_probes(dataset, protocol, episodes=1)


def test_installed_reward_bound_rejects_formula_drift(tmp_path):
    source = Path(
        importlib.metadata.distribution("mani-skill-nightly").locate_file(
            "mani_skill/envs/tasks/tabletop/pick_cube.py"
        )
    )
    contract = reward_bound_contract(source)
    assert contract["normalized_reward_min"] == 0
    assert contract["normalized_reward_max"] == 1
    assert contract["source_sha256"] == sha256(source)
    changed = tmp_path / "task.py"
    text = source.read_text()
    assert (
        "return self.compute_dense_reward(obs=obs, action=action, info=info) / 5"
        in text
    )
    changed.write_text(text.replace("info=info) / 5", "info=info) / 6"))
    with pytest.raises(ValueError, match="divisor"):
        reward_bound_contract(changed)
    changed.write_text(
        text.replace(
            "tcp_to_obj_dist = torch.linalg.norm(",
            "tcp_to_obj_dist = -torch.linalg.norm(",
        )
    )
    with pytest.raises(ValueError, match="formula"):
        reward_bound_contract(changed)
