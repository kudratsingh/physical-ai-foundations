"""Actual pre-update stochastic collection and independent receipt checks."""

import copy
import gzip
import hashlib
import json
import random

import numpy as np
import pytest
import torch

from learning.maniskill.behavior_cloning import BehaviorPolicy
from learning.maniskill.first_rollout import compare_histories, verify_first_rollout
from learning.maniskill.rng import preserve_rng


@pytest.mark.slow
def test_actual_first_rollout_matches_before_any_update(tmp_path):
    with preserve_rng():
        torch.manual_seed(191)
        teacher = BehaviorPolicy(mean=np.zeros(42), std=np.ones(42))
        teacher.save(tmp_path / "teacher.pt")
        with preserve_rng():
            expected = random.random(), np.random.random(3), torch.rand(3)
        receipt = verify_first_rollout(tmp_path / "teacher.pt", tmp_path / "capture")
        assert random.random() == expected[0]
        np.testing.assert_array_equal(np.random.random(3), expected[1])
        torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)
    assert receipt["passed"] and receipt["optimizer_empty_after"]
    assert receipt["workers"] == 4
    assert receipt["worker_seeds"] == [42, 43, 44, 45]
    assert receipt["transitions_per_arm"] == 1024
    assert receipt["timeouts_per_arm"] == {"raw": 20, "bounded": 20}
    path = tmp_path / "capture/first_rollout.jsonl.gz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["records_sha256"]
    rows = [
        json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()
    ]
    for arm in ["raw", "bounded"]:
        steps = [row for row in rows if row["arm"] == arm and "step" in row]
        assert len(steps) == 256
        assert [row["step"] for row in steps if any(row["done"])] == [
            50,
            100,
            150,
            200,
            250,
        ]
        assert all(0 <= r <= 1 for row in steps for r in row["raw_reward"])
        for row in steps:
            np.testing.assert_array_equal(
                np.clip(row["gaussian_sample"], -1, 1), row["executed_command"]
            )
        resets = [
            row["post_observation"][0][26:32] for row in steps if any(row["done"])
        ]
        assert not np.allclose(resets[0], resets[1], rtol=0, atol=1e-7)


def test_history_comparison_rejects_one_changed_raw_reward():
    history = {
        "canonical_fingerprint": "params",
        "parent_rng_before_rollout": "rng",
        "actual_transitions": 2,
        "initial_observation": [[0]],
        "records": [
            {
                "step": 1,
                "pre_observation": [[0]],
                "post_observation": [[1]],
                "gaussian_sample": [[0.2]],
                "executed_command": [[0.2]],
                "raw_reward": [0.5],
                "done": [False],
                "timeout": [False],
                "terminal_observation": [None],
            }
        ],
    }
    changed = copy.deepcopy(history)
    changed["records"][0]["raw_reward"][0] += 0.01
    with pytest.raises(ValueError, match="raw_reward"):
        compare_histories(history, changed)
