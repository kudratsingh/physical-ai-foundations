"""Independent PPO continuation and terminal-success audits without Torch/SAPIEN."""

import json
import math
from pathlib import Path
import pytest
from learning.maniskill.metrics import summarize, failure_counts

EVIDENCE = Path(__file__).resolve().parents[2] / "learning/maniskill/evidence/ppo_debug"
NAMES = [
    "baseline",
    "control_final",
    "slow_final",
    "control_selected",
    "slow_selected",
    "bc_reference",
    "bc_goal_fault",
    "bc_restored",
]


@pytest.mark.parametrize("name", NAMES)
def test_continuation_outcomes_match_geometry_and_saved_summary(name):
    data = json.loads((EVIDENCE / f"test_{name}.json").read_text())
    comparison = json.loads((EVIDENCE / "comparison.json").read_text())
    failures = json.loads((EVIDENCE / "failures.json").read_text())
    assert summarize(data["episodes"]) == comparison[name]
    assert failure_counts(data["episodes"]) == failures[name]
    assert sum(failures[name].values()) == 50
    assert [e["seed"] for e in data["episodes"]] == list(range(70000, 70050))
    for e in data["episodes"]:
        assert e["steps"] == 50
        distance = math.dist(e["final_cube_position"], e["final_goal_position"])
        static = max(abs(v) for v in e["final_arm_qvel"]) <= 0.2
        assert len(e["final_arm_qvel"]) == 7
        assert distance == pytest.approx(e["final_goal_distance"], abs=1e-7)
        assert e["success_at_end"] == (distance <= 0.025 and static)
        assert not e["success_at_end"] or e["success_once"]


def test_controlled_protocol_and_validation_selection():
    protocol = json.loads((EVIDENCE / "protocol.json").read_text())
    a, b = protocol["configurations"].values()
    assert a["learning_rate"] == 3e-4
    assert b["learning_rate"] == 1e-4
    keys = set(a) - {"learning_rate", "created_utc"}
    assert {k: a[k] for k in keys} == {k: b[k] for k in keys}
    assert a["checkpoint_sha256"] == protocol["starting_checkpoint_sha256"]
    assert set(protocol["initial_parameters_sha256"].values()) == {
        protocol["original_parameters_sha256"]
    }
    assert len(set(protocol["optimizer_sha256"].values())) == 1
    assert set(protocol["validation_seeds"]).isdisjoint(protocol["test_seeds"])
    assert set(protocol["bc_source_seed_values"]).isdisjoint(
        protocol["test_seeds"] + protocol["validation_seeds"]
    )
    for name in ["control", "slow"]:
        history = json.loads((EVIDENCE / f"{name}_validation.json").read_text())
        selected = json.loads((EVIDENCE / f"{name}_selected.json").read_text())
        assert selected == max(
            history,
            key=lambda p: (
                p["success_at_end_rate"],
                p["success_once_rate"],
                p["return_mean"],
            ),
        )
        assert [p["additional_timesteps"] for p in history] == [
            0,
            131072,
            262144,
            393216,
            524288,
        ]
        training = json.loads((EVIDENCE / f"{name}_training.json").read_text())
        assert training["actual_additional_timesteps"] == 524288
        assert training["total_timesteps"] == 1572864
        assert (
            training["final_learning_rate"]
            == protocol["configurations"][name]["learning_rate"]
        )


def test_rendered_episodes_are_the_predeclared_test_prefix():
    protocol = json.loads((EVIDENCE / "protocol.json").read_text())
    assert protocol["video_seeds"] == protocol["test_seeds"][:2]
    for name in protocol["video_policies"]:
        video = json.loads((EVIDENCE / f"video_{name}.json").read_text())
        test = json.loads((EVIDENCE / f"test_{name}.json").read_text())
        assert video["episodes"] == test["episodes"][:2]
        for seed in protocol["video_seeds"]:
            assert (EVIDENCE / f"{name}_seed{seed}.mp4").stat().st_size > 1000


def test_restoring_goal_observations_recovers_clean_behavior_exactly():
    clean = json.loads((EVIDENCE / "test_bc_reference.json").read_text())
    restored = json.loads((EVIDENCE / "test_bc_restored.json").read_text())
    assert clean["episodes"] == restored["episodes"]
