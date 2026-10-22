"""Independent audit of the controlled observation-normalization assignment."""

import json
import math
import hashlib
from pathlib import Path

import pytest
from learning.maniskill.metrics import failure_counts, summarize

EVIDENCE = (
    Path(__file__).resolve().parents[2]
    / "learning/maniskill/evidence/pickcube_normalization"
)
NAMES = ["raw_final", "normalized_final", "normalized_selected_secondary"]


def load(name):
    return json.loads((EVIDENCE / name).read_text())


@pytest.mark.parametrize("name", NAMES)
def test_full_horizon_outcomes_match_actual_geometry(name):
    data = load(f"test_{name}.json")
    assert summarize(data["episodes"]) == load("comparison.json")[name]
    assert failure_counts(data["episodes"]) == load("failures.json")[name]
    assert [e["seed"] for e in data["episodes"]] == list(range(90000, 90050))
    assert data["config"]["env_id"] == "PickCube-v1"
    for episode in data["episodes"]:
        assert episode["steps"] == 50
        distance = math.dist(
            episode["final_cube_position"], episode["final_goal_position"]
        )
        assert distance == pytest.approx(episode["final_goal_distance"], abs=1e-7)
        assert len(episode["final_arm_qvel"]) == 7
        assert all(math.isfinite(v) for v in episode["final_arm_qvel"])
        placed = distance <= 0.025
        static = max(abs(v) for v in episode["final_arm_qvel"]) <= 0.2
        assert placed == episode["final_is_obj_placed"]
        assert static == episode["final_is_robot_static"]
        assert episode["success_at_end"] == (placed and static)
        assert not episode["success_at_end"] or episode["success_once"]


def test_fixed_budget_frozen_statistics_and_predeclared_videos():
    protocol = load("config.json")
    baseline = load("baseline.json")
    contracts = load("loaded_policy_contracts.json")
    for actual in contracts.values():
        assert actual["timesteps"] == 1048576
        assert actual["observation_shape"] == [42]
        assert actual["action_shape"] == [8]
    assert baseline["loaded_contract"] == contracts["raw_final"]
    assert (
        protocol["total_timesteps"]
        == load("training.json")["actual_timesteps"]
        == baseline["training"]["actual_timesteps"]
        == 1048576
    )
    assert protocol["seed"] == baseline["config"]["seed"] == 42
    assert protocol["validation_seeds"] == list(range(80000, 80010))
    assert protocol["test_seeds"] == list(range(90000, 90050))
    assert protocol["video_seeds"] == protocol["test_seeds"][:3]
    assert set(protocol["validation_seeds"]).isdisjoint(protocol["test_seeds"])
    statistics = load("final_statistics.json")
    assert statistics["norm_obs"] and not statistics["norm_reward"]
    assert statistics["clip_obs"] == 10
    assert len(statistics["mean"]) == len(statistics["variance"]) == 42
    assert all(math.isfinite(v) and v > 0 for v in statistics["variance"])
    # One initial reset plus each environment transition, no evaluation/warmup updates.
    assert statistics["count"] == pytest.approx(1048576 + 4 + 0.0001)
    assert load("final_snapshot.json")["timesteps"] == 1048576
    for name in NAMES[:2]:
        tests = load(f"test_{name}.json")["episodes"]
        videos = load(f"video_{name}.json")["episodes"]
        for actual, reference in zip(videos, tests):
            assert len(actual["trajectory"]) == 51
            assert {k: v for k, v in actual.items() if k != "trajectory"} == {
                k: v for k, v in reference.items() if k != "trajectory"
            }
            assert (
                EVIDENCE / f"video_{name}_seed{actual['seed']}.mp4"
            ).stat().st_size > 1000


def test_paired_final_outcomes_recompute_from_shared_seeds():
    raw = load("test_raw_final.json")["episodes"]
    normalized = load("test_normalized_final.json")["episodes"]
    counts = dict(both_success=0, normalized_only=0, raw_only=0, neither_success=0)
    for a, b in zip(raw, normalized, strict=True):
        assert a["seed"] == b["seed"]
        key = {
            (True, True): "both_success",
            (False, True): "normalized_only",
            (True, False): "raw_only",
            (False, False): "neither_success",
        }[bool(a["success_at_end"]), bool(b["success_at_end"])]
        counts[key] += 1
    assert counts == load("paired_final_outcomes.json")


def test_action_audit_recomputes_from_validation_means():
    data = load("action_saturation.json")
    assert data["protocol"]["validation_seeds"] == list(range(80000, 80010))
    assert data["protocol"]["normalized_snapshot"] == load("final_snapshot.json")
    source = EVIDENCE.parents[1] / "action_saturation.py"
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == data["protocol"]["source_sha256"]
    )
    for name in NAMES[:2]:
        result = data[name]
        assert result["seeds"] == data["protocol"]["validation_seeds"]
        means = result["preclip_action_means"]
        assert result["control_steps"] == len(means) == 500
        assert all(
            len(row) == 8 and all(math.isfinite(v) for v in row) for row in means
        )
        per_joint = [
            sum(abs(row[j]) > 1 for row in means) / len(means) for j in range(7)
        ]
        assert per_joint == pytest.approx(
            result["arm_mean_outside_bounds_fraction_per_joint"]
        )
        assert sum(per_joint) / 7 == pytest.approx(
            result["arm_scalar_mean_outside_bounds_fraction"]
        )
        assert sum(any(abs(v) > 1 for v in row[:7]) for row in means) / len(
            means
        ) == pytest.approx(result["arm_any_mean_outside_bounds_fraction"])
