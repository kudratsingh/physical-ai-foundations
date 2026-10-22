"""Independent geometry and protocol audit of the bounded-action experiment."""

import hashlib
import json
import math
from pathlib import Path

import pytest
from learning.maniskill.metrics import failure_counts, summarize
from learning.maniskill.selection import holding_score

ROOT = (
    Path(__file__).resolve().parents[2] / "learning/maniskill/evidence/pickcube_bounded"
)
NAMES = ["raw_final", "bounded_final", "bounded_selected_secondary"]


def load(name):
    return json.loads((ROOT / name).read_text())


@pytest.mark.parametrize("name", NAMES)
def test_actual_geometry_supports_full_horizon_success(name):
    data = load(f"test_{name}.json")
    assert [e["seed"] for e in data["episodes"]] == list(range(110000, 110050))
    assert (
        summarize(data["episodes"]) == data["summary"] == load("comparison.json")[name]
    )
    assert failure_counts(data["episodes"]) == load("failure_counts.json")[name]
    for e in data["episodes"]:
        assert e["steps"] == 50
        distance = math.dist(e["final_cube_position"], e["final_goal_position"])
        assert len(e["final_arm_qvel"]) == 7
        speed = max(map(abs, e["final_arm_qvel"]))
        assert distance == pytest.approx(e["final_goal_distance"], abs=1e-7)
        assert e["final_is_obj_placed"] == (distance <= 0.025)
        assert e["final_is_robot_static"] == (speed <= 0.2)
        assert e["success_at_end"] == (distance <= 0.025 and speed <= 0.2)


def test_loaded_recipe_fixed_budget_and_final_first_secondary_selection():
    config = load("config.json")
    contracts = load("loaded_recipe_contracts.json")
    assert (
        config["total_timesteps"]
        == load("training.json")["actual_timesteps"]
        == 1048576
    )
    for contract in contracts["contracts"].values():
        assert contract == dict(
            timesteps=1048576, observation_shape=[42], action_shape=[8]
        )
    assert contracts["recipes"]["raw_final"] == contracts["recipes"]["bounded_final"]
    assert (
        config["source_sha256"]
        == hashlib.sha256((ROOT.parents[1] / "bounded_ppo.py").read_bytes()).hexdigest()
    )
    assert config["selection_order"] == [
        "success_at_end_rate",
        "success_once_rate",
        "return_mean",
    ]
    assert load("selected.json") == max(load("validation.json"), key=holding_score)
    assert config["validation_seeds"] == list(range(100000, 100010))
    assert config["test_seeds"] == list(range(110000, 110050))
    assert set(config["validation_seeds"]).isdisjoint(config["test_seeds"])
    for name in NAMES[:2]:
        tests = load(f"test_{name}.json")["episodes"]
        videos = load(f"video_{name}.json")["episodes"]
        assert [e["seed"] for e in videos] == config["video_seeds"] == [110000, 110001]
        for video, expected in zip(videos, tests):
            assert len(video["trajectory"]) == 51
            assert {k: v for k, v in video.items() if k != "trajectory"} == {
                k: v for k, v in expected.items() if k != "trajectory"
            }
            assert (
                ROOT / f"video_{name}_seed{video['seed']}.mp4"
            ).stat().st_size > 1000


@pytest.mark.parametrize("name", NAMES[:2])
def test_validation_trace_preserves_transform_and_real_physics(name):
    data = load(f"validation_trace_{name}.json")
    assert data["seeds"] == list(range(100000, 100010))
    near = []
    records = []
    for episode in data["episodes"]:
        assert len(episode["records"]) == episode["steps"] == 50
        previous = episode["initial"]
        for r in episode["records"]:
            assert (
                len(r["latent_gaussian_mean"])
                == len(r["latent_gaussian_std"])
                == len(r["executed_action"])
                == 8
            )
            assert all(math.isfinite(v) and v > 0 for v in r["latent_gaussian_std"])
            if name == "bounded_final":
                assert r["distribution_class"] == "SquashedDiagGaussianDistribution"
                expected = [math.tanh(v) for v in r["latent_gaussian_mean"]]
            else:
                assert r["distribution_class"] == "DiagGaussianDistribution"
                expected = [min(1, max(-1, v)) for v in r["latent_gaussian_mean"]]
            assert r["executed_action"] == pytest.approx(expected, abs=2e-7)
            assert all(abs(v) <= 1 for v in r["executed_action"])
            distance = math.dist(r["cube"], r["goal"])
            speed = max(map(abs, r["arm_qvel"]))
            assert distance == pytest.approx(r["cube_goal_distance"], abs=1e-7)
            assert r["placement"] == (distance <= 0.025)
            assert r["static"] == (speed <= 0.2)
            assert r["success"] == (distance <= 0.025 and speed <= 0.2)
            if previous["cube_goal_distance"] <= 0.05:
                near.append(r)
            previous = r
            records.append(r)
    for group, rows in [
        ("all_steps", records),
        ("pre_action_goal_distance_within_5cm", near),
    ]:
        audit = load("command_audit.json")["results"][name][group]
        assert audit["control_steps"] == len(rows)
        if rows:
            rate = sum(
                any(abs(v) >= 0.99 for v in r["executed_action"][:7]) for r in rows
            ) / len(rows)
            assert audit["executed_arm_any_near_boundary_fraction"] == rate
