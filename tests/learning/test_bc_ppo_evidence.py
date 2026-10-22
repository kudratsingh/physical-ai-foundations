"""Reconstruct paired BC/PPO results from compact, committed physical evidence."""

import hashlib
import json
import math
import statistics
from pathlib import Path

import pytest

from learning.maniskill.metrics import summarize
from learning.maniskill.selection import holding_score

ROOT = Path(__file__).resolve().parents[2] / "learning/maniskill/evidence/bc_ppo_paired"
NAMES = [
    "teacher",
    "warm_initialized",
    "warm_final",
    "control_final",
    "warm_selected_secondary",
]


def load(name):
    return json.loads((ROOT / name).read_text())


def episodes(name):
    return [
        json.loads(line) for line in (ROOT / f"{name}.jsonl").read_text().splitlines()
    ]


@pytest.mark.parametrize("name", NAMES)
def test_fresh_test_geometry_and_recomputed_metrics(name):
    data = episodes(f"test_{name}")
    assert [e["seed"] for e in data] == list(range(150000, 150050))
    for e in data:
        distance = math.dist(e["final_cube_position"], e["final_goal_position"])
        speed = max(map(abs, e["final_arm_qvel"]))
        assert e["steps"] == 50
        assert distance == pytest.approx(e["final_goal_distance"], abs=1e-7)
        assert e["success_at_end"] == (distance <= 0.025 and speed <= 0.2)
        assert not e["success_at_end"] or e["success_once"]
    assert summarize(data) == load("comparison.json")[name]
    g = load("geometry.json")[name]
    assert g["final_successes"] == sum(e["success_at_end"] for e in data)
    assert g["median_goal_distance"] == statistics.median(
        e["final_goal_distance"] for e in data
    )
    assert g["median_max_arm_speed"] == statistics.median(
        max(map(abs, e["final_arm_qvel"])) for e in data
    )


def test_teacher_transfer_and_actual_paired_initialization():
    assert episodes("test_teacher") == episodes("test_warm_initialized")
    assert episodes("preflight_teacher") == episodes("preflight_initialized")
    p = load("preflight.json")
    assert p["action_agreement"]["max_absolute_action_error"] == 0
    assert p["max_rollout_field_error"] == 0
    warm, control = [load(f"{a}_initialization.json") for a in ["warm", "control"]]
    for key in ["non_actor_fingerprint", "rng_fingerprint", "optimizer_empty"]:
        assert warm[key] == control[key]
    assert warm["optimizer_empty"] is True
    assert warm["actor_fingerprint"] != control["actor_fingerprint"]
    assert warm["non_actor_fingerprint"] == p["non_actor_fingerprint"]
    assert warm["actor_fingerprint"] == p["warm_actor_fingerprint"]
    assert control["actor_fingerprint"] == p["random_actor_fingerprint"]


@pytest.mark.parametrize("arm", ["warm", "control"])
def test_training_counts_budget_and_selection(arm):
    protocol = load("protocol.json")
    assert protocol["test_seeds"] == list(range(150000, 150050))
    assert protocol["validation_seeds"] == list(range(140000, 140010))
    assert not set(protocol["test_seeds"]) & set(protocol["validation_seeds"])
    budget = protocol["config"]["total_timesteps"]
    assert budget == 1048576
    training = load(f"{arm}_training.json")
    assert training["complete"] and training["actual_timesteps"] == budget
    metrics = load(f"{arm}_training_metrics.json")
    assert "excludes all validation and test" in metrics["scope"]
    windows = metrics["windows"]
    previous = 0
    for w in windows:
        assert w["timesteps"] == previous + w["transitions"]
        previous = w["timesteps"]
        assert 0 <= w["success"] <= min(w["placed"], w["static"]) <= w["transitions"]
        assert (
            0
            <= w["grasped_within_2_5cm"]
            <= w["grasped_within_5cm"]
            <= w["grasped_within_10cm"]
            <= w["grasped"]
            <= w["transitions"]
        )
        assert (
            0
            <= w.get("episodes_final_success", 0)
            <= w.get("episodes_any_success", 0)
            <= w.get("episodes", 0)
        )
    assert previous == budget
    totals = load("training_totals.json")[arm]
    for key, value in totals.items():
        assert value == sum(w.get(key, 0) for w in windows)
    assert totals["episodes"] == 4 * (budget // 4 // 50)
    validation = load(f"{arm}_validation.json")
    assert [v["timesteps"] for v in validation] == list(range(0, budget + 1, 65536))
    assert load(f"{arm}_selected.json") == max(validation, key=holding_score)


def test_predeclared_videos_and_finite_optimizer_trace():
    manifest = load("video_manifest.json")
    assert {(v["policy"], v["seed"]) for v in manifest} == {
        (p, s)
        for p in ["teacher", "warm_final", "control_final"]
        for s in [150000, 150001]
    }
    for item in manifest:
        path = ROOT / item["file"]
        assert path.stat().st_size > 1000
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        assert item["all_nontrajectory_fields_match"]
    trace = load("optimizer_trace.json")
    for rows in trace.values():
        assert rows[-1]["total_timesteps"] == 1048576
        assert all(math.isfinite(v) for row in rows for v in row.values())


def test_launch_source_archive_and_excluded_pilot_are_explicit():
    protocol = load("protocol.json")
    for name, expected in protocol["sources"].items():
        assert (
            hashlib.sha256((ROOT / "launch_sources" / name).read_bytes()).hexdigest()
            == expected
        )
    pilot = load("excluded_pilot.json")
    assert pilot["status"] == "excluded implementation pilot"
    assert pilot["final_test_seeds_used"] is False
    assert "NaN" in pilot["reason"]


def test_postcomparison_provenance_requires_actual_final_models_and_identity():
    p = load("provenance.json")
    assert p["complete_required"] and p["all_final_budgets_verified"]
    assert p["finals_present"] == {"warm": True, "control": True}
    assert p["teacher_initialized_test_identity"] == {
        "episodes": 50,
        "full_episode_identity": True,
        "tolerance": 0,
    }
    assert p["actual_initial_actor_equals_teacher"]
    assert p["actual_initial_nonactor_pair_matches"]
    assert p["actual_initial_optimizers_empty"]
    assert p["teacher_sha256"] == load("protocol.json")["teacher_sha256"]
    assert (
        p["protocol_sha256"]
        == hashlib.sha256((ROOT / "protocol.json").read_bytes()).hexdigest()
    )
    for arm in ["warm", "control"]:
        actual = p["actual_checkpoints"][arm]
        assert (
            actual["initial"]["checkpoint_sha256"]
            == load(f"{arm}_initialization.json")["checkpoint_sha256"]
        )
        assert actual["final"]["recipe"]["timesteps"] == 1048576
        assert actual["final"]["training"] == load(f"{arm}_training.json")
