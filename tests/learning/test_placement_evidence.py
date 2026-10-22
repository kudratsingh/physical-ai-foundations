"""Independent P0 audit: compact traces, observation-only commands and occupancy."""

import hashlib
import json
import math
from pathlib import Path
import statistics

import pytest
from learning.maniskill.metrics import summarize
from learning.maniskill.observation_stop_gate import (
    ObservationStopGate,
    placed_from_observation,
)

ROOT = (
    Path(__file__).resolve().parents[2]
    / "learning/maniskill/evidence/pickcube_placement"
)


def load(name):
    return json.loads((ROOT / name).read_text())


def episodes(name):
    return [json.loads(line) for line in (ROOT / name).read_text().splitlines()]


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def check_occupancy(data, summary):
    grasped = [r for e in data for r in e["records"] if r["is_grasped"]]
    assert len(grasped) == summary["grasped_steps"]
    assert len(data) == summary["episodes"]
    assert sum(e["steps"] for e in data) == summary["steps"]
    assert summarize(data) == summary["official"]
    for key, threshold in [("10cm", 0.10), ("5cm", 0.05), ("2_5cm", 0.025)]:
        close = [r for r in grasped if r["cube_goal_distance"] <= threshold]
        stage = summary["stages"][key]
        assert stage["grasped_steps_within"] == len(close)
        assert stage["grasped_step_denominator"] == len(grasped)
        assert stage["conditional_occupancy"] == (
            len(close) / len(grasped) if grasped else None
        )
        for field, stored in [
            ("max_arm_qvel", "mean_max_arm_speed"),
            ("arm_command_l2", "mean_arm_command_l2"),
            ("distance_change", "mean_distance_change"),
        ]:
            if close:
                assert stage[stored] == pytest.approx(
                    statistics.mean(r[field] for r in close)
                )
            else:
                assert stage[stored] is None
        hits = []
        for episode in data:
            first = next(
                (
                    r["step"]
                    for r in episode["records"]
                    if r["is_grasped"] and r["cube_goal_distance"] <= threshold
                ),
                None,
            )
            if first is not None:
                hits.append(
                    dict(seed=episode["seed"], step=first, remaining_steps=50 - first)
                )
        assert hits == stage["episode_first_hits"]


@pytest.mark.parametrize("name", ["bc_256", "raw_final", "normalized_final"])
def test_all_placement_geometry_commands_occupancy_and_first_hits(name):
    data = episodes(f"placement_{name}.jsonl")
    summary = load("placement_summary.json")[name]
    assert [e["seed"] for e in data] == list(range(130000, 130010))
    for episode in data:
        assert episode["steps"] == len(episode["records"]) == 50
        assert [r["step"] for r in episode["records"]] == list(range(1, 51))
        previous_distance = math.dist(
            episode["initial"]["cube"], episode["initial"]["goal"]
        )
        successes = []
        for r in episode["records"]:
            assert r["remaining_steps"] == 50 - r["step"]
            distance = math.dist(r["cube"], r["goal"])
            speed = max(abs(v) for v in r["arm_qvel"])
            assert len(r["arm_qvel"]) == 7
            assert distance == pytest.approx(r["cube_goal_distance"], abs=1e-7)
            assert speed == r["max_arm_qvel"]
            assert r["placement"] == (distance <= 0.025)
            assert r["static"] == (speed <= 0.2)
            assert r["success"] == (r["placement"] and r["static"])
            assert r["pre_distance"] == pytest.approx(previous_distance, abs=1e-7)
            assert r["distance_change"] == r["cube_goal_distance"] - r["pre_distance"]
            assert r["arm_command_l2"] == pytest.approx(
                math.sqrt(sum(x * x for x in r["action"][:7])), abs=5e-7
            )
            assert r["arm_command_max_abs"] == max(abs(x) for x in r["action"][:7])
            assert len(r["action"]) == 8 and max(abs(x) for x in r["action"]) <= 1
            assert r["obs_cube_to_goal"] == pytest.approx(
                [g - c for g, c in zip(r["goal"], r["cube"])], abs=1e-7
            )
            assert all(math.isfinite(value) for value in r["action"] + r["arm_qvel"])
            if r["success"]:
                successes.append(r["step"])
            previous_distance = distance
        assert episode["success_once"] == bool(successes)
        assert episode["success_at_end"] == episode["records"][-1]["success"]
        assert episode["first_success_step"] == (successes[0] if successes else None)
        assert episode["return"] == sum(r["reward"] for r in episode["records"])
    check_occupancy(data, summary)


@pytest.mark.parametrize("name", ["raw_final", "normalized_final"])
def test_observation_gate_commands_and_all_states_match_privileged_reference(name):
    data = episodes(f"hybrid_{name}.jsonl")
    checks = episodes(f"gate_checks_{name}.jsonl")
    prior_path = ROOT.parent / "pickcube_stopping" / f"{name}_override.json"
    prior = json.loads(prior_path.read_text())
    assert data == prior["episodes"]
    assert [e["seed"] for e in data] == list(range(80000, 80010))
    audit_summary = load("observation_audit.json")[name]
    assert audit_summary["states_audited"] == 510
    assert (
        audit_summary["source_sha256"]
        == hashlib.sha256(prior_path.read_bytes()).hexdigest()
    )
    assert len(checks) == 10
    for episode, audit in zip(data, checks, strict=True):
        assert episode["seed"] == audit["seed"]
        assert [r["state_step"] for r in audit["states"]] == list(range(51))
        measurements = [episode["initial"]] + episode["records"]
        gate = ObservationStopGate()
        for index, (audit_state, measured) in enumerate(
            zip(audit["states"], measurements, strict=True)
        ):
            obs = [0.0] * 42
            obs[39:42] = audit_state["cube_to_goal"]
            assert (
                placed_from_observation(obs)
                == audit_state["observed_placement"]
                == audit_state["privileged_placement"]
                == measured["placement"]
            )
            assert audit_state["cube_to_goal"] == pytest.approx(
                [g - c for g, c in zip(measured["goal"], measured["cube"])], abs=1e-7
            )
            if index < 50:
                command_record = episode["records"][index]
                command, active, _ = gate.action(
                    obs, command_record["original_action"], index
                )
                assert command.tolist() == command_record["executed_action"]
                assert active == command_record["override_active"]
    summary = load("hybrid_comparison.json")[name]
    assert "hybrid" in summary["scope"]
    assert summarize(data) == summary["official"]
    assert sum(e["active_steps"] > 0 for e in data) == summary["activated_episodes"]
    assert sum(e["active_steps"] for e in data) == summary["active_steps"]


def test_protocol_source_hashes_video_digests_and_conditional_subset():
    protocol = load("protocol.json")
    assert not protocol["training"] and not protocol["final_test_seeds_used"]
    assert protocol["placement_seeds"] == list(range(130000, 130010))
    assert protocol["observation_stop_seeds"] == list(range(80000, 80010))
    assert protocol["video_seeds"] == dict(
        observation_stop=[80000, 80001], placement=[130000, 130001]
    )
    source = ROOT.parents[1]
    assert (
        hashlib.sha256((source / "placement_diagnostic.py").read_bytes()).hexdigest()
        == protocol["source_sha256"]
    )
    assert (
        hashlib.sha256((source / "observation_stop_gate.py").read_bytes()).hexdigest()
        == protocol["gate_source_sha256"]
    )
    subset = load("placement_summary.json")["bc_final_success_subset"]
    assert "not a performance comparison" in subset["scope"]
    selected = [e for e in episodes("placement_bc_256.jsonl") if e["success_at_end"]]
    assert subset["selected_seeds"] == [e["seed"] for e in selected]
    check_occupancy(selected, subset["data"])
    manifest = load("video_manifest.json")
    assert len(manifest) == 10
    for path, entry in manifest.items():
        binary = ROOT / path
        assert binary.stat().st_size > 1000
        assert hashlib.sha256(binary.read_bytes()).hexdigest() == entry.get(
            "video_sha256", entry.get("source_sha256")
        )
        seed = int(binary.stem.split("seed")[-1])
        name = path.split("_seed")[0] + ".jsonl"
        episode = next(e for e in episodes(name) if e["seed"] == seed)
        assert digest(episode) == entry["episode_sha256"]
