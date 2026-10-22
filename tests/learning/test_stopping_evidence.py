"""Independent audits of privileged stopping evidence, without simulator imports."""

import json
import math
from pathlib import Path

import pytest
from learning.maniskill.metrics import summarize

ROOT = (
    Path(__file__).resolve().parents[2]
    / "learning/maniskill/evidence/pickcube_stopping"
)
NAMES = [
    f"{name}_{mode}"
    for name in ["raw_final", "normalized_final"]
    for mode in ["normal", "override"]
]


def load(name):
    return json.loads((ROOT / name).read_text())


@pytest.mark.parametrize("name", NAMES)
def test_privileged_diagnostic_outcomes_follow_geometry_and_override_contract(name):
    data = load(f"{name}.json")
    assert "Privileged validation diagnostic" in data["scope"]
    assert [e["seed"] for e in data["episodes"]] == list(range(80000, 80010))
    assert (
        summarize(data["episodes"])
        == data["summary"]
        == load("comparison.json")[name]["summary"]
    )
    for episode in data["episodes"]:
        assert episode["steps"] == len(episode["records"]) == 50
        assert [r["step"] for r in episode["records"]] == list(range(1, 51))
        placements = [0] if episode["initial"]["placement"] else []
        placements += [r["step"] for r in episode["records"] if r["placement"]]
        first_placement = placements[0] if placements else None
        assert first_placement == episode["first_placement_step"]
        active = []
        successes = []
        for record in episode["records"]:
            distance = math.dist(record["cube"], record["goal"])
            speed = max(abs(v) for v in record["arm_qvel"])
            assert len(record["arm_qvel"]) == 7
            assert distance == pytest.approx(record["cube_goal_distance"], abs=1e-7)
            assert speed == record["max_arm_qvel"]
            assert record["placement"] == (distance <= 0.025)
            assert record["static"] == (speed <= 0.2)
            assert record["success"] == (record["placement"] and record["static"])
            if record["success"]:
                successes.append(record["step"])
            assert record["original_action"] == [
                min(1, max(-1, v)) for v in record["gaussian_mean"]
            ]
            assert len(record["gaussian_std"]) == 8
            assert all(math.isfinite(v) and v > 0 for v in record["gaussian_std"])
            should_override = (
                data["override"]
                and first_placement is not None
                and record["step"] > first_placement
            )
            assert record["override_active"] == should_override
            assert record["executed_action"][7] == record["original_action"][7]
            if should_override:
                active.append(record)
                assert record["executed_action"][:7] == [0] * 7
                assert record["arm_target"] == pytest.approx(
                    record["pre_arm_qpos"], abs=1e-7
                )
            else:
                assert record["executed_action"] == record["original_action"]
        assert episode["return"] == pytest.approx(
            sum(r["reward"] for r in episode["records"])
        )
        assert episode["success_once"] == bool(successes)
        assert episode["success_at_end"] == episode["records"][-1]["success"]
        assert episode["first_success_step"] == (successes[0] if successes else None)
        assert episode["active_steps"] == len(active)
        assert episode["post_activation_success"] == any(r["success"] for r in active)
        assert episode["first_override_action_step"] == (
            active[0]["step"] if active else None
        )
    activated = [e for e in data["episodes"] if e["active_steps"]]
    assert data["activation"] == dict(
        episodes=len(activated),
        total_active_steps=sum(e["active_steps"] for e in activated),
        post_activation_any_success=sum(
            e["post_activation_success"] for e in activated
        ),
        activated_final_success=sum(e["success_at_end"] for e in activated),
    )
    if name.startswith("normalized"):
        assert data["frozen_statistics_unchanged"] is True


def test_predeclared_videos_baselines_and_pre_activation_equality():
    protocol = load("protocol.json")
    assert protocol["validation_seeds"] == list(range(80000, 80010))
    assert protocol["video_seeds"] == [80000, 80001]
    assert not protocol["training"] and not protocol["final_test_seeds_used"]
    assert load("baseline_equality.json")["all500_preclip_means_match"] == dict(
        raw_final=True, normalized_final=True
    )
    for name in NAMES:
        data = load(f"{name}.json")
        assert load(f"video_{name}.json")["episodes"] == data["episodes"][:2]
        for seed in protocol["video_seeds"]:
            assert (ROOT / f"video_{name}_seed{seed}.mp4").stat().st_size > 1000
    for policy in ["raw_final", "normalized_final"]:
        baseline = load(f"{policy}_normal.json")["episodes"]
        overridden = load(f"{policy}_override.json")["episodes"]
        for ordinary, changed in zip(baseline, overridden, strict=True):
            assert ordinary["seed"] == changed["seed"]
            assert ordinary["initial"] == changed["initial"]
            first_action = changed["first_override_action_step"]
            prefix = 50 if first_action is None else first_action - 1
            assert ordinary["records"][:prefix] == changed["records"][:prefix]


def test_additional_activation_examples_reference_original_outcomes():
    import hashlib

    examples = load("activation_examples.json")
    assert "Illustrative activation videos" in examples["protocol"]["scope"]
    assert len(examples["videos"]) == 4
    selection = load("plot_examples.json")["seeds"]
    for video in examples["videos"]:
        assert video["seed"] == selection[video["policy"]]
        original = load(video["reference"])["episodes"]
        expected = next(e for e in original if e["seed"] == video["seed"])
        assert video["active_steps"] == expected["active_steps"]
        assert video["success_at_end"] == expected["success_at_end"]
        assert video["all_recorded_fields_match"]
        content = (ROOT / video["video"]).read_bytes()
        assert len(content) > 1000
        assert hashlib.sha256(content).hexdigest() == video["video_sha256"]
