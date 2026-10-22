"""Independent audits of committed PickCube outcomes; no simulator dependencies."""

import json
import math
from pathlib import Path
import pytest
from learning.maniskill.metrics import failure_counts, summarize

EVIDENCE = (
    Path(__file__).resolve().parents[2] / "learning/maniskill/evidence/stage1_pickcube"
)
NAMES = ["bc_32", "bc_256", "ppo", "random", "zero"]


@pytest.mark.parametrize("name", NAMES)
def test_pickcube_metrics_match_3d_geometry_and_static_arm(name):
    data = json.loads((EVIDENCE / f"test_{name}.json").read_text())
    comparison = json.loads((EVIDENCE / "comparison.json").read_text())
    failures = json.loads((EVIDENCE / "failures.json").read_text())
    assert summarize(data["episodes"]) == comparison[name]
    assert failure_counts(data["episodes"]) == failures[name]
    assert sum(failures[name].values()) == 50
    assert [e["seed"] for e in data["episodes"]] == list(range(50000, 50050))
    assert data["config"]["env_id"] == "PickCube-v1"
    for e in data["episodes"]:
        assert e["steps"] == 50
        assert len(e["final_arm_qvel"]) == 7
        distance = math.dist(e["final_cube_position"], e["final_goal_position"])
        assert distance == pytest.approx(e["final_goal_distance"], abs=1e-7)
        placed = distance <= 0.025
        static = max(abs(v) for v in e["final_arm_qvel"]) <= 0.2
        assert e["final_is_obj_placed"] == placed
        assert e["final_is_robot_static"] == static
        assert e["success_at_end"] == (placed and static)
        assert not e["success_at_end"] or e["success_once"]
        assert (e["first_success_step"] is not None) == e["success_once"]
        assert e["lifted_5cm"] == (
            e["maximum_cube_height"] >= e["initial_cube_height"] + 0.05
        )


def test_predeclared_splits_and_rendered_rollouts_match_tests():
    protocol = json.loads((EVIDENCE / "protocol.json").read_text())
    assert len(protocol["train_ids"]) == 256
    assert len(protocol["validation_ids"]) == 64
    assert set(protocol["train_ids"]).isdisjoint(protocol["validation_ids"])
    assert (
        len(set(protocol["initial_observation_hashes"].values()))
        == protocol["retained_successful_replays"]
    )
    assert set(protocol["source_seed_values"]).isdisjoint(
        protocol["test_seeds"] + protocol["validation_rollout_seeds"]
    )
    assert set(protocol["test_seeds"]).isdisjoint(protocol["validation_rollout_seeds"])
    assert protocol["video_seeds"] == protocol["test_seeds"][:3]
    small = json.loads((EVIDENCE / "bc_32_config.json").read_text())
    large = json.loads((EVIDENCE / "bc_256_config.json").read_text())
    assert small["train_ids"] == large["train_ids"][:32] == protocol["train_ids"][:32]
    assert (
        small["validation_ids"] == large["validation_ids"] == protocol["validation_ids"]
    )
    for name, reference_name in [("bc", "bc_256"), ("ppo", "ppo")]:
        rendered = json.loads((EVIDENCE / f"video_{name}.json").read_text())
        reference = json.loads((EVIDENCE / f"test_{reference_name}.json").read_text())
        assert rendered["episodes"] == reference["episodes"][:3]
        for seed in protocol["video_seeds"]:
            assert (EVIDENCE / f"{name}_seed{seed}.mp4").stat().st_size > 1000
