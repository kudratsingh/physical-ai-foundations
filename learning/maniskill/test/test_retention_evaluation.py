"""Passive holding diagnostics preserve ordinary episodes and event timing."""

import random

import numpy as np
import pytest
import torch

from learning.maniskill.behavior_cloning import BehaviorPolicy
from learning.maniskill.experiment import Config, evaluate
from learning.maniskill.retention_evaluation import (
    evaluate_retention,
    retention_summary,
)
from learning.maniskill.rng import preserve_rng


def state(success, placed=None):
    placed = success if placed is None else placed
    return {
        "success": success,
        "geometric_success": success,
        "placed": placed,
        "official_placed": placed,
        "static": True,
        "official_static": True,
        "max_arm_speed": 0.1,
    }


def test_post_success_metrics_exclude_first_reaching_command_and_keep_lost_states():
    # The first large command reaches the goal. Two smaller commands follow,
    # immediately losing placement despite a static arm; the policy stays lost.
    records = []
    for step, (pre, post, arm, gripper) in enumerate(
        [(False, True, 1.0, 1.0), (True, False, 0.2, 0.4), (False, False, 0.0, -0.2)],
        start=1,
    ):
        records.append(
            {
                "seed": 1,
                "step": step,
                "after_first_success": step > 1,
                "pre": state(pre),
                "post": state(post),
                "command": [arm] * 7 + [gripper],
                "gripper_delta": [0.01, -0.01],
                "teacher_arm_drift_l2": np.sqrt(7) * arm,
                "teacher_gripper_drift_abs": abs(gripper),
            }
        )
    episode = {
        "success_once": True,
        "success_at_end": False,
        "final_is_robot_static": True,
    }
    result = retention_summary(records, [episode])
    assert result["post_first_success_steps"] == 2
    assert result["post_first_success_arm_command_l2_mean"] == pytest.approx(
        np.sqrt(7) * 0.1
    )
    assert result["post_first_success_gripper_command_abs_mean"] == pytest.approx(0.3)
    assert result["post_first_success_success_occupancy"] == 0
    assert result["success_to_failure_transitions"] == 1
    assert result["success_to_failure_while_post_static"] == 1
    assert result["ever_successful_but_final_failure_and_static"] == 1
    assert result["geometry_official_mismatches"] == 0
    never = retention_summary(
        [],
        [
            {
                "success_once": False,
                "success_at_end": False,
                "final_is_robot_static": True,
            }
        ],
    )
    assert never["post_first_success_success_occupancy"] is None
    assert never["post_first_success_arm_command_l2_mean"] is None


@pytest.mark.slow
def test_native_passive_teacher_does_not_change_standard_episodes_or_rng(tmp_path):
    task = Config(env_id="PickCube-v1", num_envs=1)
    with preserve_rng():
        torch.manual_seed(151)
        teacher = BehaviorPolicy(mean=np.zeros(42), std=np.ones(42))
        before = {name: value.clone() for name, value in teacher.state_dict().items()}
        standard = evaluate(task, "zero", [314159, 314160])
        with preserve_rng():
            expected = random.random(), np.random.random(3), torch.rand(3)
        measured = evaluate_retention(
            task, "zero", teacher, [314159, 314160], tmp_path / "retention.json"
        )
        assert random.random() == expected[0]
        np.testing.assert_array_equal(np.random.random(3), expected[1])
        torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)
        standard_bc = evaluate(task, teacher, [314159])
        measured_bc = evaluate_retention(task, teacher, teacher, [314159])
    assert measured["episodes"] == standard["episodes"]
    assert measured["summary"] == standard["summary"]
    assert measured["retention"]["geometry_official_mismatches"] == 0
    assert measured_bc["episodes"] == standard_bc["episodes"]
    assert measured_bc["summary"] == standard_bc["summary"]
    assert measured_bc["retention"]["teacher_arm_drift_l2_mean"] == 0
    assert measured_bc["retention"]["teacher_gripper_drift_abs_mean"] == 0
    assert len(measured["step_records"]) == 100
    for name, value in teacher.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    for seed in [314159, 314160]:
        rows = [row for row in measured["step_records"] if row["seed"] == seed]
        assert [r["step"] for r in rows] == list(range(1, 51))
        for index, row in enumerate(rows):
            assert row["command"] == [0.0] * 8
            expected_action, _ = teacher.predict(row["observation"], deterministic=True)
            np.testing.assert_array_equal(row["teacher_command"], expected_action)
            np.testing.assert_allclose(
                row["observation"][39:42],
                np.subtract(row["pre"]["goal"], row["pre"]["cube"]),
                rtol=0,
                atol=1e-7,
            )
            if index:
                assert row["pre"] == rows[index - 1]["post"]
        assert rows[-1]["truncated"] and not rows[-1]["terminated"]
