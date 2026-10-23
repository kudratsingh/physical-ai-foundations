"""No-simulator checkpoint guards, conditioned traces and report calibration."""

import copy
import gzip
import json

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from learning.maniskill.behavior_cloning import BehaviorPolicy
from learning.maniskill.metrics import summarize
from learning.maniskill.retention_compare import check_model, verify_video
from learning.maniskill.retention_evaluation import retention_summary
from learning.maniskill.retention_experiment import build
from learning.maniskill.retention_ppo import RetentionPPO
from learning.maniskill.retention_report import (
    calibration_analysis,
    export_result,
    immediate_success_summary,
    plots,
    promising_criterion,
    trace_associations,
)
from learning.maniskill.rng import preserve_rng


class ShapesOnly(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (42,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, (8,), dtype=np.float32)

    def reset(self, **kwargs):
        raise AssertionError("Evidence guard unit test must not run a simulator")

    def step(self, action):
        raise AssertionError("Evidence guard unit test must not collect transitions")


def test_actual_saved_initial_guard_rejects_final_budget_and_changed_teacher(tmp_path):
    with preserve_rng():
        torch.set_num_threads(1)
        teacher = BehaviorPolicy(mean=np.zeros(42), std=np.ones(42))
        env = DummyVecEnv([ShapesOnly] * 4)
        try:
            model = build(env, teacher, 42, 10.0)
            model.save(tmp_path / "initial")
            loaded = RetentionPPO.load(tmp_path / "initial.zip", device="cpu")
            assert check_model(loaded, teacher, 42, 10.0, 0, initial=True)[
                "teacher_frozen"
            ]
            with pytest.raises(ValueError, match="budget"):
                check_model(loaded, teacher, 42, 10.0, 262144)
            with pytest.raises(ValueError, match="coefficient"):
                check_model(loaded, teacher, 42, 1.0, 0)
            with torch.no_grad():
                loaded.teacher.net[0].weight[0, 0] += 1
            with pytest.raises(ValueError, match="teacher differs"):
                check_model(loaded, teacher, 42, 10.0, 0)
        finally:
            env.close()


def fixture_result():
    state = {
        "tcp": [0, 0, 0],
        "cube": [0, 0, 0],
        "goal": [0.1, 0, 0],
        "arm_qvel": [0] * 7,
        "gripper_qpos": [0.02, 0.02],
        "cube_goal_distance": 0.1,
        "max_arm_speed": 0.0,
        "placed": False,
        "static": True,
        "geometric_success": False,
        "success": False,
        "grasped": False,
        "official_placed": False,
        "official_static": True,
    }
    episode = {
        "seed": 7,
        "return": 25.0,
        "steps": 50,
        "success_once": False,
        "success_at_end": False,
        "first_success_step": None,
        "trajectory": [],
        "final_cube_position": [0, 0, 0],
        "final_goal_position": [0.1, 0, 0],
        "final_arm_qvel": [0] * 7,
        "grasped_once": False,
        "placed_once": False,
        "initial_cube_height": 0.0,
        "maximum_cube_height": 0.0,
        "lifted_5cm": False,
        "final_goal_distance": 0.1,
        "final_is_obj_placed": False,
        "final_is_robot_static": True,
    }
    records = [
        {
            "seed": 7,
            "step": step,
            "observation": [0.0] * 42,
            "pre": copy.deepcopy(state),
            "post": copy.deepcopy(state),
            "command": [0.0] * 8,
            "teacher_command": [1.0] * 8,
            "gripper_delta": [0.0, 0.0],
            "teacher_arm_drift_l2": float(np.sqrt(7)),
            "teacher_gripper_drift_abs": 1.0,
            "after_first_success": False,
            "raw_reward": 0.5,
            "terminated": False,
            "truncated": step == 50,
        }
        for step in range(1, 51)
    ]
    holding = retention_summary(records, [episode])
    return {
        "episodes": [episode],
        "summary": summarize([episode]),
        "step_records": records,
        "retention": holding,
        "episode_retention": {"7": holding},
    }


def test_compact_export_preserves_full_records_and_video_checks_holding_changes(
    tmp_path,
):
    result = fixture_result()
    geometry, holding, manifest = export_result(result, tmp_path, "teacher", [7])
    assert geometry["final_successes"] == 0
    assert holding["immediate_success_conditioned"]["conditioned_steps"] == 0
    assert holding["immediate_success_conditioned"]["arm_command_l2_mean"] is None
    decoded = [
        json.loads(line)
        for line in gzip.decompress(
            (tmp_path / manifest["steps"]).read_bytes()
        ).splitlines()
    ]
    assert decoded == result["step_records"]
    assert "trajectory" not in json.loads((tmp_path / "teacher.jsonl").read_text())
    verify_video(result, result, [7])
    changed = copy.deepcopy(result)
    changed["step_records"][20]["teacher_command"][7] = -1
    with pytest.raises(ValueError, match="holding records"):
        verify_video(result, changed, [7])


def test_calibration_recomputes_chosen_ratio_after_ceiling_hit():
    calibration = {
        "coefficient": 10.0,
        "ratios": [20, 25, 30, 35],
        "method": {"coefficient_clip": [0.01, 10], "target_gradient_ratio": 0.5},
        "probes": [
            {
                "ppo_actor_gradient_norm": pg,
                "teacher_actor_gradient_norm": 1.0,
                "scaled_teacher_gradient_norm": 0.0,
            }
            for pg in [40, 50, 60, 70]
        ],
    }
    derived = calibration_analysis(calibration)
    assert derived["coefficient_hit_upper_limit"]
    assert derived["unclipped_median_coefficient"] == 27.5
    assert derived["chosen_ratio_median"] == pytest.approx((0.2 + 1 / 6) / 2)
    assert derived["chosen_ratio_median"] != 0.5


def geometry_fixture():
    geometry = {"teacher": {"final_successes": 20}}
    for seed, counts in [
        (42, (20, 25, 30, 27)),
        (1042, (21, 26, 32, 27)),
        (2042, (20, 22, 30, 24)),
    ]:
        for suffix, count in zip(
            ["baseline_final", "anchored_final", "baseline_best", "anchored_best"],
            counts,
            strict=True,
        ):
            geometry[f"seed{seed}_{suffix}"] = {"final_successes": count}
    return geometry


def test_promising_rule_requires_two_paired_margins_and_less_collapse():
    geometry = geometry_fixture()
    assert promising_criterion(geometry)["promising_rule_met"]
    geometry["seed42_anchored_final"]["final_successes"] = 24
    assert not promising_criterion(geometry)["promising_rule_met"]
    geometry = geometry_fixture()
    for seed in [42, 1042, 2042]:
        geometry[f"seed{seed}_anchored_best"]["final_successes"] = 50
    assert not promising_criterion(geometry)["promising_rule_met"]


def test_command_deltas_and_grasp_associations_have_distinct_denominators():
    result = fixture_result()
    rows = result["step_records"][:3]
    for index, command in enumerate([0.1, 0.5, -0.5]):
        rows[index]["command"][7] = command
        rows[index]["after_first_success"] = index > 0
    rows[1]["pre"].update(success=True, grasped=True)
    rows[1]["post"].update(success=False, grasped=False)
    result["episodes"][0].update(success_once=True)
    derived = trace_associations(rows, result["episodes"])
    assert derived["all_steps"]["gripper_command_delta_pairs"] == 2
    assert derived["after_first_success"]["steps"] == 2
    assert derived["after_first_success"][
        "gripper_command_delta_abs_mean"
    ] == pytest.approx(0.7)
    assert derived["immediate_pre_success"]["steps"] == 1
    assert derived["immediate_pre_success"][
        "gripper_command_delta_abs_mean"
    ] == pytest.approx(0.4)
    assert derived["success_loss_and_grasp_loss_same_step"] == 1
    assert derived["ever_successful_static_displaced_final_failures"] == 1
    assert immediate_success_summary(rows)["conditioned_steps"] == 1


def test_native_plots_accept_missing_conditioned_subsets_and_no_optimizer_step(
    tmp_path,
):
    geometry = geometry_fixture()
    criterion = promising_criterion(geometry)
    results, validation, objective, diagnostics = {}, {}, {}, {}
    for seed in [42, 1042, 2042]:
        validation[str(seed)], objective[str(seed)], diagnostics[str(seed)] = {}, {}, {}
        for arm in ["baseline", "anchored"]:
            results[f"seed{seed}_{arm}_final"] = fixture_result()
            validation[str(seed)][arm] = [
                {"timesteps": 0, "success_at_end_rate": 0.4},
                {"timesteps": 1024, "success_at_end_rate": 0.3},
            ]
            objective[str(seed)][arm] = {
                "updates": [
                    {
                        "timesteps": 1024,
                        "auxiliary_loss_mean_scored_minibatches": 0.0,
                        "probe": {
                            "scaled_teacher_gradient_norm": 0,
                            "ppo_actor_gradient_norm": 1,
                            "gradient_cosine": None,
                            "exploration_variance": [0.0025] * 8,
                            "deterministic_mean_action_variance": [0.1] * 8,
                        },
                    }
                ]
            }
            diagnostics[str(seed)][arm] = {
                "updates": [{"timesteps": 1024, "optimizer_steps": 0}]
            }
    plots(tmp_path, results, geometry, criterion, validation, objective, diagnostics)
    for name in ["outcomes", "validation", "holding", "objective", "adam", "drift"]:
        assert (tmp_path / f"{name}.png").stat().st_size > 10000
