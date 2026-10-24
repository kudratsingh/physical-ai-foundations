"""No simulator: fixed-strength guards, criterion boundaries and native plots."""

import copy
import gzip
import json

import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from learning.maniskill import strength_compare
from learning.maniskill.behavior_cloning import BehaviorPolicy
from learning.maniskill.retention_experiment import build
from learning.maniskill.retention_ppo import RetentionPPO
from learning.maniskill.retention_report import export_result
from learning.maniskill.rng import preserve_rng
from learning.maniskill.strength_report import (
    paired_outcomes,
    plots,
    promising_criterion,
    trajectory_cases,
)
from learning.maniskill.test.test_retention_evidence_tools import (
    ShapesOnly,
    fixture_result,
)


def geometry_fixture():
    geometry = {"teacher": {"final_successes": 26}}
    for seed, counts in [
        (42, (30, 28, 35, 33)),
        (1042, (31, 29, 36, 34)),
        (2042, (20, 25, 30, 35)),
    ]:
        for suffix, count in zip(
            ["lambda10_final", "lambda25_final", "lambda10_best", "lambda25_best"],
            counts,
            strict=True,
        ):
            geometry[f"seed{seed}_{suffix}"] = {"final_successes": count}
    return geometry


def test_fixed_positive_strength_checkpoint_guards(tmp_path):
    with preserve_rng():
        torch.set_num_threads(1)
        teacher = BehaviorPolicy(mean=np.zeros(42), std=np.ones(42))
        env = DummyVecEnv([ShapesOnly] * 4)
        try:
            model = build(env, teacher, 42, 25.0)
            model.save(tmp_path / "initial")
            loaded = RetentionPPO.load(tmp_path / "initial.zip", device="cpu")
            assert strength_compare.check_model(
                loaded, teacher, 42, 25.0, 0, initial=True
            )["teacher_frozen"]
            with pytest.raises(ValueError, match="coefficient"):
                strength_compare.check_model(loaded, teacher, 42, 10.0, 0)
            with pytest.raises(ValueError, match="budget"):
                strength_compare.check_model(loaded, teacher, 42, 25.0, 262144)
            with torch.no_grad():
                loaded.teacher.net[0].bias[0] += 1
            with pytest.raises(ValueError, match="teacher differs"):
                strength_compare.check_model(loaded, teacher, 42, 25.0, 0)
        finally:
            env.close()


def test_incomplete_guards_stop_before_any_evaluation(monkeypatch, tmp_path):
    def fail_guard(*args):
        raise ValueError("Incomplete six-condition guard")

    def forbidden(*args, **kwargs):
        raise AssertionError("Held-out environment must not be entered")

    monkeypatch.setattr(strength_compare, "verify_checkpoints", fail_guard)
    monkeypatch.setattr(strength_compare, "evaluate_retention", forbidden)
    with pytest.raises(ValueError, match="Incomplete"):
        strength_compare.compare(tmp_path, tmp_path / "teacher.pt")


def test_predeclared_criterion_exact_boundaries():
    geometry = geometry_fixture()
    result = promising_criterion(geometry)
    assert result["promising_rule_met"]  # +5 target, -2 each other, equal collapse.
    assert (
        result["mean_signed_selected_minus_final"]["lambda10"]
        == result["mean_signed_selected_minus_final"]["lambda25"]
    )
    for key, value in [
        ("seed2042_lambda25_final", 24),
        ("seed42_lambda25_final", 27),
        ("seed1042_lambda25_final", 28),
        ("seed42_lambda25_best", 34),
    ]:
        changed = copy.deepcopy(geometry)
        changed[key]["final_successes"] = value
        assert not promising_criterion(changed)["promising_rule_met"]


def test_paired_outcomes_preserve_reset_identities_and_reject_unpaired_inputs():
    def episodes(successes):
        return {
            "episodes": [
                {"seed": s, "success_at_end": s in successes} for s in [3, 1, 2]
            ]
        }

    results = {"teacher": episodes({1, 3})}
    for seed in [42, 1042, 2042]:
        for arm, final, best in [
            ("lambda10", {1}, {2, 3}),
            ("lambda25", {2, 3}, {1, 3}),
        ]:
            results[f"seed{seed}_{arm}_final"] = episodes(final)
            results[f"seed{seed}_{arm}_best"] = episodes(best)
    row = paired_outcomes(results)["rows"][0]
    compared = row["lambda25_vs_lambda10_final"]
    assert compared["new_success_reset_seeds"] == [2, 3]
    assert compared["lost_success_reset_seeds"] == [1]
    assert compared["new_successes"] == 2 and compared["reference_successes_lost"] == 1
    assert row["lambda10_final_vs_selected"]["new_success_reset_seeds"] == [1]
    assert row["lambda10_final_vs_selected"]["lost_success_reset_seeds"] == [2, 3]
    results["seed42_lambda25_final"]["episodes"].pop()
    with pytest.raises(ValueError, match="resets differ"):
        paired_outcomes(results)


def test_strength_trace_compression_preserves_actual_records(tmp_path):
    result = fixture_result()
    _, holding, manifest = export_result(result, tmp_path, "seed42_lambda25_final", [7])
    rows = [
        json.loads(line)
        for line in gzip.decompress(
            (tmp_path / manifest["steps"]).read_bytes()
        ).splitlines()
    ]
    assert rows == result["step_records"]
    assert holding["immediate_success_conditioned"]["conditioned_steps"] == 0
    assert holding["retention"]["post_first_success_arm_command_l2_mean"] is None


def test_trajectory_selector_includes_every_qualifying_episode_in_reset_order():
    result = fixture_result()
    e = result["episodes"][0]
    e.update(success_once=True, first_success_step=10)
    other = copy.deepcopy(e)
    other["seed"] = 2
    excluded = copy.deepcopy(e)
    excluded.update(seed=9, final_goal_distance=0.01)
    result["episodes"] = [e, excluded, other]
    result["step_records"] += [
        {**r, "seed": 2} for r in copy.deepcopy(result["step_records"])
    ]
    cases = trajectory_cases(result)
    assert [c["reset_seed"] for c in cases] == [2, 7]
    assert all(len(c["steps"]) == 50 for c in cases)


def test_native_strength_plots_with_empty_subsets_and_optimizer_gaps(tmp_path):
    geometry = geometry_fixture()
    validation, objective, diagnostics, results = {}, {}, {}, {}
    for seed in [42, 1042, 2042]:
        validation[str(seed)], objective[str(seed)], diagnostics[str(seed)] = {}, {}, {}
        for arm in ["lambda10", "lambda25"]:
            results[f"seed{seed}_{arm}_final"] = fixture_result()
            validation[str(seed)][arm] = [{"timesteps": 0, "success_at_end_rate": 0.5}]
            objective[str(seed)][arm] = {
                "updates": [
                    {
                        "timesteps": 1024,
                        "auxiliary_loss_mean_scored_minibatches": 0.001,
                        "probe": {
                            "scaled_teacher_gradient_norm": 0,
                            "ppo_actor_gradient_norm": 1,
                            "exploration_variance": [0.002] * 8,
                            "deterministic_mean_action_variance": [0.01] * 8,
                        },
                    }
                ]
            }
            diagnostics[str(seed)][arm] = {
                "updates": [{"timesteps": 1024, "optimizer_steps": 0}]
            }
    cases = plots(
        tmp_path,
        results,
        geometry,
        promising_criterion(geometry),
        validation,
        objective,
        diagnostics,
    )
    assert len(cases) == 6 and all(not c["cases"] for c in cases)
    for name in ["outcomes", "validation", "holding", "objectives", "drift"]:
        assert (tmp_path / f"{name}.png").stat().st_size > 10000
    assert all((tmp_path / c["plots"][0]).is_file() for c in cases)
