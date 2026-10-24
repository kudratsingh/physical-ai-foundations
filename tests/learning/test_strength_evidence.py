"""Independent audits of strength-ablation data, geometry and gradient composition."""

import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

import pytest

from learning.maniskill.metrics import summarize
from learning.maniskill.selection import holding_score

ROOT = (
    Path(__file__).resolve().parents[2]
    / "learning/maniskill/evidence/retention_strength"
)
SEEDS = [42, 1042, 2042]
ARMS = ["lambda10", "lambda25"]
NAMES = ["teacher"] + [
    f"seed{s}_{a}_{t}" for s in SEEDS for a in ARMS for t in ["final", "best"]
]


def load(path):
    return json.loads((ROOT / path).read_text())


def lines(path):
    p = ROOT / path
    data = (
        gzip.decompress(p.read_bytes()).decode() if p.suffix == ".gz" else p.read_text()
    )
    return [json.loads(s) for s in data.splitlines()]


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def average(v):
    return statistics.mean(v) if v else None


@pytest.mark.parametrize("name", NAMES)
def test_complete_heldout_geometry_actions_event_conditioning_and_holding(name):
    episodes = lines(f"{name}.jsonl")
    rows = lines(f"{name}_steps.jsonl.gz")
    h = load(f"{name}_holding.json")
    assert [e["seed"] for e in episodes] == list(range(250000, 250050))
    assert len(rows) == 2500
    assert summarize(episodes) == load("comparison.json")[name] == h["summary"]
    for e in episodes:
        steps = [r for r in rows if r["seed"] == e["seed"]]
        assert [r["step"] for r in steps] == list(range(1, 51))
        seen = False
        first = None
        for i, r in enumerate(steps):
            assert (
                len(r["observation"]) == 42
                and len(r["command"]) == len(r["teacher_command"]) == 8
            )
            assert all(
                math.isfinite(v)
                for v in r["observation"] + r["command"] + r["teacher_command"]
            )
            assert max(map(abs, r["command"] + r["teacher_command"])) <= 1
            assert r["after_first_success"] == seen
            for state in [r["pre"], r["post"]]:
                d = math.dist(state["cube"], state["goal"])
                speed = max(map(abs, state["arm_qvel"]))
                assert d == pytest.approx(state["cube_goal_distance"], abs=1e-7)
                assert speed == state["max_arm_speed"]
                assert state["placed"] == state["official_placed"] == (d <= 0.025)
                assert state["static"] == state["official_static"] == (speed <= 0.2)
                assert (
                    state["success"]
                    == state["geometric_success"]
                    == (d <= 0.025 and speed <= 0.2)
                )
            assert r["observation"][39:42] == pytest.approx(
                [g - c for g, c in zip(r["pre"]["goal"], r["pre"]["cube"])], abs=1e-7
            )
            assert r["gripper_delta"] == pytest.approx(
                [
                    a - b
                    for a, b in zip(r["post"]["gripper_qpos"], r["pre"]["gripper_qpos"])
                ],
                abs=1e-9,
            )
            assert r["teacher_arm_drift_l2"] == pytest.approx(
                norm(
                    [a - b for a, b in zip(r["command"][:7], r["teacher_command"][:7])]
                ),
                abs=1e-6,
            )
            assert r["teacher_gripper_drift_abs"] == pytest.approx(
                abs(r["command"][7] - r["teacher_command"][7]), abs=1e-7
            )
            if i:
                assert r["pre"] == steps[i - 1]["post"]
            if r["post"]["success"] and first is None:
                first = r["step"]
            seen |= r["post"]["success"]
        last = steps[-1]["post"]
        assert (
            e["steps"] == 50
            and e["success_once"] == seen
            and e["success_at_end"] == last["success"]
        )
        assert e["first_success_step"] == first
        assert e["return"] == sum(r["raw_reward"] for r in steps)
        assert e["final_goal_distance"] == pytest.approx(
            math.dist(e["final_cube_position"], e["final_goal_position"]), abs=1e-7
        )
        assert steps[-1]["truncated"] and not any(r["terminated"] for r in steps)
    after = [r for r in rows if r["after_first_success"]]
    immediate = [r for r in rows if r["pre"]["success"]]
    losses = [r for r in immediate if not r["post"]["success"]]
    assert h["retention"]["post_first_success_steps"] == len(after)
    assert h["retention"]["success_to_failure_transitions"] == len(losses)
    assert h["retention"]["geometry_official_mismatches"] == 0
    if after:
        assert h["retention"]["post_first_success_success_occupancy"] == pytest.approx(
            average([r["post"]["success"] for r in after])
        )
    else:
        assert h["retention"]["post_first_success_success_occupancy"] is None
    assert h["immediate_success_conditioned"]["conditioned_steps"] == len(immediate)
    assert h["immediate_success_conditioned"]["success_to_failure_transitions"] == len(
        losses
    )
    associations = h["trace_associations"]
    assert associations["success_loss_steps"]["steps"] == len(losses)
    assert associations["success_loss_and_grasp_loss_same_step"] == sum(
        r["pre"]["grasped"] and not r["post"]["grasped"] for r in losses
    )
    displaced = sum(
        e["success_once"]
        and not e["success_at_end"]
        and e["final_is_robot_static"]
        and e["final_goal_distance"] > 0.025
        for e in episodes
    )
    assert associations["ever_successful_static_displaced_final_failures"] == displaced


@pytest.mark.parametrize("seed", SEEDS)
def test_all_training_counts_selection_and_objective_gradients(seed):
    initial = []
    for arm in ARMS:
        base = Path(f"seed{seed}") / arm
        initial.append(load(base / "initialization.json"))
        expected = load("protocol.json")["coefficients"][arm]
        assert initial[-1]["coefficient"] == expected
        assert initial[-1]["optimizer_empty"] is True
        training = load(base / "training.json")
        assert training["complete"] and training["actual_timesteps"] == 262144
        validation = load(base / "validation.json")
        assert [r["timesteps"] for r in validation] == list(range(0, 262145, 32768))
        assert load(base / "selected.json") == max(validation, key=holding_score)
        windows = load(base / "training_metrics.json")["windows"]
        assert sum(w["transitions"] for w in windows) == 262144
        assert sum(w.get("episodes", 0) for w in windows) == 5240
        d = load(base / "diagnostics.json")["updates"]
        o = load(base / "retention.json")["updates"]
        assert (
            [r["timesteps"] for r in d]
            == [r["timesteps"] for r in o]
            == list(range(1024, 262145, 1024))
        )
        for before, obj in zip(d, o):
            assert obj["coefficient"] == expected
            q = obj["probe"]
            pg = q["ppo_actor_gradient_norm"]
            aux = q["teacher_actor_gradient_norm"]
            cos = q["gradient_cosine"] or 0.0
            assert q["scaled_teacher_gradient_norm"] == expected * aux
            assert q["teacher_action_loss"] == pytest.approx(
                q["arm_teacher_drift_mse_sum"] + q["gripper_teacher_drift_mse"],
                abs=1e-6,
            )
            assert (
                obj["scaled_auxiliary_loss_mean_scored_minibatches"]
                == expected * obj["auxiliary_loss_mean_scored_minibatches"]
            )
            assert len(q["exploration_variance"]) == 8 and all(
                v > 0 and math.isfinite(v) for v in q["exploration_variance"]
            )
            if before["optimizer_steps"]:
                composed = math.sqrt(
                    max(
                        0.0,
                        pg * pg + (expected * aux) ** 2 + 2 * expected * pg * aux * cos,
                    )
                )
                assert composed == pytest.approx(
                    before["gradient_norm_before_clip"]["actor"], rel=2e-5, abs=1e-8
                )
            assert (
                -100.001
                <= before["values"]["min"]
                <= before["values"]["max"]
                <= 100.001
            )
            assert (
                0
                <= before["raw"]["rewards"]["min"]
                <= before["raw"]["rewards"]["max"]
                <= 1
            )
    for key in [
        "canonical_fingerprint",
        "rng_fingerprint",
        "teacher_fingerprint",
        "protocol_sha256",
    ]:
        assert initial[0][key] == initial[1][key]


def test_declared_strengths_fresh_resets_and_checkpoint_provenance():
    p, f, proof = (
        load("protocol.json"),
        load("protocol_frozen.json"),
        load("provenance.json"),
    )
    assert p["coefficients"] == {"lambda10": 10.0, "lambda25": 25.0}
    assert p["training_seeds"] == SEEDS and p["validation_seeds"] == list(
        range(230000, 230040)
    )
    assert p["test_seeds"] == list(range(250000, 250050))
    assert p["video_seeds"] == [250000, 250001]
    assert set(p["test_seeds"]).isdisjoint(range(240000, 240050))
    assert (
        f["sha256"]
        == proof["protocol_sha256"]
        == hashlib.sha256((ROOT / "protocol.json").read_bytes()).hexdigest()
    )
    assert proof["all_six_final_budgets_verified_before_evaluation"]
    assert proof["all_three_first_performance_rollout_summaries_match"]
    assert proof["reference_used_for_evaluation"] is False
    for name, digest in p["sources"].items():
        assert (
            hashlib.sha256((ROOT / "launch_sources" / name).read_bytes()).hexdigest()
            == digest
        )
    reference = load("reference_pr18_selected/manifest.json")
    assert reference["models"] == p["historical_reference"]["selected_models"]
    assert reference["evaluation_in_this_study"] is False
    assert (
        hashlib.sha256(
            (ROOT / "reference_pr18_selected/manifest.json").read_bytes()
        ).hexdigest()
        == proof["reference_pr18_selected_manifest_sha256"]
    )
    for seed in SEEDS:
        paired = []
        for arm in ARMS:
            recipe = proof["actual_checkpoints"][str(seed)][arm]
            for tag in ["initial", "final", "best"]:
                r = recipe[tag]["recipe"]
                assert r["seed"] == seed and r["coefficient"] == p["coefficients"][arm]
                assert r["teacher_frozen"] and r["critic_bound"] == 100
                steps = (
                    0
                    if tag == "initial"
                    else 262144
                    if tag == "final"
                    else load(f"seed{seed}/{arm}/selected.json")["timesteps"]
                )
                assert r["num_timesteps"] == steps
                assert r["teacher_fingerprint"] == proof["teacher_fingerprint"]
            paired.append(load(f"seed{seed}/{arm}/diagnostics.json")["updates"][0])
        for field in [
            "values",
            "returns",
            "advantages",
            "buffer_rewards_after_bootstrap",
            "raw",
        ]:
            assert paired[0][field] == paired[1][field]
    assert len(proof["actual_checkpoints"]) == 3


def test_complete_trace_hashes_native_videos_and_all_displaced_cases():
    entries = load("trace_manifest.json")
    assert {e["condition"] for e in entries} == set(NAMES)
    for entry in entries:
        for path, key in [
            (entry["episodes"], "episodes_sha256"),
            (entry["steps"], "steps_sha256"),
        ]:
            assert (
                hashlib.sha256(
                    (ROOT / entry["directory"] / path).read_bytes()
                ).hexdigest()
                == entry[key]
            )
    videos = load("video_manifest.json")
    assert len(videos) == 14
    for v in videos:
        assert (
            v["seed"] in [250000, 250001] and v["all_episode_and_holding_records_match"]
        )
        assert (
            hashlib.sha256((ROOT / v["file"]).read_bytes()).hexdigest() == v["sha256"]
        )
    cases = load("trajectory_cases.json")
    # Agent's report includes all six final learners, not just favorable cases.
    for entry in cases:
        name = entry["condition"]
        episodes = lines(f"{name}.jsonl")
        expected = [
            e
            for e in episodes
            if e["success_once"] and e["final_goal_distance"] > 0.025
        ]
        assert [r["reset_seed"] for r in entry["cases"]] == [
            e["seed"] for e in expected
        ]
        for r, e in zip(entry["cases"], expected):
            assert r["final_static"] == e["final_is_robot_static"]
            assert r["final_distance_m"] == e["final_goal_distance"]
            assert r["first_success_step"] == e["first_success_step"]


def test_predeclared_strength_criterion_from_independent_outcomes():
    counts = {n: sum(e["success_at_end"] for e in lines(f"{n}.jsonl")) for n in NAMES}
    result = load("criterion.json")
    changes = {
        s: counts[f"seed{s}_lambda25_final"] - counts[f"seed{s}_lambda10_final"]
        for s in SEEDS
    }
    collapse = {
        a: statistics.mean(
            counts[f"seed{s}_{a}_best"] - counts[f"seed{s}_{a}_final"] for s in SEEDS
        )
        for a in ARMS
    }
    target = changes[2042] >= 5
    other = all(changes[s] >= -2 for s in [42, 1042])
    stable = collapse["lambda25"] <= collapse["lambda10"]
    assert result["target_training_seed"] == 2042
    assert result["target_gain_at_least_five"] == target
    assert result["each_other_seed_loses_at_most_two"] == other
    assert result["mean_signed_collapse_not_worse"] == stable
    assert result["mean_signed_selected_minus_final"] == collapse
    assert result["promising_rule_met"] == (target and other and stable)


def test_all_paired_outcomes_preserve_episode_identities():
    actual = {
        n: {e["seed"]: e["success_at_end"] for e in lines(f"{n}.jsonl")} for n in NAMES
    }
    paired = load("paired_outcomes.json")
    assert [r["training_seed"] for r in paired["rows"]] == SEEDS
    for row in paired["rows"]:
        for key, pair in row.items():
            if key == "training_seed":
                continue
            left, right = actual[pair["reference"]], actual[pair["candidate"]]
            gain = sorted(s for s in left if right[s] and not left[s])
            loss = sorted(s for s in left if left[s] and not right[s])
            assert pair["new_success_reset_seeds"] == gain
            assert pair["lost_success_reset_seeds"] == loss
            assert pair["new_successes"] == len(gain)
            assert pair["reference_successes_lost"] == len(loss)
            assert pair["retained_successes"] == sum(left[s] and right[s] for s in left)
            assert pair["paired_resets"] == 50
