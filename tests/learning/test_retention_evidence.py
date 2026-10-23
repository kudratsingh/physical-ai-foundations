"""Independent audits of actor-retention data, geometry and gradient composition."""

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
    Path(__file__).resolve().parents[2] / "learning/maniskill/evidence/actor_retention"
)
SEEDS = [42, 1042, 2042]
ARMS = ["baseline", "anchored"]
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
    assert [e["seed"] for e in episodes] == list(range(240000, 240050))
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
        expected = (
            load("calibration_frozen.json")["coefficient"] if arm == "anchored" else 0.0
        )
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
        "calibration_sha256",
    ]:
        assert initial[0][key] == initial[1][key]


def test_calibration_receipt_provenance_and_all_saved_hashes():
    p = load("protocol.json")
    c = load("calibration.json")
    f = load("calibration_frozen.json")
    proof = load("provenance.json")
    assert p["training_seeds"] == SEEDS and p["validation_seeds"] == list(
        range(230000, 230040)
    )
    assert p["test_seeds"] == list(range(240000, 240050))
    assert c["actual_timesteps"] == 8192 and c["pilot_discarded"]
    ratios = [
        0.5 * r["ppo_actor_gradient_norm"] / r["teacher_actor_gradient_norm"]
        for r in c["probes"]
    ]
    assert ratios == c["ratios"]
    assert (
        c["coefficient"]
        == f["coefficient"]
        == min(10.0, max(0.01, statistics.median(ratios)))
    )
    assert (
        f["sha256"]
        == hashlib.sha256((ROOT / "calibration.json").read_bytes()).hexdigest()
    )
    assert (
        f["protocol_sha256"]
        == proof["protocol_sha256"]
        == hashlib.sha256((ROOT / "protocol.json").read_bytes()).hexdigest()
    )
    assert (
        proof["all_six_final_budgets_verified_before_evaluation"]
        and proof["all_three_first_performance_rollout_summaries_match"]
    )
    for name, digest in p["sources"].items():
        assert (
            hashlib.sha256((ROOT / "launch_sources" / name).read_bytes()).hexdigest()
            == digest
        )
    receipt = load("first_rollout/receipt.json")
    assert (
        receipt["passed"]
        and receipt["all_arrays_exactly_equal"]
        and receipt["workers"] == 4
        and receipt["transitions_per_arm"] == 1024
    )
    assert receipt["timeouts_per_arm"] == {"raw": 20, "bounded": 20}
    assert (
        hashlib.sha256(
            (ROOT / "first_rollout" / receipt["actual_records"]).read_bytes()
        ).hexdigest()
        == receipt["records_sha256"]
    )
    histories = lines("first_rollout/first_rollout.jsonl.gz")
    a = [
        {k: v for k, v in r.items() if k != "arm"}
        for r in histories
        if r["arm"] == "raw"
    ]
    b = [
        {k: v for k, v in r.items() if k != "arm"}
        for r in histories
        if r["arm"] == "bounded"
    ]
    assert len(a) == len(b) == 257 and a == b
    traces = load("trace_manifest.json")
    assert {e["condition"] for e in traces} == set(NAMES)
    for entry in traces:
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
            v["seed"] in [240000, 240001] and v["all_episode_and_holding_records_match"]
        )
        assert (
            hashlib.sha256((ROOT / v["file"]).read_bytes()).hexdigest() == v["sha256"]
        )


def test_predeclared_criterion_from_independent_final_outcomes():
    counts = {n: sum(e["success_at_end"] for e in lines(f"{n}.jsonl")) for n in NAMES}
    result = load("criterion.json")
    passes = 0
    bc = []
    ac = []
    for seed in SEEDS:
        b = counts[f"seed{seed}_baseline_final"]
        a = counts[f"seed{seed}_anchored_final"]
        passes += a >= counts["teacher"] + 5 and a >= b + 5
        bc.append(counts[f"seed{seed}_baseline_best"] - b)
        ac.append(counts[f"seed{seed}_anchored_best"] - a)
    assert result["training_seed_margin_passes"] == passes
    assert result["mean_baseline_selected_minus_final_successes"] == statistics.mean(bc)
    assert result["mean_anchored_selected_minus_final_successes"] == statistics.mean(ac)
    assert result["promising_rule_met"] == (
        passes >= 2 and statistics.mean(ac) < statistics.mean(bc)
    )
