"""Independent geometry, bootstrap arithmetic and actual-gradient evidence audit."""

import hashlib
import json
import math
import statistics
from pathlib import Path

import pytest

from learning.maniskill.metrics import summarize
from learning.maniskill.selection import holding_score

ROOT = Path(__file__).resolve().parents[2] / "learning/maniskill/evidence/critic_bound"
NAMES = ["teacher", "raw_final", "raw_best", "bounded_final", "bounded_best"]


def load(name):
    return json.loads((ROOT / name).read_text())


def episodes(name):
    return [json.loads(s) for s in (ROOT / f"{name}.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("name", NAMES)
def test_fresh_outcomes_from_geometry(name):
    data = episodes(f"test_{name}")
    assert [e["seed"] for e in data] == list(range(190000, 190050))
    for e in data:
        d = math.dist(e["final_cube_position"], e["final_goal_position"])
        v = max(map(abs, e["final_arm_qvel"]))
        assert e["steps"] == 50
        assert d == pytest.approx(e["final_goal_distance"], abs=1e-7)
        assert e["success_at_end"] == (d <= 0.025 and v <= 0.2)
    assert summarize(data) == load("comparison.json")[name]


@pytest.mark.parametrize("arm", ["raw", "bounded"])
def test_actual_first_step_gradient_clipping_and_bootstrap_arithmetic(arm):
    points = load(f"{arm}_diagnostics.json")["updates"]
    assert [p["timesteps"] for p in points] == list(range(1024, 262145, 1024))
    timeouts = 0
    for p in points:
        raw = p["raw"]
        r = raw["rewards"]
        t = raw["terminal_values"]
        b = p["buffer_rewards_after_bootstrap"]
        assert (
            r["count"]
            == b["count"]
            == p["values"]["count"]
            == p["returns"]["count"]
            == 1024
        )
        assert 0 <= r["min"] <= r["max"] <= 1.000001
        assert raw["termination_count"] == 0
        timeouts += raw["timeout_count"]
        assert t["count"] == raw["timeout_count"]
        correction = b["mean"] * 1024 - r["mean"] * 1024
        assert correction == pytest.approx(
            0.99 * t["mean"] * t["count"], rel=3e-6, abs=1e-4
        )
        if arm == "bounded":
            for field in ["values", "returns"]:
                assert p[field]["min"] >= -100.001 and p[field]["max"] <= 100.001
            assert t["min"] >= -100.001 and t["max"] <= 100.001
        assert 0 <= p["optimizer_steps"] <= 32
        if p["optimizer_steps"] == 0:
            assert "gradient_norm_after_clip" not in p
            continue
        before = p["gradient_norm_before_clip"]
        after = p["gradient_norm_after_clip"]
        norm = math.sqrt(sum(v * v for v in before.values()))
        assert norm == pytest.approx(p["global_norm_before_clip"], rel=1e-14)
        scale = min(1.0, 0.5 / (norm + 1e-6))
        for group in ["actor", "critic"]:
            assert after[group] == pytest.approx(
                before[group] * scale, rel=2e-5, abs=1e-10
            )
            assert p["first_step_parameter_delta_l2"][group] >= 0
        assert math.sqrt(sum(v * v for v in after.values())) <= 0.500002
    assert timeouts == 5240
    windows = load(f"{arm}_training_metrics.json")["windows"]
    assert sum(w["transitions"] for w in windows) == 262144
    assert sum(w.get("episodes", 0) for w in windows) == 5240
    validation = load(f"{arm}_validation.json")
    assert [p["timesteps"] for p in validation] == list(range(0, 262145, 32768))
    assert load(f"{arm}_selected.json") == max(validation, key=holding_score)


def test_actual_checkpoint_provenance_transfer_sources_and_videos():
    verified = load("verified.json")
    protocol = load("protocol.json")
    assert verified["complete"] and verified["all_final_budgets_verified"]
    assert (
        verified["actual_initial_actors_equal_teacher"]
        and verified["actual_initial_parameters_match"]
    )
    assert (
        episodes("preflight_teacher")
        == episodes("preflight_raw")
        == episodes("preflight_bounded")
    )
    assert [e["seed"] for e in episodes("preflight_teacher")] == list(
        range(180000, 180010)
    )
    assert verified["teacher_sha256"] == protocol["teacher_sha256"]
    assert (
        verified["protocol_sha256"]
        == hashlib.sha256((ROOT / "protocol.json").read_bytes()).hexdigest()
    )
    initial = [load(f"{a}_initialization.json") for a in ["raw", "bounded"]]
    for key in ["canonical_fingerprint", "rng_fingerprint", "optimizer_empty"]:
        assert initial[0][key] == initial[1][key]
    assert initial[0]["optimizer_empty"] is True
    for arm in ["raw", "bounded"]:
        assert (
            verified["actual_checkpoints"][arm]["final"]["recipe"]["timesteps"]
            == 262144
        )
        assert load(f"{arm}_training.json")["complete"]
    for name, digest in protocol["sources"].items():
        assert (
            hashlib.sha256((ROOT / "launch_sources" / name).read_bytes()).hexdigest()
            == digest
        )
    manifest = load("video_manifest.json")
    assert {(v["policy"], v["seed"]) for v in manifest} == {
        (p, s)
        for p in ["teacher", "raw_final", "bounded_final"]
        for s in [190000, 190001]
    }
    for item in manifest:
        assert (
            hashlib.sha256((ROOT / item["file"]).read_bytes()).hexdigest()
            == item["sha256"]
        )
        video = next(
            e for e in episodes(f"video_{item['policy']}") if e["seed"] == item["seed"]
        )
        expected = next(
            e for e in episodes(f"test_{item['policy']}") if e["seed"] == item["seed"]
        )
        assert {k: v for k, v in video.items() if k != "trajectory"} == {
            k: v for k, v in expected.items() if k != "trajectory"
        }


def test_offpolicy_audit_predictions_and_probe_selection():
    a = load("audit.json")
    assert a["probes"]["observation_count"] == 408
    assert a["probes"]["trajectory_ids"] == [161, 557, 546, 573, 405, 455, 100, 591]
    assert not a["historical_on_policy_gradient_norms"]["available"]
    assert len(a["probes"]["indices"]) == 408
    for checkpoints in a["checkpoints"].values():
        for c in checkpoints.values():
            predictions = c["predictions"]
            bound = c["theoretical_nonnegative_discounted_return_upper_bound"]
            assert len(predictions) == 408 and all(
                math.isfinite(v) for v in predictions
            )
            assert bound == pytest.approx(100.0)
            assert c["prediction_summary"]["median"] == statistics.median(predictions)
            assert (
                c["fraction_above_true_discounted_upper_bound"]
                == sum(v > bound for v in predictions) / 408
            )
            assert c["critic_clipped_normalized_absmax"] <= 10.0
    source = ROOT.parents[1] / "critic_audit.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == a["audit_source_sha256"]
