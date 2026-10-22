"""Observation-only decision boundary and physical state schema contracts."""

import inspect
import numpy as np
import pytest
from stable_baselines3 import PPO

from learning.maniskill.experiment import Config, make_env
from learning.maniskill.observation_stop_gate import (
    ObservationStopGate,
    placed_from_observation,
)
from learning.maniskill.placement_diagnostic import (
    audit_observation,
    placement_rollouts,
)
from learning.maniskill.stopping_diagnostic import state


def test_observation_only_gate_boundary_latch_and_preserved_gripper():
    # This test constructs only arrays. No simulator, info dictionary or actor
    # is passed to the gate, and unrelated observation fields cannot affect it.
    obs = np.zeros(42, dtype=np.float32)
    obs[39] = np.float32(0.025)
    assert placed_from_observation(obs)
    changed = obs.copy()
    changed[:39] = np.linspace(-50, 50, 39)
    assert placed_from_observation(changed)
    changed[39] = np.nextafter(np.float32(0.025), np.float32(1))
    assert not placed_from_observation(changed)
    gate = ObservationStopGate()
    original = np.linspace(-0.9, 0.9, 8, dtype=np.float32)
    action, active, _ = gate.action(changed, original, 0)
    np.testing.assert_array_equal(action, original)
    assert not active
    action, active, _ = gate.action(obs, original, 1)
    assert active and gate.first_placement_state_step == 1
    np.testing.assert_array_equal(action[:7], np.zeros(7))
    assert action[7] == original[7]
    action, active, placed = gate.action(changed, original, 2)
    assert active and not placed
    assert action[7] == original[7]
    np.testing.assert_array_equal(original, np.linspace(-0.9, 0.9, 8, dtype=np.float32))
    assert list(inspect.signature(gate.action).parameters) == [
        "observation",
        "policy_action",
        "state_step",
    ]
    with pytest.raises(ValueError, match="raw42"):
        placed_from_observation(np.zeros(35))


def test_real_pickcube_observation_schema_and_trace_measurements():
    env = make_env(Config(env_id="PickCube-v1"))
    try:
        obs, info = env.reset(seed=131000)
        assert audit_observation(obs, state(env), info["is_grasped"])
        broken = obs.copy()
        broken[26] += 0.01
        with pytest.raises(ValueError, match="schema"):
            audit_observation(broken, state(env), info["is_grasped"])
        model = PPO("MlpPolicy", env, n_steps=16, batch_size=16, seed=131, device="cpu")
        result = placement_rollouts(model, [131000])
        assert result[0]["steps"] == 50
        for record in result[0]["records"]:
            assert record["remaining_steps"] == 50 - record["step"]
            assert record["arm_command_l2"] == pytest.approx(
                np.linalg.norm(record["action"][:7])
            )
            assert (
                record["distance_change"]
                == record["cube_goal_distance"] - record["pre_distance"]
            )
    finally:
        env.close()
