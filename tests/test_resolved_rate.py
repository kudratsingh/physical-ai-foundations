"""Behavioral checks for the closed-loop differential controller."""
import math

import numpy as np
import pytest

from planar_arm import PlanarArm
from resolved_rate import controller_command, control_payload


def rollout(q=(30, 60), target=(1.1, 0.45), **settings):
    errors = []
    for _ in range(375):
        p = control_payload(*q, *target, **settings)
        errors.append(p["command"]["error_norm"])
        q = p["pose"]["angles_deg"]
        assert np.max(np.abs(p["applied_joint_rates"])) <= 1.5 + 1e-12
        if p["command"]["status"] != "tracking":
            break
    return p, errors


@pytest.mark.parametrize("method", ["dls", "pinv"])
def test_reaches_target_with_bounded_joint_rates(method):
    p, errors = rollout(method=method)
    assert p["command"]["status"] == "reached"
    assert errors[-1] <= 0.01
    assert all(b <= a + 1e-10 for a, b in zip(errors, errors[1:]))


@pytest.mark.parametrize("lengths", [(0.5, 1), (1, 1), (1.5, 0.25)])
def test_feedback_converges_with_other_geometries(lengths):
    arm = PlanarArm(*lengths)
    target = arm.joint_positions(0.5, 1.1)[-1]
    p, _ = rollout(q=(20, 50), target=target, l1=lengths[0], l2=lengths[1])
    assert p["command"]["status"] == "reached"


def test_dls_is_regularized_least_squares_solution_before_limits():
    arm = PlanarArm()
    q = [0.3, -0.9]
    target = arm.joint_positions(*q)[-1] + [0.02, 0.01]
    c = controller_command(arm, q, target, damping=0.2)
    J = arm.jacobian(*q)
    expected = np.linalg.solve(J.T@J + 0.2**2*np.eye(2), J.T@c["requested_velocity"])
    assert np.allclose(c["joint_rates"], expected)
    assert np.allclose(c["achieved_velocity"], J@c["joint_rates"])


def test_near_singular_damping_reduces_raw_rate_amplification():
    arm, q = PlanarArm(), [0, 1e-4]
    target = arm.joint_positions(*q)[-1] - [0.1, 0]
    plain = controller_command(arm, q, target, method="pinv")
    damped = controller_command(arm, q, target, method="dls")
    assert np.linalg.norm(plain["raw_joint_rates"]) > 1000
    assert np.linalg.norm(damped["raw_joint_rates"]) < 1
    assert plain["rate_limited"]
    assert np.max(np.abs(plain["joint_rates"])) <= 1.5


def test_exact_radial_singular_trap_is_reported_without_fake_escape():
    p = control_payload(0, 0, 1.25, 0)
    assert p["command"]["reachable"]
    assert p["command"]["status"] == "stalled"
    assert np.allclose(p["pose"]["angles_deg"], [0, 0])
    assert p["command"]["error_norm"] == 0.5


@pytest.mark.parametrize("target", [(0, 0), (4, 0)])
def test_unreachable_target_does_not_move_arm(target):
    p = control_payload(30, 60, *target)
    assert p["command"]["status"] == "unreachable"
    assert np.allclose(p["pose"]["angles_deg"], [30, 60])
    assert p["command"]["joint_rates"] == [0, 0]


def test_zero_dt_is_preview_only_and_returned_velocity_matches_returned_pose():
    for dt in (0, 0.04):
        p = control_payload(30, 60, 1.1, 0.45, dt=dt)
        if dt == 0:
            assert np.allclose(p["pose"]["angles_deg"], [30, 60])
        J = np.array(p["pose"]["differential"]["jacobian"])
        assert np.allclose(J@p["command"]["joint_rates"], p["command"]["achieved_velocity"])


def test_already_reached_target_has_zero_rates():
    p = control_payload(0, 90, 1, 0.75)
    assert p["command"]["status"] == "reached"
    assert p["command"]["joint_rates"] == [0, 0]


@pytest.mark.parametrize("dt", [-1, 0.06, math.nan])
def test_invalid_step_is_rejected(dt):
    with pytest.raises(ValueError):
        control_payload(30, 60, 1, 0.75, dt=dt)


@pytest.mark.parametrize("settings", [{"gain":0}, {"damping":-1}, {"method":"bad"},
                                      {"max_joint_speed":math.inf}, {"max_hand_speed":0}])
def test_invalid_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        controller_command(PlanarArm(), [0, 1], [1, 1], **settings)


def test_hand_speed_cap_and_joint_scaling_preserve_direction():
    c = controller_command(PlanarArm(), [0.2, 0.01], [0.5, 0.5], method="pinv")
    assert c["hand_limited"] and c["rate_limited"]
    assert np.isclose(np.linalg.norm(c["requested_velocity"]), 0.8)
    raw, rates = np.array(c["raw_joint_rates"]), np.array(c["joint_rates"])
    assert np.allclose(rates, raw*(1.5/np.max(np.abs(raw))))
