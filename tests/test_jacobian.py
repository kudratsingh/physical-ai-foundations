"""Validate differential kinematics against finite differences and MuJoCo."""
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from planar_arm import PlanarArm, pose_payload


@pytest.mark.parametrize("lengths", [(1, 0.75), (0.25, 1.5), (1.5, 0.25), (1, 1)])
def test_jacobian_matches_central_difference(lengths):
    arm = PlanarArm(*lengths)
    h = 1e-6
    for q in np.random.default_rng(314).uniform(-math.pi, math.pi, (60, 2)):
        numerical = np.column_stack([
            (arm.joint_positions(*(q + h * np.eye(2)[i]))[-1]
             - arm.joint_positions(*(q - h * np.eye(2)[i]))[-1]) / (2 * h)
            for i in range(2)
        ])
        assert np.allclose(arm.jacobian(*q), numerical, rtol=1e-8, atol=1e-9)


def test_known_straight_pose_velocities_and_null_direction():
    arm = PlanarArm()
    J = arm.jacobian(0, 0)
    assert np.allclose(J, [[0, 0], [1.75, 0.75]], atol=1e-14)
    assert np.allclose(J @ [1, 0], [0, 1.75])
    assert np.allclose(J @ [0.75, -1.75], [0, 0], atol=1e-14)


def test_determinant_and_svd_describe_same_velocity_ellipse():
    arm = PlanarArm()
    for q in np.random.default_rng(8).uniform(-math.pi, math.pi, (60, 2)):
        d = arm.differential_kinematics(*q)
        J, axes, sigma = np.array(d["jacobian"]), np.array(d["principal_directions"]).T, np.array(d["singular_values"])
        expected = arm.l1 * arm.l2 * math.sin(q[1])
        assert np.isclose(d["determinant"], expected, atol=1e-14)
        assert np.isclose(d["manipulability"], abs(expected), atol=1e-14)
        assert np.allclose(axes @ np.diag(sigma**2) @ axes.T, J @ J.T, atol=1e-14)
        assert np.isclose(d["condition"], sigma[0] / sigma[1])


@pytest.mark.parametrize("q2", [0, math.pi, -math.pi])
@pytest.mark.parametrize("lengths", [(1, 0.75), (1, 1), (0.5, 1)])
def test_extended_and_folded_configurations_have_rank_one(q2, lengths):
    result = PlanarArm(*lengths).differential_kinematics(0.4, q2)
    assert result["singular"] and result["rank"] == 1
    assert result["condition"] is None
    assert result["singular_values"][1] < 1e-14


def test_near_singular_is_not_misreported_as_exactly_singular():
    result = PlanarArm().differential_kinematics(0.3, 1e-5)
    assert not result["singular"] and result["rank"] == 2
    assert result["condition"] > 100_000
    assert result["singular_values"][1] > 0


def test_velocity_prediction_converges_as_time_step_shrinks():
    arm = PlanarArm()
    q, dq = np.array([0.4, -1.1]), np.array([0.7, -0.3])
    p, velocity = arm.joint_positions(*q)[-1], arm.jacobian(*q) @ dq
    errors = [np.linalg.norm(arm.joint_positions(*(q + dq*dt))[-1] - (p + velocity*dt))
              for dt in (0.02, 0.01, 0.005)]
    assert 3.9 < errors[0]/errors[1] < 4.1
    assert 3.9 < errors[1]/errors[2] < 4.1


def test_matches_independent_mujoco_site_jacobian():
    model = mujoco.MjModel.from_xml_path(str(Path(__file__).resolve().parents[1] / "mujoco/two_link.xml"))
    data = mujoco.MjData(model)
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for q in np.random.default_rng(55).uniform(-math.pi, math.pi, (40, 2)):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        mujoco.mj_jacSite(model, data, jacp, jacr, model.site("hand").id)
        assert np.allclose(PlanarArm().jacobian(*q), jacp[:2], atol=1e-12)
        assert np.allclose(jacr[2], [1, 1])


@pytest.mark.parametrize("q", [(math.nan, 0), (0, math.inf)])
def test_nonfinite_angles_are_rejected(q):
    with pytest.raises(ValueError):
        PlanarArm().jacobian(*q)


def test_payload_uses_radians_for_derivatives_and_reports_json_safe_singularities():
    p = pose_payload(0, 90)
    assert np.allclose(p["differential"]["jacobian"], [[-0.75, -0.75], [1, 0]])
    assert not p["differential"]["singular"]
    assert pose_payload(0, 180)["differential"]["condition"] is None
