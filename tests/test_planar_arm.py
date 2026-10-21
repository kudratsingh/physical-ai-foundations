"""Check FK against known geometry, transform composition, and MuJoCo sites."""
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from planar_arm import PlanarArm, pose_payload


@pytest.mark.parametrize("q,expected", [
    ((0, 0), [[0, 0], [1, 0], [1.75, 0]]),
    ((math.pi / 2, 0), [[0, 0], [0, 1], [0, 1.75]]),
    ((0, math.pi / 2), [[0, 0], [1, 0], [1, 0.75]]),
    ((0, math.pi), [[0, 0], [1, 0], [0.25, 0]]),
    ((math.pi, 0), [[0, 0], [-1, 0], [-1.75, 0]]),
])
def test_known_configurations(q, expected):
    assert np.allclose(PlanarArm().joint_positions(*q), expected, atol=1e-12)


def test_elbow_angle_is_relative_to_first_link():
    positions = PlanarArm().joint_positions(math.pi / 6, math.pi / 3)
    # Link 2 points at 90 degrees in the world, not at q2 = 60 degrees.
    assert np.allclose(positions[2] - positions[1], [0, 0.75], atol=1e-12)


def test_transform_chain_matches_analytic_fk():
    arm = PlanarArm(0.7, 1.3)
    for q1, q2 in np.random.default_rng(42).uniform(-math.pi, math.pi, (30, 2)):
        a, b, c = arm.transform_chain(q1, q2)
        T = arm.forward_kinematics(q1, q2)
        assert np.allclose(a @ b @ c, T, atol=1e-12)
        assert np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(T[:3, :3]), 1)


def test_link_lengths_and_workspace_bounds():
    arm = PlanarArm(0.7, 1.3)
    inner, outer = arm.workspace_radii
    for q1, q2 in np.random.default_rng(7).uniform(-math.pi, math.pi, (100, 2)):
        points = arm.joint_positions(q1, q2)
        assert np.allclose(np.linalg.norm(np.diff(points, axis=0), axis=1), [0.7, 1.3])
        assert inner - 1e-12 <= np.linalg.norm(points[-1]) <= outer + 1e-12
    assert np.isclose(np.linalg.norm(arm.joint_positions(0, 0)[-1]), outer)
    assert np.isclose(np.linalg.norm(arm.joint_positions(0, math.pi)[-1]), inner)
    assert PlanarArm(1, 1).workspace_radii == (0, 2)


def test_full_turn_is_same_pose():
    arm = PlanarArm()
    for extra in ((2 * math.pi, 0), (0, 2 * math.pi)):
        assert np.allclose(arm.forward_kinematics(0.3, -0.8),
                           arm.forward_kinematics(0.3 + extra[0], -0.8 + extra[1]))


def test_matches_independent_mujoco_kinematics():
    model_path = Path(__file__).resolve().parents[1] / "mujoco" / "two_link.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    assert model.nq == model.nv == 2
    arm = PlanarArm()
    for q in np.random.default_rng(19).uniform(-math.pi, math.pi, (30, 2)):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        expected = arm.joint_positions(*q)
        for name, xy in zip(("base", "elbow_site", "hand"), expected):
            assert np.allclose(data.site(name).xpos[:2], xy, atol=1e-12)
        T = arm.forward_kinematics(*q)
        # XML raises the arm 0.2 m; its xy geometry and orientation match exactly.
        assert np.allclose(data.site("hand").xmat.reshape(3, 3), T[:3, :3], atol=1e-12)


@pytest.mark.parametrize("length", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_lengths(length):
    with pytest.raises(ValueError):
        PlanarArm(length, 1)
    with pytest.raises(ValueError):
        PlanarArm(1, length)


@pytest.mark.parametrize("q", [(float("nan"), 0), (0, float("inf"))])
def test_rejects_nonfinite_angles(q):
    with pytest.raises(ValueError):
        PlanarArm().forward_kinematics(*q)


def test_browser_payload_converts_degrees_to_radians():
    payload = pose_payload(30, 60)
    assert np.allclose(payload["angles_rad"], [math.pi / 6, math.pi / 3])
    assert payload["hand_heading_deg"] == 90
    assert np.allclose(payload["positions"][-1], [math.sqrt(3) / 2, 1.25])
