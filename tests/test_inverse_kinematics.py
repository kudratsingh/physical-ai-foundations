"""Analytic IK, boundary/family cases, and independent forward checks."""
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from planar_arm import PlanarArm, ik_payload


def test_known_target_has_two_branches_with_expected_elbow_sides():
    arm = PlanarArm()
    result = arm.inverse_kinematics(1, 0.75)
    assert result.reachable and not result.singular and not result.infinite
    by_name = {s.branch: s for s in result.solutions}
    assert np.allclose(by_name["elbow-down"].angles, [0, math.pi/2], atol=1e-12)
    assert np.allclose(by_name["elbow-up"].angles,
                       [2*math.atan2(0.75, 1), -math.pi/2], atol=1e-12)
    for s in result.solutions:
        elbow = arm.joint_positions(*s.angles)[1]
        cross = elbow[1] - 0.75*elbow[0]
        assert cross > 0 if s.branch == "elbow-up" else cross < 0
        assert s.error < 1e-12


@pytest.mark.parametrize("lengths", [(1, 0.75), (0.75, 1), (1, 1), (0.25, 1.5)])
def test_fk_ik_roundtrips_across_workspace_and_quadrants(lengths):
    arm = PlanarArm(*lengths)
    rng = np.random.default_rng(2026)
    for q in rng.uniform(-math.pi, math.pi, (100, 2)):
        target = arm.joint_positions(*q)[-1]
        result = arm.inverse_kinematics(*target)
        assert result.reachable and len(result.solutions) == 2
        for s in result.solutions:
            assert np.allclose(arm.joint_positions(*s.angles)[-1], target, atol=1e-11)
            assert s.error < 1e-11


@pytest.mark.parametrize("lengths", [(1, 0.75), (0.75, 1)])
@pytest.mark.parametrize("direction", [(1, 0), (-1, 0), (0, 1), (0, -1)])
def test_inner_and_outer_boundaries_return_one_unique_configuration(lengths, direction):
    arm = PlanarArm(*lengths)
    for radius in arm.workspace_radii:
        target = np.array(direction)*radius
        result = arm.inverse_kinematics(*target)
        assert result.reachable and result.singular and not result.infinite
        assert len(result.solutions) == 1
        assert result.solutions[0].error < 1e-12


def test_equal_links_at_origin_preserve_shoulder_hint_as_family_representative():
    arm = PlanarArm(1, 1)
    for hint in (0, 0.7, -2.3):
        result = arm.inverse_kinematics(0, 0, shoulder_hint=hint)
        assert result.reachable and result.singular and result.infinite
        assert len(result.solutions) == 1
        s = result.solutions[0]
        assert math.isclose(s.q1, hint, abs_tol=1e-12)
        assert s.branch == "representative" and s.error < 1e-12


def test_near_origin_is_not_confused_with_infinite_family():
    result = PlanarArm(1, 1).inverse_kinematics(1e-9, -2e-9)
    assert result.reachable and not result.singular and not result.infinite
    assert len(result.solutions) == 2
    assert all(s.error < 1e-12 for s in result.solutions)


def test_nearly_equal_but_unequal_links_cannot_reach_origin():
    result = PlanarArm(1, 1 + 1e-15).inverse_kinematics(0, 0)
    assert not result.reachable and result.reason == "inside_hole"


@pytest.mark.parametrize("target,reason", [((0, 0), "inside_hole"),
    ((0.1, 0.1), "inside_hole"), ((2, 0), "outside_reach"),
    ((-3, -4), "outside_reach"), ((1e300, 0), "outside_reach")])
def test_unreachable_targets_have_no_fabricated_solutions(target, reason):
    result = PlanarArm().inverse_kinematics(*target)
    assert not result.reachable and result.reason == reason
    assert result.solutions == ()


def test_clamps_only_floating_point_noise_at_reach_boundary():
    arm = PlanarArm()
    boundary_noise = math.nextafter(1.75, math.inf)
    result = arm.inverse_kinematics(boundary_noise, 0)
    assert result.reachable and result.singular
    assert result.solutions[0].error < 1e-12
    assert not arm.inverse_kinematics(1.75 + 1e-6, 0).reachable
    assert not arm.inverse_kinematics(0.25 - 1e-6, 0).reachable


@pytest.mark.parametrize("values", [(math.nan, 0, 0), (0, math.inf, 0), (0, 0, math.nan)])
def test_rejects_nonfinite_target_or_hint(values):
    with pytest.raises(ValueError):
        PlanarArm().inverse_kinematics(*values)


def test_ik_solutions_match_independent_mujoco_sites():
    model = mujoco.MjModel.from_xml_path(str(Path(__file__).resolve().parents[1]/"mujoco"/"two_link.xml"))
    data = mujoco.MjData(model)
    arm = PlanarArm()
    for q in np.random.default_rng(41).uniform(-math.pi, math.pi, (20, 2)):
        target = arm.joint_positions(*q)[-1]
        for s in arm.inverse_kinematics(*target).solutions:
            data.qpos[:] = s.angles
            mujoco.mj_forward(model, data)
            assert np.allclose(data.site("hand").xpos[:2], target, atol=1e-11)


def test_payload_includes_both_actual_fk_poses_and_errors():
    payload = ik_payload(1, 0.75)
    assert payload["reachable"] and len(payload["solutions"]) == 2
    for s in payload["solutions"]:
        assert np.allclose(s["pose"]["positions"][-1], [1, 0.75], atol=1e-12)
        assert s["error"] < 1e-12
