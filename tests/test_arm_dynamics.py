"""Check the physical loop with actual MuJoCo steps, forces and disturbances."""
import math

import mujoco
import numpy as np
import pytest

from arm_dynamics import dynamics_model, physics_payload
from planar_arm import PlanarArm

TARGET = [12/13, 5/13]


def rollout(frames, state=None, **settings):
    frames_out = []
    for _ in range(frames):
        p = physics_payload(state or {}, TARGET, **settings)
        state = p["state"]
        frames_out.append(p)
    return p, frames_out


def test_model_has_real_motor_dynamics_and_gravity_in_the_arm_plane():
    m = dynamics_model()
    assert m.nq == m.nv == m.nu == 2
    assert m.opt.timestep == .002
    assert np.allclose(m.opt.gravity, [0, -9.81, 0])
    assert np.allclose(m.actuator_ctrlrange, [[-20, 20], [-20, 20]])
    assert np.allclose(m.actuator_gear[:, 0], [1, 1])
    assert np.allclose(m.body_mass[1:], [1, .75])


def test_one_frame_is_twenty_physics_steps_and_not_direct_joint_integration():
    p = physics_payload({}, TARGET)
    assert np.isclose(p["state"]["time"], .04)
    q = np.radians([30, 60])
    # Actual joints lag their integrated references due to finite inertia.
    assert np.linalg.norm(np.array(p["state"]["qpos"])-q) > 0
    assert np.linalg.norm(np.array(p["state"]["joint_target"])-p["state"]["qpos"]) > 1e-5


def test_default_controller_reaches_and_holds_with_nonzero_gravity_torque():
    p, _ = rollout(150)
    assert p["command"]["error_norm"] < .01
    assert p["command"]["status"] == "holding"
    assert np.linalg.norm(p["state"]["qvel"]) < .03
    assert abs(p["dynamics"]["applied_torque"][0]) > 5


def test_elbow_velocity_kick_causes_real_displacement_then_recovery():
    settled, _ = rollout(150)
    impulse = physics_payload(settled["state"], TARGET, kick=2, dt=0)
    assert np.isclose(impulse["state"]["qvel"][1]-settled["state"]["qvel"][1], 2)
    kicked = physics_payload(settled["state"], TARGET, kick=2)
    assert kicked["dynamics"]["kick"] == 2
    assert not np.allclose(kicked["pose"]["positions"][-1], settled["pose"]["positions"][-1], atol=1e-3)
    recovered, _ = rollout(100, kicked["state"])
    assert recovered["command"]["error_norm"] < .01
    assert recovered["command"]["status"] == "holding"
    assert np.linalg.norm(recovered["state"]["qvel"]) < .03


def test_torque_limits_are_enforced_by_mujoco_and_failure_remains_visible():
    p, frames = rollout(150, torque_limit=1)
    assert any(f["dynamics"]["saturated"] for f in frames)
    assert any(f["dynamics"]["reference_limited"] for f in frames)
    for f in frames:
        assert max(f["dynamics"]["peak_applied"]) <= 1+1e-12
        assert max(abs(x) for x in f["dynamics"]["applied_torque"]) <= 1+1e-12
        assert max(abs(x) for x in f["dynamics"]["joint_error"]) < .5
    assert p["command"]["error_norm"] > .1


def test_preview_preserves_state_and_reports_true_velocity_at_returned_pose():
    state = {"qpos":[.3,1.2], "qvel":[.4,-.7], "joint_target":[.31,1.21], "time":2}
    p = physics_payload(state, TARGET, dt=0)
    assert p["state"]["qpos"] == state["qpos"]
    assert p["state"]["qvel"] == state["qvel"]
    assert p["state"]["time"] == 2
    J = PlanarArm().jacobian(*state["qpos"])
    assert np.allclose(p["command"]["achieved_velocity"], J@state["qvel"])


def test_preview_is_consistent_with_independent_mujoco_site_and_applied_forces():
    p = physics_payload({}, TARGET)
    m, d = dynamics_model(), mujoco.MjData(dynamics_model())
    d.qpos[:] = p["state"]["qpos"]
    d.qvel[:] = p["state"]["qvel"]
    d.ctrl[:] = p["dynamics"]["requested_torque"]
    mujoco.mj_forward(m,d)
    assert np.allclose(d.site("hand").xpos[:2], p["pose"]["positions"][-1])
    assert np.allclose(d.qfrc_actuator, p["dynamics"]["applied_torque"])


def test_same_input_has_no_hidden_session_state_and_does_not_mutate_input():
    state={"qpos":[.4,.8], "qvel":[0,0]}
    a=physics_payload(state,TARGET)
    physics_payload({},[1,1])
    b=physics_payload(state,TARGET)
    assert a == b
    assert state == {"qpos":[.4,.8], "qvel":[0,0]}


def test_gravity_compensation_can_hold_at_rest_without_pd_position_error():
    state={"qpos":[.3,1.1], "qvel":[0,0]}
    target=PlanarArm().joint_positions(.3,1.1)[-1]
    compensated=physics_payload(state,target)
    plain=physics_payload(state,target,compensate=False)
    assert np.linalg.norm(compensated["state"]["qvel"]) < 1e-10
    assert np.linalg.norm(plain["state"]["qvel"]) > .01


@pytest.mark.parametrize("lengths",[(.5,1),(1,1),(.25,.25)])
def test_adjustable_geometry_keeps_analytic_and_physics_positions_consistent(lengths):
    p=physics_payload({},TARGET,l1=lengths[0],l2=lengths[1])
    m=dynamics_model(*lengths)
    d=mujoco.MjData(m)
    d.qpos[:]=p["state"]["qpos"]
    mujoco.mj_forward(m,d)
    assert np.allclose(d.site("hand").xpos[:2],p["pose"]["positions"][-1])


@pytest.mark.parametrize("settings",[{"dt":.02},{"kp":0},{"kd":-1},{"torque_limit":0},
 {"gravity":"yes"},{"compensate":1},{"kick":4},{"l1":math.nan},{"gain":1e300},{"damping":0}])
def test_invalid_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        physics_payload({},TARGET,**settings)


@pytest.mark.parametrize("state",[{"qpos":[math.nan,0]},{"qvel":[1]}, {"time":-1},
 {"joint_target":[1,math.inf]}, {"hold_time":"x"}])
def test_invalid_state_is_rejected(state):
    with pytest.raises(ValueError):
        physics_payload(state,TARGET)
