"""The MJCF model one_joint.xml: check the numbers the scripts rely on.

If someone edits the XML (timestep, motor limits, damping, ...), these tests
say so before the PD gains or the rollout comparisons silently change meaning.
"""
import mujoco
import numpy as np


def test_model_loads(model_data):
    model, data = model_data
    assert isinstance(model, mujoco.MjModel)
    assert isinstance(data, mujoco.MjData)


def test_one_degree_of_freedom(model_data):
    model, _ = model_data
    # one hinge -> one position, one velocity, one actuator
    assert model.nq == 1
    assert model.nv == 1
    assert model.nu == 1
    assert model.njnt == 1
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_HINGE


def test_timestep(model_data):
    model, _ = model_data
    assert model.opt.timestep == 0.002


def test_actuator_is_motor_gear1_limited_50(model_data):
    model, _ = model_data
    # A <motor> is a "direct" actuator: no dynamics, fixed gain, no bias.
    assert model.actuator_dyntype[0] == mujoco.mjtDyn.mjDYN_NONE
    assert model.actuator_gaintype[0] == mujoco.mjtGain.mjGAIN_FIXED
    assert model.actuator_biastype[0] == mujoco.mjtBias.mjBIAS_NONE
    assert model.actuator_gainprm[0, 0] == 1.0
    assert model.actuator_gear[0, 0] == 1.0
    assert model.actuator_ctrllimited[0] == 1
    assert np.allclose(model.actuator_ctrlrange[0], (-50.0, 50.0))
    # it drives the hinge
    assert model.actuator_trnid[0, 0] == 0


def test_joint_damping(model_data):
    model, _ = model_data
    assert np.isclose(model.dof_damping[0], 0.05)


def test_gravity_on(model_data):
    model, _ = model_data
    assert np.allclose(model.opt.gravity, (0.0, 0.0, -9.81))
