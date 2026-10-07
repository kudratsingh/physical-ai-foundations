"""mujoco/simulate.py: reset(), stepping, and the three-rollout comparison."""
import mujoco
import numpy as np
import pytest

import simulate
from simulate import ROLLOUTS, Rollout, reset, run_rollout


def step_for(model, data, seconds, ctrl=0.0):
    """Step with a constant ctrl and return the qpos trace (including t=0)."""
    n = int(round(seconds / model.opt.timestep))
    q = np.empty(n + 1)
    q[0] = data.qpos[0]
    for i in range(1, n + 1):
        data.ctrl[0] = ctrl
        mujoco.mj_step(model, data)
        q[i] = data.qpos[0]
    return q


def test_import_has_no_side_effects():
    # Importing must not run a simulation; the rollouts start without logs.
    assert all(ro.log is None for ro in ROLLOUTS)
    assert callable(simulate.main)


def test_reset_sets_initial_condition(model_data):
    model, data = model_data
    # dirty the state first so reset() has something to undo
    data.qpos[0], data.qvel[0], data.ctrl[0] = 2.0, 5.0, 7.0
    mujoco.mj_step(model, data)
    assert data.time > 0

    reset(model, data, qpos0=0.5, ctrl=3.0)
    assert data.qpos[0] == 0.5
    assert data.qvel[0] == 0.0
    assert data.ctrl[0] == 3.0
    assert data.time == 0.0


def test_free_swing_is_bounded_and_damped(model_data):
    model, data = model_data
    reset(model, data, qpos0=0.5, ctrl=0.0)
    q = step_for(model, data, 2.0)
    # Released from rest with no torque: it can never swing higher than it started.
    assert np.abs(q).max() <= 0.5 + 1e-6
    # Damping removes energy: the last swing peak is lower than the start.
    last_period = q[-int(round(1.0 / model.opt.timestep)):]   # ~ last second (> 1 swing)
    assert np.abs(last_period).max() < 0.5


def test_rest_at_bottom_stays_at_bottom(model_data):
    model, data = model_data
    reset(model, data, qpos0=0.0, ctrl=0.0)
    q = step_for(model, data, 2.0)
    # Hanging straight down is an equilibrium: nothing should move.
    assert np.abs(q).max() < 1e-9
    assert abs(data.qvel[0]) < 1e-9


def test_three_rollouts_differ(model_data, tmp_logs):
    model, data = model_data
    # Fresh copies so the module-level ROLLOUTS list is not mutated by tests.
    rollouts = [Rollout(ro.name, ro.qpos0, ro.ctrl) for ro in ROLLOUTS]
    for ro in rollouts:
        log = run_rollout(model, data, ro, verbose=False)
        n_steps = int(round(simulate.SIM_SECONDS / model.opt.timestep))
        assert log.shape == (n_steps + 1, 4)
        assert np.isclose(log[-1, 0], simulate.SIM_SECONDS)
        assert (tmp_logs / f"rollout_{ro.name}.csv").exists()

    # Same check as simulate.summarize(): final qpos must differ from baseline.
    base = rollouts[0]
    for ro in rollouts[1:]:
        assert abs(ro.log[-1, 1] - base.log[-1, 1]) > 1e-3, ro.name


def test_run_rollout_quiet(model_data, tmp_logs, capsys):
    model, data = model_data
    run_rollout(model, data, Rollout("quiet", 0.5, 0.0), verbose=False)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("q0", [0.5, 1.5])
def test_rollout_csv_matches_log(model_data, tmp_logs, q0):
    model, data = model_data
    ro = Rollout("csvcheck", q0, 0.0)
    log = run_rollout(model, data, ro, verbose=False)
    csv = np.loadtxt(tmp_logs / "rollout_csvcheck.csv", delimiter=",", skiprows=1)
    assert np.allclose(csv, log)
