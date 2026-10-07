"""mujoco/pd_control.py: the PD step responses behave as PD_NOTES.md claims.

Each test runs pd_rollout() for one gain setting (no printing, no plotting)
and checks the log it returns. Log columns:
    0 time, 1 q_target, 2 qpos, 3 qvel, 4 ctrl (commanded), 5 applied torque
"""
import numpy as np
import pytest

import pd_control
from pd_control import DIST_TIME, Gains, gravity_torque, metrics, pd_rollout

TARGET = 1.0
DURATION = 3.0
BAND = 0.05 * TARGET        # the +-5% settling band


def run(model_data, Kp, Kd, disturbance=True, name="test"):
    model, data = model_data
    g = Gains(name, Kp=Kp, Kd=Kd)
    tau_ff = gravity_torque(model, data, TARGET)
    return pd_rollout(model, data, g, TARGET, DURATION, disturbance, tau_ff)


@pytest.fixture
def tuned(model_data, tmp_logs):
    return run(model_data, 500.0, 40.0, name="tuned")


def test_gain_table_unchanged():
    # The other tests (and PD_NOTES.md) assume these exact settings.
    by_name = {g.name: g for g in pd_control.GAINS}
    assert set(by_name) == {"open_loop", "underdamped", "low_kp", "tuned"}
    assert (by_name["tuned"].Kp, by_name["tuned"].Kd) == (500.0, 40.0)
    assert (by_name["underdamped"].Kp, by_name["underdamped"].Kd) == (500.0, 0.0)
    assert (by_name["low_kp"].Kp, by_name["low_kp"].Kd) == (40.0, 10.0)
    assert by_name["open_loop"].open_loop


def test_rollout_log_shape(tuned, tmp_logs):
    assert tuned.shape == (int(round(DURATION / 0.002)) + 1, 6)
    assert np.allclose(tuned[:, 1], TARGET)
    assert (tmp_logs / "pd_tuned.csv").exists()


def test_tuned_settles_fast_and_stays_until_kick(tuned):
    t, q = tuned[:, 0], tuned[:, 2]
    err = np.abs(TARGET - q)
    inside = err <= BAND
    first_in = t[np.argmax(inside)]
    assert inside.any() and first_in < 0.5, f"entered the 5% band at {first_in:.3f}s"
    # From the first entry until the kick it never leaves the band again.
    window = (t >= first_in) & (t < DIST_TIME)
    assert inside[window].all()


def test_tuned_recovers_from_kick(tuned):
    t, q = tuned[:, 0], tuned[:, 2]
    after = t >= DIST_TIME
    # the kick really does knock it out of the band...
    assert (np.abs(TARGET - q[after]) > BAND).any()
    # ...and from 0.5 s after the kick onwards it is back inside it.
    back = t >= DIST_TIME + 0.5
    assert (np.abs(TARGET - q[back]) <= BAND).all()


def test_underdamped_overshoots(model_data, tmp_logs):
    log = run(model_data, 500.0, 0.0, disturbance=False, name="underdamped")
    overshoot = (log[:, 2].max() - TARGET) / TARGET
    assert overshoot > 0.30, f"overshoot only {overshoot:.1%}"


def test_low_kp_sags_below_target(model_data, tmp_logs):
    log = run(model_data, 40.0, 10.0, name="low_kp")
    m = metrics(log, TARGET, disturbance=True)
    # Gravity pulls the soft spring short of the target.
    assert m["ss_err"] > 0.2


def test_motor_torque_is_clamped(tuned):
    ctrl, tau = tuned[:, 4], tuned[:, 5]
    # The controller asks for far more than the motor can give at the step...
    assert np.abs(ctrl).max() > 50.0
    # ...but the applied torque (data.actuator_force) never exceeds the limit.
    assert np.abs(tau).max() <= 50.0 + 1e-9
    # Where the command is inside the limit, it is applied as-is (gear = 1).
    ok = np.abs(ctrl) < 50.0
    assert np.allclose(tau[1:][ok[1:]], ctrl[1:][ok[1:]])


def test_metrics_agree_with_direct_checks(tuned):
    m = metrics(tuned, TARGET, disturbance=True)
    assert m["overshoot"] < 5.0
    assert m["settle"] is not None and m["settle"] < 0.5
    assert m["recover"] is not None and m["recover"] < 0.5
    assert m["max_tau"] <= 50.0 + 1e-9


def test_open_loop_has_no_feedback(model_data, tmp_logs):
    model, data = model_data
    g = Gains("open_loop", 0.0, 0.0, open_loop=True)
    tau_ff = gravity_torque(model, data, TARGET)
    log = pd_rollout(model, data, g, TARGET, DURATION, True, tau_ff)
    # The command is the same constant every step, whatever q does.
    assert np.allclose(log[1:, 4], tau_ff)
