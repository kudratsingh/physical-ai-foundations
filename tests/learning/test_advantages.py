"""Analytic GAE limits and episode-boundary behavior for the study worksheet."""

import pytest
from learning.advantages import Transition, estimate


def test_lambda_zero_is_one_step_td_and_lambda_one_matches_discounted_return():
    trajectory = [
        Transition(1, 0.5, 0.4),
        Transition(2, 0.4, 0.2),
        Transition(3, 0.2, 999, terminated=True),
    ]
    errors, advantages, _ = estimate(trajectory, gamma=0.9, lam=0)
    assert advantages == errors
    _, _, targets = estimate(trajectory, gamma=0.9, lam=1)
    assert targets == pytest.approx([1 + 0.9 * 2 + 0.9**2 * 3, 2 + 0.9 * 3, 3])


def test_timeout_bootstraps_final_observation_but_true_terminal_does_not():
    _, a, targets = estimate([Transition(1, 0.2, 0.8, truncated=True)], gamma=0.9)
    assert targets == pytest.approx([1 + 0.9 * 0.8])
    assert a == pytest.approx([1 + 0.9 * 0.8 - 0.2])
    _, _, terminal = estimate([Transition(1, 0.2, 0.8, terminated=True)], gamma=0.9)
    assert terminal == [1]


@pytest.mark.parametrize("boundary", [dict(terminated=True), dict(truncated=True)])
def test_reset_episode_rewards_never_leak_back_across_either_boundary(boundary):
    before = [Transition(1, 0, 2, **boundary)]
    original = estimate(before, gamma=0.9, lam=0.8)[1]
    extended = estimate(
        [*before, Transition(10000, 0, 0, terminated=True)], gamma=0.9, lam=0.8
    )[1]
    assert extended[0] == original[0]


def test_three_step_hand_calculation_and_gamma_zero():
    data = [
        Transition(1, 0.5, 0.4),
        Transition(1, 0.4, 0.2),
        Transition(1, 0.2, 0.8, truncated=True),
    ]
    errors, advantages, targets = estimate(data, gamma=0.9, lam=0.8)
    assert errors == pytest.approx([0.86, 0.78, 1.52])
    assert advantages == pytest.approx([2.209568, 1.8744, 1.52])
    assert targets == pytest.approx([2.709568, 2.2744, 1.72])
    _, advantages, targets = estimate(data, gamma=0, lam=1)
    assert advantages == pytest.approx([0.5, 0.6, 0.8])
    assert targets == pytest.approx([1, 1, 1])


@pytest.mark.parametrize(
    "arguments", [dict(gamma=-0.1), dict(lam=1.1), dict(gamma=float("nan"))]
)
def test_invalid_discount_settings_rejected(arguments):
    with pytest.raises(ValueError):
        estimate([Transition(1, 0, 0)], **arguments)
