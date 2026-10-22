"""Holding must outrank a transient hit, even with a lower shaped reward."""

from learning.maniskill.selection import holding_score


def point(final, any_step, reward):
    return dict(
        success_at_end_rate=final, success_once_rate=any_step, return_mean=reward
    )


def test_holding_outranks_more_transient_hits_and_reward():
    held = point(0.4, 0.4, 10)
    transient = point(0.0, 1.0, 100)
    assert max([transient, held], key=holding_score) is held


def test_transient_success_then_return_break_equal_holding_ties():
    assert holding_score(point(0.2, 0.5, 1)) > holding_score(point(0.2, 0.4, 100))
    assert holding_score(point(0.2, 0.5, 2)) > holding_score(point(0.2, 0.5, 1))


def test_all_zero_success_uses_return_as_before():
    assert holding_score(point(0, 0, 21)) > holding_score(point(0, 0, 20))
