"""Optimizer reports must not conceal corrupt or incomplete console metrics."""

import pytest

from learning.maniskill.bc_ppo_report import optimizer_trace


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "NaN"])
def test_nonfinite_optimizer_values_are_rejected(tmp_path, bad):
    log = tmp_path / "training.log"
    log.write_text(
        f"| total_timesteps | 32768 |\n| value_loss | {bad} |\n| approx_kl | 0.01 |\n| n_updates | 32 |\n-----\n"
    )
    with pytest.raises(ValueError, match="Non-finite"):
        optimizer_trace(log)


def test_missing_loss_cannot_be_plotted_as_zero(tmp_path):
    log = tmp_path / "training.log"
    log.write_text("| total_timesteps | 32768 |\n| approx_kl | 0.01 |\n-----\n")
    with pytest.raises(ValueError, match="Incomplete optimizer table"):
        optimizer_trace(log)
