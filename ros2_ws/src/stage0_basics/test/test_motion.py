import math

import pytest

from stage0_basics.motion import joint_state


@pytest.mark.parametrize("elapsed,q,qvel", [
    (0.0, 0.0, math.pi / 2),
    (0.5, 0.5, 0.0),
    (1.0, 0.0, -math.pi / 2),
    (1.5, -0.5, 0.0),
    (2.0, 0.0, math.pi / 2),
])
def test_motion_has_expected_phase_and_units(elapsed, q, qvel):
    actual_q, actual_qvel = joint_state(elapsed)
    assert actual_q == pytest.approx(q, abs=1e-12)
    assert actual_qvel == pytest.approx(qvel, abs=1e-12)
