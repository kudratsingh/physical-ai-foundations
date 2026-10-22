"""Observation-only PickCube hybrid gate; no simulator or privileged state access."""

from dataclasses import dataclass
import numpy as np


def placed_from_observation(observation):
    """Use the ordinary raw-state obj_to_goal vector, indices39:42 (metres)."""
    values = np.asarray(observation, dtype=np.float32)
    if values.shape != (42,) or not np.isfinite(values).all():
        raise ValueError("Need one finite raw42-dimensional PickCube observation")
    # Compare float32 to float32, as the upstream torch task does at its boundary.
    return bool(np.linalg.norm(values[39:42]) <= np.float32(0.025))


@dataclass
class ObservationStopGate:
    """Latch zero arm increments from an ordinary observed placed state onward.

    This hand-coded hybrid is not PPO alone. The caller supplies the current
    policy action; this adapter never reads an env, info dict or goal actor.
    """

    latched: bool = False
    first_placement_state_step: int | None = None

    def action(self, observation, policy_action, state_step):
        placed = placed_from_observation(observation)
        if placed and not self.latched:
            self.latched = True
            self.first_placement_state_step = state_step
        original = np.asarray(policy_action)
        if original.shape != (8,) or not np.isfinite(original).all():
            raise ValueError("Need seven arm actions and one finite gripper command")
        executed = original.copy()
        if self.latched:
            executed[:7] = 0
        return executed, self.latched, placed
