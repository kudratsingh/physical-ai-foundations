"""Deliberate, internally consistent goal-observation fault for PickCube state."""

import numpy as np


def corrupt_goal(observation):
    """Replace goal by current cube position in BOTH redundant goal encodings.

    PickCube's pinned 42-state layout: agent[18], grasp[1], TCP pose[7],
    goal xyz[3], cube pose[7], TCP-to-cube[3], cube-to-goal[3].
    This changes policy input only; simulator goal and scoring stay intact.
    """
    result = np.array(observation, copy=True)
    if result.shape[-1] != 42:
        raise ValueError("Goal fault requires PickCube 42-value state observations")
    result[..., 26:29] = result[..., 29:32]
    result[..., 39:42] = 0
    return result


class GoalObservationFault:
    @property
    def policy_label(self):
        return (
            "BC-256 with deliberately corrupted goal"
            if self.enabled
            else "BC-256 with restored goal observations"
        )

    def __init__(self, policy, enabled=True):
        self.policy = policy
        self.enabled = enabled

    def predict(self, observation, deterministic=True):
        return self.policy.predict(
            corrupt_goal(observation) if self.enabled else observation,
            deterministic=deterministic,
        )
