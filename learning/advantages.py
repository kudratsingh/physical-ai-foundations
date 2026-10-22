"""Small, explicit GAE worksheet; production PPO still uses Stable-Baselines3.

Each next_value must describe the actual next observation, including the final
observation before an automatic reset. Episode boundaries break the trace;
only true termination removes value bootstrapping.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Transition:
    reward: float
    value: float
    next_value: float
    terminated: bool = False
    truncated: bool = False


def estimate(transitions, gamma=0.99, lam=0.95):
    """Return TD errors, GAE advantages and bootstrapped value targets."""
    if not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError("gamma and lambda must be in [0,1]")
    transitions = list(transitions)
    if not transitions:
        raise ValueError("At least one transition is required")
    if any(
        not all(math.isfinite(v) for v in (t.reward, t.value, t.next_value))
        for t in transitions
    ):
        raise ValueError("Rewards and value estimates must be finite")
    errors = [
        t.reward + gamma * (not t.terminated) * t.next_value - t.value
        for t in transitions
    ]
    advantages = [0.0] * len(transitions)
    following = 0.0
    for i in range(len(transitions) - 1, -1, -1):
        t = transitions[i]
        following = (
            errors[i] + gamma * lam * (not (t.terminated or t.truncated)) * following
        )
        advantages[i] = following
    targets = [a + t.value for a, t in zip(advantages, transitions)]
    return errors, advantages, targets


def example():
    sequence = [
        Transition(1, 0.5, 0.4),
        Transition(1, 0.4, 0.2),
        Transition(1, 0.2, 0.8, truncated=True),
    ]
    for label, last in [
        ("time limit", sequence[-1]),
        ("true ending", Transition(1, 0.2, 0.8, terminated=True)),
    ]:
        errors, advantages, targets = estimate(
            [*sequence[:-1], last], gamma=0.9, lam=0.8
        )
        print(f"\n{label}: gamma=0.9, lambda=0.8")
        print(" step   reward   value   next V    delta   advantage   V target")
        for i, (t, d, a, v) in enumerate(
            zip([*sequence[:-1], last], errors, advantages, targets)
        ):
            print(
                f"{i:5d} {t.reward:8.2f} {t.value:7.2f} {t.next_value:8.2f} {d:8.5f} {a:11.6f} {v:10.6f}"
            )


if __name__ == "__main__":
    example()
