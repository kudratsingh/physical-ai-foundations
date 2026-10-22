"""Evaluation summaries independent of the simulator and training libraries."""

import math
import statistics


def wilson_interval(successes, total, z=1.96):
    """95% binomial interval; a single training seed is still only one trained model."""
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Need 0 <= successes <= total and total > 0")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    return [max(0, center - radius), min(1, center + radius)]


def summarize(episodes):
    if not episodes:
        raise ValueError("Evaluation requires at least one episode")
    seeds = [e["seed"] for e in episodes]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation seeds must be unique")
    returns = [e["return"] for e in episodes]
    if not all(math.isfinite(x) for x in returns):
        raise ValueError("Non-finite episode return")
    successes = sum(bool(e["success_once"]) for e in episodes)
    return {
        "episodes": len(episodes),
        "successes": successes,
        "success_once_rate": successes / len(episodes),
        "success_at_end_rate": sum(bool(e["success_at_end"]) for e in episodes)
        / len(episodes),
        "success_once_95pct_wilson": wilson_interval(successes, len(episodes)),
        "return_mean": statistics.mean(returns),
        "return_std": statistics.stdev(returns) if len(returns) > 1 else 0.0,
        "steps_mean": statistics.mean(e["steps"] for e in episodes),
    }


def failure_counts(episodes):
    """Mutually exclusive final-outcome categories, computed from saved evidence."""
    counts = dict(
        final_success=0,
        reached_then_lost=0,
        placed_but_moving=0,
        lifted_without_success=0,
        grasped_without_lift=0,
        never_grasped=0,
    )
    for episode in episodes:
        if episode["success_at_end"]:
            category = "final_success"
        elif episode["success_once"]:
            category = "reached_then_lost"
        elif episode["placed_once"]:
            category = "placed_but_moving"
        elif episode["lifted_5cm"]:
            category = "lifted_without_success"
        elif episode["grasped_once"]:
            category = "grasped_without_lift"
        else:
            category = "never_grasped"
        counts[category] += 1
    return counts
