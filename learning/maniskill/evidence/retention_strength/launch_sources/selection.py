"""Checkpoint ranking for tasks that must finish in a successful state."""


def holding_score(summary):
    """Prefer held success, then transient success, then the shaped return."""
    return (
        summary["success_at_end_rate"],
        summary["success_once_rate"],
        summary["return_mean"],
    )
