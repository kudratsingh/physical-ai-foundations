"""Validation-only, synchronized action/state audit for the trained ablation."""

import argparse
import json
from pathlib import Path
import statistics

import numpy as np
from stable_baselines3 import PPO

from .action_trace import evaluate_trace
from stable_baselines3.common.distributions import (
    DiagGaussianDistribution,
    SquashedDiagGaussianDistribution,
)
from .experiment import Config, save_json
from .normalized_ppo import sha256


def run(output, raw_checkpoint):
    output, raw_checkpoint = Path(output), Path(raw_checkpoint)
    saved = json.loads((output / "config.json").read_text())
    config = Config(**{key: saved[key] for key in Config.__dataclass_fields__})
    models = {
        "raw_final": PPO.load(raw_checkpoint, device="cpu"),
        "bounded_final": PPO.load(output / "final.zip", device="cpu"),
    }
    if type(models["raw_final"].policy.action_dist) is not DiagGaussianDistribution:
        raise ValueError("Raw control must use the ordinary Gaussian distribution")
    if not isinstance(
        models["bounded_final"].policy.action_dist, SquashedDiagGaussianDistribution
    ):
        raise ValueError(
            "Treatment must use the tanh-transformed Gaussian distribution"
        )
    summary = {}
    for name, model in models.items():
        result = evaluate_trace(
            config,
            model,
            saved["validation_seeds"],
            output / f"validation_trace_{name}.json",
        )
        all_records = [
            record for episode in result["episodes"] for record in episode["records"]
        ]
        near = []
        for episode in result["episodes"]:
            previous = episode["initial"]
            for record in episode["records"]:
                if previous["cube_goal_distance"] <= 0.05:
                    near.append(record)
                previous = record

        def describe(records):
            if not records:
                return dict(control_steps=0)
            return dict(
                control_steps=len(records),
                executed_arm_any_near_boundary_fraction=sum(
                    any(abs(a) >= 0.99 for a in r["executed_action"][:7])
                    for r in records
                )
                / len(records),
                median_max_abs_arm_command=statistics.median(
                    max(map(abs, r["executed_action"][:7])) for r in records
                ),
                median_latent_arm_std=statistics.median(
                    v for r in records for v in r["latent_gaussian_std"][:7]
                ),
            )

        summary[name] = dict(
            all_steps=describe(all_records),
            pre_action_goal_distance_within_5cm=describe(near),
        )
    save_json(
        output / "command_audit.json",
        dict(
            scope="Post-training validation-only diagnostic, not a causal result",
            validation_seeds=saved["validation_seeds"],
            near_goal_threshold_m=0.05,
            near_boundary_threshold=0.99,
            raw_checkpoint_sha256=sha256(raw_checkpoint),
            bounded_checkpoint_sha256=sha256(output / "final.zip"),
            source_sha256=sha256(__file__),
            results=summary,
        ),
    )
    return summary


def plots(run, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run, output = Path(run), Path(output)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for name in ["raw_final", "bounded_final"]:
        data = json.loads((run / f"validation_trace_{name}.json").read_text())
        rows = [e["records"] for e in data["episodes"]]
        series = [
            [[r["cube_goal_distance"] for r in row] for row in rows],
            [[r["max_arm_qvel"] for r in row] for row in rows],
            [[max(map(abs, r["executed_action"][:7])) for r in row] for row in rows],
            [
                [statistics.median(r["latent_gaussian_std"][:7]) for r in row]
                for row in rows
            ],
        ]
        for ax, values in zip(axes.ravel(), series, strict=True):
            values = np.asarray(values)
            median = np.median(values, axis=0)
            ax.plot(np.arange(1, 51), median, label=name)
            ax.fill_between(
                np.arange(1, 51),
                np.quantile(values, 0.25, axis=0),
                np.quantile(values, 0.75, axis=0),
                alpha=0.13,
            )
    axes[0, 0].axhline(0.025, color="green", linestyle="--")
    axes[0, 1].axhline(0.2, color="green", linestyle="--")
    for ax, title, label in zip(
        axes.ravel(),
        [
            "Cube–goal error after command",
            "Maximum arm speed after command",
            "Maximum absolute executed arm command",
            "Latent exploration std (not executed-action std)",
        ],
        [
            "Distance (m)",
            "Speed (rad/s)",
            "Normalized joint increment",
            "Median of seven latent arm stds",
        ],
        strict=True,
    ):
        ax.set(title=title, ylabel=label, xlabel="Control step (20 Hz)")
        ax.legend()
    fig.suptitle("Ten paired validation resets · median and interquartile range")
    fig.tight_layout()
    fig.savefig(output / "commands.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--raw-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    run(args.run, args.raw_checkpoint)


if __name__ == "__main__":
    main()
