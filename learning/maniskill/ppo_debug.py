"""Publish an auditable, fresh-seed comparison of controlled PPO continuations."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from .bc_data import sha256_file
from .behavior_cloning import BehaviorPolicy
from .experiment import Config, evaluate, save_json
from .metrics import failure_counts, wilson_interval
from .observation_fault import GoalObservationFault
from .ppo_continuation import ADDITIONAL_STEPS, TEST_SEEDS, VALIDATION_SEEDS

NAMES = [
    "baseline",
    "control_final",
    "slow_final",
    "control_selected",
    "slow_selected",
    "bc_reference",
    "bc_goal_fault",
    "bc_restored",
]
VIDEO_NAMES = ["control_final", "slow_final", "bc_reference", "bc_goal_fault"]
VIDEO_SEEDS = TEST_SEEDS[:2]
LABELS = dict(
    baseline="PPO 1M",
    control_final="PPO +524k · 3e-4",
    slow_final="PPO +524k · 1e-4",
    control_selected="3e-4 · selected",
    slow_selected="1e-4 · selected",
    bc_reference="BC-256 reference",
    bc_goal_fault="BC · goal fault",
    bc_restored="BC · restored goal",
)


def nested_signature(value):
    """Canonical tensor fingerprints also cover Adam moments and param groups."""
    import torch

    if isinstance(value, torch.Tensor):
        return dict(
            dtype=str(value.dtype),
            shape=list(value.shape),
            sha256=hashlib.sha256(
                value.cpu().contiguous().numpy().tobytes()
            ).hexdigest(),
        )
    if isinstance(value, dict):
        return {str(k): nested_signature(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [nested_signature(v) for v in value]
    return value


def signature_sha(value):
    encoded = json.dumps(
        nested_signature(value), sort_keys=True, allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def protocol(checkpoint, bc_checkpoint, control, slow, output):
    import torch
    from stable_baselines3 import PPO

    configurations = {
        n: json.loads((p / "config.json").read_text())
        for n, p in [("control", control), ("slow", slow)]
    }
    a, b = configurations.values()
    for key in [
        "task",
        "starting_timesteps",
        "additional_steps",
        "checkpoint_sha256",
        "initial_policy_sha256",
        "seed",
        "validation_seeds",
        "test_seeds",
        "checkpoint_selection",
        "resume_semantics",
        "versions",
        "source_sha256",
    ]:
        if a[key] != b[key]:
            raise ValueError(f"Forks differ in a shared protocol field: {key}")
    if (a["learning_rate"], b["learning_rate"]) != (3e-4, 1e-4):
        raise ValueError("Unexpected learning-rate treatment")
    if (
        a["additional_steps"] != ADDITIONAL_STEPS
        or a["validation_seeds"] != VALIDATION_SEEDS
        or a["test_seeds"] != TEST_SEEDS
    ):
        raise ValueError("Unexpected budget or seed partition")
    original = PPO.load(checkpoint, device="cpu").get_parameters()
    states = {
        n: torch.load(p / "initial_parameters.pt", weights_only=True)
        for n, p in [("control", control), ("slow", slow)]
    }
    original_signature = signature_sha(original)
    if any(signature_sha(v) != original_signature for v in states.values()):
        raise ValueError(
            "Initial weights or Adam state differ from the shared checkpoint"
        )
    for p in [control, slow]:
        training = json.loads((p / "training.json").read_text())
        if (
            not training["complete"]
            or training["actual_additional_timesteps"] != ADDITIONAL_STEPS
        ):
            raise ValueError("Finish the fixed training budgets before final testing")
    bc_protocol = json.loads(
        (bc_checkpoint.parent.parent / "protocol.json").read_text()
    )
    source_seeds = bc_protocol["source_seed_values"]
    if set(source_seeds) & set(VALIDATION_SEEDS + TEST_SEEDS):
        raise ValueError("Reserved rollout seeds overlap the BC reference data")
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        bc_source_seed_values=source_seeds,
        observation_fault=dict(
            policy="BC-256 reference",
            intervention="replace goal xyz with current cube xyz and cube-to-goal displacement with zero; actual simulator goal unchanged",
            recovery="disable input fault without changing weights; clean predictions must match original exactly",
        ),
        hypothesis="Smaller optimization steps may improve goal placement and stillness; success improvement is not assumed",
        treatment={"control_learning_rate": 3e-4, "slow_learning_rate": 1e-4},
        starting_checkpoint_sha256=sha256_file(checkpoint),
        bc_reference_sha256=sha256_file(bc_checkpoint),
        initial_parameters_sha256={n: signature_sha(v) for n, v in states.items()},
        original_parameters_sha256=original_signature,
        optimizer_sha256={
            n: signature_sha(v["policy.optimizer"]) for n, v in states.items()
        },
        configurations=configurations,
        validation_seeds=VALIDATION_SEEDS,
        test_seeds=TEST_SEEDS,
        video_seeds=VIDEO_SEEDS,
        video_policies=VIDEO_NAMES,
        fixed_budget_policy_names=NAMES[:3],
        selected_policy_names=NAMES[3:5],
        bc_reference_role="Same task on fresh seeds; expert-data reference, outside the causal learning-rate comparison",
        limitation="One common starting checkpoint and one continuation seed; paired episode uncertainty does not measure training-seed variation",
        model_sha256={
            n: sha256_file(p / f"{file}.zip")
            for n, p, file in [
                ("control_final", control, "final"),
                ("slow_final", slow, "final"),
                ("control_selected", control, "best"),
                ("slow_selected", slow, "best"),
            ]
        },
    )
    save_json(output / "protocol.json", manifest)
    return Config(**a["task"])


def run(checkpoint, bc_checkpoint, control, slow, output):
    from stable_baselines3 import PPO

    task = protocol(checkpoint, bc_checkpoint, control, slow, output)
    paths = dict(
        baseline=checkpoint,
        control_final=control / "final.zip",
        slow_final=slow / "final.zip",
        control_selected=control / "best.zip",
        slow_selected=slow / "best.zip",
    )
    clean_bc = BehaviorPolicy.load(bc_checkpoint)
    fault_bc = GoalObservationFault(clean_bc)
    for name in NAMES:
        policy = (
            clean_bc
            if name in ("bc_reference", "bc_goal_fault", "bc_restored")
            else PPO.load(paths[name], device="cpu")
        )
        if name in ("bc_goal_fault", "bc_restored"):
            fault_bc.enabled = name == "bc_goal_fault"
            policy = fault_bc
        evaluate(task, policy, TEST_SEEDS, output / f"test_{name}.json", traces=True)
        if name in VIDEO_NAMES:
            evaluate(
                task,
                policy,
                VIDEO_SEEDS,
                output / f"video_{name}.json",
                traces=True,
                video=True,
            )


def optimizer_points(directory):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    points = {}
    for file in sorted((directory / "tensorboard").rglob("events.out.tfevents.*")):
        events = EventAccumulator(str(file), size_guidance={"scalars": 0})
        events.Reload()
        for tag in [
            "train/approx_kl",
            "train/clip_fraction",
            "train/std",
            "train/learning_rate",
            "train/value_loss",
        ]:
            if tag in events.Tags()["scalars"]:
                points.setdefault(tag, []).extend(
                    dict(timesteps=p.step, value=p.value) for p in events.Scalars(tag)
                )
    return points


def publish(control, slow, output, evidence):
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    evidence.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((output / "protocol.json").read_text())
    shutil.copy2(output / "protocol.json", evidence / "protocol.json")
    results = {}
    for name in NAMES:
        result = json.loads((output / f"test_{name}.json").read_text())
        for episode in result["episodes"]:
            episode.pop("trajectory", None)
        result["policy"] = LABELS[name]
        results[name] = result
        save_json(evidence / f"test_{name}.json", result)
    if results["bc_reference"]["episodes"] != results["bc_restored"]["episodes"]:
        raise ValueError(
            "Restored goal observations did not reproduce clean BC episodes"
        )
    comparison = {n: r["summary"] for n, r in results.items()}
    failures = {n: failure_counts(r["episodes"]) for n, r in results.items()}
    save_json(evidence / "comparison.json", comparison)
    save_json(evidence / "failures.json", failures)
    optimization = {}
    for name, directory in [("control", control), ("slow", slow)]:
        for file in [
            "config.json",
            "training.json",
            "validation.json",
            "selected.json",
        ]:
            shutil.copy2(directory / file, evidence / f"{name}_{file}")
        optimization[name] = optimizer_points(directory)
    save_json(evidence / "optimization.json", optimization)
    for name in VIDEO_NAMES:
        video = json.loads((output / f"video_{name}.json").read_text())
        reference = {e["seed"]: e for e in results[name]["episodes"]}
        for e in video["episodes"]:
            e.pop("trajectory", None)
            if e != reference[e["seed"]]:
                raise ValueError(f"Rendering changed episode: {name} {e['seed']}")
        video["policy"] = LABELS[name]
        save_json(evidence / f"video_{name}.json", video)
        for seed in VIDEO_SEEDS:
            shutil.copy2(
                output / f"video_{name}_seed{seed}.mp4",
                evidence / f"{name}_seed{seed}.mp4",
            )
    # Both final and selected variants are reported; choosing step zero cannot
    # be mistaken for learning a better policy during continuation.
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for name, label in [("control", "3e-4"), ("slow", "1e-4")]:
        h = json.loads((evidence / f"{name}_validation.json").read_text())
        x = [p["additional_timesteps"] for p in h]
        axes[0, 0].plot(
            x, [p["success_at_end_rate"] for p in h], marker="o", label=label
        )
        axes[0, 1].plot(x, [p["return_mean"] for p in h], marker="o", label=label)
        kl = optimization[name].get("train/approx_kl", [])
        axes[0, 2].plot(
            [
                p["timesteps"] - manifest["configurations"][name]["starting_timesteps"]
                for p in kl
            ],
            [p["value"] for p in kl],
            label=label,
            alpha=0.8,
        )
    axes[0, 0].set(
        title="Validation final success",
        xlabel="Additional transitions",
        ylabel="Success fraction",
        ylim=(-0.02, 1.02),
    )
    axes[0, 1].set(
        title="Validation return", xlabel="Additional transitions", ylabel="Mean return"
    )
    axes[0, 2].set(
        title="Observed policy update size",
        xlabel="Additional transitions",
        ylabel="Approximate KL divergence",
    )
    for ax in axes[0]:
        ax.legend()
    x = np.arange(len(NAMES))
    labels = [
        "1M PPO",
        "3e-4 final",
        "1e-4 final",
        "3e-4 sel.",
        "1e-4 sel.",
        "BC ref.",
        "Goal fault",
        "Restored",
    ]
    axes[1, 0].bar(
        x - 0.18,
        [comparison[n]["success_once_rate"] for n in NAMES],
        0.36,
        label="Any step",
    )
    axes[1, 0].bar(
        x + 0.18,
        [comparison[n]["success_at_end_rate"] for n in NAMES],
        0.36,
        label="At end",
    )
    axes[1, 0].set(
        title="50 fresh paired test episodes",
        xticks=x,
        xticklabels=labels,
        ylim=(0, 1.05),
        ylabel="Success fraction",
    )
    axes[1, 0].legend()
    for name in ["baseline", "control_final", "slow_final", "bc_reference"]:
        episodes = results[name]["episodes"]
        axes[1, 1].scatter(
            [max(e["final_goal_distance"], 1e-5) for e in episodes],
            [max(max(abs(v) for v in e["final_arm_qvel"]), 1e-5) for e in episodes],
            label=LABELS[name],
            s=15,
            alpha=0.65,
        )
    axes[1, 1].axvline(0.025, color="#666", linestyle="--")
    axes[1, 1].axhline(0.2, color="#666", linestyle="--")
    axes[1, 1].set(
        title="Terminal position and stillness",
        xlabel="3D goal distance (m; log)",
        ylabel="Max arm speed (rad/s; log)",
        xscale="log",
        yscale="log",
    )
    axes[1, 1].legend(fontsize=7)
    bottom = np.zeros(len(NAMES))
    colors = ["#2a8a65", "#c55353", "#df993f", "#8586b8", "#4b82b4", "#a9b5c2"]
    for category, color in zip(next(iter(failures.values())), colors):
        values = np.array([failures[n][category] for n in NAMES])
        axes[1, 2].bar(
            x, values, bottom=bottom, color=color, label=category.replace("_", " ")
        )
        bottom += values
    axes[1, 2].set(
        title="Exclusive outcome categories",
        xticks=x,
        xticklabels=labels,
        ylabel="Episodes",
        ylim=(0, 65),
    )
    axes[1, 2].legend(fontsize=7, ncol=2, loc="upper center")
    for ax in [axes[1, 0], axes[1, 2]]:
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(evidence / "results.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    rows = ""
    for name in NAMES:
        summary = comparison[name]
        count = round(summary["success_at_end_rate"] * 50)
        lo, hi = wilson_interval(count, 50)
        rows += f"<tr><td>{LABELS[name]}</td><td>{summary['successes']}/50</td><td>{count}/50</td><td>{100 * lo:.1f}–{100 * hi:.1f}%</td><td>{summary['return_mean']:.2f}</td></tr>"
    cards = ""
    for seed in VIDEO_SEEDS:
        for name in VIDEO_NAMES:
            e = next(e for e in results[name]["episodes"] if e["seed"] == seed)
            cards += f'<article><h3>{LABELS[name]} · seed {seed}</h3><video controls muted preload="metadata" src="{name}_seed{seed}.mp4"></video><p>Final success: {e["success_at_end"]}. Distance {e["final_goal_distance"]:.3f} m; max joint speed {max(abs(v) for v in e["final_arm_qvel"]):.3f} rad/s.</p></article>'
    template = Path(__file__).with_name("ppo_debug_report.html").read_text()
    (evidence / "index.html").write_text(
        template.replace("{{ROWS}}", rows).replace("{{CARDS}}", cards)
    )
    for name in NAMES:
        print(name, comparison[name], failures[name])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["evaluate", "publish"])
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("../stage1-pickcube/learning/runs/pickcube_ppo_seed42/final.zip"),
    )
    parser.add_argument(
        "--bc",
        type=Path,
        default=Path("../stage1-pickcube/learning/runs/pickcube_bc/bc_256/best.pt"),
    )
    parser.add_argument(
        "--control", type=Path, default=Path("learning/runs/continuation_control")
    )
    parser.add_argument(
        "--slow", type=Path, default=Path("learning/runs/continuation_slow")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("learning/runs/ppo_debug_comparison")
    )
    parser.add_argument(
        "--evidence", type=Path, default=Path("learning/maniskill/evidence/ppo_debug")
    )
    args = parser.parse_args()
    if args.command == "evaluate":
        run(args.checkpoint, args.bc, args.control, args.slow, args.output)
    else:
        publish(args.control, args.slow, args.output, args.evidence)


if __name__ == "__main__":
    main()
