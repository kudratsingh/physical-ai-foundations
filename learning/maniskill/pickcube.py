"""Stage-1 PickCube baseline: episode splits, BC fitting, paired tests and evidence.

Fit and select both clones before running `evaluate`. PPO is trained separately
with experiment.py; all three selected policies use the unchanged 50-step task.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil

from .bc_data import arrays, load_episodes, sha256_file, split_episodes
from .behavior_cloning import BehaviorPolicy, fit
from .experiment import Config, evaluate, save_json
from .metrics import failure_counts, wilson_interval

TASK = Config(
    env_id="PickCube-v1",
    total_timesteps=1048576,
    validation_seed_start=40000,
    validation_episodes=10,
    validation_interval=131072,
    test_seed_start=50000,
    test_episodes=50,
)
VALIDATION_SEEDS = list(range(40000, 40010))
TEST_SEEDS = list(range(50000, 50050))


def fit_clones(dataset, output, updates=10000):
    episodes = load_episodes(dataset, env_id=TASK.env_id)
    training, validation = split_episodes(episodes, 256, 64)
    source_seeds = {e["source_seed"] for e in episodes}
    if source_seeds & set(VALIDATION_SEEDS + TEST_SEEDS):
        raise ValueError("Source demonstration seeds overlap reserved rollout seeds")
    output.mkdir(parents=True, exist_ok=False)
    save_json(
        output / "protocol.json",
        dict(
            task=asdict(TASK),
            dataset_sha256=sha256_file(dataset),
            metadata_sha256=sha256_file(dataset.with_suffix(".json")),
            retained_successful_replays=len(episodes),
            split_seed=2026,
            train_ids=[e["id"] for e in training],
            validation_ids=[e["id"] for e in validation],
            source_seed_values=sorted(source_seeds),
            initial_observation_hashes={
                str(e["id"]): e["initial_hash"] for e in episodes
            },
            validation_rollout_seeds=VALIDATION_SEEDS,
            test_seeds=TEST_SEEDS,
            video_seeds=TEST_SEEDS[:3],
            source="Official ManiSkill RL expert trajectories; successful CPU replays only",
            clipped_scalar_fraction=sum(e["raw_clipped_scalars"] for e in episodes)
            / sum(e["scalar_count"] for e in episodes),
            train_transitions={
                "bc_32": len(arrays(training[:32])[0]),
                "bc_256": len(arrays(training)[0]),
            },
            validation_transitions=len(arrays(validation)[0]),
            success="3D cube-goal distance <= 0.025 m AND max absolute seven arm-joint velocities <= 0.2 rad/s",
            lift_diagnostic="cube center reaches at least 0.05 m above its own reset height; diagnostic only",
            checkpoint_selection="validation success_once, then final success, then negative held-out action MSE for BC or mean return for PPO",
            limitation="One training seed per condition; expert data acquisition cost is not matched to PPO compute",
        ),
    )
    for count, subset in [(32, training[:32]), (256, training)]:
        fit(
            subset,
            validation,
            output / f"bc_{count}",
            updates=updates,
            task=TASK,
            validation_seeds=VALIDATION_SEEDS,
        )


def compare(output, ppo_directory):
    """No fitting or model choice uses these reserved test episodes."""
    from stable_baselines3 import PPO

    saved = json.loads((ppo_directory / "test_ppo.json").read_text())
    if (
        saved["config"] != asdict(TASK)
        or [e["seed"] for e in saved["episodes"]] != TEST_SEEDS
    ):
        raise ValueError(
            "PPO task or test seeds differ from the predeclared comparison"
        )
    for directory in [output / "bc_32", output / "bc_256", ppo_directory]:
        if not json.loads((directory / "training.json").read_text())["complete"]:
            raise ValueError("Finish all fits before final evaluation")
    for name in ["bc_32", "bc_256"]:
        policy = BehaviorPolicy.load(output / name / "best.pt")
        evaluate(TASK, policy, TEST_SEEDS, output / f"test_{name}.json", traces=True)
    # experiment.train already evaluated its selected PPO on exactly these seeds.
    for name in ["ppo", "random", "zero"]:
        shutil.copy2(ppo_directory / f"test_{name}.json", output / f"test_{name}.json")
    save_json(
        output / "ppo_checkpoint.json",
        dict(
            sha256=sha256_file(ppo_directory / "best.zip"),
            selected=json.loads((ppo_directory / "selected.json").read_text()),
            training=json.loads((ppo_directory / "training.json").read_text()),
        ),
    )
    policies = {
        "bc": BehaviorPolicy.load(output / "bc_256/best.pt"),
        "ppo": PPO.load(ppo_directory / "best.zip", device="cpu"),
    }
    for name, policy in policies.items():
        evaluate(
            TASK,
            policy,
            TEST_SEEDS[:3],
            output / f"video_{name}.json",
            traces=True,
            video=True,
        )


def publish(output, ppo_directory, evidence):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    evidence.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in ["bc_32", "bc_256", "ppo", "random", "zero"]:
        result = json.loads((output / f"test_{name}.json").read_text())
        results[name] = result
        for episode in result["episodes"]:
            episode.pop("trajectory", None)
        save_json(evidence / f"test_{name}.json", result)
    comparison = {name: result["summary"] for name, result in results.items()}
    failures = {
        name: failure_counts(result["episodes"]) for name, result in results.items()
    }
    save_json(evidence / "comparison.json", comparison)
    save_json(evidence / "failures.json", failures)
    for name in ["protocol.json", "ppo_checkpoint.json"]:
        shutil.copy2(output / name, evidence / name)
    for name in ["bc_32", "bc_256"]:
        for file in ["config.json", "losses.json", "selected.json", "training.json"]:
            shutil.copy2(output / name / file, evidence / f"{name}_{file}")
        save_json(
            evidence / f"{name}_checkpoint.json",
            dict(sha256=sha256_file(output / name / "best.pt")),
        )
    shutil.copy2(ppo_directory / "validation.json", evidence / "ppo_validation.json")
    shutil.copy2(ppo_directory / "config.json", evidence / "ppo_config.json")
    for name in ["bc", "ppo"]:
        video = json.loads((output / f"video_{name}.json").read_text())
        reference = {
            e["seed"]: e
            for e in results["bc_256" if name == "bc" else name]["episodes"]
        }
        for episode in video["episodes"]:
            expected = reference[episode["seed"]]
            for key, value in expected.items():
                if episode[key] != value:
                    raise ValueError(
                        f"Rendered evaluation changed {name} seed {episode['seed']} field {key}"
                    )
            episode.pop("trajectory", None)
        save_json(evidence / f"video_{name}.json", video)
        for seed in TEST_SEEDS[:3]:
            shutil.copy2(
                output / f"video_{name}_seed{seed}.mp4",
                evidence / f"{name}_seed{seed}.mp4",
            )

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8))
    ppo = json.loads((ppo_directory / "validation.json").read_text())
    axes[0, 0].plot(
        [p["timesteps"] for p in ppo], [p["return_mean"] for p in ppo], marker="o"
    )
    axes[0, 0].set(
        title="PPO validation return",
        xlabel="Environment transitions",
        ylabel="Mean return",
    )
    for name, color in [("bc_32", "#2467ab"), ("bc_256", "#b35d15")]:
        history = json.loads((output / name / "losses.json").read_text())
        for field, style in [("train_mse", "--"), ("validation_mse", "-")]:
            axes[0, 1].plot(
                [p["update"] for p in history],
                [p[field] for p in history],
                color=color,
                linestyle=style,
                label=name + " " + field.removesuffix("_mse"),
            )
        axes[0, 2].plot(
            [p["update"] for p in history],
            [p["rollout"]["success_at_end_rate"] for p in history],
            marker="o",
            label=name,
        )
    axes[0, 1].set(
        title="BC action prediction error",
        xlabel="Gradient updates",
        ylabel="MSE",
        yscale="log",
    )
    axes[0, 1].legend(fontsize=8)
    axes[0, 2].set(
        title="BC validation final success",
        xlabel="Gradient updates",
        ylabel="Success fraction",
        ylim=(-0.02, 1.05),
    )
    axes[0, 2].legend()
    names = list(comparison)
    x = np.arange(len(names))
    axes[1, 0].bar(
        x - 0.18,
        [comparison[n]["success_once_rate"] for n in names],
        0.36,
        label="Any step",
    )
    axes[1, 0].bar(
        x + 0.18,
        [comparison[n]["success_at_end_rate"] for n in names],
        0.36,
        label="At end",
    )
    axes[1, 0].set(
        title="50 reserved test episodes",
        xticks=x,
        xticklabels=names,
        ylabel="Success fraction",
        ylim=(0, 1.05),
    )
    axes[1, 0].legend()
    axes[1, 1].bar(
        x - 0.18,
        [sum(e["grasped_once"] for e in results[n]["episodes"]) / 50 for n in names],
        0.36,
        label="Grasped",
    )
    axes[1, 1].bar(
        x + 0.18,
        [sum(e["lifted_5cm"] for e in results[n]["episodes"]) / 50 for n in names],
        0.36,
        label="Lifted 5 cm",
    )
    axes[1, 1].set(
        title="Diagnostic milestones, not task success",
        xticks=x,
        xticklabels=names,
        ylabel="Episode fraction",
        ylim=(0, 1.05),
    )
    axes[1, 1].legend()
    bottom = np.zeros(len(names))
    colors = ["#2a8a65", "#c55353", "#df993f", "#8586b8", "#4b82b4", "#a9b5c2"]
    for category, color in zip(next(iter(failures.values())), colors):
        values = np.array([failures[n][category] for n in names])
        axes[1, 2].bar(
            x, values, bottom=bottom, color=color, label=category.replace("_", " ")
        )
        bottom += values
    axes[1, 2].set(
        title="Mutually exclusive final outcomes",
        xticks=x,
        xticklabels=names,
        ylabel="Episodes",
        ylim=(0, 64),
    )
    axes[1, 2].legend(fontsize=7, ncol=2, loc="upper center")
    for ax in axes[1]:
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(evidence / "results.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    rows = "".join(
        f"<tr><td>{n}</td><td>{s['successes']}/50</td><td>{round(s['success_at_end_rate'] * 50)}/50</td><td>{s['return_mean']:.2f}</td></tr>"
        for n, s in comparison.items()
    )
    cards = ""
    for seed in TEST_SEEDS[:3]:
        for name in ["bc", "ppo"]:
            e = next(
                e
                for e in results["bc_256" if name == "bc" else name]["episodes"]
                if e["seed"] == seed
            )
            cards += f'<article><h3>{name.upper()} · seed {seed}</h3><video controls muted preload="metadata" src="{name}_seed{seed}.mp4"></video><p>Final success: {e["success_at_end"]}. Final distance: {e["final_goal_distance"]:.3f} m.</p></article>'
    template = Path(__file__).with_name("pickcube_report.html").read_text()
    html = template.replace("{{ROWS}}", rows).replace("{{CARDS}}", cards)
    (evidence / "index.html").write_text(html)
    for name, result in results.items():
        final = sum(e["success_at_end"] for e in result["episodes"])
        print(
            name,
            comparison[name],
            "final 95% Wilson",
            wilson_interval(final, 50),
            failures[name],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fit", "evaluate", "publish"])
    parser.add_argument("--dataset", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("learning/runs/pickcube_bc")
    )
    parser.add_argument(
        "--ppo", type=Path, default=Path("learning/runs/pickcube_ppo_seed42")
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("learning/maniskill/evidence/stage1_pickcube"),
    )
    parser.add_argument("--updates", type=int, default=10000)
    args = parser.parse_args()
    if args.command == "fit":
        if not args.dataset:
            parser.error("fit requires --dataset")
        fit_clones(args.dataset, args.output, args.updates)
    elif args.command == "evaluate":
        compare(args.output, args.ppo)
    else:
        publish(args.output, args.ppo, args.evidence)


if __name__ == "__main__":
    main()
