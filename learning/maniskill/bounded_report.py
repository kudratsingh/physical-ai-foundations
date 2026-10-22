"""Export the predeclared action-distribution experiment without rerunning tests."""

import argparse
import json
from pathlib import Path
import shutil
import statistics

from .experiment import save_json
from .metrics import failure_counts, wilson_interval

NAMES = ["raw_final", "bounded_final", "bounded_selected_secondary"]


def report(run, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = [
        "config.json",
        "training.json",
        "validation.json",
        "selected.json",
        "comparison.json",
        "provenance.json",
        "loaded_recipe_contracts.json",
    ] + [f"test_{name}.json" for name in NAMES]
    files += [f"video_{name}.json" for name in NAMES[:2]]
    files += [path.name for path in run.glob("validation_trace_*.json")]
    if (run / "command_audit.json").exists():
        files.append("command_audit.json")
    for name in files:
        shutil.copy2(run / name, output / name)
    for path in run.glob("video_*_seed*.mp4"):
        shutil.copy2(path, output / path.name)
    tests = {
        name: json.loads((run / f"test_{name}.json").read_text()) for name in NAMES
    }
    validation = json.loads((run / "validation.json").read_text())
    config = json.loads((run / "config.json").read_text())
    failures = {name: failure_counts(data["episodes"]) for name, data in tests.items()}
    geometry = {
        name: dict(
            median_goal_distance=statistics.median(
                e["final_goal_distance"] for e in data["episodes"]
            ),
            median_max_arm_speed=statistics.median(
                max(map(abs, e["final_arm_qvel"])) for e in data["episodes"]
            ),
            final_static_count=sum(
                e["final_is_robot_static"] for e in data["episodes"]
            ),
            final_success_count=sum(e["success_at_end"] for e in data["episodes"]),
        )
        for name, data in tests.items()
    }
    paired = {"both_success": 0, "bounded_only": 0, "raw_only": 0, "neither_success": 0}
    for raw, bounded in zip(
        tests["raw_final"]["episodes"], tests["bounded_final"]["episodes"], strict=True
    ):
        if raw["seed"] != bounded["seed"]:
            raise ValueError("Primary test seeds are not paired")
        paired[
            {
                (True, True): "both_success",
                (False, True): "bounded_only",
                (True, False): "raw_only",
                (False, False): "neither_success",
            }[bool(raw["success_at_end"]), bool(bounded["success_at_end"])]
        ] += 1
    save_json(output / "failure_counts.json", failures)
    save_json(output / "geometry.json", geometry)
    save_json(output / "paired_outcomes.json", paired)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = [p["timesteps"] / 1e6 for p in validation]
    axes[0, 0].plot(
        x,
        [100 * p["success_at_end_rate"] for p in validation],
        "o-",
        label="Final success",
    )
    axes[0, 0].plot(
        x,
        [100 * p["success_once_rate"] for p in validation],
        "s--",
        label="Any-step success",
    )
    axes[0, 0].set(
        title="Bounded policy validation · 10 fixed resets",
        xlabel="Training transitions (millions)",
        ylabel="Success (%)",
        ylim=(-2, 102),
    )
    axes[0, 0].legend()
    axes[0, 1].errorbar(
        x,
        [p["return_mean"] for p in validation],
        yerr=[p["return_std"] for p in validation],
        fmt="o-",
    )
    axes[0, 1].set(
        title="Unchanged task reward · mean ± sample SD",
        xlabel="Training transitions (millions)",
        ylabel="Return",
    )
    for i, name in enumerate(NAMES):
        count = geometry[name]["final_success_count"]
        n = len(tests[name]["episodes"])
        lo, hi = wilson_interval(count, n)
        rate = count / n
        axes[1, 0].bar(i, rate * 100)
        axes[1, 0].errorbar(
            i,
            rate * 100,
            yerr=[[100 * (rate - lo)], [100 * (hi - rate)]],
            fmt="none",
            color="black",
            capsize=5,
        )
        axes[1, 0].text(i, rate * 100 + 10, f"{count}/{n}", ha="center")
    axes[1, 0].set(
        title="Fresh paired test resets · 95% Wilson intervals",
        ylabel="Final success (%)",
        ylim=(0, 110),
        xticks=range(3),
        xticklabels=["Raw final", "Bounded final", "Selected bounded\n(secondary)"],
    )
    for name in NAMES[:2]:
        episodes = tests[name]["episodes"]
        axes[1, 1].scatter(
            [e["final_goal_distance"] for e in episodes],
            [max(map(abs, e["final_arm_qvel"])) for e in episodes],
            label=name,
            alpha=0.6,
        )
    axes[1, 1].axvline(0.025, color="green", linestyle="--")
    axes[1, 1].axhline(0.2, color="green", linestyle="--")
    axes[1, 1].set(
        title="Both placement and stillness are required",
        xlabel="Final cube–goal distance (m)",
        ylabel="Maximum final arm speed (rad/s)",
    )
    axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(output / "results.png", dpi=160)
    plt.close(fig)
    from .bounded_trace import plots

    plots(run, output)
    rows = ""
    for name in NAMES:
        data = tests[name]
        metrics = geometry[name]
        summary = data["summary"]
        rows += f"<tr><th>{name}</th><td>{metrics['final_success_count']}/50</td><td>{summary['successes']}/50</td><td>{summary['return_mean']:.3f} ± {summary['return_std']:.3f}</td><td>{100 * metrics['median_goal_distance']:.2f} cm</td><td>{metrics['median_max_arm_speed']:.3f} rad/s</td><td>{metrics['final_static_count']}/50</td></tr>"
    videos = ""
    for seed in config["video_seeds"]:
        videos += f'<section class="pair"><h3>Predeclared test seed {seed}</h3>'
        for name in NAMES[:2]:
            videos += f'<figure><video controls preload="metadata" src="video_{name}_seed{seed}.mp4"></video><figcaption>{name}</figcaption></figure>'
        videos += "</section>"
    links = " · ".join(
        f'<a href="{name}">{name}</a>'
        for name in files
        + ["geometry.json", "failure_counts.json", "paired_outcomes.json"]
    )
    (
        output / "index.html"
    ).write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PickCube bounded-action experiment</title><style>
body{{font:16px/1.6 system-ui;margin:0;background:#eef3f5;color:#142d3b}}main{{max-width:1180px;margin:auto;padding:28px}}h1{{font-size:36px}}img,video{{width:100%}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #ddd}}.scroll{{overflow:auto}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:20px;background:white;padding:20px;margin:20px 0}}.pair h3{{grid-column:1/-1}}figure{{margin:0}}a{{color:#00617a}}@media(max-width:700px){{.pair{{grid-template-columns:1fr}}main{{padding:14px}}h1{{font-size:28px}}}}
</style><main><p>Stage 1 · one-change action-distribution ablation</p><h1>Can a bounded PPO policy learn to place and stop?</h1>
<p>Primary fixed-budget comparison: original raw-state Gaussian PPO versus a tanh-transformed Gaussian trained from scratch. Each uses 1,048,576 transitions, seed42 and four workers. Actor/critic initial tensors match exactly. Reward, controller, observation, horizon and all PPO settings are unchanged. The bounded distribution includes the change-of-variables log probability; this is not post-hoc action clipping or a privileged stopping override.</p>
<p>Official success requires 3D goal error ≤2.5cm and all seven arm speeds ≤0.2rad/s. Evaluation runs all50steps, even after first success. Tanh can still approach the action boundary; bounded output is not proof of good control.</p>
<div class="scroll"><table><tr><th>Checkpoint</th><th>Final success</th><th>Any success</th><th>Return: mean ± SD</th><th>Median goal error</th><th>Median maximum arm speed</th><th>Static at end</th></tr>{rows}</table></div>
<img src="results.png" alt="Validation success and return, paired final success and final geometry">
<h2>Synchronized validation commands</h2><p>Ten paired validation resets show the median and interquartile range of commands and their resulting states. The exploration standard deviation is measured in latent Gaussian coordinates, before tanh.</p><img src="commands.png" alt="Paired goal distance, arm speed, executed commands, and latent exploration standard deviations">
<h2>Native rollouts</h2><p>First two predeclared test seeds, without success-based selection. Each video reproduces every stored non-trajectory outcome field.</p>{videos}
<h2>Protocol and limits</h2><p>Validation100000–100009; fresh paired tests110000–110049. Checkpoint selection now favors final success, then any-step success, then return. Validation callbacks run before the corresponding optimizer update; fixed-budget final checkpoints are saved after the last update. Selected-policy results are secondary because the original raw run used a different validation split. One training seed cannot establish algorithm-level improvement. The original raw run occurred earlier; training runtime is descriptive. The predeclared replication trigger is at least5/50 bounded final successes and improvement over the paired raw control.</p><p>Full evidence: {links}</p></main></html>""")
    return geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.run, args.output)


if __name__ == "__main__":
    main()
