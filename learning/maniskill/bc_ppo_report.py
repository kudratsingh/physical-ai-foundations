"""Compact evidence and native visual report for the paired BC/PPO experiment."""

import argparse
import json
import shutil
import statistics
from pathlib import Path

from .experiment import save_json
from .metrics import failure_counts

NAMES = [
    "teacher",
    "warm_initialized",
    "warm_final",
    "control_final",
    "warm_selected_secondary",
]
LABELS = [
    "BC teacher",
    "PPO initialized",
    "PPO warm final",
    "Matched random final",
    "Selected warm (secondary)",
]


def jsonl(path, episodes):
    path.write_text(
        "".join(
            json.dumps(e, separators=(",", ":"), allow_nan=False) + "\n"
            for e in episodes
        )
    )


def optimizer_trace(path):
    """Read SB3 console tables; logged optimizer statistics precede that update."""
    import math
    import re

    points, point = [], {}

    def append_point():
        if "total_timesteps" in point:
            required = {"value_loss", "approx_kl", "n_updates"}
            if not required <= point.keys():
                raise ValueError(
                    f"Incomplete optimizer table: {required - point.keys()}"
                )
            points.append(point)

    for line in Path(path).read_text().splitlines():
        if line.startswith("---"):
            append_point()
            point = {}
        match = re.match(r"^\|\s*([a-z_]+)\s*\|\s*(\S+)\s*\|", line)
        if match:
            try:
                value = float(match[2])
            except ValueError:
                continue
            if not math.isfinite(value):
                raise ValueError("Non-finite optimizer log")
            point[match[1]] = value
    append_point()
    return points


def report(run, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in [
        "protocol.json",
        "preflight.json",
        "comparison.json",
        "provenance.json",
    ]:
        shutil.copy2(run / name, output / name)
    if (run / "launch_sources").exists():
        shutil.copytree(
            run / "launch_sources", output / "launch_sources", dirs_exist_ok=True
        )
    if (run / "excluded_pilot.json").exists():
        shutil.copy2(run / "excluded_pilot.json", output / "excluded_pilot.json")
    for arm in ["warm", "control"]:
        for name in [
            "initialization",
            "validation",
            "selected",
            "training",
            "training_metrics",
        ]:
            shutil.copy2(run / arm / f"{name}.json", output / f"{arm}_{name}.json")
    results = {
        name: json.loads((run / f"test_{name}.json").read_text()) for name in NAMES
    }
    for name, result in results.items():
        jsonl(output / f"test_{name}.jsonl", result["episodes"])
    for name in ["preflight_teacher", "preflight_initialized"]:
        jsonl(
            output / f"{name}.jsonl",
            json.loads((run / f"{name}.json").read_text())["episodes"],
        )
    protocol = json.loads((run / "protocol.json").read_text())
    manifest = []
    for name in ["teacher", "warm_final", "control_final"]:
        video = json.loads((run / f"video_{name}.json").read_text())
        for episode in video["episodes"]:
            video_name = f"video_{name}_seed{episode['seed']}.mp4"
            shutil.copy2(run / video_name, output / video_name)
            import hashlib

            expected = next(
                e for e in results[name]["episodes"] if e["seed"] == episode["seed"]
            )
            assert {k: v for k, v in episode.items() if k != "trajectory"} == {
                k: v for k, v in expected.items() if k != "trajectory"
            }
            manifest.append(
                {
                    "policy": name,
                    "seed": episode["seed"],
                    "file": video_name,
                    "sha256": hashlib.sha256(
                        (output / video_name).read_bytes()
                    ).hexdigest(),
                    "all_nontrajectory_fields_match": True,
                }
            )
    save_json(output / "video_manifest.json", manifest)
    geometry = {
        name: {
            "final_successes": sum(e["success_at_end"] for e in data["episodes"]),
            "median_goal_distance": statistics.median(
                e["final_goal_distance"] for e in data["episodes"]
            ),
            "median_max_arm_speed": statistics.median(
                max(map(abs, e["final_arm_qvel"])) for e in data["episodes"]
            ),
            "failures": failure_counts(data["episodes"]),
        }
        for name, data in results.items()
    }
    save_json(output / "geometry.json", geometry)
    validation = {
        arm: json.loads((run / arm / "validation.json").read_text())
        for arm in ["warm", "control"]
    }
    windows = {
        arm: json.loads((run / arm / "training_metrics.json").read_text())["windows"]
        for arm in ["warm", "control"]
    }
    totals = {
        arm: {
            key: sum(w.get(key, 0) for w in rows)
            for key in set().union(*(set(w) - {"timesteps"} for w in rows))
        }
        for arm, rows in windows.items()
    }
    save_json(output / "training_totals.json", totals)
    optimization = {}
    for arm in ["warm", "control"]:
        path = run / f"{arm}_training.log"
        if not path.exists():
            raise FileNotFoundError(f"Missing optimizer console log: {path}")
        optimization[arm] = optimizer_trace(path)
    if optimization:
        save_json(output / "optimizer_trace.json", optimization)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for arm, label in [("warm", "BC initialized"), ("control", "Matched random")]:
            points = optimization[arm]
            x = [p["total_timesteps"] / 1e6 for p in points]
            axes[0].plot(x, [p["value_loss"] for p in points], label=label)
            axes[1].plot(x, [p["approx_kl"] for p in points], label=label)
        axes[0].set(
            yscale="symlog", title="Reported critic loss", ylabel="Value loss (symlog)"
        )
        axes[1].set(title="Reported policy update size", ylabel="Approximate KL")
        for ax in axes:
            ax.set_xlabel("Training transitions (millions)")
            ax.legend()
        fig.tight_layout()
        fig.savefig(output / "optimizer.png", dpi=160)
        plt.close(fig)
    fig, grid = plt.subplots(2, 2, figsize=(12, 8))
    axes = grid.ravel()
    for arm, label in [("warm", "BC initialized"), ("control", "Matched random")]:
        points = validation[arm]
        rows = windows[arm]
        axes[0].plot(
            [p["timesteps"] / 1e6 for p in points],
            [100 * p["success_at_end_rate"] for p in points],
            "o-",
            label=label,
        )
        axes[1].plot(
            [w["timesteps"] / 1e6 for w in rows],
            [
                100 * w.get("episodes_final_success", 0) / max(w.get("episodes", 0), 1)
                for w in rows
            ],
            label=label,
        )
        axes[2].plot(
            [w["timesteps"] / 1e6 for w in rows],
            [
                100 * w.get("grasped_within_2_5cm", 0) / max(w["grasped"], 1)
                for w in rows
            ],
            label=label,
        )
    axes[0].axhline(
        100
        * json.loads((run / "preflight.json").read_text())["teacher_summary"][
            "success_at_end_rate"
        ],
        color="gray",
        linestyle=":",
        label="Teacher validation",
    )
    for ax, title, ylabel in zip(
        axes[:3],
        [
            "Deterministic validation final success",
            "Stochastic training episode final success",
            "Training placement occupancy while grasped",
        ],
        [
            "Final success (%)",
            "Completed training episodes (%)",
            "Grasped steps within 2.5 cm (%)",
        ],
        strict=True,
    ):
        ax.set(
            title=title,
            xlabel="Training transitions (millions)",
            ylabel=ylabel,
            ylim=(-2, 102),
        )
        ax.legend(fontsize=8)
    bars = axes[3].bar(
        np.arange(5),
        [100 * results[n]["summary"]["success_at_end_rate"] for n in NAMES],
    )
    axes[3].bar_label(bars, fmt="%.0f%%", padding=3)
    axes[3].set(
        title="Fresh paired tests · 50 resets",
        ylabel="Final success (%)",
        ylim=(0, 110),
        xticks=np.arange(5),
        xticklabels=[
            "Teacher",
            "Initialized",
            "Warm final",
            "Random final",
            "Selected\nsecondary",
        ],
    )
    fig.tight_layout()
    fig.savefig(output / "learning.png", dpi=160)
    plt.close(fig)
    rows = ""
    for name, label in zip(NAMES, LABELS, strict=True):
        s = results[name]["summary"]
        g = geometry[name]
        rows += f"<tr><th>{label}</th><td>{g['final_successes']}/50</td><td>{s['successes']}/50</td><td>{s['return_mean']:.2f} ± {s['return_std']:.2f}</td><td>{100 * g['median_goal_distance']:.2f} cm</td><td>{g['median_max_arm_speed']:.3f} rad/s</td></tr>"
    trainingrows = ""
    for arm, label in [("warm", "BC initialized"), ("control", "Matched random")]:
        t = totals[arm]
        trainingrows += f"<tr><th>{label}</th><td>{t['transitions']:,}</td><td>{t['grasped']:,}</td><td>{t['placed']:,}</td><td>{t['success']:,}</td><td>{t.get('episodes_any_success', 0):,}/{t.get('episodes', 0):,}</td><td>{t.get('episodes_final_success', 0):,}/{t.get('episodes', 0):,}</td></tr>"
    videos = ""
    for seed in protocol["video_seeds"]:
        videos += f'<section class="videos"><h3>Predeclared test seed {seed}</h3>'
        for name, label in [
            ("teacher", "BC teacher"),
            ("warm_final", "PPO warm final"),
            ("control_final", "Matched random final"),
        ]:
            videos += f'<figure><video controls preload="metadata" src="video_{name}_seed{seed}.mp4"></video><figcaption>{label}</figcaption></figure>'
        videos += "</section>"
    selected = json.loads((run / "warm/selected.json").read_text())
    (
        output / "index.html"
    ).write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PickCube BC-initialized PPO</title><style>
body{{font:16px/1.6 system-ui;margin:0;background:#f0f4f6;color:#1d3342}}main{{max-width:1220px;margin:auto;padding:30px 20px}}h1{{font-size:38px;line-height:1.2}}.notice{{background:#fff1cf;padding:18px;border-left:5px solid #ad7c2a}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:11px;text-align:left;border-bottom:1px solid #ddd}}.scroll{{overflow:auto}}img,video{{width:100%}}.videos{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;background:white;padding:20px;margin:24px 0}}.videos h3{{grid-column:1/-1}}figure{{margin:0}}a{{color:#126f83}}@media(max-width:750px){{.videos{{grid-template-columns:1fr}}main{{padding:15px}}h1{{font-size:29px}}}}
</style><main><p>Stage 1 · fresh paired training · no stopping postprocessor</p><h1>Can PPO preserve and improve a cloned manipulation policy?</h1>
<p><strong>Selected warm PPO: {geometry["warm_selected_secondary"]["final_successes"]}/50 final successes; BC teacher: {geometry["teacher"]["final_successes"]}/50. Continued fine-tuning to the fixed budget regressed to {geometry["warm_final"]["final_successes"]}/50, matching the random control’s {geometry["control_final"]["final_successes"]}/50. The selected result is secondary; stable PPO improvement remains unproven.</strong></p>
<div class="notice">One training seed per arm. The matched random actor shares BC-derived preprocessing, ReLU architecture, exploration, critic initialization and optimizer settings. It is not the archived original Gaussian PPO recipe. Higher return with lower final success is regression. Privileged or observation-only hybrid recoveries are excluded.</div>
<p>The actor transplant reproduced BC exactly on 500 validation observations and ten full validation rollouts. Both actual initial models, critic/non-actor fingerprints and RNG digests were saved. Each learner then received 1,048,576 transitions with the original reward, controller, horizon and task success rule.</p>
<div class="scroll"><table><tr><th>Policy</th><th>Final success</th><th>Any success</th><th>Return: mean ± SD</th><th>Median goal error</th><th>Median maximum arm speed</th></tr>{rows}</table></div><img src="learning.png" alt="Deterministic validation, stochastic training success and placement occupancy, and fresh test success">
<h2>What happened during stochastic training?</h2><p>These counts exclude validation and tests. Correlated training steps are not independent trials. Official task success does not itself require a recorded grasp; rare successful states do not establish learned picking. Episode counts exclude unfinished episodes at the final rollout boundary.</p><div class="scroll"><table><tr><th>Learner</th><th>Transitions</th><th>Grasped steps</th><th>Placed steps</th><th>Successful steps</th><th>Any-success episodes</th><th>Final-success episodes</th></tr>{trainingrows}</table></div>
<h2>Optimizer stability</h2><p>These console metrics describe the preceding optimizer update at each logging point. Critic feature clipping bounds inputs, not learned values or bootstrapped targets. Rising value loss is not by itself proof of a code-path bug.</p><img src="optimizer.png" alt="Critic value loss and approximate policy KL over training"><p><a href="optimizer_trace.json">Optimizer trace</a></p>
<h2>Native held-out rollouts</h2><p>The first two predeclared test seeds, without outcome-based video selection. All stored non-trajectory fields match the corresponding test episodes.</p>{videos}
<h2>Initialization, safeguards and selection</h2><p>The actor keeps fixed BC normalization and ReLU weights. Ordinary Gaussian exploration is centered on the actor’s tanh-bounded mean, so PPO stores and scores the raw sample without inverse-tanh boundary inversion. Only critic inputs are clipped to ±10. Both arms share initial std exp(−3), learning rate 0.00003 and target-KL 0.01. KL early stopping may produce different numbers of optimizer updates; transition budgets match. No BC auxiliary loss, critic warmup, curriculum or reward modification is used.</p><p>The selected warm checkpoint is step {selected["timesteps"]:,}, ranked by final success, then any-step success, then return. It is secondary; a step-zero selection is not successful PPO improvement. Validation runs before the corresponding optimizer update and final weights after the last update.</p>
<p>An earlier squashed-Gaussian implementation pilot produced NaNs and was excluded before final testing. The revised recipe was redeclared after exact transfer and finite stochastic-update checks; its runs start fresh. Historical raw/bounded comparison caveats are not retroactively erased.</p>
<h2>Evidence</h2><p><a href="protocol.json">Protocol, versions and source hashes</a> · <a href="preflight.json">Teacher-transfer verification</a> · <a href="provenance.json">Loaded-checkpoint and source verification</a> · <a href="comparison.json">Test metrics</a> · <a href="geometry.json">Geometry and failures</a> · <a href="warm_initialization.json">Actual warm initialization</a> · <a href="control_initialization.json">Actual control initialization</a> · <a href="training_totals.json">Training totals</a> · <a href="warm_training_metrics.json">Warm stochastic windows</a> · <a href="control_training_metrics.json">Control stochastic windows</a> · <a href="video_manifest.json">Video hashes</a></p></main></html>""")
    return geometry, totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.run, args.output)


if __name__ == "__main__":
    main()
