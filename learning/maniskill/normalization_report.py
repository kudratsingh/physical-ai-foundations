"""Export the fixed-budget normalization assignment and its native visual evidence."""

import argparse
import json
from pathlib import Path
import shutil

from .experiment import save_json

NAMES = ["raw_final", "normalized_final", "normalized_selected_secondary"]
LABELS = ["Raw final", "Normalized final", "Selected normalized\n(secondary)"]


def report(run, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = [
        "config.json",
        "training.json",
        "baseline.json",
        "selected.json",
        "validation.json",
        "comparison.json",
        "loaded_policy_contracts.json",
        "failures.json",
    ]
    files += [f"test_{name}.json" for name in NAMES]
    files += [f"video_{name}.json" for name in NAMES[:2]]
    for name in files:
        shutil.copy2(run / name, output / name)
    for video in run.glob("video_*_seed*.mp4"):
        shutil.copy2(video, output / video.name)
    selected = json.loads((run / "selected.json").read_text())
    for prefix, snapshot in [("final", "final"), ("selected", selected["snapshot"])]:
        for name in ["snapshot.json", "statistics.json"]:
            shutil.copy2(run / snapshot / name, output / f"{prefix}_{name}")
    comparison = json.loads((run / "comparison.json").read_text())
    validation = json.loads((run / "validation.json").read_text())
    failures = json.loads((run / "failures.json").read_text())
    results = {
        name: json.loads((run / f"test_{name}.json").read_text()) for name in NAMES
    }
    final_stats = json.loads((run / "final/statistics.json").read_text())
    config = json.loads((run / "config.json").read_text())
    baseline = json.loads((run / "baseline.json").read_text())
    training = json.loads((run / "training.json").read_text())

    # Independent matched-pair accounting: seed is the shared reset condition.
    raw = results["raw_final"]["episodes"]
    treatment = results["normalized_final"]["episodes"]
    paired = dict(both_success=0, normalized_only=0, raw_only=0, neither_success=0)
    for a, b in zip(raw, treatment, strict=True):
        if a["seed"] != b["seed"]:
            raise ValueError("Comparison seeds are not paired")
        status = (bool(a["success_at_end"]), bool(b["success_at_end"]))
        key = {
            (True, True): "both_success",
            (False, True): "normalized_only",
            (True, False): "raw_only",
            (False, False): "neither_success",
        }[status]
        paired[key] += 1
    save_json(output / "paired_final_outcomes.json", paired)

    fig, grid = plt.subplots(2, 2, figsize=(12, 8))
    axes = grid.ravel()
    x = [p["timesteps"] / 1e6 for p in validation]
    axes[0].plot(
        x, [p["success_once_rate"] * 100 for p in validation], "o-", label="Any success"
    )
    axes[0].plot(
        x,
        [p["success_at_end_rate"] * 100 for p in validation],
        "s-",
        label="Final success",
    )
    axes[0].set(
        xlabel="Normalized PPO training (million transitions)",
        ylabel="Success (%)",
        ylim=(-3, 103),
        title="10 validation seeds; fixed throughout",
    )
    axes[0].legend()
    axes[1].errorbar(
        x,
        [p["return_mean"] for p in validation],
        yerr=[p["return_std"] for p in validation],
        fmt="o-",
        capsize=3,
        color="#207665",
    )
    axes[1].set(
        xlabel="Normalized PPO training (million transitions)",
        ylabel="Raw episode return (mean ± sample SD)",
        title="Same 10 validation seeds; original reward units",
    )
    positions = np.arange(3)
    for offset, field, label in [
        (-0.18, "success_once_rate", "Any success"),
        (0.18, "success_at_end_rate", "Final success"),
    ]:
        bars = axes[2].bar(
            positions + offset,
            [comparison[n][field] * 100 for n in NAMES],
            0.36,
            label=label,
        )
        axes[2].bar_label(bars, fmt="%.0f%%", padding=3)
    axes[2].set(
        xticks=positions,
        xticklabels=LABELS,
        ylim=(0, 105),
        ylabel="Success (%)",
        title="50 fresh paired test seeds",
    )
    axes[2].legend()
    axes[3].scatter(
        [a["final_goal_distance"] for a in raw],
        [b["final_goal_distance"] for b in treatment],
        alpha=0.65,
        s=24,
    )
    limit = (
        max(
            max(a["final_goal_distance"] for a in raw),
            max(b["final_goal_distance"] for b in treatment),
        )
        * 1.05
    )
    axes[3].plot([0, limit], [0, limit], "--", color="gray", linewidth=1)
    axes[3].axhline(0.025, color="#18846b", alpha=0.7)
    axes[3].axvline(0.025, color="#18846b", alpha=0.7)
    axes[3].set(
        xlabel="Raw final cube-goal distance (m)",
        ylabel="Normalized final cube-goal distance (m)",
        title="Distance alone excludes static-arm criterion",
    )
    fig.tight_layout()
    fig.savefig(output / "results.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    rows = "".join(
        f"<tr><td>{LABELS[i].replace(chr(10), ' ')}</td><td>{comparison[n]['successes']}/50</td>"
        f"<td>{round(comparison[n]['success_at_end_rate'] * 50)}/50</td>"
        f"<td>{comparison[n]['return_mean']:.2f} ± {comparison[n]['return_std']:.2f}</td></tr>"
        for i, n in enumerate(NAMES)
    )
    seed_start = config["test_seed_start"]
    videos = "".join(
        f'<div class="pair"><h3>Seed {seed}</h3>'
        + "".join(
            f'<figure><video controls muted loop preload="metadata" src="video_{name}_seed{seed}.mp4"></video>'
            f"<figcaption>{label}: final success {results[name]['episodes'][seed - seed_start]['success_at_end']}; "
            f"distance {results[name]['episodes'][seed - seed_start]['final_goal_distance']:.3f} m</figcaption></figure>"
            for name, label in zip(NAMES[:2], LABELS[:2])
        )
        + "</div>"
        for seed in config["video_seeds"]
    )
    failure_rows = "".join(
        f"<tr><td>{category.replace('_', ' ')}</td>"
        + "".join(f"<td>{failures[n][category]}</td>" for n in NAMES)
        + "</tr>"
        for category in failures[NAMES[0]]
    )
    delta = (
        comparison["normalized_final"]["success_at_end_rate"]
        - comparison["raw_final"]["success_at_end_rate"]
    ) * 100
    conclusion = (
        f"Observation normalization changed final success by {delta:+.0f} percentage points on this paired test set. "
        "This is one seed per treatment: it identifies a useful next experiment, not a multi-seed algorithm ranking."
    )
    audit_section = ""
    if (output / "action_saturation.json").exists():
        audit = json.loads((output / "action_saturation.json").read_text())
        raw_rate = audit["raw_final"]["arm_any_mean_outside_bounds_fraction"]
        normalized_rate = audit["normalized_final"][
            "arm_any_mean_outside_bounds_fraction"
        ]
        audit_section = (
            "<h2>Command-bound diagnostic</h2><p>On ten validation seeds, at least one "
            f"arm-joint mean exceeded the action bounds on {raw_rate:.1%} of raw-policy "
            f"steps and {normalized_rate:.1%} of normalized-policy steps. Clipping the means "
            "exactly reproduced every deterministic command. Gripper closure is excluded. "
            "This is physical command clipping, separate from PPO probability-ratio clipping; "
            "it motivates a controlled action-distribution experiment without establishing a cause. "
            '<a href="action_saturation.json">All 500 means per policy and checkpoint hashes</a>.</p>'
        )
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PickCube: observation normalization</title><style>
body{{font:16px/1.6 system-ui;background:#f2f5f4;color:#172b2a;margin:0}}main{{max-width:1150px;margin:auto;padding:36px 22px}}h1{{font-size:38px;line-height:1.15}}h2{{margin-top:42px}}.tag{{color:#207665;font-weight:700}}.card,.pair{{background:white;border-radius:15px;padding:22px;margin:22px 0}}table{{border-collapse:collapse;width:100%}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #dce5e2}}img{{width:100%}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.pair h3{{grid-column:1/-1;margin:0}}figure{{margin:0}}video{{width:100%;background:#ddd}}figcaption{{font-size:14px}}code{{font-size:14px}}a{{color:#116e65}}@media(max-width:700px){{.pair{{grid-template-columns:1fr}}main{{padding:20px 12px}}h1{{font-size:30px}}td,th{{padding:6px}}}}
</style><main><p class="tag">Stage 1 · controlled PPO assignment · Panda PickCube-v1</p>
<h1>Does scaling the observations improve grasp, place and hold?</h1>
<p>{conclusion}</p><div class="card"><p><b>Only treatment:</b> online SB3 VecNormalize observations, clip ±10; rewards stay unnormalized. No warmup. Both models train from scratch for 1,048,576 transitions with seed 42, four CPU workers and the same PPO architecture, hyperparameters and upstream task.</p>
<p><b>Primary comparison:</b> fixed-budget final checkpoints. Fresh test seeds 90000–90049; 50 control steps per episode. Official success requires cube-to-goal 3D distance ≤0.025 m <em>and</em> all seven arm-joint speeds ≤0.2 rad/s. Episodes continue through the full horizon after first success.</p>
<table><thead><tr><th>Checkpoint</th><th>Any success</th><th>Final success</th><th>Raw return: mean ± sample SD</th></tr></thead><tbody>{rows}</tbody></table></div>
<img src="results.png" alt="Validation learning curves, test success bars, and paired goal distances">
<h2>Native simulator rollouts</h2><p>The first three predeclared test seeds, with identical reset conditions for both policies. These are actual simulator frames; no success-based video selection. Use the video controls to compare the full episodes.</p>{videos}
<h2>Where episodes failed</h2><table><tr><th>Outcome</th><th>Raw final</th><th>Normalized final</th><th>Selected normalized, secondary</th></tr>{failure_rows}</table><p>Categories are mutually exclusive. “Placed but moving” means placement was reached at least once without any full official success; it does not assert the object stayed at the goal. “Lifted” means ≥5 cm above that episode’s initial cube height, a diagnostic rather than success.</p>
<h2>Statistics and selection</h2><p>Normalization uses only observations collected by the four training workers: count {final_stats["count"]:.4f}, {len(final_stats["mean"])} observation components. Snapshots save the model and running statistics together with SHA-256 checksums. Evaluation loads a separate copy with <code>training=False</code> and <code>norm_reward=False</code>; validation never updates training statistics.</p><p>The selected normalized checkpoint is step {selected["timesteps"]:,}, chosen by success at any time, then final success, then return on seeds 80000–80009. It is secondary because raw PPO originally used different selection seeds. The primary final checkpoint comparison does not use selection. Training took {training["elapsed_seconds"]:.1f} seconds versus {baseline["training"]["elapsed_seconds"]:.1f} seconds for the existing raw baseline.</p>
{audit_section}
<h2>Evidence and limitations</h2><p>One training seed per treatment. Fifty shared resets quantify these two policies, not variation over independent training runs. Training was performed on the same Mac but at different times; elapsed time is descriptive. The existing raw run was not retrained and used older validation seeds; validation does not affect PPO weights. No task reward, action scale, time horizon, curriculum, demonstration data or network architecture was changed.</p><p><a href="config.json">Protocol and versions</a> · <a href="comparison.json">Metrics</a> · <a href="failures.json">Failure counts</a> · <a href="paired_final_outcomes.json">Paired final outcomes</a> · <a href="baseline.json">Raw checkpoint provenance</a> · <a href="loaded_policy_contracts.json">Loaded policy contracts</a> · <a href="final_statistics.json">Final observation statistics</a> · <a href="final_snapshot.json">Final snapshot checksums</a></p></main></html>"""
    (output / "index.html").write_text(html)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.run, args.output)


if __name__ == "__main__":
    main()
