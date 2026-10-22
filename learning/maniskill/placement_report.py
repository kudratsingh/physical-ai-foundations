"""Export compact P0 placement diagnostics and native validation videos."""

import argparse
import json
from pathlib import Path
import shutil

from .placement_diagnostic import load_jsonl

NAMES = ["bc_256", "raw_final", "normalized_final"]
LABELS = ["BC 256", "Raw PPO", "Normalized PPO"]
COLORS = ["#187868", "#bd5146", "#b98021"]


def report(run, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for pattern in ["*.json", "*.jsonl", "*.mp4"]:
        for path in run.glob(pattern):
            shutil.copy2(path, output / path.name)
    summary = json.loads((run / "placement_summary.json").read_text())
    hybrid = json.loads((run / "hybrid_comparison.json").read_text())
    protocol = json.loads((run / "protocol.json").read_text())
    episodes = {name: load_jsonl(run / f"placement_{name}.jsonl") for name in NAMES}
    fig, grid = plt.subplots(2, 2, figsize=(12, 8))
    axes = grid.ravel()
    positions = np.arange(3)
    for offset, field, label in [
        (-0.17, "success_once_rate", "Any official success"),
        (0.17, "success_at_end_rate", "Final official success"),
    ]:
        bars = axes[0].bar(
            positions + offset,
            [summary[name]["official"][field] * 100 for name in NAMES],
            0.34,
            label=label,
        )
        axes[0].bar_label(bars, fmt="%.0f%%", padding=3)
    axes[0].set(
        xticks=positions,
        xticklabels=LABELS,
        ylabel="Success (%)",
        ylim=(0, 110),
        title="All 10 fresh shared validation resets",
    )
    axes[0].legend(fontsize=8)
    thresholds = ["10cm", "5cm", "2_5cm"]
    for index, (name, label, color) in enumerate(zip(NAMES, LABELS, COLORS)):
        bars = axes[1].bar(
            positions + (index - 1) * 0.25,
            [
                summary[name]["stages"][t]["conditional_occupancy"] * 100
                for t in thresholds
            ],
            0.25,
            label=label,
            color=color,
        )
        axes[1].bar_label(bars, fmt="%.0f", padding=2, fontsize=8)
        distances = np.asarray(
            [[r["cube_goal_distance"] for r in e["records"]] for e in episodes[name]]
        )
        steps = np.arange(1, 51)
        median = np.median(distances, axis=0)
        q1, q3 = np.quantile(distances, [0.25, 0.75], axis=0)
        axes[2].plot(steps, median, label=label, color=color)
        axes[2].fill_between(steps, q1, q3, color=color, alpha=0.12)
        grasped = [r for e in episodes[name] for r in e["records"] if r["is_grasped"]]
        axes[3].scatter(
            [r["cube_goal_distance"] for r in grasped],
            [r["arm_command_l2"] for r in grasped],
            s=9,
            alpha=0.25,
            label=label,
            color=color,
        )
    axes[1].set(
        xticks=positions,
        xticklabels=["≤10 cm", "≤5 cm", "≤2.5 cm"],
        ylim=(0, 110),
        ylabel="Pooled grasp-conditioned occupancy (%)",
        title="Denominator: all grasped steps per policy",
    )
    axes[1].legend(fontsize=8)
    axes[2].axhline(0.025, color="gray", linestyle=":", label="Placement threshold")
    axes[2].set(
        xlabel="Control step",
        ylabel="Cube-goal distance (m)",
        title="All 10 episodes: median and interquartile band",
    )
    axes[2].legend(fontsize=8)
    axes[3].axvline(0.025, color="gray", linestyle=":")
    axes[3].set(
        xlabel="Cube-goal distance while grasped (m)",
        ylabel="Executed seven-arm command L2",
        title="Synchronized observations; no causal claim",
    )
    axes[3].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "placement.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    primaryrows = ""
    detailrows = ""
    for name, label in zip(NAMES, LABELS):
        s = summary[name]
        stage = s["stages"]["2_5cm"]
        primaryrows += f"<tr><td>{label}</td><td>{s['official']['successes']}/10</td><td>{round(s['official']['success_at_end_rate'] * 10)}/10</td><td>{s['grasped_steps']}/500</td><td>{stage['grasped_steps_within']}/{s['grasped_steps']}</td></tr>"
        hits = stage["episode_first_hits"]
        remaining = ", ".join(str(hit["remaining_steps"]) for hit in hits) or "none"
        cmd = (
            "—"
            if stage["mean_arm_command_l2"] is None
            else f"{stage['mean_arm_command_l2']:.3f}"
        )
        speed = (
            "—"
            if stage["mean_max_arm_speed"] is None
            else f"{stage['mean_max_arm_speed']:.3f}"
        )
        detailrows += f"<tr><td>{label}</td><td>{cmd}</td><td>{speed}</td><td>{remaining}</td></tr>"
    subset = summary["bc_final_success_subset"]
    if subset["data"]:
        s = subset["data"]
        stage = s["stages"]["2_5cm"]
        subsettext = f"{s['episodes']} BC episodes ended successfully. Within that selected subset, {stage['grasped_steps_within']}/{s['grasped_steps']} grasped steps were within 2.5 cm; mean arm-command L2 was {stage['mean_arm_command_l2']:.3f} and mean maximum arm speed was {stage['mean_max_arm_speed']:.3f} rad/s. Selected seeds: {subset['selected_seeds']}."
    else:
        subsettext = (
            "No BC episode ended successfully, so the conditional subset is empty."
        )
    hybridrows = "".join(
        f"<tr><td>{label}</td><td>{hybrid[name]['activated_episodes']}/10</td><td>{hybrid[name]['official']['successes']}/10</td><td>{round(hybrid[name]['official']['success_at_end_rate'] * 10)}/10</td></tr>"
        for name, label in [
            ("raw_final", "Raw PPO + observation gate"),
            ("normalized_final", "Normalized PPO + observation gate"),
        ]
    )
    videos = ""
    for seed in protocol["video_seeds"]["placement"]:
        videos += f'<section class="videos"><h3>Shared validation seed {seed}</h3>'
        for name, label in zip(NAMES, LABELS):
            episode = next(e for e in episodes[name] if e["seed"] == seed)
            videos += f'<figure><video controls muted loop preload="metadata" src="placement_{name}_seed{seed}.mp4"></video><figcaption>{label}: any success {episode["success_once"]}; final {episode["success_at_end"]}</figcaption></figure>'
        videos += "</section>"
    reused = ""
    for name, label in [
        ("raw_final", "Raw hybrid"),
        ("normalized_final", "Normalized hybrid"),
    ]:
        reused += f'<section class="two"><h3>{label}: exact-trajectory verified native video reuse</h3>'
        for seed in protocol["video_seeds"]["observation_stop"]:
            reused += f'<figure><video controls muted loop preload="metadata" src="hybrid_{name}_seed{seed}.mp4"></video><figcaption>Seed {seed}; prior privileged native video reused after complete hybrid trajectory equality</figcaption></figure>'
        reused += "</section>"
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PickCube P0: observations and placement</title><style>
body{{font:16px/1.6 system-ui;background:#f2f5f3;color:#19312c;margin:0}}main{{max-width:1200px;margin:auto;padding:34px 22px}}h1{{font-size:39px;line-height:1.15}}h2{{margin-top:40px}}.notice{{background:#fff1cf;padding:20px;border-left:5px solid #b48c2a}}.card,.videos,.two{{background:white;padding:20px;margin:24px 0;border-radius:14px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #dbe3df}}.scroll{{overflow:auto}}img,video{{width:100%}}.videos{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}}.two{{display:grid;grid-template-columns:repeat(2,1fr);gap:18px}}.videos h3,.two h3{{grid-column:1/-1;margin:0}}figure{{margin:0}}figcaption{{font-size:13px}}a{{color:#176956}}@media(max-width:750px){{.videos,.two{{grid-template-columns:1fr}}main{{padding:20px 12px}}h1{{font-size:29px}}}}
</style><main><p>Stage 1 · P0 diagnostics · no training or final tests</p><h1>Observe placement, compare the approach, inspect the commands</h1>
<div class="notice">The observation-only stop is a hand-coded hybrid, not PPO alone. The BC comparison is a ten-reset validation diagnostic using an existing expert-trained clone and existing PPO checkpoints; expert data and training compute were not matched. No checkpoint was selected here.</div>
<h2>Ordinary policies on the same fresh resets</h2><p>All 10 seeds 130000–130009, all 50 steps, unchanged task and actions. Grasp-conditioned occupancy counts post-action states that are both grasped and within the stated distance. Correlated control steps are not independent trials.</p>
<div class="card scroll"><table><tr><th>Ordinary policy</th><th>Any official success</th><th>Final official success</th><th>Grasped steps/all steps</th><th>Within 2.5 cm/grasped steps</th></tr>{primaryrows}</table></div>
<img src="placement.png" alt="Validation success, grasp-conditioned goal occupancy, distance curves and executed command magnitudes">
<h2>Near-goal motion and time available</h2><p>These conditional measurements include only grasped states within 2.5 cm. The command is the executed seven-arm increment vector, not a Gaussian mean. Remaining steps are measured at each episode's first grasped 2.5 cm entry; episodes that never enter have no value.</p><div class="scroll"><table><tr><th>Policy</th><th>Mean arm-command L2</th><th>Mean maximum arm speed (rad/s)</th><th>Remaining steps at first entry</th></tr>{detailrows}</table></div>
<div class="card"><h3>BC final-success subset: conditional inspection only</h3><p>{subsettext}</p><p>This subset is selected by the observed final outcome and is not a performance comparison against unfiltered PPO. The all 10-episode table above remains primary.</p></div>
<h2>Native videos from the first two shared placement seeds</h2><p>Fixed before running, irrespective of success. Every video rerun matched its entire compact episode trace exactly; hashes are in the manifest.</p>{videos}
<h2>Can ordinary state observations reproduce the stop gate?</h2><p>Yes on all 1,020 audited reset/post-step states. The pure gate receives only raw 42 state observations. Installed schema and numeric checks identify the cube-to-goal vector at 39:42. Its float32 norm at or below 0.025 m latches zero seven-arm increments while preserving the current policy gripper. Privileged placement was audit-only and never determined a command. All 20 complete hybrid episode dictionaries matched the prior privileged override exactly.</p>
<div class="card scroll"><table><tr><th>Hand-coded hybrid on 80000–80009</th><th>Activated</th><th>Any official success</th><th>Final official success</th></tr>{hybridrows}</table></div><p>These reproduce the earlier privileged intervention outcomes; they are not newly learned PPO successes. Zero delta retargets current measured joints every step, rather than maintaining one fixed activation pose. The first two fixed seeds did not activate for either policy, so these native videos show identical behavior and are reused only after full trajectory verification.</p>{reused}
<h2>Evidence</h2><p><a href="protocol.json">Predeclared protocol</a> · <a href="placement_summary.json">All placement summaries</a> · <a href="observation_audit.json">Observation audit</a> · <a href="hybrid_comparison.json">Hybrid outcomes</a> · <a href="gate_checks_raw_final.jsonl">Raw gate vectors</a> · <a href="gate_checks_normalized_final.jsonl">Normalized gate vectors</a> · <a href="video_manifest.json">Video and episode hashes</a></p><p>Source/checkpoint hashes and versions were recorded before the final instrumented run. CPU Python/NumPy/Torch RNG streams were preserved around loading and evaluation. Frozen observation statistics remained unchanged. Only the listed validation seeds were used; final test seeds were untouched.</p></main></html>"""
    (output / "index.html").write_text(html)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report(args.run, args.output)


if __name__ == "__main__":
    main()
