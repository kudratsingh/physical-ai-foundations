"""Publish the privileged stopping diagnostic without labeling it repaired PPO."""

import argparse
import json
from pathlib import Path
import shutil

from .experiment import save_json

NAMES = [
    f"{name}_{mode}"
    for name in ["raw_final", "normalized_final"]
    for mode in ["normal", "override"]
]


def report(run, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for filename in (
        ["protocol.json", "baseline_equality.json", "comparison.json"]
        + [f"{name}.json" for name in NAMES]
        + [f"video_{name}.json" for name in NAMES]
    ):
        shutil.copy2(run / filename, output / filename)
    for filename in run.glob("*.mp4"):
        shutil.copy2(filename, output / filename.name)
    results = {name: json.loads((run / f"{name}.json").read_text()) for name in NAMES}
    protocol = json.loads((run / "protocol.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    chosen = {}
    for row, policy in enumerate(["raw_final", "normalized_final"]):
        ordinary = results[f"{policy}_normal"]["episodes"]
        override = results[f"{policy}_override"]["episodes"]
        selected = next(
            (episode for episode in override if episode["active_steps"]), override[0]
        )
        seed = selected["seed"]
        chosen[policy] = seed
        baseline = next(episode for episode in ordinary if episode["seed"] == seed)
        for episode, label, style in [
            (baseline, "Normal policy", "-"),
            (selected, "Privileged zero-arm override", "--"),
        ]:
            x = [0] + [record["step"] for record in episode["records"]]
            for column, field in enumerate(["cube_goal_distance", "max_arm_qvel"]):
                axes[row, column].plot(
                    x,
                    [episode["initial"][field]]
                    + [r[field] for r in episode["records"]],
                    style,
                    label=label,
                )
        for column, threshold in enumerate([0.025, 0.2]):
            axis = axes[row, column]
            axis.axhline(
                threshold, color="gray", linestyle=":", label="Official threshold"
            )
            if selected["first_override_action_step"] is not None:
                axis.axvline(
                    selected["first_override_action_step"],
                    color="#8750ad",
                    alpha=0.7,
                    linestyle=":",
                    label="First overridden action",
                )
            axis.set(
                xlabel="Control step",
                ylabel="Distance (m)" if column == 0 else "Max arm speed (rad/s)",
                title=f"{policy.replace('_', ' ')} · seed {seed}",
            )
            axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "stopping.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    save_json(
        output / "plot_examples.json",
        dict(
            selection="First naturally activated validation seed per policy, or first seed if none; illustrative traces, not performance subset",
            seeds=chosen,
        ),
    )
    rows = ""
    for name in NAMES:
        data = results[name]
        any_count = data["summary"]["successes"]
        final_count = sum(e["success_at_end"] for e in data["episodes"])
        placement_count = sum(
            e["first_placement_step"] is not None for e in data["episodes"]
        )
        activation = data["activation"]
        post_ratio = (
            f"{activation['post_activation_any_success']}/{activation['episodes']}"
            if activation["episodes"]
            else "—"
        )
        rows += f"<tr><td>{name.replace('_', ' ')}</td><td>{placement_count}/10</td><td>{any_count}/10</td><td>{final_count}/10</td><td>{activation['episodes']}/10</td><td>{post_ratio}</td></tr>"
    video_sections = ""
    for policy in ["raw_final", "normalized_final"]:
        for seed in protocol["video_seeds"]:
            video_sections += f'<section class="pair"><h3>{policy.replace("_", " ")} · seed {seed}</h3>'
            for mode in ["normal", "override"]:
                name = f"{policy}_{mode}"
                episode = next(
                    e for e in results[name]["episodes"] if e["seed"] == seed
                )
                video_sections += f'<figure><video controls muted loop preload="metadata" src="video_{name}_seed{seed}.mp4"></video><figcaption>{mode}: {episode["active_steps"]} overridden steps; official final success {episode["success_at_end"]}</figcaption></figure>'
            video_sections += "</section>"
    overrides = [
        results[f"{name}_override"] for name in ["raw_final", "normalized_final"]
    ]
    conclusion = (
        "; ".join(
            f"{name}: {data['activation']['episodes']}/10 episodes activated; {data['summary']['successes']}/10 official successes at any time under intervention"
            for name, data in zip(["Raw PPO", "Normalized PPO"], overrides)
        )
        + "."
    )
    activation_sections = ""
    examples_path = output / "activation_examples.json"
    if examples_path.exists():
        examples = json.loads(examples_path.read_text())
        activation_sections = "<h2>Illustrative activation cases</h2><p>These additional examples were chosen after the aggregate diagnostic: the first naturally activated validation seed per policy, matching the trace plots. They are selected examples, not a performance sample. All recorded fields match the original episodes; aggregate counts above are unchanged.</p>"
        for policy in ["raw_final", "normalized_final"]:
            activation_sections += (
                f'<section class="pair"><h3>{policy}: first natural activation</h3>'
            )
            for video in examples["videos"]:
                if video["policy"] == policy:
                    activation_sections += f'<figure><video controls muted preload="metadata" src="{video["video"]}"></video><figcaption>seed {video["seed"]}, {video["mode"]}; {video["active_steps"]} overridden steps</figcaption></figure>'
            activation_sections += "</section>"
        activation_sections += '<p><a href="activation_examples.json">Example selection and video provenance</a></p>'
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PickCube stopping diagnostic</title><style>
body{{font:16px/1.6 system-ui;background:#f4f5f3;color:#20312d;margin:0}}main{{max-width:1150px;margin:auto;padding:30px 20px}}h1{{font-size:37px;line-height:1.15}}h2{{margin-top:40px}}.notice{{background:#fff0cf;border-left:5px solid #b67917;padding:20px}}.card,.pair{{background:white;border-radius:14px;padding:20px;margin:22px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #ddd}}.scroll{{overflow:auto}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.pair h3{{grid-column:1/-1;margin:0}}figure{{margin:0}}video,img{{width:100%}}a{{color:#126f59}}@media(max-width:700px){{.pair{{grid-template-columns:1fr}}h1{{font-size:29px}}main{{padding:18px 12px}}}}
</style><main><p>Stage 1 · validation-only diagnostic</p><h1>What happens if a privileged override stops commanding arm increments after placement?</h1>
<div class="notice"><b>This is not repaired PPO or a final test result.</b> The override reads privileged task placement and changes the original policy command. All numbers below are conditional diagnostic outcomes on the same ten validation seeds. No model was trained or selected here.</div>
<p>{conclusion}</p><div class="card"><h2>Predeclared comparison</h2><p>Raw final and observation-normalized final checkpoints, each normal and overridden, on seeds 80000–80009. Each episode runs all 50 steps. Placement at step t latches zero arm increments from action t+1 onward; the current policy still supplies the gripper command every step. No forced placement, goal-local reset, added damping or changed reward.</p>
<div class="scroll"><table><tr><th>Condition</th><th>Ever placed</th><th>Any official success</th><th>Final official success</th><th>Override activated</th><th>Success after activation / activated episodes</th></tr>{rows}</table></div><p>For normal conditions the override activation columns are zero by definition. Official success still requires distance ≤0.025 m <em>and</em> all seven arm speeds ≤0.2 rad/s. Reaching placement alone does not meet that criterion.</p></div>
<h2>Zero delta recomputes a measured-state target</h2><p>The installed Panda controller uses delta mode with <code>use_target=False</code>. Its normalized ±1 arm command maps to ±0.1 rad. Each step sets target = current measured joint position + increment. Sending zero therefore recomputes the measured pose each step; it does <b>not</b> retain a fixed pose from activation or guarantee that velocity vanishes. The log records the actual target and measured pre-step positions, and checks their equality on every active step.</p>
<img src="stopping.png" alt="Paired normal and privileged-override distance and arm-speed traces"><p>Trace examples use the first naturally activated validation seed per policy (or the first seed when none activate). The aggregate table includes all ten seeds. Trace selection does not depend on success.</p>
{activation_sections}
<h2>Predeclared native simulator videos</h2><p>The first two predeclared validation seeds for every condition, regardless of activation or success. If an override never activates, paired behavior is expected to be identical. Every video rerun reproduced its entire logged trajectory exactly.</p>{video_sections}
<h2>Limits and reproducibility</h2><p>The ordinary policies reproduced all 500 pre-clip action means from the earlier validation saturation audit exactly. Frozen normalization statistics did not change. The protocol was saved before rollouts with checkpoint, source and installed controller hashes. Final test seeds remained untouched. A few naturally activated episodes cannot establish robust stopping, and any intervention success belongs to the privileged diagnostic rather than autonomous PPO.</p><p><a href="protocol.json">Predeclared protocol</a> · <a href="comparison.json">Aggregate outcomes</a> · <a href="baseline_equality.json">Earlier baseline equality</a> · <a href="raw_final_override.json">Raw override step log</a> · <a href="normalized_final_override.json">Normalized override step log</a> · <a href="plot_examples.json">Trace selection</a></p></main></html>"""
    (output / "index.html").write_text(html)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report(args.run, args.output)


if __name__ == "__main__":
    main()
