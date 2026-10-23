"""Compact evidence and native plots for the short paired critic-bound ablation.

Reads completed artifacts only; never loads a policy or starts a simulator.
"""

import argparse
import hashlib
import html
import json
import math
import shutil
import statistics
from pathlib import Path

from .experiment import save_json
from .metrics import failure_counts, summarize

ARMS = ["raw", "bounded"]
NAMES = ["teacher", "raw_final", "bounded_final", "raw_best", "bounded_best"]
LABELS = {
    "teacher": "BC teacher",
    "raw_final": "Raw critic · final",
    "bounded_final": "Bounded critic · final",
    "raw_best": "Raw selected · secondary",
    "bounded_best": "Bounded selected · secondary",
}
PRIMARY = NAMES[:3]
COLORS = {"raw": "#ba5736", "bounded": "#187f83"}


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonl(path, episodes):
    """Keep real trajectories when present; omit empty repeated arrays only."""
    Path(path).write_text(
        "".join(
            json.dumps(
                {k: v for k, v in e.items() if k != "trajectory" or v},
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for e in episodes
        )
    )


def checked_geometry(result, seeds):
    """Independently recompute final task geometry from saved positions/qvel."""
    episodes = result["episodes"]
    if [e["seed"] for e in episodes] != list(seeds):
        raise ValueError("Episode seeds/order differ from the declared protocol")
    if summarize(episodes) != result["summary"]:
        raise ValueError("Saved episode summary differs from actual episodes")
    distances, speeds, independent_success = [], [], []
    for e in episodes:
        if e["steps"] != 50:
            raise ValueError("Report requires the unchanged full 50-step horizon")
        if len(e["final_arm_qvel"]) != 7:
            raise ValueError("Need exactly seven arm velocities")
        if len(e["final_cube_position"]) != 3 or len(e["final_goal_position"]) != 3:
            raise ValueError("Need actual three-dimensional cube and goal positions")
        distance = math.dist(e["final_cube_position"], e["final_goal_position"])
        speed = max(map(abs, e["final_arm_qvel"]))
        if not math.isfinite(distance) or not math.isfinite(speed):
            raise ValueError("Nonfinite final geometry")
        if not math.isclose(distance, e["final_goal_distance"], abs_tol=1e-6):
            raise ValueError("Saved distance differs from independent 3D geometry")
        placed, static = distance <= 0.025, speed <= 0.2
        if (
            placed != e["final_is_obj_placed"]
            or static != e["final_is_robot_static"]
            or (placed and static) != e["success_at_end"]
        ):
            raise ValueError("Official final outcome differs from independent geometry")
        distances.append(distance)
        speeds.append(speed)
        independent_success.append(placed and static)
    return {
        "episodes": len(episodes),
        "final_successes": sum(independent_success),
        "any_successes": sum(e["success_once"] for e in episodes),
        "median_goal_distance_m": statistics.median(distances),
        "median_max_arm_speed_rad_s": statistics.median(speeds),
        "final_placed_count": sum(d <= 0.025 for d in distances),
        "final_static_count": sum(s <= 0.2 for s in speeds),
        "final_distance_m": distances,
        "final_max_arm_speed_rad_s": speeds,
        "independent_final_success": independent_success,
        "failure_categories": failure_counts(episodes),
    }


def gradient_series(points, field, group):
    """Missing first-step data stays NaN (a plot gap), never a measured zero."""
    values = []
    for p in points:
        if p["optimizer_steps"] == 0:
            values.append(math.nan)
        else:
            value = p[field][group]
            if not math.isfinite(value) or value < 0:
                raise ValueError("Invalid measured first-step gradient/update norm")
            values.append(value)
    return values


def plot_diagnostics(diagnostics, output):
    """Public plotting helper also accepts incomplete live telemetry for schema QA."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fields = [
        ("values", "Collected rollout critic values"),
        ("returns", "GAE lambda-return targets"),
        ("terminal", "Pre-reset timeout critic values"),
        ("buffer_rewards_after_bootstrap", "Rewards after timeout correction"),
    ]
    for arm in ARMS:
        points = diagnostics[arm]["updates"]
        x = np.asarray([p["timesteps"] / 1000 for p in points])
        for ax, (field, title) in zip(axes.flat, fields, strict=True):
            rows = [
                p["raw"]["terminal_values"] if field == "terminal" else p[field]
                for p in points
            ]
            for key, style, alpha in [("median", "-", 1), ("max", "--", 0.7)]:
                y = [r[key] if r else math.nan for r in rows]
                ax.plot(
                    x, y, style, color=COLORS[arm], alpha=alpha, label=f"{arm}: {key}"
                )
            ax.plot(
                x,
                [r["min"] if r else math.nan for r in rows],
                ":",
                color=COLORS[arm],
                alpha=0.6,
                label=f"{arm}: min",
            )
            ax.set(
                title=title,
                xlabel="Training transitions (thousands)",
                ylabel="Value / reward units",
            )
            ax.set_yscale("symlog", linthresh=1)
        for ax in axes.flat:
            ax.axhline(100, color="#64748b", linewidth=0.8, alpha=0.5)
            ax.axhline(-100, color="#64748b", linewidth=0.8, alpha=0.5)
    for ax in axes.flat:
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Actual rollout telemetry · gray lines ±100 · symlog scale")
    fig.tight_layout()
    fig.savefig(output / "bootstrap.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for arm in ARMS:
        points = diagnostics[arm]["updates"]
        x = [p["timesteps"] / 1000 for p in points]
        for group, style in [("actor", "-"), ("critic", "--")]:
            for ax, field in zip(
                axes,
                [
                    "gradient_norm_before_clip",
                    "gradient_norm_after_clip",
                    "first_step_parameter_delta_l2",
                ],
                strict=True,
            ):
                ax.plot(
                    x,
                    gradient_series(points, field, group),
                    style,
                    color=COLORS[arm],
                    label=f"{arm}: {group}",
                )
    for ax, title in zip(
        axes,
        [
            "First-step gradients before clipping",
            "First-step gradients after clipping",
            "Measured first Adam parameter delta",
        ],
        strict=True,
    ):
        ax.set(title=title, xlabel="Training transitions (thousands)", ylabel="L2 norm")
        ax.set_yscale("symlog", linthresh=1e-7)
        ax.legend(fontsize=7)
    fig.suptitle("First executed optimizer step per rollout only · gaps mean no step")
    fig.tight_layout()
    fig.savefig(output / "gradients.png", dpi=160)
    plt.close(fig)


def report(run, audit, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    run, audit, output = Path(run), Path(audit), Path(output)
    if run.resolve() == output.resolve():
        raise ValueError("Report output must be separate from the run")
    output.mkdir(parents=True, exist_ok=True)
    protocol = read(run / "protocol.json")
    if (
        protocol["config"]["total_timesteps"] != 262144
        or protocol["validation_seeds"]
        != list(
            range(
                protocol["config"]["validation_seed_start"],
                protocol["config"]["validation_seed_start"]
                + protocol["config"]["validation_episodes"],
            )
        )
        or protocol["test_seeds"]
        != list(
            range(
                protocol["config"]["test_seed_start"],
                protocol["config"]["test_seed_start"]
                + protocol["config"]["test_episodes"],
            )
        )
        or protocol["video_seeds"] != protocol["test_seeds"][:2]
    ):
        raise ValueError("Unexpected critic ablation protocol")
    required = [
        "protocol.json",
        "preflight.json",
        "stochastic_smoke.json",
        "comparison.json",
        "provenance.json",
        "verified.json",
    ]
    for name in required:
        shutil.copy2(run / name, output / name)
    shutil.copy2(audit, output / "audit.json")
    archive = run / "launch_sources"
    for name, expected in protocol["sources"].items():
        if sha256(archive / name) != expected:
            raise ValueError(f"Launch source archive mismatch: {name}")
    shutil.copytree(archive, output / "launch_sources", dirs_exist_ok=True)
    diagnostics, validation, totals = {}, {}, {}
    for arm in ARMS:
        for name in [
            "initialization",
            "training",
            "validation",
            "selected",
            "diagnostics",
            "training_metrics",
        ]:
            shutil.copy2(run / arm / f"{name}.json", output / f"{arm}_{name}.json")
        training = read(run / arm / "training.json")
        if not training["complete"] or training["actual_timesteps"] != 262144:
            raise ValueError("Incomplete or incorrect training budget")
        diagnostics[arm] = read(run / arm / "diagnostics.json")
        points = diagnostics[arm]["updates"]
        if not points or points[-1]["timesteps"] != 262144:
            raise ValueError("Missing final optimizer diagnostic")
        validation[arm] = read(run / arm / "validation.json")
        windows = read(run / arm / "training_metrics.json")["windows"]
        totals[arm] = {
            key: sum(w.get(key, 0) for w in windows)
            for key in set().union(*(set(w) - {"timesteps"} for w in windows))
        }
        if totals[arm]["transitions"] != 262144:
            raise ValueError("Stochastic counts differ from the transition budget")
    save_json(output / "training_totals.json", totals)
    preflight = {}
    for name in ["teacher", "raw", "bounded"]:
        result = read(run / f"preflight_{name}.json")
        checked_geometry(result, protocol["validation_seeds"])
        preflight[name] = result["episodes"]
        jsonl(output / f"preflight_{name}.jsonl", result["episodes"])
    if (
        preflight["teacher"] != preflight["raw"]
        or preflight["teacher"] != preflight["bounded"]
    ):
        raise ValueError("Initial policies do not exactly reproduce teacher episodes")
    results, geometry = {}, {}
    for name in NAMES:
        results[name] = read(run / f"test_{name}.json")
        geometry[name] = checked_geometry(results[name], protocol["test_seeds"])
        jsonl(output / f"test_{name}.jsonl", results[name]["episodes"])
    if read(run / "comparison.json") != {n: results[n]["summary"] for n in NAMES}:
        raise ValueError("Comparison summary differs from test evidence")
    save_json(output / "geometry.json", geometry)
    manifest = []
    for name in PRIMARY:
        video = read(run / f"video_{name}.json")
        checked_geometry(video, protocol["video_seeds"])
        jsonl(output / f"video_{name}.jsonl", video["episodes"])
        for episode in video["episodes"]:
            expected = next(
                e for e in results[name]["episodes"] if e["seed"] == episode["seed"]
            )
            if {k: v for k, v in episode.items() if k != "trajectory"} != {
                k: v for k, v in expected.items() if k != "trajectory"
            }:
                raise ValueError("Video episode differs from paired test episode")
            filename = f"video_{name}_seed{episode['seed']}.mp4"
            shutil.copy2(run / filename, output / filename)
            manifest.append(
                {
                    "policy": name,
                    "seed": episode["seed"],
                    "file": filename,
                    "sha256": sha256(output / filename),
                    "all_nontrajectory_fields_match": True,
                }
            )
    if len(manifest) != 6:
        raise ValueError("Need all six predeclared native videos")
    save_json(output / "video_manifest.json", manifest)
    plot_diagnostics(diagnostics, output)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for arm in ARMS:
        points = validation[arm]
        axes[0].plot(
            [p["timesteps"] / 1000 for p in points],
            [100 * p["success_at_end_rate"] for p in points],
            "o-",
            color=COLORS[arm],
            label=f"{arm} critic",
        )
    teacher_rate = read(run / "preflight.json")["teacher_summary"][
        "success_at_end_rate"
    ]
    axes[0].axhline(
        100 * teacher_rate, linestyle=":", color="#64748b", label="BC teacher"
    )
    axes[0].set(
        title="Deterministic validation · 10 fixed resets",
        xlabel="Training transitions (thousands)",
        ylabel="Final success (%)",
        ylim=(-2, 102),
    )
    axes[0].legend(fontsize=8)
    bars = axes[1].bar(
        np.arange(5),
        [100 * results[n]["summary"]["success_at_end_rate"] for n in NAMES],
        color=["#64748b", COLORS["raw"], COLORS["bounded"], "#dfb3a4", "#8cbfc1"],
    )
    axes[1].bar_label(bars, fmt="%.0f%%", padding=3)
    axes[1].set(
        title="Fresh paired tests · 50 resets",
        ylabel="Final success (%)",
        ylim=(0, 112),
        xticks=np.arange(5),
        xticklabels=[
            "BC",
            "Raw\nfinal",
            "Bounded\nfinal",
            "Raw\nselected\nsecondary",
            "Bounded\nselected\nsecondary",
        ],
    )
    fig.tight_layout()
    fig.savefig(output / "outcomes.png", dpi=160)
    plt.close(fig)

    rows = ""
    for name in NAMES:
        g, s = geometry[name], results[name]["summary"]
        rows += f"<tr><th>{LABELS[name]}</th><td>{g['final_successes']}/50</td><td>{g['any_successes']}/50</td><td>{s['return_mean']:.2f} ± {s['return_std']:.2f}</td><td>{100 * g['median_goal_distance_m']:.2f} cm</td><td>{g['median_max_arm_speed_rad_s']:.3f} rad/s</td></tr>"
    selected = "; ".join(
        f"{a}: {read(run / a / 'selected.json')['timesteps']:,} transitions"
        for a in ARMS
    )
    training_rows = ""
    for arm in ARMS:
        t = totals[arm]
        training_rows += f"<tr><th>{arm}</th><td>{t['transitions']:,}</td><td>{t['grasped']:,}</td><td>{t['placed']:,}</td><td>{t['success']:,}</td><td>{t.get('episodes_final_success', 0):,}/{t.get('episodes', 0):,}</td></tr>"
    videos = ""
    for seed in protocol["video_seeds"]:
        videos += f'<section class="videos"><h3>Predeclared seed {seed}</h3>'
        for name in PRIMARY:
            videos += f'<figure><video controls preload="metadata" src="video_{name}_seed{seed}.mp4"></video><figcaption>{LABELS[name]}</figcaption></figure>'
        videos += "</section>"
    links = [
        "protocol.json",
        "preflight.json",
        "provenance.json",
        "verified.json",
        "audit.json",
        "geometry.json",
        "comparison.json",
        "video_manifest.json",
        "training_totals.json",
        "raw_diagnostics.json",
        "bounded_diagnostics.json",
        "raw_training_metrics.json",
        "bounded_training_metrics.json",
        "raw_initialization.json",
        "bounded_initialization.json",
    ]
    evidence_links = " · ".join(
        f'<a href="{html.escape(n)}">{html.escape(n)}</a>' for n in links
    )
    raw_final, bound_final, teacher_final = (
        geometry[n]["final_successes"]
        for n in ["raw_final", "bounded_final", "teacher"]
    )
    if protocol.get("worker_reset_seeding"):
        reset_note = (
            "<strong>Reset-stream contract:</strong> "
            + html.escape(protocol["worker_reset_seeding"])
            + ". This newly declared condition uses fresh seed partitions; historical unseeded-worker results remain separate. Matching streams under fixed actions is regression-tested, not a claim that trained policies follow identical trajectories."
        )
    else:
        reset_note = "<strong>Reset-stream caveat:</strong> initial worker resets are seeded, but later automatic resets draw on worker Torch RNG that this historical factory did not seed. Parent fingerprints do not cover worker states; first-rollout summaries differ. These are matched initialization and recipe comparisons, not identical training trajectories. The subsequent worker-seeding repair is for future runs and does not relabel this evidence."
    (
        output / "index.html"
    ).write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PickCube critic stability diagnostic</title><style>
body{{font:16px/1.6 system-ui;margin:0;background:#f2f6f7;color:#193744}}main{{max-width:1240px;margin:auto;padding:30px 20px}}h1{{font-size:38px;line-height:1.2}}h2{{margin-top:38px}}.notice{{background:#fff0ce;border-left:5px solid #a3751f;padding:18px}}table{{border-collapse:collapse;background:white;width:100%}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #d5dee1}}.scroll{{overflow:auto}}img,video{{width:100%;background:white}}.videos{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;padding:20px;background:white;margin:24px 0}}.videos h3{{grid-column:1/-1;margin:0}}figure{{margin:0}}a{{color:#176b80}}@media(max-width:750px){{.videos{{grid-template-columns:1fr}}h1{{font-size:29px}}main{{padding:15px}}}}
</style><main><p>Stage 1 · paired critic-output ablation · original task</p><h1>Does bounding the critic stabilize BC-initialized PPO?</h1>
<p><strong>Fixed-budget final successes: bounded {bound_final}/50; raw {raw_final}/50; BC teacher {teacher_final}/50.</strong> The final models receive the same 262,144 transitions. Validation-selected models below are secondary.</p>
<div class="notice">One training seed (42) per arm and a short diagnostic budget. These results cannot establish million-transition stability or performance across training seeds. Fifty paired test resets evaluate these particular policies. No stopping override, hybrid recovery, reward change or timeout-semantics change is used.</div>
<p>Only the critic output changes: V = 100 tanh(raw V / 100). Both arms start with the same copied BC actor, canonical critic tensors, fixed observation normalization, parent RNG digest and empty Adam state. Initial teacher/raw/bounded full validation episodes match exactly. Initial value outputs need not match: the output transform is the treatment. All runs preserve the 50-step horizon and final success requires actual 3D cube–goal distance ≤ 2.5 cm and all seven arm speeds ≤ 0.2 rad/s.</p>
<div class="scroll"><table><tr><th>Policy</th><th>Final success</th><th>Any-step success</th><th>Return: mean ± SD</th><th>Median final error</th><th>Median max arm speed</th></tr>{rows}</table></div><img src="outcomes.png" alt="Deterministic validation final success and five-condition paired test success">
<h2>What the critic sees and bootstraps</h2><p>These are actual collected rollout values, GAE lambda-return targets, pre-reset terminal critic values, and rewards after timeout correction. Lines show medians, maxima and minima. Gray reference lines mark ±100; axes use a symmetric logarithmic scale. The installed normalized task reward is in [0,1], so the true continuing discounted value with gamma 0.99 is in [0,100]. The candidate permits values in [−100,100]; it clips neither rewards nor targets. Negative or large timeout-corrected rewards are not raw environment rewards.</p><img src="bootstrap.png" alt="Actual rollout values, lambda returns, timeout values and bootstrap-adjusted rewards against ±100 reference">
<h2>Actual gradients and Adam steps</h2><p>Each point measures only the first executed optimizer step of that rollout, rather than all minibatches or the whole update. Hooks observe gradients before shared global clipping, gradients at Adam entry, and actual parameter deltas after that Adam step. Actor and critic curves are separate. Gaps mean no optimizer step. A smaller clipped gradient alone does not prove a smaller Adam update; momentum and adaptation matter.</p><img src="gradients.png" alt="Measured first-step actor and critic gradients before and after clipping and actual Adam parameter deltas">
<h2>Stochastic training counts</h2><p>These counts exclude validation and tests. Steps are correlated; unfinished episodes at the final rollout boundary are excluded from completed-episode counts. A placed or successful state alone does not establish a learned pick.</p><div class="scroll"><table><tr><th>Arm</th><th>Transitions</th><th>Grasped steps</th><th>Placed steps</th><th>Successful steps</th><th>Final-success episodes</th></tr>{training_rows}</table></div>
<h2>Native paired rollouts</h2><p>The first two predeclared test seeds are shown regardless of outcome. All stored non-trajectory episode fields match the corresponding full test episodes, and all six video files have saved SHA-256 hashes.</p>{videos}
<h2>Selection and limits</h2><p>{reset_note}</p><p>Selected checkpoints: {selected}. Ranking uses final success, then any-step success, then return. Selection is secondary and uses ten fixed validation resets. Validation executes before the optimizer update at that boundary; fixed-budget final models include the last update. Step-zero selections represent the initial BC actor, not PPO improvement.</p>
<p>The historical audit uses 408 fixed off-policy demonstration states to inspect earlier checkpoints. It establishes critic miscalibration on those probes; it does not recover historical on-policy gradients or prove the cause of instability. Saved Adam buffers are momentum statistics. This experiment adds actual on-policy telemetry to test a single stabilization hypothesis.</p>
<h2>Evidence</h2><p>{evidence_links}</p><p>Episode evidence is exported as compact JSONL for all three preflight policies, all five test conditions and the native video episodes. Archived launch sources are checksum-verified against the protocol.</p></main></html>""")
    return geometry, totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.run, args.audit, args.output)


if __name__ == "__main__":
    main()
