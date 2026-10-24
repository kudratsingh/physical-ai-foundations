"""Native visual evidence for the predeclared lambda10-versus-lambda25 study."""

import argparse
import html
import math
import shutil
import statistics
from pathlib import Path

import numpy as np

from .critic_report import gradient_series
from .experiment import save_json
from .normalized_ppo import sha256
from .retention_compare import read, require, verify_video
from .retention_report import export_result
from .strength_compare import ARMS, COEFFICIENTS, SEEDS

COLORS = {"lambda10": "#bd683d", "lambda25": "#167f8d"}


def promising_criterion(geometry):
    rows = []
    for seed in SEEDS:
        counts = {
            f"{arm}_{tag}": geometry[f"seed{seed}_{arm}_{tag}"]["final_successes"]
            for arm in ARMS
            for tag in ["final", "best"]
        }
        rows.append(
            {
                "training_seed": seed,
                **counts,
                "lambda25_minus_lambda10_final": counts["lambda25_final"]
                - counts["lambda10_final"],
                "lambda10_selected_minus_final": counts["lambda10_best"]
                - counts["lambda10_final"],
                "lambda25_selected_minus_final": counts["lambda25_best"]
                - counts["lambda25_final"],
            }
        )
    target = next(r for r in rows if r["training_seed"] == 2042)
    other = [r for r in rows if r["training_seed"] != 2042]
    collapse = {
        arm: statistics.mean(r[f"{arm}_selected_minus_final"] for r in rows)
        for arm in ARMS
    }
    improved = target["lambda25_minus_lambda10_final"] >= 5
    preserved = all(r["lambda25_minus_lambda10_final"] >= -2 for r in other)
    stable = collapse["lambda25"] <= collapse["lambda10"]
    return {
        "rows": rows,
        "target_training_seed": 2042,
        "target_gain_at_least_five": improved,
        "each_other_seed_loses_at_most_two": preserved,
        "mean_signed_collapse_not_worse": stable,
        "mean_signed_selected_minus_final": collapse,
        "promising_rule_met": improved and preserved and stable,
        "rule": "lambda25 improves predeclared seed 2042 by at least5/50 versus lambda10, loses at most2/50 in EACH other training seed, and mean signed(selected-final)collapse is no larger.",
        "scope": "Descriptive fixed-strength rule, not significance. Same 50 fresh reset seeds across 3 training seeds; target 2042 was chosen from PR18 before this study.",
    }


def paired_outcomes(results):
    """Compare paired resets, without pooling across training seeds."""

    def compare(reference, candidate):
        left = {e["seed"]: e["success_at_end"] for e in results[reference]["episodes"]}
        right = {e["seed"]: e["success_at_end"] for e in results[candidate]["episodes"]}
        require(set(left) == set(right), "Paired evaluation resets differ")
        gained = sorted(s for s in left if right[s] and not left[s])
        lost = sorted(s for s in left if left[s] and not right[s])
        return {
            "reference": reference,
            "candidate": candidate,
            "new_successes": len(gained),
            "reference_successes_lost": len(lost),
            "new_success_reset_seeds": gained,
            "lost_success_reset_seeds": lost,
            "retained_successes": sum(left[s] and right[s] for s in left),
            "paired_resets": len(left),
        }

    rows = []
    for seed in SEEDS:
        row = {
            "training_seed": seed,
            "lambda25_vs_lambda10_final": compare(
                f"seed{seed}_lambda10_final", f"seed{seed}_lambda25_final"
            ),
        }
        for arm in ARMS:
            row[f"{arm}_vs_teacher_final"] = compare(
                "teacher", f"seed{seed}_{arm}_final"
            )
            row[f"{arm}_final_vs_selected"] = compare(
                f"seed{seed}_{arm}_best", f"seed{seed}_{arm}_final"
            )
        rows.append(row)
    return {
        "rows": rows,
        "scope": "Same fresh reset pairs within each comparison. Selected policies are secondary and validation-selected. Shared resets are not independent trials across training seeds.",
    }


def trajectory_cases(result):
    """All ever-successful episodes ending outside the official placement radius."""
    cases = []
    for e in sorted(result["episodes"], key=lambda e: e["seed"]):
        if e["success_once"] and e["final_goal_distance"] > 0.025:
            cases.append(
                {
                    "reset_seed": e["seed"],
                    "first_success_step": e["first_success_step"],
                    "final_distance_m": e["final_goal_distance"],
                    "final_static": e["final_is_robot_static"],
                    "final_max_arm_speed_rad_s": max(map(abs, e["final_arm_qvel"])),
                    "steps": [
                        r for r in result["step_records"] if r["seed"] == e["seed"]
                    ],
                }
            )
    return cases


def plot_trajectories(output, name, result):
    import matplotlib.pyplot as plt

    cases = trajectory_cases(result)
    files = []
    for page in range(max(1, math.ceil(len(cases) / 12))):
        subset = cases[page * 12 : (page + 1) * 12]
        rows = max(1, math.ceil(len(subset) / 3))
        fig, axes = plt.subplots(rows, 3, figsize=(13, 3.1 * rows), squeeze=False)
        for ax in axes.flat:
            ax.set_visible(False)
        if not subset:
            axes[0, 0].set_visible(True)
            axes[0, 0].text(
                0.5,
                0.5,
                "No qualifying cases",
                ha="center",
                va="center",
                transform=axes[0, 0].transAxes,
            )
            axes[0, 0].set_axis_off()
        for ax, case in zip(axes.flat, subset):
            ax.set_visible(True)
            records = case["steps"]
            x = [0, *[r["step"] for r in records]]
            distances = [
                records[0]["pre"]["cube_goal_distance"],
                *[r["post"]["cube_goal_distance"] for r in records],
            ]
            ax.plot(x, 100 * np.asarray(distances), color="#216b86")
            ax.axhline(2.5, color="#b04d37", linestyle=":", label="Placement 2.5cm")
            ax.axvline(
                case["first_success_step"],
                color="#4f8756",
                linestyle="--",
                label="First success",
            )
            ax.set(
                title=f"Reset {case['reset_seed']} · final static={case['final_static']}",
                xlabel="Control step",
                ylabel="Actual 3D cube–goal distance (cm)",
                ylim=(0, None),
            )
            ax.legend(fontsize=6)
        fig.suptitle(f"{name} · ALL successful-then-displaced cases · page {page + 1}")
        fig.tight_layout()
        filename = f"trajectories_{name}_{page + 1:02d}.png"
        fig.savefig(output / filename, dpi=140)
        plt.close(fig)
        files.append(filename)
    return {
        "condition": name,
        "selection": "ALL episodes that succeeded at least once and finish with actual 3D distance>0.025m; sorted reset seeds, no favorable-case selection",
        "cases": [{k: v for k, v in c.items() if k != "steps"} for c in cases],
        "plots": files,
    }


def plots(output, results, geometry, criterion, validation, objectives, diagnostics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for arm, offset in [("lambda10", -0.18), ("lambda25", 0.18)]:
        bars = axes[0].bar(
            x + offset,
            [geometry[f"seed{s}_{arm}_final"]["final_successes"] for s in SEEDS],
            width=0.34,
            color=COLORS[arm],
            label=arm,
        )
        axes[0].bar_label(bars, padding=3)
        axes[1].bar(
            x + offset,
            [r[f"{arm}_selected_minus_final"] for r in criterion["rows"]],
            width=0.34,
            color=COLORS[arm],
            label=arm,
        )
    axes[0].axhline(
        geometry["teacher"]["final_successes"],
        color="#64748b",
        linestyle=":",
        label="BC teacher on fresh resets",
    )
    axes[0].set(
        title="Primary: fixed-budget final",
        ylabel="Final successes / same 50 fresh resets",
        ylim=(0, 55),
    )
    axes[1].axhline(0, color="#64748b", linewidth=0.8)
    axes[1].set(
        title="Secondary: signed selection−final", ylabel="Success-count difference /50"
    )
    for ax in axes:
        ax.set_xticks(x, ["42", "1042", "2042 (target)"])
        ax.set_xlabel("Training seed")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "outcomes.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, seed in zip(axes, SEEDS, strict=True):
        for arm in ARMS:
            rows = validation[str(seed)][arm]
            ax.plot(
                [r["timesteps"] / 1000 for r in rows],
                [100 * r["success_at_end_rate"] for r in rows],
                "o-",
                color=COLORS[arm],
                label=arm,
            )
        ax.set(
            title=f"Training seed {seed}",
            xlabel="Transitions (thousands)",
            ylabel="Final success /40 validation resets (%)",
            ylim=(-2, 102),
        )
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "validation.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fields = [
        (
            "post_first_success_arm_command_l2_mean",
            "Mean arm command L2",
            "Normalized action L2",
        ),
        (
            "post_first_success_max_arm_speed_mean",
            "Mean post-state max |arm qvel|",
            "rad/s",
        ),
        (
            "post_first_success_teacher_arm_drift_l2_mean",
            "Mean deviation from BC on current state",
            "Normalized arm-action L2",
        ),
        (
            "post_first_success_success_occupancy",
            "Success occupancy after first success",
            "Fraction of conditioned steps",
        ),
        (
            "success_to_failure_transitions",
            "Success→failure transitions",
            "Events across50episodes",
        ),
        (
            "post_first_success_gripper_command_abs_mean",
            "Mean absolute gripper command",
            "Normalized absolute command",
        ),
    ]
    for ax, (field, title, ylabel) in zip(axes.flat, fields, strict=True):
        for arm in ARMS:
            vals = [results[f"seed{s}_{arm}_final"]["retention"][field] for s in SEEDS]
            ax.plot(
                x,
                [v if v is not None else np.nan for v in vals],
                "o-",
                color=COLORS[arm],
                label=arm,
            )
        ax.set_xticks(x, [str(s) for s in SEEDS])
        ax.set(title=title, xlabel="Training seed", ylabel=ylabel)
        ax.legend(fontsize=8)
        if field in [
            "post_first_success_gripper_command_abs_mean",
            "post_first_success_success_occupancy",
        ]:
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(bottom=0)
    fig.suptitle(
        "Passive post-first-success diagnostics · differing conditioned subsets"
    )
    fig.tight_layout()
    fig.savefig(output / "holding.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, seed in enumerate(SEEDS):
        for arm in ARMS:
            rows = objectives[str(seed)][arm]["updates"]
            axes[0, col].plot(
                [r["timesteps"] / 1000 for r in rows],
                [
                    r["probe"]["scaled_teacher_gradient_norm"]
                    / r["probe"]["ppo_actor_gradient_norm"]
                    if r["probe"]["ppo_actor_gradient_norm"] > 0
                    else np.nan
                    for r in rows
                ],
                color=COLORS[arm],
                label=arm,
            )
            ds = diagnostics[str(seed)][arm]["updates"]
            axes[1, col].plot(
                [r["timesteps"] / 1000 for r in ds],
                gradient_series(ds, "first_step_parameter_delta_l2", "actor"),
                color=COLORS[arm],
                label=arm,
            )
        axes[0, col].set(
            title=f"Training seed {seed}",
            ylabel="Scaled teacher /PPO actor gradient norm",
        )
        axes[1, col].set(ylabel="Measured first Adam actor delta L2")
        for row in range(2):
            axes[row, col].set_xlabel("Transitions (thousands)")
            axes[row, col].set_yscale("symlog", linthresh=1e-6)
            axes[row, col].legend(fontsize=8)
    fig.suptitle(
        "First-minibatch gradient probe (top) versus actual first optimizer step (bottom)"
    )
    fig.tight_layout()
    fig.savefig(output / "objectives.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for col, seed in enumerate(SEEDS):
        for arm in ARMS:
            rows = objectives[str(seed)][arm]["updates"]
            xvalues = [r["timesteps"] / 1000 for r in rows]
            values = [
                [r["auxiliary_loss_mean_scored_minibatches"] for r in rows],
                [float(np.mean(r["probe"]["exploration_variance"])) for r in rows],
                [
                    float(np.mean(r["probe"]["deterministic_mean_action_variance"]))
                    for r in rows
                ],
            ]
            for row in range(3):
                axes[row, col].plot(xvalues, values[row], color=COLORS[arm], label=arm)
        axes[0, col].set(
            title=f"Training seed {seed}",
            ylabel="Teacher action loss (sum over 8 actions)",
        )
        axes[1, col].set(ylabel="Mean Gaussian exploration variance")
        axes[2, col].set(ylabel="Mean across-state deterministic action variance")
        for row in range(3):
            axes[row, col].set_xlabel("Transitions (thousands)")
            axes[row, col].set_yscale("symlog", linthresh=1e-7)
            axes[row, col].legend(fontsize=8)
    fig.suptitle(
        "Teacher drift over scored minibatches · variances from first pre-update minibatch"
    )
    fig.tight_layout()
    fig.savefig(output / "drift.png", dpi=160)
    plt.close(fig)
    return [
        plot_trajectories(output, f"seed{s}_{a}_final", results[f"seed{s}_{a}_final"])
        for s in SEEDS
        for a in ARMS
    ]


def report(run, output):
    run, output = Path(run), Path(output)
    require(run.resolve() != output.resolve(), "Report output must differ from run")
    output.mkdir(parents=True, exist_ok=True)
    p, proof = read(run / "protocol.json"), read(run / "provenance.json")
    require(
        p["arms"] == ARMS
        and p["coefficients"] == COEFFICIENTS
        and p["test_seeds"] == list(range(250000, 250050)),
        "Wrong strength protocol",
    )
    require(
        proof["all_six_final_budgets_verified_before_evaluation"]
        and proof["all_three_first_performance_rollout_summaries_match"]
        and proof["protocol_sha256"] == sha256(run / "protocol.json"),
        "Missing actual checkpoint/source guard",
    )
    for filename in [
        "protocol.json",
        "protocol_frozen.json",
        "provenance.json",
        "comparison.json",
        "video_manifest.json",
    ]:
        shutil.copy2(run / filename, output / filename)
    if (run / "manualreference_reproduction.json").exists():
        shutil.copy2(
            run / "manualreference_reproduction.json",
            output / "manualreference_reproduction.json",
        )
    for name, digest in p["sources"].items():
        require(
            sha256(run / "launch_sources" / name) == digest,
            f"Changed archived source:{name}",
        )
    shutil.copytree(
        run / "launch_sources", output / "launch_sources", dirs_exist_ok=True
    )
    ref = output / "reference_pr18_selected"
    ref.mkdir(exist_ok=True)
    for path in (run / "reference_pr18_selected").glob("*.json"):
        shutil.copy2(path, ref / path.name)
    results, geometry, holding, traces = {}, {}, {}, []

    def export(name, path, seeds):
        result = read(path)
        results[name] = result
        geometry[name], holding[name], entry = export_result(
            result, output, name, seeds
        )
        traces.append({"condition": name, "directory": ".", **entry})

    export("teacher", run / "test_teacher.json", p["test_seeds"])
    validation, objectives, diagnostics = {}, {}, {}
    for seed in SEEDS:
        validation[str(seed)], objectives[str(seed)], diagnostics[str(seed)] = (
            {},
            {},
            {},
        )
        for arm in ARMS:
            src = run / f"seed{seed}" / arm
            dest = output / f"seed{seed}" / arm
            dest.mkdir(parents=True, exist_ok=True)
            for name in [
                "initialization",
                "training",
                "validation",
                "selected",
                "training_metrics",
                "retention",
                "diagnostics",
            ]:
                shutil.copy2(src / f"{name}.json", dest / f"{name}.json")
            validation[str(seed)][arm] = read(src / "validation.json")
            objectives[str(seed)][arm] = read(src / "retention.json")
            diagnostics[str(seed)][arm] = read(src / "diagnostics.json")
            for tag in ["final", "best"]:
                export(
                    f"seed{seed}_{arm}_{tag}",
                    run / f"seed{seed}" / f"test_{arm}_{tag}.json",
                    p["test_seeds"],
                )
    require(
        read(run / "comparison.json") == {n: r["summary"] for n, r in results.items()},
        "Comparison differs from episodes",
    )
    videos = {}
    manifest = read(run / "video_manifest.json")
    require(len(manifest) == 14, "Missing native videos")
    for entry in manifest:
        file = Path(entry["file"])
        require(
            sha256(run / file) == entry["sha256"]
            and entry["all_episode_and_holding_records_match"],
            "Video changed",
        )
        (output / file).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run / file, output / file)
        videos[(entry["condition"], entry["seed"])] = str(file)
    for name in ["teacher", *[f"seed{s}_{a}_final" for s in SEEDS for a in ARMS]]:
        relative = Path(".") if name == "teacher" else Path(name.split("_")[0])
        stem = (
            "video_teacher"
            if name == "teacher"
            else "video_" + name.removeprefix(str(relative) + "_")
        )
        data = read(run / relative / f"{stem}.json")
        verify_video(results[name], data, p["video_seeds"])
        export_result(data, output, f"video_{name}", p["video_seeds"])
    criterion = promising_criterion(geometry)
    paired = paired_outcomes(results)
    paired_rows = "".join(
        f"<tr><th>{r['training_seed']}</th><td>{r['lambda25_vs_lambda10_final']['new_successes']}</td><td>{r['lambda25_vs_lambda10_final']['reference_successes_lost']}</td>"
        + "".join(
            f"<td>{r[f'{a}_vs_teacher_final']['new_successes']} / {r[f'{a}_vs_teacher_final']['reference_successes_lost']}</td>"
            for a in ARMS
        )
        + "</tr>"
        for r in paired["rows"]
    )
    reproduction_notice = ""
    if (run / "manualreference_reproduction.json").exists():
        reproduction = read(run / "manualreference_reproduction.json")
        reproduction_notice = f'<p>All three λ10 final policy tensors exactly reproduce PR18: {reproduction["all_final_policy_tensors_bitwise_equal"]}. This is a read-only weight comparison, not another test evaluation. <a href="manualreference_reproduction.json">Reference reproduction audit</a>.</p>'
    cases = plots(
        output, results, geometry, criterion, validation, objectives, diagnostics
    )
    for name, data in [
        ("criterion", criterion),
        ("paired_outcomes", paired),
        ("geometry", geometry),
        ("holding", holding),
        ("trace_manifest", traces),
        ("trajectory_cases", cases),
    ]:
        save_json(output / f"{name}.json", data)
    rows = "".join(
        f"<tr><th>{r['training_seed']}</th><td>{r['lambda10_final']}/50</td><td>{r['lambda25_final']}/50</td><td>{r['lambda25_minus_lambda10_final']:+d}</td><td>{r['lambda10_best']}/50</td><td>{r['lambda25_best']}/50</td><td>{r['lambda10_selected_minus_final']:+d}</td><td>{r['lambda25_selected_minus_final']:+d}</td></tr>"
        for r in criterion["rows"]
    )
    deltas = ", ".join(
        f"seed {r['training_seed']}: {r['lambda25_minus_lambda10_final']:+d}/50"
        for r in criterion["rows"]
    )
    means = {
        a: 2 * statistics.mean(r[f"{a}_final"] for r in criterion["rows"]) for a in ARMS
    }
    result_notice = f"λ25 changes final success versus λ10 by {deltas}. Mean final success across the three training seeds is {means['lambda10']:.1f}% at λ10 and {means['lambda25']:.1f}% at λ25. These are descriptive seed-level averages."
    denominators = ""
    sections = ""
    for seed in SEEDS:
        for arm in ARMS:
            h = holding[f"seed{seed}_{arm}_final"]
            r = h["retention"]
            i = h["immediate_success_conditioned"]
            denominators += f"<tr><th>{seed} {arm}</th><td>{r['episodes_ever_successful']}</td><td>{r['post_first_success_steps']}</td><td>{i['conditioned_steps']}</td><td>{h['trace_associations']['ever_successful_static_displaced_final_failures']}</td></tr>"
        sections += f'<section data-training-seed="{seed}" class="seed-videos" {"" if seed == 42 else "hidden"}>'
        for reset in p["video_seeds"]:
            sections += (
                f'<div class="videos"><h3>Training seed {seed} · reset{reset}</h3>'
            )
            for name, label in [
                ("teacher", "BC teacher"),
                (f"seed{seed}_lambda10_final", "λ10 final"),
                (f"seed{seed}_lambda25_final", "λ25 final"),
            ]:
                sections += f'<figure><video controls preload="metadata" src="{html.escape(videos[(name, reset)])}"></video><figcaption>{label}</figcaption></figure>'
            sections += "</div>"
        sections += "</section>"
    trajectories = "".join(
        f"<details><summary>{c['condition']}: {len(c['cases'])} qualifying cases</summary>"
        + "".join(
            f'<img loading="lazy" src="{f}" alt="All successful-then-displaced cube-goal distance trajectories">'
            for f in c["plots"]
        )
        + "</details>"
        for c in cases
    )
    (output / "index.html").write_text(
        f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PickCube fixed retention strength</title><style>body{{font:16px/1.6 system-ui;background:#f1f6f7;color:#193642;margin:0}}main{{max-width:1280px;margin:auto;padding:30px 20px}}h1{{font-size:37px;line-height:1.2}}.notice{{background:#fff0cd;border-left:5px solid #a77928;padding:18px}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #d5dfe2}}.scroll{{overflow:auto}}img,video{{width:100%}}.videos{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;background:white;padding:20px;margin:24px 0}}.videos h3{{grid-column:1/-1;margin:0}}figure{{margin:0}}details{{background:white;padding:14px;margin:12px 0}}summary{{font-weight:600;cursor:pointer}}[hidden]{{display:none!important}}a{{color:#146b7e}}select{{font:inherit;padding:8px}}@media(max-width:750px){{.videos{{grid-template-columns:1fr}}h1{{font-size:28px}}}}</style><main><p>PR19 · fixed strength ablation · three paired training seeds</p><h1>Does λ25 retain the actor better than λ10?</h1><p><strong>Predeclared descriptive rule: {"met" if criterion["promising_rule_met"] else "not met"}.</strong> Target seed 2042 gain≥5/50: {criterion["target_gain_at_least_five"]}; each other seed loses≤2/50: {criterion["each_other_seed_loses_at_most_two"]}; mean signed selection−final collapse no worse: {criterion["mean_signed_collapse_not_worse"]}. BC teacher on these fresh resets: {geometry["teacher"]["final_successes"]}/50.</p><p><strong>{result_notice}</strong></p><div class="notice">The only training treatment is fixed positive teacher-loss strength 10 versus 25. Seed 2042 was targeted from PR18 before this study. There is no new pilot, calibration, strength search, adaptive budget, controller override or held-out tuning. All six fresh learners receive 262,144 transitions. Same 50 fresh reset seeds across three training seeds do not create 150 independent reset trials. This short budget cannot establish long-run stability.</div><div class="scroll"><table><tr><th>Training seed</th><th>λ10 final · primary</th><th>λ25 final · primary</th><th>Final change</th><th>λ10 selected · secondary</th><th>λ25 selected · secondary</th><th>λ10 selected−final</th><th>λ25 selected−final</th></tr>{rows}</table></div><img src="outcomes.png" alt="Three-seed primary final outcomes and secondary signed collapse"><img src="validation.png" alt="Forty-reset validation curves for both strengths in all seeds"><p>Validation resets 230000–230039 are deliberately reused. Tests 250000–250049 are fresh relative to PR18. Validation selection ranks final success, any-step success, then return, with step zero eligible. Selection precedes the boundary optimizer update; final weights include the last update.</p><h2>Paired final successes</h2><p>New/lost episodes compare the same reset seeds. Teacher gains/losses use the fresh-test teacher.</p><div class="scroll"><table><tr><th>Training seed</th><th>λ25 new vs λ10</th><th>λ10 successes lost</th><th>λ10 new/lost vs teacher</th><th>λ25 new/lost vs teacher</th></tr>{paired_rows}</table></div><p><a href="paired_outcomes.json">All paired outcomes, including selected-to-final losses</a></p><h2>Passive holding diagnostics</h2><p>After-first-success means steps strictly after the command that first produced success, including later failed states. Immediately-successful pre-state commands form a separate subset. Policies visit different states, so these are not matched-state causal effects. Missing subsets remain missing. Commands, signed gripper-command changes, measured finger motion and grasp flags are separate quantities. Same-step loss associations do not prove contact causality.</p><img src="holding.png" alt="Mean post-success commands, speeds, BC drift, occupancy and losses"><div class="scroll"><table><tr><th>Final policy</th><th>Ever-success episodes /50</th><th>After-first steps</th><th>Immediately-successful pre-state steps</th><th>Ever-successful static-displaced final losses</th></tr>{denominators}</table></div><p><a href="holding.json">All conditioned summaries and loss associations</a> · <a href="geometry.json">Independent final geometry and failures</a></p><h2>Gradient probes versus actual updates</h2><p>Both strengths use the unchanged PR18 objective and bounded critic. Top panels measure teacher/PPO actor gradient ratios on the first pre-update minibatch; bottom panels measure actual actor parameter deltas at the first executed Adam step. These are different measurements, not whole-update norms. Optimizer-step gaps remain gaps.</p><img src="objectives.png" alt="Separate first-minibatch objective probes and first actual Adam actor deltas"><img src="drift.png" alt="Teacher action loss, Gaussian exploration variance and separate across-state action variance"><p>Teacher loss averages all scored training minibatches; both variance measurements use the first pre-update minibatch. Gaussian exploration variance measures stochastic sampling. Deterministic-mean variance across different states is not temporal jitter.</p><h2>Every successful-then-displaced final case</h2><p>All episodes that achieved success and finish with actual 3D cube–goal distance&gt;2.5cm, sorted by reset seed. No favorable-case selection. A static final arm can still hold the cube outside tolerance. Vertical dashed lines mark first success; horizontal dotted lines mark placement tolerance. Full states/actions are exported separately. Expand each condition to inspect all its cases.</p>{trajectories}<p><a href="trajectory_cases.json">Complete case-selection manifest</a></p><h2>Native first-two-reset rollouts</h2><p>All 14 native files use reset 250000 or 250001, regardless of outcome. Episode fields and every holding record match full held-out evaluation.</p><label for="seed">Training seed </label><select id="seed">{"".join(f'<option value="{s}">{s}</option>' for s in SEEDS)}</select>{sections}<h2>Controls and evidence</h2>{reproduction_notice}<p>All 18 actual initial/final/selected checkpoints and six final budgets are verified before held-out evaluation. Each pair matches canonical initial tensors, teacher tensors, parent RNG manifests and actual first-rollout summaries. The original BC teacher stays frozen outside policy Adam. Historical PR18 selected checkpoints are retained by hash only as reference metadata and are not evaluated, used for initialization or used as supervision in this comparison.</p><p><a href="protocol.json">Protocol</a> · <a href="protocol_frozen.json">Frozen protocol hash</a> · <a href="criterion.json">Criterion arithmetic</a> · <a href="provenance.json">Actual checkpoint verification</a> · <a href="comparison.json">Thirteen evaluation summaries</a> · <a href="trace_manifest.json">Episode/compressed trace hashes</a> · <a href="video_manifest.json">Video hashes</a> · <a href="reference_pr18_selected/manifest.json">Historical reference manifest</a></p></main><script>document.getElementById('seed').addEventListener('change',e=>{{document.querySelectorAll('.seed-videos').forEach(s=>{{s.hidden=s.dataset.trainingSeed!==e.target.value;if(s.hidden)s.querySelectorAll('video').forEach(v=>v.pause())}})}})</script></html>"""
    )
    return criterion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.run, args.output)


if __name__ == "__main__":
    main()
