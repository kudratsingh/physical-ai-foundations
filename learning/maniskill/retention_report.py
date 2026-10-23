"""Compact, native visual evidence for the three-seed actor-retention experiment."""

import argparse
import gzip
import html
import json
import shutil
import statistics
from pathlib import Path

import numpy as np

from .critic_report import checked_geometry, gradient_series
from .experiment import save_json
from .metrics import summarize
from .normalized_ppo import sha256
from .retention_compare import ARMS, SEEDS, verify_video
from .retention_evaluation import retention_summary

COLORS = {"baseline": "#bb623d", "anchored": "#187c88"}


def read(path):
    return json.loads(Path(path).read_text())


def jsonl_bytes(rows):
    return "".join(
        json.dumps(r, separators=(",", ":"), allow_nan=False) + "\n" for r in rows
    ).encode()


def calibration_analysis(calibration):
    chosen = calibration["coefficient"]
    ratios = []
    for p in calibration["probes"]:
        pg, aux = p["ppo_actor_gradient_norm"], p["teacher_actor_gradient_norm"]
        ratios.append(
            chosen * aux / pg
            if pg > 0 and np.isfinite(pg) and np.isfinite(aux)
            else None
        )
    valid = [r for r in ratios if r is not None]
    ideal = statistics.median(calibration["ratios"])
    limits = calibration["method"]["coefficient_clip"]
    return {
        "coefficient": chosen,
        "unclipped_median_coefficient": ideal,
        "coefficient_hit_lower_limit": chosen == limits[0],
        "coefficient_hit_upper_limit": chosen == limits[1],
        "chosen_scaled_teacher_over_ppo_ratios": ratios,
        "chosen_ratio_median": statistics.median(valid) if valid else None,
        "target_ratio": calibration["method"]["target_gradient_ratio"],
        "scope": "Recomputed chosen coefficient times teacher gradient norm / PPO actor gradient norm on four calibration minibatches. Pilot's stored scaled_teacher_gradient_norm uses its coefficient zero and is not the chosen ratio.",
    }


def promising_criterion(geometry):
    """Three paired training seeds; do not pool resets into 150 trials."""
    teacher = geometry["teacher"]["final_successes"]
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
                "anchored_meets_five_success_margin_against_both": counts[
                    "anchored_final"
                ]
                >= teacher + 5
                and counts["anchored_final"] >= counts["baseline_final"] + 5,
                "baseline_selected_minus_final": counts["baseline_best"]
                - counts["baseline_final"],
                "anchored_selected_minus_final": counts["anchored_best"]
                - counts["anchored_final"],
            }
        )
    baseline_collapse = statistics.mean(
        r["baseline_selected_minus_final"] for r in rows
    )
    anchored_collapse = statistics.mean(
        r["anchored_selected_minus_final"] for r in rows
    )
    margins = sum(r["anchored_meets_five_success_margin_against_both"] for r in rows)
    return {
        "teacher_final_successes": teacher,
        "rows": rows,
        "training_seed_margin_passes": margins,
        "mean_baseline_selected_minus_final_successes": baseline_collapse,
        "mean_anchored_selected_minus_final_successes": anchored_collapse,
        "mean_collapse_decreased": anchored_collapse < baseline_collapse,
        "promising_rule_met": margins >= 2 and anchored_collapse < baseline_collapse,
        "rule": "Anchored final exceeds both teacher and paired baseline final by at least 5/50 in at least two of three training seeds, AND mean(selected-final) collapse decreases.",
        "scope": "Descriptive predeclared rule, not statistical significance. The same 50 reset seeds are reused across three training seeds.",
    }


def immediate_success_summary(records):
    """Separate states currently successful from all states after first success."""
    rows = [r for r in records if r["pre"]["success"]]

    def average(values):
        return float(np.mean(values)) if values else None

    return {
        "conditioned_steps": len(rows),
        "episodes": len({r["seed"] for r in rows}),
        "arm_command_l2_mean": average(
            [float(np.linalg.norm(r["command"][:7])) for r in rows]
        ),
        "gripper_command_abs_mean": average([abs(r["command"][7]) for r in rows]),
        "post_max_arm_speed_mean": average([r["post"]["max_arm_speed"] for r in rows]),
        "teacher_arm_drift_l2_mean": average([r["teacher_arm_drift_l2"] for r in rows]),
        "teacher_gripper_drift_abs_mean": average(
            [r["teacher_gripper_drift_abs"] for r in rows]
        ),
        "next_state_success_occupancy": average([r["post"]["success"] for r in rows]),
        "pre_grasp_fraction": average([r["pre"]["grasped"] for r in rows]),
        "post_grasp_fraction": average([r["post"]["grasped"] for r in rows]),
        "success_to_failure_transitions": sum(not r["post"]["success"] for r in rows),
        "scope": "Only commands whose immediate pre-state is officially successful; not the larger after-first-success subset.",
    }


def trace_associations(records, episodes):
    """Continuous command changes and observed loss associations, never causes."""
    previous, enriched = {}, []
    for r in records:
        prior = previous.get(r["seed"])
        delta = r["command"][7] - prior if prior is not None else None
        previous[r["seed"]] = r["command"][7]
        enriched.append((r, delta))

    def subset(rows):
        valid = [abs(delta) for _, delta in rows if delta is not None]
        return {
            "steps": len(rows),
            "episodes": len({r["seed"] for r, _ in rows}),
            "gripper_command_delta_pairs": len(valid),
            "gripper_command_delta_abs_mean": float(np.mean(valid)) if valid else None,
            "pre_grasp_fraction": float(np.mean([r["pre"]["grasped"] for r, _ in rows]))
            if rows
            else None,
            "post_grasp_fraction": float(
                np.mean([r["post"]["grasped"] for r, _ in rows])
            )
            if rows
            else None,
        }

    success_losses = [
        (r, d) for r, d in enriched if r["pre"]["success"] and not r["post"]["success"]
    ]
    grasp_losses = [
        (r, d) for r, d in enriched if r["pre"]["grasped"] and not r["post"]["grasped"]
    ]
    events = []
    for r, delta in enriched:
        lost_success = r["pre"]["success"] and not r["post"]["success"]
        lost_grasp = r["pre"]["grasped"] and not r["post"]["grasped"]
        if lost_success or lost_grasp:
            events.append(
                {
                    "seed": r["seed"],
                    "step": r["step"],
                    "success_lost": lost_success,
                    "grasp_lost": lost_grasp,
                    "pre_grasped": r["pre"]["grasped"],
                    "post_grasped": r["post"]["grasped"],
                    "gripper_command": r["command"][7],
                    "gripper_command_delta": delta,
                    "finger_position_delta_l2": float(
                        np.linalg.norm(r["gripper_delta"])
                    ),
                    "pre_cube_goal_distance": r["pre"]["cube_goal_distance"],
                    "post_cube_goal_distance": r["post"]["cube_goal_distance"],
                    "post_max_arm_speed": r["post"]["max_arm_speed"],
                    "post_static": r["post"]["static"],
                }
            )
    return {
        "all_steps": subset(enriched),
        "after_first_success": subset(
            [(r, d) for r, d in enriched if r["after_first_success"]]
        ),
        "immediate_pre_success": subset(
            [(r, d) for r, d in enriched if r["pre"]["success"]]
        ),
        "success_loss_steps": subset(success_losses),
        "grasp_loss_steps": subset(grasp_losses),
        "success_loss_and_grasp_loss_same_step": sum(
            r["pre"]["grasped"] and not r["post"]["grasped"] for r, _ in success_losses
        ),
        "success_loss_without_grasp_loss_same_step": sum(
            not (r["pre"]["grasped"] and not r["post"]["grasped"])
            for r, _ in success_losses
        ),
        "ever_successful_static_displaced_final_failures": sum(
            e["success_once"]
            and not e["success_at_end"]
            and e["final_is_robot_static"]
            and e["final_goal_distance"] > 0.025
            for e in episodes
        ),
        "loss_events": events,
        "scope": "Gripper COMMAND delta is change from the previous command in that episode; first command has no delta. Finger-position delta is measured motion in metres. Same-step associations do not establish contact or control causality. No thresholds are chosen from test outcomes.",
    }


def export_result(result, directory, stem, seeds):
    directory.mkdir(parents=True, exist_ok=True)
    geometry = checked_geometry(result, seeds)
    records, episodes = result["step_records"], result["episodes"]
    if result["summary"] != summarize(episodes) or result[
        "retention"
    ] != retention_summary(records, episodes):
        raise ValueError("Stored summary/holding aggregate differs from actual records")
    if len(records) != len(seeds) * 50:
        raise ValueError("Missing full-horizon holding records")
    for seed in seeds:
        rows = [r for r in records if r["seed"] == seed]
        if [r["step"] for r in rows] != list(range(1, 51)):
            raise ValueError("Missing/out-of-order holding steps")
        for i, r in enumerate(rows):
            if (
                np.asarray(r["observation"]).shape != (42,)
                or len(r["command"]) != 8
                or len(r["teacher_command"]) != 8
            ):
                raise ValueError("Wrong passive observation/action schema")
            if i and rows[i - 1]["post"] != r["pre"]:
                raise ValueError("Holding pre/post states are not synchronized")
    episodes_path = directory / f"{stem}.jsonl"
    episodes_path.write_bytes(
        jsonl_bytes(
            [{k: v for k, v in e.items() if k != "trajectory" or v} for e in episodes]
        )
    )
    traces_path = directory / f"{stem}_steps.jsonl.gz"
    traces_path.write_bytes(gzip.compress(jsonl_bytes(records), mtime=0))
    holding = {
        "summary": result["summary"],
        "retention": result["retention"],
        "episode_retention": result["episode_retention"],
        "immediate_success_conditioned": immediate_success_summary(records),
        "trace_associations": trace_associations(records, episodes),
    }
    save_json(directory / f"{stem}_holding.json", holding)
    return (
        geometry,
        holding,
        {
            "episodes": episodes_path.name,
            "episodes_sha256": sha256(episodes_path),
            "steps": traces_path.name,
            "steps_sha256": sha256(traces_path),
            "step_records": len(records),
        },
    )


def plots(output, results, geometry, criterion, validation, objective, diagnostics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    x = np.arange(3)
    for arm, offset in [("baseline", -0.18), ("anchored", 0.18)]:
        values = [geometry[f"seed{s}_{arm}_final"]["final_successes"] for s in SEEDS]
        bars = axes[0].bar(x + offset, values, width=0.34, color=COLORS[arm], label=arm)
        axes[0].bar_label(bars, padding=3)
        collapse = [r[f"{arm}_selected_minus_final"] for r in criterion["rows"]]
        axes[1].bar(x + offset, collapse, width=0.34, color=COLORS[arm], label=arm)
    axes[0].axhline(
        geometry["teacher"]["final_successes"],
        linestyle=":",
        color="#64748b",
        label="BC teacher",
    )
    axes[0].set(
        title="Primary: fixed-budget finals",
        ylabel="Final successes / same 50 resets",
        ylim=(0, 55),
    )
    axes[1].axhline(0, color="#64748b", linewidth=0.8)
    axes[1].set(
        title="Secondary selection minus fixed final",
        ylabel="Success-count difference / 50",
    )
    for ax in axes:
        ax.set_xticks(x, [str(s) for s in SEEDS])
        ax.set_xlabel("Training seed")
        ax.legend()
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
            ylabel="Final success / 40 validation resets (%)",
            ylim=(-2, 102),
        )
        ax.legend()
    fig.suptitle("Deterministic validation · selection is secondary")
    fig.tight_layout()
    fig.savefig(output / "validation.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fields = [
        (
            "post_first_success_arm_command_l2_mean",
            "Mean arm command magnitude",
            "Normalized command L2",
        ),
        (
            "post_first_success_gripper_command_abs_mean",
            "Mean absolute gripper command",
            "Normalized command absolute value",
        ),
        (
            "post_first_success_max_arm_speed_mean",
            "Mean post-state max |arm qvel|",
            "rad/s",
        ),
        (
            "post_first_success_teacher_arm_drift_l2_mean",
            "Mean deviation from BC on current state",
            "Arm-command L2",
        ),
        (
            "post_first_success_success_occupancy",
            "Success occupancy after first success",
            "Fraction of conditioned steps",
        ),
        (
            "success_to_failure_transitions",
            "Success → failure transitions",
            "Transitions across 50 episodes",
        ),
    ]
    for ax, (field, title, ylabel) in zip(axes.flat, fields, strict=True):
        for arm, style in [("baseline", "o-"), ("anchored", "s-")]:
            values = [
                results[f"seed{s}_{arm}_final"]["retention"][field] for s in SEEDS
            ]
            ax.plot(
                x,
                [v if v is not None else np.nan for v in values],
                style,
                color=COLORS[arm],
                label=arm,
            )
        ax.set_xticks(x, [str(s) for s in SEEDS])
        ax.set(title=title, xlabel="Training seed", ylabel=ylabel)
        if field in [
            "post_first_success_gripper_command_abs_mean",
            "post_first_success_success_occupancy",
        ]:
            ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Passive holding diagnostics · different conditioned subsets · missing subset is a gap"
    )
    fig.tight_layout()
    fig.savefig(output / "holding.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    adam_fig, adam_axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for column, seed in enumerate(SEEDS):
        for arm in ARMS:
            rows = objective[str(seed)][arm]["updates"]
            xs = [r["timesteps"] / 1000 for r in rows]
            ratios = [
                r["probe"]["scaled_teacher_gradient_norm"]
                / r["probe"]["ppo_actor_gradient_norm"]
                if r["probe"]["ppo_actor_gradient_norm"] > 0
                else np.nan
                for r in rows
            ]
            axes[0, column].plot(xs, ratios, color=COLORS[arm], label=arm)
            axes[1, column].plot(
                xs,
                [
                    r["probe"]["gradient_cosine"]
                    if r["probe"]["gradient_cosine"] is not None
                    else np.nan
                    for r in rows
                ],
                color=COLORS[arm],
                label=arm,
            )
            actual = diagnostics[str(seed)][arm]["updates"]
            for group, style in [("actor", "-"), ("critic", "--")]:
                adam_axes[column].plot(
                    [r["timesteps"] / 1000 for r in actual],
                    gradient_series(actual, "first_step_parameter_delta_l2", group),
                    style,
                    color=COLORS[arm],
                    label=f"{arm}: {group}",
                )
        axes[0, column].axhline(
            0.5, color="#64748b", linestyle=":", label="Calibration reference 0.5"
        )
        axes[0, column].set_yscale("symlog", linthresh=0.1)
        axes[0, column].set(
            title=f"Training seed {seed}",
            ylabel="Scaled teacher / PPO actor gradient norm",
        )
        axes[1, column].set(ylabel="Gradient cosine", ylim=(-1.05, 1.05))
        for row in range(2):
            axes[row, column].set_xlabel("Transitions (thousands)")
            axes[row, column].legend(fontsize=7)
        adam_axes[column].set(
            title=f"Training seed {seed}",
            xlabel="Transitions (thousands)",
            ylabel="Measured parameter-delta L2",
        )
        adam_axes[column].set_yscale("symlog", linthresh=1e-7)
        adam_axes[column].legend(fontsize=7)
    fig.suptitle(
        "Separate gradient probe · first pre-update minibatch only · coefficient is frozen"
    )
    fig.tight_layout()
    fig.savefig(output / "objective.png", dpi=160)
    plt.close(fig)
    adam_fig.suptitle(
        "Actual first Adam step per rollout · actor solid, critic dashed · no-step gaps"
    )
    adam_fig.tight_layout()
    adam_fig.savefig(output / "adam.png", dpi=160)
    plt.close(adam_fig)

    drift_fig, drift_axes = plt.subplots(3, 3, figsize=(15, 10))
    for column, seed in enumerate(SEEDS):
        for arm in ARMS:
            rows = objective[str(seed)][arm]["updates"]
            xs = [r["timesteps"] / 1000 for r in rows]
            series = [
                [r["auxiliary_loss_mean_scored_minibatches"] for r in rows],
                [float(np.mean(r["probe"]["exploration_variance"])) for r in rows],
                [
                    float(np.mean(r["probe"]["deterministic_mean_action_variance"]))
                    for r in rows
                ],
            ]
            for row, values in enumerate(series):
                drift_axes[row, column].plot(xs, values, color=COLORS[arm], label=arm)
        drift_axes[0, column].set(
            title=f"Training seed {seed}", ylabel="Teacher action MSE · sum over 8"
        )
        drift_axes[1, column].set(ylabel="Mean exploration variance")
        drift_axes[2, column].set(ylabel="Mean across-state action variance")
        for row in range(3):
            drift_axes[row, column].set_xlabel("Transitions (thousands)")
            drift_axes[row, column].set_ylim(bottom=0)
            drift_axes[row, column].legend(fontsize=7)
    drift_fig.suptitle(
        "Teacher drift and distinct action-variance measurements · state variance is not temporal jitter"
    )
    drift_fig.tight_layout()
    drift_fig.savefig(output / "drift.png", dpi=160)
    plt.close(drift_fig)


def report(run, output):
    run, output = Path(run), Path(output)
    if run.resolve() == output.resolve():
        raise ValueError("Evidence output must differ from the training run")
    output.mkdir(parents=True, exist_ok=True)
    p = read(run / "protocol.json")
    if p["training_seeds"] != SEEDS or p["test_seeds"] != list(range(240000, 240050)):
        raise ValueError("Unexpected three-seed protocol")
    provenance = read(run / "provenance.json")
    if not provenance["all_six_final_budgets_verified_before_evaluation"]:
        raise ValueError("Missing actual six-checkpoint budget guard")
    for name in [
        "protocol.json",
        "calibration.json",
        "calibration_frozen.json",
        "provenance.json",
        "comparison.json",
        "video_manifest.json",
    ]:
        shutil.copy2(run / name, output / name)
    if provenance["protocol_sha256"] != sha256(run / "protocol.json"):
        raise ValueError("Protocol changed after checkpoint verification")
    for name, expected in p["sources"].items():
        if sha256(run / "launch_sources" / name) != expected:
            raise ValueError(f"Changed archived source: {name}")
    shutil.copytree(
        run / "launch_sources", output / "launch_sources", dirs_exist_ok=True
    )
    shutil.copytree(run / "first_rollout", output / "first_rollout", dirs_exist_ok=True)
    first = read(run / "first_rollout/receipt.json")
    if (
        sha256(run / "first_rollout" / first["actual_records"])
        != first["records_sha256"]
    ):
        raise ValueError("Pre-update actual records changed")
    calibration = calibration_analysis(read(run / "calibration.json"))
    save_json(output / "calibration_analysis.json", calibration)
    results, geometry, holding, traces = {}, {}, {}, []

    def export(name, path, relative, stem, seeds):
        result = read(path)
        results[name] = result
        geometry[name], holding[name], entry = export_result(
            result, output / relative, stem, seeds
        )
        traces.append({"condition": name, "directory": str(relative), **entry})
        return result

    export("teacher", run / "test_teacher.json", Path("."), "teacher", p["test_seeds"])
    validation, objective, diagnostics = {}, {}, {}
    for seed in SEEDS:
        validation[str(seed)], objective[str(seed)], diagnostics[str(seed)] = {}, {}, {}
        for arm in ARMS:
            source = run / f"seed{seed}" / arm
            target = output / f"seed{seed}" / arm
            target.mkdir(parents=True, exist_ok=True)
            for name in [
                "initialization",
                "training",
                "validation",
                "selected",
                "training_metrics",
                "retention",
                "diagnostics",
            ]:
                shutil.copy2(source / f"{name}.json", target / f"{name}.json")
            validation[str(seed)][arm] = read(source / "validation.json")
            objective[str(seed)][arm] = read(source / "retention.json")
            diagnostics[str(seed)][arm] = read(source / "diagnostics.json")
            for tag in ["final", "best"]:
                export(
                    f"seed{seed}_{arm}_{tag}",
                    run / f"seed{seed}" / f"test_{arm}_{tag}.json",
                    Path("."),
                    f"seed{seed}_{arm}_{tag}",
                    p["test_seeds"],
                )
    if read(run / "comparison.json") != {
        name: value["summary"] for name, value in results.items()
    }:
        raise ValueError("Comparison summaries differ from actual episodes")
    manifest = read(run / "video_manifest.json")
    if len(manifest) != 14:
        raise ValueError("Missing declared native videos")
    videos = {}
    for entry in manifest:
        file = Path(entry["file"])
        if (
            sha256(run / file) != entry["sha256"]
            or not entry["all_episode_and_holding_records_match"]
        ):
            raise ValueError("Native video verification mismatch")
        dest = output / file
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run / file, dest)
        videos[(entry["condition"], entry["seed"])] = str(file)
    for name in ["teacher", *[f"seed{s}_{a}_final" for s in SEEDS for a in ARMS]]:
        relative = Path(".") if name == "teacher" else Path(name.split("_")[0])
        stem = (
            "video_teacher"
            if name == "teacher"
            else "video_" + name.removeprefix(str(relative) + "_")
        )
        video = read(run / relative / f"{stem}.json")
        verify_video(results[name], video, p["video_seeds"])
        export_result(video, output, f"video_{name}", p["video_seeds"])
    criterion = promising_criterion(geometry)
    save_json(output / "criterion.json", criterion)
    save_json(output / "geometry.json", geometry)
    save_json(output / "holding.json", holding)
    save_json(output / "trace_manifest.json", traces)
    plots(output, results, geometry, criterion, validation, objective, diagnostics)

    rows = ""
    for row in criterion["rows"]:
        rows += f"<tr><th>{row['training_seed']}</th><td>{row['baseline_final']}/50</td><td>{row['anchored_final']}/50</td><td>{row['baseline_best']}/50</td><td>{row['anchored_best']}/50</td><td>{row['baseline_selected_minus_final']:+d}</td><td>{row['anchored_selected_minus_final']:+d}</td></tr>"
    video_sections = ""
    for train_seed in SEEDS:
        video_sections += f'<section class="seed-videos" data-training-seed="{train_seed}" {"" if train_seed == SEEDS[0] else "hidden"}>'
        for test_seed in p["video_seeds"]:
            video_sections += f'<div class="videos"><h3>Reset seed {test_seed} · training seed {train_seed}</h3>'
            for condition, label in [
                ("teacher", "BC teacher"),
                (f"seed{train_seed}_baseline_final", "Baseline final"),
                (f"seed{train_seed}_anchored_final", "Anchored final"),
            ]:
                video_sections += f'<figure><video controls preload="metadata" src="{html.escape(videos[(condition, test_seed)])}"></video><figcaption>{label}</figcaption></figure>'
            video_sections += "</div>"
        video_sections += "</section>"
    denominator_rows = ""
    final_conditions = []
    for seed in SEEDS:
        for arm in ARMS:
            name = f"seed{seed}_{arm}_final"
            final_conditions.append(name)
            h = holding[name]
            r = h["retention"]
            immediate = h["immediate_success_conditioned"]
            denominator_rows += f"<tr><th>{seed} · {arm}</th><td>{r['episodes_ever_successful']}/50</td><td>{r['episodes_with_post_success_actions']}</td><td>{r['post_first_success_steps']}</td><td>{immediate['conditioned_steps']}</td></tr>"
    weakest = min(criterion["rows"], key=lambda row: row["anchored_final"])
    caveat = f"Training seed {weakest['training_seed']} finishes at {weakest['anchored_final']}/50 versus its validation-selected {weakest['anchored_best']}/50 and teacher {criterion['teacher_final_successes']}/50. Passing the aggregate rule does not mean retention is uniform across training seeds."
    loss_count = sum(
        holding[n]["retention"]["success_to_failure_transitions"]
        for n in final_conditions
    )
    grasp_loss_count = sum(
        holding[n]["trace_associations"]["success_loss_and_grasp_loss_same_step"]
        for n in final_conditions
    )
    options = "".join(f'<option value="{s}">{s}</option>' for s in SEEDS)
    flag = "met" if criterion["promising_rule_met"] else "not met"
    ratio = calibration["chosen_ratio_median"]
    ratio_text = "unavailable" if ratio is None else f"{ratio:.3f}"
    cap = (
        "upper ceiling"
        if calibration["coefficient_hit_upper_limit"]
        else "lower floor"
        if calibration["coefficient_hit_lower_limit"]
        else "interior value"
    )
    (
        output / "index.html"
    ).write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PickCube actor retention</title><style>
body{{font:16px/1.6 system-ui;margin:0;background:#f0f5f6;color:#1b3541}}main{{max-width:1280px;margin:auto;padding:30px 20px}}h1{{font-size:38px;line-height:1.2}}h2{{margin-top:36px}}.notice{{padding:18px;background:#fff0cb;border-left:5px solid #a77a24}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #d5dfe2}}.scroll{{overflow:auto}}img,video{{width:100%;background:white}}.videos{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;background:white;padding:20px;margin:24px 0}}.videos h3{{grid-column:1/-1;margin:0}}figure{{margin:0}}select{{font:inherit;padding:8px}}a{{color:#146d82}}[hidden]{{display:none!important}}@media(max-width:750px){{.videos{{grid-template-columns:1fr}}h1{{font-size:29px}}main{{padding:15px}}}}
</style><main><p>Stage 1 · three paired training seeds · passive holding diagnostics</p><h1>Can an on-policy teacher loss retain a useful manipulation actor?</h1>
<p><strong>Predeclared promising rule: {flag}.</strong> BC teacher final success: {criterion["teacher_final_successes"]}/50. Anchored final must exceed both teacher and paired baseline final by at least five successes in at least two training seeds, and mean selection-to-final collapse must decrease.</p>
<p><strong>{caveat}</strong></p>
<div class="notice">Three training seeds, each with two fixed-budget final models at 262,144 transitions. Every condition uses the same 50 held-out reset seeds: these are not 150 independent reset trials. Validation-selected checkpoints are secondary. No stopping override, privileged control, reward change or demonstration replay is used. This short budget does not establish million-transition stability.</div>
<div class="scroll"><table><tr><th>Training seed</th><th>Baseline final · primary</th><th>Anchored final · primary</th><th>Baseline selected · secondary</th><th>Anchored selected · secondary</th><th>Baseline selected−final</th><th>Anchored selected−final</th></tr>{rows}</table></div><img src="outcomes.png" alt="Seed-level fixed final success and secondary selection-to-final changes"><img src="validation.png" alt="Forty-reset deterministic validation curves for each of three training seeds">
<h2>What changed?</h2><p>Both arms use the same bounded critic and copied BC actor. The anchored arm adds coefficient × mean over current on-policy states of the squared eight-action distance between its deterministic Gaussian mean and a separate frozen BC teacher. The loss uses neither validation/test states nor a replay demonstration batch. It directly supervises the actor mean; it adds no direct critic or log-standard-deviation gradient. Both arms retain ordinary PPO and shared global gradient clipping.</p>
<p>The selected checkpoint uses final success, then any-step success, then return across 40 validation resets, with step zero eligible. Selection happens before that boundary's optimizer update; fixed finals include the last update. A selected initial actor is not learned PPO improvement.</p>
<h2>Calibration was frozen once</h2><p>A separate 8,192-transition pilot at seed 91042 was discarded. Its median ideal coefficient was {calibration["unclipped_median_coefficient"]:.4f}; the chosen coefficient is {calibration["coefficient"]:.4f} ({cap}). Recomputing chosen coefficient × teacher-gradient norm / PPO-gradient norm gives median {ratio_text}, against calibration reference 0.5. A cap hit does not mean the target ratio was reached. The pilot's own stored scaled norm is zero because that pilot used coefficient zero.</p><p><a href="calibration_analysis.json">Derived calibration ratios</a> · <a href="calibration.json">Actual pilot probes</a> · <a href="calibration_frozen.json">Frozen coefficient and hashes</a></p>
<h2>Actor objectives and actual optimizer steps</h2><p>The gradient decomposition is a separate read-only probe on the first shuffled minibatch before each update. It reports PPO and teacher contributions before combined clipping, rather than every minibatch. The Adam chart separately measures the actual first executed optimizer step per rollout; missing steps remain gaps. These are distinct measurements.</p><img src="objective.png" alt="First-minibatch scaled teacher-to-PPO gradient ratios and gradient cosine"><img src="adam.png" alt="Measured first actual Adam actor and critic parameter changes"><img src="drift.png" alt="Teacher action loss, exploration variance and separate across-state action variance"><p>Teacher loss averages all scored training minibatches; variance measurements use the first pre-update minibatch. Exploration variance is Gaussian sampling variance. Across-state deterministic-mean variance describes different inputs and is not a temporal jitter measure.</p>
<h2>What happens after placement and success?</h2><p>After-first-success summaries exclude the command that first produced success, then include all subsequent actions even if success is lost. Policies can reach success on different episodes and at different times, so conditioned subsets differ. An absent subset is missing, not a measured zero. The exported immediate-success-conditioned summaries separately restrict commands to states successful immediately before acting. Finger deltas are measured finger-position changes, distinct from normalized gripper commands.</p><img src="holding.png" alt="Conditional post-success action magnitudes, arm speeds, teacher deviation, success occupancy and loss transitions"><div class="scroll"><table><tr><th>Final policy</th><th>Ever-successful episodes</th><th>Episodes with later commands</th><th>After-first-success steps</th><th>Immediately-successful pre-state steps</th></tr>{denominator_rows}</table></div><p>Across the six final learners, {grasp_loss_count}/{loss_count} success-loss transitions coincide with a lost grasp flag. These are observed same-step associations, not proof about contact forces or earlier gripper changes. Gripper command magnitude uses a full 0–1 axis; speed is the mean of each post-state’s maximum absolute seven-arm velocity.</p><p><a href="holding.json">All holding summaries and conditioned-step counts</a> · <a href="geometry.json">Independent final 3D distances, seven-arm speeds and failure categories</a></p>
<h2>Native held-out rollouts</h2><p>First two predeclared reset seeds, regardless of outcome. Choose the training seed; the teacher is the same reference. All episode fields and every passive holding record match the corresponding full held-out evaluation. Fourteen native files are checksum-verified.</p><label for="training-seed">Training seed </label><select id="training-seed">{options}</select>{video_sections}
<h2>Pairing and evidence</h2><p>All six actual final checkpoint budgets, original teacher tensors, initial paired canonical tensors and parent RNG manifests, coefficients, calibration hashes and archived source hashes were verified before any held-out environment was created. A separate four-worker pre-update receipt compares all 256 stochastic steps, raw rewards, actions, observations and pre-reset terminal observations exactly, spanning 20 timeouts per arm. Value-head outputs and timeout-corrected rewards are excluded because they may differ by design. Worker seeding and advancing episode seeds repair the historical reset-stream gap without relabeling older experiments.</p>
<p><a href="protocol.json">Protocol</a> · <a href="criterion.json">Predeclared criterion calculation</a> · <a href="provenance.json">Actual checkpoint provenance</a> · <a href="first_rollout/receipt.json">Pre-update receipt</a> · <a href="comparison.json">All thirteen test summaries</a> · <a href="trace_manifest.json">Compact episode and compressed full-trace hashes</a> · <a href="video_manifest.json">Native video hashes</a></p></main><script>document.getElementById('training-seed').addEventListener('change',event=>{{document.querySelectorAll('.seed-videos').forEach(section=>{{section.hidden=section.dataset.trainingSeed!==event.target.value;if(section.hidden)section.querySelectorAll('video').forEach(video=>video.pause())}})}})</script></html>""")
    return criterion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.run, args.output)


if __name__ == "__main__":
    main()
