"""Verify all fixed-strength checkpoints before fresh held-out evaluation."""

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from .bc_ppo_policy import load_teacher
from .critic_experiment import canonical_fingerprint
from .experiment import Config, save_json
from .normalized_ppo import sha256
from .retention_compare import check_model, read, require, verify_video
from .retention_evaluation import evaluate_retention
from .retention_experiment import tensor_fingerprint
from .retention_ppo import RetentionPPO
from .rng import isolated_rng
from .selection import holding_score
from .strength_experiment import check_sources

SEEDS = [42, 1042, 2042]
ARMS = ["lambda10", "lambda25"]
COEFFICIENTS = {"lambda10": 10.0, "lambda25": 25.0}


@isolated_rng
def verify_checkpoints(run, teacher_path):
    """Load all 18 checkpoints and check six final budgets without a simulator."""
    torch.set_num_threads(1)
    run = Path(run)
    p, frozen = check_sources(run, teacher_path), read(run / "protocol_frozen.json")
    protocol_sha = frozen["sha256"]
    require(protocol_sha == sha256(run / "protocol.json"), "Declared protocol changed")
    require(
        p["training_seeds"] == SEEDS
        and p["arms"] == ARMS
        and p["coefficients"] == COEFFICIENTS,
        "Changed fixed-strength conditions",
    )
    require(
        p["config"]["total_timesteps"] == 262144
        and p["validation_seeds"] == list(range(230000, 230040))
        and p["test_seeds"] == list(range(250000, 250050))
        and p["video_seeds"] == [250000, 250001],
        "Changed declared task/budget/seeds",
    )
    require(sha256(teacher_path) == p["teacher_sha256"], "Original teacher changed")
    for name, expected in p["sources"].items():
        require(
            sha256(run / "launch_sources" / name) == expected
            and sha256(Path(__file__).with_name(name)) == expected,
            f"Changed frozen source:{name}",
        )
    reference = run / "reference_pr18_selected/manifest.json"
    reference_sha = sha256(reference)
    reference_manifest = read(reference)
    require(
        reference_manifest["models"] == p["historical_reference"]["selected_models"]
        and reference_manifest["evaluation_in_this_study"] is False,
        "Historical reference manifest changed",
    )
    for entry in reference_manifest["models"]:
        seed = entry["training_seed"]
        require(
            sha256(reference.parent / f"seed{seed}_best.zip")
            == entry["checkpoint_sha256"]
            and read(reference.parent / f"seed{seed}_selected.json")
            == entry["selected"],
            "Historical selected reference artifact changed",
        )
    teacher = load_teacher(teacher_path)
    tf = tensor_fingerprint(teacher)
    conditions, first_summaries = {}, {}
    for seed in SEEDS:
        conditions[str(seed)], manifests = {}, []
        for arm in ARMS:
            directory = run / f"seed{seed}" / arm
            manifest = read(directory / "initialization.json")
            manifests.append(manifest)
            require(
                manifest["optimizer_empty"] is True
                and manifest["teacher_fingerprint"] == tf
                and manifest["protocol_sha256"] == protocol_sha
                and manifest["coefficient"] == COEFFICIENTS[arm],
                "Changed initialization manifest",
            )
            path = directory / "initial.zip"
            require(
                sha256(path) == manifest["checkpoint_sha256"],
                "Initial checkpoint hash differs",
            )
            initial = RetentionPPO.load(path, device="cpu")
            entries = {
                "initial": {
                    "sha256": sha256(path),
                    "recipe": check_model(
                        initial, teacher, seed, COEFFICIENTS[arm], 0, initial=True
                    ),
                }
            }
            require(
                canonical_fingerprint(initial) == manifest["canonical_fingerprint"],
                "Loaded initial canonical tensors differ",
            )
            selected, validation = (
                read(directory / "selected.json"),
                read(directory / "validation.json"),
            )
            require(
                selected == max(validation, key=holding_score),
                "Checkpoint selection differs from final-first ranking",
            )
            training = read(directory / "training.json")
            require(
                training["complete"] and training["actual_timesteps"] == 262144,
                "Incomplete training condition",
            )
            for tag, steps in [("final", 262144), ("best", selected["timesteps"])]:
                path = directory / f"{tag}.zip"
                model = RetentionPPO.load(path, device="cpu")
                entries[tag] = {
                    "sha256": sha256(path),
                    "recipe": check_model(
                        model, teacher, seed, COEFFICIENTS[arm], steps
                    ),
                }
            conditions[str(seed)][arm] = entries
        require(
            manifests[0]["canonical_fingerprint"]
            == manifests[1]["canonical_fingerprint"]
            and manifests[0]["rng_fingerprint"] == manifests[1]["rng_fingerprint"]
            and len(manifests[0]["rng_fingerprint"]) == 64,
            "Paired initialization tensors/parent RNG differ",
        )
        first = [
            read(run / f"seed{seed}" / arm / "diagnostics.json")["updates"][0]
            for arm in ARMS
        ]
        require(all(r["timesteps"] == 1024 for r in first), "Wrong first rollout size")
        for field in [
            "values",
            "returns",
            "advantages",
            "buffer_rewards_after_bootstrap",
            "raw",
        ]:
            require(
                first[0][field] == first[1][field],
                f"First pre-update rollout summaries differ: seed{seed} {field}",
            )
        first_summaries[str(seed)] = {
            arm: {
                field: row[field]
                for field in [
                    "timesteps",
                    "values",
                    "returns",
                    "advantages",
                    "buffer_rewards_after_bootstrap",
                    "raw",
                ]
            }
            for arm, row in zip(ARMS, first, strict=True)
        }
    result = {
        "all_six_final_budgets_verified_before_evaluation": True,
        "all_three_first_performance_rollout_summaries_match": True,
        "teacher_sha256": p["teacher_sha256"],
        "teacher_fingerprint": tf,
        "protocol_sha256": protocol_sha,
        "coefficients": COEFFICIENTS,
        "actual_checkpoints": conditions,
        "first_update_summaries": first_summaries,
        "reference_pr18_selected_manifest_sha256": reference_sha,
        "reference_used_for_evaluation": False,
        "comparison_source_sha256": sha256(__file__),
    }
    save_json(run / "provenance.json", result)
    return result


@isolated_rng
def compare(run, teacher_path):
    """Use only after all training finishes; never performs policy updates."""
    run = Path(run)
    proof = verify_checkpoints(run, teacher_path)
    p = read(run / "protocol.json")
    config = Config(**p["config"])
    teacher = load_teacher(teacher_path)
    summaries, videos = {}, []

    def assess(name, policy, task, output, native):
        result = evaluate_retention(task, policy, teacher, p["test_seeds"], output)
        summaries[name] = result["summary"]
        if native:
            video_path = output.with_name(output.name.replace("test_", "video_", 1))
            video = evaluate_retention(
                task, policy, teacher, p["video_seeds"], video_path, video=True
            )
            verify_video(result, video, p["video_seeds"])
            for seed in p["video_seeds"]:
                path = video_path.with_name(f"{video_path.stem}_seed{seed}.mp4")
                require(path.stat().st_size > 0, "Empty native video")
                videos.append(
                    {
                        "condition": name,
                        "seed": seed,
                        "file": str(path.relative_to(run)),
                        "sha256": sha256(path),
                        "all_episode_and_holding_records_match": True,
                    }
                )

    assess("teacher", teacher, config, run / "test_teacher.json", True)
    for seed in SEEDS:
        for arm in ARMS:
            for tag in ["final", "best"]:
                path = run / f"seed{seed}" / arm / f"{tag}.zip"
                require(
                    sha256(path)
                    == proof["actual_checkpoints"][str(seed)][arm][tag]["sha256"],
                    "Checkpoint changed after verification",
                )
                model = RetentionPPO.load(path, device="cpu")
                assess(
                    f"seed{seed}_{arm}_{tag}",
                    model,
                    replace(config, seed=seed),
                    run / f"seed{seed}" / f"test_{arm}_{tag}.json",
                    tag == "final",
                )
    require(len(videos) == 14, "Missing predeclared native videos")
    save_json(run / "comparison.json", summaries)
    save_json(run / "video_manifest.json", videos)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    (verify_checkpoints if args.verify_only else compare)(args.run, args.teacher)


if __name__ == "__main__":
    main()
