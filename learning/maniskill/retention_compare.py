"""Guard every completed retention checkpoint before any held-out evaluation."""

import argparse
import gzip
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .bc_ppo_policy import load_teacher
from .critic_experiment import canonical_fingerprint
from .critic_policy import BoundedCriticPolicy, ValueBound
from .experiment import Config, save_json
from .first_rollout import compare_histories
from .normalized_ppo import sha256
from .retention_evaluation import evaluate_retention, retention_summary
from .retention_experiment import tensor_fingerprint
from .retention_ppo import RetentionPPO
from .rng import isolated_rng
from .selection import holding_score

SEEDS = [42, 1042, 2042]
ARMS = ["baseline", "anchored"]


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_receipt(run, protocol):
    directory = run / "first_rollout"
    receipt = read(directory / "receipt.json")
    require(
        receipt["passed"]
        and receipt["all_arrays_exactly_equal"]
        and receipt["optimizer_empty_after"],
        "First-rollout receipt failed",
    )
    require(
        receipt["teacher_sha256"] == protocol["teacher_sha256"]
        and receipt["source_sha256"] == protocol["sources"]["first_rollout.py"],
        "First-rollout teacher/source mismatch",
    )
    require(
        receipt["workers"] == 4
        and receipt["stochastic_steps_per_worker"] == 256
        and receipt["transitions_per_arm"] == 1024
        and receipt["seed"] == 42
        and receipt["worker_seeds"] == [42, 43, 44, 45],
        "Wrong first-rollout capture size",
    )
    path = directory / receipt["actual_records"]
    require(sha256(path) == receipt["records_sha256"], "Changed first-rollout records")
    rows = [
        json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()
    ]
    histories = {}
    for arm in ["raw", "bounded"]:
        headers = [r for r in rows if r["arm"] == arm and "step" not in r]
        steps = [r for r in rows if r["arm"] == arm and "step" in r]
        require(
            len(headers) == 1 and len(steps) == 256, "Incomplete first-rollout records"
        )
        require(
            np.asarray(headers[0]["initial_observation"]).shape == (4, 42),
            "Wrong first-rollout state shape",
        )
        for index, row in enumerate(steps, 1):
            require(
                row["step"] == index
                and np.asarray(row["pre_observation"]).shape == (4, 42)
                and np.asarray(row["post_observation"]).shape == (4, 42),
                "Wrong first-rollout observation sequence",
            )
            require(
                np.asarray(row["gaussian_sample"]).shape == (4, 8)
                and np.asarray(row["executed_command"]).shape == (4, 8),
                "Wrong first-rollout action shape",
            )
            require(
                len(row["raw_reward"])
                == len(row["done"])
                == len(row["terminal_observation"])
                == 4,
                "Wrong first-rollout worker count",
            )
            require(
                all(0 <= r <= 1 for r in row["raw_reward"]),
                "First-rollout rewards were not raw task rewards",
            )
            require(
                np.array_equal(
                    np.clip(row["gaussian_sample"], -1, 1), row["executed_command"]
                ),
                "First-rollout executed commands mismatch",
            )
            expected_done = index % 50 == 0
            require(
                row["done"] == row["timeout"] == [expected_done] * 4,
                "First-rollout timeout pattern differs",
            )
            require(
                all(
                    (np.asarray(v).shape == (42,)) if expected_done else v is None
                    for v in row["terminal_observation"]
                ),
                "Wrong pre-reset terminal observation schema",
            )
        require(
            sum(sum(row["timeout"]) for row in steps)
            == receipt["timeouts_per_arm"][arm]
            == 20,
            "Wrong first-rollout timeout count",
        )
        histories[arm] = {
            "initial_observation": headers[0]["initial_observation"],
            "records": steps,
            "canonical_fingerprint": receipt["canonical_fingerprint"],
            "parent_rng_before_rollout": receipt["parent_rng_before_rollout"],
            "actual_transitions": 1024,
        }
    require(
        compare_histories(histories["raw"], histories["bounded"])[
            "all_arrays_exactly_equal"
        ],
        "First-rollout actual streams differ",
    )
    return receipt


def check_model(
    model, teacher, seed, expected_coefficient, expected_steps, initial=False
):
    require(
        type(model) is RetentionPPO and type(model.policy) is BoundedCriticPolicy,
        "Wrong loaded learner/policy class",
    )
    p = model.policy
    require(
        model.num_timesteps == expected_steps,
        "Actual loaded checkpoint budget/selection mismatch",
    )
    require(
        model.retention_coefficient == expected_coefficient,
        "Actual checkpoint coefficient differs from frozen calibration",
    )
    for name, expected in {
        "seed": seed,
        "n_envs": 4,
        "n_steps": 256,
        "batch_size": 256,
        "n_epochs": 8,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.0,
        "target_kl": 0.01,
    }.items():
        require(getattr(model, name) == expected, f"Actual recipe differs: {name}")
    require(
        model.lr_schedule(1) == 3e-5 and model.clip_range(1) == 0.2,
        "Changed optimizer schedule",
    )
    require(
        p.pi_features_extractor.clip is None
        and p.vf_features_extractor.clip == 10
        and not p.share_features_extractor,
        "Changed fixed preprocessing",
    )
    require(
        isinstance(p.value_net[1], ValueBound) and p.value_net[1].bound == 100,
        "Changed critic bound",
    )
    require(
        p.net_arch == {"pi": [256, 256], "vf": [256, 256]}
        and p.activation_fn is torch.nn.ReLU,
        "Changed architecture",
    )
    require(
        all(torch.isfinite(v).all() for v in p.state_dict().values()),
        "Nonfinite policy state",
    )
    require(
        tensor_fingerprint(model.teacher) == tensor_fingerprint(teacher),
        "Checkpoint teacher differs from frozen original",
    )
    require(
        not model.teacher.training
        and not any(
            p.requires_grad or p.grad is not None for p in model.teacher.parameters()
        ),
        "Checkpoint teacher is trainable",
    )
    optimizer_ids = {
        id(v) for group in p.optimizer.param_groups for v in group["params"]
    }
    require(
        not optimizer_ids.intersection(id(v) for v in model.teacher.parameters()),
        "Teacher entered policy optimizer",
    )
    for extractor in [p.pi_features_extractor, p.vf_features_extractor]:
        require(
            torch.equal(extractor.mean, teacher.mean)
            and torch.equal(extractor.std, teacher.std),
            "Normalization differs from original teacher",
        )
    if initial:
        require(
            not p.optimizer.state
            and torch.equal(p.log_std, torch.full_like(p.log_std, -3)),
            "Initial optimizer/exploration differs",
        )
        for source, dest in [
            (teacher.net[0], p.mlp_extractor.policy_net[0]),
            (teacher.net[2], p.mlp_extractor.policy_net[2]),
            (teacher.net[4], p.action_net),
        ]:
            require(
                torch.equal(source.weight, dest.weight)
                and torch.equal(source.bias, dest.bias),
                "Initial actor differs from original teacher",
            )
    return {
        "num_timesteps": model.num_timesteps,
        "seed": model.seed,
        "coefficient": model.retention_coefficient,
        "teacher_fingerprint": tensor_fingerprint(model.teacher),
        "teacher_frozen": True,
        "optimizer_empty": not bool(p.optimizer.state),
        "policy_class": type(p).__name__,
        "critic_bound": 100,
    }


@isolated_rng
def verify_checkpoints(run, teacher_path):
    """No simulator access. All six final budgets must pass before this returns."""
    torch.set_num_threads(1)
    run = Path(run)
    p = read(run / "protocol.json")
    require(
        p["training_seeds"] == SEEDS and p["arms"] == ARMS,
        "Wrong declared six conditions",
    )
    require(
        p["config"]["total_timesteps"] == 262144
        and p["test_seeds"] == list(range(240000, 240050))
        and p["video_seeds"] == [240000, 240001],
        "Changed declared budget/test seeds",
    )
    require(sha256(teacher_path) == p["teacher_sha256"], "Original teacher changed")
    for name, expected in p["sources"].items():
        require(
            sha256(run / "launch_sources" / name) == expected
            and sha256(Path(__file__).with_name(name)) == expected,
            f"Changed frozen source:{name}",
        )
    receipt = verify_receipt(run, p)
    calibration, frozen = (
        read(run / "calibration.json"),
        read(run / "calibration_frozen.json"),
    )
    require(
        frozen["protocol_sha256"] == sha256(run / "protocol.json")
        and frozen["sha256"] == sha256(run / "calibration.json")
        and frozen["coefficient"] == calibration["coefficient"],
        "Calibration/protocol changed",
    )
    require(
        np.isfinite(frozen["coefficient"]) and 0.01 <= frozen["coefficient"] <= 10,
        "Invalid frozen coefficient",
    )
    require(
        calibration["actual_timesteps"] == 8192 and calibration["pilot_discarded"],
        "Wrong calibration budget/disposition",
    )
    teacher = load_teacher(teacher_path)
    tf = tensor_fingerprint(teacher)
    require(tf == calibration["teacher_fingerprint"], "Calibration teacher changed")
    conditions = {}
    for seed in SEEDS:
        conditions[str(seed)] = {}
        manifests = []
        for arm in ARMS:
            directory = run / f"seed{seed}" / arm
            manifest = read(directory / "initialization.json")
            manifests.append(manifest)
            require(
                manifest["optimizer_empty"] is True
                and manifest["teacher_fingerprint"] == tf
                and manifest["calibration_sha256"] == frozen["sha256"],
                "Initial manifest changed",
            )
            expected_coef = frozen["coefficient"] if arm == "anchored" else 0.0
            require(
                manifest["coefficient"] == expected_coef,
                "Initial coefficient manifest differs",
            )
            initial_path = directory / "initial.zip"
            require(
                sha256(initial_path) == manifest["checkpoint_sha256"],
                "Initial checkpoint hash differs",
            )
            initial = RetentionPPO.load(initial_path, device="cpu")
            entries = {
                "initial": {
                    "sha256": sha256(initial_path),
                    "recipe": check_model(
                        initial, teacher, seed, expected_coef, 0, initial=True
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
                "Selection differs from declared final-first ranking",
            )
            training = read(directory / "training.json")
            require(
                training["complete"] and training["actual_timesteps"] == 262144,
                "Missing final training completion",
            )
            for tag, steps in [("final", 262144), ("best", selected["timesteps"])]:
                path = directory / f"{tag}.zip"
                model = RetentionPPO.load(path, device="cpu")
                entries[tag] = {
                    "sha256": sha256(path),
                    "recipe": check_model(model, teacher, seed, expected_coef, steps),
                }
            conditions[str(seed)][arm] = entries
        require(
            manifests[0]["canonical_fingerprint"]
            == manifests[1]["canonical_fingerprint"]
            and manifests[0]["rng_fingerprint"] == manifests[1]["rng_fingerprint"]
            and len(manifests[0]["rng_fingerprint"]) == 64,
            "Paired initial parameters/parent RNG differ",
        )
        first_updates = [
            read(run / f"seed{seed}" / arm / "diagnostics.json")["updates"][0]
            for arm in ARMS
        ]
        require(
            all(row["timesteps"] == 1024 for row in first_updates),
            "Wrong first performance rollout size",
        )
        for field in [
            "values",
            "returns",
            "advantages",
            "buffer_rewards_after_bootstrap",
            "raw",
        ]:
            require(
                first_updates[0][field] == first_updates[1][field],
                f"First performance rollout summaries differ: seed{seed} {field}",
            )
    result = {
        "all_six_final_budgets_verified_before_evaluation": True,
        "all_three_first_performance_rollout_summaries_match": True,
        "teacher_sha256": p["teacher_sha256"],
        "teacher_fingerprint": tf,
        "protocol_sha256": sha256(run / "protocol.json"),
        "calibration_sha256": frozen["sha256"],
        "coefficient": frozen["coefficient"],
        "actual_checkpoints": conditions,
        "first_rollout_records_sha256": receipt["records_sha256"],
        "comparison_source_sha256": sha256(__file__),
    }
    save_json(run / "provenance.json", result)
    return result


def verify_video(full, video, seeds):
    """Compare exact passive records as well as every ordinary episode field."""
    expected_episodes = [e for e in full["episodes"] if e["seed"] in seeds]
    require(
        [e["seed"] for e in expected_episodes] == seeds,
        "Video seeds missing/out of order",
    )
    require(video["episodes"] == expected_episodes, "Video episode fields differ")
    expected_records = [r for r in full["step_records"] if r["seed"] in seeds]
    require(video["step_records"] == expected_records, "Video holding records differ")
    require(
        video["retention"] == retention_summary(expected_records, expected_episodes),
        "Video holding summary differs",
    )
    for seed in seeds:
        require(
            video["episode_retention"][str(seed)]
            == full["episode_retention"][str(seed)],
            "Video per-episode holding aggregate differs",
        )


@isolated_rng
def compare(run, teacher_path):
    """Run after all declared training conditions finish; no policy updates."""
    run = Path(run)
    # This entire guard completes before the first held-out environment is made.
    verify_checkpoints(run, teacher_path)
    p = read(run / "protocol.json")
    cfg = Config(**p["config"])
    teacher = load_teacher(teacher_path)
    summaries, video_manifest = {}, []

    def assess(name, policy, task, path, native):
        result = evaluate_retention(task, policy, teacher, p["test_seeds"], path)
        summaries[name] = result["summary"]
        if native:
            video_path = path.with_name(path.name.replace("test_", "video_", 1))
            video = evaluate_retention(
                task, policy, teacher, p["video_seeds"], video_path, video=True
            )
            verify_video(result, video, p["video_seeds"])
            for seed in p["video_seeds"]:
                file = video_path.with_name(f"{video_path.stem}_seed{seed}.mp4")
                require(file.stat().st_size > 0, "Empty native video")
                video_manifest.append(
                    {
                        "condition": name,
                        "seed": seed,
                        "file": str(file.relative_to(run)),
                        "sha256": sha256(file),
                        "all_episode_and_holding_records_match": True,
                    }
                )

    assess("teacher", teacher, cfg, run / "test_teacher.json", True)
    for seed in SEEDS:
        task = replace(cfg, seed=seed)
        for arm in ARMS:
            for tag in ["final", "best"]:
                path = run / f"seed{seed}" / f"test_{arm}_{tag}.json"
                model = RetentionPPO.load(
                    run / f"seed{seed}" / arm / f"{tag}.zip", device="cpu"
                )
                assess(f"seed{seed}_{arm}_{tag}", model, task, path, tag == "final")
    require(len(video_manifest) == 14, "Missing declared native videos")
    save_json(run / "comparison.json", summaries)
    save_json(run / "video_manifest.json", video_manifest)
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
