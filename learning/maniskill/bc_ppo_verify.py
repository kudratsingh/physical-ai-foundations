"""Read-only checkpoint/protocol guards before and after paired BC/PPO comparison.

No environment is constructed, no training occurs and RNG streams are preserved.
Use --complete after comparison to require final budgets and all held-out results.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from .bc_initialized_ppo import fingerprint
from .bc_ppo_policy import BCMeanGaussianPolicy, load_teacher
from .experiment import save_json
from .metrics import summarize
from .rng import isolated_rng

RESULTS = [
    "teacher",
    "warm_initialized",
    "warm_final",
    "control_final",
    "warm_selected_secondary",
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def assert_episode_identity(teacher, initialized):
    """Full episode identity, including speeds/geometry; no aggregate-only match."""
    require(
        len(teacher) == len(initialized), "Teacher/initialized episode counts differ"
    )
    for index, (a, b) in enumerate(zip(teacher, initialized, strict=True)):
        if a != b:
            fields = sorted(key for key in set(a) | set(b) if a.get(key) != b.get(key))
            raise ValueError(
                f"Teacher/initialized episode identity differs at index{index}, seed{a.get('seed')}, fields{fields}"
            )
    return {"episodes": len(teacher), "full_episode_identity": True, "tolerance": 0}


def recipe(model, protocol, teacher, initial=False):
    """Inspect the actual loaded recipe, not merely copied sidecar settings."""
    config = protocol["config"]
    policy = model.policy
    require(
        isinstance(policy, BCMeanGaussianPolicy), "Wrong actual loaded policy class"
    )
    require(
        model.observation_space.shape == (42,) and model.action_space.shape == (8,),
        "Wrong loaded42/8 spaces",
    )
    require(
        np.all(model.action_space.low == -1) and np.all(model.action_space.high == 1),
        "Wrong action bounds",
    )
    require(
        not policy.share_features_extractor
        and policy.pi_features_extractor is not policy.vf_features_extractor,
        "Actor/critic features must be separate",
    )
    require(
        policy.pi_features_extractor.clip is None
        and policy.vf_features_extractor.clip == protocol["critic_feature_clip"],
        "Wrong actual actor/critic clipping",
    )
    for extractor in [policy.pi_features_extractor, policy.vf_features_extractor]:
        require(
            torch.equal(extractor.mean, teacher.mean)
            and torch.equal(extractor.std, teacher.std),
            "Loaded normalization differs from teacher",
        )
    require(
        all(torch.isfinite(value).all() for value in policy.state_dict().values()),
        "Nonfinite loaded policy tensors",
    )
    require(
        policy.net_arch == {"pi": [256, 256], "vf": [256, 256]},
        "Wrong actor/critic architecture",
    )
    require(policy.activation_fn is torch.nn.ReLU, "Wrong actor/critic activation")
    actual = {
        "policy_class": type(policy).__name__,
        "distribution_class": type(policy.action_dist).__name__,
        "timesteps": int(model.num_timesteps),
        "observation_shape": list(model.observation_space.shape),
        "action_shape": list(model.action_space.shape),
        "seed": int(model.seed),
        "num_envs": int(model.n_envs),
        "rollout_steps": int(model.n_steps),
        "batch_size": int(model.batch_size),
        "epochs": int(model.n_epochs),
        "learning_rate": float(model.lr_schedule(1.0)),
        "gamma": float(model.gamma),
        "gae_lambda": float(model.gae_lambda),
        "clip_range": float(model.clip_range(1.0)),
        "entropy_coefficient": float(model.ent_coef),
        "target_kl": float(model.target_kl),
        "actor_feature_clip": policy.pi_features_extractor.clip,
        "critic_feature_clip": policy.vf_features_extractor.clip,
        "shared_features": policy.share_features_extractor,
        "network_architecture": policy.net_arch,
        "activation": policy.activation_fn.__name__,
        "log_std": policy.log_std.detach().cpu().tolist(),
        "optimizer_class": type(policy.optimizer).__name__,
        "optimizer_empty": not bool(policy.optimizer.state),
        "optimizer_groups": [
            {key: value for key, value in group.items() if key != "params"}
            for group in policy.optimizer.param_groups
        ],
        "value_coefficient": float(model.vf_coef),
        "max_gradient_norm": float(model.max_grad_norm),
        "normalize_advantage": bool(model.normalize_advantage),
    }
    for actual_key, config_key in [
        ("seed", "seed"),
        ("num_envs", "num_envs"),
        ("rollout_steps", "rollout_steps"),
        ("batch_size", "batch_size"),
        ("epochs", "epochs"),
    ]:
        require(
            actual[actual_key] == config[config_key],
            f"Loaded recipe differs on{actual_key}",
        )
    for key, expected in [
        ("learning_rate", protocol["learning_rate"]),
        ("target_kl", protocol["target_kl"]),
        ("gamma", 0.99),
        ("gae_lambda", 0.95),
        ("clip_range", 0.2),
        ("entropy_coefficient", 0.0),
    ]:
        require(actual[key] == expected, f"Loaded recipe differs on{key}")
    require(
        type(policy.action_dist).__name__ == "DiagGaussianDistribution"
        and not policy.squash_output,
        "Need ordinary Gaussian sampling around bounded mean",
    )
    require(actual["optimizer_class"] == "Adam", "Wrong optimizer")
    if initial:
        require(model.num_timesteps == 0, "Initial model has already trained")
        require(actual["optimizer_empty"], "Actual initial optimizer is not empty")
        expected = torch.full_like(policy.log_std, protocol["initial_latent_log_std"])
        require(torch.equal(policy.log_std, expected), "Wrong initial exploration std")
    return actual


@isolated_rng
def verify(run, teacher_path, complete=False):
    run, teacher_path = Path(run), Path(teacher_path)
    protocol = json.loads((run / "protocol.json").read_text())
    require(
        sha256(teacher_path) == protocol["teacher_sha256"],
        "Current teacher SHA differs from protocol",
    )
    source_hashes = {}
    for name, expected in protocol["sources"].items():
        archive, current = run / "launch_sources" / name, Path(__file__).with_name(name)
        if (run / "launch_sources").exists():
            require(
                sha256(archive) == expected, f"Archived launch source mismatch: {name}"
            )
        require(sha256(current) == expected, f"Current load source mismatch: {name}")
        source_hashes[name] = {
            "archive_sha256": expected if (run / "launch_sources").exists() else None,
            "current_sha256": expected,
        }
    preflight = json.loads((run / "preflight.json").read_text())
    teacher = load_teacher(teacher_path)
    entries, manifests, loaded = {}, {}, {}
    for arm in ["warm", "control"]:
        directory = run / arm
        manifest = json.loads((directory / "initialization.json").read_text())
        path = directory / "initial.zip"
        require(
            sha256(path) == manifest["checkpoint_sha256"],
            f"Actual{arm} initial checkpoint hash mismatch",
        )
        model = PPO.load(path, device="cpu")
        actor, nonactor = fingerprint(model, True), fingerprint(model, False)
        expected_actor = preflight[
            "warm_actor_fingerprint" if arm == "warm" else "random_actor_fingerprint"
        ]
        require(
            actor == manifest["actor_fingerprint"] == expected_actor,
            f"{arm} actual actor fingerprint mismatch",
        )
        require(
            nonactor
            == manifest["non_actor_fingerprint"]
            == preflight["non_actor_fingerprint"],
            f"{arm} actual nonactor fingerprint mismatch",
        )
        require(
            manifest["optimizer_empty"] is True,
            f"{arm} manifest optimizer is not fresh",
        )
        require(
            isinstance(manifest["rng_fingerprint"], str)
            and len(manifest["rng_fingerprint"]) == 64,
            f"{arm} invalid saved RNG fingerprint",
        )
        loaded[arm], manifests[arm] = model, manifest
        entries[arm] = {
            "initial": {
                "checkpoint_sha256": sha256(path),
                "actor_fingerprint": actor,
                "non_actor_fingerprint": nonactor,
                "recipe": recipe(model, protocol, teacher, initial=True),
            }
        }
    require(
        manifests["warm"]["rng_fingerprint"] == manifests["control"]["rng_fingerprint"],
        "Paired saved pre-learning RNG fingerprints differ",
    )
    require(
        entries["warm"]["initial"]["non_actor_fingerprint"]
        == entries["control"]["initial"]["non_actor_fingerprint"],
        "Actual paired nonactor tensors differ",
    )
    require(
        entries["warm"]["initial"]["actor_fingerprint"]
        != entries["control"]["initial"]["actor_fingerprint"],
        "Treatment/control actors are identical",
    )
    # Independently verify the actual saved warm actor still contains teacher tensors.
    warm = loaded["warm"].policy
    for source, dest in [
        (teacher.net[0], warm.mlp_extractor.policy_net[0]),
        (teacher.net[2], warm.mlp_extractor.policy_net[2]),
        (teacher.net[4], warm.action_net),
    ]:
        require(
            torch.equal(source.weight, dest.weight)
            and torch.equal(source.bias, dest.bias),
            "Actual warm initial actor differs from BC teacher",
        )
    final_presence = {}
    for arm in ["warm", "control"]:
        path, training_path = run / arm / "final.zip", run / arm / "training.json"
        final_presence[arm] = path.exists()
        if complete:
            require(
                path.exists() and training_path.exists(),
                f"Missing completed{arm} final checkpoint/training record",
            )
        if path.exists():
            model = PPO.load(path, device="cpu")
            actual = recipe(model, protocol, teacher)
            require(
                model.num_timesteps == protocol["config"]["total_timesteps"],
                f"Actual{arm} final budget mismatch",
            )
            require(training_path.exists(), f"Missing{arm} final training record")
            training = json.loads(training_path.read_text())
            require(
                training["complete"] is True
                and training["actual_timesteps"] == model.num_timesteps,
                f"{arm} training record differs from actual final model",
            )
            entries[arm]["final"] = {
                "checkpoint_sha256": sha256(path),
                "recipe": actual,
                "training": training,
            }
        elif training_path.exists():
            require(
                not json.loads(training_path.read_text()).get("complete"),
                f"{arm} claims completion without a final model",
            )
    test_entries = {}
    for name in RESULTS:
        path = run / f"test_{name}.json"
        if complete:
            require(path.exists(), f"Missing post-comparison result:{name}")
        if path.exists():
            data = json.loads(path.read_text())
            require(
                data["config"] == protocol["config"],
                f"{name} result task/config differs from protocol",
            )
            require(
                [episode["seed"] for episode in data["episodes"]]
                == protocol["test_seeds"],
                f"{name} result seeds differ from protocol",
            )
            require(
                all(episode["steps"] == 50 for episode in data["episodes"]),
                f"{name} result horizon changed",
            )
            require(
                summarize(data["episodes"]) == data["summary"],
                f"{name} result summary does not match episodes",
            )
            test_entries[name] = data
    identity = None
    if "teacher" in test_entries and "warm_initialized" in test_entries:
        identity = assert_episode_identity(
            test_entries["teacher"]["episodes"],
            test_entries["warm_initialized"]["episodes"],
        )
    if complete:
        selected_path = run / "warm/best.zip"
        selected_point = json.loads((run / "warm/selected.json").read_text())
        selected_model = PPO.load(selected_path, device="cpu")
        require(
            selected_model.num_timesteps == selected_point["timesteps"],
            "Actual selected warm checkpoint differs from selection record",
        )
        require(
            0 <= selected_model.num_timesteps <= protocol["config"]["total_timesteps"],
            "Selected checkpoint lies outside training budget",
        )
        entries["warm"]["selected_secondary"] = {
            "checkpoint_sha256": sha256(selected_path),
            "recipe": recipe(selected_model, protocol, teacher),
            "selection": selected_point,
        }
        comparison = json.loads((run / "comparison.json").read_text())
        require(
            comparison
            == {name: data["summary"] for name, data in test_entries.items()},
            "Comparison summary differs from actual result files",
        )
    provenance = {
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "complete_required": complete,
        "finals_present": final_presence,
        "all_final_budgets_verified": all(final_presence.values()),
        "source_hashes": source_hashes,
        "launch_archive_present": (run / "launch_sources").exists(),
        "protocol_sha256": sha256(run / "protocol.json"),
        "teacher_sha256": sha256(teacher_path),
        "preflight_sha256": sha256(run / "preflight.json"),
        "actual_checkpoints": entries,
        "saved_prelearning_rng_fingerprint": manifests["warm"]["rng_fingerprint"],
        "rng_note": "Saved training-start fingerprints match; current RNG streams are preserved during verification. Historical RNG state is not reconstructed from checkpoints.",
        "actual_initial_actor_equals_teacher": True,
        "actual_initial_nonactor_pair_matches": True,
        "actual_initial_optimizers_empty": True,
        "teacher_initialized_test_identity": identity,
        "result_sha256": {
            name: sha256(run / f"test_{name}.json") for name in test_entries
        },
    }
    save_json(run / "provenance.json", provenance)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument(
        "--complete",
        action="store_true",
        help="Require both completed finals and all compared held-out result files",
    )
    args = parser.parse_args()
    result = verify(args.run, args.teacher, args.complete)
    print(
        json.dumps(
            {
                "finals_present": result["finals_present"],
                "complete_required": args.complete,
                "teacher_initialized_test_identity": result[
                    "teacher_initialized_test_identity"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
