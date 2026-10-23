"""Verify actual paired critic checkpoints before and after held-out evaluation."""

import argparse
import json
from pathlib import Path

import torch

from .bc_ppo_policy import BCMeanGaussianPolicy, load_teacher
from .critic_diagnostics import DiagnosticPPO
from .critic_experiment import canonical_fingerprint
from .critic_policy import BoundedCriticPolicy, ValueBound
from .experiment import save_json
from .metrics import summarize
from .normalized_ppo import sha256
from .rng import isolated_rng


def require(condition, message):
    if not condition:
        raise ValueError(message)


def recipe(model, arm, config, initial=False):
    p = model.policy
    require(
        type(p) is (BoundedCriticPolicy if arm == "bounded" else BCMeanGaussianPolicy),
        "Wrong loaded policy class",
    )
    require(
        p.pi_features_extractor.clip is None and p.vf_features_extractor.clip == 10.0,
        "Changed feature clipping",
    )
    require(not p.share_features_extractor, "Unexpected shared features")
    if arm == "bounded":
        require(
            isinstance(p.value_net[1], ValueBound) and p.value_net[1].bound == 100.0,
            "Wrong actual value bound",
        )
    for key, value in [
        ("seed", 42),
        ("n_envs", 4),
        ("n_steps", 256),
        ("batch_size", 256),
        ("n_epochs", 8),
        ("gamma", 0.99),
        ("gae_lambda", 0.95),
        ("target_kl", 0.01),
        ("ent_coef", 0.0),
    ]:
        require(getattr(model, key) == value, f"Actual recipe mismatch:{key}")
    require(
        model.lr_schedule(1.0) == 3e-5 and model.clip_range(1.0) == 0.2,
        "Changed optimizer schedule",
    )
    require(
        p.net_arch == {"pi": [256, 256], "vf": [256, 256]}
        and p.activation_fn is torch.nn.ReLU,
        "Wrong actual architecture",
    )
    require(
        all(torch.isfinite(v).all() for v in p.state_dict().values()),
        "Nonfinite policy tensors",
    )
    if initial:
        require(
            model.num_timesteps == 0 and not p.optimizer.state,
            "Initial optimizer is not fresh",
        )
        require(
            torch.equal(p.log_std, torch.full_like(p.log_std, -3.0)),
            "Changed initial exploration",
        )
    return {
        "policy_class": type(p).__name__,
        "timesteps": model.num_timesteps,
        "seed": model.seed,
        "num_envs": model.n_envs,
        "gamma": model.gamma,
        "gae_lambda": model.gae_lambda,
        "target_kl": model.target_kl,
        "learning_rate": model.lr_schedule(1.0),
        "value_bound": 100.0 if arm == "bounded" else None,
        "optimizer_empty": not bool(p.optimizer.state),
    }


@isolated_rng
def verify(run, teacher_path, complete=False):
    run = Path(run)
    protocol = json.loads((run / "protocol.json").read_text())
    config = protocol["config"]
    require(sha256(teacher_path) == protocol["teacher_sha256"], "Teacher hash changed")
    for n, h in protocol["sources"].items():
        require(
            sha256(run / "launch_sources" / n) == h
            and sha256(Path(__file__).with_name(n)) == h,
            f"Frozen source changed:{n}",
        )
    teacher = load_teacher(teacher_path)
    pre = json.loads((run / "preflight.json").read_text())
    pre_episodes = {
        n: json.loads((run / f"preflight_{n}.json").read_text())["episodes"]
        for n in ["teacher", "raw", "bounded"]
    }
    require(
        pre_episodes["teacher"] == pre_episodes["raw"] == pre_episodes["bounded"],
        "Preflight full episodes differ",
    )
    entries = {}
    initials = []
    rng = []
    for arm in ["raw", "bounded"]:
        directory = run / arm
        manifest = json.loads((directory / "initialization.json").read_text())
        initial = DiagnosticPPO.load(directory / "initial.zip", device="cpu")
        require(
            sha256(directory / "initial.zip") == manifest["checkpoint_sha256"],
            "Initial hash changed",
        )
        require(
            canonical_fingerprint(initial)
            == manifest["canonical_fingerprint"]
            == pre["canonical_fingerprint"],
            "Initial canonical tensors changed",
        )
        require(manifest["optimizer_empty"] is True, "Manifest optimizer not fresh")
        rng.append(manifest["rng_fingerprint"])
        initials.append(canonical_fingerprint(initial))
        p = initial.policy
        for extractor in [p.pi_features_extractor, p.vf_features_extractor]:
            require(
                torch.equal(extractor.mean, teacher.mean)
                and torch.equal(extractor.std, teacher.std),
                "Initial normalization differs from teacher",
            )
        for a, b in [
            (teacher.net[0], p.mlp_extractor.policy_net[0]),
            (teacher.net[2], p.mlp_extractor.policy_net[2]),
            (teacher.net[4], p.action_net),
        ]:
            require(
                torch.equal(a.weight, b.weight) and torch.equal(a.bias, b.bias),
                "Actual initial actor differs from teacher",
            )
        entries[arm] = {
            "initial": {
                "sha256": sha256(directory / "initial.zip"),
                "recipe": recipe(initial, arm, config, True),
            }
        }
        for tag in ["final", "best"]:
            model = DiagnosticPPO.load(directory / f"{tag}.zip", device="cpu")
            expected = (
                config["total_timesteps"]
                if tag == "final"
                else json.loads((directory / "selected.json").read_text())["timesteps"]
            )
            require(
                model.num_timesteps == expected,
                "Actual loaded budget/selection differs",
            )
            if tag == "final":
                record = json.loads((directory / "training.json").read_text())
                require(
                    record["complete"] and record["actual_timesteps"] == expected,
                    "Incomplete training sidecar",
                )
            entries[arm][tag] = {
                "sha256": sha256(directory / f"{tag}.zip"),
                "recipe": recipe(model, arm, config),
            }
    require(
        len(set(rng)) == 1 and len(rng[0]) == 64 and len(set(initials)) == 1,
        "Paired initial state/RNG mismatch",
    )
    summaries = {}
    if complete:
        for name in [
            "teacher",
            "raw_final",
            "raw_best",
            "bounded_final",
            "bounded_best",
        ]:
            data = json.loads((run / f"test_{name}.json").read_text())
            require(
                data["config"] == config
                and [e["seed"] for e in data["episodes"]] == protocol["test_seeds"],
                "Changed test task/seeds",
            )
            require(all(e["steps"] == 50 for e in data["episodes"]), "Changed horizon")
            summaries[name] = summarize(data["episodes"])
            require(summaries[name] == data["summary"], "Episode summary mismatch")
        require(
            summaries == json.loads((run / "comparison.json").read_text()),
            "Comparison mismatch",
        )
    result = {
        "complete": complete,
        "teacher_sha256": sha256(teacher_path),
        "protocol_sha256": sha256(run / "protocol.json"),
        "actual_initial_actors_equal_teacher": True,
        "actual_initial_parameters_match": True,
        "paired_saved_rng": rng[0],
        "preflight_full_episode_identity": True,
        "actual_checkpoints": entries,
        "all_final_budgets_verified": True,
    }
    save_json(run / "verified.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--complete", action="store_true")
    a = p.parse_args()
    verify(a.run, a.teacher, a.complete)
    print("Actual checkpoint, source, pairing and budget guards passed")


if __name__ == "__main__":
    main()
