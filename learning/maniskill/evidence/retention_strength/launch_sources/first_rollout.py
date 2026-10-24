"""Verify paired stochastic simulator streams before the first optimizer update."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from .bc_initialized_ppo import rng_fingerprint
from .bc_ppo_policy import BCMeanGaussianPolicy, FixedNormalization, load_teacher
from .critic_experiment import canonical_fingerprint
from .critic_policy import BoundedCriticPolicy
from .experiment import Config, save_json, versions
from .rng import isolated_rng
from .seeded_env import seeded_training_env


class CaptureRollout(BaseCallback):
    """Copy environment rewards in the callback, before SB3 timeout correction."""

    def __init__(self):
        super().__init__()
        self.records = []

    def _on_training_start(self):
        self.initial_observation = self.model._last_obs.copy()

    def _on_step(self):
        self.records.append(
            {
                "step": len(self.records) + 1,
                "pre_observation": self.model._last_obs.copy().tolist(),
                "post_observation": self.locals["new_obs"].copy().tolist(),
                "gaussian_sample": self.locals["actions"].copy().tolist(),
                "executed_command": self.locals["clipped_actions"].copy().tolist(),
                "raw_reward": self.locals["rewards"].copy().tolist(),
                "done": self.locals["dones"].copy().tolist(),
                "terminal_observation": [
                    info["terminal_observation"].copy().tolist() if done else None
                    for done, info in zip(
                        self.locals["dones"], self.locals["infos"], strict=True
                    )
                ],
                "timeout": [
                    bool(info.get("TimeLimit.truncated", False))
                    for info in self.locals["infos"]
                ],
            }
        )
        return True


def build_initial(env, teacher, seed, bounded):
    """Existing BC-mean Gaussian recipe; no optimizer step is ever requested."""
    model = PPO(
        BoundedCriticPolicy if bounded else BCMeanGaussianPolicy,
        env,
        seed=seed,
        device="cpu",
        n_steps=256,
        batch_size=256,
        n_epochs=8,
        learning_rate=3e-5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0,
        target_kl=0.01,
        policy_kwargs={
            "net_arch": {"pi": [256, 256], "vf": [256, 256]},
            "activation_fn": torch.nn.ReLU,
            "log_std_init": -3,
            "features_extractor_class": FixedNormalization,
            "features_extractor_kwargs": {
                "mean": teacher.mean.tolist(),
                "std": teacher.std.tolist(),
            },
        },
    )
    with torch.no_grad():
        for source, target in [
            (teacher.net[0], model.policy.mlp_extractor.policy_net[0]),
            (teacher.net[2], model.policy.mlp_extractor.policy_net[2]),
            (teacher.net[4], model.policy.action_net),
        ]:
            target.weight.copy_(source.weight)
            target.bias.copy_(source.bias)
    return model


def capture(config, teacher, bounded):
    env = SubprocVecEnv(
        [seeded_training_env(config, rank) for rank in range(config.num_envs)],
        start_method="spawn",
    )
    try:
        model = build_initial(env, teacher, config.seed, bounded)
        before = canonical_fingerprint(model)
        capture_callback = CaptureRollout()
        _, callback = model._setup_learn(
            config.rollout_steps * config.num_envs, callback=capture_callback
        )
        callback.on_training_start({}, {})
        parent_rng = rng_fingerprint()
        if model.policy.optimizer.state:
            raise ValueError("First-rollout verifier requires an empty optimizer")
        completed = model.collect_rollouts(
            env, callback, model.rollout_buffer, n_rollout_steps=config.rollout_steps
        )
        callback.on_training_end()
        if not completed or len(capture_callback.records) != config.rollout_steps:
            raise ValueError("Incomplete pre-update rollout")
        if canonical_fingerprint(model) != before or model.policy.optimizer.state:
            raise ValueError("Model parameters or optimizer changed during capture")
        return {
            "initial_observation": capture_callback.initial_observation.tolist(),
            "records": capture_callback.records,
            "canonical_fingerprint": before,
            "parent_rng_before_rollout": parent_rng,
            "actual_transitions": model.num_timesteps,
            "optimizer_empty_after": True,
        }
    finally:
        env.close()


def compare_histories(raw, bounded, atol=1e-7):
    """Audit every captured field; value-head outputs deliberately are absent."""
    for field in [
        "canonical_fingerprint",
        "parent_rng_before_rollout",
        "actual_transitions",
    ]:
        if raw[field] != bounded[field]:
            raise ValueError(f"Unpaired first-rollout initialization: {field}")
    differences = {}
    pairs = [
        (
            "initial_observation",
            raw["initial_observation"],
            bounded["initial_observation"],
        )
    ]
    if len(raw["records"]) != len(bounded["records"]):
        raise ValueError("Different first-rollout lengths")
    for left, right in zip(raw["records"], bounded["records"], strict=True):
        for field in ["step", "done", "timeout"]:
            if left[field] != right[field]:
                raise ValueError(
                    f"First rollout differs: {field} at step {left['step']}"
                )
        for field in [
            "pre_observation",
            "post_observation",
            "gaussian_sample",
            "executed_command",
            "raw_reward",
        ]:
            pairs.append((field, left[field], right[field]))
        for a, b in zip(
            left["terminal_observation"], right["terminal_observation"], strict=True
        ):
            if (a is None) != (b is None):
                raise ValueError(
                    "Different pre-reset terminal observation availability"
                )
            if a is not None:
                pairs.append(("terminal_observation", a, b))
    bitwise = True
    for field, a, b in pairs:
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"Invalid first-rollout field: {field}")
        delta = float(np.max(np.abs(a - b)))
        differences[field] = max(differences.get(field, 0.0), delta)
        bitwise &= np.array_equal(a, b)
        if delta > atol:
            raise ValueError(
                f"First-rollout {field} differs by {delta}; allowed {atol}"
            )
    return {
        "maximum_absolute_difference": differences,
        "all_arrays_exactly_equal": bool(bitwise),
        "absolute_tolerance": atol,
    }


@isolated_rng
def verify_first_rollout(teacher_path, output, seed=42, workers=4):
    """Capture 256 steps in two separately spawned four-worker populations."""
    torch.set_num_threads(1)
    teacher_path, output = Path(teacher_path), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    teacher = load_teacher(teacher_path)
    config = Config(
        env_id="PickCube-v1", num_envs=workers, rollout_steps=256, seed=seed
    )
    histories = {
        arm: capture(config, teacher, bounded=arm == "bounded")
        for arm in ["raw", "bounded"]
    }
    comparison = compare_histories(histories["raw"], histories["bounded"])
    if histories["raw"]["actual_transitions"] != workers * config.rollout_steps:
        raise ValueError("Captured transition count differs from worker configuration")
    payload = "".join(
        json.dumps(
            {"arm": arm, "initial_observation": history["initial_observation"]},
            separators=(",", ":"),
        )
        + "\n"
        + "".join(
            json.dumps({"arm": arm, **row}, separators=(",", ":"), allow_nan=False)
            + "\n"
            for row in history["records"]
        )
        for arm, history in histories.items()
    ).encode()
    records_path = output / "first_rollout.jsonl.gz"
    records_path.write_bytes(gzip.compress(payload, mtime=0))
    receipt = {
        "passed": True,
        "scope": "Actual pre-update SB3 collection, no train()/learn() or optimizer step",
        "excluded": "Value predictions, advantages, returns and timeout-corrected buffer rewards may differ by design and are not paired fields",
        "teacher_sha256": hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": versions(),
        "seed": seed,
        "worker_seeds": list(range(seed, seed + workers)),
        "workers": workers,
        "stochastic_steps_per_worker": 256,
        "transitions_per_arm": histories["raw"]["actual_transitions"],
        "timeouts_per_arm": {
            arm: sum(sum(row["timeout"]) for row in history["records"])
            for arm, history in histories.items()
        },
        "canonical_fingerprint": histories["raw"]["canonical_fingerprint"],
        "parent_rng_before_rollout": histories["raw"]["parent_rng_before_rollout"],
        "optimizer_empty_after": all(
            h["optimizer_empty_after"] for h in histories.values()
        ),
        "actual_records": records_path.name,
        "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "uncompressed_records_sha256": hashlib.sha256(payload).hexdigest(),
        **comparison,
    }
    save_json(output / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    verify_first_rollout(args.teacher, args.output, args.seed, args.workers)


if __name__ == "__main__":
    main()
