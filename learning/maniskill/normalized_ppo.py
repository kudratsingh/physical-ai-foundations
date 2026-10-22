"""Controlled PickCube PPO observation-normalization experiment.

The raw baseline is an existing fixed-budget final checkpoint. This treatment
uses exactly the original PPO hyperparameters and adds only SB3 VecNormalize
observations (rewards remain in original task units). No extra warmup samples.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from .experiment import Config, env_thunk, evaluate, save_json, versions
from .selection import holding_score
from .rng import preserve_rng

TASK = Config(
    env_id="PickCube-v1",
    total_timesteps=1048576,
    num_envs=4,
    validation_interval=131072,
    validation_seed_start=80000,
    validation_episodes=10,
    test_seed_start=90000,
    test_episodes=50,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_normalized_env(config, instrument=False):
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        SubprocVecEnv,
        VecNormalize,
    )

    from .training_metrics import training_env

    constructors = [
        training_env(config) if instrument else env_thunk(config)
        for _ in range(config.num_envs)
    ]
    raw = (
        DummyVecEnv(constructors)
        if config.num_envs == 1
        else SubprocVecEnv(constructors, start_method="spawn")
    )
    raw.seed(config.seed)
    return VecNormalize(raw, norm_obs=True, norm_reward=False, clip_obs=10.0)


def stats_json(normalizer):
    return dict(
        count=float(normalizer.obs_rms.count),
        mean=normalizer.obs_rms.mean.tolist(),
        variance=normalizer.obs_rms.var.tolist(),
        training=normalizer.training,
        norm_obs=normalizer.norm_obs,
        norm_reward=normalizer.norm_reward,
        clip_obs=normalizer.clip_obs,
        epsilon=normalizer.epsilon,
    )


def save_snapshot(model, normalizer, directory):
    """A pair plus checksums prevents loading a model with mismatched statistics."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    model.save(directory / "model")
    normalizer.save(directory / "vecnormalize.pkl")
    save_json(directory / "statistics.json", stats_json(normalizer))
    save_json(
        directory / "snapshot.json",
        dict(
            timesteps=model.num_timesteps,
            model_sha256=sha256(directory / "model.zip"),
            statistics_sha256=sha256(directory / "vecnormalize.pkl"),
        ),
    )


class FrozenPolicy:
    """Normalize raw evaluator observations without changing training statistics."""

    policy_label = "PPO observation-normalized, frozen statistics"

    def __init__(self, model, normalizer):
        self.model, self.normalizer = model, normalizer
        self.normalizer.training = False
        self.normalizer.norm_reward = False

    @classmethod
    def load(cls, directory, config=TASK):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        directory = Path(directory)
        manifest = json.loads((directory / "snapshot.json").read_text())
        if (
            sha256(directory / "model.zip") != manifest["model_sha256"]
            or sha256(directory / "vecnormalize.pkl") != manifest["statistics_sha256"]
        ):
            raise ValueError("Model/statistics snapshot checksum mismatch")
        raw = DummyVecEnv([env_thunk(config)])
        try:
            normalizer = VecNormalize.load(directory / "vecnormalize.pkl", raw)
            model = PPO.load(directory / "model.zip", device="cpu")
        except Exception:
            raw.close()
            raise
        return cls(model, normalizer)

    def predict(self, observation, deterministic=True):
        return self.model.predict(
            self.normalizer.normalize_obs(observation), deterministic=deterministic
        )

    def close(self):
        self.normalizer.close()


def train(config, output):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from .training_metrics import TrainingMetrics

    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if config.total_timesteps % (config.num_envs * config.rollout_steps):
        raise ValueError("Budget must contain whole rollout batches")
    save_json(
        output / "config.json",
        dict(
            **asdict(config),
            versions=versions(),
            created_utc=datetime.now(timezone.utc).isoformat(),
            source_sha256=sha256(__file__),
            selection_order=["success_at_end_rate", "success_once_rate", "return_mean"],
            treatment=dict(
                norm_obs=True,
                norm_reward=False,
                clip_obs=10.0,
                extra_warmup_transitions=0,
            ),
            validation_seeds=list(
                range(
                    config.validation_seed_start,
                    config.validation_seed_start + config.validation_episodes,
                )
            ),
            test_seeds=list(
                range(
                    config.test_seed_start,
                    config.test_seed_start + config.test_episodes,
                )
            ),
            video_seeds=list(range(config.test_seed_start, config.test_seed_start + 3)),
            primary="Raw final vs normalized final at equal 1048576 transitions; no validation selection",
            secondary="Normalized validation-selected snapshot; not a matched selection comparison",
        ),
    )
    env = make_normalized_env(config, instrument=True)

    class Validation(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_eval = config.validation_interval
            self.points, self.best_score = [], None

        def _on_step(self):
            if self.num_timesteps < self.next_eval:
                return True
            snapshot = output / f"step_{self.num_timesteps}"
            save_snapshot(self.model, env, snapshot)
            before = stats_json(env)
            # PPO.load reconstructs a seeded model and resets global RNGs.
            # Preserve the learner's streams across the ENTIRE validation phase.
            with preserve_rng():
                policy = FrozenPolicy.load(snapshot, config)
                try:
                    result = evaluate(
                        config,
                        policy,
                        list(
                            range(
                                config.validation_seed_start,
                                config.validation_seed_start
                                + config.validation_episodes,
                            )
                        ),
                    )
                finally:
                    policy.close()
            if stats_json(env) != before:
                raise RuntimeError("Validation changed training observation statistics")
            point = dict(
                timesteps=self.num_timesteps,
                **result["summary"],
                stats_count=float(env.obs_rms.count),
            )
            self.points.append(point)
            score = holding_score(point)
            if self.best_score is None or score > self.best_score:
                self.best_score = score
                save_json(
                    output / "selected.json", dict(snapshot=snapshot.name, **point)
                )
            save_json(output / "validation.json", self.points)
            self.next_eval += config.validation_interval
            return True

    try:
        model = PPO(
            "MlpPolicy",
            env,
            seed=config.seed,
            device="cpu",
            verbose=1,
            n_steps=config.rollout_steps,
            batch_size=config.batch_size,
            n_epochs=config.epochs,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            tensorboard_log=str(output / "tensorboard"),
            policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        )
        started = time.monotonic()
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=CallbackList(
                [TrainingMetrics(output / "training_metrics.json"), Validation()]
            ),
            log_interval=16,
        )
        save_snapshot(model, env, output / "final")
        save_json(
            output / "training.json",
            dict(
                actual_timesteps=model.num_timesteps,
                elapsed_seconds=time.monotonic() - started,
                complete=True,
            ),
        )
    finally:
        env.close()


def compare(output, raw_checkpoint):
    """Evaluate both fixed-budget checkpoints once on the predeclared fresh seeds."""
    from stable_baselines3 import PPO
    from .metrics import failure_counts

    output, raw_checkpoint = Path(output), Path(raw_checkpoint)
    saved = json.loads((output / "config.json").read_text())
    config = Config(**{k: saved[k] for k in Config.__dataclass_fields__})
    training = json.loads((output / "training.json").read_text())
    raw_config = json.loads((raw_checkpoint.parent / "config.json").read_text())
    if not training["complete"] or training["actual_timesteps"] != 1048576:
        raise ValueError("Complete the fixed-budget treatment before final evaluation")
    for key in Config.__dataclass_fields__:
        if key.startswith("validation_") or key.startswith("test_"):
            continue
        if raw_config[key] != getattr(config, key):
            raise ValueError(f"Raw baseline differs on {key}")
    if raw_checkpoint.name != "final.zip":
        raise ValueError("Primary raw baseline must be the fixed-budget final.zip")
    seeds = list(
        range(config.test_seed_start, config.test_seed_start + config.test_episodes)
    )
    policies = {
        "raw_final": PPO.load(raw_checkpoint, device="cpu"),
        "normalized_final": FrozenPolicy.load(output / "final", config),
    }
    selected = json.loads((output / "selected.json").read_text())
    policies["normalized_selected_secondary"] = FrozenPolicy.load(
        output / selected["snapshot"], config
    )
    try:
        results = {}
        for name, policy in policies.items():
            results[name] = evaluate(
                config, policy, seeds, output / f"test_{name}.json"
            )
            if name != "normalized_selected_secondary":
                videos = evaluate(
                    config,
                    policy,
                    seeds[:3],
                    output / f"video_{name}.json",
                    traces=True,
                    video=True,
                )
                for video_episode, test_episode in zip(
                    videos["episodes"], results[name]["episodes"]
                ):
                    if {
                        k: v for k, v in video_episode.items() if k != "trajectory"
                    } != {k: v for k, v in test_episode.items() if k != "trajectory"}:
                        raise RuntimeError("Video replay differs from test rollout")
        save_json(
            output / "comparison.json", {k: v["summary"] for k, v in results.items()}
        )
        save_json(
            output / "failures.json",
            {k: failure_counts(v["episodes"]) for k, v in results.items()},
        )
        save_json(
            output / "baseline.json",
            dict(
                checkpoint_sha256=sha256(raw_checkpoint),
                config=raw_config,
                training=json.loads(
                    (raw_checkpoint.parent / "training.json").read_text()
                ),
                primary_checkpoint="final.zip",
                never_selected_on_new_test_seeds=True,
            ),
        )
    finally:
        for policy in policies.values():
            if isinstance(policy, FrozenPolicy):
                policy.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "compare"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-checkpoint", type=Path)
    args = parser.parse_args()
    if args.command == "train":
        train(TASK, args.output)
    elif args.raw_checkpoint:
        compare(args.output, args.raw_checkpoint)
    else:
        parser.error("compare requires --raw-checkpoint")


if __name__ == "__main__":
    main()
