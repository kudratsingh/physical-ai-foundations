"""One-change ablation: tanh-transformed Gaussian PPO on raw PickCube state.

Uses SB3's squashed Gaussian with its change-of-variables log probability.
This is a distribution change during training, never a post-hoc policy repair.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .experiment import Config, env_thunk, evaluate, save_json, versions
from .normalized_ppo import preserve_rng, sha256
from .selection import holding_score


class TanhGaussianPolicy(ActorCriticPolicy):
    """SB3 actor/critic and optimizer, with only the action distribution replaced.

    ActorCriticPolicy's squash_output flag is reserved for gSDE. This class leaves
    it False and transforms inside the distribution instead. SB3's outer clip is
    consequently redundant for our explicitly checked unit action bounds.
    """

    def _build(self, lr_schedule):
        if (
            self.use_sde
            or self.action_space.shape != (8,)
            or not (
                np.all(self.action_space.low == -1)
                and np.all(self.action_space.high == 1)
            )
        ):
            raise ValueError(
                "Requires eight unit-bounded actions and ordinary Gaussian noise"
            )
        self.action_dist = SquashedDiagGaussianDistribution(8)
        super()._build(lr_schedule)


TASK = Config(
    env_id="PickCube-v1",
    total_timesteps=1048576,
    num_envs=4,
    validation_interval=131072,
    validation_seed_start=100000,
    validation_episodes=10,
    test_seed_start=110000,
    test_episodes=50,
)


def policy_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.policy.state_dict().items()):
        array = value.detach().cpu().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def build_model(env, config, bounded=True):
    return PPO(
        TanhGaussianPolicy if bounded else "MlpPolicy",
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
        policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
    )


def train(config, output):
    torch.set_num_threads(1)
    if config.total_timesteps % (config.num_envs * config.rollout_steps):
        raise ValueError("Budget must contain whole PPO rollout batches")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    constructors = [env_thunk(config) for _ in range(config.num_envs)]
    env = (
        DummyVecEnv(constructors)
        if config.num_envs == 1
        else SubprocVecEnv(constructors, start_method="spawn")
    )
    env.seed(config.seed)

    class Validation(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_eval, self.points, self.best = (
                config.validation_interval,
                [],
                None,
            )

        def _on_step(self):
            if self.num_timesteps < self.next_eval:
                return True
            with preserve_rng():
                result = evaluate(
                    config,
                    self.model,
                    range(
                        config.validation_seed_start,
                        config.validation_seed_start + config.validation_episodes,
                    ),
                )
            point = dict(timesteps=self.num_timesteps, **result["summary"])
            self.points.append(point)
            score = holding_score(point)
            if self.best is None or score > self.best:
                self.best = score
                self.model.save(output / "best")
                save_json(output / "selected.json", point)
            save_json(output / "validation.json", self.points)
            self.next_eval += config.validation_interval
            return True

    try:
        # Both constructors reseed identically. Replacing the distribution must
        # leave actor/critic tensors and initial log_std exactly unchanged.
        raw = build_model(env, config, bounded=False)
        raw_initial = policy_fingerprint(raw)
        del raw
        model = build_model(env, config)
        if policy_fingerprint(model) != raw_initial:
            raise ValueError("Distribution treatment changed initial model parameters")
        save_json(
            output / "config.json",
            dict(
                **asdict(config),
                versions=versions(),
                source_sha256=sha256(__file__),
                experiment_source_sha256=sha256(
                    Path(__file__).with_name("experiment.py")
                ),
                selection_source_sha256=sha256(
                    Path(__file__).with_name("selection.py")
                ),
                treatment="tanh Gaussian action distribution; raw observations; no other recipe changes",
                raw_and_bounded_initial_policy_sha256=raw_initial,
                selection_order=[
                    "success_at_end_rate",
                    "success_once_rate",
                    "return_mean",
                ],
                primary="fixed-budget final versus original raw final on paired fresh resets",
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
                video_seeds=list(
                    range(config.test_seed_start, config.test_seed_start + 2)
                ),
            ),
        )
        started = time.monotonic()
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=Validation(),
            log_interval=16,
        )
        model.save(output / "final")
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
    from .normalization_comparison import policy_contract

    output, raw_checkpoint = Path(output), Path(raw_checkpoint)
    saved = json.loads((output / "config.json").read_text())
    config = Config(**{key: saved[key] for key in Config.__dataclass_fields__})
    baseline = json.loads((raw_checkpoint.parent / "config.json").read_text())
    for key in Config.__dataclass_fields__:
        if not key.startswith(("validation_", "test_")) and baseline[key] != getattr(
            config, key
        ):
            raise ValueError(f"Baseline training/task setting differs: {key}")
    policies = {
        "raw_final": PPO.load(raw_checkpoint, device="cpu"),
        "bounded_final": PPO.load(output / "final.zip", device="cpu"),
    }
    contracts = {key: policy_contract(model) for key, model in policies.items()}
    primary = {}
    for name, policy in policies.items():
        primary[name] = evaluate(
            config, policy, saved["test_seeds"], output / f"test_{name}.json"
        )
        video = evaluate(
            config,
            policy,
            saved["video_seeds"],
            output / f"video_{name}.json",
            traces=True,
            video=True,
        )
        for actual, expected in zip(
            video["episodes"], primary[name]["episodes"], strict=False
        ):
            if {k: v for k, v in actual.items() if k != "trajectory"} != {
                k: v for k, v in expected.items() if k != "trajectory"
            }:
                raise ValueError(
                    "Native video differs from the predeclared test rollout"
                )
    selected = PPO.load(output / "best.zip", device="cpu")
    secondary = evaluate(
        config,
        selected,
        saved["test_seeds"],
        output / "test_bounded_selected_secondary.json",
    )
    save_json(
        output / "comparison.json",
        {
            **{key: value["summary"] for key, value in primary.items()},
            "bounded_selected_secondary": secondary["summary"],
        },
    )
    save_json(
        output / "provenance.json",
        dict(
            raw_checkpoint_sha256=sha256(raw_checkpoint),
            bounded_checkpoint_sha256=sha256(output / "final.zip"),
            loaded_policy_contracts=contracts,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "compare"])
    parser.add_argument("--output", required=True, type=Path)
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
