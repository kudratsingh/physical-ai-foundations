"""Controlled PPO learning-rate forks from an identical model and optimizer.

This is a checkpoint continuation with fresh, identically seeded environments,
not a bit-exact process resume: simulator and RNG states are not in SB3 files.
"""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from .bc_data import sha256_file
from .experiment import Config, env_thunk, evaluate, save_json, versions

VALIDATION_SEEDS = list(range(60000, 60010))
TEST_SEEDS = list(range(70000, 70050))
ADDITIONAL_STEPS = 524288


def validation_score(summary):
    """Prioritize sustained success for this placement/holding experiment."""
    return (
        summary["success_at_end_rate"],
        summary["success_once_rate"],
        summary["return_mean"],
    )


def continue_training(
    checkpoint,
    directory,
    learning_rate,
    steps=ADDITIONAL_STEPS,
    num_envs=4,
    interval=131072,
    validation_seeds=None,
):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    if learning_rate <= 0 or steps <= 0 or num_envs < 1 or interval < 1:
        raise ValueError(
            "Positive learning rate, budget, workers and interval required"
        )
    validation_seeds = (
        VALIDATION_SEEDS if validation_seeds is None else list(validation_seeds)
    )
    if not validation_seeds or len(set(validation_seeds)) != len(validation_seeds):
        raise ValueError("Validation seeds must be nonempty and unique")
    if set(validation_seeds) & set(TEST_SEEDS):
        raise ValueError("Validation and final-test seeds must be disjoint")
    source = json.loads(checkpoint.with_name("config.json").read_text())
    task = Config(**{key: source[key] for key in asdict(Config()) if key in source})
    if task.env_id != "PickCube-v1":
        raise ValueError("This continuation protocol requires PickCube-v1")
    task = replace(
        task,
        num_envs=num_envs,
        total_timesteps=steps,
        validation_seed_start=validation_seeds[0],
        validation_episodes=len(validation_seeds),
        validation_interval=interval,
        test_seed_start=TEST_SEEDS[0],
        test_episodes=len(TEST_SEEDS),
    )
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    environments = [env_thunk(task) for _ in range(num_envs)]
    env = (
        DummyVecEnv(environments)
        if num_envs == 1
        else SubprocVecEnv(environments, start_method="spawn")
    )
    try:
        # SB3 recreates its constant LR schedule from this override. Its loaded
        # Adam param groups are updated from that schedule before optimization.
        model = PPO.load(checkpoint, env=env, device="cpu", learning_rate=learning_rate)
        model.tensorboard_log = str(directory / "tensorboard")
        model.set_random_seed(task.seed)
        start_steps = model.num_timesteps
        initial_parameters = model.get_parameters()
        policy_digest = hashlib.sha256()
        for name, value in sorted(initial_parameters["policy"].items()):
            policy_digest.update(name.encode())
            policy_digest.update(value.cpu().numpy().tobytes())
        # Save before any update to audit equality of actor, critic, log std and Adam state.
        torch.save(initial_parameters, directory / "initial_parameters.pt")
        save_json(
            directory / "config.json",
            dict(
                task=asdict(task),
                learning_rate=learning_rate,
                additional_steps=steps,
                starting_timesteps=start_steps,
                checkpoint_sha256=sha256_file(checkpoint),
                initial_policy_sha256=policy_digest.hexdigest(),
                seed=task.seed,
                validation_seeds=validation_seeds,
                test_seeds=TEST_SEEDS,
                only_treatment_variable="constant PPO learning rate",
                checkpoint_selection="final success, then any-step success, then return; includes untrained continuation at step zero",
                resume_semantics="model and Adam restored; new environment resets and RNG seed shared between forks; rollout buffer not restored",
                versions=versions(),
                source_sha256=sha256_file(__file__),
                created_utc=datetime.now(timezone.utc).isoformat(),
            ),
        )
        initial = evaluate(task, model, validation_seeds)
        initial_point = dict(
            additional_timesteps=0, timesteps=start_steps, **initial["summary"]
        )
        model.save(directory / "best")
        save_json(directory / "selected.json", initial_point)
        save_json(directory / "validation.json", [initial_point])

        class Validation(BaseCallback):
            def __init__(self):
                super().__init__()
                self.next_eval = start_steps + interval
                self.points = [initial_point]
                self.best = validation_score(initial["summary"])

            def _on_step(self):
                if self.num_timesteps >= self.next_eval:
                    result = evaluate(task, self.model, validation_seeds)
                    point = dict(
                        additional_timesteps=self.num_timesteps - start_steps,
                        timesteps=self.num_timesteps,
                        **result["summary"],
                    )
                    self.points.append(point)
                    if validation_score(result["summary"]) > self.best:
                        self.best = validation_score(result["summary"])
                        self.model.save(directory / "best")
                        save_json(directory / "selected.json", point)
                    save_json(directory / "validation.json", self.points)
                    self.next_eval += interval
                return True

        start = time.monotonic()
        model.learn(
            total_timesteps=steps,
            reset_num_timesteps=False,
            callback=Validation(),
            log_interval=8,
        )
        model.save(directory / "final")
        save_json(
            directory / "training.json",
            dict(
                actual_additional_timesteps=model.num_timesteps - start_steps,
                total_timesteps=model.num_timesteps,
                complete=True,
                elapsed_seconds=time.monotonic() - start,
                final_learning_rate=model.policy.optimizer.param_groups[0]["lr"],
            ),
        )
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--learning-rate", required=True, type=float)
    parser.add_argument("--steps", type=int, default=ADDITIONAL_STEPS)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--interval", type=int, default=131072)
    args = parser.parse_args()
    continue_training(
        args.checkpoint,
        args.output,
        args.learning_rate,
        args.steps,
        args.num_envs,
        args.interval,
    )


if __name__ == "__main__":
    main()
