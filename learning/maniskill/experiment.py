"""ManiSkill CPU quickstart: smoke, train PPO, and evaluate fixed held-out episodes.

Uses the upstream PushCube task and CPUGymWrapper, without changing task rewards,
initialization, action semantics or success conditions. No ROS in the training loop.
"""

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import importlib.metadata
import json
import os
import hashlib
from pathlib import Path
import platform
import time

from .metrics import summarize


@dataclass(frozen=True)
class Config:
    env_id: str = "PushCube-v1"
    obs_mode: str = "state"
    control_mode: str = "pd_joint_delta_pos"
    reward_mode: str = "normalized_dense"
    seed: int = 42
    total_timesteps: int = 65536
    num_envs: int = 4
    rollout_steps: int = 256
    batch_size: int = 256
    epochs: int = 8
    validation_interval: int = 16384
    validation_episodes: int = 5
    test_episodes: int = 20
    test_seed_start: int = 10000


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def versions():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        **{
            name: importlib.metadata.version(name)
            for name in [
                "mani-skill-nightly",
                "sapien",
                "torch",
                "stable-baselines3",
                "gymnasium",
                "numpy",
            ]
        },
    }


def make_env(config, render=False):
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("MS_ASSET_DIR", str(root / ".assets"))
    local_icd = root / ".runtime/MoltenVK_icd.json"
    if platform.system() == "Darwin" and local_icd.exists():
        os.environ.setdefault("VK_ICD_FILENAMES", str(local_icd))
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 - registers upstream tasks
    from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

    env = gym.make(
        config.env_id,
        num_envs=1,
        obs_mode=config.obs_mode,
        control_mode=config.control_mode,
        reward_mode=config.reward_mode,
        sim_backend="physx_cpu",
        render_backend="cpu",
        render_mode="rgb_array" if render else None,
        reconfiguration_freq=1,
    )
    return CPUGymWrapper(env, ignore_terminations=True, record_metrics=False)


def env_thunk(config):
    def create():
        import torch
        from stable_baselines3.common.monitor import Monitor

        torch.set_num_threads(1)
        return Monitor(make_env(config))

    return create


def trajectory_sample(env, step):
    """Privileged simulator state for rollout inspection, never extra policy input."""
    base = env.unwrapped
    return {
        "step": step,
        "tcp": base.agent.tcp.pose.p[0].cpu().tolist(),
        "cube": base.obj.pose.p[0].cpu().tolist(),
        "goal": base.goal_region.pose.p[0].cpu().tolist(),
        "qpos": base.agent.robot.get_qpos()[0].cpu().tolist(),
    }


def evaluate(config, policy, seeds, output=None, traces=False, video=False):
    import numpy as np
    import torch

    torch.set_num_threads(1)
    if video and output is None:
        raise ValueError("Video evaluation requires an output path")
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
    env = make_env(config, render=video)
    episodes = []
    try:
        for index, seed in enumerate(seeds):
            obs, _ = env.reset(seed=seed)
            env.action_space.seed(seed + 1000000)
            total, success_once, first_success, step = 0.0, False, None, 0
            trajectory = [trajectory_sample(env, 0)] if traces else []
            writer = None
            if video:
                import imageio.v2 as imageio

                output_path = Path(output).parent / (
                    Path(output).stem + f"_seed{seed}.mp4"
                )
                writer = imageio.get_writer(output_path, fps=env.unwrapped.control_freq)
            try:
                while True:
                    if policy == "random":
                        action = env.action_space.sample()
                    elif policy == "zero":
                        action = np.zeros(
                            env.action_space.shape, dtype=env.action_space.dtype
                        )
                    else:
                        action, _ = policy.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    step += 1
                    total += float(reward)
                    success = bool(info.get("success", False))
                    if success and first_success is None:
                        first_success = step
                    success_once |= success
                    if traces:
                        trajectory.append(trajectory_sample(env, step))
                    if writer:
                        writer.append_data(env.render())
                    if terminated or truncated:
                        break
            finally:
                if writer:
                    writer.close()
            terminal = trajectory_sample(env, step)
            episodes.append(
                dict(
                    seed=seed,
                    return_=total,
                    steps=step,
                    success_once=success_once,
                    success_at_end=success,
                    first_success_step=first_success,
                    trajectory=trajectory,
                    final_cube_position=terminal["cube"],
                    final_goal_position=terminal["goal"],
                )
            )
            episodes[-1]["return"] = episodes[-1].pop("return_")
            print(
                f"eval {index + 1}/{len(seeds)} seed={seed} return={total:.3f} success={success_once}",
                flush=True,
            )
    finally:
        env.close()
    result = {
        "policy": policy
        if isinstance(policy, str)
        else getattr(policy, "policy_label", "PPO deterministic"),
        "summary": summarize(episodes),
        "episodes": episodes,
        "config": asdict(config),
        "versions": versions(),
    }
    if output:
        save_json(output, result)
    return result


def train(config, directory):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    torch.set_num_threads(1)
    directory.mkdir(parents=True, exist_ok=False)
    save_json(
        directory / "config.json",
        {
            **asdict(config),
            "versions": versions(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "algorithm": "Stable-Baselines3 PPO",
            "device": "cpu",
        },
    )
    environments = [env_thunk(config) for _ in range(config.num_envs)]
    env = (
        DummyVecEnv(environments)
        if config.num_envs == 1
        else SubprocVecEnv(environments, start_method="spawn")
    )
    env.seed(config.seed)

    class Validation(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_eval = config.validation_interval
            self.points = []

        def _on_step(self):
            if self.num_timesteps >= self.next_eval:
                self.model.save(directory / "latest")
                result = evaluate(
                    config,
                    self.model,
                    list(range(1000, 1000 + config.validation_episodes)),
                )
                self.points.append(
                    {"timesteps": self.num_timesteps, **result["summary"]}
                )
                save_json(directory / "validation.json", self.points)
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
            tensorboard_log=str(directory / "tensorboard"),
            policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        )
        start = time.monotonic()
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=Validation(),
            log_interval=4,
        )
        model.save(directory / "final")
        save_json(
            directory / "training.json",
            {
                "actual_timesteps": model.num_timesteps,
                "elapsed_seconds": time.monotonic() - start,
                "complete": True,
            },
        )
    finally:
        env.close()
    seeds = list(
        range(config.test_seed_start, config.test_seed_start + config.test_episodes)
    )
    for name, policy in [("ppo", model), ("random", "random"), ("zero", "zero")]:
        evaluate(config, policy, seeds, directory / f"test_{name}.json", traces=True)
    report(directory)


def report(directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = {
        name: json.loads((directory / f"test_{name}.json").read_text())
        for name in ["ppo", "random", "zero"]
    }
    validation_file = directory / "validation.json"
    validation = (
        json.loads(validation_file.read_text()) if validation_file.exists() else []
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].plot(
        [p["timesteps"] for p in validation],
        [p["return_mean"] for p in validation],
        marker="o",
    )
    axes[0].set(
        title="Validation return · seeds 1000+",
        xlabel="Training transitions",
        ylabel="Mean episode return",
    )
    x = list(range(len(results)))
    axes[1].bar(
        [v - 0.18 for v in x],
        [r["summary"]["success_once_rate"] for r in results.values()],
        width=0.36,
        label="Any step",
    )
    axes[1].bar(
        [v + 0.18 for v in x],
        [r["summary"]["success_at_end_rate"] for r in results.values()],
        width=0.36,
        label="At end",
    )
    axes[1].set(
        title="Held-out task success",
        ylim=(0, 1),
        ylabel="Fraction of test episodes",
        xticks=x,
        xticklabels=list(results),
    )
    axes[1].legend()
    axes[2].boxplot(
        [[e["return"] for e in r["episodes"]] for r in results.values()],
        tick_labels=list(results),
    )
    axes[2].set(title="Held-out episode returns", ylabel="Normalized dense return")
    fig.tight_layout()
    fig.savefig(directory / "results.png", dpi=160)
    plt.close(fig)
    save_json(
        directory / "comparison.json",
        {name: r["summary"] for name, r in results.items()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["smoke", "train", "evaluate", "report"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--policy", choices=["ppo", "random", "zero"], default="ppo")
    parser.add_argument("--timesteps", type=int, default=65536)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if (
        args.timesteps <= 0
        or args.num_envs < 1
        or args.episodes < 1
        or args.seed_start < 2000
    ):
        parser.error(
            "Positive budgets required; test seeds must start at 2000 or higher"
        )
    config = Config(
        total_timesteps=args.timesteps,
        num_envs=args.num_envs,
        test_episodes=args.episodes,
        test_seed_start=args.seed_start,
    )
    if args.command == "smoke":
        from stable_baselines3.common.env_checker import check_env
        import torch

        torch.set_num_threads(1)
        env = make_env(config)
        try:
            check_env(env, warn=True)
            obs, _ = env.reset(seed=42)
            print("Versions", versions())
            print("Observation", env.observation_space, "Action", env.action_space)
            start = time.monotonic()
            for _ in range(200):
                obs, reward, terminated, truncated, info = env.step(
                    env.action_space.sample()
                )
                if terminated or truncated:
                    obs, _ = env.reset()
            print(
                f"PASS: 200 transitions at {200 / (time.monotonic() - start):.1f} steps/s"
            )
        finally:
            env.close()
    elif args.command == "train":
        train(
            config,
            args.output
            or Path("learning/runs")
            / datetime.now(timezone.utc).strftime("pushcube_%Y%m%dT%H%M%SZ"),
        )
    elif args.command == "report":
        if not args.output:
            parser.error("report requires --output RUN_DIRECTORY")
        report(args.output)
    else:
        if not args.output:
            parser.error("evaluate requires --output RESULTS.json")
        policy = args.policy
        if policy == "ppo":
            if not args.checkpoint:
                parser.error("PPO evaluation requires --checkpoint")
            from stable_baselines3 import PPO

            metadata = args.checkpoint.parent / "config.json"
            if metadata.exists():
                saved = json.loads(metadata.read_text())
                config = Config(**{k: saved[k] for k in asdict(Config())})
                config = replace(
                    config, test_episodes=args.episodes, test_seed_start=args.seed_start
                )
            policy = PPO.load(args.checkpoint, device="cpu")
        evaluate(
            config,
            policy,
            list(range(args.seed_start, args.seed_start + args.episodes)),
            args.output,
            traces=True,
            video=args.video,
        )


if __name__ == "__main__":
    main()
