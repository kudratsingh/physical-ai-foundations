"""Paired BC-initialized versus random-actor PPO with stochastic training telemetry."""

import argparse
from dataclasses import asdict
import hashlib
import json
import pickle
from pathlib import Path
import random
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .bc_ppo_policy import build, load_teacher, action_agreement
from .experiment import Config, evaluate, make_env, save_json, versions
from .normalized_ppo import sha256
from .selection import holding_score
from .training_metrics import training_env, TrainingMetrics

TASK = Config(
    env_id="PickCube-v1",
    total_timesteps=1048576,
    num_envs=4,
    validation_interval=65536,
    validation_seed_start=140000,
    validation_episodes=10,
    test_seed_start=150000,
    test_episodes=50,
)
SOURCES = [
    "bc_initialized_ppo.py",
    "bc_ppo_policy.py",
    "rng.py",
    "training_metrics.py",
    "experiment.py",
]


def fingerprint(model, actor):
    digest = hashlib.sha256()
    for name, value in sorted(model.policy.state_dict().items()):
        is_actor = name.startswith(("mlp_extractor.policy_net.", "action_net."))
        if is_actor != actor:
            continue
        array = value.cpu().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def rng_fingerprint():
    return hashlib.sha256(
        pickle.dumps(
            (
                random.getstate(),
                np.random.get_state(),
                torch.random.get_rng_state().numpy(),
            )
        )
    ).hexdigest()


def preflight(output, teacher_path):
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    teacher = load_teacher(teacher_path)
    env = DummyVecEnv([training_env(TASK)])
    try:
        control = build(env, teacher, initialize=False)
        warm = build(env, teacher)
        if fingerprint(control, False) != fingerprint(warm, False):
            raise ValueError("Non-actor initial tensors differ")
        if control.policy.optimizer.state or warm.policy.optimizer.state:
            raise ValueError("Need fresh optimizers")
        control.save(output / "random_initial")
        warm.save(output / "warm_initial")
        probes = []
        physical = make_env(TASK)
        try:
            for seed in range(140000, 140010):
                obs, _ = physical.reset(seed=seed)
                for _ in range(50):
                    probes.append(obs.copy())
                    action, _ = teacher.predict(obs)
                    obs, _, _, _, _ = physical.step(action)
        finally:
            physical.close()
        agreement = action_agreement(teacher, warm, probes)
        if agreement["max_absolute_action_error"] > 1e-6:
            raise ValueError(f"BC transfer mismatch: {agreement}")
        seeds = list(range(140000, 140010))
        before = evaluate(TASK, teacher, seeds, output / "preflight_teacher.json")
        after = evaluate(TASK, warm, seeds, output / "preflight_initialized.json")
        differences = []
        for a, b in zip(before["episodes"], after["episodes"], strict=True):
            errors = [
                abs(a["return"] - b["return"]),
                np.max(
                    np.abs(
                        np.array(a["final_cube_position"]) - b["final_cube_position"]
                    )
                ),
            ]
            if (
                a["success_at_end"] != b["success_at_end"]
                or a["success_once"] != b["success_once"]
                or max(errors) > 1e-4
            ):
                raise ValueError(
                    "Initialized policy does not reproduce teacher validation behavior"
                )
            differences.append(max(errors))
        save_json(
            output / "preflight.json",
            dict(
                action_agreement=agreement,
                max_rollout_field_error=max(differences),
                teacher_summary=before["summary"],
                initialized_summary=after["summary"],
                non_actor_fingerprint=fingerprint(warm, False),
                random_actor_fingerprint=fingerprint(control, True),
                warm_actor_fingerprint=fingerprint(warm, True),
            ),
        )
    finally:
        env.close()
    save_json(
        output / "protocol.json",
        dict(
            config=asdict(TASK),
            teacher_sha256=sha256(teacher_path),
            versions=versions(),
            sources={name: sha256(Path(__file__).with_name(name)) for name in SOURCES},
            validation_seeds=list(range(140000, 140010)),
            test_seeds=list(range(150000, 150050)),
            video_seeds=[150000, 150001],
            learning_rate=3e-5,
            critic_feature_clip=10.0,
            target_kl=0.01,
            initial_latent_log_std=-3,
            initial_exploration_std=float(np.exp(-3)),
            treatment="BC actor tensors only; both arms use identical fixed BC-derived normalization, ReLU architecture, Gaussian noise around a tanh-bounded mean, separate clipped critic features, fresh critic/optimizer, and exploration settings",
            primary="Warm final vs teacher, initialized warm, and fresh matched random-actor final at fixed budget; held-out resets untouched until training complete",
            interpretation="Matched control is not the archived raw-Gaussian recipe; shared preprocessing uses demonstration-derived statistics in both arms",
            checkpoint_selection=[
                "success_at_end_rate",
                "success_once_rate",
                "return_mean",
            ],
            replication_trigger="At least five additional final successes out of50 versus BOTH teacher and matched control; otherwise no improvement claim",
            training_metrics="All stochastic training transitions; no validation/test counts; actual grasp/placement/success/static/boundary and grasp-conditioned distance occupancy",
        ),
    )


def train(output, teacher_path, arm):
    torch.set_num_threads(1)
    output = Path(output)
    protocol = json.loads((output / "protocol.json").read_text())
    config = Config(**protocol["config"])
    if sha256(teacher_path) != protocol["teacher_sha256"]:
        raise ValueError("Teacher changed")
    for name, digest in protocol["sources"].items():
        if sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError(f"Source changed since declaration: {name}")
    directory = output / arm
    directory.mkdir(exist_ok=False)
    teacher = load_teacher(teacher_path)
    constructors = [training_env(config) for _ in range(config.num_envs)]
    env = (
        DummyVecEnv(constructors)
        if config.num_envs == 1
        else SubprocVecEnv(constructors, start_method="spawn")
    )
    env.seed(config.seed)

    class Validation(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next = config.validation_interval
            self.points = []
            self.best = None

        def assess(self, timesteps):
            result = evaluate(config, self.model, protocol["validation_seeds"])
            point = dict(timesteps=timesteps, **result["summary"])
            self.points.append(point)
            score = holding_score(point)
            if self.best is None or score > self.best:
                self.best = score
                self.model.save(directory / "best")
                save_json(directory / "selected.json", point)
            save_json(directory / "validation.json", self.points)

        def _on_training_start(self):
            self.assess(0)

        def _on_step(self):
            if self.num_timesteps >= self.next:
                self.assess(self.num_timesteps)
                self.next += config.validation_interval
            return True

    try:
        model = build(
            env,
            teacher,
            seed=config.seed,
            initialize=arm == "warm",
            rollout_steps=config.rollout_steps,
            batch_size=config.batch_size,
            epochs=config.epochs,
        )
        model.save(directory / "initial")
        save_json(
            directory / "initialization.json",
            dict(
                actor_fingerprint=fingerprint(model, True),
                non_actor_fingerprint=fingerprint(model, False),
                rng_fingerprint=rng_fingerprint(),
                optimizer_empty=not bool(model.policy.optimizer.state),
                checkpoint_sha256=sha256(directory / "initial.zip"),
            ),
        )
        started = time.monotonic()
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=CallbackList(
                [TrainingMetrics(directory / "training_metrics.json"), Validation()]
            ),
            log_interval=32,
        )
        model.save(directory / "final")
        save_json(
            directory / "training.json",
            dict(
                actual_timesteps=model.num_timesteps,
                elapsed_seconds=time.monotonic() - started,
                complete=True,
            ),
        )
    finally:
        env.close()


def compare(output, teacher_path):
    output = Path(output)
    protocol = json.loads((output / "protocol.json").read_text())
    config = Config(**protocol["config"])
    initial = {
        arm: json.loads((output / arm / "initialization.json").read_text())
        for arm in ["warm", "control"]
    }
    for key in ["non_actor_fingerprint", "rng_fingerprint", "optimizer_empty"]:
        if initial["warm"][key] != initial["control"][key]:
            raise ValueError(f"Paired initialization differs: {key}")
    models = {
        "teacher": load_teacher(teacher_path),
        "warm_initialized": PPO.load(output / "warm/initial.zip", device="cpu"),
        "warm_final": PPO.load(output / "warm/final.zip", device="cpu"),
        "control_final": PPO.load(output / "control/final.zip", device="cpu"),
        "warm_selected_secondary": PPO.load(output / "warm/best.zip", device="cpu"),
    }
    for arm in ["warm", "control"]:
        training = json.loads((output / arm / "training.json").read_text())
        if (
            training["actual_timesteps"] != config.total_timesteps
            or not training["complete"]
        ):
            raise ValueError("Incomplete paired training budget")
        if models[f"{arm}_final"].num_timesteps != config.total_timesteps:
            raise ValueError("Actual loaded budget mismatch")
    comparison = {}
    for name, model in models.items():
        result = evaluate(
            config, model, protocol["test_seeds"], output / f"test_{name}.json"
        )
        comparison[name] = result["summary"]
        if name in ["teacher", "warm_final", "control_final"]:
            video = evaluate(
                config,
                model,
                protocol["video_seeds"],
                output / f"video_{name}.json",
                traces=True,
                video=True,
            )
            for a, b in zip(video["episodes"], result["episodes"]):
                if {k: v for k, v in a.items() if k != "trajectory"} != {
                    k: v for k, v in b.items() if k != "trajectory"
                }:
                    raise ValueError("Video mismatch")
    save_json(output / "comparison.json", comparison)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preflight", "train", "compare"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--arm", choices=["warm", "control"])
    args = parser.parse_args()
    if args.command == "preflight":
        preflight(args.output, args.teacher)
    elif args.command == "train":
        if not args.arm:
            parser.error("train needs --arm")
        train(args.output, args.teacher, args.arm)
    else:
        compare(args.output, args.teacher)


if __name__ == "__main__":
    main()
