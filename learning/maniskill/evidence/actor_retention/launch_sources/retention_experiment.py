"""Predeclared seeded three-replicate teacher-anchored PPO ablation."""

import argparse
import hashlib
import json
import shutil
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv

from .bc_initialized_ppo import rng_fingerprint
from .bc_ppo_policy import FixedNormalization, load_teacher
from .critic_diagnostics import RolloutDiagnostics
from .critic_experiment import canonical_fingerprint
from .critic_policy import BoundedCriticPolicy
from .experiment import Config, evaluate, save_json, versions
from .normalized_ppo import sha256
from .retention_ppo import RetentionPPO, retention_probe
from .rng import preserve_rng
from .seeded_env import seeded_training_env
from .selection import holding_score
from .training_metrics import TrainingMetrics

SEEDS = [42, 1042, 2042]
ARMS = ["baseline", "anchored"]
TASK = Config(
    env_id="PickCube-v1",
    total_timesteps=262144,
    num_envs=4,
    validation_interval=32768,
    validation_seed_start=230000,
    validation_episodes=40,
    test_seed_start=240000,
    test_episodes=50,
)
SOURCES = [
    "retention_experiment.py",
    "retention_ppo.py",
    "first_rollout.py",
    "retention_evaluation.py",
    "bc_ppo_policy.py",
    "behavior_cloning.py",
    "critic_policy.py",
    "critic_diagnostics.py",
    "critic_experiment.py",
    "bc_initialized_ppo.py",
    "experiment.py",
    "metrics.py",
    "selection.py",
    "training_metrics.py",
    "rng.py",
    "seeded_env.py",
]


def tensor_fingerprint(module):
    digest = hashlib.sha256()
    for n, t in sorted(module.state_dict().items()):
        a = t.detach().cpu().numpy()
        digest.update(n.encode())
        digest.update(str(a.dtype).encode())
        digest.update(str(a.shape).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def build(env, teacher, seed, coefficient):
    model = RetentionPPO(
        BoundedCriticPolicy,
        env,
        teacher=teacher,
        retention_coefficient=coefficient,
        seed=seed,
        device="cpu",
        verbose=1,
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
        for source, dest in [
            (teacher.net[0], model.policy.mlp_extractor.policy_net[0]),
            (teacher.net[2], model.policy.mlp_extractor.policy_net[2]),
            (teacher.net[4], model.policy.action_net),
        ]:
            dest.weight.copy_(source.weight)
            dest.bias.copy_(source.bias)
    return model


def workers(config):
    env = SubprocVecEnv(
        [seeded_training_env(config, rank) for rank in range(config.num_envs)],
        start_method="spawn",
    )
    env.seed(config.seed)
    return env


def check_sources(output, teacher):
    p = json.loads((Path(output) / "protocol.json").read_text())
    if sha256(teacher) != p["teacher_sha256"]:
        raise ValueError("Teacher changed")
    for n, h in p["sources"].items():
        if sha256(Path(__file__).with_name(n)) != h:
            raise ValueError(f"Frozen source changed:{n}")
    return p


def declare(output, teacher):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save_json(
        output / "protocol.json",
        {
            "config": asdict(TASK),
            "training_seeds": SEEDS,
            "arms": ARMS,
            "teacher_sha256": sha256(teacher),
            "versions": versions(),
            "sources": {n: sha256(Path(__file__).with_name(n)) for n in SOURCES},
            "validation_seeds": list(range(230000, 230040)),
            "test_seeds": list(range(240000, 240050)),
            "video_seeds": [240000, 240001],
            "worker_reset_seeding": "Per-worker Python/NumPy/Torch seed+rank before/after construction and enhanced_determinism=True",
            "auxiliary_supervision": "Current PPO training minibatch only; frozen originalBCteacher, mean-action L2 summed over8actions then averaged overstates. No demonstration/validation/test supervision.",
            "calibration": {
                "seed": 91042,
                "transitions": 8192,
                "target_gradient_ratio": 0.5,
                "coefficient_clip": [0.01, 10.0],
                "procedure": "At final separate pilot weights, median(.5*PPOactor_grad_norm/teacher_actor_grad_norm) over4minibatches oflast1024state rollout; at least3finite nonzero ratios. Discard allpilotweights/optimizer.",
            },
            "primary": "Fixed-budget final tasksuccess, each of3trainingseeds; same50resetlist acrossseeds, not150independentresettrials",
            "secondary": "Validation-selected checkpoint finalsuccess; final-first holding_score withstep0 eligible",
            "replication": "Allsixconditions run irrespective ofpilot outcomes; noadaptivebudget",
            "promising": "Anchoredfinal exceeds both teacher andbaselinefinal by>=5/50 in at least2of3seeds, and mean(selected-final)collapse decreases",
            "holding": "Post-first-success and immediate-success-conditioned commands, velocities,teacherdrift,gripperchanges,grasp,occupancy andsuccess-to-failure transitions; passive diagnostics only",
        },
    )
    archive = output / "launch_sources"
    archive.mkdir()
    for n in SOURCES:
        shutil.copy2(Path(__file__).with_name(n), archive / n)


def calibrate(output, teacher_path):
    torch.set_num_threads(1)
    output = Path(output)
    p = check_sources(output, teacher_path)
    if (output / "calibration.json").exists():
        raise FileExistsError("Calibration already frozen")
    first = json.loads((output / "first_rollout/receipt.json").read_text())
    if not (
        first.get("passed")
        and first.get("all_arrays_exactly_equal")
        and first.get("optimizer_empty_after")
    ):
        raise ValueError("First-rollout preflight did not pass exactly")
    if (
        first["teacher_sha256"] != p["teacher_sha256"]
        or first["source_sha256"] != p["sources"]["first_rollout.py"]
    ):
        raise ValueError("First-rollout teacher/source differs from declaration")
    if (
        sha256(output / "first_rollout" / first["actual_records"])
        != first["records_sha256"]
    ):
        raise ValueError("First-rollout records changed")
    if (
        first["workers"] != TASK.num_envs
        or first["transitions_per_arm"] != TASK.num_envs * TASK.rollout_steps
    ):
        raise ValueError(
            "First-rollout configuration differs from performance training"
        )
    teacher = load_teacher(teacher_path)
    tf = tensor_fingerprint(teacher)
    cfg = replace(TASK, seed=91042, total_timesteps=8192)
    env = workers(cfg)
    try:
        model = build(env, teacher, 91042, 0.0)
        model.learn(8192, callback=RolloutDiagnostics(), log_interval=8)
        probes = []
        with preserve_rng():
            for batch in model.rollout_buffer.get(256):
                probes.append(retention_probe(model, batch))
        ratios = [
            0.5 * r["ppo_actor_gradient_norm"] / r["teacher_actor_gradient_norm"]
            for r in probes
            if r["teacher_actor_gradient_norm"] > 1e-12
            and np.isfinite(r["teacher_actor_gradient_norm"])
            and np.isfinite(r["ppo_actor_gradient_norm"])
            and r["ppo_actor_gradient_norm"] > 1e-12
        ]
        ratios = [r for r in ratios if np.isfinite(r) and r > 0]
        if len(ratios) < 3 or len(probes) != 4:
            raise ValueError("Insufficient calibration gradient ratios")
        coefficient = float(np.clip(statistics.median(ratios), 0.01, 10.0))
        if tensor_fingerprint(model.teacher) != tf:
            raise ValueError("Teacher changed")
        save_json(
            output / "calibration.json",
            {
                "coefficient": coefficient,
                "method": p["calibration"],
                "probes": probes,
                "ratios": ratios,
                "actual_timesteps": model.num_timesteps,
                "teacher_fingerprint": tf,
                "supervision": "Calibration training rollout only; no validation/test",
                "pilot_discarded": True,
            },
        )
        save_json(
            output / "calibration_frozen.json",
            {
                "sha256": sha256(output / "calibration.json"),
                "protocol_sha256": sha256(output / "protocol.json"),
                "coefficient": coefficient,
            },
        )
    finally:
        env.close()


def coefficient(output, arm):
    output = Path(output)
    f = json.loads((output / "calibration_frozen.json").read_text())
    if sha256(output / "protocol.json") != f["protocol_sha256"]:
        raise ValueError("Declared protocol changed")
    if sha256(output / "calibration.json") != f["sha256"]:
        raise ValueError("Calibration changed")
    if (
        f["coefficient"]
        != json.loads((output / "calibration.json").read_text())["coefficient"]
    ):
        raise ValueError("Coefficient mismatch")
    return f["coefficient"] if arm == "anchored" else 0.0


def train(output, teacher_path, seed, arm):
    torch.set_num_threads(1)
    output = Path(output)
    p = check_sources(output, teacher_path)
    if seed not in SEEDS or arm not in ARMS:
        raise ValueError("Undeclared condition")
    weight = coefficient(output, arm)
    cfg = replace(TASK, seed=seed)
    directory = output / f"seed{seed}" / arm
    directory.mkdir(parents=True, exist_ok=False)
    env = workers(cfg)

    class Validate(BaseCallback):
        def __init__(self):
            super().__init__()
            self.rows = []
            self.next = 32768
            self.best = None

        def assess(self):
            r = evaluate(cfg, self.model, p["validation_seeds"])
            point = dict(timesteps=self.num_timesteps, **r["summary"])
            self.rows.append(point)
            score = holding_score(point)
            if self.best is None or score > self.best:
                self.best = score
                self.model.save(directory / "best")
                save_json(directory / "selected.json", point)
            save_json(directory / "validation.json", self.rows)

        def _on_training_start(self):
            self.assess()

        def _on_step(self):
            if self.num_timesteps >= self.next:
                self.assess()
                self.next += 32768
            return True

    try:
        teacher = load_teacher(teacher_path)
        model = build(env, teacher, seed, weight)
        model.diagnostics_path = directory / "diagnostics.json"
        model.retention_path = directory / "retention.json"
        model.save(directory / "initial")
        save_json(
            directory / "initialization.json",
            {
                "canonical_fingerprint": canonical_fingerprint(model),
                "rng_fingerprint": rng_fingerprint(),
                "teacher_fingerprint": tensor_fingerprint(model.teacher),
                "optimizer_empty": not bool(model.policy.optimizer.state),
                "checkpoint_sha256": sha256(directory / "initial.zip"),
                "coefficient": weight,
                "calibration_sha256": sha256(output / "calibration.json"),
            },
        )
        start = time.monotonic()
        model.learn(
            262144,
            callback=CallbackList(
                [
                    TrainingMetrics(directory / "training_metrics.json"),
                    RolloutDiagnostics(),
                    Validate(),
                ]
            ),
            log_interval=32,
        )
        if tensor_fingerprint(model.teacher) != tensor_fingerprint(teacher):
            raise ValueError("Teacher updated")
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["declare", "calibrate", "train"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--arm", choices=ARMS)
    a = p.parse_args()
    if a.command == "declare":
        declare(a.output, a.teacher)
    elif a.command == "calibrate":
        calibrate(a.output, a.teacher)
    else:
        if not a.arm:
            p.error("train requires --arm")
        train(a.output, a.teacher, a.seed, a.arm)


if __name__ == "__main__":
    main()
