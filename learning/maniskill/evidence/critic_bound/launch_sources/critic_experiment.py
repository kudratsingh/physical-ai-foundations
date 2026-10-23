"""Paired short-budget critic-bound ablation with actual gradient telemetry."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .bc_initialized_ppo import rng_fingerprint
from .bc_ppo_policy import BCMeanGaussianPolicy, FixedNormalization, load_teacher
from .critic_diagnostics import DiagnosticPPO, RolloutDiagnostics
from .critic_policy import BoundedCriticPolicy
from .experiment import Config, evaluate, save_json, versions
from .normalized_ppo import sha256
from .selection import holding_score
from .training_metrics import TrainingMetrics, training_env

TASK = Config(
    env_id="PickCube-v1",
    total_timesteps=262144,
    num_envs=4,
    validation_interval=32768,
    validation_seed_start=180000,
    validation_episodes=10,
    test_seed_start=190000,
    test_episodes=50,
)
SOURCES = [
    "critic_experiment.py",
    "critic_policy.py",
    "critic_diagnostics.py",
    "bc_ppo_policy.py",
    "bc_initialized_ppo.py",
    "experiment.py",
    "training_metrics.py",
    "rng.py",
    "selection.py",
    "metrics.py",
    "behavior_cloning.py",
]


def canonical_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.policy.state_dict().items()):
        if name.startswith("value_net.0."):
            name = "value_net." + name.removeprefix("value_net.0.")
        array = value.cpu().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def build(env, teacher, arm):
    model = DiagnosticPPO(
        BoundedCriticPolicy if arm == "bounded" else BCMeanGaussianPolicy,
        env,
        seed=42,
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


def declare(output, teacher_path):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save_json(
        output / "protocol.json",
        {
            "config": asdict(TASK),
            "teacher_sha256": sha256(teacher_path),
            "sources": {n: sha256(Path(__file__).with_name(n)) for n in SOURCES},
            "versions": versions(),
            "validation_seeds": list(range(180000, 180010)),
            "test_seeds": list(range(190000, 190050)),
            "video_seeds": [190000, 190001],
            "arms": ["raw", "bounded"],
            "value_bound": 100.0,
            "treatment": "Only critic output changes: V=100*tanh(raw_V/100). Identical actor and canonical critic parameters, normalization, RNG and optimizer recipe. No target/reward clipping or timeout-semantics change.",
            "budget_reason": "Short diagnostic budget fixed at the previously observed regression point, before fresh test evaluation. Not evidence of million-transition stability.",
            "primary": "Fixed-budget final bounded versus raw and BC teacher; selected checkpoints secondary.",
            "seed": 42,
            "initial_state_requirement": "Canonical full policy tensor fingerprints, saved pre-learning RNG and empty optimizers must match.",
            "telemetry": "Actual first backward/optimizer step each rollout: actor/critic gradients before/after global clipping and Adam parameter deltas; rollout values, returns, raw and timeout-corrected rewards. Hooks do not alter tensors/RNG.",
            "reward_bound": "Installed PickCube normalized dense reward is in[0,1]. With gamma.99, continuing discounted value is in[0,100]. The critic permits[-100,100].",
        },
    )
    sources = output / "launch_sources"
    sources.mkdir()
    for n in SOURCES:
        (sources / n).write_bytes(Path(__file__).with_name(n).read_bytes())


def preflight(output, teacher_path):
    torch.set_num_threads(1)
    output = Path(output)
    p = json.loads((output / "protocol.json").read_text())
    if sha256(teacher_path) != p["teacher_sha256"]:
        raise ValueError("Teacher changed")
    teacher = load_teacher(teacher_path)
    env = DummyVecEnv([training_env(TASK)])
    try:
        models = {a: build(env, teacher, a) for a in ["raw", "bounded"]}
        fingerprints = {a: canonical_fingerprint(m) for a, m in models.items()}
        if len(set(fingerprints.values())) != 1:
            raise ValueError("Different initial tensors")
        results = {
            name: evaluate(
                TASK, model, p["validation_seeds"], output / f"preflight_{name}.json"
            )
            for name, model in {"teacher": teacher, **models}.items()
        }
        for arm in ["raw", "bounded"]:
            if results[arm]["episodes"] != results["teacher"]["episodes"]:
                raise ValueError("Initial actor does not exactly reproduce teacher")
        save_json(
            output / "preflight.json",
            {
                "passed": True,
                "full_episode_identity": True,
                "canonical_fingerprint": fingerprints["raw"],
                "validation_episodes": 10,
                "teacher_summary": results["teacher"]["summary"],
            },
        )
        # A separate stochastic-update pilot, never used as an initialization.
        models["bounded"].learn(4096)
        if not all(
            torch.isfinite(v).all()
            for v in models["bounded"].policy.state_dict().values()
        ):
            raise ValueError("Nonfinite stochastic pilot")
        save_json(
            output / "stochastic_smoke.json",
            {"transitions": 4096, "finite": True, "excluded_from_training": True},
        )
    finally:
        env.close()


def train(output, teacher_path, arm):
    torch.set_num_threads(1)
    output = Path(output)
    protocol = json.loads((output / "protocol.json").read_text())
    if sha256(teacher_path) != protocol["teacher_sha256"]:
        raise ValueError("Teacher changed")
    for n, h in protocol["sources"].items():
        if sha256(Path(__file__).with_name(n)) != h:
            raise ValueError(f"Source changed:{n}")
    preflight_result = json.loads((output / "preflight.json").read_text())
    if (
        not preflight_result["passed"]
        or not json.loads((output / "stochastic_smoke.json").read_text())["finite"]
    ):
        raise ValueError("Preflight did not pass")
    directory = output / arm
    directory.mkdir()
    env = SubprocVecEnv([training_env(TASK) for _ in range(4)], start_method="spawn")
    env.seed(42)

    class Validate(BaseCallback):
        def __init__(self):
            super().__init__()
            self.points = []
            self.best = None
            self.next = 32768

        def assess(self):
            result = evaluate(TASK, self.model, protocol["validation_seeds"])
            point = dict(timesteps=self.num_timesteps, **result["summary"])
            self.points.append(point)
            score = holding_score(point)
            if self.best is None or score > self.best:
                self.best = score
                self.model.save(directory / "best")
                save_json(directory / "selected.json", point)
            save_json(directory / "validation.json", self.points)

        def _on_training_start(self):
            self.assess()

        def _on_step(self):
            if self.num_timesteps >= self.next:
                self.assess()
                self.next += 32768
            return True

    try:
        model = build(env, load_teacher(teacher_path), arm)
        model.diagnostics_path = directory / "diagnostics.json"
        if canonical_fingerprint(model) != preflight_result["canonical_fingerprint"]:
            raise ValueError("Actual model differs from preflight parameters")
        if model.policy.optimizer.state:
            raise ValueError("Optimizer is not fresh")
        model.save(directory / "initial")
        save_json(
            directory / "initialization.json",
            {
                "canonical_fingerprint": canonical_fingerprint(model),
                "rng_fingerprint": rng_fingerprint(),
                "optimizer_empty": not bool(model.policy.optimizer.state),
                "checkpoint_sha256": sha256(directory / "initial.zip"),
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


def compare(output, teacher_path):
    torch.set_num_threads(1)
    output = Path(output)
    p = json.loads((output / "protocol.json").read_text())
    if sha256(teacher_path) != p["teacher_sha256"]:
        raise ValueError("Changed teacher")
    init = {
        a: json.loads((output / a / "initialization.json").read_text())
        for a in p["arms"]
    }
    for key in ["canonical_fingerprint", "rng_fingerprint", "optimizer_empty"]:
        if init["raw"][key] != init["bounded"][key]:
            raise ValueError(f"Unpaired actual initialization:{key}")
    checkpoints = {}
    for arm in p["arms"]:
        initial = DiagnosticPPO.load(output / arm / "initial.zip")
        if (
            sha256(output / arm / "initial.zip") != init[arm]["checkpoint_sha256"]
            or canonical_fingerprint(initial) != init[arm]["canonical_fingerprint"]
        ):
            raise ValueError("Actual initial tensors/hash mismatch")
        for tag in ["final", "best"]:
            model = DiagnosticPPO.load(output / arm / f"{tag}.zip")
            if tag == "final" and model.num_timesteps != 262144:
                raise ValueError("Actual final budget mismatch")
            if (
                tag == "best"
                and model.num_timesteps
                != json.loads((output / arm / "selected.json").read_text())["timesteps"]
            ):
                raise ValueError("Selected checkpoint mismatch")
            checkpoints[f"{arm}_{tag}"] = model
    checkpoints = {"teacher": load_teacher(teacher_path), **checkpoints}
    summaries = {}
    provenance = {}
    for name, model in checkpoints.items():
        result = evaluate(TASK, model, p["test_seeds"], output / f"test_{name}.json")
        summaries[name] = result["summary"]
        if name in ["teacher", "raw_final", "bounded_final"]:
            video = evaluate(
                TASK,
                model,
                p["video_seeds"],
                output / f"video_{name}.json",
                video=True,
                traces=True,
            )
            for a, b in zip(video["episodes"], result["episodes"]):
                if {k: v for k, v in a.items() if k != "trajectory"} != {
                    k: v for k, v in b.items() if k != "trajectory"
                }:
                    raise ValueError("Video mismatch")
        if name != "teacher":
            arm, tag = name.split("_")
            provenance[name] = {
                "checkpoint_sha256": sha256(output / arm / f"{tag}.zip"),
                "timesteps": model.num_timesteps,
            }
    save_json(output / "comparison.json", summaries)
    save_json(
        output / "provenance.json",
        {
            "teacher_sha256": sha256(teacher_path),
            "paired_initialization": True,
            "checkpoints": provenance,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["declare", "preflight", "train", "compare"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--arm", choices=["raw", "bounded"])
    a = parser.parse_args()
    if a.command == "declare":
        declare(a.output, a.teacher)
    elif a.command == "preflight":
        preflight(a.output, a.teacher)
    elif a.command == "train":
        if not a.arm:
            parser.error("train needs --arm")
        train(a.output, a.teacher, a.arm)
    else:
        compare(a.output, a.teacher)


if __name__ == "__main__":
    main()
