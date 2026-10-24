"""Predeclared fixed lambda10-versus-lambda25 retention-strength ablation.

Reuse the exact PR18 learner, worker factory and passive telemetry. This study
changes the positive auxiliary coefficient only; no calibration or curriculum.
"""

import argparse
import json
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from .bc_initialized_ppo import rng_fingerprint
from .bc_ppo_policy import load_teacher
from .critic_diagnostics import RolloutDiagnostics
from .critic_experiment import canonical_fingerprint
from .experiment import evaluate, save_json, versions
from .normalized_ppo import sha256
from .retention_experiment import SOURCES as RETENTION_SOURCES
from .retention_experiment import TASK as RETENTION_TASK
from .retention_experiment import build, tensor_fingerprint, workers
from .selection import holding_score
from .training_metrics import TrainingMetrics

SEEDS = [42, 1042, 2042]
COEFFICIENTS = {"lambda10": 10.0, "lambda25": 25.0}
ARMS = list(COEFFICIENTS)
TASK = replace(RETENTION_TASK, test_seed_start=250000)
SOURCES = ["strength_experiment.py", "STRENGTH_PROTOCOL.md", *RETENTION_SOURCES]


def read(path):
    return json.loads(Path(path).read_text())


def check_sources(output, teacher):
    output = Path(output)
    p = read(output / "protocol.json")
    if read(output / "protocol_frozen.json")["sha256"] != sha256(
        output / "protocol.json"
    ):
        raise ValueError("Declared protocol changed")
    if (
        p["coefficients"] != COEFFICIENTS
        or p["arms"] != ARMS
        or p["training_seeds"] != SEEDS
        or p["config"] != asdict(TASK)
        or p["validation_seeds"] != list(range(230000, 230040))
        or p["test_seeds"] != list(range(250000, 250050))
        or p["video_seeds"] != [250000, 250001]
    ):
        raise ValueError("Undeclared coefficient, recipe or reset partition")
    if sha256(teacher) != p["teacher_sha256"]:
        raise ValueError("Teacher changed")
    if set(p["sources"]) != set(SOURCES):
        raise ValueError("Incomplete frozen launch sources")
    for name, digest in p["sources"].items():
        if (
            sha256(Path(__file__).with_name(name)) != digest
            or sha256(output / "launch_sources" / name) != digest
        ):
            raise ValueError(f"Frozen source changed: {name}")
    return p


def declare(output, teacher, reference):
    output, reference = Path(output), Path(reference)
    # Verify the historical reference before creating a new run. Its selected
    # artifacts are preserved, never used as initialization or supervision.
    historical = read(reference / "protocol.json")
    if historical["teacher_sha256"] != sha256(teacher):
        raise ValueError("PR18 reference teacher differs")
    if historical["test_seeds"] != list(range(240000, 240050)):
        raise ValueError("Unexpected PR18 test partition")
    if set(historical["test_seeds"]) & set(range(250000, 250050)):
        raise ValueError("Fresh test seeds overlap PR18")
    entries = []
    for seed in SEEDS:
        source = reference / f"seed{seed}" / "anchored"
        selected = read(source / "selected.json")
        if selected != max(read(source / "validation.json"), key=holding_score):
            raise ValueError("PR18 selected checkpoint metadata differs")
        entries.append(
            {
                "training_seed": seed,
                "selected": selected,
                "checkpoint_sha256": sha256(source / "best.zip"),
            }
        )
    output.mkdir(parents=True, exist_ok=False)
    save_json(
        output / "protocol.json",
        {
            "config": asdict(TASK),
            "training_seeds": SEEDS,
            "arms": ARMS,
            "coefficients": COEFFICIENTS,
            "teacher_sha256": sha256(teacher),
            "versions": versions(),
            "sources": {n: sha256(Path(__file__).with_name(n)) for n in SOURCES},
            "validation_seeds": list(range(230000, 230040)),
            "test_seeds": list(range(250000, 250050)),
            "video_seeds": [250000, 250001],
            "coefficient_selection": "Fixed before performance training:10 is PR18 reference;25 is motivated by PR18 uncapped calibration22.73. No new calibration, coefficient search or held-out tuning.",
            "primary": "Fixed-budget final success for each paired training seed, at262144transitions; original50-step PickCube task.",
            "secondary": "40-reset validation-selected checkpoint, final-success thenany-step thenreturn; step0eligible and selection before boundary optimizerupdate.",
            "promising": "lambda25 final improves predeclared problematic seed2042 by>=5/50 versuslambda10; loses<=2/50 in EACH otherseed; and mean signed(selected-final)collapse <=lambda10. Descriptive rule, not significance.",
            "target_seed_rationale": "Seed2042 is explicitly targeted from PR18's observed retention failure; no post-hoc seed choice within this study.",
            "replication": "All six conditions run irrespective of outcomes, same paired initialization/workerstreams beforefirstupdate, no adaptive budget or stopping.",
            "scope": "Same three trainingseeds and existing40validation resets deliberately retained to isolate coefficient. New50heldout resetlist shared acrossseeds, not150independentresettrials. One fixedBCteacher andshortbudget only.",
            "holding": "Passive post-first-success/immediate-success command/speed/teacherdrift/gripper/grasp traces; inspect every successful-then-displaced finalcase. No intervention or controller override.",
            "historical_reference": {
                "commit": "e88cb5c26295fab67b3a45c74ef8b898a3608a2f",
                "protocol_sha256": sha256(reference / "protocol.json"),
                "selected_models": entries,
            },
        },
    )
    save_json(
        output / "protocol_frozen.json", {"sha256": sha256(output / "protocol.json")}
    )
    archive = output / "launch_sources"
    archive.mkdir()
    for name in SOURCES:
        shutil.copy2(Path(__file__).with_name(name), archive / name)
    dest = output / "reference_pr18_selected"
    dest.mkdir()
    for entry in entries:
        seed = entry["training_seed"]
        for filename in ["best.zip", "selected.json"]:
            shutil.copy2(
                reference / f"seed{seed}" / "anchored" / filename,
                dest / f"seed{seed}_{filename}",
            )
    save_json(
        dest / "manifest.json",
        {
            "reference_commit": historical.get("commit", "e88cb5c"),
            "models": entries,
            "evaluation_in_this_study": False,
        },
    )


def train(output, teacher_path, seed, arm):
    torch.set_num_threads(1)
    output = Path(output)
    p = check_sources(output, teacher_path)
    if seed not in SEEDS or arm not in ARMS:
        raise ValueError("Undeclared training condition")
    cfg = replace(TASK, seed=seed)
    weight = p["coefficients"][arm]
    directory = output / f"seed{seed}" / arm
    directory.mkdir(parents=True, exist_ok=False)
    env = workers(cfg)

    class Validate(BaseCallback):
        def __init__(self):
            super().__init__()
            self.rows, self.next, self.best = [], cfg.validation_interval, None

        def assess(self):
            result = evaluate(cfg, self.model, p["validation_seeds"])
            point = dict(timesteps=self.num_timesteps, **result["summary"])
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
                self.next += cfg.validation_interval
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
                "protocol_sha256": sha256(output / "protocol.json"),
            },
        )
        start = time.monotonic()
        model.learn(
            cfg.total_timesteps,
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
            raise ValueError("Frozen teacher updated")
        if model.num_timesteps != cfg.total_timesteps:
            raise ValueError("Actual training budget differs")
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["declare", "train"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.command == "declare":
        if args.reference is None:
            parser.error("declare requires --reference")
        declare(args.output, args.teacher, args.reference)
    else:
        if args.arm is None:
            parser.error("train requires --arm")
        train(args.output, args.teacher, args.seed, args.arm)


if __name__ == "__main__":
    main()
