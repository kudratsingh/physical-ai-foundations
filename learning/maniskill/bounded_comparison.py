"""Verify loaded recipe and source contracts before the bounded PPO comparison."""

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.distributions import (
    DiagGaussianDistribution,
    SquashedDiagGaussianDistribution,
)
from .bounded_ppo import compare
from .experiment import save_json
from .normalization_comparison import policy_contract
from .normalized_ppo import sha256


def recipe(model):
    return dict(
        rollout_steps=model.n_steps,
        batch_size=model.batch_size,
        epochs=model.n_epochs,
        learning_rate=model.learning_rate,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        clip_range=model.clip_range(1.0),
        entropy_coefficient=model.ent_coef,
        value_coefficient=model.vf_coef,
        max_grad_norm=model.max_grad_norm,
        normalize_advantage=model.normalize_advantage,
        use_sde=model.use_sde,
        net_arch=model.policy.net_arch,
        activation=type(model.policy.activation_fn()).__name__,
        optimizer=type(model.policy.optimizer).__name__,
        optimizer_epsilon=model.policy.optimizer.defaults["eps"],
    )


def compare_checked(output, raw_checkpoint):
    output, raw_checkpoint = Path(output), Path(raw_checkpoint)
    raw = PPO.load(raw_checkpoint, device="cpu")
    bounded = PPO.load(output / "final.zip", device="cpu")
    if raw_checkpoint.name != "final.zip":
        raise ValueError(
            "Primary control must be the original fixed-budget final checkpoint"
        )
    if type(raw.policy.action_dist) is not DiagGaussianDistribution or not isinstance(
        bounded.policy.action_dist, SquashedDiagGaussianDistribution
    ):
        raise ValueError("Expected ordinary versus tanh-transformed Gaussian policies")
    contracts = {
        name: policy_contract(model)
        for name, model in [("raw_final", raw), ("bounded_final", bounded)]
    }
    settings = {
        name: recipe(model)
        for name, model in [("raw_final", raw), ("bounded_final", bounded)]
    }
    if settings["raw_final"] != settings["bounded_final"]:
        raise ValueError(
            "Loaded PPO training recipes differ beyond action distribution"
        )
    saved = json.loads((output / "config.json").read_text())
    if saved["source_sha256"] != sha256(Path(__file__).with_name("bounded_ppo.py")):
        raise ValueError(
            "Training source changed since launch; preserve exact source before comparing"
        )
    checkpoints = {
        "raw_final": sha256(raw_checkpoint),
        "bounded_final": sha256(output / "final.zip"),
    }
    compare(output, raw_checkpoint)
    if checkpoints != {
        "raw_final": sha256(raw_checkpoint),
        "bounded_final": sha256(output / "final.zip"),
    }:
        raise ValueError("A primary checkpoint changed during evaluation")
    save_json(
        output / "loaded_recipe_contracts.json",
        dict(
            contracts=contracts,
            recipes=settings,
            checkpoint_sha256=checkpoints,
            source_sha256=sha256(__file__),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--raw-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    compare_checked(args.run, args.raw_checkpoint)


if __name__ == "__main__":
    main()
