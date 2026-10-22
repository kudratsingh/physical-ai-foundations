"""Check actual policy checkpoint contracts before the normalization comparison."""

import argparse
import json
from pathlib import Path

from .experiment import Config, save_json
from .normalized_ppo import FrozenPolicy, compare, sha256


def policy_contract(model, expected_timesteps=1048576):
    """Inspect loaded weights/spaces, rather than trusting filenames or sidecars."""
    actual = dict(
        timesteps=int(model.num_timesteps),
        observation_shape=list(model.observation_space.shape),
        action_shape=list(model.action_space.shape),
    )
    if actual["timesteps"] != expected_timesteps:
        raise ValueError("Loaded policy does not match the fixed transition budget")
    if actual["observation_shape"] != [42] or actual["action_shape"] != [8]:
        raise ValueError(
            "Loaded policy does not match PickCube state/action dimensions"
        )
    return actual


def compare_checked(output, raw_checkpoint):
    from stable_baselines3 import PPO

    output, raw_checkpoint = Path(output), Path(raw_checkpoint)
    saved = json.loads((output / "config.json").read_text())
    config = Config(**{key: saved[key] for key in Config.__dataclass_fields__})
    raw = PPO.load(raw_checkpoint, device="cpu")
    normalized = FrozenPolicy.load(output / "final", config)
    try:
        contract = {
            "raw_final": dict(
                **policy_contract(raw), checkpoint_sha256=sha256(raw_checkpoint)
            ),
            "normalized_final": dict(
                **policy_contract(normalized.model),
                checkpoint_sha256=sha256(output / "final/model.zip"),
            ),
        }
    finally:
        normalized.close()
    # The predeclared procedure evaluates the very same immutable local files.
    compare(output, raw_checkpoint)
    if (
        sha256(raw_checkpoint) != contract["raw_final"]["checkpoint_sha256"]
        or sha256(output / "final/model.zip")
        != contract["normalized_final"]["checkpoint_sha256"]
    ):
        raise ValueError("Checkpoint changed during comparison")
    save_json(output / "loaded_policy_contracts.json", contract)
    baseline_path = output / "baseline.json"
    baseline = json.loads(baseline_path.read_text())
    save_json(baseline_path, {**baseline, "loaded_contract": contract["raw_final"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    compare_checked(args.output, args.raw_checkpoint)


if __name__ == "__main__":
    main()
