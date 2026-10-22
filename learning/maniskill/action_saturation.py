"""Audit deterministic PPO action means before the controller-bound clip.

PPO probability-ratio clipping and physical action clipping are different.
This diagnostic uses validation seeds only and never changes policy parameters.
"""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .experiment import Config, make_env, save_json


def audit(policy, seeds):
    """Report per-joint mean saturation under ordinary deterministic rollouts."""
    seeds = list(seeds)
    torch.set_num_threads(1)
    env = make_env(Config(env_id="PickCube-v1"))
    means = []
    model = getattr(policy, "model", policy)
    try:
        if env.action_space.shape != (8,) or not (
            np.all(env.action_space.low == -1) and np.all(env.action_space.high == 1)
        ):
            raise ValueError("This audit requires eight actions bounded by [-1, 1]")
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            while True:
                action, _ = policy.predict(observation, deterministic=True)
                normalized = (
                    policy.normalizer.normalize_obs(observation)
                    if hasattr(policy, "normalizer")
                    else observation
                )
                with torch.no_grad():
                    tensor, _ = model.policy.obs_to_tensor(normalized)
                    mean = (
                        model.policy.get_distribution(tensor)
                        .distribution.mean.cpu()
                        .numpy()[0]
                    )
                if not np.array_equal(
                    action, np.clip(mean, env.action_space.low, env.action_space.high)
                ):
                    raise ValueError(
                        "Recorded distribution mean does not reproduce deterministic action"
                    )
                means.append(mean)
                observation, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
    finally:
        env.close()
    values = np.asarray(means)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Need finite action means from at least one complete episode")
    arm = values[:, :7]
    return dict(
        seeds=list(seeds),
        control_steps=len(values),
        preclip_action_means=values.tolist(),
        arm_any_mean_outside_bounds_fraction=float((abs(arm) > 1).any(1).mean()),
        arm_scalar_mean_outside_bounds_fraction=float((abs(arm) > 1).mean()),
        arm_mean_outside_bounds_fraction_per_joint=(abs(arm) > 1).mean(0).tolist(),
        median_absolute_mean_per_arm_joint=np.median(abs(arm), axis=0).tolist(),
        gripper_mean_outside_bounds_fraction=float((abs(values[:, 7]) > 1).mean()),
        interpretation="Diagnostic, not causal proof; pre-clip deterministic Gaussian mean compared with normalized action bounds; gripper excluded from arm rates",
    )


def main():
    from stable_baselines3 import PPO
    from .normalized_ppo import FrozenPolicy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-checkpoint", required=True, type=Path)
    parser.add_argument("--normalized-snapshot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    seeds = list(range(80000, 80010))
    raw = PPO.load(args.raw_checkpoint, device="cpu")
    normalized = FrozenPolicy.load(args.normalized_snapshot)
    try:
        results = dict(
            raw_final=audit(raw, seeds), normalized_final=audit(normalized, seeds)
        )
        save_json(
            args.output,
            dict(
                protocol=dict(
                    validation_seeds=seeds,
                    raw_checkpoint_sha256=hashlib.sha256(
                        args.raw_checkpoint.read_bytes()
                    ).hexdigest(),
                    normalized_snapshot=json.loads(
                        (args.normalized_snapshot / "snapshot.json").read_text()
                    ),
                    source_sha256=hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                    scope="post-training validation-only diagnostic; not a causal action-distribution experiment",
                ),
                **results,
            ),
        )
        print(results)
    finally:
        normalized.close()


if __name__ == "__main__":
    main()
