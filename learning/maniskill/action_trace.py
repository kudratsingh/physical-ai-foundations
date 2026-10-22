"""Synchronized PickCube action-distribution/state tracing without an override."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .action_distribution import inspect_action_distribution
from .experiment import make_env, save_json, versions
from .metrics import summarize
from .normalized_ppo import preserve_rng, stats_json
from .stopping_diagnostic import state


def evaluate_trace(config, policy, seeds, output=None, video=False):
    """Trace raw or frozen normalized Gaussian/tanh-Gaussian deterministic policies.

    Policy inputs stay unchanged. Each record pairs pre-action latent Gaussian
    parameters and transformed executed command with the resulting physical
    distance, arm speed and official flags. No stopping override is available.
    """
    if video and output is None:
        raise ValueError("Video tracing needs an output path")
    seeds = list(seeds)
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("Need nonempty unique seeds")
    torch.set_num_threads(1)
    before = stats_json(policy.normalizer) if hasattr(policy, "normalizer") else None
    episodes = []
    with preserve_rng():
        env = make_env(config, render=video)
        try:
            for seed in seeds:
                observation, _ = env.reset(seed=seed)
                initial = state(env)
                records, total, writer = [], 0.0, None
                if video:
                    import imageio.v2 as imageio

                    path = Path(output).with_name(f"{Path(output).stem}_seed{seed}.mp4")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    writer = imageio.get_writer(path, fps=env.unwrapped.control_freq)
                try:
                    while True:
                        distribution = inspect_action_distribution(policy, observation)
                        action = np.asarray(
                            distribution["executed_action"],
                            dtype=env.action_space.dtype,
                        )
                        observation, reward, terminated, truncated, info = env.step(
                            action
                        )
                        total += float(reward)
                        records.append(
                            dict(
                                step=len(records) + 1,
                                reward=float(reward),
                                **distribution,
                                **state(env, info),
                            )
                        )
                        if writer:
                            writer.append_data(env.render())
                        if terminated or truncated:
                            break
                finally:
                    if writer:
                        writer.close()
                successes = [r["step"] for r in records if r["success"]]
                episodes.append(
                    dict(
                        seed=seed,
                        return_=total,
                        steps=len(records),
                        success_once=bool(successes),
                        success_at_end=records[-1]["success"],
                        first_success_step=successes[0] if successes else None,
                        initial=initial,
                        records=records,
                    )
                )
                episodes[-1]["return"] = episodes[-1].pop("return_")
        finally:
            env.close()
    if before is not None and stats_json(policy.normalizer) != before:
        raise RuntimeError("Trace updated frozen observation statistics")
    result = dict(
        config=asdict(config),
        seeds=seeds,
        scope="Deterministic policy trace without stopping override; caller owns validation/test protocol",
        summary=summarize(episodes),
        episodes=episodes,
        frozen_statistics_unchanged=True if before is not None else None,
        versions=versions(),
    )
    if output:
        save_json(output, result)
    return result
