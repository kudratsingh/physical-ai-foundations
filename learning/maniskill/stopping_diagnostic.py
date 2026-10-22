"""Validation-only privileged placement-triggered zero-arm diagnostic.

This is not a repaired PPO policy. Zero delta targets the current measured joint
positions again each control step; it does not latch a fixed joint-space hold.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from .experiment import Config, make_env, save_json, versions
from .metrics import summarize
from .normalized_ppo import FrozenPolicy, preserve_rng, stats_json

TASK = Config(env_id="PickCube-v1")
SEEDS = list(range(80000, 80010))
CONDITIONS = [
    (name, override)
    for name in ["raw_final", "normalized_final"]
    for override in [False, True]
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def controller_contract(env):
    arm = env.unwrapped.agent.controller.controllers["arm"]
    config = arm.config
    if not (config.use_delta and not config.use_target and config.normalize_action):
        raise ValueError("Need normalized measured-state joint-delta controller")
    if config.lower != -0.1 or config.upper != 0.1 or arm.qpos.shape != (1, 7):
        raise ValueError("Unexpected Panda arm delta bounds or dimensions")
    return arm


def policy_action(policy, observation):
    model = getattr(policy, "model", policy)
    normalized = (
        policy.normalizer.normalize_obs(observation)
        if hasattr(policy, "normalizer")
        else observation
    )
    with torch.no_grad():
        tensor, _ = model.policy.obs_to_tensor(normalized)
        distribution = model.policy.get_distribution(tensor).distribution
        mean = distribution.mean.cpu().numpy()[0]
        std = distribution.stddev.cpu().numpy()[0]
    original, _ = policy.predict(observation, deterministic=True)
    if original.shape != (8,) or not np.array_equal(original, np.clip(mean, -1, 1)):
        raise ValueError(
            "Gaussian mean must reproduce eight-dimensional deterministic action"
        )
    if not (np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()):
        raise ValueError("Need finite Gaussian parameters and positive std")
    return mean, std, original


def state(env, info=None):
    base = env.unwrapped
    cube = base.cube.pose.p[0].cpu().numpy()
    goal = base.goal_site.pose.p[0].cpu().numpy()
    arm_qvel = base.agent.robot.get_qvel()[0, :7].cpu().numpy()
    if info is None:
        info = base.evaluate()
    value = dict(
        cube=cube.tolist(),
        goal=goal.tolist(),
        arm_qvel=arm_qvel.tolist(),
        cube_goal_distance=float(np.linalg.norm(cube - goal)),
        max_arm_qvel=float(np.abs(arm_qvel).max()),
        placement=bool(info["is_obj_placed"]),
        static=bool(info["is_robot_static"]),
        success=bool(info["success"]),
    )
    if value["success"] != (value["placement"] and value["static"]):
        raise ValueError("Task success contract changed")
    return value


def rollout(policy, seeds, override=False, output=None, video=False):
    """Latch after placement; original gripper is recomputed from the current state."""
    torch.set_num_threads(1)
    if video and output is None:
        raise ValueError("Video requires output path")
    env = make_env(TASK, render=video)
    if env.action_space.shape != (8,):
        env.close()
        raise ValueError("Need seven arm increments and one gripper action")
    episodes = []
    before = stats_json(policy.normalizer) if hasattr(policy, "normalizer") else None
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            arm = controller_contract(env)
            initial = state(env)
            first_placement = 0 if initial["placement"] else None
            records, total, writer = [], 0.0, None
            if video:
                import imageio.v2 as imageio

                path = Path(output).with_name(f"{Path(output).stem}_seed{seed}.mp4")
                path.parent.mkdir(parents=True, exist_ok=True)
                writer = imageio.get_writer(path, fps=env.unwrapped.control_freq)
            try:
                for step in range(1, 51):
                    mean, std, original = policy_action(policy, obs)
                    active = override and first_placement is not None
                    executed = original.copy()
                    if active:
                        executed[:7] = 0
                    pre_arm_qpos = arm.qpos[0].cpu().numpy().copy()
                    obs, reward, terminated, truncated, info = env.step(executed)
                    total += float(reward)
                    current = state(env, info)
                    target = arm._target_qpos[0].cpu().numpy().copy()
                    if active and not np.allclose(
                        target, pre_arm_qpos, rtol=0, atol=1e-7
                    ):
                        raise ValueError(
                            "Zero increment did not target current measured qpos"
                        )
                    if current["placement"] and first_placement is None:
                        first_placement = step
                    records.append(
                        dict(
                            step=step,
                            gaussian_mean=mean.tolist(),
                            gaussian_std=std.tolist(),
                            original_action=original.tolist(),
                            executed_action=executed.tolist(),
                            override_active=bool(active),
                            pre_arm_qpos=pre_arm_qpos.tolist(),
                            arm_target=target.tolist(),
                            reward=float(reward),
                            **current,
                        )
                    )
                    if writer:
                        writer.append_data(env.render())
                    if terminated or truncated:
                        if step != 50:
                            raise ValueError(
                                "Diagnostic must run full50-step task horizon"
                            )
                        break
            finally:
                if writer:
                    writer.close()
            successes = [r["step"] for r in records if r["success"]]
            activated = [r for r in records if r["override_active"]]
            episodes.append(
                dict(
                    seed=seed,
                    return_=total,
                    steps=len(records),
                    success_once=bool(successes),
                    success_at_end=records[-1]["success"],
                    first_success_step=successes[0] if successes else None,
                    first_placement_step=first_placement,
                    first_override_action_step=activated[0]["step"]
                    if activated
                    else None,
                    active_steps=len(activated),
                    post_activation_success=any(r["success"] for r in activated),
                    initial=initial,
                    records=records,
                )
            )
            episodes[-1]["return"] = episodes[-1].pop("return_")
            print(
                f"seed={seed} override={override} placed={first_placement} active_steps={len(activated)} any={bool(successes)} final={records[-1]['success']}",
                flush=True,
            )
    finally:
        env.close()
    if before is not None and stats_json(policy.normalizer) != before:
        raise RuntimeError("Diagnostic changed frozen statistics")
    activated = [e for e in episodes if e["active_steps"]]
    result = dict(
        scope="Privileged validation diagnostic; not autonomous PPO success or final test evidence",
        override=override,
        seeds=list(seeds),
        summary=summarize(episodes),
        episodes=episodes,
        activation=dict(
            episodes=len(activated),
            total_active_steps=sum(e["active_steps"] for e in episodes),
            post_activation_any_success=sum(
                e["post_activation_success"] for e in activated
            ),
            activated_final_success=sum(e["success_at_end"] for e in activated),
        ),
        frozen_statistics_unchanged=True if before is not None else None,
    )
    if output:
        save_json(output, result)
    return result


def declare(output, raw_checkpoint, normalized_snapshot):
    """Write a complete immutable protocol before any diagnostic rollout."""
    # Importing confirms the exact installed source without launching an episode.
    from mani_skill.agents.controllers.pd_joint_pos import PDJointPosController
    from mani_skill.agents.robots.panda.panda import Panda

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save_json(
        output / "protocol.json",
        dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            validation_seeds=SEEDS,
            video_seeds=SEEDS[:2],
            conditions=[
                f"{name}_{'override' if override else 'normal'}"
                for name, override in CONDITIONS
            ],
            task=asdict(TASK),
            horizon=50,
            training=False,
            final_test_seeds_used=False,
            trigger="First placed state at reset0 or after controlstep t latches zero seven-arm increments from next action t+1 through episode end; recompute original policy gripper command every step",
            semantics="Zero delta recomputes target=current measured qpos each step, NOT a guaranteed fixed hold. No added damping, fixed pose target, reward or physics modification",
            success="Official3D cube-goal distance<=0.025m AND max absolute seven-arm speeds<=0.2rad/s; privileged override outcomes reported separately from normal policy",
            source_sha256=sha256(__file__),
            raw_checkpoint_sha256=sha256(raw_checkpoint),
            normalized_model_sha256=sha256(Path(normalized_snapshot) / "model.zip"),
            normalized_statistics_sha256=sha256(
                Path(normalized_snapshot) / "vecnormalize.pkl"
            ),
            normalized_snapshot=json.loads(
                (Path(normalized_snapshot) / "snapshot.json").read_text()
            ),
            controller_source_sha256=sha256(inspect.getfile(PDJointPosController)),
            panda_source_sha256=sha256(inspect.getfile(Panda)),
            versions=versions(),
            interpretation="Post-training validation-only privileged diagnostic, not repaired PPO. No tuning or final test evaluation. If no activation, report zero activations without forced resets.",
        ),
    )


def run(output, raw_checkpoint, normalized_snapshot):
    from stable_baselines3 import PPO

    output = Path(output)
    protocol = json.loads((output / "protocol.json").read_text())
    checks = [
        (protocol["source_sha256"], __file__),
        (protocol["raw_checkpoint_sha256"], raw_checkpoint),
        (protocol["normalized_model_sha256"], Path(normalized_snapshot) / "model.zip"),
        (
            protocol["normalized_statistics_sha256"],
            Path(normalized_snapshot) / "vecnormalize.pkl",
        ),
    ]
    if any(expected != sha256(path) for expected, path in checks):
        raise ValueError("Protocol source/checkpoint hash mismatch")
    if any(
        (output / f"{name}_{'override' if override else 'normal'}.json").exists()
        for name, override in CONDITIONS
    ):
        raise ValueError("Diagnostic outputs already exist")
    with preserve_rng():
        raw = PPO.load(raw_checkpoint, device="cpu")
        normalized = FrozenPolicy.load(normalized_snapshot)
        policies = dict(raw_final=raw, normalized_final=normalized)
        try:
            results = {}
            for name, override in CONDITIONS:
                key = f"{name}_{'override' if override else 'normal'}"
                results[key] = rollout(
                    policies[name], SEEDS, override, output / f"{key}.json"
                )
                video_result = rollout(
                    policies[name],
                    SEEDS[:2],
                    override,
                    output / f"video_{key}.json",
                    video=True,
                )
                if video_result["episodes"] != results[key]["episodes"][:2]:
                    raise RuntimeError(
                        "Native video rollout differs from recorded diagnostic"
                    )
            prior_path = (
                Path(__file__).parent
                / "evidence/pickcube_normalization/action_saturation.json"
            )
            prior = json.loads(prior_path.read_text())
            equality = {}
            for name in policies:
                means = [
                    r["gaussian_mean"]
                    for e in results[f"{name}_normal"]["episodes"]
                    for r in e["records"]
                ]
                equality[name] = (
                    prior[name]["seeds"] == SEEDS
                    and prior[name]["preclip_action_means"] == means
                )
                if not equality[name]:
                    raise RuntimeError(
                        "Ordinary rollout changed from earlier validation action audit"
                    )
            save_json(
                output / "baseline_equality.json",
                dict(
                    prior_action_audit_sha256=sha256(prior_path),
                    all500_preclip_means_match=equality,
                ),
            )
            save_json(
                output / "comparison.json",
                {
                    key: dict(summary=value["summary"], activation=value["activation"])
                    for key, value in results.items()
                },
            )
        finally:
            normalized.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["declare", "run"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--raw-checkpoint", required=True, type=Path)
    parser.add_argument("--normalized-snapshot", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "declare":
        declare(args.output, args.raw_checkpoint, args.normalized_snapshot)
    else:
        run(args.output, args.raw_checkpoint, args.normalized_snapshot)


if __name__ == "__main__":
    main()
