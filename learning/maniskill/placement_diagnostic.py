"""Predeclared observation-only stop equivalence and validation placement traces.

No training, tuning or final tests. Privileged state is used for audits/metrics,
never to choose the observation-only hybrid's action.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .behavior_cloning import BehaviorPolicy
from .experiment import Config, make_env, versions, save_json
from .metrics import summarize
from .normalized_ppo import FrozenPolicy, preserve_rng, stats_json
from .observation_stop_gate import ObservationStopGate, placed_from_observation
from .stopping_diagnostic import controller_contract, policy_action, state

TASK = Config(env_id="PickCube-v1")
STOP_SEEDS = list(range(80000, 80010))
PLACEMENT_SEEDS = list(range(130000, 130010))
SCHEMA = dict(
    qpos=[0, 9],
    qvel=[9, 18],
    is_grasped=[18, 19],
    tcp_pose=[19, 26],
    goal=[26, 29],
    cube_pose=[29, 36],
    tcp_to_cube=[36, 39],
    cube_to_goal=[39, 42],
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def save_jsonl(path, episodes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(e, separators=(",", ":"), allow_nan=False) + "\n"
            for e in episodes
        )
    )


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def audit_observation(observation, measured, is_grasped):
    """Independent numeric schema check; its outputs never feed hybrid commands."""
    obs = np.asarray(observation)
    if obs.shape != (42,):
        raise ValueError("PickCube state schema must contain42 values")
    for actual, expected in [
        (obs[26:29], measured["goal"]),
        (obs[29:32], measured["cube"]),
        (obs[9:16], measured["arm_qvel"]),
        (
            obs[39:42],
            np.asarray(measured["goal"], dtype=np.float32)
            - np.asarray(measured["cube"], dtype=np.float32),
        ),
    ]:
        if not np.allclose(actual, expected, rtol=0, atol=1e-7):
            raise ValueError(
                "Ordinary state observation schema does not match measured audit"
            )
    if bool(obs[18]) != bool(is_grasped):
        raise ValueError("Observation grasp flag schema mismatch")
    if placed_from_observation(obs) != measured["placement"]:
        raise ValueError("Observation-only placement disagrees with privileged audit")
    return True


def hybrid_rollouts(policy, seeds, observation_checks=None):
    """Observation-only gate commands; privileged state audits equality only."""
    env = make_env(TASK)
    before = stats_json(policy.normalizer) if hasattr(policy, "normalizer") else None
    episodes = []
    try:
        for seed in seeds:
            obs, info = env.reset(seed=seed)
            arm = controller_contract(env)
            initial = state(env)
            audit_observation(obs, initial, info["is_grasped"])
            checks = [
                dict(
                    state_step=0,
                    cube_to_goal=obs[39:42].tolist(),
                    observed_placement=placed_from_observation(obs),
                    privileged_placement=initial["placement"],
                )
            ]
            gate = ObservationStopGate()
            records, total = [], 0.0
            for step in range(1, 51):
                mean, std, original = policy_action(policy, obs)
                # Commands are fixed BEFORE privileged state is consulted.
                executed, active, _ = gate.action(obs, original, step - 1)
                pre_qpos = arm.qpos[0].cpu().numpy().copy()
                obs, reward, terminated, truncated, info = env.step(executed)
                total += float(reward)
                measured = state(env, info)
                audit_observation(obs, measured, info["is_grasped"])
                checks.append(
                    dict(
                        state_step=step,
                        cube_to_goal=obs[39:42].tolist(),
                        observed_placement=placed_from_observation(obs),
                        privileged_placement=measured["placement"],
                    )
                )
                target = arm._target_qpos[0].cpu().numpy().copy()
                records.append(
                    dict(
                        step=step,
                        gaussian_mean=mean.tolist(),
                        gaussian_std=std.tolist(),
                        original_action=original.tolist(),
                        executed_action=executed.tolist(),
                        override_active=bool(active),
                        pre_arm_qpos=pre_qpos.tolist(),
                        arm_target=target.tolist(),
                        reward=float(reward),
                        **measured,
                    )
                )
                if terminated or truncated:
                    if step != 50:
                        raise ValueError("Require full50-step horizon")
                    break
            placements = [0] if initial["placement"] else []
            placements += [r["step"] for r in records if r["placement"]]
            successes = [r["step"] for r in records if r["success"]]
            active = [r for r in records if r["override_active"]]
            episodes.append(
                dict(
                    seed=seed,
                    steps=len(records),
                    success_once=bool(successes),
                    success_at_end=records[-1]["success"],
                    first_success_step=successes[0] if successes else None,
                    first_placement_step=placements[0] if placements else None,
                    first_override_action_step=active[0]["step"] if active else None,
                    active_steps=len(active),
                    post_activation_success=any(r["success"] for r in active),
                    initial=initial,
                    records=records,
                    **{"return": total},
                )
            )
            if observation_checks is not None:
                observation_checks.append(dict(seed=seed, states=checks))
            print(
                f"hybrid seed={seed} active={len(active)} success={bool(successes)}",
                flush=True,
            )
    finally:
        env.close()
    if before is not None and stats_json(policy.normalizer) != before:
        raise RuntimeError("Hybrid evaluation updated frozen statistics")
    return episodes


def placement_rollouts(policy, seeds, video_output=None):
    """Ordinary deterministic policy; trace placement context without overrides."""
    env = make_env(TASK, render=video_output is not None)
    before = stats_json(policy.normalizer) if hasattr(policy, "normalizer") else None
    episodes = []
    try:
        for seed in seeds:
            observation, info = env.reset(seed=seed)
            initial = state(env)
            audit_observation(observation, initial, info["is_grasped"])
            records, total, writer = [], 0.0, None
            if video_output is not None:
                import imageio.v2 as imageio

                path = Path(video_output).with_name(
                    f"{Path(video_output).stem}_seed{seed}.mp4"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                writer = imageio.get_writer(path, fps=env.unwrapped.control_freq)
            try:
                for step in range(1, 51):
                    pre_distance = float(
                        np.linalg.norm(np.asarray(observation[39:42], dtype=np.float32))
                    )
                    action, _ = policy.predict(observation, deterministic=True)
                    action = np.asarray(action, dtype=env.action_space.dtype)
                    if action.shape != (8,) or not np.isfinite(action).all():
                        raise ValueError("Require eight finite actions")
                    observation, reward, terminated, truncated, info = env.step(action)
                    measured = state(env, info)
                    audit_observation(observation, measured, info["is_grasped"])
                    total += float(reward)
                    records.append(
                        dict(
                            step=step,
                            reward=float(reward),
                            action=action.tolist(),
                            arm_command_l2=float(np.linalg.norm(action[:7])),
                            arm_command_max_abs=float(abs(action[:7]).max()),
                            is_grasped=bool(info["is_grasped"]),
                            obs_cube_to_goal=observation[39:42].tolist(),
                            pre_distance=pre_distance,
                            distance_change=measured["cube_goal_distance"]
                            - pre_distance,
                            remaining_steps=50 - step,
                            **measured,
                        )
                    )
                    if writer:
                        writer.append_data(env.render())
                    if terminated or truncated:
                        if step != 50:
                            raise ValueError("Require full50-step horizon")
                        break
            finally:
                if writer:
                    writer.close()
            successes = [r["step"] for r in records if r["success"]]
            episodes.append(
                dict(
                    seed=seed,
                    steps=len(records),
                    success_once=bool(successes),
                    success_at_end=records[-1]["success"],
                    first_success_step=successes[0] if successes else None,
                    initial=initial,
                    records=records,
                    **{"return": total},
                )
            )
            print(
                f"placement seed={seed} any={bool(successes)} final={records[-1]['success']}",
                flush=True,
            )
    finally:
        env.close()
    if before is not None and stats_json(policy.normalizer) != before:
        raise RuntimeError("Placement evaluation updated frozen statistics")
    return episodes


def occupancy_summary(episodes):
    """Pooled grasp-conditioned state occupancy; not independent Bernoulli trials."""
    records = [r for episode in episodes for r in episode["records"]]
    grasped = [r for r in records if r["is_grasped"]]
    stages = {}
    for name, threshold in [("10cm", 0.10), ("5cm", 0.05), ("2_5cm", 0.025)]:
        close = [r for r in grasped if r["cube_goal_distance"] <= threshold]
        first_hits = []
        for episode in episodes:
            hit = next(
                (
                    r["step"]
                    for r in episode["records"]
                    if r["is_grasped"] and r["cube_goal_distance"] <= threshold
                ),
                None,
            )
            if hit is not None:
                first_hits.append(
                    dict(seed=episode["seed"], step=hit, remaining_steps=50 - hit)
                )
        stages[name] = dict(
            grasped_steps_within=len(close),
            grasped_step_denominator=len(grasped),
            conditional_occupancy=len(close) / len(grasped) if grasped else None,
            mean_max_arm_speed=float(np.mean([r["max_arm_qvel"] for r in close]))
            if close
            else None,
            mean_arm_command_l2=float(np.mean([r["arm_command_l2"] for r in close]))
            if close
            else None,
            mean_distance_change=float(np.mean([r["distance_change"] for r in close]))
            if close
            else None,
            episode_first_hits=first_hits,
        )
    return dict(
        episodes=len(episodes),
        steps=len(records),
        grasped_steps=len(grasped),
        official=summarize(episodes),
        stages=stages,
    )


def declare(output, raw_checkpoint, normalized_snapshot, teacher):
    from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
    from mani_skill.envs.sapien_env import BaseEnv
    from mani_skill.agents.base_agent import BaseAgent
    from . import observation_stop_gate

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    old = Path(__file__).parent / "evidence/pickcube_stopping"
    save_json(
        output / "protocol.json",
        dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            source_sha256=sha256(__file__),
            gate_source_sha256=sha256(inspect.getfile(observation_stop_gate)),
            raw_checkpoint_sha256=sha256(raw_checkpoint),
            normalized_model_sha256=sha256(Path(normalized_snapshot) / "model.zip"),
            normalized_statistics_sha256=sha256(
                Path(normalized_snapshot) / "vecnormalize.pkl"
            ),
            teacher_sha256=sha256(teacher),
            task=asdict(TASK),
            schema=SCHEMA,
            installed_schema_source_sha256={
                cls.__name__: sha256(inspect.getfile(cls))
                for cls in [PickCubeEnv, BaseEnv, BaseAgent]
            },
            versions=versions(),
            observation_stop_seeds=STOP_SEEDS,
            placement_seeds=PLACEMENT_SEEDS,
            video_seeds=dict(
                observation_stop=STOP_SEEDS[:2], placement=PLACEMENT_SEEDS[:2]
            ),
            horizon=50,
            training=False,
            final_test_seeds_used=False,
            scope="Validation-only diagnostics. Observation-stop is a hand-coded hybrid, not PPO alone. BC final-success subset is post-hoc conditional inspection, not a performance comparison.",
            gate="Only raw ordinary42 observation, cube_to_goal39:42 norm<=float32(0.025), latched from first observed placed state; zero seven-arm increments and preserve current policy gripper.",
            audit="Privileged placement checked at reset and every post-step state only; never used to choose command. Compare entire hybrid episode dictionaries with prior privileged override, plus preclip baselines already verified there.",
            prior_privileged_hashes={
                f"{name}_override.json": sha256(old / f"{name}_override.json")
                for name in ["raw_final", "normalized_final"]
            },
            placement_conditions=["bc_256", "raw_final", "normalized_final"],
            placement_measurements="All50 post-action states: is_grasped, 10/5/2.5cm grasp-conditioned occupancy, distance changes, speed, arm command magnitude, remainingsteps. All10 episodes per policy; separately labeled BC final-success subset.",
            video_policy="First2 placement seeds rendered per policy and full episodes digest-checked. Observation-stop videos reused only after complete trajectory equality with prior privileged native videos; no duplicated video trajectoryJSON.",
        ),
    )


def run(output, raw_checkpoint, normalized_snapshot, teacher):
    from stable_baselines3 import PPO
    from . import observation_stop_gate

    torch.set_num_threads(1)
    output = Path(output)
    protocol = json.loads((output / "protocol.json").read_text())
    checks = [
        (protocol["source_sha256"], __file__),
        (protocol["gate_source_sha256"], inspect.getfile(observation_stop_gate)),
        (protocol["raw_checkpoint_sha256"], raw_checkpoint),
        (protocol["normalized_model_sha256"], Path(normalized_snapshot) / "model.zip"),
        (
            protocol["normalized_statistics_sha256"],
            Path(normalized_snapshot) / "vecnormalize.pkl",
        ),
        (protocol["teacher_sha256"], teacher),
    ]
    if any(expected != sha256(path) for expected, path in checks):
        raise ValueError("Predeclared source/checkpoint hash changed")
    if (output / "placement_bc_256.jsonl").exists():
        raise ValueError("Refuse to overwrite completed diagnostics")
    old = Path(__file__).parent / "evidence/pickcube_stopping"
    with preserve_rng():
        raw = PPO.load(raw_checkpoint, device="cpu")
        normalized = FrozenPolicy.load(normalized_snapshot)
        bc = BehaviorPolicy.load(teacher)
        models = dict(raw_final=raw, normalized_final=normalized)
        try:
            equality, hybrid_summaries, video_manifest = {}, {}, {}
            for name, model in models.items():
                checks = []
                hybrid = hybrid_rollouts(model, STOP_SEEDS, checks)
                save_jsonl(output / f"gate_checks_{name}.jsonl", checks)
                prior = json.loads((old / f"{name}_override.json").read_text())
                if (
                    sha256(old / f"{name}_override.json")
                    != protocol["prior_privileged_hashes"][f"{name}_override.json"]
                    or hybrid != prior["episodes"]
                ):
                    raise ValueError(
                        "Observation-only hybrid differs from predeclared privileged reference"
                    )
                equality[name] = dict(
                    complete_episode_equality=True,
                    states_audited=510,
                    episodes=10,
                    source_sha256=sha256(old / f"{name}_override.json"),
                )
                save_jsonl(output / f"hybrid_{name}.jsonl", hybrid)
                hybrid_summaries[name] = dict(
                    scope="Observation-only hand-coded hybrid, not PPO alone",
                    official=summarize(hybrid),
                    activated_episodes=sum(e["active_steps"] > 0 for e in hybrid),
                    active_steps=sum(e["active_steps"] for e in hybrid),
                )
                for seed in STOP_SEEDS[:2]:
                    source = old / f"video_{name}_override_seed{seed}.mp4"
                    target = output / f"hybrid_{name}_seed{seed}.mp4"
                    shutil.copy2(source, target)
                    video_manifest[target.name] = dict(
                        scope="Prior native privileged video reused after exact complete observation-only trajectory equality",
                        source_sha256=sha256(source),
                        episode_sha256=canonical_hash(
                            next(e for e in hybrid if e["seed"] == seed)
                        ),
                    )
            summaries = {}
            for name, model in dict(bc_256=bc, **models).items():
                episodes = placement_rollouts(model, PLACEMENT_SEEDS)
                save_jsonl(output / f"placement_{name}.jsonl", episodes)
                summaries[name] = occupancy_summary(episodes)
                videos = placement_rollouts(
                    model, PLACEMENT_SEEDS[:2], output / f"placement_{name}"
                )
                if videos != episodes[:2]:
                    raise ValueError(
                        "Native placement video rerun differs from full trace"
                    )
                for episode in videos:
                    path = output / f"placement_{name}_seed{episode['seed']}.mp4"
                    video_manifest[path.name] = dict(
                        scope="Predeclared native placement validation video; full episode reproduced",
                        video_sha256=sha256(path),
                        episode_sha256=canonical_hash(episode),
                    )
            bc_episodes = load_jsonl(output / "placement_bc_256.jsonl")
            successful = [e for e in bc_episodes if e["success_at_end"]]
            summaries["bc_final_success_subset"] = dict(
                scope="Post-hoc BC final-success conditional subset, not a performance comparison",
                selected_seeds=[e["seed"] for e in successful],
                data=occupancy_summary(successful) if successful else None,
            )
            save_json(output / "hybrid_comparison.json", hybrid_summaries)
            save_json(output / "observation_audit.json", equality)
            save_json(output / "placement_summary.json", summaries)
            save_json(output / "video_manifest.json", video_manifest)
        finally:
            normalized.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["declare", "run"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-checkpoint", type=Path, required=True)
    parser.add_argument("--normalized-snapshot", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    args = parser.parse_args()
    function = declare if args.command == "declare" else run
    function(args.output, args.raw_checkpoint, args.normalized_snapshot, args.teacher)


if __name__ == "__main__":
    main()
