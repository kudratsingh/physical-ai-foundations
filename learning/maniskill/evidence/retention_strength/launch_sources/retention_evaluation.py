"""Unmodified-policy evaluation with passive teacher and holding diagnostics.

Geometric simulator state and the separate teacher are diagnostics only. Neither
can alter, replace or gate the policy's executed command.
"""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .experiment import make_env, save_json, trajectory_sample, versions
from .metrics import summarize
from .rng import isolated_rng


def geometric_state(env, step, info):
    sample = trajectory_sample(env, step)
    distance = float(np.linalg.norm(np.asarray(sample["cube"]) - sample["goal"]))
    arm_qvel = sample["qvel"][:7]
    speed = max(map(abs, arm_qvel))
    return {
        "tcp": sample["tcp"],
        "cube": sample["cube"],
        "goal": sample["goal"],
        "arm_qvel": arm_qvel,
        "gripper_qpos": sample["qpos"][-2:],
        "cube_goal_distance": distance,
        "max_arm_speed": speed,
        "placed": distance <= 0.025,
        "static": speed <= 0.2,
        "geometric_success": distance <= 0.025 and speed <= 0.2,
        "success": bool(info.get("success", False)),
        "grasped": bool(info.get("is_grasped", False)),
        "official_placed": bool(info.get("is_obj_placed", False)),
        "official_static": bool(info.get("is_robot_static", False)),
    }


def mean_or_none(values):
    return float(np.mean(values)) if values else None


def retention_summary(records, episodes):
    """Actions after first success exclude the command that first produced it."""
    after = [r for r in records if r["after_first_success"]]
    transitions = [
        r for r in records if r["pre"]["success"] and not r["post"]["success"]
    ]
    geometry_mismatches = sum(
        state["success"] != state["geometric_success"]
        or state["placed"] != state["official_placed"]
        or state["static"] != state["official_static"]
        for r in records
        for state in [r["pre"], r["post"]]
    )
    return {
        "scope": "Passive deterministic evaluation; no command override. After-first-success means steps strictly after the first post-step success; later lost states remain included.",
        "episodes": len(episodes),
        "steps": len(records),
        "episodes_ever_successful": sum(e["success_once"] for e in episodes),
        "episodes_with_post_success_actions": len({r["seed"] for r in after}),
        "post_first_success_steps": len(after),
        "post_first_success_arm_command_l2_mean": mean_or_none(
            [float(np.linalg.norm(r["command"][:7])) for r in after]
        ),
        "post_first_success_arm_command_max_abs_mean": mean_or_none(
            [max(map(abs, r["command"][:7])) for r in after]
        ),
        "post_first_success_gripper_command_abs_mean": mean_or_none(
            [abs(r["command"][7]) for r in after]
        ),
        "post_first_success_gripper_motion_l2_mean": mean_or_none(
            [float(np.linalg.norm(r["gripper_delta"])) for r in after]
        ),
        "post_first_success_max_arm_speed_mean": mean_or_none(
            [r["post"]["max_arm_speed"] for r in after]
        ),
        "post_first_success_success_occupancy": mean_or_none(
            [r["post"]["success"] for r in after]
        ),
        "post_first_success_placement_occupancy": mean_or_none(
            [r["post"]["placed"] for r in after]
        ),
        "post_first_success_teacher_arm_drift_l2_mean": mean_or_none(
            [r["teacher_arm_drift_l2"] for r in after]
        ),
        "post_first_success_teacher_gripper_drift_abs_mean": mean_or_none(
            [r["teacher_gripper_drift_abs"] for r in after]
        ),
        "teacher_arm_drift_l2_mean": mean_or_none(
            [r["teacher_arm_drift_l2"] for r in records]
        ),
        "teacher_gripper_drift_abs_mean": mean_or_none(
            [r["teacher_gripper_drift_abs"] for r in records]
        ),
        "success_to_failure_transitions": len(transitions),
        "success_to_failure_while_post_static": sum(
            r["post"]["static"] for r in transitions
        ),
        "ever_successful_but_final_failure_and_static": sum(
            e["success_once"] and not e["success_at_end"] and e["final_is_robot_static"]
            for e in episodes
        ),
        "geometry_official_mismatches": geometry_mismatches,
    }


@isolated_rng
def evaluate_retention(config, policy, teacher, seeds, output=None, video=False):
    """Match experiment.evaluate episodes; add separate per-step audit records.

    ``episodes`` and ``summary`` retain the existing schema exactly (trajectory
    remains empty). ``step_records`` contains ordinary raw observations and
    pre/post simulator geometry, with the separately evaluated teacher command.
    """
    torch.set_num_threads(1)
    if config.env_id != "PickCube-v1":
        raise ValueError("Holding diagnostics require the declared PickCube task")
    if video and output is None:
        raise ValueError("Video evaluation requires an output path")
    seeds = list(seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Need nonempty unique evaluation seeds")
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
    env = make_env(config, render=video)
    episodes, records = [], []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            env.action_space.seed(seed + 1000000)
            info = env.unwrapped.get_info()
            initial = trajectory_sample(env, 0)
            total, success_once, first_success, step = 0.0, False, None, 0
            grasped_once, placed_once = False, False
            maximum_height = initial["cube"][2]
            writer = None
            if video:
                import imageio.v2 as imageio

                path = Path(output).parent / f"{Path(output).stem}_seed{seed}.mp4"
                writer = imageio.get_writer(path, fps=env.unwrapped.control_freq)
            try:
                while True:
                    pre = geometric_state(env, step, info)
                    observation = np.asarray(obs).copy()
                    if observation.shape != (42,) or not np.isfinite(observation).all():
                        raise ValueError(
                            "Need ordinary finite raw 42-component observations"
                        )
                    with torch.no_grad():
                        if isinstance(policy, str):
                            if policy == "random":
                                action = env.action_space.sample()
                            elif policy == "zero":
                                action = np.zeros(
                                    env.action_space.shape, dtype=env.action_space.dtype
                                )
                            else:
                                raise ValueError("Unknown policy label")
                        else:
                            action, _ = policy.predict(
                                observation.copy(), deterministic=True
                            )
                        action = np.asarray(action).copy()
                        teacher_command, _ = teacher.predict(
                            observation.copy(), deterministic=True
                        )
                        teacher_command = np.asarray(teacher_command).copy()
                    if action.shape != (8,) or teacher_command.shape != (8,):
                        raise ValueError(
                            "Need separate eight-component policy and teacher commands"
                        )
                    if (
                        not np.isfinite(action).all()
                        or not np.isfinite(teacher_command).all()
                    ):
                        raise ValueError("Nonfinite command")
                    # Action is fixed before passive teacher/geometry diagnostics.
                    # No correction, gating or substitution is performed.
                    after_first_success = first_success is not None
                    obs, reward, terminated, truncated, info = env.step(action)
                    step += 1
                    total += float(reward)
                    post = geometric_state(env, step, info)
                    records.append(
                        {
                            "seed": seed,
                            "step": step,
                            "observation": observation.tolist(),
                            "pre": pre,
                            "post": post,
                            "command": action.tolist(),
                            "teacher_command": teacher_command.tolist(),
                            "gripper_delta": (
                                np.asarray(post["gripper_qpos"]) - pre["gripper_qpos"]
                            ).tolist(),
                            "teacher_arm_drift_l2": float(
                                np.linalg.norm(action[:7] - teacher_command[:7])
                            ),
                            "teacher_gripper_drift_abs": float(
                                abs(action[7] - teacher_command[7])
                            ),
                            "after_first_success": after_first_success,
                            "raw_reward": float(reward),
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                        }
                    )
                    success = bool(info.get("success", False))
                    if success and first_success is None:
                        first_success = step
                    success_once |= success
                    grasped_once |= bool(info.get("is_grasped", False))
                    placed_once |= bool(info.get("is_obj_placed", False))
                    maximum_height = max(maximum_height, post["cube"][2])
                    if writer:
                        writer.append_data(env.render())
                    if terminated or truncated:
                        break
            finally:
                if writer:
                    writer.close()
            terminal = trajectory_sample(env, step)
            episodes.append(
                {
                    "seed": seed,
                    "return": total,
                    "steps": step,
                    "success_once": success_once,
                    "success_at_end": success,
                    "first_success_step": first_success,
                    "trajectory": [],
                    "final_cube_position": terminal["cube"],
                    "final_goal_position": terminal["goal"],
                    "final_arm_qvel": terminal["qvel"][:-2],
                    "grasped_once": grasped_once,
                    "placed_once": placed_once,
                    "initial_cube_height": initial["cube"][2],
                    "maximum_cube_height": maximum_height,
                    "lifted_5cm": maximum_height >= initial["cube"][2] + 0.05,
                    "final_goal_distance": float(
                        np.linalg.norm(np.array(terminal["cube"]) - terminal["goal"])
                    ),
                    "final_is_obj_placed": bool(info.get("is_obj_placed", False)),
                    "final_is_robot_static": bool(info.get("is_robot_static", False)),
                }
            )
    finally:
        env.close()
    per_episode = {
        str(e["seed"]): retention_summary(
            [r for r in records if r["seed"] == e["seed"]], [e]
        )
        for e in episodes
    }
    result = {
        "policy": policy
        if isinstance(policy, str)
        else getattr(policy, "policy_label", "PPO deterministic"),
        "summary": summarize(episodes),
        "episodes": episodes,
        "config": asdict(config),
        "versions": versions(),
        "step_records": records,
        "retention": retention_summary(records, episodes),
        "episode_retention": per_episode,
    }
    if output:
        save_json(output, result)
    return result
