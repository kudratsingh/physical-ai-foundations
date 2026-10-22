"""Task-specific geometry, state dimensions and portable clone checkpoints."""

import json
import h5py
import numpy as np
import pytest
import torch
from learning.maniskill.bc_data import load_episodes
from learning.maniskill.behavior_cloning import BehaviorPolicy
from learning.maniskill.experiment import Config, evaluate, make_env, trajectory_sample
from learning.maniskill.pickcube import failure_counts


def test_pickcube_real_success_requires_3d_placement_and_static_arm():
    env = make_env(Config(env_id="PickCube-v1"))
    try:
        obs, _ = env.reset(seed=40000)
        assert obs.shape == (42,)
        assert env.action_space.shape == (8,)
        base = env.unwrapped
        state = trajectory_sample(env, 0)
        assert state["cube"] == base.cube.pose.p[0].tolist()
        assert state["goal"] == base.goal_site.pose.p[0].tolist()
        # Move the goal rather than teleporting the dynamic cube through contacts.
        pose = base.goal_site.pose.raw_pose.clone()
        pose[:, :3] = base.cube.pose.p
        from mani_skill.utils.structs import Pose

        base.goal_site.set_pose(Pose.create(pose))
        velocity = torch.zeros_like(base.agent.robot.get_qvel())
        base.agent.robot.set_qvel(velocity)
        assert bool(base.evaluate()["success"])
        # Same xy, wrong z: the former PushCube xy criterion must not pass.
        pose[:, 2] += 0.05
        base.goal_site.set_pose(Pose.create(pose))
        assert not bool(base.evaluate()["success"])
        pose[:, :3] = base.cube.pose.p
        base.goal_site.set_pose(Pose.create(pose))
        velocity[:, 0] = 0.3
        base.agent.robot.set_qvel(velocity)
        info = base.evaluate()
        assert bool(info["is_obj_placed"])
        assert not bool(info["is_robot_static"])
        assert not bool(info["success"])
    finally:
        env.close()


def test_pickcube_bounded_clone_and_full_horizon():
    policy = BehaviorPolicy(obs_dim=42)
    result = evaluate(Config(env_id="PickCube-v1"), policy, [50000], traces=True)
    episode = result["episodes"][0]
    assert episode["steps"] == 50
    assert len(episode["trajectory"]) == 51
    assert len(episode["final_arm_qvel"]) == 7
    placed = episode["final_goal_distance"] <= 0.025
    static = max(abs(v) for v in episode["final_arm_qvel"]) <= 0.2
    assert episode["success_at_end"] == (placed and static)


def test_checkpoint_dimensions_and_legacy_schema(tmp_path):
    for dimension in [35, 42]:
        policy = BehaviorPolicy(obs_dim=dimension)
        path = tmp_path / f"policy{dimension}.pt"
        policy.save(path)
        restored = BehaviorPolicy.load(path)
        observation = np.ones(dimension, dtype=np.float32)
        assert np.array_equal(
            policy.predict(observation)[0], restored.predict(observation)[0]
        )
    legacy = tmp_path / "legacy.pt"
    policy = BehaviorPolicy()
    torch.save(dict(schema=1, state_dict=policy.state_dict()), legacy)
    assert len(BehaviorPolicy.load(legacy).mean) == 35


def test_loader_rejects_wrong_task_and_accepts_pickcube_alignment(tmp_path):
    path = tmp_path / "pick.h5"
    metadata = dict(
        env_info=dict(
            env_id="PickCube-v1",
            env_kwargs=dict(
                obs_mode="state",
                control_mode="pd_joint_delta_pos",
                sim_backend="physx_cpu",
            ),
        ),
        episodes=[dict(episode_id=0, episode_seed=123, success=True)],
    )
    with h5py.File(path, "w") as data:
        group = data.create_group("traj_0")
        group["obs"] = np.zeros((3, 42), dtype=np.float32)
        group["actions"] = np.ones((2, 8), dtype=np.float32)
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="PushCube"):
        load_episodes(path)
    assert load_episodes(path, env_id="PickCube-v1")[0]["obs"].shape == (2, 42)


def test_failure_categories_are_exclusive_and_do_not_count_a_lift_as_success():
    episode = dict(
        success_at_end=False,
        success_once=False,
        placed_once=False,
        grasped_once=False,
        lifted_5cm=False,
    )
    cases = [
        episode,
        {**episode, "grasped_once": True},
        {**episode, "grasped_once": True, "lifted_5cm": True},
        {**episode, "placed_once": True},
        {**episode, "success_once": True},
        {**episode, "success_at_end": True},
    ]
    assert set(failure_counts(cases).values()) == {1}
