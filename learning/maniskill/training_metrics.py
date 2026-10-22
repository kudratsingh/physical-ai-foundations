"""Stochastic training diagnostics; validation never contributes these counts."""

import math
from pathlib import Path
import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from .experiment import make_env, save_json


class InstrumentedTask(gym.Wrapper):
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        base = self.env.unwrapped
        distance = float(
            np.linalg.norm((base.cube.pose.p - base.goal_site.pose.p)[0].cpu().numpy())
        )
        speed = float(base.agent.robot.get_qvel()[0, :7].abs().max())
        command = float(np.max(np.abs(action[:7])))
        info = dict(
            info,
            training_diagnostics=dict(
                distance=distance,
                arm_speed=speed,
                grasped=bool(info["is_grasped"]),
                placed=bool(info["is_obj_placed"]),
                success=bool(info["success"]),
                static=speed <= 0.2,
                command_max=command,
            ),
        )
        return observation, reward, terminated, truncated, info


def training_env(config):
    """Picklable worker constructor, with identical simulation and observations."""

    def create():
        import torch
        from stable_baselines3.common.monitor import Monitor

        torch.set_num_threads(1)
        return Monitor(InstrumentedTask(make_env(config)))

    return create


class TrainingMetrics(BaseCallback):
    def __init__(self, output, interval=16384):
        super().__init__()
        self.output = Path(output)
        self.interval = interval
        self.next = interval
        self.points = []
        self.window = {}
        self.any_success = None

    def _on_step(self):
        infos = self.locals["infos"]
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        actions = np.asarray(self.locals["actions"])
        if self.any_success is None:
            self.any_success = [False] * len(infos)
        for i, (info, reward, done) in enumerate(
            zip(infos, rewards, dones, strict=True)
        ):
            d = info["training_diagnostics"]
            if not all(
                math.isfinite(d[k]) for k in ["distance", "arm_speed", "command_max"]
            ):
                raise ValueError("Non-finite training diagnostic")
            fields = dict(
                transitions=1,
                reward_sum=float(reward),
                distance_sum=d["distance"],
                arm_speed_sum=d["arm_speed"],
                grasped=d["grasped"],
                placed=d["placed"],
                success=d["success"],
                static=d["static"],
                executed_arm_near_boundary=d["command_max"] >= 0.99,
                sampled_arm_outside_bounds=bool((np.abs(actions[i, :7]) > 1).any()),
            )
            for threshold, label in [(0.1, "10cm"), (0.05, "5cm"), (0.025, "2_5cm")]:
                fields[f"grasped_within_{label}"] = (
                    d["grasped"] and d["distance"] <= threshold
                )
            self.any_success[i] |= d["success"]
            if done:
                fields.update(
                    episodes=1,
                    episodes_any_success=self.any_success[i],
                    episodes_final_success=d["success"],
                )
                self.any_success[i] = False
            for key, value in fields.items():
                self.window[key] = self.window.get(key, 0) + value
        if self.num_timesteps >= self.next:
            self.flush()
            self.next += self.interval
        return True

    def flush(self):
        if not self.window:
            return
        self.points.append(dict(timesteps=self.num_timesteps, **self.window))
        save_json(
            self.output,
            dict(
                scope="Stochastic on-policy training transitions only; excludes all validation and test rollouts",
                windows=self.points,
            ),
        )
        self.window = {}

    def _on_training_end(self):
        self.flush()
