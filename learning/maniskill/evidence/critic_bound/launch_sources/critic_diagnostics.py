"""Read-only instrumentation of SB3's actual gradients and optimizer steps."""

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from .experiment import save_json


def statistics(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite critic diagnostic")
    return {
        "count": len(values),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def parameter_group(name):
    if name == "log_std" or name.startswith(
        ("mlp_extractor.policy_net.", "action_net.")
    ):
        return "actor"
    if name.startswith(("mlp_extractor.value_net.", "value_net.")):
        return "critic"
    raise ValueError(f"Unclassified learnable parameter: {name}")


class DiagnosticPPO(PPO):
    """Same optimizer/rollout algorithm; hooks inspect, never replace gradients."""

    def train(self):
        if not hasattr(self, "diagnostics"):
            self.diagnostics = []
        groups = {"actor": [], "critic": []}
        for name, p in self.policy.named_parameters():
            if p.requires_grad:
                groups[parameter_group(name)].append(p)
        point = {
            "timesteps": self.num_timesteps,
            "values": statistics(self.rollout_buffer.values),
            "returns": statistics(self.rollout_buffer.returns),
            "advantages": statistics(self.rollout_buffer.advantages),
            "buffer_rewards_after_bootstrap": statistics(self.rollout_buffer.rewards),
            "raw": getattr(self, "rollout_diagnostics", {}),
            "optimizer_steps": 0,
        }
        before_squares = {"actor": 0.0, "critic": 0.0}
        snapshots = {}

        def gradient_hook(group):
            def inspect(gradient):
                if point["optimizer_steps"] == 0:
                    before_squares[group] += float(
                        gradient.detach().double().square().sum()
                    )

            return inspect

        def before_step(optimizer, args, kwargs):
            if point["optimizer_steps"] == 0:
                point["gradient_norm_before_clip"] = {
                    g: s**0.5 for g, s in before_squares.items()
                }
                point["gradient_norm_after_clip"] = {
                    g: sum(
                        float(p.grad.detach().double().square().sum())
                        for p in ps
                        if p.grad is not None
                    )
                    ** 0.5
                    for g, ps in groups.items()
                }
                point["global_norm_before_clip"] = sum(before_squares.values()) ** 0.5
                snapshots.update(
                    {g: [p.detach().clone() for p in ps] for g, ps in groups.items()}
                )

        def after_step(optimizer, args, kwargs):
            if point["optimizer_steps"] == 0:
                point["first_step_parameter_delta_l2"] = {
                    g: sum(
                        float((p.detach() - before).double().square().sum())
                        for p, before in zip(ps, snapshots[g], strict=True)
                    )
                    ** 0.5
                    for g, ps in groups.items()
                }
            point["optimizer_steps"] += 1

        handles = [
            p.register_hook(gradient_hook(g)) for g, ps in groups.items() for p in ps
        ]
        handles += [
            self.policy.optimizer.register_step_pre_hook(before_step),
            self.policy.optimizer.register_step_post_hook(after_step),
        ]
        try:
            super().train()
            point["optimizer_metrics"] = {
                k: float(v)
                for k, v in self.logger.name_to_value.items()
                if k.startswith("train/")
            }
            if not all(np.isfinite(v) for v in point["optimizer_metrics"].values()):
                raise ValueError("Nonfinite optimizer metric")
            self.diagnostics.append(point)
            if getattr(self, "diagnostics_path", None):
                save_json(
                    self.diagnostics_path,
                    {
                        "scope": "Actual first optimizer step per rollout; read-only hooks. Adam parameter deltas are measured separately from gradient scales.",
                        "updates": self.diagnostics,
                    },
                )
        finally:
            for handle in handles:
                handle.remove()


class RolloutDiagnostics(BaseCallback):
    """Measure rewards before SB3 adds timeout bootstrap corrections."""

    def _on_rollout_start(self):
        self.raw_rewards = []
        self.terminal_values = []
        self.timeout_count = 0
        self.termination_count = 0

    def _on_step(self):
        self.raw_rewards.extend(map(float, self.locals["rewards"]))
        for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
            if done:
                if info.get("TimeLimit.truncated", False):
                    self.timeout_count += 1
                    tensor = self.model.policy.obs_to_tensor(
                        info["terminal_observation"]
                    )[0]
                    with torch.no_grad():
                        value = float(
                            self.model.policy.predict_values(tensor).reshape(-1)[0]
                        )
                    self.terminal_values.append(value)
                else:
                    self.termination_count += 1
        return True

    def _on_rollout_end(self):
        self.model.rollout_diagnostics = {
            "rewards": statistics(self.raw_rewards),
            "timeout_count": self.timeout_count,
            "termination_count": self.termination_count,
            "terminal_values": statistics(self.terminal_values)
            if self.terminal_values
            else None,
        }
