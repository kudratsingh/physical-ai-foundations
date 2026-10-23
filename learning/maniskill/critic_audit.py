"""Existing-checkpoint critic audit on fixed BC-validation observations.

This does not create a simulator, collect a rollout, fit a model or estimate
historical on-policy gradients. Demonstrations are off-policy probes only.
"""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .experiment import save_json, versions
from .rng import isolated_rng


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_probes(dataset, split_protocol, episodes=8):
    """Return (float32[N,42] observations, complete fixed-selection provenance).

    Selection is the first `episodes` BC validation trajectory IDs in the existing
    episode split; every observation, including each demo endpoint, is included.
    Demo endpoints are not assumed to be training timeout states.
    """
    import h5py

    dataset, split_protocol = Path(dataset), Path(split_protocol)
    protocol = json.loads(split_protocol.read_text())
    if episodes < 1 or episodes > len(protocol["validation_ids"]):
        raise ValueError("Need a positive number of available BC-validation episodes")
    dataset_sha = sha256(dataset)
    if dataset_sha != protocol["dataset_sha256"]:
        raise ValueError("Probe dataset differs from saved BC split protocol")
    chosen = protocol["validation_ids"][:episodes]
    if set(chosen) & set(protocol["train_ids"]):
        raise ValueError("Probe validation IDs overlap BC training IDs")
    arrays, indices, group_keys = [], [], {}
    with h5py.File(dataset, "r") as data:
        for identifier in chosen:
            group = data[f"traj_{identifier}"]
            obs = np.asarray(group["obs"][:], dtype=np.float32)
            if obs.ndim != 2 or obs.shape[1] != 42 or not np.isfinite(obs).all():
                raise ValueError("Need finite 42-component saved state observations")
            if len(obs) != len(group["actions"]) + 1:
                raise ValueError("Saved observation/action alignment changed")
            arrays.append(obs)
            group_keys[str(identifier)] = list(group.keys())
            for index in range(len(obs)):
                indices.append(
                    {
                        "trajectory_id": int(identifier),
                        "observation_index": index,
                        "demo_length": len(obs) - 1,
                        "initial": index == 0,
                        "endpoint": index == len(obs) - 1,
                    }
                )
    observations = np.concatenate(arrays)
    return observations, {
        "dataset_sha256": dataset_sha,
        "split_protocol_sha256": sha256(split_protocol),
        "selection": "First existing BC-validation IDs, all saved observations; no success/model-based probe selection",
        "trajectory_ids": chosen,
        "observation_count": len(observations),
        "indices": indices,
        "observation_array_sha256": hashlib.sha256(observations.tobytes()).hexdigest(),
        "saved_demo_group_keys": group_keys,
        "scope": "Off-policy expert demonstration states, not on-policy PPO rollout data",
    }


def reward_bound_contract(source=None):
    """Inspect the installed PickCube formula before asserting its reward bound."""
    if source is None:
        source = importlib.metadata.distribution("mani-skill-nightly").locate_file(
            "mani_skill/envs/tasks/tabletop/pick_cube.py"
        )
    source = Path(source)
    module = ast.parse(source.read_text())
    task = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "PickCubeEnv"
    )
    functions = {
        node.name: node for node in task.body if isinstance(node, ast.FunctionDef)
    }
    dense = functions["compute_dense_reward"]
    normalized = functions["compute_normalized_dense_reward"]
    text = ast.unparse(dense)
    required = [
        "tcp_to_obj_dist = torch.linalg.norm(",
        "obj_to_goal_dist = torch.linalg.norm(",
        "reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)",
        "reward = reaching_reward",
        "reward += is_grasped",
        "place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)",
        "reward += place_reward * is_grasped",
        "static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))",
        "reward += static_reward * info['is_obj_placed']",
        "reward[info['success']] = 5",
        "return reward",
    ]
    if not all(fragment in text for fragment in required):
        raise ValueError("Installed reward formula changed; re-audit the claimed bound")
    reward_writes = []
    for node in ast.walk(dense):
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                (isinstance(target, ast.Name) and target.id == "reward")
                or (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "reward"
                )
                for target in targets
            ):
                reward_writes.append(node)
    if len(reward_writes) != 5:
        raise ValueError("Unexpected reward mutations; bound proof needs review")
    expression = normalized.body[0].value
    if not (
        isinstance(expression, ast.BinOp)
        and isinstance(expression.op, ast.Div)
        and isinstance(expression.right, ast.Constant)
        and expression.right.value == 5
        and isinstance(expression.left, ast.Call)
        and isinstance(expression.left.func, ast.Attribute)
        and expression.left.func.attr == "compute_dense_reward"
    ):
        raise ValueError("Normalized reward divisor changed")
    return {
        "source_sha256": sha256(source),
        "dense_function_line": dense.lineno,
        "normalized_function_line": normalized.lineno,
        "normalized_reward_min": 0.0,
        "normalized_reward_max": 1.0,
        "proof": "Distances and velocity norms are nonnegative; each 1-tanh(nonnegative) term is in [0,1]. Four nonnegative terms total at most 4; success overrides to 5. Division by 5 gives [0,1].",
        "note": "Source-inspected task bound, not a claim that critic predictions are bounded",
    }


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def optimizer_moments(policy):
    """Saved Adam buffers only: these are NOT pre/post-clip gradient norms."""
    result = {}
    for group_name, prefixes in [
        ("actor", ("mlp_extractor.policy_net.", "action_net.", "log_std")),
        ("critic", ("mlp_extractor.value_net.", "value_net.")),
    ]:
        first_sq, second_sum, below, total = 0.0, 0.0, 0, 0
        for name, parameter in policy.named_parameters():
            if not name.startswith(prefixes):
                continue
            state = policy.optimizer.state.get(parameter, {})
            if "exp_avg" not in state:
                continue
            m, v = (
                state["exp_avg"].detach().double(),
                state["exp_avg_sq"].detach().double(),
            )
            first_sq += float(m.square().sum())
            second_sum += float(v.sum())
            group = next(
                g
                for g in policy.optimizer.param_groups
                if any(p is parameter for p in g["params"])
            )
            below += int((v.sqrt() < group["eps"]).sum())
            total += v.numel()
        result[group_name] = {
            "exp_avg_l2": math.sqrt(first_sq),
            "sqrt_exp_avg_sq_l2": math.sqrt(second_sum),
            "entries": total,
            "sqrt_exp_avg_sq_below_adam_eps_fraction": below / total if total else None,
        }
    return {
        "scope": "Saved optimizer momentum buffers after past clipping, not actual historical gradient norms or proof of actor starvation",
        "groups": result,
    }


@isolated_rng
def audit(run, dataset, split_protocol, output, episodes=8):
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    run = Path(run)
    observations, probes = load_probes(dataset, split_protocol, episodes)
    contract = reward_bound_contract()
    protocol = json.loads((run / "protocol.json").read_text())
    if probes["observation_count"] < 1:
        raise ValueError("No saved observations")
    tensor = torch.from_numpy(observations)
    results = {}
    for arm in ["warm", "control"]:
        results[arm] = {}
        for label in ["initial", "best", "final"]:
            checkpoint = run / arm / f"{label}.zip"
            model = PPO.load(checkpoint, device="cpu")
            policy = model.policy
            with torch.no_grad():
                actor_features = policy.pi_features_extractor(tensor)
                critic_features = policy.vf_features_extractor(tensor)
                values = policy.predict_values(tensor).cpu().numpy().ravel()
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite saved critic predictions")
            if (
                policy.vf_features_extractor.clip != 10.0
                or float(critic_features.abs().max()) > 10.0
            ):
                raise ValueError("Loaded critic input clip contract changed")
            bound = contract["normalized_reward_max"] / (1.0 - model.gamma)
            masks = {
                phase: np.asarray([index[phase] for index in probes["indices"]])
                for phase in ["initial", "endpoint"]
            }
            all_parameters = list(policy.parameters())
            optimizer_ids = [
                id(p)
                for group in policy.optimizer.param_groups
                for p in group["params"]
            ]
            results[arm][label] = {
                "checkpoint_sha256": sha256(checkpoint),
                "actual_timesteps": int(model.num_timesteps),
                "gamma": float(model.gamma),
                "theoretical_nonnegative_discounted_return_upper_bound": bound,
                "predictions": values.tolist(),
                "prediction_summary": stats(values),
                "fraction_above_true_discounted_upper_bound": float(
                    (values > bound).mean()
                ),
                "demo_initial_prediction_summary": stats(values[masks["initial"]]),
                "demo_endpoint_prediction_summary": stats(values[masks["endpoint"]]),
                "actor_unclipped_normalized_absmax": float(actor_features.abs().max()),
                "critic_clipped_normalized_absmax": float(critic_features.abs().max()),
                "normalized_component_outside10_fraction": (actor_features.abs() > 10)
                .float()
                .mean(0)
                .cpu()
                .tolist(),
                "normalization_sha256": hashlib.sha256(
                    policy.pi_features_extractor.mean.cpu().numpy().tobytes()
                    + policy.pi_features_extractor.std.cpu().numpy().tobytes()
                ).hexdigest(),
                "global_clipping": {
                    "max_grad_norm": float(model.max_grad_norm),
                    "single_optimizer": type(policy.optimizer).__name__,
                    "optimizer_parameter_groups": len(policy.optimizer.param_groups),
                    "all_trainable_parameters_in_optimizer": set(optimizer_ids)
                    == {id(p) for p in all_parameters},
                    "feature_extractors_separate": not policy.share_features_extractor,
                },
                "saved_adam_moments": optimizer_moments(policy),
            }
    import stable_baselines3

    sb3 = Path(stable_baselines3.__file__).parent
    source_files = {
        "on_policy_algorithm": sb3 / "common/on_policy_algorithm.py",
        "gae": sb3 / "common/buffers.py",
        "ppo_optimizer": sb3 / "ppo/ppo.py",
        "subprocess_terminal_contract": sb3 / "common/vec_env/subproc_vec_env.py",
        "checkpoint_exclusions": sb3 / "common/base_class.py",
    }
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Read-only old-checkpoint inference on existing off-policy BC-validation observations; no simulator, rollout, training or parameter update",
        "audit_source_sha256": sha256(__file__),
        "source_protocol_sha256": sha256(run / "protocol.json"),
        "versions": versions(),
        "source_training_versions": protocol["versions"],
        "probes": probes,
        "reward_bound": contract,
        "checkpoints": results,
        "source_contract_hashes": {
            name: sha256(path) for name, path in source_files.items()
        },
        "confirmed_contracts": [
            "ManiSkill horizon truncation is preserved while task success termination is ignored by the experiment wrapper.",
            "SB3 adds gamma*V(pre-reset terminal observation) once on timeout and masks GAE across reset; no cross-episode bootstrap or double counting found.",
            "Both actor and critic use one Adam optimizer and one global gradient-norm clip over all policy parameters; separate feature extractors do not separate optimization.",
            "Critic feature clipping bounds inputs only. It does not bound learned predictions or bootstrap targets.",
        ],
        "historical_on_policy_gradient_norms": {
            "available": False,
            "reason": "PPO checkpoint excludes rollout_buffer. Saved demonstration data has observations/actions but no PPO old values/log-probabilities/advantages, rewards or actual training minibatches. Saved Adam buffers cannot recover pre-clip gradients.",
        },
        "hypotheses": [
            "An overestimated endpoint critic may inflate timeout bootstrap rewards/GAE targets and reinforce value growth; requires online timeout/target telemetry to establish.",
            "Large critic gradients may scale down actor gradients under shared global clipping; Adam adaptation means gradient rescaling alone does not prove smaller actor steps.",
            "BC-derived normalization and critic clipping may alias out-of-distribution states; quantify clipped-component occupancy on actual training observations.",
        ],
        "recommended_next_measurements": [
            "Record raw reward, timeout-added gamma*Vterminal, Vcurrent/Vnext, GAE advantage and lambda-return distributions separately; preserve pre-reset terminal and reset observations.",
            "Record actual actor/critic gradient norms before and after global clipping, total clipping factor, and actual actor/critic parameter update norms on on-policy minibatches.",
            "Test one predeclared critic stabilization factor with the same initial actor/critic tensors, RNG states, transitions and independent validation seeds; retain original evaluation task.",
            "Treat no-timeout-bootstrap as a distinct finite-horizon objective ablation, not a confirmed bug fix; remaining time is absent from the current state observation.",
        ],
    }
    save_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--split-protocol",
        type=Path,
        default=Path(__file__).parent / "evidence/stage1_pickcube/protocol.json",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=8)
    args = parser.parse_args()
    result = audit(
        args.run, args.dataset, args.split_protocol, args.output, args.episodes
    )
    print(
        json.dumps(
            {
                arm: {
                    label: entry["prediction_summary"] for label, entry in rows.items()
                }
                for arm, rows in result["checkpoints"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
