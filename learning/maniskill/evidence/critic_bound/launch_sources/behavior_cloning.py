"""Week 4: behavior cloning from official ManiSkill demonstrations, versus fixed PPO.

Episode-level data separation, train-only normalization, bounded joint actions,
validation-based checkpoint selection, and a fresh full-horizon test protocol.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn

from .bc_data import load_episodes, split_episodes, arrays, sha256_file
from .experiment import Config, evaluate, save_json, versions


class BehaviorPolicy(nn.Module):
    policy_label = "Behavior cloning deterministic"

    def __init__(self, mean=None, std=None, obs_dim=35):
        super().__init__()
        obs_dim = len(mean) if mean is not None else obs_dim
        self.register_buffer(
            "mean",
            torch.zeros(obs_dim)
            if mean is None
            else torch.as_tensor(mean, dtype=torch.float32),
        )
        self.register_buffer(
            "std",
            torch.ones(obs_dim)
            if std is None
            else torch.as_tensor(std, dtype=torch.float32).clamp(min=1e-3),
        )
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 8),
            nn.Tanh(),
        )

    def forward(self, observation):
        return self.net((observation - self.mean) / self.std)

    @torch.no_grad()
    def predict(self, observation, deterministic=True):
        self.eval()
        action = (
            self(torch.as_tensor(np.asarray(observation), dtype=torch.float32))
            .cpu()
            .numpy()
        )
        return np.clip(action, -1, 1), None

    def save(self, path):
        torch.save(
            {"schema": 2, "obs_dim": len(self.mean), "state_dict": self.state_dict()},
            path,
        )

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data["schema"] not in (1, 2):
            raise ValueError("Unsupported BC checkpoint schema")
        model = cls(obs_dim=data.get("obs_dim", 35))
        model.load_state_dict(data["state_dict"])
        model.eval()
        return model


def fit(
    episodes,
    validation,
    directory,
    updates=5000,
    seed=42,
    interval=1000,
    rollout_validation=True,
    task=None,
    validation_seeds=None,
):
    if updates < 1 or interval < 1:
        raise ValueError("Positive update and validation intervals required")
    task = task or Config()
    validation_seeds = (
        list(range(30000, 30010))
        if validation_seeds is None
        else list(validation_seeds)
    )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    observations, actions = arrays(episodes)
    val_observations, val_actions = arrays(validation)
    policy = BehaviorPolicy(observations.mean(0), observations.std(0))
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    x, y = torch.from_numpy(observations), torch.from_numpy(actions)
    vx, vy = torch.from_numpy(val_observations), torch.from_numpy(val_actions)
    generator = torch.Generator().manual_seed(seed + 1)
    history, best = [], None
    configuration = dict(
        algorithm="state-based behavior cloning",
        architecture=f"{observations.shape[1]}-256-256-8 ReLU/Tanh",
        seed=seed,
        updates=updates,
        batch_size=256,
        learning_rate=3e-4,
        train_ids=[e["id"] for e in episodes],
        validation_ids=[e["id"] for e in validation],
        train_transitions=len(x),
        validation_transitions=len(vx),
        action_label_preprocessing="clip recorded expert command to [-1,1], matching the controller",
        normalization="per-feature mean/std from training observations only, std floor 1e-3",
        validation_rollout_seeds=validation_seeds if rollout_validation else [],
        checkpoint_selection="validation success_once, then success_at_end, then negative demonstration validation MSE"
        if rollout_validation
        else "minimum demonstration validation MSE",
        task=asdict(task),
        versions=versions(),
    )
    save_json(directory / "config.json", configuration)
    start = time.monotonic()
    try:
        for update in range(1, updates + 1):
            policy.train()
            indices = torch.randint(len(x), (min(256, len(x)),), generator=generator)
            loss = nn.functional.mse_loss(policy(x[indices]), y[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if update == 1 or update % interval == 0 or update == updates:
                policy.eval()
                with torch.no_grad():
                    train_mse = nn.functional.mse_loss(policy(x), y).item()
                    validation_mse = nn.functional.mse_loss(policy(vx), vy).item()
                point = dict(
                    update=update, train_mse=train_mse, validation_mse=validation_mse
                )
                if rollout_validation:
                    result = evaluate(task, policy, validation_seeds)
                    point["rollout"] = result["summary"]
                    score = (
                        result["summary"]["success_once_rate"],
                        result["summary"]["success_at_end_rate"],
                        -validation_mse,
                    )
                else:
                    score = (-validation_mse,)
                if best is None or score > best:
                    best = score
                    policy.save(directory / "best.pt")
                    save_json(directory / "selected.json", point)
                history.append(point)
                save_json(directory / "losses.json", history)
                print(
                    f"update={update} train_mse={train_mse:.6f} validation_mse={validation_mse:.6f}",
                    flush=True,
                )
        policy.save(directory / "final.pt")
        save_json(
            directory / "training.json",
            dict(
                updates=updates, elapsed_seconds=time.monotonic() - start, complete=True
            ),
        )
    except KeyboardInterrupt:
        policy.save(directory / "interrupted.pt")
        raise
    return BehaviorPolicy.load(directory / "best.pt")


def run(
    dataset,
    ppo_path,
    output,
    updates=5000,
    test_episodes=50,
    large_count=256,
    small_count=32,
    validation_count=64,
):
    if not 0 < small_count < large_count:
        raise ValueError("Need 0 < small demonstration count < large count")
    if not Path(ppo_path).is_file():
        raise ValueError("An existing PPO checkpoint is required for the comparison")
    episodes = load_episodes(dataset)
    training, validation = split_episodes(episodes, large_count, validation_count)
    test_seeds = list(range(20000, 20000 + test_episodes))
    if (
        test_episodes < 1
        or set(test_seeds) & set(range(30000, 30010))
        or set(test_seeds) & {e["source_seed"] for e in episodes}
    ):
        raise ValueError(
            "Test seeds must be positive in count and disjoint from source demonstration seeds"
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save_json(
        output / "protocol.json",
        dict(
            task=asdict(Config()),
            dataset_path=str(Path(dataset).name),
            dataset_sha256=sha256_file(dataset),
            metadata_sha256=sha256_file(Path(dataset).with_suffix(".json")),
            retained_successful_replays=len(episodes),
            source="official ManiSkill RL demonstrations replayed in PhysX CPU",
            ppo_checkpoint_sha256=sha256_file(ppo_path),
            test_seeds=test_seeds,
            validation_rollout_seeds=list(range(30000, 30010)),
            source_seed_values=sorted({e["source_seed"] for e in episodes}),
            split_seed=2026,
            train_ids=[e["id"] for e in training],
            validation_ids=[e["id"] for e in validation],
            clipped_scalar_fraction=sum(e["raw_clipped_scalars"] for e in episodes)
            / sum(e["scalar_count"] for e in episodes),
            source_sha256=sha256_file(__file__),
        ),
    )
    policies = {}
    for name, subset in [
        (f"bc_{small_count}", training[:small_count]),
        (f"bc_{large_count}", training),
    ]:
        policies[name] = fit(subset, validation, output / name, updates=updates)
    # Models are selected using only validation before any final tests are run.
    from stable_baselines3 import PPO

    policies["ppo"] = PPO.load(ppo_path, device="cpu")
    policies.update(random="random", zero="zero")
    results = {}
    for name, policy in policies.items():
        results[name] = evaluate(
            Config(), policy, test_seeds, output / f"test_{name}.json", traces=True
        )
    save_json(
        output / "comparison.json", {name: r["summary"] for name, r in results.items()}
    )
    make_report(output)


def make_report(directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    comparison = json.loads((directory / "comparison.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for name in comparison:
        if not name.startswith("bc_"):
            continue
        color = {"bc_32": "#1f77b4", "bc_256": "#ff7f0e"}.get(name)
        history = json.loads((directory / name / "losses.json").read_text())
        axes[0].plot(
            [p["update"] for p in history],
            [p["validation_mse"] for p in history],
            marker="o",
            label=name,
        )
        if all("rollout" in p for p in history):
            axes[1].plot(
                [p["update"] for p in history],
                [p["rollout"]["success_once_rate"] for p in history],
                color=color,
                linestyle="--",
                alpha=0.5,
                label=name + " · any",
            )
            axes[1].plot(
                [p["update"] for p in history],
                [p["rollout"]["success_at_end_rate"] for p in history],
                marker="o",
                color=color,
                label=name + " · end",
            )
    axes[0].set(
        title="Held-out demonstration error",
        xlabel="Gradient updates",
        ylabel="Action MSE",
        yscale="log",
    )
    axes[1].set(
        title="Validation rollout success",
        xlabel="Gradient updates",
        ylabel="Success fraction",
        ylim=(-0.02, 1.05),
    )
    for ax in axes[:2]:
        ax.legend(fontsize=8)
    names = list(comparison)
    x = np.arange(len(names))
    axes[2].bar(
        x - 0.18,
        [comparison[n]["success_once_rate"] for n in names],
        0.36,
        label="Any step",
    )
    axes[2].bar(
        x + 0.18,
        [comparison[n]["success_at_end_rate"] for n in names],
        0.36,
        label="At end",
    )
    axes[2].set(
        title="Fresh test episodes",
        ylim=(0, 1),
        xticks=x,
        xticklabels=names,
        ylabel="Success fraction",
    )
    axes[2].tick_params(axis="x", rotation=30)
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(
        directory / "results.png", dpi=160, bbox_inches="tight", pad_inches=0.16
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "report", "video"])
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--ppo", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=20000)
    args = parser.parse_args()
    if args.command == "run":
        if not args.dataset or not args.ppo:
            parser.error("run requires --dataset and --ppo")
        run(args.dataset, args.ppo, args.output, args.updates, args.episodes)
    elif args.command == "report":
        make_report(args.output)
    else:
        if not args.checkpoint:
            parser.error("video requires --checkpoint")
        evaluate(
            Config(),
            BehaviorPolicy.load(args.checkpoint),
            list(range(args.seed_start, args.seed_start + args.episodes)),
            args.output,
            traces=True,
            video=True,
        )


if __name__ == "__main__":
    main()
