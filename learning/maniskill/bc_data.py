"""Episode-level splits and strict data alignment for state-based behavior cloning."""

import hashlib
import json
from pathlib import Path
import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_episodes(path):
    import h5py

    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    info = metadata["env_info"]
    kwargs = info["env_kwargs"]
    if (
        info["env_id"] != "PushCube-v1"
        or kwargs.get("control_mode") != "pd_joint_delta_pos"
        or kwargs.get("sim_backend") != "physx_cpu"
        or kwargs.get("obs_mode") != "state"
    ):
        raise ValueError(
            "BC requires PushCube state observations, joint-delta actions and CPU replay"
        )
    episodes = []
    with h5py.File(path, "r") as data:
        for item in metadata["episodes"]:
            if not isinstance(item.get("episode_seed"), int):
                raise ValueError("Every demonstration needs an integer source seed")
            key = f"traj_{item['episode_id']}"
            obs = np.asarray(data[key]["obs"], dtype=np.float32)
            actions = np.asarray(data[key]["actions"], dtype=np.float32)
            if (
                actions.ndim != 2
                or actions.shape[1] != 8
                or obs.shape != (len(actions) + 1, 35)
                or not len(actions)
            ):
                raise ValueError(
                    f"{key}: expected T+1 observations and T actions, dimensions 35/8"
                )
            if not np.isfinite(obs).all() or not np.isfinite(actions).all():
                raise ValueError(f"{key}: non-finite observation or action")
            if not item.get("success", False):
                raise ValueError(f"{key}: dataset contains an unsuccessful replay")
            # The terminal observation has no demonstrated action; do not shift labels.
            episodes.append(
                dict(
                    id=item["episode_id"],
                    source_seed=item.get("episode_seed"),
                    initial_hash=hashlib.sha256(obs[0].tobytes()).hexdigest(),
                    obs=obs[:-1],
                    actions=np.clip(actions, -1, 1),
                    raw_clipped_scalars=int(np.count_nonzero(np.abs(actions) > 1)),
                    scalar_count=actions.size,
                )
            )
    if len({e["id"] for e in episodes}) != len(episodes):
        raise ValueError("Duplicate trajectory ID")
    if len({e["initial_hash"] for e in episodes}) != len(episodes):
        raise ValueError("Duplicate initial observation; deduplicate before splitting")
    return episodes


def split_episodes(episodes, train_count, validation_count, seed=2026):
    if min(train_count, validation_count) < 1 or train_count + validation_count > len(
        episodes
    ):
        raise ValueError(
            "Need enough unique episodes for disjoint train/validation sets"
        )
    order = np.random.default_rng(seed).permutation(len(episodes))
    validation = [episodes[i] for i in order[:validation_count]]
    training = [
        episodes[i] for i in order[validation_count : validation_count + train_count]
    ]
    return training, validation


def arrays(episodes):
    return np.concatenate([e["obs"] for e in episodes]), np.concatenate(
        [e["actions"] for e in episodes]
    )
