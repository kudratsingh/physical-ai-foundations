"""Opt-in worker-local RNG seeding for reproducible training auto-resets.

The learner still seeds each environment's first reset through VecEnv.seed().
These additional worker-global seeds cover later unseeded ManiSkill resets,
including initialization code that draws from the worker's Torch generator.
The enhanced-determinism option binds each automatic reset to ManiSkill's
advancing per-episode seed, avoiding a repeated first scene. Historical recipes
remain available in their archived source commits.
"""

import operator
import random

import numpy as np
import torch

from .training_metrics import training_env


def seeded_training_env(config, rank):
    """Return a spawn-compatible factory with worker seed ``config.seed + rank``.

    Seed before construction, then reseed after it so constructor RNG consumption
    cannot shift the stream used by later automatic resets. Use distinct ranks
    and ``vec_env.seed(config.seed)`` before the initial vector reset. Each factory
    belongs in its own spawned worker, not several environments sharing one
    process-global RNG.
    """
    if isinstance(rank, bool):
        raise TypeError("Worker rank must be a nonnegative integer")
    try:
        rank = operator.index(rank)
    except TypeError as error:
        raise TypeError("Worker rank must be a nonnegative integer") from error
    if rank < 0:
        raise ValueError("Worker rank must be a nonnegative integer")
    worker_seed = operator.index(config.seed) + rank
    if not 0 <= worker_seed < 2**32:
        raise ValueError("Worker seed must fit NumPy's unsigned 32-bit seed range")

    def create():
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        env = training_env(config, enhanced_determinism=True)()
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        return env

    return create
