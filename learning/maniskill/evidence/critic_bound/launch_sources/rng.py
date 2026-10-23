"""Read-only evaluation must not advance the learner's random streams."""

from contextlib import contextmanager
from functools import wraps


@contextmanager
def preserve_rng():
    import random
    import numpy as np
    import torch

    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


def isolated_rng(function):
    @wraps(function)
    def run(*args, **kwargs):
        with preserve_rng():
            return function(*args, **kwargs)

    return run
