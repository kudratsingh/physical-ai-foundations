"""Shared pytest setup.

The code under test lives in plain folders (math/, mujoco/), not in an
installed package, so we put those folders on sys.path here. That makes
``from transforms import ...``, ``from simulate import ...`` and
``from pd_control import ...`` work in every test file, from any cwd.
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
for sub in ("math", "mujoco"):
    p = str(REPO / sub)
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture
def model_data():
    """A fresh (MjModel, MjData) pair loaded from mujoco/one_joint.xml."""
    from simulate import load_model
    return load_model()


@pytest.fixture
def tmp_logs(tmp_path, monkeypatch):
    """Redirect the CSV logs of simulate.py and pd_control.py into tmp_path,
    so running the tests never touches mujoco/logs/."""
    import pd_control
    import simulate
    monkeypatch.setattr(simulate, "LOG_DIR", tmp_path)
    monkeypatch.setattr(pd_control, "LOG_DIR", tmp_path)
    return tmp_path
