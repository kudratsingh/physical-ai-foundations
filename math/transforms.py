"""Week 1 (Tue/Wed): rotations and rigid transforms in NumPy.

Implement these yourself from Modern Robotics Ch. 3; do not wrap a library.
"""
import numpy as np


def rot_z(theta: float) -> np.ndarray:
    """3x3 rotation about the z-axis by theta radians."""
    raise NotImplementedError


def compose_rotations(R_ab: np.ndarray, R_bc: np.ndarray) -> np.ndarray:
    """R_ac from R_ab and R_bc."""
    raise NotImplementedError


def inverse_rotation(R: np.ndarray) -> np.ndarray:
    """Inverse of a rotation matrix (hint: it is cheap)."""
    raise NotImplementedError


def make_transform(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """4x4 homogeneous transform from rotation R and translation p."""
    raise NotImplementedError


def inverse_transform(T: np.ndarray) -> np.ndarray:
    """Inverse of a homogeneous transform without np.linalg.inv."""
    raise NotImplementedError


def compose_transforms(T_ab: np.ndarray, T_bc: np.ndarray) -> np.ndarray:
    """T_ac from T_ab and T_bc."""
    raise NotImplementedError


def transform_point(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Apply T to a 3-vector point p."""
    raise NotImplementedError
