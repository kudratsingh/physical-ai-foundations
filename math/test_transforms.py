"""Run with: .venv/bin/python -m pytest math/  (or python math/test_transforms.py)."""
import numpy as np
from transforms import (rot_z, compose_rotations, inverse_rotation, make_transform,
                        inverse_transform, compose_transforms, transform_point)


def test_rotation_is_orthonormal():
    R = rot_z(0.7)
    assert np.allclose(R.T @ R, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1.0)


def test_transform_inverse():
    T = make_transform(rot_z(0.3), np.array([1.0, 2.0, 3.0]))
    assert np.allclose(T @ inverse_transform(T), np.eye(4))


def test_composition_matches_sequential():
    T_ab = make_transform(rot_z(0.2), np.array([1.0, 0.0, 0.0]))
    T_bc = make_transform(rot_z(-0.5), np.array([0.0, 1.0, 0.0]))
    p_c = np.array([0.5, 0.5, 0.5])
    direct = transform_point(compose_transforms(T_ab, T_bc), p_c)
    sequential = transform_point(T_ab, transform_point(T_bc, p_c))
    assert np.allclose(direct, sequential)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
