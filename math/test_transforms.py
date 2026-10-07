"""Run with: .venv/bin/python -m pytest math/  (or python math/test_transforms.py)."""
import os
import sys

import numpy as np

# Make `from transforms import ...` work from any cwd (plain python or pytest).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transforms import (rot_x, rot_z, compose_rotations, inverse_rotation,  # noqa: E402
                        make_transform, inverse_transform, compose_transforms,
                        transform_point)


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


def test_inverse_rotation_is_transpose():
    R = compose_rotations(rot_z(0.4), rot_x(-1.1))
    assert np.allclose(inverse_rotation(R), R.T)
    assert np.allclose(R @ inverse_rotation(R), np.eye(3))


def test_rot_z_angles_add():
    a, b = 0.3, 1.2
    assert np.allclose(compose_rotations(rot_z(a), rot_z(b)), rot_z(a + b))


def test_inverse_transform_matches_linalg_inv():
    R = compose_rotations(rot_z(0.9), rot_x(0.25))
    T = make_transform(R, np.array([-1.0, 0.5, 2.0]))
    assert np.allclose(inverse_transform(T), np.linalg.inv(T))


def test_transform_origin_gives_translation():
    p = np.array([3.0, -2.0, 0.5])
    T = make_transform(rot_x(0.8), p)
    assert np.allclose(transform_point(T, np.zeros(3)), p)


def test_worked_example_world_base_camera():
    # Same numbers as notes/frames_se3.md and transforms.py __main__.
    T_wb = make_transform(rot_z(np.pi / 2), np.array([2.0, 1.0, 0.0]))
    T_bc = make_transform(rot_x(np.pi), np.array([0.5, 0.0, 1.0]))
    p_c = np.array([0.1, 0.2, 0.8])

    T_wc = compose_transforms(T_wb, T_bc)
    expected_T_wc = np.array([[0.0, 1.0,  0.0, 2.0],
                              [1.0, 0.0,  0.0, 1.5],
                              [0.0, 0.0, -1.0, 1.0],
                              [0.0, 0.0,  0.0, 1.0]])
    assert np.allclose(T_wc, expected_T_wc)
    assert np.allclose(transform_point(T_bc, p_c), [0.6, -0.2, 0.2])
    assert np.allclose(transform_point(T_wc, p_c), [2.2, 1.6, 0.2])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
