"""Week 1 (Tue/Wed): rotations and rigid transforms in NumPy.

Implemented from Modern Robotics Ch. 3; no library wrappers.

Subscript convention (same as the book):
    R_ab : orientation of frame {b} expressed in frame {a}.
           Its columns are b's x, y, z axes written in a's coordinates.
    T_ab : pose of frame {b} expressed in frame {a} = [[R_ab, p_ab], [0, 1]],
           where p_ab is b's origin written in a's coordinates.
    p_b  : a point (or vector) written in frame {b}'s coordinates.

Inner subscripts cancel:  R_ab @ R_bc = R_ac,  T_ab @ T_bc = T_ac,
                          T_ab @ p_b  = p_a.
"""
import numpy as np


def _check_rotation_shape(R: np.ndarray) -> None:
    assert R.shape == (3, 3), f"expected 3x3 rotation, got {R.shape}"


def _check_transform_shape(T: np.ndarray) -> None:
    assert T.shape == (4, 4), f"expected 4x4 transform, got {T.shape}"


def rot_x(theta: float) -> np.ndarray:
    """3x3 rotation about the x-axis by theta radians (right-hand rule)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,   c,  -s],
                     [0.0,   s,   c]])


def rot_y(theta: float) -> np.ndarray:
    """3x3 rotation about the y-axis by theta radians (right-hand rule).

    Note the sign flip vs rot_x/rot_z: the -s sits bottom-left because the
    cyclic order is z -> x (not x -> z).
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[  c, 0.0,   s],
                     [0.0, 1.0, 0.0],
                     [ -s, 0.0,   c]])


def rot_z(theta: float) -> np.ndarray:
    """3x3 rotation about the z-axis by theta radians (right-hand rule).

    Read as R_ab where {b} is {a} rotated by theta about a's z-axis.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[  c,  -s, 0.0],
                     [  s,   c, 0.0],
                     [0.0, 0.0, 1.0]])


def compose_rotations(R_ab: np.ndarray, R_bc: np.ndarray) -> np.ndarray:
    """R_ac from R_ab and R_bc: the inner 'b' subscripts cancel."""
    _check_rotation_shape(R_ab)
    _check_rotation_shape(R_bc)
    return R_ab @ R_bc


def inverse_rotation(R: np.ndarray) -> np.ndarray:
    """Inverse of a rotation matrix: R_ba = R_ab^-1 = R_ab^T (SO(3) is orthonormal)."""
    _check_rotation_shape(R)
    return R.T


def make_transform(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """4x4 homogeneous transform T_ab from R_ab (3x3) and p_ab (3,).

        T = [[R, p],
             [0, 1]]
    """
    _check_rotation_shape(R)
    p = np.asarray(p, dtype=float)
    assert p.shape == (3,), f"expected 3-vector, got {p.shape}"
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def inverse_transform(T: np.ndarray) -> np.ndarray:
    """T_ba from T_ab using the closed form (no np.linalg.inv):

        T^-1 = [[R^T, -R^T p],
                [0,    1    ]]
    """
    _check_transform_shape(T)
    R = T[:3, :3]
    p = T[:3, 3]
    R_inv = R.T
    return make_transform(R_inv, -R_inv @ p)


def compose_transforms(T_ab: np.ndarray, T_bc: np.ndarray) -> np.ndarray:
    """T_ac from T_ab and T_bc: the inner 'b' subscripts cancel."""
    _check_transform_shape(T_ab)
    _check_transform_shape(T_bc)
    return T_ab @ T_bc


def transform_point(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Apply T_ab to a point p_b (3,) and return p_a (3,).

    Equivalent to R_ab @ p_b + p_ab; done here by appending a 1 (homogeneous
    coordinates) so the translation rides along in one matrix multiply.
    """
    _check_transform_shape(T)
    p = np.asarray(p, dtype=float)
    assert p.shape == (3,), f"expected 3-vector, got {p.shape}"
    p_h = np.append(p, 1.0)
    return (T @ p_h)[:3]


if __name__ == "__main__":
    # Worked example from notes/frames_se3.md: world {w} -> base {b} -> camera {c}.
    np.set_printoptions(precision=3, suppress=True)

    def clean(x):
        """Round away float noise and turn -0.0 into 0.0 for readable printing."""
        return np.round(x, 12) + 0.0

    # Robot base sits at (2, 1, 0) in the world, yawed +90 deg about world z.
    T_wb = make_transform(rot_z(np.pi / 2), np.array([2.0, 1.0, 0.0]))
    # Camera is 0.5 m forward and 1.0 m up from the base, flipped 180 deg about
    # base x so its z-axis (optical axis) points straight down.
    T_bc = make_transform(rot_x(np.pi), np.array([0.5, 0.0, 1.0]))
    # A point the camera sees, in camera coordinates.
    p_c = np.array([0.1, 0.2, 0.8])

    T_wc = compose_transforms(T_wb, T_bc)
    p_b = transform_point(T_bc, p_c)
    p_w = transform_point(T_wc, p_c)

    print("T_wb (base in world):\n", clean(T_wb))
    print("T_bc (camera in base):\n", clean(T_bc))
    print("T_wc = T_wb @ T_bc (camera in world):\n", clean(T_wc))
    print("p_c (point in camera):", clean(p_c))
    print("p_b = T_bc p_c        :", clean(p_b))
    print("p_w = T_wb p_b        :", clean(transform_point(T_wb, p_b)))
    print("p_w = T_wc p_c        :", clean(p_w))
    print("T_cw = T_wc^-1:\n", clean(inverse_transform(T_wc)))
    print("round trip T_cw p_w   :", clean(transform_point(inverse_transform(T_wc), p_w)))
