# Frames, SO(3) and SE(3) (Week 1, Tuesday–Wednesday)

Source: Modern Robotics Ch. 3 (intro, rotation matrices, 3.3.1).
Code: `math/transforms.py`, tests: `math/test_transforms.py`.

## Frames I care about

A **frame** is an origin plus three orthonormal axes attached to something.

- **world {w}** (MR calls it the space frame {s}): fixed, everything is
  ultimately expressed here.
- **base {b}**: bolted to the robot's base; moves if the robot drives around.
- **body / link frames**: one per link, moves with that link.
- **end-effector {e}**: at the tool; this is what I usually want to control.
- **camera {c}**: on the sensor. By the usual vision convention z points out
  of the lens, x right, y down in the image.

All frames are **right-handed**: x × y = z. Curl the right hand's fingers from
x to y, the thumb is z. Positive rotation about an axis is counterclockwise
when that axis points at me.

## Rotation matrices and the subscript rule

`R_ab` = orientation of frame {b} **expressed in** frame {a}. Its columns are
b's x, y, z unit axes written in a's coordinates.

Subscripts cancel like units, inner ones must match:

    R_ab R_bc = R_ac          R_ab p_b = p_a

If the inner subscripts do not match, the product is a bug.

## SO(3)

The rotation matrices form the group SO(3) = { R ∈ R³ˣ³ : RᵀR = I, det R = 1 }.

- RᵀR = I: columns are orthonormal (unit axes, mutually perpendicular).
- det R = +1: right-handed (det −1 would be a mirror).
- So R⁻¹ = Rᵀ, and R_ba = R_abᵀ. Inverting is a transpose, not a solve.
- Products of rotations are rotations; in general R₁R₂ ≠ R₂R₁.

## Same matrix, two meanings

A rotation matrix gets used in two ways that are easy to confuse:

1. **Change of frame** (passive): `p_a = R_ab p_b`. The point does not move; I
   just re-write its coordinates from b's axes into a's axes.
2. **Rotating a vector** (active): `v' = R v`, both in the same frame. The
   vector physically turns, e.g. `rot_z(90°) [1,0,0] = [0,1,0]`.

The numbers are identical; only the interpretation differs. Tied to this:
`R_sb' = R R_sb` rotates about the **fixed (space)** frame's axes, while
`R_sb' = R_sb R` rotates about the **body** frame's axes.

## Homogeneous transforms and SE(3)

A pose is a rotation plus a translation. Pack them into one 4×4 matrix:

    T_ab = [ R_ab  p_ab ]      p_ab = b's origin written in a
           [ 0     1    ]

These form SE(3) (special Euclidean group). Why 4×4:

- Applying a pose is `p_a = R_ab p_b + p_ab`, which is affine, not linear.
  Appending a 1 to the point, `[p; 1]`, makes it one matrix multiply:
  `[p_a; 1] = T_ab [p_b; 1]`.
- Composition is one multiply with the same cancellation rule:
  `T_ac = T_ab T_bc`.
- The inverse has a closed form (no general matrix inverse needed):

      T_ab⁻¹ = T_ba = [ Rᵀ  −Rᵀp ]
                      [ 0    1   ]

  Intuition: un-rotate with Rᵀ, and the old translation p, seen from the
  other side and in the other axes, becomes −Rᵀp.

## Worked example: world → robot base → camera

Setup (same numbers as `python math/transforms.py`):

- Base sits at (2, 1, 0) in the world, yawed +90° about world z:

      R_wb = rot_z(90°) = [0 −1 0; 1 0 0; 0 0 1],   p_wb = (2, 1, 0)

- Camera is 0.5 m forward and 1.0 m up from the base, flipped 180° about base
  x so its optical axis (z) points straight down:

      R_bc = rot_x(180°) = [1 0 0; 0 −1 0; 0 0 −1],  p_bc = (0.5, 0, 1.0)

- The camera sees a point 0.8 m in front of the lens: p_c = (0.1, 0.2, 0.8).

**Step 1, camera → base:** p_b = R_bc p_c + p_bc

    R_bc p_c = (0.1, −0.2, −0.8)
    p_b      = (0.1 + 0.5, −0.2 + 0, −0.8 + 1.0) = (0.6, −0.2, 0.2)

**Step 2, base → world:** p_w = R_wb p_b + p_wb

    R_wb p_b = (−(−0.2), 0.6, 0.2) = (0.2, 0.6, 0.2)
    p_w      = (0.2 + 2, 0.6 + 1, 0.2 + 0) = (2.2, 1.6, 0.2)

**Same thing in one shot:** T_wc = T_wb T_bc

    R_wc = R_wb R_bc = [0 1 0; 1 0 0; 0 0 −1]
    p_wc = R_wb p_bc + p_wb = (0, 0.5, 1.0) + (2, 1, 0) = (2, 1.5, 1.0)

    T_wc = [ 0  1  0  2.0 ]
           [ 1  0  0  1.5 ]
           [ 0  0 −1  1.0 ]
           [ 0  0  0  1   ]

    T_wc [0.1, 0.2, 0.8, 1]ᵀ = (0.2 + 2, 0.1 + 1.5, −0.8 + 1.0) = (2.2, 1.6, 0.2) ✓

Sanity check: the camera is 1.0 m up looking down, the point is 0.8 m in front
of it, so it should be 0.2 m above the floor. It is.

Going back: T_cw = T_wc⁻¹ has rotation R_wcᵀ and translation
−R_wcᵀ p_wc = −(1.5, 2, −1) = (−1.5, −2, 1), and T_cw p_w recovers
(0.1, 0.2, 0.8).
