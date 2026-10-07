# Configuration space (Week 1, Monday)

Source: Modern Robotics Ch. 2 (2.1, 2.2, 2.5; skim 2.3–2.4).

## Configuration and C-space

The **configuration** of a robot is a complete specification of where every
point of the robot is. If I know the configuration, I can draw the whole robot.
I write it as a vector `q`.

The **configuration space (C-space)** is the set of all configurations the
robot can be in. It is a space with a shape (a topology), not just a list of
numbers: a revolute joint angle at 0 and at 2π is the *same* configuration, so
that coordinate lives on a circle S¹, not on a line R.

## Degrees of freedom

The **degrees of freedom (DoF)** are the dimension of the C-space: the minimum
number of real numbers needed to pin down the configuration.

A rigid body in 3D has 6 DoF (3 position + 3 orientation); in the plane it has
3 (x, y, θ). Joints take freedom away. For a mechanism this is counted by
**Grübler's formula**:

    dof = m (N − 1 − J) + Σ f_i

where m = 3 (planar) or 6 (spatial), N = number of links *including ground*,
J = number of joints, f_i = freedoms of joint i. Each joint removes
(m − f_i) constraints.

Example, planar 2R arm: m = 3, N = 3 (ground + 2 links), J = 2 revolute joints
with f = 1 each → dof = 3(3 − 1 − 2) + 2 = **2**.
(Grübler assumes the joint constraints are independent; special geometry can
break that.)

## Task space vs workspace

- **Task space**: the space I describe the *task* in, e.g. the end-effector's
  position and orientation in SE(3), or just (x, y) for a pen on paper. It is
  chosen by the task, not by the robot.
- **Workspace**: the set of end-effector configurations the robot can actually
  reach. It is a property of the robot's geometry and joint limits.

Neither one is the C-space. The C-space is about the *whole robot*; task space
and workspace are about the end-effector.

## Why a robot state is not just an XYZ coordinate

An XYZ point says where the gripper tip is, and nothing else:

- It ignores orientation: the same tip position can be reached with the tool
  pointing in many directions.
- It ignores the rest of the body: a 2R arm reaches most tip positions in two
  ways (elbow up / elbow down), and a 7-DoF arm reaches them in infinitely many.
  Those are different configurations with different collisions and dynamics.
- It ignores velocity. The *state* for dynamics is (q, q̇): configuration plus
  how fast it is changing. Same q, different q̇ → different future.

So for planning and control I work in `q` (and `q̇`), and treat XYZ as one
derived quantity (forward kinematics) among many.

## Worked examples

| system | DoF | what q contains | C-space |
|---|---|---|---|
| 1-DoF revolute joint | 1 | joint angle θ | S¹ (circle); an interval of R if joint limits stop it wrapping |
| planar 2R arm | 2 | joint angles (θ₁, θ₂) | T² = S¹ × S¹ (torus) |
| free rigid body in 3D | 6 | position (x, y, z) + orientation (e.g. rotation matrix R or unit quaternion) | R³ × SO(3) (= SE(3)) |

## Connection to MuJoCo

In MuJoCo, `data.qpos` *is* the configuration vector q and `model.nq` is its
length; for a free body nq = 7 (3 position + 4 quaternion numbers) while
nv = 6 (velocity has one number per DoF), which is exactly the gap between
"numbers I use to store q" and "dimension of the C-space".
