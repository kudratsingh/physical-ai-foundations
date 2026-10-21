"""Week 2: forward and inverse kinematics for a planar two-revolute-joint arm.

Angles are radians, lengths are metres, and positive rotation is about +z.
q1 is measured from world +x; q2 is relative to link 1, so the hand's
orientation in the world is q1 + q2. Joints have unrestricted rotation.
"""
from dataclasses import dataclass
import math

import numpy as np

from transforms import make_transform, rot_z


@dataclass(frozen=True)
class IKSolution:
    """One position-IK solution; angles are radians and error is in metres."""

    branch: str
    q1: float
    q2: float
    error: float

    @property
    def angles(self) -> tuple[float, float]:
        return self.q1, self.q2


@dataclass(frozen=True)
class IKResult:
    reachable: bool
    solutions: tuple[IKSolution, ...] = ()
    reason: str = ""
    singular: bool = False
    infinite: bool = False


@dataclass(frozen=True)
class PlanarArm:
    """Geometry shared by the analytic functions and browser explorer."""

    l1: float = 1.0
    l2: float = 0.75

    def __post_init__(self):
        for name, length in (("l1", self.l1), ("l2", self.l2)):
            if not math.isfinite(length) or length <= 0:
                raise ValueError(f"{name} must be a positive finite length in metres")

    def joint_positions(self, q1: float, q2: float) -> np.ndarray:
        """Return world (x,y) coordinates of base, elbow and hand, shape (3,2)."""
        if not math.isfinite(q1) or not math.isfinite(q2):
            raise ValueError("Joint angles must be finite radians")
        elbow = self.l1 * np.array([math.cos(q1), math.sin(q1)])
        hand = elbow + self.l2 * np.array([math.cos(q1 + q2), math.sin(q1 + q2)])
        return np.array([[0.0, 0.0], elbow, hand])

    def forward_kinematics(self, q1: float, q2: float) -> np.ndarray:
        """Return the 4x4 hand pose T_world_hand, using the Week-1 convention."""
        hand = self.joint_positions(q1, q2)[-1]
        return make_transform(rot_z(q1 + q2), np.array([*hand, 0.0]))

    def transform_chain(self, q1: float, q2: float) -> tuple[np.ndarray, ...]:
        """Return T_world_shoulder, T_shoulder_elbow, T_elbow_hand.

        The shoulder frame rotates with link 1; the elbow frame rotates with
        link 2. Translate l1 along shoulder x before the second rotation;
        translate l2 along elbow x to reach the hand.
        """
        self.joint_positions(q1, q2)  # validate angles consistently
        return (
            make_transform(rot_z(q1), np.zeros(3)),
            make_transform(rot_z(q2), np.array([self.l1, 0.0, 0.0])),
            make_transform(np.eye(3), np.array([self.l2, 0.0, 0.0])),
        )

    @property
    def workspace_radii(self) -> tuple[float, float]:
        """Minimum/maximum hand distance from the base, ignoring collisions."""
        return abs(self.l1 - self.l2), self.l1 + self.l2

    def jacobian(self, q1: float, q2: float) -> np.ndarray:
        """World xy hand-velocity Jacobian: v [m/s] = J @ qdot [rad/s].

        Each column is the hand velocity from unit speed at one joint,
        holding the other joint fixed. This is position-only, not a 6D twist.
        """
        self.joint_positions(q1, q2)  # same angle validation as FK
        s1, c1 = math.sin(q1), math.cos(q1)
        s12, c12 = math.sin(q1 + q2), math.cos(q1 + q2)
        return np.array([
            [-self.l1 * s1 - self.l2 * s12, -self.l2 * s12],
            [ self.l1 * c1 + self.l2 * c12,  self.l2 * c12],
        ])

    def differential_kinematics(self, q1: float, q2: float) -> dict:
        """Jacobian and singular values for a unit Euclidean joint-rate ball.

        A rank tolerance handles sin(pi) roundoff. condition=None means the
        mathematical condition number is infinite (JSON has no infinity).
        A large finite condition is a teaching warning, not a rank change.
        """
        J = self.jacobian(q1, q2)
        U, sigma, _ = np.linalg.svd(J)
        tolerance = 2 * np.finfo(float).eps * sigma[0]
        singular = bool(sigma[1] <= tolerance)
        return {
            "jacobian": J.tolist(),
            "determinant": float(np.linalg.det(J)),
            "singular_values": sigma.tolist(),
            "principal_directions": U.T.tolist(),
            "rank": 1 if singular else 2,
            "singular": singular,
            "condition": None if singular else float(sigma[0] / sigma[1]),
            "manipulability": float(sigma[0] * sigma[1]),
        }

    def inverse_kinematics(self, x: float, y: float, shoulder_hint: float = 0.0) -> IKResult:
        """Solve hand POSITION, leaving heading determined by the joint angles.

        Elbow-up is the elbow to the left of the base-to-target ray (q2 < 0);
        elbow-down is on its right (q2 > 0). At a radial reach limit these
        branches coincide, so return one unique configuration. At the origin
        with equal links there are infinitely many configurations; return one
        representative using shoulder_hint rather than inventing two branches.
        """
        if not all(math.isfinite(v) for v in (x, y, shoulder_hint)):
            raise ValueError("Target coordinates and shoulder hint must be finite")
        r = math.hypot(x, y)
        inner, outer = self.workspace_radii
        tolerance = 32 * math.ulp(outer)

        def solution(branch: str, q1: float, q2: float) -> IKSolution:
            # One canonical representative per angle, matching the UI interval.
            q1 = (q1 + math.pi) % (2 * math.pi) - math.pi
            q2 = (q2 + math.pi) % (2 * math.pi) - math.pi
            hand = self.joint_positions(q1, q2)[-1]
            error = math.hypot(float(hand[0]) - x, float(hand[1]) - y)
            return IKSolution(branch, q1, q2, error)

        if r == 0:
            if self.l1 != self.l2:
                return IKResult(False, reason="inside_hole")
            return IKResult(True, (solution("representative", shoulder_hint, math.pi),),
                            singular=True, infinite=True)
        if r < inner - tolerance:
            return IKResult(False, reason="inside_hole")
        if r > outer + tolerance:
            return IKResult(False, reason="outside_reach")

        # Clamp only floating-point noise at a boundary, not unreachable goals.
        bounded_r = min(outer, max(inner, r))
        a, b, t, d = self.l1 / outer, self.l2 / outer, bounded_r / outer, inner / outer
        c2 = min(1.0, max(-1.0, (t*t - a*a - b*b) / (2*a*b)))
        # Factorized triangle area preserves sin(q2) near fully folded equal
        # links, where sqrt(1-cos(q2)^2) would cancel to zero prematurely.
        s2 = math.sqrt(max(0.0, (1-t)*(1+t)*(t-d)*(t+d))) / (2*a*b)
        phi = math.atan2(y, x)
        singular = bounded_r == inner or bounded_r == outer
        branches = (("singular", s2),) if singular else (("elbow-down", s2), ("elbow-up", -s2))
        solutions = tuple(solution(branch,
                                   phi - math.atan2(b*s, a + b*c2),
                                   math.atan2(s, c2)) for branch, s in branches)
        return IKResult(True, solutions, singular=singular)


def pose_payload(q1_deg: float, q2_deg: float, l1: float = 1.0, l2: float = 0.75) -> dict:
    """Browser-facing representation; convert degrees to radians at the boundary."""
    arm = PlanarArm(l1, l2)
    q1, q2 = math.radians(q1_deg), math.radians(q2_deg)
    positions = arm.joint_positions(q1, q2)
    transforms = arm.transform_chain(q1, q2)
    return {
        "differential": arm.differential_kinematics(q1, q2),
        "angles_deg": [q1_deg, q2_deg],
        "angles_rad": [q1, q2],
        "lengths": [arm.l1, arm.l2],
        "positions": positions.tolist(),
        "hand_heading_deg": q1_deg + q2_deg,
        "workspace_radii": list(arm.workspace_radii),
        "transform": arm.forward_kinematics(q1, q2).tolist(),
        "chain": [matrix.tolist() for matrix in transforms],
    }


def ik_payload(x: float, y: float, l1: float = 1.0, l2: float = 0.75,
               shoulder_hint_deg: float = 0.0) -> dict:
    """Reachability and solutions, with each pose checked by the actual FK code."""
    arm = PlanarArm(l1, l2)
    result = arm.inverse_kinematics(x, y, math.radians(shoulder_hint_deg))
    return {
        "target": [x, y],
        "lengths": [l1, l2],
        "workspace_radii": list(arm.workspace_radii),
        "reachable": result.reachable,
        "reason": result.reason,
        "singular": result.singular,
        "infinite": result.infinite,
        "solutions": [{
            "branch": s.branch,
            "error": s.error,
            "pose": pose_payload(math.degrees(s.q1), math.degrees(s.q2), l1, l2),
        } for s in result.solutions],
    }


if __name__ == "__main__":
    arm = PlanarArm()
    q1, q2 = math.radians(30), math.radians(60)
    print("q1 = 30 deg; q2 = 60 deg; lengths = 1.0 m, 0.75 m")
    print("Base, elbow, hand [m]:\n", arm.joint_positions(q1, q2))
    print("T_world_hand:\n", arm.forward_kinematics(q1, q2))
