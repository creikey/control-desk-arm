"""Forward and inverse kinematics for the SO-101 arm.

Joint chain from TheRobotStudio's `so101_new_calib.urdf`, whose joint zeros match
LeRobot's calibrated degree readings. Positions are in meters in the base frame:
x forward, y left, z up, origin on the table surface under the base.
"""

import math

import numpy as np

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# (origin xyz, origin rpy) for each revolute joint along the chain; every axis is local +z.
_CHAIN = [
    ((0.0388353, 0.0, 0.0624), (math.pi, 0.0, -math.pi)),  # shoulder_pan
    ((-0.0303992, -0.0182778, -0.0542), (-math.pi / 2, -math.pi / 2, 0.0)),  # shoulder_lift
    ((-0.11257, -0.028, 0.0), (0.0, 0.0, math.pi / 2)),  # elbow_flex
    ((-0.1349, 0.0052, 0.0), (0.0, 0.0, -math.pi / 2)),  # wrist_flex
    ((0.0, -0.0611, 0.0181), (math.pi / 2, 0.0486795, math.pi)),  # wrist_roll
]
# Fixed offset from the wrist_roll link to the point between the gripper fingertips.
_TIP = ((-0.0079, -0.000218121, -0.0981274), (0.0, math.pi, 0.0))

# Height of the URDF base origin above the table. Measured by lowering the tip until the servo
# loads spiked; it also absorbs the small angle offsets left by calibration.
BASE_HEIGHT = 0.035

# The claw points down for picking from above (-90 = straight down).
POINTING_DEG = -80.0

# Joint limits in degrees from the URDF, shrunk slightly to stay off the hard stops.
LIMITS_DEG = {
    "shoulder_pan": (-105.0, 105.0),
    "shoulder_lift": (-95.0, 95.0),
    "elbow_flex": (-92.0, 92.0),
    "wrist_flex": (-92.0, 92.0),
    "wrist_roll": (-155.0, 160.0),
}


def _transform(xyz, rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    t = np.eye(4)
    t[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    t[:3, 3] = xyz
    return t


def _rot_z(q: float) -> np.ndarray:
    c, s = math.cos(q), math.sin(q)
    t = np.eye(4)
    t[:2, :2] = [[c, -s], [s, c]]
    return t


_CHAIN_T = [_transform(*j) for j in _CHAIN]
_TIP_T = _transform(*_TIP)


def frames(joints_deg: dict[str, float]) -> list[np.ndarray]:
    """World transforms of each joint frame (after rotation), followed by the gripper tip."""
    t = np.eye(4)
    t[2, 3] = BASE_HEIGHT
    out = []
    for name, origin in zip(ARM_JOINTS, _CHAIN_T):
        t = t @ origin @ _rot_z(math.radians(joints_deg[name]))
        out.append(t)
    out.append(t @ _TIP_T)
    return out


def tip_pose(joints_deg: dict[str, float]) -> tuple[np.ndarray, float]:
    """Gripper tip position and its pitch in degrees (0 = pointing horizontally, -90 = straight down).

    Pitch is the direction from the wrist_flex axis to the tip, so wrist roll doesn't change it.
    """
    f = frames(joints_deg)
    tip = f[-1][:3, 3]
    d = tip - f[3][:3, 3]
    pitch = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
    return tip, pitch


def solve_ik(
    target_xyz,
    target_pitch_deg: float,
    current_deg: dict[str, float],
    pitch_weight: float = 0.1,
    limits: dict[str, tuple[float, float]] = LIMITS_DEG,
    iterations: int = 200,
) -> tuple[dict[str, float], float, float]:
    """Find pan/lift/elbow/wrist_flex angles putting the tip at `target_xyz` with `target_pitch_deg`.

    Starts from `current_deg` so the arm stays in the same elbow configuration. Wrist roll is held.
    `pitch_weight` trades pitch error against position error (meters per radian).
    Returns (joint angles in degrees, position error in meters, pitch error in degrees).
    """
    names = ARM_JOINTS[:4]
    lo = np.array([limits[n][0] for n in names])
    hi = np.array([limits[n][1] for n in names])
    q = np.clip(np.array([current_deg[n] for n in names], dtype=float), lo, hi)
    target = np.array(target_xyz, dtype=float)

    def residual(qv):
        joints = {**current_deg, **dict(zip(names, qv))}
        tip, pitch = tip_pose(joints)
        return np.append(tip - target, pitch_weight * math.radians(pitch - target_pitch_deg))

    # Levenberg-Marquardt: only accept steps that reduce the error, adapting the damping.
    err = residual(q)
    cost = err @ err
    damping = 1e-6
    for _ in range(iterations):
        if cost < 1e-10:
            break
        jac = np.empty((4, 4))
        for i in range(4):
            dq = q.copy()
            dq[i] += 0.01
            jac[:, i] = (residual(dq) - err) / 0.01
        step = -np.linalg.solve(jac.T @ jac + damping * np.eye(4), jac.T @ err)
        candidate = np.clip(q + np.clip(step, -15.0, 15.0), lo, hi)
        candidate_err = residual(candidate)
        candidate_cost = candidate_err @ candidate_err
        if candidate_cost < cost:
            q, err, cost = candidate, candidate_err, candidate_cost
            damping = max(damping / 3, 1e-10)
        else:
            damping *= 4
            if damping > 1.0:
                break

    solution = {**current_deg, **dict(zip(names, (float(v) for v in q)))}
    return solution, float(np.linalg.norm(err[:3])), float(math.degrees(abs(err[3]) / pitch_weight))


# The moving jaw is a blade on the gripper hinge (so101_new_calib.urdf). Where its fingertip sits on the
# blade, and how the gripper reading (0 shut .. 100) maps to the hinge angle, were fitted to camera images
# of the claw at several openings.
_HINGE_T = _transform((0.0202, 0.0188, -0.0234), (1.5708, 0.0, 0.0))
_JAW_DEG_AT_SHUT = -59.3
_JAW_DEG_PER_UNIT = 1.395
_MOVING_TIP = np.array([0.0526, -0.0552, 0.0204, 1.0])


def moving_tip(frames_: list[np.ndarray], gripper: float) -> np.ndarray:
    """The moving jaw's fingertip, with the gripper at `gripper` (0 shut .. 100)."""
    angle = math.radians(_JAW_DEG_AT_SHUT + _JAW_DEG_PER_UNIT * gripper)
    return (frames_[4] @ _HINGE_T @ _rot_z(angle) @ _MOVING_TIP)[:3]


def grip_offset(frames_: list[np.ndarray], gripper: float) -> np.ndarray:
    """From the fixed fingertip to the grip point: halfway across the gap between the jaws, at fingertip
    height. (Opened wide, the moving jaw's tip rises well above the fingertips.)"""
    across = moving_tip(frames_, gripper) - frames_[-1][:3, 3]
    across[2] = 0.0
    return across / 2


def grip_point(joints_deg: dict[str, float]) -> np.ndarray:
    """The middle of the gap between the jaws, at fingertip height, in the world frame."""
    f = frames(joints_deg)
    return f[-1][:3, 3] + grip_offset(f, joints_deg["gripper"])
