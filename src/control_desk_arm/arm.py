"""Thin wrapper around LeRobot's SO-100/SO-101 follower for the arm on this desk.

Joint positions are in degrees (0 = calibrated middle of range) except the
gripper, which is 0 (closed) .. 100 (open).
"""

import os
import time
from pathlib import Path

from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_DIR = REPO_ROOT / "calibration"

ARM_PORT = os.environ.get("ARM_PORT", "COM3")
ARM_ID = os.environ.get("ARM_ID", "desk_arm")

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def make_arm(max_relative_target: float | None = 10.0) -> SOFollower:
    """Build (but don't connect) the arm. `max_relative_target` caps each step's jump in degrees."""
    config = SOFollowerRobotConfig(
        port=ARM_PORT,
        id=ARM_ID,
        calibration_dir=CALIBRATION_DIR,
        max_relative_target=max_relative_target,
    )
    return SOFollower(config)


def read_joints(arm: SOFollower) -> dict[str, float]:
    obs = arm.get_observation()
    return {j: float(obs[f"{j}.pos"]) for j in JOINTS}


def send_joints(arm: SOFollower, targets: dict[str, float]) -> dict[str, float]:
    """Send one position command. Returns what was actually sent after safety clipping."""
    sent = arm.send_action({f"{j}.pos": v for j, v in targets.items()})
    return {k.removesuffix(".pos"): float(v) for k, v in sent.items()}


def move_to(arm: SOFollower, targets: dict[str, float], duration: float = 2.0, hz: float = 30.0) -> None:
    """Smoothly interpolate from the current pose to `targets` over `duration` seconds."""
    start = read_joints(arm)
    goal = {**start, **targets}
    steps = max(1, int(duration * hz))
    for i in range(1, steps + 1):
        t = i / steps
        t = t * t * (3 - 2 * t)  # smoothstep easing
        send_joints(arm, {j: start[j] + (goal[j] - start[j]) * t for j in JOINTS})
        time.sleep(1 / hz)
