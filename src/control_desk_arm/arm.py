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


def connect(arm: SOFollower) -> None:
    """Connect, first resetting any motor whose overload protection tripped.

    A tripped motor drops out of LeRobot's startup check, and stays tripped until its torque is
    switched on again.
    """
    bus = arm.bus
    bus.connect(handshake=False)
    ph, port = bus.packet_handler, bus.port_handler
    try:
        for motor in bus.motors.values():
            _, result, error = ph.ping(port, motor.id)
            if result == 0 and error:
                position, _, _ = ph.read2ByteTxRx(port, motor.id, 56)  # Present_Position
                ph.write2ByteTxRx(port, motor.id, 42, position)  # Goal_Position
                ph.write1ByteTxRx(port, motor.id, 40, 1)  # Torque_Enable
                ph.write1ByteTxRx(port, motor.id, 40, 0)
    finally:
        bus.disconnect(disable_torque=False)
    arm.connect(calibrate=False)


def joint_limits(arm: SOFollower, margin_deg: float = 2.0) -> dict[str, tuple[float, float]]:
    """Each joint's range of motion recorded during calibration, shrunk by `margin_deg`."""
    limits = {}
    for name, cal in arm.calibration.items():
        half = (cal.range_max - cal.range_min) / 2 * 360 / 4095 - margin_deg
        limits[name] = (-half, half)
    return limits


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


def release(arm: SOFollower) -> None:
    """Turn torque off on every motor, even if one of them reports an error, and disconnect."""
    for motor in arm.bus.motors:
        try:
            arm.bus.write("Torque_Enable", motor, 0, num_retry=3)
        except RuntimeError as e:
            print(f"warning: couldn't release {motor}: {e}")
    arm.bus.disconnect(disable_torque=False)
