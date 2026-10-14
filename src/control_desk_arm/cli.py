"""Command-line tools for setting up and checking the arm."""

import time

from control_desk_arm.arm import CALIBRATION_DIR, JOINTS, make_arm, move_to, read_joints


def calibrate() -> None:
    """Interactive calibration. Writes calibration/<ARM_ID>.json."""
    arm = make_arm()
    arm.connect(calibrate=False)
    try:
        arm.calibrate()
    finally:
        arm.disconnect()
    print(f"Done. Calibration files are in {CALIBRATION_DIR}")


def monitor() -> None:
    """Print calibrated joint positions live with torque OFF, so you can move the arm by hand."""
    arm = make_arm()
    arm.connect(calibrate=False)
    arm.bus.disable_torque()
    print("Torque off: move the arm by hand. Ctrl+C to quit.\n")
    print("  ".join(f"{j:>13}" for j in JOINTS))
    try:
        while True:
            pos = read_joints(arm)
            print("\r" + "  ".join(f"{pos[j]:13.1f}" for j in JOINTS), end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    finally:
        arm.disconnect()


def test_motion() -> None:
    """Small, slow wiggle of each joint around its current position, then return."""
    arm = make_arm()
    arm.connect(calibrate=False)
    try:
        home = read_joints(arm)
        print("Start pose:", {j: round(v, 1) for j, v in home.items()})
        for joint, delta in [
            ("shoulder_pan", 15),
            ("shoulder_lift", -10),
            ("elbow_flex", 10),
            ("wrist_flex", 15),
            ("wrist_roll", 30),
            ("gripper", 20 if home["gripper"] < 50 else -20),
        ]:
            print(f"Moving {joint} by {delta:+}")
            move_to(arm, {joint: home[joint] + delta}, duration=1.0)
            move_to(arm, {joint: home[joint]}, duration=1.0)
        print("Done.")
    finally:
        arm.disconnect()
