"""Velocity teleoperation: inputs move the claw's grip point, wrist roll and opening at 30 Hz.

The joystick moves the grip point across the table relative to the camera image (stick up = away
from the camera); the claw keeps pointing down.
"""

import logging
import threading
import time

import numpy as np

from control_desk_arm.arm import connect, joint_limits, make_arm, move_to, read_joints, release, send_joints
from control_desk_arm.camera import CameraPose
from control_desk_arm.kinematics import LIMITS_DEG, POINTING_DEG, frames, grip_offset, grip_point, solve_ik

HZ = 30.0
MOVE_SPEED = 0.10  # m/s at full input
Z_SPEED = 0.06  # m/s
WRIST_SPEED = 120.0  # deg/s
GRIP_SPEED = 150.0  # deg/s
INPUT_STALE_S = 0.4  # stop if the input stream pauses (lag, lost packets)
IK_TOLERANCE = 0.004

GRIPPER_OPEN = 90.0
HOME_GRIP = np.array([0.25, 0.0, 0.15])
REST_JOINTS = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -88.0,
    "elbow_flex": 89.0,
    "wrist_flex": 70.0,
    "wrist_roll": 0.0,
    "gripper": 0.0,
}
NO_INPUT = (0.0, 0.0, 0.0, 0.0, 0.0)

log = logging.getLogger(__name__)


class Teleop:
    def __init__(self, pose: CameraPose):
        cam_to_robot = pose.R.T
        right, away = cam_to_robot[:, 0].copy(), cam_to_robot[:, 2].copy()
        right[2] = away[2] = 0.0
        self.right, self.away = right / np.linalg.norm(right), away / np.linalg.norm(away)

        self.state = "off"  # off | starting | on | stopping
        self.error: str | None = None
        self.grip: np.ndarray | None = None  # target grip point (m)
        self.claw = GRIPPER_OPEN
        self._input = NO_INPUT
        self._input_at = 0.0
        self._arm = None
        self._limits: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def set_input(self, x: float, y: float, z: float, wrist: float, grip: float) -> None:
        self._input = tuple(float(np.clip(v, -1.0, 1.0)) for v in (x, y, z, wrist, grip))
        self._input_at = time.monotonic()

    def stop(self) -> None:
        self._input = NO_INPUT

    def power_on(self) -> None:
        """Connect, move to the home pose with the claw open, and start the control loop. Blocking."""
        with self._lock:
            if self.state != "off":
                return
            self.state, self.error = "starting", None
            try:
                arm = make_arm(max_relative_target=None)
                connect(arm)
                calibrated = joint_limits(arm)
                self._limits = {
                    n: (max(lo, calibrated[n][0]), min(hi, calibrated[n][1])) for n, (lo, hi) in LIMITS_DEG.items()
                }
                joints = read_joints(arm)
                # Hold where it is (it may have sagged past a limit while limp) rather than at a stale goal.
                joints.update({n: float(np.clip(joints[n], lo, hi)) for n, (lo, hi) in self._limits.items()})
                send_joints(arm, joints)
                arm.bus.enable_torque(num_retry=3)

                home = {**joints, "wrist_roll": 0.0, "gripper": GRIPPER_OPEN}
                for _ in range(4):  # the grip offset depends on the final wrist pose
                    tip = HOME_GRIP - grip_offset(frames(home), 1.0)
                    home, _, _ = solve_ik(tip, POINTING_DEG, home, limits=self._limits)
                move_to(arm, home, duration=2.0)
            except Exception as e:
                log.exception("arm failed to start")
                self.state, self.error = "off", f"{type(e).__name__}: {e}"
                return
            self._arm, self.claw = arm, GRIPPER_OPEN
            self.grip = grip_point(home, 1.0)
            self.state = "on"
            self._thread = threading.Thread(target=self._loop, args=(home,), daemon=True)
            self._thread.start()

    def power_off(self) -> None:
        """Fold the arm up and release torque. Blocking."""
        with self._lock:
            if self.state != "on":
                return
            self.state = "stopping"
            self._thread.join()
            try:
                move_to(self._arm, REST_JOINTS, duration=2.5)
            finally:
                release(self._arm)
                self._arm, self.grip, self.state = None, None, "off"

    def status(self) -> dict:
        out = {"arm": self.state}
        if self.error:
            out["error"] = self.error
        if self.grip is not None:
            out["grip_cm"] = [round(float(v) * 100, 1) for v in self.grip]
            out["claw"] = round(self.claw)
        return out

    def _loop(self, joints: dict[str, float]) -> None:
        next_tick = time.perf_counter()
        while self.state == "on":
            next_tick += 1 / HZ
            try:
                joints = self._tick(joints)
            except Exception as e:
                log.exception("control loop failed")
                self.error = f"{type(e).__name__}: {e}"
            time.sleep(max(0.0, next_tick - time.perf_counter()))
            next_tick = max(next_tick, time.perf_counter() - 1 / HZ)

    def _tick(self, joints: dict[str, float]) -> dict[str, float]:
        dt = 1 / HZ
        x, y, z, wrist, grip = self._input
        if time.monotonic() - self._input_at > INPUT_STALE_S:
            x = y = z = wrist = grip = 0.0
        goal = dict(joints)

        if x or y or z:
            target = self.grip + (self.right * x + self.away * y) * MOVE_SPEED * dt
            target[2] += z * Z_SPEED * dt
            tip = target - grip_offset(frames(joints), self.claw / GRIPPER_OPEN)
            solved, err, _ = solve_ik(tip, POINTING_DEG, joints, limits=self._limits, iterations=30)
            if err > IK_TOLERANCE:  # let the claw tilt a little rather than stop dead
                solved, err, _ = solve_ik(
                    tip, POINTING_DEG, joints, pitch_weight=0.005, limits=self._limits, iterations=30
                )
            if err <= IK_TOLERANCE:
                goal.update({k: v for k, v in solved.items() if k != "wrist_roll"})
                self.grip = target

        if wrist:
            lo, hi = self._limits["wrist_roll"]
            goal["wrist_roll"] = float(np.clip(goal["wrist_roll"] + wrist * WRIST_SPEED * dt, lo, hi))

        self.claw = float(np.clip(self.claw + grip * GRIP_SPEED * dt, 0.0, GRIPPER_OPEN))
        goal["gripper"] = self.claw
        send_joints(self._arm, goal)
        return goal
