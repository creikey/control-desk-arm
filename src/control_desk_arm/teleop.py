"""Velocity teleoperation: inputs move the claw's grip point, wrist roll, tilt and opening at 30 Hz.

The joystick moves the grip point across the table relative to the camera image (stick up = away
from the camera). Tilt swings the claw about the grip point between pointing straight down and
reaching out, away from the base.
"""

import contextlib
import logging
import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from control_desk_arm.arm import connect, joint_limits, make_arm, move_to, read_joints, release, send_joints
from control_desk_arm.camera import CameraPose
from control_desk_arm.kinematics import LIMITS_DEG, POINTING_DEG, frames, grip_offset, grip_point, solve_ik

HZ = 30.0
MOVE_SPEED = 0.10  # m/s at full input
Z_SPEED = 0.06  # m/s
WRIST_SPEED = 120.0  # deg/s
GRIP_SPEED = 150.0  # deg/s
TILT_SPEED = 45.0  # deg/s
INPUT_STALE_S = 0.4  # stop if the input stream pauses (lag, lost packets)
IK_TOLERANCE = 0.004
START_ATTEMPTS = 5
RECONNECT_AFTER = 15  # consecutive failed ticks (half a second) before reconnecting to the arm

GRIPPER_OPEN = 90.0
PITCH_RANGE = (-90.0, -20.0)  # degrees: straight down .. reaching well out
HOME_GRIP = np.array([0.25, 0.0, 0.15])
REST_JOINTS = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -88.0,
    "elbow_flex": 89.0,
    "wrist_flex": 70.0,
    "wrist_roll": 0.0,
    "gripper": 0.0,
}
AXES = ("x", "y", "z", "wrist", "grip", "tilt")
NO_INPUT = (0.0,) * len(AXES)

log = logging.getLogger(__name__)


@dataclass
class Plan:
    """A move run as the inputs a person would hold, for exactly `ticks` control ticks."""

    input: tuple[float, ...]
    ticks: int
    target: np.ndarray  # where the grip point will end up
    done: threading.Event = field(default_factory=threading.Event)


class Teleop:
    def __init__(self, pose: CameraPose):
        self.pose = pose
        cam_to_robot = pose.R.T
        right, away = cam_to_robot[:, 0].copy(), cam_to_robot[:, 2].copy()
        right[2] = away[2] = 0.0
        self.right, self.away = right / np.linalg.norm(right), away / np.linalg.norm(away)

        self.state = "off"  # off | starting | on | stopping
        self.grip: np.ndarray | None = None  # target grip point (m)
        self.claw = GRIPPER_OPEN
        self.pitch = POINTING_DEG
        self.joints: dict[str, float] = {}  # last commanded
        self.applied = NO_INPUT  # the inputs in effect, whoever is driving
        self._input = NO_INPUT
        self._input_at = 0.0
        self._plan: Plan | None = None
        self._arm = None
        self._limits: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def set_input(self, *values: float) -> None:
        self._input = tuple(float(np.clip(v, -1.0, 1.0)) for v in values)
        self._input_at = time.monotonic()

    def stop(self) -> None:
        self._input = NO_INPUT
        plan, self._plan = self._plan, None
        if plan:
            plan.done.set()

    def plan_move(
        self, right: float, away: float, up: float, wrist_deg: float, tilt_deg: float, claw: float | None
    ) -> Plan:
        """Move the grip point by `right`/`away` (across the table, relative to the camera image) and
        `up` meters, turn the wrist, tilt the claw, and set it `claw` open (0 shut .. 1 open, None to
        leave it).

        Runs as held inputs scaled so every part finishes on the same tick, so the distances are exact.
        """
        claw_change = 0.0 if claw is None else float(np.clip(claw, 0, 1)) * GRIPPER_OPEN - self.claw
        tilt_deg = float(np.clip(self.pitch + tilt_deg, *PITCH_RANGE)) - self.pitch
        seconds = max(
            math.hypot(right, away) / MOVE_SPEED,
            abs(up) / Z_SPEED,
            abs(wrist_deg) / WRIST_SPEED,
            abs(claw_change) / GRIP_SPEED,
            abs(tilt_deg) / TILT_SPEED,
        )
        ticks = max(1, math.ceil(seconds * HZ))
        t = ticks / HZ
        inputs = (
            right / (MOVE_SPEED * t),
            away / (MOVE_SPEED * t),
            up / (Z_SPEED * t),
            wrist_deg / (WRIST_SPEED * t),
            claw_change / (GRIP_SPEED * t),
            tilt_deg / (TILT_SPEED * t),
        )
        target = self.grip + self.right * right + self.away * away + np.array([0.0, 0.0, up])
        self.stop()
        self._plan = Plan(inputs, ticks, target)
        self.applied = inputs
        return self._plan

    def table_point(self, u: float, v: float) -> np.ndarray:
        """The point on the table (z = 0) seen at image pixel (u, v)."""
        R, t = self.pose.R, self.pose.tvec
        center = -R.T @ t
        ray = R.T @ np.linalg.solve(self.pose.K, [u, v, 1.0])
        return center + ray * (-center[2] / ray[2])

    def table_offset(self, delta: np.ndarray) -> tuple[float, float]:
        """Split a horizontal move into its `right` and `away` parts (meters)."""
        right, away = np.linalg.solve(np.column_stack([self.right[:2], self.away[:2]]), delta[:2])
        return float(right), float(away)

    @property
    def plan_target(self) -> np.ndarray | None:
        plan = self._plan
        return None if plan is None else plan.target

    @property
    def fingertip_z(self) -> float:
        return float(frames(self.joints)[-1][2, 3])

    def jaw_ends(self, half_length: float = 0.03) -> np.ndarray:
        """Two points either side of the grip point along the line the jaws close on (seen from above).

        The moving jaw hinges about the gripper frame's y axis and the fingers point along its -z, so
        the jaws close along its x axis (so101_new_calib.urdf).
        """
        axis = frames(self.joints)[4][:3, 0].copy()
        axis[2] = 0.0
        axis *= half_length / np.linalg.norm(axis)
        return np.array([self.grip - axis, self.grip + axis])

    def jaw_image_angle(self) -> float:
        """The jaws' closing line in the camera image, in degrees counterclockwise from pointing right."""
        a, b = self.pose.project(self.jaw_ends())
        return math.degrees(math.atan2(a[1] - b[1], b[0] - a[0])) % 180

    def power_on(self) -> bool:
        """Connect, move to the home pose with the claw open, and start the control loop. Blocking.

        Retries a few times (a dropped packet or a tripped motor usually clears on the next try).
        Returns whether the arm is running.
        """
        with self._lock:
            if self.state == "on":
                return True
            self.state = "starting"
            for attempt in range(START_ATTEMPTS):
                try:
                    arm, joints = self._open()
                    home = {**joints, "wrist_roll": 0.0, "gripper": GRIPPER_OPEN}
                    for _ in range(4):  # the grip offset depends on the final wrist pose
                        tip = HOME_GRIP - grip_offset(frames(home), 1.0)
                        home, _, _ = solve_ik(tip, POINTING_DEG, home, limits=self._limits)
                    move_to(arm, home, duration=2.0)
                    break
                except Exception:
                    log.exception("arm failed to start (attempt %d of %d)", attempt + 1, START_ATTEMPTS)
                    time.sleep(1.0)
            else:
                self.state = "off"
                return False
            self._arm, self.claw, self.pitch, self.joints = arm, GRIPPER_OPEN, POINTING_DEG, home
            self.grip = grip_point(home, 1.0)
            self.state = "on"
            self._thread = threading.Thread(target=self._loop, args=(home,), daemon=True)
            self._thread.start()
            return True

    def power_off(self) -> None:
        """Fold the arm up and release torque. Blocking; never raises."""
        with self._lock:
            if self.state != "on":
                return
            self.state = "stopping"
            self.stop()
            self._thread.join()
            try:
                move_to(self._arm, REST_JOINTS, duration=2.5)
            except Exception:
                log.exception("couldn't fold the arm")
            with contextlib.suppress(Exception):
                release(self._arm)
            self._arm, self.grip, self.state = None, None, "off"

    def status(self) -> dict:
        out = {"arm": self.state}
        if self.grip is not None:
            out["grip_cm"] = [round(float(v) * 100, 1) for v in self.grip]
            out["claw"] = round(self.claw)
        return out

    def _open(self):
        """Connect and hold the arm where it is. Returns (arm, joints); frees the port on failure."""
        arm = make_arm(max_relative_target=None)
        try:
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
            return arm, joints
        except Exception:
            with contextlib.suppress(Exception):
                arm.bus.disconnect(disable_torque=False)
            raise

    def _reconnect(self, joints: dict[str, float]) -> None:
        """Reopen the connection to the arm and send it back to where it was meant to be."""
        log.warning("lost the arm; reconnecting")
        with contextlib.suppress(Exception):
            self._arm.bus.disconnect(disable_torque=False)
        try:
            self._arm, _ = self._open()
            send_joints(self._arm, joints)
        except Exception:
            log.exception("reconnect failed; will retry")
            time.sleep(1.0)

    def _loop(self, joints: dict[str, float]) -> None:
        next_tick = time.perf_counter()
        failures = 0
        while self.state == "on":
            next_tick += 1 / HZ
            try:
                joints = self._tick(joints)
                failures = 0
            except Exception:
                failures += 1
                if failures == 1:
                    log.exception("control tick failed")
                if failures >= RECONNECT_AFTER:
                    self._reconnect(joints)
                    failures = 0
            time.sleep(max(0.0, next_tick - time.perf_counter()))
            next_tick = max(next_tick, time.perf_counter() - 1 / HZ)

    def _tick(self, joints: dict[str, float]) -> dict[str, float]:
        dt = 1 / HZ
        plan = self._plan
        if plan:
            inputs = plan.input
            plan.ticks -= 1
            if plan.ticks <= 0:
                self._plan = None
                plan.done.set()
        elif time.monotonic() - self._input_at > INPUT_STALE_S:
            inputs = NO_INPUT
        else:
            inputs = self._input
        x, y, z, wrist, grip, tilt = inputs
        self.applied = inputs
        goal = dict(joints)

        if x or y or z or tilt:
            target = self.grip + (self.right * x + self.away * y) * MOVE_SPEED * dt
            target[2] += z * Z_SPEED * dt
            pitch = float(np.clip(self.pitch + tilt * TILT_SPEED * dt, *PITCH_RANGE))
            tip = target - grip_offset(frames(joints), self.claw / GRIPPER_OPEN)
            solved, err, _ = solve_ik(tip, pitch, joints, limits=self._limits, iterations=30)
            if err <= IK_TOLERANCE:
                self.pitch = pitch
            else:  # out of reach at this angle: keep the angle and let the claw give a little instead
                solved, err, _ = solve_ik(
                    tip, self.pitch, joints, pitch_weight=0.005, limits=self._limits, iterations=30
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
        self.joints = goal
        return goal
