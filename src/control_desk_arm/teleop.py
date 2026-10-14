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
from control_desk_arm.kinematics import (
    LIMITS_DEG,
    POINTING_DEG,
    frames,
    grip_offset,
    grip_point,
    solve_ik,
)

HZ = 30.0
MOVE_SPEED = 0.10  # m/s at full input
Z_SPEED = 0.06  # m/s
WRIST_SPEED = 120.0  # deg/s
GRIP_SPEED = 150.0  # deg/s
TILT_SPEED = 45.0  # deg/s
INPUT_STALE_S = 0.4  # stop if the input stream pauses (lag, lost packets)
IK_TOLERANCE = 0.004
JOINT_SPEED = 60.0  # deg/s for the fastest joint during a planned move
PAN_AXIS_X = 0.0388  # where the base turns, in front of the origin (so101_new_calib.urdf)
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
START_ATTEMPTS = 5
RECONNECT_AFTER = 15  # consecutive failed ticks (half a second) before reconnecting to the arm
MEASURE_EVERY = 3  # ticks between readings of where the arm actually is
LAG_WARN_DEG = 8.0  # log when a joint is this far from where it was told to be
# Under gravity the servos settle a few degrees short of their goals. While the arm is (nearly) still,
# each joint's shortfall is added to what it's sent, a bit at a time, so it ends up where it was told.
SAG_GAIN = 0.3
SAG_MAX_DEG = 12.0  # cap, so a claw pressing on something can't wind this up
STILL_DEG = 1.0  # the arm counts as still when no goal moved more than this since the last reading

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
    """A move, run for exactly `ticks` control ticks. With a `path`, the arm follows those joint angles
    (one per tick); otherwise it holds `input` like a person would. `input` is also what the page's
    controls show either way."""

    input: tuple[float, ...]
    ticks: int
    tip: np.ndarray  # where the fixed fingertip will end up (unaffected by the claw or what it holds)
    path: list[dict[str, float]] | None = None
    pitch: float = POINTING_DEG
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
        self.measured: dict[str, float] = {}  # last read back from the motors: where the arm really is
        self.measured_at = 0.0  # time.monotonic() of that reading
        self._sag = dict.fromkeys(ARM_JOINTS, 0.0)  # added to each joint's goal, see SAG_GAIN
        self._still_since: dict[str, float] = {}  # goals at the last reading
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
        pitch = self.pitch + tilt_deg
        claw_goal = self.claw + claw_change
        roll = float(np.clip(self.joints["wrist_roll"] + wrist_deg, *self._limits["wrist_roll"]))
        path = self._joint_path(target, pitch, claw_goal, roll, seconds)
        if path:
            ticks = len(path)
            inputs = tuple(float(np.clip(v * t * HZ / ticks, -1, 1)) for v in inputs)  # same move, new duration
        self.stop()
        tip = frames(path[-1])[-1][:3, 3] if path else target - grip_offset(frames(self.joints), self.claw)
        self._plan = Plan(inputs, ticks, tip, path, pitch)
        self.applied = inputs
        return self._plan

    def _joint_path(
        self, grip: np.ndarray, pitch: float, claw: float, roll: float, min_seconds: float
    ) -> list[dict[str, float]] | None:
        """Joint angles for each tick of a smooth move to put the grip point at `grip`, or None if no
        arm pose reaches it. Every joint moves at once, so the arm can't get stuck partway the way a
        straight-line move can; if that would sweep the fingertips toward the table, it goes via a
        raised waypoint."""
        if np.allclose(grip, self.grip) and pitch == self.pitch and roll == self.joints["wrist_roll"]:
            goal = {**self.joints, "gripper": claw}  # only the claw changes: leave the arm exactly where it is
        else:
            goal = self._solve(grip, pitch, claw, roll)
        if goal is None:
            return None
        start = dict(self.joints)
        waypoints = [start, goal]
        lowest = min(self._fingertip_z(start), self._fingertip_z(goal))
        if min(self._fingertip_z(j) for j in _blend(start, goal, 20)) < min(lowest - 0.01, 0.02):
            above = self.grip.copy()
            above[2] = max(self.grip[2], grip[2]) + 0.06
            via = self._solve(above, pitch, claw, roll)
            if via is not None:
                waypoints = [start, via, goal]
        path = []
        for a, b in zip(waypoints, waypoints[1:]):
            biggest = max(abs(b[n] - a[n]) for n in ARM_JOINTS)
            seconds = max(biggest / JOINT_SPEED, min_seconds / (len(waypoints) - 1))
            path += _blend(a, b, max(1, math.ceil(seconds * HZ)))
        return path

    def _solve(self, grip: np.ndarray, pitch: float, claw: float, roll: float) -> dict[str, float] | None:
        """The reachable arm pose closest to the current one that puts the grip point at `grip`, ending
        with the claw `claw` open.

        The pose is solved at the current claw opening: opening or closing the claw swings only the moving
        jaw, so it never moves the arm (the grip point shifts with the jaw instead)."""
        pan = math.degrees(math.atan2(grip[1], grip[0] - PAN_AXIS_X))
        seeds = [self.joints] + [
            {**self.joints, "shoulder_pan": pan, "shoulder_lift": lift, "elbow_flex": elbow, "wrist_flex": 70.0}
            for lift, elbow in ((0.0, 0.0), (-40.0, 50.0), (30.0, -30.0), (-70.0, 85.0))
        ]
        best, best_cost = None, math.inf
        for pitch_weight in (0.1, 0.005):  # exact angle if possible, else let the claw tilt a little
            for seed in seeds:
                joints = {**seed, "wrist_roll": roll, "gripper": claw}
                for _ in range(4):  # the grip offset depends on the final wrist pose
                    tip = grip - grip_offset(frames(joints), self.claw)
                    joints, err, _ = solve_ik(tip, pitch, joints, pitch_weight=pitch_weight, limits=self._limits)
                cost = sum(abs(joints[n] - self.joints[n]) for n in ARM_JOINTS)
                if err <= IK_TOLERANCE and cost < best_cost:
                    best, best_cost = joints, cost
            if best is not None:
                return best
        return None

    @staticmethod
    def _fingertip_z(joints: dict[str, float]) -> float:
        return float(frames(joints)[-1][2, 3])

    def table_point(self, u: float, v: float, height: float = 0.0) -> np.ndarray:
        """The point at `height` meters above the table seen at image pixel (u, v)."""
        R, t = self.pose.R, self.pose.tvec
        center = -R.T @ t
        ray = R.T @ np.linalg.solve(self.pose.K, [u, v, 1.0])
        return center + ray * ((height - center[2]) / ray[2])

    def table_offset(self, delta: np.ndarray) -> tuple[float, float]:
        """Split a horizontal move into its `right` and `away` parts (meters)."""
        right, away = np.linalg.solve(np.column_stack([self.right[:2], self.away[:2]]), delta[:2])
        return float(right), float(away)

    @property
    def actual_grip(self) -> np.ndarray:
        """The grip point where the arm really is (from the motors), not where it was told to go."""
        return grip_point(self.measured)

    @property
    def actual_tip(self) -> np.ndarray:
        """The fixed fingertip, where the arm really is."""
        return frames(self.measured)[-1][:3, 3]

    @property
    def fingertip_z(self) -> float:
        return float(frames(self.measured)[-1][2, 3])

    def jaw_image_angle(self) -> float:
        """Across the jaws (fixed fingertip toward the moving jaw), in degrees counterclockwise from
        pointing right in the image."""
        f = frames(self.measured)
        fixed = f[-1][:3, 3]
        a, b = self.pose.project([fixed, fixed + grip_offset(f, 30.0)])  # direction only; any opening will do
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
                        tip = HOME_GRIP - grip_offset(frames(home), GRIPPER_OPEN)
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
            self.measured = dict(home)
            self._sag = dict.fromkeys(ARM_JOINTS, 0.0)
            self.grip = grip_point(home)
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
        failures = ticks = 0
        while self.state == "on":
            next_tick += 1 / HZ
            try:
                joints = self._tick(joints)
                failures = 0
                ticks += 1
                if ticks % MEASURE_EVERY == 0:
                    self._measure()
            except Exception:
                failures += 1
                if failures == 1:
                    log.exception("control tick failed")
                if failures >= RECONNECT_AFTER:
                    self._reconnect(joints)
                    failures = 0
            time.sleep(max(0.0, next_tick - time.perf_counter()))
            next_tick = max(next_tick, time.perf_counter() - 1 / HZ)

    def _measure(self) -> None:
        with contextlib.suppress(Exception):
            self.measured = read_joints(self._arm)
            self.measured_at = time.monotonic()
            goals, previous = dict(self.joints), self._still_since
            self._still_since = goals
            if not previous or max(abs(goals[n] - previous[n]) for n in ARM_JOINTS) > STILL_DEG:
                return
            for n in ARM_JOINTS:
                short = goals[n] - self.measured[n]
                self._sag[n] = float(np.clip(self._sag[n] + SAG_GAIN * short, -SAG_MAX_DEG, SAG_MAX_DEG))
            lag = {n: round(goals[n] - self.measured[n]) for n in ARM_JOINTS
                   if abs(goals[n] - self.measured[n]) > LAG_WARN_DEG}
            if lag:
                log.warning("arm isn't where it was told to be (commanded - actual, deg): %s", lag)

    def _send(self, goal: dict[str, float]) -> None:
        """Send joint goals, with the sag correction added (and kept inside the joints' ranges)."""
        sent = dict(goal)
        for n in ARM_JOINTS:
            sent[n] = float(np.clip(goal[n] + self._sag[n], *self._limits[n]))
        send_joints(self._arm, sent)

    def _tick(self, joints: dict[str, float]) -> dict[str, float]:
        dt = 1 / HZ
        plan = self._plan
        if plan and plan.path:
            goal = plan.path[len(plan.path) - plan.ticks]
            plan.ticks -= 1
            if plan.ticks <= 0:
                self._plan, self.pitch = None, plan.pitch
                plan.done.set()
            self.applied = plan.input
            self.claw = goal["gripper"]
            self.grip = grip_point(goal)
            self._send(goal)
            self.joints = goal
            return goal
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
            tip = target - grip_offset(frames(joints), self.claw)
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

        if grip:  # the moving jaw swings and the grip point (midway between the jaws) goes with it
            self.claw = float(np.clip(self.claw + grip * GRIP_SPEED * dt, 0.0, GRIPPER_OPEN))
            goal["gripper"] = self.claw
            self.grip = grip_point(goal)
        goal["gripper"] = self.claw
        self._send(goal)
        self.joints = goal
        return goal


def _blend(a: dict[str, float], b: dict[str, float], steps: int) -> list[dict[str, float]]:
    """`steps` poses easing from `a` to `b` (ending exactly at `b`)."""
    out = []
    for i in range(1, steps + 1):
        t = i / steps
        t = t * t * (3 - 2 * t)
        out.append({n: a[n] + (b[n] - a[n]) * t for n in b})
    return out
