"""Autopilot: Gemini drives the arm by planning exact moves of the claw.

Each step it sees what the page shows (a still frame at first, then a video of its last move) plus
the claw's height, opening and wrist angle, and asks for a move: so many cm right/away/up, a wrist
turn, a claw opening. The arm runs the move as the inputs a person would have held, so the page's
controls show what it's doing, and a magenta marker previews where the claw is headed.
"""

import asyncio
import base64
import io
import json
import logging
import os
import time
from fractions import Fraction
from pathlib import Path
from typing import Callable

import av
import cv2
import numpy as np
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from control_desk_arm.teleop import AXES, Teleop

MODEL = os.environ.get("ARM_AUTOPILOT_MODEL", "google/gemini-4-argon")
# At "low", Argon sometimes misjudges a successful grab as a miss and reopens the claw; "medium" doesn't.
REASONING_EFFORT = "medium"
KEY_FILE = Path(os.environ.get("APPDATA", Path.home() / ".config")) / "control-desk-arm" / "openrouter-api-key"
SETTLE_S = 0.5  # keep filming briefly after a move, so the video ends on where the arm landed
MAX_SETTLE_S = 1.5  # ...and up to this long while it's still settling onto its target
FPS = 10
CLIP_S = 6.0  # Gemini samples about one frame per second, so clips are slowed to play this long
MAX_STEPS = 30
KEEP_MEDIA = 2  # older observations drop their image or video
DUMP_DIR = os.environ.get("ARM_AUTOPILOT_DUMP")  # if set, save each observation here for debugging

SYSTEM_PROMPT = """You drive a small robot arm with a two-jaw claw on a desk to accomplish the user's goal. You see the same webcam view as the people watching.

The claw reaches about 15 to 30 cm out from the robot's base (at the left of the image). If a move ends short of its target, you're told.

Each turn, call one tool:
- `go_over`: move the claw sideways until it is straight above a point in the image, keeping its height. Point at the middle of the top of what you want and give that surface's height above the table in cm (0 for the table itself).
- `move`: a relative move, in directions on the table as seen in the image. right_cm, away_cm (away from the camera), up_cm, wrist_deg (turns the claw; positive turns it counterclockwise in the image), tilt_deg (positive swings the fingertips outward from pointing straight down), and claw (how open: 0 shut, 1 wide open). Everything in one move happens together, so split moves when the order matters.
- `finish`: once the goal is done, or can't be done.

After each move you get a slowed-down video of it and a close-up of the claw.

The claw has a fixed jaw and a moving jaw on a hinge that rises as it opens. At claw 0.2 the fingertips are about 4 cm apart, at 0.33 about 6 cm, at 0.45 about 8 cm; past about 0.5 the moving jaw mostly goes up rather than wider, so it would come down on top of an object instead of beside it. Open it a little wider than what you're grabbing.

To pick something up: get above it, turn the wrist so the jaws will close on two opposite sides of it, set the opening, lower until the fingertips are near the table, check in the close-up that it is between the jaws, close, lift, and check it came up. Opening the claw drops whatever it holds; to put something down, go over the spot, lower it close to the surface, then open."""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "go_over",
            "description": "Move the claw sideways so it is straight above this point in the image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "minimum": 0, "maximum": 1000, "description": "From the left edge."},
                    "y": {"type": "number", "minimum": 0, "maximum": 1000, "description": "From the top edge."},
                    "height_cm": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 30,
                        "description": "Height above the table of the surface you're pointing at.",
                    },
                },
                "required": ["x", "y", "height_cm"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": "Move the claw relative to where it is now. Leave out parts that shouldn't change.",
            "parameters": {
                "type": "object",
                "properties": {
                    "right_cm": {"type": "number", "minimum": -30, "maximum": 30},
                    "away_cm": {"type": "number", "minimum": -30, "maximum": 30},
                    "up_cm": {"type": "number", "minimum": -20, "maximum": 20},
                    "wrist_deg": {"type": "number", "minimum": -180, "maximum": 180},
                    "tilt_deg": {"type": "number", "minimum": -70, "maximum": 70},
                    "claw": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Stop: the goal is done, or can't be done.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string", "description": "One short sentence for the user."}},
                "required": ["summary"],
            },
        },
    },
]

log = logging.getLogger("autopilot")



class Autopilot:
    """Runs one goal at a time. `on_change` is called whenever what the page should show changes."""

    def __init__(self, teleop: Teleop, render: Callable[[], np.ndarray], on_change: Callable[[], None]):
        self.client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=KEY_FILE.read_text().strip())
        self.teleop, self.render, self.on_change = teleop, render, on_change
        self.goal = ""
        self.phase = ""  # thinking | executing
        self.note = ""
        self.step = 0
        self.tokens_in = 0  # of the latest model call
        self.tokens_out = 0
        self.tokens_per_s = 0.0
        self.cost = 0.0  # dollars, whole run
        self._task: asyncio.Task | None = None

    @property
    def on(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self, goal: str) -> None:
        self.stop()
        self.goal, self.note, self.step, self.cost = goal, "", 0, 0.0
        self._run_id = time.strftime("%Y%m%d-%H%M%S")
        self._task = asyncio.create_task(self._run(goal))

    def stop(self) -> None:
        if self.on:
            self.note = "stopped"
        if self._task:
            self._task.cancel()
            self._task = None
        self.teleop.stop()

    def status(self) -> dict:
        return {
            "on": self.on,
            "goal": self.goal,
            "phase": self.phase,
            "note": self.note,
            "step": self.step,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_per_s": round(self.tokens_per_s),
            "cost": round(self.cost, 4),
            "input": dict(zip(AXES, self.teleop.applied)),
        }

    async def _run(self, goal: str) -> None:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Goal: {goal}"}]
        frame = self.render()
        observation = [("Camera now:", "image", _jpeg(frame)), ("Close-up around the claw:", "image", self._closeup(frame))]
        result = ""
        try:
            for _ in range(MAX_STEPS):
                state = f"{result}{self._claw_state()}"
                log.info("state: %s", state)
                self._dump(state, observation)
                messages.append(_observation_message(state, observation))
                _forget_old_media(messages)
                message = await self._complete(messages)
                messages.append(message)
                if not message.get("tool_calls"):
                    messages.append({"role": "user", "content": "Call act or finish."})
                    continue
                call, *extra = message["tool_calls"]
                args = json.loads(call["function"]["arguments"] or "{}")
                log.info("%s %s", call["function"]["name"], args)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": "ok"})
                for other in extra:
                    messages.append({"role": "tool", "tool_call_id": other["id"], "content": "Ignored: one per turn."})
                if call["function"]["name"] == "finish":
                    self.note = args.get("summary", "done")
                    return
                video, frame, result = await self._execute(call["function"]["name"], args)
                observation = [
                    ("Video of your move:", "video", video),
                    ("Close-up around the claw now:", "image", self._closeup(frame)),
                ]
            self.note = "stopped after too many steps"
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("autopilot failed")
            self.note = "stopped"
        finally:
            self.phase = ""
            self.teleop.stop()
            self.on_change()

    def _set_phase(self, phase: str) -> None:
        self.phase = phase
        self.on_change()

    def _off_target(self, plan) -> float:
        return float(np.linalg.norm(self.teleop.actual_tip - plan.tip))

    def _claw_state(self) -> str:
        t = self.teleop
        return (
            f"Fingertips {t.fingertip_z * 100:.1f} cm above the table, "
            f"tilted {t.pitch + 90:.0f} degrees out from straight down, "
            f"jaws across the image at {t.jaw_image_angle():.0f} degrees (fixed jaw toward moving jaw, counterclockwise "
            "from pointing right)."
        )

    def _dump(self, state: str, media: list[tuple[str, str, bytes]]) -> None:
        if not DUMP_DIR:
            return
        folder = Path(DUMP_DIR) / self._run_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{self.step:03d}.txt").write_text(state)
        grip = ", ".join(f"{v * 100:.1f}" for v in self.teleop.grip)
        (folder / f"{self.step:03d}_grip.txt").write_text(grip)
        for i, (_, kind, data) in enumerate(media):
            (folder / f"{self.step:03d}_{i}.{'jpg' if kind == 'image' else 'mp4'}").write_bytes(data)

    def _closeup(self, img: np.ndarray, half: int = 120) -> bytes:
        """A 2x crop of the frame around the grip point."""
        u, v = self.teleop.pose.project([self.teleop.actual_grip])[0]
        h, w = img.shape[:2]
        x0 = int(np.clip(u - half, 0, w - 2 * half))
        y0 = int(np.clip(v - half, 0, h - 2 * half))
        crop = img[y0 : y0 + 2 * half, x0 : x0 + 2 * half]
        return _jpeg(cv2.resize(crop, (4 * half, 4 * half), interpolation=cv2.INTER_CUBIC))

    async def _execute(self, name: str, args: dict) -> tuple[bytes, np.ndarray, str]:
        """Run one move, filming the page's view. Returns a slowed-down MP4 of it, the last frame, and
        a note on whether it got there."""
        if name == "go_over":
            h, w = self.render().shape[:2]
            spot = self.teleop.table_point(
                float(args["x"]) / 1000 * w, float(args["y"]) / 1000 * h, float(args.get("height_cm") or 0) / 100
            )
            right, away = self.teleop.table_offset(spot - self.teleop.grip)
            args = {"right_cm": right * 100, "away_cm": away * 100}
        cm = {key: float(args.get(key) or 0) / 100 for key in ("right_cm", "away_cm", "up_cm")}
        claw = args.get("claw")
        plan = await asyncio.to_thread(
            self.teleop.plan_move,
            cm["right_cm"], cm["away_cm"], cm["up_cm"], float(args.get("wrist_deg") or 0),
            float(args.get("tilt_deg") or 0), None if claw is None else float(claw),
        )
        self._set_phase("executing")
        loop = asyncio.get_running_loop()
        frames, start, settle_until = [], loop.time(), None
        while settle_until is None or loop.time() < settle_until:
            if settle_until is None and plan.done.is_set():
                done_at = loop.time()
                settle_until = done_at + SETTLE_S
                self.on_change()
            if settle_until is not None and self._off_target(plan) > 0.005:  # still settling: wait, up to a limit
                settle_until = min(max(settle_until, loop.time() + 0.1), done_at + MAX_SETTLE_S)
            if len(frames) <= (loop.time() - start) * FPS:
                frames.append(self.render())
            await asyncio.sleep(0.05)
        settled = time.monotonic()
        while self.teleop.measured_at < settled:  # judge the move on a reading taken after it settled
            await asyncio.sleep(0.02)
        short = self._off_target(plan) * 100
        result = f"The claw ended {short:.1f} cm from the move's target (out of reach or blocked). " if short > 1 else ""
        return await asyncio.to_thread(_mp4, frames), frames[-1], result

    async def _complete(self, messages: list[dict]) -> dict:
        for attempt in range(4):
            try:
                return await self._call(messages)
            except APIStatusError as e:
                if e.status_code == 400 and "thought signature" in str(e).lower():
                    for m in messages:
                        m.pop("reasoning_details", None)
                elif e.status_code < 500 or attempt == 3:
                    raise
            except (APIConnectionError, APITimeoutError):
                if attempt == 3:
                    raise
            await asyncio.sleep(1 + attempt)

    async def _call(self, messages: list[dict]) -> dict:
        """One model call. Returns the assistant message, and records its usage for the page."""
        loop = asyncio.get_running_loop()
        self.step += 1
        self._set_phase("thinking")
        start = loop.time()
        response = await self.client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS, reasoning_effort=REASONING_EFFORT
        )
        usage = response.usage
        self.tokens_in, self.tokens_out = usage.prompt_tokens, usage.completion_tokens
        self.tokens_per_s = self.tokens_out / max(loop.time() - start, 0.1)
        self.cost += (usage.model_extra or {}).get("cost") or 0.0
        message = response.choices[0].message.model_dump(exclude_none=True)
        message.pop("refusal", None)
        return message


def _observation_message(text: str, media: list[tuple[str, str, bytes]]) -> dict:
    """The claw's state as text, then each labeled image or video."""
    content = [{"type": "text", "text": text}]
    for label, kind, data in media:
        content.append({"type": "text", "text": label})
        if kind == "image":
            url = "data:image/jpeg;base64," + base64.b64encode(data).decode()
            content.append({"type": "image_url", "image_url": {"url": url}})
        else:
            url = "data:video/mp4;base64," + base64.b64encode(data).decode()
            content.append({"type": "video_url", "video_url": {"url": url}})
    return {"role": "user", "content": content}


def _forget_old_media(messages: list[dict]) -> None:
    observations = [m for m in messages if m["role"] == "user" and isinstance(m["content"], list)]
    for m in observations[:-KEEP_MEDIA]:
        m["content"] = m["content"][0]["text"] + " (no longer shown)"


def _jpeg(img: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def _mp4(frames: list[np.ndarray]) -> bytes:
    buf = io.BytesIO()
    with av.open(buf, "w", format="mp4", options={"movflags": "frag_keyframe+empty_moov"}) as container:
        rate = Fraction(max(1, round(len(frames) / CLIP_S * 10)), 10)
        stream = container.add_stream("libx264", rate=rate)
        stream.height, stream.width = frames[0].shape[:2]
        stream.pix_fmt, stream.options = "yuv420p", {"crf": "26"}
        for img in frames:
            container.mux(stream.encode(av.VideoFrame.from_ndarray(img, format="bgr24")))
        container.mux(stream.encode())
    return buf.getvalue()
