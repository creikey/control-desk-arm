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
from fractions import Fraction
from pathlib import Path
from typing import Callable

import av
import cv2
import numpy as np
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from control_desk_arm.teleop import AXES, GRIPPER_OPEN, Teleop

MODEL = os.environ.get("ARM_AUTOPILOT_MODEL", "google/gemini-4-argon")
KEY_FILE = Path(os.environ.get("APPDATA", Path.home() / ".config")) / "control-desk-arm" / "openrouter-api-key"
SETTLE_S = 0.3  # keep filming briefly after a move, so the video ends on where the arm landed
FPS = 10
CLIP_S = 6.0  # Gemini samples about one frame per second, so clips are slowed to play this long
MAX_STEPS = 120
KEEP_MEDIA = 2  # older observations drop their image or video

SYSTEM_PROMPT = """You drive a small robot arm on a desk to accomplish the user's goal. You see the same webcam view as the people watching. The cyan ring marks the claw's grip point (the middle of the open claw); the cyan line drops from it straight down to a dot on the table, directly below the claw. The short white bar through the ring is the line the two jaws close along: an object is only gripped if it sits between the jaws across that bar. While a move runs, a magenta ring and dot show where it is headed.

Each turn, call `move` or, once the goal is done or clearly impossible, `finish`. A move is relative to where the claw is now, in directions on the table as seen in the image:
- right_cm: toward the right of the image (negative: left).
- away_cm: away from the camera, up the image (negative: toward the camera).
- up_cm: up (negative: down).
- wrist_deg: turn the claw about its vertical axis. Positive turns the white bar counterclockwise in the image.
- tilt_deg: tilt the claw about the grip point. Positive swings its fingertips outward, away from the robot's base; negative brings it back toward pointing straight down. Useful to reach over or around things, or to grab something from the side; straight down is best for most picks.
- claw: how open the claw should be, 0 (shut) to 1 (wide open). Leave it out to keep it as is.
Everything in a move happens at once and arrives together, so split moves when the order matters (for example: open, then lower, then close, then lift). Moves are exact; make small corrections when you're close.

After each move you get a slowed-down video of it and the claw's state. To pick something up: turn the wrist so the bar crosses the object (for a long thin object like a pencil, at a right angle to it; for something round it doesn't matter), put the dot under the claw on the object, open the claw, lower it until the fingertips are about 1 cm above the table (or around the object), close it, then lift and check the object came with it."""

TOOLS = [
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
        observation = ("Camera now.", "image", _jpeg(self.render()))
        try:
            for _ in range(MAX_STEPS):
                text, kind, data = observation
                messages.append(_observation_message(f"{text} {self._claw_state()}", kind, data))
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
                observation = ("Video of your move.", "video", await self._execute(args))
            self.note = "stopped after too many steps"
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("autopilot failed")
            self.note = f"error: {type(e).__name__}"
        finally:
            self.phase = ""
            self.teleop.stop()
            self.on_change()

    def _set_phase(self, phase: str) -> None:
        self.phase = phase
        self.on_change()

    def _claw_state(self) -> str:
        t = self.teleop
        return (
            f"Fingertips {t.fingertip_z * 100:.1f} cm above the table, claw {t.claw / GRIPPER_OPEN:.0%} open, "
            f"tilted {t.pitch + 90:.0f} degrees out from straight down, "
            f"jaws closing along {t.jaw_image_angle():.0f} degrees in the image (counterclockwise from pointing right)."
        )

    async def _execute(self, args: dict) -> bytes:
        """Run one move, filming the page's view; returns a slowed-down MP4 of it."""
        cm = {key: float(args.get(key) or 0) / 100 for key in ("right_cm", "away_cm", "up_cm")}
        claw = args.get("claw")
        plan = self.teleop.plan_move(
            cm["right_cm"], cm["away_cm"], cm["up_cm"], float(args.get("wrist_deg") or 0),
            float(args.get("tilt_deg") or 0), None if claw is None else float(claw),
        )
        self._set_phase("executing")
        loop = asyncio.get_running_loop()
        frames, start, settle_until = [], loop.time(), None
        while settle_until is None or loop.time() < settle_until:
            if settle_until is None and plan.done.is_set():
                settle_until = loop.time() + SETTLE_S
                self.on_change()
            if len(frames) <= (loop.time() - start) * FPS:
                frames.append(self.render())
            await asyncio.sleep(0.05)
        return await asyncio.to_thread(_mp4, frames)

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
            model=MODEL, messages=messages, tools=TOOLS, reasoning_effort="low"
        )
        usage = response.usage
        self.tokens_in, self.tokens_out = usage.prompt_tokens, usage.completion_tokens
        self.tokens_per_s = self.tokens_out / max(loop.time() - start, 0.1)
        self.cost += (usage.model_extra or {}).get("cost") or 0.0
        message = response.choices[0].message.model_dump(exclude_none=True)
        message.pop("refusal", None)
        return message


def _observation_message(text: str, kind: str, data: bytes) -> dict:
    if kind == "image":
        media = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(data).decode()}}
    else:
        media = {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(data).decode()}}
    return {"role": "user", "content": [{"type": "text", "text": text}, media]}


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
