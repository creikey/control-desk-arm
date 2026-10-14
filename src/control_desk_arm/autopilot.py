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
SETTLE_S = 0.3  # keep filming briefly after a move, so the video ends on where the arm landed
FPS = 10
CLIP_S = 6.0  # Gemini samples about one frame per second, so clips are slowed to play this long
MAX_STEPS = 30
KEEP_MEDIA = 2  # older observations drop their image or video
DUMP_DIR = os.environ.get("ARM_AUTOPILOT_DUMP")  # if set, save each observation here for debugging

SYSTEM_PROMPT = """You drive a small robot arm on a desk to accomplish the user's goal. You see the same webcam view as the people watching. Overlaid on the image: a white line at fingertip height shows where the jaws grip. It runs from the fixed jaw's fingertip to where the moving jaw reaches at the current opening (a dot at each end); when the claw closes, whatever crosses that line gets squeezed. The cyan ring is the grip point, midway between the jaws. The translucent cyan disk is on the table, directly under the claw (like a shadow, about the claw's width), with a line up to the ring; since it lies on the table, objects standing on that spot cover it. While a move runs, a magenta ring and disk outline show where it is headed.

The claw reaches between about 15 and 30 cm out from the robot's base (the base is at the left of the image): it can't get to things further away, or right next to the base. If a move stops short, its target is out of reach.

Each turn, call one tool: `go_over`, `move`, or, once the goal is done or clearly impossible, `finish`.

`go_over` takes a point in the camera image (x from the left edge, y from the top, both 0 to 1000) and the height of what you're pointing at, and moves the claw sideways so it is straight above that point, keeping its own height. Point at the middle of the object's top surface and give that surface's height above the table in cm (a 2x2 Rubik's cube is about 5 cm tall); for a spot on the table itself, use 0. This is the accurate way to get over an object; use it first, then fine-tune with `move`.

`move` is relative to where the claw is now, in directions on the table as seen in the image:
- right_cm: toward the right of the image (negative: left).
- away_cm: away from the camera, up the image (negative: toward the camera).
- up_cm: up (negative: down).
- wrist_deg: turn the claw about its vertical axis. Positive turns the white line counterclockwise in the image.
- tilt_deg: tilt the claw about the grip point. Positive swings its fingertips outward, away from the robot's base; negative brings it back toward pointing straight down. Useful to reach over or around things, or to grab something from the side; straight down is best for most picks.
- claw: how open the claw should be, 0 (shut) to 1 (wide open). Leave it out to keep it as is.
Everything in a move happens at once and arrives together, so split moves when the order matters (for example: open, then lower, then close, then lift). Moves are exact; make small corrections when you're close.

After each move you get a slowed-down video of it, a close-up of the claw at the end, and the claw's height, tilt and jaw direction, plus a note if the move didn't reach its target (it stops short when the target is out of reach).

To pick something up:
1. `go_over` the middle of the object's top surface, with the claw high enough to clear it.
2. Turn the wrist so the white line between the fingertips crosses the object: across a long thin object like a pencil, and straight across two opposite faces of a box or cube (not across its corners). Open the claw fully.
3. Lower until the fingertips are about 1 cm above the table, so the jaws surround the object's lower half.
4. Before closing, look at the claw itself in the close-up, not the overlay (the overlay is drawn on top of everything, so its lines can look like they're on an object that is actually behind or in front of them). The fixed jaw should be right beside one side of the object and the open moving jaw beyond the opposite side, with the object centered between them. If either jaw is in front of the object (covering it from the camera), on top of it, or touching it, the claw is off-center: move to center the object between the jaws first, then check again.
5. Close the claw (a move with only `claw: 0`), then lift 5 cm.
6. Check the video and close-up: if the object came up with the claw, you have it. Only if the object was left behind on the table, open and try again from step 1.

Opening the claw drops whatever it holds. To put a held object somewhere:
1. Find the destination in the full camera view (not the close-up, which only shows the area around the claw).
2. `go_over` the middle of the destination's surface, with the object held high enough to clear things on the way.
3. Lower until the object is just above the destination's surface (about 1 cm).
4. Open the claw in a move of its own, then lift away and check the object stayed there.

If you're stuck after a few tries, call `finish` and say why."""

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

    def _claw_state(self) -> str:
        t = self.teleop
        return (
            f"Fingertips {t.fingertip_z * 100:.1f} cm above the table, "
            f"tilted {t.pitch + 90:.0f} degrees out from straight down, "
            f"jaws closing along {t.jaw_image_angle():.0f} degrees in the image (counterclockwise from pointing right)."
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
        u, v = self.teleop.pose.project([self.teleop.grip])[0]
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
                settle_until = loop.time() + SETTLE_S
                self.on_change()
            if len(frames) <= (loop.time() - start) * FPS:
                frames.append(self.render())
            await asyncio.sleep(0.05)
        short = float(np.linalg.norm(self.teleop.grip - plan.target)) * 100
        result = f"The move stopped {short:.1f} cm short of its target (out of reach). " if short > 0.5 else ""
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
