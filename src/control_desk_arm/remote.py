"""Stream the webcam to browsers over WebRTC and let one of them at a time drive the arm.

Browsers find this PC through the signaling server in `signaling/`. Each one gets a peer connection
with the video and a "control" data channel. Whoever has control keeps it until they release it,
leave, go quiet (page hidden, phone locked), or send no input for IDLE_BOOT_S.
"""

import asyncio
import json
import logging
import os
import socket
import time
from pathlib import Path

import aioice.ice
import av
import cv2
import numpy as np
import websockets
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRelay

from control_desk_arm.autopilot import Autopilot
from control_desk_arm.camera import Camera, CameraPose
from control_desk_arm.teleop import AXES, Teleop

SIGNAL_URL = os.environ.get("ARM_SIGNAL_URL", "wss://167-99-173-249.sslip.io")
TOKEN_FILE = Path(os.environ.get("APPDATA", Path.home() / ".config")) / "control-desk-arm" / "robot-token"

IDLE_BOOT_S = 30.0  # no input from the controller for this long releases control
SILENT_BOOT_S = 3.0  # the controller's page sends input 20 times a second; silence means it's gone
REST_AFTER_S = 60.0  # with nobody in control this long, fold the arm and turn torque off
MAX_GOAL_CHARS = 500
PUSH_INTERVAL_S = 0.1  # autopilot updates (token counts, phase) go out at most this often
MARKER_COLOR = (255, 255, 0)  # cyan (BGR)
PLAN_COLOR = (255, 0, 255)  # magenta: where an autopilot move is headed

log = logging.getLogger("arm-remote")


def _default_route_address(use_ipv4: bool = True, use_ipv6: bool = False) -> list[str]:
    # aioice queries STUN through every adapter, and on VPN adapters that times out after 5 s.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("1.1.1.1", 53))  # picks the route; sends nothing
        return [s.getsockname()[0]]


aioice.ice.get_host_addresses = _default_route_address


class CameraTrack(VideoStreamTrack):
    def __init__(self, render):
        super().__init__()
        self.render = render

    async def recv(self) -> av.VideoFrame:
        pts, time_base = await self.next_timestamp()
        frame = av.VideoFrame.from_ndarray(self.render(), format="bgr24")
        frame.pts, frame.time_base = pts, time_base
        return frame


class Remote:
    def __init__(self, camera: Camera, pose: CameraPose):
        self.camera, self.pose = camera, pose
        self.teleop = Teleop(pose)
        self.autopilot = Autopilot(self.teleop, self.render, self._autopilot_changed)
        self.last_push = 0.0
        self.push_pending = False
        self.track = CameraTrack(self.render)
        self.relay = MediaRelay()
        self.peers: dict[str, RTCPeerConnection] = {}
        self.channels: dict[str, object] = {}
        self.controller: str | None = None
        self.last_active = 0.0  # last real input from the controller
        self.last_heard = 0.0  # last message of any kind from the controller
        self.free_since = time.monotonic()
        self.ice: list[RTCIceServer] = []
        self.ws = None

    def render(self) -> np.ndarray:
        """The webcam, with the grip point (and any planned move's target) marked down to the table, and a
        bar through the grip point along the line the jaws close on."""
        img = self.camera.latest_frame()
        img = np.zeros((480, 640, 3), np.uint8) if img is None else img.copy()
        if self.teleop.grip is not None:
            a, b = self.pose.project(self.teleop.jaw_ends()).astype(int)
            cv2.line(img, tuple(a), tuple(b), (0, 0, 0), 5, cv2.LINE_AA)
            cv2.line(img, tuple(a), tuple(b), (255, 255, 255), 2, cv2.LINE_AA)
        for point, color in ((self.teleop.plan_target, PLAN_COLOR), (self.teleop.grip, MARKER_COLOR)):
            if point is not None:
                top, floor = self.pose.project([point, (point[0], point[1], 0.0)]).astype(int)
                cv2.line(img, tuple(top), tuple(floor), color, 1, cv2.LINE_AA)
                cv2.circle(img, tuple(floor), 3, color, -1, cv2.LINE_AA)
                cv2.circle(img, tuple(top), 6, color, 2, cv2.LINE_AA)
        return img

    async def run(self) -> None:
        token = TOKEN_FILE.read_text().strip()
        asyncio.create_task(self._watchdog())
        while True:
            try:
                async with websockets.connect(f"{SIGNAL_URL}/ws?role=robot&token={token}") as ws:
                    self.ws = ws
                    log.info("connected to %s", SIGNAL_URL)
                    async for raw in ws:
                        try:
                            await self._on_signal(json.loads(raw))
                        except Exception:
                            log.exception("couldn't handle a signaling message")
            except (OSError, websockets.WebSocketException) as e:
                log.warning("signaling connection lost (%s); reconnecting", e)
            self.ws = None
            await asyncio.sleep(2)

    async def _on_signal(self, msg: dict) -> None:
        kind, peer = msg.get("type"), msg.get("from")
        if kind == "welcome":
            self.ice = [RTCIceServer(**s) for s in msg["iceServers"]]
        elif kind == "offer":
            await self._answer(peer, msg["sdp"])
        elif kind == "bye":
            await self._drop(peer)

    async def _answer(self, peer: str, sdp: str) -> None:
        await self._drop(peer)
        pc = RTCPeerConnection(RTCConfiguration(iceServers=self.ice))
        self.peers[peer] = pc
        pc.addTrack(self.relay.subscribe(self.track, buffered=False))

        @pc.on("datachannel")
        def on_datachannel(channel):
            self.channels[peer] = channel
            channel.on("message", lambda raw: self._on_control(peer, raw))
            channel.on("close", lambda: asyncio.ensure_future(self._drop(peer)))
            self._send_status(peer)

        @pc.on("connectionstatechange")
        async def on_state():
            log.info("%s %s", peer, pc.connectionState)
            if pc.connectionState in ("failed", "closed"):
                await self._drop(peer)

        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
        await pc.setLocalDescription(await pc.createAnswer())  # gathers every candidate first
        await self.ws.send(json.dumps({"type": "answer", "to": peer, "sdp": pc.localDescription.sdp}))

    async def _drop(self, peer: str) -> None:
        self.channels.pop(peer, None)
        pc = self.peers.pop(peer, None)
        if self.controller == peer:
            self._release("left")
        if pc:
            await pc.close()
            self._broadcast_status()

    def _on_control(self, peer: str, raw) -> None:
        try:
            msg = json.loads(raw)
        except (TypeError, ValueError):
            return
        kind = msg.get("type")
        if kind == "take":
            if self.controller is None:
                self.controller = peer
                self.last_active = self.last_heard = time.monotonic()
                self._broadcast_status()
                asyncio.ensure_future(self._power_on())
            return
        if self.controller != peer:
            return  # only the controller can do anything else
        self.last_heard = time.monotonic()
        if kind == "release":
            self._release("released")
        elif kind == "autopilot":
            goal = str(msg.get("goal", "")).strip()[:MAX_GOAL_CHARS]
            if goal and self.teleop.state == "on":
                self.autopilot.start(goal)
                self._broadcast_status()
        elif kind == "autopilot_stop":
            self.autopilot.stop()
            self._broadcast_status()
        elif kind == "input" and not self.autopilot.on:
            values = [float(msg.get(k) or 0) for k in AXES]
            self.teleop.set_input(*values)
            if any(abs(v) > 0.05 for v in values):
                self.last_active = time.monotonic()

    async def _power_on(self) -> None:
        if await asyncio.to_thread(self.teleop.power_on):
            self.last_active = time.monotonic()  # starting up doesn't count as idle time
        elif self.controller:
            self._release("unavailable")
        self._broadcast_status()

    def _release(self, reason: str) -> None:
        peer, self.controller = self.controller, None
        self.autopilot.stop()
        self.free_since = time.monotonic()
        self._send(peer, {"type": "dropped", "reason": reason})
        self._broadcast_status()

    async def _watchdog(self) -> None:
        while True:
            await asyncio.sleep(0.25)
            try:
                await self._check()
            except Exception:
                log.exception("watchdog check failed")

    async def _check(self) -> None:
        now = time.monotonic()
        if self.autopilot.on:
            self.last_active = now
        if self.controller and now - self.last_heard > SILENT_BOOT_S:
            self._release("gone")
        elif self.controller and self.teleop.state != "starting" and now - self.last_active > IDLE_BOOT_S:
            self._release("idle")
        if not self.controller and self.teleop.state == "on" and now - self.free_since > REST_AFTER_S:
            await asyncio.to_thread(self.teleop.power_off)
        self._broadcast_status()

    def _autopilot_changed(self) -> None:
        """Push autopilot progress to every page right away, throttled to PUSH_INTERVAL_S."""
        if self.push_pending:
            return
        wait = self.last_push + PUSH_INTERVAL_S - time.monotonic()
        if wait > 0:
            self.push_pending = True
            asyncio.get_running_loop().call_later(wait, self._push)
        else:
            self._push()

    def _push(self) -> None:
        self.push_pending = False
        self.last_push = time.monotonic()
        self._broadcast_status()

    def _send_status(self, peer: str) -> None:
        idle_left = None
        if self.controller == peer and self.teleop.state == "on":
            idle_left = round(max(0.0, IDLE_BOOT_S - (time.monotonic() - self.last_active)), 1)
        self._send(peer, {
            "type": "status",
            "you": self.controller == peer,
            "busy": self.controller not in (None, peer),
            "viewers": len(self.peers),
            "idle_left": idle_left,
            "autopilot": self.autopilot.status(),
            **self.teleop.status(),
        })

    def _broadcast_status(self) -> None:
        for peer in list(self.channels):
            self._send_status(peer)

    def _send(self, peer: str | None, msg: dict) -> None:
        channel = self.channels.get(peer)
        if channel is not None and channel.readyState == "open":
            channel.send(json.dumps(msg))

    async def close(self) -> None:
        for peer in list(self.peers):
            await self._drop(peer)
        await asyncio.to_thread(self.teleop.power_off)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", force=True)
    for noisy in ("aioice", "aiortc", "websockets", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    camera = Camera()
    remote = Remote(camera, CameraPose())

    async def run():
        try:
            await remote.run()
        finally:
            await remote.close()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
