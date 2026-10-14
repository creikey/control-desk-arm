"""Signaling relay between the robot PC and browsers.

Everyone connects to /ws. The robot (with ?role=robot&token=ROBOT_TOKEN) receives each browser's
messages tagged with the sender's id and addresses replies with `to`. Only the WebRTC offer/answer
goes through here. Browsers also get short-lived credentials for the coturn TURN server on this
machine, used when a direct connection to the robot isn't possible.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from urllib.parse import parse_qs, urlparse

from websockets import ConnectionClosed
from websockets.asyncio.server import ServerConnection, broadcast, serve

ROBOT_TOKEN = os.environ["ROBOT_TOKEN"]
TURN_SECRET = os.environ.get("TURN_SECRET", "")
PUBLIC_IP = os.environ.get("PUBLIC_IP", "")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8765"))

CREDENTIAL_TTL_S = 12 * 3600
MAX_CLIENTS = 64

log = logging.getLogger("signaling")
robot: ServerConnection | None = None
clients: dict[str, ServerConnection] = {}


def ice_servers(turn: bool) -> list[dict]:
    if not PUBLIC_IP:
        return []
    servers = [{"urls": f"stun:{PUBLIC_IP}:3478"}]
    if turn and TURN_SECRET:
        username = f"{int(time.time()) + CREDENTIAL_TTL_S}:arm"
        digest = hmac.new(TURN_SECRET.encode(), username.encode(), hashlib.sha1).digest()
        servers.append({
            "urls": [f"turn:{PUBLIC_IP}:3478?transport=udp", f"turn:{PUBLIC_IP}:3478?transport=tcp"],
            "username": username,
            "credential": base64.b64encode(digest).decode(),
        })
    return servers


async def send(ws: ServerConnection | None, msg: dict) -> None:
    if ws is None:
        return
    try:
        await ws.send(json.dumps(msg))
    except ConnectionClosed:
        pass


async def handle(ws: ServerConnection) -> None:
    url = urlparse(ws.request.path)
    query = parse_qs(url.query)
    if url.path != "/ws":
        await ws.close(4004, "Not found")
    elif query.get("role") == ["robot"]:
        if hmac.compare_digest(query.get("token", [""])[0], ROBOT_TOKEN):
            await handle_robot(ws)
        else:
            await ws.close(4003, "Bad robot token")
    elif len(clients) >= MAX_CLIENTS:
        await ws.close(4005, "Full")
    else:
        await handle_client(ws)


async def handle_robot(ws: ServerConnection) -> None:
    global robot
    if robot is not None:
        await robot.close(4000, "Replaced by a new robot connection")
    robot = ws
    log.info("robot connected")
    await send(ws, {"type": "welcome", "iceServers": ice_servers(turn=False)})
    broadcast(clients.values(), json.dumps({"type": "robot", "online": True}))
    try:
        async for raw in ws:
            msg = json.loads(raw)
            await send(clients.get(msg.get("to")), msg)
    finally:
        if robot is ws:
            robot = None
            log.info("robot disconnected")
            broadcast(clients.values(), json.dumps({"type": "robot", "online": False}))


async def handle_client(ws: ServerConnection) -> None:
    client_id = secrets.token_hex(4)
    clients[client_id] = ws
    await send(ws, {"type": "welcome", "robot": robot is not None, "iceServers": ice_servers(turn=True)})
    try:
        async for raw in ws:
            await send(robot, {**json.loads(raw), "from": client_id})
    finally:
        del clients[client_id]
        await send(robot, {"type": "bye", "from": client_id})


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    async with serve(handle, HOST, PORT, max_size=64 * 1024) as server:
        log.info("listening on %s:%d", HOST, PORT)
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
