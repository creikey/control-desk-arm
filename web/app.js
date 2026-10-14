// Watch the desk arm's webcam over WebRTC and, when nobody else is driving, drive it.

const params = new URLSearchParams(location.search);
const SIGNAL = params.get("signal") || "wss://167-99-173-249.sslip.io";
const RELAY_ONLY = params.has("relay"); // force traffic through the TURN server, for testing
const SEND_HZ = 20;
const DROP_REASONS = {
  idle: "control released after 30 s without input",
  gone: "control released: page went to the background",
  unavailable: "the arm isn't responding right now, try again in a minute",
};

const $ = (id) => document.getElementById(id);
const video = $("video"), statusEl = $("status"), info = $("info");
const pilot = $("pilot"), pilotGoal = $("pilot-goal"), pilotPhase = $("pilot-phase"), pilotStats = $("pilot-stats");
const takeBtn = $("take"), autoBtn = $("auto"), pad = $("pad");
const dialog = $("goal-dialog"), goalInput = $("goal");

let ws, pc, dc, iceServers = [];
let state = {}, note = "", route = "", wasAuto = false, thinkingSince = 0;

// ---- connection ---------------------------------------------------------------------------------

function connectSignal() {
  ws = new WebSocket(SIGNAL + "/ws");
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === "welcome") iceServers = msg.iceServers;
    if (msg.type === "welcome" || msg.type === "robot") {
      msg.robot || msg.online ? connectPeer() : (closePeer(), show("robot offline"));
    }
    if (msg.type === "answer" && pc) pc.setRemoteDescription({ type: "answer", sdp: msg.sdp });
  };
  ws.onclose = () => {
    show("reconnecting…");
    setTimeout(connectSignal, 2000);
  };
}

async function connectPeer() {
  closePeer();
  show("connecting video…");
  const peer = (pc = new RTCPeerConnection({ iceServers, iceTransportPolicy: RELAY_ONLY ? "relay" : "all" }));
  peer.addTransceiver("video", { direction: "recvonly" });
  peer.ontrack = (e) => {
    video.srcObject = e.streams[0] || new MediaStream([e.track]);
    video.play().catch(() => {});
  };
  peer.onconnectionstatechange = () => {
    if (peer !== pc) return;
    if (peer.connectionState === "connected") {
      show("");
      showRoute(peer);
    }
    if (peer.connectionState === "failed") {
      show("connection failed, retrying…");
      setTimeout(() => peer === pc && connectPeer(), 2000);
    }
  };
  dc = peer.createDataChannel("control");
  dc.onmessage = (e) => onRobot(JSON.parse(e.data));
  dc.onclose = () => { state = {}; render(); };

  // The robot doesn't take candidates one by one, so they all go in the offer. Rather than wait for
  // gathering to formally finish (slow), send once the candidate that matters most has arrived.
  const hasTurn = iceServers.some((s) => [].concat(s.urls).some((u) => u.startsWith("turn")));
  const wanted = hasTurn ? " typ relay" : " typ srflx";
  await peer.setLocalDescription(await peer.createOffer());
  await new Promise((done) => {
    if (peer.iceGatheringState === "complete") return done();
    peer.onicegatheringstatechange = () => peer.iceGatheringState === "complete" && done();
    peer.onicecandidate = (e) => e.candidate && e.candidate.candidate.includes(wanted) && setTimeout(done, 100);
    setTimeout(done, 3000);
  });
  if (peer === pc) ws.send(JSON.stringify({ type: "offer", sdp: peer.localDescription.sdp }));
}

async function showRoute(peer) {
  const stats = await peer.getStats();
  for (const s of stats.values()) {
    const pair = s.type === "transport" && stats.get(s.selectedCandidatePairId);
    if (!pair) continue;
    const ends = [stats.get(pair.localCandidateId), stats.get(pair.remoteCandidateId)];
    route = ends.some((c) => c && c.candidateType === "relay") ? "relayed via TURN" : "direct";
  }
  render();
}

function closePeer() {
  if (pc) pc.close();
  pc = dc = null;
  state = {};
  route = "";
  render();
}

function send(msg) {
  if (dc && dc.readyState === "open") dc.send(JSON.stringify(msg));
}

// ---- control state ------------------------------------------------------------------------------

function onRobot(msg) {
  if (msg.type === "status") {
    const auto = msg.autopilot || {};
    if (wasAuto && !auto.on) {
      if (auto.note) note = `autopilot: ${auto.note}`;
      resetPad();
    }
    wasAuto = !!auto.on;
    if (auto.phase === "thinking" && (state.autopilot || {}).phase !== "thinking") thinkingSince = performance.now();
    state = msg;
  }
  if (msg.type === "dropped") note = DROP_REASONS[msg.reason] || "";
  render();
}

function show(text) {
  statusEl.textContent = text;
}

function render() {
  const open = dc && dc.readyState === "open";
  const driving = !!state.you;
  const auto = state.autopilot && state.autopilot.on ? state.autopilot : null;

  pad.hidden = !driving;
  document.body.classList.toggle("driving", driving);
  pad.classList.toggle("auto", !!auto);
  takeBtn.textContent = driving ? "release control" : !state.busy ? "take control" : auto ? "AI is driving" : "someone else is driving";
  takeBtn.disabled = !open || state.busy;
  autoBtn.hidden = !(driving && state.arm === "on");
  autoBtn.classList.toggle("on", !!auto);
  autoBtn.textContent = auto ? "✦ stop autopilot" : "✦ autopilot";
  pilot.hidden = !auto;
  if (auto) {
    pilotGoal.textContent = auto.goal;
    pilotPhase.className = auto.phase;
    pilotPhase.textContent =
      auto.phase === "thinking" ? `thinking · ${((performance.now() - thinkingSince) / 1000).toFixed(1)} s`
      : auto.phase === "executing" ? "executing"
      : "starting…";
    pilotStats.textContent = auto.tokens_out
      ? `step ${auto.step} · ${auto.tokens_in} tokens in · ${auto.tokens_out} out · ${auto.tokens_per_s} tok/s · $${auto.cost.toFixed(3)}`
      : `step ${auto.step}`;
    if (driving) showInputs(auto.input);
  }

  const parts = [];
  if (route) parts.push(route);
  if (state.viewers) parts.push(`${state.viewers} watching`);
  if (driving && state.arm === "starting") parts.push("starting arm…");
  if (driving && !auto && state.idle_left != null && state.idle_left < 10) {
    parts.push(`idle, releasing in ${Math.ceil(state.idle_left)}`);
  }
  if (note) parts.push(note);
  info.textContent = parts.join(" · ");
}

// Mirror the autopilot's controls on the pad, so you can see what it's pressing.
function showInputs(input) {
  const radius = stick.getBoundingClientRect().width / 2;
  knob.style.transform = `translate(${input.x * radius}px, ${-input.y * radius}px)`;
  for (const btn of document.querySelectorAll("#buttons button")) {
    const value = input[btn.dataset.axis];
    btn.classList.toggle("held", Math.sign(value) === Number(btn.dataset.dir));
  }
}

takeBtn.onclick = () => {
  note = "";
  send({ type: state.you ? "release" : "take" });
};

autoBtn.onclick = () => {
  if (state.autopilot && state.autopilot.on) {
    send({ type: "autopilot_stop" });
    state.autopilot.on = false; // show it stopped right away
    render();
    resetPad();
  } else {
    dialog.showModal();
    // On phones, leave the keyboard down until they tap the box, so the suggestions stay visible.
    if (matchMedia("(pointer: coarse)").matches) document.activeElement.blur();
  }
};

for (const idea of document.querySelectorAll("#ideas button")) {
  idea.onclick = () => { goalInput.value = idea.textContent; };
}
$("goal-cancel").onclick = () => dialog.close();
$("goal-form").onsubmit = () => {
  const goal = goalInput.value.trim();
  if (!goal) return false;
  note = "";
  resetPad();
  send({ type: "autopilot", goal });
};

// Leaving the page (tab closed, app switched, phone locked) gives up control, ending any autopilot.
function leave() {
  if (state.you) send({ type: "release" });
}
document.addEventListener("visibilitychange", () => document.hidden && leave());
addEventListener("pagehide", leave);

// ---- inputs -------------------------------------------------------------------------------------

const touch = { x: 0, y: 0, z: 0, wrist: 0, grip: 0, tilt: 0 };
const keys = new Set();

// Joystick output is -1..1, with a small dead zone and a curve so small offsets give fine moves.
const stick = $("stick"), knob = $("knob");
let stickPointer = null;

function moveStick(e) {
  const r = stick.getBoundingClientRect();
  const radius = r.width / 2;
  let dx = e.clientX - (r.left + radius), dy = e.clientY - (r.top + radius);
  const len = Math.hypot(dx, dy);
  if (len > radius) { dx *= radius / len; dy *= radius / len; }
  knob.style.transform = `translate(${dx}px, ${dy}px)`;
  const mag = Math.min(1, len / radius);
  const scaled = mag < 0.12 ? 0 : ((mag - 0.12) / 0.88) ** 1.6;
  touch.x = len ? (dx / len) * scaled : 0;
  touch.y = len ? (-dy / len) * scaled : 0;
}

function resetStick() {
  stickPointer = null;
  knob.style.transform = "";
  touch.x = touch.y = 0;
}

function resetPad() {
  resetStick();
  for (const axis in touch) touch[axis] = 0;
  for (const btn of document.querySelectorAll("#buttons button")) btn.classList.remove("held");
}

stick.addEventListener("pointerdown", (e) => {
  stickPointer = e.pointerId;
  stick.setPointerCapture(e.pointerId);
  moveStick(e);
});
stick.addEventListener("pointermove", (e) => e.pointerId === stickPointer && moveStick(e));
for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) {
  stick.addEventListener(type, (e) => e.pointerId === stickPointer && resetStick());
}

for (const btn of document.querySelectorAll("#buttons button")) {
  const axis = btn.dataset.axis, dir = Number(btn.dataset.dir);
  const release = () => {
    btn.classList.remove("held");
    if (touch[axis] === dir) touch[axis] = 0;
  };
  btn.addEventListener("pointerdown", (e) => {
    btn.setPointerCapture(e.pointerId);
    btn.classList.add("held");
    touch[axis] = dir;
  });
  for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) btn.addEventListener(type, release);
}
document.addEventListener("contextmenu", (e) => e.preventDefault());

const KEYMAP = {
  w: ["y", 1], arrowup: ["y", 1], s: ["y", -1], arrowdown: ["y", -1],
  d: ["x", 1], arrowright: ["x", 1], a: ["x", -1], arrowleft: ["x", -1],
  r: ["z", 1], f: ["z", -1], q: ["wrist", 1], e: ["wrist", -1], o: ["grip", 1], c: ["grip", -1],
  t: ["tilt", 1], g: ["tilt", -1],
};
addEventListener("keydown", (e) => {
  const k = e.key.toLowerCase();
  if (k in KEYMAP && !e.metaKey && !e.ctrlKey && !dialog.open) { keys.add(k); e.preventDefault(); }
});
addEventListener("keyup", (e) => keys.delete(e.key.toLowerCase()));
addEventListener("blur", () => keys.clear());

function currentInput() {
  const out = { ...touch };
  for (const k of keys) out[KEYMAP[k][0]] += KEYMAP[k][1];
  for (const axis in out) out[axis] = Math.max(-1, Math.min(1, out[axis]));
  const len = Math.hypot(out.x, out.y);
  if (len > 1) { out.x /= len; out.y /= len; }
  return out;
}

// Sent continuously while in control: it's also how the robot knows this page is still here.
setInterval(() => {
  if (state.you) send({ type: "input", ...currentInput() });
}, 1000 / SEND_HZ);

// Tick the thinking timer between updates from the robot.
setInterval(() => state.autopilot && state.autopilot.phase === "thinking" && render(), 100);

connectSignal();
