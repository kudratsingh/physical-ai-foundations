import {
  drawArm,
  drawControlHistory,
  drawDynamicsHistory,
} from "/arm_scene.js";
import { targetFromPlanPoint, jointVelocity } from "/arm_motion.mjs";
const $ = (id) => document.getElementById(id),
  fmt = (v, n = 2) => (Number.isFinite(v) ? v.toFixed(n) : "—"),
  pair = (v) => (v ? `(${v.map((x) => fmt(x, 3)).join(", ")})` : "—");
let source = "live",
  snapshot = null,
  samples = [],
  history = [],
  key = null,
  origin = 0,
  pollVersion = 0;
const camera = { yaw: -Math.PI / 4, pitch: Math.PI * 0.29, zoom: 1.2 };
const options = {
  view: "plan",
  workspace: true,
  frames: false,
  angles: false,
  referenceLabel: false,
  trail: true,
  alternate: false,
};
function redraw() {
  if (!snapshot?.pose) return;
  const p = snapshot.pose,
    c = snapshot.controller;
  const target = [Number(c.target_x), Number(c.target_y)];
  const goal = target.every(Number.isFinite)
    ? { target, reachable: true, solutions: [] }
    : null;
  options.referencePose = snapshot.reference_pose;
  const velocity = jointVelocity(
    p.differential.jacobian,
    snapshot.state.velocity,
  );
  options.velocity = velocity.speed > 1e-4 ? velocity : null;
  drawArm($("plot"), p, options, samples, camera, goal);
  drawControlHistory($("error-chart"), history);
  drawDynamicsHistory($("torque-chart"), history, "torque", 20);
}
function selectSource(next) {
  source = next;
  pollVersion++;
  snapshot = null;
  samples = [];
  history = [];
  key = null;
  $("live").setAttribute("aria-pressed", String(next === "live"));
  $("replay").setAttribute("aria-pressed", String(next === "replay"));
  $("live-controls").disabled = next !== "live";
  $("latency-label").textContent =
    next === "live"
      ? "State → command arrival"
      : "Recorded simulator command latency";
  $("source-help").textContent =
    next === "live"
      ? "Browser commands go through ROS topics and services."
      : "Read-only recorded observations. Original stamps are historical; receive freshness shows replay activity.";
  $("message").textContent =
    next === "replay"
      ? "Waiting for /replay/arm/state. Play a bag to inspect it."
      : "Connecting to live ROS…";
}
async function action(name, body = {}) {
  try {
    const response = await fetch(`/api/ros/${name}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const result = await response.json();
    $("message").textContent = result.message;
    return response.ok && result.success;
  } catch (error) {
    $("message").textContent =
      "Dashboard unavailable. Check the arm container.";
    return false;
  }
}
async function poll() {
  const version = pollVersion;
  try {
    const response = await fetch(`/api/ros/status?source=${source}`, {
      cache: "no-store",
    });
    if (!response.ok) throw new Error("unavailable");
    const s = await response.json();
    if (version !== pollVersion) return;
    snapshot = s;
    $("rate").textContent = fmt(s.state_rate_hz, 1) + " Hz";
    $("freshness").textContent = fmt(s.state_receive_age_ms, 0) + " ms";
    $("latency").textContent = fmt(s.loop_latency_ms, 1) + " ms";
    $("command-delay").textContent = fmt(s.command_transport_ms, 1) + " ms";
    $("original-age").textContent = fmt(s.state_age_ms, 0) + " ms";
    $("watchdog").textContent =
      s.simulator.watchdog === "True"
        ? "Zero torque"
        : s.simulator.watchdog === "False"
          ? "Commands fresh"
          : "—";
    $("watchdog").style.color =
      s.simulator.watchdog === "True" ? "var(--red)" : "var(--green)";
    $("stale-count").textContent = s.simulator.stale_dropped ?? "—";
    $("nodes").replaceChildren();
    for (const [name, fresh] of [
      [
        "arm_simulator",
        s.state_receive_age_ms !== null && s.state_receive_age_ms < 300,
      ],
      [
        "arm_controller",
        s.controller_receive_age_ms !== null &&
          s.controller_receive_age_ms < 500,
      ],
      ["arm_dashboard", true],
    ]) {
      const row = document.createElement("div");
      row.className = "node-line";
      const label = document.createElement("span");
      label.textContent = "/" + name;
      const status = document.createElement("strong");
      status.textContent =
        source === "replay"
          ? "Recorded stream"
          : s.nodes.includes(name)
            ? fresh
              ? "Connected"
              : "Messages stale"
            : "Not discovered";
      status.style.color = fresh ? "var(--green)" : "var(--red)";
      row.append(label, status);
      $("nodes").append(row);
    }
    $("bag-status").textContent = s.bags.recording
      ? "Recording live ROS topics…"
      : s.bags.replaying
        ? "Replaying into /replay/arm/*"
        : s.bags.bag
          ? "Recording finalized; ready to replay."
          : "No recording yet.";
    $("bag-path").textContent = s.bags.bag ?? "";
    $("record").disabled = s.bags.recording;
    $("stop-record").disabled = !s.bags.recording;
    $("stop-replay").disabled = !s.bags.replaying;
    $("play-bag").disabled = s.bags.recording || !s.bags.bag;
    if (!s.pose) {
      $("pose-result").textContent = "Waiting for measured state…";
      return;
    }
    if (
      /^(Connecting|Waiting|Dashboard disconnected)/.test(
        $("message").textContent,
      )
    )
      $("message").textContent =
        source === "live"
          ? "Live ROS connected. Select a target or try a disturbance."
          : "Recorded observations · live control is isolated.";
    const target = [
      Number(s.controller.target_x),
      Number(s.controller.target_y),
    ];
    const error = target.every(Number.isFinite)
      ? Math.hypot(...s.pose.positions[2].map((v, i) => v - target[i]))
      : null;
    const nextKey = `${source}/${s.state.epoch}`;
    if (
      key !== nextKey ||
      (history.length &&
        (s.state.sim_time < origin ||
          s.state.sequence < history.at(-1).sequence))
    ) {
      samples = [];
      history = [];
      key = nextKey;
      origin = s.state.sim_time;
    }
    if (history.at(-1)?.sequence !== s.state.sequence) {
      samples.push({
        position: s.pose.positions[2],
        angles: s.pose.angles_deg,
        time: s.state.sim_time - origin,
      });
      history.push({
        time: s.state.sim_time - origin,
        error: error ?? 0,
        sequence: s.state.sequence,
        dynamics: {
          applied_torque: s.state.effort,
          saturated: s.state.effort.some((v) => Math.abs(v) >= 19.999),
        },
      });
      if (samples.length > 600) samples.shift();
      if (history.length > 600) history.shift();
    }
    $("pose-result").textContent = pair(s.pose.positions[2]) + " m";
    $("error").textContent = fmt(error, 4) + " m";
    $("torque").textContent = pair(s.state.effort);
    $("sim-time").textContent =
      fmt(s.state.sim_time, 2) + " s / " + s.state.epoch;
    $("pause").textContent = s.state.paused
      ? "Resume physics"
      : "Pause physics";
    redraw();
  } catch (error) {
    $("message").textContent =
      "Dashboard disconnected. Check Docker and the ROS processes.";
  } finally {
    setTimeout(poll, 100);
  }
}
$("live").addEventListener("click", () => selectSource("live"));
$("replay").addEventListener("click", () => selectSource("replay"));
$("send-target").addEventListener("click", () => {
  if (!$("target-x").reportValidity() || !$("target-y").reportValidity())
    return;
  action("target", {
    target: [Number($("target-x").value), Number($("target-y").value)],
  });
});
$("other-target").addEventListener("click", () => {
  $("target-x").value = 1.1;
  $("target-y").value = 0.8;
  $("send-target").click();
});
$("reset").addEventListener("click", () => action("reset"));
$("kick").addEventListener("click", () => action("kick"));
$("pause").addEventListener("click", () =>
  action("pause", { paused: !snapshot?.state?.paused }),
);
$("delay").addEventListener("input", () => {
  $("delay-value").textContent = $("delay").value + " ms";
});
$("set-delay").addEventListener("click", () =>
  action("delay", { delay_ms: Number($("delay").value) }),
);
for (const [id, name] of [
  ["record", "record"],
  ["stop-record", "stop_record"],
  ["stop-replay", "stop_replay"],
])
  $(id).addEventListener("click", () => action(name));
$("play-bag").addEventListener("click", async () => {
  if (await action("replay")) selectSource("replay");
});
for (const [id, view] of [
  ["plan", "plan"],
  ["three", "3d"],
])
  $(id).addEventListener("click", () => {
    options.view = view;
    $("plan").setAttribute("aria-pressed", String(view === "plan"));
    $("three").setAttribute("aria-pressed", String(view === "3d"));
    redraw();
  });
let orbit = null;
$("plot").addEventListener("pointerdown", (e) => {
  if (e.button !== 0 || !snapshot?.pose) return;
  if (options.view === "3d") {
    orbit = [e.clientX, e.clientY, camera.yaw, camera.pitch];
    $("plot").setPointerCapture(e.pointerId);
    return;
  }
  if (source !== "live") return;
  const rect = $("plot").getBoundingClientRect();
  const target = targetFromPlanPoint(
    e.clientX - rect.left,
    e.clientY - rect.top,
    $("plot").clientWidth,
    $("plot").clientHeight,
    snapshot.pose.workspace_radii[1],
  );
  $("target-x").value = target[0];
  $("target-y").value = target[1];
  action("target", { target });
});
$("plot").addEventListener("pointermove", (e) => {
  if (!orbit) return;
  camera.yaw = orbit[2] - (e.clientX - orbit[0]) * 0.009;
  camera.pitch = Math.max(
    0.3,
    Math.min(1.4, orbit[3] + (e.clientY - orbit[1]) * 0.008),
  );
  redraw();
});
for (const event of ["pointerup", "pointercancel", "lostpointercapture"])
  $("plot").addEventListener(event, () => {
    orbit = null;
  });
new ResizeObserver(redraw).observe($("plot"));
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
poll();
