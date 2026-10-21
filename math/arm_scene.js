import {
  planGeometry,
  splitWrappedPath,
  velocityEllipse,
} from "/arm_motion.mjs";
const NS = "http://www.w3.org/2000/svg";
const el = (svg, tag, attrs = {}, content) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (content !== undefined) n.textContent = content;
  svg.append(n);
  return n;
};
const label = (svg, x, y, t, attrs = {}) =>
  el(svg, "text", { x, y, fill: "var(--muted)", "font-size": 11, ...attrs }, t);
const path = (svg, points, attrs = {}) =>
  el(svg, "path", {
    d: points.map((p, i) => `${i ? "L" : "M"}${p[0]},${p[1]}`).join(" "),
    fill: "none",
    stroke: "var(--line)",
    ...attrs,
  });
const clear = (svg) => {
  svg
    .querySelectorAll(":scope > :not(title):not(desc)")
    .forEach((n) => n.remove());
  const w = svg.clientWidth,
    h = svg.clientHeight;
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  return [w, h];
};
const hex = (name) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const shade = (color, f) => {
  const rgb = color
    .slice(1)
    .match(/../g)
    .map((x) => parseInt(x, 16));
  return `rgb(${rgb.map((v) => Math.round(Math.min(255, v * f))).join(",")})`;
};
const circlePoints = (center, r, z, count = 90) =>
  Array.from({ length: count + 1 }, (_, i) => {
    const a = (2 * Math.PI * i) / count;
    return [center[0] + r * Math.cos(a), center[1] + r * Math.sin(a), z];
  });

export function drawArm(svg, data, options, samples, camera, ik = null) {
  const [w, h] = clear(svg);
  if (!w || !h) return;
  const radius = data.workspace_radii[1],
    extent = radius + 0.35;
  const is3d = options.view === "3d";
  const plan = planGeometry(w, h, radius);
  const scale = is3d
    ? Math.min((w - 65) / (2 * extent), (h - 65) / (2 * extent)) * camera.zoom
    : plan.scale;
  const ox = w / 2,
    oy = is3d ? h * 0.54 : h / 2;
  const yaw = camera.yaw,
    pitch = camera.pitch;
  const right = [-Math.sin(yaw), Math.cos(yaw), 0],
    up = [
      -Math.cos(yaw) * Math.sin(pitch),
      -Math.sin(yaw) * Math.sin(pitch),
      Math.cos(pitch),
    ];
  const near = [
    Math.cos(yaw) * Math.cos(pitch),
    Math.sin(yaw) * Math.cos(pitch),
    Math.sin(pitch),
  ];
  const dot = (p, v) => p.reduce((s, x, i) => s + x * v[i], 0);
  const project = (p) =>
    is3d
      ? [ox + scale * dot(p, right), oy - scale * dot(p, up)]
      : [ox + scale * p[0], oy - scale * p[1]];
  const worldLine = (a, b, attrs) => path(svg, [project(a), project(b)], attrs);
  const [base, elbow, hand] = data.positions;
  const z = is3d ? 0.34 : 0;
  const [q1, q2] = data.angles_rad;
  const reachColor = hex("--reach"),
    blue = hex("--blue"),
    orange = hex("--orange"),
    ink = hex("--ink");
  if (options.workspace) {
    const outer = circlePoints(base, radius, 0),
      inner = circlePoints(base, data.workspace_radii[0], 0).reverse();
    const outline = outer.map(project);
    const hole = inner.map(project);
    let d =
      outline.map((p, i) => `${i ? "L" : "M"}${p.join(",")}`).join(" ") + " Z";
    if (data.workspace_radii[0] > 1e-10)
      d +=
        " " +
        hole.map((p, i) => `${i ? "L" : "M"}${p.join(",")}`).join(" ") +
        " Z";
    el(svg, "path", {
      d,
      fill: reachColor,
      opacity: 0.45,
      "fill-rule": "evenodd",
    });
    path(svg, outline, { "stroke-dasharray": "4 5" });
    if (data.workspace_radii[0] > 1e-10)
      path(svg, hole, { "stroke-dasharray": "3 4" });
  }
  const tick = radius > 2 ? 1 : 0.5;
  for (
    let v = -Math.floor(radius / tick) * tick;
    v <= radius + 1e-8;
    v += tick
  ) {
    worldLine([v, -radius - 0.15, 0], [v, radius + 0.15, 0], {
      stroke: v === 0 ? "var(--line)" : "var(--grid)",
    });
    worldLine([-radius - 0.15, v, 0], [radius + 0.15, v, 0], {
      stroke: v === 0 ? "var(--line)" : "var(--grid)",
    });
    if (!is3d && v !== 0 && (w >= 440 || Number.isInteger(v))) {
      const [x, y] = project([v, 0, 0]);
      label(svg, x, y + 17, String(v), { "text-anchor": "middle" });
      const [a, b] = project([0, v, 0]);
      label(svg, a - 9, b + 4, String(v), { "text-anchor": "end" });
    }
  }
  if (options.trail && samples.length > 1)
    path(
      svg,
      samples.map((s) => project([...s.position, is3d ? 0.02 : 0])),
      { stroke: "var(--green)", "stroke-width": 2, opacity: 0.7 },
    );
  if (options.referencePose) {
    const points = options.referencePose.positions.map((p) =>
      project([...p, z]),
    );
    path(svg, points, {
      stroke: "var(--muted)",
      "stroke-width": 4,
      "stroke-dasharray": "5 4",
      opacity: 0.65,
    });
    const hand = points[2];
    label(svg, hand[0] + 10, hand[1] - 18, "Joint reference", {
      "font-size": 10,
    });
  }
  if (ik?.reachable && options.alternate) {
    for (const solution of ik.solutions.filter(
      (s) => s.branch !== ik.selectedBranch,
    )) {
      const points = solution.pose.positions;
      path(
        svg,
        points.map((p) => project([...p, z])),
        {
          stroke: "var(--muted)",
          "stroke-width": 5,
          "stroke-dasharray": "6 5",
          opacity: 0.4,
        },
      );
      const [cx, cy] = project([...points[1], z]);
      el(svg, "circle", {
        cx,
        cy,
        r: 5,
        fill: "var(--paper)",
        stroke: "var(--muted)",
        opacity: 0.7,
      });
      label(svg, cx + 10, cy - 12, "Other elbow", { "font-size": 10 });
    }
  }
  if (!is3d) {
    const p = project([hand[0], hand[1], 0]);
    const px = project([hand[0], 0, 0]),
      py = project([0, hand[1], 0]);
    path(svg, [px, p, py], {
      stroke: "var(--muted)",
      "stroke-dasharray": "3 5",
      opacity: 0.45,
    });
    worldLine([...base, 0], [...elbow, 0], {
      stroke: blue,
      "stroke-width": 13,
      "stroke-linecap": "round",
    });
    worldLine([...elbow, 0], [...hand, 0], {
      stroke: orange,
      "stroke-width": 13,
      "stroke-linecap": "round",
    });
    for (const [p, r, fill] of [
      [base, 9, ink],
      [elbow, 7, "var(--paper)"],
      [hand, 7, ink],
    ]) {
      const [cx, cy] = project([...p, 0]);
      el(svg, "circle", { cx, cy, r, fill, stroke: ink, "stroke-width": 2 });
    }
  } else {
    const faces = [];
    const addFace = (points, fill) =>
      faces.push({
        points,
        fill,
        depth: points.reduce((s, p) => s + dot(p, near), 0) / points.length,
      });
    function cylinder(center, r, low, high, color, n = 20) {
      const lower = circlePoints(center, r, low, n).slice(0, -1),
        upper = circlePoints(center, r, high, n).slice(0, -1);
      for (let i = 0; i < n; i++) {
        const j = (i + 1) % n;
        addFace(
          [lower[i], lower[j], upper[j], upper[i]],
          shade(color, 0.68 + 0.2 * Math.cos((i * 2 * Math.PI) / n - 0.8)),
        );
      }
      addFace(upper, shade(color, 1.1));
    }
    function link(a, b, color, low, high) {
      const dx = b[0] - a[0],
        dy = b[1] - a[1],
        length = Math.hypot(dx, dy),
        p = [(-dy / length) * 0.065, (dx / length) * 0.065];
      const corners = [
        [a[0] + p[0], a[1] + p[1]],
        [b[0] + p[0], b[1] + p[1]],
        [b[0] - p[0], b[1] - p[1]],
        [a[0] - p[0], a[1] - p[1]],
      ];
      for (let i = 0; i < 4; i++) {
        const j = (i + 1) % 4;
        addFace(
          [
            [...corners[i], low],
            [...corners[j], low],
            [...corners[j], high],
            [...corners[i], high],
          ],
          shade(color, i % 2 ? 0.75 : 0.87),
        );
      }
      addFace(
        corners.map((p) => [...p, high]),
        shade(color, 1.05),
      );
    }
    cylinder(base, 0.2, 0, 0.16, hex("--shadow"));
    cylinder(base, 0.105, 0.16, 0.36, ink);
    link(base, elbow, blue, 0.23, 0.31);
    link(elbow, hand, orange, 0.25, 0.33);
    cylinder(elbow, 0.1, 0.2, 0.36, blue);
    cylinder(elbow, 0.056, 0.36, 0.375, ink);
    cylinder(hand, 0.075, 0.23, 0.38, ink);
    faces
      .sort((a, b) => a.depth - b.depth)
      .forEach((face) =>
        el(svg, "polygon", {
          points: face.points.map((p) => project(p).join(",")).join(" "),
          fill: face.fill,
          stroke: face.fill,
          "stroke-width": 0.5,
          "stroke-linejoin": "round",
        }),
      );
  }
  const frameSize = Math.min(0.34, radius * 0.23);
  function axis(origin, angle, name) {
    const a = [...origin, z + 0.05],
      c = Math.cos(angle),
      s = Math.sin(angle);
    const x = [origin[0] + frameSize * c, origin[1] + frameSize * s, z + 0.05],
      y = [origin[0] - frameSize * s, origin[1] + frameSize * c, z + 0.05];
    worldLine(a, x, { stroke: "var(--red)", "stroke-width": 1.5 });
    worldLine(a, y, { stroke: "var(--green)", "stroke-width": 1.5 });
    const [x1, y1] = project(x),
      [x2, y2] = project(y);
    label(svg, x1 + 5, y1 - 4, `x${name}`, { fill: "var(--red)" });
    label(svg, x2 + 5, y2 - 4, `y${name}`, { fill: "var(--green)" });
  }
  if (options.frames) {
    axis(base, 0, "₀");
    axis(elbow, q1 + q2, "ₑ");
    axis(hand, q1 + q2, "ₕ");
  }
  if (options.angles) {
    function arc(center, start, delta, r, text) {
      const points = Array.from({ length: 41 }, (_, i) => [
        center[0] + r * Math.cos(start + (delta * i) / 40),
        center[1] + r * Math.sin(start + (delta * i) / 40),
        z + 0.07,
      ]);
      if (Math.abs(delta) > 0.02)
        path(svg, points.map(project), {
          stroke: "var(--muted)",
          "stroke-width": 1.4,
        });
      worldLine(
        [...center, z + 0.07],
        [
          center[0] + r * 1.15 * Math.cos(start),
          center[1] + r * 1.15 * Math.sin(start),
          z + 0.07,
        ],
        { stroke: "var(--muted)", "stroke-dasharray": "3 3", opacity: 0.6 },
      );
      const mid = start + (Math.abs(delta) < 0.1 ? -0.65 : delta / 2),
        p = project([
          center[0] + (r + 0.11) * Math.cos(mid),
          center[1] + (r + 0.11) * Math.sin(mid),
          z + 0.1,
        ]);
      label(svg, p[0], p[1], text, {
        fill: ink,
        "text-anchor": "middle",
        "font-size": 11,
        "paint-order": "stroke",
        stroke: "var(--paper)",
        "stroke-width": 3,
        "stroke-linejoin": "round",
      });
    }
    arc(
      base,
      0,
      q1,
      Math.min(0.3, data.lengths[0] * 0.5),
      `q₁ ${data.angles_deg[0].toFixed(0)}°`,
    );
    arc(
      elbow,
      q1,
      q2,
      Math.min(0.22, data.lengths[1] * 0.5),
      `q₂ ${data.angles_deg[1].toFixed(0)}°`,
    );
  }
  const [hx, hy] = project([...hand, z + 0.1]);
  label(svg, hx + 12, hy - 17, "Hand", {
    fill: ink,
    "font-weight": 650,
    "paint-order": "stroke",
    stroke: "var(--paper)",
    "stroke-width": 3,
  });
  if (!is3d) {
    label(svg, w - 14, oy - 10, "x (m)", { "text-anchor": "end" });
    label(svg, ox + 9, 19, "y (m)");
  }
  if (ik) {
    const [rawX, rawY] = project([...ik.target, z]);
    const tx = Math.max(20, Math.min(w - 20, Number.isNaN(rawX) ? ox : rawX));
    const ty = Math.max(23, Math.min(h - 20, Number.isNaN(rawY) ? oy : rawY));
    const color = ik.reachable ? "var(--green)" : "var(--warn)";
    const offView = tx !== rawX || ty !== rawY;
    el(svg, "circle", {
      cx: tx,
      cy: ty,
      r: 12,
      fill: "none",
      stroke: color,
      "stroke-width": 1.8,
      "stroke-dasharray": "3 3",
    });
    path(
      svg,
      [
        [tx - 17, ty],
        [tx - 7, ty],
      ],
      { stroke: color, "stroke-width": 1.5 },
    );
    path(
      svg,
      [
        [tx + 7, ty],
        [tx + 17, ty],
      ],
      { stroke: color, "stroke-width": 1.5 },
    );
    path(
      svg,
      [
        [tx, ty - 17],
        [tx, ty - 7],
      ],
      { stroke: color, "stroke-width": 1.5 },
    );
    path(
      svg,
      [
        [tx, ty + 7],
        [tx, ty + 17],
      ],
      { stroke: color, "stroke-width": 1.5 },
    );
    label(
      svg,
      tx,
      Math.min(h - 8, ty + 30),
      offView ? "Target off view" : ik.reachable ? "Target" : "Unreachable",
      {
        fill: color,
        "font-weight": 600,
        "text-anchor": "middle",
        "paint-order": "stroke",
        stroke: "var(--paper)",
        "stroke-width": 3,
      },
    );
  }
  if (options.velocity) {
    const start = project([...hand, z]);
    const end = (v) => project([hand[0] + 0.2 * v[0], hand[1] + 0.2 * v[1], z]);
    options.velocity.contributions.forEach((v, i) =>
      arrow(
        svg,
        start,
        end(v),
        i === 0 ? "var(--blue)" : "var(--orange)",
        true,
      ),
    );
    arrow(svg, start, end(options.velocity.total), "var(--green)");
    if (options.requestedVelocity) {
      arrow(svg, start, end(options.requestedVelocity), "var(--red)", true);
      const desired = end(options.requestedVelocity);
      label(svg, desired[0] + 10, desired[1] + 18, "requested", {
        fill: "var(--red)",
        "font-size": 10,
      });
    }
    const tip = end(options.velocity.total);
    label(svg, tip[0] + 10, tip[1] - 10, "v · 0.20 s", {
      fill: "var(--green)",
      "paint-order": "stroke",
      stroke: "var(--paper)",
      "stroke-width": 3,
    });
  }
  svg.querySelector("desc").textContent =
    `Shoulder ${data.angles_deg[0].toFixed(1)} degrees, elbow ${data.angles_deg[1].toFixed(1)} degrees; hand (${hand.map((v) => v.toFixed(3)).join(", ")}) metres. ${is3d ? "3D schematic; drag to orbit." : "Plan view."}`;
}

export function cspaceGeometry(svg) {
  const w = svg.clientWidth,
    h = svg.clientHeight;
  return { w, h, left: 44, right: w - 18, top: 14, bottom: h - 36 };
}
export function drawCspace(svg, data, samples, ik = null) {
  const [w, h] = clear(svg);
  if (!w || !h) return;
  const g = cspaceGeometry(svg),
    map = (q) => [
      g.left + ((q[0] + 180) / 360) * (g.right - g.left),
      g.bottom - ((q[1] + 180) / 360) * (g.bottom - g.top),
    ];
  el(svg, "rect", {
    x: g.left,
    y: g.top,
    width: g.right - g.left,
    height: g.bottom - g.top,
    fill: "none",
    stroke: "var(--line)",
  });
  const ticks = w < 350 ? [-180, 0, 180] : [-180, -90, 0, 90, 180];
  ticks.forEach((v) => {
    const [x] = map([v, 0]),
      [, y] = map([0, v]);
    path(
      svg,
      [
        [x, g.top],
        [x, g.bottom],
      ],
      { stroke: "var(--grid)" },
    );
    path(
      svg,
      [
        [g.left, y],
        [g.right, y],
      ],
      { stroke: "var(--grid)" },
    );
    label(svg, x, g.bottom + 16, String(v), { "text-anchor": "middle" });
    label(svg, g.left - 7, y + 4, String(v), { "text-anchor": "end" });
  });
  for (const segment of splitWrappedPath(samples.map((s) => s.angles)))
    path(svg, segment.map(map), {
      stroke: "var(--blue)",
      "stroke-width": 1.8,
      opacity: 0.5,
    });
  if (ik?.infinite) {
    for (const q2 of [-180, 180])
      path(svg, [map([-180, q2]), map([180, q2])], {
        stroke: "var(--orange)",
        "stroke-width": 3,
        "stroke-dasharray": "5 4",
      });
  }
  if (ik?.reachable) {
    for (const s of ik.solutions.filter(
      (s) => s.branch !== ik.selectedBranch,
    )) {
      const [cx, cy] = map(s.pose.angles_deg);
      el(svg, "circle", {
        cx,
        cy,
        r: 6,
        fill: "var(--paper)",
        stroke: "var(--orange)",
        "stroke-width": 2,
      });
    }
  }
  const [x, y] = map(data.angles_deg);
  path(
    svg,
    [
      [g.left, y],
      [x, y],
      [x, g.bottom],
    ],
    { stroke: "var(--muted)", "stroke-dasharray": "3 4", opacity: 0.5 },
  );
  el(svg, "circle", {
    cx: x,
    cy: y,
    r: 5,
    fill: "var(--blue)",
    stroke: "var(--paper)",
    "stroke-width": 2,
  });
  label(svg, (g.left + g.right) / 2, h - 3, "Shoulder q₁ (°)", {
    "text-anchor": "middle",
  });
  label(svg, 12, (g.top + g.bottom) / 2, "Elbow q₂ (°)", {
    transform: `rotate(-90 12 ${(g.top + g.bottom) / 2})`,
    "text-anchor": "middle",
  });
}

export function drawHistory(svg, data, samples) {
  const [w, h] = clear(svg);
  if (!w || !h) return;
  const g = { left: 43, right: w - 18, top: 14, bottom: h - 36 };
  const end = samples.length ? Math.max(samples.at(-1).time, 15) : 15,
    start = end - 15,
    radius = data.workspace_radii[1];
  const x = (t) => g.left + ((t - start) / 15) * (g.right - g.left),
    y = (value) =>
      g.bottom - ((value + radius) / (2 * radius)) * (g.bottom - g.top);
  el(svg, "rect", {
    x: g.left,
    y: g.top,
    width: g.right - g.left,
    height: g.bottom - g.top,
    fill: "none",
    stroke: "var(--line)",
  });
  for (const value of [-radius, 0, radius]) {
    const py = y(value);
    path(
      svg,
      [
        [g.left, py],
        [g.right, py],
      ],
      { stroke: "var(--grid)" },
    );
    label(svg, g.left - 6, py + 4, value.toFixed(1), { "text-anchor": "end" });
  }
  for (let i = 0; i <= 3; i++) {
    const t = start + i * 5,
      px = x(t);
    label(svg, px, g.bottom + 16, t.toFixed(0), { "text-anchor": "middle" });
  }
  const visible = samples.filter((s) => s.time >= start && s.time <= end);
  for (const [index, color] of [
    [0, "var(--blue)"],
    [1, "var(--orange)"],
  ]) {
    const points = visible.map((s) => [x(s.time), y(s.position[index])]);
    path(svg, points, { stroke: color, "stroke-width": 1.8 });
    if (points.length) {
      const [cx, cy] = points.at(-1);
      el(svg, "circle", { cx, cy, r: 3, fill: color });
    }
  }
  label(svg, (g.left + g.right) / 2, h - 3, "Elapsed time (s)", {
    "text-anchor": "middle",
  });
  label(svg, 12, (g.top + g.bottom) / 2, "Position (m)", {
    transform: `rotate(-90 12 ${(g.top + g.bottom) / 2})`,
    "text-anchor": "middle",
  });
}

function arrow(svg, a, b, color, dashed = false) {
  const dx = b[0] - a[0],
    dy = b[1] - a[1],
    length = Math.hypot(dx, dy);
  if (length < 0.5) {
    el(svg, "circle", { cx: a[0], cy: a[1], r: 3, fill: color });
    return;
  }
  path(svg, [a, b], {
    stroke: color,
    "stroke-width": dashed ? 1.8 : 2.8,
    ...(dashed ? { "stroke-dasharray": "4 4" } : {}),
  });
  const ux = dx / length,
    uy = dy / length,
    head = Math.min(9, length * 0.4);
  path(
    svg,
    [
      [
        b[0] - head * ux + head * 0.45 * uy,
        b[1] - head * uy - head * 0.45 * ux,
      ],
      b,
      [
        b[0] - head * ux - head * 0.45 * uy,
        b[1] - head * uy + head * 0.45 * ux,
      ],
    ],
    { stroke: color, "stroke-width": dashed ? 1.8 : 2.8 },
  );
}

export function drawVelocity(svg, d, velocity, requested = null) {
  const [w, h] = clear(svg);
  if (!w || !h) return;
  // Keep a stable range for all rate choices at this pose: each slider is
  // bounded to ±1.5, while the ellipse uses a combined unit-rate norm.
  const J = d.jacobian;
  const extent =
    Math.max(
      0.5,
      velocity.speed,
      requested ? Math.hypot(...requested) : 0,
      1.5 * (Math.hypot(J[0][0], J[1][0]) + Math.hypot(J[0][1], J[1][1])),
    ) * 1.12;
  const scale = Math.min(w - 70, h - 62) / (2 * extent);
  const map = (v) => [w / 2 + scale * v[0], h / 2 - scale * v[1]];
  const origin = map([0, 0]);
  const tick = extent > 2 ? 1 : 0.5;
  for (
    let v = -Math.floor(extent / tick) * tick;
    v <= extent + 1e-8;
    v += tick
  ) {
    path(svg, [map([v, -extent]), map([v, extent])], {
      stroke: v === 0 ? "var(--muted)" : "var(--grid)",
    });
    path(svg, [map([-extent, v]), map([extent, v])], {
      stroke: v === 0 ? "var(--muted)" : "var(--grid)",
    });
    if (v !== 0 && (w > 330 || Number.isInteger(v))) {
      const [x, y] = map([v, 0]);
      label(svg, x, y + 16, String(v), { "text-anchor": "middle" });
      const [a, b] = map([0, v]);
      label(svg, a - 8, b + 4, String(v), { "text-anchor": "end" });
    }
  }
  path(svg, velocityEllipse(J).map(map), {
    stroke: "var(--blue)",
    "stroke-width": d.singular ? 4 : 2,
    fill: d.singular ? "none" : "var(--reach)",
    "fill-opacity": 0.4,
  });
  d.principal_directions.forEach((axis, i) => {
    const v = axis.map((x) => x * d.singular_values[i]);
    path(svg, [map(v.map((x) => -x)), map(v)], {
      stroke: "var(--muted)",
      "stroke-dasharray": "3 4",
      opacity: 0.7,
    });
  });
  velocity.contributions.forEach((v, i) => {
    arrow(svg, origin, map(v), i === 0 ? "var(--blue)" : "var(--orange)", true);
    path(svg, [map(v), map(velocity.total)], {
      stroke: i === 0 ? "var(--orange)" : "var(--blue)",
      "stroke-dasharray": "3 4",
      opacity: 0.4,
    });
  });
  if (requested) {
    arrow(svg, origin, map(requested), "var(--red)", true);
    const r = map(requested);
    label(svg, r[0] + 8, r[1] - 10, "requested", {
      fill: "var(--red)",
      "font-size": 10,
    });
  }
  arrow(svg, origin, map(velocity.total), "var(--green)");
  const tip = map(velocity.total);
  el(svg, "circle", { cx: tip[0], cy: tip[1], r: 4, fill: "var(--green)" });
  label(svg, w - 10, h / 2 - 10, "vₓ (m/s)", { "text-anchor": "end" });
  label(svg, w / 2 + 9, 16, "vᵧ (m/s)");
  svg.querySelector("desc").textContent =
    `Selected hand velocity (${velocity.total.map((v) => v.toFixed(3)).join(", ")}) metres per second. ${d.singular ? "Singular: the velocity ellipse collapses to a line." : "Ellipse maps the unit joint-rate circle through the Jacobian."}`;
}

export function drawControlHistory(svg, samples) {
  const [w, h] = clear(svg);
  if (!w || !h) return;
  const left = 45,
    right = w - 16,
    top = 18,
    bottom = h - 35;
  const duration = Math.max(1, samples.at(-1)?.time ?? 0);
  const ymax = Math.max(0.05, ...samples.map((s) => s.error)) * 1.1;
  const point = (s) => [
    left + ((right - left) * s.time) / duration,
    bottom - ((bottom - top) * s.error) / ymax,
  ];
  for (let i = 0; i <= 4; i++) {
    const y = top + ((bottom - top) * i) / 4;
    path(
      svg,
      [
        [left, y],
        [right, y],
      ],
      { stroke: "var(--grid)" },
    );
    label(svg, left - 6, y + 4, (ymax * (1 - i / 4)).toFixed(2), {
      "text-anchor": "end",
    });
    const x = left + ((right - left) * i) / 4;
    label(svg, x, bottom + 17, ((duration * i) / 4).toFixed(1), {
      "text-anchor": "middle",
    });
  }
  path(
    svg,
    [
      [left, top],
      [left, bottom],
      [right, bottom],
    ],
    { stroke: "var(--muted)" },
  );
  const toleranceY = point({ time: 0, error: 0.01 })[1];
  path(
    svg,
    [
      [left, toleranceY],
      [right, toleranceY],
    ],
    { stroke: "var(--green)", "stroke-dasharray": "4 4" },
  );
  label(svg, right - 4, toleranceY - 5, "0.01 m", {
    fill: "var(--green)",
    "text-anchor": "end",
  });
  path(svg, samples.map(point), { stroke: "var(--blue)", "stroke-width": 2.2 });
  if (samples.length) {
    const [cx, cy] = point(samples.at(-1));
    el(svg, "circle", { cx, cy, r: 4, fill: "var(--blue)" });
  }
  label(svg, (left + right) / 2, h - 3, "Simulated time (s)", {
    "text-anchor": "middle",
  });
  svg.querySelector("desc").textContent = samples.length
    ? `Position error ${samples.at(-1).error.toFixed(4)} metres after ${samples.at(-1).time.toFixed(2)} simulated seconds.`
    : "Run target tracking to record the position error.";
}

export function drawDynamicsHistory(svg, samples, kind, limit = 20) {
  const [w, h] = clear(svg);
  if (!w || !h) return;
  const data = samples.filter((s) => s.dynamics);
  const get = (s) =>
    kind === "torque" ? s.dynamics.applied_torque : s.dynamics.joint_error;
  const ymax =
    kind === "torque"
      ? limit * 1.2
      : Math.max(0.02, ...data.flatMap((s) => get(s).map(Math.abs))) * 1.2;
  const left = 47,
    right = w - 14,
    top = 18,
    bottom = h - 34,
    duration = Math.max(1, data.at(-1)?.time ?? 0);
  const map = (t, v) => [
    left + ((right - left) * t) / duration,
    (top + bottom) / 2 - ((v / ymax) * (bottom - top)) / 2,
  ];
  for (let i = -2; i <= 2; i++) {
    const v = (ymax * i) / 2,
      [, y] = map(0, v);
    path(
      svg,
      [
        [left, y],
        [right, y],
      ],
      { stroke: i === 0 ? "var(--line)" : "var(--grid)" },
    );
    label(svg, left - 6, y + 4, v.toFixed(kind === "torque" ? 0 : 3), {
      "text-anchor": "end",
    });
  }
  for (let i = 0; i <= 4; i++) {
    const x = left + ((right - left) * i) / 4;
    label(svg, x, bottom + 17, ((duration * i) / 4).toFixed(1), {
      "text-anchor": "middle",
    });
  }
  if (kind === "torque")
    for (const v of [-limit, limit]) {
      const [, y] = map(0, v);
      path(
        svg,
        [
          [left, y],
          [right, y],
        ],
        { stroke: "var(--red)", "stroke-dasharray": "4 4", opacity: 0.6 },
      );
    }
  for (let joint = 0; joint < 2; joint++)
    path(
      svg,
      data.map((s) => map(s.time, get(s)[joint])),
      {
        stroke: joint === 0 ? "var(--blue)" : "var(--orange)",
        "stroke-width": 2,
      },
    );
  for (const s of data) {
    if (kind === "torque" && s.dynamics.saturated) {
      const [cx, cy] = map(s.time, get(s)[0]);
      el(svg, "circle", { cx, cy, r: 2.4, fill: "var(--red)" });
    }
    if (s.kick) {
      const [x] = map(s.time, 0);
      path(
        svg,
        [
          [x, top],
          [x, bottom],
        ],
        { stroke: "var(--green)", "stroke-dasharray": "3 3" },
      );
      label(svg, x + 4, top + 12, "Kick", { fill: "var(--green)" });
    }
  }
  label(svg, (left + right) / 2, h - 3, "Simulated time (s)", {
    "text-anchor": "middle",
  });
}
