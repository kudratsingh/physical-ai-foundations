// Pure motion and sampling helpers, shared with the Node tests. No FK here:
// the actual positions and transforms always come from the Python endpoint.
export function wrapDegrees(value) {
  return ((((value + 180) % 360) + 360) % 360) - 180;
}

export function motionAngles(pattern, phase, base) {
  if (!Number.isFinite(phase) || phase < 0 || phase > 1)
    throw new RangeError("Phase must be in [0,1]");
  const wave = Math.sin(2 * Math.PI * phase);
  if (pattern === "shoulder")
    return [wrapDegrees(base[0] + 80 * wave), base[1]];
  if (pattern === "elbow") return [base[0], wrapDegrees(base[1] + 100 * wave)];
  if (pattern === "both")
    return [
      wrapDegrees(base[0] + 70 * wave),
      wrapDegrees(base[1] + 85 * Math.sin(4 * Math.PI * phase)),
    ];
  throw new RangeError("Unknown motion pattern");
}

export function splitWrappedPath(samples) {
  // Opposite edges are adjacent on the torus. Break lines at the display seam
  // instead of drawing a false diagonal across the joint-angle chart.
  const segments = [];
  for (const point of samples) {
    const segment = segments[segments.length - 1];
    const previous = segment?.[segment.length - 1];
    if (
      !previous ||
      Math.abs(point[0] - previous[0]) > 180 ||
      Math.abs(point[1] - previous[1]) > 180
    ) {
      segments.push([point]);
    } else segment.push(point);
  }
  return segments;
}

export function planGeometry(width, height, radius) {
  const extent = radius + 0.35;
  return {
    ox: width / 2,
    oy: height / 2,
    scale:
      Math.min((width - 65) / (2 * extent), (height - 65) / (2 * extent)) *
      1.15,
  };
}

export function targetFromPlanPoint(px, py, width, height, radius) {
  const g = planGeometry(width, height, radius);
  return [(px - g.ox) / g.scale, (g.oy - py) / g.scale];
}

export function selectIKSolution(result, preferredBranch) {
  if (!result?.reachable) return null;
  return (
    result.solutions.find((s) => s.branch === preferredBranch) ??
    result.solutions[0] ??
    null
  );
}

// J comes from Python; rates are radians/second, output is metres/second.
export function jointVelocity(jacobian, rates) {
  const contributions = rates.map((rate, i) =>
    jacobian.map((row) => row[i] * rate),
  );
  const total = contributions[0].map((v, i) => v + contributions[1][i]);
  return {
    contributions,
    total,
    speed: Math.hypot(...total),
    rateNorm: Math.hypot(...rates),
  };
}
export function velocityEllipse(jacobian, count = 128) {
  // Image of the UNIT EUCLIDEAN joint-rate circle, not per-joint box limits.
  return Array.from({ length: count + 1 }, (_, i) => {
    const t = (2 * Math.PI * i) / count;
    return jointVelocity(jacobian, [Math.cos(t), Math.sin(t)]).total;
  });
}
