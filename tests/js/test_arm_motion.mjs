import test from "node:test";
import assert from "node:assert/strict";
import {
  motionAngles,
  wrapDegrees,
  splitWrappedPath,
  planGeometry,
  targetFromPlanPoint,
  selectIKSolution,
  jointVelocity,
  velocityEllipse,
} from "../../math/arm_motion.mjs";

const close = (a, b) => Math.abs(wrapDegrees(a - b)) < 1e-9;

test("each motion starts and ends at the same physical configuration", () => {
  for (const pattern of ["shoulder", "elbow", "both"]) {
    for (const phase of [0, 1]) {
      const pose = motionAngles(pattern, phase, [170, -150]);
      assert.ok(close(pose[0], 170) && close(pose[1], -150));
    }
  }
});

test("single-joint sweeps hold the other joint fixed", () => {
  assert.deepEqual(motionAngles("shoulder", 0.25, [10, 45]), [90, 45]);
  assert.deepEqual(motionAngles("elbow", 0.25, [10, 45]), [10, 145]);
});

test("angle wrap preserves orientation at and beyond the display boundary", () => {
  assert.equal(wrapDegrees(181), -179);
  assert.equal(wrapDegrees(-181), 179);
  assert.ok(close(wrapDegrees(540), 180));
  for (let i = 0; i <= 100; i++) {
    for (const q of motionAngles("both", i / 100, [170, 170]))
      assert.ok(q >= -180 && q <= 180);
  }
});

test("C-space traces break at either periodic seam without losing samples", () => {
  const points = [
    [170, 0],
    [179, 0],
    [-179, 0],
    [-170, 179],
    [-170, -179],
  ];
  const segments = splitWrappedPath(points);
  assert.deepEqual(
    segments.map((s) => s.length),
    [2, 2, 1],
  );
  assert.deepEqual(segments.flat(), points);
  assert.deepEqual(splitWrappedPath([]), []);
});

test("invalid motion requests are rejected", () => {
  for (const phase of [-0.1, 1.1, NaN])
    assert.throws(() => motionAngles("both", phase, [0, 0]), RangeError);
  assert.throws(() => motionAngles("unknown", 0, [0, 0]), RangeError);
});

test("target picking inverts plan projection across viewport sizes and link geometry", () => {
  for (const [w, h] of [
    [800, 480],
    [320, 400],
    [1000, 600],
  ]) {
    for (const radius of [0.5, 1.75, 3]) {
      const g = planGeometry(w, h, radius);
      for (const [x, y] of [
        [0, 0],
        [1, -0.75],
        [-radius, radius],
      ]) {
        const target = targetFromPlanPoint(
          g.ox + x * g.scale,
          g.oy - y * g.scale,
          w,
          h,
          radius,
        );
        assert.ok(Math.abs(target[0] - x) < 1e-12);
        assert.ok(Math.abs(target[1] - y) < 1e-12);
      }
    }
  }
});
test("IK selection keeps the requested branch, with a singular representative fallback", () => {
  const up = { branch: "elbow-up" },
    down = { branch: "elbow-down" };
  assert.equal(
    selectIKSolution({ reachable: true, solutions: [down, up] }, "elbow-up"),
    up,
  );
  assert.equal(
    selectIKSolution({ reachable: true, solutions: [down, up] }, "elbow-down"),
    down,
  );
  const singular = { branch: "singular" };
  assert.equal(
    selectIKSolution({ reachable: true, solutions: [singular] }, "elbow-up"),
    singular,
  );
});
test("unreachable targets never fabricate a joint solution", () => {
  assert.equal(
    selectIKSolution({ reachable: false, solutions: [] }, "elbow-up"),
    null,
  );
  assert.equal(selectIKSolution(null, "elbow-up"), null);
});

test("joint contributions sum with correct signs and zero rates give zero velocity", () => {
  const J = [
    [-0.75, -0.75],
    [1, 0],
  ];
  assert.deepEqual(jointVelocity(J, [1, -1]).total, [0, 1]);
  assert.ok(jointVelocity(J, [0, 0]).total.every((v) => v === 0));
  const v = jointVelocity(J, [0.6, 0.8]);
  assert.ok(Math.abs(v.rateNorm - 1) < 1e-12);
  assert.ok(Math.abs(v.total[0] + 1.05) < 1e-12);
  assert.equal(v.total[1], 0.6);
});
test("ellipse is the image of a unit-rate circle, not joint slider box limits", () => {
  const points = velocityEllipse([
    [2, 0],
    [0, 1],
  ]);
  assert.equal(points.length, 129);
  for (const [x, y] of points)
    assert.ok(Math.abs((x * x) / 4 + y * y - 1) < 1e-12);
  assert.ok(
    Math.hypot(
      points[0][0] - points.at(-1)[0],
      points[0][1] - points.at(-1)[1],
    ) < 1e-12,
  );
});
test("singular ellipse collapses to one direction including the equal-link folded case", () => {
  for (const J of [
    [
      [0, 0],
      [1.75, 0.75],
    ],
    [
      [0, 0],
      [0, -1],
    ],
  ]) {
    for (const [x] of velocityEllipse(J)) assert.ok(x === 0);
  }
});
