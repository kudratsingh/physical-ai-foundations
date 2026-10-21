#!/usr/bin/env python3
"""Open a local interactive forward/inverse-kinematics explorer in a browser.

From this checkout's root: .venv/bin/python math/arm_viewer.py
Ctrl+C stops the server. This uses the Mac venv, not the ROS container.
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
import webbrowser

from planar_arm import ik_payload, pose_payload
from resolved_rate import control_payload
from arm_dynamics import physics_payload

HTML_PATH = Path(__file__).with_suffix(".html")
ASSETS = {f"/{name}": HTML_PATH.parent / name for name in (
    "arm_viewer.js", "arm_motion.mjs", "arm_scene.js", "arm_viewer.css")}


def query_number(values, key, default, minimum=None, maximum=None):
    raw = values.get(key, [str(default)])
    if len(raw) != 1:
        raise ValueError(f"{key} must appear once")
    value = float(raw[0])
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    if minimum is not None and not minimum <= value <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value


class ArmHandler(BaseHTTPRequestHandler):
    def send_content(self, status, content, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            # A newer slider position may cancel a request already being served.
            pass

    def do_GET(self):
        url = urlsplit(self.path)
        if url.path == "/":
            self.send_content(200, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
        elif url.path in ASSETS:
            content_type = "text/css" if url.path.endswith(".css") else "text/javascript"
            self.send_content(200, ASSETS[url.path].read_bytes(), content_type + "; charset=utf-8")
        elif url.path in ("/api/pose", "/api/ik", "/api/control"):
            try:
                values = parse_qs(url.query, keep_blank_values=True)
                l1 = query_number(values, "l1", 1, 0.25, 1.5)
                l2 = query_number(values, "l2", 0.75, 0.25, 1.5)
                if url.path == "/api/control":
                    if len(values.get("method", ["dls"])) != 1:
                        raise ValueError("method must appear once")
                    payload = control_payload(
                        query_number(values, "q1", 30, -180, 180),
                        query_number(values, "q2", 60, -180, 180),
                        query_number(values, "x", 1.1, -5, 5),
                        query_number(values, "y", 0.45, -5, 5), l1, l2,
                        dt=query_number(values, "dt", 0.04, 0, 0.05),
                        gain=query_number(values, "gain", 2, 0.1, 5),
                        damping=query_number(values, "damping", 0.12, 0.01, 1),
                        method=values.get("method", ["dls"])[0],
                    )
                elif url.path == "/api/ik":
                    x = query_number(values, "x", 1)
                    y = query_number(values, "y", 0.75)
                    hint = query_number(values, "shoulder_hint", 0, -180, 180)
                    payload = ik_payload(x, y, l1, l2, hint)
                else:
                    q1 = query_number(values, "q1", 30, -180, 180)
                    q2 = query_number(values, "q2", 60, -180, 180)
                    payload = pose_payload(q1, q2, l1, l2)
                self.send_content(200, json.dumps(payload, allow_nan=False).encode(),
                                  "application/json")
            except ValueError as error:
                self.send_content(400, json.dumps({"error": str(error)}).encode(),
                                  "application/json")
        elif url.path == "/favicon.ico":
            self.send_content(204, b"", "image/x-icon")
        else:
            self.send_content(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self):
        if urlsplit(self.path).path != "/api/physics":
            self.send_content(404, b"Not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 8192:
                raise ValueError("Request body must contain 1 to 8192 bytes")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict) or set(body) - {"state", "target", "settings", "dt", "kick"}:
                raise ValueError("Invalid physics request fields")
            settings = body.get("settings", {})
            allowed = {"l1", "l2", "kp", "kd", "torque_limit", "gravity", "compensate", "gain", "damping", "method"}
            if not isinstance(settings, dict) or set(settings)-allowed:
                raise ValueError("Invalid physics settings")
            result = physics_payload(body.get("state", {}), body.get("target", [1.1, .45]),
                                     dt=body.get("dt", .04), kick=body.get("kick", 0), **settings)
            self.send_content(200, json.dumps(result, allow_nan=False).encode(), "application/json")
        except (ValueError, TypeError, OverflowError) as error:
            self.send_content(400, json.dumps({"error": str(error)}).encode(), "application/json")

    def log_message(self, format, *args):
        # Slider updates produce many requests. Keep the user's terminal quiet.
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765,
                        help="localhost port (default 8765; 0 chooses a free port)")
    parser.add_argument("--no-open", action="store_true", help="print URL without opening it")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), ArmHandler)
    except OSError as error:
        parser.exit(1, f"Could not start viewer: {error}. Try --port 0.\n")
    with server:
        url = f"http://127.0.0.1:{server.server_port}"
        print(f"Two-link arm: {url}\nChoose joint angles (FK) or a hand target (IK); Ctrl+C here stops the viewer.",
              flush=True)
        if not args.no_open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nArm viewer stopped.")


if __name__ == "__main__":
    main()
