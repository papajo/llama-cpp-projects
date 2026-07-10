#!/usr/bin/env python3
"""Inference profiling dashboard server.

Serves a real-time dashboard on ``http://localhost:8090``.
"""

from __future__ import annotations

import json
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict

from dashboard.collector import all_runs_to_dict, generate_sample_data

HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"

RUNS: Dict[str, Any] = {}


class DashboardHandler(SimpleHTTPRequestHandler):
    """HTTP handler for the profiling dashboard."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self) -> None:
        if self.path == "/api/metrics":
            self._serve_json(all_runs_to_dict(RUNS))
        elif self.path == "/api/refresh":
            RUNS.clear()
            RUNS.update(generate_sample_data())
            self._serve_json({"status": "ok", "runs": list(RUNS.keys())})
        elif self.path == "/api/summary":
            summary = {
                name: {
                    "avg_throughput": run.avg_throughput,
                    "p95_latency": run.p95_latency,
                }
                for name, run in RUNS.items()
            }
            self._serve_json(summary)
        else:
            super().do_GET()

    def _serve_json(self, data: Any) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(f"[dashboard] {args[0]}" if args else "")


def main() -> None:
    port = int(os.environ.get("DASHBOARD_PORT", "8090"))
    RUNS.update(generate_sample_data())

    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"🌐 Inference Profiling Dashboard → http://localhost:{port}")
    print(f"   API: http://localhost:{port}/api/metrics")
    print("   Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
