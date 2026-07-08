"""Minimal performance dashboard HTTP service.

Several deployment assets (systemd units, docker dashboard image, and helper
scripts) expect `utils.performance_dashboard` to exist.

This module provides a small Flask app with two endpoints used by the repo's
smoke tests:
- `/api/status`
- `/api/hardware_info`

It is intentionally lightweight and avoids optional heavy dependencies.
"""

from __future__ import annotations

import argparse
import datetime as _dt
from dataclasses import dataclass
from typing import Any, Dict, Optional

from flask import Flask, jsonify

from common.hardware_detection import detect_hardware


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> Any:
        return jsonify({"service": "ipfs-accelerate-performance-dashboard", "ok": True})

    @app.get("/api/status")
    def status() -> Any:
        return jsonify(
            {
                "ok": True,
                "status": "running",
                "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            }
        )

    @app.get("/api/hardware_info")
    def hardware_info() -> Any:
        return jsonify(detect_hardware())

    return app


@dataclass
class PerformanceDashboard:
    """Simple Flask-based dashboard wrapper."""

    host: str = "0.0.0.0"
    port: int = 8080

    def start_dashboard(self) -> None:
        app = create_app()
        # Disable reloader so it stays stable under systemd.
        app.run(host=self.host, port=self.port, debug=False, use_reloader=False)


def start_performance_dashboard(host: str = "0.0.0.0", port: int = 8080, background: bool = False) -> Optional[Any]:
    """Start the dashboard.

    If `background=True`, runs the server in a daemon thread and returns it.
    Otherwise, blocks and returns None.
    """

    if not background:
        PerformanceDashboard(host=host, port=port).start_dashboard()
        return None

    import threading

    thread = threading.Thread(
        target=PerformanceDashboard(host=host, port=port).start_dashboard,
        name="performance-dashboard",
        daemon=True,
    )
    thread.start()
    return thread


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IPFS Accelerate Performance Dashboard")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--cli", action="store_true", help="Print a one-shot status and exit")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    if args.cli:
        info: Dict[str, Any] = {
            "ok": True,
            "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            "hardware": detect_hardware(),
        }
        # Avoid pretty-print dependencies; keep output JSON-ish.
        import json

        print(json.dumps(info, indent=2))
        return 0

    PerformanceDashboard(host=args.host, port=args.port).start_dashboard()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
