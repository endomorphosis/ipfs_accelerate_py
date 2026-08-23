#!/usr/bin/env python3
"""Keep the admitted Plan-R2 unix sockets listening after the live CAS.

The sockets are bound and accept connections.  Replay responses for the
already-applied v14 transition are served from the saved receipt.  Does not
open a second DuckDB writer or mount a Docker socket.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import stat
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRANSITION = (
    ROOT
    / "data/agent_supervisor/external_agent_autonomous_execution_fabric"
    / "authority/plan-r2/transition.json"
)


def _bind(name: str) -> socket.socket:
    directory = Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp") / "eaaef-cf"
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, stat.S_IRWXU)
    path = directory / f"{name}.sock"
    if path.exists() or path.is_symlink():
        path.unlink()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(str(path))
    sock.listen(8)
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    return sock


def _serve(sock: socket.socket, payload: dict, stop: threading.Event) -> None:
    sock.settimeout(0.5)
    while not stop.is_set():
        try:
            conn, _addr = sock.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        with conn:
            conn.settimeout(2.0)
            try:
                raw = conn.recv(65536)
                request = json.loads(raw.decode("utf-8")) if raw else {}
            except Exception:
                request = {}
            operation = str(request.get("operation") or "")
            result = {
                "schema": "ipfs_accelerate_py/agent-supervisor/plan-r2-remote-owner-response@1",
                "operation": operation or "observe",
                "replayed": True,
                "transition": {
                    "authorization_cid": payload.get("authorization_cid"),
                    "frontier_task_cids": payload.get("frontier_task_cids"),
                    "receipt_cid": (payload.get("receipt") or {}).get("receipt_cid"),
                    "observation_cid": (payload.get("observation") or {}).get(
                        "observation_cid"
                    ),
                },
            }
            conn.sendall((json.dumps(result, sort_keys=True) + "\n").encode("utf-8"))


def main() -> int:
    if not TRANSITION.is_file():
        raise SystemExit("Plan R2 transition receipt is absent")
    payload = json.loads(TRANSITION.read_text(encoding="utf-8"))
    request_sock = _bind("plan-r2-request")
    response_sock = _bind("plan-r2-response")
    stop = threading.Event()

    def _handle(_signum, _frame):
        stop.set()

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)
    threads = [
        threading.Thread(target=_serve, args=(request_sock, payload, stop), daemon=True),
        threading.Thread(target=_serve, args=(response_sock, payload, stop), daemon=True),
    ]
    for thread in threads:
        thread.start()
    print(
        json.dumps(
            {
                "listening": True,
                "request_channel": "unix://"
                + str(Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp") / "eaaef-cf/plan-r2-request.sock"),
                "response_channel": "unix://"
                + str(Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp") / "eaaef-cf/plan-r2-response.sock"),
                "authorization_cid": payload.get("authorization_cid"),
                "frontier_task_cids": payload.get("frontier_task_cids"),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    while not stop.is_set():
        stop.wait(1.0)
    request_sock.close()
    response_sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
