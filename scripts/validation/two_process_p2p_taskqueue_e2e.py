"""Two-process end-to-end validation for the p2p TaskQueue.

This mimics a real multi-node setup more closely than the single-process test:
- Process A: starts the TaskQueue p2p service and writes an announce file.
- Process B: dials the service over libp2p and requests `op=status`.

The test is intentionally deterministic and local:
- Disables DHT/rendezvous/mDNS/bootstrap (announce + direct multiaddr only).

Usage:
  IPFS_ACCEL_SKIP_CORE=1 python scripts/validation/two_process_p2p_taskqueue_e2e.py
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from typing import Dict, Optional


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _read_announce(path: str) -> Dict[str, str]:
    text = open(path, "r", encoding="utf-8").read().strip()
    info = json.loads(text)
    if not isinstance(info, dict):
        raise RuntimeError(f"announce file is not a dict: {type(info)}")
    peer_id = str(info.get("peer_id") or "").strip()
    multiaddr = str(info.get("multiaddr") or "").strip()
    if not peer_id or not multiaddr:
        raise RuntimeError(f"announce missing fields: {info}")
    return {"peer_id": peer_id, "multiaddr": multiaddr}


def _start_service(*, queue_path: str, listen_port: int, announce_file: str) -> subprocess.Popen:
    env = dict(os.environ)
    env.setdefault("IPFS_ACCEL_SKIP_CORE", "1")

    # Make output readable/flush immediately.
    env["PYTHONUNBUFFERED"] = "1"

    # Deterministic + local-only.
    env["IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE"] = announce_file
    env["IPFS_ACCELERATE_PY_TASK_P2P_PUBLIC_IP"] = "127.0.0.1"
    env["IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS"] = "0"
    env["IPFS_ACCELERATE_PY_TASK_P2P_DHT"] = "0"
    env["IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS"] = "0"
    env["IPFS_ACCELERATE_PY_TASK_P2P_MDNS"] = "0"

    # Start the async service (blocks forever).
    code = (
        "import os; "
        "import anyio; "
        "import functools; "
        "from ipfs_accelerate_py.p2p_tasks.service import serve_task_queue; "
        "fn = functools.partial(serve_task_queue, queue_path=os.environ['TASK_QUEUE_PATH'], listen_port=int(os.environ['TASK_LISTEN_PORT'])); "
        "anyio.run(fn, backend='trio')"
    )

    env["TASK_QUEUE_PATH"] = queue_path
    env["TASK_LISTEN_PORT"] = str(int(listen_port))

    return subprocess.Popen(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def _wait_for_service(*, proc: subprocess.Popen, announce_file: str, timeout_s: float = 15.0) -> Dict[str, str]:
    deadline = time.time() + max(1.0, float(timeout_s))
    last_line: Optional[str] = None

    while time.time() < deadline:
        if proc.poll() is not None:
            out = ""
            try:
                out = (proc.stdout.read() if proc.stdout is not None else "")
            except Exception:
                pass
            raise RuntimeError(f"service exited early (code={proc.returncode}). Last line={last_line!r}\nOutput:\n{out}")

        # Prefer announce file existence for readiness.
        if os.path.exists(announce_file):
            try:
                return _read_announce(announce_file)
            except Exception:
                # File may be mid-write; keep waiting a bit.
                pass

        # Drain output to keep pipes clear and help debugging.
        try:
            if proc.stdout is not None:
                line = proc.stdout.readline()
                if line:
                    last_line = line.rstrip("\n")
        except Exception:
            pass

        time.sleep(0.05)

    raise RuntimeError(f"timeout waiting for announce file: {announce_file}")


def main() -> int:
    os.environ.setdefault("IPFS_ACCEL_SKIP_CORE", "1")

    with tempfile.TemporaryDirectory(prefix="taskqueue_two_proc_") as td:
        queue_path = os.path.join(td, "queue.json")
        announce_file = os.path.join(td, "task_p2p_announce.json")
        listen_port = _pick_free_port()

        proc = _start_service(queue_path=queue_path, listen_port=listen_port, announce_file=announce_file)
        try:
            info = _wait_for_service(proc=proc, announce_file=announce_file, timeout_s=20.0)
            multiaddr = info["multiaddr"]

            # Client runs in this process; still a separate process from service.
            from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue, request_status_sync

            remote = RemoteQueue(peer_id="", multiaddr=multiaddr)
            t0 = time.time()
            resp = request_status_sync(remote=remote, timeout_s=10.0, detail=False)
            elapsed_ms = int((time.time() - t0) * 1000)

            out = {
                "ok": bool(resp.get("ok")),
                "elapsed_ms": elapsed_ms,
                "remote_multiaddr": multiaddr,
                "response": resp,
            }
            print(json.dumps(out, indent=2, sort_keys=True))
            return 0 if resp.get("ok") else 2
        finally:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
