#!/usr/bin/env python3
"""Local end-to-end validation for p2p_tasks.

This script starts a TaskQueue libp2p service in-process and then performs an
`op=status` request via the client.

It is intended to validate the protocol + discovery/dial path without requiring
multiple machines.

Usage:
  IPFS_ACCEL_SKIP_CORE=1 python scripts/validation/local_p2p_taskqueue_e2e.py

Notes:
- By default this disables DHT/rendezvous/mDNS/bootstraps to keep the test fast
  and deterministic; it dials the service directly using the service's announce
  file.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import time
import functools
from pathlib import Path


def _free_tcp_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    _, port = sock.getsockname()
    sock.close()
    return int(port)


def main() -> int:
    os.environ.setdefault("IPFS_ACCEL_SKIP_CORE", "1")

    # Keep the local test deterministic and avoid noisy network calls.
    os.environ.setdefault("IPFS_ACCELERATE_PY_TASK_P2P_DHT", "0")
    os.environ.setdefault("IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS", "0")
    os.environ.setdefault("IPFS_ACCELERATE_PY_TASK_P2P_MDNS", "0")
    os.environ.setdefault("IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS", "0")
    os.environ.setdefault("IPFS_ACCELERATE_PY_TASK_P2P_PUBLIC_IP", "127.0.0.1")

    try:
        import anyio

        from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue, request_status
        from ipfs_accelerate_py.p2p_tasks.service import serve_task_queue
    except Exception as exc:
        print(f"ERROR: imports failed: {exc}")
        return 2

    async def _run() -> int:
        port = _free_tcp_port()

        with tempfile.TemporaryDirectory(prefix="ipfs-accel-p2p-tasks-") as td:
            root = Path(td)
            queue_path = str(root / "task_queue.duckdb")
            announce_path = str(root / "task_p2p_announce.json")

            # Force both service and client to use the same announce file.
            os.environ["IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE"] = announce_path

            started = time.time()

            async with anyio.create_task_group() as tg:
                tg.start_soon(functools.partial(serve_task_queue, queue_path=queue_path, listen_port=port))

                # Wait for the service to publish its announce file.
                deadline = anyio.current_time() + 15.0
                multiaddr = ""
                while anyio.current_time() < deadline:
                    try:
                        if os.path.exists(announce_path):
                            info = json.loads(Path(announce_path).read_text("utf-8"))
                            if isinstance(info, dict):
                                multiaddr = str(info.get("multiaddr") or "").strip()
                                if multiaddr and "/p2p/" in multiaddr:
                                    break
                    except Exception:
                        pass
                    await anyio.sleep(0.05)

                if not multiaddr:
                    tg.cancel_scope.cancel()
                    print("ERROR: service did not write announce file in time")
                    return 1

                # Dial the service directly via its announced multiaddr.
                resp = await request_status(remote=RemoteQueue(multiaddr=multiaddr), timeout_s=5.0, detail=False)

                elapsed_ms = int((time.time() - started) * 1000)
                print(f"Elapsed: {elapsed_ms}ms")
                print(json.dumps(resp, indent=2, sort_keys=True))

                tg.cancel_scope.cancel()
                return 0 if bool(resp.get("ok")) else 1

    return anyio.run(_run, backend="trio")


if __name__ == "__main__":
    raise SystemExit(main())
