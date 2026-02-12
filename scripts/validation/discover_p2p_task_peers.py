#!/usr/bin/env python3
"""Discover libp2p TaskQueue peers.

This script is a quick sanity check that p2p_tasks discovery is working.
It sends a lightweight `op=status` request using the same discovery cascade
as the normal client:
- announce file (local)
- bootstrap peers (direct)
- rendezvous
- mDNS (LAN)
- DHT provider discovery (internet-wide)

Usage:
  python scripts/validation/discover_p2p_task_peers.py

Environment (optional):
  IPFS_ACCELERATE_PY_TASK_P2P_TOKEN
  IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS_NS
  IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS
"""

from __future__ import annotations

import json
import os
import time


def main() -> int:
    # Avoid importing the heavy core (and optional ipfs_kit integration) when
    # running this lightweight p2p_tasks validation script.
    os.environ.setdefault("IPFS_ACCEL_SKIP_CORE", "1")

    try:
        from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue, get_capabilities_sync
    except Exception as exc:
        print(f"ERROR: failed to import p2p_tasks client: {exc}")
        return 2

    started = time.time()
    try:
        caps = get_capabilities_sync(remote=RemoteQueue(), detail=False)
    except Exception as exc:
        print(f"ERROR: status request failed: {exc}")
        return 1

    elapsed_ms = int((time.time() - started) * 1000)
    print(f"Elapsed: {elapsed_ms}ms")
    print(json.dumps({"ok": True, "capabilities": caps}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
