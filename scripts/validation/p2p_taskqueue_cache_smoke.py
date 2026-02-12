#!/usr/bin/env python3
"""Smoke test for TaskQueue p2p cache.get/cache.set.

This validates the libp2p transport on the MCP box and confirms the TaskQueue
service is accepting cache RPCs.

Examples:
  IPFS_ACCELERATE_PY_TASK_P2P_TOKEN=... \
    python scripts/validation/p2p_taskqueue_cache_smoke.py

  IPFS_ACCELERATE_PY_TASK_P2P_TOKEN=... \
    python scripts/validation/p2p_taskqueue_cache_smoke.py \
      --multiaddr /ip4/203.0.113.10/tcp/9100/p2p/12D3KooW...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TaskQueue p2p cache smoke test")
    parser.add_argument("--peer-id", default="", help="Optional target peer id")
    parser.add_argument("--multiaddr", default="", help="Optional direct target multiaddr")
    parser.add_argument("--ttl-s", type=float, default=60.0, help="TTL seconds for the cache entry")
    parser.add_argument("--key", default="", help="Optional cache key (default: random)")
    args = parser.parse_args(argv)

    key = str(args.key or "").strip()
    if not key:
        key = f"smoke:{uuid.uuid4().hex}"

    value = {
        "ts": time.time(),
        "host": os.uname().nodename if hasattr(os, "uname") else "local",
        "key": key,
    }

    try:
        from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue, cache_get_sync, cache_set_sync
    except Exception as exc:
        print(f"ERROR: failed to import p2p_tasks client: {exc}", file=sys.stderr)
        return 2

    remote = RemoteQueue(peer_id=str(args.peer_id or "").strip(), multiaddr=str(args.multiaddr or "").strip())

    started = time.time()
    try:
        set_resp = cache_set_sync(remote=remote, key=key, value=value, ttl_s=float(args.ttl_s), timeout_s=10.0)
        get_resp = cache_get_sync(remote=remote, key=key, timeout_s=10.0)
    except Exception as exc:
        print(f"ERROR: cache RPC failed: {exc}", file=sys.stderr)
        return 1

    elapsed_ms = int((time.time() - started) * 1000)
    ok = bool(set_resp.get("ok")) and bool(get_resp.get("ok")) and bool(get_resp.get("hit")) and get_resp.get("value") == value

    print(f"Elapsed: {elapsed_ms}ms")
    print(json.dumps({"set": set_resp, "get": get_resp, "ok": ok}, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
