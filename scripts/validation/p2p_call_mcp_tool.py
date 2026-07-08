#!/usr/bin/env python3
"""Call an MCP tool over libp2p (TaskQueue op=call_tool).

This is intended as a smoke test for a remote box running the MCP server with
the TaskQueue libp2p service enabled (typically via systemd).

Examples:
  # Dial via discovery cascade (rendezvous/DHT/mDNS/announce-file):
  IPFS_ACCELERATE_PY_TASK_P2P_TOKEN=... \
    python scripts/validation/p2p_call_mcp_tool.py --tool get_server_status

  # Dial a specific peer directly:
  IPFS_ACCELERATE_PY_TASK_P2P_TOKEN=... \
    python scripts/validation/p2p_call_mcp_tool.py \
      --multiaddr /ip4/203.0.113.10/tcp/9710/p2p/12D3KooW... \
      --tool get_server_status

Notes:
  - Remote must enable tool calls: IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_TOOLS=1
  - If a shared token is set on the remote, you must set the same token here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Call an MCP tool via libp2p TaskQueue service")
    parser.add_argument(
        "--peer-id",
        default="",
        help="Optional target peer id (used for discovery filtering; ignored if --multiaddr is set)",
    )
    parser.add_argument(
        "--multiaddr",
        default="",
        help="Optional direct target multiaddr (/ip4/.../tcp/.../p2p/...)",
    )
    parser.add_argument(
        "--tool",
        required=True,
        help="MCP tool name to call on the remote (ex: get_server_status)",
    )
    parser.add_argument(
        "--args-json",
        default="{}",
        help='Tool args as JSON object (default: "{}")',
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=30.0,
        help="Timeout seconds for the remote tool execution (default: 30)",
    )
    args = parser.parse_args(argv)

    try:
        tool_args = json.loads(args.args_json)
    except Exception as exc:
        print(f"ERROR: --args-json must be valid JSON: {exc}", file=sys.stderr)
        return 2

    if tool_args is None:
        tool_args = {}
    if not isinstance(tool_args, dict):
        print("ERROR: --args-json must decode to an object/dict", file=sys.stderr)
        return 2

    try:
        from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue, call_tool_sync
    except Exception as exc:
        print(f"ERROR: failed to import p2p_tasks client: {exc}", file=sys.stderr)
        return 2

    remote = RemoteQueue(peer_id=str(args.peer_id or "").strip(), multiaddr=str(args.multiaddr or "").strip())

    started = time.time()
    try:
        resp = call_tool_sync(remote=remote, tool_name=str(args.tool), args=tool_args, timeout_s=float(args.timeout_s))
    except Exception as exc:
        print(f"ERROR: call_tool failed: {exc}", file=sys.stderr)
        return 1

    elapsed_ms = int((time.time() - started) * 1000)
    print(f"Elapsed: {elapsed_ms}ms")
    print(json.dumps(resp, indent=2, sort_keys=True))

    if not bool(resp.get("ok")):
        return 1
    # Helpful hint for the most common misconfig.
    if str(resp.get("error") or "").strip().lower() == "tools_disabled":
        print(
            "Hint: remote has tools disabled. Set IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_TOOLS=1 on the remote service.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
