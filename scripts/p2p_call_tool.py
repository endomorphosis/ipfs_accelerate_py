#!/usr/bin/env python3
"""Call an MCP tool over the TaskQueue libp2p transport.

This is a thin convenience wrapper around `ipfs_accelerate_py.p2p_tasks.client.call_tool_sync`.

Typical usage (two-box):
  - On box A (server), ensure the service writes an announce file (systemd already does):
      /var/cache/ipfs-accelerate/task_p2p_announce.json

  - On box B (client), call a tool on box A:
      ./scripts/p2p_call_tool.py \
        --announce-file /var/cache/ipfs-accelerate/task_p2p_announce.json \
        --tool get_server_status

Or pass a multiaddr directly:
      ./scripts/p2p_call_tool.py \
        --multiaddr '/ip4/10.0.0.12/tcp/9100/p2p/Qm...'
        --tool get_server_status

Args can be provided as JSON:
      ./scripts/p2p_call_tool.py --announce-file ... --tool workflow_list --args '{"limit": 5}'
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Any, Dict


def _load_announce(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.loads(handle.read())
    return data if isinstance(data, dict) else {}


def _parse_args_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--args must be valid JSON: {exc}")
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SystemExit("--args must be a JSON object (e.g. '{\"limit\": 5}')")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Call an MCP tool over TaskQueue P2P")
    parser.add_argument("--multiaddr", default="", help="Remote multiaddr (/ip4/.../tcp/.../p2p/...)")
    parser.add_argument(
        "--announce-file",
        default="",
        help="Path to a JSON announce file containing {peer_id, multiaddr}",
    )
    parser.add_argument("--peer-id", default="", help="Optional peer id hint")
    parser.add_argument("--tool", required=True, help="Tool name (e.g. get_server_status)")
    parser.add_argument("--args", default="{}", help="Tool args as JSON object")
    parser.add_argument("--timeout", type=float, default=30.0, help="Timeout in seconds")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")

    args = parser.parse_args(argv)

    if importlib.util.find_spec("libp2p") is None:
        print(
            "error: optional dependency 'libp2p' is not installed in this environment\n"
            "Install with: pip install -e '.[libp2p]'",
            file=sys.stderr,
        )
        return 2

    multiaddr = str(args.multiaddr or "").strip()
    peer_id = str(args.peer_id or "").strip()

    announce_file = str(args.announce_file or "").strip()
    if announce_file and not multiaddr:
        info = _load_announce(announce_file)
        multiaddr = str(info.get("multiaddr") or "").strip()
        if not peer_id:
            peer_id = str(info.get("peer_id") or "").strip()

    if not multiaddr:
        env_announce = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE") or os.environ.get(
            "IPFS_DATASETS_PY_TASK_P2P_ANNOUNCE_FILE"
        )
        if env_announce:
            info = _load_announce(env_announce)
            multiaddr = str(info.get("multiaddr") or "").strip()
            if not peer_id:
                peer_id = str(info.get("peer_id") or "").strip()

    if not multiaddr:
        print("error: missing remote multiaddr (use --multiaddr or --announce-file)", file=sys.stderr)
        return 2

    tool_args = _parse_args_json(str(args.args or ""))

    try:
        from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue, call_tool_sync

        remote = RemoteQueue(peer_id=peer_id, multiaddr=multiaddr)
        result = call_tool_sync(remote=remote, tool_name=str(args.tool), args=tool_args, timeout_s=float(args.timeout))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.pretty:
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
