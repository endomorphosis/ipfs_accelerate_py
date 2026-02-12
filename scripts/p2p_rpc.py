#!/usr/bin/env python3
"""Interact with the TaskQueue libp2p transport (tools + cache + tasks).

This is a convenience wrapper around `ipfs_accelerate_py.p2p_tasks.client`.

Typical two-box flow:
  1) On box A, read the server's announce file (or just its printed multiaddr)
     and copy the multiaddr to box B.

  2) From box B, run:
       ./scripts/p2p_rpc.py --multiaddr '/ip4/<BOX_A_IP>/tcp/9100/p2p/<PEER_ID>' status
       ./scripts/p2p_rpc.py --multiaddr '...' call-tool --tool get_server_status
       ./scripts/p2p_rpc.py --multiaddr '...' cache-set --key demo --value '"hello"'
       ./scripts/p2p_rpc.py --multiaddr '...' cache-get --key demo
       ./scripts/p2p_rpc.py --multiaddr '...' task-submit --task-type demo --model-name demo --payload '{"x": 1}'

Note: requires optional dependency `libp2p` (install with: `pip install -e '.[libp2p]'`).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Any, Dict, Optional


def _load_announce(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.loads(handle.read())
    return data if isinstance(data, dict) else {}


def _parse_json_obj(text: str, *, flag: str) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{flag} must be valid JSON: {exc}")
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SystemExit(f"{flag} must be a JSON object")
    return value


def _parse_json_any(text: str, *, flag: str) -> Any:
    if text == "":
        return ""
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{flag} must be valid JSON: {exc}")


def _remote_from_args(args: argparse.Namespace) -> "RemoteQueue":
    multiaddr = str(getattr(args, "multiaddr", "") or "").strip()
    peer_id = str(getattr(args, "peer_id", "") or "").strip()

    announce_file = str(getattr(args, "announce_file", "") or "").strip()
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
        raise SystemExit("missing remote multiaddr (use --multiaddr or --announce-file)")

    from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue

    return RemoteQueue(peer_id=peer_id, multiaddr=multiaddr)


def _print_result(result: Any, *, pretty: bool) -> None:
    if pretty:
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="TaskQueue P2P RPC client (tools + cache + tasks)")
    parser.add_argument("--multiaddr", default="", help="Remote multiaddr (/ip4/.../tcp/.../p2p/...)")
    parser.add_argument("--announce-file", default="", help="Path to a JSON announce file containing {peer_id, multiaddr}")
    parser.add_argument("--peer-id", default="", help="Optional peer id hint")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_status = sub.add_parser("status", help="Query remote status")
    p_status.add_argument("--timeout", type=float, default=10.0)
    p_status.add_argument("--detail", action="store_true")

    p_call = sub.add_parser("call-tool", help="Call an MCP tool via P2P")
    p_call.add_argument("--tool", required=True)
    p_call.add_argument("--args", default="{}", help="Tool args as JSON object")
    p_call.add_argument("--timeout", type=float, default=30.0)

    p_cget = sub.add_parser("cache-get", help="Fetch a cache value")
    p_cget.add_argument("--key", required=True)
    p_cget.add_argument("--timeout", type=float, default=10.0)

    p_chas = sub.add_parser("cache-has", help="Check cache existence")
    p_chas.add_argument("--key", required=True)
    p_chas.add_argument("--timeout", type=float, default=10.0)

    p_cset = sub.add_parser("cache-set", help="Set a cache value")
    p_cset.add_argument("--key", required=True)
    p_cset.add_argument("--value", required=True, help="Value as JSON (e.g. '""hello""' or '{""a"":1}')")
    p_cset.add_argument("--ttl", type=float, default=None, help="Optional TTL seconds")
    p_cset.add_argument("--timeout", type=float, default=10.0)

    p_cdel = sub.add_parser("cache-delete", help="Delete a cache key")
    p_cdel.add_argument("--key", required=True)
    p_cdel.add_argument("--timeout", type=float, default=10.0)

    p_tsub = sub.add_parser("task-submit", help="Submit a task")
    p_tsub.add_argument("--task-type", required=True)
    p_tsub.add_argument("--model-name", required=True)
    p_tsub.add_argument("--payload", default="{}", help="Payload as JSON object")

    p_tlist = sub.add_parser("task-list", help="List tasks")
    p_tlist.add_argument("--status", default=None)
    p_tlist.add_argument("--limit", type=int, default=50)
    p_tlist.add_argument("--task-types", default="", help="Comma-separated list")

    p_tget = sub.add_parser("task-get", help="Get a task by id")
    p_tget.add_argument("--task-id", required=True)

    p_twait = sub.add_parser("task-wait", help="Wait for a task")
    p_twait.add_argument("--task-id", required=True)
    p_twait.add_argument("--timeout", type=float, default=60.0)

    p_tclaim = sub.add_parser("task-claim", help="Claim next task (worker)")
    p_tclaim.add_argument("--worker-id", required=True)
    p_tclaim.add_argument("--supported-task-types", default="", help="Comma-separated list")

    p_tcomp = sub.add_parser("task-complete", help="Complete a task")
    p_tcomp.add_argument("--task-id", required=True)
    p_tcomp.add_argument("--status", default="completed")
    p_tcomp.add_argument("--result", default="{}", help="Result as JSON object")
    p_tcomp.add_argument("--error", default="")

    args = parser.parse_args(argv)

    if importlib.util.find_spec("libp2p") is None:
        print(
            "error: optional dependency 'libp2p' is not installed in this environment\n"
            "Install with: pip install -e '.[libp2p]'",
            file=sys.stderr,
        )
        return 2

    try:
        remote = _remote_from_args(args)

        from ipfs_accelerate_py.p2p_tasks.client import (
            cache_delete_sync,
            cache_get_sync,
            cache_has_sync,
            cache_set_sync,
            call_tool_sync,
            list_tasks_sync,
            request_status_sync,
            submit_task_sync,
        )

        if args.cmd == "status":
            result = request_status_sync(remote=remote, timeout_s=float(args.timeout), detail=bool(args.detail))
            _print_result(result, pretty=bool(args.pretty))
            return 0

        if args.cmd == "call-tool":
            tool_args = _parse_json_obj(str(args.args or ""), flag="--args")
            result = call_tool_sync(remote=remote, tool_name=str(args.tool), args=tool_args, timeout_s=float(args.timeout))
            _print_result(result, pretty=bool(args.pretty))
            return 0

        if args.cmd == "cache-get":
            result = cache_get_sync(remote=remote, key=str(args.key), timeout_s=float(args.timeout))
            _print_result(result, pretty=bool(args.pretty))
            return 0

        if args.cmd == "cache-has":
            result = cache_has_sync(remote=remote, key=str(args.key), timeout_s=float(args.timeout))
            _print_result(result, pretty=bool(args.pretty))
            return 0

        if args.cmd == "cache-set":
            value = _parse_json_any(str(args.value), flag="--value")
            result = cache_set_sync(
                remote=remote,
                key=str(args.key),
                value=value,
                ttl_s=args.ttl,
                timeout_s=float(args.timeout),
            )
            _print_result(result, pretty=bool(args.pretty))
            return 0

        if args.cmd == "cache-delete":
            result = cache_delete_sync(remote=remote, key=str(args.key), timeout_s=float(args.timeout))
            _print_result(result, pretty=bool(args.pretty))
            return 0

        if args.cmd == "task-submit":
            payload = _parse_json_obj(str(args.payload or ""), flag="--payload")
            task_id = submit_task_sync(
                remote=remote,
                task_type=str(args.task_type),
                model_name=str(args.model_name),
                payload=payload,
            )
            _print_result({"ok": True, "task_id": str(task_id)}, pretty=bool(args.pretty))
            return 0

        if args.cmd == "task-list":
            task_types = [t.strip() for t in str(args.task_types or "").split(",") if t.strip()]
            result = list_tasks_sync(remote=remote, status=args.status, limit=int(args.limit), task_types=task_types or None)
            _print_result(result, pretty=bool(args.pretty))
            return 0

        # For get/wait/claim/complete, use anyio because the client exposes async versions.
        if args.cmd in {"task-get", "task-wait", "task-claim", "task-complete"}:
            import anyio

            from ipfs_accelerate_py.p2p_tasks import client as p2p_client

            async def _run_async() -> Any:
                if args.cmd == "task-get":
                    return await p2p_client.get_task(remote=remote, task_id=str(args.task_id))
                if args.cmd == "task-wait":
                    return await p2p_client.wait_task(remote=remote, task_id=str(args.task_id), timeout_s=float(args.timeout))
                if args.cmd == "task-claim":
                    supported = [t.strip() for t in str(args.supported_task_types or "").split(",") if t.strip()]
                    return await p2p_client.claim_next(remote=remote, worker_id=str(args.worker_id), supported_task_types=supported or None)
                if args.cmd == "task-complete":
                    result_obj = _parse_json_obj(str(args.result or ""), flag="--result")
                    err = str(args.error or "")
                    return await p2p_client.complete_task(
                        remote=remote,
                        task_id=str(args.task_id),
                        status=str(args.status),
                        result=result_obj,
                        error=(err or None),
                    )
                return {"ok": False, "error": "unknown_cmd"}

            async_result = anyio.run(_run_async, backend="trio")
            _print_result({"ok": True, "result": async_result}, pretty=bool(args.pretty))
            return 0

        raise SystemExit(f"unknown command: {args.cmd}")

    except SystemExit:
        raise
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
