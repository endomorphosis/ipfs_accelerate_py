#!/usr/bin/env python3
"""Submit lots of text-generation tasks to one or more TaskQueue P2P peers.

Default task_type is `text-generation` and default model is `gpt2`.

Examples:
  # Submit 200 GPT-2 generations to a single peer (do not wait)
  python scripts/queue_textgen_load.py --announce-file /var/cache/ipfs-accelerate/task_p2p_announce.json --count 200

  # Submit and wait for completion (bounded concurrency)
  python scripts/queue_textgen_load.py --announce-file /var/cache/ipfs-accelerate/task_p2p_announce.json --count 50 --wait

  # Round-robin across multiple peers
  python scripts/queue_textgen_load.py --announce-file a.json --announce-file b.json --count 200 --concurrency 20 --wait

Output is a single JSON object on stdout.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple


def _load_announce(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.loads(handle.read())
    return data if isinstance(data, dict) else {}


def _print_json(obj: Any) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False))
    sys.stdout.write("\n")
    try:
        sys.stdout.flush()
    except Exception:
        pass


def _build_remote_targets(args: argparse.Namespace) -> List[Tuple[str, str]]:
    targets: list[Tuple[str, str]] = []
    for ap in args.announce_file or []:
        info = _load_announce(str(ap))
        ma = str(info.get("multiaddr") or "").strip()
        pid = str(info.get("peer_id") or "").strip()
        if not ma and not pid:
            raise SystemExit(f"announce file missing peer_id/multiaddr: {ap}")
        targets.append((pid, ma))

    for multiaddr in args.multiaddr or []:
        targets.append((str(args.peer_id or "").strip(), str(multiaddr).strip()))

    if not targets:
        # Allow peer-id-only discovery when caller sets shared discovery env.
        pid = str(args.peer_id or "").strip()
        if not pid:
            raise SystemExit("provide at least one --announce-file or --multiaddr (or set --peer-id for discovery)")
        targets.append((pid, ""))
    return targets


async def _main_async(args: argparse.Namespace) -> Dict[str, Any]:
    if args.count <= 0:
        raise SystemExit("--count must be > 0")
    concurrency = max(1, int(args.concurrency or 1))

    # Keep stdout JSON-only.
    json_stdout = sys.stdout
    sys.stdout = sys.stderr

    try:
        import anyio

        with contextlib.redirect_stdout(sys.stderr):
            from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue, submit_task, wait_task

        targets = _build_remote_targets(args)
        remotes = [RemoteQueue(peer_id=pid, multiaddr=ma) for (pid, ma) in targets]

        prompt = str(args.prompt or "Hello")
        task_type = str(args.task_type or "text-generation")
        model_name = str(args.model or "gpt2")

        payload_base: Dict[str, Any] = {
            "prompt": prompt,
            "max_new_tokens": int(args.max_new_tokens),
            "temperature": float(args.temperature),
        }
        if args.endpoint:
            payload_base["endpoint"] = str(args.endpoint)
        if args.endpoint_type:
            payload_base["endpoint_type"] = str(args.endpoint_type)

        submitted: list[Dict[str, Any]] = []
        submit_lock = anyio.Lock()
        start = time.time()

        sem = anyio.Semaphore(concurrency)

        async def _submit_one(i: int) -> None:
            remote = remotes[i % len(remotes)]
            payload = dict(payload_base)
            if args.suffix_index:
                payload["prompt"] = f"{prompt} [{i}]"
            async with sem:
                tid = await submit_task(remote=remote, task_type=task_type, model_name=model_name, payload=payload)
            async with submit_lock:
                submitted.append(
                    {
                        "task_id": str(tid),
                        "peer_id": str(getattr(remote, "peer_id", "") or ""),
                        "multiaddr": str(getattr(remote, "multiaddr", "") or ""),
                        "i": int(i),
                    }
                )

        async with anyio.create_task_group() as tg:
            for i in range(int(args.count)):
                tg.start_soon(_submit_one, i)

        submit_elapsed_s = float(time.time() - start)

        out: Dict[str, Any] = {
            "ok": True,
            "task_type": task_type,
            "model": model_name,
            "count": int(args.count),
            "concurrency": int(concurrency),
            "targets": [{"peer_id": pid, "multiaddr": ma} for (pid, ma) in targets],
            "submitted": submitted,
            "submit_elapsed_s": submit_elapsed_s,
            "wait": bool(args.wait),
        }

        if not args.wait:
            return out

        # Wait for completion (best-effort) with bounded concurrency.
        wait_sem = anyio.Semaphore(concurrency)
        results_lock = anyio.Lock()
        completed = 0
        failed = 0
        timed_out = 0
        latencies_s: list[float] = []

        async def _wait_one(item: Dict[str, Any]) -> None:
            nonlocal completed, failed, timed_out
            peer_id = str(item.get("peer_id") or "")
            task_id = str(item.get("task_id") or "")
            i = int(item.get("i") or 0)
            remote = remotes[i % len(remotes)]
            t0 = time.time()
            async with wait_sem:
                task = await wait_task(remote=remote, task_id=task_id, timeout_s=float(args.timeout_s))
            dt = float(time.time() - t0)
            async with results_lock:
                if task is None:
                    timed_out += 1
                else:
                    st = str(task.get("status") or "")
                    if st == "completed":
                        completed += 1
                    else:
                        failed += 1
                latencies_s.append(dt)

        wait_start = time.time()
        async with anyio.create_task_group() as tg:
            for item in list(submitted):
                tg.start_soon(_wait_one, item)
        wait_elapsed_s = float(time.time() - wait_start)

        lat_sorted = sorted([x for x in latencies_s if isinstance(x, (int, float))])
        def _p(pct: float) -> Optional[float]:
            if not lat_sorted:
                return None
            k = int(round((pct / 100.0) * (len(lat_sorted) - 1)))
            k = max(0, min(k, len(lat_sorted) - 1))
            return float(lat_sorted[k])

        out.update(
            {
                "wait_elapsed_s": wait_elapsed_s,
                "completed": int(completed),
                "failed": int(failed),
                "timed_out": int(timed_out),
                "latency_s": {
                    "min": float(lat_sorted[0]) if lat_sorted else None,
                    "p50": _p(50.0),
                    "p90": _p(90.0),
                    "p99": _p(99.0),
                    "max": float(lat_sorted[-1]) if lat_sorted else None,
                },
            }
        )
        return out
    finally:
        sys.stdout = json_stdout


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Queue lots of text-generation tasks on P2P peers")
    parser.add_argument(
        "--announce-file",
        action="append",
        default=[],
        help="Path to peer announce JSON {peer_id, multiaddr}. Repeatable.",
    )
    parser.add_argument(
        "--multiaddr",
        action="append",
        default=[],
        help="Remote multiaddr (/ip4/.../tcp/.../p2p/...). Repeatable.",
    )
    parser.add_argument(
        "--peer-id",
        default="",
        help="Peer-id hint (used when using discovery or when --multiaddr lacks /p2p/)",
    )

    parser.add_argument("--task-type", default="text-generation")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--suffix-index", action="store_true", help="Append [i] to prompt")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--endpoint-type", default="")

    parser.add_argument("--count", type=int, required=True, help="Total number of tasks to submit")
    parser.add_argument("--concurrency", type=int, default=10, help="Submit/wait concurrency")
    parser.add_argument("--wait", action="store_true", help="Wait for completion")
    parser.add_argument("--timeout-s", type=float, default=120.0, help="Per-task wait timeout")

    args = parser.parse_args(argv)

    # Dependency check.
    try:
        import importlib.util

        if importlib.util.find_spec("libp2p") is None:
            _print_json(
                {
                    "ok": False,
                    "error": "optional dependency 'libp2p' is not installed",
                    "hint": "pip install -e '.[libp2p]'",
                }
            )
            return 2
    except Exception:
        pass

    try:
        import anyio

        result = anyio.run(_main_async, args, backend="trio")
        _print_json(result)
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        _print_json({"ok": False, "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
