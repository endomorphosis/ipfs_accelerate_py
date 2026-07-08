#!/usr/bin/env python3
"""Round-robin Docker task load tester for the P2P task queue.

Primary use-case: validate GPU-enabled docker execution across multiple peers, e.g.
50 runs of `nvidia/cuda:12.4.0-base nvidia-smi`.

Example:
  ./scripts/queue_docker_load.py \
    --announce-file /path/to/peerA.announce.json \
    --announce-file /path/to/peerB.announce.json \
    --tasks 50 --concurrency 10 \
    --image nvidia/cuda:12.4.0-base --gpus all --command nvidia-smi

Notes:
- Workers must have docker enabled (e.g. IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_DOCKER=1)
- GPU containers require NVIDIA Container Toolkit on the worker hosts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue, submit_docker_hub_task, wait_task


def _parse_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_announce_target(path: str) -> str:
    """Return a connection target string (multiaddr) from an announce file."""
    data = _parse_json_file(path)
    # Heuristic: accept several common formats.
    if isinstance(data, str):
        return data

    if isinstance(data, dict):
        for key in ("multiaddr", "address", "addr", "target"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

        maddrs = data.get("multiaddrs")
        if isinstance(maddrs, list) and maddrs:
            for m in maddrs:
                if isinstance(m, str) and m.strip():
                    return m.strip()

    raise ValueError(f"Unrecognized announce file format: {path}")


async def _build_queues(
    *,
    targets: Sequence[str],
    shared_token: Optional[str],
    connect_timeout_s: float,
) -> List[RemoteQueue]:
    queues: List[RemoteQueue] = []
    for t in targets:
        queues.append(
            RemoteQueue(
                target=t,
                shared_token=shared_token,
                connect_timeout_s=connect_timeout_s,
            )
        )
    return queues


async def _run_one(
    *,
    queue: RemoteQueue,
    image: str,
    command: Sequence[str],
    gpus: Optional[str],
    timeout_s: Optional[float],
    stream_output: bool,
) -> Dict[str, Any]:
    """Submit one docker task and wait for completion; return a result dict."""
    submitted_at = time.time()
    task_id = await submit_docker_hub_task(
        queue,
        image=image,
        command=list(command),
        # Worker supports this payload key; non-streaming path uses docker_executor.
        stream_output=stream_output,
        # New: docker run --gpus <value>
        gpus=gpus,
    )

    result = await wait_task(queue, task_id, timeout_s=timeout_s)
    finished_at = time.time()

    out: Dict[str, Any] = {
        "task_id": task_id,
        "target": getattr(queue, "target", None),
        "submitted_at": submitted_at,
        "finished_at": finished_at,
        "duration_s": finished_at - submitted_at,
        "result": result,
    }

    # Try to pull common docker executor fields (best-effort; schema may vary).
    if isinstance(result, dict):
        for k in ("success", "exit_code", "stdout", "stderr", "error", "timed_out"):
            if k in result:
                out[k] = result.get(k)

    return out


async def main_async(args: argparse.Namespace) -> int:
    targets: List[str] = []

    if args.target:
        targets.extend([t.strip() for t in args.target if t.strip()])

    if args.announce_file:
        for p in args.announce_file:
            targets.append(_load_announce_target(p))

    if len(targets) < 2:
        raise SystemExit("Provide at least 2 peers via --target or --announce-file")

    # Keep deterministic-ish runs unless user changes.
    if args.shuffle_targets:
        random.shuffle(targets)

    queues = await _build_queues(
        targets=targets,
        shared_token=args.shared_token,
        connect_timeout_s=args.connect_timeout_s,
    )

    sem = asyncio.Semaphore(args.concurrency)

    results: List[Dict[str, Any]] = []

    async def runner(i: int) -> None:
        q = queues[i % len(queues)]
        async with sem:
            try:
                r = await _run_one(
                    queue=q,
                    image=args.image,
                    command=args.command,
                    gpus=args.gpus,
                    timeout_s=args.timeout_s,
                    stream_output=args.stream_output,
                )
                r["index"] = i
                results.append(r)
            except Exception as e:  # noqa: BLE001
                results.append(
                    {
                        "index": i,
                        "target": getattr(q, "target", None),
                        "error": str(e),
                    }
                )

    started_at = time.time()
    await asyncio.gather(*[runner(i) for i in range(args.tasks)])
    finished_at = time.time()

    # Emit JSON report to stdout.
    report: Dict[str, Any] = {
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": finished_at - started_at,
        "tasks": args.tasks,
        "concurrency": args.concurrency,
        "image": args.image,
        "command": list(args.command),
        "gpus": args.gpus,
        "targets": targets,
        "results": sorted(results, key=lambda x: x.get("index", 0)),
    }

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))

    # Exit non-zero if any task failed.
    failures = 0
    for r in results:
        if r.get("error"):
            failures += 1
            continue
        # Heuristics: prefer explicit success/exit_code.
        if "success" in r and r.get("success") is False:
            failures += 1
        elif "exit_code" in r and r.get("exit_code") not in (0, "0", None):
            failures += 1

    return 0 if failures == 0 else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument(
        "--announce-file",
        action="append",
        default=[],
        help="Path to a peer announce JSON file (can be passed multiple times)",
    )
    p.add_argument(
        "--target",
        action="append",
        default=[],
        help="Peer target (e.g. multiaddr) (can be passed multiple times)",
    )

    p.add_argument("--tasks", type=int, default=50, help="Number of docker tasks to submit")
    p.add_argument("--concurrency", type=int, default=10, help="Max in-flight tasks")
    p.add_argument("--connect-timeout-s", type=float, default=10.0)
    p.add_argument("--timeout-s", type=float, default=180.0, help="Per-task wait timeout")

    p.add_argument("--image", type=str, default="nvidia/cuda:12.4.0-base")
    p.add_argument("--gpus", type=str, default="all", help='Value for docker "--gpus" (e.g. all, device=0)')
    p.add_argument("--stream-output", action="store_true", help="Use streaming execution path")

    p.add_argument(
        "--command",
        nargs="+",
        default=["nvidia-smi"],
        help="Container command (default: nvidia-smi)",
    )

    p.add_argument("--shared-token", type=str, default=os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_SHARED_TOKEN"))
    p.add_argument("--shuffle-targets", action="store_true", help="Shuffle target order before round-robin")
    p.add_argument("--output", type=str, default=None, help="Write JSON report to this path instead of stdout")

    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
