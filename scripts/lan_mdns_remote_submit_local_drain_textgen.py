#!/usr/bin/env python3
"""LAN demo: submit GPT-2 textgen tasks to a *remote* peer and drain them locally via mesh.

This script runs on ONE machine (the "drainer"):
1) Starts a local TaskQueue worker in mesh mode (no local service needed).
2) Submits `text-generation` tasks to the remote peer's TaskQueue.
3) Waits for the remote peer to report completion.
4) Saves a JSON report including whether tasks were executed via mesh on this host.

Example:
  ./.venv/bin/python scripts/lan_mdns_remote_submit_local_drain_textgen.py \
    --remote-peer-id 12D3KooWEhisnDXrezovwTPk5kG8DGb2UQiB6ZVbqsb9Xrjp1Jzi \
    --remote-multiaddr /ip4/192.168.0.54/tcp/9101/p2p/12D3KooWEhisnDXrezovwTPk5kG8DGb2UQiB6ZVbqsb9Xrjp1Jzi \
    --count 20

Notes:
- Requires multicast/mDNS to be allowed so the local worker can discover peers.
- The local worker will only claim "mesh-safe" task types; text-generation is allowed.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Optional


def _hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "peer"


def _now_ts() -> float:
    return float(time.time())


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_mesh_worker_process(*, queue_path: str, worker_id: str, mesh_peers: str) -> None:
    # Keep this process minimal and avoid optional subsystems.
    os.environ["IPFS_KIT_DISABLE"] = "1"
    os.environ["STORAGE_FORCE_LOCAL"] = "1"
    os.environ["TRANSFORMERS_PATCH_DISABLE"] = "1"
    os.environ["IPFS_ACCEL_SKIP_CORE"] = "1"

    # Ensure text-generation works without pulling in the full accelerator stack.
    os.environ.setdefault("IPFS_ACCELERATE_PY_TASK_WORKER_MINIMAL_LLM", "1")

    # Pin mesh to the intended remote peer(s) so we don't waste cycles probing
    # unrelated LAN peers.
    if str(mesh_peers or "").strip():
        os.environ["IPFS_ACCELERATE_PY_TASK_WORKER_MESH_PEERS"] = str(mesh_peers).strip()

    from ipfs_accelerate_py.p2p_tasks.worker import run_worker

    run_worker(
        queue_path=str(queue_path),
        worker_id=str(worker_id),
        poll_interval_s=0.05,
        p2p_service=False,
        mesh=True,
        mesh_refresh_s=1.0,
        mesh_claim_interval_s=0.1,
        mesh_max_peers=25,
    )


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit remote textgen tasks and drain locally via mesh")

    parser.add_argument("--remote-peer-id", required=True, help="Remote peer_id to submit tasks to")
    parser.add_argument("--remote-multiaddr", required=True, help="Remote peer multiaddr (includes /p2p/<peer_id>)")

    parser.add_argument("--count", type=int, default=20, help="Number of text-generation tasks to submit")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent wait RPCs")
    parser.add_argument("--timeout-s", type=float, default=240.0, help="Per-task wait timeout")

    parser.add_argument("--prompt", default="The quick brown fox", help="Prompt for text generation")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.2)

    parser.add_argument("--state-dir", default="state/p2p_remote_drain", help="Directory for queue/report")
    parser.add_argument("--worker-id", default="", help="Local worker id (default: hostname + timestamp)")

    return parser.parse_args(argv)


def _extract_progress(task: Dict[str, Any]) -> Dict[str, Any]:
    result = task.get("result")
    if not isinstance(result, dict):
        return {}
    prog = result.get("progress")
    return prog if isinstance(prog, dict) else {}


async def _submit_and_wait(*, remote_peer_id: str, remote_multiaddr: str, args: argparse.Namespace, worker_id: str) -> Dict[str, Any]:
    import anyio

    from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue, submit_task_with_info, wait_task

    remote = RemoteQueue(peer_id=str(remote_peer_id), multiaddr=str(remote_multiaddr))

    submitted: List[Dict[str, Any]] = []
    for i in range(int(args.count)):
        payload = {
            "prompt": str(args.prompt),
            "max_new_tokens": int(args.max_new_tokens),
            "temperature": float(args.temperature),
            "meta": {"submitted_by": worker_id, "index": i},
        }
        info = await submit_task_with_info(remote=remote, task_type="text-generation", model_name="gpt2", payload=payload)
        submitted.append({"task_id": info.get("task_id"), "submit_info": info, "payload": payload})

    # Wait for completions with bounded concurrency.
    results_by_id: Dict[str, Any] = {}

    async def _wait_one(task_id: str) -> None:
        # `wait` is a long-poll RPC; it may return None on timeout.
        deadline = anyio.current_time() + float(args.timeout_s)
        last: Optional[Dict[str, Any]] = None
        while anyio.current_time() < deadline:
            t = await wait_task(remote=remote, task_id=str(task_id), timeout_s=min(60.0, float(args.timeout_s)))
            if t is not None:
                last = t
                break
            await anyio.sleep(0.2)
        results_by_id[str(task_id)] = last

    task_ids = [str(x.get("task_id") or "") for x in submitted if x.get("task_id")]

    async with anyio.create_task_group() as tg:
        sem = anyio.Semaphore(int(max(1, args.concurrency)))

        async def _bounded(task_id: str) -> None:
            async with sem:
                await _wait_one(task_id)

        for tid in task_ids:
            tg.start_soon(_bounded, tid)

    completed = [
        {
            "task_id": tid,
            "task": results_by_id.get(tid),
            "progress": _extract_progress(results_by_id.get(tid) or {}),
        }
        for tid in task_ids
    ]

    mesh_executed = [
        x for x in completed if isinstance(x.get("progress"), dict) and x["progress"].get("mesh") is True and x["progress"].get("worker_id") == worker_id
    ]

    return {
        "remote": {"peer_id": str(remote_peer_id), "multiaddr": str(remote_multiaddr)},
        "local": {"worker_id": str(worker_id), "host": _hostname()},
        "submitted": submitted,
        "completed": completed,
        "summary": {
            "submitted": len(submitted),
            "completed": sum(1 for x in completed if isinstance(x.get("task"), dict) and x["task"].get("status") in {"completed", "failed"}),
            "mesh_executed_by_this_host": len(mesh_executed),
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    # Keep the parent process minimal too.
    os.environ.setdefault("IPFS_KIT_DISABLE", "1")
    os.environ.setdefault("STORAGE_FORCE_LOCAL", "1")
    os.environ.setdefault("TRANSFORMERS_PATCH_DISABLE", "1")
    os.environ.setdefault("IPFS_ACCEL_SKIP_CORE", "1")
    os.environ.setdefault("IPFS_ACCELERATE_PY_TASK_WORKER_MINIMAL_LLM", "1")

    state_dir = Path(str(args.state_dir))
    state_dir.mkdir(parents=True, exist_ok=True)
    queue_path = str(state_dir / "local_queue.duckdb")

    worker_id = str(args.worker_id or "").strip() or f"drain-{_hostname()}-{int(_now_ts())}"

    report_path = state_dir / f"remote_drain_report_{worker_id}.json"

    ctx = get_context("spawn")
    proc = ctx.Process(
        target=_run_mesh_worker_process,
        kwargs={"queue_path": queue_path, "worker_id": worker_id, "mesh_peers": str(args.remote_multiaddr)},
        daemon=True,
    )
    proc.start()

    try:
        # Give the worker time to start and begin mDNS discovery.
        time.sleep(1.0)

        import anyio

        started = _now_ts()
        async def _main_async() -> Dict[str, Any]:
            return await _submit_and_wait(
                remote_peer_id=str(args.remote_peer_id),
                remote_multiaddr=str(args.remote_multiaddr),
                args=args,
                worker_id=worker_id,
            )

        report = anyio.run(_main_async, backend="trio")
        report["timing"] = {"started": started, "finished": _now_ts(), "duration_s": _now_ts() - started}

        _write_json(report_path, report)

        summary = report.get("summary") if isinstance(report, dict) else {}
        print("report", str(report_path))
        print("submitted", summary.get("submitted"))
        print("completed", summary.get("completed"))
        print("mesh_executed_by_this_host", summary.get("mesh_executed_by_this_host"))

        return 0
    finally:
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.join(timeout=3.0)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
