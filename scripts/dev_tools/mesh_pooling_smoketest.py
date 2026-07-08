#!/usr/bin/env python
"""Multi-worker pooling smoke test for `llm.generate`.

What this tests
- You can run multiple `copilot_cli` workers (typically on different machines and
  different GitHub/Copilot accounts) against the same TaskQueue service.
- Submitting many `llm.generate` tasks concurrently results in work being
  distributed across multiple workers (as evidenced by `executor_worker_id` in
  task results).

This is intentionally *not* a benchmark; it’s a correctness + distribution check.

Usage (driver)
  ./.venv/bin/python scripts/dev_tools/mesh_pooling_smoketest.py --announce /path/to/announce.json --jobs 12 --concurrency 6

You can also specify `--peer-id` and `--multiaddr` directly.

Exit codes
- 0: ran successfully
- 2: completed but did not meet distribution expectation (`--expect-workers`)
- 3: could not contact service / invalid inputs
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANNOUNCE = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")) / "ipfs_accelerate_py" / "task_p2p_announce.json"


@dataclass
class JobResult:
    task_id: str
    status: str
    executor_worker_id: str
    error: str
    prompt: str
    text: str
    submitted_at_s: float
    completed_at_s: float
    elapsed_ms: int
    result_meta: Dict[str, Any]


def _iso_utc(ts: float) -> str:
    try:
        return datetime.datetime.fromtimestamp(float(ts), tz=datetime.timezone.utc).isoformat()
    except Exception:
        return ""


def _load_announce(path: Path) -> dict:
    data = json.loads(path.read_text("utf-8") or "{}")
    if not isinstance(data, dict):
        raise ValueError("announce file is not a JSON object")
    return data


def _remote(peer_id: str, multiaddr: str):
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue  # noqa: E402

    pid = str(peer_id or "").strip()
    ma = str(multiaddr or "").strip()
    if not pid or not ma:
        raise ValueError("peer_id and multiaddr are required")
    return RemoteQueue(peer_id=pid, multiaddr=ma)


async def _run_one(*, remote, provider: str, prompt: str, timeout_s: float) -> JobResult:
    import anyio
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from ipfs_accelerate_py.p2p_tasks.client import submit_task_with_info, wait_task, cancel_task  # noqa: E402

    payload: Dict[str, Any] = {
        "provider": provider,
        "prompt": prompt,
        "chat_session_id": f"pool-smoke-{uuid.uuid4().hex}",
        "timeout": max(30.0, float(timeout_s)),
    }
    session_id = str(os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_SESSION") or "").strip()
    if session_id:
        payload["session_id"] = session_id

    t_submit_wall = time.time()
    t_submit_mono = time.monotonic()

    info = await submit_task_with_info(remote=remote, task_type="llm.generate", model_name="", payload=payload)
    task_id = str(info.get("task_id") or "").strip()
    if not task_id:
        return JobResult(
            task_id="",
            status="submit_failed",
            executor_worker_id="",
            error="missing task_id",
            prompt=prompt,
            text="",
            submitted_at_s=t_submit_wall,
            completed_at_s=time.time(),
            elapsed_ms=int(max(0.0, (time.monotonic() - t_submit_mono)) * 1000.0),
            result_meta={},
        )

    task = await wait_task(remote=remote, task_id=task_id, timeout_s=float(timeout_s))
    if task is None:
        try:
            await cancel_task(remote=remote, task_id=task_id, reason="pooling smoketest timeout")
        except Exception:
            pass
        return JobResult(
            task_id=task_id,
            status="timeout",
            executor_worker_id="",
            error="timeout",
            prompt=prompt,
            text="",
            submitted_at_s=t_submit_wall,
            completed_at_s=time.time(),
            elapsed_ms=int(max(0.0, (time.monotonic() - t_submit_mono)) * 1000.0),
            result_meta={},
        )

    status = str(task.get("status") or "").strip() or "unknown"
    err = str(task.get("error") or "").strip()

    executor = ""
    result = task.get("result")
    text = ""
    meta: Dict[str, Any] = {}
    if isinstance(result, dict):
        executor = str(result.get("executor_worker_id") or "").strip()
        text = str(result.get("text") or "")
        meta = dict(result)
    if not executor:
        executor = str(task.get("assigned_worker") or "").strip()

    await anyio.sleep(0)
    return JobResult(
        task_id=task_id,
        status=status,
        executor_worker_id=executor,
        error=err,
        prompt=prompt,
        text=text,
        submitted_at_s=t_submit_wall,
        completed_at_s=time.time(),
        elapsed_ms=int(max(0.0, (time.monotonic() - t_submit_mono)) * 1000.0),
        result_meta=meta,
    )


async def _run_all(*, remote, jobs: int, concurrency: int, provider: str, prompt: str, timeout_s: float) -> list[JobResult]:
    import anyio

    results: list[JobResult] = []
    sem = anyio.Semaphore(max(1, int(concurrency)))

    async def _wrapped() -> None:
        async with sem:
            res = await _run_one(remote=remote, provider=provider, prompt=prompt, timeout_s=timeout_s)
            results.append(res)

    async with anyio.create_task_group() as tg:
        for _ in range(max(1, int(jobs))):
            tg.start_soon(_wrapped)

    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--announce", type=str, default="", help="Path to TaskQueue service announce JSON")
    ap.add_argument("--peer-id", type=str, default="", help="Service peer_id (overrides announce)")
    ap.add_argument("--multiaddr", type=str, default="", help="Service multiaddr (overrides announce)")
    ap.add_argument("--jobs", type=int, default=12, help="Total llm.generate tasks to submit")
    ap.add_argument("--concurrency", type=int, default=6, help="Max in-flight submissions")
    ap.add_argument("--timeout-s", type=float, default=120.0, help="Per-task wait timeout")
    ap.add_argument("--provider", type=str, default="copilot_cli", help="LLM provider for llm.generate")
    ap.add_argument("--prompt", type=str, default="Return exactly: OK", help="Prompt to send")
    ap.add_argument(
        "--transcript-jsonl",
        type=str,
        default="",
        help="Optional path to write per-task transcript JSONL (prompt/text + metadata)",
    )
    ap.add_argument(
        "--expect-workers",
        type=int,
        default=2,
        help="Exit code 2 if fewer distinct executor_worker_id values are observed",
    )
    ap.add_argument(
        "--session-id",
        type=str,
        default="",
        help="Optional session_id to include in payload (also exported as IPFS_ACCELERATE_PY_TASK_P2P_SESSION for the driver)",
    )

    args = ap.parse_args()

    if isinstance(args.session_id, str) and args.session_id.strip():
        os.environ["IPFS_ACCELERATE_PY_TASK_P2P_SESSION"] = args.session_id.strip()

    peer_id = str(args.peer_id or "").strip()
    multiaddr = str(args.multiaddr or "").strip()

    if not peer_id or not multiaddr:
        ann_path = Path(str(args.announce or "").strip() or str(DEFAULT_ANNOUNCE))
        if not ann_path.exists():
            print(f"ERROR: announce file not found: {ann_path}")
            print("Provide --announce, or --peer-id + --multiaddr.")
            return 3
        try:
            ann = _load_announce(ann_path)
        except Exception as e:
            print(f"ERROR: failed to read announce JSON: {e}")
            return 3
        peer_id = str(ann.get("peer_id") or "").strip()
        multiaddr = str(ann.get("multiaddr") or "").strip()

    try:
        remote = _remote(peer_id=peer_id, multiaddr=multiaddr)
    except Exception as e:
        print(f"ERROR: invalid remote settings: {e}")
        return 3

    jobs = max(1, int(args.jobs))
    concurrency = max(1, int(args.concurrency))
    timeout_s = max(5.0, float(args.timeout_s))
    provider = str(args.provider or "").strip().lower() or "copilot_cli"
    prompt = str(args.prompt or "").strip()
    transcript_path = str(args.transcript_jsonl or "").strip()

    print("=== mesh pooling smoketest ===")
    print(f"service peer_id: {peer_id}")
    print(f"service multiaddr: {multiaddr}")
    print(f"provider: {provider}")
    print(f"jobs: {jobs}  concurrency: {concurrency}  timeout_s: {timeout_s}")

    import anyio

    t0 = time.time()
    async def _do() -> list[JobResult]:
        return await _run_all(
            remote=remote,
            jobs=jobs,
            concurrency=concurrency,
            provider=provider,
            prompt=prompt,
            timeout_s=timeout_s,
        )

    results = anyio.run(_do, backend="trio")
    dt = max(0.001, time.time() - t0)

    if transcript_path:
        run_id = f"pool-smoke-{uuid.uuid4().hex}"
        out_path = Path(transcript_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Stable order for humans: by submission time.
        ordered = sorted(results, key=lambda r: (float(r.submitted_at_s or 0.0), r.task_id))
        with out_path.open("w", encoding="utf-8") as f:
            for r in ordered:
                meta = dict(r.result_meta or {})
                record = {
                    "run_id": run_id,
                    "task_id": r.task_id,
                    "provider": provider,
                    "prompt": r.prompt,
                    "text": r.text,
                    "status": r.status,
                    "error": r.error,
                    "submitted_at": _iso_utc(r.submitted_at_s),
                    "completed_at": _iso_utc(r.completed_at_s),
                    "elapsed_ms": int(r.elapsed_ms),
                    "session_id": str(meta.get("session_id") or ""),
                    "chat_session_id": str(meta.get("chat_session_id") or ""),
                    "resume_session_id": str(meta.get("resume_session_id") or ""),
                    "executor_worker_id": str(meta.get("executor_worker_id") or r.executor_worker_id or ""),
                    "executor_peer_id": str(meta.get("executor_peer_id") or ""),
                    "executor_multiaddr": str(meta.get("executor_multiaddr") or ""),
                    "service_peer_id": peer_id,
                    "service_multiaddr": multiaddr,
                }
                f.write(json.dumps(record, sort_keys=True) + "\n")

    by_worker: Dict[str, int] = {}
    status_counts: Dict[str, int] = {}

    for r in results:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1
        wid = r.executor_worker_id or "(unknown)"
        by_worker[wid] = by_worker.get(wid, 0) + 1

    distinct_workers = [w for w in by_worker.keys() if w and w != "(unknown)"]

    print("\n--- summary ---")
    print(f"elapsed_s: {dt:.2f}")
    print(f"tasks_per_s: {len(results) / dt:.2f}")
    print(f"status_counts: {json.dumps(status_counts, sort_keys=True)}")
    print("by_worker:")
    for wid, n in sorted(by_worker.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {wid}: {n}")

    expect = max(1, int(args.expect_workers))
    if len(distinct_workers) < expect:
        print(f"\nDID NOT MEET EXPECTATION: observed {len(distinct_workers)} distinct executor_worker_id values; expected >= {expect}.")
        return 2

    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
