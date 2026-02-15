#!/usr/bin/env python3
"""Chatty smoketest for 2-peer systemd deployments.

Validates:
- Session gating: tasks tagged with session_id A should only execute on a peer
  whose TaskQueue worker session tag is A.
- Sticky resume: a second-round task with continue_session + sticky_worker_id
  should execute on the same worker that handled round 1.
- Throughput (best-effort): submits additional non-session tasks and reports
  how many distinct executors handled them.

This script is designed to run *outside* systemd, talking to already-running
TaskQueue P2P services.

Example:
  .venv/bin/python scripts/dev_tools/systemd_chatty_session_resume_smoketest.py \
    --peer-a-id Qm... --peer-a-multiaddr /ip4/.../tcp/9100/p2p/Qm... --peer-a-session S_A \
    --peer-b-id Qm... --peer-b-multiaddr /ip4/.../tcp/9100/p2p/Qm... --peer-b-session S_B \
    --provider copilot_cli --jobs 6 --timeout-s 180

Notes:
- For deterministic runs without real Copilot, ensure both peers have
  ipfs_accelerate_py_COPILOT_CLI_CMD set in their systemd env (e.g. secrets.env)
  and IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_COPILOT_CLI=1.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import anyio

from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue
from ipfs_accelerate_py.p2p_tasks.client import get_task, get_capabilities, submit_task


@dataclass
class Peer:
    name: str
    peer_id: str
    multiaddr: str
    session: str

    def remote(self) -> RemoteQueue:
        return RemoteQueue(peer_id=self.peer_id, multiaddr=self.multiaddr)


def _now() -> float:
    return time.time()


async def _get_task_retry(*, remote: RemoteQueue, task_id: str, deadline: float) -> Optional[Dict[str, Any]]:
    backoff = 0.2
    while _now() < deadline:
        try:
            return await get_task(remote=remote, task_id=str(task_id))
        except Exception:
            await anyio.sleep(backoff)
            backoff = min(2.0, backoff * 1.5)
    return None


async def _wait_completed_chatty(
    *,
    remote: RemoteQueue,
    task_id: str,
    timeout_s: float,
) -> Optional[Dict[str, Any]]:
    deadline = _now() + max(1.0, float(timeout_s))
    last_status = None

    while _now() < deadline:
        task = await _get_task_retry(remote=remote, task_id=task_id, deadline=min(deadline, _now() + 5.0))
        if isinstance(task, dict):
            status = str(task.get("status") or "").strip().lower()
            if status and status != last_status:
                last_status = status
            if status in {"completed", "failed", "cancelled"}:
                return task
        await anyio.sleep(0.25)
    return None


def _fmt_excerpt(text: str, *, max_chars: int = 200) -> str:
    s = str(text or "")
    s = " ".join(s.split())
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def _print_block(title: str, block: Dict[str, Any]) -> None:
    print("\n" + ("=" * 90))
    print(title)
    print("-" * 90)
    print(json.dumps(block, indent=2, sort_keys=True))


async def _probe_peer(peer: Peer) -> Dict[str, Any]:
    caps = await get_capabilities(remote=peer.remote(), timeout_s=8.0, detail=True)
    return caps


async def _submit_llm_task(
    *,
    peer: Peer,
    prompt: str,
    provider: str,
    model_name: str,
    session_id: Optional[str],
    sticky_worker_id: Optional[str] = None,
    chat_session_id: Optional[str] = None,
    continue_session: bool = False,
) -> str:
    payload: Dict[str, Any] = {
        "prompt": str(prompt),
        "provider": str(provider),
    }
    if session_id:
        payload["session_id"] = str(session_id)
    if sticky_worker_id:
        payload["sticky_worker_id"] = str(sticky_worker_id)
    if chat_session_id:
        payload["chat_session_id"] = str(chat_session_id)
    if continue_session:
        payload["continue_session"] = True

    tid = await submit_task(remote=peer.remote(), task_type="llm.generate", model_name=str(model_name), payload=payload)
    return str(tid)


def _result_executor(task: Dict[str, Any]) -> Tuple[str, str]:
    res = task.get("result")
    if not isinstance(res, dict):
        return ("", "")
    return (str(res.get("executor_peer_id") or "").strip(), str(res.get("executor_worker_id") or "").strip())


def _result_text(task: Dict[str, Any]) -> str:
    res = task.get("result")
    if not isinstance(res, dict):
        return ""
    return str(res.get("text") or "")


async def main_async(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--peer-a-id", required=True)
    p.add_argument("--peer-a-multiaddr", required=True)
    p.add_argument("--peer-a-session", required=True)
    p.add_argument("--peer-b-id", required=True)
    p.add_argument("--peer-b-multiaddr", required=True)
    p.add_argument("--peer-b-session", required=True)

    p.add_argument("--provider", default="copilot_cli")
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--timeout-s", type=float, default=180.0)
    args = p.parse_args(argv)

    peer_a = Peer("A", args.peer_a_id, args.peer_a_multiaddr, args.peer_a_session)
    peer_b = Peer("B", args.peer_b_id, args.peer_b_multiaddr, args.peer_b_session)

    if peer_a.session == peer_b.session:
        print("WARNING: peer sessions are identical; session-gating test is not meaningful.")

    # Probe
    for peer in (peer_a, peer_b):
        try:
            caps = await _probe_peer(peer)
            _print_block(f"Peer {peer.name} capabilities", {"peer": peer.__dict__, "capabilities": caps})
        except Exception as exc:
            _print_block(f"Peer {peer.name} probe FAILED", {"peer": peer.__dict__, "error": str(exc)})
            return 2

    # Phase 1: session gating + sticky resume
    chat_id = f"chat-{int(time.time())}"
    prompt1 = "(smoketest) Round 1: say OK and echo the word ALPHA."
    tid1 = await _submit_llm_task(
        peer=peer_a,
        prompt=prompt1,
        provider=str(args.provider),
        model_name=str(args.model),
        session_id=peer_a.session,
        chat_session_id=chat_id,
    )

    t1 = await _wait_completed_chatty(remote=peer_a.remote(), task_id=tid1, timeout_s=float(args.timeout_s))
    if not isinstance(t1, dict):
        _print_block("Round 1 TIMEOUT", {"task_id": tid1, "peer": peer_a.__dict__})
        return 3

    ex_peer1, ex_worker1 = _result_executor(t1)
    block1 = {
        "phase": "round1",
        "task_id": tid1,
        "submit_peer": peer_a.__dict__,
        "status": t1.get("status"),
        "error": t1.get("error"),
        "executor_peer_id": ex_peer1,
        "executor_worker_id": ex_worker1,
        "result_excerpt": _fmt_excerpt(_result_text(t1)),
        "result": t1.get("result"),
    }
    _print_block("Round 1 result", block1)

    if str(t1.get("status") or "").lower() != "completed":
        return 4

    if peer_a.session != peer_b.session and ex_peer1 and ex_peer1 != peer_a.peer_id:
        _print_block(
            "FAIL: Session-gated task executed on wrong peer",
            {"expected_peer_id": peer_a.peer_id, "got_peer_id": ex_peer1, "task": block1},
        )
        return 10

    if not ex_worker1:
        _print_block("FAIL: Missing executor_worker_id", {"task": block1})
        return 11

    prompt2 = "(smoketest) Round 2: continue the previous session and say OK then echo BETA."
    tid2 = await _submit_llm_task(
        peer=peer_a,
        prompt=prompt2,
        provider=str(args.provider),
        model_name=str(args.model),
        session_id=peer_a.session,
        sticky_worker_id=ex_worker1,
        chat_session_id=chat_id,
        continue_session=True,
    )

    t2 = await _wait_completed_chatty(remote=peer_a.remote(), task_id=tid2, timeout_s=float(args.timeout_s))
    if not isinstance(t2, dict):
        _print_block("Round 2 TIMEOUT", {"task_id": tid2, "peer": peer_a.__dict__, "sticky_worker_id": ex_worker1})
        return 5

    ex_peer2, ex_worker2 = _result_executor(t2)
    block2 = {
        "phase": "round2",
        "task_id": tid2,
        "submit_peer": peer_a.__dict__,
        "sticky_worker_id": ex_worker1,
        "status": t2.get("status"),
        "error": t2.get("error"),
        "executor_peer_id": ex_peer2,
        "executor_worker_id": ex_worker2,
        "result_excerpt": _fmt_excerpt(_result_text(t2)),
        "result": t2.get("result"),
    }
    _print_block("Round 2 result", block2)

    if str(t2.get("status") or "").lower() != "completed":
        return 6

    if ex_worker2 and ex_worker2 != ex_worker1:
        _print_block(
            "FAIL: Sticky resume executed on different worker",
            {"expected_worker_id": ex_worker1, "got_worker_id": ex_worker2, "task": block2},
        )
        return 12

    # Phase 2: throughput observation (best-effort)
    jobs = max(0, int(args.jobs))
    if jobs <= 0:
        print("PASS")
        return 0

    tids: list[Tuple[str, str]] = []
    for i in range(jobs):
        prompt = f"(smoketest) throughput job {i+1}/{jobs}: say OK and echo JOB{i+1}."
        submit_peer = peer_a if (i % 2 == 0) else peer_b
        tid = await _submit_llm_task(
            peer=submit_peer,
            prompt=prompt,
            provider=str(args.provider),
            model_name=str(args.model),
            session_id=None,
        )
        tids.append((submit_peer.name, tid))

    executors: Dict[str, int] = {}
    for submit_name, tid in tids:
        submit_peer = peer_a if submit_name == "A" else peer_b
        t = await _wait_completed_chatty(remote=submit_peer.remote(), task_id=tid, timeout_s=float(args.timeout_s))
        if not isinstance(t, dict):
            _print_block("Job TIMEOUT", {"task_id": tid, "submit_peer": submit_peer.__dict__})
            continue
        ex_peer, ex_worker = _result_executor(t)
        key = ex_peer or ex_worker or "unknown"
        executors[key] = executors.get(key, 0) + 1
        _print_block(
            f"Job result ({submit_name})",
            {
                "task_id": tid,
                "submit_peer": submit_peer.__dict__,
                "status": t.get("status"),
                "error": t.get("error"),
                "executor_peer_id": ex_peer,
                "executor_worker_id": ex_worker,
                "result_excerpt": _fmt_excerpt(_result_text(t)),
            },
        )

    _print_block("Throughput summary", {"jobs": jobs, "executors": executors, "distinct_executors": len(executors)})

    # If we expected >1 executor but saw only one, fail with actionable hints.
    if jobs >= 2 and len(executors) < 2:
        _print_block(
            "WARN/FAIL: Only one executor observed",
            {
                "hint": (
                    "If you expected both peers to execute copilot_cli, verify on BOTH hosts: "
                    "(1) IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_COPILOT_CLI=1, "
                    "(2) copilot CLI is installed OR ipfs_accelerate_py_COPILOT_CLI_CMD is set to a deterministic command, "
                    "(3) mesh is enabled and remote autoscale/mesh-children are enabled if you want draining."
                ),
                "executors": executors,
            },
        )
        return 20

    print("PASS")
    return 0


def main() -> int:
    try:
        return anyio.run(main_async, backend="trio")
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
