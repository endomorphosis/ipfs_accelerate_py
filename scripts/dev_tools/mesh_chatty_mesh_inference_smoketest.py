#!/usr/bin/env python
"""Chatty mesh inference smoketest.

Goals
- Demonstrate throughput scaling: multiple mesh workers drain a single peer's queue.
- Demonstrate session gating: tasks tagged with a `session_id` are only drained by
  workers configured with the same P2P session.
- Optionally exercise Copilot CLI resume/continue flags when the native `copilot`
  binary is available.

This script is designed to be human-readable: it prints per-task "chat" blocks
with metadata (executor worker/peer ids, elapsed time, session ids, etc.).

Topology (local simulation)
- Peer A: runs the TaskQueue RPC service and owns the queue.
- Peer B (mesh drainer): runs N worker threads in mesh mode (static peer list)
    and drains from A.

Notes
- For repeatability without a real Copilot install, you can set:
    ipfs_accelerate_py_COPILOT_CLI_CMD='bash -lc "echo OK"'
  (This uses the command-template mode of the provider; resume/continue flags
   require the native `copilot` binary and will be skipped if unavailable.)

Usage
  ./.venv/bin/python scripts/dev_tools/mesh_chatty_mesh_inference_smoketest.py \
      --workers 2 --jobs 16 --concurrency 8

Exit codes
- 0: PASS
- 2: FAILED expectations
- 3: invalid inputs / could not contact service
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PY = REPO_ROOT / ".venv" / "bin" / "python"


def _must_python() -> str:
    py = str(VENV_PY)
    if os.path.exists(py):
        return py
    return sys.executable


def _pick_free_port() -> int:
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _read_json(path: Path, *, timeout_s: float = 20.0) -> dict:
    deadline = time.time() + float(timeout_s)
    last_exc: Exception | None = None
    while time.time() < deadline:
        try:
            if path.exists() and path.stat().st_size > 0:
                data = json.loads(path.read_text("utf-8"))
                if isinstance(data, dict):
                    return data
        except Exception as exc:
            last_exc = exc
        time.sleep(0.05)
    raise RuntimeError(f"timed out waiting for json file: {path} ({last_exc})")


def _kill(proc: subprocess.Popen, *, name: str) -> None:
    try:
        if proc.poll() is not None:
            return
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=3)
            return
        except Exception:
            pass
        proc.kill()
    except Exception:
        pass


def _remote_from_announce(ann: dict):
    sys.path.insert(0, str(REPO_ROOT))
    from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue  # noqa: E402

    peer_id = str(ann.get("peer_id") or "").strip()
    multiaddr = str(ann.get("multiaddr") or "").strip()
    if not peer_id or not multiaddr:
        raise ValueError(f"invalid announce: {ann!r}")
    return RemoteQueue(peer_id=peer_id, multiaddr=multiaddr)


async def _wait_task(remote, task_id: str, *, timeout_s: float) -> Optional[dict]:
    sys.path.insert(0, str(REPO_ROOT))
    from ipfs_accelerate_py.p2p_tasks.client import wait_task  # noqa: E402

    deadline = time.time() + float(timeout_s)
    last_exc: Exception | None = None
    while time.time() < deadline:
        try:
            return await wait_task(remote=remote, task_id=str(task_id), timeout_s=max(1.0, deadline - time.time()))
        except Exception as exc:
            last_exc = exc
            # Transient libp2p failures (stream closed / handshake) can occur under load.
            # Retry until the overall deadline.
            import anyio

            await anyio.sleep(0.2)
    if last_exc is not None:
        raise last_exc
    return None


async def _poll_task(remote, task_id: str, *, timeout_s: float, poll_s: float = 0.2) -> Optional[dict]:
    """Poll task status over p2p until terminal.

    This avoids DuckDB lock contention (local reads) and avoids long-poll
    flakiness/latency from the `wait` RPC.
    """

    import anyio

    sys.path.insert(0, str(REPO_ROOT))
    from ipfs_accelerate_py.p2p_tasks.client import get_task  # noqa: E402

    deadline = time.time() + float(timeout_s)
    last: Optional[dict] = None
    while time.time() < deadline:
        try:
            last = await get_task(remote=remote, task_id=str(task_id))
        except Exception:
            last = None

        if isinstance(last, dict):
            st = str(last.get("status") or "").strip().lower()
            if st in {"completed", "failed"}:
                return last

        await anyio.sleep(float(max(0.05, poll_s)))
    return last


def _read_task_readonly(queue_path: str, task_id: str) -> Optional[dict]:
    """Read a task row using a read-only DuckDB connection.

    Using TaskQueue.get() opens a read-write connection, which conflicts with the
    service process holding a write lock on the DB file. Read-only polling avoids
    that lock contention.
    """

    if not task_id:
        return None
    try:
        import duckdb

        conn = duckdb.connect(str(queue_path), read_only=True)
        try:
            row = conn.execute(
                "SELECT task_id, status, assigned_worker, result_json, error FROM tasks WHERE task_id = ?",
                (str(task_id),),
            ).fetchone()
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except Exception:
        return None

    if not row:
        return None

    _tid, status, assigned_worker, result_json, error = row
    result: Any = None
    if isinstance(result_json, str) and result_json:
        try:
            result = json.loads(result_json)
        except Exception:
            result = result_json
    return {
        "task_id": str(_tid),
        "status": str(status or ""),
        "assigned_worker": str(assigned_worker) if assigned_worker else None,
        "result": result,
        "error": str(error or ""),
    }


async def _wait_task_local(queue_path: str, task_id: str, *, timeout_s: float) -> Optional[dict]:
    import anyio

    deadline = time.time() + float(timeout_s)
    last: Optional[dict] = None
    while time.time() < deadline:
        last = _read_task_readonly(queue_path, task_id)
        if isinstance(last, dict):
            st = str(last.get("status") or "").strip().lower()
            if st in {"completed", "failed"}:
                return last
        await anyio.sleep(0.1)
    return last


def _submit_task(remote, *, payload: dict) -> str:
    sys.path.insert(0, str(REPO_ROOT))
    from ipfs_accelerate_py.p2p_tasks.client import submit_task_sync  # noqa: E402

    return submit_task_sync(remote=remote, task_type="llm.generate", model_name="", payload=payload)


def _short(s: str, *, max_len: int = 180) -> str:
    t = str(s or "")
    if len(t) <= max_len:
        return t
    return t[: max(0, max_len - 3)] + "..."


@dataclass
class Completed:
    task_id: str
    status: str
    error: str
    prompt: str
    text: str
    elapsed_ms: int
    meta: Dict[str, Any]


def _render_chat(*, c: Completed, idx: int) -> str:
    meta = dict(c.meta or {})
    ex_w = str(meta.get("executor_worker_id") or "")
    ex_peer = str(meta.get("executor_peer_id") or "")
    ex_ma = str(meta.get("executor_multiaddr") or "")
    session_id = str(meta.get("session_id") or "")
    chat_session_id = str(meta.get("chat_session_id") or "")
    resume_session_id = str(meta.get("resume_session_id") or "")

    header = (
        f"\n=== chat[{idx}] task_id={c.task_id} status={c.status} elapsed_ms={c.elapsed_ms} ===\n"
        f"executor_worker_id: {ex_w or '(unknown)'}\n"
        f"executor_peer_id:   {ex_peer or '(unknown)'}\n"
        f"executor_multiaddr: {ex_ma or '(unknown)'}\n"
        f"session_id:         {session_id or '(none)'}\n"
        f"chat_session_id:    {chat_session_id or '(none)'}\n"
        f"resume_session_id:  {resume_session_id or '(none)'}\n"
    )

    body = f"User: {c.prompt}\nAssistant: {c.text}\n"
    if c.error:
        body += f"Error: {c.error}\n"
    return header + body


def _expect_only_executors(completed: Iterable[Completed], allowed_worker_ids: set[str]) -> bool:
    for c in completed:
        wid = str((c.meta or {}).get("executor_worker_id") or "") or "(unknown)"
        if wid not in allowed_worker_ids:
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Chatty mesh inference smoketest")
    ap.add_argument("--workers", type=int, default=2, help="Number of mesh draining workers to start")
    ap.add_argument("--jobs", type=int, default=16, help="Number of no-session jobs to submit")
    ap.add_argument("--concurrency", type=int, default=8, help="Max in-flight waits")
    ap.add_argument("--timeout-s", type=float, default=90.0, help="Per-task wait timeout")
    ap.add_argument(
        "--session-a",
        type=str,
        default="S1",
        help="Session tag for peer A queue (used only for the session-gating phase)",
    )
    ap.add_argument(
        "--session-b",
        type=str,
        default="S2",
        help="Session tag assigned to exactly one worker (used for session-gating phase)",
    )
    ap.add_argument(
        "--session-jobs",
        type=int,
        default=4,
        help="Number of session-bound jobs to submit (session_id=session-b)",
    )
    ap.add_argument(
        "--provider",
        type=str,
        default="copilot_cli",
        help="LLM provider (default: copilot_cli)",
    )
    ap.add_argument(
        "--transcript-jsonl",
        type=str,
        default="",
        help="Optional path to write JSONL transcript (in addition to chatty stdout)",
    )
    ap.add_argument(
        "--attempt-resume",
        action="store_true",
        help="Attempt a sticky resume/continue routing test (requires sticky_worker_id + matching session_id)",
    )

    args = ap.parse_args()

    n_workers = max(1, int(args.workers))
    jobs = max(1, int(args.jobs))
    conc = max(1, int(args.concurrency))
    timeout_s = max(5.0, float(args.timeout_s))
    provider = str(args.provider or "").strip().lower() or "copilot_cli"
    session_a = str(args.session_a or "").strip() or "S1"
    session_b = str(args.session_b or "").strip() or "S2"
    session_jobs = max(1, int(args.session_jobs))

    import shutil

    native_copilot_available = shutil.which("copilot") is not None

    py = _must_python()

    # Base env: keep everything local and deterministic.
    env_base = dict(os.environ)
    env_base["PYTHONUNBUFFERED"] = "1"
    env_base["IPFS_ACCEL_SKIP_CORE"] = env_base.get("IPFS_ACCEL_SKIP_CORE", "1")
    env_base["IPFS_ACCELERATE_PY_TASK_P2P_PUBLIC_IP"] = "127.0.0.1"
    env_base["IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS"] = "0"
    env_base["IPFS_ACCELERATE_PY_TASK_P2P_DHT"] = "0"
    env_base["IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS"] = "0"
    env_base["IPFS_ACCELERATE_PY_TASK_P2P_MDNS"] = "0"
    # Make the default provider deterministic for local runs (no real Copilot required).
    env_base.setdefault("ipfs_accelerate_py_COPILOT_CLI_CMD", 'bash -lc "echo OK"')
    # Allow copilot_cli execution on workers.
    env_base.setdefault("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_COPILOT_CLI", "1")
    # For smoketesting, allow continue_session without an explicit resume token.
    # (Still enforced by sticky_worker_id + session_id matching.)
    env_base.setdefault("IPFS_ACCELERATE_PY_TASK_WORKER_ALLOW_COPILOT_CONTINUE_WITHOUT_RESUME", "1")
    # Ensure workers advertise llm.generate.
    env_base.setdefault("IPFS_ACCELERATE_PY_TASK_WORKER_TASK_TYPES", "llm.generate")

    with tempfile.TemporaryDirectory(prefix="ipfs-accel-chatty-mesh-") as td:
        root = Path(td)

        # Peer A service (owns queue).
        a_queue = str(root / "peer_a.duckdb")
        a_announce = root / "peer_a_announce.json"
        a_port = _pick_free_port()
        env_a = dict(env_base)
        env_a["IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_HOST"] = "127.0.0.1"
        env_a["IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE"] = str(a_announce)

        svc_cmd = [
            py,
            "-m",
            "ipfs_accelerate_py.p2p_tasks.service",
            "--queue",
            a_queue,
            "--listen-port",
            str(a_port),
        ]
        svc = subprocess.Popen(svc_cmd, cwd=str(REPO_ROOT), env=env_a)

        drainers: list[subprocess.Popen] = []

        try:
            ann_a = _read_json(a_announce)
            remote_a = _remote_from_announce(ann_a)
            a_multiaddr = str(ann_a.get("multiaddr") or "").strip()

            print("=== chatty mesh inference smoketest ===")
            print(f"peerA multiaddr: {a_multiaddr}")
            print(f"provider: {provider}")
            print(f"workers: {n_workers}  jobs: {jobs}  concurrency: {conc}  timeout_s: {timeout_s}")

            # Start N mesh worker *processes* (not threads).
            # This avoids libp2p dial/handshake issues seen when running anyio/libp2p
            # request loops from multiple Python threads.
            env_b = dict(env_base)
            env_b["IPFS_ACCELERATE_PY_TASK_P2P_SESSION"] = session_b
            env_b["IPFS_ACCELERATE_PY_TASK_WORKER_MESH_PEERS"] = a_multiaddr
            env_b["IPFS_ACCELERATE_PY_TASK_WORKER_MESH_CLAIM_INTERVAL_S"] = "0.1"
            env_b["IPFS_ACCELERATE_PY_TASK_WORKER_MESH_PEER_FANOUT"] = "4"
            env_b["IPFS_ACCELERATE_PY_TASK_WORKER_MESH_CLAIM_BATCH"] = "4"

            drainer_worker_ids: list[str] = []
            for i in range(int(n_workers)):
                wid = f"drainer-w{i}-{uuid.uuid4().hex[:6]}"
                drainer_worker_ids.append(wid)
                # Each worker needs its own local DuckDB to avoid file lock contention.
                drainer_queue = str(root / f"drainer_{i}.duckdb")
                cmd = [
                    py,
                    "-m",
                    "ipfs_accelerate_py.p2p_tasks.worker",
                    "--queue",
                    drainer_queue,
                    "--worker-id",
                    wid,
                    "--mesh",
                ]
                drainers.append(subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env_b))

            # Phase 1: no-session throughput distribution.
            print("\n--- phase 1: throughput (no session_id) ---")
            no_session_tasks: list[tuple[int, str, str]] = []
            for j in range(jobs):
                prompt = f"Return exactly: OK (job {j+1}/{jobs})"
                payload = {
                    "provider": provider,
                    "prompt": prompt,
                    "chat_session_id": f"chatty-mesh-{uuid.uuid4().hex}",
                    "timeout": float(timeout_s),
                }
                tid = _submit_task(remote_a, payload=payload)
                no_session_tasks.append((j + 1, tid, prompt))

            # Phase 2: session-gated tasks should only execute on session_b worker.
            print("\n--- phase 2: session gating (session_id=session_b) ---")
            session_tasks: list[tuple[int, str, str]] = []
            for j in range(session_jobs):
                prompt = f"Return exactly: OK (session job {j+1}/{session_jobs})"
                payload = {
                    "provider": provider,
                    "prompt": prompt,
                    "chat_session_id": f"chatty-mesh-sess-{uuid.uuid4().hex}",
                    "session_id": session_b,
                    "timeout": float(timeout_s),
                }
                tid = _submit_task(remote_a, payload=payload)
                session_tasks.append((j + 1, tid, prompt))

            # Phase 2b: session mismatch should remain queued/unassigned.
            print("\n--- phase 2b: session mismatch (session_id=session_a should NOT drain) ---")
            mismatch_tasks: list[tuple[int, str, str]] = []
            for j in range(max(1, min(2, session_jobs))):
                prompt = f"Return exactly: OK (mismatch session job {j+1})"
                payload = {
                    "provider": provider,
                    "prompt": prompt,
                    "chat_session_id": f"chatty-mesh-mismatch-{uuid.uuid4().hex}",
                    "session_id": session_a,
                    "timeout": float(timeout_s),
                }
                tid = _submit_task(remote_a, payload=payload)
                mismatch_tasks.append((j + 1, tid, prompt))

            # Phase 3: optional sticky resume/continue routing test.
            resume_task: tuple[str, str] | None = None
            if bool(args.attempt_resume):
                mode = "native" if native_copilot_available else "sticky-only"
                print(f"\n--- phase 3: sticky resume/continue routing (session_id=session_b, mode={mode}) ---")
                prompt = "Return exactly: OK (resume round1)"
                payload = {
                    "provider": provider,
                    "prompt": prompt,
                    "chat_session_id": f"chatty-mesh-resume-{uuid.uuid4().hex}",
                    "session_id": session_b,
                    "timeout": float(timeout_s),
                }
                tid = _submit_task(remote_a, payload=payload)
                resume_task = (tid, prompt)

            # Wait all tasks and print chatty output.
            import anyio

            completed_no: list[Completed] = []
            completed_sess: list[Completed] = []
            completed_resume: list[Completed] = []

            sem = anyio.Semaphore(conc)

            async def _collect(which: str, idx: int, tid: str, prompt: str) -> None:
                async with sem:
                    t0 = time.monotonic()
                    try:
                        task = await _wait_task(remote_a, tid, timeout_s=timeout_s)
                    except Exception as exc:
                        task = None
                        task_exc = str(exc)
                    else:
                        task_exc = ""
                    dt_ms = int(max(0.0, (time.monotonic() - t0)) * 1000.0)
                    if not isinstance(task, dict):
                        c = Completed(
                            task_id=tid,
                            status="missing",
                            error=task_exc or "missing",
                            prompt=prompt,
                            text="",
                            elapsed_ms=dt_ms,
                            meta={},
                        )
                    else:
                        status = str(task.get("status") or "") or "unknown"
                        err = str(task.get("error") or "")
                        res = task.get("result")
                        meta = dict(res) if isinstance(res, dict) else {}
                        text = str(meta.get("text") or "")
                        c = Completed(task_id=tid, status=status, error=err, prompt=prompt, text=text, elapsed_ms=dt_ms, meta=meta)

                    print(_render_chat(c=c, idx=idx))

                    if which == "no":
                        completed_no.append(c)
                    elif which == "sess":
                        completed_sess.append(c)
                    else:
                        completed_resume.append(c)

            async def _run_all() -> None:
                async with anyio.create_task_group() as tg:
                    for idx, tid, prompt in no_session_tasks:
                        tg.start_soon(_collect, "no", idx, tid, prompt)
                    for idx, tid, prompt in session_tasks:
                        tg.start_soon(_collect, "sess", idx, tid, prompt)
                    if resume_task is not None:
                        tid, prompt = resume_task
                        tg.start_soon(_collect, "resume", jobs + session_jobs + 1, tid, prompt)

            anyio.run(_run_all, backend="trio")

            # If requested, submit round2 pinned to round1's executor.
            expected_resume_worker = ""
            if resume_task is not None and completed_resume:
                expected_resume_worker = str((completed_resume[0].meta or {}).get("executor_worker_id") or "").strip()
                if not expected_resume_worker:
                    print("FAILED: resume round1 missing executor_worker_id")
                    return 2

                prompt2 = "Return exactly: OK (resume round2)"
                payload2 = {
                    "provider": provider,
                    "prompt": prompt2,
                    "chat_session_id": str((completed_resume[0].meta or {}).get("chat_session_id") or "")
                    or f"chatty-mesh-resume2-{uuid.uuid4().hex}",
                    "session_id": session_b,
                    "sticky_worker_id": expected_resume_worker,
                    "timeout": float(timeout_s),
                }
                if native_copilot_available:
                    payload2["continue_session"] = True
                tid2 = _submit_task(remote_a, payload=payload2)

                async def _wait_round2() -> None:
                    await _collect("resume", jobs + session_jobs + 2, tid2, prompt2)

                anyio.run(_wait_round2, backend="trio")

            # Write JSONL transcript if requested.
            transcript_path = str(args.transcript_jsonl or "").strip()
            if transcript_path:
                out_path = Path(transcript_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                run_id = f"chatty-mesh-{uuid.uuid4().hex}"
                with out_path.open("w", encoding="utf-8") as f:
                    for c in list(completed_no) + list(completed_sess) + list(completed_resume):
                        meta = dict(c.meta or {})
                        record = {
                            "run_id": run_id,
                            "task_id": c.task_id,
                            "status": c.status,
                            "error": c.error,
                            "elapsed_ms": int(c.elapsed_ms),
                            "prompt": c.prompt,
                            "text": c.text,
                            "executor_worker_id": str(meta.get("executor_worker_id") or ""),
                            "executor_peer_id": str(meta.get("executor_peer_id") or ""),
                            "executor_multiaddr": str(meta.get("executor_multiaddr") or ""),
                            "session_id": str(meta.get("session_id") or ""),
                            "chat_session_id": str(meta.get("chat_session_id") or ""),
                            "resume_session_id": str(meta.get("resume_session_id") or ""),
                        }
                        f.write(json.dumps(record, sort_keys=True) + "\n")

            # Expectations.
            # Throughput: at least 2 distinct executor worker ids in phase 1.
            ex_no = {str((c.meta or {}).get("executor_worker_id") or "") for c in completed_no}
            ex_no.discard("")

            # Session gating: session_b tasks should complete (drainer runs session_b).
            ok_sess = all(str((c.meta or {}).get("session_id") or "").strip() == session_b for c in completed_sess)

            print("\n--- expectations ---")
            print(f"phase1 distinct executors: {sorted(ex_no) or ['(none)']}")
            print(f"phase2 session_b={session_b} ok={ok_sess}")

            ok_completed = all(c.status == "completed" for c in completed_no) and all(c.status == "completed" for c in completed_sess)

            expected_distinct = 2 if (n_workers >= 2 and jobs >= 2) else 1
            ok_dist = len(ex_no) >= expected_distinct
            if not ok_completed:
                print("FAILED: expected all tasks to reach status=completed")
                return 2
            if not ok_dist:
                print(f"FAILED: expected throughput distribution across >= {expected_distinct} worker(s)")
                return 2
            if not ok_sess:
                print("FAILED: expected session-bound tasks to be executed with session_id=session_b")
                return 2

            if expected_resume_worker and len(completed_resume) >= 2:
                got = str((completed_resume[-1].meta or {}).get("executor_worker_id") or "").strip()
                st2 = str(completed_resume[-1].status or "").strip().lower()
                if st2 != "completed":
                    print("FAILED: sticky resume round2 did not complete")
                    print(f"expected status=completed got status={st2!r}")
                    return 2
                if not got:
                    print("FAILED: sticky resume round2 missing executor_worker_id")
                    return 2
                if got != expected_resume_worker:
                    print("FAILED: sticky resume ran on different worker")
                    print(f"expected executor_worker_id={expected_resume_worker} got={got}")
                    return 2
            elif bool(args.attempt_resume):
                # If resume was requested, require that both round1 and round2 were executed.
                if not expected_resume_worker:
                    print("FAILED: sticky resume round1 did not produce executor_worker_id")
                    return 2
                if len(completed_resume) < 2:
                    print("FAILED: sticky resume round2 did not run")
                    return 2

            # Mismatch tasks must remain queued/unassigned.
            for _idx, tid, _prompt in mismatch_tasks:
                try:
                    import anyio

                    async def _get() -> Optional[dict]:
                        sys.path.insert(0, str(REPO_ROOT))
                        from ipfs_accelerate_py.p2p_tasks.client import get_task  # noqa: E402

                        return await get_task(remote=remote_a, task_id=str(tid))

                    t = anyio.run(_get, backend="trio")
                except Exception:
                    t = None
                t = t if isinstance(t, dict) else {}
                st = str(t.get("status") or "").strip().lower()
                aw = str(t.get("assigned_worker") or "").strip()
                if st != "queued" or aw:
                    print(f"FAILED: mismatch session task drained unexpectedly: task_id={tid} status={st} assigned_worker={aw!r}")
                    return 2

            print("\nPASS")
            return 0
        finally:
            for i, proc in enumerate(list(drainers)):
                _kill(proc, name=f"drainer[{i}]")
            _kill(svc, name="service")


if __name__ == "__main__":
    raise SystemExit(main())
