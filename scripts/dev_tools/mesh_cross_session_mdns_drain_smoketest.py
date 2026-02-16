#!/usr/bin/env python
"""Smoketest: cross-session mDNS discovery + session-gated draining.

Goal
- Prove that a mesh worker in session "Y" can discover (via mDNS) and drain
  tasks from a peer service whose own session tag is "X", as long as the task
  payload requires session "Y".

This specifically guards against regressions where mesh peer discovery filters
peers by the remote service's session tag, which would prevent draining tasks
that were enqueued on the "wrong" machine.

Topology (single host)
- Peer A: libp2p TaskQueue RPC service, session=X, queue owner.
- Worker B: mesh worker, session=Y, mDNS discovery enabled.

The test submits a session=Y llm.generate task *to Peer A* and asserts:
- It completes successfully.
- executor_worker_id matches Worker B.

Exit codes
- 0: PASS
- 2: FAIL (task failed / wrong executor / timed out)
- 3: invalid inputs / missing prerequisites

Notes
- This test uses provider=copilot_cli with a deterministic command by default
  (no real Copilot auth required): ipfs_accelerate_py_COPILOT_CLI_CMD.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict


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
        return


def _extract_executor_worker_id(task: Dict[str, Any]) -> str:
    result = task.get("result")
    if isinstance(result, dict):
        wid = str(result.get("executor_worker_id") or "").strip()
        if wid:
            return wid
    return str(task.get("assigned_worker") or "").strip()


def _wait_task_sync(remote, *, task_id: str, timeout_s: float) -> Dict[str, Any] | None:
    import anyio

    from ipfs_accelerate_py.p2p_tasks.client import wait_task

    async def _run() -> Dict[str, Any] | None:
        task = await wait_task(remote=remote, task_id=str(task_id), timeout_s=float(timeout_s))
        return task if isinstance(task, dict) else None

    return anyio.run(_run, backend="trio")


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoketest: cross-session mDNS drain")
    ap.add_argument("--timeout-s", type=float, default=90.0, help="Total time to wait for completion")
    ap.add_argument("--session-x", type=str, default="session_x", help="Service (queue owner) session tag")
    ap.add_argument("--session-y", type=str, default="session_y", help="Worker session tag")
    ap.add_argument("--prompt", type=str, default="Return exactly: OK", help="Prompt for llm.generate")
    ap.add_argument(
        "--copilot-cmd",
        type=str,
        default='bash -lc "echo OK"',
        help="Command template for copilot_cli provider (no real Copilot needed)",
    )

    args = ap.parse_args()

    timeout_s = max(10.0, float(args.timeout_s))
    session_x = str(args.session_x or "").strip()
    session_y = str(args.session_y or "").strip()
    if not session_x or not session_y:
        print("ERROR: session tags must be non-empty")
        return 3
    if session_x == session_y:
        print("ERROR: session-x and session-y must differ for this smoketest")
        return 3

    py = _must_python()

    # Base env (keep deterministic + local).
    env_base = dict(os.environ)
    env_base["PYTHONUNBUFFERED"] = "1"
    env_base.setdefault("IPFS_ACCEL_SKIP_CORE", "1")
    env_base.setdefault("IPFS_KIT_DISABLE", "1")
    env_base.setdefault("STORAGE_FORCE_LOCAL", "1")
    env_base.setdefault("TRANSFORMERS_PATCH_DISABLE", "1")

    # Make sure mDNS is enabled for this test.
    env_base["IPFS_ACCELERATE_PY_TASK_P2P_MDNS"] = "1"
    env_base["IPFS_ACCELERATE_PY_TASK_P2P_DHT"] = "0"
    env_base["IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS"] = "0"
    env_base["IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS"] = "0"

    # LLM execution: deterministic stub.
    env_base["IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_COPILOT_CLI"] = "1"
    env_base["ipfs_accelerate_py_COPILOT_CLI_CMD"] = str(args.copilot_cmd)

    # Keep worker simple.
    env_base.setdefault("IPFS_ACCELERATE_PY_TASK_WORKER_MINIMAL_LLM", "1")

    with tempfile.TemporaryDirectory(prefix="ipfs-accel-mesh-cross-session-") as td:
        root = Path(td)

        a_queue = str(root / "peer_a.duckdb")
        a_announce = root / "peer_a_announce.json"
        a_port = _pick_free_port()

        env_a = dict(env_base)
        env_a["IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_HOST"] = "127.0.0.1"
        env_a["IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE"] = str(a_announce)
        env_a["IPFS_ACCELERATE_PY_TASK_P2P_SESSION"] = session_x

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
        worker = None

        try:
            _ = _read_json(a_announce, timeout_s=30.0)

            wid = f"cross-session-drainer-{uuid.uuid4().hex[:8]}"
            b_queue = str(root / "worker_b.duckdb")

            env_b = dict(env_base)
            env_b["IPFS_ACCELERATE_PY_TASK_P2P_SESSION"] = session_y

            # Crucially, do NOT set IPFS_ACCELERATE_PY_TASK_WORKER_MESH_PEERS.
            # We want mDNS peer discovery.
            env_b["IPFS_ACCELERATE_PY_TASK_WORKER_MESH_MODE"] = "1"
            env_b["IPFS_ACCELERATE_PY_TASK_WORKER_MESH_MAX_PEERS"] = "20"
            env_b["IPFS_ACCELERATE_PY_TASK_WORKER_MESH_CLAIM_INTERVAL_S"] = "0.1"
            env_b["IPFS_ACCELERATE_PY_TASK_WORKER_TASK_TYPES"] = "llm.generate"

            worker_cmd = [
                py,
                "-m",
                "ipfs_accelerate_py.p2p_tasks.worker",
                "--queue",
                b_queue,
                "--worker-id",
                wid,
                "--mesh",
            ]

            worker = subprocess.Popen(worker_cmd, cwd=str(REPO_ROOT), env=env_b)

            # Submit to Peer A directly via RemoteQueue.
            sys.path.insert(0, str(REPO_ROOT))
            from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue  # noqa: E402
            from ipfs_accelerate_py.p2p_tasks.client import submit_task_sync  # noqa: E402

            ann = _read_json(a_announce, timeout_s=10.0)
            remote_a = RemoteQueue(peer_id=str(ann.get("peer_id") or "").strip(), multiaddr=str(ann.get("multiaddr") or "").strip())
            if not remote_a.peer_id or not remote_a.multiaddr:
                print(f"ERROR: invalid announce: {ann}")
                return 3

            payload = {
                "prompt": str(args.prompt),
                "provider": "copilot_cli",
                "session_id": session_y,
                "chat_session_id": f"chat-{uuid.uuid4().hex}",
            }

            task_id = submit_task_sync(remote=remote_a, task_type="llm.generate", model_name="gpt2", payload=payload)
            deadline = time.time() + timeout_s
            task = None
            while time.time() < deadline:
                task = _wait_task_sync(remote_a, task_id=str(task_id), timeout_s=2.0)
                if isinstance(task, dict) and str(task.get("status") or "") in {"completed", "failed"}:
                    break
                time.sleep(0.1)

            if not isinstance(task, dict):
                print("FAIL: did not receive task dict")
                return 2

            status = str(task.get("status") or "")
            if status != "completed":
                print(f"FAIL: status={status} task={task}")
                return 2

            executor = _extract_executor_worker_id(task)
            if executor != wid:
                print("FAIL: wrong executor worker")
                print(f"  expected: {wid}")
                print(f"  got:      {executor}")
                print(f"  task:     {task}")
                return 2

            print("PASS")
            print(f"  service_session: {session_x}")
            print(f"  worker_session:  {session_y}")
            print(f"  executor:        {executor}")
            return 0

        finally:
            if worker is not None:
                _kill(worker, name="worker")
            _kill(svc, name="service")


if __name__ == "__main__":
    raise SystemExit(main())
