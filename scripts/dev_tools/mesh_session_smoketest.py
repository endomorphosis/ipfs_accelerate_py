#!/usr/bin/env python
"""2-peer mesh/session-affinity smoke test.

This script runs a minimal local simulation of a LAN mesh:
- Peer A: runs the libp2p TaskQueue RPC service and owns the queue we submit to.
- Peer B: runs the worker in mesh mode (static peer list) and attempts to drain
  work from Peer A.

We submit three tasks to Peer A:
1) required session = S1 (should NOT be drained by peer B, which uses S2)
2) required session = S2 (should be drained by peer B)
3) no session requirement (should be drainable by peer B)

The test passes when:
- the S1 task remains queued on A
- the S2 task is no longer queued (completed or failed)
- the no-session task is no longer queued (completed or failed)

Notes:
- For `llm.generate` tasks, execution depends on external Copilot CLI tooling.
  The script treats both completed and failed as “drained”; the key property
  under test is session-affinity (no cross-session draining).

Usage:
  ./.venv/bin/python scripts/dev_tools/mesh_session_smoketest.py

Optional env:
  IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_COPILOT_CLI=1   (to actually run llm.generate)
  IPFS_ACCELERATE_PY_TASK_WORKER_MESH_CLAIM_INTERVAL_S=0.2
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PY = REPO_ROOT / ".venv" / "bin" / "python"


def _must_python() -> str:
    py = str(VENV_PY)
    if os.path.exists(py):
        return py
    return sys.executable


def _read_announce(path: Path, *, timeout_s: float = 20.0) -> dict:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        try:
            if path.exists() and path.stat().st_size > 0:
                return json.loads(path.read_text("utf-8"))
        except Exception:
            pass
        time.sleep(0.1)
    raise RuntimeError(f"timed out waiting for announce file: {path}")


def _remote_from_announce(ann: dict):
    sys.path.insert(0, str(REPO_ROOT))
    from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue  # noqa: E402

    peer_id = str(ann.get("peer_id") or "").strip()
    multiaddr = str(ann.get("multiaddr") or "").strip()
    return RemoteQueue(peer_id=peer_id, multiaddr=multiaddr)


def _submit_tasks(remote, *, model_name: str = "") -> tuple[str, str, str]:
    # Submit via RPC to avoid opening DuckDB directly (service holds the lock).
    sys.path.insert(0, str(REPO_ROOT))
    from ipfs_accelerate_py.p2p_tasks.client import submit_task_sync  # noqa: E402

    # Use llm.generate because it’s the Copilot CLI mesh path.
    tid_s1 = submit_task_sync(
        remote=remote,
        task_type="llm.generate",
        model_name=model_name,
        payload={"prompt": "say hello", "provider": "copilot_cli", "session_id": "S1"},
    )
    tid_s2 = submit_task_sync(
        remote=remote,
        task_type="llm.generate",
        model_name=model_name,
        payload={"prompt": "say hi", "provider": "copilot_cli", "session_id": "S2"},
    )
    tid_none = submit_task_sync(
        remote=remote,
        task_type="llm.generate",
        model_name=model_name,
        payload={"prompt": "say hey", "provider": "copilot_cli"},
    )

    return (tid_s1, tid_s2, tid_none)


def _get_task_status(remote, task_id: str) -> str:
    import anyio

    sys.path.insert(0, str(REPO_ROOT))
    from ipfs_accelerate_py.p2p_tasks.client import get_task  # noqa: E402

    async def _do():
        return await get_task(remote=remote, task_id=task_id)

    task = anyio.run(_do, backend="trio")
    if not isinstance(task, dict):
        return "missing"
    return str(task.get("status") or "") or "unknown"


def _wait_task_done(remote, task_id: str, *, timeout_s: float = 60.0) -> str:
    # Wait until the task is completed/failed, or timeout.
    import anyio

    sys.path.insert(0, str(REPO_ROOT))
    from ipfs_accelerate_py.p2p_tasks.client import wait_task  # noqa: E402

    async def _do():
        return await wait_task(remote=remote, task_id=task_id, timeout_s=float(timeout_s))

    task = anyio.run(_do, backend="trio")
    if not isinstance(task, dict):
        return "queued"
    return str(task.get("status") or "") or "unknown"


def _poll_task_is_queued(remote, task_id: str, *, timeout_s: float = 10.0) -> bool:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        status = _get_task_status(remote, task_id)
        if status == "queued":
            return True
        time.sleep(0.2)
    return False


def _kill_process(proc: subprocess.Popen, *, name: str) -> None:
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


def main() -> int:
    py = _must_python()

    with tempfile.TemporaryDirectory(prefix="ipfs-accel-mesh-smoke-") as td:
        root = Path(td)
        a_queue = str(root / "peer_a.duckdb")
        b_queue = str(root / "peer_b.duckdb")
        a_announce = root / "peer_a_announce.json"

        listen_host = "127.0.0.1"
        listen_port = 19711

        env_base = dict(os.environ)
        env_base["PYTHONUNBUFFERED"] = "1"
        env_base["IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_HOST"] = listen_host
        env_base["IPFS_ACCELERATE_PY_TASK_P2P_PUBLIC_IP"] = listen_host
        env_base["IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE"] = str(a_announce)

        # Peer A service.
        svc_cmd = [
            py,
            "-m",
            "ipfs_accelerate_py.p2p_tasks.service",
            "--queue",
            a_queue,
            "--listen-port",
            str(listen_port),
        ]
        svc = subprocess.Popen(
            svc_cmd,
            cwd=str(REPO_ROOT),
            env=env_base,
        )

        worker = None
        try:
            ann = _read_announce(a_announce)
            multiaddr = str(ann.get("multiaddr") or "").strip()
            if not multiaddr or "/p2p/" not in multiaddr:
                raise RuntimeError(f"invalid multiaddr in announce: {ann!r}")

            remote_a = _remote_from_announce(ann)

            # Submit mixed-session tasks onto peer A queue.
            tid_s1, tid_s2, tid_none = _submit_tasks(remote_a)

            # Peer B worker (mesh, static peer list, session=S2).
            env_b = dict(env_base)
            env_b["IPFS_ACCELERATE_PY_TASK_P2P_SESSION"] = "S2"
            env_b["IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_COPILOT_CLI"] = env_base.get(
                "IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_COPILOT_CLI", "0"
            )
            # Ensure worker actually advertises llm.generate for claiming.
            env_b["IPFS_ACCELERATE_PY_TASK_WORKER_TASK_TYPES"] = "llm.generate,llm_generate"
            env_b["IPFS_ACCELERATE_PY_TASK_WORKER_MESH_PEERS"] = multiaddr
            env_b["IPFS_ACCELERATE_PY_TASK_WORKER_MESH_CLAIM_INTERVAL_S"] = env_base.get(
                "IPFS_ACCELERATE_PY_TASK_WORKER_MESH_CLAIM_INTERVAL_S", "0.2"
            )

            worker_cmd = [
                py,
                "-m",
                "ipfs_accelerate_py.p2p_tasks.worker",
                "--queue",
                b_queue,
                "--worker-id",
                "peer-b",
                "--mesh",
                "--poll-interval-s",
                "0.2",
            ]
            worker = subprocess.Popen(
                worker_cmd,
                cwd=str(REPO_ROOT),
                env=env_b,
            )

            # Wait for drain of eligible tasks.
            s2_status = _wait_task_done(remote_a, tid_s2, timeout_s=60.0)
            none_status = _wait_task_done(remote_a, tid_none, timeout_s=60.0)

            # Confirm mismatched stays queued.
            s1_queued = _poll_task_is_queued(remote_a, tid_s1, timeout_s=5.0)

            # Print a small summary.
            print("\n=== mesh session smoke test ===")
            print(f"peerA multiaddr: {multiaddr}")
            print(f"task S1 (mismatch) id={tid_s1} queued={s1_queued}")
            print(f"task S2 (match)    id={tid_s2} status={s2_status}")
            print(f"task NONE          id={tid_none} status={none_status}")

            ok = bool(s1_queued) and (s2_status != "queued") and (none_status != "queued")
            if not ok:
                print("\nFAILED: expected S1 to remain queued, and S2/NONE to be drained")
                return 2

            print("\nPASS")
            return 0
        finally:
            if worker is not None:
                _kill_process(worker, name="worker")
            _kill_process(svc, name="service")


if __name__ == "__main__":
    raise SystemExit(main())
