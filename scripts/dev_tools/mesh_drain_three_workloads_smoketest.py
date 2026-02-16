#!/usr/bin/env python
"""Mesh draining smoketest for: docker.execute, gpt2 text-generation, and copilot_cli llm.generate.

This script runs a minimal local 2-peer simulation:
- Peer A: runs the libp2p TaskQueue RPC service and owns the queue.
- Peer B: runs N mesh workers that drain work from Peer A.

It submits 3 tasks to Peer A and asserts they were executed by a mesh worker:
1) docker.execute (Docker Hub container run)
2) text-generation with model_name=gpt2
3) llm.generate with provider=copilot_cli

Exit codes
- 0: PASS
- 2: FAIL (task failed or executed by unexpected worker)
- 3: invalid inputs / missing prerequisites

Prereqs
- docker.execute requires Docker on the host running the mesh workers:
    IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_DOCKER=1 and a working Docker daemon.
- llm.generate uses a deterministic fallback command by default:
    ipfs_accelerate_py_COPILOT_CLI_CMD='bash -lc "echo OK"'
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PY = REPO_ROOT / ".venv" / "bin" / "python"


@dataclass
class TaskOutcome:
    label: str
    task_id: str
    status: str
    ok: bool
    executor_worker_id: str
    error: str
    raw: Dict[str, Any]


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


def _remote_from_announce(ann: dict):
    import sys as _sys

    _sys.path.insert(0, str(REPO_ROOT))
    from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue  # noqa: E402

    peer_id = str(ann.get("peer_id") or "").strip()
    multiaddr = str(ann.get("multiaddr") or "").strip()
    if not peer_id or not multiaddr:
        raise ValueError("announce missing peer_id/multiaddr")
    return RemoteQueue(peer_id=peer_id, multiaddr=multiaddr)


async def _submit_and_wait(remote, *, task_type: str, model_name: str, payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    from ipfs_accelerate_py.p2p_tasks.client import submit_task, wait_task

    tid = await submit_task(remote=remote, task_type=str(task_type), model_name=str(model_name), payload=payload)
    task = await wait_task(remote=remote, task_id=str(tid), timeout_s=float(timeout_s))
    if task is None:
        return {"task_id": tid, "status": "timeout"}
    if isinstance(task, dict):
        task.setdefault("task_id", tid)
        return task
    return {"task_id": tid, "status": "invalid_response", "raw": task}


def _extract_executor_worker_id(task: Dict[str, Any]) -> str:
    # Prefer structured metadata in task.result.
    result = task.get("result")
    if isinstance(result, dict):
        wid = str(result.get("executor_worker_id") or "").strip()
        if wid:
            return wid
    # Fall back to assigned_worker if present.
    return str(task.get("assigned_worker") or "").strip()


def _extract_error(task: Dict[str, Any]) -> str:
    err = str(task.get("error") or "").strip()
    if err:
        return err
    res = task.get("result")
    if isinstance(res, dict):
        for k in ("error", "error_message", "stderr"):
            v = res.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Mesh drain smoketest for docker.execute + gpt2 + copilot_cli")
    ap.add_argument("--workers", type=int, default=2, help="Number of mesh workers to run")
    ap.add_argument("--timeout-s", type=float, default=180.0, help="Per-task wait timeout")
    ap.add_argument("--skip-docker", action="store_true", help="Skip docker.execute task")
    ap.add_argument("--docker-image", type=str, default="alpine:3.20", help="Docker image for docker.execute")
    ap.add_argument("--docker-cmd", type=str, default="echo OK", help="Docker command (space-separated) for docker.execute")
    ap.add_argument("--prompt", type=str, default="Return exactly: OK", help="Prompt for generation tasks")

    args = ap.parse_args()

    n_workers = max(1, int(args.workers))
    timeout_s = max(10.0, float(args.timeout_s))

    py = _must_python()

    # Keep everything local and deterministic.
    env_base = dict(os.environ)
    env_base["PYTHONUNBUFFERED"] = "1"
    env_base.setdefault("IPFS_ACCEL_SKIP_CORE", "1")
    env_base.setdefault("IPFS_KIT_DISABLE", "1")
    env_base.setdefault("STORAGE_FORCE_LOCAL", "1")
    env_base.setdefault("TRANSFORMERS_PATCH_DISABLE", "1")

    # Isolate local simulation from external peer discovery.
    env_base["IPFS_ACCELERATE_PY_TASK_P2P_PUBLIC_IP"] = "127.0.0.1"
    env_base["IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS"] = "0"
    env_base["IPFS_ACCELERATE_PY_TASK_P2P_DHT"] = "0"
    env_base["IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS"] = "0"
    env_base["IPFS_ACCELERATE_PY_TASK_P2P_MDNS"] = "0"

    # Text-gen: prefer minimal HF path for test stability.
    env_base.setdefault("IPFS_ACCELERATE_PY_TASK_WORKER_MINIMAL_LLM", "1")

    # LLM mesh provider: deterministic fallback (no real Copilot required).
    env_base.setdefault("ipfs_accelerate_py_COPILOT_CLI_CMD", 'bash -lc "echo OK"')
    env_base.setdefault("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_COPILOT_CLI", "1")

    # Docker.
    env_base.setdefault("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_DOCKER", "1")

    with tempfile.TemporaryDirectory(prefix="ipfs-accel-mesh-3workloads-") as td:
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
        worker_ids: list[str] = []

        try:
            ann = _read_json(a_announce)
            remote_a = _remote_from_announce(ann)
            a_multiaddr = str(ann.get("multiaddr") or "").strip()
            if not a_multiaddr:
                print("ERROR: missing multiaddr from announce")
                return 3

            # Peer B mesh workers.
            env_b = dict(env_base)
            env_b["IPFS_ACCELERATE_PY_TASK_WORKER_MESH_PEERS"] = a_multiaddr
            env_b["IPFS_ACCELERATE_PY_TASK_WORKER_MESH_CLAIM_INTERVAL_S"] = "0.1"
            env_b["IPFS_ACCELERATE_PY_TASK_WORKER_MESH_PEER_FANOUT"] = "4"
            env_b["IPFS_ACCELERATE_PY_TASK_WORKER_MESH_CLAIM_BATCH"] = "4"

            # Explicit allowlist so we only claim what we're testing.
            env_b["IPFS_ACCELERATE_PY_TASK_WORKER_TASK_TYPES"] = "docker.execute,text-generation,llm.generate"

            for i in range(n_workers):
                wid = f"drainer-w{i}-{uuid.uuid4().hex[:6]}"
                worker_ids.append(wid)
                w_queue = str(root / f"drainer_{i}.duckdb")
                cmd = [
                    py,
                    "-m",
                    "ipfs_accelerate_py.p2p_tasks.worker",
                    "--queue",
                    w_queue,
                    "--worker-id",
                    wid,
                    "--mesh",
                ]
                drainers.append(subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env_b))

            print("=== mesh drain 3-workloads smoketest ===")
            print(f"peerA multiaddr: {a_multiaddr}")
            print(f"mesh workers: {worker_ids}")

            import anyio

            outcomes: list[TaskOutcome] = []

            async def _run_all() -> None:
                # 1) docker.execute
                if not bool(args.skip_docker):
                    docker_payload: Dict[str, Any] = {
                        "image": str(args.docker_image),
                        "command": [str(x) for x in str(args.docker_cmd).split() if str(x).strip()],
                        "network_mode": "none",
                        "timeout": int(min(120, int(timeout_s))),
                    }
                    docker_task = await _submit_and_wait(
                        remote_a,
                        task_type="docker.execute",
                        model_name="docker",
                        payload=docker_payload,
                        timeout_s=timeout_s,
                    )
                    outcomes.append(
                        TaskOutcome(
                            label="docker.execute",
                            task_id=str(docker_task.get("task_id") or ""),
                            status=str(docker_task.get("status") or ""),
                            ok=str(docker_task.get("status") or "") == "completed",
                            executor_worker_id=_extract_executor_worker_id(docker_task),
                            error=_extract_error(docker_task),
                            raw=(docker_task if isinstance(docker_task, dict) else {}),
                        )
                    )

                # 2) gpt2 text-generation
                tg_payload: Dict[str, Any] = {
                    "prompt": str(args.prompt),
                    "max_new_tokens": 8,
                    "temperature": 0.2,
                }
                tg_task = await _submit_and_wait(
                    remote_a,
                    task_type="text-generation",
                    model_name="gpt2",
                    payload=tg_payload,
                    timeout_s=timeout_s,
                )
                outcomes.append(
                    TaskOutcome(
                        label="text-generation(gpt2)",
                        task_id=str(tg_task.get("task_id") or ""),
                        status=str(tg_task.get("status") or ""),
                        ok=str(tg_task.get("status") or "") == "completed",
                        executor_worker_id=_extract_executor_worker_id(tg_task),
                        error=_extract_error(tg_task),
                        raw=(tg_task if isinstance(tg_task, dict) else {}),
                    )
                )

                # 3) copilot_cli llm.generate
                llm_payload: Dict[str, Any] = {
                    "provider": "copilot_cli",
                    "prompt": str(args.prompt),
                    "chat_session_id": f"mesh-3workloads-{uuid.uuid4().hex}",
                    "timeout": float(timeout_s),
                }
                llm_task = await _submit_and_wait(
                    remote_a,
                    task_type="llm.generate",
                    model_name="",
                    payload=llm_payload,
                    timeout_s=timeout_s,
                )
                outcomes.append(
                    TaskOutcome(
                        label="llm.generate(copilot_cli)",
                        task_id=str(llm_task.get("task_id") or ""),
                        status=str(llm_task.get("status") or ""),
                        ok=str(llm_task.get("status") or "") == "completed",
                        executor_worker_id=_extract_executor_worker_id(llm_task),
                        error=_extract_error(llm_task),
                        raw=(llm_task if isinstance(llm_task, dict) else {}),
                    )
                )

            anyio.run(_run_all, backend="trio")

            failed = False
            for o in outcomes:
                ran_on_mesh = bool(o.executor_worker_id and o.executor_worker_id in set(worker_ids))
                print(
                    f"- {o.label}: status={o.status} executor_worker_id={o.executor_worker_id or '(missing)'} mesh={ran_on_mesh}"
                )
                if (not o.ok) and o.error:
                    print(f"  error: {o.error}")

                # Must complete successfully.
                if not o.ok:
                    failed = True
                # Must have been executed by a mesh worker.
                if not ran_on_mesh:
                    failed = True

            return 0 if not failed else 2

        finally:
            for p in drainers:
                _kill(p, name="drainer")
            _kill(svc, name="service")


if __name__ == "__main__":
    raise SystemExit(main())
