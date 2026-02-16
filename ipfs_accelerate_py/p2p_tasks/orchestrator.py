"""Task orchestration for MCP + TaskQueue p2p service.

This module implements the "server owns orchestration" model:
- The MCP process hosts the TaskQueue p2p service.
- A local orchestrator loop scales thin worker *processes* up/down based on
  backlog (local + discovered peers).
- Mesh draining is performed by the orchestrator: it claims tasks from peers
  and submits them into the local queue as proxy tasks.

Workers remain thin executors: they only run local queue tasks and (optionally)
complete remote tasks when executing a proxy payload.
"""

from __future__ import annotations

import os
import sys
import time
import socket
import threading
import subprocess
from dataclasses import dataclass
from typing import Any, Optional


def _truthy(raw: object) -> bool:
    try:
        return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}
    except Exception:
        return False


def _expected_session_tag() -> str:
    return str(os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_SESSION") or "").strip()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        return int(float(str(raw).strip()))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    try:
        return float(str(raw).strip())
    except Exception:
        return float(default)


def _default_orchestrator_id() -> str:
    return f"orch-{socket.gethostname()}"


def _default_base_worker_id() -> str:
    return f"worker-{socket.gethostname()}"


@dataclass
class OrchestratorConfig:
    queue_path: str
    orchestrator_id: str = ""
    base_worker_id: str = ""
    min_workers: int = 1
    max_workers: int = 4
    scale_poll_s: float = 2.0
    scale_down_idle_s: float = 30.0
    mesh_refresh_s: float = 5.0
    mesh_max_peers: int = 10
    mesh_claim_interval_s: float = 0.25
    mesh_peer_fanout: int = 2
    mesh_claim_batch: int = 8
    remote_status_timeout_s: float = 3.0
    remote_task_ttl_s: float = 600.0


class TaskOrchestrator:
    def __init__(
        self,
        *,
        config: OrchestratorConfig,
        accelerate_instance: object | None = None,
        supported_task_types: Optional[list[str]] = None,
    ) -> None:
        self._cfg = config
        self._accelerate = accelerate_instance
        self._supported_task_types = supported_task_types

        if not self._cfg.orchestrator_id:
            self._cfg.orchestrator_id = _default_orchestrator_id()
        if not self._cfg.base_worker_id:
            self._cfg.base_worker_id = _default_base_worker_id()

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self._lock = threading.RLock()

        @dataclass
        class _WorkerHandle:
            worker_id: str
            started_ts: float
            kind: str  # "process" | "thread"
            proc: subprocess.Popen[object] | None = None
            thread: threading.Thread | None = None
            stop_event: threading.Event | None = None

        self._WorkerHandle = _WorkerHandle  # type: ignore[assignment]
        self._workers: list[_WorkerHandle] = []
        self._last_nonzero_ts = 0.0

        # inflight remote tasks: remote_task_id -> {remote, local_task_id, ts}
        self._inflight: dict[str, dict[str, object]] = {}

        self._peers_lock = threading.RLock()
        self._peers: list[object] = []
        self._peers_thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        t = self._thread
        return bool(t and t.is_alive())

    def start(self) -> None:
        if self.running:
            return

        self._stop.clear()
        self._start_peer_discovery()
        t = threading.Thread(target=self._run, name="ipfs_accelerate_py_task_orchestrator", daemon=True)
        self._thread = t
        t.start()

    def stop(self, *, timeout_s: float = 2.0) -> None:
        self._stop.set()
        try:
            if self._thread:
                self._thread.join(timeout=max(0.1, float(timeout_s)))
        except Exception:
            pass
        self._stop_all_workers()

    def _start_peer_discovery(self) -> None:
        if self._peers_thread and self._peers_thread.is_alive():
            return

        expected_session = _expected_session_tag()

        def _loop() -> None:
            try:
                from ipfs_accelerate_py.p2p_tasks.client import discover_peers_via_mdns_sync, request_status_sync
            except Exception:
                return

            while not self._stop.is_set():
                peers: list[object] = []
                try:
                    discovered = discover_peers_via_mdns_sync(
                        timeout_s=1.0,
                        limit=int(max(1, self._cfg.mesh_max_peers)),
                        exclude_self=True,
                    )
                except Exception:
                    discovered = []

                for rq in list(discovered or []):
                    try:
                        pid = str(getattr(rq, "peer_id", "") or "").strip()
                        ma = str(getattr(rq, "multiaddr", "") or "").strip()
                    except Exception:
                        continue
                    if not pid or not ma:
                        continue

                    if expected_session:
                        try:
                            resp = request_status_sync(
                                remote=rq,
                                timeout_s=float(self._cfg.remote_status_timeout_s),
                                detail=False,
                            )
                            if not (isinstance(resp, dict) and resp.get("ok")):
                                continue
                            if str(resp.get("session") or "").strip() != expected_session:
                                continue
                        except Exception:
                            continue

                    peers.append(rq)

                with self._peers_lock:
                    self._peers = peers

                self._stop.wait(max(0.2, float(self._cfg.mesh_refresh_s)))

        t = threading.Thread(target=_loop, name="ipfs_accelerate_py_task_orchestrator_mdns", daemon=True)
        self._peers_thread = t
        t.start()

    def _snapshot_peers(self) -> list[object]:
        with self._peers_lock:
            return list(self._peers)

    def _compute_supported_task_types(self) -> list[str]:
        if isinstance(self._supported_task_types, list) and self._supported_task_types:
            return [str(x).strip() for x in self._supported_task_types if str(x).strip()]

        # Reuse the worker's capability computation (single source of truth).
        from ipfs_accelerate_py.p2p_tasks.worker import _compute_supported_task_types  # type: ignore

        return _compute_supported_task_types(
            supported_task_types=None,
            accelerate_instance=self._accelerate,
        )

    def _spawn_worker(self, *, idx: int) -> None:
        wid = f"{self._cfg.base_worker_id}-w{int(idx)}-{int(time.time()*1000) % 1000000:06d}"

        # DuckDB enforces a single-writer lock across *processes*.
        # Thread-based workers avoid this contention while still allowing
        # concurrent task execution.
        use_processes = _truthy(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_PROCESSES"))

        if use_processes:
            cmd: list[str] = [
                sys.executable,
                "-m",
                "ipfs_accelerate_py.p2p_tasks.worker",
                "--no-autoscale",
                "--no-p2p-service",
                "--no-mesh",
                "--queue",
                str(self._cfg.queue_path),
                "--worker-id",
                str(wid),
                "--poll-interval-s",
                "0.25",
            ]
            proc = subprocess.Popen(cmd, env=dict(os.environ))
            with self._lock:
                self._workers.append(self._WorkerHandle(worker_id=wid, started_ts=time.time(), kind="process", proc=proc))
            return

        stop_ev = threading.Event()

        def _run() -> None:
            try:
                from ipfs_accelerate_py.p2p_tasks.worker import run_worker

                run_worker(
                    queue_path=str(self._cfg.queue_path),
                    worker_id=str(wid),
                    poll_interval_s=0.25,
                    once=False,
                    p2p_service=False,
                    mesh=False,
                    accelerate_instance=self._accelerate,
                    supported_task_types=self._supported_task_types,
                    stop_event=stop_ev,
                )
            except Exception:
                # Best-effort: worker thread exceptions should not crash orchestrator.
                return

        t = threading.Thread(target=_run, name=f"ipfs_accelerate_py_task_worker[{wid}]", daemon=True)
        t.start()
        with self._lock:
            self._workers.append(
                self._WorkerHandle(worker_id=wid, started_ts=time.time(), kind="thread", thread=t, stop_event=stop_ev)
            )

    def _stop_all_workers(self) -> None:
        with self._lock:
            workers = list(self._workers)
            self._workers = []

        for w in workers:
            if w.kind == "thread" and w.stop_event is not None:
                try:
                    w.stop_event.set()
                except Exception:
                    pass

        for w in workers:
            if w.kind == "process" and w.proc is not None:
                try:
                    w.proc.terminate()
                except Exception:
                    pass

        for w in workers:
            if w.kind == "thread" and w.thread is not None:
                try:
                    w.thread.join(timeout=1.0)
                except Exception:
                    pass

        for w in workers:
            if w.kind == "process" and w.proc is not None:
                try:
                    w.proc.wait(timeout=1.0)
                except Exception:
                    try:
                        w.proc.kill()
                    except Exception:
                        pass

    def _stop_extra_workers(self, desired: int) -> None:
        with self._lock:
            while len(self._workers) > int(desired):
                w = self._workers.pop()
                if w.kind == "thread" and w.stop_event is not None:
                    try:
                        w.stop_event.set()
                    except Exception:
                        pass
                    continue
                if w.kind == "process" and w.proc is not None:
                    try:
                        w.proc.terminate()
                    except Exception:
                        pass
                    try:
                        w.proc.wait(timeout=1.0)
                    except Exception:
                        try:
                            w.proc.kill()
                        except Exception:
                            pass

    def _prune_dead_workers(self) -> None:
        with self._lock:
            keep: list[self._WorkerHandle] = []  # type: ignore[name-defined]
            for w in self._workers:
                try:
                    if w.kind == "process" and w.proc is not None:
                        if w.proc.poll() is None:
                            keep.append(w)
                        continue
                    if w.kind == "thread" and w.thread is not None:
                        if w.thread.is_alive():
                            keep.append(w)
                        continue
                except Exception:
                    continue
            self._workers = keep

    def _desired_workers(self, *, pending_total: int) -> int:
        mn = max(0, int(self._cfg.min_workers))
        mx = max(mn, int(self._cfg.max_workers))
        desired = int(pending_total)
        desired = max(mn, min(mx, desired if desired > 0 else mn))
        return desired

    def _remote_backlog(self, peers: list[object]) -> tuple[int, dict[str, int]]:
        total = 0
        by_type: dict[str, int] = {}
        try:
            from ipfs_accelerate_py.p2p_tasks.client import request_status_sync
        except Exception:
            return (0, {})

        for rq in peers:
            try:
                resp = request_status_sync(remote=rq, timeout_s=float(self._cfg.remote_status_timeout_s), detail=True)
            except Exception:
                continue
            if not (isinstance(resp, dict) and resp.get("ok")):
                continue
            try:
                queued = int(resp.get("queued") or 0)
            except Exception:
                queued = 0
            total += max(0, queued)

            qbt = resp.get("queued_by_type")
            if isinstance(qbt, dict):
                for k, v in qbt.items():
                    try:
                        by_type[str(k)] = int(by_type.get(str(k), 0)) + int(v)
                    except Exception:
                        continue

        return (int(total), by_type)

    def _claim_from_peers(self, *, peers: list[object], max_tasks: int) -> list[tuple[object, dict[str, Any]]]:
        if max_tasks <= 0:
            return []
        try:
            from ipfs_accelerate_py.p2p_tasks.client import claim_many_sync
        except Exception:
            return []

        supported = self._compute_supported_task_types()
        if not supported:
            return []

        session = _expected_session_tag() or None
        out: list[tuple[object, dict[str, Any]]] = []

        fanout = max(1, min(16, int(self._cfg.mesh_peer_fanout)))
        batch_n = max(1, min(64, int(self._cfg.mesh_claim_batch)))

        # Try up to fanout peers each cycle.
        for rq in list(peers)[:fanout]:
            if len(out) >= int(max_tasks):
                break
            try:
                tasks = claim_many_sync(
                    remote=rq,
                    worker_id=str(self._cfg.orchestrator_id),
                    supported_task_types=list(supported),
                    max_tasks=int(min(batch_n, max_tasks - len(out))),
                    same_task_type=True,
                    session_id=session,
                    peer_id=str(self._cfg.orchestrator_id),
                    clock=None,
                )
            except Exception:
                continue
            for t in list(tasks or []):
                if isinstance(t, dict):
                    out.append((rq, t))
        return out

    def _submit_proxy_tasks(self, *, claimed: list[tuple[object, dict[str, Any]]]) -> int:
        if not claimed:
            return 0
        from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue

        q = TaskQueue(self._cfg.queue_path)
        submitted = 0
        for remote, task in claimed:
            try:
                remote_task_id = str(task.get("task_id") or "").strip()
                task_type = str(task.get("task_type") or "").strip()
                model_name = str(task.get("model_name") or "").strip()
                payload = task.get("payload")
                if not isinstance(payload, dict):
                    payload = {"payload": payload}
                pid = str(getattr(remote, "peer_id", "") or "").strip()
                ma = str(getattr(remote, "multiaddr", "") or "").strip()
                if not (remote_task_id and task_type and model_name and pid and ma):
                    continue
            except Exception:
                continue

            # Avoid claiming the same remote task twice.
            with self._lock:
                if remote_task_id in self._inflight:
                    continue

            proxy_payload = dict(payload)
            proxy_payload.setdefault(
                "_p2p_proxy",
                {
                    "peer_id": pid,
                    "multiaddr": ma,
                    "task_id": remote_task_id,
                },
            )

            try:
                local_task_id = q.submit(task_type=task_type, model_name=model_name, payload=proxy_payload)
            except Exception:
                continue

            with self._lock:
                self._inflight[remote_task_id] = {
                    "peer_id": pid,
                    "multiaddr": ma,
                    "remote_task_id": remote_task_id,
                    "local_task_id": local_task_id,
                    "ts": time.time(),
                    "orchestrator_id": str(self._cfg.orchestrator_id),
                }
            submitted += 1

        return submitted

    def _release_stale_remote_tasks(self) -> None:
        # Best-effort release of remote tasks if local execution seems stuck.
        try:
            from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue, release_task_sync
            from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue
        except Exception:
            return

        q = TaskQueue(self._cfg.queue_path)
        now = time.time()
        with self._lock:
            items = list(self._inflight.items())

        for remote_task_id, meta in items:
            try:
                ts = float(meta.get("ts") or 0.0)
                local_task_id = str(meta.get("local_task_id") or "").strip()
                peer_id = str(meta.get("peer_id") or "").strip()
                multiaddr = str(meta.get("multiaddr") or "").strip()
                orch_id = str(meta.get("orchestrator_id") or "").strip() or str(self._cfg.orchestrator_id)
            except Exception:
                continue
            if not (local_task_id and peer_id and multiaddr):
                continue

            if (now - ts) < float(self._cfg.remote_task_ttl_s):
                # Also prune completed proxy tasks early.
                try:
                    lt = q.get(local_task_id)
                    st = str((lt or {}).get("status") or "").strip().lower()
                    if st in {"completed", "failed", "cancelled"}:
                        with self._lock:
                            self._inflight.pop(remote_task_id, None)
                except Exception:
                    pass
                continue

            # TTL exceeded: if the local proxy task isn't completed, release remote.
            try:
                lt = q.get(local_task_id)
                st = str((lt or {}).get("status") or "").strip().lower()
            except Exception:
                st = ""

            if st not in {"completed", "failed", "cancelled"}:
                try:
                    rq = RemoteQueue(peer_id=peer_id, multiaddr=multiaddr)
                    release_task_sync(remote=rq, task_id=remote_task_id, worker_id=orch_id, reason="orchestrator_ttl")
                except Exception:
                    pass
                try:
                    # Mark local as failed so it doesn't clog the queue forever.
                    from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue

                    TaskQueue(self._cfg.queue_path).complete(
                        task_id=local_task_id,
                        status="failed",
                        error=f"proxy TTL exceeded; released remote task {remote_task_id}",
                    )
                except Exception:
                    pass

            with self._lock:
                self._inflight.pop(remote_task_id, None)

    def _run(self) -> None:
        from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue

        q = TaskQueue(self._cfg.queue_path)
        last_prune_ts = 0.0
        try:
            retention_s = float(os.environ.get("IPFS_ACCELERATE_PY_TASK_QUEUE_RETENTION_S") or 86400.0)
        except Exception:
            retention_s = 86400.0
        prune_every_s = 30.0
        initial = max(1, min(int(self._cfg.max_workers), int(self._cfg.min_workers)))
        for i in range(initial):
            self._spawn_worker(idx=i)

        while not self._stop.is_set():
            self._prune_dead_workers()

            now = time.time()
            if retention_s > 0 and (now - last_prune_ts) >= prune_every_s:
                try:
                    q.prune_terminal(older_than_s=float(retention_s), limit=500)
                except Exception:
                    pass
                last_prune_ts = now

            try:
                local_pending = int(q.count(status="queued"))
            except Exception:
                local_pending = 0

            peers = self._snapshot_peers()
            remote_pending, _by_type = self._remote_backlog(peers)
            pending_total = int(local_pending) + int(remote_pending)

            if pending_total > 0:
                self._last_nonzero_ts = now

            desired = self._desired_workers(pending_total=pending_total)

            with self._lock:
                current = len(self._workers)

            if desired > current:
                for i in range(current, desired):
                    self._spawn_worker(idx=i)
            elif desired < current:
                idle_s = max(0.0, float(self._cfg.scale_down_idle_s))
                should_scale_down = (
                    idle_s <= 0.0
                    or (
                        self._last_nonzero_ts
                        and (now - self._last_nonzero_ts) >= idle_s
                    )
                    or pending_total == 0
                )
                if should_scale_down:
                    self._stop_extra_workers(desired)

            # Mesh draining: pull tasks from peers into local queue.
            try:
                local_fill_target = max(0, desired * 2)
                need = max(0, int(local_fill_target) - int(local_pending))
            except Exception:
                need = 0
            if need > 0 and peers:
                claimed = self._claim_from_peers(peers=peers, max_tasks=int(need))
                if claimed:
                    self._submit_proxy_tasks(claimed=claimed)

            self._release_stale_remote_tasks()

            self._stop.wait(max(0.2, float(self._cfg.scale_poll_s)))


def start_orchestrator_in_background(
    *,
    queue_path: str,
    accelerate_instance: object | None = None,
    supported_task_types: Optional[list[str]] = None,
) -> TaskOrchestrator:
    orch_id = str(os.environ.get("IPFS_ACCELERATE_PY_TASK_ORCHESTRATOR_ID") or "").strip()
    if not orch_id:
        orch_id = _default_orchestrator_id()
    base_worker_id = str(os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_ID") or "").strip()
    if not base_worker_id:
        base_worker_id = _default_base_worker_id()

    cfg = OrchestratorConfig(
        queue_path=str(queue_path),
        orchestrator_id=orch_id,
        base_worker_id=base_worker_id,
        min_workers=_env_int("IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_MIN", 1),
        max_workers=_env_int("IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_MAX", 4),
        scale_poll_s=_env_float("IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_POLL_S", 2.0),
        scale_down_idle_s=_env_float("IPFS_ACCELERATE_PY_TASK_WORKER_AUTOSCALE_IDLE_S", 30.0),
        mesh_refresh_s=_env_float("IPFS_ACCELERATE_PY_TASK_WORKER_MESH_REFRESH_S", 5.0),
        mesh_max_peers=_env_int("IPFS_ACCELERATE_PY_TASK_WORKER_MESH_MAX_PEERS", 10),
        mesh_claim_interval_s=_env_float("IPFS_ACCELERATE_PY_TASK_WORKER_MESH_CLAIM_INTERVAL_S", 0.25),
        mesh_peer_fanout=_env_int("IPFS_ACCELERATE_PY_TASK_WORKER_MESH_PEER_FANOUT", 2),
        mesh_claim_batch=_env_int("IPFS_ACCELERATE_PY_TASK_WORKER_MESH_CLAIM_BATCH", 8),
    )

    orch = TaskOrchestrator(
        config=cfg,
        accelerate_instance=accelerate_instance,
        supported_task_types=supported_task_types,
    )
    orch.start()
    return orch
