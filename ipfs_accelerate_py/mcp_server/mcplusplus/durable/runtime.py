"""Accelerate runtime binding to DurableExecutor@1 (RuntimeDurableAdapter@1).

MCPP-053 / track durable-accelerate: the accelerate MCP server dispatches
multi-step tasks through the primary DuckDB/Quack journaled DurableExecutor
adapter. This module is **wiring only** — it does not implement a second
journal, crash-recovery store, or private durable contract (ADR-0005 /
durable-execution.md §10.3). SQLite remains an explicit fallback.

Lifecycle:

* ``start`` / ``resume`` / ``cancel`` (plus checkpoint, recover, inspect,
  finalize) map 1:1 onto :class:`SqliteDurableExecutor` typed requests.
* Journaled transitions optionally emit Event DAG nodes via the executor's
  existing emission path (journal remains recovery authority).
* A **controlled restart hook** closes and reopens the executor against the
  same journal database so tests and operators can exercise start →
  checkpoint → restart → recover → resume without a second journal.
"""

from __future__ import annotations

import secrets
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Union

from ipfs_accelerate_py.mcp_server.mcplusplus.durable.journal import (
    ADAPTER_ID,
    DurableJournal,
)
from ipfs_accelerate_py.mcp_server.mcplusplus.durable.sqlite_executor import (
    INTERFACE_LABEL as EXECUTOR_INTERFACE,
    REQUEST_SCHEMA,
    RESULT_SCHEMA,
    SqliteDurableExecutor,
)
from ipfs_accelerate_py.mcp_server.mcplusplus.event_dag import EventDAGStore

# ---------------------------------------------------------------------------
# Interface markers
# ---------------------------------------------------------------------------

INTERFACE = "RuntimeDurableAdapter@1"
TASK_ID = "MCPP-053"
RUNTIME_NAME = "accelerate"

RestartHook = Callable[["RuntimeDurableAdapter", str], None]
PathLike = Union[str, Path]


class RuntimeDurableError(RuntimeError):
    """Base error for RuntimeDurableAdapter@1."""

    code = "runtime_durable_error"


class RuntimeNotOpenError(RuntimeDurableError):
    """Mutating dispatch attempted while the executor is closed."""

    code = "runtime_not_open"


class RuntimeDurableAdapter:
    """RuntimeDurableAdapter@1 — accelerate task dispatch over DurableExecutor.

    Parameters
    ----------
    db_path:
        DuckDB (primary) journal path owned by :class:`SqliteDurableExecutor` /
        :class:`DurableJournal`. This adapter never opens a second journal.
    clock_ms:
        Injectable clock (unix ms) forwarded to the executor.
    event_dag:
        Optional :class:`EventDAGStore`. When omitted and ``emit_events`` is
        True, a process-local store is created so journal transitions mint
        Event DAG provenance nodes.
    emit_events:
        When True (default), journal transitions emit Event DAG events via
        the executor. Emission failure never rolls back journal commits.
    restart_hook:
        Optional ``callable(adapter, boundary)`` invoked around
        :meth:`controlled_restart` (``before_close``, ``after_reopen``).
        Raising models mid-restart failure; recovery is reopen, not cleanup
        inside the hook.
    executor:
        Optional pre-built :class:`SqliteDurableExecutor`. When provided the
        adapter does not own the journal close lifecycle unless
        ``owns_executor`` is True.
    owns_executor:
        Whether :meth:`close` / :meth:`controlled_restart` may close and
        replace the executor instance.
    """

    interface = INTERFACE
    task_id = TASK_ID
    runtime = RUNTIME_NAME
    request_schema = REQUEST_SCHEMA
    result_schema = RESULT_SCHEMA

    def __init__(
        self,
        db_path: PathLike,
        *,
        clock_ms: Optional[Callable[[], int]] = None,
        event_dag: Optional[EventDAGStore] = None,
        emit_events: bool = True,
        restart_hook: Optional[RestartHook] = None,
        executor: Optional[SqliteDurableExecutor] = None,
        owns_executor: bool = True,
    ) -> None:
        self.db_path = Path(db_path)
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self._emit_events = bool(emit_events)
        self._event_dag = event_dag if event_dag is not None else (
            EventDAGStore() if self._emit_events else None
        )
        self._restart_hook = restart_hook
        self._owns_executor = bool(owns_executor)
        self._lock = threading.RLock()
        self._restart_count = 0
        self._closed = False
        self._request_seq = 0
        # correlation_id / caller task keys → latest execution_id (process-local)
        self._task_index: Dict[str, str] = {}

        if executor is not None:
            self._executor = executor
        else:
            self._executor = SqliteDurableExecutor.open(
                self.db_path,
                clock_ms=self._clock_ms,
                event_dag=self._event_dag,
                emit_events=self._emit_events,
            )

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def open(
        cls,
        db_path: PathLike,
        *,
        clock_ms: Optional[Callable[[], int]] = None,
        event_dag: Optional[EventDAGStore] = None,
        emit_events: bool = True,
        restart_hook: Optional[RestartHook] = None,
    ) -> "RuntimeDurableAdapter":
        """Open a runtime adapter bound to a SQLite DurableExecutor journal."""

        return cls(
            db_path,
            clock_ms=clock_ms,
            event_dag=event_dag,
            emit_events=emit_events,
            restart_hook=restart_hook,
        )

    def close(self) -> None:
        """Close the underlying DurableExecutor when this adapter owns it."""

        with self._lock:
            if self._closed:
                return
            if self._owns_executor:
                self._executor.close()
            self._closed = True

    def __enter__(self) -> "RuntimeDurableAdapter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @property
    def executor(self) -> SqliteDurableExecutor:
        """The bound DurableExecutor@1 adapter (SqliteDurableExecutor@1)."""

        return self._executor

    @property
    def journal(self) -> DurableJournal:
        """Journal authority — the executor's DurableJournal@1, not a second store."""

        return self._executor.journal

    @property
    def event_dag(self) -> Optional[EventDAGStore]:
        """Process-local Event DAG store used for provenance emission."""

        return self._event_dag

    @property
    def adapter_id(self) -> str:
        return ADAPTER_ID

    @property
    def executor_interface(self) -> str:
        return EXECUTOR_INTERFACE

    @property
    def restart_count(self) -> int:
        """How many times :meth:`controlled_restart` has completed successfully."""

        return self._restart_count

    @property
    def is_open(self) -> bool:
        return not self._closed

    # -- controlled restart ------------------------------------------------

    def controlled_restart(self) -> Dict[str, Any]:
        """Simulate process restart against the same durable journal.

        Closes the current :class:`SqliteDurableExecutor` and reopens a new
        instance on ``db_path``. The SQLite journal is the sole recovery
        authority — no second journal is created. Process-local Event DAG
        state is retained when the same :class:`EventDAGStore` instance is
        reattached (provenance only; not required for recover/resume).

        Invokes the optional restart hook at ``before_close`` and
        ``after_reopen`` boundaries.

        Returns
        -------
        dict
            ``ok``, ``restart_count``, ``db_path``, ``boundary`` metadata.
        """

        with self._lock:
            if not self._owns_executor:
                raise RuntimeDurableError(
                    "controlled_restart requires owns_executor=True so the "
                    "runtime can close and reopen the DurableExecutor"
                )
            if self._closed:
                raise RuntimeNotOpenError(
                    "cannot controlled_restart a closed RuntimeDurableAdapter"
                )

            self._invoke_restart_hook("before_close")
            self._executor.close()
            self._executor = SqliteDurableExecutor.open(
                self.db_path,
                clock_ms=self._clock_ms,
                event_dag=self._event_dag,
                emit_events=self._emit_events,
            )
            self._restart_count += 1
            self._invoke_restart_hook("after_reopen")
            return {
                "ok": True,
                "restart_count": self._restart_count,
                "db_path": str(self.db_path),
                "adapter_id": ADAPTER_ID,
                "executor_interface": EXECUTOR_INTERFACE,
                "runtime_interface": INTERFACE,
                "boundary": "after_reopen",
            }

    def set_restart_hook(self, hook: Optional[RestartHook]) -> None:
        """Replace the controlled-restart boundary hook."""

        with self._lock:
            self._restart_hook = hook

    def _invoke_restart_hook(self, boundary: str) -> None:
        hook = self._restart_hook
        if hook is not None:
            hook(self, boundary)

    # -- request helpers ---------------------------------------------------

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeNotOpenError(
                "RuntimeDurableAdapter is closed; reopen via open() or "
                "controlled_restart before dispatch"
            )

    def _next_request_id(self, prefix: str) -> str:
        self._request_seq += 1
        return f"{prefix}-{self._request_seq}-{secrets.token_hex(4)}"

    def _build_request(
        self,
        method: str,
        *,
        request_id: Optional[str] = None,
        **fields: Any,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "schema": REQUEST_SCHEMA,
            "method": method,
            "request_id": request_id or self._next_request_id(method),
            "issued_at_ms": int(self._clock_ms()),
        }
        for key, value in fields.items():
            if value is not None:
                body[key] = value
        return body

    def _index_task(
        self,
        *,
        execution_id: Optional[str],
        correlation_id: Optional[str],
        task_key: Optional[str],
    ) -> None:
        if not execution_id:
            return
        if isinstance(correlation_id, str) and correlation_id:
            self._task_index[f"corr:{correlation_id}"] = execution_id
        if isinstance(task_key, str) and task_key:
            self._task_index[f"task:{task_key}"] = execution_id
        self._task_index[f"exec:{execution_id}"] = execution_id

    def resolve_execution_id(
        self,
        *,
        execution_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        task_key: Optional[str] = None,
    ) -> Optional[str]:
        """Resolve a process-local task key to a durable execution_id."""

        if isinstance(execution_id, str) and execution_id:
            return execution_id
        if isinstance(correlation_id, str) and correlation_id:
            found = self._task_index.get(f"corr:{correlation_id}")
            if found:
                return found
        if isinstance(task_key, str) and task_key:
            found = self._task_index.get(f"task:{task_key}")
            if found:
                return found
        return None

    # -- typed DurableExecutor surface (accelerate dispatch) ---------------

    def handle(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        """Dispatch a typed DurableExecutor@1 request through the bound executor."""

        with self._lock:
            self._ensure_open()
            return self._executor.handle(request)

    def start(
        self,
        *,
        envelope_cid: str,
        idempotency_key: str,
        correlation_id: Optional[str] = None,
        task_key: Optional[str] = None,
        executor_did: Optional[str] = None,
        initial_checkpoint_cid: Optional[str] = None,
        parent_execution_id: Optional[str] = None,
        claim_fencing_token: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Begin durable accelerate task dispatch under an envelope identity."""

        with self._lock:
            self._ensure_open()
            req = self._build_request(
                "start",
                request_id=request_id,
                envelope_cid=envelope_cid,
                idempotency_key=idempotency_key,
                correlation_id=correlation_id,
                executor_did=executor_did,
                initial_checkpoint_cid=initial_checkpoint_cid,
                parent_execution_id=parent_execution_id,
                claim_fencing_token=claim_fencing_token,
            )
            result = self._executor.start(req)
            if result.get("ok"):
                self._index_task(
                    execution_id=result.get("execution_id"),
                    correlation_id=correlation_id,
                    task_key=task_key,
                )
            return result

    def resume(
        self,
        *,
        execution_id: str,
        fencing_token: int,
        from_checkpoint_id: Optional[str] = None,
        after_recover: bool = False,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Continue after checkpoint or process restart from journaled state."""

        with self._lock:
            self._ensure_open()
            req = self._build_request(
                "resume",
                request_id=request_id,
                execution_id=execution_id,
                fencing_token=fencing_token,
                from_checkpoint_id=from_checkpoint_id,
                after_recover=after_recover,
            )
            return self._executor.resume(req)

    def cancel(
        self,
        *,
        execution_id: str,
        reason: Optional[str] = None,
        fencing_token: Optional[int] = None,
        idempotency_key: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist cancellation; subsequent effects fail closed as cancelled."""

        with self._lock:
            self._ensure_open()
            req = self._build_request(
                "cancel",
                request_id=request_id,
                execution_id=execution_id,
                reason=reason,
                fencing_token=fencing_token,
                idempotency_key=idempotency_key,
            )
            return self._executor.cancel(req)

    def checkpoint(
        self,
        *,
        execution_id: str,
        fencing_token: int,
        idempotency_key: str,
        progress_cid: str,
        committed_side_effects: Optional[Any] = None,
        obligation_cids: Optional[Any] = None,
        state_transition_cids: Optional[Any] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Commit durable progress before further externally visible work."""

        with self._lock:
            self._ensure_open()
            req = self._build_request(
                "checkpoint",
                request_id=request_id,
                execution_id=execution_id,
                fencing_token=fencing_token,
                idempotency_key=idempotency_key,
                progress_cid=progress_cid,
                committed_side_effects=committed_side_effects,
                obligation_cids=obligation_cids,
                state_transition_cids=state_transition_cids,
            )
            return self._executor.checkpoint(req)

    def recover(
        self,
        *,
        execution_id: Optional[str] = None,
        fencing_token: Optional[int] = None,
        after_kill: bool = True,
        claim_new_fencing_token: Optional[bool] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Reconstruct runnable state after kill/restart from the journal."""

        with self._lock:
            self._ensure_open()
            req = self._build_request(
                "recover",
                request_id=request_id,
                execution_id=execution_id,
                fencing_token=fencing_token,
                after_kill=after_kill,
                claim_new_fencing_token=claim_new_fencing_token,
            )
            return self._executor.recover(req)

    def inspect(
        self,
        *,
        execution_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        include_journal: bool = False,
        include_timers: bool = False,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Read execution / journal status (read-only; does not advance fences)."""

        with self._lock:
            self._ensure_open()
            resolved = self.resolve_execution_id(
                execution_id=execution_id, correlation_id=correlation_id
            )
            req = self._build_request(
                "inspect",
                request_id=request_id,
                execution_id=resolved or execution_id,
                correlation_id=correlation_id,
                include_journal=include_journal,
                include_timers=include_timers,
            )
            return self._executor.inspect(req)

    def finalize(
        self,
        *,
        execution_id: str,
        fencing_token: int,
        terminal_status: str,
        result_cid: str,
        idempotency_key: str,
        output_cids: Optional[Any] = None,
        receipt_cid: Optional[str] = None,
        sign_receipt: Optional[bool] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Terminal binding; outputs bind to signed / local receipts."""

        with self._lock:
            self._ensure_open()
            req = self._build_request(
                "finalize",
                request_id=request_id,
                execution_id=execution_id,
                fencing_token=fencing_token,
                terminal_status=terminal_status,
                result_cid=result_cid,
                idempotency_key=idempotency_key,
                output_cids=output_cids,
                receipt_cid=receipt_cid,
                sign_receipt=sign_receipt,
            )
            return self._executor.finalize(req)

    # -- composite accelerate dispatch helpers -----------------------------

    def start_and_checkpoint(
        self,
        *,
        envelope_cid: str,
        start_idempotency_key: str,
        progress_cid: str,
        checkpoint_idempotency_key: str,
        correlation_id: Optional[str] = None,
        task_key: Optional[str] = None,
        committed_side_effects: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Convenience: start then checkpoint in one accelerate dispatch step."""

        started = self.start(
            envelope_cid=envelope_cid,
            idempotency_key=start_idempotency_key,
            correlation_id=correlation_id,
            task_key=task_key,
        )
        if not started.get("ok"):
            return {
                "ok": False,
                "stage": "start",
                "start": started,
                "checkpoint": None,
            }
        cp = self.checkpoint(
            execution_id=started["execution_id"],
            fencing_token=int(started["fencing_token"]),
            idempotency_key=checkpoint_idempotency_key,
            progress_cid=progress_cid,
            committed_side_effects=committed_side_effects,
        )
        return {
            "ok": bool(cp.get("ok")),
            "stage": "checkpoint" if cp.get("ok") else "checkpoint_failed",
            "start": started,
            "checkpoint": cp,
            "execution_id": started.get("execution_id"),
            "fencing_token": started.get("fencing_token"),
        }

    def recover_and_resume(
        self,
        *,
        execution_id: str,
        fencing_token: int,
        from_checkpoint_id: Optional[str] = None,
        after_kill: bool = True,
    ) -> Dict[str, Any]:
        """Convenience: recover after restart, then resume from journal head."""

        recovered = self.recover(
            execution_id=execution_id,
            fencing_token=fencing_token,
            after_kill=after_kill,
        )
        if not recovered.get("ok"):
            return {
                "ok": False,
                "stage": "recover",
                "recover": recovered,
                "resume": None,
            }
        resumed = self.resume(
            execution_id=execution_id,
            fencing_token=fencing_token,
            from_checkpoint_id=from_checkpoint_id,
            after_recover=True,
        )
        return {
            "ok": bool(resumed.get("ok")),
            "stage": "resume" if resumed.get("ok") else "resume_failed",
            "recover": recovered,
            "resume": resumed,
            "execution_id": execution_id,
            "fencing_token": fencing_token,
        }

    def advertisement(self) -> Dict[str, Any]:
        """Capability advertisement for accelerate durable task dispatch."""

        return {
            "interface": INTERFACE,
            "runtime": RUNTIME_NAME,
            "task_id": TASK_ID,
            "durable_executor": EXECUTOR_INTERFACE,
            "journal_adapter_id": ADAPTER_ID,
            "request_schema": REQUEST_SCHEMA,
            "result_schema": RESULT_SCHEMA,
            "methods": [
                "start",
                "resume",
                "cancel",
                "checkpoint",
                "recover",
                "inspect",
                "finalize",
                "handle",
                "controlled_restart",
            ],
            "second_journal": False,
            "emit_events": self._emit_events,
            "restart_count": self._restart_count,
            "db_path": str(self.db_path),
        }


def create_runtime_durable_adapter(
    db_path: PathLike,
    *,
    clock_ms: Optional[Callable[[], int]] = None,
    event_dag: Optional[EventDAGStore] = None,
    emit_events: bool = True,
    restart_hook: Optional[RestartHook] = None,
) -> RuntimeDurableAdapter:
    """Factory for RuntimeDurableAdapter@1 used by the accelerate runtime."""

    return RuntimeDurableAdapter.open(
        db_path,
        clock_ms=clock_ms,
        event_dag=event_dag,
        emit_events=emit_events,
        restart_hook=restart_hook,
    )


__all__ = [
    "INTERFACE",
    "TASK_ID",
    "RUNTIME_NAME",
    "RestartHook",
    "RuntimeDurableError",
    "RuntimeNotOpenError",
    "RuntimeDurableAdapter",
    "create_runtime_durable_adapter",
]
