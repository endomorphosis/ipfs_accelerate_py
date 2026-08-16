#!/usr/bin/env python3
"""Runtime tests for RuntimeDurableAdapter@1 (MCPP-053).

Acceptance (todo MCPP-053 / plan gate 17 / ADR-0005 §10.3):

* Accelerate task dispatch can start / resume / cancel through DurableExecutor
  and emit Event DAG events.
* Runtime test exercises start and resume after a **controlled restart hook**.
* No second journal implementation — wiring only over SqliteDurableExecutor@1
  and DurableJournal@1.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest

# Ensure repo roots are importable when pytest is launched from the workspace.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ACCEL_ROOT = _REPO_ROOT / "ipfs_accelerate_py"
for _p in (_REPO_ROOT, _ACCEL_ROOT):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ipfs_accelerate_py.mcp_server.mcplusplus.durable import runtime as runtime_mod
from ipfs_accelerate_py.mcp_server.mcplusplus.durable.journal import (
    ADAPTER_ID,
    INTERFACE_LABEL as JOURNAL_INTERFACE,
    DurableJournal,
)
from ipfs_accelerate_py.mcp_server.mcplusplus.durable.runtime import (
    INTERFACE,
    TASK_ID,
    RuntimeDurableAdapter,
    create_runtime_durable_adapter,
)
from ipfs_accelerate_py.mcp_server.mcplusplus.durable.sqlite_executor import (
    INTERFACE_LABEL as EXECUTOR_INTERFACE,
    REQUEST_SCHEMA,
    RESULT_SCHEMA,
    SqliteDurableExecutor,
)
from ipfs_accelerate_py.mcp_server.mcplusplus.event_dag import EventDAGStore
from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _clock(start_ms: int = 1_700_000_000_000) -> Callable[[], int]:
    state = {"now": start_ms}

    def now() -> int:
        return int(state["now"])

    now.advance = lambda ms: state.__setitem__("now", state["now"] + ms)  # type: ignore[attr-defined]
    return now


def _db(tmp_path: Path, name: str = "runtime-durable.sqlite3") -> Path:
    return tmp_path / name


def _open(
    path: Path,
    *,
    clock_ms: Optional[Callable[[], int]] = None,
    event_dag: Optional[EventDAGStore] = None,
    emit_events: bool = True,
    restart_hook: Optional[Callable[[RuntimeDurableAdapter, str], None]] = None,
) -> RuntimeDurableAdapter:
    return RuntimeDurableAdapter.open(
        path,
        clock_ms=clock_ms,
        event_dag=event_dag,
        emit_events=emit_events,
        restart_hook=restart_hook,
    )


# ---------------------------------------------------------------------------
# Interface / wiring
# ---------------------------------------------------------------------------


def test_interface_markers_and_advertisement(tmp_path: Path) -> None:
    path = _db(tmp_path)
    with _open(path) as rt:
        assert INTERFACE == "RuntimeDurableAdapter@1"
        assert TASK_ID == "MCPP-053"
        assert rt.interface == INTERFACE
        assert rt.executor_interface == "SqliteDurableExecutor@1"
        assert EXECUTOR_INTERFACE == "SqliteDurableExecutor@1"
        assert JOURNAL_INTERFACE == "DurableJournal@1"
        assert rt.adapter_id == ADAPTER_ID
        assert isinstance(rt.executor, SqliteDurableExecutor)
        assert isinstance(rt.journal, DurableJournal)
        assert rt.journal is rt.executor.journal

        ad = rt.advertisement()
        assert ad["interface"] == INTERFACE
        assert ad["runtime"] == "accelerate"
        assert ad["second_journal"] is False
        assert ad["durable_executor"] == EXECUTOR_INTERFACE
        assert "start" in ad["methods"]
        assert "resume" in ad["methods"]
        assert "cancel" in ad["methods"]
        assert "controlled_restart" in ad["methods"]


def test_factory_create_runtime_durable_adapter(tmp_path: Path) -> None:
    path = _db(tmp_path)
    rt = create_runtime_durable_adapter(path, emit_events=False)
    try:
        assert isinstance(rt, RuntimeDurableAdapter)
        assert rt.is_open
    finally:
        rt.close()


def test_no_second_journal_implementation() -> None:
    """Runtime module must wire SqliteDurableExecutor / DurableJournal only."""

    src = inspect.getsource(runtime_mod)
    # Must not define its own journal/store types or open raw sqlite schemas.
    assert "class " not in src or "class RuntimeDurable" in src
    assert "CREATE TABLE" not in src
    assert "sqlite3.connect" not in src
    assert "DurableJournal.open" not in src  # journal owned by executor
    assert "SqliteDurableExecutor" in src
    assert "second journal" in src.lower() or "does not implement a second journal" in src.lower()

    # Class docstring and module docstring state wiring-only intent.
    mod_doc = inspect.getdoc(runtime_mod) or ""
    cls_doc = inspect.getdoc(RuntimeDurableAdapter) or ""
    assert "DurableExecutor" in mod_doc
    assert "second journal" in mod_doc.lower() or "wiring only" in mod_doc.lower()
    assert "SqliteDurableExecutor" in cls_doc or "DurableExecutor" in cls_doc

    # Journal property is the executor's journal object (identity, not a fork).
    # Verified in open tests via ``rt.journal is rt.executor.journal``.


# ---------------------------------------------------------------------------
# Start / Event DAG emission
# ---------------------------------------------------------------------------


def test_start_emits_event_dag_and_journals_running(tmp_path: Path) -> None:
    path = _db(tmp_path)
    dag = EventDAGStore()
    envelope = _cid("envelope-runtime-start")
    with _open(path, event_dag=dag, emit_events=True) as rt:
        result = rt.start(
            envelope_cid=envelope,
            idempotency_key="rt-start-1",
            correlation_id="corr-runtime-1",
            task_key="task-runtime-1",
        )
        assert result["ok"] is True
        assert result["schema"] == RESULT_SCHEMA
        assert result["status"] == "running"
        assert result["fencing_token"] == 1
        assert result["envelope_cid"] == envelope
        assert result["execution_id"].startswith("dexec_")
        assert result["event_cid"], "start must emit an Event DAG event_cid"
        assert dag.has_event(result["event_cid"])
        event = dag.get_event(result["event_cid"])
        assert event is not None
        assert event["payload"]["transition"] == "started"
        assert event["payload"]["execution_id"] == result["execution_id"]

        # Process-local task index resolves correlation / task keys.
        assert (
            rt.resolve_execution_id(correlation_id="corr-runtime-1")
            == result["execution_id"]
        )
        assert (
            rt.resolve_execution_id(task_key="task-runtime-1")
            == result["execution_id"]
        )

        records = rt.journal.list_records(result["execution_id"])
        assert len(records) == 1
        assert records[0]["transition"] == "started"


def test_start_idempotent_returns_same_execution(tmp_path: Path) -> None:
    path = _db(tmp_path)
    envelope = _cid("envelope-idem")
    with _open(path, emit_events=False) as rt:
        a = rt.start(envelope_cid=envelope, idempotency_key="same-key")
        b = rt.start(envelope_cid=envelope, idempotency_key="same-key")
        assert a["ok"] and b["ok"]
        assert a["execution_id"] == b["execution_id"]
        assert b.get("idempotent_replay") is True


# ---------------------------------------------------------------------------
# Controlled restart: start → checkpoint → restart → recover → resume
# ---------------------------------------------------------------------------


def test_start_and_resume_after_controlled_restart_hook(tmp_path: Path) -> None:
    """Acceptance: start and resume after a controlled restart hook."""

    path = _db(tmp_path)
    clock = _clock()
    boundaries: List[str] = []
    dag = EventDAGStore()

    def hook(adapter: RuntimeDurableAdapter, boundary: str) -> None:
        boundaries.append(boundary)
        # Adapter remains open only after reopen; journal path is stable.
        assert adapter.db_path == path

    progress = _cid("progress-after-step-1")
    effect_cid = _cid("effect-once")
    envelope = _cid("envelope-restart")

    with _open(
        path, clock_ms=clock, event_dag=dag, emit_events=True, restart_hook=hook
    ) as rt:
        started = rt.start(
            envelope_cid=envelope,
            idempotency_key="restart-start-key",
            correlation_id="corr-restart",
            task_key="task-restart",
        )
        assert started["ok"] is True
        eid = started["execution_id"]
        fence = int(started["fencing_token"])
        start_event = started["event_cid"]
        assert start_event and dag.has_event(start_event)

        cp = rt.checkpoint(
            execution_id=eid,
            fencing_token=fence,
            idempotency_key="cp-step-1",
            progress_cid=progress,
            committed_side_effects=[
                {
                    "kind": "http_call",
                    "idempotency_key": "fx-once",
                    "effect_cid": effect_cid,
                    "compensatable": True,
                }
            ],
        )
        assert cp["ok"] is True
        assert cp["checkpoint_id"]
        assert cp["event_cid"], "checkpoint must emit Event DAG event"
        assert dag.has_event(cp["event_cid"])
        checkpoint_id = cp["checkpoint_id"]
        pre_restart_seq = int(cp["journal_seq"])

        # Controlled restart: close + reopen same journal (no second journal).
        restart_meta = rt.controlled_restart()
        assert restart_meta["ok"] is True
        assert restart_meta["restart_count"] == 1
        assert restart_meta["db_path"] == str(path)
        assert boundaries == ["before_close", "after_reopen"]
        assert rt.restart_count == 1
        assert rt.is_open
        # Same journal class / adapter id; new executor instance on same path.
        assert rt.adapter_id == ADAPTER_ID
        assert isinstance(rt.journal, DurableJournal)
        assert rt.journal is rt.executor.journal

        # Journal reconstructs committed state without replaying side effects.
        replayed = rt.journal.replay(eid)
        assert replayed["execution"]["status"] == "running"
        assert replayed["execution"]["progress_cid"] == progress
        assert replayed["execution"]["last_checkpoint_id"] == checkpoint_id
        assert "fx-once" in replayed["side_effects_not_replayed"]

        recovered = rt.recover(
            execution_id=eid, fencing_token=fence, after_kill=True
        )
        assert recovered["ok"] is True
        assert recovered["status"] == "running"
        assert len(recovered["recovered"]) == 1
        receipt = recovered["crash_recovery_receipt"]
        assert receipt is not None
        assert receipt["adapter_id"] == ADAPTER_ID
        assert "fx-once" in receipt["side_effects_not_replayed"]

        resumed = rt.resume(
            execution_id=eid,
            fencing_token=fence,
            from_checkpoint_id=checkpoint_id,
            after_recover=True,
        )
        assert resumed["ok"] is True
        assert resumed["status"] == "running"
        assert resumed["last_checkpoint_id"] == checkpoint_id
        assert resumed["event_cid"], "resume must emit Event DAG event"
        assert dag.has_event(resumed["event_cid"])
        assert int(resumed["journal_seq"]) > pre_restart_seq

        # Composite helper path also works post-restart (idempotent recover).
        again = rt.recover_and_resume(
            execution_id=eid,
            fencing_token=fence,
            from_checkpoint_id=checkpoint_id,
        )
        # Second recover on already-recovered running exec may still succeed
        # depending on executor policy; if ok, resume continues; if not, ensure
        # inspect still shows running from journal.
        inspected = rt.inspect(execution_id=eid, include_journal=True)
        assert inspected["ok"] is True
        assert inspected["status"] == "running"
        assert inspected["last_checkpoint_id"] == checkpoint_id
        transitions = [r["transition"] for r in inspected["journal_records"]]
        assert "started" in transitions
        assert "checkpointed" in transitions
        assert "recovered" in transitions
        assert "resumed" in transitions
        # Side-effect key still marked committed once.
        assert "fx-once" in rt.journal.replay(eid)["side_effects_not_replayed"]
        # Silence unused if recover_and_resume shape changes.
        assert "ok" in again


def test_recover_and_resume_helper_after_restart(tmp_path: Path) -> None:
    path = _db(tmp_path)
    progress = _cid("progress-helper")
    with _open(path, emit_events=True) as rt:
        started = rt.start(
            envelope_cid=_cid("env-helper"),
            idempotency_key="helper-start",
        )
        eid = started["execution_id"]
        fence = int(started["fencing_token"])
        cp = rt.checkpoint(
            execution_id=eid,
            fencing_token=fence,
            idempotency_key="helper-cp",
            progress_cid=progress,
        )
        assert cp["ok"]
        checkpoint_id = cp["checkpoint_id"]

        rt.controlled_restart()
        combo = rt.recover_and_resume(
            execution_id=eid,
            fencing_token=fence,
            from_checkpoint_id=checkpoint_id,
        )
        assert combo["ok"] is True
        assert combo["stage"] == "resume"
        assert combo["recover"]["ok"] is True
        assert combo["resume"]["ok"] is True
        assert combo["resume"]["status"] == "running"


# ---------------------------------------------------------------------------
# Cancel through DurableExecutor
# ---------------------------------------------------------------------------


def test_cancel_through_runtime_and_persists_across_restart(tmp_path: Path) -> None:
    path = _db(tmp_path)
    dag = EventDAGStore()
    with _open(path, event_dag=dag, emit_events=True) as rt:
        started = rt.start(
            envelope_cid=_cid("env-cancel"),
            idempotency_key="cancel-start",
        )
        eid = started["execution_id"]
        fence = int(started["fencing_token"])

        cancelled = rt.cancel(
            execution_id=eid,
            fencing_token=fence,
            reason="operator-abort",
            idempotency_key="cancel-once",
        )
        assert cancelled["ok"] is True
        assert cancelled["status"] == "cancelled"
        assert cancelled["event_cid"]
        assert dag.has_event(cancelled["event_cid"])

        rt.controlled_restart()
        inspected = rt.inspect(execution_id=eid, include_journal=True)
        assert inspected["ok"] is True
        assert inspected["status"] == "cancelled"
        assert inspected["cancel_state"] == "cancelled"

        # Resume after cancel fails closed.
        resumed = rt.resume(execution_id=eid, fencing_token=fence)
        assert resumed["ok"] is False
        assert resumed["error"]["code"] in {"cancelled", "terminal_execution"}


# ---------------------------------------------------------------------------
# Finalize + handle dispatch
# ---------------------------------------------------------------------------


def test_finalize_after_resume_binds_receipt(tmp_path: Path) -> None:
    path = _db(tmp_path)
    with _open(path, emit_events=True) as rt:
        started = rt.start(
            envelope_cid=_cid("env-final"),
            idempotency_key="final-start",
        )
        eid = started["execution_id"]
        fence = int(started["fencing_token"])
        cp = rt.checkpoint(
            execution_id=eid,
            fencing_token=fence,
            idempotency_key="final-cp",
            progress_cid=_cid("progress-final"),
        )
        assert cp["ok"]
        rt.controlled_restart()
        combo = rt.recover_and_resume(
            execution_id=eid,
            fencing_token=fence,
            from_checkpoint_id=cp["checkpoint_id"],
        )
        assert combo["ok"]

        fin = rt.finalize(
            execution_id=eid,
            fencing_token=fence,
            terminal_status="succeeded",
            result_cid=_cid("result-final"),
            idempotency_key="final-done",
        )
        assert fin["ok"] is True
        assert fin["status"] == "succeeded" or fin.get("terminal_status") == "succeeded"
        assert fin.get("receipt_cid") or fin.get("result_cid")
        assert fin.get("event_cid")


def test_handle_dispatches_typed_requests(tmp_path: Path) -> None:
    path = _db(tmp_path)
    with _open(path, emit_events=False) as rt:
        started = rt.handle(
            {
                "schema": REQUEST_SCHEMA,
                "method": "start",
                "request_id": "h-start",
                "issued_at_ms": 1_700_000_000_000,
                "envelope_cid": _cid("env-handle"),
                "idempotency_key": "handle-key",
            }
        )
        assert started["ok"] is True
        eid = started["execution_id"]
        inspected = rt.handle(
            {
                "schema": REQUEST_SCHEMA,
                "method": "inspect",
                "request_id": "h-inspect",
                "issued_at_ms": 1_700_000_000_001,
                "execution_id": eid,
            }
        )
        assert inspected["ok"] is True
        assert inspected["execution_id"] == eid


def test_start_and_checkpoint_helper(tmp_path: Path) -> None:
    path = _db(tmp_path)
    with _open(path, emit_events=True) as rt:
        combo = rt.start_and_checkpoint(
            envelope_cid=_cid("env-sac"),
            start_idempotency_key="sac-start",
            progress_cid=_cid("sac-progress"),
            checkpoint_idempotency_key="sac-cp",
            correlation_id="corr-sac",
            task_key="task-sac",
        )
        assert combo["ok"] is True
        assert combo["stage"] == "checkpoint"
        assert combo["start"]["ok"] is True
        assert combo["checkpoint"]["ok"] is True
        assert combo["execution_id"]


def test_closed_adapter_rejects_dispatch(tmp_path: Path) -> None:
    path = _db(tmp_path)
    rt = _open(path, emit_events=False)
    rt.close()
    assert not rt.is_open
    with pytest.raises(runtime_mod.RuntimeNotOpenError):
        rt.start(envelope_cid=_cid("env-closed"), idempotency_key="x")


def test_restart_hook_receives_boundaries_in_order(tmp_path: Path) -> None:
    path = _db(tmp_path)
    seen: List[str] = []

    def hook(_adapter: RuntimeDurableAdapter, boundary: str) -> None:
        seen.append(boundary)

    with _open(path, emit_events=False, restart_hook=hook) as rt:
        rt.start(envelope_cid=_cid("env-hook"), idempotency_key="hook-start")
        rt.controlled_restart()
        rt.controlled_restart()
        assert seen == [
            "before_close",
            "after_reopen",
            "before_close",
            "after_reopen",
        ]
        assert rt.restart_count == 2
