"""Unit tests for SqliteDurableExecutor@1 and DurableJournal@1 (MCPP-051).

Acceptance (todo MCPP-051 / plan gate 17 / ADR-0005):

* Journal replay reconstructs runnable state after process restart
* Idempotent retry / checkpoint does not re-commit side effects
* Cancel state persists across reopen
* Stale fencing tokens are rejected on exclusive mutators and recover
* Restate / Dapr absence is a documented non-blocker
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pytest

from ipfs_accelerate_py.mcp_server.mcplusplus.durable.journal import (
    ADAPTER_ID,
    INTERFACE_LABEL as JOURNAL_INTERFACE,
    DurableJournal,
)
from ipfs_accelerate_py.mcp_server.mcplusplus.durable.sqlite_executor import (
    INTERFACE_LABEL as EXECUTOR_INTERFACE,
    RESTATE_DAPR_NON_BLOCKER,
    REQUEST_SCHEMA,
    RESULT_SCHEMA,
    SqliteDurableExecutor,
)
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
    now.set = lambda ms: state.__setitem__("now", int(ms))  # type: ignore[attr-defined]
    return now


def _db(tmp_path: Path, name: str = "durable.sqlite3") -> Path:
    return tmp_path / name


def _open(
    path: Path,
    *,
    clock_ms: Optional[Callable[[], int]] = None,
) -> SqliteDurableExecutor:
    return SqliteDurableExecutor.open(path, clock_ms=clock_ms, emit_events=False)


def _base_req(method: str, request_id: str, **extra: Any) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "schema": REQUEST_SCHEMA,
        "method": method,
        "request_id": request_id,
        "issued_at_ms": 1_700_000_000_000,
    }
    body.update(extra)
    return body


def _start(
    ex: SqliteDurableExecutor,
    *,
    request_id: str = "start-1",
    idempotency_key: str = "start-key-1",
    envelope: Optional[str] = None,
    claim_fencing_token: Optional[int] = None,
) -> Dict[str, Any]:
    req = _base_req(
        "start",
        request_id,
        envelope_cid=envelope or _cid("envelope-default"),
        idempotency_key=idempotency_key,
        correlation_id="corr-1",
    )
    if claim_fencing_token is not None:
        req["claim_fencing_token"] = claim_fencing_token
    return ex.start(req)


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------


def test_opens_with_wal_journal_mode(tmp_path: Path) -> None:
    path = _db(tmp_path)
    with _open(path) as ex:
        assert ex.journal_mode() == "wal"
        assert ex.adapter_id == ADAPTER_ID
        assert EXECUTOR_INTERFACE == "SqliteDurableExecutor@1"
        assert JOURNAL_INTERFACE == "DurableJournal@1"
        assert ex.journal.db_version() == DurableJournal.DB_VERSION


def test_start_journals_running_execution(tmp_path: Path) -> None:
    path = _db(tmp_path)
    envelope = _cid("envelope-start")
    with _open(path) as ex:
        result = _start(ex, envelope=envelope)
        assert result["ok"] is True
        assert result["schema"] == RESULT_SCHEMA
        assert result["status"] == "running"
        assert result["fencing_token"] == 1
        assert result["journal_seq"] == 1
        assert result["idempotent_replay"] is False
        assert result["envelope_cid"] == envelope
        assert result["journal_record_cid"]
        assert isinstance(result["execution_id"], str)
        assert result["execution_id"].startswith("dexec_")

        records = ex.journal.list_records(result["execution_id"])
        assert len(records) == 1
        assert records[0]["transition"] == "started"
        assert records[0]["journal_seq"] == 1


# ---------------------------------------------------------------------------
# Journal replay
# ---------------------------------------------------------------------------


def test_journal_replay_reconstructs_state_after_restart(tmp_path: Path) -> None:
    """Process restart: reopen DB and reconstruct from journal records."""

    path = _db(tmp_path)
    envelope = _cid("envelope-replay")
    progress = _cid("progress-1")
    effect_cid = _cid("effect-payload-1")

    with _open(path) as ex:
        started = _start(ex, envelope=envelope)
        eid = started["execution_id"]
        fence = started["fencing_token"]

        cp = ex.checkpoint(
            _base_req(
                "checkpoint",
                "cp-1",
                execution_id=eid,
                fencing_token=fence,
                idempotency_key="cp-key-1",
                progress_cid=progress,
                committed_side_effects=[
                    {
                        "kind": "http_call",
                        "idempotency_key": "fx-1",
                        "effect_cid": effect_cid,
                        "compensatable": True,
                    }
                ],
            )
        )
        assert cp["ok"] is True
        assert cp["checkpoint_id"]
        assert cp["journal_seq"] == 2
        checkpoint_id = cp["checkpoint_id"]
        frontier = cp["journal_seq"]

    # Simulate process death: new executor on the same durable file.
    with _open(path) as recovered:
        replayed = recovered.journal.replay(eid)
        projected = replayed["execution"]
        assert projected["status"] == "running"
        assert projected["envelope_cid"] == envelope
        assert projected["progress_cid"] == progress
        assert projected["last_checkpoint_id"] == checkpoint_id
        assert projected["journal_seq"] == frontier
        assert projected["fencing_token"] == fence
        assert "fx-1" in replayed["side_effects_not_replayed"]
        assert len(replayed["records"]) == 2
        assert [r["transition"] for r in replayed["records"]] == [
            "started",
            "checkpointed",
        ]

        # recover() journals a recovery marker without replaying side effects.
        rec = recovered.recover(
            _base_req(
                "recover",
                "recover-1",
                execution_id=eid,
                fencing_token=fence,
                after_kill=True,
            )
        )
        assert rec["ok"] is True
        assert rec["status"] == "running"
        assert len(rec["recovered"]) == 1
        assert rec["recovered"][0]["execution_id"] == eid
        receipt = rec["crash_recovery_receipt"]
        assert receipt["schema"] == "mcp++/durable/crash-recovery-receipt@1"
        assert receipt["adapter_id"] == ADAPTER_ID
        assert "fx-1" in receipt["side_effects_not_replayed"]
        assert eid in receipt["execution_ids"]

        inspected = recovered.inspect(
            _base_req(
                "inspect",
                "inspect-1",
                execution_id=eid,
                include_journal=True,
            )
        )
        assert inspected["ok"] is True
        assert inspected["status"] == "running"
        assert inspected["last_checkpoint_id"] == checkpoint_id
        assert inspected["journal_frontier_seq"] == frontier + 1  # recovered append
        transitions = [r["transition"] for r in inspected["journal_records"]]
        assert transitions == ["started", "checkpointed", "recovered"]


def test_resume_after_recover_continues_from_checkpoint(tmp_path: Path) -> None:
    path = _db(tmp_path)
    progress = _cid("progress-resume")
    with _open(path) as ex:
        started = _start(ex)
        eid = started["execution_id"]
        fence = started["fencing_token"]
        cp = ex.checkpoint(
            _base_req(
                "checkpoint",
                "cp-1",
                execution_id=eid,
                fencing_token=fence,
                idempotency_key="cp-resume",
                progress_cid=progress,
            )
        )
        checkpoint_id = cp["checkpoint_id"]

    with _open(path) as ex:
        rec = ex.recover(
            _base_req(
                "recover",
                "recover-1",
                execution_id=eid,
                fencing_token=fence,
                after_kill=True,
            )
        )
        assert rec["ok"] is True
        resumed = ex.resume(
            _base_req(
                "resume",
                "resume-1",
                execution_id=eid,
                fencing_token=fence,
                from_checkpoint_id=checkpoint_id,
                after_recover=True,
            )
        )
        assert resumed["ok"] is True
        assert resumed["status"] == "running"
        assert resumed["last_checkpoint_id"] == checkpoint_id


# ---------------------------------------------------------------------------
# Idempotent retry / start / checkpoint
# ---------------------------------------------------------------------------


def test_start_idempotent_retry_returns_same_execution(tmp_path: Path) -> None:
    path = _db(tmp_path)
    envelope = _cid("envelope-idem")
    with _open(path) as ex:
        first = _start(ex, request_id="s1", idempotency_key="start-same", envelope=envelope)
        second = _start(ex, request_id="s2", idempotency_key="start-same", envelope=envelope)
        assert first["ok"] and second["ok"]
        assert first["execution_id"] == second["execution_id"]
        assert second["idempotent_replay"] is True
        assert second["journal_seq"] == first["journal_seq"]
        # Only one journal root.
        assert len(ex.journal.list_records(first["execution_id"])) == 1

        conflict = _start(
            ex,
            request_id="s3",
            idempotency_key="start-same",
            envelope=_cid("envelope-other"),
        )
        assert conflict["ok"] is False
        assert conflict["error"]["code"] == "idempotency_conflict"


def test_checkpoint_idempotent_retry_does_not_recommit_side_effects(
    tmp_path: Path,
) -> None:
    path = _db(tmp_path)
    progress = _cid("progress-idem")
    effect_cid = _cid("fx-payload")
    with _open(path) as ex:
        started = _start(ex)
        eid = started["execution_id"]
        fence = started["fencing_token"]
        side_effects = [
            {
                "kind": "tool_invoke",
                "idempotency_key": "fx-commit-1",
                "effect_cid": effect_cid,
            }
        ]
        req_body = dict(
            execution_id=eid,
            fencing_token=fence,
            idempotency_key="cp-idem-1",
            progress_cid=progress,
            committed_side_effects=side_effects,
        )
        first = ex.checkpoint(_base_req("checkpoint", "cp-a", **req_body))
        assert first["ok"] is True
        seq = first["journal_seq"]
        checkpoint_id = first["checkpoint_id"]

        # Exact retry: same result, no new journal row, effect still once.
        second = ex.checkpoint(_base_req("checkpoint", "cp-b", **req_body))
        assert second["ok"] is True
        assert second["checkpoint_id"] == checkpoint_id
        assert second["journal_seq"] == seq
        effects = ex.journal.committed_side_effects(eid)
        assert len(effects) == 1
        assert effects[0]["idempotency_key"] == "fx-commit-1"

        # Different payload under same key fails closed.
        bad = ex.checkpoint(
            _base_req(
                "checkpoint",
                "cp-c",
                execution_id=eid,
                fencing_token=fence,
                idempotency_key="cp-idem-1",
                progress_cid=_cid("progress-other"),
                committed_side_effects=side_effects,
            )
        )
        assert bad["ok"] is False
        assert bad["error"]["code"] == "idempotency_conflict"


def test_retry_is_idempotent_and_skips_committed_side_effects(tmp_path: Path) -> None:
    path = _db(tmp_path)
    with _open(path) as ex:
        started = _start(ex)
        eid = started["execution_id"]
        fence = started["fencing_token"]
        ex.checkpoint(
            _base_req(
                "checkpoint",
                "cp-1",
                execution_id=eid,
                fencing_token=fence,
                idempotency_key="cp-before-retry",
                progress_cid=_cid("progress-before-retry"),
                committed_side_effects=[
                    {
                        "kind": "publish",
                        "idempotency_key": "fx-already",
                        "effect_cid": _cid("fx-already-body"),
                    }
                ],
            )
        )

        first = ex.retry(
            _base_req(
                "retry",
                "retry-1",
                execution_id=eid,
                fencing_token=fence,
                idempotency_key="retry-key-1",
                reason="transient_error",
                max_attempts=5,
            )
        )
        assert first["ok"] is True
        assert first["attempt"] == 1
        assert "fx-already" in first["side_effects_not_replayed"]
        seq = first["journal_seq"]

        # Idempotent method retry returns prior result without advancing attempt.
        second = ex.retry(
            _base_req(
                "retry",
                "retry-2",
                execution_id=eid,
                fencing_token=fence,
                idempotency_key="retry-key-1",
                reason="transient_error",
                max_attempts=5,
            )
        )
        assert second["ok"] is True
        assert second["attempt"] == 1
        assert second["journal_seq"] == seq
        assert "fx-already" in second["side_effects_not_replayed"]

        # New key advances attempt; still does not re-commit side effects.
        third = ex.retry(
            _base_req(
                "retry",
                "retry-3",
                execution_id=eid,
                fencing_token=fence,
                idempotency_key="retry-key-2",
                reason="another_try",
            )
        )
        assert third["ok"] is True
        assert third["attempt"] == 2
        assert "fx-already" in third["side_effects_not_replayed"]
        assert len(ex.journal.committed_side_effects(eid)) == 1


def test_duplicate_side_effect_key_filtered_on_later_checkpoint(tmp_path: Path) -> None:
    path = _db(tmp_path)
    with _open(path) as ex:
        started = _start(ex)
        eid = started["execution_id"]
        fence = started["fencing_token"]
        effect = {
            "kind": "write",
            "idempotency_key": "fx-once",
            "effect_cid": _cid("fx-once-body"),
        }
        first = ex.checkpoint(
            _base_req(
                "checkpoint",
                "cp-1",
                execution_id=eid,
                fencing_token=fence,
                idempotency_key="cp-a",
                progress_cid=_cid("p1"),
                committed_side_effects=[effect],
            )
        )
        assert first["ok"] is True
        second = ex.checkpoint(
            _base_req(
                "checkpoint",
                "cp-2",
                execution_id=eid,
                fencing_token=fence,
                idempotency_key="cp-b",
                progress_cid=_cid("p2"),
                committed_side_effects=[effect],  # already committed
            )
        )
        assert second["ok"] is True
        assert len(ex.journal.committed_side_effects(eid)) == 1


# ---------------------------------------------------------------------------
# Cancel persistence
# ---------------------------------------------------------------------------


def test_cancel_persists_across_reopen(tmp_path: Path) -> None:
    path = _db(tmp_path)
    with _open(path) as ex:
        started = _start(ex)
        eid = started["execution_id"]
        fence = started["fencing_token"]
        # Schedule a timer so cancel must clear outstanding work.
        timer = ex.timer(
            _base_req(
                "timer",
                "timer-1",
                execution_id=eid,
                fencing_token=fence,
                timer_id="t-wait",
                durable=True,
                delay_ms=60_000,
            )
        )
        assert timer["ok"] is True
        assert timer["status"] == "paused"

        cancelled = ex.cancel(
            _base_req(
                "cancel",
                "cancel-1",
                execution_id=eid,
                fencing_token=fence,
                reason="operator_abort",
                idempotency_key="cancel-key-1",
            )
        )
        assert cancelled["ok"] is True
        assert cancelled["status"] == "cancelled"

    with _open(path) as recovered:
        execution = recovered.journal.get_execution(eid)
        assert execution["status"] == "cancelled"
        assert execution["cancel_state"] == "cancelled"
        assert execution["cancel_reason"] == "operator_abort"

        inspected = recovered.inspect(
            _base_req(
                "inspect",
                "inspect-cancel",
                execution_id=eid,
                include_timers=True,
                include_journal=True,
            )
        )
        assert inspected["ok"] is True
        assert inspected["status"] == "cancelled"
        assert inspected["cancel_state"] == "cancelled"
        timer_statuses = {t["timer_id"]: t["status"] for t in inspected["timers"]}
        assert timer_statuses.get("t-wait") == "cancelled"

        # Mutators fail closed after cancel.
        resume = recovered.resume(
            _base_req(
                "resume",
                "resume-after-cancel",
                execution_id=eid,
                fencing_token=fence,
            )
        )
        assert resume["ok"] is False
        assert resume["error"]["code"] in {"cancelled", "terminal_execution"}

        cp = recovered.checkpoint(
            _base_req(
                "checkpoint",
                "cp-after-cancel",
                execution_id=eid,
                fencing_token=fence,
                idempotency_key="cp-nope",
                progress_cid=_cid("progress-nope"),
            )
        )
        assert cp["ok"] is False
        assert cp["error"]["code"] in {"cancelled", "terminal_execution"}

        # Cancel itself is idempotent.
        again = recovered.cancel(
            _base_req(
                "cancel",
                "cancel-2",
                execution_id=eid,
                fencing_token=fence,
                reason="operator_abort",
            )
        )
        assert again["ok"] is True
        assert again["status"] == "cancelled"


# ---------------------------------------------------------------------------
# Stale fence reject
# ---------------------------------------------------------------------------


def test_stale_fence_rejected_on_checkpoint_and_resume(tmp_path: Path) -> None:
    path = _db(tmp_path)
    with _open(path) as ex:
        started = _start(ex, claim_fencing_token=3)
        eid = started["execution_id"]
        assert started["fencing_token"] == 3
        fence = 3

        stale_cp = ex.checkpoint(
            _base_req(
                "checkpoint",
                "cp-stale",
                execution_id=eid,
                fencing_token=2,  # stale
                idempotency_key="cp-stale",
                progress_cid=_cid("p-stale"),
            )
        )
        assert stale_cp["ok"] is False
        assert stale_cp["error"]["code"] == "stale_fence"
        assert stale_cp["error"]["details"]["presented_fencing_token"] == 2
        assert stale_cp["error"]["details"]["current_fencing_token"] == 3

        missing = ex.checkpoint(
            _base_req(
                "checkpoint",
                "cp-missing-fence",
                execution_id=eid,
                idempotency_key="cp-missing",
                progress_cid=_cid("p-missing"),
            )
        )
        assert missing["ok"] is False
        assert missing["error"]["code"] == "fencing_token_required"

        stale_resume = ex.resume(
            _base_req(
                "resume",
                "resume-stale",
                execution_id=eid,
                fencing_token=1,
            )
        )
        assert stale_resume["ok"] is False
        assert stale_resume["error"]["code"] == "stale_fence"

        # Current fence still works.
        ok_cp = ex.checkpoint(
            _base_req(
                "checkpoint",
                "cp-ok",
                execution_id=eid,
                fencing_token=fence,
                idempotency_key="cp-ok",
                progress_cid=_cid("p-ok"),
            )
        )
        assert ok_cp["ok"] is True


def test_stale_fence_rejected_on_recover(tmp_path: Path) -> None:
    path = _db(tmp_path)
    with _open(path) as ex:
        started = _start(ex, claim_fencing_token=5)
        eid = started["execution_id"]
        fence = started["fencing_token"]
        ex.checkpoint(
            _base_req(
                "checkpoint",
                "cp-1",
                execution_id=eid,
                fencing_token=fence,
                idempotency_key="cp-before-recover",
                progress_cid=_cid("progress-before-recover"),
            )
        )

    with _open(path) as recovered:
        stale = recovered.recover(
            _base_req(
                "recover",
                "recover-stale",
                execution_id=eid,
                fencing_token=1,
                after_kill=True,
            )
        )
        assert stale["ok"] is False
        assert stale["status"] == "rejected"
        assert stale["error"]["code"] == "stale_fence"
        assert len(stale["rejected_stale"]) == 1
        assert stale["rejected_stale"][0]["presented_fencing_token"] == 1
        assert stale["rejected_stale"][0]["current_fencing_token"] == 5
        assert stale["recovered"] == []

        # claim_new_fencing_token advances the epoch and succeeds.
        claimed = recovered.recover(
            _base_req(
                "recover",
                "recover-claim",
                execution_id=eid,
                claim_new_fencing_token=True,
                after_kill=True,
            )
        )
        assert claimed["ok"] is True
        assert claimed["recovered"][0]["fencing_token"] == 6

        # Old fence is now stale.
        after = recovered.resume(
            _base_req(
                "resume",
                "resume-old-fence",
                execution_id=eid,
                fencing_token=5,
            )
        )
        assert after["ok"] is False
        assert after["error"]["code"] == "stale_fence"

        ok = recovered.resume(
            _base_req(
                "resume",
                "resume-new-fence",
                execution_id=eid,
                fencing_token=6,
            )
        )
        assert ok["ok"] is True


def test_finalize_requires_current_fence(tmp_path: Path) -> None:
    path = _db(tmp_path)
    with _open(path) as ex:
        started = _start(ex)
        eid = started["execution_id"]
        fence = started["fencing_token"]
        result_cid = _cid("result-final")

        stale = ex.finalize(
            _base_req(
                "finalize",
                "fin-stale",
                execution_id=eid,
                fencing_token=0,
                terminal_status="succeeded",
                result_cid=result_cid,
                idempotency_key="fin-1",
            )
        )
        assert stale["ok"] is False
        assert stale["error"]["code"] == "stale_fence"

        done = ex.finalize(
            _base_req(
                "finalize",
                "fin-ok",
                execution_id=eid,
                fencing_token=fence,
                terminal_status="succeeded",
                result_cid=result_cid,
                idempotency_key="fin-1",
            )
        )
        assert done["ok"] is True
        assert done["status"] == "succeeded"
        assert done["receipt_cid"]
        assert done["result_cid"] == result_cid

        # Terminal: further mutators fail closed.
        again_cp = ex.checkpoint(
            _base_req(
                "checkpoint",
                "cp-after-fin",
                execution_id=eid,
                fencing_token=fence,
                idempotency_key="cp-after",
                progress_cid=_cid("p-after"),
            )
        )
        assert again_cp["ok"] is False
        assert again_cp["error"]["code"] == "terminal_execution"


# ---------------------------------------------------------------------------
# Timers / signal (supporting durable surface)
# ---------------------------------------------------------------------------


def test_durable_timer_survives_reopen(tmp_path: Path) -> None:
    path = _db(tmp_path)
    clock = _clock()
    with _open(path, clock_ms=clock) as ex:
        started = _start(ex)
        eid = started["execution_id"]
        fence = started["fencing_token"]
        scheduled = ex.timer(
            _base_req(
                "timer",
                "timer-1",
                execution_id=eid,
                fencing_token=fence,
                timer_id="deadline",
                durable=True,
                fire_at_ms=clock() + 5_000,
            )
        )
        assert scheduled["ok"] is True
        fire_at = scheduled["fire_at_ms"]

    with _open(path, clock_ms=clock) as recovered:
        timers = recovered.journal.list_timers(eid)
        assert len(timers) == 1
        assert timers[0]["timer_id"] == "deadline"
        assert timers[0]["status"] == "scheduled"
        assert timers[0]["fire_at_ms"] == fire_at


def test_signal_is_journaled(tmp_path: Path) -> None:
    path = _db(tmp_path)
    with _open(path) as ex:
        started = _start(ex)
        eid = started["execution_id"]
        fence = started["fencing_token"]
        sig = ex.signal(
            _base_req(
                "signal",
                "sig-1",
                execution_id=eid,
                fencing_token=fence,
                signal_name="user_input",
                payload_cid=_cid("signal-payload"),
            )
        )
        assert sig["ok"] is True
        assert sig["accepted"] is True
        records = ex.journal.list_records(eid)
        assert any(r["transition"] == "signalled" for r in records)


# ---------------------------------------------------------------------------
# Restate / Dapr non-blocker documentation
# ---------------------------------------------------------------------------


def test_restate_dapr_absence_is_documented_non_blocker() -> None:
    """Restate/Dapr are optional; SQLite journal is the mandatory adapter."""

    assert "non-blocker" in RESTATE_DAPR_NON_BLOCKER.lower()
    assert "Restate" in RESTATE_DAPR_NON_BLOCKER
    assert "Dapr" in RESTATE_DAPR_NON_BLOCKER
    assert "SQLite" in RESTATE_DAPR_NON_BLOCKER

    # Module and class docstrings carry the same acceptance language.
    module_doc = inspect.getdoc(SqliteDurableExecutor) or ""
    assert "DurableExecutor" in module_doc or EXECUTOR_INTERFACE

    import ipfs_accelerate_py.mcp_server.mcplusplus.durable.sqlite_executor as mod

    src = inspect.getsource(mod)
    assert "Restate" in src and "Dapr" in src
    assert "non-blocker" in src.lower() or "non blocker" in src.lower()

    adr = Path(
        "ipfs_accelerate_py/mcplusplus/docs/architecture/decisions/0005-durable-executor.md"
    )
    assert adr.is_file()
    text = adr.read_text(encoding="utf-8")
    assert "documented non-blocker" in text
    assert "Restate" in text and "Dapr" in text
    assert "SQLite" in text


def test_handle_dispatches_methods(tmp_path: Path) -> None:
    path = _db(tmp_path)
    with _open(path) as ex:
        started = ex.handle(
            _base_req(
                "start",
                "h1",
                envelope_cid=_cid("env-handle"),
                idempotency_key="handle-start",
            )
        )
        assert started["ok"] is True
        eid = started["execution_id"]
        inspected = ex.handle(
            _base_req("inspect", "h2", execution_id=eid)
        )
        assert inspected["ok"] is True
        assert inspected["execution_id"] == eid

        unknown = ex.handle(_base_req("not_a_method", "h3"))  # type: ignore[arg-type]
        # method validation happens before dispatch via METHODS set
        assert unknown["ok"] is False
        assert unknown["error"]["code"] == "unknown_method"
