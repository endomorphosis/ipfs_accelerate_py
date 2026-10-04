"""Native authority tests for a task-only external callback guard.

These tests use real DuckDB claim, lease, attempt, token, and completion rows.
An external marker deliberately survives a rejected post-check: the guard is
not a transaction over the callback's store. Thread and subprocess controls
exercise the landed process-serialized adapter's actual file locks.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import select
import subprocess
import sys
import threading
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
    AttemptStatus,
    DatabaseCoordinationBoundsError,
    DatabaseCoordinationConflictError,
    DatabaseCoordinationError,
    DatabaseCoordinationExpiredError,
    DatabaseCoordinationNotReadyError,
    DatabaseCoordinationStaleFenceError,
    DatabaseCoordinator,
    LeaseState,
    TaskClaim,
    duckdb_available,
    open_database_coordinator,
    open_process_serialized_database_coordinator,
)


pytestmark = pytest.mark.skipif(
    not duckdb_available(), reason="native DuckDB coordination is required"
)


_ACTIVE_EVIDENCE: dict[str, Any] | None = None
_EVIDENCE_LOCK = threading.Lock()
_MAX_WITNESS_BYTES = 8 * 1024 * 1024


def _native_tables(connection, *, wrapped: bool) -> dict[str, Any]:
    """Export actual columns/values through the already held native connection.

    Callback exports use only a connection captured before guard entry. This
    is test observation, never a replacement authority decision or public
    coordinator re-entry. The closed export uses native read-only DuckDB.
    """
    metadata = connection.execute(
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema = 'main' ORDER BY table_name, ordinal_position"
    ).fetchall()
    columns: dict[str, list[str]] = {}
    for row in metadata:
        table, column = (row["table_name"], row["column_name"]) if wrapped else row
        assert re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", table)
        columns.setdefault(table, []).append(column)
    tables = {}
    for table, names in columns.items():
        values = connection.execute('SELECT * FROM "' + table + '"').fetchall()
        records = [dict(row) if wrapped else dict(zip(names, row)) for row in values]
        records.sort(key=lambda row: json.dumps(row, sort_keys=True, allow_nan=False))
        tables[table] = {"columns": names, "rows": records, "row_count": len(records)}
    return {"tables": tables, "sql_statement_count": 1 + len(columns)}


def _clock_value(coordinator) -> int:
    return int(coordinator._clock_ms())


def _execute(coordinator, claim, callback, *, minimum_remaining_ms=0):
    """Record a bounded observation around the actual public native call."""
    evidence = _ACTIVE_EVIDENCE
    if evidence is None:
        return coordinator.execute_with_task_claim_fence(
            claim, callback, minimum_remaining_ms=minimum_remaining_ms
        )
    record: dict[str, Any] = {
        "operation": "execute_with_task_claim_fence",
        "coordinator_type": type(coordinator).__name__,
        "authority_path": str(coordinator.database_path),
        "claim": claim.to_dict() if isinstance(claim, TaskClaim) else dict(claim),
        "minimum_remaining_ms": minimum_remaining_ms,
        "clock_before_ms": _clock_value(coordinator),
        "callback_entered": False,
        "callback_count": 0,
        "guard_returned": False,
    }
    with _EVIDENCE_LOCK:
        record["operation_ordinal"] = len(evidence["guard_operations"])
        evidence["guard_operations"].append(record)
    connection = None
    if isinstance(coordinator, DatabaseCoordinator) and not coordinator._fenced_callback_active:
        connection = coordinator._require()
        record["native_before_rows"] = _native_tables(connection, wrapped=True)

    def observed_callback(lease):
        record["callback_entered"] = True
        record["callback_count"] += 1
        record["supplied_native_lease"] = lease.to_dict()
        record["clock_callback_entry_ms"] = _clock_value(coordinator)
        if connection is not None:
            record["native_callback_entry_rows"] = _native_tables(connection, wrapped=True)
        try:
            result = callback(lease)
            record["callback_returned"] = True
            record["callback_result_type"] = type(result).__name__
            if result is None or type(result) in (str, int, bool, dict, list):
                record["callback_result"] = result
            return result
        except BaseException as error:
            record["callback_returned"] = False
            record["callback_error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            record["clock_callback_return_ms"] = _clock_value(coordinator)
            if connection is not None:
                record["native_callback_return_rows"] = _native_tables(connection, wrapped=True)

    try:
        result = coordinator.execute_with_task_claim_fence(
            claim, observed_callback if callable(callback) else callback,
            minimum_remaining_ms=minimum_remaining_ms,
        )
        record["guard_returned"] = True
        record["guard_result_type"] = type(result).__name__
        return result
    except BaseException as error:
        record["guard_error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        record["clock_after_ms"] = _clock_value(coordinator)
        if connection is not None:
            record["native_after_rows"] = _native_tables(connection, wrapped=True)


@pytest.fixture(autouse=True)
def actual_native_witness(request, tmp_path: Path):
    global _ACTIVE_EVIDENCE
    witness: dict[str, Any] = {
        "schema": "native-task-claim-fenced-callback-case-evidence@1",
        "node_id": request.node.nodeid,
        "guard_operations": [],
        "initial_native_authorities": [],
        "closed_native_authorities": [],
        "external_files": [],
        "scope": {
            "actual_native_duckdb": True,
            "callback_sql_observation_authority": False,
            "native_corruption_controls_are_supported_clients": False,
            "failed_guard_asserts_no_external_effects": False,
            "model_training": False,
        },
    }
    assert _ACTIVE_EVIDENCE is None
    _ACTIVE_EVIDENCE = witness
    try:
        yield witness
    finally:
        _ACTIVE_EVIDENCE = None
        import duckdb
        for path in sorted(tmp_path.rglob("*.duckdb")):
            if path.is_symlink() or not path.is_file():
                continue
            before_bytes = path.read_bytes()
            metadata = path.stat()
            closed: dict[str, Any] = {
                "path": str(path), "bytes": len(before_bytes),
                "sha256": hashlib.sha256(before_bytes).hexdigest(),
                "mode": metadata.st_mode, "uid": metadata.st_uid,
                "gid": metadata.st_gid, "nlink": metadata.st_nlink,
            }
            try:
                with duckdb.connect(str(path), read_only=True, config={"threads": "1", "memory_limit": "256MB", "enable_external_access": "false"}) as connection:
                    closed["native_read_only_rows"] = _native_tables(connection, wrapped=False)
            except BaseException as error:
                closed["export_error"] = {"type": type(error).__name__, "message": str(error)}
            closed["after_read_only_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            closed["native_read_only_file_bytes_unchanged"] = closed["sha256"] == closed["after_read_only_sha256"]
            witness["closed_native_authorities"].append(closed)
        for path in sorted(tmp_path.rglob("*")):
            if path.is_symlink() or not path.is_file() or path.suffix == ".duckdb" or path.name.startswith("."):
                continue
            content = path.read_bytes()
            assert len(content) <= 64 * 1024
            witness["external_files"].append({
                "path": str(path), "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "content_hex": content.hex(),
                "role": "test_external_marker_or_retained_native_auxiliary",
            })
        body = json.dumps(witness, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
        assert len(body) <= _MAX_WITNESS_BYTES
        (tmp_path / "task-claim-guard-evidence.json").write_bytes(body)


class Clock:
    def __init__(self, now: int = 1_000_000) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now

    def advance(self, milliseconds: int) -> None:
        self.now += milliseconds


def _claim(coordinator: Any, *, lease_ms: int = 10_000) -> TaskClaim:
    coordinator.register_task(task_cid="task:guard", task_id="GUARD")
    claim = coordinator.claim_task(
        task_cid="task:guard",
        owner_session_id="session:owner",
        idempotency_key="guard-attempt",
        lease_ms=lease_ms,
    )
    if _ACTIVE_EVIDENCE is not None:
        _ACTIVE_EVIDENCE["initial_native_authorities"].append({
            "path": str(coordinator.database_path),
            "claim": claim.to_dict(),
            "full_native_registry": coordinator.coordination_registry_projection(),
        })
    return claim


@pytest.fixture
def native(tmp_path: Path):
    clock = Clock()
    coordinator = open_database_coordinator(
        tmp_path / "authority.duckdb", clock_ms=clock, default_lease_ms=10_000
    )
    claim = _claim(coordinator)
    try:
        yield coordinator, clock, claim
    finally:
        coordinator.close()


def _rows(coordinator: DatabaseCoordinator, claim: TaskClaim) -> dict[str, Any]:
    connection = coordinator._require()

    def rows(sql, parameters=()):
        return [dict(row) for row in connection.execute(sql, parameters).fetchall()]

    return {
        "claim": rows(
            "SELECT * FROM task_claims WHERE claim_id = ?", [claim.claim_id]
        ),
        "lease": rows(
            "SELECT * FROM fenced_leases WHERE lease_id = ?", [claim.lease_id]
        ),
        "attempt": rows(
            "SELECT * FROM task_attempts WHERE attempt_id = ?", [claim.attempt_id]
        ),
        "tokens": rows(
            "SELECT * FROM token_history ORDER BY scope_key, fencing_token, fence_epoch"
        ),
        "completion": rows(
            "SELECT * FROM task_completions WHERE task_cid = ?", [claim.task_cid]
        ),
        "events": coordinator.lease_events(),
    }


def test_returns_actual_callback_object_inside_native_transaction(native) -> None:
    coordinator, _clock, claim = native
    connection = coordinator._require()
    result = object()
    observations = []

    def callback(protected_lease):
        observations.append(connection.in_transaction)
        row = connection.execute(
            "SELECT owner_session_id, status FROM task_attempts WHERE attempt_id = ?",
            [claim.attempt_id],
        ).fetchone()
        assert row is not None
        assert (row["owner_session_id"], row["status"]) == (
            claim.owner_session_id, AttemptStatus.RUNNING.value
        )
        assert protected_lease == claim.as_fenced_lease()
        return result

    assert _execute(coordinator, claim, callback) is result
    assert observations == [True]
    assert not connection.in_transaction
    assert coordinator.protect_task_claim(claim).lease_id == claim.lease_id


@pytest.mark.parametrize("remaining,minimum,admitted", [
    (5_001, 5_000, True),
    (5_000, 5_000, False),
    (4_999, 5_000, False),
    (1, 0, True),
    (0, 0, False),
])
def test_actual_deadline_requires_strict_remaining_budget(
    native, remaining: int, minimum: int, admitted: bool
) -> None:
    coordinator, clock, claim = native
    clock.advance(10_000 - remaining)
    calls = []
    callback = lambda protected_lease: calls.append("entered")
    if admitted:
        _execute(coordinator, 
            claim, callback, minimum_remaining_ms=minimum
        )
        assert calls == ["entered"]
    else:
        with pytest.raises(DatabaseCoordinationExpiredError):
            _execute(coordinator, 
                claim, callback, minimum_remaining_ms=minimum
            )
        assert calls == []


def test_current_renewed_deadline_overrides_old_projection_deadline(native) -> None:
    coordinator, clock, claim = native
    unchanged_projection = claim.to_dict()
    clock.advance(9_000)
    renewed = coordinator.renew(claim.as_fenced_lease(), lease_ms=20_000)
    assert renewed.lease_id == claim.lease_id
    assert (renewed.fencing_token, renewed.fence_epoch) == (
        claim.fencing_token, claim.fence_epoch
    )
    assert unchanged_projection == claim.to_dict()
    assert renewed.expires_at_ms > claim.expires_at_ms
    assert _execute(coordinator, 
        unchanged_projection, lambda protected_lease: "renewed", minimum_remaining_ms=5_000
    ) == "renewed"


def test_shortened_same_fence_renewal_supplies_actual_deadline_to_callback(native) -> None:
    coordinator, clock, claim = native
    original_projection = claim.to_dict()
    clock.advance(1_000)
    shortened = coordinator.renew(claim.as_fenced_lease(), lease_ms=6_000)
    assert shortened.expires_at_ms < claim.expires_at_ms
    assert (shortened.lease_id, shortened.fencing_token, shortened.fence_epoch) == (
        claim.lease_id, claim.fencing_token, claim.fence_epoch
    )
    seen = []

    def callback(protected_lease):
        seen.append(protected_lease)
        assert protected_lease.expires_at_ms == shortened.expires_at_ms
        assert protected_lease.expires_at_ms != original_projection["expires_at_ms"]
        return "actual shortened deadline"

    assert _execute(coordinator, 
        original_projection, callback, minimum_remaining_ms=5_000
    ) == "actual shortened deadline"
    assert seen == [shortened]
    clock.advance(1_000)
    with pytest.raises(DatabaseCoordinationExpiredError):
        _execute(coordinator, 
            original_projection, callback, minimum_remaining_ms=5_000
        )
    assert seen == [shortened]


def test_callback_may_consume_entry_budget_while_authority_stays_live(native) -> None:
    coordinator, clock, claim = native

    def callback(protected_lease):
        clock.advance(9_999)
        return "acknowledged before expiry"

    assert _execute(coordinator, 
        claim, callback, minimum_remaining_ms=5_000
    ) == "acknowledged before expiry"
    assert coordinator.protect_task_claim(claim).expires_at_ms == clock.now + 1


@pytest.mark.parametrize("field", [
    "task_cid", "claim_id", "attempt_id", "attempt_number",
    "owner_session_id", "lease_id", "fencing_token", "fence_epoch",
])
def test_one_changed_identity_field_prevents_callback(native, field: str) -> None:
    coordinator, _clock, claim = native
    identity = claim.to_dict()
    identity[field] = identity[field] + 1 if isinstance(identity[field], int) else "wrong:" + str(identity[field])
    calls = []
    with pytest.raises(DatabaseCoordinationStaleFenceError):
        _execute(coordinator, identity, lambda protected_lease: calls.append(field))
    assert calls == []
    assert coordinator.protect_task_claim(claim).lease_id == claim.lease_id


@pytest.mark.parametrize("lifecycle", ["expired", "released", "replaced"])
def test_unchanged_claim_projection_cannot_hide_native_lease_change(
    native, lifecycle: str
) -> None:
    coordinator, clock, claim = native
    projection = claim.to_dict()
    if lifecycle == "released":
        coordinator.release(claim.as_fenced_lease(), reason="native retirement")
    else:
        clock.advance(10_000)
        if lifecycle == "replaced":
            successor = coordinator.claim_task(
                task_cid=claim.task_cid,
                owner_session_id="session:successor",
                idempotency_key="successor-attempt",
            )
            assert successor.lease_id != claim.lease_id
            assert successor.fencing_token > claim.fencing_token
    calls = []
    with pytest.raises((DatabaseCoordinationExpiredError, DatabaseCoordinationStaleFenceError)):
        _execute(coordinator, projection, lambda protected_lease: calls.append("entered"))
    assert calls == []
    assert projection == claim.to_dict()


@pytest.mark.parametrize("projection", [
    "claim_owner", "lease_owner", "lease_expiry", "attempt_owner",
    "attempt_status", "latest_token", "logical_completion",
])
def test_native_projection_change_is_checked_before_callback(native, projection: str) -> None:
    coordinator, clock, claim = native
    connection = coordinator._require()
    _mutate(connection, clock, claim, projection)
    coordinator._commit_if_idle(connection)
    calls = []
    expected_error = DatabaseCoordinationNotReadyError if projection == "logical_completion" else DatabaseCoordinationError
    with pytest.raises(expected_error):
        _execute(coordinator, claim, lambda protected_lease: calls.append("entered"))
    assert calls == []


def _mutate(connection, clock: Clock, claim: TaskClaim, projection: str) -> None:
    if projection == "claim_owner":
        connection.execute("UPDATE task_claims SET owner_session_id = ? WHERE claim_id = ?", ["session:other", claim.claim_id])
    elif projection == "lease_owner":
        connection.execute("UPDATE fenced_leases SET owner_session_id = ? WHERE lease_id = ?", ["session:other", claim.lease_id])
    elif projection == "lease_expiry":
        connection.execute("UPDATE fenced_leases SET expires_at_ms = expires_at_ms + 1 WHERE lease_id = ?", [claim.lease_id])
    elif projection == "attempt_owner":
        connection.execute("UPDATE task_attempts SET owner_session_id = ? WHERE attempt_id = ?", ["session:other", claim.attempt_id])
    elif projection == "attempt_status":
        connection.execute("UPDATE task_attempts SET status = ? WHERE attempt_id = ?", [AttemptStatus.FAILED.value, claim.attempt_id])
    elif projection == "latest_token":
        connection.execute(
            "INSERT INTO token_history(scope_key, fencing_token, fence_epoch, recorded_at_ms) VALUES (?, ?, ?, ?)",
            [claim.as_fenced_lease().scope_key, claim.fencing_token + 1, claim.fence_epoch + 1, clock.now],
        )
    elif projection == "logical_completion":
        connection.execute(
            "INSERT INTO task_completions(task_cid, completed_at_ms, status, body_json) VALUES (?, ?, ?, ?)",
            [claim.task_cid, clock.now, "prepared", "{}"],
        )
    else:
        raise AssertionError(projection)


@pytest.mark.parametrize("projection", [
    "claim_owner", "lease_owner", "lease_expiry", "attempt_owner",
    "attempt_status", "latest_token", "logical_completion",
])
def test_full_native_postcheck_rolls_back_guard_but_keeps_external_effect(
    native, tmp_path: Path, projection: str
) -> None:
    coordinator, clock, claim = native
    connection = coordinator._require()
    before = _rows(coordinator, claim)
    marker = tmp_path / "external-effect.json"
    receipt = {"attempt_id": claim.attempt_id, "outcome": "callback returned"}

    def callback(protected_lease):
        # A captured native connection is used only to inject a corruption
        # control. Public coordinator re-entry is independently refused.
        _mutate(connection, clock, claim, projection)
        marker.write_text(json.dumps(receipt), encoding="utf-8")
        return receipt

    expected_error = DatabaseCoordinationNotReadyError if projection == "logical_completion" else DatabaseCoordinationError
    with pytest.raises(expected_error):
        _execute(coordinator, claim, callback)
    assert json.loads(marker.read_text(encoding="utf-8")) == receipt
    assert _rows(coordinator, claim) == before
    assert not connection.in_transaction
    assert _execute(coordinator, claim, lambda protected_lease: "reusable") == "reusable"


@pytest.mark.parametrize("clock_change", [10_000, 10_001, -1])
def test_postcheck_clock_rejection_keeps_possible_external_effect(
    native, tmp_path: Path, clock_change: int
) -> None:
    coordinator, clock, claim = native
    before = _rows(coordinator, claim)
    marker = tmp_path / "callback-effect"

    def callback(protected_lease):
        marker.write_bytes(b"external effect cannot be rolled back")
        clock.advance(clock_change)
        return "possible effect"

    with pytest.raises((DatabaseCoordinationExpiredError, DatabaseCoordinationStaleFenceError)):
        _execute(coordinator, claim, callback)
    assert marker.read_bytes() == b"external effect cannot be rolled back"
    assert _rows(coordinator, claim) == before
    assert not coordinator._require().in_transaction


@pytest.mark.parametrize("operation", ["read", "renew", "close", "nested_guard"])
def test_suppressed_native_callback_reentry_still_refuses(native, operation: str) -> None:
    coordinator, _clock, claim = native
    before = _rows(coordinator, claim)
    operations = {
        "read": lambda: coordinator.get_task_claim(claim.claim_id),
        "renew": lambda: coordinator.renew(claim.as_fenced_lease()),
        "close": coordinator.close,
        "nested_guard": lambda: _execute(coordinator, claim, lambda protected_lease: "nested"),
    }

    def callback(protected_lease):
        with pytest.raises(DatabaseCoordinationConflictError, match="re-enter"):
            operations[operation]()
        return "exception was suppressed"

    with pytest.raises(DatabaseCoordinationConflictError, match="re-enter"):
        _execute(coordinator, claim, callback)
    assert _rows(coordinator, claim) == before
    assert _execute(coordinator, claim, lambda protected_lease: "usable") == "usable"


def test_callback_exception_propagates_without_fabricating_completion(native, tmp_path: Path) -> None:
    coordinator, _clock, claim = native
    before = _rows(coordinator, claim)
    marker = tmp_path / "callback-started"
    failure = RuntimeError("external callback failed after possible effect")

    def callback(protected_lease):
        marker.write_bytes(b"possible effect")
        raise failure

    with pytest.raises(RuntimeError) as error:
        _execute(coordinator, claim, callback)
    assert error.value is failure
    assert marker.read_bytes() == b"possible effect"
    assert _rows(coordinator, claim) == before
    assert coordinator.get_task_attempt(claim.attempt_id).status is AttemptStatus.RUNNING
    assert _execute(coordinator, claim, lambda protected_lease: "retry guard only") == "retry guard only"


@pytest.mark.parametrize("minimum", [-1, True, 1.5, "5000"])
def test_invalid_budget_prevents_callback(native, minimum) -> None:
    coordinator, _clock, claim = native
    calls = []
    with pytest.raises((DatabaseCoordinationBoundsError, TypeError, ValueError)):
        _execute(coordinator, 
            claim, lambda protected_lease: calls.append("entered"), minimum_remaining_ms=minimum
        )
    assert calls == []


def test_noncallable_rejected_without_guard_transaction(native) -> None:
    coordinator, _clock, claim = native
    before = _rows(coordinator, claim)
    with pytest.raises(TypeError, match="callable"):
        _execute(coordinator, claim, None)
    assert _rows(coordinator, claim) == before


@pytest.mark.parametrize("keyword_callback", [False, True])
def test_serialized_alias_reentry_cannot_be_suppressed(tmp_path: Path, keyword_callback: bool) -> None:
    clock = Clock()
    path = tmp_path / "serialized.duckdb"
    first = open_process_serialized_database_coordinator(path, clock_ms=clock)
    second = open_process_serialized_database_coordinator(path, clock_ms=clock)
    try:
        claim = _claim(first)

        def callback(protected_lease):
            with pytest.raises(DatabaseCoordinationConflictError, match="re-enter"):
                second.protect_task_claim(claim)
            return "suppressed alias error"

        with pytest.raises(DatabaseCoordinationConflictError, match="re-enter"):
            if keyword_callback:
                _execute(first, claim=claim, callback=callback)
            else:
                _execute(first, claim, callback)
        assert _execute(second, claim, lambda protected_lease: "released") == "released"
        assert second.get_task_attempt(claim.attempt_id).status is AttemptStatus.RUNNING
    finally:
        second.close()
        first.close()


def _capture_thread_result(results, function) -> None:
    try:
        results.append(("result", function()))
    except BaseException as error:
        results.append(("error", error))


def test_two_serialized_adapters_wait_for_callback_ack_in_threads(tmp_path: Path) -> None:
    clock = Clock()
    path = tmp_path / "thread-authority.duckdb"
    first = open_process_serialized_database_coordinator(path, clock_ms=clock)
    second = open_process_serialized_database_coordinator(path, clock_ms=clock)
    entered = threading.Event()
    release = threading.Event()
    contender_started = threading.Event()
    contender_entered = threading.Event()
    holder_results = []
    contender_results = []
    holder = contender = None
    try:
        claim = _claim(first)

        def holding_callback(protected_lease):
            entered.set()
            assert release.wait(10), "callback ACK was not released"
            return "holder acknowledged"

        holder = threading.Thread(target=_capture_thread_result, args=(
            holder_results, lambda: _execute(first, claim, holding_callback)
        ))
        holder.start()
        assert entered.wait(10)

        def contending_operation():
            contender_started.set()
            return _execute(second, 
                claim, lambda protected_lease: (contender_entered.set(), "contender acknowledged")[1]
            )

        contender = threading.Thread(target=_capture_thread_result, args=(contender_results, contending_operation))
        contender.start()
        assert contender_started.wait(10)
        assert not contender_entered.wait(0.2)
        release.set()
        holder.join(10)
        contender.join(10)
        assert not holder.is_alive() and not contender.is_alive()
        assert holder_results == [("result", "holder acknowledged")]
        assert contender_results == [("result", "contender acknowledged")]
    finally:
        release.set()
        for thread in (holder, contender):
            if thread is not None:
                thread.join(10)
        second.close()
        first.close()


_PROCESS_CONTENDER = r'''
import fcntl, json, pathlib, sys
from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import open_process_serialized_database_coordinator
path = pathlib.Path(sys.argv[1])
claim = json.loads(sys.argv[2])
now = int(sys.argv[3])
for suffix in (".serialized-coordinator.lock", ".lock"):
    lock_path = path.with_name("." + path.name + suffix)
    with lock_path.open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("busy:" + suffix, flush=True)
        else:
            raise AssertionError("callback did not retain " + suffix)
coordinator = open_process_serialized_database_coordinator(path, clock_ms=lambda: now, require_existing_authority=True)
try:
    result = coordinator.execute_with_task_claim_fence(claim, lambda protected_lease: "contender acknowledged")
    print("result:" + result, flush=True)
finally:
    coordinator.close()
'''


def test_serialized_callback_retains_both_os_file_locks_until_ack(tmp_path: Path) -> None:
    clock = Clock()
    path = tmp_path / "process-authority.duckdb"
    first = open_process_serialized_database_coordinator(path, clock_ms=clock)
    entered = threading.Event()
    release = threading.Event()
    results = []
    holder = None
    child = None
    try:
        claim = _claim(first)

        def callback(protected_lease):
            entered.set()
            assert release.wait(15), "callback ACK was not released"
            return "holder acknowledged"

        holder = threading.Thread(target=_capture_thread_result, args=(
            results, lambda: _execute(first, claim, callback)
        ))
        holder.start()
        assert entered.wait(10)
        child = subprocess.Popen(
            [sys.executable, "-B", "-c", _PROCESS_CONTENDER, str(path), json.dumps(claim.to_dict()), str(clock.now)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        assert child.stdout is not None
        assert select.select([child.stdout], [], [], 10)[0], "child did not report lock probe"
        first_line = child.stdout.readline().strip()
        second_line = child.stdout.readline().strip()
        assert first_line == "busy:.serialized-coordinator.lock"
        assert second_line == "busy:.lock"
        assert child.poll() is None
        assert not select.select([child.stdout], [], [], 0.2)[0]
        release.set()
        stdout, stderr = child.communicate(timeout=15)
        if _ACTIVE_EVIDENCE is not None:
            _ACTIVE_EVIDENCE["process_controls"] = [{
                "role": "actual_native_authority_contender",
                "pid": child.pid, "returncode": child.returncode,
                "script_sha256": hashlib.sha256(_PROCESS_CONTENDER.encode()).hexdigest(),
                "argv": child.args,
                "stdout": first_line + "\n" + second_line + "\n" + stdout,
                "stderr": stderr,
            }]
        holder.join(10)
        assert child.returncode == 0, stderr
        assert stdout.strip() == "result:contender acknowledged"
        assert results == [("result", "holder acknowledged")]
        assert not holder.is_alive()
    finally:
        release.set()
        if holder is not None:
            holder.join(10)
        if child is not None and child.poll() is None:
            child.kill()
            child.communicate(timeout=10)
        first.close()


def test_serialized_callback_failure_releases_both_file_locks(tmp_path: Path) -> None:
    clock = Clock()
    path = tmp_path / "failed-authority.duckdb"
    first = open_process_serialized_database_coordinator(path, clock_ms=clock)
    second = open_process_serialized_database_coordinator(path, clock_ms=clock)
    try:
        claim = _claim(first)

        def callback(protected_lease):
            raise RuntimeError("callback failed")

        with pytest.raises(RuntimeError, match="callback failed"):
            _execute(first, claim, callback)
        # Probe each actual OS lock with a different open file description.
        import fcntl
        for lock_path in (
            first.serialization_lock_path, path.with_name("." + path.name + ".lock")
        ):
            with lock_path.open("a+b") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        assert _execute(second, claim, lambda protected_lease: "available") == "available"
        assert second.get_task_attempt(claim.attempt_id).status is AttemptStatus.RUNNING
    finally:
        second.close()
        first.close()


@pytest.mark.parametrize("serialized", [False, True])
def test_require_existing_authority_never_installs_missing_database(tmp_path: Path, serialized: bool) -> None:
    path = tmp_path / "missing" / "authority.duckdb"
    factory = open_process_serialized_database_coordinator if serialized else open_database_coordinator
    with pytest.raises(DatabaseCoordinationError):
        factory(path, require_existing_authority=True)
    assert not path.exists()
    assert not path.parent.exists()


@pytest.mark.parametrize("serialized", [False, True])
def test_require_existing_authority_rejects_incomplete_schema_without_repair(tmp_path: Path, serialized: bool) -> None:
    import duckdb
    path = tmp_path / "malformed.duckdb"
    with duckdb.connect(str(path), config={"threads": "1", "memory_limit": "256MB", "enable_external_access": "false"}) as connection:
        connection.execute("CREATE TABLE unrelated(value INTEGER)")
        connection.execute("INSERT INTO unrelated VALUES (7)")
    before_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    factory = open_process_serialized_database_coordinator if serialized else open_database_coordinator
    with pytest.raises(DatabaseCoordinationError):
        factory(path, require_existing_authority=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before_digest
    with duckdb.connect(str(path), read_only=True, config={"threads": "1", "memory_limit": "256MB", "enable_external_access": "false"}) as connection:
        assert connection.execute("SHOW TABLES").fetchall() == [("unrelated",)]
        assert connection.execute("SELECT value FROM unrelated").fetchall() == [(7,)]


@pytest.mark.parametrize("serialized", [False, True])
def test_require_existing_authority_opens_actual_native_registry(tmp_path: Path, serialized: bool) -> None:
    clock = Clock()
    path = tmp_path / "existing.duckdb"
    with open_database_coordinator(path, clock_ms=clock) as creator:
        claim = _claim(creator)
    factory = open_process_serialized_database_coordinator if serialized else open_database_coordinator
    with factory(path, clock_ms=clock, require_existing_authority=True) as guard:
        assert _execute(guard, claim, lambda protected_lease: "existing") == "existing"


@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize("corruption", ["missing_index", "schema_identity"])
def test_require_existing_authority_does_not_repair_damaged_native_registry(
    tmp_path: Path, serialized: bool, corruption: str
) -> None:
    import duckdb
    path = tmp_path / "damaged-native.duckdb"
    with open_database_coordinator(path) as creator:
        _claim(creator)
    with duckdb.connect(str(path), config={"threads": "1", "memory_limit": "256MB", "enable_external_access": "false"}) as connection:
        if corruption == "missing_index":
            connection.execute("DROP INDEX task_claims_task_idx")
        else:
            connection.execute(
                "UPDATE coordination_metadata SET value = ? WHERE key = 'schema'",
                ["unreviewed-schema@999"],
            )
    before_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    factory = open_process_serialized_database_coordinator if serialized else open_database_coordinator
    with pytest.raises(DatabaseCoordinationStaleFenceError):
        factory(path, require_existing_authority=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before_digest
    with duckdb.connect(str(path), read_only=True, config={"threads": "1", "memory_limit": "256MB", "enable_external_access": "false"}) as connection:
        if corruption == "missing_index":
            assert connection.execute(
                "SELECT index_name FROM duckdb_indexes() WHERE index_name = 'task_claims_task_idx'"
            ).fetchall() == []
        else:
            assert connection.execute(
                "SELECT value FROM coordination_metadata WHERE key = 'schema'"
            ).fetchone() == ("unreviewed-schema@999",)


@pytest.mark.parametrize("serialized", [False, True])
def test_require_existing_authority_rejects_symlink_alias(tmp_path: Path, serialized: bool) -> None:
    path = tmp_path / "authority.duckdb"
    with open_database_coordinator(path):
        pass
    alias = tmp_path / "alias.duckdb"
    alias.symlink_to(path)
    before_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    factory = open_process_serialized_database_coordinator if serialized else open_database_coordinator
    with pytest.raises(DatabaseCoordinationStaleFenceError):
        factory(alias, require_existing_authority=True)
    assert alias.is_symlink()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before_digest
    assert not tmp_path.joinpath(".alias.duckdb.lock").exists()
    assert not tmp_path.joinpath(".alias.duckdb.serialized-coordinator.lock").exists()
