"""Native file-queue tests for exact CodebaseIR recovery without scans."""
from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py.p2p_tasks.codebase_federated_dispatch import (
    TASK_TYPE, CodebaseDispatchError, CodebaseQueueDispatcher,
    codebase_dispatch_request_sha256,
)
from ipfs_accelerate_py.p2p_tasks.codebase_federated_history import find_codebase_federated_task
from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue


@pytest.fixture
def queue(tmp_path):
    native = TaskQueue(str(tmp_path / "history.duckdb"))
    try:
        yield native
    finally:
        native.close()


@pytest.fixture
def payload():
    return {"schema": "codebase-federated-artifact-request@1",
            "declaration": {"work_binding": {"client_id": "client-1", "attempt": 2, "fence": 7},
                            "artifacts": {"round": {"sha256": "a" * 64}}}}


def _request(payload, model="codebase-model"):
    return codebase_dispatch_request_sha256(payload, model)


def _find(queue, payload, model="codebase-model"):
    return find_codebase_federated_task(queue, request_sha256=_request(payload, model),
                                      payload=payload, model_name=model)


def _forbid_mutation(monkeypatch, queue):
    def forbidden(*args, **kwargs):
        pytest.fail("recovery scanned, submitted or mutated the native queue")
    for name in ("list", "submit", "submit_once", "submit_with_outcome", "claim", "claim_next",
                 "claim_next_many", "complete", "update", "retry", "recover_expired_leases",
                 "heartbeat", "cancel", "release", "delete", "prune_terminal", "_get_conn"):
        monkeypatch.setattr(queue, name, forbidden)


@pytest.mark.parametrize("unrelated", [1001, 10000])
def test_exact_index_recovers_after_native_inventory_bound(queue, payload, monkeypatch, unrelated):
    # Populate a real indexed native table efficiently; the unrelated tasks are
    # older and have the same type, so the visibility API always omits our task.
    with queue._conn_lock:
        queue._get_conn().execute(
            "INSERT INTO tasks(task_id, task_type, model_name, payload_json, status, "
            "assigned_worker, created_at, updated_at, result_json, attempt, idempotency_key) "
            "SELECT 'old:' || CAST(i AS VARCHAR), ?, 'other-model', '{}', 'completed', "
            "'other-worker', i, i, '{}', 1, 'other:' || CAST(i AS VARCHAR) FROM range(?) AS t(i)",
            [TASK_TYPE, unrelated],
        )
    calls = []
    dispatcher = CodebaseQueueDispatcher(queue, lambda task: calls.append(task) or {"retained": True})
    assert dispatcher.dispatch(payload, "codebase-model") == {"retained": True}
    target = queue.get(calls[0]["task_id"])
    inventory = queue.list(status="completed", task_types=[TASK_TYPE], limit=10000)
    assert len(inventory) == 1000 and target["task_id"] not in {row["task_id"] for row in inventory}
    statements, closed = [], []
    connect = queue._connect

    class SelectOnlyConnection:
        def __init__(self):
            self.native = connect()

        def execute(self, sql, params=()):
            statements.append((sql, params))
            assert sql.startswith("SELECT ")
            return self.native.execute(sql, params)

        def close(self):
            self.native.close()
            closed.append(True)

    _forbid_mutation(monkeypatch, queue)
    monkeypatch.setattr(queue, "_connect", SelectOnlyConnection)
    assert _find(queue, payload) == target
    assert len(statements) == 2 and len(closed) == 2
    assert statements[0] == (
        "SELECT task_id FROM tasks WHERE idempotency_key=? LIMIT 2", (_request(payload),))
    assert "WHERE task_id = ?" in statements[1][0]
    assert not any("ORDER BY" in sql for sql, _ in statements)
    assert len(calls) == 1


def test_absent_request_does_not_submit_or_change_queue(queue, payload, monkeypatch):
    _forbid_mutation(monkeypatch, queue)
    assert _find(queue, payload) is None
    assert queue.count() == 0


def test_detached_completed_row_survives_native_queue_restart(queue, payload):
    calls = []
    dispatcher = CodebaseQueueDispatcher(queue, lambda task: calls.append(task) or {"nested": {"step": 1}})
    dispatcher.dispatch(payload, "codebase-model")
    expected = queue.get(calls[0]["task_id"])
    found = _find(queue, payload)
    found["payload"]["schema"] = "caller-change"
    found["result"]["nested"]["step"] = 100
    assert queue.get(expected["task_id"]) == expected
    queue.close()
    reopened = TaskQueue(queue.path)
    try:
        assert _find(reopened, payload) == expected
        assert reopened.get(expected["task_id"]) == expected
    finally:
        reopened.close()
    assert len(calls) == 1


@pytest.mark.parametrize("status", ["queued", "running", "completed", "failed", "cancelled"])
def test_lookup_preserves_native_state_and_attempt_without_claiming(queue, payload, status):
    task_id = queue.submit_once(idempotency_key=_request(payload), task_type=TASK_TYPE,
                                model_name="codebase-model", payload=payload)
    if status in {"running", "completed", "failed"}:
        queue.claim(task_id=task_id, worker_id="retained-worker")
        if status in {"completed", "failed"}:
            queue.complete(task_id=task_id, worker_id="retained-worker", status=status,
                           result={"retained": True} if status == "completed" else None,
                           error="retained failure" if status == "failed" else None)
    if status == "cancelled":
        assert queue.cancel(task_id=task_id, reason="retained cancel")
    expected = queue.get(task_id)
    assert expected["status"] == status
    assert _find(queue, payload) == expected
    assert _find(queue, payload) == expected
    assert queue.get(task_id) == expected


@pytest.mark.parametrize("bad", [None, 3, True, "", "A" * 64, "a" * 63, "a" * 65, "a" * 64 + "\n"])
def test_invalid_request_identity_rejected_before_native_read(queue, payload, monkeypatch, bad):
    monkeypatch.setattr(queue, "_connect", lambda: pytest.fail("invalid request reached the database"))
    with pytest.raises(CodebaseDispatchError, match="request SHA-256"):
        find_codebase_federated_task(queue, request_sha256=bad, payload=payload, model_name="model")


@pytest.mark.parametrize("bad", [{1: "stringified"}, {"value": (1, 2)}, {"value": float("nan")},
                                 {"value": "x" * (1024 * 1024)}])
def test_non_native_or_oversized_payload_rejected_before_native_read(queue, monkeypatch, bad):
    monkeypatch.setattr(queue, "_connect", lambda: pytest.fail("invalid request reached the database"))
    with pytest.raises(CodebaseDispatchError):
        find_codebase_federated_task(queue, request_sha256="a" * 64, payload=bad, model_name="model")


@pytest.mark.parametrize("bad", [None, 12, "", " leading", "model\n", "m" * 513])
def test_invalid_model_rejected_before_native_read(queue, payload, monkeypatch, bad):
    monkeypatch.setattr(queue, "_connect", lambda: pytest.fail("invalid model reached the database"))
    with pytest.raises(CodebaseDispatchError):
        find_codebase_federated_task(queue, request_sha256=_request(payload), payload=payload, model_name=bad)


def test_wrong_full_request_commitment_rejected_before_read(queue, payload, monkeypatch):
    monkeypatch.setattr(queue, "_connect", lambda: pytest.fail("wrong request reached the database"))
    with pytest.raises(CodebaseDispatchError, match="full task/model/payload"):
        find_codebase_federated_task(queue, request_sha256="0" * 64, payload=payload, model_name="model")


def test_nonnative_or_remote_queue_rejected_without_connection(payload):
    remote = object.__new__(TaskQueue)
    remote._quack = True
    for bad in (object(), remote):
        with pytest.raises(CodebaseDispatchError, match="local native TaskQueue"):
            _find(bad, payload)


@pytest.mark.parametrize("field,bad", [("task_type", "other-task"), ("model_name", "other-model"),
                                      ("payload_json", '{"schema":"different"}')])
def test_indexed_row_with_different_scope_is_rejected(queue, payload, field, bad):
    task_id = queue.submit_once(idempotency_key=_request(payload), task_type=TASK_TYPE,
                                model_name="codebase-model", payload=payload)
    with queue._conn_lock:
        queue._get_conn().execute("UPDATE tasks SET " + field + "=? WHERE task_id=?", [bad, task_id])
    with pytest.raises(CodebaseDispatchError, match="exact request scope"):
        _find(queue, payload)


def test_malformed_native_payload_is_rejected(queue, payload):
    task_id = queue.submit_once(idempotency_key=_request(payload), task_type=TASK_TYPE,
                                model_name="codebase-model", payload=payload)
    with queue._conn_lock:
        queue._get_conn().execute("UPDATE tasks SET payload_json='invalid-json' WHERE task_id=?", [task_id])
    with pytest.raises(CodebaseDispatchError, match="malformed"):
        _find(queue, payload)


@pytest.mark.parametrize("change", ["missing", "key", "task_id", "extra"])
def test_read_between_index_and_native_get_cannot_alias_changed_row(queue, payload, monkeypatch, change):
    task_id = queue.submit_once(idempotency_key=_request(payload), task_type=TASK_TYPE,
                                model_name="codebase-model", payload=payload)
    row = queue.get(task_id)
    if change == "key":
        row["idempotency_key"] = "0" * 64
    elif change == "task_id":
        row["task_id"] = "another-native-task"
    elif change == "extra":
        row["unexpected"] = True
    monkeypatch.setattr(queue, "get", lambda observed: None if change == "missing" else row)
    with pytest.raises(CodebaseDispatchError, match="disappeared or changed identity"):
        _find(queue, payload)


def test_duplicate_corrupt_native_identity_is_rejected(queue, payload):
    task_id = queue.submit_once(idempotency_key=_request(payload), task_type=TASK_TYPE,
                                model_name="codebase-model", payload=payload)
    with queue._conn_lock:
        connection = queue._get_conn()
        connection.execute("DROP INDEX idx_tasks_idempotency")
        connection.execute(
            "INSERT INTO tasks(task_id, task_type, model_name, payload_json, status, created_at, "
            "updated_at, idempotency_key) SELECT 'duplicate', task_type, model_name, payload_json, "
            "status, created_at, updated_at, idempotency_key FROM tasks WHERE task_id=?", [task_id])
    with pytest.raises(CodebaseDispatchError, match="duplicate native request"):
        _find(queue, payload)


def test_fresh_lookup_connection_closes_on_query_failure(queue, payload, monkeypatch):
    closed = []

    class BrokenRead:
        def execute(self, sql, params):
            raise RuntimeError("native read interrupted")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(queue, "_connect", BrokenRead)
    with pytest.raises(RuntimeError, match="native read interrupted"):
        _find(queue, payload)
    assert closed == [True]


def test_native_nonfinite_result_is_rejected_as_inert_json(queue, payload):
    task_id = queue.submit_once(idempotency_key=_request(payload), task_type=TASK_TYPE,
                                model_name="codebase-model", payload=payload)
    with queue._conn_lock:
        queue._get_conn().execute("UPDATE tasks SET result_json=? WHERE task_id=?",
                                  [json.dumps({"bad": float("nan")}), task_id])
    with pytest.raises(CodebaseDispatchError, match="finite"):
        _find(queue, payload)
