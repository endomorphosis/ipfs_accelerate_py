"""Real typed owner and native filtered consumer, with no second queue writer."""

import json
from dataclasses import replace

import pytest

from test.api.test_agent_supervisor_owner_merge_queue import (
    owner as owner_fixture,
    enqueue,
    request,
)
from ipfs_accelerate_py.agent_supervisor.merge import owner_merge_queue as port
from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import MergeQueueFenceError
from ipfs_accelerate_py.agent_supervisor.merge.merge_train import MergeTrain
from ipfs_accelerate_py.agent_supervisor.merge.owner_merge_queue_adapter import (
    OwnerMergeQueueAdapter,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    TypedStateOwnerError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    ControlPlaneBoundsError,
)

owner = owner_fixture


def snapshot(api, operation="pending_requests", **overrides):
    return api.call(operation, **{"limit": 32, "after_request_id": None, **overrides})


def all_rows(conn):
    return [
        tuple(row[i] for i in range(len(row)))
        for row in conn.execute(
            "SELECT * FROM merge_requests ORDER BY request_id"
        ).fetchall()
    ]


def test_snapshot_exact_target_pages_and_no_expired_claim_reaping(owner, monkeypatch):
    gateway, conn, bind, attach, retained, *_ = owner
    bind()
    api, _ = attach()
    items = [request(enqueue(api, task=f"task:page:{i}")) for i in range(5)]
    conn.execute("UPDATE merge_requests SET claimed_at=1 WHERE status='processing'")
    before = all_rows(conn)
    monkeypatch.setattr(
        port._OwnerQueue, "_purge_stale", lambda _: pytest.fail("read reaper")
    )
    seen = []
    cursor = ""
    while True:
        result = snapshot(api, limit=2, after_request_id=cursor)
        assert result["completion_authority"] is False
        rows = json.loads(result["requests_json"])
        assert len(rows) <= 2
        if not rows:
            break
        seen.extend(row["request_id"] for row in rows)
        cursor = rows[-1]["request_id"]
    assert seen == sorted(item["request_id"] for item in items)
    processing = json.loads(snapshot(api, "processing_requests")["requests_json"])
    assert [row["request_id"] for row in processing] == [retained[2].request_id]
    assert processing[0]["claim_token"] == retained[2].claim_token
    assert all_rows(conn) == before
    assert not conn.in_transaction


@pytest.mark.parametrize(
    "overrides",
    [
        {"limit": True},
        {"limit": 0},
        {"limit": 257},
        {"limit": 2.0},
        {"after_request_id": []},
        {"after_request_id": " bad"},
        {"sql": "SELECT 1"},
    ],
)
def test_snapshot_rejects_invalid_bounds_and_open_fields(owner, overrides):
    _, conn, bind, attach, *_ = owner
    bind()
    api, _ = attach()
    before = all_rows(conn)
    with pytest.raises((TypedStateOwnerError, ControlPlaneBoundsError)):
        snapshot(api, **overrides)
    assert all_rows(conn) == before


@pytest.mark.parametrize(
    "operation", ["pending_requests", "processing_requests", "defer"]
)
def test_new_operations_require_separate_explicit_grants(owner, operation):
    _, conn, bind, attach, *_ = owner
    bind()
    api, _ = attach(
        operations=port.SERVICE_OPERATIONS - {"legacy.merge_queue." + operation}
    )
    before = all_rows(conn)
    with pytest.raises(TypedStateOwnerError):
        api.call(operation, limit=1, after_request_id=None)
    assert all_rows(conn) == before


def test_snapshot_denies_malformed_preserved_dedupe_without_mutation(owner):
    _, conn, bind, attach, *_ = owner
    bind()
    api, _ = attach()
    row = request(enqueue(api))
    conn.execute(
        "UPDATE merge_requests SET dedupe_key='malformed' WHERE request_id=?",
        [row["request_id"]],
    )
    before = all_rows(conn)
    with pytest.raises(TypedStateOwnerError):
        snapshot(api)
    assert all_rows(conn) == before


def test_snapshot_byte_limit_denies_instead_of_silent_truncation(owner, monkeypatch):
    _, conn, bind, attach, *_ = owner
    bind()
    api, _ = attach()
    enqueue(api)
    before = all_rows(conn)
    monkeypatch.setattr(port, "MAX_SNAPSHOT_BYTES", 16)
    with pytest.raises(TypedStateOwnerError):
        snapshot(api)
    assert all_rows(conn) == before


@pytest.mark.parametrize("change", ["revoke", "expire", "detach"])
def test_snapshot_revalidates_admission_after_read(owner, monkeypatch, change):
    gateway, conn, bind, attach, *_ = owner
    bind()
    api, grant = attach()
    enqueue(api)
    native = gateway._legacy_merge_queue_service._queue.pending_requests

    def read_then_change(**kwargs):
        rows = native(**kwargs)
        if change == "revoke":
            gateway.revoke_grant(grant.grant_id)
        elif change == "expire":
            with gateway._grants_lock:
                for token, value in tuple(gateway._grants.items()):
                    if value.grant_id == grant.grant_id:
                        gateway._grants[token] = replace(
                            grant, issued_at=1, expires_at=1001
                        )
        else:
            conn.execute(
                "UPDATE client_sessions SET status='detached' WHERE session_id=?",
                [api.connection.session_id],
            )
        return rows

    monkeypatch.setattr(
        gateway._legacy_merge_queue_service._queue, "pending_requests", read_then_change
    )
    before = all_rows(conn)
    with pytest.raises(TypedStateOwnerError):
        snapshot(api)
    assert all_rows(conn) == before


def test_native_filtered_consumer_claims_only_selected_task_and_defers_without_retry(
    owner, monkeypatch
):
    gateway, conn, bind, attach, retained, _, files, queue_dir = owner
    bind()
    api, _ = attach()
    queue = OwnerMergeQueueAdapter(api)
    queue.bind_target("repo:one", "main")
    ignored = queue.enqueue(
        branch_name="work/ignore", task_id="task:ignore", commit_sha="e" * 40
    )
    selected = queue.enqueue(
        branch_name="work/selected", task_id="task:selected", commit_sha="f" * 40
    )
    native = MergeTrain.__new__(MergeTrain)
    native.queue = queue
    native.owner_id = "consumer:one"
    native.request_filter = None
    claim = native._dequeue(request_filter=lambda row: row.task_id == "task:selected")
    assert claim.request_id == selected.request_id
    assert queue.get(ignored.request_id).status == "pending"
    assert queue.owns_claim(claim)
    assert queue.processing_requests()[-1].request_id == claim.request_id
    assert native._dequeue(request_filter=lambda row: row.task_id == "missing") is None
    deferred = queue.defer(claim, "checkout lock held", delay_seconds=60.25)
    assert (deferred.status, deferred.attempt, deferred.failure_count) == (
        "pending",
        1,
        0,
    )
    assert deferred.claim_generation == claim.claim_generation + 1
    assert not queue.owns_claim(claim)
    assert (
        native._dequeue(request_filter=lambda row: row.task_id == "task:selected")
        is None
    )
    with pytest.raises(TypedStateOwnerError):
        queue.complete(claim)
    service_queue = gateway._legacy_merge_queue_service._queue
    monkeypatch.setattr(service_queue, "_clock", lambda: deferred.retry_not_before + 1)
    reclaimed = native._dequeue(
        request_filter=lambda row: row.task_id == "task:selected"
    )
    assert reclaimed.claim_generation > deferred.claim_generation
    queue.complete(reclaimed, {"observational": True})
    assert queue.get(selected.request_id).status == "completed"
    assert files == {
        str(path.relative_to(queue_dir)): path.read_bytes()
        for path in queue_dir.rglob("*.json")
    }


@pytest.mark.parametrize("delay", [True, -1, 86400000, "2", None, {}, float("inf")])
def test_deferral_rejects_invalid_delay_preserving_claim(owner, delay):
    _, conn, bind, attach, *_ = owner
    bind()
    api, _ = attach()
    queue = OwnerMergeQueueAdapter(api)
    added = queue.enqueue(
        branch_name="work/one", task_id="task:one", commit_sha="e" * 40
    )
    claim = queue.claim_pending_request(added)
    before = all_rows(conn)
    with pytest.raises((TypedStateOwnerError, ValueError)):
        queue.defer(claim, delay_seconds=delay)
    assert all_rows(conn) == before
    assert queue.owns_claim(claim)


def test_adapter_rejects_consumer_target_and_attempt_changes(owner):
    _, conn, bind, attach, *_ = owner
    bind()
    api, _ = attach()
    queue = OwnerMergeQueueAdapter(api)
    before = all_rows(conn)
    with pytest.raises(MergeQueueFenceError):
        queue.dequeue(consumer_id="consumer:old")
    with pytest.raises(MergeQueueFenceError):
        queue.bind_target("repo:other", "main")
    with pytest.raises(port.OwnerMergeQueueError):
        queue.enqueue(branch_name="work/one", task_id="task:one", attempt=2)
    assert all_rows(conn) == before


def test_adapter_failure_uses_bounded_retry_then_quarantine(owner):
    _, conn, bind, attach, *_ = owner
    bind(max_attempts=2)
    api, _ = attach()
    queue = OwnerMergeQueueAdapter(api)
    added = queue.enqueue(
        branch_name="work/one", task_id="task:one", commit_sha="e" * 40
    )
    for _ in range(2):
        claim = queue.claim_pending_request(added.request_id)
        assert queue.fail(claim, "failed validation", retryable=True) is None
    assert queue.get(added.request_id).status == "quarantined"
    assert queue.claim_pending_request(added.request_id) is None
