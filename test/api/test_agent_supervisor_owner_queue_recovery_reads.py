"""Native recovery observations over a real owner, with no acceptance authority."""

from dataclasses import replace

import pytest

from test.api.test_agent_supervisor_owner_merge_queue import owner as owner_fixture
from test.api.test_agent_supervisor_owner_merge_queue_adapter import all_rows
from ipfs_accelerate_py.agent_supervisor.merge import owner_merge_queue as port
from ipfs_accelerate_py.agent_supervisor.merge.owner_merge_queue_adapter import (
    OwnerMergeQueueAdapter,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    TypedStateOwnerError,
)

owner = owner_fixture
NEW_OPERATIONS = ("completed_requests", "quarantined_requests", "has_pending_for_task")


def add(queue, name, **kwargs):
    return queue.enqueue(
        branch_name="work/" + name,
        task_id="task:" + name,
        commit_sha="a" * 40,
        **kwargs,
    )


def test_active_task_check_retains_cooldown_expired_claim_and_exact_target(
    owner, monkeypatch
):
    gateway, conn, bind, attach, retained, *_ = owner
    bind()
    api, _ = attach()
    queue = OwnerMergeQueueAdapter(api)
    pending = add(
        queue,
        "active",
        canonical_task_id="Canonical:ACTIVE",
        canonical_task_key="key:ACTIVE",
    )
    claimed = queue.claim_pending_request(pending)
    queue.defer(claimed, "busy", delay_seconds=1000)
    monkeypatch.setattr(
        port._OwnerQueue, "_purge_stale", lambda _: pytest.fail("reaper")
    )
    assert queue.pending_requests() == ()
    conn.execute(
        "UPDATE merge_requests SET claimed_at=1 WHERE request_id=?",
        [retained[2].request_id],
    )
    before = all_rows(conn)
    for identity in ("task:ACTIVE", "canonical:active", "key:active", "task:retained"):
        assert queue.has_pending_for_task(identity)
    assert queue.has_pending_for_task("task:active", commit_sha="A" * 40)
    assert not queue.has_pending_for_task("task:active", commit_sha="b" * 40)
    for identity in ("task:other", "task:unbound", "missing"):
        assert not queue.has_pending_for_task(identity)
    assert all_rows(conn) == before
    assert not conn.in_transaction


@pytest.mark.parametrize("ordered_by_request_id", [False, True])
def test_terminal_pages_preserve_filter_before_limit_and_native_keysets(
    owner, monkeypatch, ordered_by_request_id
):
    _, conn, bind, attach, retained, *_ = owner
    bind()
    api, _ = attach()
    queue = OwnerMergeQueueAdapter(api)
    completed = []
    quarantined = []
    for i in range(6):
        row = add(queue, f"completed-{i}", metadata={"schema": "candidate@3"})
        claim = queue.claim_pending_request(row)
        queue.complete(
            claim,
            {"schema": "accepted" if i % 2 else "irrelevant", "reason": "verified"},
        )
        completed.append(row.request_id)
        row = add(queue, f"quarantined-{i}")
        queue.quarantine(queue.claim_pending_request(row), "rejected")
        quarantined.append(row.request_id)
    # Other targets and unbound legacy rows must never appear, even terminal.
    conn.execute(
        "UPDATE merge_requests SET status='completed' WHERE request_id IN (?, ?)",
        [retained[0].request_id, retained[1].request_id],
    )
    before = all_rows(conn)
    monkeypatch.setattr(
        port._OwnerQueue, "_purge_stale", lambda _: pytest.fail("reaper")
    )
    seen = []
    cursor = ""
    while True:
        rows = queue.completed_requests(
            limit=1,
            before_request_id=cursor,
            completion_schema="accepted",
            ordered_by_request_id=ordered_by_request_id,
            completion_reason="verified",
            metadata_schema="candidate@3",
        )
        if not rows:
            break
        seen.extend(row.request_id for row in rows)
        cursor = rows[-1].request_id
    assert seen == sorted(completed[1::2], reverse=True)
    assert queue.completed_requests(require_completion_absent=True) == ()
    seen = []
    cursor = ""
    while True:
        rows = queue.quarantined_requests(limit=2, after_request_id=cursor)
        if not rows:
            break
        seen.extend(row.request_id for row in rows)
        cursor = rows[-1].request_id
    assert seen == sorted(quarantined)
    assert all_rows(conn) == before
    assert not conn.in_transaction


@pytest.mark.parametrize("operation", NEW_OPERATIONS)
def test_recovery_reads_need_separate_grants(owner, operation):
    _, conn, bind, attach, *_ = owner
    bind()
    api, _ = attach(
        operations=port.SERVICE_OPERATIONS - {"legacy.merge_queue." + operation}
    )
    before = all_rows(conn)
    queue = OwnerMergeQueueAdapter(api)
    with pytest.raises(TypedStateOwnerError):
        if operation == "has_pending_for_task":
            queue.has_pending_for_task("task:retained")
        else:
            getattr(queue, operation)()
    assert all_rows(conn) == before


@pytest.mark.parametrize("operation", NEW_OPERATIONS)
@pytest.mark.parametrize("change", ("revoke", "expire", "detach"))
def test_recovery_reads_recheck_admission_after_read(
    owner, monkeypatch, operation, change
):
    gateway, conn, bind, attach, *_ = owner
    bind()
    api, grant = attach()
    queue = OwnerMergeQueueAdapter(api)
    native = getattr(gateway._legacy_merge_queue_service._queue, operation)

    def read_then_change(*args, **kwargs):
        result = native(*args, **kwargs)
        if change == "revoke":
            gateway.revoke_grant(grant.grant_id)
        elif change == "expire":
            with gateway._grants_lock:
                for token, current in tuple(gateway._grants.items()):
                    if current.grant_id == grant.grant_id:
                        gateway._grants[token] = replace(
                            grant, issued_at=1, expires_at=1001
                        )
        else:
            conn.execute(
                "UPDATE client_sessions SET status='detached' WHERE session_id=?",
                [api.connection.session_id],
            )
        return result

    monkeypatch.setattr(
        gateway._legacy_merge_queue_service._queue, operation, read_then_change
    )
    before = all_rows(conn)
    with pytest.raises(TypedStateOwnerError):
        if operation == "has_pending_for_task":
            queue.has_pending_for_task("task:retained")
        else:
            getattr(queue, operation)()
    assert all_rows(conn) == before


@pytest.mark.parametrize("operation", NEW_OPERATIONS)
def test_recovery_reads_deny_malformed_preserved_identity(owner, operation):
    _, conn, bind, attach, *_ = owner
    bind()
    api, _ = attach()
    queue = OwnerMergeQueueAdapter(api)
    row = add(queue, "malformed")
    status = {
        "completed_requests": "completed",
        "quarantined_requests": "quarantined",
        "has_pending_for_task": "pending",
    }[operation]
    conn.execute(
        "UPDATE merge_requests SET status=?, dedupe_key=? WHERE request_id=?",
        [status, "bad", row.request_id],
    )
    before = all_rows(conn)
    with pytest.raises(TypedStateOwnerError):
        if operation == "has_pending_for_task":
            queue.has_pending_for_task(row.task_id)
        else:
            getattr(queue, operation)()
    assert all_rows(conn) == before


@pytest.mark.parametrize(
    "kwargs",
    (
        {"require_completion_absent": 1},
        {"ordered_by_request_id": "yes"},
        {"before_request_id": []},
        {"metadata_schema": " bad"},
        {"limit": 257},
    ),
)
def test_completion_page_rejects_non_native_fields(owner, kwargs):
    _, conn, bind, attach, *_ = owner
    bind()
    api, _ = attach()
    before = all_rows(conn)
    with pytest.raises(TypedStateOwnerError):
        OwnerMergeQueueAdapter(api).completed_requests(**kwargs)
    assert all_rows(conn) == before
