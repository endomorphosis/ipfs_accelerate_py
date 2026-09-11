"""Owner-side quarantine preserves task bytes and survives hostile bundle clients."""

from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources import owner_task_quarantine as q
from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as ds
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    QuackStateServerMutationError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    canonical_json_bytes,
    content_identity,
)
from test.api.test_agent_supervisor_quack_owner_mutation import (
    _server,
    _signed_request,
    _transition_request,
)


def quarantine_request(server, token, *, prior=None, revoke=False, **changes):
    c = server._connection
    task_revision = c.execute(
        "SELECT revision FROM tasks WHERE task_cid = 'task:test'"
    ).fetchone()[0]
    body = {
        "schema": q.SCHEMA,
        "state": "revoked" if revoke else "active",
        "revision": prior["revision"] + 1 if prior else 1,
        "previous_event_id": prior["event_id"] if prior else "",
        "task_cid": "task:test",
        "task_revision": task_revision,
        "workspace_custody_cid": "workspace:custody",
        "workspace_root": "/test/old-workspaces",
        "fresh_workspace_root": "/test/fresh-workspaces",
        "attempt_id": "attempt:retained",
        "execution_store_id": "execution:retained",
        "execution_owner_id": "owner:retained",
        "retained_state_cid": "state:retained",
        "owner_binding": server._mutation_binding(),
        "diagnosis": "entered_callback_state_missing",
        **changes,
    }
    seq = (
        c.execute(
            "SELECT COALESCE(MAX(sequence),0) FROM domain_events WHERE stream_id='stream:intent'"
        ).fetchone()[0]
        + 1
    )
    global_seq = (
        c.execute(
            "SELECT COALESCE(MAX(global_sequence),0) FROM domain_events"
        ).fetchone()[0]
        + 1
    )
    at = "2026-09-11T00:00:00Z"
    envelope = {
        "schema": "ipfs_accelerate_py/agent-supervisor/intent-event@1",
        "event_type": q.EVENT,
        "subject_id": "task:test",
        "body": body,
        "recorded_at": at,
        "owner_id": "owner:retained",
    }
    event_id = content_identity(
        {
            "stream_id": "stream:intent",
            "sequence": seq,
            "global_sequence": global_seq,
            "event_type": q.EVENT,
            "body": envelope,
        }
    )
    event = [
        event_id,
        "stream:intent",
        seq,
        global_seq,
        q.EVENT,
        body["task_cid"],
        body["attempt_id"],
        "session:test",
        at,
        canonical_json_bytes(envelope).decode(),
    ]
    steps = [
        {"template_id": ds.QUACK_MUTATION_DOMAIN_EVENT_INSERT, "parameters": event},
        {
            "template_id": ds.QUACK_MUTATION_QUARANTINE_ANCHOR,
            "parameters": [
                q.ANCHOR_KEY,
                canonical_json_bytes(q.next_anchor(c, event_id)).decode(),
                at,
            ],
        },
    ]
    return _signed_request(server, token, operation=q.OPERATION, steps=steps)


def apply(server, request):
    valid = server._validate_mutation_request(request, request_id=request["request_id"])
    return server._execute_mutation_request(valid)


def task_snapshot(server):
    return {
        name: server._connection.execute(f"SELECT * FROM {name} ORDER BY 1").fetchall()
        for name in (
            "tasks",
            "task_revisions",
            "completion_receipts",
            "leases",
            "validation_runs",
            "validation_results",
        )
    }


def test_owner_bundle_preserves_rows_blocks_foreign_status_and_replays(tmp_path):
    server, _, token, _ = _server(tmp_path)
    try:
        before = task_snapshot(server)
        request = quarantine_request(server, token)
        assert ds._quack_mutation_operation(request["steps"]) == q.OPERATION
        apply(server, request)
        assert task_snapshot(server) == before
        head = q.heads(server._connection)["attempt:retained"]
        assert head["state"] == "active"
        apply(server, request)
        assert task_snapshot(server) == before
        import copy

        tampered_steps = copy.deepcopy(request["steps"])
        tampered_steps[1]["parameters"][1] = "{}"
        tampered_replay = _signed_request(
            server, token, operation=q.OPERATION, steps=tampered_steps
        )
        with pytest.raises(
            QuackStateServerMutationError, match="replay_integrity_failure"
        ):
            apply(server, tampered_replay)
        assert task_snapshot(server) == before
        forged_lane = _transition_request(
            server, token, session_id="foreign-lane", owner_id="foreign-lane"
        )
        with pytest.raises(
            QuackStateServerMutationError, match="task_custody_quarantined"
        ):
            apply(server, forged_lane)
        assert task_snapshot(server) == before
    finally:
        server.stop()


@pytest.mark.parametrize(
    "change",
    [
        {"task_revision": 99},
        {"owner_binding": {}},
        {"previous_event_id": "missing"},
        {"revision": True},
        {"diagnosis": "operator says safe"},
        {"extra": "forged"},
    ],
)
def test_owner_rejects_unbound_quarantine_without_effects(tmp_path, change):
    server, _, token, _ = _server(tmp_path)
    try:
        before = task_snapshot(server)
        with pytest.raises((q.QuarantineDenied, QuackStateServerMutationError)):
            apply(server, quarantine_request(server, token, **change))
        assert task_snapshot(server) == before
        assert q.heads(server._connection) == {}
    finally:
        server.stop()


@pytest.mark.parametrize(
    "fault",
    ["delete_event", "delete_anchor", "rewrite_event", "oversize", "bad_anchor"],
)
def test_missing_or_corrupt_population_is_globally_closed(tmp_path, fault):
    server, _, token, _ = _server(tmp_path)
    try:
        apply(server, quarantine_request(server, token))
        c = server._connection
        if fault == "delete_event":
            c.execute("DELETE FROM domain_events WHERE event_type = ?", [q.EVENT])
        elif fault == "delete_anchor":
            c.execute(
                "DELETE FROM control_plane_metadata WHERE key = ?", [q.ANCHOR_KEY]
            )
        elif fault == "rewrite_event":
            c.execute(
                "UPDATE domain_events SET event_type = 'hidden' WHERE event_type = ?",
                [q.EVENT],
            )
        elif fault == "oversize":
            c.execute(
                "UPDATE domain_events SET body_json = ? WHERE event_type = ?",
                ["x" * (q.MAX_BYTES + 1), q.EVENT],
            )
        else:
            c.execute(
                "UPDATE control_plane_metadata SET value = '{}' WHERE key = ?",
                [q.ANCHOR_KEY],
            )
        with pytest.raises(q.QuarantineDenied):
            q.assert_task_unfenced(c, "a-different-independent-task")
    finally:
        server.stop()


def test_revocation_cannot_release_retained_task_or_reactivate(tmp_path):
    server, _, token, _ = _server(tmp_path)
    try:
        before = task_snapshot(server)
        apply(server, quarantine_request(server, token))
        prior = q.heads(server._connection)["attempt:retained"]
        apply(server, quarantine_request(server, token, prior=prior, revoke=True))
        revoked = q.heads(server._connection)["attempt:retained"]
        assert revoked["state"] == "revoked"
        with pytest.raises(q.QuarantineDenied):
            q.assert_task_unfenced(server._connection, "task:test")
        with pytest.raises((q.QuarantineDenied, QuackStateServerMutationError)):
            apply(server, quarantine_request(server, token, prior=revoked))
        assert task_snapshot(server) == before
    finally:
        server.stop()


def test_full_history_cannot_issue_another_anchor_or_forget_its_fence(
    tmp_path, monkeypatch
):
    server, _, token, _ = _server(tmp_path)
    try:
        apply(server, quarantine_request(server, token))
        connection = server._connection
        before = q.heads(connection)
        # The last available history slot can remain valid, but it cannot
        # authorize truncation or an additional native quarantine revision.
        monkeypatch.setattr(q, "MAX_HISTORY", 1)
        assert q.heads(connection) == before
        with pytest.raises(q.QuarantineDenied, match="quarantine_population_bound"):
            q.next_anchor(connection, "event:another")
        with pytest.raises(q.QuarantineDenied, match="task_custody_quarantined"):
            q.assert_task_unfenced(connection, "task:test")
        assert q.heads(connection) == before
    finally:
        server.stop()
