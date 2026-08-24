"""Closed owner-mutation protocol tests for the Quack state owner."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime import quack_state_server as quack_server_module
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    MUTATION_REQUEST_NAME,
    QUACK_ISOLATION_RECEIPT_SCHEMA,
    FakeQuackTransport,
    QuackStateServerIsolationError,
    build_server,
)
from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as duckdb_state_module
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    canonical_json_bytes,
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_transactions import (
    TransactionError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
    TaskSourceConflictError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    _QUACK_OWNER_MUTATION_SQL_TO_TEMPLATE,
    QUACK_MUTATION_COMPLETION_RECEIPT_INSERT,
    QUACK_MUTATION_DOMAIN_EVENT_INSERT,
    QUACK_MUTATION_LEASE_QUEUE_BACKOFF_INSERT,
    QUACK_MUTATION_LEASE_QUEUE_BACKOFF_UPDATE,
    QUACK_MUTATION_QUEUE_BACKOFF,
    QUACK_MUTATION_TASK_REVISION_INSERT,
    QUACK_MUTATION_TASK_STATUS_CAS,
    QUACK_MUTATION_TASK_STATUS_TRANSITION,
    QUACK_OWNER_MUTATION_MAX_PARAMETERS,
    QUACK_OWNER_MUTATION_PROTOCOL_REVISION,
    QUACK_OWNER_MUTATION_REQUEST_SCHEMA,
    QUACK_OWNER_MUTATION_REQUEST_TTL_MS,
    DuckDBConnection,
    DuckDBConnectionPolicyError,
    _normalize_quack_mutation_sql,
    _quack_mutation_operation,
    _validate_quack_mutation_parameters,
    _validate_quack_mutation_result,
    quack_owner_mutation_content_id,
    quack_owner_mutation_mac,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    COMPLETION_EVIDENCE_SCHEMA,
    QUEUE_ENTRY_SCHEMA,
    open_intent_repository,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
    probe_quack_capabilities,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
    QuackStateClient,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_database_task_source import (
    TypedDatabaseTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA,
    TYPED_DATABASE_CLAIM_RESERVATION_SCHEMA,
    TYPED_STATE_OWNER_SOCKET_ENV,
    TYPED_STATE_OWNER_TOKEN_ENV,
    build_control_plane_operation_catalog,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(), reason="DuckDB is required for owner mutation tests"
)


def _seed(database: Path) -> None:
    repo = open_intent_repository(database, owner_id="seed")
    try:
        repo.upsert_objective(
            objective_id="objective:test", objective_alias="O", title="Objective"
        )
        repo.upsert_goal(
            goal_cid="goal:test",
            goal_alias="G",
            title="Goal",
            objective_id="objective:test",
        )
        repo.upsert_plan(
            plan_cid="plan:test",
            goal_cid="goal:test",
            plan_alias="P",
        )
        repo.upsert_task(
            task_cid="task:test",
            task_alias="T",
            goal_cid="goal:test",
            plan_cid="plan:test",
            objective_id="objective:test",
            ordinal=1,
            status="ready",
        )
    finally:
        repo.close()


def _raw_quack_query(uri: str, token: str, sql: str) -> list[Any]:
    import duckdb

    client = duckdb.connect(":memory:")
    try:
        client.execute("LOAD quack")
        return client.execute(
            "SELECT * FROM quack_query(?, ?, token := ?, disable_ssl := true)",
            [uri, sql, token],
        ).fetchall()
    finally:
        client.close()


def _server(tmp_path: Path):
    database = tmp_path / "control.duckdb"
    _seed(database)
    server = build_server(
        database_path=database,
        state_dir=tmp_path / "state",
        store_id=str(database),
        repository_id="repository:test",
        transport=FakeQuackTransport(),
        capability_probe=lambda **_kwargs: probe_quack_capabilities(),
    )
    identity = server.start()
    token = server._vault.resolve(identity.secret_handle)  # noqa: SLF001
    return server, identity, token, database


def _typed_task_source(
    server: Any,
    identity: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    client_id: str,
    allowed_command_operations: tuple[str, ...],
    clock_ms: Any | None = None,
) -> TypedDatabaseTaskSource:
    """Attach one process-bound typed client without retaining its grant in env."""

    token, grant = server.issue_typed_client_grant_record(
        client_id=client_id,
        process_birth_id=identity.process_birth_id,
        allowed_operations=tuple(sorted(build_control_plane_operation_catalog())),
        allowed_command_operations=allowed_command_operations,
        peer_pid=os.getpid(),
    )
    assert grant.client_id == client_id
    assert grant.process_birth_id == identity.process_birth_id
    assert grant.peer_pid == os.getpid()
    monkeypatch.setenv(
        TYPED_STATE_OWNER_SOCKET_ENV,
        str(server.typed_command_socket_path()),
    )
    monkeypatch.setenv(TYPED_STATE_OWNER_TOKEN_ENV, token)
    client = QuackStateClient(
        owner_id=client_id,
        store_id=identity.store_id,
        process_birth_id=identity.process_birth_id,
    )
    try:
        client.attach(identity.listen_uri, server_id=identity.server_id)
    except Exception:
        client.close()
        raise
    finally:
        monkeypatch.delenv(TYPED_STATE_OWNER_TOKEN_ENV, raising=False)
    assert TYPED_STATE_OWNER_TOKEN_ENV not in os.environ
    return TypedDatabaseTaskSource(client, clock_ms=clock_ms)


def _typed_claim_receipt(
    source: TypedDatabaseTaskSource,
    *,
    lane: str,
    claimed_from_revision: int,
) -> dict[str, Any]:
    return {
        "operation": "database_claim",
        "claim_phase_schema": TYPED_DATABASE_CLAIM_RESERVATION_SCHEMA,
        "claim_process_attestation": dict(source.claim_process_attestation()),
        "claim_id": f"claim:{lane}",
        "attempt_id": f"attempt:{lane}",
        "attempt_number": 1,
        "lease_id": f"lease:{lane}",
        "owner_session_id": f"session:{lane}",
        "fencing_token": 1,
        "fence_epoch": 1,
        "claimed_from_revision": int(claimed_from_revision),
    }


def _isolation_receipt(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    owner_write_root = tmp_path / "control"
    database = owner_write_root / "control.duckdb"
    state = owner_write_root / "quack-owner"
    repository = tmp_path / "repository"
    state.mkdir(exist_ok=True)
    repository.mkdir(exist_ok=True)
    port_probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        port_probe.bind(("127.0.0.1", 0))
        published_port = int(port_probe.getsockname()[1])
    finally:
        port_probe.close()
    unsigned = {
        "schema": QUACK_ISOLATION_RECEIPT_SCHEMA,
        "runtime": "docker",
        "container_id": "a" * 64,
        "container_hostname": "ipfs-accelerate-pcpc-quack-owner-v1",
        "network_mode": "bridge",
        "container_bind_host": "0.0.0.0",
        "container_port": published_port,
        "published_host": "127.0.0.1",
        "published_port": published_port,
        "published_protocol": "tcp",
        "owner_write_root": str(owner_write_root.resolve()),
        "database_path": str(database.resolve()),
        "state_dir": str(state.resolve()),
        "repository_path": str(repository.resolve()),
        "allowed_rw_mount_targets": [str(owner_write_root.resolve())],
        "issuer": "test:isolation",
        "issued_at": "2026-08-20T20:00:00Z",
    }
    receipt = {**unsigned, "receipt_cid": content_identity(unsigned)}
    path = state / "isolation.json"
    path.write_bytes(canonical_json_bytes(receipt))
    path.chmod(0o600)
    return path, receipt


def _isolation_server_kwargs(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "host": receipt["published_host"],
        "port": receipt["published_port"],
        "container_bind_host": receipt["container_bind_host"],
        "container_port": receipt["container_port"],
    }


def _admitted_observation(_config: Any, receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/quack-owner-isolation-observation@1",
        "container_marker_regular": True,
        "container_id": receipt["container_id"],
        "container_hostname": receipt["container_hostname"],
        "root_read_only": True,
        "repository_read_only": True,
        "rw_host_bind_targets": sorted(receipt["allowed_rw_mount_targets"]),
        "docker_socket_absent": True,
        "host_proc_hidden": True,
        "private_home": True,
        "provider_auth_absent": True,
    }


def _signed_request(
    server: Any,
    token: str,
    *,
    operation: str,
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    binding = server._mutation_binding()  # noqa: SLF001
    semantic = {
        "schema": "ipfs_accelerate_py/agent-supervisor/quack-owner-mutation-semantic@1",
        "protocol_revision": QUACK_OWNER_MUTATION_PROTOCOL_REVISION,
        "operation": operation,
        "binding": binding,
        "steps": steps,
    }
    request_id = quack_owner_mutation_content_id(semantic)
    issued_at_ms = int(time.time() * 1000)
    unsigned = {
        "schema": QUACK_OWNER_MUTATION_REQUEST_SCHEMA,
        "protocol_revision": QUACK_OWNER_MUTATION_PROTOCOL_REVISION,
        "request_id": request_id,
        "issued_at_ms": issued_at_ms,
        "expires_at_ms": issued_at_ms + QUACK_OWNER_MUTATION_REQUEST_TTL_MS,
        "operation": operation,
        "binding": binding,
        "steps": steps,
    }
    request_cid = quack_owner_mutation_content_id(unsigned)
    authenticated = {**unsigned, "request_cid": request_cid}
    return {
        **authenticated,
        "auth_mac": quack_owner_mutation_mac(authenticated, token),
    }


def _transition_request(
    server: Any,
    token: str,
    *,
    session_id: str,
    owner_id: str,
    new_status: str = "in_progress",
    recorded_at: str = "2026-08-20T20:00:00Z",
) -> dict[str, Any]:
    connection = server._connection  # noqa: SLF001
    task = connection.execute(
        "SELECT task_alias, goal_cid, status, revision, body_json "
        "FROM tasks WHERE task_cid = 'task:test'"
    ).fetchone()
    head = connection.execute(
        "SELECT COALESCE(MAX(sequence), 0), COALESCE(MAX(global_sequence), 0) "
        "FROM domain_events WHERE stream_id = 'stream:intent'"
    ).fetchone()
    assert task is not None and head is not None
    revision = int(task[3]) + 1
    inner = {
        "task_cid": "task:test",
        "task_alias": str(task[0]),
        "goal_cid": str(task[1]),
        "previous_status": str(task[2]),
        "status": new_status,
        "revision": revision,
        "receipt": {},
        "recorded_at": recorded_at,
    }
    event_body = {
        "schema": "ipfs_accelerate_py/agent-supervisor/intent-event@1",
        "event_type": "intent.task_status_changed",
        "subject_id": "task:test",
        "body": inner,
        "recorded_at": recorded_at,
        "owner_id": owner_id,
    }
    sequence = int(head[0]) + 1
    global_sequence = int(head[1]) + 1
    event_id = content_identity(
        {
            "stream_id": "stream:intent",
            "sequence": sequence,
            "global_sequence": global_sequence,
            "event_type": "intent.task_status_changed",
            "body": event_body,
        }
    )
    body_json = str(task[4])
    steps = [
        {
            "template_id": QUACK_MUTATION_TASK_STATUS_CAS,
            "parameters": [
                new_status,
                revision,
                recorded_at,
                body_json,
                "task:test",
                int(task[3]),
            ],
        },
        {
            "template_id": QUACK_MUTATION_TASK_REVISION_INSERT,
            "parameters": [
                "task:test", revision, new_status, body_json, recorded_at
            ],
        },
        {
            "template_id": QUACK_MUTATION_DOMAIN_EVENT_INSERT,
            "parameters": [
                event_id,
                "stream:intent",
                sequence,
                global_sequence,
                "intent.task_status_changed",
                "task:test",
                "",
                session_id,
                recorded_at,
                canonical_json_bytes(event_body).decode("utf-8"),
            ],
        },
    ]
    return _signed_request(
        server,
        token,
        operation=QUACK_MUTATION_TASK_STATUS_TRANSITION,
        steps=steps,
    )


def _completion_request(
    server: Any,
    token: str,
    *,
    evidence_digests: list[str],
    session_id: str = "lane:completion",
    forged_evidence_digest: str = "",
) -> dict[str, Any]:
    connection = server._connection  # noqa: SLF001
    task = connection.execute(
        "SELECT task_alias, goal_cid, status, revision, body_json "
        "FROM tasks WHERE task_cid = 'task:test'"
    ).fetchone()
    head = connection.execute(
        "SELECT COALESCE(MAX(sequence), 0), COALESCE(MAX(global_sequence), 0) "
        "FROM domain_events WHERE stream_id = 'stream:intent'"
    ).fetchone()
    assert task is not None and head is not None and task[2] == "in_progress"
    recorded_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    revision = int(task[3]) + 1
    receipt: dict[str, Any] = {}
    evidence_digest = content_identity(
        {
            "task_cid": "task:test",
            "revision": revision,
            "receipt": receipt,
            "evidence_digests": evidence_digests,
        }
    )
    persisted_evidence_digest = forged_evidence_digest or evidence_digest
    receipt_cid = content_identity(
        {
            "namespace": "completion-receipt",
            "task_cid": "task:test",
            "revision": revision,
            "evidence_digest": persisted_evidence_digest,
        }
    )
    inner = {
        "task_cid": "task:test",
        "task_alias": str(task[0]),
        "goal_cid": str(task[1]),
        "previous_status": "in_progress",
        "status": "completed",
        "revision": revision,
        "receipt": receipt,
        "recorded_at": recorded_at,
        "completion_receipt_cid": receipt_cid,
        "evidence_digest": persisted_evidence_digest,
    }
    event_body = {
        "schema": "ipfs_accelerate_py/agent-supervisor/intent-event@1",
        "event_type": "intent.completion_recorded",
        "subject_id": "task:test",
        "body": inner,
        "recorded_at": recorded_at,
        "owner_id": "test:completion",
    }
    sequence = int(head[0]) + 1
    global_sequence = int(head[1]) + 1
    event_id = content_identity(
        {
            "stream_id": "stream:intent",
            "sequence": sequence,
            "global_sequence": global_sequence,
            "event_type": "intent.completion_recorded",
            "body": event_body,
        }
    )
    body_json = str(task[4])
    completion_body = canonical_json_bytes(
        {
            "schema": COMPLETION_EVIDENCE_SCHEMA,
            "receipt": receipt,
            "evidence_digests": evidence_digests,
            "revision": revision,
        }
    ).decode("utf-8")
    steps = [
        {
            "template_id": QUACK_MUTATION_TASK_STATUS_CAS,
            "parameters": [
                "completed", revision, recorded_at, body_json, "task:test", int(task[3])
            ],
        },
        {
            "template_id": QUACK_MUTATION_TASK_REVISION_INSERT,
            "parameters": ["task:test", revision, "completed", body_json, recorded_at],
        },
        {
            "template_id": QUACK_MUTATION_COMPLETION_RECEIPT_INSERT,
            "parameters": [
                receipt_cid,
                "task:test",
                str(task[1]),
                "",
                "",
                0,
                recorded_at,
                "",
                persisted_evidence_digest,
                completion_body,
            ],
        },
        {
            "template_id": QUACK_MUTATION_DOMAIN_EVENT_INSERT,
            "parameters": [
                event_id,
                "stream:intent",
                sequence,
                global_sequence,
                "intent.completion_recorded",
                "task:test",
                "",
                session_id,
                recorded_at,
                canonical_json_bytes(event_body).decode("utf-8"),
            ],
        },
    ]
    return _signed_request(
        server,
        token,
        operation=QUACK_MUTATION_TASK_STATUS_TRANSITION,
        steps=steps,
    )


_QUEUE_BACKOFF_INSERT_SQL = """
    INSERT INTO leases (
        task_cid, claim_cid, resolution_cid, claimant_did,
        logical_epoch, fencing_token, expires_at_ms, attempt,
        state, started_at_ms, release_reason, retry_not_before_ms,
        owner_session_id, fence_epoch, revision, extension_schema,
        extension_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""
_QUEUE_BACKOFF_UPDATE_SQL = """
    UPDATE leases SET
        attempt = ?, retry_not_before_ms = ?,
        release_reason = ?, state = 'released',
        extension_schema = ?, extension_json = ?,
        revision = revision + 1
    WHERE task_cid = ?
"""


def _queue_backoff_request(
    server: Any,
    token: str,
    *,
    session_id: str,
    owner_id: str,
    delay_ms: int = 60_000,
    reason: str = "backoff",
    selection_penalty: int = 0,
    now_ms: int = 1_700_000_000_000,
    recorded_at: str = "2026-08-20T20:00:00Z",
) -> dict[str, Any]:
    connection = server._connection  # noqa: SLF001
    task = connection.execute(
        "SELECT 1 FROM tasks WHERE task_cid = 'task:test'"
    ).fetchone()
    lease = connection.execute(
        "SELECT attempt FROM leases WHERE task_cid = 'task:test'"
    ).fetchone()
    head = connection.execute(
        "SELECT COALESCE(MAX(sequence), 0), COALESCE(MAX(global_sequence), 0) "
        "FROM domain_events WHERE stream_id = 'stream:intent'"
    ).fetchone()
    assert task is not None and head is not None
    inserting = lease is None
    attempt = 1 if inserting else int(lease[0]) + 1
    retry_not_before_ms = now_ms + delay_ms
    extension = canonical_json_bytes(
        {
            "consecutive_failures": attempt,
            "reason": reason,
            "selection_penalty": selection_penalty,
        }
    ).decode("utf-8")
    inner = {
        "attempt": attempt,
        "delay_ms": delay_ms,
        "reason": reason,
        "retry_not_before_ms": retry_not_before_ms,
        "revision": attempt,
        "selection_penalty": selection_penalty,
        "task_cid": "task:test",
    }
    event_body = {
        "schema": "ipfs_accelerate_py/agent-supervisor/intent-event@1",
        "event_type": "intent.queue_backoff",
        "subject_id": "task:test",
        "body": inner,
        "recorded_at": recorded_at,
        "owner_id": owner_id,
    }
    sequence = int(head[0]) + 1
    global_sequence = int(head[1]) + 1
    event_id = content_identity(
        {
            "stream_id": "stream:intent",
            "sequence": sequence,
            "global_sequence": global_sequence,
            "event_type": "intent.queue_backoff",
            "body": event_body,
        }
    )
    if inserting:
        lease_step = {
            "template_id": QUACK_MUTATION_LEASE_QUEUE_BACKOFF_INSERT,
            "parameters": [
                "task:test",
                "claim:queue:task:test",
                "resolution:queue:task:test",
                owner_id,
                1,
                1,
                0,
                attempt,
                "released",
                now_ms,
                reason,
                retry_not_before_ms,
                session_id,
                1,
                1,
                QUEUE_ENTRY_SCHEMA,
                extension,
            ],
        }
    else:
        lease_step = {
            "template_id": QUACK_MUTATION_LEASE_QUEUE_BACKOFF_UPDATE,
            "parameters": [
                attempt,
                retry_not_before_ms,
                reason,
                QUEUE_ENTRY_SCHEMA,
                extension,
                "task:test",
            ],
        }
    steps = [
        lease_step,
        {
            "template_id": QUACK_MUTATION_DOMAIN_EVENT_INSERT,
            "parameters": [
                event_id,
                "stream:intent",
                sequence,
                global_sequence,
                "intent.queue_backoff",
                "task:test",
                "",
                session_id,
                recorded_at,
                canonical_json_bytes(event_body).decode("utf-8"),
            ],
        },
    ]
    return _signed_request(
        server,
        token,
        operation=QUACK_MUTATION_QUEUE_BACKOFF,
        steps=steps,
    )


def _publish(server: Any, request: dict[str, Any]) -> Path:
    inbox = server.mutation_inbox_path()
    inbox.mkdir(parents=True, exist_ok=True)
    path = inbox / f"{request['request_id']}.request.json"
    path.write_bytes(canonical_json_bytes(request) + b"\n")
    return path


def _done(server: Any, request: dict[str, Any]) -> dict[str, Any]:
    path = server.mutation_inbox_path() / f"{request['request_id']}.done.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_authenticated_bundle_is_atomic_and_exact_replay_is_idempotent(tmp_path: Path) -> None:
    server, _identity, token, database = _server(tmp_path)
    request = _transition_request(server, token, session_id="lane:one", owner_id="one")
    try:
        _publish(server, request)
        assert server.service_mutation_inbox() == 1
        first = _done(server, request)
        assert first["ok"] is True
        assert first["result_mac"]

        _publish(server, request)
        assert server.service_mutation_inbox() == 1
        assert _done(server, request) == first
        assert server._connection.execute(  # noqa: SLF001
            "SELECT COUNT(*) FROM task_revisions "
            "WHERE task_cid = 'task:test' AND revision = 2"
        ).fetchone()[0] == 1
        assert server._connection.execute(  # noqa: SLF001
            "SELECT COUNT(*) FROM domain_events WHERE event_id = ?",
            [request["steps"][-1]["parameters"][0]],
        ).fetchone()[0] == 1
    finally:
        server.stop()
    with DatabaseTaskSource(database, install_schema=False) as source:
        assert source.projection_matches_events() is True


def test_two_remote_bundles_have_one_winner_and_typed_loser(tmp_path: Path) -> None:
    server, _identity, token, database = _server(tmp_path)
    first = _transition_request(server, token, session_id="lane:one", owner_id="one")
    second = _transition_request(server, token, session_id="lane:two", owner_id="two")
    assert first["request_id"] != second["request_id"]
    try:
        _publish(server, first)
        _publish(server, second)
        assert server.service_mutation_inbox(max_requests=2) == 2
        results = [_done(server, first), _done(server, second)]
        assert sorted((item["ok"] for item in results)) == [False, True]
        loser = next(item for item in results if not item["ok"])
        assert loser["error_code"] == "cas_conflict"
        assert server._connection.execute(  # noqa: SLF001
            "SELECT COUNT(*) FROM task_revisions "
            "WHERE task_cid = 'task:test' AND revision = 2"
        ).fetchone()[0] == 1
        assert server._connection.execute(  # noqa: SLF001
            "SELECT COUNT(*) FROM domain_events WHERE task_cid = 'task:test' "
            "AND event_type = 'intent.task_status_changed'"
        ).fetchone()[0] == 1
    finally:
        server.stop()
    with DatabaseTaskSource(database, install_schema=False) as source:
        assert source.projection_matches_events() is True


@pytest.mark.parametrize(
    ("new_status", "expected"),
    [
        ("blocked", True),
        ("retrying", True),
        ("ready", True),
        ("failed", False),
        ("cancelled", False),
    ],
)
def test_in_progress_recovery_transitions_are_closed(
    tmp_path: Path, new_status: str, expected: bool
) -> None:
    server, _identity, token, database = _server(tmp_path)
    try:
        claim = _transition_request(
            server, token, session_id="lane:claim", owner_id="claim"
        )
        _publish(server, claim)
        assert server.service_mutation_inbox() == 1
        assert _done(server, claim)["ok"] is True
        recover = _transition_request(
            server,
            token,
            session_id=f"lane:{new_status}",
            owner_id=new_status,
            new_status=new_status,
            recorded_at="2026-08-20T20:00:01Z",
        )
        _publish(server, recover)
        assert server.service_mutation_inbox() == 1
        result = _done(server, recover)
        if expected:
            assert result["ok"] is True
            assert server._connection.execute(  # noqa: SLF001
                "SELECT status FROM tasks WHERE task_cid = 'task:test'"
            ).fetchone()[0] == new_status
        else:
            assert result["ok"] is False
            assert result["error_code"] == "transition_invalid"
            assert server._connection.execute(  # noqa: SLF001
                "SELECT status FROM tasks WHERE task_cid = 'task:test'"
            ).fetchone()[0] == "in_progress"
    finally:
        server.stop()
    with DatabaseTaskSource(database, install_schema=False) as source:
        assert source.projection_matches_events() is True


def test_blocked_and_retrying_can_reenter_execution(tmp_path: Path) -> None:
    server, _identity, token, database = _server(tmp_path)
    try:
        for index, status in enumerate(
            ("in_progress", "blocked", "retrying", "in_progress")
        ):
            request = _transition_request(
                server,
                token,
                session_id=f"lane:{index}",
                owner_id=f"owner:{index}",
                new_status=status,
                recorded_at=f"2026-08-20T20:00:0{index}Z",
            )
            _publish(server, request)
            assert server.service_mutation_inbox() == 1
            result = _done(server, request)
            assert result["ok"] is True, result
        assert server._connection.execute(  # noqa: SLF001
            "SELECT status FROM tasks WHERE task_cid = 'task:test'"
        ).fetchone()[0] == "in_progress"
    finally:
        server.stop()
    with DatabaseTaskSource(database, install_schema=False) as source:
        assert source.projection_matches_events() is True


def test_completed_cannot_be_reopened(tmp_path: Path) -> None:
    server, _identity, token, database = _server(tmp_path)
    try:
        claim = _transition_request(
            server, token, session_id="lane:claim", owner_id="claim"
        )
        _publish(server, claim)
        assert server.service_mutation_inbox() == 1
        assert _done(server, claim)["ok"] is True
        server._connection.execute(  # noqa: SLF001
            "UPDATE tasks SET status = 'completed' WHERE task_cid = 'task:test'"
        )
        reopen = _transition_request(
            server,
            token,
            session_id="lane:reopen",
            owner_id="reopen",
            new_status="in_progress",
            recorded_at="2026-08-20T21:00:00Z",
        )
        _publish(server, reopen)
        assert server.service_mutation_inbox() == 1
        result = _done(server, reopen)
        assert result["ok"] is False
        assert result["error_code"] == "transition_invalid"
    finally:
        server.stop()


@pytest.mark.parametrize("attack", ["malformed", "forged_mac", "wrong_generation"])
def test_malformed_forged_and_wrong_generation_requests_are_rejected(
    tmp_path: Path, attack: str
) -> None:
    server, _identity, token, _database = _server(tmp_path)
    request = _transition_request(server, token, session_id="lane:one", owner_id="one")
    if attack == "malformed":
        request = {"request_id": request["request_id"]}
    elif attack == "forged_mac":
        request["auth_mac"] = "0" * 64
    else:
        request["binding"]["generation"] += 1
        unsigned = dict(request)
        unsigned.pop("auth_mac")
        unsigned.pop("request_cid")
        request["request_cid"] = quack_owner_mutation_content_id(unsigned)
        authenticated = {**unsigned, "request_cid": request["request_cid"]}
        request["auth_mac"] = quack_owner_mutation_mac(authenticated, token)
    path = server.mutation_inbox_path() / f"{request['request_id']}.request.json"
    try:
        _publish(server, request)
        assert server.service_mutation_inbox() == 1
        assert not path.exists()
        assert not (server.mutation_inbox_path() / f"{request['request_id']}.done.json").exists()
        task = server._connection.execute(  # noqa: SLF001
            "SELECT status, revision FROM tasks WHERE task_cid = 'task:test'"
        ).fetchone()
        assert (task[0], task[1]) == ("ready", 1)
    finally:
        server.stop()


def test_stale_claim_reconciles_exact_committed_effects(tmp_path: Path) -> None:
    server, _identity, token, _database = _server(tmp_path)
    request = _transition_request(server, token, session_id="lane:one", owner_id="one")
    try:
        _publish(server, request)
        server.service_mutation_inbox()
        done = server.mutation_inbox_path() / f"{request['request_id']}.done.json"
        done.unlink()
        processing = server.mutation_inbox_path() / f"{request['request_id']}.processing.json"
        processing.write_bytes(canonical_json_bytes(request) + b"\n")
        old = time.time() - 60
        os.utime(processing, (old, old))
        server.service_mutation_inbox()
        reconciled = _done(server, request)
        assert reconciled["ok"] is True
        assert reconciled["observed"]["reconciled_after_interruption"] is True
        assert reconciled["observed"]["read_replica"]["live"] is True
        assert (
            reconciled["observed"]["read_replica"]["refresh_sequence"] >= 3
        )
    finally:
        server.stop()


def test_expired_unclaimed_request_is_refused_without_effect(tmp_path: Path) -> None:
    server, _identity, token, _database = _server(tmp_path)
    request = _transition_request(server, token, session_id="lane:one", owner_id="one")
    expired_at = int(time.time() * 1000) - 1
    request["expires_at_ms"] = expired_at
    request["issued_at_ms"] = expired_at - QUACK_OWNER_MUTATION_REQUEST_TTL_MS
    unsigned = dict(request)
    unsigned.pop("auth_mac")
    unsigned.pop("request_cid")
    request["request_cid"] = quack_owner_mutation_content_id(unsigned)
    authenticated = {**unsigned, "request_cid": request["request_cid"]}
    request["auth_mac"] = quack_owner_mutation_mac(authenticated, token)
    done = server.mutation_inbox_path() / f"{request['request_id']}.done.json"
    try:
        _publish(server, request)
        assert server.service_mutation_inbox() == 1
        assert not done.exists()
        task = server._connection.execute(  # noqa: SLF001
            "SELECT status, revision FROM tasks WHERE task_cid = 'task:test'"
        ).fetchone()
        assert (task[0], task[1]) == ("ready", 1)
    finally:
        server.stop()


def test_owner_claims_inode_before_cancel_and_recreate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, _identity, token, _database = _server(tmp_path)
    request = _transition_request(server, token, session_id="lane:one", owner_id="one")
    request_path = _publish(server, request)
    processing = server.mutation_inbox_path() / f"{request['request_id']}.processing.json"
    cancelled = server.mutation_inbox_path() / f"{request['request_id']}.cancelled.json"
    reached_claim = threading.Event()
    allow_claim = threading.Event()
    real_replace = os.replace
    worker: threading.Thread | None = None

    def gated_replace(source: Any, target: Any) -> None:
        if (
            worker is not None
            and threading.current_thread() is worker
            and Path(target) == processing
        ):
            reached_claim.set()
            assert allow_claim.wait(timeout=5)
        real_replace(source, target)

    monkeypatch.setattr(quack_server_module.os, "replace", gated_replace)
    serviced: list[int] = []
    worker = threading.Thread(
        target=lambda: serviced.append(server.service_mutation_inbox()),
        daemon=True,
    )
    try:
        worker.start()
        assert reached_claim.wait(timeout=5)
        # The timeout side wins the original request pathname, then a forged
        # same-name file appears.  The owner must read only what it actually
        # claims after the race, never cached pre-claim bytes.
        real_replace(request_path, cancelled)
        forged = dict(request)
        forged["auth_mac"] = "0" * 64
        request_path.write_bytes(canonical_json_bytes(forged) + b"\n")
        allow_claim.set()
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert serviced == [1]
        assert not (
            server.mutation_inbox_path() / f"{request['request_id']}.done.json"
        ).exists()
        task = server._connection.execute(  # noqa: SLF001
            "SELECT status, revision FROM tasks WHERE task_cid = 'task:test'"
        ).fetchone()
        assert (task[0], task[1]) == ("ready", 1)
    finally:
        allow_claim.set()
        if worker is not None:
            worker.join(timeout=1)
        cancelled.unlink(missing_ok=True)
        server.stop()


@pytest.mark.parametrize(
    "gate",
    ["required_digest", "required_kind", "stale", "removed", "forged_digest"],
)
def test_owner_rechecks_exact_current_completion_evidence(
    tmp_path: Path,
    gate: str,
) -> None:
    server, _identity, token, _database = _server(tmp_path)
    digest = "sha256:" + ("ab" * 32)
    try:
        claim = _transition_request(
            server, token, session_id="lane:one", owner_id="one"
        )
        _publish(server, claim)
        assert server.service_mutation_inbox() == 1
        assert _done(server, claim)["ok"] is True

        if gate == "required_digest":
            policy = {"required_digest": "sha256:" + ("cd" * 32)}
        elif gate == "required_kind":
            policy = {"evidence_kind": "proof"}
        else:
            policy = {"required_digest": digest}
        server._connection.execute(  # noqa: SLF001
            "INSERT INTO task_acceptance "
            "(task_cid, ordinal, criterion, evidence_policy_json) "
            "VALUES (?, ?, ?, ?)",
            [
                "task:test",
                0,
                "must be independently validated",
                canonical_json_bytes(policy).decode("utf-8"),
            ],
        )
        created_at = (
            "2000-01-01T00:00:00Z"
            if gate == "stale"
            else datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
        server._connection.execute(  # noqa: SLF001
            "INSERT INTO evidence_nodes "
            "(evidence_id, parent_evidence_id, task_cid, evidence_kind, "
            "digest, created_at, body_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ["evidence:test", "", "task:test", "validation", digest, created_at, "{}"],
        )
        completion = _completion_request(
            server,
            token,
            evidence_digests=[digest],
            forged_evidence_digest=(
                content_identity({"forged": "completion-evidence"})
                if gate == "forged_digest"
                else ""
            ),
        )
        if gate == "removed":
            server._connection.execute(  # noqa: SLF001
                "DELETE FROM evidence_nodes WHERE evidence_id = 'evidence:test'"
            )
        _publish(server, completion)
        assert server.service_mutation_inbox() == 1
        result = _done(server, completion)
        assert result["ok"] is False
        assert result["error_code"] == (
            "completion_receipt_invalid"
            if gate == "forged_digest"
            else "completion_evidence_stale"
        )
        task = server._connection.execute(  # noqa: SLF001
            "SELECT status, revision FROM tasks WHERE task_cid = 'task:test'"
        ).fetchone()
        assert (task[0], task[1]) == ("in_progress", 2)
    finally:
        server.stop()


@pytest.mark.parametrize("corruption", ["event", "updated_at"])
def test_stale_replay_rejects_mismatched_persisted_effect(
    tmp_path: Path,
    corruption: str,
) -> None:
    server, _identity, token, _database = _server(tmp_path)
    request = _transition_request(server, token, session_id="lane:one", owner_id="one")
    try:
        _publish(server, request)
        assert server.service_mutation_inbox() == 1
        done = server.mutation_inbox_path() / f"{request['request_id']}.done.json"
        done.unlink()
        if corruption == "event":
            event_id = request["steps"][-1]["parameters"][0]
            server._connection.execute(  # noqa: SLF001
                "UPDATE domain_events SET session_id = 'forged' WHERE event_id = ?",
                [event_id],
            )
        else:
            server._connection.execute(  # noqa: SLF001
                "UPDATE tasks SET updated_at = 'forged' WHERE task_cid = 'task:test'"
            )
        processing = (
            server.mutation_inbox_path()
            / f"{request['request_id']}.processing.json"
        )
        processing.write_bytes(canonical_json_bytes(request) + b"\n")
        old = time.time() - 60
        os.utime(processing, (old, old))

        assert server.service_mutation_inbox() == 0
        result = _done(server, request)
        assert result["ok"] is False
        assert result["error_code"] == "replay_integrity_failure"
    finally:
        server.stop()


def test_forged_done_receipt_is_rejected_and_cannot_suppress_request(tmp_path: Path) -> None:
    server, _identity, token, _database = _server(tmp_path)
    request = _transition_request(server, token, session_id="lane:one", owner_id="one")
    done = server.mutation_inbox_path() / f"{request['request_id']}.done.json"
    try:
        forged = server._mutation_result(  # noqa: SLF001
            request, ok=True, rowcounts=[1, 1, 1], observed={"forged": False}
        )
        forged["observed"]["forged"] = True
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_bytes(canonical_json_bytes(forged) + b"\n")
        with pytest.raises(DuckDBConnectionPolicyError, match="invalid result receipt"):
            _validate_quack_mutation_result(forged, request=request, token=token)

        _publish(server, request)
        assert server.service_mutation_inbox() == 1
        valid = _done(server, request)
        assert valid["ok"] is True
        assert valid["observed"].get("forged") is None
        # This hermetic server intentionally uses FakeQuackTransport, so it
        # cannot produce the authoritative read-replica proof required by the
        # transport-level validator.  The security property under test is the
        # exact signed owner envelope: the forged file is rejected and the
        # owner replaces it with a canonical CID/MAC-bound result.
        assert server._existing_result_is_valid(done, request) is True  # noqa: SLF001
    finally:
        server.stop()


def test_request_filename_is_closed_to_canonical_cid() -> None:
    assert MUTATION_REQUEST_NAME.fullmatch("0" * 32 + ".request.json") is None
    request_id = content_identity({"request": "bounded"})
    assert MUTATION_REQUEST_NAME.fullmatch(f"{request_id}.request.json") is not None


def test_connection_commit_dispatches_buffer_and_nested_begin_preserves_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AttachedConnection:
        description = None

        def __init__(self) -> None:
            self.statements: list[str] = []

        def execute(self, sql: str, _parameters: Any = None) -> AttachedConnection:
            self.statements.append(" ".join(sql.strip().upper().split()))
            self.description = None
            return self

        def fetchall(self) -> list[Any]:
            return []

    attached = AttachedConnection()
    connection = DuckDBConnection.wrap(attached)
    connection._default_catalog = "control_plane"  # noqa: SLF001
    connection._quack_mutation_binding = {"store_id": "/state/control.duckdb"}  # noqa: SLF001
    connection._quack_mutation_token = "test-token"  # noqa: SLF001
    dispatched: list[list[dict[str, Any]]] = []

    def dispatch(
        steps: Any, *, binding: Any, token: str
    ) -> Any:
        assert binding == {"store_id": "/state/control.duckdb"}
        assert token == "test-token"
        dispatched.append([dict(step) for step in steps])
        return duckdb_state_module._empty_duckdb_cursor(rowcount=1)  # noqa: SLF001

    monkeypatch.setattr(
        duckdb_state_module,
        "_execute_quack_owner_mutation_bundle",
        dispatch,
    )
    connection.execute("BEGIN TRANSACTION")
    connection.execute(
        """
        UPDATE tasks SET status = ?, revision = ?, updated_at = ?, body_json = ?
        WHERE task_cid = ? AND revision = ?
        """,
        ["in_progress", 2, "2026-08-20T20:00:00Z", "{}", "task:test", 1],
    )
    with pytest.raises(DuckDBConnectionPolicyError, match="already active"):
        connection.execute("BEGIN TRANSACTION")
    assert len(connection._quack_pending_mutations) == 1  # noqa: SLF001

    connection.commit()

    assert len(dispatched) == 1
    assert dispatched[0][0]["template_id"] == QUACK_MUTATION_TASK_STATUS_CAS
    assert attached.statements[-1] == "ROLLBACK"
    assert connection.in_transaction is False
    assert connection._quack_pending_mutations == []  # noqa: SLF001


def test_queue_backoff_sql_is_in_closed_catalog_and_forms_admitted_bundle() -> None:
    insert_id = _QUACK_OWNER_MUTATION_SQL_TO_TEMPLATE[
        _normalize_quack_mutation_sql(_QUEUE_BACKOFF_INSERT_SQL)
    ]
    update_id = _QUACK_OWNER_MUTATION_SQL_TO_TEMPLATE[
        _normalize_quack_mutation_sql(_QUEUE_BACKOFF_UPDATE_SQL)
    ]
    assert insert_id == QUACK_MUTATION_LEASE_QUEUE_BACKOFF_INSERT
    assert update_id == QUACK_MUTATION_LEASE_QUEUE_BACKOFF_UPDATE
    assert (
        _quack_mutation_operation(
            [
                {"template_id": insert_id, "parameters": []},
                {"template_id": QUACK_MUTATION_DOMAIN_EVENT_INSERT, "parameters": []},
            ]
        )
        == QUACK_MUTATION_QUEUE_BACKOFF
    )
    assert (
        _quack_mutation_operation(
            [
                {"template_id": update_id, "parameters": []},
                {"template_id": QUACK_MUTATION_DOMAIN_EVENT_INSERT, "parameters": []},
            ]
        )
        == QUACK_MUTATION_QUEUE_BACKOFF
    )
    seventeen = ["task:test"] * 11 + [1, 1, 0, 1, 1, 1]
    assert len(seventeen) == QUACK_OWNER_MUTATION_MAX_PARAMETERS
    _validate_quack_mutation_parameters(seventeen)
    with pytest.raises(
        DuckDBConnectionPolicyError, match="parameter count exceeds its bound"
    ):
        _validate_quack_mutation_parameters(seventeen + ["overflow"])
    unknown = _normalize_quack_mutation_sql(
        "UPDATE leases SET state = ? WHERE task_cid = ?"
    )
    assert unknown not in _QUACK_OWNER_MUTATION_SQL_TO_TEMPLATE


def test_queue_backoff_insert_and_update_are_atomic_and_replay_idempotent(
    tmp_path: Path,
) -> None:
    server, _identity, token, database = _server(tmp_path)
    try:
        insert = _queue_backoff_request(
            server, token, session_id="lane:backoff", owner_id="backoff"
        )
        _publish(server, insert)
        assert server.service_mutation_inbox() == 1
        first = _done(server, insert)
        assert first["ok"] is True, first
        lease = server._connection.execute(  # noqa: SLF001
            "SELECT attempt, state, retry_not_before_ms FROM leases "
            "WHERE task_cid = 'task:test'"
        ).fetchone()
        assert lease is not None
        assert (lease[0], lease[1], lease[2]) == (1, "released", 1_700_000_060_000)

        _publish(server, insert)
        assert server.service_mutation_inbox() == 1
        assert _done(server, insert) == first
        assert server._connection.execute(  # noqa: SLF001
            "SELECT COUNT(*) FROM leases WHERE task_cid = 'task:test'"
        ).fetchone()[0] == 1
        assert server._connection.execute(  # noqa: SLF001
            "SELECT COUNT(*) FROM domain_events WHERE event_id = ?",
            [insert["steps"][-1]["parameters"][0]],
        ).fetchone()[0] == 1

        update = _queue_backoff_request(
            server,
            token,
            session_id="lane:backoff-2",
            owner_id="backoff",
            recorded_at="2026-08-20T20:00:01Z",
        )
        _publish(server, update)
        assert server.service_mutation_inbox() == 1
        second = _done(server, update)
        assert second["ok"] is True, second
        lease_after = server._connection.execute(  # noqa: SLF001
            "SELECT attempt, state, retry_not_before_ms FROM leases "
            "WHERE task_cid = 'task:test'"
        ).fetchone()
        assert lease_after is not None
        assert (lease_after[0], lease_after[1], lease_after[2]) == (
            2,
            "released",
            1_700_000_060_000,
        )

        _publish(server, update)
        assert server.service_mutation_inbox() == 1
        assert _done(server, update) == second
    finally:
        server.stop()
    with DatabaseTaskSource(database, install_schema=False) as source:
        assert source.projection_matches_events() is True


def test_connection_commit_dispatches_queue_backoff_insert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AttachedConnection:
        description = None

        def __init__(self) -> None:
            self.statements: list[str] = []

        def execute(self, sql: str, _parameters: Any = None) -> AttachedConnection:
            self.statements.append(" ".join(sql.strip().upper().split()))
            self.description = None
            return self

        def fetchall(self) -> list[Any]:
            return []

    attached = AttachedConnection()
    connection = DuckDBConnection.wrap(attached)
    connection._default_catalog = "control_plane"  # noqa: SLF001
    connection._quack_mutation_binding = {"store_id": "/state/control.duckdb"}  # noqa: SLF001
    connection._quack_mutation_token = "test-token"  # noqa: SLF001
    dispatched: list[list[dict[str, Any]]] = []

    def dispatch(steps: Any, *, binding: Any, token: str) -> Any:
        assert binding == {"store_id": "/state/control.duckdb"}
        assert token == "test-token"
        dispatched.append([dict(step) for step in steps])
        assert _quack_mutation_operation(steps) == QUACK_MUTATION_QUEUE_BACKOFF
        return duckdb_state_module._empty_duckdb_cursor(rowcount=1)  # noqa: SLF001

    monkeypatch.setattr(
        duckdb_state_module,
        "_execute_quack_owner_mutation_bundle",
        dispatch,
    )
    connection.execute("BEGIN TRANSACTION")
    connection.execute(
        _QUEUE_BACKOFF_INSERT_SQL,
        [
            "task:test",
            "claim:queue:task:test",
            "resolution:queue:task:test",
            "owner",
            1,
            1,
            0,
            1,
            "released",
            1_700_000_000_000,
            "backoff",
            1_700_000_060_000,
            "lane:one",
            1,
            1,
            QUEUE_ENTRY_SCHEMA,
            "{}",
        ],
    )
    connection.execute(
        """
        INSERT INTO domain_events (
            event_id, stream_id, sequence, global_sequence, event_type,
            task_cid, attempt_id, session_id, recorded_at, body_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "event:1",
            "stream:intent",
            1,
            1,
            "intent.queue_backoff",
            "task:test",
            "",
            "lane:one",
            "2026-08-20T20:00:00Z",
            "{}",
        ],
    )
    connection.commit()
    assert len(dispatched) == 1
    assert dispatched[0][0]["template_id"] == QUACK_MUTATION_LEASE_QUEUE_BACKOFF_INSERT
    assert dispatched[0][1]["template_id"] == QUACK_MUTATION_DOMAIN_EVENT_INSERT


def test_truthful_looking_receipt_cannot_self_admit_outside_observed_container(
    tmp_path: Path,
) -> None:
    database = tmp_path / "control" / "control.duckdb"
    _seed(database)
    receipt_path, _receipt = _isolation_receipt(tmp_path)
    server = build_server(
        database_path=database,
        state_dir=receipt_path.parent,
        **_isolation_server_kwargs(_receipt),
        store_id=str(database),
        transport=FakeQuackTransport(),
        isolation_receipt_path=receipt_path,
        capability_probe=lambda **_kwargs: probe_quack_capabilities(),
    )
    with pytest.raises(QuackStateServerIsolationError, match="live container isolation"):
        server.start()


def test_invalid_isolation_receipt_cannot_reach_migration_or_change_database(
    tmp_path: Path,
) -> None:
    database = tmp_path / "control" / "control.duckdb"
    _seed(database)
    database_before = database.read_bytes()
    receipt_path, receipt = _isolation_receipt(tmp_path)
    receipt["issuer"] = "forged-without-rebinding-cid"
    receipt_path.write_bytes(canonical_json_bytes(receipt))
    receipt_path.chmod(0o600)
    migration_calls: list[Path] = []
    open_calls: list[Path] = []

    def forbidden_migration(path: Path) -> Any:
        migration_calls.append(path)
        path.write_bytes(b"migration-reached-before-isolation")
        return None

    def forbidden_open(path: Path) -> Any:
        open_calls.append(path)
        raise AssertionError("connection open reached before isolation admission")

    server = build_server(
        database_path=database,
        state_dir=receipt_path.parent,
        **_isolation_server_kwargs(receipt),
        store_id=str(database),
        transport=FakeQuackTransport(),
        isolation_receipt_path=receipt_path,
        isolation_observer=_admitted_observation,
        capability_probe=lambda **_kwargs: probe_quack_capabilities(),
        migrate=forbidden_migration,
        connection_factory=forbidden_open,
    )
    with pytest.raises(
        QuackStateServerIsolationError,
        match="identity or required controls",
    ):
        server.start()

    assert migration_calls == []
    assert open_calls == []
    assert database.read_bytes() == database_before
    assert not server.owner_marker_path().exists()


def test_isolation_observation_mismatch_rejects_valid_receipt(tmp_path: Path) -> None:
    database = tmp_path / "control" / "control.duckdb"
    _seed(database)
    database_before = database.read_bytes()
    receipt_path, _receipt = _isolation_receipt(tmp_path)
    migration_calls: list[Path] = []
    open_calls: list[Path] = []

    def mismatch(config: Any, receipt: dict[str, Any]) -> dict[str, Any]:
        observed = _admitted_observation(config, receipt)
        observed["repository_read_only"] = False
        return observed

    def forbidden_migration(path: Path) -> Any:
        migration_calls.append(path)
        path.write_bytes(b"migration-reached-before-isolation")
        return None

    def forbidden_open(path: Path) -> Any:
        open_calls.append(path)
        raise AssertionError("connection open reached before isolation admission")

    server = build_server(
        database_path=database,
        state_dir=receipt_path.parent,
        **_isolation_server_kwargs(_receipt),
        store_id=str(database),
        transport=FakeQuackTransport(),
        isolation_receipt_path=receipt_path,
        isolation_observer=mismatch,
        capability_probe=lambda **_kwargs: probe_quack_capabilities(),
        migrate=forbidden_migration,
        connection_factory=forbidden_open,
    )
    with pytest.raises(QuackStateServerIsolationError, match="live container isolation"):
        server.start()
    assert migration_calls == []
    assert open_calls == []
    assert database.read_bytes() == database_before
    assert not server.owner_marker_path().exists()


def test_concurrent_typed_database_task_source_cas_has_one_loser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two process-bound typed clients contend through one exclusive owner."""

    database = tmp_path / "control" / "control.duckdb"
    _seed(database)
    receipt_path, _receipt = _isolation_receipt(tmp_path)
    server = build_server(
        database_path=database,
        state_dir=receipt_path.parent,
        **_isolation_server_kwargs(_receipt),
        store_id="casf-typed-concurrent-cas-v1",
        repository_id="repository:test",
        isolation_receipt_path=receipt_path,
        isolation_observer=_admitted_observation,
        capability_probe=lambda **_kwargs: probe_quack_capabilities(),
    )
    identity = server.start()
    assert server._connection.execute(  # noqa: SLF001
        "SELECT current_setting('enable_external_access')"
    ).fetchone()[0] is False
    transport_settings = server._transport_connection.execute(  # noqa: SLF001
        "SELECT current_setting('access_mode'), "
        "current_setting('enable_external_access')"
    ).fetchone()
    assert (transport_settings[0], transport_settings[1]) == ("read_only", False)
    replica = server.status()["read_replica"]
    assert replica == server.ready()["read_replica"]
    assert replica["authority"] == "non_authoritative_read_replica"
    assert replica["path"] == str(
        database.with_name("control.read-replica.duckdb")
    )
    assert replica["source_database_path"] == str(database)
    assert replica["server_id"] == identity.server_id
    assert replica["database_uuid"] == identity.database_uuid
    assert replica["generation"] == identity.generation
    assert replica["live"] is True
    transport_token = server._vault.resolve(identity.secret_handle)  # noqa: SLF001
    assert transport_token not in os.environ.values()

    assert _raw_quack_query(
        identity.listen_uri,
        transport_token,
        "SELECT status, revision FROM tasks WHERE task_cid = 'task:test'",
    ) == [("ready", 1)]
    escape = database.parent / "raw-quack-copy-escape.csv"
    for adversarial_sql in (
        "UPDATE tasks SET status = 'completed' WHERE task_cid = 'task:test'",
        f"COPY (SELECT 1) TO '{escape}'",
        f"SELECT * FROM read_text('{database}')",
        "SELECT * FROM read_text('https://example.com/')",
    ):
        with pytest.raises(
            Exception,
            match="read-only|read only|READ_ONLY|disabled by configuration",
        ):
            _raw_quack_query(
                identity.listen_uri,
                transport_token,
                adversarial_sql,
            )
    assert not escape.exists()

    sources: list[TypedDatabaseTaskSource] = []
    for index in (1, 2):
        sources.append(
            _typed_task_source(
                server,
                identity,
                monkeypatch,
                client_id=f"database-implementation-daemon:typed-lane-{index}",
                allowed_command_operations=(
                    "task.status.cas.receipt",
                    "task.validation.record.passed",
                ),
            )
        )
    initial = sources[0].get_task("task:test")
    assert initial is not None and (initial.status, initial.revision) == ("ready", 1)
    barrier = threading.Barrier(2)
    outcomes: list[tuple[str, Any, TypedDatabaseTaskSource]] = []
    guard = threading.Lock()

    def contend(source: TypedDatabaseTaskSource, lane: str) -> None:
        claim_receipt = _typed_claim_receipt(
            source,
            lane=lane,
            claimed_from_revision=initial.revision,
        )
        barrier.wait(timeout=5)
        try:
            value = source.compare_and_set_status(
                "task:test",
                expected_revision=initial.revision,
                status="in_progress",
                receipt=claim_receipt,
            )
        except TaskSourceConflictError as exc:
            result: tuple[str, Any, TypedDatabaseTaskSource] = (
                "rejected",
                exc,
                source,
            )
        except TransactionError as exc:
            # Once the winning claim is committed, the owner rejects the
            # losing receipt as stale claim authority. If the loser has not
            # submitted yet, the adapter instead rejects its stale revision.
            assert "authorization_denied" in str(exc)
            result = ("rejected", exc, source)
        else:
            result = ("success", value, source)
        with guard:
            outcomes.append(result)

    threads = [
        threading.Thread(
            target=contend,
            args=(source, f"lane-{index}"),
        )
        for index, source in enumerate(sources, start=1)
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)
        assert all(not thread.is_alive() for thread in threads)
        assert sorted(item[0] for item in outcomes) == ["rejected", "success"], [
            (
                label,
                type(value).__name__,
                getattr(value, "kind", None),
                str(value),
                dict(getattr(value, "details", {})),
            )
            for label, value, _source in outcomes
        ]
        winner = next(item[2] for item in outcomes if item[0] == "success")
        after_cas = winner.get_task("task:test")
        assert after_cas is not None
        assert (after_cas.status, after_cas.revision) == ("in_progress", 2)
        winning_claim = after_cas.body["completion_receipt"]
        assert winning_claim["claim_process_attestation"]["client_id"].startswith(
            "database-implementation-daemon:typed-lane-"
        )
        assert not tuple(server.mutation_inbox_path().glob("*.request.json"))
        assert not tuple(server.mutation_inbox_path().glob("*.done.json"))

        admitted = winner.compare_and_set_status(
            after_cas.task_cid,
            after_cas.revision,
            "in_progress",
            {
                **winning_claim,
                "operation": "database_attempt_admitted",
                "claim_phase_schema": TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA,
                "admitted_from_revision": after_cas.revision,
                "attempt_execution_phase": "claimed",
                "attempt_execution_revision": 1,
            },
        )
        evidence_digest = "sha256:" + ("ab" * 32)
        validation = winner.record_validation_result(
            task_cid=admitted.task.task_cid,
            outcome="passed",
            evidence_digest=evidence_digest,
            argv=["python", "-m", "pytest", "-q"],
            attempt_id=winning_claim["attempt_id"],
        )
        assert validation.changed is True

        read_stopping = threading.Event()
        read_failures: list[BaseException] = []
        read_revisions: list[int] = []
        first_read_samples = [threading.Event() for _source in sources]

        def read_during_completion(
            source: TypedDatabaseTaskSource,
            first_read_sample: threading.Event,
        ) -> None:
            while not read_stopping.is_set():
                try:
                    observed_task = source.get_task("task:test")
                    if observed_task is not None:
                        read_revisions.append(observed_task.revision)
                        first_read_sample.set()
                except BaseException as exc:  # retained for exact assertion
                    read_failures.append(exc)
                    return
                read_stopping.wait(0.002)

        readers = [
            threading.Thread(
                target=read_during_completion,
                args=(source, first_read_sample),
                daemon=True,
            )
            for source, first_read_sample in zip(
                sources, first_read_samples, strict=True
            )
        ]
        for reader in readers:
            reader.start()
        first_samples_observed = [
            first_read_sample.wait(timeout=5)
            for first_read_sample in first_read_samples
        ]
        if not all(first_samples_observed):
            read_stopping.set()
            for reader in readers:
                reader.join(timeout=5)
        assert all(first_samples_observed), read_failures
        completed = winner.compare_and_set_status(
            admitted.task.task_cid,
            admitted.task.revision,
            "completed",
            {"operation": "database_complete", "evidence_digest": evidence_digest},
            evidence_digests=[evidence_digest],
        )
        read_stopping.set()
        for reader in readers:
            reader.join(timeout=5)
        assert all(not reader.is_alive() for reader in readers)
        assert not read_failures
        assert read_revisions
        assert completed.task.status == "completed"
        assert completed.task.revision == admitted.task.revision + 1
        assert completed.receipt_cid
        final = winner.get_task("task:test")
        assert final is not None
        assert (final.status, final.revision) == (
            "completed",
            completed.task.revision,
        )
        assert not tuple(server.mutation_inbox_path().glob("*.request.json"))
        assert not tuple(server.mutation_inbox_path().glob("*.done.json"))
    finally:
        for source in sources:
            source.close()
        server.stop()

    with DatabaseTaskSource(database, install_schema=False) as local:
        final = local.get_task("task:test")
        assert final is not None
        assert (final.status, final.revision) == ("completed", 4)


def test_typed_cas_is_authoritative_when_replica_refresh_fails_and_restart_repairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Typed owner commands neither depend on nor mutate through the replica."""

    database = tmp_path / "control" / "control.duckdb"
    _seed(database)
    receipt_path, _receipt = _isolation_receipt(tmp_path)

    def make_server():
        return build_server(
            database_path=database,
            state_dir=receipt_path.parent,
            **_isolation_server_kwargs(_receipt),
            store_id="casf-typed-replica-recovery-v1",
            repository_id="repository:test",
            isolation_receipt_path=receipt_path,
            isolation_observer=_admitted_observation,
            capability_probe=lambda **_kwargs: probe_quack_capabilities(),
        )

    server = make_server()
    identity = server.start()
    transport_token = server._vault.resolve(identity.secret_handle)  # noqa: SLF001
    source = _typed_task_source(
        server,
        identity,
        monkeypatch,
        client_id="database-implementation-daemon:typed-replica-independence",
        allowed_command_operations=("task.status.cas.receipt",),
    )
    refresh_calls = {"count": 0}

    def fail_copy() -> tuple[str, int]:
        refresh_calls["count"] += 1
        raise AssertionError(
            "typed owner command attempted non-authoritative replica settlement"
        )

    monkeypatch.setattr(server, "_copy_authoritative_read_replica", fail_copy)
    try:
        ready = source.get_task("task:test")
        assert ready is not None
        claim = _typed_claim_receipt(
            source,
            lane="typed-replica-independence",
            claimed_from_revision=ready.revision,
        )
        claimed = source.compare_and_set_status(
            ready.task_cid,
            ready.revision,
            "in_progress",
            claim,
        )
        assert claimed.changed is True
        assert (claimed.task.status, claimed.task.revision) == ("in_progress", 2)
        assert refresh_calls["count"] == 0
        assert server.lifecycle.value == "ready"
        assert server.status()["read_replica"]["live"] is True
        # The HTTP/Quack file is explicitly non-authoritative and remains the
        # last checkpoint. Typed reads observe the canonical owner immediately.
        assert _raw_quack_query(
            identity.listen_uri,
            transport_token,
            "SELECT status, revision FROM tasks WHERE task_cid = 'task:test'",
        ) == [("ready", 1)]
        observed = source.get_task("task:test")
        assert observed is not None
        assert (observed.status, observed.revision) == ("in_progress", 2)
        assert not tuple(server.mutation_inbox_path().glob("*.request.json"))
        assert not tuple(server.mutation_inbox_path().glob("*.done.json"))
    finally:
        source.close()
        server.stop()

    # A stale or corrupted non-authoritative replica is never reused. A new
    # owner generation overwrites it from the canonical writer before serving.
    replica_path = database.with_name("control.read-replica.duckdb")
    replica_path.write_bytes(b"drifted-non-authoritative-replica")
    restarted = make_server()
    restarted_identity = restarted.start()
    restarted_token = restarted._vault.resolve(  # noqa: SLF001
        restarted_identity.secret_handle
    )
    try:
        assert restarted_identity.generation > identity.generation
        assert restarted.status()["read_replica"]["live"] is True
        assert restarted.status()["read_replica"]["sha256"].startswith("sha256:")
        assert _raw_quack_query(
            restarted_identity.listen_uri,
            restarted_token,
            "SELECT status, revision FROM tasks WHERE task_cid = 'task:test'",
        ) == [("in_progress", 2)]
    finally:
        restarted.stop()


def test_ops_start_serve_loop_services_owner_inbox(tmp_path: Path) -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "ops"
        / "agent_supervisor"
        / "quack_state_server.py"
    )
    spec = importlib.util.spec_from_file_location("quack_state_server_ops_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Lifecycle:
        value = "ready"

    class Server:
        lifecycle = Lifecycle()
        serviced = 0

        def stop_control_path(self) -> Path:
            return tmp_path / "stop"

        def service_mutation_inbox(self, *, max_requests: int) -> int:
            assert max_requests == 32
            self.serviced += 1
            self.lifecycle.value = "stopped"
            return 0

        def stop(self) -> dict[str, Any]:
            return {"stopped": True}

    server = Server()
    assert module._serve_until_stop(server) == {"stopped": True}
    assert server.serviced == 1


def test_typed_database_task_source_records_process_bound_retry_cooldown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retry cooldowns use an exact typed claim, not the legacy HMAC inbox."""

    database = tmp_path / "control" / "control.duckdb"
    _seed(database)
    receipt_path, _receipt = _isolation_receipt(tmp_path)
    server = build_server(
        database_path=database,
        state_dir=receipt_path.parent,
        **_isolation_server_kwargs(_receipt),
        store_id="casf-typed-retry-cooldown-v1",
        repository_id="repository:test",
        isolation_receipt_path=receipt_path,
        isolation_observer=_admitted_observation,
        capability_probe=lambda **_kwargs: probe_quack_capabilities(),
    )
    identity = server.start()
    clock = {"now_ms": 1_000}
    source = _typed_task_source(
        server,
        identity,
        monkeypatch,
        client_id="database-implementation-daemon:typed-backoff",
        allowed_command_operations=(
            "task.status.cas.receipt",
            "task.retry.cooldown.record",
        ),
        clock_ms=lambda: clock["now_ms"],
    )
    try:
        ready = source.get_task("task:test")
        assert ready is not None
        claim = _typed_claim_receipt(
            source,
            lane="typed-backoff",
            claimed_from_revision=ready.revision,
        )
        claimed = source.compare_and_set_status(
            ready.task_cid,
            ready.revision,
            "in_progress",
            claim,
        )
        cooldown_identity = {
            name: claim[name]
            for name in (
                "attempt_id",
                "claim_id",
                "lease_id",
                "owner_session_id",
                "attempt_number",
                "fencing_token",
                "fence_epoch",
            )
        }
        reason = "database_portal_retry:attempt:typed-backoff:empty_or_no_change"
        first = source.record_task_retry_cooldown(
            task_cid=claimed.task.task_cid,
            expected_task_revision=claimed.task.revision,
            expected_task_status="in_progress",
            delay_ms=60_000,
            reason=reason,
            selection_penalty=100,
            now_ms=clock["now_ms"],
            **cooldown_identity,
        )
        assert first.changed is True
        entry = source.get_queue_entry(claimed.task.task_cid)
        assert entry is not None
        assert entry.attempt == 1
        assert entry.reason == reason
        assert entry.selection_penalty == 100
        assert entry.retry_not_before_ms == 61_000

        replay = source.record_task_retry_cooldown(
            task_cid=claimed.task.task_cid,
            expected_task_revision=claimed.task.revision,
            expected_task_status="in_progress",
            delay_ms=60_000,
            reason=reason,
            selection_penalty=100,
            now_ms=60_999,
            **cooldown_identity,
        )
        assert replay.changed is False
        assert replay.event_id == first.event_id

        retrying = source.compare_and_set_status(
            claimed.task.task_cid,
            claimed.task.revision,
            "retrying",
            {
                "operation": "database_portal_retry",
                **cooldown_identity,
                "queue_reason": reason,
                "backoff_ms": 60_000,
                "retry_not_before_ms": entry.retry_not_before_ms,
                "control_expected_revision": claimed.task.revision,
            },
        )
        assert retrying.task.status == "retrying"
        clock["now_ms"] = 60_999
        assert source.ready_tasks().tasks == ()
        clock["now_ms"] = 61_000
        assert tuple(task.task_cid for task in source.ready_tasks().tasks) == (
            "task:test",
        )
        assert not tuple(server.mutation_inbox_path().glob("*.request.json"))
        assert not tuple(server.mutation_inbox_path().glob("*.done.json"))
    finally:
        source.close()
        server.stop()
