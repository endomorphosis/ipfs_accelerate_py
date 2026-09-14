"""Focused fail-closed tests for the SAWM Quack owner mutation protocol."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.merge_resolver import (
    invoke_llm_resolver,
)
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    DATABASE_PROGRAM_ENV_NAMES,
    DATABASE_PROGRAM_JSON_ENV,
    STATE_AUTHORITY_MODE_ENV,
    STATE_QUACK_MUTATION_BINDING_ENV,
    STATE_QUACK_MUTATION_DIR_ENV,
    provider_subprocess_environment,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    QuackStateServerTokenError,
    TokenVault,
    retire_token_handoff,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    canonical_json_bytes,
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    QUEUE_ENTRY_SCHEMA,
    open_intent_repository,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_owner_mutation import (
    MUTATION_SQL_TEMPLATES,
    QUACK_MUTATION_DOMAIN_EVENT_INSERT,
    QUACK_MUTATION_EVIDENCE_DELETE,
    QUACK_MUTATION_EVIDENCE_INSERT,
    QUACK_MUTATION_LEASE_QUEUE_BACKOFF_INSERT,
    QUACK_MUTATION_LEASE_QUEUE_BACKOFF_UPDATE,
    QUACK_MUTATION_TASK_REVISION_INSERT,
    QUACK_MUTATION_TASK_STATUS_CAS,
    QUACK_OWNER_MUTATION_PROTOCOL_REVISION,
    QUACK_OWNER_MUTATION_REQUEST_SCHEMA,
    QuackOwnerMutationError,
    build_mutation_request,
    execute_owner_mutation,
    mutation_operation,
    mutation_step,
    open_mutation_inbox_directory,
    read_envelope_at,
    service_mutation_inbox,
    validate_mutation_request,
    validate_mutation_result,
    write_envelope_atomic_at,
)

TOKEN = "sawm-quack-test-token"
REQUIRES_DUCKDB = pytest.mark.skipif(
    not duckdb_available(), reason="DuckDB is required for owner transaction tests"
)


@pytest.fixture
def short_root() -> Iterator[Path]:
    """Keep owner inbox paths comfortably below local transport path limits."""

    with tempfile.TemporaryDirectory(prefix="sq-", dir="/tmp") as directory:
        yield Path(directory)


def _binding(*, generation: int = 7) -> dict[str, Any]:
    return {
        "server_id": "server:sawm-test",
        "store_id": "store:sawm-test",
        "database_uuid": "database:sawm-test",
        "schema_revision": 1,
        "schema_fingerprint": "schema:sawm-test",
        "generation": generation,
        "process_birth_id": "birth:sawm-test",
        "listen_uri": "quack://127.0.0.1:17421",
        "extension_fingerprint": "none",
    }


def _seed(database: Path) -> None:
    repository = open_intent_repository(database, owner_id="sawm-test")
    try:
        repository.upsert_objective(
            objective_id="objective:test",
            objective_alias="O",
            title="Objective",
        )
        repository.upsert_goal(
            goal_cid="goal:test",
            goal_alias="G",
            title="Goal",
            objective_id="objective:test",
        )
        repository.upsert_plan(
            plan_cid="plan:test",
            goal_cid="goal:test",
            plan_alias="P",
        )
        repository.upsert_task(
            task_cid="task:test",
            task_alias="T",
            goal_cid="goal:test",
            plan_cid="plan:test",
            objective_id="objective:test",
            ordinal=1,
            status="ready",
        )
    finally:
        repository.close()


def _transition_request(
    connection: Any,
    *,
    binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    task = connection.execute(
        "SELECT task_alias, goal_cid, status, revision, body_json "
        "FROM tasks WHERE task_cid = 'task:test'"
    ).fetchone()
    head = connection.execute(
        "SELECT COALESCE(MAX(sequence), 0), "
        "(SELECT COALESCE(MAX(global_sequence), 0) FROM domain_events) "
        "FROM domain_events WHERE stream_id = 'stream:intent'"
    ).fetchone()
    assert task is not None and head is not None

    recorded_at = "2026-08-28T00:00:00Z"
    revision = int(task[3]) + 1
    event_body = {
        "schema": "ipfs_accelerate_py/agent-supervisor/intent-event@1",
        "event_type": "intent.task_status_changed",
        "subject_id": "task:test",
        "body": {
            "task_cid": "task:test",
            "task_alias": str(task[0]),
            "goal_cid": str(task[1]),
            "previous_status": str(task[2]),
            "status": "in_progress",
            "revision": revision,
            "receipt": {},
            "recorded_at": recorded_at,
        },
        "recorded_at": recorded_at,
        "owner_id": "sawm-test",
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
                "in_progress",
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
                "task:test",
                revision,
                "in_progress",
                body_json,
                recorded_at,
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
                "lane:sawm-test",
                recorded_at,
                canonical_json_bytes(event_body).decode("utf-8"),
            ],
        },
    ]
    return build_mutation_request(
        steps=steps,
        binding=dict(binding or _binding()),
        token=TOKEN,
    )


def _queue_backoff_request(connection: Any) -> dict[str, Any]:
    lease = connection.execute(
        "SELECT attempt FROM leases WHERE task_cid = 'task:test'"
    ).fetchone()
    head = connection.execute(
        "SELECT COALESCE(MAX(sequence), 0), "
        "(SELECT COALESCE(MAX(global_sequence), 0) FROM domain_events) "
        "FROM domain_events WHERE stream_id = 'stream:intent'"
    ).fetchone()
    assert head is not None

    inserting = lease is None
    attempt = 1 if inserting else int(lease[0]) + 1
    started_at_ms = 1_700_000_000_000 + attempt
    delay_ms = 60_000
    retry_not_before_ms = started_at_ms + delay_ms
    reason = f"backoff:{attempt}"
    selection_penalty = attempt
    recorded_at = f"2026-08-28T00:00:0{attempt}Z"
    extension = canonical_json_bytes(
        {
            "selection_penalty": selection_penalty,
            "consecutive_failures": attempt,
            "reason": reason,
        }
    ).decode("utf-8")
    event_body = {
        "schema": "ipfs_accelerate_py/agent-supervisor/intent-event@1",
        "event_type": "intent.queue_backoff",
        "subject_id": "task:test",
        "body": {
            "task_cid": "task:test",
            "attempt": attempt,
            "retry_not_before_ms": retry_not_before_ms,
            "delay_ms": delay_ms,
            "selection_penalty": selection_penalty,
            "reason": reason,
            "revision": attempt,
        },
        "recorded_at": recorded_at,
        "owner_id": "sawm-test",
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
                "sawm-test",
                1,
                1,
                0,
                attempt,
                "released",
                started_at_ms,
                reason,
                retry_not_before_ms,
                "lane:sawm-test",
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
    return build_mutation_request(
        steps=[
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
                    "lane:sawm-test",
                    recorded_at,
                    canonical_json_bytes(event_body).decode("utf-8"),
                ],
            },
        ],
        binding=_binding(),
        token=TOKEN,
    )


def _publish(inbox: Path, request: Mapping[str, Any]) -> None:
    descriptor = open_mutation_inbox_directory(inbox.resolve())
    try:
        write_envelope_atomic_at(
            descriptor,
            f"{request['request_id']}.request.json",
            request,
        )
    finally:
        os.close(descriptor)


def _read_done(inbox: Path, request: Mapping[str, Any]) -> dict[str, Any]:
    descriptor = open_mutation_inbox_directory(inbox.resolve())
    try:
        return read_envelope_at(
            descriptor,
            f"{request['request_id']}.done.json",
        )
    finally:
        os.close(descriptor)


def test_protocol_v2_is_closed_schema_and_never_carries_raw_sql() -> None:
    assert QUACK_OWNER_MUTATION_PROTOCOL_REVISION == 2
    assert QUACK_OWNER_MUTATION_REQUEST_SCHEMA.endswith("@2")

    admitted = mutation_step(
        MUTATION_SQL_TEMPLATES[QUACK_MUTATION_EVIDENCE_DELETE],
        ["evidence:test"],
    )
    assert admitted == {
        "template_id": QUACK_MUTATION_EVIDENCE_DELETE,
        "parameters": ["evidence:test"],
    }
    assert "sql" not in admitted

    with pytest.raises(QuackOwnerMutationError) as raw_sql:
        mutation_step("DELETE FROM tasks WHERE task_cid = ?", ["task:test"])
    assert raw_sql.value.code == "template_not_allowlisted"

    with pytest.raises(QuackOwnerMutationError) as extra_step_field:
        mutation_operation([{**admitted, "sql": "DELETE FROM evidence_nodes"}])
    assert extra_step_field.value.code == "operation_shape_invalid"

    syntactic_steps = [
        admitted,
        {
            "template_id": QUACK_MUTATION_EVIDENCE_INSERT,
            "parameters": ["evidence:test", "", "task:test", "test", "d", "t", "{}"],
        },
        {
            "template_id": QUACK_MUTATION_DOMAIN_EVENT_INSERT,
            "parameters": ["e", "s", 1, 1, "t", "task:test", "", "", "t", "{}"],
        },
    ]
    request = build_mutation_request(
        steps=syntactic_steps,
        binding=_binding(),
        token=TOKEN,
    )
    assert all(set(step) == {"template_id", "parameters"} for step in request["steps"])
    assert "sql" not in request and "raw_sql" not in request

    with pytest.raises(QuackOwnerMutationError) as extra_request_field:
        validate_mutation_request(
            {**request, "raw_sql": "DROP TABLE tasks"},
            request_id=request["request_id"],
            binding=_binding(),
            token=TOKEN,
        )
    assert extra_request_field.value.code == "request_schema_invalid"


@REQUIRES_DUCKDB
def test_owner_transaction_rolls_back_after_injected_second_write_failure(
    short_root: Path,
) -> None:
    database = short_root / "c.duckdb"
    _seed(database)
    connection = open_duckdb_connection(database, prefer_quack=False)
    request = _transition_request(connection)

    class FailOnRevisionInsert:
        def execute(
            self,
            sql: str,
            parameters: Any = None,
        ) -> Any:
            if sql == MUTATION_SQL_TEMPLATES[QUACK_MUTATION_TASK_REVISION_INSERT]:
                raise RuntimeError("injected owner-side second-write failure")
            if parameters is None:
                return connection.execute(sql)
            return connection.execute(sql, parameters)

    try:
        with pytest.raises(RuntimeError, match="injected owner-side"):
            execute_owner_mutation(
                FailOnRevisionInsert(),
                request,
                refresh_replica=lambda: {"live": True},
            )
        task = connection.execute(
            "SELECT status, revision FROM tasks WHERE task_cid = 'task:test'"
        ).fetchone()
        assert (task[0], task[1]) == ("ready", 1)
        assert connection.execute(
            "SELECT COUNT(*) FROM task_revisions "
            "WHERE task_cid = 'task:test' AND revision = 2"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM domain_events WHERE event_id = ?",
            [request["steps"][-1]["parameters"][0]],
        ).fetchone()[0] == 0
    finally:
        connection.close()


@REQUIRES_DUCKDB
def test_exact_authenticated_replay_is_idempotent(short_root: Path) -> None:
    database = short_root / "c.duckdb"
    inbox = short_root / "i"
    _seed(database)
    connection = open_duckdb_connection(database, prefer_quack=False)
    request = _transition_request(connection)
    refreshes = 0

    def refresh_replica() -> dict[str, Any]:
        nonlocal refreshes
        refreshes += 1
        return {"live": True, "refresh_sequence": refreshes}

    try:
        _publish(inbox, request)
        assert service_mutation_inbox(
            connection,
            inbox=inbox,
            binding=_binding(),
            token=TOKEN,
            refresh_replica=refresh_replica,
        ) == 1
        first_bytes = (
            inbox / f"{request['request_id']}.done.json"
        ).read_bytes()
        first = _read_done(inbox, request)
        assert validate_mutation_result(first, request=request, token=TOKEN) == 1

        replay_rowcounts, replay_observed = execute_owner_mutation(
            connection,
            request,
            refresh_replica=refresh_replica,
        )
        assert replay_rowcounts == [1, 1, 1]
        assert replay_observed["idempotent_replay"] is True
        assert replay_observed["request_id"] == request["request_id"]

        _publish(inbox, request)
        assert service_mutation_inbox(
            connection,
            inbox=inbox,
            binding=_binding(),
            token=TOKEN,
            refresh_replica=refresh_replica,
        ) == 1
        assert (inbox / f"{request['request_id']}.done.json").read_bytes() == first_bytes
        assert _read_done(inbox, request) == first
        assert refreshes == 2
        assert connection.execute(
            "SELECT COUNT(*) FROM task_revisions "
            "WHERE task_cid = 'task:test' AND revision = 2"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM domain_events WHERE event_id = ?",
            [request["steps"][-1]["parameters"][0]],
        ).fetchone()[0] == 1
    finally:
        connection.close()


@REQUIRES_DUCKDB
def test_authenticated_forged_composite_is_not_admitted_as_exact_replay(
    short_root: Path,
) -> None:
    database = short_root / "c.duckdb"
    _seed(database)
    connection = open_duckdb_connection(database, prefer_quack=False)
    request = _transition_request(connection)
    refreshes = 0

    def refresh_replica() -> dict[str, Any]:
        nonlocal refreshes
        refreshes += 1
        return {"live": True, "refresh_sequence": refreshes}

    try:
        execute_owner_mutation(
            connection,
            request,
            refresh_replica=refresh_replica,
        )
        unrelated = connection.execute(
            "SELECT event_id, stream_id, sequence, global_sequence, event_type, "
            "task_cid, attempt_id, session_id, recorded_at, body_json "
            "FROM domain_events WHERE event_type <> 'intent.task_status_changed' "
            "ORDER BY global_sequence LIMIT 1"
        ).fetchone()
        assert unrelated is not None
        forged = build_mutation_request(
            steps=[
                request["steps"][0],
                request["steps"][1],
                {
                    "template_id": QUACK_MUTATION_DOMAIN_EVENT_INSERT,
                    "parameters": [unrelated[index] for index in range(10)],
                },
            ],
            binding=_binding(),
            token=TOKEN,
        )

        with pytest.raises(QuackOwnerMutationError) as rejected:
            execute_owner_mutation(
                connection,
                forged,
                refresh_replica=refresh_replica,
            )
        assert rejected.value.code == "event_binding_invalid"
        assert refreshes == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM task_revisions "
            "WHERE task_cid = 'task:test' AND revision = 2"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM domain_events WHERE event_id = ?",
            [request["steps"][-1]["parameters"][0]],
        ).fetchone()[0] == 1
    finally:
        connection.close()


@REQUIRES_DUCKDB
def test_interrupted_processing_rejects_forged_composite_replay(
    short_root: Path,
) -> None:
    database = short_root / "c.duckdb"
    inbox = short_root / "i"
    _seed(database)
    connection = open_duckdb_connection(database, prefer_quack=False)
    request = _transition_request(connection)
    refreshes = 0

    def refresh_replica() -> dict[str, Any]:
        nonlocal refreshes
        refreshes += 1
        return {"live": True, "refresh_sequence": refreshes}

    try:
        execute_owner_mutation(
            connection,
            request,
            refresh_replica=refresh_replica,
        )
        unrelated = connection.execute(
            "SELECT event_id, stream_id, sequence, global_sequence, event_type, "
            "task_cid, attempt_id, session_id, recorded_at, body_json "
            "FROM domain_events WHERE event_type <> 'intent.task_status_changed' "
            "ORDER BY global_sequence LIMIT 1"
        ).fetchone()
        assert unrelated is not None
        forged = build_mutation_request(
            steps=[
                request["steps"][0],
                request["steps"][1],
                {
                    "template_id": QUACK_MUTATION_DOMAIN_EVENT_INSERT,
                    "parameters": [unrelated[index] for index in range(10)],
                },
            ],
            binding=_binding(),
            token=TOKEN,
        )
        processing_name = f"{forged['request_id']}.processing.json"
        done_name = f"{forged['request_id']}.done.json"
        descriptor = open_mutation_inbox_directory(inbox.resolve())
        try:
            write_envelope_atomic_at(
                descriptor,
                processing_name,
                forged,
            )
        finally:
            os.close(descriptor)

        assert service_mutation_inbox(
            connection,
            inbox=inbox,
            binding=_binding(),
            token=TOKEN,
            refresh_replica=refresh_replica,
        ) == 0
        assert not (inbox / processing_name).exists()
        assert not (inbox / done_name).exists()
        assert refreshes == 1
    finally:
        connection.close()


@REQUIRES_DUCKDB
def test_queue_replay_rejects_rows_from_two_individually_admitted_bundles(
    short_root: Path,
) -> None:
    database = short_root / "c.duckdb"
    _seed(database)
    connection = open_duckdb_connection(database, prefer_quack=False)
    refreshes = 0

    def refresh_replica() -> dict[str, Any]:
        nonlocal refreshes
        refreshes += 1
        return {"live": True, "refresh_sequence": refreshes}

    try:
        first = _queue_backoff_request(connection)
        execute_owner_mutation(
            connection,
            first,
            refresh_replica=refresh_replica,
        )
        first_replay = execute_owner_mutation(
            connection,
            first,
            refresh_replica=refresh_replica,
        )[1]
        assert first_replay["idempotent_replay"] is True

        second = _queue_backoff_request(connection)
        execute_owner_mutation(
            connection,
            second,
            refresh_replica=refresh_replica,
        )
        second_replay = execute_owner_mutation(
            connection,
            second,
            refresh_replica=refresh_replica,
        )[1]
        assert second_replay["idempotent_replay"] is True

        forged = build_mutation_request(
            steps=[second["steps"][0], first["steps"][1]],
            binding=_binding(),
            token=TOKEN,
        )
        with pytest.raises(QuackOwnerMutationError) as rejected:
            execute_owner_mutation(
                connection,
                forged,
                refresh_replica=refresh_replica,
            )
        assert rejected.value.code == "lease_binding_invalid"
        assert refreshes == 4
        lease = connection.execute(
            "SELECT attempt, retry_not_before_ms, release_reason "
            "FROM leases WHERE task_cid = 'task:test'"
        ).fetchone()
        assert (lease[0], lease[1], lease[2]) == tuple(
            second["steps"][0]["parameters"][:3]
        )
    finally:
        connection.close()


@REQUIRES_DUCKDB
@pytest.mark.parametrize(
    ("attack", "expected_code"),
    [("forged_mac", "request_mac_invalid"), ("wrong_generation", "request_binding_invalid")],
)
def test_forged_and_wrong_generation_requests_have_no_effect_or_signed_oracle(
    short_root: Path,
    attack: str,
    expected_code: str,
) -> None:
    database = short_root / "c.duckdb"
    inbox = short_root / "i"
    _seed(database)
    connection = open_duckdb_connection(database, prefer_quack=False)
    if attack == "wrong_generation":
        request = _transition_request(connection, binding=_binding(generation=8))
    else:
        request = _transition_request(connection)
        request["auth_mac"] = "0" * 64

    try:
        with pytest.raises(QuackOwnerMutationError) as rejected:
            validate_mutation_request(
                request,
                request_id=request["request_id"],
                binding=_binding(),
                token=TOKEN,
            )
        assert rejected.value.code == expected_code

        _publish(inbox, request)
        assert service_mutation_inbox(
            connection,
            inbox=inbox,
            binding=_binding(),
            token=TOKEN,
            refresh_replica=lambda: pytest.fail(
                "rejected request must not refresh the read replica"
            ),
        ) == 1
        assert not (inbox / f"{request['request_id']}.done.json").exists()
        assert not (inbox / f"{request['request_id']}.request.json").exists()
        assert not (inbox / f"{request['request_id']}.processing.json").exists()
        task = connection.execute(
            "SELECT status, revision FROM tasks WHERE task_cid = 'task:test'"
        ).fetchone()
        assert (task[0], task[1]) == ("ready", 1)
    finally:
        connection.close()


def test_provider_environment_scrubs_quack_authority_and_raw_secrets() -> None:
    ambient = {
        "PATH": "/usr/bin",
        "SAWM_PROVIDER_SENTINEL": "preserved",
        "QUACK_TOKEN": "raw-owner-token",
        "TENANT_QUACK_SECRET": "raw-tenant-secret",
        STATE_AUTHORITY_MODE_ENV: "quack",
        STATE_QUACK_MUTATION_DIR_ENV: "/tmp/private-owner-inbox",
        STATE_QUACK_MUTATION_BINDING_ENV: '{"generation":7}',
        DATABASE_PROGRAM_JSON_ENV: '{"authority_mode":"quack"}',
    }
    original = dict(ambient)

    cleaned = provider_subprocess_environment(ambient)

    assert cleaned["PATH"] == "/usr/bin"
    assert cleaned["SAWM_PROVIDER_SENTINEL"] == "preserved"
    assert "QUACK_TOKEN" not in cleaned
    assert "TENANT_QUACK_SECRET" not in cleaned
    assert STATE_QUACK_MUTATION_DIR_ENV not in cleaned
    assert STATE_QUACK_MUTATION_BINDING_ENV not in cleaned
    assert all(name not in cleaned for name in DATABASE_PROGRAM_ENV_NAMES)
    assert ambient == original


def test_merge_resolver_real_child_cannot_observe_state_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = tmp_path / "resolver_environment_probe.py"
    denied_names = sorted(
        set(DATABASE_PROGRAM_ENV_NAMES)
        | {
            "IPFS_ACCELERATE_AGENT_QUACK_TOKEN",
            "SAWM_QUACK_TOKEN",
            "TENANT_QUACK_SECRET",
        }
    )
    probe.write_text(
        "import json, os\n"
        f"denied = {denied_names!r}\n"
        "print(json.dumps({'sentinel': os.environ.get('SAWM_PROVIDER_SENTINEL'), "
        "'visible': sorted(name for name in denied if name in os.environ)}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SAWM_PROVIDER_SENTINEL", "preserved")
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "raw-owner-token")
    monkeypatch.setenv("SAWM_QUACK_TOKEN", "raw-owner-token")
    monkeypatch.setenv("TENANT_QUACK_SECRET", "raw-tenant-secret")
    monkeypatch.setenv(STATE_AUTHORITY_MODE_ENV, "quack")
    monkeypatch.setenv(STATE_QUACK_MUTATION_DIR_ENV, "/tmp/private-owner-inbox")
    monkeypatch.setenv(STATE_QUACK_MUTATION_BINDING_ENV, '{"generation":7}')
    monkeypatch.setenv(DATABASE_PROGRAM_JSON_ENV, '{"authority_mode":"quack"}')

    result = invoke_llm_resolver(
        {"found": True, "repo_root": str(tmp_path), "prompt": "probe"},
        command_template=(
            f"{shlex.quote(sys.executable)} {shlex.quote(str(probe))}"
        ),
        timeout_seconds=5,
    )

    assert result["applied"] is True
    child_observation = json.loads(result["llm_stdout"])
    assert child_observation == {"sentinel": "preserved", "visible": []}


def test_retired_token_handoff_is_absent_from_real_provider_child(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "quack-owner"
    vault = TokenVault(state_dir)
    handle = vault.mint(secret_handle="env://SAWM_QUACK_TOKEN", generation=2)
    token = vault.resolve(handle.handle)
    handoff_paths = tuple(state_dir.glob("*.quack-token"))
    assert len(handoff_paths) == 1

    retired = retire_token_handoff(
        state_dir=state_dir,
        secret_handle=handle.handle,
        expected_token=token,
    )
    assert retired["retired"] is True
    assert retired["already_absent"] is False
    replay = retire_token_handoff(
        state_dir=state_dir,
        secret_handle=handle.handle,
        expected_token=token,
    )
    assert replay["retired"] is True
    assert replay["already_absent"] is True

    provider_environment = provider_subprocess_environment(
        {
            "PATH": os.environ.get("PATH", ""),
            "IPFS_ACCELERATE_AGENT_QUACK_TOKEN": token,
            "SAWM_QUACK_TOKEN": token,
            STATE_QUACK_MUTATION_DIR_ENV: str(state_dir / "mutations"),
            "SAWM_PROVIDER_SENTINEL": "preserved",
        }
    )
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, os, pathlib, sys; "
                "path = pathlib.Path(sys.argv[1]); "
                "print(json.dumps({'handoff_exists': path.exists(), "
                "'quack_token_visible': "
                "'IPFS_ACCELERATE_AGENT_QUACK_TOKEN' in os.environ, "
                "'sawm_token_visible': 'SAWM_QUACK_TOKEN' in os.environ, "
                "'sentinel': os.environ.get('SAWM_PROVIDER_SENTINEL')}))"
            ),
            str(handoff_paths[0]),
        ],
        check=True,
        capture_output=True,
        env=provider_environment,
        text=True,
    )
    assert json.loads(child.stdout) == {
        "handoff_exists": False,
        "quack_token_visible": False,
        "sawm_token_visible": False,
        "sentinel": "preserved",
    }
    vault.destroy()


def test_token_handoff_mismatch_fails_closed_without_unlinking(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "quack-owner"
    vault = TokenVault(state_dir)
    handle = vault.mint(secret_handle="env://SAWM_QUACK_TOKEN", generation=2)
    handoff = next(state_dir.glob("*.quack-token"))

    with pytest.raises(
        QuackStateServerTokenError,
        match="does not match the authenticated owner",
    ):
        retire_token_handoff(
            state_dir=state_dir,
            secret_handle=handle.handle,
            expected_token="wrong-token-value",
        )

    assert handoff.is_file()
    vault.destroy()
