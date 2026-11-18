"""Typed, repository-backed Quack state-owner command tests."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
    TaskSourceCompletionError,
    TaskSourceConflictError,
    TaskSourceIntegrityError,
    TaskSourceUnknownOutcomeError,
    execute_quack_owner_command,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    QUACK_OWNER_COMMAND_COMPARE_AND_SET_STATUS,
    QUACK_OWNER_COMMAND_MAX_BYTES,
    QUACK_OWNER_COMMAND_REARM_BLOCKED_TASK,
    QUACK_OWNER_COMMAND_RECORD_EVIDENCE,
    QUACK_OWNER_COMMAND_RECORD_QUEUE_BACKOFF,
    QUACK_OWNER_COMMAND_RECORD_QUEUE_BACKOFF_AND_CAS_STATUS,
    QUACK_OWNER_COMMAND_RECORD_QUEUE_RETRY,
    QUACK_OWNER_COMMAND_RECORD_VALIDATION_RESULT,
    QUACK_OWNER_COMMAND_RECOVER_LEFTOVER_WAIT_DEFERRAL_BUDGET,
    QUACK_OWNER_COMMAND_REQUEST_SCHEMA,
    QUACK_OWNER_COMMAND_RESPONSE_SCHEMA,
    DuckDBConnection,
    DuckDBConnectionPolicyError,
    QuackOwnerCommandRemoteError,
    open_duckdb_connection,
    quack_owner_command_response,
    quack_owner_command_signature,
    submit_quack_owner_command,
    validate_quack_owner_command,
    validate_quack_owner_command_request,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    IntentRepository,
    IntentRepositoryConflictError,
)


def _materialize_one_task(path: Path) -> None:
    with DatabaseTaskSource(path) as source:
        source.materialize(
            {
                "repository_tree_id": "tree:typed-owner-test",
                "objectives": [
                    {
                        "goal_id": "G1",
                        "goal_cid": "goal:typed-owner-test",
                        "objective_id": "objective:typed-owner-test",
                        "title": "Typed owner commands",
                    }
                ],
                "taskboard": [
                    {
                        "task_id": "T1",
                        "task_cid": "task:typed-owner-test",
                        "goal_cid": "goal:typed-owner-test",
                        "status": "ready",
                    }
                ],
            }
        )


def _signed_request(*, now_ms: int = 1_000_000) -> dict[str, object]:
    request: dict[str, object] = {
        "schema": QUACK_OWNER_COMMAND_REQUEST_SCHEMA,
        "request_id": "a" * 32,
        "issued_at_ms": now_ms,
        "writer_identity": "supervisor-process:123",
        "store_id": "data/control.duckdb",
        "store_generation": "generation-1",
        "command": QUACK_OWNER_COMMAND_RECORD_EVIDENCE,
        "payload": {
            "task_cid": "task:typed-owner-test",
            "evidence_kind": "validation",
            "digest": "sha256:" + ("1" * 64),
            "body": {"producer": "pytest"},
        },
    }
    request["signature"] = quack_owner_command_signature(request, "token_value_123")
    return request


def test_owner_request_is_closed_typed_and_has_no_sql() -> None:
    request = _signed_request()
    command, payload = validate_quack_owner_command_request(
        request,
        token="token_value_123",
        expected_request_id="a" * 32,
        expected_store_id="data/control.duckdb",
        expected_store_generation="generation-1",
        now_ms=1_000_000,
    )
    assert command == QUACK_OWNER_COMMAND_RECORD_EVIDENCE
    assert payload["task_cid"] == "task:typed-owner-test"
    assert "sql" not in request
    assert "parameters" not in request
    with pytest.raises(DuckDBConnectionPolicyError, match="closed schema"):
        validate_quack_owner_command(
            command,
            {**payload, "sql": "DELETE FROM tasks"},
        )


def test_request_authentication_binds_generation_freshness_and_hmac() -> None:
    request = _signed_request()
    for changes, message in (
        ({"store_generation": "generation-2"}, "binding"),
        ({"issued_at_ms": 600_000}, "stale"),
        ({"signature": "0" * 64}, "authorization"),
    ):
        changed = {**request, **changes}
        with pytest.raises(DuckDBConnectionPolicyError, match=message):
            validate_quack_owner_command_request(
                changed,
                token="token_value_123",
                expected_request_id="a" * 32,
                expected_store_id="data/control.duckdb",
                expected_store_generation="generation-1",
                now_ms=1_000_000,
            )


def test_submit_round_trips_bound_typed_result(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    inbox = tmp_path / "mutations"
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR", str(inbox))
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "token_value_123")
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_QUACK_TOKEN_FILE",
        str(tmp_path / "round-trip.quack-token"),
    )
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", "data/control.duckdb")
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION", "generation-1")

    def owner() -> None:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            requests = list(inbox.glob("*.request.json"))
            if requests:
                request_path = requests[0]
                request = json.loads(request_path.read_text(encoding="utf-8"))
                assert set(request).isdisjoint({"sql", "parameters"})
                response = quack_owner_command_response(
                    request,
                    token="token_value_123",
                    result={"schema": "typed-result@1", "value": 7},
                )
                request_path.with_name(
                    request_path.name.replace(".request.json", ".done.json")
                ).write_text(json.dumps(response), encoding="utf-8")
                return
            time.sleep(0.01)
        raise AssertionError("typed command request did not arrive")

    thread = threading.Thread(target=owner)
    thread.start()
    result = submit_quack_owner_command(
        QUACK_OWNER_COMMAND_RECORD_EVIDENCE,
        {
            "task_cid": "task:typed-owner-test",
            "evidence_kind": "validation",
            "digest": "sha256:" + ("2" * 64),
        },
        timeout_seconds=2,
    )
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert not list(inbox.glob("*.request.json"))
    assert result == {"schema": "typed-result@1", "value": 7}


def test_stable_request_id_rejects_response_for_different_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inbox = tmp_path / "mutations"
    token = "token_value_123"
    store_id = "data/control.duckdb"
    store_generation = "generation-1"
    request_id = "b" * 32
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR", str(inbox))
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", token)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", store_id)
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION",
        store_generation,
    )
    inbox.mkdir(parents=True)
    prior_request = {
        "schema": QUACK_OWNER_COMMAND_REQUEST_SCHEMA,
        "request_id": request_id,
        "issued_at_ms": int(time.time() * 1000),
        "writer_identity": "supervisor-process:123",
        "store_id": store_id,
        "store_generation": store_generation,
        "command": QUACK_OWNER_COMMAND_RECORD_EVIDENCE,
        "payload": {
            "task_cid": "task:typed-owner-test",
            "evidence_kind": "validation",
            "digest": "sha256:" + ("1" * 64),
        },
    }
    prior_request["signature"] = quack_owner_command_signature(
        prior_request,
        token,
    )
    stale_response = quack_owner_command_response(
        prior_request,
        token=token,
        result={"schema": "typed-result@1", "value": "stale"},
    )
    (inbox / f"{request_id}.done.json").write_text(
        json.dumps(stale_response),
        encoding="utf-8",
    )

    with pytest.raises(QuackOwnerCommandRemoteError) as caught:
        submit_quack_owner_command(
            QUACK_OWNER_COMMAND_RECORD_EVIDENCE,
            {
                "task_cid": "task:typed-owner-test",
                "evidence_kind": "validation",
                "digest": "sha256:" + ("2" * 64),
            },
            timeout_seconds=0.1,
            request_id=request_id,
        )
    assert caught.value.code == "command_timeout_unknown_outcome"
    assert caught.value.request_id == request_id
    assert isinstance(caught.value.__cause__, DuckDBConnectionPolicyError)
    assert (inbox / f"{request_id}.request.json").is_file()


def test_stable_request_id_rejects_stale_failure_for_prior_publication(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inbox = tmp_path / "mutations"
    token = "token_value_123"
    store_id = "data/control.duckdb"
    store_generation = "generation-1"
    request_id = "c" * 32
    payload = {
        "task_cid": "task:typed-owner-test",
        "evidence_kind": "validation",
        "digest": "sha256:" + ("3" * 64),
    }
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR", str(inbox))
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", token)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", store_id)
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION",
        store_generation,
    )
    inbox.mkdir(parents=True)
    prior_request = {
        "schema": QUACK_OWNER_COMMAND_REQUEST_SCHEMA,
        "request_id": request_id,
        "issued_at_ms": int(time.time() * 1000) - 1_000,
        "writer_identity": "supervisor-process:123",
        "store_id": store_id,
        "store_generation": store_generation,
        "command": QUACK_OWNER_COMMAND_RECORD_EVIDENCE,
        "payload": payload,
    }
    prior_request["signature"] = quack_owner_command_signature(
        prior_request,
        token,
    )
    stale_failure = quack_owner_command_response(
        prior_request,
        token=token,
        error_code="prior_definitive_failure",
        error_message="the prior publication was rejected",
    )
    (inbox / f"{request_id}.done.json").write_text(
        json.dumps(stale_failure),
        encoding="utf-8",
    )

    with pytest.raises(QuackOwnerCommandRemoteError) as caught:
        submit_quack_owner_command(
            QUACK_OWNER_COMMAND_RECORD_EVIDENCE,
            payload,
            timeout_seconds=0.1,
            request_id=request_id,
        )
    assert caught.value.code == "command_timeout_unknown_outcome"
    assert caught.value.request_id == request_id
    assert isinstance(caught.value.__cause__, DuckDBConnectionPolicyError)
    replacement = json.loads(
        (inbox / f"{request_id}.request.json").read_text(encoding="utf-8")
    )
    assert replacement["signature"] != prior_request["signature"]


def test_submit_rejects_forged_owner_response(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    inbox = tmp_path / "mutations"
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR", str(inbox))
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "token_value_123")
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_QUACK_TOKEN_FILE",
        str(tmp_path / "forged-response.quack-token"),
    )
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", "data/control.duckdb")
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION", "generation-1")

    def forger() -> None:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            requests = list(inbox.glob("*.request.json"))
            if requests:
                request_path = requests[0]
                request = json.loads(request_path.read_text(encoding="utf-8"))
                forged = {
                    "schema": QUACK_OWNER_COMMAND_RESPONSE_SCHEMA,
                    "request_id": request["request_id"],
                    "command": request["command"],
                    "store_id": request["store_id"],
                    "store_generation": request["store_generation"],
                    "ok": True,
                    "result": {"schema": "forged-result@1", "changed": True},
                    "signature": "0" * 64,
                }
                request_path.with_name(
                    request_path.name.replace(".request.json", ".done.json")
                ).write_text(json.dumps(forged), encoding="utf-8")
                return
            time.sleep(0.01)
        raise AssertionError("typed command request did not arrive")

    thread = threading.Thread(target=forger)
    thread.start()
    with pytest.raises(QuackOwnerCommandRemoteError) as caught:
        submit_quack_owner_command(
            QUACK_OWNER_COMMAND_RECORD_EVIDENCE,
            {
                "task_cid": "task:typed-owner-test",
                "evidence_kind": "validation",
                "digest": "sha256:" + ("4" * 64),
            },
            timeout_seconds=0.2,
        )
    assert caught.value.code == "command_timeout_unknown_outcome"
    assert len(caught.value.request_id) == 32
    assert isinstance(caught.value.__cause__, DuckDBConnectionPolicyError)
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert len(list(inbox.glob("*.request.json"))) == 1


@pytest.mark.parametrize("response_kind", ["symlink", "oversize"])
def test_submit_rejects_unsafe_owner_response_file(
    response_kind: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    inbox = tmp_path / "mutations"
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR", str(inbox))
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "token_value_123")
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_QUACK_TOKEN_FILE",
        str(tmp_path / "unsafe-response.quack-token"),
    )
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", "data/control.duckdb")
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION", "generation-1")

    def unsafe_writer() -> None:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            requests = list(inbox.glob("*.request.json"))
            if requests:
                request_path = requests[0]
                done_path = request_path.with_name(
                    request_path.name.replace(".request.json", ".done.json")
                )
                if response_kind == "symlink":
                    target = tmp_path / "forged-response.json"
                    target.write_text("{}", encoding="utf-8")
                    done_path.symlink_to(target)
                else:
                    done_path.write_bytes(b"x" * (QUACK_OWNER_COMMAND_MAX_BYTES + 1))
                return
            time.sleep(0.01)
        raise AssertionError("typed command request did not arrive")

    thread = threading.Thread(target=unsafe_writer)
    thread.start()
    with pytest.raises(QuackOwnerCommandRemoteError) as caught:
        submit_quack_owner_command(
            QUACK_OWNER_COMMAND_RECORD_EVIDENCE,
            {
                "task_cid": "task:typed-owner-test",
                "evidence_kind": "validation",
                "digest": "sha256:" + ("5" * 64),
            },
            timeout_seconds=0.2,
        )
    assert caught.value.code == "command_timeout_unknown_outcome"
    assert len(caught.value.request_id) == 32
    assert isinstance(caught.value.__cause__, DuckDBConnectionPolicyError)
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert len(list(inbox.glob("*.request.json"))) == 1


def test_bound_repository_command_is_atomic_and_restart_idempotent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "control.duckdb"
    _materialize_one_task(path)
    owner_connection = open_duckdb_connection(path)
    try:
        first_repo = IntentRepository(
            path,
            bound_connection=owner_connection,
            install_schema=False,
            owner_id="owner:typed-test",
            session_id="session:first",
        )
        kwargs = {
            "request_id": "b" * 32,
            "store_id": "data/control.duckdb",
            "store_generation": "generation-1",
        }
        payload = {
            "task_cid": "task:typed-owner-test",
            "evidence_kind": "validation",
            "digest": "sha256:" + ("3" * 64),
        }
        first = execute_quack_owner_command(
            first_repo,
            QUACK_OWNER_COMMAND_RECORD_EVIDENCE,
            payload,
            **kwargs,
        )
        first_repo.close()
        owner_connection.close()

        # A new connection and repository model an owner process restart while
        # the durable idempotency record remains in the authoritative database.
        owner_connection = open_duckdb_connection(path)
        second_repo = IntentRepository(
            path,
            bound_connection=owner_connection,
            install_schema=False,
            owner_id="owner:typed-test",
            session_id="session:second",
        )
        replay = execute_quack_owner_command(
            second_repo,
            QUACK_OWNER_COMMAND_RECORD_EVIDENCE,
            payload,
            **kwargs,
        )
        assert replay == first
        event_count = owner_connection.execute(
            "SELECT COUNT(*) FROM domain_events WHERE event_type = ?",
            ["intent.evidence_recorded"],
        ).fetchone()
        assert event_count is not None and int(event_count[0]) == 1
        record_count = owner_connection.execute(
            "SELECT COUNT(*) FROM idempotency_records WHERE idempotency_key = ?",
            ["quack-owner-command:" + ("b" * 32)],
        ).fetchone()
        assert record_count is not None and int(record_count[0]) == 1
        second_repo.close()
        # Repository lifecycle never closes the owner's injected connection.
        assert owner_connection.execute("SELECT 1").fetchone()[0] == 1
    finally:
        owner_connection.close()


def test_owner_guarded_retry_preserves_exact_absolute_deadline(
    tmp_path: Path,
) -> None:
    path = tmp_path / "control.duckdb"
    _materialize_one_task(path)
    owner_connection = open_duckdb_connection(path)
    try:
        repository = IntentRepository(
            path,
            bound_connection=owner_connection,
            install_schema=False,
            owner_id="owner:typed-test",
            session_id="session:absolute-deadline",
        )
        repository._clock_ms = lambda: 1_000_123  # type: ignore[method-assign]
        task = repository.get_task("task:typed-owner-test")
        assert task is not None
        claim_receipt = {
            "operation": "database_claim",
            "attempt_id": "attempt:absolute-deadline",
        }
        repository.cas_task_status(
            task_cid="task:typed-owner-test",
            expected_revision=int(task["revision"]),
            new_status="in_progress",
            receipt=claim_receipt,
        )
        claimed = repository.get_task("task:typed-owner-test")
        assert claimed is not None
        reason = "database_portal_retry:attempt:absolute-deadline:capacity"
        transition = {
            "operation": "database_portal_capacity_retry",
            "queue_reason": reason,
            "queue_reused": False,
            "queue_receipt": {},
            "retry_not_before_ms": 0,
        }
        payload = {
            "task_cid": "task:typed-owner-test",
            "expected_revision": int(claimed["revision"]),
            "expected_control_receipt": claim_receipt,
            "status": "retrying",
            "receipt": transition,
            "delay_ms": 999_877,
            "reason": reason,
            "exact_retry_not_before_ms": 2_000_000,
        }
        assert validate_quack_owner_command(
            QUACK_OWNER_COMMAND_RECORD_QUEUE_BACKOFF_AND_CAS_STATUS,
            payload,
        )["exact_retry_not_before_ms"] == 2_000_000
        result = execute_quack_owner_command(
            repository,
            QUACK_OWNER_COMMAND_RECORD_QUEUE_BACKOFF_AND_CAS_STATUS,
            payload,
            request_id="d" * 32,
            store_id="data/control.duckdb",
            store_generation="generation-1",
        )
        assert result["retry_not_before_ms"] == 2_000_000
        assert result["transition_receipt"]["retry_not_before_ms"] == 2_000_000
        assert result["cas_result"]["task"]["body"]["completion_receipt"][
            "retry_not_before_ms"
        ] == 2_000_000
        with pytest.raises(
            DuckDBConnectionPolicyError,
            match="non-negative integer",
        ):
            validate_quack_owner_command(
                QUACK_OWNER_COMMAND_RECORD_QUEUE_BACKOFF_AND_CAS_STATUS,
                {**payload, "exact_retry_not_before_ms": -1},
            )
        repository.close()
    finally:
        owner_connection.close()


def test_owner_generic_queue_status_rejects_leftover_wait_recovery(
    tmp_path: Path,
) -> None:
    path = tmp_path / "control.duckdb"
    _materialize_one_task(path)
    owner_connection = open_duckdb_connection(path)
    try:
        repository = IntentRepository(
            path,
            bound_connection=owner_connection,
            install_schema=False,
            owner_id="owner:typed-test",
            session_id="session:reject-leftover-wait",
        )
        task = repository.get_task("task:typed-owner-test")
        assert task is not None
        claim_receipt = {"operation": "database_claim"}
        repository.cas_task_status(
            task_cid="task:typed-owner-test",
            expected_revision=int(task["revision"]),
            new_status="in_progress",
            receipt=claim_receipt,
        )
        claimed = repository.get_task("task:typed-owner-test")
        assert claimed is not None
        blocked_receipt = {
            "operation": "database_portal_typed_deferral_budget_exhausted"
        }
        repository.cas_task_status(
            task_cid="task:typed-owner-test",
            expected_revision=int(claimed["revision"]),
            new_status="blocked",
            receipt=blocked_receipt,
        )
        blocked = repository.get_task("task:typed-owner-test")
        assert blocked is not None
        payload = {
            "task_cid": "task:typed-owner-test",
            "expected_revision": int(blocked["revision"]),
            "expected_control_receipt": blocked_receipt,
            "status": "retrying",
            "receipt": {
                "operation": (
                    "database_portal_leftover_wait_"
                    "deferral_budget_retry_recovery"
                )
            },
            "delay_ms": 0,
            "reason": "leftover_wait_deferral_budget_cleared",
        }

        with pytest.raises(
            (TaskSourceConflictError, TaskSourceIntegrityError),
        ):
            execute_quack_owner_command(
                repository,
                QUACK_OWNER_COMMAND_RECORD_QUEUE_BACKOFF_AND_CAS_STATUS,
                payload,
                request_id="e" * 32,
                store_id="data/control.duckdb",
                store_generation="generation-1",
            )

        observed = repository.get_task("task:typed-owner-test")
        assert observed is not None
        assert observed["status"] == "blocked"
        assert int(observed["revision"]) == int(blocked["revision"])
        assert repository.get_queue_entry("task:typed-owner-test") is None
        repository.close()
    finally:
        owner_connection.close()


def test_owner_leftover_wait_recovery_command_rejects_incomplete_seed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "control.duckdb"
    _materialize_one_task(path)
    owner_connection = open_duckdb_connection(path)
    try:
        repository = IntentRepository(
            path,
            bound_connection=owner_connection,
            install_schema=False,
            owner_id="owner:typed-test",
            session_id="session:leftover-wait-dedicated",
        )
        task = repository.get_task("task:typed-owner-test")
        assert task is not None
        repository.cas_task_status(
            task_cid="task:typed-owner-test",
            expected_revision=int(task["revision"]),
            new_status="in_progress",
            receipt={"operation": "database_claim"},
        )
        claimed = repository.get_task("task:typed-owner-test")
        assert claimed is not None
        blocked_receipt = {
            "operation": "database_portal_typed_deferral_budget_exhausted"
        }
        repository.cas_task_status(
            task_cid="task:typed-owner-test",
            expected_revision=int(claimed["revision"]),
            new_status="blocked",
            receipt=blocked_receipt,
        )
        blocked = repository.get_task("task:typed-owner-test")
        assert blocked is not None
        payload = {
            "task_cid": "task:typed-owner-test",
            "expected_revision": int(blocked["revision"]),
            "expected_control_receipt": blocked_receipt,
            "status": "retrying",
            "receipt": {
                "operation": (
                    "database_portal_leftover_wait_"
                    "deferral_budget_retry_recovery"
                )
            },
            "delay_ms": 0,
            "reason": "leftover_wait_deferral_budget_cleared",
        }

        with pytest.raises(
            TaskSourceConflictError,
            match="exhausted typed-deferral task remains blocked",
        ):
            execute_quack_owner_command(
                repository,
                QUACK_OWNER_COMMAND_RECOVER_LEFTOVER_WAIT_DEFERRAL_BUDGET,
                payload,
                request_id="f" * 32,
                store_id="data/control.duckdb",
                store_generation="generation-1",
            )

        observed = repository.get_task("task:typed-owner-test")
        assert observed is not None
        assert observed["status"] == "blocked"
        assert int(observed["revision"]) == int(blocked["revision"])
        assert repository.get_queue_entry("task:typed-owner-test") is None
        repository.close()
    finally:
        owner_connection.close()


def test_owner_dispatches_exact_five_live_mutations(tmp_path: Path) -> None:
    path = tmp_path / "control.duckdb"
    _materialize_one_task(path)
    owner_connection = open_duckdb_connection(path)
    try:
        repository = IntentRepository(
            path,
            bound_connection=owner_connection,
            install_schema=False,
            owner_id="owner:typed-test",
            session_id="session:five-commands",
        )
        bindings = {
            "store_id": "data/control.duckdb",
            "store_generation": "generation-1",
        }
        calls = (
            (
                QUACK_OWNER_COMMAND_RECORD_QUEUE_BACKOFF,
                {
                    "task_cid": "task:typed-owner-test",
                    "delay_ms": 1,
                    "reason": "retry-later",
                    "selection_penalty": 1,
                },
            ),
            (
                QUACK_OWNER_COMMAND_RECORD_QUEUE_RETRY,
                {"task_cid": "task:typed-owner-test"},
            ),
            (
                QUACK_OWNER_COMMAND_RECORD_EVIDENCE,
                {
                    "task_cid": "task:typed-owner-test",
                    "evidence_kind": "validation",
                    "digest": "sha256:" + ("6" * 64),
                },
            ),
            (
                QUACK_OWNER_COMMAND_RECORD_VALIDATION_RESULT,
                {
                    "task_cid": "task:typed-owner-test",
                    "outcome": "passed",
                    "evidence_digest": "sha256:" + ("7" * 64),
                    "argv": ["pytest", "-q"],
                },
            ),
            (
                QUACK_OWNER_COMMAND_COMPARE_AND_SET_STATUS,
                {
                    "task_cid_or_alias": "task:typed-owner-test",
                    "expected_revision": 1,
                    "status": "running",
                },
            ),
        )
        results = [
            execute_quack_owner_command(
                repository,
                command,
                payload,
                request_id=f"{index:x}" * 32,
                **bindings,
            )
            for index, (command, payload) in enumerate(calls, start=1)
        ]
        assert all(result.get("schema") for result in results)
        assert results[-1]["task"]["status"] == "running"
        assert (
            len(
                owner_connection.execute(
                    "SELECT idempotency_key FROM idempotency_records "
                    "WHERE command_kind LIKE 'record_%' OR "
                    "command_kind = 'compare_and_set_status'"
                ).fetchall()
            )
            == 5
        )
        repository.close()
    finally:
        owner_connection.close()


def test_owner_rearms_blocked_task_without_client_revision(tmp_path: Path) -> None:
    path = tmp_path / "control.duckdb"
    _materialize_one_task(path)
    with IntentRepository(
        path,
        install_schema=False,
        owner_id="owner:typed-test",
        session_id="session:block",
    ) as repository:
        task = repository.get_task("task:typed-owner-test")
        assert task is not None
        repository.block_task(
            task_cid="task:typed-owner-test",
            blocker_kind="dependency",
            blocker_id="task:other",
            reason="waiting",
            expected_revision=int(task["revision"]),
        )
    owner_connection = open_duckdb_connection(path)
    try:
        repository = IntentRepository(
            path,
            bound_connection=owner_connection,
            install_schema=False,
            owner_id="owner:typed-test",
            session_id="session:rearm",
        )
        result = execute_quack_owner_command(
            repository,
            QUACK_OWNER_COMMAND_REARM_BLOCKED_TASK,
            {
                "task_cid_or_alias": "task:typed-owner-test",
                "receipt": {"operation": "database_declared_outputs_on_head_rearm"},
            },
            request_id="c" * 32,
            store_id="data/control.duckdb",
            store_generation="generation-1",
        )
        assert result["changed"] is True
        assert result["previous_status"] == "blocked"
        assert result["task"]["status"] == "retrying"
        replay = execute_quack_owner_command(
            repository,
            QUACK_OWNER_COMMAND_REARM_BLOCKED_TASK,
            {"task_cid_or_alias": "task:typed-owner-test"},
            request_id="d" * 32,
            store_id="data/control.duckdb",
            store_generation="generation-1",
        )
        assert replay["changed"] is False
        assert replay["task"]["status"] == "retrying"
        repository.close()
    finally:
        owner_connection.close()


def test_owner_rearm_preserves_terminal_task_as_idempotent_noop(tmp_path: Path) -> None:
    path = tmp_path / "control.duckdb"
    _materialize_one_task(path)
    owner_connection = open_duckdb_connection(path)
    try:
        owner_connection.execute(
            "UPDATE tasks SET status = 'completed' WHERE task_cid = ?",
            ["task:typed-owner-test"],
        )
        repository = IntentRepository(
            path,
            bound_connection=owner_connection,
            install_schema=False,
            owner_id="owner:typed-test",
            session_id="session:terminal-rearm",
        )
        result = execute_quack_owner_command(
            repository,
            QUACK_OWNER_COMMAND_REARM_BLOCKED_TASK,
            {
                "task_cid_or_alias": "task:typed-owner-test",
                "receipt": {"operation": "database_declared_outputs_on_head_rearm"},
            },
            request_id="e" * 32,
            store_id="data/control.duckdb",
            store_generation="generation-1",
        )
        assert result["changed"] is False
        assert result["previous_status"] == "completed"
        assert result["task"]["status"] == "completed"
        repository.close()
    finally:
        owner_connection.close()


def test_quack_transport_sql_mutations_fail_closed() -> None:
    class NeverExecute:
        description = None

        def execute(self, *_args, **_kwargs):
            raise AssertionError("raw Quack mutation reached the connection")

        def close(self) -> None:
            return None

    connection = DuckDBConnection.wrap(NeverExecute())
    connection._default_catalog = "control_plane"  # noqa: SLF001
    with pytest.raises(
        DuckDBConnectionPolicyError,
        match="quack owner mutation requires an explicit transaction",
    ):
        connection.execute(
            "UPDATE tasks SET status = ? WHERE task_cid = ?",
            ["blocked", "task:typed-owner-test"],
        )
    with pytest.raises(DuckDBConnectionPolicyError, match="executemany"):
        connection.executemany("INSERT INTO tasks VALUES (?)", [["x"]])
    with pytest.raises(DuckDBConnectionPolicyError, match="scripts"):
        connection.executescript("DELETE FROM tasks")


@pytest.mark.parametrize("unknown_code", [
    "command_timeout_unknown_outcome",
    "read_replica_refresh_unknown_outcome",
    "unknown_external_outcome",
])
def test_database_task_source_maps_typed_owner_failures(monkeypatch, unknown_code) -> None:
    source = DatabaseTaskSource(
        "quack:127.0.0.1:45123",
        install_schema=False,
    )

    def conflict(*_args, **_kwargs):
        raise QuackOwnerCommandRemoteError("conflict", "stale CAS")

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source.submit_quack_owner_command",
        conflict,
    )
    with pytest.raises(TaskSourceConflictError, match="stale CAS"):
        source.compare_and_set_status("task:1", 1, "running")

    def completion(*_args, **_kwargs):
        raise QuackOwnerCommandRemoteError("completion_refused", "current evidence required")

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source.submit_quack_owner_command",
        completion,
    )
    with pytest.raises(TaskSourceCompletionError, match="current evidence"):
        source.compare_and_set_status("task:1", 1, "completed")

    def unknown(*_args, **_kwargs):
        raise QuackOwnerCommandRemoteError(
            unknown_code,
            "reconcile exact request",
            request_id="e" * 32,
        )

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source.submit_quack_owner_command",
        unknown,
    )
    with pytest.raises(TaskSourceUnknownOutcomeError) as raised:
        source.record_queue_retry(task_cid="task:1")
    assert raised.value.request_id == "e" * 32
    source.close()


def test_block_and_unblock_use_revision_predicates(tmp_path: Path) -> None:
    path = tmp_path / "control.duckdb"
    _materialize_one_task(path)
    with IntentRepository(path, install_schema=False) as repository:
        task = repository.get_task("task:typed-owner-test")
        assert task is not None
        revision = int(task["revision"])
        with pytest.raises(IntentRepositoryConflictError, match="blocking"):
            repository.block_task(
                task_cid="task:typed-owner-test",
                blocker_kind="dependency",
                blocker_id="task:other",
                reason="waiting",
                expected_revision=revision + 1,
            )
        repository.block_task(
            task_cid="task:typed-owner-test",
            blocker_kind="dependency",
            blocker_id="task:other",
            reason="waiting",
            expected_revision=revision,
        )
        blocked = repository.get_task("task:typed-owner-test")
        assert blocked is not None and blocked["status"] == "blocked"
        with pytest.raises(IntentRepositoryConflictError, match="unblocking"):
            repository.unblock_task(
                task_cid="task:typed-owner-test",
                expected_revision=revision,
            )
        repository.unblock_task(
            task_cid="task:typed-owner-test",
            expected_revision=int(blocked["revision"]),
        )
        ready = repository.get_task("task:typed-owner-test")
        assert ready is not None and ready["status"] == "ready"


# Current-reason recovery controls are pure fixtures, not live task authority.
def _false_terminal_guard_cases():
    repair = "false_terminal_blocked_supervisor_bug"
    local = "local_validation_pending_native_admission"
    return [
        ("current-infrastructure", {"blocked_reason": "quack_transport_unavailable: socket closed"}, repair, repair),
        ("current-process-absence", {"reason": "ProcessLookupError: original child exited"}, repair, repair),
        ("historical-positive-only", {"blocked_reason": "waiting for reviewed definition",
            "history": [{"reason": "quack_transport_unavailable"}]}, None, None),
        ("description-positive-only", {"description": "fix quack_transport_unavailable later"}, None, None),
        ("current-reason-precedes-receipt", {"blocked_reason": "waiting for reviewed definition",
            "completion_receipt": {"operation": "quack_transport_unavailable"}}, None, None),
        ("current-UNKNOWN", {"blocked_reason": "quack_transport_unavailable",
            "completion_receipt": {"failure_kind": "administrative_operator_stop_outcome_unknown",
                "automatic_retry_admitted": False}}, None, None),
        ("current-callback-UNKNOWN", {"blocked_reason": "quack_transport_unavailable",
            "custody_observation": {"callback_outcome": "unknown"}}, None, None),
        ("explicit-retry-denial", {"blocked_reason": "quack_transport_unavailable",
            "automatic_retry_admitted": False}, None, None),
        ("malformed-retry-denial", {"blocked_reason": "quack_transport_unavailable",
            "automatic_retry_admitted": "false"}, None, None),
        ("exhausted-counter", {"blocked_reason": "quack_transport_unavailable",
            "attempt_budget": {"spent_native_ordinal": 2, "max_task_attempts": 2}}, None, None),
        ("exhausted-current-reason", {"blocked_reason": "typed_portal_deferral_budget_exhausted"}, None, None),
        ("retained-owner", {"blocked_reason": "quack_transport_unavailable",
            "details": {"retained_attempt_settlement_required": True}}, None, None),
        ("protected-path", {"blocked_reason": "implementation_protected_path_mutated"}, None, None),
        ("local-validation", {"blocked_reason": "local_validation_pending_native_admission"}, None, local),
        ("history-negative-is-not-current", {"blocked_reason": "quack_transport_unavailable",
            "history": [{"provider_outcome": "unknown", "automatic_retry_admitted": False}]}, repair, repair),
        ("selected-retained-UNKNOWN", {"blocked_reason": "quack_transport_unavailable",
            "retained_execution_outcomes": [{"task_cid": "task:authored", "task_revision": 13,
                "completion_receipt": {"failure_kind": "administrative_operator_stop_outcome_unknown",
                    "automatic_retry_admitted": False}, "sha256": "0" * 64}]}, None, None),
        ("uppercase-provider-UNKNOWN", {"blocked_reason": "quack_transport_unavailable", "provider_outcome": " UNKNOWN "}, None, None),
        ("uppercase-callback-UNKNOWN", {"blocked_reason": "quack_transport_unavailable", "callback_outcome": "UNKNOWN"}, None, None),
        ("uppercase-effect-UNKNOWN", {"blocked_reason": "quack_transport_unavailable", "provider_effect_state": "UNKNOWN"}, None, None),
        ("uppercase-state-UNKNOWN", {"blocked_reason": "quack_transport_unavailable", "callback_state": "STARTED_OUTCOME_UNKNOWN"}, None, None),
        ("malformed-provider-outcome", {"blocked_reason": "quack_transport_unavailable", "provider_outcome": False}, None, None),
        ("unrecognized-provider-outcome", {"blocked_reason": "quack_transport_unavailable", "provider_outcome": "ambiguous_pending"}, None, None),
        ("exhausted-attempts-counter", {"blocked_reason": "quack_transport_unavailable", "attempt_budget": {"max_attempts": 2, "attempts": 2}}, None, None),
        ("exhausted-selected-attempt-history", {"blocked_reason": "quack_transport_unavailable", "attempt_history": {"max_attempts": 2, "attempts": 2}}, None, None),
        ("settled-provider-metadata", {"blocked_reason": "quack_transport_unavailable", "provider_outcome": " SUCCEEDED "}, repair, repair),
        ("negative-attempt_number", {"blocked_reason": "quack_transport_unavailable", "attempt_budget": {"max_attempts": 2, "attempt_number": -1}}, None, None),
        ("negative-native_attempt_number", {"blocked_reason": "quack_transport_unavailable", "attempt_budget": {"max_attempts": 2, "native_attempt_number": -1}}, None, None),
        ("negative-spent_native_ordinal", {"blocked_reason": "quack_transport_unavailable", "attempt_budget": {"max_attempts": 2, "spent_native_ordinal": -1}}, None, None),
        ("negative-attempts_used", {"blocked_reason": "quack_transport_unavailable", "attempt_budget": {"max_attempts": 2, "attempts_used": -1}}, None, None),
        ("negative-attempts", {"blocked_reason": "quack_transport_unavailable", "attempt_budget": {"max_attempts": 2, "attempts": -1}}, None, None),
        ("negative-counter-without-maximum", {"blocked_reason": "quack_transport_unavailable", "attempt_history": {"attempts": -1}}, None, None),
        ("zero-unspent-counter", {"blocked_reason": "quack_transport_unavailable", "attempt_budget": {"max_attempts": 2, "attempts": 0}}, repair, repair),
        ("positive-unspent-counter", {"blocked_reason": "quack_transport_unavailable", "attempt_budget": {"max_attempts": 2, "attempts": 1}}, repair, repair),
        ("malformed-body", None, None, None),
    ]


@pytest.mark.parametrize("candidate_ready", [False, True])
@pytest.mark.parametrize("case", _false_terminal_guard_cases(), ids=lambda case: case[0])
def test_false_terminal_rearm_uses_current_reason_and_preserves_denials(case, candidate_ready):
    from copy import deepcopy
    from ipfs_accelerate_py.agent_supervisor.task_sources.quack_owner_command import (
        false_terminal_blocked_rearm_reason,
    )
    _, body, without_candidate, with_candidate = case
    before = deepcopy(body)
    expected = with_candidate if candidate_ready else without_candidate
    assert false_terminal_blocked_rearm_reason(body, candidate_ready=candidate_ready) == expected
    assert body == before


def _metadata_stage_payload() -> dict[str, object]:
    return {
        "parent_task_cid": "task:typed-owner-test",
        "expected_parent_revision": 2,
        "expected_parent_spec_cid": "spec:retained-parent",
        "expected_parent_dependencies": [],
        "work_unit_cid": "unit:selected-metadata",
        "source_generation": "source:generation-1",
        "selected_scope_cid": "scope:metadata-only",
        "requirement_cid": "requirement:missing-owner",
        "prerequisite_kind": "discover_registered_preparer",
        "handler_registration_cid": "handler:registered-metadata-discovery",
        "budget_snapshot_cid": "budget:remaining-snapshot-1",
    }


@pytest.fixture
def metadata_stage_schema(monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.task_sources import quack_owner_command

    # Schema validation must not best-effort mirror into any metadata index.
    monkeypatch.setattr(quack_owner_command, "_mirror_quack_owner_command", lambda _command: None)
    assert quack_owner_command.QUACK_OWNER_COMMAND_STAGE_METADATA_PREREQUISITE == "stage_metadata_prerequisite"
    return quack_owner_command.QUACK_OWNER_COMMAND_STAGE_METADATA_PREREQUISITE


@pytest.mark.parametrize("dependencies", [
    [], ["task:existing-a", "task:existing-b"], [f"task:{number:03}" for number in range(256)],
])
def test_metadata_stage_schema_accepts_only_typed_references(metadata_stage_schema, dependencies):
    from copy import deepcopy

    payload = _metadata_stage_payload()
    payload["expected_parent_dependencies"] = dependencies
    before = deepcopy(payload)
    result = validate_quack_owner_command(metadata_stage_schema, payload)
    assert result == before and payload == before
    assert result is not payload


@pytest.mark.parametrize("field", list(_metadata_stage_payload()))
def test_metadata_stage_schema_requires_every_binding(metadata_stage_schema, field):
    payload = _metadata_stage_payload()
    del payload[field]
    with pytest.raises(DuckDBConnectionPolicyError, match="closed schema"):
        validate_quack_owner_command(metadata_stage_schema, payload)


@pytest.mark.parametrize("extra", [
    {"command": "python3 unregistered.py"}, {"sql": "DELETE FROM tasks"},
    {"status": "ready"}, {"remaining_budget": {"max_tokens": 1000}}, {"admission_granted": True},
])
def test_metadata_stage_schema_refuses_commands_status_and_budget_overrides(metadata_stage_schema, extra):
    with pytest.raises(DuckDBConnectionPolicyError, match="closed schema"):
        validate_quack_owner_command(metadata_stage_schema, {**_metadata_stage_payload(), **extra})


@pytest.mark.parametrize("revision", [True, False, 0, -1, "1", 1.0, 2 ** 63])
def test_metadata_stage_schema_refuses_invalid_parent_revisions(metadata_stage_schema, revision):
    payload = {**_metadata_stage_payload(), "expected_parent_revision": revision}
    with pytest.raises(DuckDBConnectionPolicyError):
        validate_quack_owner_command(metadata_stage_schema, payload)


@pytest.mark.parametrize("dependencies", [
    ["task:b", "task:a"], ["task:a", "task:a"], "task:a", [True], [""],
    [f"task:{number:03}" for number in range(257)],
])
def test_metadata_stage_schema_refuses_ambiguous_or_oversized_dependencies(metadata_stage_schema, dependencies):
    payload = {**_metadata_stage_payload(), "expected_parent_dependencies": dependencies}
    with pytest.raises(DuckDBConnectionPolicyError):
        validate_quack_owner_command(metadata_stage_schema, payload)


@pytest.mark.parametrize("kind", ["", "Discover_owner", "discover-owner", "discover owner", "a" * 97, True])
def test_metadata_stage_schema_refuses_untyped_handler_kinds(metadata_stage_schema, kind):
    payload = {**_metadata_stage_payload(), "prerequisite_kind": kind}
    with pytest.raises(DuckDBConnectionPolicyError):
        validate_quack_owner_command(metadata_stage_schema, payload)


@pytest.mark.parametrize("field", [
    "parent_task_cid", "expected_parent_spec_cid", "work_unit_cid", "source_generation",
    "selected_scope_cid", "requirement_cid", "handler_registration_cid", "budget_snapshot_cid",
])
def test_metadata_stage_schema_refuses_empty_identity_references(metadata_stage_schema, field):
    payload = {**_metadata_stage_payload(), field: ""}
    with pytest.raises(DuckDBConnectionPolicyError):
        validate_quack_owner_command(metadata_stage_schema, payload)


@pytest.mark.parametrize("reference", [False, "scope:bad\x00reference", "scope:bad\nreference", "s" * 4097, chr(0xD800)])
def test_metadata_stage_schema_refuses_unsafe_reference_transport(metadata_stage_schema, reference):
    payload = {**_metadata_stage_payload(), "selected_scope_cid": reference}
    with pytest.raises(DuckDBConnectionPolicyError):
        validate_quack_owner_command(metadata_stage_schema, payload)


def _metadata_stage_bound_payload(repository):
    from ipfs_accelerate_py.agent_supervisor.task_sources.execution_packet_admission import authority_spec_cid

    parent = repository.get_task("task:typed-owner-test")
    assert parent is not None
    return {
        **_metadata_stage_payload(),
        "expected_parent_revision": int(parent["revision"]),
        "expected_parent_spec_cid": authority_spec_cid(parent),
        "expected_parent_dependencies": sorted(parent["dependencies"]),
    }


def _metadata_stage_context(*, parent, payload, store_id, store_generation, **_kwargs):
    return {
        **{field: payload[field] for field in (
            "parent_task_cid", "work_unit_cid", "source_generation", "selected_scope_cid",
            "requirement_cid", "prerequisite_kind", "handler_registration_cid", "budget_snapshot_cid",
        )},
        "parent_revision": parent["revision"],
        "parent_spec_cid": payload["expected_parent_spec_cid"],
        "parent_dependencies": list(parent["dependencies"]),
        "store_id": store_id,
        "store_generation": store_generation,
        "handler_purpose": "control_metadata_analysis",
        "remaining_budget": {
            "cpu_threads": 1, "memory_mib": 64, "storage_bytes": 1_048_576,
            "max_children": 0, "max_tokens": 0, "deadline_seconds": 10, "max_attempts": 2,
        },
        "scope_allows_prerequisite": True,
        "metadata_work_permitted": True,
        "handler_registered": True,
    }


def _metadata_stage_snapshot(owner):
    from copy import deepcopy

    cids = owner.connection.execute("SELECT task_cid FROM tasks ORDER BY task_cid").fetchall()
    return {
        "tasks": [deepcopy(dict(owner.repository.get_task(str(row[0])))) for row in cids],
        "revisions": owner.connection.execute(
            "SELECT * FROM task_revisions ORDER BY task_cid, revision"
        ).fetchall(),
        "events": owner.connection.execute("SELECT * FROM domain_events ORDER BY global_sequence").fetchall(),
        "idempotency": owner.connection.execute(
            "SELECT * FROM idempotency_records ORDER BY idempotency_key"
        ).fetchall(),
    }


@pytest.fixture
def metadata_stage_owner(tmp_path, metadata_stage_schema):
    from types import SimpleNamespace

    path = tmp_path / "metadata-stage.duckdb"
    _materialize_one_task(path)
    connection = open_duckdb_connection(path)
    repository = IntentRepository(
        path, bound_connection=connection, install_schema=False,
        owner_id="owner:metadata-stage-test", session_id="session:metadata-stage-first",
    )
    owner = None
    try:
        task = repository.get_task("task:typed-owner-test")
        assert task is not None
        repository.upsert_task(
            task_cid=task["task_cid"], task_alias=task["task_alias"],
            goal_cid=task["goal_cid"], plan_cid=task["plan_cid"], objective_id=task["objective_id"],
            ordinal=task["ordinal"], status=task["status"], priority=task["priority"],
            dependencies=list(task["dependencies"]),
            body={**task["body"], "attempt_budget": {"max_attempts": 3, "attempts": 2}},
            acceptance=[{"criterion": "retained parent acceptance", "evidence_kind": "validation"}],
            validations=[["python3", "retained_parent_validator.py"]],
            outputs=[{"path": "retained/metadata.json", "effect_id": "effect:retained-parent"}],
            expected_revision=int(task["revision"]),
        )
        task = repository.get_task("task:typed-owner-test")
        repository.block_task(
            task_cid="task:typed-owner-test", expected_revision=int(task["revision"]),
            blocker_kind="dependency", blocker_id="requirement:missing-owner",
            reason="waiting for registered metadata prerequisite",
        )
        owner = SimpleNamespace(
            path=path, connection=connection, repository=repository,
            command=metadata_stage_schema, factory=_metadata_stage_context,
        )
        yield owner
    finally:
        (owner.repository if owner is not None else repository).close()
        (owner.connection if owner is not None else connection).close()


def _execute_metadata_stage(owner, *, payload=None, **overrides):
    kwargs = {
        "request_id": "5" * 32, "store_id": "data/control.duckdb", "store_generation": "generation-1",
        "prerequisite_context_factory": owner.factory,
        "fictional_prerequisite_fixture": True,
    }
    kwargs.update(overrides)
    return execute_quack_owner_command(
        owner.repository, owner.command,
        _metadata_stage_bound_payload(owner.repository) if payload is None else payload,
        **kwargs,
    )


def test_metadata_stage_owner_creates_only_blocked_child_and_cas_dependency(metadata_stage_owner):
    from copy import deepcopy

    owner = metadata_stage_owner
    before = _metadata_stage_snapshot(owner)
    parent_before = deepcopy(dict(owner.repository.get_task("task:typed-owner-test")))
    payload = _metadata_stage_bound_payload(owner.repository)
    result = _execute_metadata_stage(owner, payload=payload)
    assert result["staged"] is True and result["duplicate"] is False
    assert result["fictional_fixture"] is True
    assert result["execution_admitted"] is result["completion_authority"] is False
    assert result["parent_task_cid"] == payload["parent_task_cid"]
    child = owner.repository.get_task(result["child_task_cid"])
    assert child is not None and child["task_cid"] != parent_before["task_cid"]
    assert child["task_alias"] == result["child_task_alias"]
    assert child["status"] == "blocked" and child["dependencies"] == ()
    assert child["body"]["execution_packet_required"] is True
    assert child["body"]["fictional_fixture"] is True
    assert child["body"]["budget_snapshot_cid"] == payload["budget_snapshot_cid"]
    assert "remaining_budget" not in child["body"]
    parent_after = owner.repository.get_task(parent_before["task_cid"])
    assert parent_after["dependencies"] == (child["task_cid"],)
    assert int(parent_after["revision"]) == int(parent_before["revision"]) + 1
    for field in ("task_cid", "task_alias", "goal_cid", "plan_cid", "objective_id", "ordinal",
                  "status", "priority", "body", "acceptance", "validations", "outputs"):
        assert parent_after[field] == parent_before[field], field
    assert child["task_cid"] not in {task["task_cid"] for task in owner.repository.select_ready_tasks()}
    after = _metadata_stage_snapshot(owner)
    assert len(after["tasks"]) == len(before["tasks"]) + 1
    assert all(row in after["revisions"] for row in before["revisions"])
    assert all(row in after["events"] for row in before["events"])
    owner.repository.assert_projection_matches_events()


def test_metadata_stage_owner_replay_survives_owner_restart_without_context_or_reset(metadata_stage_owner):
    owner = metadata_stage_owner
    payload = _metadata_stage_bound_payload(owner.repository)
    first = _execute_metadata_stage(owner, payload=payload)
    before = _metadata_stage_snapshot(owner)
    owner.repository.close()
    owner.connection.close()
    owner.connection = open_duckdb_connection(owner.path)
    owner.repository = IntentRepository(
        owner.path, bound_connection=owner.connection, install_schema=False,
        owner_id="owner:metadata-stage-test", session_id="session:metadata-stage-restarted",
    )
    replay = _execute_metadata_stage(owner, payload=payload, prerequisite_context_factory=None)
    assert replay == first
    assert _metadata_stage_snapshot(owner) == before
    assert owner.connection.execute("SELECT 1").fetchone()[0] == 1


def test_metadata_stage_owner_fresh_duplicate_keeps_child_and_original_budget_provenance(metadata_stage_owner):
    owner = metadata_stage_owner
    original_payload = _metadata_stage_bound_payload(owner.repository)
    first = _execute_metadata_stage(owner, payload=original_payload)
    before = _metadata_stage_snapshot(owner)
    payload = _metadata_stage_bound_payload(owner.repository)
    payload["budget_snapshot_cid"] = "budget:separately-current-snapshot"

    def fresh_context(**kwargs):
        context = _metadata_stage_context(**kwargs)
        context["remaining_budget"]["max_attempts"] = 1
        return context

    result = _execute_metadata_stage(owner, payload=payload, request_id="6" * 32,
                                     prerequisite_context_factory=fresh_context)
    assert result["duplicate"] is True and result["staged"] is False
    assert result["child_task_cid"] == first["child_task_cid"]
    assert result["parent_receipt"] is result["child_receipt"] is None
    after = _metadata_stage_snapshot(owner)
    for field in ("tasks", "revisions", "events"):
        assert after[field] == before[field]
    child = owner.repository.get_task(first["child_task_cid"])
    assert child["body"]["budget_snapshot_cid"] == original_payload["budget_snapshot_cid"]


@pytest.mark.parametrize("stale_field", ["expected_parent_revision", "expected_parent_spec_cid", "expected_parent_dependencies"])
def test_metadata_stage_owner_refuses_stale_parent_without_partial_rows(metadata_stage_owner, stale_field):
    owner = metadata_stage_owner
    payload = _metadata_stage_bound_payload(owner.repository)
    payload[stale_field] = {
        "expected_parent_revision": int(payload["expected_parent_revision"]) + 1,
        "expected_parent_spec_cid": "spec:stale-parent",
        "expected_parent_dependencies": ["task:unexpected-dependency"],
    }[stale_field]
    before = _metadata_stage_snapshot(owner)
    with pytest.raises(TaskSourceConflictError, match="metadata prerequisite"):
        _execute_metadata_stage(owner, payload=payload)
    assert _metadata_stage_snapshot(owner) == before


@pytest.mark.parametrize("condition", [
    "missing_factory", "missing_budget", "exhausted_budget", "handler_false", "generation_mismatch",
    "scope_denied", "metadata_denied", "missing_handler", "token_budget", "boolean_budget", "extra_budget", "overflow_budget",
])
def test_metadata_stage_owner_refuses_missing_authority_handler_or_budget_without_mutation(metadata_stage_owner, condition):
    owner = metadata_stage_owner
    before = _metadata_stage_snapshot(owner)

    def context_factory(**kwargs):
        context = _metadata_stage_context(**kwargs)
        if condition == "missing_budget":
            del context["remaining_budget"]
        elif condition == "exhausted_budget":
            context["remaining_budget"]["max_attempts"] = 0
        elif condition == "handler_false":
            context["handler_registered"] = False
        elif condition == "generation_mismatch":
            context["store_generation"] = "generation-other"
        elif condition == "scope_denied":
            context["scope_allows_prerequisite"] = False
        elif condition == "metadata_denied":
            context["metadata_work_permitted"] = False
        elif condition == "missing_handler":
            del context["handler_registration_cid"]
        elif condition == "token_budget":
            context["remaining_budget"]["max_tokens"] = 1
        elif condition == "boolean_budget":
            context["remaining_budget"]["cpu_threads"] = True
        elif condition == "extra_budget":
            context["remaining_budget"]["fresh_attempts"] = 10
        elif condition == "overflow_budget":
            context["remaining_budget"]["cpu_threads"] = 2 ** 63
        return context

    factory = None if condition == "missing_factory" else context_factory
    with pytest.raises(TaskSourceIntegrityError, match="metadata prerequisite"):
        _execute_metadata_stage(owner, prerequisite_context_factory=factory)
    assert _metadata_stage_snapshot(owner) == before


@pytest.mark.parametrize("binding", ["request_id", "store_id", "store_generation"])
def test_metadata_stage_owner_requires_explicit_owner_request_bindings(metadata_stage_owner, binding):
    owner = metadata_stage_owner
    before = _metadata_stage_snapshot(owner)
    with pytest.raises(TaskSourceIntegrityError, match="metadata prerequisite"):
        _execute_metadata_stage(owner, **{binding: ""})
    assert _metadata_stage_snapshot(owner) == before


@pytest.mark.parametrize("fail_after", [1, 2])
def test_metadata_stage_owner_event_failure_rolls_back_child_edge_and_idempotency(metadata_stage_owner, monkeypatch, fail_after):
    owner = metadata_stage_owner
    before = _metadata_stage_snapshot(owner)
    original_append = owner.repository._append_event
    calls = []

    def append(*args, **kwargs):
        calls.append(None)
        if len(calls) == fail_after:
            raise RuntimeError("injected metadata prerequisite event failure")
        return original_append(*args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(owner.repository, "_append_event", append)
        with pytest.raises(RuntimeError, match="injected metadata prerequisite event failure"):
            _execute_metadata_stage(owner)
    assert len(calls) == fail_after
    assert _metadata_stage_snapshot(owner) == before
    owner.repository.assert_projection_matches_events()


def test_metadata_stage_owner_callback_parent_mutation_is_refused_and_rolled_back(metadata_stage_owner):
    owner = metadata_stage_owner
    before = _metadata_stage_snapshot(owner)

    def mutating_context(**kwargs):
        context = _metadata_stage_context(**kwargs)
        owner.repository.cas_task_status(
            task_cid=kwargs["parent"]["task_cid"], expected_revision=kwargs["parent"]["revision"],
            new_status="ready", receipt={"operation": "fictional-context-mutation"},
        )
        return context

    with pytest.raises(TaskSourceConflictError, match="metadata prerequisite"):
        _execute_metadata_stage(owner, prerequisite_context_factory=mutating_context)
    assert _metadata_stage_snapshot(owner) == before


@pytest.mark.parametrize("case", _false_terminal_guard_cases(), ids=lambda case: case[0])
def test_false_terminal_database_recovery_cannot_borrow_history_or_denied_retry(case):
    from copy import deepcopy
    from types import SimpleNamespace
    _, body, expected, _ = case
    before = deepcopy(body)
    record = SimpleNamespace(task_alias="FIXTURE-1", task_cid="task:authored", revision=4, body=body)
    calls = []
    def rearm(subject, *, receipt):
        assert subject is record
        calls.append(receipt)
        return SimpleNamespace(changed=True, revision=5)
    source = SimpleNamespace(
        _intent=SimpleNamespace(uses_quack_transport=True),
        list_tasks=lambda **_kwargs: SimpleNamespace(tasks=(record,)),
        rearm_blocked_task=rearm,
    )
    result = DatabaseTaskSource.unstall_stale_in_progress_tasks(source, skip_stale_in_progress=True)
    assert calls == ([] if expected is None else [{"operation": expected}])
    assert len(result["unstalled"]) == (0 if expected is None else 1)
    assert body == before


def _metadata_stage_restart_owner(owner):
    owner.repository.close()
    owner.connection.close()
    owner.connection = open_duckdb_connection(owner.path)
    owner.repository = IntentRepository(
        owner.path, bound_connection=owner.connection, install_schema=False,
        owner_id="owner:metadata-stage-test", session_id="session:metadata-stage-contract-restart",
    )


@pytest.mark.parametrize("restart_owner", [False, True])
def test_metadata_stage_owner_rejects_child_reuse_after_parent_authority_changes(
    metadata_stage_owner, restart_owner,
):
    owner = metadata_stage_owner
    first_payload = _metadata_stage_bound_payload(owner.repository)
    first = _execute_metadata_stage(owner, payload=first_payload)
    parent = owner.repository.get_task("task:typed-owner-test")
    original_spec = _metadata_stage_bound_payload(owner.repository)["expected_parent_spec_cid"]
    owner.repository.upsert_task(
        task_cid=parent["task_cid"], task_alias=parent["task_alias"],
        goal_cid=parent["goal_cid"], plan_cid=parent["plan_cid"], objective_id=parent["objective_id"],
        ordinal=parent["ordinal"], status=parent["status"], priority=parent["priority"],
        dependencies=list(parent["dependencies"]), body=dict(parent["body"]),
        acceptance=[{"criterion": "new parent acceptance authority", "evidence_kind": "validation"}],
        validations=[["python3", "changed_parent_validator.py"]],
        outputs=[{"path": "retained/changed-metadata.json", "effect_id": "effect:changed-parent"}],
        expected_revision=int(parent["revision"]),
    )
    if restart_owner:
        _metadata_stage_restart_owner(owner)
    payload = _metadata_stage_bound_payload(owner.repository)
    assert payload["expected_parent_spec_cid"] != original_spec
    assert payload["expected_parent_dependencies"] == [first["child_task_cid"]]
    before = _metadata_stage_snapshot(owner)
    with pytest.raises(TaskSourceConflictError, match="metadata prerequisite"):
        _execute_metadata_stage(owner, payload=payload, request_id="7" * 32)
    assert _metadata_stage_snapshot(owner) == before
    owner.repository.assert_projection_matches_events()


@pytest.mark.parametrize("restart_owner", [False, True])
def test_metadata_stage_owner_rejects_child_reuse_after_parent_revision_only_changes(
    metadata_stage_owner, restart_owner,
):
    from copy import deepcopy

    owner = metadata_stage_owner
    first = _execute_metadata_stage(owner)
    parent = deepcopy(dict(owner.repository.get_task("task:typed-owner-test")))
    original_spec = _metadata_stage_bound_payload(owner.repository)["expected_parent_spec_cid"]
    owner.repository.upsert_task(
        task_cid=parent["task_cid"], task_alias=parent["task_alias"],
        goal_cid=parent["goal_cid"], plan_cid=parent["plan_cid"], objective_id=parent["objective_id"],
        ordinal=parent["ordinal"], status=parent["status"], priority=parent["priority"],
        dependencies=list(parent["dependencies"]), body=deepcopy(parent["body"]),
        identity=deepcopy(parent["identity"]),
        outputs=[deepcopy(item["effect"]) for item in parent["outputs"]],
        acceptance=[deepcopy(item["evidence_policy"]) for item in parent["acceptance"]],
        validations=[{**deepcopy(item["policy"]), "argv": deepcopy(item["argv"])}
                     for item in parent["validations"]],
        expected_revision=int(parent["revision"]),
    )
    if restart_owner:
        _metadata_stage_restart_owner(owner)
    current = owner.repository.get_task(parent["task_cid"])
    assert current["revision"] == parent["revision"] + 1
    for field, retained in parent.items():
        if field not in {"revision", "updated_at"}:
            assert current[field] == retained, field
    payload = _metadata_stage_bound_payload(owner.repository)
    assert payload["expected_parent_spec_cid"] == original_spec
    assert payload["expected_parent_dependencies"] == [first["child_task_cid"]]
    before = _metadata_stage_snapshot(owner)
    with pytest.raises(TaskSourceConflictError, match="metadata prerequisite"):
        _execute_metadata_stage(owner, payload=payload, request_id="9" * 32)
    assert _metadata_stage_snapshot(owner) == before
    owner.repository.assert_projection_matches_events()


@pytest.mark.parametrize("restart_replay", [False, True])
def test_metadata_stage_owner_preserves_nonempty_parent_relations_and_spent_budget(
    metadata_stage_owner, restart_replay,
):
    from copy import deepcopy

    owner = metadata_stage_owner
    parent_before = deepcopy(dict(owner.repository.get_task("task:typed-owner-test")))
    relations = {field: deepcopy(parent_before[field]) for field in ("acceptance", "validations", "outputs")}
    assert all(relations.values()), "The preservation control requires every relation family to be nonempty."
    assert parent_before["body"]["attempt_budget"] == {"max_attempts": 3, "attempts": 2}
    owner.repository.assert_projection_matches_events()
    payload = _metadata_stage_bound_payload(owner.repository)
    first = _execute_metadata_stage(owner, payload=payload)
    if restart_replay:
        _metadata_stage_restart_owner(owner)
        assert _execute_metadata_stage(owner, payload=payload, prerequisite_context_factory=None) == first
    parent_after = owner.repository.get_task(parent_before["task_cid"])
    for field, retained in relations.items():
        assert parent_after[field] == retained, field
    assert parent_after["body"] == parent_before["body"]
    assert parent_after["status"] == parent_before["status"] == "blocked"
    assert parent_after["dependencies"] == (first["child_task_cid"],)
    owner.repository.assert_projection_matches_events()


def test_metadata_stage_owner_late_parent_upsert_failure_rolls_back_relations_and_history(
    metadata_stage_owner, monkeypatch,
):
    from copy import deepcopy

    owner = metadata_stage_owner
    parent = deepcopy(dict(owner.repository.get_task("task:typed-owner-test")))
    relations = {field: deepcopy(parent[field]) for field in ("acceptance", "validations", "outputs")}
    assert all(relations.values())
    before = _metadata_stage_snapshot(owner)
    original_upsert = owner.repository.upsert_task
    completed_parent_updates = []

    def fail_after_parent_upsert(*args, **kwargs):
        receipt = original_upsert(*args, **kwargs)
        if kwargs.get("task_cid") == parent["task_cid"]:
            completed_parent_updates.append(deepcopy(dict(owner.repository.get_task(parent["task_cid"]))))
            raise RuntimeError("injected failure after complete parent relation update")
        return receipt

    with monkeypatch.context() as patcher:
        patcher.setattr(owner.repository, "upsert_task", fail_after_parent_upsert)
        with pytest.raises(RuntimeError, match="injected failure after complete parent relation update"):
            _execute_metadata_stage(owner)
    assert len(completed_parent_updates) == 1
    assert completed_parent_updates[0]["dependencies"] != parent["dependencies"]
    for field, retained in relations.items():
        assert completed_parent_updates[0][field] == retained, field
    assert _metadata_stage_snapshot(owner) == before
    owner.repository.assert_projection_matches_events()


@pytest.mark.parametrize("fixture_flag", ["omitted", False])
def test_metadata_stage_production_seam_refuses_before_context_source_or_sql(
    metadata_stage_schema, monkeypatch, fixture_flag,
):
    from ipfs_accelerate_py.agent_supervisor.task_sources import database_task_source

    reached = []

    class NeverExecute:
        def execute(self, *_args, **_kwargs):
            reached.append("sql")
            raise AssertionError("production refusal reached SQL")

        def transaction(self, *_args, **_kwargs):
            reached.append("transaction")
            raise AssertionError("production refusal reached a transaction")

    def forbidden_source(*_args, **_kwargs):
        reached.append("source-wrapper")
        raise AssertionError("production refusal constructed a source wrapper")

    def supplied_context(**_kwargs):
        reached.append("context-factory")
        raise AssertionError("a supplied callback cannot replace the missing production resolver")

    repository = IntentRepository.__new__(IntentRepository)
    repository._bound_connection = NeverExecute()
    before = dict(repository.__dict__)
    monkeypatch.setattr(database_task_source, "DatabaseTaskSource", forbidden_source)
    extra = {} if fixture_flag == "omitted" else {"fictional_prerequisite_fixture": fixture_flag}
    with pytest.raises(TaskSourceIntegrityError, match="production metadata prerequisite"):
        execute_quack_owner_command(
            repository, metadata_stage_schema, _metadata_stage_payload(),
            request_id="8" * 32, store_id="data/control.duckdb", store_generation="generation-1",
            prerequisite_context_factory=supplied_context, **extra,
        )
    assert reached == []
    assert repository.__dict__ == before
