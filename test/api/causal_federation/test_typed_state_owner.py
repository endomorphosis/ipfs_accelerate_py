from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources import (
    typed_state_owner as typed_owner,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    CommandKind,
    StateCommand,
    StoreGeneration,
    canonical_json_bytes,
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_transactions import (
    OptimisticConflictError,
    result_digest,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    install_control_plane_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    COMPLETION_EVIDENCE_SCHEMA,
    INTENT_COMPLETION_PROJECTION_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
    QuackClientTransportError,
    QuackStateClient,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    COMPLETION_PROGRESS_SNAPSHOT_OPERATION,
    STATUS_BOOTSTRAP_ALLOWED_OPERATIONS,
    STATUS_BOOTSTRAP_CLIENT_ID,
    TYPED_COMPLETION_PROGRESS_SNAPSHOT_SCHEMA,
    TYPED_STATE_OWNER_SOCKET_ENV,
    TYPED_STATE_OWNER_TOKEN_ENV,
    TypedStateOwnerAuthorizationError,
    TypedStateOwnerConnection,
    TypedStateOwnerError,
    TypedStateOwnerGateway,
    TypedStateOwnerProtocolError,
    TypedStateOwnerRemoteError,
    build_control_plane_operation_catalog,
)


def _install(db: Path) -> None:
    install_control_plane_schema(
        db,
        application_version="0.0.45",
        tool_version="1.5.2",
        owner_id="typed-owner-test",
    )
    with open_duckdb_connection(db) as connection:
        connection.execute(
            """
            INSERT INTO goals (
                goal_cid, goal_alias, objective_id, parent_goal_cid, ordinal,
                title, status, created_at, updated_at, revision, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "goal:typed-owner",
                "G-TYPED",
                "objective:typed-owner",
                "",
                1,
                "Typed owner",
                "open",
                "1970-01-01T00:00:00Z",
                "1970-01-01T00:00:00Z",
                0,
                "{}",
            ],
        )
        connection.execute(
            """
            INSERT INTO tasks (
                task_cid, task_alias, goal_cid, plan_cid, objective_id,
                ordinal, status, revision, priority, created_at, updated_at,
                identity_json, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "task:typed-owner",
                "CASF-TYPED",
                "goal:typed-owner",
                "",
                "objective:typed-owner",
                1,
                "ready",
                0,
                "P0",
                "1970-01-01T00:00:00Z",
                "1970-01-01T00:00:00Z",
                "{}",
                "{}",
            ],
        )
    seed = QuackStateClient(
        owner_id="typed-owner-test:seed",
        store_id="control.duckdb",
    )
    try:
        seed.attach(db, seed_generation=True)
    finally:
        seed.close()


def _gateway(db: Path, socket_path: Path) -> tuple[TypedStateOwnerGateway, Any]:
    connection = open_duckdb_connection(db)
    row = connection.execute(
        """
        SELECT generation, schema_revision, fence_epoch, revision,
               database_uuid, birth_id
        FROM store_generations ORDER BY generation DESC LIMIT 1
        """
    ).fetchone()
    assert row is not None
    identity = {
        "server_id": "server:typed-owner-test",
        "store_id": "control.duckdb",
        "generation": int(row[0]),
        "schema_revision": int(row[1]),
        "fence_epoch": int(row[2]),
        "revision": int(row[3]),
        "database_uuid": str(row[4]),
        "process_birth_id": str(row[5] or "birth:typed-owner-test"),
    }
    gateway = TypedStateOwnerGateway(
        connection=connection,
        socket_path=socket_path,
        store_id="control.duckdb",
        identity=identity,
    )
    gateway.start()
    return gateway, connection


def _read_operations() -> tuple[str, ...]:
    catalog = build_control_plane_operation_catalog()
    return tuple(name for name, operation in catalog.items() if not operation.mutation)


def _task_grant_operations() -> tuple[str, ...]:
    return (
        *_read_operations(),
        "txn_advance_store_revision",
        "txn_record_idempotency",
        "txn_cas_task_status",
        "executor_insert_task_revision_history",
    )


def _attached_client(client_id: str) -> QuackStateClient:
    client = QuackStateClient(
        owner_id=client_id,
        store_id="control.duckdb",
    )
    client.attach(
        "quack:127.0.0.1:7777",
        server_id="server:typed-owner-test",
    )
    return client


def _install_protected_qualification_completion_fixture(
    db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Seed the exact legacy gap while substituting only its unavailable body."""

    task_cid = "baguqeerakwvsckoysv5edcru3makxvmcwjjm2alzam5umsyqbkp3efngyqpa"
    dependency_specs = (
        (
            "LGCVF-112",
            "baguqeeramxjolmqp2rh7r5vfrrs5ne3mqqhbxh3pn74k27h5tbpf4luasfkq",
            5,
        ),
        (
            "LGCVF-111",
            "baguqeerau4vdgcyn3sdorik7zawwgwrhy7mxxydb6gashctytbmdzwtmumea",
            4,
        ),
    )
    prior_receipt = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "typed-database-legacy-unstall-recovery@1"
        ),
        "operation": "database_claim_lost_sidecar_recovery",
    }
    prior_body = {
        "completion_receipt": prior_receipt,
        "fixture": "protected-qualification-legacy-gap",
    }
    prior_body_json = canonical_json_bytes(prior_body).decode("utf-8")
    prior_body_sha256 = "sha256:" + hashlib.sha256(
        prior_body_json.encode("utf-8")
    ).hexdigest()
    monkeypatch.setattr(
        typed_owner,
        "_LGCVF_PROTECTED_QUALIFICATION_COMPLETION_PRIOR_BODY_SHA256",
        prior_body_sha256,
    )

    db.parent.mkdir(mode=0o700, parents=True)
    _install(db)
    dependency_pins: list[dict[str, Any]] = []
    with open_duckdb_connection(db) as connection:
        connection.execute(
            """
            INSERT INTO tasks (
                task_cid, task_alias, goal_cid, plan_cid, objective_id,
                ordinal, status, revision, priority, created_at, updated_at,
                identity_json, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                task_cid,
                "LGCVF-113",
                "goal:typed-owner",
                "plan:protected-qualification",
                "objective:typed-owner",
                113,
                "retrying",
                8,
                "P0",
                "1970-01-01T00:00:00Z",
                "1970-01-08T00:00:00Z",
                "{}",
                prior_body_json,
            ],
        )
        connection.execute(
            "INSERT INTO task_revisions "
            "(task_cid, revision, status, body_json, recorded_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                task_cid,
                1,
                "ready",
                "{}",
                "1970-01-01T00:00:00Z",
            ],
        )
        for alias, dependency_cid, revision in dependency_specs:
            dependency_body_json = canonical_json_bytes(
                {"fixture": alias}
            ).decode("utf-8")
            connection.execute(
                """
                INSERT INTO tasks (
                    task_cid, task_alias, goal_cid, plan_cid, objective_id,
                    ordinal, status, revision, priority, created_at, updated_at,
                    identity_json, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    dependency_cid,
                    alias,
                    "goal:typed-owner",
                    "plan:protected-qualification",
                    "objective:typed-owner",
                    int(alias.rsplit("-", 1)[1]),
                    "completed",
                    revision,
                    "P0",
                    "1970-01-01T00:00:00Z",
                    f"1970-01-0{revision}T00:00:00Z",
                    "{}",
                    dependency_body_json,
                ],
            )
            connection.execute(
                "INSERT INTO task_dependencies "
                "(task_cid, dependency_task_cid, kind) VALUES (?, ?, ?)",
                [task_cid, dependency_cid, "hard"],
            )
            dependency_pins.append(
                {
                    "task_cid": dependency_cid,
                    "task_alias": alias,
                    "status": "completed",
                    "revision": revision,
                    "body_sha256": "sha256:"
                    + hashlib.sha256(
                        dependency_body_json.encode("utf-8")
                    ).hexdigest(),
                }
            )

    def fixture_cid(name: str) -> str:
        return content_identity({"protected_qualification_fixture": name})

    def fixture_sha256(digit: str) -> str:
        return "sha256:" + digit * 64
    qualification_result_cid = fixture_cid("qualification-result")
    qualification_stable_projection_cid = fixture_cid(
        "qualification-stable-projection"
    )
    qualification_artifact_sha256 = fixture_sha256("7")
    source_continuity = {
        "approved_branch": (
            "agent/logic-governed-compositional-verification-fabric-v1"
        ),
        "resolved_remote_head": "1" * 40,
        "current_head": "2" * 40,
        "current_tree": "3" * 40,
        "candidate_worktree_clean": True,
        "datasets_head": "4" * 40,
        "datasets_tree": "5" * 40,
        "datasets_worktree_clean": True,
        "python_bytecode_quarantine": {
            "enabled": True,
            "ephemeral": True,
            "ignored_worktree_pycache": "quarantined_not_imported",
            "outside_candidate_root": True,
            "private": True,
        },
        "superproject_runtime_inventory": {
            "tracked_object_count": 1,
            "tracked_inventory_root": fixture_sha256("1"),
        },
        "datasets_runtime_inventory": {
            "tracked_object_count": 1,
            "tracked_inventory_root": fixture_sha256("2"),
        },
    }
    qualification = {
        "argv": [
            "python",
            "scripts/qualify_logic_governed_compositional_verification_fabric.py",
            "--check",
        ],
        "validator_path": (
            "scripts/qualify_logic_governed_compositional_verification_fabric.py"
        ),
        "validator_blob_oid": "6" * 40,
        "validator_sha256": fixture_sha256("3"),
        "artifact_path": (
            "data/agent_supervisor/logic_governed_compositional_verification_"
            "fabric/independent_qualification_result.json"
        ),
        "artifact_blob_oid": "7" * 40,
        "artifact_sha256": qualification_artifact_sha256,
        "stored_result_cid": qualification_result_cid,
        "stored_stable_projection_cid": (
            qualification_stable_projection_cid
        ),
        "replay_stable_projection_cid": qualification_stable_projection_cid,
        "replay_exit_code": 0,
        "recorded_at": "1970-01-08T00:00:00Z",
        "passed": True,
        "test_qualification_complete": True,
        "independent_fixed_manifest_executed": True,
        "candidate_suites_are_self_authority": False,
        "task_implementation_complete": False,
        "objective_complete": False,
        "release_qualified": False,
        "production_authoritative": False,
        "production_authorized": False,
    }
    reviewed_pins = {
        "target_generation": "lgcvf-run-v39",
        "source_provenance_cid": fixture_cid("source-provenance"),
        "stopped_state_continuity_receipt_cid": fixture_cid(
            "stopped-continuity"
        ),
        "stopped_controller_status_cid": fixture_cid("stopped-status"),
        "source_continuity": source_continuity,
        "databases": {
            "control": {
                "path": str(db),
                "sha256": fixture_sha256("4"),
            },
            "coordination": {
                "path": str(db.with_name("control.coordination.duckdb")),
                "sha256": fixture_sha256("5"),
            },
            "execution": {
                "path": str(db.with_name("control.execution.duckdb")),
                "sha256": fixture_sha256("6"),
            },
        },
        "task": {
            "task_cid": task_cid,
            "task_alias": "LGCVF-113",
            "status": "retrying",
            "revision": 8,
            "body_sha256": prior_body_sha256,
            "prior_receipt_schema": prior_receipt["schema"],
            "prior_receipt_operation": prior_receipt["operation"],
            "prior_receipt_cid": content_identity(prior_receipt),
            "history_observed_revisions": [1],
            "history_missing_revisions": list(range(2, 9)),
            "gap_preserved": True,
            "repair_authority": False,
        },
        "dependencies": dependency_pins,
        "dependency_binding_cid": content_identity(dependency_pins),
        "qualification": qualification,
    }
    reviewed_preflight = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "lgcvf-protected-qualification-completion-preflight@1"
        ),
        "operation": (
            typed_owner.TYPED_DATABASE_PROTECTED_QUALIFICATION_COMPLETION_OPERATION
        ),
        "reviewed_pins": reviewed_pins,
    }
    return {
        "task_cid": task_cid,
        "prior_receipt": prior_receipt,
        "prior_body": prior_body,
        "prior_body_sha256": prior_body_sha256,
        "dependency_cids": [item["task_cid"] for item in dependency_pins],
        "reviewed_preflight": reviewed_preflight,
        "reviewed_preflight_cid": content_identity(reviewed_preflight),
        "qualification_result_cid": qualification_result_cid,
        "qualification_stable_projection_cid": (
            qualification_stable_projection_cid
        ),
        "qualification_artifact_sha256": qualification_artifact_sha256,
    }


def test_default_owner_socket_path_compacts_an_overlong_store_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(TYPED_STATE_OWNER_SOCKET_ENV, raising=False)
    long_store = tmp_path.joinpath(*("long-segment" for _ in range(12)), "control.duckdb")

    first = typed_owner.typed_owner_socket_path(str(long_store))
    second = typed_owner.typed_owner_socket_path(str(long_store))

    assert first == second
    assert len(os.fsencode(first)) <= 100
    assert first.parent.name == f"ipfs-accelerate-typed-owner-{os.geteuid()}"
    assert first.suffix == ".sock"


def test_explicit_owner_socket_path_is_not_silently_rewritten(
    tmp_path: Path,
) -> None:
    explicit = tmp_path / "launcher-owned" / "owner.sock"
    assert typed_owner.typed_owner_socket_path("ignored", str(explicit)) == explicit


def test_protected_qualification_completion_is_atomic_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "run-v39" / "control.duckdb"
    socket_path = tmp_path / "protected-qualification-owner.sock"
    fixture = _install_protected_qualification_completion_fixture(
        db,
        monkeypatch,
    )
    gateway, owner_connection = _gateway(db, socket_path)
    client_id = (
        typed_owner.TYPED_DATABASE_PROTECTED_QUALIFICATION_COMPLETION_CLIENT_ID
    )
    process_birth_id = "birth:protected-qualification-completion-test"
    allowed_operations = (
        "whoami_metadata",
        "load_store_generation",
        "txn_load_generation",
        "txn_lookup_idempotency",
        "txn_advance_store_revision",
        "txn_record_idempotency",
        "executor_insert_validation_run",
        "executor_insert_validation_result",
        "executor_insert_validation_evidence",
        "executor_cas_task_status_receipt",
        "executor_insert_task_revision_history",
    )
    token, _grant = gateway.issue_grant(
        client_id=client_id,
        process_birth_id=process_birth_id,
        allowed_operations=allowed_operations,
        allowed_command_operations=(
            typed_owner.TYPED_DATABASE_PROTECTED_QUALIFICATION_COMPLETION_COMMAND,
        ),
        entity_scopes={
            "task_cid": fixture["task_cid"],
            "reviewed_preflight_cid": fixture["reviewed_preflight_cid"],
        },
        peer_pid=os.getpid(),
    )
    monkeypatch.setenv(TYPED_STATE_OWNER_SOCKET_ENV, str(socket_path))
    monkeypatch.setenv(TYPED_STATE_OWNER_TOKEN_ENV, token)
    client = QuackStateClient(
        owner_id=client_id,
        store_id="control.duckdb",
        process_birth_id=process_birth_id,
    )

    def complete() -> Any:
        return client.complete_protected_qualification_legacy_gap(
            task_cid=fixture["task_cid"],
            task_alias="LGCVF-113",
            expected_task_revision=8,
            expected_prior_body_sha256=fixture["prior_body_sha256"],
            expected_prior_receipt_schema=fixture["prior_receipt"]["schema"],
            expected_prior_receipt_operation=fixture["prior_receipt"][
                "operation"
            ],
            expected_prior_receipt_cid=content_identity(
                fixture["prior_receipt"]
            ),
            expected_history_revisions=[1],
            expected_dependency_cids=fixture["dependency_cids"],
            prior_body=fixture["prior_body"],
            reviewed_preflight_cid=fixture["reviewed_preflight_cid"],
            reviewed_preflight=fixture["reviewed_preflight"],
            qualification_result_cid=fixture["qualification_result_cid"],
            qualification_stable_projection_cid=fixture[
                "qualification_stable_projection_cid"
            ],
            qualification_artifact_sha256=fixture[
                "qualification_artifact_sha256"
            ],
        )

    before_revision_row = owner_connection.execute(
        "SELECT revision FROM store_generations ORDER BY generation DESC LIMIT 1"
    ).fetchone()
    assert before_revision_row is not None
    before_revision = int(before_revision_row[0])
    try:
        client.attach(
            "quack:127.0.0.1:7777",
            server_id="server:typed-owner-test",
        )
        accepted = complete()
        replay = complete()
    finally:
        client.close()
        gateway.stop()

    assert (accepted.outcome.value, accepted.changed) == ("accepted", True)
    assert (replay.outcome.value, replay.changed) == (
        "idempotent_replay",
        False,
    )
    assert dict(replay.result) == dict(accepted.result)
    assert replay.result_digest == accepted.result_digest
    assert accepted.result["task_revision"] == 9
    assert accepted.result["completion_receipt_cid"]

    task_row = owner_connection.execute(
        "SELECT status, revision, body_json FROM tasks WHERE task_cid = ?",
        [fixture["task_cid"]],
    ).fetchone()
    assert task_row is not None
    assert (str(task_row[0]), int(task_row[1])) == ("completed", 9)
    completed_body = json.loads(str(task_row[2]))
    completion_receipt = completed_body["completion_receipt"]
    assert completion_receipt["receipt_cid"] == accepted.result[
        "completion_receipt_cid"
    ]
    assert completion_receipt["reviewed_preflight_cid"] == fixture[
        "reviewed_preflight_cid"
    ]
    assert completion_receipt["history_observed_revisions"] == [1]
    assert completion_receipt["history_missing_revisions"] == list(range(2, 9))
    assert completion_receipt["appended_revision"] == 9
    assert completion_receipt["gap_preserved"] is True
    assert completion_receipt["repair_performed"] is False
    assert completion_receipt["fabricated_revisions"] == []

    history_rows = owner_connection.execute(
        "SELECT revision FROM task_revisions WHERE task_cid = ? ORDER BY revision",
        [fixture["task_cid"]],
    ).fetchall()
    assert [int(row[0]) for row in history_rows] == [1, 9]
    for table, key, value in (
        ("validation_runs", "run_id", accepted.result["run_id"]),
        ("validation_results", "result_id", accepted.result["result_id"]),
        ("evidence_nodes", "evidence_id", accepted.result["evidence_id"]),
    ):
        count_row = owner_connection.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {key} = ?",  # noqa: S608
            [value],
        ).fetchone()
        assert count_row is not None and int(count_row[0]) == 1
    after_revision_row = owner_connection.execute(
        "SELECT revision FROM store_generations ORDER BY generation DESC LIMIT 1"
    ).fetchone()
    assert after_revision_row is not None
    assert int(after_revision_row[0]) == before_revision + 1
    owner_connection.close()


def test_closed_owner_command_is_atomic_and_rolls_back_callback_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    gateway, connection = _gateway(db, socket_path)
    token, _grant = gateway.issue_grant(
        client_id="client:typed-owner-test",
        allowed_operations=_task_grant_operations(),
        allowed_command_operations=("task.status.cas",),
        peer_pid=os.getpid(),
    )
    monkeypatch.setenv(TYPED_STATE_OWNER_SOCKET_ENV, str(socket_path))
    monkeypatch.setenv(TYPED_STATE_OWNER_TOKEN_ENV, token)
    client = QuackStateClient(
        owner_id="client:typed-owner-test",
        store_id="control.duckdb",
    )
    try:
        client.attach("quack:127.0.0.1:7777", server_id="server:typed-owner-test")
        accepted = client.cas_task_status(
            task_cid="task:typed-owner",
            expected_task_revision=0,
            new_status="claimed",
            idempotency_key="idem:typed-owner:accepted",
            command_id="command:typed-owner:accepted",
        )
        assert accepted.changed is True
        before = client.load_generation()
        session = client.session
        assert session is not None
        command = StateCommand(
            command_id="command:typed-owner:rollback",
            command_kind=CommandKind.CLAIM,
            store_id="control.duckdb",
            session_id=session.session_id,
            expected_generation=before.generation,
            expected_revision=before.revision,
            fence_epoch=before.fence_epoch,
            idempotency_key="idem:typed-owner:rollback",
            parameters={
                "operation": "task.status.cas",
                "task_cid": "task:typed-owner",
                "expected_task_revision": 1,
                "status": "failed",
            },
        )

        def fail_after_state_mutation(transaction: Any, *_args: Any) -> dict[str, Any]:
            transaction.cas_row_revision(
                table="tasks",
                key_column="task_cid",
                key_value="task:typed-owner",
                expected_revision=1,
                assignments={
                    "status": "failed",
                    "updated_at": "1970-01-01T00:00:01Z",
                },
            )
            raise RuntimeError("do not commit")

        with pytest.raises(RuntimeError, match="do not commit"):
            client.submit_command(command, apply=fail_after_state_mutation)
        task = client.execute("select_task_by_cid", {"task_cid": "task:typed-owner"})[0]
        after = client.load_generation()
        assert task["status"] == "claimed"
        assert int(task["revision"]) == 1
        assert after.revision == before.revision
        assert client.execute("lookup_idempotency", ["idem:typed-owner:rollback"]) == ()
    finally:
        client.close()
        gateway.stop()
        connection.close()


def test_error_code_preserves_duckdb_fatal_detail() -> None:
    class FatalException(Exception):
        pass

    code = TypedStateOwnerGateway._error_code(
        FatalException("INTERNAL Error: Failed to checkpoint")
    )
    assert code.startswith("operation_failed:FatalException:")
    assert "checkpoint" in code


def test_owner_rebuilds_task_status_indexes_after_art_fatal(
    tmp_path: Path,
) -> None:
    class FatalException(Exception):
        pass

    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    gateway, connection = _gateway(db, socket_path)
    try:
        names_before = {
            str(row[0])
            for row in connection.execute(
                "SELECT index_name FROM duckdb_indexes() WHERE table_name = 'tasks'"
            ).fetchall()
        }
        assert "tasks_status_idx" in names_before
        assert "tasks_goal_idx" in names_before
        gateway._recover_exclusive_handle_after_native_fatal(
            FatalException(
                "FATAL Error: Invalid Input Error: Failed to delete all rows "
                "from index. Only deleted 0 out of 1 rows."
            )
        )
        names = {
            str(row[0])
            for row in connection.execute(
                "SELECT index_name FROM duckdb_indexes() WHERE table_name = 'tasks'"
            ).fetchall()
        }
        assert "tasks_status_idx" in names
        assert "tasks_goal_idx" in names
        connection.execute(
            "UPDATE tasks SET status = ?, revision = ? "
            "WHERE task_cid = ? AND revision = ?",
            ["in_progress", 1, "task:typed-owner", 0],
        )
        row = connection.execute(
            "SELECT status, revision FROM tasks WHERE task_cid = 'task:typed-owner'"
        ).fetchone()
        if hasattr(row, "get"):
            observed = (row["status"], int(row["revision"]))
        else:
            observed = (row[0], int(row[1]))
        assert observed == ("in_progress", 1)
    finally:
        gateway.stop()
        connection.close()


def test_owner_drops_status_indexes_around_receipt_cas(
    tmp_path: Path,
) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    gateway, connection = _gateway(db, socket_path)
    try:
        command = StateCommand(
            command_id="cmd:status-cas",
            command_kind=CommandKind.CLAIM,
            store_id="control.duckdb",
            session_id="session:status-cas",
            expected_generation=1,
            expected_revision=1,
            fence_epoch=1,
            idempotency_key="idem:status-cas",
            parameters={"operation": "task.status.cas.receipt"},
        )
        dropped = gateway._task_status_cas_index_sql(command)
        assert any("tasks_status_idx" in sql for sql in dropped)
        names = {
            str(row[0])
            for row in connection.execute(
                "SELECT index_name FROM duckdb_indexes() WHERE table_name = 'tasks'"
            ).fetchall()
        }
        assert "tasks_status_idx" not in names
        gateway._restore_task_status_cas_indexes(dropped)
        names = {
            str(row[0])
            for row in connection.execute(
                "SELECT index_name FROM duckdb_indexes() WHERE table_name = 'tasks'"
            ).fetchall()
        }
        assert "tasks_status_idx" in names
        assert "tasks_goal_idx" in names
    finally:
        gateway.stop()
        connection.close()


def test_gateway_attach_recovers_after_exclusive_owner_poison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    gateway, connection = _gateway(db, socket_path)
    token, _grant = gateway.issue_grant(
        client_id="client:recover-after-poison",
        allowed_operations=_read_operations(),
        peer_pid=os.getpid(),
    )
    monkeypatch.setenv(TYPED_STATE_OWNER_SOCKET_ENV, str(socket_path))
    monkeypatch.setenv(TYPED_STATE_OWNER_TOKEN_ENV, token)
    with connection._execution_condition:
        connection._poison_locked()
    client = QuackStateClient(
        owner_id="client:recover-after-poison",
        store_id="control.duckdb",
    )
    try:
        client.attach("quack:127.0.0.1:7777", server_id="server:typed-owner-test")
        generation = client.load_generation()
        assert int(generation.generation) >= 1
    finally:
        client.close()
        gateway.stop()
        connection.close()


def test_commit_observer_runs_after_owner_transaction_lock_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    gateway, connection = _gateway(db, socket_path)
    client_id = "client:post-commit-lock"
    token, _grant = gateway.issue_grant(
        client_id=client_id,
        allowed_operations=_task_grant_operations(),
        allowed_command_operations=("task.status.cas",),
        peer_pid=os.getpid(),
    )
    monkeypatch.setenv(TYPED_STATE_OWNER_SOCKET_ENV, str(socket_path))
    monkeypatch.setenv(TYPED_STATE_OWNER_TOKEN_ENV, token)
    observed: list[tuple[bool, str, tuple[str, ...]]] = []

    def observe_commit(command: StateCommand, manifest: tuple[Any, ...]) -> None:
        lock_available = gateway._transaction_lock.acquire(blocking=False)
        if lock_available:
            gateway._transaction_lock.release()
        observed.append(
            (
                lock_available,
                str(command.parameters.get("operation") or ""),
                tuple(str(item[0]) for item in manifest),
            )
        )

    gateway.bind_commit_observer(observe_commit)
    client = _attached_client(client_id)
    try:
        accepted = client.cas_task_status(
            task_cid="task:typed-owner",
            expected_task_revision=0,
            new_status="claimed",
            idempotency_key="idem:post-commit-lock",
            command_id="command:post-commit-lock",
        )
        assert accepted.changed is True
        assert observed == [
            (
                True,
                "task.status.cas",
                (
                    "txn_cas_task_status",
                    "executor_insert_task_revision_history",
                    "txn_advance_store_revision",
                    "txn_record_idempotency",
                ),
            )
        ]
    finally:
        client.close()
        gateway.stop()
        connection.close()


def test_blocked_downstream_observer_does_not_block_later_owner_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    gateway, connection = _gateway(db, socket_path)
    client_id = "client:blocked-downstream"
    token, _grant = gateway.issue_grant(
        client_id=client_id,
        allowed_operations=_task_grant_operations(),
        allowed_command_operations=("task.status.cas",),
        peer_pid=os.getpid(),
    )
    monkeypatch.setenv(TYPED_STATE_OWNER_SOCKET_ENV, str(socket_path))
    monkeypatch.setenv(TYPED_STATE_OWNER_TOKEN_ENV, token)
    first_observer_entered = threading.Event()
    release_first_observer = threading.Event()
    observer_guard = threading.Lock()
    observer_lock_available: list[bool] = []
    observer_calls = 0

    def blocked_downstream_observer(
        _command: StateCommand,
        _manifest: tuple[Any, ...],
    ) -> None:
        nonlocal observer_calls
        lock_available = gateway._transaction_lock.acquire(blocking=False)
        if lock_available:
            gateway._transaction_lock.release()
        with observer_guard:
            observer_calls += 1
            call_number = observer_calls
            observer_lock_available.append(lock_available)
        if call_number == 1:
            first_observer_entered.set()
            release_first_observer.wait(timeout=10.0)

    gateway.bind_commit_observer(blocked_downstream_observer)
    first_client = _attached_client(client_id)
    second_client = _attached_client(client_id)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(
                first_client.cas_task_status,
                task_cid="task:typed-owner",
                expected_task_revision=0,
                new_status="claimed",
                idempotency_key="idem:blocked-downstream:first",
                command_id="command:blocked-downstream:first",
            )
            assert first_observer_entered.wait(timeout=5.0)
            try:
                second_call = executor.submit(
                    second_client.cas_task_status,
                    task_cid="task:typed-owner",
                    expected_task_revision=1,
                    new_status="running",
                    idempotency_key="idem:blocked-downstream:second",
                    command_id="command:blocked-downstream:second",
                )
                second = second_call.result(timeout=5.0)
                assert second.changed is True
                assert first.done() is False
            finally:
                release_first_observer.set()
            assert first.result(timeout=5.0).changed is True
        assert observer_calls == 2
        assert observer_lock_available == [True, True]
        task = second_client.execute(
            "select_task_by_cid",
            {"task_cid": "task:typed-owner"},
        )[0]
        assert task["status"] == "running"
        assert int(task["revision"]) == 2
    finally:
        release_first_observer.set()
        first_client.close()
        second_client.close()
        gateway.stop()
        connection.close()


def test_failing_downstream_observer_cannot_undo_or_disable_owner_operations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    gateway, connection = _gateway(db, socket_path)
    client_id = "client:failing-downstream"
    token, _grant = gateway.issue_grant(
        client_id=client_id,
        allowed_operations=_task_grant_operations(),
        allowed_command_operations=("task.status.cas",),
        peer_pid=os.getpid(),
    )
    monkeypatch.setenv(TYPED_STATE_OWNER_SOCKET_ENV, str(socket_path))
    monkeypatch.setenv(TYPED_STATE_OWNER_TOKEN_ENV, token)
    observer_lock_available: list[bool] = []

    def failing_downstream_observer(
        _command: StateCommand,
        _manifest: tuple[Any, ...],
    ) -> None:
        lock_available = gateway._transaction_lock.acquire(blocking=False)
        if lock_available:
            gateway._transaction_lock.release()
        observer_lock_available.append(lock_available)
        raise RuntimeError("downstream DuckLake-style observer unavailable")

    gateway.bind_commit_observer(failing_downstream_observer)
    client = _attached_client(client_id)
    try:
        first = client.cas_task_status(
            task_cid="task:typed-owner",
            expected_task_revision=0,
            new_status="claimed",
            idempotency_key="idem:failing-downstream:first",
            command_id="command:failing-downstream:first",
        )
        second = client.cas_task_status(
            task_cid="task:typed-owner",
            expected_task_revision=1,
            new_status="running",
            idempotency_key="idem:failing-downstream:second",
            command_id="command:failing-downstream:second",
        )
        assert first.changed is True
        assert second.changed is True
        assert observer_lock_available == [True, True]
        assert gateway.capability()["last_observer_error_type"] == "RuntimeError"
        task = client.execute(
            "select_task_by_cid",
            {"task_cid": "task:typed-owner"},
        )[0]
        assert task["status"] == "running"
        assert int(task["revision"]) == 2
    finally:
        client.close()
        gateway.stop()
        connection.close()


def test_grant_rejects_self_admission_unknown_mutation_and_scope_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    gateway, connection = _gateway(db, socket_path)
    token, _grant = gateway.issue_grant(
        client_id="client:bounded",
        allowed_operations=("whoami_metadata", "load_store_generation"),
        peer_pid=os.getpid(),
    )
    scoped_token, _scoped_grant = gateway.issue_grant(
        client_id="client:scoped",
        allowed_operations=("casf_select_federation",),
        peer_pid=os.getpid(),
        tenant_id="tenant:one",
        federation_id="federation:one",
    )
    monkeypatch.setenv(TYPED_STATE_OWNER_SOCKET_ENV, str(socket_path))
    monkeypatch.setenv(TYPED_STATE_OWNER_TOKEN_ENV, token)
    try:
        wrong = QuackStateClient(owner_id="client:self-promoted")
        with pytest.raises(QuackClientTransportError):
            wrong.attach("quack:127.0.0.1:7777")
        raw = TypedStateOwnerConnection(
            socket_path=socket_path,
            token=token,
            client_id="client:bounded",
            process_birth_id="caller-asserted-and-nonauthoritative",
            store_id="control.duckdb",
        )
        try:
            assert raw.execute_operation("whoami_metadata").fetchone() is not None
            with pytest.raises(TypedStateOwnerRemoteError) as denied:
                raw.execute_operation(
                    "seed_client_session",
                    [
                        "session:forged",
                        "server:forged",
                        "client:admin",
                        "birth:forged",
                        "1970-01-01T00:00:00Z",
                        "1970-01-01T00:00:00Z",
                        1,
                        1,
                        "attached",
                        0,
                    ],
                )
            assert denied.value.error_code == "authorization_denied"
        finally:
            raw.close()
        scoped = TypedStateOwnerConnection(
            socket_path=socket_path,
            token=scoped_token,
            client_id="client:scoped",
            process_birth_id="birth:scoped",
            store_id="control.duckdb",
        )
        try:
            assert scoped.execute_operation(
                "casf_select_federation",
                ["federation:one", "tenant:one"],
            ).fetchone() is None
            with pytest.raises(TypedStateOwnerRemoteError) as wrong_scope:
                scoped.execute_operation(
                    "casf_select_federation",
                    ["federation:two", "tenant:two"],
                )
            assert wrong_scope.value.error_code == "authorization_denied"
        finally:
            scoped.close()
    finally:
        gateway.stop()
        connection.close()


def test_live_grant_revocation_is_checked_on_every_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    gateway, connection = _gateway(db, socket_path)
    token, grant = gateway.issue_grant(
        client_id="client:revocable",
        allowed_operations=("whoami_metadata", "load_store_generation"),
        peer_pid=os.getpid(),
    )
    monkeypatch.setenv(TYPED_STATE_OWNER_SOCKET_ENV, str(socket_path))
    monkeypatch.setenv(TYPED_STATE_OWNER_TOKEN_ENV, token)
    client = QuackStateClient(owner_id="client:revocable")
    try:
        client.attach("quack:127.0.0.1:7777")
        assert client.execute("whoami_metadata")
        gateway.revoke_grant(grant.grant_id)
        with pytest.raises(QuackClientTransportError) as denied:
            client.execute("whoami_metadata")
        assert isinstance(denied.value.__cause__, TypedStateOwnerRemoteError)
        assert denied.value.__cause__.error_code == "authorization_denied"
    finally:
        client.close()
        gateway.stop()
        connection.close()


def test_transaction_idempotency_lookup_cannot_cross_admitted_command_key(
    tmp_path: Path,
) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    with open_duckdb_connection(db) as seed:
        seed.execute(
            """
            INSERT INTO idempotency_records (
                idempotency_key, command_kind, command_id, store_id,
                session_id, result_digest, created_at, expires_at, body_json,
                tenant_id, federation_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "idem:tenant-b:secret",
                "claim",
                "command:tenant-b",
                "control.duckdb",
                "session:tenant-b",
                "digest:tenant-b",
                "1970-01-01T00:00:00Z",
                "9999-12-31T23:59:59Z",
                '{"private":"tenant-b"}',
                "tenant:b",
                "federation:b",
            ],
        )
    gateway, connection = _gateway(db, socket_path)
    token, _grant = gateway.issue_grant(
        client_id="client:tenant-a-child",
        allowed_operations=(
            "load_store_generation",
            "txn_load_generation",
            "txn_lookup_idempotency",
        ),
        allowed_command_operations=("task.status.cas",),
        tenant_id="tenant:a",
        federation_id="federation:a",
        peer_pid=os.getpid(),
    )
    client = TypedStateOwnerConnection(
        socket_path=socket_path,
        token=token,
        client_id="client:tenant-a-child",
        process_birth_id="birth:tenant-a-child",
        store_id="control.duckdb",
    )
    try:
        generation = client.execute_operation("load_store_generation").fetchone()
        assert generation is not None
        command = StateCommand(
            command_id="command:tenant-a",
            command_kind=CommandKind.CLAIM,
            store_id="control.duckdb",
            session_id=client.session_id,
            expected_generation=int(generation[0]),
            expected_revision=int(generation[3]),
            fence_epoch=int(generation[2]),
            idempotency_key="idem:tenant-a:admitted",
            parameters={
                "operation": "task.status.cas",
                "task_cid": "task:typed-owner",
                "expected_task_revision": 0,
                "status": "claimed",
                "tenant_id": "tenant:a",
                "federation_id": "federation:a",
            },
        )
        client.prepare_command(command)
        client.execute("BEGIN TRANSACTION")
        assert client.execute_operation(
            "txn_lookup_idempotency", [command.idempotency_key]
        ).fetchone() is None
        with pytest.raises(TypedStateOwnerRemoteError) as denied:
            client.execute_operation(
                "txn_lookup_idempotency", ["idem:tenant-b:secret"]
            )
        assert denied.value.error_code == "authorization_denied"
    finally:
        client.close()
        gateway.stop()
        connection.close()
    with open_duckdb_connection(db) as check:
        row = check.execute(
            "SELECT body_json FROM idempotency_records WHERE idempotency_key = ?",
            ["idem:tenant-b:secret"],
        ).fetchone()
        assert row is not None
        assert row["body_json"] == '{"private":"tenant-b"}'


def test_live_grant_expiry_is_checked_on_every_request(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    gateway, connection = _gateway(db, socket_path)
    token, _grant = gateway.issue_grant(
        client_id="client:expiring",
        allowed_operations=("whoami_metadata",),
        peer_pid=os.getpid(),
        ttl_seconds=1.0,
    )
    client = TypedStateOwnerConnection(
        socket_path=socket_path,
        token=token,
        client_id="client:expiring",
        process_birth_id="birth:expiring",
        store_id="control.duckdb",
    )
    try:
        assert client.execute_operation("whoami_metadata").fetchone() is not None
        time.sleep(1.05)
        with pytest.raises(TypedStateOwnerRemoteError) as denied:
            client.execute_operation("whoami_metadata")
        assert denied.value.error_code == "authorization_denied"
    finally:
        client.close()
        gateway.stop()
        connection.close()


def test_completion_progress_is_reserved_for_status_session_grants(
    tmp_path: Path,
) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    gateway, connection = _gateway(db, socket_path)
    try:
        with pytest.raises(
            TypedStateOwnerAuthorizationError,
            match="absent from the server catalog",
        ):
            gateway.issue_grant(
                client_id="client:forged-progress",
                allowed_operations=(COMPLETION_PROGRESS_SNAPSHOT_OPERATION,),
                peer_pid=os.getpid(),
            )
    finally:
        gateway.stop()
        connection.close()


def test_completion_progress_accepts_canonical_empty_control_receipt() -> None:
    owner_identity = {
        "server_id": "server:progress:001",
        "process_birth_id": "birth:progress:001",
        "store_id": "store:progress:001",
        "database_uuid": "12345678-1234-4678-9234-567812345678",
        "generation": 3,
        "fence_epoch": 7,
    }
    request = typed_owner.completion_progress_request(
        owner_identity,
        ["task:progress:001"],
    )
    evidence_digests = ["sha256:" + ("2b" * 32)]
    evidence_digest = content_identity(
        {
            "task_cid": "task:progress:001",
            "revision": 2,
            "receipt": {},
            "evidence_digests": evidence_digests,
        }
    )
    receipt_cid = content_identity(
        {
            "namespace": "completion-receipt",
            "task_cid": "task:progress:001",
            "revision": 2,
            "evidence_digest": evidence_digest,
        }
    )
    projection = {
        "schema": INTENT_COMPLETION_PROJECTION_SCHEMA,
        "event_watermark": 10,
        "task_states": [
            {
                "task_cid": "task:progress:001",
                "status": "completed",
                "revision": 2,
            }
        ],
        "completion_receipts": [
            {
                "receipt_cid": receipt_cid,
                "task_cid": "task:progress:001",
                "goal_cid": "goal:progress:001",
                "attempt_id": "",
                "claim_cid": "",
                "fencing_token": 0,
                "completed_at": "2026-08-26T12:00:00Z",
                "validation_run_id": "",
                "evidence_digest": evidence_digest,
                "body": {
                    "schema": COMPLETION_EVIDENCE_SCHEMA,
                    "receipt": {},
                    "evidence_digests": evidence_digests,
                    "revision": 2,
                },
            }
        ],
    }
    projection["projection_cid"] = content_identity(projection)
    snapshot = {
        "schema": TYPED_COMPLETION_PROGRESS_SNAPSHOT_SCHEMA,
        "request_cid": request["request_cid"],
        "owner_identity": owner_identity,
        "store_generation": StoreGeneration(
            store_id="store:progress:001",
            generation=3,
            schema_revision=14,
            fence_epoch=7,
            revision=19,
            database_uuid="12345678-1234-4678-9234-567812345678",
            birth_id="birth:progress:001",
        ).to_dict(),
        "completion_projection": projection,
    }
    snapshot["snapshot_cid"] = content_identity(snapshot)

    validated = typed_owner.validate_completion_progress_snapshot(
        snapshot,
        request=request,
    )

    assert validated["completion_projection"]["completion_receipts"][0]["body"][
        "receipt"
    ] == {}


@pytest.mark.parametrize(
    "request_id",
    ("", " request:leading-space", "request:embedded space", "r" * 257, 7),
)
def test_request_ids_are_compact_on_open_and_every_followup_request(
    tmp_path: Path,
    request_id: Any,
) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    gateway, connection = _gateway(db, socket_path)
    token, _grant = gateway.issue_grant(
        client_id="client:request-id",
        allowed_operations=("whoami_metadata",),
        peer_pid=os.getpid(),
    )

    def _connect() -> socket.socket:
        channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        channel.settimeout(2.0)
        channel.connect(str(socket_path))
        return channel

    def _open_payload(correlation_id: Any) -> dict[str, Any]:
        return {
            "schema": typed_owner.TYPED_STATE_OWNER_SCHEMA,
            "action": "open",
            "request_id": correlation_id,
            "token": token,
            "client_id": "client:request-id",
            "process_birth_id": "birth:request-id",
            "store_id": "control.duckdb",
        }

    try:
        with _connect() as invalid_open:
            typed_owner._send_frame(  # noqa: SLF001
                invalid_open,
                _open_payload(request_id),
            )
            with pytest.raises((OSError, TypedStateOwnerProtocolError)):
                typed_owner._receive_frame(invalid_open)  # noqa: SLF001

        with _connect() as invalid_followup:
            typed_owner._send_frame(  # noqa: SLF001
                invalid_followup,
                _open_payload("request:valid-open"),
            )
            opened = typed_owner._receive_frame(invalid_followup)  # noqa: SLF001
            assert opened["ok"] is True
            typed_owner._send_frame(  # noqa: SLF001
                invalid_followup,
                {
                    "schema": typed_owner.TYPED_STATE_OWNER_SCHEMA,
                    "action": "execute",
                    "request_id": request_id,
                    "operation": "whoami_metadata",
                    "parameters": [],
                },
            )
            with pytest.raises((OSError, TypedStateOwnerProtocolError)):
                typed_owner._receive_frame(invalid_followup)  # noqa: SLF001
    finally:
        gateway.stop()
        connection.close()


def test_partial_response_timeout_poison_closes_the_client_transport() -> None:
    client_socket, server_socket = socket.socketpair()
    client_socket.settimeout(0.1)
    client = object.__new__(TypedStateOwnerConnection)
    client._socket = client_socket  # noqa: SLF001
    client._request_lock = threading.RLock()  # noqa: SLF001
    client._closed = False  # noqa: SLF001
    client._active = True  # noqa: SLF001
    client._prepared_command = object()  # noqa: SLF001
    client._request_index = 0  # noqa: SLF001
    partial_sent = threading.Event()
    release_server = threading.Event()

    def _send_partial_response() -> None:
        try:
            typed_owner._receive_frame(server_socket)  # noqa: SLF001
            server_socket.sendall((64).to_bytes(4, "big") + b"{")
            partial_sent.set()
            release_server.wait(timeout=2.0)
        finally:
            server_socket.close()

    server = threading.Thread(target=_send_partial_response, daemon=True)
    server.start()
    try:
        with pytest.raises(OSError):
            client._request("execute", operation="whoami_metadata", parameters=[])  # noqa: SLF001
        assert partial_sent.wait(timeout=1.0)
        assert client._closed is True  # noqa: SLF001
        assert client._active is False  # noqa: SLF001
        assert client._prepared_command is None  # noqa: SLF001
        assert client_socket.fileno() == -1
        with pytest.raises(TypedStateOwnerProtocolError, match="closed"):
            client._request("execute", operation="whoami_metadata", parameters=[])  # noqa: SLF001
    finally:
        release_server.set()
        server.join(timeout=2.0)


def _independent_reader(
    receive: Any,
    result: Any,
    socket_path: str,
) -> None:
    token = receive.recv()
    try:
        connection = TypedStateOwnerConnection(
            socket_path=Path(socket_path),
            token=token,
            client_id="client:independent",
            process_birth_id="birth:independent",
            store_id="control.duckdb",
        )
        try:
            result.send(connection.execute_operation("whoami_metadata").fetchone())
        finally:
            connection.close()
    except BaseException as exc:  # pragma: no cover - parent reports exact type
        result.send(("error", type(exc).__name__))


def _independent_status_reader(
    receive: Any,
    result: Any,
    socket_path: str,
) -> None:
    token = receive.recv()
    try:
        connection = TypedStateOwnerConnection(
            socket_path=Path(socket_path),
            token=token,
            client_id=STATUS_BOOTSTRAP_CLIENT_ID,
            process_birth_id=f"birth:status:{os.getpid()}",
            store_id="control.duckdb",
            status_bootstrap=True,
        )
        try:
            count = connection.execute_operation("count_tasks").fetchone()
            completion_snapshot = connection.completion_progress_snapshot(
                ["task:typed-owner"]
            )
            denied: list[str] = []
            for operation, parameters in (
                ("casf_select_federation", ["federation:forged", "tenant:forged"]),
                (
                    "casf_select_supervisor_bootstrap_health",
                    [
                        "tenant:other",
                        "federation:other",
                        "supervisor:other",
                        "subscription:other",
                        "consumer:other",
                        "event:other",
                        "acknowledgement:other",
                        "attempt:other",
                        "9999-12-31T23:59:59Z",
                    ],
                ),
                (
                    "seed_client_session",
                    [
                        "session:forged",
                        "server:forged",
                        "client:forged",
                        "birth:forged",
                        "1970-01-01T00:00:00Z",
                        "1970-01-01T00:00:00Z",
                        1,
                        1,
                        "attached",
                        0,
                    ],
                ),
            ):
                try:
                    connection.execute_operation(operation, parameters)
                except TypedStateOwnerRemoteError as exc:
                    denied.append(exc.error_code)
            raw_sql_denied = False
            try:
                connection.execute("SELECT 1")
            except TypedStateOwnerProtocolError:
                raw_sql_denied = True
            result.send(
                {
                    "count": count,
                    "completion_snapshot": dict(completion_snapshot),
                    "grant": dict(connection.grant),
                    "denied": denied,
                    "raw_sql_denied": raw_sql_denied,
                }
            )
        finally:
            connection.close()
    except BaseException as exc:  # pragma: no cover - parent reports exact type
        result.send({"error": type(exc).__name__})


def test_grant_is_kernel_bound_to_an_independent_process(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    gateway, connection = _gateway(db, socket_path)
    context = multiprocessing.get_context("spawn")
    token_receive, token_send = context.Pipe(duplex=False)
    result_receive, result_send = context.Pipe(duplex=False)
    process = context.Process(
        target=_independent_reader,
        args=(token_receive, result_send, str(socket_path)),
    )
    process.start()
    try:
        token, _grant = gateway.issue_grant(
            client_id="client:independent",
            allowed_operations=("whoami_metadata",),
            peer_pid=process.pid,
        )
        token_send.send(token)
        assert result_receive.poll(10.0)
        assert result_receive.recv() is not None
    finally:
        process.join(timeout=10.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
        gateway.stop()
        connection.close()
    assert process.exitcode == 0


def test_status_bootstrap_rebinds_each_distinct_process_read_only(
    tmp_path: Path,
) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    gateway, connection = _gateway(db, socket_path)
    connection.execute(
        """
        INSERT INTO federations VALUES (
            'federation:status', 'tenant:status', 'program:status',
            'objective:status', 1, 'policy:status', 1, 'catalog:status',
            1, 1, 'semantic:status', 'admitted', 1, 1, 1, 1,
            'issuer:status', 'evidence:status', '9999-12-31T23:59:59Z',
            '1970-01-01T00:00:00Z', '1970-01-01T00:00:00Z',
            'content:status', '{}'
        )
        """
    )
    connection.execute(
        """
        INSERT INTO supervisor_instances (
            supervisor_id, repository_id, process_birth_id, started_at,
            status, revision, tenant_id, federation_id, lifecycle_state
        ) VALUES (
            'supervisor:status', 'repository:status', 'logical:not-started',
            '1970-01-01T00:00:00Z', 'ADMITTED', 1,
            'tenant:status', 'federation:status', 'ADMITTED'
        )
        """
    )
    connection.execute(
        """
        INSERT INTO event_subscriptions (
            subscription_id, tenant_id, federation_id, consumer_id,
            supervisor_id, revision, maximum_batch, maximum_pending,
            maximum_fanout, retry_budget, status, created_at, updated_at
        ) VALUES (
            'subscription:status', 'tenant:status', 'federation:status',
            'consumer:status', 'supervisor:status', 1, 1, 1, 1, 1,
            'active', '1970-01-01T00:00:00Z', '1970-01-01T00:00:00Z'
        )
        """
    )
    bootstrap_token = gateway.configure_status_bootstrap()
    gateway.bind_status_bootstrap_scope()
    context = multiprocessing.get_context("spawn")
    observed: list[dict[str, Any]] = []
    try:
        for _index in range(2):
            token_receive, token_send = context.Pipe(duplex=False)
            result_receive, result_send = context.Pipe(duplex=False)
            process = context.Process(
                target=_independent_status_reader,
                args=(token_receive, result_send, str(socket_path)),
            )
            process.start()
            token_send.send(bootstrap_token)
            assert result_receive.poll(10.0)
            payload = result_receive.recv()
            process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
            assert process.exitcode == 0
            assert payload.get("error") is None
            assert payload["count"] == (1,)
            completion_snapshot = payload["completion_snapshot"]
            assert completion_snapshot["schema"] == (
                TYPED_COMPLETION_PROGRESS_SNAPSHOT_SCHEMA
            )
            assert completion_snapshot["completion_projection"]["task_states"] == [
                {
                    "task_cid": "task:typed-owner",
                    "status": "ready",
                    "revision": 0,
                }
            ]
            assert completion_snapshot["snapshot_cid"].startswith("b")
            assert payload["denied"] == [
                "authorization_denied",
                "authorization_denied",
                "authorization_denied",
            ]
            assert payload["raw_sql_denied"] is True
            grant = payload["grant"]
            assert grant["client_id"] == STATUS_BOOTSTRAP_CLIENT_ID
            assert grant["peer_pid"] == process.pid
            assert grant["peer_uid"] == os.getuid()
            assert set(grant["allowed_operations"]) == set(
                STATUS_BOOTSTRAP_ALLOWED_OPERATIONS
            )
            assert COMPLETION_PROGRESS_SNAPSHOT_OPERATION in grant[
                "allowed_operations"
            ]
            assert grant["allowed_command_operations"] == []
            assert grant["tenant_id"] == "tenant:status"
            assert grant["federation_id"] == "federation:status"
            assert grant["entity_scopes"] == {
                "consumer_id": "consumer:status",
                "subscription_id": "subscription:status",
                "supervisor_id": "supervisor:status",
            }
            assert grant["authority_profile"] == "dedicated_store_status_portfolio"
            assert 0 < int(grant["expires_at"]) - int(grant["issued_at"]) <= 60_000
            observed.append(payload)

        assert observed[0]["grant"]["peer_pid"] != observed[1]["grant"]["peer_pid"]
        # Connection-local grants are retired instead of accumulating as an
        # ambient same-UID capability after each status process exits.
        deadline = time.monotonic() + 2.0
        while gateway.capability()["active_grants"] and time.monotonic() < deadline:
            time.sleep(0.01)
        assert gateway.capability()["active_grants"] == 0

        connection.execute(
            """
            INSERT INTO federations VALUES (
                'federation:other', 'tenant:other', 'program:other',
                'objective:other', 1, 'policy:other', 1, 'catalog:other',
                1, 1, 'semantic:other', 'admitted', 1, 1, 1, 1,
                'issuer:other', 'evidence:other', '9999-12-31T23:59:59Z',
                '1970-01-01T00:00:00Z', '1970-01-01T00:00:00Z',
                'content:other', '{}'
            )
            """
        )
        with pytest.raises(TypedStateOwnerError):
            TypedStateOwnerConnection(
                socket_path=socket_path,
                token=bootstrap_token,
                client_id=STATUS_BOOTSTRAP_CLIENT_ID,
                process_birth_id="birth:multiple-federations",
                store_id="control.duckdb",
                status_bootstrap=True,
            )
        connection.execute(
            "DELETE FROM federations WHERE federation_id = 'federation:other'"
        )

        for token, client_id, store_id in (
            ("0" * 64, STATUS_BOOTSTRAP_CLIENT_ID, "control.duckdb"),
            (bootstrap_token, "client:forged", "control.duckdb"),
            (bootstrap_token, STATUS_BOOTSTRAP_CLIENT_ID, "other.duckdb"),
        ):
            with pytest.raises(TypedStateOwnerError):
                TypedStateOwnerConnection(
                    socket_path=socket_path,
                    token=token,
                    client_id=client_id,
                    process_birth_id="birth:rejected-status",
                    store_id=store_id,
                    status_bootstrap=True,
                )

        original_uid = gateway._status_bootstrap_uid
        gateway._status_bootstrap_uid = original_uid + 1
        try:
            with pytest.raises(TypedStateOwnerError):
                TypedStateOwnerConnection(
                    socket_path=socket_path,
                    token=bootstrap_token,
                    client_id=STATUS_BOOTSTRAP_CLIENT_ID,
                    process_birth_id="birth:wrong-uid",
                    store_id="control.duckdb",
                    status_bootstrap=True,
                )
        finally:
            gateway._status_bootstrap_uid = original_uid
    finally:
        gateway.stop()
        connection.close()


def test_budget_policy_and_usage_query_are_tenant_scoped(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    gateway, connection = _gateway(db, socket_path)
    token, grant = gateway.issue_grant(
        client_id="client:budget",
        allowed_operations=("casf_select_active_admission_budget_usage",),
        allowed_command_operations=(
            "budget.reserve",
            "budget.release",
            "federation.create",
        ),
        tenant_id="tenant:one",
        peer_pid=os.getpid(),
    )
    client = TypedStateOwnerConnection(
        socket_path=socket_path,
        token=token,
        client_id="client:budget",
        process_birth_id="birth:budget",
        store_id="control.duckdb",
    )
    try:
        assert client.execute_operation(
            "casf_select_active_admission_budget_usage",
            ["tenant:one", "9999-12-31T23:59:59Z"],
        ).fetchall() == []
        with pytest.raises(TypedStateOwnerRemoteError) as denied:
            client.execute_operation(
                "casf_select_active_admission_budget_usage",
                ["tenant:two", "9999-12-31T23:59:59Z"],
            )
        assert denied.value.error_code == "authorization_denied"
        assert grant.federation_id == ""
        assert grant.allowed_command_operations == frozenset(
            {"budget.reserve", "budget.release", "federation.create"}
        )
        policies = typed_owner._COMMAND_MUTATION_CATALOG
        assert {
            "casf_insert_admission_budget_reservation",
            "casf_insert_admission_budget_dimension",
        }.issubset(policies["budget.reserve"])
        assert "casf_transition_admission_budget_reservation" in policies[
            "budget.release"
        ]
        assert "casf_transition_admission_budget_reservation" in policies[
            "federation.create"
        ]
    finally:
        client.close()
        gateway.stop()
        connection.close()


@pytest.mark.parametrize(
    ("operation", "include_partial_state"),
    (("federation.create", True), ("supervisor.register", False)),
)
def test_partial_federation_commands_cannot_commit(
    tmp_path: Path,
    operation: str,
    include_partial_state: bool,
) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    gateway, owner_connection = _gateway(db, socket_path)
    allowed = [
        "load_store_generation",
        "txn_advance_store_revision",
        "txn_record_idempotency",
    ]
    if include_partial_state:
        allowed.append("casf_insert_federation")
    token, _grant = gateway.issue_grant(
        client_id=f"client:partial:{operation}",
        allowed_operations=tuple(allowed),
        allowed_command_operations=(operation,),
        tenant_id="tenant:one",
        federation_id="federation:one",
        peer_pid=os.getpid(),
    )
    client = TypedStateOwnerConnection(
        socket_path=socket_path,
        token=token,
        client_id=f"client:partial:{operation}",
        process_birth_id=f"birth:partial:{operation}",
        store_id="control.duckdb",
    )
    head = client.execute_operation("load_store_generation").fetchone()
    assert head is not None
    command = StateCommand(
        command_id=f"command:partial:{operation}",
        command_kind=CommandKind.APPEND,
        store_id="control.duckdb",
        session_id=client.session_id,
        expected_generation=int(head[0]),
        expected_revision=int(head[3]),
        fence_epoch=int(head[2]),
        idempotency_key=f"idempotency:partial:{operation}",
        parameters={
            "operation": operation,
            "tenant_id": "tenant:one",
            "federation_id": "federation:one",
        },
    )
    try:
        client.prepare_command(command)
        client.execute("BEGIN TRANSACTION")
        if include_partial_state:
            client.execute_operation(
                "casf_insert_federation",
                [
                    "federation:one",
                    "tenant:one",
                    "program:one",
                    "objective:one",
                    1,
                    "policy:one",
                    1,
                    "catalog:one",
                    int(head[0]),
                    1,
                    "semantic:one",
                    "admitted",
                    12,
                    256,
                    1,
                    int(head[2]),
                    "issuer:one",
                    "evidence:one",
                    "9999-12-31T23:59:59Z",
                    "1970-01-01T00:00:00Z",
                    "1970-01-01T00:00:00Z",
                    "content:one",
                    "{}",
                ],
            )
        client.execute_operation(
            "txn_advance_store_revision",
            [int(head[3]) + 1, int(head[0]), int(head[3]), int(head[2])],
        )
        client.execute_operation(
            "txn_record_idempotency",
            [
                command.idempotency_key,
                command.command_kind.value,
                command.command_id,
                command.store_id,
                command.session_id,
                "sha256:partial",
                "1970-01-01T00:00:00Z",
                None,
                "{}",
            ],
        )
        with pytest.raises(TypedStateOwnerRemoteError) as denied:
            client.commit()
        assert denied.value.error_code == "authorization_denied"
    finally:
        client.close()
    assert owner_connection.execute(
        "SELECT revision FROM store_generations ORDER BY generation DESC LIMIT 1"
    ).fetchone()[0] == int(head[3])
    assert owner_connection.execute(
        "SELECT COUNT(*) FROM idempotency_records WHERE idempotency_key = ?",
        [command.idempotency_key],
    ).fetchone()[0] == 0
    assert owner_connection.execute(
        "SELECT COUNT(*) FROM federations WHERE federation_id = 'federation:one'"
    ).fetchone()[0] == 0
    gateway.stop()
    owner_connection.close()


@pytest.mark.parametrize(
    "attack",
    ("duplicate", "out_of_order", "omitted_history"),
)
def test_manifest_rejects_duplicate_and_out_of_order_mutations(
    tmp_path: Path,
    attack: str,
) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    gateway, owner_connection = _gateway(db, socket_path)
    token, _grant = gateway.issue_grant(
        client_id=f"client:manifest:{attack}",
        allowed_operations=_task_grant_operations(),
        allowed_command_operations=("task.status.cas",),
        peer_pid=os.getpid(),
    )
    client = TypedStateOwnerConnection(
        socket_path=socket_path,
        token=token,
        client_id=f"client:manifest:{attack}",
        process_birth_id=f"birth:manifest:{attack}",
        store_id="control.duckdb",
    )
    head = client.execute_operation("load_store_generation").fetchone()
    assert head is not None
    command = StateCommand(
        command_id=f"command:manifest:{attack}",
        command_kind=CommandKind.CLAIM,
        store_id="control.duckdb",
        session_id=client.session_id,
        expected_generation=int(head[0]),
        expected_revision=int(head[3]),
        fence_epoch=int(head[2]),
        idempotency_key=f"idempotency:manifest:{attack}",
        parameters={
            "operation": "task.status.cas",
            "task_cid": "task:typed-owner",
            "expected_task_revision": 0,
            "status": "claimed",
        },
    )
    try:
        client.prepare_command(command)
        client.execute("BEGIN TRANSACTION")
        if attack == "duplicate":
            parameters = [
                "claimed",
                "1970-01-01T00:00:01Z",
                1,
                "task:typed-owner",
                0,
            ]
            client.execute_operation("txn_cas_task_status", parameters)
            with pytest.raises(TypedStateOwnerRemoteError) as denied:
                client.execute_operation("txn_cas_task_status", parameters)
        elif attack == "out_of_order":
            with pytest.raises(TypedStateOwnerRemoteError) as denied:
                client.execute_operation(
                    "txn_record_idempotency",
                    [
                        command.idempotency_key,
                        command.command_kind.value,
                        command.command_id,
                        command.store_id,
                        command.session_id,
                        "sha256:out-of-order",
                        "1970-01-01T00:00:00Z",
                        None,
                        "{}",
                    ],
                )
        else:
            client.execute_operation(
                "txn_cas_task_status",
                [
                    "claimed",
                    "1970-01-01T00:00:01Z",
                    1,
                    "task:typed-owner",
                    0,
                ],
            )
            client.execute_operation(
                "txn_advance_store_revision",
                [int(head[3]) + 1, int(head[0]), int(head[3]), int(head[2])],
            )
            client.execute_operation(
                "txn_record_idempotency",
                [
                    command.idempotency_key,
                    command.command_kind.value,
                    command.command_id,
                    command.store_id,
                    command.session_id,
                    "sha256:omitted-history",
                    "1970-01-01T00:00:00Z",
                    None,
                    "{}",
                ],
            )
            with pytest.raises(TypedStateOwnerRemoteError) as denied:
                client.commit()
        assert denied.value.error_code == "authorization_denied"
    finally:
        client.close()
    task = owner_connection.execute(
        "SELECT status, revision FROM tasks WHERE task_cid = 'task:typed-owner'"
    ).fetchone()
    assert task is not None
    assert (task[0], task[1]) == ("ready", 0)
    gateway.stop()
    owner_connection.close()


@pytest.mark.parametrize(
    "attack",
    (
        "missing",
        "reordered",
        "duplicate",
        "mismatched",
        "mismatched_timestamp",
        "mismatched_idempotency",
        "bool_store_seal",
        "stale_cas",
        "gapped_history",
    ),
)
def test_receipt_manifest_requires_exact_atomic_task_revision_history(
    tmp_path: Path,
    attack: str,
) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    if attack == "gapped_history":
        with open_duckdb_connection(db) as seed:
            seed.execute(
                "UPDATE tasks SET revision = 2 WHERE task_cid = 'task:typed-owner'"
            )
            seed.execute(
                """
                INSERT INTO task_revisions (
                    task_cid, revision, status, body_json, recorded_at
                ) VALUES ('task:typed-owner', 1, 'ready', '{}',
                          '1970-01-01T00:00:00Z')
                """
            )
    gateway, owner_connection = _gateway(db, socket_path)
    baseline_task = owner_connection.execute(
        "SELECT status, revision FROM tasks WHERE task_cid = 'task:typed-owner'"
    ).fetchone()
    assert baseline_task is not None
    baseline_task = (baseline_task[0], baseline_task[1])
    baseline_history_count = owner_connection.execute(
        "SELECT COUNT(*) FROM task_revisions "
        "WHERE task_cid = 'task:typed-owner'"
    ).fetchone()[0]
    client_id = f"client:receipt-history:{attack}"
    token, _grant = gateway.issue_grant(
        client_id=client_id,
        allowed_operations=(
            *_task_grant_operations(),
            "executor_cas_task_status_receipt",
            "executor_insert_task_revision",
        ),
        allowed_command_operations=("task.status.cas.receipt",),
        peer_pid=os.getpid(),
    )
    client = TypedStateOwnerConnection(
        socket_path=socket_path,
        token=token,
        client_id=client_id,
        process_birth_id=f"birth:receipt-history:{attack}",
        store_id="control.duckdb",
    )
    head = client.execute_operation("load_store_generation").fetchone()
    assert head is not None
    expected_task_revision = (
        2
        if attack == "gapped_history"
        else 1
        if attack == "stale_cas"
        else 0
    )
    command = StateCommand(
        command_id=f"command:receipt-history:{attack}",
        command_kind=CommandKind.CLAIM,
        store_id="control.duckdb",
        session_id=client.session_id,
        expected_generation=int(head[0]),
        expected_revision=int(head[3]),
        fence_epoch=int(head[2]),
        idempotency_key=f"idempotency:receipt-history:{attack}",
        parameters={
            "operation": "task.status.cas.receipt",
            "task_cid": "task:typed-owner",
            "expected_task_revision": expected_task_revision,
            "status": "claimed",
            "body_json": "{}",
        },
    )
    recorded_at = "1970-01-01T00:00:01Z"
    receipt_parameters = [
        "claimed",
        expected_task_revision + 1,
        recorded_at,
        "{}",
        "task:typed-owner",
        expected_task_revision,
    ]
    history_parameters = [
        "task:typed-owner",
        expected_task_revision + 1,
        "claimed",
        "{}",
        recorded_at,
    ]
    try:
        client.prepare_command(command)
        client.execute("BEGIN TRANSACTION")
        if attack == "reordered":
            with pytest.raises(TypedStateOwnerRemoteError) as denied:
                client.execute_operation(
                    "executor_insert_task_revision", history_parameters
                )
        elif attack == "stale_cas":
            # The merged owner validates exact single-row task mutations at
            # execute time, before a stale caller can append history or seals.
            with pytest.raises(TypedStateOwnerRemoteError) as denied:
                client.execute_operation(
                    "executor_cas_task_status_receipt", receipt_parameters
                )
        else:
            client.execute_operation(
                "executor_cas_task_status_receipt", receipt_parameters
            )
            if attack == "mismatched":
                bad_history = list(history_parameters)
                bad_history[3] = '{"forged":true}'
                with pytest.raises(TypedStateOwnerRemoteError) as denied:
                    client.execute_operation(
                        "executor_insert_task_revision", bad_history
                    )
            elif attack == "mismatched_timestamp":
                bad_history = list(history_parameters)
                bad_history[4] = "1970-01-01T00:00:02Z"
                with pytest.raises(TypedStateOwnerRemoteError) as denied:
                    client.execute_operation(
                        "executor_insert_task_revision", bad_history
                    )
            elif attack == "duplicate":
                client.execute_operation(
                    "executor_insert_task_revision", history_parameters
                )
                with pytest.raises(TypedStateOwnerRemoteError) as denied:
                    client.execute_operation(
                        "executor_insert_task_revision", history_parameters
                    )
            elif attack == "bool_store_seal":
                client.execute_operation(
                    "executor_insert_task_revision", history_parameters
                )
                with pytest.raises(TypedStateOwnerRemoteError) as denied:
                    client.execute_operation(
                        "txn_advance_store_revision",
                        [True, int(head[0]), int(head[3]), int(head[2])],
                    )
            elif attack in {"gapped_history", "mismatched_idempotency"}:
                client.execute_operation(
                    "executor_insert_task_revision", history_parameters
                )
                client.execute_operation(
                    "txn_advance_store_revision",
                    [
                        int(head[3]) + 1,
                        int(head[0]),
                        int(head[3]),
                        int(head[2]),
                    ],
                )
                client.execute_operation(
                    "txn_record_idempotency",
                    [
                        command.idempotency_key,
                        command.command_kind.value,
                        command.command_id,
                        command.store_id,
                        command.session_id,
                        (
                            "sha256:forged"
                            if attack == "mismatched_idempotency"
                            else result_digest({})
                        ),
                        recorded_at,
                        None,
                        "{}",
                    ],
                )
                with pytest.raises(TypedStateOwnerRemoteError) as denied:
                    client.commit()
            else:
                client.execute_operation(
                    "txn_advance_store_revision",
                    [
                        int(head[3]) + 1,
                        int(head[0]),
                        int(head[3]),
                        int(head[2]),
                    ],
                )
                client.execute_operation(
                    "txn_record_idempotency",
                    [
                        command.idempotency_key,
                        command.command_kind.value,
                        command.command_id,
                        command.store_id,
                        command.session_id,
                        result_digest({}),
                        recorded_at,
                        None,
                        "{}",
                    ],
                )
                with pytest.raises(TypedStateOwnerRemoteError) as denied:
                    client.commit()
        assert denied.value.error_code == "authorization_denied"
    finally:
        client.close()
    task_row = owner_connection.execute(
        "SELECT status, revision FROM tasks WHERE task_cid = 'task:typed-owner'"
    ).fetchone()
    assert task_row is not None
    assert (task_row[0], task_row[1]) == baseline_task
    assert owner_connection.execute(
        "SELECT COUNT(*) FROM task_revisions "
        "WHERE task_cid = 'task:typed-owner'"
    ).fetchone()[0] == baseline_history_count
    assert owner_connection.execute(
        "SELECT revision FROM store_generations ORDER BY generation DESC LIMIT 1"
    ).fetchone()[0] == int(head[3])
    assert owner_connection.execute(
        "SELECT COUNT(*) FROM idempotency_records WHERE idempotency_key = ?",
        [command.idempotency_key],
    ).fetchone()[0] == 0
    gateway.stop()
    owner_connection.close()


def test_retry_cooldown_extension_rejects_bool_for_integer_identity() -> None:
    extension = {
        "schema": typed_owner.TYPED_RETRY_COOLDOWN_SCHEMA,
        "task_cid": "task:typed-owner",
        "expected_task_revision": 0,
        "attempt_id": "attempt:one",
        "claim_id": "claim:one",
        "lease_id": "lease:one",
        "owner_session_id": "owner:one",
        "attempt_number": 1,
        "fencing_token": 1,
        "fence_epoch": 1,
        "delay_ms": 100,
        "started_at_ms": 1_000,
        "retry_not_before_ms": 1_100,
        "selection_penalty": 0,
        "consecutive_failures": 1,
        "reason": "bounded retry",
        "expected_queue_revision": -1,
        "expected_queue_attempt": 0,
    }
    parameters = {
        **extension,
        "operation": "task.retry.cooldown.record",
        "expected_task_status": "in_progress",
        "resolution_cid": content_identity(
            {
                "typed_retry_cooldown": extension,
                "started_at_ms": extension["started_at_ms"],
            }
        ),
        "extension_schema": typed_owner.TYPED_RETRY_COOLDOWN_SCHEMA,
        "extension_json": canonical_json_bytes(extension).decode("utf-8"),
    }
    assert typed_owner._validated_retry_cooldown_parameters(parameters)

    forged_extension = {**extension, "fencing_token": True}
    forged = {
        **parameters,
        "extension_json": canonical_json_bytes(forged_extension).decode(
            "utf-8"
        ),
    }
    with pytest.raises(
        TypedStateOwnerAuthorizationError,
        match="extension or resolution identity differs",
    ):
        typed_owner._validated_retry_cooldown_parameters(forged)

    dead_extension = {
        **extension,
        "delay_ms": 0,
        "retry_not_before_ms": extension["started_at_ms"],
        "reason": typed_owner.TYPED_DATABASE_CLAIM_RECOVERY_REASON,
    }
    dead_recovery = {
        **dead_extension,
        "operation": typed_owner.TYPED_DATABASE_CLAIM_RECOVERY_COMMAND,
        "expected_task_status": "in_progress",
        "resolution_cid": content_identity(
            {
                "typed_retry_cooldown": dead_extension,
                "started_at_ms": dead_extension["started_at_ms"],
            }
        ),
        "extension_schema": typed_owner.TYPED_RETRY_COOLDOWN_SCHEMA,
        "extension_json": canonical_json_bytes(dead_extension).decode(
            "utf-8"
        ),
        "status": "retrying",
        "body_json": (
            '{"completion_receipt":{},"completion_receipt":{"forged":true}}'
        ),
    }
    with pytest.raises(
        TypedStateOwnerAuthorizationError,
        match="dead claim recovery task body is malformed",
    ):
        typed_owner._validated_dead_claim_recovery_parameters(dead_recovery)


@pytest.mark.parametrize("attack", ("unknown_field", "duplicate_body_key"))
def test_receipt_command_admission_rejects_nonclosed_json_or_fields(
    tmp_path: Path,
    attack: str,
) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    gateway, owner_connection = _gateway(db, socket_path)
    client_id = f"client:receipt-admission:{attack}"
    token, _grant = gateway.issue_grant(
        client_id=client_id,
        allowed_operations=(
            *_task_grant_operations(),
            "executor_cas_task_status_receipt",
            "executor_insert_task_revision",
        ),
        allowed_command_operations=("task.status.cas.receipt",),
        peer_pid=os.getpid(),
    )
    client = TypedStateOwnerConnection(
        socket_path=socket_path,
        token=token,
        client_id=client_id,
        process_birth_id=f"birth:receipt-admission:{attack}",
        store_id="control.duckdb",
    )
    head = client.execute_operation("load_store_generation").fetchone()
    assert head is not None
    parameters: dict[str, Any] = {
        "operation": "task.status.cas.receipt",
        "task_cid": "task:typed-owner",
        "expected_task_revision": 0,
        "status": "claimed",
        "body_json": (
            '{"completion_receipt":{},"completion_receipt":{"forged":true}}'
            if attack == "duplicate_body_key"
            else "{}"
        ),
    }
    if attack == "unknown_field":
        parameters["forged"] = True
    command = StateCommand(
        command_id=f"command:receipt-admission:{attack}",
        command_kind=CommandKind.CLAIM,
        store_id="control.duckdb",
        session_id=client.session_id,
        expected_generation=int(head[0]),
        expected_revision=int(head[3]),
        fence_epoch=int(head[2]),
        idempotency_key=f"idempotency:receipt-admission:{attack}",
        parameters=parameters,
    )
    try:
        client.prepare_command(command)
        with pytest.raises(TypedStateOwnerRemoteError) as denied:
            client.execute("BEGIN TRANSACTION")
        assert denied.value.error_code == "authorization_denied"
    finally:
        client.close()
        gateway.stop()
        owner_connection.close()


def test_expected_receipt_owner_rejects_bool_for_integer_identity(
    tmp_path: Path,
) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    current_body_json = canonical_json_bytes(
        {"completion_receipt": {"fencing_token": 1}}
    ).decode("utf-8")
    with open_duckdb_connection(db) as seed:
        seed.execute(
            "UPDATE tasks SET body_json = ? WHERE task_cid = ?",
            [current_body_json, "task:typed-owner"],
        )
    gateway, owner_connection = _gateway(db, socket_path)
    client_id = "client:receipt-bool-int"
    token, _grant = gateway.issue_grant(
        client_id=client_id,
        allowed_operations=(
            *_task_grant_operations(),
            "executor_cas_task_status_receipt",
            "executor_insert_task_revision",
        ),
        allowed_command_operations=("task.status.cas.receipt",),
        peer_pid=os.getpid(),
    )
    client = TypedStateOwnerConnection(
        socket_path=socket_path,
        token=token,
        client_id=client_id,
        process_birth_id="birth:receipt-bool-int",
        store_id="control.duckdb",
    )
    head = client.execute_operation("load_store_generation").fetchone()
    assert head is not None
    command = StateCommand(
        command_id="command:receipt-bool-int",
        command_kind=CommandKind.CLAIM,
        store_id="control.duckdb",
        session_id=client.session_id,
        expected_generation=int(head[0]),
        expected_revision=int(head[3]),
        fence_epoch=int(head[2]),
        idempotency_key="idempotency:receipt-bool-int",
        parameters={
            "operation": "task.status.cas.receipt",
            "task_cid": "task:typed-owner",
            "expected_task_revision": 0,
            "status": "claimed",
            "body_json": current_body_json,
            "expected_control_receipt_json": canonical_json_bytes(
                {"fencing_token": True}
            ).decode("utf-8"),
        },
    )
    try:
        client.prepare_command(command)
        client.execute("BEGIN TRANSACTION")
        with pytest.raises(TypedStateOwnerRemoteError) as denied:
            client.execute_operation(
                "executor_cas_task_status_receipt",
                [
                    "claimed",
                    1,
                    "1970-01-01T00:00:01Z",
                    current_body_json,
                    "task:typed-owner",
                    0,
                ],
            )
        assert denied.value.error_code == "authorization_denied"
    finally:
        client.close()
        task_row = owner_connection.execute(
            "SELECT status, revision FROM tasks WHERE task_cid = ?",
            ["task:typed-owner"],
        ).fetchone()
        assert task_row is not None
        assert (task_row[0], task_row[1]) == ("ready", 0)
        gateway.stop()
        owner_connection.close()


def test_stale_store_revision_is_rejected_before_domain_mutation(
    tmp_path: Path,
) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    with open_duckdb_connection(db) as seed:
        seed.execute("UPDATE store_generations SET revision = revision + 1")
    gateway, owner_connection = _gateway(db, socket_path)
    client_id = "client:stale-store-seal"
    token, _grant = gateway.issue_grant(
        client_id=client_id,
        allowed_operations=_task_grant_operations(),
        allowed_command_operations=("task.status.cas",),
        peer_pid=os.getpid(),
    )
    client = TypedStateOwnerConnection(
        socket_path=socket_path,
        token=token,
        client_id=client_id,
        process_birth_id="birth:stale-store-seal",
        store_id="control.duckdb",
    )
    head = client.execute_operation("load_store_generation").fetchone()
    assert head is not None and int(head[3]) == 1
    command = StateCommand(
        command_id="command:stale-store-seal",
        command_kind=CommandKind.CLAIM,
        store_id="control.duckdb",
        session_id=client.session_id,
        expected_generation=int(head[0]),
        expected_revision=0,
        fence_epoch=int(head[2]),
        idempotency_key="idempotency:stale-store-seal",
        parameters={
            "operation": "task.status.cas",
            "task_cid": "task:typed-owner",
            "expected_task_revision": 0,
            "status": "claimed",
        },
    )
    try:
        client.prepare_command(command)
        with pytest.raises(OptimisticConflictError):
            client.execute("BEGIN TRANSACTION")
    finally:
        client.close()
        task_row = owner_connection.execute(
            "SELECT status, revision FROM tasks WHERE task_cid = ?",
            ["task:typed-owner"],
        ).fetchone()
        assert task_row is not None
        assert (task_row[0], task_row[1]) == ("ready", 0)
        assert owner_connection.execute(
            "SELECT revision FROM store_generations"
        ).fetchone()[0] == 1
        gateway.stop()
        owner_connection.close()


def test_owner_rejects_zero_row_task_cas_before_history_or_seals(
    tmp_path: Path,
) -> None:
    db = tmp_path / "control.duckdb"
    socket_path = tmp_path / "owner.sock"
    _install(db)
    with open_duckdb_connection(db) as connection:
        connection.execute(
            "UPDATE tasks SET status = 'claimed', revision = 1, "
            "updated_at = '1970-01-01T00:00:01Z' "
            "WHERE task_cid = 'task:typed-owner'"
        )
    gateway, owner_connection = _gateway(db, socket_path)
    token, _grant = gateway.issue_grant(
        client_id="client:zero-row-cas",
        allowed_operations=_task_grant_operations(),
        allowed_command_operations=("task.status.cas",),
        peer_pid=os.getpid(),
    )
    client = TypedStateOwnerConnection(
        socket_path=socket_path,
        token=token,
        client_id="client:zero-row-cas",
        process_birth_id="birth:zero-row-cas",
        store_id="control.duckdb",
    )
    head = client.execute_operation("load_store_generation").fetchone()
    assert head is not None
    command = StateCommand(
        command_id="command:zero-row-cas",
        command_kind=CommandKind.CLAIM,
        store_id="control.duckdb",
        session_id=client.session_id,
        expected_generation=int(head[0]),
        expected_revision=int(head[3]),
        fence_epoch=int(head[2]),
        idempotency_key="idempotency:zero-row-cas",
        parameters={
            "operation": "task.status.cas",
            "task_cid": "task:typed-owner",
            "expected_task_revision": 0,
            "status": "claimed",
        },
    )
    try:
        client.prepare_command(command)
        client.execute("BEGIN TRANSACTION")
        with pytest.raises(TypedStateOwnerRemoteError) as denied:
            client.execute_operation(
                "txn_cas_task_status",
                [
                    "claimed",
                    "1970-01-01T00:00:02Z",
                    1,
                    "task:typed-owner",
                    0,
                ],
            )
        assert denied.value.error_code == "authorization_denied"
    finally:
        client.close()
    task_row = owner_connection.execute(
        "SELECT status, revision FROM tasks "
        "WHERE task_cid = 'task:typed-owner'"
    ).fetchone()
    assert task_row is not None and (task_row[0], task_row[1]) == ("claimed", 1)
    history_count = owner_connection.execute(
        "SELECT COUNT(*) FROM task_revisions "
        "WHERE task_cid = 'task:typed-owner'"
    ).fetchone()
    assert history_count is not None and history_count[0] == 0
    idempotency_count = owner_connection.execute(
        "SELECT COUNT(*) FROM idempotency_records "
        "WHERE idempotency_key = 'idempotency:zero-row-cas'"
    ).fetchone()
    assert idempotency_count is not None and idempotency_count[0] == 0
    gateway.stop()
    owner_connection.close()
