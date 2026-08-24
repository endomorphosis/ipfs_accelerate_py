"""Qualification for the configured CASF admitted-executor boundary."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    OwnerLiveness,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    FakeQuackTransport,
    build_server,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    canonical_json_bytes,
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_transactions import (
    RetryPolicy,
    TransactionError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
    TaskRecord,
    TaskSourceIntegrityError,
    TaskSourceSnapshot,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
    QuackStateClient,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import (
    DETERMINISTIC_ONLY_EXECUTION_MODE,
    GROK_CODEX_EXECUTION_MODE,
    TaskExecutionRoutePolicy,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_database_task_source import (
    TypedDatabaseTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA,
    TYPED_DATABASE_CLAIM_RESERVATION_SCHEMA,
    TYPED_RETRY_COOLDOWN_SCHEMA,
    TYPED_STATE_OWNER_SOCKET_ENV,
    TYPED_STATE_OWNER_TOKEN_ENV,
    TypedStateOwnerAuthorizationError,
    TypedStateOwnerConnection,
    TypedStateOwnerError,
    build_control_plane_operation_catalog,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    ATTEMPT_PHASE_CLAIMED,
    ATTEMPT_PHASE_FAILED,
    DATABASE_POST_MERGE_RECOVERY_SCHEMA,
    POST_MERGE_DECLARED_OUTPUT_REPAIR_SCHEMA,
    DatabaseImplementationAuthorityError,
    DatabaseImplementationDaemon,
)
from test.api.causal_federation.test_bootstrap_runtime import (
    _capability,
    _migrate,
)

ROOT = Path(__file__).resolve().parents[3]
OPERATOR_PATH = ROOT / "scripts/run_agent_supervisor_causal_event_federation.py"
CONFIG = ROOT / "config/agent_supervisor_causal_event_federation_scheduler.json"
CHILD = Path(__file__).with_name("executor_bootstrap_child.py")


def _execution_route_policy(
    *aliases: str,
    deterministic_aliases: tuple[str, ...] = (),
    plan_root_cid: str = "plan:test-execution-route",
    repository_tree_id: str = "tree:test-execution-route",
) -> TaskExecutionRoutePolicy:
    tasks = tuple(
        TaskRecord(
            task_cid=f"task:test-route:{index}",
            task_alias=alias,
            goal_cid="goal:test-route",
            plan_cid=plan_root_cid,
            ordinal=index,
            status="ready",
            revision=1,
        )
        for index, alias in enumerate(aliases)
    )
    snapshot = TaskSourceSnapshot(
        source_schema="test-task-source@1",
        schema_version=1,
        plan_root_cid=plan_root_cid,
        repository_tree_id=repository_tree_id,
        projection_cid="projection:test-execution-route",
        formal_plan_id=plan_root_cid,
        source_identity="source:test-execution-route",
        revision=1,
        event_cursor=0,
        goal_count=1,
        task_count=len(tasks),
        dependency_count=0,
        terminal=False,
        objective_count=1,
        plan_count=1,
    )
    deterministic = set(deterministic_aliases)
    return TaskExecutionRoutePolicy.seal(
        snapshot=snapshot,
        tasks=tasks,
        execution_modes={
            task.task_alias: (
                DETERMINISTIC_ONLY_EXECUTION_MODE
                if task.task_alias in deterministic
                else GROK_CODEX_EXECUTION_MODE
            )
            for task in tasks
        },
    )


def _operator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("casf_admitted_operator", OPERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _eventually(path: Path, process: subprocess.Popen[bytes]) -> dict[str, Any]:
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(
                "executor helper exited before readiness: "
                f"returncode={process.returncode}, "
                f"stdout={stdout.decode(errors='replace')!r}, "
                f"stderr={stderr.decode(errors='replace')!r}"
            )
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.025)
    pytest.fail("executor helper did not publish readiness")


def test_admitted_plan_uses_configured_builder_and_env_only_handle() -> None:
    operator = _operator()
    board, _config = operator._load_config(CONFIG)
    plan = operator._launch_plan(board, admit_task_execution=True)
    command = operator._executor_command(
        board,
        operator._runtime_paths(board),
        bootstrap_descriptor=17,
    )
    environment = operator._executor_environment(
        board,
        plan["provider_route_preflight"],
        owner_identity={"generation": 3, "schema_revision": 3},
    )

    assert plan["task_execution_admitted"] is True
    assert plan["maximum_active_subagents"] == 1
    assert "--implement" in command
    assert "--state-owner-bootstrap-fd" in command
    assert "--endpoint-secret-handle" not in command
    assert board.resolved_database_program().endpoint_secret_handle not in command
    assert environment[
        "IPFS_ACCELERATE_AGENT_STATE_ENDPOINT_SECRET_HANDLE"
    ] == "handle:casf-v1"
    assert operator.STATE_TOKEN_ENV not in environment
    assert operator.STATE_OWNER_SOCKET_ENV not in environment
    assert plan["execution_route_expected_counts"] == {
        "task_count": 44,
        "deterministic_task_count": 31,
        "model_task_count": 13,
    }


def test_execution_route_policy_fails_closed_on_population_and_task_drift() -> None:
    task = TaskRecord(
        task_cid="task:test-route:0",
        task_alias="CASF-DRIFT",
        goal_cid="goal:test-route",
        plan_cid="plan:test-execution-route",
        ordinal=0,
        status="ready",
        revision=1,
    )
    snapshot = TaskSourceSnapshot(
        source_schema="test-task-source@1",
        schema_version=1,
        plan_root_cid=task.plan_cid,
        repository_tree_id="tree:test-execution-route",
        projection_cid="projection:test-execution-route",
        formal_plan_id=task.plan_cid,
        source_identity="source:test-execution-route",
        revision=1,
        event_cursor=0,
        goal_count=1,
        task_count=1,
        dependency_count=0,
        terminal=False,
        objective_count=1,
        plan_count=1,
    )
    with pytest.raises(TaskSourceIntegrityError, match="exact task population"):
        TaskExecutionRoutePolicy.seal(
            snapshot=snapshot,
            tasks=(task,),
            execution_modes={},
        )
    with pytest.raises(TaskSourceIntegrityError, match="exact task population"):
        TaskExecutionRoutePolicy.seal(
            snapshot=snapshot,
            tasks=(task,),
            execution_modes={
                task.task_alias: DETERMINISTIC_ONLY_EXECUTION_MODE,
                "CASF-EXTRA": GROK_CODEX_EXECUTION_MODE,
            },
        )

    policy = TaskExecutionRoutePolicy.seal(
        snapshot=snapshot,
        tasks=(task,),
        execution_modes={task.task_alias: DETERMINISTIC_ONLY_EXECUTION_MODE},
    )
    binding = policy.binding_for_task(task).to_dict()
    summary = policy.public_summary()
    assert summary["task_count"] == 1
    assert summary["deterministic_task_count"] == 1
    assert summary["model_task_count"] == 0
    with pytest.raises(TaskSourceIntegrityError):
        policy.binding_for_task(replace(task, revision=2))
    with pytest.raises(TaskSourceIntegrityError):
        policy.binding_for_task(replace(task, body={"description": "drift"}))
    with pytest.raises(TaskSourceIntegrityError):
        policy.binding_for_task(replace(task, plan_cid="plan:other"))

    adapter = object.__new__(TypedDatabaseTaskSource)
    adapter._execution_route_policy = policy  # type: ignore[attr-defined]
    adapter._require_execution_route_plan_root = lambda: policy  # type: ignore[method-assign]
    carried = replace(
        task,
        revision=2,
        status="retrying",
        body={
            "completion_receipt": {
                "operation": "database_portal_validation_retry",
                "execution_route_binding": binding,
                "execution_route_policy_id": policy.policy_id,
                "execution_route_origin_revision": task.revision,
            }
        },
    )
    assert dict(adapter.execution_route_binding_for_task(carried)) == binding
    with pytest.raises(TaskSourceIntegrityError, match="carried"):
        adapter.execution_route_binding_for_task(
            replace(carried, body={"completion_receipt": {}})
        )


def test_status_cas_preserves_execution_route_across_requeue() -> None:
    policy = _execution_route_policy(
        "CASF-REQUEUE",
        deterministic_aliases=("CASF-REQUEUE",),
    )
    task = TaskRecord(
        task_cid="task:test-route:0",
        task_alias="CASF-REQUEUE",
        goal_cid="goal:test-route",
        plan_cid="plan:test-execution-route",
        ordinal=0,
        status="in_progress",
        revision=2,
        body={},
    )
    binding = policy.entries[0]
    route = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "task-execution-route-binding@1"
        ),
        "policy_id": policy.policy_id,
        "plan_root_cid": policy.plan_root_cid,
        "repository_tree_id": policy.repository_tree_id,
        "source_revision": policy.source_revision,
        **binding.to_dict(),
    }
    task = replace(
        task,
        body={
            "completion_receipt": {
                "operation": "database_claim",
                "execution_route_binding": route,
                "execution_route_policy_id": policy.policy_id,
                "execution_route_origin_revision": 1,
            }
        },
    )
    observed: dict[str, Any] = {}

    class _Source:
        @staticmethod
        def get(_task_cid: str) -> TaskRecord:
            return task

        @staticmethod
        def validate_execution_route_binding(
            value: Mapping[str, Any],
            *,
            task: TaskRecord,
            allow_claim_revision: bool,
        ) -> Mapping[str, Any]:
            assert task.revision == 2
            assert allow_claim_revision is True
            assert dict(value) == route
            return route

        @staticmethod
        def compare_and_set_status(
            task_cid: str,
            *,
            expected_revision: int,
            status: str,
            receipt: Mapping[str, Any],
            evidence_digests: Any,
        ) -> dict[str, Any]:
            observed.update(
                {
                    "task_cid": task_cid,
                    "expected_revision": expected_revision,
                    "status": status,
                    "receipt": dict(receipt),
                    "evidence_digests": evidence_digests,
                }
            )
            return observed

    daemon = SimpleNamespace(task_source=_Source(), markdown_path=None)
    result = DatabaseImplementationDaemon._cas_task_status_database(
        daemon,
        task.task_cid,
        expected_revision=task.revision,
        new_status="retrying",
        receipt={"operation": "database_portal_validation_retry"},
    )

    assert result["receipt"]["execution_route_binding"] == route
    assert result["receipt"]["execution_route_policy_id"] == policy.policy_id
    assert result["receipt"]["execution_route_origin_revision"] == 1


def test_admitted_health_requires_exact_live_executor_and_authoritative_progress() -> None:
    operator = _operator()
    from test.api.causal_federation.test_operator import (
        _first_tranche_authority,
        _first_tranche_runtime,
    )

    runtime = _first_tranche_runtime(
        runtime_updates={
            "task_execution_admitted": True,
            "executor": {
                "available": True,
                "supervisor_process_bound": True,
                "executor_process_bound": True,
                "supervisor_liveness": "alive",
                "executor_liveness": "alive",
                "status_fresh": True,
                "clean_error_state": True,
                "task_state": {},
            },
        }
    )
    runtime["outbox_worker"]["watermark"] = 23
    runtime["outbox_worker"]["committed_sequence"] = 23
    progressing_authority = _first_tranche_authority()
    progressing_authority.update({"event_cursor": 22, "active_count": 1})
    result = operator.classify_health(
        owner_liveness="alive",
        master_liveness="alive",
        task_authority=progressing_authority,
        runtime=runtime,
        baseline={"event_cursor": 20, "completed_count": 12},
        within_startup_grace=False,
    )
    assert result["classification"] == "progressing"
    assert result["healthy"] is True

    runtime["executor"]["clean_error_state"] = False
    unhealthy = operator.classify_health(
        owner_liveness="alive",
        master_liveness="alive",
        task_authority=_first_tranche_authority(),
        runtime=runtime,
        baseline={"event_cursor": 20, "completed_count": 12},
        within_startup_grace=False,
    )
    assert unhealthy["classification"] == "stuck"
    assert unhealthy["healthy"] is False


def test_runtime_projection_read_retries_only_a_bounded_parse_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    projection = tmp_path / "supervisor-status.json"
    projection.write_text("{}", encoding="utf-8")
    attempts = 0

    def _flaky_read(_path: Path) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise operator.OperatorError("transient partial JSON")
        return {"status": "running"}

    monkeypatch.setattr(operator, "_json_object", _flaky_read)

    assert operator._read_optional_json(
        projection, transient_retry_attempts=3
    ) == {"status": "running"}
    assert attempts == 3


def test_typed_database_task_source_reads_claims_and_records_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "typed-control.duckdb"
    source = DatabaseTaskSource(database)
    source.materialize(
        {
            "repository_tree_id": "tree:typed-executor",
            "plan_root_cid": "plan:typed-executor",
            "goals": [
                {
                    "goal_cid": "goal:typed-executor",
                    "goal_alias": "CASF-G-TYPED",
                    "title": "Typed executor",
                }
            ],
            "tasks": [
                {
                    "task_cid": "task:typed-executor",
                    "task_id": "CASF-TYPED",
                    "goal_cid": "goal:typed-executor",
                    "status": "ready",
                    "body": {"No-change completion": "allowed"},
                    "outputs": [{"path": "pyproject.toml", "effect": {}}],
                    "validations": [{"argv": ["/usr/bin/true"], "policy": {}}],
                }
            ],
        }
    )
    source.close()
    server = build_server(
        database_path=database,
        state_dir=tmp_path / "typed-owner",
        repository_id="repository:ipfs_accelerate_py",
        store_id="casf-typed-executor-v1",
        transport=FakeQuackTransport(),
        capability_probe=_capability,
        migrate=_migrate,
        connection_factory=open_duckdb_connection,
        owner_liveness_probe=lambda _birth: OwnerLiveness.DEAD,
    )
    identity = server.start()
    client_id = "database-implementation-daemon:typed-test"
    operator_birth_id = identity.process_birth_id
    token, grant = server.issue_typed_client_grant_record(
        client_id=client_id,
        process_birth_id=operator_birth_id,
        allowed_operations=tuple(build_control_plane_operation_catalog()),
        allowed_command_operations=(
            "task.status.cas.receipt",
            "task.validation.record.passed",
            "task.validation.record.nonpassing",
        ),
        peer_pid=os.getpid(),
    )
    monkeypatch.setenv(
        TYPED_STATE_OWNER_SOCKET_ENV, str(server.typed_command_socket_path())
    )
    monkeypatch.setenv(TYPED_STATE_OWNER_TOKEN_ENV, token)
    client = QuackStateClient(
        owner_id=client_id,
        store_id=identity.store_id,
        process_birth_id=operator_birth_id,
    )
    adapter: TypedDatabaseTaskSource | None = None
    try:
        client.attach(identity.listen_uri, server_id=identity.server_id)
        adapter = TypedDatabaseTaskSource(client)
        borrowed_projection = TypedDatabaseTaskSource(
            client,
            owns_client=False,
        )
        borrowed_projection.close()
        snapshot = adapter.snapshot()
        assert snapshot.task_count == 1
        assert snapshot.repository_tree_id == "tree:typed-executor"
        assert adapter.ready_tasks().tasks[0].task_alias == "CASF-TYPED"
        ready = adapter.get("CASF-TYPED")
        assert ready is not None
        with pytest.raises(TaskSourceIntegrityError, match="closed typed vocabulary"):
            adapter.compare_and_set_status(
                ready.task_cid,
                ready.revision,
                "invented-status",
            )
        with pytest.raises(TransactionError, match="authorization_denied"):
            client.cas_task_status(
                task_cid=ready.task_cid,
                expected_task_revision=ready.revision,
                new_status="invented-status",
                idempotency_key="executor-cas:unknown-status",
                body={"operation": "unknown-status-negative-test"},
            )
        ready = adapter.get("CASF-TYPED")
        assert ready is not None and ready.revision == 1
        with pytest.raises(TransactionError, match="authorization_denied"):
            adapter.compare_and_set_status(
                ready.task_cid,
                ready.revision,
                "in_progress",
                {"operation": "database_claim"},
            )
        with pytest.raises(TransactionError, match="authorization_denied"):
            adapter.compare_and_set_status(
                ready.task_cid,
                ready.revision,
                "in_progress",
                {
                    "operation": "database_claim",
                    "claim_phase_schema": "foreign-claim-phase@1",
                },
            )
        with pytest.raises(TransactionError, match="authorization_denied"):
            client.cas_task_status(
                task_cid=ready.task_cid,
                expected_task_revision=ready.revision,
                new_status="in_progress",
                idempotency_key="executor-cas:receiptless-in-progress",
            )
        assert adapter.get(ready.task_cid).revision == ready.revision
        claim_receipt = {
            "operation": "database_claim",
            "claim_phase_schema": TYPED_DATABASE_CLAIM_RESERVATION_SCHEMA,
            "claim_process_attestation": dict(
                adapter.claim_process_attestation()
            ),
            "claim_id": "claim:typed-test",
            "attempt_id": "attempt:typed-test",
            "attempt_number": 1,
            "lease_id": "lease:typed-test",
            "owner_session_id": "session:typed-test",
            "fencing_token": 1,
            "fence_epoch": 1,
            "claimed_from_revision": ready.revision,
        }
        claimed = adapter.compare_and_set_status(
            ready.task_cid,
            ready.revision,
            "in_progress",
            claim_receipt,
        )
        assert claimed.changed is True
        assert claimed.task.body["completion_receipt"]["operation"] == "database_claim"
        admitted_receipt = {
            **claim_receipt,
            "operation": "database_attempt_admitted",
            "claim_phase_schema": TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA,
            "admitted_from_revision": claimed.task.revision,
            "attempt_execution_phase": "claimed",
            "attempt_execution_revision": 1,
        }
        claimed = adapter.compare_and_set_status(
            claimed.task.task_cid,
            claimed.task.revision,
            "in_progress",
            admitted_receipt,
        )
        assert claimed.task.body["completion_receipt"]["operation"] == (
            "database_attempt_admitted"
        )
        evidence_digest = "sha256:" + ("a" * 64)
        validation = adapter.record_validation_result(
            task_cid=claimed.task.task_cid,
            outcome="passed",
            evidence_digest=evidence_digest,
            argv=["/usr/bin/true"],
            attempt_id="attempt:typed-test",
            body={"status": "passed"},
        )
        assert validation.changed is True
        completed = adapter.compare_and_set_status(
            claimed.task.task_cid,
            claimed.task.revision,
            "completed",
            {"operation": "database_complete", "evidence_digest": evidence_digest},
            evidence_digests=[evidence_digest],
        )
        assert completed.task.status == "completed"
        assert adapter.snapshot().terminal is True

        with pytest.raises(TypedStateOwnerAuthorizationError):
            TypedStateOwnerConnection(
                socket_path=server.typed_command_socket_path(),
                token="",
                client_id=client_id,
                process_birth_id=operator_birth_id,
                store_id=identity.store_id,
            )
        with pytest.raises(TypedStateOwnerError):
            TypedStateOwnerConnection(
                socket_path=server.typed_command_socket_path(),
                token=token,
                client_id=client_id,
                process_birth_id="birth:wrong",
                store_id=identity.store_id,
            )
    finally:
        if adapter is not None:
            adapter.close()
        else:
            client.close()
        server.revoke_typed_client_grant(grant.grant_id)
        with pytest.raises(TypedStateOwnerError):
            TypedStateOwnerConnection(
                socket_path=server.typed_command_socket_path(),
                token=token,
                client_id=client_id,
                process_birth_id=operator_birth_id,
                store_id=identity.store_id,
            )
        server.stop()

    connection = open_duckdb_connection(database)
    try:
        assert connection.execute("SELECT COUNT(*) FROM validation_runs").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM validation_results").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM evidence_nodes").fetchone()[0] == 1
    finally:
        connection.close()


def test_typed_daemon_promotes_local_attempt_before_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "typed-claim-barrier.duckdb"
    source = DatabaseTaskSource(database)
    source.materialize(
        {
            "repository_tree_id": "tree:typed-claim-barrier",
            "plan_root_cid": "plan:typed-claim-barrier",
            "goals": [
                {
                    "goal_cid": "goal:typed-claim-barrier",
                    "goal_alias": "CASF-G-TYPED-CLAIM-BARRIER",
                    "title": "Typed claim barrier",
                }
            ],
            "tasks": [
                {
                    "task_cid": "task:typed-claim-barrier",
                    "task_id": "CASF-TYPED-CLAIM-BARRIER",
                    "goal_cid": "goal:typed-claim-barrier",
                    "status": "ready",
                }
            ],
        }
    )
    source.close()
    server = build_server(
        database_path=database,
        state_dir=tmp_path / "typed-claim-barrier-owner",
        repository_id="repository:ipfs_accelerate_py",
        store_id="casf-typed-claim-barrier-v1",
        transport=FakeQuackTransport(),
        capability_probe=_capability,
        migrate=_migrate,
        connection_factory=open_duckdb_connection,
        owner_liveness_probe=lambda _birth: OwnerLiveness.DEAD,
    )
    identity = server.start()
    operator = _operator()
    client_id = "database-implementation-daemon:typed-claim-barrier"
    token, grant = server.issue_typed_client_grant_record(
        client_id=client_id,
        process_birth_id=identity.process_birth_id,
        allowed_operations=tuple(
            sorted(operator.EXECUTOR_OWNER_ALLOWED_OPERATIONS)
        ),
        allowed_command_operations=tuple(
            sorted(operator.EXECUTOR_OWNER_COMMAND_OPERATIONS)
        ),
        peer_pid=os.getpid(),
    )
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
    adapter: TypedDatabaseTaskSource | None = None
    daemon: DatabaseImplementationDaemon | None = None
    try:
        client.attach(identity.listen_uri, server_id=identity.server_id)
        unsealed = TypedDatabaseTaskSource(client, owns_client=False)
        route_policy = unsealed.seal_execution_route_policy(
            {
                "CASF-TYPED-CLAIM-BARRIER": (
                    DETERMINISTIC_ONLY_EXECUTION_MODE
                )
            }
        )
        unsealed.close()
        adapter = TypedDatabaseTaskSource(
            client,
            execution_route_policy=route_policy,
        )
        daemon = DatabaseImplementationDaemon(
            database_path=database,
            coordination_path=tmp_path / "typed-claim-barrier-coordination.duckdb",
            execution_path=tmp_path / "typed-claim-barrier-execution.duckdb",
            owner_session_id="session:typed-claim-barrier",
            authority_mode="quack",
            task_source_kind="duckdb",
            quack_uri=identity.listen_uri,
            task_source=adapter,
            close_task_source=False,
            lease_ms=5_000,
            clock_ms=lambda: 1_000,
            provider_fn=lambda _attempt: {"status": "ok", "accepted": True},
            effect_fn=lambda _attempt, _provider: {"status": "applied"},
            validation_fn=lambda _attempt, _effect: {
                "outcome": "passed",
                "evidence_digest": "sha256:" + "a" * 64,
            },
            require_real_execution=True,
        ).open()
        promote = daemon._promote_typed_attempt_admission

        def fail_before_admission(*_args: Any, **_kwargs: Any) -> None:
            raise DatabaseImplementationAuthorityError(
                "simulated crash after local attempt insert"
            )

        monkeypatch.setattr(
            daemon,
            "_promote_typed_attempt_admission",
            fail_before_admission,
        )
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="simulated crash",
        ):
            daemon.claim_next()
        running = daemon.list_running_attempts()
        assert len(running) == 1
        reservation = adapter.get(running[0].task_cid)
        assert reservation is not None
        assert reservation.body["completion_receipt"]["operation"] == (
            "database_claim"
        )
        assert daemon._shared_claim_binding_for_this_owner(reservation) == {
            "claim_id": running[0].claim_id,
            "attempt_id": running[0].attempt_id,
            "lease_id": running[0].lease_id,
            "owner_session_id": running[0].owner_session_id,
            "fencing_token": running[0].fencing_token,
            "fence_epoch": running[0].fence_epoch,
            "attempt_number": running[0].attempt_number,
            "operation": "database_claim",
            "claim_phase_schema": TYPED_DATABASE_CLAIM_RESERVATION_SCHEMA,
        }
        provider_calls: list[str] = []
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="simulated crash",
        ):
            daemon.run_provider(
                running[0],
                provider_fn=lambda attempt: provider_calls.append(
                    attempt.attempt_id
                )
                or {"status": "ok", "accepted": True},
            )
        assert provider_calls == []

        monkeypatch.setattr(
            daemon,
            "_promote_typed_attempt_admission",
            promote,
        )
        context_attempt = daemon.commit_phase(running[0], "context")
        _updated, _result, duplicated = daemon.run_provider(
            context_attempt,
            provider_fn=lambda attempt: provider_calls.append(
                attempt.attempt_id
            )
            or {"status": "ok", "accepted": True},
        )
        assert duplicated is False
        assert provider_calls == [running[0].attempt_id]
        admitted = adapter.get(running[0].task_cid)
        assert admitted is not None
        admitted_receipt = admitted.body["completion_receipt"]
        assert admitted_receipt["operation"] == "database_attempt_admitted"
        assert admitted_receipt["claim_phase_schema"] == (
            TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA
        )
        assert admitted_receipt["attempt_id"] == running[0].attempt_id
        assert admitted_receipt["attempt_execution_phase"] == ATTEMPT_PHASE_CLAIMED
        assert admitted_receipt["attempt_execution_revision"] == 1
        assert context_attempt.revision > running[0].revision
    finally:
        if daemon is not None:
            daemon.close()
        if adapter is not None:
            adapter.close()
        else:
            client.close()
        server.revoke_typed_client_grant(grant.grant_id)
        server.stop()


def test_typed_retry_cooldown_is_claim_bound_replay_safe_and_deadline_gated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "typed-retry-cooldown.duckdb"
    clock = {"now_ms": 1_000}
    retry_claim = {
        "attempt_id": "attempt:typed-retry-recovery",
        "claim_id": "claim:typed-retry-recovery",
        "lease_id": "lease:typed-retry-recovery",
        "owner_session_id": "session:typed-retry-recovery",
        "attempt_number": 1,
        "fencing_token": 19,
        "fence_epoch": 7,
    }
    retry_reason = (
        "database_portal_retry:attempt:typed-retry-recovery:typed_deferral"
    )
    blocked_claim = {
        "attempt_id": "attempt:typed-blocked-recovery",
        "claim_id": "claim:typed-blocked-recovery",
        "lease_id": "lease:typed-blocked-recovery",
        "owner_session_id": "session:typed-blocked-recovery",
        "attempt_number": 1,
        "fencing_token": 29,
        "fence_epoch": 13,
    }
    post_merge_claim = {
        "attempt_id": "attempt:typed-post-merge-recovery",
        "claim_id": "claim:typed-post-merge-recovery",
        "lease_id": "lease:typed-post-merge-recovery",
        "owner_session_id": "session:typed-post-merge-recovery",
        "attempt_number": 2,
        "fencing_token": 37,
        "fence_epoch": 17,
    }
    bool_claim = {
        "attempt_id": "attempt:typed-bool-receipt",
        "claim_id": "claim:typed-bool-receipt",
        "lease_id": "lease:typed-bool-receipt",
        "owner_session_id": "session:typed-bool-receipt",
        "attempt_number": True,
        "fencing_token": 41,
        "fence_epoch": 19,
    }
    legacy_claim = {
        "attempt_id": "attempt:typed-legacy-receipt",
        "claim_id": "claim:typed-legacy-receipt",
        "lease_id": "lease:typed-legacy-receipt",
        "owner_session_id": "session:typed-legacy-receipt",
        "attempt_number": 1,
        "fencing_token": 43,
        "fence_epoch": 23,
    }
    source = DatabaseTaskSource(database)
    source.materialize(
        {
            "repository_tree_id": "tree:typed-retry-cooldown",
            "plan_root_cid": "plan:typed-retry-cooldown",
            "goals": [
                {
                    "goal_cid": "goal:typed-retry-cooldown",
                    "goal_alias": "CASF-G-TYPED-RETRY",
                    "title": "Typed retry cooldown",
                }
            ],
            "tasks": [
                {
                    "task_cid": "task:typed-retry-cooldown",
                    "task_id": "CASF-TYPED-RETRY",
                    "goal_cid": "goal:typed-retry-cooldown",
                    "status": "ready",
                },
                {
                    "task_cid": "task:typed-retry-recovery",
                    "task_id": "CASF-TYPED-RETRY-RECOVERY",
                    "goal_cid": "goal:typed-retry-cooldown",
                    "status": "retrying",
                    "completion_receipt": {
                        "operation": "database_portal_retry",
                        **retry_claim,
                        "queue_reason": retry_reason,
                        "backoff_ms": 6_000,
                        "retry_not_before_ms": 7_000,
                        "control_expected_revision": 0,
                    },
                },
                {
                    "task_cid": "task:typed-blocked-recovery",
                    "task_id": "CASF-TYPED-BLOCKED-RECOVERY",
                    "goal_cid": "goal:typed-retry-cooldown",
                    "status": "blocked",
                    "completion_receipt": {
                        "operation": "database_portal_terminal_failure",
                        **blocked_claim,
                        "reason": "portal_provider_failed",
                        "retryable": False,
                        "control_expected_status": "in_progress",
                        "control_expected_revision": 0,
                    },
                },
                {
                    "task_cid": "task:typed-post-merge-recovery",
                    "task_id": "CASF-TYPED-POST-MERGE-RECOVERY",
                    "goal_cid": "goal:typed-retry-cooldown",
                    "status": "blocked",
                    "completion_receipt": {
                        "operation": "database_portal_terminal_failure",
                        **post_merge_claim,
                        "reason": "post_merge_declared_outputs_missing",
                        "retryable": False,
                        "control_expected_status": "in_progress",
                        "control_expected_revision": 0,
                    },
                },
                {
                    "task_cid": "task:typed-bool-receipt",
                    "task_id": "CASF-TYPED-BOOL-RECEIPT",
                    "goal_cid": "goal:typed-retry-cooldown",
                    "status": "in_progress",
                    "completion_receipt": {
                        "operation": "database_claim",
                        **bool_claim,
                        "claimed_from_revision": 0,
                    },
                },
                {
                    "task_cid": "task:typed-legacy-receipt",
                    "task_id": "CASF-TYPED-LEGACY-RECEIPT",
                    "goal_cid": "goal:typed-retry-cooldown",
                    "status": "in_progress",
                    "completion_receipt": {
                        "operation": "database_claim",
                        **legacy_claim,
                        "claimed_from_revision": 0,
                    },
                },
            ],
        }
    )
    source.close()
    server = build_server(
        database_path=database,
        state_dir=tmp_path / "typed-retry-owner",
        repository_id="repository:ipfs_accelerate_py",
        store_id="casf-typed-retry-cooldown-v1",
        transport=FakeQuackTransport(),
        capability_probe=_capability,
        migrate=_migrate,
        connection_factory=open_duckdb_connection,
        owner_liveness_probe=lambda _birth: OwnerLiveness.DEAD,
    )
    identity = server.start()
    operator = _operator()
    client_id = "database-implementation-daemon:typed-retry-test"
    token, grant = server.issue_typed_client_grant_record(
        client_id=client_id,
        process_birth_id=identity.process_birth_id,
        allowed_operations=tuple(
            sorted(operator.EXECUTOR_OWNER_ALLOWED_OPERATIONS)
        ),
        allowed_command_operations=tuple(
            sorted(operator.EXECUTOR_OWNER_COMMAND_OPERATIONS)
        ),
        peer_pid=os.getpid(),
    )
    monkeypatch.setenv(
        TYPED_STATE_OWNER_SOCKET_ENV, str(server.typed_command_socket_path())
    )
    monkeypatch.setenv(TYPED_STATE_OWNER_TOKEN_ENV, token)
    client = QuackStateClient(
        owner_id=client_id,
        store_id=identity.store_id,
        process_birth_id=identity.process_birth_id,
        retry_policy=RetryPolicy(
            max_attempts=1,
            base_delay_seconds=0.0,
            max_delay_seconds=0.0,
            jitter_ratio=0.0,
        ),
    )
    adapter: TypedDatabaseTaskSource | None = None
    try:
        client.attach(identity.listen_uri, server_id=identity.server_id)
        adapter = TypedDatabaseTaskSource(
            client,
            clock_ms=lambda: clock["now_ms"],
        )

        bool_task = adapter.get("CASF-TYPED-BOOL-RECEIPT")
        assert bool_task is not None
        with pytest.raises(TransactionError, match="authorization_denied"):
            adapter.record_task_retry_cooldown(
                task_cid=bool_task.task_cid,
                expected_task_revision=bool_task.revision,
                expected_task_status="in_progress",
                attempt_id=bool_claim["attempt_id"],
                claim_id=bool_claim["claim_id"],
                lease_id=bool_claim["lease_id"],
                owner_session_id=bool_claim["owner_session_id"],
                attempt_number=1,
                fencing_token=bool_claim["fencing_token"],
                fence_epoch=bool_claim["fence_epoch"],
                delay_ms=1,
                reason=(
                    "database_portal_retry:attempt:typed-bool-receipt:"
                    "strict_numeric_receipt"
                ),
                now_ms=clock["now_ms"],
            )

        legacy_task = adapter.get("CASF-TYPED-LEGACY-RECEIPT")
        assert legacy_task is not None
        with pytest.raises(TransactionError, match="authorization_denied"):
            adapter.record_task_retry_cooldown(
                task_cid=legacy_task.task_cid,
                expected_task_revision=legacy_task.revision,
                expected_task_status="in_progress",
                delay_ms=1,
                reason=(
                    "database_portal_retry:attempt:typed-legacy-receipt:"
                    "missing_typed_reservation"
                ),
                now_ms=clock["now_ms"],
                **legacy_claim,
            )

        recovery = adapter.get("CASF-TYPED-RETRY-RECOVERY")
        assert recovery is not None
        assert recovery.status == "retrying"
        generation_before_rejection = client.load_generation()
        gateway = server._command_gateway
        assert gateway is not None
        validate_manifest = gateway._validate_semantic_manifest

        def reject_cooldown_post_state(
            command: Any,
            manifest: Any,
            authority: Any,
        ) -> None:
            if command.parameters.get("operation") == (
                "task.retry.cooldown.record"
            ):
                raise TypedStateOwnerAuthorizationError(
                    "forced retry cooldown post-state rejection"
                )
            validate_manifest(command, manifest, authority)

        with monkeypatch.context() as rejection:
            rejection.setattr(
                gateway,
                "_validate_semantic_manifest",
                reject_cooldown_post_state,
            )
            with pytest.raises(TransactionError, match="authorization_denied"):
                adapter.record_task_retry_cooldown(
                    task_cid=recovery.task_cid,
                    expected_task_revision=recovery.revision - 1,
                    expected_task_status="retrying",
                    delay_ms=6_000,
                    reason=retry_reason,
                    now_ms=clock["now_ms"],
                    **retry_claim,
                )
        generation_after_rejection = client.load_generation()
        assert generation_after_rejection.revision == (
            generation_before_rejection.revision
        )
        assert adapter.get_queue_entry(recovery.task_cid) is None

        recovery_receipt = adapter.record_task_retry_cooldown(
            task_cid=recovery.task_cid,
            expected_task_revision=recovery.revision - 1,
            expected_task_status="retrying",
            delay_ms=6_000,
            reason=retry_reason,
            now_ms=clock["now_ms"],
            **retry_claim,
        )
        assert recovery_receipt.changed is True
        recovery_entry = adapter.get_queue_entry(recovery.task_cid)
        assert recovery_entry is not None
        assert recovery_entry.retry_not_before_ms == 7_000

        blocked = adapter.get("CASF-TYPED-BLOCKED-RECOVERY")
        assert blocked is not None and blocked.status == "blocked"
        blocked_reason = (
            "database_portal_retry:attempt:typed-blocked-recovery:"
            "portal_candidate_retry"
        )
        blocked_receipt = adapter.record_task_retry_cooldown(
            task_cid=blocked.task_cid,
            expected_task_revision=blocked.revision,
            expected_task_status="blocked",
            delay_ms=0,
            reason=blocked_reason,
            now_ms=clock["now_ms"],
            **blocked_claim,
        )
        assert blocked_receipt.changed is True
        blocked_entry = adapter.get_queue_entry(blocked.task_cid)
        assert blocked_entry is not None
        assert blocked_entry.reason == blocked_reason

        ready = adapter.get("CASF-TYPED-RETRY")
        assert ready is not None
        claim = {
            "operation": "database_claim",
            "claim_phase_schema": TYPED_DATABASE_CLAIM_RESERVATION_SCHEMA,
            "claim_process_attestation": dict(
                adapter.claim_process_attestation()
            ),
            "attempt_id": "attempt:typed-retry-cooldown",
            "claim_id": "claim:typed-retry-cooldown",
            "lease_id": "lease:typed-retry-cooldown",
            "owner_session_id": "session:typed-retry-cooldown",
            "attempt_number": 1,
            "fencing_token": 17,
            "fence_epoch": 5,
            "claimed_from_revision": ready.revision,
        }
        claimed = adapter.compare_and_set_status(
            ready.task_cid,
            ready.revision,
            "in_progress",
            claim,
        )
        claimed = adapter.compare_and_set_status(
            claimed.task.task_cid,
            claimed.task.revision,
            "in_progress",
            {
                **claim,
                "operation": "database_attempt_admitted",
                "claim_phase_schema": TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA,
                "admitted_from_revision": claimed.task.revision,
                "attempt_execution_phase": "claimed",
                "attempt_execution_revision": 1,
            },
        )
        cooldown = {
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
        with pytest.raises(TransactionError, match="authorization_denied"):
            adapter.record_task_retry_cooldown(
                task_cid=claimed.task.task_cid,
                expected_task_revision=claimed.task.revision + 1,
                expected_task_status="in_progress",
                delay_ms=5_000,
                reason=(
                    "database_portal_retry:attempt:typed-retry-cooldown:"
                    "worktree_lifecycle_claim_exists"
                ),
                now_ms=clock["now_ms"],
                **cooldown,
            )
        assert adapter.get_queue_entry(claimed.task.task_cid) is None

        captured: dict[str, Any] = {}
        submit = client.submit_command

        def capture_submit(
            command: Any,
            *,
            apply: Any = None,
            refresh_on_conflict: bool = True,
        ) -> Any:
            captured["command"] = command
            captured["apply"] = apply
            return submit(
                command,
                apply=apply,
                refresh_on_conflict=refresh_on_conflict,
            )

        monkeypatch.setattr(client, "submit_command", capture_submit)
        queue_reason = (
            "database_portal_retry:attempt:typed-retry-cooldown:"
            "worktree_lifecycle_claim_exists"
        )
        first = adapter.record_task_retry_cooldown(
            task_cid=claimed.task.task_cid,
            expected_task_revision=claimed.task.revision,
            expected_task_status="in_progress",
            delay_ms=5_000,
            reason=queue_reason,
            now_ms=clock["now_ms"],
            **cooldown,
        )
        assert first.changed is True
        first_command = captured["command"]
        first_apply = captured["apply"]
        replay = adapter.record_task_retry_cooldown(
            task_cid=claimed.task.task_cid,
            expected_task_revision=claimed.task.revision,
            expected_task_status="in_progress",
            delay_ms=5_000,
            reason=queue_reason,
            now_ms=5_999,
            **cooldown,
        )
        assert replay.changed is False
        assert replay.event_id == first.event_id
        assert captured["command"].command_id == first_command.command_id
        assert captured["command"].idempotency_key == first_command.idempotency_key

        altered_parameters = dict(first_command.parameters)
        altered_extension = json.loads(altered_parameters["extension_json"])
        altered_extension["selection_penalty"] = 1
        altered_parameters["selection_penalty"] = 1
        altered_parameters["extension_json"] = canonical_json_bytes(
            altered_extension
        ).decode("utf-8")
        altered_parameters["resolution_cid"] = content_identity(
            {
                "typed_retry_cooldown": altered_extension,
                "started_at_ms": altered_extension["started_at_ms"],
            }
        )
        altered = replace(first_command, parameters=altered_parameters)
        with pytest.raises(TransactionError, match="authorization_denied"):
            submit(altered, apply=first_apply, refresh_on_conflict=False)

        entry = adapter.get_queue_entry(claimed.task.task_cid)
        assert entry is not None
        assert entry.attempt == 1
        assert entry.retry_not_before_ms == 6_000
        retrying = adapter.compare_and_set_status(
            claimed.task.task_cid,
            claimed.task.revision,
            "retrying",
            {
                "operation": "database_portal_retry",
                **cooldown,
                "queue_reason": queue_reason,
                "backoff_ms": 5_000,
                "retry_not_before_ms": entry.retry_not_before_ms,
                "control_expected_revision": claimed.task.revision,
            },
        )
        assert retrying.task.status == "retrying"
        assert adapter.ready_tasks().tasks == ()
        clock["now_ms"] = 5_999
        assert adapter.ready_tasks().tasks == ()
        clock["now_ms"] = 6_000
        assert tuple(task.task_alias for task in adapter.ready_tasks().tasks) == (
            "CASF-TYPED-RETRY",
        )
        clock["now_ms"] = 7_000
        assert tuple(task.task_alias for task in adapter.ready_tasks().tasks) == (
            "CASF-TYPED-RETRY",
            "CASF-TYPED-RETRY-RECOVERY",
        )

        retrying = adapter.get(claimed.task.task_cid)
        assert retrying is not None and retrying.status == "retrying"
        second_claim = {
            "operation": "database_claim",
            "claim_phase_schema": TYPED_DATABASE_CLAIM_RESERVATION_SCHEMA,
            "claim_process_attestation": dict(
                adapter.claim_process_attestation()
            ),
            "attempt_id": "attempt:typed-retry-cooldown:2",
            "claim_id": "claim:typed-retry-cooldown:2",
            "lease_id": "lease:typed-retry-cooldown:2",
            "owner_session_id": "session:typed-retry-cooldown:2",
            "attempt_number": 2,
            "fencing_token": 23,
            "fence_epoch": 11,
            "claimed_from_revision": retrying.revision,
        }
        claimed_again = adapter.compare_and_set_status(
            retrying.task_cid,
            retrying.revision,
            "in_progress",
            second_claim,
        )
        claimed_again = adapter.compare_and_set_status(
            claimed_again.task.task_cid,
            claimed_again.task.revision,
            "in_progress",
            {
                **second_claim,
                "operation": "database_attempt_admitted",
                "claim_phase_schema": TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA,
                "admitted_from_revision": claimed_again.task.revision,
                "attempt_execution_phase": "claimed",
                "attempt_execution_revision": 1,
            },
        )
        second_cooldown = {
            name: second_claim[name]
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
        second_reason = (
            "database_portal_retry:attempt:typed-retry-cooldown:2:"
            "worktree_lifecycle_claim_exists"
        )
        second = adapter.record_task_retry_cooldown(
            task_cid=claimed_again.task.task_cid,
            expected_task_revision=claimed_again.task.revision,
            expected_task_status="in_progress",
            delay_ms=2_000,
            reason=second_reason,
            now_ms=clock["now_ms"],
            **second_cooldown,
        )
        second_command = captured["command"]
        second_apply = captured["apply"]
        second_row = adapter._retry_cooldown_row(claimed_again.task.task_cid)
        assert second.changed is True
        assert second.revision == 2
        assert second_row is not None
        assert {
            "task_cid": second_row["task_cid"],
            "claim_cid": second_row["claim_cid"],
            "claimant_did": second_row["claimant_did"],
            "logical_epoch": second_row["logical_epoch"],
            "fencing_token": second_row["fencing_token"],
            "attempt": second_row["attempt"],
            "owner_session_id": second_row["owner_session_id"],
            "fence_epoch": second_row["fence_epoch"],
            "revision": second_row["revision"],
        } == {
            "task_cid": claimed_again.task.task_cid,
            "claim_cid": second_claim["claim_id"],
            "claimant_did": second_claim["owner_session_id"],
            "logical_epoch": second_claim["fence_epoch"],
            "fencing_token": second_claim["fencing_token"],
            "attempt": 2,
            "owner_session_id": second_claim["owner_session_id"],
            "fence_epoch": second_claim["fence_epoch"],
            "revision": 2,
        }
        assert second_row["extension"]["expected_queue_revision"] == 1
        assert second_row["extension"]["expected_queue_attempt"] == 1

        clock["now_ms"] = 8_000
        second_replay = adapter.record_task_retry_cooldown(
            task_cid=claimed_again.task.task_cid,
            expected_task_revision=claimed_again.task.revision,
            expected_task_status="in_progress",
            delay_ms=2_000,
            reason=second_reason,
            **second_cooldown,
        )
        assert second_replay.changed is False
        assert second_replay.revision == 2

        with pytest.raises(ValueError, match="same-attempt replay identity"):
            adapter.record_task_retry_cooldown(
                task_cid=claimed_again.task.task_cid,
                expected_task_revision=claimed_again.task.revision,
                expected_task_status="in_progress",
                delay_ms=2_000,
                reason=second_reason + ":altered",
                **second_cooldown,
            )
        with pytest.raises(ValueError, match="newer queue row"):
            adapter.record_task_retry_cooldown(
                task_cid=claimed_again.task.task_cid,
                expected_task_revision=claimed_again.task.revision,
                expected_task_status="in_progress",
                delay_ms=2_000,
                reason=queue_reason,
                **cooldown,
            )

        stale_parameters = dict(second_command.parameters)
        stale_extension = json.loads(stale_parameters["extension_json"])
        stale_reason = second_reason + ":stale-queue-revision"
        stale_extension["reason"] = stale_reason
        stale_parameters["reason"] = stale_reason
        stale_parameters["extension_json"] = canonical_json_bytes(
            stale_extension
        ).decode("utf-8")
        stale_parameters["resolution_cid"] = content_identity(
            {
                "typed_retry_cooldown": stale_extension,
                "started_at_ms": stale_extension["started_at_ms"],
            }
        )
        stale_material = {
            name: value
            for name, value in stale_parameters.items()
            if name not in {"extension_schema", "extension_json"}
        }
        stale_digest = hashlib.sha256(
            canonical_json_bytes(stale_material)
        ).hexdigest()
        stale_command = replace(
            second_command,
            command_id=f"cmd:retry-cooldown:{stale_digest}",
            idempotency_key=f"executor-retry-cooldown:{stale_digest}",
            expected_revision=client.load_generation().revision,
            parameters=stale_parameters,
        )
        stale_result = submit(
            stale_command,
            apply=second_apply,
            refresh_on_conflict=False,
        )
        assert stale_result.accepted is False
        assert stale_result.changed is False
        after_stale = adapter._retry_cooldown_row(claimed_again.task.task_cid)
        assert after_stale is not None
        assert dict(after_stale) == dict(second_row)

        retrying_again = adapter.compare_and_set_status(
            claimed_again.task.task_cid,
            claimed_again.task.revision,
            "retrying",
            {
                "operation": "database_portal_retry",
                **second_cooldown,
                "queue_reason": second_reason,
                "backoff_ms": 2_000,
                "retry_not_before_ms": 9_000,
                "control_expected_revision": claimed_again.task.revision,
            },
        )
        assert retrying_again.task.status == "retrying"
        with pytest.raises(
            TaskSourceIntegrityError,
            match="differs from the expected attempt",
        ):
            adapter.validate_retrying_task_cooldown(
                retrying_again.task.task_cid,
                expected_attempt_identity={
                    name: cooldown[name]
                    for name in (
                        "attempt_id",
                        "claim_id",
                        "lease_id",
                        "owner_session_id",
                        "attempt_number",
                        "fencing_token",
                        "fence_epoch",
                    )
                },
                expected_reason=queue_reason,
                expected_delay_ms=5_000,
            )
        exact_second_entry = adapter.validate_retrying_task_cooldown(
            retrying_again.task.task_cid,
            expected_attempt_identity=second_cooldown,
            expected_reason=second_reason,
            expected_delay_ms=2_000,
        )
        assert exact_second_entry.attempt == 2

        post_merge_task = adapter.get("CASF-TYPED-POST-MERGE-RECOVERY")
        assert post_merge_task is not None and post_merge_task.status == "blocked"
        post_merge_attempt = SimpleNamespace(
            task_cid=post_merge_task.task_cid,
            task_alias=post_merge_task.task_alias,
            committed_phase=ATTEMPT_PHASE_FAILED,
            status="failed",
            started_at_ms=500,
            finished_at_ms=900,
            revision=4,
            **post_merge_claim,
        )
        repair_receipt_body = {
            "schema": POST_MERGE_DECLARED_OUTPUT_REPAIR_SCHEMA,
            "candidate_commit": "a" * 40,
            "repair_commit": "b" * 40,
            "task_ids": [post_merge_task.task_alias],
        }
        repair_receipt_id = content_identity(repair_receipt_body)
        repair_receipt = {
            **repair_receipt_body,
            "receipt_id": repair_receipt_id,
        }
        post_merge_evidence_body = {
            "schema": DATABASE_POST_MERGE_RECOVERY_SCHEMA,
            "request_id": "request:typed-post-merge-recovery",
            "task_cid": post_merge_task.task_cid,
            "task_alias": post_merge_task.task_alias,
            "candidate_commit": "a" * 40,
            "source_attempt_id": post_merge_attempt.attempt_id,
            "source_claim_id": post_merge_attempt.claim_id,
            "source_lease_id": post_merge_attempt.lease_id,
            "source_fencing_token": post_merge_attempt.fencing_token,
            "source_fence_epoch": post_merge_attempt.fence_epoch,
            "source_binding_id": "sha256:" + "c" * 64,
            "source_projection_immutable_digest": "sha256:" + "d" * 64,
            "repair_commit": "b" * 40,
            "repair_receipt_id": repair_receipt_id,
            "repair_receipt": repair_receipt,
        }
        post_merge_evidence = {
            **post_merge_evidence_body,
            "evidence_id": (
                DatabaseImplementationDaemon._database_portal_evidence_digest(
                    post_merge_evidence_body
                )
            ),
        }

        def post_merge_control_cas(
            task_cid: str,
            *,
            expected_revision: int,
            new_status: str,
            receipt: Mapping[str, Any],
            evidence_digests: Any = None,
        ) -> Any:
            return adapter.compare_and_set_status(
                task_cid,
                expected_revision,
                new_status,
                receipt,
                evidence_digests=evidence_digests,
            )

        post_merge_daemon = SimpleNamespace(
            task_source=adapter,
            _require_execution_authority=lambda _operation: None,
            _verified_post_merge_declared_output_repair_receipt=(
                lambda value: dict(value)
            ),
            _database_portal_evidence_digest=(
                lambda value: DatabaseImplementationDaemon._database_portal_evidence_digest(
                    value
                )
            ),
            _automatic_claim_forbidden=lambda _task: False,
            _latest_failed_attempts=lambda: [post_merge_attempt],
            _post_merge_source_admitted=(
                lambda _raw, _attempt, _task: True
            ),
            _is_post_merge_declared_outputs_missing_terminal=(
                lambda _attempt, _task: True
            ),
            _reconcile_failed_attempt_coordination=(
                lambda _attempt: {
                    "attempt_id": post_merge_attempt.attempt_id,
                    "claim_id": post_merge_attempt.claim_id,
                    "attempt_number": post_merge_attempt.attempt_number,
                }
            ),
            _protect_retry_transition_authority=(
                lambda _attempt, _coordination: None
            ),
            _cas_task_status_database=post_merge_control_cas,
            _now_ms=lambda: clock["now_ms"],
        )
        post_merge_daemon._verified_post_merge_declared_output_recovery_state = (
            lambda attempt, selected_task, expected_evidence=None: (
                DatabaseImplementationDaemon._verified_post_merge_declared_output_recovery_state(
                    post_merge_daemon,
                    attempt,
                    selected_task,
                    expected_evidence=expected_evidence,
                )
            )
        )
        post_merge_result = (
            DatabaseImplementationDaemon.recover_blocked_post_merge_declared_outputs(
                post_merge_daemon,
                post_merge_evidence,
            )
        )
        assert post_merge_result["changed"] is True
        post_merge_entry = adapter.get_queue_entry(post_merge_task.task_cid)
        assert post_merge_entry is not None
        post_merge_updated = adapter.get(post_merge_task.task_cid)
        assert post_merge_updated is not None
        assert post_merge_updated.status == "retrying"
        post_merge_control_receipt = post_merge_updated.body["completion_receipt"]
        assert post_merge_control_receipt["operation"] == (
            "database_post_merge_declared_outputs_repair_recovery"
        )
        assert post_merge_control_receipt["attempt_id"] == (
            post_merge_attempt.attempt_id
        )
        assert post_merge_control_receipt["queue_reason"] == (
            post_merge_entry.reason
        )
        assert post_merge_control_receipt["retry_not_before_ms"] == (
            post_merge_entry.retry_not_before_ms
        )
        post_merge_replay = (
            DatabaseImplementationDaemon.recover_blocked_post_merge_declared_outputs(
                post_merge_daemon,
                post_merge_evidence,
            )
        )
        assert post_merge_replay["changed"] is False
        assert post_merge_replay["status"] == "retrying"

        # An exact idempotency replay must validate the durable row before
        # the owner can reproduce its prior receipt.  Simulate corruption
        # below the typed boundary and prove the replay fails closed without
        # advancing any owner generation or mutating the damaged row again.
        gateway._connection.execute(
            "UPDATE leases SET state = 'active' WHERE task_cid = ?",
            [claimed_again.task.task_cid],
        )
        corrupt_generation = client.load_generation()
        with pytest.raises(ValueError, match="prior queue state is malformed"):
            adapter.record_task_retry_cooldown(
                task_cid=claimed_again.task.task_cid,
                expected_task_revision=claimed_again.task.revision,
                expected_task_status="in_progress",
                delay_ms=2_000,
                reason=second_reason,
                now_ms=clock["now_ms"],
                **second_cooldown,
            )
        after_corrupt_replay = client.load_generation()
        assert after_corrupt_replay.revision == corrupt_generation.revision
        corrupt_row = gateway._connection.execute(
            "SELECT state, revision FROM leases WHERE task_cid = ?",
            [claimed_again.task.task_cid],
        ).fetchone()
        assert corrupt_row is not None
        assert (corrupt_row["state"], corrupt_row["revision"]) == (
            "active",
            2,
        )
    finally:
        if adapter is not None:
            adapter.close()
        else:
            client.close()
        server.revoke_typed_client_grant(grant.grant_id)
        server.stop()


def test_typed_database_task_source_pages_transport_for_public_maximum() -> None:
    task_count = 501
    rows = [
        {
            "task_cid": f"task:paged:{ordinal:04d}",
            "task_alias": f"CASF-PAGED-{ordinal:04d}",
            "goal_cid": "goal:paged",
            "plan_cid": "plan:paged",
            "objective_id": "objective:paged",
            "ordinal": ordinal,
            "status": "quarantined",
            "revision": 1,
            "priority": "normal",
            "identity_json": json.dumps({"repository_tree_id": "tree:paged"}),
            "body_json": "{}",
            "dependencies_json": "[]",
            "outputs_json": "[]",
            "acceptance_json": "[]",
            "validations_json": "[]",
        }
        for ordinal in range(task_count)
    ]

    class _PagedClient:
        def __init__(self) -> None:
            self.page_requests: list[dict[str, int]] = []

        @staticmethod
        def load_generation() -> SimpleNamespace:
            return SimpleNamespace(content_id="generation:stable", revision=7)

        def execute(
            self, operation: str, parameters: Mapping[str, Any] | None = None
        ) -> tuple[Mapping[str, Any], ...]:
            if operation == "executor_control_snapshot":
                return (
                    {
                        "objective_count": 1,
                        "goal_count": 1,
                        "plan_count": 1,
                        "task_count": task_count,
                        "dependency_count": 0,
                        "event_watermark": 0,
                        "goals_json": "[]",
                        "plans_json": "[]",
                        "tasks_json": "[]",
                    },
                )
            assert operation == "executor_task_projection_page"
            request = {key: int(value) for key, value in dict(parameters or {}).items()}
            self.page_requests.append(request)
            offset = request["offset"]
            return tuple(rows[offset : offset + request["limit"]])

    client = _PagedClient()
    adapter = object.__new__(TypedDatabaseTaskSource)
    adapter._client = client  # type: ignore[attr-defined]
    adapter._closed = False  # type: ignore[attr-defined]
    adapter.path = Path("typed-state-owner")
    adapter.database_path = adapter.path

    page = adapter.list_tasks(status="quarantined", limit=1_000)

    assert len(page.tasks) == task_count
    assert page.next_cursor == ""
    assert client.page_requests == [
        {"limit": 500, "offset": 0},
        {"limit": 1, "offset": 500},
    ]


@pytest.mark.parametrize(
    "lease_kind",
    ("legacy", "malformed_typed", "forged_typed", "valid_old", "missing"),
)
def test_typed_ready_projection_rejects_a_foreign_lease_row(
    lease_kind: str,
) -> None:
    task_row = {
        "task_cid": "task:foreign-lease",
        "task_alias": "CASF-FOREIGN-LEASE",
        "goal_cid": "goal:foreign-lease",
        "plan_cid": "plan:foreign-lease",
        "objective_id": "objective:foreign-lease",
        "ordinal": 1,
        "status": "retrying",
        "revision": 3,
        "priority": "normal",
        "identity_json": json.dumps(
            {"repository_tree_id": "tree:foreign-lease"}
        ),
        "body_json": json.dumps(
            {
                "completion_receipt": {
                    "operation": "database_portal_retry",
                    "attempt_id": "attempt:newer",
                    "claim_id": "claim:newer",
                    "lease_id": "lease:newer",
                    "owner_session_id": "owner:newer",
                    "attempt_number": 9,
                    "fencing_token": 9,
                    "fence_epoch": 9,
                    "queue_reason": "typed-backoff",
                    "backoff_ms": 98_000,
                    "retry_not_before_ms": 99_000,
                    "control_expected_revision": 2,
                }
            }
        ),
        "dependencies_json": "[]",
        "outputs_json": "[]",
        "acceptance_json": "[]",
        "validations_json": "[]",
    }
    extension_schema = (
        "ipfs_accelerate_py/agent-supervisor/queue-entry@1"
        if lease_kind == "legacy"
        else TYPED_RETRY_COOLDOWN_SCHEMA
    )
    if lease_kind == "malformed_typed":
        extension = {
            "schema": TYPED_RETRY_COOLDOWN_SCHEMA,
            "task_cid": task_row["task_cid"],
            "claim_id": "claim:legacy-queue",
            "attempt_number": 1,
            "retry_not_before_ms": 99_000,
        }
    elif lease_kind == "forged_typed":
        extension = {
            "schema": TYPED_RETRY_COOLDOWN_SCHEMA,
            "task_cid": task_row["task_cid"],
            "expected_task_revision": -9,
            "attempt_id": "",
            "claim_id": "claim:legacy-queue",
            "lease_id": "",
            "owner_session_id": "owner:legacy-queue",
            "attempt_number": 1,
            "fencing_token": 1,
            "fence_epoch": 1,
            "delay_ms": -5,
            "started_at_ms": 1_000,
            "retry_not_before_ms": 995,
            "selection_penalty": -1,
            "consecutive_failures": -1,
            "reason": "legacy-backoff",
            "expected_queue_revision": -1,
            "expected_queue_attempt": 0,
            "unexpected": "caller-selected-extension",
        }
    elif lease_kind == "valid_old":
        extension = {
            "schema": TYPED_RETRY_COOLDOWN_SCHEMA,
            "task_cid": task_row["task_cid"],
            "expected_task_revision": 2,
            "attempt_id": "attempt:legacy-queue",
            "claim_id": "claim:legacy-queue",
            "lease_id": "lease:legacy-queue",
            "owner_session_id": "owner:legacy-queue",
            "attempt_number": 1,
            "fencing_token": 1,
            "fence_epoch": 1,
            "delay_ms": 98_000,
            "started_at_ms": 1_000,
            "retry_not_before_ms": 99_000,
            "selection_penalty": 0,
            "consecutive_failures": 1,
            "reason": "typed-backoff",
            "expected_queue_revision": -1,
            "expected_queue_attempt": 0,
        }
    else:
        extension = {
            "selection_penalty": 0,
            "consecutive_failures": 1,
            "reason": "legacy-backoff",
        }
    resolution_cid = (
        content_identity(
            {
                "typed_retry_cooldown": extension,
                "started_at_ms": extension.get("started_at_ms", 1_000),
            }
        )
        if lease_kind in {"forged_typed", "valid_old"}
        else "resolution:legacy-queue"
    )
    foreign_page_row = {
        "task_cid": task_row["task_cid"],
        "claim_cid": "claim:legacy-queue",
        "resolution_cid": resolution_cid,
        "claimant_did": "owner:legacy-queue",
        "logical_epoch": 1,
        "fencing_token": 1,
        "expires_at_ms": 0,
        "attempt": 1,
        "state": "released",
        "started_at_ms": 1_000,
        "release_reason": str(extension.get("reason") or "legacy-backoff"),
        "retry_not_before_ms": int(
            extension.get("retry_not_before_ms", 99_000)
        ),
        "owner_session_id": "owner:legacy-queue",
        "fence_epoch": 1,
        "revision": 1,
        "extension_schema": extension_schema,
        "extension_json": json.dumps(extension),
    }

    class _ForeignLeaseClient:
        @staticmethod
        def load_generation() -> SimpleNamespace:
            return SimpleNamespace(content_id="generation:foreign", revision=9)

        @staticmethod
        def execute(
            operation: str,
            _parameters: Mapping[str, Any] | None = None,
        ) -> tuple[Mapping[str, Any], ...]:
            if operation == "executor_control_snapshot":
                return (
                    {
                        "objective_count": 1,
                        "goal_count": 1,
                        "plan_count": 1,
                        "task_count": 1,
                        "dependency_count": 0,
                        "event_watermark": 0,
                        "goals_json": "[]",
                        "plans_json": "[]",
                        "tasks_json": "[]",
                    },
                )
            if operation == "executor_task_projection_page":
                return (task_row,)
            if operation == "executor_retry_cooldown_page":
                return () if lease_kind == "missing" else (foreign_page_row,)
            if operation == "executor_retry_cooldown_by_task":
                return () if lease_kind == "missing" else (foreign_page_row,)
            raise AssertionError(operation)

    adapter = object.__new__(TypedDatabaseTaskSource)
    adapter._client = _ForeignLeaseClient()  # type: ignore[attr-defined]
    adapter._clock_ms = lambda: 1_000  # type: ignore[attr-defined]
    adapter._closed = False  # type: ignore[attr-defined]
    adapter.path = Path("typed-state-owner")
    adapter.database_path = adapter.path

    with pytest.raises(
        TaskSourceIntegrityError,
        match="foreign|differs|no typed cooldown",
    ):
        adapter.ready_tasks()
    if lease_kind in {"legacy", "malformed_typed", "forged_typed"}:
        with pytest.raises(TaskSourceIntegrityError, match="foreign"):
            adapter.get_queue_entry(task_row["task_cid"])
    elif lease_kind == "valid_old":
        assert adapter.get_queue_entry(task_row["task_cid"]) is not None
    else:
        assert adapter.get_queue_entry(task_row["task_cid"]) is None


def test_executor_typed_operation_catalog_is_closed_and_full_fidelity() -> None:
    catalog = build_control_plane_operation_catalog()

    assert catalog["executor_task_projection_page"].parameter_names == (
        "limit",
        "offset",
    )
    assert catalog["executor_task_projection_page"].mutation is False
    assert catalog["executor_task_projection_by_identity"].parameter_names == (
        "task_identity",
        "task_alias",
    )
    assert catalog["executor_control_snapshot"].parameter_names == ()
    assert catalog["executor_control_snapshot"].mutation is False
    assert catalog["executor_retry_cooldown_by_task"].parameter_names == (
        "task_cid",
    )
    assert catalog["executor_retry_cooldown_page"].parameter_names == (
        "limit",
        "offset",
    )
    assert catalog["executor_insert_retry_cooldown"].mutation is True
    assert catalog["executor_update_retry_cooldown"].mutation is True


@pytest.mark.timeout(10)
def test_executor_bootstrap_broker_stop_unblocks_an_accepted_peer(
    tmp_path: Path,
) -> None:
    operator = _operator()
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    address = "\0casf-broker-stop-" + os.urandom(8).hex()
    listener.bind(address)
    listener.listen(1)
    broker = operator._ExecutorBootstrapBroker(
        channel=listener,
        server=SimpleNamespace(revoke_typed_client_grant=lambda _grant: None),
        board=SimpleNamespace(),
        paths={
            "executor_history": tmp_path / "history.json",
            "executor_current": tmp_path / "current.json",
        },
        supervisor_birth=operator._process_birth(os.getpid()),
        execution_route_policy=_execution_route_policy("CASF-BROKER-STOP"),
    )
    peer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    broker.start()
    try:
        peer.connect(address)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with broker._accepted_lock:
                if broker._accepted is not None:
                    break
            time.sleep(0.01)
        else:
            pytest.fail("broker did not accept the intentionally stalled peer")

        broker.stop()

        assert broker._thread.is_alive() is False
        assert broker.active_grant_id == ""
    finally:
        peer.close()
        if broker._thread.is_alive():
            broker.stop()


def test_executor_retirement_freezes_supervisor_before_reading_latest_rotation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    supervisor_birth = {"pid": 101, "start_time_ticks": 1, "boot_id": "boot"}
    stale_executor = {"pid": 202, "start_time_ticks": 2, "boot_id": "boot"}
    latest_executor = {"pid": 303, "start_time_ticks": 3, "boot_id": "boot"}
    current_path = tmp_path / "executor-current.json"
    current_path.write_text(
        json.dumps(
            {
                "supervisor_process_birth": supervisor_birth,
                "executor_process_birth": stale_executor,
            }
        ),
        encoding="utf-8",
    )
    events: list[Any] = []

    class _Broker:
        @staticmethod
        def stop() -> None:
            events.append("broker.stop")

    def _terminate(birth: Mapping[str, Any], *, grace_seconds: float) -> str:
        events.append(("terminate", dict(birth), grace_seconds))
        if dict(birth) == supervisor_birth:
            current_path.write_text(
                json.dumps(
                    {
                        "supervisor_process_birth": supervisor_birth,
                        "executor_process_birth": latest_executor,
                    }
                ),
                encoding="utf-8",
            )
        return "terminated"

    monkeypatch.setattr(operator, "_terminate_birth", _terminate)

    cleanup, failures = operator._retire_configured_executor(
        paths={"executor_current": current_path},
        supervisor_birth=supervisor_birth,
        broker=_Broker(),
        fallback_executor_birth=stale_executor,
        grace_seconds=7.0,
    )

    assert failures == []
    assert events == [
        "broker.stop",
        ("terminate", supervisor_birth, 7.0),
        ("terminate", latest_executor, 7.0),
        ("terminate", stale_executor, 7.0),
    ]
    assert [item["role"] for item in cleanup] == [
        "executor_supervisor",
        "executor_daemon",
        "executor_daemon",
    ]


@pytest.mark.timeout(60)
def test_real_duckdb_owner_bootstrap_claim_restart_status_and_stop(
    tmp_path: Path,
) -> None:
    operator = _operator()
    database = tmp_path / "control.duckdb"
    source = DatabaseTaskSource(database)
    source.materialize(
        {
            "repository_tree_id": "tree:casf-executor-e2e",
            "plan_root_cid": "plan:casf-executor-e2e",
            "goals": [
                {
                    "goal_cid": "goal:casf-executor-e2e",
                    "goal_alias": "CASF-G-E2E",
                    "title": "Executor E2E",
                }
            ],
            "tasks": [
                {
                    "task_cid": "task:casf-executor-e2e",
                    "task_id": "CASF-E2E",
                    "goal_cid": "goal:casf-executor-e2e",
                    "status": "ready",
                }
            ],
        }
    )
    source.close()
    server = build_server(
        database_path=database,
        state_dir=tmp_path / "owner",
        repository_id="repository:ipfs_accelerate_py",
        store_id="casf-executor-e2e-v1",
        transport=FakeQuackTransport(),
        capability_probe=_capability,
        migrate=_migrate,
        connection_factory=open_duckdb_connection,
        owner_liveness_probe=lambda _birth: OwnerLiveness.DEAD,
    )
    identity = server.start()
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind("\0casf-executor-e2e-" + os.urandom(8).hex())
    listener.listen(4)
    paths = {
        "owner_socket": server.typed_command_socket_path(),
        "executor_current": tmp_path / "executor-current.json",
        "executor_history": tmp_path / "executor-history.json",
        "executor_state": tmp_path,
        "executor_supervisor_status": tmp_path / "executor-status.json",
    }
    program = SimpleNamespace(
        quack_endpoint=identity.listen_uri,
        store_generation=identity.store_id,
    )
    board = SimpleNamespace(resolved_database_program=lambda: program)
    supervisor_birth = operator._process_birth(os.getpid())
    broker = operator._ExecutorBootstrapBroker(
        channel=listener,
        server=server,
        board=board,
        paths=paths,
        supervisor_birth=supervisor_birth,
        execution_route_policy=_execution_route_policy("CASF-E2E"),
    )
    revoked: list[str] = []
    original_revoke = server.revoke_typed_client_grant

    def record_revoke(grant_id: str) -> None:
        revoked.append(grant_id)
        original_revoke(grant_id)

    server.revoke_typed_client_grant = record_revoke  # type: ignore[method-assign]
    broker.start()
    children: list[subprocess.Popen[bytes]] = []
    try:
        first_output = tmp_path / "first.json"
        first = subprocess.Popen(
            [
                sys.executable,
                str(CHILD),
                "--bootstrap-fd",
                str(listener.fileno()),
                "--client-id",
                f"database-implementation-daemon:{operator.EXECUTOR_OWNER_SESSION_ID}",
                "--store-id",
                identity.store_id,
                "--output",
                str(first_output),
                "--claim",
                "--hold-seconds",
                "30",
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=(listener.fileno(),),
            start_new_session=True,
        )
        children.append(first)
        first_result = _eventually(first_output, first)
        first_grant = broker.active_grant_id
        assert first_result["claimed"] is True
        assert first_result["provider_received_token"] is False
        assert first_result["provider_received_socket"] is False
        assert first_result["route_policy_in_argv"] is False
        assert first_result["route_policy_in_environment"] is False
        assert first_result["granted_operations"] == sorted(
            operator.EXECUTOR_OWNER_ALLOWED_OPERATIONS
        )
        assert first_result["granted_command_operations"] == sorted(
            operator.EXECUTOR_OWNER_COMMAND_OPERATIONS
        )
        assert first_result["unrelated_read_denied"] is True
        assert set(first_result["granted_operations"]).isdisjoint(
            set(build_control_plane_operation_catalog())
            - operator.EXECUTOR_OWNER_ALLOWED_OPERATIONS
        )
        assert "token" not in " ".join(first.args).lower()
        assert str(server.typed_command_socket_path()) not in first.args

        first_birth = operator._process_birth(first.pid)
        assert operator._terminate_birth(first_birth, grace_seconds=5.0) in {
            "terminated",
            "killed",
        }
        first.wait(timeout=5.0)

        second_output = tmp_path / "second.json"
        second = subprocess.Popen(
            [
                sys.executable,
                str(CHILD),
                "--bootstrap-fd",
                str(listener.fileno()),
                "--client-id",
                f"database-implementation-daemon:{operator.EXECUTOR_OWNER_SESSION_ID}",
                "--store-id",
                identity.store_id,
                "--output",
                str(second_output),
                "--hold-seconds",
                "30",
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=(listener.fileno(),),
            start_new_session=True,
        )
        children.append(second)
        second_result = _eventually(second_output, second)
        assert second_result["attached"] is True
        assert broker.active_grant_id != first_grant
        assert first_grant in revoked

        second_birth = operator._process_birth(second.pid)
        paths["executor_supervisor_status"].write_text(
            json.dumps(
                {
                    "status": "running",
                    "supervisor_pid": os.getpid(),
                    "supervisor_pid_alive": True,
                    "daemon_pid": second.pid,
                    "daemon_pid_alive": True,
                    "current_status_path": str(tmp_path / "actual-task-state.json"),
                    "stalled_without_active_worker": False,
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "actual-task-state.json").write_text(
            json.dumps({"selection_idle_reason": "no_ready_tasks"}),
            encoding="utf-8",
        )
        projection = operator._executor_runtime_projection(
            paths,
            expected_supervisor_birth=supervisor_birth,
        )
        assert projection["supervisor_process_bound"] is True
        assert projection["executor_process_bound"] is True
        assert projection["clean_error_state"] is True
        assert projection["birth_rotation_count"] == 2
        assert projection["task_state_path"].endswith("actual-task-state.json")

        assert operator._terminate_birth(second_birth, grace_seconds=5.0) in {
            "terminated",
            "killed",
        }
        second.wait(timeout=5.0)
        assert operator._birth_liveness(second_birth) == "dead"
    finally:
        for process in children:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5.0)
        broker.stop()
        server.stop()


@pytest.mark.timeout(360)
def test_actual_configured_supervisor_routes_mixed_generation_without_leaks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    operator = _operator()
    board, _config = operator._load_config(CONFIG)
    repository_tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    protected_branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    protected_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    database = tmp_path / "managed-control.duckdb"
    managed_store_id = "data/casf-managed-e2e-runtime/control.duckdb"
    source = DatabaseTaskSource(database)
    source.materialize(
        {
            "repository_tree_id": repository_tree,
            "plan_root_cid": "plan:managed-mixed-route",
            "goals": [
                {
                    "goal_cid": "goal:managed-no-change",
                    "goal_alias": "CASF-G-MANAGED",
                    "title": "Managed typed mixed-route execution",
                }
            ],
            "tasks": [
                {
                    "task_cid": "task:managed-deterministic",
                    "task_id": "CASF-MANAGED-DET",
                    "goal_cid": "goal:managed-no-change",
                    "status": "ready",
                    "ordinal": 0,
                    "description": "Revalidate an output already present on target",
                    "No-change completion": "allowed",
                    "outputs": [{"path": "pyproject.toml", "effect": {}}],
                    "validations": [{"argv": ["/usr/bin/true"], "policy": {}}],
                },
                {
                    "task_cid": "task:managed-model",
                    "task_id": "CASF-MANAGED-MODEL",
                    "goal_cid": "goal:managed-no-change",
                    "status": "ready",
                    "ordinal": 1,
                    "description": "Require configured provider execution",
                    "dependencies": ["task:managed-deterministic"],
                    "outputs": [
                        {
                            "path": "data/casf-model-provider-output.txt",
                            "effect": {},
                        }
                    ],
                    "validations": [
                        {"argv": ["/usr/bin/true"], "policy": {}}
                    ],
                }
            ],
        }
    )
    source.close()
    server = build_server(
        database_path=database,
        state_dir=tmp_path / "managed-owner",
        repository_id="repository:ipfs_accelerate_py",
        store_id="casf-managed-no-change-v1",
        transport=FakeQuackTransport(),
        capability_probe=_capability,
        migrate=_migrate,
        connection_factory=open_duckdb_connection,
        owner_liveness_probe=lambda _birth: OwnerLiveness.DEAD,
    )
    identity = server.start()
    request.addfinalizer(server.stop)
    runtime_root = Path(
        tempfile.mkdtemp(prefix=".casf-managed-e2e-", dir=ROOT / "data")
    )
    temporary_branch = f"test/casf-managed-e2e-{runtime_root.name.removeprefix('.')}"
    temporary_ref = f"refs/heads/{temporary_branch}"
    subprocess.run(
        ["git", "update-ref", temporary_ref, "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    isolated_repo = runtime_root / "repository"
    subprocess.run(
        ["git", "worktree", "add", str(isolated_repo), temporary_branch],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    operator.ROOT = isolated_repo

    def _cleanup_isolated_runtime() -> None:
        listing = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        candidates: list[Path] = []
        for line in listing.stdout.splitlines():
            if not line.startswith("worktree "):
                continue
            candidate = Path(line.removeprefix("worktree ")).resolve(strict=False)
            try:
                candidate.relative_to(runtime_root.resolve())
            except ValueError:
                continue
            candidates.append(candidate)
        for candidate in sorted(
            candidates, key=lambda item: len(item.parts), reverse=True
        ):
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(candidate)],
                cwd=ROOT,
                check=False,
                capture_output=True,
            )
        shutil.rmtree(runtime_root, ignore_errors=True)
        subprocess.run(
            ["git", "update-ref", "-d", temporary_ref],
            cwd=ROOT,
            check=False,
            capture_output=True,
        )

    request.addfinalizer(_cleanup_isolated_runtime)
    relative_runtime = str(Path(managed_store_id).parent)
    runtime_paths = {
        "root": relative_runtime,
        "state": f"{relative_runtime}/state",
        "worktrees": f"{relative_runtime}/worktrees",
        "merge_queue": f"{relative_runtime}/merge-queue",
        "logs": f"{relative_runtime}/logs",
        "evidence": f"{relative_runtime}/evidence",
        "quack_owner": f"{relative_runtime}/owner",
        "generated_runtime_artifacts_are_completion_authority": False,
    }
    program = replace(
        board.resolved_database_program(),
        quack_endpoint=identity.listen_uri,
        store_id=managed_store_id,
        store_generation=identity.store_id,
        event_store_path=f"{relative_runtime}/events",
        runtime_registry_path=f"{relative_runtime}/registry",
        worktree_root=f"{relative_runtime}/worktrees",
    )
    payload = dict(board.payload)
    payload.update(
        {
            "merge_target_branch": temporary_branch,
            "runtime_paths": runtime_paths,
            "daemon_interval_seconds": 0.25,
            "check_interval_seconds": 0.25,
            "max_restarts": 2,
        }
    )
    managed_board = replace(
        board,
        config_path=isolated_repo / CONFIG.relative_to(ROOT),
        repo_root=isolated_repo,
        payload=payload,
        runtime_paths=runtime_paths,
        database_program=program,
        merge_target_branch=temporary_branch,
    )
    paths = operator._runtime_paths(managed_board)
    paths["owner_socket"] = server.typed_command_socket_path()

    observer_id = "client:casf-managed-e2e-observer"
    observer_token = server.issue_typed_client_grant(
        client_id=observer_id,
        process_birth_id=identity.process_birth_id,
        allowed_operations=tuple(build_control_plane_operation_catalog()),
        allowed_command_operations=(),
        peer_pid=os.getpid(),
    )
    monkeypatch.setenv(
        TYPED_STATE_OWNER_SOCKET_ENV, str(server.typed_command_socket_path())
    )
    monkeypatch.setenv(TYPED_STATE_OWNER_TOKEN_ENV, observer_token)
    observer_client = QuackStateClient(
        owner_id=observer_id,
        store_id=identity.store_id,
        process_birth_id=identity.process_birth_id,
    )
    observer_client.attach(identity.listen_uri, server_id=identity.server_id)
    observer = TypedDatabaseTaskSource(observer_client)
    managed_route_policy = observer.seal_execution_route_policy(
        {
            "CASF-MANAGED-DET": DETERMINISTIC_ONLY_EXECUTION_MODE,
            "CASF-MANAGED-MODEL": GROK_CODEX_EXECUTION_MODE,
        }
    )
    initial_generation = observer_client.load_generation()
    provider_command = (
        f"{shlex.quote(sys.executable)} -c "
        + shlex.quote(
            "from pathlib import Path; "
            "output = Path('data/casf-model-provider-output.txt'); "
            "output.parent.mkdir(parents=True, exist_ok=True); "
            "output.write_text('model provider dispatched\\n', encoding='utf-8')"
        )
    )
    supervisor: subprocess.Popen[Any] | None = None
    supervisor_birth: Mapping[str, Any] | None = None
    broker: Any = None
    executor_birth: Mapping[str, Any] | None = None
    try:
        supervisor, supervisor_birth, broker = operator._spawn_configured_executor(
            server=server,
            board=managed_board,
            paths=paths,
            owner_identity=identity.to_dict(),
            execution_route_policy=managed_route_policy,
            implementation_command=provider_command,
        )
        deadline = time.monotonic() + 240.0
        deterministic = None
        model = None
        while time.monotonic() < deadline:
            if supervisor.poll() is not None:
                pytest.fail(
                    "configured supervisor exited during no-change execution: "
                    + paths["executor_log"].read_text(
                        encoding="utf-8", errors="replace"
                    )[-8_000:]
                )
            deterministic = observer.get("CASF-MANAGED-DET")
            model = observer.get("CASF-MANAGED-MODEL")
            if (
                deterministic is not None
                and deterministic.status == "completed"
                and model is not None
                and model.status == "completed"
            ):
                break
            time.sleep(0.25)
        current_generation = observer_client.load_generation()
        assert current_generation.generation == initial_generation.generation
        assert current_generation.database_uuid == initial_generation.database_uuid
        assert current_generation.revision > initial_generation.revision
        current = json.loads(paths["executor_current"].read_text(encoding="utf-8"))
        executor_birth = current["executor_process_birth"]
        assert current["execution_route_policy"] == (
            managed_route_policy.public_summary()
        )
        assert current["execution_route_policy"]["task_count"] == 2
        assert current["execution_route_policy"]["deterministic_task_count"] == 1
        assert current["execution_route_policy"]["model_task_count"] == 1
        projection = operator._executor_runtime_projection(
            paths,
            expected_supervisor_birth=supervisor_birth,
        )
        assert projection["supervisor_process_bound"] is True
        assert projection["executor_process_bound"] is True
        assert projection["clean_error_state"] is True
        assert projection["task_state_path"].startswith(str(paths["executor_state"]))
        assert operator._read_optional_json(paths["executor_readiness"])[
            "broker_failed"
        ] is False
        assert "token" not in " ".join(supervisor.args).lower()
        assert str(server.typed_command_socket_path()) not in supervisor.args
        assert managed_route_policy.policy_id not in " ".join(supervisor.args)
        assert "CASF-MANAGED-DET" not in " ".join(supervisor.args)
        assert "CASF-MANAGED-MODEL" not in " ".join(supervisor.args)
        assert temporary_branch in supervisor.args
        assert (
            subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            == protected_branch
        )
        assert (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            == protected_head
        )
        diagnostics = {
            "deterministic": (
                {
                    "alias": deterministic.task_alias,
                    "status": deterministic.status,
                    "revision": deterministic.revision,
                    "completion_receipt": deterministic.body.get(
                        "completion_receipt"
                    ),
                }
                if deterministic is not None
                else None
            ),
            "model": (
                {
                    "alias": model.task_alias,
                    "status": model.status,
                    "revision": model.revision,
                    "completion_receipt": model.body.get("completion_receipt"),
                }
                if model is not None
                else None
            ),
            "executor_current": {
                "executor_process_birth": current.get("executor_process_birth"),
                "execution_route_policy": current.get("execution_route_policy"),
                "readiness_state": current.get("readiness_state"),
                "status": current.get("status"),
            },
            "supervisor_status": operator._read_optional_json(
                paths["executor_supervisor_status"]
            ),
            "readiness": operator._read_optional_json(
                paths["executor_readiness"]
            ),
            "log_tail": paths["executor_log"].read_text(
                encoding="utf-8", errors="replace"
            )[-8_000:],
        }
    finally:
        observer.close()
        if supervisor_birth is not None:
            cleanup, failures = operator._retire_configured_executor(
                paths=paths,
                supervisor_birth=supervisor_birth,
                broker=broker,
                fallback_executor_birth=executor_birth,
                grace_seconds=10.0,
            )
            assert failures == []
            assert cleanup
            assert operator._birth_liveness(supervisor_birth) == "dead"
            if executor_birth is not None:
                assert operator._birth_liveness(executor_birth) == "dead"
        elif supervisor is not None and supervisor.poll() is None:
            os.killpg(supervisor.pid, signal.SIGKILL)
            supervisor.wait(timeout=5.0)

    execution_database = (
        paths["executor_state"] / "casf_executor_database_execution.duckdb"
    )
    execution = open_duckdb_connection(execution_database)
    try:
        phase_rows = execution.execute(
            """
            SELECT
                attempts.task_alias,
                phases.phase,
                CAST(phases.body_json AS VARCHAR)
            FROM database_task_attempts AS attempts
            JOIN attempt_phases AS phases USING (attempt_id)
            ORDER BY attempts.task_alias, phases.committed_at_ms
            """
        ).fetchall()
    finally:
        execution.close()
    evidence: dict[str, list[Any]] = {}
    compact_phase_evidence: dict[str, list[Any]] = {}
    for row in phase_rows:
        alias, phase, body_json = row[0], row[1], row[2]
        decoded = json.loads(str(body_json))
        evidence.setdefault(str(alias), []).append(decoded)
        compact_phase_evidence.setdefault(str(alias), []).append(
            {
                "phase": str(phase),
                "body_json_prefix": str(body_json)[:2_000],
            }
        )
    diagnostic_text = json.dumps(
        {"runtime": diagnostics, "phase_evidence": compact_phase_evidence},
        indent=2,
        sort_keys=True,
        default=str,
    )

    assert deterministic is not None and deterministic.status == "completed", (
        diagnostic_text
    )
    assert model is not None and model.status == "completed", diagnostic_text

    def _contains_provider_disposition(value: Any, expected: bool) -> bool:
        if isinstance(value, Mapping):
            return value.get("provider_dispatched") is expected or any(
                _contains_provider_disposition(item, expected)
                for item in value.values()
            )
        if isinstance(value, list):
            return any(
                _contains_provider_disposition(item, expected) for item in value
            )
        return False

    assert any(
        _contains_provider_disposition(item, False)
        for item in evidence.get("CASF-MANAGED-DET", [])
    ), diagnostic_text
    assert not any(
        _contains_provider_disposition(item, True)
        for item in evidence.get("CASF-MANAGED-DET", [])
    ), diagnostic_text
    assert any(
        _contains_provider_disposition(item, True)
        for item in evidence.get("CASF-MANAGED-MODEL", [])
    ), diagnostic_text


def test_launch_modes_are_unambiguous_and_no_change_remains_explicit() -> None:
    operator = _operator()
    parser = operator._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "launch",
                "--allow-coordinator-only",
                "--admit-task-execution",
            ]
        )
    parsed = parser.parse_args(
        ["launch", "--admit-task-execution", "--executor-mode", "no-change"]
    )
    assert parsed.admit_task_execution is True
    assert parsed.executor_mode == "no-change"
