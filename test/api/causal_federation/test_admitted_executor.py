"""Qualification for the configured CASF admitted-executor boundary."""

from __future__ import annotations

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
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_transactions import (
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
    TYPED_STATE_OWNER_SOCKET_ENV,
    TYPED_STATE_OWNER_TOKEN_ENV,
    TypedStateOwnerAuthorizationError,
    TypedStateOwnerConnection,
    TypedStateOwnerError,
    build_control_plane_operation_catalog,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
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
        claimed = adapter.compare_and_set_status(
            ready.task_cid,
            ready.revision,
            "in_progress",
            {"operation": "database_claim", "claim_id": "claim:typed-test"},
        )
        assert claimed.changed is True
        assert claimed.task.body["completion_receipt"]["operation"] == "database_claim"
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
