"""Qualification for the configured CASF admitted-executor boundary."""

from __future__ import annotations

import importlib.util
import json
import os
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
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
    QuackStateClient,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_database_task_source import (
    TypedDatabaseTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    TYPED_STATE_OWNER_SOCKET_ENV,
    TYPED_STATE_OWNER_TOKEN_ENV,
    build_control_plane_operation_catalog,
)
from test.api.causal_federation.test_bootstrap_runtime import (
    _capability,
    _migrate,
)

ROOT = Path(__file__).resolve().parents[3]
OPERATOR_PATH = ROOT / "scripts/run_agent_supervisor_causal_event_federation.py"
CONFIG = ROOT / "config/agent_supervisor_causal_event_federation_scheduler.json"
CHILD = Path(__file__).with_name("executor_bootstrap_child.py")


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
    token = server.issue_typed_client_grant(
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
        snapshot = adapter.snapshot()
        assert snapshot.task_count == 1
        assert snapshot.repository_tree_id == "tree:typed-executor"
        assert adapter.ready_tasks().tasks[0].task_alias == "CASF-TYPED"
        ready = adapter.get("CASF-TYPED")
        assert ready is not None
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
    finally:
        if adapter is not None:
            adapter.close()
        else:
            client.close()
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


@pytest.mark.timeout(180)
def test_actual_configured_supervisor_completes_typed_no_change_task(
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
    source = DatabaseTaskSource(database)
    source.materialize(
        {
            "repository_tree_id": repository_tree,
            "plan_root_cid": "plan:managed-no-change",
            "goals": [
                {
                    "goal_cid": "goal:managed-no-change",
                    "goal_alias": "CASF-G-MANAGED",
                    "title": "Managed typed no-change execution",
                }
            ],
            "tasks": [
                {
                    "task_cid": "task:managed-no-change",
                    "task_id": "CASF-MANAGED",
                    "goal_cid": "goal:managed-no-change",
                    "status": "ready",
                    "description": "Revalidate an output already present on target",
                    "No-change completion": "allowed",
                    "outputs": [{"path": "pyproject.toml", "effect": {}}],
                    "validations": [{"argv": ["/usr/bin/true"], "policy": {}}],
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
    relative_runtime = "data/casf-managed-e2e-runtime"
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
        store_id=f"{relative_runtime}/control.duckdb",
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
            implementation_command="/usr/bin/true",
        )
        deadline = time.monotonic() + 120.0
        completed = None
        while time.monotonic() < deadline:
            if supervisor.poll() is not None:
                pytest.fail(
                    "configured supervisor exited during no-change execution: "
                    + paths["executor_log"].read_text(
                        encoding="utf-8", errors="replace"
                    )[-8_000:]
                )
            completed = observer.get("CASF-MANAGED")
            if completed is not None and completed.status == "completed":
                break
            time.sleep(0.25)
        assert completed is not None and completed.status == "completed", (
            paths["executor_log"].read_text(encoding="utf-8", errors="replace")[-8_000:]
        )
        current = json.loads(paths["executor_current"].read_text(encoding="utf-8"))
        executor_birth = current["executor_process_birth"]
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
