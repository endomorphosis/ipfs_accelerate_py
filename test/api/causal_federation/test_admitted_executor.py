"""Qualification for the configured CASF admitted-executor boundary."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
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
