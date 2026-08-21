from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
OPERATOR_PATH = ROOT / "scripts/run_agent_supervisor_residual_intelligence.py"
CONFIG_PATH = ROOT / "config/agent_supervisor_residual_intelligence_scheduler.json"


def _operator():
    spec = importlib.util.spec_from_file_location("vrif_operator", OPERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bold_and_plain_board_metadata_are_equivalent() -> None:
    operator = _operator()
    assert operator._metadata_value("completed") == "completed"
    assert operator._metadata_value("** completed") == "completed"
    goals = operator._goal_blocks("## VRIF-G000 Root\n- **Status:** active\n- **Parent:**\n")
    assert goals == [("VRIF-G000", "Root", {"status": "active", "parent": ""})]


def test_runtime_config_closes_authority_and_training_fallbacks() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    projection = config["initial_projection"]
    assert projection["task_count"] == 33
    assert projection["task_dependency_count"] == 111
    assert projection["goal_count"] == 9
    assert projection["ready_task_ids"] == [
        "VRIF-009",
        "VRIF-010",
        "VRIF-011",
        "VRIF-012",
    ]
    assert config["database_program"]["authority_mode"] == "quack"
    assert config["database_program"]["task_source_kind"] == "duckdb"
    assert config["database_program"]["failover_policy"] == "fail_closed"
    assert config["ducklake_projection_program"]["authority"] is False
    assert config["training_policy"]["training_enabled_at_bootstrap"] is False
    assert config["reconciliation_guardrail_enabled"] is False


def test_owner_command_vocabulary_has_no_raw_sql_surface() -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        DuckDBConnectionPolicyError,
        validate_quack_owner_command,
    )

    assert validate_quack_owner_command(
        "record_queue_retry", {"task_cid": "task-cid"}
    ) == {"task_cid": "task-cid"}
    with pytest.raises(DuckDBConnectionPolicyError):
        validate_quack_owner_command(
            "execute_sql", {"sql": "UPDATE tasks SET status='completed'"}
        )
    with pytest.raises(DuckDBConnectionPolicyError):
        validate_quack_owner_command(
            "record_queue_retry",
            {"task_cid": "task-cid", "sql": "DELETE FROM tasks"},
        )


def test_owner_command_inbox_requires_signed_store_bound_envelope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources import database_task_source
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        QUACK_OWNER_COMMAND_REQUEST_SCHEMA,
        QUACK_OWNER_COMMAND_RESPONSE_SCHEMA,
        quack_owner_command_signature,
    )

    operator = _operator()
    request_id = "a" * 32
    token = "vrif_test_token_0123456789"
    payload = {
        "schema": QUACK_OWNER_COMMAND_REQUEST_SCHEMA,
        "request_id": request_id,
        "issued_at_ms": int(time.time() * 1000),
        "writer_identity": "supervisor-process:1234",
        "store_id": "data/agent_supervisor/residual_intelligence_foundry/control.duckdb",
        "store_generation": "vrif-v1",
        "command": "record_queue_retry",
        "payload": {"task_cid": "task-cid"},
    }
    payload["signature"] = quack_owner_command_signature(payload, token=token)
    request_path = tmp_path / f"{request_id}.request.json"
    request_path.write_text(json.dumps(payload), encoding="utf-8")

    calls: list[tuple[object, str, object, object]] = []

    def execute(
        repository: object,
        command: str,
        command_payload: object,
        **bindings: object,
    ) -> dict[str, object]:
        calls.append((repository, command, command_payload, bindings))
        return {"schema": "test-result@1", "changed": True}

    repository = object()
    monkeypatch.setattr(database_task_source, "execute_quack_owner_command", execute)
    operator._process_owner_commands(
        repository,
        tmp_path,
        token=token,
        expected_store_id=payload["store_id"],
        expected_store_generation=payload["store_generation"],
    )
    assert calls == [
        (
            repository,
            "record_queue_retry",
            {"task_cid": "task-cid"},
            {
                "request_id": request_id,
                "store_id": payload["store_id"],
                "store_generation": "vrif-v1",
            },
        )
    ]
    result = json.loads((tmp_path / f"{request_id}.done.json").read_text())
    assert result["schema"] == QUACK_OWNER_COMMAND_RESPONSE_SCHEMA
    assert result["ok"] is True
    assert result["result"] == {"schema": "test-result@1", "changed": True}


def test_owner_command_inbox_rejects_raw_sql_envelopes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources import database_task_source

    operator = _operator()
    request_id = "b" * 32
    request_path = tmp_path / f"{request_id}.request.json"
    request_path.write_text(
        json.dumps(
            {
                "sql": "UPDATE tasks SET status = 'completed'",
                "parameters": None,
            }
        ),
        encoding="utf-8",
    )

    def execute(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("raw SQL must never reach the typed dispatcher")

    monkeypatch.setattr(database_task_source, "execute_quack_owner_command", execute)
    operator._process_owner_commands(
        object(),
        tmp_path,
        token="vrif_test_token_0123456789",
        expected_store_id="store",
        expected_store_generation="generation",
    )
    result = json.loads((tmp_path / f"{request_id}.done.json").read_text())
    assert result["ok"] is False
    assert result["error_code"] == "owner_error"
    assert result["error_message"] == "typed owner command rejected"


def test_state_credential_process_denies_same_uid_environment_probe() -> None:
    script = """
import subprocess
import sys
from ipfs_accelerate_py.agent_supervisor.runtime.process_security import harden_state_authority_process
assert harden_state_authority_process() is True
probe = subprocess.run(
    [sys.executable, '-c', "import os; open(f'/proc/{os.getppid()}/environ', 'rb').read()"],
    capture_output=True,
    text=True,
)
print(probe.returncode)
print('PermissionError' in probe.stderr)
"""
    environment = dict(os.environ)
    environment["IPFS_ACCELERATE_AGENT_QUACK_TOKEN"] = "isolated_test_token_123456"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.splitlines() == ["1", "True"]


def test_real_launch_token_vault_unlink_is_fail_closed(tmp_path: Path) -> None:
    operator = _operator()
    token_path = tmp_path / "owner.quack-token"
    token_path.write_text("opaque-test-token", encoding="utf-8")
    token_path.chmod(0o600)
    operator._unlink_token_vault(token_path)
    assert not token_path.exists()

    target = tmp_path / "unsafe-target"
    target.write_text("must-remain", encoding="utf-8")
    link = tmp_path / "unsafe.quack-token"
    link.symlink_to(target)
    with pytest.raises(operator.OperatorError, match="unsafe Quack token vault"):
        operator._unlink_token_vault(link)
    assert link.is_symlink()
    assert target.read_text(encoding="utf-8") == "must-remain"


def test_provider_environment_removes_state_authority_credentials() -> None:
    from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
        provider_subprocess_environment,
    )

    cleaned = provider_subprocess_environment(
        {
            "PATH": "/usr/bin",
            "IPFS_ACCELERATE_AGENT_QUACK_TOKEN": "not-for-provider",
            "IPFS_ACCELERATE_AGENT_OWNER_STATE_TOKEN": "not-for-provider",
        }
    )
    assert cleaned == {"PATH": "/usr/bin"}


def test_implementation_provider_receives_scrubbed_parent_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        PortalImplementationDaemon,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_runtime import (
        launch_process_child,
    )

    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "not-for-implementation-provider"
    )
    monkeypatch.setenv("VRIF_TEST_PROVIDER_API_KEY", "preserved-provider-key")

    class _Daemon:
        @staticmethod
        def _canonical_ref(_task: object) -> str:
            return "task-cid:vrif-test"

    environment = PortalImplementationDaemon._implementation_process_environment(
        _Daemon(),
        SimpleNamespace(task_id="VRIF-TEST"),
        attempt=2,
        checkpoint_dir=tmp_path,
    )
    assert "IPFS_ACCELERATE_AGENT_QUACK_TOKEN" not in environment
    assert environment["VRIF_TEST_PROVIDER_API_KEY"] == "preserved-provider-key"
    assert environment["IPFS_ACCELERATE_AGENT_TASK_ID"] == "VRIF-TEST"
    child = launch_process_child(
        [
            sys.executable,
            "-c",
            "import os; print(os.environ.get('IPFS_ACCELERATE_AGENT_QUACK_TOKEN', 'absent'))",
        ],
        cwd=ROOT,
        env=environment,
        replace_env=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = child.communicate(timeout=5)
    assert child.returncode == 0, stderr
    assert stdout.strip() == "absent"


def test_auto_rescue_worktree_command_receives_no_state_credential(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        PortalImplementationDaemon,
    )

    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "not-for-auto-rescue-command"
    )

    class _Daemon:
        implementation_timeout = 5

        @staticmethod
        def _record_event(_event: str, _payload: object) -> None:
            return None

    command = (
        f'{sys.executable} -c "import os; '
        "print(os.environ.get('IPFS_ACCELERATE_AGENT_QUACK_TOKEN', 'absent'))\""
    )
    results = PortalImplementationDaemon._run_auto_rescue_materialize_commands(
        _Daemon(),
        workspace_path=tmp_path,
        log_path=tmp_path / "auto-rescue.log",
        commands=(command,),
        task=SimpleNamespace(task_id="VRIF-TEST"),
    )
    assert results[0]["ok"] is True
    assert results[0]["output_tail"].strip() == "absent"
