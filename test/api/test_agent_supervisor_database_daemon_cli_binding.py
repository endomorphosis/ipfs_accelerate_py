"""CLI wiring preserves native board scope and binds one execution adapter."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import multi_supervisor_runner, process_security
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as runtime
from ipfs_accelerate_py.agent_supervisor.todo_daemon import eaaef_host_admitted_daemon_gateway as host_gateway


@pytest.mark.parametrize("native_host_binding", [False, True])
def test_database_cli_forwards_namespace_and_binds_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_host_binding: bool,
) -> None:
    calls = []
    captured = {}
    expected_gateway = object() if native_host_binding else None
    expected_dispatcher = object() if native_host_binding else None

    # These are CLI boundary fixtures. Real process/grant admission is covered
    # by the Quack integration suite; no provider or database is opened here.
    monkeypatch.setattr(process_security, "harden_state_authority_process", lambda: calls.append("harden"))
    monkeypatch.setattr(process_security, "capture_state_authority_credentials", lambda: calls.append("capture"))
    monkeypatch.setattr(multi_supervisor_runner, "preload_sealed_native_dependency_from_environment", lambda: calls.append("preload"))
    monkeypatch.setattr(runtime, "_IMPORTED_CONTROL_PLANE_CAPSULE", (
        SimpleNamespace(source_head="a" * 40, source_tree="b" * 40)
        if native_host_binding else None
    ))
    monkeypatch.setattr(host_gateway, "build_eaaef_host_admitted_command_gateway", lambda **_kwargs: expected_gateway)
    monkeypatch.setattr(host_gateway, "build_eaaef_host_admitted_container_dispatcher_factory", lambda **_kwargs: expected_dispatcher)
    monkeypatch.setattr(runtime, "database_program_from_daemon_namespace", lambda _args: SimpleNamespace(
        authority_mode="quack", task_source_kind="duckdb", store_id="board:test",
        quack_endpoint="quack:127.0.0.1:43123", schema_revision="test-profile@1",
    ))
    monkeypatch.setattr(runtime, "resolve_database_implementation_paths", lambda *_args, **_kwargs: {
        "database_path": tmp_path / "control.duckdb",
        "coordination_path": tmp_path / "coordination.duckdb",
        "execution_path": tmp_path / "execution.duckdb",
    })

    class Daemon:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            calls.append("construct")

        def run_once(self):
            calls.append("run")
            return {"unchanged": True, "write_count": 0}

        def close(self):
            calls.append("close")

    def bind(daemon, args, **kwargs):
        assert isinstance(daemon, Daemon)
        assert args.board_namespace == "board:exact-namespace"
        assert kwargs["external_agent_container_dispatcher_factory"] is expected_dispatcher
        calls.append("bind")

    monkeypatch.setattr(runtime, "DatabaseImplementationDaemon", Daemon)
    monkeypatch.setattr(runtime, "bind_database_portal_execution_from_args", bind)
    monkeypatch.setattr(runtime, "materialize_database_task_state_compatibility_projection", lambda *_args, **_kwargs: None)
    runtime.main([
        "--once", "--task-source-kind", "duckdb", "--authority-mode", "quack",
        "--state-dir", str(tmp_path), "--board-namespace", "board:exact-namespace",
        "--task-prefix", "TEST", "--owner-session-id", "lane:test",
        "--quack-endpoint", "quack:127.0.0.1:43123",
    ])

    assert captured["board_namespace"] == "board:exact-namespace"
    assert captured["task_prefix"] == "TEST"
    assert captured["quack_command_gateway"] is expected_gateway
    assert calls == ["harden", "capture", "preload", "construct", "bind", "run", "close"]
