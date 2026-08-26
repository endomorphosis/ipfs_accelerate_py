from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    BOARD_EXTENSION_INSTALL_POLICY_ENV,
    BOARD_EXTENSION_INSTALL_POLICY_LOAD_ONLY,
    DATABASE_PROGRAM_JSON_ENV,
    QUACK_TOKEN_FILE_ENV,
    RUNTIME_REGISTRY_PATH_ENV,
    STATE_AUTHORITY_MODE_ENV,
    STATE_ENDPOINT_SECRET_HANDLE_ENV,
    STATE_FAILOVER_POLICY_ENV,
    STATE_QUACK_MUTATION_DIR_ENV,
    TASK_SOURCE_KIND_ENV,
    DatabaseProgramConfig,
    DatabaseProgramConfigError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    LEGACY_BOARD_UNSTALL_DISABLED,
    LEGACY_BOARD_UNSTALL_POLICY_ENV,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_supervisor as supervisor_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_loop import (
    SupervisorLoop,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_runtime import (
    SUPERVISED_CHILD_IDENTITY_PATH_ENV,
    SUPERVISED_CHILD_OWNER_SCOPE_ENV,
    SupervisedChildSpec,
)

_LEGACY_AUTHORITY_ARGS = (
    "--task-source-kind",
    "legacy-markdown",
    "--authority-mode",
    "legacy_markdown",
    "--state-failover-policy",
    "fail_closed",
    "--explicit-legacy-task-source",
)


def test_supervisor_rejects_invalid_accepted_control_plane_pin(
    tmp_path: Path,
) -> None:
    args = supervisor_module.parse_args(
        [
            "--todo-path",
            str(tmp_path / "tasks.md"),
            "--state-dir",
            str(tmp_path / "state"),
            "--worktree-root",
            str(tmp_path / "worktrees"),
            "--accepted-control-plane-pin-json",
            "{}",
            *_LEGACY_AUTHORITY_ARGS,
        ]
    )

    with pytest.raises(
        ValueError,
        match="accepted control-plane pin fields are not exact",
    ):
        supervisor_module.supervisor_config_from_args(
            args,
            repo_root=tmp_path,
        )


def test_supervisor_accepts_and_forwards_explicit_legacy_authority(
    tmp_path: Path,
) -> None:
    todo_path = tmp_path / "tasks.md"
    todo_path.write_text("# Tasks\n", encoding="utf-8")
    state_dir = tmp_path / "state"
    args = supervisor_module.parse_args(
        [
            "--once",
            "--implement",
            "--todo-path",
            str(todo_path),
            "--state-dir",
            str(state_dir),
            "--worktree-root",
            str(tmp_path / "worktrees"),
            *_LEGACY_AUTHORITY_ARGS,
        ]
    )

    program = supervisor_module.database_program_from_cli_namespace(
        args,
        environ={},
    )
    assert program is not None
    assert program.task_source_kind == "legacy-markdown"
    assert program.authority_mode == "legacy_markdown"
    assert program.failover_policy == "fail_closed"
    assert program.explicit_legacy is True
    assert program.worktree_root == ""

    config = supervisor_module.supervisor_config_from_args(
        args,
        repo_root=tmp_path,
    )
    assert config.database_program == program
    command = supervisor_module.PortalImplementationSupervisor(
        config
    )._build_daemon_command()
    assert command.count("--task-source-kind") == 1
    source_index = command.index("--task-source-kind")
    assert command[source_index + 1] == "legacy-markdown"

    child_env = supervisor_module._managed_daemon_child_environment(
        database_program=config.database_program,
    )
    assert child_env[TASK_SOURCE_KIND_ENV] == "legacy-markdown"
    assert child_env[STATE_AUTHORITY_MODE_ENV] == "legacy_markdown"
    assert child_env[STATE_FAILOVER_POLICY_ENV] == "fail_closed"
    assert DATABASE_PROGRAM_JSON_ENV in child_env

    daemon_entrypoint = (
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon"
    )
    daemon_argv = command[command.index(daemon_entrypoint) + 1 :]
    daemon_args = daemon_module.parse_args(daemon_argv)
    assert daemon_args.worktree_root == tmp_path / "worktrees"
    daemon_program = daemon_module.database_program_from_daemon_namespace(
        daemon_args,
        environ=child_env,
    )
    assert daemon_program == program


def test_supervisor_round_trips_full_quack_authority_without_raw_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        LEGACY_BOARD_UNSTALL_POLICY_ENV,
        LEGACY_BOARD_UNSTALL_DISABLED,
    )
    monkeypatch.delenv("QUACK_TOKEN", raising=False)
    monkeypatch.delenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", raising=False)
    args = supervisor_module.parse_args(
        [
            "--todo-path",
            str(tmp_path / "tasks.md"),
            "--state-dir",
            str(tmp_path / "state"),
            "--worktree-root",
            str(tmp_path / "worktrees"),
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "quack",
            "--endpoint-secret-handle",
            "env://QUACK_TOKEN",
            "--quack-endpoint",
            "quack:127.0.0.1:45123",
            "--state-store-id",
            "control.duckdb",
            "--state-store-generation",
            "gen-7",
            "--state-schema-revision",
            "schema-v1",
            "--event-store-path",
            "state/events",
            "--runtime-registry-path",
            "state/registry",
            "--export-profile",
            "operator-export",
            "--state-failover-policy",
            "fail_closed",
        ]
    )
    program = supervisor_module.database_program_from_cli_namespace(
        args,
        environ={},
    )
    assert program is not None
    assert program.authority_mode == "quack"
    assert program.task_source_kind == "duckdb"
    assert program.endpoint_secret_handle == "env://QUACK_TOKEN"
    assert program.quack_endpoint == "quack:127.0.0.1:45123"
    assert program.store_id == "control.duckdb"
    assert program.store_generation == "gen-7"
    assert program.schema_revision == "schema-v1"
    assert program.worktree_root == ""

    child_env = supervisor_module._managed_daemon_child_environment(
        database_program=program,
        repo_root=tmp_path,
    )
    assert child_env[STATE_AUTHORITY_MODE_ENV] == "quack"
    assert child_env[TASK_SOURCE_KIND_ENV] == "duckdb"
    assert child_env[RUNTIME_REGISTRY_PATH_ENV] == str(
        (tmp_path / "state" / "registry").resolve()
    )
    assert child_env[STATE_QUACK_MUTATION_DIR_ENV] == str(
        (tmp_path / "state" / "registry" / "mutations").resolve()
    )
    assert (
        child_env[LEGACY_BOARD_UNSTALL_POLICY_ENV]
        == LEGACY_BOARD_UNSTALL_DISABLED
    )
    assert "QUACK_TOKEN" not in child_env



def test_managed_daemon_forwards_env_secret_handle_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QUACK_TOKEN", "admitted-parent-token")
    (tmp_path / "tasks.md").write_text("# Tasks\n", encoding="utf-8")
    (tmp_path / "state").mkdir()
    args = supervisor_module.parse_args(
        [
            "--todo-path",
            str(tmp_path / "tasks.md"),
            "--state-dir",
            str(tmp_path / "state"),
            "--worktree-root",
            str(tmp_path / "worktrees"),
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "quack",
            "--endpoint-secret-handle",
            "env://QUACK_TOKEN",
            "--quack-endpoint",
            "quack:127.0.0.1:45123",
            "--state-store-id",
            "control.duckdb",
            "--state-store-generation",
            "gen-7",
            "--state-schema-revision",
            "schema-v1",
            "--event-store-path",
            "state/events",
            "--runtime-registry-path",
            "state/registry",
            "--export-profile",
            "operator-export",
            "--state-failover-policy",
            "fail_closed",
        ]
    )
    program = supervisor_module.database_program_from_cli_namespace(
        args,
        environ={},
    )
    child_env = supervisor_module._managed_daemon_child_environment(
        database_program=program,
        repo_root=tmp_path,
    )
    assert child_env["QUACK_TOKEN"] == "admitted-parent-token"


def test_lgcvf_bootstrap_daemon_omits_root_token_and_passes_sealed_prelaunch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bootstrap listener replaces root-token inheritance at daemon birth."""

    root_token_name = "IPFS_ACCELERATE_AGENT_QUACK_TOKEN"
    marker = tmp_path / ".ephemeral-token-persistence-disabled"
    marker.write_text(
        "trusted controller keeps the Quack attach credential in memory\n",
        encoding="utf-8",
    )
    marker.chmod(0o400)
    monkeypatch.setenv(root_token_name, "owner-root-token-must-not-cross")
    monkeypatch.setenv(QUACK_TOKEN_FILE_ENV, str(marker / "unavailable"))
    monkeypatch.setenv(
        BOARD_EXTENSION_INSTALL_POLICY_ENV,
        BOARD_EXTENSION_INSTALL_POLICY_LOAD_ONLY,
    )

    program = DatabaseProgramConfig(
        authority_mode="quack",
        task_source_kind="duckdb",
        endpoint_secret_handle=f"env://{root_token_name}",
        quack_endpoint="quack:127.0.0.1:45123",
        store_id="data/lgcvf/control.duckdb",
        store_generation="lgcvf-test-v1",
        schema_revision="datasets-authoritative-operational-v1",
        event_store_path="state/events",
        runtime_registry_path="state/registry",
        export_profile="operator-export",
        failover_policy="fail_closed",
    )
    capsule_descriptor, native_descriptor = os.pipe()
    context = SimpleNamespace(
        capsule_pin_json="sealed-capsule-pin",
        capsule_descriptor=capsule_descriptor,
        admission_json="sealed-live-admission",
        native_launch_json="sealed-native-launch",
        native_descriptor=native_descriptor,
        pass_fds=(capsule_descriptor, native_descriptor),
    )
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(
        b"\0ipfs-accelerate-lgcvf-daemon-env-test-"
        + str(os.getpid()).encode("ascii")
    )
    listener.listen(1)
    supervisor = object.__new__(
        supervisor_module.PortalImplementationSupervisor
    )
    supervisor.config = SimpleNamespace(
        configured_board_live_context=context,
        database_program=program,
        repo_root=tmp_path,
        state_owner_bootstrap_fd=listener.fileno(),
        state_owner_bootstrap_store_id=program.store_id,
        database_owner_session_id="lgcvf-quack-lane-2",
    )
    command = (
        "/sealed/python",
        "-m",
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon",
        "--state-owner-bootstrap-fd",
        str(listener.fileno()),
        "--state-owner-bootstrap-store-id",
        program.store_id,
    )
    identity_path = tmp_path / "daemon.identity.json"
    owner_scope = {"lane": "lgcvf-quack-lane-2"}
    monkeypatch.setattr(
        supervisor_module,
        "verify_lgcvf_configured_board_live_context",
        lambda **_kwargs: context,
    )
    monkeypatch.setattr(
        supervisor_module.PortalImplementationSupervisor,
        "_build_daemon_command",
        lambda _self: list(command),
    )
    monkeypatch.setattr(
        supervisor_module.PortalImplementationSupervisor,
        "_managed_daemon_identity_path",
        lambda _self: identity_path,
    )
    monkeypatch.setattr(
        supervisor_module.PortalImplementationSupervisor,
        "_managed_daemon_owner_scope",
        lambda _self: owner_scope,
    )
    try:
        child_env, verified_context = (
            supervisor._lgcvf_live_managed_daemon_environment(command)
        )
        assert verified_context is context
        assert root_token_name not in child_env
        assert child_env[STATE_ENDPOINT_SECRET_HANDLE_ENV] == (
            f"env://{root_token_name}"
        )
        assert child_env[STATE_AUTHORITY_MODE_ENV] == "quack"
        assert child_env[TASK_SOURCE_KIND_ENV] == "duckdb"
        assert DATABASE_PROGRAM_JSON_ENV in child_env

        expected_env = {
            **child_env,
            SUPERVISED_CHILD_IDENTITY_PATH_ENV: str(identity_path),
            SUPERVISED_CHILD_OWNER_SCOPE_ENV: json.dumps(
                owner_scope,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        }
        child_spec = SupervisedChildSpec(
            repo_root=tmp_path,
            command=command,
            log_path=tmp_path / "daemon.log",
            child_pid_path=tmp_path / "daemon.pid",
            env=expected_env,
            inherit_environment=False,
            pass_fds=(*context.pass_fds, listener.fileno()),
        )
        # The sealed pre-Popen verifier still accepts the exact command,
        # context descriptors, bootstrap listener, and non-secret authority.
        supervisor._verify_lgcvf_live_supervisor_loop_child_launch(child_spec)
    finally:
        listener.close()
        os.close(capsule_descriptor)
        os.close(native_descriptor)


def test_direct_start_preserves_plan_bound_and_bootstrap_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The non-LGCVF daemon birth retains both accepted authority FDs."""

    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 43210

    def capture_popen(command: list[str], **kwargs: object) -> FakeProcess:
        captured["command"] = command
        captured.update(kwargs)
        return FakeProcess()

    supervisor = object.__new__(
        supervisor_module.PortalImplementationSupervisor
    )
    supervisor.config = SimpleNamespace(
        database_program=None,
        repo_root=tmp_path,
        state_dir=tmp_path,
        state_prefix="test",
        configured_board_live_context=None,
        plan_bound_dispatch=True,
        accepted_control_plane_descriptor=11,
        state_owner_bootstrap_fd=12,
    )
    supervisor.ensure_managed_daemon_pid_file = lambda: {"blocked": False}
    supervisor._build_daemon_command = lambda: ["python3", "daemon.py"]
    supervisor._write_managed_daemon_identity = lambda **_kwargs: None
    monkeypatch.setattr(
        supervisor_module,
        "_requires_eaaef_implementation_daemon_birth",
        lambda _program: False,
    )
    monkeypatch.setattr(
        supervisor_module,
        "_managed_daemon_child_environment",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(supervisor_module.subprocess, "Popen", capture_popen)

    process = supervisor._start_daemon()

    assert process.pid == 43210
    assert captured["command"] == ["python3", "daemon.py"]
    assert captured["pass_fds"] == (11, 12)
    assert captured["start_new_session"] is True


def test_direct_supervisor_round_trips_embedded_one_writer_authority(
    tmp_path: Path,
) -> None:
    todo_path = tmp_path / "tasks.md"
    todo_path.write_text("# Tasks\n", encoding="utf-8")
    state_dir = tmp_path / "state"
    store_id = "data/agent_supervisor/lgcvf-bootstrap/control.duckdb"
    args = supervisor_module.parse_args(
        [
            "--todo-path",
            str(todo_path),
            "--state-dir",
            str(state_dir),
            "--worktree-root",
            str(tmp_path / "worktrees"),
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded",
            "--state-store-id",
            store_id,
            "--state-store-generation",
            "lgcvf-bootstrap-v1",
            "--state-schema-revision",
            "datasets-authoritative-operational-v1",
            "--state-failover-policy",
            "fail_closed",
        ]
    )

    config = supervisor_module.supervisor_config_from_args(
        args,
        repo_root=tmp_path,
    )
    program = config.database_program
    assert program is not None
    assert program.authority_mode == "embedded"
    assert program.task_source_kind == "duckdb"
    assert program.store_id == store_id
    assert not program.endpoint_secret_handle
    assert not program.quack_endpoint

    supervisor = supervisor_module.PortalImplementationSupervisor(config)
    command = supervisor._build_daemon_command()
    assert command.count("--task-source-kind") == 1
    assert command[command.index("--task-source-kind") + 1] == "duckdb"
    assert command.count("--authority-mode") == 1
    assert command[command.index("--authority-mode") + 1] == "embedded"
    assert command.count("--state-store-id") == 1
    assert command[command.index("--state-store-id") + 1] == store_id
    assert command.count("--state-store-generation") == 1
    assert (
        command[command.index("--state-store-generation") + 1]
        == "lgcvf-bootstrap-v1"
    )
    assert command.count("--state-schema-revision") == 1
    assert (
        command[command.index("--state-schema-revision") + 1]
        == "datasets-authoritative-operational-v1"
    )
    assert supervisor._managed_daemon_matches_command_line(" ".join(command))

    for option, stale_value in (
        ("--task-source-kind", "markdown"),
        ("--authority-mode", "embedded_exclusive"),
        ("--state-store-id", "different-control.duckdb"),
        ("--state-store-generation", "stale-generation"),
        ("--state-schema-revision", "stale-schema"),
        ("--state-failover-policy", "require_explicit_operator"),
    ):
        stale_command = list(command)
        stale_command[stale_command.index(option) + 1] = stale_value
        assert not supervisor._managed_daemon_matches_command_line(
            " ".join(stale_command)
        )

    injected_command = [
        *command,
        "--runtime-registry-path",
        "different/registry",
    ]
    assert not supervisor._managed_daemon_matches_command_line(
        " ".join(injected_command)
    )

    authority_env = supervisor_module._managed_daemon_child_environment(
        database_program=program,
    )
    loop_config = supervisor.build_supervisor_loop_config()
    child_env = dict(loop_config.child_env)
    assert {
        key: value
        for key, value in child_env.items()
        if key
        not in {
            SUPERVISED_CHILD_IDENTITY_PATH_ENV,
            SUPERVISED_CHILD_OWNER_SCOPE_ENV,
        }
    } == authority_env
    assert child_env[SUPERVISED_CHILD_IDENTITY_PATH_ENV] == str(
        supervisor._managed_daemon_identity_path()
    )
    assert json.loads(child_env[SUPERVISED_CHILD_OWNER_SCOPE_ENV]) == (
        supervisor._managed_daemon_owner_scope()
    )
    assert loop_config.spec.launch_env == child_env
    loop = SupervisorLoop(loop_config)
    assert loop._child_spec("initial").env == child_env
    assert loop._child_spec("restart").env == child_env
    assert str(supervisor_module.REPO_ROOT) in child_env["PYTHONPATH"].split(os.pathsep)
    daemon_entrypoint = (
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon"
    )
    daemon_argv = command[command.index(daemon_entrypoint) + 1 :]
    daemon_args = daemon_module.parse_args(daemon_argv)
    assert daemon_module.database_program_from_daemon_namespace(
        daemon_args,
        environ=authority_env,
    ) == program
    # The immutable non-secret store authority is also reconstructable from
    # argv alone.  A missing environment binding must never reinterpret the
    # Markdown projection as the DuckDB store.
    assert daemon_module.database_program_from_daemon_namespace(
        daemon_args,
        environ={},
    ) == program

    provider_env = supervisor.provider_subprocess_environment(
        {"QUACK_TOKEN": "must-not-cross", **child_env}
    )
    assert "QUACK_TOKEN" not in provider_env
    assert DATABASE_PROGRAM_JSON_ENV not in provider_env
    assert STATE_AUTHORITY_MODE_ENV not in provider_env
    assert TASK_SOURCE_KIND_ENV not in provider_env


def test_source_change_reload_preserves_embedded_programmatic_launch_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An embedded ``main(argv)`` reload must never fall back to CLI defaults."""

    repo = tmp_path.resolve()
    todo_path = repo / "docs" / "lgcvf.todo.md"
    todo_path.parent.mkdir()
    todo_path.write_text("# LGCVF tasks\n", encoding="utf-8")
    for relative in ("policy/operator-seal.json", "policy/benchmark.json"):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    original_argv = [
        "--todo-path",
        str(todo_path),
        "--state-dir",
        str(repo / "run-v16" / "state"),
        "--state-prefix",
        "portal",
        "--task-prefix",
        "## LGCVF-",
        "--stale-seconds",
        "1800",
        "--check-interval",
        "17",
        "--max-restarts",
        "10",
        "--max-task-attempts",
        "3",
        "--daemon-interval",
        "23",
        "--implement",
        "--implementation-command",
        "python -m deterministic_provider",
        "--implementation-timeout",
        "901",
        "--implementation-max-timeout",
        "1201",
        "--validation-max-workers",
        "4",
        "--worktree-root",
        str(repo / "run-v16" / "worktrees"),
        "--worktree-submodule-path",
        "ipfs_datasets_py",
        "--implementation-protected-path",
        "policy/operator-seal.json",
        "--implementation-protected-path",
        "policy/benchmark.json",
        "--task-source-kind",
        "duckdb",
        "--authority-mode",
        "embedded",
        "--state-store-id",
        "run-v16/control.duckdb",
        "--state-store-generation",
        "lgcvf-run-v16",
        "--state-schema-revision",
        "datasets-authoritative-operational-v1",
        "--state-failover-policy",
        "fail_closed",
        "--task-shard-count",
        "1",
        "--task-shard-index",
        "0",
        "--strict-task-sharding",
    ]
    parsed = supervisor_module.parse_args(original_argv)
    config = supervisor_module.supervisor_config_from_args(
        parsed,
        repo_root=repo,
    )
    supervisor = supervisor_module.PortalImplementationSupervisor(config)
    original_child_command = supervisor._build_daemon_command()

    class ExecRequested(Exception):
        pass

    calls: list[tuple[str, list[str]]] = []

    def fake_execv(executable: str, arguments: list[str]) -> None:
        calls.append((executable, arguments))
        raise ExecRequested

    # Reproduce the failed production shape: the accepted arguments were
    # passed to main(argv), while sys.argv belonged to an embedding launcher.
    monkeypatch.setattr(supervisor_module.sys, "argv", ["embedded-launcher"])
    monkeypatch.setattr(supervisor_module.os, "execv", fake_execv)

    with pytest.raises(ExecRequested):
        supervisor._reload_for_control_plane_update()

    assert len(calls) == 1
    executable, reload_command = calls[0]
    assert executable == supervisor_module.sys.executable
    assert reload_command[:3] == [
        supervisor_module.sys.executable,
        "-m",
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor",
    ]
    assert reload_command[3:] == original_argv

    reloaded_args = supervisor_module.parse_args(reload_command[3:])
    reloaded_config = supervisor_module.supervisor_config_from_args(
        reloaded_args,
        repo_root=repo,
    )
    reloaded_child_command = supervisor_module.PortalImplementationSupervisor(
        reloaded_config
    )._build_daemon_command()

    # Exact command equality covers every forwarded policy field and repeated
    # protected path.  The focused assertions make the safety-critical values
    # visible if this regression ever fails.
    assert reloaded_child_command == original_child_command
    assert "--implement" in reloaded_child_command
    assert reloaded_child_command[
        reloaded_child_command.index("--task-prefix") + 1
    ] == "## LGCVF-"
    assert reloaded_child_command[
        reloaded_child_command.index("--max-task-attempts") + 1
    ] == "3"
    assert reloaded_child_command[
        reloaded_child_command.index("--validation-max-workers") + 1
    ] == "4"
    protected = [
        reloaded_child_command[index + 1]
        for index, token in enumerate(reloaded_child_command[:-1])
        if token == "--implementation-protected-path"
    ]
    assert protected == [
        "policy/operator-seal.json",
        "policy/benchmark.json",
    ]
    assert str(todo_path) in reloaded_child_command
    assert "docs/211_SERVICE_NAVIGATION_PORTAL_TODO.md" not in (
        reloaded_child_command
    )


def test_supervisor_rejects_inconsistent_authority_selection(
    tmp_path: Path,
) -> None:
    args = supervisor_module.parse_args(
        [
            "--todo-path",
            str(tmp_path / "tasks.md"),
            "--state-dir",
            str(tmp_path / "state"),
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "legacy_markdown",
            "--explicit-legacy-task-source",
        ]
    )
    with pytest.raises(DatabaseProgramConfigError, match="legacy_markdown"):
        supervisor_module.database_program_from_cli_namespace(
            args,
            environ={},
        )
