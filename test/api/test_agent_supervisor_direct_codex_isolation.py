"""Focused tests for the direct Codex external isolation boundary."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime import (
    configured_board_scheduler as scheduler_module,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    multi_supervisor_runner as multi_runner_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_supervisor as supervisor_module,
)

IMAGE_ID = "sha256:" + "4" * 64
CODEX_SHA256 = "7" * 64


@pytest.fixture(autouse=True)
def _isolate_ambient_trusted_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Require each test to opt into trusted control-plane runtime bindings."""

    monkeypatch.delenv(multi_runner_module.TRUSTED_DUCKDB_HOME_ENV, raising=False)
    monkeypatch.delenv(
        multi_runner_module.TRUSTED_PYTHON_USER_BASE_ENV,
        raising=False,
    )
    for name in multi_runner_module.TRUSTED_RUNTIME_CACHE_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def _isolation_payload(credential: Path) -> dict[str, object]:
    return {
        "schema": daemon_module.PROVIDER_EXTERNAL_ISOLATION_SCHEMA,
        "required": True,
        "backend": "docker",
        "provider_id": "codex",
        "runtime_executable": "/usr/bin/docker",
        "runtime_endpoint": "unix:///run/user/1000/docker.sock",
        "image_id": IMAGE_ID,
        "image_os": "linux",
        "image_architecture": "arm64",
        "image_label": "test-v1",
        "container_executable": "/usr/local/bin/codex",
        "container_executable_sha256": CODEX_SHA256,
        "credential_file": str(credential),
        "network": "bridge",
        "pids_limit": 128,
        "memory_bytes": 1024 * 1024 * 1024,
        "cpus": 1,
        "tmpfs_size_bytes": 64 * 1024 * 1024,
    }


def _credential(tmp_path: Path) -> Path:
    credential = tmp_path / "auth.json"
    credential.write_text("{}\n", encoding="utf-8")
    credential.chmod(0o600)
    return credential


def _linked_workspace(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "repository"
    common = repository / ".git"
    git_dir = common / "worktrees" / "task"
    workspace = tmp_path / "task-workspace"
    git_dir.mkdir(parents=True)
    workspace.mkdir()
    (git_dir / "commondir").write_text("../..\n", encoding="utf-8")
    (workspace / ".git").write_text(
        f"gitdir: {git_dir}\n",
        encoding="utf-8",
    )
    return repository, workspace


def _fake_host_validation(value, *, verify_host=True):
    del verify_host
    return daemon_module.ExternalProviderIsolationConfig.parse(value)


def _database_program_environment(
    repository: Path,
) -> tuple[dict[str, str], Path]:
    control_root = (
        repository
        / "state"
        / "agent_supervisor_proof_carrying_procedure_compiler"
        / "control"
    )
    control_root.mkdir(parents=True, exist_ok=True)
    program = multi_runner_module.DatabaseProgramConfig(
        authority_mode="quack",
        task_source_kind="duckdb",
        endpoint_secret_handle="env://QUACK_TOKEN",
        quack_endpoint="quack:127.0.0.1:45123",
        store_id=(
            control_root.relative_to(repository) / "control.duckdb"
        ).as_posix(),
        store_generation="17",
        schema_revision="9",
        runtime_registry_path=(
            "state/agent_supervisor_proof_carrying_procedure_compiler/registry"
        ),
        failover_policy="fail_closed",
    )
    return program.environment(repository_root=repository), control_root


def _mounts(command: list[str]) -> list[str]:
    return [
        command[index + 1]
        for index, item in enumerate(command[:-1])
        if item == "--mount"
    ]


def test_direct_codex_command_has_a_fail_closed_external_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential = _credential(tmp_path)
    repository, workspace = _linked_workspace(tmp_path)
    runtime_root = tmp_path / "runtime" / "control"
    runtime_root.mkdir(parents=True)
    config = daemon_module.ExternalProviderIsolationConfig.parse(
        _isolation_payload(credential)
    )
    monkeypatch.setattr(
        daemon_module,
        "validate_external_provider_isolation_config",
        _fake_host_validation,
    )
    monkeypatch.setenv(
        daemon_module.PROVIDER_EXTERNAL_ISOLATION_ENV,
        config.environment_json(),
    )
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "must-not-pass")
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_QUACK_ENDPOINT",
        "quack:127.0.0.1:45671",
    )

    command = daemon_module._codex_implementation_command(
        codex="/unsafe/host/codex",
        workspace_path=workspace,
        repository_root=repository,
        model_override="gpt-test",
        reasoning_effort_override="medium",
    )

    assert command[:4] == [
        "/usr/bin/env",
        "-i",
        "HOME=/nonexistent",
        "PATH=/usr/bin:/bin",
    ]
    assert command[4:8] == [
        "/usr/bin/docker",
        "--host=unix:///run/user/1000/docker.sock",
        "run",
        "--pull=never",
    ]
    assert "--rm" in command
    assert "-i" in command
    assert "--read-only" in command
    assert "--network=bridge" in command
    assert "--pid=host" not in command
    assert "--ipc=private" in command
    assert "--uts=host" not in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges:true" in command
    assert "--pids-limit=128" in command
    assert "--memory=1073741824" in command
    assert "--memory-swap=1073741824" in command
    assert "--cpus=1" in command
    assert command[command.index("--user") + 1] == "0:0"
    mounts = _mounts(command)
    writable_mounts = [
        item for item in mounts if item.endswith(",readonly=false")
    ]
    assert (
        f"type=bind,src={workspace},dst={workspace},readonly=false"
        in writable_mounts
    )
    auth_mounts = [
        item for item in mounts if "dst=/opt/codex-home/auth.json" in item
    ]
    assert len(auth_mounts) == 1
    assert str(credential) not in auth_mounts[0]
    assert auth_mounts[0].endswith(",readonly=false")
    assert not any(str(credential) in item for item in mounts)
    assert not any(str(credential) in item for item in command)
    assert (
        f"type=bind,src={repository / '.git'},"
        f"dst={repository / '.git'},readonly"
    ) in mounts
    assert (
        f"type=bind,src={workspace / '.git'},"
        f"dst={workspace / '.git'},readonly"
    ) in mounts
    assert not any(str(runtime_root) in item for item in mounts)
    vendor = daemon_module._host_codex_vendor_binaries()
    if vendor is not None:
        host_codex, host_companion = vendor
        assert (
            f"type=bind,src={host_codex},"
            "dst=/usr/local/bin/codex,readonly"
        ) in mounts
        assert (
            f"type=bind,src={host_companion},"
            "dst=/usr/local/bin/codex-code-mode-host,readonly"
        ) in mounts
    assert not any("/proc" in item for item in mounts)
    assert not any("docker.sock" in item for item in mounts)
    assert not any("/unsafe/host/codex" in item for item in command)
    image_index = command.index(IMAGE_ID)
    assert command[image_index + 1] == "-i"
    assert "--dangerously-bypass-approvals-and-sandbox" in command[image_index:]
    assert not any(
        daemon_module.PROVIDER_EXTERNAL_ISOLATION_ENV in item
        for item in command
    )
    assert not any("QUACK" in item.upper() for item in command[image_index:])
    assert not any("45671" in item for item in command[image_index:])


def test_required_isolation_never_falls_back_to_direct_codex(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential = _credential(tmp_path)
    _repository, workspace = _linked_workspace(tmp_path)
    config = daemon_module.ExternalProviderIsolationConfig.parse(
        _isolation_payload(credential)
    )
    monkeypatch.setenv(
        daemon_module.PROVIDER_EXTERNAL_ISOLATION_ENV,
        config.environment_json(),
    )

    def unavailable(_value, *, verify_host=True):
        del verify_host
        raise ValueError("Docker unavailable")

    monkeypatch.setattr(
        daemon_module,
        "validate_external_provider_isolation_config",
        unavailable,
    )
    with pytest.raises(ValueError, match="Docker unavailable"):
        daemon_module._codex_implementation_command(
            codex="/usr/local/bin/codex",
            workspace_path=workspace,
        )


def test_required_isolation_rejects_direct_command_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential = _credential(tmp_path)
    config = daemon_module.ExternalProviderIsolationConfig.parse(
        _isolation_payload(credential)
    )
    repository = tmp_path / "repository"
    workspace = repository / "worktree"
    workspace.mkdir(parents=True)
    monkeypatch.setenv(
        daemon_module.PROVIDER_EXTERNAL_ISOLATION_ENV,
        config.environment_json(),
    )
    monkeypatch.setenv(daemon_module.IMPLEMENTATION_PROVIDER_ENV, "codex")
    daemon = daemon_module.PortalImplementationDaemon(
        todo_path=repository / "tasks.todo.md",
        state_path=repository / "state.json",
        strategy_path=repository / "strategy.json",
        events_path=repository / "events.jsonl",
        repo_root=repository,
        worktree_root=repository,
        implementation_command="/usr/local/bin/codex exec -",
    )

    with pytest.raises(
        daemon_module.ImplementationRetryDeferred,
        match="rejects a direct command override",
    ):
        daemon._build_implementation_command(workspace)


def _scheduler_payload(
    isolation: dict[str, object],
) -> dict[str, object]:
    return {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "external_isolation_test.scheduler_config@1"
        ),
        "taskboard_path": "docs/tasks.md",
        "objectives_path": "docs/objectives.md",
        "plan_path": "docs/plan.md",
        "validator_path": "scripts/validate.py",
        "task_prefix": "TEST-",
        "board_namespace": "external-isolation-test",
        "merge_target_branch": "main",
        "max_lanes": 1,
        "strict_task_sharding": True,
        "idle_lane_work_stealing": "",
        "exit_when_all_tracks_terminal": True,
        "objective_refill_enabled": False,
        "codebase_refill_enabled": False,
        "poll_interval_seconds": 1,
        "daemon_interval_seconds": 1,
        "check_interval_seconds": 1,
        "stale_seconds": 60,
        "watchdog_startup_grace_seconds": 60,
        "max_restarts": 1,
        "max_task_attempts": 1,
        "implementation_retry_budget": 1,
        "validation_retry_budget": 1,
        "merge_retry_budget": 1,
        "implementation_timeout_seconds": 60,
        "implementation_max_timeout_seconds": 120,
        "implementation_log_stall_seconds": 60,
        "worktree_submodule_paths": [],
        "protected_paths": ["config/scheduler.json"],
        "runtime_paths": {
            "root": "state/runtime",
            "state": "state/runtime/state",
            "worktrees": "state/runtime/worktrees",
            "merge_queue": "state/runtime/merge-queue",
            "logs": "state/runtime/logs",
        },
        "lanes": [
            {
                "index": 0,
                "name": "test-lane-0",
                "strict_shard_remainder": 0,
            }
        ],
        "provider": {
            "provider_id": "codex",
            "max_concurrency": 1,
            "external_isolation": isolation,
        },
    }


def test_scheduler_seals_isolation_and_refuses_unavailable_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential = _credential(tmp_path)
    isolation = _isolation_payload(credential)
    repo = tmp_path / "repo"
    config_path = repo / "config" / "scheduler.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(_scheduler_payload(isolation)),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        daemon_module,
        "validate_external_provider_isolation_config",
        _fake_host_validation,
    )

    board = scheduler_module.load_configured_board(
        config_path,
        repo_root=repo,
    )
    plan = scheduler_module.configured_board_launch_plan(
        board,
        implement=True,
        detach=True,
        stamp="20260820T000000Z",
    )
    environment = plan["environment"]
    assert environment[scheduler_module.PROVIDER_ENV] == "codex"
    sealed = environment[scheduler_module.EXTERNAL_PROVIDER_ISOLATION_ENV]
    assert json.loads(sealed) == daemon_module.ExternalProviderIsolationConfig.parse(
        isolation
    ).to_dict()

    host_checks: list[bool] = []

    def unavailable(value, *, verify_host=True):
        host_checks.append(bool(verify_host))
        if verify_host:
            raise ValueError("immutable image unavailable")
        return daemon_module.ExternalProviderIsolationConfig.parse(value)

    monkeypatch.setattr(
        daemon_module,
        "validate_external_provider_isolation_config",
        unavailable,
    )
    # Structural loading remains deterministic in the socket-free validation
    # container.  The trusted launch-plan preflight repeats exact host/image
    # admission and fails closed before a child can be launched.
    board = scheduler_module.load_configured_board(config_path, repo_root=repo)
    assert host_checks == [False]
    with pytest.raises(
        scheduler_module.ConfiguredBoardError,
        match="launch preflight failed: immutable image unavailable",
    ):
        scheduler_module.configured_board_launch_plan(
            board,
            implement=True,
            detach=True,
            stamp="20260820T000001Z",
        )
    assert host_checks == [False, True]


def test_external_validation_command_isolates_linked_task_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential = _credential(tmp_path)
    repository, workspace = _linked_workspace(tmp_path)
    database_environment, control_root = _database_program_environment(
        repository
    )
    monkeypatch.setattr(
        daemon_module,
        "validate_external_provider_isolation_config",
        _fake_host_validation,
    )
    monkeypatch.setenv(
        multi_runner_module.REPOSITORY_ROOT_ENV,
        str(repository),
    )
    monkeypatch.setenv(
        multi_runner_module.DATABASE_PROGRAM_JSON_ENV,
        database_environment[multi_runner_module.DATABASE_PROGRAM_JSON_ENV],
    )

    command, docker_environment, _config, receipt = (
        daemon_module._docker_external_validation_command(
            spec=SimpleNamespace(
                command="python -m pytest -q test_example.py",
                raw_command="python -m pytest -q test_example.py",
            ),
            workspace_path=workspace,
            timeout_seconds=120.0,
            environment={
                "IPFS_ACCELERATE_AGENT_QUACK_TOKEN": "must-not-pass",
                daemon_module.PROVIDER_EXTERNAL_ISOLATION_ENV: "must-not-pass",
            },
            isolation_value=_isolation_payload(credential),
            container_name="pcpc-validation-linked-test",
        )
    )

    assert docker_environment == {
        "HOME": "/nonexistent",
        "PATH": "/usr/bin:/bin",
    }
    assert command[:4] == [
        "/usr/bin/docker",
        "--host=unix:///run/user/1000/docker.sock",
        "run",
        "--pull=never",
    ]
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--pid=host" not in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges:true" in command
    assert "--pids-limit=128" in command
    assert "--memory=1073741824" in command
    assert "--cpus=1" in command
    mounts = _mounts(command)
    assert [item for item in mounts if item.endswith(",readonly=false")] == [
        f"type=bind,src={workspace},dst={workspace},readonly=false"
    ]
    assert (
        f"type=bind,src={repository / '.git'},"
        f"dst={repository / '.git'},readonly"
    ) in mounts
    assert (
        f"type=bind,src={workspace / '.git'},"
        f"dst={workspace / '.git'},readonly"
    ) in mounts
    assert not any(str(credential) in item for item in mounts)
    assert not any(str(control_root) in item for item in mounts)
    assert not any("docker.sock" in item for item in mounts)
    assert not any("/proc" in item for item in mounts)
    assert receipt["control_root_masked"] is False
    image_index = command.index(IMAGE_ID)
    container_environment = command[image_index + 1 :]
    assert not any("QUACK" in item for item in container_environment)
    assert not any(
        daemon_module.PROVIDER_EXTERNAL_ISOLATION_ENV in item
        for item in container_environment
    )
    assert not any("auth.json" in item for item in command)


def test_external_validation_masks_control_state_in_main_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential = _credential(tmp_path)
    repository = tmp_path / "repository"
    (repository / ".git").mkdir(parents=True)
    database_environment, control_root = _database_program_environment(
        repository
    )
    token = control_root / "quack-owner" / "handle_pcpc-v1.quack-token"
    token.parent.mkdir(parents=True)
    token.write_text("host-secret\n", encoding="utf-8")
    monkeypatch.setattr(
        daemon_module,
        "validate_external_provider_isolation_config",
        _fake_host_validation,
    )
    monkeypatch.setenv(
        multi_runner_module.REPOSITORY_ROOT_ENV,
        str(repository),
    )
    monkeypatch.setenv(
        multi_runner_module.DATABASE_PROGRAM_JSON_ENV,
        database_environment[multi_runner_module.DATABASE_PROGRAM_JSON_ENV],
    )

    command, _docker_environment, _config, receipt = (
        daemon_module._docker_external_validation_command(
            spec=SimpleNamespace(
                command="python -m pytest -q test_example.py",
                raw_command="python -m pytest -q test_example.py",
            ),
            workspace_path=repository,
            timeout_seconds=120.0,
            environment={},
            isolation_value=_isolation_payload(credential),
            container_name="pcpc-validation-main-test",
        )
    )

    mounts = _mounts(command)
    assert [item for item in mounts if item.endswith(",readonly=false")] == [
        f"type=bind,src={repository},dst={repository},readonly=false"
    ]
    assert (
        f"type=bind,src={repository / '.git'},"
        f"dst={repository / '.git'},readonly"
    ) in mounts
    tmpfs_values = [
        command[index + 1]
        for index, item in enumerate(command[:-1])
        if item == "--tmpfs"
    ]
    assert any(
        item.startswith(f"{control_root}:rw,nosuid,nodev,noexec,mode=000,")
        for item in tmpfs_values
    )
    assert receipt["control_root_masked"] is True
    assert receipt["control_root"] == str(control_root)
    assert not any(str(token) in item for item in mounts)
    assert not any(str(credential) in item for item in mounts)


def test_external_validation_has_no_host_fallback_when_isolation_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential = _credential(tmp_path)
    repository, workspace = _linked_workspace(tmp_path)
    database_environment, _control_root = _database_program_environment(
        repository
    )
    isolation = daemon_module.ExternalProviderIsolationConfig.parse(
        _isolation_payload(credential)
    ).environment_json()
    monkeypatch.setenv(
        daemon_module.PROVIDER_EXTERNAL_ISOLATION_ENV,
        isolation,
    )
    monkeypatch.setenv(
        multi_runner_module.REPOSITORY_ROOT_ENV,
        str(repository),
    )
    monkeypatch.setenv(
        multi_runner_module.DATABASE_PROGRAM_JSON_ENV,
        database_environment[multi_runner_module.DATABASE_PROGRAM_JSON_ENV],
    )

    def unavailable(_value, *, verify_host=True):
        assert verify_host is True
        raise ValueError("Docker unavailable")

    monkeypatch.setattr(
        daemon_module,
        "validate_external_provider_isolation_config",
        unavailable,
    )
    result = daemon_module.PortalImplementationDaemon._validation_command_runner(
        spec=SimpleNamespace(
            command="python -m pytest -q test_example.py",
            raw_command="python -m pytest -q test_example.py",
        ),
        workspace_path=workspace,
        timeout_seconds=60.0,
        environment={},
    )

    assert result["returncode"] == 75
    assert result["infrastructure_failure"] is True
    assert result["error"] == "external_validation_isolation_unavailable"
    assert "Docker unavailable" in result["reason"]


@pytest.mark.parametrize("main_checkout", [False, True])
def test_live_external_validation_denies_host_authority_and_runs_pytest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    main_checkout: bool,
) -> None:
    """Opt-in local-Docker regression; default unit collection is hermetic."""

    isolation = os.environ.get(
        daemon_module.PROVIDER_EXTERNAL_ISOLATION_ENV,
        "",
    ).strip()
    if not isolation or os.environ.get(
        "IPFS_ACCELERATE_RUN_DOCKER_ISOLATION_TESTS",
        "",
    ) != "1":
        pytest.skip("set the sealed isolation JSON and opt-in Docker test flag")
    if main_checkout:
        repository = tmp_path / "repository"
        (repository / ".git").mkdir(parents=True)
        workspace = repository
    else:
        repository, workspace = _linked_workspace(tmp_path)
    database_environment, control_root = _database_program_environment(
        repository
    )
    token = control_root / "quack-owner" / "handle_pcpc-v1.quack-token"
    token.parent.mkdir(parents=True)
    token.write_text("host-secret-must-not-cross\n", encoding="utf-8")
    probe = workspace / "test_external_validation_probe.py"
    probe.write_text(
        "\n".join(
            (
                "import os",
                "import socket",
                "from pathlib import Path",
                "",
                "def test_external_validation_boundary():",
                f"    token = Path({str(token)!r})",
                "    try:",
                "        token.read_bytes()",
                "    except OSError:",
                "        pass",
                "    else:",
                "        raise AssertionError('host token was readable')",
                f"    assert not Path('/proc/{os.getpid()}').exists()",
                "    assert not Path('/var/run/docker.sock').exists()",
                "    assert not Path('/opt/codex-home/auth.json').exists()",
                "    assert not any('QUACK' in key for key in os.environ)",
                "    assert not any('EXTERNAL_ISOLATION' in key for key in os.environ)",
                "    root_probe = Path('/pcpc-root-write-must-fail')",
                "    try:",
                "        root_probe.write_text('denied', encoding='utf-8')",
                "    except OSError:",
                "        pass",
                "    else:",
                "        raise AssertionError('container root was writable')",
                "    network_probe = socket.socket()",
                "    network_probe.settimeout(0.2)",
                "    try:",
                "        network_probe.connect(('1.1.1.1', 53))",
                "    except OSError:",
                "        pass",
                "    else:",
                "        raise AssertionError('external network was reachable')",
                "    finally:",
                "        network_probe.close()",
                "    workspace_probe = Path(__file__).parent / 'probe-write.tmp'",
                "    workspace_probe.write_text('ok', encoding='utf-8')",
                "    assert workspace_probe.read_text(encoding='utf-8') == 'ok'",
                "    workspace_probe.unlink()",
                "",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        multi_runner_module.REPOSITORY_ROOT_ENV,
        str(repository),
    )
    monkeypatch.setenv(
        multi_runner_module.DATABASE_PROGRAM_JSON_ENV,
        database_environment[multi_runner_module.DATABASE_PROGRAM_JSON_ENV],
    )

    result = daemon_module._run_external_validation_in_container(
        spec=SimpleNamespace(
            command="python -m pytest -q test_external_validation_probe.py",
            raw_command="python -m pytest -q test_external_validation_probe.py",
        ),
        workspace_path=workspace,
        timeout_seconds=120.0,
        environment={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        isolation_value=isolation,
    )

    assert result["returncode"] == 0, result
    assert "1 passed" in result["output"]
    receipt = result["external_validation_isolation_receipt"]
    assert receipt["image_id"] == json.loads(isolation)["image_id"]
    assert receipt["network_mode"] == "none"
    assert receipt["credential_mounted"] is False
    assert receipt["docker_socket_mounted"] is False
    assert receipt["host_pid_namespace"] is False
    assert receipt["control_root_masked"] is main_checkout
    assert receipt["container_removed"] is True


def test_live_direct_provider_boundary_denies_host_and_keeps_auth_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opt-in denial probe for the exact direct-provider Docker boundary."""

    isolation = os.environ.get(
        daemon_module.PROVIDER_EXTERNAL_ISOLATION_ENV,
        "",
    ).strip()
    if not isolation or os.environ.get(
        "IPFS_ACCELERATE_RUN_DOCKER_ISOLATION_TESTS",
        "",
    ) != "1":
        pytest.skip("set the sealed isolation JSON and opt-in Docker test flag")
    config = daemon_module.validate_external_provider_isolation_config(
        isolation
    )
    repository, workspace = _linked_workspace(tmp_path)
    _database_environment, control_root = _database_program_environment(
        repository
    )
    token = control_root / "quack-owner" / "handle_pcpc-v1.quack-token"
    token.parent.mkdir(parents=True, exist_ok=True)
    token.write_text("host-secret-must-not-cross\n", encoding="utf-8")
    host_auth_before = Path(config.credential_file).read_bytes()
    inner = [
        config.container_executable,
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C",
        str(workspace),
        "-",
    ]
    command = daemon_module._docker_codex_implementation_command(
        inner_command=inner,
        workspace_path=workspace,
        repository_root=repository,
        config=config,
    )
    image_index = command.index(config.image_id)
    codex_index = command.index(config.container_executable, image_index)
    probe_script = "\n".join(
        (
            "import json",
            "import os",
            "from pathlib import Path",
            f"workspace = Path({str(workspace)!r})",
            f"token = Path({str(token)!r})",
            f"auth_source = Path({config.credential_file!r})",
            "auth_target = Path('/opt/codex-home/auth.json')",
            "results = {}",
            "results['auth_at_intended_target'] = auth_target.is_file()",
            "results['auth_source_path_absent'] = not auth_source.exists()",
            "try:",
            "    auth_target.read_bytes()",
            "    auth_readable = True",
            "except OSError:",
            "    auth_readable = False",
            "try:",
            "    auth_target.write_bytes(b'refreshed-copy')",
            "    auth_writable = True",
            "except OSError:",
            "    auth_writable = False",
            "results['auth_copy_writable'] = auth_readable and auth_writable",
            "try:",
            "    token.read_bytes()",
            "    token_readable = True",
            "except OSError:",
            "    token_readable = False",
            "results['host_state_canary_absent'] = not token_readable",
            f"results['host_pid_absent'] = not Path('/proc/{os.getpid()}').exists()",
            "results['docker_socket_absent'] = not Path('/var/run/docker.sock').exists()",
            "results['quack_environment_absent'] = not any('QUACK' in key for key in os.environ)",
            "root_probe = Path('/pcpc-provider-root-write-must-fail')",
            "try:",
            "    root_probe.write_text('denied', encoding='utf-8')",
            "    root_writable = True",
            "except OSError:",
            "    root_writable = False",
            "results['root_write_denied'] = not root_writable",
            "workspace_probe = workspace / 'provider-probe-write.tmp'",
            "try:",
            "    workspace_probe.write_text('ok', encoding='utf-8')",
            "    workspace_writable = workspace_probe.read_text(encoding='utf-8') == 'ok'",
            "    workspace_probe.unlink()",
            "except OSError:",
            "    workspace_writable = False",
            "results['workspace_writable'] = workspace_writable",
            "print(json.dumps(results, sort_keys=True))",
            "raise SystemExit(0 if all(results.values()) else 1)",
        )
    )
    probe_command = [
        *command[:codex_index],
        str(daemon_module._VALIDATION_CONTAINER_PYTHON),
        "-c",
        probe_script,
    ]

    completed = subprocess.run(
        probe_command,
        stdin=subprocess.DEVNULL,
        text=True,
        capture_output=True,
        timeout=30.0,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    results = json.loads(completed.stdout.strip().splitlines()[-1])
    assert results == {
        "auth_at_intended_target": True,
        "auth_copy_writable": True,
        "auth_source_path_absent": True,
        "docker_socket_absent": True,
        "host_pid_absent": True,
        "host_state_canary_absent": True,
        "quack_environment_absent": True,
        "root_write_denied": True,
        "workspace_writable": True,
    }
    assert Path(config.credential_file).read_bytes() == host_auth_before
    assert host_auth_before != b"refreshed-copy"


def test_sealed_isolation_survives_profile_gate_and_daemon_handoffs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential = _credential(tmp_path)
    isolation = daemon_module.ExternalProviderIsolationConfig.parse(
        _isolation_payload(credential)
    ).environment_json()
    repository = tmp_path / "repo"
    config_path = repository / "config" / "scheduler.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(_scheduler_payload(json.loads(isolation))),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        daemon_module,
        "validate_external_provider_isolation_config",
        _fake_host_validation,
    )
    board = scheduler_module.load_configured_board(
        config_path,
        repo_root=repository,
    )
    launch_plan = scheduler_module.configured_board_launch_plan(
        board,
        implement=True,
        detach=True,
        stamp="20260820T010101Z",
    )
    state_root = repository / "state"
    run_root = state_root / "runs" / "lane-0"
    scheduler_environment = dict(launch_plan["environment"])
    scheduler_environment.update(
        {
            multi_runner_module.STATE_STORE_LIVE_GENERATION_ENV: "17",
            multi_runner_module.STATE_LIVE_SCHEMA_REVISION_ENV: "9",
        }
    )
    profile = multi_runner_module.LifecycleProfile(
        target_id="supervisor-track:lane-0",
        run_id="test-run",
        configuration_root="sha256:" + "1" * 64,
        repository_root=str(repository),
        state_root=str(state_root),
        run_root=str(run_root),
        argv=("/usr/bin/python3", "-c", "raise SystemExit(0)"),
        cwd=str(repository),
        environment=multi_runner_module._plan_bound_profile_environment(
            scheduler_environment
        ),
    )
    gate_input = profile.launch_environment(0)
    gate_input["HOSTILE_AMBIENT"] = "must-not-cross"
    lane_environment = (
        multi_runner_module._plan_bound_positive_child_environment(
            gate_input
        )
    )
    assert lane_environment[
        daemon_module.PROVIDER_EXTERNAL_ISOLATION_ENV
    ] == isolation
    assert lane_environment[
        multi_runner_module.STATE_STORE_LIVE_GENERATION_ENV
    ] == "17"
    assert lane_environment[
        multi_runner_module.STATE_LIVE_SCHEMA_REVISION_ENV
    ] == "9"
    assert lane_environment[
        multi_runner_module.REPOSITORY_ROOT_ENV
    ] == str(repository)
    assert "HOSTILE_AMBIENT" not in lane_environment

    with monkeypatch.context() as lane_process:
        for name, value in lane_environment.items():
            lane_process.setenv(name, value)
        daemon_environment = supervisor_module._managed_daemon_child_environment()
    assert daemon_environment[
        daemon_module.PROVIDER_EXTERNAL_ISOLATION_ENV
    ] == isolation
    assert daemon_environment[
        multi_runner_module.STATE_STORE_LIVE_GENERATION_ENV
    ] == "17"
    assert daemon_environment[
        multi_runner_module.STATE_LIVE_SCHEMA_REVISION_ENV
    ] == "9"
    assert daemon_environment[
        multi_runner_module.REPOSITORY_ROOT_ENV
    ] == str(repository)

    provider_environment = multi_runner_module.provider_subprocess_environment(
        lane_environment
    )
    assert daemon_module.PROVIDER_EXTERNAL_ISOLATION_ENV not in (
        provider_environment
    )
    assert multi_runner_module.STATE_STORE_LIVE_GENERATION_ENV not in (
        provider_environment
    )
    assert multi_runner_module.STATE_LIVE_SCHEMA_REVISION_ENV not in (
        provider_environment
    )
    assert multi_runner_module.REPOSITORY_ROOT_ENV not in provider_environment


def test_trusted_duckdb_home_is_profile_bound_and_removed_from_provider(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    trusted_home = (
        repository
        / "state"
        / "qualification-homes"
        / ("a" * 64)
    )
    trusted_home.mkdir(parents=True, mode=0o700)
    xdg_cache = trusted_home / ".cache" / "xdg"
    cuda_cache = trusted_home / ".cache" / "cuda"
    xdg_cache.mkdir(parents=True, mode=0o700)
    cuda_cache.mkdir(mode=0o700)
    (trusted_home / ".cache").chmod(0o700)
    trusted_home.chmod(0o500)
    python_user_base = repository / "python-user-base"
    python_user_base.mkdir(mode=0o700)
    environment = {
        "HOME": str(trusted_home),
        multi_runner_module.TRUSTED_DUCKDB_HOME_ENV: str(trusted_home),
        multi_runner_module.TRUSTED_PYTHON_USER_BASE_ENV: str(python_user_base),
        multi_runner_module.TRUSTED_XDG_CACHE_HOME_ENV: "/tmp/hostile-xdg",
        multi_runner_module.TRUSTED_CUDA_CACHE_PATH_ENV: "/tmp/hostile-cuda",
        multi_runner_module.TRUSTED_CUDA_CACHE_DISABLE_ENV: "0",
        multi_runner_module.TRUSTED_PYTHONDONTWRITEBYTECODE_ENV: "0",
    }

    profile_environment = dict(
        multi_runner_module._trusted_duckdb_profile_environment(
            environment,
            repository_root=repository,
        )
    )
    assert profile_environment["HOME"] == str(trusted_home)
    assert profile_environment[
        multi_runner_module.TRUSTED_DUCKDB_HOME_ENV
    ] == str(trusted_home)
    assert profile_environment[multi_runner_module.TRUSTED_XDG_CACHE_HOME_ENV] == str(
        xdg_cache
    )
    assert profile_environment[multi_runner_module.TRUSTED_CUDA_CACHE_PATH_ENV] == str(
        cuda_cache
    )
    assert profile_environment[multi_runner_module.TRUSTED_CUDA_CACHE_DISABLE_ENV] == "1"
    assert profile_environment[
        multi_runner_module.TRUSTED_PYTHONDONTWRITEBYTECODE_ENV
    ] == "1"

    lane_environment = {
        **profile_environment,
        multi_runner_module.REPOSITORY_ROOT_ENV: str(repository),
    }
    projected = multi_runner_module._plan_bound_positive_child_environment(
        lane_environment
    )
    assert projected["HOME"] == str(trusted_home)
    assert projected[multi_runner_module.TRUSTED_XDG_CACHE_HOME_ENV] == str(xdg_cache)
    assert projected[multi_runner_module.TRUSTED_CUDA_CACHE_PATH_ENV] == str(cuda_cache)

    previous = {
        name: os.environ.get(name)
        for name in (
            "HOME",
            multi_runner_module.TRUSTED_DUCKDB_HOME_ENV,
            multi_runner_module.TRUSTED_PYTHON_USER_BASE_ENV,
            multi_runner_module.REPOSITORY_ROOT_ENV,
            *multi_runner_module.TRUSTED_RUNTIME_CACHE_ENV_NAMES,
        )
    }
    try:
        os.environ.update(projected)
        managed = supervisor_module._managed_daemon_child_environment()
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    assert managed["HOME"] == str(trusted_home)
    assert managed[multi_runner_module.TRUSTED_DUCKDB_HOME_ENV] == str(
        trusted_home
    )
    assert managed[multi_runner_module.TRUSTED_XDG_CACHE_HOME_ENV] == str(xdg_cache)
    assert managed[multi_runner_module.TRUSTED_CUDA_CACHE_PATH_ENV] == str(cuda_cache)

    provider_environment = multi_runner_module.provider_subprocess_environment(
        projected
    )
    assert "HOME" not in provider_environment
    assert multi_runner_module.TRUSTED_DUCKDB_HOME_ENV not in provider_environment
    assert (
        multi_runner_module.TRUSTED_PYTHON_USER_BASE_ENV
        not in provider_environment
    )
    assert not (
        set(multi_runner_module.TRUSTED_RUNTIME_CACHE_ENV_NAMES)
        & set(provider_environment)
    )

    unpaired = multi_runner_module._plan_bound_positive_child_environment(
        {
            multi_runner_module.TRUSTED_PYTHON_USER_BASE_ENV: (
                "/tmp/hostile-python-user-base"
            ),
            multi_runner_module.TRUSTED_XDG_CACHE_HOME_ENV: "/tmp/hostile-xdg",
            multi_runner_module.TRUSTED_CUDA_CACHE_PATH_ENV: "/tmp/hostile-cuda",
            multi_runner_module.TRUSTED_CUDA_CACHE_DISABLE_ENV: "0",
            multi_runner_module.TRUSTED_PYTHONDONTWRITEBYTECODE_ENV: "0",
        }
    )
    assert (
        multi_runner_module.TRUSTED_PYTHON_USER_BASE_ENV not in unpaired
    )
    assert not set(multi_runner_module.TRUSTED_RUNTIME_CACHE_ENV_NAMES) & set(unpaired)

    unpaired_provider = multi_runner_module.provider_subprocess_environment(
        {
            name: "hostile"
            for name in multi_runner_module.TRUSTED_RUNTIME_CACHE_ENV_NAMES
        }
    )
    assert not set(multi_runner_module.TRUSTED_RUNTIME_CACHE_ENV_NAMES) & set(
        unpaired_provider
    )

    with pytest.raises(ValueError, match="binding is incomplete"):
        multi_runner_module._trusted_duckdb_profile_environment(
            {**environment, "HOME": str(Path.home())},
            repository_root=repository,
        )

    trusted_home.chmod(0o700)
    with pytest.raises(ValueError, match="immutable owned directory"):
        multi_runner_module._trusted_duckdb_profile_environment(
            environment,
            repository_root=repository,
        )
    trusted_home.chmod(0o500)
    xdg_cache.chmod(0o755)
    with pytest.raises(ValueError, match="cache directory is unsafe"):
        multi_runner_module._trusted_duckdb_profile_environment(
            environment,
            repository_root=repository,
        )
    xdg_cache.chmod(0o700)

    with pytest.raises(ValueError, match="must be paired"):
        multi_runner_module._trusted_duckdb_profile_environment(
            {
                "HOME": str(trusted_home),
                multi_runner_module.TRUSTED_DUCKDB_HOME_ENV: str(trusted_home),
            },
            repository_root=repository,
        )
    trusted_home.chmod(0o700)


def test_external_isolation_contract_rejects_unknown_fields(
    tmp_path: Path,
) -> None:
    payload = _isolation_payload(_credential(tmp_path))
    payload["arbitrary_mount"] = "/"
    with pytest.raises(ValueError, match="unknown or missing fields"):
        daemon_module.ExternalProviderIsolationConfig.parse(payload)


@pytest.mark.parametrize(
    "endpoint",
    (
        "tcp://127.0.0.1:2375",
        "unix:///tmp/docker.sock",
        "unix:///run/user/01000/docker.sock",
    ),
)
def test_external_isolation_contract_rejects_noncanonical_local_endpoints(
    tmp_path: Path,
    endpoint: str,
) -> None:
    payload = _isolation_payload(_credential(tmp_path))
    payload["runtime_endpoint"] = endpoint

    with pytest.raises(ValueError, match="canonical local socket"):
        daemon_module.validate_external_provider_isolation_config(
            payload,
            verify_host=False,
        )


def test_external_isolation_host_admission_remains_uid_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _isolation_payload(_credential(tmp_path))
    monkeypatch.setattr(
        daemon_module,
        "_DOCKER_LOCAL_ENDPOINTS",
        frozenset(
            {
                "unix:///var/run/docker.sock",
                "unix:///run/user/0/docker.sock",
            }
        ),
    )

    parsed = daemon_module.validate_external_provider_isolation_config(
        payload,
        verify_host=False,
    )
    assert parsed.runtime_endpoint == "unix:///run/user/1000/docker.sock"
    with pytest.raises(ValueError, match="not admitted for the current runtime"):
        daemon_module.validate_external_provider_isolation_config(
            payload,
            verify_host=True,
        )


def test_host_validation_rejects_image_codex_digest_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _isolation_payload(_credential(tmp_path))

    def mismatched_inspect(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"{IMAGE_ID}|linux|arm64|test-v1|{'0' * 64}\n",
            stderr="",
        )

    monkeypatch.setattr(daemon_module.subprocess, "run", mismatched_inspect)
    with pytest.raises(ValueError, match="image identity is unavailable"):
        daemon_module.validate_external_provider_isolation_config(payload)


def test_host_validation_hashes_image_codex_bytes_and_rejects_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _isolation_payload(_credential(tmp_path))
    commands: list[list[str]] = []

    def mismatched_bytes(command, **_kwargs):
        commands.append(command)
        if "inspect" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    f"{IMAGE_ID}|linux|arm64|test-v1|{CODEX_SHA256}\n"
                ),
                stderr="",
            )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"{'0' * 64}  /usr/local/bin/codex\n",
            stderr="",
        )

    monkeypatch.setattr(daemon_module.subprocess, "run", mismatched_bytes)
    with pytest.raises(ValueError, match="executable digest mismatch"):
        daemon_module.validate_external_provider_isolation_config(payload)

    assert len(commands) == 2
    digest_command = commands[1]
    assert digest_command[:4] == [
        "/usr/bin/docker",
        "--host=unix:///run/user/1000/docker.sock",
        "run",
        "--pull=never",
    ]
    assert "--network=none" in digest_command
    assert "--read-only" in digest_command
    assert "--cap-drop=ALL" in digest_command
    assert "--security-opt=no-new-privileges:true" in digest_command
    assert "--pids-limit=16" in digest_command
    assert "--memory=268435456" in digest_command
    assert "--cpus=0.25" in digest_command
    assert "--mount" not in digest_command
    assert digest_command[-2:] == [IMAGE_ID, "/usr/local/bin/codex"]


def _configure_host_cli_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    binary_name: str,
) -> tuple[object, Path, Path]:
    credential = _credential(tmp_path)
    repository, workspace = _linked_workspace(tmp_path)
    fake_cli = tmp_path / binary_name
    fake_cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_cli.chmod(0o755)
    config = daemon_module.ExternalProviderIsolationConfig.parse(
        _isolation_payload(credential)
    )
    monkeypatch.setattr(
        daemon_module,
        "validate_external_provider_isolation_config",
        _fake_host_validation,
    )
    monkeypatch.setenv(
        daemon_module.PROVIDER_EXTERNAL_ISOLATION_ENV,
        config.environment_json(),
    )
    monkeypatch.setenv(
        daemon_module._LIFECYCLE_REPOSITORY_ROOT_ENV,
        str(repository),
    )
    monkeypatch.setattr(
        daemon_module,
        "_host_cli_binary",
        lambda name, _expected=binary_name, _path=str(fake_cli): (
            _path if name == _expected else None
        ),
    )
    return config, workspace, fake_cli


def _assert_host_cli_docker_wrap(
    command: list[str],
    *,
    config: object,
    family: str,
    workspace: Path,
) -> None:
    assert command[4:8] == [
        "/usr/bin/docker",
        "--host=unix:///run/user/1000/docker.sock",
        "run",
        "--pull=never",
    ]
    assert config.image_id in command
    assert f"HOME={daemon_module._CLI_CONTAINER_HOME}" in command
    assert f"--user" in command
    assert f"{os.getuid()}:{os.getgid()}" in command
    mount_specs = _mounts(command)
    assert any(
        spec.startswith("type=bind,src=/usr,dst=/usr,")
        or ",src=/usr,dst=/usr," in spec
        for spec in mount_specs
    )
    assert any(f"dst={workspace}" in spec or spec.endswith(f"dst={workspace},readonly=false") for spec in mount_specs)
    assert f"PATH={daemon_module._CLI_CONTAINER_BIN}:" in " ".join(
        item for item in command if item.startswith("PATH=")
    )
    del family


def test_isolation_wraps_claude_host_cli_in_docker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, workspace, _fake = _configure_host_cli_isolation(
        tmp_path, monkeypatch, binary_name="claude"
    )
    command = daemon_module._claude_implementation_command(
        workspace_path=workspace,
    )
    _assert_host_cli_docker_wrap(
        command, config=config, family="claude", workspace=workspace
    )
    assert "ipfs_accelerate_py.agent_supervisor.cli_implement_runner" in command
    assert "--provider" in command
    assert "claude" in command


def test_isolation_wraps_gemini_host_cli_in_docker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, workspace, _fake = _configure_host_cli_isolation(
        tmp_path, monkeypatch, binary_name="gemini"
    )
    command = daemon_module._gemini_implementation_command(
        workspace_path=workspace,
    )
    _assert_host_cli_docker_wrap(
        command, config=config, family="gemini", workspace=workspace
    )
    assert "gemini" in command


def test_isolation_wraps_mistral_vibe_host_cli_in_docker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, workspace, _fake = _configure_host_cli_isolation(
        tmp_path, monkeypatch, binary_name="vibe"
    )
    command = daemon_module._mistral_implementation_command(
        workspace_path=workspace,
    )
    _assert_host_cli_docker_wrap(
        command, config=config, family="mistral", workspace=workspace
    )
    assert "mistral" in command


def test_isolation_wraps_goose_host_cli_in_docker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, workspace, fake = _configure_host_cli_isolation(
        tmp_path, monkeypatch, binary_name="goose"
    )
    monkeypatch.setattr(daemon_module, "_goose_binary", lambda: str(fake))
    monkeypatch.setattr(
        daemon_module, "_resolve_meta_spark_api_key", lambda: "test-key"
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    command = daemon_module._goose_meta_spark_command(workspace_path=workspace)
    _assert_host_cli_docker_wrap(
        command, config=config, family="goose", workspace=workspace
    )
    assert "ipfs_accelerate_py.agent_supervisor.integrations.meta_spark_goose_runner" in command
    assert not any(item.startswith("OPENAI_API_KEY=") for item in command)
    assert not any("test-key" in item for item in command)
    assert any(
        f"dst={daemon_module._CLI_CONTAINER_HOME / daemon_module._ISOLATED_ENV_FILE_NAME}"
        in item
        for item in _mounts(command)
    )
    assert str(daemon_module._CLI_CONTAINER_HOME / daemon_module._ISOLATED_ENV_EXEC_NAME) in command


def test_isolation_wraps_copilot_host_cli_in_docker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, workspace, fake = _configure_host_cli_isolation(
        tmp_path, monkeypatch, binary_name="copilot"
    )
    command = daemon_module._copilot_fallback_command(
        codex=None,
        copilot=str(fake),
        workspace_path=workspace,
    )
    _assert_host_cli_docker_wrap(
        command, config=config, family="copilot", workspace=workspace
    )
    assert str(fake) in command


def test_sealed_grok_isolation_prefers_pcpc_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner

    monkeypatch.delenv(grok_cli_runner._SEALED_PROVIDER_ISOLATION_ENV, raising=False)
    assert grok_cli_runner._sealed_provider_isolation_image_id() == ""
    payload = {
        "schema": daemon_module.PROVIDER_EXTERNAL_ISOLATION_SCHEMA,
        "image_id": IMAGE_ID,
        "runtime_endpoint": "unix:///run/user/1000/docker.sock",
    }
    monkeypatch.setenv(
        grok_cli_runner._SEALED_PROVIDER_ISOLATION_ENV,
        json.dumps(payload),
    )
    assert grok_cli_runner._sealed_provider_isolation_image_id() == IMAGE_ID
    assert grok_cli_runner._docker_isolation_host() == (
        "unix:///run/user/1000/docker.sock"
    )


def _implementation_daemon(
    repository: Path,
    *,
    implementation_command: str = "",
) -> daemon_module.PortalImplementationDaemon:
    return daemon_module.PortalImplementationDaemon(
        todo_path=repository / "tasks.todo.md",
        state_path=repository / "state.json",
        strategy_path=repository / "strategy.json",
        events_path=repository / "events.jsonl",
        repo_root=repository,
        worktree_root=repository,
        implementation_command=implementation_command,
    )


def test_isolation_allows_host_grok_without_wrapping_docker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential = _credential(tmp_path)
    repository, workspace = _linked_workspace(tmp_path)
    config = daemon_module.ExternalProviderIsolationConfig.parse(
        _isolation_payload(credential)
    )
    monkeypatch.setattr(
        daemon_module,
        "validate_external_provider_isolation_config",
        _fake_host_validation,
    )
    monkeypatch.setenv(
        daemon_module.PROVIDER_EXTERNAL_ISOLATION_ENV,
        config.environment_json(),
    )
    monkeypatch.setenv(daemon_module.IMPLEMENTATION_PROVIDER_ENV, "grok")
    monkeypatch.setattr(
        daemon_module,
        "_configured_agent_implementation_route_plan",
        lambda _repo: None,
    )
    monkeypatch.setattr(daemon_module, "_grok_cli_available", lambda: True)
    monkeypatch.setattr(
        daemon_module,
        "_grok_binary",
        lambda: "/opt/providers/grok",
    )
    monkeypatch.setattr(
        daemon_module,
        "_grok_cli_command",
        lambda **kwargs: [
            "/opt/providers/grok",
            "agent",
            str(kwargs["workspace_path"]),
        ],
    )
    daemon_module._ISOLATED_CODEX_CODE_MODE_HOST_PROBE_CACHE.clear()
    daemon = _implementation_daemon(repository)

    command = daemon._build_implementation_command(workspace)

    assert command[:2] == ["/opt/providers/grok", "agent"]
    assert "docker" not in command
    assert config.image_id not in command


def test_isolation_auto_selects_grok_when_isolated_codex_cannot_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential = _credential(tmp_path)
    repository, workspace = _linked_workspace(tmp_path)
    config = daemon_module.ExternalProviderIsolationConfig.parse(
        _isolation_payload(credential)
    )
    monkeypatch.setattr(
        daemon_module,
        "validate_external_provider_isolation_config",
        _fake_host_validation,
    )
    monkeypatch.setenv(
        daemon_module.PROVIDER_EXTERNAL_ISOLATION_ENV,
        config.environment_json(),
    )
    monkeypatch.delenv(daemon_module.IMPLEMENTATION_PROVIDER_ENV, raising=False)
    monkeypatch.setattr(
        daemon_module,
        "_configured_agent_implementation_route_plan",
        lambda _repo: None,
    )
    monkeypatch.setattr(daemon_module, "_grok_cli_available", lambda: True)
    monkeypatch.setattr(
        daemon_module,
        "_grok_binary",
        lambda: "/opt/providers/grok",
    )
    monkeypatch.setattr(
        daemon_module,
        "_codex_ready_for_automatic_routing",
        lambda: False,
    )
    monkeypatch.setattr(
        daemon_module,
        "_secondary_implementation_cli_ready",
        lambda: False,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_provider_auto import (
        AutoProviderDecision,
        AutoProviderSelection,
    )

    def _select_grok(**_kwargs):
        return AutoProviderSelection(
            decision=AutoProviderDecision.GROK,
            selected_provider="grok",
        )

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_provider_auto.select_auto_implementation_provider",
        _select_grok,
    )
    monkeypatch.setattr(
        daemon_module,
        "resolve_agent_implementation_route",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        daemon_module,
        "_grok_cli_command",
        lambda **kwargs: [
            "/opt/providers/grok",
            "agent",
            str(kwargs["workspace_path"]),
        ],
    )
    daemon_module._ISOLATED_CODEX_CODE_MODE_HOST_PROBE_CACHE.clear()
    daemon = _implementation_daemon(repository)
    monkeypatch.setattr(daemon, "_provider_capacity_latch_states", lambda: {})

    command = daemon._build_implementation_command(workspace)

    assert command[:2] == ["/opt/providers/grok", "agent"]
    assert "docker" not in command


def test_scheduler_accepts_auto_provider_with_codex_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential = _credential(tmp_path)
    isolation = _isolation_payload(credential)
    repo = tmp_path / "repo"
    config_path = repo / "config" / "scheduler.json"
    config_path.parent.mkdir(parents=True)
    payload = _scheduler_payload(isolation)
    payload["provider"]["provider_id"] = "auto"
    config_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        daemon_module,
        "validate_external_provider_isolation_config",
        _fake_host_validation,
    )

    board = scheduler_module.load_configured_board(
        config_path,
        repo_root=repo,
    )
    plan = scheduler_module.configured_board_launch_plan(
        board,
        implement=True,
        detach=True,
        stamp="20260820T000000Z",
    )
    environment = plan["environment"]
    assert environment[scheduler_module.PROVIDER_ENV] == "auto"
    sealed = environment[scheduler_module.EXTERNAL_PROVIDER_ISOLATION_ENV]
    assert json.loads(sealed)["provider_id"] == "codex"


def test_install_ephemeral_credential_rejects_symlink(tmp_path: Path) -> None:
    from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner

    real = tmp_path / "real.json"
    real.write_text("{}\n", encoding="utf-8")
    real.chmod(0o600)
    link = tmp_path / "auth.json"
    link.symlink_to(real)
    dest = tmp_path / "copy" / "auth.json"
    with pytest.raises(ValueError, match="unavailable"):
        grok_cli_runner._install_ephemeral_credential(link, dest)
    assert not dest.exists()


def test_isolated_grok_home_copies_operator_auth(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner

    operator_home = tmp_path / "operator-grok"
    operator_home.mkdir()
    auth = operator_home / "auth.json"
    payload = b'{"access_token":"operator-secret"}\n'
    auth.write_bytes(payload)
    auth.chmod(0o600)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".git").mkdir()

    temporary_home, env, _policy, denied = grok_cli_runner._isolated_grok_home(
        base_env={
            "GROK_HOME": str(operator_home),
            "HOME": str(tmp_path / "sealed"),
            "PATH": "/usr/bin:/bin",
        },
        child_env={"PATH": "/usr/bin:/bin"},
        codex_fallback_command=(),
        workspace=workspace,
    )
    try:
        grok_home = Path(temporary_home.name)
        copied = grok_home / "auth.json"
        nested = grok_home / ".grok" / "auth.json"
        assert copied.is_file()
        assert not copied.is_symlink()
        assert nested.is_file()
        assert copied.read_bytes() == payload
        assert nested.read_bytes() == payload
        assert copied.stat().st_mode & 0o777 == 0o600
        assert copied.stat().st_ino != auth.stat().st_ino
        assert auth.read_bytes() == payload
        assert env["GROK_HOME"] == str(grok_home)
        assert env["HOME"] == str(grok_home)
        assert any(path == auth or Path(path) == auth for path in denied)
    finally:
        temporary_home.cleanup()


def test_host_cli_wrap_keeps_secret_values_off_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator_home = tmp_path / "operator"
    goose_dir = operator_home / ".config" / "goose"
    goose_dir.mkdir(parents=True)
    secrets = goose_dir / "secrets.yaml"
    secrets.write_text("key: host-secret-file\n", encoding="utf-8")
    secrets.chmod(0o600)
    config, workspace, fake = _configure_host_cli_isolation(
        tmp_path, monkeypatch, binary_name="goose"
    )
    monkeypatch.setattr(daemon_module, "_goose_binary", lambda: str(fake))
    monkeypatch.setattr(daemon_module, "_operator_home_dir", lambda: operator_home)
    monkeypatch.setattr(
        daemon_module, "_resolve_meta_spark_api_key", lambda: "env-secret-value"
    )
    for name in daemon_module._HOST_CLI_ENV_NAMES["goose"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "env-secret-value")
    command = daemon_module._goose_meta_spark_command(workspace_path=workspace)
    _assert_host_cli_docker_wrap(
        command, config=config, family="goose", workspace=workspace
    )
    assert not any("env-secret-value" in item for item in command)
    assert not any("host-secret-file" in item for item in command)
    assert not any(item.startswith("OPENAI_API_KEY=") for item in command)
    assert not any(str(secrets) in item for item in command)
    mounts = _mounts(command)
    secret_mounts = [
        item for item in mounts if "dst=/opt/cli-home/.config/goose/secrets.yaml" in item
    ]
    assert len(secret_mounts) == 1
    assert str(secrets) not in secret_mounts[0]
    env_file_dst = (
        f"dst={daemon_module._CLI_CONTAINER_HOME / daemon_module._ISOLATED_ENV_FILE_NAME},"
    )
    env_file_mounts = [item for item in mounts if env_file_dst in item]
    assert len(env_file_mounts) == 1
    env_src = None
    for part in env_file_mounts[0].split(","):
        if part.startswith("src="):
            env_src = Path(part[4:])
            break
    assert env_src is not None
    assert env_src.read_text(encoding="utf-8") == "OPENAI_API_KEY=env-secret-value\n"
    assert env_src.stat().st_mode & 0o777 == 0o600


def test_isolated_codex_without_code_mode_host_is_not_auto_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential = _credential(tmp_path)
    config = daemon_module.ExternalProviderIsolationConfig.parse(
        _isolation_payload(credential)
    )
    monkeypatch.setattr(daemon_module, "_host_codex_vendor_binaries", lambda: None)
    daemon_module._ISOLATED_CODEX_CODE_MODE_HOST_PROBE_CACHE.clear()
    assert (
        daemon_module._isolated_codex_code_mode_host_ready(config, probe=False)
        is False
    )

    def missing_host(command, **_kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="")

    monkeypatch.setattr(daemon_module.subprocess, "run", missing_host)
    assert (
        daemon_module._isolated_codex_code_mode_host_ready(config, probe=True)
        is False
    )
    assert (
        daemon_module._ISOLATED_CODEX_CODE_MODE_HOST_PROBE_CACHE[config.image_id]
        is False
    )


def test_isolation_still_wraps_explicit_codex(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential = _credential(tmp_path)
    repository, workspace = _linked_workspace(tmp_path)
    config = daemon_module.ExternalProviderIsolationConfig.parse(
        _isolation_payload(credential)
    )
    monkeypatch.setattr(
        daemon_module,
        "validate_external_provider_isolation_config",
        _fake_host_validation,
    )
    monkeypatch.setenv(
        daemon_module.PROVIDER_EXTERNAL_ISOLATION_ENV,
        config.environment_json(),
    )
    monkeypatch.setenv(daemon_module.IMPLEMENTATION_PROVIDER_ENV, "codex")
    monkeypatch.setattr(
        daemon_module,
        "_configured_agent_implementation_route_plan",
        lambda _repo: None,
    )
    daemon = _implementation_daemon(repository)

    command = daemon._build_implementation_command(workspace)

    assert command[4:8] == [
        "/usr/bin/docker",
        "--host=unix:///run/user/1000/docker.sock",
        "run",
        "--pull=never",
    ]
    assert config.image_id in command
    assert "--dangerously-bypass-approvals-and-sandbox" in command
    assert "features.code_mode=false" not in command


def test_host_codex_vendor_pair_makes_isolated_codex_auto_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential = _credential(tmp_path)
    config = daemon_module.ExternalProviderIsolationConfig.parse(
        _isolation_payload(credential)
    )
    vendor_bin = tmp_path / "vendor-bin"
    vendor_bin.mkdir()
    host_codex = vendor_bin / "codex"
    host_companion = vendor_bin / "codex-code-mode-host"
    host_codex.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    host_companion.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    host_codex.chmod(0o755)
    host_companion.chmod(0o755)
    monkeypatch.setattr(
        daemon_module,
        "_host_codex_vendor_binaries",
        lambda: (host_codex.resolve(), host_companion.resolve()),
    )
    monkeypatch.setattr(
        daemon_module,
        "validate_external_provider_isolation_config",
        _fake_host_validation,
    )
    monkeypatch.setenv(
        daemon_module.PROVIDER_EXTERNAL_ISOLATION_ENV,
        config.environment_json(),
    )
    daemon_module._ISOLATED_CODEX_CODE_MODE_HOST_PROBE_CACHE.clear()
    assert (
        daemon_module._isolated_codex_code_mode_host_ready(config, probe=False)
        is True
    )
    assert daemon_module._codex_ready_for_automatic_routing() is True


def test_isolation_bind_mounts_newer_host_codex_vendor_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential = _credential(tmp_path)
    repository, workspace = _linked_workspace(tmp_path)
    vendor_bin = tmp_path / "vendor-bin"
    vendor_bin.mkdir()
    host_codex = vendor_bin / "codex"
    host_companion = vendor_bin / "codex-code-mode-host"
    host_codex.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    host_companion.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    host_codex.chmod(0o755)
    host_companion.chmod(0o755)
    config = daemon_module.ExternalProviderIsolationConfig.parse(
        _isolation_payload(credential)
    )
    monkeypatch.setattr(
        daemon_module,
        "validate_external_provider_isolation_config",
        _fake_host_validation,
    )
    monkeypatch.setattr(
        daemon_module,
        "_host_codex_vendor_binaries",
        lambda: (host_codex.resolve(), host_companion.resolve()),
    )
    monkeypatch.setenv(
        daemon_module.PROVIDER_EXTERNAL_ISOLATION_ENV,
        config.environment_json(),
    )
    command = daemon_module._codex_implementation_command(
        codex="/unsafe/host/codex",
        workspace_path=workspace,
        repository_root=repository,
    )
    mounts = _mounts(command)
    assert (
        f"type=bind,src={host_codex.resolve()},"
        "dst=/usr/local/bin/codex,readonly"
    ) in mounts
    assert (
        f"type=bind,src={host_companion.resolve()},"
        "dst=/usr/local/bin/codex-code-mode-host,readonly"
    ) in mounts
    assert config.container_executable in command
    assert "features.code_mode=false" not in command


def test_grok_command_attaches_trusted_codex_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        daemon_module,
        "_grok_binary",
        lambda: "/home/barberb/.local/bin/grok",
    )
    monkeypatch.setattr(daemon_module, "_grok_cli_available", lambda: True)
    monkeypatch.setattr(
        daemon_module,
        "_host_cli_binary",
        lambda name: "/usr/local/bin/codex" if name == "codex" else None,
    )
    command = daemon_module._grok_cli_command(
        workspace_path=workspace,
        failure_receipt_nonce="ab" * 32,
    )
    assert "--codex-fallback-command-json" in command
    payload = json.loads(
        command[command.index("--codex-fallback-command-json") + 1]
    )
    assert payload[0] == "/usr/local/bin/codex"
    assert "gpt-5.6-terra" in payload
