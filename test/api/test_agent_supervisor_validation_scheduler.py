from __future__ import annotations

import hashlib
import io
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.engine import (
    command_runner_from_legacy_function,
    run_validation_commands,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask,
    TodoImplementationDaemon,
)
from ipfs_accelerate_py.agent_supervisor.validation.validation_commands import (
    ValidationStage,
    build_validation_commands,
    select_validation_commands,
)
from ipfs_accelerate_py.agent_supervisor.validation.validation_runtime import (
    VALIDATION_NPM_CACHE_ENV,
    VALIDATION_PATH_ENV,
    VALIDATION_PLAYWRIGHT_BROWSERS_PATH_ENV,
    VALIDATION_PYTHON_ENV,
    VALIDATION_PYTHON_INTERPRETER_SHA256_ENV,
    VALIDATION_PYTHON_INTERPRETER_STAT_ENV,
    VALIDATION_PYTHON_LAUNCHER_MODE_ENV,
    VALIDATION_PYTHON_LAUNCHER_POLICY_SHA256_ENV,
    VALIDATION_PYTHON_LAUNCHER_SHA256_ENV,
    VALIDATION_PYTHONPATH_ENV,
    VALIDATION_SUPERVISOR_STATE_ROOT_ENV,
    ValidationRuntimeError,
    build_validation_environment,
    build_hermetic_validation_runtime,
    canonical_validation_environment_contract,
    sealed_validation_python_runner,
    validation_argv_command,
    validation_environment_for_runner,
    validation_python_launcher_environment,
    validation_python_executable,
    validation_shell_command,
)
from ipfs_accelerate_py.agent_supervisor.validation.validation_scheduler import (
    HermeticValidationPolicy,
    ValidationResultCache,
    ValidationScheduler,
    _validation_result_digest,
    build_validation_cache_key,
    collect_dependency_state,
    hermetic_validation_runner,
)


def _result(spec, *, returncode: int = 0) -> dict[str, object]:
    return {
        "command": spec.command,
        "raw_command": spec.raw_command,
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:01+00:00",
        "returncode": returncode,
        "output": f"output:{spec.command}",
    }


def _sealed_daemon_environment() -> dict[str, str]:
    return validation_environment_for_runner(
        build_validation_environment(),
        TodoImplementationDaemon._validation_command_runner,
    )


def test_authority_validation_rejects_remote_docker_context_before_probes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DOCKER_CONTEXT", "remote-builder")
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.setattr(
        daemon_module.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "remote Docker context must fail before runtime probes"
        ),
    )
    monkeypatch.setattr(
        daemon_module.shutil,
        "which",
        lambda *_args, **_kwargs: pytest.fail(
            "remote Docker context must fail before executable discovery"
        ),
    )

    contract = (
        TodoImplementationDaemon._authority_validation_isolation_contract()
    )

    assert contract["available"] is False
    assert (
        contract["reason"]
        == "authority_validation_nonlocal_docker_forbidden"
    )
    assert contract["configured_docker_context"] == "remote-builder"
    assert contract["configured_docker_host"] == ""
    assert contract["docker_endpoint"] == "unix:///run/docker.sock"


def test_authority_validation_ignores_path_and_rejects_unapproved_root_binary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_bin = tmp_path / "writable-bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_docker.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_docker.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin))
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    monkeypatch.setattr(
        daemon_module,
        "AUTHORITY_VALIDATION_DOCKER_SHA256",
        "0" * 64,
    )
    monkeypatch.setattr(
        daemon_module.shutil,
        "which",
        lambda *_args, **_kwargs: pytest.fail(
            "authority validation must not discover TCB binaries from PATH"
        ),
    )
    monkeypatch.setattr(
        daemon_module.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "an unapproved root binary must fail before runtime probes"
        ),
    )

    contract = (
        TodoImplementationDaemon._authority_validation_isolation_contract()
    )

    assert contract["available"] is False
    assert contract["reason"] == "authority_validation_tcb_binary_unapproved"
    assert contract["docker_path"] == "/usr/bin/docker"
    assert contract["docker_path"] != str(fake_docker)
    assert contract["docker_sha256"] != "0" * 64


def test_authority_validation_does_not_dispatch_without_isolation_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    unavailable = {
        "available": False,
        "backend": "docker-local-cuda",
        "docker_endpoint": "unix:///run/docker.sock",
        "reason": "authority_validation_docker_contract_unavailable",
        "contract_id": "unavailable-contract",
    }
    monkeypatch.setattr(
        TodoImplementationDaemon,
        "_authority_validation_isolation_contract",
        staticmethod(lambda: unavailable),
    )
    monkeypatch.setattr(
        daemon_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail(
            "validation command dispatched without an isolation contract"
        ),
    )

    result = TodoImplementationDaemon._authority_validation_command_runner(
        spec=SimpleNamespace(command="true", raw_command="true"),
        workspace_path=workspace,
        timeout_seconds=10,
        environment=_sealed_daemon_environment(),
    )

    assert result["returncode"] == 75
    assert result["infrastructure_failure"] is True
    assert result["error"] == "authority_validation_isolation_unavailable"
    assert (
        result["reason"]
        == "authority_validation_docker_contract_unavailable"
    )
    assert result["authority_validation_isolation"] == unavailable
    assert "authority_validation_isolation_receipt" not in result


def test_authority_validation_docker_argv_enforces_local_bounded_read_only_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()
    image_id = f"sha256:{'a' * 64}"
    gpu_uuid = "GPU-11111111-2222-3333-4444-555555555555"
    contract = {
        "available": True,
        "backend": "docker-local-cuda",
        "docker_path": "/usr/bin/docker",
        "docker_endpoint": "unix:///run/docker.sock",
        "image_reference": "registry.invalid/unpinned:latest",
        "image_id": image_id,
        "gpu_uuid": gpu_uuid,
        "contract_id": "pinned-local-cuda-contract",
    }
    monkeypatch.setattr(
        TodoImplementationDaemon,
        "_authority_validation_isolation_contract",
        staticmethod(lambda: contract),
    )
    popen_calls: list[tuple[list[str], dict[str, object]]] = []
    control_calls: list[tuple[list[str], dict[str, object]]] = []

    class CompletedDockerProcess:
        pid = 424242
        returncode = 0

        def __init__(self) -> None:
            self.stdout = io.BytesIO(b"validated\n")

        def poll(self) -> int:
            return self.returncode

    def fake_popen(
        command: list[str],
        **kwargs: object,
    ) -> CompletedDockerProcess:
        popen_calls.append((list(command), dict(kwargs)))
        return CompletedDockerProcess()

    def fake_run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        control_calls.append((list(command), dict(kwargs)))
        return subprocess.CompletedProcess(command, 0, "")

    monkeypatch.setattr(daemon_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(daemon_module.subprocess, "run", fake_run)

    result = TodoImplementationDaemon._authority_validation_command_runner(
        spec=SimpleNamespace(command="true", raw_command="true"),
        workspace_path=workspace,
        timeout_seconds=10,
        environment=_sealed_daemon_environment(),
    )

    assert result["returncode"] == 0
    assert len(popen_calls) == 1
    docker_argv, popen_kwargs = popen_calls[0]
    assert docker_argv[:4] == [
        "/usr/bin/docker",
        "--host",
        "unix:///run/docker.sock",
        "run",
    ]
    assert image_id in docker_argv
    assert contract["image_reference"] not in docker_argv
    assert f"--gpus=device={gpu_uuid}" in docker_argv
    assert "--rm" in docker_argv
    assert "--pull=never" in docker_argv
    assert "--init" in docker_argv
    assert "--network=none" in docker_argv
    assert "--read-only" in docker_argv
    assert "--cap-drop=ALL" in docker_argv
    assert "--security-opt=no-new-privileges:true" in docker_argv
    assert "--log-driver=none" in docker_argv
    assert "--pids-limit=256" in docker_argv
    assert "--cpus=4" in docker_argv
    assert "--memory=4294967296" in docker_argv
    assert "--memory-swap=4294967296" in docker_argv
    assert "--shm-size=1073741824" in docker_argv
    assert (
        "--tmpfs=/tmp:rw,nosuid,nodev,exec,size=1073741824,mode=1777"
        in docker_argv
    )
    assert f"--workdir={workspace}" in docker_argv
    mount_arguments = [
        argument
        for argument in docker_argv
        if argument.startswith("--mount=")
    ]
    assert mount_arguments == [
        f"--mount=type=bind,src={workspace},dst={workspace},readonly"
    ]
    assert "-v" not in docker_argv
    assert "--volume" not in docker_argv
    assert not any(
        argument.startswith(("-v=", "--volume="))
        for argument in docker_argv
    )
    assert not any(
        f"src={broad_path}" in argument
        for broad_path in ("/home", "/usr", "/etc", "/opt")
        for argument in mount_arguments
    )
    assert popen_kwargs["cwd"] == workspace
    assert popen_kwargs["start_new_session"] is (os.name == "posix")
    docker_environment = popen_kwargs["env"]
    assert isinstance(docker_environment, dict)
    assert docker_environment["DOCKER_HOST"] == "unix:///run/docker.sock"
    assert "DOCKER_CONTEXT" not in docker_environment
    assert control_calls
    assert all(
        command[:3]
        == ["/usr/bin/docker", "--host", "unix:///run/docker.sock"]
        for command, _kwargs in control_calls
    )
    receipt = result["authority_validation_isolation_receipt"]
    assert receipt["image_id"] == image_id
    assert receipt["gpu_uuid"] == gpu_uuid
    assert receipt["workspace_path"] == str(workspace)
    assert receipt["workspace_read_only"] is True
    assert receipt["container_root_read_only"] is True
    assert receipt["network_mode"] == "none"
    assert receipt["capabilities_dropped"] == "all"
    assert receipt["cgroup_process_limit"] == 256
    assert receipt["cpu_limit"] == 4
    assert receipt["memory_limit_bytes"] == 4294967296
    assert receipt["tmpfs_limit_bytes"] == 1073741824
    assert receipt["output_limit_bytes"] == 16777216
    assert receipt["output_bounded"] is True
    assert receipt["container_log_driver"] == "none"
    assert receipt["container_removed"] is True
    assert receipt["process_tree_quiesced"] is True


@pytest.mark.parametrize(
    "start_new_session",
    (False, True),
    ids=("same-pgid", "setsid"),
)
def test_authority_validation_descendants_cannot_mutate_workspace_after_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    start_new_session: bool,
) -> None:
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    contract = (
        TodoImplementationDaemon._authority_validation_isolation_contract()
    )
    if contract.get("available") is not True:
        pytest.skip(str(contract.get("reason") or "Docker CUDA unavailable"))
    monkeypatch.setattr(
        TodoImplementationDaemon,
        "_authority_validation_isolation_contract",
        staticmethod(lambda: contract),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    delayed_mutation = workspace / "delayed-mutation.txt"
    immediate_mutation = workspace / "immediate-mutation.txt"
    child_source = (
        "import pathlib, time; "
        "time.sleep(0.75); "
        f"pathlib.Path({str(delayed_mutation)!r}).write_text("
        "'escaped', encoding='utf-8')"
    )
    (workspace / "spawn_descendant.py").write_text(
        "\n".join(
            (
                "import pathlib",
                "import subprocess",
                "import sys",
                f"immediate = pathlib.Path({str(immediate_mutation)!r})",
                "try:",
                "    immediate.write_text('unexpected', encoding='utf-8')",
                "except OSError:",
                "    print('workspace-read-only', flush=True)",
                "else:",
                "    raise SystemExit(91)",
                f"child_source = {child_source!r}",
                "child = subprocess.Popen(",
                "    [sys.executable, '-c', child_source],",
                "    stdin=subprocess.DEVNULL,",
                "    stdout=subprocess.DEVNULL,",
                "    stderr=subprocess.DEVNULL,",
                f"    start_new_session={start_new_session!r},",
                ")",
                "print(f'child={child.pid}', flush=True)",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = TodoImplementationDaemon._authority_validation_command_runner(
        spec=SimpleNamespace(
            command="python spawn_descendant.py",
            raw_command="python spawn_descendant.py",
        ),
        workspace_path=workspace,
        timeout_seconds=30,
        environment=_sealed_daemon_environment(),
    )

    assert result["returncode"] == 0, result["output"]
    assert "workspace-read-only" in str(result["output"])
    assert "child=" in str(result["output"])
    receipt = result["authority_validation_isolation_receipt"]
    assert receipt["workspace_read_only"] is True
    assert receipt["private_pid_namespace"] is True
    assert receipt["container_removed"] is True
    assert receipt["process_tree_quiesced"] is True
    time.sleep(1.0)
    assert not immediate_mutation.exists()
    assert not delayed_mutation.exists()


def test_canonical_validation_environment_contract_ignores_provider_path() -> None:
    provider_only_path = (
        "/home/test/.elan/bin:/home/test/.local/theorem-provers/bin"
    )

    contract = canonical_validation_environment_contract(
        {"PATH": provider_only_path}
    )
    expected = build_validation_environment({"PATH": provider_only_path})

    assert contract["path"] == expected["PATH"]
    assert provider_only_path not in str(contract["path"])
    assert contract["path_entries"] == tuple(
        expected["PATH"].split(os.pathsep)
    )
    assert contract["inherited_path_ignored"] is True
    assert contract["writable_toolchain_paths_rejected"] is True
    assert contract["path_override_active"] is False
    assert contract["path_override_environment_variable"] == (
        VALIDATION_PATH_ENV
    )
    assert contract["python_interpreter"] == expected["PYTHON"]
    assert contract["base_home"] == expected["HOME"]
    assert contract["base_xdg"] == {
        key: expected[key]
        for key in (
            "XDG_CACHE_HOME",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
            "XDG_STATE_HOME",
        )
    }


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def _repo(path: Path) -> str:
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.name", "Validation Test")
    _git(path, "config", "user.email", "validation@example.invalid")
    (path / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='1'\n", encoding="utf-8")
    (path / "src").mkdir()
    (path / "src" / "alpha.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(path, "add", "-A")
    _git(path, "commit", "-qm", "baseline")
    return _git(path, "rev-parse", "HEAD")


def test_validation_runtime_scrubs_hooks_secrets_and_inherited_path(
    tmp_path: Path,
) -> None:
    trusted_bin = Path("/usr/bin").resolve()
    approved_npm_cache = tmp_path / "approved-npm-cache"
    approved_npm_cache.mkdir()
    approved_playwright_browsers = tmp_path / "approved-playwright-browsers"
    approved_playwright_browsers.mkdir()
    source = {
        "AWS_SECRET_ACCESS_KEY": "secret",
        "BASH_ENV": str(tmp_path / "bash-env"),
        "CARGO_HOME": str(tmp_path / "cargo-home"),
        "ENV": str(tmp_path / "env"),
        "GRADLE_USER_HOME": str(tmp_path / "gradle-home"),
        "HOME": str(tmp_path / "home"),
        "NPM_CONFIG_CACHE": str(tmp_path / "inherited-npm-cache"),
        "NPM_CONFIG_OFFLINE": "true",
        "PATH": str(tmp_path / "hostile-bin"),
        "PROMPT_COMMAND": "touch compromised",
        "PYTHON": str(tmp_path / "hostile-python"),
        "RUSTUP_HOME": str(tmp_path / "rustup-home"),
        VALIDATION_NPM_CACHE_ENV: str(approved_npm_cache),
        VALIDATION_PATH_ENV: str(trusted_bin),
        VALIDATION_PLAYWRIGHT_BROWSERS_PATH_ENV: str(
            approved_playwright_browsers
        ),
    }

    environment = build_validation_environment(source)

    assert environment["PATH"] == str(trusted_bin)
    assert environment["HOME"] == "/nonexistent/ipfs-accelerate-validation"
    assert environment["XDG_CONFIG_HOME"] == environment["HOME"]
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHON"] == str(Path(sys.executable).resolve())
    assert environment[VALIDATION_PYTHON_LAUNCHER_MODE_ENV].endswith(
        ":canonical-direct"
    )
    assert (
        len(environment[VALIDATION_PYTHON_LAUNCHER_POLICY_SHA256_ENV])
        == 64
    )
    assert len(environment[VALIDATION_PYTHON_LAUNCHER_SHA256_ENV]) == 64
    assert len(environment[VALIDATION_PYTHON_INTERPRETER_SHA256_ENV]) == 64
    assert environment[VALIDATION_PYTHON_INTERPRETER_STAT_ENV].startswith(
        '{"device":'
    )
    assert environment["NPM_CONFIG_CACHE"] == str(approved_npm_cache.resolve())
    assert environment["NPM_CONFIG_OFFLINE"] == "true"
    assert environment["PLAYWRIGHT_BROWSERS_PATH"] == str(
        approved_playwright_browsers.resolve()
    )
    assert environment["NPM_CONFIG_GLOBALCONFIG"] == "/dev/null"
    assert environment["NPM_CONFIG_USERCONFIG"] == "/dev/null/npmrc"
    assert (
        environment["NPM_CONFIG_USERCONFIG"]
        != environment["NPM_CONFIG_GLOBALCONFIG"]
    )
    assert environment["GIT_TERMINAL_PROMPT"] == "0"
    assert environment["PYTHONHASHSEED"] == "0"
    assert not {
        "AWS_SECRET_ACCESS_KEY",
        "BASH_ENV",
        "CARGO_HOME",
        "ENV",
        "GRADLE_USER_HOME",
        "PROMPT_COMMAND",
        "RUSTUP_HOME",
        VALIDATION_NPM_CACHE_ENV,
        VALIDATION_PATH_ENV,
        VALIDATION_PLAYWRIGHT_BROWSERS_PATH_ENV,
    } & set(environment)
    shell_command = validation_shell_command("test -f artifact")
    assert shell_command[:4] == ["/bin/bash", "--noprofile", "--norc", "-c"]
    assert shell_command[4].endswith(
        "readonly -f _ipfs_accelerate_validation_python python python3 pytest; "
        "test -f artifact"
    )
    for nested_shell in (
        "bash -lc 'python -c \"raise SystemExit(0)\"'",
        "true && bash -lc 'python -V'",
        "command bash -lc 'python -V'",
        ":; /bin/sh -c 'python -V'",
        "env SAFE=1 /bin/bash -c 'python -V'",
        "true&&bash -lc 'python -V'",
        "echo x|bash -lc 'python -V'",
        "true;/bin/sh -c 'python -V'",
    ):
        with pytest.raises(
            ValidationRuntimeError,
            match="nested validation shells",
        ):
            validation_shell_command(nested_shell)
    with pytest.raises(
        ValidationRuntimeError,
        match="must provide command text with -c",
    ):
        validation_argv_command(("/bin/bash", "validation-script.sh"))
    for wrapped_shell in (
        ("env", "SAFE=1", "bash", "-lc", "python -V"),
        ("command", "/bin/sh", "-c", "python -V"),
    ):
        with pytest.raises(
            ValidationRuntimeError,
            match="wrapped validation shells",
        ):
            validation_argv_command(wrapped_shell)
    for dynamic_shell in (
        "echo `bash -lc 'python -V'`",
        "echo $(bash -lc 'python -V')",
        "eval \"bash -lc 'python -V'\"",
    ):
        with pytest.raises(
            ValidationRuntimeError,
            match="dynamic command substitution|dynamic shell evaluation",
        ):
            validation_shell_command(dynamic_shell)

    with pytest.raises(ValidationRuntimeError, match="must be absolute"):
        build_validation_environment({VALIDATION_PATH_ENV: "relative/bin"})
    writable_bin = tmp_path / "writable-bin"
    writable_bin.mkdir()
    with pytest.raises(ValidationRuntimeError, match="must not be writable"):
        build_validation_environment({VALIDATION_PATH_ENV: str(writable_bin)})
    replaceable_bin = tmp_path / "replaceable-bin"
    replaceable_bin.mkdir()
    replaceable_bin.chmod(0o555)
    # A user namespace may report the chmod-555 leaf itself as writable due to
    # its mapped root capability; either the leaf or its replaceable ancestor
    # must still be rejected.
    with pytest.raises(ValidationRuntimeError, match="must not be writable"):
        build_validation_environment({VALIDATION_PATH_ENV: str(replaceable_bin)})


def test_validation_runtime_propagates_only_canonical_readonly_supervisor_state_root(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state_root = tmp_path / "program-state"
    state_root.mkdir()
    source = {
        VALIDATION_SUPERVISOR_STATE_ROOT_ENV: str(state_root),
    }

    environment = build_validation_environment(source)

    assert environment[VALIDATION_SUPERVISOR_STATE_ROOT_ENV] == str(
        state_root.resolve()
    )
    report = ValidationScheduler().run(
        [
            "test \"$LPR_STATE_ROOT\" = "
            f"{shlex.quote(str(state_root.resolve()))}"
        ],
        workspace_path=workspace,
        changed_files=["pyproject.toml"],
        target_commit="test-commit",
        dependency_state="test-dependencies",
        environment=source,
    )
    assert report["passed"] is True

    for command in (
        "LPR_STATE_ROOT=/tmp python -c 'raise SystemExit(0)'",
        "env LPR_STATE_ROOT=/tmp python -c 'raise SystemExit(0)'",
        "export LPR_STATE_ROOT=/tmp; python -c 'raise SystemExit(0)'",
    ):
        with pytest.raises(
            ValidationRuntimeError,
            match="may not override the supervisor state root",
        ):
            validation_shell_command(command)
    for command in (
        "env -i python -c 'raise SystemExit(0)'",
        "env - python -c 'raise SystemExit(0)'",
        "env -iu LPR_STATE_ROOT python -c 'raise SystemExit(0)'",
        "env -u LPR_STATE_ROOT python -c 'raise SystemExit(0)'",
        "env --unset=LPR_STATE_ROOT python -c 'raise SystemExit(0)'",
        "env -S '-u LPR_STATE_ROOT' python -c 'raise SystemExit(0)'",
    ):
        with pytest.raises(
            ValidationRuntimeError,
            match="may not use env options inside the protected environment",
        ):
            validation_shell_command(command)

    with pytest.raises(ValidationRuntimeError, match="must be an absolute directory"):
        build_validation_environment(
            {VALIDATION_SUPERVISOR_STATE_ROOT_ENV: "relative/state"}
        )


def test_real_validation_runner_ignores_profile_bash_env_and_path_injection(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    hostile_bin = tmp_path / "hostile-bin"
    hostile_bin.mkdir()
    profile_marker = tmp_path / "profile-loaded"
    bash_env_marker = tmp_path / "bash-env-loaded"
    path_marker = tmp_path / "path-shadow-ran"
    (home / ".bash_profile").write_text(
        f"touch {shlex.quote(str(profile_marker))}\n",
        encoding="utf-8",
    )
    bash_env = tmp_path / "bash-env"
    bash_env.write_text(
        f"touch {shlex.quote(str(bash_env_marker))}\n",
        encoding="utf-8",
    )
    shadow_python = hostile_bin / "python"
    shadow_python.write_text(
        f"#!/bin/sh\ntouch {shlex.quote(str(path_marker))}\nexit 97\n",
        encoding="utf-8",
    )
    shadow_python.chmod(0o755)
    trusted_path = os.pathsep.join(("/usr/bin", "/bin"))
    hostile_environment = {
        "BASH_ENV": str(bash_env),
        "ENV": str(bash_env),
        "HOME": str(home),
        "PATH": str(hostile_bin),
        "VALIDATION_SECRET": "must-not-leak",
        VALIDATION_PATH_ENV: trusted_path,
    }
    command = (
        f"test ! -e {shlex.quote(str(profile_marker))} "
        f"&& test ! -e {shlex.quote(str(bash_env_marker))} "
        '&& test -z "${VALIDATION_SECRET-}" '
        '&& test "$HOME" = /nonexistent/ipfs-accelerate-validation '
        '&& test "$XDG_CONFIG_HOME" = "$HOME" '
        "&& python -c 'raise SystemExit(0)'"
    )

    report = ValidationScheduler().run(
        [command],
        workspace_path=workspace,
        changed_files=["pyproject.toml"],
        target_commit="test-commit",
        dependency_state="test-dependencies",
        environment=hostile_environment,
    )

    assert report["passed"] is True
    assert not profile_marker.exists()
    assert not bash_env_marker.exists()
    assert not path_marker.exists()
    expected_environment = build_validation_environment(hostile_environment)
    expected_key = build_validation_cache_key(
        target_commit="test-commit",
        command=build_validation_commands([command])[0],
        environment=expected_environment,
        dependency_state="test-dependencies",
        relevant_environment_keys=expected_environment,
    )
    assert report["results"][0]["cache_key"] == expected_key.digest


def test_validation_runtime_reuses_supervisor_python_and_installed_pytest(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    expected_python = str(Path(sys.executable).resolve())
    command = (
        "TASK_PREFIX=works python -c 'import os, sys; "
        "assert os.environ[\"TASK_PREFIX\"] == \"works\"; print(sys.executable)' "
        "&& python -m pytest --version "
        "&& pytest --version"
    )

    report = ValidationScheduler().run(
        [command],
        workspace_path=workspace,
        changed_files=["pyproject.toml"],
        target_commit="test-commit",
        dependency_state="test-dependencies",
    )

    assert report["passed"] is True
    output = str(report["results"][0]["output"])
    assert output.splitlines()[0] == expected_python
    assert output.count("pytest ") == 2
    environment = build_validation_environment()
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert str(Path(pytest.__file__).parent.parent.resolve()) in environment.get(
        "PYTHONPATH", ""
    ).split(os.pathsep)


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="sealed memfd launchers are Linux-specific",
)
def test_validation_runtime_seals_nested_python_launcher_and_cleans_descriptor(
) -> None:
    import fcntl

    environment = _sealed_daemon_environment()
    launcher_path = ""

    with validation_python_launcher_environment(environment) as (
        child_environment,
        receipt,
    ):
        launcher_path = child_environment["PYTHON"]
        payload = Path(launcher_path).read_bytes()
        descriptor = os.open(launcher_path, os.O_RDONLY)
        try:
            required_seals = (
                fcntl.F_SEAL_WRITE
                | fcntl.F_SEAL_GROW
                | fcntl.F_SEAL_SHRINK
                | fcntl.F_SEAL_SEAL
            )
            assert (
                fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)
                & required_seals
                == required_seals
            )
        finally:
            os.close(descriptor)
        assert receipt.sealed is True
        assert receipt.executable == launcher_path
        assert receipt.content_sha256 == hashlib.sha256(payload).hexdigest()
        assert (
            receipt.interpreter_sha256
            == child_environment[
                VALIDATION_PYTHON_INTERPRETER_SHA256_ENV
            ]
        )
        assert (
            receipt.interpreter_stat
            == child_environment[
                VALIDATION_PYTHON_INTERPRETER_STAT_ENV
            ]
        )
        assert (
            receipt.mode
            == child_environment[VALIDATION_PYTHON_LAUNCHER_MODE_ENV]
        )
        assert (
            receipt.policy_sha256
            == child_environment[
                VALIDATION_PYTHON_LAUNCHER_POLICY_SHA256_ENV
            ]
        )
        assert (
            child_environment[
                VALIDATION_PYTHON_LAUNCHER_SHA256_ENV
            ]
            == receipt.content_sha256
        )
        assert child_environment["PYTHONNOUSERSITE"] == "1"

    assert launcher_path
    assert not Path(launcher_path).exists()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="sealed memfd launchers are Linux-specific",
)
def test_daemon_nested_python_keeps_approved_packages_after_pythonpath_replace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "fixture_value.py").write_text(
        "VALUE = 'workspace-import'\n",
        encoding="utf-8",
    )
    (workspace / "hostile-bash-env").write_text(
        "touch bash-env-executed\nexit 42\n",
        encoding="utf-8",
    )
    (workspace / "nested_launcher.py").write_text(
        """
import os
import pathlib
import subprocess

environment = dict(os.environ)
environment["PYTHONPATH"] = str(pathlib.Path.cwd())
environment["PYTHONNOUSERSITE"] = "0"
environment["BASH_ENV"] = str(pathlib.Path.cwd() / "hostile-bash-env")
pathlib.Path("launcher-path.txt").write_text(
    environment["PYTHON"],
    encoding="utf-8",
)
completed = subprocess.run(
    [
        environment["PYTHON"],
        "-c",
        (
            "import os, site, fixture_value, pytest; "
            "assert os.environ['PYTHONNOUSERSITE'] == '1'; "
            "assert site.ENABLE_USER_SITE is False; "
            "assert fixture_value.VALUE == 'workspace-import'; "
            "print(pytest.__version__)"
        ),
    ],
    env=environment,
    text=True,
    capture_output=True,
    check=False,
)
site_probe = subprocess.run(
    [
        environment["PYTHON"],
        "-E",
        "-c",
        (
            "import site; "
            "assert site.ENABLE_USER_SITE is False"
        ),
    ],
    env=environment,
    text=True,
    capture_output=True,
    check=False,
)
print(completed.stdout, end="")
print(completed.stderr, end="")
print(site_probe.stdout, end="")
print(site_probe.stderr, end="")
raise SystemExit(completed.returncode or site_probe.returncode)
""".lstrip(),
        encoding="utf-8",
    )
    spec = SimpleNamespace(
        command="python nested_launcher.py",
        raw_command="python nested_launcher.py",
    )

    result = TodoImplementationDaemon._validation_command_runner(
        spec=spec,
        workspace_path=workspace,
        timeout_seconds=30,
        environment=_sealed_daemon_environment(),
    )

    assert result["returncode"] == 0, result["output"]
    assert pytest.__version__ in str(result["output"])
    assert not (workspace / "bash-env-executed").exists()
    launcher_receipt = result["validation_python_launcher"]
    assert launcher_receipt["sealed"] is True
    environment = _sealed_daemon_environment()
    assert (
        launcher_receipt["content_sha256"]
        == environment[VALIDATION_PYTHON_LAUNCHER_SHA256_ENV]
    )
    assert (
        launcher_receipt["mode"]
        == environment[VALIDATION_PYTHON_LAUNCHER_MODE_ENV]
    )
    assert (
        launcher_receipt["policy_sha256"]
        == environment[
            VALIDATION_PYTHON_LAUNCHER_POLICY_SHA256_ENV
        ]
    )
    assert (
        launcher_receipt["interpreter_sha256"]
        == environment[VALIDATION_PYTHON_INTERPRETER_SHA256_ENV]
    )
    assert (
        launcher_receipt["interpreter_stat"]
        == environment[VALIDATION_PYTHON_INTERPRETER_STAT_ENV]
    )
    launcher_path = (workspace / "launcher-path.txt").read_text(
        encoding="utf-8"
    )
    assert launcher_path.startswith(f"/proc/{os.getpid()}/fd/")
    assert not Path(launcher_path).exists()


def test_daemon_classifies_python_launcher_failure_as_infrastructure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class FailedLauncher:
        def __enter__(self):
            raise ValidationRuntimeError("kernel sealing unavailable")

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(
        (
            "ipfs_accelerate_py.agent_supervisor.todo_daemon."
            "implementation_daemon.validation_python_launcher_environment"
        ),
        lambda _environment: FailedLauncher(),
    )
    spec = SimpleNamespace(command="true", raw_command="true")

    result = TodoImplementationDaemon._validation_command_runner(
        spec=spec,
        workspace_path=workspace,
        timeout_seconds=30,
        environment=_sealed_daemon_environment(),
    )

    assert result["returncode"] == 75
    assert result["infrastructure_failure"] is True
    assert (
        result["error"]
        == "validation_environment_python_launcher_unavailable"
    )
    assert (
        result["reason"]
        == "sealed_validation_python_launcher_unavailable"
    )
    assert "kernel sealing unavailable" in str(result["output"])


def test_daemon_classifies_child_launcher_exec_denial_as_infrastructure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    environment = _sealed_daemon_environment()

    def denied_run(*_args: object, **_kwargs: object):
        raise PermissionError("procfd execution denied")

    monkeypatch.setattr(
        (
            "ipfs_accelerate_py.agent_supervisor.todo_daemon."
            "implementation_daemon.subprocess.run"
        ),
        denied_run,
    )
    spec = SimpleNamespace(command="true", raw_command="true")

    result = TodoImplementationDaemon._validation_command_runner(
        spec=spec,
        workspace_path=workspace,
        timeout_seconds=30,
        environment=environment,
    )

    assert result["returncode"] == 75
    assert result["infrastructure_failure"] is True
    assert (
        result["error"]
        == "validation_environment_python_launcher_exec_unavailable"
    )
    assert (
        result["reason"]
        == "sealed_validation_python_launcher_child_probe_failed"
    )
    assert "procfd execution denied" in str(result["output"])
    receipt = result["validation_python_launcher"]
    assert (
        receipt["content_sha256"]
        == environment[VALIDATION_PYTHON_LAUNCHER_SHA256_ENV]
    )
    assert (
        receipt["interpreter_sha256"]
        == environment[VALIDATION_PYTHON_INTERPRETER_SHA256_ENV]
    )
    assert (
        receipt["interpreter_stat"]
        == environment[VALIDATION_PYTHON_INTERPRETER_STAT_ENV]
    )
    assert (
        receipt["mode"]
        == environment[VALIDATION_PYTHON_LAUNCHER_MODE_ENV]
    )
    assert (
        receipt["policy_sha256"]
        == environment[
            VALIDATION_PYTHON_LAUNCHER_POLICY_SHA256_ENV
        ]
    )


@pytest.mark.parametrize(
    "command",
    ("bash -c 'true'", "eval true"),
)
def test_daemon_classifies_invalid_shell_command_as_policy_rejection(
    tmp_path: Path,
    command: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = SimpleNamespace(command=command, raw_command=command)

    result = TodoImplementationDaemon._validation_command_runner(
        spec=spec,
        workspace_path=workspace,
        timeout_seconds=30,
        environment=_sealed_daemon_environment(),
    )

    assert result["returncode"] == 78
    assert result["error"] == "validation_command_policy_rejected"
    assert (
        result["reason"]
        == "validation_shell_command_policy_violation"
    )
    assert result["infrastructure_failure"] is False
    assert "validation_python_launcher" not in result


def test_validation_runtime_extends_task_local_pythonpath_with_approved_packages(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "fixture_value.py").write_text(
        "VALUE = 'workspace-import'\n",
        encoding="utf-8",
    )
    command = (
        "PYTHONPATH=. python -c 'import fixture_value, pytest; "
        "assert fixture_value.VALUE == \"workspace-import\"; "
        "print(pytest.__version__)' "
        "&& PYTHONPATH=. pytest --version"
    )

    report = ValidationScheduler().run(
        [command],
        workspace_path=workspace,
        changed_files=["fixture_value.py"],
        target_commit="test-commit",
        dependency_state="test-dependencies",
    )

    assert report["passed"] is True
    output = str(report["results"][0]["output"])
    assert pytest.__version__ in output
    assert "pytest " in output


def test_validation_runtime_canonicalizes_replaceable_python_launcher(
    tmp_path: Path,
) -> None:
    interpreter = tmp_path / "replaceable-python"
    interpreter.symlink_to(Path(sys.executable).resolve())
    environment = {
        VALIDATION_PATH_ENV: os.pathsep.join(("/usr/bin", "/bin")),
        VALIDATION_PYTHON_ENV: str(interpreter),
    }

    report = ValidationScheduler().run(
        ["python -c 'import sys; print(sys.executable)'"],
        workspace_path=tmp_path,
        changed_files=["pyproject.toml"],
        target_commit="test-commit",
        dependency_state="test-dependencies",
        environment=environment,
    )

    assert report["passed"] is True
    assert str(report["results"][0]["output"]).strip() == str(
        Path(sys.executable).resolve()
    )
    child_environment = build_validation_environment(environment)
    assert child_environment[
        "IPFS_ACCELERATE_VALIDATION_PYTHON_EXECUTABLE"
    ] == str(Path(sys.executable).resolve())
    assert "PYTHONPATH" not in child_environment
    assert child_environment["PYTHONNOUSERSITE"] == "1"
    assert validation_python_executable(environment) != str(interpreter)


def test_validation_runtime_does_not_reinject_inherited_pythonpath(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hostile = tmp_path / "hostile" / "site-packages"
    hostile.mkdir(parents=True)
    monkeypatch.setattr(sys, "path", [str(hostile), *sys.path])

    environment = build_validation_environment()

    assert str(hostile.resolve()) not in environment.get("PYTHONPATH", "").split(
        os.pathsep
    )
    with pytest.raises(ValidationRuntimeError, match="must not be writable"):
        build_validation_environment(
            {VALIDATION_PYTHONPATH_ENV: str(hostile)}
        )


def test_legacy_argv_validation_normalizes_login_shell_and_scrubs_bash_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "bash-env-ran"
    bash_env = tmp_path / "bash-env"
    bash_env.write_text(
        f"touch {shlex.quote(str(marker))}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("BASH_ENV", str(bash_env))

    results = run_validation_commands(
        repo_root=tmp_path,
        commands=(
            (
                "/bin/bash",
                "-lc",
                "python -c 'import sys; print(sys.executable)'",
            ),
        ),
        timeout_seconds=10,
    )

    assert results[0].ok
    assert results[0].command[:4] == (
        "/bin/bash",
        "--noprofile",
        "--norc",
        "-c",
    )
    assert results[0].stdout.strip() == str(Path(sys.executable).resolve())
    assert not marker.exists()


def test_legacy_adapter_forwards_sanitized_validation_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "bash-env-ran"
    bash_env = tmp_path / "bash-env"
    bash_env.write_text(
        f"touch {shlex.quote(str(marker))}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("BASH_ENV", str(bash_env))
    monkeypatch.setenv("VALIDATION_SECRET", "must-not-leak")
    captured_environment: dict[str, str] = {}

    def legacy_runner(
        command,
        *,
        cwd,
        timeout,
        input_text=None,
        environment=None,
    ):
        assert environment is not None
        captured_environment.update(
            {str(key): str(value) for key, value in environment.items()}
        )
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=captured_environment,
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": list(command),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    results = run_validation_commands(
        repo_root=tmp_path,
        commands=(("/bin/bash", "-lc", "python -c 'print(\"safe\")'"),),
        timeout_seconds=10,
        run_command_fn=command_runner_from_legacy_function(legacy_runner),
    )

    assert results[0].ok
    assert results[0].stdout.strip() == "safe"
    assert not marker.exists()
    assert "BASH_ENV" not in captured_environment
    assert "VALIDATION_SECRET" not in captured_environment


def test_legacy_adapter_rejects_runner_without_environment_contract() -> None:
    def unsafe_legacy_runner(command, *, cwd, timeout, input_text=None):
        return {
            "command": command,
            "returncode": 0,
            "stdout": "",
            "stderr": "",
        }

    with pytest.raises(
        ValidationRuntimeError,
        match="must accept an environment keyword",
    ):
        command_runner_from_legacy_function(unsafe_legacy_runner)


def test_cheap_checks_run_before_expensive_tests_and_fail_fast(tmp_path: Path) -> None:
    calls: list[str] = []

    def runner(*, spec, **_kwargs):
        calls.append(spec.command)
        return _result(spec, returncode=7 if spec.stage == ValidationStage.CHEAP else 0)

    scheduler = ValidationScheduler(max_workers=2, resource_budget=2, runner=runner)
    report = scheduler.run(
        ["pytest tests/test_alpha.py", "git diff --check"],
        workspace_path=tmp_path,
        changed_files=["src/alpha.py"],
        target_commit="abc",
        dependency_state="deps",
    )

    assert calls == ["git diff --check"]
    assert report["passed"] is False
    assert report["returncode"] == 7
    assert report["failed_command"] == "git diff --check"
    assert [item["stage"] for item in report["stages"]] == ["cheap"]


def test_independent_validations_run_in_parallel_under_weighted_budget(tmp_path: Path) -> None:
    lock = threading.Lock()
    release = threading.Event()
    two_running = threading.Event()
    active = 0
    maximum_active = 0

    def runner(*, spec, **_kwargs):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
            if active == 2:
                two_running.set()
        assert release.wait(timeout=5)
        with lock:
            active -= 1
        return _result(spec)

    scheduler = ValidationScheduler(max_workers=3, resource_budget=2, runner=runner)
    commands = [
        "pytest tests/test_alpha.py",
        "pytest tests/test_beta.py",
        "pytest tests/test_gamma.py",
    ]
    outcome: dict[str, object] = {}

    def schedule() -> None:
        outcome.update(
            scheduler.run(
                commands,
                workspace_path=tmp_path,
                changed_files=["pyproject.toml"],
                target_commit="abc",
                dependency_state="deps",
            )
        )

    thread = threading.Thread(target=schedule)
    thread.start()
    assert two_running.wait(timeout=5)
    release.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert outcome["passed"] is True
    assert maximum_active == 2
    assert [item["command"] for item in outcome["results"]] == commands


def test_cache_and_result_digests_bind_python_launcher_policy() -> None:
    canonical_environment = build_validation_environment()
    environment = _sealed_daemon_environment()
    base_key = build_validation_cache_key(
        target_commit="commit-a",
        command="pytest tests/test_alpha.py",
        environment=environment,
        dependency_state={"lock": "one"},
    )
    canonical_key = build_validation_cache_key(
        target_commit="commit-a",
        command="pytest tests/test_alpha.py",
        environment=canonical_environment,
        dependency_state={"lock": "one"},
    )
    assert canonical_key.digest != base_key.digest

    for variable in (
        VALIDATION_PYTHON_INTERPRETER_SHA256_ENV,
        VALIDATION_PYTHON_INTERPRETER_STAT_ENV,
        VALIDATION_PYTHON_LAUNCHER_MODE_ENV,
        VALIDATION_PYTHON_LAUNCHER_POLICY_SHA256_ENV,
        VALIDATION_PYTHON_LAUNCHER_SHA256_ENV,
    ):
        changed = dict(environment)
        changed[variable] = f"{changed[variable]}-changed"
        variant = build_validation_cache_key(
            target_commit="commit-a",
            command="pytest tests/test_alpha.py",
            environment=changed,
            dependency_state={"lock": "one"},
        )
        assert variant.digest != base_key.digest

    result = {
        "command": "pytest tests/test_alpha.py",
        "returncode": 0,
        "output": "passed",
        "validation_python_launcher": {
            "content_sha256": environment[
                VALIDATION_PYTHON_LAUNCHER_SHA256_ENV
            ],
            "interpreter_sha256": environment[
                VALIDATION_PYTHON_INTERPRETER_SHA256_ENV
            ],
            "interpreter_stat": environment[
                VALIDATION_PYTHON_INTERPRETER_STAT_ENV
            ],
            "mode": environment[VALIDATION_PYTHON_LAUNCHER_MODE_ENV],
            "policy_sha256": environment[
                VALIDATION_PYTHON_LAUNCHER_POLICY_SHA256_ENV
            ],
            "sealed": True,
        },
    }
    base_result_digest = _validation_result_digest(
        result,
        cache_key=base_key,
    )
    changed_result = {
        **result,
        "validation_python_launcher": {
            **result["validation_python_launcher"],
            "sealed": False,
        },
    }
    assert (
        _validation_result_digest(changed_result, cache_key=base_key)
        != base_result_digest
    )


def test_validation_cache_separates_canonical_and_sealed_runners(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cache_dir = tmp_path / "cache"
    calls: list[str] = []

    def canonical_runner(*, spec, **_kwargs):
        calls.append("canonical")
        return _result(spec)

    @sealed_validation_python_runner
    def sealed_runner(*, spec, environment, **_kwargs):
        calls.append("sealed")
        result = _result(spec)
        result["validation_python_launcher"] = {
            "content_sha256": environment[
                VALIDATION_PYTHON_LAUNCHER_SHA256_ENV
            ],
            "interpreter_sha256": environment[
                VALIDATION_PYTHON_INTERPRETER_SHA256_ENV
            ],
            "interpreter_stat": environment[
                VALIDATION_PYTHON_INTERPRETER_STAT_ENV
            ],
            "mode": environment[VALIDATION_PYTHON_LAUNCHER_MODE_ENV],
            "policy_sha256": environment[
                VALIDATION_PYTHON_LAUNCHER_POLICY_SHA256_ENV
            ],
            "sealed": True,
        }
        return result

    common = {
        "workspace_path": workspace,
        "changed_files": ["pyproject.toml"],
        "target_commit": "same-commit",
        "dependency_state": "same-dependencies",
    }
    canonical = ValidationScheduler(
        cache_dir=cache_dir,
        runner=canonical_runner,
    ).run(["true"], **common)
    sealed_environment = validation_environment_for_runner(
        build_validation_environment(),
        sealed_runner,
    )
    sealed_key = build_validation_cache_key(
        target_commit="same-commit",
        command="true",
        environment=sealed_environment,
        dependency_state="same-dependencies",
    )
    stale_without_receipt = {
        "command": "true",
        "raw_command": "true",
        "returncode": 0,
        "output": "stale",
    }
    assert ValidationResultCache(cache_dir).put(
        sealed_key,
        stale_without_receipt,
    )
    first_sealed = ValidationScheduler(
        cache_dir=cache_dir,
        runner=sealed_runner,
    ).run(["true"], **common)
    replayed_sealed = ValidationScheduler(
        cache_dir=cache_dir,
        runner=sealed_runner,
    ).run(["true"], **common)

    assert canonical["results"][0]["cache_hit"] is False
    assert first_sealed["results"][0]["cache_hit"] is False
    assert replayed_sealed["results"][0]["cache_hit"] is True
    assert (
        canonical["results"][0]["cache_key"]
        != first_sealed["results"][0]["cache_key"]
    )
    assert calls == ["canonical", "sealed"]


def test_fresh_sealed_runner_requires_exact_launcher_receipt(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    @sealed_validation_python_runner
    def missing_receipt_runner(*, spec, **_kwargs):
        return _result(spec)

    report = ValidationScheduler(runner=missing_receipt_runner).run(
        ["true"],
        workspace_path=workspace,
        changed_files=["pyproject.toml"],
        target_commit="same-commit",
        dependency_state="same-dependencies",
    )

    result = report["results"][0]
    assert result["returncode"] == 75
    assert result["infrastructure_failure"] is True
    assert (
        result["error"]
        == "validation_environment_python_launcher_receipt_mismatch"
    )
    assert result["outcome"] == "infrastructure_failure"
    assert result["classification"] == "infrastructure_failure"
    assert result["authoritative"] is False
    assert result["stable"] is False


def test_hermetic_runtime_always_resanitizes_supplied_environment(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime = build_hermetic_validation_runtime(
        command="true",
        workspace_path=workspace,
        repository_tree_id="tree:test",
        environment={
            "AWS_SECRET_ACCESS_KEY": "must-not-survive",
            "BASH_ENV": str(tmp_path / "hostile-startup-hook"),
        },
        timeout_seconds=30,
        cancellation_id="validation:test",
        isolation_executable="/bin/true",
    )

    environment = dict(runtime.environment)
    assert "AWS_SECRET_ACCESS_KEY" not in environment
    assert "BASH_ENV" not in environment


def test_hermetic_scheduler_rejects_actual_daemon_runner_before_execution(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    report = ValidationScheduler(
        runner=TodoImplementationDaemon._validation_command_runner,
        hermetic_policy=HermeticValidationPolicy(),
    ).run(
        ['test -z "${IPFS_ACCELERATE_VALIDATION_RUNTIME_ID-}"'],
        workspace_path=workspace,
        changed_files=["pyproject.toml"],
        target_commit="same-commit",
        dependency_state="same-dependencies",
    )

    assert report["passed"] is False
    result = report["results"][0]
    assert result["returncode"] == 75
    assert (
        result["error"]
        == "hermetic_validation_runner_capability_missing"
    )
    assert result["reason"] == (
        "hermetic_runner_does_not_consume_runtime_context"
    )
    assert result["outcome"] == "infrastructure_failure"
    assert result["classification"] == "infrastructure_failure"
    assert result["authoritative"] is False
    assert result["stable"] is False


def test_marked_hermetic_runner_must_return_exact_runtime_receipt(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    @hermetic_validation_runner
    def receiptless_runner(*, spec, runtime_context, **_kwargs):
        assert runtime_context is not None
        return _result(spec)

    report = ValidationScheduler(
        runner=receiptless_runner,
        hermetic_policy=HermeticValidationPolicy(),
    ).run(
        ["true"],
        workspace_path=workspace,
        changed_files=["pyproject.toml"],
        target_commit="same-commit",
        dependency_state="same-dependencies",
    )

    assert report["passed"] is False
    result = report["results"][0]
    assert result["returncode"] == 75
    assert result["error"] == "hermetic_runtime_receipt_mismatch"
    assert result["outcome"] == "infrastructure_failure"
    assert result["authoritative"] is False
    assert result["stable"] is False
    assert result["attempts"][0]["observed_runtime_id"] == ""
    assert result["attempts"][0]["expected_runtime_id"]
    assert result.get("runtime_id", "") == ""
    assert result.get("cancellation_id", "") == ""


@pytest.mark.parametrize("stale_attempt_count", (None, "not-a-number"))
def test_hermetic_cache_rejects_success_without_exact_runtime_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stale_attempt_count: object,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cache = ValidationResultCache(tmp_path / "cache")
    monkeypatch.setattr(
        cache,
        "get",
        lambda _key: {
            "returncode": 0,
            "output": "forged stale cache authority",
            "attempt_count": stale_attempt_count,
        },
    )
    calls: list[int] = []

    @hermetic_validation_runner
    def runner(*, spec, runtime_context, attempt_number, **_kwargs):
        calls.append(attempt_number)
        result = _result(spec)
        result.update(
            {
                "runtime_id": runtime_context.runtime_id,
                "cancellation_id": runtime_context.cancellation_id,
            }
        )
        return result

    report = ValidationScheduler(
        cache=cache,
        runner=runner,
        hermetic_policy=HermeticValidationPolicy(),
    ).run(
        ["true"],
        workspace_path=workspace,
        changed_files=["pyproject.toml"],
        target_commit="same-commit",
        dependency_state="same-dependencies",
    )

    assert report["passed"] is True
    assert calls == [1, 2]
    result = report["results"][0]
    assert result["cache_hit"] is False
    assert result["authoritative"] is True
    assert result["runtime_id"]
    assert result["cancellation_id"]
    base_digest = _validation_result_digest(result)
    tampered = dict(result)
    tampered["runtime_id"] = f"{result['runtime_id']}-tampered"
    assert _validation_result_digest(tampered) != base_digest


def test_hermetic_cache_reuses_exact_runtime_receipts(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    calls: list[int] = []

    @hermetic_validation_runner
    def runner(*, spec, runtime_context, attempt_number, **_kwargs):
        calls.append(attempt_number)
        result = _result(spec)
        result.update(
            {
                "runtime_id": runtime_context.runtime_id,
                "cancellation_id": runtime_context.cancellation_id,
            }
        )
        return result

    scheduler = ValidationScheduler(
        cache_dir=tmp_path / "cache",
        runner=runner,
        hermetic_policy=HermeticValidationPolicy(),
    )
    common = {
        "workspace_path": workspace,
        "changed_files": ["pyproject.toml"],
        "target_commit": "same-commit",
        "dependency_state": "same-dependencies",
    }

    first = scheduler.run(["true"], **common)
    replay = scheduler.run(["true"], **common)

    assert first["passed"] is True
    assert first["results"][0]["cache_hit"] is False
    assert replay["passed"] is True
    assert replay["results"][0]["cache_hit"] is True
    assert replay["results"][0]["authoritative"] is True
    assert calls == [1, 2]


def test_hermetic_scheduler_rejects_dual_sealed_runner_composition(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    calls: list[str] = []

    @sealed_validation_python_runner
    @hermetic_validation_runner
    def dual_runner(*, spec, runtime_context, **_kwargs):
        calls.append(spec.command)
        result = _result(spec)
        result.update(
            {
                "runtime_id": runtime_context.runtime_id,
                "cancellation_id": runtime_context.cancellation_id,
            }
        )
        return result

    report = ValidationScheduler(
        runner=dual_runner,
        hermetic_policy=HermeticValidationPolicy(),
    ).run(
        ["true"],
        workspace_path=workspace,
        changed_files=["pyproject.toml"],
        target_commit="same-commit",
        dependency_state="same-dependencies",
    )

    assert calls == []
    assert report["passed"] is False
    result = report["results"][0]
    assert result["returncode"] == 75
    assert (
        result["error"]
        == "hermetic_sealed_runner_composition_unsupported"
    )
    assert result["outcome"] == "infrastructure_failure"
    assert result["authoritative"] is False


def test_cache_key_includes_commit_command_relevant_environment_and_dependencies() -> None:
    base = build_validation_cache_key(
        target_commit="commit-a",
        command="pytest tests/test_alpha.py",
        environment={"PYTHONPATH": "src", "IGNORED_SECRET": "one"},
        dependency_state={"lock": "one"},
    )
    same = build_validation_cache_key(
        target_commit="commit-a",
        command="pytest tests/test_alpha.py",
        environment={"IGNORED_SECRET": "two", "PYTHONPATH": "src"},
        dependency_state={"lock": "one"},
    )

    assert base.digest == same.digest
    variants = [
        build_validation_cache_key(
            target_commit="commit-b",
            command="pytest tests/test_alpha.py",
            environment={"PYTHONPATH": "src"},
            dependency_state={"lock": "one"},
        ),
        build_validation_cache_key(
            target_commit="commit-a",
            command="pytest tests/test_beta.py",
            environment={"PYTHONPATH": "src"},
            dependency_state={"lock": "one"},
        ),
        build_validation_cache_key(
            target_commit="commit-a",
            command="pytest tests/test_alpha.py",
            environment={"PYTHONPATH": "lib"},
            dependency_state={"lock": "one"},
        ),
        build_validation_cache_key(
            target_commit="commit-a",
            command="pytest tests/test_alpha.py",
            environment={"PYTHONPATH": "src"},
            dependency_state={"lock": "two"},
        ),
    ]
    assert all(item.digest != base.digest for item in variants)
    assert base.to_dict()["target_commit"] == "commit-a"


def test_success_cache_is_durable_and_dirty_or_dependency_content_invalidates(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    commit = _repo(repo)
    calls = 0

    def runner(*, spec, **_kwargs):
        nonlocal calls
        calls += 1
        return _result(spec)

    cache_dir = tmp_path / "cache"
    command = "pytest tests/test_alpha.py"
    scheduler = ValidationScheduler(cache_dir=cache_dir, runner=runner)
    first = scheduler.run(
        [command],
        workspace_path=repo,
        changed_files=["src/alpha.py"],
        target_commit=commit,
    )
    second = ValidationScheduler(cache_dir=cache_dir, runner=runner).run(
        [command],
        workspace_path=repo,
        changed_files=["src/alpha.py"],
        target_commit=commit,
    )

    assert calls == 1
    assert first["cache_misses"] == 1
    assert second["cache_hits"] == 1

    (repo / "src" / "alpha.py").write_text("VALUE = 2\n", encoding="utf-8")
    dirty = ValidationScheduler(cache_dir=cache_dir, runner=runner).run(
        [command],
        workspace_path=repo,
        changed_files=["src/alpha.py"],
        target_commit=commit,
    )
    assert calls == 2
    assert dirty["cache_hits"] == 0

    (repo / "pyproject.toml").write_text(
        "[project]\nname='fixture'\nversion='2'\n", encoding="utf-8"
    )
    dependency_changed = ValidationScheduler(cache_dir=cache_dir, runner=runner).run(
        [command],
        workspace_path=repo,
        changed_files=["src/alpha.py"],
        target_commit=commit,
    )
    assert calls == 3
    assert dependency_changed["cache_hits"] == 0


def test_failures_are_not_cached(tmp_path: Path) -> None:
    spec = build_validation_commands(["git diff --check"])[0]
    cache = ValidationResultCache(tmp_path / "cache")
    key = build_validation_cache_key(
        target_commit="abc", command=spec, dependency_state="deps", environment={}
    )

    assert cache.put(key, {"returncode": 1}) is False
    assert cache.get(key) is None


def test_impact_selection_is_explainable_and_dependency_changes_are_conservative() -> None:
    commands = [
        "git diff --check",
        "pytest tests/test_alpha.py",
        "pytest tests/test_beta.py",
        "custom-validation --all",
    ]
    narrow = select_validation_commands(commands, ["src/alpha.py"])
    decisions = {item.spec.command: item for item in narrow.items}

    assert decisions["git diff --check"].selected is True
    assert decisions["pytest tests/test_alpha.py"].selected is True
    assert decisions["pytest tests/test_alpha.py"].matched_paths == ("src/alpha.py",)
    assert decisions["pytest tests/test_beta.py"].selected is False
    assert decisions["pytest tests/test_beta.py"].reason == "no_changed_path_matches_command_target"
    assert decisions["custom-validation --all"].selected is True
    assert decisions["custom-validation --all"].reason == "global_or_unknown_impact"

    broad = select_validation_commands(commands, ["pyproject.toml"])
    assert all(item.selected for item in broad.items)
    assert broad.to_dict()["changed_files"] == ["pyproject.toml"]

    prefixed_test = select_validation_commands(
        ["pytest test/api/test_agent_supervisor_validation_scheduler.py"],
        ["ipfs_accelerate_py/agent_supervisor/validation_scheduler.py"],
    )
    assert prefixed_test.items[0].selected is True
    assert prefixed_test.items[0].reason == "changed_path_matches_command_target"

    ci_change = select_validation_commands(commands, [".github/workflows/test.yml"])
    assert all(item.selected for item in ci_change.items)


def test_pre_merge_escalation_runs_unrelated_targeted_validation(tmp_path: Path) -> None:
    calls: list[str] = []

    def runner(*, spec, **_kwargs):
        calls.append(spec.command)
        return _result(spec, returncode=9 if "beta" in spec.command else 0)

    report = ValidationScheduler(max_workers=2, resource_budget=2, runner=runner).run(
        ["pytest tests/test_alpha.py", "pytest tests/test_beta.py"],
        workspace_path=tmp_path,
        changed_files=["src/alpha.py"],
        target_commit="abc",
        dependency_state="deps",
        require_full_validation=True,
        scope="pre_merge",
    )

    assert set(calls) == {"pytest tests/test_alpha.py", "pytest tests/test_beta.py"}
    assert report["passed"] is False
    assert report["selection"]["escalated"] is True
    beta = next(
        item for item in report["selection"]["decisions"] if "beta" in item["command"]
    )
    assert beta["reason"] == "pre_merge_broad_escalation"
    assert beta["stage"] == "broad"


def test_dependency_state_records_candidate_content_not_only_head(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _repo(repo)
    before = collect_dependency_state(repo, changed_files=["src/alpha.py"])
    (repo / "src" / "alpha.py").write_text("VALUE = 99\n", encoding="utf-8")
    after = collect_dependency_state(repo, changed_files=["src/alpha.py"])

    assert before["candidate_content_sha256"] != after["candidate_content_sha256"]


def test_daemon_uses_full_pre_merge_scope_and_preserves_result_contract(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class Scheduler:
        def run(self, commands, **kwargs):
            captured["commands"] = list(commands)
            captured.update(kwargs)
            return {
                "attempted": True,
                "passed": False,
                "returncode": 6,
                "results": [
                    {
                        "command": commands[0],
                        "returncode": 6,
                        "stage": "cheap",
                        "output": "failed\n",
                    }
                ],
                "failed_command": commands[0],
                "selection": {"scope": "pre_merge", "changed_files": ["src/a.py"]},
            }

    daemon = TodoImplementationDaemon(
        todo_path=tmp_path / "todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=tmp_path,
        validation_scheduler=Scheduler(),  # type: ignore[arg-type]
    )
    task = PortalTask(
        task_id="REF-043",
        title="validation scheduler",
        status="todo",
        completion="manual",
        priority="P1",
        track="g9",
        validation=["git diff --check"],
    )

    report = daemon._run_validation_commands(tmp_path, task, tmp_path / "validation.log")

    assert captured["commands"] == ["git diff --check"]
    assert captured["require_full_validation"] is True
    assert captured["scope"] == "pre_merge"
    assert callable(captured["runner"])
    assert report["attempted"] is True
    assert report["passed"] is False
    assert report["returncode"] == 6
    assert report["failed_command"] == "git diff --check"


def test_compound_bare_diff_check_covers_committed_candidate_from_baseline(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    baseline = _repo(repo)
    (repo / "src" / "alpha.py").write_text(
        "VALUE = 1  \n",
        encoding="utf-8",
    )
    _git(repo, "add", "src/alpha.py")
    _git(repo, "commit", "-qm", "model-created clean candidate")
    assert _git(repo, "status", "--porcelain") == ""
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state.json",
        strategy_path=repo / "strategy.json",
        events_path=repo / "events.jsonl",
        repo_root=repo,
        validation_cache_dir=repo / "validation-cache",
        merge_queue_dir=repo / "merge-queue",
    )
    declared_validation = (
        "test -f src/alpha.py && rg -q 'VALUE' src/alpha.py && "
        "rg -qi 'value' src/alpha.py && git diff --check"
    )
    task = PortalTask(
        task_id="REF-043",
        title="committed candidate whitespace",
        status="todo",
        completion="manual",
        priority="P1",
        track="validation",
        validation=[declared_validation],
    )

    assert daemon._declares_bare_git_diff_check(task.validation) is True
    assert daemon._declares_bare_git_diff_check(
        ("printf '%s\\n' 'git diff --check'",)
    ) is False

    report = daemon._run_validation_commands(
        repo,
        task,
        repo / "validation.log",
        baseline_ref=baseline,
    )

    assert report["passed"] is False
    assert report["reason"] == "candidate_diff_check_failed"
    invariant = report["candidate_diff_check"]
    assert invariant["stage"] == "candidate_invariant"
    assert invariant["returncode"] != 0
    assert invariant["command"].startswith(
        f"git diff --check {baseline}"
    )
    assert "src/alpha.py:1: trailing whitespace" in (
        repo / "validation.log"
    ).read_text(encoding="utf-8")


def test_daemon_python_validation_imports_configured_worktree_packages(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _repo(repo)
    provider_root = repo / "external" / "provider"
    provider_root.mkdir(parents=True)
    (provider_root / "sibling_provider.py").write_text(
        "VALUE = 7\n",
        encoding="utf-8",
    )
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state.json",
        strategy_path=repo / "strategy.json",
        events_path=repo / "events.jsonl",
        repo_root=repo,
        worktree_submodule_paths=("external/provider",),
        worktree_pool_enabled=False,
        validation_cache_dir=repo / "validation-cache",
        merge_queue_dir=repo / "merge-queue",
    )
    task = PortalTask(
        task_id="REF-044",
        title="worktree package validation",
        status="todo",
        completion="manual",
        priority="P1",
        track="validation",
        validation=[
            "python3 -c 'import sibling_provider; "
            "assert sibling_provider.VALUE == 7'"
        ],
    )

    report = daemon._run_validation_commands(
        repo,
        task,
        repo / "validation.log",
    )

    assert report["passed"] is True
    assert report["results"][0]["command"].startswith(
        "export PYTHONPATH=external/provider && python3 "
    )
    assert (
        "added configured worktree package roots to PYTHONPATH"
        in (repo / "validation.log").read_text(encoding="utf-8")
    )


def test_daemon_pythonpath_export_covers_root_chained_validation_commands(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _repo(repo)
    provider_root = repo / "external" / "provider"
    provider_root.mkdir(parents=True)
    (provider_root / "sibling_provider.py").write_text(
        "VALUE = 9\n",
        encoding="utf-8",
    )
    tool_root = repo / "tools"
    tool_root.mkdir()
    (tool_root / "check.py").write_text(
        "import sibling_provider\nassert sibling_provider.VALUE == 9\n",
        encoding="utf-8",
    )
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state.json",
        strategy_path=repo / "strategy.json",
        events_path=repo / "events.jsonl",
        repo_root=repo,
        worktree_submodule_paths=("external/provider",),
        worktree_pool_enabled=False,
        validation_cache_dir=repo / "validation-cache",
        merge_queue_dir=repo / "merge-queue",
    )
    task = PortalTask(
        task_id="REF-044-ROOT-CHAIN",
        title="chained repository-root worktree validation",
        status="todo",
        completion="manual",
        priority="P1",
        track="validation",
        validation=[
            "python3 -c 'import sibling_provider; "
            "assert sibling_provider.VALUE == 9' && python3 tools/check.py"
        ],
    )

    report = daemon._run_validation_commands(
        repo,
        task,
        repo / "validation.log",
    )

    assert report["passed"] is True
    assert report["results"][0]["command"].startswith(
        "export PYTHONPATH=external/provider && python3 "
    )


def test_daemon_python_validation_after_cd_keeps_worktree_packages_importable(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _repo(repo)
    package_root = repo / "package"
    package_root.mkdir()
    provider_root = repo / "external" / "provider"
    provider_root.mkdir(parents=True)
    (provider_root / "sibling_provider.py").write_text(
        "VALUE = 11\n",
        encoding="utf-8",
    )
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state.json",
        strategy_path=repo / "strategy.json",
        events_path=repo / "events.jsonl",
        repo_root=repo,
        worktree_submodule_paths=("external/provider",),
        worktree_pool_enabled=False,
        validation_cache_dir=repo / "validation-cache",
        merge_queue_dir=repo / "merge-queue",
    )
    task = PortalTask(
        task_id="REF-044-CD",
        title="package-local worktree validation",
        status="todo",
        completion="manual",
        priority="P1",
        track="validation",
        validation=[
            "cd package && python3 -c 'import sibling_provider; "
            "assert sibling_provider.VALUE == 11'"
        ],
    )

    report = daemon._run_validation_commands(
        repo,
        task,
        repo / "validation.log",
    )

    assert report["passed"] is True
    assert report["results"][0]["command"].startswith(
        "cd package && export PYTHONPATH=../external/provider && python3 "
    )


def test_daemon_pythonpath_export_covers_chained_validation_commands(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _repo(repo)
    package_root = repo / "package"
    (package_root / "local_package").mkdir(parents=True)
    (package_root / "local_package" / "__init__.py").write_text(
        "VALUE = 13\n",
        encoding="utf-8",
    )
    benchmark_root = package_root / "benchmarks"
    benchmark_root.mkdir()
    (benchmark_root / "run.py").write_text(
        "import local_package\nassert local_package.VALUE == 13\n",
        encoding="utf-8",
    )
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state.json",
        strategy_path=repo / "strategy.json",
        events_path=repo / "events.jsonl",
        repo_root=repo,
        worktree_submodule_paths=("package",),
        worktree_pool_enabled=False,
        validation_cache_dir=repo / "validation-cache",
        merge_queue_dir=repo / "merge-queue",
    )
    task = PortalTask(
        task_id="REF-044-CHAIN",
        title="chained package-local worktree validation",
        status="todo",
        completion="manual",
        priority="P1",
        track="validation",
        validation=[
            "cd package && python3 -c 'import local_package; "
            "assert local_package.VALUE == 13' && python3 benchmarks/run.py"
        ],
    )

    report = daemon._run_validation_commands(
        repo,
        task,
        repo / "validation.log",
    )

    assert report["passed"] is True
    assert report["results"][0]["command"].startswith(
        "cd package && export PYTHONPATH=. && python3 "
    )


def test_daemon_preserves_explicit_validation_pythonpath(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _repo(repo)
    (repo / "external" / "provider").mkdir(parents=True)
    captured: dict[str, object] = {}

    class Scheduler:
        def run(self, commands, **_kwargs):
            captured["commands"] = tuple(commands)
            return {
                "attempted": True,
                "passed": True,
                "returncode": 0,
                "results": [],
            }

    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state.json",
        strategy_path=repo / "strategy.json",
        events_path=repo / "events.jsonl",
        repo_root=repo,
        validation_scheduler=Scheduler(),  # type: ignore[arg-type]
        worktree_submodule_paths=("external/provider",),
    )
    command = "PYTHONPATH=src python3 -m pytest tests/unit -q"
    task = PortalTask(
        task_id="REF-045",
        title="explicit validation path",
        status="todo",
        completion="manual",
        priority="P1",
        track="validation",
        validation=[command],
    )

    daemon._run_validation_commands(repo, task, repo / "validation.log")

    assert captured["commands"] == (command,)


def test_daemon_binds_task_validation_to_proposal_local_impact_graph(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class Scheduler:
        def run_validated(self, proposal_validation, commands, **kwargs):
            captured["proposal_validation"] = proposal_validation
            captured["commands"] = tuple(commands)
            captured.update(kwargs)
            return {
                "attempted": True,
                "passed": True,
                "returncode": 0,
                "results": [],
            }

    daemon = TodoImplementationDaemon(
        todo_path=tmp_path / "todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=tmp_path,
        validation_scheduler=Scheduler(),  # type: ignore[arg-type]
    )
    task = PortalTask(
        task_id="IRF-010",
        title="proposal-local validation",
        status="todo",
        completion="manual",
        priority="P0",
        track="platform",
        validation=["python -m pytest tests/unit/test_identity.py -q"],
    )
    proposal_validation = SimpleNamespace(
        accepted=True,
        findings=(),
        proposal=SimpleNamespace(
            proposal_id="proposal:fixture",
            repository_tree_id="tree:fixture",
            changed_paths=("src/identity.py", "tests/unit/test_identity.py"),
        ),
        policy=SimpleNamespace(policy_id="policy:fixture"),
        receipt=SimpleNamespace(receipt_id="receipt:fixture"),
    )

    report = daemon._run_validation_commands(
        tmp_path,
        task,
        tmp_path / "validation.log",
        proposal_validation=proposal_validation,
    )

    commands = captured["commands"]
    graph = captured["impact_graph"]
    assert captured["require_impact_graph"] is True
    assert captured["require_full_validation"] is True
    assert captured["scope"] == "pre_merge"
    assert len(commands) == 1
    assert commands[0].validation_id.startswith("declared:")
    assert graph.graph_version == "declared-validation-plan-v1"
    assert graph.required_validations(
        graph.affected_paths(
            ("src/identity.py", "tests/unit/test_identity.py")
        )
    )
    assert report["passed"] is True
    assert report["validation_plan_binding"]["graph_id"] == graph.graph_id
