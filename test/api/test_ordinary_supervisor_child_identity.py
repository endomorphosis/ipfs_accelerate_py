"""Ordinary Portal loops retain native child identity and execution policy."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import shlex
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_supervisor as portal,
    supervisor_runtime as runtime,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_loop import SupervisorLoop


@pytest.fixture(autouse=True)
def _benign_launch_environment(monkeypatch):
    # These tests need no owner credential; do not request a real broker grant.
    for name in (
        "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SOCKET",
        "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD",
    ):
        monkeypatch.delenv(name, raising=False)


def _loop(tmp_path: Path, *options: str):
    board = tmp_path / "tasks.todo.md"
    board.write_text("# Native identity controls\n", encoding="utf-8")
    script = tmp_path / "benign_daemon.py"
    script.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    args = portal.parse_args([
        "--todo-path", str(board), "--state-dir", str(tmp_path / "state"),
        "--max-task-attempts", "3", *options,
    ])
    config = replace(
        portal.supervisor_config_from_args(args, repo_root=tmp_path),
        daemon_script_path=script,
        implementation_protected_paths=("src/protected.py",),
    )
    supervisor = portal.PortalImplementationSupervisor(config)
    configured = supervisor.build_supervisor_loop_config()
    spec = SupervisorLoop(configured)._child_spec("native-identity-control")
    assert spec.command == tuple(supervisor._build_daemon_command())
    assert spec.pass_fds == configured.spec.pass_fds
    assert spec.worker_credential_handoff == configured.worker_credential_handoff is False
    assert spec.start_new_session is True
    assert spec.stdin_devnull is True
    return supervisor, configured, spec


@contextmanager
def _native_child(spec):
    child = runtime.launch_supervised_child(spec)
    birth = runtime.read_process_birth(child.pid)
    assert birth is not None and birth.parent_pid == os.getpid()
    try:
        deadline = time.monotonic() + 5
        while runtime.read_process_command_argv(child.pid) != tuple(spec.command):
            assert time.monotonic() < deadline, "native child never reached its configured entry"
            time.sleep(0.01)
        yield child
    finally:
        # Preserve custody independently of deliberately corrupted test markers.
        if runtime.read_process_birth(child.pid) == birth:
            os.kill(child.pid, signal.SIGTERM)
        try:
            os.waitpid(child.pid, 0)
        except ChildProcessError:
            pass


def _identity(supervisor, child, spec):
    return runtime.write_supervised_child_identity(
        supervisor._managed_daemon_identity_path(), pid=child.pid,
        command=spec.command, owner_scope=supervisor._managed_daemon_owner_scope(),
        require_direct_child=True,
    )


def test_constructed_ordinary_loop_keeps_native_identity_and_exact_policy(tmp_path):
    supervisor, configured, spec = _loop(tmp_path)
    assert configured.child_env[runtime.SUPERVISED_CHILD_IDENTITY_PATH_ENV] == str(
        supervisor._managed_daemon_identity_path()
    )
    assert json.loads(configured.child_env[runtime.SUPERVISED_CHILD_OWNER_SCOPE_ENV]) == (
        supervisor._managed_daemon_owner_scope()
    )
    with _native_child(spec) as child:
        assert child.identity_path == supervisor._managed_daemon_identity_path()
        assert child.identity_process_birth == runtime.read_process_birth(child.pid)
        assert child.owned_process_group_id == os.getpgid(child.pid) == child.pid
        assert os.getsid(child.pid) == child.pid
        assert runtime.read_process_command_argv(child.pid) == spec.command
        assert supervisor._managed_daemon_matches_command_line(
            shlex.join(runtime.read_process_command_argv(child.pid))
        )
        adopted = runtime.adopt_or_launch_supervised_child(
            spec, launch_lock_path=configured.spec.resolve(configured.spec.supervisor_lock_path),
        )
        assert adopted is not None
        assert adopted.pid == child.pid
        assert adopted.identity_record_id == child.identity_record_id
        assert adopted.identity_process_birth == child.identity_process_birth


@pytest.mark.parametrize("override", [
    ("--max-task-attempts", "0"),
    ("--validation-max-workers", "1"),
    ("--implementation-protected-path", "src/unprotected.py"),
    ("--state-prefix", "foreign-prefix"),
])
def test_native_adoption_rejects_appended_execution_policy_override(tmp_path, override):
    supervisor, _configured, expected = _loop(tmp_path)
    foreign = replace(expected, command=(*expected.command, *override))
    with _native_child(foreign) as child:
        assert not supervisor._managed_daemon_matches_command_line(
            shlex.join(runtime.read_process_command_argv(child.pid))
        )
        before = {path: path.read_bytes() for path in (
            supervisor._managed_daemon_pid_path(),
            supervisor._managed_daemon_identity_path(),
        ) if path.exists()}
        with pytest.raises(RuntimeError, match="ownership identity mismatch"):
            runtime.adopt_supervised_child(expected)
        assert runtime.read_process_birth(child.pid) is not None
        assert {path: path.read_bytes() for path in before} == before


def test_native_adoption_rejects_a_benign_bootstrap_that_only_contains_expected_argv(tmp_path):
    _supervisor, _configured, expected = _loop(tmp_path)
    foreign = replace(expected, command=(
        expected.command[0], "-c", "import time; time.sleep(60)",
        *expected.command[1:],
    ))
    with _native_child(foreign) as child:
        with pytest.raises(RuntimeError, match="ownership identity mismatch"):
            runtime.adopt_supervised_child(expected)
        assert runtime.read_process_birth(child.pid) is not None


@pytest.mark.parametrize("marker", ["pid", "identity"])
def test_native_adoption_recovers_only_exact_live_missing_marker(tmp_path, marker):
    supervisor, _configured, spec = _loop(tmp_path)
    with _native_child(spec) as child:
        path = (supervisor._managed_daemon_pid_path() if marker == "pid"
                else supervisor._managed_daemon_identity_path())
        path.unlink(missing_ok=True)
        adopted = runtime.adopt_supervised_child(spec)
        assert adopted is not None and adopted.pid == child.pid
        assert adopted.identity_path == supervisor._managed_daemon_identity_path()
        identity = runtime.load_supervised_child_identity(adopted.identity_path)
        assert identity is not None
        assert identity.process_birth == runtime.read_process_birth(child.pid)
        assert identity.command == spec.command
        assert dict(identity.owner_scope) == supervisor._managed_daemon_owner_scope()
        assert supervisor._managed_daemon_pid_path().read_text().strip() == str(child.pid)


@pytest.mark.parametrize("change", ["birth", "boot", "scope", "command", "malformed"])
def test_native_adoption_rejects_changed_identity_without_touching_child(tmp_path, change):
    supervisor, _configured, spec = _loop(tmp_path)
    with _native_child(spec) as child:
        identity = _identity(supervisor, child, spec)
        if change == "birth":
            identity = replace(identity, process_birth=replace(
                identity.process_birth,
                start_time_ticks=identity.process_birth.start_time_ticks + 1,
            ), record_id="")
        elif change == "boot":
            identity = replace(identity, process_birth=replace(
                identity.process_birth, boot_id="a-different-boot",
            ), record_id="")
        elif change == "scope":
            identity = replace(identity, owner_scope={"repo_root": str(tmp_path / "foreign")},
                               record_id="")
        elif change == "command":
            identity = replace(identity, command=(*identity.command, "--foreign"), record_id="")
        path = supervisor._managed_daemon_identity_path()
        path.write_text("unavailable\n" if change == "malformed" else json.dumps(identity.to_dict()))
        retained = path.read_bytes()
        with pytest.raises(RuntimeError, match="ownership identity mismatch"):
            runtime.adopt_supervised_child(spec)
        assert path.read_bytes() == retained
        assert runtime.read_process_command_argv(child.pid) == spec.command


def test_native_adoption_refuses_session_relaxation(tmp_path):
    _supervisor, _configured, spec = _loop(tmp_path)
    with _native_child(spec):
        with pytest.raises(RuntimeError, match="dedicated process session"):
            runtime.adopt_supervised_child(replace(spec, start_new_session=False))


def test_native_adoption_does_not_mint_identity_for_an_inherited_session(tmp_path):
    supervisor, _configured, expected = _loop(tmp_path)
    unprotected = replace(expected, start_new_session=False, env={
        name: value for name, value in expected.env.items()
        if name not in (runtime.SUPERVISED_CHILD_IDENTITY_PATH_ENV,
                        runtime.SUPERVISED_CHILD_OWNER_SCOPE_ENV)
    })
    with _native_child(unprotected) as child:
        assert os.getsid(child.pid) != child.pid
        assert os.getpgid(child.pid) != child.pid
        before_pid = supervisor._managed_daemon_pid_path().read_bytes()
        with pytest.raises(RuntimeError, match="dedicated process session"):
            runtime.adopt_supervised_child(expected)
        assert not supervisor._managed_daemon_identity_path().exists()
        assert supervisor._managed_daemon_pid_path().read_bytes() == before_pid


@pytest.mark.parametrize("workers", ["1", "4", "256"])
def test_validation_worker_cli_factory_and_child_argv_are_identical(tmp_path, workers):
    supervisor, _configured, spec = _loop(tmp_path, "--validation-max-workers", workers)
    assert supervisor.config.validation_max_workers == int(workers)
    index = spec.command.index("--validation-max-workers")
    assert spec.command[index + 1] == workers
    assert supervisor._managed_daemon_matches_command_line(shlex.join(spec.command))


@pytest.mark.parametrize("workers", ["0", "-1", "257", "1.5", "true"])
def test_invalid_validation_worker_cli_is_rejected(workers):
    with pytest.raises(SystemExit) as stopped:
        portal.parse_args(["--validation-max-workers", workers])
    assert stopped.value.code == 2


@pytest.mark.parametrize("workers", [0, -1, 257, True, 1.5])
def test_invalid_validation_worker_config_is_not_clamped(tmp_path, workers):
    supervisor, _configured, _spec = _loop(tmp_path)
    with pytest.raises(ValueError, match="validation.*workers"):
        replace(supervisor.config, validation_max_workers=workers)
