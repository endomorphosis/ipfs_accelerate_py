from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sys
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon,
    supervisor,
    supervisor_runtime,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.core import ManagedDaemonSpec
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTaskState,
    state_file_repair_reason,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor,
    PortalSupervisorConfig,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_loop import (
    SupervisorLoop,
    SupervisorLoopConfig,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_runtime import (
    run_process_group_stream,
)
from test.api.test_llm_router_agent_implementation_route import _signed_high_plan


@pytest.fixture
def sealed_control_plane_fd() -> Iterator[tuple[int, str]]:
    if not hasattr(os, "memfd_create") or not hasattr(fcntl, "F_ADD_SEALS"):
        pytest.skip("Linux sealed memfds are unavailable")
    descriptor = os.memfd_create(
        "ipfs-accelerate-accepted-control-plane",
        os.MFD_ALLOW_SEALING,
    )
    archive = (
        b"import os,sys,time\n"
        b"marker=os.environ.get('WATCHDOG_TEST_MARKER','')\n"
        b"if marker:\n"
        b" open(marker,'w',encoding='utf-8').write(sys.stdin.read())\n"
        b"else:\n"
        b" time.sleep(30)\n"
    )
    os.write(descriptor, archive)
    fcntl.fcntl(
        descriptor,
        fcntl.F_ADD_SEALS,
        fcntl.F_SEAL_WRITE
        | fcntl.F_SEAL_SHRINK
        | fcntl.F_SEAL_GROW
        | fcntl.F_SEAL_SEAL,
    )
    try:
        yield descriptor, "sha256:" + hashlib.sha256(archive).hexdigest()
    finally:
        os.close(descriptor)


def _sealed_receipt_case(
    tmp_path,
    *,
    descriptor: int,
    archive_sha256: str,
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    workspace = tmp_path / "task-worktree"
    workspace.mkdir(exist_ok=True)
    task_id = "VGO-009"
    attempt = 1
    revision = "baguqeera" + "a" * 48
    latch_body = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "protected-implementation-attempt-latch@1"
        ),
        "task_id": task_id,
        "attempt": attempt,
        "task_revision_cid": revision,
        "board_namespace": "verified-gui-optimizer-v1",
        "route_id": "reviewed-grok-terra-route-v1",
        "invocation_id": "baguqeera" + "b" * 48,
        "logical_attempt_id": "baguqeera" + "c" * 48,
        "worktree_id": "baguqeera" + "d" * 48,
        "provider_attempt_store": str(tmp_path / "provider-attempts"),
        "provider_attempt_store_identity": "sha256:" + "1" * 64,
    }
    latch = {**latch_body, "latch_id": content_identity(latch_body)}
    latch_key = content_identity(
        {
            "task_id": task_id,
            "attempt": attempt,
            "task_revision_cid": revision,
        }
    )
    status = {
        "heartbeat_at": datetime.now(UTC).isoformat(),
        "active_phase": "implementing",
        "active_phase_started_at": (
            datetime.now(UTC) - timedelta(minutes=10)
        ).isoformat(),
        "active_task_id": task_id,
        "active_attempt": attempt,
        "active_task_cid": revision,
        "active_worktree_path": str(workspace),
        "implementation_in_progress": True,
        "worktree_no_child_stall_seconds": 60,
        "protected_implementation_attempts": {latch_key: latch},
    }
    control_plane = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "accepted-control-plane-pin@2"
        ),
        "runner_path": str(tmp_path / "grok_cli_runner.py"),
        "runner_sha256": "sha256:" + "2" * 64,
        "capsule_root": str(tmp_path / "capsule"),
        "capsule_id": "accepted-test-capsule",
        "source_head": "3" * 40,
        "source_tree": "4" * 40,
        "archive_sha256": archive_sha256,
    }
    invocation = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "provider-fallback-invocation@2"
        ),
        "task_id": task_id,
        "attempt": attempt,
        "task_revision_cid": revision,
        "workspace_path": str(workspace),
        "route_id": latch["route_id"],
        "invocation_id": latch["invocation_id"],
        "logical_attempt_id": latch["logical_attempt_id"],
        "worktree_id": latch["worktree_id"],
        "provider_attempt_store": latch["provider_attempt_store"],
        "provider_attempt_store_identity": latch[
            "provider_attempt_store_identity"
        ],
        "control_plane": control_plane,
    }
    binding = {
        "authorization": {"board_namespace": latch["board_namespace"]},
        "invocation_binding": invocation,
        "route_id": latch["route_id"],
    }
    argv = [
        sys.executable,
        "-I",
        f"/proc/self/fd/{descriptor}",
        "--workspace",
        str(workspace),
        "--model",
        "grok-4.5",
        "--agent-implementation-route-json",
        json.dumps(
            binding,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ),
    ]
    pid = os.getpid()
    owner_pid = os.getppid()
    fd_stat = os.fstat(descriptor)
    exe_stat = os.stat("/proc/self/exe")
    seals = int(fcntl.fcntl(descriptor, fcntl.F_GET_SEALS))
    receipt_body = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "provider-runner-birth@1"
        ),
        "task_id": task_id,
        "attempt": attempt,
        "task_revision_cid": revision,
        "workspace_path": str(workspace),
        "latch_id": latch["latch_id"],
        "route_id": latch["route_id"],
        "invocation_id": latch["invocation_id"],
        "logical_attempt_id": latch["logical_attempt_id"],
        "worktree_id": latch["worktree_id"],
        "owner_pid": owner_pid,
        "owner_start_ticks": supervisor._process_start_ticks(owner_pid),
        "pid": pid,
        "start_ticks": supervisor._process_start_ticks(pid),
        "argv_sha256": supervisor._argv_sha256(argv),
        "executable_device": exe_stat.st_dev,
        "executable_inode": exe_stat.st_ino,
        "descriptor_number": descriptor,
        "descriptor_device": fd_stat.st_dev,
        "descriptor_inode": fd_stat.st_ino,
        "descriptor_size": fd_stat.st_size,
        "descriptor_seals": seals,
        "archive_sha256": archive_sha256,
    }
    status["active_provider_runner"] = {
        **receipt_body,
        "receipt_id": content_identity(receipt_body),
    }
    item = {
        "pid": pid,
        "cmdline": " ".join(argv),
        "argv": tuple(argv),
        "start_ticks": receipt_body["start_ticks"],
    }
    return status, argv, item


def _readdress_receipt(status: dict[str, Any]) -> None:
    receipt = status["active_provider_runner"]
    body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    receipt["receipt_id"] = content_identity(body)


def _fake_control_plane(command: list[str]) -> SimpleNamespace:
    route_index = command.index("--agent-implementation-route-json")
    binding = json.loads(command[route_index + 1])
    values = dict(binding["invocation_binding"]["control_plane"])
    return SimpleNamespace(**values, as_dict=lambda: dict(values))


def _birth_callback_case(
    tmp_path,
    *,
    descriptor: int,
    archive_sha256: str,
) -> tuple[
    PortalImplementationDaemon,
    PortalTaskState,
    list[str],
    Path,
    Any,
]:
    status, command, _ = _sealed_receipt_case(
        tmp_path,
        descriptor=descriptor,
        archive_sha256=archive_sha256,
    )
    workspace = tmp_path / "task-worktree"
    state = PortalTaskState(
        **{
            key: value
            for key, value in status.items()
            if key in PortalTaskState.__dataclass_fields__
        }
    )
    state.active_provider_runner = {}
    daemon = object.__new__(PortalImplementationDaemon)
    daemon.state_path = tmp_path / "state.json"
    daemon._scoped_control_plane_launch = SimpleNamespace(
        descriptor=descriptor,
        executable_path=f"/proc/self/fd/{descriptor}",
        archive_sha256=archive_sha256,
        seals=int(fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)),
        capsule_id="accepted-test-capsule",
    )
    daemon._scoped_control_plane = _fake_control_plane(command)
    daemon._canonical_ref = lambda _task: status["active_task_cid"]
    callback = daemon._provider_runner_started_callback(
        state,
        task=SimpleNamespace(task_id=status["active_task_id"]),
        attempt=status["active_attempt"],
        command=command,
        workspace_path=workspace,
    )
    assert callback is not None
    return daemon, state, command, workspace, callback


def _preserve_process_identity(
    monkeypatch: pytest.MonkeyPatch,
    *,
    argv: list[str],
) -> None:
    real_argv = supervisor._process_command_argv
    real_parent = supervisor._process_parent_pid
    monkeypatch.setattr(
        supervisor,
        "_process_command_argv",
        lambda pid: tuple(argv) if pid == os.getpid() else real_argv(pid),
    )
    monkeypatch.setattr(
        supervisor,
        "_process_parent_pid",
        lambda pid: os.getppid() if pid == os.getpid() else real_parent(pid),
    )
    real_stat = supervisor.os.stat
    real_readlink = supervisor.os.readlink
    descriptor = int(argv[2].rsplit("/", 1)[1])

    def receipt_stat(path, *args, **kwargs):
        path_text = os.fspath(path)
        if path_text == f"/proc/{os.getppid()}/fd/{descriptor}":
            return real_stat(f"/proc/{os.getpid()}/fd/{descriptor}")
        return real_stat(path, *args, **kwargs)

    def receipt_readlink(path, *args, **kwargs):
        path_text = os.fspath(path)
        if path_text == f"/proc/{os.getppid()}/fd/{descriptor}":
            return real_readlink(f"/proc/{os.getpid()}/fd/{descriptor}")
        return real_readlink(path, *args, **kwargs)

    monkeypatch.setattr(supervisor.os, "stat", receipt_stat)
    monkeypatch.setattr(supervisor.os, "readlink", receipt_readlink)


def test_watchdog_recognizes_exact_daemon_owned_sealed_runner(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    sealed_control_plane_fd: tuple[int, str],
) -> None:
    descriptor, digest = sealed_control_plane_fd
    status, argv, item = _sealed_receipt_case(
        tmp_path,
        descriptor=descriptor,
        archive_sha256=digest,
    )
    _preserve_process_identity(monkeypatch, argv=argv)
    monkeypatch.setattr(supervisor, "descendant_processes", lambda _pid: [item])

    worker = supervisor.worktree_phase_worker_status(
        status,
        daemon_pid=os.getppid(),
        threshold_seconds=60,
    )

    assert worker["active_worker_count"] == 1
    assert worker["stalled_without_active_worker"] is False
    assert supervisor.active_codex_exec_workers(os.getppid(), status) == [item]


def test_outer_supervisor_keeps_exact_sealed_runner(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    sealed_control_plane_fd: tuple[int, str],
) -> None:
    descriptor, digest = sealed_control_plane_fd
    status, argv, item = _sealed_receipt_case(
        tmp_path,
        descriptor=descriptor,
        archive_sha256=digest,
    )
    now = datetime.now(UTC)
    status.update(
        {
            "heartbeat_at": now.isoformat(),
            "last_progress_at": now.isoformat(),
            "last_implementation_task_id": status["active_task_id"],
            "last_implementation_started_at": now.isoformat(),
        }
    )
    state = PortalTaskState(
        **{
            key: value
            for key, value in status.items()
            if key in PortalTaskState.__dataclass_fields__
        }
    )
    config = PortalSupervisorConfig(
        todo_path=tmp_path / "todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        state_dir=tmp_path / "state",
        stale_seconds=3600,
        implementation_timeout=3600,
        implementation_log_stall_seconds=60,
    )
    outer = PortalImplementationSupervisor(config)
    state.save(config.state_path)
    _preserve_process_identity(monkeypatch, argv=argv)
    monkeypatch.setattr(supervisor, "descendant_processes", lambda _pid: [item])
    monkeypatch.setattr(outer, "_read_managed_daemon_pid", lambda: os.getppid())

    assert outer._active_agent_worker_processes() == [item]
    stuck, reason = outer.is_stuck(state, now_ts=now.timestamp())

    assert stuck is False
    assert reason == ""


def test_supervisor_loop_graces_exact_sealed_runner_disappearance(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    sealed_control_plane_fd: tuple[int, str],
) -> None:
    descriptor, digest = sealed_control_plane_fd
    status, argv, item = _sealed_receipt_case(
        tmp_path,
        descriptor=descriptor,
        archive_sha256=digest,
    )
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    spec = ManagedDaemonSpec(
        name="sealed-daemon",
        schema="test.daemon",
        repo_root=tmp_path,
        daemon_dir=state_dir,
        runner=(sys.executable, "-c", "pass"),
        status_path=state_dir / "daemon.json",
        supervisor_status_path=state_dir / "supervisor.json",
        supervisor_pid_path=state_dir / "supervisor.pid",
        child_pid_path=state_dir / "child.pid",
        supervisor_out_path=state_dir / "supervisor.out",
        ensure_status_path=state_dir / "ensure.json",
        ensure_check_path=state_dir / "check.json",
    )
    clock = [100.0]
    loop = SupervisorLoop(
        SupervisorLoopConfig(
            spec=spec,
            command=(sys.executable, "-c", "pass"),
            log_prefix="child",
            watchdog_stale_after_seconds=60,
        ),
        monotonic=lambda: clock[0],
    )
    live = [True]
    _preserve_process_identity(monkeypatch, argv=argv)
    monkeypatch.setattr(
        supervisor,
        "descendant_processes",
        lambda _pid: [item] if live[0] else [],
    )
    child = SimpleNamespace(pid=os.getppid())

    assert loop.default_watchdog(child, status).action == "continue"
    live[0] = False
    clock[0] = 110.0
    assert loop.default_watchdog(child, status).action == "continue"
    clock[0] = 161.0
    expired = loop.default_watchdog(child, status)

    assert expired.action == "recycle"
    assert expired.reason == "worktree_phase_without_active_child"
    assert expired.detail["worker_absence_age_seconds"] == 61.0


@pytest.mark.parametrize(
    "mutation",
    [
        "no_receipt",
        "receipt_pid",
        "receipt_cid",
        "receipt_attempt_bool",
        "latch_attempt_bool",
        "latch_extra_field",
        "relative_workspace",
        "duplicate_workspace",
        "missing_isolation",
        "low_descriptor",
        "recovery",
        "route_and_recovery",
        "owner_birth",
        "fd_inode",
    ],
)
def test_watchdog_rejects_sealed_runner_lookalikes(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    sealed_control_plane_fd: tuple[int, str],
    mutation: str,
) -> None:
    descriptor, digest = sealed_control_plane_fd
    status, argv, item = _sealed_receipt_case(
        tmp_path,
        descriptor=descriptor,
        archive_sha256=digest,
    )
    if mutation == "no_receipt":
        status["active_provider_runner"] = {}
    elif mutation == "receipt_pid":
        status["active_provider_runner"]["pid"] += 1
        _readdress_receipt(status)
    elif mutation == "receipt_cid":
        status["active_provider_runner"]["route_id"] = "tampered"
    elif mutation == "receipt_attempt_bool":
        status["active_provider_runner"]["attempt"] = True
        _readdress_receipt(status)
    elif mutation.startswith("latch_"):
        latch = next(iter(status["protected_implementation_attempts"].values()))
        if mutation == "latch_attempt_bool":
            latch["attempt"] = True
        else:
            latch["extra"] = "forged"
        latch_body = {
            key: value for key, value in latch.items() if key != "latch_id"
        }
        latch["latch_id"] = content_identity(latch_body)
        status["active_provider_runner"]["latch_id"] = latch["latch_id"]
        _readdress_receipt(status)
    elif mutation == "relative_workspace":
        argv[4] = "relative/worktree"
        item["argv"] = tuple(argv)
    elif mutation == "duplicate_workspace":
        argv.extend(["--workspace", status["active_worktree_path"]])
        item["argv"] = tuple(argv)
    elif mutation == "missing_isolation":
        argv[1] = "-P"
        item["argv"] = tuple(argv)
    elif mutation == "low_descriptor":
        argv[2] = "/proc/self/fd/2"
        item["argv"] = tuple(argv)
    elif mutation == "recovery":
        argv = argv[:5] + ["--agent-implementation-recovery-json", "{}"]
        item["argv"] = tuple(argv)
    elif mutation == "route_and_recovery":
        argv.extend(["--agent-implementation-recovery-json", "{}"])
        item["argv"] = tuple(argv)
    elif mutation == "owner_birth":
        status["active_provider_runner"]["owner_start_ticks"] += 1
        _readdress_receipt(status)
    elif mutation == "fd_inode":
        status["active_provider_runner"]["descriptor_inode"] += 1
        _readdress_receipt(status)
    _preserve_process_identity(monkeypatch, argv=argv)
    monkeypatch.setattr(supervisor, "descendant_processes", lambda _pid: [item])

    worker = supervisor.worktree_phase_worker_status(
        status,
        daemon_pid=os.getppid(),
        threshold_seconds=60,
    )

    assert worker["active_worker_count"] == 0
    assert worker["stalled_without_active_worker"] is True


def test_watchdog_proc_disappearance_race_fails_closed(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    sealed_control_plane_fd: tuple[int, str],
) -> None:
    descriptor, digest = sealed_control_plane_fd
    status, argv, item = _sealed_receipt_case(
        tmp_path,
        descriptor=descriptor,
        archive_sha256=digest,
    )
    _preserve_process_identity(monkeypatch, argv=argv)
    real_readlink = supervisor.os.readlink
    calls = [0]

    def disappearing(path):
        if str(path).endswith(f"/fd/{descriptor}"):
            calls[0] += 1
            if calls[0] > 2:
                raise FileNotFoundError(path)
        return real_readlink(path)

    monkeypatch.setattr(supervisor.os, "readlink", disappearing)

    assert not supervisor._sealed_agent_worker_process(
        item,
        status,
        daemon_pid=os.getppid(),
    )


def test_birth_callback_persists_before_prompt_and_round_trips(
    tmp_path,
    sealed_control_plane_fd: tuple[int, str],
) -> None:
    descriptor, digest = sealed_control_plane_fd
    daemon, state, command, workspace, callback = _birth_callback_case(
        tmp_path,
        descriptor=descriptor,
        archive_sha256=digest,
    )
    marker = tmp_path / "prompt-delivered"
    observed: dict[str, Any] = {}

    def assert_birth_before_prompt(process) -> None:
        callback(process)
        assert not marker.exists()
        durable = PortalTaskState.load(daemon.state_path)
        item = {
            "pid": process.pid,
            "cmdline": " ".join(command),
            "argv": tuple(command),
            "start_ticks": supervisor._process_start_ticks(process.pid),
        }
        assert supervisor._sealed_agent_worker_process(
            item,
            vars(durable),
            daemon_pid=os.getpid(),
        )
        observed["pid"] = process.pid
        observed["receipt"] = dict(durable.active_provider_runner)

    completed = run_process_group_stream(
        command,
        cwd=workspace,
        stdout=None,
        env={"WATCHDOG_TEST_MARKER": str(marker)},
        pass_fds=(descriptor,),
        input_text="signed implementation prompt",
        timeout_seconds=10,
        on_started=assert_birth_before_prompt,
    )

    assert completed.returncode == 0
    assert marker.read_text(encoding="utf-8") == "signed implementation prompt"
    assert observed["pid"] > 1
    assert observed["receipt"]["owner_pid"] == os.getpid()
    assert PortalTaskState.load(daemon.state_path).active_provider_runner == observed[
        "receipt"
    ]
    daemon._mark_implementation_finished(state, finished_at=datetime.now(UTC).isoformat())
    state.save(daemon.state_path)
    assert PortalTaskState.load(daemon.state_path).active_provider_runner == {}


def test_birth_callback_waits_for_transient_empty_procfs_cmdline(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    sealed_control_plane_fd: tuple[int, str],
) -> None:
    descriptor, digest = sealed_control_plane_fd
    daemon, _state, command, workspace, callback = _birth_callback_case(
        tmp_path,
        descriptor=descriptor,
        archive_sha256=digest,
    )
    marker = tmp_path / "prompt-delivered-after-handshake"
    real_read_bytes = Path.read_bytes
    cmdline_reads = 0

    def transient_empty_cmdline(path: Path) -> bytes:
        nonlocal cmdline_reads
        if path.name == "cmdline" and path.parent.parent == Path("/proc"):
            cmdline_reads += 1
            if cmdline_reads <= 3:
                return b""
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", transient_empty_cmdline)

    completed = run_process_group_stream(
        command,
        cwd=workspace,
        stdout=None,
        env={"WATCHDOG_TEST_MARKER": str(marker)},
        pass_fds=(descriptor,),
        input_text="prompt released after birth handshake",
        timeout_seconds=10,
        on_started=callback,
    )

    assert completed.returncode == 0
    assert cmdline_reads >= 5  # handshake retry plus post-save recheck
    assert marker.read_text(encoding="utf-8") == (
        "prompt released after birth handshake"
    )
    assert PortalTaskState.load(daemon.state_path).active_provider_runner


def test_birth_callback_empty_procfs_cmdline_deadline_fails_closed(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    sealed_control_plane_fd: tuple[int, str],
) -> None:
    descriptor, digest = sealed_control_plane_fd
    daemon, state, command, workspace, callback = _birth_callback_case(
        tmp_path,
        descriptor=descriptor,
        archive_sha256=digest,
    )
    marker = tmp_path / "prompt-must-remain-withheld"
    real_read_bytes = Path.read_bytes
    observed_pid = 0

    def persistently_empty_cmdline(path: Path) -> bytes:
        if path.name == "cmdline" and path.parent.parent == Path("/proc"):
            return b""
        return real_read_bytes(path)

    def capture_then_persist(process) -> None:
        nonlocal observed_pid
        observed_pid = process.pid
        callback(process)

    monkeypatch.setattr(Path, "read_bytes", persistently_empty_cmdline)
    monkeypatch.setattr(
        implementation_daemon,
        "PROVIDER_RUNNER_BIRTH_TIMEOUT_SECONDS",
        0.02,
    )
    monkeypatch.setattr(
        implementation_daemon,
        "PROVIDER_RUNNER_BIRTH_POLL_SECONDS",
        0.001,
    )

    with pytest.raises(
        RuntimeError,
        match="birth identity was not published",
    ):
        run_process_group_stream(
            command,
            cwd=workspace,
            stdout=None,
            env={"WATCHDOG_TEST_MARKER": str(marker)},
            pass_fds=(descriptor,),
            input_text="prompt must never be released",
            timeout_seconds=10,
            on_started=capture_then_persist,
        )

    assert observed_pid > 1
    assert not os.path.exists(f"/proc/{observed_pid}")
    # The fixture opens the marker before blocking in sys.stdin.read(); an
    # empty file proves no prompt bytes crossed the callback effect barrier.
    if marker.exists():
        assert marker.read_bytes() == b""
    assert state.active_provider_runner == {}
    assert not daemon.state_path.exists()


def test_birth_callback_rejects_latch_mutation_during_handshake(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    sealed_control_plane_fd: tuple[int, str],
) -> None:
    descriptor, digest = sealed_control_plane_fd
    daemon, state, command, workspace, callback = _birth_callback_case(
        tmp_path,
        descriptor=descriptor,
        archive_sha256=digest,
    )
    marker = tmp_path / "prompt-must-remain-withheld-after-latch-drift"
    real_read_bytes = Path.read_bytes
    mutated = False
    observed_pid = 0

    def mutate_latch_during_empty_cmdline(path: Path) -> bytes:
        nonlocal mutated
        if (
            not mutated
            and path.name == "cmdline"
            and path.parent.parent == Path("/proc")
        ):
            mutated = True
            latch = next(
                iter(state.protected_implementation_attempts.values())
            )
            latch["route_id"] = "drifted-during-birth-handshake"
            return b""
        return real_read_bytes(path)

    def capture_then_persist(process) -> None:
        nonlocal observed_pid
        observed_pid = process.pid
        callback(process)

    monkeypatch.setattr(
        Path,
        "read_bytes",
        mutate_latch_during_empty_cmdline,
    )

    with pytest.raises(RuntimeError, match="birth identity drifted"):
        run_process_group_stream(
            command,
            cwd=workspace,
            stdout=None,
            env={"WATCHDOG_TEST_MARKER": str(marker)},
            pass_fds=(descriptor,),
            input_text="prompt must never cross a drifted latch",
            timeout_seconds=10,
            on_started=capture_then_persist,
        )

    assert mutated
    assert observed_pid > 1
    assert not os.path.exists(f"/proc/{observed_pid}")
    if marker.exists():
        assert marker.read_bytes() == b""
    assert state.active_provider_runner == {}
    assert not daemon.state_path.exists()


def test_provider_launch_boundary_refreshes_workerless_grace(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    sealed_control_plane_fd: tuple[int, str],
) -> None:
    descriptor, digest = sealed_control_plane_fd
    daemon, state, _command, workspace, _callback = _birth_callback_case(
        tmp_path,
        descriptor=descriptor,
        archive_sha256=digest,
    )
    fixed_now = datetime.now(UTC)
    state.active_phase_started_at = (
        fixed_now - timedelta(minutes=10)
    ).isoformat()
    monkeypatch.setattr(implementation_daemon, "utc_now", fixed_now.isoformat)
    monkeypatch.setattr(supervisor, "descendant_processes", lambda _pid: [])

    before = supervisor.worktree_phase_worker_status(
        vars(state),
        daemon_pid=os.getpid(),
        threshold_seconds=60,
        now=fixed_now,
    )
    assert before["stalled_without_active_worker"] is True

    task = SimpleNamespace(task_id=state.active_task_id)
    daemon._mark_provider_launch_boundary(
        state,
        task=task,
        attempt=state.active_attempt,
        workspace_path=workspace,
    )
    durable = PortalTaskState.load(daemon.state_path)

    assert durable.active_phase == "implementing"
    assert durable.active_phase_detail == "provider_launch_birth"
    assert durable.active_phase_started_at == fixed_now.isoformat()
    assert durable.heartbeat_at == fixed_now.isoformat()
    assert durable.last_progress_at == fixed_now.isoformat()
    assert durable.active_provider_runner == {}
    old_log = tmp_path / "old-provider.log"
    old_log.write_text("provider launch pending\n", encoding="utf-8")
    old_mtime = fixed_now.timestamp() - 10 * 60
    os.utime(old_log, (old_mtime, old_mtime))
    durable.active_log_path = str(old_log)
    durable.last_implementation_log_path = str(old_log)
    durable.last_implementation_task_id = durable.active_task_id
    durable.last_implementation_started_at = (
        fixed_now - timedelta(minutes=10)
    ).isoformat()
    durable.save(daemon.state_path)
    after = supervisor.worktree_phase_worker_status(
        vars(durable),
        daemon_pid=os.getpid(),
        threshold_seconds=60,
        now=fixed_now + timedelta(seconds=1),
    )
    assert after["stalled_without_active_worker"] is False

    state_dir = tmp_path / "supervisor-state"
    state_dir.mkdir()
    spec = ManagedDaemonSpec(
        name="provider-launch-daemon",
        schema="test.daemon",
        repo_root=tmp_path,
        daemon_dir=state_dir,
        runner=(sys.executable, "-c", "pass"),
        status_path=state_dir / "daemon.json",
        supervisor_status_path=state_dir / "supervisor.json",
        supervisor_pid_path=state_dir / "supervisor.pid",
        child_pid_path=state_dir / "child.pid",
        supervisor_out_path=state_dir / "supervisor.out",
        ensure_status_path=state_dir / "ensure.json",
        ensure_check_path=state_dir / "check.json",
    )
    loop = SupervisorLoop(
        SupervisorLoopConfig(
            spec=spec,
            command=(sys.executable, "-c", "pass"),
            log_prefix="child",
            watchdog_stale_after_seconds=60,
            status_static_fields={"worktree_no_child_stall_seconds": 60},
        ),
        monotonic=lambda: 161.0,
    )
    loop._worker_tracking_generation = before["tracking_generation"]
    loop._last_worker_seen_monotonic = 100.0
    decision = loop.default_watchdog(
        SimpleNamespace(pid=os.getpid(), log_path=None),
        vars(durable),
    )
    assert decision.action == "continue"
    assert loop._worker_tracking_generation == after["tracking_generation"]
    assert loop._last_worker_seen_monotonic is None

    config = PortalSupervisorConfig(
        todo_path=tmp_path / "todo.md",
        state_path=daemon.state_path,
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        state_dir=state_dir,
        stale_seconds=3600,
        implementation_timeout=3600,
        implementation_log_stall_seconds=60,
    )
    outer = PortalImplementationSupervisor(config)
    monkeypatch.setattr(outer, "_read_managed_daemon_pid", os.getpid)
    outer._worktree_worker_generation = before["tracking_generation"]
    outer._last_worktree_worker_seen_monotonic = time.monotonic() - 61
    assert (
        outer._worktree_phase_without_worker_reason(
            durable,
            now_ts=fixed_now.timestamp() + 1,
        )
        == ""
    )
    assert outer._worktree_worker_generation == after["tracking_generation"]
    assert outer._last_worktree_worker_seen_monotonic is None
    monkeypatch.setattr(outer, "_active_agent_worker_processes", list)
    monkeypatch.setattr(outer, "_active_validation_subprocess_exists", bool)
    assert (
        outer._implementation_log_stall_reason(
            durable,
            now_ts=fixed_now.timestamp() + 1,
        )
        == ""
    )
    stuck, reason = outer.is_stuck(
        durable,
        now_ts=fixed_now.timestamp() + 1,
    )
    assert stuck is False
    assert reason == ""

    expired_at = fixed_now.timestamp() + 61
    assert "implementation log stalled" in (
        outer._implementation_log_stall_reason(
            durable,
            now_ts=expired_at,
        )
    )
    stuck, reason = outer.is_stuck(durable, now_ts=expired_at)
    assert stuck is True
    assert (
        "no active worker" in reason
        or "implementation log stalled" in reason
    )


def test_birth_callback_accepts_canonical_signed_route_fixture(tmp_path) -> None:
    workspace, route, invocation = _signed_high_plan(tmp_path)
    authorization = route.authorization
    assert authorization is not None
    launch = llm_router.seal_agent_implementation_control_plane_capsule(
        invocation.control_plane
    )
    try:
        latch_body = {
            "schema": (
                "ipfs_accelerate_py.agent_supervisor."
                "protected-implementation-attempt-latch@1"
            ),
            "task_id": invocation.task_id,
            "attempt": invocation.attempt,
            "task_revision_cid": invocation.task_revision_cid,
            "board_namespace": authorization.board_namespace,
            "route_id": invocation.route_id,
            "invocation_id": invocation.invocation_id,
            "logical_attempt_id": invocation.logical_attempt_id,
            "worktree_id": invocation.worktree_id,
            "provider_attempt_store": invocation.provider_attempt_store,
            "provider_attempt_store_identity": (
                invocation.provider_attempt_store_identity
            ),
        }
        latch = {**latch_body, "latch_id": content_identity(latch_body)}
        latch_key = content_identity(
            {
                "task_id": invocation.task_id,
                "attempt": invocation.attempt,
                "task_revision_cid": invocation.task_revision_cid,
            }
        )
        state = PortalTaskState(
            active_task_id=invocation.task_id,
            active_attempt=invocation.attempt,
            active_task_cid=invocation.task_revision_cid,
            active_worktree_path=str(workspace.resolve()),
            implementation_in_progress=True,
            protected_implementation_attempts={latch_key: latch},
        )
        command = [
            sys.executable,
            "-I",
            launch.executable_path,
            "--workspace",
            str(workspace.resolve()),
            "--model",
            "grok-4.5",
            "--agent-implementation-route-json",
            json.dumps(
                route.as_binding_dict(),
                sort_keys=True,
                separators=(",", ":"),
            ),
        ]
        daemon = object.__new__(PortalImplementationDaemon)
        daemon._scoped_control_plane = invocation.control_plane
        daemon._scoped_control_plane_launch = launch
        daemon._canonical_ref = lambda _task: invocation.task_revision_cid

        callback = daemon._provider_runner_started_callback(
            state,
            task=SimpleNamespace(task_id=invocation.task_id),
            attempt=invocation.attempt,
            command=command,
            workspace_path=workspace,
        )

        assert callback is not None
    finally:
        os.close(launch.descriptor)


def test_scoped_fresh_callback_is_mandatory_but_recovery_is_not(
    tmp_path,
    sealed_control_plane_fd: tuple[int, str],
) -> None:
    descriptor, digest = sealed_control_plane_fd
    status, command, _ = _sealed_receipt_case(
        tmp_path,
        descriptor=descriptor,
        archive_sha256=digest,
    )
    daemon = object.__new__(PortalImplementationDaemon)
    daemon._scoped_control_plane_launch = SimpleNamespace(
        descriptor=descriptor,
        executable_path=f"/proc/self/fd/{descriptor}",
        archive_sha256=digest,
        seals=int(fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)),
        capsule_id="accepted-test-capsule",
    )
    daemon._scoped_control_plane = _fake_control_plane(command)
    state = PortalTaskState()
    task = SimpleNamespace(task_id=status["active_task_id"])
    malformed = list(command)
    malformed[malformed.index("--agent-implementation-route-json") + 1] = (
        "--missing-route-value"
    )

    with pytest.raises(RuntimeError, match="birth receipt cannot be constructed"):
        daemon._provider_runner_started_callback(
            state,
            task=task,
            attempt=1,
            command=malformed,
            workspace_path=tmp_path / "task-worktree",
        )

    wrong_descriptor = list(command)
    wrong_descriptor[2] = f"/proc/self/fd/{descriptor + 1}"
    with pytest.raises(RuntimeError, match="command is malformed"):
        daemon._provider_runner_started_callback(
            state,
            task=task,
            attempt=1,
            command=wrong_descriptor,
            workspace_path=tmp_path / "task-worktree",
        )

    recovery = command[:5] + ["--agent-implementation-recovery-json", "{}"]
    assert (
        daemon._provider_runner_started_callback(
            state,
            task=task,
            attempt=1,
            command=recovery,
            workspace_path=tmp_path / "task-worktree",
        )
        is None
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_key",
        "noncanonical",
        "attempt_bool",
        "attempt_float",
        "route_id",
        "board_namespace",
        "task_revision",
        "workspace",
        "store_identity",
        "control_plane",
    ],
)
def test_scoped_fresh_callback_rejects_locator_drift(
    tmp_path,
    sealed_control_plane_fd: tuple[int, str],
    mutation: str,
) -> None:
    descriptor, digest = sealed_control_plane_fd
    status, command, _ = _sealed_receipt_case(
        tmp_path,
        descriptor=descriptor,
        archive_sha256=digest,
    )
    state = PortalTaskState(
        **{
            key: value
            for key, value in status.items()
            if key in PortalTaskState.__dataclass_fields__
        }
    )
    daemon = object.__new__(PortalImplementationDaemon)
    daemon._scoped_control_plane_launch = SimpleNamespace(
        descriptor=descriptor,
        executable_path=f"/proc/self/fd/{descriptor}",
        archive_sha256=digest,
        seals=int(fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)),
        capsule_id="accepted-test-capsule",
    )
    daemon._scoped_control_plane = _fake_control_plane(command)
    daemon._canonical_ref = lambda _task: status["active_task_cid"]
    route_index = command.index("--agent-implementation-route-json")
    raw = command[route_index + 1]
    binding = json.loads(raw)
    if mutation == "duplicate_key":
        command[route_index + 1] = raw[:-1] + ',"route_id":"duplicate"}'
    elif mutation == "noncanonical":
        command[route_index + 1] = json.dumps(binding, indent=2)
    else:
        invocation = binding["invocation_binding"]
        if mutation == "attempt_bool":
            invocation["attempt"] = True
        elif mutation == "attempt_float":
            invocation["attempt"] = 1.0
        elif mutation == "route_id":
            invocation["route_id"] = "drifted"
        elif mutation == "board_namespace":
            binding["authorization"]["board_namespace"] = "drifted"
        elif mutation == "task_revision":
            invocation["task_revision_cid"] = "drifted"
        elif mutation == "workspace":
            invocation["workspace_path"] = str(tmp_path / "other")
        elif mutation == "store_identity":
            invocation["provider_attempt_store_identity"] = "sha256:" + "9" * 64
        elif mutation == "control_plane":
            invocation["control_plane"]["archive_sha256"] = "sha256:" + "9" * 64
        command[route_index + 1] = json.dumps(
            binding,
            sort_keys=True,
            separators=(",", ":"),
        )

    with pytest.raises(RuntimeError):
        daemon._provider_runner_started_callback(
            state,
            task=SimpleNamespace(task_id=status["active_task_id"]),
            attempt=1,
            command=command,
            workspace_path=tmp_path / "task-worktree",
        )


def test_unsealed_route_json_keeps_existing_launch_behavior() -> None:
    daemon = object.__new__(PortalImplementationDaemon)
    daemon._scoped_control_plane_launch = SimpleNamespace(
        executable_path="/proc/self/fd/99"
    )

    callback = daemon._provider_runner_started_callback(
        SimpleNamespace(),
        task=SimpleNamespace(task_id="TASK-1"),
        attempt=1,
        command=[
            sys.executable,
            "-m",
            "ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner",
            "--agent-implementation-route-json",
            "{}",
        ],
        workspace_path=SimpleNamespace(),
    )

    assert callback is None


def test_state_loader_rejects_boolean_active_attempt(tmp_path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text('{"active_attempt":true}', encoding="utf-8")

    assert state_file_repair_reason(state_path) == "malformed_state_metadata"
    assert PortalTaskState.load(state_path).active_attempt == 0


def test_callback_failure_strictly_fences_child(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}
    cleanup_calls: list[dict[str, Any]] = []
    real_terminate = supervisor_runtime.terminate_pid_tree

    def tracked_terminate(pid, **kwargs):
        cleanup_calls.append({"pid": pid, **kwargs})
        return real_terminate(pid, **kwargs)

    monkeypatch.setattr(
        supervisor_runtime,
        "terminate_pid_tree",
        tracked_terminate,
    )
    descendant_path = tmp_path / "forked-child.pid"
    code = (
        "import os,pathlib,time; "
        "child=os.fork(); "
        f"path=pathlib.Path({str(descendant_path)!r}); "
        "(path.write_text(str(child)) if child else time.sleep(30)); "
        "os._exit(0) if child else None"
    )

    def fail(process) -> None:
        observed["pid"] = process.pid
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                state = (
                    Path(f"/proc/{process.pid}/stat")
                    .read_text(encoding="utf-8")
                    .rsplit(")", 1)[1]
                    .split()[0]
                )
            except OSError:
                state = ""
            if descendant_path.exists() and state == "Z":
                break
            time.sleep(0.01)
        assert descendant_path.exists()
        observed["descendant_pid"] = int(
            descendant_path.read_text(encoding="utf-8")
        )
        assert state == "Z"
        raise SystemExit("receipt persistence interrupted")

    with pytest.raises(SystemExit, match="receipt persistence interrupted"):
        run_process_group_stream(
            [sys.executable, "-c", code],
            cwd=tmp_path,
            stdout=None,
            input_text="prompt withheld until callback",
            timeout_seconds=60,
            on_started=fail,
        )

    assert observed["pid"] > 1
    assert not os.path.exists(f"/proc/{observed['pid']}")
    assert not os.path.exists(f"/proc/{observed['descendant_pid']}")
    assert len(cleanup_calls) == 1
    assert cleanup_calls[0]["pid"] == observed["pid"]
    assert cleanup_calls[0]["freeze_first"] is True
    assert cleanup_calls[0]["require_gone"] is True
    assert cleanup_calls[0]["owned_process_group_id"] == observed["pid"]


@pytest.mark.parametrize(
    "cmdline",
    [
        (
            "/home/example/.local/bin/python -m "
            "ipfs_accelerate_py.agent_supervisor.grok_cli_runner "
            "--workspace /tmp/task --model grok-4.5"
        ),
        (
            "/usr/bin/python3 -P -m "
            "ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner "
            "--workspace /tmp/task --model grok-4.5"
        ),
        (
            "/usr/bin/python3 /opt/ipfs/agent_supervisor/grok_cli_runner.py "
            "--workspace /tmp/task --model grok-4.5"
        ),
    ],
)
def test_watchdog_recognizes_packaged_grok_runner(
    monkeypatch: pytest.MonkeyPatch,
    cmdline: str,
) -> None:
    now = datetime.now(UTC)
    monkeypatch.setattr(
        supervisor,
        "descendant_processes",
        lambda _pid: [{"pid": 4322, "cmdline": cmdline}],
    )

    status = supervisor.worktree_phase_worker_status(
        {
            "active_phase": "implementing",
            "active_phase_started_at": (now - timedelta(minutes=10)).isoformat(),
        },
        daemon_pid=1234,
        threshold_seconds=60,
        now=now,
    )

    assert status["active_worker_count"] == 1
    assert status["active_worker_pids"] == [4322]
    assert status["stalled_without_active_worker"] is False


@pytest.mark.parametrize(
    "cmdline",
    [
        "/usr/bin/python3 -m pytest test/api -q",
        (
            "/usr/bin/python3 -m "
            "ipfs_accelerate_py.agent_supervisor.grok_cli_runner_helper"
        ),
        "/usr/bin/python3 /opt/ipfs/agent_supervisor/not_grok_cli_runner.py",
    ],
)
def test_watchdog_does_not_treat_arbitrary_python_as_agent_worker(
    monkeypatch: pytest.MonkeyPatch,
    cmdline: str,
) -> None:
    now = datetime.now(UTC)
    monkeypatch.setattr(
        supervisor,
        "descendant_processes",
        lambda _pid: [
            {
                "pid": 4323,
                "cmdline": cmdline,
            }
        ],
    )

    status = supervisor.worktree_phase_worker_status(
        {
            "active_phase": "implementing",
            "active_phase_started_at": (now - timedelta(minutes=10)).isoformat(),
        },
        daemon_pid=1234,
        threshold_seconds=60,
        now=now,
    )

    assert status["active_worker_count"] == 0
    assert status["stalled_without_active_worker"] is True


def test_supervisor_loop_graces_packaged_runner_disappearance(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    state_dir = repo / "state"
    state_dir.mkdir(parents=True)
    spec = ManagedDaemonSpec(
        name="test-daemon",
        schema="test.daemon",
        repo_root=repo,
        daemon_dir=state_dir,
        runner=(sys.executable, "-c", "pass"),
        status_path=state_dir / "daemon_status.json",
        supervisor_status_path=state_dir / "supervisor_status.json",
        supervisor_pid_path=state_dir / "supervisor.pid",
        child_pid_path=state_dir / "child.pid",
        supervisor_out_path=state_dir / "supervisor.out",
        ensure_status_path=state_dir / "ensure_status.json",
        ensure_check_path=state_dir / "ensure_check.json",
    )
    clock = [100.0]
    loop = SupervisorLoop(
        SupervisorLoopConfig(
            spec=spec,
            command=(sys.executable, "-c", "pass"),
            log_prefix="child",
            watchdog_stale_after_seconds=60,
        ),
        monotonic=lambda: clock[0],
    )
    runner_live = [True]
    monkeypatch.setattr(
        supervisor,
        "descendant_processes",
        lambda _pid: (
            [
                {
                    "pid": 4324,
                    "cmdline": (
                        "/usr/bin/python3.12 -m "
                        "ipfs_accelerate_py.agent_supervisor.grok_cli_runner "
                        "--workspace /tmp/task --model grok-4.5"
                    ),
                }
            ]
            if runner_live[0]
            else []
        ),
    )
    now = datetime.now(UTC)
    status = {
        "heartbeat_at": now.isoformat(),
        "active_phase": "implementing",
        "active_phase_started_at": (now - timedelta(minutes=10)).isoformat(),
        "worktree_no_child_stall_seconds": 60,
    }
    child = SimpleNamespace(pid=1234)

    live = loop.default_watchdog(child, status)
    assert live.action == "continue"
    assert loop._last_worker_status["active_worker_count"] == 1
    assert loop._last_worker_status["worker_absence_age_seconds"] == 0.0

    runner_live[0] = False
    clock[0] = 110.0
    within_grace = loop.default_watchdog(child, status)
    assert within_grace.action == "continue"
    assert loop._last_worker_status["active_worker_count"] == 0
    assert loop._last_worker_status["worker_absence_age_seconds"] == 10.0

    clock[0] = 161.0
    expired = loop.default_watchdog(child, status)
    assert expired.action == "recycle"
    assert expired.reason == "worktree_phase_without_active_child"
    assert expired.detail["worker_absence_age_seconds"] == 61.0
