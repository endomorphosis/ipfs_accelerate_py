"""Focused ASEH bootstrap tests for the existing Quack supervisor handoff."""

from __future__ import annotations

import fcntl
import hashlib
import inspect
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import zipfile
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    current_process_birth,
    read_process_birth,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    configured_board_scheduler as configured_scheduler,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    multi_supervisor_runner as multi_runner,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    process_security as process_security_module,
)
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    DatabaseProgramConfig,
    provider_subprocess_environment,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    QuackStateServerControlError,
    QuackStateServerReadyError,
    TypedStateOwnerGrantBroker,
    build_server,
)
from ipfs_accelerate_py.agent_supervisor.task_sources import (
    typed_state_owner as typed_state_owner_module,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
    TaskSourceBoundsError,
    TaskSourceUnknownOutcomeError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    QUACK_OWNER_COMMAND_COMPARE_AND_SET_STATUS,
    QuackOwnerCommandRemoteError,
    reset_quack_transport_cache,
    submit_quack_owner_command,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    task_authority_spec_cid,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
    QuackCapabilityStatus,
    probe_quack_capabilities,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    DATABASE_TASK_COMMANDS,
    TYPED_STATE_OWNER_GRANT_BROKER_SCHEMA,
    TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV,
    TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_ENV,
    TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_FILENAME,
    TYPED_STATE_OWNER_SOCKET_ENV,
    TypedStateOwnerAuthorizationError,
    TypedStateOwnerConnection,
    TypedStateOwnerError,
    TypedStateOwnerRemoteError,
    kernel_process_birth_id,
    request_database_task_command_credential,
    request_quack_attach_credential,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    supervisor as todo_supervisor,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    supervisor_loop as supervisor_loop_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    supervisor_runtime as supervisor_runtime_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.core import ManagedDaemonSpec
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_loop import (
    SupervisorLoop,
    SupervisorLoopConfig,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_runtime import (
    SUPERVISED_CHILD_IDENTITY_PATH_ENV,
    SUPERVISED_CHILD_OWNER_SCOPE_ENV,
)
from ipfs_accelerate_py.agent_supervisor.validation.validation_runtime import (
    build_validation_environment,
)

from scripts import run_agent_supervisor_efficiency_state_hardening as aseh_operator


def _sealed_validation_execution_fixture(
    command: tuple[str, ...],
    *,
    candidate_head: str,
    candidate_tree: str,
    authorization_witness: dict[str, str],
    environment_identity: str,
    stdout_digest: str,
    stderr_digest: str,
    returncode: int = 0,
) -> dict[str, object] | None:
    if (
        aseh_operator._parse_receipt_validation_python_command(
            command,
            require_known=True,
        )
        is None
    ):
        return None
    native_mode = (
        "self_qualified"
        if aseh_operator._uses_historical_self_qualified_native_route(command)
        else "preloaded"
    )
    execution_context = aseh_operator._sealed_validation_execution_context(
        declared=command,
        candidate_head=candidate_head,
        candidate_tree=candidate_tree,
        authorization_witness=authorization_witness,
        environment_identity=environment_identity,
        native_authorization_id=(
            aseh_operator.ASEH_R11_NATIVE_DEPENDENCY_AUTHORIZATION_ID
        ),
    )
    execution_context_cid = aseh_operator._identity(execution_context)
    ready = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "aseh-sealed-validation-ready@1"
        ),
        "nonce": "a" * 64,
        "pid": 43210,
        "start_time_ticks": 123456,
        "boot_id": "00000000-0000-0000-0000-000000000000",
        "declared_argv_sha256": aseh_operator._identity(list(command)),
        "bootstrap_sha256": (
            aseh_operator.ASEH_RECEIPT_VALIDATION_BOOTSTRAP_SHA256
        ),
        "native_authorization_id": (
            aseh_operator.ASEH_R11_NATIVE_DEPENDENCY_AUTHORIZATION_ID
        ),
        "native_mode": native_mode,
        "interpreter_sha256": (
            aseh_operator.ASEH_R11_NATIVE_DEPENDENCY_PIN[
                "python_executable_sha256"
            ]
        ),
        "execution_context_cid": execution_context_cid,
    }
    ready_bytes = (
        aseh_operator._canonical_json(ready) + "\n"
    ).encode("utf-8")
    ready_sha = "sha256:" + hashlib.sha256(ready_bytes).hexdigest()
    release = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "aseh-sealed-validation-release@1"
        ),
        "nonce": ready["nonce"],
        "ready_sha256": ready_sha,
    }
    completion = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "aseh-sealed-validation-completion@1"
        ),
        "nonce": ready["nonce"],
        "pid": ready["pid"],
        "returncode": returncode,
        "ready_sha256": ready_sha,
        "native_mode": native_mode,
        "descendants_drained": True,
        "execution_context_cid": execution_context_cid,
    }
    release_bytes = (
        aseh_operator._canonical_json(release) + "\n"
    ).encode("utf-8")
    completion_bytes = (
        aseh_operator._canonical_json(completion) + "\n"
    ).encode("utf-8")
    executor_contract_cid = aseh_operator._identity(
        aseh_operator._sealed_receipt_validation_executor_contract()
    )
    completion_receipt_sha256 = (
        "sha256:" + hashlib.sha256(completion_bytes).hexdigest()
    )
    execution_binding = aseh_operator._sealed_validation_execution_binding(
        execution_context=execution_context,
        executor_contract_cid=executor_contract_cid,
        ready_receipt_sha256=ready_sha,
        completion_receipt_sha256=completion_receipt_sha256,
        returncode=returncode,
        stdout_digest=stdout_digest,
        stderr_digest=stderr_digest,
    )
    evidence: dict[str, object] = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "aseh-sealed-validation-execution@1"
        ),
        "executor_contract_cid": executor_contract_cid,
        "ready": ready,
        "release": release,
        "completion": completion,
        "ready_receipt_sha256": ready_sha,
        "release_receipt_sha256": (
            "sha256:" + hashlib.sha256(release_bytes).hexdigest()
        ),
        "completion_receipt_sha256": completion_receipt_sha256,
        "execution_context": execution_context,
        "execution_context_cid": execution_context_cid,
        "execution_binding": execution_binding,
        "execution_binding_cid": aseh_operator._identity(execution_binding),
    }
    evidence["evidence_cid"] = aseh_operator._identity(evidence)
    return evidence


def _assert_process_not_executable(pid: int) -> None:
    deadline = time.monotonic() + 2.0
    while True:
        try:
            raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        except OSError:
            return
        if raw.rsplit(")", 1)[1].split()[0] == "Z":
            return
        if time.monotonic() >= deadline:
            pytest.fail(f"process {pid} remained executable")
        time.sleep(0.01)


def test_aseh_scheduler_group_fence_survives_leader_exit(
    tmp_path: Path,
) -> None:
    child_marker = tmp_path / "scheduler-child.pid"
    release = tmp_path / "release-leader"
    process = subprocess.Popen(
        (
            sys.executable,
            "-c",
            (
                "import os,signal,time,pathlib; child=os.fork(); "
                f"marker=pathlib.Path({str(child_marker)!r}); "
                f"release=pathlib.Path({str(release)!r}); "
                "(marker.write_text(str(os.getpid())) if child==0 else None); "
                "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                "(time.sleep(60) if child==0 else "
                "exec('while not release.exists():\\n time.sleep(.01)'))"
            ),
        ),
        start_new_session=True,
    )
    try:
        start_time_ticks = aseh_operator._dedicated_process_group_birth(process)
        deadline = time.monotonic() + 5.0
        while not child_marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert child_marker.exists()
        release.write_text("release\n", encoding="utf-8")
        process.wait(timeout=5.0)
        child_pid = int(child_marker.read_text(encoding="utf-8"))

        aseh_operator._terminate_scheduler(process, start_time_ticks)

        _assert_process_not_executable(child_pid)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5.0)


def test_aseh_forced_owner_group_escalation_reaps_term_ignoring_tree(
    tmp_path: Path,
) -> None:
    child_marker = tmp_path / "owner-child.pid"
    process = subprocess.Popen(
        (
            sys.executable,
            "-c",
            (
                "import os,signal,time,pathlib; child=os.fork(); "
                f"marker=pathlib.Path({str(child_marker)!r}); "
                "(marker.write_text(str(os.getpid())) if child==0 else None); "
                "signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"
            ),
        ),
        start_new_session=True,
    )
    try:
        start_time_ticks = aseh_operator._dedicated_process_group_birth(process)
        deadline = time.monotonic() + 5.0
        while not child_marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert child_marker.exists()
        child_pid = int(child_marker.read_text(encoding="utf-8"))
        started = time.monotonic()

        aseh_operator._terminate_dedicated_process_group(
            process,
            start_time_ticks=start_time_ticks,
            grace_seconds=0.05,
        )

        assert time.monotonic() - started < 3.0
        assert process.poll() is not None
        _assert_process_not_executable(child_pid)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5.0)

_CWD_OWNER_DIR = Path("/proc/self/cwd/quack-owner")


def test_state_authority_handoff_rejects_unqualified_ptrace_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        process_security_module,
        "_read_bounded_proc_text",
        lambda *_args, **_kwargs: "0\n",
    )
    monkeypatch.setattr(
        process_security_module,
        "_same_user_namespace_ptrace_capability_pids",
        lambda: (),
    )
    with pytest.raises(
        process_security_module.StateAuthorityProcessIsolationError,
        match="ptrace_scope >= 1",
    ):
        process_security_module.require_state_authority_handoff_ptrace_protection()


def test_state_authority_handoff_rejects_cap_sys_ptrace_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        process_security_module,
        "_read_bounded_proc_text",
        lambda *_args, **_kwargs: "1\n",
    )
    monkeypatch.setattr(
        process_security_module,
        "_same_user_namespace_ptrace_capability_pids",
        lambda: (4321,),
    )
    with pytest.raises(
        process_security_module.StateAuthorityProcessIsolationError,
        match="CAP_SYS_PTRACE",
    ):
        process_security_module.require_state_authority_handoff_ptrace_protection()


@pytest.mark.skipif(
    not hasattr(os, "memfd_create"),
    reason="sealed Linux memfd handoff is required",
)
def test_state_authority_handoff_redeems_sealed_fd_after_actual_exec() -> None:
    """Exercise the post-exec SCM_RIGHTS path without changing pytest's dumpability."""

    repository_root = Path.cwd().resolve()
    child_code = "\n".join(
        (
            "import json, os",
            (
                "from ipfs_accelerate_py.agent_supervisor.runtime.process_security "
                "import receive_state_authority_child_handoff, state_authority_pass_fds"
            ),
            "assert receive_state_authority_child_handoff(os.environ) is True",
            "descriptors = state_authority_pass_fds(os.environ)",
            "assert len(descriptors) == 1",
            "descriptor = descriptors[0]",
            "os.lseek(descriptor, 0, os.SEEK_SET)",
            "payload = os.read(descriptor, 256).decode('ascii')",
            "inheritable = os.get_inheritable(descriptor)",
            "os.close(descriptor)",
            "print(json.dumps({'payload': payload, 'inheritable': inheritable}))",
        )
    )
    parent_code = "\n".join(
        (
            "import fcntl, json, os, subprocess, sys",
            (
                "from ipfs_accelerate_py.agent_supervisor.runtime.process_security "
                "import STATE_AUTHORITY_PARENT_LOSS_TERMINATE, "
                "prepare_state_authority_child_handoff"
            ),
            "flags = int(getattr(os, 'MFD_CLOEXEC', 1)) | int(getattr(os, 'MFD_ALLOW_SEALING', 2))",
            "descriptor = os.memfd_create('aseh-exec-handoff', flags=flags)",
            "os.write(descriptor, b'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')",
            (
                "seals = int(getattr(fcntl, 'F_SEAL_SEAL', 1)) | "
                "int(getattr(fcntl, 'F_SEAL_SHRINK', 2)) | "
                "int(getattr(fcntl, 'F_SEAL_GROW', 4)) | "
                "int(getattr(fcntl, 'F_SEAL_WRITE', 8))"
            ),
            "fcntl.fcntl(descriptor, int(getattr(fcntl, 'F_ADD_SEALS', 1033)), seals)",
            "environment = {'PATH': '/usr/bin:/bin', 'IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SOCKET': '/tmp/aseh-exec-handoff.sock', 'IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD': str(descriptor)}",
            (
                "handoff = prepare_state_authority_child_handoff("
                "environment, parent_loss_policy="
                "STATE_AUTHORITY_PARENT_LOSS_TERMINATE)"
            ),
            "assert 'IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD' not in environment",
            f"child_code = {child_code!r}",
            "process = None",
            "try:",
            f"    process = subprocess.Popen([sys.executable, '-c', child_code], cwd={str(repository_root)!r}, env=environment, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)",
            "    handoff.deliver(process)",
            "    stdout, stderr = process.communicate(timeout=5.0)",
            "    if process.returncode != 0: raise RuntimeError(stderr.decode(errors='replace'))",
            "    print(stdout.decode('utf-8').strip())",
            "finally:",
            "    handoff.close()",
            "    if process is not None and process.poll() is None:",
            "        process.kill()",
            "        process.wait(timeout=2.0)",
            "    os.close(descriptor)",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", parent_code],
        cwd=repository_root,
        env={"PATH": "/usr/bin:/bin"},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15.0,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    assert json.loads(completed.stdout) == {
        "payload": "x" * 64,
        "inheritable": False,
    }


@pytest.mark.skipif(
    not hasattr(os, "memfd_create"),
    reason="sealed Linux memfd handoff is required",
)
def test_state_authority_handoff_parent_loss_signals_redeemed_child(
    tmp_path: Path,
) -> None:
    repository_root = Path.cwd().resolve()
    child_pid_path = tmp_path / "child.pid"
    child_ready_path = tmp_path / "child.ready"
    child_term_path = tmp_path / "child.term"
    child_code = "\n".join(
        (
            "import os, pathlib, signal, time",
            (
                "from ipfs_accelerate_py.agent_supervisor.runtime.process_security "
                "import receive_state_authority_child_handoff"
            ),
            f"pid_path = pathlib.Path({str(child_pid_path)!r})",
            f"ready_path = pathlib.Path({str(child_ready_path)!r})",
            f"term_path = pathlib.Path({str(child_term_path)!r})",
            "pid_path.write_text(str(os.getpid()))",
            (
                "signal.signal(signal.SIGTERM, lambda *_args: "
                "(term_path.write_text('term\\n'), raise_exit()))"
            ),
            "assert receive_state_authority_child_handoff(os.environ) is True",
            "ready_path.write_text('ready\\n')",
            "while True: time.sleep(.05)",
        )
    ).replace(
        "import os, pathlib, signal, time",
        (
            "import os, pathlib, signal, time\n"
            "def raise_exit(): raise SystemExit(0)"
        ),
    )
    parent_code = "\n".join(
        (
            "import fcntl, os, pathlib, subprocess, sys, time",
            (
                "from ipfs_accelerate_py.agent_supervisor.runtime.process_security "
                "import STATE_AUTHORITY_PARENT_LOSS_TERMINATE, "
                "prepare_state_authority_child_handoff"
            ),
            "flags = int(getattr(os, 'MFD_CLOEXEC', 1)) | int(getattr(os, 'MFD_ALLOW_SEALING', 2))",
            "descriptor = os.memfd_create('aseh-parent-loss', flags=flags)",
            "os.write(descriptor, b'x' * 64)",
            (
                "seals = int(getattr(fcntl, 'F_SEAL_SEAL', 1)) | "
                "int(getattr(fcntl, 'F_SEAL_SHRINK', 2)) | "
                "int(getattr(fcntl, 'F_SEAL_GROW', 4)) | "
                "int(getattr(fcntl, 'F_SEAL_WRITE', 8))"
            ),
            "fcntl.fcntl(descriptor, int(getattr(fcntl, 'F_ADD_SEALS', 1033)), seals)",
            "environment = {'PATH': '/usr/bin:/bin', 'IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SOCKET': '/tmp/aseh-parent-loss.sock', 'IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD': str(descriptor)}",
            (
                "handoff = prepare_state_authority_child_handoff("
                "environment, parent_loss_policy="
                "STATE_AUTHORITY_PARENT_LOSS_TERMINATE)"
            ),
            f"child_code = {child_code!r}",
            f"process = subprocess.Popen([sys.executable, '-c', child_code], cwd={str(repository_root)!r}, env=environment, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)",
            "handoff.deliver(process)",
            f"ready = pathlib.Path({str(child_ready_path)!r})",
            "deadline = time.monotonic() + 5.0",
            "while not ready.exists() and time.monotonic() < deadline: time.sleep(.01)",
            "if not ready.exists(): raise SystemExit(79)",
            "os._exit(0)",
        )
    )
    parent = subprocess.Popen(
        (sys.executable, "-c", parent_code),
        cwd=repository_root,
        env={"PATH": "/usr/bin:/bin"},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    stderr = b""
    try:
        _stdout, stderr = parent.communicate(timeout=10.0)
        assert parent.returncode == 0, stderr.decode(errors="replace")
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 5.0
        observed_term = ""
        while time.monotonic() < deadline:
            try:
                observed_term = child_term_path.read_text(encoding="utf-8")
            except FileNotFoundError:
                observed_term = ""
            if observed_term == "term\n":
                break
            time.sleep(0.01)
        assert observed_term == "term\n"
        _assert_process_not_executable(child_pid)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=2.0)


@pytest.mark.skipif(
    not hasattr(os, "memfd_create"),
    reason="sealed Linux memfd handoff is required",
)
def test_state_authority_handoff_detached_policy_survives_launcher_exit(
    tmp_path: Path,
) -> None:
    repository_root = Path.cwd().resolve()
    child_pid_path = tmp_path / "detached-child.pid"
    child_ready_path = tmp_path / "detached-child.ready"
    child_stop_path = tmp_path / "detached-child.stop"
    child_code = "\n".join(
        (
            "import os, pathlib, time",
            (
                "from ipfs_accelerate_py.agent_supervisor.runtime.process_security "
                "import receive_state_authority_child_handoff"
            ),
            f"pid_path = pathlib.Path({str(child_pid_path)!r})",
            f"ready_path = pathlib.Path({str(child_ready_path)!r})",
            f"stop_path = pathlib.Path({str(child_stop_path)!r})",
            "pid_path.write_text(str(os.getpid()))",
            "assert receive_state_authority_child_handoff(os.environ) is True",
            "ready_path.write_text('ready\\n')",
            "while not stop_path.exists(): time.sleep(.02)",
        )
    )
    parent_code = "\n".join(
        (
            "import fcntl, os, pathlib, subprocess, sys, time",
            (
                "from ipfs_accelerate_py.agent_supervisor.runtime.process_security "
                "import STATE_AUTHORITY_PARENT_LOSS_DETACHED, "
                "prepare_state_authority_child_handoff"
            ),
            "flags = int(getattr(os, 'MFD_CLOEXEC', 1)) | int(getattr(os, 'MFD_ALLOW_SEALING', 2))",
            "descriptor = os.memfd_create('aseh-detached-handoff', flags=flags)",
            "os.write(descriptor, b'x' * 64)",
            (
                "seals = int(getattr(fcntl, 'F_SEAL_SEAL', 1)) | "
                "int(getattr(fcntl, 'F_SEAL_SHRINK', 2)) | "
                "int(getattr(fcntl, 'F_SEAL_GROW', 4)) | "
                "int(getattr(fcntl, 'F_SEAL_WRITE', 8))"
            ),
            "fcntl.fcntl(descriptor, int(getattr(fcntl, 'F_ADD_SEALS', 1033)), seals)",
            "environment = {'PATH': '/usr/bin:/bin', 'IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SOCKET': '/tmp/aseh-detached-handoff.sock', 'IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD': str(descriptor)}",
            (
                "handoff = prepare_state_authority_child_handoff("
                "environment, parent_loss_policy="
                "STATE_AUTHORITY_PARENT_LOSS_DETACHED)"
            ),
            f"child_code = {child_code!r}",
            f"process = subprocess.Popen([sys.executable, '-c', child_code], cwd={str(repository_root)!r}, env=environment, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)",
            "handoff.deliver(process)",
            f"ready = pathlib.Path({str(child_ready_path)!r})",
            "deadline = time.monotonic() + 5.0",
            "while not ready.exists() and time.monotonic() < deadline: time.sleep(.01)",
            "if not ready.exists(): raise SystemExit(79)",
            "os._exit(0)",
        )
    )
    parent = subprocess.Popen(
        (sys.executable, "-c", parent_code),
        cwd=repository_root,
        env={"PATH": "/usr/bin:/bin"},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    child_pid = 0
    child_start_ticks = 0
    try:
        _stdout, stderr = parent.communicate(timeout=10.0)
        assert parent.returncode == 0, stderr.decode(errors="replace")
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        child_stat = Path(f"/proc/{child_pid}/stat").read_text(
            encoding="utf-8"
        )
        child_start_ticks = int(
            child_stat[child_stat.rfind(")") + 2 :].split()[19]
        )
        assert child_ready_path.read_text(encoding="utf-8") == "ready\n"
        assert Path(f"/proc/{child_pid}").exists()
        child_stop_path.write_text("stop\n", encoding="utf-8")
        _assert_process_not_executable(child_pid)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=2.0)
        if child_pid > 1 and Path(f"/proc/{child_pid}").exists():
            try:
                current = Path(f"/proc/{child_pid}/stat").read_text(
                    encoding="utf-8"
                )
                current_start = int(
                    current[current.rfind(")") + 2 :].split()[19]
                )
                if current_start == child_start_ticks:
                    os.kill(child_pid, signal.SIGKILL)
            except (OSError, ValueError):
                pass


def test_state_authority_handoff_requires_exact_parent_loss_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        process_security_module,
        "state_authority_pass_fds",
        lambda _environment: (9,),
    )
    environment = {
        "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD": "9",
    }
    with pytest.raises(
        process_security_module.StateAuthorityProcessIsolationError,
        match="parent-loss policy is required",
    ):
        process_security_module.prepare_state_authority_child_handoff(
            dict(environment)
        )
    with pytest.raises(
        process_security_module.StateAuthorityProcessIsolationError,
        match="parent-loss policy is required",
    ):
        process_security_module.prepare_state_authority_child_handoff(
            dict(environment),
            parent_loss_policy="ambient_override",
        )


@pytest.mark.skipif(
    not hasattr(os, "memfd_create"),
    reason="sealed Linux memfd handoff is required",
)
def test_state_authority_handoff_policy_mismatch_denies_descriptor() -> None:
    """The child cannot weaken the exact parent-authenticated lifecycle mode."""

    repository_root = Path.cwd().resolve()
    child_code = "\n".join(
        (
            "import json, os",
            (
                "from ipfs_accelerate_py.agent_supervisor.runtime.process_security "
                "import receive_state_authority_child_handoff"
            ),
            "denied = False",
            "try:",
            "    receive_state_authority_child_handoff(os.environ)",
            "except BaseException:",
            "    denied = True",
            (
                "print(json.dumps({'denied': denied, 'descriptor_present': "
                "bool(os.environ.get('IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD'))}))"
            ),
            "raise SystemExit(0 if denied else 79)",
        )
    )
    parent_code = "\n".join(
        (
            "import fcntl, json, os, subprocess, sys",
            (
                "from ipfs_accelerate_py.agent_supervisor.runtime.process_security "
                "import STATE_AUTHORITY_HANDOFF_PARENT_LOSS_POLICY_ENV, "
                "STATE_AUTHORITY_PARENT_LOSS_DETACHED, "
                "STATE_AUTHORITY_PARENT_LOSS_TERMINATE, "
                "StateAuthorityProcessIsolationError, "
                "prepare_state_authority_child_handoff"
            ),
            (
                "flags = int(getattr(os, 'MFD_CLOEXEC', 1)) | "
                "int(getattr(os, 'MFD_ALLOW_SEALING', 2))"
            ),
            "descriptor = os.memfd_create('aseh-policy-mismatch', flags=flags)",
            "os.write(descriptor, b'x' * 64)",
            (
                "seals = int(getattr(fcntl, 'F_SEAL_SEAL', 1)) | "
                "int(getattr(fcntl, 'F_SEAL_SHRINK', 2)) | "
                "int(getattr(fcntl, 'F_SEAL_GROW', 4)) | "
                "int(getattr(fcntl, 'F_SEAL_WRITE', 8))"
            ),
            (
                "fcntl.fcntl(descriptor, "
                "int(getattr(fcntl, 'F_ADD_SEALS', 1033)), seals)"
            ),
            (
                "environment = {'PATH': '/usr/bin:/bin', "
                "'IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SOCKET': "
                "'/tmp/aseh-policy-mismatch.sock', "
                "'IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD': "
                "str(descriptor)}"
            ),
            (
                "handoff = prepare_state_authority_child_handoff("
                "environment, parent_loss_policy="
                "STATE_AUTHORITY_PARENT_LOSS_TERMINATE)"
            ),
            (
                "environment[STATE_AUTHORITY_HANDOFF_PARENT_LOSS_POLICY_ENV] = "
                "STATE_AUTHORITY_PARENT_LOSS_DETACHED"
            ),
            f"child_code = {child_code!r}",
            "process = None",
            "try:",
            (
                f"    process = subprocess.Popen([sys.executable, '-c', child_code], cwd={str(repository_root)!r}, "
                "env=environment, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)"
            ),
            "    denied = False",
            "    try:",
            "        handoff.deliver(process, timeout_seconds=.5)",
            "    except StateAuthorityProcessIsolationError:",
            "        denied = True",
            "    stdout, stderr = process.communicate(timeout=5.0)",
            "    if process.returncode != 0: raise RuntimeError(stderr.decode(errors='replace'))",
            "    child = json.loads(stdout)",
            "    print(json.dumps({'parent_denied': denied, 'child': child}))",
            "finally:",
            "    handoff.close()",
            "    if process is not None and process.poll() is None:",
            "        process.kill()",
            "        process.wait(timeout=2.0)",
            "    os.close(descriptor)",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", parent_code],
        cwd=repository_root,
        env={"PATH": "/usr/bin:/bin"},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15.0,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    assert json.loads(completed.stdout) == {
        "parent_denied": True,
        "child": {"denied": True, "descriptor_present": False},
    }


def test_state_authority_parent_loss_policy_is_explicit_at_all_launchers() -> None:
    """Every authority-bearing production edge declares its lifecycle owner."""

    terminate = "parent_loss_policy=STATE_AUTHORITY_PARENT_LOSS_TERMINATE"
    detached = "parent_loss_policy=STATE_AUTHORITY_PARENT_LOSS_DETACHED"

    terminate_sources = (
        inspect.getsource(
            configured_scheduler._launch_foreground_plan_bound_coordinator
        ),
        inspect.getsource(multi_runner.start_track),
        inspect.getsource(aseh_operator._run_supervisor_owner),
        inspect.getsource(supervisor_runtime_module.launch_supervised_child),
    )
    detached_sources = (
        inspect.getsource(
            configured_scheduler._launch_detached_plan_bound_coordinator
        ),
        inspect.getsource(
            configured_scheduler._launch_detached_receipt_coordinator
        ),
        inspect.getsource(multi_runner.launch_detached),
    )
    for source in terminate_sources:
        assert terminate in "".join(source.split())
    for source in detached_sources:
        assert detached in "".join(source.split())
    generic_source = "".join(
        inspect.getsource(supervisor_runtime_module.launch_process_child).split()
    )
    assert "parent_loss_policy=parent_loss_policy" in generic_source


def test_supervisor_runtime_fences_child_when_authority_delivery_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FailingHandoff:
        pass_fds: tuple[int, ...] = ()

        def deliver(self, _process: object) -> None:
            events.append("deliver")
            raise process_security_module.StateAuthorityProcessIsolationError(
                "injected delivery failure"
            )

        def close(self) -> None:
            events.append("close")

    class Child:
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            events.append("terminate")
            self.returncode = -signal.SIGTERM

        def kill(self) -> None:
            events.append("kill")
            self.returncode = -signal.SIGKILL

        def wait(self, *, timeout: float) -> int:
            assert timeout == 1.0
            events.append("wait")
            assert self.returncode is not None
            return self.returncode

    monkeypatch.setattr(
        process_security_module,
        "prepare_state_authority_child_handoff",
        lambda _environment, **_kwargs: FailingHandoff(),
    )
    monkeypatch.setattr(
        supervisor_runtime_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: Child(),
    )

    with pytest.raises(
        process_security_module.StateAuthorityProcessIsolationError,
        match="injected delivery failure",
    ):
        supervisor_runtime_module.launch_process_child(
            [sys.executable, "-c", "pass"],
            cwd=Path.cwd(),
            inherit_environment=False,
            start_new_session=False,
        )

    assert events == ["deliver", "close", "terminate", "wait"]


def test_foreground_master_pid_recovers_only_a_proven_dead_owner(
    tmp_path: Path,
) -> None:
    exited = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    dead_pid = int(exited.pid)
    assert exited.wait(timeout=5) == 0

    pid_path = tmp_path / "state" / "configured-board-master.pid"
    pid_path.parent.mkdir()
    pid_path.write_text(f"{dead_pid}\n", encoding="ascii")
    pid_path.chmod(0o600)

    multi_runner._adopt_or_create_current_master_pid_projection(  # noqa: SLF001
        pid_path
    )

    assert pid_path.read_text(encoding="ascii") == f"{os.getpid()}\n"
    current_projection = pid_path.stat()
    assert current_projection.st_nlink == 1
    assert current_projection.st_mode & 0o777 == 0o600
    quarantines = tuple(pid_path.parent.glob(f".{pid_path.name}.stale-*.quarantine"))
    decisions = tuple(pid_path.parent.glob(f".{pid_path.name}.stale-*.decision.json"))
    receipts = tuple(pid_path.parent.glob(f".{pid_path.name}.stale-*.receipt.json"))
    assert len(quarantines) == len(decisions) == len(receipts) == 1
    assert quarantines[0].read_text(encoding="ascii") == f"{dead_pid}\n"
    decision = json.loads(decisions[0].read_text(encoding="utf-8"))
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert decision["schema"] == multi_runner.STALE_DETACHED_MASTER_PID_DECISION_SCHEMA
    assert decision["decision"] == "quarantine_authorized"
    assert decision["legacy_pid"] == dead_pid
    assert decision["liveness_evidence"]["errno"] == "ESRCH"
    assert receipt["schema"] == multi_runner.STALE_DETACHED_MASTER_PID_RECEIPT_SCHEMA
    assert receipt["outcome"] == "quarantined"
    assert receipt["legacy_pid"] == dead_pid


def test_foreground_master_pid_refuses_a_live_owner(tmp_path: Path) -> None:
    live = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    pid_path = tmp_path / "state" / "configured-board-master.pid"
    pid_path.parent.mkdir()
    pid_path.write_text(f"{live.pid}\n", encoding="ascii")
    pid_path.chmod(0o600)
    try:
        with pytest.raises(
            ValueError,
            match="master PID projection names a live process",
        ):
            multi_runner._adopt_or_create_current_master_pid_projection(  # noqa: SLF001
                pid_path
            )
        assert pid_path.read_text(encoding="ascii") == f"{live.pid}\n"
        assert not tuple(pid_path.parent.glob(f".{pid_path.name}.stale-*"))
    finally:
        live.terminate()
        live.wait(timeout=5)


def test_aseh_task_authority_spec_excludes_typed_lifecycle_fields_only() -> None:
    task = {
        "task_cid": "task:aseh-authority-spec",
        "task_alias": "ASEH-001",
        "goal_cid": "goal:aseh-authority-spec",
        "objective_id": "objective:aseh-authority-spec",
        "ordinal": 1,
        "priority": "P0",
        "identity": {
            "task_cid": "task:aseh-authority-spec",
            "task_alias": "ASEH-001",
            "repository_tree_id": "tree:aseh-authority-spec",
        },
        "body": {"acceptance_conditions": "sealed acceptance"},
        "extension_schema": "",
        "extension": {},
        "dependencies": [],
        "outputs": [],
        "acceptance": [],
        "validations": [],
    }
    sealed = task_authority_spec_cid(task)
    lifecycle = {
        **task,
        "status": "in_progress",
        "revision": 4,
        "body": {
            **task["body"],
            "completion_receipt": {
                "operation": "database_claim",
                "claim_id": "claim:aseh-authority-spec",
            },
            "unknown_callback_reopen_count": 1,
        },
    }
    assert task_authority_spec_cid(lifecycle) == sealed

    forged = {**lifecycle, "body": dict(lifecycle["body"])}
    forged["body"]["acceptance_conditions"] = "reduced acceptance"
    assert task_authority_spec_cid(forged) != sealed


def _cwd_owner_socket(name: str) -> Path:
    """Bind Unix sockets through the same AF_UNIX-safe cwd alias as production."""

    return _CWD_OWNER_DIR / name


def test_aseh_parallel_quack_lanes_require_strict_deterministic_sharding() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (
            repository_root
            / "config/agent_supervisor_efficiency_state_hardening_scheduler.json"
        ).read_text(encoding="utf-8")
    )

    assert config["database_program"]["authority_mode"] == "quack"
    assert config["max_lanes"] == 4
    assert config["strict_task_sharding"] is True
    assert config["idle_lane_work_stealing"] == ""


def _materialize_one_task(path: Path) -> None:
    with DatabaseTaskSource(path) as source:
        source.materialize(
            {
                "repository_tree_id": "tree:aseh-bootstrap-test",
                "objectives": [
                    {
                        "goal_id": "ASEH-G000",
                        "goal_cid": "goal:aseh-bootstrap-test",
                        "objective_id": "objective:aseh-bootstrap-test",
                        "title": "Prove the canonical handoff",
                    }
                ],
                "taskboard": [
                    {
                        "task_id": "ASEH-000",
                        "task_cid": "task:aseh-bootstrap-test",
                        "goal_cid": "goal:aseh-bootstrap-test",
                        "status": "ready",
                    }
                ],
            }
        )


def test_aseh_offline_continuity_replay_never_mutates_authoritative_db(
    tmp_path: Path,
) -> None:
    database = tmp_path / "control.duckdb"
    _materialize_one_task(database)
    before = database.read_bytes()
    before_stat = database.stat()

    assert (
        aseh_operator._projection_matches_events_on_disposable_copy(database)
        is True
    )
    with aseh_operator._read_only_database_task_source(
        database,
        owner_id="aseh-test-read-only",
        repository_tree_id="tree:aseh-bootstrap-test",
        plan_root_cid="",
    ) as source:
        assert source.intent.uses_bound_connection is True
        assert source.snapshot().task_count == 1

    after_stat = database.stat()
    assert database.read_bytes() == before
    assert after_stat.st_size == before_stat.st_size
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns


def _sealed_memfd(value: str) -> int:
    flags = int(getattr(os, "MFD_CLOEXEC", 0x0001)) | int(
        getattr(os, "MFD_ALLOW_SEALING", 0x0002)
    )
    descriptor = os.memfd_create("aseh-test-secret", flags=flags)
    os.write(descriptor, value.encode("ascii"))
    seals = (
        int(getattr(fcntl, "F_SEAL_SEAL", 0x0001))
        | int(getattr(fcntl, "F_SEAL_SHRINK", 0x0002))
        | int(getattr(fcntl, "F_SEAL_GROW", 0x0004))
        | int(getattr(fcntl, "F_SEAL_WRITE", 0x0008))
    )
    fcntl.fcntl(
        descriptor,
        int(getattr(fcntl, "F_ADD_SEALS", 1033)),
        seals,
    )
    return descriptor


def test_grant_broker_recovers_only_same_uid_stale_socket(
    tmp_path: Path,
) -> None:
    assert DATABASE_TASK_COMMANDS == frozenset(
        {
            "compare_and_set_status",
            "rearm_blocked_task",
            "record_queue_backoff",
            "record_queue_retry",
            "record_evidence",
            "record_validation_result",
        }
    )
    broker_path = tmp_path / "owner" / "grants.sock"
    broker_path.parent.mkdir()

    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(broker_path))
    stale.close()

    credential_calls: list[str] = []

    def credential(
        _kind: str,
        _client: str,
        _birth: str,
        _pid: int,
    ) -> str:
        credential_calls.append(_kind)
        return "fixed_test_credential"

    broker = TypedStateOwnerGrantBroker(
        socket_path=broker_path,
        bootstrap_secret="0" * 64,
        store_id="store:test",
        resolve_credential=credential,
    )
    broker.start()
    try:
        assert broker.alive() is True
        channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        channel.settimeout(1)
        try:
            channel.connect(str(broker_path))
            channel.sendall(
                (
                    json.dumps(
                        {
                            "schema": TYPED_STATE_OWNER_GRANT_BROKER_SCHEMA,
                            "credential_kind": "caller_selected_mutation_scope",
                            "bootstrap_secret": "0" * 64,
                            "client_id": "client:unknown-kind",
                            "process_birth_id": kernel_process_birth_id(),
                            "store_id": "store:test",
                        },
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("utf-8")
            )
            with channel.makefile("rb") as stream:
                response = json.loads(stream.readline())
        finally:
            channel.close()
        assert response["ok"] is False
        assert response["error_code"] == "grant_denied"
        assert credential_calls == []
        assert broker.alive() is True
    finally:
        broker.stop()
    assert not broker_path.exists()

    live = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    live.bind(str(broker_path))
    live.listen(1)
    blocked = TypedStateOwnerGrantBroker(
        socket_path=broker_path,
        bootstrap_secret="1" * 64,
        store_id="store:test",
        resolve_credential=credential,
    )
    try:
        with pytest.raises(QuackStateServerControlError, match="live listener"):
            blocked.start()
        assert broker_path.is_socket()
    finally:
        live.close()
        broker_path.unlink()

    broker_path.write_text("not a socket", encoding="utf-8")
    unsafe = TypedStateOwnerGrantBroker(
        socket_path=broker_path,
        bootstrap_secret="2" * 64,
        store_id="store:test",
        resolve_credential=credential,
    )
    with pytest.raises(QuackStateServerControlError, match="same-UID socket"):
        unsafe.start()
    assert broker_path.read_text(encoding="utf-8") == "not a socket"


@pytest.mark.skipif(
    not hasattr(os, "memfd_create"),
    reason="sealed Linux memfd handoff is required",
)
def test_grant_broker_reclaims_only_a_proved_stale_socket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = probe_quack_capabilities(allow_network_install=False)
    if capability.status is not QuackCapabilityStatus.COMPATIBLE:
        pytest.skip(f"reviewed preinstalled Quack unavailable: {capability.status.value}")

    database = tmp_path / "control.duckdb"
    owner_dir = tmp_path / "quack-owner"
    monkeypatch.chdir(tmp_path)
    broker_socket = _cwd_owner_socket(TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_FILENAME)
    _materialize_one_task(database)
    owner_dir.mkdir(parents=True, exist_ok=True)

    stale_listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale_listener.bind(str(broker_socket))
    stale_listener.close()
    assert broker_socket.is_socket()

    server = build_server(
        database_path=database,
        state_dir=owner_dir,
        repository_root=tmp_path,
        port=0,
        store_id=str(database),
        secret_handle="handle:aseh-stale-broker-test",
        typed_command_socket_path=_cwd_owner_socket("typed-owner.sock"),
    )
    server.start()
    try:
        handoff = dict(server.start_supervisor_grant_broker())
        assert Path(
            handoff[TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_ENV]
        ).is_socket()
        assert server.ready()["live"] is True
    finally:
        server.stop()
        reset_quack_transport_cache()
    assert not broker_socket.exists()

    live_listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    live_listener.bind(str(broker_socket))
    live_listener.listen(1)
    second = build_server(
        database_path=database,
        state_dir=owner_dir,
        repository_root=tmp_path,
        port=0,
        store_id=str(database),
        secret_handle="handle:aseh-live-broker-test",
        typed_command_socket_path=_cwd_owner_socket("typed-owner.sock"),
    )
    second.start()
    try:
        with pytest.raises(
            QuackStateServerControlError,
            match="already serves a live listener",
        ):
            second.start_supervisor_grant_broker()
        assert broker_socket.is_socket()
    finally:
        second.stop()
        live_listener.close()
        broker_socket.unlink(missing_ok=True)
        reset_quack_transport_cache()


@pytest.mark.skipif(
    not hasattr(os, "memfd_create"),
    reason="sealed Linux memfd handoff is required",
)
def test_real_configured_supervisor_handoff_reads_and_mutates_via_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    preload_native = (
        os.environ.get("IPFS_ACCELERATE_AGENT_TEST_PRELOAD_QUACK_NATIVE")
        == "1"
    )
    if preload_native:
        assert (
            os.environ.get(
                "IPFS_ACCELERATE_AGENT_REQUIRE_LIVE_NATIVE_DEPENDENCY_VALIDATION"
            )
            == "1"
        )
        source_text = str(
            os.environ.get(
                "IPFS_ACCELERATE_AGENT_LIVE_NATIVE_DEPENDENCY_SOURCE"
            )
            or ""
        )
        assert source_text
        source = Path(source_text)
        assert (
            source.is_absolute()
            and not source.is_symlink()
            and source.resolve(strict=True) == source
            and source.name == "_duckdb.cpython-312-aarch64-linux-gnu.so"
            and os.lstat(source).st_nlink == 1
        )
        pin = llm_router.inspect_agent_supervisor_native_dependency_source(
            source,
            distribution_version="1.5.5",
            engine_version="v1.5.5",
        )
        assert pin.as_dict() == aseh_operator.ASEH_R11_NATIVE_DEPENDENCY_PIN
        native_launch = llm_router.seal_agent_supervisor_native_dependency(
            source,
            expected_pin=pin,
            accepted_authorization_id=aseh_operator._identity(
                b"aseh-r11-real-configured-supervisor-handoff-fixture"
            ),
        )

        def close_native_descriptor() -> None:
            try:
                os.close(native_launch.descriptor.descriptor)
            except OSError:
                pass

        request.addfinalizer(close_native_descriptor)
        qualification_home = aseh_operator._build_aseh_qualification_home(
            {"runtime": tmp_path / "runtime"}
        )
        assert (
            aseh_operator._validate_aseh_qualification_home(
                qualification_home
            )
            == qualification_home
        )
        for name, value in (
            aseh_operator._sealed_owner_delegation_environment(
                qualification_home
            ).items()
        ):
            monkeypatch.setenv(name, value)
        native_module = llm_router.preload_agent_supervisor_native_dependency(
            native_launch
        )
        assert native_module is sys.modules["_duckdb"]
        assert native_module is sys.modules["duckdb"]

    capability = probe_quack_capabilities(allow_network_install=False)
    if capability.status is not QuackCapabilityStatus.COMPATIBLE:
        if preload_native:
            pytest.fail(
                "required reviewed DuckDB/Quack capability is not compatible: "
                f"{capability.status.value}"
            )
        pytest.skip(f"reviewed preinstalled Quack unavailable: {capability.status.value}")

    database = tmp_path / "control.duckdb"
    owner_dir = tmp_path / "quack-owner"
    monkeypatch.chdir(tmp_path)
    owner_socket = Path("/proc/self/cwd/quack-owner/custom-typed-owner.sock")
    _materialize_one_task(database)
    server = build_server(
        database_path=database,
        state_dir=owner_dir,
        repository_root=tmp_path,
        host="127.0.0.1",
        port=0,
        repository_id="repository:aseh-bootstrap-test",
        store_id=str(database),
        secret_handle="handle:aseh-bootstrap-test",
        typed_command_socket_path=owner_socket,
    )
    identity = server.start()
    try:
        handoff = dict(server.start_supervisor_grant_broker())
        monkeypatch.setenv(
            TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_ENV,
            handoff[TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_ENV],
        )
        monkeypatch.setenv(
            TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV,
            handoff[TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV],
        )
        monkeypatch.setenv(TYPED_STATE_OWNER_SOCKET_ENV, str(owner_socket))
        monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", str(database))
        monkeypatch.setenv(
            "IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION",
            str(identity.generation),
        )
        monkeypatch.setenv(
            "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR",
            str(owner_dir / "mutations"),
        )
        # A stale legacy value must never shadow the live broker exchange.
        monkeypatch.setenv(
            "IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "stale_legacy_token"
        )

        assert not (owner_dir / "handle_aseh-bootstrap-test.quack-token").exists()
        assert server.status()["configured_supervisor_credential_broker"] == {
            "available": True,
            "server_owned": True,
            "socket_path": "/proc/self/cwd/quack-owner/typed-state-owner-grants.sock",
            "credential_published": False,
            "task_mutation_path": "typed_state_owner_database_task_command",
            "last_error_type": "",
        }
        assert handoff[TYPED_STATE_OWNER_SOCKET_ENV] == str(owner_socket)
        assert server.status_path().is_file()
        assert not any("ControlPlaneBoundsError" in item for item in server.logs())

        def forbidden_legacy_inbox(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("filesystem mutation servicing was invoked")

        for service_name in (
            "process_mutation_inbox",
            "service_mutation_inbox",
            "service_database_task_command_inbox",
        ):
            monkeypatch.setattr(server, service_name, forbidden_legacy_inbox)
        refresh_before = int(
            server.status()["read_replica"]["refresh_sequence"]
        )
        with DatabaseTaskSource(
            identity.listen_uri,
            owner_id="aseh-test-supervisor",
            install_schema=False,
        ) as source:
            snapshot = source.snapshot()
            assert snapshot.task_count == 1
            ready = source.ready_tasks(limit=10).tasks
            assert [item.task_alias for item in ready] == ["ASEH-000"]
            sealed_projection = source.plan_projection(
                task_cids=[ready[0].task_cid]
            )
            sealed_authority = aseh_operator._task_authority_spec_cids(  # noqa: SLF001
                sealed_projection
            )
            sealed_task = sealed_projection["tasks"][0]
            sealed_projection_spec_cid = str(sealed_task["spec_cid"])
            changed = source.compare_and_set_status(
                ready[0],
                ready[0].revision,
                "in_progress",
                receipt={
                    "operation": "database_claim",
                    "claim_id": "claim:aseh-bootstrap-test",
                    "attempt_id": "attempt:aseh-bootstrap-test",
                    "owner_session_id": "owner:aseh-bootstrap-test",
                    "lease_id": "lease:aseh-bootstrap-test",
                    "fencing_token": 1,
                    "fence_epoch": 1,
                    "claimed_from_revision": 1,
                },
            )
            assert changed.changed is True
            assert changed.task.status == "in_progress"
            requeued = source.compare_and_set_status(
                changed.task,
                changed.revision,
                "todo",
                receipt={
                    "operation": "requeue_unimplemented_stale_attempt",
                    "attempt_id": "attempt:aseh-bootstrap-test",
                    "unknown_callback_reopen_count": 1,
                },
            )
            assert requeued.changed is True
            assert requeued.task.status == "todo"
            assert requeued.task.revision == 3
            reclaimed = source.compare_and_set_status(
                requeued.task,
                requeued.revision,
                "in_progress",
                receipt={
                    "operation": "database_claim",
                    "claim_id": "claim:aseh-bootstrap-test:retry-1",
                    "attempt_id": "attempt:aseh-bootstrap-test:retry-1",
                    "owner_session_id": "owner:aseh-bootstrap-test",
                    "lease_id": "lease:aseh-bootstrap-test:retry-1",
                    "fencing_token": 2,
                    "fence_epoch": 1,
                    "claimed_from_revision": 3,
                },
            )
            assert reclaimed.changed is True
            assert reclaimed.task.status == "in_progress"
            assert reclaimed.task.revision == 4
            # A successful owner acknowledgement is also a synchronous
            # Quack-publication barrier.  Strict sharding re-reads this exact
            # binding immediately after reclaim and must not see a stale
            # ``todo`` projection.
            observed = source.get("ASEH-000")
            assert observed is not None
            assert observed.status == "in_progress"
            assert observed.revision == reclaimed.revision
            assert observed.body["unknown_callback_reopen_count"] == 1
            assert observed.body["completion_receipt"] == {
                "operation": "database_claim",
                "claim_id": "claim:aseh-bootstrap-test:retry-1",
                "attempt_id": "attempt:aseh-bootstrap-test:retry-1",
                "owner_session_id": "owner:aseh-bootstrap-test",
                "lease_id": "lease:aseh-bootstrap-test:retry-1",
                "fencing_token": 2,
                "fence_epoch": 1,
                "claimed_from_revision": 3,
                "unknown_callback_reopen_count": 1,
            }
            lifecycle_projection = source.plan_projection(
                task_cids=[observed.task_cid]
            )
            lifecycle_task = lifecycle_projection["tasks"][0]
            assert lifecycle_task["task_cid"] == sealed_task["task_cid"]
            assert lifecycle_task["spec_cid"] != sealed_projection_spec_cid
            assert (
                aseh_operator._task_authority_spec_cids(  # noqa: SLF001
                    lifecycle_projection
                )
                == sealed_authority
            )

            forged_task = {
                **lifecycle_task,
                "body": dict(lifecycle_task["body"]),
            }
            forged_task["body"]["acceptance_conditions"] = "reduced acceptance"
            assert (
                aseh_operator._task_authority_spec_cids(  # noqa: SLF001
                    {"tasks": [forged_task]}
                )
                != sealed_authority
            )
            # The canonical repository classifies a stale lower revision as
            # a bounds failure; the typed gateway must preserve that code.
            with pytest.raises(TaskSourceBoundsError):
                source.compare_and_set_status(ready[0], 0, "failed")
        live_status = server.status()
        assert live_status["read_replica"]["live"] is True
        assert int(live_status["read_replica"]["refresh_sequence"]) > refresh_before
        assert int(live_status["read_replica"]["refresh_sequence"]) >= 2
        mutation_dir = owner_dir / "mutations"
        assert not mutation_dir.exists() or not tuple(mutation_dir.glob("*.json"))
        assert server._connection is not None  # noqa: SLF001
        idempotency = server._connection.execute(  # noqa: SLF001
            "SELECT COUNT(*) FROM idempotency_records "
            "WHERE command_kind = 'compare_and_set_status'"
        ).fetchone()
        assert idempotency is not None and int(idempotency[0]) == 3

        descriptor = int(handoff[TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV])
        for safe_environment in (
            provider_subprocess_environment(os.environ),
            build_validation_environment(os.environ),
        ):
            assert TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_ENV not in safe_environment
            assert (
                TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV
                not in safe_environment
            )
            child = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import os,sys; fd=int(sys.argv[1]); "
                        "\ntry: os.fstat(fd)"
                        "\nexcept OSError: raise SystemExit(0)"
                        "\nraise SystemExit(1)"
                    ),
                    str(descriptor),
                ],
                env=safe_environment,
                check=False,
            )
            assert child.returncode == 0

        raw_transport_token = request_quack_attach_credential(
            store_id=str(database),
            client_id="aseh-test-persistence-audit",
            process_birth_id=kernel_process_birth_id(),
            timeout_seconds=2,
        )
        with pytest.raises(TypedStateOwnerError):
            TypedStateOwnerConnection(
                socket_path=owner_socket,
                token=raw_transport_token,
                client_id="raw-read-token-cannot-mutate",
                process_birth_id=kernel_process_birth_id(),
                store_id=str(database),
                timeout_seconds=2,
            )

        command_client_id = f"grant-inspection:{os.getpid()}"
        process_birth_id = kernel_process_birth_id()
        command_grant = request_database_task_command_credential(
            store_id=str(database),
            client_id=command_client_id,
            process_birth_id=process_birth_id,
            timeout_seconds=2,
        )
        copied = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import json,sys; from pathlib import Path; "
                    "from ipfs_accelerate_py.agent_supervisor.task_sources."
                    "typed_state_owner import TypedStateOwnerConnection; "
                    "p=json.loads(sys.stdin.read()); "
                    "\ntry: c=TypedStateOwnerConnection(socket_path=Path(p['socket']), "
                    "token=p['token'], client_id=p['client'], "
                    "process_birth_id=p['birth'], store_id=p['store'], "
                    "timeout_seconds=2)"
                    "\nexcept BaseException: raise SystemExit(0)"
                    "\nc.close(); raise SystemExit(7)"
                ),
            ],
            input=json.dumps(
                {
                    "socket": str(owner_socket.resolve()),
                    "token": command_grant,
                    "client": command_client_id,
                    "birth": process_birth_id,
                    "store": str(database),
                }
            ),
            text=True,
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            check=False,
        )
        assert copied.returncode == 0
        command_connection = TypedStateOwnerConnection(
            socket_path=owner_socket,
            token=command_grant,
            client_id=command_client_id,
            process_birth_id=process_birth_id,
            store_id=str(database),
            timeout_seconds=2,
        )
        try:
            with pytest.raises(TypedStateOwnerError):
                TypedStateOwnerConnection(
                    socket_path=owner_socket,
                    token=command_grant,
                    client_id=command_client_id,
                    process_birth_id=process_birth_id,
                    store_id=str(database),
                    timeout_seconds=2,
                )
            assert set(
                command_connection.grant["allowed_database_task_commands"]
            ) == set(DATABASE_TASK_COMMANDS)
            assert command_connection.grant["allowed_operations"] == []
            assert command_connection.grant["allowed_command_operations"] == []
            assert (
                int(command_connection.grant["expires_at"])
                - int(command_connection.grant["issued_at"])
                <= 60_000
            )
        finally:
            command_connection.close()
        deadline = time.monotonic() + 2
        while (
            server.status()["typed_command_gateway"]["active_grants"]
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        with pytest.raises(TypedStateOwnerError):
            TypedStateOwnerConnection(
                socket_path=owner_socket,
                token=command_grant,
                client_id=command_client_id,
                process_birth_id=process_birth_id,
                store_id=str(database),
                timeout_seconds=2,
            )

        expiring_client_id = f"expiring-task-grant:{os.getpid()}"
        expiring_grant = server.issue_typed_client_grant(
            client_id=expiring_client_id,
            process_birth_id=process_birth_id,
            allowed_database_task_commands=("record_queue_retry",),
            peer_pid=os.getpid(),
            ttl_seconds=1,
        )
        server.issue_typed_client_grant(
            client_id=f"orphaned-task-grant:{os.getpid()}",
            process_birth_id=process_birth_id,
            allowed_database_task_commands=("record_queue_retry",),
            peer_pid=os.getpid(),
            ttl_seconds=1,
        )
        expiring_connection = TypedStateOwnerConnection(
            socket_path=owner_socket,
            token=expiring_grant,
            client_id=expiring_client_id,
            process_birth_id=process_birth_id,
            store_id=str(database),
            timeout_seconds=2,
        )
        try:
            time.sleep(1.05)
            with pytest.raises(TypedStateOwnerRemoteError) as expired:
                expiring_connection.execute_database_task_command(
                    "record_queue_retry",
                    {"task_cid": "task:aseh-bootstrap-test"},
                    command_request_id="f" * 32,
                )
            assert expired.value.error_code == "authorization_denied"
        finally:
            expiring_connection.close()
        assert server.status()["typed_command_gateway"]["active_grants"] == 0

        # Drop the first post-commit success response at the typed gateway.
        # The client must surface an unknown outcome for reconciliation and
        # must not manufacture a second command/request behind the caller's
        # back.
        assert server._connection is not None  # noqa: SLF001
        before_drop = server._connection.execute(  # noqa: SLF001
            "SELECT COUNT(*) FROM idempotency_records "
            "WHERE command_kind = 'record_queue_backoff'"
        ).fetchone()
        assert before_drop is not None
        real_send_frame = typed_state_owner_module._send_frame  # noqa: SLF001
        dropped = threading.Event()

        def drop_post_commit_response(
            channel: socket.socket,
            payload: dict[str, object],
        ) -> None:
            if (
                not dropped.is_set()
                and payload.get("ok") is True
                and "result" in payload
            ):
                dropped.set()
                try:
                    channel.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                channel.close()
                raise OSError("injected post-commit response loss")
            real_send_frame(channel, payload)

        monkeypatch.setattr(
            typed_state_owner_module,
            "_send_frame",
            drop_post_commit_response,
        )
        try:
            with DatabaseTaskSource(
                identity.listen_uri,
                owner_id="aseh-test-unknown-outcome",
                install_schema=False,
            ) as source:
                with pytest.raises(TaskSourceUnknownOutcomeError):
                    source.record_queue_backoff(
                        task_cid="task:aseh-bootstrap-test",
                        delay_ms=2_000,
                        reason="dropped-response-test",
                    )
        finally:
            monkeypatch.setattr(
                typed_state_owner_module,
                "_send_frame",
                real_send_frame,
            )
        assert dropped.is_set()
        after_drop = server._connection.execute(  # noqa: SLF001
            "SELECT COUNT(*) FROM idempotency_records "
            "WHERE command_kind = 'record_queue_backoff'"
        ).fetchone()
        assert after_drop is not None
        assert int(after_drop[0]) == int(before_drop[0]) + 1
        with DatabaseTaskSource(
            identity.listen_uri,
            owner_id="aseh-test-reconcile-observation",
            install_schema=False,
        ) as source:
            entry = source.get_queue_entry("task:aseh-bootstrap-test")
            assert entry is not None
            assert entry.reason == "dropped-response-test"

        raw_bootstrap_secret = os.pread(descriptor, 257, 0)
        for path in owner_dir.rglob("*"):
            if not path.is_file():
                continue
            body = path.read_bytes()
            assert raw_transport_token.encode("ascii") not in body
            assert raw_bootstrap_secret not in body

        # Status publication is part of the acknowledgement barrier too. A
        # failure after the command commit must quarantine the owner and
        # surface an unknown outcome, never an ordinary retryable error.
        assert server._connection is not None  # noqa: SLF001
        before_publication_failure = server._connection.execute(  # noqa: SLF001
            "SELECT COUNT(*) FROM idempotency_records "
            "WHERE command_kind = 'record_queue_retry'"
        ).fetchone()
        assert before_publication_failure is not None

        def fail_status_publication() -> None:
            raise OSError("injected status publication failure")

        real_write_status = server._write_status  # noqa: SLF001
        monkeypatch.setattr(server, "_write_status", fail_status_publication)
        try:
            with DatabaseTaskSource(
                identity.listen_uri,
                owner_id="aseh-test-publication-failure",
                install_schema=False,
            ) as source:
                with pytest.raises(TaskSourceUnknownOutcomeError):
                    source.record_queue_retry(
                        task_cid="task:aseh-bootstrap-test"
                    )
        finally:
            monkeypatch.setattr(server, "_write_status", real_write_status)
        after_publication_failure = server._connection.execute(  # noqa: SLF001
            "SELECT COUNT(*) FROM idempotency_records "
            "WHERE command_kind = 'record_queue_retry'"
        ).fetchone()
        assert after_publication_failure is not None
        assert int(after_publication_failure[0]) == (
            int(before_publication_failure[0]) + 1
        )
        assert server.status()["lifecycle"] == "failed"
        assert server.status()["read_replica"]["live"] is False
    finally:
        # This process is both the real owner and its configured-supervisor
        # client.  Release the client-side Quack pool before asking the owner
        # to prove that its endpoint closed; production processes reach the
        # same ordering when supervisor children terminate before the owner.
        reset_quack_transport_cache()
        server.stop()
        assert not any("transport stop warning" in item for item in server.logs())


@pytest.mark.skipif(
    not hasattr(os, "memfd_create"),
    reason="sealed Linux memfd handoff is required",
)
def test_live_broker_status_replays_only_an_exact_disposable_replica(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = probe_quack_capabilities(allow_network_install=False)
    if capability.status is not QuackCapabilityStatus.COMPATIBLE:
        pytest.skip(f"reviewed preinstalled Quack unavailable: {capability.status.value}")

    database = tmp_path / "control.duckdb"
    owner_dir = tmp_path / "owner"
    tree_id = "tree:aseh-bootstrap-test"
    monkeypatch.chdir(tmp_path)
    _materialize_one_task(database)
    with DatabaseTaskSource(
        database,
        install_schema=False,
        repository_tree_id=tree_id,
    ) as source:
        sealed_snapshot = source.snapshot().to_dict()
        goal = source.get_goal("goal:aseh-bootstrap-test")
        assert goal is not None
    bootstrap = {
        "schema": aseh_operator.BOOTSTRAP_SCHEMA,
        "source_head": "commit:aseh-bootstrap-test",
        "repository_tree_id": tree_id,
        "plan_root_cid": sealed_snapshot["plan_root_cid"],
        "source_forest": {},
        "source_identities": {},
        "database_task_source_receipt": {},
        "snapshot": sealed_snapshot,
        "integrity": {
            "goal_records": {
                "ASEH-G000": aseh_operator._immutable_goal_record(goal)
            }
        },
        "initial_ready_task_ids": ["ASEH-000"],
        "bootstrap_validation": {},
        "recovered_after_interrupted_materialization": False,
        "authority": {},
        "ducklake_projection": {},
    }
    bootstrap["bootstrap_receipt_id"] = aseh_operator._identity(bootstrap)
    bootstrap_path = tmp_path / "bootstrap.json"
    aseh_operator._atomic_json(bootstrap_path, bootstrap)
    paths = {
        "runtime": tmp_path,
        "database": database,
        "owner": owner_dir,
        "bootstrap_receipt": bootstrap_path,
    }
    server = build_server(
        database_path=database,
        state_dir=owner_dir,
        repository_root=tmp_path,
        port=0,
        store_id=str(database),
        secret_handle="handle:aseh-live-status-test",
    )
    identity = server.start()
    try:
        handoff = dict(server.start_supervisor_grant_broker())
        monkeypatch.setenv(
            TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_ENV,
            handoff[TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_ENV],
        )
        monkeypatch.setenv(
            TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV,
            handoff[TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV],
        )
        monkeypatch.setenv(
            TYPED_STATE_OWNER_SOCKET_ENV,
            handoff[TYPED_STATE_OWNER_SOCKET_ENV],
        )
        monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", str(database))
        monkeypatch.setenv(
            "IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION",
            str(identity.generation),
        )
        monkeypatch.delenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", raising=False)
        board = SimpleNamespace(
            resolved_database_program=lambda: SimpleNamespace(
                quack_endpoint=identity.listen_uri
            )
        )
        real_replay = DatabaseTaskSource.projection_matches_events
        replay_transports: list[bool] = []

        def reject_live_rebuild(source: DatabaseTaskSource) -> bool:
            replay_transports.append(source.intent.uses_quack_transport)
            if source.intent.uses_quack_transport:
                raise AssertionError("live Quack projection rebuild was invoked")
            return real_replay(source)

        monkeypatch.setattr(
            DatabaseTaskSource,
            "projection_matches_events",
            reject_live_rebuild,
        )
        with aseh_operator._LIVE_REPLAY_CACHE_LOCK:
            aseh_operator._LIVE_REPLAY_CACHE.clear()
        event_cursor_before = int(
            server._connection.execute(  # noqa: SLF001
                "SELECT COALESCE(MAX(global_sequence), 0) FROM domain_events"
            ).fetchone()[0]
        )
        owner_status = server.status()
        real_retire = aseh_operator._retire_live_replay_directory

        def fail_retirement(directory: Path) -> None:
            (directory / "unexpected-artifact").write_text(
                "not admitted\n", encoding="utf-8"
            )
            real_retire(directory)

        monkeypatch.setattr(
            aseh_operator,
            "_retire_live_replay_directory",
            fail_retirement,
        )
        for _attempt in range(2):
            with pytest.raises(
                aseh_operator.OperatorError,
                match="unexpected artifacts: unexpected-artifact",
            ):
                aseh_operator._broker_status_query(
                    board, paths, owner_status=owner_status
                )
            with aseh_operator._LIVE_REPLAY_CACHE_LOCK:
                assert aseh_operator._LIVE_REPLAY_CACHE == {}
        assert replay_transports == [False, False]
        replay_directories = tuple(tmp_path.glob(".live-projection-replay*"))
        assert len(replay_directories) == 2
        monkeypatch.setattr(
            aseh_operator,
            "_retire_live_replay_directory",
            real_retire,
        )
        for replay_directory in replay_directories:
            (replay_directory / "unexpected-artifact").unlink()
            real_retire(replay_directory)

        report = aseh_operator._broker_status_query(
            board, paths, owner_status=owner_status
        )
        assert report["available"] is True
        owner_binding_fields = (
            "server_id", "store_id", "database_uuid", "schema_revision",
            "schema_fingerprint", "generation", "process_birth_id",
            "listen_uri", "extension_fingerprint",
        )
        assert owner_status["storage_schema_fingerprint"] != (
            owner_status["identity"]["schema_fingerprint"]
        )
        assert report["owner_binding"] == {
            field: owner_status["identity"][field]
            for field in owner_binding_fields
        }
        assert report["ready_task_ids"] == ["ASEH-000"]
        assert report["event_cursor"] == event_cursor_before
        assert report["projection_matches_events"] is True
        witness = report["projection_reconciliation"]
        assert witness["authoritative"] is False
        assert witness["mutation_authority"] is False
        assert replay_transports == [False, False, False]
        assert not tuple(tmp_path.glob(".live-projection-replay*"))
        assert int(
            server._connection.execute(  # noqa: SLF001
                "SELECT COALESCE(MAX(global_sequence), 0) FROM domain_events"
            ).fetchone()[0]
        ) == event_cursor_before

        mismatched_identity = json.loads(json.dumps(owner_status))
        mismatched_identity["identity"]["generation"] += 1
        with pytest.raises(
            aseh_operator.OperatorError,
            match="owner binding differs from published owner",
        ):
            aseh_operator._broker_status_query(
                board, paths, owner_status=mismatched_identity
            )
        mismatched_storage = json.loads(json.dumps(owner_status))
        mismatched_storage["storage_schema_fingerprint"] = "storage:stale"
        with pytest.raises(
            aseh_operator.OperatorError,
            match="storage schema differs from published owner",
        ):
            aseh_operator._broker_status_query(
                board, paths, owner_status=mismatched_storage
            )

        # An exact cache hit still reopens and re-hashes the owner-published
        # bytes.  Same-sized path tampering cannot reuse the prior witness.
        replica_path = Path(owner_status["read_replica"]["path"])
        descriptor = os.open(replica_path, os.O_RDWR | os.O_CLOEXEC)
        try:
            original = os.pread(descriptor, 1, 0)
            assert original
            changed = bytes([original[0] ^ 0xFF])
            assert os.pwrite(descriptor, changed, 0) == 1
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        with pytest.raises(
            aseh_operator.OperatorError,
            match="published replica bytes differ",
        ):
            aseh_operator._admit_live_projection_shadow_replay(
                paths=paths,
                owner_status=owner_status,
                expected_snapshot=report["snapshot"],
            )
        assert replay_transports == [False, False, False]

        tampered_status = json.loads(json.dumps(server.status()))
        tampered_status["read_replica"]["sha256"] = f"sha256:{'0' * 64}"
        with pytest.raises(aseh_operator.OperatorError):
            aseh_operator._admit_live_projection_shadow_replay(
                paths=paths,
                owner_status=tampered_status,
                expected_snapshot=report["snapshot"],
            )
        assert int(
            server._connection.execute(  # noqa: SLF001
                "SELECT COALESCE(MAX(global_sequence), 0) FROM domain_events"
            ).fetchone()[0]
        ) == event_cursor_before
    finally:
        with aseh_operator._LIVE_REPLAY_CACHE_LOCK:
            aseh_operator._LIVE_REPLAY_CACHE.clear()
        reset_quack_transport_cache()
        server.stop()


@pytest.mark.skipif(
    not hasattr(os, "memfd_create"),
    reason="sealed Linux memfd handoff is required",
)
def test_typed_owner_never_acknowledges_an_unpublished_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = probe_quack_capabilities(allow_network_install=False)
    if capability.status is not QuackCapabilityStatus.COMPATIBLE:
        pytest.skip(
            f"reviewed preinstalled Quack unavailable: {capability.status.value}"
        )

    database = tmp_path / "control.duckdb"
    owner_dir = tmp_path / "quack-owner"
    monkeypatch.chdir(tmp_path)
    owner_socket = _cwd_owner_socket("typed-owner.sock")
    _materialize_one_task(database)
    server = build_server(
        database_path=database,
        state_dir=owner_dir,
        repository_root=tmp_path,
        port=0,
        store_id=str(database),
        secret_handle="handle:aseh-publication-failure-test",
        typed_command_socket_path=owner_socket,
    )
    identity = server.start()
    try:
        handoff = dict(server.start_supervisor_grant_broker())
        monkeypatch.setenv(
            TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_ENV,
            handoff[TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_ENV],
        )
        monkeypatch.setenv(
            TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV,
            handoff[TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV],
        )
        monkeypatch.setenv(TYPED_STATE_OWNER_SOCKET_ENV, str(owner_socket))
        monkeypatch.setenv(
            "IPFS_ACCELERATE_AGENT_STATE_STORE_ID", str(database)
        )
        monkeypatch.setenv(
            "IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION",
            str(identity.generation),
        )
        monkeypatch.setenv(
            "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR",
            str(owner_dir / "mutations"),
        )

        def fail_publication(*_args: object, **_kwargs: object) -> dict[str, object]:
            raise RuntimeError("injected read-replica publication failure")

        monkeypatch.setattr(server, "_refresh_read_replica", fail_publication)

        with pytest.raises(QuackOwnerCommandRemoteError) as unknown:
            submit_quack_owner_command(
                QUACK_OWNER_COMMAND_COMPARE_AND_SET_STATUS,
                {
                    "task_cid_or_alias": "ASEH-000",
                    "expected_revision": 1,
                    "status": "in_progress",
                    "receipt": {"operation": "database_claim"},
                    "evidence_digests": None,
                },
                timeout_seconds=0.5,
            )
        assert unknown.value.code == "read_replica_refresh_unknown_outcome"
        mutation_dir = owner_dir / "mutations"
        assert not mutation_dir.exists() or not tuple(
            mutation_dir.glob("*.json")
        )
        assert server._connection is not None  # noqa: SLF001
        row = server._connection.execute(  # noqa: SLF001
            "SELECT status, revision FROM tasks WHERE task_alias = 'ASEH-000'"
        ).fetchone()
        assert row is not None
        assert (str(row[0]), int(row[1])) == ("in_progress", 2)
        idempotency = server._connection.execute(  # noqa: SLF001
            "SELECT COUNT(*) FROM idempotency_records "
            "WHERE command_kind = 'compare_and_set_status'"
        ).fetchone()
        assert idempotency is not None and int(idempotency[0]) == 1
        assert server.status()["lifecycle"] == "failed"
        assert server.status()["read_replica"]["live"] is False
    finally:
        reset_quack_transport_cache()
        server.stop()


@pytest.mark.skipif(
    not hasattr(os, "memfd_create"),
    reason="sealed Linux memfd handoff is required",
)
def test_broker_denial_and_slow_peer_do_not_break_later_delivery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = probe_quack_capabilities(allow_network_install=False)
    if capability.status is not QuackCapabilityStatus.COMPATIBLE:
        pytest.skip(f"reviewed preinstalled Quack unavailable: {capability.status.value}")

    database = tmp_path / "control.duckdb"
    owner_dir = tmp_path / "quack-owner"
    monkeypatch.chdir(tmp_path)
    _materialize_one_task(database)
    server = build_server(
        database_path=database,
        state_dir=owner_dir,
        repository_root=tmp_path,
        port=0,
        store_id=str(database),
        secret_handle="handle:aseh-broker-denial-test",
        typed_command_socket_path=_cwd_owner_socket("typed-owner.sock"),
    )
    server.start()
    wrong_fd = -1
    slow: socket.socket | None = None
    try:
        handoff = dict(server.start_supervisor_grant_broker())
        monkeypatch.setenv(
            TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_ENV,
            handoff[TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_ENV],
        )
        monkeypatch.setenv(
            TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV,
            handoff[TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV],
        )
        monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", str(database))

        wrong_fd = _sealed_memfd("0" * 64)
        monkeypatch.setenv(
            TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV, str(wrong_fd)
        )
        with pytest.raises(TypedStateOwnerAuthorizationError, match="denied"):
            request_quack_attach_credential(
                store_id=str(database),
                client_id="aseh-test-wrong-secret",
                process_birth_id=kernel_process_birth_id(),
                timeout_seconds=2,
            )
        assert server.status()["configured_supervisor_credential_broker"][
            "available"
        ] is True

        monkeypatch.setenv(
            TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV,
            handoff[TYPED_STATE_OWNER_GRANT_BROKER_SECRET_FD_ENV],
        )
        with pytest.raises(TypedStateOwnerAuthorizationError, match="denied"):
            request_quack_attach_credential(
                store_id=str(database),
                client_id="caller-selected-label-is-diagnostic-only",
                process_birth_id="birth:caller-selected-not-kernel-derived",
                timeout_seconds=2,
            )
        assert server.status()["configured_supervisor_credential_broker"][
            "available"
        ] is True

        slow = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        slow.connect(handoff[TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_ENV])
        started = time.monotonic()
        token = request_quack_attach_credential(
            store_id=str(database),
            client_id="aseh-test-after-slow-peer",
            process_birth_id=kernel_process_birth_id(),
            timeout_seconds=3,
        )
        assert token
        assert time.monotonic() - started < 2
        assert server.status()["configured_supervisor_credential_broker"][
            "available"
        ] is True

        def fail_resolver(
            _kind: str,
            _client: str,
            _birth: str,
            _pid: int,
        ) -> str:
            raise RuntimeError("injected broker authority failure")

        assert server._grant_broker is not None  # noqa: SLF001
        monkeypatch.setattr(
            server._grant_broker,  # noqa: SLF001
            "_resolve_credential",
            fail_resolver,
        )
        with pytest.raises(TypedStateOwnerAuthorizationError, match="denied"):
            request_quack_attach_credential(
                store_id=str(database),
                client_id="aseh-test-owner-failure",
                process_birth_id=kernel_process_birth_id(),
                timeout_seconds=2,
            )
        broker_status = server.status()[
            "configured_supervisor_credential_broker"
        ]
        assert broker_status["available"] is False
        assert broker_status["last_error_type"] == "RuntimeError"
        with pytest.raises(QuackStateServerReadyError, match="credential broker"):
            server.ready()
    finally:
        if slow is not None:
            slow.close()
        if wrong_fd >= 0:
            os.close(wrong_fd)
        server.stop()
        reset_quack_transport_cache()


def _aseh_health_fixture(
    tmp_path: Path,
    *,
    status: str = "todo",
    revision: int = 1,
    event_cursor: int = 10,
    ready: bool = True,
    active: bool = False,
    worker_count: int = 0,
    observed_at: float,
    lane_mtime_ns: int,
    lane_stalled: bool = False,
    delayed_retry_not_before_ms: int = 0,
    parallel_blocked: bool = False,
) -> tuple[SimpleNamespace, dict[str, Path], dict[str, object]]:
    task_cid = "task:aseh-health"
    blocked_task_cid = "task:aseh-health-blocked"
    task_count = 2 if parallel_blocked else 1
    goal_record = {
        "goal_cid": "goal:aseh-health",
        "goal_alias": "ASEH-G000",
        "objective_id": "objective:aseh-root",
        "parent_goal_cid": "",
        "ordinal": 1,
        "title": "ASEH health",
        "body": {"priority": "P0"},
    }
    plan_record = {
        "plan_cid": "plan:aseh-health",
        "goal_cid": "goal:aseh-health",
        "plan_alias": "ASEH-PLAN-R1",
        "body": {"plan_cid": "plan:aseh-health"},
    }
    objective_record = {
        "objective_id": "objective:aseh-root",
        "objective_alias": "ASEH-G000",
        "parent_objective_id": "",
        "title": "ASEH root objective",
        "priority": "P0",
        "body": {"program_id": aseh_operator.PROGRAM},
        "extension_schema": "",
        "extension": {},
    }
    bootstrap_snapshot = {
        "source_schema": "source@1",
        "schema_version": "1",
        "plan_root_cid": "plan:aseh-health",
        "repository_tree_id": "tree:aseh-health",
        "formal_plan_id": "formal:aseh-health",
        "source_identity": "",
        "projection_cid": "projection:aseh-health",
        "event_cursor": 10,
        "task_count": task_count,
        "goal_count": 1,
        "dependency_count": 0,
        "objective_count": 1,
        "plan_count": 1,
    }
    bootstrap_snapshot["source_identity"] = aseh_operator.content_identity(
        {
            "plan_root_cid": bootstrap_snapshot["plan_root_cid"],
            "repository_tree_id": bootstrap_snapshot["repository_tree_id"],
            "projection_cid": bootstrap_snapshot["projection_cid"],
        }
    )
    sealed_task_statuses = {"ASEH-000": "todo"}
    sealed_task_revisions = {"ASEH-000": 1}
    sealed_task_cids = {"ASEH-000": task_cid}
    sealed_owner_bindings = {
        "ASEH-000": {
            "owning_repository": "ipfs_accelerate_py",
            "base_revision": "commit:aseh-health",
            "base_repository_tree_id": "tree:aseh-health",
            "source_forest_cid": "forest:aseh-health",
            "owner_source_identity": "source-owner:aseh-health",
        }
    }
    sealed_task_dependencies = {"ASEH-000": []}
    sealed_task_authority_spec_cids = {
        "ASEH-000": "sha256:aseh-health-authority-spec"
    }
    if parallel_blocked:
        sealed_task_statuses["ASEH-001"] = "todo"
        sealed_task_revisions["ASEH-001"] = 1
        sealed_task_cids["ASEH-001"] = blocked_task_cid
        sealed_owner_bindings["ASEH-001"] = dict(
            sealed_owner_bindings["ASEH-000"]
        )
        sealed_task_dependencies["ASEH-001"] = []
        sealed_task_authority_spec_cids["ASEH-001"] = (
            "sha256:aseh-health-blocked-authority-spec"
        )
    integrity = {
        "schema": "ipfs_accelerate_py/agent-supervisor/aseh-integrity@1",
        "projection_matches_events": True,
        "projection_cid": "projection:aseh-health",
        "event_cursor": 10,
        "task_statuses": sealed_task_statuses,
        "task_revisions": sealed_task_revisions,
        "task_cids": sealed_task_cids,
        "task_owner_bindings": sealed_owner_bindings,
        "task_dependencies": sealed_task_dependencies,
        "task_authority_spec_cids": sealed_task_authority_spec_cids,
        "goal_records": {"ASEH-G000": goal_record},
        "goal_edges": [],
        "plan_record": plan_record,
        "objective_record": objective_record,
        "task_count": task_count,
        "goal_count": 1,
        "dependency_count": 0,
        "objective_count": 1,
        "plan_count": 1,
    }
    integrity["integrity_receipt_id"] = aseh_operator._identity(integrity)
    bootstrap = {
        "schema": aseh_operator.BOOTSTRAP_SCHEMA,
        "source_head": "commit:aseh-health",
        "repository_tree_id": "tree:aseh-health",
        "plan_root_cid": "plan:aseh-health",
        "source_forest": {"forest_cid": "forest:aseh-health"},
        "source_identities": {"operator": "source:operator"},
        "database_task_source_receipt": {},
        "snapshot": bootstrap_snapshot,
        "integrity": integrity,
        "initial_ready_task_ids": ["ASEH-000"],
        "bootstrap_validation": {},
        "recovered_after_interrupted_materialization": False,
        "authority": {},
        "ducklake_projection": {},
    }
    bootstrap["bootstrap_receipt_id"] = aseh_operator._identity(bootstrap)
    bootstrap_path = tmp_path / "bootstrap.json"
    aseh_operator._atomic_json(bootstrap_path, bootstrap)

    process_birth = current_process_birth().to_dict()
    binding = {
        "server_id": "server:aseh-health",
        "store_id": "store:aseh-health",
        "database_uuid": "database:aseh-health",
        "schema_revision": 3,
        "schema_fingerprint": "schema:aseh-health",
        "generation": 9,
        "process_birth_id": aseh_operator._state_owner_process_birth_id(
            process_birth
        ),
        "listen_uri": "quack:127.0.0.1:1",
        "extension_fingerprint": "extensions:aseh-health",
    }
    owner_identity = {
        **binding,
        "process_birth": process_birth,
        "status": "ready",
    }
    current_snapshot = {**bootstrap_snapshot, "event_cursor": event_cursor}
    database_path = tmp_path / "control.duckdb"
    replica_binding = {
        "path": str(tmp_path / "control.read-replica.duckdb"),
        "source_database_path": str(database_path),
        "server_id": binding["server_id"],
        "database_uuid": binding["database_uuid"],
        "generation": binding["generation"],
        "schema_revision": binding["schema_revision"],
        "schema_fingerprint": binding["schema_fingerprint"],
        "storage_schema_fingerprint": "storage-schema:aseh-health",
        "sha256": f"sha256:{'a' * 64}",
        "size_bytes": 1,
        "refresh_sequence": 1,
    }
    replay_witness = {
        "schema": aseh_operator.LIVE_REPLAY_SCHEMA,
        "method": "disposable_exact_owner_published_replica_replay",
        "authoritative": False,
        "mutation_authority": False,
        "projection_matches_events": True,
        "replica": replica_binding,
        "projection_cid": current_snapshot["projection_cid"],
        "event_cursor": current_snapshot["event_cursor"],
        "cache_key": aseh_operator._identity(
            {
                "replica": replica_binding,
                "projection_cid": current_snapshot["projection_cid"],
                "event_cursor": current_snapshot["event_cursor"],
                "plan_root_cid": current_snapshot["plan_root_cid"],
                "repository_tree_id": current_snapshot["repository_tree_id"],
            }
        ),
    }
    replay_witness["witness_cid"] = aseh_operator._identity(replay_witness)
    ready_task_ids = ["ASEH-000"] if ready else []
    live_task_statuses = {"ASEH-000": status}
    live_task_revisions = {"ASEH-000": revision}
    if parallel_blocked:
        live_task_statuses["ASEH-001"] = "blocked"
        live_task_revisions["ASEH-001"] = 2
    authority = {
        "available": True,
        "transport": "quack",
        "credential_path": "sealed_memfd_broker",
        "projection_matches_events": True,
        "projection_reconciliation": replay_witness,
        "owner_binding": binding,
        "snapshot": current_snapshot,
        "task_statuses": live_task_statuses,
        "task_revisions": live_task_revisions,
        "task_cids": integrity["task_cids"],
        "task_owner_bindings": integrity["task_owner_bindings"],
        "task_dependencies": integrity["task_dependencies"],
        "task_authority_spec_cids": integrity["task_authority_spec_cids"],
        "goal_records": integrity["goal_records"],
        "goal_edges": integrity["goal_edges"],
        "plan_record": integrity["plan_record"],
        "objective_record": integrity["objective_record"],
        "ready_task_ids": ready_task_ids,
        "ready_count": len(ready_task_ids),
        "queue_entries": {
            "ASEH-000": {
                "task_cid": task_cid,
                "retry_not_before_ms": delayed_retry_not_before_ms,
            }
        } if delayed_retry_not_before_ms else {},
        "query_started_at_ms": int(observed_at * 1_000),
        "delayed_ready_task_ids": (
            ["ASEH-000"] if delayed_retry_not_before_ms else []
        ),
        "active_count": int(active),
        "blocked_count": int(parallel_blocked),
        "terminal_count": int(status in aseh_operator.TERMINAL_STATUSES),
        "event_cursor": event_cursor,
    }
    sample: dict[str, object] = {
        "observed_at": observed_at,
        "scheduler": {
            "pid": 4242,
            "process_group": 4242,
            "alive": True,
            "returncode": None,
        },
        "owner_status": {
            "lifecycle": "ready",
            "database_path": str(database_path),
            "storage_schema_fingerprint": "storage-schema:aseh-health",
            "identity": owner_identity,
            "read_replica": {
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "read-replica-observation@1"
                ),
                "authority": "non_authoritative_read_replica",
                "live": True,
                **replica_binding,
            },
            "configured_supervisor_credential_broker": {
                "available": True,
                "last_error_type": "",
            },
        },
        "authority": authority,
        "lanes": [
            {
                "lane": 0,
                "fresh": True,
                "admissible": True,
                "watchdog_admissible": True,
                "mtime_ns": lane_mtime_ns,
                "worker_metrics_available": True,
                "worker_census_method": "linux-procfs-descendant-census@1",
                "worker_root_pid": 4321,
                "worker_root_start_time_ticks": 987654,
                "worker_root_boot_id": "boot-id",
                "worker_root_identity_source": "supervised_child_identity",
                "active_worker_count": worker_count,
                "active_worker_pids": list(range(5000, 5000 + worker_count)),
                "worker_descendant_count": worker_count,
                "worker_descendant_pids": list(
                    range(5000, 5000 + worker_count)
                ),
                "worker_phase_guarded": lane_stalled,
                "worker_phase_available": lane_stalled,
                "worker_phase_age_seconds": 0.5 if lane_stalled else None,
                "worker_stall_evidence_available": lane_stalled,
                "stalled_without_active_worker": (
                    True if lane_stalled else None
                ),
            }
        ],
    }
    board = SimpleNamespace(
        max_lanes=1,
        payload={
            "stale_seconds": 2.0,
            "watchdog_startup_grace_seconds": 1.0,
            "initial_projection": {
                "task_count": task_count,
                "goal_count": 1,
                "task_dependency_count": 0,
            },
        },
    )
    return board, {
        "bootstrap_receipt": bootstrap_path,
        "database": database_path,
        "runtime": tmp_path,
    }, sample


def test_aseh_status_sample_rejects_replica_generation_change_during_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.time()
    board, paths, sample = _aseh_health_fixture(
        tmp_path,
        observed_at=now,
        lane_mtime_ns=int(now * 1_000_000_000),
    )
    before = json.loads(json.dumps(sample["owner_status"]))
    after = json.loads(json.dumps(before))
    after["read_replica"]["refresh_sequence"] += 1
    observations = iter(
        (before, after) * aseh_operator.STATUS_REPLICA_STABILITY_ATTEMPTS
    )
    server = SimpleNamespace(status=lambda: next(observations))
    scheduler = SimpleNamespace(pid=os.getpid(), poll=lambda: None)
    broker_calls = []
    monkeypatch.setattr(
        aseh_operator,
        "_broker_status_query",
        lambda *_args, **_kwargs: broker_calls.append(True)
        or {"available": True},
    )
    monkeypatch.setattr(
        aseh_operator,
        "_lane_status_observations",
        lambda *_args, **_kwargs: [],
    )

    observed = aseh_operator._status_sample(
        board, paths, server, scheduler
    )

    assert observed["authority"]["available"] is False
    assert observed["authority"]["error_type"] == "OperatorError"
    assert observed["owner_status"]["read_replica"]["refresh_sequence"] == (
        after["read_replica"]["refresh_sequence"]
    )
    assert len(broker_calls) == (
        aseh_operator.STATUS_REPLICA_STABILITY_ATTEMPTS
    )


def test_aseh_status_sample_retries_the_whole_query_until_replica_is_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.time()
    board, paths, sample = _aseh_health_fixture(
        tmp_path,
        observed_at=now,
        lane_mtime_ns=int(now * 1_000_000_000),
    )
    before = json.loads(json.dumps(sample["owner_status"]))
    after = json.loads(json.dumps(before))
    after["read_replica"]["refresh_sequence"] += 1
    observations = iter((before, after, after, after))
    server = SimpleNamespace(status=lambda: next(observations))
    scheduler = SimpleNamespace(pid=os.getpid(), poll=lambda: None)
    broker_calls = []
    monkeypatch.setattr(
        aseh_operator,
        "_broker_status_query",
        lambda *_args, **_kwargs: broker_calls.append(True)
        or {"available": True},
    )
    monkeypatch.setattr(
        aseh_operator,
        "_lane_status_observations",
        lambda *_args, **_kwargs: [],
    )

    observed = aseh_operator._status_sample(
        board, paths, server, scheduler
    )

    assert observed["authority"]["available"] is True
    assert len(broker_calls) == 2
    assert observed["owner_status"]["read_replica"]["refresh_sequence"] == (
        after["read_replica"]["refresh_sequence"]
    )


@pytest.mark.parametrize(
    "failure_kind",
    ("endpoint_refused", "replica_replaced"),
)
def test_aseh_status_sample_retries_exact_owner_publication_races(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    import duckdb

    now = time.time()
    board, paths, sample = _aseh_health_fixture(
        tmp_path,
        observed_at=now,
        lane_mtime_ns=int(now * 1_000_000_000),
    )
    quack_uri = "quack:127.0.0.1:45123"
    board.resolved_database_program = lambda: SimpleNamespace(
        quack_endpoint=quack_uri
    )
    owner_status = json.loads(json.dumps(sample["owner_status"]))
    observations = iter((owner_status, owner_status) * 2)
    server = SimpleNamespace(status=lambda: next(observations))
    scheduler = SimpleNamespace(pid=os.getpid(), poll=lambda: None)
    calls = 0

    def query(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            if failure_kind == "endpoint_refused":
                raise duckdb.IOException(
                    "IO Error: Failed to send message: IO Error: Could not "
                    "connect to server error for HTTP POST to "
                    "'http://127.0.0.1:45123/quack'"
                )
            raise aseh_operator.OperatorError(
                "published replica changed during shadow copy"
            )
        return {"available": True}

    monkeypatch.setattr(aseh_operator, "_broker_status_query", query)
    monkeypatch.setattr(
        aseh_operator,
        "_lane_status_observations",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        aseh_operator, "STATUS_REPLICA_RETRY_DELAY_SECONDS", 0
    )

    observed = aseh_operator._status_sample(
        board, paths, server, scheduler
    )

    assert observed["authority"] == {"available": True}
    assert calls == 2


def test_aseh_status_sample_does_not_retry_foreign_replica_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.time()
    board, paths, sample = _aseh_health_fixture(
        tmp_path,
        observed_at=now,
        lane_mtime_ns=int(now * 1_000_000_000),
    )
    board.resolved_database_program = lambda: SimpleNamespace(
        quack_endpoint="quack:127.0.0.1:45123"
    )
    owner_status = json.loads(json.dumps(sample["owner_status"]))
    observations = iter((owner_status, owner_status))
    server = SimpleNamespace(status=lambda: next(observations))
    scheduler = SimpleNamespace(pid=os.getpid(), poll=lambda: None)
    calls = 0

    def query(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise aseh_operator.OperatorError(
            "published replica file identity is unsafe"
        )

    monkeypatch.setattr(aseh_operator, "_broker_status_query", query)
    monkeypatch.setattr(
        aseh_operator,
        "_lane_status_observations",
        lambda *_args, **_kwargs: [],
    )

    observed = aseh_operator._status_sample(
        board, paths, server, scheduler
    )

    assert observed["authority"]["available"] is False
    assert observed["authority"]["error"] == (
        "published replica file identity is unsafe"
    )
    assert calls == 1


def test_aseh_external_status_binds_receipt_to_current_owner_incarnation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.time()
    board, paths, before = _aseh_health_fixture(
        tmp_path,
        observed_at=now - 0.25,
        lane_mtime_ns=int((now - 0.25) * 1_000_000_000),
    )
    _board, _paths, current = _aseh_health_fixture(
        tmp_path,
        observed_at=now,
        lane_mtime_ns=int(now * 1_000_000_000),
    )
    receipt = aseh_operator._health_receipt(
        board,
        paths,
        samples=(before, current),
        launched_at=now - 0.5,
        last_progress_at=now,
        failure={},
    )
    assert receipt["healthy"] is True
    owner_directory = tmp_path / "owner"
    status_receipt = tmp_path / "live-status.json"
    paths.update(
        {
            "owner": owner_directory,
            "status_receipt": status_receipt,
        }
    )
    owner_status_path = owner_directory / "quack-state-server.status.json"
    aseh_operator._atomic_json(owner_status_path, current["owner_status"])
    aseh_operator._atomic_json(status_receipt, receipt)
    monkeypatch.setattr(aseh_operator, "_load", lambda _path: (board, {}))
    monkeypatch.setattr(aseh_operator, "_paths", lambda _board: paths)

    exit_code, exact = aseh_operator.status(
        tmp_path / "unused-config.json", require_ready=True
    )
    assert exit_code == 0
    assert exact["healthy"] is True
    assert exact["broker_authenticated_receipt"] is True

    # A recent healthy receipt from the prior owner must not make a newly
    # ready incarnation healthy after restart.
    restarted = json.loads(json.dumps(current["owner_status"]))
    restarted_identity = restarted["identity"]
    restarted_identity["server_id"] = "server:aseh-health-restarted"
    restarted_identity["database_uuid"] = "database:aseh-health-restarted"
    restarted_identity["generation"] += 1
    restarted_identity["process_birth_id"] = "birth:aseh-health-restarted"
    restarted_identity["process_birth"] = {"pid": os.getpid() + 1}
    restarted_identity["listen_uri"] = "quack:127.0.0.1:2"
    restarted_replica = restarted["read_replica"]
    for field in (
        "server_id", "database_uuid", "generation", "schema_revision",
        "schema_fingerprint",
    ):
        restarted_replica[field] = restarted_identity[field]
    restarted_replica["refresh_sequence"] += 1
    restarted_replica["sha256"] = f"sha256:{'b' * 64}"
    aseh_operator._atomic_json(owner_status_path, restarted)

    exit_code, stale = aseh_operator.status(
        tmp_path / "unused-config.json", require_ready=True
    )
    assert exit_code == 1
    assert stale["owner_ready"] is True
    assert stale["healthy"] is False
    assert stale["broker_authenticated_receipt"] is False
    assert stale["receipt_error"]["reason"] == (
        "live_status_receipt_unavailable_or_invalid"
    )


def test_aseh_external_status_accepts_only_monotonic_replica_successor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A read refresh is valid, but rollback/equal-sequence drift is not."""

    now = time.time()
    board, paths, before = _aseh_health_fixture(
        tmp_path,
        observed_at=now - 0.25,
        lane_mtime_ns=int((now - 0.25) * 1_000_000_000),
    )
    _board, _paths, current = _aseh_health_fixture(
        tmp_path,
        observed_at=now,
        lane_mtime_ns=int(now * 1_000_000_000),
    )
    receipt = aseh_operator._health_receipt(
        board,
        paths,
        samples=(before, current),
        launched_at=now - 0.5,
        last_progress_at=now,
        failure={},
    )
    owner_directory = tmp_path / "owner"
    status_receipt = tmp_path / "live-status.json"
    paths.update({"owner": owner_directory, "status_receipt": status_receipt})
    owner_status_path = owner_directory / "quack-state-server.status.json"
    aseh_operator._atomic_json(status_receipt, receipt)
    monkeypatch.setattr(aseh_operator, "_load", lambda _path: (board, {}))
    monkeypatch.setattr(aseh_operator, "_paths", lambda _board: paths)

    successor = json.loads(json.dumps(current["owner_status"]))
    successor_replica = successor["read_replica"]
    successor_replica["refresh_sequence"] += 1
    successor_replica["sha256"] = f"sha256:{'b' * 64}"
    successor_replica["size_bytes"] += 1
    aseh_operator._atomic_json(owner_status_path, successor)

    exit_code, admitted = aseh_operator.status(
        tmp_path / "unused-config.json", require_ready=True
    )
    assert exit_code == 0
    assert admitted["healthy"] is True
    assert admitted["broker_authenticated_receipt"] is True

    equal_sequence_drift = json.loads(json.dumps(current["owner_status"]))
    equal_sequence_drift["read_replica"]["sha256"] = f"sha256:{'c' * 64}"
    aseh_operator._atomic_json(owner_status_path, equal_sequence_drift)
    exit_code, rejected_equal = aseh_operator.status(
        tmp_path / "unused-config.json", require_ready=True
    )
    assert exit_code == 1
    assert rejected_equal["broker_authenticated_receipt"] is False

    later_before = json.loads(json.dumps(before))
    later_current = json.loads(json.dumps(current))
    for sample in (later_before, later_current):
        sample_replica = sample["owner_status"]["read_replica"]
        sample_replica["refresh_sequence"] = 2
        sample_replica["sha256"] = f"sha256:{'d' * 64}"
        sample_replica["size_bytes"] += 2
    later_receipt = aseh_operator._health_receipt(
        board,
        paths,
        samples=(later_before, later_current),
        launched_at=now - 0.5,
        last_progress_at=now,
        failure={},
    )
    aseh_operator._atomic_json(status_receipt, later_receipt)
    aseh_operator._atomic_json(owner_status_path, current["owner_status"])
    exit_code, rejected_rollback = aseh_operator.status(
        tmp_path / "unused-config.json", require_ready=True
    )
    assert exit_code == 1
    assert rejected_rollback["broker_authenticated_receipt"] is False

    aseh_operator._atomic_json(status_receipt, receipt)
    storage_drift = json.loads(json.dumps(successor))
    storage_drift["storage_schema_fingerprint"] = "storage-schema:changed"
    storage_drift["read_replica"]["storage_schema_fingerprint"] = (
        "storage-schema:changed"
    )
    aseh_operator._atomic_json(owner_status_path, storage_drift)
    exit_code, rejected_identity = aseh_operator.status(
        tmp_path / "unused-config.json", require_ready=True
    )
    assert exit_code == 1
    assert rejected_identity["broker_authenticated_receipt"] is False


def test_aseh_external_status_rejects_abruptly_dead_bound_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    try:
        child_birth = read_process_birth(child.pid)
        assert child_birth is not None
        now = time.time()
        board, paths, before = _aseh_health_fixture(
            tmp_path,
            observed_at=now - 0.25,
            lane_mtime_ns=int((now - 0.25) * 1_000_000_000),
        )
        _board, _paths, current = _aseh_health_fixture(
            tmp_path,
            observed_at=now,
            lane_mtime_ns=int(now * 1_000_000_000),
        )
        receipt = aseh_operator._health_receipt(
            board,
            paths,
            samples=(before, current),
            launched_at=now - 0.5,
            last_progress_at=now,
            failure={},
        )
        assert receipt["healthy"] is True
        child_birth_payload = child_birth.to_dict()
        child_birth_id = aseh_operator._state_owner_process_birth_id(
            child_birth_payload
        )
        receipt = json.loads(json.dumps(receipt))
        for sample in receipt["samples"]:
            sample_identity = sample["owner_status"]["identity"]
            sample_identity["process_birth"] = child_birth_payload
            sample_identity["process_birth_id"] = child_birth_id
            sample["authority"]["owner_binding"][
                "process_birth_id"
            ] = child_birth_id
        unsigned_receipt = dict(receipt)
        unsigned_receipt.pop("receipt_cid")
        receipt["receipt_cid"] = aseh_operator._identity(unsigned_receipt)

        child_status = json.loads(json.dumps(current["owner_status"]))
        child_status["identity"]["process_birth"] = child_birth_payload
        child_status["identity"]["process_birth_id"] = child_birth_id
        owner_directory = tmp_path / "owner"
        status_receipt = tmp_path / "live-status.json"
        paths.update(
            {
                "owner": owner_directory,
                "status_receipt": status_receipt,
            }
        )
        reused_status = json.loads(json.dumps(child_status))
        reused_birth = reused_status["identity"]["process_birth"]
        reused_birth["start_time_ticks"] += 1
        reused_status["identity"][
            "process_birth_id"
        ] = aseh_operator._state_owner_process_birth_id(reused_birth)
        with pytest.raises(aseh_operator.OperatorError, match="not alive"):
            aseh_operator._owner_incarnation_binding(reused_status, paths)

        owner_status_path = owner_directory / "quack-state-server.status.json"
        aseh_operator._atomic_json(owner_status_path, child_status)
        aseh_operator._atomic_json(status_receipt, receipt)
        monkeypatch.setattr(aseh_operator, "_load", lambda _path: (board, {}))
        monkeypatch.setattr(aseh_operator, "_paths", lambda _board: paths)

        exit_code, alive = aseh_operator.status(
            tmp_path / "unused-config.json", require_ready=True
        )
        assert exit_code == 0
        assert alive["healthy"] is True

        child.kill()
        assert child.wait(timeout=5) == -signal.SIGKILL
        exit_code, dead = aseh_operator.status(
            tmp_path / "unused-config.json", require_ready=True
        )
        assert exit_code == 1
        assert dead["owner_ready"] is True
        assert dead["healthy"] is False
        assert dead["broker_authenticated_receipt"] is False
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_aseh_health_heartbeat_only_cannot_mask_stuck_board(
    tmp_path: Path,
) -> None:
    now = time.time()
    board, paths, before = _aseh_health_fixture(
        tmp_path,
        observed_at=now - 1.0,
        lane_mtime_ns=int((now - 1.0) * 1_000_000_000),
    )
    _board, _paths, current = _aseh_health_fixture(
        tmp_path,
        observed_at=now,
        lane_mtime_ns=int(now * 1_000_000_000),
    )
    receipt = aseh_operator._health_receipt(
        board,
        paths,
        samples=(before, current),
        launched_at=now - 100.0,
        last_progress_at=now - 100.0,
        failure={},
    )

    assert receipt["progress_evidence"] == []
    assert receipt["liveness_evidence"] == ["lane_heartbeat_advanced"]
    assert receipt["stuck"] is True
    assert receipt["healthy"] is False

    cursor_only = json.loads(json.dumps(current))
    cursor_only["authority"]["event_cursor"] = 11
    cursor_only["authority"]["snapshot"]["event_cursor"] = 11
    assert aseh_operator._authoritative_progress_between(
        current, cursor_only
    ) == []

    current["lanes"][0]["stalled_without_active_worker"] = True  # type: ignore[index]
    stalled = aseh_operator._health_receipt(
        board,
        paths,
        samples=(before, current),
        launched_at=now - 0.5,
        last_progress_at=now,
        failure={},
    )
    assert stalled["lane_stalled_without_active_worker"] is True
    assert stalled["stuck"] is True
    assert stalled["healthy"] is False


def test_aseh_health_requires_two_sample_semantic_authority_and_exact_terminal(
    tmp_path: Path,
) -> None:
    now = time.time()
    board, paths, before = _aseh_health_fixture(
        tmp_path,
        observed_at=now - 1.0,
        lane_mtime_ns=int((now - 1.0) * 1_000_000_000),
    )
    _board, _paths, current = _aseh_health_fixture(
        tmp_path,
        observed_at=now,
        lane_mtime_ns=int(now * 1_000_000_000),
    )
    forged = json.loads(json.dumps(current))
    forged["authority"].pop("projection_reconciliation")
    forged["authority"]["projection_matches_events"] = True
    forged_receipt = aseh_operator._health_receipt(
        board,
        paths,
        samples=(before, forged),
        launched_at=now - 0.5,
        last_progress_at=now,
        failure={},
    )
    assert forged_receipt["source_identity_admitted"] is False
    assert forged_receipt["healthy"] is False

    before["authority"]["objective_record"]["title"] = "amended"  # type: ignore[index]
    repaired = aseh_operator._health_receipt(
        board,
        paths,
        samples=(before, current),
        launched_at=now - 0.5,
        last_progress_at=now,
        failure={},
    )
    assert repaired["semantic_corpus_admitted"] is False
    assert repaired["healthy"] is False
    assert aseh_operator._post_admission_health_action(
        repaired,
        prior_available=True,
        current_available=True,
        unhealthy_edges=0,
    )[:2] == ("fail", "authoritative_health_admission_lost")

    _board, _paths, terminal_before = _aseh_health_fixture(
        tmp_path,
        status="failed",
        revision=2,
        ready=False,
        observed_at=now - 1.0,
        lane_mtime_ns=int((now - 1.0) * 1_000_000_000),
    )
    _board, _paths, terminal_current = _aseh_health_fixture(
        tmp_path,
        status="failed",
        revision=2,
        ready=False,
        observed_at=now,
        lane_mtime_ns=int(now * 1_000_000_000),
    )
    terminal_receipt = aseh_operator._health_receipt(
        board,
        paths,
        samples=(terminal_before, terminal_current),
        launched_at=now - 10.0,
        last_progress_at=now - 1.0,
        failure={},
    )
    assert terminal_receipt["terminal"] is True
    assert terminal_receipt["healthy"] is True
    assert aseh_operator._post_admission_health_action(
        terminal_receipt,
        prior_available=True,
        current_available=True,
        unhealthy_edges=0,
    )[:2] == ("stop", "")

    terminal_current["authority"]["task_authority_spec_cids"][  # type: ignore[index]
        "ASEH-000"
    ] = "b" + ("a" * 60)
    rejected_terminal = aseh_operator._health_receipt(
        board,
        paths,
        samples=(terminal_before, terminal_current),
        launched_at=now - 10.0,
        last_progress_at=now - 1.0,
        failure={},
    )
    assert rejected_terminal["terminal"] is True
    assert rejected_terminal["healthy"] is False
    assert aseh_operator._post_admission_health_action(
        rejected_terminal,
        prior_available=True,
        current_available=True,
        unhealthy_edges=0,
    )[:2] == ("fail", "authoritative_terminal_not_admitted")


def test_aseh_restart_admits_only_monotonic_lifecycle_on_sealed_corpus(
    tmp_path: Path,
) -> None:
    now = time.time()
    _board, paths, sample = _aseh_health_fixture(
        tmp_path,
        observed_at=now,
        lane_mtime_ns=int(now * 1_000_000_000),
    )
    bootstrap = aseh_operator._secure_runtime_json(
        paths["bootstrap_receipt"],
        max_bytes=aseh_operator.STATUS_RECEIPT_MAX_BYTES,
    )
    snapshot = dict(bootstrap["snapshot"])
    snapshot["event_cursor"] = int(snapshot["event_cursor"]) + 1
    snapshot["projection_cid"] = "projection:advanced"
    snapshot["source_identity"] = aseh_operator.content_identity(
        {
            "plan_root_cid": snapshot["plan_root_cid"],
            "repository_tree_id": snapshot["repository_tree_id"],
            "projection_cid": snapshot["projection_cid"],
        }
    )
    integrity = json.loads(json.dumps(bootstrap["integrity"]))
    integrity["event_cursor"] = snapshot["event_cursor"]
    integrity["task_revisions"]["ASEH-000"] += 1
    integrity["task_statuses"]["ASEH-000"] = "in_progress"
    integrity["projection_cid"] = "projection:advanced"
    integrity["integrity_receipt_id"] = aseh_operator._identity(
        {
            key: value
            for key, value in integrity.items()
            if key != "integrity_receipt_id"
        }
    )
    aseh_operator._admit_current_projection_against_bootstrap(
        bootstrap, snapshot, integrity
    )

    integrity["objective_record"]["title"] = "amended"
    with pytest.raises(aseh_operator.OperatorError, match="immutable corpus"):
        aseh_operator._admit_current_projection_against_bootstrap(
            bootstrap, snapshot, integrity
        )


def test_aseh_lane_status_projects_worker_watchdog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aseh_operator, "ROOT", tmp_path)
    state_root = tmp_path / "state"
    status_path = state_root / "lane-0" / "aseh_lane_0_supervisor_status.json"
    status_path.parent.mkdir(parents=True)
    payload = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "todo_implementation_supervisor.supervisor"
        ),
        "repo_root": str(tmp_path),
        "task_prefix": "## ASEH-",
        "state_prefix": "aseh_lane_0",
        "status": "running",
        "run_id": "run-1",
        "daemon_pid": 4321,
        "worker_metrics_available": True,
        "worker_metrics_unavailable_reason": "",
        "worker_census_method": "linux-procfs-descendant-census@1",
        "worker_root_pid": 4321,
        "worker_root_start_time_ticks": 987654,
        "worker_root_boot_id": "boot-id",
        "worker_root_identity_source": "supervised_child_identity",
        "worker_observed_at_ns": time.time_ns(),
        "worker_observation_generation": (
            "run-1:4321:987654:boot-id"
        ),
        "active_worker_count": 0,
        "active_worker_pids": [],
        "worker_descendant_count": 0,
        "worker_descendant_pids": [],
        "worker_phase": "",
        "worker_phase_available": False,
        "worker_phase_known": True,
        "worker_phase_known_non_worktree": False,
        "worker_phase_guarded": False,
        "worker_phase_age_seconds": None,
        "worker_stall_evidence_available": False,
        "worker_stall_evidence_unavailable_reason": "phase_not_guarded",
        "stalled_without_active_worker": None,
    }
    status_path.write_text(json.dumps(payload), encoding="utf-8")
    board = SimpleNamespace(
        max_lanes=1,
        task_prefix="ASEH",
        task_header_prefix="## ASEH-",
        repo_root=tmp_path,
        runtime_paths={"state": "state"},
        path=lambda value: tmp_path / Path(value),
    )

    observations = aseh_operator._lane_status_observations(
        board, now=time.time()
    )

    assert observations[0]["watchdog_admissible"] is True
    assert observations[0]["active_worker_count"] == 0
    assert observations[0]["worker_phase_age_seconds"] is None
    assert observations[0]["stalled_without_active_worker"] is None

    # A no-work maintenance pass publishes this fresh, quiescent state until
    # the next managed-daemon cycle.  It remains an admitted live lane when
    # the exact worker census and watchdog evidence above are current.
    payload["status"] = "agentic_maintenance_completed"
    payload["worker_observed_at_ns"] = time.time_ns()
    status_path.write_text(json.dumps(payload), encoding="utf-8")
    maintenance_completed = aseh_operator._lane_status_observations(
        board, now=time.time()
    )
    assert maintenance_completed[0]["admissible"] is True
    assert maintenance_completed[0]["watchdog_admissible"] is True

    payload["status"] = "agentic_maintenance_failed"
    payload["worker_observed_at_ns"] = time.time_ns()
    status_path.write_text(json.dumps(payload), encoding="utf-8")
    maintenance_failed = aseh_operator._lane_status_observations(
        board, now=time.time()
    )
    assert maintenance_failed[0]["admissible"] is False

    payload["status"] = "running"

    payload["worker_observed_at_ns"] = time.time_ns() + 250_000_000
    status_path.write_text(json.dumps(payload), encoding="utf-8")
    future_obs = aseh_operator._lane_status_observations(
        board, now=time.time()
    )
    assert future_obs[0]["watchdog_admissible"] is False
    assert future_obs[0]["worker_observation_age_seconds"] < 0.0
    payload["worker_observed_at_ns"] = time.time_ns()

    payload.update(
        {
            "worker_phase": "validating_reconciled_candidate",
            "worker_phase_available": True,
            "worker_phase_known": True,
            "worker_phase_known_non_worktree": True,
        }
    )
    status_path.write_text(json.dumps(payload), encoding="utf-8")
    known_non_worktree = aseh_operator._lane_status_observations(
        board,
        now=time.time(),
    )
    assert known_non_worktree[0]["watchdog_admissible"] is True

    payload.update(
        {
            "worker_phase": "implementng",
            "worker_phase_available": True,
            "worker_phase_known": False,
            "worker_phase_known_non_worktree": False,
            "worker_stall_evidence_unavailable_reason": "phase_unknown",
        }
    )
    status_path.write_text(json.dumps(payload), encoding="utf-8")
    unknown_phase = aseh_operator._lane_status_observations(
        board,
        now=time.time(),
    )
    assert unknown_phase[0]["watchdog_admissible"] is False

    payload.update(
        {
            "worker_phase": "validating",
            "worker_phase_available": True,
            "worker_phase_known": True,
            "worker_phase_known_non_worktree": True,
            "worker_stall_evidence_unavailable_reason": "phase_not_guarded",
        }
    )
    status_path.write_text(json.dumps(payload), encoding="utf-8")
    validating = aseh_operator._lane_status_observations(
        board,
        now=time.time(),
    )
    assert validating[0]["watchdog_admissible"] is True
    assert validating[0]["worker_phase_known_non_worktree"] is True

    payload.update(
        {
            "worker_phase": "implementng",
            "worker_phase_available": True,
            "worker_phase_known": True,
            "worker_phase_known_non_worktree": True,
        }
    )
    status_path.write_text(json.dumps(payload), encoding="utf-8")
    forged_phase = aseh_operator._lane_status_observations(
        board,
        now=time.time(),
    )
    assert forged_phase[0]["watchdog_admissible"] is False

    payload.update(
        {
            "worker_phase": "",
            "worker_phase_available": False,
            "worker_phase_known": None,
            "worker_phase_known_non_worktree": None,
            "worker_phase_guarded": None,
            "worker_stall_evidence_unavailable_reason": "worker_metrics_unavailable",
            "worker_metrics_available": False,
            "worker_metrics_unavailable_reason": "procfs_unavailable",
            "active_worker_count": None,
            "active_worker_pids": None,
            "worker_descendant_count": None,
            "worker_descendant_pids": None,
        }
    )
    status_path.write_text(json.dumps(payload), encoding="utf-8")
    unavailable = aseh_operator._lane_status_observations(
        board,
        now=time.time(),
    )
    assert unavailable[0]["watchdog_admissible"] is False
    assert unavailable[0]["active_worker_count"] is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("worker_root_boot_id", ""),
        ("worker_root_identity_source", "captured_before_census"),
        ("worker_root_pid", 4322),
        ("worker_observed_at_ns", 1),
        ("worker_observed_at_ns", time.time_ns() + 60_000_000_000),
        ("worker_observation_generation", "run-1:4321:1:boot-id"),
    ],
)
def test_aseh_lane_status_rejects_unsealed_worker_root_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    monkeypatch.setattr(aseh_operator, "ROOT", tmp_path)
    state_root = tmp_path / "state"
    status_path = state_root / "lane-0" / "aseh_lane_0_supervisor_status.json"
    status_path.parent.mkdir(parents=True)
    payload = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "todo_implementation_supervisor.supervisor"
        ),
        "repo_root": str(tmp_path),
        "task_prefix": "## ASEH-",
        "state_prefix": "aseh_lane_0",
        "status": "running",
        "run_id": "run-1",
        "daemon_pid": 4321,
        "worker_metrics_available": True,
        "worker_census_method": "linux-procfs-descendant-census@1",
        "worker_root_pid": 4321,
        "worker_root_start_time_ticks": 987654,
        "worker_root_boot_id": "boot-id",
        "worker_root_identity_source": "supervised_child_identity",
        "worker_observed_at_ns": time.time_ns(),
        "worker_observation_generation": (
            "run-1:4321:987654:boot-id"
        ),
        "active_worker_count": 0,
        "active_worker_pids": [],
        "worker_descendant_count": 0,
        "worker_descendant_pids": [],
        "worker_phase": "",
        "worker_phase_available": False,
        "worker_phase_known": True,
        "worker_phase_known_non_worktree": False,
        "worker_phase_guarded": False,
        "worker_stall_evidence_available": False,
        "worker_stall_evidence_unavailable_reason": "phase_not_guarded",
        "stalled_without_active_worker": None,
        field: value,
    }
    status_path.write_text(json.dumps(payload), encoding="utf-8")
    board = SimpleNamespace(
        max_lanes=1,
        task_prefix="ASEH",
        task_header_prefix="## ASEH-",
        repo_root=tmp_path,
        runtime_paths={"state": "state"},
        path=lambda item: tmp_path / Path(item),
    )

    observations = aseh_operator._lane_status_observations(
        board,
        now=time.time(),
    )

    assert observations[0]["watchdog_admissible"] is False


def test_known_non_worktree_phase_allowlist_is_exact_and_not_guarded() -> None:
    expected = frozenset(
        {
            "merge_queue",
            "merge_reconciliation",
            "validating",
            "validating_reconciled_candidate",
        }
    )
    assert todo_supervisor.KNOWN_NON_WORKTREE_PHASES == expected

    for phase in sorted(expected):
        status = todo_supervisor.worktree_phase_worker_status(
            {"active_phase": phase},
            daemon_pid=1234,
            threshold_seconds=60,
            descendants=[],
        )
        assert status["required"] is False
        assert status["phase_known"] is True
        assert status["phase_known_non_worktree"] is True
        assert status["stall_evidence_unavailable_reason"] == "phase_not_guarded"
        assert status["stalled_without_active_worker"] is None

    unknown = todo_supervisor.worktree_phase_worker_status(
        {"active_phase": "implementng"},
        daemon_pid=1234,
        threshold_seconds=60,
        descendants=[],
    )
    assert unknown["phase_known"] is False
    assert unknown["phase_known_non_worktree"] is False
    assert unknown["stall_evidence_unavailable_reason"] == "phase_unknown"


def test_worker_observation_binds_exact_run_child_and_root_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = object.__new__(SupervisorLoop)
    loop.last_run_id = "run-7"
    loop._last_worker_status = {}
    monkeypatch.setattr(
        supervisor_loop_module.time,
        "time_ns",
        lambda: 1_700_000_000_000_000_000,
    )

    observed = loop._record_worker_observation(
        SimpleNamespace(pid=4321),
        {
            "worker_metrics_available": True,
            "worker_root_start_time_ticks": 987654,
            "worker_root_boot_id": "boot-id",
        },
    )

    assert observed["worker_observed_at_ns"] == 1_700_000_000_000_000_000
    assert observed["worker_observation_generation"] == (
        "run-7:4321:987654:boot-id"
    )
    assert loop._last_worker_status == observed

    unavailable = loop._record_worker_observation(
        SimpleNamespace(pid=4321),
        {"worker_metrics_available": False},
    )
    assert unavailable["worker_observed_at_ns"] == 1_700_000_000_000_000_000
    assert unavailable["worker_observation_generation"] == ""


def test_supervisor_loop_publishes_worker_census_before_startup_grace(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    state_dir = repo / "state"
    state_dir.mkdir(parents=True)
    identity_path = state_dir / "child.identity.json"
    command = (
        sys.executable,
        "-c",
        "import time; time.sleep(0.2)",
    )
    spec = ManagedDaemonSpec(
        name="aseh-census-test",
        schema="test.aseh-census",
        repo_root=repo,
        daemon_dir=state_dir,
        runner=command,
        status_path=state_dir / "daemon_status.json",
        supervisor_status_path=state_dir / "supervisor_status.json",
        supervisor_pid_path=state_dir / "supervisor.pid",
        child_pid_path=state_dir / "child.pid",
        supervisor_out_path=state_dir / "supervisor.out",
        ensure_status_path=state_dir / "ensure_status.json",
        ensure_check_path=state_dir / "ensure_check.json",
    )
    watchdog_calls: list[bool] = []
    loop = SupervisorLoop(
        SupervisorLoopConfig(
            spec=spec,
            command=command,
            log_prefix="child",
            heartbeat_seconds=0.01,
            poll_seconds=0.01,
            watchdog_startup_grace_seconds=3600,
            max_restarts=1,
            child_env={
                SUPERVISED_CHILD_IDENTITY_PATH_ENV: str(identity_path),
                SUPERVISED_CHILD_OWNER_SCOPE_ENV: json.dumps(
                    {"test": "aseh-pre-grace-census"},
                    sort_keys=True,
                ),
            },
        ),
        watchdog_hook=lambda *_args: watchdog_calls.append(True),
    )
    snapshots: list[dict[str, object]] = []
    original_write = loop._safe_write_status

    def record_status(*args, **kwargs) -> None:
        original_write(*args, **kwargs)
        snapshots.append(
            json.loads(
                (state_dir / "supervisor_status.json").read_text(
                    encoding="utf-8"
                )
            )
        )

    loop._safe_write_status = record_status  # type: ignore[method-assign]

    result = loop.run()

    live = [
        item
        for item in snapshots
        if item.get("status") in {"starting", "running"}
    ]
    assert result.status == "child_exited"
    assert watchdog_calls == []
    assert {item["status"] for item in live} == {"starting", "running"}
    available_live = [
        item for item in live if item["worker_metrics_available"] is True
    ]
    assert {item["status"] for item in available_live} == {
        "starting",
        "running",
    }
    assert all(
        item["active_worker_count"] is None
        for item in live
        if item["worker_metrics_available"] is False
    )
    assert all(
        item["worker_census_method"]
        == "linux-procfs-descendant-census@1"
        for item in available_live
    )
    assert all(
        item["worker_root_identity_source"] == "supervised_child_identity"
        for item in available_live
    )
    assert all(
        type(item["worker_root_start_time_ticks"]) is int
        for item in available_live
    )
    assert all(bool(item["worker_root_boot_id"]) for item in available_live)
    assert all(
        type(item["active_worker_count"]) is int for item in available_live
    )
    assert all(
        isinstance(item["active_worker_pids"], list)
        for item in available_live
    )
    assert all(item["worker_phase"] == "" for item in available_live)
    assert all(
        item["worker_phase_guarded"] is False for item in available_live
    )
    assert all(item["worker_phase_known"] is True for item in available_live)
    assert all(
        item["worker_phase_known_non_worktree"] is False
        for item in available_live
    )
    assert all(
        item["stalled_without_active_worker"] is None
        for item in available_live
    )
    assert all(
        type(item["worker_observed_at_ns"]) is int
        and item["worker_observed_at_ns"] > 0
        for item in available_live
    )
    assert all(
        item["worker_observation_generation"]
        == (
            f"{item['run_id']}:{item['worker_root_pid']}:"
            f"{item['worker_root_start_time_ticks']}:"
            f"{item['worker_root_boot_id']}"
        )
        for item in available_live
    )


def test_supervisor_loop_publishes_unavailable_census_without_false_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    state_dir = repo / "state"
    state_dir.mkdir(parents=True)
    spec = ManagedDaemonSpec(
        name="aseh-census-test",
        schema="test.aseh-census",
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
    loop = SupervisorLoop(
        SupervisorLoopConfig(
            spec=spec,
            command=(sys.executable, "-c", "pass"),
            log_prefix="child",
        )
    )
    birth = current_process_birth()
    child = SimpleNamespace(
        pid=os.getpid(),
        identity_process_birth=birth,
    )

    def unavailable_census(_pid: int) -> list[dict[str, object]]:
        raise OSError("procfs unavailable")

    monkeypatch.setattr(
        supervisor_loop_module,
        "procfs_descendant_processes",
        unavailable_census,
    )

    loop._observe_worker_status(child, {})
    loop._write_status("running", child=child)
    status = json.loads(
        (state_dir / "supervisor_status.json").read_text(encoding="utf-8")
    )

    assert status["worker_metrics_available"] is False
    assert status["worker_metrics_unavailable_reason"] == "worker_census_unavailable"
    assert status["active_worker_count"] is None
    assert status["active_worker_pids"] is None
    assert status["worker_descendant_count"] is None
    assert status["worker_descendant_pids"] is None
    assert status["stalled_without_active_worker"] is None


def test_cleanup_watchdog_cannot_extend_implementation_worker_lease() -> None:
    command = (
        "/usr/bin/python3 -m "
        "ipfs_accelerate_py.agent_supervisor.grok_cli_runner "
        "--internal-docker-cleanup-watchdog --container-id provider-1"
    )

    assert todo_supervisor._is_agent_worker_command(command) is False


@pytest.mark.parametrize("mutation", ["reparent", "pid_reuse_after_argv"])
def test_procfs_worker_census_rejects_ancestry_and_pid_races(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    proc_root = tmp_path / "proc"
    root_pid = 4100
    worker_pid = 4101

    def stat_record(pid: int, parent: int, start_ticks: int) -> str:
        fields = ["0"] * 50
        fields[0] = "S"
        fields[1] = str(parent)
        fields[19] = str(start_ticks)
        return f"{pid} (worker) {' '.join(fields)}\n"

    for pid, parent, start_ticks in (
        (root_pid, 1, 100),
        (worker_pid, root_pid, 101),
    ):
        process_dir = proc_root / str(pid)
        process_dir.mkdir(parents=True)
        (process_dir / "stat").write_text(
            stat_record(pid, parent, start_ticks),
            encoding="utf-8",
        )
        (process_dir / "cmdline").write_bytes(
            b"grok\x00--workspace\x00/tmp/task\x00"
        )

    original = todo_supervisor._strict_procfs_process_identity
    worker_reads = 0

    def raced_identity(path: Path):
        nonlocal worker_reads
        observed = original(path)
        if path == proc_root / str(worker_pid) / "stat":
            worker_reads += 1
            if mutation == "reparent" and worker_reads == 2:
                return (1, 101, "S")
            if mutation == "pid_reuse_after_argv" and worker_reads == 3:
                return (root_pid, 202, "S")
        return observed

    monkeypatch.setattr(
        todo_supervisor,
        "_strict_procfs_process_identity",
        raced_identity,
    )

    with pytest.raises(OSError, match="ancestry or identity changed"):
        todo_supervisor.procfs_descendant_processes(
            root_pid,
            proc_root=proc_root,
        )


def test_aseh_health_zero_frontier_dependency_deadlock_is_blocked_and_stuck(
    tmp_path: Path,
) -> None:
    now = time.time()
    board, paths, before = _aseh_health_fixture(
        tmp_path,
        ready=False,
        observed_at=now - 0.25,
        lane_mtime_ns=int((now - 0.25) * 1_000_000_000),
    )
    _board, _paths, current = _aseh_health_fixture(
        tmp_path,
        ready=False,
        observed_at=now,
        lane_mtime_ns=int(now * 1_000_000_000),
    )
    receipt = aseh_operator._health_receipt(
        board,
        paths,
        samples=(before, current),
        launched_at=now - 0.5,
        last_progress_at=now,
        failure={},
    )

    assert receipt["dependency_deadlock"] is True
    assert receipt["blocked"] is True
    assert receipt["stuck"] is True
    assert receipt["healthy"] is False


def test_aseh_health_gives_exact_blocked_reconciliation_a_bounded_window(
    tmp_path: Path,
) -> None:
    now = time.time()
    board, paths, before = _aseh_health_fixture(
        tmp_path,
        status="blocked",
        revision=2,
        event_cursor=11,
        ready=False,
        observed_at=now - 0.25,
        lane_mtime_ns=int((now - 0.25) * 1_000_000_000),
    )
    _board, _paths, current = _aseh_health_fixture(
        tmp_path,
        status="blocked",
        revision=2,
        event_cursor=11,
        ready=False,
        observed_at=now,
        lane_mtime_ns=int(now * 1_000_000_000),
    )
    before["authority"]["blocked_count"] = 1  # type: ignore[index]
    current["authority"]["blocked_count"] = 1  # type: ignore[index]

    receipt = aseh_operator._health_receipt(
        board,
        paths,
        samples=(before, current),
        launched_at=now - 0.5,
        last_progress_at=now - 0.25,
        failure={},
    )

    assert receipt["blocked"] is True
    assert receipt["stuck"] is True
    assert receipt["healthy"] is False
    assert receipt["blocked_recovery_admitted"] is True
    edges = 2
    for _index in range(8):
        action, reason, edges = (
            aseh_operator._post_admission_health_action(
                receipt,
                prior_available=True,
                current_available=True,
                unhealthy_edges=edges,
            )
        )
        assert (action, reason) == ("continue", "")
        assert edges == 0

    expired = aseh_operator._health_receipt(
        board,
        paths,
        samples=(before, current),
        launched_at=now - 10.0,
        last_progress_at=now - 3.0,
        failure={},
    )
    assert expired["blocked_recovery_admitted"] is False
    assert aseh_operator._post_admission_health_action(
        expired,
        prior_available=True,
        current_available=True,
        unhealthy_edges=edges,
    )[:2] == ("fail", "authoritative_board_blocked")

    current["authority"]["objective_record"]["title"] = (  # type: ignore[index]
        "unsealed mutation"
    )
    rejected = aseh_operator._health_receipt(
        board,
        paths,
        samples=(before, current),
        launched_at=now - 0.5,
        last_progress_at=now - 0.25,
        failure={},
    )
    assert rejected["blocked_recovery_admitted"] is False
    assert aseh_operator._post_admission_health_action(
        rejected,
        prior_available=True,
        current_available=True,
        unhealthy_edges=0,
    )[:2] == ("fail", "authoritative_board_blocked")


def test_aseh_health_blocked_reconciliation_allows_startup_lane_refresh(
    tmp_path: Path,
) -> None:
    now = time.time()
    board, paths, before = _aseh_health_fixture(
        tmp_path,
        status="blocked",
        revision=2,
        event_cursor=11,
        ready=False,
        observed_at=now - 0.25,
        lane_mtime_ns=int((now - 100.0) * 1_000_000_000),
    )
    _board, _paths, current = _aseh_health_fixture(
        tmp_path,
        status="blocked",
        revision=2,
        event_cursor=11,
        ready=False,
        observed_at=now,
        lane_mtime_ns=int((now - 100.0) * 1_000_000_000),
    )
    for sample in (before, current):
        sample["authority"]["blocked_count"] = 1  # type: ignore[index]
        sample["lanes"][0]["fresh"] = False  # type: ignore[index]
        sample["lanes"][0]["watchdog_admissible"] = False  # type: ignore[index]

    startup = aseh_operator._health_receipt(
        board,
        paths,
        samples=(before, current),
        launched_at=now - 0.5,
        last_progress_at=now - 0.25,
        failure={},
    )
    assert startup["startup_grace_active"] is True
    assert startup["lane_heartbeat_fresh"] is False
    assert startup["blocked_recovery_admitted"] is True

    after_startup = aseh_operator._health_receipt(
        board,
        paths,
        samples=(before, current),
        launched_at=now - 2.0,
        last_progress_at=now - 0.25,
        failure={},
    )
    assert after_startup["startup_grace_active"] is False
    assert after_startup["blocked_recovery_admitted"] is False


def test_aseh_health_admits_parallel_blocked_recovery_only_during_startup(
    tmp_path: Path,
) -> None:
    now = time.time()
    board, paths, before = _aseh_health_fixture(
        tmp_path,
        status="todo",
        revision=1,
        event_cursor=10,
        ready=True,
        observed_at=now - 0.25,
        lane_mtime_ns=int((now - 0.25) * 1_000_000_000),
        parallel_blocked=True,
    )
    _board, _paths, current = _aseh_health_fixture(
        tmp_path,
        status="todo",
        revision=1,
        event_cursor=10,
        ready=True,
        observed_at=now,
        lane_mtime_ns=int(now * 1_000_000_000),
        parallel_blocked=True,
    )

    startup = aseh_operator._health_receipt(
        board,
        paths,
        samples=(before, current),
        launched_at=now - 0.5,
        last_progress_at=now - 0.25,
        failure={},
    )
    assert startup["blocked"] is True
    assert startup["dependency_deadlock"] is False
    assert startup["blocked_recovery_scope"] == "parallel_startup"
    assert startup["blocked_recovery_admitted"] is True
    assert startup["healthy"] is False

    expired = aseh_operator._health_receipt(
        board,
        paths,
        samples=(before, current),
        launched_at=now - 2.0,
        last_progress_at=now - 0.25,
        failure={},
    )
    assert expired["startup_grace_active"] is False
    assert expired["blocked_recovery_scope"] == ""
    assert expired["blocked_recovery_admitted"] is False


def test_aseh_health_admits_exact_delayed_retry_frontier(
    tmp_path: Path,
) -> None:
    now = time.time()
    retry_at_ms = int((now + 30.0) * 1_000)
    board, paths, before = _aseh_health_fixture(
        tmp_path,
        ready=False,
        observed_at=now - 0.25,
        lane_mtime_ns=int((now - 0.25) * 1_000_000_000),
        delayed_retry_not_before_ms=retry_at_ms,
    )
    _board, _paths, current = _aseh_health_fixture(
        tmp_path,
        ready=False,
        observed_at=now,
        lane_mtime_ns=int(now * 1_000_000_000),
        delayed_retry_not_before_ms=retry_at_ms,
    )
    receipt = aseh_operator._health_receipt(
        board,
        paths,
        samples=(before, current),
        launched_at=now - 100.0,
        last_progress_at=now - 100.0,
        failure={},
    )

    assert receipt["delayed_frontier_admitted"] is True
    assert receipt["delayed_ready_task_ids"] == ["ASEH-000"]
    assert receipt["dependency_deadlock"] is False
    assert receipt["blocked"] is False
    assert receipt["stuck"] is False
    assert receipt["healthy"] is True


def test_aseh_health_rejects_outage_progress_and_bounds_recovery_edges(
    tmp_path: Path,
) -> None:
    now = time.time()
    board, paths, before = _aseh_health_fixture(
        tmp_path,
        observed_at=now - 0.25,
        lane_mtime_ns=int((now - 0.25) * 1_000_000_000),
    )
    _board, _paths, current = _aseh_health_fixture(
        tmp_path,
        status="claimed",
        revision=2,
        event_cursor=11,
        ready=False,
        active=True,
        observed_at=now,
        lane_mtime_ns=int(now * 1_000_000_000),
    )
    unavailable = json.loads(json.dumps(before))
    unavailable["authority"] = {
        "available": False,
        "task_statuses": {},
        "task_revisions": {},
        "event_cursor": 0,
    }
    assert aseh_operator._authoritative_progress_between(
        unavailable, current
    ) == []
    assert aseh_operator._authoritative_progress_between(
        before, unavailable
    ) == []

    rollback = json.loads(json.dumps(current))
    rollback["authority"]["task_revisions"]["ASEH-000"] = 0
    rollback["authority"]["event_cursor"] = 12
    rollback["authority"]["snapshot"]["event_cursor"] = 12
    rejected = aseh_operator._health_receipt(
        board,
        paths,
        samples=(current, rollback),
        launched_at=now - 1.0,
        last_progress_at=now,
        failure={},
    )
    assert rejected["task_authority_pair"]["revision_monotonic"] is False
    assert rejected["healthy"] is False

    unhealthy = {
        "healthy": False,
        "blocked": False,
        "stuck": False,
        "scheduler_alive": True,
        "owner_ready": True,
        "broker_ready": True,
    }
    edges = 0
    for prior_available, current_available in (
        (True, False),
        (False, True),
    ):
        action, reason, edges = aseh_operator._post_admission_health_action(
            unhealthy,
            prior_available=prior_available,
            current_available=current_available,
            unhealthy_edges=edges,
        )
        assert (action, reason) == ("continue", "")
    action, reason, edges = aseh_operator._post_admission_health_action(
        unhealthy,
        prior_available=True,
        current_available=False,
        unhealthy_edges=edges,
    )
    assert action == "fail"
    assert reason == "authoritative_status_recovery_grace_exhausted"
    assert edges == 3


def test_aseh_post_admission_grace_is_exclusive_to_typed_lane_loss() -> None:
    lane_only = {
        "healthy": False,
        "blocked": False,
        "stuck": False,
        "terminal": False,
        "scheduler_alive": True,
        "owner_ready": True,
        "broker_ready": True,
        "health_without_lane_admitted": True,
        "lane_heartbeat_fresh": False,
    }

    edges = 0
    for _index in range(2):
        action, reason, edges = aseh_operator._post_admission_health_action(
            lane_only,
            prior_available=True,
            current_available=True,
            unhealthy_edges=edges,
        )
        assert (action, reason) == ("continue", "")
    action, reason, edges = aseh_operator._post_admission_health_action(
        lane_only,
        prior_available=True,
        current_available=True,
        unhealthy_edges=edges,
    )
    assert (action, reason, edges) == (
        "fail",
        "authoritative_health_admission_lost",
        3,
    )

    healthy = {**lane_only, "healthy": True, "lane_heartbeat_fresh": True}
    assert aseh_operator._post_admission_health_action(
        healthy,
        prior_available=True,
        current_available=True,
        unhealthy_edges=2,
    ) == ("continue", "", 0)

    for field, value in (
        ("scheduler_alive", False),
        ("owner_ready", False),
        ("broker_ready", False),
        ("health_without_lane_admitted", False),
        ("lane_heartbeat_fresh", None),
    ):
        degraded = {**lane_only, field: value}
        action, _reason, next_edges = (
            aseh_operator._post_admission_health_action(
                degraded,
                prior_available=True,
                current_available=True,
                unhealthy_edges=0,
            )
        )
        assert action == "fail", field
        assert next_edges == 0

    scheduler_lost = {**lane_only, "scheduler_alive": False}
    assert aseh_operator._post_admission_health_action(
        scheduler_lost,
        prior_available=True,
        current_available=False,
        unhealthy_edges=0,
    )[:2] == ("fail", "authoritative_scheduler_not_live")


def test_aseh_startup_fails_after_two_unavailable_authority_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.time()
    board, fixture_paths, current = _aseh_health_fixture(
        tmp_path,
        status="claimed",
        revision=2,
        event_cursor=11,
        ready=False,
        active=True,
        observed_at=now - 0.25,
        lane_mtime_ns=int((now - 0.25) * 1_000_000_000),
    )
    paths = {
        **fixture_paths,
        "status_receipt": tmp_path / "live-status.json",
    }
    unavailable = json.loads(json.dumps(current))
    unavailable["authority"] = {
        "available": False,
        "error": "published replica file identity is unsafe",
        "error_type": "OperatorError",
        "ready_count": 0,
        "active_count": 0,
        "blocked_count": 0,
        "terminal_count": 0,
        "event_cursor": 0,
        "task_statuses": {},
        "task_revisions": {},
    }
    samples = [unavailable, json.loads(json.dumps(unavailable))]
    recorded_failure: dict[str, object] = {}

    def fake_sample(*_args: object, **_kwargs: object) -> dict[str, object]:
        assert samples, "startup admission sampled past the stable pair"
        return samples.pop(0)

    monkeypatch.setattr(aseh_operator, "_status_sample", fake_sample)
    monkeypatch.setattr(aseh_operator, "STATUS_SAMPLE_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(
        aseh_operator,
        "_record_control_failure",
        lambda _paths, _failure, _event, **fields: (
            recorded_failure.update(fields)
        ),
    )

    with pytest.raises(
        aseh_operator.OperatorError,
        match="two consecutive authoritative status samples unavailable",
    ):
        aseh_operator._await_initial_health(
            board,
            paths,
            server=SimpleNamespace(),
            scheduler=SimpleNamespace(pid=4242, poll=lambda: None),
            launched_at=now - 0.5,
            failure={},
            failure_event=threading.Event(),
            shutdown_requested=threading.Event(),
            received_signal={},
        )

    assert recorded_failure == {
        "reason_code": "authoritative_status_unavailable_two_samples",
        "error_type": "ASEHHealthQueryFailure",
    }
    assert samples == []


def test_aseh_startup_honors_admitted_blocked_recovery_past_thirty_seconds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board = SimpleNamespace(
        payload={"watchdog_startup_grace_seconds": 300.0}
    )
    paths = {"status_receipt": tmp_path / "live-status.json"}
    samples = [
        {"observed_at": float(index), "authority": {"available": True}}
        for index in range(5)
    ]
    blocked = {
        "blocked": True,
        "stuck": True,
        "blocked_recovery_admitted": True,
        "healthy": False,
        "receipt_cid": "receipt:blocked",
    }
    healthy = {
        "blocked": False,
        "stuck": False,
        "blocked_recovery_admitted": False,
        "healthy": True,
        "receipt_cid": "receipt:healthy",
    }
    receipts = [dict(blocked), dict(blocked), dict(blocked), healthy]
    monotonic_values = iter((100.0, 100.0, 111.0, 122.0, 133.0))
    observed_monotonic: list[float] = []

    def fake_monotonic() -> float:
        value = next(monotonic_values)
        observed_monotonic.append(value)
        return value

    def fake_sample(*_args: object, **_kwargs: object) -> dict[str, object]:
        assert samples
        return samples.pop(0)

    def fake_health(*_args: object, **_kwargs: object) -> dict[str, object]:
        assert receipts
        return receipts.pop(0)

    monkeypatch.setattr(aseh_operator, "_status_sample", fake_sample)
    monkeypatch.setattr(aseh_operator, "_health_receipt", fake_health)
    monkeypatch.setattr(
        aseh_operator,
        "_authoritative_progress_between",
        lambda *_args: [],
    )
    monkeypatch.setattr(aseh_operator, "STATUS_SAMPLE_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(
        aseh_operator,
        "time",
        SimpleNamespace(monotonic=fake_monotonic),
    )

    admitted, last_progress_at = aseh_operator._await_initial_health(
        board,
        paths,
        server=SimpleNamespace(),
        scheduler=SimpleNamespace(pid=4242, poll=lambda: None),
        launched_at=90.0,
        failure={},
        failure_event=threading.Event(),
        shutdown_requested=threading.Event(),
        received_signal={},
    )

    assert admitted == healthy
    assert last_progress_at == 90.0
    assert observed_monotonic[-1] - observed_monotonic[0] > 30.0
    assert receipts == []
    assert samples == []


def test_aseh_startup_fails_when_blocked_recovery_admission_is_lost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board = SimpleNamespace(
        payload={"watchdog_startup_grace_seconds": 300.0}
    )
    paths = {"status_receipt": tmp_path / "live-status.json"}
    available = {"observed_at": 1.0, "authority": {"available": True}}
    samples = [dict(available), dict(available), dict(available)]
    receipts = [
        {
            "blocked": True,
            "stuck": True,
            "blocked_recovery_admitted": True,
            "healthy": False,
        },
        {
            "blocked": True,
            "stuck": True,
            "blocked_recovery_admitted": False,
            "owner_ready": False,
            "healthy": False,
        },
    ]
    recorded_failure: dict[str, object] = {}

    def fake_sample(*_args: object, **_kwargs: object) -> dict[str, object]:
        assert samples
        return samples.pop(0)

    def fake_health(*_args: object, **_kwargs: object) -> dict[str, object]:
        assert receipts
        return receipts.pop(0)

    monkeypatch.setattr(aseh_operator, "_status_sample", fake_sample)
    monkeypatch.setattr(aseh_operator, "_health_receipt", fake_health)
    monkeypatch.setattr(
        aseh_operator,
        "_authoritative_progress_between",
        lambda *_args: [],
    )
    monkeypatch.setattr(aseh_operator, "STATUS_SAMPLE_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(
        aseh_operator,
        "_record_control_failure",
        lambda _paths, _failure, _event, **fields: (
            recorded_failure.update(fields)
        ),
    )

    with pytest.raises(
        aseh_operator.OperatorError,
        match="foreground health admission failed closed",
    ):
        aseh_operator._await_initial_health(
            board,
            paths,
            server=SimpleNamespace(),
            scheduler=SimpleNamespace(pid=4242, poll=lambda: None),
            launched_at=time.time(),
            failure={},
            failure_event=threading.Event(),
            shutdown_requested=threading.Event(),
            received_signal={},
        )

    assert recorded_failure == {
        "reason_code": "authoritative_blocked_recovery_grace_exhausted",
        "error_type": "ASEHHealthGateFailure",
    }
    assert receipts == []
    assert samples == []


def test_aseh_health_accepts_fast_claim_before_first_sample(
    tmp_path: Path,
) -> None:
    now = time.time()
    board, paths, before = _aseh_health_fixture(
        tmp_path,
        status="claimed",
        revision=2,
        event_cursor=11,
        ready=False,
        active=True,
        observed_at=now - 0.25,
        lane_mtime_ns=int((now - 0.25) * 1_000_000_000),
    )
    _board, _paths, current = _aseh_health_fixture(
        tmp_path,
        status="claimed",
        revision=2,
        event_cursor=11,
        ready=False,
        active=True,
        observed_at=now,
        lane_mtime_ns=int(now * 1_000_000_000),
    )
    receipt = aseh_operator._health_receipt(
        board,
        paths,
        samples=(before, current),
        launched_at=now - 0.5,
        last_progress_at=now - 0.5,
        failure={},
        require_authoritative_progress=True,
    )

    assert "authoritative_event_since_bootstrap" in receipt["progress_evidence"]
    assert "authoritative_status_since_bootstrap" in receipt["progress_evidence"]
    assert "authoritative_revision_since_bootstrap" in receipt["progress_evidence"]
    assert receipt["authoritative_progress_admitted"] is True
    assert receipt["healthy"] is True


def test_aseh_stale_bootstrap_is_rejected_before_owner_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.time()
    _board, fixture_paths, _sample = _aseh_health_fixture(
        tmp_path,
        observed_at=now,
        lane_mtime_ns=int(now * 1_000_000_000),
    )
    database = tmp_path / "control.duckdb"
    database.touch()
    paths = {**fixture_paths, "database": database}
    board = SimpleNamespace(config_path=tmp_path / "board.json")
    config: dict[str, object] = {}
    current_population = {
        "source_head": "commit:current",
        "repository_tree_id": "tree:current",
        "plan_root_cid": "plan:current",
        "source_forest": {"forest_cid": "forest:current"},
        "source_identities": {"operator": "source:current"},
    }
    owner_built = False

    def build_owner(_board: object, _paths: object) -> object:
        nonlocal owner_built
        owner_built = True
        return object()

    monkeypatch.setattr(aseh_operator, "_load", lambda _path: (board, config))
    monkeypatch.setattr(aseh_operator, "_paths", lambda _board: paths)
    monkeypatch.setattr(
        aseh_operator,
        "_assert_clean_tree",
        lambda _board: (
            str(current_population["source_head"]),
            str(current_population["repository_tree_id"]),
        ),
    )
    monkeypatch.setattr(
        aseh_operator,
        "_candidate_authorization_witness",
        lambda **_kwargs: {"stable": "yes"},
    )
    monkeypatch.setattr(
        aseh_operator,
        "_git",
        lambda *_args: ("a" * 40) + " " + ("b" * 40),
    )
    monkeypatch.setattr(
        aseh_operator,
        "_population",
        lambda _board, _config: current_population,
    )
    monkeypatch.setattr(aseh_operator, "_build_server", build_owner)
    monkeypatch.setattr(
        configured_scheduler,
        "preflight_configured_board",
        lambda _board: {"valid": True},
    )

    with pytest.raises(
        aseh_operator.OperatorError,
        match="source forest|native-seal anchor",
    ):
        aseh_operator.run_supervisor(board.config_path, implement=True, duration=1)
    assert owner_built is False


def test_aseh_canonical_merge_suffix_admits_only_exact_two_parent_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
        checkout_repository_id,
    )
    from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import MergeRequest

    def git(*args: str) -> str:
        result = subprocess.run(
            ("git", *args), cwd=tmp_path, text=True, capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    git("init", "-b", "main")
    git("config", "user.email", "aseh-continuity@example.invalid")
    git("config", "user.name", "ASEH Continuity")
    (tmp_path / "base.txt").write_text("base\n", encoding="utf-8")
    git("add", "base.txt")
    git("commit", "-m", "sealed base")
    base = git("rev-parse", "HEAD")
    git("checkout", "-b", "candidate")
    (tmp_path / "output.txt").write_text("admitted\n", encoding="utf-8")
    git("add", "output.txt")
    git("commit", "-m", "ASEH-000 exact output")
    candidate = git("rev-parse", "HEAD")
    candidate_tree = git("rev-parse", "HEAD^{tree}")
    git("checkout", "main")
    git("merge", "--no-ff", "--no-edit", "candidate")
    integrated = git("rev-parse", "HEAD")

    monkeypatch.setattr(aseh_operator, "ROOT", tmp_path)
    task_cid = "task:aseh-continuity"
    board = SimpleNamespace(
        protected_paths=("protected.py",),
        merge_target_branch="main",
    )
    bootstrap = {
        "integrity": {"task_revisions": {"ASEH-000": 1}}
    }
    integrity = {
        "task_statuses": {"ASEH-000": "completed"},
        "task_revisions": {"ASEH-000": 3},
        "task_cids": {"ASEH-000": task_cid},
    }
    metadata = {
        "schema": "ipfs_accelerate_py/agent-supervisor/merge-candidate@3",
        "target_binding_schema": (
            "ipfs_accelerate_py/agent-supervisor/merge-target-binding@1"
        ),
        "target_repository_id": checkout_repository_id(tmp_path),
        "target_branch": "main",
        "candidate_tree": candidate_tree,
        "repository_tree_id": f"git-tree:{candidate_tree}",
        "baseline_ref": base,
        "changed_submodule_paths": [],
        "completion_task_cids": {"ASEH-000": task_cid},
        "task": {"outputs": ["output.txt"]},
        "validation_proof": {
            "passed": True,
            "target_commit": candidate,
            "target_tree": candidate_tree,
        },
    }
    request = MergeRequest(
        request_id="request:aseh-continuity",
        branch_name="candidate",
        task_id="ASEH-000",
        priority="P1",
        lane_id="lane-0",
        enqueued_at=1.0,
        metadata=metadata,
        commit_sha=candidate,
        canonical_task_id=task_cid,
        canonical_task_key=task_cid,
        status="completed",
    )
    proof = aseh_operator._admit_canonical_merge_suffix(
        board,
        base_head=base,
        target_head=integrated,
        bootstrap=bootstrap,
        integrity=integrity,
        task_outputs={"ASEH-000": ("output.txt",)},
        completed_requests=(request,),
    )
    assert proof["integrations"][0]["candidate_commit"] == candidate
    assert proof["integrations"][0]["changed_paths"] == ["output.txt"]

    monkeypatch.setattr(
        aseh_operator, "REPAIR_FOLLOWUP_TRANSITION_FIRST_PARENT", base
    )
    monkeypatch.setattr(
        aseh_operator, "REPAIR_FOLLOWUP_TRANSITION_BASE_HEAD", integrated
    )
    monkeypatch.setattr(
        aseh_operator, "REPAIR_FOLLOWUP_TRANSITION_CANDIDATE", candidate
    )
    monkeypatch.setattr(
        aseh_operator, "REPAIR_FOLLOWUP_TRANSITION_TASK_ALIAS", "ASEH-000"
    )
    integrity["task_statuses"]["ASEH-000"] = "blocked"
    nonterminal = aseh_operator._admit_canonical_merge_suffix(
        board,
        base_head=base,
        target_head=integrated,
        bootstrap=bootstrap,
        integrity=integrity,
        task_outputs={"ASEH-000": ("output.txt",)},
        completed_requests=(request,),
        admission_mode="followup_repair_base",
    )
    assert nonterminal["schema"].endswith(
        "/aseh-nonterminal-integration-base@1"
    )
    assert nonterminal["task_completion_admitted"] is False
    assert nonterminal["completion_authoritative"] is False
    assert nonterminal["integrations"][0]["observed_task_status"] == "blocked"

    for restart_status in ("ready", "claimed", "in_progress", "running"):
        integrity["task_statuses"]["ASEH-000"] = restart_status
        restart_witness = aseh_operator._admit_canonical_merge_suffix(
            board,
            base_head=base,
            target_head=integrated,
            bootstrap=bootstrap,
            integrity=integrity,
            task_outputs={"ASEH-000": ("output.txt",)},
            completed_requests=(request,),
            admission_mode="followup_repair_base",
        )
        assert restart_witness["task_completion_admitted"] is False
        assert restart_witness["integrations"][0][
            "observed_task_status"
        ] == restart_status
    integrity["task_statuses"]["ASEH-000"] = "completed"

    metadata["baseline_ref"] = "HEAD"
    with pytest.raises(aseh_operator.OperatorError, match="exact commit"):
        aseh_operator._admit_canonical_merge_suffix(
            board,
            base_head=base,
            target_head=integrated,
            bootstrap=bootstrap,
            integrity=integrity,
            task_outputs={"ASEH-000": ("output.txt",)},
            completed_requests=(request,),
        )
    metadata["baseline_ref"] = base

    git("checkout", "-b", "omitted-candidate-output", base)
    git("merge", "--no-ff", "-s", "ours", "--no-edit", "candidate")
    omitted = git("rev-parse", "HEAD")
    with pytest.raises(aseh_operator.OperatorError, match="output differs"):
        aseh_operator._admit_canonical_merge_suffix(
            board,
            base_head=base,
            target_head=omitted,
            bootstrap=bootstrap,
            integrity=integrity,
            task_outputs={"ASEH-000": ("output.txt",)},
            completed_requests=(request,),
        )
    git("checkout", "main")

    metadata["candidate_tree"] = "0" * 40
    with pytest.raises(aseh_operator.OperatorError, match="validation binding"):
        aseh_operator._admit_canonical_merge_suffix(
            board,
            base_head=base,
            target_head=integrated,
            bootstrap=bootstrap,
            integrity=integrity,
            task_outputs={"ASEH-000": ("output.txt",)},
            completed_requests=(request,),
        )
    metadata["candidate_tree"] = candidate_tree

    protected_board = SimpleNamespace(
        protected_paths=("output.txt",),
        merge_target_branch="main",
    )
    with pytest.raises(aseh_operator.OperatorError, match="protected-path"):
        aseh_operator._admit_canonical_merge_suffix(
            protected_board,
            base_head=base,
            target_head=integrated,
            bootstrap=bootstrap,
            integrity=integrity,
            task_outputs={"ASEH-000": ("output.txt",)},
            completed_requests=(request,),
        )

    (tmp_path / "arbitrary.txt").write_text("escape\n", encoding="utf-8")
    git("add", "arbitrary.txt")
    git("commit", "-m", "arbitrary child")
    arbitrary = git("rev-parse", "HEAD")
    with pytest.raises(aseh_operator.OperatorError, match="non-canonical"):
        aseh_operator._admit_canonical_merge_suffix(
            board,
            base_head=base,
            target_head=arbitrary,
            bootstrap=bootstrap,
            integrity=integrity,
            task_outputs={"ASEH-000": ("output.txt",)},
            completed_requests=(request,),
        )


def test_aseh_repair_transition_receipt_is_closed_and_non_mutating() -> None:
    receipt = {
        "schema": aseh_operator.REPAIR_TRANSITION_SCHEMA,
        "task_id": aseh_operator.REPAIR_TRANSITION_TASK_ID,
        "stable_identity": "agent-supervisor-efficiency/ASEH-BOOTSTRAP-002",
        "program_id": aseh_operator.PROGRAM,
        "bootstrap_receipt_id": "sha256:" + ("1" * 64),
        "plan_root_cid": "plan:sealed",
        "repository_tree_id": "tree:sealed",
        "base_head": aseh_operator.REPAIR_TRANSITION_BASE_HEAD,
        "base_tree": "2" * 40,
        "repair_head": "3" * 40,
        "repair_tree": "4" * 40,
        "changed_paths": list(aseh_operator.REPAIR_TRANSITION_CHANGED_PATHS),
        "patch_digest": "sha256:" + ("5" * 64),
        "dependencies": ["ASEH-BOOTSTRAP-001", "ASEH-000"],
        "owning_repository": "ipfs_accelerate_py",
        "risk_class": "R4_SECURITY_OR_PROTOCOL_SENSITIVE",
        "authority_requirement": "explicit bootstrap repair authority",
        "validation_results": [],
        "terminal_success_criteria": "exact repair",
        "terminal_non_success_criteria": "all drift rejected",
        "semantic_corpus_changed": False,
        "database_mutated": False,
        "authorized_at": 1.0,
    }
    receipt["receipt_cid"] = aseh_operator._identity(receipt)
    assert aseh_operator._repair_transition_receipt_id(receipt) == (
        receipt["receipt_cid"]
    )
    receipt["database_mutated"] = True
    with pytest.raises(aseh_operator.OperatorError, match="schema"):
        aseh_operator._repair_transition_receipt_id(receipt)


def test_aseh_repair_transition_followup_receipt_is_closed_and_chained() -> None:
    witness = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "aseh-nonterminal-integration-witness@1"
        ),
        "base_head": "1" * 40,
        "base_tree": "2" * 40,
        "target_head": "3" * 40,
        "target_tree": "4" * 40,
        "request_id": "request:aseh-followup",
        "merge_request_cid": "sha256:" + ("5" * 64),
        "task_id": "ASEH-001",
        "task_cid": "task:aseh-followup",
        "candidate_commit": "6" * 40,
        "candidate_tree": "7" * 40,
        "integration_commit": "3" * 40,
        "integration_tree": "4" * 40,
        "baseline_ref": "8" * 40,
        "changed_paths": ["sealed.json"],
        "validation_proof_cid": "sha256:" + ("9" * 64),
        "completion_authoritative": False,
        "task_completion_admitted": False,
    }
    witness["receipt_cid"] = aseh_operator._identity(witness)
    receipt = {
        "schema": aseh_operator.REPAIR_FOLLOWUP_TRANSITION_SCHEMA,
        "task_id": aseh_operator.REPAIR_TRANSITION_TASK_ID,
        "stable_identity": (
            f"{aseh_operator.PROGRAM}/"
            f"{aseh_operator.REPAIR_TRANSITION_TASK_ID}@ASEH-PLAN-R2"
        ),
        "program_id": aseh_operator.PROGRAM,
        "transition_revision": 2,
        "bootstrap_receipt_id": "sha256:" + ("a" * 64),
        "previous_receipt_cid": "sha256:" + ("b" * 64),
        "base_integration_witness": witness,
        "authorization_task_observation": {
            "task_id": "ASEH-001",
            "task_cid": "task:aseh-followup",
            "status": "blocked",
            "revision": 6,
            "completion_authoritative": False,
            "observed_at": 1.0,
        },
        "plan_root_cid": "plan:sealed",
        "repository_tree_id": "tree:sealed",
        "base_head": "3" * 40,
        "base_tree": "4" * 40,
        "repair_head": "c" * 40,
        "repair_tree": "d" * 40,
        "changed_paths": list(
            aseh_operator.REPAIR_FOLLOWUP_TRANSITION_CHANGED_PATHS
        ),
        "patch_digest": "sha256:" + ("e" * 64),
        "dependencies": [
            "ASEH-BOOTSTRAP-002@ASEH-PLAN-R1",
            "ASEH-001",
        ],
        "owning_repository": "ipfs_accelerate_py",
        "risk_class": "R4_SECURITY_OR_PROTOCOL_SENSITIVE",
        "authority_requirement": "explicit follow-up repair authority",
        "validation_results": [],
        "terminal_success_criteria": "exact automatic recovery",
        "terminal_non_success_criteria": "all drift rejected",
        "semantic_corpus_changed": False,
        "database_mutated": False,
        "authorized_at": 1.0,
    }
    receipt["receipt_cid"] = aseh_operator._identity(receipt)
    assert aseh_operator._repair_followup_transition_receipt_id(receipt) == (
        receipt["receipt_cid"]
    )

    receipt["transition_revision"] = 3
    receipt["receipt_cid"] = aseh_operator._identity(
        {key: value for key, value in receipt.items() if key != "receipt_cid"}
    )
    with pytest.raises(aseh_operator.OperatorError, match="schema"):
        aseh_operator._repair_followup_transition_receipt_id(receipt)


def test_aseh_repair_clean_launch_transition_is_closed_and_chained() -> None:
    receipt = {
        "schema": aseh_operator.REPAIR_CLEAN_LAUNCH_TRANSITION_SCHEMA,
        "task_id": aseh_operator.REPAIR_TRANSITION_TASK_ID,
        "stable_identity": (
            f"{aseh_operator.PROGRAM}/"
            f"{aseh_operator.REPAIR_TRANSITION_TASK_ID}@ASEH-PLAN-R3"
        ),
        "program_id": aseh_operator.PROGRAM,
        "transition_revision": 3,
        "bootstrap_receipt_id": "sha256:" + ("a" * 64),
        "previous_receipt_cid": "sha256:" + ("b" * 64),
        "plan_root_cid": "plan:sealed",
        "repository_tree_id": "tree:sealed",
        "base_head": "1" * 40,
        "base_tree": "2" * 40,
        "repair_head": "3" * 40,
        "repair_tree": "4" * 40,
        "changed_paths": list(
            aseh_operator.REPAIR_CLEAN_LAUNCH_TRANSITION_CHANGED_PATHS
        ),
        "patch_digest": "sha256:" + ("5" * 64),
        "dependencies": ["ASEH-BOOTSTRAP-002@ASEH-PLAN-R2"],
        "owning_repository": "ipfs_accelerate_py",
        "risk_class": "R4_SECURITY_OR_PROTOCOL_SENSITIVE",
        "authority_requirement": "explicit clean-launch repair authority",
        "validation_results": [],
        "terminal_success_criteria": "validation leaves checkout clean",
        "terminal_non_success_criteria": "all drift rejected",
        "semantic_corpus_changed": False,
        "database_mutated": False,
        "authorized_at": 1.0,
    }
    receipt["receipt_cid"] = aseh_operator._identity(receipt)
    assert aseh_operator._repair_clean_launch_transition_receipt_id(
        receipt
    ) == receipt["receipt_cid"]

    receipt["previous_receipt_cid"] = "sha256:" + ("c" * 64)
    with pytest.raises(aseh_operator.OperatorError, match="CID"):
        aseh_operator._repair_clean_launch_transition_receipt_id(receipt)


def test_aseh_repair_runtime_hardening_transition_is_closed_and_chained(
) -> None:
    receipt = {
        "schema": aseh_operator.REPAIR_RUNTIME_HARDENING_TRANSITION_SCHEMA,
        "task_id": aseh_operator.REPAIR_TRANSITION_TASK_ID,
        "stable_identity": (
            f"{aseh_operator.PROGRAM}/"
            f"{aseh_operator.REPAIR_TRANSITION_TASK_ID}@ASEH-PLAN-R4"
        ),
        "program_id": aseh_operator.PROGRAM,
        "transition_revision": 4,
        "bootstrap_receipt_id": "sha256:" + ("a" * 64),
        "previous_receipt_cid": "sha256:" + ("b" * 64),
        "plan_root_cid": "plan:sealed",
        "repository_tree_id": "tree:sealed",
        "base_head": "1" * 40,
        "base_tree": "2" * 40,
        "repair_head": "3" * 40,
        "repair_tree": "4" * 40,
        "changed_paths": list(
            aseh_operator.REPAIR_RUNTIME_HARDENING_TRANSITION_CHANGED_PATHS
        ),
        "patch_digest": "sha256:" + ("5" * 64),
        "dependencies": ["ASEH-BOOTSTRAP-002@ASEH-PLAN-R3"],
        "owning_repository": "ipfs_accelerate_py",
        "risk_class": "R4_SECURITY_OR_PROTOCOL_SENSITIVE",
        "authority_requirement": "explicit runtime-hardening authority",
        "validation_results": [],
        "terminal_success_criteria": "automatic bounded recovery",
        "terminal_non_success_criteria": "all drift rejected",
        "semantic_corpus_changed": False,
        "database_mutated": False,
        "authorized_at": 1.0,
    }
    receipt["receipt_cid"] = aseh_operator._identity(receipt)
    assert (
        aseh_operator._repair_runtime_hardening_transition_receipt_id(
            receipt
        )
        == receipt["receipt_cid"]
    )

    receipt["transition_revision"] = 3
    receipt["receipt_cid"] = aseh_operator._identity(
        {
            key: value
            for key, value in receipt.items()
            if key != "receipt_cid"
        }
    )
    with pytest.raises(aseh_operator.OperatorError, match="schema"):
        aseh_operator._repair_runtime_hardening_transition_receipt_id(
            receipt
        )


def test_aseh_repair_quack_recovery_transition_is_closed_and_chained(
) -> None:
    receipt = {
        "schema": aseh_operator.REPAIR_QUACK_RECOVERY_TRANSITION_SCHEMA,
        "task_id": aseh_operator.REPAIR_TRANSITION_TASK_ID,
        "stable_identity": (
            f"{aseh_operator.PROGRAM}/"
            f"{aseh_operator.REPAIR_TRANSITION_TASK_ID}@ASEH-PLAN-R5"
        ),
        "program_id": aseh_operator.PROGRAM,
        "transition_revision": 5,
        "bootstrap_receipt_id": "sha256:" + ("a" * 64),
        "previous_receipt_cid": "sha256:" + ("b" * 64),
        "plan_root_cid": "plan:sealed",
        "repository_tree_id": "tree:sealed",
        "base_head": "1" * 40,
        "base_tree": "2" * 40,
        "repair_head": "3" * 40,
        "repair_tree": "4" * 40,
        "changed_paths": list(
            aseh_operator.REPAIR_QUACK_RECOVERY_TRANSITION_CHANGED_PATHS
        ),
        "patch_digest": "sha256:" + ("5" * 64),
        "dependencies": ["ASEH-BOOTSTRAP-002@ASEH-PLAN-R4"],
        "owning_repository": "ipfs_accelerate_py",
        "risk_class": "R4_SECURITY_OR_PROTOCOL_SENSITIVE",
        "authority_requirement": "explicit Quack-recovery authority",
        "validation_results": [],
        "terminal_success_criteria": "exact automatic recovery",
        "terminal_non_success_criteria": "all drift rejected",
        "semantic_corpus_changed": False,
        "database_mutated": False,
        "authorized_at": 1.0,
    }
    receipt["receipt_cid"] = aseh_operator._identity(receipt)
    assert (
        aseh_operator._repair_quack_recovery_transition_receipt_id(
            receipt
        )
        == receipt["receipt_cid"]
    )

    receipt["transition_revision"] = 4
    receipt["receipt_cid"] = aseh_operator._identity(
        {
            key: value
            for key, value in receipt.items()
            if key != "receipt_cid"
        }
    )
    with pytest.raises(aseh_operator.OperatorError, match="schema"):
        aseh_operator._repair_quack_recovery_transition_receipt_id(receipt)


def test_aseh_repair_parallel_blocked_startup_transition_is_closed_and_chained(
) -> None:
    receipt = {
        "schema": (
            aseh_operator.REPAIR_PARALLEL_BLOCKED_STARTUP_TRANSITION_SCHEMA
        ),
        "task_id": aseh_operator.REPAIR_TRANSITION_TASK_ID,
        "stable_identity": (
            f"{aseh_operator.PROGRAM}/"
            f"{aseh_operator.REPAIR_TRANSITION_TASK_ID}@ASEH-PLAN-R6"
        ),
        "program_id": aseh_operator.PROGRAM,
        "transition_revision": 6,
        "bootstrap_receipt_id": "sha256:" + ("a" * 64),
        "previous_receipt_cid": "sha256:" + ("b" * 64),
        "plan_root_cid": "plan:sealed",
        "repository_tree_id": "tree:sealed",
        "base_head": "1" * 40,
        "base_tree": "2" * 40,
        "repair_head": "3" * 40,
        "repair_tree": "4" * 40,
        "changed_paths": list(
            aseh_operator.REPAIR_PARALLEL_BLOCKED_STARTUP_TRANSITION_CHANGED_PATHS
        ),
        "patch_digest": "sha256:" + ("5" * 64),
        "dependencies": ["ASEH-BOOTSTRAP-002@ASEH-PLAN-R5"],
        "owning_repository": "ipfs_accelerate_py",
        "risk_class": "R4_SECURITY_OR_PROTOCOL_SENSITIVE",
        "authority_requirement": "explicit parallel startup authority",
        "validation_results": [],
        "terminal_success_criteria": "bounded automatic recovery",
        "terminal_non_success_criteria": "all drift rejected",
        "semantic_corpus_changed": False,
        "database_mutated": False,
        "authorized_at": 1.0,
    }
    receipt["receipt_cid"] = aseh_operator._identity(receipt)
    assert (
        aseh_operator._repair_parallel_blocked_startup_transition_receipt_id(
            receipt
        )
        == receipt["receipt_cid"]
    )

    receipt["transition_revision"] = 5
    receipt["receipt_cid"] = aseh_operator._identity(
        {
            key: value
            for key, value in receipt.items()
            if key != "receipt_cid"
        }
    )
    with pytest.raises(aseh_operator.OperatorError, match="schema"):
        aseh_operator._repair_parallel_blocked_startup_transition_receipt_id(
            receipt
        )


def test_aseh_repair_quack_publication_contention_transition_is_closed_and_chained(
) -> None:
    receipt = {
        "schema": (
            aseh_operator.REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_SCHEMA
        ),
        "task_id": aseh_operator.REPAIR_TRANSITION_TASK_ID,
        "stable_identity": (
            f"{aseh_operator.PROGRAM}/"
            f"{aseh_operator.REPAIR_TRANSITION_TASK_ID}@ASEH-PLAN-R7"
        ),
        "program_id": aseh_operator.PROGRAM,
        "transition_revision": 7,
        "bootstrap_receipt_id": "sha256:" + ("a" * 64),
        "previous_receipt_cid": "sha256:" + ("b" * 64),
        "plan_root_cid": "plan:sealed",
        "repository_tree_id": "tree:sealed",
        "base_head": "1" * 40,
        "base_tree": "2" * 40,
        "repair_head": "3" * 40,
        "repair_tree": "4" * 40,
        "changed_paths": list(
            aseh_operator.REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_CHANGED_PATHS
        ),
        "patch_digest": "sha256:" + ("5" * 64),
        "dependencies": ["ASEH-BOOTSTRAP-002@ASEH-PLAN-R6"],
        "owning_repository": "ipfs_accelerate_py",
        "risk_class": "R4_SECURITY_OR_PROTOCOL_SENSITIVE",
        "authority_requirement": (
            aseh_operator.REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_AUTHORITY
        ),
        "validation_results": [],
        "terminal_success_criteria": "bounded automatic recovery",
        "terminal_non_success_criteria": "all drift rejected",
        "semantic_corpus_changed": False,
        "database_mutated": False,
        "authorized_at": 1.0,
    }
    receipt["receipt_cid"] = aseh_operator._identity(receipt)
    assert (
        aseh_operator._repair_quack_publication_contention_transition_receipt_id(
            receipt
        )
        == receipt["receipt_cid"]
    )

    receipt["previous_receipt_cid"] = "sha256:" + ("c" * 64)
    with pytest.raises(aseh_operator.OperatorError, match="CID"):
        aseh_operator._repair_quack_publication_contention_transition_receipt_id(
            receipt
        )


def test_aseh_repair_quack_recovery_replay_transition_is_closed_and_chained(
) -> None:
    receipt = {
        "schema": aseh_operator.REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_SCHEMA,
        "task_id": aseh_operator.REPAIR_TRANSITION_TASK_ID,
        "stable_identity": (
            f"{aseh_operator.PROGRAM}/"
            f"{aseh_operator.REPAIR_TRANSITION_TASK_ID}@ASEH-PLAN-R8"
        ),
        "program_id": aseh_operator.PROGRAM,
        "transition_revision": 8,
        "bootstrap_receipt_id": "sha256:" + ("a" * 64),
        "previous_receipt_cid": "sha256:" + ("b" * 64),
        "plan_root_cid": "plan:sealed",
        "repository_tree_id": "tree:sealed",
        "base_head": "1" * 40,
        "base_tree": "2" * 40,
        "repair_head": "3" * 40,
        "repair_tree": "4" * 40,
        "changed_paths": list(
            aseh_operator.REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_CHANGED_PATHS
        ),
        "patch_digest": "sha256:" + ("5" * 64),
        "dependencies": ["ASEH-BOOTSTRAP-002@ASEH-PLAN-R7"],
        "owning_repository": "ipfs_accelerate_py",
        "risk_class": "R4_SECURITY_OR_PROTOCOL_SENSITIVE",
        "authority_requirement": (
            aseh_operator.REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_AUTHORITY
        ),
        "validation_results": [],
        "terminal_success_criteria": "idempotent exact replay",
        "terminal_non_success_criteria": "all drift rejected",
        "semantic_corpus_changed": False,
        "database_mutated": False,
        "authorized_at": 1.0,
    }
    receipt["receipt_cid"] = aseh_operator._identity(receipt)
    assert (
        aseh_operator._repair_quack_recovery_replay_transition_receipt_id(
            receipt
        )
        == receipt["receipt_cid"]
    )

    receipt["transition_revision"] = 7
    receipt["receipt_cid"] = aseh_operator._identity(
        {
            key: value
            for key, value in receipt.items()
            if key != "receipt_cid"
        }
    )
    with pytest.raises(aseh_operator.OperatorError, match="schema"):
        aseh_operator._repair_quack_recovery_replay_transition_receipt_id(
            receipt
        )


def test_aseh_repair_control_receipt_lifecycle_transition_is_closed_and_chained(
) -> None:
    receipt = {
        "schema": (
            aseh_operator.REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_SCHEMA
        ),
        "task_id": aseh_operator.REPAIR_TRANSITION_TASK_ID,
        "stable_identity": (
            f"{aseh_operator.PROGRAM}/"
            f"{aseh_operator.REPAIR_TRANSITION_TASK_ID}@ASEH-PLAN-R9"
        ),
        "program_id": aseh_operator.PROGRAM,
        "transition_revision": 9,
        "bootstrap_receipt_id": "sha256:" + ("a" * 64),
        "previous_receipt_cid": "sha256:" + ("b" * 64),
        "plan_root_cid": "plan:sealed",
        "repository_tree_id": "tree:sealed",
        "base_head": "1" * 40,
        "base_tree": "2" * 40,
        "repair_head": "3" * 40,
        "repair_tree": "4" * 40,
        "changed_paths": list(
            aseh_operator.REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_CHANGED_PATHS
        ),
        "patch_digest": "sha256:" + ("5" * 64),
        "dependencies": ["ASEH-BOOTSTRAP-002@ASEH-PLAN-R8"],
        "owning_repository": "ipfs_accelerate_py",
        "risk_class": "R4_SECURITY_OR_PROTOCOL_SENSITIVE",
        "authority_requirement": (
            aseh_operator.REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_AUTHORITY
        ),
        "validation_results": [],
        "terminal_success_criteria": "exact lifecycle replay",
        "terminal_non_success_criteria": "all drift rejected",
        "semantic_corpus_changed": False,
        "database_mutated": False,
        "authorized_at": 1.0,
    }
    receipt["receipt_cid"] = aseh_operator._identity(receipt)
    assert (
        aseh_operator._repair_control_receipt_lifecycle_transition_receipt_id(
            receipt
        )
        == receipt["receipt_cid"]
    )

    receipt["unknown"] = True
    receipt["receipt_cid"] = aseh_operator._identity(
        {
            key: value
            for key, value in receipt.items()
            if key != "receipt_cid"
        }
    )
    with pytest.raises(aseh_operator.OperatorError, match="schema"):
        aseh_operator._repair_control_receipt_lifecycle_transition_receipt_id(
            receipt
        )


def test_aseh_repair_provider_lease_ownership_transition_is_closed_and_chained(
) -> None:
    receipt = {
        "schema": (
            aseh_operator.REPAIR_PROVIDER_LEASE_OWNERSHIP_TRANSITION_SCHEMA
        ),
        "task_id": aseh_operator.REPAIR_TRANSITION_TASK_ID,
        "stable_identity": (
            f"{aseh_operator.PROGRAM}/"
            f"{aseh_operator.REPAIR_TRANSITION_TASK_ID}@ASEH-PLAN-R10"
        ),
        "program_id": aseh_operator.PROGRAM,
        "transition_revision": 10,
        "bootstrap_receipt_id": "sha256:" + ("a" * 64),
        "previous_receipt_cid": "sha256:" + ("b" * 64),
        "plan_root_cid": "plan:sealed",
        "repository_tree_id": "tree:sealed",
        "base_head": "1" * 40,
        "base_tree": "2" * 40,
        "repair_head": "3" * 40,
        "repair_tree": "4" * 40,
        "changed_paths": list(
            aseh_operator.REPAIR_PROVIDER_LEASE_OWNERSHIP_TRANSITION_CHANGED_PATHS
        ),
        "patch_digest": "sha256:" + ("5" * 64),
        "dependencies": ["ASEH-BOOTSTRAP-002@ASEH-PLAN-R9"],
        "owning_repository": "ipfs_accelerate_py",
        "risk_class": "R4_SECURITY_OR_PROTOCOL_SENSITIVE",
        "authority_requirement": (
            aseh_operator.REPAIR_PROVIDER_LEASE_OWNERSHIP_TRANSITION_AUTHORITY
        ),
        "validation_results": [],
        "terminal_success_criteria": "exact provider lease ownership",
        "terminal_non_success_criteria": "all drift rejected",
        "semantic_corpus_changed": False,
        "database_mutated": False,
        "authorized_at": 1.0,
    }
    receipt["receipt_cid"] = aseh_operator._identity(receipt)
    assert (
        aseh_operator._repair_provider_lease_ownership_transition_receipt_id(
            receipt
        )
        == receipt["receipt_cid"]
    )

    receipt["transition_revision"] = 9
    receipt["receipt_cid"] = aseh_operator._identity(
        {
            key: value
            for key, value in receipt.items()
            if key != "receipt_cid"
        }
    )
    with pytest.raises(aseh_operator.OperatorError, match="schema"):
        aseh_operator._repair_provider_lease_ownership_transition_receipt_id(
            receipt
        )


def test_aseh_repair_provider_cleanup_fence_transition_is_closed_and_chained(
) -> None:
    repair_head = "3" * 40
    repair_tree = "4" * 40
    candidate_witness = {
        "head": repair_head,
        "tree": repair_tree,
        "branch_ref": "refs/heads/aseh-r11-fixture",
        "index_entries_digest": "sha256:" + ("8" * 64),
        "index_flags_digest": "sha256:" + ("9" * 64),
        "status_digest": aseh_operator._identity(b""),
        "head_reflog_digest": "sha256:" + ("a" * 64),
        "branch_reflog_digest": "sha256:" + ("b" * 64),
    }
    known_baseline = {
        "schema": (
            aseh_operator.REPAIR_PROVIDER_CLEANUP_FENCE_KNOWN_BASELINE_SCHEMA
        ),
        "source_head": (
            aseh_operator.REPAIR_PROVIDER_CLEANUP_FENCE_TRANSITION_BASE_HEAD
        ),
        "source_tree": (
            aseh_operator.REPAIR_PROVIDER_CLEANUP_FENCE_KNOWN_BASELINE_TREE
        ),
        "command": list(
            aseh_operator.REPAIR_PROVIDER_CLEANUP_FENCE_KNOWN_BASELINE_COMMAND
        ),
        "expected_returncode": 1,
        "observed_returncode": 1,
        "first_failing_node": (
            aseh_operator.REPAIR_PROVIDER_CLEANUP_FENCE_KNOWN_BASELINE_NODE
        ),
        "normalized_failure_class": (
            "prompt_v3_branch_specific_convergence_failure"
        ),
        "environment_identity": (
            aseh_operator._r11_validation_environment_identity(
                aseh_operator._r11_validation_environment(
                    Path("/sealed-checkout")
                ),
                checkout=Path("/sealed-checkout"),
            )
        ),
        "stdout_digest": "sha256:" + ("6" * 64),
        "stderr_digest": "sha256:" + ("7" * 64),
        "authoritative_for_r11": False,
        "blocks_broad_quality_or_promotion_claim": True,
        "corpus_changed": False,
        "observed_at": 1.0,
    }
    known_baseline["receipt_cid"] = aseh_operator._identity(known_baseline)
    receipt = {
        "schema": (
            aseh_operator.REPAIR_PROVIDER_CLEANUP_FENCE_TRANSITION_SCHEMA
        ),
        "task_id": aseh_operator.REPAIR_TRANSITION_TASK_ID,
        "stable_identity": (
            f"{aseh_operator.PROGRAM}/"
            f"{aseh_operator.REPAIR_TRANSITION_TASK_ID}@ASEH-PLAN-R11"
        ),
        "program_id": aseh_operator.PROGRAM,
        "transition_revision": 11,
        "bootstrap_receipt_id": "sha256:" + ("a" * 64),
        "previous_receipt_cid": "sha256:" + ("b" * 64),
        "plan_root_cid": "plan:sealed",
        "repository_tree_id": "tree:sealed",
        "base_head": "1" * 40,
        "base_tree": "2" * 40,
        "repair_head": repair_head,
        "repair_tree": repair_tree,
        "changed_paths": list(
            aseh_operator.REPAIR_PROVIDER_CLEANUP_FENCE_TRANSITION_CHANGED_PATHS
        ),
        "patch_digest": "sha256:" + ("5" * 64),
        "dependencies": ["ASEH-BOOTSTRAP-002@ASEH-PLAN-R10"],
        "owning_repository": "ipfs_accelerate_py",
        "risk_class": "R4_SECURITY_OR_PROTOCOL_SENSITIVE",
        "authority_requirement": (
            aseh_operator.REPAIR_PROVIDER_CLEANUP_FENCE_TRANSITION_AUTHORITY
        ),
        "validation_results": [],
        "known_baseline_failures": [known_baseline],
        "native_dependency_authorization": (
            aseh_operator._r11_native_dependency_authorization(
                candidate_head=repair_head,
                candidate_tree=repair_tree,
                candidate_authorization_witness=candidate_witness,
            )
        ),
        "terminal_success_criteria": (
            aseh_operator.REPAIR_PROVIDER_CLEANUP_FENCE_TRANSITION_SUCCESS
        ),
        "terminal_non_success_criteria": (
            aseh_operator.REPAIR_PROVIDER_CLEANUP_FENCE_TRANSITION_NON_SUCCESS
        ),
        "semantic_corpus_changed": False,
        "database_mutated": False,
        "authorized_at": 1.0,
    }
    receipt["receipt_cid"] = aseh_operator._identity(receipt)
    assert (
        aseh_operator._repair_provider_cleanup_fence_transition_receipt_id(
            receipt
        )
        == receipt["receipt_cid"]
    )

    receipt["transition_revision"] = 10
    receipt["receipt_cid"] = aseh_operator._identity(
        {
            key: value
            for key, value in receipt.items()
            if key != "receipt_cid"
        }
    )
    with pytest.raises(aseh_operator.OperatorError, match="schema"):
        aseh_operator._repair_provider_cleanup_fence_transition_receipt_id(
            receipt
        )


def test_aseh_repair_sealed_owner_identity_transition_is_closed_and_chained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repair_head = "3" * 40
    repair_tree = "4" * 40
    witness = {
        "head": repair_head,
        "tree": repair_tree,
        "branch_ref": "refs/heads/aseh-r12-fixture",
        "index_entries_digest": "sha256:" + ("8" * 64),
        "index_flags_digest": "sha256:" + ("9" * 64),
        "status_digest": aseh_operator._identity(b""),
        "head_reflog_digest": "sha256:" + ("a" * 64),
        "branch_reflog_digest": "sha256:" + ("b" * 64),
    }
    receipt = {
        "schema": (
            aseh_operator.REPAIR_SEALED_OWNER_IDENTITY_TRANSITION_SCHEMA
        ),
        "task_id": aseh_operator.REPAIR_TRANSITION_TASK_ID,
        "stable_identity": (
            f"{aseh_operator.PROGRAM}/"
            f"{aseh_operator.REPAIR_TRANSITION_TASK_ID}@ASEH-PLAN-R12"
        ),
        "program_id": aseh_operator.PROGRAM,
        "transition_revision": 12,
        "bootstrap_receipt_id": "sha256:" + ("1" * 64),
        "previous_receipt_cid": "sha256:" + ("2" * 64),
        "plan_root_cid": "plan:sealed",
        "repository_tree_id": "tree:sealed",
        "base_head": (
            aseh_operator.REPAIR_SEALED_OWNER_IDENTITY_TRANSITION_BASE_HEAD
        ),
        "base_tree": "5" * 40,
        "repair_head": repair_head,
        "repair_tree": repair_tree,
        "changed_paths": list(
            aseh_operator
            .REPAIR_SEALED_OWNER_IDENTITY_TRANSITION_CHANGED_PATHS
        ),
        "patch_digest": "sha256:" + ("6" * 64),
        "dependencies": ["ASEH-BOOTSTRAP-002@ASEH-PLAN-R11"],
        "owning_repository": "ipfs_accelerate_py",
        "risk_class": "R4_SECURITY_OR_PROTOCOL_SENSITIVE",
        "authority_requirement": (
            aseh_operator.REPAIR_SEALED_OWNER_IDENTITY_TRANSITION_AUTHORITY
        ),
        "validation_results": [],
        "candidate_authorization_witness": witness,
        "terminal_success_criteria": (
            aseh_operator.REPAIR_SEALED_OWNER_IDENTITY_TRANSITION_SUCCESS
        ),
        "terminal_non_success_criteria": (
            aseh_operator.REPAIR_SEALED_OWNER_IDENTITY_TRANSITION_NON_SUCCESS
        ),
        "semantic_corpus_changed": False,
        "database_mutated": False,
        "authorized_at": 1.0,
    }
    receipt["receipt_cid"] = aseh_operator._identity(receipt)
    assert (
        aseh_operator
        ._repair_sealed_owner_identity_transition_receipt_id(receipt)
        == receipt["receipt_cid"]
    )

    monkeypatch.setattr(
        aseh_operator,
        "_repair_provider_cleanup_fence_transition_receipt_id",
        lambda _payload: "sha256:" + ("2" * 64),
    )
    bootstrap = {
        "bootstrap_receipt_id": "sha256:" + ("1" * 64),
        "plan_root_cid": "plan:sealed",
        "repository_tree_id": "tree:sealed",
    }
    for field, value in (
        ("previous_receipt_cid", "sha256:" + ("7" * 64)),
        ("base_head", "0" * 40),
        ("changed_paths", ["out/of/scope.py"]),
    ):
        malformed = dict(receipt)
        malformed[field] = value
        malformed["receipt_cid"] = aseh_operator._identity(
            {
                key: item
                for key, item in malformed.items()
                if key != "receipt_cid"
            }
        )
        with pytest.raises(aseh_operator.OperatorError, match="authority differs"):
            aseh_operator._validate_repair_sealed_owner_identity_transition(
                malformed,
                bootstrap=bootstrap,
                previous_receipt={},
                rerun_validations=False,
            )

    receipt["transition_revision"] = 11
    receipt["receipt_cid"] = aseh_operator._identity(
        {
            key: value
            for key, value in receipt.items()
            if key != "receipt_cid"
        }
    )
    with pytest.raises(aseh_operator.OperatorError, match="schema"):
        aseh_operator._repair_sealed_owner_identity_transition_receipt_id(
            receipt
        )


def test_aseh_r12_sealed_owner_identity_routes_only_board_check_to_launch_tree(
) -> None:
    commands = (
        aseh_operator.REPAIR_SEALED_OWNER_IDENTITY_TRANSITION_VALIDATIONS
    )
    assert [
        aseh_operator._r12_validation_working_tree_scope(command)
        for command in commands
    ] == [
        "immutable_candidate_checkout",
        "immutable_candidate_checkout",
        "candidate_authorization_worktree",
        "immutable_candidate_checkout",
    ]


def test_aseh_receipt_validation_python_is_stable_under_resolved_interpreter_alias(
) -> None:
    group_names = [
        "REPAIR_TRANSITION_VALIDATIONS",
        "REPAIR_FOLLOWUP_TRANSITION_VALIDATIONS",
        "REPAIR_CLEAN_LAUNCH_TRANSITION_VALIDATIONS",
        "REPAIR_RUNTIME_HARDENING_TRANSITION_VALIDATIONS",
        "REPAIR_QUACK_RECOVERY_TRANSITION_VALIDATIONS",
        "REPAIR_PARALLEL_BLOCKED_STARTUP_TRANSITION_VALIDATIONS",
        "REPAIR_QUACK_PUBLICATION_CONTENTION_TRANSITION_VALIDATIONS",
        "REPAIR_QUACK_RECOVERY_REPLAY_TRANSITION_VALIDATIONS",
        "REPAIR_CONTROL_RECEIPT_LIFECYCLE_TRANSITION_VALIDATIONS",
        "REPAIR_PROVIDER_LEASE_OWNERSHIP_TRANSITION_VALIDATIONS",
        "REPAIR_PROVIDER_CLEANUP_FENCE_TRANSITION_VALIDATIONS",
        "REPAIR_SEALED_OWNER_IDENTITY_TRANSITION_VALIDATIONS",
        "REPAIR_VALIDATION_EXECUTOR_IDENTITY_TRANSITION_VALIDATIONS",
        "REPAIR_SEALED_RECEIPT_VALIDATION_TRANSITION_VALIDATIONS",
    ]
    probe = (
        "import json,runpy,sys;"
        "sys.path.extend(['/usr/lib/python3/dist-packages',"
        "'/usr/local/lib/python3.12/dist-packages']);"
        "m=runpy.run_path("
        "'scripts/run_agent_supervisor_efficiency_state_hardening.py');"
        f"names={group_names!r};"
        "print(json.dumps({'runtime':sys.executable,"
        "'stable':m['ASEH_RECEIPT_VALIDATION_PYTHON'],"
        "'groups':{name:[list(item) for item in m[name]] for name in names},"
        "'baseline':list(m['REPAIR_PROVIDER_CLEANUP_FENCE_KNOWN_BASELINE_COMMAND'])},"
        "sort_keys=True))"
    )
    completed = subprocess.run(
        ("/usr/bin/python3.12", "-I", "-S", "-c", probe),
        cwd=Path(__file__).resolve().parents[2],
        env={
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C.UTF-8",
            "LANG": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    assert observed["runtime"] == "/usr/bin/python3.12"
    assert observed["stable"] == "/usr/bin/python3"
    assert {
        name: len(commands) for name, commands in observed["groups"].items()
    } == dict(
        zip(
            group_names,
            [5, 6, 2, 7, 5, 3, 3, 2, 2, 2, 15, 4, 4, 4],
            strict=True,
        )
    )
    for commands in observed["groups"].values():
        for command in commands:
            assert "/usr/bin/python3.12" not in command
            if command[0] == "/usr/bin/git":
                assert "/usr/bin/python3" not in command
            else:
                assert command.count("/usr/bin/python3") == 1
    assert observed["baseline"][0] == "/usr/bin/python3"


def test_aseh_receipt_validation_python_qualifies_exact_alias_and_rejects_other_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        aseh_operator,
        "_ASEH_RECEIPT_VALIDATION_PYTHON_IDENTITY",
        None,
    )
    monkeypatch.setattr(aseh_operator.sys, "executable", "/usr/bin/python3.12")
    assert (
        aseh_operator._trusted_receipt_validation_python()
        == "/usr/bin/python3"
    )

    monkeypatch.setattr(aseh_operator.sys, "executable", "/usr/bin/false")
    with pytest.raises(aseh_operator.OperatorError, match="identity drifted"):
        aseh_operator._trusted_receipt_validation_python()


def test_aseh_receipt_validation_python_qualifies_nested_env_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualified: list[bool] = []
    monkeypatch.setattr(
        aseh_operator,
        "_trusted_receipt_validation_python",
        lambda: qualified.append(True) or "/usr/bin/python3",
    )
    monkeypatch.setattr(
        aseh_operator.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="",
            stderr="",
        ),
    )

    aseh_operator._run(
        (
            "/usr/bin/env",
            "IPFS_ACCELERATE_AGENT_REQUIRE_LIVE_NATIVE_DEPENDENCY_VALIDATION=1",
            "/usr/bin/python3",
            "-m",
            "pytest",
            "-q",
        )
    )

    assert qualified == [True]


def test_aseh_sealed_receipt_validation_bootstrap_is_compilable_and_content_bound(
) -> None:
    source = aseh_operator.ASEH_RECEIPT_VALIDATION_BOOTSTRAP
    compile(source, "<aseh-sealed-receipt-validation>", "exec")
    assert "\0" not in source
    assert aseh_operator.ASEH_RECEIPT_VALIDATION_BOOTSTRAP_SHA256 == (
        "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()
    )
    contract = aseh_operator._sealed_receipt_validation_executor_contract()
    assert contract["effective_flags"] == ["-S", "-P", "-c"]
    assert contract["bootstrap_sha256"] == (
        aseh_operator.ASEH_RECEIPT_VALIDATION_BOOTSTRAP_SHA256
    )
    assert contract["native_authorization_id"] == (
        aseh_operator.ASEH_R11_NATIVE_DEPENDENCY_AUTHORIZATION_ID
    )
    assert contract["owner_or_database_acquired_inside_scope"] is False
    assert len(contract["handshake_schemas"]) == 3
    assert "PYTHONPATH" not in contract["startup_environment_keys"]


def test_aseh_sealed_receipt_validation_command_grammar_is_bounded() -> None:
    command = aseh_operator.REPAIR_TRANSITION_VALIDATIONS[0]
    parsed = aseh_operator._parse_receipt_validation_python_command(
        command,
        require_known=True,
    )
    assert parsed is not None
    assert parsed[0] == command
    assert parsed[2][:2] == ("-m", "pytest")

    with pytest.raises(aseh_operator.OperatorError, match="environment"):
        aseh_operator._parse_receipt_validation_python_command(
            (
                "/usr/bin/env",
                "PYTHONPATH=/tmp/hostile",
                "/usr/bin/python3",
                "-m",
                "pytest",
            ),
            require_known=False,
        )
    with pytest.raises(aseh_operator.OperatorError, match="grammar"):
        aseh_operator._parse_receipt_validation_python_command(
            ("/usr/bin/python3", "-c", "raise SystemExit(0)"),
            require_known=False,
        )
    direct = [
        command
        for command in (
            aseh_operator
            .REPAIR_PROVIDER_CLEANUP_FENCE_TRANSITION_VALIDATIONS
        )
        if aseh_operator._uses_historical_self_qualified_native_route(command)
    ]
    assert direct
    assert all(command[0] == "/usr/bin/env" for command in direct)


@pytest.mark.parametrize(
    ("command", "expected_stdout"),
    [
        (aseh_operator.REPAIR_TRANSITION_VALIDATIONS[0], "8 passed"),
        (
            aseh_operator
            .REPAIR_SEALED_RECEIPT_VALIDATION_TRANSITION_VALIDATIONS[0],
            "",
        ),
    ],
    ids=["pytest", "py_compile"],
)
def test_aseh_sealed_receipt_validation_executor_runs_r1_without_user_site(
    monkeypatch: pytest.MonkeyPatch,
    command: tuple[str, ...],
    expected_stdout: str,
) -> None:
    source = Path(
        "/home/barberb/.local/lib/python3.12/site-packages/"
        + str(
            aseh_operator
            .ASEH_R11_NATIVE_DEPENDENCY_PIN["extension_filename"]
        )
    )
    if not source.is_file():
        if os.environ.get(
            "IPFS_ACCELERATE_AGENT_REQUIRE_LIVE_NATIVE_DEPENDENCY_VALIDATION"
        ) == "1":
            pytest.fail("reviewed DuckDB native source is unavailable")
        pytest.skip("reviewed DuckDB native source is unavailable")
    pin = llm_router.inspect_agent_supervisor_native_dependency_source(
        source,
        distribution_version="1.5.5",
        engine_version="v1.5.5",
    )
    assert pin.as_dict() == aseh_operator.ASEH_R11_NATIVE_DEPENDENCY_PIN
    native = llm_router.seal_agent_supervisor_native_dependency(
        source,
        expected_pin=pin,
        accepted_authorization_id=(
            aseh_operator.ASEH_R11_NATIVE_DEPENDENCY_AUTHORIZATION_ID
        ),
    )
    interpreter = multi_runner.retain_control_plane_interpreter(
        "/usr/bin/python3"
    )
    monkeypatch.setattr(
        aseh_operator,
        "_assert_candidate_authorization_witness",
        lambda *_args, **_kwargs: None,
    )
    environment = aseh_operator._r11_validation_environment(
        aseh_operator.ROOT
    )
    try:
        with aseh_operator._sealed_receipt_validation_executor_scope(
            interpreter=interpreter,
            native_dependency=native,
            system_dependency_directories_json=(
                multi_runner.trusted_system_dependency_directories_json()
            ),
            candidate_head="1" * 40,
            candidate_tree="2" * 40,
            authorization_witness={"test": "bounded"},
            base_environment=environment,
        ):
            completed = aseh_operator._run(
                command,
                timeout=120,
                env=environment,
                cwd=aseh_operator.ROOT,
            )
        assert completed.returncode == 0, completed.stderr
        assert expected_stdout in completed.stdout
        evidence = completed.aseh_sealed_execution_evidence
        assert evidence["completion"]["descendants_drained"] is True
        aseh_operator._validate_sealed_validation_execution_evidence(
            evidence,
            declared=command,
            candidate_head="1" * 40,
            candidate_tree="2" * 40,
            authorization_witness={"test": "bounded"},
            environment_identity=(
                aseh_operator._r11_command_environment_identity(
                    environment,
                    command,
                    checkout=aseh_operator.ROOT,
                )
            ),
            returncode=0,
            stdout_digest=completed.aseh_sealed_stdout_digest,
            stderr_digest=completed.aseh_sealed_stderr_digest,
        )
        replayed = json.loads(json.dumps(evidence))
        replayed_witness = {"test": "different-tree"}
        replayed_context = (
            aseh_operator._sealed_validation_execution_context(
                declared=command,
                candidate_head="1" * 40,
                candidate_tree="3" * 40,
                authorization_witness=replayed_witness,
                environment_identity=(
                    aseh_operator._r11_command_environment_identity(
                        environment,
                        command,
                        checkout=aseh_operator.ROOT,
                    )
                ),
                native_authorization_id=(
                    aseh_operator
                    .ASEH_R11_NATIVE_DEPENDENCY_AUTHORIZATION_ID
                ),
            )
        )
        replayed_context_cid = aseh_operator._identity(replayed_context)
        replayed["execution_context"] = replayed_context
        replayed["execution_context_cid"] = replayed_context_cid
        replayed_binding = dict(replayed["execution_binding"])
        replayed_binding["execution_context_cid"] = replayed_context_cid
        replayed["execution_binding"] = replayed_binding
        replayed["execution_binding_cid"] = aseh_operator._identity(
            replayed_binding
        )
        replayed["evidence_cid"] = aseh_operator._identity(
            {
                key: value
                for key, value in replayed.items()
                if key != "evidence_cid"
            }
        )
        with pytest.raises(
            aseh_operator.OperatorError,
            match="readiness evidence",
        ):
            aseh_operator._validate_sealed_validation_execution_evidence(
                replayed,
                declared=command,
                candidate_head="1" * 40,
                candidate_tree="3" * 40,
                authorization_witness=replayed_witness,
                environment_identity=(
                    aseh_operator._r11_command_environment_identity(
                        environment,
                        command,
                        checkout=aseh_operator.ROOT,
                    )
                ),
                returncode=0,
                stdout_digest=completed.aseh_sealed_stdout_digest,
                stderr_digest=completed.aseh_sealed_stderr_digest,
            )
        assert aseh_operator._ASEH_RECEIPT_VALIDATION_EXECUTOR is None
    finally:
        os.close(interpreter.descriptor)
        os.close(native.descriptor.descriptor)


@pytest.mark.parametrize(
    "invalid_output",
    [False, True],
    ids=["utf8", "invalid_utf8"],
)
def test_aseh_sealed_receipt_validation_monitor_drains_setsid_descendant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_output: bool,
) -> None:
    source = Path(
        "/home/barberb/.local/lib/python3.12/site-packages/"
        + str(
            aseh_operator
            .ASEH_R11_NATIVE_DEPENDENCY_PIN["extension_filename"]
        )
    )
    if not source.is_file():
        pytest.skip("reviewed DuckDB native source is unavailable")
    marker = tmp_path / "detached.pid"
    test_path = tmp_path / "test_detached.py"
    test_path.write_text(
        (
        "import os,pathlib,time\n"
        "def test_detached():\n"
        "    child=os.fork()\n"
        "    if child==0:\n"
        "        os.setsid()\n"
        f"        pathlib.Path({str(marker)!r}).write_text(str(os.getpid()))\n"
        "        time.sleep(60)\n"
        "        os._exit(0)\n"
        "    deadline=time.monotonic()+5\n"
        f"    marker=pathlib.Path({str(marker)!r})\n"
        "    while not marker.exists() and time.monotonic()<deadline:\n"
        "        time.sleep(.01)\n"
        "    assert marker.exists()\n"
        + ("    os.write(1,b'\\xff')\n" if invalid_output else "")
        ),
        encoding="utf-8",
    )
    command = (
        "/usr/bin/python3",
        "-m",
        "pytest",
        "-q",
        *(("-s",) if invalid_output else ()),
        test_path.name,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_receipt_validation_matrices",
        lambda: ((command,),),
    )
    pin = llm_router.inspect_agent_supervisor_native_dependency_source(
        source,
        distribution_version="1.5.5",
        engine_version="v1.5.5",
    )
    native = llm_router.seal_agent_supervisor_native_dependency(
        source,
        expected_pin=pin,
        accepted_authorization_id=(
            aseh_operator.ASEH_R11_NATIVE_DEPENDENCY_AUTHORIZATION_ID
        ),
    )
    interpreter = multi_runner.retain_control_plane_interpreter(
        "/usr/bin/python3"
    )
    monkeypatch.setattr(
        aseh_operator,
        "_assert_candidate_authorization_witness",
        lambda *_args, **_kwargs: None,
    )
    environment = aseh_operator._r11_validation_environment(tmp_path)
    try:
        with aseh_operator._sealed_receipt_validation_executor_scope(
            interpreter=interpreter,
            native_dependency=native,
            system_dependency_directories_json=(
                multi_runner.trusted_system_dependency_directories_json()
            ),
            candidate_head="3" * 40,
            candidate_tree="4" * 40,
            authorization_witness={"test": "descendant-drain"},
            base_environment=environment,
        ):
            if invalid_output:
                with pytest.raises(
                    aseh_operator.OperatorError,
                    match="output is not UTF-8",
                ):
                    aseh_operator._run(
                        command,
                        timeout=30,
                        env=environment,
                        cwd=tmp_path,
                    )
                completed = None
            else:
                completed = aseh_operator._run(
                    command,
                    timeout=30,
                    env=environment,
                    cwd=tmp_path,
                )
        if completed is not None:
            assert completed.returncode == 0, completed.stderr
        assert marker.is_file()
        detached_pid = int(marker.read_text(encoding="utf-8"))
        _assert_process_not_executable(detached_pid)
        if completed is not None:
            assert (
                completed.aseh_sealed_execution_evidence["completion"]
                ["descendants_drained"]
                is True
            )
    finally:
        os.close(interpreter.descriptor)
        os.close(native.descriptor.descriptor)


def test_aseh_sealed_receipt_validation_owner_loss_drains_descendants(
    tmp_path: Path,
) -> None:
    source = Path(
        "/home/barberb/.local/lib/python3.12/site-packages/"
        + str(
            aseh_operator
            .ASEH_R11_NATIVE_DEPENDENCY_PIN["extension_filename"]
        )
    )
    if not source.is_file():
        pytest.skip("reviewed DuckDB native source is unavailable")
    marker = tmp_path / "owner-loss-descendant.pid"
    validator = tmp_path / "test_owner_loss.py"
    validator.write_text(
        "import os,pathlib,time\n"
        "def test_owner_loss():\n"
        "    child=os.fork()\n"
        "    if child==0:\n"
        "        os.setsid()\n"
        f"        pathlib.Path({str(marker)!r}).write_text(str(os.getpid()))\n"
        "        time.sleep(60)\n"
        "        os._exit(0)\n"
        "    time.sleep(60)\n",
        encoding="utf-8",
    )
    command = (
        "/usr/bin/python3",
        "-m",
        "pytest",
        "-q",
        validator.name,
    )
    helper = tmp_path / "validation_owner.py"
    helper.write_text(
        "import os,sys\n"
        f"sys.path.insert(0,{str(aseh_operator.ROOT)!r})\n"
        "from pathlib import Path\n"
        "from ipfs_accelerate_py import llm_router\n"
        "from ipfs_accelerate_py.agent_supervisor.runtime import "
        "multi_supervisor_runner as mr\n"
        "from scripts import "
        "run_agent_supervisor_efficiency_state_hardening as m\n"
        f"source=Path({str(source)!r})\n"
        "pin=llm_router.inspect_agent_supervisor_native_dependency_source("
        "source,distribution_version='1.5.5',engine_version='v1.5.5')\n"
        "native=llm_router.seal_agent_supervisor_native_dependency("
        "source,expected_pin=pin,accepted_authorization_id="
        "m.ASEH_R11_NATIVE_DEPENDENCY_AUTHORIZATION_ID)\n"
        "interpreter=mr.retain_control_plane_interpreter('/usr/bin/python3')\n"
        "m._assert_candidate_authorization_witness=lambda *args,**kwargs:None\n"
        f"command={command!r}\n"
        "m._receipt_validation_matrices=lambda:((command,),)\n"
        f"working=Path({str(tmp_path)!r})\n"
        "environment=m._r11_validation_environment(working)\n"
        "try:\n"
        "    with m._sealed_receipt_validation_executor_scope("
        "interpreter=interpreter,native_dependency=native,"
        "system_dependency_directories_json="
        "mr.trusted_system_dependency_directories_json(),"
        "candidate_head='7'*40,candidate_tree='8'*40,"
        "authorization_witness={'test':'owner-loss'},"
        "base_environment=environment):\n"
        "        m._run(command,timeout=120,env=environment,cwd=working)\n"
        "finally:\n"
        "    os.close(interpreter.descriptor)\n"
        "    os.close(native.descriptor.descriptor)\n",
        encoding="utf-8",
    )
    owner = subprocess.Popen(
        (sys.executable, str(helper)),
        cwd=aseh_operator.ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    detached_pid = 0
    try:
        deadline = time.monotonic() + 15.0
        while not marker.is_file() and time.monotonic() < deadline:
            if owner.poll() is not None:
                stdout, stderr = owner.communicate()
                pytest.fail(
                    "validation owner exited before descendant birth: "
                    + (stdout + stderr).decode("utf-8", errors="replace")
                )
            time.sleep(0.01)
        assert marker.is_file()
        detached_pid = int(marker.read_text(encoding="utf-8"))
        owner.kill()
        owner.wait(timeout=5.0)
        _assert_process_not_executable(detached_pid)
    finally:
        if owner.poll() is None:
            os.killpg(owner.pid, signal.SIGKILL)
            owner.wait(timeout=5.0)
        if detached_pid > 1:
            try:
                os.kill(detached_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_aseh_r13_validation_executor_identity_routes_only_board_check_to_launch_tree(
) -> None:
    commands = (
        aseh_operator.REPAIR_VALIDATION_EXECUTOR_IDENTITY_TRANSITION_VALIDATIONS
    )
    assert [
        aseh_operator._r13_validation_working_tree_scope(command)
        for command in commands
    ] == [
        "immutable_candidate_checkout",
        "immutable_candidate_checkout",
        "candidate_authorization_worktree",
        "immutable_candidate_checkout",
    ]


def test_aseh_r14_sealed_receipt_validation_routes_only_board_check_to_launch_tree(
) -> None:
    commands = (
        aseh_operator.REPAIR_SEALED_RECEIPT_VALIDATION_TRANSITION_VALIDATIONS
    )
    assert [
        aseh_operator._r14_validation_working_tree_scope(command)
        for command in commands
    ] == [
        "immutable_candidate_checkout",
        "immutable_candidate_checkout",
        "candidate_authorization_worktree",
        "immutable_candidate_checkout",
    ]


def test_aseh_r14_authorization_holds_lock_and_sealed_executor_through_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board = object()
    paths: dict[str, Path] = {}
    candidate_head = "1" * 40
    candidate_tree = "2" * 40
    anchor = {
        "head": "3" * 40,
        "tree": "4" * 40,
        "witness": {"anchor": "r13"},
    }
    interpreter_fd = os.open("/dev/null", os.O_RDONLY)
    native_fd = os.open("/dev/null", os.O_RDONLY)
    interpreter = SimpleNamespace(descriptor=interpreter_fd)
    native = SimpleNamespace(
        descriptor=SimpleNamespace(descriptor=native_fd)
    )
    ordering: list[str] = []
    lock_active = False
    executor_active = False

    @contextmanager
    def guard(_paths: object) -> object:
        nonlocal lock_active
        lock_active = True
        ordering.append("lock_enter")
        try:
            yield 99
        finally:
            ordering.append("lock_exit")
            lock_active = False

    @contextmanager
    def scope(**kwargs: object) -> object:
        nonlocal executor_active
        assert lock_active is True
        assert kwargs["candidate_head"] == candidate_head
        assert kwargs["candidate_tree"] == candidate_tree
        executor_active = True
        ordering.append("executor_enter")
        try:
            yield None
        finally:
            ordering.append("executor_exit")
            executor_active = False

    def seal_native(**kwargs: object) -> object:
        assert lock_active is True
        assert kwargs["candidate_head"] == anchor["head"]
        assert kwargs["candidate_tree"] == anchor["tree"]
        assert kwargs["candidate_authorization_witness"] == anchor["witness"]
        assert kwargs["launch_admission"] is None
        assert kwargs["active_candidate_head"] == candidate_head
        assert kwargs["active_candidate_tree"] == candidate_tree
        assert kwargs["active_candidate_authorization_witness"] == {
            "candidate": "exact"
        }
        ordering.append("native_seal")
        return native

    def authorize_locked(**kwargs: object) -> dict[str, object]:
        assert lock_active is True
        assert executor_active is True
        assert kwargs["authorization_directory_fd"] == 99
        ordering.append("receipt_publication")
        return {"ok": True}

    monkeypatch.setattr(aseh_operator, "_load", lambda _path: (board, {}))
    monkeypatch.setattr(aseh_operator, "_paths", lambda _board: paths)
    monkeypatch.setattr(
        aseh_operator,
        "_repair_transition_authorization_guard",
        guard,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_assert_clean_tree",
        lambda _board: (candidate_head, candidate_tree),
    )
    monkeypatch.setattr(
        aseh_operator,
        "_r14_native_seal_anchor",
        lambda *_args, **_kwargs: anchor,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_candidate_authorization_witness",
        lambda **_kwargs: {"candidate": "exact"},
    )
    monkeypatch.setattr(
        aseh_operator,
        "_build_aseh_qualification_home",
        lambda _paths: tmp_path,
    )
    monkeypatch.setattr(
        multi_runner,
        "retain_control_plane_interpreter",
        lambda _python: ordering.append("retain_interpreter") or interpreter,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_seal_r11_native_dependency",
        seal_native,
    )
    monkeypatch.setattr(
        multi_runner,
        "trusted_system_dependency_directories_json",
        lambda: "[]",
    )
    monkeypatch.setattr(
        aseh_operator,
        "_sealed_owner_delegation_environment",
        lambda _home: {},
    )
    monkeypatch.setattr(
        aseh_operator,
        "_sealed_receipt_validation_executor_scope",
        scope,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_authorize_repair_transition_locked",
        authorize_locked,
    )

    assert aseh_operator.authorize_repair_transition(tmp_path / "board.json") == {
        "ok": True
    }
    assert ordering == [
        "lock_enter",
        "retain_interpreter",
        "native_seal",
        "executor_enter",
        "receipt_publication",
        "executor_exit",
        "lock_exit",
    ]
    for descriptor in (interpreter_fd, native_fd):
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_aseh_r14_sealed_receipt_validation_requires_exact_ordered_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_head = "a" * 40
    candidate_tree = "b" * 40
    chain = [
        {
            "schema": schema,
            "transition_revision": None if index == 0 else index + 1,
            "receipt_cid": "sha256:" + f"{index + 1:064x}",
        }
        for index, schema in enumerate(
            aseh_operator.ASEH_R14_REPAIR_TRANSITION_CHAIN_SCHEMAS
        )
    ]
    for index in range(1, len(chain)):
        chain[index]["previous_receipt_cid"] = chain[index - 1][
            "receipt_cid"
        ]
    chain[-2]["repair_head"] = (
        aseh_operator.REPAIR_SEALED_RECEIPT_VALIDATION_TRANSITION_BASE_HEAD
    )
    chain[-1].update(
        {"repair_head": candidate_head, "repair_tree": candidate_tree}
    )
    admission = {
        "runtime_source_head": candidate_head,
        "runtime_repository_tree_id": candidate_tree,
        "repair_transition": dict(chain[-1]),
        "repair_transition_chain": chain,
    }
    monkeypatch.setattr(
        aseh_operator,
        "_git",
        lambda *_args: (
            aseh_operator
            .REPAIR_SEALED_RECEIPT_VALIDATION_TRANSITION_BASE_HEAD
        ),
    )
    aseh_operator._assert_exact_run_launch_admission(
        admission,
        candidate_head=candidate_head,
        candidate_tree=candidate_tree,
    )

    for malformed in (
        chain[:-1],
        [*chain[:-2], chain[-1], chain[-2]],
        [
            *chain[:-2],
            {**chain[-2], "receipt_cid": chain[-1]["receipt_cid"]},
            {**chain[-1], "receipt_cid": chain[-2]["receipt_cid"]},
        ],
    ):
        with pytest.raises(aseh_operator.OperatorError, match="R14"):
            aseh_operator._admit_exact_r14_transition_chain(malformed)


def test_aseh_repair_validation_executor_identity_transition_is_closed_and_chained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repair_head = "7" * 40
    repair_tree = "8" * 40
    base_tree = "9" * 40
    prior_cid = "sha256:" + ("a" * 64)
    patch_digest = "sha256:" + ("b" * 64)
    dataset_commit = "c" * 40
    kit_commit = "d" * 40
    witness = {
        "head": repair_head,
        "tree": repair_tree,
        "branch_ref": "refs/heads/aseh-r13-fixture",
        "index_entries_digest": "sha256:" + ("1" * 64),
        "index_flags_digest": "sha256:" + ("2" * 64),
        "status_digest": aseh_operator._identity(b""),
        "head_reflog_digest": "sha256:" + ("3" * 64),
        "branch_reflog_digest": "sha256:" + ("4" * 64),
    }
    validations = [
        {
            "argv": list(command),
            "candidate_head": repair_head,
            "candidate_tree": repair_tree,
            "environment_identity": (
                aseh_operator._r11_command_environment_identity(
                    aseh_operator._r11_validation_environment(
                        Path("/sealed-checkout")
                    ),
                    command,
                    checkout=Path("/sealed-checkout"),
                )
            ),
            "working_tree_scope": (
                aseh_operator._r13_validation_working_tree_scope(command)
            ),
            "returncode": 0,
            "stdout_digest": "sha256:" + ("5" * 64),
            "stderr_digest": "sha256:" + ("6" * 64),
        }
        for command in (
            aseh_operator
            .REPAIR_VALIDATION_EXECUTOR_IDENTITY_TRANSITION_VALIDATIONS
        )
    ]
    bootstrap = {
        "bootstrap_receipt_id": "sha256:" + ("e" * 64),
        "plan_root_cid": "plan:r13",
        "repository_tree_id": "tree:bootstrap",
        "source_forest": {
            "by_owner": {
                "ipfs_datasets_py": {"commit": dataset_commit},
                "ipfs_kit_py": {"commit": kit_commit},
            }
        },
    }
    receipt = {
        "schema": (
            aseh_operator
            .REPAIR_VALIDATION_EXECUTOR_IDENTITY_TRANSITION_SCHEMA
        ),
        "task_id": aseh_operator.REPAIR_TRANSITION_TASK_ID,
        "stable_identity": (
            f"{aseh_operator.PROGRAM}/"
            f"{aseh_operator.REPAIR_TRANSITION_TASK_ID}@ASEH-PLAN-R13"
        ),
        "program_id": aseh_operator.PROGRAM,
        "transition_revision": 13,
        "bootstrap_receipt_id": bootstrap["bootstrap_receipt_id"],
        "previous_receipt_cid": prior_cid,
        "plan_root_cid": bootstrap["plan_root_cid"],
        "repository_tree_id": bootstrap["repository_tree_id"],
        "base_head": (
            aseh_operator
            .REPAIR_VALIDATION_EXECUTOR_IDENTITY_TRANSITION_BASE_HEAD
        ),
        "base_tree": base_tree,
        "repair_head": repair_head,
        "repair_tree": repair_tree,
        "changed_paths": list(
            aseh_operator
            .REPAIR_VALIDATION_EXECUTOR_IDENTITY_TRANSITION_CHANGED_PATHS
        ),
        "patch_digest": patch_digest,
        "dependencies": ["ASEH-BOOTSTRAP-002@ASEH-PLAN-R12"],
        "owning_repository": "ipfs_accelerate_py",
        "risk_class": "R4_SECURITY_OR_PROTOCOL_SENSITIVE",
        "authority_requirement": (
            aseh_operator
            .REPAIR_VALIDATION_EXECUTOR_IDENTITY_TRANSITION_AUTHORITY
        ),
        "validation_results": validations,
        "candidate_authorization_witness": witness,
        "terminal_success_criteria": (
            aseh_operator
            .REPAIR_VALIDATION_EXECUTOR_IDENTITY_TRANSITION_SUCCESS
        ),
        "terminal_non_success_criteria": (
            aseh_operator
            .REPAIR_VALIDATION_EXECUTOR_IDENTITY_TRANSITION_NON_SUCCESS
        ),
        "semantic_corpus_changed": False,
        "database_mutated": False,
        "authorized_at": 1.0,
    }
    receipt["receipt_cid"] = aseh_operator._identity(receipt)

    def fake_git(*arguments: str) -> str:
        if arguments[:3] == ("show", "-s", "--format=%P"):
            return (
                aseh_operator
                .REPAIR_VALIDATION_EXECUTOR_IDENTITY_TRANSITION_BASE_HEAD
            )
        if arguments[:1] == ("rev-parse",):
            revision = arguments[1]
            if revision.endswith(":ipfs_datasets_py"):
                return dataset_commit
            if revision.endswith(":ipfs_kit_py"):
                return kit_commit
            if revision == repair_head + "^{tree}":
                return repair_tree
            return base_tree
        raise AssertionError(arguments)

    monkeypatch.setattr(aseh_operator, "_git", fake_git)
    monkeypatch.setattr(
        aseh_operator,
        "_git_changed_paths",
        lambda *_args: (
            aseh_operator
            .REPAIR_VALIDATION_EXECUTOR_IDENTITY_TRANSITION_CHANGED_PATHS
        ),
    )
    monkeypatch.setattr(
        aseh_operator,
        "_git_patch_digest",
        lambda *_args: patch_digest,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_repair_sealed_owner_identity_transition_receipt_id",
        lambda _payload: prior_cid,
    )

    admitted = (
        aseh_operator
        ._validate_repair_validation_executor_identity_transition(
            receipt,
            bootstrap=bootstrap,
            previous_receipt={"receipt_cid": prior_cid},
            rerun_validations=False,
        )
    )
    assert admitted["transition_revision"] == 13
    assert admitted["previous_receipt_cid"] == prior_cid
    assert admitted["candidate_authorization_witness"] == witness

    malformed = json.loads(json.dumps(receipt))
    malformed["validation_results"][0]["argv"][0] = "/usr/bin/python3.12"
    malformed["receipt_cid"] = aseh_operator._identity(
        {key: value for key, value in malformed.items() if key != "receipt_cid"}
    )
    with pytest.raises(
        aseh_operator.OperatorError,
        match="validation differs",
    ):
        aseh_operator._validate_repair_validation_executor_identity_transition(
            malformed,
            bootstrap=bootstrap,
            previous_receipt={"receipt_cid": prior_cid},
            rerun_validations=False,
        )


def test_aseh_repair_sealed_receipt_validation_transition_is_closed_and_chained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repair_head = "a" * 40
    repair_tree = "b" * 40
    base_tree = "c" * 40
    prior_cid = "sha256:" + ("d" * 64)
    patch_digest = "sha256:" + ("e" * 64)
    dataset_commit = "1" * 40
    kit_commit = "2" * 40
    witness = {
        "head": repair_head,
        "tree": repair_tree,
        "branch_ref": "refs/heads/aseh-r14-fixture",
        "index_entries_digest": "sha256:" + ("3" * 64),
        "index_flags_digest": "sha256:" + ("4" * 64),
        "status_digest": aseh_operator._identity(b""),
        "head_reflog_digest": "sha256:" + ("5" * 64),
        "branch_reflog_digest": "sha256:" + ("6" * 64),
    }
    validations = [
        {
            "argv": list(command),
            "candidate_head": repair_head,
            "candidate_tree": repair_tree,
            "environment_identity": (
                aseh_operator._r11_command_environment_identity(
                    aseh_operator._r11_validation_environment(
                        Path("/sealed-checkout")
                    ),
                    command,
                    checkout=Path("/sealed-checkout"),
                )
            ),
            "working_tree_scope": (
                aseh_operator._r14_validation_working_tree_scope(command)
            ),
            "returncode": 0,
            "stdout_digest": "sha256:" + ("7" * 64),
            "stderr_digest": "sha256:" + ("8" * 64),
            "sealed_execution_evidence": (
                _sealed_validation_execution_fixture(
                    command,
                    candidate_head=repair_head,
                    candidate_tree=repair_tree,
                    authorization_witness=witness,
                    environment_identity=(
                        aseh_operator._r11_command_environment_identity(
                            aseh_operator._r11_validation_environment(
                                Path("/sealed-checkout")
                            ),
                            command,
                            checkout=Path("/sealed-checkout"),
                        )
                    ),
                    stdout_digest="sha256:" + ("7" * 64),
                    stderr_digest="sha256:" + ("8" * 64),
                )
            ),
        }
        for command in (
            aseh_operator
            .REPAIR_SEALED_RECEIPT_VALIDATION_TRANSITION_VALIDATIONS
        )
    ]
    bootstrap = {
        "bootstrap_receipt_id": "sha256:" + ("9" * 64),
        "plan_root_cid": "plan:r14",
        "repository_tree_id": "tree:bootstrap",
        "source_forest": {
            "by_owner": {
                "ipfs_datasets_py": {"commit": dataset_commit},
                "ipfs_kit_py": {"commit": kit_commit},
            }
        },
    }
    receipt = {
        "schema": (
            aseh_operator.REPAIR_SEALED_RECEIPT_VALIDATION_TRANSITION_SCHEMA
        ),
        "task_id": aseh_operator.REPAIR_TRANSITION_TASK_ID,
        "stable_identity": (
            f"{aseh_operator.PROGRAM}/"
            f"{aseh_operator.REPAIR_TRANSITION_TASK_ID}@ASEH-PLAN-R14"
        ),
        "program_id": aseh_operator.PROGRAM,
        "transition_revision": 14,
        "bootstrap_receipt_id": bootstrap["bootstrap_receipt_id"],
        "previous_receipt_cid": prior_cid,
        "plan_root_cid": bootstrap["plan_root_cid"],
        "repository_tree_id": bootstrap["repository_tree_id"],
        "base_head": (
            aseh_operator
            .REPAIR_SEALED_RECEIPT_VALIDATION_TRANSITION_BASE_HEAD
        ),
        "base_tree": base_tree,
        "repair_head": repair_head,
        "repair_tree": repair_tree,
        "changed_paths": list(
            aseh_operator
            .REPAIR_SEALED_RECEIPT_VALIDATION_TRANSITION_CHANGED_PATHS
        ),
        "patch_digest": patch_digest,
        "dependencies": ["ASEH-BOOTSTRAP-002@ASEH-PLAN-R13"],
        "owning_repository": "ipfs_accelerate_py",
        "risk_class": "R4_SECURITY_OR_PROTOCOL_SENSITIVE",
        "authority_requirement": (
            aseh_operator
            .REPAIR_SEALED_RECEIPT_VALIDATION_TRANSITION_AUTHORITY
        ),
        "validation_results": validations,
        "candidate_authorization_witness": witness,
        "sealed_validation_executor_contract": (
            aseh_operator._sealed_receipt_validation_executor_contract()
        ),
        "terminal_success_criteria": (
            aseh_operator
            .REPAIR_SEALED_RECEIPT_VALIDATION_TRANSITION_SUCCESS
        ),
        "terminal_non_success_criteria": (
            aseh_operator
            .REPAIR_SEALED_RECEIPT_VALIDATION_TRANSITION_NON_SUCCESS
        ),
        "semantic_corpus_changed": False,
        "database_mutated": False,
        "authorized_at": 1.0,
    }
    receipt["receipt_cid"] = aseh_operator._identity(receipt)

    def fake_git(*arguments: str) -> str:
        if arguments[:3] == ("show", "-s", "--format=%P"):
            return (
                aseh_operator
                .REPAIR_SEALED_RECEIPT_VALIDATION_TRANSITION_BASE_HEAD
            )
        if arguments[:1] == ("rev-parse",):
            revision = arguments[1]
            if revision.endswith(":ipfs_datasets_py"):
                return dataset_commit
            if revision.endswith(":ipfs_kit_py"):
                return kit_commit
            if revision == repair_head + "^{tree}":
                return repair_tree
            return base_tree
        raise AssertionError(arguments)

    monkeypatch.setattr(aseh_operator, "_git", fake_git)
    monkeypatch.setattr(
        aseh_operator,
        "_git_changed_paths",
        lambda *_args: (
            aseh_operator
            .REPAIR_SEALED_RECEIPT_VALIDATION_TRANSITION_CHANGED_PATHS
        ),
    )
    monkeypatch.setattr(
        aseh_operator,
        "_git_patch_digest",
        lambda *_args: patch_digest,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_repair_validation_executor_identity_transition_receipt_id",
        lambda _payload: prior_cid,
    )

    admitted = aseh_operator._validate_repair_sealed_receipt_validation_transition(
        receipt,
        bootstrap=bootstrap,
        previous_receipt={"receipt_cid": prior_cid},
        rerun_validations=False,
    )
    assert admitted["transition_revision"] == 14
    assert admitted["previous_receipt_cid"] == prior_cid
    assert admitted["sealed_validation_executor_contract"] == (
        aseh_operator._sealed_receipt_validation_executor_contract()
    )

    malformed = json.loads(json.dumps(receipt))
    malformed["sealed_validation_executor_contract"]["effective_flags"] = [
        "-S",
        "-c",
    ]
    malformed["receipt_cid"] = aseh_operator._identity(
        {key: value for key, value in malformed.items() if key != "receipt_cid"}
    )
    with pytest.raises(aseh_operator.OperatorError, match="schema"):
        aseh_operator._validate_repair_sealed_receipt_validation_transition(
            malformed,
            bootstrap=bootstrap,
            previous_receipt={"receipt_cid": prior_cid},
            rerun_validations=False,
        )

    forged = json.loads(json.dumps(receipt))
    evidence = next(
        item["sealed_execution_evidence"]
        for item in forged["validation_results"]
        if isinstance(item["sealed_execution_evidence"], dict)
    )
    evidence["completion"]["descendants_drained"] = False
    completion_bytes = (
        aseh_operator._canonical_json(evidence["completion"]) + "\n"
    ).encode("utf-8")
    evidence["completion_receipt_sha256"] = (
        "sha256:" + hashlib.sha256(completion_bytes).hexdigest()
    )
    evidence["evidence_cid"] = aseh_operator._identity(
        {
            key: value
            for key, value in evidence.items()
            if key != "evidence_cid"
        }
    )
    forged["receipt_cid"] = aseh_operator._identity(
        {key: value for key, value in forged.items() if key != "receipt_cid"}
    )
    with pytest.raises(aseh_operator.OperatorError, match="completion evidence"):
        aseh_operator._validate_repair_sealed_receipt_validation_transition(
            forged,
            bootstrap=bootstrap,
            previous_receipt={"receipt_cid": prior_cid},
            rerun_validations=False,
        )

    output_replayed = json.loads(json.dumps(receipt))
    observed = next(
        item
        for item in output_replayed["validation_results"]
        if isinstance(item["sealed_execution_evidence"], dict)
    )
    observed["stdout_digest"] = "sha256:" + ("0" * 64)
    output_replayed["receipt_cid"] = aseh_operator._identity(
        {
            key: value
            for key, value in output_replayed.items()
            if key != "receipt_cid"
        }
    )
    with pytest.raises(
        aseh_operator.OperatorError,
        match="execution evidence CID differs",
    ):
        aseh_operator._validate_repair_sealed_receipt_validation_transition(
            output_replayed,
            bootstrap=bootstrap,
            previous_receipt={"receipt_cid": prior_cid},
            rerun_validations=False,
        )


def test_aseh_r12_sealed_owner_identity_requires_r11_launch_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_head = "3" * 40
    candidate_tree = "4" * 40
    admission = {
        "runtime_source_head": candidate_head,
        "runtime_repository_tree_id": candidate_tree,
        "repair_transition": {
            "schema": (
                aseh_operator
                .REPAIR_SEALED_OWNER_IDENTITY_TRANSITION_SCHEMA
            ),
            "repair_head": candidate_head,
            "repair_tree": candidate_tree,
        },
        "repair_transition_chain": [],
    }
    monkeypatch.setattr(
        aseh_operator,
        "_git",
        lambda *_args: (
            aseh_operator.REPAIR_SEALED_OWNER_IDENTITY_TRANSITION_BASE_HEAD
        ),
    )

    with pytest.raises(
        aseh_operator.OperatorError,
        match="current R12 candidate lacks",
    ):
        aseh_operator._assert_exact_run_launch_admission(
            admission,
            candidate_head=candidate_head,
            candidate_tree=candidate_tree,
        )

    admission["repair_transition_chain"] = [
        {
            "schema": (
                aseh_operator.REPAIR_PROVIDER_CLEANUP_FENCE_TRANSITION_SCHEMA
            ),
            "repair_head": (
                aseh_operator
                .REPAIR_SEALED_OWNER_IDENTITY_TRANSITION_BASE_HEAD
            ),
        }
    ]
    aseh_operator._assert_exact_run_launch_admission(
        admission,
        candidate_head=candidate_head,
        candidate_tree=candidate_tree,
    )


def test_aseh_r13_validation_executor_identity_requires_r12_launch_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_head = "5" * 40
    candidate_tree = "6" * 40
    chain = [
        {
            "schema": schema,
            "transition_revision": None if index == 0 else index + 1,
            "receipt_cid": "sha256:" + f"{index + 1:064x}",
        }
        for index, schema in enumerate(
            aseh_operator.ASEH_REPAIR_TRANSITION_CHAIN_SCHEMAS
        )
    ]
    for index in range(1, len(chain)):
        chain[index]["previous_receipt_cid"] = chain[index - 1][
            "receipt_cid"
        ]
    chain[-2]["repair_head"] = (
        aseh_operator.REPAIR_VALIDATION_EXECUTOR_IDENTITY_TRANSITION_BASE_HEAD
    )
    chain[-1].update(
        {"repair_head": candidate_head, "repair_tree": candidate_tree}
    )
    admission = {
        "runtime_source_head": candidate_head,
        "runtime_repository_tree_id": candidate_tree,
        "repair_transition": {
            "schema": (
                aseh_operator
                .REPAIR_VALIDATION_EXECUTOR_IDENTITY_TRANSITION_SCHEMA
            ),
            "repair_head": candidate_head,
            "repair_tree": candidate_tree,
            "previous_receipt_cid": chain[-2]["receipt_cid"],
            "receipt_cid": chain[-1]["receipt_cid"],
        },
        "repair_transition_chain": [],
    }
    monkeypatch.setattr(
        aseh_operator,
        "_git",
        lambda *_args: (
            aseh_operator
            .REPAIR_VALIDATION_EXECUTOR_IDENTITY_TRANSITION_BASE_HEAD
        ),
    )

    with pytest.raises(
        aseh_operator.OperatorError,
        match="R13 repair transition chain differs",
    ):
        aseh_operator._assert_exact_run_launch_admission(
            admission,
            candidate_head=candidate_head,
            candidate_tree=candidate_tree,
        )

    admission["repair_transition_chain"] = chain
    aseh_operator._assert_exact_run_launch_admission(
        admission,
        candidate_head=candidate_head,
        candidate_tree=candidate_tree,
    )

    cid_swapped = [dict(item) for item in chain]
    cid_swapped[-3]["receipt_cid"], cid_swapped[-2]["receipt_cid"] = (
        cid_swapped[-2]["receipt_cid"],
        cid_swapped[-3]["receipt_cid"],
    )
    admission["repair_transition_chain"] = cid_swapped
    with pytest.raises(
        aseh_operator.OperatorError,
        match="R13 repair transition chain differs",
    ):
        aseh_operator._assert_exact_run_launch_admission(
            admission,
            candidate_head=candidate_head,
            candidate_tree=candidate_tree,
        )

    reordered = [dict(item) for item in chain]
    reordered[4], reordered[5] = reordered[5], reordered[4]
    admission["repair_transition_chain"] = reordered
    with pytest.raises(
        aseh_operator.OperatorError,
        match="R13 repair transition chain differs",
    ):
        aseh_operator._assert_exact_run_launch_admission(
            admission,
            candidate_head=candidate_head,
            candidate_tree=candidate_tree,
        )


def test_aseh_repair_sealed_owner_identity_transition_publication_is_fenced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_head = "3" * 40
    candidate_tree = "4" * 40
    previous_cid = "sha256:" + ("2" * 64)
    witness = {
        "head": candidate_head,
        "tree": candidate_tree,
        "branch_ref": "refs/heads/aseh-r12-fixture",
        "index_entries_digest": "sha256:" + ("5" * 64),
        "index_flags_digest": "sha256:" + ("6" * 64),
        "status_digest": aseh_operator._identity(b""),
        "head_reflog_digest": "sha256:" + ("7" * 64),
        "branch_reflog_digest": "sha256:" + ("8" * 64),
    }
    prior_chain = [
        {"receipt_cid": f"receipt:r{revision}"}
        for revision in range(1, 11)
    ] + [{"receipt_cid": previous_cid}]
    events: list[tuple[str, object]] = []

    def git(*args: str) -> str:
        if args[:3] == ("show", "-s", "--format=%P"):
            return (
                aseh_operator
                .REPAIR_SEALED_OWNER_IDENTITY_TRANSITION_BASE_HEAD
            )
        if args == ("rev-parse", f"{candidate_head}^{{tree}}"):
            return candidate_tree
        if args == (
            "rev-parse",
            aseh_operator.REPAIR_SEALED_OWNER_IDENTITY_TRANSITION_BASE_HEAD
            + "^{tree}",
        ):
            return "9" * 40
        raise AssertionError(args)

    def assert_witness(
        _witness: object,
        *,
        boundary: str,
        **_kwargs: object,
    ) -> None:
        events.append(("witness", boundary))

    def publish(
        _path: Path,
        _receipt: object,
        *,
        authority_directory_fd: int | None = None,
    ) -> None:
        events.append(("publish", authority_directory_fd))

    monkeypatch.setattr(aseh_operator, "_git", git)
    monkeypatch.setattr(
        aseh_operator,
        "_git_changed_paths",
        lambda *_args: (
            aseh_operator.REPAIR_SEALED_OWNER_IDENTITY_TRANSITION_CHANGED_PATHS
        ),
    )
    monkeypatch.setattr(
        aseh_operator,
        "_git_patch_digest",
        lambda *_args: "sha256:" + ("a" * 64),
    )
    monkeypatch.setattr(
        aseh_operator,
        "_candidate_authorization_witness",
        lambda **_kwargs: witness,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_run_repair_sealed_owner_identity_transition_validations",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        aseh_operator,
        "_validate_repair_sealed_owner_identity_transition",
        lambda *_args, **_kwargs: {
            "repair_head": candidate_head,
            "repair_tree": candidate_tree,
            "receipt_cid": "receipt:r12",
        },
    )
    monkeypatch.setattr(
        aseh_operator,
        "_assert_candidate_authorization_witness",
        assert_witness,
    )
    monkeypatch.setattr(aseh_operator, "_atomic_json_create", publish)
    monkeypatch.setattr(
        aseh_operator,
        "_admit_materialized_launch",
        lambda *_args, **_kwargs: pytest.fail(
            "pre-publication admission cannot admit a one-parent R12 repair"
        ),
    )

    result = (
        aseh_operator
        ._authorize_repair_sealed_owner_identity_transition_if_applicable(
            board=object(),
            config={},
            paths={
                "repair_sealed_owner_identity_transition_receipt": (
                    tmp_path / "repair-r12.json"
                )
            },
            bootstrap={
                "plan_root_cid": "plan:sealed",
                "repository_tree_id": "tree:sealed",
            },
            bootstrap_id="bootstrap:sealed",
            head=candidate_head,
            previous_receipt={"receipt_cid": previous_cid},
            previous_transition={
                "repair_head": (
                    aseh_operator
                    .REPAIR_SEALED_OWNER_IDENTITY_TRANSITION_BASE_HEAD
                ),
                "receipt_cid": previous_cid,
            },
            prior_receipt_chain=prior_chain,
            authorization_directory_fd=73,
        )
    )

    assert result is not None
    assert len(result["repair_transition_chain"]) == 12
    assert events[-3:] == [
        ("witness", "immediately before R12 receipt publication"),
        ("publish", 73),
        ("witness", "after R12 receipt publication"),
    ]


def test_aseh_repair_sealed_owner_identity_transition_replay_admits_full_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r12_path = tmp_path / "repair-r12.json"
    r12_path.touch()
    previous_cid = "receipt:r11"
    r12_cid = "receipt:r12"
    repair_head = "3" * 40
    prior_chain = [
        {"receipt_cid": f"receipt:r{revision}"}
        for revision in range(1, 12)
    ]
    receipt = {"repair_head": repair_head, "receipt_cid": r12_cid}
    expected_chain = [*prior_chain, receipt]
    admitted_chain = [
        {"receipt_cid": item["receipt_cid"]} for item in expected_chain
    ]

    monkeypatch.setattr(
        aseh_operator,
        "_secure_runtime_json",
        lambda *_args, **_kwargs: receipt,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_validate_repair_sealed_owner_identity_transition",
        lambda *_args, **_kwargs: {
            "schema": (
                aseh_operator
                .REPAIR_SEALED_OWNER_IDENTITY_TRANSITION_SCHEMA
            ),
            "repair_head": repair_head,
            "repair_tree": "4" * 40,
            "receipt_cid": r12_cid,
        },
    )
    monkeypatch.setattr(aseh_operator, "_git", lambda *_args: "")
    monkeypatch.setattr(
        aseh_operator,
        "_admit_materialized_launch",
        lambda *_args, **_kwargs: {
            "admission_cid": "admission:r12",
            "runtime_source_head": repair_head,
            "repair_transition": {
                "schema": (
                    aseh_operator
                    .REPAIR_SEALED_OWNER_IDENTITY_TRANSITION_SCHEMA
                ),
                "repair_head": repair_head,
                "receipt_cid": r12_cid,
            },
            "repair_transition_chain": admitted_chain,
            "canonical_continuity": {
                "provider_cleanup_fence_to_sealed_owner_identity": {}
            },
        },
    )

    result = (
        aseh_operator
        ._authorize_repair_sealed_owner_identity_transition_if_applicable(
            board=object(),
            config={},
            paths={
                "repair_sealed_owner_identity_transition_receipt": r12_path
            },
            bootstrap={},
            bootstrap_id="bootstrap:sealed",
            head=repair_head,
            previous_receipt={"receipt_cid": previous_cid},
            previous_transition={
                "repair_head": (
                    aseh_operator
                    .REPAIR_SEALED_OWNER_IDENTITY_TRANSITION_BASE_HEAD
                ),
                "receipt_cid": previous_cid,
            },
            prior_receipt_chain=prior_chain,
            authorization_directory_fd=73,
        )
    )

    assert result == {
        "schema": aseh_operator.OPERATOR_SCHEMA,
        "command": "authorize-repair-transition",
        "ok": True,
        "idempotent_replay": True,
        "repair_transition_receipt": receipt,
        "repair_transition_chain": expected_chain,
        "current_admission_cid": "admission:r12",
        "runtime_source_head": repair_head,
    }


def test_aseh_repair_validation_executor_identity_transition_publication_is_fenced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_head = "a" * 40
    candidate_tree = "b" * 40
    previous_cid = "receipt:r12"
    witness = {
        "head": candidate_head,
        "tree": candidate_tree,
        "branch_ref": "refs/heads/aseh-r13-fixture",
        "index_entries_digest": "sha256:" + ("1" * 64),
        "index_flags_digest": "sha256:" + ("2" * 64),
        "status_digest": aseh_operator._identity(b""),
        "head_reflog_digest": "sha256:" + ("3" * 64),
        "branch_reflog_digest": "sha256:" + ("4" * 64),
    }
    prior_chain = [
        {"receipt_cid": f"receipt:r{revision}"}
        for revision in range(1, 12)
    ] + [{"receipt_cid": previous_cid}]
    events: list[tuple[str, object]] = []

    def git(*args: str) -> str:
        if args[:3] == ("show", "-s", "--format=%P"):
            return (
                aseh_operator
                .REPAIR_VALIDATION_EXECUTOR_IDENTITY_TRANSITION_BASE_HEAD
            )
        if args == ("rev-parse", f"{candidate_head}^{{tree}}"):
            return candidate_tree
        if args == (
            "rev-parse",
            aseh_operator
            .REPAIR_VALIDATION_EXECUTOR_IDENTITY_TRANSITION_BASE_HEAD
            + "^{tree}",
        ):
            return "c" * 40
        raise AssertionError(args)

    monkeypatch.setattr(aseh_operator, "_git", git)
    monkeypatch.setattr(
        aseh_operator,
        "_git_changed_paths",
        lambda *_args: (
            aseh_operator
            .REPAIR_VALIDATION_EXECUTOR_IDENTITY_TRANSITION_CHANGED_PATHS
        ),
    )
    monkeypatch.setattr(
        aseh_operator,
        "_git_patch_digest",
        lambda *_args: "sha256:" + ("d" * 64),
    )
    monkeypatch.setattr(
        aseh_operator,
        "_candidate_authorization_witness",
        lambda **_kwargs: witness,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_run_repair_validation_executor_identity_transition_validations",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        aseh_operator,
        "_validate_repair_validation_executor_identity_transition",
        lambda *_args, **_kwargs: {
            "repair_head": candidate_head,
            "repair_tree": candidate_tree,
            "receipt_cid": "receipt:r13",
        },
    )
    monkeypatch.setattr(
        aseh_operator,
        "_assert_candidate_authorization_witness",
        lambda _witness, *, boundary, **_kwargs: events.append(
            ("witness", boundary)
        ),
    )
    monkeypatch.setattr(
        aseh_operator,
        "_atomic_json_create",
        lambda _path, _receipt, *, authority_directory_fd=None: events.append(
            ("publish", authority_directory_fd)
        ),
    )
    monkeypatch.setattr(
        aseh_operator,
        "_admit_materialized_launch",
        lambda *_args, **_kwargs: pytest.fail(
            "pre-publication admission cannot admit a one-parent R13 repair"
        ),
    )

    result = (
        aseh_operator
        ._authorize_repair_validation_executor_identity_transition_if_applicable(
            board=object(),
            config={},
            paths={
                "repair_validation_executor_identity_transition_receipt": (
                    tmp_path / "repair-r13.json"
                )
            },
            bootstrap={
                "plan_root_cid": "plan:sealed",
                "repository_tree_id": "tree:sealed",
            },
            bootstrap_id="bootstrap:sealed",
            head=candidate_head,
            previous_receipt={"receipt_cid": previous_cid},
            previous_transition={
                "repair_head": (
                    aseh_operator
                    .REPAIR_VALIDATION_EXECUTOR_IDENTITY_TRANSITION_BASE_HEAD
                ),
                "receipt_cid": previous_cid,
            },
            prior_receipt_chain=prior_chain,
            authorization_directory_fd=74,
        )
    )

    assert result is not None
    assert len(result["repair_transition_chain"]) == 13
    assert events[-3:] == [
        ("witness", "immediately before R13 receipt publication"),
        ("publish", 74),
        ("witness", "after R13 receipt publication"),
    ]


def test_aseh_repair_validation_executor_identity_transition_replay_admits_full_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r13_path = tmp_path / "repair-r13.json"
    r13_path.touch()
    previous_cid = "receipt:r12"
    r13_cid = "receipt:r13"
    repair_head = "a" * 40
    prior_chain = [
        {"receipt_cid": f"receipt:r{revision}"}
        for revision in range(1, 13)
    ]
    receipt = {"repair_head": repair_head, "receipt_cid": r13_cid}
    expected_chain = [*prior_chain, receipt]
    admitted_chain = [
        {"receipt_cid": item["receipt_cid"]} for item in expected_chain
    ]

    monkeypatch.setattr(
        aseh_operator,
        "_secure_runtime_json",
        lambda *_args, **_kwargs: receipt,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_validate_repair_validation_executor_identity_transition",
        lambda *_args, **_kwargs: {
            "schema": (
                aseh_operator
                .REPAIR_VALIDATION_EXECUTOR_IDENTITY_TRANSITION_SCHEMA
            ),
            "repair_head": repair_head,
            "repair_tree": "b" * 40,
            "receipt_cid": r13_cid,
        },
    )
    monkeypatch.setattr(aseh_operator, "_git", lambda *_args: "")
    monkeypatch.setattr(
        aseh_operator,
        "_admit_materialized_launch",
        lambda *_args, **_kwargs: {
            "admission_cid": "admission:r13",
            "runtime_source_head": repair_head,
            "repair_transition": {
                "schema": (
                    aseh_operator
                    .REPAIR_VALIDATION_EXECUTOR_IDENTITY_TRANSITION_SCHEMA
                ),
                "repair_head": repair_head,
                "receipt_cid": r13_cid,
            },
            "repair_transition_chain": admitted_chain,
            "canonical_continuity": {
                "sealed_owner_identity_to_validation_executor_identity": {}
            },
        },
    )

    result = (
        aseh_operator
        ._authorize_repair_validation_executor_identity_transition_if_applicable(
            board=object(),
            config={},
            paths={
                "repair_validation_executor_identity_transition_receipt": (
                    r13_path
                )
            },
            bootstrap={},
            bootstrap_id="bootstrap:sealed",
            head=repair_head,
            previous_receipt={"receipt_cid": previous_cid},
            previous_transition={
                "repair_head": (
                    aseh_operator
                    .REPAIR_VALIDATION_EXECUTOR_IDENTITY_TRANSITION_BASE_HEAD
                ),
                "receipt_cid": previous_cid,
            },
            prior_receipt_chain=prior_chain,
            authorization_directory_fd=74,
        )
    )

    assert result == {
        "schema": aseh_operator.OPERATOR_SCHEMA,
        "command": "authorize-repair-transition",
        "ok": True,
        "idempotent_replay": True,
        "repair_transition_receipt": receipt,
        "repair_transition_chain": expected_chain,
        "current_admission_cid": "admission:r13",
        "runtime_source_head": repair_head,
    }


def test_aseh_repair_sealed_owner_identity_delegates_only_full_chain_to_r13(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r12_path = tmp_path / "repair-r12.json"
    r12_path.touch()
    r13_path = tmp_path / "repair-r13.json"
    head = "d" * 40
    r12_cid = "sha256:" + ("c" * 64)
    r12_receipt = {"receipt_cid": r12_cid, "repair_head": "b" * 40}
    prior_chain = [
        {"receipt_cid": "sha256:" + f"{revision:064x}"}
        for revision in range(1, 12)
    ]
    delegated: list[dict[str, object]] = []
    sentinel = {"delegated": "r13"}

    monkeypatch.setattr(
        aseh_operator,
        "_secure_runtime_json",
        lambda *_args, **_kwargs: r12_receipt,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_validate_repair_sealed_owner_identity_transition",
        lambda *_args, **_kwargs: {
            "repair_head": "b" * 40,
            "repair_tree": "e" * 40,
            "receipt_cid": r12_cid,
        },
    )
    monkeypatch.setattr(aseh_operator, "_git", lambda *_args: "")

    def delegate(**kwargs: object) -> dict[str, str]:
        delegated.append(dict(kwargs))
        return sentinel

    monkeypatch.setattr(
        aseh_operator,
        "_authorize_repair_validation_executor_identity_transition_if_applicable",
        delegate,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_admit_materialized_launch",
        lambda *_args, **_kwargs: pytest.fail(
            "R12 replay must delegate R13 before current-suffix admission"
        ),
    )

    result = (
        aseh_operator
        ._authorize_repair_sealed_owner_identity_transition_if_applicable(
            board=object(),
            config={},
            paths={
                "repair_sealed_owner_identity_transition_receipt": r12_path,
                "repair_validation_executor_identity_transition_receipt": (
                    r13_path
                ),
            },
            bootstrap={},
            bootstrap_id="bootstrap:sealed",
            head=head,
            previous_receipt={"receipt_cid": prior_chain[-1]["receipt_cid"]},
            previous_transition={
                "repair_head": (
                    aseh_operator
                    .REPAIR_SEALED_OWNER_IDENTITY_TRANSITION_BASE_HEAD
                ),
                "receipt_cid": prior_chain[-1]["receipt_cid"],
            },
            prior_receipt_chain=prior_chain,
            authorization_directory_fd=75,
        )
    )

    assert result is sentinel
    assert len(delegated) == 1
    assert delegated[0]["previous_receipt"] is r12_receipt
    assert delegated[0]["previous_transition"]["receipt_cid"] == r12_cid
    assert len(delegated[0]["prior_receipt_chain"]) == 12
    assert delegated[0]["prior_receipt_chain"][-1] is r12_receipt
    assert delegated[0]["authorization_directory_fd"] == 75


def test_aseh_repair_validation_executor_identity_delegates_full_chain_to_r14(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r13_path = tmp_path / "repair-r13.json"
    r13_path.touch()
    r14_path = tmp_path / "repair-r14.json"
    head = "e" * 40
    r13_cid = "sha256:" + ("d" * 64)
    r13_receipt = {"receipt_cid": r13_cid, "repair_head": "c" * 40}
    prior_chain = [
        {"receipt_cid": "sha256:" + f"{revision:064x}"}
        for revision in range(1, 13)
    ]
    delegated: list[dict[str, object]] = []
    sentinel = {"delegated": "r14"}
    monkeypatch.setattr(
        aseh_operator,
        "_secure_runtime_json",
        lambda *_args, **_kwargs: r13_receipt,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_validate_repair_validation_executor_identity_transition",
        lambda *_args, **_kwargs: {
            "repair_head": "c" * 40,
            "repair_tree": "f" * 40,
            "receipt_cid": r13_cid,
        },
    )
    monkeypatch.setattr(aseh_operator, "_git", lambda *_args: "")

    def delegate(**kwargs: object) -> dict[str, str]:
        delegated.append(dict(kwargs))
        return sentinel

    monkeypatch.setattr(
        aseh_operator,
        "_authorize_repair_sealed_receipt_validation_transition_if_applicable",
        delegate,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_admit_materialized_launch",
        lambda *_args, **_kwargs: pytest.fail(
            "R13 replay must delegate R14 before current-suffix admission"
        ),
    )
    result = (
        aseh_operator
        ._authorize_repair_validation_executor_identity_transition_if_applicable(
            board=object(),
            config={},
            paths={
                "repair_validation_executor_identity_transition_receipt": (
                    r13_path
                ),
                "repair_sealed_receipt_validation_transition_receipt": (
                    r14_path
                ),
            },
            bootstrap={},
            bootstrap_id="bootstrap:sealed",
            head=head,
            previous_receipt={
                "receipt_cid": prior_chain[-1]["receipt_cid"]
            },
            previous_transition={
                "repair_head": (
                    aseh_operator
                    .REPAIR_VALIDATION_EXECUTOR_IDENTITY_TRANSITION_BASE_HEAD
                ),
                "receipt_cid": prior_chain[-1]["receipt_cid"],
            },
            prior_receipt_chain=prior_chain,
            authorization_directory_fd=76,
        )
    )
    assert result is sentinel
    assert len(delegated) == 1
    assert delegated[0]["previous_receipt"] is r13_receipt
    assert delegated[0]["previous_transition"]["receipt_cid"] == r13_cid
    assert len(delegated[0]["prior_receipt_chain"]) == 13
    assert delegated[0]["prior_receipt_chain"][-1] is r13_receipt
    assert delegated[0]["authorization_directory_fd"] == 76


def test_aseh_r13_validation_executor_identity_rejects_merge_parent_before_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior_cid = "sha256:" + ("c" * 64)
    prior_chain = [
        {"receipt_cid": "sha256:" + f"{revision:064x}"}
        for revision in range(1, 12)
    ] + [{"receipt_cid": prior_cid}]
    monkeypatch.setattr(
        aseh_operator,
        "_git",
        lambda *_args: (
            aseh_operator
            .REPAIR_VALIDATION_EXECUTOR_IDENTITY_TRANSITION_BASE_HEAD
            + " "
            + ("f" * 40)
        ),
    )
    monkeypatch.setattr(
        aseh_operator,
        "_run_repair_validation_executor_identity_transition_validations",
        lambda **_kwargs: pytest.fail("merge-parent candidate was validated"),
    )
    monkeypatch.setattr(
        aseh_operator,
        "_atomic_json_create",
        lambda *_args, **_kwargs: pytest.fail("merge-parent receipt published"),
    )

    result = (
        aseh_operator
        ._authorize_repair_validation_executor_identity_transition_if_applicable(
            board=object(),
            config={},
            paths={
                "repair_validation_executor_identity_transition_receipt": (
                    tmp_path / "repair-r13.json"
                )
            },
            bootstrap={},
            bootstrap_id="bootstrap:sealed",
            head="d" * 40,
            previous_receipt={"receipt_cid": prior_cid},
            previous_transition={
                "repair_head": (
                    aseh_operator
                    .REPAIR_VALIDATION_EXECUTOR_IDENTITY_TRANSITION_BASE_HEAD
                ),
                "receipt_cid": prior_cid,
            },
            prior_receipt_chain=prior_chain,
            authorization_directory_fd=75,
        )
    )

    assert result is None


def test_aseh_r14_sealed_receipt_validation_reuses_r11_native_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = {
        "bootstrap_receipt": tmp_path / "bootstrap.json",
        "repair_provider_lease_ownership_transition_receipt": (
            tmp_path / "repair-r10.json"
        ),
        "repair_provider_cleanup_fence_transition_receipt": (
            tmp_path / "repair-r11.json"
        ),
        "repair_sealed_owner_identity_transition_receipt": (
            tmp_path / "repair-r12.json"
        ),
        "repair_validation_executor_identity_transition_receipt": (
            tmp_path / "repair-r13.json"
        ),
        "repair_sealed_receipt_validation_transition_receipt": (
            tmp_path / "repair-r14.json"
        ),
    }
    payloads = {
        path: {"name": name} for name, path in paths.items()
    }
    r11_witness = {"head": "1" * 40}
    r12_head = "3" * 40
    r12_tree = "4" * 40
    candidate_head = r12_head
    candidate_tree = r12_tree
    r12_witness = {"head": r12_head, "tree": r12_tree}
    r13_head = "6" * 40
    r13_tree = "7" * 40
    r13_witness = {"head": r13_head, "tree": r13_tree}
    r14_head = "8" * 40
    r14_tree = "9" * 40
    r14_witness = {"head": r14_head, "tree": r14_tree}
    r11_cid = "sha256:" + ("a" * 64)
    r12_cid = "sha256:" + ("b" * 64)
    r13_cid = "sha256:" + ("c" * 64)
    r14_cid = "sha256:" + ("d" * 64)
    authorization_id = (
        aseh_operator.ASEH_R11_NATIVE_DEPENDENCY_AUTHORIZATION_ID
    )
    authorization = {
        "candidate_authorization_witness": r11_witness,
        "native_dependency_pin": dict(
            aseh_operator.ASEH_R11_NATIVE_DEPENDENCY_PIN
        ),
        "authorization_id": authorization_id,
    }
    observed_witnesses: list[object] = []
    sealed: list[tuple[object, object, str]] = []
    launch = object()

    monkeypatch.setattr(
        aseh_operator,
        "_secure_runtime_json",
        lambda path, **_kwargs: payloads[path],
    )
    monkeypatch.setattr(
        aseh_operator,
        "_validate_repair_provider_cleanup_fence_transition",
        lambda *_args, **_kwargs: {
            "repair_head": (
                aseh_operator
                .REPAIR_SEALED_OWNER_IDENTITY_TRANSITION_BASE_HEAD
            ),
            "repair_tree": "2" * 40,
            "native_dependency_authorization": authorization,
            "receipt_cid": r11_cid,
        },
    )
    monkeypatch.setattr(
        aseh_operator,
        "_validate_repair_sealed_owner_identity_transition",
        lambda *_args, **_kwargs: {
            "schema": (
                aseh_operator
                .REPAIR_SEALED_OWNER_IDENTITY_TRANSITION_SCHEMA
            ),
            "repair_head": r12_head,
            "repair_tree": r12_tree,
            "candidate_authorization_witness": r12_witness,
            "receipt_cid": r12_cid,
        },
    )
    monkeypatch.setattr(
        aseh_operator,
        "_validate_repair_validation_executor_identity_transition",
        lambda *_args, **_kwargs: {
            "schema": (
                aseh_operator
                .REPAIR_VALIDATION_EXECUTOR_IDENTITY_TRANSITION_SCHEMA
            ),
            "repair_head": r13_head,
            "repair_tree": r13_tree,
            "candidate_authorization_witness": r13_witness,
            "receipt_cid": r13_cid,
        },
    )
    monkeypatch.setattr(
        aseh_operator,
        "_validate_repair_sealed_receipt_validation_transition",
        lambda *_args, **_kwargs: {
            "schema": (
                aseh_operator
                .REPAIR_SEALED_RECEIPT_VALIDATION_TRANSITION_SCHEMA
            ),
            "repair_head": r14_head,
            "repair_tree": r14_tree,
            "candidate_authorization_witness": r14_witness,
            "receipt_cid": r14_cid,
        },
    )
    monkeypatch.setattr(
        aseh_operator,
        "_assert_candidate_authorization_witness",
        lambda witness, **_kwargs: observed_witnesses.append(witness),
    )
    monkeypatch.setattr(
        aseh_operator.importlib.util,
        "find_spec",
        lambda _name: SimpleNamespace(origin="/sealed/_duckdb.so"),
    )
    pin = SimpleNamespace(
        as_dict=lambda: dict(aseh_operator.ASEH_R11_NATIVE_DEPENDENCY_PIN)
    )
    monkeypatch.setattr(
        llm_router,
        "inspect_agent_supervisor_native_dependency_source",
        lambda *_args, **_kwargs: pin,
    )

    def seal_native(
        source: object,
        *,
        expected_pin: object,
        accepted_authorization_id: str,
    ) -> object:
        sealed.append(
            (source, expected_pin, accepted_authorization_id)
        )
        return launch

    monkeypatch.setattr(
        llm_router,
        "seal_agent_supervisor_native_dependency",
        seal_native,
    )

    assert aseh_operator._seal_r11_native_dependency(
        paths=paths,
        candidate_head=candidate_head,
        candidate_tree=candidate_tree,
        candidate_authorization_witness=r12_witness,
        launch_admission=None,
    ) is launch
    assert observed_witnesses == [r12_witness, r12_witness]
    assert sealed == [
        (Path("/sealed/_duckdb.so"), pin, authorization_id)
    ]

    assert aseh_operator._seal_r11_native_dependency(
        paths=paths,
        candidate_head=r13_head,
        candidate_tree=r13_tree,
        candidate_authorization_witness=r13_witness,
        launch_admission=None,
    ) is launch
    assert observed_witnesses[-2:] == [r13_witness, r13_witness]

    assert aseh_operator._seal_r11_native_dependency(
        paths=paths,
        candidate_head=r14_head,
        candidate_tree=r14_tree,
        candidate_authorization_witness=r14_witness,
        launch_admission=None,
    ) is launch
    assert observed_witnesses[-2:] == [r14_witness, r14_witness]

    monkeypatch.setattr(aseh_operator, "_git", lambda *_args: "")
    assert aseh_operator._seal_r11_native_dependency(
        paths=paths,
        candidate_head=r13_head,
        candidate_tree=r13_tree,
        candidate_authorization_witness=r13_witness,
        launch_admission=None,
        active_candidate_head=r14_head,
        active_candidate_tree=r14_tree,
        active_candidate_authorization_witness=r14_witness,
    ) is launch
    assert observed_witnesses[-2:] == [r14_witness, r14_witness]

    descendant_head = "e" * 40
    descendant_tree = "f" * 40
    descendant_witness = {
        "head": descendant_head,
        "tree": descendant_tree,
    }
    with pytest.raises(
        aseh_operator.OperatorError,
        match="current candidate is not admitted",
    ):
        aseh_operator._seal_r11_native_dependency(
            paths=paths,
            candidate_head=descendant_head,
            candidate_tree=descendant_tree,
            candidate_authorization_witness=descendant_witness,
            launch_admission=None,
        )

    descendant_chain = [
        {
            "schema": schema,
            "transition_revision": None if index == 0 else index + 1,
            "receipt_cid": "sha256:" + f"{index + 1:064x}",
        }
        for index, schema in enumerate(
            aseh_operator.ASEH_R14_REPAIR_TRANSITION_CHAIN_SCHEMAS
        )
    ]
    descendant_chain[-4]["receipt_cid"] = r11_cid
    descendant_chain[-3]["receipt_cid"] = r12_cid
    descendant_chain[-2].update(
        {
            "receipt_cid": r13_cid,
            "repair_head": r13_head,
            "repair_tree": r13_tree,
        }
    )
    descendant_chain[-1].update(
        {
            "receipt_cid": r14_cid,
            "repair_head": r14_head,
            "repair_tree": r14_tree,
        }
    )
    for index in range(1, len(descendant_chain)):
        descendant_chain[index]["previous_receipt_cid"] = (
            descendant_chain[index - 1]["receipt_cid"]
        )
    descendant_admission = {
        "runtime_source_head": descendant_head,
        "runtime_repository_tree_id": descendant_tree,
        "repair_transition": {
            "schema": (
                aseh_operator
                .REPAIR_SEALED_RECEIPT_VALIDATION_TRANSITION_SCHEMA
            ),
            "repair_head": r14_head,
            "repair_tree": r14_tree,
            "previous_receipt_cid": r13_cid,
            "receipt_cid": r14_cid,
        },
        "repair_transition_chain": descendant_chain,
        "canonical_continuity": {
            "provider_cleanup_fence_to_sealed_owner_identity": {},
            "sealed_owner_identity_to_validation_executor_identity": {},
            "validation_executor_identity_to_sealed_receipt_validation": {},
            "repair_to_current": {},
        },
    }
    assert aseh_operator._seal_r11_native_dependency(
        paths=paths,
        candidate_head=descendant_head,
        candidate_tree=descendant_tree,
        candidate_authorization_witness=descendant_witness,
        launch_admission=descendant_admission,
    ) is launch
    assert observed_witnesses[-2:] == [
        descendant_witness,
        descendant_witness,
    ]
    assert sealed[-1] == (
        Path("/sealed/_duckdb.so"),
        pin,
        authorization_id,
    )

    cid_swapped = dict(descendant_admission)
    swapped_chain = [dict(item) for item in descendant_chain]
    swapped_chain[-3]["receipt_cid"], swapped_chain[-2]["receipt_cid"] = (
        swapped_chain[-2]["receipt_cid"],
        swapped_chain[-3]["receipt_cid"],
    )
    cid_swapped["repair_transition_chain"] = swapped_chain
    with pytest.raises(
        aseh_operator.OperatorError,
        match="current candidate is not admitted",
    ):
        aseh_operator._seal_r11_native_dependency(
            paths=paths,
            candidate_head=descendant_head,
            candidate_tree=descendant_tree,
            candidate_authorization_witness=descendant_witness,
            launch_admission=cid_swapped,
        )

    truncated = dict(descendant_admission)
    truncated["repair_transition_chain"] = descendant_chain[-3:]
    with pytest.raises(
        aseh_operator.OperatorError,
        match="current candidate is not admitted",
    ):
        aseh_operator._seal_r11_native_dependency(
            paths=paths,
            candidate_head=descendant_head,
            candidate_tree=descendant_tree,
            candidate_authorization_witness=descendant_witness,
            launch_admission=truncated,
        )


def test_aseh_repair_provider_cleanup_fence_transition_binds_validation_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repair_head = "3" * 40
    repair_tree = "4" * 40
    base_tree = "2" * 40
    sibling_commits = {
        "ipfs_datasets_py": "6" * 40,
        "ipfs_kit_py": "7" * 40,
    }
    bootstrap = {
        "bootstrap_receipt_id": "bootstrap:sealed",
        "plan_root_cid": "plan:sealed",
        "repository_tree_id": "tree:sealed",
        "source_forest": {
            "by_owner": {
                owner: {"commit": commit}
                for owner, commit in sibling_commits.items()
            }
        },
    }
    candidate_witness = {
        "head": repair_head,
        "tree": repair_tree,
        "branch_ref": "refs/heads/aseh-r11-fixture",
        "index_entries_digest": "sha256:" + ("a" * 64),
        "index_flags_digest": "sha256:" + ("b" * 64),
        "status_digest": aseh_operator._identity(b""),
        "head_reflog_digest": "sha256:" + ("c" * 64),
        "branch_reflog_digest": "sha256:" + ("d" * 64),
    }
    validation_results = [
        {
            "argv": list(command),
            "candidate_head": repair_head,
            "candidate_tree": repair_tree,
            "environment_identity": (
                aseh_operator._r11_command_environment_identity(
                    aseh_operator._r11_validation_environment(
                        Path("/sealed-checkout")
                    ),
                    command,
                    checkout=Path("/sealed-checkout"),
                )
            ),
            "working_tree_scope": (
                aseh_operator._r11_validation_working_tree_scope(command)
            ),
            "returncode": 0,
            "stdout_digest": "sha256:" + ("8" * 64),
            "stderr_digest": "sha256:" + ("9" * 64),
        }
        for command in (
            aseh_operator.REPAIR_PROVIDER_CLEANUP_FENCE_TRANSITION_VALIDATIONS
        )
    ]
    receipt = {
        "stable_identity": (
            f"{aseh_operator.PROGRAM}/"
            f"{aseh_operator.REPAIR_TRANSITION_TASK_ID}@ASEH-PLAN-R11"
        ),
        "previous_receipt_cid": "previous:sealed",
        "bootstrap_receipt_id": "bootstrap:sealed",
        "plan_root_cid": "plan:sealed",
        "repository_tree_id": "tree:sealed",
        "base_head": (
            aseh_operator.REPAIR_PROVIDER_CLEANUP_FENCE_TRANSITION_BASE_HEAD
        ),
        "base_tree": base_tree,
        "repair_head": repair_head,
        "repair_tree": repair_tree,
        "changed_paths": list(
            aseh_operator.REPAIR_PROVIDER_CLEANUP_FENCE_TRANSITION_CHANGED_PATHS
        ),
        "patch_digest": "sha256:" + ("5" * 64),
        "dependencies": ["ASEH-BOOTSTRAP-002@ASEH-PLAN-R10"],
        "owning_repository": "ipfs_accelerate_py",
        "risk_class": "R4_SECURITY_OR_PROTOCOL_SENSITIVE",
        "authority_requirement": (
            aseh_operator.REPAIR_PROVIDER_CLEANUP_FENCE_TRANSITION_AUTHORITY
        ),
        "validation_results": validation_results,
        "known_baseline_failures": [{}],
        "native_dependency_authorization": (
            aseh_operator._r11_native_dependency_authorization(
                candidate_head=repair_head,
                candidate_tree=repair_tree,
                candidate_authorization_witness=candidate_witness,
            )
        ),
        "authorized_at": 1.0,
    }

    monkeypatch.setattr(
        aseh_operator,
        "_repair_provider_cleanup_fence_transition_receipt_id",
        lambda _receipt: "receipt:sealed",
    )
    monkeypatch.setattr(
        aseh_operator,
        "_repair_provider_cleanup_fence_known_baseline_receipt_id",
        lambda _receipt: "baseline:sealed",
    )
    monkeypatch.setattr(
        aseh_operator,
        "_repair_provider_lease_ownership_transition_receipt_id",
        lambda _receipt: "previous:sealed",
    )
    monkeypatch.setattr(
        aseh_operator,
        "_git_changed_paths",
        lambda _base, _repair: (
            aseh_operator.REPAIR_PROVIDER_CLEANUP_FENCE_TRANSITION_CHANGED_PATHS
        ),
    )
    monkeypatch.setattr(
        aseh_operator,
        "_git_patch_digest",
        lambda _base, _repair: "sha256:" + ("5" * 64),
    )

    def fake_git(*arguments: str, **_kwargs: object) -> str:
        if arguments[:3] == ("show", "-s", "--format=%P"):
            return aseh_operator.REPAIR_PROVIDER_CLEANUP_FENCE_TRANSITION_BASE_HEAD
        if arguments == (
            "rev-parse",
            aseh_operator.REPAIR_PROVIDER_CLEANUP_FENCE_TRANSITION_BASE_HEAD
            + "^{tree}",
        ):
            return base_tree
        if arguments == ("rev-parse", repair_head + "^{tree}"):
            return repair_tree
        for owner, commit in sibling_commits.items():
            if arguments == ("rev-parse", f"{repair_head}:{owner}"):
                return commit
        raise AssertionError(f"unexpected git arguments: {arguments!r}")

    monkeypatch.setattr(aseh_operator, "_git", fake_git)

    admitted = aseh_operator._validate_repair_provider_cleanup_fence_transition(
        receipt,
        bootstrap=bootstrap,
        previous_receipt={},
        rerun_validations=False,
    )
    assert admitted["repair_head"] == repair_head
    assert admitted["repair_tree"] == repair_tree

    validation_results[0]["candidate_tree"] = "a" * 40
    with pytest.raises(aseh_operator.OperatorError, match="validation differs"):
        aseh_operator._validate_repair_provider_cleanup_fence_transition(
            receipt,
            bootstrap=bootstrap,
            previous_receipt={},
            rerun_validations=False,
        )


def test_aseh_repair_runtime_hardening_transition_rejects_wrong_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_cid = "sha256:" + ("b" * 64)
    receipt = {
        "schema": aseh_operator.REPAIR_RUNTIME_HARDENING_TRANSITION_SCHEMA,
        "task_id": aseh_operator.REPAIR_TRANSITION_TASK_ID,
        "stable_identity": (
            f"{aseh_operator.PROGRAM}/"
            f"{aseh_operator.REPAIR_TRANSITION_TASK_ID}@ASEH-PLAN-R4"
        ),
        "program_id": aseh_operator.PROGRAM,
        "transition_revision": 4,
        "bootstrap_receipt_id": "sha256:" + ("a" * 64),
        "previous_receipt_cid": "sha256:" + ("c" * 64),
        "plan_root_cid": "plan:sealed",
        "repository_tree_id": "tree:sealed",
        "base_head": (
            aseh_operator.REPAIR_RUNTIME_HARDENING_TRANSITION_BASE_HEAD
        ),
        "base_tree": "2" * 40,
        "repair_head": "3" * 40,
        "repair_tree": "4" * 40,
        "changed_paths": list(
            aseh_operator.REPAIR_RUNTIME_HARDENING_TRANSITION_CHANGED_PATHS
        ),
        "patch_digest": "sha256:" + ("5" * 64),
        "dependencies": ["ASEH-BOOTSTRAP-002@ASEH-PLAN-R3"],
        "owning_repository": "ipfs_accelerate_py",
        "risk_class": "R4_SECURITY_OR_PROTOCOL_SENSITIVE",
        "authority_requirement": (
            "the operator explicitly directed the bootstrap engineering "
            "agent to fix the existing supervisor so it automatically "
            "recovers ASEH false-completion, startup, and shutdown faults "
            "without state-writer or checkout contention"
        ),
        "validation_results": [],
        "terminal_success_criteria": "automatic bounded recovery",
        "terminal_non_success_criteria": "all drift rejected",
        "semantic_corpus_changed": False,
        "database_mutated": False,
        "authorized_at": 1.0,
    }
    receipt["receipt_cid"] = aseh_operator._identity(receipt)
    monkeypatch.setattr(
        aseh_operator,
        "_repair_clean_launch_transition_receipt_id",
        lambda _payload: previous_cid,
    )

    with pytest.raises(aseh_operator.OperatorError, match="authority differs"):
        aseh_operator._validate_repair_runtime_hardening_transition(
            receipt,
            bootstrap={
                "bootstrap_receipt_id": "sha256:" + ("a" * 64),
                "plan_root_cid": "plan:sealed",
                "repository_tree_id": "tree:sealed",
            },
            previous_receipt={},
            rerun_validations=False,
        )


def test_aseh_repair_clean_launch_transition_rejects_rehashed_wrong_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_cid = "sha256:" + ("b" * 64)
    receipt = {
        "schema": aseh_operator.REPAIR_CLEAN_LAUNCH_TRANSITION_SCHEMA,
        "task_id": aseh_operator.REPAIR_TRANSITION_TASK_ID,
        "stable_identity": (
            f"{aseh_operator.PROGRAM}/"
            f"{aseh_operator.REPAIR_TRANSITION_TASK_ID}@ASEH-PLAN-R3"
        ),
        "program_id": aseh_operator.PROGRAM,
        "transition_revision": 3,
        "bootstrap_receipt_id": "sha256:" + ("a" * 64),
        "previous_receipt_cid": "sha256:" + ("c" * 64),
        "plan_root_cid": "plan:sealed",
        "repository_tree_id": "tree:sealed",
        "base_head": aseh_operator.REPAIR_CLEAN_LAUNCH_TRANSITION_BASE_HEAD,
        "base_tree": "2" * 40,
        "repair_head": "3" * 40,
        "repair_tree": "4" * 40,
        "changed_paths": list(
            aseh_operator.REPAIR_CLEAN_LAUNCH_TRANSITION_CHANGED_PATHS
        ),
        "patch_digest": "sha256:" + ("5" * 64),
        "dependencies": ["ASEH-BOOTSTRAP-002@ASEH-PLAN-R2"],
        "owning_repository": "ipfs_accelerate_py",
        "risk_class": "R4_SECURITY_OR_PROTOCOL_SENSITIVE",
        "authority_requirement": (
            "the operator explicitly directed the bootstrap engineering "
            "agent to fix the existing supervisor so admission validation "
            "cannot materialize credentials in or dirty the launch checkout"
        ),
        "validation_results": [],
        "terminal_success_criteria": "validation leaves checkout clean",
        "terminal_non_success_criteria": "all drift rejected",
        "semantic_corpus_changed": False,
        "database_mutated": False,
        "authorized_at": 1.0,
    }
    receipt["receipt_cid"] = aseh_operator._identity(receipt)
    monkeypatch.setattr(
        aseh_operator,
        "_repair_followup_transition_receipt_id",
        lambda _payload: previous_cid,
    )

    with pytest.raises(aseh_operator.OperatorError, match="authority differs"):
        aseh_operator._validate_repair_clean_launch_transition(
            receipt,
            bootstrap={
                "bootstrap_receipt_id": "sha256:" + ("a" * 64),
                "plan_root_cid": "plan:sealed",
                "repository_tree_id": "tree:sealed",
            },
            previous_receipt={},
            rerun_validations=False,
        )


def test_aseh_repair_clean_launch_transition_publication_is_create_only(
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / "repair-r3.json"
    first = {"revision": 3, "receipt_cid": "sha256:" + ("1" * 64)}
    replacement = {"revision": 3, "receipt_cid": "sha256:" + ("2" * 64)}

    aseh_operator._atomic_json_create(receipt_path, first)
    original = receipt_path.read_bytes()
    with pytest.raises(aseh_operator.OperatorError, match="already exists"):
        aseh_operator._atomic_json_create(receipt_path, replacement)

    assert receipt_path.read_bytes() == original
    assert receipt_path.stat().st_mode & 0o777 == 0o600
    assert aseh_operator._secure_runtime_json(
        receipt_path, max_bytes=4096
    ) == first


def test_aseh_repair_provider_cleanup_fence_transition_publication_race_is_create_only(
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / "repair-r11.json"
    payloads = [
        {"revision": 11, "receipt_cid": "sha256:" + (token * 64)}
        for token in ("1", "2")
    ]
    barrier = threading.Barrier(2)
    admitted: list[dict[str, object]] = []
    rejected: list[BaseException] = []
    result_lock = threading.Lock()

    def publish(payload: dict[str, object]) -> None:
        try:
            barrier.wait(timeout=2.0)
            aseh_operator._atomic_json_create(receipt_path, payload)
            with result_lock:
                admitted.append(payload)
        except BaseException as exc:
            with result_lock:
                rejected.append(exc)

    threads = [
        threading.Thread(target=publish, args=(payload,))
        for payload in payloads
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)

    assert all(not thread.is_alive() for thread in threads)
    assert len(admitted) == 1
    assert len(rejected) == 1
    assert isinstance(rejected[0], aseh_operator.OperatorError)
    assert aseh_operator._secure_runtime_json(
        receipt_path,
        max_bytes=4096,
    ) == admitted[0]


def test_aseh_repair_provider_cleanup_fence_transition_public_name_cannot_remint(
    tmp_path: Path,
) -> None:
    target = tmp_path / "unrelated.json"
    target.write_text("unchanged", encoding="utf-8")
    receipt_path = tmp_path / "repair-r11.json"
    receipt_path.symlink_to(target)

    with pytest.raises(aseh_operator.OperatorError, match="already exists"):
        aseh_operator._atomic_json_create(
            receipt_path,
            {"revision": 11},
        )

    assert target.read_text(encoding="utf-8") == "unchanged"
    assert receipt_path.is_symlink()


def test_aseh_repair_provider_cleanup_fence_transition_parent_swap_cannot_remint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_directory = tmp_path / "bootstrap"
    authority_directory.mkdir()
    displaced_directory = tmp_path / "bootstrap.displaced"
    receipt_path = authority_directory / "repair-r11.json"
    original_rename = aseh_operator._rename_noreplace
    swapped = False

    def swap_parent_then_publish(
        directory_fd: int,
        source_name: str,
        target_name: str,
    ) -> None:
        nonlocal swapped
        assert swapped is False
        os.replace(authority_directory, displaced_directory)
        authority_directory.mkdir()
        swapped = True
        original_rename(directory_fd, source_name, target_name)

    monkeypatch.setattr(
        aseh_operator,
        "_rename_noreplace",
        swap_parent_then_publish,
    )

    with pytest.raises(aseh_operator.OperatorError, match="directory changed"):
        with aseh_operator._anchored_directory_descriptor(
            authority_directory
        ) as authority_directory_fd:
            aseh_operator._atomic_json_create(
                receipt_path,
                {"revision": 11},
                authority_directory_fd=authority_directory_fd,
            )

    assert swapped is True
    assert receipt_path.exists() is False
    assert aseh_operator._secure_runtime_json(
        displaced_directory / receipt_path.name,
        max_bytes=4096,
    ) == {"revision": 11}


@pytest.mark.parametrize(
    ("sealed_transition", "move_after_validation", "expected_error"),
    (
        (False, False, "lacks its exact admitted validation seal"),
        (True, True, "candidate moved after validation"),
    ),
)
def test_aseh_repair_provider_cleanup_fence_transition_preflight_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sealed_transition: bool,
    move_after_validation: bool,
    expected_error: str,
) -> None:
    bootstrap_path = tmp_path / "bootstrap.json"
    database_path = tmp_path / "state.duckdb"
    bootstrap_path.write_bytes(b"{}")
    database_path.write_bytes(b"database")
    board = object()
    candidate_head = "3" * 40
    candidate_tree = "4" * 40
    transition = {
        "schema": (
            aseh_operator.REPAIR_PROVIDER_CLEANUP_FENCE_TRANSITION_SCHEMA
            if sealed_transition
            else "unsealed"
        ),
        "repair_head": candidate_head,
        "repair_tree": candidate_tree,
        "receipt_cid": "receipt:sealed",
    }
    admission = {
        "admission_cid": "admission:sealed",
        "runtime_source_head": candidate_head,
        "runtime_repository_tree_id": candidate_tree,
        "repair_transition": transition,
    }
    monkeypatch.setattr(
        configured_scheduler,
        "preflight_configured_board",
        lambda _board: {"valid": True, "errors": []},
    )
    monkeypatch.setattr(
        aseh_operator,
        "_load",
        lambda _path: (board, {}),
    )
    monkeypatch.setattr(
        aseh_operator,
        "_paths",
        lambda _board: {
            "bootstrap_receipt": bootstrap_path,
            "database": database_path,
        },
    )
    monkeypatch.setattr(
        aseh_operator,
        "_candidate_authorization_witness",
        lambda **_kwargs: {"epoch": "stable"},
    )
    monkeypatch.setattr(
        aseh_operator,
        "_admit_materialized_launch",
        lambda *_args, **_kwargs: admission,
    )

    def fake_git(*arguments: str, **_kwargs: object) -> str:
        if arguments == ("rev-parse", "HEAD"):
            return candidate_head
        if arguments == ("rev-parse", "HEAD^{tree}"):
            return candidate_tree
        if arguments == ("show", "-s", "--format=%P", candidate_head):
            return aseh_operator.REPAIR_PROVIDER_CLEANUP_FENCE_TRANSITION_BASE_HEAD
        raise AssertionError(f"unexpected git arguments: {arguments!r}")

    monkeypatch.setattr(aseh_operator, "_git", fake_git)

    def assert_witness(*_args: object, **_kwargs: object) -> None:
        if move_after_validation:
            raise aseh_operator.OperatorError("candidate moved after validation")

    monkeypatch.setattr(
        aseh_operator,
        "_assert_candidate_authorization_witness",
        assert_witness,
    )

    returncode, report = aseh_operator.preflight(tmp_path / "config.json")

    assert returncode == 1
    assert report["valid"] is False
    assert report["sealed_launch_admission"]["admitted"] is False
    assert expected_error in report["errors"][-1]


def test_aseh_repair_provider_cleanup_fence_transition_lock_replacement_contends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / "bootstrap" / ".authorize-repair-transition.lock"
    paths = {"repair_transition_authorization_lock": lock_path}
    monkeypatch.setattr(
        aseh_operator,
        "AUTHORIZATION_TRANSITION_LOCK_TIMEOUT_SECONDS",
        0.05,
    )

    with pytest.raises(aseh_operator.OperatorError, match="lock name changed"):
        with aseh_operator._repair_transition_authorization_guard(paths):
            displaced = lock_path.with_name(lock_path.name + ".displaced")
            os.replace(lock_path, displaced)
            lock_path.write_bytes(b"")
            lock_path.chmod(0o600)
            with pytest.raises(
                aseh_operator.OperatorError,
                match="authorization_contended",
            ):
                with aseh_operator._repair_transition_authorization_guard(paths):
                    pytest.fail("replacement name acquired a second authority")


def test_aseh_repair_provider_cleanup_fence_transition_witness_rejects_movement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=repository,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        assert completed.returncode == 0, completed.stderr
        return completed.stdout.strip()

    git("init", "--quiet", "--initial-branch=aseh")
    git("config", "user.name", "ASEH Test")
    git("config", "user.email", "aseh@example.invalid")
    tracked = repository / "tracked.txt"
    tracked.write_text("one\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "--quiet", "-m", "one")
    first_head = git("rev-parse", "HEAD")
    first_tree = git("rev-parse", "HEAD^{tree}")
    monkeypatch.setattr(aseh_operator, "ROOT", repository)
    witness = aseh_operator._candidate_authorization_witness(
        expected_head=first_head,
        expected_tree=first_tree,
    )

    tracked.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(aseh_operator.OperatorError, match="dirty or moving"):
        aseh_operator._assert_candidate_authorization_witness(
            witness,
            expected_head=first_head,
            expected_tree=first_tree,
            boundary="post-validation dirtiness",
        )
    tracked.write_text("two\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "--quiet", "-m", "two")
    git("reset", "--hard", "--quiet", first_head)
    assert git("status", "--porcelain=v1", "--untracked-files=all") == ""
    assert git("rev-parse", "HEAD") == first_head
    assert git("rev-parse", "HEAD^{tree}") == first_tree

    with pytest.raises(
        aseh_operator.OperatorError,
        match="authorization boundary",
    ):
        aseh_operator._assert_candidate_authorization_witness(
            witness,
            expected_head=first_head,
            expected_tree=first_tree,
            boundary="clean HEAD moved away and back",
        )


@pytest.mark.parametrize(
    "index_flag",
    ("--assume-unchanged", "--skip-worktree"),
)
def test_aseh_repair_provider_cleanup_fence_transition_witness_rejects_hidden_index_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    index_flag: str,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=repository,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        assert completed.returncode == 0, completed.stderr
        return completed.stdout.strip()

    git("init", "--quiet", "--initial-branch=aseh")
    git("config", "user.name", "ASEH Test")
    git("config", "user.email", "aseh@example.invalid")
    tracked = repository / "operator.py"
    tracked.write_text("sealed = True\n", encoding="utf-8")
    git("add", "operator.py")
    git("commit", "--quiet", "-m", "sealed candidate")
    candidate_head = git("rev-parse", "HEAD")
    candidate_tree = git("rev-parse", "HEAD^{tree}")
    git("update-index", index_flag, "operator.py")
    tracked.write_text("sealed = False\n", encoding="utf-8")
    assert git("status", "--porcelain=v1", "--untracked-files=all") == ""

    monkeypatch.setattr(aseh_operator, "ROOT", repository)
    with pytest.raises(
        aseh_operator.OperatorError,
        match="exceptional tracked entry",
    ):
        aseh_operator._candidate_authorization_witness(
            expected_head=candidate_head,
            expected_tree=candidate_tree,
        )


@pytest.mark.parametrize(
    "index_flag",
    ("--assume-unchanged", "--skip-worktree"),
)
def test_aseh_direct_run_clean_gate_rejects_hidden_mutated_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    index_flag: str,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ("/usr/bin/git", *arguments),
            cwd=repository,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        assert completed.returncode == 0, completed.stderr
        return completed.stdout.strip()

    git("init", "--quiet", "--initial-branch=aseh")
    git("config", "user.name", "ASEH Test")
    git("config", "user.email", "aseh@example.invalid")
    runtime = repository / "runtime.py"
    runtime.write_text("SEALED = True\n", encoding="utf-8")
    git("add", "runtime.py")
    git("commit", "--quiet", "-m", "sealed runtime")
    git("update-index", index_flag, "runtime.py")
    runtime.write_text("SEALED = False\n", encoding="utf-8")
    assert git("status", "--porcelain=v1", "--untracked-files=all") == ""

    monkeypatch.setattr(aseh_operator, "ROOT", repository)
    with pytest.raises(aseh_operator.OperatorError, match="exceptional tracked entry"):
        aseh_operator._assert_clean_tree(
            SimpleNamespace(merge_target_branch="aseh")
        )


def test_aseh_preseal_git_ignores_forged_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(
        ("/usr/bin/git", "init", "--quiet", "--initial-branch=aseh"),
        cwd=repository,
        check=True,
    )
    hostile = tmp_path / "hostile"
    hostile.mkdir()
    sentinel = tmp_path / "forged-git-executed"
    forged = hostile / "git"
    forged.write_text(
        "#!/bin/sh\ntouch " + str(sentinel) + "\nexit 99\n",
        encoding="utf-8",
    )
    forged.chmod(0o755)
    monkeypatch.setenv("PATH", str(hostile))
    monkeypatch.setattr(aseh_operator, "ROOT", repository)

    assert aseh_operator._git("rev-parse", "--show-toplevel") == str(repository)
    assert not sentinel.exists()


def test_aseh_preseal_git_rejects_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aseh_operator, "_TRUSTED_GIT_IDENTITY", (0,))
    with pytest.raises(aseh_operator.OperatorError, match="identity drifted"):
        aseh_operator._trusted_git_executable()


def test_aseh_r11_validation_environment_rejects_ambient_startup_and_plugins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hostile = {
        "PYTEST_ADDOPTS": "--collect-only",
        "PYTEST_PLUGINS": "hostile_plugin",
        "PYTHONPATH": str(tmp_path / "hostile-python"),
        "PYTHONHOME": str(tmp_path / "hostile-home"),
        "PYTHONSTARTUP": str(tmp_path / "startup.py"),
        "PYTHONINSPECT": "1",
        "PYTHONWARNINGS": "ignore",
        "LD_PRELOAD": str(tmp_path / "hostile.so"),
        "LD_LIBRARY_PATH": str(tmp_path / "hostile-lib"),
        "DYLD_INSERT_LIBRARIES": str(tmp_path / "hostile.dylib"),
    }
    for name, value in hostile.items():
        monkeypatch.setenv(name, value)
    checkout = tmp_path / "exact-checkout"
    checkout.mkdir()

    environment = aseh_operator._r11_validation_environment(checkout)

    assert environment["PYTHONPATH"] == str(checkout)
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert not (set(hostile) - {"PYTHONPATH"}).intersection(environment)
    assert aseh_operator._r11_validation_environment_identity(
        environment,
        checkout=checkout,
    ).startswith("sha256:")


def test_aseh_r11_validation_environment_routes_only_board_check_to_launch_tree(
) -> None:
    commands = (
        aseh_operator.REPAIR_PROVIDER_CLEANUP_FENCE_TRANSITION_VALIDATIONS
    )
    assert [
        aseh_operator._r11_validation_working_tree_scope(command)
        for command in commands
    ].count("candidate_authorization_worktree") == 1
    assert (
        aseh_operator._r11_validation_working_tree_scope(commands[-2])
        == "candidate_authorization_worktree"
    )
    assert all(
        aseh_operator._r11_validation_working_tree_scope(command)
        == "immutable_candidate_checkout"
        for index, command in enumerate(commands)
        if index != len(commands) - 2
    )


def test_aseh_r11_validation_environment_executes_board_check_in_launch_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    immutable_checkout = tmp_path / "immutable-candidate"
    authorization_worktree = tmp_path / "authorization-worktree"
    immutable_checkout.mkdir()
    authorization_worktree.mkdir()
    environment = aseh_operator._r11_validation_environment(
        immutable_checkout
    )
    calls: list[tuple[tuple[str, ...], Path]] = []

    monkeypatch.setattr(aseh_operator, "ROOT", authorization_worktree)
    monkeypatch.setattr(
        aseh_operator,
        "_exact_candidate_validation_checkout",
        lambda **_kwargs: nullcontext((immutable_checkout, environment)),
    )
    monkeypatch.setattr(
        aseh_operator,
        "_assert_candidate_authorization_witness",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_assert_r11_validation_checkout_identity",
        lambda *_args, **_kwargs: None,
    )

    def fake_run(
        command: tuple[str, ...],
        **kwargs: object,
    ) -> SimpleNamespace:
        calls.append((tuple(command), Path(str(kwargs["cwd"]))))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(aseh_operator, "_run", fake_run)
    results = (
        aseh_operator
        ._run_repair_provider_cleanup_fence_transition_validations(
            candidate_head="1" * 40,
            candidate_tree="2" * 40,
            authorization_witness={"sealed": "witness"},
        )
    )

    assert len(calls) == len(
        aseh_operator.REPAIR_PROVIDER_CLEANUP_FENCE_TRANSITION_VALIDATIONS
    )
    assert calls[-2][1] == authorization_worktree
    assert all(
        cwd == immutable_checkout
        for index, (_command, cwd) in enumerate(calls)
        if index != len(calls) - 2
    )
    assert results[-2]["working_tree_scope"] == (
        "candidate_authorization_worktree"
    )
    assert all(
        result["working_tree_scope"] == "immutable_candidate_checkout"
        for index, result in enumerate(results)
        if index != len(results) - 2
    )


def test_aseh_sealed_owner_loads_operator_only_from_verified_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    operator_path = repository / configured_scheduler.ASEH_SEALED_OWNER_OPERATOR
    operator_path.parent.mkdir(parents=True)
    sentinel = tmp_path / "live-operator-executed"
    operator_path.write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).touch()\n",
        encoding="utf-8",
    )
    config = repository / "config.json"
    config.write_text("{}", encoding="utf-8")
    qualification_home = tmp_path / "qualification-homes" / "fixture"
    qualification_home.mkdir(parents=True)
    sealed_source = (
        "def _run_supervisor_owner(config_path, **kwargs):\n"
        "    assert kwargs['sealed_control_plane_descriptor'] >= 3\n"
        "    assert kwargs['retained_interpreter']['descriptor'] >= 3\n"
        "    assert kwargs['native_dependency_launch']['accepted'] is True\n"
        "    assert kwargs['system_dependency_directories_json'] == '[]'\n"
        f"    assert kwargs['sealed_owner_environment']['HOME'] == {str(qualification_home)!r}\n"
        "    return 73\n"
    ).encode("utf-8")
    manifest = {
        "capsule_id": "sha256:" + ("1" * 64),
        "source_head": "2" * 40,
        "source_tree": "3" * 40,
        "files": {
            configured_scheduler.ASEH_SEALED_OWNER_OPERATOR: (
                "sha256:" + hashlib.sha256(sealed_source).hexdigest()
            )
        },
    }
    archive_path = tmp_path / "capsule.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            ".agent-control-plane-manifest.json",
            json.dumps(manifest, sort_keys=True),
        )
        archive.writestr(
            configured_scheduler.ASEH_SEALED_OWNER_OPERATOR,
            sealed_source,
        )
    capsule_descriptor = os.open(archive_path, os.O_RDONLY)
    interpreter = multi_runner.retain_control_plane_interpreter(sys.executable)
    pin = SimpleNamespace(
        capsule_id=manifest["capsule_id"],
        source_head=manifest["source_head"],
        source_tree=manifest["source_tree"],
        as_dict=lambda: {"sealed": True},
    )
    monkeypatch.setattr(
        configured_scheduler,
        "parse_accepted_control_plane_pin",
        lambda _value: pin,
    )
    monkeypatch.setattr(
        configured_scheduler,
        "verify_agent_implementation_sealed_control_plane",
        lambda _pin, descriptor: f"/proc/self/fd/{descriptor}",
    )
    native_json = '{"accepted":true}'
    native_dependency = SimpleNamespace(
        descriptor=SimpleNamespace(descriptor=92),
        to_json=lambda: native_json,
        as_dict=lambda: {"accepted": True},
    )
    monkeypatch.setattr(
        configured_scheduler,
        "parse_agent_supervisor_native_dependency_launch",
        lambda _value: native_dependency,
    )
    monkeypatch.setattr(
        configured_scheduler,
        "verify_agent_supervisor_native_dependency_sealed_fd",
        lambda _launch: "/proc/self/fd/92",
    )
    monkeypatch.setattr(
        configured_scheduler,
        "admit_trusted_system_dependency_directories",
        lambda _value: (),
    )
    for name in (
        "IPFS_ACCELERATE_AGENT_QUACK_TOKEN",
        "IPFS_ACCELERATE_AGENT_OWNER_STATE_TOKEN",
        "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD",
        "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SOCKET",
        "IPFS_ACCELERATE_AGENT_STATE_OWNER_SOCKET",
        "IPFS_ACCELERATE_AGENT_STATE_GRANT_HANDOFF_ADDRESS",
        "IPFS_ACCELERATE_AGENT_STATE_GRANT_HANDOFF_PARENT_PID",
        "IPFS_ACCELERATE_AGENT_STATE_GRANT_HANDOFF_PARENT_START",
        "IPFS_ACCELERATE_AGENT_STATE_GRANT_HANDOFF_BOOT_ID",
        "IPFS_ACCELERATE_AGENT_STATE_GRANT_HANDOFF_PARENT_LOSS_POLICY",
    ):
        monkeypatch.delenv(name, raising=False)
    exact_environment = aseh_operator._sealed_owner_delegation_environment(
        qualification_home
    )
    for name in tuple(os.environ):
        monkeypatch.delenv(name, raising=False)
    for name, value in exact_environment.items():
        monkeypatch.setenv(name, value)
    native_alias = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "_duckdb", native_alias)
    monkeypatch.setitem(sys.modules, "duckdb", native_alias)
    parent_fences: list[dict[str, object]] = []
    monkeypatch.setattr(
        process_security_module,
        "arm_state_authority_parent_death_signal",
        lambda **kwargs: parent_fences.append(dict(kwargs)),
    )
    try:
        result = configured_scheduler._run_aseh_sealed_owner(
            [
                configured_scheduler.ASEH_SEALED_OWNER_MARKER,
                "sealed-pin",
                str(capsule_descriptor),
                str(interpreter.descriptor),
                interpreter.argv0,
                interpreter.sha256,
                str(repository),
                str(config),
                "1",
                "1.0",
                aseh_operator._sealed_owner_delegation_environment_identity(
                    qualification_home
                ),
                "92",
                native_json,
                "[]",
                str(qualification_home),
                "123",
                "456",
                "fixture-boot",
            ]
        )
    finally:
        os.close(interpreter.descriptor)
        os.close(capsule_descriptor)
    assert result == 73
    assert (
        aseh_operator._sealed_owner_delegation_environment_identity(
            qualification_home
        )
        == configured_scheduler._identity(exact_environment)
    )
    assert (
        aseh_operator._sealed_owner_delegation_environment_identity(
            qualification_home
        )
        != aseh_operator._identity(exact_environment)
    )
    assert parent_fences == [
        {
            "expected_parent_pid": 123,
            "expected_parent_start_time_ticks": 456,
            "expected_boot_id": "fixture-boot",
        }
    ]
    assert not sentinel.exists()


def test_aseh_failed_sealed_delegation_closes_fds_without_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py import llm_router

    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    board = SimpleNamespace(config_path=config)
    candidate_head = "1" * 40
    candidate_tree = "2" * 40
    qualification_home = tmp_path / "qualification-homes" / "fixture"
    qualification_home.mkdir(parents=True)
    opened: list[int] = []
    capsule_parents: list[Path] = []
    native_launch_admissions: list[object] = []
    ordering: list[str] = []
    pin = SimpleNamespace(
        source_head=candidate_head,
        source_tree=candidate_tree,
        capsule_root=str(tmp_path / "unused-capsule"),
    )

    def materialize(**kwargs: object) -> object:
        ordering.append("capsule")
        capsule_parents.append(Path(str(kwargs["capsule_parent"])))
        pin.capsule_root = str(capsule_parents[-1] / "capsule")
        return pin

    def seal(_pin: object) -> object:
        path = tmp_path / "sealed.zip"
        path.write_bytes(b"sealed")
        descriptor = os.open(path, os.O_RDONLY)
        opened.append(descriptor)
        return SimpleNamespace(descriptor=descriptor)

    def retain(_python: str) -> object:
        descriptor = os.open(sys.executable, os.O_RDONLY)
        opened.append(descriptor)
        return SimpleNamespace(
            descriptor=descriptor,
            argv0=str(Path(sys.executable).resolve()),
            sha256="sha256:" + ("3" * 64),
            executable_path=f"/proc/self/fd/{descriptor}",
        )

    def seal_native(**kwargs: object) -> object:
        ordering.append("native_seal")
        native_launch_admissions.append(kwargs.get("launch_admission"))
        path = tmp_path / "sealed-native.so"
        path.write_bytes(b"sealed-native")
        descriptor = os.open(path, os.O_RDONLY)
        opened.append(descriptor)
        return SimpleNamespace(
            descriptor=SimpleNamespace(descriptor=descriptor),
            accepted_authorization_id="sha256:" + ("4" * 64),
            to_json=lambda: '{"native":"sealed"}',
        )

    monkeypatch.setattr(aseh_operator, "_load", lambda _path: (board, {}))
    monkeypatch.setattr(
        aseh_operator,
        "_assert_clean_tree",
        lambda _board: (candidate_head, candidate_tree),
    )
    monkeypatch.setattr(
        aseh_operator,
        "_candidate_authorization_witness",
        lambda **_kwargs: {"stable": "yes"},
    )
    monkeypatch.setattr(
        aseh_operator,
        "_git",
        lambda *_args: "f" * 40,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_assert_candidate_authorization_witness",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_build_server",
        lambda *_args, **_kwargs: pytest.fail("delegate acquired an owner"),
    )
    monkeypatch.setattr(
        llm_router,
        "materialize_agent_implementation_control_plane_capsule",
        materialize,
    )
    monkeypatch.setattr(
        llm_router,
        "seal_agent_implementation_control_plane_capsule",
        seal,
    )
    monkeypatch.setattr(multi_runner, "retain_control_plane_interpreter", retain)
    monkeypatch.setattr(
        aseh_operator,
        "_seal_r11_native_dependency",
        seal_native,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_r14_native_seal_anchor",
        lambda *_args, **_kwargs: {
            "head": "a" * 40,
            "tree": "b" * 40,
            "witness": {"anchor": "r14"},
        },
    )

    @contextmanager
    def validation_scope(**_kwargs: object) -> object:
        ordering.append("executor_enter")
        yield None
        ordering.append("executor_exit")

    monkeypatch.setattr(
        aseh_operator,
        "_sealed_receipt_validation_executor_scope",
        validation_scope,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_admit_materialized_launch",
        lambda *_args, **_kwargs: ordering.append("admission") or {},
    )
    monkeypatch.setattr(
        aseh_operator,
        "_assert_exact_run_launch_admission",
        lambda *_args, **_kwargs: ordering.append("assert_admission"),
    )
    monkeypatch.setattr(aseh_operator, "_paths", lambda _board: {})
    monkeypatch.setattr(
        aseh_operator,
        "_build_aseh_qualification_home",
        lambda _paths: qualification_home,
    )
    monkeypatch.setattr(
        multi_runner,
        "trusted_system_dependency_directories_json",
        lambda: "[]",
    )
    monkeypatch.setattr(
        multi_runner,
        "accepted_control_plane_pin_json",
        lambda _pin: "{}",
    )
    monkeypatch.setattr(
        multi_runner,
        "build_sealed_control_plane_module_command",
        lambda **_kwargs: [sys.executable, "-c", "pass"],
    )
    monkeypatch.setattr(
        configured_scheduler,
        "_cleanup_plan_bound_control_plane",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        aseh_operator.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("injected delegation failure")
        ),
    )
    for name in (
        "IPFS_ACCELERATE_AGENT_QUACK_TOKEN",
        "IPFS_ACCELERATE_AGENT_OWNER_STATE_TOKEN",
        "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD",
        "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SOCKET",
        "IPFS_ACCELERATE_AGENT_STATE_OWNER_SOCKET",
        "IPFS_ACCELERATE_AGENT_STATE_GRANT_HANDOFF_ADDRESS",
        "IPFS_ACCELERATE_AGENT_STATE_GRANT_HANDOFF_PARENT_PID",
        "IPFS_ACCELERATE_AGENT_STATE_GRANT_HANDOFF_PARENT_START",
        "IPFS_ACCELERATE_AGENT_STATE_GRANT_HANDOFF_BOOT_ID",
        "IPFS_ACCELERATE_AGENT_STATE_GRANT_HANDOFF_PARENT_LOSS_POLICY",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(OSError, match="delegation failure"):
        aseh_operator.run_supervisor(config, implement=True, duration=1.0)

    assert len(opened) == 3
    assert native_launch_admissions == [None]
    assert ordering == [
        "native_seal",
        "executor_enter",
        "admission",
        "assert_admission",
        "executor_exit",
        "capsule",
    ]
    for descriptor in opened:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert capsule_parents and not capsule_parents[0].exists()


def test_aseh_sealed_owner_rechecks_candidate_before_owner_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.json"
    bootstrap = tmp_path / "bootstrap.json"
    database = tmp_path / "control.duckdb"
    for path in (config, bootstrap, database):
        path.write_text("{}", encoding="utf-8")
    candidate_head = "4" * 40
    candidate_tree = "5" * 40
    board = SimpleNamespace(config_path=config)
    pin = SimpleNamespace(
        source_head=candidate_head,
        source_tree=candidate_tree,
        capsule_root=str(tmp_path / "capsule"),
    )
    interpreter = SimpleNamespace(
        descriptor=91,
        argv0="/usr/bin/python3.12",
        sha256="sha256:" + ("6" * 64),
        executable_path="/proc/self/fd/91",
    )
    server = SimpleNamespace(
        start=lambda: pytest.fail("moving candidate acquired owner authority"),
        stop=lambda: {"stopped": True},
    )
    native_dependency = SimpleNamespace(
        descriptor=SimpleNamespace(descriptor=92),
        accepted_authorization_id="sha256:" + ("8" * 64),
        pin=SimpleNamespace(
            as_dict=lambda: dict(aseh_operator.ASEH_R11_NATIVE_DEPENDENCY_PIN)
        ),
    )
    ordering: list[str] = []

    @contextmanager
    def validation_scope(**_kwargs: object) -> object:
        ordering.append("executor_enter")
        yield None
        ordering.append("executor_exit")

    monkeypatch.setattr(aseh_operator, "_load", lambda _path: (board, {}))
    monkeypatch.setattr(
        aseh_operator,
        "_assert_clean_tree",
        lambda _board: (candidate_head, candidate_tree),
    )
    monkeypatch.setattr(
        aseh_operator,
        "_candidate_authorization_witness",
        lambda **_kwargs: {"stable": "yes"},
    )
    monkeypatch.setattr(
        aseh_operator,
        "_paths",
        lambda _board: {
            "bootstrap_receipt": bootstrap,
            "database": database,
        },
    )
    monkeypatch.setattr(
        configured_scheduler,
        "preflight_configured_board",
        lambda _board: ordering.append("preflight")
        or {"valid": True, "errors": []},
    )
    monkeypatch.setattr(
        aseh_operator,
        "_admit_materialized_launch",
        lambda *_args: ordering.append("admission") or {
            "runtime_source_head": candidate_head,
            "runtime_repository_tree_id": candidate_tree,
        },
    )
    monkeypatch.setattr(
        aseh_operator,
        "_git",
        lambda *_args, **_kwargs: "",
    )
    monkeypatch.setattr(
        aseh_operator,
        "_build_server",
        lambda *_args: ordering.append("build_server") or server,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_stop_signal_handlers",
        lambda *_args: nullcontext(),
    )
    monkeypatch.setattr(
        multi_runner,
        "parse_accepted_control_plane_pin",
        lambda _value: pin,
    )
    monkeypatch.setattr(
        multi_runner,
        "verify_agent_implementation_sealed_control_plane",
        lambda _pin, descriptor: f"/proc/self/fd/{descriptor}",
    )
    monkeypatch.setattr(
        multi_runner,
        "admit_retained_control_plane_interpreter",
        lambda **_kwargs: interpreter,
    )
    monkeypatch.setattr(
        llm_router,
        "parse_agent_supervisor_native_dependency_launch",
        lambda _value: native_dependency,
    )
    monkeypatch.setattr(
        llm_router,
        "verify_agent_supervisor_native_dependency_sealed_fd",
        lambda _launch: "/proc/self/fd/92",
    )
    monkeypatch.setattr(
        multi_runner,
        "admit_trusted_system_dependency_directories",
        lambda _value: (),
    )
    native_alias = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "_duckdb", native_alias)
    monkeypatch.setitem(sys.modules, "duckdb", native_alias)
    sealed_environment = dict(os.environ)
    monkeypatch.setattr(
        aseh_operator,
        "_sealed_owner_delegation_environment",
        lambda _qualification_home: dict(sealed_environment),
    )
    monkeypatch.setattr(
        aseh_operator,
        "_validate_aseh_qualification_home",
        lambda qualification_home: qualification_home,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_sealed_receipt_validation_executor_scope",
        validation_scope,
    )
    monkeypatch.setattr(
        process_security_module,
        "make_state_authority_process_nondumpable",
        lambda: None,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_assert_candidate_authorization_witness",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            aseh_operator.OperatorError("candidate moved before owner start")
        ),
    )

    with pytest.raises(aseh_operator.OperatorError, match="moved before owner"):
        aseh_operator._run_supervisor_owner(
            config,
            implement=True,
            duration=1.0,
            sealed_control_plane_pin={"pin": "sealed"},
            sealed_control_plane_descriptor=90,
            retained_interpreter={
                "descriptor": 91,
                "argv0": interpreter.argv0,
                "sha256": interpreter.sha256,
            },
            native_dependency_launch={"native": "sealed"},
            system_dependency_directories_json="[]",
            sealed_owner_environment=sealed_environment,
        )
    assert ordering == [
        "executor_enter",
        "preflight",
        "admission",
        "executor_exit",
        "build_server",
    ]


def test_aseh_r14_sealed_receipt_validation_transition_is_active_admission_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap_path = tmp_path / "bootstrap.json"
    repair_path = tmp_path / "repair-r1.json"
    followup_path = tmp_path / "repair-r2.json"
    clean_launch_path = tmp_path / "repair-r3.json"
    runtime_hardening_path = tmp_path / "repair-r4.json"
    quack_recovery_path = tmp_path / "repair-r5.json"
    parallel_startup_path = tmp_path / "repair-r6.json"
    publication_contention_path = tmp_path / "repair-r7.json"
    recovery_replay_path = tmp_path / "repair-r8.json"
    control_receipt_lifecycle_path = tmp_path / "repair-r9.json"
    provider_lease_ownership_path = tmp_path / "repair-r10.json"
    provider_cleanup_fence_path = tmp_path / "repair-r11.json"
    sealed_owner_identity_path = tmp_path / "repair-r12.json"
    validation_executor_identity_path = tmp_path / "repair-r13.json"
    sealed_receipt_validation_path = tmp_path / "repair-r14.json"
    database_path = tmp_path / "control.duckdb"
    for path in (
        bootstrap_path,
        repair_path,
        followup_path,
        clean_launch_path,
        runtime_hardening_path,
        quack_recovery_path,
        parallel_startup_path,
        publication_contention_path,
        recovery_replay_path,
        control_receipt_lifecycle_path,
        provider_lease_ownership_path,
        provider_cleanup_fence_path,
        sealed_owner_identity_path,
        validation_executor_identity_path,
        sealed_receipt_validation_path,
        database_path,
    ):
        path.touch()
    paths = {
        "bootstrap_receipt": bootstrap_path,
        "repair_transition_receipt": repair_path,
        "repair_followup_transition_receipt": followup_path,
        "repair_clean_launch_transition_receipt": clean_launch_path,
        "repair_runtime_hardening_transition_receipt": (
            runtime_hardening_path
        ),
        "repair_quack_recovery_transition_receipt": quack_recovery_path,
        "repair_parallel_blocked_startup_transition_receipt": (
            parallel_startup_path
        ),
        "repair_quack_publication_contention_transition_receipt": (
            publication_contention_path
        ),
        "repair_quack_recovery_replay_transition_receipt": (
            recovery_replay_path
        ),
        "repair_control_receipt_lifecycle_transition_receipt": (
            control_receipt_lifecycle_path
        ),
        "repair_provider_lease_ownership_transition_receipt": (
            provider_lease_ownership_path
        ),
        "repair_provider_cleanup_fence_transition_receipt": (
            provider_cleanup_fence_path
        ),
        "repair_sealed_owner_identity_transition_receipt": (
            sealed_owner_identity_path
        ),
        "repair_validation_executor_identity_transition_receipt": (
            validation_executor_identity_path
        ),
        "repair_sealed_receipt_validation_transition_receipt": (
            sealed_receipt_validation_path
        ),
        "database": database_path,
    }
    bootstrap = {
        "source_head": "0" * 40,
        "repository_tree_id": "tree:sealed",
        "plan_root_cid": "plan:sealed",
        "source_forest": {"forest_cid": "forest:sealed"},
        "source_identities": {"identity": "sealed"},
        "bootstrap_receipt_id": "bootstrap:sealed",
    }
    r1_receipt = {"revision": 1}
    r2_receipt = {"revision": 2}
    r3_receipt = {"revision": 3}
    r4_receipt = {"revision": 4}
    r5_receipt = {"revision": 5}
    r6_receipt = {"revision": 6}
    r7_receipt = {"revision": 7}
    r8_receipt = {"revision": 8}
    r9_receipt = {"revision": 9}
    r10_receipt = {"revision": 10}
    r11_receipt = {"revision": 11}
    r12_receipt = {"revision": 12}
    r3_head = aseh_operator.REPAIR_RUNTIME_HARDENING_TRANSITION_BASE_HEAD
    r4_head = "4" * 40
    r5_head = "5" * 40
    r6_head = "6" * 40
    r7_head = "7" * 40
    r8_head = "8" * 40
    r9_head = "9" * 40
    r10_head = (
        aseh_operator.REPAIR_PROVIDER_CLEANUP_FENCE_TRANSITION_BASE_HEAD
    )
    r11_head = (
        aseh_operator.REPAIR_SEALED_OWNER_IDENTITY_TRANSITION_BASE_HEAD
    )
    r12_head = (
        aseh_operator
        .REPAIR_VALIDATION_EXECUTOR_IDENTITY_TRANSITION_BASE_HEAD
    )
    r13_head = (
        aseh_operator
        .REPAIR_SEALED_RECEIPT_VALIDATION_TRANSITION_BASE_HEAD
    )
    r14_head = "d" * 40
    population = {
        "source_head": r14_head,
        "repository_tree_id": "tree:runtime",
        "plan_root_cid": "plan:sealed",
        "source_forest": {"forest_cid": "forest:runtime"},
        "source_identities": {"identity": "runtime"},
    }
    r1 = {
        "repair_head": aseh_operator.REPAIR_FOLLOWUP_TRANSITION_FIRST_PARENT,
        "transition_revision": 1,
        "receipt_cid": "receipt:r1",
    }
    r2 = {
        "repair_head": aseh_operator.REPAIR_CLEAN_LAUNCH_TRANSITION_BASE_HEAD,
        "transition_revision": 2,
        "receipt_cid": "receipt:r2",
    }
    r3 = {
        "base_head": aseh_operator.REPAIR_CLEAN_LAUNCH_TRANSITION_BASE_HEAD,
        "repair_head": r3_head,
        "transition_revision": 3,
        "receipt_cid": "receipt:r3",
    }
    r4 = {
        "base_head": r3_head,
        "repair_head": r4_head,
        "transition_revision": 4,
        "receipt_cid": "receipt:r4",
    }
    r5 = {
        "base_head": r4_head,
        "repair_head": r5_head,
        "transition_revision": 5,
        "receipt_cid": "receipt:r5",
    }
    r6 = {
        "base_head": r5_head,
        "repair_head": r6_head,
        "transition_revision": 6,
        "receipt_cid": "receipt:r6",
    }
    r7 = {
        "base_head": r6_head,
        "repair_head": r7_head,
        "transition_revision": 7,
        "receipt_cid": "receipt:r7",
    }
    r8 = {
        "base_head": r7_head,
        "repair_head": r8_head,
        "transition_revision": 8,
        "receipt_cid": "receipt:r8",
    }
    r9 = {
        "base_head": r8_head,
        "repair_head": r9_head,
        "transition_revision": 9,
        "receipt_cid": "receipt:r9",
    }
    r10 = {
        "base_head": r9_head,
        "repair_head": r10_head,
        "transition_revision": 10,
        "receipt_cid": "receipt:r10",
    }
    r11 = {
        "base_head": r10_head,
        "repair_head": r11_head,
        "transition_revision": 11,
        "receipt_cid": "receipt:r11",
    }
    r12 = {
        "schema": (
            aseh_operator.REPAIR_SEALED_OWNER_IDENTITY_TRANSITION_SCHEMA
        ),
        "base_head": r11_head,
        "repair_head": r12_head,
        "transition_revision": 12,
        "receipt_cid": "receipt:r12",
    }
    r13 = {
        "schema": (
            aseh_operator
            .REPAIR_VALIDATION_EXECUTOR_IDENTITY_TRANSITION_SCHEMA
        ),
        "base_head": r12_head,
        "repair_head": r13_head,
        "transition_revision": 13,
        "receipt_cid": "receipt:r13",
    }
    r14 = {
        "schema": (
            aseh_operator
            .REPAIR_SEALED_RECEIPT_VALIDATION_TRANSITION_SCHEMA
        ),
        "base_head": r13_head,
        "repair_head": r14_head,
        "transition_revision": 14,
        "receipt_cid": "receipt:r14",
    }
    payloads = {
        bootstrap_path: bootstrap,
        repair_path: r1_receipt,
        followup_path: r2_receipt,
        clean_launch_path: r3_receipt,
        runtime_hardening_path: r4_receipt,
        quack_recovery_path: r5_receipt,
        parallel_startup_path: r6_receipt,
        publication_contention_path: r7_receipt,
        recovery_replay_path: r8_receipt,
        control_receipt_lifecycle_path: r9_receipt,
        provider_lease_ownership_path: r10_receipt,
        provider_cleanup_fence_path: r11_receipt,
        sealed_owner_identity_path: r12_receipt,
        validation_executor_identity_path: {"revision": 13},
        sealed_receipt_validation_path: {"revision": 14},
    }
    suffix_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        aseh_operator, "_population", lambda _board, _config: population
    )
    monkeypatch.setattr(
        aseh_operator,
        "_secure_runtime_json",
        lambda path, **_kwargs: payloads[path],
    )
    monkeypatch.setattr(
        aseh_operator,
        "_bootstrap_receipt_id",
        lambda _payload: "bootstrap:sealed",
    )
    monkeypatch.setattr(
        aseh_operator,
        "_validate_repair_transition",
        lambda *_args, **_kwargs: r1,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_validate_repair_followup_transition",
        lambda *_args, **_kwargs: r2,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_validate_repair_clean_launch_transition",
        lambda *_args, **_kwargs: r3,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_validate_repair_runtime_hardening_transition",
        lambda *_args, **_kwargs: r4,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_validate_repair_quack_recovery_transition",
        lambda *_args, **_kwargs: r5,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_validate_repair_parallel_blocked_startup_transition",
        lambda *_args, **_kwargs: r6,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_validate_repair_quack_publication_contention_transition",
        lambda *_args, **_kwargs: r7,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_validate_repair_quack_recovery_replay_transition",
        lambda *_args, **_kwargs: r8,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_validate_repair_control_receipt_lifecycle_transition",
        lambda *_args, **_kwargs: r9,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_validate_repair_provider_lease_ownership_transition",
        lambda *_args, **_kwargs: r10,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_validate_repair_provider_cleanup_fence_transition",
        lambda *_args, **_kwargs: r11,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_validate_repair_sealed_owner_identity_transition",
        lambda *_args, **_kwargs: r12,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_validate_repair_validation_executor_identity_transition",
        lambda *_args, **_kwargs: r13,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_validate_repair_sealed_receipt_validation_transition",
        lambda *_args, **_kwargs: r14,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_read_continuity_state",
        lambda *_args, **_kwargs: (
            {"projection_cid": "projection:current", "event_cursor": 79},
            [],
            {"task_statuses": {}, "task_revisions": {}},
            {},
            [],
        ),
    )
    monkeypatch.setattr(
        aseh_operator,
        "_admit_repair_followup_base",
        lambda *_args, **_kwargs: {"schema": "followup-base"},
    )

    def admit_suffix(
        _board: object,
        *,
        base_head: str,
        target_head: str,
        **_kwargs: object,
    ) -> dict[str, object]:
        suffix_calls.append((base_head, target_head))
        return {
            "schema": "canonical-suffix",
            "base_head": base_head,
            "target_head": target_head,
            "integrations": [],
        }

    monkeypatch.setattr(
        aseh_operator, "_admit_canonical_merge_suffix", admit_suffix
    )

    admission = aseh_operator._admit_materialized_launch(
        object(), {}, paths
    )

    assert admission["repair_transition"] == r14
    assert [
        item.get("transition_revision", 1)
        for item in admission["repair_transition_chain"]
    ] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    assert suffix_calls[-1] == (r14_head, r14_head)
    assert admission["canonical_continuity"][
        "followup_to_clean_launch"
    ] == r3
    assert admission["canonical_continuity"][
        "clean_launch_to_runtime_hardening"
    ] == r4
    assert admission["canonical_continuity"][
        "runtime_hardening_to_quack_recovery"
    ] == r5
    assert admission["canonical_continuity"][
        "quack_recovery_to_parallel_blocked_startup"
    ] == r6
    assert admission["canonical_continuity"][
        "parallel_blocked_startup_to_quack_publication_contention"
    ] == r7
    assert admission["canonical_continuity"][
        "quack_publication_contention_to_quack_recovery_replay"
    ] == r8
    assert admission["canonical_continuity"][
        "quack_recovery_replay_to_control_receipt_lifecycle"
    ] == r9
    assert admission["canonical_continuity"][
        "control_receipt_lifecycle_to_provider_lease_ownership"
    ] == r10
    assert admission["canonical_continuity"][
        "provider_lease_ownership_to_provider_cleanup_fence"
    ] == r11
    assert admission["canonical_continuity"][
        "provider_cleanup_fence_to_sealed_owner_identity"
    ] == r12
    assert admission["canonical_continuity"][
        "sealed_owner_identity_to_validation_executor_identity"
    ] == r13
    assert admission["canonical_continuity"][
        "validation_executor_identity_to_sealed_receipt_validation"
    ] == r14


def test_aseh_repair_authorization_replay_rejects_head_regression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap_path = tmp_path / "bootstrap.json"
    repair_path = tmp_path / "repair.json"
    bootstrap_path.touch()
    repair_path.touch()
    board = object()
    config: dict[str, object] = {}
    paths = {
        "bootstrap_receipt": bootstrap_path,
        "repair_transition_receipt": repair_path,
        "repair_transition_authorization_lock": tmp_path / "repair.lock",
    }
    bootstrap = {"bootstrap_receipt_id": "bootstrap:sealed"}
    prior = {"repair_head": "a" * 40}
    launch_called = False

    monkeypatch.setattr(aseh_operator, "_load", lambda _path: (board, config))
    monkeypatch.setattr(aseh_operator, "_paths", lambda _board: paths)
    monkeypatch.setattr(
        aseh_operator,
        "_assert_clean_tree",
        lambda _board: ("b" * 40, "c" * 40),
    )
    monkeypatch.setattr(
        aseh_operator,
        "_r14_native_seal_anchor",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        aseh_operator,
        "_population",
        lambda _board, _config: {"source_head": "b" * 40},
    )
    monkeypatch.setattr(
        aseh_operator,
        "_secure_runtime_json",
        lambda path, **_kwargs: bootstrap if path == bootstrap_path else prior,
    )
    monkeypatch.setattr(
        aseh_operator, "_bootstrap_receipt_id", lambda _payload: "bootstrap:sealed"
    )
    monkeypatch.setattr(
        aseh_operator,
        "_validate_repair_transition",
        lambda *_args, **_kwargs: {"repair_head": prior["repair_head"]},
    )

    def reject_regression(*args: str, **_kwargs: object) -> str:
        assert args[:2] == ("merge-base", "--is-ancestor")
        raise aseh_operator.OperatorError("repair is not an ancestor")

    def launch(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal launch_called
        launch_called = True
        return {}

    monkeypatch.setattr(aseh_operator, "_git", reject_regression)
    monkeypatch.setattr(aseh_operator, "_admit_materialized_launch", launch)

    with pytest.raises(aseh_operator.OperatorError, match="not an ancestor"):
        aseh_operator.authorize_repair_transition(tmp_path / "board.json")
    assert launch_called is False


def test_aseh_scheduler_uses_only_complete_live_owner_generation_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program = DatabaseProgramConfig(
        authority_mode="quack",
        task_source_kind="duckdb",
        endpoint_secret_handle="handle:aseh-live-test",
        quack_endpoint="quack:127.0.0.1:41487",
        store_id="store:aseh-live-test",
        store_generation="1",
        schema_revision="1",
        runtime_registry_path="registry",
        failover_policy="fail_closed",
    )
    board = SimpleNamespace(
        database_program=program,
        resolved_database_program=lambda: program,
        path=lambda value: tmp_path / Path(value),
    )
    registry = tmp_path / "registry"
    registry.mkdir()
    descriptor = os.open("/dev/null", os.O_RDONLY)
    birth = current_process_birth()
    status_identity = {
        "store_id": program.store_id,
        "generation": 8,
        "schema_revision": 3,
        "process_birth_id": "birth:aseh-live-test",
        "process_birth": birth.to_dict(),
    }
    monkeypatch.setattr(
        configured_scheduler,
        "_read_stable_regular_json",
        lambda _path: (
            {"lifecycle": "ready", "identity": status_identity},
            {"state": "present"},
        ),
    )
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_STORE_LIVE_GENERATION", "9"
    )
    with pytest.raises(configured_scheduler.ConfiguredBoardError, match="incomplete"):
        configured_scheduler._database_program_with_admitted_live_owner(board)

    inherited = program.to_dict()
    inherited["store_generation"] = "9"
    inherited["schema_revision"] = "3"
    environment = {
        "IPFS_ACCELERATE_AGENT_STATE_STORE_LIVE_GENERATION": "9",
        "IPFS_ACCELERATE_AGENT_STATE_LIVE_SCHEMA_REVISION": "3",
        "IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION": "9",
        "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION": "3",
        "IPFS_ACCELERATE_AGENT_DATABASE_PROGRAM_JSON": json.dumps(inherited),
        "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SOCKET": str(
            registry / "typed-state-owner-grants.sock"
        ),
        "IPFS_ACCELERATE_AGENT_STATE_OWNER_SOCKET": str(
            registry / "typed-state-owner.sock"
        ),
        "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD": str(descriptor),
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    try:
        with pytest.raises(
            configured_scheduler.ConfiguredBoardError,
            match="exact owner status",
        ):
            configured_scheduler._database_program_with_admitted_live_owner(board)

        status_identity["generation"] = 9
        admitted = configured_scheduler._database_program_with_admitted_live_owner(
            board
        )
        assert admitted.store_generation == "9"
        assert admitted.schema_revision == "3"
        assert "--state-store-generation" in admitted.cli_args()
        assert admitted.cli_args()[
            admitted.cli_args().index("--state-store-generation") + 1
        ] == "9"
    finally:
        os.close(descriptor)


def test_aseh_stop_signal_handlers_request_cleanup_and_restore() -> None:
    requested = threading.Event()
    received: dict[str, int] = {}
    prior = signal.getsignal(signal.SIGTERM)

    with aseh_operator._stop_signal_handlers(requested, received):
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        handler(signal.SIGTERM, None)
        assert requested.is_set()
        assert received == {"signum": signal.SIGTERM}

    assert signal.getsignal(signal.SIGTERM) == prior
