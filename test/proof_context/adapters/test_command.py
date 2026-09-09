"""PCCE-032 adversarial tests for the argv-only command adapter."""

from __future__ import annotations

import errno
import fcntl
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from ipfs_accelerate_py.proof_context.adapters import command as command_module
from ipfs_accelerate_py.proof_context.adapters.base import CancellationToken, execute_propose
from ipfs_accelerate_py.proof_context.adapters.command import (
    CommandAdapter,
    CommandPolicy,
    decode_structured_output,
    invoke_command,
)
from ipfs_accelerate_py.proof_context.adapters.models import (
    CONTEXT_PACK_SCHEMA,
    MODEL_ROUTE_DECISION_SCHEMA,
    TASK_SPECIFICATION_SCHEMA,
    ContextPack,
    ModelRouteDecision,
    TaskSpecification,
)
from ipfs_accelerate_py.proof_context.errors import (
    BoundaryViolationError,
    MalformedError,
    ProofCancelledError,
    ProofTimeoutError,
    UnavailableCapabilityError,
)

CID = "bafkreihxffr7ppivqrjwb3pafumcoh6mg7vyoglc6gpnmbtjgnq2pdajru"
CID_B = "bafkreihxffr7ppivqrjwb3pafumcoh6mg7vyoglc6gpnmbtjgnq2pdajrv"
CID_C = "bafkreihxffr7ppivqrjwb3pafumcoh6mg7vyoglc6gpnmbtjgnq2pdajrw"
OWNED = "src/demo.py"


def _policy(
    tmp_path: Path, code: str, *, timeout: float = 2, extra: tuple[str, ...] = ()
) -> CommandPolicy:
    binary = os.path.realpath(sys.executable)
    return CommandPolicy(
        binary,
        (binary,),
        str(tmp_path),
        (str(tmp_path),),
        ("-c", code, *extra),
        timeout_seconds=timeout,
    )


def _records() -> tuple[TaskSpecification, ContextPack, ModelRouteDecision]:
    task = TaskSpecification.from_mapping(
        {
            "schema": TASK_SPECIFICATION_SCHEMA,
            "task_id": "PCCE-032",
            "objective_id": "PCCE-G300",
            "repository_state_cid": CID,
            "owned_paths": [OWNED],
            "declared_files": [OWNED],
            "route_cid": CID_B,
            "provenance": "live",
        }
    )
    pack = ContextPack.from_mapping(
        {
            "schema": CONTEXT_PACK_SCHEMA,
            "pack_cid": CID_C,
            "repository_state_cid": CID,
            "sufficiency": "sufficient",
            "task_id": "PCCE-032",
            "provenance": "live",
        }
    )
    route = ModelRouteDecision.from_mapping(
        {
            "schema": MODEL_ROUTE_DECISION_SCHEMA,
            "decision_cid": CID_B,
            "task_id": "PCCE-032",
            "repository_state_cid": CID,
            "provider": "local",
            "model": "agent",
            "revision": "r1",
            "tier": "medium",
            "provenance": "live",
        }
    )
    return task, pack, route


def _provider_code(patch: object) -> str:
    return (
        "import json,sys; r=json.load(sys.stdin); "
        f"patch={patch!r}; "
        "print(json.dumps({'task_id':r['task_id'],"
        "'repository_state_cid':r['repository_state_cid'],"
        "'pack_cid':r['pack_cid'],'route_cid':r['route_cid'],"
        "'declared_files':r['declared_files'],'patch':patch,"
        "'model':r['model'],'revision':r['revision'],'token_count':1,"
        "'cached_token_count':0,'latency_ms':1,'cost_micros':0}))"
    )


def _lock_is_available(path: Path) -> bool:
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(stream, fcntl.LOCK_UN)
        return True


def _kill_lock_holder(path: Path) -> None:
    metadata = path.stat()
    identity = (os.major(metadata.st_dev), os.minor(metadata.st_dev), metadata.st_ino)

    def anchored_owner_pids() -> set[int]:
        owners: set[int] = set()
        for line in Path("/proc/locks").read_text().splitlines():
            fields = line.split()
            if len(fields) < 6 or not fields[4].isdecimal():
                continue
            device_inode = fields[5].split(":")
            if len(device_inode) != 3:
                continue
            try:
                observed = (
                    int(device_inode[0], 16),
                    int(device_inode[1], 16),
                    int(device_inode[2]),
                )
            except ValueError:
                continue
            if observed == identity:
                owners.add(int(fields[4]))
        return owners

    # Anchor the exact owner before revalidating the full device/inode identity;
    # pidfd signalling cannot follow a recycled numeric PID to an unrelated task.
    for pid in anchored_owner_pids():
        try:
            pidfd = os.pidfd_open(pid)
        except ProcessLookupError:
            continue
        try:
            if pid not in anchored_owner_pids():
                continue
            try:
                signal.pidfd_send_signal(pidfd, signal.SIGKILL)
            except ProcessLookupError:
                pass
        finally:
            os.close(pidfd)


def _wait_for_lock(path: Path, *, timeout: float = 2) -> None:
    deadline = time.monotonic() + timeout
    while _lock_is_available(path) and time.monotonic() < deadline:
        time.sleep(0.005)
    assert not _lock_is_available(path)


def _double_fork_detached_code(lock_file: Path, action: str) -> str:
    detached = (
        "import fcntl,os,signal,time\n"
        "os.setsid()\n"
        "child = os.fork()\n"
        "if child:\n"
        "    os._exit(0)\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"lock = open({str(lock_file)!r}, 'r')\n"
        "fcntl.flock(lock, fcntl.LOCK_EX)\n"
        "print('ready', flush=True)\n"
        "time.sleep(10)\n"
    )
    return (
        "import subprocess,sys,time\n"
        f"detached = subprocess.Popen([sys.executable, '-c', {detached!r}], "
        "stdout=subprocess.PIPE, text=True)\n"
        "assert detached.stdout is not None\n"
        "assert detached.stdout.readline() == 'ready\\n'\n"
        f"{action}\n"
    )


def _guardian_attack_code(
    lock_file: Path,
    requested_signal: int,
) -> str:
    detached = (
        "import fcntl,json,os,signal,time\n"
        "os.setsid()\n"
        "session_id = os.getsid(0)\n"
        "child = os.fork()\n"
        "if child:\n"
        "    os._exit(0)\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"lock = open({str(lock_file)!r}, 'r')\n"
        "fcntl.flock(lock, fcntl.LOCK_EX)\n"
        "print(json.dumps({'pid': os.getpid(), 'session': session_id}), flush=True)\n"
        "time.sleep(10)\n"
    )
    return (
        "import json,os,subprocess,sys\n"
        "wrapper = os.getppid()\n"
        f"detached = subprocess.Popen([sys.executable, '-c', {detached!r}], "
        "stdout=subprocess.PIPE, text=True)\n"
        "assert detached.stdout is not None\n"
        "identity = json.loads(detached.stdout.readline())\n"
        "try:\n"
        f"    os.kill(wrapper, {requested_signal})\n"
        "except PermissionError:\n"
        "    outcome = 'blocked'\n"
        "else:\n"
        "    outcome = 'allowed'\n"
        "print(json.dumps({'wrapper': wrapper, 'identity': identity, 'outcome': outcome}))\n"
    )


def _adapter_host_code(cwd: Path, lock_file: Path, timeout: float) -> str:
    provider = _double_fork_detached_code(lock_file, "time.sleep(10)")
    return (
        "import os,sys\n"
        "from ipfs_accelerate_py.proof_context.adapters.command import "
        "CommandPolicy,invoke_command\n"
        "from ipfs_accelerate_py.proof_context.errors import ProofTimeoutError\n"
        "binary=os.path.realpath(sys.executable)\n"
        f"policy=CommandPolicy(binary,(binary,),{str(cwd)!r},({str(cwd)!r},),"
        f"('-c',{provider!r}),timeout_seconds={timeout!r})\n"
        "try:\n"
        "    invoke_command(policy,{'x': 1})\n"
        "except ProofTimeoutError:\n"
        "    raise SystemExit(42)\n"
        "raise SystemExit(0)\n"
    )


def test_policy_rejects_relative_or_non_allowlisted_executable_and_cwd(tmp_path: Path) -> None:
    binary = os.path.realpath(sys.executable)
    with pytest.raises(BoundaryViolationError):
        CommandPolicy("python", (binary,), str(tmp_path), (str(tmp_path),))
    with pytest.raises(BoundaryViolationError):
        CommandPolicy(binary, (binary,), str(tmp_path), ("/tmp",))


def test_argv_is_literal_and_environment_is_hermetic(tmp_path: Path) -> None:
    marker = tmp_path / "injected"
    code = "import json,os,sys; print(json.dumps({'argv':sys.argv[1:], 'env':sorted(os.environ), 'cwd':os.getcwd()}))"
    policy = _policy(tmp_path, code, extra=(f"; touch {marker}",))
    result = invoke_command(policy, {"hello": "world"})
    observed = json.loads(result.stdout)
    assert observed["argv"] == [f"; touch {marker}"]
    assert not marker.exists()
    assert observed["cwd"] == str(tmp_path)
    assert "HOME" in observed["env"] and "PATH" in observed["env"]
    assert "AWS_SECRET_ACCESS_KEY" not in observed["env"]


def test_provider_runs_in_private_proc_and_run_without_namespace_capabilities(
    tmp_path: Path,
) -> None:
    code = """
import ctypes,json,os
libc = ctypes.CDLL(None, use_errno=True)
def denied(call):
    ctypes.set_errno(0)
    result = call()
    return [result, ctypes.get_errno()]
status = open('/proc/self/status').read().splitlines()
caps = {line.split(':', 1)[0]: int(line.split()[1], 16)
        for line in status if line.startswith(('CapInh:', 'CapPrm:', 'CapEff:', 'CapBnd:', 'CapAmb:'))}
observed = {
    'pid': os.getpid(),
    'ppid': os.getppid(),
    'proc_pids': sorted(int(name) for name in os.listdir('/proc') if name.isdecimal()),
    'run': os.listdir('/run'),
    'caps': caps,
    'mount': denied(lambda: libc.mount(b'tmpfs', b'/run', b'tmpfs', 0, b'')),
    'unshare': denied(lambda: libc.unshare(0x10000000)),
    'dumpable': denied(lambda: libc.prctl(4, 1, 0, 0, 0)),
}
print(json.dumps(observed))
"""
    observed = json.loads(invoke_command(_policy(tmp_path, code), {"x": 1}).stdout)
    assert observed["pid"] == 2 and observed["ppid"] == 1
    assert observed["proc_pids"] == [1, 2]
    assert observed["run"] == []
    assert set(observed["caps"].values()) == {0}
    assert observed["mount"] == [-1, 1]
    assert observed["unshare"] == [-1, 1]
    assert observed["dumpable"] == [-1, 1]


def test_provider_has_private_disabled_network_and_cannot_reach_host_loopback(
    tmp_path: Path,
) -> None:
    host_network_namespace = os.stat("/proc/self/ns/net").st_ino
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        code = f"""
import errno,json,os,socket
try:
    socket.socket(socket.AF_INET, socket.SOCK_STREAM)
except OSError as exc:
    socket_errno = exc.errno
else:
    socket_errno = 0
interfaces = sorted(
    line.split(':', 1)[0].strip()
    for line in open('/proc/net/dev').read().splitlines()
    if ':' in line
)
routes = open('/proc/net/route').read().splitlines()[1:]
print(json.dumps({{
    'normal': True,
    'network_namespace': os.stat('/proc/self/ns/net').st_ino,
    'interfaces': interfaces,
    'routes': routes,
    'socket_errno': socket_errno,
    'target_port': {port},
}}))
"""
        execution = invoke_command(_policy(tmp_path, code), {"x": 1})
        observed = json.loads(execution.stdout)
        assert execution.returncode == 0 and observed["normal"] is True
        assert observed["network_namespace"] != host_network_namespace
        assert observed["interfaces"] == ["lo"]
        assert observed["routes"] == []
        assert observed["socket_errno"] == errno.EPERM
        assert observed["target_port"] == port
        listener.setblocking(False)
        with pytest.raises(BlockingIOError):
            listener.accept()


def test_provider_has_no_inherited_supplementary_group_authority(tmp_path: Path) -> None:
    protected: Path | None = None
    for candidate in (Path("/var/log/auth.log.1"), Path("/var/log/syslog")):
        try:
            metadata = candidate.stat()
            with candidate.open("rb") as stream:
                stream.read(0)
        except (FileNotFoundError, PermissionError, OSError):
            continue
        if (
            candidate.is_file()
            and metadata.st_uid != os.geteuid()
            and metadata.st_gid in os.getgroups()
            and metadata.st_mode & 0o040
            and not metadata.st_mode & 0o004
        ):
            protected = candidate
            break

    code = f"""
import errno,json,os
protected = {str(protected) if protected is not None else None!r}
if protected is None:
    opened = None
    open_errno = None
else:
    try:
        descriptor = os.open(protected, os.O_RDONLY)
    except OSError as exc:
        opened = False
        open_errno = exc.errno
    else:
        os.close(descriptor)
        opened = True
        open_errno = 0
print(json.dumps({{'groups': os.getgroups(), 'opened': opened, 'errno': open_errno}}))
"""
    observed = json.loads(invoke_command(_policy(tmp_path, code), {"x": 1}).stdout)
    assert observed["groups"] == []
    if protected is None:
        assert observed["opened"] is None and observed["errno"] is None
    else:
        assert observed["opened"] is False and observed["errno"] == errno.EACCES


def test_provider_cannot_create_pathname_unix_socket_or_reach_host_broker(
    tmp_path: Path,
) -> None:
    broker_path = tmp_path / "host-broker.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as broker:
        broker.bind(str(broker_path))
        broker.listen()
        code = f"""
import json,socket
try:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
except OSError as exc:
    socket_errno = exc.errno
    connect_errno = None
else:
    try:
        client.connect({str(broker_path)!r})
    except OSError as exc:
        connect_errno = exc.errno
    else:
        connect_errno = 0
    finally:
        client.close()
    socket_errno = 0
print(json.dumps({{'normal': True, 'socket_errno': socket_errno,
                  'connect_errno': connect_errno}}))
"""
        observed = json.loads(invoke_command(_policy(tmp_path, code), {"x": 1}).stdout)
        assert observed == {
            "normal": True,
            "socket_errno": errno.EPERM,
            "connect_errno": None,
        }
        broker.setblocking(False)
        with pytest.raises(BlockingIOError):
            broker.accept()


def test_provider_can_write_only_private_runtime_paths_and_not_cwd(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical.py"
    canonical.write_text("original")
    original_stat = canonical.stat()
    outside = tmp_path.parent / f"{tmp_path.name}-outside-host.py"
    outside.write_text("outside-original")
    marker = tmp_path / "forbidden-marker"
    renamed = tmp_path / "renamed.py"
    linked = tmp_path / "linked.py"
    symlinked = tmp_path / "symlinked.py"
    code = f"""
import json,os,pathlib
def denied(operation):
    try:
        operation()
    except OSError as exc:
        return exc.errno
    return 0
home = pathlib.Path(os.environ['HOME'])
home_file = home / 'provider-output'
run_file = pathlib.Path('/run/provider-output')
home_file.write_text('home')
run_file.write_text('run')
observed = {{
    'marker': denied(lambda: pathlib.Path({str(marker)!r}).write_text('forbidden')),
    'truncate': denied(lambda: pathlib.Path({str(canonical)!r}).write_text('changed')),
    'rename': denied(lambda: os.rename({str(canonical)!r}, {str(renamed)!r})),
    'unlink': denied(lambda: os.unlink({str(canonical)!r})),
    'link': denied(lambda: os.link({str(canonical)!r}, {str(linked)!r})),
    'symlink': denied(lambda: os.symlink({str(canonical)!r}, {str(symlinked)!r})),
    'chmod': denied(lambda: os.chmod({str(canonical)!r}, 0o777)),
    'utime': denied(lambda: os.utime({str(canonical)!r}, None)),
    'xattr': denied(lambda: os.setxattr({str(canonical)!r}, b'user.pcce', b'x')),
    'outside_truncate': denied(
        lambda: pathlib.Path({str(outside)!r}).write_text('outside-changed')
    ),
    'outside_unlink': denied(lambda: os.unlink({str(outside)!r})),
    'home': home_file.read_text(),
    'run': run_file.read_text(),
}}
print(json.dumps(observed))
"""
    observed = json.loads(invoke_command(_policy(tmp_path, code), {"x": 1}).stdout)
    assert all(
        observed[field] in {errno.EACCES, errno.EPERM}
        for field in (
            "marker",
            "truncate",
            "rename",
            "unlink",
            "link",
            "symlink",
            "chmod",
            "utime",
            "xattr",
            "outside_truncate",
            "outside_unlink",
        )
    )
    assert observed["home"] == "home" and observed["run"] == "run"
    assert not marker.exists() and not renamed.exists()
    assert not linked.exists() and not symlinked.exists()
    assert canonical.read_text() == "original"
    assert canonical.stat().st_mode == original_stat.st_mode
    assert canonical.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert "user.pcce" not in os.listxattr(canonical)
    assert outside.read_text() == "outside-original"


def test_provider_device_writes_are_limited_to_effect_free_devices(tmp_path: Path) -> None:
    code = """
import json,os
def write_device(path):
    try:
        descriptor = os.open(path, os.O_WRONLY)
        try:
            os.write(descriptor, b'x')
        finally:
            os.close(descriptor)
    except OSError as exc:
        return exc.errno
    return 0
print(json.dumps({path: write_device(path) for path in (
    '/dev/null', '/dev/full', '/dev/zero', '/dev/random', '/dev/urandom'
)}))
"""
    observed = json.loads(invoke_command(_policy(tmp_path, code), {"x": 1}).stdout)
    assert observed["/dev/null"] == 0
    assert observed["/dev/full"] == errno.ENOSPC
    assert observed["/dev/zero"] == 0
    assert observed["/dev/random"] == errno.EACCES
    assert observed["/dev/urandom"] == errno.EACCES


def test_policy_rejects_cwd_symlink_drift_before_spawn(tmp_path: Path) -> None:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    policy = _policy(cwd, "print('{}')")
    moved = tmp_path / "moved-cwd"
    cwd.rename(moved)
    cwd.symlink_to(moved, target_is_directory=True)
    with pytest.raises(BoundaryViolationError):
        invoke_command(policy, {"x": 1})


def test_policy_rejects_cwd_hidden_below_private_runtime_mount() -> None:
    cwd = os.path.realpath("/proc/self")
    binary = os.path.realpath(sys.executable)
    policy = CommandPolicy(binary, (binary,), cwd, (cwd,), ("-c", "print('{}')"))
    with pytest.raises(BoundaryViolationError, match="private runtime mount"):
        invoke_command(policy, {"x": 1})


def test_policy_rejects_executable_rename_replacement_before_spawn(tmp_path: Path) -> None:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    executable = tmp_path / "provider-python"
    shutil.copy2(os.path.realpath(sys.executable), executable)
    policy = CommandPolicy(
        str(executable),
        (str(executable),),
        str(cwd),
        (str(cwd),),
        ("-c", "print('{}')"),
    )
    moved = tmp_path / "original-provider-python"
    executable.rename(moved)
    shutil.copy2(moved, executable)
    with pytest.raises(BoundaryViolationError):
        invoke_command(policy, {"x": 1})


def test_fd_bound_user_owned_script_executes_with_only_its_anchor_inherited(
    tmp_path: Path,
) -> None:
    script = tmp_path / "provider-script"
    script.write_text(
        "#!/usr/bin/python3\n"
        "import json,os\n"
        "targets=[]\n"
        "for name in os.listdir('/proc/self/fd'):\n"
        "    if int(name) <= 2:\n"
        "        continue\n"
        "    try:\n"
        "        targets.append(os.readlink('/proc/self/fd/' + name))\n"
        "    except OSError:\n"
        "        pass\n"
        "print(json.dumps({'normal': True, 'fd_targets': sorted(targets)}))\n"
    )
    script.chmod(0o700)
    policy = CommandPolicy(
        str(script),
        (str(script),),
        str(tmp_path),
        (str(tmp_path),),
    )
    execution = invoke_command(policy, {"x": 1})
    observed = json.loads(execution.stdout)
    assert execution.returncode == 0 and observed["normal"] is True
    assert observed["fd_targets"] == [str(script)]


def test_namespace_capability_failure_never_executes_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "provider-started"
    monkeypatch.setattr(command_module, "_NAMESPACE_GATE_SOURCE", "import os; os._exit(125)")
    with pytest.raises(UnavailableCapabilityError):
        invoke_command(
            _policy(tmp_path, f"from pathlib import Path; Path({str(marker)!r}).touch()"),
            {"x": 1},
        )
    assert not marker.exists()


def test_insufficient_landlock_fails_before_provider_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        command_module,
        "_NAMESPACE_GATE_SOURCE",
        command_module._NAMESPACE_GATE_SOURCE.replace(
            "LANDLOCK_MINIMUM_ABI = 6",
            "LANDLOCK_MINIMUM_ABI = 99",
        ),
    )
    with pytest.raises(UnavailableCapabilityError):
        invoke_command(_policy(tmp_path, "print('provider-ran')"), {"x": 1})


def test_missing_identity_mapping_helper_never_starts_launcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = False

    def fail_mapping_helpers() -> None:
        raise UnavailableCapabilityError("injected missing mapping helper")

    def record_start(*_args: object, **_kwargs: object) -> object:
        nonlocal started
        started = True
        raise AssertionError("launcher must not start")

    monkeypatch.setattr(command_module, "_require_trusted_mapping_helpers", fail_mapping_helpers)
    monkeypatch.setattr(command_module.subprocess, "Popen", record_start)
    with pytest.raises(UnavailableCapabilityError, match="missing mapping helper"):
        invoke_command(_policy(tmp_path, "print('{}')"), {"x": 1})
    assert started is False


def test_namespace_control_failure_never_executes_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "provider-started"

    def fail_control_channel(*_args: object) -> object:
        raise OSError("injected namespace control failure")

    monkeypatch.setattr(command_module.socket, "socketpair", fail_control_channel)
    with pytest.raises(UnavailableCapabilityError):
        invoke_command(
            _policy(tmp_path, f"from pathlib import Path; Path({str(marker)!r}).touch()"),
            {"x": 1},
        )
    assert not marker.exists()


def test_pre_handshake_failure_kills_the_complete_trusted_launcher_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_file = tmp_path / "unreleased-gate.lock"
    stalled_gate = (
        "import fcntl,time;"
        f"lock=open({str(lock_file)!r},'w');"
        "fcntl.flock(lock,fcntl.LOCK_EX);"
        "lock.write('ready');lock.flush();"
        "time.sleep(10)"
    )
    monkeypatch.setattr(command_module, "_NAMESPACE_GATE_SOURCE", stalled_gate)
    with pytest.raises(ProofTimeoutError):
        invoke_command(_policy(tmp_path, "raise AssertionError", timeout=0.2), {"x": 1})
    assert lock_file.read_text() == "ready"
    assert _lock_is_available(lock_file)


def test_output_is_bounded_redacted_and_nonzero_does_not_decode(tmp_path: Path) -> None:
    execution = invoke_command(
        _policy(tmp_path, "import sys; sys.stderr.write('token=super-secret\\n'); print('{}')"),
        {"x": 1},
    )
    assert b"super-secret" not in execution.log_bytes and b"[redacted]" in execution.log_bytes
    oversized = _policy(tmp_path, "import sys; sys.stdout.write('x' * 2600000)")
    with pytest.raises(BoundaryViolationError):
        invoke_command(oversized, {"x": 1})


@pytest.mark.parametrize(
    "output", [b"", b"[]", b"{} trailing", b"```json\\n{}\\n```", b'{"patch":"x"}']
)
def test_structured_decoder_is_closed_and_fail_closed(output: bytes) -> None:
    with pytest.raises(MalformedError):
        decode_structured_output(output)


def test_structured_decoder_rejects_duplicate_json_object_keys() -> None:
    proposal = {
        "task_id": "PCCE-032",
        "repository_state_cid": CID,
        "pack_cid": CID_B,
        "route_cid": CID_C,
        "declared_files": [OWNED],
        "patch": "first",
        "model": "agent",
        "revision": "r1",
        "token_count": 1,
        "cached_token_count": 0,
        "latency_ms": 1,
        "cost_micros": 0,
    }
    encoded = json.dumps(proposal, separators=(",", ":"))
    duplicated = encoded[:-1] + ',"patch":"second"}'
    with pytest.raises(MalformedError):
        decode_structured_output(duplicated)


def test_timeout_and_cancellation_kill_the_process_group(tmp_path: Path) -> None:
    sleeper = _policy(tmp_path, "import time; time.sleep(5)", timeout=0.1)
    with pytest.raises(ProofTimeoutError):
        invoke_command(sleeper, {"x": 1})
    token = CancellationToken()
    threading.Timer(0.05, token.cancel).start()
    with pytest.raises(ProofCancelledError):
        invoke_command(_policy(tmp_path, "import time; time.sleep(5)"), {"x": 1}, token)


@pytest.mark.parametrize(
    ("trigger", "error"),
    [
        ("timeout", ProofTimeoutError),
        ("cancel", ProofCancelledError),
        ("output", BoundaryViolationError),
    ],
)
def test_aborted_invocation_terminates_term_ignoring_descendant(
    tmp_path: Path,
    trigger: str,
    error: type[Exception],
) -> None:
    lock_file = tmp_path / "child.lock"
    lock_file.write_text("")
    child_code = (
        "import fcntl,signal,time;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        f"lock=open({str(lock_file)!r},'r');"
        "fcntl.flock(lock,fcntl.LOCK_EX);"
        "print('ready',flush=True);"
        "time.sleep(10)"
    )
    code = (
        "import subprocess,sys,time;"
        f"child=subprocess.Popen([sys.executable, '-c', {child_code!r}],"
        "stdout=subprocess.PIPE,text=True);"
        "assert child.stdout is not None;"
        "assert child.stdout.readline() == 'ready\\n';"
        + ("sys.stdout.write('x' * 2600000); sys.stdout.flush();" if trigger == "output" else "")
        + "time.sleep(10)"
    )
    token = CancellationToken() if trigger == "cancel" else None
    if token is not None:
        threading.Timer(0.3, token.cancel).start()
    started = time.monotonic()
    with pytest.raises(error):
        invoke_command(
            _policy(tmp_path, code, timeout=0.3 if trigger == "timeout" else 2),
            {"x": 1},
            token,
        )
    assert time.monotonic() - started < 2
    for _ in range(30):
        if _lock_is_available(lock_file):
            break
        time.sleep(0.02)
    assert _lock_is_available(lock_file)


@pytest.mark.parametrize(
    ("cancel", "error"),
    [(False, ProofTimeoutError), (True, ProofCancelledError)],
)
def test_blocked_stdin_remains_timeout_and_cancellation_responsive(
    tmp_path: Path,
    cancel: bool,
    error: type[Exception],
) -> None:
    token = CancellationToken() if cancel else None
    if token is not None:
        threading.Timer(0.05, token.cancel).start()
    started = time.monotonic()
    with pytest.raises(error):
        invoke_command(
            _policy(tmp_path, "import time; time.sleep(5)", timeout=0.1 if not cancel else 2),
            {"payload": "x" * 200_000},
            token,
        )
    assert time.monotonic() - started < 1.5


@pytest.mark.parametrize(
    ("trigger", "action", "error"),
    [
        ("timeout", "time.sleep(10)", ProofTimeoutError),
        ("cancel", "time.sleep(10)", ProofCancelledError),
        (
            "output",
            "sys.stdout.write('x' * 2600000); sys.stdout.flush(); time.sleep(10)",
            BoundaryViolationError,
        ),
        ("fast-exit", "pass", None),
    ],
)
def test_pid_namespace_cleans_double_forked_setsid_descendant(
    tmp_path: Path,
    trigger: str,
    action: str,
    error: type[Exception] | None,
) -> None:
    lock_file = tmp_path / "detached.lock"
    lock_file.write_text("")
    token = CancellationToken() if trigger == "cancel" else None
    if token is not None:
        threading.Timer(0.4, token.cancel).start()
    policy = _policy(
        tmp_path,
        _double_fork_detached_code(lock_file, action),
        timeout=0.4 if trigger == "timeout" else 2,
    )
    started = time.monotonic()
    try:
        if error is None:
            assert invoke_command(policy, {"x": 1}).returncode == 0
        else:
            with pytest.raises(error):
                invoke_command(policy, {"x": 1}, token)
        assert time.monotonic() - started < 2.5
        assert _lock_is_available(lock_file)
    finally:
        if not _lock_is_available(lock_file):
            _kill_lock_holder(lock_file)


@pytest.mark.parametrize("requested_signal", [signal.SIGKILL, signal.SIGSTOP])
def test_provider_cannot_kill_or_stop_its_pid_namespace_guardian(
    tmp_path: Path,
    requested_signal: signal.Signals,
) -> None:
    lock_file = tmp_path / "attacking-detached.lock"
    lock_file.write_text("")
    started = time.monotonic()
    try:
        execution = invoke_command(
            _policy(
                tmp_path,
                _guardian_attack_code(
                    lock_file,
                    int(requested_signal),
                ),
                # Namespace setup plus the intentional fork/setsid attack can
                # exceed 400 ms on the qualified arm64 kernel under load. Keep
                # the behavior bounded without conflating launch latency with
                # a successful signal escape.
                timeout=2,
            ),
            {"x": 1},
        )
        assert execution.returncode == 0
        assert time.monotonic() - started < 2.5
        observed = json.loads(execution.stdout)
        assert observed["outcome"] == "blocked"
        assert observed["identity"]["session"] != observed["wrapper"]
        assert _lock_is_available(lock_file)
    finally:
        if not _lock_is_available(lock_file):
            _kill_lock_holder(lock_file)


@pytest.mark.parametrize(
    ("parent_signal", "policy_timeout", "expected_returncode"),
    [
        (signal.SIGKILL, 10.0, -signal.SIGKILL),
        (signal.SIGSTOP, 0.8, 42),
    ],
)
def test_adapter_parent_death_or_hang_cannot_strand_namespace_descendants(
    tmp_path: Path,
    parent_signal: signal.Signals,
    policy_timeout: float,
    expected_returncode: int,
) -> None:
    lock_file = tmp_path / f"adapter-parent-{int(parent_signal)}.lock"
    lock_file.write_text("")
    repository_root = Path(command_module.__file__).resolve().parents[3]
    invoker = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _adapter_host_code(tmp_path, lock_file, policy_timeout),
        ],
        cwd=repository_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        _wait_for_lock(lock_file, timeout=3)
        pidfd = os.pidfd_open(invoker.pid)
        try:
            signal.pidfd_send_signal(pidfd, parent_signal)
            if parent_signal == signal.SIGSTOP:
                deadline = time.monotonic() + 2.5
                while not _lock_is_available(lock_file) and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert _lock_is_available(lock_file)
                signal.pidfd_send_signal(pidfd, signal.SIGCONT)
            assert invoker.wait(timeout=2) == expected_returncode
        finally:
            os.close(pidfd)
        deadline = time.monotonic() + 2.5
        while not _lock_is_available(lock_file) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert _lock_is_available(lock_file)
    finally:
        if invoker.poll() is None:
            invoker.send_signal(signal.SIGCONT)
            invoker.kill()
            invoker.wait(timeout=2)
        if not _lock_is_available(lock_file):
            _kill_lock_holder(lock_file)


@pytest.mark.parametrize("failure_stage", ["anchor", "nonblocking", "selector"])
def test_every_post_spawn_setup_failure_cleans_the_started_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    lock_file = tmp_path / f"setup-{failure_stage}.lock"
    lock_file.write_text("")

    def injected_failure() -> None:
        _wait_for_lock(lock_file)
        raise RuntimeError(f"injected {failure_stage} failure")

    if failure_stage == "anchor":

        def fail_anchor(_process: subprocess.Popen[bytes]) -> object:
            injected_failure()

        monkeypatch.setattr(command_module, "_capture_started_launcher", fail_anchor)
    elif failure_stage == "nonblocking":

        def fail_nonblocking(_descriptor: int, _blocking: bool) -> None:
            injected_failure()

        monkeypatch.setattr(command_module.os, "set_blocking", fail_nonblocking)
    else:
        selector_factory = command_module.selectors.DefaultSelector

        class PartiallyRegisteredSelector:
            def __init__(self) -> None:
                self.inner = selector_factory()
                self.registrations = 0

            def register(self, fileobj: object, events: int) -> object:
                self.registrations += 1
                if self.registrations == 2:
                    injected_failure()
                return self.inner.register(fileobj, events)

            def unregister(self, fileobj: object) -> object:
                return self.inner.unregister(fileobj)

            def close(self) -> None:
                self.inner.close()

        monkeypatch.setattr(
            command_module.selectors,
            "DefaultSelector",
            PartiallyRegisteredSelector,
        )

    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match=f"injected {failure_stage} failure"):
            invoke_command(
                _policy(
                    tmp_path,
                    _double_fork_detached_code(lock_file, "time.sleep(10)"),
                ),
                {"x": 1},
            )
        assert time.monotonic() - started < 2
        assert _lock_is_available(lock_file)
    finally:
        if not _lock_is_available(lock_file):
            _kill_lock_holder(lock_file)


def test_namespace_cleanup_does_not_kill_unrelated_process(tmp_path: Path) -> None:
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        start_new_session=True,
    )
    try:
        with pytest.raises(ProofTimeoutError):
            invoke_command(
                _policy(tmp_path, "import time; time.sleep(10)", timeout=0.1),
                {"x": 1},
            )
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        try:
            unrelated.wait(timeout=1)
        except subprocess.TimeoutExpired:
            unrelated.kill()
            unrelated.wait(timeout=1)


def test_namespace_cleanup_is_isolated_between_concurrent_invocations(tmp_path: Path) -> None:
    peer_lock = tmp_path / "peer.lock"
    peer_lock.write_text("")
    peer_results: list[object] = []
    peer_errors: list[BaseException] = []

    def run_peer() -> None:
        code = (
            "import fcntl,time;"
            f"lock=open({str(peer_lock)!r},'r');"
            "fcntl.flock(lock,fcntl.LOCK_EX);"
            "time.sleep(.5); print('{}')"
        )
        try:
            peer_results.append(invoke_command(_policy(tmp_path, code), {"peer": True}))
        except BaseException as exc:
            peer_errors.append(exc)

    peer = threading.Thread(target=run_peer)
    peer.start()
    deadline = time.monotonic() + 1
    while _lock_is_available(peer_lock) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not _lock_is_available(peer_lock)
    with pytest.raises(ProofTimeoutError):
        invoke_command(
            _policy(tmp_path, "import time; time.sleep(10)", timeout=0.1),
            {"abort": True},
        )
    peer.join(timeout=2)
    assert not peer.is_alive()
    assert not peer_errors
    assert len(peer_results) == 1
    assert _lock_is_available(peer_lock)


def test_adapter_accepts_only_strict_identity_bound_proposal(tmp_path: Path) -> None:
    task, pack, route = _records()
    result = execute_propose(
        CommandAdapter(
            _policy(tmp_path, _provider_code("diff --git a/src/demo.py b/src/demo.py\n"))
        ),
        task,
        pack,
        route,
    )
    assert result.proposal.provenance == result.invocation.provenance == "live"
    assert result.accepted is result.approved is False


def test_adapter_rejects_patch_outside_declared_scope(tmp_path: Path) -> None:
    task, pack, route = _records()
    with pytest.raises(BoundaryViolationError):
        execute_propose(
            CommandAdapter(
                _policy(tmp_path, _provider_code("diff --git a/src/secret.py b/src/secret.py\n"))
            ),
            task,
            pack,
            route,
        )


@pytest.mark.parametrize("preamble", ["", "@@ -0,0 +0,0 @@\n"])
def test_adapter_rejects_foreign_unified_diff_file_markers(
    tmp_path: Path,
    preamble: str,
) -> None:
    patch = (
        "diff --git a/src/demo.py b/src/demo.py\n"
        f"{preamble}"
        "--- a/src/secret.py\n"
        "+++ b/src/secret.py\n"
        "@@ -1 +1 @@\n-old\n+new\n"
    )
    task, pack, route = _records()
    with pytest.raises(BoundaryViolationError):
        execute_propose(
            CommandAdapter(_policy(tmp_path, _provider_code(patch))),
            task,
            pack,
            route,
        )


def test_adapter_rejects_non_text_patch_value(tmp_path: Path) -> None:
    task, pack, route = _records()
    with pytest.raises(MalformedError):
        execute_propose(
            CommandAdapter(_policy(tmp_path, _provider_code(["not", "a", "patch"]))),
            task,
            pack,
            route,
        )
