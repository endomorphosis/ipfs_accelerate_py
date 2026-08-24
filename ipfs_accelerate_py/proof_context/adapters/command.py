"""Hermetic argv-only local coding-agent adapter (PCCE-032).

This is deliberately a small subprocess boundary: a policy names exact binary
paths and working directories, no parent environment is inherited, and every
invocation runs below a handshake-gated Linux PID-namespace init.  The kernel
owns descendant teardown when that init exits; private procfs/runtime mounts,
an unconfigured private network namespace, cleared supplementary groups,
provider socket denial, a write-denying Landlock domain, capability removal,
and inherited seccomp fences bound its process, network, and mutation effects.
Host read/path allowlisting and hostile same-UID mutation by a separate host
actor remain general sandbox concerns rather than claims of this adapter.  The
adapter accepts exactly one JSON object on stdout; diagnostics are never
interpreted as a proposal.
"""

from __future__ import annotations

import array
import base64
import hashlib
import json
import os
import re
import select
import selectors
import signal
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.proof_context.adapters.base import (
    APPROVAL_AUTHORITY,
    CANONICAL_BRANCH_AUTHORITY,
    AdapterResult,
    CancellationToken,
    admit_adapter_result,
    bind_adapter_request,
)
from ipfs_accelerate_py.proof_context.adapters.models import (
    CODING_AGENT_INVOCATION_SCHEMA,
    MAX_LOG_BYTES,
    MAX_PROVIDER_OUTPUT_BYTES,
    PATCH_PROPOSAL_SCHEMA,
    CodingAgentInvocation,
    ContextPack,
    ModelRouteDecision,
    PatchProposal,
    TaskSpecification,
    admit_bounded_patch,
    admit_non_negative_int,
    admit_path_list,
    admit_relative_path,
    assert_declared_scope,
    wire_canonical_utf8,
)
from ipfs_accelerate_py.proof_context.errors import (
    BoundaryViolationError,
    IdentityInconsistentError,
    MalformedError,
    ProofCancelledError,
    ProofTimeoutError,
    UnavailableCapabilityError,
    redact_text,
)

ADAPTER: Final[str] = "CommandAdapter@0.1"
COMMAND_CONTRACT: Final[str] = "local-agent-json-argv@1"
REQUEST_SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1/command-request@1"
MAX_ARGUMENTS: Final[int] = 64
MAX_ARGUMENT_BYTES: Final[int] = 16_384
_OUTPUT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "task_id",
        "repository_state_cid",
        "pack_cid",
        "route_cid",
        "declared_files",
        "patch",
        "model",
        "revision",
        "token_count",
        "cached_token_count",
        "latency_ms",
        "cost_micros",
    }
)
_SECRET = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|password|authorization|credential)s?\\s*[:=]\\s*\\S+"
)
_PATCH_PATH = re.compile(r"diff --git a/(.+?) b/(.+)")
_HUNK_HEADER = re.compile(r"@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@(?: .*)?")
_SAFE_ENVIRONMENT_KEYS: Final[frozenset[str]] = frozenset({"TERM", "TZ", "NO_COLOR"})
_IO_POLL_SECONDS: Final[float] = 0.01
_NAMESPACE_CLEANUP_SECONDS: Final[float] = 2.0
_REAP_GRACE_SECONDS: Final[float] = 1.0
_NAMESPACE_HANDSHAKE_BYTES: Final[int] = 4096
_NAMESPACE_FD_COUNT: Final[int] = 6
_NAMESPACE_GATE_SOURCE: Final[str] = r"""
import array
import ctypes
import errno
import fcntl
import json
import os
import select
import socket
import stat
import struct
import sys
import time

PR_SET_DUMPABLE = 4
PR_SET_NO_NEW_PRIVS = 38
PR_CAPBSET_DROP = 24
LINUX_CAPABILITY_VERSION_3 = 0x20080522
SCMP_ACT_ALLOW = 0x7FFF0000
SCMP_ACT_ERRNO = 0x00050000
SCMP_CMP_EQ = 4
SCMP_CMP_MASKED_EQ = 7
FAILURE = 125
LANDLOCK_CREATE_RULESET = 444
LANDLOCK_ADD_RULE = 445
LANDLOCK_RESTRICT_SELF = 446
LANDLOCK_CREATE_RULESET_VERSION = 1
LANDLOCK_RULE_PATH_BENEATH = 1
LANDLOCK_MINIMUM_ABI = 6
LANDLOCK_SCOPE_ABSTRACT_UNIX_SOCKET = 1 << 0
LANDLOCK_SCOPE_SIGNAL = 1 << 1
LANDLOCK_ACCESS_FS_WRITE_FILE = 1 << 1
LANDLOCK_ACCESS_FS_TRUNCATE = 1 << 14
LANDLOCK_ACCESS_FS_IOCTL_DEV = 1 << 15
LANDLOCK_ACCESS_FS_WRITE = (
    LANDLOCK_ACCESS_FS_WRITE_FILE |
    sum(1 << index for index in range(4, 16))
)
LANDLOCK_ACCESS_FS_DEVICE = (
    LANDLOCK_ACCESS_FS_WRITE_FILE |
    LANDLOCK_ACCESS_FS_TRUNCATE |
    LANDLOCK_ACCESS_FS_IOCTL_DEV
)
DENIED = (
    "mount", "umount2", "pivot_root", "move_mount", "fsopen", "fsconfig",
    "fsmount", "open_tree", "mount_setattr", "unshare", "setns",
)
CLONE_NAMESPACE_FLAGS = (
    0x00020000,  # CLONE_NEWNS
    0x02000000,  # CLONE_NEWCGROUP
    0x04000000,  # CLONE_NEWUTS
    0x08000000,  # CLONE_NEWIPC
    0x10000000,  # CLONE_NEWUSER
    0x20000000,  # CLONE_NEWPID
    0x40000000,  # CLONE_NEWNET
)

class CapHeader(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

class CapData(ctypes.Structure):
    _fields_ = [
        ("effective", ctypes.c_uint32),
        ("permitted", ctypes.c_uint32),
        ("inheritable", ctypes.c_uint32),
    ]

class ScmpArgCmp(ctypes.Structure):
    _fields_ = [
        ("arg", ctypes.c_uint),
        ("op", ctypes.c_int),
        ("datum_a", ctypes.c_uint64),
        ("datum_b", ctypes.c_uint64),
    ]

class LandlockRulesetAttr(ctypes.Structure):
    _fields_ = [
        ("handled_access_fs", ctypes.c_uint64),
        ("handled_access_net", ctypes.c_uint64),
        ("scoped", ctypes.c_uint64),
    ]

class LandlockPathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
    ]

def fail():
    os._exit(FAILURE)

def syscall_number(library, name):
    number = library.seccomp_syscall_resolve_name(name.encode("ascii"))
    if number < 0:
        raise RuntimeError("unresolved syscall:" + name)
    return number

def install_seccomp():
    library = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    library.seccomp_init.argtypes = (ctypes.c_uint32,)
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes = (ctypes.c_char_p,)
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int
    library.seccomp_rule_add.argtypes = (
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint,
    )
    library.seccomp_rule_add.restype = ctypes.c_int
    library.seccomp_rule_add_array.argtypes = (
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint,
        ctypes.POINTER(ScmpArgCmp),
    )
    library.seccomp_rule_add_array.restype = ctypes.c_int
    library.seccomp_load.argtypes = (ctypes.c_void_p,)
    library.seccomp_load.restype = ctypes.c_int
    library.seccomp_release.argtypes = (ctypes.c_void_p,)
    context = library.seccomp_init(SCMP_ACT_ALLOW)
    if not context:
        raise RuntimeError("seccomp_init")
    try:
        denied = SCMP_ACT_ERRNO | errno.EPERM
        for name in DENIED:
            if library.seccomp_rule_add(
                context, denied, syscall_number(library, name), 0
            ) != 0:
                raise RuntimeError("seccomp_rule_add:" + name)

        # clone3 carries flags through a pointed structure, so expose it as
        # unavailable and let ordinary runtimes fall back to clone/fork.
        if library.seccomp_rule_add(
            context,
            SCMP_ACT_ERRNO | errno.ENOSYS,
            syscall_number(library, "clone3"),
            0,
        ) != 0:
            raise RuntimeError("seccomp_rule_add:clone3")
        clone_number = syscall_number(library, "clone")
        for namespace_flag in CLONE_NAMESPACE_FLAGS:
            comparison = ScmpArgCmp(
                0, SCMP_CMP_MASKED_EQ, namespace_flag, namespace_flag
            )
            if library.seccomp_rule_add_array(
                context, denied, clone_number, 1, ctypes.byref(comparison)
            ) != 0:
                raise RuntimeError("seccomp_rule_add:clone")
        dumpable = ScmpArgCmp(0, SCMP_CMP_EQ, PR_SET_DUMPABLE, 0)
        if library.seccomp_rule_add_array(
            context,
            denied,
            syscall_number(library, "prctl"),
            1,
            ctypes.byref(dumpable),
        ) != 0:
            raise RuntimeError("seccomp_rule_add:prctl")
        if library.seccomp_load(context) != 0:
            raise RuntimeError("seccomp_load")
    finally:
        library.seccomp_release(context)

def install_provider_seccomp():
    library = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    library.seccomp_init.argtypes = (ctypes.c_uint32,)
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes = (ctypes.c_char_p,)
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int
    library.seccomp_rule_add.argtypes = (
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint,
    )
    library.seccomp_rule_add.restype = ctypes.c_int
    library.seccomp_load.argtypes = (ctypes.c_void_p,)
    library.seccomp_load.restype = ctypes.c_int
    library.seccomp_release.argtypes = (ctypes.c_void_p,)
    context = library.seccomp_init(SCMP_ACT_ALLOW)
    if not context:
        raise RuntimeError("provider seccomp_init")
    try:
        denied = SCMP_ACT_ERRNO | errno.EPERM
        for name in (
            "socket", "socketpair", "connect", "io_uring_setup", "pidfd_getfd",
        ):
            if library.seccomp_rule_add(
                context, denied, syscall_number(library, name), 0
            ) != 0:
                raise RuntimeError("provider seccomp rule:" + name)
        # Landlock does not mediate these inode-metadata mutations.  Deny each
        # syscall exposed on the supported architecture before untrusted exec.
        for name in (
            "chmod", "fchmod", "fchmodat", "fchmodat2",
            "chown", "fchown", "lchown", "fchownat",
            "utime", "utimes", "futimesat", "utimensat",
            "setxattr", "lsetxattr", "fsetxattr",
            "removexattr", "lremovexattr", "fremovexattr",
        ):
            number = library.seccomp_syscall_resolve_name(name.encode("ascii"))
            if number >= 0 and library.seccomp_rule_add(context, denied, number, 0) != 0:
                raise RuntimeError("provider seccomp metadata rule:" + name)
        if library.seccomp_load(context) != 0:
            raise RuntimeError("provider seccomp_load")
    finally:
        library.seccomp_release(context)

def landlock_abi():
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    abi = int(
        libc.syscall(
            LANDLOCK_CREATE_RULESET,
            ctypes.c_void_p(),
            ctypes.c_size_t(0),
            ctypes.c_uint(LANDLOCK_CREATE_RULESET_VERSION),
        )
    )
    if abi < LANDLOCK_MINIMUM_ABI:
        raise RuntimeError("Landlock ABI is unavailable")
    return abi

def install_provider_landlock(home_fd):
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    landlock_abi()
    attributes = LandlockRulesetAttr(
        LANDLOCK_ACCESS_FS_WRITE,
        0,
        LANDLOCK_SCOPE_ABSTRACT_UNIX_SOCKET | LANDLOCK_SCOPE_SIGNAL,
    )
    ruleset_fd = int(
        libc.syscall(
            LANDLOCK_CREATE_RULESET,
            ctypes.byref(attributes),
            ctypes.sizeof(attributes),
            ctypes.c_uint(0),
        )
    )
    if ruleset_fd < 0:
        raise OSError(ctypes.get_errno(), "Landlock ruleset creation")
    opened = []

    def add_rule(descriptor, access):
        rule = LandlockPathBeneathAttr(access, descriptor)
        if int(
            libc.syscall(
                LANDLOCK_ADD_RULE,
                ruleset_fd,
                LANDLOCK_RULE_PATH_BENEATH,
                ctypes.byref(rule),
                ctypes.c_uint(0),
            )
        ) != 0:
            raise OSError(ctypes.get_errno(), "Landlock path rule")

    def open_rule(path, access, *, expected_device=None):
        resolved = os.path.realpath(path)
        descriptor = os.open(resolved, os.O_PATH | os.O_CLOEXEC | os.O_NOFOLLOW)
        opened.append(descriptor)
        if expected_device is not None:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISCHR(metadata.st_mode)
                or (os.major(metadata.st_rdev), os.minor(metadata.st_rdev))
                != expected_device
            ):
                raise RuntimeError("Landlock runtime device is not trusted")
        add_rule(descriptor, access)

    try:
        add_rule(home_fd, LANDLOCK_ACCESS_FS_WRITE)
        open_rule("/run", LANDLOCK_ACCESS_FS_WRITE)
        for device, identity in (
            ("/dev/null", (1, 3)),
            ("/dev/full", (1, 7)),
            ("/dev/zero", (1, 5)),
        ):
            open_rule(device, LANDLOCK_ACCESS_FS_DEVICE, expected_device=identity)
        if int(libc.syscall(LANDLOCK_RESTRICT_SELF, ruleset_fd, ctypes.c_uint(0))) != 0:
            raise OSError(ctypes.get_errno(), "Landlock restriction")
    finally:
        for descriptor in opened:
            os.close(descriptor)
        os.close(ruleset_fd)

def close_unexpected_fds(allowed):
    for item in os.listdir("/proc/self/fd"):
        try:
            descriptor = int(item)
        except ValueError:
            continue
        if descriptor not in allowed:
            try:
                os.close(descriptor)
            except OSError as exc:
                if exc.errno != errno.EBADF:
                    raise

def drop_capabilities(libc):
    last_cap = int(open("/proc/sys/kernel/cap_last_cap", encoding="ascii").read())
    for capability in range(last_cap + 1):
        if libc.prctl(PR_CAPBSET_DROP, capability, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "capability bounding-set drop")
    header = CapHeader(LINUX_CAPABILITY_VERSION_3, 0)
    data = (CapData * 2)()
    libc.capset.argtypes = (ctypes.POINTER(CapHeader), ctypes.POINTER(CapData))
    libc.capset.restype = ctypes.c_int
    if libc.capset(ctypes.byref(header), data) != 0:
        raise OSError(ctypes.get_errno(), "capability-set drop")
    status = open("/proc/self/status", encoding="ascii").read().splitlines()
    for field in ("CapInh:", "CapPrm:", "CapEff:", "CapBnd:", "CapAmb:"):
        value = next(line.split()[1] for line in status if line.startswith(field))
        if int(value, 16) != 0:
            raise RuntimeError("capabilities remain after drop")

def normalized_status(status):
    if os.WIFEXITED(status):
        return min(255, os.WEXITSTATUS(status))
    if os.WIFSIGNALED(status):
        return min(255, 128 + os.WTERMSIG(status))
    return FAILURE

def id_map(path):
    mapping = []
    for line in open(path, encoding="ascii"):
        fields = line.split()
        if len(fields) != 3:
            raise RuntimeError("identity map is malformed")
        mapping.append([int(item) for item in fields])
    if (
        len(mapping) != 2
        or mapping[0][0] != 0
        or mapping[0][2] != 1
        or mapping[1][0] != 1
        or mapping[1][2] <= 0
    ):
        raise RuntimeError("identity map is incomplete")
    return mapping

try:
    if len(sys.argv) < 10 or sys.argv[9] != "--":
        raise RuntimeError("malformed guarded argv")
    control_fd = int(sys.argv[1])
    cwd_fd = int(sys.argv[2])
    executable_fd = int(sys.argv[3])
    home_fd = int(sys.argv[4])
    nonce = sys.argv[5]
    timeout_ns = int(sys.argv[6])
    cwd = sys.argv[7]
    executable = sys.argv[8]
    anchored_fds = (control_fd, cwd_fd, executable_fd, home_fd)
    if (
        any(descriptor < 3 for descriptor in anchored_fds)
        or len(set(anchored_fds)) != len(anchored_fds)
        or len(nonce) != 64
        or any(character not in "0123456789abcdef" for character in nonce)
        or str(timeout_ns) != sys.argv[6]
        or not 0 < timeout_ns <= 3_600_000_000_000
        or not cwd.startswith("/")
        or not executable.startswith("/")
    ):
        raise RuntimeError("malformed guarded identity")
    for descriptor in anchored_fds:
        os.set_inheritable(descriptor, False)
    control = socket.socket(fileno=control_fd)

    if (
        os.getpid() != 1
        or os.getppid() != 0
        or os.getuid() != 0
        or os.getgid() != 0
    ):
        raise RuntimeError("gate is not namespace PID 1")
    uid_mapping = id_map("/proc/self/uid_map")
    gid_mapping = id_map("/proc/self/gid_map")
    if open("/proc/self/setgroups", encoding="ascii").read().strip() != "allow":
        raise RuntimeError("supplementary groups cannot be cleared")
    os.setgroups([])
    if os.getgroups():
        raise RuntimeError("supplementary groups remain")
    cwd_stat = os.fstat(cwd_fd)
    executable_stat = os.fstat(executable_fd)
    home_stat = os.fstat(home_fd)
    if (
        not stat.S_ISDIR(cwd_stat.st_mode)
        or not stat.S_ISREG(executable_stat.st_mode)
        or not stat.S_ISDIR(home_stat.st_mode)
        or os.path.realpath("/proc/self/fd/" + str(cwd_fd)) != cwd
        or os.path.realpath("/proc/self/fd/" + str(executable_fd)) != executable
        or os.path.realpath("/proc/self/fd/" + str(home_fd)) != os.environ.get("HOME")
        or os.environ.get("TMPDIR") != os.environ.get("HOME")
    ):
        raise RuntimeError("guarded path identity drifted")

    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = (
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    )
    libc.prctl.restype = ctypes.c_int
    libc.mount.argtypes = (
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_ulong, ctypes.c_char_p,
    )
    libc.mount.restype = ctypes.c_int
    if libc.mount(b"tmpfs", b"/run", b"tmpfs", 14, b"size=65536,mode=755") != 0:
        raise OSError(ctypes.get_errno(), "private /run mount")
    if os.listdir("/run"):
        raise RuntimeError("private /run is not empty")
    for path, anchored in (
        (cwd, cwd_stat),
        (executable, executable_stat),
        (os.environ["HOME"], home_stat),
    ):
        visible = os.stat(path, follow_symlinks=False)
        if (visible.st_dev, visible.st_ino) != (anchored.st_dev, anchored.st_ino):
            raise RuntimeError("guarded path was hidden or replaced")
    os.fchdir(cwd_fd)

    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "no_new_privs")
    if libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "non-dumpable gate")
    install_seccomp()
    drop_capabilities(libc)
    admitted_landlock_abi = landlock_abi()

    # A fresh network namespace contains only a down loopback device.  Verify
    # that state after the capability drop so the admitted command cannot gain
    # loopback or external routes by reconfiguring the namespace.
    if {name for _index, name in socket.if_nameindex()} != {"lo"}:
        raise RuntimeError("network namespace has an unexpected interface")
    network_probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        request = struct.pack("16sH14x", b"lo", 0)
        response = fcntl.ioctl(network_probe, 0x8913, request)  # SIOCGIFFLAGS
        if struct.unpack("16sH14x", response)[1] & 1:  # IFF_UP
            raise RuntimeError("private loopback interface is enabled")
    finally:
        network_probe.close()

    owned_fds = [
        os.pidfd_open(1),
        os.open("/proc/self/ns/pid", os.O_RDONLY | os.O_CLOEXEC),
        os.open("/proc/self/ns/mnt", os.O_RDONLY | os.O_CLOEXEC),
        os.open("/proc/self/ns/user", os.O_RDONLY | os.O_CLOEXEC),
        os.open("/proc/self/ns/net", os.O_RDONLY | os.O_CLOEXEC),
        os.open("/proc", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC),
    ]
    metadata = {
        "schema": "pcce-command-namespace-gate@3",
        "nonce": nonce,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "pid_namespace": os.fstat(owned_fds[1]).st_ino,
        "mount_namespace": os.fstat(owned_fds[2]).st_ino,
        "user_namespace": os.fstat(owned_fds[3]).st_ino,
        "network_namespace": os.fstat(owned_fds[4]).st_ino,
        "uid_map": uid_mapping,
        "gid_map": gid_mapping,
        "groups": os.getgroups(),
        "landlock_abi": admitted_landlock_abi,
        "cwd_device": cwd_stat.st_dev,
        "cwd_inode": cwd_stat.st_ino,
        "executable_device": executable_stat.st_dev,
        "executable_inode": executable_stat.st_ino,
        "home_device": home_stat.st_dev,
        "home_inode": home_stat.st_ino,
        "proc_device": os.fstat(owned_fds[5]).st_dev,
        "run_device": os.stat("/run").st_dev,
    }
    rights = array.array("i", owned_fds)
    control.sendmsg(
        [json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("ascii")],
        [(socket.SOL_SOCKET, socket.SCM_RIGHTS, rights)],
    )
    for descriptor in owned_fds:
        os.close(descriptor)
    gate_deadline = time.monotonic() + timeout_ns / 1_000_000_000
    while True:
        remaining = gate_deadline - time.monotonic()
        if remaining <= 0:
            control.send(b"T")
            fail()
        readable, _, _ = select.select(
            (control,), (), (), min(0.01, remaining)
        )
        if readable:
            if control.recv(2) != b"R":
                fail()
            break

    child = os.fork()
    if child == 0:
        control.close()
        try:
            install_provider_seccomp()
            install_provider_landlock(home_fd)
            os.close(cwd_fd)
            os.close(home_fd)
            close_unexpected_fds({0, 1, 2, executable_fd})
            # A shebang interpreter must reopen the fd-backed script after
            # execveat resolves it; keep only this read-only identity anchor.
            os.set_inheritable(executable_fd, True)
            os.execve(executable_fd, [executable, *sys.argv[10:]], os.environ)
        except BaseException:
            os._exit(FAILURE)

    os.close(cwd_fd)
    os.close(executable_fd)
    os.close(home_fd)

    while True:
        waited, status = os.waitpid(child, os.WNOHANG)
        if waited == child:
            os._exit(normalized_status(status))
        remaining = gate_deadline - time.monotonic()
        if remaining <= 0:
            control.send(b"T")
            try:
                os.kill(child, 9)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(child, 0)
            except ChildProcessError:
                pass
            fail()
        readable, _, _ = select.select(
            (control,), (), (), min(0.01, remaining)
        )
        if readable:
            try:
                os.kill(child, 9)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(child, 0)
            except ChildProcessError:
                pass
            fail()
except BaseException:
    fail()
"""


def _cid(value: Any) -> str:
    raw = wire_canonical_utf8(value).encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return "b" + base64.b32encode(b"\x01\x55\x12\x20" + digest).decode().lower().rstrip("=")


def _path(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise MalformedError(f"{field_name} must be a non-empty path")
    if not os.path.isabs(value):
        raise BoundaryViolationError(f"{field_name} must be absolute")
    resolved = os.path.realpath(value)
    if resolved != value or not os.path.exists(resolved):
        raise BoundaryViolationError(f"{field_name} must be an existing canonical path")
    return resolved


def _directory_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
    )


def _executable_identity(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def redact_command_log(value: bytes | str) -> bytes:
    text = value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)
    text = _SECRET.sub("[redacted]", text.replace("\x00", ""))
    text = redact_text(text)
    return text.encode("utf-8")[:MAX_LOG_BYTES]


@dataclass(frozen=True)
class CommandPolicy:
    """Immutable explicit inputs controlling the sole executable and cwd."""

    executable: str
    allowed_executables: tuple[str, ...]
    cwd: str
    allowed_cwds: tuple[str, ...]
    arguments: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float = 120.0
    _admitted_executable_identity: tuple[int, int, int, int, int, int, int, int] = field(
        init=False, repr=False, compare=False
    )
    _admitted_cwd_identity: tuple[int, int, int, int, int] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        executable = _path(self.executable, field_name="executable")
        allowed = tuple(
            _path(item, field_name="allowed_executables") for item in self.allowed_executables
        )
        cwd = _path(self.cwd, field_name="cwd")
        cwds = tuple(_path(item, field_name="allowed_cwds") for item in self.allowed_cwds)
        if (
            executable not in allowed
            or cwd not in cwds
            or not os.path.isdir(cwd)
            or not os.path.isfile(executable)
            or not os.access(executable, os.X_OK)
        ):
            raise BoundaryViolationError("command policy is outside its immutable allowlist")
        try:
            executable_metadata = os.stat(executable, follow_symlinks=False)
            cwd_metadata = os.stat(cwd, follow_symlinks=False)
        except OSError as exc:
            raise BoundaryViolationError("command policy identity cannot be captured") from exc
        if not 0 < self.timeout_seconds <= 3600:
            raise MalformedError("timeout_seconds is outside the safe bound")
        if len(self.arguments) > MAX_ARGUMENTS:
            raise BoundaryViolationError("too many command arguments")
        args: list[str] = []
        for arg in self.arguments:
            if (
                not isinstance(arg, str)
                or "\x00" in arg
                or len(arg.encode("utf-8")) > MAX_ARGUMENT_BYTES
            ):
                raise MalformedError("command argument is malformed")
            args.append(arg)
        safe_env: dict[str, str] = {}
        for key, value in self.environment.items():
            if not isinstance(key, str) or not re.fullmatch(r"[A-Z_][A-Z0-9_]{0,63}", key):
                raise MalformedError("environment key is malformed")
            if (
                key not in _SAFE_ENVIRONMENT_KEYS
                or _SECRET.search(key)
                or not isinstance(value, str)
                or "\x00" in value
                or len(value) > 4096
            ):
                raise BoundaryViolationError("command environment contains a forbidden value")
            safe_env[key] = value
        object.__setattr__(self, "executable", executable)
        object.__setattr__(self, "allowed_executables", allowed)
        object.__setattr__(self, "cwd", cwd)
        object.__setattr__(self, "allowed_cwds", cwds)
        object.__setattr__(self, "arguments", tuple(args))
        object.__setattr__(self, "environment", MappingProxyType(safe_env))
        object.__setattr__(
            self,
            "_admitted_executable_identity",
            _executable_identity(executable_metadata),
        )
        object.__setattr__(self, "_admitted_cwd_identity", _directory_identity(cwd_metadata))


@dataclass(frozen=True)
class CommandExecution:
    stdout: bytes
    stderr: bytes
    returncode: int
    latency_ms: int

    @property
    def log_bytes(self) -> bytes:
        return redact_command_log(self.stderr)


@dataclass(frozen=True)
class _PolicyAnchors:
    """Open identities passed unchanged through the trusted namespace launch."""

    cwd_fd: int
    executable_fd: int
    home_fd: int
    home_identity: tuple[int, int]


@dataclass(frozen=True)
class _StartedLauncher:
    """Identity anchor retained until the trusted namespace launcher is reaped."""

    leader_pid: int
    pgid: int
    session_id: int
    leader_start_time: int


@dataclass(frozen=True)
class _StartedNamespace:
    """Kernel identity and lifetime handles for one gated command namespace."""

    control: socket.socket
    init_pidfd: int
    pid_namespace_fd: int
    mount_namespace_fd: int
    user_namespace_fd: int
    network_namespace_fd: int
    procfs_fd: int
    init_host_pid: int
    pid_namespace_inode: int


def _process_stat(pid: int) -> tuple[str, int, int, int] | None:
    """Return Linux state, process group, session, and immutable start time."""

    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise UnavailableCapabilityError("process identity cannot be inspected") from exc
    fields = raw.rsplit(")", 1)
    if len(fields) != 2:
        raise UnavailableCapabilityError("process identity is malformed")
    values = fields[1].split()
    if len(values) < 20:
        raise UnavailableCapabilityError("process identity is incomplete")
    try:
        return values[0], int(values[2]), int(values[3]), int(values[19])
    except ValueError as exc:
        raise UnavailableCapabilityError("process identity is malformed") from exc


def _capture_started_launcher(process: subprocess.Popen[bytes]) -> _StartedLauncher:
    identity = _process_stat(process.pid)
    if identity is None:
        raise UnavailableCapabilityError("started process identity is unavailable")
    _, pgid, session_id, start_time = identity
    if pgid != process.pid or session_id != process.pid:
        raise UnavailableCapabilityError("started process is not its isolated session leader")
    return _StartedLauncher(process.pid, pgid, session_id, start_time)


def _require_safe_process_supervision() -> None:
    required_os_features = (
        "P_PID",
        "WEXITED",
        "WNOHANG",
        "WNOWAIT",
        "waitid",
        "pidfd_open",
        "O_PATH",
        "O_NOFOLLOW",
        "O_CLOEXEC",
        "O_DIRECTORY",
    )
    required_socket_features = (
        "AF_UNIX",
        "SOCK_SEQPACKET",
        "SOL_SOCKET",
        "SO_PASSCRED",
        "SCM_CREDENTIALS",
        "SCM_RIGHTS",
        "MSG_CMSG_CLOEXEC",
        "MSG_TRUNC",
        "MSG_CTRUNC",
        "CMSG_SPACE",
        "socketpair",
    )
    required_select_features = ("poll", "POLLIN", "POLLERR", "POLLHUP")
    if (
        any(not hasattr(os, name) for name in required_os_features)
        or not Path("/proc/self/stat").is_file()
        or not hasattr(signal, "pidfd_send_signal")
        or any(not hasattr(socket, name) for name in required_socket_features)
        or any(not hasattr(select, name) for name in required_select_features)
        or os.execve not in os.supports_fd
        or os.uname().sysname != "Linux"
        or os.uname().machine not in {"aarch64", "x86_64"}
        or signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL
    ):
        raise UnavailableCapabilityError("safe PID-namespace supervision is unavailable")


def _trusted_root_executable(path: Path, *, capability: str) -> str:
    try:
        resolved = path.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as exc:
        raise UnavailableCapabilityError(f"trusted {capability} is unavailable") from exc
    if (
        not resolved.is_file()
        or not os.access(resolved, os.X_OK)
        or metadata.st_uid != 0
        or metadata.st_mode & 0o022
    ):
        raise UnavailableCapabilityError(f"trusted {capability} is unavailable")
    return str(resolved)


def _require_trusted_mapping_helpers() -> None:
    for helper in (Path("/usr/bin/newuidmap"), Path("/usr/bin/newgidmap")):
        resolved = _trusted_root_executable(helper, capability="identity mapping helper")
        if not os.stat(resolved, follow_symlinks=False).st_mode & stat.S_ISUID:
            raise UnavailableCapabilityError("trusted identity mapping helper is unavailable")


def _open_policy_anchors(policy: CommandPolicy, home: str) -> _PolicyAnchors:
    descriptors: list[int] = []
    try:
        if any(
            path == mount or path.startswith(f"{mount}/")
            for path in (policy.cwd, policy.executable, home)
            for mount in ("/proc", "/run")
        ):
            raise BoundaryViolationError(
                "command policy path would be hidden by a private runtime mount"
            )
        cwd_fd = os.open(
            policy.cwd,
            os.O_PATH | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        descriptors.append(cwd_fd)
        executable_fd = os.open(
            policy.executable,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        descriptors.append(executable_fd)
        home_fd = os.open(
            home,
            os.O_PATH | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        descriptors.append(home_fd)
        cwd_metadata = os.fstat(cwd_fd)
        executable_metadata = os.fstat(executable_fd)
        home_metadata = os.fstat(home_fd)
        if (
            _directory_identity(cwd_metadata) != policy._admitted_cwd_identity
            or _executable_identity(executable_metadata) != policy._admitted_executable_identity
            or os.path.realpath(f"/proc/self/fd/{cwd_fd}") != policy.cwd
            or os.path.realpath(f"/proc/self/fd/{executable_fd}") != policy.executable
            or os.path.realpath(f"/proc/self/fd/{home_fd}") != home
            or not stat.S_ISDIR(home_metadata.st_mode)
            or home_metadata.st_uid != os.geteuid()
            or home_metadata.st_mode & 0o077
        ):
            raise BoundaryViolationError("command policy path identity drifted")
        return _PolicyAnchors(
            cwd_fd=cwd_fd,
            executable_fd=executable_fd,
            home_fd=home_fd,
            home_identity=(home_metadata.st_dev, home_metadata.st_ino),
        )
    except OSError as exc:
        _close_descriptors(descriptors)
        raise BoundaryViolationError("command policy paths cannot be anchored") from exc
    except BaseException:
        _close_descriptors(descriptors)
        raise


def _namespace_argv(
    policy: CommandPolicy,
    control_descriptor: int,
    anchors: _PolicyAnchors,
    nonce: str,
    timeout_ns: int,
) -> list[str]:
    """Return the trusted namespace gate argv; only the admitted tail is variable."""

    try:
        interpreter_path = Path(sys.executable).resolve(strict=True)
    except OSError as exc:
        raise UnavailableCapabilityError("trusted Python interpreter is unavailable") from exc
    interpreter = _trusted_root_executable(interpreter_path, capability="Python interpreter")
    busybox = _trusted_root_executable(
        Path("/usr/bin/busybox"), capability="BusyBox namespace launcher"
    )
    unshare = _trusted_root_executable(
        Path("/usr/bin/unshare"), capability="util-linux namespace launcher"
    )
    _require_trusted_mapping_helpers()
    return [
        busybox,
        "env",
        unshare,
        "--map-auto",
        "--map-user=0",
        "--map-group=0",
        "--fork",
        "--pid",
        "--net",
        "--mount",
        "--mount-proc",
        "--propagation",
        "private",
        interpreter,
        "-I",
        "-c",
        _NAMESPACE_GATE_SOURCE,
        str(control_descriptor),
        str(anchors.cwd_fd),
        str(anchors.executable_fd),
        str(anchors.home_fd),
        nonce,
        str(timeout_ns),
        policy.cwd,
        policy.executable,
        "--",
        *policy.arguments,
    ]


def _open_namespace_control_pair() -> tuple[socket.socket, socket.socket]:
    parent_control: socket.socket | None = None
    child_control: socket.socket | None = None
    try:
        parent_control, child_control = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        parent_control.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        parent_control.setblocking(False)
        return parent_control, child_control
    except (OSError, ValueError) as exc:
        for control in (parent_control, child_control):
            if control is not None:
                try:
                    control.close()
                except OSError:
                    pass
        raise UnavailableCapabilityError("safe namespace control channel is unavailable") from exc


def _close_descriptors(descriptors: list[int] | tuple[int, ...]) -> None:
    for descriptor in descriptors:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _receive_namespace_handshake(
    control: socket.socket,
    nonce: str,
    policy: CommandPolicy,
    home_identity: tuple[int, int],
) -> _StartedNamespace:
    """Validate the gated PID-1 sender and adopt its kernel lifetime handles."""

    descriptors: list[int] = []
    try:
        packet, ancillary, flags, _address = control.recvmsg(
            _NAMESPACE_HANDSHAKE_BYTES,
            socket.CMSG_SPACE(_NAMESPACE_FD_COUNT * array.array("i").itemsize)
            + socket.CMSG_SPACE(struct.calcsize("3i")),
            socket.MSG_CMSG_CLOEXEC,
        )
        if not packet or flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
            raise UnavailableCapabilityError("namespace gate handshake is truncated")
        credentials: list[tuple[int, int, int]] = []
        for level, kind, value in ancillary:
            if level != socket.SOL_SOCKET:
                continue
            if kind == socket.SCM_RIGHTS:
                admitted = array.array("i")
                admitted.frombytes(value[: len(value) - (len(value) % admitted.itemsize)])
                descriptors.extend(admitted)
            elif kind == socket.SCM_CREDENTIALS and len(value) >= struct.calcsize("3i"):
                credentials.append(struct.unpack("3i", value[: struct.calcsize("3i")]))
        if len(descriptors) != _NAMESPACE_FD_COUNT or len(credentials) != 1:
            raise UnavailableCapabilityError("namespace gate handles are incomplete")
        sender_pid, sender_uid, sender_gid = credentials[0]
        if sender_uid != os.geteuid() or sender_gid != os.getegid() or sender_pid <= 0:
            raise UnavailableCapabilityError("namespace gate sender identity is inconsistent")

        try:
            metadata = json.loads(packet.decode("ascii"))
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise UnavailableCapabilityError("namespace gate metadata is malformed") from exc
        expected_fields = {
            "schema",
            "nonce",
            "pid",
            "ppid",
            "pid_namespace",
            "mount_namespace",
            "user_namespace",
            "network_namespace",
            "uid_map",
            "gid_map",
            "groups",
            "landlock_abi",
            "cwd_device",
            "cwd_inode",
            "executable_device",
            "executable_inode",
            "home_device",
            "home_inode",
            "proc_device",
            "run_device",
        }
        if (
            not isinstance(metadata, dict)
            or set(metadata) != expected_fields
            or metadata["schema"] != "pcce-command-namespace-gate@3"
            or metadata["nonce"] != nonce
            or metadata["pid"] != 1
            or metadata["ppid"] != 0
        ):
            raise UnavailableCapabilityError("namespace gate metadata is inconsistent")

        def valid_identity_map(value: object, outer_id: int) -> bool:
            return (
                isinstance(value, list)
                and len(value) == 2
                and all(
                    isinstance(row, list)
                    and len(row) == 3
                    and all(type(item) is int and item >= 0 for item in row)
                    for row in value
                )
                and value[0] == [0, outer_id, 1]
                and value[1][0] == 1
                and value[1][2] > 0
                and not (value[1][1] <= outer_id < value[1][1] + value[1][2])
            )

        if (
            not valid_identity_map(metadata["uid_map"], os.geteuid())
            or not valid_identity_map(metadata["gid_map"], os.getegid())
            or metadata["groups"] != []
            or type(metadata["landlock_abi"]) is not int
            or metadata["landlock_abi"] < 6
            or (metadata["cwd_device"], metadata["cwd_inode"]) != policy._admitted_cwd_identity[:2]
            or (metadata["executable_device"], metadata["executable_inode"])
            != policy._admitted_executable_identity[:2]
            or (metadata["home_device"], metadata["home_inode"]) != home_identity
        ):
            raise UnavailableCapabilityError("guarded command identity is inconsistent")

        (
            init_pidfd,
            pid_namespace_fd,
            mount_namespace_fd,
            user_namespace_fd,
            network_namespace_fd,
            procfs_fd,
        ) = descriptors
        fdinfo = Path(f"/proc/self/fdinfo/{init_pidfd}").read_text(encoding="ascii")
        fdinfo_fields = {
            key: value.strip()
            for key, value in (line.split(":", 1) for line in fdinfo.splitlines() if ":" in line)
        }
        if int(fdinfo_fields.get("Pid", "-1")) != sender_pid:
            raise UnavailableCapabilityError("namespace PID 1 pidfd is inconsistent")
        nested_pids = tuple(map(int, fdinfo_fields.get("NSpid", "").split()))
        if len(nested_pids) < 2 or nested_pids[0] != sender_pid or nested_pids[-1] != 1:
            raise UnavailableCapabilityError("namespace PID mapping is inconsistent")

        namespace_bindings = (
            (pid_namespace_fd, "pid", "pid_namespace"),
            (mount_namespace_fd, "mnt", "mount_namespace"),
            (user_namespace_fd, "user", "user_namespace"),
            (network_namespace_fd, "net", "network_namespace"),
        )
        for descriptor, namespace_kind, metadata_field in namespace_bindings:
            inode = os.fstat(descriptor).st_ino
            if metadata[metadata_field] != inode:
                raise UnavailableCapabilityError("namespace descriptor metadata drifted")
            if os.readlink(f"/proc/self/fd/{descriptor}") != f"{namespace_kind}:[{inode}]":
                raise UnavailableCapabilityError("namespace descriptor type is inconsistent")
            if inode == os.stat(f"/proc/self/ns/{namespace_kind}").st_ino:
                raise UnavailableCapabilityError("namespace gate did not isolate its namespaces")
        if not isinstance(metadata["run_device"], int) or metadata["run_device"] <= 0:
            raise UnavailableCapabilityError("private runtime mount is inconsistent")
        if metadata["run_device"] == os.stat("/run").st_dev:
            raise UnavailableCapabilityError("runtime mount was not isolated")
        if (
            not isinstance(metadata["proc_device"], int)
            or metadata["proc_device"] != os.fstat(procfs_fd).st_dev
            or metadata["proc_device"] == os.stat("/proc").st_dev
            or "1" not in os.listdir(procfs_fd)
        ):
            raise UnavailableCapabilityError("private procfs mount is inconsistent")
        if _pidfd_exited(init_pidfd):
            raise UnavailableCapabilityError("namespace PID 1 exited before release")
        return _StartedNamespace(
            control=control,
            init_pidfd=init_pidfd,
            pid_namespace_fd=pid_namespace_fd,
            mount_namespace_fd=mount_namespace_fd,
            user_namespace_fd=user_namespace_fd,
            network_namespace_fd=network_namespace_fd,
            procfs_fd=procfs_fd,
            init_host_pid=sender_pid,
            pid_namespace_inode=os.fstat(pid_namespace_fd).st_ino,
        )
    except BlockingIOError:
        _close_descriptors(descriptors)
        raise
    except UnavailableCapabilityError:
        _close_descriptors(descriptors)
        raise
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        _close_descriptors(descriptors)
        raise UnavailableCapabilityError("namespace gate handshake cannot be verified") from exc
    except BaseException:
        _close_descriptors(descriptors)
        raise


def _pidfd_exited(pidfd: int) -> bool:
    poller = select.poll()
    poller.register(pidfd, select.POLLIN | select.POLLERR | select.POLLHUP)
    return bool(poller.poll(0))


def _leader_is_anchored(group: _StartedLauncher) -> bool:
    identity = _process_stat(group.leader_pid)
    return identity is not None and identity[1:] == (
        group.pgid,
        group.session_id,
        group.leader_start_time,
    )


def _leader_exited(group: _StartedLauncher) -> bool:
    """Observe exit without reaping the leader and releasing its PID/PGID anchor."""

    if not _leader_is_anchored(group):
        raise UnavailableCapabilityError("started process identity was lost before cleanup")
    try:
        status = os.waitid(
            os.P_PID,
            group.leader_pid,
            os.WEXITED | os.WNOHANG | os.WNOWAIT,
        )
    except ChildProcessError as exc:
        raise UnavailableCapabilityError("started process was reaped outside the adapter") from exc
    return status is not None


def _unreaped_child_exited(process: subprocess.Popen[bytes]) -> bool:
    """Observe a direct child's exit while retaining its kernel PID anchor."""

    try:
        status = os.waitid(
            os.P_PID,
            process.pid,
            os.WEXITED | os.WNOHANG | os.WNOWAIT,
        )
    except ChildProcessError as exc:
        raise UnavailableCapabilityError("started process was reaped outside the adapter") from exc
    return status is not None


def _launcher_exited(process: subprocess.Popen[bytes], group: _StartedLauncher | None) -> bool:
    return _leader_exited(group) if group is not None else _unreaped_child_exited(process)


def _group_has_live_members(pgid: int, session_id: int) -> bool:
    try:
        entries = os.scandir("/proc")
    except OSError as exc:
        raise UnavailableCapabilityError("trusted launcher group cannot be inspected") from exc
    with entries:
        for entry in entries:
            if not entry.name.isdecimal():
                continue
            identity = _process_stat(int(entry.name))
            if identity is None:
                continue
            state, observed_pgid, observed_session, _start_time = identity
            if observed_pgid == pgid and observed_session == session_id and state not in {"Z", "X"}:
                return True
    return False


def _kill_unreleased_launcher_group(
    process: subprocess.Popen[bytes], group: _StartedLauncher | None
) -> None:
    """Kill the still-trusted pre-release session as one race-free unit."""

    pgid = process.pid if group is None else group.pgid
    session_id = process.pid if group is None else group.session_id
    if group is not None and not _leader_is_anchored(group):
        raise UnavailableCapabilityError("refusing to signal an unanchored launcher group")
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + _NAMESPACE_CLEANUP_SECONDS
    while _group_has_live_members(pgid, session_id):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise UnavailableCapabilityError(
                "trusted launcher group did not terminate within its bound"
            )
        time.sleep(min(_IO_POLL_SECONDS, remaining))


def _namespace_has_live_members(namespace: _StartedNamespace) -> bool:
    try:
        entries = os.listdir(namespace.procfs_fd)
    except OSError as exc:
        raise UnavailableCapabilityError("PID namespace membership cannot be inspected") from exc
    for entry in entries:
        if not entry.isdecimal():
            continue
        descriptor = -1
        try:
            descriptor = os.open(
                f"{entry}/stat",
                os.O_RDONLY | os.O_CLOEXEC,
                dir_fd=namespace.procfs_fd,
            )
            raw = os.read(descriptor, 8192).decode("ascii")
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise UnavailableCapabilityError("PID namespace membership cannot be proven") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        fields = raw.rsplit(")", 1)
        if len(fields) != 2 or not fields[1].split():
            raise UnavailableCapabilityError("PID namespace process state is malformed")
        if fields[1].split()[0] not in {"Z", "X"}:
            return True
    return False


def _terminate_started_namespace(namespace: _StartedNamespace) -> None:
    """Kill namespace PID 1, then prove its anchored namespace has no live member."""

    if not _pidfd_exited(namespace.init_pidfd):
        try:
            signal.pidfd_send_signal(namespace.init_pidfd, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + _NAMESPACE_CLEANUP_SECONDS
    while not _pidfd_exited(namespace.init_pidfd):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise UnavailableCapabilityError("namespace PID 1 did not terminate within its bound")
        time.sleep(min(_IO_POLL_SECONDS, remaining))
    while _namespace_has_live_members(namespace):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise UnavailableCapabilityError(
                "PID namespace descendant cleanup could not be proven within its bound"
            )
        time.sleep(min(_IO_POLL_SECONDS, remaining))


def _terminate_started_launcher(
    process: subprocess.Popen[bytes], group: _StartedLauncher | None
) -> None:
    if _launcher_exited(process, group):
        return
    if group is not None and not _leader_is_anchored(group):
        raise UnavailableCapabilityError("refusing to signal an unanchored namespace launcher")
    grace_deadline = time.monotonic() + 0.25
    while not _launcher_exited(process, group) and time.monotonic() < grace_deadline:
        time.sleep(_IO_POLL_SECONDS)
    if _launcher_exited(process, group):
        return
    try:
        os.kill(process.pid, signal.SIGCONT)
        os.kill(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + _REAP_GRACE_SECONDS
    while not _launcher_exited(process, group):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise UnavailableCapabilityError(
                "namespace launcher did not terminate within its bound"
            )
        time.sleep(min(_IO_POLL_SECONDS, remaining))


def _close_started_namespace(namespace: _StartedNamespace) -> None:
    try:
        namespace.control.close()
    finally:
        _close_descriptors(
            (
                namespace.init_pidfd,
                namespace.pid_namespace_fd,
                namespace.mount_namespace_fd,
                namespace.user_namespace_fd,
                namespace.network_namespace_fd,
                namespace.procfs_fd,
            )
        )


def invoke_command(
    policy: CommandPolicy, request: Mapping[str, Any], cancellation: CancellationToken | None = None
) -> CommandExecution:
    """Execute one allowlisted argv process with no inherited environment."""
    if cancellation is not None:
        cancellation.check()
    _require_safe_process_supervision()
    payload = wire_canonical_utf8(dict(request)).encode("utf-8")
    if len(payload) > MAX_PROVIDER_OUTPUT_BYTES:
        raise BoundaryViolationError("command request exceeds the frozen byte bound")
    with tempfile.TemporaryDirectory(prefix="pcce-command-") as home:
        env = {
            "HOME": home,
            "TMPDIR": home,
            "XDG_CACHE_HOME": f"{home}/.cache",
            "XDG_CONFIG_HOME": f"{home}/.config",
            "XDG_DATA_HOME": f"{home}/.local/share",
            "XDG_STATE_HOME": f"{home}/.local/state",
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }
        env.update(policy.environment)
        anchors: _PolicyAnchors | None = _open_policy_anchors(policy, home)
        home_identity = anchors.home_identity
        parent_control: socket.socket | None = None
        child_control: socket.socket | None = None
        try:
            parent_control, child_control = _open_namespace_control_pair()
            nonce = os.urandom(32).hex()
            timeout_ns = max(1, int(policy.timeout_seconds * 1_000_000_000))
            supervised_argv = _namespace_argv(
                policy,
                child_control.fileno(),
                anchors,
                nonce,
                timeout_ns,
            )
        except BaseException:
            for control in (parent_control, child_control):
                if control is not None:
                    control.close()
            _close_descriptors((anchors.cwd_fd, anchors.executable_fd, anchors.home_fd))
            raise
        assert parent_control is not None and child_control is not None
        group: _StartedLauncher | None = None
        namespace: _StartedNamespace | None = None
        selector: selectors.BaseSelector | None = None
        stdin: Any = None
        stdout_pipe: Any = None
        stderr_pipe: Any = None
        stdout, stderr = bytearray(), bytearray()
        destinations: dict[Any, bytearray] = {}
        remaining_budget = MAX_PROVIDER_OUTPUT_BYTES
        payload_view = memoryview(payload)
        payload_offset = 0
        exceeded = False
        reason: str | None = None
        returncode: int | None = None

        def close_stream(stream: Any) -> None:
            if stream is None:
                return
            if selector is not None:
                try:
                    selector.unregister(stream)
                except (KeyError, OSError, ValueError):
                    pass
            if not getattr(stream, "closed", True):
                stream.close()

        def collect_output(stream: Any) -> None:
            nonlocal exceeded, remaining_budget
            try:
                chunk = os.read(stream.fileno(), 65_536)
            except BlockingIOError:
                return
            if not chunk:
                close_stream(stream)
                return
            available = min(remaining_budget, len(chunk))
            if available:
                destinations[stream].extend(chunk[:available])
                remaining_budget -= available
            if available < len(chunk):
                exceeded = True

        try:
            started = time.monotonic()
            process = subprocess.Popen(
                supervised_argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd="/",
                env=env,
                shell=False,
                close_fds=True,
                pass_fds=(
                    child_control.fileno(),
                    anchors.cwd_fd,
                    anchors.executable_fd,
                    anchors.home_fd,
                ),
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            parent_control.close()
            child_control.close()
            _close_descriptors((anchors.cwd_fd, anchors.executable_fd, anchors.home_fd))
            raise UnavailableCapabilityError("trusted namespace launcher is unavailable") from exc
        except OSError as exc:
            parent_control.close()
            child_control.close()
            _close_descriptors((anchors.cwd_fd, anchors.executable_fd, anchors.home_fd))
            raise UnavailableCapabilityError(
                "trusted namespace launcher cannot be started"
            ) from exc

        try:
            stdin, stdout_pipe, stderr_pipe = process.stdin, process.stdout, process.stderr
            child_control.close()
            _close_descriptors((anchors.cwd_fd, anchors.executable_fd, anchors.home_fd))
            anchors = None
            deadline = started + policy.timeout_seconds
            while namespace is None and reason is None:
                if cancellation is not None and cancellation.cancelled:
                    reason = "cancelled"
                    break
                if time.monotonic() >= deadline:
                    reason = "timeout"
                    break
                if _unreaped_child_exited(process):
                    raise UnavailableCapabilityError(
                        "namespace launcher exited before its gated handshake"
                    )
                try:
                    namespace = _receive_namespace_handshake(
                        parent_control,
                        nonce,
                        policy,
                        home_identity,
                    )
                except BlockingIOError:
                    remaining = deadline - time.monotonic()
                    select.select(
                        (parent_control,),
                        (),
                        (),
                        min(_IO_POLL_SECONDS, max(0.0, remaining)),
                    )
            if namespace is not None:
                while True:
                    try:
                        if parent_control.send(b"R") != 1:
                            raise UnavailableCapabilityError(
                                "namespace gate release was incomplete"
                            )
                        break
                    except BlockingIOError:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            reason = "timeout"
                            break
                        select.select(
                            (),
                            (parent_control,),
                            (),
                            min(_IO_POLL_SECONDS, remaining),
                        )
            if reason is not None:
                returncode = None
            else:
                assert namespace is not None
            group = _capture_started_launcher(process)
            if stdin is None or stdout_pipe is None or stderr_pipe is None:
                raise UnavailableCapabilityError("command pipes are unavailable")
            destinations = {stdout_pipe: stdout, stderr_pipe: stderr}
            for stream in (stdin, stdout_pipe, stderr_pipe):
                os.set_blocking(stream.fileno(), False)
            selector = selectors.DefaultSelector()
            selector.register(stdin, selectors.EVENT_WRITE)
            selector.register(stdout_pipe, selectors.EVENT_READ)
            selector.register(stderr_pipe, selectors.EVENT_READ)
            while reason is None and not _pidfd_exited(namespace.init_pidfd):
                if cancellation is not None and cancellation.cancelled:
                    reason = "cancelled"
                elif exceeded:
                    reason = "output"
                elif _leader_exited(group):
                    reason = "launcher"
                elif time.monotonic() >= deadline:
                    reason = "timeout"
                if reason:
                    break
                wait_seconds = min(_IO_POLL_SECONDS, max(0.0, deadline - time.monotonic()))
                assert selector is not None
                for key, _ in selector.select(wait_seconds):
                    stream = key.fileobj
                    if stream is stdin:
                        try:
                            written = os.write(stream.fileno(), payload_view[payload_offset:])
                        except (BrokenPipeError, ConnectionResetError):
                            close_stream(stream)
                        else:
                            payload_offset += written
                            if payload_offset == len(payload):
                                close_stream(stream)
                    else:
                        collect_output(stream)
            if reason is None:
                try:
                    gate_notice = parent_control.recv(2)
                except BlockingIOError:
                    gate_notice = None
                if gate_notice == b"T":
                    reason = "timeout"
                elif gate_notice not in (None, b""):
                    raise UnavailableCapabilityError(
                        "namespace gate sent an invalid terminal notice"
                    )
        finally:
            cleanup_error: BaseException | None = None

            def remember_cleanup_error(exc: BaseException) -> None:
                nonlocal cleanup_error
                if cleanup_error is None:
                    cleanup_error = exc

            try:
                close_stream(stdin)
            except BaseException as exc:
                remember_cleanup_error(exc)
            try:
                if namespace is None:
                    _kill_unreleased_launcher_group(process, group)
                else:
                    _terminate_started_namespace(namespace)
            except BaseException as exc:
                remember_cleanup_error(exc)
            try:
                _terminate_started_launcher(process, group)
            except BaseException as exc:
                remember_cleanup_error(exc)
            try:
                process.wait(timeout=_REAP_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                remember_cleanup_error(
                    UnavailableCapabilityError("namespace launcher could not be reaped")
                )
            except BaseException as exc:
                remember_cleanup_error(exc)
            else:
                returncode = process.returncode
                for stream in (stdout_pipe, stderr_pipe):
                    if stream is None or stream not in destinations:
                        continue
                    try:
                        while not stream.closed:
                            before = len(destinations[stream])
                            collect_output(stream)
                            if stream.closed or len(destinations[stream]) == before:
                                break
                    except BaseException as exc:
                        remember_cleanup_error(exc)
            for stream in (stdout_pipe, stderr_pipe):
                try:
                    close_stream(stream)
                except BaseException as exc:
                    remember_cleanup_error(exc)
            if selector is not None:
                try:
                    selector.close()
                except BaseException as exc:
                    remember_cleanup_error(exc)
            try:
                child_control.close()
            except BaseException as exc:
                remember_cleanup_error(exc)
            if anchors is not None:
                try:
                    _close_descriptors((anchors.cwd_fd, anchors.executable_fd, anchors.home_fd))
                    anchors = None
                except BaseException as exc:
                    remember_cleanup_error(exc)
            if namespace is None:
                try:
                    parent_control.close()
                except BaseException as exc:
                    remember_cleanup_error(exc)
            else:
                try:
                    _close_started_namespace(namespace)
                except BaseException as exc:
                    remember_cleanup_error(exc)
            if cleanup_error is not None:
                raise cleanup_error
        latency = max(0, int((time.monotonic() - started) * 1000))
    if cancellation is not None and cancellation.cancelled:
        raise ProofCancelledError("command invocation cancelled")
    if exceeded:
        raise BoundaryViolationError("command output exceeds the frozen byte bound")
    # A successful process may finish at the deadline; only a still-running process was killed.
    if reason == "timeout":
        raise ProofTimeoutError("command invocation timed out")
    if reason == "launcher":
        raise UnavailableCapabilityError("namespace launcher exited before PID 1")
    assert returncode is not None
    return CommandExecution(bytes(stdout), bytes(stderr), int(returncode), latency)


def decode_structured_output(stdout: bytes | str) -> Mapping[str, Any]:
    """Decode exactly one closed JSON proposal object; no fences or prose."""
    raw = stdout.encode("utf-8") if isinstance(stdout, str) else bytes(stdout)
    if not raw or len(raw) > MAX_PROVIDER_OUTPUT_BYTES:
        raise MalformedError("command stdout is empty or exceeds its bound")
    try:
        text = raw.decode("utf-8")

        def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            admitted: dict[str, Any] = {}
            for key, item in pairs:
                if key in admitted:
                    raise ValueError("duplicate JSON object key")
                admitted[key] = item
            return admitted

        def reject_non_finite_number(value: str) -> None:
            raise ValueError(f"non-finite JSON number {value}")

        decoder = json.JSONDecoder(
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_non_finite_number,
        )
        value, index = decoder.raw_decode(text.lstrip())
        if text.lstrip()[index:].strip():
            raise ValueError("trailing data")
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise MalformedError("command stdout is not exactly one JSON object") from exc
    if not isinstance(value, dict) or set(value) != _OUTPUT_FIELDS:
        raise MalformedError("command proposal has an invalid closed field set")
    return MappingProxyType(value)


def _admit_patch_scope(patch: bytes, declared: tuple[str, ...], task: TaskSpecification) -> None:
    """Require every changed path to be both declared and owned."""
    text = patch.decode("utf-8", "strict")
    lines = text.splitlines()
    if not lines:
        raise MalformedError("command patch must be a unified diff")

    def admit_scoped_path(raw: str, *, prefix: str | None = None) -> str | None:
        if raw == "/dev/null":
            if prefix is None:
                raise MalformedError("command diff header cannot use /dev/null")
            return None
        if prefix is not None:
            marker = f"{prefix}/"
            if not raw.startswith(marker):
                raise MalformedError("command patch has a malformed file marker")
            raw = raw[len(marker) :]
        path = admit_relative_path(raw, field="patch_path")
        assert_declared_scope((path,), task.owned_paths, task.declared_files)
        if path not in declared:
            raise BoundaryViolationError(
                "command patch path is not declared",
                details={"field": "declared_files"},
            )
        return path

    index = 0
    while index < len(lines):
        match = _PATCH_PATH.fullmatch(lines[index])
        if match is None:
            raise MalformedError("command patch must contain only git diff sections")
        old_path = admit_scoped_path(match.group(1))
        new_path = admit_scoped_path(match.group(2))
        if old_path is None or new_path is None:
            raise MalformedError("command diff header must name both repository paths")
        index += 1
        hunk_remaining: tuple[int, int] | None = None
        file_markers_seen = False
        while index < len(lines) and not lines[index].startswith("diff --git "):
            line = lines[index]
            if hunk_remaining is not None:
                old_remaining, new_remaining = hunk_remaining
                if old_remaining == 0 and new_remaining == 0:
                    hunk_remaining = None
                    continue
                if line.startswith(" "):
                    old_remaining -= 1
                    new_remaining -= 1
                elif line.startswith("-"):
                    old_remaining -= 1
                elif line.startswith("+"):
                    new_remaining -= 1
                elif line == r"\ No newline at end of file":
                    index += 1
                    continue
                else:
                    raise MalformedError("command patch hunk has a malformed content line")
                if old_remaining < 0 or new_remaining < 0:
                    raise MalformedError("command patch hunk exceeds its declared line count")
                hunk_remaining = (old_remaining, new_remaining)
            elif line.startswith("@@ "):
                hunk = _HUNK_HEADER.fullmatch(line)
                if hunk is None:
                    raise MalformedError("command patch has a malformed hunk header")
                hunk_remaining = (
                    int(hunk.group(1)) if hunk.group(1) is not None else 1,
                    int(hunk.group(2)) if hunk.group(2) is not None else 1,
                )
            elif line.startswith("--- "):
                if file_markers_seen:
                    raise MalformedError("command patch repeats its file markers")
                if index + 1 >= len(lines) or not lines[index + 1].startswith("+++ "):
                    raise MalformedError("command patch has an unpaired file marker")
                admitted_old = admit_scoped_path(line[4:], prefix="a")
                admitted_new = admit_scoped_path(lines[index + 1][4:], prefix="b")
                if admitted_old not in {None, old_path} or admitted_new not in {None, new_path}:
                    raise BoundaryViolationError(
                        "command patch file markers drift from its diff header"
                    )
                if admitted_old is None and admitted_new is None:
                    raise MalformedError("command patch file markers cannot both be /dev/null")
                file_markers_seen = True
                index += 1
            elif line.startswith("+++ "):
                raise MalformedError("command patch has an unpaired file marker")
            elif line.startswith(("rename from ", "copy from ")):
                metadata_path = admit_scoped_path(line.split(" ", 2)[2])
                if metadata_path != old_path:
                    raise BoundaryViolationError(
                        "command patch source path drifts from its diff header"
                    )
            elif line.startswith(("rename to ", "copy to ")):
                metadata_path = admit_scoped_path(line.split(" ", 2)[2])
                if metadata_path != new_path:
                    raise BoundaryViolationError(
                        "command patch destination path drifts from its diff header"
                    )
            index += 1
        if hunk_remaining not in {None, (0, 0)}:
            raise MalformedError("command patch hunk ends before its declared line count")


def build_command_request(
    task: TaskSpecification, pack: ContextPack, route: ModelRouteDecision
) -> Mapping[str, Any]:
    bind_adapter_request(task, pack, route)
    return MappingProxyType(
        {
            "schema": REQUEST_SCHEMA,
            "task_id": task.task_id,
            "objective_id": task.objective_id,
            "repository_state_cid": task.repository_state_cid,
            "pack_cid": pack.pack_cid,
            "route_cid": route.decision_cid,
            "owned_paths": list(task.owned_paths),
            "declared_files": list(task.declared_files or task.owned_paths),
            "provider": route.provider,
            "model": route.model,
            "revision": route.revision or "unspecified",
            "tier": route.tier,
            "instruction": "Return exactly one JSON proposal object and no prose; never approve or apply a patch.",
        }
    )


@dataclass
class CommandAdapter:
    policy: CommandPolicy

    def __post_init__(self) -> None:
        self.accepted = False
        self.approved = False
        self.approval_authority = APPROVAL_AUTHORITY
        self.canonical_branch_authority = CANONICAL_BRANCH_AUTHORITY

    def cancel(self, cancellation: CancellationToken) -> None:
        cancellation.cancel()

    def propose(
        self,
        task: TaskSpecification,
        context_pack: ContextPack,
        route: ModelRouteDecision,
        cancellation: CancellationToken | None = None,
    ) -> AdapterResult:
        request = build_command_request(task, context_pack, route)
        execution = invoke_command(self.policy, request, cancellation)
        if execution.returncode != 0:
            raise UnavailableCapabilityError("local command returned a non-zero exit status")
        output = decode_structured_output(execution.stdout)
        for name, expected in (
            ("task_id", task.task_id),
            ("repository_state_cid", task.repository_state_cid),
            ("pack_cid", context_pack.pack_cid),
            ("route_cid", route.decision_cid),
            ("model", route.model),
            ("revision", route.revision or "unspecified"),
        ):
            if output[name] != expected:
                raise IdentityInconsistentError(
                    f"command response {name} drifted", details={"field": name}
                )
        declared = admit_path_list(
            output["declared_files"], field="declared_files", min_items=1, max_items=1024
        )
        assert_declared_scope(declared, task.owned_paths, task.declared_files)
        patch = admit_bounded_patch(output["patch"])
        _admit_patch_scope(patch, declared, task)
        # Validate every reported counter even though elapsed wall time is authoritative.
        admit_non_negative_int(output["latency_ms"], field="latency_ms")
        invocation_body = {
            "schema": CODING_AGENT_INVOCATION_SCHEMA,
            "task_id": task.task_id,
            "repository_state_cid": task.repository_state_cid,
            "route_cid": route.decision_cid,
            "provider": route.provider,
            "model": route.model,
            "revision": route.revision or "unspecified",
            "tier": route.tier,
            "token_count": admit_non_negative_int(output["token_count"], field="token_count"),
            "cached_token_count": admit_non_negative_int(
                output["cached_token_count"], field="cached_token_count"
            ),
            "latency_ms": execution.latency_ms,
            "cost_micros": admit_non_negative_int(output["cost_micros"], field="cost_micros"),
            "response_artifact_cid": _cid(
                {"request": dict(request), "stdout": execution.stdout.decode("utf-8")}
            ),
            "provenance": "live",
        }
        invocation_body["invocation_cid"] = _cid(invocation_body)
        proposal_body = {
            "schema": PATCH_PROPOSAL_SCHEMA,
            "task_id": task.task_id,
            "repository_state_cid": task.repository_state_cid,
            "declared_files": list(declared),
            "invocation_cid": invocation_body["invocation_cid"],
            "patch_cid": _cid(patch.decode("utf-8", "replace")),
            "provenance": "live",
        }
        proposal_body["proposal_cid"] = _cid(proposal_body)
        result = AdapterResult(
            PatchProposal.from_mapping(proposal_body),
            CodingAgentInvocation.from_mapping(invocation_body),
            patch_bytes=patch,
            log_bytes=execution.log_bytes,
        )
        return admit_adapter_result(task, context_pack, route, result, cancellation=cancellation)


__all__ = [
    "ADAPTER",
    "COMMAND_CONTRACT",
    "CommandAdapter",
    "CommandExecution",
    "CommandPolicy",
    "build_command_request",
    "decode_structured_output",
    "invoke_command",
    "redact_command_log",
]
