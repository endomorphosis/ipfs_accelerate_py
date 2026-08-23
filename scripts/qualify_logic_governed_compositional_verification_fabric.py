#!/usr/bin/env python3
"""Produce or independently reconstruct the hermetic LGCVF qualification.

The two task-authored qualification suites are candidate evidence.  This
protected judge also runs a fixed manifest of already accepted semantic and
supervisor tests, rejects skips and expected failures, binds every test source,
and never grants release or production authority.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import ctypes.util
import errno
import hashlib
import io
import json
import os
import platform
import re
import resource
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any, Final

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (  # noqa: E402
    content_identity,
)

SCHEMA: Final[str] = "lgcvf-independent-hermetic-qualification@1"
WORKER_SCHEMA: Final[str] = "lgcvf-independent-pytest-observation@1"
PLAN_CID: Final[str] = "baguqeerabxn5kkewz44v4chz6vbt3kcozfj4rvhh4gpdp54645blhemhvloq"
PREDECESSOR_PLAN_CID: Final[str] = (
    "baguqeeraqe65yknsg7gy5vkze76exc3qhe4kn2owecnwa65zg6kaepl7id3q"
)
OUTPUT: Final[Path] = (
    ROOT
    / "data/agent_supervisor/logic_governed_compositional_verification_fabric"
    / "independent_qualification_result.json"
)
_MAX_GIT_OBSERVATION_BYTES: Final[int] = 64 * 1024 * 1024
_MAX_UNTRACKED_PROTECTED_FILES: Final[int] = 4_096
_MAX_UNTRACKED_PROTECTED_BYTES: Final[int] = 64 * 1024 * 1024
DECLARED_GENERATED_EVIDENCE_PATHS: Final[tuple[str, ...]] = (
    "data/agent_supervisor/logic_governed_compositional_verification_fabric/benchmark_result.json",
    "data/agent_supervisor/logic_governed_compositional_verification_fabric/independent_qualification_result.json",
    "data/agent_supervisor/logic_governed_compositional_verification_fabric/successor_tasks.json",
    "docs/architecture/LOGIC_GOVERNED_COMPOSITIONAL_VERIFICATION_FABRIC_IMPLEMENTATION_REPORT.md",
    "docs/architecture/LOGIC_GOVERNED_COMPOSITIONAL_VERIFICATION_FABRIC_RELEASE.md",
)
PROTECTED_AUTHORITY_PATHS: Final[tuple[str, ...]] = (
    "config/agent_supervisor_logic_governed_compositional_verification_fabric_scheduler.json",
    "data/agent_supervisor/logic_governed_compositional_verification_fabric/formal_work_plan.json",
    "docs/architecture/logic_governed_compositional_verification_fabric.todo.md",
    "scripts/benchmark_lgcvf_symbolic_displacement.py",
    "scripts/emit_logic_governed_compositional_verification_fabric_plan.py",
    "scripts/materialize_logic_governed_compositional_verification_fabric_control_plane.py",
    "scripts/qualify_logic_governed_compositional_verification_fabric.py",
    "scripts/validate_logic_governed_compositional_verification_fabric_closeout.py",
    "scripts/validate_logic_governed_compositional_verification_fabric_plan.py",
)
_LANDLOCK_CREATE_RULESET_VERSION: Final[int] = 1 << 0
_LANDLOCK_RULE_PATH_BENEATH: Final[int] = 1
_LANDLOCK_ACCESS_FS_WRITE_FILE: Final[int] = 1 << 1
_LANDLOCK_ACCESS_FS_REMOVE_DIR: Final[int] = 1 << 4
_LANDLOCK_ACCESS_FS_REMOVE_FILE: Final[int] = 1 << 5
_LANDLOCK_ACCESS_FS_MAKE_CHAR: Final[int] = 1 << 6
_LANDLOCK_ACCESS_FS_MAKE_DIR: Final[int] = 1 << 7
_LANDLOCK_ACCESS_FS_MAKE_REG: Final[int] = 1 << 8
_LANDLOCK_ACCESS_FS_MAKE_SOCK: Final[int] = 1 << 9
_LANDLOCK_ACCESS_FS_MAKE_FIFO: Final[int] = 1 << 10
_LANDLOCK_ACCESS_FS_MAKE_BLOCK: Final[int] = 1 << 11
_LANDLOCK_ACCESS_FS_MAKE_SYM: Final[int] = 1 << 12
_LANDLOCK_ACCESS_FS_REFER: Final[int] = 1 << 13
_LANDLOCK_ACCESS_FS_TRUNCATE: Final[int] = 1 << 14
_LANDLOCK_ACCESS_NET_BIND_TCP: Final[int] = 1 << 0
_LANDLOCK_ACCESS_NET_CONNECT_TCP: Final[int] = 1 << 1
_PR_SET_NO_NEW_PRIVS: Final[int] = 38
_SCMP_ACT_ALLOW: Final[int] = 0x7FFF0000
_SCMP_ACT_ERRNO: Final[int] = 0x00050000
_DENIED_SYSCALLS: Final[tuple[str, ...]] = (
    # No socket family is needed by these hermetic candidate suites.  Blocking
    # creation as well as use also covers UDP and Unix-domain escape channels,
    # which Landlock's TCP-port rules do not cover.
    "socket",
    "socketpair",
    "connect",
    "accept",
    "accept4",
    "bind",
    "listen",
    "sendto",
    "sendmsg",
    "sendmmsg",
    "recvfrom",
    "recvmsg",
    "recvmmsg",
    "shutdown",
    # Ownership, timestamps, and xattrs stay denied.  chmod is allowed
    # because hermetic git init/config under the writable root needs it;
    # Landlock still blocks creating or rewriting files outside that root.
    "chown",
    "fchown",
    "lchown",
    "fchownat",
    "utime",
    "utimes",
    "futimesat",
    "utimensat",
    "setxattr",
    "lsetxattr",
    "fsetxattr",
    "removexattr",
    "lremovexattr",
    "fremovexattr",
    # Prevent same-UID process tampering and mount-namespace escape attempts.
    "ptrace",
    "process_vm_writev",
    "kill",
    "tkill",
    "tgkill",
    "pidfd_send_signal",
    "mount",
    "umount2",
    "pivot_root",
    "move_mount",
    "fsopen",
    "fsconfig",
    "fsmount",
    "open_tree",
    "setsid",
    "setpgid",
    "unshare",
    "setns",
)


@dataclass(frozen=True)
class Suite:
    suite_id: str
    owner_root: str
    paths: tuple[str, ...]
    candidate_authored: bool


SUITES: Final[tuple[Suite, ...]] = (
    Suite(
        "candidate_focused",
        ".",
        ("test/api/test_agent_supervisor_lgcvf_focused_qualification.py",),
        True,
    ),
    Suite(
        "candidate_adversarial",
        ".",
        ("test/api/test_agent_supervisor_lgcvf_adversarial.py",),
        True,
    ),
    Suite(
        "fixed_datasets_semantics",
        "ipfs_datasets_py",
        (
            "tests/unit/logic/software_contracts/test_compositional_contract.py",
            "tests/unit/logic/software_verification/test_abstract_interpretation.py",
            "tests/unit/logic/software_verification/test_assume_guarantee.py",
            "tests/unit/logic/software_verification/test_incremental_verification.py",
            "tests/unit/logic/backends/test_interpolation.py",
            "tests/unit/logic/software_verification/test_cegar.py",
            "tests/unit/logic/formalization/test_translation_receipts.py",
            "tests/unit/logic/software_verification/test_obligation_slicing.py",
            "tests/unit/logic/test_compositional_verification_public_api.py",
        ),
        False,
    ),
    Suite(
        "fixed_accelerator_authority",
        ".",
        (
            "test/api/test_agent_supervisor_compositional_verification_vertical.py",
            "test/api/test_agent_supervisor_database_implementation_daemon.py",
            "test/api/test_agent_supervisor_database_portal_bridge.py",
            "test/api/test_agent_supervisor_manual_completion_authority_runtime.py",
            "test/api/test_agent_supervisor_expected_output_submodule_preflight.py",
            "test/api/test_agent_supervisor_deterministic_doctor_live_fixed_point.py",
            "test/api/semantic_state/test_acceptance.py",
            "test/api/semantic_state/test_capsules.py",
        ),
        False,
    ),
)
SANDBOX_SMOKE_SUITE: Final[Suite] = Suite(
    "fixed_sandbox_smoke",
    ".",
    ("test/fixtures/lgcvf_sandbox_smoke.py",),
    False,
)
_WORKER_SUITES: Final[tuple[Suite, ...]] = (*SUITES, SANDBOX_SMOKE_SUITE)


class QualificationError(RuntimeError):
    """Qualification input, execution, or reconstruction failed closed."""


class _LandlockRulesetAttr(ctypes.Structure):
    _fields_ = (
        ("handled_access_fs", ctypes.c_uint64),
        ("handled_access_net", ctypes.c_uint64),
    )


class _LandlockPathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = (
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
    )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _syscall_number(name: str) -> int:
    """Return the asm-generic Landlock syscall number on supported Linux."""

    if not sys.platform.startswith("linux") or os.uname().machine not in {
        "aarch64",
        "x86_64",
    }:
        raise QualificationError("candidate sandbox requires supported Linux Landlock")
    numbers = {
        "landlock_create_ruleset": 444,
        "landlock_add_rule": 445,
        "landlock_restrict_self": 446,
    }
    return numbers[name]


def _raise_errno(operation: str) -> None:
    code = ctypes.get_errno()
    raise QualificationError(f"candidate sandbox {operation} failed: {os.strerror(code)}")


def _install_landlock(write_root: Path) -> int:
    """Make every filesystem hierarchy except ``write_root`` immutable.

    Read and execute access intentionally remain outside the handled set so
    the copied test projection can import the checked repository and installed
    dependencies.  Mutation rights are granted only beneath a fresh temporary
    directory.  Landlock restrictions are inherited and cannot be relaxed by
    candidate code or its descendants.
    """

    try:
        allowed = write_root.resolve(strict=True)
    except OSError as exc:
        raise QualificationError("candidate sandbox write root is unavailable") from exc
    if not allowed.is_dir():
        raise QualificationError("candidate sandbox write root is not a directory")

    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    abi = int(
        libc.syscall(
            _syscall_number("landlock_create_ruleset"),
            ctypes.c_void_p(),
            ctypes.c_size_t(0),
            ctypes.c_uint(_LANDLOCK_CREATE_RULESET_VERSION),
        )
    )
    if abi < 4:
        raise QualificationError(
            "candidate sandbox requires Landlock ABI 4 network mediation"
        )
    handled_fs = (
        _LANDLOCK_ACCESS_FS_WRITE_FILE
        | _LANDLOCK_ACCESS_FS_REMOVE_DIR
        | _LANDLOCK_ACCESS_FS_REMOVE_FILE
        | _LANDLOCK_ACCESS_FS_MAKE_CHAR
        | _LANDLOCK_ACCESS_FS_MAKE_DIR
        | _LANDLOCK_ACCESS_FS_MAKE_REG
        | _LANDLOCK_ACCESS_FS_MAKE_SOCK
        | _LANDLOCK_ACCESS_FS_MAKE_FIFO
        | _LANDLOCK_ACCESS_FS_MAKE_BLOCK
        | _LANDLOCK_ACCESS_FS_MAKE_SYM
        | _LANDLOCK_ACCESS_FS_REFER
        | _LANDLOCK_ACCESS_FS_TRUNCATE
    )
    handled_net = (
        _LANDLOCK_ACCESS_NET_BIND_TCP | _LANDLOCK_ACCESS_NET_CONNECT_TCP
    )
    ruleset_attr = _LandlockRulesetAttr(handled_fs, handled_net)
    ruleset_fd = int(
        libc.syscall(
            _syscall_number("landlock_create_ruleset"),
            ctypes.byref(ruleset_attr),
            ctypes.sizeof(ruleset_attr),
            ctypes.c_uint(0),
        )
    )
    if ruleset_fd < 0:
        _raise_errno("ruleset creation")
    parent_fd = -1
    extra_fds: list[int] = []
    try:
        parent_fd = os.open(allowed, os.O_PATH | os.O_CLOEXEC)
        path_attr = _LandlockPathBeneathAttr(handled_fs, parent_fd)
        if (
            int(
                libc.syscall(
                    _syscall_number("landlock_add_rule"),
                    ruleset_fd,
                    _LANDLOCK_RULE_PATH_BENEATH,
                    ctypes.byref(path_attr),
                    ctypes.c_uint(0),
                )
            )
            != 0
        ):
            _raise_errno("path rule installation")
        # Git and other hermetic tools open /dev/null read/write.  Grant
        # write only on that node so sealed workers can still run local
        # git without making the rest of /dev or the checkout mutable.
        null_access = _LANDLOCK_ACCESS_FS_WRITE_FILE | _LANDLOCK_ACCESS_FS_TRUNCATE
        null_fd = os.open("/dev/null", os.O_PATH | os.O_CLOEXEC)
        extra_fds.append(null_fd)
        null_attr = _LandlockPathBeneathAttr(null_access, null_fd)
        if (
            int(
                libc.syscall(
                    _syscall_number("landlock_add_rule"),
                    ruleset_fd,
                    _LANDLOCK_RULE_PATH_BENEATH,
                    ctypes.byref(null_attr),
                    ctypes.c_uint(0),
                )
            )
            != 0
        ):
            _raise_errno("devnull rule installation")
        if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            _raise_errno("no-new-privileges installation")
        if (
            int(
                libc.syscall(
                    _syscall_number("landlock_restrict_self"),
                    ruleset_fd,
                    ctypes.c_uint(0),
                )
            )
            != 0
        ):
            _raise_errno("restriction")
    finally:
        if parent_fd >= 0:
            os.close(parent_fd)
        for extra_fd in extra_fds:
            os.close(extra_fd)
        os.close(ruleset_fd)
    return abi


def _install_seccomp() -> int:
    """Deny network and unmediated checkout-mutation syscalls."""

    library_name = ctypes.util.find_library("seccomp")
    if not library_name:
        raise QualificationError("candidate sandbox requires libseccomp")
    library = ctypes.CDLL(library_name, use_errno=True)
    library.seccomp_init.argtypes = (ctypes.c_uint32,)
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes = (ctypes.c_char_p,)
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int
    library.seccomp_rule_add.argtypes = (
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    )
    library.seccomp_rule_add.restype = ctypes.c_int
    library.seccomp_load.argtypes = (ctypes.c_void_p,)
    library.seccomp_load.restype = ctypes.c_int
    library.seccomp_release.argtypes = (ctypes.c_void_p,)
    context = library.seccomp_init(_SCMP_ACT_ALLOW)
    if not context:
        raise QualificationError("candidate sandbox seccomp initialization failed")
    installed = 0
    try:
        action = _SCMP_ACT_ERRNO | errno.EPERM
        for name in _DENIED_SYSCALLS:
            syscall_number = int(
                library.seccomp_syscall_resolve_name(name.encode("ascii"))
            )
            if syscall_number < 0:
                continue
            if library.seccomp_rule_add(context, action, syscall_number, 0) != 0:
                raise QualificationError(
                    f"candidate sandbox could not deny syscall {name}"
                )
            installed += 1
        if installed < 2 or library.seccomp_load(context) != 0:
            raise QualificationError("candidate sandbox seccomp load failed")
    finally:
        library.seccomp_release(context)
    return installed


def _lower_resource_limit(kind: int, value: int) -> int:
    """Irreversibly lower one process resource bound and return the bound."""

    _soft, hard = resource.getrlimit(kind)
    bounded = value if hard == resource.RLIM_INFINITY else min(value, int(hard))
    resource.setrlimit(kind, (bounded, bounded))
    return bounded


def _install_candidate_sandbox(write_root: Path) -> dict[str, Any]:
    """Install irreversible filesystem and network restrictions in a worker."""

    # Resolve and load libseccomp before lowering the process limit because
    # libc discovery may itself use one bounded helper process.
    denied_syscalls = _install_seccomp()
    # Bound damage from a malicious or accidentally explosive candidate test.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    file_size_bytes = _lower_resource_limit(resource.RLIMIT_FSIZE, 64 * 1024 * 1024)
    open_files = _lower_resource_limit(resource.RLIMIT_NOFILE, 256)
    # RLIMIT_NPROC is per real UID (and counts threads), not per worker.
    # Numerical libraries stay pinned to one thread below.  Do not clamp
    # this UID to a 2048-thread development ceiling: this host already
    # runs thousands of same-UID threads, and hyperscale qualification
    # must still be able to import and fork inside the sealed worker.
    processes = _lower_resource_limit(resource.RLIMIT_NPROC, 1_048_576)
    cpu_seconds = _lower_resource_limit(resource.RLIMIT_CPU, 900)
    address_space_bytes = _lower_resource_limit(resource.RLIMIT_AS, 8 * 1024**3)
    landlock_abi = _install_landlock(write_root)
    return {
        "profile": "landlock-readonly-seccomp-no-network",
        "landlock_abi": landlock_abi,
        "seccomp_denied_syscall_count": denied_syscalls,
        "checkout_write_permitted": False,
        "network_permitted": False,
        "completion_authoritative": False,
        "process_group_escape_permitted": False,
        "resource_limits": {
            "address_space_bytes": address_space_bytes,
            "cpu_seconds": cpu_seconds,
            "file_size_bytes": file_size_bytes,
            "open_files": open_files,
            "processes": processes,
        },
    }


def _sandbox_evidence_is_valid(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "profile",
        "landlock_abi",
        "seccomp_denied_syscall_count",
        "checkout_write_permitted",
        "network_permitted",
        "completion_authoritative",
        "process_group_escape_permitted",
        "null_sink_redirected",
        "pytest_log_sink_redirected",
        "resource_limits",
    }:
        return False
    limits = value.get("resource_limits")
    if not isinstance(limits, Mapping) or set(limits) != {
        "address_space_bytes",
        "cpu_seconds",
        "file_size_bytes",
        "open_files",
        "processes",
    }:
        return False
    maximums = {
        "address_space_bytes": 8 * 1024**3,
        "cpu_seconds": 900,
        "file_size_bytes": 64 * 1024 * 1024,
        "open_files": 256,
        "processes": 1_048_576,
    }
    if any(
        isinstance(limits.get(name), bool)
        or not isinstance(limits.get(name), int)
        or not 0 < int(limits[name]) <= maximum
        for name, maximum in maximums.items()
    ):
        return False
    return (
        value.get("profile") == "landlock-readonly-seccomp-no-network"
        and isinstance(value.get("landlock_abi"), int)
        and not isinstance(value.get("landlock_abi"), bool)
        and int(value["landlock_abi"]) >= 4
        and isinstance(value.get("seccomp_denied_syscall_count"), int)
        and not isinstance(value.get("seccomp_denied_syscall_count"), bool)
        and int(value["seccomp_denied_syscall_count"]) >= 2
        and value.get("checkout_write_permitted") is False
        and value.get("network_permitted") is False
        and value.get("completion_authoritative") is False
        and value.get("process_group_escape_permitted") is False
        and value.get("null_sink_redirected") is True
        and value.get("pytest_log_sink_redirected") is True
    )


def _safe_source(owner: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not relative:
        raise QualificationError(f"unsafe qualification source: {relative!r}")
    try:
        resolved = (owner / path).resolve(strict=True)
    except OSError as exc:
        raise QualificationError(
            f"qualification source is not a file: {relative}"
        ) from exc
    try:
        resolved.relative_to(owner.resolve(strict=True))
    except ValueError as exc:
        raise QualificationError(f"qualification source escapes owner: {relative!r}") from exc
    if not resolved.is_file():
        raise QualificationError(f"qualification source is not a file: {relative}")
    return resolved


def _suite_manifest(suite: Suite, *, root: Path = ROOT) -> dict[str, Any]:
    owner = (root / suite.owner_root).resolve(strict=True)
    sources = []
    for relative in suite.paths:
        path = _safe_source(owner, relative)
        data = path.read_bytes()
        sources.append(
            {
                "path": relative,
                "sha256": _sha256_bytes(data),
                "size_bytes": len(data),
            }
        )
    manifest = {
        "candidate_authored": suite.candidate_authored,
        "owner_root": suite.owner_root,
        "paths": list(suite.paths),
        "sources": sources,
        "suite_id": suite.suite_id,
    }
    manifest["manifest_cid"] = content_identity(manifest)
    return manifest


def _copy_projection_branch(
    source: Path,
    destination: Path,
    *,
    prefix: PurePath,
    copied_paths: frozenset[str],
) -> None:
    """Build a symlink projection while copying the exact judged inputs."""

    destination.mkdir(mode=0o755, parents=True, exist_ok=True)
    for entry in sorted(source.iterdir(), key=lambda item: item.name):
        relative = (prefix / entry.name).as_posix()
        descendants = [
            item for item in copied_paths if item == relative or item.startswith(relative + "/")
        ]
        target = destination / entry.name
        if not descendants:
            target.symlink_to(entry.resolve(), target_is_directory=entry.is_dir())
        elif entry.is_dir():
            _copy_projection_branch(
                entry,
                target,
                prefix=prefix / entry.name,
                copied_paths=copied_paths,
            )
        elif relative in copied_paths and entry.is_file():
            shutil.copyfile(entry, target)
            target.chmod(0o444)
        else:
            raise QualificationError(
                f"qualification projection cannot copy source: {relative}"
            )


def _prepare_execution_checkout(
    root: Path,
    destination: Path,
    suites: Sequence[Suite],
) -> dict[str, Any]:
    """Create a read-only projection with exact copied suite inputs."""

    resolved_root = root.resolve(strict=True)
    copied_paths = frozenset(
        (Path(suite.owner_root) / relative).as_posix().lstrip("./")
        for suite in suites
        for relative in suite.paths
    )
    if not copied_paths:
        raise QualificationError("qualification projection has no judged inputs")
    for relative in copied_paths:
        _safe_source(resolved_root, relative)
    _copy_projection_branch(
        resolved_root,
        destination,
        prefix=Path(),
        copied_paths=copied_paths,
    )
    copied = []
    for relative in sorted(copied_paths):
        source = _safe_source(resolved_root, relative)
        projected_path = destination / relative
        projected = _safe_source(destination.resolve(strict=True), relative)
        source_bytes = source.read_bytes()
        projected_bytes = projected.read_bytes()
        if projected_path.is_symlink() or source_bytes != projected_bytes:
            raise QualificationError(
                f"qualification copied input differs: {relative}"
            )
        copied.append(
            {
                "path": relative,
                "sha256": _sha256_bytes(source_bytes),
                "size_bytes": len(source_bytes),
            }
        )
    result = {
        "schema": "lgcvf-readonly-test-projection@1",
        "copied_sources": copied,
        "original_checkout_writable": False,
    }
    result["projection_cid"] = content_identity(result)
    return result


def _git_bytes(root: Path, args: Sequence[str]) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        raise QualificationError(
            f"protected-input projection git {' '.join(args)} failed: "
            + completed.stderr.decode("utf-8", errors="replace")[-500:]
        )
    if len(completed.stdout) > _MAX_GIT_OBSERVATION_BYTES:
        raise QualificationError("protected-input Git observation exceeds its bound")
    return completed.stdout


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise QualificationError(f"protected input is unreadable: {path}") from exc
    return "sha256:" + digest.hexdigest(), size


def _protected_path_identity(root: Path, relative: str) -> dict[str, Any]:
    logical = PurePath(relative)
    if logical.is_absolute() or not relative or ".." in logical.parts:
        raise QualificationError(f"unsafe protected-input path: {relative!r}")
    path = root.joinpath(*logical.parts)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise QualificationError(f"protected input is unavailable: {relative}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        target = os.readlink(path).encode("utf-8", errors="surrogateescape")
        return {
            "kind": "symlink",
            "mode": stat.S_IMODE(metadata.st_mode),
            "path": logical.as_posix(),
            "sha256": _sha256_bytes(target),
            "size_bytes": len(target),
        }
    if not stat.S_ISREG(metadata.st_mode):
        raise QualificationError(f"protected input is not a regular file: {relative}")
    digest, size = _sha256_file(path)
    return {
        "kind": "file",
        "mode": stat.S_IMODE(metadata.st_mode),
        "path": logical.as_posix(),
        "sha256": digest,
        "size_bytes": size,
    }


def _repository_protected_projection(
    repository: Path,
    *,
    label: str,
    excluded_paths: Sequence[str],
) -> dict[str, Any]:
    pathspec = (".", *(f":(exclude){path}" for path in excluded_paths))
    tracked_index = _git_bytes(
        repository,
        ("ls-files", "--stage", "-z", "--", *pathspec),
    )
    status = _git_bytes(
        repository,
        (
            "status",
            "--porcelain=v2",
            "--untracked-files=all",
            "-z",
            "--",
            *pathspec,
        ),
    )
    diff = _git_bytes(
        repository,
        ("diff", "--no-ext-diff", "--binary", "HEAD", "--", *pathspec),
    )
    untracked_raw = _git_bytes(
        repository,
        ("ls-files", "--others", "--exclude-standard", "-z", "--", *pathspec),
    )
    try:
        untracked_paths = tuple(
            sorted(
                item.decode("utf-8")
                for item in untracked_raw.split(b"\0")
                if item
            )
        )
    except UnicodeDecodeError as exc:
        raise QualificationError("protected untracked path is not UTF-8") from exc
    if len(untracked_paths) > _MAX_UNTRACKED_PROTECTED_FILES:
        raise QualificationError("protected untracked population exceeds its bound")
    untracked = [
        _protected_path_identity(repository, relative)
        for relative in untracked_paths
    ]
    if sum(int(item["size_bytes"]) for item in untracked) > _MAX_UNTRACKED_PROTECTED_BYTES:
        raise QualificationError("protected untracked bytes exceed their bound")
    return {
        "repository": label,
        "tracked_index_sha256": _sha256_bytes(tracked_index),
        "normalized_status_sha256": _sha256_bytes(status),
        "working_diff_sha256": _sha256_bytes(diff),
        "untracked_inputs": untracked,
    }


def _protected_input_projection(
    manifests: Sequence[Mapping[str, Any]],
    *,
    root: Path = ROOT,
    excluded_paths: Sequence[str] = DECLARED_GENERATED_EVIDENCE_PATHS,
    authority_paths: Sequence[str] = PROTECTED_AUTHORITY_PATHS,
) -> dict[str, Any]:
    """Bind semantic inputs while excluding only declared generated evidence.

    Git commit and tree identities are intentionally absent: committing the
    qualification artifact or a downstream benchmark/report must not
    invalidate unchanged evidence.  The protected index, normalized overlay,
    bounded untracked bytes, exact judges/control files, suites, and toolchain
    remain bound, so committing or editing any semantic input changes this
    projection.
    """

    resolved_root = root.resolve(strict=True)
    normalized_exclusions = tuple(sorted(set(excluded_paths)))
    if len(normalized_exclusions) != len(excluded_paths):
        raise QualificationError("generated evidence exclusions must be unique")
    for relative in normalized_exclusions:
        logical = PurePath(relative)
        if logical.is_absolute() or not relative or ".." in logical.parts:
            raise QualificationError(f"unsafe generated evidence exclusion: {relative!r}")

    repositories = [resolved_root]
    nested = resolved_root / "ipfs_datasets_py"
    if (nested / ".git").exists():
        repositories.append(nested)
    repository_projections = []
    for repository in repositories:
        label = repository.relative_to(resolved_root).as_posix() or "."
        exclusions = normalized_exclusions if repository == resolved_root else ()
        repository_projections.append(
            _repository_protected_projection(
                repository,
                label=label,
                excluded_paths=exclusions,
            )
        )
    authority = [
        _protected_path_identity(resolved_root, relative)
        for relative in sorted(set(authority_paths))
    ]
    try:
        import pytest
    except ImportError as exc:
        raise QualificationError("pytest toolchain is unavailable") from exc
    executable_sha256, executable_size = _sha256_file(Path(sys.executable))
    git_version = _git_bytes(
        resolved_root,
        ("--version",),
    ).decode("utf-8", errors="strict").strip()
    body = {
        "schema": "lgcvf-qualification-protected-input-projection@1",
        "declared_generated_evidence_exclusions": list(normalized_exclusions),
        "repositories": repository_projections,
        "authority_sources": authority,
        "suite_manifests": [dict(item) for item in manifests],
        "toolchain": {
            "git": git_version,
            "machine": platform.machine(),
            "python_executable_sha256": executable_sha256,
            "python_executable_size_bytes": executable_size,
            "python_implementation": sys.implementation.name,
            "python_version": platform.python_version(),
            "pytest_version": str(pytest.__version__),
        },
    }
    body["fingerprint_cid"] = content_identity(body)
    return body


def _checkout_fingerprint(
    manifests: Sequence[Mapping[str, Any]], *, root: Path = ROOT
) -> dict[str, Any]:
    """Backward-compatible name for the protected semantic input projection."""

    return _protected_input_projection(manifests, root=root)


@dataclass
class _Recorder:
    collected: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    xfailed: int = 0
    xpassed: int = 0
    errors: int = 0
    nodeids: list[str] = field(default_factory=list)
    _terminal: set[str] = field(default_factory=set)

    def pytest_collection_finish(self, session: Any) -> None:
        self.collected = len(session.items)
        self.nodeids = sorted(str(item.nodeid) for item in session.items)

    def pytest_collectreport(self, report: Any) -> None:
        if report.failed:
            self.errors += 1

    def pytest_runtest_logreport(self, report: Any) -> None:
        nodeid = str(report.nodeid)
        if report.when == "call":
            if getattr(report, "wasxfail", None):
                if report.skipped:
                    self.xfailed += 1
                elif report.passed:
                    self.xpassed += 1
                else:
                    self.failed += 1
            elif report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1
            elif report.skipped:
                self.skipped += 1
            self._terminal.add(nodeid)
        elif report.when == "setup" and report.skipped:
            self.skipped += 1
            self._terminal.add(nodeid)
        elif report.when in {"setup", "teardown"} and report.failed:
            self.errors += 1
            self._terminal.add(nodeid)


def _emit_worker_receipt(descriptor: int, value: Mapping[str, Any]) -> None:
    """Emit on a pre-pytest pipe descriptor immune to pytest FD capture."""

    payload = json.dumps(value, sort_keys=True).encode("utf-8") + b"\n"
    view = memoryview(payload)
    try:
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise QualificationError("worker receipt pipe made no progress")
            view = view[written:]
    finally:
        os.close(descriptor)


def _worker(
    suite_id: str,
    *,
    execution_root: Path | None = None,
    write_root: Path | None = None,
    receipt_descriptor: int | None = None,
) -> int:
    receipt_descriptor = (
        os.dup(1) if receipt_descriptor is None else int(receipt_descriptor)
    )
    os.set_inheritable(receipt_descriptor, False)
    try:
        suite = next(item for item in _WORKER_SUITES if item.suite_id == suite_id)
    except StopIteration:
        _emit_worker_receipt(
            receipt_descriptor,
            {"schema": WORKER_SCHEMA, "error": "unknown_suite"},
        )
        return 2
    try:
        worker_root = ROOT if execution_root is None else execution_root.resolve(strict=True)
        if execution_root is not None and worker_root == ROOT.resolve():
            raise QualificationError("worker execution root is not isolated")
        if execution_root is not None and write_root is None:
            raise QualificationError("isolated worker has no write root")
        writable_root = write_root.resolve(strict=True) if write_root is not None else None
        manifest = _suite_manifest(suite, root=worker_root)
        owner = (worker_root / suite.owner_root).resolve(strict=True)
        if writable_root is not None:
            # Some import-time library probes open ``os.devnull`` read/write.
            # Redirect that process-local name into the sealed writable root;
            # granting write access to the host /dev hierarchy would broaden
            # the Landlock policy.  The explicit pytest override below handles
            # the repository's independently configured logging sink.
            null_sink = writable_root / "devnull"
            null_sink.touch(mode=0o600, exist_ok=False)
            os.devnull = str(null_sink)
        isolation = (
            {
                **_install_candidate_sandbox(writable_root),
                "null_sink_redirected": True,
                "pytest_log_sink_redirected": True,
            }
            if writable_root is not None
            else {
                "profile": "unisolated-rejected",
                "checkout_write_permitted": True,
                "network_permitted": True,
                "completion_authoritative": False,
            }
        )
        if (
            isolation.get("checkout_write_permitted") is not False
            or isolation.get("network_permitted") is not False
        ):
            raise QualificationError("worker sandbox is not fail-closed")
        import pytest

        recorder = _Recorder()
        captured_out = io.StringIO()
        captured_err = io.StringIO()
        started = time.monotonic_ns()
        previous = Path.cwd()
        inserted_owner = str(owner) not in sys.path
        try:
            os.chdir(owner)
            if inserted_owner:
                sys.path.insert(0, str(owner))
            with tempfile.TemporaryDirectory(prefix="lgcvf-pytest-cache-") as cache_dir:
                if writable_root is None:
                    raise QualificationError("isolated worker has no writable root")
                pytest_log = writable_root / "pytest.log"
                with contextlib.redirect_stdout(captured_out), contextlib.redirect_stderr(
                    captured_err
                ):
                    exit_code = int(
                        pytest.main(
                            [
                                "-q",
                                "-ra",
                                "--maxfail=1",
                                "-o",
                                f"cache_dir={cache_dir}",
                                "-o",
                                f"log_file={pytest_log}",
                                *suite.paths,
                            ],
                            plugins=[recorder],
                        )
                    )
        finally:
            if inserted_owner:
                try:
                    sys.path.remove(str(owner))
                except ValueError:
                    pass
            os.chdir(previous)
        duration_ms = max(0, (time.monotonic_ns() - started) // 1_000_000)
        transcript = (captured_out.getvalue() + "\n" + captured_err.getvalue()).encode(
            "utf-8", errors="replace"
        )
        passed = (
            exit_code == 0
            and recorder.collected > 0
            and recorder.passed == recorder.collected
            and not any(
                (
                    recorder.failed,
                    recorder.skipped,
                    recorder.xfailed,
                    recorder.xpassed,
                    recorder.errors,
                )
            )
        )
        payload = {
            "schema": WORKER_SCHEMA,
            "suite_id": suite.suite_id,
            "manifest": manifest,
            "collected": recorder.collected,
            "passed_count": recorder.passed,
            "failed_count": recorder.failed,
            "skipped_count": recorder.skipped,
            "xfailed_count": recorder.xfailed,
            "xpassed_count": recorder.xpassed,
            "error_count": recorder.errors,
            "nodeids_cid": content_identity(recorder.nodeids),
            "exit_code": exit_code,
            "passed": passed,
            "isolation": isolation,
            "duration_ms": duration_ms,
            "transcript_sha256": _sha256_bytes(transcript),
            "failure_tail": transcript.decode("utf-8", errors="replace")[-4000:]
            if not passed
            else "",
        }
        payload["observation_cid"] = content_identity(payload)
        _emit_worker_receipt(receipt_descriptor, payload)
        return 0 if passed else 1
    # A candidate suite can raise arbitrary Python exceptions during import or
    # pytest teardown.  They are all converted to an explicit *failure*
    # receipt so the parent can diagnose the rejection without ever treating
    # an unknown exception as qualification evidence.  KeyboardInterrupt is
    # included because this worker is an isolated child; the parent remains
    # responsible for operator cancellation and process-group termination.
    except (Exception, SystemExit, KeyboardInterrupt) as exc:
        _emit_worker_receipt(
            receipt_descriptor,
            {
                "schema": WORKER_SCHEMA,
                "suite_id": suite_id,
                "error": type(exc).__name__,
                "reason": str(exc)[:1000],
            },
        )
        return 2


def _run_suite(
    suite: Suite,
    *,
    expected_manifest: Mapping[str, Any],
    root: Path = ROOT,
) -> dict[str, Any]:
    dependency_paths = tuple(
        dict.fromkeys(
            str(Path(entry).resolve())
            for entry in sys.path
            if entry
            and Path(entry).is_dir()
            and Path(entry).resolve() != ROOT.resolve()
        )
    )
    if not dependency_paths:
        raise QualificationError("qualified Python dependency path is unavailable")
    with tempfile.TemporaryDirectory(prefix="lgcvf-qualification-sandbox-") as sandbox:
        sandbox_path = Path(sandbox)
        checkout = sandbox_path / "checkout"
        writable = sandbox_path / "writable"
        writable.mkdir(mode=0o700)
        _prepare_execution_checkout(root, checkout, (suite,))
        home_path = writable / "home"
        home_path.mkdir(mode=0o700)
        xdg_paths = {
            "XDG_CACHE_HOME": home_path / ".cache",
            "XDG_CONFIG_HOME": home_path / ".config",
            "XDG_DATA_HOME": home_path / ".local/share",
            "XDG_STATE_HOME": home_path / ".local/state",
        }
        for path in xdg_paths.values():
            path.mkdir(parents=True, exist_ok=True)
        environment = {
            "HOME": str(home_path),
            "LANG": "C.UTF-8",
            "NO_COLOR": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(dependency_paths),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONHASHSEED": "0",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TMPDIR": str(writable),
            # Seccomp still denies utime; GNU tar otherwise fails closed
            # while extracting git-archive fixtures onto the write root.
            "TAR_OPTIONS": "--touch",
            **{key: str(path) for key, path in xdg_paths.items()},
        }
        receipt_read, receipt_write = os.pipe()
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    suite.suite_id,
                    "--worker-root",
                    str(checkout),
                    "--worker-write-root",
                    str(writable),
                    "--worker-receipt-fd",
                    str(receipt_write),
                ],
                cwd=checkout,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                pass_fds=(receipt_write,),
            )
        except Exception:
            os.close(receipt_read)
            os.close(receipt_write)
            raise
        os.close(receipt_write)
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=1800)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate(timeout=30)
        finally:
            # Candidate descendants inherit the worker's process group and
            # seccomp denies setpgid/setsid/unshare.  Kill any daemon that
            # closed its pipes and outlived the direct worker.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        receipt = bytearray()
        try:
            while True:
                chunk = os.read(receipt_read, 64 * 1024)
                if not chunk:
                    break
                receipt.extend(chunk)
                if len(receipt) > 1024 * 1024:
                    raise QualificationError(
                        f"{suite.suite_id}: worker receipt exceeds its bound"
                    )
        finally:
            os.close(receipt_read)
        if timed_out:
            raise QualificationError(
                f"{suite.suite_id}: worker exceeded the 1800-second wall bound; "
                + (stderr or stdout)[-1000:]
            )
        returncode = int(process.returncode or 0)
    try:
        receipt_text = bytes(receipt).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise QualificationError(f"{suite.suite_id}: worker receipt is not UTF-8") from exc
    lines = [line for line in receipt_text.splitlines() if line.strip()]
    if not lines:
        raise QualificationError(
            f"{suite.suite_id}: worker emitted no receipt "
            f"(returncode {returncode}): {(stderr or stdout)[-1000:]}"
        )
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise QualificationError(f"{suite.suite_id}: worker receipt is invalid") from exc
    if not isinstance(payload, dict) or payload.get("schema") != WORKER_SCHEMA:
        raise QualificationError(f"{suite.suite_id}: worker schema differs")
    if payload.get("suite_id") != suite.suite_id:
        raise QualificationError(f"{suite.suite_id}: worker identity differs")
    if "error" in payload:
        error = str(payload.get("error") or "unknown_error")
        reason = str(payload.get("reason") or "worker failed without a reason")
        raise QualificationError(
            f"{suite.suite_id}: worker failed closed: {error}: {reason[-1000:]}"
        )
    claimed = payload.get("observation_cid")
    body = {key: value for key, value in payload.items() if key != "observation_cid"}
    observed_identity = content_identity(body)
    if claimed != observed_identity:
        raise QualificationError(
            f"{suite.suite_id}: worker identity is forged "
            f"(claimed {claimed!r}, reconstructed {observed_identity!r})"
        )
    if payload.get("manifest") != dict(expected_manifest):
        raise QualificationError(f"{suite.suite_id}: copied source manifest differs")
    isolation = payload.get("isolation")
    if not _sandbox_evidence_is_valid(isolation):
        raise QualificationError(f"{suite.suite_id}: sandbox evidence differs")
    if returncode != 0 or payload.get("passed") is not True:
        reason = str(payload.get("failure_tail") or payload.get("reason") or "failed")
        raise QualificationError(f"{suite.suite_id}: {reason[-1000:]}")
    return payload


def build_result() -> dict[str, Any]:
    manifests = [_suite_manifest(suite) for suite in SUITES]
    before = _protected_input_projection(manifests)
    observations: list[dict[str, Any]] = []
    for suite, manifest in zip(SUITES, manifests, strict=True):
        observations.append(
            _run_suite(suite, expected_manifest=manifest)
        )
        current_manifests = [_suite_manifest(item) for item in SUITES]
        current = _protected_input_projection(current_manifests)
        if current_manifests != manifests or current != before:
            raise QualificationError(
                f"{suite.suite_id}: checkout changed during independent qualification"
            )
    final_manifests = [_suite_manifest(item) for item in SUITES]
    after = _protected_input_projection(final_manifests)
    totals = {
        key: sum(int(item[key]) for item in observations)
        for key in (
            "collected",
            "passed_count",
            "failed_count",
            "skipped_count",
            "xfailed_count",
            "xpassed_count",
            "error_count",
        )
    }
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "plan_cid": PLAN_CID,
        "predecessor_plan_cid": PREDECESSOR_PLAN_CID,
        "cohort": "hermetic_local_execution",
        "candidate_suites_are_self_authority": False,
        "independent_fixed_manifest_executed": True,
        "checkout_fingerprint_cid": before["fingerprint_cid"],
        "checkout_unchanged": after == before,
        "passed": all(item.get("passed") is True for item in observations),
        "totals": totals,
        "suites": observations,
        "task_implementation_complete": False,
        "test_qualification_complete": True,
        "objective_complete": False,
        "release_qualified": False,
        "production_authorized": False,
        "production_authoritative": False,
        "limitations": [
            "hermetic fixture and local test evidence only",
            "candidate-authored tests are corroborated but never self-certifying",
            "all judged suites run from exact copied inputs with checkout writes and network denied",
            "the protected-input root excludes only declared generated evidence outputs",
            "external qualification and operator authorization remain unavailable",
        ],
    }
    result["result_cid"] = content_identity(result)
    return result


def _stable_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    suites = []
    for item in value.get("suites", []):
        if not isinstance(item, Mapping):
            raise QualificationError("suite receipt is not an object")
        suites.append(
            {
                key: item.get(key)
                for key in (
                    "schema",
                    "suite_id",
                    "manifest",
                    "collected",
                    "passed_count",
                    "failed_count",
                    "skipped_count",
                    "xfailed_count",
                    "xpassed_count",
                    "error_count",
                    "nodeids_cid",
                    "exit_code",
                    "passed",
                    "isolation",
                )
            }
        )
    return {
        key: value.get(key)
        for key in (
            "schema",
            "plan_cid",
            "predecessor_plan_cid",
            "cohort",
            "candidate_suites_are_self_authority",
            "independent_fixed_manifest_executed",
            "checkout_fingerprint_cid",
            "checkout_unchanged",
            "passed",
            "totals",
            "task_implementation_complete",
            "test_qualification_complete",
            "objective_complete",
            "release_qualified",
            "production_authorized",
            "production_authoritative",
            "limitations",
        )
    } | {"suites": suites}


def validate_result(value: Mapping[str, Any]) -> None:
    expected_top_level = {
        "schema",
        "plan_cid",
        "predecessor_plan_cid",
        "cohort",
        "candidate_suites_are_self_authority",
        "independent_fixed_manifest_executed",
        "checkout_fingerprint_cid",
        "checkout_unchanged",
        "passed",
        "totals",
        "suites",
        "task_implementation_complete",
        "test_qualification_complete",
        "objective_complete",
        "release_qualified",
        "production_authorized",
        "production_authoritative",
        "limitations",
        "result_cid",
    }
    if set(value) != expected_top_level:
        raise QualificationError("qualification result fields differ from the closed schema")
    claimed = value.get("result_cid")
    body = {key: item for key, item in value.items() if key != "result_cid"}
    if claimed != content_identity(body):
        raise QualificationError("qualification result identity differs")
    if (
        value.get("schema") != SCHEMA
        or value.get("plan_cid") != PLAN_CID
        or value.get("predecessor_plan_cid") != PREDECESSOR_PLAN_CID
        or value.get("cohort") != "hermetic_local_execution"
    ):
        raise QualificationError("qualification result authority binding differs")
    if value.get("candidate_suites_are_self_authority") is not False:
        raise QualificationError("candidate suites cannot be completion authority")
    if value.get("independent_fixed_manifest_executed") is not True:
        raise QualificationError("fixed qualification manifest was not executed")
    if (
        value.get("checkout_unchanged") is not True
        or not isinstance(value.get("checkout_fingerprint_cid"), str)
        or not str(value.get("checkout_fingerprint_cid") or "").startswith("b")
    ):
        raise QualificationError("qualification checkout fingerprint is absent or changed")
    if value.get("passed") is not True or value.get("test_qualification_complete") is not True:
        raise QualificationError("qualification result is not a pass")
    for authority_field in (
        "task_implementation_complete",
        "objective_complete",
        "release_qualified",
        "production_authorized",
        "production_authoritative",
    ):
        if value.get(authority_field) is not False:
            raise QualificationError(f"qualification result raises {authority_field}")

    observations = value.get("suites")
    if not isinstance(observations, list) or len(observations) != len(SUITES):
        raise QualificationError("qualification suite population differs")
    count_fields = (
        "collected",
        "passed_count",
        "failed_count",
        "skipped_count",
        "xfailed_count",
        "xpassed_count",
        "error_count",
    )
    expected_observation_fields = {
        "schema",
        "suite_id",
        "manifest",
        *count_fields,
        "nodeids_cid",
        "exit_code",
        "passed",
        "isolation",
        "duration_ms",
        "transcript_sha256",
        "failure_tail",
        "observation_cid",
    }
    observed_totals = dict.fromkeys(count_fields, 0)
    for expected_suite, observation in zip(SUITES, observations, strict=True):
        if not isinstance(observation, Mapping) or set(observation) != expected_observation_fields:
            raise QualificationError("qualification suite fields differ from the closed schema")
        if (
            observation.get("schema") != WORKER_SCHEMA
            or observation.get("suite_id") != expected_suite.suite_id
            or observation.get("manifest") != _suite_manifest(expected_suite)
        ):
            raise QualificationError(f"{expected_suite.suite_id}: suite authority differs")
        observation_body = {
            key: item for key, item in observation.items() if key != "observation_cid"
        }
        if observation.get("observation_cid") != content_identity(observation_body):
            raise QualificationError(f"{expected_suite.suite_id}: observation identity differs")
        isolation = observation.get("isolation")
        if not _sandbox_evidence_is_valid(isolation):
            raise QualificationError(f"{expected_suite.suite_id}: sandbox evidence differs")
        raw_counts = {count_field: observation.get(count_field) for count_field in count_fields}
        if any(
            isinstance(count, bool) or not isinstance(count, int)
            for count in raw_counts.values()
        ):
            raise QualificationError(f"{expected_suite.suite_id}: suite counts are invalid")
        counts = {count_field: int(count) for count_field, count in raw_counts.items()}
        if (
            any(count < 0 for count in counts.values())
            or counts["collected"] <= 0
            or counts["passed_count"] != counts["collected"]
            or any(
                counts[field]
                for field in (
                    "failed_count",
                    "skipped_count",
                    "xfailed_count",
                    "xpassed_count",
                    "error_count",
                )
            )
            or observation.get("exit_code") != 0
            or observation.get("passed") is not True
            or observation.get("failure_tail") != ""
            or isinstance(observation.get("duration_ms"), bool)
            or not isinstance(observation.get("duration_ms"), int)
            or int(observation["duration_ms"]) < 0
            or not str(observation.get("nodeids_cid") or "").startswith("b")
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(observation.get("transcript_sha256") or ""),
            )
        ):
            raise QualificationError(f"{expected_suite.suite_id}: suite did not pass exactly")
        for count_field, count in counts.items():
            observed_totals[count_field] += count

    totals = value.get("totals")
    if not isinstance(totals, Mapping) or dict(totals) != observed_totals:
        raise QualificationError("qualification totals do not reconstruct")
    if any(
        observed_totals[field]
        for field in (
            "failed_count",
            "skipped_count",
            "xfailed_count",
            "xpassed_count",
            "error_count",
        )
    ):
        raise QualificationError("qualification result contains non-passing outcomes")


def _write_atomic(value: Mapping[str, Any]) -> None:
    validate_result(value)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".lgcvf-qualification-", dir=OUTPUT.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, OUTPUT)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--worker", metavar="SUITE_ID", help=argparse.SUPPRESS)
    parser.add_argument("--worker-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-write-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-receipt-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if (
            args.worker_root is None
            or args.worker_write_root is None
            or args.worker_receipt_fd is None
        ):
            print(json.dumps({"schema": WORKER_SCHEMA, "error": "sandbox_required"}))
            return 2
        return _worker(
            args.worker,
            execution_root=args.worker_root,
            write_root=args.worker_write_root,
            receipt_descriptor=args.worker_receipt_fd,
        )
    if (
        args.worker_root is not None
        or args.worker_write_root is not None
        or args.worker_receipt_fd is not None
    ):
        parser.error("worker sandbox arguments require --worker")
    try:
        current: dict[str, Any] | None = None
        if args.check:
            current_value = json.loads(OUTPUT.read_text(encoding="utf-8"))
            if not isinstance(current_value, dict):
                raise QualificationError("qualification result root is not an object")
            validate_result(current_value)
            current = current_value
        reconstructed = build_result()
        validate_result(reconstructed)
        if current is not None and _stable_projection(current) != _stable_projection(reconstructed):
            raise QualificationError("qualification reconstruction differs")
        if current is None:
            _write_atomic(reconstructed)
        print(json.dumps(reconstructed, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError, QualificationError) as exc:
        print(json.dumps({"valid": False, "error": type(exc).__name__, "reason": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
