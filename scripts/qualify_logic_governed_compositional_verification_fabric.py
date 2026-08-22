#!/usr/bin/env python3
"""Produce or independently reconstruct the hermetic LGCVF qualification.

The two task-authored qualification suites are candidate evidence.  This
protected judge also runs a fixed manifest of already accepted semantic and
supervisor tests, rejects skips and expected failures, binds every test source,
and never grants release or production authority.
"""

from __future__ import annotations

import argparse
import atexit
import base64
import binascii
import contextlib
import csv
import ctypes
import ctypes.util
import email.parser
import email.policy
import errno
import gc
import grp
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import io
import json
import os
import platform
import pwd
import re
import resource
import select
import shutil
import signal
import site
import stat
import struct
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from threading import Thread
from typing import Any, Final

_RECOVERY_PYCACHE_CAPSULE_ATTRIBUTE: Final[str] = (
    "_lgcvf_isolated_recovery_pycache_capsule_v1"
)
_RECOVERY_PYCACHE_CAPSULE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/lgcvf-isolated-recovery-pycache@1"
)
_RECOVERY_INITIAL_STDLIB_ATTRIBUTE: Final[str] = (
    "_lgcvf_initial_isolated_stdlib_paths_v1"
)
_RECOVERY_WORKER_BOOTSTRAP_ATTRIBUTE: Final[str] = (
    "_lgcvf_qualification_runtime_bootstrap_v1"
)
_RECOVERY_EXPECTED_ISOLATED_STDLIB_PATHS: Final[tuple[str, ...]] = (
    "/usr/lib/python312.zip",
    "/usr/lib/python3.12",
    "/usr/lib/python3.12/lib-dynload",
)
if not hasattr(sys, _RECOVERY_INITIAL_STDLIB_ATTRIBUTE):
    if tuple(sys.path) == _RECOVERY_EXPECTED_ISOLATED_STDLIB_PATHS:
        setattr(
            sys,
            _RECOVERY_INITIAL_STDLIB_ATTRIBUTE,
            _RECOVERY_EXPECTED_ISOLATED_STDLIB_PATHS,
        )
_RECOVERY_INITIAL_ISOLATED_STDLIB_PATHS = getattr(
    sys, _RECOVERY_INITIAL_STDLIB_ATTRIBUTE, None
)
_RECOVERY_WORKER_BOOTSTRAP_CAPSULE = getattr(
    sys, _RECOVERY_WORKER_BOOTSTRAP_ATTRIBUTE, None
)


def _validated_isolated_recovery_pycache_capsule(
    capsule: object,
) -> tuple[tempfile.TemporaryDirectory[str], Path, tuple[int, int]]:
    """Validate one process-lifetime, empty, owner-private pycache capsule."""

    if not isinstance(capsule, tuple) or len(capsule) != 7:
        raise RuntimeError("protected recovery pycache isolation conflict")
    schema, process_id, directory, root_text, device, inode, seal = capsule
    if (
        schema != _RECOVERY_PYCACHE_CAPSULE_SCHEMA
        or type(process_id) is not int
        or process_id != os.getpid()
        or not isinstance(directory, tempfile.TemporaryDirectory)
        or type(root_text) is not str
        or type(device) is not int
        or type(inode) is not int
        or type(seal) is not object
        or getattr(sys, _RECOVERY_PYCACHE_CAPSULE_ATTRIBUTE, None) is not capsule
        or sys.pycache_prefix != root_text
        or directory.name != root_text
    ):
        raise RuntimeError("protected recovery pycache isolation conflict")
    root = Path(root_text)
    if not root.is_absolute() or root.as_posix() != root_text:
        raise RuntimeError("protected recovery pycache isolation conflict")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(root, flags)
    except OSError as exc:
        raise RuntimeError("protected recovery pycache isolation conflict") from exc
    try:
        status = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(status.st_mode)
            or (status.st_dev, status.st_ino) != (device, inode)
            or status.st_uid != os.geteuid()
            or stat.S_IMODE(status.st_mode) != 0o700
        ):
            raise RuntimeError("protected recovery pycache isolation conflict")
        with os.scandir(descriptor) as entries:
            if next(entries, None) is not None:
                raise RuntimeError("protected recovery pycache isolation is not empty")
    finally:
        os.close(descriptor)
    return directory, root, (device, inode)


def _validate_external_isolated_recovery_pycache_prefix(prefix: object) -> None:
    """Accept only the scheduler's empty, owner-private bootstrap prefix."""

    if type(prefix) is not str:
        raise RuntimeError("protected recovery pycache isolation conflict")
    root = Path(prefix)
    if not root.is_absolute() or root.as_posix() != prefix:
        raise RuntimeError("protected recovery pycache isolation conflict")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(root, flags)
    except OSError as exc:
        raise RuntimeError("protected recovery pycache isolation conflict") from exc
    try:
        status = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(status.st_mode)
            or status.st_uid != os.geteuid()
            or stat.S_IMODE(status.st_mode) != 0o700
        ):
            raise RuntimeError("protected recovery pycache isolation conflict")
        with os.scandir(descriptor) as entries:
            if next(entries, None) is not None:
                raise RuntimeError("protected recovery pycache isolation is not empty")
    finally:
        os.close(descriptor)


def _install_isolated_recovery_pycache() -> tuple[
    tempfile.TemporaryDirectory[str] | None,
    Path | None,
    tuple[int, int] | None,
    object | None,
]:
    """Redirect exact isolated execution away from repository bytecode caches."""

    flags = sys.flags
    if (
        flags.isolated != 1
        or flags.ignore_environment != 1
        or flags.no_site != 1
        or flags.safe_path is not True
        or flags.dont_write_bytecode != 1
        or sys.dont_write_bytecode is not True
    ):
        return None, None, None, None
    if hasattr(sys, _RECOVERY_PYCACHE_CAPSULE_ATTRIBUTE):
        existing = getattr(sys, _RECOVERY_PYCACHE_CAPSULE_ATTRIBUTE)
        directory, root, identity = (
            _validated_isolated_recovery_pycache_capsule(existing)
        )
        return directory, root, identity, existing
    previous_prefix = sys.pycache_prefix
    if previous_prefix is not None:
        _validate_external_isolated_recovery_pycache_prefix(previous_prefix)
    directory = tempfile.TemporaryDirectory(prefix="lgcvf-isolated-pycache-")
    root = Path(directory.name).resolve(strict=True)
    os.chmod(root, 0o700)
    status = root.lstat()
    sys.pycache_prefix = str(root)
    capsule = (
        _RECOVERY_PYCACHE_CAPSULE_SCHEMA,
        os.getpid(),
        directory,
        str(root),
        status.st_dev,
        status.st_ino,
        object(),
    )
    setattr(sys, _RECOVERY_PYCACHE_CAPSULE_ATTRIBUTE, capsule)
    try:
        observed_directory, observed_root, identity = (
            _validated_isolated_recovery_pycache_capsule(capsule)
        )
    except BaseException:
        if getattr(sys, _RECOVERY_PYCACHE_CAPSULE_ATTRIBUTE, None) is capsule:
            delattr(sys, _RECOVERY_PYCACHE_CAPSULE_ATTRIBUTE)
        sys.pycache_prefix = previous_prefix
        directory.cleanup()
        raise
    return observed_directory, observed_root, identity, capsule


(
    _ISOLATED_RECOVERY_PYCACHE_DIRECTORY,
    _ISOLATED_RECOVERY_PYCACHE_ROOT,
    _ISOLATED_RECOVERY_PYCACHE_IDENTITY,
    _ISOLATED_RECOVERY_PYCACHE_CAPSULE,
) = _install_isolated_recovery_pycache()

ROOT: Final[Path] = Path(__file__).resolve().parents[1]

_MAX_RECOVERY_IMPORT_ROOT_ENTRIES: Final[int] = 32_768
_MAX_RECOVERY_TRACKED_PATH_BYTES: Final[int] = 16 * 1024 * 1024
_MAX_RECOVERY_IMPORT_DEPTH: Final[int] = 64
_MAX_RECOVERY_IMPORT_FILE_BYTES: Final[int] = 32 * 1024 * 1024
_MAX_RECOVERY_IMPORT_TOTAL_BYTES: Final[int] = 512 * 1024 * 1024
_RECOVERY_GIT_CONFIG_OVERRIDES: Final[tuple[str, ...]] = (
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.untrackedCache=false",
    "-c",
    "core.trustctime=true",
    "-c",
    "core.checkStat=default",
    "-c",
    "core.attributesFile=/dev/null",
)
_RECOVERY_OMITTED_SOURCE_SYMLINKS: Final[dict[str, str]] = {
    "test/run_dashboard_tests.py": (
        "/home/barberb/ipfs_accelerate_py/test/duckdb_api/visualization/"
        "advanced_visualization/test_customizable_dashboard.py"
    ),
}


def _tracked_recovery_import_entries(
    root: Path,
    *,
    pathspecs: tuple[str, ...],
) -> dict[str, tuple[str, str]]:
    """Bind ordinary stage-zero entries exactly to the current HEAD tree."""

    tracked_payload = _bounded_recovery_git(
        root,
        "ls-files",
        "-v",
        "-z",
        "--cached",
        "--",
        *pathspecs,
    )
    stage_payload = _bounded_recovery_git(
        root, "ls-files", "-s", "-z", "--cached", "--", *pathspecs
    )
    head_payload = _bounded_recovery_git(
        root, "ls-tree", "-r", "-z", "--full-tree", "HEAD"
    )
    if any(
        payload and not payload.endswith(b"\0")
        for payload in (tracked_payload, stage_payload, head_payload)
    ):
        raise RuntimeError("protected recovery import inventory is unavailable")
    tracked: set[str] = set()
    for raw in tracked_payload.split(b"\0")[:-1]:
        if not raw.startswith(b"H "):
            raise RuntimeError("protected recovery tracked path has index flags")
        try:
            value = raw[2:].decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise RuntimeError(
                "protected recovery tracked path is not UTF-8"
            ) from exc
        logical = PurePath(value)
        if (
            not value
            or logical.is_absolute()
            or ".." in logical.parts
            or logical.as_posix() != value
            or value in tracked
        ):
            raise RuntimeError("protected recovery tracked-path set is ambiguous")
        tracked.add(value)

    def parse_entries(payload: bytes, *, head: bool) -> dict[str, tuple[str, str]]:
        result: dict[str, tuple[str, str]] = {}
        for raw in payload.split(b"\0")[:-1]:
            try:
                metadata, path_bytes = raw.split(b"\t", 1)
                fields = metadata.decode("ascii", errors="strict").split(" ")
                value = path_bytes.decode("utf-8", errors="strict")
            except (UnicodeDecodeError, ValueError) as exc:
                raise RuntimeError(
                    "protected recovery Git entry is malformed"
                ) from exc
            logical = PurePath(value)
            if (
                not value
                or logical.is_absolute()
                or ".." in logical.parts
                or logical.as_posix() != value
                or value in result
            ):
                raise RuntimeError("protected recovery Git entry is ambiguous")
            if head:
                if (
                    len(fields) != 3
                    or fields[1] not in {"blob", "commit"}
                    or (fields[0] == "160000") != (fields[1] == "commit")
                ):
                    raise RuntimeError("protected recovery HEAD entry differs")
                mode, object_id = fields[0], fields[2]
            else:
                if len(fields) != 3 or fields[2] != "0":
                    raise RuntimeError("protected recovery index entry differs")
                mode, object_id = fields[0], fields[1]
            if (
                mode not in {"100644", "100755", "120000", "160000"}
                or len(object_id) not in {40, 64}
                or any(character not in "0123456789abcdef" for character in object_id)
            ):
                raise RuntimeError("protected recovery Git blob identity differs")
            result[value] = (mode, object_id)
        return result

    stage = parse_entries(stage_payload, head=False)
    head_all = parse_entries(head_payload, head=True)
    head = {value: head_all[value] for value in tracked if value in head_all}
    if set(stage) != tracked or head != stage:
        raise RuntimeError(
            "protected recovery ordinary index differs from current HEAD"
        )
    return stage


def _git_blob_matches(payload: bytes, object_id: str) -> bool:
    """Return whether raw bytes have the exact repository blob identity."""

    constructor = hashlib.sha1 if len(object_id) == 40 else hashlib.sha256
    digest = constructor(f"blob {len(payload)}\0".encode("ascii"))
    digest.update(payload)
    return digest.hexdigest() == object_id


def _bounded_recovery_git(root: Path, *arguments: str) -> bytes:
    """Run one fixed local Git observation with bounded stdout and time."""

    git = Path("/usr/bin/git")
    try:
        git_status = git.lstat()
    except OSError as exc:
        raise RuntimeError("protected recovery import inventory is unavailable") from exc
    if (
        not stat.S_ISREG(git_status.st_mode)
        or git_status.st_uid != 0
        or git_status.st_mode & 0o022
    ):
        raise RuntimeError("protected recovery import inventory is unavailable")
    process = subprocess.Popen(
        (
            str(git),
            *_RECOVERY_GIT_CONFIG_OVERRIDES,
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(root),
            *arguments,
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env={
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "HOME": "/nonexistent",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        },
        close_fds=True,
    )
    if process.stdout is None:
        process.kill()
        process.wait()
        raise RuntimeError("protected recovery import inventory is unavailable")
    payload = bytearray()
    deadline = time.monotonic() + 15.0
    try:
        descriptor = process.stdout.fileno()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("protected recovery import inventory timed out")
            readable, _, _ = select.select((descriptor,), (), (), remaining)
            if not readable:
                raise RuntimeError("protected recovery import inventory timed out")
            block = os.read(
                descriptor,
                min(
                    64 * 1024,
                    _MAX_RECOVERY_TRACKED_PATH_BYTES + 1 - len(payload),
                ),
            )
            if not block:
                break
            payload.extend(block)
            if len(payload) > _MAX_RECOVERY_TRACKED_PATH_BYTES:
                raise RuntimeError("protected recovery tracked-path set is too large")
        returncode = process.wait(timeout=max(0.1, deadline - time.monotonic()))
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.wait()
        raise
    finally:
        process.stdout.close()
    if returncode != 0:
        raise RuntimeError("protected recovery import inventory is unavailable")
    return bytes(payload)


def _git_object_substitution_state(root: Path) -> tuple[str, int, int]:
    """Reject legacy grafts and replacement refs in the exact Git common dir."""

    raw_common = _bounded_recovery_git(
        root, "rev-parse", "--path-format=absolute", "--git-common-dir"
    )
    if (
        not raw_common.endswith(b"\n")
        or raw_common.count(b"\n") != 1
        or b"\0" in raw_common
        or len(raw_common) > 4_096
    ):
        raise RuntimeError("protected recovery Git common directory is ambiguous")
    try:
        common = Path(raw_common[:-1].decode("utf-8", errors="strict"))
        resolved = common.resolve(strict=True)
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            "protected recovery Git common directory is unavailable"
        ) from exc
    if not common.is_absolute() or resolved != common:
        raise RuntimeError("protected recovery Git common directory is unsafe")

    uid, private_gid = _private_recovery_import_principal()
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        common_fd = os.open(common, flags)
    except OSError as exc:
        raise RuntimeError(
            "protected recovery Git common directory is unavailable"
        ) from exc
    try:
        common_status = os.fstat(common_fd)
        common_mode = stat.S_IMODE(common_status.st_mode)
        if (
            not stat.S_ISDIR(common_status.st_mode)
            or common_status.st_uid != uid
            or common_mode & 0o002
            or (common_mode & 0o020 and common_status.st_gid != private_gid)
        ):
            raise RuntimeError(
                "protected recovery Git common directory permissions differ"
            )
        for directory, leaf in (
            ("info", "grafts"),
            ("info", "attributes"),
            ("refs", "replace"),
        ):
            try:
                directory_fd = os.open(directory, flags, dir_fd=common_fd)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise RuntimeError(
                    "protected recovery Git substitution path is unsafe"
                ) from exc
            try:
                try:
                    os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    raise RuntimeError(
                        "protected recovery Git object substitution is present"
                    )
            finally:
                os.close(directory_fd)
    finally:
        os.close(common_fd)
    replacements = _bounded_recovery_git(
        root, "for-each-ref", "--format=%(refname)", "refs/replace"
    )
    if replacements:
        raise RuntimeError("protected recovery Git replacement refs are present")
    config_names = _bounded_recovery_git(root, "config", "--name-only", "--list")
    if any(line.lower().startswith(b"filter.") for line in config_names.splitlines()):
        raise RuntimeError("protected recovery Git filter drivers are present")
    return str(common), common_status.st_dev, common_status.st_ino


def _clean_recovery_import_source(root: Path) -> tuple[str, str] | None:
    """Bind one clean repository HEAD/tree before checkout code can execute."""

    if _ISOLATED_RECOVERY_PYCACHE_DIRECTORY is None:
        return None
    substitution_before = _git_object_substitution_state(root)
    head = _bounded_recovery_git(root, "rev-parse", "--verify", "HEAD").strip()
    tree = _bounded_recovery_git(
        root, "rev-parse", "--verify", "HEAD^{tree}"
    ).strip()
    status = _bounded_recovery_git(
        root,
        "status",
        "--porcelain=v2",
        "-z",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    if (
        status
        or _git_object_substitution_state(root) != substitution_before
        or len(head) not in {40, 64}
        or len(tree) != len(head)
        or any(character not in b"0123456789abcdef" for character in head + tree)
    ):
        raise RuntimeError("protected recovery import source is not clean")
    return head.decode("ascii"), tree.decode("ascii")


def _datasets_recovery_import_source(root: Path) -> tuple[str, str, str] | None:
    """Bind the clean nested datasets checkout to the outer Git gitlink."""

    if _ISOLATED_RECOVERY_PYCACHE_DIRECTORY is None:
        return None
    nested = root / "ipfs_datasets_py"
    nested_identity = _clean_recovery_import_source(nested)
    if nested_identity is None:
        raise RuntimeError("protected recovery datasets source is unavailable")
    head, tree = nested_identity
    stage = _bounded_recovery_git(
        root, "ls-files", "-s", "-z", "--", "ipfs_datasets_py"
    )
    tracked = _bounded_recovery_git(
        root, "ls-files", "-v", "-z", "--", "ipfs_datasets_py"
    )
    committed = _bounded_recovery_git(
        root, "ls-tree", "-z", "HEAD", "--", "ipfs_datasets_py"
    )
    expected_stage = f"160000 {head} 0\tipfs_datasets_py\0".encode("ascii")
    expected_commit = f"160000 commit {head}\tipfs_datasets_py\0".encode("ascii")
    if (
        stage != expected_stage
        or tracked != b"H ipfs_datasets_py\0"
        or committed != expected_commit
    ):
        raise RuntimeError("protected recovery datasets gitlink differs")
    return head, tree, head


def _private_recovery_import_principal() -> tuple[int, int]:
    """Require the current primary group to be a single-principal private group."""

    uid = os.geteuid()
    try:
        account = pwd.getpwuid(uid)
        group = grp.getgrgid(account.pw_gid)
    except KeyError as exc:
        raise RuntimeError("protected recovery import principal is ambiguous") from exc
    if (
        account.pw_uid != uid
        or account.pw_gid != os.getegid()
        or group.gr_gid != account.pw_gid
        or group.gr_mem
    ):
        raise RuntimeError("protected recovery import principal is ambiguous")
    primary_accounts = []
    for index, candidate in enumerate(pwd.getpwall()):
        if index >= 4_096:
            raise RuntimeError("protected recovery account database is too large")
        if candidate.pw_gid == account.pw_gid:
            primary_accounts.append((candidate.pw_uid, candidate.pw_name))
    if primary_accounts != [(uid, account.pw_name)]:
        raise RuntimeError("protected recovery import group is not private")
    return uid, account.pw_gid


def _scan_isolated_recovery_import_roots(
    root: Path,
    *,
    roots: tuple[str, ...] = ("scripts", "ipfs_accelerate_py", "test"),
    tracked_pathspecs: tuple[str, ...] = (
        "scripts",
        "ipfs_accelerate_py",
        "test",
        ":(top,glob)*.py",
        ":(top,glob)*.pyc",
        ":(top,glob)*.pyo",
        ":(top,glob)*.so",
    ),
    whole_repository: bool = False,
    root_import_candidates: bool = True,
) -> tuple[
    tuple[str, str, int, int, int, int, int, int, int, int, str], ...
] | None:
    """Close importable checkout files before the checkout enters ``sys.path``."""

    if _ISOLATED_RECOVERY_PYCACHE_DIRECTORY is None:
        return None
    git_entries = _tracked_recovery_import_entries(root, pathspecs=tracked_pathspecs)
    tracked = frozenset(git_entries)
    gitlink_paths = frozenset(
        value for value, (mode, _object_id) in git_entries.items() if mode == "160000"
    )
    uid, private_gid = _private_recovery_import_principal()
    records: list[
        tuple[str, str, int, int, int, int, int, int, int, int, str]
    ] = []
    entry_count = 0
    total_file_bytes = 0
    observed_blob_paths: set[str] = set()
    observed_link_paths: set[str] = set()
    directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)

    native_suffixes = (
        ".so",
        ".pyd",
        ".dylib",
        *importlib.machinery.EXTENSION_SUFFIXES,
    )

    def is_import_candidate(value: str) -> bool:
        return value.endswith(".py") or value.endswith(native_suffixes)

    def is_gitlink_import_artifact(value: str) -> bool:
        return value.endswith((".py", ".pyc", ".pyo", *native_suffixes))

    def verify_blob_file(
        directory_fd: int,
        name: str,
        status: os.stat_result,
        value: str,
    ) -> str:
        """Stream one stable file and join it to its index/HEAD blob."""

        nonlocal total_file_bytes
        git_entry = git_entries.get(value)
        if git_entry is None or git_entry[0] in {"120000", "160000"}:
            raise RuntimeError(
                f"protected recovery import candidate is untracked: {value}"
            )
        descriptor = os.open(name, file_flags, dir_fd=directory_fd)
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                )
                != (
                    status.st_dev,
                    status.st_ino,
                    status.st_size,
                    status.st_mtime_ns,
                    status.st_ctime_ns,
                )
                or before.st_size < 0
                or before.st_size > _MAX_RECOVERY_IMPORT_FILE_BYTES
            ):
                raise RuntimeError(
                    f"protected recovery import candidate identity differs: {value}"
                )
            expected_mode = "100755" if stat.S_IMODE(before.st_mode) & 0o111 else "100644"
            mode, object_id = git_entry
            if mode != expected_mode:
                raise RuntimeError(
                    f"protected recovery import candidate mode differs: {value}"
                )
            total_file_bytes += before.st_size
            if total_file_bytes > _MAX_RECOVERY_IMPORT_TOTAL_BYTES:
                raise RuntimeError("protected recovery import bytes exceed their bound")
            constructor = hashlib.sha1 if len(object_id) == 40 else hashlib.sha256
            digest = constructor(f"blob {before.st_size}\0".encode("ascii"))
            observed = 0
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                observed += len(block)
                if observed > before.st_size:
                    raise RuntimeError(
                        f"protected recovery import candidate grew: {value}"
                    )
                digest.update(block)
            after = os.fstat(descriptor)
            if (
                observed != before.st_size
                or (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                )
                != (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                )
                or digest.hexdigest() != object_id
            ):
                raise RuntimeError(
                    f"protected recovery import candidate differs from HEAD: {value}"
                )
        finally:
            os.close(descriptor)
        observed_blob_paths.add(value)
        return f"{mode}:{object_id}"

    def record(
        status: os.stat_result,
        value: str,
        kind: str,
        detail: str = "",
    ) -> None:
        mode = stat.S_IMODE(status.st_mode)
        if (
            status.st_uid != uid
            or status.st_nlink < 1
            or mode & 0o002
            or (mode & 0o020 and status.st_gid != private_gid)
            or (kind == "directory" and mode & 0o500 != 0o500)
            or (kind != "directory" and status.st_nlink != 1)
            or (kind != "directory" and mode & 0o400 != 0o400)
        ):
            raise RuntimeError(
                f"protected recovery import candidate permissions differ: {value}"
            )
        records.append(
            (
                value,
                kind,
                status.st_dev,
                status.st_ino,
                status.st_size,
                status.st_mtime_ns,
                status.st_uid,
                status.st_gid,
                mode,
                status.st_nlink,
                detail,
            )
        )

    def visit(
        directory_fd: int,
        relative: PurePath,
        depth: int,
        *,
        inside_gitlink: bool = False,
    ) -> None:
        nonlocal entry_count
        if depth > _MAX_RECOVERY_IMPORT_DEPTH:
            raise RuntimeError("protected recovery import tree is too deep")
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                if entry.name == ".git":
                    continue
                entry_count += 1
                if entry_count > _MAX_RECOVERY_IMPORT_ROOT_ENTRIES:
                    raise RuntimeError("protected recovery import tree is too large")
                child = relative / entry.name
                value = child.as_posix()
                status = entry.stat(follow_symlinks=False)
                if value in gitlink_paths and not stat.S_ISDIR(status.st_mode):
                    raise RuntimeError(
                        f"protected recovery Gitlink placeholder is unsafe: {value}"
                    )
                child_inside_gitlink = inside_gitlink or value in gitlink_paths
                if (
                    child_inside_gitlink
                    and not stat.S_ISDIR(status.st_mode)
                    and is_gitlink_import_artifact(entry.name)
                ):
                    raise RuntimeError(
                        "protected recovery Gitlink contains an import candidate: "
                        f"{value}"
                    )
                if stat.S_ISLNK(status.st_mode):
                    target = os.readlink(entry.name, dir_fd=directory_fd)
                    importable_link = entry.name.endswith(
                        (
                            ".py",
                            ".pyc",
                            ".pyo",
                            ".so",
                            *importlib.machinery.EXTENSION_SUFFIXES,
                        )
                    )
                    admitted_omission = (
                        not whole_repository
                        and _RECOVERY_OMITTED_SOURCE_SYMLINKS.get(value) == target
                    )
                    git_entry = git_entries.get(value)
                    target_bytes = target.encode("utf-8", errors="surrogateescape")
                    if (
                        value not in tracked
                        or git_entry is None
                        or git_entry[0] != "120000"
                        or not _git_blob_matches(target_bytes, git_entry[1])
                        or (importable_link and not admitted_omission)
                    ):
                        raise RuntimeError(
                            f"protected recovery import tree contains an unsafe link: {value}"
                        )
                    normalized_target = os.path.normpath(
                        (child.parent / target).as_posix()
                    )
                    if (
                        not target
                        or (
                            importable_link
                            and not admitted_omission
                            and (
                                os.path.isabs(target)
                                or normalized_target == ".."
                                or normalized_target.startswith("../")
                            )
                        )
                    ):
                        raise RuntimeError(
                            f"protected recovery import link escapes its repository: {value}"
                        )
                    after = os.stat(
                        entry.name, dir_fd=directory_fd, follow_symlinks=False
                    )
                    if (
                        not stat.S_ISLNK(after.st_mode)
                        or (after.st_dev, after.st_ino, after.st_mtime_ns)
                        != (status.st_dev, status.st_ino, status.st_mtime_ns)
                    ):
                        raise RuntimeError("protected recovery import link changed")
                    if status.st_uid != uid or status.st_nlink != 1:
                        raise RuntimeError(
                            f"protected recovery import link ownership differs: {value}"
                        )
                    records.append(
                        (
                            value,
                            "symlink",
                            status.st_dev,
                            status.st_ino,
                            status.st_size,
                            status.st_mtime_ns,
                            status.st_uid,
                            status.st_gid,
                            stat.S_IMODE(status.st_mode),
                            status.st_nlink,
                            target,
                        )
                    )
                    observed_link_paths.add(value)
                    continue
                if stat.S_ISDIR(status.st_mode):
                    if entry.name == "__pycache__" and not child_inside_gitlink:
                        continue
                    record(status, value, "directory")
                    child_fd = os.open(entry.name, directory_flags, dir_fd=directory_fd)
                    try:
                        opened = os.fstat(child_fd)
                        if (
                            not stat.S_ISDIR(opened.st_mode)
                            or (opened.st_dev, opened.st_ino)
                            != (status.st_dev, status.st_ino)
                        ):
                            raise RuntimeError(
                                "protected recovery import directory changed"
                            )
                        visit(
                            child_fd,
                            child,
                            depth + 1,
                            inside_gitlink=child_inside_gitlink,
                        )
                    finally:
                        os.close(child_fd)
                    continue
                python_source = entry.name.endswith(".py")
                bytecode = entry.name.endswith((".pyc", ".pyo"))
                native = entry.name.endswith(native_suffixes)
                if bytecode and "__pycache__" not in child.parts:
                    raise RuntimeError(
                        f"protected recovery import tree contains bytecode: {value}"
                    )
                if not python_source and not native:
                    continue
                if not stat.S_ISREG(status.st_mode) or value not in tracked:
                    raise RuntimeError(
                        f"protected recovery import candidate is untracked: {value}"
                    )
                detail = verify_blob_file(directory_fd, entry.name, status, value)
                record(
                    status,
                    value,
                    "python" if python_source else "native",
                    detail,
                )

    root_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
    root_flags |= getattr(os, "O_NOFOLLOW", 0)
    root_fd = os.open(root, root_flags)
    try:
        record(os.fstat(root_fd), ".", "directory")
        if whole_repository:
            visit(root_fd, PurePath("."), 0)
        for name in (() if whole_repository else roots):
            directory_fd = os.open(name, directory_flags, dir_fd=root_fd)
            try:
                record(os.fstat(directory_fd), name, "directory")
                visit(directory_fd, PurePath(name), 0)
            finally:
                os.close(directory_fd)
        if root_import_candidates and not whole_repository:
            with os.scandir(root_fd) as entries:
                for entry in entries:
                    entry_count += 1
                    if entry_count > _MAX_RECOVERY_IMPORT_ROOT_ENTRIES:
                        raise RuntimeError(
                            "protected recovery import tree is too large"
                        )
                    name = entry.name
                    python_source = name.endswith(".py")
                    bytecode = name.endswith((".pyc", ".pyo"))
                    native = name.endswith(native_suffixes)
                    if not python_source and not bytecode and not native:
                        continue
                    status = entry.stat(follow_symlinks=False)
                    if bytecode:
                        raise RuntimeError(
                            f"protected recovery import tree contains bytecode: {name}"
                        )
                    if (
                        stat.S_ISLNK(status.st_mode)
                        or not stat.S_ISREG(status.st_mode)
                        or name not in tracked
                    ):
                        raise RuntimeError(
                            f"protected recovery import candidate is untracked: {name}"
                        )
                    detail = verify_blob_file(root_fd, name, status, name)
                    record(
                        status,
                        name,
                        "python" if python_source else "native",
                        detail,
                    )
    finally:
        os.close(root_fd)
    expected_blob_paths = {
        value
        for value, (mode, _object_id) in git_entries.items()
        if mode in {"100644", "100755"} and is_import_candidate(value)
    }
    expected_link_paths = {
        value for value, (mode, _object_id) in git_entries.items() if mode == "120000"
    }
    if (
        observed_blob_paths != expected_blob_paths
        or observed_link_paths != expected_link_paths
    ):
        raise RuntimeError(
            "protected recovery import filesystem differs from index and HEAD"
        )
    return tuple(sorted(records))


if _RECOVERY_WORKER_BOOTSTRAP_CAPSULE is None:
    _ISOLATED_RECOVERY_SOURCE_IDENTITY = _clean_recovery_import_source(ROOT)
    _ISOLATED_RECOVERY_IMPORT_INVENTORY = _scan_isolated_recovery_import_roots(ROOT)
    _ISOLATED_RECOVERY_DATASETS_SOURCE_IDENTITY = _datasets_recovery_import_source(ROOT)
    _ISOLATED_RECOVERY_DATASETS_IMPORT_INVENTORY = (
        _scan_isolated_recovery_import_roots(
            ROOT / "ipfs_datasets_py",
            roots=(),
            tracked_pathspecs=(".",),
            whole_repository=True,
            root_import_candidates=False,
        )
    )
else:
    _ISOLATED_RECOVERY_SOURCE_IDENTITY = None
    _ISOLATED_RECOVERY_IMPORT_INVENTORY = None
    _ISOLATED_RECOVERY_DATASETS_SOURCE_IDENTITY = None
    _ISOLATED_RECOVERY_DATASETS_IMPORT_INVENTORY = None

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (  # noqa: E402
    content_identity,
)

SCHEMA: Final[str] = "lgcvf-independent-hermetic-qualification@1"
WORKER_SCHEMA: Final[str] = "lgcvf-independent-pytest-observation@1"
RECOVERY_SCHEMA: Final[str] = "lgcvf-independent-recovery-qualification@9"
RECOVERY_WORKER_SCHEMA: Final[str] = (
    "lgcvf-independent-recovery-pytest-observation@9"
)
RECOVERY_UNAVAILABLE_SCHEMA: Final[str] = (
    "lgcvf-independent-recovery-qualification-unavailable@3"
)
_RECOVERY_UNAVAILABLE_CLASSIFICATIONS: Final[dict[tuple[str, str], str]] = {
    ("pin", "identity_mismatch"): "qualification_runtime",
}
RECOVERY_LIMITATIONS: Final[tuple[str, ...]] = (
    "the six historical task-authored tests are replay evidence, not independent completion oracles",
    "provider-import observation covers the pytest worker process only",
    "descendants inherit OS network denial but their Python import tables are not observed",
    "LGCVF-051 and LGCVF-060 alone admit one anonymous private 4096-byte rwxp libffi page inside their sealed ephemeral workers after pinned Z3 initialization; the controller, materializer, and the other four workers require zero writable-executable mappings",
    "each suite worker exits immediately after two-phase parent attestation, so the qualified libffi page cannot persist into controller or materialization authority",
    "solver ELF dependencies and the pinned CPython lib-dynload inventory are content-bound; transitive dependencies of root-owned standard-library extensions remain an explicit pinned-host platform boundary",
    "full suite-task live observations are parent-validated in both attestation phases but are not persisted or reconstructed by later direct verification; compact receipts retain their exact CIDs and config-pinned semantic commitments",
    "LGCVF-061 deliberately makes the projected Z3 Python namespace policy-unavailable in the sealed main suite worker, so its live-Z3 CEGAR path is not exercised; the exact MetaPathFinder denial and candidate-phase audit open boundary assume trusted tracked source; direct transient sys.modules mutation, custom in-memory loaders, and C-native import/file bypasses are outside that boundary; after candidate quiescence exactly one owner-thread nonreentrant tracked-source directory-component open is admitted by a source-bound pending capability, not by any global relative-z3 exemption; Python's open audit tuple omits dir_fd, so its root binding relies on the exact pending logical path, trusted reader code identity, and post-open descriptor validation; the separate owner-thread runtime reread remains scoped, and no process-tree namespace denial is claimed",
    "tracked recovery tests may invoke source-bound external subprocess tools whose executable internals are outside the in-process native-module closure",
    "current-source, merge, semantic, and control-plane evidence must be cross-checked by recovery admission",
    "this receipt grants no task, objective, release, or production authority",
)
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
_MAX_WORKER_TRANSCRIPT_BYTES: Final[int] = 4 * 1024 * 1024
_MAX_WORKER_RECEIPT_BYTES: Final[int] = 32 * 1024
_MAX_WORKER_ATTESTATION_BYTES: Final[int] = 64 * 1024
_MAX_RECOVERY_EXECUTABLE_MAPPINGS: Final[int] = 4_096
_MAX_RECOVERY_PROC_ENTRIES: Final[int] = 65_536
_MAX_WORKER_ACK_SECONDS: Final[int] = 60
_MAX_PYTEST_METADATA_BYTES: Final[int] = 1024 * 1024
_MAX_PYTEST_RECORD_BYTES: Final[int] = 16 * 1024 * 1024
_MAX_PYTEST_DISTRIBUTION_CANDIDATES: Final[int] = 32
_PYTEST_EXTERNAL_RECORD_PATHS: Final[frozenset[str]] = frozenset(
    {"../../../bin/py.test", "../../../bin/pytest"}
)
_MAX_DUCKDB_DISTRIBUTION_CANDIDATES: Final[int] = 8
_MAX_DUCKDB_SITE_ROOT_ENTRIES: Final[int] = 4_096
_MAX_DUCKDB_RUNTIME_FILE_BYTES: Final[int] = 64 * 1024 * 1024
_DUCKDB_RUNTIME_SOURCE_PATHS: Final[tuple[str, ...]] = (
    "duckdb/__init__.py",
    "duckdb/_dbapi_type_object.py",
    "duckdb/_version.py",
    "duckdb/sqltypes/__init__.py",
    "duckdb/value/__init__.py",
    "duckdb/value/constant/__init__.py",
)
_DUCKDB_RUNTIME_MODULE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "_duckdb",
        "_duckdb._func",
        "_duckdb._sqltypes",
        "duckdb",
        "duckdb._dbapi_type_object",
        "duckdb._version",
        "duckdb.sqltypes",
        "duckdb.value",
        "duckdb.value.constant",
    }
)
_ACTIVE_DUCKDB_RUNTIME_CID: str | None = None
_ACTIVE_DUCKDB_RUNTIME_EVIDENCE: dict[str, Any] | None = None
_ACTIVE_DUCKDB_RUNTIME_DIRECTORY: tempfile.TemporaryDirectory[str] | None = None
_ACTIVE_DUCKDB_RUNTIME_PROJECTION: Path | None = None
_ACTIVE_DUCKDB_RUNTIME_MODULES: dict[str, Any] | None = None

_QUALIFICATION_RUNTIME_COMPONENT_SCHEMA: Final[str] = (
    "lgcvf-qualification-runtime-component@1"
)
_QUALIFICATION_RUNTIME_BUNDLE_SCHEMA: Final[str] = (
    "lgcvf-qualification-runtime-bundle@1"
)
_QUALIFICATION_RUNTIME_PROJECTION_SCHEMA: Final[str] = (
    "lgcvf-qualification-runtime-projection@1"
)
_QUALIFICATION_RUNTIME_BOOTSTRAP_SCHEMA: Final[str] = (
    "lgcvf-qualification-runtime-bootstrap@1"
)
_RECOVERY_WORKER_PYCACHE_SCHEMA: Final[str] = (
    "lgcvf-recovery-worker-pycache@2"
)
_RECOVERY_CONFIG_RELATIVE_PATH: Final[str] = (
    "config/agent_supervisor_logic_governed_compositional_verification_fabric_scheduler.json"
)
_MAX_QUALIFICATION_RUNTIME_SITE_ENTRIES: Final[int] = 4_096
_MAX_QUALIFICATION_RUNTIME_DISTRIBUTIONS: Final[int] = 32
_MAX_QUALIFICATION_RUNTIME_RECORD_BYTES: Final[int] = 16 * 1024 * 1024
_MAX_QUALIFICATION_RUNTIME_FILE_BYTES: Final[int] = 64 * 1024 * 1024
_MAX_QUALIFICATION_RUNTIME_PAYLOAD_FILES: Final[int] = 619
_QUALIFICATION_RUNTIME_PAYLOAD_BYTES: Final[int] = 87_243_248
_RECOVERY_Z3_REQUIRED_TASKS: Final[frozenset[str]] = frozenset(
    {"LGCVF-051", "LGCVF-060"}
)
_RECOVERY_Z3_IMPORT_DENIED_TASKS: Final[frozenset[str]] = frozenset(
    {"LGCVF-061"}
)
_QUALIFICATION_RUNTIME_PROTECTED_MODULE_ROOTS: Final[frozenset[str]] = frozenset(
    {
        "pytest",
        "_pytest",
        "py",
        "pluggy",
        "iniconfig",
        "packaging",
        "pygments",
        "multiformats",
        "multiformats_config",
        "typing_validation",
        "typing_extensions",
        "bases",
        "cvc5",
        "z3",
    }
)
_QUALIFICATION_RUNTIME_EXTERNAL_RECORD_PATHS: Final[
    dict[str, frozenset[str]]
] = {
    "pytest": frozenset({"../../../bin/py.test", "../../../bin/pytest"}),
    "z3-solver": frozenset({"../../../bin/z3"}),
}
_QUALIFICATION_RUNTIME_REQUIRES_PYTHON: Final[dict[str, str]] = {
    "pytest": ">=3.10",
    "pluggy": ">=3.9",
    "iniconfig": ">=3.10",
    "packaging": ">=3.8",
    "pygments": ">=3.7",
    "multiformats": ">=3.7",
    "multiformats-config": ">=3.7",
    "typing-validation": ">=3.7",
    "typing-extensions": ">=3.9",
    "bases": ">=3.7",
    "cvc5": "",
    "z3-solver": "",
}


@dataclass(frozen=True)
class _QualificationRuntimeComponentSpec:
    ordinal: int
    role: str
    normalized_name: str
    distribution_directory: str
    version: str
    allowed_roots: tuple[str, ...]
    package_entry: str
    active_dependencies: tuple[str, ...]
    expected_file_count: int
    expected_total_bytes: int
    provenance_kind: str = "wheel_record"


_QUALIFICATION_RUNTIME_COMPONENTS: Final[
    tuple[_QualificationRuntimeComponentSpec, ...]
] = (
    _QualificationRuntimeComponentSpec(
        1,
        "runner",
        "pytest",
        "pytest-9.1.1.dist-info",
        "9.1.1",
        ("pytest", "_pytest", "py.py", "pytest-9.1.1.dist-info"),
        "pytest/__init__.py",
        ("iniconfig>=1.0.1", "packaging>=22", "pluggy<2,>=1.5", "pygments>=2.7.2"),
        90,
        1_367_928,
    ),
    _QualificationRuntimeComponentSpec(
        2, "runner", "pluggy", "pluggy-1.6.0.dist-info", "1.6.0",
        ("pluggy", "pluggy-1.6.0.dist-info"), "pluggy/__init__.py", (), 15, 66_302,
    ),
    _QualificationRuntimeComponentSpec(
        3, "runner", "iniconfig", "iniconfig-2.3.0.dist-info", "2.3.0",
        ("iniconfig", "iniconfig-2.3.0.dist-info"), "iniconfig/__init__.py", (), 11, 18_639,
    ),
    _QualificationRuntimeComponentSpec(
        4, "runner", "packaging", "packaging-26.2.dist-info", "26.2",
        ("packaging", "packaging-26.2.dist-info"), "packaging/__init__.py", (), 28, 383_121,
    ),
    _QualificationRuntimeComponentSpec(
        5, "runner", "pygments", "pygments-2.17.2.dist-info", "2.17.2",
        ("pygments", "pygments-2.17.2.dist-info"), "pygments/__init__.py", (), 325, 4_286_552,
        "debian_dpkg_md5sums",
    ),
    _QualificationRuntimeComponentSpec(
        6, "semantic", "multiformats", "multiformats-0.3.1.post4.dist-info", "0.3.1.post4",
        ("multiformats", "multiformats-0.3.1.post4.dist-info"), "multiformats/__init__.py",
        ("typing-extensions>=4.6.0", "typing-validation>=1.1.0", "bases>=0.3.0", "multiformats-config>=0.3.0"),
        31, 208_479,
    ),
    _QualificationRuntimeComponentSpec(
        7, "semantic", "multiformats-config", "multiformats_config-0.3.1.dist-info", "0.3.1",
        ("multiformats_config", "multiformats_config-0.3.1.dist-info"),
        "multiformats_config/__init__.py", ("multiformats",), 13, 114_530,
    ),
    _QualificationRuntimeComponentSpec(
        8, "semantic", "typing-validation", "typing_validation-1.2.12.dist-info", "1.2.12",
        ("typing_validation", "typing_validation-1.2.12.dist-info"),
        "typing_validation/__init__.py", (), 11, 84_391,
    ),
    _QualificationRuntimeComponentSpec(
        9, "semantic", "typing-extensions", "typing_extensions-4.16.0.dist-info", "4.16.0",
        ("typing_extensions.py", "typing_extensions-4.16.0.dist-info"),
        "typing_extensions.py", (), 6, 182_924,
    ),
    _QualificationRuntimeComponentSpec(
        10, "semantic", "bases", "bases-0.3.0.dist-info", "0.3.0",
        ("bases", "bases-0.3.0.dist-info"), "bases/__init__.py",
        ("typing-extensions>=4.6.0", "typing-validation>=1.1.0"), 20, 136_201,
    ),
    _QualificationRuntimeComponentSpec(
        11, "solver", "cvc5", "cvc5-1.3.3.dist-info", "1.3.3",
        ("cvc5", "cvc5.libs", "cvc5-1.3.3.dist-info"), "cvc5/__init__.py", (),
        23, 39_518_407,
    ),
    _QualificationRuntimeComponentSpec(
        12, "solver", "z3-solver", "z3_solver-4.15.4.0.dist-info", "4.15.4.0",
        ("z3", "z3_solver-4.15.4.0.dist-info"), "z3/__init__.py", (),
        46, 40_875_774,
    ),
)


@dataclass
class _ResolvedQualificationRuntime:
    bundle: dict[str, Any]
    components: tuple[dict[str, Any], ...]
    payload_manifest: tuple[dict[str, Any], ...]
    payload_bytes: dict[str, bytes]
    native_source_observation: dict[str, Any]


@dataclass
class _ActiveQualificationRuntime:
    resolved: _ResolvedQualificationRuntime
    directory: tempfile.TemporaryDirectory[str]
    root: Path
    root_fd: int
    projection: dict[str, Any]
    control_manifest_path: str


_ACTIVE_QUALIFICATION_RUNTIME: _ActiveQualificationRuntime | None = None
_ACTIVE_QUALIFICATION_RUNTIME_DEPTH = 0
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
_SECCOMP_POLICY_SCHEMA: Final[str] = "lgcvf-recovery-seccomp-policy@2"
_DENIED_NETWORK_SYSCALLS: Final[tuple[str, ...]] = (
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
)
_CRITICAL_CONTAINMENT_SYSCALLS: Final[tuple[str, ...]] = (
    "setsid",
    "setpgid",
    "unshare",
    "setns",
)
_GIT_HELPER_METADATA_SYSCALLS: Final[tuple[str, ...]] = (
    "chmod",
    "fchmod",
    "fchmodat",
    "fchmodat2",
    "utime",
    "utimes",
    "futimesat",
    "utimensat",
)
_DENIED_SYSCALLS: Final[tuple[str, ...]] = (
    *_DENIED_NETWORK_SYSCALLS,
    # Landlock deliberately does not mediate these metadata operations.  A
    # candidate must not alter even modes, ownership, timestamps, or xattrs of
    # a protected checkout while its tests are being judged.
    *_GIT_HELPER_METADATA_SYSCALLS,
    "chown",
    "fchown",
    "lchown",
    "fchownat",
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


@dataclass(frozen=True)
class RecoveryValidation:
    """One closed historical validation replay; never a completion oracle."""

    task_id: str
    task_cid: str
    owner_root: str
    path: str

    @property
    def suite(self) -> Suite:
        return Suite(
            "recovery_" + self.task_id.casefold().replace("-", "_"),
            self.owner_root,
            (self.path,),
            True,
        )

    def validation_spec(self) -> dict[str, Any]:
        declared = f"python -m pytest -q {self.path}"
        if self.owner_root == "ipfs_datasets_py":
            working_directory = "ipfs_datasets_py"
            python_path = ".:.."
            historical = (
                "cd ipfs_datasets_py && export PYTHONPATH=.:.. && " + declared
            )
        elif self.owner_root == ".":
            working_directory = "."
            python_path = "ipfs_datasets_py"
            historical = "export PYTHONPATH=ipfs_datasets_py && " + declared
        else:  # pragma: no cover - constants below are the closed authority.
            raise QualificationError(
                f"{self.task_id}: unsupported recovery repository owner"
            )
        spec: dict[str, Any] = {
            "task_id": self.task_id,
            "task_cid": self.task_cid,
            "declared_command": declared,
            "historical_command": historical,
            "argv": ["python", "-m", "pytest", "-q", self.path],
            "working_directory": working_directory,
            "python_path": python_path,
            "timeout_seconds": 900,
            "provider_route": "none",
            "network_client_required": False,
        }
        spec["validation_spec_cid"] = content_identity(spec)
        return spec


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
RECOVERY_VALIDATIONS: Final[tuple[RecoveryValidation, ...]] = (
    RecoveryValidation(
        "LGCVF-051",
        "baguqeera5akr2w56xcy6mus4td2ghwaita5k52lnqkh54tgooqq35fai72nq",
        "ipfs_datasets_py",
        "tests/unit/logic/test_compositional_verification_public_api.py",
    ),
    RecoveryValidation(
        "LGCVF-060",
        "baguqeeratlqqozbenktvzhzzk36ewsti3mzegge2ynft24s3mznewzgswfoq",
        "ipfs_datasets_py",
        "tests/unit/logic/backends/test_interpolation.py",
    ),
    RecoveryValidation(
        "LGCVF-061",
        "baguqeeraqopotj43fgxcfptvziv3g3kna4wcwhtzarcobv2obvs32njoggjq",
        "ipfs_datasets_py",
        "tests/unit/logic/software_verification/test_cegar.py",
    ),
    RecoveryValidation(
        "LGCVF-070",
        "baguqeeraej2zz7zlrd2l5p6adjnzinnitzuqmxx4agfhfzekixdqm2mnqyda",
        "ipfs_datasets_py",
        "tests/unit/logic/formalization/test_translation_receipts.py",
    ),
    RecoveryValidation(
        "LGCVF-071",
        "baguqeerar3vmbqw7f2qk6mjyhsx3hq7gpbnqcydt7cecm6og5xejbd2vz6cq",
        "ipfs_datasets_py",
        "tests/unit/logic/software_verification/test_obligation_slicing.py",
    ),
    RecoveryValidation(
        "LGCVF-080",
        "baguqeera22uu4o4ux6kzp4fgv5gxqupas3nhdjtbrtt73x2kc6mhtxkdbtwq",
        ".",
        "test/api/test_agent_supervisor_program_repair_egraph.py",
    ),
)
_RECOVERY_BY_SUITE_ID: Final[dict[str, RecoveryValidation]] = {
    item.suite.suite_id: item for item in RECOVERY_VALIDATIONS
}
_WORKER_SUITES: Final[tuple[Suite, ...]] = (
    *SUITES,
    SANDBOX_SMOKE_SUITE,
    *(item.suite for item in RECOVERY_VALIDATIONS),
)

# This is a protected-judge policy, not a provider router.  It mirrors the
# external model SDK roots denied by the deterministic runtime and adds the
# known internal invocation surfaces whose import would invalidate a zero-model
# recovery observation.
_FORBIDDEN_RECOVERY_PROVIDER_IMPORTS: Final[tuple[str, ...]] = (
    "anthropic",
    "azure.ai.inference",
    "azure.ai.openai",
    "cohere",
    "google.genai",
    "google.generativeai",
    "groq",
    "huggingface_hub",
    "langchain",
    "langchain_openai",
    "litellm",
    "mistralai",
    "ollama",
    "openai",
    "transformers",
    "vertexai",
    "vllm",
    "ipfs_accelerate_py.llm",
    "ipfs_accelerate_py.llm_router",
    "ipfs_accelerate_py.backends",
    "ipfs_accelerate_py.api_backends",
    "ipfs_accelerate_py.common.meta_model_api",
    "ipfs_accelerate_py.agent_supervisor.todo_daemon.llm",
    "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_provider_auto",
    "ipfs_accelerate_py.agent_supervisor.todo_daemon.production_provider_cli",
    "ipfs_accelerate_py.agent_supervisor.todo_daemon.residual_provider_invocation",
    "ipfs_accelerate_py.agent_supervisor.provider_fallback_runner",
    "ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner",
    "ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_embedding_provider",
    "ipfs_accelerate_py.agent_supervisor.integrations.llm_merge_resolver_fallback",
)
_FORBIDDEN_RECOVERY_PROVIDER_EXECUTABLES: Final[tuple[str, ...]] = (
    "anthropic",
    "claude",
    "codex",
    "gemini",
    "grok",
    "ollama",
    "openai",
)


class QualificationError(RuntimeError):
    """Qualification input, execution, or reconstruction failed closed."""


class QualificationRuntimeUnavailable(QualificationError):
    """One classified dependency-runtime failure safe to report as unavailable."""

    def __init__(
        self,
        *,
        reason_code: str,
        phase: str,
        expected_runtime_cid: str,
        observed_runtime_cid: str = "",
        component: str = "",
        detail: str,
    ) -> None:
        expected_component = _RECOVERY_UNAVAILABLE_CLASSIFICATIONS.get(
            (phase, reason_code)
        )
        if expected_component is None or component != expected_component:
            raise QualificationError(
                "qualification runtime unavailable classification differs"
            )
        super().__init__(detail)
        self.reason_code = reason_code
        self.phase = phase
        self.expected_runtime_cid = expected_runtime_cid
        self.observed_runtime_cid = observed_runtime_cid
        self.component = component
        self.detail = detail


def _strict_json_loads(value: str, *, noun: str) -> Any:
    """Decode JSON while rejecting duplicate object keys at every depth."""

    def closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise QualificationError(
                    f"{noun} contains duplicate JSON object key: {key}"
                )
            result[key] = item
        return result

    return json.loads(value, object_pairs_hook=closed_object)


def _is_canonical_content_cid(value: object) -> bool:
    """Recognize the exact CIDv1/base32/raw/sha2-256 identity encoding."""

    if not isinstance(value, str) or not re.fullmatch(r"b[a-z2-7]{60}", value):
        return False
    encoded = value[1:]
    try:
        raw = base64.b32decode(encoded.upper() + "====", casefold=False)
    except (binascii.Error, ValueError):
        return False
    return (
        len(raw) == 37
        and raw[:5] == b"\x01\xa9\x02\x12\x20"
        and base64.b32encode(raw).decode("ascii").rstrip("=").lower()
        == encoded
    )


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


def _install_landlock(write_root: Path, *, permit_git_helpers: bool = False) -> int:
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
        # git and subprocess helpers open /dev/null O_RDWR.  Device inodes are
        # not valid PATH_BENEATH parents, so allow WRITE_FILE under /dev
        # without create/remove rights.  Only fixed suites that invoke git
        # receive this exception; candidate suites keep the tighter policy.
        if permit_git_helpers and Path("/dev").is_dir():
            extra_fd = os.open("/dev", os.O_PATH | os.O_DIRECTORY | os.O_CLOEXEC)
            extra_fds.append(extra_fd)
            extra_attr = _LandlockPathBeneathAttr(
                _LANDLOCK_ACCESS_FS_WRITE_FILE | _LANDLOCK_ACCESS_FS_TRUNCATE,
                extra_fd,
            )
            if (
                int(
                    libc.syscall(
                        _syscall_number("landlock_add_rule"),
                        ruleset_fd,
                        _LANDLOCK_RULE_PATH_BENEATH,
                        ctypes.byref(extra_attr),
                        ctypes.c_uint(0),
                    )
                )
                != 0
            ):
                _raise_errno("path rule installation for /dev")
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
        for extra_fd in extra_fds:
            os.close(extra_fd)
        if parent_fd >= 0:
            os.close(parent_fd)
        os.close(ruleset_fd)
    return abi


def _resolve_seccomp_rules(
    resolver: Callable[[bytes], int],
) -> tuple[tuple[str, int], ...]:
    """Resolve the closed deny policy and reject any missing network rule."""

    rules: list[tuple[str, int]] = []
    unresolved_required: list[str] = []
    required = set(_DENIED_NETWORK_SYSCALLS) | set(
        _CRITICAL_CONTAINMENT_SYSCALLS
    )
    for name in _DENIED_SYSCALLS:
        syscall_number = int(resolver(name.encode("ascii")))
        if syscall_number < 0:
            if name in required:
                unresolved_required.append(name)
            continue
        rules.append((name, syscall_number))
    if unresolved_required:
        raise QualificationError(
            "candidate sandbox cannot resolve required containment syscalls: "
            + ", ".join(unresolved_required)
        )
    resolved_names = {name for name, _number in rules}
    if not required.issubset(resolved_names):
        raise QualificationError("candidate sandbox required deny policy is incomplete")
    return tuple(rules)


def _seccomp_policy_evidence(installed: Sequence[str]) -> dict[str, Any]:
    """Bind the exact requested, installed, and unavailable seccomp rules."""

    installed_names = tuple(map(str, installed))
    if (
        len(installed_names) != len(set(installed_names))
        or any(name not in _DENIED_SYSCALLS for name in installed_names)
        or installed_names
        != tuple(name for name in _DENIED_SYSCALLS if name in set(installed_names))
        or not set(_DENIED_NETWORK_SYSCALLS).issubset(installed_names)
        or not set(_CRITICAL_CONTAINMENT_SYSCALLS).issubset(installed_names)
    ):
        raise QualificationError("candidate sandbox installed seccomp policy differs")
    evidence: dict[str, Any] = {
        "schema": _SECCOMP_POLICY_SCHEMA,
        "default_action": "allow",
        "deny_action": "errno:EPERM",
        "required_network_syscalls": list(_DENIED_NETWORK_SYSCALLS),
        "required_containment_syscalls": list(
            _CRITICAL_CONTAINMENT_SYSCALLS
        ),
        "requested_syscalls": list(_DENIED_SYSCALLS),
        "installed_syscalls": list(installed_names),
        "unavailable_optional_syscalls": [
            name for name in _DENIED_SYSCALLS if name not in installed_names
        ],
    }
    evidence["policy_cid"] = content_identity(evidence)
    return evidence


def _install_seccomp(*, permit_git_helpers: bool = False) -> tuple[str, ...]:
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
    installed: list[str] = []
    try:
        action = _SCMP_ACT_ERRNO | errno.EPERM
        rules = _resolve_seccomp_rules(library.seccomp_syscall_resolve_name)
        skipped = set(_GIT_HELPER_METADATA_SYSCALLS) if permit_git_helpers else set()
        for name, syscall_number in rules:
            if name in skipped:
                continue
            if library.seccomp_rule_add(context, action, syscall_number, 0) != 0:
                raise QualificationError(
                    f"candidate sandbox could not deny syscall {name}"
                )
            installed.append(name)
        if not set(_DENIED_NETWORK_SYSCALLS).issubset(installed):
            raise QualificationError("candidate sandbox network deny policy is incomplete")
        if not set(_CRITICAL_CONTAINMENT_SYSCALLS).issubset(installed):
            raise QualificationError(
                "candidate sandbox containment deny policy is incomplete"
            )
        if library.seccomp_load(context) != 0:
            raise QualificationError("candidate sandbox seccomp load failed")
    finally:
        library.seccomp_release(context)
    return tuple(installed)


def _lower_resource_limit(kind: int, value: int) -> int:
    """Irreversibly lower one process resource bound and return the bound."""

    _soft, hard = resource.getrlimit(kind)
    bounded = value if hard == resource.RLIM_INFINITY else min(value, int(hard))
    resource.setrlimit(kind, (bounded, bounded))
    return bounded


def _install_candidate_sandbox(
    write_root: Path,
    *,
    bind_seccomp_policy: bool = False,
    permit_git_helpers: bool = False,
) -> dict[str, Any]:
    """Install irreversible filesystem and network restrictions in a worker."""

    # Resolve and load libseccomp before lowering the process limit because
    # libc discovery may itself use one bounded helper process.
    denied_syscalls = _install_seccomp(permit_git_helpers=permit_git_helpers)
    # Bound damage from a malicious or accidentally explosive candidate test.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    file_size_bytes = _lower_resource_limit(resource.RLIMIT_FSIZE, 64 * 1024 * 1024)
    open_files = _lower_resource_limit(resource.RLIMIT_NOFILE, 256)
    # RLIMIT_NPROC is per real UID (and counts threads), not per worker.  The
    # development host legitimately runs more than 4,096 same-UID threads, so a
    # 4,096 ceiling prevents even git/pytest helper forks.  Keep a finite
    # ceiling and separately pin numerical libraries to one thread in the
    # sealed worker environment.
    processes = _lower_resource_limit(resource.RLIMIT_NPROC, 65_536)
    cpu_seconds = _lower_resource_limit(resource.RLIMIT_CPU, 900)
    address_space_bytes = _lower_resource_limit(resource.RLIMIT_AS, 8 * 1024**3)
    landlock_abi = _install_landlock(
        write_root, permit_git_helpers=permit_git_helpers
    )
    result = {
        "profile": "landlock-readonly-seccomp-no-network",
        "landlock_abi": landlock_abi,
        "seccomp_denied_syscall_count": len(denied_syscalls),
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
    if bind_seccomp_policy:
        result["seccomp_policy"] = _seccomp_policy_evidence(denied_syscalls)
    return result


def _sandbox_evidence_is_valid(
    value: Any,
    *,
    require_recovery_policy: bool = False,
) -> bool:
    expected_fields = {
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
    }
    if require_recovery_policy:
        expected_fields.add("seccomp_policy")
        expected_fields.add("worker_pycache")
    if not isinstance(value, Mapping) or set(value) != expected_fields:
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
        "processes": 65_536,
    }
    if any(
        isinstance(limits.get(name), bool)
        or not isinstance(limits.get(name), int)
        or not 0 < int(limits[name]) <= maximum
        for name, maximum in maximums.items()
    ):
        return False
    policy_valid = True
    if require_recovery_policy:
        policy = value.get("seccomp_policy")
        if not isinstance(policy, Mapping) or set(policy) != {
            "schema",
            "default_action",
            "deny_action",
            "required_network_syscalls",
            "required_containment_syscalls",
            "requested_syscalls",
            "installed_syscalls",
            "unavailable_optional_syscalls",
            "policy_cid",
        }:
            return False
        installed = policy.get("installed_syscalls")
        if not isinstance(installed, list) or any(
            not isinstance(name, str) for name in installed
        ):
            return False
        try:
            expected_policy = _seccomp_policy_evidence(installed)
        except QualificationError:
            return False
        policy_valid = (
            dict(policy) == expected_policy
            and policy.get("required_network_syscalls")
            == list(_DENIED_NETWORK_SYSCALLS)
            and policy.get("required_containment_syscalls")
            == list(_CRITICAL_CONTAINMENT_SYSCALLS)
            and policy.get("requested_syscalls") == list(_DENIED_SYSCALLS)
            and value.get("seccomp_denied_syscall_count") == len(installed)
        )
        worker_pycache = value.get("worker_pycache")
        if not isinstance(worker_pycache, Mapping) or set(worker_pycache) != {
            "schema",
            "write_root_relative_path",
            "activation",
            "python_prefix_active",
            "dont_write_bytecode",
            "owner_matches_worker",
            "mode_octal",
            "root_identity",
            "empty_before",
            "empty_after",
        }:
            return False
        pycache_identity = worker_pycache.get("root_identity")
        if (
            worker_pycache.get("schema") != _RECOVERY_WORKER_PYCACHE_SCHEMA
            or worker_pycache.get("write_root_relative_path")
            != "<temporary:python-pycache-*>"
            or worker_pycache.get("activation") != "shared_bootstrap_capsule"
            or worker_pycache.get("python_prefix_active") is not True
            or worker_pycache.get("dont_write_bytecode") is not True
            or worker_pycache.get("owner_matches_worker") is not True
            or worker_pycache.get("mode_octal") != "0700"
            or worker_pycache.get("empty_before") is not True
            or worker_pycache.get("empty_after") is not True
            or not isinstance(pycache_identity, Mapping)
            or set(pycache_identity)
            != {"dev", "ino", "uid", "gid", "mode", "nlink"}
            or any(
                isinstance(pycache_identity.get(name), bool)
                or not isinstance(pycache_identity.get(name), int)
                for name in pycache_identity
            )
            or pycache_identity.get("uid") != os.geteuid()
            or pycache_identity.get("mode") != 0o700
        ):
            return False
    return (
        value.get("profile") == "landlock-readonly-seccomp-no-network"
        and isinstance(value.get("landlock_abi"), int)
        and not isinstance(value.get("landlock_abi"), bool)
        and int(value["landlock_abi"]) >= 4
        and isinstance(value.get("seccomp_denied_syscall_count"), int)
        and not isinstance(value.get("seccomp_denied_syscall_count"), bool)
        and int(value["seccomp_denied_syscall_count"])
        >= len(
            set(_DENIED_NETWORK_SYSCALLS)
            | set(_CRITICAL_CONTAINMENT_SYSCALLS)
        )
        and value.get("checkout_write_permitted") is False
        and value.get("network_permitted") is False
        and value.get("completion_authoritative") is False
        and value.get("process_group_escape_permitted") is False
        and value.get("null_sink_redirected") is True
        and value.get("pytest_log_sink_redirected") is True
        and policy_valid
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
        if entry.is_symlink():
            if descendants:
                raise QualificationError(
                    f"qualification selected a symlink input: {relative}"
                )
            continue
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


def _projection_omitted_symlinks(
    source: Path,
    *,
    prefix: PurePath,
    copied_paths: frozenset[str],
) -> list[dict[str, str]]:
    """Return exact source symlinks omitted along projected branches."""

    omitted: list[dict[str, str]] = []
    for entry in sorted(source.iterdir(), key=lambda item: item.name):
        relative = (prefix / entry.name).as_posix()
        descendants = [
            item
            for item in copied_paths
            if item == relative or item.startswith(relative + "/")
        ]
        if entry.is_symlink():
            if descendants:
                raise QualificationError(
                    f"qualification selected a symlink input: {relative}"
                )
            omitted.append(
                {
                    "path": relative,
                    "git_target": os.readlink(entry),
                    "disposition": "omitted_source_symlink",
                }
            )
        elif descendants and entry.is_dir():
            omitted.extend(
                _projection_omitted_symlinks(
                    entry,
                    prefix=prefix / entry.name,
                    copied_paths=copied_paths,
                )
            )
    return omitted


def _execution_projection_receipt(
    root: Path,
    suites: Sequence[Suite],
) -> dict[str, Any]:
    """Build the closed read-only projection receipt without writing it."""

    resolved_root = root.resolve(strict=True)
    copied_paths = frozenset(
        (Path(suite.owner_root) / relative).as_posix().lstrip("./")
        for suite in suites
        for relative in suite.paths
    )
    if not copied_paths:
        raise QualificationError("qualification projection has no judged inputs")
    copied = []
    for relative in sorted(copied_paths):
        source_bytes = _safe_source(resolved_root, relative).read_bytes()
        copied.append(
            {
                "path": relative,
                "sha256": _sha256_bytes(source_bytes),
                "size_bytes": len(source_bytes),
            }
        )
    result: dict[str, Any] = {
        "schema": "lgcvf-readonly-test-projection@2",
        "copied_sources": copied,
        "omitted_source_symlinks": _projection_omitted_symlinks(
            resolved_root,
            prefix=PurePath(),
            copied_paths=copied_paths,
        ),
        "original_checkout_writable": False,
    }
    result["projection_cid"] = content_identity(result)
    return result


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
    expected = _execution_projection_receipt(resolved_root, suites)
    _copy_projection_branch(
        resolved_root,
        destination,
        prefix=Path(),
        copied_paths=copied_paths,
    )
    for item in expected["copied_sources"]:
        relative = str(item["path"])
        source = _safe_source(resolved_root, relative)
        projected_path = destination / relative
        projected = _safe_source(destination.resolve(strict=True), relative)
        source_bytes = source.read_bytes()
        projected_bytes = projected.read_bytes()
        if projected_path.is_symlink() or source_bytes != projected_bytes:
            raise QualificationError(
                f"qualification copied input differs: {relative}"
            )
    for item in expected["omitted_source_symlinks"]:
        try:
            (destination / str(item["path"])).lstat()
        except FileNotFoundError:
            continue
        raise QualificationError(
            f"qualification projection retained a source symlink: {item['path']}"
        )
    return expected


def _recovery_projection_source_paths(
    root: Path,
    suites: Sequence[Suite],
) -> tuple[str, ...]:
    """Select the closed Python/native/config dependency projection."""

    selected = {
        (Path(suite.owner_root) / relative).as_posix().lstrip("./")
        for suite in suites
        for relative in suite.paths
    }
    source_roots = (
        root / "ipfs_accelerate_py",
        root / "scripts",
        root / "ipfs_datasets_py/ipfs_datasets_py",
    )
    for source_root in source_roots:
        if not source_root.is_dir() or source_root.is_symlink():
            raise QualificationError("recovery projection source root differs")
        for candidate in source_root.rglob("*"):
            if candidate.is_symlink() or not candidate.is_file():
                continue
            if candidate.name.endswith(".py") or candidate.name.endswith(".so") or candidate.name.endswith(
                tuple(importlib.machinery.EXTENSION_SUFFIXES)
            ):
                selected.add(candidate.relative_to(root).as_posix())
    for candidate in root.iterdir():
        if (
            not candidate.is_symlink()
            and candidate.is_file()
            and (
                candidate.name.endswith(".py")
                or candidate.name.endswith(".so")
                or candidate.name.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES))
            )
        ):
            selected.add(candidate.relative_to(root).as_posix())
    # The accelerator's root conftest imports these shared fixtures at
    # collection time.  Copy the closed helper package instead of exposing
    # the rest of ``test/`` (which contains unrelated environments, reports,
    # and external symlinks) to the recovery worker.
    common_helpers = root / "test/common"
    if not common_helpers.is_dir() or common_helpers.is_symlink():
        raise QualificationError("recovery test helper root differs")
    for candidate in common_helpers.iterdir():
        if candidate.is_symlink() or not candidate.is_file():
            continue
        if candidate.name.endswith(".py"):
            selected.add(candidate.relative_to(root).as_posix())
    for suite in suites:
        selected_path = Path(suite.owner_root) / suite.paths[0]
        parent = selected_path.parent
        while parent != Path("."):
            for support_name in ("conftest.py", "__init__.py"):
                support = root / parent / support_name
                if support.is_file() and not support.is_symlink():
                    selected.add(support.relative_to(root).as_posix())
            parent = parent.parent
    for relative in (
        _RECOVERY_CONFIG_RELATIVE_PATH,
        "conftest.py",
        "test/__init__.py",
        "pytest.ini",
        "pyproject.toml",
        "ipfs_datasets_py/conftest.py",
        "ipfs_datasets_py/pytest.ini",
        "ipfs_datasets_py/pyproject.toml",
    ):
        candidate = root / relative
        if candidate.is_file() and not candidate.is_symlink():
            selected.add(relative)
    if not selected or len(selected) > 8_192:
        raise QualificationError("recovery projection population exceeds its bound")
    return tuple(sorted(selected))


def _recovery_projection_omitted_symlinks(root: Path) -> list[dict[str, str]]:
    """Inventory every source symlink excluded from recovery worker reach."""

    omitted: list[dict[str, str]] = []
    count = 0

    def visit(directory: Path, prefix: PurePath, depth: int) -> None:
        nonlocal count
        if depth > _MAX_RECOVERY_IMPORT_DEPTH:
            raise QualificationError("recovery projection source tree is too deep")
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.name in {".git", "__pycache__"}:
                    continue
                count += 1
                if count > 2 * _MAX_RECOVERY_IMPORT_ROOT_ENTRIES:
                    raise QualificationError("recovery projection source tree is too large")
                relative = prefix / entry.name
                status = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(status.st_mode):
                    omitted.append(
                        {
                            "path": relative.as_posix(),
                            "git_target": os.readlink(entry.path),
                            "disposition": "omitted_source_symlink",
                        }
                    )
                elif stat.S_ISDIR(status.st_mode):
                    visit(Path(entry.path), relative, depth + 1)

    for relative in ("scripts", "ipfs_accelerate_py", "test", "ipfs_datasets_py"):
        source = root / relative
        if not source.is_dir() or source.is_symlink():
            raise QualificationError("recovery projection source root differs")
        visit(source, PurePath(relative), 0)
    paths = [item["path"] for item in omitted]
    if len(paths) != len(set(paths)):
        raise QualificationError("recovery projection symlink inventory is ambiguous")
    return sorted(omitted, key=lambda item: item["path"])


def _recovery_projection_directory_flags() -> int:
    """Return the one exact no-follow directory-open flag set."""

    return (
        os.O_RDONLY
        | os.O_CLOEXEC
        | os.O_DIRECTORY
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _read_recovery_projection_source(
    root: Path,
    relative: str,
    *,
    git_entry: tuple[str, str] | None = None,
    required_git_mode: str | None = None,
    source_revalidation_guard: _RecoveryZ3ImportDenialGuard | None = None,
) -> bytes:
    """Read one bounded regular source through a no-follow descriptor chain."""

    logical = PurePath(relative)
    if (
        not relative
        or logical.is_absolute()
        or ".." in logical.parts
        or logical.as_posix() != relative
    ):
        raise QualificationError("recovery projection path differs")
    directory_flags = _recovery_projection_directory_flags()
    file_flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(root, directory_flags)
    except OSError as exc:
        raise QualificationError(
            "recovery projection source is unavailable"
        ) from exc
    directory_chain = [descriptor]
    directory_observations: list[tuple[int, ...]] = []
    stable_directory_fields = (
        "st_dev",
        "st_ino",
        "st_uid",
        "st_gid",
        "st_mode",
        "st_nlink",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    try:
        lexical_root = root.lstat()
        root_status = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(root_status.st_mode)
            or (root_status.st_dev, root_status.st_ino)
            != (lexical_root.st_dev, lexical_root.st_ino)
            or not _qualification_source_mode_is_safe(
                root_status,
                owner_uid=os.geteuid(),
            )
            or stat.S_IMODE(root_status.st_mode) & 0o500 != 0o500
        ):
            raise QualificationError("recovery projection root identity differs")
        directory_observations.append(
            tuple(getattr(root_status, key) for key in stable_directory_fields)
        )
        for component_index, component in enumerate(logical.parts[:-1]):
            guarded_open = False
            if source_revalidation_guard is not None:
                guarded_open = (
                    source_revalidation_guard.register_trusted_source_directory_open(
                        logical_path=relative,
                        component_index=component_index,
                        component=component,
                        directory_flags=directory_flags,
                    )
                )
            child = os.open(component, directory_flags, dir_fd=descriptor)
            directory_chain.append(child)
            child_status = os.fstat(child)
            if (
                not stat.S_ISDIR(child_status.st_mode)
                or not _qualification_source_mode_is_safe(
                    child_status,
                    owner_uid=os.geteuid(),
                )
                or stat.S_IMODE(child_status.st_mode) & 0o500 != 0o500
            ):
                raise QualificationError(
                    "recovery projection directory identity differs"
                )
            if guarded_open:
                source_revalidation_guard.confirm_trusted_source_directory_open()
            directory_observations.append(
                tuple(
                    getattr(child_status, key)
                    for key in stable_directory_fields
                )
            )
            descriptor = child
        source_fd = os.open(logical.name, file_flags, dir_fd=descriptor)
        try:
            before = os.fstat(source_fd)
            expected_mode = (
                "100755"
                if stat.S_IMODE(before.st_mode) & 0o111
                else "100644"
            )
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size < 0
                or before.st_size > 16 * 1024 * 1024
                or not _qualification_source_mode_is_safe(
                    before,
                    owner_uid=os.geteuid(),
                )
                or not stat.S_IMODE(before.st_mode) & 0o400
                or (git_entry is not None and git_entry[0] != expected_mode)
                or (
                    required_git_mode is not None
                    and required_git_mode != expected_mode
                )
            ):
                raise QualificationError(
                    "recovery projection source identity differs"
                )
            chunks: list[bytes] = []
            observed = 0
            while True:
                block = os.read(
                    source_fd,
                    min(1024 * 1024, before.st_size + 1 - observed),
                )
                if not block:
                    break
                chunks.append(block)
                observed += len(block)
                if observed > before.st_size:
                    raise QualificationError(
                        "recovery projection source grew while read"
                    )
            after = os.fstat(source_fd)
            payload = b"".join(chunks)
            if (
                observed != before.st_size
                or (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                )
                != (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                )
                or (
                    git_entry is not None
                    and not _git_blob_matches(payload, git_entry[1])
                )
                or any(
                    tuple(
                        getattr(os.fstat(directory_fd), key)
                        for key in stable_directory_fields
                    )
                    != observed_directory
                    for directory_fd, observed_directory in zip(
                        directory_chain,
                        directory_observations,
                        strict=True,
                    )
                )
            ):
                raise QualificationError(
                    "recovery projection source differs from index and HEAD"
                )
            return payload
        finally:
            os.close(source_fd)
    except OSError as exc:
        raise QualificationError(
            "recovery projection source is unavailable"
        ) from exc
    finally:
        for directory_fd in reversed(directory_chain):
            os.close(directory_fd)


def _recovery_qualification_policy(
    root: Path,
    *,
    head_bound: bool,
) -> tuple[str, dict[str, Any]]:
    """Read the sole expected runtime CID from exact canonical config bytes."""

    resolved = root.resolve(strict=True)
    git_entry: tuple[str, str] | None = None
    if head_bound:
        try:
            entries = _tracked_recovery_import_entries(
                resolved,
                pathspecs=(_RECOVERY_CONFIG_RELATIVE_PATH,),
            )
        except RuntimeError as exc:
            raise QualificationError(
                "recovery qualification configuration authority differs"
            ) from exc
        if set(entries) != {_RECOVERY_CONFIG_RELATIVE_PATH}:
            raise QualificationError(
                "recovery qualification configuration is not exactly tracked"
            )
        git_entry = entries[_RECOVERY_CONFIG_RELATIVE_PATH]
        if git_entry[0] != "100644":
            raise QualificationError(
                "recovery qualification configuration mode differs"
            )
    payload = _read_recovery_projection_source(
        resolved,
        _RECOVERY_CONFIG_RELATIVE_PATH,
        git_entry=git_entry,
        required_git_mode="100644",
    )
    try:
        decoded = _strict_json_loads(
            payload.decode("utf-8", errors="strict"),
            noun="recovery qualification configuration",
        )
    except UnicodeDecodeError as exc:
        raise QualificationError(
            "recovery qualification configuration is not UTF-8"
        ) from exc
    if not isinstance(decoded, dict):
        raise QualificationError(
            "recovery qualification configuration root differs"
        )
    policy = decoded.get("fresh_generation_recovery")
    if (
        not isinstance(policy, dict)
        or policy.get("schema")
        != (
            "ipfs_accelerate_py/agent-supervisor/"
            "lgcvf-fresh-generation-recovery-policy@3"
        )
        or set(policy) != {
            "schema",
            "source_generation",
            "source_runtime_root",
            "target_generation",
            "target_runtime_root",
            "duckdb_runtime_cid",
            "qualification_runtime_cid",
            "verification_python_executable",
            "verification_python_executable_sha256",
            "retained_revision_receipt_path",
            "retained_revision_receipt_sha256",
            "retained_revision_receipt_cid",
            "retained_successor_revision_cid",
            "retained_delta_cid",
            "retained_completion_binding_cid",
            "retained_protected_blocker_binding_cid",
            "wrong_default_incident_manifest_path",
            "wrong_default_incident_manifest_sha256",
            "wrong_default_incident_manifest_cid",
            "contaminated_coordination_projection_path",
            "contaminated_coordination_projection_sha256",
            "contaminated_coordination_projection_manifest_cid",
            "contaminated_coordination_rejected_record_set_cid",
            "rejected_contaminated_coordination_projection_root",
            "construction_completed_task_ids",
            "recovered_completed_task_ids",
            "rejected_synthetic_task_ids",
            "preserved_blocked_task_ids",
            "merge_completions",
        }
    ):
        raise QualificationError("recovery qualification policy fields differ")
    expected_runtime_cid = policy.get("qualification_runtime_cid")
    if not _is_canonical_content_cid(expected_runtime_cid):
        raise QualificationError("qualification runtime policy is absent")
    evidence: dict[str, Any] = {
        "schema": "lgcvf-qualification-runtime-policy-binding@1",
        "path": _RECOVERY_CONFIG_RELATIVE_PATH,
        "mode": git_entry[0] if git_entry is not None else "copied_100644",
        "head_blob_oid": git_entry[1] if git_entry is not None else "copied_projection",
        "sha256": _sha256_bytes(payload),
        "size_bytes": len(payload),
        "qualification_runtime_cid": expected_runtime_cid,
        "ordinary_index_equals_head": head_bound,
    }
    evidence["policy_binding_cid"] = content_identity(evidence)
    return expected_runtime_cid, evidence


def _recovery_projection_git_entries(
    root: Path,
    paths: Sequence[str],
) -> dict[str, tuple[str, str]]:
    """Bind every selected projection file to ordinary index and HEAD blobs."""

    grouped: dict[Path, list[tuple[str, str]]] = {root: []}
    nested_root = root / "ipfs_datasets_py"
    grouped[nested_root] = []
    prefix = "ipfs_datasets_py/"
    for relative in paths:
        if relative.startswith(prefix):
            grouped[nested_root].append((relative, relative[len(prefix) :]))
        else:
            grouped[root].append((relative, relative))
    result: dict[str, tuple[str, str]] = {}
    for repository, values in grouped.items():
        for offset in range(0, len(values), 256):
            batch = values[offset : offset + 256]
            try:
                entries = _tracked_recovery_import_entries(
                    repository,
                    pathspecs=tuple(item[1] for item in batch),
                )
            except RuntimeError as exc:
                raise QualificationError(
                    "recovery projection Git authority differs"
                ) from exc
            if set(entries) != {item[1] for item in batch}:
                raise QualificationError(
                    "recovery projection source is not exactly tracked"
                )
            for public_path, repository_path in batch:
                entry = entries[repository_path]
                if entry[0] not in {"100644", "100755"}:
                    raise QualificationError(
                        "recovery projection selected a non-file Git entry"
                    )
                result[public_path] = entry
    if set(result) != set(paths):
        raise QualificationError("recovery projection Git population differs")
    return result


def _recovery_projection_manifest(
    root: Path,
    paths: Sequence[str],
    *,
    head_bound: bool = True,
    source_revalidation_guard: _RecoveryZ3ImportDenialGuard | None = None,
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    """Read and content-address every closed recovery projection file."""

    manifest: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    total = 0
    git_entries = _recovery_projection_git_entries(root, paths) if head_bound else {}
    for relative in paths:
        logical = PurePath(relative)
        if logical.is_absolute() or ".." in logical.parts or logical.as_posix() != relative:
            raise QualificationError("recovery projection path differs")
        payload = _read_recovery_projection_source(
            root,
            relative,
            git_entry=git_entries.get(relative),
            source_revalidation_guard=source_revalidation_guard,
        )
        total += len(payload)
        if total > 256 * 1024 * 1024:
            raise QualificationError("recovery projection bytes exceed their bound")
        payloads[relative] = payload
        manifest.append(
            {
                "path": relative,
                "sha256": _sha256_bytes(payload),
                "size_bytes": len(payload),
            }
        )
    return manifest, payloads


def _recovery_execution_projection_receipt(
    root: Path,
    suites: Sequence[Suite],
) -> dict[str, Any]:
    """Build the compact closed-copy projection receipt."""

    paths = _recovery_projection_source_paths(root, suites)
    manifest, _payloads = _recovery_projection_manifest(root, paths)
    selected_sources = []
    for suite in suites:
        relative = (Path(suite.owner_root) / suite.paths[0]).as_posix().lstrip("./")
        selected_sources.append(next(item for item in manifest if item["path"] == relative))
    body: dict[str, Any] = {
        "schema": "lgcvf-closed-recovery-test-projection@2",
        "selected_sources": selected_sources,
        "copied_source_count": len(manifest),
        "copied_source_bytes": sum(int(item["size_bytes"]) for item in manifest),
        "copied_source_manifest_root": content_identity(manifest),
        "omitted_source_symlinks": _recovery_projection_omitted_symlinks(root),
        "contains_live_source_links": False,
        "original_checkout_writable": False,
    }
    body["projection_cid"] = content_identity(body)
    return body


def _prepare_recovery_execution_checkout(
    root: Path,
    destination: Path,
    suites: Sequence[Suite],
) -> dict[str, Any]:
    """Copy the exact closed recovery dependency projection with no links."""

    paths = _recovery_projection_source_paths(root, suites)
    manifest, payloads = _recovery_projection_manifest(root, paths)
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    for item in manifest:
        relative = str(item["path"])
        target = destination.joinpath(*PurePath(relative).parts)
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o400,
        )
        try:
            view = memoryview(payloads[relative])
            while view:
                written = os.write(descriptor, view)
                if written < 1:
                    raise QualificationError("recovery projection write made no progress")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for directory, directories, _files in os.walk(destination, topdown=False):
        for name in directories:
            os.chmod(Path(directory) / name, 0o500)
        os.chmod(directory, 0o500)
    expected = _recovery_execution_projection_receipt(root, suites)
    copied_manifest, _ = _recovery_projection_manifest(
        destination,
        paths,
        head_bound=False,
    )
    if (
        content_identity(copied_manifest) != expected["copied_source_manifest_root"]
        or any(path.is_symlink() for path in destination.rglob("*"))
    ):
        raise QualificationError("closed recovery projection differs after copy")
    return expected


def _git_bytes(root: Path, args: Sequence[str]) -> bytes:
    try:
        substitution_before = _git_object_substitution_state(root)
    except RuntimeError as exc:
        raise QualificationError(
            "protected-input Git object substitution differs"
        ) from exc
    completed = subprocess.run(
        [
            "/usr/bin/git",
            *_RECOVERY_GIT_CONFIG_OVERRIDES,
            "-c",
            "core.hooksPath=/dev/null",
            *args,
        ],
        cwd=root,
        env={
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "HOME": "/nonexistent",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        },
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
    try:
        if _git_object_substitution_state(root) != substitution_before:
            raise QualificationError(
                "protected-input Git object substitution changed"
            )
    except RuntimeError as exc:
        raise QualificationError(
            "protected-input Git object substitution differs"
        ) from exc
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


def _read_owned_regular_at(
    directory_fd: int,
    name: str,
    *,
    noun: str,
    limit: int,
) -> bytes:
    """Read one bounded regular file without following a pathname link."""

    if not name or "/" in name or name in {".", ".."}:
        raise QualificationError(f"{noun} has an unsafe file name")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise QualificationError(f"{noun} is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid not in {0, os.geteuid()}
            or before.st_nlink != 1
            or before.st_size < 0
            or before.st_size > limit
        ):
            raise QualificationError(f"{noun} is not a bounded owned regular file")
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, limit + 1 - observed))
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            if observed > limit:
                raise QualificationError(f"{noun} exceeds its byte bound")
        after = os.fstat(descriptor)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or observed != before.st_size
        ):
            raise QualificationError(f"{noun} changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _open_owned_directory_at(
    directory_fd: int,
    name: str,
    *,
    noun: str,
) -> int:
    """Open one owned directory component without following a symlink."""

    if not name or "/" in name or name in {".", ".."}:
        raise QualificationError(f"{noun} has an unsafe directory name")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise QualificationError(f"{noun} is unavailable") from exc
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid not in {
        0,
        os.geteuid(),
    }:
        os.close(descriptor)
        raise QualificationError(f"{noun} is not an owned directory")
    return descriptor


def _pytest_site_roots() -> tuple[Path, ...]:
    """Return only interpreter-derived package roots, never ambient sys.path."""

    raw_roots: list[str] = []
    try:
        account_home = Path(pwd.getpwuid(os.geteuid()).pw_dir)
    except (KeyError, OSError) as exc:
        raise QualificationError("interpreter account home is unavailable") from exc
    if not account_home.is_absolute():
        raise QualificationError("interpreter account home is not absolute")
    userbase = str(account_home / ".local")
    for key in ("purelib", "platlib"):
        candidate = sysconfig.get_path(
            key,
            scheme="posix_user",
            vars={"userbase": userbase},
        )
        if candidate:
            raw_roots.append(candidate)
    try:
        raw_roots.extend(site.getsitepackages())
    except AttributeError:
        pass
    for key in ("purelib", "platlib"):
        candidate = sysconfig.get_path(key)
        if candidate:
            raw_roots.append(candidate)

    roots: list[Path] = []
    for raw_root in raw_roots:
        lexical = Path(raw_root)
        if not lexical.is_absolute():
            raise QualificationError("pytest distribution root is not absolute")
        try:
            metadata = lexical.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise QualificationError("pytest distribution root is unreadable") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise QualificationError("pytest distribution root is not a real directory")
        try:
            resolved = lexical.resolve(strict=True)
        except OSError as exc:
            raise QualificationError("pytest distribution root cannot be resolved") from exc
        if resolved != lexical or resolved in roots:
            if resolved != lexical:
                raise QualificationError("pytest distribution root contains a symlink")
            continue
        roots.append(resolved)
    return tuple(roots)


def _record_sha256_size(
    encoded_digest: str,
    encoded_size: str,
    *,
    noun: str,
) -> tuple[bytes, int]:
    """Decode one canonical sha256/size RECORD identity."""

    try:
        algorithm, payload = encoded_digest.split("=", 1)
        if algorithm != "sha256" or not payload or "=" in payload:
            raise ValueError
        padding = "=" * ((4 - len(payload) % 4) % 4)
        digest = base64.b64decode(
            payload + padding,
            altchars=b"-_",
            validate=True,
        )
        if (
            len(digest) != hashlib.sha256().digest_size
            or base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
            != payload
            or not encoded_size.isascii()
            or not encoded_size.isdecimal()
            or str(int(encoded_size)) != encoded_size
        ):
            raise ValueError
    except (ValueError, binascii.Error) as exc:
        raise QualificationError(f"{noun} has a malformed identity") from exc
    return digest, int(encoded_size)


def _strict_record_relative_path(
    relative: str,
    *,
    noun: str,
    external_paths: frozenset[str] = frozenset(),
) -> None:
    """Reject ambiguous RECORD paths before any identity matching."""

    components = relative.split("/")
    if relative not in external_paths and (
        not relative
        or relative.startswith("/")
        or "\\" in relative
        or any(
            ord(character) < 32 or ord(character) == 127 for character in relative
        )
        or re.match(r"^[A-Za-z]:", relative) is not None
        or any(component in {"", ".", ".."} for component in components)
    ):
        raise QualificationError(f"{noun} contains an unsafe RECORD path")


def _pytest_record_identity(
    record: bytes,
    *,
    noun: str,
    distribution_name: str,
    metadata_bytes: bytes,
) -> tuple[bytes, int]:
    """Extract package identity and bind the exact dist-info METADATA bytes."""

    try:
        rows = csv.reader(io.StringIO(record.decode("utf-8", errors="strict")))
        package_matches: list[tuple[str, str]] = []
        metadata_matches: list[tuple[str, str]] = []
        observed_paths: set[str] = set()
        metadata_path = f"{distribution_name}/METADATA"
        for row in rows:
            if len(row) != 3:
                raise QualificationError(f"{noun} contains a malformed RECORD row")
            relative, digest, size = row
            if relative in observed_paths:
                raise QualificationError(f"{noun} contains a duplicate RECORD path")
            observed_paths.add(relative)
            _strict_record_relative_path(
                relative,
                noun=noun,
                external_paths=_PYTEST_EXTERNAL_RECORD_PATHS,
            )
            if relative == "pytest/__init__.py":
                package_matches.append((digest, size))
            if relative == metadata_path:
                metadata_matches.append((digest, size))
    except (csv.Error, UnicodeDecodeError) as exc:
        raise QualificationError(f"{noun} is not a valid UTF-8 RECORD") from exc
    if len(package_matches) != 1:
        raise QualificationError(f"{noun} does not bind exactly one pytest package")
    if len(metadata_matches) != 1:
        raise QualificationError(f"{noun} does not bind exactly one METADATA file")
    package_identity = _record_sha256_size(
        *package_matches[0], noun=f"{noun} pytest package"
    )
    metadata_identity = _record_sha256_size(
        *metadata_matches[0], noun=f"{noun} METADATA"
    )
    if metadata_identity != (hashlib.sha256(metadata_bytes).digest(), len(metadata_bytes)):
        raise QualificationError(f"{noun} METADATA identity differs")
    return package_identity


def _pytest_distribution_version() -> str:
    """Resolve pytest from RECORD-bound bytes without importing its code.

    Isolated launch admission deliberately excludes the user site from
    ``sys.path``.  Qualification execution may nevertheless have used a
    user-site pytest.  This resolver treats distribution metadata and package
    bytes strictly as bounded data, and selects the sole distribution whose
    RECORD identity matches the installed package at the same interpreter-
    derived site root.  Stale dist-info directories therefore cannot win by
    name or search order, and no user-site Python is executed.
    """

    matches: list[str] = []
    candidate_count = 0
    directory_pattern = re.compile(
        r"pytest-([0-9A-Za-z][0-9A-Za-z.!+_-]{0,127})\.dist-info"
    )
    for root in _pytest_site_roots():
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            root_fd = os.open(root, flags)
        except OSError as exc:
            raise QualificationError("pytest distribution root cannot be opened") from exc
        try:
            root_status = os.fstat(root_fd)
            if not stat.S_ISDIR(root_status.st_mode) or root_status.st_uid not in {
                0,
                os.geteuid(),
            }:
                raise QualificationError("pytest distribution root is not owned")
            names: list[str] = []
            with os.scandir(root_fd) as entries:
                for entry in entries:
                    if not directory_pattern.fullmatch(entry.name):
                        continue
                    names.append(entry.name)
                    candidate_count += 1
                    if candidate_count > _MAX_PYTEST_DISTRIBUTION_CANDIDATES:
                        raise QualificationError(
                            "pytest distribution population exceeds its bound"
                        )
            names.sort()
            if not names:
                continue
            package_fd = _open_owned_directory_at(
                root_fd,
                "pytest",
                noun="pytest package directory",
            )
            try:
                package = _read_owned_regular_at(
                    package_fd,
                    "__init__.py",
                    noun="pytest package entry point",
                    limit=_MAX_PYTEST_METADATA_BYTES,
                )
            finally:
                os.close(package_fd)
            package_digest = hashlib.sha256(package).digest()
            package_size = len(package)

            for name in names:
                candidate_fd = _open_owned_directory_at(
                    root_fd,
                    name,
                    noun="pytest distribution directory",
                )
                try:
                    metadata_bytes = _read_owned_regular_at(
                        candidate_fd,
                        "METADATA",
                        noun="pytest distribution METADATA",
                        limit=_MAX_PYTEST_METADATA_BYTES,
                    )
                    record_bytes = _read_owned_regular_at(
                        candidate_fd,
                        "RECORD",
                        noun="pytest distribution RECORD",
                        limit=_MAX_PYTEST_RECORD_BYTES,
                    )
                finally:
                    os.close(candidate_fd)
                try:
                    metadata = email.parser.BytesParser(
                        policy=email.policy.compat32
                    ).parsebytes(metadata_bytes, headersonly=True)
                except (TypeError, ValueError) as exc:
                    raise QualificationError(
                        "pytest distribution METADATA is malformed"
                    ) from exc
                names_found = metadata.get_all("Name", [])
                versions_found = metadata.get_all("Version", [])
                if (
                    metadata.defects
                    or len(names_found) != 1
                    or len(versions_found) != 1
                    or re.sub(r"[-_.]+", "-", str(names_found[0])).casefold()
                    != "pytest"
                ):
                    raise QualificationError(
                        "pytest distribution METADATA identity differs"
                    )
                version = str(versions_found[0]).strip()
                directory_match = directory_pattern.fullmatch(name)
                if (
                    directory_match is None
                    or version != directory_match.group(1)
                    or not re.fullmatch(
                        r"[0-9A-Za-z][0-9A-Za-z.!+_-]{0,127}", version
                    )
                ):
                    raise QualificationError(
                        "pytest distribution version is not canonical"
                    )
                record_digest, record_size = _pytest_record_identity(
                    record_bytes,
                    noun=f"pytest {version} RECORD",
                    distribution_name=name,
                    metadata_bytes=metadata_bytes,
                )
                if record_digest == package_digest and record_size == package_size:
                    matches.append(version)
        finally:
            os.close(root_fd)
    if len(matches) != 1:
        raise QualificationError(
            "pytest distribution does not have one exact RECORD-matched installation"
        )
    return matches[0]


def _read_owned_regular_relative(
    root_fd: int,
    relative: str,
    *,
    noun: str,
    limit: int,
) -> bytes:
    """Read a strict relative file through an fd-scoped no-follow walk."""

    _strict_record_relative_path(relative, noun=noun)
    components = relative.split("/")
    directory_fd = os.dup(root_fd)
    try:
        for component in components[:-1]:
            child_fd = _open_owned_directory_at(
                directory_fd,
                component,
                noun=f"{noun} directory",
            )
            os.close(directory_fd)
            directory_fd = child_fd
        return _read_owned_regular_at(
            directory_fd,
            components[-1],
            noun=noun,
            limit=limit,
        )
    finally:
        os.close(directory_fd)


def _private_primary_group_identity() -> tuple[int, int]:
    """Return the current uid/private primary gid or fail closed."""

    uid = os.geteuid()
    try:
        account = pwd.getpwuid(uid)
        group = grp.getgrgid(account.pw_gid)
        flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open("/etc/passwd", flags)
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != 0
                or before.st_nlink != 1
                or before.st_size < 0
                or before.st_size > 1024 * 1024
                or stat.S_IMODE(before.st_mode) & 0o022
            ):
                raise QualificationError("qualification passwd authority differs")
            passwd_bytes = os.read(descriptor, before.st_size + 1)
            after = os.fstat(descriptor)
            if (
                len(passwd_bytes) != before.st_size
                or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            ):
                raise QualificationError("qualification passwd authority changed")
        finally:
            os.close(descriptor)
        accounts: list[tuple[int, str]] = []
        lines = passwd_bytes.decode("utf-8", errors="strict").splitlines()
        if len(lines) > 4_096:
            raise QualificationError(
                "qualification runtime account population exceeds its bound"
            )
        for line in lines:
            fields = line.split(":")
            if len(fields) != 7 or not fields[2].isdecimal() or not fields[3].isdecimal():
                raise QualificationError("qualification passwd authority is malformed")
            if int(fields[3]) == account.pw_gid:
                accounts.append((int(fields[2]), fields[0]))
    except (KeyError, OSError) as exc:
        raise QualificationError("qualification runtime account identity is unavailable") from exc
    if (
        account.pw_uid != uid
        or account.pw_gid != os.getegid()
        or sorted(accounts) != [(uid, account.pw_name)]
        or any(member != account.pw_name for member in group.gr_mem)
    ):
        raise QualificationError("qualification runtime primary group is not private")
    return uid, account.pw_gid


def _qualification_source_mode_is_safe(metadata: os.stat_result, *, owner_uid: int) -> bool:
    """Validate one source mode against the closed owner/private-group policy."""

    mode = stat.S_IMODE(metadata.st_mode)
    if metadata.st_uid != owner_uid or mode & 0o002:
        return False
    if mode & 0o020:
        if owner_uid == 0:
            return False
        uid, private_gid = _private_primary_group_identity()
        if owner_uid != uid or metadata.st_gid != private_gid:
            return False
    return True


def _read_qualification_runtime_relative(
    root_fd: int,
    relative: str,
    *,
    noun: str,
    owner_uid: int,
    limit: int = _MAX_QUALIFICATION_RUNTIME_FILE_BYTES,
) -> tuple[bytes, dict[str, Any]]:
    """Stable-read one runtime source with a no-follow component walk."""

    _strict_record_relative_path(relative, noun=noun)
    components = relative.split("/")
    directory_fd = os.dup(root_fd)
    try:
        root_status = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(root_status.st_mode)
            or not _qualification_source_mode_is_safe(root_status, owner_uid=owner_uid)
            or not (stat.S_IMODE(root_status.st_mode) & 0o500) == 0o500
        ):
            raise QualificationError(f"{noun} root directory identity differs")
        for component in components[:-1]:
            flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
            flags |= getattr(os, "O_NOFOLLOW", 0)
            child_fd = os.open(component, flags, dir_fd=directory_fd)
            child_status = os.fstat(child_fd)
            if (
                not stat.S_ISDIR(child_status.st_mode)
                or not _qualification_source_mode_is_safe(
                    child_status, owner_uid=owner_uid
                )
                or not (stat.S_IMODE(child_status.st_mode) & 0o500) == 0o500
            ):
                os.close(child_fd)
                raise QualificationError(f"{noun} directory identity differs")
            os.close(directory_fd)
            directory_fd = child_fd
        flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        source_fd = os.open(components[-1], flags, dir_fd=directory_fd)
        try:
            before = os.fstat(source_fd)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size < 0
                or before.st_size > limit
                or not _qualification_source_mode_is_safe(before, owner_uid=owner_uid)
                or not stat.S_IMODE(before.st_mode) & 0o400
            ):
                raise QualificationError(f"{noun} file identity differs")
            chunks: list[bytes] = []
            observed = 0
            while True:
                block = os.read(source_fd, min(1024 * 1024, limit + 1 - observed))
                if not block:
                    break
                chunks.append(block)
                observed += len(block)
                if observed > limit:
                    raise QualificationError(f"{noun} exceeds its byte bound")
            after = os.fstat(source_fd)
            stable_fields = (
                "st_dev",
                "st_ino",
                "st_uid",
                "st_gid",
                "st_mode",
                "st_nlink",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
            if (
                observed != before.st_size
                or tuple(getattr(before, key) for key in stable_fields)
                != tuple(getattr(after, key) for key in stable_fields)
            ):
                raise QualificationError(f"{noun} changed while it was read")
            payload = b"".join(chunks)
            return payload, {
                "dev": before.st_dev,
                "ino": before.st_ino,
                "uid": before.st_uid,
                "gid": before.st_gid,
                "mode": stat.S_IMODE(before.st_mode),
                "nlink": before.st_nlink,
                "size_bytes": before.st_size,
                "mtime_ns": before.st_mtime_ns,
                "ctime_ns": before.st_ctime_ns,
                "sha256": _sha256_bytes(payload),
            }
        finally:
            os.close(source_fd)
    except OSError as exc:
        raise QualificationError(f"{noun} is unavailable") from exc
    finally:
        os.close(directory_fd)


def _qualification_runtime_is_bytecode(path: str) -> bool:
    logical = PurePath(path)
    return (
        "__pycache__" in logical.parts
        or logical.name.endswith(".pyc")
        or logical.name.endswith(".pyo")
    )


def _qualification_runtime_is_native(path: str) -> bool:
    name = PurePath(path).name
    return (
        name.endswith((".so", ".pyd", ".dylib"))
        or ".so." in name
        or name.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES))
    )


def _qualification_runtime_path_allowed(
    relative: str, allowed_roots: tuple[str, ...]
) -> bool:
    return any(
        relative == root or relative.startswith(root + "/")
        for root in allowed_roots
    )


def _elf_string(table: bytes, offset: int, *, noun: str) -> str:
    if offset < 0 or offset >= len(table):
        raise QualificationError(f"{noun} ELF string offset differs")
    end = table.find(b"\0", offset)
    if end < 0:
        raise QualificationError(f"{noun} ELF string is unterminated")
    try:
        return table[offset:end].decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise QualificationError(f"{noun} ELF string is not UTF-8") from exc


def _elf64_binding(payload: bytes, *, noun: str) -> dict[str, Any]:
    """Parse the bounded AArch64 ELF identity without executing a tool."""

    header_format = "<16sHHIQQQIHHHHHH"
    if len(payload) < struct.calcsize(header_format):
        raise QualificationError(f"{noun} is not a complete ELF file")
    values = struct.unpack_from(header_format, payload)
    ident = values[0]
    if (
        ident[:4] != b"\x7fELF"
        or ident[4] != 2
        or ident[5] != 1
        or ident[6] != 1
        or values[2] != 183
    ):
        raise QualificationError(f"{noun} ELF platform differs")
    elf_type = int(values[1])
    phoff = int(values[5])
    phentsize = int(values[9])
    phnum = int(values[10])
    program_format = "<IIQQQQQQ"
    if (
        phentsize != struct.calcsize(program_format)
        or phnum <= 0
        or phnum > 256
        or phoff < 0
        or phoff + phnum * phentsize > len(payload)
    ):
        raise QualificationError(f"{noun} ELF program headers differ")
    programs = [
        struct.unpack_from(program_format, payload, phoff + index * phentsize)
        for index in range(phnum)
    ]

    def virtual_to_offset(address: int, size: int) -> int:
        for program in programs:
            if program[0] != 1:
                continue
            file_offset, virtual, file_size = int(program[2]), int(program[3]), int(program[5])
            if address >= virtual and address + size <= virtual + file_size:
                result = file_offset + address - virtual
                if result < 0 or result + size > len(payload):
                    break
                return result
        raise QualificationError(f"{noun} ELF virtual address differs")

    interpreter = ""
    build_id = ""
    for program in programs:
        segment_type, offset, file_size = int(program[0]), int(program[2]), int(program[5])
        if offset < 0 or file_size < 0 or offset + file_size > len(payload):
            raise QualificationError(f"{noun} ELF segment differs")
        if segment_type == 3:
            raw = payload[offset : offset + file_size]
            if not raw.endswith(b"\0"):
                raise QualificationError(f"{noun} ELF interpreter differs")
            interpreter = raw[:-1].decode("utf-8", errors="strict")
        elif segment_type == 4:
            cursor = offset
            limit = offset + file_size
            while cursor + 12 <= limit:
                name_size, description_size, note_type = struct.unpack_from(
                    "<III", payload, cursor
                )
                cursor += 12
                name = payload[cursor : cursor + name_size]
                cursor += (name_size + 3) & ~3
                description = payload[cursor : cursor + description_size]
                cursor += (description_size + 3) & ~3
                if cursor > limit:
                    raise QualificationError(f"{noun} ELF note differs")
                if note_type == 3 and name.rstrip(b"\0") == b"GNU":
                    if build_id:
                        raise QualificationError(f"{noun} has duplicate ELF build IDs")
                    build_id = description.hex()

    dynamic_entries: list[tuple[int, int]] = []
    for program in programs:
        if program[0] != 2:
            continue
        offset, file_size = int(program[2]), int(program[5])
        if file_size % 16 or offset + file_size > len(payload):
            raise QualificationError(f"{noun} ELF dynamic table differs")
        for cursor in range(offset, offset + file_size, 16):
            tag, value = struct.unpack_from("<qQ", payload, cursor)
            dynamic_entries.append((int(tag), int(value)))
            if tag == 0:
                break
    string_addresses = [value for tag, value in dynamic_entries if tag == 5]
    string_sizes = [value for tag, value in dynamic_entries if tag == 10]
    if len(string_addresses) != 1 or len(string_sizes) != 1:
        raise QualificationError(f"{noun} ELF dynamic strings differ")
    string_offset = virtual_to_offset(string_addresses[0], string_sizes[0])
    strings = payload[string_offset : string_offset + string_sizes[0]]

    def values_for(tag: int) -> list[str]:
        return [_elf_string(strings, value, noun=noun) for key, value in dynamic_entries if key == tag]

    sonames = values_for(14)
    rpaths = values_for(15)
    runpaths = values_for(29)
    if len(sonames) > 1 or len(rpaths) > 1 or len(runpaths) > 1:
        raise QualificationError(f"{noun} ELF dynamic identity is ambiguous")
    return {
        "elf_class": "ELF64",
        "data_encoding": "little_endian",
        "machine": "AArch64",
        "elf_type": elf_type,
        "build_id": build_id,
        "pt_interp": interpreter,
        "soname": sonames[0] if sonames else "",
        "needed": values_for(1),
        "rpath": rpaths[0] if rpaths else "",
        "runpath": runpaths[0] if runpaths else "",
    }


def _runtime_manifest_entry(path: str, payload: bytes) -> dict[str, Any]:
    native = _qualification_runtime_is_native(path)
    return {
        "path": path,
        "sha256": _sha256_bytes(payload),
        "size_bytes": len(payload),
        "projection_mode_octal": "0500" if native else "0400",
        "native": native,
    }


def _runtime_metadata_headers(
    payload: bytes, spec: _QualificationRuntimeComponentSpec
) -> tuple[str, dict[str, Any]]:
    try:
        metadata = email.parser.BytesParser(policy=email.policy.compat32).parsebytes(
            payload, headersonly=True
        )
    except (TypeError, ValueError) as exc:
        raise QualificationError(
            f"{spec.normalized_name} METADATA is malformed"
        ) from exc
    names = metadata.get_all("Name", [])
    versions = metadata.get_all("Version", [])
    requires_python = metadata.get_all("Requires-Python", [])
    normalized = (
        re.sub(r"[-_.]+", "-", str(names[0])).casefold() if names else ""
    )
    if (
        metadata.defects
        or len(names) != 1
        or len(versions) != 1
        or normalized != spec.normalized_name
        or str(versions[0]).strip() != spec.version
        or len(requires_python) > 1
    ):
        raise QualificationError(
            f"{spec.normalized_name} METADATA identity differs"
        )
    observed_requires_python = (
        str(requires_python[0]).strip() if requires_python else ""
    )
    if observed_requires_python != _QUALIFICATION_RUNTIME_REQUIRES_PYTHON[
        spec.normalized_name
    ]:
        raise QualificationError(
            f"{spec.normalized_name} Requires-Python differs"
        )
    dependencies = tuple(
        str(value).strip() for value in metadata.get_all("Requires-Dist", [])
    )
    active: list[str] = []
    inactive: list[str] = []
    for dependency in dependencies:
        requirement, separator, marker = dependency.partition(";")
        requirement = requirement.strip()
        marker = marker.strip()
        if not separator:
            active.append(requirement)
            continue
        normalized_marker = re.sub(r"\s+", " ", marker).strip().casefold()
        inactive_marker = (
            "extra" in normalized_marker
            or normalized_marker in {
                'sys_platform == "win32"',
                'python_version < "3.11"',
                'python_version < "3.9"',
                "python_version < '3.8'",
            }
            or (
                "python_version < '3.8'" in normalized_marker
                and "extra == 'plugins'" in normalized_marker
            )
        )
        if not inactive_marker:
            raise QualificationError(
                f"{spec.normalized_name} dependency marker is not closed"
            )
        inactive.append(dependency)
    normalized_active = [
        re.sub(r"\s+", "", value).casefold() for value in active
    ]
    expected_active = [
        re.sub(r"\s+", "", value).casefold()
        for value in spec.active_dependencies
    ]
    if normalized_active != expected_active:
        raise QualificationError(
            f"{spec.normalized_name} active dependency closure differs"
        )
    dependency_partition = {
        "environment": {
            "python_version": "3.12",
            "sys_platform": "linux",
            "extras": [],
        },
        "active": active,
        "inactive": inactive,
    }
    return observed_requires_python, dependency_partition


def _runtime_wheel_tags(payload: bytes, *, component: str) -> list[str]:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise QualificationError(f"{component} WHEEL is not UTF-8") from exc
    tags: list[str] = []
    for line in text.splitlines():
        if not line.startswith("Tag:"):
            continue
        tag = line.partition(":")[2].strip()
        if not re.fullmatch(r"[A-Za-z0-9_.]+-[A-Za-z0-9_.]+-[A-Za-z0-9_.]+", tag):
            raise QualificationError(f"{component} WHEEL tag is malformed")
        tags.append(tag)
    if not tags or len(tags) != len(set(tags)):
        raise QualificationError(f"{component} WHEEL tags differ")
    return tags


def _runtime_record_rows(
    payload: bytes, *, component: str, external_paths: frozenset[str]
) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    observed: set[str] = set()
    try:
        reader = csv.reader(io.StringIO(payload.decode("utf-8", errors="strict")))
        for row in reader:
            if len(row) != 3:
                raise QualificationError(f"{component} RECORD row differs")
            relative, digest, size = row
            if relative in observed:
                raise QualificationError(f"{component} RECORD path is duplicated")
            observed.add(relative)
            _strict_record_relative_path(
                relative,
                noun=f"{component} RECORD",
                external_paths=external_paths,
            )
            rows.append((relative, digest, size))
    except (csv.Error, UnicodeDecodeError) as exc:
        raise QualificationError(f"{component} RECORD is malformed") from exc
    if not rows or len(rows) > 8_192:
        raise QualificationError(f"{component} RECORD population differs")
    return rows


def _candidate_distribution_directories(
    root_fd: int, spec: _QualificationRuntimeComponentSpec
) -> list[str]:
    normalized_stem = re.sub(r"[-_.]+", "_", spec.normalized_name)
    names: list[str] = []
    observed = 0
    with os.scandir(root_fd) as entries:
        for entry in entries:
            observed += 1
            if observed > _MAX_QUALIFICATION_RUNTIME_SITE_ENTRIES:
                raise QualificationError("qualification site root population exceeds its bound")
            folded = entry.name.casefold()
            if not folded.endswith(".dist-info"):
                continue
            prefix = folded[: -len(".dist-info")].split("-", 1)[0]
            if re.sub(r"[-_.]+", "_", prefix) != normalized_stem:
                continue
            names.append(entry.name)
            if len(names) > _MAX_QUALIFICATION_RUNTIME_DISTRIBUTIONS:
                raise QualificationError(
                    f"{spec.normalized_name} distribution population exceeds its bound"
                )
    return sorted(names)


def _wheel_runtime_component(
    spec: _QualificationRuntimeComponentSpec,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], dict[str, bytes]]:
    matches: list[tuple[dict[str, Any], tuple[dict[str, Any], ...], dict[str, bytes]]] = []
    for site_root in _pytest_site_roots():
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
        try:
            root_fd = os.open(site_root, flags)
        except OSError as exc:
            raise QualificationError("qualification site root is unavailable") from exc
        try:
            root_status = os.fstat(root_fd)
            owner_uid = root_status.st_uid
            if owner_uid not in {0, os.geteuid()} or not _qualification_source_mode_is_safe(
                root_status, owner_uid=owner_uid
            ):
                raise QualificationError("qualification site root identity differs")
            for candidate_name in _candidate_distribution_directories(root_fd, spec):
                candidate_fd = _open_owned_directory_at(
                    root_fd,
                    candidate_name,
                    noun=f"{spec.normalized_name} distribution directory",
                )
                try:
                    metadata_bytes, _metadata_source = _read_qualification_runtime_relative(
                        candidate_fd,
                        "METADATA",
                        noun=f"{spec.normalized_name} METADATA",
                        owner_uid=owner_uid,
                        limit=_MAX_PYTEST_METADATA_BYTES,
                    )
                    try:
                        _runtime_metadata_headers(metadata_bytes, spec)
                    except QualificationError:
                        # Other installed versions are not candidates for this exact
                        # policy component; malformed exact-name/version metadata is
                        # rejected below when its directory name collides.
                        if candidate_name == spec.distribution_directory:
                            raise
                        continue
                    if candidate_name != spec.distribution_directory:
                        raise QualificationError(
                            f"{spec.normalized_name} exact distribution is ambiguous"
                        )
                    record_bytes, _record_source = _read_qualification_runtime_relative(
                        candidate_fd,
                        "RECORD",
                        noun=f"{spec.normalized_name} RECORD",
                        owner_uid=owner_uid,
                        limit=_MAX_QUALIFICATION_RUNTIME_RECORD_BYTES,
                    )
                finally:
                    os.close(candidate_fd)
                external_paths = _QUALIFICATION_RUNTIME_EXTERNAL_RECORD_PATHS.get(
                    spec.normalized_name, frozenset()
                )
                rows = _runtime_record_rows(
                    record_bytes,
                    component=spec.normalized_name,
                    external_paths=external_paths,
                )
                record_path = f"{spec.distribution_directory}/RECORD"
                payloads: dict[str, bytes] = {}
                manifest: list[dict[str, Any]] = []
                omitted: list[str] = []
                record_matches = 0
                for relative, digest, size in rows:
                    if relative in external_paths:
                        omitted.append(relative)
                        continue
                    if _qualification_runtime_is_bytecode(relative):
                        continue
                    if relative.endswith(".pth"):
                        raise QualificationError(
                            f"{spec.normalized_name} RECORD contains a pth file"
                        )
                    if not _qualification_runtime_path_allowed(relative, spec.allowed_roots):
                        raise QualificationError(
                            f"{spec.normalized_name} RECORD escapes its allowed roots"
                        )
                    if relative == record_path:
                        record_matches += 1
                        if digest or size:
                            raise QualificationError(
                                f"{spec.normalized_name} RECORD self row is not blank"
                            )
                        item_bytes = record_bytes
                    else:
                        expected_digest, expected_size = _record_sha256_size(
                            digest,
                            size,
                            noun=f"{spec.normalized_name} RECORD {relative}",
                        )
                        item_bytes, _source = _read_qualification_runtime_relative(
                            root_fd,
                            relative,
                            noun=f"{spec.normalized_name} runtime {relative}",
                            owner_uid=owner_uid,
                        )
                        if (hashlib.sha256(item_bytes).digest(), len(item_bytes)) != (
                            expected_digest,
                            expected_size,
                        ):
                            raise QualificationError(
                                f"{spec.normalized_name} runtime bytes differ from RECORD"
                            )
                    payloads[relative] = item_bytes
                    manifest.append(_runtime_manifest_entry(relative, item_bytes))
                if record_matches != 1 or spec.package_entry not in payloads:
                    raise QualificationError(
                        f"{spec.normalized_name} RECORD authority differs"
                    )
                metadata_path = f"{spec.distribution_directory}/METADATA"
                wheel_path = f"{spec.distribution_directory}/WHEEL"
                if metadata_path not in payloads or wheel_path not in payloads:
                    raise QualificationError(
                        f"{spec.normalized_name} distribution control files are absent"
                    )
                requires_python, dependency_partition = _runtime_metadata_headers(
                    payloads[metadata_path], spec
                )
                wheel_tags = _runtime_wheel_tags(
                    payloads[wheel_path], component=spec.normalized_name
                )
                manifest.sort(key=lambda item: str(item["path"]))
                if (
                    len(manifest) != spec.expected_file_count
                    or sum(int(item["size_bytes"]) for item in manifest)
                    != spec.expected_total_bytes
                ):
                    raise QualificationError(
                        f"{spec.normalized_name} runtime population differs"
                    )
                native_files = [
                    {
                        **{key: item[key] for key in ("path", "sha256", "size_bytes")},
                        **_elf64_binding(
                            payloads[str(item["path"])],
                            noun=f"{spec.normalized_name} native {item['path']}",
                        ),
                    }
                    for item in manifest
                    if item["native"] is True
                ]
                native_binding = {
                    "native_file_count": len(native_files),
                    "native_files": native_files,
                    "native_file_root": content_identity(native_files),
                }
                component: dict[str, Any] = {
                    "schema": _QUALIFICATION_RUNTIME_COMPONENT_SCHEMA,
                    "ordinal": spec.ordinal,
                    "role": spec.role,
                    "normalized_name": spec.normalized_name,
                    "version": spec.version,
                    "provenance_kind": spec.provenance_kind,
                    "allowed_roots": list(spec.allowed_roots),
                    "metadata": {
                        "path": metadata_path,
                        "sha256": _sha256_bytes(payloads[metadata_path]),
                        "size_bytes": len(payloads[metadata_path]),
                        "requires_python": requires_python,
                    },
                    "provenance_manifest": {
                        "kind": "wheel_record",
                        "path_token": record_path,
                        "sha256": _sha256_bytes(record_bytes),
                        "size_bytes": len(record_bytes),
                    },
                    "wheel_tags": wheel_tags,
                    "active_dependencies": dependency_partition,
                    "omitted_paths": sorted(omitted),
                    "file_count": len(manifest),
                    "total_bytes": sum(int(item["size_bytes"]) for item in manifest),
                    "file_manifest_root": content_identity(manifest),
                    "native_binding": native_binding,
                }
                if sorted(omitted) != sorted(external_paths):
                    raise QualificationError(
                        f"{spec.normalized_name} external RECORD omissions differ"
                    )
                component["component_cid"] = content_identity(component)
                matches.append((component, tuple(manifest), payloads))
        finally:
            os.close(root_fd)
    if len(matches) != 1:
        raise QualificationError(
            f"{spec.normalized_name} does not have one exact RECORD installation"
        )
    return matches[0]


_PYGMENTS_DPKG_LIST_SHA256: Final[str] = (
    "sha256:860fa2944c67a9a74d35ed41bef30c74a49a4da4ecb413aa8152ea4762166d06"
)
_PYGMENTS_DPKG_MD5SUMS_SHA256: Final[str] = (
    "sha256:047cfa17d9d1e9e765133d02194b2442573b6b0d3afbc9d7db8630c9082afee5"
)
_PYGMENTS_METADATA_SHA256: Final[str] = (
    "sha256:4b348aab273b98656f0ee2b8c4307cd1d471d06957a8ecdd12f6b00caf08e7f2"
)
_PYGMENTS_DPKG_OMITTED_PATHS: Final[frozenset[str]] = frozenset(
    {
        "usr/bin/pygmentize",
        "usr/share/bash-completion/completions/pygmentize",
        "usr/share/doc/python3-pygments/changelog.Debian.gz",
        "usr/share/doc/python3-pygments/copyright",
        "usr/share/man/man1/pygmentize.1.gz",
    }
)


def _pygments_runtime_component(
    spec: _QualificationRuntimeComponentSpec,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], dict[str, bytes]]:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    root_fd = os.open("/", flags)
    try:
        list_bytes, _ = _read_qualification_runtime_relative(
            root_fd,
            "var/lib/dpkg/info/python3-pygments.list",
            noun="Pygments dpkg list",
            owner_uid=0,
            limit=1024 * 1024,
        )
        md5_bytes, _ = _read_qualification_runtime_relative(
            root_fd,
            "var/lib/dpkg/info/python3-pygments.md5sums",
            noun="Pygments dpkg md5sums",
            owner_uid=0,
            limit=1024 * 1024,
        )
        status_bytes, _ = _read_qualification_runtime_relative(
            root_fd,
            "var/lib/dpkg/status",
            noun="Pygments dpkg status",
            owner_uid=0,
            limit=16 * 1024 * 1024,
        )
        if (
            _sha256_bytes(list_bytes) != _PYGMENTS_DPKG_LIST_SHA256
            or _sha256_bytes(md5_bytes) != _PYGMENTS_DPKG_MD5SUMS_SHA256
        ):
            raise QualificationError("Pygments dpkg provenance differs")
        try:
            listed = {
                line.strip().lstrip("/")
                for line in list_bytes.decode("utf-8", errors="strict").splitlines()
                if line.strip()
            }
            md5_rows = [
                tuple(line.split(maxsplit=1))
                for line in md5_bytes.decode("utf-8", errors="strict").splitlines()
                if line.strip()
            ]
        except UnicodeDecodeError as exc:
            raise QualificationError("Pygments dpkg provenance is not UTF-8") from exc
        if any(len(row) != 2 for row in md5_rows):
            raise QualificationError("Pygments dpkg md5sums is malformed")
        try:
            status_text = status_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise QualificationError("Pygments dpkg status is not UTF-8") from exc
        status_stanzas = [
            stanza
            for stanza in re.split(r"\n\s*\n", status_text)
            if any(line == "Package: python3-pygments" for line in stanza.splitlines())
        ]
        if len(status_stanzas) != 1:
            raise QualificationError("Pygments dpkg package status is ambiguous")
        status_fields: dict[str, str] = {}
        for line in status_stanzas[0].splitlines():
            if not line or line[0].isspace():
                continue
            key, separator, value = line.partition(":")
            if not separator or key in status_fields:
                raise QualificationError("Pygments dpkg package status is malformed")
            status_fields[key] = value.strip()
        if {
            "Package": status_fields.get("Package"),
            "Status": status_fields.get("Status"),
            "Architecture": status_fields.get("Architecture"),
            "Version": status_fields.get("Version"),
        } != {
            "Package": "python3-pygments",
            "Status": "install ok installed",
            "Architecture": "all",
            "Version": "2.17.2+dfsg-1",
        }:
            raise QualificationError("Pygments dpkg package identity differs")
        payloads: dict[str, bytes] = {}
        manifest: list[dict[str, Any]] = []
        omitted: list[str] = []
        prefix = "usr/lib/python3/dist-packages/"
        seen: set[str] = set()
        for raw_digest, raw_path in md5_rows:
            source_relative = str(raw_path).strip().lstrip("/")
            if source_relative in seen or source_relative not in listed:
                raise QualificationError("Pygments dpkg path authority differs")
            seen.add(source_relative)
            if not re.fullmatch(r"[0-9a-f]{32}", str(raw_digest)):
                raise QualificationError("Pygments dpkg digest is malformed")
            if not source_relative.startswith(prefix):
                omitted.append(source_relative)
                continue
            projected = source_relative[len(prefix) :]
            _strict_record_relative_path(projected, noun="Pygments dpkg payload")
            if not _qualification_runtime_path_allowed(projected, spec.allowed_roots):
                omitted.append(source_relative)
                continue
            if _qualification_runtime_is_bytecode(projected) or projected.endswith(".pth"):
                raise QualificationError("Pygments dpkg payload contains forbidden code")
            payload, _source = _read_qualification_runtime_relative(
                root_fd,
                source_relative,
                noun=f"Pygments runtime {projected}",
                owner_uid=0,
            )
            digest = hashlib.md5(payload, usedforsecurity=False).hexdigest()
            if digest != raw_digest:
                raise QualificationError("Pygments bytes differ from dpkg provenance")
            payloads[projected] = payload
            manifest.append(_runtime_manifest_entry(projected, payload))
        manifest.sort(key=lambda item: str(item["path"]))
        if (
            len(manifest) != spec.expected_file_count
            or sum(int(item["size_bytes"]) for item in manifest)
            != spec.expected_total_bytes
            or frozenset(omitted) != _PYGMENTS_DPKG_OMITTED_PATHS
        ):
            raise QualificationError("Pygments runtime population differs")
        metadata_path = f"{spec.distribution_directory}/METADATA"
        if (
            metadata_path not in payloads
            or _sha256_bytes(payloads[metadata_path]) != _PYGMENTS_METADATA_SHA256
        ):
            raise QualificationError("Pygments METADATA differs")
        requires_python, dependency_partition = _runtime_metadata_headers(
            payloads[metadata_path], spec
        )
        component: dict[str, Any] = {
            "schema": _QUALIFICATION_RUNTIME_COMPONENT_SCHEMA,
            "ordinal": spec.ordinal,
            "role": spec.role,
            "normalized_name": spec.normalized_name,
            "version": spec.version,
            "provenance_kind": spec.provenance_kind,
            "allowed_roots": list(spec.allowed_roots),
            "metadata": {
                "path": metadata_path,
                "sha256": _sha256_bytes(payloads[metadata_path]),
                "size_bytes": len(payloads[metadata_path]),
                "requires_python": requires_python,
            },
            "provenance_manifest": {
                "kind": "debian_dpkg_md5sums",
                "path_token": "dpkg:python3-pygments.md5sums",
                "sha256": _sha256_bytes(md5_bytes),
                "size_bytes": len(md5_bytes),
                "package_list_path_token": "dpkg:python3-pygments.list",
                "package_list_sha256": _sha256_bytes(list_bytes),
                "package_list_size_bytes": len(list_bytes),
                "package_status_stanza_sha256": _sha256_bytes(
                    status_stanzas[0].encode("utf-8")
                ),
                "package_version": status_fields["Version"],
                "package_architecture": status_fields["Architecture"],
                "package_status": status_fields["Status"],
            },
            "wheel_tags": [],
            "active_dependencies": dependency_partition,
            "omitted_paths": sorted(omitted),
            "file_count": len(manifest),
            "total_bytes": sum(int(item["size_bytes"]) for item in manifest),
            "file_manifest_root": content_identity(manifest),
            "native_binding": {
                "native_file_count": 0,
                "native_files": [],
                "native_file_root": content_identity([]),
            },
        }
        component["component_cid"] = content_identity(component)
        return component, tuple(manifest), payloads
    except OSError as exc:
        raise QualificationError("Pygments dpkg provenance is unavailable") from exc
    finally:
        os.close(root_fd)


_QUALIFICATION_HOST_OBJECTS: Final[
    tuple[tuple[str, str, str, tuple[tuple[str, str], ...]], ...]
] = (
    (
        "ld-linux-aarch64.so.1",
        "/lib/ld-linux-aarch64.so.1",
        "usr/lib/aarch64-linux-gnu/ld-linux-aarch64.so.1",
        (
            ("/lib", "usr/lib"),
            (
                "/usr/lib/ld-linux-aarch64.so.1",
                "aarch64-linux-gnu/ld-linux-aarch64.so.1",
            ),
        ),
    ),
    (
        "libc.so.6",
        "/lib/aarch64-linux-gnu/libc.so.6",
        "usr/lib/aarch64-linux-gnu/libc.so.6",
        (("/lib", "usr/lib"),),
    ),
    (
        "libm.so.6",
        "/lib/aarch64-linux-gnu/libm.so.6",
        "usr/lib/aarch64-linux-gnu/libm.so.6",
        (("/lib", "usr/lib"),),
    ),
    (
        "libgcc_s.so.1",
        "/lib/aarch64-linux-gnu/libgcc_s.so.1",
        "usr/lib/aarch64-linux-gnu/libgcc_s.so.1",
        (("/lib", "usr/lib"),),
    ),
    (
        "libpthread.so.0",
        "/lib/aarch64-linux-gnu/libpthread.so.0",
        "usr/lib/aarch64-linux-gnu/libpthread.so.0",
        (("/lib", "usr/lib"),),
    ),
    (
        "libstdc++.so.6",
        "/lib/aarch64-linux-gnu/libstdc++.so.6",
        "usr/lib/aarch64-linux-gnu/libstdc++.so.6.0.33",
        (
            ("/lib", "usr/lib"),
            (
                "/usr/lib/aarch64-linux-gnu/libstdc++.so.6",
                "libstdc++.so.6.0.33",
            ),
        ),
    ),
    (
        "libz.so.1",
        "/lib/aarch64-linux-gnu/libz.so.1",
        "usr/lib/aarch64-linux-gnu/libz.so.1.3",
        (
            ("/lib", "usr/lib"),
            ("/usr/lib/aarch64-linux-gnu/libz.so.1", "libz.so.1.3"),
        ),
    ),
    (
        "libexpat.so.1",
        "/lib/aarch64-linux-gnu/libexpat.so.1",
        "usr/lib/aarch64-linux-gnu/libexpat.so.1.9.1",
        (
            ("/lib", "usr/lib"),
            (
                "/usr/lib/aarch64-linux-gnu/libexpat.so.1",
                "libexpat.so.1.9.1",
            ),
        ),
    ),
    (
        "libreadline.so.8",
        "/lib/aarch64-linux-gnu/libreadline.so.8",
        "usr/lib/aarch64-linux-gnu/libreadline.so.8.2",
        (
            ("/lib", "usr/lib"),
            (
                "/usr/lib/aarch64-linux-gnu/libreadline.so.8",
                "libreadline.so.8.2",
            ),
        ),
    ),
    (
        "libtinfo.so.6",
        "/lib/aarch64-linux-gnu/libtinfo.so.6",
        "usr/lib/aarch64-linux-gnu/libtinfo.so.6.4",
        (
            ("/lib", "usr/lib"),
            (
                "/usr/lib/aarch64-linux-gnu/libtinfo.so.6",
                "libtinfo.so.6.4",
            ),
        ),
    ),
    (
        "libffi.so.8",
        "/lib/aarch64-linux-gnu/libffi.so.8",
        "usr/lib/aarch64-linux-gnu/libffi.so.8.1.4",
        (
            ("/lib", "usr/lib"),
            (
                "/usr/lib/aarch64-linux-gnu/libffi.so.8",
                "libffi.so.8.1.4",
            ),
        ),
    ),
    (
        "libcrypto.so.3",
        "/lib/aarch64-linux-gnu/libcrypto.so.3",
        "usr/lib/aarch64-linux-gnu/libcrypto.so.3",
        (("/lib", "usr/lib"),),
    ),
    (
        "libssl.so.3",
        "/lib/aarch64-linux-gnu/libssl.so.3",
        "usr/lib/aarch64-linux-gnu/libssl.so.3",
        (("/lib", "usr/lib"),),
    ),
    (
        "libsqlite3.so.0",
        "/lib/aarch64-linux-gnu/libsqlite3.so.0",
        "usr/lib/aarch64-linux-gnu/libsqlite3.so.0.8.6",
        (
            ("/lib", "usr/lib"),
            (
                "/usr/lib/aarch64-linux-gnu/libsqlite3.so.0",
                "libsqlite3.so.0.8.6",
            ),
        ),
    ),
)
_RECOVERY_080_PREBOUND_HOST_SONAMES: Final[tuple[str, ...]] = (
    "libcrypto.so.3",
    "libssl.so.3",
    "libsqlite3.so.0",
)
_QUALIFICATION_LOADER_ENVIRONMENT_VARIABLES: Final[tuple[str, ...]] = (
    "LD_AUDIT",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "GLIBC_TUNABLES",
)


def _stable_symlink_identity(path: str, expected_target: str) -> dict[str, Any]:
    try:
        before = os.lstat(path)
        target = os.readlink(path)
        after = os.lstat(path)
    except OSError as exc:
        raise QualificationError("native host symlink identity is unavailable") from exc
    if (
        not stat.S_ISLNK(before.st_mode)
        or before.st_uid != 0
        or before.st_nlink != 1
        or target != expected_target
        or (before.st_dev, before.st_ino, before.st_uid, before.st_gid, before.st_mode)
        != (after.st_dev, after.st_ino, after.st_uid, after.st_gid, after.st_mode)
    ):
        raise QualificationError("native host symlink identity differs")
    return {
        "path_token": "system:" + path,
        "target": target,
        "uid": before.st_uid,
        "gid": before.st_gid,
        "mode": stat.S_IMODE(before.st_mode),
        "nlink": before.st_nlink,
    }


def _native_host_runtime_binding(
    payload_native_files: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind the solver DT_NEEDED graph and pinned interpreter baseline."""

    if any(
        name == "GLIBC_TUNABLES" or name.startswith("LD_")
        for name in os.environ
    ):
        raise QualificationError("native loader environment is not empty")
    try:
        os.lstat("/etc/ld.so.preload")
    except FileNotFoundError:
        preload_absent = True
    except OSError as exc:
        raise QualificationError("native loader preload state is unavailable") from exc
    else:
        raise QualificationError("native loader preload file is present")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    root_fd = os.open("/", flags)
    observations: dict[str, Any] = {}
    try:
        cache_bytes, cache_observation = _read_qualification_runtime_relative(
            root_fd,
            "etc/ld.so.cache",
            noun="native loader cache",
            owner_uid=0,
            limit=4 * 1024 * 1024,
        )
        executable_bytes, executable_observation = _read_qualification_runtime_relative(
            root_fd,
            "usr/bin/python3.12",
            noun="qualification interpreter",
            owner_uid=0,
            limit=16 * 1024 * 1024,
        )
        interpreter_elf = _elf64_binding(
            executable_bytes, noun="qualification interpreter"
        )
        objects: list[dict[str, Any]] = []
        object_observations: list[dict[str, Any]] = []
        for soname, logical_path, real_relative, symlink_spec in _QUALIFICATION_HOST_OBJECTS:
            chain = [
                _stable_symlink_identity(path, target)
                for path, target in symlink_spec
            ]
            real_bytes, observed = _read_qualification_runtime_relative(
                root_fd,
                real_relative,
                noun=f"native host object {soname}",
                owner_uid=0,
            )
            elf = _elf64_binding(real_bytes, noun=f"native host object {soname}")
            if elf["soname"] not in {"", soname}:
                raise QualificationError("native host SONAME differs")
            objects.append(
                {
                    "soname": soname,
                    "logical_path_token": "system:" + logical_path,
                    "symlink_chain": chain,
                    "real_path_token": "system:/" + real_relative,
                    "sha256": _sha256_bytes(real_bytes),
                    "size_bytes": len(real_bytes),
                    "build_id": elf["build_id"],
                    "mode": observed["mode"],
                    "uid": observed["uid"],
                    "gid": observed["gid"],
                    "nlink": observed["nlink"],
                    "elf_type": elf["elf_type"],
                    "soname_field": elf["soname"],
                    "needed": elf["needed"],
                    "rpath": elf["rpath"],
                    "runpath": elf["runpath"],
                }
            )
            object_observations.append(
                {
                    "soname": soname,
                    "real_path": "/" + real_relative,
                    **observed,
                }
            )
        host_sonames = {str(item["soname"]) for item in objects}
        payload_sonames = {
            str(item.get("soname") or PurePath(str(item["path"])).name)
            for item in payload_native_files
        }
        allowed_needed = host_sonames | payload_sonames
        edges: list[dict[str, Any]] = []
        for item in [*payload_native_files, *objects]:
            source = str(item.get("path") or item.get("soname") or "")
            needed = [str(value) for value in item.get("needed") or ()]
            if any(value not in allowed_needed for value in needed):
                raise QualificationError(
                    f"native dependency closure escapes its admitted graph: {source}"
                )
            edges.append({"source": source, "needed": needed})
        interpreter_needed = [str(value) for value in interpreter_elf["needed"]]
        if any(value not in host_sonames for value in interpreter_needed):
            raise QualificationError("qualification interpreter dependency closure differs")
        edges.append({"source": "python:/usr/bin/python3.12", "needed": interpreter_needed})
        body: dict[str, Any] = {
            "schema": "lgcvf-native-host-runtime-binding@1",
            "closure_scope": (
                "solver_elf_needed_closure_interpreter_direct_baseline_"
                "and_prebound_stdlib_dependencies"
            ),
            "elf_class": "ELF64",
            "data_encoding": "little_endian",
            "machine": "AArch64",
            "interpreter": {
                "logical_path_token": "python:/usr/bin/python3.12",
                "sha256": _sha256_bytes(executable_bytes),
                "size_bytes": len(executable_bytes),
                "build_id": interpreter_elf["build_id"],
                "pt_interp": interpreter_elf["pt_interp"],
                "needed": interpreter_needed,
            },
            "loader_cache": {
                "path_token": "system:/etc/ld.so.cache",
                "sha256": _sha256_bytes(cache_bytes),
                "size_bytes": len(cache_bytes),
                "root_owned": cache_observation["uid"] == 0,
                "mode": cache_observation["mode"],
                "nlink": cache_observation["nlink"],
            },
            "preload_absent": preload_absent,
            "environment_loader_variables_absent": True,
            "objects": objects,
            "dependency_edges": edges,
            "closure_complete": True,
            "stdlib_extension_native_boundary": (
                "lgcvf_080_selected_host_dependencies_prebound_"
                "other_late_host_dependencies_fail_closed"
            ),
        }
        body["host_runtime_cid"] = content_identity(body)
        observations = {
            "loader_cache": cache_observation,
            "interpreter": executable_observation,
            "objects": object_observations,
            "preload_absent": True,
            "environment_loader_variables_absent": True,
        }
        return body, observations
    except OSError as exc:
        raise QualificationError("native host runtime is unavailable") from exc
    finally:
        os.close(root_fd)


def _qualification_python_runtime_binding(
    native_host: Mapping[str, Any],
) -> dict[str, Any]:
    executable = native_host.get("interpreter")
    if not isinstance(executable, Mapping):
        raise QualificationError("qualification interpreter evidence is absent")
    extension_suffix = str(sysconfig.get_config_var("EXT_SUFFIX") or "")
    soabi = str(sysconfig.get_config_var("SOABI") or "")
    expected_paths = _RECOVERY_EXPECTED_ISOLATED_STDLIB_PATHS
    if (
        Path(sys.executable).resolve(strict=True) != Path("/usr/bin/python3.12")
        or platform.python_version() != "3.12.3"
        or sys.implementation.name != "cpython"
        or sys.implementation.cache_tag != "cpython-312"
        or soabi != "cpython-312-aarch64-linux-gnu"
        or extension_suffix != ".cpython-312-aarch64-linux-gnu.so"
        or platform.machine() != "aarch64"
        or platform.system() != "Linux"
        or sys.byteorder != "little"
        or _RECOVERY_INITIAL_ISOLATED_STDLIB_PATHS != expected_paths
    ):
        raise QualificationError("qualification Python runtime identity differs")
    stdlib_extensions, stdlib_extension_root, stdlib_extension_bytes = (
        _stdlib_extension_native_manifest()
    )
    ctypes_path = "/usr/lib/python3.12/lib-dynload/_ctypes.cpython-312-aarch64-linux-gnu.so"
    ctypes_extension = stdlib_extensions.get(ctypes_path)
    host_objects = native_host.get("objects")
    libffi_objects = (
        [item for item in host_objects if item.get("soname") == "libffi.so.8"]
        if isinstance(host_objects, list)
        and all(isinstance(item, Mapping) for item in host_objects)
        else []
    )
    if not isinstance(ctypes_extension, Mapping) or len(libffi_objects) != 1:
        raise QualificationError("ctypes/libffi W+X producer binding is absent")
    ctypes_libffi_producer: dict[str, Any] = {
        "schema": "lgcvf-z3-libffi-rwx-producer-binding@1",
        "ctypes_extension": dict(ctypes_extension),
        "libffi_host_object": dict(libffi_objects[0]),
        "qualified_worker_scope": ["LGCVF-051", "LGCVF-060"],
        "expected_anonymous_mapping": (
            _expected_worker_writable_executable_mappings(z3_required=True)[0]
        ),
        "controller_rwx_permitted": False,
        "non_z3_worker_rwx_permitted": False,
    }
    ctypes_libffi_producer["producer_binding_cid"] = content_identity(
        ctypes_libffi_producer
    )
    return {
        "executable_path_token": "python:/usr/bin/python3.12",
        "executable_sha256": executable.get("sha256"),
        "executable_size_bytes": executable.get("size_bytes"),
        "implementation": sys.implementation.name,
        "version": platform.python_version(),
        "cache_tag": sys.implementation.cache_tag,
        "soabi": soabi,
        "extension_suffix": extension_suffix,
        "machine": platform.machine(),
        "system": platform.system(),
        "byteorder": sys.byteorder,
        "initial_isolated_stdlib_paths": list(expected_paths),
        "required_flags": ["-I", "-S", "-B"],
        "stdlib_extension_binding": {
            "schema": "lgcvf-stdlib-extension-native-binding@1",
            "directory_path_token": "stdlib:/usr/lib/python3.12/lib-dynload",
            "file_count": len(stdlib_extensions),
            "total_bytes": stdlib_extension_bytes,
            "file_manifest_root": stdlib_extension_root,
            "root_owned_regular_nofollow": True,
            "transitive_native_dependencies": (
                "pinned_root_owned_host_platform_not_enumerated"
            ),
        },
        "z3_libffi_rwx_producer_binding": ctypes_libffi_producer,
    }


def _resolve_qualification_runtime() -> _ResolvedQualificationRuntime:
    """Resolve the one exact sealed pytest/semantic/solver runtime as data."""

    components: list[dict[str, Any]] = []
    payload_manifest: list[dict[str, Any]] = []
    payload_bytes: dict[str, bytes] = {}
    for spec in _QUALIFICATION_RUNTIME_COMPONENTS:
        if spec.provenance_kind == "debian_dpkg_md5sums":
            component, manifest, files = _pygments_runtime_component(spec)
        else:
            component, manifest, files = _wheel_runtime_component(spec)
        if any(path in payload_bytes for path in files):
            raise QualificationError("qualification runtime payload paths overlap")
        components.append(component)
        payload_manifest.extend(dict(item) for item in manifest)
        payload_bytes.update(files)
    payload_manifest.sort(key=lambda item: str(item["path"]))
    if (
        len(components) != len(_QUALIFICATION_RUNTIME_COMPONENTS)
        or len(payload_manifest) != _MAX_QUALIFICATION_RUNTIME_PAYLOAD_FILES
        or len(payload_bytes) != _MAX_QUALIFICATION_RUNTIME_PAYLOAD_FILES
        or sum(int(item["size_bytes"]) for item in payload_manifest)
        != _QUALIFICATION_RUNTIME_PAYLOAD_BYTES
    ):
        raise QualificationError("qualification runtime aggregate population differs")
    payload_native = [
        native
        for component in components
        for native in component["native_binding"]["native_files"]
    ]
    host_runtime, host_observation = _native_host_runtime_binding(payload_native)
    python_runtime = _qualification_python_runtime_binding(host_runtime)
    component_summaries = [
        {
            "ordinal": item["ordinal"],
            "role": item["role"],
            "normalized_name": item["normalized_name"],
            "version": item["version"],
            "file_count": item["file_count"],
            "total_bytes": item["total_bytes"],
            "file_manifest_root": item["file_manifest_root"],
            "component_cid": item["component_cid"],
        }
        for item in components
    ]
    omissions = {
        "schema": "lgcvf-qualification-runtime-omissions@1",
        "bytecode_suffixes": [".pyc", ".pyo"],
        "bytecode_directories": ["__pycache__"],
        "pth_processed": False,
        "external_record_paths": {
            name: sorted(paths)
            for name, paths in sorted(
                _QUALIFICATION_RUNTIME_EXTERNAL_RECORD_PATHS.items()
            )
        },
        "component_omitted_paths": [
            {
                "ordinal": item["ordinal"],
                "normalized_name": item["normalized_name"],
                "omitted_paths": item["omitted_paths"],
            }
            for item in components
        ],
    }
    native_platform = {
        "schema": "lgcvf-qualification-native-platform@1",
        "solver_payload_native_files": payload_native,
        "solver_payload_native_root": content_identity(payload_native),
        "native_host_runtime": host_runtime,
        "native_host_runtime_root": host_runtime["host_runtime_cid"],
        "actual_solver_mapping_required": True,
    }
    bundle: dict[str, Any] = {
        "schema": _QUALIFICATION_RUNTIME_BUNDLE_SCHEMA,
        "components": component_summaries,
        "component_count": len(_QUALIFICATION_RUNTIME_COMPONENTS),
        "file_count": len(payload_manifest),
        "total_bytes": sum(int(item["size_bytes"]) for item in payload_manifest),
        "file_manifest_root": content_identity(payload_manifest),
        "omission_manifest_root": content_identity(omissions),
        "python_runtime_binding": python_runtime,
        "native_platform_binding": native_platform,
        "recovery_suite_task_policy": _recovery_suite_task_policy_matrix(),
        "pycache_projected": False,
        "pth_processed": False,
        "plugin_autoload": False,
    }
    bundle["runtime_cid"] = content_identity(bundle)
    return _ResolvedQualificationRuntime(
        bundle=bundle,
        components=tuple(components),
        payload_manifest=tuple(payload_manifest),
        payload_bytes=payload_bytes,
        native_source_observation=host_observation,
    )


def qualification_runtime_bundle_evidence() -> dict[str, Any]:
    """Return the compact observed bundle without projecting or importing it."""

    if _ACTIVE_QUALIFICATION_RUNTIME is not None:
        _validate_active_qualification_runtime(_ACTIVE_QUALIFICATION_RUNTIME)
        return dict(_ACTIVE_QUALIFICATION_RUNTIME.resolved.bundle)
    return dict(_resolve_qualification_runtime().bundle)


def _write_qualification_runtime_file(
    root_fd: int,
    *,
    relative: str,
    payload: bytes,
    mode: int,
) -> None:
    _strict_record_relative_path(relative, noun="qualification runtime projection")
    if mode not in {0o400, 0o500}:
        raise QualificationError("qualification runtime projection mode differs")
    components = relative.split("/")
    directory_fd = os.dup(root_fd)
    try:
        for component in components[:-1]:
            try:
                os.mkdir(component, 0o700, dir_fd=directory_fd)
            except FileExistsError:
                pass
            child_fd = _open_owned_directory_at(
                directory_fd,
                component,
                noun="qualification runtime projection directory",
            )
            os.close(directory_fd)
            directory_fd = child_fd
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(
            components[-1], flags, mode, dir_fd=directory_fd
        )
        try:
            os.fchmod(descriptor, mode)
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise QualificationError(
                        "qualification runtime projection write stalled"
                    )
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(directory_fd)
    except OSError as exc:
        raise QualificationError(
            "qualification runtime projection cannot be created"
        ) from exc
    finally:
        os.close(directory_fd)


def _seal_qualification_runtime_directories(root_fd: int) -> None:
    """Seal every projection directory owner-read/execute after population."""

    def seal(directory_fd: int) -> None:
        child_directories: list[int] = []
        try:
            with os.scandir(directory_fd) as entries:
                for entry in entries:
                    metadata = entry.stat(follow_symlinks=False)
                    if stat.S_ISDIR(metadata.st_mode):
                        child = os.open(
                            entry.name,
                            os.O_RDONLY
                            | os.O_CLOEXEC
                            | os.O_DIRECTORY
                            | getattr(os, "O_NOFOLLOW", 0),
                            dir_fd=directory_fd,
                        )
                        child_directories.append(child)
                    elif not stat.S_ISREG(metadata.st_mode):
                        raise QualificationError(
                            "qualification runtime projection contains a special entry"
                        )
            for child in child_directories:
                seal(child)
                os.close(child)
            os.fchmod(directory_fd, 0o500)
            os.fsync(directory_fd)
        except BaseException:
            for child in child_directories:
                try:
                    os.close(child)
                except OSError:
                    pass
            raise

    seal(root_fd)


def _qualification_projection_expected_files(
    active: _ActiveQualificationRuntime,
) -> dict[str, dict[str, Any]]:
    expected = {
        str(item["path"]): dict(item)
        for item in active.resolved.payload_manifest
    }
    control_bytes = active.resolved.payload_bytes.get(active.control_manifest_path)
    if control_bytes is not None:
        raise QualificationError("qualification control path overlaps payload")
    control = active.projection.get("control_file")
    if not isinstance(control, Mapping):
        raise QualificationError("qualification projection control evidence is absent")
    expected[active.control_manifest_path] = {
        "path": active.control_manifest_path,
        "sha256": control.get("sha256"),
        "size_bytes": control.get("size_bytes"),
        "projection_mode_octal": "0400",
        "native": False,
    }
    return expected


def _validate_qualification_runtime_projection(
    active: _ActiveQualificationRuntime,
) -> None:
    """Reconstruct the exact sealed projection through the retained root fd."""

    root_status = os.fstat(active.root_fd)
    root_identity = active.projection.get("root_identity")
    if (
        not stat.S_ISDIR(root_status.st_mode)
        or root_status.st_uid != os.geteuid()
        or stat.S_IMODE(root_status.st_mode) != 0o500
        or not isinstance(root_identity, Mapping)
        or root_identity
        != {
            "dev": root_status.st_dev,
            "ino": root_status.st_ino,
            "uid": root_status.st_uid,
            "gid": root_status.st_gid,
            "mode": stat.S_IMODE(root_status.st_mode),
            "nlink": root_status.st_nlink,
        }
    ):
        raise QualificationError("qualification runtime projection root differs")
    expected = _qualification_projection_expected_files(active)
    observed: set[str] = set()
    observed_directories: set[str] = set()
    expected_directories: set[str] = set()
    for relative in expected:
        parts = relative.split("/")[:-1]
        expected_directories.update("/".join(parts[:index]) for index in range(1, len(parts) + 1))

    def visit(directory_fd: int, prefix: str) -> None:
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                relative = f"{prefix}/{entry.name}" if prefix else entry.name
                metadata = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(metadata.st_mode):
                    if (
                        relative not in expected_directories
                        or metadata.st_uid != os.geteuid()
                        or stat.S_IMODE(metadata.st_mode) != 0o500
                    ):
                        raise QualificationError(
                            "qualification runtime projection directory differs"
                        )
                    child = os.open(
                        entry.name,
                        os.O_RDONLY
                        | os.O_CLOEXEC
                        | os.O_DIRECTORY
                        | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=directory_fd,
                    )
                    try:
                        opened = os.fstat(child)
                        if (opened.st_dev, opened.st_ino) != (
                            metadata.st_dev,
                            metadata.st_ino,
                        ):
                            raise QualificationError(
                                "qualification projection directory changed"
                            )
                        observed_directories.add(relative)
                        visit(child, relative)
                    finally:
                        os.close(child)
                    continue
                item = expected.get(relative)
                if (
                    item is None
                    or not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or metadata.st_nlink != 1
                    or stat.S_IMODE(metadata.st_mode)
                    != int(str(item["projection_mode_octal"]), 8)
                ):
                    raise QualificationError(
                        "qualification runtime projection file differs"
                    )
                observed.add(relative)

    visit(active.root_fd, "")
    if observed != set(expected) or observed_directories != expected_directories:
        raise QualificationError("qualification runtime projection inventory differs")
    for relative, item in expected.items():
        payload, _observation = _read_qualification_runtime_relative(
            active.root_fd,
            relative,
            noun=f"qualification projected runtime {relative}",
            owner_uid=os.geteuid(),
        )
        if (
            _sha256_bytes(payload) != item["sha256"]
            or len(payload) != item["size_bytes"]
        ):
            raise QualificationError("qualification runtime projection content differs")
    body = {
        key: item
        for key, item in active.projection.items()
        if key != "projection_cid"
    }
    if active.projection.get("projection_cid") != content_identity(body):
        raise QualificationError("qualification runtime projection identity differs")


def _restore_qualification_projection_for_cleanup(root_fd: int) -> None:
    """Make a sealed private tree removable without following links."""

    def restore(directory_fd: int) -> None:
        os.fchmod(directory_fd, 0o700)
        child_directories: list[int] = []
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                metadata = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(metadata.st_mode):
                    child = os.open(
                        entry.name,
                        os.O_RDONLY
                        | os.O_CLOEXEC
                        | os.O_DIRECTORY
                        | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=directory_fd,
                    )
                    child_directories.append(child)
                elif stat.S_ISREG(metadata.st_mode):
                    os.chmod(entry.name, 0o600, dir_fd=directory_fd, follow_symlinks=False)
        for child in child_directories:
            try:
                restore(child)
            finally:
                os.close(child)

    restore(root_fd)


def _build_qualification_runtime_projection(
    resolved: _ResolvedQualificationRuntime,
) -> _ActiveQualificationRuntime:
    directory = tempfile.TemporaryDirectory(prefix="lgcvf-qualification-runtime-")
    root = Path(directory.name).resolve(strict=True)
    os.chmod(root, 0o700)
    root_fd = os.open(
        root,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    control_path = ".lgcvf-runtime/control-manifest.json"
    try:
        for item in resolved.payload_manifest:
            relative = str(item["path"])
            _write_qualification_runtime_file(
                root_fd,
                relative=relative,
                payload=resolved.payload_bytes[relative],
                mode=int(str(item["projection_mode_octal"]), 8),
            )
        control_body: dict[str, Any] = {
            "schema": "lgcvf-qualification-runtime-control-manifest@1",
            "bundle": resolved.bundle,
            "components": list(resolved.components),
            "payload_manifest": list(resolved.payload_manifest),
        }
        control_body["manifest_cid"] = content_identity(control_body)
        control_bytes = json.dumps(
            control_body, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        _write_qualification_runtime_file(
            root_fd,
            relative=control_path,
            payload=control_bytes,
            mode=0o400,
        )
        _seal_qualification_runtime_directories(root_fd)
        root_status = os.fstat(root_fd)
        control_entry = _runtime_manifest_entry(control_path, control_bytes)
        control_entry["projection_mode_octal"] = "0400"
        projection: dict[str, Any] = {
            "schema": _QUALIFICATION_RUNTIME_PROJECTION_SCHEMA,
            "runtime_cid": resolved.bundle["runtime_cid"],
            "bundle_manifest_cid": control_body["manifest_cid"],
            "component_cids": [item["component_cid"] for item in resolved.components],
            "component_count": len(_QUALIFICATION_RUNTIME_COMPONENTS),
            "file_count": len(resolved.payload_manifest),
            "total_bytes": sum(
                int(item["size_bytes"]) for item in resolved.payload_manifest
            ),
            "file_manifest_root": resolved.bundle["file_manifest_root"],
            "control_file_count": 1,
            "control_total_bytes": len(control_bytes),
            "control_file_root": content_identity([control_entry]),
            "control_file": control_entry,
            "root_identity": {
                "dev": root_status.st_dev,
                "ino": root_status.st_ino,
                "uid": root_status.st_uid,
                "gid": root_status.st_gid,
                "mode": stat.S_IMODE(root_status.st_mode),
                "nlink": root_status.st_nlink,
            },
            "regular_files_only": True,
            "single_link_files": True,
            "owner_private": True,
            "contains_symlinks": False,
            "contains_bytecode": False,
            "contains_pth": False,
            "sealed_read_only": True,
        }
        projection["projection_cid"] = content_identity(projection)
        active = _ActiveQualificationRuntime(
            resolved=resolved,
            directory=directory,
            root=root,
            root_fd=root_fd,
            projection=projection,
            control_manifest_path=control_path,
        )
        _validate_qualification_runtime_projection(active)
        return active
    except BaseException:
        try:
            _restore_qualification_projection_for_cleanup(root_fd)
        except BaseException:
            pass
        os.close(root_fd)
        directory.cleanup()
        raise


def _cleanup_active_qualification_runtime() -> None:
    global _ACTIVE_QUALIFICATION_RUNTIME, _ACTIVE_QUALIFICATION_RUNTIME_DEPTH
    active = _ACTIVE_QUALIFICATION_RUNTIME
    _ACTIVE_QUALIFICATION_RUNTIME = None
    _ACTIVE_QUALIFICATION_RUNTIME_DEPTH = 0
    if active is None:
        return
    try:
        _restore_qualification_projection_for_cleanup(active.root_fd)
    finally:
        os.close(active.root_fd)
        active.directory.cleanup()


atexit.register(_cleanup_active_qualification_runtime)


def _validate_active_qualification_runtime(
    active: _ActiveQualificationRuntime,
) -> None:
    """Revalidate the sealed projection and solver host-native boundary."""

    _validate_qualification_runtime_projection(active)
    native_platform = active.resolved.bundle.get("native_platform_binding")
    payload_native = (
        native_platform.get("solver_payload_native_files")
        if isinstance(native_platform, Mapping)
        else None
    )
    if not isinstance(payload_native, list):
        raise QualificationError("qualification native runtime authority is absent")
    observed_host, _observation = _native_host_runtime_binding(payload_native)
    if (
        not isinstance(native_platform, Mapping)
        or native_platform.get("native_host_runtime") != observed_host
        or native_platform.get("native_host_runtime_root")
        != observed_host.get("host_runtime_cid")
    ):
        raise QualificationError("qualification native host runtime differs")
    if (
        active.resolved.bundle.get("recovery_suite_task_policy")
        != _recovery_suite_task_policy_matrix()
    ):
        raise QualificationError("qualification suite task policy differs")


@contextlib.contextmanager
def isolated_qualification_runtime(*, expected_runtime_cid: str) -> Any:
    """Resolve and retain one exact sealed runtime for all six workers."""

    _require_isolated_recovery_runtime()
    global _ACTIVE_QUALIFICATION_RUNTIME, _ACTIVE_QUALIFICATION_RUNTIME_DEPTH
    if _ACTIVE_QUALIFICATION_RUNTIME is None:
        if not _is_canonical_content_cid(expected_runtime_cid):
            raise QualificationError(
                "qualification runtime policy identity is absent or invalid"
            )
        resolved = _resolve_qualification_runtime()
        observed_cid = str(resolved.bundle.get("runtime_cid") or "")
        if not _is_canonical_content_cid(observed_cid):
            raise QualificationError(
                "qualification runtime observed identity is absent or invalid"
            )
        if observed_cid != expected_runtime_cid:
            raise QualificationRuntimeUnavailable(
                reason_code="identity_mismatch",
                phase="pin",
                expected_runtime_cid=expected_runtime_cid,
                observed_runtime_cid=observed_cid,
                component="qualification_runtime",
                detail="qualification runtime differs from policy",
            )
        _ACTIVE_QUALIFICATION_RUNTIME = _build_qualification_runtime_projection(
            resolved
        )
    active = _ACTIVE_QUALIFICATION_RUNTIME
    if active.resolved.bundle.get("runtime_cid") != expected_runtime_cid:
        raise QualificationError("active qualification runtime identity conflicts")
    _validate_active_qualification_runtime(active)
    _ACTIVE_QUALIFICATION_RUNTIME_DEPTH += 1
    try:
        yield active
        _validate_active_qualification_runtime(active)
    finally:
        _ACTIVE_QUALIFICATION_RUNTIME_DEPTH -= 1
        if _ACTIVE_QUALIFICATION_RUNTIME_DEPTH == 0:
            _validate_active_qualification_runtime(active)
            _cleanup_active_qualification_runtime()


def _duckdb_record_rows(
    record_bytes: bytes,
    *,
    noun: str,
) -> dict[str, tuple[str, str]]:
    """Decode one closed DuckDB RECORD path map."""

    rows: dict[str, tuple[str, str]] = {}
    try:
        parsed = csv.reader(io.StringIO(record_bytes.decode("utf-8", errors="strict")))
        for row in parsed:
            if len(row) != 3:
                raise QualificationError(f"{noun} contains a malformed RECORD row")
            relative, digest, size = row
            _strict_record_relative_path(relative, noun=noun)
            if relative in rows:
                raise QualificationError(f"{noun} contains a duplicate RECORD path")
            rows[relative] = (digest, size)
    except (csv.Error, UnicodeDecodeError) as exc:
        raise QualificationError(f"{noun} is not a valid UTF-8 RECORD") from exc
    return rows


def _duckdb_distribution_candidate_names(root_fd: int) -> list[str]:
    """Stream and bound candidate discovery without listing a site root."""

    pattern = re.compile(
        r"duckdb-([0-9A-Za-z][0-9A-Za-z.!+_-]{0,127})\.dist-info"
    )
    names: list[str] = []
    observed_entries = 0
    with os.scandir(root_fd) as entries:
        for entry in entries:
            observed_entries += 1
            if observed_entries > _MAX_DUCKDB_SITE_ROOT_ENTRIES:
                raise QualificationError("DuckDB site root population exceeds its bound")
            if not pattern.fullmatch(entry.name):
                continue
            names.append(entry.name)
            if len(names) > _MAX_DUCKDB_DISTRIBUTION_CANDIDATES:
                raise QualificationError(
                    "DuckDB distribution population exceeds its bound"
                )
    names.sort()
    return names


def _resolve_bound_duckdb_runtime() -> tuple[dict[str, Any], dict[str, bytes]]:
    """Resolve the sole RECORD-bound DuckDB runtime without importing it."""

    matches: list[tuple[dict[str, Any], dict[str, bytes]]] = []
    observed_candidates = 0
    version_pattern = re.compile(
        r"duckdb-([0-9A-Za-z][0-9A-Za-z.!+_-]{0,127})\.dist-info"
    )
    extension_suffix = str(sysconfig.get_config_var("EXT_SUFFIX") or "")
    soabi = str(sysconfig.get_config_var("SOABI") or "")
    if (
        not extension_suffix
        or extension_suffix not in importlib.machinery.EXTENSION_SUFFIXES
        or not soabi
    ):
        raise QualificationError("Python native-extension identity is unavailable")
    for root in _pytest_site_roots():
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            root_fd = os.open(root, flags)
        except OSError as exc:
            raise QualificationError("DuckDB distribution root cannot be opened") from exc
        try:
            root_status = os.fstat(root_fd)
            if not stat.S_ISDIR(root_status.st_mode) or root_status.st_uid not in {
                0,
                os.geteuid(),
            }:
                raise QualificationError("DuckDB distribution root is not owned")
            candidates = _duckdb_distribution_candidate_names(root_fd)
            observed_candidates += len(candidates)
            if observed_candidates > _MAX_DUCKDB_DISTRIBUTION_CANDIDATES:
                raise QualificationError(
                    "DuckDB distribution population exceeds its bound"
                )
            for distribution_name in candidates:
                candidate_fd = _open_owned_directory_at(
                    root_fd,
                    distribution_name,
                    noun="DuckDB distribution directory",
                )
                try:
                    metadata_bytes = _read_owned_regular_at(
                        candidate_fd,
                        "METADATA",
                        noun="DuckDB distribution METADATA",
                        limit=_MAX_PYTEST_METADATA_BYTES,
                    )
                    record_bytes = _read_owned_regular_at(
                        candidate_fd,
                        "RECORD",
                        noun="DuckDB distribution RECORD",
                        limit=_MAX_PYTEST_RECORD_BYTES,
                    )
                finally:
                    os.close(candidate_fd)
                try:
                    metadata = email.parser.BytesParser(
                        policy=email.policy.compat32
                    ).parsebytes(metadata_bytes, headersonly=True)
                except (TypeError, ValueError) as exc:
                    raise QualificationError(
                        "DuckDB distribution METADATA is malformed"
                    ) from exc
                names_found = metadata.get_all("Name", [])
                versions_found = metadata.get_all("Version", [])
                version_match = version_pattern.fullmatch(distribution_name)
                version = str(versions_found[0]).strip() if versions_found else ""
                if (
                    metadata.defects
                    or len(names_found) != 1
                    or len(versions_found) != 1
                    or re.sub(r"[-_.]+", "-", str(names_found[0])).casefold()
                    != "duckdb"
                    or version_match is None
                    or version != version_match.group(1)
                ):
                    raise QualificationError("DuckDB distribution identity differs")

                rows = _duckdb_record_rows(
                    record_bytes,
                    noun=f"DuckDB {version} RECORD",
                )
                metadata_path = f"{distribution_name}/METADATA"
                if metadata_path not in rows:
                    raise QualificationError(
                        "DuckDB RECORD does not bind its exact METADATA"
                    )
                metadata_identity = _record_sha256_size(
                    *rows[metadata_path], noun="DuckDB METADATA"
                )
                if metadata_identity != (
                    hashlib.sha256(metadata_bytes).digest(),
                    len(metadata_bytes),
                ):
                    raise QualificationError("DuckDB METADATA identity differs")

                native_candidates = sorted(
                    path
                    for path in rows
                    if "/" not in path
                    and path.startswith("_duckdb")
                    and path.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES))
                )
                expected_native_path = f"_duckdb{extension_suffix}"
                if native_candidates != [expected_native_path]:
                    raise QualificationError(
                        "DuckDB RECORD does not bind one native runtime"
                    )
                runtime_paths = (*_DUCKDB_RUNTIME_SOURCE_PATHS, expected_native_path)
                runtime_files: list[dict[str, Any]] = []
                runtime_bytes: dict[str, bytes] = {}
                candidate_matches = True
                for relative in runtime_paths:
                    if relative not in rows:
                        raise QualificationError(
                            f"DuckDB RECORD omits runtime file: {relative}"
                        )
                    expected_identity = _record_sha256_size(
                        *rows[relative], noun=f"DuckDB runtime {relative}"
                    )
                    observed = _read_owned_regular_relative(
                        root_fd,
                        relative,
                        noun=f"DuckDB runtime {relative}",
                        limit=_MAX_DUCKDB_RUNTIME_FILE_BYTES,
                    )
                    observed_identity = (hashlib.sha256(observed).digest(), len(observed))
                    if expected_identity != observed_identity:
                        candidate_matches = False
                        break
                    runtime_bytes[relative] = observed
                    runtime_files.append(
                        {
                            "path": relative,
                            "sha256": "sha256:" + observed_identity[0].hex(),
                            "size_bytes": observed_identity[1],
                        }
                    )
                if not candidate_matches:
                    continue
                runtime_files.sort(key=lambda item: str(item["path"]))
                projected_bytes = {
                    **runtime_bytes,
                    metadata_path: metadata_bytes,
                    f"{distribution_name}/RECORD": record_bytes,
                }
                body: dict[str, Any] = {
                    "schema": "lgcvf-bound-duckdb-runtime@1",
                    "distribution_name": "duckdb",
                    "distribution_directory": distribution_name,
                    "version": version,
                    "python_cache_tag": str(sys.implementation.cache_tag),
                    "python_soabi": soabi,
                    "native_extension_suffix": extension_suffix,
                    "machine": platform.machine(),
                    "platform": platform.system(),
                    "metadata": {
                        "path": metadata_path,
                        "sha256": _sha256_bytes(metadata_bytes),
                        "size_bytes": len(metadata_bytes),
                    },
                    "record": {
                        "path": f"{distribution_name}/RECORD",
                        "sha256": _sha256_bytes(record_bytes),
                        "size_bytes": len(record_bytes),
                    },
                    "runtime_files": runtime_files,
                    "native_extension_path": expected_native_path,
                    "projected_site_root_only": True,
                    "pycache_projected": False,
                    "pth_processed": False,
                }
                body["runtime_cid"] = content_identity(body)
                matches.append((body, projected_bytes))
        finally:
            os.close(root_fd)
    if observed_candidates != 1 or len(matches) != 1:
        raise QualificationError(
            "DuckDB runtime does not have one exact RECORD-matched distribution"
        )
    return matches[0]


def bound_duckdb_runtime_evidence() -> dict[str, Any]:
    """Return the content-bound DuckDB runtime identity without importing it."""

    evidence, _files = _resolve_bound_duckdb_runtime()
    return evidence


def _require_isolated_recovery_runtime() -> None:
    """Reject protected recovery work outside ``python -I -S -B``."""

    flags = sys.flags
    try:
        cpython_flags = tuple(
            ctypes.c_int.in_dll(ctypes.pythonapi, name).value
            for name in (
                "Py_IsolatedFlag",
                "Py_IgnoreEnvironmentFlag",
                "Py_NoSiteFlag",
                "Py_DontWriteBytecodeFlag",
            )
        )
    except (AttributeError, OSError, ValueError):
        cpython_flags = ()
    if (
        flags.isolated != 1
        or flags.ignore_environment != 1
        or flags.no_site != 1
        or flags.safe_path is not True
        or flags.dont_write_bytecode != 1
        or sys.dont_write_bytecode is not True
        or cpython_flags != (1, 1, 1, 1)
    ):
        raise QualificationError(
            "protected recovery requires python -I -S -B"
        )
    _require_isolated_recovery_pycache()
    if _RECOVERY_WORKER_BOOTSTRAP_CAPSULE is not None:
        _validate_recovery_worker_bootstrap()
        return
    if (
        _ISOLATED_RECOVERY_SOURCE_IDENTITY is None
        or _clean_recovery_import_source(ROOT)
        != _ISOLATED_RECOVERY_SOURCE_IDENTITY
        or
        _ISOLATED_RECOVERY_IMPORT_INVENTORY is None
        or _scan_isolated_recovery_import_roots(ROOT)
        != _ISOLATED_RECOVERY_IMPORT_INVENTORY
        or _ISOLATED_RECOVERY_DATASETS_SOURCE_IDENTITY is None
        or _datasets_recovery_import_source(ROOT)
        != _ISOLATED_RECOVERY_DATASETS_SOURCE_IDENTITY
        or _ISOLATED_RECOVERY_DATASETS_IMPORT_INVENTORY is None
        or _scan_isolated_recovery_import_roots(
            ROOT / "ipfs_datasets_py",
            roots=(),
            tracked_pathspecs=(".",),
            whole_repository=True,
            root_import_candidates=False,
        )
        != _ISOLATED_RECOVERY_DATASETS_IMPORT_INVENTORY
    ):
        raise QualificationError("protected recovery import inventory differs")


def _require_isolated_recovery_pycache() -> None:
    """Require the early, owner-private, empty bytecode lookup root."""

    root = _ISOLATED_RECOVERY_PYCACHE_ROOT
    identity = _ISOLATED_RECOVERY_PYCACHE_IDENTITY
    capsule = _ISOLATED_RECOVERY_PYCACHE_CAPSULE
    if (
        _ISOLATED_RECOVERY_PYCACHE_DIRECTORY is None
        or root is None
        or identity is None
        or capsule is None
        or sys.pycache_prefix != str(root)
    ):
        raise QualificationError("protected recovery pycache isolation differs")
    try:
        observed_directory, observed_root, observed_identity = (
            _validated_isolated_recovery_pycache_capsule(capsule)
        )
    except RuntimeError as exc:
        raise QualificationError(
            str(exc)
        ) from exc
    if (
        observed_directory is not _ISOLATED_RECOVERY_PYCACHE_DIRECTORY
        or observed_root != root
        or observed_identity != identity
    ):
        raise QualificationError("protected recovery pycache isolation differs")


def _duckdb_module_names() -> set[str]:
    return {
        name
        for name in sys.modules
        if name == "duckdb"
        or name.startswith("duckdb.")
        or name == "_duckdb"
        or name.startswith("_duckdb.")
    }


def _write_bound_duckdb_projection(
    root_fd: int,
    *,
    relative: str,
    payload: bytes,
    executable: bool,
) -> None:
    """Create one capsule file through a no-follow directory walk."""

    _strict_record_relative_path(relative, noun="DuckDB projected runtime")
    components = relative.split("/")
    directory_fd = os.dup(root_fd)
    try:
        for component in components[:-1]:
            try:
                os.mkdir(component, 0o700, dir_fd=directory_fd)
            except FileExistsError:
                pass
            child_fd = _open_owned_directory_at(
                directory_fd,
                component,
                noun="DuckDB projected runtime directory",
            )
            os.fchmod(child_fd, 0o700)
            os.close(directory_fd)
            directory_fd = child_fd
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        mode = 0o500 if executable else 0o400
        try:
            descriptor = os.open(
                components[-1],
                flags,
                mode,
                dir_fd=directory_fd,
            )
        except OSError as exc:
            raise QualificationError("DuckDB runtime projection cannot be created") from exc
        try:
            os.fchmod(descriptor, mode)
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise QualificationError("DuckDB runtime projection write stalled")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _bound_duckdb_projection_expected_files(
    evidence: Mapping[str, Any],
) -> dict[str, tuple[str, int, int]]:
    """Return the exact content and mode contract for one admitted capsule."""

    expected: dict[str, tuple[str, int, int]] = {}
    native_path = str(evidence.get("native_extension_path") or "")
    runtime_files = evidence.get("runtime_files")
    if not isinstance(runtime_files, list):
        raise QualificationError("DuckDB runtime file evidence is absent")
    for value in runtime_files:
        if not isinstance(value, Mapping) or set(value) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise QualificationError("DuckDB runtime file evidence differs")
        relative = str(value["path"])
        if relative in expected:
            raise QualificationError("DuckDB runtime file evidence is duplicated")
        expected[relative] = (
            str(value["sha256"]),
            int(value["size_bytes"]),
            0o500 if relative == native_path else 0o400,
        )
    for evidence_field in ("metadata", "record"):
        value = evidence.get(evidence_field)
        if not isinstance(value, Mapping) or set(value) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise QualificationError(f"DuckDB {evidence_field} evidence differs")
        relative = str(value["path"])
        if relative in expected:
            raise QualificationError("DuckDB projected runtime path is duplicated")
        expected[relative] = (
            str(value["sha256"]),
            int(value["size_bytes"]),
            0o400,
        )
    expected_runtime_paths = set(_DUCKDB_RUNTIME_SOURCE_PATHS) | {native_path}
    if set(expected) - {
        str(evidence["metadata"]["path"]),
        str(evidence["record"]["path"]),
    } != expected_runtime_paths:
        raise QualificationError("DuckDB projected runtime file set differs")
    return expected


def _validate_bound_duckdb_projection(
    projection: Path,
    evidence: Mapping[str, Any],
) -> None:
    """Revalidate the exact private capsule inventory and content."""

    try:
        lexical = projection.lstat()
    except OSError as exc:
        raise QualificationError("DuckDB runtime projection is unavailable") from exc
    if (
        not stat.S_ISDIR(lexical.st_mode)
        or lexical.st_uid != os.geteuid()
        or stat.S_IMODE(lexical.st_mode) != 0o700
    ):
        raise QualificationError("DuckDB runtime projection root differs")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        root_fd = os.open(projection, flags)
    except OSError as exc:
        raise QualificationError("DuckDB runtime projection cannot be opened") from exc
    try:
        opened = os.fstat(root_fd)
        if (opened.st_dev, opened.st_ino) != (lexical.st_dev, lexical.st_ino):
            raise QualificationError("DuckDB runtime projection changed while opened")
        expected_files = _bound_duckdb_projection_expected_files(evidence)
        expected_directories = {
            "/".join(PurePath(relative).parts[:ordinal])
            for relative in expected_files
            for ordinal in range(1, len(PurePath(relative).parts))
        }
        observed_files: set[str] = set()
        observed_directories: set[str] = set()

        def inspect(directory_fd: int, prefix: str) -> None:
            with os.scandir(directory_fd) as entries:
                for entry in entries:
                    relative = f"{prefix}/{entry.name}" if prefix else entry.name
                    metadata = entry.stat(follow_symlinks=False)
                    if stat.S_ISDIR(metadata.st_mode):
                        if (
                            relative not in expected_directories
                            or metadata.st_uid != os.geteuid()
                            or stat.S_IMODE(metadata.st_mode) != 0o700
                        ):
                            raise QualificationError(
                                "DuckDB runtime projection directory differs"
                            )
                        child_fd = _open_owned_directory_at(
                            directory_fd,
                            entry.name,
                            noun="DuckDB projected runtime directory",
                        )
                        try:
                            child = os.fstat(child_fd)
                            if (child.st_dev, child.st_ino) != (
                                metadata.st_dev,
                                metadata.st_ino,
                            ):
                                raise QualificationError(
                                    "DuckDB projection directory changed while opened"
                                )
                            observed_directories.add(relative)
                            inspect(child_fd, relative)
                        finally:
                            os.close(child_fd)
                        continue
                    if not stat.S_ISREG(metadata.st_mode):
                        raise QualificationError(
                            "DuckDB runtime projection contains a special entry"
                        )
                    expected = expected_files.get(relative)
                    if (
                        expected is None
                        or metadata.st_uid != os.geteuid()
                        or metadata.st_nlink != 1
                        or stat.S_IMODE(metadata.st_mode) != expected[2]
                    ):
                        raise QualificationError("DuckDB runtime projection file differs")
                    observed_files.add(relative)

        inspect(root_fd, "")
        if observed_files != set(expected_files) or observed_directories != expected_directories:
            raise QualificationError("DuckDB runtime projection inventory differs")
        for relative, (digest, size, _mode) in expected_files.items():
            payload = _read_owned_regular_relative(
                root_fd,
                relative,
                noun=f"DuckDB projected runtime {relative}",
                limit=_MAX_DUCKDB_RUNTIME_FILE_BYTES,
            )
            if _sha256_bytes(payload) != digest or len(payload) != size:
                raise QualificationError("DuckDB runtime projection content differs")
    finally:
        os.close(root_fd)


def _module_file(module: Any) -> Path | None:
    value = getattr(module, "__file__", None)
    if not isinstance(value, str) or not value:
        return None
    try:
        return Path(value).resolve(strict=True)
    except OSError as exc:
        raise QualificationError("imported module origin is unavailable") from exc


def _validate_bound_duckdb_modules(
    projection: Path,
    evidence: Mapping[str, Any],
) -> None:
    """Require all admitted native/package modules to originate in the capsule."""

    observed_names = _duckdb_module_names()
    if observed_names != _DUCKDB_RUNTIME_MODULE_NAMES:
        raise QualificationError("bound DuckDB runtime module set differs")
    if _ACTIVE_DUCKDB_RUNTIME_MODULES is not None and any(
        sys.modules[name] is not module
        for name, module in _ACTIVE_DUCKDB_RUNTIME_MODULES.items()
    ):
        raise QualificationError("bound DuckDB runtime module identity differs")
    duckdb = sys.modules.get("duckdb")
    native = sys.modules.get("_duckdb")
    if duckdb is None or native is None:
        raise QualificationError("bound DuckDB runtime modules are absent")
    projection = projection.resolve(strict=True)
    package_origin = _module_file(duckdb)
    native_origin = _module_file(native)
    expected_package = (projection / "duckdb/__init__.py").resolve(strict=True)
    expected_native = (
        projection / str(evidence["native_extension_path"])
    ).resolve(strict=True)
    if (
        str(getattr(duckdb, "__version__", "")) != evidence.get("version")
        or package_origin != expected_package
        or native_origin != expected_native
    ):
        raise QualificationError("imported DuckDB runtime identity differs")
    allowed_python = {
        (projection / relative).resolve(strict=True)
        for relative in _DUCKDB_RUNTIME_SOURCE_PATHS
    }
    for name in observed_names:
        origin = _module_file(sys.modules[name])
        if origin is None:
            raise QualificationError(f"DuckDB module has no capsule origin: {name}")
        if origin != expected_native and origin not in allowed_python:
            raise QualificationError(f"DuckDB module escaped its capsule: {name}")
        cached = getattr(sys.modules[name], "__cached__", None)
        if isinstance(cached, str) and Path(cached).exists():
            raise QualificationError("DuckDB runtime loaded a bytecode cache")


def _closed_import_meta_path() -> list[Any]:
    return [
        importlib.machinery.BuiltinImporter,
        importlib.machinery.FrozenImporter,
        importlib.machinery.PathFinder,
    ]


@contextlib.contextmanager
def _bound_duckdb_import_scope(
    projection: Path,
    *,
    hidden_module_roots: frozenset[str] = frozenset(),
) -> Any:
    original_path = list(sys.path)
    original_meta_path = list(sys.meta_path)
    original_dont_write_bytecode = sys.dont_write_bytecode
    site_roots = _pytest_site_roots()

    def ambient_path(value: str) -> bool:
        if not value:
            return False
        try:
            resolved = Path(value).resolve(strict=True)
        except OSError:
            return False
        return any(
            resolved == root or resolved.is_relative_to(root) for root in site_roots
        )

    hidden_modules: dict[str, Any] = {}
    for name, module in tuple(sys.modules.items()):
        if name.partition(".")[0] in hidden_module_roots:
            hidden_modules[name] = module
            sys.modules.pop(name, None)
    sys.path[:] = [
        str(projection),
        *[
            value
            for value in original_path
            if value != str(projection) and not ambient_path(value)
        ],
    ]
    sys.meta_path[:] = _closed_import_meta_path()
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        for name in hidden_modules:
            replacement = sys.modules.get(name)
            if replacement is not None and replacement is not hidden_modules[name]:
                sys.modules.pop(name, None)
        sys.modules.update(hidden_modules)
        sys.path[:] = original_path
        sys.meta_path[:] = original_meta_path
        sys.dont_write_bytecode = original_dont_write_bytecode


def _cleanup_bound_duckdb_runtime() -> None:
    global _ACTIVE_DUCKDB_RUNTIME_DIRECTORY, _ACTIVE_DUCKDB_RUNTIME_MODULES
    global _ACTIVE_DUCKDB_RUNTIME_PROJECTION

    directory = _ACTIVE_DUCKDB_RUNTIME_DIRECTORY
    _ACTIVE_DUCKDB_RUNTIME_DIRECTORY = None
    _ACTIVE_DUCKDB_RUNTIME_MODULES = None
    _ACTIVE_DUCKDB_RUNTIME_PROJECTION = None
    if directory is not None:
        directory.cleanup()


atexit.register(_cleanup_bound_duckdb_runtime)


@contextlib.contextmanager
def isolated_bound_duckdb_runtime(
    *,
    expected_runtime_cid: str,
) -> Any:
    """Admit one process-lifetime content-bound DuckDB runtime capsule."""

    _require_isolated_recovery_runtime()
    global _ACTIVE_DUCKDB_RUNTIME_CID, _ACTIVE_DUCKDB_RUNTIME_DIRECTORY
    global _ACTIVE_DUCKDB_RUNTIME_EVIDENCE, _ACTIVE_DUCKDB_RUNTIME_PROJECTION
    global _ACTIVE_DUCKDB_RUNTIME_MODULES

    evidence, projected = _resolve_bound_duckdb_runtime()
    if not expected_runtime_cid or evidence.get("runtime_cid") != expected_runtime_cid:
        raise QualificationError("DuckDB runtime differs from the configured identity")
    if _ACTIVE_DUCKDB_RUNTIME_CID is None:
        if _duckdb_module_names():
            raise QualificationError(
                "DuckDB runtime was preloaded before isolated admission"
            )
        optional_roots = frozenset(
            {
                "numpy",
                "pandas",
                "pyarrow",
                "polars",
                "adbc_driver_manager",
                "adbc_driver_duckdb",
            }
        )
        imported_before = set(sys.modules)
        directory = tempfile.TemporaryDirectory(prefix="lgcvf-bound-duckdb-")
        projection = Path(directory.name)
        os.chmod(projection, 0o700)
        try:
            flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            root_fd = os.open(projection, flags)
            try:
                for relative, payload in sorted(projected.items()):
                    _write_bound_duckdb_projection(
                        root_fd,
                        relative=relative,
                        payload=payload,
                        executable=relative == evidence["native_extension_path"],
                    )
                os.fsync(root_fd)
            finally:
                os.close(root_fd)
            _validate_bound_duckdb_projection(projection, evidence)
            with _bound_duckdb_import_scope(
                projection,
                hidden_module_roots=optional_roots,
            ):
                importlib.invalidate_caches()
                importlib.import_module("duckdb")
                _validate_bound_duckdb_modules(projection, evidence)
                new_optional = optional_roots & set(sys.modules)
                if new_optional:
                    raise QualificationError(
                        "DuckDB runtime imported an optional dependency"
                    )
                site_roots = _pytest_site_roots()
                for name in set(sys.modules) - imported_before:
                    origin = _module_file(sys.modules[name])
                    if origin is not None and any(
                        origin.is_relative_to(root) for root in site_roots
                    ):
                        raise QualificationError(
                            f"DuckDB admission imported ambient package: {name}"
                        )
            _ACTIVE_DUCKDB_RUNTIME_CID = expected_runtime_cid
            _ACTIVE_DUCKDB_RUNTIME_EVIDENCE = dict(evidence)
            _ACTIVE_DUCKDB_RUNTIME_DIRECTORY = directory
            _ACTIVE_DUCKDB_RUNTIME_MODULES = {
                name: sys.modules[name] for name in _DUCKDB_RUNTIME_MODULE_NAMES
            }
            _ACTIVE_DUCKDB_RUNTIME_PROJECTION = projection
            with _bound_duckdb_import_scope(projection):
                yield dict(evidence)
                _validate_bound_duckdb_modules(projection, evidence)
                _validate_bound_duckdb_projection(projection, evidence)
            return
        except BaseException:
            if _ACTIVE_DUCKDB_RUNTIME_CID is None:
                _ACTIVE_DUCKDB_RUNTIME_EVIDENCE = None
                _ACTIVE_DUCKDB_RUNTIME_DIRECTORY = None
                _ACTIVE_DUCKDB_RUNTIME_MODULES = None
                _ACTIVE_DUCKDB_RUNTIME_PROJECTION = None
                for name in _duckdb_module_names() - imported_before:
                    sys.modules.pop(name, None)
                directory.cleanup()
            raise
    if (
        _ACTIVE_DUCKDB_RUNTIME_CID != expected_runtime_cid
        or _ACTIVE_DUCKDB_RUNTIME_EVIDENCE != evidence
        or _ACTIVE_DUCKDB_RUNTIME_PROJECTION is None
        or _ACTIVE_DUCKDB_RUNTIME_DIRECTORY is None
        or _ACTIVE_DUCKDB_RUNTIME_MODULES is None
    ):
        raise QualificationError("active DuckDB runtime identity differs")
    projection = _ACTIVE_DUCKDB_RUNTIME_PROJECTION
    _validate_bound_duckdb_projection(projection, evidence)
    _validate_bound_duckdb_modules(projection, evidence)
    with _bound_duckdb_import_scope(projection):
        yield dict(evidence)
        _validate_bound_duckdb_modules(projection, evidence)
        _validate_bound_duckdb_projection(projection, evidence)


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
            "pytest_version": _pytest_distribution_version(),
        },
    }
    body["fingerprint_cid"] = content_identity(body)
    return body


def _checkout_fingerprint(
    manifests: Sequence[Mapping[str, Any]], *, root: Path = ROOT
) -> dict[str, Any]:
    """Backward-compatible name for the protected semantic input projection."""

    return _protected_input_projection(manifests, root=root)


def _matches_recovery_provider_import(name: object) -> bool:
    if not isinstance(name, str):
        return False
    normalized = name.strip().casefold()
    return any(
        normalized == prefix or normalized.startswith(prefix + ".")
        for prefix in _FORBIDDEN_RECOVERY_PROVIDER_IMPORTS
    )


class _RecoveryProviderImportFinder(importlib.abc.MetaPathFinder):
    """Deny a provider import before its loader can execute side effects."""

    def __init__(self, attempts: list[str]) -> None:
        self._attempts = attempts

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: object | None = None,
    ) -> None:
        del path, target
        if _matches_recovery_provider_import(fullname):
            self._attempts.append(fullname)
            raise QualificationError(
                f"recovery provider import denied before execution: {fullname}"
            )
        return None


class _RecoveryProviderGuard:
    """Process-local import/CLI guard supplementing the OS network boundary."""

    def __init__(self, *, retain_until_process_exit: bool = False) -> None:
        self.import_attempts: list[str] = []
        self.process_attempts: list[str] = []
        self._finder = _RecoveryProviderImportFinder(self.import_attempts)
        self._active = False
        self._retain_until_process_exit = retain_until_process_exit

    @staticmethod
    def policy() -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": "lgcvf-recovery-provider-denial-policy@1",
            "provider_route": "none",
            "network_client_required": False,
            "forbidden_import_prefixes": list(_FORBIDDEN_RECOVERY_PROVIDER_IMPORTS),
            "forbidden_executable_names": list(
                _FORBIDDEN_RECOVERY_PROVIDER_EXECUTABLES
            ),
            "import_audit_hook_installed": True,
            "import_meta_path_guard_installed": True,
            "subprocess_audit_hook_installed": True,
            "os_system_denied": True,
            "provider_import_observation_scope": "qualification_worker_process_only",
            "descendant_network_denial": "inherited_seccomp_and_landlock",
        }
        value["policy_cid"] = content_identity(value)
        return value

    def _audit(self, event: str, arguments: tuple[Any, ...]) -> None:
        if not self._active:
            return
        if event == "import" and arguments:
            name = arguments[0]
            if _matches_recovery_provider_import(name):
                normalized = str(name)
                self.import_attempts.append(normalized)
                raise QualificationError(
                    f"recovery provider import denied by audit hook: {normalized}"
                )
        if event not in {
            "os.exec",
            "os.posix_spawn",
            "os.spawn",
            "os.system",
            "subprocess.Popen",
        } or not arguments:
            return
        if event == "os.system":
            self.process_attempts.append("<os.system>")
            raise QualificationError("recovery shell process denied by audit hook")
        tokens: list[str] = []
        for raw in arguments[:2]:
            values: Sequence[object]
            if isinstance(raw, (list, tuple)):
                values = raw
            else:
                values = (raw,)
            for value in values:
                if isinstance(value, bytes):
                    tokens.append(value.decode("utf-8", errors="replace"))
                elif isinstance(value, (str, os.PathLike)):
                    tokens.append(os.fspath(value))
        forbidden = next(
            (
                PurePath(token.strip()).name.casefold()
                for token in tokens
                if token.strip()
                and PurePath(token.strip()).name.casefold()
                in _FORBIDDEN_RECOVERY_PROVIDER_EXECUTABLES
            ),
            "",
        )
        provider_module = next(
            (
                tokens[index + 1]
                for index, token in enumerate(tokens[:-1])
                if token == "-m"
                and _matches_recovery_provider_import(tokens[index + 1])
            ),
            "",
        )
        denied = forbidden or (f"python:-m:{provider_module}" if provider_module else "")
        if denied:
            self.process_attempts.append(denied)
            raise QualificationError(
                f"recovery provider process denied by audit hook: {denied}"
            )

    def __enter__(self) -> _RecoveryProviderGuard:
        already_loaded = sorted(
            name for name in sys.modules if _matches_recovery_provider_import(name)
        )
        if already_loaded:
            raise QualificationError(
                "recovery worker started with forbidden provider modules loaded: "
                + ", ".join(already_loaded[:10])
            )
        self._active = True
        sys.addaudithook(self._audit)
        sys.meta_path.insert(0, self._finder)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if self._retain_until_process_exit:
            return False
        self._active = False
        try:
            sys.meta_path.remove(self._finder)
        except ValueError:
            # Removal by judged code is itself evidence that the policy was
            # tampered with and must reject the observation.
            self.import_attempts.append("<meta_path_guard_removed>")
        return False

    def imported_modules(self) -> list[str]:
        return sorted(
            name for name in sys.modules if _matches_recovery_provider_import(name)
        )


class _RecoveryZ3ImportDenialFinder(importlib.abc.MetaPathFinder):
    """Give the intended 061 import the ordinary typed-unavailable shape."""

    def __init__(self, guard: _RecoveryZ3ImportDenialGuard) -> None:
        self._guard = guard

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: object | None = None,
    ) -> None:
        del path, target
        if fullname == "z3" or fullname.startswith("z3."):
            self._guard.deny_meta_import(fullname)
        return None


class _RecoveryZ3ImportDenialGuard:
    """Deny 061's Z3 namespace before loading, with an irreversible file boundary."""

    def __init__(
        self,
        *,
        task_id: str,
        suite_id: str,
        runtime_root: Path,
        recorder: _Recorder,
        source_projection_root: str,
        provider_guard: _RecoveryProviderGuard,
    ) -> None:
        self.task_id = task_id
        self.suite_id = suite_id
        self.runtime_root = runtime_root.resolve(strict=True)
        self.z3_root = (self.runtime_root / "z3").resolve(strict=True)
        if not self.z3_root.is_relative_to(self.runtime_root):
            raise QualificationError("recovery Z3 namespace escaped its runtime")
        self.policy = _recovery_z3_import_denial_policy(
            task_id=task_id,
            suite_id=suite_id,
        )
        self.enabled = task_id in _RECOVERY_Z3_IMPORT_DENIED_TASKS
        if not _is_canonical_content_cid(source_projection_root):
            raise QualificationError(
                "recovery Z3 source projection identity differs"
            )
        self.source_projection_root = source_projection_root
        self.provider_guard = provider_guard
        self.recorder = recorder
        self.meta_denials: list[dict[str, Any]] = []
        self.open_boundary_denials: list[dict[str, Any]] = []
        self._finder = _RecoveryZ3ImportDenialFinder(self)
        self._meta_path_before: tuple[object, ...] | None = None
        self._pytest_candidate_meta_path: tuple[object, ...] | None = None
        self._meta_path_active = False
        self._meta_path_restored = not self.enabled
        self._pytest_meta_path_admission_count = 0
        self._pytest_meta_path_call_validation_count = 0
        self._pytest_meta_path_sessionfinish_count = 0
        self._pytest_meta_path_unconfigure_count = 0
        self._pytest_meta_path_return_restoration_count = 0
        self._pytest_meta_path_candidate_tuple_validated = False
        self._pytest_meta_path_bootstrap_tuple_restored = False
        self._audit_installed = False
        self._owner_thread_ident = threading.get_ident()
        self._owner_native_thread_id = threading.get_native_id()
        self._trusted_revalidation_depth = 0
        self._trusted_revalidation_open_count = 0
        self._trusted_revalidation_scope_entry_count = 0
        self._trusted_revalidation_scope_exit_count = 0
        self._trusted_revalidation_scope_completed = False
        self._trusted_source_revalidation_depth = 0
        self._trusted_source_revalidation_scope_entry_count = 0
        self._trusted_source_revalidation_scope_exit_count = 0
        self._trusted_source_revalidation_scope_completed = False
        self._trusted_source_pending_event: dict[str, Any] | None = None
        self._trusted_source_pending_audit_consumed = False
        self._trusted_source_confirmation_count = 0
        self._trusted_source_events: list[dict[str, Any]] = []
        self._trusted_source_failed = False
        self._trusted_source_reader_code = (
            _read_recovery_projection_source.__code__
        )
        self._trusted_source_caller_code_identity_validated = False
        self._candidate_expected_threads: list[dict[str, Any]] | None = None
        self._candidate_expected_tasks: list[int] | None = None
        self._candidate_expected_children: list[int] | None = None
        self._owner_phase = (
            "candidate_execution" if self.enabled else "not_applicable"
        )
        recorder.bind_z3_import_denial_guard(self)

    @staticmethod
    def _loaded_z3_modules() -> list[str]:
        return sorted(
            name
            for name in sys.modules
            if name == "z3" or name.startswith("z3.")
        )

    def _path_targets_z3(self, target: object) -> tuple[bool, str]:
        if isinstance(target, bytes):
            decoded = os.fsdecode(target)
        elif isinstance(target, (str, os.PathLike)):
            decoded = os.fspath(target)
            if isinstance(decoded, bytes):
                decoded = os.fsdecode(decoded)
        else:
            return False, ""
        if not isinstance(decoded, str) or not decoded or "\x00" in decoded:
            return False, ""
        normalized_relative = os.path.normpath(decoded)
        relative_parts = PurePath(normalized_relative).parts
        relative_z3 = (
            bool(relative_parts)
            and relative_parts[0] == "z3"
            and all(part not in {"", ".", ".."} for part in relative_parts)
        )
        try:
            resolved = Path(decoded).resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            resolved = Path(os.path.abspath(decoded))
        under_z3 = resolved == self.z3_root or resolved.is_relative_to(self.z3_root)
        if not (relative_z3 or under_z3):
            return False, ""
        if under_z3:
            try:
                suffix = resolved.relative_to(self.z3_root).as_posix()
            except ValueError:
                suffix = ""
            return True, "runtime:z3" + (("/" + suffix) if suffix else "")
        return True, "relative:" + normalized_relative[:256]

    def _expected_trusted_source_events(self) -> list[dict[str, Any]]:
        """Close policy templates over this worker's source projection root."""

        return [
            {**dict(item), "source_projection_root": self.source_projection_root}
            for item in self.policy["expected_trusted_source_events"]
        ]

    def _audit(self, event: str, arguments: tuple[Any, ...]) -> None:
        if not self.enabled or event not in {"open", "open_code"} or not arguments:
            return
        denied, token = self._path_targets_z3(arguments[0])
        if not denied:
            return
        pending = self._trusted_source_pending_event
        if pending is not None:
            try:
                caller = sys._getframe(1)
            except (ValueError, RuntimeError) as exc:
                self._trusted_source_failed = True
                raise QualificationError(
                    "recovery trusted source caller is unavailable"
                ) from exc
            exact_source_open = (
                self._owner_phase == "source_revalidation_pending"
                and self._trusted_source_revalidation_depth == 1
                and not self._trusted_source_failed
                and threading.get_ident() == self._owner_thread_ident
                and threading.get_native_id() == self._owner_native_thread_id
                and event == "open"
                and len(arguments) == 3
                and type(arguments[0]) is str
                and arguments[0] == "z3"
                and arguments[1] is None
                and type(arguments[2]) is int
                and arguments[2] == _recovery_projection_directory_flags()
                and caller.f_code is self._trusted_source_reader_code
                and pending
                == self._expected_trusted_source_events()[
                    len(self._trusted_source_events)
                ]
            )
            if not exact_source_open:
                self._trusted_source_failed = True
                raise QualificationError(
                    "recovery trusted source audit capability differs"
                )
            self._trusted_source_pending_event = None
            self._trusted_source_pending_audit_consumed = True
            self._trusted_source_caller_code_identity_validated = True
            self._trusted_source_events.append(dict(pending))
            return
        if (
            self._owner_phase == "post_candidate_revalidation"
            and self._trusted_revalidation_depth == 1
            and threading.get_ident() == self._owner_thread_ident
            and threading.get_native_id() == self._owner_native_thread_id
        ):
            if self._trusted_revalidation_open_count >= 4096:
                raise QualificationError(
                    "recovery Z3 trusted reread event bound was exceeded"
                )
            del token
            self._trusted_revalidation_open_count += 1
            return
        if not self.open_boundary_denials:
            self.open_boundary_denials.append(
                {
                    "ordinal": 1,
                    "event": event,
                    "path_token": token,
                    "disposition": "denied_by_irreversible_audit_open_boundary",
                }
            )
        raise ModuleNotFoundError(
            "z3 namespace is policy-unavailable inside the sealed LGCVF-061 worker",
            name="z3",
        )

    def register_trusted_source_directory_open(
        self,
        *,
        logical_path: str,
        component_index: int,
        component: str,
        directory_flags: int,
    ) -> bool:
        """Arm only the config-pinned source component immediately before open."""

        if not self.enabled:
            return False
        expected_events = self._expected_trusted_source_events()
        if len(self._trusted_source_events) >= len(expected_events):
            return False
        expected = expected_events[len(self._trusted_source_events)]
        candidate = {
            "event": "open",
            "logical_path": logical_path,
            "component_index": component_index,
            "component": component,
            "directory_flags": directory_flags,
            "caller_code_identity": (
                "_read_recovery_projection_source.__code__"
            ),
            "source_projection_root": self.source_projection_root,
            "disposition": (
                "permitted_by_one_use_trusted_source_revalidation_capability"
            ),
        }
        if candidate != expected:
            return False
        try:
            caller = sys._getframe(1)
        except (ValueError, RuntimeError) as exc:
            self._trusted_source_failed = True
            raise QualificationError(
                "recovery trusted source registration caller is unavailable"
            ) from exc
        if (
            self._owner_phase != "source_revalidation_pending"
            or self._trusted_source_revalidation_depth != 1
            or self._trusted_source_failed
            or self._trusted_source_pending_event is not None
            or self._trusted_source_pending_audit_consumed
            or self._trusted_source_confirmation_count
            != len(self._trusted_source_events)
            or threading.get_ident() != self._owner_thread_ident
            or threading.get_native_id() != self._owner_native_thread_id
            or caller.f_code is not self._trusted_source_reader_code
            or not self._meta_path_identity_exact()
            or self._loaded_z3_modules()
            or self._z3_file_descriptor_evidence()
            or self._z3_native_mapping_evidence()
            or self.open_boundary_denials
        ):
            self._trusted_source_failed = True
            raise QualificationError(
                "recovery trusted source registration boundary differs"
            )
        self._trusted_source_pending_event = dict(candidate)
        return True

    def confirm_trusted_source_directory_open(self) -> None:
        """Confirm that the audit hook consumed the armed open capability."""

        if (
            not self.enabled
            or self._owner_phase != "source_revalidation_pending"
            or self._trusted_source_revalidation_depth != 1
            or self._trusted_source_failed
            or self._trusted_source_pending_event is not None
            or not self._trusted_source_pending_audit_consumed
            or self._trusted_source_confirmation_count + 1
            != len(self._trusted_source_events)
        ):
            self._trusted_source_failed = True
            raise QualificationError(
                "recovery trusted source audit consumption differs"
            )
        self._trusted_source_pending_audit_consumed = False
        self._trusted_source_confirmation_count += 1

    def deny_meta_import(self, fullname: str) -> None:
        if not self.enabled:
            raise QualificationError("recovery Z3 meta denial is not admitted")
        before = self._meta_path_before
        if (
            self._owner_phase != "candidate_execution"
            or before is None
            or not self._meta_path_identity_exact()
            or self._loaded_z3_modules()
            or self._z3_file_descriptor_evidence()
            or self._z3_native_mapping_evidence()
            or threading.get_ident() != self._owner_thread_ident
            or threading.get_native_id() != self._owner_native_thread_id
        ):
            raise QualificationError("recovery Z3 meta denial boundary differs")
        call_ordinal, nodeid = self.recorder.current_call_context()
        denial = {
            "ordinal": len(self.meta_denials) + 1,
            "pytest_call_ordinal": call_ordinal,
            "nodeid": nodeid,
            "module": fullname,
            "disposition": "denied_before_loader_as_typed_unavailable",
            "meta_path_identity_exact": True,
            "owner_thread_only": True,
            "z3_modules_absent": True,
            "z3_file_descriptor_count": 0,
            "z3_file_descriptor_root": content_identity([]),
            "z3_native_mapping_count": 0,
            "z3_native_mapping_root": content_identity([]),
        }
        expected = self.policy["expected_meta_denials"]
        if len(self.meta_denials) >= len(expected) or denial != expected[
            len(self.meta_denials)
        ]:
            raise QualificationError("recovery Z3 meta denial sequence differs")
        self.meta_denials.append(denial)
        raise ModuleNotFoundError(
            "z3 namespace is policy-unavailable inside the sealed LGCVF-061 worker",
            name=fullname,
        )

    def __enter__(self) -> _RecoveryZ3ImportDenialGuard:
        loaded = self._loaded_z3_modules()
        if (
            self.policy["z3_module_absence_required"]
            and (
                loaded
                or self._z3_file_descriptor_evidence()
                or self._z3_native_mapping_evidence()
            )
        ):
            raise QualificationError(
                "recovery worker started with policy-denied Z3 state loaded"
            )
        if not self.enabled:
            return self
        if self._audit_installed or self._meta_path_before is not None:
            raise QualificationError("recovery Z3 import guard was reused")
        self._meta_path_before = tuple(sys.meta_path)
        if self._finder in self._meta_path_before:
            raise QualificationError("recovery Z3 meta-path guard was preloaded")
        sys.addaudithook(self._audit)
        self._audit_installed = True
        sys.meta_path.insert(0, self._finder)
        if not self._meta_path_identity_exact():
            raise QualificationError("recovery Z3 meta-path admission differs")
        self._meta_path_active = True
        self._meta_path_restored = False
        self._pytest_meta_path_bootstrap_tuple_restored = True
        return self

    def _z3_file_descriptor_evidence(self) -> list[str]:
        """Return only stable runtime-relative tokens, never host descriptor IDs."""
        return _recovery_process_z3_file_descriptor_evidence(
            os.getpid(),
            self.z3_root,
        )

    def _meta_path_identity_exact(self) -> bool:
        before = self._meta_path_before
        if before is None:
            return False
        if (
            self._pytest_meta_path_admission_count == 1
            and self._pytest_meta_path_return_restoration_count == 0
        ):
            expected = self._pytest_candidate_meta_path
        else:
            expected = (self._finder, *before)
        return expected is not None and len(sys.meta_path) == len(expected) and all(
            observed is admitted
            for observed, admitted in zip(sys.meta_path, expected, strict=True)
        )

    @staticmethod
    def _is_exact_pytest_assertion_rewrite_hook(value: object) -> bool:
        hook_type = type(value)
        return (
            hook_type.__module__ == "_pytest.assertion.rewrite"
            and hook_type.__qualname__ == "AssertionRewritingHook"
        )

    def admit_pytest_meta_path_lifecycle(self, *, rewrite_hook: object) -> None:
        """Seal pytest's one known assertion-rewrite insertion by identity."""

        if not self.enabled:
            return
        before = self._meta_path_before
        if (
            before is None
            or self._pytest_meta_path_admission_count != 0
            or self._pytest_candidate_meta_path is not None
            or rewrite_hook in before
            or len(sys.meta_path) != len(before) + 2
            or sys.meta_path[0] is not rewrite_hook
            or not self._is_exact_pytest_assertion_rewrite_hook(rewrite_hook)
            or sys.meta_path[1] is not self._finder
            or any(
                observed is not expected
                for observed, expected in zip(
                    sys.meta_path[2:], before, strict=True
                )
            )
        ):
            raise QualificationError("recovery pytest meta-path admission differs")
        sys.meta_path[:] = [self._finder, rewrite_hook, *before]
        self._pytest_candidate_meta_path = tuple(sys.meta_path)
        self._pytest_meta_path_admission_count = 1
        self._pytest_meta_path_bootstrap_tuple_restored = False
        if not self._meta_path_identity_exact():
            raise QualificationError("recovery pytest meta-path seal differs")

    def validate_pytest_call_meta_path(self, *, ordinal: int, nodeid: str) -> None:
        if not self.enabled:
            return
        expected_nodeids = _recovery_061_expected_pytest_call_nodeids()
        if (
            ordinal != self._pytest_meta_path_call_validation_count + 1
            or ordinal < 1
            or ordinal > len(expected_nodeids)
            or nodeid != expected_nodeids[ordinal - 1]
            or self._pytest_meta_path_admission_count != 1
            or self._pytest_meta_path_return_restoration_count != 0
            or not self._meta_path_identity_exact()
        ):
            raise QualificationError("recovery pytest call meta-path seal differs")
        self._pytest_meta_path_call_validation_count = ordinal

    def validate_pytest_sessionfinish_meta_path(self) -> None:
        if not self.enabled:
            return
        if (
            self._pytest_meta_path_sessionfinish_count != 0
            or self._pytest_meta_path_call_validation_count
            != len(_recovery_061_expected_pytest_call_nodeids())
            or not self._meta_path_identity_exact()
        ):
            raise QualificationError("recovery pytest sessionfinish meta-path differs")
        self._pytest_meta_path_sessionfinish_count = 1
        self._pytest_meta_path_candidate_tuple_validated = True

    def validate_pytest_unconfigure_meta_path(self) -> None:
        if not self.enabled:
            return
        if (
            self._pytest_meta_path_unconfigure_count != 0
            or self._pytest_meta_path_sessionfinish_count != 1
            or not self._meta_path_identity_exact()
        ):
            raise QualificationError("recovery pytest unconfigure meta-path differs")
        self._pytest_meta_path_unconfigure_count = 1

    def validate_pytest_return_meta_path(self) -> None:
        """Require pytest to remove only its sealed rewrite hook on return."""

        if not self.enabled:
            return
        before = self._meta_path_before
        expected = (self._finder, *(before or ()))
        if (
            before is None
            or self._pytest_meta_path_return_restoration_count != 0
            or self._pytest_meta_path_admission_count != 1
            or self._pytest_meta_path_sessionfinish_count != 1
            or self._pytest_meta_path_unconfigure_count != 1
            or len(sys.meta_path) != len(expected)
            or any(
                observed is not admitted
                for observed, admitted in zip(sys.meta_path, expected, strict=True)
            )
        ):
            raise QualificationError("recovery pytest return meta-path differs")
        self._pytest_meta_path_return_restoration_count = 1
        self._pytest_meta_path_bootstrap_tuple_restored = True
        if not self._meta_path_identity_exact():
            raise QualificationError("recovery pytest bootstrap meta-path differs")

    def _z3_native_mapping_evidence(self) -> list[dict[str, Any]]:
        return _recovery_policy_denied_z3_native_mapping_evidence(
            _native_executable_mappings(),
            self.z3_root,
        )

    def complete_candidate_execution(
        self,
        *,
        expected_threads: Sequence[Mapping[str, Any]],
        expected_tasks: Sequence[int],
        expected_children: Sequence[int],
        pytest_exit_code: int,
        expected_sys_path: Sequence[str],
        transcript_within_bound: bool,
        loaded_origins_captured: bool,
        importer_cache_entry_cleared: bool,
    ) -> None:
        """Seal the judged interval before one-use source revalidation."""

        if not self.enabled:
            return
        before = self._meta_path_before
        observed_threads = _recovery_live_thread_population()
        observed_tasks = _recovery_kernel_task_population()
        observed_children = _recovery_child_process_population(
            task_ids=observed_tasks
        )
        if (
            self._owner_phase != "candidate_execution"
            or before is None
            or not self._meta_path_identity_exact()
            or self._pytest_meta_path_admission_count != 1
            or self._pytest_meta_path_call_validation_count
            != len(_recovery_061_expected_pytest_call_nodeids())
            or self._pytest_meta_path_sessionfinish_count != 1
            or self._pytest_meta_path_unconfigure_count != 1
            or self._pytest_meta_path_return_restoration_count != 1
            or not self._pytest_meta_path_candidate_tuple_validated
            or not self._pytest_meta_path_bootstrap_tuple_restored
            or self.meta_denials != self.policy["expected_meta_denials"]
            or self.open_boundary_denials
            or self._loaded_z3_modules()
            or self._z3_file_descriptor_evidence()
            or self._z3_native_mapping_evidence()
            or self.recorder.ordered_call_nodeids
            != _recovery_061_expected_pytest_call_nodeids()
            or self.recorder._current_call_nodeid
            or self.recorder._current_call_ordinal != 0
            or pytest_exit_code != 0
            or self.recorder.collected
            != len(_recovery_061_expected_pytest_call_nodeids())
            or self.recorder.passed != self.recorder.collected
            or any(
                value != 0
                for value in (
                    self.recorder.failed,
                    self.recorder.skipped,
                    self.recorder.xfailed,
                    self.recorder.xpassed,
                    self.recorder.errors,
                )
            )
            or tuple(sys.path) != tuple(expected_sys_path)
            or transcript_within_bound is not True
            or loaded_origins_captured is not True
            or importer_cache_entry_cleared is not True
            or observed_threads != [dict(item) for item in expected_threads]
            or observed_tasks != list(expected_tasks)
            or observed_children != list(expected_children)
            or self.provider_guard.imported_modules()
            or self.provider_guard.import_attempts
            or self.provider_guard.process_attempts
            or self._trusted_source_revalidation_depth != 0
            or self._trusted_source_revalidation_scope_entry_count != 0
            or self._trusted_source_revalidation_scope_exit_count != 0
            or self._trusted_source_revalidation_scope_completed
            or self._trusted_source_pending_event is not None
            or self._trusted_source_pending_audit_consumed
            or self._trusted_source_confirmation_count != 0
            or self._trusted_source_events
            or self._trusted_source_failed
        ):
            raise QualificationError("recovery Z3 candidate boundary differs")
        self._candidate_expected_threads = [dict(item) for item in expected_threads]
        self._candidate_expected_tasks = list(expected_tasks)
        self._candidate_expected_children = list(expected_children)
        self._owner_phase = "source_revalidation_pending"

    @contextlib.contextmanager
    def trusted_source_revalidation(self) -> Iterator[None]:
        """Admit one exact tracked-source component open after candidate closure."""

        if not self.enabled:
            yield
            return
        before = self._meta_path_before
        if (
            self._owner_phase != "source_revalidation_pending"
            or self._trusted_source_revalidation_depth != 0
            or self._trusted_source_revalidation_scope_entry_count != 0
            or self._trusted_source_revalidation_scope_exit_count != 0
            or self._trusted_source_revalidation_scope_completed
            or self._trusted_source_pending_event is not None
            or self._trusted_source_pending_audit_consumed
            or self._trusted_source_confirmation_count != 0
            or self._trusted_source_events
            or self._trusted_source_failed
            or threading.get_ident() != self._owner_thread_ident
            or threading.get_native_id() != self._owner_native_thread_id
            or before is None
            or not self._meta_path_identity_exact()
            or self._pytest_meta_path_return_restoration_count != 1
            or not self._pytest_meta_path_bootstrap_tuple_restored
            or self._loaded_z3_modules()
            or self._z3_file_descriptor_evidence()
            or self._z3_native_mapping_evidence()
            or self.open_boundary_denials
            or self.provider_guard.imported_modules()
            or self.provider_guard.import_attempts
            or self.provider_guard.process_attempts
            or self._candidate_expected_threads is None
            or self._candidate_expected_tasks is None
            or self._candidate_expected_children is None
            or _recovery_live_thread_population()
            != self._candidate_expected_threads
            or _recovery_kernel_task_population()
            != self._candidate_expected_tasks
            or _recovery_child_process_population(
                task_ids=self._candidate_expected_tasks
            )
            != self._candidate_expected_children
        ):
            self._trusted_source_failed = True
            raise QualificationError(
                "recovery trusted source revalidation admission differs"
            )
        self._trusted_source_revalidation_depth = 1
        self._trusted_source_revalidation_scope_entry_count = 1
        body_completed = False
        try:
            yield
            body_completed = True
        except BaseException:
            self._trusted_source_failed = True
            raise
        finally:
            self._trusted_source_revalidation_depth = 0
            self._trusted_source_revalidation_scope_exit_count += 1
            closure_exact = (
                body_completed
                and not self._trusted_source_failed
                and self._trusted_source_pending_event is None
                and not self._trusted_source_pending_audit_consumed
                and self._trusted_source_confirmation_count
                == self.policy["expected_trusted_source_event_count"]
                and self._trusted_source_events
                == self._expected_trusted_source_events()
                and content_identity(self._trusted_source_events)
                == content_identity(self._expected_trusted_source_events())
                and self._trusted_source_revalidation_scope_exit_count == 1
                and not self.open_boundary_denials
                and self._meta_path_identity_exact()
                and not self._loaded_z3_modules()
                and not self._z3_file_descriptor_evidence()
                and not self._z3_native_mapping_evidence()
                and not self.provider_guard.imported_modules()
                and not self.provider_guard.import_attempts
                and not self.provider_guard.process_attempts
                and _recovery_live_thread_population()
                == self._candidate_expected_threads
                and _recovery_kernel_task_population()
                == self._candidate_expected_tasks
                and _recovery_child_process_population(
                    task_ids=self._candidate_expected_tasks or []
                )
                == self._candidate_expected_children
            )
            if not closure_exact:
                self._trusted_source_failed = True
                self._trusted_source_pending_event = None
                self._trusted_source_pending_audit_consumed = False
                self._trusted_source_revalidation_scope_completed = False
                if body_completed:
                    raise QualificationError(
                        "recovery trusted source revalidation closure differs"
                    )
            else:
                self._trusted_source_revalidation_scope_completed = True
                self._owner_phase = "post_candidate_revalidation"

    @contextlib.contextmanager
    def trusted_runtime_revalidation(self) -> Iterator[None]:
        """Permit only the owner thread's bounded post-candidate runtime reread."""

        if not self.enabled:
            yield
            return
        before = self._meta_path_before
        if (
            self._owner_phase != "post_candidate_revalidation"
            or self._trusted_revalidation_depth != 0
            or self._trusted_revalidation_scope_entry_count != 0
            or self._trusted_revalidation_scope_exit_count != 0
            or self._trusted_revalidation_scope_completed
            or not self._trusted_source_revalidation_scope_completed
            or self._trusted_source_revalidation_scope_entry_count != 1
            or self._trusted_source_revalidation_scope_exit_count != 1
            or self._trusted_source_events
            != self._expected_trusted_source_events()
            or self._trusted_source_confirmation_count
            != self.policy["expected_trusted_source_event_count"]
            or self._trusted_source_failed
            or self.provider_guard.imported_modules()
            or self.provider_guard.import_attempts
            or self.provider_guard.process_attempts
            or threading.get_ident() != self._owner_thread_ident
            or threading.get_native_id() != self._owner_native_thread_id
            or before is None
            or not self._meta_path_identity_exact()
            or self._pytest_meta_path_return_restoration_count != 1
            or not self._pytest_meta_path_bootstrap_tuple_restored
            or self._loaded_z3_modules()
            or self._z3_file_descriptor_evidence()
            or self._z3_native_mapping_evidence()
        ):
            raise QualificationError("recovery Z3 trusted reread admission differs")
        self._trusted_revalidation_depth = 1
        self._trusted_revalidation_scope_entry_count += 1
        body_completed = False
        try:
            yield
            body_completed = True
        finally:
            self._trusted_revalidation_depth = 0
            self._trusted_revalidation_scope_exit_count += 1
            if (
                not self._meta_path_identity_exact()
                or self._loaded_z3_modules()
                or self._z3_file_descriptor_evidence()
                or self._z3_native_mapping_evidence()
            ):
                self._trusted_revalidation_scope_completed = False
                raise QualificationError("recovery Z3 trusted reread closure differs")
            self._trusted_revalidation_scope_completed = body_completed

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        del exc_type, exc, traceback
        return False

    def evidence(self, *, phase: str) -> dict[str, Any]:
        if phase not in {"prepared", "final"}:
            raise QualificationError("recovery Z3 denial evidence phase differs")
        policy = _recovery_z3_import_denial_policy(
            task_id=self.task_id,
            suite_id=self.suite_id,
        )
        expected_meta = (
            [] if phase == "prepared" else list(policy["expected_meta_denials"])
        )
        modules = self._loaded_z3_modules()
        expected_meta_state = "active_exact" if self.enabled else "not_applicable"
        observed_meta_state = (
            "active_exact"
            if self._meta_path_active
            and self._meta_path_identity_exact()
            else "not_applicable"
        )
        expected_owner_phase = (
            "candidate_execution"
            if self.enabled and phase == "prepared"
            else "post_candidate_revalidation"
            if self.enabled
            else "not_applicable"
        )
        z3_fds = self._z3_file_descriptor_evidence()
        z3_native_mappings = (
            self._z3_native_mapping_evidence() if self.enabled else []
        )
        observed_call_nodeids = (
            list(self.recorder.ordered_call_nodeids) if self.enabled else []
        )
        final_enabled = self.enabled and phase == "final"
        expected_lifecycle_disposition = (
            "candidate_and_trusted_source_completed_bootstrap_restored"
            if final_enabled
            else "candidate_not_started"
            if self.enabled
            else "not_applicable"
        )
        if (
            self.policy != policy
            or self.meta_denials != expected_meta
            or self.open_boundary_denials
            or observed_meta_state != expected_meta_state
            or self._owner_phase != expected_owner_phase
            or self._trusted_revalidation_depth != 0
            or self._trusted_source_revalidation_depth != 0
            or (policy["z3_module_absence_required"] and modules)
            or (policy["z3_module_absence_required"] and z3_fds)
            or (self.enabled and z3_native_mappings)
            or (self.enabled and not self._audit_installed)
            or self.provider_guard.imported_modules()
            or self.provider_guard.import_attempts
            or self.provider_guard.process_attempts
            or self._pytest_meta_path_admission_count != (1 if final_enabled else 0)
            or self._pytest_meta_path_call_validation_count
            != (len(_recovery_061_expected_pytest_call_nodeids()) if final_enabled else 0)
            or self._pytest_meta_path_sessionfinish_count
            != (1 if final_enabled else 0)
            or self._pytest_meta_path_unconfigure_count
            != (1 if final_enabled else 0)
            or self._pytest_meta_path_return_restoration_count
            != (1 if final_enabled else 0)
            or self._pytest_meta_path_candidate_tuple_validated is not final_enabled
            or self._pytest_meta_path_bootstrap_tuple_restored is not self.enabled
            or self._trusted_source_failed
            or self._trusted_source_pending_event is not None
            or self._trusted_source_pending_audit_consumed
            or (
                phase == "final"
                and self.enabled
                and self.recorder.ordered_call_nodeids
                != _recovery_061_expected_pytest_call_nodeids()
            )
            or (
                phase == "prepared"
                and self.enabled
                and observed_call_nodeids
            )
            or (
                phase == "final"
                and self.enabled
                and (
                    self._trusted_source_revalidation_scope_entry_count != 1
                    or self._trusted_source_revalidation_scope_exit_count != 1
                    or not self._trusted_source_revalidation_scope_completed
                    or self._trusted_source_events
                    != self._expected_trusted_source_events()
                    or self._trusted_source_confirmation_count
                    != policy["expected_trusted_source_event_count"]
                    or not self._trusted_source_caller_code_identity_validated
                    or self._trusted_revalidation_open_count < 1
                    or self._trusted_revalidation_scope_entry_count != 1
                    or self._trusted_revalidation_scope_exit_count != 1
                    or not self._trusted_revalidation_scope_completed
                )
            )
            or (
                phase == "prepared"
                and self.enabled
                and (
                    self._trusted_source_revalidation_scope_entry_count != 0
                    or self._trusted_source_revalidation_scope_exit_count != 0
                    or self._trusted_source_revalidation_scope_completed
                    or self._trusted_source_events
                    or self._trusted_source_confirmation_count != 0
                )
            )
        ):
            raise QualificationError("recovery Z3 import denial evidence differs")
        value: dict[str, Any] = {
            "schema": "lgcvf-recovery-z3-import-denial-evidence@3",
            "phase": phase,
            "task_id": self.task_id,
            "suite_id": self.suite_id,
            "policy_cid": policy["policy_cid"],
            "policy_disposition": policy["disposition"],
            "top_level_import_audit_claimed": False,
            "irreversible_audit_open_boundary_installed": self._audit_installed,
            "meta_path_guard_state": observed_meta_state,
            "owner_phase": self._owner_phase,
            "process_exit_removal_boundary": self.enabled,
            "pytest_meta_path_lifecycle_disposition": (
                expected_lifecycle_disposition
            ),
            "pytest_meta_path_admission_count": (
                self._pytest_meta_path_admission_count
            ),
            "pytest_meta_path_call_start_validation_count": (
                self._pytest_meta_path_call_validation_count
            ),
            "pytest_meta_path_sessionfinish_validation_count": (
                self._pytest_meta_path_sessionfinish_count
            ),
            "pytest_meta_path_unconfigure_validation_count": (
                self._pytest_meta_path_unconfigure_count
            ),
            "pytest_meta_path_return_restoration_count": (
                self._pytest_meta_path_return_restoration_count
            ),
            "pytest_meta_path_candidate_tuple_validated": (
                self._pytest_meta_path_candidate_tuple_validated
            ),
            "pytest_meta_path_bootstrap_tuple_restored": (
                self._pytest_meta_path_bootstrap_tuple_restored
            ),
            "trusted_revalidation_owner_thread_only": self.enabled,
            "trusted_revalidation_scope_closed": (
                self._trusted_revalidation_depth == 0
            ),
            "trusted_source_revalidation_disposition": policy[
                "trusted_source_revalidation_disposition"
            ],
            "trusted_source_revalidation_source_projection_root": (
                self.source_projection_root
            ),
            "trusted_source_revalidation_expected_event_count": policy[
                "expected_trusted_source_event_count"
            ],
            "trusted_source_revalidation_expected_event_root": policy[
                "expected_trusted_source_event_root"
            ],
            "ordered_trusted_source_revalidation_events": [
                dict(item) for item in self._trusted_source_events
            ],
            "trusted_source_revalidation_event_count": len(
                self._trusted_source_events
            ),
            "trusted_source_revalidation_event_root": content_identity(
                self._trusted_source_events
            ),
            "trusted_source_revalidation_scope_entry_count": (
                self._trusted_source_revalidation_scope_entry_count
            ),
            "trusted_source_revalidation_scope_exit_count": (
                self._trusted_source_revalidation_scope_exit_count
            ),
            "trusted_source_revalidation_scope_completed": (
                self._trusted_source_revalidation_scope_completed
            ),
            "trusted_source_revalidation_pending_empty": (
                self._trusted_source_pending_event is None
                and not self._trusted_source_pending_audit_consumed
            ),
            "trusted_source_revalidation_confirmation_count": (
                self._trusted_source_confirmation_count
            ),
            "trusted_source_revalidation_owner_thread_only": self.enabled,
            "trusted_source_revalidation_caller_code_identity_exact": (
                self._trusted_source_caller_code_identity_validated
            ),
            "trusted_source_revalidation_descriptor_identity_validated": (
                final_enabled
                and self._trusted_source_confirmation_count
                == policy["expected_trusted_source_event_count"]
            ),
            "trusted_source_revalidation_global_z3_exemption": False,
            "trusted_source_revalidation_audit_dirfd_observed": False,
            "trusted_source_revalidation_telemetry_authoritative": (
                self.enabled
            ),
            "trusted_source_revalidation_telemetry_reconstructed": False,
            "ordered_meta_denials": [dict(item) for item in self.meta_denials],
            "meta_denial_count": len(self.meta_denials),
            "meta_denial_root": content_identity(self.meta_denials),
            "ordered_open_boundary_denials": [
                dict(item) for item in self.open_boundary_denials
            ],
            "open_boundary_denial_count": len(self.open_boundary_denials),
            "open_boundary_denial_root": content_identity(
                self.open_boundary_denials
            ),
            "trusted_revalidation_scope_entry_count": (
                self._trusted_revalidation_scope_entry_count
            ),
            "trusted_revalidation_scope_exit_count": (
                self._trusted_revalidation_scope_exit_count
            ),
            "trusted_revalidation_scope_completed": (
                self._trusted_revalidation_scope_completed
            ),
            "trusted_revalidation_permitted_z3_open_count": (
                self._trusted_revalidation_open_count
            ),
            "trusted_revalidation_telemetry_authoritative": False,
            "trusted_revalidation_telemetry_reconstructed": False,
            "z3_modules_absent": not modules,
            "z3_file_descriptor_count": len(z3_fds),
            "z3_file_descriptor_root": content_identity(z3_fds),
            "policy_denied_z3_native_mapping_count": len(z3_native_mappings),
            "policy_denied_z3_native_mapping_root": content_identity(
                z3_native_mappings
            ),
            "z3_loader_executed": bool(modules),
            "pytest_call_count": len(observed_call_nodeids),
            "pytest_call_nodeid_root": content_identity(
                observed_call_nodeids
            ),
            "policy_namespace_unavailability": policy[
                "policy_namespace_unavailability"
            ],
            "live_z3_cegar_disposition": policy["live_z3_cegar_disposition"],
            "candidate_reason_interpretation": policy[
                "candidate_reason_interpretation"
            ],
            "infrastructure_not_proof": True,
            "cache_authority": False,
            "completion_authoritative": False,
        }
        value["evidence_cid"] = content_identity(value)
        return value


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
    ordered_call_nodeids: list[str] = field(default_factory=list)
    _current_call_nodeid: str = ""
    _current_call_ordinal: int = 0
    _terminal: set[str] = field(default_factory=set)
    _z3_import_denial_guard: Any | None = None

    def bind_z3_import_denial_guard(
        self, guard: _RecoveryZ3ImportDenialGuard
    ) -> None:
        if self._z3_import_denial_guard is not None:
            raise QualificationError("recovery Z3 denial recorder was rebound")
        self._z3_import_denial_guard = guard

    def pytest_load_initial_conftests(
        self, early_config: Any, parser: Any, args: list[str]
    ) -> None:
        del parser, args
        guard = self._z3_import_denial_guard
        if guard is None or not guard.enabled:
            return
        pluginmanager = getattr(early_config, "pluginmanager", None)
        rewrite_hook = getattr(pluginmanager, "rewrite_hook", None)
        if rewrite_hook is None:
            raise QualificationError("recovery pytest rewrite hook is absent")
        guard.admit_pytest_meta_path_lifecycle(rewrite_hook=rewrite_hook)

    def pytest_sessionfinish(self, session: Any, exitstatus: Any) -> None:
        del session, exitstatus
        guard = self._z3_import_denial_guard
        if guard is not None:
            guard.validate_pytest_sessionfinish_meta_path()

    def pytest_unconfigure(self, config: Any) -> None:
        del config
        guard = self._z3_import_denial_guard
        if guard is not None:
            guard.validate_pytest_unconfigure_meta_path()

    def pytest_collection_finish(self, session: Any) -> None:
        self.collected = len(session.items)
        self.nodeids = sorted(str(item.nodeid) for item in session.items)

    def pytest_collectreport(self, report: Any) -> None:
        if report.failed:
            self.errors += 1

    def current_call_context(self) -> tuple[int, str]:
        if (
            self._current_call_ordinal < 1
            or not self._current_call_nodeid
            or self.ordered_call_nodeids[-1] != self._current_call_nodeid
        ):
            raise QualificationError("recovery pytest call context is absent")
        return self._current_call_ordinal, self._current_call_nodeid

    def pytest_runtest_logreport(self, report: Any) -> None:
        nodeid = str(report.nodeid)
        if report.when == "setup" and report.passed:
            self.ordered_call_nodeids.append(nodeid)
            self._current_call_nodeid = nodeid
            self._current_call_ordinal = len(self.ordered_call_nodeids)
            guard = self._z3_import_denial_guard
            if guard is not None:
                guard.validate_pytest_call_meta_path(
                    ordinal=self._current_call_ordinal,
                    nodeid=nodeid,
                )
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
            self._current_call_nodeid = ""
            self._current_call_ordinal = 0
        elif report.when == "setup" and report.skipped:
            self.skipped += 1
            self._terminal.add(nodeid)
        elif report.when in {"setup", "teardown"} and report.failed:
            self.errors += 1
            self._terminal.add(nodeid)


class _BoundedTextCapture(io.StringIO):
    """In-memory text capture sharing one hard UTF-8 byte budget."""

    def __init__(self, budget: list[int]) -> None:
        super().__init__()
        self._budget = budget

    def write(self, value: str) -> int:
        encoded_size = len(value.encode("utf-8", errors="replace"))
        if encoded_size > self._budget[0]:
            raise QualificationError("recovery worker transcript exceeded its bound")
        self._budget[0] -= encoded_size
        return super().write(value)


def _emit_worker_receipt(descriptor: int, value: Mapping[str, Any]) -> None:
    """Emit on a pre-pytest pipe descriptor immune to pytest FD capture."""

    payload = _canonical_bytes(value) + b"\n"
    if len(payload) > _MAX_WORKER_RECEIPT_BYTES:
        attempted_bytes = len(payload)
        payload = _canonical_bytes(
            {
                "schema": value.get("schema", WORKER_SCHEMA),
                "suite_id": value.get("suite_id", ""),
                "error": "receipt_too_large",
                "reason": "worker receipt exceeded its predeclared pipe bound",
                "attempted_bytes": attempted_bytes,
                "limit_bytes": _MAX_WORKER_RECEIPT_BYTES,
            }
        ) + b"\n"
    view = memoryview(payload)
    try:
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise QualificationError("worker receipt pipe made no progress")
            view = view[written:]
    finally:
        os.close(descriptor)


def _recovery_worker_pycache_state(
    write_root: Path,
    *,
    expected_identity: tuple[int, int] | None = None,
) -> tuple[tuple[int, int], dict[str, Any]]:
    """Validate the fresh, process-scoped bytecode lookup root."""

    expected = _ISOLATED_RECOVERY_PYCACHE_ROOT
    if (
        expected is None
        or expected.parent != write_root
        or not expected.name.startswith("python-pycache-")
        or expected.is_symlink()
    ):
        raise QualificationError("recovery worker pycache path differs")
    if (
        sys.pycache_prefix != str(expected)
        or sys.dont_write_bytecode is not True
    ):
        raise QualificationError("recovery worker pycache isolation differs")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(expected, flags)
    except OSError as exc:
        raise QualificationError("recovery worker pycache is unavailable") from exc
    try:
        status = os.fstat(descriptor)
        identity = (status.st_dev, status.st_ino)
        if (
            not stat.S_ISDIR(status.st_mode)
            or status.st_uid != os.geteuid()
            or stat.S_IMODE(status.st_mode) != 0o700
            or (expected_identity is not None and identity != expected_identity)
        ):
            raise QualificationError("recovery worker pycache identity differs")
        with os.scandir(descriptor) as entries:
            if next(entries, None) is not None:
                raise QualificationError("recovery worker pycache is not empty")
    finally:
        os.close(descriptor)
    return identity, {
        "schema": _RECOVERY_WORKER_PYCACHE_SCHEMA,
        "write_root_relative_path": "<temporary:python-pycache-*>",
        "activation": "shared_bootstrap_capsule",
        "python_prefix_active": True,
        "dont_write_bytecode": True,
        "owner_matches_worker": True,
        "mode_octal": "0700",
        "root_identity": {
            "dev": status.st_dev,
            "ino": status.st_ino,
            "uid": status.st_uid,
            "gid": status.st_gid,
            "mode": stat.S_IMODE(status.st_mode),
            "nlink": status.st_nlink,
        },
    }


def _validate_worker_execution_projection(
    value: Mapping[str, Any],
    *,
    worker_root: Path,
    suite: Suite,
) -> dict[str, Any]:
    """Validate the parent's closed projection and every worker-visible path."""

    if set(value) != {
        "schema",
        "selected_sources",
        "copied_source_count",
        "copied_source_bytes",
        "copied_source_manifest_root",
        "omitted_source_symlinks",
        "contains_live_source_links",
        "original_checkout_writable",
        "projection_cid",
    } or value.get("schema") != "lgcvf-closed-recovery-test-projection@2":
        raise QualificationError("recovery worker projection fields differ")
    body = {key: item for key, item in value.items() if key != "projection_cid"}
    if value.get("projection_cid") != content_identity(body):
        raise QualificationError("recovery worker projection identity differs")
    selected = value.get("selected_sources")
    omitted = value.get("omitted_source_symlinks")
    if (
        value.get("original_checkout_writable") is not False
        or value.get("contains_live_source_links") is not False
        or not isinstance(selected, list)
        or not isinstance(omitted, list)
        or len(selected) != 1
        or len(omitted) > 256
    ):
        raise QualificationError("recovery worker projection population differs")
    paths = _recovery_projection_source_paths(worker_root, (suite,))
    manifest, _payloads = _recovery_projection_manifest(
        worker_root,
        paths,
        head_bound=False,
    )
    selected_relative = (Path(suite.owner_root) / suite.paths[0]).as_posix().lstrip("./")
    expected_selected = [
        next(item for item in manifest if item["path"] == selected_relative)
    ]
    if (
        selected != expected_selected
        or value.get("copied_source_count") != len(manifest)
        or value.get("copied_source_bytes")
        != sum(int(item["size_bytes"]) for item in manifest)
        or value.get("copied_source_manifest_root") != content_identity(manifest)
        or any(path.is_symlink() for path in worker_root.rglob("*"))
    ):
        raise QualificationError("recovery worker copied source bytes differ")
    omitted_paths: set[str] = set()
    for item in omitted:
        if not isinstance(item, Mapping) or set(item) != {
            "path",
            "git_target",
            "disposition",
        }:
            raise QualificationError("recovery worker omitted symlink differs")
        relative = str(item.get("path") or "")
        logical = PurePath(relative)
        if (
            not relative
            or logical.is_absolute()
            or ".." in logical.parts
            or logical.as_posix() != relative
            or relative in omitted_paths
            or not isinstance(item.get("git_target"), str)
            or not item.get("git_target")
            or item.get("disposition") != "omitted_source_symlink"
        ):
            raise QualificationError("recovery worker omitted symlink path differs")
        omitted_paths.add(relative)
        try:
            worker_root.joinpath(*logical.parts).lstat()
        except FileNotFoundError:
            continue
        raise QualificationError("recovery worker retained an omitted symlink")
    return dict(value)


_RECOVERY_WORKER_BOOTSTRAP = r'''
import ctypes, json, os, runpy, stat, sys, tempfile

def closed_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise RuntimeError("duplicate bootstrap projection field")
        value[key] = item
    return value

expected_stdlib = (
    "/usr/lib/python312.zip",
    "/usr/lib/python3.12",
    "/usr/lib/python3.12/lib-dynload",
)
if tuple(sys.path) != expected_stdlib:
    raise RuntimeError("worker initial stdlib path differs")
flags = sys.flags
c_flags = tuple(
    ctypes.c_int.in_dll(ctypes.pythonapi, name).value
    for name in (
        "Py_IsolatedFlag",
        "Py_IgnoreEnvironmentFlag",
        "Py_NoSiteFlag",
        "Py_DontWriteBytecodeFlag",
    )
)
if (
    flags.isolated != 1
    or flags.ignore_environment != 1
    or flags.no_site != 1
    or flags.safe_path is not True
    or flags.dont_write_bytecode != 1
    or sys.dont_write_bytecode is not True
    or c_flags != (1, 1, 1, 1)
):
    raise RuntimeError("worker interpreter flags differ")
if len(sys.argv) < 12:
    raise RuntimeError("worker bootstrap arguments are absent")
(
    runtime_root,
    runtime_fd_text,
    expected_runtime_cid,
    projection_text,
    checkout,
    owner_root,
    writable,
    qualifier_path,
    *qualifier_arguments,
) = sys.argv[1:]
if owner_root not in {".", "ipfs_datasets_py"}:
    raise RuntimeError("worker owner root differs")
runtime_fd = int(runtime_fd_text)
projection_raw = __import__("base64").b64decode(
    projection_text.encode("ascii"), altchars=b"-_", validate=True
)
if len(projection_raw) > 64 * 1024:
    raise RuntimeError("worker runtime projection exceeds its bound")
projection = json.loads(
    projection_raw.decode("utf-8"), object_pairs_hook=closed_object
)
root_identity = projection.get("root_identity")
root_status = os.fstat(runtime_fd)
runtime_lstat = os.lstat(runtime_root)
if (
    projection.get("schema") != "lgcvf-qualification-runtime-projection@1"
    or projection.get("runtime_cid") != expected_runtime_cid
    or not isinstance(root_identity, dict)
    or root_identity
    != {
        "dev": root_status.st_dev,
        "ino": root_status.st_ino,
        "uid": root_status.st_uid,
        "gid": root_status.st_gid,
        "mode": stat.S_IMODE(root_status.st_mode),
        "nlink": root_status.st_nlink,
    }
    or not stat.S_ISDIR(root_status.st_mode)
    or root_status.st_uid != os.geteuid()
    or stat.S_IMODE(root_status.st_mode) != 0o500
    or (runtime_lstat.st_dev, runtime_lstat.st_ino)
    != (root_status.st_dev, root_status.st_ino)
    or os.path.realpath(runtime_root) != runtime_root
):
    raise RuntimeError("worker runtime projection root differs")
for value in (checkout, writable, qualifier_path):
    if not os.path.isabs(value) or os.path.realpath(value) != value:
        raise RuntimeError("worker bootstrap path differs")
if qualifier_path != os.path.join(
    checkout,
    "scripts/qualify_logic_governed_compositional_verification_fabric.py",
):
    raise RuntimeError("worker qualifier path differs")
expected_environment = {
    "GIT_ATTR_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "HOME": os.path.join(writable, "home"),
    "IPFS_DATASETS_AUTO_INSTALL": "0",
    "IPFS_DATASETS_AUTO_INSTALL_TEST_DEPS": "0",
    "IPFS_DATASETS_PY_LAZY_INSTALL_ERGOAI": "0",
    "IPFS_DATASETS_PY_MINIMAL_IMPORTS": "1",
    "IPFS_KIT_AUTO_INSTALL_DEPS": "0",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "MKL_NUM_THREADS": "1",
    "NO_COLOR": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PATH": "/usr/bin:/bin",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    "TMPDIR": writable,
    "TZ": "UTC",
    "XDG_CACHE_HOME": os.path.join(writable, "home/.cache"),
    "XDG_CONFIG_HOME": os.path.join(writable, "home/.config"),
    "XDG_DATA_HOME": os.path.join(writable, "home/.local/share"),
    "XDG_STATE_HOME": os.path.join(writable, "home/.local/state"),
}
if dict(os.environ) != expected_environment:
    raise RuntimeError("worker environment differs")
write_status = os.lstat(writable)
if (
    not stat.S_ISDIR(write_status.st_mode)
    or write_status.st_uid != os.geteuid()
    or stat.S_IMODE(write_status.st_mode) != 0o700
):
    raise RuntimeError("worker writable root differs")
pycache_directory = tempfile.TemporaryDirectory(
    prefix="python-pycache-", dir=writable
)
pycache_root = os.path.realpath(pycache_directory.name)
os.chmod(pycache_root, 0o700)
pycache_status = os.lstat(pycache_root)
if os.listdir(pycache_root):
    raise RuntimeError("worker pycache root is not empty")
sys.pycache_prefix = pycache_root
pycache_capsule = (
    "ipfs_accelerate_py/lgcvf-isolated-recovery-pycache@1",
    os.getpid(),
    pycache_directory,
    pycache_root,
    pycache_status.st_dev,
    pycache_status.st_ino,
    object(),
)
setattr(sys, "_lgcvf_isolated_recovery_pycache_capsule_v1", pycache_capsule)
setattr(sys, "_lgcvf_initial_isolated_stdlib_paths_v1", expected_stdlib)
owner = checkout if owner_root == "." else os.path.join(checkout, owner_root)
counterpart = os.path.join(checkout, "ipfs_datasets_py") if owner_root == "." else checkout
closed_path = (runtime_root, owner, counterpart, *expected_stdlib)
sys.path[:] = closed_path
bootstrap_capsule = (
    "lgcvf-qualification-runtime-bootstrap@1",
    os.getpid(),
    runtime_root,
    runtime_fd,
    expected_runtime_cid,
    projection,
    checkout,
    owner_root,
    writable,
    closed_path,
    tuple(sorted(expected_environment.items())),
    pycache_capsule,
    object(),
)
setattr(sys, "_lgcvf_qualification_runtime_bootstrap_v1", bootstrap_capsule)
sys.argv[:] = [qualifier_path, *qualifier_arguments]
runpy.run_path(qualifier_path, run_name="__main__")
'''.strip()


def _normalized_recovery_worker_environment(write_root: Path) -> dict[str, str]:
    home = write_root / "home"
    return {
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": str(home),
        "IPFS_DATASETS_AUTO_INSTALL": "0",
        "IPFS_DATASETS_AUTO_INSTALL_TEST_DEPS": "0",
        "IPFS_DATASETS_PY_LAZY_INSTALL_ERGOAI": "0",
        "IPFS_DATASETS_PY_MINIMAL_IMPORTS": "1",
        "IPFS_KIT_AUTO_INSTALL_DEPS": "0",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "MKL_NUM_THREADS": "1",
        "NO_COLOR": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PATH": "/usr/bin:/bin",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "TMPDIR": str(write_root),
        "TZ": "UTC",
        "XDG_CACHE_HOME": str(home / ".cache"),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_DATA_HOME": str(home / ".local/share"),
        "XDG_STATE_HOME": str(home / ".local/state"),
    }


def _normalized_recovery_worker_environment_evidence() -> dict[str, str]:
    return {
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": "<worker-writable>/home",
        "IPFS_DATASETS_AUTO_INSTALL": "0",
        "IPFS_DATASETS_AUTO_INSTALL_TEST_DEPS": "0",
        "IPFS_DATASETS_PY_LAZY_INSTALL_ERGOAI": "0",
        "IPFS_DATASETS_PY_MINIMAL_IMPORTS": "1",
        "IPFS_KIT_AUTO_INSTALL_DEPS": "0",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "MKL_NUM_THREADS": "1",
        "NO_COLOR": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PATH": "/usr/bin:/bin",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "TMPDIR": "<worker-writable>",
        "TZ": "UTC",
        "XDG_CACHE_HOME": "<worker-writable>/home/.cache",
        "XDG_CONFIG_HOME": "<worker-writable>/home/.config",
        "XDG_DATA_HOME": "<worker-writable>/home/.local/share",
        "XDG_STATE_HOME": "<worker-writable>/home/.local/state",
    }


def _qualification_runtime_bootstrap_evidence(
    projection: Mapping[str, Any],
) -> dict[str, Any]:
    root_identity = projection.get("root_identity")
    if not isinstance(root_identity, Mapping):
        raise QualificationError("qualification runtime root identity is absent")
    evidence: dict[str, Any] = {
        "schema": _QUALIFICATION_RUNTIME_BOOTSTRAP_SCHEMA,
        "runtime_cid": projection.get("runtime_cid"),
        "projection_cid": projection.get("projection_cid"),
        "runtime_root_identity": dict(root_identity),
        "path_policy": {
            "runtime_projection_first": True,
            "owner_checkout_second": True,
            "counterpart_checkout_third": True,
            "isolated_stdlib_paths": list(
                _RECOVERY_EXPECTED_ISOLATED_STDLIB_PATHS
            ),
            "ambient_site_paths": False,
        },
        "environment_policy": _normalized_recovery_worker_environment_evidence(),
        "interpreter_flags": ["-I", "-S", "-B"],
        "pycache_capsule_schema": _RECOVERY_PYCACHE_CAPSULE_SCHEMA,
    }
    evidence["bootstrap_cid"] = content_identity(evidence)
    return evidence


def _validate_recovery_worker_bootstrap() -> dict[str, Any]:
    capsule = _RECOVERY_WORKER_BOOTSTRAP_CAPSULE
    if not isinstance(capsule, tuple) or len(capsule) != 13:
        raise QualificationError("recovery worker bootstrap is absent")
    (
        schema,
        process_id,
        runtime_root_text,
        runtime_fd,
        runtime_cid,
        projection,
        checkout_text,
        owner_root,
        writable_text,
        closed_path,
        environment,
        pycache_capsule,
        seal,
    ) = capsule
    if (
        schema != _QUALIFICATION_RUNTIME_BOOTSTRAP_SCHEMA
        or type(process_id) is not int
        or process_id != os.getpid()
        or type(runtime_root_text) is not str
        or type(runtime_fd) is not int
        or type(runtime_cid) is not str
        or not isinstance(projection, Mapping)
        or type(checkout_text) is not str
        or owner_root not in {".", "ipfs_datasets_py"}
        or type(writable_text) is not str
        or not isinstance(closed_path, tuple)
        or not isinstance(environment, tuple)
        or type(seal) is not object
        or getattr(sys, _RECOVERY_WORKER_BOOTSTRAP_ATTRIBUTE, None) is not capsule
        or pycache_capsule is not _ISOLATED_RECOVERY_PYCACHE_CAPSULE
    ):
        raise QualificationError("recovery worker bootstrap identity differs")
    runtime_root = Path(runtime_root_text)
    checkout = Path(checkout_text)
    writable = Path(writable_text)
    root_status = os.fstat(runtime_fd)
    lexical_status = runtime_root.lstat()
    root_identity = projection.get("root_identity")
    expected_path = (
        runtime_root_text,
        str(checkout if owner_root == "." else checkout / owner_root),
        str(checkout / "ipfs_datasets_py" if owner_root == "." else checkout),
        *_RECOVERY_EXPECTED_ISOLATED_STDLIB_PATHS,
    )
    expected_root_identity = {
            "dev": root_status.st_dev,
            "ino": root_status.st_ino,
            "uid": root_status.st_uid,
            "gid": root_status.st_gid,
            "mode": stat.S_IMODE(root_status.st_mode),
            "nlink": root_status.st_nlink,
    }
    if (
        projection.get("schema") != _QUALIFICATION_RUNTIME_PROJECTION_SCHEMA
        or projection.get("runtime_cid") != runtime_cid
        or projection.get("projection_cid") is None
    ):
        raise QualificationError("recovery worker bootstrap projection differs")
    if not isinstance(root_identity, Mapping) or dict(root_identity) != expected_root_identity:
        raise QualificationError("recovery worker bootstrap root identity differs")
    if (lexical_status.st_dev, lexical_status.st_ino) != (
        root_status.st_dev,
        root_status.st_ino,
    ):
        raise QualificationError("recovery worker bootstrap lexical root differs")
    if tuple(closed_path) != expected_path:
        raise QualificationError("recovery worker bootstrap path receipt differs")
    if tuple(sys.path) != expected_path:
        raise QualificationError(
            "recovery worker bootstrap process path differs: "
            + str(len(sys.path))
        )
    expected_environment = _normalized_recovery_worker_environment(writable)
    if tuple(environment) != tuple(sorted(expected_environment.items())):
        raise QualificationError("recovery worker bootstrap environment receipt differs")
    if dict(os.environ) != expected_environment:
        observed_environment = dict(os.environ)
        differing_keys = sorted(
            set(observed_environment).symmetric_difference(expected_environment)
            | {
                key
                for key in set(observed_environment).intersection(expected_environment)
                if observed_environment[key] != expected_environment[key]
            }
        )
        raise QualificationError(
            "recovery worker bootstrap process environment differs: "
            + ",".join(differing_keys[:32])
        )
    if (
        _RECOVERY_INITIAL_ISOLATED_STDLIB_PATHS
        != _RECOVERY_EXPECTED_ISOLATED_STDLIB_PATHS
    ):
        raise QualificationError("recovery worker initial stdlib identity differs")
    return _qualification_runtime_bootstrap_evidence(projection)


def _worker_qualification_runtime_state() -> tuple[
    dict[str, Any], dict[str, Any], Path, dict[str, Any]
]:
    """Reconstruct the inherited projection and its host binding in the child."""

    bootstrap = _validate_recovery_worker_bootstrap()
    capsule = _RECOVERY_WORKER_BOOTSTRAP_CAPSULE
    runtime_root = Path(str(capsule[2]))
    runtime_fd = int(capsule[3])
    projection = dict(capsule[5])
    control_path = ".lgcvf-runtime/control-manifest.json"
    control_bytes, _control_observation = _read_qualification_runtime_relative(
        runtime_fd,
        control_path,
        noun="worker qualification runtime control manifest",
        owner_uid=os.geteuid(),
        limit=4 * 1024 * 1024,
    )
    control_entry = projection.get("control_file")
    if (
        not isinstance(control_entry, Mapping)
        or control_entry.get("path") != control_path
        or control_entry.get("sha256") != _sha256_bytes(control_bytes)
        or control_entry.get("size_bytes") != len(control_bytes)
        or control_entry.get("projection_mode_octal") != "0400"
        or control_entry.get("native") is not False
    ):
        raise QualificationError("worker runtime control identity differs")
    try:
        control = _strict_json_loads(
            control_bytes.decode("utf-8", errors="strict"),
            noun="worker qualification runtime control manifest",
        )
    except UnicodeDecodeError as exc:
        raise QualificationError("worker runtime control is not UTF-8") from exc
    if not isinstance(control, dict) or set(control) != {
        "schema",
        "bundle",
        "components",
        "payload_manifest",
        "manifest_cid",
    }:
        raise QualificationError("worker runtime control fields differ")
    control_body = {
        key: item for key, item in control.items() if key != "manifest_cid"
    }
    bundle = control.get("bundle")
    components = control.get("components")
    payload_manifest = control.get("payload_manifest")
    if (
        control.get("schema")
        != "lgcvf-qualification-runtime-control-manifest@1"
        or control.get("manifest_cid") != content_identity(control_body)
        or control.get("manifest_cid") != projection.get("bundle_manifest_cid")
        or not isinstance(bundle, dict)
        or not isinstance(components, list)
        or not isinstance(payload_manifest, list)
        or bundle.get("schema") != _QUALIFICATION_RUNTIME_BUNDLE_SCHEMA
        or bundle.get("runtime_cid") != projection.get("runtime_cid")
        or bundle.get("runtime_cid") != bootstrap.get("runtime_cid")
        or bundle.get("runtime_cid")
        != content_identity(
            {key: item for key, item in bundle.items() if key != "runtime_cid"}
        )
        or bundle.get("recovery_suite_task_policy")
        != _recovery_suite_task_policy_matrix()
        or len(components) != len(_QUALIFICATION_RUNTIME_COMPONENTS)
        or len(payload_manifest) != _MAX_QUALIFICATION_RUNTIME_PAYLOAD_FILES
    ):
        raise QualificationError("worker runtime control authority differs")
    observed_component_cids: list[str] = []
    for ordinal, component in enumerate(components, start=1):
        if (
            not isinstance(component, dict)
            or component.get("schema") != _QUALIFICATION_RUNTIME_COMPONENT_SCHEMA
            or component.get("ordinal") != ordinal
            or component.get("component_cid")
            != content_identity(
                {
                    key: item
                    for key, item in component.items()
                    if key != "component_cid"
                }
            )
        ):
            raise QualificationError("worker runtime component authority differs")
        observed_component_cids.append(str(component["component_cid"]))
    normalized_manifest: list[dict[str, Any]] = []
    observed_paths: set[str] = set()
    for item in payload_manifest:
        if not isinstance(item, dict) or set(item) != {
            "path",
            "sha256",
            "size_bytes",
            "projection_mode_octal",
            "native",
        }:
            raise QualificationError("worker runtime payload manifest fields differ")
        path = str(item.get("path") or "")
        _strict_record_relative_path(path, noun="worker runtime payload")
        if (
            path in observed_paths
            or item.get("projection_mode_octal")
            not in ({"0500"} if item.get("native") is True else {"0400"})
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(item.get("sha256") or ""))
            or isinstance(item.get("size_bytes"), bool)
            or not isinstance(item.get("size_bytes"), int)
            or int(item["size_bytes"]) < 0
            or bool(item.get("native")) != _qualification_runtime_is_native(path)
        ):
            raise QualificationError("worker runtime payload manifest differs")
        observed_paths.add(path)
        normalized_manifest.append(dict(item))
    if (
        normalized_manifest
        != sorted(normalized_manifest, key=lambda item: str(item["path"]))
        or content_identity(normalized_manifest) != bundle.get("file_manifest_root")
        or len(normalized_manifest) != bundle.get("file_count")
        or sum(int(item["size_bytes"]) for item in normalized_manifest)
        != bundle.get("total_bytes")
        or bundle.get("file_count") != _MAX_QUALIFICATION_RUNTIME_PAYLOAD_FILES
        or bundle.get("total_bytes") != _QUALIFICATION_RUNTIME_PAYLOAD_BYTES
        or observed_component_cids != projection.get("component_cids")
    ):
        raise QualificationError("worker runtime payload authority differs")
    synthetic_resolved = _ResolvedQualificationRuntime(
        bundle=bundle,
        components=tuple(components),
        payload_manifest=tuple(normalized_manifest),
        payload_bytes={},
        native_source_observation={},
    )
    synthetic_active = _ActiveQualificationRuntime(
        resolved=synthetic_resolved,
        directory=_ISOLATED_RECOVERY_PYCACHE_DIRECTORY,  # unused by validation
        root=runtime_root,
        root_fd=runtime_fd,
        projection=projection,
        control_manifest_path=control_path,
    )
    _validate_qualification_runtime_projection(synthetic_active)
    native_platform = bundle.get("native_platform_binding")
    payload_native = (
        native_platform.get("solver_payload_native_files")
        if isinstance(native_platform, Mapping)
        else None
    )
    if not isinstance(payload_native, list):
        raise QualificationError("worker native runtime authority is absent")
    observed_host, _host_observation = _native_host_runtime_binding(payload_native)
    if (
        not isinstance(native_platform, Mapping)
        or native_platform.get("native_host_runtime") != observed_host
        or native_platform.get("native_host_runtime_root")
        != observed_host.get("host_runtime_cid")
    ):
        raise QualificationError("worker native host runtime differs")
    expected_runtime_cid, policy_binding = _recovery_qualification_policy(
        ROOT,
        head_bound=False,
    )
    if expected_runtime_cid != bundle.get("runtime_cid"):
        raise QualificationError("worker qualification runtime policy differs")
    return bundle, bootstrap, runtime_root, policy_binding


_RECOVERY_WORKER_NATIVE_HANDLES: list[Any] = []


def _stdlib_extension_native_manifest(
) -> tuple[dict[str, dict[str, Any]], str, int]:
    """Bind the exact root-owned stdlib extension allowlist as raw bytes."""

    relative_root = "usr/lib/python3.12/lib-dynload"
    root_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
    root_flags |= getattr(os, "O_NOFOLLOW", 0)
    system_fd = os.open("/", root_flags)
    directory_fd = os.open(relative_root, root_flags, dir_fd=system_fd)
    try:
        directory_status = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(directory_status.st_mode)
            or directory_status.st_uid != 0
            or directory_status.st_nlink < 1
            or stat.S_IMODE(directory_status.st_mode) & 0o022
        ):
            raise QualificationError("stdlib extension directory identity differs")
        names: list[str] = []
        observed_entries = 0
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                observed_entries += 1
                if observed_entries > 4_096:
                    raise QualificationError(
                        "stdlib extension directory population exceeds its bound"
                    )
                if not _qualification_runtime_is_native(entry.name):
                    raise QualificationError(
                        "stdlib extension directory contains an unbound entry"
                    )
                names.append(entry.name)
        names.sort()
        manifest: list[dict[str, Any]] = []
        by_path: dict[str, dict[str, Any]] = {}
        for name in names:
            relative = f"{relative_root}/{name}"
            payload, observed = _read_qualification_runtime_relative(
                system_fd,
                relative,
                noun="stdlib extension allowlist member",
                owner_uid=0,
            )
            absolute = "/" + relative
            item = {
                "path_token": "stdlib-lib-dynload:" + name,
                "sha256": _sha256_bytes(payload),
                "size_bytes": len(payload),
                "mode": observed["mode"],
                "uid": observed["uid"],
                "gid": observed["gid"],
                "nlink": observed["nlink"],
            }
            manifest.append(item)
            by_path[absolute] = item
        if not manifest:
            raise QualificationError("stdlib extension allowlist is empty")
        return (
            by_path,
            content_identity(manifest),
            sum(int(item["size_bytes"]) for item in manifest),
        )
    except OSError as exc:
        raise QualificationError("stdlib extension allowlist is unavailable") from exc
    finally:
        os.close(directory_fd)
        os.close(system_fd)


def _native_executable_mappings(
    process_id: int | None = None,
) -> list[dict[str, Any]]:
    """Return every executable VMA without collapsing repeated file mappings."""

    if process_id is not None and (
        isinstance(process_id, bool) or not isinstance(process_id, int) or process_id <= 0
    ):
        raise QualificationError("worker native process identity is malformed")
    process_token = "self" if process_id is None else str(process_id)
    flags = os.O_RDONLY | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(f"/proc/{process_token}/maps", flags)
        try:
            chunks = bytearray()
            while len(chunks) <= 4 * 1024 * 1024:
                chunk = os.read(descriptor, min(64 * 1024, 4 * 1024 * 1024 + 1 - len(chunks)))
                if not chunk:
                    break
                chunks.extend(chunk)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise QualificationError("worker native process map is unavailable") from exc
    maps_bytes = bytes(chunks)
    if len(maps_bytes) > 4 * 1024 * 1024 or b" (deleted)" in maps_bytes:
        raise QualificationError("worker native process map differs")
    result: list[dict[str, Any]] = []
    for raw_line in maps_bytes.splitlines():
        fields = raw_line.split(maxsplit=5)
        if len(fields) < 5 or b"x" not in fields[1]:
            continue
        try:
            address_text = fields[0].decode("ascii", errors="strict")
            permissions = fields[1].decode("ascii", errors="strict")
            offset = fields[2].decode("ascii", errors="strict")
            device = fields[3].decode("ascii", errors="strict")
            inode = fields[4].decode("ascii", errors="strict")
        except UnicodeDecodeError as exc:
            raise QualificationError("worker executable mapping is malformed") from exc
        address = address_text.split("-", 1)
        if (
            len(address) != 2
            or re.fullmatch(r"[0-9a-f]+", address[0]) is None
            or re.fullmatch(r"[0-9a-f]+", address[1]) is None
            or int(address[0], 16) >= int(address[1], 16)
            or re.fullmatch(r"[r-][w-][x-][ps]", permissions) is None
            or re.fullmatch(r"[0-9a-f]+", offset) is None
            or re.fullmatch(r"[0-9a-f]+:[0-9a-f]+", device) is None
            or not inode.isdecimal()
        ):
            raise QualificationError("worker executable mapping is malformed")
        label = ""
        resolved_path: str | None = None
        path_kind = "anonymous"
        if len(fields) == 6:
            try:
                label = fields[5].decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise QualificationError(
                    "worker executable mapped path is malformed"
                ) from exc
            if label.startswith("/"):
                try:
                    resolved_path = str(Path(label).resolve(strict=True))
                except OSError as exc:
                    raise QualificationError(
                        "worker executable mapped path is unavailable"
                    ) from exc
                path_kind = "absolute_file"
            elif re.fullmatch(r"\[[A-Za-z0-9_.:-]+\]", label) is not None:
                path_kind = "kernel"
            else:
                raise QualificationError("worker executable map label differs")
        result.append(
            {
                "address_start": address[0],
                "address_end": address[1],
                "permissions": permissions,
                "offset": offset,
                "device": device,
                "inode": inode,
                "path_kind": path_kind,
                "resolved_path": resolved_path,
                "label": label if path_kind == "kernel" else "",
            }
        )
        if len(result) > _MAX_RECOVERY_EXECUTABLE_MAPPINGS:
            raise QualificationError(
                "worker executable mapping population exceeds its bound"
            )
    if not result:
        raise QualificationError("worker executable mapping population is empty")
    result.sort(key=lambda item: int(str(item["address_start"]), 16))
    if len({content_identity(item) for item in result}) != len(result):
        raise QualificationError("worker executable mapping population differs")
    return result


def _native_executable_mapping_root(
    mappings: Sequence[Mapping[str, Any]],
) -> str:
    """Commit the exact internal VMA list, including addresses and identities."""

    return content_identity([dict(item) for item in mappings])


def _normalized_executable_mapping_signatures(
    mappings: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build a public VMA commitment without ASLR or lexical host paths."""

    result: list[dict[str, Any]] = []
    for item in mappings:
        try:
            size_bytes = int(str(item["address_end"]), 16) - int(
                str(item["address_start"]), 16
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise QualificationError("worker executable mapping range is malformed") from exc
        path_kind = item.get("path_kind")
        resolved_path = item.get("resolved_path")
        label = item.get("label")
        if path_kind == "absolute_file" and isinstance(resolved_path, str):
            path_token = "absolute:" + _sha256_bytes(resolved_path.encode("utf-8"))
        elif path_kind == "kernel" and isinstance(label, str):
            path_token = "kernel:" + label
        elif path_kind == "anonymous" and resolved_path is None and label == "":
            path_token = "anonymous"
        else:
            raise QualificationError("worker executable mapping path differs")
        result.append(
            {
                "kind": path_kind,
                "permissions": item.get("permissions"),
                "offset": item.get("offset"),
                "size_bytes": size_bytes,
                "path_token": path_token,
            }
        )
    result.sort(key=_canonical_bytes)
    return result


def _writable_executable_mapping_signatures(
    mappings: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize every W+X mapping without retaining randomized addresses."""

    result: list[dict[str, Any]] = []
    for item in mappings:
        permissions = item.get("permissions")
        if not isinstance(permissions, str) or "w" not in permissions or "x" not in permissions:
            continue
        try:
            size_bytes = int(str(item["address_end"]), 16) - int(
                str(item["address_start"]), 16
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise QualificationError("W+X mapping range is malformed") from exc
        path_kind = item.get("path_kind")
        resolved_path = item.get("resolved_path")
        label = item.get("label")
        if path_kind not in {"absolute_file", "anonymous", "kernel"}:
            raise QualificationError("W+X mapping path kind differs")
        result.append(
            {
                "kind": path_kind,
                "permissions": permissions,
                "offset": item.get("offset"),
                "device": item.get("device"),
                "inode": item.get("inode"),
                "size_bytes": size_bytes,
                "path_token": (
                    "absolute:" + _sha256_bytes(str(resolved_path).encode())
                    if path_kind == "absolute_file"
                    else "kernel:" + str(label)
                    if path_kind == "kernel"
                    else "anonymous"
                ),
            }
        )
    result.sort(key=_canonical_bytes)
    return result


def _controller_zero_wx_observation(*, phase: str) -> dict[str, Any]:
    """Prove the authority-bearing parent never carries a writable code VMA."""

    if phase not in {
        "qualification_entry",
        "before_worker",
        "prepared_parent_inspection",
        "final_parent_inspection",
        "after_worker_exit",
        "qualification_final",
        "materializer_entry",
        "materializer_final",
    }:
        raise QualificationError("controller W+X observation phase differs")
    mappings = _native_executable_mappings()
    writable = _writable_executable_mapping_signatures(mappings)
    if writable:
        raise QualificationError("controller contains a writable executable mapping")
    value: dict[str, Any] = {
        "schema": "lgcvf-recovery-controller-zero-wx-observation@1",
        "phase": phase,
        "executable_mapping_count": len(mappings),
        "normalized_executable_mapping_root": content_identity(
            _normalized_executable_mapping_signatures(mappings)
        ),
        "controller_wx_mapping_count": 0,
        "controller_rwx_permitted": False,
    }
    value["observation_cid"] = content_identity(value)
    return value


def _validate_controller_zero_wx_observation(
    value: Any,
    *,
    phase: str,
) -> dict[str, Any]:
    """Validate a historical normalized controller observation without replay."""

    if not isinstance(value, Mapping):
        raise QualificationError("controller W+X observation is absent")
    body = {key: item for key, item in value.items() if key != "observation_cid"}
    executable_count = value.get("executable_mapping_count")
    if (
        set(value)
        != {
            "schema",
            "phase",
            "executable_mapping_count",
            "normalized_executable_mapping_root",
            "controller_wx_mapping_count",
            "controller_rwx_permitted",
            "observation_cid",
        }
        or value.get("schema")
        != "lgcvf-recovery-controller-zero-wx-observation@1"
        or value.get("phase") != phase
        or isinstance(executable_count, bool)
        or not isinstance(executable_count, int)
        or executable_count <= 0
        or not _is_canonical_content_cid(
            value.get("normalized_executable_mapping_root")
        )
        or isinstance(value.get("controller_wx_mapping_count"), bool)
        or not isinstance(value.get("controller_wx_mapping_count"), int)
        or value.get("controller_wx_mapping_count") != 0
        or value.get("controller_rwx_permitted") is not False
        or value.get("observation_cid") != content_identity(body)
    ):
        raise QualificationError("controller W+X observation differs")
    return dict(value)


def _recovery_live_thread_population() -> list[dict[str, Any]]:
    """Bind the exact live thread population around judged pytest execution."""

    result: list[dict[str, Any]] = []
    for thread in threading.enumerate():
        if thread.ident is None or thread.native_id is None:
            raise QualificationError("recovery worker thread identity is absent")
        result.append(
            {
                "ident": int(thread.ident),
                "native_id": int(thread.native_id),
                "name": str(thread.name),
                "daemon": bool(thread.daemon),
            }
        )
    result.sort(key=lambda item: (int(item["ident"]), str(item["name"])))
    if not result or len(result) > 128:
        raise QualificationError("recovery worker thread population differs")
    return result


def _recovery_kernel_task_population(process_id: int | None = None) -> list[int]:
    """Bind every kernel-visible task in the one-shot recovery worker."""

    if process_id is not None and (
        isinstance(process_id, bool) or not isinstance(process_id, int) or process_id <= 0
    ):
        raise QualificationError("kernel task process identity is malformed")
    process_token = "self" if process_id is None else str(process_id)
    result: list[int] = []
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(f"/proc/{process_token}/task", flags)
    try:
        with os.scandir(descriptor) as entries:
            for entry in entries:
                if not entry.name.isascii() or not entry.name.isdecimal():
                    raise QualificationError("kernel task identity is malformed")
                metadata = entry.stat(follow_symlinks=False)
                if not stat.S_ISDIR(metadata.st_mode):
                    raise QualificationError("kernel task entry differs")
                result.append(int(entry.name))
                if len(result) > 128:
                    raise QualificationError(
                        "kernel task population exceeds its bound"
                    )
    except OSError as exc:
        raise QualificationError("kernel task population is unavailable") from exc
    finally:
        os.close(descriptor)
    result.sort()
    if not result or len(set(result)) != len(result):
        raise QualificationError("kernel task population differs")
    return result


def _read_recovery_task_file(
    task_descriptor: int,
    name: str,
    *,
    limit: int,
) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=task_descriptor)
    try:
        before = os.fstat(descriptor)
        payload = os.read(descriptor, limit + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        len(payload) > limit
        or not stat.S_ISREG(before.st_mode)
        or (before.st_dev, before.st_ino, before.st_mode)
        != (after.st_dev, after.st_ino, after.st_mode)
    ):
        raise QualificationError("kernel task identity file differs")
    return payload


def _recovery_kernel_task_records(
    process_id: int | None = None,
) -> list[dict[str, Any]]:
    """Stable-read every task identity and its nofollow proc directory inode."""

    if process_id is not None and (
        isinstance(process_id, bool) or not isinstance(process_id, int) or process_id <= 0
    ):
        raise QualificationError("kernel task process identity is malformed")
    process_token = "self" if process_id is None else str(process_id)
    task_root_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
    task_root_flags |= getattr(os, "O_NOFOLLOW", 0)
    task_flags = task_root_flags
    root_descriptor = os.open(f"/proc/{process_token}/task", task_root_flags)
    try:
        def enumerate_ids() -> list[int]:
            identities: list[int] = []
            with os.scandir(root_descriptor) as entries:
                for entry in entries:
                    if not entry.name.isascii() or not entry.name.isdecimal():
                        raise QualificationError("kernel task identity is malformed")
                    metadata = entry.stat(follow_symlinks=False)
                    if not stat.S_ISDIR(metadata.st_mode):
                        raise QualificationError("kernel task entry differs")
                    identities.append(int(entry.name))
                    if len(identities) > 128:
                        raise QualificationError(
                            "kernel task population exceeds its bound"
                        )
            identities.sort()
            if not identities or len(set(identities)) != len(identities):
                raise QualificationError("kernel task population differs")
            return identities

        task_ids = enumerate_ids()
        records: list[dict[str, Any]] = []
        for task_id in task_ids:
            descriptor = os.open(
                str(task_id),
                task_flags,
                dir_fd=root_descriptor,
            )
            try:
                before = os.fstat(descriptor)
                stat_bytes = _read_recovery_task_file(
                    descriptor,
                    "stat",
                    limit=16 * 1024,
                )
                comm_bytes = _read_recovery_task_file(
                    descriptor,
                    "comm",
                    limit=256,
                )
                after = os.fstat(descriptor)
                lexical = os.stat(
                    str(task_id),
                    dir_fd=root_descriptor,
                    follow_symlinks=False,
                )
            finally:
                os.close(descriptor)
            if (
                not stat.S_ISDIR(before.st_mode)
                or (before.st_dev, before.st_ino, before.st_mode)
                != (after.st_dev, after.st_ino, after.st_mode)
                or (before.st_dev, before.st_ino, before.st_mode)
                != (lexical.st_dev, lexical.st_ino, lexical.st_mode)
            ):
                raise QualificationError("kernel task directory identity differs")
            closing = stat_bytes.rfind(b") ")
            if (
                closing < 1
                or not stat_bytes.startswith(str(task_id).encode("ascii") + b" (")
            ):
                raise QualificationError("kernel task stat identity differs")
            fields = stat_bytes[closing + 2 :].split()
            if (
                len(fields) < 20
                or not fields[19].isascii()
                or not fields[19].isdigit()
                or int(fields[19]) <= 0
            ):
                raise QualificationError("kernel task start identity differs")
            try:
                comm = comm_bytes.rstrip(b"\n").decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise QualificationError("kernel task comm identity differs") from exc
            if not comm or len(comm.encode("utf-8")) > 64 or "\n" in comm:
                raise QualificationError("kernel task comm identity differs")
            records.append(
                {
                    "tid": task_id,
                    "start_time_ticks": int(fields[19]),
                    "comm": comm,
                    "dir_dev": int(before.st_dev),
                    "dir_ino": int(before.st_ino),
                }
            )
        if enumerate_ids() != task_ids:
            raise QualificationError("kernel task population raced inspection")
    except OSError as exc:
        raise QualificationError("kernel task identity is unavailable") from exc
    finally:
        os.close(root_descriptor)
    return records


def _normalized_kernel_task_role_evidence(
    records: Sequence[Mapping[str, Any]],
    *,
    task_id: str,
) -> list[dict[str, Any]]:
    if task_id == "LGCVF-051":
        roles: list[str | int] = [
            "leader" if index == 0 else f"z3_timeout_task_{index}"
            for index in range(len(records))
        ]
    elif task_id in {
        "LGCVF-060",
        "LGCVF-061",
        "LGCVF-070",
        "LGCVF-071",
        "LGCVF-080",
    }:
        roles = list(range(len(records)))
    else:
        raise QualificationError("normalized kernel task suite differs")
    normalized = [
        {"role": role, "comm": str(item.get("comm") or "")}
        for role, item in zip(roles, records, strict=True)
    ]
    if (
        not normalized
        or len(normalized) > 128
        or any(not item["comm"] for item in normalized)
    ):
        raise QualificationError("normalized kernel task roles differ")
    return normalized


def _hold_recovery_task_directories(
    process_id: int,
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Hold every PREPARED task inode until FINAL/terminal cleanup."""

    root_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
    root_flags |= getattr(os, "O_NOFOLLOW", 0)
    if (
        not records
        or len(records) > 128
        or len({item.get("tid") for item in records}) != len(records)
    ):
        raise QualificationError("held kernel task population differs")
    root_descriptor = os.open(f"/proc/{process_id}/task", root_flags)
    held: list[dict[str, Any]] = []
    try:
        for raw in records:
            record = dict(raw)
            task_id = record.get("tid")
            if isinstance(task_id, bool) or not isinstance(task_id, int):
                raise QualificationError("held kernel task identity differs")
            descriptor = os.open(
                str(task_id),
                root_flags,
                dir_fd=root_descriptor,
            )
            try:
                metadata = os.fstat(descriptor)
                if (
                    not stat.S_ISDIR(metadata.st_mode)
                    or metadata.st_dev != record.get("dir_dev")
                    or metadata.st_ino != record.get("dir_ino")
                ):
                    raise QualificationError("held kernel task inode differs")
                held.append({"record": record, "descriptor": descriptor})
            except BaseException:
                os.close(descriptor)
                raise
    except BaseException:
        for item in held:
            os.close(int(item["descriptor"]))
        raise
    finally:
        os.close(root_descriptor)
    return held


def _revalidate_held_recovery_task_directories(
    process_id: int,
    held: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expected = [dict(item["record"]) for item in held]
    observed = _recovery_kernel_task_records(process_id)
    if observed != expected:
        raise QualificationError("held kernel task population changed")
    root_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
    root_flags |= getattr(os, "O_NOFOLLOW", 0)
    root_descriptor = os.open(f"/proc/{process_id}/task", root_flags)
    try:
        for item in held:
            record = dict(item["record"])
            descriptor = int(item["descriptor"])
            held_status = os.fstat(descriptor)
            reopened = os.open(
                str(record["tid"]),
                root_flags,
                dir_fd=root_descriptor,
            )
            try:
                reopened_status = os.fstat(reopened)
            finally:
                os.close(reopened)
            expected_identity = (record["dir_dev"], record["dir_ino"])
            if (
                not stat.S_ISDIR(held_status.st_mode)
                or (held_status.st_dev, held_status.st_ino) != expected_identity
                or (reopened_status.st_dev, reopened_status.st_ino)
                != expected_identity
            ):
                raise QualificationError("held kernel task directory was replaced")
    except OSError as exc:
        raise QualificationError("held kernel task directory is unavailable") from exc
    finally:
        os.close(root_descriptor)
    return observed


def _close_held_recovery_task_directories(
    held: list[dict[str, Any]],
) -> None:
    """Verify, close, and consume one task-directory lease exactly once."""

    items = list(held)
    held.clear()
    errors: list[BaseException] = []
    for item in items:
        descriptor = int(item["descriptor"])
        try:
            status = os.fstat(descriptor)
            record = item.get("record")
            if (
                not isinstance(record, Mapping)
                or not stat.S_ISDIR(status.st_mode)
                or (status.st_dev, status.st_ino)
                != (record.get("dir_dev"), record.get("dir_ino"))
            ):
                errors.append(
                    QualificationError(
                        "held kernel task descriptor identity changed"
                    )
                )
        except BaseException as exc:
            errors.append(exc)
        finally:
            try:
                os.close(descriptor)
            except OSError as exc:
                errors.append(exc)
    if errors:
        raise QualificationError(
            "held kernel task descriptor cleanup failed"
        ) from errors[0]


def _recovery_process_stat_identity(
    process_id: int | None = None,
    *,
    require_positive_group: bool = True,
) -> dict[str, int]:
    """Read PID, process-group, session, and start-time tokens from procfs."""

    if process_id is not None and (
        isinstance(process_id, bool) or not isinstance(process_id, int) or process_id <= 0
    ):
        raise QualificationError("worker process identity is malformed")
    process_token = "self" if process_id is None else str(process_id)
    flags = os.O_RDONLY | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(f"/proc/{process_token}/stat", flags)
        try:
            payload = os.read(descriptor, 16 * 1024 + 1)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise QualificationError("worker process start identity is unavailable") from exc
    if len(payload) > 16 * 1024 or b"\n" in payload.rstrip(b"\n"):
        raise QualificationError("worker process start identity is malformed")
    closing = payload.rfind(b") ")
    if closing < 1:
        raise QualificationError("worker process start identity is malformed")
    prefix = payload[:closing]
    expected_pid = os.getpid() if process_id is None else process_id
    if not prefix.startswith(str(expected_pid).encode("ascii") + b" ("):
        raise QualificationError("worker process start identity differs")
    fields = payload[closing + 2 :].split()
    if (
        len(fields) < 20
        or any(
            not fields[index].isascii() or not fields[index].isdigit()
            for index in (2, 3, 19)
        )
    ):
        raise QualificationError("worker process start identity is malformed")
    process_group = int(fields[2])
    session = int(fields[3])
    start_time = int(fields[19])
    if require_positive_group and (
        process_group <= 0 or session <= 0 or start_time <= 0
    ):
        raise QualificationError("worker process start identity differs")
    return {
        "process_id": expected_pid,
        "process_group_id": process_group,
        "session_id": session,
        "start_time_ticks": start_time,
    }


def _recovery_process_start_time_ticks(process_id: int | None = None) -> int:
    """Read the kernel process start-time token used with a held pidfd."""

    return _recovery_process_stat_identity(process_id)["start_time_ticks"]


def _recovery_process_group_population(
    leader_process_id: int | None = None,
    *,
    require_leader: bool = True,
) -> list[dict[str, int]]:
    """Bind the entire protected process group/session, including reparented peers."""

    leader = os.getpid() if leader_process_id is None else leader_process_id
    if isinstance(leader, bool) or not isinstance(leader, int) or leader <= 0:
        raise QualificationError("worker process-group leader is malformed")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        proc_descriptor = os.open("/proc", flags)
    except OSError as exc:
        raise QualificationError("worker process-group inventory is unavailable") from exc
    observed = 0
    result: list[dict[str, int]] = []
    try:
        with os.scandir(proc_descriptor) as entries:
            for entry in entries:
                observed += 1
                if observed > _MAX_RECOVERY_PROC_ENTRIES:
                    raise QualificationError(
                        "worker process-group inventory exceeds its bound"
                    )
                if not entry.name.isascii() or not entry.name.isdecimal():
                    continue
                process_id = int(entry.name)
                try:
                    identity = _recovery_process_stat_identity(
                        process_id,
                        require_positive_group=False,
                    )
                except QualificationError as exc:
                    # Processes outside the held group may exit while procfs is
                    # streamed.  The held leader and any observed matching peer
                    # are checked again by the stable outer snapshot.
                    if process_id == leader:
                        raise
                    cause = exc.__cause__
                    if isinstance(cause, OSError) and cause.errno in {
                        errno.ENOENT,
                        errno.ESRCH,
                    }:
                        continue
                    raise
                if (
                    identity["process_group_id"] == leader
                    or identity["session_id"] == leader
                ):
                    if (
                        identity["start_time_ticks"] <= 0
                        or identity["process_group_id"] != leader
                        or identity["session_id"] != leader
                    ):
                        raise QualificationError(
                            "worker process-group/session identity differs"
                        )
                    result.append(identity)
                    if len(result) > 128:
                        raise QualificationError(
                            "worker process-group population exceeds its bound"
                        )
    finally:
        os.close(proc_descriptor)
    result.sort(key=lambda item: int(item["process_id"]))
    if len({item["process_id"] for item in result}) != len(result):
        raise QualificationError("worker process-group population differs")
    leader_entries = [item for item in result if item["process_id"] == leader]
    if require_leader and len(leader_entries) != 1:
        raise QualificationError("worker process-group leader differs")
    if not require_leader and leader_entries:
        raise QualificationError("terminal worker process-group leader remains")
    return result


def _recovery_child_process_population(
    process_id: int | None = None,
    *,
    task_ids: Sequence[int] | None = None,
) -> list[int]:
    """Bind every direct child reported for every kernel task."""

    if process_id is not None and (
        isinstance(process_id, bool) or not isinstance(process_id, int) or process_id <= 0
    ):
        raise QualificationError("child process owner identity is malformed")
    expected_tasks = (
        _recovery_kernel_task_population(process_id)
        if task_ids is None
        else list(task_ids)
    )
    if (
        not expected_tasks
        or expected_tasks != sorted(set(expected_tasks))
        or len(expected_tasks) > 128
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in expected_tasks)
    ):
        raise QualificationError("child process task population differs")
    process_token = "self" if process_id is None else str(process_id)
    result: list[int] = []
    flags = os.O_RDONLY | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    for task_id in expected_tasks:
        try:
            descriptor = os.open(
                f"/proc/{process_token}/task/{task_id}/children", flags
            )
            try:
                payload = os.read(descriptor, 16 * 1024 + 1)
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise QualificationError("child process population is unavailable") from exc
        if len(payload) > 16 * 1024:
            raise QualificationError("child process population exceeds its bound")
        for token in payload.split():
            if not token.isascii() or not token.isdigit():
                raise QualificationError("child process identity is malformed")
            result.append(int(token))
            if len(result) > 128:
                raise QualificationError(
                    "child process population exceeds its bound"
                )
    result.sort()
    if len(set(result)) != len(result):
        raise QualificationError("child process population differs")
    return result


def _wait_for_recovery_execution_quiescence(
    *,
    expected_threads: Sequence[Mapping[str, Any]],
    expected_tasks: Sequence[int],
    expected_children: Sequence[int],
    timeout_seconds: float = 2.0,
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    """Give pytest-owned teardown tasks one small bounded window to disappear."""

    deadline = time.monotonic() + timeout_seconds
    while True:
        threads = _recovery_live_thread_population()
        tasks = _recovery_kernel_task_population()
        children = _recovery_child_process_population(task_ids=tasks)
        if (
            threads == [dict(item) for item in expected_threads]
            and tasks == list(expected_tasks)
            and children == list(expected_children)
        ):
            return threads, tasks, children
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            diagnostic_classes: list[str] = []
            for task_id in tasks:
                flags = os.O_RDONLY | os.O_CLOEXEC
                flags |= getattr(os, "O_NOFOLLOW", 0)
                try:
                    descriptor = os.open(f"/proc/self/task/{task_id}/comm", flags)
                    try:
                        raw_name = os.read(descriptor, 256)
                    finally:
                        os.close(descriptor)
                    name = raw_name.rstrip(b"\n").decode(
                        "utf-8", errors="strict"
                    )
                except (OSError, UnicodeError, QualificationError):
                    name = "<unavailable>"
                diagnostic_classes.append(name[:64])
            diagnostic_classes.sort()
            raise QualificationError(
                "recovery execution tasks did not quiesce: "
                + str(len(expected_tasks))
                + ":"
                + str(len(tasks))
                + ":"
                + content_identity(list(expected_tasks))
                + ":"
                + content_identity(tasks)
                + ":"
                + content_identity(diagnostic_classes)
                + ":"
                + json.dumps(diagnostic_classes, separators=(",", ":"))
            )
        try:
            select.select([], [], [], min(0.05, remaining))
        except OSError as exc:
            raise QualificationError(
                "recovery execution quiescence wait failed"
            ) from exc


def _recovery_suite_native_policy(*, z3_required: bool) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": "lgcvf-recovery-suite-native-policy@1",
        "z3_required": bool(z3_required),
        "controller_rwx_permitted": False,
        "worker_rwx_disposition": (
            "exact_z3_libffi_anonymous_4096_rwxp"
            if z3_required
            else "zero_writable_executable"
        ),
    }
    value["policy_cid"] = content_identity(value)
    return value


_RECOVERY_061_PYTEST_CALL_NAMES: Final[tuple[str, ...]] = (
    "test_closed_dispositions_are_exactly_the_required_terminals",
    "test_safe_cfg_is_proved_without_refinement",
    "test_spurious_trace_refines_with_validated_interpolant",
    "test_spurious_trace_refines_with_validated_unsat_core",
    "test_spurious_trace_refines_with_weakest_precondition",
    "test_spurious_trace_refines_with_reviewed_predicate",
    "test_unreviewed_predicate_is_rejected",
    "test_real_trace_remains_counterexample",
    "test_real_trace_is_not_refined_away",
    "test_iteration_budget_exhausts_on_remaining_spurious_trace",
    "test_predicate_budget_exhausts_when_refinement_cannot_grow",
    "test_timeout_terminates",
    "test_path_timeout_terminates",
    "test_unavailable_solver_terminates",
    "test_unknown_path_check_terminates",
    "test_unknown_when_no_refinement_authority_applies",
    "test_every_run_has_exactly_one_closed_disposition",
    "test_refinement_binds_partitions_vocabulary_theory_provider_bounds_and_identities",
    "test_non_interpolant_refinement_does_not_fabricate_an_interpolant",
    "test_receipt_identity_is_stable_for_identical_runs",
    "test_malformed_system_is_rejected",
    "test_disproved_receipt_requires_a_real_counterexample",
    "test_spurious_list_cannot_hold_a_real_trace",
    "test_scripted_solver_can_answer_by_query_id",
    "test_live_incremental_smt_real_trace_stays_a_counterexample",
    "test_live_incremental_smt_spurious_trace_is_refined_or_typed",
    "test_default_backends_never_fabricate_an_interpolant_on_the_spurious_example",
)


def _recovery_061_expected_pytest_call_nodeids() -> list[str]:
    prefix = "tests/unit/logic/software_verification/test_cegar.py::"
    return [prefix + name for name in _RECOVERY_061_PYTEST_CALL_NAMES]


_RECOVERY_061_TRUSTED_SOURCE_LOGICAL_PATH: Final[str] = (
    "ipfs_datasets_py/ipfs_datasets_py/logic/backends/z3/compiler.py"
)


def _recovery_z3_expected_trusted_source_events(
    *, task_id: str,
    source_projection_root: str | None = None,
) -> list[dict[str, Any]]:
    """Return the exact host-ephemeral-free one-use source capability."""

    if task_id != "LGCVF-061":
        return []
    if source_projection_root is not None and not _is_canonical_content_cid(
        source_projection_root
    ):
        raise QualificationError(
            "recovery trusted source projection identity differs"
        )
    return [
        {
            "event": "open",
            "logical_path": _RECOVERY_061_TRUSTED_SOURCE_LOGICAL_PATH,
            "component_index": 4,
            "component": "z3",
            "directory_flags": _recovery_projection_directory_flags(),
            "caller_code_identity": (
                "_read_recovery_projection_source.__code__"
            ),
            **(
                {"source_projection_root": source_projection_root}
                if source_projection_root is not None
                else {}
            ),
            "disposition": (
                "permitted_by_one_use_trusted_source_revalidation_capability"
            ),
        }
    ]


def _recovery_z3_import_denial_policy(
    *, task_id: str, suite_id: str
) -> dict[str, Any]:
    """Return the config-pinned, least-capability Z3 namespace policy."""

    expected_suite = "recovery_" + task_id.casefold().replace("-", "_")
    if suite_id != expected_suite or task_id not in {
        item.task_id for item in RECOVERY_VALIDATIONS
    }:
        raise QualificationError("recovery Z3 import policy identity differs")
    call_nodeids: list[str] = []
    if task_id in _RECOVERY_Z3_IMPORT_DENIED_TASKS:
        disposition = "z3_import_denied_as_typed_unavailable"
        call_nodeids = _recovery_061_expected_pytest_call_nodeids()
        expected_denials = [
            {
                "ordinal": ordinal,
                "pytest_call_ordinal": call_ordinal,
                "nodeid": call_nodeids[call_ordinal - 1],
                "module": "z3",
                "disposition": "denied_before_loader_as_typed_unavailable",
                "meta_path_identity_exact": True,
                "owner_thread_only": True,
                "z3_modules_absent": True,
                "z3_file_descriptor_count": 0,
                "z3_file_descriptor_root": content_identity([]),
                "z3_native_mapping_count": 0,
                "z3_native_mapping_root": content_identity([]),
            }
            for ordinal, call_ordinal in enumerate((25, 26, 27), start=1)
        ]
        module_absence_required = True
        namespace_unavailability = True
        live_cegar_disposition = "not_exercised_policy_namespace_unavailable"
        candidate_reason_interpretation = (
            "z3 Python API is not installed means policy namespace unavailable "
            "inside the sealed LGCVF-061 worker"
        )
    elif task_id in _RECOVERY_Z3_REQUIRED_TASKS:
        disposition = "bound_z3_import_admitted_for_other_suite_semantics"
        expected_denials = []
        module_absence_required = False
        namespace_unavailability = False
        live_cegar_disposition = "not_applicable"
        candidate_reason_interpretation = "not_applicable"
    else:
        disposition = "z3_import_not_expected"
        expected_denials = []
        module_absence_required = True
        namespace_unavailability = False
        live_cegar_disposition = "not_applicable"
        candidate_reason_interpretation = "not_applicable"
    expected_trusted_source_events = (
        _recovery_z3_expected_trusted_source_events(task_id=task_id)
    )
    value: dict[str, Any] = {
        "schema": "lgcvf-recovery-z3-import-denial-policy@3",
        "task_id": task_id,
        "suite_id": suite_id,
        "disposition": disposition,
        "denial_active": task_id in _RECOVERY_Z3_IMPORT_DENIED_TASKS,
        "denied_module_prefixes": (
            ["z3"] if task_id in _RECOVERY_Z3_IMPORT_DENIED_TASKS else []
        ),
        "interception": (
            "meta_path_semantic_denial_with_candidate_phase_audit_open_boundary"
            if task_id in _RECOVERY_Z3_IMPORT_DENIED_TASKS
            else "not_applicable"
        ),
        "candidate_denial_interval": (
            "guard_install_through_pytest_temp_cleanup_and_execution_quiescence"
            if task_id in _RECOVERY_Z3_IMPORT_DENIED_TASKS
            else "not_applicable"
        ),
        "trusted_postvalidation_read_scope": (
            "one_use_source_capability_then_owner_thread_runtime_revalidation"
            if task_id in _RECOVERY_Z3_IMPORT_DENIED_TASKS
            else "not_applicable"
        ),
        "trusted_source_revalidation_disposition": (
            "owner_thread_nonreentrant_exact_projection_component_once"
            if task_id in _RECOVERY_Z3_IMPORT_DENIED_TASKS
            else "not_applicable"
        ),
        "trusted_source_revalidation_global_z3_exemption": False,
        "trusted_source_revalidation_audit_dirfd_observed": False,
        "expected_trusted_source_events": expected_trusted_source_events,
        "expected_trusted_source_event_count": len(
            expected_trusted_source_events
        ),
        "expected_trusted_source_event_root": content_identity(
            expected_trusted_source_events
        ),
        "meta_path_removal_boundary": (
            "sealed_worker_process_exit"
            if task_id in _RECOVERY_Z3_IMPORT_DENIED_TASKS
            else "not_applicable"
        ),
        "pytest_meta_path_lifecycle": (
            "exact_assertion_rewrite_identity_reposition_and_return_restoration"
            if task_id in _RECOVERY_Z3_IMPORT_DENIED_TASKS
            else "not_applicable"
        ),
        "pytest_candidate_meta_path_seal_required": (
            task_id in _RECOVERY_Z3_IMPORT_DENIED_TASKS
        ),
        "process_tree_namespace_denial_claimed": False,
        "trusted_tracked_source_boundary": True,
        "expected_meta_denials": expected_denials,
        "expected_meta_denial_count": len(expected_denials),
        "expected_meta_denial_root": content_identity(expected_denials),
        "expected_pytest_call_count": len(call_nodeids),
        "expected_pytest_call_nodeid_root": content_identity(call_nodeids),
        "expected_open_boundary_denial_count": 0,
        "expected_open_boundary_denial_root": content_identity([]),
        "z3_module_absence_required": module_absence_required,
        "unbound_native_loads_permitted": False,
        "policy_namespace_unavailability": namespace_unavailability,
        "live_z3_cegar_disposition": live_cegar_disposition,
        "candidate_reason_interpretation": candidate_reason_interpretation,
        "infrastructure_not_proof": True,
        "cache_authority": False,
        "completion_authoritative": False,
    }
    value["policy_cid"] = content_identity(value)
    return value


def _recovery_z3_import_policy_commitments(
    *, task_id: str, suite_id: str
) -> dict[str, Any]:
    policy = _recovery_z3_import_denial_policy(
        task_id=task_id,
        suite_id=suite_id,
    )
    return {
        "z3_import_policy_disposition": policy["disposition"],
        "z3_import_policy_cid": policy["policy_cid"],
        "z3_expected_meta_denial_count": policy[
            "expected_meta_denial_count"
        ],
        "z3_expected_meta_denial_root": policy["expected_meta_denial_root"],
        "z3_expected_open_boundary_denial_count": policy[
            "expected_open_boundary_denial_count"
        ],
        "z3_expected_open_boundary_denial_root": policy[
            "expected_open_boundary_denial_root"
        ],
        "z3_trusted_source_revalidation_disposition": policy[
            "trusted_source_revalidation_disposition"
        ],
        "z3_expected_trusted_source_event_count": policy[
            "expected_trusted_source_event_count"
        ],
        "z3_expected_trusted_source_event_root": policy[
            "expected_trusted_source_event_root"
        ],
        "z3_policy_namespace_unavailability": policy[
            "policy_namespace_unavailability"
        ],
        "z3_live_cegar_disposition": policy["live_z3_cegar_disposition"],
        "z3_candidate_reason_interpretation": policy[
            "candidate_reason_interpretation"
        ],
    }


_RECOVERY_060_PUBLIC_PREFIX_DIAGNOSTIC_REFERENCE_CID: Final[str] = (
    "baguqeeradrkb7ml4pvbxmnspufmyy2tgchhpvjyje7tasllxsovbhmdtbmlq"
)
# Derived from the closed, host-ephemeral-free execution body below.  This is
# intentionally distinct from the five-run diagnostic reference above: the
# protected worker reconstructs this identity from every executed check.
_RECOVERY_060_PUBLIC_PREFIX_EXECUTION_CID: Final[str] = (
    "baguqeeraxn25g2cvpas3d4igbxshfzmpa4lciza5mgbvpzaf75n4ystgjseq"
)
_RECOVERY_060_PUBLIC_PREFIX_EVENT_ROOT: Final[str] = (
    "baguqeerabufnwaetwv3eyxcjkr3raeecjtxm3kqb6rojblrzu6wjqlse37ka"
)
_RECOVERY_060_PUBLIC_PREFIX_OPERATION_ROOT: Final[str] = (
    "baguqeera6utmafcvhggljsbwbdzd7cbvcesb75nlgdekbtiau77boah4kejq"
)
_RECOVERY_060_PUBLIC_PREFIX_TRANSITION_ROOT: Final[str] = (
    "baguqeerazsz4ciiduaro5o52qch2k54lwy2zyjmjpnofcgatsi7umeusgptq"
)
_RECOVERY_TASK_NORMALIZED_ROOTS: Final[dict[int, str]] = {
    1: "baguqeeraj6gw3kkp32qrdmcywqvqpoxivablrsd57kaywizei3vh3gg7b6da",
    2: "baguqeerafbstf6kghu3tzy7syialblzrjnrrikdff3fj6oxzteuk77o7sfva",
    3: "baguqeera2diqx7gvndqocoi3m3plcfpwq5mwklnbyozbadgtwpatfglwlihq",
}
_RECOVERY_051_TASK_NORMALIZED_ROOTS: Final[dict[int, str]] = {
    1: "baguqeeraojhboa6zm5ar2yq7u2arnzfkh7t3qt3rnyax3bqsax7go35mcufa",
    2: "baguqeeraqa6eji3lu2l25m37w7wcm3f4zni4skaahk6aywym3hh3irlb33fa",
    3: "baguqeerane2peoplh3wa5yh33xu42dmv725xsjgr6nfs36kiygmmosmu37ya",
}


def _recovery_task_normalized_roots(task_id: str) -> dict[int, str]:
    if task_id == "LGCVF-051":
        return _RECOVERY_051_TASK_NORMALIZED_ROOTS
    if task_id in {
        "LGCVF-060",
        "LGCVF-061",
        "LGCVF-070",
        "LGCVF-071",
        "LGCVF-080",
    }:
        return _RECOVERY_TASK_NORMALIZED_ROOTS
    raise QualificationError("recovery task normalized-root policy differs")


_RECOVERY_051_PUBLIC_PREFIX_DIAGNOSTIC_REFERENCE_CID: Final[str] = (
    "baguqeeraqzr2u5shnpnhtha6kd3yyzkoywfgjahthh6upvd4kfc53b3jw3rq"
)
_RECOVERY_051_FIVE_PROBE_AGGREGATE_CID: Final[str] = (
    "baguqeeral4g5m5vfgprrb7eoltx66ns2wtjxksyvhxpjbuz7yh545xcjjgda"
)
_RECOVERY_051_DIAGNOSTIC_PROFILE_CID: Final[str] = (
    "baguqeerauxwxb7zv5dfnp7gxu4ulvkurt6eq6ldavjphwgc7idmj5hwz7q5q"
)
_RECOVERY_051_HISTORICAL_NORMALIZED_EVENT_ROOT: Final[str] = (
    "baguqeeraazpqgnqxxleikj3gv5qtlppik3lb5jpiy3ybymxu5srt23atyokq"
)
_RECOVERY_051_PUBLIC_PREFIX_OPERATION_ROOT: Final[str] = (
    "baguqeeranutn5ek6ficbc27vzxf67vlomrxxgvbgvyq3rrs2s2wtregpyf4a"
)
_RECOVERY_051_PUBLIC_PREFIX_TRANSITION_ROOT: Final[str] = (
    "baguqeera6entqf75vfa7vyxvrg5qh5lpvqp353dwatlhcpapfeneeu27s76a"
)
_RECOVERY_051_TASK_POPULATION_AGGREGATE_ROOT: Final[str] = (
    "baguqeeral36im225pbbtuxxsjtjl5hqnipbl44plao3ev2t5gglrxvpofn3q"
)
_RECOVERY_051_CORE: Final[tuple[str, ...]] = (
    "A-to-B:obligation:0:guarantee",
    "A-to-B:obligation:0:negated-assumption",
)
_RECOVERY_051_LIFETIME_BOUNDARIES: Final[tuple[str, ...]] = (
    "cold_process_before_public_discharge",
    "public_discharge_returned_before_reference_drop",
    "drop_graph_receipt_result_references",
    "get_verification_api_reset_true",
    "gc_collect_after_reset",
    "stable_20_samples_50ms_task_count_3",
)


def _recovery_051_expected_diagnostic_check_events() -> list[dict[str, Any]]:
    return [
        {
            "ordinal": 1,
            "assertions": [
                (
                    "(=> track__bf1cdc460ab18e7ff01ceac3 "
                    "(and (<= 0 value_0) (>= 10 value_0)))"
                ),
                (
                    "(=> track__844d345912bc2c8e15452e18 "
                    "(not (and (<= 0 value_0) (>= 20 value_0))))"
                ),
            ],
            "assumptions": [
                "track__bf1cdc460ab18e7ff01ceac3",
                "track__844d345912bc2c8e15452e18",
            ],
            "result": "unsat",
            "before_task_count": 1,
            "after_task_count": 2,
        },
        {
            "ordinal": 2,
            "assertions": [
                "(and (<= 0 value_0) (>= 10 value_0))",
                "(not (and (<= 0 value_0) (>= 20 value_0)))",
            ],
            "assumptions": [],
            "result": "unsat",
            "before_task_count": 2,
            "after_task_count": 3,
        },
    ]


def _recovery_051_expected_check_events() -> list[dict[str, Any]]:
    diagnostic_events = _recovery_051_expected_diagnostic_check_events()
    phases = (
        "tracked_assumption_unsat",
        "independent_core_validation_unsat",
    )
    values: list[dict[str, Any]] = []
    for event, phase, before_root, after_root in zip(
        diagnostic_events,
        phases,
        (
            _RECOVERY_051_TASK_NORMALIZED_ROOTS[1],
            _RECOVERY_051_TASK_NORMALIZED_ROOTS[2],
        ),
        (
            _RECOVERY_051_TASK_NORMALIZED_ROOTS[2],
            _RECOVERY_051_TASK_NORMALIZED_ROOTS[3],
        ),
        strict=True,
    ):
        main = event["ordinal"] == 1
        values.append(
            {
                "ordinal": event["ordinal"],
                "operation_ordinal": 1,
                "operation": "verification_api.discharge_assume_guarantee",
                "phase": phase,
                "timeout_ms": 5_000,
                "assertions": list(event["assertions"]),
                "assertion_count": len(event["assertions"]),
                "assertion_root": content_identity(event["assertions"]),
                "assumptions": list(event["assumptions"]),
                "assumption_count": len(event["assumptions"]),
                "assumption_root": content_identity(event["assumptions"]),
                "result": event["result"],
                "result_root": content_identity(event["result"]),
                "produced_core_count": len(_RECOVERY_051_CORE) if main else 0,
                "produced_core_root": content_identity(
                    list(_RECOVERY_051_CORE) if main else []
                ),
                "consumed_core_count": 0 if main else len(_RECOVERY_051_CORE),
                "consumed_core_root": content_identity(
                    [] if main else list(_RECOVERY_051_CORE)
                ),
                "before_task_count": event["before_task_count"],
                "after_task_count": event["after_task_count"],
                "before_normalized_task_root": before_root,
                "after_normalized_task_root": after_root,
                "task_identity_relation": "one_addition_preserving_prior",
            }
        )
    return values


def _recovery_051_expected_operation_evidence() -> list[dict[str, Any]]:
    return [
        {
            "ordinal": 1,
            "operation": "verification_api.discharge_assume_guarantee",
            "lifetime_begin": "cold_process_before_public_discharge",
            "lifetime_return": (
                "public_discharge_returned_before_reference_drop"
            ),
            "producer_range": [0, 10],
            "consumer_range": [0, 20],
            "maximum_obligations": 256,
            "graph_cid": (
                "bafkreieqlh3et3v3m2o67on6mdwoihykzwjppcrzwfjqgnmz223i4t6mg4"
            ),
            "contract_root": (
                "bafkreiekhiynhjkk5ozdsjd63ua737njn3xdih6ljorfs5dqlnjrpuswsy"
            ),
            "receipt_cid": (
                "bafkreicxqwezcvs73id2rk5542zqlpkfonb42frrohrb44ws3d7gev226a"
            ),
            "solver_receipt_id": (
                "bafkreibgtaglg677ibg3srzzm3eol2uzuwdhrruuars5yqw54kjpotg2iy"
            ),
            "disposition": "proved",
            "obligation_count": 1,
            "graph_body_root": (
                "baguqeerazwmcpi4lgac3xv5vu2da26xpeasnlezaldzquh77mkqnseq23zhq"
            ),
            "contract_body_root": (
                "baguqeeratntazskzor6xrp6ucp25hnzusmx6ndxnkhh76bzn2knl2uvu3paq"
            ),
            "clause_body_root": (
                "baguqeerauydhdu72pe3aydg7fu3psipzmc2apciqqo3qeqr6ld4c2tuleacq"
            ),
            "edge_body_root": (
                "baguqeeraws3zab6ofjfz6qkzhwozo4zlmott5jsrbvmlrtshspj3zgqdzz7q"
            ),
            "obligation_body_root": (
                "baguqeera6pa3juanimnzb7tmngkncrnvzummnqqtnb5ivbl7a2eba6k5kygq"
            ),
            "receipt_body_root": (
                "baguqeerasgzywtwchox7ym2mkcy7zwhrbyx5jgabxrqolknjpgagqthskd6q"
            ),
            "replay_root": (
                "baguqeerazhcpwzm7zkokfourpje5ulbwt666xkg45jiynjrz3wkfw5ulv2oq"
            ),
        }
    ]


def _recovery_051_expected_transition_evidence() -> list[dict[str, Any]]:
    events = _recovery_051_expected_check_events()
    values: list[dict[str, Any]] = []
    for event in events:
        main = event["ordinal"] == 1
        values.append(
            {
                "ordinal": event["ordinal"],
                "operation_ordinal": 1,
                "operation": "verification_api.discharge_assume_guarantee",
                "phase": event["phase"],
                "before_task_count": event["before_task_count"],
                "after_task_count": event["after_task_count"],
                "before_normalized_task_root": event[
                    "before_normalized_task_root"
                ],
                "after_normalized_task_root": event[
                    "after_normalized_task_root"
                ],
                "assertion_count": len(event["assertions"]),
                "assertion_root": content_identity(event["assertions"]),
                "assumption_count": len(event["assumptions"]),
                "assumption_root": content_identity(event["assumptions"]),
                "result": event["result"],
                "result_root": content_identity(event["result"]),
                "produced_core_count": len(_RECOVERY_051_CORE) if main else 0,
                "produced_core_root": content_identity(
                    list(_RECOVERY_051_CORE) if main else []
                ),
                "consumed_core_count": 0 if main else len(_RECOVERY_051_CORE),
                "consumed_core_root": content_identity(
                    [] if main else list(_RECOVERY_051_CORE)
                ),
            }
        )
    return values


def _recovery_051_expected_execution_body() -> dict[str, Any]:
    return {
        "schema": "lgcvf-recovery-051-public-prefix-execution@1",
        "diagnostic_reference_cid": (
            _RECOVERY_051_PUBLIC_PREFIX_DIAGNOSTIC_REFERENCE_CID
        ),
        "historical_five_probe_aggregate_cid": (
            _RECOVERY_051_FIVE_PROBE_AGGREGATE_CID
        ),
        "historical_diagnostic_profile_cid": (
            _RECOVERY_051_DIAGNOSTIC_PROFILE_CID
        ),
        "historical_normalized_event_root": (
            _RECOVERY_051_HISTORICAL_NORMALIZED_EVENT_ROOT
        ),
        "diagnostic_reference_disposition": (
            "historical_non_authoritative_reference"
        ),
        "diagnostic_reference_authoritative": False,
        "ordered_operation_evidence": (
            _recovery_051_expected_operation_evidence()
        ),
        "ordered_check_events": _recovery_051_expected_check_events(),
        "ordered_transition_evidence": (
            _recovery_051_expected_transition_evidence()
        ),
        "ordered_task_counts": [1, 2, 3],
        "ordered_normalized_task_roots": [
            _RECOVERY_051_TASK_NORMALIZED_ROOTS[count]
            for count in (1, 2, 3)
        ],
        "task_population_aggregate_root": (
            _RECOVERY_051_TASK_POPULATION_AGGREGATE_ROOT
        ),
        "lifetime_boundaries": list(_RECOVERY_051_LIFETIME_BOUNDARIES),
        "helper_references_dropped": True,
        "facade_reset_disposition": "performed",
        "gc_collection_count": 1,
        "stability_sample_scope": "after_public_operation_reset_gc",
        "stability_checkpoint_count": 1,
        "stability_samples_per_checkpoint": 20,
        "stability_interval_ms": 50,
        "task_identity_continuity_enforced": True,
        "prior_task_identity_preserved_on_growth": True,
        "instrumentation_restored": True,
        "infrastructure_not_proof": True,
        "cache_authority": False,
        "completion_authoritative": False,
    }


def _recovery_051_task_fixed_point_policy() -> dict[str, Any]:
    events = _recovery_051_expected_check_events()
    diagnostic_events = _recovery_051_expected_diagnostic_check_events()
    operations = _recovery_051_expected_operation_evidence()
    transitions = _recovery_051_expected_transition_evidence()
    if (
        content_identity(diagnostic_events)
        != _RECOVERY_051_HISTORICAL_NORMALIZED_EVENT_ROOT
        or content_identity(operations)
        != _RECOVERY_051_PUBLIC_PREFIX_OPERATION_ROOT
        or content_identity(transitions)
        != _RECOVERY_051_PUBLIC_PREFIX_TRANSITION_ROOT
    ):
        raise QualificationError("LGCVF-051 fixed-point policy roots differ")
    z3_import_policy = _recovery_z3_import_denial_policy(
        task_id="LGCVF-051", suite_id="recovery_lgcvf_051"
    )
    value: dict[str, Any] = {
        "schema": "lgcvf-recovery-suite-task-fixed-point@2",
        "task_id": "LGCVF-051",
        "suite_id": "recovery_lgcvf_051",
        "disposition": "public_prefix_fixed_point_required",
        "semantic_diagnostic_reference_cid": (
            _RECOVERY_051_PUBLIC_PREFIX_DIAGNOSTIC_REFERENCE_CID
        ),
        "historical_five_probe_aggregate_cid": (
            _RECOVERY_051_FIVE_PROBE_AGGREGATE_CID
        ),
        "historical_diagnostic_profile_cid": (
            _RECOVERY_051_DIAGNOSTIC_PROFILE_CID
        ),
        "historical_normalized_event_root": (
            _RECOVERY_051_HISTORICAL_NORMALIZED_EVENT_ROOT
        ),
        "diagnostic_reference_disposition": (
            "historical_non_authoritative_reference"
        ),
        "diagnostic_reference_authoritative": False,
        "semantic_execution_cid": content_identity(
            _recovery_051_expected_execution_body()
        ),
        "semantic_check_event_count": 2,
        "semantic_check_event_root": content_identity(events),
        "semantic_operation_evidence_count": 1,
        "semantic_operation_evidence_root": (
            _RECOVERY_051_PUBLIC_PREFIX_OPERATION_ROOT
        ),
        "semantic_transition_evidence_count": 2,
        "semantic_transition_evidence_root": (
            _RECOVERY_051_PUBLIC_PREFIX_TRANSITION_ROOT
        ),
        "timeout_ms": 5_000,
        "logic": "QF_LIA",
        "deterministic_seed": 0,
        "memory_limit_mib": 512,
        "ordered_expected_task_counts": [1, 2, 3],
        "ordered_normalized_task_roots": [
            _RECOVERY_051_TASK_NORMALIZED_ROOTS[count]
            for count in (1, 2, 3)
        ],
        "ordered_operations": operations,
        "lifetime_boundaries": list(_RECOVERY_051_LIFETIME_BOUNDARIES),
        "facade_reset_disposition": "performed",
        "gc_collection_count": 1,
        "stability_sample_scope": "after_public_operation_reset_gc",
        "stability_checkpoint_count": 1,
        "stability_samples_per_checkpoint": 20,
        "stability_interval_ms": 50,
        "z3_import_policy_disposition": z3_import_policy["disposition"],
        "z3_import_policy_cid": z3_import_policy["policy_cid"],
        "exact_task_identity_required": True,
        "parent_task_directory_fd_lease_required": True,
        "recomputed_per_fresh_worker": True,
        "infrastructure_not_proof": True,
        "cache_authority": False,
        "completion_authoritative": False,
    }
    value["profile_cid"] = content_identity(value)
    return value


def _recovery_060_task_fixed_point_policy() -> dict[str, Any]:
    """Return the exact public semantic prefix that fixes Z3 task identity."""

    z3_import_policy = _recovery_z3_import_denial_policy(
        task_id="LGCVF-060", suite_id="recovery_lgcvf_060"
    )
    value: dict[str, Any] = {
        "schema": "lgcvf-recovery-suite-task-fixed-point@2",
        "task_id": "LGCVF-060",
        "suite_id": "recovery_lgcvf_060",
        "disposition": "public_prefix_fixed_point_required",
        "semantic_diagnostic_reference_cid": (
            _RECOVERY_060_PUBLIC_PREFIX_DIAGNOSTIC_REFERENCE_CID
        ),
        "diagnostic_reference_disposition": (
            "historical_non_authoritative_reference"
        ),
        "diagnostic_reference_authoritative": False,
        "semantic_execution_cid": _RECOVERY_060_PUBLIC_PREFIX_EXECUTION_CID,
        "semantic_check_event_count": 12,
        "semantic_check_event_root": _RECOVERY_060_PUBLIC_PREFIX_EVENT_ROOT,
        "semantic_operation_evidence_count": 2,
        "semantic_operation_evidence_root": (
            _RECOVERY_060_PUBLIC_PREFIX_OPERATION_ROOT
        ),
        "semantic_transition_evidence_count": 2,
        "semantic_transition_evidence_root": (
            _RECOVERY_060_PUBLIC_PREFIX_TRANSITION_ROOT
        ),
        "timeout_ms": 5_000,
        "ordered_expected_task_counts": [1, 2, 3],
        "ordered_normalized_task_roots": [
            _RECOVERY_TASK_NORMALIZED_ROOTS[count] for count in (1, 2, 3)
        ],
        "ordered_operations": [
            {
                "ordinal": 1,
                "operation": (
                    "compute_and_validate_interpolant_then_explicit_readmit"
                ),
                "partition_a_root": (
                    "baguqeerawuy72n6holxfjry74cykxhstwkilik7p5g2x7jbef2voqlrmukga"
                ),
                "partition_b_root": (
                    "baguqeeral4i6wqipjbtegxowvzgpq5e3bbuqd4ndd3bukextrejpto4vtkha"
                ),
                "computed_receipt_cid": (
                    "bafkreiduryhkbdzgn3alyhs7ij7elkzfqcfbrr6ku3jo4paqgpxtjq6b3y"
                ),
                "readmitted_receipt_cid": (
                    "bafkreiduryhkbdzgn3alyhs7ij7elkzfqcfbrr6ku3jo4paqgpxtjq6b3y"
                ),
                "transition": {
                    "before_count": 1,
                    "after_count": 2,
                    "assertion_count": 2,
                    "assertion_root": (
                        "baguqeera6653gfja3utzy7aamuw6crebjtkvzfthgttgxpfehxcxx7snsqaq"
                    ),
                    "assumption_count": 2,
                    "assumption_root": (
                        "baguqeerau3xlkezpksybvcgmybit7wjl76xhrttlm5yqvqnpybmnwdcb42ia"
                    ),
                    "result": "unsat",
                    "core_count": 2,
                    "core_root": (
                        "baguqeeram7unu3vzskxjpw3e6ndgs5qqppvlyuo6iof5c3isiagycs67237a"
                    ),
                },
                "lifetime_boundary": (
                    "helper_return_drop_all_references_gc_collect"
                ),
            },
            {
                "ordinal": 2,
                "operation": "first_admit_interpolant_x_le_15",
                "partition_a_root": (
                    "baguqeerawuy72n6holxfjry74cykxhstwkilik7p5g2x7jbef2voqlrmukga"
                ),
                "partition_b_root": (
                    "baguqeeral4i6wqipjbtegxowvzgpq5e3bbuqd4ndd3bukextrejpto4vtkha"
                ),
                "interpolant_root": (
                    "baguqeerasap4t6e3a77dz2b7ubm5y4dmykldnrhp25wqsls4mboidgrj7cha"
                ),
                "receipt_cid": (
                    "bafkreicqdlybhby4dcokk3g5a5st2qepnksdyf4inyokv6bv7dawh2sxf4"
                ),
                "transition": {
                    "before_count": 2,
                    "after_count": 3,
                    "assertion_count": 2,
                    "assertion_root": (
                        "baguqeeramzl5nt4o2cc673y4cx5ldkz7u3eqb2j6igwcrw7ixrqnuswttcya"
                    ),
                    "assumption_count": 0,
                    "assumption_root": (
                        "baguqeeraj5j43immfovaya2uxnpzupwl4xwrfk2nryi3vbz4f4irmeqcxfcq"
                    ),
                    "result": "unsat",
                    "core_count": 0,
                    "core_root": (
                        "baguqeeraj5j43immfovaya2uxnpzupwl4xwrfk2nryi3vbz4f4irmeqcxfcq"
                    ),
                },
                "lifetime_boundary": (
                    "helper_return_drop_all_references_gc_collect"
                ),
            },
        ],
        "lifetime_boundaries": [
            "helper_a_return_drop_all_references_gc_collect",
            "helper_b_return_drop_all_references_gc_collect",
        ],
        "facade_reset_disposition": "not_applicable",
        "gc_collection_count": 2,
        "stability_sample_scope": "after_each_public_operation_transition",
        "stability_checkpoint_count": 2,
        "stability_samples_per_checkpoint": 20,
        "stability_interval_ms": 50,
        "z3_import_policy_disposition": z3_import_policy["disposition"],
        "z3_import_policy_cid": z3_import_policy["policy_cid"],
        "exact_task_identity_required": True,
        "parent_task_directory_fd_lease_required": True,
        "recomputed_per_fresh_worker": True,
        "infrastructure_not_proof": True,
        "cache_authority": False,
        "completion_authoritative": False,
    }
    value["profile_cid"] = content_identity(value)
    return value


def _recovery_suite_task_policy(*, task_id: str, suite_id: str) -> dict[str, Any]:
    expected_suite = "recovery_" + task_id.casefold().replace("-", "_")
    if suite_id != expected_suite:
        raise QualificationError("recovery suite task-policy identity differs")
    if task_id == "LGCVF-060":
        return _recovery_060_task_fixed_point_policy()
    if task_id == "LGCVF-051":
        return _recovery_051_task_fixed_point_policy()
    elif task_id in {"LGCVF-061", "LGCVF-070", "LGCVF-071", "LGCVF-080"}:
        z3_import_policy = _recovery_z3_import_denial_policy(
            task_id=task_id, suite_id=suite_id
        )
        value = {
            "schema": "lgcvf-recovery-suite-task-fixed-point@2",
            "task_id": task_id,
            "suite_id": suite_id,
            "disposition": (
                "single_task_z3_import_denied_as_typed_unavailable"
                if task_id == "LGCVF-061"
                else "single_task_no_z3"
            ),
            "expected_task_count": 1,
            "expected_normalized_task_root": _RECOVERY_TASK_NORMALIZED_ROOTS[1],
            "semantic_execution_cid": "",
            "semantic_check_event_count": 0,
            "semantic_check_event_root": content_identity([]),
            "semantic_operation_evidence_count": 0,
            "semantic_operation_evidence_root": content_identity([]),
            "semantic_transition_evidence_count": 0,
            "semantic_transition_evidence_root": content_identity([]),
            "lifetime_boundaries": [],
            "facade_reset_disposition": "not_applicable",
            "gc_collection_count": 0,
            "stability_sample_scope": "not_applicable",
            "stability_checkpoint_count": 0,
            "stability_samples_per_checkpoint": 0,
            "stability_interval_ms": 0,
            "z3_import_policy_disposition": z3_import_policy["disposition"],
            "z3_import_policy_cid": z3_import_policy["policy_cid"],
            "exact_task_identity_required": True,
            "parent_task_directory_fd_lease_required": True,
            "recomputed_per_fresh_worker": True,
            "infrastructure_not_proof": True,
            "cache_authority": False,
            "completion_authoritative": False,
        }
    else:
        raise QualificationError("recovery suite task policy is unknown")
    value["profile_cid"] = content_identity(value)
    return value


def _recovery_suite_task_policy_matrix() -> dict[str, Any]:
    profiles = [
        _recovery_suite_task_policy(task_id=item.task_id, suite_id=item.suite.suite_id)
        for item in RECOVERY_VALIDATIONS
    ]
    value: dict[str, Any] = {
        "schema": "lgcvf-recovery-suite-task-policy-matrix@2",
        "ordered_profiles": profiles,
        "profile_count": len(profiles),
        "fresh_worker_recomputation_required": True,
        "process_local_task_identity_not_cacheable": True,
        "completion_authoritative": False,
    }
    value["matrix_cid"] = content_identity(value)
    return value


def _stable_recovery_kernel_task_records(
    *,
    task_id: str,
    expected_count: int,
    sample_count: int,
    interval_ms: int,
) -> tuple[list[dict[str, Any]], str]:
    if (
        isinstance(expected_count, bool)
        or expected_count <= 0
        or isinstance(sample_count, bool)
        or sample_count < 0
        or isinstance(interval_ms, bool)
        or interval_ms < 0
    ):
        raise QualificationError("recovery task stability policy differs")
    expected = _recovery_kernel_task_records()
    if len(expected) != expected_count:
        raise QualificationError("recovery task fixed-point count differs")
    for _ in range(sample_count):
        try:
            select.select([], [], [], interval_ms / 1_000)
        except OSError as exc:
            raise QualificationError("recovery task stability wait failed") from exc
        if _recovery_kernel_task_records() != expected:
            raise QualificationError("recovery task fixed-point identity differs")
    normalized = _normalized_kernel_task_role_evidence(
        expected,
        task_id=task_id,
    )
    normalized_root = content_identity(normalized)
    expected_root = _recovery_task_normalized_roots(task_id).get(expected_count)
    if normalized_root != expected_root:
        raise QualificationError("recovery task normalized identity differs")
    return expected, normalized_root


def _recovery_051_public_prefix_fixed_point(
    z3_module: Any,
    *,
    runtime_cid: str,
    source_projection_root: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the exact public compositional discharge that fixes Z3 tasks."""

    from ipfs_datasets_py.logic.software_contracts.compositional import (
        CompositionalContract,
        SemanticContractClause,
    )
    from ipfs_datasets_py.logic.software_contracts.contracts import (
        BoundedPredicate,
        ContractAuthority,
        ContractProvenance,
    )
    from ipfs_datasets_py.logic.software_contracts.semantic_index.models import (
        RepositoryState,
    )
    from ipfs_datasets_py.logic.software_verification.assume_guarantee import (
        ComponentCompositionGraph,
        CompositionEdge,
        DischargeDisposition,
    )
    from ipfs_datasets_py.logic.verification_api import (
        discharge_assume_guarantee,
        get_verification_api,
    )

    policy = _recovery_051_task_fixed_point_policy()
    sample_count = int(policy["stability_samples_per_checkpoint"])
    interval_ms = int(policy["stability_interval_ms"])

    def provenance() -> Any:
        return ContractProvenance(
            fact_kind="inferred",
            authority=ContractAuthority(
                authority_id="authority:public-api-test",
                rank="inference",
                owner="ipfs_datasets_py.logic",
                revision="public-api-test@1",
            ),
            source_path="fixture.py",
            source_symbol="fixture.value",
        )

    def roots() -> dict[str, str]:
        return {
            name: f"sha256:{index:064x}"
            for index, name in enumerate(
                (
                    "source_root",
                    "ast_root",
                    "symbol_version_root",
                    "interface_root",
                    "configuration_root",
                    "toolchain_root",
                ),
                1,
            )
        }

    def clause(clause_id: str, kind: str, lower: int, upper: int) -> Any:
        return SemanticContractClause(
            clause_id=clause_id,
            kind=kind,
            support="typed_inline",
            predicate=BoundedPredicate(
                predicate_id=f"{clause_id}:predicate",
                role="assumption" if kind == "assumption" else "postcondition",
                operator="range_int",
                subject="return",
                provenance=provenance(),
                arguments=(lower, upper),
            ),
        )

    def contract(
        component: str,
        *,
        assumptions: tuple[Any, ...] = (),
        guarantees: tuple[Any, ...] = (),
    ) -> Any:
        return CompositionalContract(
            contract_id=f"contract:{component}",
            component_id=component,
            component_kind="callable",
            provenance=provenance(),
            assumptions=assumptions,
            guarantees=guarantees,
            confidence="conservative",
            semantic_support_class="supported_subset",
            **roots(),
        )

    private_events: list[dict[str, Any]] = []
    solver_timeouts: dict[int, int] = {}
    active_operation: dict[str, Any] = {
        "ordinal": 0,
        "operation": "not_started",
    }
    original_set = z3_module.Solver.set
    original_check = z3_module.Solver.check

    def traced_set(solver: Any, *args: Any, **kwargs: Any) -> Any:
        timeout = kwargs.get("timeout")
        if isinstance(timeout, int) and not isinstance(timeout, bool):
            solver_timeouts[id(solver)] = timeout
        return original_set(solver, *args, **kwargs)

    def traced_check(solver: Any, *assumptions: Any) -> Any:
        event_ordinal = len(private_events) + 1
        phases = {
            1: "tracked_assumption_unsat",
            2: "independent_core_validation_unsat",
        }
        phase = phases.get(event_ordinal)
        if (
            phase is None
            or active_operation != {
                "ordinal": 1,
                "operation": "verification_api.discharge_assume_guarantee",
            }
        ):
            raise QualificationError(
                "LGCVF-051 fixed-point check sequence differs"
            )
        before_records = _recovery_kernel_task_records()
        assertions = [item.sexpr() for item in solver.assertions()]
        assumption_text = [item.sexpr() for item in assumptions]
        result = original_check(solver, *assumptions)
        after_records = _recovery_kernel_task_records()
        if (
            len(after_records) != len(before_records) + 1
            or not all(record in after_records for record in before_records)
        ):
            raise QualificationError(
                "LGCVF-051 fixed-point task growth differs"
            )
        private_events.append(
            {
                "ordinal": event_ordinal,
                "operation_ordinal": active_operation["ordinal"],
                "operation": active_operation["operation"],
                "phase": phase,
                "timeout_ms": solver_timeouts.get(id(solver)),
                "assertions": assertions,
                "assumptions": assumption_text,
                "result": str(result),
                "before_records": before_records,
                "after_records": after_records,
            }
        )
        return result

    def public_discharge() -> tuple[dict[str, Any], list[str]]:
        producer = contract(
            "A",
            guarantees=(clause("A:guarantee", "guarantee", 0, 10),),
        )
        consumer = contract(
            "B",
            assumptions=(clause("B:assumption", "assumption", 0, 20),),
        )
        state = RepositoryState("repository:public-compositional-api")
        edge = CompositionEdge(
            edge_id="A-to-B",
            producer_component_id="A",
            consumer_component_id="B",
            guarantee_clause_ids=("A:guarantee",),
            assumption_clause_ids=("B:assumption",),
        )
        graph = ComponentCompositionGraph(
            semantic_state_root=state.state_cid,
            contracts=(producer, consumer),
            edges=(edge,),
        )
        active_operation.update(
            {
                "ordinal": 1,
                "operation": "verification_api.discharge_assume_guarantee",
            }
        )
        receipt = discharge_assume_guarantee(
            graph,
            expected_semantic_state_root=state.state_cid,
            expected_contract_root=graph.contract_root,
        )
        if (
            receipt.disposition is not DischargeDisposition.PROVED
            or len(receipt.obligations) != 1
        ):
            raise QualificationError(
                "LGCVF-051 public fixed-point discharge differs"
            )
        obligation = receipt.obligations[0]
        if list(obligation.unsat_core) != list(_RECOVERY_051_CORE):
            raise QualificationError("LGCVF-051 public fixed-point core differs")
        operation = {
            "ordinal": 1,
            "operation": "verification_api.discharge_assume_guarantee",
            "lifetime_begin": "cold_process_before_public_discharge",
            "lifetime_return": (
                "public_discharge_returned_before_reference_drop"
            ),
            "producer_range": [0, 10],
            "consumer_range": [0, 20],
            "maximum_obligations": 256,
            "graph_cid": graph.graph_cid,
            "contract_root": graph.contract_root,
            "receipt_cid": receipt.receipt_cid,
            "solver_receipt_id": obligation.solver_receipt_id,
            "disposition": receipt.disposition.value,
            "obligation_count": len(receipt.obligations),
            "graph_body_root": content_identity(graph.to_dict()),
            "contract_body_root": content_identity(
                [producer.to_dict(), consumer.to_dict()]
            ),
            "clause_body_root": content_identity(
                [
                    producer.guarantees[0].to_dict(),
                    consumer.assumptions[0].to_dict(),
                ]
            ),
            "edge_body_root": content_identity(edge.to_dict()),
            "obligation_body_root": content_identity(obligation.to_dict()),
            "receipt_body_root": content_identity(receipt.to_dict()),
            "replay_root": content_identity(receipt.replay_data.to_dict()),
        }
        return operation, list(obligation.unsat_core)

    baseline_records, baseline_root = _stable_recovery_kernel_task_records(
        task_id="LGCVF-051",
        expected_count=1,
        sample_count=0,
        interval_ms=0,
    )
    instrumentation_restored = False
    try:
        z3_module.Solver.set = traced_set
        z3_module.Solver.check = traced_check
        operation_evidence, live_core = public_discharge()
    finally:
        z3_module.Solver.set = original_set
        z3_module.Solver.check = original_check
        instrumentation_restored = (
            z3_module.Solver.set is original_set
            and z3_module.Solver.check is original_check
        )
    if not instrumentation_restored:
        raise QualificationError("LGCVF-051 fixed-point instrumentation remained")
    replacement = get_verification_api(reset=True)
    del replacement
    gc.collect()
    terminal_records, terminal_root = _stable_recovery_kernel_task_records(
        task_id="LGCVF-051",
        expected_count=3,
        sample_count=sample_count,
        interval_ms=interval_ms,
    )
    if len(private_events) != 2:
        raise QualificationError("LGCVF-051 fixed-point check count differs")
    if private_events[0]["before_records"] != baseline_records:
        raise QualificationError("LGCVF-051 fixed-point baseline differs")
    if (
        private_events[1]["before_records"]
        != private_events[0]["after_records"]
        or terminal_records != private_events[1]["after_records"]
    ):
        raise QualificationError(
            "LGCVF-051 fixed-point task identity continuity differs"
        )
    middle_records = private_events[0]["after_records"]
    middle_root = content_identity(
        _normalized_kernel_task_role_evidence(
            middle_records,
            task_id="LGCVF-051",
        )
    )
    normalized_events: list[dict[str, Any]] = []
    for private in private_events:
        main = private["ordinal"] == 1
        event = {
            "ordinal": private["ordinal"],
            "operation_ordinal": private["operation_ordinal"],
            "operation": private["operation"],
            "phase": private["phase"],
            "timeout_ms": private["timeout_ms"],
            "assertions": list(private["assertions"]),
            "assertion_count": len(private["assertions"]),
            "assertion_root": content_identity(private["assertions"]),
            "assumptions": list(private["assumptions"]),
            "assumption_count": len(private["assumptions"]),
            "assumption_root": content_identity(private["assumptions"]),
            "result": private["result"],
            "result_root": content_identity(private["result"]),
            "produced_core_count": len(live_core) if main else 0,
            "produced_core_root": content_identity(live_core if main else []),
            "consumed_core_count": 0 if main else len(live_core),
            "consumed_core_root": content_identity([] if main else live_core),
            "before_task_count": len(private["before_records"]),
            "after_task_count": len(private["after_records"]),
            "before_normalized_task_root": content_identity(
                _normalized_kernel_task_role_evidence(
                    private["before_records"],
                    task_id="LGCVF-051",
                )
            ),
            "after_normalized_task_root": content_identity(
                _normalized_kernel_task_role_evidence(
                    private["after_records"],
                    task_id="LGCVF-051",
                )
            ),
            "task_identity_relation": "one_addition_preserving_prior",
        }
        normalized_events.append(event)
    expected_events = _recovery_051_expected_check_events()
    if (
        operation_evidence != policy["ordered_operations"][0]
        or normalized_events != expected_events
        or [baseline_root, middle_root, terminal_root]
        != policy["ordered_normalized_task_roots"]
    ):
        raise QualificationError("LGCVF-051 public fixed-point evidence differs")
    transition_evidence = [
        {
            key: event[key]
            for key in (
                "ordinal",
                "operation_ordinal",
                "operation",
                "phase",
                "before_task_count",
                "after_task_count",
                "before_normalized_task_root",
                "after_normalized_task_root",
                "assertion_count",
                "assertion_root",
                "assumption_count",
                "assumption_root",
                "result",
                "result_root",
                "produced_core_count",
                "produced_core_root",
                "consumed_core_count",
                "consumed_core_root",
            )
        }
        for event in normalized_events
    ]
    normalized_task_populations = [
        _normalized_kernel_task_role_evidence(records, task_id="LGCVF-051")
        for records in (baseline_records, middle_records, terminal_records)
    ]
    semantic_body: dict[str, Any] = {
        "schema": "lgcvf-recovery-051-public-prefix-execution@1",
        "diagnostic_reference_cid": (
            _RECOVERY_051_PUBLIC_PREFIX_DIAGNOSTIC_REFERENCE_CID
        ),
        "historical_five_probe_aggregate_cid": (
            _RECOVERY_051_FIVE_PROBE_AGGREGATE_CID
        ),
        "historical_diagnostic_profile_cid": (
            _RECOVERY_051_DIAGNOSTIC_PROFILE_CID
        ),
        "historical_normalized_event_root": (
            _RECOVERY_051_HISTORICAL_NORMALIZED_EVENT_ROOT
        ),
        "diagnostic_reference_disposition": (
            "historical_non_authoritative_reference"
        ),
        "diagnostic_reference_authoritative": False,
        "ordered_operation_evidence": [operation_evidence],
        "ordered_check_events": normalized_events,
        "ordered_transition_evidence": transition_evidence,
        "ordered_task_counts": [
            len(baseline_records),
            len(middle_records),
            len(terminal_records),
        ],
        "ordered_normalized_task_roots": [
            baseline_root,
            middle_root,
            terminal_root,
        ],
        "task_population_aggregate_root": content_identity(
            normalized_task_populations
        ),
        "lifetime_boundaries": list(_RECOVERY_051_LIFETIME_BOUNDARIES),
        "helper_references_dropped": True,
        "facade_reset_disposition": "performed",
        "gc_collection_count": 1,
        "stability_sample_scope": "after_public_operation_reset_gc",
        "stability_checkpoint_count": 1,
        "stability_samples_per_checkpoint": sample_count,
        "stability_interval_ms": interval_ms,
        "task_identity_continuity_enforced": True,
        "prior_task_identity_preserved_on_growth": True,
        "instrumentation_restored": instrumentation_restored,
        "infrastructure_not_proof": True,
        "cache_authority": False,
        "completion_authoritative": False,
    }
    expected_semantic_body = _recovery_051_expected_execution_body()
    if (
        transition_evidence != _recovery_051_expected_transition_evidence()
        or semantic_body != expected_semantic_body
    ):
        raise QualificationError("LGCVF-051 semantic execution differs")
    semantic_execution_cid = content_identity(semantic_body)
    if semantic_execution_cid != policy["semantic_execution_cid"]:
        raise QualificationError(
            "LGCVF-051 public fixed-point semantic identity differs: "
            + semantic_execution_cid
        )
    value: dict[str, Any] = {
        "schema": "lgcvf-recovery-suite-task-observation@4",
        "task_id": "LGCVF-051",
        "suite_id": "recovery_lgcvf_051",
        "runtime_cid": runtime_cid,
        "source_projection_root": source_projection_root,
        "policy_matrix_cid": _recovery_suite_task_policy_matrix()[
            "matrix_cid"
        ],
        "profile": policy,
        "profile_cid": policy["profile_cid"],
        **_recovery_z3_import_policy_commitments(
            task_id="LGCVF-051", suite_id="recovery_lgcvf_051"
        ),
        "semantic_diagnostic_reference_cid": (
            _RECOVERY_051_PUBLIC_PREFIX_DIAGNOSTIC_REFERENCE_CID
        ),
        "diagnostic_reference_disposition": (
            "historical_non_authoritative_reference"
        ),
        "diagnostic_reference_authoritative": False,
        "semantic_execution": semantic_body,
        "semantic_execution_cid": semantic_execution_cid,
        "ordered_operation_evidence": [operation_evidence],
        "ordered_transition_evidence": transition_evidence,
        "z3_check_event_count": len(normalized_events),
        "ordered_task_counts": [1, 2, 3],
        "ordered_normalized_task_roots": [
            baseline_root,
            middle_root,
            terminal_root,
        ],
        "lifetime_boundary_count": len(policy["lifetime_boundaries"]),
        "lifetime_boundary_root": content_identity(
            policy["lifetime_boundaries"]
        ),
        "helper_references_dropped": True,
        "facade_reset_disposition": "performed",
        "gc_collection_count": 1,
        "stability_sample_scope": "after_public_operation_reset_gc",
        "stability_checkpoint_count": 1,
        "stability_samples_per_checkpoint": sample_count,
        "stability_interval_ms": interval_ms,
        "stability_verified": True,
        "instrumentation_restored": True,
        "fresh_worker_recomputed": True,
        "infrastructure_not_proof": True,
        "cache_authority": False,
        "completion_authoritative": False,
    }
    value["observation_cid"] = content_identity(value)
    return value, [dict(item) for item in terminal_records]


def _recovery_060_public_prefix_fixed_point(
    z3_module: Any,
    *,
    runtime_cid: str,
    source_projection_root: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Create the exact public API task fixed point, retaining no proof objects."""

    from ipfs_datasets_py.logic.backends.smt.compiler import (
        SmtTerm,
        SmtTermKind,
        term_and,
        term_int,
        term_symbol,
    )
    from ipfs_datasets_py.logic.backends.smt.interpolation import (
        InterpolationStatus,
        admit_interpolant,
        compute_and_validate_interpolant,
    )

    policy = _recovery_060_task_fixed_point_policy()
    sample_count = int(policy["stability_samples_per_checkpoint"])
    interval_ms = int(policy["stability_interval_ms"])

    def range_term(symbol: str, lower: int, upper: int) -> Any:
        value = term_symbol(symbol)
        return term_and(
            SmtTerm(
                SmtTermKind.GE,
                arguments=(value, term_int(lower)),
            ),
            SmtTerm(
                SmtTermKind.LE,
                arguments=(value, term_int(upper)),
            ),
        )

    def le_term(symbol: str, upper: int) -> Any:
        return SmtTerm(
            SmtTermKind.LE,
            arguments=(term_symbol(symbol), term_int(upper)),
        )

    events: list[dict[str, Any]] = []
    private_task_transitions: list[
        tuple[list[dict[str, Any]], list[dict[str, Any]]]
    ] = []
    timeouts: dict[int, int] = {}
    active_operation = {"ordinal": 0}
    original_set = z3_module.Solver.set
    original_check = z3_module.Solver.check
    original_unsat_core = z3_module.Solver.unsat_core

    def traced_set(solver: Any, *args: Any, **kwargs: Any) -> Any:
        timeout = kwargs.get("timeout")
        if isinstance(timeout, int) and not isinstance(timeout, bool):
            timeouts[id(solver)] = timeout
        return original_set(solver, *args, **kwargs)

    def traced_check(solver: Any, *assumptions: Any) -> Any:
        before_records = _recovery_kernel_task_records()
        assertions = [str(item) for item in solver.assertions()]
        assumption_text = [str(item) for item in assumptions]
        result = original_check(solver, *assumptions)
        core = (
            [str(item) for item in original_unsat_core(solver)]
            if str(result) == "unsat"
            else []
        )
        after_records = _recovery_kernel_task_records()
        before_normalized_root = content_identity(
            _normalized_kernel_task_role_evidence(
                before_records,
                task_id="LGCVF-060",
            )
        )
        after_normalized_root = content_identity(
            _normalized_kernel_task_role_evidence(
                after_records,
                task_id="LGCVF-060",
            )
        )
        if after_records == before_records:
            task_identity_relation = "unchanged"
        elif (
            len(after_records) == len(before_records) + 1
            and all(record in after_records for record in before_records)
        ):
            task_identity_relation = "one_addition_preserving_prior"
        else:
            raise QualificationError(
                "LGCVF-060 fixed-point task identity changed unexpectedly"
            )
        operation_event_ordinal = 1 + sum(
            1
            for item in events
            if item["operation_ordinal"] == active_operation["ordinal"]
        )
        events.append(
            {
                "operation_ordinal": active_operation["ordinal"],
                "operation_event_ordinal": operation_event_ordinal,
                "timeout_ms": timeouts.get(id(solver)),
                "assertion_count": len(assertions),
                "assertion_root": content_identity(assertions),
                "assumption_count": len(assumption_text),
                "assumption_root": content_identity(assumption_text),
                "result": str(result),
                "core_count": len(core),
                "core_root": content_identity(core),
                "before_count": len(before_records),
                "after_count": len(after_records),
                "before_normalized_task_root": before_normalized_root,
                "after_normalized_task_root": after_normalized_root,
                "task_identity_relation": task_identity_relation,
            }
        )
        private_task_transitions.append((before_records, after_records))
        return result

    def helper_a() -> dict[str, Any]:
        active_operation["ordinal"] = 1
        partition_a = range_term("x", 0, 10)
        partition_b = range_term("x", 20, 30)
        receipt = compute_and_validate_interpolant(partition_a, partition_b)
        if (
            receipt.status is not InterpolationStatus.VALIDATED
            or receipt.interpolant is None
        ):
            raise QualificationError("LGCVF-060 fixed-point compute was not validated")
        admitted = admit_interpolant(
            partition_a,
            partition_b,
            receipt.interpolant,
            provider=receipt.provider,
            provider_version=receipt.provider_version,
            interpolation_api=receipt.interpolation_api,
            independent_validator_version=receipt.independent_validator_version,
        )
        if (
            admitted.status is not InterpolationStatus.VALIDATED
            or admitted.receipt_cid != receipt.receipt_cid
        ):
            raise QualificationError("LGCVF-060 fixed-point re-admission differs")
        return {
            "ordinal": 1,
            "operation": "compute_and_validate_interpolant_then_explicit_readmit",
            "partition_a_root": content_identity(partition_a.to_dict()),
            "partition_b_root": content_identity(partition_b.to_dict()),
            "computed_receipt_cid": receipt.receipt_cid,
            "readmitted_receipt_cid": admitted.receipt_cid,
            "status": admitted.status.value,
            "objects_retained_after_helper_return": False,
        }

    def helper_b() -> dict[str, Any]:
        active_operation["ordinal"] = 2
        partition_a = range_term("x", 0, 10)
        partition_b = range_term("x", 20, 30)
        interpolant = le_term("x", 15)
        receipt = admit_interpolant(partition_a, partition_b, interpolant)
        if receipt.status is not InterpolationStatus.VALIDATED:
            raise QualificationError("LGCVF-060 fixed-point admission was not validated")
        return {
            "ordinal": 2,
            "operation": "first_admit_interpolant_x_le_15",
            "partition_a_root": content_identity(partition_a.to_dict()),
            "partition_b_root": content_identity(partition_b.to_dict()),
            "interpolant_root": content_identity(interpolant.to_dict()),
            "receipt_cid": receipt.receipt_cid,
            "status": receipt.status.value,
            "objects_retained_after_helper_return": False,
        }

    baseline_records, baseline_root = _stable_recovery_kernel_task_records(
        task_id="LGCVF-060",
        expected_count=1,
        sample_count=0,
        interval_ms=0,
    )
    instrumentation_restored = False
    try:
        z3_module.Solver.set = traced_set
        z3_module.Solver.check = traced_check
        helper_a_evidence = helper_a()
        gc.collect()
        after_a_records, after_a_root = _stable_recovery_kernel_task_records(
            task_id="LGCVF-060",
            expected_count=2,
            sample_count=sample_count,
            interval_ms=interval_ms,
        )
        helper_b_evidence = helper_b()
        gc.collect()
        after_b_records, after_b_root = _stable_recovery_kernel_task_records(
            task_id="LGCVF-060",
            expected_count=3,
            sample_count=sample_count,
            interval_ms=interval_ms,
        )
    finally:
        z3_module.Solver.set = original_set
        z3_module.Solver.check = original_check
        instrumentation_restored = (
            z3_module.Solver.set is original_set
            and z3_module.Solver.check is original_check
            and z3_module.Solver.unsat_core is original_unsat_core
        )
    if not instrumentation_restored:
        raise QualificationError("LGCVF-060 fixed-point instrumentation remained")
    transitions = [
        event for event in events if event["before_count"] != event["after_count"]
    ]
    expected_operations = policy["ordered_operations"]
    expected_transitions = [
        {
            "operation_ordinal": item["ordinal"],
            "timeout_ms": policy["timeout_ms"],
            **dict(item["transition"]),
            "before_normalized_task_root": (
                _RECOVERY_TASK_NORMALIZED_ROOTS[
                    int(item["transition"]["before_count"])
                ]
            ),
            "after_normalized_task_root": (
                _RECOVERY_TASK_NORMALIZED_ROOTS[
                    int(item["transition"]["after_count"])
                ]
            ),
            "task_identity_relation": "one_addition_preserving_prior",
        }
        for item in expected_operations
    ]
    transition_cores = [
        {
            key: item
            for key, item in event.items()
            if key != "operation_event_ordinal"
        }
        for event in transitions
    ]
    current_records = baseline_records
    for before_records, after_records in private_task_transitions:
        if before_records != current_records:
            raise QualificationError(
                "LGCVF-060 fixed-point task identity continuity differs"
            )
        if after_records != before_records:
            if (
                len(after_records) != len(before_records) + 1
                or not all(record in after_records for record in before_records)
            ):
                raise QualificationError(
                    "LGCVF-060 fixed-point task growth differs"
                )
        current_records = after_records
    if current_records != after_b_records:
        raise QualificationError(
            "LGCVF-060 fixed-point terminal task identity differs"
        )
    operation_event_counts = [
        sum(1 for event in events if event["operation_ordinal"] == ordinal)
        for ordinal in (1, 2)
    ]
    if (
        len(events) != 12
        or operation_event_counts != [8, 4]
        or transition_cores != expected_transitions
        or helper_a_evidence
        != {
            "ordinal": 1,
            "operation": expected_operations[0]["operation"],
            "partition_a_root": expected_operations[0]["partition_a_root"],
            "partition_b_root": expected_operations[0]["partition_b_root"],
            "computed_receipt_cid": expected_operations[0][
                "computed_receipt_cid"
            ],
            "readmitted_receipt_cid": expected_operations[0][
                "readmitted_receipt_cid"
            ],
            "status": "validated",
            "objects_retained_after_helper_return": False,
        }
        or helper_b_evidence
        != {
            "ordinal": 2,
            "operation": expected_operations[1]["operation"],
            "partition_a_root": expected_operations[1]["partition_a_root"],
            "partition_b_root": expected_operations[1]["partition_b_root"],
            "interpolant_root": expected_operations[1]["interpolant_root"],
            "receipt_cid": expected_operations[1]["receipt_cid"],
            "status": "validated",
            "objects_retained_after_helper_return": False,
        }
        or [len(baseline_records), len(after_a_records), len(after_b_records)]
        != policy["ordered_expected_task_counts"]
        or [baseline_root, after_a_root, after_b_root]
        != policy["ordered_normalized_task_roots"]
        or private_task_transitions[0][0] != baseline_records
        or private_task_transitions[7][1] != after_a_records
        or private_task_transitions[8][0] != after_a_records
        or private_task_transitions[-1][1] != after_b_records
    ):
        raise QualificationError("LGCVF-060 public fixed-point evidence differs")
    semantic_body: dict[str, Any] = {
        "schema": "lgcvf-recovery-060-public-prefix-execution@1",
        "diagnostic_reference_cid": (
            _RECOVERY_060_PUBLIC_PREFIX_DIAGNOSTIC_REFERENCE_CID
        ),
        "diagnostic_reference_disposition": (
            "historical_non_authoritative_reference"
        ),
        "diagnostic_reference_authoritative": False,
        "ordered_operation_evidence": [helper_a_evidence, helper_b_evidence],
        "ordered_check_events": events,
        "ordered_transition_evidence": transitions,
        "operation_event_counts": operation_event_counts,
        "ordered_task_counts": [1, 2, 3],
        "ordered_normalized_task_roots": [
            baseline_root,
            after_a_root,
            after_b_root,
        ],
        "helper_references_dropped": True,
        # This already-qualified semantic body remains byte-for-byte @1.
        # Scope-aware lifecycle facts are config-pinned in the profile and
        # cross-bound by the @2 observation/receipt without rewriting its CID.
        "gc_collection_count": 2,
        "stability_sample_count_per_transition": sample_count,
        "stability_interval_ms": interval_ms,
        "task_identity_continuity_enforced": True,
        "prior_task_identity_preserved_on_growth": True,
        "instrumentation_restored": True,
        "infrastructure_not_proof": True,
        "cache_authority": False,
        "completion_authoritative": False,
    }
    semantic_execution_cid = content_identity(semantic_body)
    if semantic_execution_cid != policy["semantic_execution_cid"]:
        raise QualificationError(
            "LGCVF-060 public fixed-point semantic identity differs: "
            + semantic_execution_cid
        )
    value: dict[str, Any] = {
        "schema": "lgcvf-recovery-suite-task-observation@4",
        "task_id": "LGCVF-060",
        "suite_id": "recovery_lgcvf_060",
        "runtime_cid": runtime_cid,
        "source_projection_root": source_projection_root,
        "policy_matrix_cid": _recovery_suite_task_policy_matrix()[
            "matrix_cid"
        ],
        "profile": policy,
        "profile_cid": policy["profile_cid"],
        **_recovery_z3_import_policy_commitments(
            task_id="LGCVF-060", suite_id="recovery_lgcvf_060"
        ),
        "semantic_diagnostic_reference_cid": (
            _RECOVERY_060_PUBLIC_PREFIX_DIAGNOSTIC_REFERENCE_CID
        ),
        "diagnostic_reference_disposition": (
            "historical_non_authoritative_reference"
        ),
        "diagnostic_reference_authoritative": False,
        "semantic_execution": semantic_body,
        "semantic_execution_cid": semantic_execution_cid,
        "ordered_operation_evidence": [helper_a_evidence, helper_b_evidence],
        "ordered_transition_evidence": transitions,
        "z3_check_event_count": len(events),
        "ordered_task_counts": [1, 2, 3],
        "ordered_normalized_task_roots": [
            baseline_root,
            after_a_root,
            after_b_root,
        ],
        "lifetime_boundary_count": len(policy["lifetime_boundaries"]),
        "lifetime_boundary_root": content_identity(
            policy["lifetime_boundaries"]
        ),
        "helper_references_dropped": True,
        "facade_reset_disposition": "not_applicable",
        "gc_collection_count": policy["gc_collection_count"],
        "stability_sample_scope": policy["stability_sample_scope"],
        "stability_checkpoint_count": policy["stability_checkpoint_count"],
        "stability_samples_per_checkpoint": sample_count,
        "stability_interval_ms": interval_ms,
        "stability_verified": True,
        "instrumentation_restored": True,
        "fresh_worker_recomputed": True,
        "infrastructure_not_proof": True,
        "cache_authority": False,
        "completion_authoritative": False,
    }
    value["observation_cid"] = content_identity(value)
    return value, [dict(item) for item in after_b_records]


def _recovery_suite_task_fixed_point(
    recovery: RecoveryValidation,
    suite: Suite,
    *,
    z3_module: Any | None,
    runtime_cid: str,
    source_projection_root: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    policy = _recovery_suite_task_policy(
        task_id=recovery.task_id,
        suite_id=suite.suite_id,
    )
    if recovery.task_id in {"LGCVF-051", "LGCVF-060"}:
        if z3_module is None:
            raise QualificationError(
                f"{recovery.task_id} Z3 runtime is absent"
            )
        fixed_point = (
            _recovery_051_public_prefix_fixed_point
            if recovery.task_id == "LGCVF-051"
            else _recovery_060_public_prefix_fixed_point
        )
        observation, terminal_records = fixed_point(
                z3_module,
                runtime_cid=runtime_cid,
                source_projection_root=source_projection_root,
            )
        validated = _validate_recovery_suite_task_observation(
            observation,
            task_id=recovery.task_id,
            suite_id=suite.suite_id,
            runtime_cid=runtime_cid,
            source_projection_root=source_projection_root,
        )
        if _recovery_kernel_task_records() != terminal_records:
            raise QualificationError(
                f"{recovery.task_id} task identity changed after fixed point"
            )
        return validated, terminal_records
    records, normalized_root = _stable_recovery_kernel_task_records(
        task_id=recovery.task_id,
        expected_count=int(policy["expected_task_count"]),
        sample_count=0,
        interval_ms=0,
    )
    value = {
        "schema": "lgcvf-recovery-suite-task-observation@4",
        "task_id": recovery.task_id,
        "suite_id": suite.suite_id,
        "runtime_cid": runtime_cid,
        "source_projection_root": source_projection_root,
        "policy_matrix_cid": _recovery_suite_task_policy_matrix()[
            "matrix_cid"
        ],
        "profile": policy,
        "profile_cid": policy["profile_cid"],
        **_recovery_z3_import_policy_commitments(
            task_id=recovery.task_id, suite_id=suite.suite_id
        ),
        "semantic_diagnostic_reference_cid": "",
        "diagnostic_reference_disposition": "not_applicable",
        "diagnostic_reference_authoritative": False,
        "semantic_execution": None,
        "semantic_execution_cid": "",
        "ordered_operation_evidence": [],
        "ordered_transition_evidence": [],
        "z3_check_event_count": 0,
        "ordered_task_counts": [len(records)],
        "ordered_normalized_task_roots": [normalized_root],
        "lifetime_boundary_count": 0,
        "lifetime_boundary_root": content_identity([]),
        "helper_references_dropped": True,
        "facade_reset_disposition": "not_applicable",
        "gc_collection_count": 0,
        "stability_sample_scope": "not_applicable",
        "stability_checkpoint_count": 0,
        "stability_samples_per_checkpoint": 0,
        "stability_interval_ms": 0,
        "stability_verified": True,
        "instrumentation_restored": True,
        "fresh_worker_recomputed": True,
        "infrastructure_not_proof": True,
        "cache_authority": False,
        "completion_authoritative": False,
    }
    value["observation_cid"] = content_identity(value)
    validated = _validate_recovery_suite_task_observation(
        value,
        task_id=recovery.task_id,
        suite_id=suite.suite_id,
        runtime_cid=runtime_cid,
        source_projection_root=source_projection_root,
    )
    if _recovery_kernel_task_records() != records:
        raise QualificationError("recovery task identity changed after fixed point")
    return validated, [dict(item) for item in records]


_RECOVERY_SUITE_TASK_OBSERVATION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "task_id",
        "suite_id",
        "runtime_cid",
        "source_projection_root",
        "policy_matrix_cid",
        "profile",
        "profile_cid",
        "z3_import_policy_disposition",
        "z3_import_policy_cid",
        "z3_expected_meta_denial_count",
        "z3_expected_meta_denial_root",
        "z3_expected_open_boundary_denial_count",
        "z3_expected_open_boundary_denial_root",
        "z3_trusted_source_revalidation_disposition",
        "z3_expected_trusted_source_event_count",
        "z3_expected_trusted_source_event_root",
        "z3_policy_namespace_unavailability",
        "z3_live_cegar_disposition",
        "z3_candidate_reason_interpretation",
        "semantic_diagnostic_reference_cid",
        "diagnostic_reference_disposition",
        "diagnostic_reference_authoritative",
        "semantic_execution",
        "semantic_execution_cid",
        "ordered_operation_evidence",
        "ordered_transition_evidence",
        "z3_check_event_count",
        "ordered_task_counts",
        "ordered_normalized_task_roots",
        "lifetime_boundary_count",
        "lifetime_boundary_root",
        "helper_references_dropped",
        "facade_reset_disposition",
        "gc_collection_count",
        "stability_sample_scope",
        "stability_checkpoint_count",
        "stability_samples_per_checkpoint",
        "stability_interval_ms",
        "stability_verified",
        "instrumentation_restored",
        "fresh_worker_recomputed",
        "infrastructure_not_proof",
        "cache_authority",
        "completion_authoritative",
        "observation_cid",
    }
)


def _validate_recovery_suite_task_observation(
    observation: Mapping[str, Any],
    *,
    task_id: str,
    suite_id: str,
    runtime_cid: str,
    source_projection_root: str,
) -> dict[str, Any]:
    """Validate the deterministic task policy and its fresh worker evidence."""

    body = {
        key: item for key, item in observation.items() if key != "observation_cid"
    }
    profile = _recovery_suite_task_policy(task_id=task_id, suite_id=suite_id)
    matrix = _recovery_suite_task_policy_matrix()
    z3_commitments = _recovery_z3_import_policy_commitments(
        task_id=task_id,
        suite_id=suite_id,
    )
    numeric_observation_fields = (
        "z3_expected_meta_denial_count",
        "z3_expected_open_boundary_denial_count",
        "z3_expected_trusted_source_event_count",
        "z3_check_event_count",
        "lifetime_boundary_count",
        "gc_collection_count",
        "stability_checkpoint_count",
        "stability_samples_per_checkpoint",
        "stability_interval_ms",
    )
    if (
        set(observation) != _RECOVERY_SUITE_TASK_OBSERVATION_FIELDS
        or observation.get("schema")
        != "lgcvf-recovery-suite-task-observation@4"
        or observation.get("task_id") != task_id
        or observation.get("suite_id") != suite_id
        or observation.get("runtime_cid") != runtime_cid
        or observation.get("source_projection_root") != source_projection_root
        or observation.get("policy_matrix_cid") != matrix["matrix_cid"]
        or observation.get("profile") != profile
        or observation.get("profile_cid") != profile["profile_cid"]
        or any(
            observation.get(key) != item
            for key, item in z3_commitments.items()
        )
        or observation.get("observation_cid") != content_identity(body)
        or observation.get("helper_references_dropped") is not True
        or observation.get("stability_verified") is not True
        or observation.get("instrumentation_restored") is not True
        or observation.get("fresh_worker_recomputed") is not True
        or observation.get("infrastructure_not_proof") is not True
        or observation.get("cache_authority") is not False
        or observation.get("completion_authoritative") is not False
        or any(
            isinstance(observation.get(field), bool)
            or not isinstance(observation.get(field), int)
            or int(observation[field]) < 0
            for field in numeric_observation_fields
        )
        or not isinstance(observation.get("ordered_task_counts"), list)
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in observation.get("ordered_task_counts", [])
        )
        or not _is_canonical_content_cid(
            observation.get("lifetime_boundary_root")
        )
    ):
        raise QualificationError("recovery suite task observation differs")
    if profile.get("disposition") == "public_prefix_fixed_point_required":
        semantic = observation.get("semantic_execution")
        if not isinstance(semantic, Mapping):
            raise QualificationError(
                f"{task_id} semantic execution is absent"
            )
        semantic_body = dict(semantic)
        events = semantic.get("ordered_check_events")
        transitions = semantic.get("ordered_transition_evidence")
        operations = semantic.get("ordered_operation_evidence")
        semantic_cid = content_identity(semantic_body)
        if (
            semantic.get("diagnostic_reference_cid")
            != profile["semantic_diagnostic_reference_cid"]
            or semantic.get("diagnostic_reference_disposition")
            != "historical_non_authoritative_reference"
            or semantic.get("diagnostic_reference_authoritative") is not False
            or not isinstance(events, list)
            or not isinstance(transitions, list)
            or not isinstance(operations, list)
            or profile.get("semantic_check_event_count") != len(events)
            or profile.get("semantic_check_event_root")
            != content_identity(events)
            or profile.get("semantic_operation_evidence_count")
            != len(operations)
            or profile.get("semantic_operation_evidence_root")
            != content_identity(operations)
            or profile.get("semantic_transition_evidence_count")
            != len(transitions)
            or profile.get("semantic_transition_evidence_root")
            != content_identity(transitions)
            or semantic.get("ordered_task_counts")
            != profile["ordered_expected_task_counts"]
            or semantic.get("ordered_normalized_task_roots")
            != profile["ordered_normalized_task_roots"]
            or semantic.get("helper_references_dropped") is not True
            or semantic.get("task_identity_continuity_enforced") is not True
            or semantic.get("prior_task_identity_preserved_on_growth") is not True
            or semantic.get("instrumentation_restored") is not True
            or semantic.get("infrastructure_not_proof") is not True
            or semantic.get("cache_authority") is not False
            or semantic.get("completion_authoritative") is not False
            or semantic_cid != profile["semantic_execution_cid"]
            or observation.get("semantic_execution_cid") != semantic_cid
            or observation.get("semantic_diagnostic_reference_cid")
            != profile["semantic_diagnostic_reference_cid"]
            or observation.get("diagnostic_reference_disposition")
            != "historical_non_authoritative_reference"
            or observation.get("diagnostic_reference_authoritative") is not False
            or observation.get("ordered_operation_evidence") != operations
            or observation.get("ordered_transition_evidence") != transitions
            or observation.get("z3_check_event_count") != len(events)
            or observation.get("ordered_task_counts")
            != profile["ordered_expected_task_counts"]
            or observation.get("ordered_normalized_task_roots")
            != profile["ordered_normalized_task_roots"]
            or observation.get("lifetime_boundary_count")
            != len(profile["lifetime_boundaries"])
            or observation.get("lifetime_boundary_root")
            != content_identity(profile["lifetime_boundaries"])
            or observation.get("facade_reset_disposition")
            != profile["facade_reset_disposition"]
            or observation.get("gc_collection_count")
            != profile["gc_collection_count"]
            or observation.get("stability_sample_scope")
            != profile["stability_sample_scope"]
            or observation.get("stability_checkpoint_count")
            != profile["stability_checkpoint_count"]
            or observation.get("stability_samples_per_checkpoint")
            != profile["stability_samples_per_checkpoint"]
            or observation.get("stability_interval_ms")
            != profile["stability_interval_ms"]
        ):
            raise QualificationError(f"{task_id} semantic execution differs")
        if task_id == "LGCVF-051":
            if semantic_body != _recovery_051_expected_execution_body():
                raise QualificationError("LGCVF-051 semantic body differs")
        elif task_id == "LGCVF-060":
            expected_semantic_fields = {
                "schema",
                "diagnostic_reference_cid",
                "diagnostic_reference_disposition",
                "diagnostic_reference_authoritative",
                "ordered_operation_evidence",
                "ordered_check_events",
                "ordered_transition_evidence",
                "operation_event_counts",
                "ordered_task_counts",
                "ordered_normalized_task_roots",
                "helper_references_dropped",
                "gc_collection_count",
                "stability_sample_count_per_transition",
                "stability_interval_ms",
                "task_identity_continuity_enforced",
                "prior_task_identity_preserved_on_growth",
                "instrumentation_restored",
                "infrastructure_not_proof",
                "cache_authority",
                "completion_authoritative",
            }
            if (
                set(semantic) != expected_semantic_fields
                or semantic.get("schema")
                != "lgcvf-recovery-060-public-prefix-execution@1"
                or len(events) != 12
                or len(transitions) != 2
                or len(operations) != 2
                or semantic.get("operation_event_counts") != [8, 4]
                or semantic.get("gc_collection_count")
                != profile["gc_collection_count"]
                or semantic.get("stability_sample_count_per_transition")
                != profile["stability_samples_per_checkpoint"]
                or semantic.get("stability_interval_ms")
                != profile["stability_interval_ms"]
            ):
                raise QualificationError("LGCVF-060 semantic body differs")
            for ordinal, event in enumerate(events, start=1):
                if (
                    not isinstance(event, Mapping)
                    or set(event)
                    != {
                        "operation_ordinal",
                        "operation_event_ordinal",
                        "timeout_ms",
                        "assertion_count",
                        "assertion_root",
                        "assumption_count",
                        "assumption_root",
                        "result",
                        "core_count",
                        "core_root",
                        "before_count",
                        "after_count",
                        "before_normalized_task_root",
                        "after_normalized_task_root",
                        "task_identity_relation",
                    }
                    or isinstance(event.get("operation_ordinal"), bool)
                    or event.get("operation_ordinal") not in {1, 2}
                    or isinstance(event.get("operation_event_ordinal"), bool)
                    or not isinstance(event.get("operation_event_ordinal"), int)
                    or event.get("operation_event_ordinal") <= 0
                    or event.get("timeout_ms") != profile["timeout_ms"]
                    or any(
                        isinstance(event.get(field), bool)
                        or not isinstance(event.get(field), int)
                        or event.get(field) < 0
                        for field in (
                            "assertion_count",
                            "assumption_count",
                            "core_count",
                            "before_count",
                            "after_count",
                        )
                    )
                    or event.get("result") not in {"sat", "unsat", "unknown"}
                    or event.get("task_identity_relation")
                    not in {"unchanged", "one_addition_preserving_prior"}
                    or any(
                        not _is_canonical_content_cid(event.get(field))
                        for field in (
                            "assertion_root",
                            "assumption_root",
                            "core_root",
                            "before_normalized_task_root",
                            "after_normalized_task_root",
                        )
                    )
                ):
                    raise QualificationError(
                        f"LGCVF-060 semantic event {ordinal} differs"
                    )
        else:
            raise QualificationError("recovery public-prefix suite differs")
    elif profile.get("disposition") in {
        "single_task_no_z3",
        "single_task_z3_import_denied_as_typed_unavailable",
    }:
        if (
            observation.get("semantic_diagnostic_reference_cid") != ""
            or observation.get("diagnostic_reference_disposition")
            != "not_applicable"
            or observation.get("diagnostic_reference_authoritative") is not False
            or observation.get("semantic_execution") is not None
            or observation.get("semantic_execution_cid") != ""
            or profile.get("semantic_execution_cid") != ""
            or profile.get("semantic_check_event_count") != 0
            or profile.get("semantic_check_event_root") != content_identity([])
            or profile.get("semantic_operation_evidence_count") != 0
            or profile.get("semantic_operation_evidence_root")
            != content_identity([])
            or profile.get("semantic_transition_evidence_count") != 0
            or profile.get("semantic_transition_evidence_root")
            != content_identity([])
            or observation.get("ordered_operation_evidence") != []
            or observation.get("ordered_transition_evidence") != []
            or observation.get("z3_check_event_count") != 0
            or observation.get("ordered_task_counts")
            != [profile["expected_task_count"]]
            or observation.get("ordered_normalized_task_roots")
            != [profile["expected_normalized_task_root"]]
            or observation.get("lifetime_boundary_count") != 0
            or observation.get("lifetime_boundary_root")
            != content_identity([])
            or observation.get("facade_reset_disposition")
            != profile["facade_reset_disposition"]
            or observation.get("gc_collection_count")
            != profile["gc_collection_count"]
            or observation.get("stability_sample_scope")
            != profile["stability_sample_scope"]
            or observation.get("stability_checkpoint_count")
            != profile["stability_checkpoint_count"]
            or observation.get("stability_samples_per_checkpoint")
            != profile["stability_samples_per_checkpoint"]
            or observation.get("stability_interval_ms")
            != profile["stability_interval_ms"]
        ):
            raise QualificationError("recovery single-task observation differs")
    else:
        raise QualificationError("recovery task observation disposition differs")
    return dict(observation)


_RECOVERY_SUITE_TASK_RECEIPT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "task_id",
        "suite_id",
        "runtime_cid",
        "source_projection_root",
        "policy_matrix_cid",
        "profile_cid",
        "z3_import_policy_disposition",
        "z3_import_policy_cid",
        "z3_expected_meta_denial_count",
        "z3_expected_meta_denial_root",
        "z3_expected_open_boundary_denial_count",
        "z3_expected_open_boundary_denial_root",
        "z3_trusted_source_revalidation_disposition",
        "z3_expected_trusted_source_event_count",
        "z3_expected_trusted_source_event_root",
        "z3_policy_namespace_unavailability",
        "z3_live_cegar_disposition",
        "z3_candidate_reason_interpretation",
        "semantic_execution_cid",
        "full_live_observation_cid",
        "semantic_check_event_count",
        "semantic_check_event_root",
        "semantic_operation_evidence_count",
        "semantic_operation_evidence_root",
        "semantic_transition_evidence_count",
        "semantic_transition_evidence_root",
        "ordered_task_counts",
        "ordered_normalized_task_roots",
        "lifetime_boundary_count",
        "lifetime_boundary_root",
        "helper_references_dropped",
        "facade_reset_disposition",
        "gc_collection_count",
        "stability_sample_scope",
        "stability_checkpoint_count",
        "stability_samples_per_checkpoint",
        "stability_interval_ms",
        "stability_verified",
        "instrumentation_restored",
        "fresh_worker_recomputed",
        "full_live_observation_persisted",
        "full_live_observation_parent_validated",
        "direct_verifier_reconstructs_full_live_observation",
        "disposition",
        "infrastructure_not_proof",
        "cache_authority",
        "completion_authoritative",
        "receipt_cid",
    }
)


def _recovery_suite_task_receipt_expectations(
    *,
    task_id: str,
    suite_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the config-pinned compact task-receipt commitments."""

    profile = _recovery_suite_task_policy(task_id=task_id, suite_id=suite_id)
    matrix = _recovery_suite_task_policy_matrix()
    if profile.get("disposition") == "public_prefix_fixed_point_required":
        expected = {
            "semantic_execution_cid": profile["semantic_execution_cid"],
            "semantic_check_event_count": profile[
                "semantic_check_event_count"
            ],
            "semantic_check_event_root": profile["semantic_check_event_root"],
            "semantic_operation_evidence_count": profile[
                "semantic_operation_evidence_count"
            ],
            "semantic_operation_evidence_root": profile[
                "semantic_operation_evidence_root"
            ],
            "semantic_transition_evidence_count": profile[
                "semantic_transition_evidence_count"
            ],
            "semantic_transition_evidence_root": profile[
                "semantic_transition_evidence_root"
            ],
            "ordered_task_counts": list(profile["ordered_expected_task_counts"]),
            "ordered_normalized_task_roots": list(
                profile["ordered_normalized_task_roots"]
            ),
            "lifetime_boundary_count": len(profile["lifetime_boundaries"]),
            "lifetime_boundary_root": content_identity(
                profile["lifetime_boundaries"]
            ),
            "facade_reset_disposition": profile[
                "facade_reset_disposition"
            ],
            "gc_collection_count": profile["gc_collection_count"],
            "stability_sample_scope": profile["stability_sample_scope"],
            "stability_checkpoint_count": profile[
                "stability_checkpoint_count"
            ],
            "stability_samples_per_checkpoint": profile[
                "stability_samples_per_checkpoint"
            ],
            "stability_interval_ms": profile["stability_interval_ms"],
        }
    else:
        if profile.get("disposition") not in {
            "single_task_no_z3",
            "single_task_z3_import_denied_as_typed_unavailable",
        }:
            raise QualificationError("recovery suite task receipt is unresolved")
        expected = {
            "semantic_execution_cid": profile["semantic_execution_cid"],
            "semantic_check_event_count": profile[
                "semantic_check_event_count"
            ],
            "semantic_check_event_root": profile["semantic_check_event_root"],
            "semantic_operation_evidence_count": profile[
                "semantic_operation_evidence_count"
            ],
            "semantic_operation_evidence_root": profile[
                "semantic_operation_evidence_root"
            ],
            "semantic_transition_evidence_count": profile[
                "semantic_transition_evidence_count"
            ],
            "semantic_transition_evidence_root": profile[
                "semantic_transition_evidence_root"
            ],
            "ordered_task_counts": [profile["expected_task_count"]],
            "ordered_normalized_task_roots": [
                profile["expected_normalized_task_root"]
            ],
            "lifetime_boundary_count": len(profile["lifetime_boundaries"]),
            "lifetime_boundary_root": content_identity(
                profile["lifetime_boundaries"]
            ),
            "facade_reset_disposition": profile[
                "facade_reset_disposition"
            ],
            "gc_collection_count": profile["gc_collection_count"],
            "stability_sample_scope": profile["stability_sample_scope"],
            "stability_checkpoint_count": profile[
                "stability_checkpoint_count"
            ],
            "stability_samples_per_checkpoint": profile[
                "stability_samples_per_checkpoint"
            ],
            "stability_interval_ms": profile["stability_interval_ms"],
        }
    return profile, {
        **expected,
        **_recovery_z3_import_policy_commitments(
            task_id=task_id,
            suite_id=suite_id,
        ),
        "policy_matrix_cid": matrix["matrix_cid"],
    }


def _recovery_suite_task_receipt_from_live_observation(
    observation: Mapping[str, Any],
    *,
    task_id: str,
    suite_id: str,
    runtime_cid: str,
    source_projection_root: str,
) -> dict[str, Any]:
    """Project a fully validated live observation to parent-persistable evidence."""

    live = _validate_recovery_suite_task_observation(
        observation,
        task_id=task_id,
        suite_id=suite_id,
        runtime_cid=runtime_cid,
        source_projection_root=source_projection_root,
    )
    profile, expected = _recovery_suite_task_receipt_expectations(
        task_id=task_id,
        suite_id=suite_id,
    )
    if profile.get("disposition") == "public_prefix_fixed_point_required":
        semantic = live.get("semantic_execution")
        if not isinstance(semantic, Mapping):
            raise QualificationError(
                f"{task_id} live semantic execution is absent"
            )
        events = semantic.get("ordered_check_events")
        operations = semantic.get("ordered_operation_evidence")
        transitions = semantic.get("ordered_transition_evidence")
        if (
            not isinstance(events, list)
            or not isinstance(operations, list)
            or not isinstance(transitions, list)
        ):
            raise QualificationError(
                f"{task_id} live semantic commitments are absent"
            )
        observed_commitments = {
            **_recovery_z3_import_policy_commitments(
                task_id=task_id,
                suite_id=suite_id,
            ),
            "semantic_execution_cid": live["semantic_execution_cid"],
            "semantic_check_event_count": len(events),
            "semantic_check_event_root": content_identity(events),
            "semantic_operation_evidence_count": len(operations),
            "semantic_operation_evidence_root": content_identity(operations),
            "semantic_transition_evidence_count": len(transitions),
            "semantic_transition_evidence_root": content_identity(transitions),
            "ordered_task_counts": list(live["ordered_task_counts"]),
            "ordered_normalized_task_roots": list(
                live["ordered_normalized_task_roots"]
            ),
            "lifetime_boundary_count": live["lifetime_boundary_count"],
            "lifetime_boundary_root": live["lifetime_boundary_root"],
            "facade_reset_disposition": live[
                "facade_reset_disposition"
            ],
            "gc_collection_count": live["gc_collection_count"],
            "stability_sample_scope": live["stability_sample_scope"],
            "stability_checkpoint_count": live[
                "stability_checkpoint_count"
            ],
            "stability_samples_per_checkpoint": live[
                "stability_samples_per_checkpoint"
            ],
            "stability_interval_ms": live["stability_interval_ms"],
            "policy_matrix_cid": live["policy_matrix_cid"],
        }
    else:
        observed_commitments = {
            **_recovery_z3_import_policy_commitments(
                task_id=task_id,
                suite_id=suite_id,
            ),
            "semantic_execution_cid": live["semantic_execution_cid"],
            "semantic_check_event_count": live["z3_check_event_count"],
            "semantic_check_event_root": content_identity([]),
            "semantic_operation_evidence_count": len(
                live["ordered_operation_evidence"]
            ),
            "semantic_operation_evidence_root": content_identity(
                live["ordered_operation_evidence"]
            ),
            "semantic_transition_evidence_count": len(
                live["ordered_transition_evidence"]
            ),
            "semantic_transition_evidence_root": content_identity(
                live["ordered_transition_evidence"]
            ),
            "ordered_task_counts": list(live["ordered_task_counts"]),
            "ordered_normalized_task_roots": list(
                live["ordered_normalized_task_roots"]
            ),
            "lifetime_boundary_count": live["lifetime_boundary_count"],
            "lifetime_boundary_root": live["lifetime_boundary_root"],
            "facade_reset_disposition": live[
                "facade_reset_disposition"
            ],
            "gc_collection_count": live["gc_collection_count"],
            "stability_sample_scope": live["stability_sample_scope"],
            "stability_checkpoint_count": live[
                "stability_checkpoint_count"
            ],
            "stability_samples_per_checkpoint": live[
                "stability_samples_per_checkpoint"
            ],
            "stability_interval_ms": live["stability_interval_ms"],
            "policy_matrix_cid": live["policy_matrix_cid"],
        }
    if observed_commitments != expected:
        raise QualificationError("recovery live task commitments differ")
    value: dict[str, Any] = {
        "schema": "lgcvf-recovery-suite-task-receipt@4",
        "task_id": task_id,
        "suite_id": suite_id,
        "runtime_cid": runtime_cid,
        "source_projection_root": source_projection_root,
        "policy_matrix_cid": expected["policy_matrix_cid"],
        "profile_cid": profile["profile_cid"],
        "z3_import_policy_disposition": expected[
            "z3_import_policy_disposition"
        ],
        "z3_import_policy_cid": expected["z3_import_policy_cid"],
        "z3_expected_meta_denial_count": expected[
            "z3_expected_meta_denial_count"
        ],
        "z3_expected_meta_denial_root": expected[
            "z3_expected_meta_denial_root"
        ],
        "z3_expected_open_boundary_denial_count": expected[
            "z3_expected_open_boundary_denial_count"
        ],
        "z3_expected_open_boundary_denial_root": expected[
            "z3_expected_open_boundary_denial_root"
        ],
        "z3_trusted_source_revalidation_disposition": expected[
            "z3_trusted_source_revalidation_disposition"
        ],
        "z3_expected_trusted_source_event_count": expected[
            "z3_expected_trusted_source_event_count"
        ],
        "z3_expected_trusted_source_event_root": expected[
            "z3_expected_trusted_source_event_root"
        ],
        "z3_policy_namespace_unavailability": expected[
            "z3_policy_namespace_unavailability"
        ],
        "z3_live_cegar_disposition": expected["z3_live_cegar_disposition"],
        "z3_candidate_reason_interpretation": expected[
            "z3_candidate_reason_interpretation"
        ],
        "semantic_execution_cid": expected["semantic_execution_cid"],
        "full_live_observation_cid": live["observation_cid"],
        "semantic_check_event_count": expected[
            "semantic_check_event_count"
        ],
        "semantic_check_event_root": expected["semantic_check_event_root"],
        "semantic_operation_evidence_count": expected[
            "semantic_operation_evidence_count"
        ],
        "semantic_operation_evidence_root": expected[
            "semantic_operation_evidence_root"
        ],
        "semantic_transition_evidence_count": expected[
            "semantic_transition_evidence_count"
        ],
        "semantic_transition_evidence_root": expected[
            "semantic_transition_evidence_root"
        ],
        "ordered_task_counts": expected["ordered_task_counts"],
        "ordered_normalized_task_roots": expected[
            "ordered_normalized_task_roots"
        ],
        "lifetime_boundary_count": expected["lifetime_boundary_count"],
        "lifetime_boundary_root": expected["lifetime_boundary_root"],
        "helper_references_dropped": True,
        "facade_reset_disposition": expected[
            "facade_reset_disposition"
        ],
        "gc_collection_count": expected["gc_collection_count"],
        "stability_sample_scope": expected["stability_sample_scope"],
        "stability_checkpoint_count": expected[
            "stability_checkpoint_count"
        ],
        "stability_samples_per_checkpoint": expected[
            "stability_samples_per_checkpoint"
        ],
        "stability_interval_ms": expected["stability_interval_ms"],
        "stability_verified": True,
        "instrumentation_restored": True,
        "fresh_worker_recomputed": True,
        "full_live_observation_persisted": False,
        "full_live_observation_parent_validated": True,
        "direct_verifier_reconstructs_full_live_observation": False,
        "disposition": "parent_live_validated_cid_not_reconstructed",
        "infrastructure_not_proof": True,
        "cache_authority": False,
        "completion_authoritative": False,
    }
    value["receipt_cid"] = content_identity(value)
    return value


def _validate_recovery_suite_task_receipt(
    value: Any,
    *,
    task_id: str,
    suite_id: str,
    runtime_cid: str,
    source_projection_root: str,
    full_live_observation_cid: str,
) -> dict[str, Any]:
    """Validate compact parent evidence without claiming full reconstruction."""

    if not isinstance(value, Mapping):
        raise QualificationError("recovery suite task receipt is absent")
    profile, expected = _recovery_suite_task_receipt_expectations(
        task_id=task_id,
        suite_id=suite_id,
    )
    body = {key: item for key, item in value.items() if key != "receipt_cid"}
    numeric_fields = (
        "z3_expected_meta_denial_count",
        "z3_expected_open_boundary_denial_count",
        "z3_expected_trusted_source_event_count",
        "semantic_check_event_count",
        "semantic_operation_evidence_count",
        "semantic_transition_evidence_count",
        "lifetime_boundary_count",
        "gc_collection_count",
        "stability_checkpoint_count",
        "stability_samples_per_checkpoint",
        "stability_interval_ms",
    )
    if (
        set(value) != _RECOVERY_SUITE_TASK_RECEIPT_FIELDS
        or value.get("schema") != "lgcvf-recovery-suite-task-receipt@4"
        or value.get("task_id") != task_id
        or value.get("suite_id") != suite_id
        or value.get("runtime_cid") != runtime_cid
        or value.get("source_projection_root") != source_projection_root
        or value.get("policy_matrix_cid") != expected["policy_matrix_cid"]
        or value.get("profile_cid") != profile["profile_cid"]
        or any(
            value.get(key) != expected[key]
            for key in _recovery_z3_import_policy_commitments(
                task_id=task_id,
                suite_id=suite_id,
            )
        )
        or value.get("semantic_execution_cid")
        != expected["semantic_execution_cid"]
        or value.get("full_live_observation_cid")
        != full_live_observation_cid
        or not _is_canonical_content_cid(full_live_observation_cid)
        or any(
            isinstance(value.get(field), bool)
            or not isinstance(value.get(field), int)
            or int(value[field]) < 0
            for field in numeric_fields
        )
        or value.get("semantic_check_event_count")
        != expected["semantic_check_event_count"]
        or value.get("semantic_check_event_root")
        != expected["semantic_check_event_root"]
        or value.get("semantic_operation_evidence_count")
        != expected["semantic_operation_evidence_count"]
        or value.get("semantic_operation_evidence_root")
        != expected["semantic_operation_evidence_root"]
        or value.get("semantic_transition_evidence_count")
        != expected["semantic_transition_evidence_count"]
        or value.get("semantic_transition_evidence_root")
        != expected["semantic_transition_evidence_root"]
        or value.get("ordered_task_counts") != expected["ordered_task_counts"]
        or not isinstance(value.get("ordered_task_counts"), list)
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in value["ordered_task_counts"]
        )
        or value.get("ordered_normalized_task_roots")
        != expected["ordered_normalized_task_roots"]
        or value.get("lifetime_boundary_count")
        != expected["lifetime_boundary_count"]
        or value.get("lifetime_boundary_root")
        != expected["lifetime_boundary_root"]
        or value.get("helper_references_dropped") is not True
        or value.get("facade_reset_disposition")
        != expected["facade_reset_disposition"]
        or value.get("gc_collection_count") != expected["gc_collection_count"]
        or value.get("stability_sample_scope")
        != expected["stability_sample_scope"]
        or value.get("stability_checkpoint_count")
        != expected["stability_checkpoint_count"]
        or value.get("stability_samples_per_checkpoint")
        != expected["stability_samples_per_checkpoint"]
        or value.get("stability_interval_ms")
        != expected["stability_interval_ms"]
        or value.get("stability_verified") is not True
        or value.get("instrumentation_restored") is not True
        or value.get("fresh_worker_recomputed") is not True
        or value.get("full_live_observation_persisted") is not False
        or value.get("full_live_observation_parent_validated") is not True
        or value.get("direct_verifier_reconstructs_full_live_observation")
        is not False
        or value.get("disposition")
        != "parent_live_validated_cid_not_reconstructed"
        or value.get("infrastructure_not_proof") is not True
        or value.get("cache_authority") is not False
        or value.get("completion_authoritative") is not False
        or value.get("receipt_cid") != content_identity(body)
    ):
        raise QualificationError("recovery suite task receipt differs")
    roots = value.get("ordered_normalized_task_roots")
    if not isinstance(roots, list) or any(
        not _is_canonical_content_cid(item) for item in roots
    ) or not _is_canonical_content_cid(value.get("lifetime_boundary_root")):
        raise QualificationError("recovery suite task receipt roots differ")
    return dict(value)


def _expected_worker_writable_executable_mappings(
    *, z3_required: bool,
) -> list[dict[str, Any]]:
    if not z3_required:
        return []
    return [
        {
            "kind": "anonymous",
            "permissions": "rwxp",
            "offset": "00000000",
            "device": "00:00",
            "inode": "0",
            "size_bytes": 4096,
            "path_token": "anonymous",
        }
    ]


def _write_bounded_json_line(
    descriptor: int,
    value: Mapping[str, Any],
    *,
    noun: str,
) -> None:
    payload = _canonical_bytes(value) + b"\n"
    if len(payload) > _MAX_WORKER_ATTESTATION_BYTES:
        raise QualificationError(f"{noun} exceeds its bound")
    view = memoryview(payload)
    try:
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise QualificationError(f"{noun} pipe made no progress")
            view = view[written:]
    except OSError as exc:
        raise QualificationError(f"{noun} pipe write failed") from exc


def _read_bounded_json_line(
    descriptor: int,
    *,
    noun: str,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    deadline = (
        None if timeout_seconds is None else time.monotonic() + timeout_seconds
    )
    payload = bytearray()
    while True:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise QualificationError(f"{noun} timed out")
            try:
                readable, _, _ = select.select([descriptor], [], [], remaining)
            except OSError as exc:
                raise QualificationError(f"{noun} pipe wait failed") from exc
            if not readable:
                raise QualificationError(f"{noun} timed out")
        try:
            chunk = os.read(
                descriptor,
                min(4096, _MAX_WORKER_ATTESTATION_BYTES + 1 - len(payload)),
            )
        except OSError as exc:
            raise QualificationError(f"{noun} pipe read failed") from exc
        if not chunk:
            raise QualificationError(f"{noun} pipe closed before its message")
        newline = chunk.find(b"\n")
        if newline >= 0:
            payload.extend(chunk[:newline])
            if chunk[newline + 1 :]:
                raise QualificationError(f"{noun} contains duplicate messages")
            break
        payload.extend(chunk)
        if len(payload) > _MAX_WORKER_ATTESTATION_BYTES:
            raise QualificationError(f"{noun} exceeds its bound")
    try:
        text = payload.decode("utf-8", errors="strict")
        value = _strict_json_loads(text, noun=noun)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationError(f"{noun} is malformed") from exc
    if not isinstance(value, dict) or _canonical_bytes(value) != bytes(payload):
        raise QualificationError(f"{noun} is not canonical")
    return value


def _require_pipe_eof(
    descriptor: int,
    *,
    noun: str,
    timeout_seconds: float,
) -> None:
    try:
        readable, _, _ = select.select([descriptor], [], [], timeout_seconds)
    except OSError as exc:
        raise QualificationError(f"{noun} pipe wait failed") from exc
    if not readable:
        raise QualificationError(f"{noun} pipe did not close within its bound")
    try:
        trailing = os.read(descriptor, _MAX_WORKER_ATTESTATION_BYTES + 1)
    except OSError as exc:
        raise QualificationError(f"{noun} pipe close read failed") from exc
    if trailing:
        raise QualificationError(f"{noun} contains trailing messages")


class _RecoveryNativeLoadGuard:
    """Irreversibly deny unbound in-process native loading in one worker."""

    def __init__(
        self,
        runtime_root: Path,
        bundle: Mapping[str, Any],
        *,
        z3_required: bool,
    ) -> None:
        self.runtime_root = runtime_root.resolve(strict=True)
        self.z3_required = bool(z3_required)
        native_platform = bundle.get("native_platform_binding")
        payload_native = (
            native_platform.get("solver_payload_native_files")
            if isinstance(native_platform, Mapping)
            else None
        )
        if not isinstance(payload_native, list):
            raise QualificationError("native load guard payload authority is absent")
        self.projected: dict[str, dict[str, Any]] = {}
        projected_summary: list[dict[str, Any]] = []
        for item in payload_native:
            if not isinstance(item, Mapping):
                raise QualificationError("native load guard payload differs")
            relative = str(item.get("path") or "")
            _strict_record_relative_path(relative, noun="native load guard payload")
            if relative.startswith("z3/") and not self.z3_required:
                continue
            absolute = self.runtime_root / relative
            try:
                status = absolute.lstat()
            except OSError as exc:
                raise QualificationError(
                    "native load guard projected member is unavailable"
                ) from exc
            if (
                not stat.S_ISREG(status.st_mode)
                or status.st_uid != os.geteuid()
                or status.st_nlink != 1
                or stat.S_IMODE(status.st_mode) != 0o500
                or status.st_size != item.get("size_bytes")
                or str(absolute.resolve(strict=True)) != str(absolute)
            ):
                raise QualificationError(
                    "native load guard projected member identity differs"
                )
            normalized = {
                "path_token": "runtime:" + relative,
                "sha256": item.get("sha256"),
                "size_bytes": item.get("size_bytes"),
            }
            self.projected[str(absolute)] = normalized
            projected_summary.append(normalized)
        projected_summary.sort(key=lambda item: str(item["path_token"]))
        self.projected_root = content_identity(projected_summary)
        self.suite_native_policy = _recovery_suite_native_policy(
            z3_required=self.z3_required
        )
        (
            self.stdlib,
            self.stdlib_root,
            self.stdlib_total_bytes,
        ) = _stdlib_extension_native_manifest()
        python_runtime = bundle.get("python_runtime_binding")
        stdlib_binding = (
            python_runtime.get("stdlib_extension_binding")
            if isinstance(python_runtime, Mapping)
            else None
        )
        rwx_producer = (
            python_runtime.get("z3_libffi_rwx_producer_binding")
            if isinstance(python_runtime, Mapping)
            else None
        )
        if (
            not isinstance(stdlib_binding, Mapping)
            or stdlib_binding.get("schema")
            != "lgcvf-stdlib-extension-native-binding@1"
            or stdlib_binding.get("file_count") != len(self.stdlib)
            or stdlib_binding.get("total_bytes") != self.stdlib_total_bytes
            or stdlib_binding.get("file_manifest_root") != self.stdlib_root
            or not isinstance(rwx_producer, Mapping)
            or rwx_producer.get("schema")
            != "lgcvf-z3-libffi-rwx-producer-binding@1"
            or rwx_producer.get("producer_binding_cid")
            != content_identity(
                {
                    key: item
                    for key, item in rwx_producer.items()
                    if key != "producer_binding_cid"
                }
            )
            or rwx_producer.get("qualified_worker_scope")
            != ["LGCVF-051", "LGCVF-060"]
            or rwx_producer.get("expected_anonymous_mapping")
            != _expected_worker_writable_executable_mappings(z3_required=True)[0]
        ):
            raise QualificationError("stdlib extension policy binding differs")
        self.rwx_producer_binding_cid = str(
            rwx_producer["producer_binding_cid"]
        )
        self.attempts: list[dict[str, Any]] = []
        self.unauthorized_attempts: list[dict[str, Any]] = []
        self._sequence = 0

        def audit(event: str, arguments: tuple[Any, ...]) -> None:
            if event == "mmap.__new__":
                self._sequence += 1
                self.unauthorized_attempts.append(
                    {
                        "sequence": self._sequence,
                        "event": event,
                        "target_token": "anonymous_executable_capability",
                        "disposition": "denied",
                    }
                )
                raise QualificationError(
                    "Python mmap is denied during recovery tests"
                )
            if event == "ctypes.dlopen":
                target = arguments[0] if arguments else None
                self._admit(event, target)
                return
            if event != "import" or len(arguments) < 2:
                return
            target = arguments[1]
            if not isinstance(target, str) or not _qualification_runtime_is_native(
                target
            ):
                return
            self._admit("extension_import", target)

        self._audit_hook = audit
        sys.addaudithook(audit)

    def _admit(self, event: str, target: Any) -> None:
        self._sequence += 1
        if not isinstance(target, str) or not target or not Path(target).is_absolute():
            denied = {
                "sequence": self._sequence,
                "event": event,
                "target_token": "invalid:" + _sha256_bytes(repr(target).encode()),
                "disposition": "denied",
            }
            self.unauthorized_attempts.append(denied)
            raise QualificationError("unbound native load target is denied")
        if "\x00" in target or os.path.normpath(target) != target:
            denied = {
                "sequence": self._sequence,
                "event": event,
                "target_token": "invalid:" + _sha256_bytes(target.encode()),
                "disposition": "denied",
            }
            self.unauthorized_attempts.append(denied)
            raise QualificationError("noncanonical native load target is denied")
        admitted = self.projected.get(target) or self.stdlib.get(target)
        if admitted is None:
            denied = {
                "sequence": self._sequence,
                "event": event,
                "target_token": "unbound:" + _sha256_bytes(target.encode()),
                "disposition": "denied",
            }
            self.unauthorized_attempts.append(denied)
            raise QualificationError("ambient native load target is denied")
        try:
            status = os.lstat(target)
        except OSError as exc:
            raise QualificationError("admitted native load target disappeared") from exc
        expected_uid = os.geteuid() if target in self.projected else 0
        expected_mode = 0o500 if target in self.projected else int(admitted["mode"])
        if (
            not stat.S_ISREG(status.st_mode)
            or status.st_uid != expected_uid
            or status.st_nlink != 1
            or stat.S_IMODE(status.st_mode) != expected_mode
            or status.st_size != admitted["size_bytes"]
            or str(Path(target).resolve(strict=True)) != target
        ):
            raise QualificationError("admitted native load target identity changed")
        self.attempts.append(
            {
                "sequence": self._sequence,
                "event": event,
                "target_token": admitted["path_token"],
                "disposition": "allowed",
            }
        )

    def evidence(
        self,
        *,
        before_maps: Sequence[Mapping[str, Any]],
        after_maps: Sequence[Mapping[str, Any]],
        before_threads: Sequence[Mapping[str, Any]],
        after_threads: Sequence[Mapping[str, Any]],
        before_kernel_tasks: Sequence[int],
        after_kernel_tasks: Sequence[int],
        before_child_processes: Sequence[int],
        after_child_processes: Sequence[int],
        diagnostic_phase: str = "unspecified",
    ) -> dict[str, Any]:
        before_by_identity = {
            content_identity(item): dict(item) for item in before_maps
        }
        after_by_identity = {
            content_identity(item): dict(item) for item in after_maps
        }
        if (
            len(before_by_identity) != len(before_maps)
            or len(after_by_identity) != len(after_maps)
        ):
            raise QualificationError("worker executable mapping identity differs")
        added_identities = sorted(set(after_by_identity).difference(before_by_identity))
        additions: list[str] = []
        unauthorized_additions: list[str] = []
        expected_rwx = _expected_worker_writable_executable_mappings(
            z3_required=self.z3_required
        )
        for mapping_identity in added_identities:
            mapping = after_by_identity[mapping_identity]
            path = mapping.get("resolved_path")
            path = path if isinstance(path, str) else ""
            admitted = self.projected.get(path) or self.stdlib.get(path)
            candidate_signature = _writable_executable_mapping_signatures(
                [mapping]
            )
            if (
                admitted is None
                and self.z3_required
                and candidate_signature == expected_rwx
            ):
                additions.append("z3-libffi-anonymous-rwx:4096")
                continue
            if admitted is None:
                raw_label = str(mapping.get("label") or "anonymous")
                unauthorized_additions.append(
                    "unbound-map:"
                    + raw_label[:64]
                    + ":"
                    + PurePath(path).name[:128]
                    + ":"
                    + mapping_identity
                )
            else:
                additions.append(str(admitted["path_token"]))
        additions = sorted(set(additions))
        expected_projected = {
            "runtime:" + path
            for path in _expected_solver_native_mapping_evidence(
                z3_required=self.z3_required
            )[
                "projected_solver_paths"
            ]
        }
        missing_projected = sorted(expected_projected.difference(additions))
        before_rwx = _writable_executable_mapping_signatures(before_maps)
        after_rwx = _writable_executable_mapping_signatures(after_maps)
        expected_after_rwx = expected_rwx
        normalized_before_threads = [dict(item) for item in before_threads]
        normalized_after_threads = [dict(item) for item in after_threads]
        normalized_before_tasks = list(before_kernel_tasks)
        normalized_after_tasks = list(after_kernel_tasks)
        normalized_before_children = list(before_child_processes)
        normalized_after_children = list(after_child_processes)
        if (
            self.unauthorized_attempts
            or unauthorized_additions
            or missing_projected
            or normalized_after_threads != normalized_before_threads
            or normalized_after_tasks != normalized_before_tasks
            or normalized_before_children
            or normalized_after_children
            or before_rwx
            or after_rwx != expected_after_rwx
        ):
            diagnostic = [
                *(str(item["target_token"]) for item in self.unauthorized_attempts),
                *unauthorized_additions,
                *("missing:" + item for item in missing_projected),
                *(
                    ["python-thread-population-differs"]
                    if normalized_after_threads != normalized_before_threads
                    else []
                ),
                *(
                    [
                        "kernel-task-population-differs:"
                        + str(len(normalized_before_tasks))
                        + ":"
                        + str(len(normalized_after_tasks))
                        + ":"
                        + content_identity(normalized_before_tasks)
                        + ":"
                        + content_identity(normalized_after_tasks)
                    ]
                    if normalized_after_tasks != normalized_before_tasks
                    else []
                ),
                *(
                    ["child-process-population-differs"]
                    if normalized_before_children or normalized_after_children
                    else []
                ),
                *(["pre-solver-wx-present"] if before_rwx else []),
                *(
                    ["post-solver-wx-differs"]
                    if after_rwx != expected_after_rwx
                    else []
                ),
            ]
            raise QualificationError(
                "unbound native execution was observed during "
                + diagnostic_phase
                + ": "
                + ",".join(diagnostic[:16])
            )
        normalized_thread_population = sorted(
            (
                {
                    "name": str(item.get("name") or ""),
                    "daemon": bool(item.get("daemon")),
                }
                for item in normalized_before_threads
            ),
            key=_canonical_bytes,
        )
        value: dict[str, Any] = {
            "schema": "lgcvf-recovery-native-load-guard@3",
            "suite_native_policy": self.suite_native_policy,
            "installed_before_solver_import": True,
            "irreversible_until_worker_exit": True,
            "projected_native_manifest_root": self.projected_root,
            "stdlib_extension_file_count": len(self.stdlib),
            "stdlib_extension_total_bytes": self.stdlib_total_bytes,
            "stdlib_extension_manifest_root": self.stdlib_root,
            "rwx_producer_binding_cid": self.rwx_producer_binding_cid,
            "ordered_attempts": list(self.attempts),
            "ordered_attempt_root": content_identity(self.attempts),
            "pre_solver_executable_mapping_root": content_identity(
                _normalized_executable_mapping_signatures(before_maps)
            ),
            "pre_solver_executable_mapping_count": len(before_maps),
            "post_solver_executable_mapping_root": content_identity(
                _normalized_executable_mapping_signatures(after_maps)
            ),
            "post_solver_executable_mapping_count": len(after_maps),
            "pre_solver_writable_executable_mappings": before_rwx,
            "post_solver_writable_executable_mappings": after_rwx,
            "writable_executable_limitation": (
                "z3_libffi_rwx_4k_ephemeral_worker"
                if self.z3_required
                else "none"
            ),
            "pre_pytest_thread_count": len(normalized_before_threads),
            "pre_pytest_thread_population_root": content_identity(
                normalized_thread_population
            ),
            "thread_population_restored": True,
            "pre_pytest_kernel_task_count": len(normalized_before_tasks),
            "kernel_task_population_restored": True,
            "pre_pytest_child_process_count": len(normalized_before_children),
            "child_process_population_restored": True,
            "post_pytest_mapping_additions": additions,
            "post_pytest_mapping_addition_root": content_identity(additions),
            "unauthorized_attempts": [],
            "unauthorized_mapping_additions": [],
        }
        value["guard_cid"] = content_identity(value)
        return value


def _validate_recovery_z3_import_denial_evidence(
    value: Any,
    *,
    phase: str,
    task_id: str,
    suite_id: str,
    source_projection_root: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or phase not in {"prepared", "final"}:
        raise QualificationError("recovery Z3 import denial evidence is absent")
    expected_policy = _recovery_z3_import_denial_policy(
        task_id=task_id,
        suite_id=suite_id,
    )
    enabled = task_id in _RECOVERY_Z3_IMPORT_DENIED_TASKS
    expected_meta = (
        []
        if phase == "prepared"
        else list(expected_policy["expected_meta_denials"])
    )
    expected_calls = (
        _recovery_061_expected_pytest_call_nodeids()
        if enabled and phase == "final"
        else []
    )
    expected_trusted_source_events = (
        [
            {**dict(item), "source_projection_root": source_projection_root}
            for item in expected_policy["expected_trusted_source_events"]
        ]
        if enabled and phase == "final"
        else []
    )
    body = {key: item for key, item in value.items() if key != "evidence_cid"}
    expected_fields = {
        "schema",
        "phase",
        "task_id",
        "suite_id",
        "policy_cid",
        "policy_disposition",
        "top_level_import_audit_claimed",
        "irreversible_audit_open_boundary_installed",
        "meta_path_guard_state",
        "owner_phase",
        "process_exit_removal_boundary",
        "pytest_meta_path_lifecycle_disposition",
        "pytest_meta_path_admission_count",
        "pytest_meta_path_call_start_validation_count",
        "pytest_meta_path_sessionfinish_validation_count",
        "pytest_meta_path_unconfigure_validation_count",
        "pytest_meta_path_return_restoration_count",
        "pytest_meta_path_candidate_tuple_validated",
        "pytest_meta_path_bootstrap_tuple_restored",
        "trusted_revalidation_owner_thread_only",
        "trusted_revalidation_scope_closed",
        "trusted_source_revalidation_disposition",
        "trusted_source_revalidation_source_projection_root",
        "trusted_source_revalidation_expected_event_count",
        "trusted_source_revalidation_expected_event_root",
        "ordered_trusted_source_revalidation_events",
        "trusted_source_revalidation_event_count",
        "trusted_source_revalidation_event_root",
        "trusted_source_revalidation_scope_entry_count",
        "trusted_source_revalidation_scope_exit_count",
        "trusted_source_revalidation_scope_completed",
        "trusted_source_revalidation_pending_empty",
        "trusted_source_revalidation_confirmation_count",
        "trusted_source_revalidation_owner_thread_only",
        "trusted_source_revalidation_caller_code_identity_exact",
        "trusted_source_revalidation_descriptor_identity_validated",
        "trusted_source_revalidation_global_z3_exemption",
        "trusted_source_revalidation_audit_dirfd_observed",
        "trusted_source_revalidation_telemetry_authoritative",
        "trusted_source_revalidation_telemetry_reconstructed",
        "ordered_meta_denials",
        "meta_denial_count",
        "meta_denial_root",
        "ordered_open_boundary_denials",
        "open_boundary_denial_count",
        "open_boundary_denial_root",
        "trusted_revalidation_scope_entry_count",
        "trusted_revalidation_scope_exit_count",
        "trusted_revalidation_scope_completed",
        "trusted_revalidation_permitted_z3_open_count",
        "trusted_source_revalidation_expected_event_count",
        "trusted_source_revalidation_event_count",
        "trusted_source_revalidation_scope_entry_count",
        "trusted_source_revalidation_scope_exit_count",
        "trusted_source_revalidation_confirmation_count",
        "trusted_revalidation_telemetry_authoritative",
        "trusted_revalidation_telemetry_reconstructed",
        "z3_modules_absent",
        "z3_file_descriptor_count",
        "z3_file_descriptor_root",
        "policy_denied_z3_native_mapping_count",
        "policy_denied_z3_native_mapping_root",
        "z3_loader_executed",
        "pytest_call_count",
        "pytest_call_nodeid_root",
        "policy_namespace_unavailability",
        "live_z3_cegar_disposition",
        "candidate_reason_interpretation",
        "infrastructure_not_proof",
        "cache_authority",
        "completion_authoritative",
        "evidence_cid",
    }
    numeric_fields = (
        "meta_denial_count",
        "open_boundary_denial_count",
        "pytest_meta_path_admission_count",
        "pytest_meta_path_call_start_validation_count",
        "pytest_meta_path_sessionfinish_validation_count",
        "pytest_meta_path_unconfigure_validation_count",
        "pytest_meta_path_return_restoration_count",
        "trusted_revalidation_scope_entry_count",
        "trusted_revalidation_scope_exit_count",
        "trusted_revalidation_permitted_z3_open_count",
        "trusted_source_revalidation_expected_event_count",
        "trusted_source_revalidation_event_count",
        "trusted_source_revalidation_scope_entry_count",
        "trusted_source_revalidation_scope_exit_count",
        "trusted_source_revalidation_confirmation_count",
        "z3_file_descriptor_count",
        "policy_denied_z3_native_mapping_count",
        "pytest_call_count",
    )
    if any(
        isinstance(value.get(field), bool) or not isinstance(value.get(field), int)
        for field in numeric_fields
    ):
        raise QualificationError("recovery Z3 import denial numeric evidence differs")
    trusted_open_count = int(
        value.get("trusted_revalidation_permitted_z3_open_count", -1)
    )
    expected_entry_count = 1 if enabled and phase == "final" else 0
    expected_exit_count = expected_entry_count
    expected_scope_completed = enabled and phase == "final"
    expected_lifecycle_disposition = (
        "candidate_and_trusted_source_completed_bootstrap_restored"
        if expected_scope_completed
        else "candidate_not_started"
        if enabled
        else "not_applicable"
    )
    expected_modules_absent = task_id not in _RECOVERY_Z3_REQUIRED_TASKS
    expected_loader_executed = task_id in _RECOVERY_Z3_REQUIRED_TASKS
    if (
        set(value) != expected_fields
        or value.get("schema") != "lgcvf-recovery-z3-import-denial-evidence@3"
        or value.get("phase") != phase
        or value.get("task_id") != task_id
        or value.get("suite_id") != suite_id
        or value.get("policy_cid") != expected_policy["policy_cid"]
        or value.get("policy_disposition") != expected_policy["disposition"]
        or value.get("top_level_import_audit_claimed") is not False
        or value.get("irreversible_audit_open_boundary_installed") is not enabled
        or value.get("meta_path_guard_state")
        != ("active_exact" if enabled else "not_applicable")
        or value.get("owner_phase")
        != (
            "candidate_execution"
            if enabled and phase == "prepared"
            else "post_candidate_revalidation"
            if enabled
            else "not_applicable"
        )
        or value.get("process_exit_removal_boundary") is not enabled
        or value.get("pytest_meta_path_lifecycle_disposition")
        != expected_lifecycle_disposition
        or value.get("pytest_meta_path_admission_count")
        != (1 if expected_scope_completed else 0)
        or value.get("pytest_meta_path_call_start_validation_count")
        != (len(expected_calls) if expected_scope_completed else 0)
        or value.get("pytest_meta_path_sessionfinish_validation_count")
        != (1 if expected_scope_completed else 0)
        or value.get("pytest_meta_path_unconfigure_validation_count")
        != (1 if expected_scope_completed else 0)
        or value.get("pytest_meta_path_return_restoration_count")
        != (1 if expected_scope_completed else 0)
        or value.get("pytest_meta_path_candidate_tuple_validated")
        is not expected_scope_completed
        or value.get("pytest_meta_path_bootstrap_tuple_restored") is not enabled
        or value.get("trusted_revalidation_owner_thread_only") is not enabled
        or value.get("trusted_revalidation_scope_closed") is not True
        or value.get("trusted_source_revalidation_disposition")
        != expected_policy["trusted_source_revalidation_disposition"]
        or value.get("trusted_source_revalidation_source_projection_root")
        != source_projection_root
        or not _is_canonical_content_cid(source_projection_root)
        or value.get("trusted_source_revalidation_expected_event_count")
        != expected_policy["expected_trusted_source_event_count"]
        or value.get("trusted_source_revalidation_expected_event_root")
        != expected_policy["expected_trusted_source_event_root"]
        or value.get("ordered_trusted_source_revalidation_events")
        != expected_trusted_source_events
        or value.get("trusted_source_revalidation_event_count")
        != len(expected_trusted_source_events)
        or value.get("trusted_source_revalidation_event_root")
        != content_identity(expected_trusted_source_events)
        or value.get("trusted_source_revalidation_scope_entry_count")
        != expected_entry_count
        or value.get("trusted_source_revalidation_scope_exit_count")
        != expected_exit_count
        or value.get("trusted_source_revalidation_scope_completed")
        is not expected_scope_completed
        or value.get("trusted_source_revalidation_pending_empty") is not True
        or value.get("trusted_source_revalidation_confirmation_count")
        != len(expected_trusted_source_events)
        or value.get("trusted_source_revalidation_owner_thread_only")
        is not enabled
        or value.get(
            "trusted_source_revalidation_caller_code_identity_exact"
        ) is not (enabled and phase == "final")
        or value.get(
            "trusted_source_revalidation_descriptor_identity_validated"
        ) is not (enabled and phase == "final")
        or value.get("trusted_source_revalidation_global_z3_exemption")
        is not False
        or value.get("trusted_source_revalidation_audit_dirfd_observed")
        is not False
        or value.get("trusted_source_revalidation_telemetry_authoritative")
        is not enabled
        or value.get("trusted_source_revalidation_telemetry_reconstructed")
        is not False
        or value.get("ordered_meta_denials") != expected_meta
        or value.get("meta_denial_count") != len(expected_meta)
        or value.get("meta_denial_root") != content_identity(expected_meta)
        or value.get("ordered_open_boundary_denials") != []
        or value.get("open_boundary_denial_count") != 0
        or value.get("open_boundary_denial_root") != content_identity([])
        or value.get("trusted_revalidation_scope_entry_count")
        != expected_entry_count
        or value.get("trusted_revalidation_scope_exit_count")
        != expected_exit_count
        or value.get("trusted_revalidation_scope_completed")
        is not expected_scope_completed
        or (enabled and phase == "final" and not 1 <= trusted_open_count <= 4096)
        or ((not enabled or phase == "prepared") and trusted_open_count != 0)
        or value.get("trusted_revalidation_telemetry_authoritative") is not False
        or value.get("trusted_revalidation_telemetry_reconstructed") is not False
        or value.get("z3_modules_absent") is not expected_modules_absent
        or value.get("z3_file_descriptor_count") != 0
        or value.get("z3_file_descriptor_root") != content_identity([])
        or value.get("policy_denied_z3_native_mapping_count") != 0
        or value.get("policy_denied_z3_native_mapping_root")
        != content_identity([])
        or value.get("z3_loader_executed") is not expected_loader_executed
        or value.get("pytest_call_count") != len(expected_calls)
        or value.get("pytest_call_nodeid_root") != content_identity(expected_calls)
        or value.get("policy_namespace_unavailability")
        is not expected_policy["policy_namespace_unavailability"]
        or value.get("live_z3_cegar_disposition")
        != expected_policy["live_z3_cegar_disposition"]
        or value.get("candidate_reason_interpretation")
        != expected_policy["candidate_reason_interpretation"]
        or value.get("infrastructure_not_proof") is not True
        or value.get("cache_authority") is not False
        or value.get("completion_authoritative") is not False
        or value.get("evidence_cid") != content_identity(body)
    ):
        raise QualificationError("recovery Z3 import denial evidence differs")
    return dict(value)


def _public_z3_import_denial_commitments(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "z3_import_denial_evidence_cid": evidence["evidence_cid"],
        "z3_import_policy_cid": evidence["policy_cid"],
        "z3_import_policy_disposition": evidence["policy_disposition"],
        "z3_meta_denial_count": evidence["meta_denial_count"],
        "z3_meta_denial_root": evidence["meta_denial_root"],
        "z3_open_boundary_denial_count": evidence[
            "open_boundary_denial_count"
        ],
        "z3_open_boundary_denial_root": evidence["open_boundary_denial_root"],
        "z3_modules_absent": evidence["z3_modules_absent"],
        "z3_file_descriptor_count": evidence["z3_file_descriptor_count"],
        "z3_file_descriptor_root": evidence["z3_file_descriptor_root"],
        "z3_policy_denied_native_mapping_count": evidence[
            "policy_denied_z3_native_mapping_count"
        ],
        "z3_policy_denied_native_mapping_root": evidence[
            "policy_denied_z3_native_mapping_root"
        ],
        "z3_owner_phase": evidence["owner_phase"],
        "z3_trusted_revalidation_scope_completed": evidence[
            "trusted_revalidation_scope_completed"
        ],
        "z3_trusted_revalidation_owner_thread_only": evidence[
            "trusted_revalidation_owner_thread_only"
        ],
        "z3_trusted_revalidation_permitted_open_count": evidence[
            "trusted_revalidation_permitted_z3_open_count"
        ],
        "z3_trusted_revalidation_telemetry_authoritative": evidence[
            "trusted_revalidation_telemetry_authoritative"
        ],
        "z3_trusted_revalidation_telemetry_reconstructed": evidence[
            "trusted_revalidation_telemetry_reconstructed"
        ],
        "z3_trusted_source_revalidation_disposition": evidence[
            "trusted_source_revalidation_disposition"
        ],
        "z3_trusted_source_revalidation_source_projection_root": evidence[
            "trusted_source_revalidation_source_projection_root"
        ],
        "z3_trusted_source_revalidation_expected_event_count": evidence[
            "trusted_source_revalidation_expected_event_count"
        ],
        "z3_trusted_source_revalidation_expected_event_root": evidence[
            "trusted_source_revalidation_expected_event_root"
        ],
        "z3_trusted_source_revalidation_event_count": evidence[
            "trusted_source_revalidation_event_count"
        ],
        "z3_trusted_source_revalidation_event_root": evidence[
            "trusted_source_revalidation_event_root"
        ],
        "z3_trusted_source_revalidation_scope_entry_count": evidence[
            "trusted_source_revalidation_scope_entry_count"
        ],
        "z3_trusted_source_revalidation_scope_exit_count": evidence[
            "trusted_source_revalidation_scope_exit_count"
        ],
        "z3_trusted_source_revalidation_scope_completed": evidence[
            "trusted_source_revalidation_scope_completed"
        ],
        "z3_trusted_source_revalidation_pending_empty": evidence[
            "trusted_source_revalidation_pending_empty"
        ],
        "z3_trusted_source_revalidation_confirmation_count": evidence[
            "trusted_source_revalidation_confirmation_count"
        ],
        "z3_trusted_source_revalidation_owner_thread_only": evidence[
            "trusted_source_revalidation_owner_thread_only"
        ],
        "z3_trusted_source_revalidation_caller_code_identity_exact": evidence[
            "trusted_source_revalidation_caller_code_identity_exact"
        ],
        "z3_trusted_source_revalidation_descriptor_identity_validated": evidence[
            "trusted_source_revalidation_descriptor_identity_validated"
        ],
        "z3_trusted_source_revalidation_global_z3_exemption": evidence[
            "trusted_source_revalidation_global_z3_exemption"
        ],
        "z3_trusted_source_revalidation_audit_dirfd_observed": evidence[
            "trusted_source_revalidation_audit_dirfd_observed"
        ],
        "z3_trusted_source_revalidation_telemetry_authoritative": evidence[
            "trusted_source_revalidation_telemetry_authoritative"
        ],
        "z3_trusted_source_revalidation_telemetry_reconstructed": evidence[
            "trusted_source_revalidation_telemetry_reconstructed"
        ],
        "z3_policy_namespace_unavailability": evidence[
            "policy_namespace_unavailability"
        ],
        "z3_live_cegar_disposition": evidence["live_z3_cegar_disposition"],
        "z3_candidate_reason_interpretation": evidence[
            "candidate_reason_interpretation"
        ],
    }


_RECOVERY_PUBLIC_Z3_IMPORT_DENIAL_FIELDS: Final[frozenset[str]] = frozenset(
    _public_z3_import_denial_commitments(
        {
            "evidence_cid": "",
            "policy_cid": "",
            "policy_disposition": "",
            "meta_denial_count": 0,
            "meta_denial_root": "",
            "open_boundary_denial_count": 0,
            "open_boundary_denial_root": "",
            "z3_modules_absent": True,
            "z3_file_descriptor_count": 0,
            "z3_file_descriptor_root": "",
            "policy_denied_z3_native_mapping_count": 0,
            "policy_denied_z3_native_mapping_root": "",
            "owner_phase": "",
            "trusted_revalidation_scope_completed": False,
            "trusted_revalidation_owner_thread_only": False,
            "trusted_revalidation_permitted_z3_open_count": 0,
            "trusted_revalidation_telemetry_authoritative": False,
            "trusted_revalidation_telemetry_reconstructed": False,
            "trusted_source_revalidation_disposition": "",
            "trusted_source_revalidation_source_projection_root": "",
            "trusted_source_revalidation_expected_event_count": 0,
            "trusted_source_revalidation_expected_event_root": "",
            "trusted_source_revalidation_event_count": 0,
            "trusted_source_revalidation_event_root": "",
            "trusted_source_revalidation_scope_entry_count": 0,
            "trusted_source_revalidation_scope_exit_count": 0,
            "trusted_source_revalidation_scope_completed": False,
            "trusted_source_revalidation_pending_empty": True,
            "trusted_source_revalidation_confirmation_count": 0,
            "trusted_source_revalidation_owner_thread_only": False,
            "trusted_source_revalidation_caller_code_identity_exact": False,
            "trusted_source_revalidation_descriptor_identity_validated": False,
            "trusted_source_revalidation_global_z3_exemption": False,
            "trusted_source_revalidation_audit_dirfd_observed": False,
            "trusted_source_revalidation_telemetry_authoritative": False,
            "trusted_source_revalidation_telemetry_reconstructed": False,
            "policy_namespace_unavailability": False,
            "live_z3_cegar_disposition": "",
            "candidate_reason_interpretation": "",
        }
    )
)


def _validate_public_z3_import_denial_commitments(
    value: Mapping[str, Any],
    *,
    phase: str,
    task_id: str,
    suite_id: str,
    source_projection_root: str,
) -> dict[str, Any]:
    policy = _recovery_z3_import_denial_policy(
        task_id=task_id,
        suite_id=suite_id,
    )
    enabled = task_id in _RECOVERY_Z3_IMPORT_DENIED_TASKS
    expected_meta = (
        [] if phase == "prepared" else list(policy["expected_meta_denials"])
    )
    observed = {key: value.get(key) for key in _RECOVERY_PUBLIC_Z3_IMPORT_DENIAL_FIELDS}
    numeric_fields = (
        "z3_meta_denial_count",
        "z3_open_boundary_denial_count",
        "z3_file_descriptor_count",
        "z3_policy_denied_native_mapping_count",
        "z3_trusted_revalidation_permitted_open_count",
        "z3_trusted_source_revalidation_expected_event_count",
        "z3_trusted_source_revalidation_event_count",
        "z3_trusted_source_revalidation_scope_entry_count",
        "z3_trusted_source_revalidation_scope_exit_count",
        "z3_trusted_source_revalidation_confirmation_count",
    )
    if any(
        isinstance(observed.get(field), bool)
        or not isinstance(observed.get(field), int)
        for field in numeric_fields
    ):
        raise QualificationError("public Z3 denial numeric evidence differs")
    permitted_open_count = int(
        observed.get("z3_trusted_revalidation_permitted_open_count", -1)
    )
    expected_modules_absent = task_id not in _RECOVERY_Z3_REQUIRED_TASKS
    expected_source_events = (
        [
            {**dict(item), "source_projection_root": source_projection_root}
            for item in policy["expected_trusted_source_events"]
        ]
        if enabled and phase == "final"
        else []
    )
    if (
        not _is_canonical_content_cid(
            observed.get("z3_import_denial_evidence_cid")
        )
        or observed.get("z3_import_policy_cid") != policy["policy_cid"]
        or observed.get("z3_import_policy_disposition") != policy["disposition"]
        or observed.get("z3_meta_denial_count") != len(expected_meta)
        or observed.get("z3_meta_denial_root") != content_identity(expected_meta)
        or observed.get("z3_open_boundary_denial_count") != 0
        or observed.get("z3_open_boundary_denial_root") != content_identity([])
        or observed.get("z3_modules_absent") is not expected_modules_absent
        or observed.get("z3_file_descriptor_count") != 0
        or observed.get("z3_file_descriptor_root") != content_identity([])
        or observed.get("z3_policy_denied_native_mapping_count") != 0
        or observed.get("z3_policy_denied_native_mapping_root")
        != content_identity([])
        or observed.get("z3_owner_phase")
        != (
            "candidate_execution"
            if enabled and phase == "prepared"
            else "post_candidate_revalidation"
            if enabled
            else "not_applicable"
        )
        or observed.get("z3_trusted_revalidation_scope_completed")
        is not (enabled and phase == "final")
        or observed.get("z3_trusted_revalidation_owner_thread_only") is not enabled
        or (
            enabled
            and phase == "final"
            and not 1 <= permitted_open_count <= 4096
        )
        or (
            (not enabled or phase == "prepared")
            and permitted_open_count != 0
        )
        or observed.get("z3_trusted_revalidation_telemetry_authoritative")
        is not False
        or observed.get("z3_trusted_revalidation_telemetry_reconstructed")
        is not False
        or observed.get("z3_trusted_source_revalidation_disposition")
        != policy["trusted_source_revalidation_disposition"]
        or observed.get(
            "z3_trusted_source_revalidation_source_projection_root"
        ) != source_projection_root
        or not _is_canonical_content_cid(source_projection_root)
        or observed.get(
            "z3_trusted_source_revalidation_expected_event_count"
        ) != policy["expected_trusted_source_event_count"]
        or observed.get(
            "z3_trusted_source_revalidation_expected_event_root"
        ) != policy["expected_trusted_source_event_root"]
        or observed.get("z3_trusted_source_revalidation_event_count")
        != len(expected_source_events)
        or observed.get("z3_trusted_source_revalidation_event_root")
        != content_identity(expected_source_events)
        or observed.get(
            "z3_trusted_source_revalidation_scope_entry_count"
        ) != (1 if enabled and phase == "final" else 0)
        or observed.get(
            "z3_trusted_source_revalidation_scope_exit_count"
        ) != (1 if enabled and phase == "final" else 0)
        or observed.get("z3_trusted_source_revalidation_scope_completed")
        is not (enabled and phase == "final")
        or observed.get("z3_trusted_source_revalidation_pending_empty")
        is not True
        or observed.get(
            "z3_trusted_source_revalidation_confirmation_count"
        ) != len(expected_source_events)
        or observed.get("z3_trusted_source_revalidation_owner_thread_only")
        is not enabled
        or observed.get(
            "z3_trusted_source_revalidation_caller_code_identity_exact"
        ) is not (enabled and phase == "final")
        or observed.get(
            "z3_trusted_source_revalidation_descriptor_identity_validated"
        ) is not (enabled and phase == "final")
        or observed.get(
            "z3_trusted_source_revalidation_global_z3_exemption"
        ) is not False
        or observed.get(
            "z3_trusted_source_revalidation_audit_dirfd_observed"
        ) is not False
        or observed.get(
            "z3_trusted_source_revalidation_telemetry_authoritative"
        ) is not enabled
        or observed.get(
            "z3_trusted_source_revalidation_telemetry_reconstructed"
        ) is not False
        or observed.get("z3_policy_namespace_unavailability")
        is not policy["policy_namespace_unavailability"]
        or observed.get("z3_live_cegar_disposition")
        != policy["live_z3_cegar_disposition"]
        or observed.get("z3_candidate_reason_interpretation")
        != policy["candidate_reason_interpretation"]
    ):
        raise QualificationError("public Z3 import denial evidence differs")
    return observed


def _worker_live_attestation(
    *,
    phase: str,
    launch_nonce: str,
    suite_id: str,
    runtime_cid: str,
    z3_required: bool,
    suite_task_observation: Mapping[str, Any],
    suite_task_terminal_records: Sequence[Mapping[str, Any]],
    native_guard_evidence: Mapping[str, Any],
    z3_import_denial_evidence: Mapping[str, Any],
    executable_mappings: Sequence[Mapping[str, Any]],
    thread_population: Sequence[Mapping[str, Any]],
    kernel_task_ids: Sequence[int],
    kernel_task_records: Sequence[Mapping[str, Any]],
    child_process_ids: Sequence[int],
    process_group_population: Sequence[Mapping[str, int]],
) -> dict[str, Any]:
    if phase not in {"prepared", "final"}:
        raise QualificationError("worker attestation phase differs")
    if re.fullmatch(r"[0-9a-f]{64}", launch_nonce) is None:
        raise QualificationError("worker attestation nonce differs")
    normalized_mappings = _normalized_executable_mapping_signatures(
        executable_mappings
    )
    task_policy_matrix = _recovery_suite_task_policy_matrix()
    recovery = _RECOVERY_BY_SUITE_ID.get(suite_id)
    if recovery is None:
        raise QualificationError("worker attestation recovery identity is absent")
    compact_task_receipt = _recovery_suite_task_receipt_from_live_observation(
        suite_task_observation,
        task_id=recovery.task_id,
        suite_id=suite_id,
        runtime_cid=runtime_cid,
        source_projection_root=str(
            suite_task_observation.get("source_projection_root") or ""
        ),
    )
    z3_import_denial_evidence = _validate_recovery_z3_import_denial_evidence(
        z3_import_denial_evidence,
        phase=phase,
        task_id=recovery.task_id,
        suite_id=suite_id,
        source_projection_root=str(
            suite_task_observation.get("source_projection_root") or ""
        ),
    )
    terminal_records = [dict(item) for item in suite_task_terminal_records]
    current_records = [dict(item) for item in kernel_task_records]
    if current_records != terminal_records:
        raise QualificationError(
            "worker task identity changed after its fixed point"
        )
    body: dict[str, Any] = {
        "schema": "lgcvf-recovery-worker-live-attestation@4",
        "phase": phase,
        "launch_nonce": launch_nonce,
        "suite_id": suite_id,
        "process_id": os.getpid(),
        "process_start_time_ticks": _recovery_process_start_time_ticks(),
        "runtime_cid": runtime_cid,
        "suite_native_policy": _recovery_suite_native_policy(
            z3_required=z3_required
        ),
        "suite_task_policy_matrix_cid": task_policy_matrix["matrix_cid"],
        "suite_task_profile_cid": suite_task_observation.get("profile_cid"),
        "suite_task_observation": dict(suite_task_observation),
        "full_live_observation_cid": suite_task_observation.get("observation_cid"),
        "suite_task_receipt_cid": compact_task_receipt.get("receipt_cid"),
        "suite_task_terminal_records": terminal_records,
        "suite_task_terminal_record_root": content_identity(
            terminal_records
        ),
        "native_guard_cid": native_guard_evidence.get("guard_cid"),
        "z3_import_denial_evidence": dict(z3_import_denial_evidence),
        "z3_import_denial_evidence_cid": z3_import_denial_evidence.get(
            "evidence_cid"
        ),
        "executable_mapping_count": len(executable_mappings),
        "executable_mapping_root": _native_executable_mapping_root(
            executable_mappings
        ),
        "normalized_executable_mapping_root": content_identity(
            normalized_mappings
        ),
        "writable_executable_mappings": (
            _writable_executable_mapping_signatures(executable_mappings)
        ),
        "thread_population": [dict(item) for item in thread_population],
        "thread_population_root": content_identity(
            [dict(item) for item in thread_population]
        ),
        "kernel_task_ids": list(kernel_task_ids),
        "kernel_task_root": content_identity(list(kernel_task_ids)),
        "kernel_task_records": [dict(item) for item in kernel_task_records],
        "kernel_task_record_root": content_identity(
            [dict(item) for item in kernel_task_records]
        ),
        "child_process_ids": list(child_process_ids),
        "child_process_root": content_identity(list(child_process_ids)),
        "process_group_population": [dict(item) for item in process_group_population],
        "process_group_root": content_identity(
            [dict(item) for item in process_group_population]
        ),
    }
    body["attestation_cid"] = content_identity(body)
    return body


def _public_worker_self_observation(
    attestation: Mapping[str, Any],
) -> dict[str, Any]:
    suite_id = attestation.get("suite_id")
    recovery = (
        _RECOVERY_BY_SUITE_ID.get(suite_id)
        if isinstance(suite_id, str)
        else None
    )
    if recovery is None:
        raise QualificationError("worker normalized self suite differs")
    phase = attestation.get("phase")
    z3_import_evidence = _validate_recovery_z3_import_denial_evidence(
        attestation.get("z3_import_denial_evidence"),
        phase=str(phase or ""),
        task_id=recovery.task_id,
        suite_id=suite_id,
        source_projection_root=str(
            (
                attestation.get("suite_task_observation")
                if isinstance(
                    attestation.get("suite_task_observation"), Mapping
                )
                else {}
            ).get("source_projection_root")
            or ""
        ),
    )
    value: dict[str, Any] = {
        "schema": "lgcvf-recovery-worker-normalized-self-observation@4",
        "phase": attestation.get("phase"),
        "suite_id": attestation.get("suite_id"),
        "runtime_cid": attestation.get("runtime_cid"),
        "suite_native_policy": attestation.get("suite_native_policy"),
        "suite_task_policy_matrix_cid": attestation.get(
            "suite_task_policy_matrix_cid"
        ),
        "suite_task_profile_cid": attestation.get("suite_task_profile_cid"),
        "full_live_observation_cid": attestation.get(
            "full_live_observation_cid"
        ),
        "suite_task_receipt_cid": attestation.get(
            "suite_task_receipt_cid"
        ),
        "fixed_point_to_prepared_identity_restored": True,
        "native_guard_cid": attestation.get("native_guard_cid"),
        **_public_z3_import_denial_commitments(z3_import_evidence),
        "executable_mapping_count": attestation.get("executable_mapping_count"),
        "normalized_executable_mapping_root": attestation.get(
            "normalized_executable_mapping_root"
        ),
        "writable_executable_mappings": attestation.get(
            "writable_executable_mappings"
        ),
        "thread_count": len(attestation.get("thread_population", [])),
        "kernel_task_count": len(attestation.get("kernel_task_ids", [])),
        "normalized_kernel_task_root": content_identity(
            _normalized_kernel_task_role_evidence(
                attestation.get("kernel_task_records", []),
                task_id=recovery.task_id,
            )
        ),
        "child_process_count": len(attestation.get("child_process_ids", [])),
        "children_absent": attestation.get("child_process_ids") == [],
        "process_group_count": len(
            attestation.get("process_group_population", [])
        ),
        "process_group_leader_only": len(
            attestation.get("process_group_population", [])
        )
        == 1,
    }
    value["self_observation_cid"] = content_identity(value)
    return value


def _validate_worker_parent_ack(
    ack: Mapping[str, Any],
    *,
    attestation: Mapping[str, Any],
) -> dict[str, Any]:
    suite_id = attestation.get("suite_id")
    if not isinstance(suite_id, str):
        raise QualificationError("worker acknowledgement suite differs")
    recovery = _RECOVERY_BY_SUITE_ID.get(suite_id)
    if recovery is None:
        raise QualificationError(f"{suite_id}: recovery task identity is absent")
    suite_task_observation = attestation.get("suite_task_observation")
    if not isinstance(suite_task_observation, Mapping):
        raise QualificationError("worker task observation is absent")
    validated_task_observation = _validate_recovery_suite_task_observation(
        suite_task_observation,
        task_id=recovery.task_id,
        suite_id=suite_id,
        runtime_cid=str(attestation.get("runtime_cid") or ""),
        source_projection_root=str(
            suite_task_observation.get("source_projection_root") or ""
        ),
    )
    compact_task_receipt = _recovery_suite_task_receipt_from_live_observation(
        validated_task_observation,
        task_id=recovery.task_id,
        suite_id=suite_id,
        runtime_cid=str(attestation.get("runtime_cid") or ""),
        source_projection_root=str(
            validated_task_observation.get("source_projection_root") or ""
        ),
    )
    z3_import_denial_evidence = _validate_recovery_z3_import_denial_evidence(
        attestation.get("z3_import_denial_evidence"),
        phase=str(attestation.get("phase") or ""),
        task_id=recovery.task_id,
        suite_id=suite_id,
        source_projection_root=str(
            validated_task_observation.get("source_projection_root") or ""
        ),
    )
    public_z3_commitments = _public_z3_import_denial_commitments(
        z3_import_denial_evidence
    )
    body = {key: item for key, item in ack.items() if key != "ack_cid"}
    parent_observation = ack.get("parent_observation")
    if not isinstance(parent_observation, Mapping):
        raise QualificationError("worker parent observation is absent")
    parent_body = {
        key: item
        for key, item in parent_observation.items()
        if key != "parent_observation_cid"
    }
    if (
        set(ack)
        != {
            "schema",
            "phase",
            "launch_nonce",
            "suite_id",
            "process_id",
            "process_start_time_ticks",
            "child_attestation_cid",
            "observed_executable_mapping_root",
            "observed_kernel_task_root",
            "observed_kernel_task_record_root",
            "suite_task_terminal_record_root",
            "suite_task_policy_matrix_cid",
            "suite_task_profile_cid",
            "full_live_observation_cid",
            "suite_task_receipt_cid",
            "fixed_point_to_prepared_identity_restored",
            "z3_import_denial_evidence_cid",
            "parent_z3_file_descriptor_count",
            "parent_z3_file_descriptor_root",
            "parent_policy_denied_z3_native_mapping_count",
            "parent_policy_denied_z3_native_mapping_root",
            "observed_process_group_root",
            "parent_observation",
            "parent_observation_cid",
            "admitted",
            "ack_cid",
        }
        or ack.get("schema") != "lgcvf-recovery-worker-parent-ack@4"
        or ack.get("phase") != attestation.get("phase")
        or ack.get("launch_nonce") != attestation.get("launch_nonce")
        or ack.get("suite_id") != attestation.get("suite_id")
        or ack.get("process_id") != attestation.get("process_id")
        or ack.get("process_start_time_ticks")
        != attestation.get("process_start_time_ticks")
        or ack.get("child_attestation_cid")
        != attestation.get("attestation_cid")
        or ack.get("observed_executable_mapping_root")
        != attestation.get("executable_mapping_root")
        or ack.get("observed_kernel_task_root")
        != attestation.get("kernel_task_root")
        or ack.get("observed_kernel_task_record_root")
        != attestation.get("kernel_task_record_root")
        or ack.get("suite_task_terminal_record_root")
        != attestation.get("suite_task_terminal_record_root")
        or ack.get("suite_task_policy_matrix_cid")
        != attestation.get("suite_task_policy_matrix_cid")
        or ack.get("suite_task_profile_cid")
        != attestation.get("suite_task_profile_cid")
        or ack.get("full_live_observation_cid")
        != attestation.get("full_live_observation_cid")
        or ack.get("full_live_observation_cid")
        != validated_task_observation["observation_cid"]
        or ack.get("suite_task_receipt_cid")
        != attestation.get("suite_task_receipt_cid")
        or ack.get("suite_task_receipt_cid")
        != compact_task_receipt["receipt_cid"]
        or ack.get("fixed_point_to_prepared_identity_restored") is not True
        or ack.get("fixed_point_to_prepared_identity_restored")
        != parent_observation.get(
            "fixed_point_to_prepared_identity_restored"
        )
        or ack.get("observed_process_group_root")
        != attestation.get("process_group_root")
        or ack.get("z3_import_denial_evidence_cid")
        != z3_import_denial_evidence["evidence_cid"]
        or any(
            isinstance(ack.get(field), bool)
            or not isinstance(ack.get(field), int)
            for field in (
                "parent_z3_file_descriptor_count",
                "parent_policy_denied_z3_native_mapping_count",
            )
        )
        or ack.get("parent_z3_file_descriptor_count") != 0
        or ack.get("parent_z3_file_descriptor_count")
        != parent_observation.get("parent_z3_file_descriptor_count")
        or ack.get("parent_z3_file_descriptor_root") != content_identity([])
        or ack.get("parent_z3_file_descriptor_root")
        != parent_observation.get("parent_z3_file_descriptor_root")
        or ack.get("parent_policy_denied_z3_native_mapping_count") != 0
        or ack.get("parent_policy_denied_z3_native_mapping_count")
        != parent_observation.get(
            "parent_policy_denied_z3_native_mapping_count"
        )
        or ack.get("parent_policy_denied_z3_native_mapping_root")
        != content_identity([])
        or ack.get("parent_policy_denied_z3_native_mapping_root")
        != parent_observation.get(
            "parent_policy_denied_z3_native_mapping_root"
        )
        or ack.get("parent_observation_cid")
        != parent_observation.get("parent_observation_cid")
        or parent_observation.get("parent_observation_cid")
        != content_identity(parent_body)
        or ack.get("admitted") is not True
        or ack.get("ack_cid") != content_identity(body)
    ):
        raise QualificationError("worker parent acknowledgement differs")
    expected_parent_fields = {
        "schema",
        "phase",
        "suite_id",
        "runtime_cid",
        "suite_native_policy",
        "suite_task_policy_matrix_cid",
        "suite_task_profile_cid",
        "full_live_observation_cid",
        "suite_task_receipt_cid",
        "fixed_point_to_prepared_identity_restored",
        "native_guard_cid",
        *_RECOVERY_PUBLIC_Z3_IMPORT_DENIAL_FIELDS,
        "parent_z3_file_descriptor_count",
        "parent_z3_file_descriptor_root",
        "parent_policy_denied_z3_native_mapping_count",
        "parent_policy_denied_z3_native_mapping_root",
        "pidfd_bound",
        "process_live",
        "process_start_time_matched",
        "executable_mapping_count",
        "normalized_executable_mapping_root",
        "writable_executable_mappings",
        "kernel_task_count",
        "normalized_kernel_task_root",
        "task_directory_identity_held",
        "task_directory_identity_restored",
        "child_process_count",
        "children_absent",
        "process_group_count",
        "process_group_leader_only",
        "prepared_task_population_restored",
        "prepared_writable_executable_mapping_restored",
        "prepared_mappings_retained",
        "prepared_process_group_restored",
        "controller_zero_wx_observation",
        "controller_zero_wx_observation_cid",
        "parent_observation_cid",
    }
    expected_task_profile = _recovery_suite_task_policy(
        task_id=recovery.task_id,
        suite_id=suite_id,
    )
    if (
        set(parent_observation) != expected_parent_fields
        or parent_observation.get("schema")
        != "lgcvf-recovery-worker-parent-observation@4"
        or parent_observation.get("phase") != attestation.get("phase")
        or parent_observation.get("suite_id") != attestation.get("suite_id")
        or parent_observation.get("runtime_cid") != attestation.get("runtime_cid")
        or parent_observation.get("suite_native_policy")
        != attestation.get("suite_native_policy")
        or parent_observation.get("suite_task_policy_matrix_cid")
        != _recovery_suite_task_policy_matrix()["matrix_cid"]
        or parent_observation.get("suite_task_policy_matrix_cid")
        != attestation.get("suite_task_policy_matrix_cid")
        or parent_observation.get("suite_task_profile_cid")
        != attestation.get("suite_task_profile_cid")
        or parent_observation.get("suite_task_profile_cid")
        != expected_task_profile["profile_cid"]
        or parent_observation.get("full_live_observation_cid")
        != attestation.get("full_live_observation_cid")
        or parent_observation.get("full_live_observation_cid")
        != validated_task_observation["observation_cid"]
        or parent_observation.get("suite_task_receipt_cid")
        != attestation.get("suite_task_receipt_cid")
        or parent_observation.get("suite_task_receipt_cid")
        != compact_task_receipt["receipt_cid"]
        or parent_observation.get("fixed_point_to_prepared_identity_restored")
        is not True
        or parent_observation.get("native_guard_cid")
        != attestation.get("native_guard_cid")
        or any(
            parent_observation.get(key) != item
            for key, item in public_z3_commitments.items()
        )
        or any(
            isinstance(parent_observation.get(field), bool)
            or not isinstance(parent_observation.get(field), int)
            for field in (
                "parent_z3_file_descriptor_count",
                "parent_policy_denied_z3_native_mapping_count",
            )
        )
        or parent_observation.get("parent_z3_file_descriptor_count") != 0
        or parent_observation.get("parent_z3_file_descriptor_root")
        != content_identity([])
        or parent_observation.get(
            "parent_policy_denied_z3_native_mapping_count"
        )
        != 0
        or parent_observation.get(
            "parent_policy_denied_z3_native_mapping_root"
        )
        != content_identity([])
        or parent_observation.get("pidfd_bound") is not True
        or parent_observation.get("process_live") is not True
        or parent_observation.get("process_start_time_matched") is not True
        or parent_observation.get("executable_mapping_count")
        != attestation.get("executable_mapping_count")
        or parent_observation.get("normalized_executable_mapping_root")
        != attestation.get("normalized_executable_mapping_root")
        or parent_observation.get("writable_executable_mappings")
        != attestation.get("writable_executable_mappings")
        or parent_observation.get("kernel_task_count")
        != len(attestation.get("kernel_task_ids", []))
        or parent_observation.get("normalized_kernel_task_root")
        != content_identity(
            _normalized_kernel_task_role_evidence(
                attestation.get("kernel_task_records", []),
                task_id=recovery.task_id,
            )
        )
        or parent_observation.get("task_directory_identity_held") is not True
        or parent_observation.get("task_directory_identity_restored") is not True
        or parent_observation.get("child_process_count") != 0
        or parent_observation.get("children_absent") is not True
        or parent_observation.get("process_group_count") != 1
        or parent_observation.get("process_group_leader_only") is not True
        or parent_observation.get("prepared_task_population_restored") is not True
        or parent_observation.get("prepared_writable_executable_mapping_restored")
        is not True
        or parent_observation.get("prepared_mappings_retained") is not True
        or parent_observation.get("prepared_process_group_restored") is not True
        or not isinstance(
            parent_observation.get("controller_zero_wx_observation"), Mapping
        )
        or parent_observation.get("controller_zero_wx_observation_cid")
        != parent_observation.get("controller_zero_wx_observation", {}).get(
            "observation_cid"
        )
        or parent_observation.get("controller_zero_wx_observation", {}).get(
            "controller_wx_mapping_count"
        )
        != 0
    ):
        raise QualificationError("worker parent live observation differs")
    return dict(parent_observation)


def _public_attestation_phase_evidence(
    attestation: Mapping[str, Any],
    parent_observation: Mapping[str, Any],
) -> dict[str, Any]:
    self_observation = _public_worker_self_observation(attestation)
    z3_import_denial_evidence = attestation.get("z3_import_denial_evidence")
    if not isinstance(z3_import_denial_evidence, Mapping):
        raise QualificationError("worker Z3 import denial phase evidence is absent")
    value: dict[str, Any] = {
        "schema": "lgcvf-recovery-worker-attestation-phase@5",
        "phase": attestation.get("phase"),
        "self_observation": self_observation,
        "parent_observation": dict(parent_observation),
        "parent_observation_cid": parent_observation.get(
            "parent_observation_cid"
        ),
        "z3_import_denial_evidence": dict(z3_import_denial_evidence),
        "z3_import_denial_evidence_cid": z3_import_denial_evidence.get(
            "evidence_cid"
        ),
        "parent_admitted": True,
        "post_ack_state_rechecked": True,
    }
    value["phase_cid"] = content_identity(value)
    return value


def _worker_attestation_phase(
    *,
    phase: str,
    attestation_descriptor: int,
    acknowledgement_descriptor: int,
    launch_nonce: str,
    suite_id: str,
    runtime_cid: str,
    z3_required: bool,
    suite_task_observation: Mapping[str, Any],
    suite_task_terminal_records: Sequence[Mapping[str, Any]],
    native_load_guard: _RecoveryNativeLoadGuard,
    native_guard_evidence: Mapping[str, Any],
    z3_import_denial_guard: _RecoveryZ3ImportDenialGuard,
    z3_import_denial_evidence: Mapping[str, Any],
    native_maps_before_solver: Sequence[Mapping[str, Any]],
    native_threads_before_pytest: Sequence[Mapping[str, Any]],
    native_kernel_tasks_before_pytest: Sequence[int],
    native_children_before_pytest: Sequence[int],
) -> dict[str, Any]:
    maps = _native_executable_mappings()
    threads = _recovery_live_thread_population()
    tasks = _recovery_kernel_task_population()
    task_records = _recovery_kernel_task_records()
    expected_terminal_records = [
        dict(item) for item in suite_task_terminal_records
    ]
    if (
        [int(item["tid"]) for item in task_records] != tasks
        or task_records != expected_terminal_records
    ):
        raise QualificationError("worker kernel task records differ")
    children = _recovery_child_process_population(task_ids=tasks)
    process_group = _recovery_process_group_population()
    observed_native_guard = native_load_guard.evidence(
        before_maps=native_maps_before_solver,
        after_maps=maps,
        before_threads=native_threads_before_pytest,
        after_threads=threads,
        before_kernel_tasks=native_kernel_tasks_before_pytest,
        after_kernel_tasks=tasks,
        before_child_processes=native_children_before_pytest,
        after_child_processes=children,
        diagnostic_phase=phase,
    )
    if observed_native_guard != native_guard_evidence:
        raise QualificationError("worker native guard changed before attestation")
    recovery = _RECOVERY_BY_SUITE_ID.get(suite_id)
    if recovery is None:
        raise QualificationError("worker Z3 denial recovery identity is absent")
    observed_z3_import_denial = z3_import_denial_guard.evidence(phase=phase)
    if (
        observed_z3_import_denial != z3_import_denial_evidence
        or _validate_recovery_z3_import_denial_evidence(
            observed_z3_import_denial,
            phase=phase,
            task_id=recovery.task_id,
            suite_id=suite_id,
            source_projection_root=str(
                suite_task_observation.get("source_projection_root") or ""
            ),
        )
        != observed_z3_import_denial
    ):
        raise QualificationError("worker Z3 import denial changed before attestation")
    attestation = _worker_live_attestation(
        phase=phase,
        launch_nonce=launch_nonce,
        suite_id=suite_id,
        runtime_cid=runtime_cid,
        z3_required=z3_required,
        suite_task_observation=suite_task_observation,
        suite_task_terminal_records=expected_terminal_records,
        native_guard_evidence=observed_native_guard,
        z3_import_denial_evidence=observed_z3_import_denial,
        executable_mappings=maps,
        thread_population=threads,
        kernel_task_ids=tasks,
        kernel_task_records=task_records,
        child_process_ids=children,
        process_group_population=process_group,
    )
    _write_bounded_json_line(
        attestation_descriptor,
        attestation,
        noun=f"{phase} worker attestation",
    )
    ack = _read_bounded_json_line(
        acknowledgement_descriptor,
        noun=f"{phase} parent acknowledgement",
        timeout_seconds=_MAX_WORKER_ACK_SECONDS,
    )
    parent_observation = _validate_worker_parent_ack(
        ack,
        attestation=attestation,
    )
    after_maps = _native_executable_mappings()
    after_threads = _recovery_live_thread_population()
    after_tasks = _recovery_kernel_task_population()
    after_task_records = _recovery_kernel_task_records()
    after_children = _recovery_child_process_population(task_ids=after_tasks)
    after_process_group = _recovery_process_group_population()
    after_native_guard = native_load_guard.evidence(
        before_maps=native_maps_before_solver,
        after_maps=after_maps,
        before_threads=native_threads_before_pytest,
        after_threads=after_threads,
        before_kernel_tasks=native_kernel_tasks_before_pytest,
        after_kernel_tasks=after_tasks,
        before_child_processes=native_children_before_pytest,
        after_child_processes=after_children,
        diagnostic_phase=phase + "_post_ack",
    )
    after_z3_import_denial = z3_import_denial_guard.evidence(phase=phase)
    if (
        after_maps != maps
        or after_threads != threads
        or after_tasks != tasks
        or after_task_records != task_records
        or after_children != children
        or after_process_group != process_group
        or after_native_guard != observed_native_guard
        or after_z3_import_denial != observed_z3_import_denial
        or _recovery_process_start_time_ticks()
        != attestation.get("process_start_time_ticks")
    ):
        raise QualificationError("worker live state changed across parent acknowledgement")
    return _public_attestation_phase_evidence(attestation, parent_observation)


def _worker_attestation_barrier_evidence(
    prepared: Mapping[str, Any],
    final: Mapping[str, Any],
) -> dict[str, Any]:
    prepared_self = prepared.get("self_observation")
    final_self = final.get("self_observation")
    prepared_parent = prepared.get("parent_observation")
    final_parent = final.get("parent_observation")
    prepared_z3 = prepared.get("z3_import_denial_evidence")
    final_z3 = final.get("z3_import_denial_evidence")
    if not isinstance(prepared_self, Mapping) or not isinstance(final_self, Mapping):
        raise QualificationError("worker attestation self evidence is absent")
    if not isinstance(prepared_parent, Mapping) or not isinstance(final_parent, Mapping):
        raise QualificationError("worker attestation parent evidence is absent")
    if not isinstance(prepared_z3, Mapping) or not isinstance(final_z3, Mapping):
        raise QualificationError("worker attestation Z3 evidence is absent")
    if (
        prepared.get("phase") != "prepared"
        or final.get("phase") != "final"
        or prepared_self.get("suite_native_policy")
        != final_self.get("suite_native_policy")
        or prepared_self.get("runtime_cid") != final_self.get("runtime_cid")
        or prepared_self.get("suite_task_policy_matrix_cid")
        != final_self.get("suite_task_policy_matrix_cid")
        or prepared_self.get("suite_task_profile_cid")
        != final_self.get("suite_task_profile_cid")
        or prepared_self.get("full_live_observation_cid")
        != final_self.get("full_live_observation_cid")
        or prepared_self.get("suite_task_receipt_cid")
        != final_self.get("suite_task_receipt_cid")
        or prepared_self.get("z3_import_policy_cid")
        != final_self.get("z3_import_policy_cid")
        or prepared_parent.get("z3_import_policy_cid")
        != prepared_self.get("z3_import_policy_cid")
        or final_parent.get("z3_import_policy_cid")
        != final_self.get("z3_import_policy_cid")
        or prepared_parent.get("z3_import_denial_evidence_cid")
        != prepared_self.get("z3_import_denial_evidence_cid")
        or final_parent.get("z3_import_denial_evidence_cid")
        != final_self.get("z3_import_denial_evidence_cid")
        or prepared.get("z3_import_denial_evidence_cid")
        != prepared_self.get("z3_import_denial_evidence_cid")
        or final.get("z3_import_denial_evidence_cid")
        != final_self.get("z3_import_denial_evidence_cid")
        or prepared_z3.get("evidence_cid")
        != prepared_self.get("z3_import_denial_evidence_cid")
        or final_z3.get("evidence_cid")
        != final_self.get("z3_import_denial_evidence_cid")
        or prepared_z3.get("evidence_cid") == final_z3.get("evidence_cid")
        or prepared_parent.get("parent_z3_file_descriptor_count") != 0
        or final_parent.get("parent_z3_file_descriptor_count") != 0
        or prepared_parent.get("parent_z3_file_descriptor_root")
        != content_identity([])
        or final_parent.get("parent_z3_file_descriptor_root")
        != content_identity([])
        or prepared_parent.get(
            "parent_policy_denied_z3_native_mapping_count"
        )
        != 0
        or final_parent.get(
            "parent_policy_denied_z3_native_mapping_count"
        )
        != 0
        or prepared_parent.get("suite_task_policy_matrix_cid")
        != prepared_self.get("suite_task_policy_matrix_cid")
        or final_parent.get("suite_task_policy_matrix_cid")
        != final_self.get("suite_task_policy_matrix_cid")
        or prepared_parent.get("suite_task_profile_cid")
        != prepared_self.get("suite_task_profile_cid")
        or final_parent.get("suite_task_profile_cid")
        != final_self.get("suite_task_profile_cid")
        or prepared_parent.get("full_live_observation_cid")
        != prepared_self.get("full_live_observation_cid")
        or final_parent.get("full_live_observation_cid")
        != final_self.get("full_live_observation_cid")
        or prepared_parent.get("suite_task_receipt_cid")
        != prepared_self.get("suite_task_receipt_cid")
        or final_parent.get("suite_task_receipt_cid")
        != final_self.get("suite_task_receipt_cid")
        or prepared_self.get("fixed_point_to_prepared_identity_restored")
        is not True
        or final_self.get("fixed_point_to_prepared_identity_restored")
        is not True
        or prepared_parent.get("fixed_point_to_prepared_identity_restored")
        is not True
        or final_parent.get("fixed_point_to_prepared_identity_restored")
        is not True
        or prepared_self.get("writable_executable_mappings")
        != final_self.get("writable_executable_mappings")
        or prepared_self.get("thread_count") != final_self.get("thread_count")
        or prepared_self.get("kernel_task_count")
        != final_self.get("kernel_task_count")
        or prepared_self.get("child_process_count") != 0
        or final_self.get("child_process_count") != 0
        or prepared_self.get("process_group_count") != 1
        or final_self.get("process_group_count") != 1
        or prepared_self.get("process_group_leader_only") is not True
        or final_self.get("process_group_leader_only") is not True
        or prepared_parent.get("prepared_task_population_restored") is not True
        or final_parent.get("prepared_task_population_restored") is not True
        or prepared_parent.get("prepared_writable_executable_mapping_restored")
        is not True
        or final_parent.get("prepared_writable_executable_mapping_restored")
        is not True
        or prepared_parent.get("prepared_mappings_retained") is not True
        or final_parent.get("prepared_mappings_retained") is not True
        or prepared_parent.get("prepared_process_group_restored") is not True
        or final_parent.get("prepared_process_group_restored") is not True
    ):
        raise QualificationError("worker attestation phases disagree")
    value: dict[str, Any] = {
        "schema": "lgcvf-recovery-worker-parent-attestation-barrier@5",
        "suite_native_policy": prepared_self["suite_native_policy"],
        "suite_task_policy_matrix_cid": prepared_self[
            "suite_task_policy_matrix_cid"
        ],
        "suite_task_profile_cid": prepared_self["suite_task_profile_cid"],
        "full_live_observation_cid": prepared_self[
            "full_live_observation_cid"
        ],
        "suite_task_receipt_cid": prepared_self["suite_task_receipt_cid"],
        "fixed_point_to_prepared_identity_restored": True,
        "z3_import_policy_cid": prepared_self["z3_import_policy_cid"],
        "prepared_z3_import_denial_evidence_cid": prepared_self[
            "z3_import_denial_evidence_cid"
        ],
        "final_z3_import_denial_evidence_cid": final_self[
            "z3_import_denial_evidence_cid"
        ],
        "z3_import_denial_parent_live_validated": True,
        "full_z3_import_denial_evidence_persisted": True,
        "direct_verifier_validates_full_z3_import_denial_evidence": True,
        "prepared": dict(prepared),
        "final": dict(final),
        "parent_admitted_before_pytest": True,
        "parent_admitted_before_receipt": True,
        "writable_executable_mapping_stable": True,
        "prepared_normalized_executable_mapping_root": prepared_self.get(
            "normalized_executable_mapping_root"
        ),
        "final_normalized_executable_mapping_root": final_self.get(
            "normalized_executable_mapping_root"
        ),
        "prepared_mappings_retained": True,
        "kernel_task_population_restored": True,
        "children_absent": True,
        "process_group_leader_only": True,
    }
    value["barrier_cid"] = content_identity(value)
    return value


def _parent_attestation_barrier_from_child_cid(
    claimed_cid: Any,
    expected_barrier: Mapping[str, Any],
) -> dict[str, Any]:
    """Recover the parent-owned barrier after validating the child's CID link."""

    expected = dict(expected_barrier)
    body = {key: item for key, item in expected.items() if key != "barrier_cid"}
    expected_cid = expected.get("barrier_cid")
    if (
        expected.get("schema")
        != "lgcvf-recovery-worker-parent-attestation-barrier@5"
        or not _is_canonical_content_cid(expected_cid)
        or expected_cid != content_identity(body)
        or claimed_cid != expected_cid
    ):
        raise QualificationError("worker parent attestation barrier CID differs")
    return expected


def _validate_public_worker_self_observation(
    value: Any,
    *,
    phase: str,
    suite_id: str,
    runtime_cid: str,
    native_guard_cid: str,
    z3_required: bool,
    task_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise QualificationError("worker normalized self observation is absent")
    recovery = _RECOVERY_BY_SUITE_ID.get(suite_id)
    if recovery is None:
        raise QualificationError("worker normalized self suite differs")
    z3_commitments = _validate_public_z3_import_denial_commitments(
        value,
        phase=phase,
        task_id=recovery.task_id,
        suite_id=suite_id,
        source_projection_root=str(
            task_receipt.get("source_projection_root") or ""
        ),
    )
    body = {
        key: item for key, item in value.items() if key != "self_observation_cid"
    }
    numeric_fields = (
        "executable_mapping_count",
        "thread_count",
        "kernel_task_count",
        "child_process_count",
        "process_group_count",
    )
    if (
        set(value)
        != {
            "schema",
            "phase",
            "suite_id",
            "runtime_cid",
            "suite_native_policy",
            "suite_task_policy_matrix_cid",
            "suite_task_profile_cid",
            "full_live_observation_cid",
            "suite_task_receipt_cid",
            "fixed_point_to_prepared_identity_restored",
            "native_guard_cid",
            *_RECOVERY_PUBLIC_Z3_IMPORT_DENIAL_FIELDS,
            "executable_mapping_count",
            "normalized_executable_mapping_root",
            "writable_executable_mappings",
            "thread_count",
            "kernel_task_count",
            "normalized_kernel_task_root",
            "child_process_count",
            "children_absent",
            "process_group_count",
            "process_group_leader_only",
            "self_observation_cid",
        }
        or value.get("schema")
        != "lgcvf-recovery-worker-normalized-self-observation@4"
        or value.get("phase") != phase
        or value.get("suite_id") != suite_id
        or value.get("runtime_cid") != runtime_cid
        or value.get("suite_native_policy")
        != _recovery_suite_native_policy(z3_required=z3_required)
        or value.get("suite_task_policy_matrix_cid")
        != task_receipt.get("policy_matrix_cid")
        or value.get("suite_task_profile_cid")
        != task_receipt.get("profile_cid")
        or value.get("full_live_observation_cid")
        != task_receipt.get("full_live_observation_cid")
        or value.get("suite_task_receipt_cid")
        != task_receipt.get("receipt_cid")
        or value.get("fixed_point_to_prepared_identity_restored") is not True
        or value.get("native_guard_cid") != native_guard_cid
        or any(value.get(key) != item for key, item in z3_commitments.items())
        or any(
            isinstance(value.get(field), bool)
            or not isinstance(value.get(field), int)
            or int(value[field]) < 0
            for field in numeric_fields
        )
        or int(value.get("executable_mapping_count", 0)) <= 0
        or int(value.get("thread_count", 0)) <= 0
        or int(value.get("kernel_task_count", 0)) <= 0
        or value.get("normalized_kernel_task_root")
        != task_receipt.get("ordered_normalized_task_roots", [None])[-1]
        or value.get("child_process_count") != 0
        or value.get("children_absent") is not True
        or value.get("process_group_count") != 1
        or value.get("process_group_leader_only") is not True
        or not _is_canonical_content_cid(
            value.get("normalized_executable_mapping_root")
        )
        or value.get("writable_executable_mappings")
        != _expected_worker_writable_executable_mappings(
            z3_required=z3_required
        )
        or value.get("self_observation_cid") != content_identity(body)
    ):
        raise QualificationError("worker normalized self observation differs")
    return dict(value)


def _validate_public_worker_parent_observation(
    value: Any,
    *,
    phase: str,
    suite_id: str,
    runtime_cid: str,
    native_guard_cid: str,
    z3_required: bool,
    self_observation: Mapping[str, Any],
    task_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise QualificationError("worker normalized parent observation is absent")
    recovery = _RECOVERY_BY_SUITE_ID.get(suite_id)
    if recovery is None:
        raise QualificationError("worker normalized parent suite differs")
    z3_commitments = _validate_public_z3_import_denial_commitments(
        value,
        phase=phase,
        task_id=recovery.task_id,
        suite_id=suite_id,
        source_projection_root=str(
            task_receipt.get("source_projection_root") or ""
        ),
    )
    body = {
        key: item for key, item in value.items() if key != "parent_observation_cid"
    }
    controller_phase = (
        "prepared_parent_inspection"
        if phase == "prepared"
        else "final_parent_inspection"
    )
    controller = _validate_controller_zero_wx_observation(
        value.get("controller_zero_wx_observation"),
        phase=controller_phase,
    )
    numeric_parent_fields = (
        "executable_mapping_count",
        "kernel_task_count",
        "child_process_count",
        "process_group_count",
        "parent_z3_file_descriptor_count",
        "parent_policy_denied_z3_native_mapping_count",
    )
    if (
        set(value)
        != {
            "schema",
            "phase",
            "suite_id",
            "runtime_cid",
            "suite_native_policy",
            "suite_task_policy_matrix_cid",
            "suite_task_profile_cid",
            "full_live_observation_cid",
            "suite_task_receipt_cid",
            "fixed_point_to_prepared_identity_restored",
            "native_guard_cid",
            *_RECOVERY_PUBLIC_Z3_IMPORT_DENIAL_FIELDS,
            "parent_z3_file_descriptor_count",
            "parent_z3_file_descriptor_root",
            "parent_policy_denied_z3_native_mapping_count",
            "parent_policy_denied_z3_native_mapping_root",
            "pidfd_bound",
            "process_live",
            "process_start_time_matched",
            "executable_mapping_count",
            "normalized_executable_mapping_root",
            "writable_executable_mappings",
            "kernel_task_count",
            "normalized_kernel_task_root",
            "task_directory_identity_held",
            "task_directory_identity_restored",
            "child_process_count",
            "children_absent",
            "process_group_count",
            "process_group_leader_only",
            "prepared_task_population_restored",
            "prepared_writable_executable_mapping_restored",
            "prepared_mappings_retained",
            "prepared_process_group_restored",
            "controller_zero_wx_observation",
            "controller_zero_wx_observation_cid",
            "parent_observation_cid",
        }
        or value.get("schema")
        != "lgcvf-recovery-worker-parent-observation@4"
        or value.get("phase") != phase
        or value.get("suite_id") != suite_id
        or value.get("runtime_cid") != runtime_cid
        or value.get("suite_native_policy")
        != _recovery_suite_native_policy(z3_required=z3_required)
        or value.get("suite_task_policy_matrix_cid")
        != task_receipt.get("policy_matrix_cid")
        or value.get("suite_task_policy_matrix_cid")
        != self_observation.get("suite_task_policy_matrix_cid")
        or value.get("suite_task_profile_cid")
        != task_receipt.get("profile_cid")
        or value.get("suite_task_profile_cid")
        != self_observation.get("suite_task_profile_cid")
        or value.get("full_live_observation_cid")
        != task_receipt.get("full_live_observation_cid")
        or value.get("full_live_observation_cid")
        != self_observation.get("full_live_observation_cid")
        or value.get("suite_task_receipt_cid")
        != task_receipt.get("receipt_cid")
        or value.get("suite_task_receipt_cid")
        != self_observation.get("suite_task_receipt_cid")
        or value.get("fixed_point_to_prepared_identity_restored") is not True
        or value.get("fixed_point_to_prepared_identity_restored")
        != self_observation.get("fixed_point_to_prepared_identity_restored")
        or value.get("native_guard_cid") != native_guard_cid
        or any(value.get(key) != item for key, item in z3_commitments.items())
        or any(
            value.get(key) != self_observation.get(key)
            for key in _RECOVERY_PUBLIC_Z3_IMPORT_DENIAL_FIELDS
        )
        or value.get("parent_z3_file_descriptor_count") != 0
        or value.get("parent_z3_file_descriptor_root") != content_identity([])
        or value.get("parent_policy_denied_z3_native_mapping_count") != 0
        or value.get("parent_policy_denied_z3_native_mapping_root")
        != content_identity([])
        or value.get("pidfd_bound") is not True
        or value.get("process_live") is not True
        or value.get("process_start_time_matched") is not True
        or any(
            isinstance(value.get(field), bool)
            or not isinstance(value.get(field), int)
            or int(value[field]) < 0
            for field in numeric_parent_fields
        )
        or value.get("executable_mapping_count")
        != self_observation.get("executable_mapping_count")
        or value.get("normalized_executable_mapping_root")
        != self_observation.get("normalized_executable_mapping_root")
        or value.get("writable_executable_mappings")
        != self_observation.get("writable_executable_mappings")
        or value.get("kernel_task_count")
        != self_observation.get("kernel_task_count")
        or value.get("normalized_kernel_task_root")
        != self_observation.get("normalized_kernel_task_root")
        or value.get("task_directory_identity_held") is not True
        or value.get("task_directory_identity_restored") is not True
        or value.get("child_process_count") != 0
        or value.get("children_absent") is not True
        or value.get("process_group_count") != 1
        or value.get("process_group_leader_only") is not True
        or any(
            value.get(field) is not True
            for field in (
                "prepared_task_population_restored",
                "prepared_writable_executable_mapping_restored",
                "prepared_mappings_retained",
                "prepared_process_group_restored",
            )
        )
        or value.get("controller_zero_wx_observation") != controller
        or value.get("controller_zero_wx_observation_cid")
        != controller["observation_cid"]
        or value.get("parent_observation_cid") != content_identity(body)
    ):
        raise QualificationError("worker normalized parent observation differs")
    return dict(value)


def _validate_public_attestation_phase(
    value: Any,
    *,
    phase: str,
    suite_id: str,
    runtime_cid: str,
    native_guard_cid: str,
    z3_required: bool,
    task_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise QualificationError("worker public attestation phase is absent")
    recovery = _RECOVERY_BY_SUITE_ID.get(suite_id)
    if recovery is None:
        raise QualificationError("worker public attestation suite differs")
    z3_import_denial_evidence = _validate_recovery_z3_import_denial_evidence(
        value.get("z3_import_denial_evidence"),
        phase=phase,
        task_id=recovery.task_id,
        suite_id=suite_id,
        source_projection_root=str(
            task_receipt.get("source_projection_root") or ""
        ),
    )
    self_observation = _validate_public_worker_self_observation(
        value.get("self_observation"),
        phase=phase,
        suite_id=suite_id,
        runtime_cid=runtime_cid,
        native_guard_cid=native_guard_cid,
        z3_required=z3_required,
        task_receipt=task_receipt,
    )
    parent_observation = _validate_public_worker_parent_observation(
        value.get("parent_observation"),
        phase=phase,
        suite_id=suite_id,
        runtime_cid=runtime_cid,
        native_guard_cid=native_guard_cid,
        z3_required=z3_required,
        self_observation=self_observation,
        task_receipt=task_receipt,
    )
    body = {key: item for key, item in value.items() if key != "phase_cid"}
    if (
        set(value)
        != {
            "schema",
            "phase",
            "self_observation",
            "parent_observation",
            "parent_observation_cid",
            "z3_import_denial_evidence",
            "z3_import_denial_evidence_cid",
            "parent_admitted",
            "post_ack_state_rechecked",
            "phase_cid",
        }
        or value.get("schema") != "lgcvf-recovery-worker-attestation-phase@5"
        or value.get("phase") != phase
        or value.get("self_observation") != self_observation
        or value.get("parent_observation") != parent_observation
        or value.get("parent_observation_cid")
        != parent_observation["parent_observation_cid"]
        or value.get("z3_import_denial_evidence")
        != z3_import_denial_evidence
        or value.get("z3_import_denial_evidence_cid")
        != z3_import_denial_evidence["evidence_cid"]
        or value.get("z3_import_denial_evidence_cid")
        != self_observation.get("z3_import_denial_evidence_cid")
        or value.get("z3_import_denial_evidence_cid")
        != parent_observation.get("z3_import_denial_evidence_cid")
        or value.get("parent_admitted") is not True
        or value.get("post_ack_state_rechecked") is not True
        or value.get("phase_cid") != content_identity(body)
    ):
        raise QualificationError("worker public attestation phase differs")
    return dict(value)


def _validate_worker_attestation_barrier(
    value: Any,
    *,
    suite_id: str,
    runtime_cid: str,
    native_guard_cid: str,
    z3_required: bool,
    task_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise QualificationError("worker parent attestation barrier is absent")
    raw_prepared = value.get("prepared") if isinstance(value, Mapping) else None
    raw_prepared_self = (
        raw_prepared.get("self_observation")
        if isinstance(raw_prepared, Mapping)
        else None
    )
    prepared_native_guard_cid = (
        raw_prepared_self.get("native_guard_cid")
        if isinstance(raw_prepared_self, Mapping)
        else None
    )
    if not _is_canonical_content_cid(prepared_native_guard_cid):
        raise QualificationError("prepared native guard identity differs")
    prepared = _validate_public_attestation_phase(
        value.get("prepared"),
        phase="prepared",
        suite_id=suite_id,
        runtime_cid=runtime_cid,
        native_guard_cid=str(prepared_native_guard_cid),
        z3_required=z3_required,
        task_receipt=task_receipt,
    )
    final = _validate_public_attestation_phase(
        value.get("final"),
        phase="final",
        suite_id=suite_id,
        runtime_cid=runtime_cid,
        native_guard_cid=native_guard_cid,
        z3_required=z3_required,
        task_receipt=task_receipt,
    )
    reconstructed = _worker_attestation_barrier_evidence(prepared, final)
    if dict(value) != reconstructed:
        raise QualificationError("worker parent attestation barrier differs")
    return reconstructed


def _validate_worker_parent_terminal_observation(
    value: Any,
    *,
    worker_observation_cid: str,
    barrier: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise QualificationError("worker parent terminal observation is absent")
    observations = value.get("controller_zero_wx_observations")
    expected_phases = [
        "before_worker",
        "prepared_parent_inspection",
        "final_parent_inspection",
        "after_worker_exit",
    ]
    if not isinstance(observations, list) or len(observations) != len(expected_phases):
        raise QualificationError("worker terminal controller evidence is absent")
    normalized = [
        _validate_controller_zero_wx_observation(item, phase=phase)
        for item, phase in zip(observations, expected_phases, strict=True)
    ]
    prepared_parent = barrier.get("prepared")
    prepared_parent = (
        prepared_parent.get("parent_observation")
        if isinstance(prepared_parent, Mapping)
        else None
    )
    final_parent = barrier.get("final")
    final_parent = (
        final_parent.get("parent_observation")
        if isinstance(final_parent, Mapping)
        else None
    )
    body = {
        key: item for key, item in value.items() if key != "terminal_observation_cid"
    }
    terminal_numeric_fields = (
        "waitid_code",
        "waitid_status",
        "process_returncode",
        "controller_wx_mapping_count",
    )
    if (
        set(value)
        != {
            "schema",
            "waitid_code",
            "waitid_status",
            "pidfd_signaled_after_wait",
            "zombie_leader_reserved_before_reap",
            "pre_reap_process_group_leader_only",
            "pre_reap_signal_probe_present",
            "process_exited",
            "process_returncode",
            "receipt_pipe_closed",
            "process_group_empty_after_exit",
            "process_group_signal_probe_esrch",
            "cleanup_performed",
            "controller_zero_wx_observations",
            "controller_zero_wx_observation_root",
            "controller_wx_mapping_count",
            "attestation_barrier_cid",
            "suite_task_receipt_cid",
            "worker_observation_cid",
            "terminal_observation_cid",
        }
        or value.get("schema")
        != "lgcvf-recovery-worker-parent-terminal-observation@2"
        or any(
            isinstance(value.get(field), bool)
            or not isinstance(value.get(field), int)
            for field in terminal_numeric_fields
        )
        or value.get("waitid_code") != getattr(os, "CLD_EXITED", 1)
        or value.get("waitid_status") != 0
        or value.get("pidfd_signaled_after_wait") is not True
        or value.get("zombie_leader_reserved_before_reap") is not True
        or value.get("pre_reap_process_group_leader_only") is not True
        or value.get("pre_reap_signal_probe_present") is not True
        or value.get("process_exited") is not True
        or value.get("process_returncode") != 0
        or value.get("receipt_pipe_closed") is not True
        or value.get("process_group_empty_after_exit") is not True
        or value.get("process_group_signal_probe_esrch") is not True
        or value.get("cleanup_performed") is not False
        or value.get("controller_zero_wx_observations") != normalized
        or not isinstance(prepared_parent, Mapping)
        or not isinstance(final_parent, Mapping)
        or normalized[1]
        != prepared_parent.get("controller_zero_wx_observation")
        or normalized[2] != final_parent.get("controller_zero_wx_observation")
        or value.get("controller_zero_wx_observation_root")
        != content_identity(normalized)
        or value.get("controller_wx_mapping_count") != 0
        or value.get("attestation_barrier_cid") != barrier.get("barrier_cid")
        or value.get("suite_task_receipt_cid")
        != barrier.get("suite_task_receipt_cid")
        or value.get("worker_observation_cid") != worker_observation_cid
        or value.get("terminal_observation_cid") != content_identity(body)
    ):
        raise QualificationError("worker parent terminal observation differs")
    return dict(value)


def _validate_controller_zero_wx_aggregate(
    value: Any,
    *,
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise QualificationError("qualification controller W+X evidence is absent")
    entry = _validate_controller_zero_wx_observation(
        value.get("entry"), phase="qualification_entry"
    )
    final = _validate_controller_zero_wx_observation(
        value.get("final"), phase="qualification_final"
    )
    ordered = value.get("ordered_suite_observations")
    if not isinstance(ordered, list) or len(ordered) != len(observations):
        raise QualificationError("qualification controller suite evidence differs")
    normalized_ordered: list[list[dict[str, Any]]] = []
    phases = [
        "before_worker",
        "prepared_parent_inspection",
        "final_parent_inspection",
        "after_worker_exit",
    ]
    for raw_population, observation in zip(ordered, observations, strict=True):
        if not isinstance(raw_population, list) or len(raw_population) != len(phases):
            raise QualificationError("qualification controller phase population differs")
        normalized = [
            _validate_controller_zero_wx_observation(item, phase=phase)
            for item, phase in zip(raw_population, phases, strict=True)
        ]
        terminal = observation.get(
            "qualification_runtime_parent_terminal_observation"
        )
        if (
            not isinstance(terminal, Mapping)
            or terminal.get("controller_zero_wx_observations") != normalized
        ):
            raise QualificationError(
                "qualification controller evidence is not linked to its worker"
            )
        normalized_ordered.append(normalized)
    body = {
        key: item for key, item in value.items() if key != "controller_evidence_cid"
    }
    if (
        set(value)
        != {
            "schema",
            "entry",
            "ordered_suite_observations",
            "final",
            "controller_wx_mapping_count",
            "controller_rwx_permitted",
            "controller_evidence_cid",
        }
        or value.get("schema") != "lgcvf-recovery-controller-zero-wx@1"
        or value.get("entry") != entry
        or value.get("ordered_suite_observations") != normalized_ordered
        or value.get("final") != final
        or isinstance(value.get("controller_wx_mapping_count"), bool)
        or not isinstance(value.get("controller_wx_mapping_count"), int)
        or value.get("controller_wx_mapping_count") != 0
        or value.get("controller_rwx_permitted") is not False
        or value.get("controller_evidence_cid") != content_identity(body)
    ):
        raise QualificationError("qualification controller W+X evidence differs")
    return dict(value)


def _pidfd_process_is_live(pidfd: int) -> bool:
    try:
        readable, _, _ = select.select([pidfd], [], [], 0)
    except OSError as exc:
        raise QualificationError("worker pidfd inspection failed") from exc
    return not readable


def _wait_and_inspect_recovery_worker_terminal(
    process: subprocess.Popen[str],
    *,
    pidfd: int,
    deadline: float,
) -> dict[str, Any]:
    """Inspect the exited worker while its zombie leader still reserves its PGID."""

    reaped = False
    unexpected_group_member = False
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, 0)
        try:
            readable, _, _ = select.select([pidfd], [], [], remaining)
        except OSError as exc:
            raise QualificationError("worker pidfd terminal wait failed") from exc
        if not readable:
            raise subprocess.TimeoutExpired(process.args, remaining)
        try:
            result = os.waitid(
                os.P_PIDFD,
                pidfd,
                os.WEXITED | os.WNOWAIT,
            )
        except (AttributeError, OSError) as exc:
            raise QualificationError("worker waitid terminal observation failed") from exc
        if (
            result is None
            or result.si_pid != process.pid
            or result.si_code
            not in {
                getattr(os, "CLD_EXITED", 1),
                getattr(os, "CLD_KILLED", 2),
                getattr(os, "CLD_DUMPED", 3),
            }
        ):
            raise QualificationError("worker waitid terminal identity differs")
        group = _recovery_process_group_population(process.pid)
        try:
            os.killpg(process.pid, 0)
        except OSError as exc:
            raise QualificationError(
                "worker zombie-reserved process group is absent"
            ) from exc
        if (
            len(group) != 1
            or group[0].get("process_id") != process.pid
            or group[0].get("process_group_id") != process.pid
            or group[0].get("session_id") != process.pid
        ):
            unexpected_group_member = True
            # The unreaped zombie leader still owns this numeric PGID, so this
            # cleanup signal cannot target a subsequently reused process group.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            cleanup_deadline = time.monotonic() + 5.0
            while time.monotonic() < cleanup_deadline:
                group = _recovery_process_group_population(process.pid)
                if (
                    len(group) == 1
                    and group[0].get("process_id") == process.pid
                ):
                    break
                select.select([], [], [], 0.02)
            else:
                raise QualificationError(
                    "worker process group did not terminate within cleanup bound"
                )
        returncode = process.wait(timeout=30)
        reaped = True
        if _pidfd_process_is_live(pidfd):
            raise QualificationError("worker pidfd remained live after reap")
        terminal_group = _recovery_process_group_population(
            process.pid,
            require_leader=False,
        )
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            signal_probe_absent = True
        except PermissionError:
            signal_probe_absent = False
        else:
            signal_probe_absent = False
        if terminal_group or not signal_probe_absent:
            raise QualificationError("worker process group remains after reap")
        if unexpected_group_member:
            raise QualificationError("worker created a terminal process-group peer")
        return {
            "returncode": int(returncode),
            "waitid_code": int(result.si_code),
            "waitid_status": int(result.si_status),
            "pidfd_signaled_after_wait": True,
            "zombie_leader_reserved_before_reap": True,
            "pre_reap_process_group_leader_only": True,
            "pre_reap_signal_probe_present": True,
            "process_group_empty_after_exit": True,
            "process_group_signal_probe_esrch": True,
            "cleanup_performed": False,
        }
    except BaseException:
        if not reaped:
            # The live or zombie leader still reserves its PGID here.  Cleanup
            # is therefore safe even if terminal inspection itself failed.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                pass
        raise


def _parent_allowed_postprepared_native_paths(
    runtime: _ActiveQualificationRuntime,
    *,
    z3_required: bool,
) -> set[str]:
    native_platform = runtime.resolved.bundle.get("native_platform_binding")
    payload = (
        native_platform.get("solver_payload_native_files")
        if isinstance(native_platform, Mapping)
        else None
    )
    if not isinstance(payload, list):
        raise QualificationError("parent native payload authority is absent")
    result: set[str] = set()
    for item in payload:
        if not isinstance(item, Mapping):
            raise QualificationError("parent native payload authority differs")
        relative = str(item.get("path") or "")
        _strict_record_relative_path(relative, noun="parent native payload")
        if relative.startswith("z3/") and not z3_required:
            continue
        result.add(str((runtime.root / relative).resolve(strict=True)))
    stdlib, _stdlib_root, _stdlib_bytes = _stdlib_extension_native_manifest()
    result.update(stdlib)
    return result


def _recovery_process_z3_file_descriptor_evidence(
    process_id: int,
    z3_root: Path,
) -> list[str]:
    """Inventory Z3-root descriptor targets without exposing descriptor numbers."""

    fd_root = Path(f"/proc/{process_id}/fd")
    tokens: list[str] = []
    try:
        names = sorted(os.listdir(fd_root), key=int)
    except (OSError, ValueError) as exc:
        raise QualificationError(
            "recovery Z3 file-descriptor inventory is unavailable"
        ) from exc
    if len(names) > 4096:
        raise QualificationError("recovery Z3 file-descriptor inventory is unbounded")
    for name in names:
        if not name.isdecimal():
            raise QualificationError("recovery Z3 file-descriptor inventory differs")
        try:
            target = os.readlink(fd_root / name)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise QualificationError(
                "recovery Z3 file-descriptor identity is unavailable"
            ) from exc
        deleted = target.endswith(" (deleted)")
        candidate = target[: -len(" (deleted)")] if deleted else target
        if not os.path.isabs(candidate):
            continue
        try:
            resolved = Path(candidate).resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            continue
        if resolved == z3_root or resolved.is_relative_to(z3_root):
            relative = resolved.relative_to(z3_root).as_posix()
            tokens.append(
                "runtime:z3"
                + (("/" + relative) if relative != "." else "")
                + (":deleted" if deleted else "")
            )
    return sorted(tokens)


def _recovery_policy_denied_z3_native_mapping_evidence(
    mappings: Sequence[Mapping[str, Any]],
    z3_root: Path,
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for mapping in mappings:
        resolved = mapping.get("resolved_path")
        label = str(mapping.get("label") or "")
        under_runtime = (
            isinstance(resolved, str)
            and (
                Path(resolved) == z3_root
                or Path(resolved).is_relative_to(z3_root)
            )
        )
        libz3_named = (
            isinstance(resolved, str)
            and PurePath(resolved).name.startswith("libz3.so")
        ) or PurePath(label).name.startswith("libz3.so")
        if under_runtime or libz3_named:
            values.append(
                {
                    "path_disposition": (
                        "runtime_z3" if under_runtime else "libz3_named"
                    ),
                    "mapping_identity": content_identity(mapping),
                }
            )
    return sorted(values, key=_canonical_bytes)


def _parent_worker_live_observation(
    attestation: Mapping[str, Any],
    *,
    phase: str,
    launch_nonce: str,
    suite_id: str,
    process_id: int,
    pidfd: int,
    runtime: _ActiveQualificationRuntime,
    source_projection_root: str,
    z3_required: bool,
    prepared_exact_state: Mapping[str, Any] | None,
    controller_observation: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    recovery = _RECOVERY_BY_SUITE_ID.get(suite_id)
    if recovery is None:
        raise QualificationError(f"{suite_id}: recovery task identity is absent")
    expected_task_profile = _recovery_suite_task_policy(
        task_id=recovery.task_id,
        suite_id=suite_id,
    )
    task_observation = attestation.get("suite_task_observation")
    if not isinstance(task_observation, Mapping):
        raise QualificationError(
            f"{suite_id}: {phase} worker task observation is absent"
        )
    validated_task_observation = _validate_recovery_suite_task_observation(
        task_observation,
        task_id=recovery.task_id,
        suite_id=suite_id,
        runtime_cid=str(runtime.resolved.bundle.get("runtime_cid") or ""),
        source_projection_root=source_projection_root,
    )
    z3_import_denial_evidence = _validate_recovery_z3_import_denial_evidence(
        attestation.get("z3_import_denial_evidence"),
        phase=phase,
        task_id=recovery.task_id,
        suite_id=suite_id,
        source_projection_root=source_projection_root,
    )
    compact_task_receipt = _recovery_suite_task_receipt_from_live_observation(
        validated_task_observation,
        task_id=recovery.task_id,
        suite_id=suite_id,
        runtime_cid=str(runtime.resolved.bundle.get("runtime_cid") or ""),
        source_projection_root=source_projection_root,
    )
    body = {
        key: item for key, item in attestation.items() if key != "attestation_cid"
    }
    expected_fields = {
        "schema",
        "phase",
        "launch_nonce",
        "suite_id",
        "process_id",
        "process_start_time_ticks",
        "runtime_cid",
        "suite_native_policy",
        "suite_task_policy_matrix_cid",
        "suite_task_profile_cid",
        "suite_task_observation",
        "full_live_observation_cid",
        "suite_task_receipt_cid",
        "suite_task_terminal_records",
        "suite_task_terminal_record_root",
        "native_guard_cid",
        "z3_import_denial_evidence",
        "z3_import_denial_evidence_cid",
        "executable_mapping_count",
        "executable_mapping_root",
        "normalized_executable_mapping_root",
        "writable_executable_mappings",
        "thread_population",
        "thread_population_root",
        "kernel_task_ids",
        "kernel_task_root",
        "kernel_task_records",
        "kernel_task_record_root",
        "child_process_ids",
        "child_process_root",
        "process_group_population",
        "process_group_root",
        "attestation_cid",
    }
    if (
        set(attestation) != expected_fields
        or attestation.get("schema")
        != "lgcvf-recovery-worker-live-attestation@4"
        or attestation.get("phase") != phase
        or attestation.get("launch_nonce") != launch_nonce
        or attestation.get("suite_id") != suite_id
        or attestation.get("process_id") != process_id
        or attestation.get("runtime_cid")
        != runtime.resolved.bundle.get("runtime_cid")
        or attestation.get("suite_native_policy")
        != _recovery_suite_native_policy(z3_required=z3_required)
        or attestation.get("suite_task_policy_matrix_cid")
        != _recovery_suite_task_policy_matrix()["matrix_cid"]
        or attestation.get("suite_task_profile_cid")
        != expected_task_profile["profile_cid"]
        or attestation.get("suite_task_observation")
        != validated_task_observation
        or attestation.get("full_live_observation_cid")
        != validated_task_observation["observation_cid"]
        or attestation.get("suite_task_receipt_cid")
        != compact_task_receipt["receipt_cid"]
        or not isinstance(attestation.get("suite_task_terminal_records"), list)
        or attestation.get("suite_task_terminal_record_root")
        != content_identity(attestation.get("suite_task_terminal_records"))
        or not _is_canonical_content_cid(attestation.get("native_guard_cid"))
        or attestation.get("z3_import_denial_evidence")
        != z3_import_denial_evidence
        or attestation.get("z3_import_denial_evidence_cid")
        != z3_import_denial_evidence["evidence_cid"]
        or attestation.get("attestation_cid") != content_identity(body)
    ):
        raise QualificationError(f"{suite_id}: {phase} worker attestation differs")
    if not _pidfd_process_is_live(pidfd):
        raise QualificationError(f"{suite_id}: worker exited before {phase} inspection")
    expected_controller_phase = (
        "prepared_parent_inspection"
        if phase == "prepared"
        else "final_parent_inspection"
    )
    if (
        controller_observation
        != _controller_zero_wx_observation(phase=expected_controller_phase)
        or controller_observation.get("controller_wx_mapping_count") != 0
    ):
        raise QualificationError(f"{suite_id}: controller W+X state differs")
    start_time = _recovery_process_start_time_ticks(process_id)
    tasks = _recovery_kernel_task_population(process_id)
    task_records = _recovery_kernel_task_records(process_id)
    if [int(item["tid"]) for item in task_records] != tasks:
        raise QualificationError(f"{suite_id}: worker task records differ")
    children = _recovery_child_process_population(process_id, task_ids=tasks)
    process_group = _recovery_process_group_population(process_id)
    mappings = _native_executable_mappings(process_id)
    z3_root = (runtime.root / "z3").resolve(strict=True)
    parent_z3_fds = _recovery_process_z3_file_descriptor_evidence(
        process_id,
        z3_root,
    )
    denied_z3_mappings = (
        _recovery_policy_denied_z3_native_mapping_evidence(mappings, z3_root)
        if recovery.task_id in _RECOVERY_Z3_IMPORT_DENIED_TASKS
        else []
    )
    after_tasks = _recovery_kernel_task_population(process_id)
    after_task_records = _recovery_kernel_task_records(process_id)
    after_process_group = _recovery_process_group_population(process_id)
    after_start_time = _recovery_process_start_time_ticks(process_id)
    after_parent_z3_fds = _recovery_process_z3_file_descriptor_evidence(
        process_id,
        z3_root,
    )
    if (
        tasks != after_tasks
        or task_records != after_task_records
        or process_group != after_process_group
        or start_time != after_start_time
        or parent_z3_fds != after_parent_z3_fds
        or (
            recovery.task_id in _RECOVERY_Z3_IMPORT_DENIED_TASKS
            and (parent_z3_fds or denied_z3_mappings)
        )
    ):
        raise QualificationError(
            f"{suite_id}: {phase} worker state changed during inspection"
        )
    normalized_mappings = _normalized_executable_mapping_signatures(mappings)
    threads = attestation.get("thread_population")
    if not isinstance(threads, list):
        raise QualificationError(f"{suite_id}: worker thread attestation is absent")
    normalized_threads: list[dict[str, Any]] = []
    for item in threads:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"ident", "native_id", "name", "daemon"}
            or isinstance(item.get("ident"), bool)
            or not isinstance(item.get("ident"), int)
            or isinstance(item.get("native_id"), bool)
            or not isinstance(item.get("native_id"), int)
            or not isinstance(item.get("name"), str)
            or not isinstance(item.get("daemon"), bool)
        ):
            raise QualificationError(f"{suite_id}: worker thread attestation differs")
        normalized_threads.append(dict(item))
    writable_mappings = _writable_executable_mapping_signatures(mappings)
    expected_writable = _expected_worker_writable_executable_mappings(
        z3_required=z3_required
    )
    if (
        attestation.get("process_start_time_ticks") != start_time
        or attestation.get("executable_mapping_count") != len(mappings)
        or attestation.get("executable_mapping_root")
        != _native_executable_mapping_root(mappings)
        or attestation.get("normalized_executable_mapping_root")
        != content_identity(normalized_mappings)
        or attestation.get("writable_executable_mappings") != writable_mappings
        or writable_mappings != expected_writable
        or attestation.get("thread_population_root")
        != content_identity(normalized_threads)
        or attestation.get("kernel_task_ids") != tasks
        or attestation.get("kernel_task_root") != content_identity(tasks)
        or attestation.get("kernel_task_records") != task_records
        or attestation.get("kernel_task_record_root")
        != content_identity(task_records)
        or attestation.get("suite_task_terminal_records") != task_records
        or attestation.get("suite_task_terminal_record_root")
        != content_identity(task_records)
        or attestation.get("child_process_ids") != children
        or attestation.get("child_process_root") != content_identity(children)
        or attestation.get("process_group_population") != process_group
        or attestation.get("process_group_root") != content_identity(process_group)
        or children
        or len(process_group) != 1
        or process_group[0].get("process_id") != process_id
    ):
        raise QualificationError(f"{suite_id}: {phase} live worker state differs")
    mapping_identities = {content_identity(item) for item in mappings}
    prepared_task_population_restored = True
    prepared_writable_mapping_restored = True
    prepared_mappings_retained = True
    prepared_process_group_restored = True
    task_directory_identity_held = False
    task_directory_identity_restored = False
    held_task_directories: list[dict[str, Any]]
    if prepared_exact_state is not None:
        prepared_tasks = prepared_exact_state.get("kernel_task_ids")
        prepared_task_records = prepared_exact_state.get("kernel_task_records")
        prepared_held = prepared_exact_state.get("held_task_directories")
        prepared_writable = prepared_exact_state.get("writable_executable_mappings")
        prepared_mapping_identities = prepared_exact_state.get("mapping_identities")
        prepared_process_group = prepared_exact_state.get("process_group_population")
        prepared_task_observation = prepared_exact_state.get(
            "full_live_task_observation"
        )
        prepared_task_receipt = prepared_exact_state.get("compact_task_receipt")
        if (
            not isinstance(prepared_tasks, list)
            or not isinstance(prepared_task_records, list)
            or not isinstance(prepared_held, list)
            or not isinstance(prepared_writable, list)
            or not isinstance(prepared_mapping_identities, set)
            or not isinstance(prepared_process_group, list)
            or not isinstance(prepared_task_observation, Mapping)
            or not isinstance(prepared_task_receipt, Mapping)
            or dict(prepared_task_observation) != validated_task_observation
            or dict(prepared_task_receipt) != compact_task_receipt
        ):
            raise QualificationError(f"{suite_id}: prepared worker state differs")
        prepared_task_population_restored = tasks == prepared_tasks
        if task_records != prepared_task_records:
            raise QualificationError(
                f"{suite_id}: prepared task identity was not restored"
            )
        held_task_directories = prepared_held
        task_directory_identity_restored = (
            _revalidate_held_recovery_task_directories(
                process_id,
                held_task_directories,
            )
            == task_records
        )
        task_directory_identity_held = True
        prepared_writable_mapping_restored = writable_mappings == prepared_writable
        prepared_mappings_retained = prepared_mapping_identities.issubset(
            mapping_identities
        )
        prepared_process_group_restored = process_group == prepared_process_group
        allowed_new_paths = _parent_allowed_postprepared_native_paths(
            runtime,
            z3_required=z3_required,
        )
        for item in mappings:
            identity = content_identity(item)
            if identity in prepared_mapping_identities:
                continue
            if (
                item.get("path_kind") != "absolute_file"
                or item.get("resolved_path") not in allowed_new_paths
            ):
                raise QualificationError(
                    f"{suite_id}: unbound executable mapping appeared after admission"
                )
        if (
            not prepared_task_population_restored
            or not task_directory_identity_restored
            or not prepared_writable_mapping_restored
            or not prepared_mappings_retained
            or not prepared_process_group_restored
        ):
            raise QualificationError(
                f"{suite_id}: live worker state changed after admission"
            )
    else:
        # Open only after all fallible public-observation reconstruction below;
        # the successful final open/revalidate/return sequence transfers the
        # lease directly to the outer coordinator.
        held_task_directories = []
        task_directory_identity_restored = True
        task_directory_identity_held = True
    normalized_task_root = content_identity(
        _normalized_kernel_task_role_evidence(
            task_records,
            task_id=recovery.task_id,
        )
    )
    expected_task_counts = validated_task_observation.get(
        "ordered_task_counts"
    )
    expected_task_roots = validated_task_observation.get(
        "ordered_normalized_task_roots"
    )
    if (
        not isinstance(expected_task_counts, list)
        or not expected_task_counts
        or not isinstance(expected_task_roots, list)
        or not expected_task_roots
        or len(tasks) != expected_task_counts[-1]
        or normalized_task_root != expected_task_roots[-1]
    ):
        if prepared_exact_state is None:
            _close_held_recovery_task_directories(held_task_directories)
        raise QualificationError(
            f"{suite_id}: task fixed-point live state differs"
        )
    public: dict[str, Any] = {
        "schema": "lgcvf-recovery-worker-parent-observation@4",
        "phase": phase,
        "suite_id": suite_id,
        "runtime_cid": runtime.resolved.bundle["runtime_cid"],
        "suite_native_policy": _recovery_suite_native_policy(
            z3_required=z3_required
        ),
        "suite_task_policy_matrix_cid": _recovery_suite_task_policy_matrix()[
            "matrix_cid"
        ],
        "suite_task_profile_cid": expected_task_profile["profile_cid"],
        "full_live_observation_cid": attestation["full_live_observation_cid"],
        "suite_task_receipt_cid": attestation["suite_task_receipt_cid"],
        "fixed_point_to_prepared_identity_restored": True,
        "native_guard_cid": attestation["native_guard_cid"],
        **_public_z3_import_denial_commitments(z3_import_denial_evidence),
        "parent_z3_file_descriptor_count": len(parent_z3_fds),
        "parent_z3_file_descriptor_root": content_identity(parent_z3_fds),
        "parent_policy_denied_z3_native_mapping_count": len(
            denied_z3_mappings
        ),
        "parent_policy_denied_z3_native_mapping_root": content_identity(
            denied_z3_mappings
        ),
        "pidfd_bound": True,
        "process_live": True,
        "process_start_time_matched": True,
        "executable_mapping_count": len(mappings),
        "normalized_executable_mapping_root": content_identity(
            normalized_mappings
        ),
        "writable_executable_mappings": writable_mappings,
        "kernel_task_count": len(tasks),
        "normalized_kernel_task_root": normalized_task_root,
        "task_directory_identity_held": task_directory_identity_held,
        "task_directory_identity_restored": task_directory_identity_restored,
        "child_process_count": len(children),
        "children_absent": True,
        "process_group_count": len(process_group),
        "process_group_leader_only": True,
        "prepared_task_population_restored": prepared_task_population_restored,
        "prepared_writable_executable_mapping_restored": (
            prepared_writable_mapping_restored
        ),
        "prepared_mappings_retained": prepared_mappings_retained,
        "prepared_process_group_restored": prepared_process_group_restored,
        "controller_zero_wx_observation": dict(controller_observation),
        "controller_zero_wx_observation_cid": controller_observation[
            "observation_cid"
        ],
    }
    public["parent_observation_cid"] = content_identity(public)
    exact_state = {
        "process_start_time_ticks": start_time,
        "mappings": mappings,
        "mapping_identities": mapping_identities,
        "kernel_task_ids": tasks,
        "kernel_task_records": task_records,
        "held_task_directories": held_task_directories,
        "child_process_ids": children,
        "process_group_population": process_group,
        "process_group_root": content_identity(process_group),
        "writable_executable_mappings": writable_mappings,
        "normalized_executable_mapping_root": public[
            "normalized_executable_mapping_root"
        ],
        "full_live_task_observation": validated_task_observation,
        "compact_task_receipt": compact_task_receipt,
    }
    if prepared_exact_state is None:
        held_task_directories = _hold_recovery_task_directories(
            process_id,
            task_records,
        )
        try:
            if (
                _revalidate_held_recovery_task_directories(
                    process_id,
                    held_task_directories,
                )
                != task_records
            ):
                raise QualificationError(
                    f"{suite_id}: prepared task directory identity differs"
                )
        except BaseException:
            try:
                _close_held_recovery_task_directories(held_task_directories)
            except BaseException:
                pass
            raise
        exact_state["held_task_directories"] = held_task_directories
    return public, exact_state


def _parent_worker_ack(
    attestation: Mapping[str, Any],
    parent_observation: Mapping[str, Any],
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": "lgcvf-recovery-worker-parent-ack@4",
        "phase": attestation.get("phase"),
        "launch_nonce": attestation.get("launch_nonce"),
        "suite_id": attestation.get("suite_id"),
        "process_id": attestation.get("process_id"),
        "process_start_time_ticks": attestation.get("process_start_time_ticks"),
        "child_attestation_cid": attestation.get("attestation_cid"),
        "observed_executable_mapping_root": attestation.get(
            "executable_mapping_root"
        ),
        "observed_kernel_task_root": attestation.get("kernel_task_root"),
        "observed_kernel_task_record_root": attestation.get(
            "kernel_task_record_root"
        ),
        "suite_task_policy_matrix_cid": attestation.get(
            "suite_task_policy_matrix_cid"
        ),
        "suite_task_profile_cid": attestation.get("suite_task_profile_cid"),
        "full_live_observation_cid": attestation.get(
            "full_live_observation_cid"
        ),
        "suite_task_receipt_cid": attestation.get(
            "suite_task_receipt_cid"
        ),
        "fixed_point_to_prepared_identity_restored": True,
        "z3_import_denial_evidence_cid": attestation.get(
            "z3_import_denial_evidence_cid"
        ),
        "parent_z3_file_descriptor_count": parent_observation.get(
            "parent_z3_file_descriptor_count"
        ),
        "parent_z3_file_descriptor_root": parent_observation.get(
            "parent_z3_file_descriptor_root"
        ),
        "parent_policy_denied_z3_native_mapping_count": parent_observation.get(
            "parent_policy_denied_z3_native_mapping_count"
        ),
        "parent_policy_denied_z3_native_mapping_root": parent_observation.get(
            "parent_policy_denied_z3_native_mapping_root"
        ),
        "suite_task_terminal_record_root": attestation.get(
            "suite_task_terminal_record_root"
        ),
        "observed_process_group_root": attestation.get("process_group_root"),
        "parent_observation": dict(parent_observation),
        "parent_observation_cid": parent_observation.get(
            "parent_observation_cid"
        ),
        "admitted": True,
    }
    value["ack_cid"] = content_identity(value)
    return value


def _expected_solver_native_mapping_evidence(
    *, z3_required: bool,
) -> dict[str, Any]:
    projected_paths = [
        "cvc5.libs/libcvc5-72ca88c7.so.1",
        "cvc5.libs/libcvc5parser-f4f50221.so.1",
        "cvc5.libs/libgmp-1fa5f074.so.10.5.0",
        "cvc5.libs/libpoly-eb10d276.so.0.2.1",
        "cvc5.libs/libpolyxx-19bd257c.so.0.2.1",
        "cvc5/cvc5_python_base.cpython-312-aarch64-linux-gnu.so",
    ]
    if z3_required:
        projected_paths.append("z3/lib/libz3.so")
    value: dict[str, Any] = {
        "schema": "lgcvf-qualification-runtime-native-mapping@2",
        "z3_required": z3_required,
        "projected_solver_paths": projected_paths,
        "z3_resource_path": "z3/lib" if z3_required else None,
        "z3_mapped_path": "z3/lib/libz3.so" if z3_required else None,
        "deleted_mappings": False,
    }
    value["mapping_cid"] = content_identity(value)
    return value


def _load_bound_native_host_objects(
    bundle: Mapping[str, Any],
    *,
    task_id: str,
) -> None:
    """Force solver SONAME reuse from already validated absolute host bytes."""

    native_platform = bundle.get("native_platform_binding")
    host = (
        native_platform.get("native_host_runtime")
        if isinstance(native_platform, Mapping)
        else None
    )
    objects = host.get("objects") if isinstance(host, Mapping) else None
    if not isinstance(objects, list):
        raise QualificationError("worker native host objects are absent")
    selected_sonames = (
        set(_RECOVERY_080_PREBOUND_HOST_SONAMES)
        if task_id == "LGCVF-080"
        else set()
    )
    loaded_selected_sonames: set[str] = set()
    mode = int(getattr(os, "RTLD_NOW", 2)) | int(getattr(os, "RTLD_GLOBAL", 0x100))
    for item in objects:
        if not isinstance(item, Mapping):
            raise QualificationError("worker native host object differs")
        soname = str(item.get("soname") or "")
        if (
            soname in _RECOVERY_080_PREBOUND_HOST_SONAMES
            and soname not in selected_sonames
        ):
            continue
        token = str(item.get("real_path_token") or "")
        if not token.startswith("system:/"):
            raise QualificationError("worker native host path token differs")
        path = token[len("system:") :]
        try:
            handle = ctypes.CDLL(path, mode=mode)
        except OSError as exc:
            raise QualificationError("worker native host object cannot be loaded") from exc
        _RECOVERY_WORKER_NATIVE_HANDLES.append(handle)
        if soname in selected_sonames:
            loaded_selected_sonames.add(soname)
    if loaded_selected_sonames != selected_sonames:
        raise QualificationError("worker native host preload scope differs")


def _validate_runtime_module_origins(
    runtime_root: Path,
    runtime_bundle: Mapping[str, Any],
    *,
    z3_required: bool,
) -> dict[str, Any]:
    """Close every protected module origin and the actual solver library map."""

    runtime_root = runtime_root.resolve(strict=True)
    python_runtime = runtime_bundle.get("python_runtime_binding")
    producer = (
        python_runtime.get("z3_libffi_rwx_producer_binding")
        if isinstance(python_runtime, Mapping)
        else None
    )
    producer_body = (
        {
            key: item
            for key, item in producer.items()
            if key != "producer_binding_cid"
        }
        if isinstance(producer, Mapping)
        else {}
    )
    ctypes_module = sys.modules.get("_ctypes")
    ctypes_origin = _module_file(ctypes_module) if ctypes_module is not None else None
    expected_ctypes_origin = Path(
        "/usr/lib/python3.12/lib-dynload/_ctypes.cpython-312-aarch64-linux-gnu.so"
    )
    if (
        not isinstance(producer, Mapping)
        or producer.get("schema") != "lgcvf-z3-libffi-rwx-producer-binding@1"
        or producer.get("producer_binding_cid") != content_identity(producer_body)
        or ctypes_origin != expected_ctypes_origin
        or producer.get("ctypes_extension", {}).get("path_token")
        != "stdlib-lib-dynload:_ctypes.cpython-312-aarch64-linux-gnu.so"
        or producer.get("libffi_host_object", {}).get("soname") != "libffi.so.8"
    ):
        raise QualificationError("ctypes/libffi W+X producer identity differs")
    modules: list[dict[str, str]] = []
    for name, module in sorted(sys.modules.items()):
        if name.partition(".")[0] not in _QUALIFICATION_RUNTIME_PROTECTED_MODULE_ROOTS:
            continue
        origin = _module_file(module)
        if origin is None or not origin.is_relative_to(runtime_root):
            raise QualificationError(f"qualification runtime module escaped: {name}")
        modules.append(
            {"name": name, "origin": origin.relative_to(runtime_root).as_posix()}
        )
    required_roots = {"pytest", "cvc5"} | ({"z3"} if z3_required else set())
    if not required_roots.issubset({item["name"].partition(".")[0] for item in modules}):
        raise QualificationError("qualification runtime required modules are absent")
    if not z3_required and any(
        item["name"].partition(".")[0] == "z3" for item in modules
    ):
        raise QualificationError("z3 loaded for a cvc5-only recovery suite")
    if z3_required:
        try:
            import builtins

            import z3.z3core as z3core

            z3_resource_root = Path(
                str(z3core._z3_lib_resource_path)
            ).resolve(strict=True)
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            raise QualificationError("bound z3 library resource is unavailable") from exc
        expected_z3 = (runtime_root / "z3/lib/libz3.so").resolve(strict=True)
        if (
            hasattr(builtins, "Z3_LIB_DIRS")
            or z3_resource_root != expected_z3.parent
        ):
            raise QualificationError("z3 loaded outside the runtime projection")
    try:
        maps_bytes = Path("/proc/self/maps").read_bytes()
    except OSError as exc:
        raise QualificationError("worker native process map is unavailable") from exc
    if len(maps_bytes) > 4 * 1024 * 1024 or b" (deleted)" in maps_bytes:
        raise QualificationError("worker native process map differs")
    mapped: set[Path] = set()
    for raw_line in maps_bytes.splitlines():
        fields = raw_line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith(b"/"):
            continue
        try:
            mapped.add(Path(os.fsdecode(fields[5])).resolve(strict=True))
        except OSError as exc:
            raise QualificationError("worker native mapped path is unavailable") from exc
    expected_projected_paths = [
        "cvc5/cvc5_python_base.cpython-312-aarch64-linux-gnu.so",
        "cvc5.libs/libcvc5-72ca88c7.so.1",
        "cvc5.libs/libcvc5parser-f4f50221.so.1",
        "cvc5.libs/libgmp-1fa5f074.so.10.5.0",
        "cvc5.libs/libpoly-eb10d276.so.0.2.1",
        "cvc5.libs/libpolyxx-19bd257c.so.0.2.1",
    ]
    if z3_required:
        expected_projected_paths.append("z3/lib/libz3.so")
    expected_projected = {
        (runtime_root / path).resolve(strict=True)
        for path in expected_projected_paths
    }
    if not expected_projected.issubset(mapped):
        raise QualificationError("worker solver native mapping differs")
    libffi_real_path = Path(
        "/usr/lib/aarch64-linux-gnu/libffi.so.8.1.4"
    ).resolve(strict=True)
    if libffi_real_path not in mapped:
        raise QualificationError("worker libffi producer mapping differs")
    native_mapping = _expected_solver_native_mapping_evidence(
        z3_required=z3_required
    )
    if native_mapping["projected_solver_paths"] != sorted(
        path.relative_to(runtime_root).as_posix() for path in expected_projected
    ):
        raise QualificationError("worker solver native mapping policy differs")
    return {
        "module_origins": modules,
        "module_origin_root": content_identity(modules),
        "native_mapping": native_mapping,
        "z3_libffi_rwx_producer_binding_cid": producer[
            "producer_binding_cid"
        ],
    }


def _logical_legal_data_import_path_policy(
    suite: Suite,
    execution_projection: Mapping[str, Any],
    manifest: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reconstruct the source-derived temporary path policy without paths."""

    if suite.owner_root != "ipfs_datasets_py":
        evidence: dict[str, Any] = {
            "schema": "lgcvf-recovery-temporary-import-path-policy@1",
            "disposition": "not_required_for_accelerator_owner",
            "runtime_projection_first": True,
            "temporary_path_position": None,
            "source_projection_root": execution_projection.get(
                "copied_source_manifest_root"
            ),
            "file_count": 0,
            "total_bytes": 0,
            "file_manifest_root": content_identity([]),
            "direct_import_candidates": [],
            "protected_runtime_collisions": [],
            "contains_symlink_or_special": False,
            "contains_bytecode_or_native": False,
        }
        evidence["policy_cid"] = content_identity(evidence)
        return evidence
    prefix = "ipfs_datasets_py/ipfs_datasets_py/processors/legal_data/"
    legal_manifest = [
        dict(item) for item in manifest if str(item.get("path") or "").startswith(prefix)
    ]
    if not legal_manifest or len(legal_manifest) > 4_096:
        raise QualificationError("tracked legal-data file population differs")
    candidates: set[str] = set()
    for item in legal_manifest:
        remainder = str(item["path"])[len(prefix) :]
        first = remainder.partition("/")[0]
        if not first or first in {".", ".."}:
            raise QualificationError("tracked legal-data import path differs")
        name = PurePath(first).name
        if _qualification_runtime_is_bytecode(remainder) or _qualification_runtime_is_native(
            remainder
        ):
            raise QualificationError(
                "tracked legal-data path contains bytecode or native code"
            )
        if "/" in remainder:
            candidates.add(first)
        elif name.endswith(".py"):
            candidates.add(name[:-3])
        elif name.endswith((".pyc", ".pyo")):
            raise QualificationError("tracked legal-data path contains sourceless code")
    collisions = sorted(
        candidates.intersection(_QUALIFICATION_RUNTIME_PROTECTED_MODULE_ROOTS)
    )
    if collisions:
        raise QualificationError(
            "tracked legal-data path shadows the qualification runtime"
        )
    evidence = {
        "schema": "lgcvf-recovery-temporary-import-path-policy@1",
        "disposition": "preinserted_tracked_datasets_legal_data",
        "relative_path": prefix[:-1],
        "runtime_projection_first": True,
        "temporary_path_position": 1,
        "source_projection_root": execution_projection.get(
            "copied_source_manifest_root"
        ),
        "file_count": len(legal_manifest),
        "total_bytes": sum(int(item["size_bytes"]) for item in legal_manifest),
        "file_manifest_root": content_identity(legal_manifest),
        "direct_import_candidates": sorted(candidates),
        "protected_runtime_collisions": collisions,
        "contains_symlink_or_special": False,
        "contains_bytecode_or_native": False,
    }
    evidence["policy_cid"] = content_identity(evidence)
    return evidence


def _legal_data_import_path_policy(
    worker_root: Path,
    suite: Suite,
    execution_projection: Mapping[str, Any],
    *,
    source_revalidation_guard: _RecoveryZ3ImportDenialGuard | None = None,
) -> tuple[Path | None, dict[str, Any]]:
    """Bind the sole tracked conftest import-path addition below runtime."""

    if suite.owner_root != "ipfs_datasets_py":
        return None, _logical_legal_data_import_path_policy(
            suite,
            execution_projection,
            (),
        )

    relative_root = "ipfs_datasets_py/ipfs_datasets_py/processors/legal_data"
    legal_root = worker_root / relative_root
    try:
        root_status = legal_root.lstat()
    except OSError as exc:
        raise QualificationError("tracked legal-data import root is absent") from exc
    if (
        not stat.S_ISDIR(root_status.st_mode)
        or root_status.st_uid != os.geteuid()
        or stat.S_IMODE(root_status.st_mode) != 0o500
    ):
        raise QualificationError("tracked legal-data import root differs")
    paths = _recovery_projection_source_paths(worker_root, (suite,))
    manifest, _payloads = _recovery_projection_manifest(
        worker_root,
        paths,
        head_bound=False,
        source_revalidation_guard=source_revalidation_guard,
    )
    if content_identity(manifest) != execution_projection.get(
        "copied_source_manifest_root"
    ):
        raise QualificationError("tracked legal-data source projection differs")
    prefix = relative_root + "/"
    legal_manifest = [item for item in manifest if str(item["path"]).startswith(prefix)]
    if not legal_manifest or len(legal_manifest) > 4_096:
        raise QualificationError("tracked legal-data file population differs")
    observed_files: set[str] = set()
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
    flags |= getattr(os, "O_NOFOLLOW", 0)
    legal_fd = os.open(legal_root, flags)
    opened_root_status = os.fstat(legal_fd)
    if (
        not stat.S_ISDIR(opened_root_status.st_mode)
        or (opened_root_status.st_dev, opened_root_status.st_ino)
        != (root_status.st_dev, root_status.st_ino)
        or opened_root_status.st_uid != root_status.st_uid
        or stat.S_IMODE(opened_root_status.st_mode)
        != stat.S_IMODE(root_status.st_mode)
    ):
        os.close(legal_fd)
        raise QualificationError("tracked legal-data opened root differs")

    def visit_legal(directory_fd: int, relative: str, depth: int) -> None:
        if depth > 64:
            raise QualificationError("tracked legal-data tree is too deep")
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                child_relative = f"{relative}/{entry.name}" if relative else entry.name
                metadata = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(metadata.st_mode):
                    if (
                        metadata.st_uid != os.geteuid()
                        or stat.S_IMODE(metadata.st_mode) != 0o500
                    ):
                        raise QualificationError(
                            "tracked legal-data directory mode differs"
                        )
                    child = os.open(entry.name, flags, dir_fd=directory_fd)
                    try:
                        opened = os.fstat(child)
                        if (opened.st_dev, opened.st_ino) != (
                            metadata.st_dev,
                            metadata.st_ino,
                        ):
                            raise QualificationError(
                                "tracked legal-data directory changed"
                            )
                        visit_legal(child, child_relative, depth + 1)
                    finally:
                        os.close(child)
                elif (
                    stat.S_ISREG(metadata.st_mode)
                    and metadata.st_uid == os.geteuid()
                    and metadata.st_nlink == 1
                    and stat.S_IMODE(metadata.st_mode) == 0o400
                ):
                    observed_files.add(prefix + child_relative)
                    if len(observed_files) > 4_096:
                        raise QualificationError(
                            "tracked legal-data file population exceeds its bound"
                        )
                else:
                    raise QualificationError(
                        "tracked legal-data tree contains a special entry"
                    )

    try:
        visit_legal(legal_fd, "", 0)
    finally:
        os.close(legal_fd)
    if observed_files != {str(item["path"]) for item in legal_manifest}:
        raise QualificationError("tracked legal-data tree inventory differs")
    evidence = _logical_legal_data_import_path_policy(
        suite,
        execution_projection,
        manifest,
    )
    return legal_root, evidence


def _legal_data_loaded_module_origins(
    worker_root: Path,
    suite: Suite,
    legal_root: Path | None,
) -> list[dict[str, str]]:
    """Bind every module actually loaded through the temporary tracked path."""

    if legal_root is None:
        return []
    paths = set(_recovery_projection_source_paths(worker_root, (suite,)))
    result: list[dict[str, str]] = []
    resolved_legal = legal_root.resolve(strict=True)
    for name, module in sorted(sys.modules.items()):
        origin = _module_file(module)
        if origin is None or not origin.is_relative_to(resolved_legal):
            continue
        relative = origin.relative_to(worker_root).as_posix()
        if relative not in paths:
            raise QualificationError(
                "temporary legal-data module escaped the copied projection"
            )
        result.append({"name": name, "origin": relative})
    return result


def _worker(
    suite_id: str,
    *,
    execution_root: Path | None = None,
    write_root: Path | None = None,
    receipt_descriptor: int | None = None,
    attestation_descriptor: int | None = None,
    acknowledgement_descriptor: int | None = None,
    attestation_nonce: str | None = None,
    execution_projection: Mapping[str, Any] | None = None,
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
    recovery = _RECOVERY_BY_SUITE_ID.get(suite_id)
    z3_required = (
        recovery is not None and recovery.task_id in _RECOVERY_Z3_REQUIRED_TASKS
    )
    worker_schema = RECOVERY_WORKER_SCHEMA if recovery is not None else WORKER_SCHEMA
    try:
        runtime_bundle: dict[str, Any] | None = None
        runtime_bootstrap: dict[str, Any] | None = None
        runtime_policy_binding: dict[str, Any] | None = None
        runtime_root: Path | None = None
        runtime_import_evidence: dict[str, Any] | None = None
        native_load_guard: _RecoveryNativeLoadGuard | None = None
        z3_import_denial_guard: _RecoveryZ3ImportDenialGuard | None = None
        prepared_z3_import_denial_evidence: dict[str, Any] | None = None
        final_z3_import_denial_evidence: dict[str, Any] | None = None
        native_maps_before_solver: list[dict[str, Any]] | None = None
        native_threads_before_pytest: list[dict[str, Any]] | None = None
        native_kernel_tasks_before_pytest: list[int] | None = None
        native_children_before_pytest: list[int] | None = None
        native_load_evidence: dict[str, Any] | None = None
        prepared_attestation_evidence: dict[str, Any] | None = None
        final_attestation_evidence: dict[str, Any] | None = None
        attestation_barrier_evidence: dict[str, Any] | None = None
        suite_task_observation: dict[str, Any] | None = None
        suite_task_terminal_records: list[dict[str, Any]] | None = None
        temporary_import_path: Path | None = None
        temporary_import_policy: dict[str, Any] | None = None
        temporary_import_observation: dict[str, Any] | None = None
        if recovery is not None:
            # This guard lives in the public worker API, not just the CLI or
            # parent launcher.  No copied source, manifest, or dependency code
            # is inspected before the immutable -I/-S/-B and sealed-runtime
            # bootstrap authority has been reconstructed.
            _require_isolated_recovery_runtime()
            if (
                isinstance(attestation_descriptor, bool)
                or not isinstance(attestation_descriptor, int)
                or isinstance(acknowledgement_descriptor, bool)
                or not isinstance(acknowledgement_descriptor, int)
                or attestation_descriptor < 0
                or acknowledgement_descriptor < 0
                or len(
                    {
                        receipt_descriptor,
                        attestation_descriptor,
                        acknowledgement_descriptor,
                    }
                )
                != 3
                or not isinstance(attestation_nonce, str)
                or re.fullmatch(r"[0-9a-f]{64}", attestation_nonce) is None
            ):
                raise QualificationError("recovery worker attestation capability differs")
            os.set_inheritable(attestation_descriptor, False)
            os.set_inheritable(acknowledgement_descriptor, False)
            (
                runtime_bundle,
                runtime_bootstrap,
                runtime_root,
                runtime_policy_binding,
            ) = (
                _worker_qualification_runtime_state()
            )
            _load_bound_native_host_objects(
                runtime_bundle,
                task_id=recovery.task_id,
            )
        worker_root = ROOT if execution_root is None else execution_root.resolve(strict=True)
        if execution_root is not None:
            worker_script_root = ROOT.resolve()
            if recovery is None and worker_root == worker_script_root:
                raise QualificationError("worker execution root is not isolated")
            if recovery is not None and worker_root != worker_script_root:
                raise QualificationError(
                    "recovery worker did not execute from its copied projection"
                )
        if execution_root is not None and write_root is None:
            raise QualificationError("isolated worker has no write root")
        writable_root = write_root.resolve(strict=True) if write_root is not None else None
        worker_pycache_identity: tuple[int, int] | None = None
        worker_pycache: dict[str, Any] | None = None
        if recovery is not None:
            if writable_root is None:
                raise QualificationError("isolated worker has no write root")
            if execution_projection is None:
                raise QualificationError("recovery worker projection is absent")
            execution_projection = _validate_worker_execution_projection(
                execution_projection,
                worker_root=worker_root,
                suite=suite,
            )
            temporary_import_path, temporary_import_policy = (
                _legal_data_import_path_policy(
                    worker_root,
                    suite,
                    execution_projection,
                )
            )
            worker_pycache_identity, worker_pycache = (
                _recovery_worker_pycache_state(writable_root)
            )
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
                **_install_candidate_sandbox(
                    writable_root,
                    bind_seccomp_policy=recovery is not None,
                    permit_git_helpers=(
                        recovery is None and not suite.candidate_authored
                    ),
                ),
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
        if recovery is not None:
            recovery_file_size = _lower_resource_limit(
                resource.RLIMIT_FSIZE, _MAX_WORKER_TRANSCRIPT_BYTES
            )
            isolation = {
                **isolation,
                "resource_limits": {
                    **dict(isolation["resource_limits"]),
                    "file_size_bytes": recovery_file_size,
                },
            }
        if (
            isolation.get("checkout_write_permitted") is not False
            or isolation.get("network_permitted") is not False
        ):
            raise QualificationError("worker sandbox is not fail-closed")
        recorder = _Recorder()
        capture_budget = [_MAX_WORKER_TRANSCRIPT_BYTES]
        captured_out: io.StringIO = (
            _BoundedTextCapture(capture_budget)
            if recovery is not None
            else io.StringIO()
        )
        captured_err: io.StringIO = (
            _BoundedTextCapture(capture_budget)
            if recovery is not None
            else io.StringIO()
        )
        started = time.monotonic_ns()
        previous = Path.cwd()
        required_import_paths = [] if recovery is not None else [str(owner)]
        inserted_import_paths: list[str] = []
        provider_guard = (
            _RecoveryProviderGuard(retain_until_process_exit=True)
            if recovery is not None
            else None
        )
        guard_context: Any = (
            provider_guard if provider_guard is not None else contextlib.nullcontext()
        )
        try:
            os.chdir(owner)
            for import_path in reversed(required_import_paths):
                if import_path not in sys.path:
                    sys.path.insert(0, import_path)
                    inserted_import_paths.append(import_path)
            with guard_context:
                sealed_worker_path = tuple(sys.path)
                if temporary_import_path is not None:
                    if str(temporary_import_path) in sys.path_importer_cache:
                        raise QualificationError(
                            "temporary qualification import path cache is preloaded"
                        )
                    sys.path.insert(1, str(temporary_import_path))
                execution_worker_path = tuple(sys.path)
                if (
                    execution_worker_path[0] != sealed_worker_path[0]
                    or (
                        temporary_import_path is not None
                        and execution_worker_path
                        != (
                            sealed_worker_path[0],
                            str(temporary_import_path),
                            *sealed_worker_path[1:],
                        )
                    )
                    or (
                        temporary_import_path is None
                        and execution_worker_path != sealed_worker_path
                    )
                ):
                    raise QualificationError(
                        "temporary qualification import path position differs"
                    )
                if recovery is not None:
                    if runtime_root is None or runtime_bundle is None:
                        raise QualificationError(
                            "qualification runtime projection is absent"
                        )
                    native_maps_before_solver = _native_executable_mappings()
                    native_load_guard = _RecoveryNativeLoadGuard(
                        runtime_root,
                        runtime_bundle,
                        z3_required=z3_required,
                    )
                    z3_import_denial_guard = _RecoveryZ3ImportDenialGuard(
                        task_id=recovery.task_id,
                        suite_id=suite.suite_id,
                        runtime_root=runtime_root,
                        recorder=recorder,
                        provider_guard=provider_guard,
                        source_projection_root=str(
                            execution_projection[
                                "copied_source_manifest_root"
                            ]
                        ),
                    )
                    z3_import_denial_guard.__enter__()
                    import cvc5  # noqa: F401
                    import pytest
                    z3_module: Any | None = None
                    if z3_required:
                        import z3

                        z3_module = z3

                    (
                        suite_task_observation,
                        suite_task_terminal_records,
                    ) = _recovery_suite_task_fixed_point(
                            recovery,
                            suite,
                            z3_module=z3_module,
                            runtime_cid=str(runtime_bundle["runtime_cid"]),
                            source_projection_root=str(
                                execution_projection[
                                    "copied_source_manifest_root"
                                ]
                            ),
                    )

                    runtime_import_evidence = _validate_runtime_module_origins(
                        runtime_root,
                        runtime_bundle,
                        z3_required=z3_required,
                    )
                    native_threads_before_pytest = (
                        _recovery_live_thread_population()
                    )
                    native_kernel_tasks_before_pytest = (
                        _recovery_kernel_task_population()
                    )
                    native_children_before_pytest = (
                        _recovery_child_process_population(
                            task_ids=native_kernel_tasks_before_pytest
                        )
                    )
                    prepared_native_load_evidence = native_load_guard.evidence(
                        before_maps=native_maps_before_solver,
                        after_maps=_native_executable_mappings(),
                        before_threads=native_threads_before_pytest,
                        after_threads=_recovery_live_thread_population(),
                        before_kernel_tasks=native_kernel_tasks_before_pytest,
                        after_kernel_tasks=_recovery_kernel_task_population(),
                        before_child_processes=native_children_before_pytest,
                        after_child_processes=_recovery_child_process_population(),
                        diagnostic_phase="prepared",
                    )
                    prepared_z3_import_denial_evidence = (
                        z3_import_denial_guard.evidence(phase="prepared")
                    )
                    prepared_attestation_evidence = _worker_attestation_phase(
                        phase="prepared",
                        attestation_descriptor=attestation_descriptor,
                        acknowledgement_descriptor=acknowledgement_descriptor,
                        launch_nonce=attestation_nonce,
                        suite_id=suite.suite_id,
                        runtime_cid=str(runtime_bundle["runtime_cid"]),
                        z3_required=z3_required,
                        suite_task_observation=suite_task_observation,
                        suite_task_terminal_records=(
                            suite_task_terminal_records
                        ),
                        native_load_guard=native_load_guard,
                        native_guard_evidence=prepared_native_load_evidence,
                        z3_import_denial_guard=z3_import_denial_guard,
                        z3_import_denial_evidence=(
                            prepared_z3_import_denial_evidence
                        ),
                        native_maps_before_solver=native_maps_before_solver,
                        native_threads_before_pytest=native_threads_before_pytest,
                        native_kernel_tasks_before_pytest=(
                            native_kernel_tasks_before_pytest
                        ),
                        native_children_before_pytest=(
                            native_children_before_pytest
                        ),
                    )
                else:
                    import pytest

                sealed_environ = dict(os.environ) if recovery is not None else None
                with tempfile.TemporaryDirectory(
                    prefix="lgcvf-pytest-cache-"
                ) as cache_dir:
                    if writable_root is None:
                        raise QualificationError("isolated worker has no writable root")
                    pytest_log = writable_root / "pytest.log"
                    with contextlib.redirect_stdout(
                        captured_out
                    ), contextlib.redirect_stderr(captured_err):
                        pytest_arguments = [
                            "-q",
                            "-ra",
                            "--maxfail=1",
                            "-o",
                            f"cache_dir={cache_dir}",
                            "-o",
                            f"log_file={pytest_log}",
                        ]
                        if recovery is not None and suite.owner_root == ".":
                            pytest_arguments.extend(
                                ("--rootdir", str(worker_root))
                            )
                        pytest_arguments.extend(suite.paths)
                        exit_code = int(
                            pytest.main(
                                pytest_arguments,
                                plugins=[recorder],
                            )
                        )
                    if recovery is not None:
                        if sealed_environ is None:
                            raise QualificationError(
                                "recovery worker sealed environment is absent"
                            )
                        os.environ.clear()
                        os.environ.update(sealed_environ)
                        if z3_import_denial_guard is None:
                            raise QualificationError(
                                "recovery Z3 import denial guard is absent"
                            )
                        z3_import_denial_guard.validate_pytest_return_meta_path()
                if recovery is not None:
                    if (
                        native_load_guard is None
                        or native_maps_before_solver is None
                        or native_threads_before_pytest is None
                        or native_kernel_tasks_before_pytest is None
                        or native_children_before_pytest is None
                    ):
                        raise QualificationError(
                            "recovery native load guard is absent"
                        )
                    (
                        quiescent_threads,
                        quiescent_tasks,
                        quiescent_children,
                    ) = _wait_for_recovery_execution_quiescence(
                        expected_threads=native_threads_before_pytest,
                        expected_tasks=native_kernel_tasks_before_pytest,
                        expected_children=native_children_before_pytest,
                    )
                    native_load_evidence = native_load_guard.evidence(
                        before_maps=native_maps_before_solver,
                        after_maps=_native_executable_mappings(),
                        before_threads=native_threads_before_pytest,
                        after_threads=quiescent_threads,
                        before_kernel_tasks=native_kernel_tasks_before_pytest,
                        after_kernel_tasks=quiescent_tasks,
                        before_child_processes=native_children_before_pytest,
                        after_child_processes=quiescent_children,
                        diagnostic_phase="post_pytest",
                    )
                if recovery is not None:
                    before_paths = list(execution_worker_path)
                    after_paths = list(sys.path)
                    added_paths = [
                        path
                        for path in after_paths
                        if path not in before_paths
                    ]
                    removed_paths = [
                        path
                        for path in before_paths
                        if path not in after_paths
                    ]
                    sealed_checkout = worker_root.resolve(strict=True)
                    ancestor_additions = True
                    for path in added_paths:
                        try:
                            added_root = Path(path).resolve(strict=True)
                            sealed_checkout.relative_to(added_root)
                        except (OSError, ValueError):
                            ancestor_additions = False
                            break
                    if removed_paths or not ancestor_additions:
                        raise QualificationError(
                            "pytest changed the sealed qualification import path: added="
                            + ",".join(dict.fromkeys(added_paths))
                            + "; removed="
                            + ",".join(dict.fromkeys(removed_paths))
                        )
                    sys.path[:] = before_paths
                if recovery is not None:
                    loaded_origins = _legal_data_loaded_module_origins(
                        worker_root,
                        suite,
                        temporary_import_path,
                    )
                    if temporary_import_path is not None:
                        if sys.path[1] != str(temporary_import_path):
                            raise QualificationError(
                                "temporary qualification import path moved"
                            )
                        del sys.path[1]
                        sys.path_importer_cache.pop(str(temporary_import_path), None)
                    if tuple(sys.path) != sealed_worker_path:
                        raise QualificationError(
                            "qualification import path was not restored"
                        )
                    if temporary_import_policy is None:
                        raise QualificationError(
                            "temporary qualification import policy is absent"
                        )
                    if z3_import_denial_guard is None:
                        raise QualificationError(
                            "recovery Z3 import denial guard is absent"
                        )
                    if (
                        native_threads_before_pytest is None
                        or native_kernel_tasks_before_pytest is None
                        or native_children_before_pytest is None
                    ):
                        raise QualificationError(
                            "recovery candidate population authority is absent"
                        )
                    z3_import_denial_guard.complete_candidate_execution(
                        expected_threads=native_threads_before_pytest,
                        expected_tasks=native_kernel_tasks_before_pytest,
                        expected_children=native_children_before_pytest,
                        pytest_exit_code=exit_code,
                        expected_sys_path=sealed_worker_path,
                        transcript_within_bound=(
                            len(
                                (
                                    captured_out.getvalue()
                                    + "\n"
                                    + captured_err.getvalue()
                                ).encode("utf-8", errors="replace")
                            )
                            <= _MAX_WORKER_TRANSCRIPT_BYTES
                        ),
                        loaded_origins_captured=isinstance(
                            loaded_origins, list
                        ),
                        importer_cache_entry_cleared=(
                            temporary_import_path is None
                            or str(temporary_import_path)
                            not in sys.path_importer_cache
                        ),
                    )
                    with (
                        z3_import_denial_guard.trusted_source_revalidation()
                    ):
                        (
                            after_temporary_path,
                            after_temporary_policy,
                        ) = _legal_data_import_path_policy(
                            worker_root,
                            suite,
                            execution_projection,
                            source_revalidation_guard=(
                                z3_import_denial_guard
                            ),
                        )
                        if (
                            after_temporary_path != temporary_import_path
                            or after_temporary_policy
                            != temporary_import_policy
                        ):
                            raise QualificationError(
                                "temporary qualification import policy changed"
                            )
                        temporary_import_observation = {
                            "schema": (
                                "lgcvf-recovery-temporary-import-path-observation@1"
                            ),
                            "policy": temporary_import_policy,
                            "loaded_module_origins": loaded_origins,
                            "loaded_module_origin_root": content_identity(
                                loaded_origins
                            ),
                            "importer_cache_entry_cleared": (
                                temporary_import_path is None
                                or str(temporary_import_path)
                                not in sys.path_importer_cache
                            ),
                            "sealed_path_restored": True,
                        }
                        temporary_import_observation["observation_cid"] = (
                            content_identity(temporary_import_observation)
                        )
        finally:
            for import_path in inserted_import_paths:
                try:
                    sys.path.remove(import_path)
                except ValueError:
                    pass
            os.chdir(previous)
        duration_ms = max(0, (time.monotonic_ns() - started) // 1_000_000)
        if recovery is not None:
            if worker_pycache_identity is None or worker_pycache is None:
                raise QualificationError("recovery worker pycache evidence is absent")
            after_identity, after_pycache = _recovery_worker_pycache_state(
                writable_root,
                expected_identity=worker_pycache_identity,
            )
            if after_identity != worker_pycache_identity or after_pycache != worker_pycache:
                raise QualificationError("recovery worker pycache evidence differs")
            isolation = {
                **isolation,
                "worker_pycache": {
                    **worker_pycache,
                    "empty_before": True,
                    "empty_after": True,
                },
            }
            if z3_import_denial_guard is None:
                raise QualificationError("recovery Z3 import denial guard is absent")
            with z3_import_denial_guard.trusted_runtime_revalidation():
                (
                    after_bundle,
                    after_bootstrap,
                    after_runtime_root,
                    after_policy_binding,
                ) = _worker_qualification_runtime_state()
                after_import_evidence = _validate_runtime_module_origins(
                    after_runtime_root,
                    after_bundle,
                    z3_required=z3_required,
                )
            final_z3_import_denial_evidence = (
                z3_import_denial_guard.evidence(phase="final")
            )
            before_modules = {
                (item["name"], item["origin"])
                for item in runtime_import_evidence.get("module_origins", [])
                if isinstance(item, Mapping)
                and isinstance(item.get("name"), str)
                and isinstance(item.get("origin"), str)
            } if runtime_import_evidence is not None else set()
            after_modules = {
                (item["name"], item["origin"])
                for item in after_import_evidence.get("module_origins", [])
                if isinstance(item, Mapping)
                and isinstance(item.get("name"), str)
                and isinstance(item.get("origin"), str)
            }
            if (
                runtime_bundle is None
                or runtime_bootstrap is None
                or runtime_root is None
                or runtime_import_evidence is None
                or native_load_evidence is None
                or prepared_z3_import_denial_evidence is None
                or final_z3_import_denial_evidence is None
                or after_bundle != runtime_bundle
                or after_bootstrap != runtime_bootstrap
                or after_runtime_root != runtime_root
                or runtime_policy_binding is None
                or after_policy_binding != runtime_policy_binding
                or not before_modules
                or not before_modules.issubset(after_modules)
                or after_import_evidence.get("native_mapping")
                != runtime_import_evidence.get("native_mapping")
            ):
                raise QualificationError(
                    "qualification runtime changed during worker execution"
                )
        transcript = (captured_out.getvalue() + "\n" + captured_err.getvalue()).encode(
            "utf-8", errors="replace"
        )
        transcript_within_bound = len(transcript) <= _MAX_WORKER_TRANSCRIPT_BYTES
        provider_imports = (
            provider_guard.imported_modules() if provider_guard is not None else []
        )
        provider_import_attempts = (
            sorted(set(provider_guard.import_attempts))
            if provider_guard is not None
            else []
        )
        provider_process_attempts = (
            sorted(set(provider_guard.process_attempts))
            if provider_guard is not None
            else []
        )
        passed = (
            exit_code == 0
            and recorder.collected > 0
            and recorder.passed == recorder.collected
            and transcript_within_bound
            and not provider_imports
            and not provider_import_attempts
            and not provider_process_attempts
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
            "schema": worker_schema,
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
        if recovery is not None:
            if (
                runtime_bundle is None
                or runtime_bootstrap is None
                or runtime_import_evidence is None
                or runtime_policy_binding is None
                or temporary_import_observation is None
                or native_load_evidence is None
                or suite_task_observation is None
                or suite_task_terminal_records is None
            ):
                raise QualificationError("qualification runtime receipt is absent")
            if (
                native_load_guard is None
                or native_maps_before_solver is None
                or native_threads_before_pytest is None
                or native_kernel_tasks_before_pytest is None
                or native_children_before_pytest is None
                or z3_import_denial_guard is None
            ):
                raise QualificationError("qualification native closure is absent")
            final_native_load_evidence = native_load_guard.evidence(
                before_maps=native_maps_before_solver,
                after_maps=_native_executable_mappings(),
                before_threads=native_threads_before_pytest,
                after_threads=_recovery_live_thread_population(),
                before_kernel_tasks=native_kernel_tasks_before_pytest,
                after_kernel_tasks=_recovery_kernel_task_population(),
                before_child_processes=native_children_before_pytest,
                after_child_processes=_recovery_child_process_population(),
                diagnostic_phase="pre_final_attestation",
            )
            if final_native_load_evidence != native_load_evidence:
                raise QualificationError(
                    "qualification native closure changed after pytest"
                )
            native_load_evidence = final_native_load_evidence
            if prepared_attestation_evidence is None:
                raise QualificationError(
                    "qualification prepared attestation is absent"
                )
            final_attestation_evidence = _worker_attestation_phase(
                phase="final",
                attestation_descriptor=attestation_descriptor,
                acknowledgement_descriptor=acknowledgement_descriptor,
                launch_nonce=attestation_nonce,
                suite_id=suite.suite_id,
                runtime_cid=str(runtime_bundle["runtime_cid"]),
                z3_required=z3_required,
                suite_task_observation=suite_task_observation,
                suite_task_terminal_records=suite_task_terminal_records,
                native_load_guard=native_load_guard,
                native_guard_evidence=native_load_evidence,
                z3_import_denial_guard=z3_import_denial_guard,
                z3_import_denial_evidence=final_z3_import_denial_evidence,
                native_maps_before_solver=native_maps_before_solver,
                native_threads_before_pytest=native_threads_before_pytest,
                native_kernel_tasks_before_pytest=native_kernel_tasks_before_pytest,
                native_children_before_pytest=native_children_before_pytest,
            )
            os.close(attestation_descriptor)
            os.close(acknowledgement_descriptor)
            attestation_barrier_evidence = (
                _worker_attestation_barrier_evidence(
                    prepared_attestation_evidence,
                    final_attestation_evidence,
                )
            )
            if attestation_barrier_evidence is None:
                raise QualificationError("qualification attestation barrier is absent")
            payload.update(
                {
                    "task_id": recovery.task_id,
                    "task_cid": recovery.task_cid,
                    "validation_spec": recovery.validation_spec(),
                    "execution_api": "pytest.main",
                    "pytest_control_argv": [
                        "-q",
                        "-ra",
                        "--maxfail=1",
                    ],
                    "cache_reused": False,
                    "transcript_size_bytes": len(transcript),
                    "transcript_limit_bytes": _MAX_WORKER_TRANSCRIPT_BYTES,
                    "provider_policy": _RecoveryProviderGuard.policy(),
                    "provider_imports_observed": provider_imports,
                    "provider_import_attempts": provider_import_attempts,
                    "provider_process_attempts": provider_process_attempts,
                    "candidate_authored": True,
                    "self_authority": False,
                    "completion_authoritative": False,
                    "readonly_projection": execution_projection,
                    "qualification_runtime_cid": runtime_bundle["runtime_cid"],
                    "qualification_runtime_projection_cid": runtime_bootstrap[
                        "projection_cid"
                    ],
                    "qualification_runtime_component_root": content_identity(
                        runtime_bundle["components"]
                    ),
                    "qualification_runtime_file_manifest_root": runtime_bundle[
                        "file_manifest_root"
                    ],
                    "qualification_runtime_bootstrap": runtime_bootstrap,
                    "qualification_runtime_policy_binding": (
                        runtime_policy_binding
                    ),
                    "qualification_runtime_module_origin_root": (
                        after_import_evidence["module_origin_root"]
                    ),
                    "qualification_runtime_module_origins": (
                        after_import_evidence["module_origins"]
                    ),
                    "qualification_runtime_native_mapping": (
                        after_import_evidence["native_mapping"]
                    ),
                    "qualification_runtime_native_load_guard": (
                        native_load_evidence
                    ),
                    "qualification_runtime_suite_task_full_live_observation_cid": (
                        suite_task_observation["observation_cid"]
                    ),
                    "qualification_runtime_suite_task_receipt_cid": (
                        attestation_barrier_evidence[
                            "suite_task_receipt_cid"
                        ]
                    ),
                    "qualification_runtime_parent_attestation_barrier_cid": (
                        attestation_barrier_evidence["barrier_cid"]
                    ),
                    "qualification_runtime_z3_import_denial_prepared_cid": (
                        attestation_barrier_evidence[
                            "prepared_z3_import_denial_evidence_cid"
                        ]
                    ),
                    "qualification_runtime_z3_import_denial_final_cid": (
                        attestation_barrier_evidence[
                            "final_z3_import_denial_evidence_cid"
                        ]
                    ),
                    "qualification_runtime_temporary_import_path": (
                        temporary_import_observation
                    ),
                }
            )
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
                "schema": worker_schema,
                "suite_id": suite_id,
                "error": type(exc).__name__,
                "reason": str(exc)[:1000],
            },
        )
        return 2


class _BoundedReceiptPipeDrain:
    """Drain a worker's capability pipe concurrently without unbounded memory."""

    def __init__(
        self,
        descriptor: int,
        *,
        maximum_bytes: int = _MAX_WORKER_RECEIPT_BYTES,
    ) -> None:
        self._descriptor = descriptor
        self._maximum_bytes = maximum_bytes
        self._payload = bytearray()
        self._overflow = False
        self._error: BaseException | None = None
        self._thread = Thread(
            target=self._drain,
            name="lgcvf-recovery-receipt-drain",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def _drain(self) -> None:
        observed = 0
        try:
            while True:
                chunk = os.read(self._descriptor, 16 * 1024)
                if not chunk:
                    break
                observed += len(chunk)
                if observed > self._maximum_bytes:
                    # Continue draining so an oversized or compromised worker
                    # cannot block on a full pipe before the parent rejects it.
                    self._overflow = True
                    continue
                self._payload.extend(chunk)
        except BaseException as exc:  # delivered to and rejected by the parent
            self._error = exc
        finally:
            try:
                os.close(self._descriptor)
            except OSError:
                pass

    def finish(self, *, timeout_seconds: int = 30) -> bytes:
        self._thread.join(timeout_seconds)
        if self._thread.is_alive():
            try:
                os.close(self._descriptor)
            except OSError:
                pass
            self._thread.join(1)
            raise QualificationError("worker receipt pipe did not close within its bound")
        if self._error is not None:
            raise QualificationError("worker receipt pipe read failed") from self._error
        if self._overflow:
            raise QualificationError("worker receipt exceeds its pipe bound")
        return bytes(self._payload)


def _abort_spawned_worker_setup(
    process: subprocess.Popen[Any],
    *,
    descriptors: Sequence[int],
    pidfd: int | None,
    stream_handles: Sequence[Any],
) -> None:
    """Kill/reap a just-spawned worker and consume all parent-owned handles."""

    cleanup_errors: list[BaseException] = []
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except BaseException as exc:
        cleanup_errors.append(exc)
    try:
        process.wait(timeout=30)
    except BaseException as exc:
        cleanup_errors.append(exc)
    unique_descriptors: list[int] = []
    for descriptor in descriptors:
        if (
            isinstance(descriptor, bool)
            or not isinstance(descriptor, int)
            or descriptor < 0
            or descriptor in unique_descriptors
        ):
            cleanup_errors.append(
                QualificationError("worker setup descriptor ownership differs")
            )
            continue
        unique_descriptors.append(descriptor)
    if pidfd is not None:
        if (
            isinstance(pidfd, bool)
            or not isinstance(pidfd, int)
            or pidfd < 0
            or pidfd in unique_descriptors
        ):
            cleanup_errors.append(
                QualificationError("worker setup pidfd ownership differs")
            )
        else:
            unique_descriptors.append(pidfd)
    for descriptor in unique_descriptors:
        try:
            os.close(descriptor)
        except BaseException as exc:
            cleanup_errors.append(exc)
    unique_handles: list[Any] = []
    for handle in (
        *stream_handles,
        process.stdout,
        process.stderr,
    ):
        if handle is None or any(handle is item for item in unique_handles):
            continue
        unique_handles.append(handle)
        try:
            handle.close()
        except BaseException as exc:
            cleanup_errors.append(exc)
    if process.returncode is None:
        cleanup_errors.append(
            QualificationError("worker setup failure left an unreaped process")
        )
    if cleanup_errors:
        raise QualificationError("worker setup cleanup failed") from cleanup_errors[0]


def _bounded_worker_stream(path: Path, *, suite_id: str, label: str) -> str:
    """Read a recovery worker stream only after enforcing its hard bound."""

    try:
        size = path.stat().st_size
    except OSError as exc:
        raise QualificationError(f"{suite_id}: {label} stream is unavailable") from exc
    if size >= _MAX_WORKER_TRANSCRIPT_BYTES:
        raise QualificationError(f"{suite_id}: {label} stream reached its hard bound")
    try:
        return path.read_bytes().decode("utf-8", errors="replace")
    except OSError as exc:
        raise QualificationError(f"{suite_id}: {label} stream is unreadable") from exc


def _run_suite(
    suite: Suite,
    *,
    expected_manifest: Mapping[str, Any],
    root: Path = ROOT,
    qualification_runtime: _ActiveQualificationRuntime | None = None,
) -> dict[str, Any]:
    recovery = _RECOVERY_BY_SUITE_ID.get(suite.suite_id)
    expected_worker_schema = (
        RECOVERY_WORKER_SCHEMA if recovery is not None else WORKER_SCHEMA
    )
    source_root = root.resolve(strict=True)
    qualification_policy_binding: dict[str, Any] | None = None
    controller_wx_observations: list[dict[str, Any]] = []
    if recovery is not None:
        if qualification_runtime is None:
            raise QualificationError("qualification runtime is required")
        expected_runtime_cid, qualification_policy_binding = (
            _recovery_qualification_policy(source_root, head_bound=True)
        )
        if (
            qualification_runtime.resolved.bundle.get("runtime_cid")
            != expected_runtime_cid
        ):
            raise QualificationError("qualification runtime policy differs")
        _validate_active_qualification_runtime(qualification_runtime)
        controller_wx_observations.append(
            _controller_zero_wx_observation(phase="before_worker")
        )
        dependency_paths: tuple[str, ...] = ()
    else:
        dependency_paths = tuple(
            dict.fromkeys(
                str(resolved)
                for entry in sys.path
                if entry
                and Path(entry).is_dir()
                and not (
                    (resolved := Path(entry).resolve()).is_relative_to(source_root)
                )
            )
        )
    if recovery is None and not dependency_paths:
        raise QualificationError("qualified Python dependency path is unavailable")
    with tempfile.TemporaryDirectory(prefix="lgcvf-qualification-sandbox-") as sandbox:
        sandbox_path = Path(sandbox)
        checkout = sandbox_path / "checkout"
        writable = sandbox_path / "writable"
        writable.mkdir(mode=0o700)
        execution_projection = (
            _prepare_recovery_execution_checkout(root, checkout, (suite,))
            if recovery is not None
            else _prepare_execution_checkout(root, checkout, (suite,))
        )
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
        if recovery is not None:
            environment = _normalized_recovery_worker_environment(writable)
        else:
            projected_import_paths = (
                str(checkout),
                str(checkout / "ipfs_datasets_py"),
            )
            worker_pycache = writable / "python-pycache"
            worker_pycache.mkdir(mode=0o700)
            environment = {
                "HOME": str(home_path),
                "IPFS_DATASETS_PY_MINIMAL_IMPORTS": "1",
                "LANG": "C.UTF-8",
                "NO_COLOR": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin",
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPYCACHEPREFIX": str(worker_pycache),
                "PYTHONPATH": os.pathsep.join(
                    (*projected_import_paths, *dependency_paths)
                ),
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                "PYTHONHASHSEED": "0",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "TMPDIR": str(writable),
                **{key: str(path) for key, path in xdg_paths.items()},
            }
        setup_descriptors: list[int] = []
        setup_stream_handles: list[Any] = []
        receipt_read: int
        receipt_write: int
        attestation_read: int | None = None
        attestation_write: int | None = None
        acknowledgement_read: int | None = None
        acknowledgement_write: int | None = None
        attestation_nonce: str | None = None
        stdout_path = writable / "worker.stdout"
        stderr_path = writable / "worker.stderr"
        stdout_handle: Any | None = None
        stderr_handle: Any | None = None
        try:
            receipt_read, receipt_write = os.pipe()
            setup_descriptors.extend((receipt_read, receipt_write))
            if recovery is not None:
                attestation_read, attestation_write = os.pipe()
                setup_descriptors.extend((attestation_read, attestation_write))
                acknowledgement_read, acknowledgement_write = os.pipe()
                setup_descriptors.extend(
                    (acknowledgement_read, acknowledgement_write)
                )
                attestation_nonce = os.urandom(32).hex()
                stdout_handle = stdout_path.open("wb")
                setup_stream_handles.append(stdout_handle)
                stderr_handle = stderr_path.open("wb")
                setup_stream_handles.append(stderr_handle)
        except BaseException as exc:
            cleanup_errors: list[BaseException] = []
            for descriptor in reversed(setup_descriptors):
                try:
                    os.close(descriptor)
                except BaseException as cleanup_exc:
                    cleanup_errors.append(cleanup_exc)
            for handle in reversed(setup_stream_handles):
                try:
                    handle.close()
                except BaseException as cleanup_exc:
                    cleanup_errors.append(cleanup_exc)
            if cleanup_errors:
                raise QualificationError(
                    "worker capability acquisition and cleanup failed"
                ) from cleanup_errors[0]
            raise QualificationError("worker capability acquisition failed") from exc
        try:
            qualifier_path = (
                checkout
                / "scripts/qualify_logic_governed_compositional_verification_fabric.py"
            ).resolve(strict=True)
            if recovery is not None:
                if qualification_runtime is None:  # already checked above
                    raise QualificationError("qualification runtime is required")
                encoded_projection = base64.urlsafe_b64encode(
                    json.dumps(
                        execution_projection,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).decode("ascii")
                encoded_runtime_projection = base64.urlsafe_b64encode(
                    json.dumps(
                        qualification_runtime.projection,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).decode("ascii")
                worker_arguments = [
                    "/usr/bin/python3.12",
                    "-I",
                    "-S",
                    "-B",
                    "-c",
                    _RECOVERY_WORKER_BOOTSTRAP,
                    str(qualification_runtime.root),
                    str(qualification_runtime.root_fd),
                    str(qualification_runtime.resolved.bundle["runtime_cid"]),
                    encoded_runtime_projection,
                    str(checkout.resolve(strict=True)),
                    recovery.owner_root,
                    str(writable.resolve(strict=True)),
                    str(qualifier_path),
                    "--worker",
                    suite.suite_id,
                    "--worker-root",
                    str(checkout.resolve(strict=True)),
                    "--worker-write-root",
                    str(writable.resolve(strict=True)),
                    "--worker-receipt-fd",
                    str(receipt_write),
                    "--worker-attestation-fd",
                    str(attestation_write),
                    "--worker-ack-fd",
                    str(acknowledgement_read),
                    "--worker-attestation-nonce",
                    str(attestation_nonce),
                    "--worker-projection",
                    encoded_projection,
                ]
            else:
                worker_arguments = [
                    sys.executable,
                    str(qualifier_path),
                    "--worker",
                    suite.suite_id,
                    "--worker-root",
                    str(checkout),
                    "--worker-write-root",
                    str(writable),
                    "--worker-receipt-fd",
                    str(receipt_write),
                ]
            process = subprocess.Popen(
                worker_arguments,
                cwd=checkout,
                env=environment,
                stdout=stdout_handle if stdout_handle is not None else subprocess.PIPE,
                stderr=stderr_handle if stderr_handle is not None else subprocess.PIPE,
                text=True,
                start_new_session=True,
                pass_fds=(
                    (
                        receipt_write,
                        qualification_runtime.root_fd,
                        attestation_write,
                        acknowledgement_read,
                    )
                    if recovery is not None
                    and qualification_runtime is not None
                    and attestation_write is not None
                    and acknowledgement_read is not None
                    else (receipt_write,)
                ),
            )
        except BaseException as exc:
            cleanup_errors = []
            for descriptor in reversed(setup_descriptors):
                try:
                    os.close(descriptor)
                except BaseException as cleanup_exc:
                    cleanup_errors.append(cleanup_exc)
            for handle in reversed(setup_stream_handles):
                try:
                    handle.close()
                except BaseException as cleanup_exc:
                    cleanup_errors.append(cleanup_exc)
            if cleanup_errors:
                raise QualificationError(
                    "worker spawn admission and cleanup failed"
                ) from cleanup_errors[0]
            raise QualificationError("worker spawn admission failed") from exc
        post_spawn_owned_descriptors = set(setup_descriptors)

        def close_post_spawn_owned(descriptor: int) -> None:
            if descriptor not in post_spawn_owned_descriptors:
                raise QualificationError("worker descriptor ownership differs")
            os.close(descriptor)
            post_spawn_owned_descriptors.remove(descriptor)

        try:
            close_post_spawn_owned(receipt_write)
        except BaseException as exc:
            try:
                _abort_spawned_worker_setup(
                    process,
                    descriptors=tuple(sorted(post_spawn_owned_descriptors)),
                    pidfd=None,
                    stream_handles=(stdout_handle, stderr_handle),
                )
            except BaseException as cleanup_exc:
                raise QualificationError(
                    "worker descriptor transfer and cleanup failed"
                ) from cleanup_exc
            raise QualificationError("worker descriptor transfer failed") from exc
        pidfd: int | None = None
        if recovery is not None:
            if (
                attestation_read is None
                or attestation_write is None
                or acknowledgement_read is None
                or acknowledgement_write is None
                or attestation_nonce is None
                or qualification_runtime is None
            ):
                try:
                    _abort_spawned_worker_setup(
                        process,
                        descriptors=tuple(
                            sorted(post_spawn_owned_descriptors)
                        ),
                        pidfd=None,
                        stream_handles=(stdout_handle, stderr_handle),
                    )
                except BaseException as cleanup_exc:
                    raise QualificationError(
                        "worker attestation state and cleanup differ"
                    ) from cleanup_exc
                raise QualificationError("worker attestation pipes are absent")
            try:
                pidfd = os.pidfd_open(process.pid, 0)
                os.set_inheritable(pidfd, False)
            except (AttributeError, OSError) as exc:
                try:
                    _abort_spawned_worker_setup(
                        process,
                        descriptors=tuple(sorted(post_spawn_owned_descriptors)),
                        pidfd=pidfd,
                        stream_handles=(stdout_handle, stderr_handle),
                    )
                except BaseException as cleanup_exc:
                    raise QualificationError(
                        "worker pidfd admission and cleanup failed"
                    ) from cleanup_exc
                raise QualificationError("worker pidfd is unavailable") from exc
            try:
                close_post_spawn_owned(attestation_write)
                close_post_spawn_owned(acknowledgement_read)
            except BaseException as exc:
                try:
                    _abort_spawned_worker_setup(
                        process,
                        descriptors=tuple(
                            sorted(post_spawn_owned_descriptors)
                        ),
                        pidfd=pidfd,
                        stream_handles=(stdout_handle, stderr_handle),
                    )
                except BaseException as cleanup_exc:
                    raise QualificationError(
                        "worker attestation transfer and cleanup failed"
                    ) from cleanup_exc
                raise QualificationError(
                    "worker attestation transfer failed"
                ) from exc
        receipt_drain = _BoundedReceiptPipeDrain(receipt_read)
        try:
            receipt_drain.start()
        except BaseException as exc:
            try:
                _abort_spawned_worker_setup(
                    process,
                    descriptors=tuple(sorted(post_spawn_owned_descriptors)),
                    pidfd=pidfd,
                    stream_handles=(stdout_handle, stderr_handle),
                )
            except BaseException as cleanup_exc:
                raise QualificationError(
                    "worker receipt-drain admission and cleanup failed"
                ) from cleanup_exc
            raise QualificationError("worker receipt drain failed to start") from exc
        if receipt_read not in post_spawn_owned_descriptors:
            raise QualificationError("worker receipt descriptor transfer differs")
        post_spawn_owned_descriptors.remove(receipt_read)
        # The surviving recovery pipe ends are transferred to the barrier
        # coordinator below.  Ordinary workers have no additional pipe ends.
        expected_transferred = (
            {attestation_read, acknowledgement_write}
            if recovery is not None
            else set()
        )
        if post_spawn_owned_descriptors != expected_transferred:
            raise QualificationError("worker pipe ownership transfer differs")
        post_spawn_owned_descriptors.clear()
        timed_out = False
        terminal_pidfd_signaled = recovery is None
        terminal_process_group_empty = recovery is None
        terminal_process_group_probe_absent = recovery is None
        wall_timeout = (
            recovery.validation_spec()["timeout_seconds"]
            if recovery is not None
            else 1800
        )
        parent_started = time.monotonic()
        barrier_error: BaseException | None = None
        expected_attestation_barrier: dict[str, Any] | None = None
        expected_suite_task_receipt: dict[str, Any] | None = None
        held_task_directory_lease: list[dict[str, Any]] = []
        if recovery is not None:
            if (
                attestation_read is None
                or acknowledgement_write is None
                or attestation_nonce is None
                or pidfd is None
                or qualification_runtime is None
            ):
                raise QualificationError("worker attestation parent state is absent")
            try:
                prepared_attestation = _read_bounded_json_line(
                    attestation_read,
                    noun=f"{suite.suite_id}: prepared worker attestation",
                    timeout_seconds=max(
                        0.001,
                        wall_timeout - (time.monotonic() - parent_started),
                    ),
                )
                prepared_controller_observation = (
                    _controller_zero_wx_observation(
                        phase="prepared_parent_inspection"
                    )
                )
                controller_wx_observations.append(
                    prepared_controller_observation
                )
                prepared_parent_observation, prepared_exact_state = (
                    _parent_worker_live_observation(
                        prepared_attestation,
                        phase="prepared",
                        launch_nonce=attestation_nonce,
                        suite_id=suite.suite_id,
                        process_id=process.pid,
                        pidfd=pidfd,
                        runtime=qualification_runtime,
                        source_projection_root=str(
                            execution_projection["copied_source_manifest_root"]
                        ),
                        z3_required=recovery.task_id
                        in _RECOVERY_Z3_REQUIRED_TASKS,
                        prepared_exact_state=None,
                        controller_observation=prepared_controller_observation,
                    )
                )
                transferred_lease = prepared_exact_state.get(
                    "held_task_directories"
                )
                if (
                    held_task_directory_lease
                    or not isinstance(transferred_lease, list)
                    or not transferred_lease
                ):
                    raise QualificationError(
                        f"{suite.suite_id}: prepared task lease differs"
                    )
                held_task_directory_lease = transferred_lease
                derived_task_receipt = prepared_exact_state.get(
                    "compact_task_receipt"
                )
                if not isinstance(derived_task_receipt, Mapping):
                    raise QualificationError(
                        f"{suite.suite_id}: compact task receipt is absent"
                    )
                expected_suite_task_receipt = dict(derived_task_receipt)
                prepared_ack = _parent_worker_ack(
                    prepared_attestation,
                    prepared_parent_observation,
                )
                _write_bounded_json_line(
                    acknowledgement_write,
                    prepared_ack,
                    noun=f"{suite.suite_id}: prepared parent acknowledgement",
                )
                final_attestation = _read_bounded_json_line(
                    attestation_read,
                    noun=f"{suite.suite_id}: final worker attestation",
                    timeout_seconds=max(
                        0.001,
                        wall_timeout - (time.monotonic() - parent_started),
                    ),
                )
                final_controller_observation = _controller_zero_wx_observation(
                    phase="final_parent_inspection"
                )
                controller_wx_observations.append(final_controller_observation)
                final_parent_observation, _final_exact_state = (
                    _parent_worker_live_observation(
                        final_attestation,
                        phase="final",
                        launch_nonce=attestation_nonce,
                        suite_id=suite.suite_id,
                        process_id=process.pid,
                        pidfd=pidfd,
                        runtime=qualification_runtime,
                        source_projection_root=str(
                            execution_projection["copied_source_manifest_root"]
                        ),
                        z3_required=recovery.task_id
                        in _RECOVERY_Z3_REQUIRED_TASKS,
                        prepared_exact_state=prepared_exact_state,
                        controller_observation=final_controller_observation,
                    )
                )
                final_ack = _parent_worker_ack(
                    final_attestation,
                    final_parent_observation,
                )
                _write_bounded_json_line(
                    acknowledgement_write,
                    final_ack,
                    noun=f"{suite.suite_id}: final parent acknowledgement",
                )
                expected_attestation_barrier = (
                    _worker_attestation_barrier_evidence(
                        _public_attestation_phase_evidence(
                            prepared_attestation,
                            prepared_parent_observation,
                        ),
                        _public_attestation_phase_evidence(
                            final_attestation,
                            final_parent_observation,
                        ),
                    )
                )
                _require_pipe_eof(
                    attestation_read,
                    noun=f"{suite.suite_id}: worker attestation",
                    timeout_seconds=min(
                        _MAX_WORKER_ACK_SECONDS,
                        max(
                            0.001,
                            wall_timeout - (time.monotonic() - parent_started),
                        ),
                    ),
                )
            except BaseException as exc:
                barrier_error = exc
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            finally:
                for descriptor in (attestation_read, acknowledgement_write):
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
        terminal_inspection_error: BaseException | None = None
        held_task_lease_error: BaseException | None = None
        terminal_evidence: dict[str, Any] | None = None
        stdout_value: str | None = None
        stderr_value: str | None = None
        if recovery is not None:
            if pidfd is None:
                raise QualificationError("worker terminal pidfd is absent")
            try:
                terminal_evidence = _wait_and_inspect_recovery_worker_terminal(
                    process,
                    pidfd=pidfd,
                    deadline=parent_started + wall_timeout,
                )
                terminal_pidfd_signaled = bool(
                    terminal_evidence["pidfd_signaled_after_wait"]
                )
                terminal_process_group_empty = bool(
                    terminal_evidence["process_group_empty_after_exit"]
                )
                terminal_process_group_probe_absent = bool(
                    terminal_evidence["process_group_signal_probe_esrch"]
                )
            except subprocess.TimeoutExpired:
                timed_out = True
            except BaseException as exc:
                terminal_inspection_error = exc
            finally:
                try:
                    os.close(pidfd)
                except BaseException as exc:
                    if terminal_inspection_error is None:
                        terminal_inspection_error = exc
                try:
                    _close_held_recovery_task_directories(
                        held_task_directory_lease
                    )
                except BaseException as exc:
                    held_task_lease_error = exc
        else:
            try:
                stdout_value, stderr_value = process.communicate(
                    timeout=max(
                        0.001,
                        wall_timeout - (time.monotonic() - parent_started),
                    )
                )
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout_value, stderr_value = process.communicate(timeout=30)
        receipt = receipt_drain.finish()
        if stdout_handle is not None and stderr_handle is not None:
            stdout_handle.flush()
            stderr_handle.flush()
            stdout_handle.close()
            stderr_handle.close()
            stdout = _bounded_worker_stream(
                stdout_path, suite_id=suite.suite_id, label="stdout"
            )
            stderr = _bounded_worker_stream(
                stderr_path, suite_id=suite.suite_id, label="stderr"
            )
        else:
            stdout = str(stdout_value or "")
            stderr = str(stderr_value or "")
        barrier_child_diagnostic = ""
        if barrier_error is not None:
            try:
                diagnostic_text = receipt.decode("utf-8", errors="strict")
                diagnostic_payload = _strict_json_loads(
                    diagnostic_text[:-1],
                    noun=f"{suite.suite_id}: diagnostic worker receipt",
                )
                if (
                    receipt.endswith(b"\n")
                    and receipt.count(b"\n") == 1
                    and isinstance(diagnostic_payload, Mapping)
                    and _canonical_bytes(diagnostic_payload) + b"\n" == receipt
                    and diagnostic_payload.get("schema") == expected_worker_schema
                    and diagnostic_payload.get("suite_id") == suite.suite_id
                    and isinstance(diagnostic_payload.get("error"), str)
                    and isinstance(diagnostic_payload.get("reason"), str)
                ):
                    error_token = re.sub(
                        r"[^A-Za-z0-9_.-]",
                        "_",
                        str(diagnostic_payload["error"]),
                    )[:128]
                    reason_token = str(diagnostic_payload["reason"])[-1000:]
                    barrier_child_diagnostic = (
                        "; child=" + error_token + ":" + reason_token
                    )
            except (UnicodeDecodeError, json.JSONDecodeError, QualificationError):
                barrier_child_diagnostic = ""
        if timed_out:
            raise QualificationError(
                f"{suite.suite_id}: worker exceeded the {wall_timeout}-second wall bound; "
                + (stderr or stdout)[-1000:]
            )
        if barrier_error is not None:
            raise QualificationError(
                f"{suite.suite_id}: worker attestation failed: {barrier_error}; "
                + (stderr or stdout)[-1000:]
                + barrier_child_diagnostic
            ) from barrier_error
        if terminal_inspection_error is not None:
            raise QualificationError(
                f"{suite.suite_id}: worker terminal inspection failed"
            ) from terminal_inspection_error
        if held_task_lease_error is not None:
            raise QualificationError(
                f"{suite.suite_id}: worker task-directory lease cleanup failed"
            ) from held_task_lease_error
        returncode = int(process.returncode or 0)
        if recovery is not None and (
            terminal_evidence is None
            or not terminal_pidfd_signaled
            or not terminal_process_group_empty
            or not terminal_process_group_probe_absent
        ):
            raise QualificationError(
                f"{suite.suite_id}: worker terminal process state differs"
            )
        if recovery is not None:
            controller_wx_observations.append(
                _controller_zero_wx_observation(phase="after_worker_exit")
            )
        if recovery is not None:
            if qualification_runtime is None:
                raise QualificationError("qualification runtime is required")
            _validate_active_qualification_runtime(qualification_runtime)
            after_runtime_cid, after_policy_binding = (
                _recovery_qualification_policy(source_root, head_bound=True)
            )
            if (
                qualification_policy_binding is None
                or after_policy_binding != qualification_policy_binding
                or after_runtime_cid
                != qualification_runtime.resolved.bundle.get("runtime_cid")
            ):
                raise QualificationError(
                    "qualification runtime policy changed during worker execution"
                )
    try:
        receipt_text = receipt.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise QualificationError(f"{suite.suite_id}: worker receipt is not UTF-8") from exc
    if (
        not receipt.endswith(b"\n")
        or receipt.count(b"\n") != 1
        or not receipt[:-1]
    ):
        raise QualificationError(
            f"{suite.suite_id}: worker emitted no exact receipt "
            f"(returncode {returncode}): {(stderr or stdout)[-1000:]}"
        )
    try:
        payload = _strict_json_loads(
            receipt_text[:-1], noun=f"{suite.suite_id}: worker receipt"
        )
    except json.JSONDecodeError as exc:
        raise QualificationError(f"{suite.suite_id}: worker receipt is invalid") from exc
    if not isinstance(payload, dict) or payload.get("schema") != expected_worker_schema:
        raise QualificationError(f"{suite.suite_id}: worker schema differs")
    if _canonical_bytes(payload) + b"\n" != receipt:
        raise QualificationError(f"{suite.suite_id}: worker receipt is not canonical")
    if payload.get("suite_id") != suite.suite_id:
        raise QualificationError(f"{suite.suite_id}: worker identity differs")
    if "error" in payload:
        error = str(payload.get("error") or "unknown_error")
        reason = str(payload.get("reason") or "worker failed without a reason")
        attempted = payload.get("attempted_bytes")
        limit = payload.get("limit_bytes")
        bounded_size_detail = (
            f" (attempted {attempted} bytes; limit {limit} bytes)"
            if error == "receipt_too_large"
            and isinstance(attempted, int)
            and not isinstance(attempted, bool)
            and isinstance(limit, int)
            and not isinstance(limit, bool)
            and attempted > limit > 0
            else ""
        )
        raise QualificationError(
            f"{suite.suite_id}: worker failed closed: {error}: {reason[-1000:]}"
            + bounded_size_detail
        )
    if recovery is not None and set(payload) != _RECOVERY_WORKER_CHILD_FIELDS:
        raise QualificationError(
            f"{suite.suite_id}: worker terminal receipt fields differ"
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
    if recovery is not None and payload.get("readonly_projection") != execution_projection:
        raise QualificationError(f"{suite.suite_id}: read-only projection differs")
    if recovery is not None:
        if expected_attestation_barrier is None:
            raise QualificationError(
                f"{suite.suite_id}: parent attestation barrier is absent"
            )
        expected_attestation_barrier = (
            _parent_attestation_barrier_from_child_cid(
                payload.get(
                    "qualification_runtime_parent_attestation_barrier_cid"
                ),
                expected_attestation_barrier,
            )
        )
        if (
            payload.get(
                "qualification_runtime_z3_import_denial_prepared_cid"
            )
            != expected_attestation_barrier.get(
                "prepared_z3_import_denial_evidence_cid"
            )
            or payload.get(
                "qualification_runtime_z3_import_denial_final_cid"
            )
            != expected_attestation_barrier.get(
                "final_z3_import_denial_evidence_cid"
            )
        ):
            raise QualificationError(
                f"{suite.suite_id}: Z3 import denial CID linkage differs"
            )
    if recovery is not None:
        if expected_suite_task_receipt is None:
            raise QualificationError("parent-derived task receipt is absent")
        if (
            payload.get(
                "qualification_runtime_suite_task_full_live_observation_cid"
            )
            != expected_suite_task_receipt["full_live_observation_cid"]
            or payload.get(
                "qualification_runtime_suite_task_receipt_cid"
            )
            != expected_suite_task_receipt["receipt_cid"]
            or expected_attestation_barrier.get("full_live_observation_cid")
            != expected_suite_task_receipt["full_live_observation_cid"]
            or expected_attestation_barrier.get("suite_task_receipt_cid")
            != expected_suite_task_receipt["receipt_cid"]
        ):
            raise QualificationError(
                f"{suite.suite_id}: compact task receipt linkage differs"
            )
    isolation = payload.get("isolation")
    if not _sandbox_evidence_is_valid(
        isolation,
        require_recovery_policy=recovery is not None,
    ):
        raise QualificationError(f"{suite.suite_id}: sandbox evidence differs")
    if returncode != 0 or payload.get("passed") is not True:
        reason = str(payload.get("failure_tail") or payload.get("reason") or "failed")
        raise QualificationError(f"{suite.suite_id}: {reason[-1000:]}")
    if recovery is not None:
        worker_observation_cid = str(payload.pop("observation_cid"))
        raw_stdout = stdout.encode("utf-8", errors="replace")
        raw_stderr = stderr.encode("utf-8", errors="replace")
        if expected_attestation_barrier is None:
            raise QualificationError("parent attestation barrier is absent")
        if terminal_evidence is None:
            raise QualificationError("parent terminal evidence is absent")
        if [item.get("phase") for item in controller_wx_observations] != [
            "before_worker",
            "prepared_parent_inspection",
            "final_parent_inspection",
            "after_worker_exit",
        ] or any(
            item.get("controller_wx_mapping_count") != 0
            for item in controller_wx_observations
        ):
            raise QualificationError("controller W+X observation population differs")
        terminal_observation: dict[str, Any] = {
            "schema": "lgcvf-recovery-worker-parent-terminal-observation@2",
            "waitid_code": terminal_evidence["waitid_code"],
            "waitid_status": terminal_evidence["waitid_status"],
            "pidfd_signaled_after_wait": terminal_evidence[
                "pidfd_signaled_after_wait"
            ],
            "zombie_leader_reserved_before_reap": terminal_evidence[
                "zombie_leader_reserved_before_reap"
            ],
            "pre_reap_process_group_leader_only": terminal_evidence[
                "pre_reap_process_group_leader_only"
            ],
            "pre_reap_signal_probe_present": terminal_evidence[
                "pre_reap_signal_probe_present"
            ],
            "process_exited": True,
            "process_returncode": returncode,
            "receipt_pipe_closed": True,
            "process_group_empty_after_exit": terminal_evidence[
                "process_group_empty_after_exit"
            ],
            "process_group_signal_probe_esrch": (
                terminal_process_group_probe_absent
            ),
            "cleanup_performed": terminal_evidence["cleanup_performed"],
            "controller_zero_wx_observations": controller_wx_observations,
            "controller_zero_wx_observation_root": content_identity(
                controller_wx_observations
            ),
            "controller_wx_mapping_count": 0,
            "attestation_barrier_cid": expected_attestation_barrier[
                "barrier_cid"
            ],
            "suite_task_receipt_cid": expected_suite_task_receipt[
                "receipt_cid"
            ],
            "worker_observation_cid": worker_observation_cid,
        }
        terminal_observation["terminal_observation_cid"] = content_identity(
            terminal_observation
        )
        payload.update(
            {
                "worker_observation_cid": worker_observation_cid,
                "qualification_runtime_parent_attestation_barrier": (
                    expected_attestation_barrier
                ),
                "qualification_runtime_parent_terminal_observation": (
                    terminal_observation
                ),
                "qualification_runtime_suite_task_receipt": (
                    expected_suite_task_receipt
                ),
                "raw_stdout_size_bytes": len(raw_stdout),
                "raw_stdout_sha256": _sha256_bytes(raw_stdout),
                "raw_stderr_size_bytes": len(raw_stderr),
                "raw_stderr_sha256": _sha256_bytes(raw_stderr),
            }
        )
        payload["observation_cid"] = content_identity(payload)
    return payload


def _git_text(root: Path, *arguments: str) -> str:
    try:
        return _git_bytes(root, arguments).decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise QualificationError("recovery Git identity is not UTF-8") from exc


def _recovery_source_binding(*, root: Path = ROOT) -> dict[str, Any]:
    """Bind a clean superproject and its exact datasets gitlink checkout."""

    resolved = root.resolve(strict=True)
    expected_qualification_runtime_cid, qualification_policy_binding = (
        _recovery_qualification_policy(resolved, head_bound=True)
    )
    qualification_runtime = qualification_runtime_bundle_evidence()
    if (
        qualification_runtime.get("runtime_cid")
        != expected_qualification_runtime_cid
    ):
        raise QualificationError(
            "recovery qualification runtime differs from configuration"
        )
    if Path(_git_text(resolved, "rev-parse", "--show-toplevel")).resolve() != resolved:
        raise QualificationError("recovery source root is not the Git worktree root")
    nested = resolved / "ipfs_datasets_py"
    if not (nested / ".git").exists():
        raise QualificationError("recovery datasets checkout is not a nested Git repository")
    accelerator_status = _git_bytes(
        resolved, ("status", "--porcelain=v1", "--untracked-files=all")
    )
    datasets_status = _git_bytes(
        nested, ("status", "--porcelain=v1", "--untracked-files=all")
    )
    if accelerator_status or datasets_status:
        raise QualificationError("recovery requires clean accelerator and datasets overlays")

    accelerator_head = _git_text(resolved, "rev-parse", "HEAD")
    accelerator_tree = _git_text(resolved, "rev-parse", "HEAD^{tree}")
    datasets_head = _git_text(nested, "rev-parse", "HEAD")
    datasets_tree = _git_text(nested, "rev-parse", "HEAD^{tree}")
    gitlink_fields = _git_text(
        resolved, "ls-tree", "HEAD", "--", "ipfs_datasets_py"
    ).split(None, 3)
    if (
        len(gitlink_fields) != 4
        or gitlink_fields[0] != "160000"
        or gitlink_fields[1] != "commit"
        or gitlink_fields[3] != "ipfs_datasets_py"
    ):
        raise QualificationError("recovery datasets path is not a Git gitlink")
    datasets_gitlink = gitlink_fields[2]
    if datasets_head != datasets_gitlink:
        raise QualificationError("recovery datasets checkout differs from the gitlink")
    for label, identity in (
        ("accelerator HEAD", accelerator_head),
        ("accelerator tree", accelerator_tree),
        ("datasets HEAD", datasets_head),
        ("datasets tree", datasets_tree),
    ):
        if not re.fullmatch(r"[0-9a-f]{40,64}", identity):
            raise QualificationError(f"recovery {label} is malformed")

    qualifier_relative = (
        "scripts/qualify_logic_governed_compositional_verification_fabric.py"
    )
    qualifier = _protected_path_identity(resolved, qualifier_relative)
    executing_digest, executing_size = _sha256_file(Path(__file__).resolve())
    if (
        qualifier.get("kind") != "file"
        or qualifier.get("sha256") != executing_digest
        or qualifier.get("size_bytes") != executing_size
    ):
        raise QualificationError("recovery qualifier differs from the executing judge")
    executable = Path(sys.executable).resolve(strict=True)
    executable_digest, executable_size = _sha256_file(executable)
    duckdb_runtime = bound_duckdb_runtime_evidence()
    body: dict[str, Any] = {
        "schema": "lgcvf-recovery-source-binding@3",
        "repository_topology": "accelerator_with_datasets_gitlink",
        "accelerator_head": accelerator_head,
        "accelerator_tree": accelerator_tree,
        "datasets_gitlink": datasets_gitlink,
        "datasets_head": datasets_head,
        "datasets_tree": datasets_tree,
        "recursive_submodule_status": _git_text(
            resolved, "submodule", "status", "--recursive"
        ),
        "accelerator_clean": True,
        "datasets_clean": True,
        "accelerator_status_sha256": _sha256_bytes(accelerator_status),
        "datasets_status_sha256": _sha256_bytes(datasets_status),
        "qualifier": qualifier,
        "qualification_runtime": {
            "runtime_cid": expected_qualification_runtime_cid,
            "policy_binding": qualification_policy_binding,
            "bundle": qualification_runtime,
        },
        "toolchain": {
            "git": _git_text(resolved, "--version"),
            "machine": platform.machine(),
            "platform": platform.system(),
            "platform_release": platform.release(),
            "python_executable": str(executable),
            "python_executable_sha256": executable_digest,
            "python_executable_size_bytes": executable_size,
            "python_implementation": sys.implementation.name,
            "python_version": platform.python_version(),
            "pytest_version": _pytest_distribution_version(),
            "duckdb_runtime": duckdb_runtime,
        },
    }
    body["source_binding_cid"] = content_identity(body)
    return body


def _preregistered_recovery_manifest(*, root: Path = ROOT) -> dict[str, Any]:
    entries = []
    for ordinal, validation in enumerate(RECOVERY_VALIDATIONS, start=1):
        entry = {
            "ordinal": ordinal,
            "task_id": validation.task_id,
            "task_cid": validation.task_cid,
            "validation_spec": validation.validation_spec(),
            "suite_manifest": _suite_manifest(validation.suite, root=root),
            "candidate_authored": True,
            "self_authority": False,
        }
        entry["entry_cid"] = content_identity(entry)
        entries.append(entry)
    value: dict[str, Any] = {
        "schema": "lgcvf-preregistered-recovery-validation-manifest@1",
        "plan_cid": PLAN_CID,
        "ordered_entries": entries,
    }
    value["manifest_cid"] = content_identity(value)
    return value


def _recovery_projection_commitments(
    source_binding: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the source-derived omission and richer per-suite commitments."""

    if len(observations) != len(RECOVERY_VALIDATIONS):
        raise QualificationError("recovery projection population differs")
    canonical_omissions: list[dict[str, str]] | None = None
    suites: list[dict[str, Any]] = []
    for validation, observation in zip(
        RECOVERY_VALIDATIONS, observations, strict=True
    ):
        projection = observation.get("readonly_projection")
        if not isinstance(projection, Mapping):
            raise QualificationError("recovery projection receipt is absent")
        omissions = projection.get("omitted_source_symlinks")
        if not isinstance(omissions, list):
            raise QualificationError("recovery projection omission set is absent")
        closed_omissions = [dict(item) for item in omissions if isinstance(item, Mapping)]
        if len(closed_omissions) != len(omissions):
            raise QualificationError("recovery projection omission set differs")
        if canonical_omissions is None:
            canonical_omissions = closed_omissions
        elif canonical_omissions != closed_omissions:
            raise QualificationError("recovery suite omission sets disagree")
        suites.append(
            {
                "suite_id": validation.suite.suite_id,
                "task_id": validation.task_id,
                "task_cid": validation.task_cid,
                "projection_cid": projection.get("projection_cid"),
                "copied_source_manifest_root": projection.get(
                    "copied_source_manifest_root"
                ),
            }
        )
    scoped_omissions = []
    for item in canonical_omissions or []:
        path = str(item.get("path") or "")
        scoped_omissions.append(
            {
                "scope": "datasets_gitlink"
                if path.startswith("ipfs_datasets_py/")
                else "accelerator",
                "path": path,
                "git_target": item.get("git_target"),
                "disposition": item.get("disposition"),
            }
        )
    omission: dict[str, Any] = {
        "schema": "lgcvf-recovery-validation-projection-omission@1",
        "accelerator_head": source_binding.get("accelerator_head"),
        "accelerator_tree": source_binding.get("accelerator_tree"),
        "datasets_gitlink": source_binding.get("datasets_gitlink"),
        "datasets_tree": source_binding.get("datasets_tree"),
        "omitted_source_symlinks": sorted(
            scoped_omissions, key=lambda item: (item["scope"], item["path"])
        ),
    }
    omission["commitment_cid"] = content_identity(omission)
    evidence: dict[str, Any] = {
        "schema": "lgcvf-recovery-validation-projection-evidence@1",
        "source_binding_cid": source_binding.get("source_binding_cid"),
        "omission_root": omission["commitment_cid"],
        "ordered_suites": suites,
    }
    evidence["commitment_cid"] = content_identity(evidence)
    return omission, evidence


_RECOVERY_COUNT_FIELDS: Final[tuple[str, ...]] = (
    "collected",
    "passed_count",
    "failed_count",
    "skipped_count",
    "xfailed_count",
    "xpassed_count",
    "error_count",
)
_RECOVERY_OBSERVATION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "suite_id",
        "manifest",
        *_RECOVERY_COUNT_FIELDS,
        "nodeids_cid",
        "exit_code",
        "passed",
        "isolation",
        "duration_ms",
        "transcript_sha256",
        "failure_tail",
        "task_id",
        "task_cid",
        "validation_spec",
        "execution_api",
        "pytest_control_argv",
        "cache_reused",
        "transcript_size_bytes",
        "transcript_limit_bytes",
        "provider_policy",
        "provider_imports_observed",
        "provider_import_attempts",
        "provider_process_attempts",
        "candidate_authored",
        "self_authority",
        "completion_authoritative",
        "readonly_projection",
        "qualification_runtime_cid",
        "qualification_runtime_projection_cid",
        "qualification_runtime_component_root",
        "qualification_runtime_file_manifest_root",
        "qualification_runtime_bootstrap",
        "qualification_runtime_policy_binding",
        "qualification_runtime_module_origins",
        "qualification_runtime_module_origin_root",
        "qualification_runtime_native_mapping",
        "qualification_runtime_native_load_guard",
        "qualification_runtime_suite_task_full_live_observation_cid",
        "qualification_runtime_suite_task_receipt_cid",
        "qualification_runtime_suite_task_receipt",
        "qualification_runtime_parent_attestation_barrier_cid",
        "qualification_runtime_z3_import_denial_prepared_cid",
        "qualification_runtime_z3_import_denial_final_cid",
        "qualification_runtime_parent_attestation_barrier",
        "qualification_runtime_parent_terminal_observation",
        "qualification_runtime_temporary_import_path",
        "worker_observation_cid",
        "raw_stdout_size_bytes",
        "raw_stdout_sha256",
        "raw_stderr_size_bytes",
        "raw_stderr_sha256",
        "observation_cid",
    }
)
_RECOVERY_WORKER_CHILD_FIELDS: Final[frozenset[str]] = (
    _RECOVERY_OBSERVATION_FIELDS
    - {
        "qualification_runtime_suite_task_receipt",
        "qualification_runtime_parent_attestation_barrier",
        "qualification_runtime_parent_terminal_observation",
        "worker_observation_cid",
        "raw_stdout_size_bytes",
        "raw_stdout_sha256",
        "raw_stderr_size_bytes",
        "raw_stderr_sha256",
    }
)


def _validate_recovery_observation(
    value: Mapping[str, Any],
    validation: RecoveryValidation,
    *,
    root: Path = ROOT,
    expected_runtime_bundle: Mapping[str, Any] | None = None,
) -> None:
    if set(value) != _RECOVERY_OBSERVATION_FIELDS:
        raise QualificationError(
            f"{validation.task_id}: recovery observation fields differ"
        )
    if (
        value.get("schema") != RECOVERY_WORKER_SCHEMA
        or value.get("suite_id") != validation.suite.suite_id
        or value.get("task_id") != validation.task_id
        or value.get("task_cid") != validation.task_cid
        or value.get("validation_spec") != validation.validation_spec()
        or value.get("manifest") != _suite_manifest(validation.suite, root=root)
        or value.get("readonly_projection")
        != _recovery_execution_projection_receipt(root, (validation.suite,))
    ):
        raise QualificationError(
            f"{validation.task_id}: recovery observation authority differs"
        )
    expected_runtime_cid, parent_policy_binding = _recovery_qualification_policy(
        root.resolve(strict=True),
        head_bound=True,
    )
    runtime_bundle = dict(
        expected_runtime_bundle
        if expected_runtime_bundle is not None
        else qualification_runtime_bundle_evidence()
    )
    if runtime_bundle.get("runtime_cid") != expected_runtime_cid:
        raise QualificationError(
            f"{validation.task_id}: qualification runtime policy differs"
        )
    if (
        runtime_bundle.get("recovery_suite_task_policy")
        != _recovery_suite_task_policy_matrix()
    ):
        raise QualificationError(
            f"{validation.task_id}: qualification task policy differs"
        )
    expected_component_root = content_identity(runtime_bundle.get("components"))
    copied_policy_binding = {
        **parent_policy_binding,
        "mode": "copied_100644",
        "head_blob_oid": "copied_projection",
        "ordinary_index_equals_head": False,
    }
    copied_policy_binding.pop("policy_binding_cid", None)
    copied_policy_binding["policy_binding_cid"] = content_identity(
        copied_policy_binding
    )
    bootstrap = value.get("qualification_runtime_bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise QualificationError(
            f"{validation.task_id}: qualification runtime bootstrap is absent"
        )
    bootstrap_body = {
        key: item for key, item in bootstrap.items() if key != "bootstrap_cid"
    }
    root_identity = bootstrap.get("runtime_root_identity")
    if (
        bootstrap.get("schema") != _QUALIFICATION_RUNTIME_BOOTSTRAP_SCHEMA
        or bootstrap.get("runtime_cid") != expected_runtime_cid
        or bootstrap.get("projection_cid")
        != value.get("qualification_runtime_projection_cid")
        or bootstrap.get("bootstrap_cid") != content_identity(bootstrap_body)
        or not isinstance(root_identity, Mapping)
        or set(root_identity) != {"dev", "ino", "uid", "gid", "mode", "nlink"}
        or root_identity.get("uid") != os.geteuid()
        or root_identity.get("mode") != 0o500
        or bootstrap.get("path_policy")
        != _qualification_runtime_bootstrap_evidence(
            {
                "runtime_cid": expected_runtime_cid,
                "projection_cid": bootstrap.get("projection_cid"),
                "root_identity": dict(root_identity),
            }
        )["path_policy"]
        or bootstrap.get("environment_policy")
        != _normalized_recovery_worker_environment_evidence()
        or bootstrap.get("interpreter_flags") != ["-I", "-S", "-B"]
        or bootstrap.get("pycache_capsule_schema")
        != _RECOVERY_PYCACHE_CAPSULE_SCHEMA
    ):
        raise QualificationError(
            f"{validation.task_id}: qualification runtime bootstrap differs"
        )
    if _ACTIVE_QUALIFICATION_RUNTIME is not None:
        expected_bootstrap = _qualification_runtime_bootstrap_evidence(
            _ACTIVE_QUALIFICATION_RUNTIME.projection
        )
        if bootstrap != expected_bootstrap:
            raise QualificationError(
                f"{validation.task_id}: live qualification bootstrap differs"
            )
    module_origins = value.get("qualification_runtime_module_origins")
    payload_paths = {
        str(item.get("path") or "")
        for item in _resolve_qualification_runtime().payload_manifest
    } if _ACTIVE_QUALIFICATION_RUNTIME is None else {
        str(item.get("path") or "")
        for item in _ACTIVE_QUALIFICATION_RUNTIME.resolved.payload_manifest
    }
    if not isinstance(module_origins, list):
        raise QualificationError(
            f"{validation.task_id}: qualification module origins are absent"
        )
    normalized_origins: list[dict[str, str]] = []
    observed_module_names: set[str] = set()
    for item in module_origins:
        if not isinstance(item, Mapping) or set(item) != {"name", "origin"}:
            raise QualificationError(
                f"{validation.task_id}: qualification module origin differs"
            )
        name = item.get("name")
        origin = item.get("origin")
        if (
            not isinstance(name, str)
            or not isinstance(origin, str)
            or name in observed_module_names
            or name.partition(".")[0]
            not in _QUALIFICATION_RUNTIME_PROTECTED_MODULE_ROOTS
            or origin not in payload_paths
        ):
            raise QualificationError(
                f"{validation.task_id}: qualification module escaped its bundle"
            )
        observed_module_names.add(name)
        normalized_origins.append({"name": name, "origin": origin})
    observed_runtime_roots = {
        name.partition(".")[0] for name in observed_module_names
    }
    required_runtime_roots = {"pytest", "cvc5"} | (
        {"z3"} if validation.task_id in _RECOVERY_Z3_REQUIRED_TASKS else set()
    )
    if (
        normalized_origins
        != sorted(normalized_origins, key=lambda item: item["name"])
        or not required_runtime_roots.issubset(observed_runtime_roots)
        or (
            validation.task_id not in _RECOVERY_Z3_REQUIRED_TASKS
            and "z3" in observed_runtime_roots
        )
        or value.get("qualification_runtime_module_origin_root")
        != content_identity(normalized_origins)
    ):
        raise QualificationError(
            f"{validation.task_id}: qualification module origin root differs"
        )
    expected_projection = _recovery_execution_projection_receipt(
        root,
        (validation.suite,),
    )
    full_live_observation_cid = value.get(
        "qualification_runtime_suite_task_full_live_observation_cid"
    )
    suite_task_receipt_cid = value.get(
        "qualification_runtime_suite_task_receipt_cid"
    )
    task_receipt = _validate_recovery_suite_task_receipt(
        value.get("qualification_runtime_suite_task_receipt"),
        task_id=validation.task_id,
        suite_id=validation.suite.suite_id,
        runtime_cid=expected_runtime_cid,
        source_projection_root=str(
            expected_projection["copied_source_manifest_root"]
        ),
        full_live_observation_cid=str(full_live_observation_cid or ""),
    )
    if (
        full_live_observation_cid != task_receipt["full_live_observation_cid"]
        or suite_task_receipt_cid != task_receipt["receipt_cid"]
    ):
        raise QualificationError(
            f"{validation.task_id}: suite task receipt linkage differs"
        )
    source_paths = _recovery_projection_source_paths(root, (validation.suite,))
    source_manifest, _source_payloads = _recovery_projection_manifest(
        root,
        source_paths,
    )
    expected_temporary_policy = _logical_legal_data_import_path_policy(
        validation.suite,
        expected_projection,
        source_manifest,
    )
    temporary = value.get("qualification_runtime_temporary_import_path")
    if not isinstance(temporary, Mapping) or set(temporary) != {
        "schema",
        "policy",
        "loaded_module_origins",
        "loaded_module_origin_root",
        "importer_cache_entry_cleared",
        "sealed_path_restored",
        "observation_cid",
    }:
        raise QualificationError(
            f"{validation.task_id}: temporary import observation fields differ"
        )
    loaded_origins = temporary.get("loaded_module_origins")
    legal_prefix = "ipfs_datasets_py/ipfs_datasets_py/processors/legal_data/"
    if not isinstance(loaded_origins, list):
        raise QualificationError(
            f"{validation.task_id}: temporary import origins are absent"
        )
    normalized_loaded: list[dict[str, str]] = []
    loaded_names: set[str] = set()
    source_path_set = set(source_paths)
    temporary_candidates = set(
        expected_temporary_policy.get("direct_import_candidates") or ()
    )
    for item in loaded_origins:
        if not isinstance(item, Mapping) or set(item) != {"name", "origin"}:
            raise QualificationError(
                f"{validation.task_id}: temporary import origin differs"
            )
        name = item.get("name")
        origin = item.get("origin")
        origin_remainder = (
            origin[len(legal_prefix) :]
            if isinstance(origin, str) and origin.startswith(legal_prefix)
            else ""
        )
        origin_first = origin_remainder.partition("/")[0]
        compatible_root = (
            PurePath(origin_first).stem
            if "/" not in origin_remainder
            else origin_first
        )
        if (
            not isinstance(name, str)
            or not isinstance(origin, str)
            or name in loaded_names
            or not origin.startswith(legal_prefix)
            or origin not in source_path_set
            or name.partition(".")[0] not in temporary_candidates
            or name.partition(".")[0] != compatible_root
        ):
            raise QualificationError(
                f"{validation.task_id}: temporary import escaped its projection"
            )
        loaded_names.add(name)
        normalized_loaded.append({"name": name, "origin": origin})
    temporary_body = {
        key: item for key, item in temporary.items() if key != "observation_cid"
    }
    if (
        temporary.get("schema")
        != "lgcvf-recovery-temporary-import-path-observation@1"
        or temporary.get("policy") != expected_temporary_policy
        or normalized_loaded
        != sorted(normalized_loaded, key=lambda item: item["name"])
        or (
            validation.owner_root != "ipfs_datasets_py"
            and normalized_loaded != []
        )
        or temporary.get("loaded_module_origin_root")
        != content_identity(normalized_loaded)
        or temporary.get("importer_cache_entry_cleared") is not True
        or temporary.get("sealed_path_restored") is not True
        or temporary.get("observation_cid") != content_identity(temporary_body)
    ):
        raise QualificationError(
            f"{validation.task_id}: temporary import observation differs"
        )
    if (
        value.get("qualification_runtime_cid") != expected_runtime_cid
        or value.get("qualification_runtime_component_root")
        != expected_component_root
        or value.get("qualification_runtime_file_manifest_root")
        != runtime_bundle.get("file_manifest_root")
        or value.get("qualification_runtime_policy_binding")
        != copied_policy_binding
        or value.get("qualification_runtime_native_mapping")
        != _expected_solver_native_mapping_evidence(
            z3_required=validation.task_id in _RECOVERY_Z3_REQUIRED_TASKS
        )
    ):
        raise QualificationError(
            f"{validation.task_id}: qualification runtime evidence differs"
        )
    native_guard = value.get("qualification_runtime_native_load_guard")
    if not isinstance(native_guard, Mapping):
        raise QualificationError(
            f"{validation.task_id}: native load guard evidence is absent"
        )
    native_guard_body = {
        key: item for key, item in native_guard.items() if key != "guard_cid"
    }
    native_platform = runtime_bundle.get("native_platform_binding")
    payload_native = (
        native_platform.get("solver_payload_native_files")
        if isinstance(native_platform, Mapping)
        else None
    )
    if not isinstance(payload_native, list):
        raise QualificationError(
            f"{validation.task_id}: native payload evidence is absent"
        )
    projected_native = sorted(
        (
            {
                "path_token": "runtime:" + str(item.get("path") or ""),
                "sha256": item.get("sha256"),
                "size_bytes": item.get("size_bytes"),
            }
            for item in payload_native
            if isinstance(item, Mapping)
            and (
                validation.task_id in _RECOVERY_Z3_REQUIRED_TASKS
                or not str(item.get("path") or "").startswith("z3/")
            )
        ),
        key=lambda item: str(item["path_token"]),
    )
    (
        stdlib_native,
        stdlib_native_root,
        stdlib_native_bytes,
    ) = _stdlib_extension_native_manifest()
    python_runtime_binding = runtime_bundle.get("python_runtime_binding")
    stdlib_binding = (
        python_runtime_binding.get("stdlib_extension_binding")
        if isinstance(python_runtime_binding, Mapping)
        else None
    )
    producer_binding = (
        python_runtime_binding.get("z3_libffi_rwx_producer_binding")
        if isinstance(python_runtime_binding, Mapping)
        else None
    )
    allowed_tokens = {
        str(item["path_token"]) for item in projected_native
    } | {
        str(item["path_token"]) for item in stdlib_native.values()
    }
    attempts = native_guard.get("ordered_attempts")
    additions = native_guard.get("post_pytest_mapping_additions")
    if not isinstance(attempts, list) or not isinstance(additions, list):
        raise QualificationError(
            f"{validation.task_id}: native load guard population is absent"
        )
    normalized_attempts: list[dict[str, Any]] = []
    z3_required = validation.task_id in _RECOVERY_Z3_REQUIRED_TASKS
    expected_native_policy: dict[str, Any] = {
        "schema": "lgcvf-recovery-suite-native-policy@1",
        "z3_required": z3_required,
        "controller_rwx_permitted": False,
        "worker_rwx_disposition": (
            "exact_z3_libffi_anonymous_4096_rwxp"
            if z3_required
            else "zero_writable_executable"
        ),
    }
    expected_native_policy["policy_cid"] = content_identity(
        expected_native_policy
    )
    expected_rwx = (
        [
            {
                "kind": "anonymous",
                "permissions": "rwxp",
                "offset": "00000000",
                "device": "00:00",
                "inode": "0",
                "size_bytes": 4096,
                "path_token": "anonymous",
            }
        ]
        if z3_required
        else []
    )
    for sequence, item in enumerate(attempts, start=1):
        if (
            not isinstance(item, Mapping)
            or set(item)
            != {"sequence", "event", "target_token", "disposition"}
            or isinstance(item.get("sequence"), bool)
            or not isinstance(item.get("sequence"), int)
            or item.get("sequence") != sequence
            or item.get("event") not in {"ctypes.dlopen", "extension_import"}
            or item.get("target_token") not in allowed_tokens
            or item.get("disposition") != "allowed"
        ):
            raise QualificationError(
                f"{validation.task_id}: native load attempt differs"
            )
        normalized_attempts.append(dict(item))
    if (
        set(native_guard)
        != {
            "schema",
            "suite_native_policy",
            "installed_before_solver_import",
            "irreversible_until_worker_exit",
            "projected_native_manifest_root",
            "stdlib_extension_file_count",
            "stdlib_extension_total_bytes",
            "stdlib_extension_manifest_root",
            "rwx_producer_binding_cid",
            "ordered_attempts",
            "ordered_attempt_root",
            "pre_solver_executable_mapping_root",
            "pre_solver_executable_mapping_count",
            "post_solver_executable_mapping_root",
            "post_solver_executable_mapping_count",
            "pre_solver_writable_executable_mappings",
            "post_solver_writable_executable_mappings",
            "writable_executable_limitation",
            "pre_pytest_thread_count",
            "pre_pytest_thread_population_root",
            "thread_population_restored",
            "pre_pytest_kernel_task_count",
            "kernel_task_population_restored",
            "pre_pytest_child_process_count",
            "child_process_population_restored",
            "post_pytest_mapping_additions",
            "post_pytest_mapping_addition_root",
            "unauthorized_attempts",
            "unauthorized_mapping_additions",
            "guard_cid",
        }
        or native_guard.get("schema") != "lgcvf-recovery-native-load-guard@3"
        or native_guard.get("suite_native_policy") != expected_native_policy
        or native_guard.get("installed_before_solver_import") is not True
        or native_guard.get("irreversible_until_worker_exit") is not True
        or native_guard.get("projected_native_manifest_root")
        != content_identity(projected_native)
        or native_guard.get("stdlib_extension_file_count") != len(stdlib_native)
        or native_guard.get("stdlib_extension_total_bytes")
        != stdlib_native_bytes
        or native_guard.get("stdlib_extension_manifest_root")
        != stdlib_native_root
        or not isinstance(stdlib_binding, Mapping)
        or stdlib_binding.get("schema")
        != "lgcvf-stdlib-extension-native-binding@1"
        or stdlib_binding.get("file_count") != len(stdlib_native)
        or stdlib_binding.get("total_bytes") != stdlib_native_bytes
        or stdlib_binding.get("file_manifest_root") != stdlib_native_root
        or not isinstance(producer_binding, Mapping)
        or producer_binding.get("schema")
        != "lgcvf-z3-libffi-rwx-producer-binding@1"
        or native_guard.get("rwx_producer_binding_cid")
        != producer_binding.get("producer_binding_cid")
        or native_guard.get("ordered_attempt_root")
        != content_identity(normalized_attempts)
        or not _is_canonical_content_cid(
            native_guard.get("pre_solver_executable_mapping_root")
        )
        or isinstance(native_guard.get("pre_solver_executable_mapping_count"), bool)
        or not isinstance(native_guard.get("pre_solver_executable_mapping_count"), int)
        or int(native_guard.get("pre_solver_executable_mapping_count", 0)) <= 0
        or not _is_canonical_content_cid(
            native_guard.get("post_solver_executable_mapping_root")
        )
        or isinstance(native_guard.get("post_solver_executable_mapping_count"), bool)
        or not isinstance(native_guard.get("post_solver_executable_mapping_count"), int)
        or int(native_guard.get("post_solver_executable_mapping_count", 0))
        < int(native_guard.get("pre_solver_executable_mapping_count", 0))
        or native_guard.get("pre_solver_writable_executable_mappings") != []
        or native_guard.get("post_solver_writable_executable_mappings")
        != expected_rwx
        or native_guard.get("writable_executable_limitation")
        != (
            "z3_libffi_rwx_4k_ephemeral_worker" if z3_required else "none"
        )
        or isinstance(native_guard.get("pre_pytest_thread_count"), bool)
        or not isinstance(native_guard.get("pre_pytest_thread_count"), int)
        or int(native_guard.get("pre_pytest_thread_count", 0)) <= 0
        or not _is_canonical_content_cid(
            native_guard.get("pre_pytest_thread_population_root")
        )
        or native_guard.get("thread_population_restored") is not True
        or isinstance(native_guard.get("pre_pytest_kernel_task_count"), bool)
        or not isinstance(native_guard.get("pre_pytest_kernel_task_count"), int)
        or int(native_guard.get("pre_pytest_kernel_task_count", 0)) <= 0
        or native_guard.get("kernel_task_population_restored") is not True
        or isinstance(native_guard.get("pre_pytest_child_process_count"), bool)
        or not isinstance(native_guard.get("pre_pytest_child_process_count"), int)
        or native_guard.get("pre_pytest_child_process_count") != 0
        or native_guard.get("child_process_population_restored") is not True
        or additions != sorted(set(additions))
        or any(
            item not in allowed_tokens
            and not (z3_required and item == "z3-libffi-anonymous-rwx:4096")
            for item in additions
        )
        or not {
            "runtime:" + path
            for path in _expected_solver_native_mapping_evidence(
                z3_required=z3_required
            )[
                "projected_solver_paths"
            ]
        }.issubset(set(additions))
        or native_guard.get("post_pytest_mapping_addition_root")
        != content_identity(additions)
        or native_guard.get("unauthorized_attempts") != []
        or native_guard.get("unauthorized_mapping_additions") != []
        or native_guard.get("guard_cid") != content_identity(native_guard_body)
        or not (
            {
                "runtime:cvc5/cvc5_python_base.cpython-312-aarch64-linux-gnu.so"
            }
            | ({"runtime:z3/lib/libz3.so"} if z3_required else set())
        ).issubset({str(item["target_token"]) for item in normalized_attempts})
    ):
        raise QualificationError(
            f"{validation.task_id}: native load guard evidence differs"
        )
    barrier = _validate_worker_attestation_barrier(
        value.get("qualification_runtime_parent_attestation_barrier"),
        suite_id=validation.suite.suite_id,
        runtime_cid=expected_runtime_cid,
        native_guard_cid=str(native_guard["guard_cid"]),
        z3_required=z3_required,
        task_receipt=task_receipt,
    )
    if (
        value.get("qualification_runtime_parent_attestation_barrier_cid")
        != barrier["barrier_cid"]
        or value.get(
            "qualification_runtime_z3_import_denial_prepared_cid"
        )
        != barrier["prepared_z3_import_denial_evidence_cid"]
        or value.get("qualification_runtime_z3_import_denial_final_cid")
        != barrier["final_z3_import_denial_evidence_cid"]
    ):
        raise QualificationError(
            f"{validation.task_id}: parent attestation barrier CID differs"
        )
    prepared_self = barrier["prepared"]["self_observation"]
    final_self = barrier["final"]["self_observation"]
    if (
        prepared_self.get("thread_count")
        != native_guard.get("pre_pytest_thread_count")
        or final_self.get("thread_count")
        != native_guard.get("pre_pytest_thread_count")
        or prepared_self.get("kernel_task_count")
        != native_guard.get("pre_pytest_kernel_task_count")
        or final_self.get("kernel_task_count")
        != native_guard.get("pre_pytest_kernel_task_count")
        or prepared_self.get("child_process_count")
        != native_guard.get("pre_pytest_child_process_count")
        or final_self.get("child_process_count")
        != native_guard.get("pre_pytest_child_process_count")
        or final_self.get("executable_mapping_count")
        != native_guard.get("post_solver_executable_mapping_count")
        or final_self.get("normalized_executable_mapping_root")
        != native_guard.get("post_solver_executable_mapping_root")
        or final_self.get("writable_executable_mappings")
        != native_guard.get("post_solver_writable_executable_mappings")
    ):
        raise QualificationError(
            f"{validation.task_id}: native guard and barrier populations differ"
        )
    observation_body = {
        key: item for key, item in value.items() if key != "observation_cid"
    }
    if value.get("observation_cid") != content_identity(observation_body):
        raise QualificationError(
            f"{validation.task_id}: recovery observation identity differs"
        )
    worker_body = {
        key: item
        for key, item in observation_body.items()
        if key
        not in {
            "worker_observation_cid",
            "qualification_runtime_parent_attestation_barrier",
            "qualification_runtime_parent_terminal_observation",
            "qualification_runtime_suite_task_receipt",
            "raw_stdout_size_bytes",
            "raw_stdout_sha256",
            "raw_stderr_size_bytes",
            "raw_stderr_sha256",
        }
    }
    if value.get("worker_observation_cid") != content_identity(worker_body):
        raise QualificationError(
            f"{validation.task_id}: worker observation identity differs"
        )
    _validate_worker_parent_terminal_observation(
        value.get("qualification_runtime_parent_terminal_observation"),
        worker_observation_cid=str(value.get("worker_observation_cid") or ""),
        barrier=barrier,
    )
    raw_counts = {field: value.get(field) for field in _RECOVERY_COUNT_FIELDS}
    if any(
        isinstance(count, bool) or not isinstance(count, int)
        for count in raw_counts.values()
    ):
        raise QualificationError(f"{validation.task_id}: recovery counts are invalid")
    counts = {field: int(count) for field, count in raw_counts.items()}
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
        or isinstance(value.get("exit_code"), bool)
        or not isinstance(value.get("exit_code"), int)
        or value.get("exit_code") != 0
        or value.get("passed") is not True
        or value.get("failure_tail") != ""
        or value.get("execution_api") != "pytest.main"
        or value.get("pytest_control_argv")
        != ["-q", "-ra", "--maxfail=1"]
        or value.get("cache_reused") is not False
        or value.get("candidate_authored") is not True
        or value.get("self_authority") is not False
        or value.get("completion_authoritative") is not False
        or value.get("provider_policy") != _RecoveryProviderGuard.policy()
        or value.get("provider_imports_observed") != []
        or value.get("provider_import_attempts") != []
        or value.get("provider_process_attempts") != []
        or not _sandbox_evidence_is_valid(
            value.get("isolation"), require_recovery_policy=True
        )
    ):
        raise QualificationError(
            f"{validation.task_id}: recovery suite did not pass exactly"
        )
    for numeric_field in (
        "duration_ms",
        "transcript_size_bytes",
        "transcript_limit_bytes",
        "raw_stdout_size_bytes",
        "raw_stderr_size_bytes",
    ):
        item = value.get(numeric_field)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise QualificationError(
                f"{validation.task_id}: recovery {numeric_field} is invalid"
            )
    if (
        int(value["duration_ms"]) > validation.validation_spec()["timeout_seconds"] * 1000
        or value.get("transcript_limit_bytes") != _MAX_WORKER_TRANSCRIPT_BYTES
        or int(value["transcript_size_bytes"]) > _MAX_WORKER_TRANSCRIPT_BYTES
        or int(value["raw_stdout_size_bytes"]) >= _MAX_WORKER_TRANSCRIPT_BYTES
        or int(value["raw_stderr_size_bytes"]) >= _MAX_WORKER_TRANSCRIPT_BYTES
        or not str(value.get("nodeids_cid") or "").startswith("b")
        or any(
            not re.fullmatch(r"sha256:[0-9a-f]{64}", str(value.get(field) or ""))
            for field in (
                "transcript_sha256",
                "raw_stdout_sha256",
                "raw_stderr_sha256",
            )
        )
    ):
        raise QualificationError(
            f"{validation.task_id}: recovery output or duration bound differs"
        )


def _recovery_unavailable(reason: QualificationRuntimeUnavailable) -> dict[str, Any]:
    expected_component = _RECOVERY_UNAVAILABLE_CLASSIFICATIONS.get(
        (reason.phase, reason.reason_code)
    )
    if (
        expected_component is None
        or reason.component != expected_component
        or not _is_canonical_content_cid(reason.expected_runtime_cid)
        or not _is_canonical_content_cid(reason.observed_runtime_cid)
        or reason.expected_runtime_cid == reason.observed_runtime_cid
    ):
        raise QualificationError(
            "qualification runtime unavailable evidence differs"
        )
    population = [
        {
            "task_id": item.task_id,
            "task_cid": item.task_cid,
            "validation_spec_cid": item.validation_spec()["validation_spec_cid"],
        }
        for item in RECOVERY_VALIDATIONS
    ]
    detail = ":".join(
        (
            reason.phase,
            reason.reason_code,
            reason.component or "qualification_runtime",
        )
    )[:512]
    value: dict[str, Any] = {
        "schema": RECOVERY_UNAVAILABLE_SCHEMA,
        "qualification_schema": RECOVERY_SCHEMA,
        "plan_cid": PLAN_CID,
        "disposition": "unavailable",
        "ordered_task_population": population,
        "reason_code": reason.reason_code,
        "phase": reason.phase,
        "expected_runtime_cid": reason.expected_runtime_cid,
        "observed_runtime_cid": reason.observed_runtime_cid,
        "component": reason.component,
        "detail": detail,
        "detail_sha256": _sha256_bytes(detail.encode("utf-8")),
        "task_authority": False,
        "objective_authority": False,
        "release_authority": False,
        "production_authority": False,
        "self_authority": False,
        "completion_authoritative": False,
        "task_implementation_complete": False,
        "test_qualification_complete": False,
        "objective_complete": False,
        "release_qualified": False,
        "production_authorized": False,
        "production_authoritative": False,
    }
    value["receipt_cid"] = content_identity(value)
    return value


def run_preregistered_recovery_qualification(
    *, root: Path = ROOT
) -> dict[str, Any]:
    """Run exactly six task-bound replays inside the protected OS sandbox."""

    _require_isolated_recovery_runtime()
    controller_entry = _controller_zero_wx_observation(
        phase="qualification_entry"
    )
    expected_runtime_cid, initial_policy_binding = _recovery_qualification_policy(
        root.resolve(strict=True),
        head_bound=True,
    )
    try:
        with isolated_qualification_runtime(
            expected_runtime_cid=expected_runtime_cid
        ) as qualification_runtime:
            manifest = _preregistered_recovery_manifest(root=root)
            before = _recovery_source_binding(root=root)
            observations: list[dict[str, Any]] = []
            entries = manifest["ordered_entries"]
            for validation, entry in zip(
                RECOVERY_VALIDATIONS,
                entries,
                strict=True,
            ):
                observation = _run_suite(
                    validation.suite,
                    expected_manifest=entry["suite_manifest"],
                    root=root,
                    qualification_runtime=qualification_runtime,
                )
                _validate_recovery_observation(observation, validation, root=root)
                observations.append(observation)
                if _recovery_source_binding(root=root) != before:
                    raise QualificationError(
                        f"{validation.task_id}: recovery source changed during replay"
                    )
            after = _recovery_source_binding(root=root)
            after_runtime_cid, after_policy_binding = (
                _recovery_qualification_policy(
                    root.resolve(strict=True),
                    head_bound=True,
                )
            )
            if (
                after_runtime_cid != expected_runtime_cid
                or after_policy_binding != initial_policy_binding
            ):
                raise QualificationError(
                    "qualification runtime policy changed during recovery"
                )
            _validate_active_qualification_runtime(qualification_runtime)
            totals = {
                field: sum(int(item[field]) for item in observations)
                for field in _RECOVERY_COUNT_FIELDS
            }
            omission_commitment, projection_evidence = (
                _recovery_projection_commitments(before, observations)
            )
            runtime_evidence = dict(qualification_runtime.resolved.bundle)
            common_projection_cid = qualification_runtime.projection[
                "projection_cid"
            ]
            controller_final = _controller_zero_wx_observation(
                phase="qualification_final"
            )
            controller_evidence: dict[str, Any] = {
                "schema": "lgcvf-recovery-controller-zero-wx@1",
                "entry": controller_entry,
                "ordered_suite_observations": [
                    item["qualification_runtime_parent_terminal_observation"][
                        "controller_zero_wx_observations"
                    ]
                    for item in observations
                ],
                "final": controller_final,
                "controller_wx_mapping_count": 0,
                "controller_rwx_permitted": False,
            }
            controller_evidence["controller_evidence_cid"] = content_identity(
                controller_evidence
            )
            result: dict[str, Any] = {
                "schema": RECOVERY_SCHEMA,
                "plan_cid": PLAN_CID,
                "predecessor_plan_cid": PREDECESSOR_PLAN_CID,
                "cohort": "hermetic_local_execution",
                "disposition": "passed",
                "validation_role": "candidate_authored_historical_replay",
                "recovery_manifest": manifest,
                "source_binding_before": before,
                "source_binding_after": after,
                "source_unchanged": after == before,
                "exact_suite_membership": True,
                "passed": all(
                    item.get("passed") is True for item in observations
                ),
                "totals": totals,
                "suites": observations,
                "qualification_runtime_cid": expected_runtime_cid,
                "qualification_runtime_evidence": runtime_evidence,
                "qualification_runtime_evidence_cid": runtime_evidence[
                    "runtime_cid"
                ],
                "qualification_runtime_projection_cid": common_projection_cid,
                "qualification_runtime_worker_consistent": all(
                    item.get("qualification_runtime_cid")
                    == expected_runtime_cid
                    and item.get("qualification_runtime_projection_cid")
                    == common_projection_cid
                    for item in observations
                ),
                "qualification_runtime_controller_zero_wx": (
                    controller_evidence
                ),
                "validation_projection_omission_commitment": (
                    omission_commitment
                ),
                "validation_projection_omission_root": omission_commitment[
                    "commitment_cid"
                ],
                "validation_projection_evidence_commitment": projection_evidence,
                "validation_projection_evidence_root": projection_evidence[
                    "commitment_cid"
                ],
                "provider_route": "none",
                "network_permitted": False,
                "cache_reused": False,
                "candidate_authored_replay": True,
                "self_authority": False,
                "completion_authoritative": False,
                "task_implementation_complete": False,
                "test_qualification_complete": False,
                "objective_complete": False,
                "release_qualified": False,
                "production_authorized": False,
                "production_authoritative": False,
                "limitations": list(RECOVERY_LIMITATIONS),
            }
            result["receipt_cid"] = content_identity(result)
            return verify_preregistered_recovery_qualification(
                result,
                root=root,
            )
    except QualificationRuntimeUnavailable as exc:
        return _recovery_unavailable(exc)


_RECOVERY_RESULT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "plan_cid",
        "predecessor_plan_cid",
        "cohort",
        "disposition",
        "validation_role",
        "recovery_manifest",
        "source_binding_before",
        "source_binding_after",
        "source_unchanged",
        "exact_suite_membership",
        "passed",
        "totals",
        "suites",
        "qualification_runtime_cid",
        "qualification_runtime_evidence",
        "qualification_runtime_evidence_cid",
        "qualification_runtime_projection_cid",
        "qualification_runtime_worker_consistent",
        "qualification_runtime_controller_zero_wx",
        "validation_projection_omission_commitment",
        "validation_projection_omission_root",
        "validation_projection_evidence_commitment",
        "validation_projection_evidence_root",
        "provider_route",
        "network_permitted",
        "cache_reused",
        "candidate_authored_replay",
        "self_authority",
        "completion_authoritative",
        "task_implementation_complete",
        "test_qualification_complete",
        "objective_complete",
        "release_qualified",
        "production_authorized",
        "production_authoritative",
        "limitations",
        "receipt_cid",
    }
)
_RECOVERY_UNAVAILABLE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "qualification_schema",
        "plan_cid",
        "disposition",
        "ordered_task_population",
        "reason_code",
        "phase",
        "expected_runtime_cid",
        "observed_runtime_cid",
        "component",
        "detail",
        "detail_sha256",
        "task_authority",
        "objective_authority",
        "release_authority",
        "production_authority",
        "self_authority",
        "completion_authoritative",
        "task_implementation_complete",
        "test_qualification_complete",
        "objective_complete",
        "release_qualified",
        "production_authorized",
        "production_authoritative",
        "receipt_cid",
    }
)


def verify_preregistered_recovery_qualification(
    value: Mapping[str, Any],
    *,
    root: Path = ROOT,
    require_passed: bool = True,
) -> dict[str, Any]:
    """Content-check a recovery receipt against this exact clean checkout."""

    _require_isolated_recovery_runtime()
    if value.get("schema") == RECOVERY_UNAVAILABLE_SCHEMA:
        if set(value) != _RECOVERY_UNAVAILABLE_FIELDS:
            raise QualificationError("recovery unavailable receipt fields differ")
        body = {key: item for key, item in value.items() if key != "receipt_cid"}
        detail = value.get("detail")
        classification = (value.get("phase"), value.get("reason_code"))
        expected_component = _RECOVERY_UNAVAILABLE_CLASSIFICATIONS.get(
            classification
        )
        expected_population = [
            {
                "task_id": item.task_id,
                "task_cid": item.task_cid,
                "validation_spec_cid": item.validation_spec()[
                    "validation_spec_cid"
                ],
            }
            for item in RECOVERY_VALIDATIONS
        ]
        if expected_component is None:
            raise QualificationError(
                "recovery unavailable classification differs"
            )
        expected_runtime_cid, _policy_binding = _recovery_qualification_policy(
            root.resolve(strict=True),
            head_bound=True,
        )
        observed_runtime_cid = str(
            _resolve_qualification_runtime().bundle.get("runtime_cid") or ""
        )
        expected_detail = ":".join(
            (
                str(classification[0]),
                str(classification[1]),
                expected_component,
            )
        )[:512]
        if (
            value.get("receipt_cid") != content_identity(body)
            or value.get("qualification_schema") != RECOVERY_SCHEMA
            or value.get("plan_cid") != PLAN_CID
            or value.get("disposition") != "unavailable"
            or value.get("ordered_task_population") != expected_population
            or value.get("expected_runtime_cid") != expected_runtime_cid
            or value.get("observed_runtime_cid") != observed_runtime_cid
            or value.get("expected_runtime_cid")
            == value.get("observed_runtime_cid")
            or not _is_canonical_content_cid(value.get("expected_runtime_cid"))
            or not _is_canonical_content_cid(value.get("observed_runtime_cid"))
            or value.get("component") != expected_component
            or not isinstance(detail, str)
            or detail != expected_detail
            or value.get("detail_sha256")
            != _sha256_bytes(expected_detail.encode("utf-8"))
            or any(
                value.get(field) is not False
                for field in (
                    "task_authority",
                    "objective_authority",
                    "release_authority",
                    "production_authority",
                    "self_authority",
                    "completion_authoritative",
                    "task_implementation_complete",
                    "test_qualification_complete",
                    "objective_complete",
                    "release_qualified",
                    "production_authorized",
                    "production_authoritative",
                )
            )
        ):
            raise QualificationError("recovery unavailable receipt differs")
        if require_passed:
            raise QualificationError("recovery qualification is unavailable")
        return dict(value)

    if set(value) != _RECOVERY_RESULT_FIELDS:
        raise QualificationError("recovery qualification fields differ")
    body = {key: item for key, item in value.items() if key != "receipt_cid"}
    if value.get("receipt_cid") != content_identity(body):
        raise QualificationError("recovery qualification identity differs")
    current_source = _recovery_source_binding(root=root)
    expected_manifest = _preregistered_recovery_manifest(root=root)
    expected_runtime_cid, _policy_binding = _recovery_qualification_policy(
        root.resolve(strict=True),
        head_bound=True,
    )
    expected_runtime_evidence = qualification_runtime_bundle_evidence()
    if (
        value.get("schema") != RECOVERY_SCHEMA
        or value.get("plan_cid") != PLAN_CID
        or value.get("predecessor_plan_cid") != PREDECESSOR_PLAN_CID
        or value.get("cohort") != "hermetic_local_execution"
        or value.get("disposition") != "passed"
        or value.get("validation_role")
        != "candidate_authored_historical_replay"
        or value.get("recovery_manifest") != expected_manifest
        or value.get("source_binding_before") != current_source
        or value.get("source_binding_after") != current_source
        or value.get("source_unchanged") is not True
        or value.get("exact_suite_membership") is not True
        or value.get("passed") is not True
        or value.get("provider_route") != "none"
        or value.get("network_permitted") is not False
        or value.get("cache_reused") is not False
        or value.get("candidate_authored_replay") is not True
        or value.get("self_authority") is not False
        or value.get("completion_authoritative") is not False
        or value.get("qualification_runtime_cid") != expected_runtime_cid
        or value.get("qualification_runtime_evidence")
        != expected_runtime_evidence
        or value.get("qualification_runtime_evidence_cid")
        != expected_runtime_evidence.get("runtime_cid")
        or value.get("qualification_runtime_evidence_cid")
        != expected_runtime_cid
        or value.get("qualification_runtime_worker_consistent") is not True
        or not _is_canonical_content_cid(
            value.get("qualification_runtime_projection_cid")
        )
    ):
        raise QualificationError("recovery qualification authority binding differs")
    for authority_field in (
        "task_implementation_complete",
        "test_qualification_complete",
        "objective_complete",
        "release_qualified",
        "production_authorized",
        "production_authoritative",
    ):
        if value.get(authority_field) is not False:
            raise QualificationError(
                f"recovery qualification raises {authority_field}"
            )
    observations = value.get("suites")
    if not isinstance(observations, list) or len(observations) != len(
        RECOVERY_VALIDATIONS
    ):
        raise QualificationError("recovery suite population differs")
    totals = dict.fromkeys(_RECOVERY_COUNT_FIELDS, 0)
    for validation, observation in zip(
        RECOVERY_VALIDATIONS, observations, strict=True
    ):
        if not isinstance(observation, Mapping):
            raise QualificationError("recovery observation is not an object")
        _validate_recovery_observation(
            observation,
            validation,
            root=root,
            expected_runtime_bundle=expected_runtime_evidence,
        )
        for count_field in _RECOVERY_COUNT_FIELDS:
            totals[count_field] += int(observation[count_field])
        if (
            observation.get("qualification_runtime_cid")
            != expected_runtime_cid
            or observation.get("qualification_runtime_projection_cid")
            != value.get("qualification_runtime_projection_cid")
        ):
            raise QualificationError(
                "recovery worker qualification runtime differs"
            )
    _validate_controller_zero_wx_aggregate(
        value.get("qualification_runtime_controller_zero_wx"),
        observations=[dict(item) for item in observations],
    )
    if not isinstance(value.get("totals"), Mapping) or dict(value["totals"]) != totals:
        raise QualificationError("recovery totals do not reconstruct")
    (
        expected_omission_commitment,
        expected_projection_evidence,
    ) = _recovery_projection_commitments(
        current_source,
        [dict(item) for item in observations],
    )
    if (
        value.get("validation_projection_omission_commitment")
        != expected_omission_commitment
        or value.get("validation_projection_omission_root")
        != expected_omission_commitment["commitment_cid"]
        or value.get("validation_projection_evidence_commitment")
        != expected_projection_evidence
        or value.get("validation_projection_evidence_root")
        != expected_projection_evidence["commitment_cid"]
    ):
        raise QualificationError("recovery projection omission commitment differs")
    limitations = value.get("limitations")
    if limitations != list(RECOVERY_LIMITATIONS):
        raise QualificationError("recovery limitations differ from mandatory caveats")
    if require_passed and value.get("passed") is not True:
        raise QualificationError("recovery qualification did not pass")
    return dict(value)


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
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true")
    action.add_argument("--recovery", action="store_true")
    action.add_argument("--verify-recovery", type=Path, metavar="RECEIPT")
    parser.add_argument("--worker", metavar="SUITE_ID", help=argparse.SUPPRESS)
    parser.add_argument("--worker-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-write-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-receipt-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-attestation-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-ack-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-attestation-nonce", help=argparse.SUPPRESS)
    parser.add_argument("--worker-projection", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if args.check or args.recovery or args.verify_recovery is not None:
            parser.error("qualification actions cannot be combined with --worker")
        if (
            args.worker_root is None
            or args.worker_write_root is None
            or args.worker_receipt_fd is None
        ):
            print(json.dumps({"schema": WORKER_SCHEMA, "error": "sandbox_required"}))
            return 2
        execution_projection: Mapping[str, Any] | None = None
        if args.worker_projection is not None:
            try:
                encoded = args.worker_projection.encode("ascii")
                if len(encoded) > 64 * 1024:
                    raise QualificationError("worker projection exceeds its bound")
                decoded = base64.b64decode(encoded, altchars=b"-_", validate=True)
                raw_projection = _strict_json_loads(
                    decoded.decode("utf-8"), noun="worker projection"
                )
            except (UnicodeDecodeError, UnicodeEncodeError, ValueError) as exc:
                raise QualificationError("worker projection is malformed") from exc
            if not isinstance(raw_projection, dict):
                raise QualificationError("worker projection root is not an object")
            execution_projection = raw_projection
        return _worker(
            args.worker,
            execution_root=args.worker_root,
            write_root=args.worker_write_root,
            receipt_descriptor=args.worker_receipt_fd,
            attestation_descriptor=args.worker_attestation_fd,
            acknowledgement_descriptor=args.worker_ack_fd,
            attestation_nonce=args.worker_attestation_nonce,
            execution_projection=execution_projection,
        )
    if (
        args.worker_root is not None
        or args.worker_write_root is not None
        or args.worker_receipt_fd is not None
        or args.worker_attestation_fd is not None
        or args.worker_ack_fd is not None
        or args.worker_attestation_nonce is not None
        or args.worker_projection is not None
    ):
        parser.error("worker sandbox arguments require --worker")
    try:
        if args.recovery:
            recovery = run_preregistered_recovery_qualification()
            verify_preregistered_recovery_qualification(
                recovery,
                require_passed=recovery.get("disposition") == "passed",
            )
            print(json.dumps(recovery, indent=2, sort_keys=True))
            return 0 if recovery.get("disposition") == "passed" else 1
        if args.verify_recovery is not None:
            recovery_value = _strict_json_loads(
                args.verify_recovery.read_text(encoding="utf-8"),
                noun="recovery receipt",
            )
            if not isinstance(recovery_value, dict):
                raise QualificationError("recovery receipt root is not an object")
            verified = verify_preregistered_recovery_qualification(recovery_value)
            print(json.dumps(verified, indent=2, sort_keys=True))
            return 0
        current: dict[str, Any] | None = None
        if args.check:
            current_value = _strict_json_loads(
                OUTPUT.read_text(encoding="utf-8"),
                noun="stored qualification result",
            )
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
