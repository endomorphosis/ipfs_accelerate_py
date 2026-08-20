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
import subprocess
import sys
import sysconfig
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from threading import Thread
from typing import Any, Final


def _install_isolated_recovery_pycache() -> tuple[
    tempfile.TemporaryDirectory[str] | None,
    Path | None,
    tuple[int, int] | None,
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
        return None, None, None
    directory = tempfile.TemporaryDirectory(prefix="lgcvf-isolated-pycache-")
    root = Path(directory.name).resolve(strict=True)
    os.chmod(root, 0o700)
    status = root.lstat()
    sys.pycache_prefix = str(root)
    return directory, root, (status.st_dev, status.st_ino)


(
    _ISOLATED_RECOVERY_PYCACHE_DIRECTORY,
    _ISOLATED_RECOVERY_PYCACHE_ROOT,
    _ISOLATED_RECOVERY_PYCACHE_IDENTITY,
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

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (  # noqa: E402
    content_identity,
)

SCHEMA: Final[str] = "lgcvf-independent-hermetic-qualification@1"
WORKER_SCHEMA: Final[str] = "lgcvf-independent-pytest-observation@1"
RECOVERY_SCHEMA: Final[str] = "lgcvf-independent-recovery-qualification@3"
RECOVERY_WORKER_SCHEMA: Final[str] = (
    "lgcvf-independent-recovery-pytest-observation@3"
)
RECOVERY_UNAVAILABLE_SCHEMA: Final[str] = (
    "lgcvf-independent-recovery-qualification-unavailable@1"
)
RECOVERY_LIMITATIONS: Final[tuple[str, ...]] = (
    "the six historical task-authored tests are replay evidence, not independent completion oracles",
    "provider-import observation covers the pytest worker process only",
    "descendants inherit OS network denial but their Python import tables are not observed",
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
_SECCOMP_POLICY_SCHEMA: Final[str] = "lgcvf-recovery-seccomp-policy@1"
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
_DENIED_SYSCALLS: Final[tuple[str, ...]] = (
    *_DENIED_NETWORK_SYSCALLS,
    # Landlock deliberately does not mediate these metadata operations.  A
    # candidate must not alter even modes, ownership, timestamps, or xattrs of
    # a protected checkout while its tests are being judged.
    "chmod",
    "fchmod",
    "fchmodat",
    "fchmodat2",
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
        os.close(ruleset_fd)
    return abi


def _resolve_seccomp_rules(
    resolver: Callable[[bytes], int],
) -> tuple[tuple[str, int], ...]:
    """Resolve the closed deny policy and reject any missing network rule."""

    rules: list[tuple[str, int]] = []
    unresolved_network: list[str] = []
    for name in _DENIED_SYSCALLS:
        syscall_number = int(resolver(name.encode("ascii")))
        if syscall_number < 0:
            if name in _DENIED_NETWORK_SYSCALLS:
                unresolved_network.append(name)
            continue
        rules.append((name, syscall_number))
    if unresolved_network:
        raise QualificationError(
            "candidate sandbox cannot resolve required network syscalls: "
            + ", ".join(unresolved_network)
        )
    resolved_names = {name for name, _number in rules}
    if not set(_DENIED_NETWORK_SYSCALLS).issubset(resolved_names):
        raise QualificationError("candidate sandbox network deny policy is incomplete")
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
    ):
        raise QualificationError("candidate sandbox installed seccomp policy differs")
    evidence: dict[str, Any] = {
        "schema": _SECCOMP_POLICY_SCHEMA,
        "default_action": "allow",
        "deny_action": "errno:EPERM",
        "required_network_syscalls": list(_DENIED_NETWORK_SYSCALLS),
        "requested_syscalls": list(_DENIED_SYSCALLS),
        "installed_syscalls": list(installed_names),
        "unavailable_optional_syscalls": [
            name for name in _DENIED_SYSCALLS if name not in installed_names
        ],
    }
    evidence["policy_cid"] = content_identity(evidence)
    return evidence


def _install_seccomp() -> tuple[str, ...]:
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
        for name, syscall_number in rules:
            if library.seccomp_rule_add(context, action, syscall_number, 0) != 0:
                raise QualificationError(
                    f"candidate sandbox could not deny syscall {name}"
                )
            installed.append(name)
        if not set(_DENIED_NETWORK_SYSCALLS).issubset(installed):
            raise QualificationError("candidate sandbox network deny policy is incomplete")
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
) -> dict[str, Any]:
    """Install irreversible filesystem and network restrictions in a worker."""

    # Resolve and load libseccomp before lowering the process limit because
    # libc discovery may itself use one bounded helper process.
    denied_syscalls = _install_seccomp()
    # Bound damage from a malicious or accidentally explosive candidate test.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    file_size_bytes = _lower_resource_limit(resource.RLIMIT_FSIZE, 64 * 1024 * 1024)
    open_files = _lower_resource_limit(resource.RLIMIT_NOFILE, 256)
    # RLIMIT_NPROC is per real UID (and counts threads), not per worker.  The
    # development host legitimately runs more than 2,048 same-UID threads, so a
    # lower ceiling prevents even deterministic library imports.  Keep a
    # finite ceiling and separately pin numerical libraries to one thread in
    # the sealed worker environment.
    processes = _lower_resource_limit(resource.RLIMIT_NPROC, 4_096)
    cpu_seconds = _lower_resource_limit(resource.RLIMIT_CPU, 900)
    address_space_bytes = _lower_resource_limit(resource.RLIMIT_AS, 8 * 1024**3)
    landlock_abi = _install_landlock(write_root)
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
        "processes": 4_096,
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
            and policy.get("requested_syscalls") == list(_DENIED_SYSCALLS)
            and value.get("seccomp_denied_syscall_count") == len(installed)
        )
        if value.get("worker_pycache") != {
            "schema": "lgcvf-recovery-worker-pycache-isolation@1",
            "write_root_relative_path": "python-pycache",
            "environment_variable": "PYTHONPYCACHEPREFIX",
            "python_prefix_active": True,
            "dont_write_bytecode": True,
            "owner_matches_worker": True,
            "mode_octal": "0700",
            "empty_before": True,
            "empty_after": True,
        }:
            return False
    return (
        value.get("profile") == "landlock-readonly-seccomp-no-network"
        and isinstance(value.get("landlock_abi"), int)
        and not isinstance(value.get("landlock_abi"), bool)
        and int(value["landlock_abi"]) >= 4
        and isinstance(value.get("seccomp_denied_syscall_count"), int)
        and not isinstance(value.get("seccomp_denied_syscall_count"), bool)
        and int(value["seccomp_denied_syscall_count"])
        >= len(_DENIED_NETWORK_SYSCALLS)
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


def _read_recovery_projection_source(
    root: Path,
    relative: str,
    *,
    git_entry: tuple[str, str] | None = None,
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
    directory_flags = (
        os.O_RDONLY
        | os.O_CLOEXEC
        | os.O_DIRECTORY
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(root, directory_flags)
    except OSError as exc:
        raise QualificationError(
            "recovery projection source is unavailable"
        ) from exc
    try:
        for component in logical.parts[:-1]:
            child = os.open(component, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        source_fd = os.open(logical.name, file_flags, dir_fd=descriptor)
        before = os.fstat(source_fd)
        expected_mode = "100755" if stat.S_IMODE(before.st_mode) & 0o111 else "100644"
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or before.st_size < 0
            or before.st_size > 16 * 1024 * 1024
            or (git_entry is not None and git_entry[0] != expected_mode)
        ):
            raise QualificationError("recovery projection source identity differs")
        try:
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
        os.close(descriptor)


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
        "schema": "lgcvf-closed-recovery-test-projection@1",
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
    if (
        _ISOLATED_RECOVERY_PYCACHE_DIRECTORY is None
        or root is None
        or identity is None
        or sys.pycache_prefix != str(root)
    ):
        raise QualificationError("protected recovery pycache isolation differs")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(root, flags)
    except OSError as exc:
        raise QualificationError(
            "protected recovery pycache isolation differs"
        ) from exc
    try:
        status = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(status.st_mode)
            or (status.st_dev, status.st_ino) != identity
            or status.st_uid != os.geteuid()
            or stat.S_IMODE(status.st_mode) != 0o700
        ):
            raise QualificationError("protected recovery pycache isolation differs")
        with os.scandir(descriptor) as entries:
            if next(entries, None) is not None:
                raise QualificationError(
                    "protected recovery pycache isolation is not empty"
                )
    finally:
        os.close(descriptor)


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

    def __init__(self) -> None:
        self.import_attempts: list[str] = []
        self.process_attempts: list[str] = []
        self._finder = _RecoveryProviderImportFinder(self.import_attempts)
        self._active = False

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

    payload = json.dumps(value, sort_keys=True).encode("utf-8") + b"\n"
    if len(payload) > _MAX_WORKER_RECEIPT_BYTES:
        payload = json.dumps(
            {
                "schema": value.get("schema", WORKER_SCHEMA),
                "suite_id": value.get("suite_id", ""),
                "error": "receipt_too_large",
                "reason": "worker receipt exceeded its predeclared pipe bound",
            },
            sort_keys=True,
        ).encode("utf-8") + b"\n"
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

    lexical = write_root / "python-pycache"
    expected = lexical.resolve(strict=True)
    if expected.parent != write_root or lexical.is_symlink():
        raise QualificationError("recovery worker pycache path differs")
    if (
        os.environ.get("PYTHONPYCACHEPREFIX") != str(expected)
        or sys.pycache_prefix != str(expected)
        or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
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
        "schema": "lgcvf-recovery-worker-pycache-isolation@1",
        "write_root_relative_path": "python-pycache",
        "environment_variable": "PYTHONPYCACHEPREFIX",
        "python_prefix_active": True,
        "dont_write_bytecode": True,
        "owner_matches_worker": True,
        "mode_octal": "0700",
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
    } or value.get("schema") != "lgcvf-closed-recovery-test-projection@1":
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


def _worker(
    suite_id: str,
    *,
    execution_root: Path | None = None,
    write_root: Path | None = None,
    receipt_descriptor: int | None = None,
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
    worker_schema = RECOVERY_WORKER_SCHEMA if recovery is not None else WORKER_SCHEMA
    try:
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
        required_import_paths = [str(owner)]
        if recovery is not None:
            if recovery.owner_root == "ipfs_datasets_py":
                required_import_paths.append(str(worker_root))
            else:
                required_import_paths.append(str(worker_root / "ipfs_datasets_py"))
        inserted_import_paths: list[str] = []
        provider_guard = _RecoveryProviderGuard() if recovery is not None else None
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
                import pytest

                with tempfile.TemporaryDirectory(
                    prefix="lgcvf-pytest-cache-"
                ) as cache_dir:
                    if writable_root is None:
                        raise QualificationError("isolated worker has no writable root")
                    pytest_log = writable_root / "pytest.log"
                    with contextlib.redirect_stdout(
                        captured_out
                    ), contextlib.redirect_stderr(captured_err):
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
) -> dict[str, Any]:
    recovery = _RECOVERY_BY_SUITE_ID.get(suite.suite_id)
    expected_worker_schema = (
        RECOVERY_WORKER_SCHEMA if recovery is not None else WORKER_SCHEMA
    )
    source_root = root.resolve(strict=True)
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
    if not dependency_paths:
        raise QualificationError("qualified Python dependency path is unavailable")
    with tempfile.TemporaryDirectory(prefix="lgcvf-qualification-sandbox-") as sandbox:
        sandbox_path = Path(sandbox)
        checkout = sandbox_path / "checkout"
        writable = sandbox_path / "writable"
        writable.mkdir(mode=0o700)
        worker_pycache = writable / "python-pycache"
        worker_pycache.mkdir(mode=0o700)
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
        projected_import_paths = (
            str(checkout),
            str(checkout / "ipfs_datasets_py"),
        )
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
        receipt_read, receipt_write = os.pipe()
        stdout_path = writable / "worker.stdout"
        stderr_path = writable / "worker.stderr"
        stdout_handle = stdout_path.open("wb") if recovery is not None else None
        stderr_handle = stderr_path.open("wb") if recovery is not None else None
        try:
            worker_arguments = [
                sys.executable,
                str(
                    checkout
                    / "scripts/qualify_logic_governed_compositional_verification_fabric.py"
                ),
                "--worker",
                suite.suite_id,
                "--worker-root",
                str(checkout),
                "--worker-write-root",
                str(writable),
                "--worker-receipt-fd",
                str(receipt_write),
            ]
            if recovery is not None:
                encoded_projection = base64.urlsafe_b64encode(
                    json.dumps(
                        execution_projection,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).decode("ascii")
                worker_arguments.extend(
                    ("--worker-projection", encoded_projection)
                )
            process = subprocess.Popen(
                worker_arguments,
                cwd=checkout,
                env=environment,
                stdout=stdout_handle if stdout_handle is not None else subprocess.PIPE,
                stderr=stderr_handle if stderr_handle is not None else subprocess.PIPE,
                text=True,
                start_new_session=True,
                pass_fds=(receipt_write,),
            )
        except Exception:
            os.close(receipt_read)
            os.close(receipt_write)
            if stdout_handle is not None:
                stdout_handle.close()
            if stderr_handle is not None:
                stderr_handle.close()
            raise
        os.close(receipt_write)
        receipt_drain = _BoundedReceiptPipeDrain(receipt_read)
        try:
            receipt_drain.start()
        except Exception:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                os.close(receipt_read)
            except OSError:
                pass
            if stdout_handle is not None:
                stdout_handle.close()
            if stderr_handle is not None:
                stderr_handle.close()
            raise
        timed_out = False
        wall_timeout = (
            recovery.validation_spec()["timeout_seconds"]
            if recovery is not None
            else 1800
        )
        try:
            stdout_value, stderr_value = process.communicate(timeout=wall_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout_value, stderr_value = process.communicate(timeout=30)
        finally:
            # Candidate descendants inherit the worker's process group and
            # seccomp denies setpgid/setsid/unshare.  Kill any daemon that
            # closed its pipes and outlived the direct worker.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
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
        if timed_out:
            raise QualificationError(
                f"{suite.suite_id}: worker exceeded the {wall_timeout}-second wall bound; "
                + (stderr or stdout)[-1000:]
            )
        returncode = int(process.returncode or 0)
    try:
        receipt_text = receipt.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise QualificationError(f"{suite.suite_id}: worker receipt is not UTF-8") from exc
    lines = [line for line in receipt_text.splitlines() if line.strip()]
    if not lines:
        raise QualificationError(
            f"{suite.suite_id}: worker emitted no receipt "
            f"(returncode {returncode}): {(stderr or stdout)[-1000:]}"
        )
    try:
        payload = _strict_json_loads(
            lines[-1], noun=f"{suite.suite_id}: worker receipt"
        )
    except json.JSONDecodeError as exc:
        raise QualificationError(f"{suite.suite_id}: worker receipt is invalid") from exc
    if not isinstance(payload, dict) or payload.get("schema") != expected_worker_schema:
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
    if recovery is not None and payload.get("readonly_projection") != execution_projection:
        raise QualificationError(f"{suite.suite_id}: read-only projection differs")
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
        payload.update(
            {
                "worker_observation_cid": worker_observation_cid,
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
        "schema": "lgcvf-recovery-source-binding@2",
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
        "worker_observation_cid",
        "raw_stdout_size_bytes",
        "raw_stdout_sha256",
        "raw_stderr_size_bytes",
        "raw_stderr_sha256",
        "observation_cid",
    }
)


def _validate_recovery_observation(
    value: Mapping[str, Any],
    validation: RecoveryValidation,
    *,
    root: Path = ROOT,
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


def _recovery_unavailable(reason: BaseException) -> dict[str, Any]:
    population = [
        {
            "task_id": item.task_id,
            "task_cid": item.task_cid,
            "validation_spec_cid": item.validation_spec()["validation_spec_cid"],
        }
        for item in RECOVERY_VALIDATIONS
    ]
    detail = f"{type(reason).__name__}: {reason}"[:2000]
    value: dict[str, Any] = {
        "schema": RECOVERY_UNAVAILABLE_SCHEMA,
        "qualification_schema": RECOVERY_SCHEMA,
        "plan_cid": PLAN_CID,
        "disposition": "unavailable",
        "ordered_task_population": population,
        "reason": detail,
        "reason_sha256": _sha256_bytes(detail.encode("utf-8")),
        "self_authority": False,
        "completion_authoritative": False,
        "release_qualified": False,
        "production_authorized": False,
    }
    value["receipt_cid"] = content_identity(value)
    return value


def run_preregistered_recovery_qualification(
    *, root: Path = ROOT
) -> dict[str, Any]:
    """Run exactly six task-bound replays inside the protected OS sandbox."""

    _require_isolated_recovery_runtime()
    try:
        manifest = _preregistered_recovery_manifest(root=root)
        before = _recovery_source_binding(root=root)
        observations: list[dict[str, Any]] = []
        entries = manifest["ordered_entries"]
        for validation, entry in zip(RECOVERY_VALIDATIONS, entries, strict=True):
            observation = _run_suite(
                validation.suite,
                expected_manifest=entry["suite_manifest"],
                root=root,
            )
            _validate_recovery_observation(observation, validation, root=root)
            observations.append(observation)
            if _recovery_source_binding(root=root) != before:
                raise QualificationError(
                    f"{validation.task_id}: recovery source changed during replay"
                )
        after = _recovery_source_binding(root=root)
        totals = {
            field: sum(int(item[field]) for item in observations)
            for field in _RECOVERY_COUNT_FIELDS
        }
        omission_commitment, projection_evidence = _recovery_projection_commitments(
            before, observations
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
            "passed": all(item.get("passed") is True for item in observations),
            "totals": totals,
            "suites": observations,
            "validation_projection_omission_commitment": omission_commitment,
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
        return verify_preregistered_recovery_qualification(result, root=root)
    except (OSError, ValueError, json.JSONDecodeError, QualificationError) as exc:
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
        "reason",
        "reason_sha256",
        "self_authority",
        "completion_authoritative",
        "release_qualified",
        "production_authorized",
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
        reason = value.get("reason")
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
        if (
            value.get("receipt_cid") != content_identity(body)
            or value.get("qualification_schema") != RECOVERY_SCHEMA
            or value.get("plan_cid") != PLAN_CID
            or value.get("disposition") != "unavailable"
            or value.get("ordered_task_population") != expected_population
            or not isinstance(reason, str)
            or len(reason) > 2000
            or value.get("reason_sha256")
            != _sha256_bytes(str(reason).encode("utf-8"))
            or any(
                value.get(field) is not False
                for field in (
                    "self_authority",
                    "completion_authoritative",
                    "release_qualified",
                    "production_authorized",
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
        _validate_recovery_observation(observation, validation, root=root)
        for count_field in _RECOVERY_COUNT_FIELDS:
            totals[count_field] += int(observation[count_field])
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
            execution_projection=execution_projection,
        )
    if (
        args.worker_root is not None
        or args.worker_write_root is not None
        or args.worker_receipt_fd is not None
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
